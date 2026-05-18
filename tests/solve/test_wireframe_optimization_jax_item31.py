"""Item 31 tests for fixed-state wireframe RCLS JAX solve helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import BiotSavart, Current, ToroidalField, coils_via_symmetries
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.field.toroidal_field_jax import ToroidalFieldJAX
from simsopt.geo import (
    SurfaceRZFourier,
    ToroidalWireframe,
    create_equally_spaced_curves,
)
from simsopt.solve.wireframe_optimization import (
    bnorm_obj_matrices,
    get_gsco_iteration,
    optimize_wireframe,
    rcls_wireframe,
    regularized_constrained_least_squares,
)
from simsopt.solve.wireframe_optimization_jax import (
    _gsco_opposite_candidate_index,
    bnorm_obj_matrices_jax,
    get_gsco_iteration_jax,
    greedy_stellarator_coil_optimization_jax,
    gsco_wireframe_jax,
    optimize_wireframe_jax,
    rcls_wireframe_jax,
    regularized_constrained_least_squares_jax,
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


@dataclass
class _WireframeFixture:
    n_segments: int
    free_segments: np.ndarray
    C: np.ndarray
    d: np.ndarray
    currents: np.ndarray

    def constraint_matrices(
        self,
        *,
        assume_no_crossings: bool,
        remove_constrained_segments: bool,
    ):
        assert assume_no_crossings is True
        assert remove_constrained_segments is True
        return self.C, self.d

    def unconstrained_segments(self):
        return self.free_segments


@dataclass
class _GSCOFixture:
    currents: np.ndarray
    loops: np.ndarray
    free_loops: np.ndarray
    segments: np.ndarray
    connected_segments: np.ndarray

    def get_cell_key(self):
        return self.loops

    def get_free_cells(self, *, form: str):
        assert form == "logical"
        return self.free_loops


def _least_squares_problem():
    rng = np.random.default_rng(3101)
    A = np.ascontiguousarray(rng.standard_normal(size=(9, 4)))
    b = np.ascontiguousarray(rng.standard_normal(size=(9, 1)))
    C = np.ascontiguousarray(np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]))
    d = np.ascontiguousarray(np.array([[0.25], [-0.1]]))
    W_matrix = np.ascontiguousarray(
        np.array(
            [
                [0.2, 0.01, 0.0, 0.0],
                [0.01, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.3, -0.02],
                [0.0, 0.0, -0.02, 0.35],
            ]
        )
    )
    return A, b, C, d, W_matrix


def _gsco_problem():
    rng = np.random.default_rng(3104)
    A = np.ascontiguousarray(rng.standard_normal(size=(5, 6)))
    b = np.ascontiguousarray(rng.standard_normal(size=(5, 1)))
    loops = np.ascontiguousarray(np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int64))
    free_loops = np.ascontiguousarray(np.ones(2, dtype=np.int64))
    segments = np.ascontiguousarray(
        np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
            dtype=np.int64,
        )
    )
    connections = np.ascontiguousarray(
        np.array(
            [[0, 3, 4, 0], [0, 1, 5, 0], [1, 2, 4, 0], [2, 3, 5, 0]],
            dtype=np.int64,
        )
    )
    x_init = np.ascontiguousarray(np.zeros((6, 1), dtype=np.float64))
    loop_count_init = np.ascontiguousarray(np.zeros(2, dtype=np.int64))
    return A, b, loops, free_loops, segments, connections, x_init, loop_count_init


def _synthetic_gsco_problem(
    *,
    n_grid: int,
    n_loops: int,
    seed: int,
    n_segments: int | None = None,
):
    if n_segments is None:
        n_segments = 4 * n_loops

    loops = np.arange(4 * n_loops, dtype=np.int64).reshape(n_loops, 4)
    if n_segments > 4 * n_loops:
        loops[-1] = np.arange(n_segments - 4, n_segments, dtype=np.int64)
    free_loops = np.ones((n_loops,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    A = np.zeros((n_grid, n_segments), dtype=np.float64)
    active_columns = np.unique(loops.reshape(-1))
    A[:, active_columns] = rng.standard_normal(size=(n_grid, active_columns.size))
    b = rng.standard_normal(size=(n_grid, 1))

    nodes = np.arange(n_segments, dtype=np.int64)
    segments = np.stack([nodes, np.roll(nodes, -1)], axis=1)
    connections = np.zeros((n_segments, 4), dtype=np.int64)
    connections[:, 0] = nodes
    x_init = np.zeros((n_segments, 1), dtype=np.float64)
    loop_count_init = np.zeros((n_loops,), dtype=np.int64)
    return (
        np.ascontiguousarray(A),
        np.ascontiguousarray(b),
        np.ascontiguousarray(loops),
        np.ascontiguousarray(free_loops),
        np.ascontiguousarray(segments),
        np.ascontiguousarray(connections),
        np.ascontiguousarray(x_init),
        np.ascontiguousarray(loop_count_init),
    )


def _surf_torus(nfp: int, rmaj: float, rmin: float) -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=nfp, mpol=1, ntor=0)
    surface.set_rc(0, 0, rmaj)
    surface.set_rc(1, 0, rmin)
    surface.set_zs(1, 0, rmin)
    return surface


def _public_wireframe() -> ToroidalWireframe:
    return ToroidalWireframe(_surf_torus(nfp=2, rmaj=2.0, rmin=0.7), 4, 6)


def _equally_spaced_coils(surface: SurfaceRZFourier, *, ncoils: int, current: float):
    curves = create_equally_spaced_curves(
        ncoils,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.35,
        order=3,
    )
    currents = [Current(current) for _ in range(ncoils)]
    return coils_via_symmetries(curves, currents, surface.nfp, True)


def _public_optimize_problem(seed: int = 3105):
    wireframe = _public_wireframe()
    rng = np.random.default_rng(seed)
    A = np.ascontiguousarray(
        rng.standard_normal(size=(wireframe.n_segments + 10, wireframe.n_segments))
    )
    b = np.ascontiguousarray(rng.standard_normal(size=(A.shape[0], 1)))
    return wireframe, A, b


def _compare_gsco_result(actual, expected) -> None:
    (
        x_expected,
        loop_count_expected,
        iter_hist_expected,
        curr_hist_expected,
        loop_hist_expected,
        f_B_hist_expected,
        f_S_hist_expected,
        f_hist_expected,
    ) = expected
    history_length = int(np.asarray(actual.history_length))
    valid_history = slice(0, history_length)

    np.testing.assert_allclose(np.asarray(actual.x), x_expected, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(np.asarray(actual.loop_count), loop_count_expected)
    np.testing.assert_array_equal(
        np.asarray(actual.iter_history)[valid_history], iter_hist_expected
    )
    np.testing.assert_allclose(
        np.asarray(actual.curr_history)[valid_history],
        curr_hist_expected,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.loop_history)[valid_history], loop_hist_expected
    )
    np.testing.assert_allclose(
        np.asarray(actual.f_B_history)[valid_history],
        f_B_hist_expected,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(actual.f_S_history)[valid_history],
        f_S_hist_expected,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(actual.f_history)[valid_history],
        f_hist_expected,
        rtol=_RTOL,
        atol=_ATOL,
    )


def _run_cpp_gsco(
    no_crossing,
    no_new_coils,
    match_current,
    A,
    b,
    default_current,
    max_current,
    max_loop_count,
    loops,
    free_loops,
    segments,
    connections,
    lambda_S,
    max_iter,
    x_init,
    loop_count_init,
):
    return sopp.GSCO(
        no_crossing,
        no_new_coils,
        match_current,
        A,
        b,
        abs(default_current),
        abs(max_current),
        abs(max_loop_count),
        loops,
        free_loops,
        segments,
        connections,
        lambda_S,
        max_iter,
        x_init,
        loop_count_init,
        1,
    )


@pytest.mark.parametrize(
    "W",
    (
        0.25,
        np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        np.array([[0.2], [0.3], [0.4], [0.5]], dtype=np.float64),
        np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float64),
        _least_squares_problem()[-1],
    ),
)
def test_regularized_constrained_least_squares_jax_matches_cpu(W) -> None:
    A, b, C, d, _ = _least_squares_problem()
    expected = regularized_constrained_least_squares(A, b, W, C, d)
    actual = regularized_constrained_least_squares_jax(A, b, W, C, d)

    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_regularized_constrained_least_squares_jits_under_transfer_guard() -> None:
    A, b, C, d, _ = _least_squares_problem()
    W = jnp.asarray(np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64))
    A_device = jax.device_put(jnp.asarray(A))
    b_device = jax.device_put(jnp.asarray(b))
    C_device = jax.device_put(jnp.asarray(C))
    d_device = jax.device_put(jnp.asarray(d))

    @jax.jit
    def _solve(A_data, b_data, W_data, C_data, d_data):
        return regularized_constrained_least_squares_jax(
            A_data, b_data, W_data, C_data, d_data
        )

    _solve(A_device, b_device, W, C_device, d_device).block_until_ready()
    with jax.transfer_guard("disallow"):
        out = _solve(A_device, b_device, W, C_device, d_device)
        out.block_until_ready()

    assert out.shape == (4, 1)
    assert np.all(np.isfinite(np.asarray(out)))


def test_regularized_constrained_least_squares_handles_no_constraints() -> None:
    rng = np.random.default_rng(3103)
    A = np.ascontiguousarray(rng.standard_normal(size=(7, 3)))
    b = np.ascontiguousarray(rng.standard_normal(size=(7, 1)))
    C = np.zeros((0, 3), dtype=np.float64)
    d = np.zeros((0, 1), dtype=np.float64)
    W = 0.15

    expected = regularized_constrained_least_squares(A, b, W, C, d)
    actual = regularized_constrained_least_squares_jax(A, b, W, C, d)

    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_regularized_constrained_least_squares_exact_rank_deficient_matches_cpu() -> None:
    A = np.ascontiguousarray(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 2.0, 2.0],
                [0.0, 3.0, 3.0],
            ],
            dtype=np.float64,
        )
    )
    b = np.ascontiguousarray(np.array([[1.0], [1.0], [2.0], [3.0]]))
    C = np.zeros((0, 3), dtype=np.float64)
    d = np.zeros((0, 1), dtype=np.float64)
    W = 0.0

    expected = regularized_constrained_least_squares(A, b, W, C, d)
    actual = regularized_constrained_least_squares_jax(A, b, W, C, d)

    assert np.linalg.matrix_rank(A.T @ A) < A.shape[1]
    np.testing.assert_allclose(
        expected,
        np.array([[1.0], [0.5], [0.5]], dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_regularized_constrained_least_squares_rank_deficient_matches_cpu() -> None:
    lhs = np.array(
        [[0.52239503, 0.49949821], [0.49949821, 0.47760498]],
        dtype=np.float64,
    )
    rhs = np.array([[0.90535587], [0.44637457]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(lhs)
    A = np.ascontiguousarray(np.diag(np.sqrt(eigvals)) @ eigvecs.T)
    b = np.ascontiguousarray(np.linalg.solve(A.T, rhs))
    C = np.zeros((0, 2), dtype=np.float64)
    d = np.zeros((0, 1), dtype=np.float64)
    W = 0.0

    expected = regularized_constrained_least_squares(A, b, W, C, d)
    actual = regularized_constrained_least_squares_jax(A, b, W, C, d)

    assert np.linalg.cond(lhs) > 1.0e8
    assert np.max(np.abs(expected)) > 1.0e7
    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize(
    "reg_W",
    (
        0.15,
        np.array([0.11, 0.13, 0.17, 0.19, 0.23], dtype=np.float64),
        np.diag(np.array([0.11, 0.13, 0.17, 0.19, 0.23], dtype=np.float64)),
    ),
)
def test_rcls_wireframe_jax_matches_cpu_without_mutating_wireframe(reg_W) -> None:
    rng = np.random.default_rng(3102)
    Amat = np.ascontiguousarray(rng.standard_normal(size=(8, 5)))
    bvec = np.ascontiguousarray(rng.standard_normal(size=(8, 1)))
    fixture = _WireframeFixture(
        n_segments=5,
        free_segments=np.array([0, 2, 4], dtype=np.int64),
        C=np.ascontiguousarray(np.array([[1.0, -1.0, 0.5]], dtype=np.float64)),
        d=np.ascontiguousarray(np.array([[0.2]], dtype=np.float64)),
        currents=np.full(5, 7.0, dtype=np.float64),
    )
    cpu_fixture = _WireframeFixture(
        n_segments=fixture.n_segments,
        free_segments=fixture.free_segments,
        C=fixture.C,
        d=fixture.d,
        currents=fixture.currents.copy(),
    )

    x_cpu, f_B_cpu, f_R_cpu, f_cpu = rcls_wireframe(
        cpu_fixture,
        Amat,
        bvec,
        reg_W,
        assume_no_crossings=True,
        verbose=False,
    )
    result = rcls_wireframe_jax(
        fixture,
        Amat,
        bvec,
        reg_W,
        assume_no_crossings=True,
    )

    np.testing.assert_allclose(np.asarray(result.x), x_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(result.f_B), f_B_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(result.f_R), f_R_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(result.f), f_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(fixture.currents, np.full(5, 7.0, dtype=np.float64))


@pytest.mark.parametrize("area_weighted", (False, True))
def test_bnorm_obj_matrices_jax_matches_cpu_public_surface_mode(area_weighted) -> None:
    wireframe = _public_wireframe()
    plasma_surface = _surf_torus(nfp=2, rmaj=1.85, rmin=0.45)

    expected_A, expected_b = bnorm_obj_matrices(
        wireframe,
        plasma_surface,
        area_weighted=area_weighted,
        verbose=False,
    )
    actual_A, actual_b = bnorm_obj_matrices_jax(
        wireframe,
        plasma_surface,
        area_weighted=area_weighted,
        verbose=False,
    )

    np.testing.assert_allclose(actual_A, expected_A, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(actual_b, expected_b, rtol=_RTOL, atol=_ATOL)


def test_bnorm_obj_matrices_jax_matches_cpu_ext_field_and_target() -> None:
    wireframe = _public_wireframe()
    plasma_surface = _surf_torus(nfp=2, rmaj=1.85, rmin=0.45)
    ext_field = ToroidalFieldJAX(1.0, 0.25)
    normal = plasma_surface.normal()
    unit_normal = normal / np.linalg.norm(normal, axis=2)[:, :, None]
    bnorm_target = 0.07 * unit_normal[:, :, 2] + 0.02 * unit_normal[:, :, 0]

    expected_A, expected_b = bnorm_obj_matrices(
        wireframe,
        plasma_surface,
        ext_field=ext_field,
        bnorm_target=bnorm_target,
        area_weighted=False,
        verbose=False,
    )
    actual_A, actual_b = bnorm_obj_matrices_jax(
        wireframe,
        plasma_surface,
        ext_field=ext_field,
        bnorm_target=bnorm_target,
        area_weighted=False,
        verbose=False,
    )

    np.testing.assert_allclose(actual_A, expected_A, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(actual_b, expected_b, rtol=_RTOL, atol=_ATOL)


def test_bnorm_obj_matrices_jax_matches_cpu_biotsavart_ext_field() -> None:
    wireframe = _public_wireframe()
    plasma_surface = _surf_torus(nfp=2, rmaj=1.85, rmin=0.45)
    current = 2.5e4
    cpu_coils = _equally_spaced_coils(plasma_surface, ncoils=2, current=current)
    jax_coils = _equally_spaced_coils(plasma_surface, ncoils=2, current=current)
    ext_field_cpu = BiotSavart(cpu_coils)
    ext_field_jax = BiotSavartJAX(jax_coils)

    expected_A, expected_b = bnorm_obj_matrices(
        wireframe,
        plasma_surface,
        ext_field=ext_field_cpu,
        area_weighted=False,
        verbose=False,
    )
    actual_A, actual_b = bnorm_obj_matrices_jax(
        wireframe,
        plasma_surface,
        ext_field=ext_field_jax,
        area_weighted=False,
        verbose=False,
    )

    assert ext_field_jax.get_points_cart_ref() is None
    np.testing.assert_allclose(actual_A, expected_A, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(actual_b, expected_b, rtol=_RTOL, atol=_ATOL)


def test_bnorm_obj_matrices_jax_rejects_cpu_ext_field() -> None:
    wireframe = _public_wireframe()
    plasma_surface = _surf_torus(nfp=2, rmaj=1.85, rmin=0.45)
    ext_field = ToroidalField(1.0, 0.25)

    with pytest.raises(ValueError, match="JAX-native MagneticField"):
        bnorm_obj_matrices_jax(
            wireframe,
            plasma_surface,
            ext_field=ext_field,
            area_weighted=False,
            verbose=False,
        )


def test_optimize_wireframe_jax_rcls_matches_public_cpu_and_mutates() -> None:
    cpu_wireframe, A, b = _public_optimize_problem()
    jax_wireframe, _, _ = _public_optimize_problem()
    params = {"reg_W": 0.1, "assume_no_crossings": False}

    expected = optimize_wireframe(
        cpu_wireframe,
        "rcls",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )
    actual = optimize_wireframe_jax(
        jax_wireframe,
        "rcls",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )

    np.testing.assert_allclose(actual["x"], expected["x"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(actual["f_B"], expected["f_B"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(actual["f_R"], expected["f_R"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(actual["f"], expected["f"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        jax_wireframe.currents,
        cpu_wireframe.currents,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_optimize_wireframe_jax_rcls_matches_cpu_with_public_constraints() -> None:
    cpu_wireframe, A, b = _public_optimize_problem(seed=3110)
    jax_wireframe, _, _ = _public_optimize_problem(seed=3110)
    for wireframe in (cpu_wireframe, jax_wireframe):
        wireframe.set_poloidal_current(0.05)
        wireframe.set_toroidal_current(-0.03)
        wireframe.set_segments_constrained(
            [0, wireframe.n_tor_segments, wireframe.n_tor_segments + 1]
        )
    params = {"reg_W": 0.1, "assume_no_crossings": False}

    expected = optimize_wireframe(
        cpu_wireframe,
        "rcls",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )
    actual = optimize_wireframe_jax(
        jax_wireframe,
        "rcls",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )

    for key in ("x", "f_B", "f_R", "f"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        jax_wireframe.currents,
        cpu_wireframe.currents,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_gsco_jax_matches_cpp_fixed_state_baseline() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    expected = _run_cpp_gsco(
        False,
        False,
        False,
        A,
        b,
        0.2,
        np.inf,
        0,
        loops,
        free_loops,
        segments,
        connections,
        0.15,
        5,
        x_init,
        loop_count_init,
    )

    actual = greedy_stellarator_coil_optimization_jax(
        False,
        False,
        False,
        A,
        b,
        0.2,
        np.inf,
        0,
        loops,
        free_loops,
        segments,
        connections,
        0.15,
        5,
        x_init,
        loop_count_init,
    )

    _compare_gsco_result(actual, expected)


def test_gsco_jax_large_200_grid_50_loops_matches_cpp() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _synthetic_gsco_problem(n_grid=200, n_loops=50, seed=3110)
    )
    expected = _run_cpp_gsco(
        False,
        False,
        False,
        A,
        b,
        0.2,
        1.0,
        3,
        loops,
        free_loops,
        segments,
        connections,
        0.05,
        50,
        x_init,
        loop_count_init,
    )

    actual = greedy_stellarator_coil_optimization_jax(
        False,
        False,
        False,
        A,
        b,
        0.2,
        1.0,
        3,
        loops,
        free_loops,
        segments,
        connections,
        0.05,
        50,
        x_init,
        loop_count_init,
    )

    assert A.shape == (200, 200)
    assert loops.shape == (50, 4)
    _compare_gsco_result(actual, expected)


def test_gsco_jax_topology_with_more_than_int16_segments_matches_cpp() -> None:
    n_segments = 2**15 + 8
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _synthetic_gsco_problem(
            n_grid=8,
            n_loops=2,
            n_segments=n_segments,
            seed=3111,
        )
    )
    expected = _run_cpp_gsco(
        False,
        False,
        False,
        A,
        b,
        0.2,
        1.0,
        2,
        loops,
        free_loops,
        segments,
        connections,
        0.05,
        2,
        x_init,
        loop_count_init,
    )

    actual = greedy_stellarator_coil_optimization_jax(
        False,
        False,
        False,
        A,
        b,
        0.2,
        1.0,
        2,
        loops,
        free_loops,
        segments,
        connections,
        0.05,
        2,
        x_init,
        loop_count_init,
    )

    assert segments.shape[0] > 2**15
    assert int(loops.max()) > 2**15
    _compare_gsco_result(actual, expected)


def test_gsco_opposite_candidate_index_wraps_negative_to_positive() -> None:
    """Undo detection maps both current directions to the opposite candidate."""

    n_loops = 3
    candidates = jnp.arange(2 * n_loops, dtype=jnp.int32)
    actual = jax.vmap(lambda opt_ind: _gsco_opposite_candidate_index(opt_ind, n_loops))(
        candidates
    )

    np.testing.assert_array_equal(
        np.asarray(jax.device_get(actual)),
        np.array([3, 4, 5, 0, 1, 2], dtype=np.int32),
    )


def test_gsco_jax_stop_none_eligible_matches_cpp() -> None:
    A = np.zeros((2, 4), dtype=np.float64)
    b = np.ones((2, 1), dtype=np.float64)
    loops = np.array([[0, 1, 2, 3]], dtype=np.int64)
    free_loops = np.array([0], dtype=np.int64)
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    connections = np.zeros((4, 4), dtype=np.int64)
    x_init = np.zeros((4, 1), dtype=np.float64)
    loop_count_init = np.zeros(1, dtype=np.int64)

    expected = _run_cpp_gsco(
        False,
        False,
        False,
        A,
        b,
        0.2,
        np.inf,
        0,
        loops,
        free_loops,
        segments,
        connections,
        0.1,
        4,
        x_init,
        loop_count_init,
    )
    actual = greedy_stellarator_coil_optimization_jax(
        False,
        False,
        False,
        A,
        b,
        0.2,
        np.inf,
        0,
        loops,
        free_loops,
        segments,
        connections,
        0.1,
        4,
        x_init,
        loop_count_init,
    )

    _compare_gsco_result(actual, expected)
    assert int(np.asarray(actual.history_length)) == 1


def test_gsco_jax_undo_branch_matches_cpp() -> None:
    A = np.zeros((2, 4), dtype=np.float64)
    b = np.zeros((2, 1), dtype=np.float64)
    loops = np.array([[0, 1, 2, 3]], dtype=np.int64)
    free_loops = np.array([1], dtype=np.int64)
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    connections = np.zeros((4, 4), dtype=np.int64)
    x_init = np.zeros((4, 1), dtype=np.float64)
    loop_count_init = np.zeros(1, dtype=np.int64)

    expected = _run_cpp_gsco(
        False,
        False,
        False,
        A,
        b,
        0.2,
        np.inf,
        0,
        loops,
        free_loops,
        segments,
        connections,
        0.1,
        4,
        x_init,
        loop_count_init,
    )
    actual = greedy_stellarator_coil_optimization_jax(
        False,
        False,
        False,
        A,
        b,
        0.2,
        np.inf,
        0,
        loops,
        free_loops,
        segments,
        connections,
        0.1,
        4,
        x_init,
        loop_count_init,
    )

    _compare_gsco_result(actual, expected)
    assert int(np.asarray(actual.history_length)) == 3
    np.testing.assert_allclose(np.asarray(actual.x), x_init, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    (
        "no_crossing",
        "no_new_coils",
        "match_current",
        "default_current",
        "max_current",
        "max_loop_count",
        "x_init_override",
    ),
    (
        (False, False, False, 0.25, 0.5, 1, None),
        (
            False,
            True,
            True,
            0.25,
            1.0,
            2,
            np.array([[0.25], [0.25], [-0.25], [-0.25], [0.0], [0.0]]),
        ),
        (True, False, False, 0.25, 1.0, 2, None),
    ),
)
def test_gsco_jax_matches_cpp_eligibility_options(
    no_crossing,
    no_new_coils,
    match_current,
    default_current,
    max_current,
    max_loop_count,
    x_init_override,
) -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    if x_init_override is not None:
        x_init = np.ascontiguousarray(x_init_override, dtype=np.float64)
    expected = _run_cpp_gsco(
        no_crossing,
        no_new_coils,
        match_current,
        A,
        b,
        abs(default_current),
        abs(max_current),
        abs(max_loop_count),
        loops,
        free_loops,
        segments,
        connections,
        0.2,
        4,
        x_init,
        loop_count_init,
    )

    actual = greedy_stellarator_coil_optimization_jax(
        no_crossing,
        no_new_coils,
        match_current,
        A,
        b,
        default_current,
        max_current,
        max_loop_count,
        loops,
        free_loops,
        segments,
        connections,
        0.2,
        4,
        x_init,
        loop_count_init,
    )

    _compare_gsco_result(actual, expected)


def test_gsco_wireframe_jax_wrapper_matches_cpp_without_mutating_wireframe() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    fixture = _GSCOFixture(
        currents=x_init.ravel().copy(),
        loops=loops,
        free_loops=free_loops,
        segments=segments,
        connected_segments=connections,
    )
    expected = _run_cpp_gsco(
        False,
        False,
        False,
        A,
        b,
        0.2,
        np.inf,
        0,
        loops,
        free_loops,
        segments,
        connections,
        0.15,
        5,
        x_init,
        loop_count_init,
    )

    actual = gsco_wireframe_jax(
        fixture,
        A,
        b,
        0.15,
        False,
        False,
        0.2,
        np.inf,
        5,
        1,
        loop_count_init=loop_count_init,
        verbose=False,
    )

    _compare_gsco_result(actual, expected)
    np.testing.assert_array_equal(fixture.currents, x_init.ravel())


def test_optimize_wireframe_jax_gsco_matches_public_cpu_and_iteration_helper() -> None:
    cpu_wireframe, A, b = _public_optimize_problem(seed=3106)
    jax_wireframe, _, _ = _public_optimize_problem(seed=3106)
    params = {
        "lambda_S": 0.1,
        "max_iter": 3,
        "print_interval": 1,
        "default_current": 0.2,
        "no_crossing": False,
        "match_current": False,
    }

    expected = optimize_wireframe(
        cpu_wireframe,
        "gsco",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )
    actual = optimize_wireframe_jax(
        jax_wireframe,
        "gsco",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )

    for key in ("x", "f_B", "f_S", "f", "f_B_hist", "f_S_hist", "f_hist"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=_RTOL, atol=_ATOL)
    for key in ("loop_count", "iter_hist", "loop_hist"):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_allclose(
        actual["curr_hist"],
        expected["curr_hist"],
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        jax_wireframe.currents,
        cpu_wireframe.currents,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        get_gsco_iteration_jax(2, actual, jax_wireframe),
        get_gsco_iteration(2, expected, cpu_wireframe),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_optimize_wireframe_jax_gsco_accepts_present_none_initial_state() -> None:
    cpu_wireframe, A, b = _public_optimize_problem(seed=3107)
    jax_wireframe, _, _ = _public_optimize_problem(seed=3107)
    params = {
        "lambda_S": 0.1,
        "max_iter": 2,
        "print_interval": 1,
        "default_current": 0.2,
        "no_crossing": False,
        "match_current": False,
        "x_init": None,
        "loop_count_init": None,
    }

    expected = optimize_wireframe(
        cpu_wireframe,
        "gsco",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )
    actual = optimize_wireframe_jax(
        jax_wireframe,
        "gsco",
        params,
        Amat=A,
        bvec=b,
        verbose=False,
    )

    np.testing.assert_allclose(actual["x"], expected["x"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(actual["loop_count"], expected["loop_count"])


def test_gsco_jax_jits_under_transfer_guard() -> None:
    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _gsco_problem()
    )
    A_device = jax.device_put(jnp.asarray(A))
    b_device = jax.device_put(jnp.asarray(b))
    loops_device = jax.device_put(jnp.asarray(loops, dtype=jnp.int32))
    free_loops_device = jax.device_put(jnp.asarray(free_loops, dtype=jnp.int32))
    segments_device = jax.device_put(jnp.asarray(segments, dtype=jnp.int32))
    connections_device = jax.device_put(jnp.asarray(connections, dtype=jnp.int32))
    x_init_device = jax.device_put(jnp.asarray(x_init))
    loop_count_device = jax.device_put(jnp.asarray(loop_count_init, dtype=jnp.int32))

    @jax.jit
    def _solve(
        A_data,
        b_data,
        loops_data,
        free_data,
        segments_data,
        connections_data,
        x_data,
        count_data,
    ):
        return greedy_stellarator_coil_optimization_jax(
            False,
            False,
            False,
            A_data,
            b_data,
            0.2,
            np.inf,
            0,
            loops_data,
            free_data,
            segments_data,
            connections_data,
            0.15,
            5,
            x_data,
            count_data,
        )

    _solve(
        A_device,
        b_device,
        loops_device,
        free_loops_device,
        segments_device,
        connections_device,
        x_init_device,
        loop_count_device,
    ).x.block_until_ready()
    with jax.transfer_guard("disallow"):
        out = _solve(
            A_device,
            b_device,
            loops_device,
            free_loops_device,
            segments_device,
            connections_device,
            x_init_device,
            loop_count_device,
        )
        out.x.block_until_ready()

    assert int(np.asarray(out.history_length)) == 5
    assert np.all(np.isfinite(np.asarray(out.f_history[:5])))
