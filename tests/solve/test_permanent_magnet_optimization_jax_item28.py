"""Item 28 tests for the fixed-state PM solve-level JAX wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsoptpp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.permanent_magnet_grid_jax import (
    PermanentMagnetGridJAX,
    mwpgp_alpha_from_grid,
)
from simsopt.jax_core.pm_optimization import PMOptimizationSpec, mwpgp_solve
from simsopt.solve.permanent_magnet_optimization import (
    GPMO,
    projection_L2_balls,
    prox_l0,
    prox_l1,
    setup_initial_condition,
)
from simsopt.solve.permanent_magnet_optimization_jax import (
    GPMO_ArbVec_backtracking_jax,
    GPMO_ArbVec_jax,
    GPMO_backtracking_jax,
    GPMO_baseline_jax,
    GPMO_multi_jax,
    projection_L2_balls_jax,
    prox_l0_jax,
    prox_l1_jax,
    relax_and_split_jax,
    setup_initial_condition_jax,
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_PM_FIXED_STEP = parity_ladder_tolerances("pm_mwpgp_fixed_step")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]
_STATE_TRACE_RTOL = _PM_FIXED_STEP["state_trace_rtol"]
_STATE_TRACE_ATOL = _PM_FIXED_STEP["state_trace_atol"]


@dataclass
class _InitialConditionCPUGrid:
    ndipoles: int
    m_maxima: np.ndarray
    m0: np.ndarray


@dataclass(frozen=True)
class _PlasmaBoundary:
    nfp: int
    stellsym: bool


@dataclass(frozen=True)
class _GPMOPlasmaBoundary:
    normal_array: np.ndarray
    nfp: int = 1
    stellsym: bool = False

    def normal(self) -> np.ndarray:
        return self.normal_array


@dataclass(frozen=True)
class _CPUGrid:
    A_obj: np.ndarray
    b_obj: np.ndarray
    ATb: np.ndarray
    ATA_scale: float
    m0: np.ndarray
    m: np.ndarray
    m_proxy: np.ndarray
    m_maxima: np.ndarray
    dipole_grid_xyz: np.ndarray
    coordinate_flag: str
    R0: float
    plasma_boundary: _PlasmaBoundary
    nphi: int
    ntheta: int
    ndipoles: int


@dataclass
class _GPMOCPUGrid:
    A_obj: np.ndarray
    b_obj: np.ndarray
    ATb: np.ndarray
    ATA_scale: float
    m0: np.ndarray
    m: np.ndarray
    m_proxy: np.ndarray
    m_maxima: np.ndarray
    dipole_grid_xyz: np.ndarray
    coordinate_flag: str
    R0: float
    plasma_boundary: _GPMOPlasmaBoundary
    nphi: int
    ntheta: int
    ndipoles: int


def _synthetic_grid(seed: int = 28) -> PermanentMagnetGridJAX:
    rng = np.random.default_rng(seed)
    ndipoles = 4
    nquad = 17
    A_obj = np.ascontiguousarray(rng.standard_normal(size=(nquad, ndipoles * 3)))
    b_obj = np.ascontiguousarray(rng.standard_normal(size=(nquad,)))
    ATb = A_obj.T @ b_obj
    ATA_scale = float(np.linalg.svd(A_obj, compute_uv=False)[0] ** 2)
    m0 = np.zeros(ndipoles * 3, dtype=np.float64)
    m_maxima = np.full(ndipoles, 0.55, dtype=np.float64)
    dipoles = np.ascontiguousarray(rng.standard_normal(size=(ndipoles, 3)))
    return PermanentMagnetGridJAX.from_cpu(
        _CPUGrid(
            A_obj=A_obj,
            b_obj=b_obj,
            ATb=ATb,
            ATA_scale=ATA_scale,
            m0=m0,
            m=m0,
            m_proxy=m0,
            m_maxima=m_maxima,
            dipole_grid_xyz=dipoles,
            coordinate_flag="cartesian",
            R0=0.0,
            plasma_boundary=_PlasmaBoundary(nfp=1, stellsym=False),
            nphi=1,
            ntheta=nquad,
            ndipoles=ndipoles,
        )
    )


def _gpmo_cpu_grid(seed: int = 2801) -> _GPMOCPUGrid:
    rng = np.random.default_rng(seed)
    ndipoles = 5
    nquad = 13
    A_obj = np.ascontiguousarray(rng.standard_normal(size=(nquad, ndipoles * 3)))
    b_obj = np.ascontiguousarray(rng.standard_normal(size=(nquad,)))
    ATb = A_obj.T @ b_obj
    ATA_scale = float(np.linalg.svd(A_obj, compute_uv=False)[0] ** 2)
    m0 = np.zeros(ndipoles * 3, dtype=np.float64)
    m_maxima = 0.3 + rng.random(size=ndipoles)
    dipoles = np.ascontiguousarray(rng.standard_normal(size=(ndipoles, 3)))
    normal = np.ones((nquad, 3), dtype=np.float64)
    return _GPMOCPUGrid(
        A_obj=A_obj,
        b_obj=b_obj,
        ATb=ATb,
        ATA_scale=ATA_scale,
        m0=m0,
        m=m0.copy(),
        m_proxy=m0.copy(),
        m_maxima=m_maxima,
        dipole_grid_xyz=dipoles,
        coordinate_flag="cartesian",
        R0=0.0,
        plasma_boundary=_GPMOPlasmaBoundary(normal_array=normal),
        nphi=1,
        ntheta=nquad,
        ndipoles=ndipoles,
    )


def _gpmo_pol_vectors(seed: int, ndipoles: int, n_vectors: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(size=(ndipoles, n_vectors, 3))
    norms = np.linalg.norm(raw, axis=2)
    return np.ascontiguousarray(raw / norms[:, :, None])


def test_projection_and_prox_helpers_match_cpu_oracles():
    rng = np.random.default_rng(4)
    moments = rng.standard_normal(size=12)
    m_maxima = np.array([0.4, 0.6, 0.8, 1.0], dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(projection_L2_balls_jax(moments, m_maxima)),
        projection_L2_balls(moments, m_maxima),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(prox_l0_jax(moments, m_maxima, reg_l0=0.1, nu=0.5)),
        prox_l0(moments, m_maxima, reg_l0=0.1, nu=0.5),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(prox_l1_jax(moments, m_maxima, reg_l1=0.1, nu=0.5)),
        prox_l1(moments, m_maxima, reg_l1=0.1, nu=0.5),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_setup_initial_condition_matches_cpu_projection_contract():
    m_maxima = np.array([0.5, 0.7], dtype=np.float64)
    m0 = np.array([0.2, 0.1, -0.3, 0.0, 0.6, 0.1], dtype=np.float64)
    cpu_grid = _InitialConditionCPUGrid(
        ndipoles=2,
        m_maxima=m_maxima,
        m0=np.zeros(6, dtype=np.float64),
    )
    setup_initial_condition(cpu_grid, m0)

    grid = _synthetic_grid()
    projected = setup_initial_condition_jax(
        PermanentMagnetGridJAX(
            A_obj=grid.A_obj[:2, :6],
            b_obj=grid.b_obj[:2],
            ATb=grid.ATb[:2],
            ATA_scale=grid.ATA_scale,
            m0=jnp.zeros((2, 3), dtype=jnp.float64),
            m=jnp.zeros((2, 3), dtype=jnp.float64),
            m_proxy=jnp.zeros((2, 3), dtype=jnp.float64),
            m_maxima=jnp.asarray(m_maxima),
            dipole_grid_xyz=grid.dipole_grid_xyz[:2],
            coordinate_flag="cartesian",
            R0=0.0,
            nfp=1,
            stellsym=False,
            nphi=1,
            ntheta=2,
            ndipoles=2,
        ),
        m0,
    )

    np.testing.assert_allclose(
        np.asarray(projected).reshape(-1),
        cpu_grid.m0,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_setup_initial_condition_jax_explicit_m0_is_traceable_under_guard():
    grid = _synthetic_grid()
    m0 = jax.device_put(
        jnp.asarray(
            np.array(
                [
                    [0.2, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, -0.3],
                    [0.2, -0.1, 0.0],
                ],
                dtype=np.float64,
            )
        )
    )
    expected = projection_L2_balls(
        np.asarray(m0).reshape(-1), np.asarray(grid.m_maxima)
    )

    with jax.transfer_guard("disallow"):
        direct = setup_initial_condition_jax(grid, m0)
        direct.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(direct).reshape(-1),
        expected,
        rtol=_RTOL,
        atol=_ATOL,
    )

    @jax.jit
    def _setup(grid_data: PermanentMagnetGridJAX, moments: jax.Array):
        return setup_initial_condition_jax(grid_data, moments)

    _setup(grid, m0).block_until_ready()
    with jax.transfer_guard("disallow"):
        projected = _setup(grid, m0)
        projected.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(projected).reshape(-1),
        expected,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_setup_initial_condition_jax_rejects_infeasible_explicit_m0():
    grid = _synthetic_grid()
    m0 = jax.device_put(
        jnp.asarray(
            np.array(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, -3.0],
                    [0.2, -0.1, 0.0],
                ],
                dtype=np.float64,
            )
        )
    )

    with pytest.raises(ValueError, match="maximum bound constraints"):
        setup_initial_condition_jax(grid, m0)


def test_relax_and_split_jax_matches_direct_mwpgp_fixed_step():
    grid = _synthetic_grid()
    n_steps = 7
    result = relax_and_split_jax(grid, max_iter=n_steps, reg_l2=0.0)
    spec = PMOptimizationSpec(
        m_maxima=grid.m_maxima,
        m_proxy=grid.m0,
        nu=jnp.asarray(1.0e100, dtype=jnp.float64),
        reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
        alpha=mwpgp_alpha_from_grid(grid),
    )
    expected, expected_history = mwpgp_solve(
        spec,
        grid.A_obj,
        grid.ATb,
        grid.m0,
        n_steps=n_steps,
    )

    np.testing.assert_allclose(
        np.asarray(result.m),
        np.asarray(expected),
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.errors),
        np.asarray(expected_history[-1:]),
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    assert result.m_history.shape == (1, grid.ndipoles, 3)
    assert result.m_proxy_history.shape == (0, grid.ndipoles, 3)


def test_relax_and_split_jax_matches_cpp_mwpgp_oracle_one_convex_step():
    grid = _synthetic_grid(seed=2028)
    n_steps = 5
    alpha = float(np.asarray(mwpgp_alpha_from_grid(grid)))

    _, _, _, cpp_m = simsoptpp.MwPGP_algorithm(
        np.asarray(grid.A_obj),
        np.asarray(grid.b_obj),
        np.asarray(grid.ATb),
        np.asarray(grid.m0),
        np.asarray(grid.m0),
        np.asarray(grid.m_maxima),
        alpha,
        1.0e100,
        0.0,
        0.0,
        0.0,
        0.0,
        n_steps,
        0.0,
        False,
    )
    result = relax_and_split_jax(grid, max_iter=n_steps)

    np.testing.assert_allclose(
        np.asarray(result.m),
        cpp_m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )


def test_relax_and_split_jax_updates_l0_proxy_for_fixed_outer_steps():
    grid = _synthetic_grid(seed=2030)
    result = relax_and_split_jax(
        grid,
        max_iter=4,
        max_iter_RS=2,
        nu=5.0,
        reg_l0=0.02,
    )
    expected_proxy = prox_l0_jax(result.m, grid.m_maxima, reg_l0=0.02, nu=5.0)
    initial_proxy = prox_l0_jax(grid.m0, grid.m_maxima, reg_l0=0.02, nu=5.0)
    proxies_used = jnp.concatenate(
        [initial_proxy[None, :, :], result.m_proxy_history[:-1]], axis=0
    )
    expected_errors = []
    for step in range(2):
        m_step = result.m_history[step]
        residual = grid.A_obj @ jnp.reshape(m_step, (-1,)) - grid.b_obj
        r2 = 0.5 * jnp.sum(residual * residual)
        n2 = 0.5 * jnp.sum((m_step - proxies_used[step]) ** 2) / 5.0
        expected_errors.append(r2 + n2)

    assert result.errors.shape == (2,)
    assert result.m_history.shape == (2, grid.ndipoles, 3)
    assert result.m_proxy_history.shape == (2, grid.ndipoles, 3)
    np.testing.assert_allclose(
        np.asarray(result.errors),
        np.asarray(expected_errors),
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.m_proxy),
        np.asarray(expected_proxy),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_relax_and_split_jax_jits_under_strict_transfer_guard():
    grid = _synthetic_grid(seed=2031)
    m0 = jax.device_put(jnp.asarray(np.asarray(grid.m0) + 0.05))

    @jax.jit
    def _run(grid_data: PermanentMagnetGridJAX, initial: jax.Array):
        return relax_and_split_jax(grid_data, initial, max_iter=3).m

    _run(grid, m0).block_until_ready()
    with jax.transfer_guard("disallow"):
        out = _run(grid, m0)
        out.block_until_ready()

    assert out.shape == (grid.ndipoles, 3)
    assert np.all(np.isfinite(np.asarray(out)))


def test_gpmo_baseline_jax_matches_cpu_baseline_wrapper():
    cpu_grid = _gpmo_cpu_grid(seed=2802)
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    K = 4
    reg_l2 = 0.15
    GPMO(
        cpu_grid,
        algorithm="baseline",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
        single_direction=-1,
    )
    result = GPMO_baseline_jax(jax_grid, K=K, reg_l2=reg_l2, single_direction=-1)

    assert result.m_history.shape == (K, jax_grid.ndipoles, 3)
    assert result.x_history.shape == (K, jax_grid.ndipoles, 3)
    np.testing.assert_allclose(
        np.asarray(result.m_history),
        np.asarray(result.x_history) * np.asarray(jax_grid.m_maxima)[None, :, None],
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.m_history[-1]).reshape(-1),
        np.asarray(result.m).reshape(-1),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    residual = np.asarray(jax_grid.A_obj) @ np.asarray(result.m).reshape(-1)
    residual -= np.asarray(jax_grid.b_obj)
    np.testing.assert_allclose(
        np.asarray(result.residual),
        residual,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_gpmo_multi_jax_matches_cpu_multi_wrapper():
    cpu_grid = _gpmo_cpu_grid(seed=2803)
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    K = 2
    Nadjacent = 2
    reg_l2 = 0.0
    GPMO(
        cpu_grid,
        algorithm="multi",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
        dipole_grid_xyz=cpu_grid.dipole_grid_xyz,
        single_direction=-1,
        Nadjacent=Nadjacent,
    )
    result = GPMO_multi_jax(
        jax_grid,
        K=K,
        reg_l2=reg_l2,
        single_direction=-1,
        Nadjacent=Nadjacent,
    )

    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    residual = np.asarray(jax_grid.A_obj) @ np.asarray(result.m).reshape(-1)
    residual -= np.asarray(jax_grid.b_obj)
    np.testing.assert_allclose(
        np.asarray(result.residual),
        residual,
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("single_direction", (-1, 0, 1, 2))
def test_gpmo_multi_jax_wrapper_matches_cpu_multi_with_l2_and_direction(
    single_direction: int,
):
    cpu_grid = _gpmo_cpu_grid(seed=2810 + single_direction)
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    K = 2
    Nadjacent = 2
    reg_l2 = 0.13
    GPMO(
        cpu_grid,
        algorithm="multi",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
        dipole_grid_xyz=cpu_grid.dipole_grid_xyz,
        single_direction=single_direction,
        Nadjacent=Nadjacent,
    )
    result = GPMO_multi_jax(
        jax_grid,
        K=K,
        reg_l2=reg_l2,
        single_direction=single_direction,
        Nadjacent=Nadjacent,
    )

    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )


def test_gpmo_arbvec_jax_matches_cpu_arbvec_wrapper():
    cpu_grid = _gpmo_cpu_grid(seed=2820)
    pol_vectors = _gpmo_pol_vectors(seed=2821, ndipoles=cpu_grid.ndipoles)
    cpu_grid.pol_vectors = pol_vectors
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    K = 3
    reg_l2 = 0.12
    GPMO(
        cpu_grid,
        algorithm="ArbVec",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
    )
    result = GPMO_ArbVec_jax(jax_grid, K=K, reg_l2=reg_l2)

    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    residual = np.asarray(jax_grid.A_obj) @ np.asarray(result.m).reshape(-1)
    residual -= np.asarray(jax_grid.b_obj)
    np.testing.assert_allclose(
        np.asarray(result.residual),
        residual,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_gpmo_arbvec_jax_uses_from_fixed_state_pol_vectors():
    rng = np.random.default_rng(2822)
    ndipoles = 4
    nphi = 2
    ntheta = 3
    nquad = nphi * ntheta
    points = np.ascontiguousarray(rng.standard_normal(size=(nquad, 3)))
    normal = np.ascontiguousarray(rng.standard_normal(size=(nquad, 3)))
    Bn = np.ascontiguousarray(rng.standard_normal(size=(nphi, ntheta)))
    dipoles = np.ascontiguousarray(rng.standard_normal(size=(ndipoles, 3)))
    m_maxima = np.ascontiguousarray(0.3 + rng.random(size=ndipoles))
    pol_vectors = _gpmo_pol_vectors(seed=2823, ndipoles=ndipoles, n_vectors=4)
    grid = PermanentMagnetGridJAX.from_fixed_state(
        plasma_points=points,
        normal=normal,
        Bn=Bn,
        dipole_grid_xyz=dipoles,
        m_maxima=m_maxima,
        nfp=1,
        stellsym=False,
        coordinate_flag="cartesian",
        R0=0.0,
        pol_vectors=pol_vectors,
    )
    K = 3
    reg_l2 = 0.11
    mmax_vec = np.repeat(m_maxima, 3)
    A_scaled = np.asarray(grid.A_obj) * mmax_vec[None, :]
    _, _, _, x_cpp = simsoptpp.GPMO_ArbVec(
        np.ascontiguousarray(A_scaled.T),
        np.asarray(grid.b_obj),
        np.sqrt(reg_l2) * mmax_vec,
        np.linalg.norm(normal, axis=1),
        pol_vectors,
        K=K,
        verbose=False,
        nhistory=K,
    )
    result = GPMO_ArbVec_jax(grid, K=K, reg_l2=reg_l2)

    np.testing.assert_allclose(
        np.asarray(result.x),
        x_cpp,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )


def test_gpmo_arbvec_jax_accepts_explicit_pol_vectors_without_grid_staging():
    cpu_grid = _gpmo_cpu_grid(seed=2824)
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    assert jax_grid.pol_vectors is None
    pol_vectors = _gpmo_pol_vectors(seed=2825, ndipoles=cpu_grid.ndipoles)
    cpu_grid.pol_vectors = pol_vectors
    K = 3
    reg_l2 = 0.12
    GPMO(
        cpu_grid,
        algorithm="ArbVec",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
    )
    result = GPMO_ArbVec_jax(jax_grid, K=K, reg_l2=reg_l2, pol_vectors=pol_vectors)

    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )


def test_gpmo_backtracking_jax_matches_cpu_backtracking_wrapper():
    cpu_grid = _gpmo_cpu_grid(seed=2830)
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    K = 8
    Nadjacent = 3
    backtracking = 2
    max_nMagnets = cpu_grid.ndipoles
    reg_l2 = 0.0
    GPMO(
        cpu_grid,
        algorithm="backtracking",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
        dipole_grid_xyz=cpu_grid.dipole_grid_xyz,
        single_direction=-1,
        Nadjacent=Nadjacent,
        backtracking=backtracking,
        max_nMagnets=max_nMagnets,
    )
    result = GPMO_backtracking_jax(
        jax_grid,
        K=K,
        reg_l2=reg_l2,
        single_direction=-1,
        Nadjacent=Nadjacent,
        backtracking=backtracking,
        max_nMagnets=max_nMagnets,
    )

    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    residual = np.asarray(jax_grid.A_obj) @ np.asarray(result.m).reshape(-1)
    residual -= np.asarray(jax_grid.b_obj)
    np.testing.assert_allclose(
        np.asarray(result.residual),
        residual,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_gpmo_backtracking_jax_jits_under_strict_transfer_guard():
    jax_grid = PermanentMagnetGridJAX.from_cpu(_gpmo_cpu_grid(seed=2831))

    @jax.jit
    def _run(grid_data: PermanentMagnetGridJAX):
        return GPMO_backtracking_jax(
            grid_data,
            K=6,
            Nadjacent=2,
            backtracking=2,
            max_nMagnets=4,
        ).m

    _run(jax_grid).block_until_ready()
    with jax.transfer_guard("disallow"):
        out = _run(jax_grid)
        out.block_until_ready()

    assert out.shape == (jax_grid.ndipoles, 3)
    assert np.all(np.isfinite(np.asarray(out)))


def test_gpmo_arbvec_backtracking_jax_matches_cpu_arbvec_backtracking_wrapper():
    cpu_grid = _gpmo_cpu_grid(seed=2840)
    pol_vectors = _gpmo_pol_vectors(seed=2841, ndipoles=cpu_grid.ndipoles)
    cpu_grid.pol_vectors = pol_vectors
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    K = 8
    Nadjacent = 3
    backtracking = 2
    thresh_angle = float(np.pi / 4.0)
    max_nMagnets = cpu_grid.ndipoles
    reg_l2 = 0.0
    GPMO(
        cpu_grid,
        algorithm="ArbVec_backtracking",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
        dipole_grid_xyz=cpu_grid.dipole_grid_xyz,
        Nadjacent=Nadjacent,
        backtracking=backtracking,
        thresh_angle=thresh_angle,
        max_nMagnets=max_nMagnets,
    )
    result = GPMO_ArbVec_backtracking_jax(
        jax_grid,
        K=K,
        reg_l2=reg_l2,
        Nadjacent=Nadjacent,
        backtracking=backtracking,
        thresh_angle=thresh_angle,
        max_nMagnets=max_nMagnets,
    )
    assert result.m_history.shape == (K, jax_grid.ndipoles, 3)
    assert result.x_history.shape == (K, jax_grid.ndipoles, 3)
    np.testing.assert_allclose(
        np.asarray(result.m_history),
        np.asarray(result.x_history) * np.asarray(jax_grid.m_maxima)[None, :, None],
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    residual = np.asarray(jax_grid.A_obj) @ np.asarray(result.m).reshape(-1)
    residual -= np.asarray(jax_grid.b_obj)
    np.testing.assert_allclose(
        np.asarray(result.residual),
        residual,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_gpmo_arbvec_backtracking_jax_with_m_init():
    """Host wrapper threads ``m_init`` through to the normalized init routine."""

    cpu_grid = _gpmo_cpu_grid(seed=2842)
    pol_vectors = _gpmo_pol_vectors(seed=2843, ndipoles=cpu_grid.ndipoles)
    cpu_grid.pol_vectors = pol_vectors
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    K = 6
    Nadjacent = 2
    backtracking = 2
    thresh_angle = float(np.pi)
    max_nMagnets = cpu_grid.ndipoles
    reg_l2 = 0.0

    # Build a physical ``m_init`` seeded by the first signed pol vector.
    m_init = np.zeros((cpu_grid.ndipoles, 3))
    m_init[0] = pol_vectors[0, 0] * cpu_grid.m_maxima[0]
    m_init[2] = -pol_vectors[2, 1] * cpu_grid.m_maxima[2]

    GPMO(
        cpu_grid,
        algorithm="ArbVec_backtracking",
        K=K,
        nhistory=K,
        verbose=False,
        reg_l2=reg_l2,
        dipole_grid_xyz=cpu_grid.dipole_grid_xyz,
        Nadjacent=Nadjacent,
        backtracking=backtracking,
        thresh_angle=thresh_angle,
        max_nMagnets=max_nMagnets,
        m_init=m_init,
    )
    result = GPMO_ArbVec_backtracking_jax(
        jax_grid,
        K=K,
        reg_l2=reg_l2,
        Nadjacent=Nadjacent,
        backtracking=backtracking,
        thresh_angle=thresh_angle,
        max_nMagnets=max_nMagnets,
        m_init=m_init,
    )
    np.testing.assert_allclose(
        np.asarray(result.m).reshape(-1),
        cpu_grid.m,
        rtol=_STATE_TRACE_RTOL,
        atol=_STATE_TRACE_ATOL,
    )
    assert int(result.initial_num_nonzero) == 2


def test_gpmo_arbvec_backtracking_jax_jits_under_strict_transfer_guard():
    cpu_grid = _gpmo_cpu_grid(seed=2844)
    pol_vectors = _gpmo_pol_vectors(seed=2845, ndipoles=cpu_grid.ndipoles)
    cpu_grid.pol_vectors = pol_vectors
    jax_grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)

    @jax.jit
    def _run(grid_data: PermanentMagnetGridJAX):
        return GPMO_ArbVec_backtracking_jax(
            grid_data,
            K=6,
            Nadjacent=2,
            backtracking=2,
            thresh_angle=float(np.pi / 3.0),
            max_nMagnets=4,
        ).m

    _run(jax_grid).block_until_ready()
    with jax.transfer_guard("disallow"):
        out = _run(jax_grid)
        out.block_until_ready()

    assert out.shape == (jax_grid.ndipoles, 3)
    assert np.all(np.isfinite(np.asarray(out)))


def test_general_greedy_gpmo_is_not_claimed_by_the_jax_wrapper():
    import simsopt.solve.permanent_magnet_optimization_jax as pmo_jax

    assert hasattr(pmo_jax, "GPMO_ArbVec_backtracking_jax")
    assert hasattr(pmo_jax, "GPMO_ArbVec_jax")
    assert hasattr(pmo_jax, "GPMO_backtracking_jax")
    assert hasattr(pmo_jax, "GPMO_baseline_jax")
    assert hasattr(pmo_jax, "GPMO_multi_jax")
    assert not hasattr(pmo_jax, "GPMO_jax")
    with pytest.raises(AttributeError):
        getattr(pmo_jax, "GPMO")
