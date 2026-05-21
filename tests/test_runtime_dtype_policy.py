from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

_FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_FLOOR = float(np.sqrt(np.finfo(np.float32).eps))
_FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_CAP = 1e-3
_RUNTIME_DTYPE_CASES = (
    ("jax_cpu_float32_smoke", jnp.float32),
    ("jax_cpu_parity", jnp.float64),
)


def _assert_arrays_dtype(expected_dtype, *arrays) -> None:
    expected = np.dtype(expected_dtype)
    for array in arrays:
        assert np.dtype(array.dtype) == expected


def _restore_backend_config(config) -> None:
    from simsopt.backend import set_backend

    set_backend(
        config.mode,
        strict=config.strict,
        debug_nans=config.debug_nans,
        disable_jit=config.disable_jit,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
        xla_gpu_preallocate=config.xla_gpu_preallocate,
        xla_gpu_mem_fraction=config.xla_gpu_mem_fraction,
        xla_gpu_allocator=config.xla_gpu_allocator,
        tf_gpu_allocator=config.tf_gpu_allocator,
        configure_runtime=False,
    )


@contextmanager
def _temporary_backend(mode: str):
    from simsopt.backend import get_backend_config, set_backend

    previous = get_backend_config()
    try:
        set_backend(mode, configure_runtime=False)
        yield
    finally:
        _restore_backend_config(previous)


def test_jax_mps_smoke_policy_runtime_dtype():
    from simsopt.backend import get_backend_policy

    with _temporary_backend("jax_mps_smoke"):
        policy = get_backend_policy()

        assert policy.runtime_dtype == "float32"
        assert policy.host_dtype == "float32"
        assert policy.tolerance_tier == "float32_smoke"
        assert policy.linear_solve_tolerance_floor == pytest.approx(
            _FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_FLOOR
        )
        assert policy.linear_solve_tolerance_cap == pytest.approx(
            _FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_CAP
        )


def test_jax_cpu_float32_smoke_policy_runtime_dtype():
    from simsopt.backend import get_backend_policy

    with _temporary_backend("jax_cpu_float32_smoke"):
        policy = get_backend_policy()

        assert policy.runtime_dtype == "float32"
        assert policy.host_dtype == "float32"
        assert policy.tolerance_tier == "float32_smoke"
        assert policy.linear_solve_tolerance_floor == pytest.approx(
            _FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_FLOOR
        )
        assert policy.linear_solve_tolerance_cap == pytest.approx(
            _FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_CAP
        )


def test_float64_modes_keep_float64_policy_dtype():
    from simsopt.backend import get_backend_policy

    for mode in (
        "native_cpu",
        "jax_cpu_fast",
        "jax_cpu_parity",
        "jax_gpu_fast",
        "jax_gpu_parity",
    ):
        policy = get_backend_policy(mode)

        assert policy.runtime_dtype == "float64"
        assert policy.host_dtype == "float64"
        assert policy.linear_solve_tolerance_floor == pytest.approx(1e-14)
        assert policy.linear_solve_tolerance_cap == pytest.approx(1e-10)


def test_require_runtime_dtype_follows_backend_policy():
    from simsopt.backend import set_backend
    from simsopt.jax_core._math_utils import require_runtime_dtype

    with _temporary_backend("jax_mps_smoke"):
        require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float32))
        with pytest.raises(TypeError, match="x must have runtime dtype float32"):
            require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float64))

        set_backend("jax_cpu_parity", configure_runtime=False)
        require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float64))
        with pytest.raises(TypeError, match="x must have runtime dtype float64"):
            require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float32))


def test_as_runtime_value_uses_policy_dtype_for_host_values():
    from simsopt.backend import set_backend
    from simsopt.jax_core._math_utils import as_runtime_value

    with _temporary_backend("jax_mps_smoke"):
        reference32 = jnp.asarray([0.0], dtype=jnp.float32)
        value32 = as_runtime_value(
            np.asarray([1.0, 2.0], dtype=np.float64),
            reference=reference32,
        )
        assert value32.dtype == jnp.float32

        set_backend("jax_cpu_parity", configure_runtime=False)
        reference64 = jnp.asarray([0.0], dtype=jnp.float64)
        value64 = as_runtime_value(
            np.asarray([1.0, 2.0], dtype=np.float32),
            reference=reference64,
        )
        assert value64.dtype == jnp.float64


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    _RUNTIME_DTYPE_CASES,
)
def test_squared_flux_field_dofs_follow_runtime_policy_dtype(mode, expected_dtype):
    from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX

    objective = object.__new__(SquaredFluxJAX)
    objective.field = SimpleNamespace(x=np.asarray([1.0, 2.0, 3.0], dtype=np.float64))

    with _temporary_backend(mode):
        assert objective._gather_field_free_dofs().dtype == expected_dtype


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    _RUNTIME_DTYPE_CASES,
)
def test_qfm_surface_coil_spec_dofs_follow_runtime_policy_dtype(mode, expected_dtype):
    from simsopt.geo.qfmsurface_jax import QfmSurfaceJAX

    class CapturingBiotSavart:
        def __init__(self) -> None:
            self.x = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
            self.observed_dtype = None

        def coil_set_spec_from_dofs(self, dofs):
            self.observed_dtype = dofs.dtype
            return dofs

    qfm_surface = object.__new__(QfmSurfaceJAX)

    with _temporary_backend(mode):
        biotsavart = CapturingBiotSavart()
        qfm_surface._coil_set_spec(biotsavart)
        assert biotsavart.observed_dtype == expected_dtype


def _pm_dtype_inputs():
    points = np.array(
        [
            [1.10, 0.20, -0.15],
            [1.35, -0.10, 0.22],
            [1.75, 0.30, 0.05],
            [2.05, -0.25, -0.18],
        ],
        dtype=np.float64,
        order="C",
    )
    normal = np.array(
        [
            [0.40, -0.10, 0.20],
            [-0.25, 0.55, 0.12],
            [0.30, 0.25, -0.35],
            [-0.45, -0.20, 0.30],
        ],
        dtype=np.float64,
        order="C",
    )
    Bn = np.array([[0.12, -0.18], [0.22, -0.05]], dtype=np.float64, order="C")
    dipoles = np.array(
        [
            [0.42, 0.18, -0.20],
            [0.75, -0.22, 0.31],
            [1.05, 0.12, 0.08],
        ],
        dtype=np.float64,
        order="C",
    )
    m_maxima = np.asarray([0.4, 0.6, 0.8], dtype=np.float64)
    pol_vectors = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    return points, normal, Bn, dipoles, m_maxima, pol_vectors


def _wireframe_dtype_arrays():
    loops = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    free_loops = np.ones((1,), dtype=np.int64)
    segments = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    connections = np.asarray(
        [[0, 3, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0], [2, 3, 0, 0]],
        dtype=np.int64,
    )
    A = np.asarray(
        [
            [1.0, -0.2, 0.1, 0.0],
            [0.0, 0.7, -0.3, 0.4],
            [0.3, 0.0, 0.5, -0.6],
            [-0.4, 0.2, 0.0, 0.8],
        ],
        dtype=np.float64,
    )
    b = np.asarray([[0.2], [-0.1], [0.05], [0.3]], dtype=np.float64)
    x_init = np.zeros((4, 1), dtype=np.float64)
    loop_count_init = np.zeros((1,), dtype=np.int64)
    return A, b, loops, free_loops, segments, connections, x_init, loop_count_init


def _wireframe_object_case():
    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe

    surface = SurfaceRZFourier(nfp=1, mpol=1, ntor=0)
    surface.set_rc(0, 0, 1.8)
    surface.set_rc(1, 0, 0.35)
    surface.set_zs(1, 0, 0.35)
    wireframe = ToroidalWireframe(surface, 2, 4)
    wireframe.currents[:] = np.linspace(
        -1.0,
        1.0,
        wireframe.n_segments,
        dtype=np.float64,
    )
    return wireframe, surface


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    _RUNTIME_DTYPE_CASES,
)
def test_pm_workflow_variants_follow_runtime_policy_dtype(mode, expected_dtype):
    from simsopt.jax_core.pm_optimization import (
        GPMOArbVecBacktrackingSpec,
        GPMOArbVecSpec,
        GPMOBacktrackingSpec,
        GPMOBaselineSpec,
        GPMOMultiSpec,
    )
    from simsopt.jax_core.pm_workflow import (
        pm_gpmo_arbvec_backtracking_initial_state,
        pm_gpmo_arbvec_backtracking_live_loop_jax,
        pm_gpmo_arbvec_initial_state,
        pm_gpmo_arbvec_live_loop_jax,
        pm_gpmo_baseline_initial_state,
        pm_gpmo_backtracking_initial_state,
        pm_gpmo_backtracking_live_loop_jax,
        pm_gpmo_live_loop_jax,
        pm_gpmo_multi_initial_state,
        pm_gpmo_multi_live_loop_jax,
    )

    _, _, _, dipoles, m_maxima, pol_vectors = _pm_dtype_inputs()
    ndipoles = dipoles.shape[0]
    A = np.asarray(
        [
            [0.8, -0.2, 0.1, 0.0, 0.3, -0.4, 0.2, 0.1, -0.3],
            [0.1, 0.7, -0.5, 0.2, -0.1, 0.0, 0.4, -0.2, 0.3],
            [-0.2, 0.1, 0.6, -0.3, 0.2, 0.5, 0.0, 0.1, -0.4],
            [0.3, 0.0, -0.1, 0.7, -0.2, 0.1, -0.5, 0.4, 0.2],
        ],
        dtype=np.float64,
    )
    b = np.asarray([0.25, -0.5, 0.75, -0.1], dtype=np.float64)
    reg_l2 = np.asarray(0.0, dtype=np.float64)
    baseline_spec = GPMOBaselineSpec(
        m_maxima=m_maxima,
        reg_l2=reg_l2,
        single_direction=-1,
    )
    multi_spec = GPMOMultiSpec(
        m_maxima=m_maxima,
        reg_l2=reg_l2,
        dipole_grid_xyz=dipoles,
        Nadjacent=1,
    )
    arbvec_spec = GPMOArbVecSpec(
        m_maxima=m_maxima,
        reg_l2=reg_l2,
        pol_vectors=pol_vectors,
    )
    backtracking_spec = GPMOBacktrackingSpec(
        m_maxima=m_maxima,
        reg_l2=reg_l2,
        dipole_grid_xyz=dipoles,
        Nadjacent=1,
        backtracking=1,
        max_nMagnets=2,
    )
    arbvec_backtracking_spec = GPMOArbVecBacktrackingSpec(
        m_maxima=m_maxima,
        reg_l2=reg_l2,
        dipole_grid_xyz=dipoles,
        pol_vectors=pol_vectors,
        Nadjacent=1,
        backtracking=1,
        max_nMagnets=2,
    )

    with _temporary_backend(mode):
        baseline_state = pm_gpmo_baseline_initial_state(
            A,
            b,
            ndipoles=ndipoles,
            history_capacity=2,
        )
        multi_state = pm_gpmo_multi_initial_state(
            A,
            b,
            multi_spec,
            history_capacity=2,
        )
        arbvec_state = pm_gpmo_arbvec_initial_state(
            A,
            b,
            arbvec_spec,
            history_capacity=2,
        )
        backtracking_state = pm_gpmo_backtracking_initial_state(
            A,
            b,
            backtracking_spec,
            history_capacity=2,
        )
        arbvec_backtracking_state = pm_gpmo_arbvec_backtracking_initial_state(
            A,
            b,
            arbvec_backtracking_spec,
            history_capacity=2,
        )

        results = (
            pm_gpmo_live_loop_jax(baseline_state, baseline_spec, A, max_steps=1),
            pm_gpmo_multi_live_loop_jax(multi_state, multi_spec, A, max_steps=1),
            pm_gpmo_arbvec_live_loop_jax(arbvec_state, arbvec_spec, A, max_steps=1),
            pm_gpmo_backtracking_live_loop_jax(
                backtracking_state,
                backtracking_spec,
                A,
                max_steps=1,
            ),
            pm_gpmo_arbvec_backtracking_live_loop_jax(
                arbvec_backtracking_state,
                arbvec_backtracking_spec,
                A,
                max_steps=1,
            ),
        )

        for state in (
            baseline_state,
            multi_state,
            arbvec_state,
            backtracking_state,
            arbvec_backtracking_state,
        ):
            _assert_arrays_dtype(expected_dtype, state.x, state.residual)

        for result in results:
            _assert_arrays_dtype(expected_dtype, result.x, result.residual_history)


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    _RUNTIME_DTYPE_CASES,
)
def test_pm_grid_and_solve_wrappers_follow_runtime_policy_dtype(mode, expected_dtype):
    from simsopt.geo.permanent_magnet_grid_jax import (
        PermanentMagnetGridJAX,
        mwpgp_alpha_from_grid,
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

    points, normal, Bn, dipoles, mmax, pol_vectors = _pm_dtype_inputs()
    moments = np.asarray(
        [[0.1, -0.2, 0.0], [0.05, 0.15, -0.1], [0.2, 0.0, 0.1]],
        dtype=np.float64,
    )

    with _temporary_backend(mode):
        grid = PermanentMagnetGridJAX.from_fixed_state(
            plasma_points=points,
            normal=normal,
            Bn=Bn,
            dipole_grid_xyz=dipoles,
            m_maxima=mmax,
            nfp=2,
            stellsym=True,
            coordinate_flag="cartesian",
            R0=1.25,
            m0=moments,
            pol_vectors=pol_vectors,
        )

        _assert_arrays_dtype(
            expected_dtype,
            grid.A_obj,
            grid.b_obj,
            grid.m0,
            grid.pol_vectors,
            mwpgp_alpha_from_grid(grid),
        )

        projected = projection_L2_balls_jax(moments, mmax)
        l0_thresholded = prox_l0_jax(moments, mmax, reg_l0=0.05, nu=1.5)
        l1_thresholded = prox_l1_jax(moments, mmax, reg_l1=0.05, nu=1.5)
        initial = setup_initial_condition_jax(grid, m0=moments)
        relaxed = relax_and_split_jax(
            grid,
            m0=moments,
            alpha=np.asarray(0.01, dtype=np.float64),
            max_iter=1,
            max_iter_RS=1,
            nu=10.0,
            reg_l1=0.01,
        )
        gpmo_results = (
            GPMO_baseline_jax(grid, K=1),
            GPMO_multi_jax(grid, K=1, Nadjacent=1),
            GPMO_ArbVec_jax(grid, K=1, pol_vectors=pol_vectors),
            GPMO_backtracking_jax(
                grid,
                K=1,
                Nadjacent=1,
                backtracking=1,
                max_nMagnets=2,
            ),
            GPMO_ArbVec_backtracking_jax(
                grid,
                K=1,
                Nadjacent=1,
                backtracking=1,
                max_nMagnets=2,
                pol_vectors=pol_vectors,
                m_init=moments,
            ),
        )

        for array in (projected, l0_thresholded, l1_thresholded, initial, relaxed.m):
            _assert_arrays_dtype(expected_dtype, array)
            assert jnp.all(jnp.isfinite(array))

        for result in gpmo_results:
            _assert_arrays_dtype(expected_dtype, result.m, result.residual_history)


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    _RUNTIME_DTYPE_CASES,
)
def test_wireframe_workflow_variants_follow_runtime_policy_dtype(mode, expected_dtype):
    from simsopt.jax_core.wireframe_workflow import WireframeGSCOLiveParams
    from simsopt.jax_core.wireframe_workflow import (
        greedy_stellarator_coil_optimization_jax,
        wireframe_gsco_multistep_loop_jax,
    )

    A, b, loops, free_loops, segments, connections, x_init, loop_count_init = (
        _wireframe_dtype_arrays()
    )

    with _temporary_backend(mode):
        A_device = jnp.asarray(A, dtype=expected_dtype)
        result = greedy_stellarator_coil_optimization_jax(
            False,
            False,
            False,
            A,
            b,
            0.25,
            1.0,
            2,
            loops,
            free_loops,
            segments,
            connections,
            0.1,
            1,
            x_init,
            loop_count_init,
        )
        sampled = greedy_stellarator_coil_optimization_jax(
            False,
            False,
            False,
            A,
            b,
            0.25,
            1.0,
            2,
            loops,
            free_loops,
            segments,
            connections,
            0.1,
            1,
            x_init,
            loop_count_init,
            record_every=2,
        )
        params = WireframeGSCOLiveParams(
            A=A_device,
            loops=jnp.asarray(loops, dtype=jnp.int32),
            free_loops=jnp.asarray(free_loops, dtype=jnp.int32),
            segments=jnp.asarray(segments, dtype=jnp.int32),
            connections=jnp.asarray(connections, dtype=jnp.int32),
            default_current=jnp.asarray(0.25, dtype=expected_dtype),
            max_current=jnp.asarray(1.0, dtype=expected_dtype),
            lambda_s=jnp.asarray(0.1, dtype=expected_dtype),
            tol=jnp.asarray(0.001, dtype=expected_dtype),
            max_loop_count=1,
            no_crossing=False,
            no_new_coils=False,
            match_current=False,
        )
        multistep = wireframe_gsco_multistep_loop_jax(
            params,
            b,
            x_init,
            loop_count_init,
            loops,
            np.asarray([[0, 0, 0, 0]], dtype=np.int64),
            np.zeros((x_init.size,), dtype=bool),
            max_iter_per_step=1,
            max_outer_steps=1,
            initial_current_fraction=0.25,
            current_scale=1.0,
            min_coil_size=1,
            final_max_current=1.0,
        )

        _assert_arrays_dtype(
            expected_dtype,
            result.x,
            result.f_history,
            result.curr_history,
            sampled.x,
            sampled.curr_history,
            multistep.x,
        )


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    _RUNTIME_DTYPE_CASES,
)
def test_wireframe_field_and_solve_wrappers_follow_runtime_policy_dtype(
    mode,
    expected_dtype,
):
    from simsopt.field.wireframefield_jax import WireframeFieldJAX
    from simsopt.solve.wireframe_optimization_jax import (
        bnorm_obj_matrices_jax,
        get_gsco_iteration_jax,
        gsco_wireframe_jax,
        optimize_wireframe_jax,
        rcls_wireframe_jax,
        regularized_constrained_least_squares_jax,
    )

    class RCLSWireframe:
        n_segments = 3

        def constraint_matrices(
            self,
            *,
            assume_no_crossings: bool,
            remove_constrained_segments: bool,
        ):
            assert assume_no_crossings is True
            assert remove_constrained_segments is True
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 1))

        def unconstrained_segments(self):
            return np.asarray([0, 2], dtype=np.int64)

    class GSCOWireframe:
        currents = np.zeros((4,), dtype=np.float64)
        segments = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        connected_segments = np.asarray(
            [[0, 3, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0], [2, 3, 0, 0]],
            dtype=np.int64,
        )

        def get_cell_key(self):
            return np.asarray([[0, 1, 2, 3]], dtype=np.int64)

        def get_free_cells(self, *, form: str):
            assert form == "logical"
            return np.ones((1,), dtype=np.int64)

    A, b, *_wireframe_arrays, x_init, loop_count_init = _wireframe_dtype_arrays()
    expected_host_dtype = np.dtype(expected_dtype)

    with _temporary_backend(mode):
        rcls_direct = regularized_constrained_least_squares_jax(
            np.eye(2, dtype=np.float64),
            np.asarray([[1.0], [0.25]], dtype=np.float64),
            0.1,
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            np.asarray([[0.0]], dtype=np.float64),
        )
        rcls_wrapped = rcls_wireframe_jax(
            RCLSWireframe(),
            np.eye(3, dtype=np.float64),
            np.asarray([[1.0], [0.5], [-0.25]], dtype=np.float64),
            0.1,
            assume_no_crossings=True,
        )
        gsco_wrapped = gsco_wireframe_jax(
            GSCOWireframe(),
            A,
            b,
            lambda_S=0.1,
            no_crossing=False,
            match_current=False,
            default_current=0.25,
            max_current=1.0,
            max_iter=2,
            print_interval=1,
            record_every=2,
        )
        wireframe, surface = _wireframe_object_case()
        field = WireframeFieldJAX(wireframe)
        field.set_points(np.ascontiguousarray(surface.gamma().reshape((-1, 3))))
        normal_matrix = field.dBnormal_by_dsegmentcurrents_matrix(surface)
        bnorm_target = np.zeros(surface.gamma().shape[:2], dtype=np.float64)
        identity = np.eye(wireframe.n_segments, dtype=np.float64)
        zero_currents = np.zeros((wireframe.n_segments, 1), dtype=np.float64)
        A_surface, b_surface = bnorm_obj_matrices_jax(
            wireframe,
            surface,
            bnorm_target=bnorm_target,
            verbose=False,
        )
        rcls_result = optimize_wireframe_jax(
            wireframe,
            "rcls",
            {"reg_W": 0.1, "assume_no_crossings": False},
            Amat=identity,
            bvec=zero_currents,
            verbose=False,
        )
        surface_result = optimize_wireframe_jax(
            wireframe,
            "rcls",
            {"reg_W": 0.1, "assume_no_crossings": False},
            surf_plas=surface,
            bnorm_target=bnorm_target,
            verbose=False,
        )
        gsco_sampled_result = optimize_wireframe_jax(
            wireframe,
            "gsco",
            {
                "lambda_S": 0.1,
                "max_iter": 2,
                "print_interval": 1,
                "default_current": 0.25,
                "no_crossing": False,
                "match_current": False,
                "record_every": 2,
            },
            Amat=identity,
            bvec=zero_currents,
            verbose=False,
        )
        gsco_result = optimize_wireframe_jax(
            wireframe,
            "gsco",
            {
                "lambda_S": 0.1,
                "max_iter": 2,
                "print_interval": 1,
                "default_current": 0.25,
                "no_crossing": False,
                "match_current": False,
            },
            Amat=identity,
            bvec=zero_currents,
            verbose=False,
        )
        replay = get_gsco_iteration_jax(0, gsco_result, wireframe)

        _assert_arrays_dtype(
            expected_dtype,
            rcls_direct,
            rcls_wrapped.x,
            gsco_wrapped.x,
            gsco_wrapped.curr_history,
            field._nodes_device,
            field._points_device,
        )
        _assert_arrays_dtype(
            expected_host_dtype,
            normal_matrix,
            A_surface,
            b_surface,
            rcls_result["x"],
            rcls_result["Amat"],
            surface_result["x"],
            surface_result["Amat"],
            surface_result["bvec"],
            gsco_sampled_result["x"],
            gsco_sampled_result["curr_hist"],
            gsco_result["x"],
            gsco_result["curr_hist"],
            replay,
        )
        assert jnp.all(jnp.isfinite(rcls_direct))


def test_axis0_entries_preserves_empty_axis0_tuple():
    from simsopt.jax_core._math_utils import axis0_entries

    assert axis0_entries(jnp.zeros((0, 3))) == ()


def test_surface_rzfourier_coefficient_scatter_forward_is_transfer_clean():
    from simsopt.jax_core.surface_rzfourier import _scatter_coefficients

    positions = np.asarray([0, 2, 4], dtype=np.int32)

    def scatter_from_dofs(dofs):
        return _scatter_coefficients(
            positions,
            dofs,
            target_size=8,
            source_offset=1,
        )

    dofs = jax.device_put(np.arange(5.0, dtype=np.float64))

    with jax.transfer_guard("disallow"):
        scattered = scatter_from_dofs(dofs)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(scattered)),
        np.asarray([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def test_surface_rzfourier_scatter_vjp_is_transfer_clean():
    from simsopt.jax_core.surface_rzfourier import _scatter_coefficients

    positions = np.asarray([0, 2, 4], dtype=np.int32)

    def scatter_from_dofs(dofs):
        return _scatter_coefficients(
            positions,
            dofs,
            target_size=8,
            source_offset=1,
        )

    dofs = jax.device_put(np.arange(5.0, dtype=np.float64))
    cotangent = jax.device_put(np.ones(8, dtype=np.float64))

    with jax.transfer_guard("disallow"):
        _, pullback = jax.vjp(scatter_from_dofs, dofs)
        (gradient,) = pullback(cotangent)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(gradient)),
        np.asarray([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64),
    )


def test_cws_rz_curve_pullback_is_transfer_clean():
    from simsopt.jax_core import (
        make_curve_cwsfourier_rz_spec,
        make_surface_rzfourier_spec,
    )
    from simsopt.jax_core.curve_geometry import (
        curve_gamma_and_dash_from_dofs,
        curve_pullback_from_dofs,
    )

    surface = make_surface_rzfourier_spec(
        rc=jnp.asarray([[1.0], [0.25]], dtype=jnp.float64),
        zs=jnp.asarray([[0.0], [0.2]], dtype=jnp.float64),
        quadpoints_phi=jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        quadpoints_theta=jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        nfp=1,
        stellsym=True,
    )
    curve = make_curve_cwsfourier_rz_spec(
        dofs=jnp.asarray([0.1, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=jnp.float64),
        quadpoints=jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        surface=surface,
        order=1,
    )
    gamma, gammadash = curve_gamma_and_dash_from_dofs(curve, curve.dofs)
    gamma_cotangent = jax.device_put(np.ones(gamma.shape, dtype=np.float64))
    gammadash_cotangent = jax.device_put(np.ones(gammadash.shape, dtype=np.float64))

    with jax.transfer_guard("disallow"):
        coeff_cotangent, surface_cotangent = curve_pullback_from_dofs(
            curve,
            curve.dofs,
            gamma_cotangent,
            gammadash_cotangent,
        )

    assert coeff_cotangent.shape == curve.dofs.shape
    assert surface_cotangent is not None
    assert surface_cotangent.shape == (3,)
    assert np.all(np.isfinite(np.asarray(jax.device_get(coeff_cotangent))))
    assert np.all(np.isfinite(np.asarray(jax.device_get(surface_cotangent))))


def test_as_runtime_float64_uses_runtime_policy_dtype_for_host_values():
    from simsopt.backend.dtypes import as_runtime_float64

    with _temporary_backend("jax_mps_smoke"):
        reference32 = jnp.asarray([0.0], dtype=jnp.float32)

        value32 = as_runtime_float64(
            np.asarray([1.0, 2.0], dtype=np.float64),
            reference=reference32,
        )

        assert value32.dtype == jnp.float32


def test_as_runtime_float64_alias_does_not_gate_on_host_reference_dtype():
    from simsopt.backend.dtypes import as_runtime_float64

    with _temporary_backend("jax_mps_smoke"):
        host64 = np.asarray([1.0, 2.0], dtype=np.float64)

        value32 = as_runtime_float64(host64, reference=host64)

        assert value32.dtype == jnp.float32


def test_as_jax_float64_compat_alias_uses_runtime_policy_dtype():
    from simsopt.backend.dtypes import as_jax_float64

    with _temporary_backend("jax_mps_smoke"):
        value = as_jax_float64(np.asarray([1.0, 2.0], dtype=np.float64))

        assert value.dtype == jnp.float32


def test_runtime_device_put_uses_policy_dtype_for_float_hosts():
    from simsopt.backend.dtypes import runtime_device_put

    with _temporary_backend("jax_mps_smoke"):
        value = runtime_device_put(np.asarray([1.0, 2.0], dtype=np.float64))
        indices = runtime_device_put([0, 1], dtype=np.int32)

        assert value.dtype == jnp.float32
        assert indices.dtype == jnp.int32


def test_runtime_device_put_resolves_explicit_float_dtype_through_policy():
    from simsopt.backend.dtypes import runtime_device_put

    with _temporary_backend("jax_mps_smoke"):
        value = runtime_device_put([1.0, 2.0], dtype=np.float64)

        assert value.dtype == jnp.float32


def test_boozer_optimizer_backend_auto_uses_policy_default():
    from simsopt.backend import set_backend
    from simsopt.geo import boozersurface_jax

    with _temporary_backend("native_cpu"):
        native_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        set_backend("jax_cpu_fast", configure_runtime=False)
        jax_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        assert native_options["optimizer_backend"] == "scipy"
        assert jax_options["optimizer_backend"] == "ondevice"

        set_backend("jax_mps_smoke", configure_runtime=False)
        mps_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        assert mps_options["optimizer_backend"] == "scipy"


def test_boozer_ls_mps_smoke_default_avoids_target_x64_gate(monkeypatch):
    from simsopt.geo import boozersurface_jax
    from simsopt.geo import optimizer_jax as optimizer_module

    with _temporary_backend("jax_mps_smoke"):
        monkeypatch.setattr(optimizer_module, "_x64_enabled", lambda: False)

        default_options = boozersurface_jax._normalize_solver_options({}, "ls")
        auto_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        assert default_options["optimizer_backend"] == "scipy"
        assert auto_options["optimizer_backend"] == "scipy"
        optimizer_module.require_target_backend_x64(
            default_options["optimizer_backend"]
        )
        optimizer_module.require_target_backend_x64(auto_options["optimizer_backend"])

        optimizer_module.require_target_backend_x64("ondevice")


def test_parity_target_backend_still_requires_x64(monkeypatch):
    from simsopt.geo import optimizer_jax as optimizer_module

    with _temporary_backend("jax_cpu_parity"):
        monkeypatch.setattr(optimizer_module, "_x64_enabled", lambda: False)

        with pytest.raises(RuntimeError, match="requires jax_enable_x64=True"):
            optimizer_module.require_target_backend_x64("ondevice")


def test_cpu_float32_smoke_target_backend_uses_float32_policy_gate(monkeypatch):
    from simsopt.geo import optimizer_jax as optimizer_module

    with _temporary_backend("jax_cpu_float32_smoke"):
        monkeypatch.setattr(optimizer_module, "_x64_enabled", lambda: False)

        optimizer_module.require_target_backend_x64("ondevice")


def test_boozer_ls_mps_smoke_default_reaches_reference_method(monkeypatch):
    from simsopt.geo import optimizer_jax as optimizer_module
    from simsopt.geo.boozersurface_jax import (
        BoozerSurfaceJAX,
        _normalize_solver_options,
    )

    with _temporary_backend("jax_mps_smoke"):
        monkeypatch.setattr(optimizer_module, "_x64_enabled", lambda: False)
        options = _normalize_solver_options({}, "ls")
        options["limited_memory"] = False
        options["force_ondevice_limited_memory"] = False

        with pytest.warns(RuntimeWarning, match="legacy adapter seam"):
            method = BoozerSurfaceJAX._resolve_optimizer_method(
                SimpleNamespace(options=options),
                optimize_G=True,
            )

        assert method == "bfgs"


def test_boozer_linearization_residency_uses_policy_default():
    from simsopt.backend import set_backend
    from simsopt.geo import boozersurface_jax

    with _temporary_backend("native_cpu"):
        native_options = boozersurface_jax._normalize_solver_options({}, "ls")

        set_backend("jax_cpu_fast", configure_runtime=False)
        jax_options = boozersurface_jax._normalize_solver_options({}, "ls")

        assert native_options["linearization_residency"] == "host"
        assert jax_options["linearization_residency"] == "device"


def test_boozer_residual_accepts_float32_under_mps_policy():
    from simsopt.geo.boozer_residual_jax import (
        boozer_residual_scalar,
        boozer_residual_vector,
    )

    with _temporary_backend("jax_mps_smoke"):
        B = jnp.ones((2, 3, 3), dtype=jnp.float32)
        xphi = jnp.full((2, 3, 3), 2.0, dtype=jnp.float32)
        xtheta = jnp.full((2, 3, 3), -0.5, dtype=jnp.float32)

        scalar_value = boozer_residual_scalar(
            np.float32(1.25),
            np.float32(-0.2),
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )
        vector_value = boozer_residual_vector(
            np.float32(1.25),
            np.float32(-0.2),
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )

        assert scalar_value.dtype == jnp.float32
        assert vector_value.dtype == jnp.float32
        assert bool(jnp.isfinite(scalar_value))
        assert bool(jnp.all(jnp.isfinite(vector_value)))
