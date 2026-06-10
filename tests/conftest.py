"""Root-level test fixtures shared across the JAX test suite."""

from __future__ import annotations

from contextlib import contextmanager
import os
import shlex
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_REPO_ROOT / "src")

import numpy as np
import pytest

from simsopt_jax.backend import get_backend_config, invalidate_backend_cache, set_backend

try:
    import jax
except ModuleNotFoundError:
    jax = None


def _force_x64(jax_module) -> None:
    jax_module.config.update("jax_enable_x64", True)
    if jax_module.config.jax_enable_x64 is not True:
        raise RuntimeError("tests/conftest.py requires jax_enable_x64=True")


if jax is not None:
    _force_x64(jax)

_BACKEND_RUNTIME_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_DEBUG",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_DISABLE_JIT",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
    "SIMSOPT_JAX_POINT_CHUNK_SIZE",
    "SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE",
    "SIMSOPT_JAX_CHUNK_AUTOTUNE",
    "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB",
    "SIMSOPT_JAX_GPU_PREALLOCATE",
    "SIMSOPT_JAX_GPU_MEM_FRACTION",
    "SIMSOPT_JAX_GPU_ALLOCATOR",
    "SIMSOPT_TF_GPU_ALLOCATOR",
    "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_CPU",
    "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU",
    "SIMSOPT_JAX_SHARDING",
    "SIMSOPT_JAX_SHARDING_AXIS",
    "SIMSOPT_JAX_COIL_SHARDING_AXIS",
    "SIMSOPT_JAX_MIN_POINTS_TO_SHARD",
    "SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD",
    "SIMSOPT_JAX_MIN_COILS_TO_SHARD",
    "SIMSOPT_JAX_DISTRIBUTED_INIT",
    "SIMSOPT_JAX_COORDINATOR_ADDRESS",
    "SIMSOPT_JAX_NUM_PROCESSES",
    "SIMSOPT_JAX_PROCESS_ID",
    "SIMSOPT_JAX_LOCAL_DEVICE_IDS",
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_CLIENT_MEM_FRACTION",
    "TF_GPU_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
)
_JAX_RUNTIME_CONFIG_DEFAULTS = {
    "jax_enable_x64": True,
    "jax_debug_nans": False,
    "jax_transfer_guard": None,
    "jax_platforms": None,
    "jax_compilation_cache_dir": None,
}
_PARITY_SEED_BASE = 1729
_PARITY_LANE_TO_MODE = {
    "cpu": "jax_cpu_parity",
    "gpu": "jax_gpu_parity",
}
_PARITY_MODE_TO_LANE = {
    "jax_cpu_parity": "cpu",
    "jax_gpu_parity": "gpu",
    "jax_gpu_fast": "gpu",
}
_REDUCTION_ACCEPTANCE_TIERS = {
    "biotsavart_chunked_dense": {
        "cpu": (1e-12, 1e-14),
        "gpu": (1e-12, 1e-13),
    },
    "biotsavart_accumulation_order": {
        "cpu": (1e-12, 1e-14),
        "gpu": (1e-12, 2e-13),
    },
    "integral_bdotn_normalized_stress": {
        "cpu": (1e-12, 1e-14),
        "gpu": (1e-12, 1e-14),
    },
    "boozer_residual_floor_vector": {
        "cpu": (1e-12, 1e-24),
        "gpu": (1e-10, 1e-22),
    },
    "boozer_residual_floor_scalar": {
        "cpu": (1e-12, 1e-15),
        "gpu": (1e-10, 1e-14),
    },
}
_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_exclude_nondeterministic_ops",)
_STALE_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_deterministic_ops",)
_DEFAULT_GPU_DETERMINISM_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"


def _require_jax():
    if jax is None:
        pytest.skip("JAX not installed in current environment")
    return jax


class _TwoRankReplayComm:
    size = 2

    def __init__(self, rank, expected_tys, expected_hits):
        self.rank = rank
        self._expected_tys = expected_tys
        self._expected_hits = expected_hits
        self._allgather_calls = 0

    def allgather(self, value):
        expected = (self._expected_tys, self._expected_hits)[self._allgather_calls]
        _assert_list_arrays_close(value, expected[self.rank])
        self._allgather_calls += 1
        return expected

    def allreduce(self, value):
        return value


def _assert_list_arrays_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_array, expected_array)


def _flatten_rank_values(rank_values):
    return [item for values in rank_values for item in values]


def _assert_two_rank_replay_matches(
    no_comm_tys,
    no_comm_hits,
    run_with_comm,
    *,
    expected_tys_by_rank=None,
    expected_hits_by_rank=None,
):
    expected_tys = (
        expected_tys_by_rank
        if expected_tys_by_rank is not None
        else [no_comm_tys[:1], no_comm_tys[1:]]
    )
    expected_hits = (
        expected_hits_by_rank
        if expected_hits_by_rank is not None
        else [no_comm_hits[:1], no_comm_hits[1:]]
    )
    expected_global_tys = _flatten_rank_values(expected_tys)
    expected_global_hits = _flatten_rank_values(expected_hits)
    assert len(expected_global_tys) == len(no_comm_tys) == 2
    assert len(expected_global_hits) == len(no_comm_hits) == 2
    for rank in range(_TwoRankReplayComm.size):
        comm_tys, comm_hits = run_with_comm(
            _TwoRankReplayComm(rank, expected_tys, expected_hits)
        )
        assert len(comm_tys) == len(expected_global_tys) == 2
        assert len(comm_hits) == len(expected_global_hits) == 2
        _assert_list_arrays_close(comm_tys, expected_global_tys)
        _assert_list_arrays_close(comm_hits, expected_global_hits)


@pytest.fixture
def assert_two_rank_replay_matches():
    return _assert_two_rank_replay_matches


def _loaded_backend_module():
    module = sys.modules.get("simsopt_jax.backend")
    if module is not None and hasattr(module, "invalidate_backend_cache"):
        return module
    return None


def _loaded_jax_core_module():
    module = sys.modules.get("simsopt_jax.core")
    if module is not None and hasattr(module, "invalidate_kernel_cache"):
        return module
    return None


def _invalidate_loaded_kernel_cache() -> None:
    jax_core_module = _loaded_jax_core_module()
    if jax_core_module is not None:
        jax_core_module.invalidate_kernel_cache()


def _invalidate_loaded_backend_state() -> None:
    backend_module = _loaded_backend_module()
    if backend_module is not None:
        backend_module.invalidate_backend_cache()
    _invalidate_loaded_kernel_cache()


def _snapshot_loaded_jax_runtime_config() -> dict[str, object]:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return dict(_JAX_RUNTIME_CONFIG_DEFAULTS)
    return {
        name: getattr(jax_module.config, name) for name in _JAX_RUNTIME_CONFIG_DEFAULTS
    }


def _restore_loaded_jax_runtime_config(snapshot: dict[str, object]) -> None:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return
    for name, value in snapshot.items():
        jax_module.config.update(name, value)


def _restore_backend_runtime_env(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _split_xla_flag_tokens(xla_flags: str | None) -> tuple[str, ...]:
    if not xla_flags:
        return ()
    try:
        return tuple(shlex.split(xla_flags))
    except ValueError:
        return tuple(xla_flags.split())


def ensure_gpu_determinism_xla_flag(
    env: dict[str, str],
    *,
    deterministic_flag: str = _DEFAULT_GPU_DETERMINISM_XLA_FLAG,
) -> None:
    tokens = _split_xla_flag_tokens(env.get("XLA_FLAGS"))
    rewritten = [
        token
        for token in tokens
        if not any(
            token == flag_name or token.startswith(f"{flag_name}=")
            for flag_name in (
                _GPU_DETERMINISM_XLA_FLAGS + _STALE_GPU_DETERMINISM_XLA_FLAGS
            )
        )
    ]
    rewritten.append(deterministic_flag)
    env["XLA_FLAGS"] = " ".join(rewritten)


@pytest.fixture(autouse=True)
def _guard_backend_runtime_state():
    env_snapshot = {name: os.environ.get(name) for name in _BACKEND_RUNTIME_ENV_VARS}
    jax_config_snapshot = _snapshot_loaded_jax_runtime_config()
    _invalidate_loaded_backend_state()
    try:
        yield
    finally:
        _restore_backend_runtime_env(env_snapshot)
        _restore_loaded_jax_runtime_config(jax_config_snapshot)
        _invalidate_loaded_backend_state()


def _activate_backend_mode(monkeypatch, request, *, mode, strict):
    _require_jax()
    invalidate_backend_cache()
    previous = get_backend_config()
    if mode == "jax_gpu_parity":
        merged_env = dict(os.environ)
        ensure_gpu_determinism_xla_flag(merged_env)
        monkeypatch.setenv("XLA_FLAGS", merged_env["XLA_FLAGS"])
    requested_transfer_guard = os.environ.get("SIMSOPT_JAX_TRANSFER_GUARD")
    if mode == "native_cpu":
        transfer_guard = None
    elif requested_transfer_guard not in (None, ""):
        transfer_guard = requested_transfer_guard
    else:
        transfer_guard = "log"
    set_backend(
        mode,
        strict=strict,
        transfer_guard=transfer_guard,
        configure_runtime=False,
    )
    _invalidate_loaded_kernel_cache()
    _apply_test_transfer_guard(mode, transfer_guard)

    def _restore_backend_mode():
        invalidate_backend_cache()
        set_backend(
            previous.mode,
            strict=previous.strict,
            debug_nans=previous.debug_nans,
            transfer_guard=previous.transfer_guard,
            compilation_cache_dir=previous.compilation_cache_dir,
            configure_runtime=False,
        )
        _invalidate_loaded_kernel_cache()
        _apply_test_transfer_guard(previous.mode, previous.transfer_guard)

    request.addfinalizer(_restore_backend_mode)


def _apply_test_transfer_guard(mode, transfer_guard=None):
    if mode == "native_cpu":
        jax_module = sys.modules.get("jax")
        if jax_module is not None:
            jax_module.config.update("jax_transfer_guard", "allow")
        return
    jax_module = _require_jax()
    jax_module.config.update(
        "jax_transfer_guard",
        "log" if transfer_guard in (None, "") else transfer_guard,
    )


def enable_strict_jax_backend(monkeypatch, request, mode="jax_gpu_parity"):
    """Activate strict JAX backend mode for a single test.

    Sets the backend mode and strict env vars, invalidates the config cache,
    and registers a finalizer to restore the env and re-invalidate after
    test-local backend mode changes.
    """
    _activate_backend_mode(monkeypatch, request, mode=mode, strict=True)


def enable_non_strict_jax_backend(monkeypatch, request, mode):
    """Activate non-strict JAX backend mode for a single test.

    Sets the backend mode, removes the strict env var, and invalidates the
    config cache.  ``mode`` is required — callers must be explicit about which
    backend mode they intend (e.g. ``"jax_cpu_parity"`` or ``"jax_gpu_parity"``).
    """
    _activate_backend_mode(monkeypatch, request, mode=mode, strict=False)


def parity_mode(lane: str) -> str:
    try:
        return _PARITY_LANE_TO_MODE[lane]
    except KeyError as exc:
        raise ValueError(f"Unknown parity lane {lane!r}") from exc


def parity_seed(seed: int = 0) -> int:
    return _PARITY_SEED_BASE + seed


def parity_rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(parity_seed(seed))


def _parity_device_for_lane(jax_module, lane: str):
    if lane not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown parity lane {lane!r}; expected 'cpu' or 'gpu'.")
    for device in jax_module.devices():
        if device.platform == lane:
            return device
    if lane == "gpu":
        pytest.skip("CUDA GPU not available")
    if lane == "cpu":
        pytest.skip("CPU JAX backend not available")


def parity_device(lane: str):
    return _parity_device_for_lane(_require_jax(), lane)


@contextmanager
def parity_default_device(lane: str):
    jax_module = _require_jax()
    with jax_module.default_device(_parity_device_for_lane(jax_module, lane)):
        yield


def _block_until_ready(value, *, jax_module):
    return jax_module.tree.map(
        lambda leaf: (
            leaf.block_until_ready() if isinstance(leaf, jax_module.Array) else leaf
        ),
        value,
    )


def host_materialize(value):
    jax_module = _require_jax()
    return jax_module.device_get(_block_until_ready(value, jax_module=jax_module))


def host_array(value, *, dtype=None):
    return np.asarray(host_materialize(value), dtype=dtype)


def host_scalar(value) -> float:
    return float(np.asarray(host_materialize(value), dtype=np.float64))


def device_float64(value):
    jax_module = _require_jax()
    return jax_module.numpy.asarray(
        np.asarray(value, dtype=np.float64),
        dtype=jax_module.numpy.float64,
    )


@pytest.fixture(params=("cpu", "gpu"), ids=("cpu_parity", "gpu_parity"))
def parity_lane(request):
    return request.param


def enable_strict_parity_backend(monkeypatch, request, lane: str) -> None:
    enable_strict_jax_backend(monkeypatch, request, mode=parity_mode(lane))


def assert_array_on_device(array, device):
    jax_module = _require_jax()
    assert isinstance(array, jax_module.Array)
    assert array.devices() == {device}


def assert_arrays_on_device(device, *arrays):
    for array in arrays:
        assert_array_on_device(array, device)


def relative_error(actual, reference):
    """Return ``|actual - reference| / (|reference| + 1e-30)``."""
    return abs(actual - reference) / (abs(reference) + 1e-30)


def _parity_lane_key(lane_or_mode: str) -> str:
    if lane_or_mode in _PARITY_LANE_TO_MODE:
        return lane_or_mode
    try:
        return _PARITY_MODE_TO_LANE[lane_or_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown parity lane or mode {lane_or_mode!r}") from exc


def parity_acceptance_tolerance(tier: str, lane_or_mode: str) -> tuple[float, float]:
    try:
        tolerances = _REDUCTION_ACCEPTANCE_TIERS[tier]
    except KeyError as exc:
        raise ValueError(f"Unknown parity acceptance tier {tier!r}") from exc

    lane = _parity_lane_key(lane_or_mode)

    return tolerances[lane]


def parity_acceptance_modes(
    tier: str, *modes: str
) -> tuple[tuple[str, float, float], ...]:
    return tuple((mode, *parity_acceptance_tolerance(tier, mode)) for mode in modes)


_STRICT_GPU_EXCLUDED_MARKERS = {
    "adapter_boundary": "adapter-boundary test is not part of strict jax_gpu_parity",
    "native_cpu_reference": "native CPU/reference test is not part of strict jax_gpu_parity",
    "build_compat_native": "native build-compat test is not part of strict jax_gpu_parity",
}

_ADAPTER_BOUNDARY_TEST_FILES = frozenset(
    {
        "field/test_force_item09_closeout.py",
        "field/test_magnetic_axis_helpers_jax_item21.py",
        "field/test_selffieldforces.py",
        "geo/test_curvexyzfouriersymmetries_spec_jax.py",
        "geo/test_qfmsurface_jax.py",
        "geo/test_surface_fourier_jax.py",
        "geo/test_surface_garabedian_jax.py",
        "geo/test_surface_henneberg_jax.py",
        "geo/test_surface_objectives_jax.py",
        "geo/test_surface_rzfourier_jax.py",
        "geo/test_surface_rzfourier_jax_item06_closeout.py",
        "geo/test_surface_rzfourier_transfer_guard_jax.py",
        "geo/test_surface_xyz_tensor_clamped_jax.py",
        "objectives/test_fluxobjective_jax_parity.py",
    }
)

_NATIVE_CPU_REFERENCE_TEST_FILES = frozenset(
    {
        "integration/test_non_banana_example_cpp_jax_cpu_parity.py",
        "jax/solve/test_compat_shim_translation.py",
    }
)

_NATIVE_CPU_REFERENCE_NODE_PREFIXES = frozenset(
    {
        "geo/test_boozer_derivatives_jax.py::TestBoozerDirectCppOracles::",
        "geo/test_boozer_derivatives_jax.py::TestBoozerResidualJacobianComposed::test_jacobian_matches_cpp_dresidual_dc_oracle",
        "geo/test_boozer_derivatives_jax.py::TestBoozerHessianComposed::test_hessian_matches_cpp_boozer_residual_ds2_oracle",
        "geo/test_boozer_residual_jax.py::TestBoozerResidualScalar::test_scalar_matches_cpp_oracle",
        "geo/test_boozer_residual_jax.py::TestBoozerResidualParityStress::test_vector_reduction_near_tolerance_floor_matches_cpp_scalar",
        "geo/test_boozer_residual_jax.py::TestBoozerResidualParityStress::test_scalar_residual_norm_near_tolerance_floor_matches_cpp_oracle",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_scipy_bfgs_strips_limited_memory_and_callback_options",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_reference_scipy_adapters_materialize_host_contract",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_reference_scipy_adapter_rejects_non_scalar_objective",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_reference_lbfgs_trace_uses_host_core_without_scipy",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_scipy_minimize_does_not_cache_unmarked_objective",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_quadratic",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_refines_nontrivial_gmres_residual",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_dense_steps_use_materialized_solve",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_backtracks_finite_norm_increasing_operator_steps",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_backtracks_nonfinite_value_candidate",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_rejects_nonfinite_gradient_norm_candidate",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_nonfinite_operator_step_fails_without_dense_fallback",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_exact_linear_system",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_exact_materializes_dense_jacobian_once_at_final_iterate",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_exact_skips_dense_jacobian_when_ceiling_is_exceeded",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_backtracks_gradient_increasing_step",
        "geo/test_boozersurface_jax.py::TestNewtonPolishBoozer::",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_backend_mutation_does_not_rewrite_dense_linearization_default",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_instantiation_defaults_optimizer_backend_from_runtime_contract",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_reference_ls_reuses_cached_scipy_value_and_grad_transform",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_lbfgs_public_api_uses_options_default_when_limited_memory_omitted",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_host_scipy_quasi_newton_uses_cpu_ordered_value_grad_without_trace",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_ls_api_accepts_weight_inv_modB_override",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_manual_ls_api_supports_baseline_demo_sequence",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_restores_cpu_api",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_nonstellsym_stays_native_without_root",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_nonstellsym_uses_full_jacobian_solve",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_newton_api_routes_without_legacy_vectorize_kwarg",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_private_options_rejected_with_scipy_backend",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_newton_polish_returns_stabilized_hessian_when_requested",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_routes_backend_contract_to_expected_method",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_scipy_bfgs_pre_newton_contract_uses_cpu_call_shape",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_reference_lm_forwards_least_squares_options",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_jax_least_squares_solves_simple_structured_problem",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_jax_minimize_adam_solves_simple_structured_problem",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_jax_minimize_adam_supports_explicit_value_and_grad",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_reference_minimize_supports_structured_explicit_value_and_grad",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_jax_least_squares_pytree_hot_path_skips_flattening_adapter",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_same_input_same_result_without_res_identity",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_emits_newton_progress_updates",
        "geo/test_boozersurface_jax.py::TestMixedQuadratureBoozer::test_run_code_ls_converges",
        "geo/test_boozersurface_jax.py::TestMixedQuadratureBoozer::test_penalty_matches_uniform",
        "integration/test_single_stage_jax_cpu_reference.py::TestBoozerResidualCPUParity::",
        "integration/test_single_stage_jax_cpu_reference.py::TestAdjointSolveConsistency::",
    }
)

_BUILD_COMPAT_NATIVE_TEST_FILES = frozenset(
    {
        "geo/test_boozersurface.py",
    }
)


def skip_strict_gpu_collection(reason: str) -> None:
    if os.environ.get("SIMSOPT_BACKEND_MODE") == "jax_gpu_parity":
        pytest.skip(reason, allow_module_level=True)


def _item_nodeid_matches(item, prefixes: frozenset[str]) -> bool:
    nodeid = item.nodeid
    test_root_relative_nodeid = nodeid.removeprefix("tests/")
    return any(
        nodeid.startswith(prefix) or test_root_relative_nodeid.startswith(prefix)
        for prefix in prefixes
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark heavy JAX suites so they can be sharded consistently."""
    tests_root = Path(__file__).resolve().parent
    strict_gpu_mode = os.environ.get("SIMSOPT_BACKEND_MODE") == "jax_gpu_parity"
    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            relpath = path.relative_to(tests_root)
        except ValueError:
            continue

        relpath_str = relpath.as_posix()
        if relpath.parts and relpath.parts[0] == "jax":
            item.add_marker(pytest.mark.jax_gpu_pure)
        if relpath.parts and relpath.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.jax_gpu_integration)
            item.add_marker(pytest.mark.slow)
        if relpath_str in _ADAPTER_BOUNDARY_TEST_FILES:
            item.add_marker(pytest.mark.adapter_boundary)
        if (
            relpath_str in _NATIVE_CPU_REFERENCE_TEST_FILES
            or _item_nodeid_matches(item, _NATIVE_CPU_REFERENCE_NODE_PREFIXES)
        ):
            item.add_marker(pytest.mark.native_cpu_reference)
        if relpath_str in _BUILD_COMPAT_NATIVE_TEST_FILES:
            item.add_marker(pytest.mark.build_compat_native)
        if relpath_str in {
            "integration/test_single_stage_jax.py",
            "integration/test_single_stage_jax_cpu_reference.py",
            "integration/test_single_stage_dof_mapping.py",
            "integration/test_single_stage_physics_parity.py",
            "geo/test_single_stage_example.py",
        }:
            item.add_marker(pytest.mark.single_stage)
            item.add_marker(pytest.mark.slow)
        if relpath_str in {
            "integration/test_stage2_jax.py",
            "integration/test_stage2_target_lane_purity.py",
        }:
            item.add_marker(pytest.mark.stage2)
            item.add_marker(pytest.mark.slow)
        if relpath_str in {
            "geo/test_boozersurface_jax.py",
            "geo/test_boozersurface_jax_private.py",
        }:
            item.add_marker(pytest.mark.boozer)
            item.add_marker(pytest.mark.slow)
        if strict_gpu_mode:
            for marker_name, reason in _STRICT_GPU_EXCLUDED_MARKERS.items():
                if any(item.iter_markers(marker_name)):
                    item.add_marker(pytest.mark.skip(reason=reason))
                    break
