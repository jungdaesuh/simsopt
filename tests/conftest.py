"""Root-level test fixtures shared across the JAX test suite."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import jax
except ModuleNotFoundError:
    jax = None
else:
    jax.config.update("jax_enable_x64", True)

_BACKEND_RUNTIME_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
    "SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE",
    "SIMSOPT_JAX_CHUNK_AUTOTUNE",
    "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB",
    "SIMSOPT_JAX_SHARDING",
    "SIMSOPT_JAX_SHARDING_AXIS",
    "SIMSOPT_JAX_MIN_POINTS_TO_SHARD",
    "SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD",
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


def _require_jax():
    if jax is None:
        pytest.skip("JAX not installed in current environment")
    return jax


def _loaded_backend_module():
    module = sys.modules.get("simsopt.backend")
    if module is not None and hasattr(module, "invalidate_backend_cache"):
        return module
    return None


def _loaded_jax_core_module():
    module = sys.modules.get("simsopt.jax_core")
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
        name: getattr(jax_module.config, name)
        for name in _JAX_RUNTIME_CONFIG_DEFAULTS
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


@pytest.fixture(autouse=True)
def _guard_backend_runtime_state():
    env_snapshot = {
        name: os.environ.get(name) for name in _BACKEND_RUNTIME_ENV_VARS
    }
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
    from simsopt.backend import get_backend_config, invalidate_backend_cache, set_backend

    invalidate_backend_cache()
    previous = get_backend_config()
    requested_transfer_guard = os.environ.get("SIMSOPT_JAX_TRANSFER_GUARD")
    if mode == "native_cpu":
        transfer_guard = None
    elif requested_transfer_guard not in (None, ""):
        transfer_guard = requested_transfer_guard
    else:
        transfer_guard = "log"
    backend_env_ctx = monkeypatch.context()
    backend_env = backend_env_ctx.__enter__()
    set_backend(
        mode,
        strict=strict,
        transfer_guard=transfer_guard,
        configure_runtime=False,
    )
    _invalidate_loaded_kernel_cache()
    _apply_test_transfer_guard(mode, transfer_guard)

    def _restore_backend_mode():
        backend_env_ctx.__exit__(None, None, None)
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
    for device in jax_module.devices():
        if device.platform == lane:
            return device
    if lane == "gpu":
        pytest.skip("CUDA GPU not available")
    raise RuntimeError(f"No JAX device available for parity lane {lane!r}")


def parity_device(lane: str):
    return _parity_device_for_lane(_require_jax(), lane)


@contextmanager
def parity_default_device(lane: str):
    jax_module = _require_jax()
    with jax_module.default_device(_parity_device_for_lane(jax_module, lane)):
        yield


def _block_until_ready(value, *, jax_module):
    return jax_module.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready()
        if isinstance(leaf, jax_module.Array)
        else leaf,
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


def parity_acceptance_tolerance(
    tier: str, lane_or_mode: str
) -> tuple[float, float]:
    try:
        tolerances = _REDUCTION_ACCEPTANCE_TIERS[tier]
    except KeyError as exc:
        raise ValueError(f"Unknown parity acceptance tier {tier!r}") from exc

    lane = _parity_lane_key(lane_or_mode)

    return tolerances[lane]


def parity_acceptance_modes(
    tier: str, *modes: str
) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (mode, *parity_acceptance_tolerance(tier, mode)) for mode in modes
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark heavy JAX suites so they can be sharded consistently."""
    tests_root = Path(__file__).resolve().parent
    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            relpath = path.relative_to(tests_root)
        except ValueError:
            continue

        relpath_str = relpath.as_posix()
        if relpath.parts and relpath.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        if relpath_str in {
            "integration/test_single_stage_jax.py",
            "integration/test_single_stage_jax_cpu_reference.py",
            "geo/test_single_stage_example.py",
        }:
            item.add_marker(pytest.mark.single_stage)
            item.add_marker(pytest.mark.slow)
        if relpath_str == "integration/test_stage2_jax.py":
            item.add_marker(pytest.mark.stage2)
            item.add_marker(pytest.mark.slow)
        if relpath_str in {
            "geo/test_boozersurface_jax.py",
            "geo/test_boozersurface_jax_private.py",
        }:
            item.add_marker(pytest.mark.boozer)
            item.add_marker(pytest.mark.slow)
