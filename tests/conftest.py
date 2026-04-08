"""Root-level test fixtures shared across the JAX test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

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
    "jax_enable_x64": False,
    "jax_debug_nans": False,
    "jax_transfer_guard": None,
    "jax_platforms": None,
    "jax_compilation_cache_dir": None,
}


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
    import jax

    jax.config.update(
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


def relative_error(actual, reference):
    """Return ``|actual - reference| / (|reference| + 1e-30)``."""
    return abs(actual - reference) / (abs(reference) + 1e-30)


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
