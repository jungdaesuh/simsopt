"""Backend/device helpers for integration tests with a local conftest."""

from __future__ import annotations

import os
import shlex
import sys

import pytest

from simsopt_jax.backend import get_backend_config, invalidate_backend_cache, set_backend

try:
    import jax
except ModuleNotFoundError:
    jax = None

_PARITY_LANE_TO_MODE = {
    "cpu": "jax_cpu_parity",
    "gpu": "jax_gpu_parity",
}
_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_exclude_nondeterministic_ops",)
_STALE_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_deterministic_ops",)
_DEFAULT_GPU_DETERMINISM_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"


def _require_jax():
    if jax is None:
        pytest.skip("JAX not installed in current environment")
    return jax


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


def enable_strict_jax_backend(monkeypatch, request, mode="jax_gpu_parity"):
    _activate_backend_mode(monkeypatch, request, mode=mode, strict=True)


def enable_non_strict_jax_backend(monkeypatch, request, mode):
    _activate_backend_mode(monkeypatch, request, mode=mode, strict=False)


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
