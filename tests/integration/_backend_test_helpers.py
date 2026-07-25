"""Backend/device helpers for integration tests with a local conftest."""

from __future__ import annotations

from contextlib import contextmanager
import os
import shlex
import sys

import numpy as np
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
_PARITY_MODE_TO_LANE = {
    "jax_cpu_parity": "cpu",
    "jax_gpu_parity": "gpu",
    "jax_gpu_fast": "gpu",
}
_PARITY_SEED_BASE = 1729
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
    lane = _PARITY_MODE_TO_LANE.get(mode)
    if lane is not None and not _parity_lane_available(lane):
        pytest.skip(
            "CUDA GPU not available"
            if lane == "gpu"
            else "CPU JAX backend not available"
        )
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
            precision=previous.precision,
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


def _parity_lane_available(lane: str) -> bool:
    jax_module = _require_jax()
    if lane not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown parity lane {lane!r}; expected 'cpu' or 'gpu'.")
    return any(device.platform == lane for device in jax_module.devices())


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
