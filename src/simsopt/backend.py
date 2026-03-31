"""
Backend selection for the simsopt JAX lane.

This module keeps the legacy environment-variable contract working while adding
an explicit mode-based public API for the new runtime surface:

- ``native_cpu``
- ``jax_cpu_parity``
- ``jax_gpu_parity``
- ``jax_gpu_fast``

The mode API is the SSOT. The older ``SIMSOPT_BACKEND`` /
``SIMSOPT_JAX_PLATFORM`` pair is still read and written for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

_VALID_BACKENDS = ("cpu", "jax")
_VALID_PLATFORMS = ("cpu", "cuda")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

_BACKEND_ENV = "SIMSOPT_BACKEND"
_BACKEND_LEGACY_ENV = "STAGE2_BACKEND"
_PLATFORM_ENV = "SIMSOPT_JAX_PLATFORM"
_PLATFORM_LEGACY_ENV = "SIMSOPT_JAX_BACKEND"
_MODE_ENV = "SIMSOPT_BACKEND_MODE"
_STRICT_ENV = "SIMSOPT_BACKEND_STRICT"
_JAX_PLATFORMS_ENV = "JAX_PLATFORMS"
_EXPLICIT_SELECTOR_ENV_VARS = (
    _MODE_ENV,
    _BACKEND_ENV,
    _BACKEND_LEGACY_ENV,
    _PLATFORM_ENV,
    _PLATFORM_LEGACY_ENV,
)
_SYNCED_RUNTIME_ENV_VALUES = (
    (_MODE_ENV, "mode"),
    (_STRICT_ENV, "strict"),
    (_BACKEND_ENV, "backend"),
    (_BACKEND_LEGACY_ENV, "backend"),
    (_PLATFORM_ENV, "jax_platform"),
    (_PLATFORM_LEGACY_ENV, "jax_platform"),
    (_JAX_PLATFORMS_ENV, "jax_platform"),
)

VALID_BACKEND_MODES = (
    "native_cpu",
    "jax_cpu_parity",
    "jax_gpu_parity",
    "jax_gpu_fast",
)

_MODE_TO_RUNTIME = {
    "native_cpu": ("cpu", "cpu"),
    "jax_cpu_parity": ("jax", "cpu"),
    "jax_gpu_parity": ("jax", "cuda"),
    "jax_gpu_fast": ("jax", "cuda"),
}


@dataclass(frozen=True)
class BackendConfig:
    mode: str
    backend: str
    jax_platform: str
    strict: bool = False


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in _TRUTHY_ENV_VALUES


def _validate_backend(value: str, *, source: str) -> str:
    if value not in _VALID_BACKENDS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_BACKENDS}"
        )
    return value


def _validate_platform(value: str, *, source: str) -> str:
    if value not in _VALID_PLATFORMS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_PLATFORMS}"
        )
    return value


def _validate_mode(mode: str) -> str:
    if mode not in VALID_BACKEND_MODES:
        raise ValueError(
            f"Backend mode {mode!r} is not valid. Accepted: {VALID_BACKEND_MODES}"
        )
    return mode


def _config_from_mode(mode: str, *, strict: bool) -> BackendConfig:
    backend, jax_platform = _MODE_TO_RUNTIME[_validate_mode(mode)]
    return BackendConfig(
        mode=mode,
        backend=backend,
        jax_platform=jax_platform,
        strict=bool(strict),
    )


def _resolve_legacy_value(
    primary_env: str,
    legacy_env: str,
    default: str,
    *,
    validator,
) -> str:
    raw_value = os.environ.get(primary_env)
    source = primary_env
    if raw_value is None:
        raw_value = os.environ.get(legacy_env, default)
        source = legacy_env if legacy_env in os.environ else "(default)"
    return validator(raw_value, source=source)


def _mode_from_legacy_env(backend: str, platform: str) -> str:
    if backend == "cpu":
        return "native_cpu"
    if platform == "cpu":
        return "jax_cpu_parity"
    return "jax_gpu_parity"


def get_backend_config() -> BackendConfig:
    """Return the resolved backend configuration."""
    strict = _env_bool(_STRICT_ENV)
    mode = os.environ.get(_MODE_ENV)
    if mode is not None:
        return _config_from_mode(mode, strict=strict)

    backend = _resolve_legacy_value(
        _BACKEND_ENV,
        _BACKEND_LEGACY_ENV,
        "cpu",
        validator=_validate_backend,
    )
    platform = _resolve_legacy_value(
        _PLATFORM_ENV,
        _PLATFORM_LEGACY_ENV,
        "cpu",
        validator=_validate_platform,
    )

    return _config_from_mode(
        _mode_from_legacy_env(backend, platform),
        strict=strict,
    )


def get_backend_mode() -> str:
    """Return the resolved backend mode."""
    return get_backend_config().mode


def get_backend() -> str:
    """Return the active compute backend: ``'cpu'`` or ``'jax'``."""
    return get_backend_config().backend


def is_jax_backend() -> bool:
    """``True`` when the JAX code path is selected."""
    return get_backend() == "jax"


def get_jax_platform() -> str:
    """Return the resolved JAX device platform: ``'cpu'`` or ``'cuda'``."""
    return get_backend_config().jax_platform


def is_backend_strict() -> bool:
    """``True`` when strict fallback rejection is enabled."""
    return get_backend_config().strict


def should_eagerly_configure_jax() -> bool:
    """Return whether package import should eagerly configure the JAX runtime."""
    explicit_selector_present = any(
        name in os.environ for name in _EXPLICIT_SELECTOR_ENV_VARS
    )
    return explicit_selector_present and is_jax_backend()


def apply_jax_runtime_config() -> None:
    """Apply the resolved JAX runtime settings to the active process."""
    config = get_backend_config()
    if config.backend != "jax":
        return
    import jax

    jax.config.update("jax_platforms", config.jax_platform)
    jax.config.update("jax_enable_x64", True)


def set_backend(
    mode: str,
    *,
    strict: bool = False,
    configure_runtime: bool = True,
) -> BackendConfig:
    """Set the active backend mode for the current process.

    This keeps the legacy env vars in sync so existing scripts and subprocess
    helpers continue to work unchanged.
    """
    config = _config_from_mode(mode, strict=bool(strict))
    for env_name, attribute_name in _SYNCED_RUNTIME_ENV_VALUES:
        value = getattr(config, attribute_name)
        if attribute_name == "strict":
            os.environ[env_name] = "1" if value else "0"
        else:
            os.environ[env_name] = value
    if configure_runtime:
        apply_jax_runtime_config()
    return config
