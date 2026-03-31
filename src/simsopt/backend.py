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

_MODE_POLICY_DEFAULTS = {
    "native_cpu": {
        "parity_mode": False,
        "requires_x64": True,
        "chunk_policy": "host_reference",
        "tolerance_tier": "cpu_reference",
        "compilation_cache_policy": "not_applicable",
        "provenance_label": "native_cpu",
    },
    "jax_cpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "provenance_label": "jax_cpu_parity",
    },
    "jax_gpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "provenance_label": "jax_gpu_parity",
    },
    "jax_gpu_fast": {
        "parity_mode": False,
        "requires_x64": True,
        "chunk_policy": "performance_tuned",
        "tolerance_tier": "fast",
        "compilation_cache_policy": "optional_persistent",
        "provenance_label": "jax_gpu_fast",
    },
}


@dataclass(frozen=True)
class BackendConfig:
    mode: str
    backend: str
    jax_platform: str
    strict: bool = False


@dataclass(frozen=True)
class BackendPolicy:
    mode: str
    backend: str
    jax_platform: str
    strict: bool
    parity_mode: bool
    requires_x64: bool
    chunk_policy: str
    tolerance_tier: str
    compilation_cache_policy: str
    provenance_label: str


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


def _resolve_mode(mode: str | None = None) -> str:
    if mode is None:
        return get_backend_mode()
    return _validate_mode(mode)


def _get_mode_policy_defaults(mode: str) -> dict[str, object]:
    return _MODE_POLICY_DEFAULTS[_validate_mode(mode)]


def _policy_from_config(config: BackendConfig) -> BackendPolicy:
    defaults = _get_mode_policy_defaults(config.mode)
    return BackendPolicy(
        mode=config.mode,
        backend=config.backend,
        jax_platform=config.jax_platform,
        strict=config.strict,
        parity_mode=bool(defaults["parity_mode"]),
        requires_x64=bool(defaults["requires_x64"]),
        chunk_policy=str(defaults["chunk_policy"]),
        tolerance_tier=str(defaults["tolerance_tier"]),
        compilation_cache_policy=str(defaults["compilation_cache_policy"]),
        provenance_label=str(defaults["provenance_label"]),
    )


def _runtime_env_value(attribute_name: str, value: object) -> str:
    if attribute_name == "strict":
        return "1" if bool(value) else "0"
    return str(value)


def get_backend_policy(mode: str | None = None) -> BackendPolicy:
    """Return the numerical-policy contract for a backend mode."""
    if mode is None:
        config = get_backend_config()
    else:
        resolved_mode = _resolve_mode(mode)
        current_config = get_backend_config()
        config = (
            current_config
            if current_config.mode == resolved_mode
            else _config_from_mode(resolved_mode, strict=False)
        )
    return _policy_from_config(config)


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


def is_parity_mode(mode: str | None = None) -> bool:
    """``True`` when the resolved mode is a parity lane."""
    return get_backend_policy(mode).parity_mode


def requires_x64(mode: str | None = None) -> bool:
    """``True`` when the resolved mode requires float64 JAX execution."""
    return get_backend_policy(mode).requires_x64


def get_chunk_policy(mode: str | None = None) -> str:
    """Return the default chunking policy label for the resolved mode."""
    return get_backend_policy(mode).chunk_policy


def get_tolerance_tier(mode: str | None = None) -> str:
    """Return the tolerance policy label for the resolved mode."""
    return get_backend_policy(mode).tolerance_tier


def get_compilation_cache_policy(mode: str | None = None) -> str:
    """Return the compilation-cache policy label for the resolved mode."""
    return get_backend_policy(mode).compilation_cache_policy


def get_provenance_label(mode: str | None = None) -> str:
    """Return the provenance label that should tag outputs from the mode."""
    return get_backend_policy(mode).provenance_label


def raise_if_strict_jax_fallback(*, component: str, detail: str) -> None:
    """Reject CPU or mixed fallback behavior when strict JAX mode is active."""
    config = get_backend_config()
    if config.backend != "jax" or not config.strict:
        return
    raise RuntimeError(
        f"{component} cannot use {detail} while simsopt backend mode "
        f"{config.mode!r} has strict=True. Select a JAX-native path or "
        "disable strict mode."
    )


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
    jax.config.update("jax_enable_x64", requires_x64(config.mode))


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
        os.environ[env_name] = _runtime_env_value(
            attribute_name, getattr(config, attribute_name)
        )
    if configure_runtime:
        apply_jax_runtime_config()
    return config
