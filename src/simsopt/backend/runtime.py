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
_DEBUG_NANS_ENV = "SIMSOPT_JAX_DEBUG_NANS"
_TRANSFER_GUARD_ENV = "SIMSOPT_JAX_TRANSFER_GUARD"
_COMPILATION_CACHE_DIR_ENV = "SIMSOPT_JAX_COMPILATION_CACHE_DIR"
_COIL_CHUNK_SIZE_ENV = "SIMSOPT_JAX_COIL_CHUNK_SIZE"
_QUADRATURE_BLOCK_SIZE_ENV = "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE"
_JAX_PLATFORMS_ENV = "JAX_PLATFORMS"
_VALID_TRANSFER_GUARDS = ("allow", "log", "disallow")
_GUARDRAIL_ENV_VARS = (
    _DEBUG_NANS_ENV,
    _TRANSFER_GUARD_ENV,
    _COMPILATION_CACHE_DIR_ENV,
)
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
    (_DEBUG_NANS_ENV, "debug_nans"),
    (_TRANSFER_GUARD_ENV, "transfer_guard"),
    (_COMPILATION_CACHE_DIR_ENV, "compilation_cache_dir"),
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

_FIELD_KERNEL_DEFAULTS = {
    "native_cpu": {"coil_chunk_size": 0, "quadrature_block_size": 0},
    "jax_cpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0},
    "jax_gpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0},
    "jax_gpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64},
}
_POINT_CHUNK_SIZE_BY_POLICY = {
    "host_reference": 0,
    "stable_default": 256,
    "performance_tuned": 1024,
}
_FIELD_KERNEL_ENV_BY_KEY = {
    "coil_chunk_size": _COIL_CHUNK_SIZE_ENV,
    "quadrature_block_size": _QUADRATURE_BLOCK_SIZE_ENV,
}
_DEFAULT_TRANSFER_GUARD_BY_MODE = {
    "native_cpu": None,
    "jax_cpu_parity": "log",
    "jax_gpu_parity": "log",
    "jax_gpu_fast": None,
}


@dataclass(frozen=True)
class BackendConfig:
    mode: str
    backend: str
    jax_platform: str
    strict: bool = False
    debug_nans: bool = False
    transfer_guard: str | None = None
    compilation_cache_dir: str | None = None


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
    debug_nans: bool
    transfer_guard: str | None
    compilation_cache_dir: str | None


@dataclass(frozen=True)
class FieldKernelTuning:
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int


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


def _validate_transfer_guard(value: str | None, *, source: str) -> str | None:
    if value in (None, ""):
        return None
    if value not in _VALID_TRANSFER_GUARDS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_TRANSFER_GUARDS}"
        )
    return value


def _default_compilation_cache_dir(mode: str) -> str | None:
    del mode
    return None


def _optional_env_value(name: str) -> str | None:
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return None
    return raw_value


def _optional_positive_int_env(name: str) -> int | None:
    raw_value = _optional_env_value(name)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name}={raw_value!r} must be >= 0")
    return value


def _field_kernel_value(mode: str, key: str) -> int:
    env_name = _FIELD_KERNEL_ENV_BY_KEY[key]
    value = _optional_positive_int_env(env_name)
    if value is not None:
        return value
    return _FIELD_KERNEL_DEFAULTS[mode][key]


def _resolve_debug_nans(debug_nans: bool | None) -> bool:
    if debug_nans is None:
        return (
            _env_bool(_DEBUG_NANS_ENV)
            if _optional_env_value(_DEBUG_NANS_ENV)
            else False
        )
    return bool(debug_nans)


def _default_transfer_guard(mode: str) -> str | None:
    return _DEFAULT_TRANSFER_GUARD_BY_MODE[_validate_mode(mode)]


def _resolve_transfer_guard(mode: str, transfer_guard: str | None) -> str | None:
    env_value = _optional_env_value(_TRANSFER_GUARD_ENV)
    if transfer_guard is None and env_value is not None:
        return _validate_transfer_guard(
            env_value,
            source=_TRANSFER_GUARD_ENV,
        )
    if transfer_guard is None:
        return _default_transfer_guard(mode)
    return _validate_transfer_guard(
        transfer_guard,
        source="transfer_guard",
    )


def _resolve_compilation_cache_dir(
    mode: str,
    compilation_cache_dir: str | None,
) -> str | None:
    if compilation_cache_dir is not None:
        return compilation_cache_dir or None
    env_value = _optional_env_value(_COMPILATION_CACHE_DIR_ENV)
    if env_value is not None:
        return env_value
    return _default_compilation_cache_dir(mode)


def _config_from_mode(
    mode: str,
    *,
    strict: bool,
    debug_nans: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
) -> BackendConfig:
    mode = _validate_mode(mode)
    backend, jax_platform = _MODE_TO_RUNTIME[mode]
    return BackendConfig(
        mode=mode,
        backend=backend,
        jax_platform=jax_platform,
        strict=bool(strict),
        debug_nans=_resolve_debug_nans(debug_nans),
        transfer_guard=_resolve_transfer_guard(mode, transfer_guard),
        compilation_cache_dir=_resolve_compilation_cache_dir(
            mode,
            compilation_cache_dir,
        ),
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
        debug_nans=config.debug_nans,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
    )


def _runtime_env_value(attribute_name: str, value: object) -> str:
    if attribute_name in {"strict", "debug_nans"}:
        return "1" if bool(value) else "0"
    if value is None:
        return ""
    return str(value)


_cached_backend_policy: BackendPolicy | None = None


def get_backend_policy(mode: str | None = None) -> BackendPolicy:
    """Return the numerical-policy contract for a backend mode."""
    global _cached_backend_policy
    if mode is None:
        if _cached_backend_policy is not None:
            return _cached_backend_policy
        policy = _policy_from_config(get_backend_config())
        _cached_backend_policy = policy
        return policy
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


_cached_backend_config: BackendConfig | None = None


def get_backend_config() -> BackendConfig:
    """Return the resolved backend configuration.

    The result is cached after first resolution. Call
    ``invalidate_backend_cache()`` or ``set_backend()`` to clear.
    """
    global _cached_backend_config
    if _cached_backend_config is not None:
        return _cached_backend_config

    strict = _env_bool(_STRICT_ENV)
    mode = os.environ.get(_MODE_ENV)
    if mode is not None:
        config = _config_from_mode(mode, strict=strict)
    else:
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
        config = _config_from_mode(
            _mode_from_legacy_env(backend, platform),
            strict=strict,
        )

    _cached_backend_config = config
    return config


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


_cached_field_kernel_tuning: FieldKernelTuning | None = None


def get_field_kernel_tuning(mode: str | None = None) -> FieldKernelTuning:
    """Return the low-level field-kernel tuning contract for the resolved mode."""
    global _cached_field_kernel_tuning
    if mode is None and _cached_field_kernel_tuning is not None:
        return _cached_field_kernel_tuning
    resolved_mode = _resolve_mode(mode)
    policy = get_backend_policy(resolved_mode)
    tuning = FieldKernelTuning(
        mode=resolved_mode,
        chunk_policy=policy.chunk_policy,
        coil_chunk_size=_field_kernel_value(resolved_mode, "coil_chunk_size"),
        quadrature_block_size=_field_kernel_value(
            resolved_mode,
            "quadrature_block_size",
        ),
    )
    if mode is None:
        _cached_field_kernel_tuning = tuning
    return tuning


def get_coil_chunk_size(mode: str | None = None) -> int:
    """Return the low-level Biot-Savart coil-axis chunk size."""
    return get_field_kernel_tuning(mode).coil_chunk_size


def get_quadrature_block_size(mode: str | None = None) -> int:
    """Return the low-level Biot-Savart quadrature-block size."""
    return get_field_kernel_tuning(mode).quadrature_block_size


def get_point_chunk_size(mode: str | None = None) -> int:
    """Return the grouped-field point chunk size for the resolved mode."""
    chunk_policy = get_chunk_policy(mode)
    return _POINT_CHUNK_SIZE_BY_POLICY.get(chunk_policy, 0)


def get_debug_nans(mode: str | None = None) -> bool:
    """Return the debug-NaN runtime guardrail state for the resolved mode."""
    return get_backend_policy(mode).debug_nans


def get_transfer_guard(mode: str | None = None) -> str | None:
    """Return the active JAX transfer-guard policy for the resolved mode."""
    return get_backend_policy(mode).transfer_guard


def get_compilation_cache_dir(mode: str | None = None) -> str | None:
    """Return the active JAX compilation-cache directory for the resolved mode."""
    return get_backend_policy(mode).compilation_cache_dir


def invalidate_backend_cache() -> None:
    """Clear the cached backend configuration and derived caches.

    Call this after mutating ``SIMSOPT_*`` environment variables directly
    (outside of ``set_backend()``) so the next ``get_backend_config()`` call
    re-reads the environment.  Test fixtures should call this when they
    manipulate env vars via ``monkeypatch`` or context managers.
    """
    global _cached_backend_config, _cached_backend_policy, _cached_field_kernel_tuning
    _cached_backend_config = None
    _cached_backend_policy = None
    _cached_field_kernel_tuning = None


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
    jax.config.update("jax_debug_nans", config.debug_nans)
    jax.config.update("jax_transfer_guard", config.transfer_guard)
    if config.compilation_cache_dir is not None:
        jax.config.update("jax_compilation_cache_dir", config.compilation_cache_dir)


def set_backend(
    mode: str,
    *,
    strict: bool = False,
    debug_nans: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
    configure_runtime: bool = True,
) -> BackendConfig:
    """Set the active backend mode for the current process.

    This keeps the legacy env vars in sync so existing scripts and subprocess
    helpers continue to work unchanged.  Also updates the config cache so
    subsequent ``get_backend_config()`` calls are free.
    """
    global _cached_backend_config, _cached_backend_policy, _cached_field_kernel_tuning
    config = _config_from_mode(
        mode,
        strict=bool(strict),
        debug_nans=debug_nans,
        transfer_guard=transfer_guard,
        compilation_cache_dir=compilation_cache_dir,
    )
    _cached_backend_config = config
    _cached_backend_policy = None
    _cached_field_kernel_tuning = None
    for env_name, attribute_name in _SYNCED_RUNTIME_ENV_VALUES:
        os.environ[env_name] = _runtime_env_value(
            attribute_name, getattr(config, attribute_name)
        )
    if configure_runtime:
        apply_jax_runtime_config()
    return config
