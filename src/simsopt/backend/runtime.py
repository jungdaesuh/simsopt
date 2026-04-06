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
import subprocess
from typing import Callable
import warnings

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
_PAIRWISE_PENALTY_CHUNK_SIZE_ENV = "SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE"
_CHUNK_AUTOTUNE_ENV = "SIMSOPT_JAX_CHUNK_AUTOTUNE"
_GPU_MEMORY_TOTAL_MB_ENV = "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB"
_SHARDING_STRATEGY_ENV = "SIMSOPT_JAX_SHARDING"
_SHARDING_AXIS_ENV = "SIMSOPT_JAX_SHARDING_AXIS"
_MIN_POINTS_TO_SHARD_ENV = "SIMSOPT_JAX_MIN_POINTS_TO_SHARD"
_JAX_PLATFORMS_ENV = "JAX_PLATFORMS"
_VALID_TRANSFER_GUARDS = ("allow", "log", "disallow")
_VALID_SHARDING_STRATEGIES = ("none", "points")
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

_NO_CI_REPRODUCIBILITY_DEFAULTS = {
    "gpu_reduction_order_max_ulp": None,
    "gpu_reduction_order_rel_tol": None,
    "gpu_reproducibility_seed": None,
    "gpu_reproducibility_sample_size": None,
    "tolerance_ratchet_factor": None,
}

_MODE_POLICY_DEFAULTS = {
    "native_cpu": {
        "parity_mode": False,
        "requires_x64": True,
        "chunk_policy": "host_reference",
        "tolerance_tier": "cpu_reference",
        "compilation_cache_policy": "not_applicable",
        "provenance_label": "native_cpu",
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
    "jax_cpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "provenance_label": "jax_cpu_parity",
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
    "jax_gpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "provenance_label": "jax_gpu_parity",
        "gpu_reduction_order_max_ulp": 10,
        "gpu_reduction_order_rel_tol": 1e-12,
        "gpu_reproducibility_seed": 1729,
        "gpu_reproducibility_sample_size": 1000,
        "tolerance_ratchet_factor": 10.0,
    },
    "jax_gpu_fast": {
        "parity_mode": False,
        "requires_x64": True,
        "chunk_policy": "performance_tuned",
        "tolerance_tier": "fast",
        "compilation_cache_policy": "optional_persistent",
        "provenance_label": "jax_gpu_fast",
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
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
_PAIRWISE_PENALTY_CHUNK_SIZE_BY_POLICY = dict(_POINT_CHUNK_SIZE_BY_POLICY)
_MODE_SHARDING_DEFAULTS = {
    "native_cpu": "none",
    "jax_cpu_parity": "none",
    "jax_gpu_parity": "none",
    "jax_gpu_fast": "points",
}
_DEFAULT_SHARDING_AXIS_NAME = "d"
_MIN_POINTS_TO_SHARD_BY_POLICY = {
    "host_reference": 1 << 30,
    "stable_default": 4096,
    "performance_tuned": 2048,
}
_AUTOTUNED_CHUNK_SIZES_BY_POLICY = {
    "host_reference": (),
    "stable_default": (
        (
            8192,
            {
                "coil_chunk_size": 8,
                "quadrature_block_size": 0,
                "point_chunk_size": 128,
                "pairwise_penalty_chunk_size": 128,
            },
        ),
        (
            16384,
            {
                "coil_chunk_size": 16,
                "quadrature_block_size": 0,
                "point_chunk_size": 256,
                "pairwise_penalty_chunk_size": 256,
            },
        ),
        (
            32768,
            {
                "coil_chunk_size": 32,
                "quadrature_block_size": 0,
                "point_chunk_size": 512,
                "pairwise_penalty_chunk_size": 512,
            },
        ),
        (
            None,
            {
                "coil_chunk_size": 64,
                "quadrature_block_size": 0,
                "point_chunk_size": 1024,
                "pairwise_penalty_chunk_size": 1024,
            },
        ),
    ),
    "performance_tuned": (
        (
            8192,
            {
                "coil_chunk_size": 32,
                "quadrature_block_size": 32,
                "point_chunk_size": 512,
                "pairwise_penalty_chunk_size": 512,
            },
        ),
        (
            16384,
            {
                "coil_chunk_size": 64,
                "quadrature_block_size": 64,
                "point_chunk_size": 1024,
                "pairwise_penalty_chunk_size": 1024,
            },
        ),
        (
            32768,
            {
                "coil_chunk_size": 128,
                "quadrature_block_size": 128,
                "point_chunk_size": 2048,
                "pairwise_penalty_chunk_size": 2048,
            },
        ),
        (
            None,
            {
                "coil_chunk_size": 256,
                "quadrature_block_size": 256,
                "point_chunk_size": 4096,
                "pairwise_penalty_chunk_size": 4096,
            },
        ),
    ),
}
_FIELD_KERNEL_ENV_BY_KEY = {
    "coil_chunk_size": _COIL_CHUNK_SIZE_ENV,
    "quadrature_block_size": _QUADRATURE_BLOCK_SIZE_ENV,
}
_DEFAULT_TRANSFER_GUARD_BY_MODE = {
    "native_cpu": None,
    "jax_cpu_parity": "log",
    "jax_gpu_parity": "log",
    "jax_gpu_fast": "log",
}
_BackendCacheClearCallbackKey = tuple[str, str]
_backend_cache_clear_callbacks: dict[
    _BackendCacheClearCallbackKey, Callable[[], None]
] = {}


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
    gpu_reduction_order_max_ulp: int | None
    gpu_reduction_order_rel_tol: float | None
    gpu_reproducibility_seed: int | None
    gpu_reproducibility_sample_size: int | None
    tolerance_ratchet_factor: float | None
    debug_nans: bool
    transfer_guard: str | None
    compilation_cache_dir: str | None


@dataclass(frozen=True)
class FieldKernelTuning:
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int


@dataclass(frozen=True)
class ChunkTuning:
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int
    point_chunk_size: int
    pairwise_penalty_chunk_size: int
    autotuned: bool
    autotune_source: str | None
    gpu_total_memory_mb: int | None


@dataclass(frozen=True)
class ShardingTuning:
    mode: str
    strategy: str
    mesh_axis_name: str
    min_points_to_shard: int
    device_count: int
    active: bool
    platform: str


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


def _validate_sharding_strategy(value: str, *, source: str) -> str:
    if value not in _VALID_SHARDING_STRATEGIES:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_SHARDING_STRATEGIES}"
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


def _optional_nonempty_env(name: str) -> str | None:
    raw_value = _optional_env_value(name)
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if stripped == "":
        return None
    return stripped


def _point_chunk_size_default(chunk_policy: str) -> int:
    return _POINT_CHUNK_SIZE_BY_POLICY.get(chunk_policy, 0)


def _pairwise_penalty_chunk_size_default(chunk_policy: str) -> int:
    return _PAIRWISE_PENALTY_CHUNK_SIZE_BY_POLICY.get(chunk_policy, 0)


def _resolve_chunk_autotune_enabled(policy: BackendPolicy) -> bool:
    raw_value = _optional_env_value(_CHUNK_AUTOTUNE_ENV)
    if raw_value is not None:
        return raw_value.strip().lower() in _TRUTHY_ENV_VALUES
    return policy.backend == "jax" and policy.jax_platform == "cuda"


def _resolve_sharding_strategy(mode: str, policy: BackendPolicy) -> str:
    del policy
    raw_value = _optional_nonempty_env(_SHARDING_STRATEGY_ENV)
    if raw_value is not None:
        return _validate_sharding_strategy(raw_value, source=_SHARDING_STRATEGY_ENV)
    return _MODE_SHARDING_DEFAULTS[mode]


def _resolve_sharding_axis_name() -> str:
    raw_value = _optional_nonempty_env(_SHARDING_AXIS_ENV)
    if raw_value is None:
        return _DEFAULT_SHARDING_AXIS_NAME
    return raw_value


def _resolve_min_points_to_shard(policy: BackendPolicy) -> int:
    env_value = _optional_positive_int_env(_MIN_POINTS_TO_SHARD_ENV)
    if env_value is not None:
        return env_value
    return _MIN_POINTS_TO_SHARD_BY_POLICY.get(policy.chunk_policy, 0)


def _detect_local_jax_device_count(policy: BackendPolicy) -> int:
    try:
        import jax
    except ImportError:
        return 0
    try:
        backend_name = "gpu" if policy.jax_platform == "cuda" else policy.jax_platform
        return len(jax.local_devices(backend=backend_name))
    except Exception:
        return 0


def _parse_visible_cuda_device_index() -> int | None:
    raw_value = _optional_env_value("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return None
    first = raw_value.split(",", 1)[0].strip()
    if not first or first in {"-1", "none", "NoDevFiles"}:
        return None
    try:
        value = int(first)
    except ValueError:
        return None
    return value if value >= 0 else None


def _detect_active_jax_cuda_device_index() -> int | None:
    env_index = _parse_visible_cuda_device_index()
    if env_index is not None:
        return env_index
    try:
        import jax
    except ImportError:
        return None
    try:
        devices = jax.local_devices(backend="gpu")
    except Exception:
        return None
    if not devices:
        return None
    device = devices[0]
    for attr in ("local_hardware_id", "id"):
        value = getattr(device, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _parse_nvidia_smi_memory_row(raw_row: str) -> tuple[int, int] | None:
    fields = [field.strip() for field in raw_row.split(",")]
    if len(fields) != 2:
        return None
    try:
        return int(float(fields[0])), int(float(fields[1]))
    except ValueError:
        return None


def _query_gpu_total_memory_mb_from_nvidia_smi(device_index: int | None = None) -> int | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total",
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.extend(["-i", str(device_index)])
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    for line in lines:
        parsed = _parse_nvidia_smi_memory_row(line)
        if parsed is None:
            continue
        index, value = parsed
        if device_index is not None and index != device_index:
            continue
        if value > 0:
            return value
    return None


def _resolve_gpu_total_memory_mb(policy: BackendPolicy) -> tuple[int | None, str | None]:
    if policy.jax_platform != "cuda":
        return None, None
    env_value = _optional_positive_int_env(_GPU_MEMORY_TOTAL_MB_ENV)
    if env_value is not None:
        if env_value == 0:
            raise ValueError(f"{_GPU_MEMORY_TOTAL_MB_ENV} must be > 0 when set")
        return env_value, _GPU_MEMORY_TOTAL_MB_ENV
    device_index = _detect_active_jax_cuda_device_index()
    detected = _query_gpu_total_memory_mb_from_nvidia_smi(device_index)
    if detected is None:
        return None, None
    if device_index is None:
        return detected, "nvidia-smi"
    return detected, f"nvidia-smi[{device_index}]"


def _resolve_autotuned_chunk_sizes(
    chunk_policy: str,
    gpu_total_memory_mb: int | None,
) -> dict[str, int] | None:
    if gpu_total_memory_mb is None:
        return None
    buckets = _AUTOTUNED_CHUNK_SIZES_BY_POLICY.get(chunk_policy, ())
    for max_total_mb, sizes in buckets:
        if max_total_mb is None or gpu_total_memory_mb <= max_total_mb:
            return dict(sizes)
    return None


def _static_chunk_sizes(mode: str, chunk_policy: str) -> dict[str, int]:
    return {
        "coil_chunk_size": _FIELD_KERNEL_DEFAULTS[mode]["coil_chunk_size"],
        "quadrature_block_size": _FIELD_KERNEL_DEFAULTS[mode][
            "quadrature_block_size"
        ],
        "point_chunk_size": _point_chunk_size_default(chunk_policy),
        "pairwise_penalty_chunk_size": _pairwise_penalty_chunk_size_default(
            chunk_policy
        ),
    }


def _apply_chunk_env_overrides(chunk_sizes: dict[str, int]) -> dict[str, int]:
    resolved = dict(chunk_sizes)
    for key, env_name in _FIELD_KERNEL_ENV_BY_KEY.items():
        value = _optional_positive_int_env(env_name)
        if value is not None:
            resolved[key] = value
    pairwise_value = _optional_positive_int_env(_PAIRWISE_PENALTY_CHUNK_SIZE_ENV)
    if pairwise_value is not None:
        resolved["pairwise_penalty_chunk_size"] = pairwise_value
    return resolved


def _build_chunk_tuning(
    mode: str,
    policy: BackendPolicy,
) -> ChunkTuning:
    chunk_sizes = _static_chunk_sizes(mode, policy.chunk_policy)
    autotuned = False
    autotune_source = None
    gpu_total_memory_mb = None
    if _resolve_chunk_autotune_enabled(policy):
        gpu_total_memory_mb, autotune_source = _resolve_gpu_total_memory_mb(policy)
        autotuned_chunk_sizes = _resolve_autotuned_chunk_sizes(
            policy.chunk_policy,
            gpu_total_memory_mb,
        )
        if autotuned_chunk_sizes is not None:
            chunk_sizes.update(autotuned_chunk_sizes)
            autotuned = True
    chunk_sizes = _apply_chunk_env_overrides(chunk_sizes)
    effective_chunk_policy = policy.chunk_policy
    if policy.transfer_guard == "disallow":
        effective_chunk_policy = f"{policy.chunk_policy}_dense_audit"
        chunk_sizes["coil_chunk_size"] = 0
        chunk_sizes["quadrature_block_size"] = 0
        chunk_sizes["point_chunk_size"] = 0
    return ChunkTuning(
        mode=mode,
        chunk_policy=effective_chunk_policy,
        coil_chunk_size=chunk_sizes["coil_chunk_size"],
        quadrature_block_size=chunk_sizes["quadrature_block_size"],
        point_chunk_size=chunk_sizes["point_chunk_size"],
        pairwise_penalty_chunk_size=chunk_sizes["pairwise_penalty_chunk_size"],
        autotuned=autotuned,
        autotune_source=autotune_source,
        gpu_total_memory_mb=gpu_total_memory_mb,
    )


def _build_sharding_tuning(
    mode: str,
    policy: BackendPolicy,
) -> ShardingTuning:
    strategy = _resolve_sharding_strategy(mode, policy)
    if policy.backend != "jax":
        strategy = "none"
    device_count = _detect_local_jax_device_count(policy)
    return ShardingTuning(
        mode=mode,
        strategy=strategy,
        mesh_axis_name=_resolve_sharding_axis_name(),
        min_points_to_shard=_resolve_min_points_to_shard(policy),
        device_count=device_count,
        active=strategy != "none" and device_count > 1,
        platform=policy.jax_platform,
    )


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
        gpu_reduction_order_max_ulp=defaults["gpu_reduction_order_max_ulp"],
        gpu_reduction_order_rel_tol=defaults["gpu_reduction_order_rel_tol"],
        gpu_reproducibility_seed=defaults["gpu_reproducibility_seed"],
        gpu_reproducibility_sample_size=defaults["gpu_reproducibility_sample_size"],
        tolerance_ratchet_factor=defaults["tolerance_ratchet_factor"],
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
_warned_jax_fallbacks: set[tuple[str, str, str]] = set()


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
_cached_chunk_tuning: ChunkTuning | None = None
_cached_sharding_tuning: ShardingTuning | None = None


def get_chunk_tuning(mode: str | None = None) -> ChunkTuning:
    """Return the resolved chunk sizes and autotuning metadata."""
    global _cached_chunk_tuning
    if mode is None and _cached_chunk_tuning is not None:
        return _cached_chunk_tuning
    resolved_mode = _resolve_mode(mode)
    tuning = _build_chunk_tuning(
        resolved_mode,
        get_backend_policy(resolved_mode),
    )
    if mode is None:
        _cached_chunk_tuning = tuning
    return tuning


def get_sharding_tuning(mode: str | None = None) -> ShardingTuning:
    """Return the resolved sharding strategy and mesh activation metadata."""
    global _cached_sharding_tuning
    if mode is None and _cached_sharding_tuning is not None:
        return _cached_sharding_tuning
    resolved_mode = _resolve_mode(mode)
    tuning = _build_sharding_tuning(
        resolved_mode,
        get_backend_policy(resolved_mode),
    )
    if mode is None:
        _cached_sharding_tuning = tuning
    return tuning


def get_field_kernel_tuning(mode: str | None = None) -> FieldKernelTuning:
    """Return the low-level field-kernel tuning contract for the resolved mode."""
    global _cached_field_kernel_tuning
    if mode is None and _cached_field_kernel_tuning is not None:
        return _cached_field_kernel_tuning
    chunk_tuning = get_chunk_tuning(mode)
    tuning = FieldKernelTuning(
        mode=chunk_tuning.mode,
        chunk_policy=chunk_tuning.chunk_policy,
        coil_chunk_size=chunk_tuning.coil_chunk_size,
        quadrature_block_size=chunk_tuning.quadrature_block_size,
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
    return get_chunk_tuning(mode).point_chunk_size


def get_pairwise_penalty_chunk_size(mode: str | None = None) -> int:
    """Return the pairwise-penalty block size for curve/surface reductions."""
    return get_chunk_tuning(mode).pairwise_penalty_chunk_size


def get_sharding_strategy(mode: str | None = None) -> str:
    """Return the resolved sharding strategy label for the mode."""
    return get_sharding_tuning(mode).strategy


def should_shard_points(mode: str | None = None) -> bool:
    """Return ``True`` when point-axis sharding is active for the mode."""
    tuning = get_sharding_tuning(mode)
    return tuning.active and tuning.strategy == "points"


def get_debug_nans(mode: str | None = None) -> bool:
    """Return the debug-NaN runtime guardrail state for the resolved mode."""
    return get_backend_policy(mode).debug_nans


def get_transfer_guard(mode: str | None = None) -> str | None:
    """Return the active JAX transfer-guard policy for the resolved mode."""
    return get_backend_policy(mode).transfer_guard


def get_compilation_cache_dir(mode: str | None = None) -> str | None:
    """Return the active JAX compilation-cache directory for the resolved mode."""
    return get_backend_policy(mode).compilation_cache_dir


def _backend_cache_clear_callback_key(
    callback: Callable[[], None],
) -> _BackendCacheClearCallbackKey:
    return (callback.__module__, callback.__qualname__)


def register_backend_cache_clear(callback: Callable[[], None]) -> None:
    """Register a callback that should run whenever backend caches are cleared."""
    _backend_cache_clear_callbacks[_backend_cache_clear_callback_key(callback)] = (
        callback
    )


def _run_backend_cache_clear_callbacks() -> None:
    for callback in _backend_cache_clear_callbacks.values():
        callback()


def _reset_backend_runtime_caches() -> None:
    global _cached_backend_policy, _cached_chunk_tuning, _cached_field_kernel_tuning, _cached_sharding_tuning
    _cached_backend_policy = None
    _cached_chunk_tuning = None
    _cached_field_kernel_tuning = None
    _cached_sharding_tuning = None
    _run_backend_cache_clear_callbacks()
    _warned_jax_fallbacks.clear()


def invalidate_backend_cache() -> None:
    """Clear the cached backend configuration and derived caches.

    Call this after mutating ``SIMSOPT_*`` environment variables directly
    (outside of ``set_backend()``) so the next ``get_backend_config()`` call
    re-reads the environment.  Test fixtures should call this when they
    manipulate env vars via ``monkeypatch`` or context managers.
    """
    global _cached_backend_config
    _cached_backend_config = None
    _reset_backend_runtime_caches()


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


def warn_if_jax_fallback(*, component: str, detail: str) -> None:
    """Warn once when non-strict JAX mode uses a legacy fallback path."""
    config = get_backend_config()
    if config.backend != "jax" or config.strict:
        return

    cache_key = (config.mode, component, detail)
    if cache_key in _warned_jax_fallbacks:
        return
    _warned_jax_fallbacks.add(cache_key)
    warnings.warn(
        f"{component} is using {detail} while simsopt backend mode "
        f"{config.mode!r} is active. This path should be treated as a legacy "
        "adapter seam; enable strict mode to reject it.",
        RuntimeWarning,
        stacklevel=2,
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
    if config.transfer_guard is not None:
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
    global _cached_backend_config
    config = _config_from_mode(
        mode,
        strict=bool(strict),
        debug_nans=debug_nans,
        transfer_guard=transfer_guard,
        compilation_cache_dir=compilation_cache_dir,
    )
    _cached_backend_config = config
    _reset_backend_runtime_caches()
    for env_name, attribute_name in _SYNCED_RUNTIME_ENV_VALUES:
        os.environ[env_name] = _runtime_env_value(
            attribute_name, getattr(config, attribute_name)
        )
    if configure_runtime:
        apply_jax_runtime_config()
    return config
