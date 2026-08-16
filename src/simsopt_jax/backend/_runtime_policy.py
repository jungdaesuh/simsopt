"""Policy selection, validation, and config/policy resolution for the JAX backend.

Owns the mode/precision/dtype/residency contract and pure builders that turn
environment and keyword inputs into :class:`BackendConfig` /
:class:`BackendPolicy`. Process-global caches and configure-before-JAX side
effects live in :mod:`simsopt_jax.backend.runtime`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypeVar, cast

import numpy as np

from simsopt_jax.numerical_policy import CertificateDType

_ExplicitT = TypeVar("_ExplicitT")
_ResolvedT = TypeVar("_ResolvedT")

PrecisionSelection = Literal["mode_default", "fp64", "mixed"]
ResolvedPrecision = Literal["fp32_smoke", "fp64", "mixed"]
BackendMode = Literal[
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_cpu_float32_smoke",
    "jax_gpu_fast",
    "jax_gpu_parity",
]
JaxDevice = Literal["cpu", "gpu"]
ExecutionIntent = Literal["fast", "parity"]

_VALID_BACKENDS = ("cpu", "jax")
_VALID_PLATFORMS = ("cpu", "cuda")
_VALID_POLICY_DTYPES = ("float32", "float64")
_VALID_PRECISION_SELECTIONS = ("mode_default", "fp64", "mixed")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

_BACKEND_ENV = "SIMSOPT_BACKEND"
_BACKEND_LEGACY_ENV = "STAGE2_BACKEND"
_PLATFORM_ENV = "SIMSOPT_JAX_PLATFORM"
_PLATFORM_LEGACY_ENV = "SIMSOPT_JAX_BACKEND"
_MODE_ENV = "SIMSOPT_BACKEND_MODE"
_PRECISION_ENV = "SIMSOPT_PRECISION"
_OBSOLETE_MIXED_PRECISION_ENV = "SIMSOPT_MIXED_PRECISION"
_STRICT_ENV = "SIMSOPT_BACKEND_STRICT"
_TARGET_LANE_STRICT_ENV = "SIMSOPT_TARGET_LANE_STRICT"
_DEBUG_ENV = "SIMSOPT_DEBUG"
_DEBUG_NANS_ENV = "SIMSOPT_JAX_DEBUG_NANS"
_DISABLE_JIT_ENV = "SIMSOPT_JAX_DISABLE_JIT"
_TRANSFER_GUARD_ENV = "SIMSOPT_JAX_TRANSFER_GUARD"
_COMPILATION_CACHE_DIR_ENV = "SIMSOPT_JAX_COMPILATION_CACHE_DIR"
_JAX_COMPILATION_CACHE_DIR_ENV = "JAX_COMPILATION_CACHE_DIR"
_COIL_CHUNK_SIZE_ENV = "SIMSOPT_JAX_COIL_CHUNK_SIZE"
_QUADRATURE_BLOCK_SIZE_ENV = "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE"
_POINT_CHUNK_SIZE_ENV = "SIMSOPT_JAX_POINT_CHUNK_SIZE"
_PAIRWISE_PENALTY_CHUNK_SIZE_ENV = "SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE"
_CHUNK_AUTOTUNE_ENV = "SIMSOPT_JAX_CHUNK_AUTOTUNE"
_GPU_MEMORY_TOTAL_MB_ENV = "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB"
_GPU_PREALLOCATE_ENV = "SIMSOPT_JAX_GPU_PREALLOCATE"
_GPU_MEM_FRACTION_ENV = "SIMSOPT_JAX_GPU_MEM_FRACTION"
_GPU_ALLOCATOR_ENV = "SIMSOPT_JAX_GPU_ALLOCATOR"
_TF_GPU_ALLOCATOR_OVERRIDE_ENV = "SIMSOPT_TF_GPU_ALLOCATOR"
_MAX_DENSE_JACOBIAN_BYTES_CPU_ENV = "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_CPU"
_MAX_DENSE_JACOBIAN_BYTES_GPU_ENV = "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU"
_SHARDING_STRATEGY_ENV = "SIMSOPT_JAX_SHARDING"
_SHARDING_AXIS_ENV = "SIMSOPT_JAX_SHARDING_AXIS"
_SHARDING_COIL_AXIS_ENV = "SIMSOPT_JAX_COIL_SHARDING_AXIS"
_MIN_POINTS_TO_SHARD_ENV = "SIMSOPT_JAX_MIN_POINTS_TO_SHARD"
_MIN_PAIRWISE_ROWS_TO_SHARD_ENV = "SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD"
_MIN_COILS_TO_SHARD_ENV = "SIMSOPT_JAX_MIN_COILS_TO_SHARD"
_DISTRIBUTED_INIT_ENV = "SIMSOPT_JAX_DISTRIBUTED_INIT"
_DISTRIBUTED_COORDINATOR_ADDRESS_ENV = "SIMSOPT_JAX_COORDINATOR_ADDRESS"
_DISTRIBUTED_NUM_PROCESSES_ENV = "SIMSOPT_JAX_NUM_PROCESSES"
_DISTRIBUTED_PROCESS_ID_ENV = "SIMSOPT_JAX_PROCESS_ID"
_DISTRIBUTED_LOCAL_DEVICE_IDS_ENV = "SIMSOPT_JAX_LOCAL_DEVICE_IDS"
_JAX_PLATFORMS_ENV = "JAX_PLATFORMS"
_XLA_FLAGS_ENV = "XLA_FLAGS"
_XLA_PYTHON_CLIENT_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
_XLA_PYTHON_CLIENT_MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"
_XLA_PYTHON_CLIENT_ALLOCATOR_ENV = "XLA_PYTHON_CLIENT_ALLOCATOR"
_XLA_CLIENT_MEM_FRACTION_ENV = "XLA_CLIENT_MEM_FRACTION"
_TF_GPU_ALLOCATOR_ENV = "TF_GPU_ALLOCATOR"
_VALID_TRANSFER_GUARDS = ("allow", "log", "disallow")
_VALID_DEFAULT_RESIDENCIES = ("device", "host")
_VALID_DEFAULT_OPTIMIZER_BACKENDS = ("scipy", "ondevice")
_VALID_GPU_ALLOCATORS = ("platform", "vmm")
_VALID_TF_GPU_ALLOCATORS = ("cuda_malloc_async",)
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
    (_PRECISION_ENV, "precision"),
    (_STRICT_ENV, "strict"),
    (_DEBUG_NANS_ENV, "debug_nans"),
    (_DISABLE_JIT_ENV, "disable_jit"),
    (_TRANSFER_GUARD_ENV, "transfer_guard"),
    (_COMPILATION_CACHE_DIR_ENV, "compilation_cache_dir"),
    (_GPU_PREALLOCATE_ENV, "xla_gpu_preallocate"),
    (_GPU_MEM_FRACTION_ENV, "xla_gpu_mem_fraction"),
    (_GPU_ALLOCATOR_ENV, "xla_gpu_allocator"),
    (_TF_GPU_ALLOCATOR_OVERRIDE_ENV, "tf_gpu_allocator"),
    (_BACKEND_ENV, "backend"),
    (_BACKEND_LEGACY_ENV, "backend"),
    (_PLATFORM_ENV, "jax_platform"),
    (_PLATFORM_LEGACY_ENV, "jax_platform"),
    (_JAX_PLATFORMS_ENV, "jax_platforms"),
)
VALID_BACKEND_MODES: tuple[BackendMode, ...] = (
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_cpu_float32_smoke",
    "jax_gpu_fast",
    "jax_gpu_parity",
)

_JAX_EXECUTION_MODES: dict[tuple[JaxDevice, ExecutionIntent], BackendMode] = {
    ("cpu", "fast"): "jax_cpu_fast",
    ("cpu", "parity"): "jax_cpu_parity",
    ("gpu", "fast"): "jax_gpu_fast",
    ("gpu", "parity"): "jax_gpu_parity",
}

_MODE_TO_RUNTIME = {
    "native_cpu": ("cpu", "cpu"),
    "jax_cpu_fast": ("jax", "cpu"),
    "jax_cpu_parity": ("jax", "cpu"),
    "jax_cpu_float32_smoke": ("jax", "cpu"),
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
_NO_GPU_MEMORY_DEFAULTS = {
    "xla_gpu_preallocate": None,
    "xla_gpu_mem_fraction": None,
    "xla_gpu_allocator": None,
    "tf_gpu_allocator": None,
}
_GPU_MEMORY_MODE_DEFAULTS = {
    "xla_gpu_preallocate": False,
    "xla_gpu_mem_fraction": None,
    "xla_gpu_allocator": None,
    "tf_gpu_allocator": None,
}
_DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU = 4 * 1024 * 1024 * 1024
_DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU = 256 * 1024 * 1024
_FLOAT64_LINEAR_SOLVE_TOLERANCE_FLOOR = 1e-14
_FLOAT64_LINEAR_SOLVE_TOLERANCE_CAP = 1e-10
_FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_FLOOR = float(np.sqrt(np.finfo(np.float32).eps))
_FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_CAP = 1e-3
_FLOAT64_LINEAR_SOLVE_DEFAULTS = {
    "linear_solve_tolerance_floor": _FLOAT64_LINEAR_SOLVE_TOLERANCE_FLOOR,
    "linear_solve_tolerance_cap": _FLOAT64_LINEAR_SOLVE_TOLERANCE_CAP,
}
_FLOAT32_SMOKE_LINEAR_SOLVE_DEFAULTS = {
    "linear_solve_tolerance_floor": _FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_FLOOR,
    "linear_solve_tolerance_cap": _FLOAT32_SMOKE_LINEAR_SOLVE_TOLERANCE_CAP,
}
_HOST_CALLBACK_SUPPORTED_DEFAULTS = {"supports_host_callback": True}
_BUFFER_DONATION_SUPPORTED_DEFAULTS = {"supports_buffer_donation": True}

_MODE_POLICY_DEFAULTS = {
    "native_cpu": {
        "parity_mode": False,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "default_residency": "host",
        "default_optimizer_backend": "scipy",
        **_HOST_CALLBACK_SUPPORTED_DEFAULTS,
        **_BUFFER_DONATION_SUPPORTED_DEFAULTS,
        "chunk_policy": "host_reference",
        "tolerance_tier": "cpu_reference",
        "compilation_cache_policy": "not_applicable",
        "matmul_precision": "highest",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU,
        "provenance_label": "native_cpu",
        **_FLOAT64_LINEAR_SOLVE_DEFAULTS,
        **_NO_GPU_MEMORY_DEFAULTS,
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
    "jax_cpu_fast": {
        "parity_mode": False,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "default_residency": "device",
        "default_optimizer_backend": "ondevice",
        **_HOST_CALLBACK_SUPPORTED_DEFAULTS,
        **_BUFFER_DONATION_SUPPORTED_DEFAULTS,
        "chunk_policy": "performance_tuned",
        "tolerance_tier": "fast",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "default",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU,
        "provenance_label": "jax_cpu_fast",
        **_FLOAT64_LINEAR_SOLVE_DEFAULTS,
        **_NO_GPU_MEMORY_DEFAULTS,
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
    "jax_cpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "default_residency": "device",
        "default_optimizer_backend": "ondevice",
        **_HOST_CALLBACK_SUPPORTED_DEFAULTS,
        **_BUFFER_DONATION_SUPPORTED_DEFAULTS,
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "highest",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU,
        "provenance_label": "jax_cpu_parity",
        **_FLOAT64_LINEAR_SOLVE_DEFAULTS,
        **_NO_GPU_MEMORY_DEFAULTS,
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
    "jax_cpu_float32_smoke": {
        "parity_mode": False,
        "requires_x64": False,
        "runtime_dtype": "float32",
        "host_dtype": "float32",
        "default_residency": "device",
        "default_optimizer_backend": "ondevice",
        **_HOST_CALLBACK_SUPPORTED_DEFAULTS,
        **_BUFFER_DONATION_SUPPORTED_DEFAULTS,
        "chunk_policy": "stable_default",
        "tolerance_tier": "float32_smoke",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "default",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU,
        "provenance_label": "jax_cpu_float32_smoke",
        **_FLOAT32_SMOKE_LINEAR_SOLVE_DEFAULTS,
        **_NO_GPU_MEMORY_DEFAULTS,
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
    "jax_gpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "default_residency": "device",
        "default_optimizer_backend": "ondevice",
        **_HOST_CALLBACK_SUPPORTED_DEFAULTS,
        **_BUFFER_DONATION_SUPPORTED_DEFAULTS,
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "highest",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU,
        "provenance_label": "jax_gpu_parity",
        **_FLOAT64_LINEAR_SOLVE_DEFAULTS,
        **_GPU_MEMORY_MODE_DEFAULTS,
        "gpu_reduction_order_max_ulp": 10,
        "gpu_reduction_order_rel_tol": 1e-12,
        "gpu_reproducibility_seed": 1729,
        "gpu_reproducibility_sample_size": 1000,
        "tolerance_ratchet_factor": 10.0,
    },
    "jax_gpu_fast": {
        "parity_mode": False,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "default_residency": "device",
        "default_optimizer_backend": "ondevice",
        **_HOST_CALLBACK_SUPPORTED_DEFAULTS,
        **_BUFFER_DONATION_SUPPORTED_DEFAULTS,
        "chunk_policy": "performance_tuned",
        "tolerance_tier": "fast",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "default",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU,
        "provenance_label": "jax_gpu_fast",
        **_FLOAT64_LINEAR_SOLVE_DEFAULTS,
        **_GPU_MEMORY_MODE_DEFAULTS,
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
}

_DEFAULT_TRANSFER_GUARD_BY_MODE = {
    "native_cpu": None,
    "jax_cpu_fast": "log",
    "jax_cpu_parity": "log",
    "jax_cpu_float32_smoke": "log",
    "jax_gpu_parity": "log",
    "jax_gpu_fast": "log",
}


@dataclass(frozen=True)
class BackendConfig:
    mode: BackendMode
    backend: str
    jax_platform: str
    precision: PrecisionSelection = "mode_default"
    strict: bool = False
    debug_nans: bool = False
    disable_jit: bool = False
    transfer_guard: str | None = None
    compilation_cache_dir: str | None = None
    xla_gpu_preallocate: bool | None = None
    xla_gpu_mem_fraction: float | None = None
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None


@dataclass(frozen=True)
class BackendPolicy:
    """Numerical-policy contract for one resolved backend mode.

    The GPU reproducibility fields are reporting/acceptance metadata for parity
    lanes. They document the expected tolerance budget and sampling defaults
    used by CI and diagnostics. For CUDA parity lanes, runtime configuration
    validates the required pre-import XLA determinism flags, but the policy
    fields themselves do not directly force kernel execution behavior.
    """

    mode: BackendMode
    backend: str
    jax_platform: str
    strict: bool
    parity_mode: bool
    requires_x64: bool
    precision: PrecisionSelection
    resolved_precision: ResolvedPrecision
    runtime_dtype: str
    host_dtype: str
    compute_dtype: str
    certificate_dtype: CertificateDType | None
    default_residency: str
    default_optimizer_backend: str
    supports_host_callback: bool
    supports_buffer_donation: bool
    chunk_policy: str
    tolerance_tier: str
    compilation_cache_policy: str
    matmul_precision: str
    max_dense_jacobian_bytes: int | None
    linear_solve_tolerance_floor: float
    linear_solve_tolerance_cap: float | None
    provenance_label: str
    xla_gpu_preallocate: bool | None
    xla_gpu_mem_fraction: float | None
    xla_gpu_allocator: Literal["platform", "vmm"] | None
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None
    gpu_reduction_order_max_ulp: int | None
    gpu_reduction_order_rel_tol: float | None
    gpu_reproducibility_seed: int | None
    gpu_reproducibility_sample_size: int | None
    tolerance_ratchet_factor: float | None
    debug_nans: bool
    disable_jit: bool
    transfer_guard: str | None
    compilation_cache_dir: str | None


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in _TRUTHY_ENV_VALUES


def _parse_bool_value(raw_value: str, *, source: str) -> bool:
    value = raw_value.strip().lower()
    if value in _TRUTHY_ENV_VALUES:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{source}={raw_value!r} must be a boolean value")


def _optional_bool_env(name: str) -> bool | None:
    raw_value = _optional_env_value(name)
    if raw_value is None:
        return None
    return _parse_bool_value(raw_value, source=name)


def _validate_gpu_allocator(
    value: object | None,
    *,
    source: str,
) -> Literal["platform", "vmm"] | None:
    if value in (None, ""):
        return None
    if value == "platform":
        return "platform"
    if value == "vmm":
        return "vmm"
    raise ValueError(
        f"{source}={value!r} is not valid. Accepted: {_VALID_GPU_ALLOCATORS}"
    )


def _validate_tf_gpu_allocator(
    value: object | None,
    *,
    source: str,
) -> Literal["cuda_malloc_async"] | None:
    if value in (None, ""):
        return None
    if value == "cuda_malloc_async":
        return "cuda_malloc_async"
    raise ValueError(
        f"{source}={value!r} is not valid. Accepted: {_VALID_TF_GPU_ALLOCATORS}"
    )


def _validate_backend(value: str, *, source: str) -> str:
    if value not in _VALID_BACKENDS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_BACKENDS}"
        )
    return value


def _validate_platform(value: str, *, source: str) -> str:
    value = value.lower()
    if value not in _VALID_PLATFORMS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_PLATFORMS}"
        )
    return value


def _validate_mode(mode: str) -> BackendMode:
    if mode not in VALID_BACKEND_MODES:
        raise ValueError(
            f"Backend mode {mode!r} is not valid. Accepted: {VALID_BACKEND_MODES}"
        )
    return cast(BackendMode, mode)


@dataclass(frozen=True)
class JaxExecutionProfile:
    """Resolved JAX placement, numerical intent, and evidence eligibility."""

    device: JaxDevice
    intent: ExecutionIntent
    mode: BackendMode
    certification_eligible: bool


def resolve_jax_execution_profile(
    device: JaxDevice | str,
    intent: ExecutionIntent | str = "fast",
) -> JaxExecutionProfile:
    """Resolve the public orthogonal JAX selector to one canonical mode."""
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device={device!r} is not valid. Accepted: ('cpu', 'gpu')")
    if intent not in ("fast", "parity"):
        raise ValueError(
            f"intent={intent!r} is not valid. Accepted: ('fast', 'parity')"
        )
    resolved_device = cast(JaxDevice, device)
    resolved_intent = cast(ExecutionIntent, intent)
    return JaxExecutionProfile(
        device=resolved_device,
        intent=resolved_intent,
        mode=_JAX_EXECUTION_MODES[(resolved_device, resolved_intent)],
        certification_eligible=resolved_intent == "parity",
    )


def _validate_precision_selection(
    value: object,
    *,
    source: str,
) -> PrecisionSelection:
    if value not in _VALID_PRECISION_SELECTIONS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_PRECISION_SELECTIONS}"
        )
    return cast(PrecisionSelection, value)


def _reject_obsolete_precision_environment() -> None:
    if _OBSOLETE_MIXED_PRECISION_ENV in os.environ:
        raise ValueError(
            f"{_OBSOLETE_MIXED_PRECISION_ENV} is not supported; "
            f"use {_PRECISION_ENV}=mixed instead."
        )


def _validate_precision_for_mode(
    mode: str,
    precision: PrecisionSelection,
) -> PrecisionSelection:
    if mode == "jax_cpu_float32_smoke" and precision != "mode_default":
        raise ValueError(
            "jax_cpu_float32_smoke only supports precision='mode_default'; "
            "its full-FP32 contract cannot be overridden."
        )
    if mode == "native_cpu" and precision == "mixed":
        raise ValueError(
            "native_cpu does not support mixed precision; use precision='fp64' "
            "or precision='mode_default'."
        )
    return precision


def _resolved_precision_for_mode(
    mode: str,
    precision: PrecisionSelection,
) -> ResolvedPrecision:
    if mode == "jax_cpu_float32_smoke":
        return "fp32_smoke"
    if precision == "mixed":
        return "mixed"
    return "fp64"


def _validate_transfer_guard(value: str | None, *, source: str) -> str | None:
    if value in (None, ""):
        return None
    if value not in _VALID_TRANSFER_GUARDS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_TRANSFER_GUARDS}"
        )
    return value


def _validate_default_residency(value: object, *, mode: str) -> str:
    residency = str(value)
    if residency not in _VALID_DEFAULT_RESIDENCIES:
        raise ValueError(
            f"Backend mode {mode!r} has unsupported default_residency={residency!r}. "
            f"Accepted: {_VALID_DEFAULT_RESIDENCIES}."
        )
    return residency


def _validate_default_optimizer_backend(value: object, *, mode: str) -> str:
    optimizer_backend = str(value)
    if optimizer_backend not in _VALID_DEFAULT_OPTIMIZER_BACKENDS:
        raise ValueError(
            f"Backend mode {mode!r} has unsupported "
            f"default_optimizer_backend={optimizer_backend!r}. "
            f"Accepted: {_VALID_DEFAULT_OPTIMIZER_BACKENDS}."
        )
    return optimizer_backend


def _default_compilation_cache_dir(mode: str) -> str | None:
    resolved_mode = _validate_mode(mode)
    backend, _platform = _MODE_TO_RUNTIME[resolved_mode]
    if backend != "jax":
        return None
    return str(Path.home() / ".cache" / "simsopt-jax-xla")


def _optional_env_value(name: str) -> str | None:
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return None
    return raw_value


def _optional_nonneg_int_env(name: str) -> int | None:
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


def _runtime_jax_platform_value(platform: str) -> str:
    # Single canonical helper for lowering ``BackendConfig.jax_platform`` to the
    # value JAX expects in ``JAX_PLATFORMS`` / ``jax.config["jax_platforms"]``.
    # Currently identity (cpu/cuda already match upstream casing); kept as
    # a single edit site so a future platform whose JAX name diverges from the
    # simsopt mode token can be remapped here without touching call sites.
    return platform


_CUDA_WITH_CPU_FALLBACK_PLATFORMS = "cuda,cpu"


def _runtime_jax_platforms_value(platform: str) -> str:
    if platform != "cuda":
        return _runtime_jax_platform_value(platform)
    requested_platforms = _optional_env_value(_JAX_PLATFORMS_ENV)
    requested_parts = (
        ()
        if requested_platforms is None
        else tuple(part.strip().lower() for part in requested_platforms.split(","))
    )
    if "cuda" in requested_parts and "cpu" in requested_parts:
        return _CUDA_WITH_CPU_FALLBACK_PLATFORMS
    return _runtime_jax_platform_value(platform)


def _runtime_jax_backend_name(platform: str) -> str:
    if platform == "cuda":
        return "gpu"
    return _runtime_jax_platform_value(platform)


def _primary_jax_platforms_env_platform() -> str | None:
    platforms = _optional_env_value(_JAX_PLATFORMS_ENV)
    if platforms is None:
        return None
    parts = tuple(part.strip().lower() for part in platforms.split(",") if part.strip())
    if not parts:
        return None
    primary_platform = parts[0]
    return primary_platform if primary_platform in _VALID_PLATFORMS else None


def _resolve_kwarg(
    explicit: _ExplicitT | None,
    *,
    parse_explicit: Callable[[_ExplicitT], _ResolvedT],
    env_names: tuple[str, ...],
    parse_env: Callable[[str, str], _ResolvedT],
    read_default: Callable[[], _ResolvedT],
) -> _ResolvedT:
    if explicit is not None:
        return parse_explicit(explicit)
    for env_name in env_names:
        env_value = _optional_env_value(env_name)
        if env_value is not None:
            return parse_env(env_value, env_name)
    return read_default()


def _optional_bool_policy_default(value: object) -> bool | None:
    return None if value is None else bool(value)


def _debug_overlay_enabled() -> bool:
    return bool(_optional_bool_env(_DEBUG_ENV))


def _default_transfer_guard(mode: str) -> str | None:
    return _DEFAULT_TRANSFER_GUARD_BY_MODE[_validate_mode(mode)]


def _validate_mem_fraction_value(value: object, *, source: str) -> float:
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{source}={value!r} must be in (0, 1]")
    return fraction


def _config_from_mode(
    mode: str,
    *,
    strict: bool,
    precision: PrecisionSelection | None = None,
    debug_nans: bool | None = None,
    disable_jit: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
    xla_gpu_preallocate: bool | None = None,
    xla_gpu_mem_fraction: float | None = None,
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None,
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None,
) -> BackendConfig:
    _reject_obsolete_precision_environment()
    mode = _validate_mode(mode)
    backend, jax_platform = _MODE_TO_RUNTIME[mode]
    debug_overlay = _debug_overlay_enabled()
    defaults = _get_mode_policy_defaults(mode)
    resolved_precision = _validate_precision_for_mode(
        mode,
        _resolve_kwarg(
            precision,
            parse_explicit=lambda value: _validate_precision_selection(
                value,
                source="precision",
            ),
            env_names=(_PRECISION_ENV,),
            parse_env=lambda value, source: _validate_precision_selection(
                value,
                source=source,
            ),
            read_default=lambda: "mode_default",
        ),
    )
    if debug_overlay:
        resolved_debug_nans = True
        resolved_disable_jit = True
        resolved_transfer_guard = "disallow"
    else:
        resolved_debug_nans = _resolve_kwarg(
            debug_nans,
            parse_explicit=bool,
            env_names=(_DEBUG_NANS_ENV,),
            parse_env=lambda value, source: value.strip().lower() in _TRUTHY_ENV_VALUES,
            read_default=lambda: False,
        )
        resolved_disable_jit = _resolve_kwarg(
            disable_jit,
            parse_explicit=bool,
            env_names=(_DISABLE_JIT_ENV,),
            parse_env=lambda value, source: _parse_bool_value(value, source=source),
            read_default=lambda: False,
        )
        resolved_transfer_guard = _resolve_kwarg(
            transfer_guard,
            parse_explicit=lambda value: _validate_transfer_guard(
                value,
                source="transfer_guard",
            ),
            env_names=(_TRANSFER_GUARD_ENV,),
            parse_env=lambda value, source: _validate_transfer_guard(
                value,
                source=source,
            ),
            read_default=lambda: _default_transfer_guard(mode),
        )
    resolved_compilation_cache_dir = _resolve_kwarg(
        compilation_cache_dir,
        parse_explicit=lambda value: value or None,
        env_names=(_COMPILATION_CACHE_DIR_ENV, _JAX_COMPILATION_CACHE_DIR_ENV),
        parse_env=lambda value, source: value,
        read_default=lambda: _default_compilation_cache_dir(mode),
    )
    resolved_xla_gpu_preallocate = _resolve_kwarg(
        xla_gpu_preallocate,
        parse_explicit=bool,
        env_names=(_GPU_PREALLOCATE_ENV,),
        parse_env=lambda value, source: _parse_bool_value(value, source=source),
        read_default=lambda: _optional_bool_policy_default(
            defaults["xla_gpu_preallocate"]
        ),
    )
    resolved_xla_gpu_mem_fraction = _resolve_kwarg(
        xla_gpu_mem_fraction,
        parse_explicit=lambda value: _validate_mem_fraction_value(
            value,
            source="xla_gpu_mem_fraction",
        ),
        env_names=(_GPU_MEM_FRACTION_ENV,),
        parse_env=lambda value, source: _validate_mem_fraction_value(
            value,
            source=source,
        ),
        read_default=lambda: _optional_float_policy_default(
            defaults["xla_gpu_mem_fraction"]
        ),
    )
    resolved_xla_gpu_allocator = _resolve_kwarg(
        xla_gpu_allocator,
        parse_explicit=lambda value: _validate_gpu_allocator(
            value,
            source="xla_gpu_allocator",
        ),
        env_names=(_GPU_ALLOCATOR_ENV,),
        parse_env=lambda value, source: _validate_gpu_allocator(
            value,
            source=source,
        ),
        read_default=lambda: _validate_gpu_allocator(
            defaults["xla_gpu_allocator"],
            source=f"{mode}.xla_gpu_allocator",
        ),
    )
    resolved_tf_gpu_allocator = _resolve_kwarg(
        tf_gpu_allocator,
        parse_explicit=lambda value: _validate_tf_gpu_allocator(
            value,
            source="tf_gpu_allocator",
        ),
        env_names=(_TF_GPU_ALLOCATOR_OVERRIDE_ENV,),
        parse_env=lambda value, source: _validate_tf_gpu_allocator(
            value,
            source=source,
        ),
        read_default=lambda: _validate_tf_gpu_allocator(
            defaults["tf_gpu_allocator"],
            source=f"{mode}.tf_gpu_allocator",
        ),
    )
    return BackendConfig(
        mode=mode,
        backend=backend,
        jax_platform=jax_platform,
        precision=resolved_precision,
        strict=bool(strict) or debug_overlay,
        debug_nans=resolved_debug_nans,
        disable_jit=resolved_disable_jit,
        transfer_guard=resolved_transfer_guard,
        compilation_cache_dir=resolved_compilation_cache_dir,
        xla_gpu_preallocate=resolved_xla_gpu_preallocate,
        xla_gpu_mem_fraction=resolved_xla_gpu_mem_fraction,
        xla_gpu_allocator=resolved_xla_gpu_allocator,
        tf_gpu_allocator=resolved_tf_gpu_allocator,
    )


def _get_mode_policy_defaults(mode: str) -> dict[str, object]:
    return _MODE_POLICY_DEFAULTS[_validate_mode(mode)]


def _resolve_policy_max_dense_jacobian_bytes(
    config: BackendConfig,
    defaults: dict[str, object],
) -> int | None:
    env_name = (
        _MAX_DENSE_JACOBIAN_BYTES_CPU_ENV
        if config.jax_platform == "cpu"
        else _MAX_DENSE_JACOBIAN_BYTES_GPU_ENV
    )
    env_value = _optional_nonneg_int_env(env_name)
    if env_value is not None:
        return env_value
    default = defaults["max_dense_jacobian_bytes"]
    return None if default is None else int(default)


def _validate_policy_dtype(value: object, *, mode: str, field: str) -> str:
    dtype_name = str(value)
    if dtype_name not in _VALID_POLICY_DTYPES:
        raise ValueError(
            f"Backend mode {mode!r} has unsupported {field}={dtype_name!r}. "
            f"Accepted: {_VALID_POLICY_DTYPES}."
        )
    return dtype_name


def _optional_float_policy_default(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _policy_from_config(config: BackendConfig) -> BackendPolicy:
    defaults = _get_mode_policy_defaults(config.mode)
    resolved_precision = _resolved_precision_for_mode(
        config.mode,
        config.precision,
    )
    compute_dtype = (
        "float32" if resolved_precision in ("fp32_smoke", "mixed") else "float64"
    )
    certificate_dtype: CertificateDType | None = (
        "float64" if resolved_precision == "mixed" else None
    )
    return BackendPolicy(
        mode=config.mode,
        backend=config.backend,
        jax_platform=config.jax_platform,
        strict=config.strict,
        parity_mode=bool(defaults["parity_mode"]),
        requires_x64=bool(defaults["requires_x64"]),
        precision=config.precision,
        resolved_precision=resolved_precision,
        runtime_dtype=_validate_policy_dtype(
            defaults["runtime_dtype"],
            mode=config.mode,
            field="runtime_dtype",
        ),
        host_dtype=_validate_policy_dtype(
            defaults["host_dtype"],
            mode=config.mode,
            field="host_dtype",
        ),
        compute_dtype=compute_dtype,
        certificate_dtype=certificate_dtype,
        default_residency=_validate_default_residency(
            defaults["default_residency"],
            mode=config.mode,
        ),
        default_optimizer_backend=_validate_default_optimizer_backend(
            defaults["default_optimizer_backend"],
            mode=config.mode,
        ),
        supports_host_callback=bool(defaults["supports_host_callback"]),
        supports_buffer_donation=bool(defaults["supports_buffer_donation"]),
        chunk_policy=str(defaults["chunk_policy"]),
        tolerance_tier=str(defaults["tolerance_tier"]),
        compilation_cache_policy=str(defaults["compilation_cache_policy"]),
        matmul_precision=(
            "highest"
            if resolved_precision == "mixed"
            else str(defaults["matmul_precision"])
        ),
        max_dense_jacobian_bytes=_resolve_policy_max_dense_jacobian_bytes(
            config,
            defaults,
        ),
        linear_solve_tolerance_floor=float(defaults["linear_solve_tolerance_floor"]),
        linear_solve_tolerance_cap=_optional_float_policy_default(
            defaults["linear_solve_tolerance_cap"]
        ),
        provenance_label=str(defaults["provenance_label"]),
        xla_gpu_preallocate=config.xla_gpu_preallocate,
        xla_gpu_mem_fraction=config.xla_gpu_mem_fraction,
        xla_gpu_allocator=config.xla_gpu_allocator,
        tf_gpu_allocator=config.tf_gpu_allocator,
        gpu_reduction_order_max_ulp=defaults["gpu_reduction_order_max_ulp"],
        gpu_reduction_order_rel_tol=defaults["gpu_reduction_order_rel_tol"],
        gpu_reproducibility_seed=defaults["gpu_reproducibility_seed"],
        gpu_reproducibility_sample_size=defaults["gpu_reproducibility_sample_size"],
        tolerance_ratchet_factor=defaults["tolerance_ratchet_factor"],
        debug_nans=config.debug_nans,
        disable_jit=config.disable_jit,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
    )


def _runtime_env_value(attribute_name: str, value: object) -> str:
    if value is None:
        return ""
    if attribute_name in {
        "strict",
        "debug_nans",
        "disable_jit",
        "xla_gpu_preallocate",
    }:
        return "1" if bool(value) else "0"
    if attribute_name == "jax_platforms":
        return _runtime_jax_platforms_value(str(value))
    if attribute_name == "jax_platform":
        return _runtime_jax_platform_value(str(value))
    return str(value)


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


def _resolve_legacy_platform(backend: str) -> str:
    raw_value = os.environ.get(_PLATFORM_ENV)
    source = _PLATFORM_ENV
    if raw_value is None:
        raw_value = os.environ.get(_PLATFORM_LEGACY_ENV)
        source = _PLATFORM_LEGACY_ENV
    if raw_value is None:
        raw_value = "cuda" if backend == "jax" else "cpu"
        source = "(default)"
    return _validate_platform(raw_value, source=source)


def _mode_from_legacy_env(backend: str, platform: str) -> BackendMode:
    if backend == "cpu":
        return "native_cpu"
    if platform == "cpu":
        return resolve_jax_execution_profile("cpu").mode
    return resolve_jax_execution_profile("gpu").mode


def is_float32_smoke_policy(policy: BackendPolicy) -> bool:
    """Return True when ``policy`` describes a float32 smoke-tolerance lane."""
    return (
        not policy.requires_x64
        and policy.runtime_dtype == "float32"
        and policy.tolerance_tier == "float32_smoke"
    )
