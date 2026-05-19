"""
Backend selection for the simsopt JAX lane.

This module keeps the legacy environment-variable contract working while adding
an explicit mode-based public API for the new runtime surface:

- ``native_cpu``
- ``jax_cpu_fast``
- ``jax_cpu_parity``
- ``jax_gpu_fast``
- ``jax_gpu_parity``
- ``jax_mps_smoke``

The mode API is the SSOT. The older ``SIMSOPT_BACKEND`` /
``SIMSOPT_JAX_PLATFORM`` pair is still read and written for compatibility.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import importlib.util
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from typing import Callable, Literal
import warnings

_LOGGER = logging.getLogger(__name__)

_VALID_BACKENDS = ("cpu", "jax")
_VALID_PLATFORMS = ("cpu", "cuda", "mps")
_VALID_POLICY_DTYPES = ("float32", "float64")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

# Deprecated selectors removed in favour of the jax-mps Apple Silicon lane.
# jax-metal (last release 2024-10) is unmaintained and incompatible with
# jaxlib 0.10; selecting it must hard-fail with a pointer to jax_mps_smoke.
_DEPRECATED_PLATFORMS = {"metal": "mps"}
_DEPRECATED_MODES = {"jax_metal_smoke": "jax_mps_smoke"}
_MPS_NOT_AVAILABLE_HINT = (
    "jax-mps is not importable. Install with envs/jax-mps.yml (python=3.13) "
    "or pick a cpu/cuda mode."
)
_DEPRECATION_HINT_FMT = (
    "{name}={value!r} was removed (jax-metal is unmaintained upstream and "
    "incompatible with jaxlib 0.10). Use {replacement!r} with envs/jax-mps.yml."
)

_BACKEND_ENV = "SIMSOPT_BACKEND"
_BACKEND_LEGACY_ENV = "STAGE2_BACKEND"
_PLATFORM_ENV = "SIMSOPT_JAX_PLATFORM"
_PLATFORM_LEGACY_ENV = "SIMSOPT_JAX_BACKEND"
_MODE_ENV = "SIMSOPT_BACKEND_MODE"
_STRICT_ENV = "SIMSOPT_BACKEND_STRICT"
_TARGET_LANE_STRICT_ENV = "SIMSOPT_TARGET_LANE_STRICT"
_DEBUG_ENV = "SIMSOPT_DEBUG"
_DEBUG_NANS_ENV = "SIMSOPT_JAX_DEBUG_NANS"
_DISABLE_JIT_ENV = "SIMSOPT_JAX_DISABLE_JIT"
_TRANSFER_GUARD_ENV = "SIMSOPT_JAX_TRANSFER_GUARD"
_COMPILATION_CACHE_DIR_ENV = "SIMSOPT_JAX_COMPILATION_CACHE_DIR"
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
_VALID_SHARDING_STRATEGIES = (
    "none",
    "points",
    "pairwise_rows",
    "hybrid",
    "coil_groups",
    "points_coils",
)
_POINT_AXIS_SHARDING_STRATEGIES = frozenset(("points", "pairwise_rows", "hybrid"))
_POINT_OWNED_SHARDING_STRATEGIES = frozenset(("points", "hybrid", "points_coils"))
_PAIRWISE_ROW_SHARDING_STRATEGIES = frozenset(("pairwise_rows", "hybrid"))
_COIL_AXIS_SHARDING_STRATEGIES = frozenset(("coil_groups", "points_coils"))
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
_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_exclude_nondeterministic_ops",)
_STALE_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_deterministic_ops",)

VALID_BACKEND_MODES = (
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_gpu_fast",
    "jax_gpu_parity",
    "jax_mps_smoke",
)

_MODE_TO_RUNTIME = {
    "native_cpu": ("cpu", "cpu"),
    "jax_cpu_fast": ("jax", "cpu"),
    "jax_cpu_parity": ("jax", "cpu"),
    "jax_gpu_parity": ("jax", "cuda"),
    "jax_gpu_fast": ("jax", "cuda"),
    "jax_mps_smoke": ("jax", "mps"),
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

_MODE_POLICY_DEFAULTS = {
    "native_cpu": {
        "parity_mode": False,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "default_residency": "host",
        "default_optimizer_backend": "scipy",
        "chunk_policy": "host_reference",
        "tolerance_tier": "cpu_reference",
        "compilation_cache_policy": "not_applicable",
        "matmul_precision": "highest",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU,
        "provenance_label": "native_cpu",
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
        "chunk_policy": "performance_tuned",
        "tolerance_tier": "fast",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "default",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU,
        "provenance_label": "jax_cpu_fast",
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
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "highest",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU,
        "provenance_label": "jax_cpu_parity",
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
        "chunk_policy": "stable_default",
        "tolerance_tier": "parity",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "highest",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU,
        "provenance_label": "jax_gpu_parity",
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
        "chunk_policy": "performance_tuned",
        "tolerance_tier": "fast",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "default",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU,
        "provenance_label": "jax_gpu_fast",
        **_GPU_MEMORY_MODE_DEFAULTS,
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
    "jax_mps_smoke": {
        "parity_mode": False,
        "requires_x64": False,
        "runtime_dtype": "float32",
        "host_dtype": "float32",
        "default_residency": "device",
        "default_optimizer_backend": "ondevice",
        "chunk_policy": "stable_default",
        "tolerance_tier": "smoke",
        "compilation_cache_policy": "optional_persistent",
        "matmul_precision": "default",
        "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU,
        "provenance_label": "jax_mps_smoke",
        **_NO_GPU_MEMORY_DEFAULTS,
        **_NO_CI_REPRODUCIBILITY_DEFAULTS,
    },
}

_FIELD_KERNEL_DEFAULTS = {
    "native_cpu": {"coil_chunk_size": 0, "quadrature_block_size": 0},
    "jax_cpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64},
    "jax_cpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0},
    "jax_gpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0},
    "jax_gpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64},
    "jax_mps_smoke": {"coil_chunk_size": 16, "quadrature_block_size": 0},
}
_POINT_CHUNK_SIZE_BY_POLICY = {
    "host_reference": 0,
    "stable_default": 256,
    "performance_tuned": 1024,
}
_PAIRWISE_PENALTY_CHUNK_SIZE_BY_POLICY = dict(_POINT_CHUNK_SIZE_BY_POLICY)
_MODE_SHARDING_DEFAULTS = {
    "native_cpu": "none",
    "jax_cpu_fast": "none",
    "jax_cpu_parity": "none",
    "jax_gpu_parity": "none",
    "jax_gpu_fast": "hybrid",
    "jax_mps_smoke": "none",
}
_DEFAULT_SHARDING_AXIS_NAME = "d"
_DEFAULT_COIL_SHARDING_AXIS_NAME = "coil"
_MIN_POINTS_TO_SHARD_BY_POLICY = {
    "host_reference": 1 << 30,
    "stable_default": 4096,
    "performance_tuned": 2048,
}
_MIN_PAIRWISE_ROWS_TO_SHARD_BY_POLICY = {
    "host_reference": 1 << 30,
    "stable_default": 64,
    "performance_tuned": 32,
}
_MIN_COILS_TO_SHARD_BY_POLICY = {
    "host_reference": 1 << 30,
    "stable_default": 8,
    "performance_tuned": 4,
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
    "point_chunk_size": _POINT_CHUNK_SIZE_ENV,
}
_DEFAULT_TRANSFER_GUARD_BY_MODE = {
    "native_cpu": None,
    "jax_cpu_fast": "log",
    "jax_cpu_parity": "log",
    "jax_gpu_parity": "log",
    "jax_gpu_fast": "log",
    "jax_mps_smoke": "log",
}
_BackendCacheClearCallbackKey = tuple[str, str]
_backend_runtime_lock = threading.RLock()
_target_lane_purity_depth = ContextVar("simsopt_target_lane_purity_depth", default=0)
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
    disable_jit: bool = False
    transfer_guard: str | None = None
    compilation_cache_dir: str | None = None
    xla_gpu_preallocate: bool | None = None
    xla_gpu_mem_fraction: float | None = None
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None


def _xla_flag_value(token: str, flag_name: str) -> bool | None:
    if token == flag_name:
        return True
    if not token.startswith(f"{flag_name}="):
        return None
    _, raw_value = token.split("=", 1)
    return raw_value.strip().lower() in _TRUTHY_ENV_VALUES


def _split_xla_flag_tokens(xla_flags: str | None) -> tuple[str, ...]:
    # External-input parse contract: tokenize XLA_FLAGS env value via shlex.
    # The narrow ValueError catch is a boundary parser (malformed user input
    # returns an empty tuple), not a runtime error swallow.
    if not xla_flags:
        return ()
    try:
        return tuple(shlex.split(xla_flags))
    except ValueError:
        return ()


def _xla_flags_enable_gpu_determinism(xla_flags: str | None) -> bool:
    effective_values: dict[str, bool] = {}
    for token in _split_xla_flag_tokens(xla_flags):
        for flag_name in _GPU_DETERMINISM_XLA_FLAGS:
            resolved = _xla_flag_value(token, flag_name)
            if resolved is None:
                continue
            effective_values[flag_name] = resolved
            break
    return any(effective_values.values())


def _xla_flags_include_stale_gpu_determinism(xla_flags: str | None) -> bool:
    for token in _split_xla_flag_tokens(xla_flags):
        if any(
            token == flag_name or token.startswith(f"{flag_name}=")
            for flag_name in _STALE_GPU_DETERMINISM_XLA_FLAGS
        ):
            return True
    return False


def _stale_cuda_determinism_message() -> str:
    stale_flags = " or ".join(_STALE_GPU_DETERMINISM_XLA_FLAGS)
    return (
        f"{_XLA_FLAGS_ENV} contains stale CUDA determinism flag {stale_flags}. "
        f"Use {_enabled_gpu_determinism_flags_text()} before importing or "
        "touching JAX devices."
    )


def _enabled_gpu_determinism_flags_text() -> str:
    return " or ".join(
        f"{flag_name}=true" for flag_name in _GPU_DETERMINISM_XLA_FLAGS
    )


@dataclass(frozen=True)
class BackendPolicy:
    """Numerical-policy contract for one resolved backend mode.

    The GPU reproducibility fields are reporting/acceptance metadata for parity
    lanes. They document the expected tolerance budget and sampling defaults
    used by CI and diagnostics. For CUDA parity lanes, runtime configuration
    validates the required pre-import XLA determinism flags, but the policy
    fields themselves do not directly force kernel execution behavior.
    """

    mode: str
    backend: str
    jax_platform: str
    strict: bool
    parity_mode: bool
    requires_x64: bool
    runtime_dtype: str
    host_dtype: str
    default_residency: str
    default_optimizer_backend: str
    chunk_policy: str
    tolerance_tier: str
    compilation_cache_policy: str
    matmul_precision: str
    max_dense_jacobian_bytes: int | None
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


@dataclass(frozen=True)
class FieldKernelTuning:
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int
    point_chunk_size: int


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
    point_axis_name: str
    coil_axis_name: str
    mesh_axes: tuple[str, ...]
    point_device_count: int
    coil_device_count: int
    reduced_axis_name: str | None
    min_points_to_shard: int
    min_pairwise_rows_to_shard: int
    min_coils_to_shard: int
    device_count: int
    local_device_count: int
    active: bool
    platform: str
    distributed_enabled: bool
    distributed_initialized: bool


@dataclass(frozen=True)
class DistributedRuntimeConfig:
    enabled: bool
    coordinator_address: str | None
    num_processes: int | None
    process_id: int | None
    local_device_ids: tuple[int, ...] | None
    initialized: bool


def _with_distributed_initialized(
    config: DistributedRuntimeConfig,
    *,
    initialized: bool,
) -> DistributedRuntimeConfig:
    return DistributedRuntimeConfig(
        enabled=config.enabled,
        coordinator_address=config.coordinator_address,
        num_processes=config.num_processes,
        process_id=config.process_id,
        local_device_ids=config.local_device_ids,
        initialized=initialized,
    )


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


def _optional_fraction_env(name: str) -> float | None:
    raw_value = _optional_env_value(name)
    if raw_value is None:
        return None
    value = float(raw_value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name}={raw_value!r} must be in (0, 1]")
    return value


def _validate_gpu_allocator(
    value: str | None,
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
    value: str | None,
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


def target_lane_purity_requested() -> bool:
    """Return whether strict target-lane purity checks are requested."""
    return _env_bool(_TARGET_LANE_STRICT_ENV)


def target_lane_purity_active() -> bool:
    """Return whether the current stack is inside the target-lane guard."""
    return _target_lane_purity_depth.get() > 0


class _StrictTargetLanePurity:
    def __enter__(self):
        self._token = _target_lane_purity_depth.set(_target_lane_purity_depth.get() + 1)
        return self

    def __exit__(self, exc_type, exc, traceback):
        _target_lane_purity_depth.reset(self._token)
        return False


def strict_target_lane_purity():
    """Activate strict legacy-entry blocking for the current context stack."""
    return _StrictTargetLanePurity()


def raise_if_target_lane_bypass(entry: str) -> None:
    """Raise when a guarded target-lane value/grad re-enters legacy code."""
    if target_lane_purity_requested() and target_lane_purity_active():
        raise RuntimeError(f"target-lane bypass: {entry}")


def _validate_backend(value: str, *, source: str) -> str:
    if value not in _VALID_BACKENDS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_BACKENDS}"
        )
    return value


def _raise_deprecated_selector(name: str, value: str, replacement: str) -> None:
    raise ValueError(
        _DEPRECATION_HINT_FMT.format(name=name, value=value, replacement=replacement)
    )


def _validate_platform(value: str, *, source: str) -> str:
    value = value.lower()
    replacement = _DEPRECATED_PLATFORMS.get(value)
    if replacement is not None:
        _raise_deprecated_selector(source, value, replacement)
    if value not in _VALID_PLATFORMS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_PLATFORMS}"
        )
    return value


def _validate_mode(mode: str) -> str:
    replacement = _DEPRECATED_MODES.get(mode)
    if replacement is not None:
        _raise_deprecated_selector("Backend mode", mode, replacement)
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


def _validate_sharding_strategy(value: str, *, source: str) -> str:
    if value not in _VALID_SHARDING_STRATEGIES:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_SHARDING_STRATEGIES}"
        )
    return value


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


def _resolve_coil_sharding_axis_name() -> str:
    raw_value = _optional_nonempty_env(_SHARDING_COIL_AXIS_ENV)
    if raw_value is None:
        return _DEFAULT_COIL_SHARDING_AXIS_NAME
    return raw_value


def _resolve_min_coils_to_shard(policy: BackendPolicy) -> int:
    value = _optional_nonneg_int_env(_MIN_COILS_TO_SHARD_ENV)
    if value is not None:
        return value
    return _MIN_COILS_TO_SHARD_BY_POLICY[policy.chunk_policy]


def _factor_device_count_2d(device_count: int) -> tuple[int, int]:
    """Factor ``device_count`` into ``(point_count, coil_count)`` for a 2D mesh.

    Picks the factor pair closest to a square mesh. ``device_count`` must be
    positive; prime device counts yield ``(1, device_count)`` which still
    constitutes a valid 2D mesh per the JAX shard_map contract.
    """
    if device_count <= 0:
        raise ValueError("points_coils sharding requires device_count > 0.")
    best = (1, device_count)
    best_aspect = float(device_count)
    for point_count in range(1, int(device_count**0.5) + 1):
        if device_count % point_count != 0:
            continue
        coil_count = device_count // point_count
        aspect = max(point_count, coil_count) / min(point_count, coil_count)
        if aspect < best_aspect:
            best = (point_count, coil_count)
            best_aspect = aspect
    return best


def _strategy_device_counts(strategy: str, device_count: int) -> tuple[int, int]:
    if strategy == "none":
        return (0, 0)
    if device_count <= 0:
        raise ValueError(f"{strategy} sharding requires device_count > 0.")
    if strategy == "coil_groups":
        return (1, device_count)
    if strategy in _POINT_AXIS_SHARDING_STRATEGIES:
        return (device_count, 1)
    if strategy == "points_coils":
        return _factor_device_count_2d(device_count)
    raise ValueError(f"unsupported sharding strategy {strategy!r}")


def _strategy_mesh_axis_names(
    strategy: str,
    *,
    point_axis_name: str,
    coil_axis_name: str,
) -> tuple[str, ...]:
    if strategy == "coil_groups":
        return (coil_axis_name,)
    if strategy in _POINT_AXIS_SHARDING_STRATEGIES:
        return (point_axis_name,)
    if strategy == "points_coils":
        return (point_axis_name, coil_axis_name)
    return ()


def _strategy_reduced_axis_name(strategy: str, *, coil_axis_name: str) -> str | None:
    if strategy in _COIL_AXIS_SHARDING_STRATEGIES:
        return coil_axis_name
    return None


def _runtime_jax_platform_value(platform: str) -> str:
    # Identity seam: callers (env lowering, transfer-guard expected-platforms
    # map, set_backend sync) read the JAX_PLATFORMS value through this helper
    # so future platform-specific case mapping has a single edit site.
    return platform


# Mirror of ``repo_bootstrap.with_cpu_callback_lane`` for the single-platform
# config slot. Repeated here because ``repo_bootstrap`` is not importable from
# the installed ``simsopt`` package; keep the value synchronized with that
# helper so the entrypoint and runtime layers agree on JAX_PLATFORMS.
_CUDA_WITH_CPU_FALLBACK_PLATFORMS = "cuda,cpu"


def _runtime_jax_platforms_value(platform: str) -> str:
    if platform == "cuda":
        return _CUDA_WITH_CPU_FALLBACK_PLATFORMS
    return _runtime_jax_platform_value(platform)


def _runtime_jax_backend_name(platform: str) -> str:
    if platform == "cuda":
        return "gpu"
    return _runtime_jax_platform_value(platform)


def _resolve_min_points_to_shard(policy: BackendPolicy) -> int:
    env_value = _optional_nonneg_int_env(_MIN_POINTS_TO_SHARD_ENV)
    if env_value is not None:
        return env_value
    return _MIN_POINTS_TO_SHARD_BY_POLICY.get(policy.chunk_policy, 0)


def _resolve_min_pairwise_rows_to_shard(policy: BackendPolicy) -> int:
    env_value = _optional_nonneg_int_env(_MIN_PAIRWISE_ROWS_TO_SHARD_ENV)
    if env_value is not None:
        return env_value
    return _MIN_PAIRWISE_ROWS_TO_SHARD_BY_POLICY.get(policy.chunk_policy, 0)


def _detect_local_jax_device_count(policy: BackendPolicy) -> int:
    # ImportError boundary: simsopt is importable without JAX installed.
    # Once JAX is present, enumeration errors must surface — the caller
    # has already gated on policy.backend == "jax".
    try:
        import jax
    except ImportError:
        return 0
    backend_name = _runtime_jax_backend_name(policy.jax_platform)
    return len(jax.local_devices(backend=backend_name))


def _detect_global_jax_device_count(policy: BackendPolicy) -> int:
    # Same ImportError boundary as _detect_local_jax_device_count.
    try:
        import jax
    except ImportError:
        return 0
    backend_name = _runtime_jax_backend_name(policy.jax_platform)
    return len(jax.devices(backend=backend_name))


def _visible_cuda_device_selector() -> str | None:
    raw_value = _optional_env_value("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return None
    first = raw_value.split(",", 1)[0].strip()
    if not first or first in {"-1", "none", "NoDevFiles"}:
        return None
    return first


def _parse_visible_cuda_device_index() -> int | None:
    # External-input parse contract: CUDA_VISIBLE_DEVICES env value.
    # The narrow ValueError catch handles non-integer selectors (e.g. UUIDs
    # or whitespace), not runtime errors.
    selector = _visible_cuda_device_selector()
    if selector is None:
        return None
    try:
        value = int(selector)
    except ValueError:
        return None
    return value if value >= 0 else None


def _detect_imported_jax_cuda_device_index() -> int | None:
    jax = sys.modules.get("jax")
    if jax is None:
        return None
    distributed = get_distributed_runtime_config()
    if distributed.enabled:
        distributed_module = getattr(jax, "distributed", None)
        is_initialized = getattr(distributed_module, "is_initialized", None)
        if not callable(is_initialized):
            return None
        if not bool(is_initialized()):
            return None
    local_devices = getattr(jax, "local_devices", None)
    if not callable(local_devices):
        return None
    try:
        devices = local_devices(backend="gpu")
    except RuntimeError:
        # GPU backend not available on this host: detection returns None.
        return None
    if not devices:
        return None
    device = devices[0]
    for attr in ("local_hardware_id", "id"):
        value = getattr(device, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _detect_active_jax_cuda_device_index() -> int | None:
    runtime_index = _detect_imported_jax_cuda_device_index()
    if runtime_index is not None:
        return runtime_index
    return _parse_visible_cuda_device_index()


def _detect_active_jax_cuda_device_selector() -> int | str | None:
    runtime_index = _detect_imported_jax_cuda_device_index()
    if runtime_index is not None:
        return runtime_index
    return _visible_cuda_device_selector()


def _parse_nvidia_smi_indexed_value_row(raw_row: str) -> tuple[int, float] | None:
    # External-input parse contract: a single nvidia-smi CSV row. The
    # narrow ValueError catch handles malformed rows from the external
    # tool, not runtime errors in this process.
    fields = [field.strip() for field in raw_row.split(",")]
    if len(fields) != 2:
        return None
    try:
        return int(float(fields[0])), float(fields[1])
    except ValueError:
        return None


def _query_gpu_metric_mb_from_nvidia_smi(
    metric_name: str,
    device_selector: int | str | None = None,
) -> float | None:
    command = [
        "nvidia-smi",
        f"--query-gpu=index,{metric_name}",
        "--format=csv,noheader,nounits",
    ]
    if device_selector is not None:
        command.extend(["-i", str(device_selector)])
    # External-tool availability boundary: nvidia-smi may be absent from
    # PATH (FileNotFoundError) or exit non-zero on hosts without an
    # NVIDIA driver (CalledProcessError). Both are expected absence
    # signals, not runtime errors; return None so the caller can continue
    # the ordered GPU-detection chain.
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
        parsed = _parse_nvidia_smi_indexed_value_row(line)
        if parsed is None:
            continue
        index, value = parsed
        if isinstance(device_selector, int) and index != device_selector:
            continue
        if value >= 0:
            return float(value)
    return None


def _query_gpu_total_memory_mb_from_nvidia_smi(
    device_selector: int | str | None = None,
) -> int | None:
    value = _query_gpu_metric_mb_from_nvidia_smi("memory.total", device_selector)
    if value is None or value <= 0:
        return None
    return int(value)


def _resolve_gpu_total_memory_mb(
    policy: BackendPolicy,
) -> tuple[int | None, str | None]:
    if policy.jax_platform != "cuda":
        return None, None
    env_value = _optional_nonneg_int_env(_GPU_MEMORY_TOTAL_MB_ENV)
    if env_value is not None:
        if env_value == 0:
            raise ValueError(f"{_GPU_MEMORY_TOTAL_MB_ENV} must be > 0 when set")
        return env_value, _GPU_MEMORY_TOTAL_MB_ENV
    device_selector = _detect_active_jax_cuda_device_selector()
    detected = _query_gpu_total_memory_mb_from_nvidia_smi(device_selector)
    if detected is None:
        return None, None
    if device_selector is None:
        return detected, "nvidia-smi"
    return detected, f"nvidia-smi[{device_selector}]"


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
        "quadrature_block_size": _FIELD_KERNEL_DEFAULTS[mode]["quadrature_block_size"],
        "point_chunk_size": _point_chunk_size_default(chunk_policy),
        "pairwise_penalty_chunk_size": _pairwise_penalty_chunk_size_default(
            chunk_policy
        ),
    }


def _apply_chunk_env_overrides(chunk_sizes: dict[str, int]) -> dict[str, int]:
    resolved = dict(chunk_sizes)
    for key, env_name in _FIELD_KERNEL_ENV_BY_KEY.items():
        value = _optional_nonneg_int_env(env_name)
        if value is not None:
            resolved[key] = value
    pairwise_value = _optional_nonneg_int_env(_PAIRWISE_PENALTY_CHUNK_SIZE_ENV)
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
    distributed = get_distributed_runtime_config()
    if policy.backend != "jax":
        strategy = "none"
        local_device_count = 0
        device_count = 0
    else:
        local_device_count = _detect_local_jax_device_count(policy)
        device_count = (
            _detect_global_jax_device_count(policy)
            if distributed.initialized
            else local_device_count
        )
    if (
        mode == "jax_gpu_parity"
        and strategy == "none"
        and local_device_count > 1
        and (mode, local_device_count) not in _logged_sharding_notices
    ):
        _logged_sharding_notices.add((mode, local_device_count))
        _LOGGER.info(
            "jax_gpu_parity defaults to single-device execution on %s local "
            "devices; set SIMSOPT_JAX_SHARDING=hybrid to opt in before the "
            "round-3 multi-GPU parity proof is available.",
            local_device_count,
        )
    point_axis_name = _resolve_sharding_axis_name()
    coil_axis_name = _resolve_coil_sharding_axis_name()
    point_device_count, coil_device_count = _strategy_device_counts(
        strategy,
        device_count,
    )
    return ShardingTuning(
        mode=mode,
        strategy=strategy,
        mesh_axis_name=point_axis_name,
        point_axis_name=point_axis_name,
        coil_axis_name=coil_axis_name,
        mesh_axes=_strategy_mesh_axis_names(
            strategy,
            point_axis_name=point_axis_name,
            coil_axis_name=coil_axis_name,
        ),
        point_device_count=point_device_count,
        coil_device_count=coil_device_count,
        reduced_axis_name=_strategy_reduced_axis_name(
            strategy,
            coil_axis_name=coil_axis_name,
        ),
        min_points_to_shard=_resolve_min_points_to_shard(policy),
        min_pairwise_rows_to_shard=_resolve_min_pairwise_rows_to_shard(policy),
        min_coils_to_shard=_resolve_min_coils_to_shard(policy),
        device_count=device_count,
        local_device_count=local_device_count,
        active=strategy != "none" and device_count > 1,
        platform=policy.jax_platform,
        distributed_enabled=distributed.enabled,
        distributed_initialized=distributed.initialized,
    )


def _resolve_debug_nans(debug_nans: bool | None) -> bool:
    if debug_nans is None:
        return (
            _env_bool(_DEBUG_NANS_ENV)
            if _optional_env_value(_DEBUG_NANS_ENV)
            else False
        )
    return bool(debug_nans)


def _debug_overlay_enabled() -> bool:
    return bool(_optional_bool_env(_DEBUG_ENV))


def _resolve_disable_jit(disable_jit: bool | None) -> bool:
    if disable_jit is None:
        env_value = _optional_bool_env(_DISABLE_JIT_ENV)
        return False if env_value is None else env_value
    return bool(disable_jit)


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


def _resolve_xla_gpu_preallocate(
    mode: str,
    xla_gpu_preallocate: bool | None,
) -> bool | None:
    if xla_gpu_preallocate is not None:
        return bool(xla_gpu_preallocate)
    env_value = _optional_bool_env(_GPU_PREALLOCATE_ENV)
    if env_value is not None:
        return env_value
    defaults = _get_mode_policy_defaults(mode)
    default = defaults["xla_gpu_preallocate"]
    return None if default is None else bool(default)


def _validate_mem_fraction_value(value: float, *, source: str) -> float:
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{source}={value!r} must be in (0, 1]")
    return fraction


def _resolve_xla_gpu_mem_fraction(
    mode: str,
    xla_gpu_mem_fraction: float | None,
) -> float | None:
    if xla_gpu_mem_fraction is not None:
        return _validate_mem_fraction_value(
            xla_gpu_mem_fraction,
            source="xla_gpu_mem_fraction",
        )
    env_value = _optional_fraction_env(_GPU_MEM_FRACTION_ENV)
    if env_value is not None:
        return env_value
    defaults = _get_mode_policy_defaults(mode)
    default = defaults["xla_gpu_mem_fraction"]
    return None if default is None else float(default)


def _resolve_xla_gpu_allocator(
    mode: str,
    xla_gpu_allocator: Literal["platform", "vmm"] | None,
) -> Literal["platform", "vmm"] | None:
    if xla_gpu_allocator is not None:
        return _validate_gpu_allocator(
            xla_gpu_allocator,
            source="xla_gpu_allocator",
        )
    env_value = _optional_env_value(_GPU_ALLOCATOR_ENV)
    if env_value is not None:
        return _validate_gpu_allocator(env_value, source=_GPU_ALLOCATOR_ENV)
    defaults = _get_mode_policy_defaults(mode)
    return _validate_gpu_allocator(
        defaults["xla_gpu_allocator"],
        source=f"{mode}.xla_gpu_allocator",
    )


def _resolve_tf_gpu_allocator(
    mode: str,
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None,
) -> Literal["cuda_malloc_async"] | None:
    if tf_gpu_allocator is not None:
        return _validate_tf_gpu_allocator(
            tf_gpu_allocator,
            source="tf_gpu_allocator",
        )
    env_value = _optional_env_value(_TF_GPU_ALLOCATOR_OVERRIDE_ENV)
    if env_value is not None:
        return _validate_tf_gpu_allocator(
            env_value,
            source=_TF_GPU_ALLOCATOR_OVERRIDE_ENV,
        )
    defaults = _get_mode_policy_defaults(mode)
    return _validate_tf_gpu_allocator(
        defaults["tf_gpu_allocator"],
        source=f"{mode}.tf_gpu_allocator",
    )


def _config_from_mode(
    mode: str,
    *,
    strict: bool,
    debug_nans: bool | None = None,
    disable_jit: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
    xla_gpu_preallocate: bool | None = None,
    xla_gpu_mem_fraction: float | None = None,
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None,
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None,
) -> BackendConfig:
    mode = _validate_mode(mode)
    backend, jax_platform = _MODE_TO_RUNTIME[mode]
    debug_overlay = _debug_overlay_enabled()
    return BackendConfig(
        mode=mode,
        backend=backend,
        jax_platform=jax_platform,
        strict=bool(strict) or debug_overlay,
        debug_nans=True if debug_overlay else _resolve_debug_nans(debug_nans),
        disable_jit=True if debug_overlay else _resolve_disable_jit(disable_jit),
        transfer_guard=(
            "disallow"
            if debug_overlay
            else _resolve_transfer_guard(mode, transfer_guard)
        ),
        compilation_cache_dir=_resolve_compilation_cache_dir(
            mode,
            compilation_cache_dir,
        ),
        xla_gpu_preallocate=_resolve_xla_gpu_preallocate(
            mode,
            xla_gpu_preallocate,
        ),
        xla_gpu_mem_fraction=_resolve_xla_gpu_mem_fraction(
            mode,
            xla_gpu_mem_fraction,
        ),
        xla_gpu_allocator=_resolve_xla_gpu_allocator(mode, xla_gpu_allocator),
        tf_gpu_allocator=_resolve_tf_gpu_allocator(mode, tf_gpu_allocator),
    )


def _resolve_mode(mode: str | None = None) -> str:
    if mode is None:
        return get_backend_mode()
    return _validate_mode(mode)


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


def _policy_from_config(config: BackendConfig) -> BackendPolicy:
    defaults = _get_mode_policy_defaults(config.mode)
    return BackendPolicy(
        mode=config.mode,
        backend=config.backend,
        jax_platform=config.jax_platform,
        strict=config.strict,
        parity_mode=bool(defaults["parity_mode"]),
        requires_x64=bool(defaults["requires_x64"]),
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
        default_residency=_validate_default_residency(
            defaults["default_residency"],
            mode=config.mode,
        ),
        default_optimizer_backend=_validate_default_optimizer_backend(
            defaults["default_optimizer_backend"],
            mode=config.mode,
        ),
        chunk_policy=str(defaults["chunk_policy"]),
        tolerance_tier=str(defaults["tolerance_tier"]),
        compilation_cache_policy=str(defaults["compilation_cache_policy"]),
        matmul_precision=str(defaults["matmul_precision"]),
        max_dense_jacobian_bytes=_resolve_policy_max_dense_jacobian_bytes(
            config,
            defaults,
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


_cached_backend_policy: BackendPolicy | None = None
_warned_jax_fallbacks: set[tuple[str, str, str]] = set()
_logged_sharding_notices: set[tuple[str, int]] = set()


def get_backend_policy(mode: str | None = None) -> BackendPolicy:
    """Return the numerical-policy contract for a backend mode."""
    global _cached_backend_policy
    with _backend_runtime_lock:
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


def _mode_from_legacy_env(backend: str, platform: str) -> str:
    if backend == "cpu":
        return "native_cpu"
    if platform == "cpu":
        return "jax_cpu_parity"
    if platform == "mps":
        return "jax_mps_smoke"
    return "jax_gpu_parity"


_cached_backend_config: BackendConfig | None = None


def get_backend_config() -> BackendConfig:
    """Return the resolved backend configuration.

    The result is cached after first resolution. Call
    ``invalidate_backend_cache()`` or ``set_backend()`` to clear.
    """
    global _cached_backend_config
    with _backend_runtime_lock:
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
            platform = _resolve_legacy_platform(backend)
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
_cached_distributed_runtime_config: DistributedRuntimeConfig | None = None


def _jax_distributed_runtime_is_initialized() -> bool:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return False
    distributed_module = getattr(jax_module, "distributed", None)
    is_initialized = getattr(distributed_module, "is_initialized", None)
    if not callable(is_initialized):
        return False
    return bool(is_initialized())


def _invalidate_distributed_tuning_caches() -> None:
    global _cached_chunk_tuning, _cached_field_kernel_tuning, _cached_sharding_tuning
    with _backend_runtime_lock:
        _cached_chunk_tuning = None
        _cached_field_kernel_tuning = None
        _cached_sharding_tuning = None


def _cache_distributed_initialized_config(
    config: DistributedRuntimeConfig,
) -> DistributedRuntimeConfig:
    global _cached_distributed_runtime_config
    initialized_config = _with_distributed_initialized(config, initialized=True)
    with _backend_runtime_lock:
        _cached_distributed_runtime_config = initialized_config
        _invalidate_distributed_tuning_caches()
        return initialized_config


def _resolve_distributed_runtime_config(
    config: DistributedRuntimeConfig,
) -> DistributedRuntimeConfig:
    if (
        config.enabled
        and not config.initialized
        and _jax_distributed_runtime_is_initialized()
    ):
        return _cache_distributed_initialized_config(config)
    return config


def get_chunk_tuning(mode: str | None = None) -> ChunkTuning:
    """Return the resolved chunk sizes and autotuning metadata."""
    global _cached_chunk_tuning
    with _backend_runtime_lock:
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
    with _backend_runtime_lock:
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
    with _backend_runtime_lock:
        if mode is None and _cached_field_kernel_tuning is not None:
            return _cached_field_kernel_tuning
        chunk_tuning = get_chunk_tuning(mode)
        tuning = FieldKernelTuning(
            mode=chunk_tuning.mode,
            chunk_policy=chunk_tuning.chunk_policy,
            coil_chunk_size=chunk_tuning.coil_chunk_size,
            quadrature_block_size=chunk_tuning.quadrature_block_size,
            point_chunk_size=chunk_tuning.point_chunk_size,
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


def get_active_cuda_device_index(mode: str | None = None) -> int | None:
    """Return the active CUDA device index implied by env or JAX runtime state."""
    policy = get_backend_policy(mode)
    if policy.jax_platform != "cuda":
        return None
    return _detect_active_jax_cuda_device_index()


def query_active_gpu_memory_mb(mode: str | None = None) -> float | None:
    """Return coarse memory usage for the active CUDA device when available."""
    policy = get_backend_policy(mode)
    if policy.jax_platform != "cuda":
        return None
    device_selector = _detect_active_jax_cuda_device_selector()
    return _query_gpu_metric_mb_from_nvidia_smi("memory.used", device_selector)


def get_sharding_strategy(mode: str | None = None) -> str:
    """Return the resolved sharding strategy label for the mode."""
    return get_sharding_tuning(mode).strategy


def should_shard_points(mode: str | None = None) -> bool:
    """Return ``True`` when point-axis sharding is active for the mode."""
    tuning = get_sharding_tuning(mode)
    return tuning.active and tuning.strategy in _POINT_OWNED_SHARDING_STRATEGIES


def should_shard_pairwise_rows(mode: str | None = None) -> bool:
    """Return ``True`` when row-owned pairwise sharding is active for the mode."""
    tuning = get_sharding_tuning(mode)
    return tuning.active and tuning.strategy in _PAIRWISE_ROW_SHARDING_STRATEGIES


def should_shard_coil_groups(mode: str | None = None) -> bool:
    """Return ``True`` when the coil axis is sharded by the active strategy.

    Includes both the 1D ``coil_groups`` mesh and the 2D ``points_coils``
    mesh; the predicate signals "coil axis collective is active," not
    "1D-only mesh." Callers that need the 1D variant specifically should
    compare ``get_sharding_strategy(mode) == 'coil_groups'`` directly.
    """
    tuning = get_sharding_tuning(mode)
    return tuning.active and tuning.strategy in _COIL_AXIS_SHARDING_STRATEGIES


def get_debug_nans(mode: str | None = None) -> bool:
    """Return the debug-NaN runtime guardrail state for the resolved mode."""
    return get_backend_policy(mode).debug_nans


def get_disable_jit(mode: str | None = None) -> bool:
    """Return the active JAX disable-JIT debug policy for the resolved mode."""
    return get_backend_policy(mode).disable_jit


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
    with _backend_runtime_lock:
        _backend_cache_clear_callbacks[_backend_cache_clear_callback_key(callback)] = (
            callback
        )


def _run_backend_cache_clear_callbacks() -> None:
    with _backend_runtime_lock:
        callbacks = tuple(_backend_cache_clear_callbacks.values())
    for callback in callbacks:
        callback()


def _reset_backend_runtime_caches() -> None:
    global _cached_backend_policy, _cached_distributed_runtime_config
    with _backend_runtime_lock:
        _cached_backend_policy = None
        _invalidate_distributed_tuning_caches()
        _cached_distributed_runtime_config = None
        _warned_jax_fallbacks.clear()
    _run_backend_cache_clear_callbacks()


def _parse_local_device_ids(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value in (None, ""):
        return None
    values = []
    for field in raw_value.split(","):
        stripped = field.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 0:
            raise ValueError(
                f"{_DISTRIBUTED_LOCAL_DEVICE_IDS_ENV} entries must be >= 0."
            )
        values.append(value)
    return tuple(values) if values else None


def _build_distributed_runtime_config() -> DistributedRuntimeConfig:
    enabled = _env_bool(_DISTRIBUTED_INIT_ENV)
    coordinator_address = _optional_nonempty_env(_DISTRIBUTED_COORDINATOR_ADDRESS_ENV)
    num_processes = _optional_nonneg_int_env(_DISTRIBUTED_NUM_PROCESSES_ENV)
    process_id = _optional_nonneg_int_env(_DISTRIBUTED_PROCESS_ID_ENV)
    local_device_ids = _parse_local_device_ids(
        _optional_nonempty_env(_DISTRIBUTED_LOCAL_DEVICE_IDS_ENV)
    )
    if not enabled:
        return DistributedRuntimeConfig(
            enabled=False,
            coordinator_address=None,
            num_processes=None,
            process_id=None,
            local_device_ids=None,
            initialized=False,
        )

    missing = [
        name
        for name, value in (
            (_DISTRIBUTED_COORDINATOR_ADDRESS_ENV, coordinator_address),
            (_DISTRIBUTED_NUM_PROCESSES_ENV, num_processes),
            (_DISTRIBUTED_PROCESS_ID_ENV, process_id),
        )
        if value is None
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            "Distributed JAX bootstrap requires the following env vars when "
            f"{_DISTRIBUTED_INIT_ENV}=1: {missing_list}."
        )
    if int(num_processes) <= 0:
        raise ValueError(
            f"{_DISTRIBUTED_NUM_PROCESSES_ENV} must be > 0 when "
            f"{_DISTRIBUTED_INIT_ENV}=1."
        )
    if int(process_id) >= int(num_processes):
        raise ValueError(
            f"{_DISTRIBUTED_PROCESS_ID_ENV}={process_id} must be smaller than "
            f"{_DISTRIBUTED_NUM_PROCESSES_ENV}={num_processes}."
        )
    return DistributedRuntimeConfig(
        enabled=True,
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
        initialized=False,
    )


def get_distributed_runtime_config() -> DistributedRuntimeConfig:
    """Return the configured distributed-JAX bootstrap contract."""
    global _cached_distributed_runtime_config
    with _backend_runtime_lock:
        if _cached_distributed_runtime_config is None:
            _cached_distributed_runtime_config = _build_distributed_runtime_config()
        _cached_distributed_runtime_config = _resolve_distributed_runtime_config(
            _cached_distributed_runtime_config
        )
        return _cached_distributed_runtime_config


def maybe_initialize_distributed_jax() -> DistributedRuntimeConfig:
    """Initialize multi-host JAX when explicitly configured through env vars."""
    config = get_distributed_runtime_config()
    if not config.enabled:
        return config

    import jax

    distributed_module = getattr(jax, "distributed", None)
    if distributed_module is None:
        raise RuntimeError("Installed JAX runtime does not expose jax.distributed.")
    is_initialized = getattr(distributed_module, "is_initialized", None)
    if callable(is_initialized) and bool(is_initialized()):
        return _cache_distributed_initialized_config(config)

    initialize = getattr(distributed_module, "initialize", None)
    if initialize is None:
        raise RuntimeError(
            "Installed JAX runtime does not expose jax.distributed.initialize."
        )
    initialize(
        coordinator_address=config.coordinator_address,
        num_processes=int(config.num_processes),
        process_id=int(config.process_id),
        local_device_ids=(
            None if config.local_device_ids is None else list(config.local_device_ids)
        ),
    )
    return _cache_distributed_initialized_config(config)


def invalidate_backend_cache() -> None:
    """Clear the cached backend configuration and derived caches.

    Call this after mutating ``SIMSOPT_*`` environment variables directly
    (outside of ``set_backend()``) so the next ``get_backend_config()`` call
    re-reads the environment.  Test fixtures should call this when they
    manipulate env vars via ``monkeypatch`` or context managers.
    """
    global _cached_backend_config
    with _backend_runtime_lock:
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
    with _backend_runtime_lock:
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


def _env_platforms_request_cuda() -> bool:
    platforms = _optional_env_value(_JAX_PLATFORMS_ENV)
    if platforms is None:
        return False
    return any(part.strip().lower() == "cuda" for part in platforms.split(","))


def _imported_jax_reports_cuda() -> bool:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return False
    default_backend = getattr(jax_module, "default_backend", None)
    if not callable(default_backend):
        return False
    active_backend = str(default_backend())
    return active_backend in _expected_runtime_backend_names("cuda")


def _raise_or_warn_runtime_issue(config: BackendConfig, message: str) -> None:
    if config.mode == "jax_gpu_parity" or config.strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def validate_cuda_determinism_environment() -> None:
    """Validate direct CUDA platform selection before JAX initializes.

    This covers users who set ``JAX_PLATFORMS=cuda`` directly instead of going
    through ``SIMSOPT_BACKEND_MODE=jax_gpu_*``. Runtime mode configuration has a
    stricter policy-aware check in ``apply_jax_runtime_config``.
    """
    if not (_env_platforms_request_cuda() or _imported_jax_reports_cuda()):
        return
    xla_flags = os.environ.get(_XLA_FLAGS_ENV)
    if _xla_flags_include_stale_gpu_determinism(xla_flags):
        message = _stale_cuda_determinism_message()
        config = get_backend_config()
        _raise_or_warn_runtime_issue(config, message)
        return
    if _xla_flags_enable_gpu_determinism(xla_flags):
        return
    message = (
        f"{_JAX_PLATFORMS_ENV}=cuda selects CUDA execution, but {_XLA_FLAGS_ENV} "
        f"does not enable {_enabled_gpu_determinism_flags_text()}. Set "
        f"{_XLA_FLAGS_ENV} before "
        "importing or touching JAX devices, because changing XLA flags after "
        "JAX backend initialization has no effect."
    )
    config = get_backend_config()
    _raise_or_warn_runtime_issue(config, message)


def _expected_runtime_backend_names(jax_platform: str) -> frozenset[str]:
    if jax_platform == "cuda":
        return frozenset({"cuda", "gpu"})
    return frozenset({jax_platform})


def _validate_initialized_jax_runtime(jax_module, config: BackendConfig) -> None:
    default_backend = getattr(jax_module, "default_backend", None)
    if not callable(default_backend):
        return
    active_backend = str(default_backend())
    expected_backends = _expected_runtime_backend_names(config.jax_platform)
    if active_backend in expected_backends:
        return
    message = (
        f"Requested JAX platform {config.jax_platform!r} for backend mode "
        f"{config.mode!r}, but the active JAX default backend is "
        f"{active_backend!r}. Set backend environment variables before "
        "importing or touching JAX devices."
    )
    _raise_or_warn_runtime_issue(config, message)


def _validate_cuda_parity_determinism_env(
    config: BackendConfig,
    policy: BackendPolicy,
) -> None:
    if config.jax_platform != "cuda":
        return
    xla_flags = os.environ.get(_XLA_FLAGS_ENV)
    if _xla_flags_include_stale_gpu_determinism(xla_flags):
        message = _stale_cuda_determinism_message()
        _raise_or_warn_runtime_issue(config, message)
        return
    if _xla_flags_enable_gpu_determinism(xla_flags):
        return
    message = (
        f"Backend mode {config.mode!r} selects CUDA execution, but "
        f"{_XLA_FLAGS_ENV} does not enable "
        f"{_enabled_gpu_determinism_flags_text()}. Set {_XLA_FLAGS_ENV} before "
        "importing or touching JAX devices, because changing XLA flags after JAX "
        "backend initialization has no effect."
    )
    _raise_or_warn_runtime_issue(config, message)


def _set_runtime_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
        return
    os.environ[name] = value


def _gpu_memory_runtime_env(config: BackendConfig) -> tuple[tuple[str, str | None], ...]:
    if config.xla_gpu_preallocate is None:
        preallocate = None
    else:
        preallocate = "true" if config.xla_gpu_preallocate else "false"

    if config.xla_gpu_allocator == "vmm":
        python_mem_fraction = None
        client_mem_fraction = (
            None
            if config.xla_gpu_mem_fraction is None
            else str(config.xla_gpu_mem_fraction)
        )
    else:
        python_mem_fraction = (
            None
            if config.xla_gpu_mem_fraction is None
            else str(config.xla_gpu_mem_fraction)
        )
        client_mem_fraction = None

    return (
        (_XLA_PYTHON_CLIENT_PREALLOCATE_ENV, preallocate),
        (_XLA_PYTHON_CLIENT_ALLOCATOR_ENV, config.xla_gpu_allocator),
        (_XLA_PYTHON_CLIENT_MEM_FRACTION_ENV, python_mem_fraction),
        (_XLA_CLIENT_MEM_FRACTION_ENV, client_mem_fraction),
        (_TF_GPU_ALLOCATOR_ENV, config.tf_gpu_allocator),
    )


def _gpu_memory_runtime_env_matches(config: BackendConfig) -> bool:
    for name, expected in _gpu_memory_runtime_env(config):
        actual = os.environ.get(name)
        if expected is None:
            if actual is not None:
                return False
            continue
        if actual != expected:
            return False
    return True


def _assert_jax_not_imported_for_gpu_memory_config(config: BackendConfig) -> None:
    jax_module = sys.modules.get("jax")
    if (
        config.jax_platform != "cuda"
        or jax_module is None
        or getattr(jax_module, "__name__", None) != "jax"
        or "jax._src" not in sys.modules
    ):
        return
    if _gpu_memory_runtime_env_matches(config):
        return
    raise RuntimeError(
        "JAX GPU memory environment variables must be resolved before "
        "importing jax. Call simsopt.config.set_backend(...) or set "
        "SIMSOPT_BACKEND_MODE before importing or touching JAX devices."
    )


def _apply_jax_gpu_memory_env(config: BackendConfig) -> None:
    if config.jax_platform != "cuda":
        return
    _assert_jax_not_imported_for_gpu_memory_config(config)
    for name, value in _gpu_memory_runtime_env(config):
        _set_runtime_env(name, value)


def _probe_mps_plugin() -> None:
    """Fail closed when the jax-mps PJRT plugin is not importable.

    Uses ``find_spec`` rather than ``import`` so the plugin registers normally
    during the subsequent ``import jax`` call with ``JAX_PLATFORMS=mps`` set.
    """
    if importlib.util.find_spec("jax_plugins.mps") is None:
        raise RuntimeError(_MPS_NOT_AVAILABLE_HINT)


def apply_jax_runtime_config() -> None:
    """Apply the resolved JAX runtime settings to the active process."""
    config = get_backend_config()
    if config.backend != "jax":
        return
    policy = get_backend_policy(config.mode)
    _validate_cuda_parity_determinism_env(config, policy)
    _apply_jax_gpu_memory_env(config)
    if config.jax_platform == "mps":
        _probe_mps_plugin()

    import jax

    jax.config.update(
        "jax_platforms",
        _runtime_jax_platforms_value(config.jax_platform),
    )
    jax.config.update("jax_enable_x64", policy.requires_x64)
    jax.config.update("jax_default_matmul_precision", policy.matmul_precision)
    jax.config.update("jax_debug_nans", config.debug_nans)
    jax.config.update("jax_disable_jit", config.disable_jit)
    if config.transfer_guard is not None:
        jax.config.update("jax_transfer_guard", config.transfer_guard)
    if config.compilation_cache_dir is not None:
        jax.config.update("jax_compilation_cache_dir", config.compilation_cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    _validate_initialized_jax_runtime(jax, config)


class _CpuDeviceConstructionContext:
    def __init__(self):
        self._context = None

    def __enter__(self):
        import jax

        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("JAX did not report an addressable CPU device.")
        cpu_device = cpu_devices[0]
        self._context = jax.default_device(cpu_device)
        self._context.__enter__()
        return cpu_device

    def __exit__(self, exc_type, exc, traceback):
        if self._context is None:
            return False
        return self._context.__exit__(exc_type, exc, traceback)


def with_cpu_device_for_construction():
    """Return a context manager that defaults fresh JAX arrays to CPU."""
    return _CpuDeviceConstructionContext()


def set_backend(
    mode: str,
    *,
    strict: bool = False,
    debug_nans: bool | None = None,
    disable_jit: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
    xla_gpu_preallocate: bool | None = None,
    xla_gpu_mem_fraction: float | None = None,
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None,
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None,
    configure_runtime: bool = True,
) -> BackendConfig:
    """Set the active backend mode for the current process.

    This keeps the legacy env vars in sync so existing scripts and subprocess
    helpers continue to work unchanged. GPU memory keywords resolve the
    pre-import JAX/XLA allocator env vars explicitly; env overrides still sit
    between mode defaults and these arguments. Also updates the config cache so
    subsequent ``get_backend_config()`` calls are free.
    """
    global _cached_backend_config
    config = _config_from_mode(
        mode,
        strict=bool(strict),
        debug_nans=debug_nans,
        disable_jit=disable_jit,
        transfer_guard=transfer_guard,
        compilation_cache_dir=compilation_cache_dir,
        xla_gpu_preallocate=xla_gpu_preallocate,
        xla_gpu_mem_fraction=xla_gpu_mem_fraction,
        xla_gpu_allocator=xla_gpu_allocator,
        tf_gpu_allocator=tf_gpu_allocator,
    )
    with _backend_runtime_lock:
        _cached_backend_config = config
        _reset_backend_runtime_caches()
        for env_name, attribute_name in _SYNCED_RUNTIME_ENV_VALUES:
            config_attribute_name = (
                "jax_platform" if attribute_name == "jax_platforms" else attribute_name
            )
            os.environ[env_name] = _runtime_env_value(
                attribute_name,
                getattr(config, config_attribute_name),
            )
    if configure_runtime:
        apply_jax_runtime_config()
    return config


def use_runtime(
    mode: str,
    *,
    debug: bool = False,
    strict: bool = False,
    debug_nans: bool | None = None,
    disable_jit: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
    xla_gpu_preallocate: bool | None = None,
    xla_gpu_mem_fraction: float | None = None,
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None,
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None,
    configure_runtime: bool = True,
) -> BackendConfig:
    """Set runtime mode with the strict debug overlay when requested."""
    return set_backend(
        mode,
        strict=bool(strict) or bool(debug),
        debug_nans=True if debug else debug_nans,
        disable_jit=True if debug else disable_jit,
        transfer_guard="disallow" if debug else transfer_guard,
        compilation_cache_dir=compilation_cache_dir,
        xla_gpu_preallocate=xla_gpu_preallocate,
        xla_gpu_mem_fraction=xla_gpu_mem_fraction,
        xla_gpu_allocator=xla_gpu_allocator,
        tf_gpu_allocator=tf_gpu_allocator,
        configure_runtime=configure_runtime,
    )
