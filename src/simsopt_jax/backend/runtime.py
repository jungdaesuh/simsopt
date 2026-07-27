"""
Backend selection for the simsopt JAX lane.

This module keeps the legacy environment-variable contract working while adding
an explicit mode-based public API for the new runtime surface:

- ``native_cpu``
- ``jax_cpu_fast``
- ``jax_cpu_parity``
- ``jax_cpu_float32_smoke``
- ``jax_gpu_fast``
- ``jax_gpu_parity``

The mode API is the SSOT. The older ``SIMSOPT_BACKEND`` /
``SIMSOPT_JAX_PLATFORM`` pair is still read and written for compatibility.

Internal knowledge is split along stable boundaries:

- :mod:`simsopt_jax.backend._runtime_policy` — mode/precision validation and
  config/policy resolution
- :mod:`simsopt_jax.backend._runtime_tuning` — chunk/sharding/field-kernel and
  distributed topology builders
- this module — process-global lifecycle, configure-before-JAX, and the public
  facade re-exports
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
import threading
import warnings
from contextvars import ContextVar
from typing import Callable, Literal, overload

# Public + private re-exports: callers and tests keep importing from this facade.
from simsopt_jax.backend._runtime_policy import (  # noqa: F401
    _BACKEND_ENV,
    _BACKEND_LEGACY_ENV,
    _COMPILATION_CACHE_DIR_ENV,
    _CUDA_WITH_CPU_FALLBACK_PLATFORMS,
    _DEBUG_ENV,
    _DEBUG_NANS_ENV,
    _DEFAULT_TRANSFER_GUARD_BY_MODE,
    _DISABLE_JIT_ENV,
    _EXPLICIT_SELECTOR_ENV_VARS,
    _GUARDRAIL_ENV_VARS,
    _JAX_COMPILATION_CACHE_DIR_ENV,
    _JAX_PLATFORMS_ENV,
    _MODE_ENV,
    _MODE_POLICY_DEFAULTS,
    _MODE_TO_RUNTIME,
    _OBSOLETE_MIXED_PRECISION_ENV,
    _PLATFORM_ENV,
    _PLATFORM_LEGACY_ENV,
    _PRECISION_ENV,
    _STRICT_ENV,
    _SYNCED_RUNTIME_ENV_VALUES,
    _TARGET_LANE_STRICT_ENV,
    _TF_GPU_ALLOCATOR_ENV,
    _TRANSFER_GUARD_ENV,
    _TRUTHY_ENV_VALUES,
    _VALID_BACKENDS,
    _VALID_DEFAULT_OPTIMIZER_BACKENDS,
    _VALID_DEFAULT_RESIDENCIES,
    _VALID_GPU_ALLOCATORS,
    _VALID_PLATFORMS,
    _VALID_POLICY_DTYPES,
    _VALID_PRECISION_SELECTIONS,
    _VALID_TF_GPU_ALLOCATORS,
    _VALID_TRANSFER_GUARDS,
    _XLA_CLIENT_MEM_FRACTION_ENV,
    _XLA_FLAGS_ENV,
    _XLA_PYTHON_CLIENT_ALLOCATOR_ENV,
    _XLA_PYTHON_CLIENT_MEM_FRACTION_ENV,
    _XLA_PYTHON_CLIENT_PREALLOCATE_ENV,
    VALID_BACKEND_MODES,
    BackendConfig,
    BackendMode,
    BackendPolicy,
    ExecutionIntent,
    JaxDevice,
    JaxExecutionProfile,
    PrecisionSelection,
    ResolvedPrecision,
    _config_from_mode,
    _debug_overlay_enabled,
    _default_compilation_cache_dir,
    _default_transfer_guard,
    _env_bool,
    _get_mode_policy_defaults,
    _mode_from_legacy_env,
    _optional_bool_env,
    _optional_bool_policy_default,
    _optional_env_value,
    _optional_float_policy_default,
    _optional_nonempty_env,
    _optional_nonneg_int_env,
    _parse_bool_value,
    _policy_from_config,
    _primary_jax_platforms_env_platform,
    _reject_obsolete_precision_environment,
    _resolve_kwarg,
    _resolve_legacy_platform,
    _resolve_legacy_value,
    _resolve_policy_max_dense_jacobian_bytes,
    _resolved_precision_for_mode,
    _runtime_env_value,
    _runtime_jax_backend_name,
    _runtime_jax_platform_value,
    _runtime_jax_platforms_value,
    _validate_backend,
    _validate_default_optimizer_backend,
    _validate_default_residency,
    _validate_gpu_allocator,
    _validate_mem_fraction_value,
    _validate_mode,
    _validate_platform,
    _validate_policy_dtype,
    _validate_precision_for_mode,
    _validate_precision_selection,
    _validate_tf_gpu_allocator,
    _validate_transfer_guard,
    is_float32_smoke_policy,
    resolve_jax_execution_profile,
)
from simsopt_jax.backend._runtime_tuning import (  # noqa: F401
    _AUTOTUNED_CHUNK_SIZES_BY_POLICY,
    _COIL_AXIS_SHARDING_STRATEGIES,
    _DEFAULT_COIL_SHARDING_AXIS_NAME,
    _DEFAULT_SHARDING_AXIS_NAME,
    _FIELD_KERNEL_DEFAULTS,
    _FIELD_KERNEL_ENV_BY_KEY,
    _MIN_COILS_TO_SHARD_BY_POLICY,
    _MIN_PAIRWISE_ROWS_TO_SHARD_BY_POLICY,
    _MIN_POINTS_TO_SHARD_BY_POLICY,
    _MIXED_BIOT_SAVART_SOURCE_TILE_SIZE,
    _MODE_SHARDING_DEFAULTS,
    _PAIRWISE_PENALTY_CHUNK_SIZE_BY_POLICY,
    _PAIRWISE_ROW_SHARDING_STRATEGIES,
    _POINT_AXIS_SHARDING_STRATEGIES,
    _POINT_CHUNK_SIZE_BY_POLICY,
    _POINT_OWNED_SHARDING_STRATEGIES,
    _VALID_SHARDING_STRATEGIES,
    ChunkTuning,
    DistributedRuntimeConfig,
    FieldKernelTuning,
    ShardingTuning,
    _apply_chunk_env_overrides,
    _build_chunk_tuning,
    _build_distributed_runtime_config,
    _build_sharding_tuning,
    _detect_active_jax_cuda_device_index,
    _detect_active_jax_cuda_device_selector,
    _detect_global_jax_device_count,
    _detect_imported_jax_cuda_device_index,
    _detect_local_jax_device_count,
    _factor_device_count_2d,
    _pairwise_penalty_chunk_size_default,
    _parse_local_device_ids,
    _parse_nvidia_smi_indexed_value_row,
    _parse_visible_cuda_device_index,
    _point_chunk_size_default,
    _query_gpu_metric_mb_from_nvidia_smi,
    _query_gpu_total_memory_mb_from_nvidia_smi,
    _resolve_autotuned_chunk_sizes,
    _resolve_chunk_autotune_enabled,
    _resolve_coil_sharding_axis_name,
    _resolve_gpu_total_memory_mb,
    _resolve_min_coils_to_shard,
    _resolve_min_pairwise_rows_to_shard,
    _resolve_min_points_to_shard,
    _resolve_sharding_axis_name,
    _resolve_sharding_strategy,
    _static_chunk_sizes,
    _strategy_device_counts,
    _strategy_mesh_axis_names,
    _strategy_reduced_axis_name,
    _validate_sharding_strategy,
    _visible_cuda_device_selector,
    _with_distributed_initialized,
)
from simsopt_jax.numerical_policy import CertificateDType

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "VALID_BACKEND_MODES",
    "BackendConfig",
    "BackendMode",
    "BackendPolicy",
    "ChunkTuning",
    "DistributedRuntimeConfig",
    "ExecutionIntent",
    "FieldKernelTuning",
    "JaxDevice",
    "JaxExecutionProfile",
    "PrecisionSelection",
    "ResolvedPrecision",
    "ShardingTuning",
    "apply_jax_runtime_config",
    "get_active_cuda_device_index",
    "get_backend",
    "get_backend_config",
    "get_backend_mode",
    "get_backend_policy",
    "get_certificate_dtype",
    "get_chunk_policy",
    "get_chunk_tuning",
    "get_coil_chunk_size",
    "get_compilation_cache_dir",
    "get_compilation_cache_policy",
    "get_compute_dtype",
    "get_debug_nans",
    "get_disable_jit",
    "get_distributed_runtime_config",
    "get_field_kernel_tuning",
    "get_jax_platform",
    "get_pairwise_penalty_chunk_size",
    "get_point_chunk_size",
    "get_precision",
    "get_provenance_label",
    "get_quadrature_block_size",
    "get_resolved_precision",
    "get_runtime_jax_device",
    "get_sharding_strategy",
    "get_sharding_tuning",
    "get_tolerance_tier",
    "get_transfer_guard",
    "invalidate_backend_cache",
    "is_backend_strict",
    "is_float32_smoke_policy",
    "is_jax_backend",
    "is_mixed_precision_enabled",
    "is_parity_mode",
    "maybe_initialize_distributed_jax",
    "query_active_gpu_memory_mb",
    "raise_if_strict_jax_fallback",
    "raise_if_target_lane_bypass",
    "register_backend_cache_clear",
    "requires_x64",
    "resolve_jax_execution_profile",
    "set_backend",
    "should_eagerly_configure_jax",
    "should_shard_coil_groups",
    "should_shard_pairwise_rows",
    "should_shard_points",
    "strict_target_lane_purity",
    "target_lane_purity_active",
    "target_lane_purity_requested",
    "use_runtime",
    "validate_cuda_determinism_environment",
    "warn_if_jax_fallback",
    "with_cpu_device_for_construction",
]

_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_exclude_nondeterministic_ops",)
_STALE_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_deterministic_ops",)
_CPU_OPT_PRESET_FLAG_NAME = "--xla_cpu_opt_preset"
_CPU_OPT_PRESET_FAST_COMPILE = f"{_CPU_OPT_PRESET_FLAG_NAME}=FAST_COMPILE"


_BackendCacheClearCallbackKey = tuple[str, str]
_backend_runtime_lock = threading.RLock()
_target_lane_purity_depth = ContextVar("simsopt_target_lane_purity_depth", default=0)
_backend_cache_clear_callbacks: dict[
    _BackendCacheClearCallbackKey, Callable[[], None]
] = {}


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
    return " or ".join(f"{flag_name}=true" for flag_name in _GPU_DETERMINISM_XLA_FLAGS)


def _xla_flags_with_cpu_compile_preset(xla_flags: str | None) -> str:
    """Return ``xla_flags`` with the CPU FAST_COMPILE preset appended.

    Idempotent and non-destructive: existing tokens are preserved verbatim, and
    a caller-provided ``--xla_cpu_opt_preset`` (any value) is respected rather
    than overridden. ``None``/empty input yields just the preset token.
    """
    stripped_xla_flags = "" if xla_flags is None else xla_flags.strip()
    tokens = _split_xla_flag_tokens(xla_flags)
    if any(
        token == _CPU_OPT_PRESET_FLAG_NAME
        or token.startswith(f"{_CPU_OPT_PRESET_FLAG_NAME}=")
        for token in tokens
    ):
        return xla_flags or ""
    if not stripped_xla_flags:
        return _CPU_OPT_PRESET_FAST_COMPILE
    return f"{stripped_xla_flags} {_CPU_OPT_PRESET_FAST_COMPILE}"


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


def _resolve_mode(mode: str | None = None) -> str:
    if mode is None:
        return get_backend_mode()
    return _validate_mode(mode)


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


def get_precision(mode: str | None = None) -> PrecisionSelection:
    """Return the explicit precision selection for a backend mode."""
    return get_backend_policy(mode).precision


def get_resolved_precision(mode: str | None = None) -> ResolvedPrecision:
    """Return the effective precision route for a backend mode."""
    return get_backend_policy(mode).resolved_precision


def get_compute_dtype(mode: str | None = None) -> str:
    """Return the compute dtype name for a backend mode."""
    return get_backend_policy(mode).compute_dtype


def is_mixed_precision_enabled(mode: str | None = None) -> bool:
    """Return whether compute arrays intentionally differ from certificates."""
    return get_backend_policy(mode).resolved_precision == "mixed"


def get_certificate_dtype(mode: str | None = None) -> CertificateDType | None:
    """Return the optional FP64 certificate dtype for a backend mode."""
    return get_backend_policy(mode).certificate_dtype


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
            mixed_biot_savart_source_tile_size=(_MIXED_BIOT_SAVART_SOURCE_TILE_SIZE),
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


def get_runtime_jax_device(mode: str | None = None):
    """Return the first local JAX device for the active runtime policy."""
    policy = get_backend_policy(mode)
    if policy.backend == "jax":
        platform = policy.jax_platform
    else:
        platform = _primary_jax_platforms_env_platform()
    if platform is None:
        return None

    import jax

    backend_name = _runtime_jax_backend_name(platform)
    return jax.local_devices(backend=backend_name)[0]


def get_active_cuda_device_index(mode: str | None = None) -> int | None:
    """Return the active CUDA device index implied by env or JAX runtime state."""
    policy = get_backend_policy(mode)
    if (
        policy.jax_platform != "cuda"
        and _detect_active_jax_cuda_device_selector() is None
    ):
        return None
    return _detect_active_jax_cuda_device_index()


def query_active_gpu_memory_mb(mode: str | None = None) -> float | None:
    """Return coarse memory usage for the active CUDA device when available."""
    policy = get_backend_policy(mode)
    device_selector = _detect_active_jax_cuda_device_selector()
    if policy.jax_platform != "cuda" and device_selector is None:
        return None
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


def _gpu_memory_runtime_env(
    config: BackendConfig,
) -> tuple[tuple[str, str | None], ...]:
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
        "importing jax. Call simsopt_jax.config.set_backend(...) or set "
        "SIMSOPT_BACKEND_MODE before importing or touching JAX devices."
    )


def _apply_jax_gpu_memory_env(config: BackendConfig) -> None:
    if config.jax_platform != "cuda":
        return
    _assert_jax_not_imported_for_gpu_memory_config(config)
    for name, value in _gpu_memory_runtime_env(config):
        _set_runtime_env(name, value)


def _apply_cpu_compile_preset_env(config: BackendConfig, policy: BackendPolicy) -> None:
    """Pull the FAST_COMPILE CPU preset into ``XLA_FLAGS`` before JAX inits.

    XLA reads ``XLA_FLAGS`` only at backend initialization, so this runs in the
    pre-``import jax`` region of :func:`apply_jax_runtime_config`. Applied to
    non-parity CPU lanes only: ``xla_cpu_opt_preset`` is inert on the CUDA
    backend (whose XLA flags carry the determinism contract), and the preset
    reduces XLA optimization passes -- which can shift CPU reduction order, so
    it is withheld from the bit-exact ``*_parity`` lanes.
    """
    if config.jax_platform == "cuda" or policy.parity_mode:
        return
    _set_runtime_env(
        _XLA_FLAGS_ENV,
        _xla_flags_with_cpu_compile_preset(os.environ.get(_XLA_FLAGS_ENV)),
    )


def apply_jax_runtime_config() -> None:
    """Apply the resolved JAX runtime settings to the active process."""
    config = get_backend_config()
    if config.backend != "jax":
        return
    policy = get_backend_policy(config.mode)
    _validate_cuda_parity_determinism_env(config, policy)
    _apply_jax_gpu_memory_env(config)
    _apply_cpu_compile_preset_env(config, policy)

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
        # Follow JAX's documented GPU persistent-cache setting. Wider XLA cache
        # modes can force nvlink through container CUDA toolkits that differ
        # from the NVIDIA libraries bundled with the JAX wheel.
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
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


@overload
def set_backend(
    mode: Literal["jax"],
    *,
    device: JaxDevice,
    intent: ExecutionIntent = "fast",
    precision: PrecisionSelection | None = None,
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
) -> BackendConfig: ...


@overload
def set_backend(
    mode: BackendMode,
    *,
    device: None = None,
    intent: None = None,
    precision: PrecisionSelection | None = None,
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
) -> BackendConfig: ...


def set_backend(
    mode: BackendMode | Literal["jax"],
    *,
    device: JaxDevice | None = None,
    intent: ExecutionIntent | None = None,
    precision: PrecisionSelection | None = None,
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
    if mode == "jax":
        if device is None:
            raise ValueError("set_backend('jax') requires device='cpu' or device='gpu'")
        resolved_mode = resolve_jax_execution_profile(
            device,
            "fast" if intent is None else intent,
        ).mode
    else:
        if device is not None or intent is not None:
            raise ValueError(
                "A canonical backend mode cannot be combined with device or intent"
            )
        resolved_mode = _validate_mode(mode)
    config = _config_from_mode(
        resolved_mode,
        strict=bool(strict),
        precision=precision,
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


@overload
def use_runtime(
    mode: Literal["jax"],
    *,
    device: JaxDevice,
    intent: ExecutionIntent = "fast",
    precision: PrecisionSelection | None = None,
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
) -> BackendConfig: ...


@overload
def use_runtime(
    mode: BackendMode,
    *,
    device: None = None,
    intent: None = None,
    precision: PrecisionSelection | None = None,
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
) -> BackendConfig: ...


def use_runtime(
    mode: BackendMode | Literal["jax"],
    *,
    device: JaxDevice | None = None,
    intent: ExecutionIntent | None = None,
    precision: PrecisionSelection | None = None,
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
        device=device,
        intent=intent,
        precision=precision,
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
