"""Public runtime configuration helpers for simsopt."""

from __future__ import annotations

from .backend import (
    BackendConfig,
    BackendPolicy,
    VALID_BACKEND_MODES,
    apply_jax_runtime_config,
    get_backend,
    get_backend_config,
    get_backend_mode,
    get_backend_policy,
    get_chunk_policy,
    get_compilation_cache_policy,
    get_jax_platform,
    get_provenance_label,
    get_tolerance_tier,
    is_backend_strict,
    is_jax_backend,
    is_parity_mode,
    requires_x64,
    set_backend,
)

__all__ = [
    "BackendConfig",
    "BackendPolicy",
    "VALID_BACKEND_MODES",
    "apply_jax_runtime_config",
    "get_backend",
    "get_backend_config",
    "get_backend_mode",
    "get_backend_policy",
    "get_chunk_policy",
    "get_compilation_cache_policy",
    "get_jax_platform",
    "get_provenance_label",
    "get_tolerance_tier",
    "is_backend_strict",
    "is_jax_backend",
    "is_parity_mode",
    "requires_x64",
    "set_backend",
]
