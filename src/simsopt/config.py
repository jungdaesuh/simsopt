"""Public runtime configuration helpers for simsopt."""

from __future__ import annotations

from .backend import (
    BackendConfig,
    VALID_BACKEND_MODES,
    apply_jax_runtime_config,
    get_backend,
    get_backend_config,
    get_backend_mode,
    get_jax_platform,
    is_backend_strict,
    is_jax_backend,
    set_backend,
)

__all__ = [
    "BackendConfig",
    "VALID_BACKEND_MODES",
    "apply_jax_runtime_config",
    "get_backend",
    "get_backend_config",
    "get_backend_mode",
    "get_jax_platform",
    "is_backend_strict",
    "is_jax_backend",
    "set_backend",
]
