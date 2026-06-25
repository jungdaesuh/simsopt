"""Compatibility facade for single-stage inner Boozer backend routing."""

from __future__ import annotations

from simsopt_jax.geo.optimizers.single_stage_routing import (
    resolve_boozer_least_squares_algorithm,
    resolve_boozer_limited_memory,
    resolve_boozer_optimizer_backend,
    resolve_boozer_optimizer_method,
    resolve_single_stage_jax_boozer_optimizer_backend,
)

__all__ = (
    "resolve_boozer_least_squares_algorithm",
    "resolve_boozer_limited_memory",
    "resolve_boozer_optimizer_backend",
    "resolve_boozer_optimizer_method",
    "resolve_single_stage_jax_boozer_optimizer_backend",
)
