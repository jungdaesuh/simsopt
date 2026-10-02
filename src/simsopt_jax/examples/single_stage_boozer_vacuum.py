"""Outer-optimization constants of the BoozerQA JAX mirror."""

from __future__ import annotations

from typing import Final

OUTER_GRADIENT_TOLERANCE: Final[float] = 1.0e-15
JAX_FAST_DRIVER_ID: Final[str] = "simsopt_jax_host_lbfgsb_with_traceable_boozer_newton"
JAX_PARITY_DRIVER_ID: Final[str] = "simsopt_jax_host_bfgs_with_traceable_boozer_newton"


__all__ = (
    "JAX_FAST_DRIVER_ID",
    "JAX_PARITY_DRIVER_ID",
    "OUTER_GRADIENT_TOLERANCE",
)
