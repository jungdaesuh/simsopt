"""Shared native-default budget for the VMEC-free Boozer example pair."""

from typing import Final

NATIVE_ITERATIONS: Final[int] = 1_000
OUTER_GRADIENT_TOLERANCE: Final[float] = 1.0e-15
JAX_FAST_DRIVER_ID: Final[str] = "simsopt_jax_host_lbfgsb_with_traceable_boozer_newton"
JAX_OPTAX_DRIVER_ID: Final[str] = "simsopt_jax_optax_lbfgs_with_traceable_boozer_newton"
JAX_PARITY_DRIVER_ID: Final[str] = "simsopt_jax_host_bfgs_with_traceable_boozer_newton"

__all__ = (
    "JAX_FAST_DRIVER_ID",
    "JAX_OPTAX_DRIVER_ID",
    "JAX_PARITY_DRIVER_ID",
    "NATIVE_ITERATIONS",
    "OUTER_GRADIENT_TOLERANCE",
)
