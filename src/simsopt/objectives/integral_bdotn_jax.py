"""Compatibility shim for the pure JAX ``integral_BdotN`` kernels.

The implementation lives in :mod:`simsopt.jax_core.integral_bdotn`.
This module preserves the historical public import path.
"""

from ..jax_core.integral_bdotn import (
    integral_BdotN,
    residual_BdotN,
    signed_BdotN_flux,
)

__all__ = ["integral_BdotN", "residual_BdotN", "signed_BdotN_flux"]
