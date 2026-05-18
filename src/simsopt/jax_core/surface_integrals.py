"""Shared pure JAX surface integral helpers."""

import jax.numpy as jnp

from ._math_utils import as_jax_float64 as _as_jax_float64
from ._vector_norms import norm3 as _norm3

__all__ = ["surface_area", "surface_volume"]


def surface_volume(gamma, normal):
    """Compute the volume enclosed by a toroidal surface."""
    nphi, ntheta = gamma.shape[:2]
    integrand = jnp.sum(gamma * normal, axis=-1)
    return jnp.sum(integrand) / _as_jax_float64(3.0 * nphi * ntheta)


def surface_area(normal):
    """Compute the area of a toroidal surface."""
    nphi, ntheta = normal.shape[:2]
    norm_n = _norm3(normal)[..., 0]
    return jnp.sum(norm_n) / _as_jax_float64(nphi * ntheta)
