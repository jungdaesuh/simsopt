"""Shared pure JAX surface integral helpers."""

import jax
import jax.numpy as jnp

from ._device_scalars import two_pi as _two_pi
from ._math_utils import as_jax_float64 as _as_jax_float64
from ._vector_norms import norm3 as _norm3

__all__ = [
    "surface_area",
    "surface_major_radius",
    "surface_mean_cross_sectional_area",
    "surface_volume",
]


@jax.jit
def surface_volume(gamma, normal):
    """Compute the volume enclosed by a toroidal surface."""
    nphi, ntheta = gamma.shape[:2]
    integrand = jnp.sum(gamma * normal, axis=-1)
    return jnp.sum(integrand) / _as_jax_float64(3.0 * nphi * ntheta)


@jax.jit
def surface_area(normal):
    """Compute the area of a toroidal surface."""
    nphi, ntheta = normal.shape[:2]
    norm_n = jnp.reshape(_norm3(normal), normal.shape[:-1])
    return jnp.sum(norm_n) / _as_jax_float64(nphi * ntheta)


@jax.jit
def surface_mean_cross_sectional_area(gamma, xphi, xtheta):
    """Compute the absolute mean toroidal cross-sectional area."""

    x, y, _z = jnp.split(gamma, (1, 2), axis=2)
    xphi_x, xphi_y, xphi_z = jnp.split(xphi, (1, 2), axis=2)
    xtheta_x, xtheta_y, xtheta_z = jnp.split(xtheta, (1, 2), axis=2)
    radius_squared = x * x + y * y
    jacobian_00 = (x * xphi_y - y * xphi_x) / radius_squared
    jacobian_01 = (x * xtheta_y - y * xtheta_x) / radius_squared
    dz_dtheta = xtheta_z - xphi_z * jacobian_01 / jacobian_00
    signed_area = jnp.mean(
        jnp.sqrt(radius_squared) * dz_dtheta * jacobian_00
    ) / _two_pi(radius_squared)
    return jnp.abs(signed_area)


@jax.jit
def surface_major_radius(gamma, xphi, xtheta):
    """Compute the VMEC-style major radius from volume and mean area."""

    normal = jnp.cross(xphi, xtheta)
    volume = jnp.abs(surface_volume(gamma, normal))
    mean_area = surface_mean_cross_sectional_area(gamma, xphi, xtheta)
    pi = _two_pi(mean_area) / _as_jax_float64(2.0)
    minor_radius_squared = mean_area / pi
    return volume / (_two_pi(volume) * pi * minor_radius_squared)
