"""Pure JAX quasisymmetry metrics shared by objective front ends."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["non_quasi_symmetric_ratio"]


def non_quasi_symmetric_ratio(
    xphi: jax.Array,
    xtheta: jax.Array,
    magnetic_field: jax.Array,
    *,
    axis: int,
) -> jax.Array:
    """Return the surface-weighted non-QS magnetic-field ratio."""

    normal = jnp.cross(xphi, xtheta)
    differential_area = jnp.linalg.norm(normal, axis=-1)
    mod_B = jnp.linalg.norm(magnetic_field, axis=-1)
    quasi_symmetric_B = jnp.sum(mod_B * differential_area, axis=axis) / jnp.sum(
        differential_area,
        axis=axis,
    )
    quasi_symmetric_B = jnp.expand_dims(quasi_symmetric_B, axis=axis)
    non_quasi_symmetric_B = mod_B - quasi_symmetric_B
    return jnp.sum(differential_area * non_quasi_symmetric_B**2) / jnp.sum(
        differential_area * quasi_symmetric_B**2
    )
