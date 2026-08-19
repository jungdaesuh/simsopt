"""Pure-JAX implementation of the flat-675 two-column Boozer system."""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
from simsopt_jax.core._math_utils import explicit_rsqrt
from simsopt_jax.core.reductions import pairwise_sum_axis

JaxFlat675BoozerSystemArrays: TypeAlias = tuple[jax.Array, jax.Array]


def build_flat675_boozer_system_arrays(
    magnetic_field: jax.Array,
    toroidal_tangent: jax.Array,
    poloidal_tangent: jax.Array,
    *,
    weight_by_inverse_field_magnitude: bool,
) -> JaxFlat675BoozerSystemArrays:
    """Build numerical A/b arrays without claiming candidate provenance."""

    field_squared = pairwise_sum_axis(
        magnetic_field * magnetic_field,
        axis=-1,
    )
    if weight_by_inverse_field_magnitude:
        point_weight = explicit_rsqrt(field_squared)
    else:
        point_weight = jnp.ones_like(field_squared)

    weighted_field_squared = point_weight * field_squared
    iota_column = -(weighted_field_squared[..., None] * poloidal_tangent).reshape(-1)
    current_column = (point_weight[..., None] * magnetic_field).reshape(-1)
    right_hand_side = (weighted_field_squared[..., None] * toroidal_tangent).reshape(-1)

    residual_count = jnp.asarray(magnetic_field.size, dtype=magnetic_field.dtype)
    normalization = jnp.sqrt(residual_count)
    design_matrix = jnp.stack((iota_column, current_column), axis=1) / normalization
    return design_matrix, right_hand_side / normalization


__all__ = [
    "JaxFlat675BoozerSystemArrays",
    "build_flat675_boozer_system_arrays",
]
