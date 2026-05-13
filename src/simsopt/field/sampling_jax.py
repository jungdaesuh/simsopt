"""JAX public wrappers for weighted curve/surface sampling."""

from ..jax_core.sampling import (
    draw_uniform_on_curve_jax,
    draw_uniform_on_surface_jax,
    sample_weighted_indices_jax,
)

__all__ = [
    "draw_uniform_on_curve_jax",
    "draw_uniform_on_surface_jax",
    "sample_weighted_indices_jax",
]
