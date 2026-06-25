"""FITPACK-compatible one-dimensional B-spline evaluation in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64

__all__ = [
    "bspline_deriv_1d",
    "bspline_eval_1d",
]


def _active_coefficients(knots: jax.Array, coeffs: jax.Array, degree: int) -> jax.Array:
    ncoeffs = int(knots.shape[0]) - int(degree) - 1
    return coeffs[:ncoeffs]


def _safe_divide(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    zero_denominator = denominator == 0.0
    safe_denominator = jnp.where(zero_denominator, 1.0, denominator)
    return jnp.where(zero_denominator, 0.0, numerator / safe_denominator)


def _bspline_eval_scalar(
    knots: jax.Array, coeffs: jax.Array, degree: int, x: jax.Array
) -> jax.Array:
    degree_int = int(degree)
    active_coeffs = _active_coefficients(knots, coeffs, degree_int)
    ncoeffs = int(active_coeffs.shape[0])
    interval = jnp.searchsorted(knots, x, side="right") - 1
    interval = jnp.clip(interval, degree_int, ncoeffs - 1)
    local_index = interval - degree_int + jnp.arange(degree_int + 1)
    values = jnp.take(active_coeffs, local_index, mode="clip")
    for r in range(1, degree_int + 1):
        for j in range(degree_int, r - 1, -1):
            knot_index = interval - degree_int + j
            left = knots[knot_index]
            right = knots[knot_index + degree_int + 1 - r]
            alpha = _safe_divide(x - left, right - left)
            updated = (1.0 - alpha) * values[j - 1] + alpha * values[j]
            values = values.at[j].set(updated)
    return values[degree_int]


def bspline_eval_1d(
    knots: jax.Array, coeffs: jax.Array, degree: int, x: jax.Array
) -> jax.Array:
    """Evaluate a FITPACK ``(t, c, k)`` spline representation at ``x``."""
    knots_jax = as_jax_float64(knots)
    coeffs_jax = as_jax_float64(coeffs)
    x_jax = as_jax_float64(x)
    flat = jnp.ravel(x_jax)
    evaluated = jax.vmap(
        lambda x_scalar: _bspline_eval_scalar(
            knots_jax, coeffs_jax, int(degree), x_scalar
        )
    )(flat)
    return jnp.reshape(evaluated, x_jax.shape)


def bspline_deriv_1d(
    knots: jax.Array, coeffs: jax.Array, degree: int, x: jax.Array
) -> jax.Array:
    """Evaluate the first derivative of a FITPACK ``(t, c, k)`` spline."""
    degree_int = int(degree)
    knots_jax = as_jax_float64(knots)
    coeffs_jax = as_jax_float64(coeffs)
    active_coeffs = _active_coefficients(knots_jax, coeffs_jax, degree_int)
    denominators = (
        knots_jax[degree_int + 1 : degree_int + active_coeffs.shape[0]]
        - knots_jax[1 : active_coeffs.shape[0]]
    )
    derivative_coeffs = degree_int * _safe_divide(
        active_coeffs[1:] - active_coeffs[:-1], denominators
    )
    return bspline_eval_1d(knots_jax[1:-1], derivative_coeffs, degree_int - 1, x)
