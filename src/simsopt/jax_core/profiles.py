"""Pure JAX radial-profile kernels."""

from __future__ import annotations

import numbers

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64
from ._spline_utils import bspline_deriv_1d, bspline_eval_1d

__all__ = [
    "profile_polynomial_dfds",
    "profile_polynomial_value",
    "profile_pressure_dfds",
    "profile_pressure_value",
    "profile_scaled_dfds",
    "profile_scaled_value",
    "profile_spline_dfds",
    "profile_spline_value",
]


def _horner_ascending(coeffs: jax.Array, s: jax.Array) -> jax.Array:
    coeffs_jax = as_jax_float64(coeffs)
    s_jax = as_jax_float64(s)
    result = jnp.zeros_like(s_jax, dtype=coeffs_jax.dtype)
    for coeff in jnp.flip(coeffs_jax, axis=0):
        result = result * s_jax + coeff
    return result


def profile_polynomial_value(coeffs: jax.Array, s: jax.Array) -> jax.Array:
    """Evaluate an ascending-power polynomial profile at ``s``."""
    return _horner_ascending(coeffs, s)


def profile_polynomial_dfds(coeffs: jax.Array, s: jax.Array) -> jax.Array:
    """Evaluate ``d/ds`` for an ascending-power polynomial profile."""
    coeffs_jax = as_jax_float64(coeffs)
    if int(coeffs_jax.shape[0]) <= 1:
        return jnp.zeros_like(as_jax_float64(s), dtype=coeffs_jax.dtype)
    powers = jnp.arange(1, coeffs_jax.shape[0], dtype=coeffs_jax.dtype)
    derivative_coeffs = coeffs_jax[1:] * powers
    return _horner_ascending(derivative_coeffs, s)


def profile_scaled_value(scale: jax.Array, base_value: jax.Array) -> jax.Array:
    """Scale a profile value."""
    return as_jax_float64(scale) * as_jax_float64(base_value)


def profile_scaled_dfds(scale: jax.Array, base_dfds: jax.Array) -> jax.Array:
    """Scale a profile derivative."""
    return as_jax_float64(scale) * as_jax_float64(base_dfds)


def profile_pressure_value(values_pairs: tuple[jax.Array, ...]) -> jax.Array:
    """Evaluate ``sum_j f_{2j}(s) f_{2j+1}(s)`` for profile pairs."""
    total = jnp.zeros_like(as_jax_float64(values_pairs[0]))
    for left, right in zip(values_pairs[0::2], values_pairs[1::2]):
        total = total + as_jax_float64(left) * as_jax_float64(right)
    return total


def profile_pressure_dfds(
    values_pairs: tuple[jax.Array, ...], dfds_pairs: tuple[jax.Array, ...]
) -> jax.Array:
    """Evaluate the product-rule derivative for profile-pressure pairs."""
    total = jnp.zeros_like(as_jax_float64(values_pairs[0]))
    for left, right, dleft, dright in zip(
        values_pairs[0::2],
        values_pairs[1::2],
        dfds_pairs[0::2],
        dfds_pairs[1::2],
    ):
        total = total + as_jax_float64(dleft) * as_jax_float64(right)
        total = total + as_jax_float64(left) * as_jax_float64(dright)
    return total


def _spline_degree_index(degree: int) -> jax.Array:
    if isinstance(degree, numbers.Integral):
        degree_int = int(degree)
        if degree_int < 1 or degree_int > 5:
            raise ValueError("profile spline degree must be in [1, 5].")
        return jnp.asarray(degree_int, dtype=jnp.int32)
    return jnp.asarray(degree, dtype=jnp.int32)


def _raise_invalid_spline_degree() -> None:
    raise ValueError("profile spline degree must be in [1, 5].")


def _invalid_spline_value_branch(operands) -> jax.Array:
    jax.debug.callback(_raise_invalid_spline_degree)
    return jnp.zeros_like(as_jax_float64(operands[2]))


def profile_spline_value(
    knots: jax.Array, coeffs: jax.Array, degree: int, s: jax.Array
) -> jax.Array:
    """Evaluate a host-fitted FITPACK spline profile at ``s``."""
    branches = (
        _invalid_spline_value_branch,
        *tuple(
            lambda operands, degree_int=degree_int: bspline_eval_1d(
                operands[0], operands[1], degree_int, operands[2]
            )
            for degree_int in range(1, 6)
        ),
        _invalid_spline_value_branch,
    )
    return jax.lax.switch(
        _spline_degree_index(degree),
        branches,
        (knots, coeffs, s),
    )


def profile_spline_dfds(
    knots: jax.Array, coeffs: jax.Array, degree: int, s: jax.Array
) -> jax.Array:
    """Evaluate the first derivative of a host-fitted FITPACK spline."""
    branches = (
        _invalid_spline_value_branch,
        *tuple(
            lambda operands, degree_int=degree_int: bspline_deriv_1d(
                operands[0], operands[1], degree_int, operands[2]
            )
            for degree_int in range(1, 6)
        ),
        _invalid_spline_value_branch,
    )
    return jax.lax.switch(
        _spline_degree_index(degree),
        branches,
        (knots, coeffs, s),
    )
