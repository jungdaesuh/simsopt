"""Permutation-invariant observables for symmetric surface parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


def parameter_invariants(parameters: np.ndarray) -> np.ndarray:
    """Return quotient coordinates for two interchangeable ellipse semi-axes."""
    values = np.asarray(parameters, dtype=np.float64)
    return np.asarray((np.sum(values), np.prod(values)), dtype=np.float64)


def global_column_swap_jacobian_invariants(
    left: np.ndarray | jax.Array,
    right: np.ndarray | jax.Array,
) -> tuple[
    np.ndarray | jax.Array,
    np.ndarray | jax.Array,
    np.ndarray | jax.Array,
]:
    """Identify two Jacobian columns up to one global column exchange.

    The normalized difference association couples every residual row, so an
    independent row-wise column swap cannot masquerade as a global exchange.
    Differences at roundoff scale are treated as exact column coincidence so
    the normalization does not amplify an arbitrary native/JAX sign.
    """
    column_difference = left - right
    difference_scale = abs(column_difference).max()
    value_scale = abs(left).max() + abs(right).max()
    roundoff_threshold = (
        128.0 * np.finfo(np.dtype(left.dtype)).eps * (value_scale + (value_scale == 0))
    )
    active = difference_scale > roundoff_threshold
    normalized_difference = (
        column_difference / (difference_scale + (difference_scale == 0))
    ) * active
    return (
        left + right,
        left * right,
        normalized_difference[:, None] * normalized_difference[None, :],
    )


__all__ = [
    "global_column_swap_jacobian_invariants",
    "parameter_invariants",
]
