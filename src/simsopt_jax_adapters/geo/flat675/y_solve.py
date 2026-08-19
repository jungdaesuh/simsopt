"""Float64 two-column least-squares solve for the flat-675 inner state.

The flat formulation keeps only ``(iota, G)`` in the inner state and closes
them in one economy QR against the Boozer system, so the outer objective is a
function of the 675 outer coordinates alone.  Beside the solution the record
publishes the singular values, the numerical rank they imply, and whether the
whole solve stayed finite — the conditioning evidence the archived campaign
certificate can be checked against.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

FLAT675_Y_COLUMN_COUNT: Final[int] = 2


@dataclass(frozen=True, slots=True)
class Flat675YSolution:
    """Fixed-shape device result of one float64 two-column QR solve."""

    solution: jax.Array
    singular_values: jax.Array
    numerical_rank: jax.Array
    numerics_finite: jax.Array


def _validated_inputs(
    design_matrix: jax.Array,
    right_hand_side: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    matrix = jnp.asarray(design_matrix)
    rhs = jnp.asarray(right_hand_side)
    if matrix.ndim != 2 or matrix.shape[1] != FLAT675_Y_COLUMN_COUNT:
        raise ValueError("flat-675 y design matrix must have shape (m, 2).")
    if matrix.shape[0] < FLAT675_Y_COLUMN_COUNT:
        raise ValueError("flat-675 y design matrix must have at least two rows.")
    if rhs.ndim != 1 or rhs.shape[0] != matrix.shape[0]:
        raise ValueError("flat-675 y right-hand side must have shape (m,).")
    if matrix.dtype != jnp.dtype(jnp.float64) or rhs.dtype != jnp.dtype(jnp.float64):
        raise TypeError("flat-675 y QR solve requires float64 inputs.")
    return matrix, rhs


def solve_flat675_y_qr(
    design_matrix: jax.Array,
    right_hand_side: jax.Array,
) -> Flat675YSolution:
    """Solve ``min ||A y - b||`` for the two inner scalars by economy QR."""
    matrix, rhs = _validated_inputs(design_matrix, right_hand_side)
    orthogonal, triangular = jnp.linalg.qr(matrix, mode="reduced")
    solution = jsp_linalg.solve_triangular(
        triangular,
        orthogonal.T @ rhs,
        lower=False,
    )
    singular_values = jnp.linalg.svd(triangular, compute_uv=False)

    # The rank threshold is LAPACK's: the largest singular value scaled by the
    # problem's leading dimension and one machine epsilon.
    epsilon = jnp.asarray(sys.float_info.epsilon, dtype=matrix.dtype)
    rank_threshold = (
        jnp.asarray(max(matrix.shape), dtype=matrix.dtype)
        * epsilon
        * singular_values[0]
    )
    return Flat675YSolution(
        solution=solution,
        singular_values=singular_values,
        numerical_rank=jnp.sum(singular_values > rank_threshold, dtype=jnp.int32),
        numerics_finite=(
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(rhs))
            & jnp.all(jnp.isfinite(solution))
            & jnp.all(jnp.isfinite(singular_values))
        ),
    )


__all__ = [
    "FLAT675_Y_COLUMN_COUNT",
    "Flat675YSolution",
    "solve_flat675_y_qr",
]
