"""Float64 two-column least-squares solve for the flat-675 inner state.

The flat formulation keeps only ``(iota, G)`` in the inner state and closes
them in one economy QR against the Boozer system, so the outer objective is a
function of the 675 outer coordinates alone.  The returned record carries the
conditioning and residual numbers that make the solve auditable; the objective
itself reads only ``solution``.
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
    rank_threshold: jax.Array
    numerical_rank: jax.Array
    condition_number: jax.Array
    residual_l2_norm: jax.Array
    relative_fit_residual: jax.Array
    normal_residual_l2_norm: jax.Array
    relative_normal_residual: jax.Array
    numerics_finite: jax.Array


jax.tree_util.register_dataclass(
    Flat675YSolution,
    data_fields=[
        "solution",
        "singular_values",
        "rank_threshold",
        "numerical_rank",
        "condition_number",
        "residual_l2_norm",
        "relative_fit_residual",
        "normal_residual_l2_norm",
        "relative_normal_residual",
        "numerics_finite",
    ],
    meta_fields=[],
)


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

    residual = matrix @ solution - rhs
    maximum_singular_value = singular_values[0]
    minimum_singular_value = singular_values[-1]
    epsilon = jnp.asarray(sys.float_info.epsilon, dtype=matrix.dtype)
    rank_threshold = (
        jnp.asarray(max(matrix.shape), dtype=matrix.dtype)
        * epsilon
        * maximum_singular_value
    )
    numerical_rank = jnp.sum(singular_values > rank_threshold, dtype=jnp.int32)
    condition_number = jnp.where(
        minimum_singular_value > jnp.asarray(0.0, dtype=matrix.dtype),
        maximum_singular_value / minimum_singular_value,
        jnp.asarray(jnp.inf, dtype=matrix.dtype),
    )

    residual_l2_norm = jnp.linalg.norm(residual)
    rhs_l2_norm = jnp.linalg.norm(rhs)
    solution_l2_norm = jnp.linalg.norm(solution)
    tiny = jnp.asarray(jnp.finfo(matrix.dtype).tiny, dtype=matrix.dtype)
    normal_residual_l2_norm = jnp.linalg.norm(matrix.T @ residual)
    normal_residual_scale = maximum_singular_value * jnp.maximum(
        rhs_l2_norm,
        maximum_singular_value * solution_l2_norm,
    )
    relative_normal_residual = normal_residual_l2_norm / jnp.maximum(
        normal_residual_scale,
        tiny,
    )
    return Flat675YSolution(
        solution=solution,
        singular_values=singular_values,
        rank_threshold=rank_threshold,
        numerical_rank=numerical_rank,
        condition_number=condition_number,
        residual_l2_norm=residual_l2_norm,
        relative_fit_residual=residual_l2_norm / jnp.maximum(rhs_l2_norm, tiny),
        normal_residual_l2_norm=normal_residual_l2_norm,
        relative_normal_residual=relative_normal_residual,
        numerics_finite=(
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(rhs))
            & jnp.all(jnp.isfinite(solution))
            & jnp.all(jnp.isfinite(singular_values))
            & jnp.isfinite(condition_number)
        ),
    )


__all__ = [
    "FLAT675_Y_COLUMN_COUNT",
    "Flat675YSolution",
    "solve_flat675_y_qr",
]
