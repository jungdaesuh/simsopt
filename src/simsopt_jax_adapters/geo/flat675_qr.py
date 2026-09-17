"""Leaf QR primitive shared by flat-675 and reduced nested-LS callers.

This module deliberately lives outside :mod:`flat675`: reduced nested-LS
needs the two-column projection while the flat-675 package also imports the
polish layer that depends on reduced nested-LS. Keeping the primitive here
makes that dependency one-way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from simsopt_jax.backend.dtypes import runtime_device_put

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


@jax.jit
def _solve_flat675_y_qr_device(
    matrix: jax.Array,
    rhs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the fixed-shape QR algebra with rank scalars staged on device."""

    orthogonal, triangular = jnp.linalg.qr(matrix, mode="reduced")
    solution = jsp_linalg.solve_triangular(
        triangular,
        orthogonal.T @ rhs,
        lower=False,
    )
    singular_values = jnp.linalg.svd(triangular, compute_uv=False)

    # Constants enter at the device boundary, so an enclosing JIT never
    # materializes rank-threshold scalars on the host.
    epsilon = runtime_device_put(np.finfo(np.float64).eps, dtype=jnp.float64)
    leading_dimension = runtime_device_put(
        np.asarray(max(matrix.shape), dtype=np.float64),
        dtype=jnp.float64,
    )
    rank_threshold = leading_dimension * epsilon * singular_values[0]
    return (
        solution,
        singular_values,
        jnp.sum(singular_values > rank_threshold, dtype=jnp.int32),
        (
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(rhs))
            & jnp.all(jnp.isfinite(solution))
            & jnp.all(jnp.isfinite(singular_values))
        ),
    )


def solve_flat675_y_qr(
    design_matrix: jax.Array,
    right_hand_side: jax.Array,
) -> Flat675YSolution:
    """Solve ``min ||A y - b||`` for the two inner scalars by economy QR."""

    matrix, rhs = _validated_inputs(design_matrix, right_hand_side)
    solution, singular_values, numerical_rank, numerics_finite = (
        _solve_flat675_y_qr_device(matrix, rhs)
    )
    return Flat675YSolution(
        solution=solution,
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        numerics_finite=numerics_finite,
    )


__all__ = [
    "FLAT675_Y_COLUMN_COUNT",
    "Flat675YSolution",
    "solve_flat675_y_qr",
]
