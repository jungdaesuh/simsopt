"""Tangent-space Gauss--Newton step for a small-residual objective.

For ``Phi = 0.5 ||R||^2`` with a residual that stays small near the solution,
the Gauss--Newton model ``0.5 ||J d + R||^2`` is locally almost exact -- the
projected GN trust-region route measured reduction ratios with median 0.9994
and terminal 1.0000 on this problem -- so the model deserves a FULL step.  What
that route lacked was permission to take one: its trust ball capped the step
far below Newton scale.  This module produces the step; a line search, not a
ball, decides how far to go along it.

The module owns one piece of knowledge: how a residual Jacobian and a tangent
projector become a search direction confined to the tangent space.

The subproblem solved is ``min_d ||J P d + R||^2 + lambda ||d||^2`` over the
tangent space, whose normal equations are ``(P J^T J P + lambda I) d = -P J^T
R``.  The Levenberg term is for rank safety only, sized relative to the
reduced Hessian's own diagonal, and is deliberately far too small to act as a
trust region.  Because ``P J^T J P`` annihilates normal directions and the
right-hand side is already tangent, the regularized solve returns a tangent
step; it is re-projected anyway so that tangency is certified rather than
argued.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def materialize_residual_jacobian(
    residuals: Callable[[jax.Array], jax.Array],
    coordinates: jax.Array,
    *,
    batch_width: int = 4,
) -> jax.Array:
    """Materialize ``dR/dz`` by chunked forward-mode passes.

    Forward mode costs one pass per INPUT, which is the cheaper direction here
    because the residual vector is longer than the coordinate vector.  The
    passes are chunked rather than vmapped all at once because a single vmap
    over every input holds that many sets of forward intermediates live, which
    exhausts device memory on a problem of this size.
    """

    if batch_width < 1:
        raise ValueError("batch_width must be positive")

    dimension = coordinates.shape[0]
    basis = jnp.eye(dimension, dtype=coordinates.dtype)

    def push(tangents: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda tangent: jax.jvp(residuals, (coordinates,), (tangent,))[1]
        )(tangents)

    complete_count = dimension // batch_width
    complete_width = complete_count * batch_width
    columns: list[jax.Array] = []
    if complete_count:
        chunks = basis[:complete_width].reshape(
            (complete_count, batch_width, dimension)
        )
        columns.append(jax.lax.map(push, chunks).reshape((complete_width, -1)))
    if complete_width < dimension:
        columns.append(push(basis[complete_width:]))
    return jnp.concatenate(columns, axis=0).T if len(columns) > 1 else columns[0].T


class TangentGaussNewtonStep(NamedTuple):
    """One reduced Gauss--Newton step and the evidence behind it."""

    direction: jax.Array
    predicted_reduction: jax.Array
    levenberg: jax.Array
    reduced_hessian_reciprocal_condition: jax.Array
    tangency_relative_residual: jax.Array
    positive_definite: jax.Array
    all_finite: jax.Array


def solve_tangent_gauss_newton(
    residual_jacobian: jax.Array,
    projection: jax.Array,
    projected_gradient: jax.Array,
    *,
    levenberg_relative: float,
) -> TangentGaussNewtonStep:
    """Solve the tangent-space Gauss--Newton subproblem for one direction.

    ``projected_gradient`` must already be ``P J^T R``, the tangent gradient of
    ``Phi``, which the caller has from its own point evaluation.
    ``predicted_reduction`` is the model's own decrease, ``-(g.d + 0.5 d.H d)``,
    which a caller can compare against the realized decrease to test how exact
    the model actually is.
    """

    dtype = projected_gradient.dtype
    gauss_newton_hessian = residual_jacobian.T @ residual_jacobian
    reduced = projection @ gauss_newton_hessian @ projection
    reduced = 0.5 * (reduced + reduced.T)
    dimension = reduced.shape[0]
    levenberg = jnp.asarray(levenberg_relative, dtype=dtype) * jnp.maximum(
        jnp.max(jnp.diag(reduced)), jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    )
    regularized = reduced + levenberg * jnp.eye(dimension, dtype=dtype)

    factor = jnp.linalg.cholesky(regularized)
    positive_definite = jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.diag(factor) > 0.0)
    safe_factor = jnp.where(positive_definite, factor, jnp.eye(dimension, dtype=dtype))
    solved = jsp.linalg.cho_solve((safe_factor, True), projected_gradient)
    direction = -(projection @ solved)

    # ``reduced`` is zero on the normal space, so ``levenberg`` is an eigenvalue
    # of ``regularized`` there whatever the objective does.  Taking the smallest
    # eigenvalue outright would therefore report ``levenberg / largest`` on every
    # iteration and describe the regularizer rather than the solve; the tangent
    # block is the part the direction is actually drawn from, and the projector's
    # trace is its rank.
    eigenvalues = jnp.linalg.eigvalsh(regularized)
    largest = jnp.max(eigenvalues)
    tangent_rank = jnp.round(jnp.trace(projection)).astype(jnp.int32)
    in_tangent_block = jnp.arange(dimension) >= dimension - tangent_rank
    smallest = jnp.min(jnp.where(in_tangent_block, eigenvalues, largest))
    reciprocal_condition = jnp.where(
        largest > 0.0, smallest / largest, jnp.zeros((), dtype=dtype)
    )
    # Tangency is measured against the projector itself: a tangent step is a
    # fixed point of P, so the residual of that identity is the defect.
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    tangency_relative_residual = jnp.linalg.norm(
        projection @ direction - direction, ord=jnp.inf
    ) / jnp.maximum(tiny, jnp.linalg.norm(direction, ord=jnp.inf))
    predicted_reduction = -(
        projected_gradient @ direction
        + 0.5 * (direction @ (gauss_newton_hessian @ direction))
    )
    return TangentGaussNewtonStep(
        direction=direction,
        predicted_reduction=predicted_reduction,
        levenberg=levenberg,
        reduced_hessian_reciprocal_condition=reciprocal_condition,
        tangency_relative_residual=tangency_relative_residual,
        positive_definite=positive_definite,
        all_finite=(
            positive_definite
            & jnp.all(jnp.isfinite(direction))
            & jnp.isfinite(predicted_reduction)
        ),
    )


__all__ = (
    "TangentGaussNewtonStep",
    "materialize_residual_jacobian",
    "solve_tangent_gauss_newton",
)
