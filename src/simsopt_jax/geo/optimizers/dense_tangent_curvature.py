"""Dense secant curvature for a tangent-space quasi-Newton step.

At a few hundred variables a dense Hessian approximation is affordable -- one
Cholesky per iteration is microseconds next to a constraint-Jacobian
materialization -- and it is the model a native dense-BFGS solve actually uses.
This module owns one piece of knowledge: how accepted curvature pairs turn into
a dense positive-definite Hessian approximation and how that approximation
turns one projected gradient into a search direction.

The stored operator is the DIRECT Hessian approximation ``B``, not its inverse,
for two reasons: Powell damping is written in ``s.B s``, which is one matrix
vector product here; and a Cholesky factorization both delivers the direction
and proves the operator is still positive definite, which a stored inverse
would only imply.

Curvature pairs are formed from tangent vectors, so every update acts inside
the tangent space and ``B`` keeps its initial scaling on the normal
directions.  That is what keeps it invertible even though the objective has no
curvature to report there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

# Powell's damping threshold: a pair is rewritten toward ``B s`` whenever its
# curvature falls below this fraction of the model's own predicted curvature,
# which keeps ``B`` positive definite without discarding the pair.
_POWELL_FLOOR = 0.2


class DenseTangentCurvature(NamedTuple):
    """A dense Hessian approximation and whether it has been scaled yet.

    ``hessian`` starts as the identity; ``scaled`` records whether the first
    accepted pair has already replaced that with the standard ``(y.y)/(s.y)``
    scaling, which is the direct-form counterpart of initializing an inverse
    Hessian with ``(s.y)/(y.y) I``.
    """

    hessian: jax.Array
    scaled: jax.Array
    damped_updates: jax.Array
    updates: jax.Array


def empty_dense_tangent_curvature(
    dimension: int,
    dtype: jnp.dtype,
) -> DenseTangentCurvature:
    """Return the identity model, which steps along the projected gradient."""

    return DenseTangentCurvature(
        hessian=jnp.eye(dimension, dtype=dtype),
        scaled=jnp.asarray(False),
        damped_updates=jnp.asarray(0, dtype=jnp.int32),
        updates=jnp.asarray(0, dtype=jnp.int32),
    )


def update_dense_tangent_curvature(
    curvature: DenseTangentCurvature,
    step: jax.Array,
    gradient_change: jax.Array,
) -> DenseTangentCurvature:
    """Apply one Powell-damped BFGS update, keeping ``B`` positive definite.

    Powell's rewrite replaces ``y`` by ``theta y + (1 - theta) B s`` with the
    largest ``theta`` that still leaves ``s.r`` at ``_POWELL_FLOOR`` of
    ``s.B s``.  An undamped pair passes through with ``theta = 1``, so this
    reduces to textbook BFGS whenever the pair is already well curved.
    """

    dtype = step.dtype
    hessian = curvature.hessian
    # Scale the identity by the first pair, the standard B0 = (y.y)/(s.y) I.
    pair_curvature = step @ gradient_change
    scale = jnp.where(
        (~curvature.scaled) & (pair_curvature > 0.0),
        (gradient_change @ gradient_change)
        / jnp.where(pair_curvature > 0.0, pair_curvature, jnp.ones((), dtype=dtype)),
        jnp.ones((), dtype=dtype),
    )
    hessian = hessian * scale

    hessian_step = hessian @ step
    model_curvature = step @ hessian_step
    needs_damping = pair_curvature < _POWELL_FLOOR * model_curvature
    denominator = model_curvature - pair_curvature
    theta = jnp.where(
        needs_damping & (denominator > 0.0),
        (1.0 - _POWELL_FLOOR)
        * model_curvature
        / jnp.where(denominator > 0.0, denominator, jnp.ones((), dtype=dtype)),
        jnp.ones((), dtype=dtype),
    )
    damped_change = theta * gradient_change + (1.0 - theta) * hessian_step
    damped_curvature = step @ damped_change

    usable = (
        (model_curvature > 0.0)
        & (damped_curvature > 0.0)
        & jnp.all(jnp.isfinite(damped_change))
    )
    safe_damped_curvature = jnp.where(
        usable, damped_curvature, jnp.ones((), dtype=dtype)
    )
    safe_model_curvature = jnp.where(usable, model_curvature, jnp.ones((), dtype=dtype))
    updated = (
        hessian
        - jnp.outer(hessian_step, hessian_step) / safe_model_curvature
        + jnp.outer(damped_change, damped_change) / safe_damped_curvature
    )
    updated = 0.5 * (updated + updated.T)
    return DenseTangentCurvature(
        hessian=jnp.where(usable, updated, hessian),
        scaled=curvature.scaled | (pair_curvature > 0.0),
        damped_updates=curvature.damped_updates
        + (usable & needs_damping).astype(jnp.int32),
        updates=curvature.updates + usable.astype(jnp.int32),
    )


class DenseTangentDirection(NamedTuple):
    """One search direction and the evidence that the model produced it."""

    direction: jax.Array
    positive_definite: jax.Array
    all_finite: jax.Array


def dense_tangent_direction(
    curvature: DenseTangentCurvature,
    projected_gradient: jax.Array,
    project: Callable[[jax.Array], jax.Array],
) -> DenseTangentDirection:
    """Return ``-P B^-1 P g``, falling back to the projected gradient if unusable.

    ``project`` maps a vector into the tangent space and is applied on the way
    out, so the direction is tangent even though ``B`` is inverted in the
    ambient space.
    The Cholesky both solves the system and certifies definiteness: a
    nonpositive diagonal means the model is no longer a metric and the caller
    gets steepest descent instead of a silently wrong step.
    """

    factor = jnp.linalg.cholesky(curvature.hessian)
    positive_definite = jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.diag(factor) > 0.0)
    safe_factor = jnp.where(
        positive_definite, factor, jnp.eye(factor.shape[0], dtype=factor.dtype)
    )
    solved = jsp.linalg.cho_solve((safe_factor, True), projected_gradient)
    direction = -project(solved)
    usable = positive_definite & jnp.all(jnp.isfinite(direction))
    return DenseTangentDirection(
        direction=jnp.where(usable, direction, -projected_gradient),
        positive_definite=positive_definite,
        all_finite=usable,
    )


__all__ = (
    "DenseTangentCurvature",
    "DenseTangentDirection",
    "dense_tangent_direction",
    "empty_dense_tangent_curvature",
    "update_dense_tangent_curvature",
)
