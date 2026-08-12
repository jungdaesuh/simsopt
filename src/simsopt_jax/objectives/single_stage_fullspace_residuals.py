"""Certification surface for fullspace Gauss--Newton objective residuals."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .single_stage_fullspace import (
    FullSpaceObjectiveResiduals,
    FullSpaceProblem,
    evaluate_fullspace_with_objective_residuals,
    flatten_fullspace_objective_residuals,
    fullspace_objective_residual_vector,
    fullspace_value_and_grad,
)


class ObjectiveResidualReconstruction(NamedTuple):
    """Value and gradient identity certificate for the residual contract."""

    reconstructed_value: jax.Array
    authoritative_value: jax.Array
    value_scaled_defect: jax.Array
    gradient_scaled_defect: jax.Array
    residual_valid: jax.Array
    all_finite: jax.Array


def _scaled_scalar_defect(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    one = jnp.asarray(1.0, dtype=lhs.dtype)
    scale = jnp.maximum(one, jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)))
    return jnp.abs(lhs - rhs) / scale


def _scaled_vector_defect(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    one = jnp.asarray(1.0, dtype=lhs.dtype)
    lhs_norm = jnp.linalg.norm(lhs, ord=jnp.inf)
    rhs_norm = jnp.linalg.norm(rhs, ord=jnp.inf)
    scale = jnp.maximum(one, jnp.maximum(lhs_norm, rhs_norm))
    return jnp.linalg.norm(lhs - rhs, ord=jnp.inf) / scale


def fullspace_objective_residuals(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> FullSpaceObjectiveResiduals:
    """Return canonical residual blocks without changing scalar authority."""

    return evaluate_fullspace_with_objective_residuals(
        z,
        problem,
    ).objective_residuals


def certify_fullspace_objective_residuals(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> ObjectiveResidualReconstruction:
    """Certify residual reconstruction of the authoritative value and gradient."""

    residual_evaluation = evaluate_fullspace_with_objective_residuals(z, problem)
    residual = residual_evaluation.objective_residual_vector
    reconstructed_value = 0.5 * jnp.vdot(residual, residual)
    authoritative_value, authoritative_gradient = fullspace_value_and_grad(z, problem)
    _value, pullback = jax.vjp(
        lambda candidate: fullspace_objective_residual_vector(candidate, problem),
        z,
    )
    reconstructed_gradient = pullback(residual)[0]
    value_scaled_defect = _scaled_scalar_defect(
        reconstructed_value,
        authoritative_value,
    )
    gradient_scaled_defect = _scaled_vector_defect(
        reconstructed_gradient,
        authoritative_gradient,
    )
    all_finite = (
        residual_evaluation.residual_valid
        & jnp.isfinite(reconstructed_value)
        & jnp.isfinite(authoritative_value)
        & jnp.all(jnp.isfinite(reconstructed_gradient))
        & jnp.all(jnp.isfinite(authoritative_gradient))
        & jnp.isfinite(value_scaled_defect)
        & jnp.isfinite(gradient_scaled_defect)
    )
    return ObjectiveResidualReconstruction(
        reconstructed_value=reconstructed_value,
        authoritative_value=authoritative_value,
        value_scaled_defect=value_scaled_defect,
        gradient_scaled_defect=gradient_scaled_defect,
        residual_valid=residual_evaluation.residual_valid,
        all_finite=all_finite,
    )


__all__ = [
    "FullSpaceObjectiveResiduals",
    "ObjectiveResidualReconstruction",
    "certify_fullspace_objective_residuals",
    "flatten_fullspace_objective_residuals",
    "fullspace_objective_residual_vector",
    "fullspace_objective_residuals",
]
