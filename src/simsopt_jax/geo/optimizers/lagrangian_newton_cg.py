"""Ball-free tangent Newton--CG step on the reduced Lagrangian curvature.

On a constrained manifold the curvature that governs a step is not the
objective's.  With ``lambda`` the least-squares multipliers at a feasible
point, the reduced Hessian is ``P (grad2 Phi + sum_i lambda_i grad2 c_i) P``:
the constraint-curvature term is what the manifold's own bending contributes,
and it is absent from every Gauss--Newton model of the objective alone.  When
most of the gradient lies in the constraint row space the multipliers are
large, that term dominates at Newton scale, and an objective-only model
proposes directions no line search will accept -- which is exactly what the
projected Gauss--Newton arm measured.

This module owns two pieces of knowledge: how the multipliers that define the
Lagrangian are recovered from an already factored constraint Gram, and how a
truncated conjugate-gradient solve turns matrix-free Lagrangian curvature into
a tangent search direction.  There is no trust region here at all.  A Euclidean
ball would demote the curvature model to a direction choice -- the metric arm
measured Newton-scale directions at 152-1950x a self-regulating radius -- so
the step length is decided downstream by a line search on the retracted
objective, and the only thing bounding the solve is the conjugate-gradient
forcing sequence.

The curvature operator is never materialized.  ``Hv`` is one
forward-over-reverse pass through the Lagrangian, which costs a small multiple
of a gradient evaluation, against the residual Jacobian materialization the
Gauss--Newton arm needed at 25 s per iteration.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dense_sqp import JointValueConstraints
from .projected_hvp_trust_region import (
    CertifiedGramProjector,
    HessianVectorProduct,
    project_with_certified_gram,
    solve_certified_gram,
)

# Maps one tangent residual to the search-direction seed of a preconditioned
# conjugate-gradient step.  The operator must be symmetric positive definite on
# the tangent space; whatever normal component it returns is projected away by
# the solve before the seed is used, so it need not be tangent itself.
TangentPreconditioner = Callable[[jax.Array], jax.Array]


class TangentNewtonCgTermination(IntEnum):
    """Terminal reason for one truncated tangent Newton--CG solve."""

    FORCING_SATISFIED = 0
    NEGATIVE_CURVATURE = 1
    ITERATION_LIMIT = 2
    NONFINITE = 3


class LeastSquaresMultipliers(NamedTuple):
    """Multipliers of the linearized stationarity, with their solve evidence.

    ``multipliers`` minimize ``||g + A.T lambda||`` at a feasible point, which
    is the same certified Gram solve the campaign's finalize step performs.
    """

    multipliers: jax.Array
    norm: jax.Array
    relative_residual: jax.Array
    forward_error_bound: jax.Array
    all_finite: jax.Array


class TangentNewtonCgStep(NamedTuple):
    """One reduced-Lagrangian Newton direction and the evidence behind it.

    ``direction`` is exactly zero when the very first conjugate-gradient
    iteration met negative curvature, since no iterate had yet been taken;
    ``negative_curvature_before_any_step`` reports that case so a caller can
    fall back to another model rather than test a zero step.

    ``usable`` is deliberately narrow: finite, and a step that exists.  It
    states nothing about ``tangency_relative_residual`` and nothing about the
    sign of ``Pg . d``, both of which are published beside it for a caller to
    judge.  A poorly tangent step is not unusable -- the retraction takes its
    verdict from the true constraints, so the cost of one is trials, not an
    off-manifold point -- and a step that measures as ascent is handled by the
    caller falling back to its own store, which is where the evidence about
    what to do next lives.  Folding either into ``usable`` would put a policy
    threshold with no measured band behind it inside a solve whose job is to
    report what it found.
    """

    direction: jax.Array
    direction_norm: jax.Array
    predicted_reduction: jax.Array
    forcing_eta: jax.Array
    iterations: jax.Array
    hvp_evaluations: jax.Array
    termination: jax.Array
    negative_curvature: jax.Array
    negative_curvature_before_any_step: jax.Array
    terminal_curvature: jax.Array
    terminal_normalized_curvature: jax.Array
    curvature_rayleigh_history: jax.Array
    initial_residual_norm: jax.Array
    final_residual_norm: jax.Array
    residual_target: jax.Array
    tangency_relative_residual: jax.Array
    all_finite: jax.Array
    usable: jax.Array


class _NewtonCgState(NamedTuple):
    step: jax.Array
    hessian_step: jax.Array
    residual: jax.Array
    direction: jax.Array
    residual_inner_product: jax.Array
    active: jax.Array
    iterations: jax.Array
    termination: jax.Array
    negative_curvature_before_any_step: jax.Array
    terminal_curvature: jax.Array
    terminal_normalized_curvature: jax.Array
    curvature_rayleigh_history: jax.Array
    all_finite: jax.Array


def least_squares_multipliers(
    projector: CertifiedGramProjector,
    objective_gradient: jax.Array,
) -> LeastSquaresMultipliers:
    """Recover ``lambda = -(A A.T)^-1 A g`` from an already factored Gram.

    The factorization is the caller's -- the same one its tangent projection
    spent -- so this costs one triangular solve, not a second Jacobian.
    """

    solve = solve_certified_gram(
        projector, -(projector.constraint_jacobian @ objective_gradient)
    )
    return LeastSquaresMultipliers(
        multipliers=solve.solution,
        norm=jnp.linalg.norm(solve.solution),
        relative_residual=solve.relative_residual,
        forward_error_bound=solve.forward_error_bound,
        all_finite=solve.all_finite,
    )


def lagrangian_hessian_vector_product(
    joint_value_constraints: JointValueConstraints,
    coordinates: jax.Array,
    multipliers: jax.Array,
) -> HessianVectorProduct:
    """Return the matrix-free Hessian of ``Phi + lambda.c`` at one point.

    ``multipliers`` are frozen into the returned operator: within one step the
    Lagrangian whose curvature is being modelled is a fixed scalar function, so
    the operator is the exact Hessian of that function and is symmetric.  The
    product is forward-over-reverse -- a JVP of the Lagrangian gradient --
    which traverses the primal graph twice regardless of the dimension.
    """

    def lagrangian(values: jax.Array) -> jax.Array:
        objective, constraints = joint_value_constraints(values)
        return objective + multipliers @ constraints

    gradient = jax.grad(lagrangian)

    def product(vector: jax.Array) -> jax.Array:
        return jax.jvp(gradient, (coordinates,), (vector,))[1]

    return product


def solve_tangent_newton_cg(
    hessian_vector_product: HessianVectorProduct,
    projector: CertifiedGramProjector,
    projected_gradient: jax.Array,
    *,
    maximum_iterations: int,
    forcing_maximum: float = 0.5,
    preconditioner: TangentPreconditioner | None = None,
) -> TangentNewtonCgStep:
    """Solve ``P H P d = -P g`` inside the tangent space by truncated CG.

    Termination follows an Eisenstat--Walker forcing sequence,
    ``eta = min(forcing_maximum, sqrt(||P g||))``: loose while the iterate is
    far from stationary, tightening on its own as the projected gradient
    falls, so the solve is never more exact than the outer step can use.
    Negative curvature ends the solve at the current iterate -- past that point
    the quadratic model is unbounded below and following it further is not a
    descent statement about the true objective.

    A supplied ``preconditioner`` is projected before use, so it changes the
    conjugate directions without ever taking an iterate off the tangent space.
    """

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not 0.0 < forcing_maximum < 1.0:
        raise ValueError("forcing_maximum must lie strictly inside (0, 1)")

    dtype = projected_gradient.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

    def project(vector: jax.Array) -> jax.Array:
        return project_with_certified_gram(projector, vector).projected

    def seed(residual: jax.Array) -> jax.Array:
        if preconditioner is None:
            return residual
        return project(preconditioner(residual))

    initial_residual_norm = jnp.linalg.norm(projected_gradient)
    forcing_eta = jnp.minimum(
        jnp.asarray(forcing_maximum, dtype=dtype), jnp.sqrt(initial_residual_norm)
    )
    residual_target = forcing_eta * initial_residual_norm
    initial_seed = seed(projected_gradient)
    zero_vector = jnp.zeros_like(projected_gradient)
    initial = _NewtonCgState(
        step=zero_vector,
        hessian_step=zero_vector,
        residual=projected_gradient,
        direction=-initial_seed,
        residual_inner_product=projected_gradient @ initial_seed,
        active=initial_residual_norm > residual_target,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        termination=jnp.asarray(
            int(TangentNewtonCgTermination.FORCING_SATISFIED), dtype=jnp.int32
        ),
        negative_curvature_before_any_step=jnp.asarray(False),
        terminal_curvature=jnp.zeros((), dtype=dtype),
        terminal_normalized_curvature=jnp.zeros((), dtype=dtype),
        # One Rayleigh quotient per conjugate direction actually explored.
        # Each lies between the extreme eigenvalues of the reduced Hessian, so
        # the filled prefix is a measured window onto its spectrum; entries for
        # iterations never run stay NaN.
        curvature_rayleigh_history=jnp.full(
            (maximum_iterations,), jnp.nan, dtype=dtype
        ),
        all_finite=jnp.asarray(True),
    )

    def active_iteration(state: _NewtonCgState) -> _NewtonCgState:
        hessian_direction = project(hessian_vector_product(state.direction))
        curvature = state.direction @ hessian_direction
        rayleigh = curvature / jnp.maximum(tiny, state.direction @ state.direction)
        normalized_curvature = curvature / jnp.maximum(
            tiny,
            jnp.linalg.norm(state.direction) * jnp.linalg.norm(hessian_direction),
        )
        nonpositive = ~(curvature > 0.0)
        safe_curvature = jnp.where(nonpositive, jnp.ones((), dtype=dtype), curvature)
        step_length = jnp.where(
            nonpositive,
            jnp.zeros((), dtype=dtype),
            state.residual_inner_product / safe_curvature,
        )
        next_step = state.step + step_length * state.direction
        next_hessian_step = state.hessian_step + step_length * hessian_direction
        next_residual = state.residual + step_length * hessian_direction
        next_seed = seed(next_residual)
        next_inner_product = next_residual @ next_seed
        beta = next_inner_product / jnp.where(
            state.residual_inner_product > 0.0, state.residual_inner_product, tiny
        )
        converged = jnp.linalg.norm(next_residual) <= residual_target
        finite = (
            jnp.isfinite(curvature)
            & jnp.all(jnp.isfinite(hessian_direction))
            & jnp.all(jnp.isfinite(next_step))
        )
        # A zero ``step_length`` already holds the iterate, the accumulated
        # curvature and the residual at their incoming values, so negative
        # curvature needs no separate branch to leave them alone.
        return _NewtonCgState(
            step=next_step,
            hessian_step=next_hessian_step,
            residual=next_residual,
            direction=-next_seed + beta * state.direction,
            residual_inner_product=next_inner_product,
            active=~nonpositive & ~converged & finite,
            iterations=state.iterations + 1,
            termination=jnp.where(
                nonpositive,
                int(TangentNewtonCgTermination.NEGATIVE_CURVATURE),
                jnp.where(
                    finite,
                    int(TangentNewtonCgTermination.FORCING_SATISFIED),
                    int(TangentNewtonCgTermination.NONFINITE),
                ),
            ).astype(jnp.int32),
            negative_curvature_before_any_step=(
                state.negative_curvature_before_any_step
                | (nonpositive & (state.iterations == 0))
            ),
            terminal_curvature=curvature,
            terminal_normalized_curvature=normalized_curvature,
            curvature_rayleigh_history=state.curvature_rayleigh_history.at[
                state.iterations
            ].set(rayleigh),
            all_finite=state.all_finite & finite,
        )

    def iteration(_index: int, state: _NewtonCgState) -> _NewtonCgState:
        return jax.lax.cond(
            state.active, active_iteration, lambda current: current, state
        )

    solved = jax.lax.fori_loop(0, maximum_iterations, iteration, initial)
    termination = jnp.where(
        solved.active,
        int(TangentNewtonCgTermination.ITERATION_LIMIT),
        solved.termination,
    ).astype(jnp.int32)
    direction = solved.step
    # The model decrease of the step actually returned.  ``hessian_step`` is
    # the projected curvature applied along the way, which for a tangent step
    # equals the unprojected one contracted against it.
    predicted_reduction = -(
        projected_gradient @ direction + 0.5 * (direction @ solved.hessian_step)
    )
    direction_norm = jnp.linalg.norm(direction)
    all_finite = (
        projector.all_finite
        & solved.all_finite
        & jnp.all(jnp.isfinite(direction))
        & jnp.isfinite(predicted_reduction)
    )
    tangency_relative_residual = jnp.linalg.norm(
        projector.constraint_jacobian @ direction, ord=jnp.inf
    ) / jnp.maximum(tiny, jnp.linalg.norm(direction, ord=jnp.inf))
    return TangentNewtonCgStep(
        direction=direction,
        direction_norm=direction_norm,
        predicted_reduction=predicted_reduction,
        forcing_eta=forcing_eta,
        iterations=solved.iterations,
        hvp_evaluations=solved.iterations,
        termination=termination,
        negative_curvature=(
            termination == int(TangentNewtonCgTermination.NEGATIVE_CURVATURE)
        ),
        negative_curvature_before_any_step=solved.negative_curvature_before_any_step,
        terminal_curvature=solved.terminal_curvature,
        terminal_normalized_curvature=solved.terminal_normalized_curvature,
        curvature_rayleigh_history=solved.curvature_rayleigh_history,
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=jnp.linalg.norm(solved.residual),
        residual_target=residual_target,
        tangency_relative_residual=tangency_relative_residual,
        all_finite=all_finite,
        usable=all_finite & (direction_norm > 0.0),
    )


__all__ = (
    "LeastSquaresMultipliers",
    "TangentNewtonCgStep",
    "TangentNewtonCgTermination",
    "TangentPreconditioner",
    "lagrangian_hessian_vector_product",
    "least_squares_multipliers",
    "solve_tangent_newton_cg",
)
