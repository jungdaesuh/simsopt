"""Certified projected trust-region primitives for matrix-free Hessian actions.

The module isolates one equality-constrained curvature experiment.  It owns the
dense Gram projection and nonlinear restoration certificates while keeping the
primal Hessian action matrix-free and entirely inside JAX control flow.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .dense_sqp import JointValueConstraints, JointVJPRows, materialize_joint_vjp_rows

HessianVectorProduct = Callable[[jax.Array], jax.Array]
# Maps one tangent residual to the search-direction seed of a preconditioned
# conjugate-gradient step.  The operator must be symmetric positive definite on
# the tangent space, and must return tangent vectors.
TangentPreconditioner = Callable[[jax.Array], jax.Array]


class ProjectedSteihaugTermination(IntEnum):
    """Terminal reason for one projected Steihaug solve."""

    INTERIOR_CONVERGED = 0
    TRUST_BOUNDARY = 1
    NONPOSITIVE_CURVATURE = 2
    ITERATION_LIMIT = 3


class CertifiedGramProjector(NamedTuple):
    """Dense equality projector with an eigenvalue-based solve certificate."""

    constraint_jacobian: jax.Array
    gram_matrix: jax.Array
    cholesky_factor: jax.Array
    reciprocal_condition: jax.Array
    factorization_relative_residual: jax.Array
    all_finite: jax.Array


class CertifiedGramSolve(NamedTuple):
    """One refined solve with the factored equality Gram matrix."""

    solution: jax.Array
    relative_residual: jax.Array
    reciprocal_condition: jax.Array
    forward_error_bound: jax.Array
    all_finite: jax.Array


class CertifiedProjection(NamedTuple):
    """A null-space projection and the certificate for its Gram solve."""

    projected: jax.Array
    tangency_relative_residual: jax.Array
    solve_relative_residual: jax.Array
    solve_forward_error_bound: jax.Array
    all_finite: jax.Array


class CertifiedMinimumNormCorrection(NamedTuple):
    """Minimum-norm correction for one supplied Jacobian and residual."""

    correction: jax.Array
    linearized_constraints: jax.Array
    relative_residual: jax.Array
    reciprocal_condition: jax.Array
    forward_error_bound: jax.Array
    all_finite: jax.Array


class ProjectedSteihaugResult(NamedTuple):
    """Fixed-shape result of one matrix-free projected trust-region solve."""

    tangential_step: jax.Array
    combined_step: jax.Array
    hessian_combined_step: jax.Array
    predicted_reduction: jax.Array
    tangency_relative_residual: jax.Array
    combined_step_norm: jax.Array
    initial_projected_residual_norm: jax.Array
    final_projected_residual_norm: jax.Array
    projected_residual_target: jax.Array
    iterations: jax.Array
    hvp_evaluations: jax.Array
    termination: jax.Array
    hit_boundary: jax.Array
    encountered_nonpositive_curvature: jax.Array
    terminal_curvature: jax.Array
    terminal_normalized_curvature: jax.Array
    all_finite: jax.Array


class ProjectedHvpCanaryEndpoint(NamedTuple):
    """One corrected endpoint and its curvature and linear-solve evidence."""

    coordinates: jax.Array
    multipliers: jax.Array
    objective: jax.Array
    constraints: jax.Array
    stationarity: jax.Array
    scaled_stationarity_inf: jax.Array
    scaled_feasibility_inf: jax.Array
    tangent_step: jax.Array
    tangent_step_norm: jax.Array
    model_step_norm: jax.Array
    applied_step: jax.Array
    applied_step_norm: jax.Array
    predicted_reduction: jax.Array
    tangency_relative_residual: jax.Array
    cg_iterations: jax.Array
    cg_termination: jax.Array
    cg_hit_boundary: jax.Array
    cg_negative_curvature: jax.Array
    cg_hvp_evaluations: jax.Array
    cg_initial_projected_residual_norm: jax.Array
    cg_final_projected_residual_norm: jax.Array
    cg_projected_residual_target: jax.Array
    correction: jax.Array
    correction_relative_residual: jax.Array
    correction_forward_error_bound: jax.Array
    multiplier_projection_relative_residual: jax.Array
    multiplier_projection_forward_error_bound: jax.Array
    usable: jax.Array
    all_finite: jax.Array


class ProjectedHvpCanaryResult(NamedTuple):
    """Paired identity/exact-HVP one-step curvature diagnostic."""

    initial: ProjectedHvpCanaryEndpoint
    identity: ProjectedHvpCanaryEndpoint
    exact: ProjectedHvpCanaryEndpoint
    exact_hvp_bilinear_symmetry_relative_defect: jax.Array
    both_variants_usable: jax.Array
    exact_hvp_supported: jax.Array
    all_finite: jax.Array


class ProjectedCurvatureCanaryResult(NamedTuple):
    """Paired identity/candidate-HVP one-step curvature diagnostic."""

    initial: ProjectedHvpCanaryEndpoint
    identity: ProjectedHvpCanaryEndpoint
    candidate: ProjectedHvpCanaryEndpoint
    candidate_hvp_bilinear_symmetry_relative_defect: jax.Array
    candidate_terminal_normalized_curvature: jax.Array
    candidate_valid: jax.Array
    both_variants_usable: jax.Array
    candidate_supported: jax.Array
    all_finite: jax.Array


class _SteihaugState(NamedTuple):
    tangential_step: jax.Array
    hessian_tangential_step: jax.Array
    residual: jax.Array
    direction: jax.Array
    active: jax.Array
    iterations: jax.Array
    hvp_evaluations: jax.Array
    termination: jax.Array
    terminal_curvature: jax.Array
    terminal_normalized_curvature: jax.Array


def _relative_inf_residual(
    matrix: jax.Array,
    solution: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    residual = matrix @ solution - right_hand_side
    tiny = jnp.asarray(jnp.finfo(matrix.dtype).tiny, dtype=matrix.dtype)
    denominator = jnp.maximum(
        tiny,
        jnp.linalg.norm(matrix, ord=jnp.inf) * jnp.linalg.norm(solution, ord=jnp.inf)
        + jnp.linalg.norm(right_hand_side, ord=jnp.inf),
    )
    return jnp.linalg.norm(residual, ord=jnp.inf) / denominator


def _forward_error_bound(
    relative_residual: jax.Array,
    reciprocal_condition: jax.Array,
) -> jax.Array:
    return jnp.where(
        reciprocal_condition > relative_residual,
        relative_residual / (reciprocal_condition - relative_residual),
        jnp.asarray(jnp.inf, dtype=relative_residual.dtype),
    )


def factor_certified_gram_projector(
    constraint_jacobian: jax.Array,
) -> CertifiedGramProjector:
    """Factor ``A A.T`` and record rank and reconstruction evidence."""

    gram_matrix = constraint_jacobian @ constraint_jacobian.T
    gram_matrix = 0.5 * (gram_matrix + gram_matrix.T)
    cholesky_factor = jnp.linalg.cholesky(gram_matrix)
    eigenvalues = jnp.linalg.eigvalsh(gram_matrix)
    largest_eigenvalue = jnp.max(eigenvalues)
    reciprocal_condition = jnp.where(
        largest_eigenvalue > 0.0,
        jnp.min(eigenvalues) / largest_eigenvalue,
        jnp.asarray(0.0, dtype=gram_matrix.dtype),
    )
    reconstructed = cholesky_factor @ cholesky_factor.T
    tiny = jnp.asarray(jnp.finfo(gram_matrix.dtype).tiny, dtype=gram_matrix.dtype)
    factorization_relative_residual = jnp.linalg.norm(
        reconstructed - gram_matrix, ord=jnp.inf
    ) / jnp.maximum(tiny, jnp.linalg.norm(gram_matrix, ord=jnp.inf))
    all_finite = (
        jnp.all(jnp.isfinite(cholesky_factor))
        & jnp.all(jnp.diag(cholesky_factor) > 0.0)
        & jnp.all(jnp.isfinite(eigenvalues))
        & jnp.isfinite(reciprocal_condition)
        & (reciprocal_condition > 0.0)
        & jnp.isfinite(factorization_relative_residual)
    )
    return CertifiedGramProjector(
        constraint_jacobian=constraint_jacobian,
        gram_matrix=gram_matrix,
        cholesky_factor=cholesky_factor,
        reciprocal_condition=reciprocal_condition,
        factorization_relative_residual=factorization_relative_residual,
        all_finite=all_finite,
    )


def solve_certified_gram(
    projector: CertifiedGramProjector,
    right_hand_side: jax.Array,
) -> CertifiedGramSolve:
    """Solve one Gram system with one refinement and a forward-error bound."""

    solution = jsp.linalg.cho_solve((projector.cholesky_factor, True), right_hand_side)
    first_residual = right_hand_side - projector.gram_matrix @ solution
    solution = solution + jsp.linalg.cho_solve(
        (projector.cholesky_factor, True), first_residual
    )
    relative_residual = _relative_inf_residual(
        projector.gram_matrix, solution, right_hand_side
    )
    forward_error_bound = _forward_error_bound(
        relative_residual, projector.reciprocal_condition
    )
    all_finite = (
        projector.all_finite
        & jnp.all(jnp.isfinite(solution))
        & jnp.isfinite(relative_residual)
        & jnp.isfinite(forward_error_bound)
    )
    return CertifiedGramSolve(
        solution=solution,
        relative_residual=relative_residual,
        reciprocal_condition=projector.reciprocal_condition,
        forward_error_bound=forward_error_bound,
        all_finite=all_finite,
    )


def project_with_certified_gram(
    projector: CertifiedGramProjector,
    vector: jax.Array,
) -> CertifiedProjection:
    """Project a vector into ``null(A)`` and certify the Gram solve."""

    solve = solve_certified_gram(projector, projector.constraint_jacobian @ vector)
    projected = vector - projector.constraint_jacobian.T @ solve.solution
    tiny = jnp.asarray(jnp.finfo(vector.dtype).tiny, dtype=vector.dtype)
    tangency_relative_residual = jnp.linalg.norm(
        projector.constraint_jacobian @ projected, ord=jnp.inf
    ) / jnp.maximum(tiny, jnp.linalg.norm(projected, ord=jnp.inf))
    all_finite = (
        solve.all_finite
        & jnp.all(jnp.isfinite(projected))
        & jnp.isfinite(tangency_relative_residual)
    )
    return CertifiedProjection(
        projected=projected,
        tangency_relative_residual=tangency_relative_residual,
        solve_relative_residual=solve.relative_residual,
        solve_forward_error_bound=solve.forward_error_bound,
        all_finite=all_finite,
    )


def certified_correction_with_projector(
    projector: CertifiedGramProjector,
    constraints: jax.Array,
) -> CertifiedMinimumNormCorrection:
    """Return the minimum-norm solution of ``A delta = -c`` for a factored ``A``.

    ``constraints`` need not have been evaluated at the point where ``A`` was
    materialized.  When it was not, this is a chord step -- normal to the rows
    of the frozen ``A`` rather than to the manifold at the current point -- and
    ``linearized_constraints`` reports the residual the frozen rows predict.
    """

    constraint_jacobian = projector.constraint_jacobian
    solve = solve_certified_gram(projector, -constraints)
    correction = constraint_jacobian.T @ solve.solution
    linearized_constraints = constraints + constraint_jacobian @ correction
    tiny = jnp.asarray(jnp.finfo(constraints.dtype).tiny, dtype=constraints.dtype)
    relative_residual = jnp.linalg.norm(
        linearized_constraints, ord=jnp.inf
    ) / jnp.maximum(
        tiny,
        jnp.linalg.norm(constraint_jacobian, ord=jnp.inf)
        * jnp.linalg.norm(correction, ord=jnp.inf)
        + jnp.linalg.norm(constraints, ord=jnp.inf),
    )
    all_finite = (
        solve.all_finite
        & jnp.all(jnp.isfinite(correction))
        & jnp.all(jnp.isfinite(linearized_constraints))
        & jnp.isfinite(relative_residual)
    )
    return CertifiedMinimumNormCorrection(
        correction=correction,
        linearized_constraints=linearized_constraints,
        relative_residual=relative_residual,
        reciprocal_condition=solve.reciprocal_condition,
        forward_error_bound=solve.forward_error_bound,
        all_finite=all_finite,
    )


def certified_minimum_norm_correction(
    constraint_jacobian: jax.Array,
    constraints: jax.Array,
) -> CertifiedMinimumNormCorrection:
    """Return the certified minimum-norm solution of ``A delta = -c``."""

    return certified_correction_with_projector(
        factor_certified_gram_projector(constraint_jacobian), constraints
    )


def _boundary_root(
    offset: jax.Array,
    direction: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    quadratic = direction @ direction
    linear = 2.0 * (offset @ direction)
    constant = offset @ offset - radius * radius
    discriminant = jnp.maximum(linear * linear - 4.0 * quadratic * constant, 0.0)
    safe_quadratic = jnp.where(quadratic > 0.0, quadratic, 1.0)
    return jnp.maximum((-linear + jnp.sqrt(discriminant)) / (2.0 * safe_quadratic), 0.0)


def solve_projected_steihaug(
    hessian_vector_product: HessianVectorProduct,
    lagrangian_gradient: jax.Array,
    normal_step: jax.Array,
    projector: CertifiedGramProjector,
    *,
    trust_radius: float,
    maximum_iterations: int,
    projected_residual_tolerance: float,
    preconditioner: TangentPreconditioner | None = None,
) -> ProjectedSteihaugResult:
    """Solve one equality-projected trust subproblem using callable HVPs.

    The trust region is always the Euclidean ball, so a supplied preconditioner
    changes only the conjugate-gradient search directions and leaves every step
    norm, boundary root, and termination certificate measured as before.
    """

    seed_direction = (
        (lambda residual: residual) if preconditioner is None else preconditioner
    )
    radius = jnp.asarray(trust_radius, dtype=lagrangian_gradient.dtype)
    zero_vector = jnp.zeros_like(normal_step)
    normal_is_nonzero = jnp.linalg.norm(normal_step) > 0.0
    hessian_normal_step, normal_hvp_evaluations = jax.lax.cond(
        normal_is_nonzero,
        lambda step: (
            hessian_vector_product(step),
            jnp.asarray(1, dtype=jnp.int32),
        ),
        lambda step: (jnp.zeros_like(step), jnp.asarray(0, dtype=jnp.int32)),
        normal_step,
    )
    initial_projection = project_with_certified_gram(
        projector, lagrangian_gradient + hessian_normal_step
    )
    initial_residual = initial_projection.projected
    initial_residual_norm = jnp.linalg.norm(initial_residual)
    residual_target = jnp.asarray(
        projected_residual_tolerance, dtype=lagrangian_gradient.dtype
    ) * jnp.maximum(
        jnp.asarray(1.0, dtype=lagrangian_gradient.dtype), initial_residual_norm
    )
    initially_active = initial_residual_norm > residual_target
    initial = _SteihaugState(
        tangential_step=zero_vector,
        hessian_tangential_step=zero_vector,
        residual=initial_residual,
        direction=-seed_direction(initial_residual),
        active=initially_active,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        hvp_evaluations=normal_hvp_evaluations,
        termination=jnp.asarray(
            int(ProjectedSteihaugTermination.INTERIOR_CONVERGED), dtype=jnp.int32
        ),
        terminal_curvature=jnp.asarray(0.0, dtype=lagrangian_gradient.dtype),
        terminal_normalized_curvature=jnp.asarray(0.0, dtype=lagrangian_gradient.dtype),
    )

    def active_iteration(state: _SteihaugState) -> _SteihaugState:
        hessian_direction = hessian_vector_product(state.direction)
        projected_hessian_direction = project_with_certified_gram(
            projector, hessian_direction
        ).projected
        curvature = state.direction @ projected_hessian_direction
        curvature_scale = jnp.maximum(
            jnp.asarray(jnp.finfo(lagrangian_gradient.dtype).tiny),
            jnp.linalg.norm(state.direction)
            * jnp.linalg.norm(projected_hessian_direction),
        )
        normalized_curvature = curvature / curvature_scale
        nonpositive_curvature = curvature <= 0.0
        residual_inner_product = state.residual @ seed_direction(state.residual)
        safe_curvature = jnp.where(nonpositive_curvature, 1.0, curvature)
        alpha = residual_inner_product / safe_curvature
        unconstrained_step = state.tangential_step + alpha * state.direction
        crosses_boundary = jnp.linalg.norm(normal_step + unconstrained_step) >= radius
        hits_boundary = nonpositive_curvature | crosses_boundary
        tau = _boundary_root(
            normal_step + state.tangential_step, state.direction, radius
        )
        step_length = jnp.where(hits_boundary, tau, alpha)
        next_step = state.tangential_step + step_length * state.direction
        next_hessian_step = (
            state.hessian_tangential_step + step_length * hessian_direction
        )
        next_residual = state.residual + step_length * projected_hessian_direction
        next_seed = seed_direction(next_residual)
        next_residual_inner_product = next_residual @ next_seed
        # Without a preconditioner the conjugate-gradient inner product is the
        # squared residual norm, so the convergence test reuses that same
        # contraction rather than emitting a second one.
        next_residual_squared = (
            next_residual_inner_product
            if preconditioner is None
            else next_residual @ next_residual
        )
        beta = next_residual_inner_product / jnp.maximum(
            residual_inner_product,
            jnp.asarray(jnp.finfo(lagrangian_gradient.dtype).tiny),
        )
        next_direction = -next_seed + beta * state.direction
        converged = jnp.sqrt(next_residual_squared) <= residual_target
        next_active = ~hits_boundary & ~converged
        termination = jnp.where(
            nonpositive_curvature,
            int(ProjectedSteihaugTermination.NONPOSITIVE_CURVATURE),
            jnp.where(
                crosses_boundary,
                int(ProjectedSteihaugTermination.TRUST_BOUNDARY),
                int(ProjectedSteihaugTermination.INTERIOR_CONVERGED),
            ),
        )
        return _SteihaugState(
            tangential_step=next_step,
            hessian_tangential_step=next_hessian_step,
            residual=next_residual,
            direction=next_direction,
            active=next_active,
            iterations=state.iterations + 1,
            hvp_evaluations=state.hvp_evaluations + 1,
            termination=jnp.asarray(termination, dtype=jnp.int32),
            terminal_curvature=curvature,
            terminal_normalized_curvature=normalized_curvature,
        )

    def iteration(_index: int, state: _SteihaugState) -> _SteihaugState:
        return jax.lax.cond(
            state.active, active_iteration, lambda current: current, state
        )

    solved = jax.lax.fori_loop(0, maximum_iterations, iteration, initial)
    termination = jnp.where(
        solved.active,
        int(ProjectedSteihaugTermination.ITERATION_LIMIT),
        solved.termination,
    ).astype(jnp.int32)
    tangential_step = solved.tangential_step
    combined_step = normal_step + tangential_step
    hessian_combined_step = hessian_normal_step + solved.hessian_tangential_step
    predicted_reduction = -(
        lagrangian_gradient @ combined_step
        + 0.5 * (combined_step @ hessian_combined_step)
    )
    final_projected_residual = project_with_certified_gram(
        projector, lagrangian_gradient + hessian_combined_step
    ).projected
    final_projected_residual_norm = jnp.linalg.norm(final_projected_residual)
    combined_step_norm = jnp.linalg.norm(combined_step)
    tangency_relative_residual = jnp.linalg.norm(
        projector.constraint_jacobian @ tangential_step, ord=jnp.inf
    ) / jnp.maximum(
        jnp.asarray(jnp.finfo(tangential_step.dtype).tiny),
        jnp.linalg.norm(tangential_step, ord=jnp.inf),
    )
    hit_boundary = (termination == int(ProjectedSteihaugTermination.TRUST_BOUNDARY)) | (
        termination == int(ProjectedSteihaugTermination.NONPOSITIVE_CURVATURE)
    )
    all_finite = (
        projector.all_finite
        & initial_projection.all_finite
        & jnp.all(jnp.isfinite(tangential_step))
        & jnp.all(jnp.isfinite(combined_step))
        & jnp.all(jnp.isfinite(hessian_combined_step))
        & jnp.isfinite(predicted_reduction)
        & jnp.isfinite(final_projected_residual_norm)
        & jnp.isfinite(solved.terminal_curvature)
        & jnp.isfinite(solved.terminal_normalized_curvature)
    )
    return ProjectedSteihaugResult(
        tangential_step=tangential_step,
        combined_step=combined_step,
        hessian_combined_step=hessian_combined_step,
        predicted_reduction=predicted_reduction,
        tangency_relative_residual=tangency_relative_residual,
        combined_step_norm=combined_step_norm,
        initial_projected_residual_norm=initial_residual_norm,
        final_projected_residual_norm=final_projected_residual_norm,
        projected_residual_target=residual_target,
        iterations=solved.iterations,
        hvp_evaluations=solved.hvp_evaluations,
        termination=termination,
        hit_boundary=hit_boundary,
        encountered_nonpositive_curvature=(
            termination == int(ProjectedSteihaugTermination.NONPOSITIVE_CURVATURE)
        ),
        terminal_curvature=solved.terminal_curvature,
        terminal_normalized_curvature=solved.terminal_normalized_curvature,
        all_finite=all_finite,
    )


def exact_hvp_bilinear_symmetry_relative_defect(
    hessian_vector_product: HessianVectorProduct,
    coordinates: jax.Array,
) -> jax.Array:
    """Measure HVP bilinear symmetry with two deterministic dense probes."""

    indices = jnp.arange(1, coordinates.shape[0] + 1, dtype=coordinates.dtype)
    first = jnp.sin(indices)
    second = jnp.cos(indices * jnp.asarray(0.5, dtype=coordinates.dtype))
    first = first / jnp.linalg.norm(first)
    second = second / jnp.linalg.norm(second)
    hessian_first = hessian_vector_product(first)
    hessian_second = hessian_vector_product(second)
    first_action = first @ hessian_second
    second_action = second @ hessian_first
    denominator = jnp.maximum(
        jnp.asarray(jnp.finfo(coordinates.dtype).tiny),
        jnp.linalg.norm(first) * jnp.linalg.norm(hessian_second)
        + jnp.linalg.norm(second) * jnp.linalg.norm(hessian_first),
    )
    return jnp.abs(first_action - second_action) / denominator


def _project_multipliers(
    objective_gradient: jax.Array,
    projector: CertifiedGramProjector,
) -> CertifiedGramSolve:
    return solve_certified_gram(
        projector,
        -(projector.constraint_jacobian @ objective_gradient),
    )


def _endpoint_from_step(
    joint_value_constraints: JointValueConstraints,
    initial_coordinates: jax.Array,
    steihaug: ProjectedSteihaugResult,
    *,
    trust_radius: float,
    feasibility_tolerance: float,
    linear_residual_tolerance: float,
) -> ProjectedHvpCanaryEndpoint:
    trial_coordinates = initial_coordinates + steihaug.combined_step
    trial_rows = materialize_joint_vjp_rows(joint_value_constraints, trial_coordinates)
    correction = certified_minimum_norm_correction(
        trial_rows.constraint_jacobian, trial_rows.constraints
    )
    endpoint_coordinates = trial_coordinates + correction.correction
    endpoint_rows = materialize_joint_vjp_rows(
        joint_value_constraints, endpoint_coordinates
    )
    endpoint_projector = factor_certified_gram_projector(
        endpoint_rows.constraint_jacobian
    )
    multiplier_projection = _project_multipliers(
        endpoint_rows.objective_gradient, endpoint_projector
    )
    stationarity = (
        endpoint_rows.objective_gradient
        + endpoint_rows.constraint_jacobian.T @ multiplier_projection.solution
    )
    scaled_stationarity_inf = jnp.linalg.norm(stationarity, ord=jnp.inf)
    scaled_feasibility_inf = jnp.linalg.norm(endpoint_rows.constraints, ord=jnp.inf)
    applied_step = steihaug.combined_step + correction.correction
    all_finite = (
        steihaug.all_finite
        & correction.all_finite
        & multiplier_projection.all_finite
        & jnp.isfinite(endpoint_rows.objective)
        & jnp.all(jnp.isfinite(endpoint_rows.constraints))
        & jnp.all(jnp.isfinite(endpoint_rows.objective_gradient))
        & jnp.all(jnp.isfinite(endpoint_rows.constraint_jacobian))
        & jnp.all(jnp.isfinite(stationarity))
        & jnp.isfinite(scaled_stationarity_inf)
        & jnp.isfinite(scaled_feasibility_inf)
        & jnp.all(jnp.isfinite(applied_step))
    )
    accepted_termination = steihaug.termination != int(
        ProjectedSteihaugTermination.ITERATION_LIMIT
    )
    radius = jnp.asarray(trust_radius, dtype=endpoint_rows.objective.dtype)
    radius_tolerance = (
        jnp.asarray(64.0, dtype=endpoint_rows.objective.dtype)
        * jnp.finfo(endpoint_rows.objective.dtype).eps
        * jnp.maximum(jnp.asarray(1.0, dtype=endpoint_rows.objective.dtype), radius)
    )
    step_within_radius = steihaug.combined_step_norm <= radius + radius_tolerance
    boundary_radius_valid = (~steihaug.hit_boundary) | (
        jnp.abs(steihaug.combined_step_norm - radius) <= radius_tolerance
    )
    interior_residual_valid = (
        steihaug.termination != int(ProjectedSteihaugTermination.INTERIOR_CONVERGED)
    ) | (steihaug.final_projected_residual_norm <= steihaug.projected_residual_target)
    usable = (
        all_finite
        & accepted_termination
        & step_within_radius
        & boundary_radius_valid
        & interior_residual_valid
        & (steihaug.predicted_reduction > 0.0)
        & (steihaug.tangency_relative_residual <= linear_residual_tolerance)
        & (correction.relative_residual <= linear_residual_tolerance)
        & (correction.forward_error_bound < 1.0e-7)
        & (multiplier_projection.relative_residual <= linear_residual_tolerance)
        & (multiplier_projection.forward_error_bound < 1.0e-7)
        & (scaled_feasibility_inf <= feasibility_tolerance)
    )
    return ProjectedHvpCanaryEndpoint(
        coordinates=endpoint_coordinates,
        multipliers=multiplier_projection.solution,
        objective=endpoint_rows.objective,
        constraints=endpoint_rows.constraints,
        stationarity=stationarity,
        scaled_stationarity_inf=scaled_stationarity_inf,
        scaled_feasibility_inf=scaled_feasibility_inf,
        tangent_step=steihaug.tangential_step,
        tangent_step_norm=jnp.linalg.norm(steihaug.tangential_step),
        model_step_norm=steihaug.combined_step_norm,
        applied_step=applied_step,
        applied_step_norm=jnp.linalg.norm(applied_step),
        predicted_reduction=steihaug.predicted_reduction,
        tangency_relative_residual=steihaug.tangency_relative_residual,
        cg_iterations=steihaug.iterations,
        cg_termination=steihaug.termination,
        cg_hit_boundary=steihaug.hit_boundary,
        cg_negative_curvature=steihaug.encountered_nonpositive_curvature,
        cg_hvp_evaluations=steihaug.hvp_evaluations,
        cg_initial_projected_residual_norm=(steihaug.initial_projected_residual_norm),
        cg_final_projected_residual_norm=steihaug.final_projected_residual_norm,
        cg_projected_residual_target=steihaug.projected_residual_target,
        correction=correction.correction,
        correction_relative_residual=correction.relative_residual,
        correction_forward_error_bound=correction.forward_error_bound,
        multiplier_projection_relative_residual=(
            multiplier_projection.relative_residual
        ),
        multiplier_projection_forward_error_bound=(
            multiplier_projection.forward_error_bound
        ),
        usable=usable,
        all_finite=all_finite,
    )


def _initial_endpoint(
    rows: JointVJPRows,
    multipliers: CertifiedGramSolve,
) -> ProjectedHvpCanaryEndpoint:
    stationarity = (
        rows.objective_gradient + rows.constraint_jacobian.T @ multipliers.solution
    )
    zero_vector = jnp.zeros_like(rows.objective_gradient)
    zero = jnp.asarray(0.0, dtype=rows.objective.dtype)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    scaled_stationarity_inf = jnp.linalg.norm(stationarity, ord=jnp.inf)
    scaled_feasibility_inf = jnp.linalg.norm(rows.constraints, ord=jnp.inf)
    all_finite = (
        multipliers.all_finite
        & jnp.isfinite(rows.objective)
        & jnp.all(jnp.isfinite(rows.constraints))
        & jnp.all(jnp.isfinite(rows.objective_gradient))
        & jnp.all(jnp.isfinite(rows.constraint_jacobian))
        & jnp.all(jnp.isfinite(stationarity))
    )
    return ProjectedHvpCanaryEndpoint(
        coordinates=jnp.zeros_like(rows.objective_gradient),
        multipliers=multipliers.solution,
        objective=rows.objective,
        constraints=rows.constraints,
        stationarity=stationarity,
        scaled_stationarity_inf=scaled_stationarity_inf,
        scaled_feasibility_inf=scaled_feasibility_inf,
        tangent_step=zero_vector,
        tangent_step_norm=zero,
        model_step_norm=zero,
        applied_step=zero_vector,
        applied_step_norm=zero,
        predicted_reduction=zero,
        tangency_relative_residual=zero,
        cg_iterations=zero_int,
        cg_termination=jnp.asarray(
            int(ProjectedSteihaugTermination.INTERIOR_CONVERGED), dtype=jnp.int32
        ),
        cg_hit_boundary=jnp.asarray(False),
        cg_negative_curvature=jnp.asarray(False),
        cg_hvp_evaluations=zero_int,
        cg_initial_projected_residual_norm=scaled_stationarity_inf,
        cg_final_projected_residual_norm=scaled_stationarity_inf,
        cg_projected_residual_target=zero,
        correction=zero_vector,
        correction_relative_residual=zero,
        correction_forward_error_bound=zero,
        multiplier_projection_relative_residual=multipliers.relative_residual,
        multiplier_projection_forward_error_bound=multipliers.forward_error_bound,
        usable=all_finite,
        all_finite=all_finite,
    )


def _validate_canary_options(
    *,
    trust_radius: float,
    maximum_iterations: int,
    projected_residual_tolerance: float,
    feasibility_tolerance: float,
    linear_residual_tolerance: float,
) -> None:
    if not math.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("trust_radius must be finite and positive")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    tolerances = (
        projected_residual_tolerance,
        feasibility_tolerance,
        linear_residual_tolerance,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("canary tolerances must be finite and nonnegative")


def _run_projected_curvature_canary(
    joint_value_constraints: JointValueConstraints,
    initial_coordinates: jax.Array,
    initial_rows: JointVJPRows,
    initial_projector: CertifiedGramProjector,
    initial_multipliers: CertifiedGramSolve,
    lagrangian_gradient: jax.Array,
    candidate_hvp: HessianVectorProduct,
    candidate_valid: jax.Array | bool,
    *,
    trust_radius: float,
    maximum_iterations: int,
    projected_residual_tolerance: float,
    feasibility_tolerance: float,
    linear_residual_tolerance: float,
) -> ProjectedCurvatureCanaryResult:
    symmetry_defect = exact_hvp_bilinear_symmetry_relative_defect(
        candidate_hvp, initial_coordinates
    )
    normal_step = jnp.zeros_like(initial_coordinates)
    identity_hvp = lambda vector: vector
    identity_step = solve_projected_steihaug(
        identity_hvp,
        lagrangian_gradient,
        normal_step,
        initial_projector,
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
        projected_residual_tolerance=projected_residual_tolerance,
    )
    candidate_step = solve_projected_steihaug(
        candidate_hvp,
        lagrangian_gradient,
        normal_step,
        initial_projector,
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
        projected_residual_tolerance=projected_residual_tolerance,
    )
    initial = _initial_endpoint(initial_rows, initial_multipliers)._replace(
        coordinates=initial_coordinates
    )
    identity = _endpoint_from_step(
        joint_value_constraints,
        initial_coordinates,
        identity_step,
        trust_radius=trust_radius,
        feasibility_tolerance=feasibility_tolerance,
        linear_residual_tolerance=linear_residual_tolerance,
    )
    candidate = _endpoint_from_step(
        joint_value_constraints,
        initial_coordinates,
        candidate_step,
        trust_radius=trust_radius,
        feasibility_tolerance=feasibility_tolerance,
        linear_residual_tolerance=linear_residual_tolerance,
    )
    candidate_valid_array = jnp.asarray(candidate_valid, dtype=jnp.bool_)
    candidate_hvp_valid = (
        candidate_valid_array
        & jnp.isfinite(symmetry_defect)
        & (symmetry_defect <= 1.0e-10)
        & initial_projector.all_finite
        & (
            initial_projector.factorization_relative_residual
            <= linear_residual_tolerance
        )
        & initial_multipliers.all_finite
        & (initial_multipliers.relative_residual <= linear_residual_tolerance)
        & (initial_multipliers.forward_error_bound < 1.0e-7)
    )
    both_variants_usable = identity.usable & candidate.usable & candidate_hvp_valid
    candidate_supported = (
        both_variants_usable
        & (candidate.scaled_stationarity_inf <= 0.5 * initial.scaled_stationarity_inf)
        & (candidate.scaled_stationarity_inf <= 0.5 * identity.scaled_stationarity_inf)
    )
    all_finite = (
        initial.all_finite
        & identity.all_finite
        & candidate.all_finite
        & jnp.isfinite(symmetry_defect)
    )
    return ProjectedCurvatureCanaryResult(
        initial=initial,
        identity=identity,
        candidate=candidate,
        candidate_hvp_bilinear_symmetry_relative_defect=symmetry_defect,
        candidate_terminal_normalized_curvature=(
            candidate_step.terminal_normalized_curvature
        ),
        candidate_valid=candidate_valid_array,
        both_variants_usable=both_variants_usable,
        candidate_supported=candidate_supported,
        all_finite=all_finite,
    )


def run_projected_curvature_canary(
    joint_value_constraints: JointValueConstraints,
    initial_coordinates: jax.Array,
    candidate_hvp: HessianVectorProduct,
    *,
    candidate_valid: jax.Array | bool,
    trust_radius: float = 2.0**-10,
    maximum_iterations: int = 32,
    projected_residual_tolerance: float = 1.0e-10,
    feasibility_tolerance: float = 1.0e-10,
    linear_residual_tolerance: float = 1.0e-10,
) -> ProjectedCurvatureCanaryResult:
    """Compare identity and a named-by-caller candidate HVP at one state."""

    _validate_canary_options(
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
        projected_residual_tolerance=projected_residual_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        linear_residual_tolerance=linear_residual_tolerance,
    )
    initial_rows = materialize_joint_vjp_rows(
        joint_value_constraints, initial_coordinates
    )
    initial_projector = factor_certified_gram_projector(
        initial_rows.constraint_jacobian
    )
    initial_multipliers = _project_multipliers(
        initial_rows.objective_gradient, initial_projector
    )
    lagrangian_gradient = (
        initial_rows.objective_gradient
        + initial_rows.constraint_jacobian.T @ initial_multipliers.solution
    )

    return _run_projected_curvature_canary(
        joint_value_constraints,
        initial_coordinates,
        initial_rows,
        initial_projector,
        initial_multipliers,
        lagrangian_gradient,
        candidate_hvp,
        candidate_valid,
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
        projected_residual_tolerance=projected_residual_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        linear_residual_tolerance=linear_residual_tolerance,
    )


def run_projected_hvp_canary(
    joint_value_constraints: JointValueConstraints,
    initial_coordinates: jax.Array,
    *,
    trust_radius: float = 2.0**-10,
    maximum_iterations: int = 32,
    projected_residual_tolerance: float = 1.0e-10,
    feasibility_tolerance: float = 1.0e-10,
    linear_residual_tolerance: float = 1.0e-10,
) -> ProjectedHvpCanaryResult:
    """Compare identity and exact-HVP projected steps at one shared state."""

    _validate_canary_options(
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
        projected_residual_tolerance=projected_residual_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        linear_residual_tolerance=linear_residual_tolerance,
    )
    initial_rows = materialize_joint_vjp_rows(
        joint_value_constraints, initial_coordinates
    )
    initial_projector = factor_certified_gram_projector(
        initial_rows.constraint_jacobian
    )
    initial_multipliers = _project_multipliers(
        initial_rows.objective_gradient, initial_projector
    )
    lagrangian_gradient = (
        initial_rows.objective_gradient
        + initial_rows.constraint_jacobian.T @ initial_multipliers.solution
    )

    fixed_multipliers = initial_multipliers.solution

    def lagrangian(coordinates: jax.Array) -> jax.Array:
        objective, constraints = joint_value_constraints(coordinates)
        return objective + jnp.vdot(fixed_multipliers, constraints)

    _gradient, exact_hvp = jax.linearize(jax.grad(lagrangian), initial_coordinates)
    generic_result = _run_projected_curvature_canary(
        joint_value_constraints,
        initial_coordinates,
        initial_rows,
        initial_projector,
        initial_multipliers,
        lagrangian_gradient,
        exact_hvp,
        jnp.asarray(True),
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
        projected_residual_tolerance=projected_residual_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        linear_residual_tolerance=linear_residual_tolerance,
    )
    return ProjectedHvpCanaryResult(
        initial=generic_result.initial,
        identity=generic_result.identity,
        exact=generic_result.candidate,
        exact_hvp_bilinear_symmetry_relative_defect=(
            generic_result.candidate_hvp_bilinear_symmetry_relative_defect
        ),
        both_variants_usable=generic_result.both_variants_usable,
        exact_hvp_supported=generic_result.candidate_supported,
        all_finite=generic_result.all_finite,
    )


__all__ = (
    "CertifiedGramProjector",
    "CertifiedGramSolve",
    "CertifiedMinimumNormCorrection",
    "CertifiedProjection",
    "ProjectedCurvatureCanaryResult",
    "ProjectedHvpCanaryEndpoint",
    "ProjectedHvpCanaryResult",
    "ProjectedSteihaugResult",
    "ProjectedSteihaugTermination",
    "TangentPreconditioner",
    "certified_correction_with_projector",
    "certified_minimum_norm_correction",
    "exact_hvp_bilinear_symmetry_relative_defect",
    "factor_certified_gram_projector",
    "project_with_certified_gram",
    "run_projected_curvature_canary",
    "run_projected_hvp_canary",
    "solve_certified_gram",
    "solve_projected_steihaug",
)
