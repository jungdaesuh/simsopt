"""Device-resident projected Gauss--Newton trust-region optimizer.

Callers supply one authoritative scalar-objective/equality function and one
canonical objective-residual function.  The optimizer retains one residual
linearization per attempt and owns all bounded acceptance and radius control.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from simsopt_jax.runtime.trace_annotations import PhaseId, device_scope

from .dense_sqp import JointValueConstraints, JointVJPRows, materialize_joint_vjp_rows
from .projected_hvp_trust_region import (
    ProjectedSteihaugTermination,
    exact_hvp_bilinear_symmetry_relative_defect,
    factor_certified_gram_projector,
    project_with_certified_gram,
    solve_certified_gram,
    solve_projected_steihaug,
)

ObjectiveResiduals = Callable[[jax.Array], jax.Array]

_STEP_BOUND_SAFEGUARD_BACKTRACKS = 2
_STEP_BOUND_SAFEGUARD_SUBTRIALS = 1 + _STEP_BOUND_SAFEGUARD_BACKTRACKS


class ProjectedGaussNewtonStatus(IntEnum):
    """Terminal status of one bounded Gauss--Newton solve."""

    RUNNING = 0
    BOUNDED_COMPLETE = 1
    ATTEMPT_LIMIT = 2
    FATAL_CURRENT_STATE = 3
    FATAL_STEIHAUG = 4
    FATAL_CURVATURE = 5
    DEVICE_QUALITY_CANDIDATE = 6


class ProjectedGaussNewtonAttemptOutcome(IntEnum):
    """One attempt's accepted, retryable, or fatal disposition."""

    INACTIVE = 0
    ACCEPTED = 1
    RETRY_OBJECTIVE = 2
    RETRY_NONFINITE = 3
    RETRY_CORRECTION_CERTIFICATE = 4
    RETRY_FEASIBILITY = 5
    RETRY_STEP_BOUNDS = 6
    FATAL_CURRENT_STATE = 7
    FATAL_STEIHAUG = 8
    FATAL_CURVATURE = 9


@dataclass(frozen=True, slots=True)
class ProjectedGaussNewtonOptions:
    """Immutable numerical policy for the bounded trust-region state machine."""

    maximum_accepted_steps: int = 8
    maximum_attempts: int = 12
    initial_trust_radius: float = 2.0**-10
    minimum_trust_radius: float = 2.0**-20
    maximum_trust_radius: float = 2.0**-4
    maximum_steihaug_iterations: int = 32
    projected_residual_tolerance: float = 1.0e-10
    linear_residual_tolerance: float = 1.0e-10
    corrected_feasibility_tolerance: float = 1.0e-10
    forward_error_tolerance: float = 1.0e-7
    residual_value_defect_tolerance: float = 1.0e-12
    residual_gradient_defect_tolerance: float = 1.0e-10
    normalized_curvature_tolerance: float = 1.0e-10
    maximum_correction_step_ratio: float = 1.0e-3
    maximum_corrected_radius_excess: float = 1.0e-6
    mechanism_rotation_threshold: float = 1.0e-3
    maximum_nonlinear_corrections: int = 1
    enable_step_bound_safeguard: bool = False


_DEFAULT_OPTIONS = ProjectedGaussNewtonOptions()


class ProjectedGaussNewtonHistory(NamedTuple):
    """Fixed-shape scalar evidence for every bounded attempt."""

    outcome: jax.Array
    accepted_step_number: jax.Array
    current_objective: jax.Array
    current_feasibility_inf: jax.Array
    current_stationarity_inf: jax.Array
    candidate_objective: jax.Array
    candidate_feasibility_inf: jax.Array
    actual_reduction: jax.Array
    predicted_reduction: jax.Array
    reduction_ratio: jax.Array
    trust_radius: jax.Array
    next_trust_radius: jax.Array
    tangent_step_norm: jax.Array
    correction_norm: jax.Array
    nonlinear_corrections: jax.Array
    applied_step_norm: jax.Array
    correction_step_ratio: jax.Array
    maximum_individual_correction_step_ratio: jax.Array
    correction_path_step_ratio: jax.Array
    corrected_radius_ratio: jax.Array
    steihaug_iterations: jax.Array
    steihaug_hvp_evaluations: jax.Array
    steihaug_solve_calls: jax.Array
    steihaug_termination: jax.Array
    terminal_normalized_curvature: jax.Array
    residual_value_defect: jax.Array
    residual_gradient_defect: jax.Array
    hvp_symmetry_defect: jax.Array
    probe_normalized_curvature: jax.Array
    direction_rotation: jax.Array
    correction_relative_residual: jax.Array
    correction_forward_error_bound: jax.Array
    trial_gram_factorization_relative_residual: jax.Array
    trial_gram_solve_relative_residual: jax.Array
    current_projection_tangency_relative_residual: jax.Array
    current_projection_solve_relative_residual: jax.Array
    current_projection_forward_error_bound: jax.Array
    steihaug_tangency_relative_residual: jax.Array
    steihaug_final_projected_residual_norm: jax.Array
    steihaug_projected_residual_target: jax.Array
    steihaug_hit_boundary: jax.Array
    steihaug_residual_projection_tangency_relative_residual: jax.Array
    steihaug_residual_projection_solve_relative_residual: jax.Array
    steihaug_residual_projection_forward_error_bound: jax.Array
    subtrial_count: jax.Array
    selected_subtrial_index: jax.Array
    subtrial_trust_radius: jax.Array
    subtrial_outcome: jax.Array
    subtrial_actual_reduction: jax.Array
    subtrial_predicted_reduction: jax.Array
    subtrial_maximum_individual_correction_step_ratio: jax.Array
    subtrial_correction_path_step_ratio: jax.Array
    subtrial_corrected_radius_ratio: jax.Array
    subtrial_steihaug_iterations: jax.Array
    subtrial_steihaug_hvp_evaluations: jax.Array
    subtrial_steihaug_solve_calls: jax.Array
    subtrial_total_hvp_evaluations: jax.Array
    subtrial_nonlinear_corrections: jax.Array
    subtrial_joint_evaluations: jax.Array
    subtrial_joint_linearizations: jax.Array
    subtrial_joint_value_evaluations: jax.Array
    subtrial_objective_residual_linearizations: jax.Array
    subtrial_gram_factorizations: jax.Array
    subtrial_gram_solves: jax.Array


class ProjectedGaussNewtonFinalCertificate(NamedTuple):
    """Current-state certificates recomputed at the returned coordinates."""

    coordinates_finite: jax.Array
    residual_value_defect: jax.Array
    residual_gradient_defect: jax.Array
    hvp_symmetry_defect: jax.Array
    probe_normalized_curvature: jax.Array
    gram_factorization_relative_residual: jax.Array
    multiplier_relative_residual: jax.Array
    multiplier_forward_error_bound: jax.Array
    projection_tangency_relative_residual: jax.Array
    projection_solve_relative_residual: jax.Array
    projection_forward_error_bound: jax.Array
    all_finite: jax.Array
    certified: jax.Array


class ProjectedGaussNewtonAcceptedState(NamedTuple):
    """Corrected accepted candidate exposed to a route-owned quality predicate."""

    optimizer_coordinates: jax.Array
    objective: jax.Array
    constraints: jax.Array
    scaled_feasibility_inf: jax.Array


AcceptedStateQualityPredicate = Callable[[ProjectedGaussNewtonAcceptedState], jax.Array]


def _never_accept_quality(
    _accepted_state: ProjectedGaussNewtonAcceptedState,
) -> jax.Array:
    return jnp.asarray(False)


class ProjectedGaussNewtonLoopResult(NamedTuple):
    """Timed device-loop output before endpoint derivative finalization."""

    optimizer_coordinates: jax.Array
    trust_radius: jax.Array
    accepted_steps: jax.Array
    attempts: jax.Array
    retryable_rejections: jax.Array
    status: jax.Array
    fatal: jax.Array
    bounded_complete: jax.Array
    device_quality_candidate_reached: jax.Array
    first_quality_attempt: jax.Array
    first_quality_accepted_step: jax.Array
    accepted_optimizer_coordinates: jax.Array
    accepted_state_mask: jax.Array
    mechanism_exercised: jax.Array
    all_accepted_states_finite: jax.Array
    history: ProjectedGaussNewtonHistory


class ProjectedGaussNewtonResult(NamedTuple):
    """Fixed-shape final state and attempt evidence for one bounded solve."""

    optimizer_coordinates: jax.Array
    multipliers: jax.Array
    objective: jax.Array
    constraints: jax.Array
    objective_gradient: jax.Array
    constraint_jacobian: jax.Array
    stationarity: jax.Array
    scaled_stationarity_inf: jax.Array
    scaled_feasibility_inf: jax.Array
    trust_radius: jax.Array
    accepted_steps: jax.Array
    attempts: jax.Array
    retryable_rejections: jax.Array
    status: jax.Array
    fatal: jax.Array
    bounded_complete: jax.Array
    mechanism_exercised: jax.Array
    all_accepted_states_finite: jax.Array
    all_finite: jax.Array
    final_certificate: ProjectedGaussNewtonFinalCertificate
    usable: jax.Array
    history: ProjectedGaussNewtonHistory
    device_quality_candidate_reached: jax.Array | bool = False
    first_quality_attempt: jax.Array | int = 0
    first_quality_accepted_step: jax.Array | int = 0


class _AttemptTelemetry(NamedTuple):
    outcome: jax.Array
    accepted_step_number: jax.Array
    current_objective: jax.Array
    current_feasibility_inf: jax.Array
    current_stationarity_inf: jax.Array
    candidate_objective: jax.Array
    candidate_feasibility_inf: jax.Array
    actual_reduction: jax.Array
    predicted_reduction: jax.Array
    reduction_ratio: jax.Array
    trust_radius: jax.Array
    next_trust_radius: jax.Array
    tangent_step_norm: jax.Array
    correction_norm: jax.Array
    nonlinear_corrections: jax.Array
    applied_step_norm: jax.Array
    correction_step_ratio: jax.Array
    maximum_individual_correction_step_ratio: jax.Array
    correction_path_step_ratio: jax.Array
    corrected_radius_ratio: jax.Array
    steihaug_iterations: jax.Array
    steihaug_hvp_evaluations: jax.Array
    steihaug_solve_calls: jax.Array
    steihaug_termination: jax.Array
    terminal_normalized_curvature: jax.Array
    residual_value_defect: jax.Array
    residual_gradient_defect: jax.Array
    hvp_symmetry_defect: jax.Array
    probe_normalized_curvature: jax.Array
    direction_rotation: jax.Array
    correction_relative_residual: jax.Array
    correction_forward_error_bound: jax.Array
    trial_gram_factorization_relative_residual: jax.Array
    trial_gram_solve_relative_residual: jax.Array
    current_projection_tangency_relative_residual: jax.Array
    current_projection_solve_relative_residual: jax.Array
    current_projection_forward_error_bound: jax.Array
    steihaug_tangency_relative_residual: jax.Array
    steihaug_final_projected_residual_norm: jax.Array
    steihaug_projected_residual_target: jax.Array
    steihaug_hit_boundary: jax.Array
    steihaug_residual_projection_tangency_relative_residual: jax.Array
    steihaug_residual_projection_solve_relative_residual: jax.Array
    steihaug_residual_projection_forward_error_bound: jax.Array
    subtrial_count: jax.Array
    selected_subtrial_index: jax.Array
    subtrial_trust_radius: jax.Array
    subtrial_outcome: jax.Array
    subtrial_actual_reduction: jax.Array
    subtrial_predicted_reduction: jax.Array
    subtrial_maximum_individual_correction_step_ratio: jax.Array
    subtrial_correction_path_step_ratio: jax.Array
    subtrial_corrected_radius_ratio: jax.Array
    subtrial_steihaug_iterations: jax.Array
    subtrial_steihaug_hvp_evaluations: jax.Array
    subtrial_steihaug_solve_calls: jax.Array
    subtrial_total_hvp_evaluations: jax.Array
    subtrial_nonlinear_corrections: jax.Array
    subtrial_joint_evaluations: jax.Array
    subtrial_joint_linearizations: jax.Array
    subtrial_joint_value_evaluations: jax.Array
    subtrial_objective_residual_linearizations: jax.Array
    subtrial_gram_factorizations: jax.Array
    subtrial_gram_solves: jax.Array


class _LoopState(NamedTuple):
    coordinates: jax.Array
    trust_radius: jax.Array
    accepted_steps: jax.Array
    attempts: jax.Array
    retryable_rejections: jax.Array
    status: jax.Array
    fatal: jax.Array
    device_quality_candidate_reached: jax.Array
    first_quality_attempt: jax.Array
    first_quality_accepted_step: jax.Array
    accepted_optimizer_coordinates: jax.Array
    accepted_state_mask: jax.Array
    mechanism_exercised: jax.Array
    all_accepted_states_finite: jax.Array
    history: ProjectedGaussNewtonHistory


class _StepBoundSafeguardState(NamedTuple):
    """Fixed-shape evidence and selected state for one safeguarded attempt."""

    result: _LoopState
    subtrial_count: jax.Array
    selected_subtrial_index: jax.Array
    subtrial_trust_radius: jax.Array
    subtrial_outcome: jax.Array
    subtrial_actual_reduction: jax.Array
    subtrial_predicted_reduction: jax.Array
    subtrial_maximum_individual_correction_step_ratio: jax.Array
    subtrial_correction_path_step_ratio: jax.Array
    subtrial_corrected_radius_ratio: jax.Array
    subtrial_steihaug_iterations: jax.Array
    subtrial_steihaug_hvp_evaluations: jax.Array
    subtrial_steihaug_solve_calls: jax.Array
    subtrial_total_hvp_evaluations: jax.Array
    subtrial_nonlinear_corrections: jax.Array
    subtrial_joint_evaluations: jax.Array
    subtrial_joint_linearizations: jax.Array
    subtrial_joint_value_evaluations: jax.Array
    subtrial_objective_residual_linearizations: jax.Array
    subtrial_gram_factorizations: jax.Array
    subtrial_gram_solves: jax.Array
    retry_step_bounds: jax.Array


class _NormalCorrection(NamedTuple):
    """One certified Newton correction normal to the linearized constraints."""

    step: jax.Array
    all_finite: jax.Array
    factorization_relative_residual: jax.Array
    linearized_constraint_relative_residual: jax.Array
    solve_relative_residual: jax.Array
    solve_forward_error_bound: jax.Array


class _NonlinearRetractionState(NamedTuple):
    """Fixed-shape cumulative state for nonlinear normal retraction."""

    coordinates: jax.Array
    correction: jax.Array
    maximum_individual_correction_norm: jax.Array
    correction_path_norm: jax.Array
    nonlinear_corrections: jax.Array
    all_finite: jax.Array
    factorization_relative_residual: jax.Array
    linearized_constraint_relative_residual: jax.Array
    solve_relative_residual: jax.Array
    solve_forward_error_bound: jax.Array


def _validate_options(options: ProjectedGaussNewtonOptions) -> None:
    if options.maximum_accepted_steps < 1:
        raise ValueError("maximum_accepted_steps must be positive")
    if options.maximum_attempts < options.maximum_accepted_steps:
        raise ValueError("maximum_attempts must cover maximum_accepted_steps")
    if options.maximum_steihaug_iterations < 1:
        raise ValueError("maximum_steihaug_iterations must be positive")
    if (
        not isinstance(options.maximum_nonlinear_corrections, int)
        or isinstance(options.maximum_nonlinear_corrections, bool)
        or options.maximum_nonlinear_corrections < 1
    ):
        raise ValueError("maximum_nonlinear_corrections must be a positive integer")
    if not isinstance(options.enable_step_bound_safeguard, bool):
        raise TypeError("enable_step_bound_safeguard must be a bool")
    radii = (
        options.minimum_trust_radius,
        options.initial_trust_radius,
        options.maximum_trust_radius,
    )
    if any(not math.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise ValueError("trust radii must be finite and positive")
    if not radii[0] <= radii[1] <= radii[2]:
        raise ValueError("trust radii must be ordered minimum through maximum")
    tolerances = (
        options.projected_residual_tolerance,
        options.linear_residual_tolerance,
        options.corrected_feasibility_tolerance,
        options.forward_error_tolerance,
        options.residual_value_defect_tolerance,
        options.residual_gradient_defect_tolerance,
        options.normalized_curvature_tolerance,
        options.maximum_correction_step_ratio,
        options.maximum_corrected_radius_excess,
        options.mechanism_rotation_threshold,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("optimizer tolerances must be finite and nonnegative")


def _trial_correction_certified(
    correction_all_finite: jax.Array,
    factorization_relative_residual: jax.Array,
    correction_relative_residual: jax.Array,
    solve_relative_residual: jax.Array,
    solve_forward_error_bound: jax.Array,
    *,
    linear_residual_tolerance: float,
    forward_error_tolerance: float,
) -> jax.Array:
    """Apply the complete trial-correction certificate without hidden coupling."""

    return (
        correction_all_finite
        & (factorization_relative_residual <= linear_residual_tolerance)
        & (correction_relative_residual <= linear_residual_tolerance)
        & (solve_relative_residual <= linear_residual_tolerance)
        & (solve_forward_error_bound < forward_error_tolerance)
    )


def _correction_path_bounds_valid(
    maximum_individual_correction_step_ratio: jax.Array,
    correction_path_step_ratio: jax.Array,
    nonlinear_corrections: jax.Array,
    corrected_radius_ratio: jax.Array,
    *,
    correction_ratio_limit: jax.Array,
    corrected_radius_limit: jax.Array,
) -> jax.Array:
    """Certify every correction, the traversed path, and the corrected radius."""

    path_ratio_limit = (
        nonlinear_corrections.astype(correction_path_step_ratio.dtype)
        * correction_ratio_limit
    )
    return (
        (maximum_individual_correction_step_ratio <= correction_ratio_limit)
        & (correction_path_step_ratio <= path_ratio_limit)
        & (corrected_radius_ratio <= corrected_radius_limit)
    )


def _normal_correction(rows: JointVJPRows) -> _NormalCorrection:
    """Solve and certify one Newton correction for the supplied nonlinear rows."""

    projector = factor_certified_gram_projector(rows.constraint_jacobian)
    correction_solve = solve_certified_gram(projector, -rows.constraints)
    step = rows.constraint_jacobian.T @ correction_solve.solution
    linearized_constraints = rows.constraints + rows.constraint_jacobian @ step
    tiny = jnp.asarray(
        jnp.finfo(rows.constraints.dtype).tiny,
        dtype=rows.constraints.dtype,
    )
    linearized_constraint_relative_residual = jnp.linalg.norm(
        linearized_constraints, ord=jnp.inf
    ) / jnp.maximum(
        tiny,
        jnp.linalg.norm(rows.constraint_jacobian, ord=jnp.inf)
        * jnp.linalg.norm(step, ord=jnp.inf)
        + jnp.linalg.norm(rows.constraints, ord=jnp.inf),
    )
    all_finite = (
        projector.all_finite
        & correction_solve.all_finite
        & jnp.all(jnp.isfinite(step))
        & jnp.all(jnp.isfinite(linearized_constraints))
        & jnp.isfinite(linearized_constraint_relative_residual)
    )
    return _NormalCorrection(
        step=step,
        all_finite=all_finite,
        factorization_relative_residual=projector.factorization_relative_residual,
        linearized_constraint_relative_residual=(
            linearized_constraint_relative_residual
        ),
        solve_relative_residual=correction_solve.relative_residual,
        solve_forward_error_bound=correction_solve.forward_error_bound,
    )


def _extend_nonlinear_retraction(
    joint_value_constraints: JointValueConstraints,
    feasibility_tolerance: float,
    _iteration: jax.Array,
    state: _NonlinearRetractionState,
) -> _NonlinearRetractionState:
    """Apply one more certified correction only while nonlinear feasibility fails."""

    _, constraints = joint_value_constraints(state.coordinates)
    feasibility = jnp.linalg.norm(constraints, ord=jnp.inf)
    correction_required = (
        state.all_finite
        & jnp.isfinite(feasibility)
        & (feasibility > feasibility_tolerance)
    )

    def correct(active: _NonlinearRetractionState) -> _NonlinearRetractionState:
        rows = materialize_joint_vjp_rows(joint_value_constraints, active.coordinates)
        correction = _normal_correction(rows)
        correction_norm = jnp.linalg.norm(correction.step)
        return _NonlinearRetractionState(
            coordinates=active.coordinates + correction.step,
            correction=active.correction + correction.step,
            maximum_individual_correction_norm=jnp.maximum(
                active.maximum_individual_correction_norm,
                correction_norm,
            ),
            correction_path_norm=active.correction_path_norm + correction_norm,
            nonlinear_corrections=active.nonlinear_corrections
            + jnp.asarray(1, dtype=jnp.int32),
            all_finite=active.all_finite & correction.all_finite,
            factorization_relative_residual=jnp.maximum(
                active.factorization_relative_residual,
                correction.factorization_relative_residual,
            ),
            linearized_constraint_relative_residual=jnp.maximum(
                active.linearized_constraint_relative_residual,
                correction.linearized_constraint_relative_residual,
            ),
            solve_relative_residual=jnp.maximum(
                active.solve_relative_residual,
                correction.solve_relative_residual,
            ),
            solve_forward_error_bound=jnp.maximum(
                active.solve_forward_error_bound,
                correction.solve_forward_error_bound,
            ),
        )

    return jax.lax.cond(correction_required, correct, lambda active: active, state)


def _empty_history(
    maximum_attempts: int, dtype: jnp.dtype
) -> ProjectedGaussNewtonHistory:
    floating = lambda: jnp.full((maximum_attempts,), jnp.nan, dtype=dtype)
    integer = lambda value: jnp.full((maximum_attempts,), value, dtype=jnp.int32)
    subtrial_floating = lambda: jnp.full(
        (maximum_attempts, _STEP_BOUND_SAFEGUARD_SUBTRIALS), jnp.nan, dtype=dtype
    )
    subtrial_integer = lambda value: jnp.full(
        (maximum_attempts, _STEP_BOUND_SAFEGUARD_SUBTRIALS),
        value,
        dtype=jnp.int32,
    )
    return ProjectedGaussNewtonHistory(
        outcome=integer(int(ProjectedGaussNewtonAttemptOutcome.INACTIVE)),
        accepted_step_number=integer(0),
        current_objective=floating(),
        current_feasibility_inf=floating(),
        current_stationarity_inf=floating(),
        candidate_objective=floating(),
        candidate_feasibility_inf=floating(),
        actual_reduction=floating(),
        predicted_reduction=floating(),
        reduction_ratio=floating(),
        trust_radius=floating(),
        next_trust_radius=floating(),
        tangent_step_norm=floating(),
        correction_norm=floating(),
        nonlinear_corrections=integer(0),
        applied_step_norm=floating(),
        correction_step_ratio=floating(),
        maximum_individual_correction_step_ratio=floating(),
        correction_path_step_ratio=floating(),
        corrected_radius_ratio=floating(),
        steihaug_iterations=integer(0),
        steihaug_hvp_evaluations=integer(0),
        steihaug_solve_calls=integer(0),
        steihaug_termination=integer(
            int(ProjectedSteihaugTermination.INTERIOR_CONVERGED)
        ),
        terminal_normalized_curvature=floating(),
        residual_value_defect=floating(),
        residual_gradient_defect=floating(),
        hvp_symmetry_defect=floating(),
        probe_normalized_curvature=floating(),
        direction_rotation=floating(),
        correction_relative_residual=floating(),
        correction_forward_error_bound=floating(),
        trial_gram_factorization_relative_residual=floating(),
        trial_gram_solve_relative_residual=floating(),
        current_projection_tangency_relative_residual=floating(),
        current_projection_solve_relative_residual=floating(),
        current_projection_forward_error_bound=floating(),
        steihaug_tangency_relative_residual=floating(),
        steihaug_final_projected_residual_norm=floating(),
        steihaug_projected_residual_target=floating(),
        steihaug_hit_boundary=jnp.zeros((maximum_attempts,), dtype=jnp.bool_),
        steihaug_residual_projection_tangency_relative_residual=floating(),
        steihaug_residual_projection_solve_relative_residual=floating(),
        steihaug_residual_projection_forward_error_bound=floating(),
        subtrial_count=integer(0),
        selected_subtrial_index=integer(-1),
        subtrial_trust_radius=subtrial_floating(),
        subtrial_outcome=subtrial_integer(
            int(ProjectedGaussNewtonAttemptOutcome.INACTIVE)
        ),
        subtrial_actual_reduction=subtrial_floating(),
        subtrial_predicted_reduction=subtrial_floating(),
        subtrial_maximum_individual_correction_step_ratio=subtrial_floating(),
        subtrial_correction_path_step_ratio=subtrial_floating(),
        subtrial_corrected_radius_ratio=subtrial_floating(),
        subtrial_steihaug_iterations=subtrial_integer(0),
        subtrial_steihaug_hvp_evaluations=subtrial_integer(0),
        subtrial_steihaug_solve_calls=subtrial_integer(0),
        subtrial_total_hvp_evaluations=subtrial_integer(0),
        subtrial_nonlinear_corrections=subtrial_integer(0),
        subtrial_joint_evaluations=subtrial_integer(0),
        subtrial_joint_linearizations=subtrial_integer(0),
        subtrial_joint_value_evaluations=subtrial_integer(0),
        subtrial_objective_residual_linearizations=subtrial_integer(0),
        subtrial_gram_factorizations=subtrial_integer(0),
        subtrial_gram_solves=subtrial_integer(0),
    )


def _record_attempt(
    history: ProjectedGaussNewtonHistory,
    index: jax.Array,
    telemetry: _AttemptTelemetry,
) -> ProjectedGaussNewtonHistory:
    return ProjectedGaussNewtonHistory(
        *(
            values.at[index].set(value)
            for values, value in zip(history, telemetry, strict=True)
        )
    )


def _scaled_defect(
    candidate: jax.Array,
    reference: jax.Array,
) -> jax.Array:
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=candidate.dtype),
        jnp.maximum(jnp.abs(candidate), jnp.abs(reference)),
    )
    return jnp.abs(candidate - reference) / scale


def _gradient_scaled_defect(
    candidate: jax.Array,
    reference: jax.Array,
) -> jax.Array:
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=candidate.dtype),
        jnp.maximum(
            jnp.linalg.norm(candidate, ord=jnp.inf),
            jnp.linalg.norm(reference, ord=jnp.inf),
        ),
    )
    return jnp.linalg.norm(candidate - reference, ord=jnp.inf) / scale


def _normalized_curvature(vector: jax.Array, action: jax.Array) -> jax.Array:
    denominator = jnp.linalg.norm(vector) * jnp.linalg.norm(action)
    return jnp.where(denominator > 0.0, (vector @ action) / denominator, 0.0)


def _direction_rotation(
    initial_direction: jax.Array,
    tangent_step: jax.Array,
) -> jax.Array:
    direction_norm_squared = initial_direction @ initial_direction
    tangent_norm = jnp.linalg.norm(tangent_step)
    parallel_squared = jnp.where(
        direction_norm_squared > 0.0,
        (tangent_step @ initial_direction) ** 2 / direction_norm_squared,
        0.0,
    )
    orthogonal_norm = jnp.sqrt(
        jnp.maximum(tangent_step @ tangent_step - parallel_squared, 0.0)
    )
    finite_nonzero = (
        jnp.isfinite(direction_norm_squared)
        & jnp.isfinite(tangent_norm)
        & (direction_norm_squared > 0.0)
        & (tangent_norm > 0.0)
    )
    return jnp.where(finite_nonzero, orthogonal_norm / tangent_norm, 0.0)


def _ratio_radius_update(
    radius: jax.Array,
    step_norm: jax.Array,
    reduction_ratio: jax.Array,
    minimum_radius: jax.Array,
    maximum_radius: jax.Array,
) -> jax.Array:
    proposed = jnp.where(
        ~jnp.isfinite(reduction_ratio) | (reduction_ratio < 0.25),
        0.25 * step_norm,
        jnp.where(
            reduction_ratio > 0.75,
            jnp.maximum(2.0 * step_norm, radius),
            radius,
        ),
    )
    return jnp.clip(proposed, minimum_radius, maximum_radius)


def run_projected_gauss_newton_trust_region_loop(
    joint_value_constraints: JointValueConstraints,
    objective_residuals: ObjectiveResiduals,
    initial_coordinates: jax.Array,
    *,
    options: ProjectedGaussNewtonOptions = _DEFAULT_OPTIONS,
    accepted_state_quality_predicate: AcceptedStateQualityPredicate = (
        _never_accept_quality
    ),
) -> ProjectedGaussNewtonLoopResult:
    """Run only the bounded device loop, excluding endpoint finalization."""

    _validate_options(options)
    dtype = initial_coordinates.dtype
    minimum_radius = jnp.asarray(options.minimum_trust_radius, dtype=dtype)
    maximum_radius = jnp.asarray(options.maximum_trust_radius, dtype=dtype)
    correction_ratio_limit = jnp.asarray(
        options.maximum_correction_step_ratio, dtype=dtype
    )
    corrected_radius_limit = jnp.asarray(
        1.0 + options.maximum_corrected_radius_excess, dtype=dtype
    )
    nan = jnp.asarray(jnp.nan, dtype=dtype)

    initial_state = _LoopState(
        coordinates=initial_coordinates,
        trust_radius=jnp.asarray(options.initial_trust_radius, dtype=dtype),
        accepted_steps=jnp.asarray(0, dtype=jnp.int32),
        attempts=jnp.asarray(0, dtype=jnp.int32),
        retryable_rejections=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(int(ProjectedGaussNewtonStatus.RUNNING), dtype=jnp.int32),
        fatal=jnp.asarray(False),
        device_quality_candidate_reached=jnp.asarray(False),
        first_quality_attempt=jnp.asarray(0, dtype=jnp.int32),
        first_quality_accepted_step=jnp.asarray(0, dtype=jnp.int32),
        accepted_optimizer_coordinates=(
            jnp.zeros(
                (options.maximum_accepted_steps + 1, initial_coordinates.shape[0]),
                dtype=dtype,
            )
            .at[0]
            .set(initial_coordinates)
        ),
        accepted_state_mask=(
            jnp.zeros((options.maximum_accepted_steps + 1,), dtype=jnp.bool_)
            .at[0]
            .set(True)
        ),
        mechanism_exercised=jnp.asarray(False),
        all_accepted_states_finite=jnp.asarray(True),
        history=_empty_history(options.maximum_attempts, dtype),
    )

    def attempt(state: _LoopState) -> _LoopState:
        coordinates = state.coordinates
        with device_scope(PhaseId.GNTR_CURRENT_LINEARIZATION):
            rows = materialize_joint_vjp_rows(joint_value_constraints, coordinates)
            residual, residual_jvp = jax.linearize(objective_residuals, coordinates)
            residual_transpose = jax.linear_transpose(
                residual_jvp, jnp.zeros_like(coordinates)
            )

        def gn_hvp(vector: jax.Array) -> jax.Array:
            return residual_transpose(residual_jvp(vector))[0]

        with device_scope(PhaseId.GNTR_CURRENT_CERTIFICATES):
            reconstructed_objective = 0.5 * jnp.vdot(residual, residual)
            reconstructed_gradient = residual_transpose(residual)[0]
            value_defect = _scaled_defect(reconstructed_objective, rows.objective)
            gradient_defect = _gradient_scaled_defect(
                reconstructed_gradient, rows.objective_gradient
            )
            projector = factor_certified_gram_projector(rows.constraint_jacobian)
            multiplier_projection = solve_certified_gram(
                projector,
                -(rows.constraint_jacobian @ rows.objective_gradient),
            )
            stationarity = (
                rows.objective_gradient
                + rows.constraint_jacobian.T @ multiplier_projection.solution
            )
            stationarity_inf = jnp.linalg.norm(stationarity, ord=jnp.inf)
            feasibility_inf = jnp.linalg.norm(rows.constraints, ord=jnp.inf)
            projected_stationarity = project_with_certified_gram(
                projector, stationarity
            )
            initial_direction = -projected_stationarity.projected
            symmetry_defect = exact_hvp_bilinear_symmetry_relative_defect(
                gn_hvp, coordinates
            )
            probe_indices = jnp.arange(
                1, coordinates.shape[0] + 1, dtype=coordinates.dtype
            )
            probe = jnp.sin(probe_indices)
            probe = probe / jnp.linalg.norm(probe)
            probe_action = gn_hvp(probe)
            probe_curvature = _normalized_curvature(probe, probe_action)
            rows_finite = (
                jnp.all(jnp.isfinite(coordinates))
                & jnp.isfinite(rows.objective)
                & jnp.all(jnp.isfinite(rows.constraints))
                & jnp.all(jnp.isfinite(rows.objective_gradient))
                & jnp.all(jnp.isfinite(rows.constraint_jacobian))
                & jnp.all(jnp.isfinite(stationarity))
                & jnp.isfinite(stationarity_inf)
                & jnp.isfinite(feasibility_inf)
            )
            curvature_finite = (
                jnp.isfinite(symmetry_defect)
                & jnp.all(jnp.isfinite(probe_action))
                & jnp.isfinite(probe_curvature)
            )
            current_base_valid = (
                rows_finite
                & jnp.all(jnp.isfinite(residual))
                & jnp.isfinite(reconstructed_objective)
                & jnp.all(jnp.isfinite(reconstructed_gradient))
                & (value_defect <= options.residual_value_defect_tolerance)
                & (gradient_defect <= options.residual_gradient_defect_tolerance)
                & projector.all_finite
                & (
                    projector.factorization_relative_residual
                    <= options.linear_residual_tolerance
                )
                & multiplier_projection.all_finite
                & (
                    multiplier_projection.relative_residual
                    <= options.linear_residual_tolerance
                )
                & (
                    multiplier_projection.forward_error_bound
                    < options.forward_error_tolerance
                )
                & projected_stationarity.all_finite
                & (
                    projected_stationarity.tangency_relative_residual
                    <= options.linear_residual_tolerance
                )
                & (
                    projected_stationarity.solve_relative_residual
                    <= options.linear_residual_tolerance
                )
                & (
                    projected_stationarity.solve_forward_error_bound
                    < options.forward_error_tolerance
                )
                & (feasibility_inf <= options.corrected_feasibility_tolerance)
                & curvature_finite
                & (symmetry_defect <= options.residual_gradient_defect_tolerance)
            )
            probe_curvature_valid = (
                probe_curvature >= -options.normalized_curvature_tolerance
            )

        base_telemetry = _AttemptTelemetry(
            outcome=jnp.asarray(
                int(ProjectedGaussNewtonAttemptOutcome.FATAL_CURRENT_STATE),
                dtype=jnp.int32,
            ),
            accepted_step_number=jnp.asarray(0, dtype=jnp.int32),
            current_objective=rows.objective,
            current_feasibility_inf=feasibility_inf,
            current_stationarity_inf=stationarity_inf,
            candidate_objective=nan,
            candidate_feasibility_inf=nan,
            actual_reduction=nan,
            predicted_reduction=nan,
            reduction_ratio=nan,
            trust_radius=state.trust_radius,
            next_trust_radius=state.trust_radius,
            tangent_step_norm=nan,
            correction_norm=nan,
            nonlinear_corrections=jnp.asarray(0, dtype=jnp.int32),
            applied_step_norm=nan,
            correction_step_ratio=nan,
            maximum_individual_correction_step_ratio=nan,
            correction_path_step_ratio=nan,
            corrected_radius_ratio=nan,
            steihaug_iterations=jnp.asarray(0, dtype=jnp.int32),
            steihaug_hvp_evaluations=jnp.asarray(0, dtype=jnp.int32),
            steihaug_solve_calls=jnp.asarray(0, dtype=jnp.int32),
            steihaug_termination=jnp.asarray(
                int(ProjectedSteihaugTermination.INTERIOR_CONVERGED),
                dtype=jnp.int32,
            ),
            terminal_normalized_curvature=nan,
            residual_value_defect=value_defect,
            residual_gradient_defect=gradient_defect,
            hvp_symmetry_defect=symmetry_defect,
            probe_normalized_curvature=probe_curvature,
            direction_rotation=nan,
            correction_relative_residual=nan,
            correction_forward_error_bound=nan,
            trial_gram_factorization_relative_residual=nan,
            trial_gram_solve_relative_residual=nan,
            current_projection_tangency_relative_residual=(
                projected_stationarity.tangency_relative_residual
            ),
            current_projection_solve_relative_residual=(
                projected_stationarity.solve_relative_residual
            ),
            current_projection_forward_error_bound=(
                projected_stationarity.solve_forward_error_bound
            ),
            steihaug_tangency_relative_residual=nan,
            steihaug_final_projected_residual_norm=nan,
            steihaug_projected_residual_target=nan,
            steihaug_hit_boundary=jnp.asarray(False),
            steihaug_residual_projection_tangency_relative_residual=nan,
            steihaug_residual_projection_solve_relative_residual=nan,
            steihaug_residual_projection_forward_error_bound=nan,
            subtrial_count=jnp.asarray(0, dtype=jnp.int32),
            selected_subtrial_index=jnp.asarray(-1, dtype=jnp.int32),
            subtrial_trust_radius=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), nan, dtype=dtype
            ),
            subtrial_outcome=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,),
                int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
                dtype=jnp.int32,
            ),
            subtrial_actual_reduction=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), nan, dtype=dtype
            ),
            subtrial_predicted_reduction=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), nan, dtype=dtype
            ),
            subtrial_maximum_individual_correction_step_ratio=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), nan, dtype=dtype
            ),
            subtrial_correction_path_step_ratio=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), nan, dtype=dtype
            ),
            subtrial_corrected_radius_ratio=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), nan, dtype=dtype
            ),
            subtrial_steihaug_iterations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_steihaug_hvp_evaluations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_steihaug_solve_calls=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_total_hvp_evaluations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_nonlinear_corrections=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_joint_evaluations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_joint_linearizations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_joint_value_evaluations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_objective_residual_linearizations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_gram_factorizations=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
            subtrial_gram_solves=jnp.zeros(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32
            ),
        )

        def fatal_current(current: _LoopState) -> _LoopState:
            outcome = jnp.where(
                current_base_valid,
                int(ProjectedGaussNewtonAttemptOutcome.FATAL_CURVATURE),
                int(ProjectedGaussNewtonAttemptOutcome.FATAL_CURRENT_STATE),
            ).astype(jnp.int32)
            status = jnp.where(
                current_base_valid,
                int(ProjectedGaussNewtonStatus.FATAL_CURVATURE),
                int(ProjectedGaussNewtonStatus.FATAL_CURRENT_STATE),
            ).astype(jnp.int32)
            telemetry = base_telemetry._replace(outcome=outcome)
            return current._replace(
                attempts=current.attempts + 1,
                status=status,
                fatal=jnp.asarray(True),
                history=_record_attempt(current.history, current.attempts, telemetry),
            )

        def valid_current(current: _LoopState) -> _LoopState:
            with device_scope(PhaseId.GNTR_STEIHAUG):
                steihaug = solve_projected_steihaug(
                    gn_hvp,
                    stationarity,
                    jnp.zeros_like(coordinates),
                    projector,
                    trust_radius=state.trust_radius,
                    maximum_iterations=options.maximum_steihaug_iterations,
                    projected_residual_tolerance=options.projected_residual_tolerance,
                )
                terminal_curvature_valid = jnp.isfinite(
                    steihaug.terminal_normalized_curvature
                ) & (
                    steihaug.terminal_normalized_curvature
                    >= -options.normalized_curvature_tolerance
                )
                terminal_residual_projection = project_with_certified_gram(
                    projector, stationarity + steihaug.hessian_combined_step
                )
            radius_roundoff = (
                jnp.asarray(64.0, dtype=dtype)
                * jnp.finfo(dtype).eps
                * jnp.maximum(jnp.asarray(1.0, dtype=dtype), state.trust_radius)
            )
            interior_termination = steihaug.termination == int(
                ProjectedSteihaugTermination.INTERIOR_CONVERGED
            )
            boundary_termination = steihaug.termination == int(
                ProjectedSteihaugTermination.TRUST_BOUNDARY
            )
            null_curvature_termination = steihaug.termination == int(
                ProjectedSteihaugTermination.NONPOSITIVE_CURVATURE
            )
            boundary_like_termination = (
                boundary_termination | null_curvature_termination
            )
            termination_certificate_valid = (
                (interior_termination | boundary_like_termination)
                & (steihaug.hit_boundary == boundary_like_termination)
                & (
                    steihaug.encountered_nonpositive_curvature
                    == null_curvature_termination
                )
                & (
                    ~interior_termination
                    | (
                        steihaug.final_projected_residual_norm
                        <= steihaug.projected_residual_target
                    )
                )
                & (
                    ~boundary_like_termination
                    | (
                        jnp.abs(steihaug.combined_step_norm - state.trust_radius)
                        <= radius_roundoff
                    )
                )
            )
            steihaug_valid = (
                steihaug.all_finite
                & jnp.all(jnp.isfinite(steihaug.tangential_step))
                & jnp.all(jnp.isfinite(steihaug.combined_step))
                & (steihaug.combined_step_norm <= state.trust_radius + radius_roundoff)
                & (
                    steihaug.tangency_relative_residual
                    <= options.linear_residual_tolerance
                )
                & terminal_residual_projection.all_finite
                & (
                    terminal_residual_projection.tangency_relative_residual
                    <= options.linear_residual_tolerance
                )
                & (
                    terminal_residual_projection.solve_relative_residual
                    <= options.linear_residual_tolerance
                )
                & (
                    terminal_residual_projection.solve_forward_error_bound
                    < options.forward_error_tolerance
                )
                & termination_certificate_valid
            )
            rotation = _direction_rotation(initial_direction, steihaug.tangential_step)
            step_telemetry = base_telemetry._replace(
                tangent_step_norm=jnp.linalg.norm(steihaug.tangential_step),
                steihaug_iterations=steihaug.iterations,
                steihaug_hvp_evaluations=steihaug.hvp_evaluations,
                steihaug_solve_calls=jnp.asarray(1, dtype=jnp.int32),
                steihaug_termination=steihaug.termination,
                terminal_normalized_curvature=(steihaug.terminal_normalized_curvature),
                direction_rotation=rotation,
                steihaug_tangency_relative_residual=(
                    steihaug.tangency_relative_residual
                ),
                steihaug_final_projected_residual_norm=(
                    steihaug.final_projected_residual_norm
                ),
                steihaug_projected_residual_target=(steihaug.projected_residual_target),
                steihaug_hit_boundary=steihaug.hit_boundary,
                steihaug_residual_projection_tangency_relative_residual=(
                    terminal_residual_projection.tangency_relative_residual
                ),
                steihaug_residual_projection_solve_relative_residual=(
                    terminal_residual_projection.solve_relative_residual
                ),
                steihaug_residual_projection_forward_error_bound=(
                    terminal_residual_projection.solve_forward_error_bound
                ),
            )

            def fatal_step(active: _LoopState) -> _LoopState:
                curvature_failure = steihaug_valid & ~terminal_curvature_valid
                outcome = jnp.where(
                    curvature_failure,
                    int(ProjectedGaussNewtonAttemptOutcome.FATAL_CURVATURE),
                    int(ProjectedGaussNewtonAttemptOutcome.FATAL_STEIHAUG),
                ).astype(jnp.int32)
                status = jnp.where(
                    curvature_failure,
                    int(ProjectedGaussNewtonStatus.FATAL_CURVATURE),
                    int(ProjectedGaussNewtonStatus.FATAL_STEIHAUG),
                ).astype(jnp.int32)
                return active._replace(
                    attempts=active.attempts + 1,
                    status=status,
                    fatal=jnp.asarray(True),
                    history=_record_attempt(
                        active.history,
                        active.attempts,
                        step_telemetry._replace(outcome=outcome),
                    ),
                )

            def evaluate_trial(active: _LoopState) -> _LoopState:
                tangent_step = steihaug.tangential_step
                with device_scope(PhaseId.GNTR_TRIAL_EVALUATION):
                    trial_coordinates = coordinates + tangent_step
                    trial_rows = materialize_joint_vjp_rows(
                        joint_value_constraints, trial_coordinates
                    )
                with device_scope(PhaseId.GNTR_NONLINEAR_CORRECTION):
                    first_correction = _normal_correction(trial_rows)
                    retraction = jax.lax.fori_loop(
                        1,
                        options.maximum_nonlinear_corrections,
                        lambda iteration, state: _extend_nonlinear_retraction(
                            joint_value_constraints,
                            options.corrected_feasibility_tolerance,
                            iteration,
                            state,
                        ),
                        _NonlinearRetractionState(
                            coordinates=coordinates
                            + (tangent_step + first_correction.step),
                            correction=first_correction.step,
                            maximum_individual_correction_norm=jnp.linalg.norm(
                                first_correction.step
                            ),
                            correction_path_norm=jnp.linalg.norm(first_correction.step),
                            nonlinear_corrections=jnp.asarray(1, dtype=jnp.int32),
                            all_finite=first_correction.all_finite,
                            factorization_relative_residual=(
                                first_correction.factorization_relative_residual
                            ),
                            linearized_constraint_relative_residual=(
                                first_correction.linearized_constraint_relative_residual
                            ),
                            solve_relative_residual=(
                                first_correction.solve_relative_residual
                            ),
                            solve_forward_error_bound=(
                                first_correction.solve_forward_error_bound
                            ),
                        ),
                    )
                    correction = retraction.correction
                    correction_all_finite = retraction.all_finite
                    correction_relative_residual = (
                        retraction.linearized_constraint_relative_residual
                    )
                    applied_step = tangent_step + correction
                with device_scope(PhaseId.GNTR_CORRECTED_CANDIDATE_EVALUATION):
                    candidate_coordinates = retraction.coordinates
                    candidate_objective, candidate_constraints = (
                        joint_value_constraints(candidate_coordinates)
                    )
                    applied_hessian = gn_hvp(applied_step)
                    predicted_reduction = -(
                        reconstructed_gradient @ applied_step
                        + 0.5 * (applied_step @ applied_hessian)
                    )
                    actual_reduction = rows.objective - candidate_objective
                    reduction_ratio = actual_reduction / predicted_reduction
                with device_scope(PhaseId.GNTR_ACCEPTANCE_RADIUS_UPDATE):
                    tangent_norm = jnp.linalg.norm(tangent_step)
                    correction_norm = jnp.linalg.norm(correction)
                    applied_norm = jnp.linalg.norm(applied_step)
                    correction_denominator = jnp.minimum(
                        tangent_norm, state.trust_radius
                    )
                    correction_step_ratio = jnp.where(
                        correction_denominator > 0.0,
                        correction_norm / correction_denominator,
                        jnp.where(correction_norm == 0.0, 0.0, jnp.inf),
                    )
                    corrected_radius_ratio = applied_norm / state.trust_radius
                    maximum_individual_correction_step_ratio = jnp.where(
                        correction_denominator > 0.0,
                        retraction.maximum_individual_correction_norm
                        / correction_denominator,
                        jnp.where(
                            retraction.maximum_individual_correction_norm == 0.0,
                            0.0,
                            jnp.inf,
                        ),
                    )
                    correction_path_step_ratio = jnp.where(
                        correction_denominator > 0.0,
                        retraction.correction_path_norm / correction_denominator,
                        jnp.where(
                            retraction.correction_path_norm == 0.0,
                            0.0,
                            jnp.inf,
                        ),
                    )
                    candidate_feasibility = jnp.linalg.norm(
                        candidate_constraints, ord=jnp.inf
                    )
                trial_rows_finite = (
                    jnp.all(jnp.isfinite(tangent_step))
                    & jnp.all(jnp.isfinite(trial_coordinates))
                    & jnp.isfinite(trial_rows.objective)
                    & jnp.all(jnp.isfinite(trial_rows.constraints))
                    & jnp.all(jnp.isfinite(trial_rows.objective_gradient))
                    & jnp.all(jnp.isfinite(trial_rows.constraint_jacobian))
                )
                candidate_finite = (
                    jnp.all(jnp.isfinite(correction))
                    & jnp.all(jnp.isfinite(applied_step))
                    & jnp.all(jnp.isfinite(candidate_coordinates))
                    & jnp.isfinite(candidate_objective)
                    & jnp.all(jnp.isfinite(candidate_constraints))
                    & jnp.all(jnp.isfinite(applied_hessian))
                    & jnp.isfinite(predicted_reduction)
                    & jnp.isfinite(actual_reduction)
                    & jnp.isfinite(applied_norm)
                )
                correction_valid = _trial_correction_certified(
                    correction_all_finite,
                    retraction.factorization_relative_residual,
                    correction_relative_residual,
                    retraction.solve_relative_residual,
                    retraction.solve_forward_error_bound,
                    linear_residual_tolerance=options.linear_residual_tolerance,
                    forward_error_tolerance=options.forward_error_tolerance,
                )
                feasibility_valid = (
                    candidate_feasibility <= options.corrected_feasibility_tolerance
                )
                bounds_valid = _correction_path_bounds_valid(
                    maximum_individual_correction_step_ratio,
                    correction_path_step_ratio,
                    retraction.nonlinear_corrections,
                    corrected_radius_ratio,
                    correction_ratio_limit=correction_ratio_limit,
                    corrected_radius_limit=corrected_radius_limit,
                )
                structurally_valid = (
                    trial_rows_finite
                    & candidate_finite
                    & correction_valid
                    & feasibility_valid
                    & bounds_valid
                )
                with device_scope(PhaseId.GNTR_ACCEPTANCE_RADIUS_UPDATE):
                    accepted = (
                        structurally_valid
                        & (predicted_reduction > 0.0)
                        & (actual_reduction > 0.0)
                    )
                    ratio_radius = _ratio_radius_update(
                        state.trust_radius,
                        applied_norm,
                        reduction_ratio,
                        minimum_radius,
                        maximum_radius,
                    )
                    forced_radius = jnp.clip(
                        0.25 * state.trust_radius, minimum_radius, maximum_radius
                    )
                    next_radius = jnp.where(
                        structurally_valid, ratio_radius, forced_radius
                    )
                    retry_outcome = jnp.where(
                        ~trial_rows_finite,
                        int(ProjectedGaussNewtonAttemptOutcome.RETRY_NONFINITE),
                        jnp.where(
                            ~correction_valid,
                            int(
                                ProjectedGaussNewtonAttemptOutcome.RETRY_CORRECTION_CERTIFICATE
                            ),
                            jnp.where(
                                ~candidate_finite,
                                int(ProjectedGaussNewtonAttemptOutcome.RETRY_NONFINITE),
                                jnp.where(
                                    ~feasibility_valid,
                                    int(
                                        ProjectedGaussNewtonAttemptOutcome.RETRY_FEASIBILITY
                                    ),
                                    jnp.where(
                                        ~bounds_valid,
                                        int(
                                            ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS
                                        ),
                                        int(
                                            ProjectedGaussNewtonAttemptOutcome.RETRY_OBJECTIVE
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                    outcome = jnp.where(
                        accepted,
                        int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED),
                        retry_outcome,
                    ).astype(jnp.int32)
                    accepted_steps = active.accepted_steps + accepted.astype(jnp.int32)
                accepted_state = ProjectedGaussNewtonAcceptedState(
                    optimizer_coordinates=candidate_coordinates,
                    objective=candidate_objective,
                    constraints=candidate_constraints,
                    scaled_feasibility_inf=candidate_feasibility,
                )
                quality_hit = accepted & jnp.asarray(
                    jax.lax.cond(
                        accepted,
                        accepted_state_quality_predicate,
                        _never_accept_quality,
                        accepted_state,
                    ),
                    dtype=jnp.bool_,
                )
                mechanism_this_step = (
                    accepted
                    & (active.accepted_steps >= 1)
                    & (steihaug.hvp_evaluations >= 2)
                    & (rotation >= options.mechanism_rotation_threshold)
                )
                with device_scope(PhaseId.GNTR_ACCEPTANCE_RADIUS_UPDATE):
                    accepted_optimizer_coordinates = jax.lax.cond(
                        accepted,
                        lambda ledger: ledger.at[accepted_steps].set(
                            candidate_coordinates
                        ),
                        lambda ledger: ledger,
                        active.accepted_optimizer_coordinates,
                    )
                    accepted_state_mask = jax.lax.cond(
                        accepted,
                        lambda mask: mask.at[accepted_steps].set(True),
                        lambda mask: mask,
                        active.accepted_state_mask,
                    )
                telemetry = step_telemetry._replace(
                    outcome=outcome,
                    accepted_step_number=jnp.where(
                        accepted, accepted_steps, jnp.asarray(0, dtype=jnp.int32)
                    ),
                    candidate_objective=candidate_objective,
                    candidate_feasibility_inf=candidate_feasibility,
                    actual_reduction=actual_reduction,
                    predicted_reduction=predicted_reduction,
                    reduction_ratio=reduction_ratio,
                    next_trust_radius=next_radius,
                    tangent_step_norm=tangent_norm,
                    correction_norm=correction_norm,
                    nonlinear_corrections=retraction.nonlinear_corrections,
                    applied_step_norm=applied_norm,
                    correction_step_ratio=correction_step_ratio,
                    maximum_individual_correction_step_ratio=(
                        maximum_individual_correction_step_ratio
                    ),
                    correction_path_step_ratio=correction_path_step_ratio,
                    corrected_radius_ratio=corrected_radius_ratio,
                    correction_relative_residual=correction_relative_residual,
                    correction_forward_error_bound=(
                        retraction.solve_forward_error_bound
                    ),
                    trial_gram_factorization_relative_residual=(
                        retraction.factorization_relative_residual
                    ),
                    trial_gram_solve_relative_residual=(
                        retraction.solve_relative_residual
                    ),
                )
                with device_scope(PhaseId.GNTR_ACCEPTANCE_RADIUS_UPDATE):
                    return active._replace(
                        coordinates=jnp.where(
                            accepted, candidate_coordinates, active.coordinates
                        ),
                        trust_radius=next_radius,
                        accepted_steps=accepted_steps,
                        attempts=active.attempts + 1,
                        retryable_rejections=(
                            active.retryable_rejections + (~accepted).astype(jnp.int32)
                        ),
                        status=jnp.where(
                            quality_hit,
                            int(ProjectedGaussNewtonStatus.DEVICE_QUALITY_CANDIDATE),
                            active.status,
                        ).astype(jnp.int32),
                        device_quality_candidate_reached=(
                            active.device_quality_candidate_reached | quality_hit
                        ),
                        first_quality_attempt=jnp.where(
                            quality_hit,
                            active.attempts + 1,
                            active.first_quality_attempt,
                        ),
                        first_quality_accepted_step=jnp.where(
                            quality_hit,
                            accepted_steps,
                            active.first_quality_accepted_step,
                        ),
                        accepted_optimizer_coordinates=(accepted_optimizer_coordinates),
                        accepted_state_mask=accepted_state_mask,
                        mechanism_exercised=(
                            active.mechanism_exercised | mechanism_this_step
                        ),
                        all_accepted_states_finite=(
                            active.all_accepted_states_finite
                            & (~accepted | candidate_finite)
                        ),
                        history=_record_attempt(
                            active.history, active.attempts, telemetry
                        ),
                    )

            return jax.lax.cond(
                steihaug_valid & terminal_curvature_valid,
                evaluate_trial,
                fatal_step,
                current,
            )

        return jax.lax.cond(
            current_base_valid & probe_curvature_valid,
            valid_current,
            fatal_current,
            state,
        )

    def subtrial_work(
        trial_state: _LoopState,
        result: _LoopState,
    ) -> tuple[jax.Array, ...]:
        index = trial_state.attempts
        history = result.history
        outcome = history.outcome[index]
        steihaug_iterations = history.steihaug_iterations[index]
        steihaug_hvp_evaluations = history.steihaug_hvp_evaluations[index]
        steihaug_solve_calls = history.steihaug_solve_calls[index]
        nonlinear_corrections = history.nonlinear_corrections[index]
        trial_evaluated = nonlinear_corrections > 0
        # Every shadow attempt repeats the two symmetry HVPs and one probe HVP;
        # a completed trial adds the applied-step prediction HVP.
        total_hvp_evaluations = (
            jnp.asarray(3, dtype=jnp.int32)
            + steihaug_hvp_evaluations
            + trial_evaluated.astype(jnp.int32)
        )
        joint_evaluations = jnp.asarray(1, dtype=jnp.int32) + jnp.where(
            trial_evaluated,
            jnp.asarray(options.maximum_nonlinear_corrections, dtype=jnp.int32)
            + nonlinear_corrections,
            jnp.asarray(0, dtype=jnp.int32),
        )
        # One current linearization plus one per correction are
        # derivative-bearing calls. Retraction checks and the final candidate
        # account for exactly maximum_nonlinear_corrections value-only calls.
        joint_linearizations = jnp.asarray(1, dtype=jnp.int32) + nonlinear_corrections
        joint_value_evaluations = jnp.where(
            trial_evaluated,
            jnp.asarray(options.maximum_nonlinear_corrections, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        objective_residual_linearizations = jnp.asarray(1, dtype=jnp.int32)
        gram_factorizations = jnp.asarray(1, dtype=jnp.int32) + nonlinear_corrections
        gram_solves = (
            # Current multiplier/tangency solves, correction solves, and, when
            # reached, Steihaug initial/final/terminal projections plus one
            # projected solve per HVP iteration.
            jnp.asarray(2, dtype=jnp.int32)
            + nonlinear_corrections
            + steihaug_solve_calls
            * (jnp.asarray(3, dtype=jnp.int32) + steihaug_hvp_evaluations)
        )
        retry_step_bounds = (
            (outcome == int(ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS))
            & (history.predicted_reduction[index] > 0.0)
            & (history.actual_reduction[index] > 0.0)
            & (result.trust_radius < trial_state.trust_radius)
        )
        return (
            outcome,
            history.actual_reduction[index],
            history.predicted_reduction[index],
            history.maximum_individual_correction_step_ratio[index],
            history.correction_path_step_ratio[index],
            history.corrected_radius_ratio[index],
            steihaug_iterations,
            steihaug_hvp_evaluations,
            steihaug_solve_calls,
            total_hvp_evaluations,
            nonlinear_corrections,
            joint_evaluations,
            joint_linearizations,
            joint_value_evaluations,
            objective_residual_linearizations,
            gram_factorizations,
            gram_solves,
            retry_step_bounds,
        )

    def safeguarded_attempt(state: _LoopState) -> _LoopState:
        integer_zeros = jnp.zeros((_STEP_BOUND_SAFEGUARD_SUBTRIALS,), dtype=jnp.int32)
        real_nans = jnp.full((_STEP_BOUND_SAFEGUARD_SUBTRIALS,), nan, dtype=dtype)
        # Subtrial 0 is seeded as pending at the incoming trust radius, so the
        # rolled loop below runs it exactly like every backtracked subtrial.
        ledger = _StepBoundSafeguardState(
            result=state,
            subtrial_count=jnp.asarray(0, dtype=jnp.int32),
            selected_subtrial_index=jnp.asarray(0, dtype=jnp.int32),
            subtrial_trust_radius=real_nans,
            subtrial_outcome=jnp.full(
                (_STEP_BOUND_SAFEGUARD_SUBTRIALS,),
                int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
                dtype=jnp.int32,
            ),
            subtrial_actual_reduction=real_nans,
            subtrial_predicted_reduction=real_nans,
            subtrial_maximum_individual_correction_step_ratio=real_nans,
            subtrial_correction_path_step_ratio=real_nans,
            subtrial_corrected_radius_ratio=real_nans,
            subtrial_steihaug_iterations=integer_zeros,
            subtrial_steihaug_hvp_evaluations=integer_zeros,
            subtrial_steihaug_solve_calls=integer_zeros,
            subtrial_total_hvp_evaluations=integer_zeros,
            subtrial_nonlinear_corrections=integer_zeros,
            subtrial_joint_evaluations=integer_zeros,
            subtrial_joint_linearizations=integer_zeros,
            subtrial_joint_value_evaluations=integer_zeros,
            subtrial_objective_residual_linearizations=integer_zeros,
            subtrial_gram_factorizations=integer_zeros,
            subtrial_gram_solves=integer_zeros,
            retry_step_bounds=jnp.asarray(True),
        )

        def subtrial(active: _StepBoundSafeguardState) -> _StepBoundSafeguardState:
            trial_state = state._replace(trust_radius=active.result.trust_radius)
            result = attempt(trial_state)
            (
                outcome,
                actual_reduction,
                predicted_reduction,
                maximum_individual_correction_step_ratio,
                correction_path_step_ratio,
                corrected_radius_ratio,
                steihaug_iterations,
                steihaug_hvp_evaluations,
                steihaug_solve_calls,
                total_hvp_evaluations,
                nonlinear_corrections,
                joint_evaluations,
                joint_linearizations,
                joint_value_evaluations,
                objective_residual_linearizations,
                gram_factorizations,
                gram_solves,
                retry_step_bounds,
            ) = subtrial_work(trial_state, result)
            index = active.subtrial_count
            return _StepBoundSafeguardState(
                result=result,
                subtrial_count=index + jnp.asarray(1, dtype=jnp.int32),
                selected_subtrial_index=index,
                subtrial_trust_radius=active.subtrial_trust_radius.at[index].set(
                    trial_state.trust_radius
                ),
                subtrial_outcome=active.subtrial_outcome.at[index].set(outcome),
                subtrial_actual_reduction=(
                    active.subtrial_actual_reduction.at[index].set(actual_reduction)
                ),
                subtrial_predicted_reduction=(
                    active.subtrial_predicted_reduction.at[index].set(
                        predicted_reduction
                    )
                ),
                subtrial_maximum_individual_correction_step_ratio=(
                    active.subtrial_maximum_individual_correction_step_ratio.at[
                        index
                    ].set(maximum_individual_correction_step_ratio)
                ),
                subtrial_correction_path_step_ratio=(
                    active.subtrial_correction_path_step_ratio.at[index].set(
                        correction_path_step_ratio
                    )
                ),
                subtrial_corrected_radius_ratio=(
                    active.subtrial_corrected_radius_ratio.at[index].set(
                        corrected_radius_ratio
                    )
                ),
                subtrial_steihaug_iterations=(
                    active.subtrial_steihaug_iterations.at[index].set(
                        steihaug_iterations
                    )
                ),
                subtrial_steihaug_hvp_evaluations=(
                    active.subtrial_steihaug_hvp_evaluations.at[index].set(
                        steihaug_hvp_evaluations
                    )
                ),
                subtrial_steihaug_solve_calls=(
                    active.subtrial_steihaug_solve_calls.at[index].set(
                        steihaug_solve_calls
                    )
                ),
                subtrial_total_hvp_evaluations=(
                    active.subtrial_total_hvp_evaluations.at[index].set(
                        total_hvp_evaluations
                    )
                ),
                subtrial_nonlinear_corrections=(
                    active.subtrial_nonlinear_corrections.at[index].set(
                        nonlinear_corrections
                    )
                ),
                subtrial_joint_evaluations=(
                    active.subtrial_joint_evaluations.at[index].set(joint_evaluations)
                ),
                subtrial_joint_linearizations=(
                    active.subtrial_joint_linearizations.at[index].set(
                        joint_linearizations
                    )
                ),
                subtrial_joint_value_evaluations=(
                    active.subtrial_joint_value_evaluations.at[index].set(
                        joint_value_evaluations
                    )
                ),
                subtrial_objective_residual_linearizations=(
                    active.subtrial_objective_residual_linearizations.at[index].set(
                        objective_residual_linearizations
                    )
                ),
                subtrial_gram_factorizations=(
                    active.subtrial_gram_factorizations.at[index].set(
                        gram_factorizations
                    )
                ),
                subtrial_gram_solves=active.subtrial_gram_solves.at[index].set(
                    gram_solves
                ),
                retry_step_bounds=retry_step_bounds,
            )

        def subtrial_when_pending(
            _index: int, current: _StepBoundSafeguardState
        ) -> _StepBoundSafeguardState:
            return jax.lax.cond(
                current.retry_step_bounds,
                subtrial,
                lambda live: live,
                current,
            )

        ledger = jax.lax.fori_loop(
            0,
            _STEP_BOUND_SAFEGUARD_SUBTRIALS,
            subtrial_when_pending,
            ledger,
        )

        index = state.attempts
        history = ledger.result.history._replace(
            subtrial_count=ledger.result.history.subtrial_count.at[index].set(
                ledger.subtrial_count
            ),
            selected_subtrial_index=(
                ledger.result.history.selected_subtrial_index.at[index].set(
                    ledger.selected_subtrial_index
                )
            ),
            subtrial_trust_radius=(
                ledger.result.history.subtrial_trust_radius.at[index].set(
                    ledger.subtrial_trust_radius
                )
            ),
            subtrial_outcome=ledger.result.history.subtrial_outcome.at[index].set(
                ledger.subtrial_outcome
            ),
            subtrial_actual_reduction=(
                ledger.result.history.subtrial_actual_reduction.at[index].set(
                    ledger.subtrial_actual_reduction
                )
            ),
            subtrial_predicted_reduction=(
                ledger.result.history.subtrial_predicted_reduction.at[index].set(
                    ledger.subtrial_predicted_reduction
                )
            ),
            subtrial_maximum_individual_correction_step_ratio=(
                ledger.result.history.subtrial_maximum_individual_correction_step_ratio.at[
                    index
                ].set(ledger.subtrial_maximum_individual_correction_step_ratio)
            ),
            subtrial_correction_path_step_ratio=(
                ledger.result.history.subtrial_correction_path_step_ratio.at[index].set(
                    ledger.subtrial_correction_path_step_ratio
                )
            ),
            subtrial_corrected_radius_ratio=(
                ledger.result.history.subtrial_corrected_radius_ratio.at[index].set(
                    ledger.subtrial_corrected_radius_ratio
                )
            ),
            subtrial_steihaug_iterations=(
                ledger.result.history.subtrial_steihaug_iterations.at[index].set(
                    ledger.subtrial_steihaug_iterations
                )
            ),
            subtrial_steihaug_hvp_evaluations=(
                ledger.result.history.subtrial_steihaug_hvp_evaluations.at[index].set(
                    ledger.subtrial_steihaug_hvp_evaluations
                )
            ),
            subtrial_steihaug_solve_calls=(
                ledger.result.history.subtrial_steihaug_solve_calls.at[index].set(
                    ledger.subtrial_steihaug_solve_calls
                )
            ),
            subtrial_total_hvp_evaluations=(
                ledger.result.history.subtrial_total_hvp_evaluations.at[index].set(
                    ledger.subtrial_total_hvp_evaluations
                )
            ),
            subtrial_nonlinear_corrections=(
                ledger.result.history.subtrial_nonlinear_corrections.at[index].set(
                    ledger.subtrial_nonlinear_corrections
                )
            ),
            subtrial_joint_evaluations=(
                ledger.result.history.subtrial_joint_evaluations.at[index].set(
                    ledger.subtrial_joint_evaluations
                )
            ),
            subtrial_joint_linearizations=(
                ledger.result.history.subtrial_joint_linearizations.at[index].set(
                    ledger.subtrial_joint_linearizations
                )
            ),
            subtrial_joint_value_evaluations=(
                ledger.result.history.subtrial_joint_value_evaluations.at[index].set(
                    ledger.subtrial_joint_value_evaluations
                )
            ),
            subtrial_objective_residual_linearizations=(
                ledger.result.history.subtrial_objective_residual_linearizations.at[
                    index
                ].set(ledger.subtrial_objective_residual_linearizations)
            ),
            subtrial_gram_factorizations=(
                ledger.result.history.subtrial_gram_factorizations.at[index].set(
                    ledger.subtrial_gram_factorizations
                )
            ),
            subtrial_gram_solves=(
                ledger.result.history.subtrial_gram_solves.at[index].set(
                    ledger.subtrial_gram_solves
                )
            ),
        )
        return ledger.result._replace(history=history)

    def iteration(_index: int, state: _LoopState) -> _LoopState:
        active = (
            ~state.fatal
            & ~state.device_quality_candidate_reached
            & (state.accepted_steps < options.maximum_accepted_steps)
            & (state.attempts < options.maximum_attempts)
        )
        attempt_body = (
            safeguarded_attempt if options.enable_step_bound_safeguard else attempt
        )
        return jax.lax.cond(active, attempt_body, lambda current: current, state)

    solved = jax.lax.fori_loop(0, options.maximum_attempts, iteration, initial_state)
    bounded_complete = solved.accepted_steps >= options.maximum_accepted_steps
    terminal_status = jnp.where(
        solved.fatal,
        solved.status,
        jnp.where(
            solved.device_quality_candidate_reached,
            int(ProjectedGaussNewtonStatus.DEVICE_QUALITY_CANDIDATE),
            jnp.where(
                bounded_complete,
                int(ProjectedGaussNewtonStatus.BOUNDED_COMPLETE),
                int(ProjectedGaussNewtonStatus.ATTEMPT_LIMIT),
            ),
        ),
    ).astype(jnp.int32)

    return ProjectedGaussNewtonLoopResult(
        optimizer_coordinates=solved.coordinates,
        trust_radius=solved.trust_radius,
        accepted_steps=solved.accepted_steps,
        attempts=solved.attempts,
        retryable_rejections=solved.retryable_rejections,
        status=terminal_status,
        fatal=solved.fatal,
        bounded_complete=bounded_complete,
        device_quality_candidate_reached=solved.device_quality_candidate_reached,
        first_quality_attempt=solved.first_quality_attempt,
        first_quality_accepted_step=solved.first_quality_accepted_step,
        accepted_optimizer_coordinates=solved.accepted_optimizer_coordinates,
        accepted_state_mask=solved.accepted_state_mask,
        mechanism_exercised=solved.mechanism_exercised,
        all_accepted_states_finite=solved.all_accepted_states_finite,
        history=solved.history,
    )


def finalize_projected_gauss_newton_trust_region(
    joint_value_constraints: JointValueConstraints,
    objective_residuals: ObjectiveResiduals,
    loop_result: ProjectedGaussNewtonLoopResult,
    *,
    options: ProjectedGaussNewtonOptions = _DEFAULT_OPTIONS,
) -> ProjectedGaussNewtonResult:
    """Build untimed endpoint derivatives and certificates from a loop result."""

    _validate_options(options)

    final_rows = materialize_joint_vjp_rows(
        joint_value_constraints, loop_result.optimizer_coordinates
    )
    final_projector = factor_certified_gram_projector(final_rows.constraint_jacobian)
    final_multipliers = solve_certified_gram(
        final_projector,
        -(final_rows.constraint_jacobian @ final_rows.objective_gradient),
    )
    final_stationarity = (
        final_rows.objective_gradient
        + final_rows.constraint_jacobian.T @ final_multipliers.solution
    )
    final_stationarity_inf = jnp.linalg.norm(final_stationarity, ord=jnp.inf)
    final_feasibility_inf = jnp.linalg.norm(final_rows.constraints, ord=jnp.inf)
    final_projected_stationarity = project_with_certified_gram(
        final_projector, final_stationarity
    )
    final_residual, final_residual_jvp = jax.linearize(
        objective_residuals, loop_result.optimizer_coordinates
    )
    final_residual_transpose = jax.linear_transpose(
        final_residual_jvp, jnp.zeros_like(loop_result.optimizer_coordinates)
    )

    def final_gn_hvp(vector: jax.Array) -> jax.Array:
        return final_residual_transpose(final_residual_jvp(vector))[0]

    final_reconstructed_objective = 0.5 * jnp.vdot(final_residual, final_residual)
    final_reconstructed_gradient = final_residual_transpose(final_residual)[0]
    final_value_defect = _scaled_defect(
        final_reconstructed_objective, final_rows.objective
    )
    final_gradient_defect = _gradient_scaled_defect(
        final_reconstructed_gradient, final_rows.objective_gradient
    )
    final_symmetry_defect = exact_hvp_bilinear_symmetry_relative_defect(
        final_gn_hvp, loop_result.optimizer_coordinates
    )
    final_probe_indices = jnp.arange(
        1,
        loop_result.optimizer_coordinates.shape[0] + 1,
        dtype=loop_result.optimizer_coordinates.dtype,
    )
    final_probe = jnp.sin(final_probe_indices)
    final_probe = final_probe / jnp.linalg.norm(final_probe)
    final_probe_action = final_gn_hvp(final_probe)
    final_probe_curvature = _normalized_curvature(final_probe, final_probe_action)
    final_finite = (
        jnp.all(jnp.isfinite(loop_result.optimizer_coordinates))
        & jnp.isfinite(final_rows.objective)
        & jnp.all(jnp.isfinite(final_rows.constraints))
        & jnp.all(jnp.isfinite(final_rows.objective_gradient))
        & jnp.all(jnp.isfinite(final_rows.constraint_jacobian))
        & jnp.all(jnp.isfinite(final_residual))
        & jnp.isfinite(final_reconstructed_objective)
        & jnp.all(jnp.isfinite(final_reconstructed_gradient))
        & final_projector.all_finite
        & final_multipliers.all_finite
        & final_projected_stationarity.all_finite
        & jnp.all(jnp.isfinite(final_stationarity))
        & jnp.isfinite(final_stationarity_inf)
        & jnp.isfinite(final_feasibility_inf)
        & jnp.isfinite(final_value_defect)
        & jnp.isfinite(final_gradient_defect)
        & jnp.isfinite(final_symmetry_defect)
        & jnp.all(jnp.isfinite(final_probe_action))
        & jnp.isfinite(final_probe_curvature)
    )
    final_certified = (
        final_finite
        & (
            final_projector.factorization_relative_residual
            <= options.linear_residual_tolerance
        )
        & (final_multipliers.relative_residual <= options.linear_residual_tolerance)
        & (final_multipliers.forward_error_bound < options.forward_error_tolerance)
        & (
            final_projected_stationarity.tangency_relative_residual
            <= options.linear_residual_tolerance
        )
        & (
            final_projected_stationarity.solve_relative_residual
            <= options.linear_residual_tolerance
        )
        & (
            final_projected_stationarity.solve_forward_error_bound
            < options.forward_error_tolerance
        )
        & (final_feasibility_inf <= options.corrected_feasibility_tolerance)
        & (final_value_defect <= options.residual_value_defect_tolerance)
        & (final_gradient_defect <= options.residual_gradient_defect_tolerance)
        & (final_symmetry_defect <= options.residual_gradient_defect_tolerance)
        & (final_probe_curvature >= -options.normalized_curvature_tolerance)
    )
    final_certificate = ProjectedGaussNewtonFinalCertificate(
        coordinates_finite=jnp.all(jnp.isfinite(loop_result.optimizer_coordinates)),
        residual_value_defect=final_value_defect,
        residual_gradient_defect=final_gradient_defect,
        hvp_symmetry_defect=final_symmetry_defect,
        probe_normalized_curvature=final_probe_curvature,
        gram_factorization_relative_residual=(
            final_projector.factorization_relative_residual
        ),
        multiplier_relative_residual=final_multipliers.relative_residual,
        multiplier_forward_error_bound=final_multipliers.forward_error_bound,
        projection_tangency_relative_residual=(
            final_projected_stationarity.tangency_relative_residual
        ),
        projection_solve_relative_residual=(
            final_projected_stationarity.solve_relative_residual
        ),
        projection_forward_error_bound=(
            final_projected_stationarity.solve_forward_error_bound
        ),
        all_finite=final_finite,
        certified=final_certified,
    )
    return ProjectedGaussNewtonResult(
        optimizer_coordinates=loop_result.optimizer_coordinates,
        multipliers=final_multipliers.solution,
        objective=final_rows.objective,
        constraints=final_rows.constraints,
        objective_gradient=final_rows.objective_gradient,
        constraint_jacobian=final_rows.constraint_jacobian,
        stationarity=final_stationarity,
        scaled_stationarity_inf=final_stationarity_inf,
        scaled_feasibility_inf=final_feasibility_inf,
        trust_radius=loop_result.trust_radius,
        accepted_steps=loop_result.accepted_steps,
        attempts=loop_result.attempts,
        retryable_rejections=loop_result.retryable_rejections,
        status=loop_result.status,
        fatal=loop_result.fatal,
        bounded_complete=loop_result.bounded_complete,
        mechanism_exercised=loop_result.mechanism_exercised,
        all_accepted_states_finite=loop_result.all_accepted_states_finite,
        all_finite=final_finite & loop_result.all_accepted_states_finite,
        final_certificate=final_certificate,
        usable=(
            ~loop_result.fatal
            & loop_result.all_accepted_states_finite
            & final_certified
        ),
        history=loop_result.history,
        device_quality_candidate_reached=(loop_result.device_quality_candidate_reached),
        first_quality_attempt=loop_result.first_quality_attempt,
        first_quality_accepted_step=loop_result.first_quality_accepted_step,
    )


def run_projected_gauss_newton_trust_region(
    joint_value_constraints: JointValueConstraints,
    objective_residuals: ObjectiveResiduals,
    initial_coordinates: jax.Array,
    *,
    options: ProjectedGaussNewtonOptions = _DEFAULT_OPTIONS,
    accepted_state_quality_predicate: AcceptedStateQualityPredicate = (
        _never_accept_quality
    ),
) -> ProjectedGaussNewtonResult:
    """Run the legacy combined loop and endpoint-finalization path."""

    loop_result = run_projected_gauss_newton_trust_region_loop(
        joint_value_constraints,
        objective_residuals,
        initial_coordinates,
        options=options,
        accepted_state_quality_predicate=accepted_state_quality_predicate,
    )
    return finalize_projected_gauss_newton_trust_region(
        joint_value_constraints,
        objective_residuals,
        loop_result,
        options=options,
    )


__all__ = (
    "AcceptedStateQualityPredicate",
    "ObjectiveResiduals",
    "ProjectedGaussNewtonAcceptedState",
    "ProjectedGaussNewtonAttemptOutcome",
    "ProjectedGaussNewtonFinalCertificate",
    "ProjectedGaussNewtonHistory",
    "ProjectedGaussNewtonLoopResult",
    "ProjectedGaussNewtonOptions",
    "ProjectedGaussNewtonResult",
    "ProjectedGaussNewtonStatus",
    "finalize_projected_gauss_newton_trust_region",
    "run_projected_gauss_newton_trust_region",
    "run_projected_gauss_newton_trust_region_loop",
)
