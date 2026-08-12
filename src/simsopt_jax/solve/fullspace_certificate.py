"""Fail-closed endpoint certification for the promoting CFS-SQP1 route.

This module deliberately owns a formulation-specific contract.  It does not
translate the joint equality-constrained solve into the nested solver's
``inner_success`` vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.geo.optimizers.dense_sqp import DenseSQPStatus
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceProblem,
    FullSpaceRawTerms,
    evaluate_fullspace,
    flatten_fullspace_constraints,
    fullspace_kkt_primitives,
)
from simsopt_jax.solve.fullspace import (
    CfsSqp1Policy,
    FullSpaceRoute,
    FullSpaceScaling,
    sqp_route_policy,
)
from simsopt_jax.solve.fullspace_sqp import (
    SCHEMA_VERSION as CFS_SQP1_RESULT_SCHEMA_VERSION,
)
from simsopt_jax.solve.fullspace_sqp import CfsSqp1Result

SCHEMA_VERSION: Final = "single-stage-fullspace-certificate-v1"
CFS_SQP1_CERTIFICATE_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-cfs-sqp1-endpoint-certificate-v1"
)
OBJECTIVE_MAXIMUM: Final = 4.4822247e-08
PROJECTION_STATE_INFINITY_TOLERANCE: Final = 1.0e-12
PROJECTION_OBJECTIVE_TOLERANCE: Final = 1.0e-15
PROJECTION_CONSTRAINT_INFINITY_TOLERANCE: Final = 1.0e-12
CROSS_EVALUATOR_RTOL: Final = 1.0e-12
CROSS_EVALUATOR_ATOL: Final = 1.0e-15
BRANCH_STATE_INFINITY_TOLERANCE: Final = 1.0e-10
TRACED_IOTA_TOLERANCE: Final = 1.0e-7
INACTIVE_HARDWARE_TERMS: Final = (
    "curvature",
    "curve_curve",
    "curve_surface",
    "surface_vessel",
)


class OptimizerTermination(StrEnum):
    """Normalized full-space solver terminal classifications."""

    CONVERGED = "CONVERGED"
    INCOMPLETE = "INCOMPLETE"
    NONFINITE = "NONFINITE"
    OPTIMIZER_REJECTED = "OPTIMIZER_REJECTED"


@dataclass(frozen=True, slots=True)
class FixedStateEvidence:
    """Independent observations of state intentionally excluded from ``z``."""

    expected_first_current: jax.Array
    observed_first_current: jax.Array
    expected_fixed_dofs: jax.Array
    observed_fixed_dofs: jax.Array


@dataclass(frozen=True, slots=True)
class InactiveHardwareEvidence:
    """Metrics and frozen zero weights in the canonical hardware-term order."""

    names: tuple[str, ...]
    metrics: jax.Array
    weights: jax.Array


@dataclass(frozen=True, slots=True)
class ObjectiveReferenceEvidence:
    """The historical native objective used by the engineering gate."""

    native_reference_objective: jax.Array


@dataclass(frozen=True, slots=True)
class CrossEvaluatorEvidence:
    """Independent objective evaluations at both exchanged endpoints."""

    performed: bool
    native_on_jax_endpoint_objective: jax.Array
    jax_on_native_endpoint_objective: jax.Array


@dataclass(frozen=True, slots=True)
class FieldLineEvidence:
    """Optimizer-independent Poincare and traced-iota checks."""

    performed: bool
    poincare_closed: bool
    traced_iota: jax.Array


@dataclass(frozen=True, slots=True)
class BranchEvidence:
    """Authoritative exact-solve reproduction of the endpoint inner root."""

    performed: bool
    exact_solve_succeeded: bool
    material_branch_switch: bool
    reproduced_state_infinity_difference: jax.Array
    basin_classification: str


@dataclass(frozen=True, slots=True)
class ProjectionEvidence:
    """Explicit pre/post states for an optional final exact projection."""

    evaluated: bool
    used: bool
    pre_state: jax.Array
    post_state: jax.Array


@dataclass(frozen=True, slots=True)
class EndpointNumerics:
    """Direct FP64 endpoint quantities used by the certificate."""

    state: jax.Array
    objective: jax.Array
    raw_objective_terms: FullSpaceRawTerms
    objective_ledger_consistent: bool
    constraints: jax.Array
    scaled_constraints: jax.Array
    objective_gradient: jax.Array
    stationarity_gradient: jax.Array
    boozer_residual_infinity_norm: jax.Array
    volume_residual_absolute: jax.Array
    scaled_feasibility_infinity_norm: jax.Array
    raw_kkt_stationarity_infinity_norm: jax.Array
    iota: jax.Array
    major_radius: jax.Array
    one_sided_length_penalty: jax.Array
    all_finite_fp64: bool


@dataclass(frozen=True, slots=True)
class CertificateChecks:
    """One independently inspectable Boolean per promotion boundary."""

    optimizer_termination: bool
    solver_result_consistent: bool
    finite_fp64: bool
    objective_ledger_consistent: bool
    scaled_feasibility: bool
    raw_kkt_stationarity: bool
    fixed_state_preserved: bool
    inactive_hardware_terms_valid: bool
    objective_threshold: bool
    objective_reference_valid: bool
    cross_evaluator: bool
    field_line: bool
    branch: bool
    projection_bound_to_solver_endpoint: bool
    projection_immaterial: bool
    pre_projection_certifiable: bool
    post_projection_certifiable: bool


@dataclass(frozen=True, slots=True)
class CfsAl1EndpointCertificate:
    """Complete host-side verdict and its immutable scientific evidence."""

    schema_version: str
    route: FullSpaceRoute
    termination: OptimizerTermination
    pre_projection: EndpointNumerics
    post_projection: EndpointNumerics
    multipliers: jax.Array
    inactive_hardware: InactiveHardwareEvidence
    objective_reference: ObjectiveReferenceEvidence
    cross_evaluator: CrossEvaluatorEvidence
    field_line: FieldLineEvidence
    branch: BranchEvidence
    projection: ProjectionEvidence
    checks: CertificateChecks
    certified: bool


@dataclass(frozen=True, slots=True)
class CfsSqp1EndpointCertificate(CfsAl1EndpointCertificate):
    """Distinct CFS-SQP1 envelope over the shared scientific evidence."""


# The evidence schema is formulation-specific but route-independent.  Retain the
# original public name for callers that already consume published receipts.
FullSpaceEndpointCertificate = CfsAl1EndpointCertificate


_CertificateT = TypeVar(
    "_CertificateT",
    bound=CfsAl1EndpointCertificate,
)


def _is_float64(array: jax.Array) -> bool:
    return np.asarray(array).dtype == np.dtype(np.float64)


def _finite_fp64(array: jax.Array) -> bool:
    value = np.asarray(array)
    return value.dtype == np.dtype(np.float64) and bool(np.all(np.isfinite(value)))


def _scalar_bool(value: jax.Array) -> bool:
    array = np.asarray(value)
    return array.shape == () and bool(array)


def _scalar_int(value: jax.Array) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("optimizer counter evidence must be an integer scalar")
    return int(array)


def _require_vector(name: str, value: jax.Array, size: int | None = None) -> None:
    array = np.asarray(value)
    if array.ndim != 1 or (size is not None and array.shape != (size,)):
        expected = "a vector" if size is None else f"shape ({size},)"
        raise ValueError(f"{name} must have {expected}")


def _require_scalar(name: str, value: jax.Array) -> None:
    if np.asarray(value).shape != ():
        raise ValueError(f"{name} must be a scalar")


def _endpoint_numerics(
    state: jax.Array,
    multipliers: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> EndpointNumerics:
    evaluation = evaluate_fullspace(state, problem)
    constraints = flatten_fullspace_constraints(evaluation.constraints)
    scaled_constraints = constraints * scaling.constraint_inverse_scale
    objective_gradient = jax.grad(
        lambda candidate: evaluate_fullspace(candidate, problem).weighted_total
    )(state)
    kkt = fullspace_kkt_primitives(state, multipliers, problem)
    arrays = (
        state,
        evaluation.weighted_total,
        constraints,
        scaled_constraints,
        objective_gradient,
        multipliers,
        kkt.stationarity_residual,
        evaluation.observables.iota,
        evaluation.observables.major_radius,
        evaluation.raw_terms.non_qs,
        evaluation.raw_terms.residual,
        evaluation.raw_terms.iota,
        evaluation.raw_terms.major_radius,
        evaluation.raw_terms.length,
    )
    recomposed_objective = (
        problem.config.non_qs_weight * evaluation.raw_terms.non_qs
        + problem.config.residual_weight * evaluation.raw_terms.residual
        + problem.config.iota_weight * evaluation.raw_terms.iota
        + problem.config.major_radius_weight * evaluation.raw_terms.major_radius
        + problem.config.length_weight * evaluation.raw_terms.length
    )
    return EndpointNumerics(
        state=state,
        objective=evaluation.weighted_total,
        raw_objective_terms=evaluation.raw_terms,
        objective_ledger_consistent=bool(
            np.array_equal(
                np.asarray(recomposed_objective),
                np.asarray(evaluation.weighted_total),
            )
        ),
        constraints=constraints,
        scaled_constraints=scaled_constraints,
        objective_gradient=objective_gradient,
        stationarity_gradient=kkt.stationarity_residual,
        boozer_residual_infinity_norm=jnp.max(jnp.abs(evaluation.constraints.boozer)),
        volume_residual_absolute=jnp.abs(evaluation.constraints.volume),
        scaled_feasibility_infinity_norm=jnp.max(jnp.abs(scaled_constraints)),
        raw_kkt_stationarity_infinity_norm=kkt.stationarity_inf,
        iota=evaluation.observables.iota,
        major_radius=evaluation.observables.major_radius,
        one_sided_length_penalty=evaluation.raw_terms.length,
        all_finite_fp64=all(_finite_fp64(array) for array in arrays),
    )


def _sqp_certificate_policy(route: FullSpaceRoute) -> CfsSqp1Policy:
    if route is not FullSpaceRoute.CFS_SQP1:
        raise ValueError(
            f"SQP endpoint certification requires CFS-SQP1; got {route.value}"
        )
    return sqp_route_policy(route)


def _endpoint_core_passes(
    endpoint: EndpointNumerics,
    route: FullSpaceRoute,
) -> tuple[bool, bool, bool]:
    policy = _sqp_certificate_policy(route)
    constraint_tolerance = policy.scaled_feasibility_tolerance
    stationarity_tolerance = policy.raw_kkt_stationarity_tolerance
    objective_maximum = policy.objective_maximum
    feasibility = bool(
        np.asarray(endpoint.scaled_feasibility_infinity_norm) <= constraint_tolerance
    )
    stationarity = bool(
        np.asarray(endpoint.raw_kkt_stationarity_infinity_norm)
        <= stationarity_tolerance
    )
    objective = bool(np.asarray(endpoint.objective) <= objective_maximum)
    return feasibility, stationarity, objective


def _fixed_state_passes(evidence: FixedStateEvidence) -> bool:
    _require_scalar("expected_first_current", evidence.expected_first_current)
    _require_scalar("observed_first_current", evidence.observed_first_current)
    _require_vector("expected_fixed_dofs", evidence.expected_fixed_dofs)
    _require_vector(
        "observed_fixed_dofs",
        evidence.observed_fixed_dofs,
        np.asarray(evidence.expected_fixed_dofs).size,
    )
    arrays = (
        evidence.expected_first_current,
        evidence.observed_first_current,
        evidence.expected_fixed_dofs,
        evidence.observed_fixed_dofs,
    )
    return bool(
        all(_finite_fp64(array) for array in arrays)
        and np.array_equal(
            np.asarray(evidence.expected_first_current),
            np.asarray(evidence.observed_first_current),
        )
        and np.array_equal(
            np.asarray(evidence.expected_fixed_dofs),
            np.asarray(evidence.observed_fixed_dofs),
        )
    )


def _normalized_sqp_termination(result: CfsSqp1Result) -> OptimizerTermination:
    optimizer = result.optimizer
    if (
        result.schema_version != CFS_SQP1_RESULT_SCHEMA_VERSION
        or result.route is not FullSpaceRoute.CFS_SQP1
    ):
        return OptimizerTermination.OPTIMIZER_REJECTED
    always_available_floating_evidence = (
        optimizer.optimizer_coordinates,
        optimizer.multipliers,
        optimizer.bfgs_matrix,
        optimizer.objective,
        optimizer.constraints,
        optimizer.objective_gradient,
        optimizer.constraint_jacobian,
        optimizer.stationarity,
        optimizer.merit_penalty,
        result.endpoint.physical_state,
        result.endpoint.physical_objective,
        result.endpoint.raw_constraints,
        result.endpoint.scaled_constraints,
        result.endpoint.scaled_multipliers,
        result.endpoint.raw_multipliers,
        result.endpoint.raw_stationarity_residual,
        result.optimizer_stationarity_tolerance,
        result.stationarity_scaling_error_infinity_norm,
    )
    kkt_solves = _scalar_int(optimizer.kkt_solves)
    solve_diagnostics = (
        optimizer.final_kkt_relative_residual,
        optimizer.final_kkt_reciprocal_condition,
        optimizer.final_kkt_solution_scaled_residual,
        optimizer.final_schur_relative_residual,
        optimizer.final_bfgs_cholesky_relative_pivot,
        optimizer.final_schur_cholesky_relative_pivot,
        optimizer.selected_regularization,
    )
    solve_diagnostics_available = (
        all(_finite_fp64(value) for value in solve_diagnostics)
        if kkt_solves > 0
        else all(
            _is_float64(value) and bool(np.all(np.isnan(np.asarray(value))))
            for value in solve_diagnostics
        )
    )
    if (
        not _scalar_bool(result.endpoint.all_finite)
        or not _scalar_bool(result.all_finite)
        or not _scalar_bool(optimizer.all_finite)
        or not _scalar_bool(optimizer.all_accepted_states_finite)
        or not all(_finite_fp64(value) for value in always_available_floating_evidence)
        or not solve_diagnostics_available
    ):
        return OptimizerTermination.NONFINITE

    policy = _sqp_certificate_policy(FullSpaceRoute.CFS_SQP1)
    status = _scalar_int(optimizer.status)
    counters = (
        _scalar_int(optimizer.iterations),
        _scalar_int(optimizer.joint_evaluations),
        _scalar_int(optimizer.derivative_builds),
        kkt_solves,
        _scalar_int(optimizer.line_search_evaluations),
        _scalar_int(optimizer.rejected_nonfinite_trials),
        _scalar_int(optimizer.bfgs_resets),
        _scalar_int(optimizer.regularization_uses),
        _scalar_int(optimizer.regularization_candidates_tested),
    )
    iterations, joint_evaluations, *other_counters = counters
    history = optimizer.history
    history_float_arrays = (
        history.objective,
        history.feasibility_infinity_norm,
        history.stationarity_infinity_norm,
        history.step_length,
        history.kkt_relative_residual,
    )
    history_status = np.asarray(history.status)
    history_shapes_valid = bool(
        all(
            np.asarray(value).shape == (policy.maximum_iterations,)
            for value in history_float_arrays
        )
        and history_status.shape == (policy.maximum_iterations,)
        and np.issubdtype(history_status.dtype, np.integer)
    )
    if (
        any(counter < 0 for counter in counters)
        or iterations > policy.maximum_iterations
        or joint_evaluations > policy.maximum_joint_evaluations
        or any(counter > policy.maximum_joint_evaluations for counter in other_counters)
        or not history_shapes_valid
        or status == int(DenseSQPStatus.RUNNING)
        or status
        in {
            int(DenseSQPStatus.ITERATION_LIMIT),
            int(DenseSQPStatus.EVALUATION_LIMIT),
        }
    ):
        return OptimizerTermination.INCOMPLETE
    valid_statuses = {int(member) for member in DenseSQPStatus}
    if not all(
        np.asarray(value).dtype == np.dtype(np.float64)
        and bool(np.all(np.isfinite(np.asarray(value)[:iterations])))
        for value in history_float_arrays
    ) or not set(history_status[:iterations]).issubset(valid_statuses):
        return OptimizerTermination.NONFINITE
    if (
        status == int(DenseSQPStatus.CONVERGED)
        and _scalar_bool(optimizer.converged)
        and not _scalar_bool(optimizer.fatal)
        and not _scalar_bool(optimizer.failed)
        and _scalar_bool(result.converged)
    ):
        return OptimizerTermination.CONVERGED
    return OptimizerTermination.OPTIMIZER_REJECTED


def _sqp_solver_result_is_consistent(
    result: CfsSqp1Result,
    endpoint: EndpointNumerics,
    scaling: FullSpaceScaling,
) -> bool:
    optimizer = result.optimizer
    reported = result.endpoint
    expected_physical_state = (
        scaling.bootstrap_anchor
        + optimizer.optimizer_coordinates * scaling.variable_scale
    )
    expected_raw_multipliers = scaling.constraint_inverse_scale * optimizer.multipliers
    expected_optimizer_stationarity = (
        scaling.variable_scale * endpoint.stationarity_gradient
    )
    policy = _sqp_certificate_policy(FullSpaceRoute.CFS_SQP1)
    expected_optimizer_tolerance = policy.raw_kkt_stationarity_tolerance * np.min(
        np.abs(np.asarray(scaling.variable_scale))
    )
    stationarity_scaling_error = np.max(
        np.abs(
            np.asarray(optimizer.stationarity)
            - np.asarray(expected_optimizer_stationarity)
        )
    )
    stationarity_comparison_scale = max(
        1.0,
        float(np.max(np.abs(np.asarray(optimizer.stationarity)))),
        float(np.max(np.abs(np.asarray(expected_optimizer_stationarity)))),
    )
    derivative_identity_tolerance = 1.0e-12 + 1.0e-10 * stationarity_comparison_scale
    kkt_solves = _scalar_int(optimizer.kkt_solves)
    if kkt_solves == 0:
        solve_diagnostics_consistent = bool(
            all(
                _is_float64(value) and bool(np.all(np.isnan(np.asarray(value))))
                for value in (
                    optimizer.final_kkt_relative_residual,
                    optimizer.final_kkt_reciprocal_condition,
                    optimizer.final_kkt_solution_scaled_residual,
                    optimizer.final_schur_relative_residual,
                    optimizer.final_bfgs_cholesky_relative_pivot,
                    optimizer.final_schur_cholesky_relative_pivot,
                    optimizer.selected_regularization,
                )
            )
        )
    else:
        solve_diagnostics_consistent = bool(
            _finite_fp64(optimizer.final_kkt_relative_residual)
            and _finite_fp64(optimizer.final_kkt_reciprocal_condition)
            and _finite_fp64(optimizer.final_kkt_solution_scaled_residual)
            and _finite_fp64(optimizer.final_schur_relative_residual)
            and _finite_fp64(optimizer.final_bfgs_cholesky_relative_pivot)
            and _finite_fp64(optimizer.final_schur_cholesky_relative_pivot)
            and _finite_fp64(optimizer.selected_regularization)
            and float(np.asarray(optimizer.final_kkt_relative_residual))
            <= policy.kkt_relative_residual_tolerance
            and float(np.asarray(optimizer.final_kkt_reciprocal_condition))
            > float(np.asarray(optimizer.final_kkt_solution_scaled_residual))
            and float(np.asarray(optimizer.final_kkt_solution_scaled_residual))
            <= policy.kkt_solution_scaled_residual_tolerance
            and (
                float(np.asarray(optimizer.final_kkt_solution_scaled_residual))
                / (
                    float(np.asarray(optimizer.final_kkt_reciprocal_condition))
                    - float(np.asarray(optimizer.final_kkt_solution_scaled_residual))
                )
                < policy.kkt_forward_error_tolerance
            )
            and float(np.asarray(optimizer.final_schur_relative_residual))
            <= policy.schur_relative_residual_tolerance
            and float(np.asarray(optimizer.selected_regularization))
            in policy.regularization_ladder
        )
    return bool(
        _scalar_bool(result.solver_result_consistent)
        and np.array_equal(
            np.asarray(expected_physical_state), np.asarray(reported.physical_state)
        )
        and np.array_equal(
            np.asarray(reported.physical_state), np.asarray(endpoint.state)
        )
        and np.array_equal(
            np.asarray(optimizer.multipliers),
            np.asarray(reported.scaled_multipliers),
        )
        and np.array_equal(
            np.asarray(expected_raw_multipliers),
            np.asarray(reported.raw_multipliers),
        )
        and np.isclose(
            np.asarray(optimizer.objective),
            np.asarray(endpoint.objective),
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        and np.array_equal(
            np.asarray(reported.physical_objective), np.asarray(endpoint.objective)
        )
        and np.allclose(
            np.asarray(optimizer.constraints),
            np.asarray(endpoint.scaled_constraints),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        and np.array_equal(
            np.asarray(reported.raw_constraints), np.asarray(endpoint.constraints)
        )
        and np.array_equal(
            np.asarray(reported.scaled_constraints),
            np.asarray(endpoint.scaled_constraints),
        )
        and np.array_equal(
            np.asarray(reported.raw_stationarity_residual),
            np.asarray(endpoint.stationarity_gradient),
        )
        and stationarity_scaling_error <= derivative_identity_tolerance
        and float(np.asarray(result.stationarity_scaling_error_infinity_norm))
        == stationarity_scaling_error
        and np.array_equal(
            np.asarray(reported.raw_constraint_infinity_norm),
            np.asarray(jnp.max(jnp.abs(endpoint.constraints))),
        )
        and np.array_equal(
            np.asarray(reported.scaled_constraint_infinity_norm),
            np.asarray(endpoint.scaled_feasibility_infinity_norm),
        )
        and np.array_equal(
            np.asarray(reported.raw_kkt_stationarity_infinity_norm),
            np.asarray(endpoint.raw_kkt_stationarity_infinity_norm),
        )
        and solve_diagnostics_consistent
        and float(np.asarray(result.optimizer_stationarity_tolerance))
        == expected_optimizer_tolerance
        and _scalar_int(optimizer.derivative_builds)
        == _scalar_int(optimizer.iterations) + 1
        and _scalar_int(optimizer.joint_evaluations)
        == _scalar_int(optimizer.derivative_builds)
        + _scalar_int(optimizer.line_search_evaluations)
    )


def _inactive_hardware_passes(evidence: InactiveHardwareEvidence) -> bool:
    if evidence.names != INACTIVE_HARDWARE_TERMS:
        return False
    _require_vector("inactive hardware metrics", evidence.metrics, 4)
    _require_vector("inactive hardware weights", evidence.weights, 4)
    return bool(
        _finite_fp64(evidence.metrics)
        and _finite_fp64(evidence.weights)
        and np.array_equal(np.asarray(evidence.weights), np.zeros(4, dtype=np.float64))
    )


def _projection_is_immaterial(
    projection: ProjectionEvidence,
    pre: EndpointNumerics,
    post: EndpointNumerics,
) -> bool:
    if not projection.evaluated:
        return False
    state_change = np.max(np.abs(np.asarray(post.state) - np.asarray(pre.state)))
    objective_change = abs(float(np.asarray(post.objective - pre.objective)))
    constraint_change = np.max(
        np.abs(np.asarray(post.constraints) - np.asarray(pre.constraints))
    )
    if not projection.used:
        return bool(
            state_change == 0.0 and objective_change == 0.0 and constraint_change == 0.0
        )
    return bool(
        state_change <= PROJECTION_STATE_INFINITY_TOLERANCE
        and objective_change <= PROJECTION_OBJECTIVE_TOLERANCE
        and constraint_change <= PROJECTION_CONSTRAINT_INFINITY_TOLERANCE
    )


def _validate_certificate_inputs(
    *,
    state_name: str,
    solver_physical_state: jax.Array,
    multiplier_name: str,
    raw_multipliers: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    objective_reference: ObjectiveReferenceEvidence,
    cross_evaluator: CrossEvaluatorEvidence,
    field_line: FieldLineEvidence,
    branch: BranchEvidence,
    projection: ProjectionEvidence,
) -> None:
    state_size = problem.layout.total_dof_count
    equality_size = scaling.constraint_inverse_scale.size
    _require_vector(state_name, solver_physical_state, state_size)
    _require_vector(multiplier_name, raw_multipliers, equality_size)
    _require_vector("projection.pre_state", projection.pre_state, state_size)
    _require_vector("projection.post_state", projection.post_state, state_size)
    _require_scalar(
        "native_reference_objective", objective_reference.native_reference_objective
    )
    _require_scalar(
        "native_on_jax_endpoint_objective",
        cross_evaluator.native_on_jax_endpoint_objective,
    )
    _require_scalar(
        "jax_on_native_endpoint_objective",
        cross_evaluator.jax_on_native_endpoint_objective,
    )
    _require_scalar("traced_iota", field_line.traced_iota)
    _require_scalar(
        "reproduced_state_infinity_difference",
        branch.reproduced_state_infinity_difference,
    )


def _certify_recomputed_endpoint(
    *,
    certificate_type: type[_CertificateT],
    schema_version: str,
    route: FullSpaceRoute,
    solver_physical_state: jax.Array,
    raw_multipliers: jax.Array,
    termination: OptimizerTermination,
    solver_result_consistent: bool,
    pre: EndpointNumerics,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    fixed_state: FixedStateEvidence,
    inactive_hardware: InactiveHardwareEvidence,
    objective_reference: ObjectiveReferenceEvidence,
    cross_evaluator: CrossEvaluatorEvidence,
    field_line: FieldLineEvidence,
    branch: BranchEvidence,
    projection: ProjectionEvidence,
) -> _CertificateT:
    """Build a certificate from route-normalized solver evidence.

    External audits are mandatory typed arguments.  An audit may explicitly
    report ``performed=False``, but absence can never be mistaken for success.
    """

    _validate_certificate_inputs(
        state_name="solver_physical_state",
        solver_physical_state=solver_physical_state,
        multiplier_name="raw_multipliers",
        raw_multipliers=raw_multipliers,
        problem=problem,
        scaling=scaling,
        objective_reference=objective_reference,
        cross_evaluator=cross_evaluator,
        field_line=field_line,
        branch=branch,
        projection=projection,
    )

    post = _endpoint_numerics(projection.post_state, raw_multipliers, problem, scaling)
    pre_feasible, pre_stationary, pre_objective = _endpoint_core_passes(pre, route)
    post_feasible, post_stationary, post_objective = _endpoint_core_passes(post, route)

    multipliers_finite_fp64 = _finite_fp64(raw_multipliers)
    finite_fp64 = bool(
        pre.all_finite_fp64
        and post.all_finite_fp64
        and multipliers_finite_fp64
        and _is_float64(scaling.constraint_inverse_scale)
    )
    fixed_pass = _fixed_state_passes(fixed_state)
    inactive_pass = _inactive_hardware_passes(inactive_hardware)
    native_reference = objective_reference.native_reference_objective
    reference_valid = bool(
        _finite_fp64(native_reference)
        and float(np.asarray(native_reference)) == OBJECTIVE_MAXIMUM
    )
    cross_pass = bool(
        cross_evaluator.performed
        and _finite_fp64(cross_evaluator.native_on_jax_endpoint_objective)
        and _finite_fp64(cross_evaluator.jax_on_native_endpoint_objective)
        and np.isclose(
            np.asarray(cross_evaluator.native_on_jax_endpoint_objective),
            np.asarray(post.objective),
            rtol=CROSS_EVALUATOR_RTOL,
            atol=CROSS_EVALUATOR_ATOL,
        )
        and np.isclose(
            np.asarray(cross_evaluator.jax_on_native_endpoint_objective),
            np.asarray(native_reference),
            rtol=CROSS_EVALUATOR_RTOL,
            atol=CROSS_EVALUATOR_ATOL,
        )
    )
    field_pass = bool(
        field_line.performed
        and field_line.poincare_closed
        and _finite_fp64(field_line.traced_iota)
        and np.isclose(
            np.asarray(field_line.traced_iota),
            np.asarray(post.iota),
            rtol=0.0,
            atol=TRACED_IOTA_TOLERANCE,
        )
    )
    branch_pass = bool(
        branch.performed
        and branch.exact_solve_succeeded
        and not branch.material_branch_switch
        and bool(branch.basin_classification.strip())
        and _finite_fp64(branch.reproduced_state_infinity_difference)
        and float(np.asarray(branch.reproduced_state_infinity_difference))
        <= BRANCH_STATE_INFINITY_TOLERANCE
    )
    projection_bound = bool(
        np.array_equal(
            np.asarray(projection.pre_state), np.asarray(solver_physical_state)
        )
    )
    projection_immaterial = _projection_is_immaterial(projection, pre, post)
    pre_certifiable = bool(
        pre.all_finite_fp64
        and pre.objective_ledger_consistent
        and pre_feasible
        and pre_stationary
        and pre_objective
    )
    post_certifiable = bool(
        post.all_finite_fp64
        and post.objective_ledger_consistent
        and post_feasible
        and post_stationary
        and post_objective
    )
    checks = CertificateChecks(
        optimizer_termination=termination is OptimizerTermination.CONVERGED,
        solver_result_consistent=solver_result_consistent,
        finite_fp64=finite_fp64,
        objective_ledger_consistent=bool(
            pre.objective_ledger_consistent and post.objective_ledger_consistent
        ),
        scaled_feasibility=post_feasible,
        raw_kkt_stationarity=post_stationary,
        fixed_state_preserved=fixed_pass,
        inactive_hardware_terms_valid=inactive_pass,
        objective_threshold=post_objective,
        objective_reference_valid=reference_valid,
        cross_evaluator=cross_pass,
        field_line=field_pass,
        branch=branch_pass,
        projection_bound_to_solver_endpoint=projection_bound,
        projection_immaterial=projection_immaterial,
        pre_projection_certifiable=pre_certifiable,
        post_projection_certifiable=post_certifiable,
    )
    certified = all(
        (
            checks.optimizer_termination,
            checks.solver_result_consistent,
            checks.finite_fp64,
            checks.objective_ledger_consistent,
            checks.scaled_feasibility,
            checks.raw_kkt_stationarity,
            checks.fixed_state_preserved,
            checks.inactive_hardware_terms_valid,
            checks.objective_threshold,
            checks.objective_reference_valid,
            checks.cross_evaluator,
            checks.field_line,
            checks.branch,
            checks.projection_bound_to_solver_endpoint,
            checks.projection_immaterial,
            checks.pre_projection_certifiable,
            checks.post_projection_certifiable,
        )
    )
    return certificate_type(
        schema_version=schema_version,
        route=route,
        termination=termination,
        pre_projection=pre,
        post_projection=post,
        multipliers=raw_multipliers,
        inactive_hardware=inactive_hardware,
        objective_reference=objective_reference,
        cross_evaluator=cross_evaluator,
        field_line=field_line,
        branch=branch,
        projection=projection,
        checks=checks,
        certified=certified,
    )


def certify_cfs_sqp1_endpoint(
    *,
    result: CfsSqp1Result,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    fixed_state: FixedStateEvidence,
    inactive_hardware: InactiveHardwareEvidence,
    objective_reference: ObjectiveReferenceEvidence,
    cross_evaluator: CrossEvaluatorEvidence,
    field_line: FieldLineEvidence,
    branch: BranchEvidence,
    projection: ProjectionEvidence,
) -> CfsSqp1EndpointCertificate:
    """Normalize CFS-SQP1 termination and recompute all scientific evidence."""

    _sqp_certificate_policy(FullSpaceRoute.CFS_SQP1)
    _validate_certificate_inputs(
        state_name="result.endpoint.physical_state",
        solver_physical_state=result.endpoint.physical_state,
        multiplier_name="result.endpoint.raw_multipliers",
        raw_multipliers=result.endpoint.raw_multipliers,
        problem=problem,
        scaling=scaling,
        objective_reference=objective_reference,
        cross_evaluator=cross_evaluator,
        field_line=field_line,
        branch=branch,
        projection=projection,
    )
    pre = _endpoint_numerics(
        projection.pre_state,
        result.endpoint.raw_multipliers,
        problem,
        scaling,
    )
    return _certify_recomputed_endpoint(
        certificate_type=CfsSqp1EndpointCertificate,
        schema_version=CFS_SQP1_CERTIFICATE_SCHEMA_VERSION,
        route=FullSpaceRoute.CFS_SQP1,
        solver_physical_state=result.endpoint.physical_state,
        raw_multipliers=result.endpoint.raw_multipliers,
        termination=_normalized_sqp_termination(result),
        solver_result_consistent=_sqp_solver_result_is_consistent(
            result,
            pre,
            scaling,
        ),
        pre=pre,
        problem=problem,
        scaling=scaling,
        fixed_state=fixed_state,
        inactive_hardware=inactive_hardware,
        objective_reference=objective_reference,
        cross_evaluator=cross_evaluator,
        field_line=field_line,
        branch=branch,
        projection=projection,
    )


__all__ = (
    "BRANCH_STATE_INFINITY_TOLERANCE",
    "CFS_SQP1_CERTIFICATE_SCHEMA_VERSION",
    "CROSS_EVALUATOR_ATOL",
    "CROSS_EVALUATOR_RTOL",
    "INACTIVE_HARDWARE_TERMS",
    "OBJECTIVE_MAXIMUM",
    "PROJECTION_CONSTRAINT_INFINITY_TOLERANCE",
    "PROJECTION_OBJECTIVE_TOLERANCE",
    "PROJECTION_STATE_INFINITY_TOLERANCE",
    "SCHEMA_VERSION",
    "TRACED_IOTA_TOLERANCE",
    "BranchEvidence",
    "CertificateChecks",
    "CfsAl1EndpointCertificate",
    "CfsSqp1EndpointCertificate",
    "CrossEvaluatorEvidence",
    "EndpointNumerics",
    "FieldLineEvidence",
    "FixedStateEvidence",
    "FullSpaceEndpointCertificate",
    "InactiveHardwareEvidence",
    "ObjectiveReferenceEvidence",
    "OptimizerTermination",
    "ProjectionEvidence",
    "certify_cfs_sqp1_endpoint",
)
