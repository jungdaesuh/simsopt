"""CFS-SQP1 adapter over the frozen single-stage full-space problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.geo.optimizers.dense_sqp import (
    DenseSQPOptions,
    DenseSQPResult,
    PreparedDenseSQP,
    materialize_joint_vjp_rows,
    prepare_dense_sqp,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceProblem,
    evaluate_fullspace,
    flatten_fullspace_constraints,
    fullspace_kkt_primitives,
)
from simsopt_jax.solve.fullspace import (
    FullSpaceRoute,
    FullSpaceScaling,
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
    sqp_route_policy,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-sqp1-result-v1"
CFS_SQP1_FINAL_MAXIMUM_ITERATIONS: Final = 100
CFS_SQP1_CANARY_MAXIMUM_ITERATIONS: Final = frozenset((1, 10, 100))


@dataclass(frozen=True, slots=True)
class CfsSqp1JointEvaluation:
    """Physical and scaled quantities from one optimizer coordinate."""

    physical_state: jax.Array
    physical_objective: jax.Array
    raw_constraints: jax.Array
    scaled_constraints: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    CfsSqp1JointEvaluation,
    data_fields=[
        "physical_state",
        "physical_objective",
        "raw_constraints",
        "scaled_constraints",
        "all_finite",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class CfsSqp1JointLinearization:
    """One retained joint primal and its exact objective/equality VJP rows."""

    physical_objective: jax.Array
    scaled_constraints: jax.Array
    objective_gradient: jax.Array
    constraint_jacobian: jax.Array
    joint_vjp_rows: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    CfsSqp1JointLinearization,
    data_fields=[
        "physical_objective",
        "scaled_constraints",
        "objective_gradient",
        "constraint_jacobian",
        "joint_vjp_rows",
        "all_finite",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class CfsSqp1EndpointDiagnostics:
    """Independently formed raw-physics endpoint KKT quantities."""

    physical_state: jax.Array
    physical_objective: jax.Array
    raw_constraints: jax.Array
    scaled_constraints: jax.Array
    scaled_multipliers: jax.Array
    raw_multipliers: jax.Array
    raw_stationarity_residual: jax.Array
    raw_constraint_infinity_norm: jax.Array
    scaled_constraint_infinity_norm: jax.Array
    raw_kkt_stationarity_infinity_norm: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    CfsSqp1EndpointDiagnostics,
    data_fields=[
        "physical_state",
        "physical_objective",
        "raw_constraints",
        "scaled_constraints",
        "scaled_multipliers",
        "raw_multipliers",
        "raw_stationarity_residual",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
        "all_finite",
    ],
    meta_fields=[],
)


def cfs_sqp1_joint_value_constraints(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(Phi, D q)`` at ``z = z0 + S u`` without host work."""

    physical_state = fullspace_physical_coordinates(
        optimizer_coordinates,
        scaling,
    )
    evaluation = evaluate_fullspace(physical_state, problem)
    raw_constraints = flatten_fullspace_constraints(evaluation.constraints)
    return (
        evaluation.weighted_total,
        raw_constraints * scaling.constraint_inverse_scale,
    )


def cfs_sqp1_joint_evaluation(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> CfsSqp1JointEvaluation:
    """Return named same-state SQP quantities for diagnostics and adapters."""

    physical_state = fullspace_physical_coordinates(
        optimizer_coordinates,
        scaling,
    )
    evaluation = evaluate_fullspace(physical_state, problem)
    raw_constraints = flatten_fullspace_constraints(evaluation.constraints)
    scaled_constraints = raw_constraints * scaling.constraint_inverse_scale
    all_finite = (
        jnp.all(jnp.isfinite(physical_state))
        & jnp.isfinite(evaluation.weighted_total)
        & jnp.all(jnp.isfinite(raw_constraints))
        & jnp.all(jnp.isfinite(scaled_constraints))
    )
    return CfsSqp1JointEvaluation(
        physical_state=physical_state,
        physical_objective=evaluation.weighted_total,
        raw_constraints=raw_constraints,
        scaled_constraints=scaled_constraints,
        all_finite=all_finite,
    )


def cfs_sqp1_joint_linearization(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> CfsSqp1JointLinearization:
    """Retain one joint primal and its exact objective-then-equality VJP rows."""

    policy = sqp_route_policy(FullSpaceRoute.CFS_SQP1)
    rows = materialize_joint_vjp_rows(
        lambda candidate: cfs_sqp1_joint_value_constraints(
            candidate,
            problem,
            scaling,
        ),
        optimizer_coordinates,
        batch_width=policy.reverse_row_batch_width,
    )
    all_finite = (
        jnp.isfinite(rows.objective)
        & jnp.all(jnp.isfinite(rows.constraints))
        & jnp.all(jnp.isfinite(rows.objective_gradient))
        & jnp.all(jnp.isfinite(rows.constraint_jacobian))
        & jnp.all(jnp.isfinite(rows.joint_rows))
    )
    return CfsSqp1JointLinearization(
        physical_objective=rows.objective,
        scaled_constraints=rows.constraints,
        objective_gradient=rows.objective_gradient,
        constraint_jacobian=rows.constraint_jacobian,
        joint_vjp_rows=rows.joint_rows,
        all_finite=all_finite,
    )


def cfs_sqp1_joint_vjp_rows(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Materialize the exact objective row followed by all 255 equality rows."""

    return cfs_sqp1_joint_linearization(
        optimizer_coordinates,
        problem,
        scaling,
    ).joint_vjp_rows


def cfs_sqp1_raw_multipliers(
    scaled_multipliers: jax.Array,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Map scaled equality duals to the raw Boozer-then-volume convention."""

    return scaling.constraint_inverse_scale * scaled_multipliers


def cfs_sqp1_endpoint_diagnostics(
    optimizer_coordinates: jax.Array,
    scaled_multipliers: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> CfsSqp1EndpointDiagnostics:
    """Recompute raw KKT evidence independently of the generic SQP result."""

    joint = cfs_sqp1_joint_evaluation(
        optimizer_coordinates,
        problem,
        scaling,
    )
    raw_multipliers = cfs_sqp1_raw_multipliers(scaled_multipliers, scaling)
    kkt = fullspace_kkt_primitives(
        joint.physical_state,
        raw_multipliers,
        problem,
    )
    all_finite = (
        joint.all_finite
        & jnp.all(jnp.isfinite(scaled_multipliers))
        & jnp.all(jnp.isfinite(raw_multipliers))
        & kkt.all_finite
    )
    return CfsSqp1EndpointDiagnostics(
        physical_state=joint.physical_state,
        physical_objective=joint.physical_objective,
        raw_constraints=joint.raw_constraints,
        scaled_constraints=joint.scaled_constraints,
        scaled_multipliers=scaled_multipliers,
        raw_multipliers=raw_multipliers,
        raw_stationarity_residual=kkt.stationarity_residual,
        raw_constraint_infinity_norm=jnp.max(jnp.abs(joint.raw_constraints)),
        scaled_constraint_infinity_norm=jnp.max(jnp.abs(joint.scaled_constraints)),
        raw_kkt_stationarity_infinity_norm=kkt.stationarity_inf,
        all_finite=all_finite,
    )


@dataclass(frozen=True, slots=True)
class CfsSqp1Result:
    """Route-specific result with generic telemetry and raw endpoint evidence."""

    schema_version: str
    route: FullSpaceRoute
    optimizer: DenseSQPResult
    endpoint: CfsSqp1EndpointDiagnostics
    optimizer_stationarity_tolerance: jax.Array
    stationarity_scaling_error_infinity_norm: jax.Array
    solver_result_consistent: jax.Array
    all_finite: jax.Array
    converged: jax.Array

    @property
    def status(self) -> jax.Array:
        return self.optimizer.status

    @property
    def failed(self) -> jax.Array:
        return self.optimizer.failed

    @property
    def fatal(self) -> jax.Array:
        return self.optimizer.fatal

    @property
    def optimizer_coordinates(self) -> jax.Array:
        return self.optimizer.optimizer_coordinates

    @property
    def physical_state(self) -> jax.Array:
        return self.endpoint.physical_state

    @property
    def scaled_multipliers(self) -> jax.Array:
        return self.endpoint.scaled_multipliers

    @property
    def raw_multipliers(self) -> jax.Array:
        return self.endpoint.raw_multipliers

    @property
    def physical_objective(self) -> jax.Array:
        return self.endpoint.physical_objective

    @property
    def raw_constraint_infinity_norm(self) -> jax.Array:
        return self.endpoint.raw_constraint_infinity_norm

    @property
    def scaled_constraint_infinity_norm(self) -> jax.Array:
        return self.endpoint.scaled_constraint_infinity_norm

    @property
    def raw_kkt_stationarity_infinity_norm(self) -> jax.Array:
        return self.endpoint.raw_kkt_stationarity_infinity_norm

    @property
    def iterations(self) -> jax.Array:
        return self.optimizer.iterations

    @property
    def joint_evaluations(self) -> jax.Array:
        return self.optimizer.joint_evaluations

    @property
    def derivative_builds(self) -> jax.Array:
        return self.optimizer.derivative_builds

    @property
    def kkt_solves(self) -> jax.Array:
        return self.optimizer.kkt_solves

    @property
    def line_search_evaluations(self) -> jax.Array:
        return self.optimizer.line_search_evaluations

    @property
    def rejected_nonfinite_trials(self) -> jax.Array:
        return self.optimizer.rejected_nonfinite_trials

    @property
    def bfgs_resets(self) -> jax.Array:
        return self.optimizer.bfgs_resets

    @property
    def regularization_uses(self) -> jax.Array:
        return self.optimizer.regularization_uses

    @property
    def final_kkt_relative_residual(self) -> jax.Array:
        return self.optimizer.final_kkt_relative_residual

    @property
    def final_kkt_reciprocal_condition(self) -> jax.Array:
        return self.optimizer.final_kkt_reciprocal_condition

    @property
    def final_kkt_solution_scaled_residual(self) -> jax.Array:
        return self.optimizer.final_kkt_solution_scaled_residual

    @property
    def final_schur_relative_residual(self) -> jax.Array:
        return self.optimizer.final_schur_relative_residual

    @property
    def selected_regularization(self) -> jax.Array:
        return self.optimizer.selected_regularization

    @property
    def regularization_candidates_tested(self) -> jax.Array:
        return self.optimizer.regularization_candidates_tested

    @property
    def all_accepted_states_finite(self) -> jax.Array:
        return self.optimizer.all_accepted_states_finite


@dataclass(frozen=True, slots=True)
class PreparedCfsSqp1:
    """Prepared CFS-SQP1 program starting from one frozen full-space state."""

    problem: FullSpaceProblem
    scaling: FullSpaceScaling
    initial_optimizer_coordinates: jax.Array
    initial_scaled_multipliers: jax.Array
    optimizer_stationarity_tolerance: float
    optimizer: PreparedDenseSQP

    def run_solver(self) -> DenseSQPResult:
        """Dispatch only the compiled solver from pristine prepared inputs."""

        return self.optimizer.run(
            self.initial_optimizer_coordinates,
            self.initial_scaled_multipliers,
        )

    def finalize_result(
        self,
        optimizer_result: DenseSQPResult,
    ) -> CfsSqp1Result:
        """Form independent raw endpoint diagnostics outside timed execution."""

        endpoint = cfs_sqp1_endpoint_diagnostics(
            optimizer_result.optimizer_coordinates,
            optimizer_result.multipliers,
            self.problem,
            self.scaling,
        )
        policy = sqp_route_policy(FullSpaceRoute.CFS_SQP1)
        optimizer_all_finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (
                        optimizer_result.optimizer_coordinates,
                        optimizer_result.multipliers,
                        optimizer_result.bfgs_matrix,
                        optimizer_result.objective,
                        optimizer_result.constraints,
                        optimizer_result.objective_gradient,
                        optimizer_result.constraint_jacobian,
                        optimizer_result.stationarity,
                        optimizer_result.merit_penalty,
                    )
                )
            )
        )
        solve_diagnostics_finite = (optimizer_result.kkt_solves == 0) | (
            jnp.isfinite(optimizer_result.final_kkt_relative_residual)
            & jnp.isfinite(optimizer_result.final_kkt_reciprocal_condition)
            & jnp.isfinite(optimizer_result.final_kkt_solution_scaled_residual)
            & jnp.isfinite(optimizer_result.final_schur_relative_residual)
            & jnp.isfinite(optimizer_result.selected_regularization)
        )
        all_finite = (
            endpoint.all_finite
            & optimizer_result.all_finite
            & optimizer_result.all_accepted_states_finite
            & optimizer_all_finite
            & solve_diagnostics_finite
        )
        expected_optimizer_stationarity = (
            self.scaling.variable_scale * endpoint.raw_stationarity_residual
        )
        stationarity_scaling_error = jnp.max(
            jnp.abs(optimizer_result.stationarity - expected_optimizer_stationarity)
        )
        dtype = endpoint.physical_objective.dtype
        comparison_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=dtype),
            jnp.maximum(
                jnp.max(jnp.abs(optimizer_result.stationarity)),
                jnp.max(jnp.abs(expected_optimizer_stationarity)),
            ),
        )
        derivative_identity_tolerance = (
            jnp.asarray(1.0e-12, dtype=dtype)
            + jnp.asarray(1.0e-10, dtype=dtype) * comparison_scale
        )
        solver_result_consistent = (
            jnp.isclose(
                optimizer_result.objective,
                endpoint.physical_objective,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            & jnp.allclose(
                optimizer_result.constraints,
                endpoint.scaled_constraints,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            & (stationarity_scaling_error <= derivative_identity_tolerance)
        )
        converged = (
            optimizer_result.converged
            & (~optimizer_result.fatal)
            & (~optimizer_result.failed)
            & all_finite
            & solver_result_consistent
            & (
                endpoint.physical_objective
                <= jnp.asarray(
                    policy.objective_maximum,
                    dtype=endpoint.physical_objective.dtype,
                )
            )
            & (
                endpoint.scaled_constraint_infinity_norm
                <= jnp.asarray(
                    policy.scaled_feasibility_tolerance,
                    dtype=endpoint.physical_objective.dtype,
                )
            )
            & (
                endpoint.raw_kkt_stationarity_infinity_norm
                <= jnp.asarray(
                    policy.raw_kkt_stationarity_tolerance,
                    dtype=endpoint.physical_objective.dtype,
                )
            )
        )
        return CfsSqp1Result(
            schema_version=SCHEMA_VERSION,
            route=FullSpaceRoute.CFS_SQP1,
            optimizer=optimizer_result,
            endpoint=endpoint,
            optimizer_stationarity_tolerance=jnp.asarray(
                self.optimizer_stationarity_tolerance,
                dtype=endpoint.physical_objective.dtype,
            ),
            stationarity_scaling_error_infinity_norm=(stationarity_scaling_error),
            solver_result_consistent=solver_result_consistent,
            all_finite=all_finite,
            converged=converged,
        )

    def run(self) -> CfsSqp1Result:
        """Run the solver and finalize its endpoint for compatibility callers."""

        return self.finalize_result(self.run_solver())


def _optimizer_stationarity_tolerance(
    scaling: FullSpaceScaling,
) -> float:
    """Return a sufficient optimizer-coordinate bound for raw stationarity."""

    policy = sqp_route_policy(FullSpaceRoute.CFS_SQP1)
    minimum_variable_scale = float(np.min(np.abs(np.asarray(scaling.variable_scale))))
    if not np.isfinite(minimum_variable_scale) or minimum_variable_scale <= 0.0:
        raise ValueError("CFS-SQP1 variable scaling must be finite and positive")
    return policy.raw_kkt_stationarity_tolerance * minimum_variable_scale


def _dense_sqp_options(
    optimizer_stationarity_tolerance: float,
    maximum_iterations: int,
) -> DenseSQPOptions:
    policy = sqp_route_policy(FullSpaceRoute.CFS_SQP1)
    return DenseSQPOptions(
        maximum_iterations=maximum_iterations,
        maximum_joint_evaluations=policy.maximum_joint_evaluations,
        reverse_row_batch_width=policy.reverse_row_batch_width,
        objective_maximum=policy.objective_maximum,
        feasibility_tolerance=policy.scaled_feasibility_tolerance,
        stationarity_tolerance=optimizer_stationarity_tolerance,
        maximum_consecutive_bfgs_resets=(policy.maximum_consecutive_bfgs_resets),
        regularization_ladder=policy.regularization_ladder,
        kkt_relative_residual_tolerance=(policy.kkt_relative_residual_tolerance),
        schur_relative_residual_tolerance=(policy.schur_relative_residual_tolerance),
        kkt_forward_error_tolerance=policy.kkt_forward_error_tolerance,
        kkt_solution_scaled_residual_tolerance=(
            policy.kkt_solution_scaled_residual_tolerance
        ),
        curvature_fraction=policy.powell_curvature_fraction,
        initial_bfgs_identity_scale=policy.initial_bfgs_identity_scale,
        merit_initial=policy.initial_merit_penalty,
        merit_multiplier_margin=policy.merit_multiplier_margin,
        armijo_coefficient=policy.armijo_coefficient,
        candidate_steps=policy.candidate_step_sizes,
        maximum_identity_retries=policy.maximum_identity_retries,
    )


def prepare_cfs_sqp1(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    *,
    maximum_iterations: int = CFS_SQP1_FINAL_MAXIMUM_ITERATIONS,
) -> PreparedCfsSqp1:
    """Prepare the frozen callback-free CFS-SQP1 solve."""

    if (
        type(maximum_iterations) is not int
        or maximum_iterations not in CFS_SQP1_CANARY_MAXIMUM_ITERATIONS
    ):
        raise ValueError("CFS-SQP1 maximum_iterations must be one of {1, 10, 100}")

    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    initial_optimizer_coordinates = fullspace_optimizer_coordinates(
        initial_physical_state,
        scaling,
    )
    optimizer_stationarity_tolerance = _optimizer_stationarity_tolerance(scaling)

    def joint_value_constraints(
        optimizer_coordinates: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return cfs_sqp1_joint_value_constraints(
            optimizer_coordinates,
            problem,
            scaling,
        )

    optimizer = prepare_dense_sqp(
        joint_value_constraints,
        initial_optimizer_coordinates,
        options=_dense_sqp_options(
            optimizer_stationarity_tolerance,
            maximum_iterations,
        ),
    )
    policy = sqp_route_policy(FullSpaceRoute.CFS_SQP1)
    initial_scaled_multipliers = jnp.full(
        scaling.constraint_inverse_scale.shape,
        policy.initial_multiplier,
        dtype=initial_optimizer_coordinates.dtype,
    )
    return PreparedCfsSqp1(
        problem=problem,
        scaling=scaling,
        initial_optimizer_coordinates=initial_optimizer_coordinates,
        initial_scaled_multipliers=initial_scaled_multipliers,
        optimizer_stationarity_tolerance=optimizer_stationarity_tolerance,
        optimizer=optimizer,
    )


__all__ = (
    "CFS_SQP1_CANARY_MAXIMUM_ITERATIONS",
    "CFS_SQP1_FINAL_MAXIMUM_ITERATIONS",
    "SCHEMA_VERSION",
    "CfsSqp1EndpointDiagnostics",
    "CfsSqp1JointEvaluation",
    "CfsSqp1JointLinearization",
    "CfsSqp1Result",
    "PreparedCfsSqp1",
    "cfs_sqp1_endpoint_diagnostics",
    "cfs_sqp1_joint_evaluation",
    "cfs_sqp1_joint_linearization",
    "cfs_sqp1_joint_value_constraints",
    "cfs_sqp1_joint_vjp_rows",
    "cfs_sqp1_raw_multipliers",
    "prepare_cfs_sqp1",
)
