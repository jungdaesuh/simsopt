"""CFS-FTR1 adapter over the frozen single-stage full-space problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.geo.optimizers.filter_trust_region_sqp import (
    FilterTrustRegionSQPOptions,
    FilterTrustRegionSQPResult,
    PreparedFilterTrustRegionSQP,
    prepare_filter_trust_region_sqp,
)
from simsopt_jax.objectives.single_stage_fullspace import FullSpaceProblem
from simsopt_jax.solve.fullspace import (
    FullSpaceRoute,
    FullSpaceScaling,
    ftr_route_policy,
    fullspace_optimizer_coordinates,
    fullspace_scaling_from_bootstrap,
)
from simsopt_jax.solve.fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_value_constraints,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-ftr1-result-v1"
CFS_FTR1_FINAL_MAXIMUM_ITERATIONS: Final = 100
CFS_FTR1_CANARY_MAXIMUM_ITERATIONS: Final = frozenset((10, 100))


@dataclass(frozen=True, slots=True)
class CfsFtr1Result:
    """Route result combining generic FTR telemetry and raw endpoint evidence."""

    schema_version: str
    route: FullSpaceRoute
    optimizer: FilterTrustRegionSQPResult
    endpoint: CfsSqp1EndpointDiagnostics
    optimizer_stationarity_tolerance: jax.Array
    stationarity_scaling_error_infinity_norm: jax.Array
    solver_result_consistent: jax.Array
    solve_certificates_valid: jax.Array
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
    def accepted_iterations(self) -> jax.Array:
        return self.optimizer.accepted_iterations

    @property
    def derivative_builds(self) -> jax.Array:
        return self.optimizer.derivative_builds

    @property
    def radius(self) -> jax.Array:
        return self.optimizer.radius

    @property
    def final_normal_relative_residual(self) -> jax.Array:
        return self.optimizer.final_normal_relative_residual

    @property
    def final_normal_forward_error_bound(self) -> jax.Array:
        return self.optimizer.final_normal_forward_error_bound

    @property
    def final_tangency_relative_residual(self) -> jax.Array:
        return self.optimizer.final_tangency_relative_residual

    @property
    def final_multiplier_projection_relative_residual(self) -> jax.Array:
        return self.optimizer.final_multiplier_projection_relative_residual

    @property
    def final_multiplier_projection_forward_error_bound(self) -> jax.Array:
        return self.optimizer.final_multiplier_projection_forward_error_bound

    @property
    def all_accepted_states_finite(self) -> jax.Array:
        return self.optimizer.all_accepted_states_finite


@dataclass(frozen=True, slots=True)
class PreparedCfsFtr1:
    """Prepared CFS-FTR1 program bound to one full-space problem and scaling."""

    problem: FullSpaceProblem
    scaling: FullSpaceScaling
    initial_optimizer_coordinates: jax.Array
    initial_scaled_multipliers: jax.Array
    optimizer_stationarity_tolerance: float
    optimizer: PreparedFilterTrustRegionSQP

    def run_solver(self) -> FilterTrustRegionSQPResult:
        """Dispatch only the compiled device-resident optimizer."""

        return self.optimizer.run(
            self.initial_optimizer_coordinates,
            self.initial_scaled_multipliers,
        )

    def finalize_result(
        self,
        optimizer_result: FilterTrustRegionSQPResult,
    ) -> CfsFtr1Result:
        """Form independent raw endpoint diagnostics outside timed execution."""

        endpoint = cfs_sqp1_endpoint_diagnostics(
            optimizer_result.optimizer_coordinates,
            optimizer_result.multipliers,
            self.problem,
            self.scaling,
        )
        policy = ftr_route_policy(FullSpaceRoute.CFS_FTR1)
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
        optimizer_arrays_finite = jnp.all(
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
                        optimizer_result.radius,
                    )
                )
            )
        )
        retained_solve_certificates_finite = (
            jnp.isfinite(optimizer_result.final_normal_relative_residual)
            & jnp.isfinite(optimizer_result.final_normal_forward_error_bound)
            & jnp.isfinite(optimizer_result.final_tangency_relative_residual)
            & jnp.isfinite(
                optimizer_result.final_multiplier_projection_relative_residual
            )
            & jnp.isfinite(
                optimizer_result.final_multiplier_projection_forward_error_bound
            )
        )
        retained_solve_certificates_valid = (
            retained_solve_certificates_finite
            & (
                optimizer_result.final_normal_relative_residual
                <= jnp.asarray(
                    policy.linear_solve_relative_residual_tolerance,
                    dtype=dtype,
                )
            )
            & (
                optimizer_result.final_tangency_relative_residual
                <= jnp.asarray(
                    policy.tangency_relative_residual_tolerance,
                    dtype=dtype,
                )
            )
            & (
                optimizer_result.final_normal_forward_error_bound
                < jnp.asarray(
                    policy.linear_solve_forward_error_tolerance,
                    dtype=dtype,
                )
            )
            & (
                optimizer_result.final_multiplier_projection_relative_residual
                <= jnp.asarray(
                    policy.linear_solve_relative_residual_tolerance,
                    dtype=dtype,
                )
            )
            & (
                optimizer_result.final_multiplier_projection_forward_error_bound
                < jnp.asarray(
                    policy.linear_solve_forward_error_tolerance,
                    dtype=dtype,
                )
            )
        )
        all_finite = (
            endpoint.all_finite
            & optimizer_result.all_finite
            & optimizer_result.all_accepted_states_finite
            & optimizer_arrays_finite
            & retained_solve_certificates_finite
        )
        converged = (
            optimizer_result.converged
            & (~optimizer_result.fatal)
            & (~optimizer_result.failed)
            & all_finite
            & solver_result_consistent
            & retained_solve_certificates_valid
            & (
                endpoint.physical_objective
                <= jnp.asarray(policy.objective_maximum, dtype=dtype)
            )
            & (
                endpoint.scaled_constraint_infinity_norm
                <= jnp.asarray(policy.scaled_feasibility_tolerance, dtype=dtype)
            )
            & (
                endpoint.raw_kkt_stationarity_infinity_norm
                <= jnp.asarray(policy.raw_kkt_stationarity_tolerance, dtype=dtype)
            )
        )
        return CfsFtr1Result(
            schema_version=SCHEMA_VERSION,
            route=FullSpaceRoute.CFS_FTR1,
            optimizer=optimizer_result,
            endpoint=endpoint,
            optimizer_stationarity_tolerance=jnp.asarray(
                self.optimizer_stationarity_tolerance,
                dtype=dtype,
            ),
            stationarity_scaling_error_infinity_norm=stationarity_scaling_error,
            solver_result_consistent=solver_result_consistent,
            solve_certificates_valid=retained_solve_certificates_valid,
            all_finite=all_finite,
            converged=converged,
        )

    def run(self) -> CfsFtr1Result:
        """Run the solver and finalize its endpoint for compatibility callers."""

        return self.finalize_result(self.run_solver())


def _optimizer_stationarity_tolerance(scaling: FullSpaceScaling) -> float:
    policy = ftr_route_policy(FullSpaceRoute.CFS_FTR1)
    minimum_variable_scale = float(np.min(np.abs(np.asarray(scaling.variable_scale))))
    if not np.isfinite(minimum_variable_scale) or minimum_variable_scale <= 0.0:
        raise ValueError("CFS-FTR1 variable scaling must be finite and positive")
    return policy.raw_kkt_stationarity_tolerance * minimum_variable_scale


def _filter_trust_region_options(
    optimizer_stationarity_tolerance: float,
    maximum_iterations: int,
) -> FilterTrustRegionSQPOptions:
    """Map the route policy into the generic optimizer's immutable controls."""

    policy = ftr_route_policy(FullSpaceRoute.CFS_FTR1)
    return FilterTrustRegionSQPOptions(
        maximum_iterations=maximum_iterations,
        maximum_joint_evaluations=policy.maximum_joint_evaluations,
        reverse_row_batch_width=policy.reverse_row_batch_width,
        objective_maximum=policy.objective_maximum,
        feasibility_tolerance=policy.scaled_feasibility_tolerance,
        stationarity_tolerance=optimizer_stationarity_tolerance,
        initial_bfgs_identity_scale=policy.initial_bfgs_identity_scale,
        curvature_fraction=policy.powell_curvature_fraction,
        maximum_consecutive_bfgs_resets=policy.maximum_consecutive_bfgs_resets,
        initial_trust_radius=policy.initial_trust_radius,
        minimum_trust_radius=policy.minimum_trust_radius,
        maximum_trust_radius=policy.maximum_trust_radius,
        normal_radius_fraction=policy.normal_radius_fraction,
        maximum_tangential_cg_iterations=(policy.maximum_tangential_cg_iterations),
        filter_gamma_feasibility=policy.filter_gamma_feasibility,
        filter_gamma_objective=policy.filter_gamma_objective,
        objective_step_threshold=policy.objective_step_threshold,
        acceptance_ratio=policy.acceptance_ratio,
        radius_shrink_ratio=policy.radius_shrink_ratio,
        expansion_ratio=policy.expansion_ratio,
        radius_contraction=policy.radius_contraction,
        radius_expansion=policy.radius_expansion,
        boundary_fraction=policy.boundary_fraction,
        tangency_relative_residual_tolerance=(
            policy.tangency_relative_residual_tolerance
        ),
        gram_relative_residual_tolerance=(
            policy.linear_solve_relative_residual_tolerance
        ),
        multiplier_projection_relative_residual_tolerance=(
            policy.linear_solve_relative_residual_tolerance
        ),
        linear_solve_forward_error_tolerance=(
            policy.linear_solve_forward_error_tolerance
        ),
    )


def prepare_cfs_ftr1(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    *,
    maximum_iterations: int = CFS_FTR1_FINAL_MAXIMUM_ITERATIONS,
) -> PreparedCfsFtr1:
    """Prepare the frozen callback-free CFS-FTR1 solve."""

    if (
        type(maximum_iterations) is not int
        or maximum_iterations not in CFS_FTR1_CANARY_MAXIMUM_ITERATIONS
    ):
        raise ValueError("CFS-FTR1 maximum_iterations must be one of {10, 100}")

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

    optimizer = prepare_filter_trust_region_sqp(
        joint_value_constraints,
        initial_optimizer_coordinates,
        options=_filter_trust_region_options(
            optimizer_stationarity_tolerance,
            maximum_iterations,
        ),
    )
    initial_scaled_multipliers = jnp.zeros(
        scaling.constraint_inverse_scale.shape,
        dtype=initial_optimizer_coordinates.dtype,
    )
    return PreparedCfsFtr1(
        problem=problem,
        scaling=scaling,
        initial_optimizer_coordinates=initial_optimizer_coordinates,
        initial_scaled_multipliers=initial_scaled_multipliers,
        optimizer_stationarity_tolerance=optimizer_stationarity_tolerance,
        optimizer=optimizer,
    )


__all__ = (
    "CFS_FTR1_CANARY_MAXIMUM_ITERATIONS",
    "CFS_FTR1_FINAL_MAXIMUM_ITERATIONS",
    "SCHEMA_VERSION",
    "CfsFtr1Result",
    "PreparedCfsFtr1",
    "prepare_cfs_ftr1",
)
