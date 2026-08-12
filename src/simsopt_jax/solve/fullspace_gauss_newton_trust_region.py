"""CFS-GNTR1 adapter over the frozen single-stage full-space problem."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import jax
import jax.numpy as jnp

from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonOptions,
    ProjectedGaussNewtonResult,
    run_projected_gauss_newton_trust_region,
)
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    factor_certified_gram_projector,
    solve_certified_gram,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceProblem,
    fullspace_objective_residual_vector,
)
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    certify_fullspace_objective_residuals,
)

from .fullspace import (
    FullSpaceScaling,
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
)
from .fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_linearization,
    cfs_sqp1_joint_value_constraints,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-gntr1-result-v1"
ROUTE: Final = "CFS-GNTR1"
CFS_GNTR1_STATE_SIZE: Final = 716
CFS_GNTR1_EQUALITY_SIZE: Final = 255
CFS_GNTR1_OBJECTIVE_RESIDUAL_SIZE: Final = 2110
CFS_GNTR1_OPTIONS: Final = ProjectedGaussNewtonOptions()

_PreparedRun = Callable[[jax.Array], ProjectedGaussNewtonResult]


@dataclass(frozen=True, slots=True)
class CfsGntr1Result:
    """Generic GNTR evidence paired with independent raw endpoint authority."""

    schema_version: str
    route: str
    optimizer_result: ProjectedGaussNewtonResult
    initial_endpoint: CfsSqp1EndpointDiagnostics
    final_endpoint: CfsSqp1EndpointDiagnostics
    residual_value_defect: jax.Array
    residual_gradient_relative_defect: jax.Array
    stationarity_scaling_relative_defect: jax.Array
    objective_residual_size: jax.Array
    state_size: jax.Array
    equality_size: jax.Array
    bootstrap_matches_initial: jax.Array
    dimensions_valid: jax.Array
    fp64_valid: jax.Array
    residual_contract_valid: jax.Array
    current_state_certificates_valid: jax.Array
    solver_result_consistent: jax.Array
    all_finite: jax.Array
    canary_usable: jax.Array

    @property
    def physical_state(self) -> jax.Array:
        return self.final_endpoint.physical_state

    @property
    def scaled_multipliers(self) -> jax.Array:
        return self.final_endpoint.scaled_multipliers

    @property
    def raw_multipliers(self) -> jax.Array:
        return self.final_endpoint.raw_multipliers

    @property
    def physical_objective(self) -> jax.Array:
        return self.final_endpoint.physical_objective

    @property
    def scaled_constraint_infinity_norm(self) -> jax.Array:
        return self.final_endpoint.scaled_constraint_infinity_norm

    @property
    def raw_kkt_stationarity_infinity_norm(self) -> jax.Array:
        return self.final_endpoint.raw_kkt_stationarity_infinity_norm


@dataclass(frozen=True, slots=True)
class PreparedCfsGntr1:
    """One compiled GNTR1 solve with untimed independent endpoint finalization."""

    problem: FullSpaceProblem
    scaling: FullSpaceScaling
    initial_physical_state: jax.Array
    initial_optimizer_coordinates: jax.Array
    objective_residual_size: int
    options: ProjectedGaussNewtonOptions
    _run_prepared: _PreparedRun = field(repr=False, compare=False)

    def run_solver(self) -> ProjectedGaussNewtonResult:
        """Execute only the compiled device-resident bounded optimizer."""

        return self._run_prepared(self.initial_optimizer_coordinates)

    def finalize_result(
        self,
        optimizer_result: ProjectedGaussNewtonResult,
    ) -> CfsGntr1Result:
        """Recompute raw initial/final endpoint evidence outside timed execution."""

        initial_linearization = cfs_sqp1_joint_linearization(
            self.initial_optimizer_coordinates,
            self.problem,
            self.scaling,
        )
        initial_projector = factor_certified_gram_projector(
            initial_linearization.constraint_jacobian
        )
        initial_multipliers = solve_certified_gram(
            initial_projector,
            -(
                initial_linearization.constraint_jacobian
                @ initial_linearization.objective_gradient
            ),
        )
        initial_endpoint = cfs_sqp1_endpoint_diagnostics(
            self.initial_optimizer_coordinates,
            initial_multipliers.solution,
            self.problem,
            self.scaling,
        )
        final_endpoint = cfs_sqp1_endpoint_diagnostics(
            optimizer_result.optimizer_coordinates,
            optimizer_result.multipliers,
            self.problem,
            self.scaling,
        )
        reconstruction = certify_fullspace_objective_residuals(
            self.initial_physical_state,
            self.problem,
        )

        expected_scaled_stationarity = (
            self.scaling.variable_scale * final_endpoint.raw_stationarity_residual
        )
        stationarity_difference = jnp.linalg.norm(
            optimizer_result.stationarity - expected_scaled_stationarity,
            ord=jnp.inf,
        )
        stationarity_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=optimizer_result.stationarity.dtype),
            jnp.maximum(
                jnp.linalg.norm(optimizer_result.stationarity, ord=jnp.inf),
                jnp.linalg.norm(expected_scaled_stationarity, ord=jnp.inf),
            ),
        )
        stationarity_scaling_relative_defect = (
            stationarity_difference / stationarity_scale
        )
        solver_result_consistent = (
            jnp.isclose(
                optimizer_result.objective,
                final_endpoint.physical_objective,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            & jnp.allclose(
                optimizer_result.constraints,
                final_endpoint.scaled_constraints,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            & (stationarity_scaling_relative_defect <= 1.0e-10)
        )

        state_size = jnp.asarray(
            optimizer_result.optimizer_coordinates.size,
            dtype=jnp.int32,
        )
        equality_size = jnp.asarray(optimizer_result.constraints.size, dtype=jnp.int32)
        objective_residual_size = jnp.asarray(
            self.objective_residual_size,
            dtype=jnp.int32,
        )
        dimensions_valid = (
            (state_size == CFS_GNTR1_STATE_SIZE)
            & (equality_size == CFS_GNTR1_EQUALITY_SIZE)
            & (objective_residual_size == CFS_GNTR1_OBJECTIVE_RESIDUAL_SIZE)
        )
        fp64_valid = (optimizer_result.optimizer_coordinates.dtype == jnp.float64) & (
            optimizer_result.constraints.dtype == jnp.float64
        )
        bootstrap_matches_initial = jnp.array_equal(
            self.initial_physical_state,
            self.scaling.bootstrap_anchor,
        )
        residual_contract_valid = (
            reconstruction.residual_valid
            & reconstruction.all_finite
            & (
                reconstruction.value_scaled_defect
                <= self.options.residual_value_defect_tolerance
            )
            & (
                reconstruction.gradient_scaled_defect
                <= self.options.residual_gradient_defect_tolerance
            )
        )
        attempt_mask = (
            jnp.arange(self.options.maximum_attempts, dtype=jnp.int32)
            < optimizer_result.attempts
        )
        history = optimizer_result.history
        current_state_certificates_valid = jnp.all(
            ~attempt_mask
            | (
                jnp.isfinite(history.residual_value_defect)
                & jnp.isfinite(history.residual_gradient_defect)
                & jnp.isfinite(history.hvp_symmetry_defect)
                & jnp.isfinite(history.probe_normalized_curvature)
                & jnp.isfinite(history.current_projection_tangency_relative_residual)
                & jnp.isfinite(history.current_projection_solve_relative_residual)
                & jnp.isfinite(history.current_projection_forward_error_bound)
                & jnp.isfinite(history.trial_gram_factorization_relative_residual)
                & jnp.isfinite(history.trial_gram_solve_relative_residual)
                & (
                    history.residual_value_defect
                    <= self.options.residual_value_defect_tolerance
                )
                & (
                    history.residual_gradient_defect
                    <= self.options.residual_gradient_defect_tolerance
                )
                & (
                    history.hvp_symmetry_defect
                    <= self.options.residual_gradient_defect_tolerance
                )
                & (
                    history.probe_normalized_curvature
                    >= -self.options.normalized_curvature_tolerance
                )
                & (
                    history.current_projection_tangency_relative_residual
                    <= self.options.linear_residual_tolerance
                )
                & (
                    history.current_projection_solve_relative_residual
                    <= self.options.linear_residual_tolerance
                )
                & (
                    history.current_projection_forward_error_bound
                    < self.options.forward_error_tolerance
                )
                & (
                    history.trial_gram_factorization_relative_residual
                    <= self.options.linear_residual_tolerance
                )
                & (
                    history.trial_gram_solve_relative_residual
                    <= self.options.linear_residual_tolerance
                )
            )
        )
        all_finite = (
            optimizer_result.all_finite
            & optimizer_result.all_accepted_states_finite
            & optimizer_result.final_certificate.all_finite
            & initial_linearization.all_finite
            & initial_projector.all_finite
            & initial_multipliers.all_finite
            & initial_endpoint.all_finite
            & final_endpoint.all_finite
            & reconstruction.all_finite
            & jnp.isfinite(stationarity_scaling_relative_defect)
        )
        canary_usable = (
            optimizer_result.bounded_complete
            & ~optimizer_result.fatal
            & optimizer_result.usable
            & optimizer_result.final_certificate.certified
            & dimensions_valid
            & fp64_valid
            & bootstrap_matches_initial
            & residual_contract_valid
            & current_state_certificates_valid
            & solver_result_consistent
            & all_finite
        )
        return CfsGntr1Result(
            schema_version=SCHEMA_VERSION,
            route=ROUTE,
            optimizer_result=optimizer_result,
            initial_endpoint=initial_endpoint,
            final_endpoint=final_endpoint,
            residual_value_defect=reconstruction.value_scaled_defect,
            residual_gradient_relative_defect=(reconstruction.gradient_scaled_defect),
            stationarity_scaling_relative_defect=(stationarity_scaling_relative_defect),
            objective_residual_size=objective_residual_size,
            state_size=state_size,
            equality_size=equality_size,
            bootstrap_matches_initial=bootstrap_matches_initial,
            dimensions_valid=dimensions_valid,
            fp64_valid=fp64_valid,
            residual_contract_valid=residual_contract_valid,
            current_state_certificates_valid=current_state_certificates_valid,
            solver_result_consistent=solver_result_consistent,
            all_finite=all_finite,
            canary_usable=canary_usable,
        )

    def run(self) -> CfsGntr1Result:
        """Run the compiled solver and then form independent endpoint evidence."""

        return self.finalize_result(self.run_solver())


def prepare_cfs_gntr1(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
) -> PreparedCfsGntr1:
    """Compile the frozen callback-free eight-step CFS-GNTR1 canary."""

    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    initial_optimizer_coordinates = fullspace_optimizer_coordinates(
        initial_physical_state,
        scaling,
    )

    def joint_value_constraints(
        optimizer_coordinates: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return cfs_sqp1_joint_value_constraints(
            optimizer_coordinates,
            problem,
            scaling,
        )

    def objective_residuals(optimizer_coordinates: jax.Array) -> jax.Array:
        physical_state = fullspace_physical_coordinates(
            optimizer_coordinates,
            scaling,
        )
        return fullspace_objective_residual_vector(physical_state, problem)

    residual_shape = jax.eval_shape(
        objective_residuals,
        initial_optimizer_coordinates,
    )
    if residual_shape.ndim != 1:
        raise ValueError("CFS-GNTR1 objective residual must be a vector")
    if residual_shape.dtype != initial_optimizer_coordinates.dtype:
        raise TypeError("CFS-GNTR1 objective residual dtype must match coordinates")

    def run_prepared(coordinates: jax.Array) -> ProjectedGaussNewtonResult:
        return run_projected_gauss_newton_trust_region(
            joint_value_constraints,
            objective_residuals,
            coordinates,
            options=CFS_GNTR1_OPTIONS,
        )

    executable = jax.jit(run_prepared).lower(initial_optimizer_coordinates).compile()
    return PreparedCfsGntr1(
        problem=problem,
        scaling=scaling,
        initial_physical_state=initial_physical_state,
        initial_optimizer_coordinates=initial_optimizer_coordinates,
        objective_residual_size=residual_shape.size,
        options=CFS_GNTR1_OPTIONS,
        _run_prepared=executable,
    )


__all__ = (
    "CFS_GNTR1_EQUALITY_SIZE",
    "CFS_GNTR1_OBJECTIVE_RESIDUAL_SIZE",
    "CFS_GNTR1_OPTIONS",
    "CFS_GNTR1_STATE_SIZE",
    "ROUTE",
    "SCHEMA_VERSION",
    "CfsGntr1Result",
    "PreparedCfsGntr1",
    "prepare_cfs_gntr1",
)
