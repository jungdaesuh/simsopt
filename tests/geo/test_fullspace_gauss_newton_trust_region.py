from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.solve.fullspace_gauss_newton_trust_region as gntr_module
from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonAttemptOutcome,
    ProjectedGaussNewtonFinalCertificate,
    ProjectedGaussNewtonHistory,
    ProjectedGaussNewtonOptions,
    ProjectedGaussNewtonResult,
    ProjectedGaussNewtonStatus,
)
from simsopt_jax.objectives.single_stage_fullspace import FullSpaceProblem
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    ObjectiveResidualReconstruction,
)
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_gauss_newton_trust_region import (
    CFS_GNTR1_OPTIONS,
    prepare_cfs_gntr1,
)
from simsopt_jax.solve.fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    CfsSqp1JointLinearization,
)

jax.config.update("jax_enable_x64", True)

_STATE_SIZE = 716
_EQUALITY_SIZE = 255
_RESIDUAL_SIZE = 2110
_CONSTRAINT_MATRIX = jnp.pad(
    jnp.diag(jnp.linspace(1.0, 2.0, _EQUALITY_SIZE, dtype=jnp.float64)),
    ((0, 0), (0, _STATE_SIZE - _EQUALITY_SIZE)),
)


def _scaling() -> FullSpaceScaling:
    return FullSpaceScaling(
        bootstrap_anchor=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        variable_scale=jnp.linspace(0.5, 1.5, _STATE_SIZE, dtype=jnp.float64),
        constraint_inverse_scale=jnp.linspace(
            0.75,
            1.25,
            _EQUALITY_SIZE,
            dtype=jnp.float64,
        ),
    )


def _problem() -> FullSpaceProblem:
    return cast(
        "FullSpaceProblem",
        SimpleNamespace(layout=SimpleNamespace(total_dof_count=_STATE_SIZE)),
    )


def _joint(
    coordinates: jax.Array,
    _problem_value: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> tuple[jax.Array, jax.Array]:
    physical_state = scaling.bootstrap_anchor + scaling.variable_scale * coordinates
    objective = 0.5 * jnp.vdot(physical_state, physical_state)
    constraints = scaling.constraint_inverse_scale * (
        _CONSTRAINT_MATRIX @ physical_state
    )
    return objective, constraints


def _residual_vector(
    physical_state: jax.Array,
    _problem_value: FullSpaceProblem,
) -> jax.Array:
    return jnp.pad(physical_state, (0, _RESIDUAL_SIZE - _STATE_SIZE))


def _joint_linearization(
    coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> CfsSqp1JointLinearization:
    objective, constraints = _joint(coordinates, problem, scaling)
    physical_state = scaling.bootstrap_anchor + scaling.variable_scale * coordinates
    objective_gradient = scaling.variable_scale * physical_state
    constraint_jacobian = (
        scaling.constraint_inverse_scale[:, None]
        * _CONSTRAINT_MATRIX
        * scaling.variable_scale[None, :]
    )
    joint_vjp_rows = jnp.concatenate(
        (objective_gradient[None, :], constraint_jacobian),
        axis=0,
    )
    return CfsSqp1JointLinearization(
        physical_objective=objective,
        scaled_constraints=constraints,
        objective_gradient=objective_gradient,
        constraint_jacobian=constraint_jacobian,
        joint_vjp_rows=joint_vjp_rows,
        all_finite=jnp.asarray(True),
    )


def _endpoint(
    coordinates: jax.Array,
    scaled_multipliers: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> CfsSqp1EndpointDiagnostics:
    objective, scaled_constraints = _joint(coordinates, problem, scaling)
    physical_state = scaling.bootstrap_anchor + scaling.variable_scale * coordinates
    raw_constraints = _CONSTRAINT_MATRIX @ physical_state
    raw_multipliers = scaling.constraint_inverse_scale * scaled_multipliers
    raw_stationarity = physical_state + _CONSTRAINT_MATRIX.T @ raw_multipliers
    return CfsSqp1EndpointDiagnostics(
        physical_state=physical_state,
        physical_objective=objective,
        raw_constraints=raw_constraints,
        scaled_constraints=scaled_constraints,
        scaled_multipliers=scaled_multipliers,
        raw_multipliers=raw_multipliers,
        raw_stationarity_residual=raw_stationarity,
        raw_constraint_infinity_norm=jnp.linalg.norm(raw_constraints, ord=jnp.inf),
        scaled_constraint_infinity_norm=jnp.linalg.norm(
            scaled_constraints,
            ord=jnp.inf,
        ),
        raw_kkt_stationarity_infinity_norm=jnp.linalg.norm(
            raw_stationarity,
            ord=jnp.inf,
        ),
        all_finite=jnp.asarray(True),
    )


def _history(options: ProjectedGaussNewtonOptions) -> ProjectedGaussNewtonHistory:
    index = jnp.arange(options.maximum_attempts, dtype=jnp.int32)
    active = index < options.maximum_accepted_steps
    floating = jnp.where(active, 0.0, jnp.nan).astype(jnp.float64)
    return ProjectedGaussNewtonHistory(
        outcome=jnp.where(
            active,
            int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED),
            int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
        ).astype(jnp.int32),
        accepted_step_number=jnp.where(active, index + 1, 0),
        current_objective=floating,
        current_feasibility_inf=floating,
        current_stationarity_inf=floating,
        candidate_objective=floating,
        candidate_feasibility_inf=floating,
        actual_reduction=floating,
        predicted_reduction=floating,
        reduction_ratio=floating,
        trust_radius=floating,
        next_trust_radius=floating,
        tangent_step_norm=floating,
        correction_norm=floating,
        applied_step_norm=floating,
        correction_step_ratio=floating,
        corrected_radius_ratio=floating,
        nonlinear_corrections=jnp.where(active, 1, 0).astype(jnp.int32),
        steihaug_iterations=jnp.where(active, 1, 0).astype(jnp.int32),
        steihaug_hvp_evaluations=jnp.where(active, 1, 0).astype(jnp.int32),
        steihaug_termination=jnp.where(active, 1, 0).astype(jnp.int32),
        terminal_normalized_curvature=floating,
        residual_value_defect=floating,
        residual_gradient_defect=floating,
        hvp_symmetry_defect=floating,
        probe_normalized_curvature=floating,
        direction_rotation=floating,
        correction_relative_residual=floating,
        correction_forward_error_bound=floating,
        current_projection_tangency_relative_residual=floating,
        current_projection_solve_relative_residual=floating,
        current_projection_forward_error_bound=floating,
        steihaug_tangency_relative_residual=floating,
        steihaug_final_projected_residual_norm=floating,
        steihaug_projected_residual_target=floating,
        steihaug_hit_boundary=jnp.zeros(
            (options.maximum_attempts,),
            dtype=jnp.bool_,
        ),
        steihaug_residual_projection_tangency_relative_residual=floating,
        steihaug_residual_projection_solve_relative_residual=floating,
        steihaug_residual_projection_forward_error_bound=floating,
        trial_gram_factorization_relative_residual=floating,
        trial_gram_solve_relative_residual=floating,
    )


def _fake_optimizer(
    joint_value_constraints: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    objective_residuals: Callable[[jax.Array], jax.Array],
    coordinates: jax.Array,
    *,
    options: ProjectedGaussNewtonOptions,
) -> ProjectedGaussNewtonResult:
    objective, constraints = joint_value_constraints(coordinates)
    residual = objective_residuals(coordinates)
    zero_coordinates = jnp.zeros_like(coordinates)
    zero_multipliers = jnp.zeros_like(constraints)
    objective_gradient = jnp.zeros_like(coordinates)
    constraint_jacobian = jnp.zeros(
        (constraints.size, coordinates.size),
        dtype=coordinates.dtype,
    )
    residual_is_expected_shape = residual.size == _RESIDUAL_SIZE
    final_certificate = ProjectedGaussNewtonFinalCertificate(
        coordinates_finite=jnp.asarray(True),
        residual_value_defect=jnp.asarray(0.0, dtype=coordinates.dtype),
        residual_gradient_defect=jnp.asarray(0.0, dtype=coordinates.dtype),
        hvp_symmetry_defect=jnp.asarray(0.0, dtype=coordinates.dtype),
        probe_normalized_curvature=jnp.asarray(0.0, dtype=coordinates.dtype),
        gram_factorization_relative_residual=jnp.asarray(
            0.0,
            dtype=coordinates.dtype,
        ),
        multiplier_relative_residual=jnp.asarray(0.0, dtype=coordinates.dtype),
        multiplier_forward_error_bound=jnp.asarray(0.0, dtype=coordinates.dtype),
        projection_tangency_relative_residual=jnp.asarray(
            0.0,
            dtype=coordinates.dtype,
        ),
        projection_solve_relative_residual=jnp.asarray(
            0.0,
            dtype=coordinates.dtype,
        ),
        projection_forward_error_bound=jnp.asarray(0.0, dtype=coordinates.dtype),
        all_finite=jnp.asarray(True),
        certified=jnp.asarray(True),
    )
    return ProjectedGaussNewtonResult(
        optimizer_coordinates=zero_coordinates,
        multipliers=zero_multipliers,
        objective=objective,
        constraints=constraints,
        objective_gradient=objective_gradient,
        constraint_jacobian=constraint_jacobian,
        stationarity=objective_gradient,
        scaled_stationarity_inf=jnp.asarray(0.0, dtype=coordinates.dtype),
        scaled_feasibility_inf=jnp.asarray(0.0, dtype=coordinates.dtype),
        trust_radius=jnp.asarray(options.initial_trust_radius, dtype=coordinates.dtype),
        accepted_steps=jnp.asarray(options.maximum_accepted_steps, dtype=jnp.int32),
        attempts=jnp.asarray(options.maximum_accepted_steps, dtype=jnp.int32),
        retryable_rejections=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(
            int(ProjectedGaussNewtonStatus.BOUNDED_COMPLETE),
            dtype=jnp.int32,
        ),
        fatal=jnp.asarray(False),
        bounded_complete=jnp.asarray(True),
        mechanism_exercised=jnp.asarray(True),
        all_accepted_states_finite=jnp.asarray(residual_is_expected_shape),
        all_finite=jnp.asarray(residual_is_expected_shape),
        final_certificate=final_certificate,
        usable=jnp.asarray(residual_is_expected_shape),
        history=_history(options),
    )


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch):
    scaling = _scaling()
    monkeypatch.setattr(
        gntr_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem_value: scaling,
    )
    monkeypatch.setattr(gntr_module, "cfs_sqp1_joint_value_constraints", _joint)
    monkeypatch.setattr(
        gntr_module,
        "fullspace_objective_residual_vector",
        _residual_vector,
    )
    monkeypatch.setattr(
        gntr_module,
        "cfs_sqp1_joint_linearization",
        _joint_linearization,
    )
    monkeypatch.setattr(gntr_module, "cfs_sqp1_endpoint_diagnostics", _endpoint)
    monkeypatch.setattr(
        gntr_module,
        "certify_fullspace_objective_residuals",
        lambda _state, _problem_value: ObjectiveResidualReconstruction(
            reconstructed_value=jnp.asarray(0.0, dtype=jnp.float64),
            authoritative_value=jnp.asarray(0.0, dtype=jnp.float64),
            value_scaled_defect=jnp.asarray(0.0, dtype=jnp.float64),
            gradient_scaled_defect=jnp.asarray(0.0, dtype=jnp.float64),
            residual_valid=jnp.asarray(True),
            all_finite=jnp.asarray(True),
        ),
    )
    monkeypatch.setattr(
        gntr_module,
        "run_projected_gauss_newton_trust_region",
        _fake_optimizer,
    )
    return prepare_cfs_gntr1(
        _problem(),
        scaling.bootstrap_anchor,
        scaling.bootstrap_anchor,
    )


def test_frozen_gntr1_policy_matches_ssot() -> None:
    assert CFS_GNTR1_OPTIONS.maximum_accepted_steps == 8
    assert CFS_GNTR1_OPTIONS.maximum_attempts == 12
    assert CFS_GNTR1_OPTIONS.initial_trust_radius == 2.0**-10
    assert CFS_GNTR1_OPTIONS.minimum_trust_radius == 2.0**-20
    assert CFS_GNTR1_OPTIONS.maximum_trust_radius == 2.0**-4
    assert CFS_GNTR1_OPTIONS.maximum_steihaug_iterations == 32


def test_prepared_solver_is_compiled_and_finalization_is_independent(prepared) -> None:
    optimizer_result = prepared.run_solver()
    result = prepared.finalize_result(optimizer_result)

    assert optimizer_result.history.outcome.shape == (12,)
    assert optimizer_result.accepted_steps == 8
    assert optimizer_result.attempts == 8
    assert result.schema_version == "single-stage-fullspace-cfs-gntr1-result-v1"
    assert result.route == "CFS-GNTR1"
    assert result.objective_residual_size == _RESIDUAL_SIZE
    assert result.state_size == _STATE_SIZE
    assert result.equality_size == _EQUALITY_SIZE
    assert result.bootstrap_matches_initial
    assert result.dimensions_valid
    assert result.fp64_valid
    assert result.residual_contract_valid
    assert result.current_state_certificates_valid
    assert optimizer_result.final_certificate.certified
    assert optimizer_result.usable
    assert result.solver_result_consistent
    assert result.all_finite
    assert result.canary_usable
    np.testing.assert_array_equal(
        result.initial_endpoint.physical_state,
        jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
    )
    np.testing.assert_array_equal(
        result.final_endpoint.physical_state,
        jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
    )


def test_independent_endpoint_rejects_inconsistent_solver_result(prepared) -> None:
    optimizer_result = prepared.run_solver()
    inconsistent = optimizer_result._replace(
        objective=jnp.asarray(1.0, dtype=jnp.float64)
    )

    result = prepared.finalize_result(inconsistent)

    assert not result.solver_result_consistent
    assert not result.canary_usable


def test_adapter_consumes_core_final_certificate_fail_closed(prepared) -> None:
    optimizer_result = prepared.run_solver()

    unusable = prepared.finalize_result(
        optimizer_result._replace(usable=jnp.asarray(False))
    )
    uncertified = prepared.finalize_result(
        optimizer_result._replace(
            final_certificate=optimizer_result.final_certificate._replace(
                certified=jnp.asarray(False)
            )
        )
    )

    assert not unusable.canary_usable
    assert not uncertified.canary_usable
