from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonAcceptedState,
    ProjectedGaussNewtonAttemptOutcome,
    ProjectedGaussNewtonLoopResult,
    ProjectedGaussNewtonOptions,
    ProjectedGaussNewtonStatus,
    _correction_path_bounds_valid,
    _trial_correction_certified,
    finalize_projected_gauss_newton_trust_region,
    run_projected_gauss_newton_trust_region,
    run_projected_gauss_newton_trust_region_loop,
)
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    ProjectedSteihaugTermination,
)
from simsopt_jax.runtime import trace_annotations
from simsopt_jax.runtime.trace_annotations import (
    GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256,
    GNTR_DIAGNOSTIC_PHASES,
    normalized_jax_ir,
    trace_session,
)

jax.config.update("jax_enable_x64", True)


def _linear_equality_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    return 0.5 * values[0] ** 2, jnp.reshape(values[1], (1,))


def _linear_objective_residual(values: jax.Array) -> jax.Array:
    return jnp.reshape(values[0], (1,))


def _nonlinear_retraction_joint(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return 0.5 * values[0] ** 2, jnp.reshape(values[1] - values[0] ** 2, (1,))


def _curved_residual_problem(
    quadratic_coefficient: float,
):
    def residual(values: jax.Array) -> jax.Array:
        return jnp.reshape(
            1.0 + values[0] + quadratic_coefficient * values[0] ** 2,
            (1,),
        )

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        residual_value = residual(values)
        return 0.5 * jnp.vdot(residual_value, residual_value), jnp.reshape(
            values[1], (1,)
        )

    return joint, residual


def test_boundary_step_is_accepted_and_expands_radius() -> None:
    radius = 0.25
    result = run_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=radius,
            maximum_trust_radius=1.0,
        ),
    )

    np.testing.assert_allclose(result.optimizer_coordinates, [0.75, 0.0], atol=1e-14)
    assert result.status == int(ProjectedGaussNewtonStatus.BOUNDED_COMPLETE)
    assert result.accepted_steps == 1
    assert result.attempts == 1
    assert result.history.outcome[0] == int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED)
    assert result.history.reduction_ratio[0] == 1.0
    assert result.history.next_trust_radius[0] == 0.5
    assert result.trust_radius == 0.5


def test_nonfinite_candidate_retries_without_changing_accepted_state() -> None:
    def nonfinite_candidate_joint(
        values: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        objective = jnp.where(values[0] < 0.9, jnp.nan, 0.5 * values[0] ** 2)
        return objective, jnp.reshape(values[1], (1,))

    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        nonfinite_candidate_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=1.0,
        ),
    )

    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.status == int(ProjectedGaussNewtonStatus.ATTEMPT_LIMIT)
    assert not result.fatal
    assert result.retryable_rejections == 1
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_NONFINITE
    )
    assert result.trust_radius == 0.0625


def test_residual_reconstruction_failure_is_fatal() -> None:
    result = run_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        lambda values: jnp.reshape(2.0 * values[0], (1,)),
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
        ),
    )

    assert result.fatal
    assert result.status == int(ProjectedGaussNewtonStatus.FATAL_CURRENT_STATE)
    assert result.accepted_steps == 0
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.FATAL_CURRENT_STATE
    )
    assert result.history.residual_value_defect[0] > 0.0


def test_steihaug_exhaustion_is_fatal_not_retryable() -> None:
    curvature_scale = jnp.asarray([1.0, 10.0, 0.0], dtype=jnp.float64)

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        residual = curvature_scale[:2] * values[:2]
        return 0.5 * jnp.vdot(residual, residual), jnp.reshape(values[2], (1,))

    def residual(values: jax.Array) -> jax.Array:
        return curvature_scale[:2] * values[:2]

    result = run_projected_gauss_newton_trust_region(
        joint,
        residual,
        jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=10.0,
            maximum_trust_radius=10.0,
            maximum_steihaug_iterations=1,
        ),
    )

    assert result.fatal
    assert result.status == int(ProjectedGaussNewtonStatus.FATAL_STEIHAUG)
    assert result.retryable_rejections == 0
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.FATAL_STEIHAUG
    )


def test_corrected_prediction_uses_full_applied_step() -> None:
    def nonlinear_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.reshape(values[1] - values[0] ** 2, (1,))

    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        nonlinear_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=1.0e-5,
            minimum_trust_radius=1.0e-8,
            maximum_trust_radius=1.0e-3,
        ),
    )

    applied_step = result.optimizer_coordinates - initial
    expected_prediction = -(applied_step[0] + 0.5 * applied_step[0] ** 2)
    assert result.accepted_steps == 1
    assert result.history.correction_norm[0] > 0.0
    np.testing.assert_allclose(
        result.history.predicted_reduction[0], expected_prediction, rtol=0.0, atol=1e-15
    )
    assert result.history.correction_step_ratio[0] <= 1.0e-3
    assert result.scaled_feasibility_inf <= 1.0e-10


def test_retryable_step_bound_failure_forces_quarter_radius() -> None:
    def nonlinear_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.reshape(values[1] - values[0] ** 2, (1,))

    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        nonlinear_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=1.0e-5,
            minimum_trust_radius=1.0e-8,
            maximum_trust_radius=1.0e-3,
            maximum_correction_step_ratio=1.0e-12,
        ),
    )

    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS
    )
    assert result.history.next_trust_radius[0] == 2.5e-6
    assert result.history.correction_step_ratio[0] > 1.0e-12


def test_step_bound_safeguard_accepts_quartered_subtrial_and_records_work() -> None:
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    common_options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=1,
        maximum_attempts=1,
        initial_trust_radius=1.0e-5,
        minimum_trust_radius=1.0e-8,
        maximum_trust_radius=1.0e-3,
        maximum_correction_step_ratio=5.0e-7,
    )
    disabled = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=common_options,
    )
    enabled = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=replace(common_options, enable_step_bound_safeguard=True),
    )

    assert disabled.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS
    )
    assert disabled.accepted_steps == 0
    assert enabled.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.ACCEPTED
    )
    assert enabled.accepted_steps == 1
    assert enabled.attempts == 1
    assert enabled.retryable_rejections == 0
    assert enabled.history.subtrial_count[0] == 2
    assert enabled.history.selected_subtrial_index[0] == 1
    np.testing.assert_array_equal(
        enabled.history.subtrial_outcome[0],
        [
            int(ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS),
            int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED),
            int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
        ],
    )
    np.testing.assert_allclose(
        enabled.history.subtrial_trust_radius[0, :2],
        [1.0e-5, 2.5e-6],
        rtol=0.0,
        atol=0.0,
    )
    assert np.isnan(enabled.history.subtrial_trust_radius[0, 2])
    assert (
        enabled.history.subtrial_maximum_individual_correction_step_ratio[0, 0]
        > common_options.maximum_correction_step_ratio
    )
    assert (
        enabled.history.subtrial_maximum_individual_correction_step_ratio[0, 1]
        <= common_options.maximum_correction_step_ratio
    )
    assert enabled.history.trust_radius[0] == 2.5e-6
    assert (
        enabled.history.maximum_individual_correction_step_ratio[0]
        == enabled.history.subtrial_maximum_individual_correction_step_ratio[0, 1]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_joint_evaluations[0], [3, 3, 0]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_nonlinear_corrections[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_joint_linearizations[0], [2, 2, 0]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_joint_value_evaluations[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_objective_residual_linearizations[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_gram_factorizations[0], [2, 2, 0]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_steihaug_solve_calls[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_total_hvp_evaluations[0, :2],
        enabled.history.subtrial_steihaug_hvp_evaluations[0, :2] + 4,
    )
    np.testing.assert_array_equal(
        enabled.history.subtrial_gram_solves[0, :2],
        enabled.history.subtrial_steihaug_hvp_evaluations[0, :2] + 6,
    )


def test_whole_solver_jits_with_enabled_step_bound_safeguard() -> None:
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=1,
        maximum_attempts=1,
        initial_trust_radius=1.0e-5,
        minimum_trust_radius=1.0e-8,
        maximum_trust_radius=1.0e-3,
        maximum_correction_step_ratio=5.0e-7,
        enable_step_bound_safeguard=True,
    )

    def run(initial: jax.Array):
        return run_projected_gauss_newton_trust_region(
            _nonlinear_retraction_joint,
            _linear_objective_residual,
            initial,
            options=options,
        )

    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    executable = jax.jit(run).lower(initial).compile()
    first = executable(initial)
    second = executable(initial)

    for first_leaf, second_leaf in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        np.testing.assert_array_equal(second_leaf, first_leaf)
    for field in first.history:
        assert field.shape[0] == options.maximum_attempts
        assert field.ndim in {1, 2}
        if field.ndim == 2:
            assert field.shape == (options.maximum_attempts, 3)

    assert first.status == int(ProjectedGaussNewtonStatus.BOUNDED_COMPLETE)
    assert first.accepted_steps == 1
    assert first.attempts == 1
    assert first.retryable_rejections == 0
    assert first.history.subtrial_count[0] == 2
    assert first.history.selected_subtrial_index[0] == 1
    np.testing.assert_array_equal(
        first.history.subtrial_outcome[0],
        [
            int(ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS),
            int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED),
            int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
        ],
    )
    np.testing.assert_array_equal(
        first.history.subtrial_trust_radius[0], [1.0e-5, 2.5e-6, np.nan]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_steihaug_solve_calls[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_steihaug_hvp_evaluations[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_total_hvp_evaluations[0], [5, 5, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_nonlinear_corrections[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_joint_evaluations[0], [3, 3, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_joint_linearizations[0], [2, 2, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_joint_value_evaluations[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_objective_residual_linearizations[0], [1, 1, 0]
    )
    np.testing.assert_array_equal(
        first.history.subtrial_gram_factorizations[0], [2, 2, 0]
    )
    np.testing.assert_array_equal(first.history.subtrial_gram_solves[0], [7, 7, 0])


def test_step_bound_safeguard_uses_at_most_two_quartered_subtrials() -> None:
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=1.0e-5,
            minimum_trust_radius=1.0e-8,
            maximum_trust_radius=1.0e-3,
            maximum_correction_step_ratio=2.0e-7,
            enable_step_bound_safeguard=True,
        ),
    )

    assert result.accepted_steps == 1
    assert result.history.subtrial_count[0] == 3
    assert result.history.selected_subtrial_index[0] == 2
    np.testing.assert_array_equal(
        result.history.subtrial_outcome[0],
        [
            int(ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS),
            int(ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS),
            int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED),
        ],
    )
    np.testing.assert_allclose(
        result.history.subtrial_trust_radius[0],
        [1.0e-5, 2.5e-6, 6.25e-7],
        rtol=0.0,
        atol=0.0,
    )


def test_step_bound_safeguard_exhaustion_keeps_every_bound_failure() -> None:
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    correction_limit = 1.0e-12
    result = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=1.0e-5,
            minimum_trust_radius=1.0e-8,
            maximum_trust_radius=1.0e-3,
            maximum_correction_step_ratio=correction_limit,
            enable_step_bound_safeguard=True,
        ),
    )

    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.accepted_steps == 0
    assert result.retryable_rejections == 1
    assert result.history.subtrial_count[0] == 3
    assert result.history.selected_subtrial_index[0] == 2
    np.testing.assert_array_equal(
        result.history.subtrial_outcome[0],
        int(ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS),
    )
    assert np.all(
        result.history.subtrial_maximum_individual_correction_step_ratio[0]
        > correction_limit
    )
    assert result.history.next_trust_radius[0] == 1.5625e-7


def test_step_bound_safeguard_stops_on_nonfinite_backtrack_and_initial_fatal() -> None:
    def nonfinite_quartered_candidate_joint(
        values: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        nonfinite = (values[1] < 1.0) & (values[1] > 0.999995)
        objective = jnp.where(nonfinite, jnp.nan, 0.5 * values[0] ** 2)
        return objective, jnp.reshape(values[1] - values[0] ** 2, (1,))

    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    nonfinite = run_projected_gauss_newton_trust_region(
        nonfinite_quartered_candidate_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=1.0e-5,
            minimum_trust_radius=1.0e-8,
            maximum_trust_radius=1.0e-3,
            maximum_correction_step_ratio=8.0e-7,
            enable_step_bound_safeguard=True,
        ),
    )
    fatal = run_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        lambda values: jnp.reshape(2.0 * values[0], (1,)),
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            enable_step_bound_safeguard=True,
        ),
    )

    assert nonfinite.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_NONFINITE
    )
    assert nonfinite.history.subtrial_count[0] == 2
    np.testing.assert_array_equal(
        nonfinite.history.subtrial_outcome[0],
        [
            int(ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS),
            int(ProjectedGaussNewtonAttemptOutcome.RETRY_NONFINITE),
            int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
        ],
    )
    assert fatal.fatal
    assert fatal.history.subtrial_count[0] == 1
    assert fatal.history.subtrial_outcome[0, 0] == int(
        ProjectedGaussNewtonAttemptOutcome.FATAL_CURRENT_STATE
    )
    assert fatal.history.subtrial_steihaug_solve_calls[0, 0] == 0
    assert fatal.history.subtrial_total_hvp_evaluations[0, 0] == 3
    assert fatal.history.subtrial_joint_linearizations[0, 0] == 1
    assert fatal.history.subtrial_objective_residual_linearizations[0, 0] == 1
    assert fatal.history.subtrial_gram_factorizations[0, 0] == 1
    assert fatal.history.subtrial_gram_solves[0, 0] == 2


def test_step_bound_safeguard_requires_positive_actual_reduction() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return jnp.reshape(1.0 + values[0] + 10.0 * values[0] ** 2, (1,))

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        residual_value = residual(values)
        return 0.5 * jnp.vdot(residual_value, residual_value), jnp.reshape(
            values[1] - values[0] ** 2, (1,)
        )

    result = run_projected_gauss_newton_trust_region(
        joint,
        residual,
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=1.0,
            maximum_correction_step_ratio=1.0e-12,
            maximum_nonlinear_corrections=3,
            enable_step_bound_safeguard=True,
        ),
    )

    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_STEP_BOUNDS
    )
    assert result.history.actual_reduction[0] < 0.0
    assert result.history.subtrial_count[0] == 1


def test_whole_loop_jits_with_fixed_history_and_exercises_rotation() -> None:
    curvature_scale = jnp.asarray([1.0, 10.0, 0.0], dtype=jnp.float64)
    options = replace(
        ProjectedGaussNewtonOptions(),
        maximum_accepted_steps=2,
        maximum_attempts=3,
        initial_trust_radius=0.8,
        maximum_trust_radius=10.0,
    )

    def run(initial: jax.Array):
        def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            residual = curvature_scale[:2] * values[:2]
            return 0.5 * jnp.vdot(residual, residual), jnp.reshape(values[2], (1,))

        def residual(values: jax.Array) -> jax.Array:
            return curvature_scale[:2] * values[:2]

        return run_projected_gauss_newton_trust_region(
            joint, residual, initial, options=options
        )

    initial = jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float64)
    executable = jax.jit(run).lower(initial).compile()
    result = executable(initial)

    assert result.status == int(ProjectedGaussNewtonStatus.BOUNDED_COMPLETE)
    assert result.accepted_steps == 2
    assert result.attempts == 2
    assert result.mechanism_exercised
    assert result.history.steihaug_hvp_evaluations[1] >= 2
    assert result.history.direction_rotation[1] >= 1.0e-3
    for field in result.history:
        assert field.shape[0] == options.maximum_attempts
        assert field.ndim in {1, 2}
        if field.ndim == 2:
            assert field.shape[1] == 3
    compiled_text = executable.runtime_executable().hlo_modules()[0].to_string()
    assert "host_callback" not in compiled_text
    assert "io_callback" not in compiled_text


def test_authoritative_gradient_drives_stationarity_and_multipliers() -> None:
    perturbation = 2.0e-10
    multiplier_gradient = 5.0e-11

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = (
            0.5 * values[0] ** 2
            + perturbation * (values[0] - 1.0) * (values[0] - 0.75)
            + multiplier_gradient * values[1]
        )
        return objective, jnp.reshape(values[1], (1,))

    result = run_projected_gauss_newton_trust_region(
        joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=1.0,
        ),
    )

    assert 0.0 < result.history.residual_gradient_defect[0] <= 1.0e-10
    np.testing.assert_allclose(
        result.history.current_stationarity_inf[0],
        1.0 + 0.25 * perturbation,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.history.predicted_reduction[0], 0.21875, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        result.multipliers, [-multiplier_gradient], rtol=0.0, atol=1.0e-20
    )
    assert result.final_certificate.residual_gradient_defect > 0.0
    assert result.final_certificate.certified
    assert result.usable


def test_ignored_nonfinite_coordinate_fails_current_and_final_certificates() -> None:
    initial = jnp.asarray([1.0, 0.0, jnp.nan], dtype=jnp.float64)

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.reshape(values[1], (1,))

    result = run_projected_gauss_newton_trust_region(
        joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
        ),
    )

    assert result.fatal
    assert result.status == int(ProjectedGaussNewtonStatus.FATAL_CURRENT_STATE)
    assert not result.final_certificate.coordinates_finite
    assert not result.final_certificate.certified
    assert not result.all_finite
    assert not result.usable


def test_low_ratio_accepted_step_contracts_to_quarter_step_norm() -> None:
    joint, residual = _curved_residual_problem(1.8)
    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        joint,
        residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.5,
            maximum_trust_radius=1.0,
        ),
    )

    assert result.accepted_steps == 1
    assert 0.0 < result.history.reduction_ratio[0] < 0.25
    np.testing.assert_allclose(result.trust_radius, 0.125, rtol=0.0, atol=1e-15)


def test_intermediate_ratio_keeps_radius() -> None:
    joint, residual = _curved_residual_problem(1.0)
    result = run_projected_gauss_newton_trust_region(
        joint,
        residual,
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.5,
            maximum_trust_radius=1.0,
        ),
    )

    assert result.accepted_steps == 1
    assert 0.25 <= result.history.reduction_ratio[0] <= 0.75
    assert result.trust_radius == 0.5


def test_high_ratio_expansion_clamps_to_maximum_radius() -> None:
    result = run_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=0.3,
        ),
    )

    assert result.history.reduction_ratio[0] > 0.75
    assert result.trust_radius == 0.3


def test_forced_contraction_clamps_to_minimum_radius() -> None:
    def nonfinite_candidate_joint(
        values: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        objective = jnp.where(values[0] < 0.9, jnp.nan, 0.5 * values[0] ** 2)
        return objective, jnp.reshape(values[1], (1,))

    result = run_projected_gauss_newton_trust_region(
        nonfinite_candidate_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            minimum_trust_radius=0.25,
            maximum_trust_radius=1.0,
        ),
    )

    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_NONFINITE
    )
    assert result.trust_radius == 0.25


def test_objective_rejection_preserves_coordinates_and_uses_ratio_radius() -> None:
    joint, residual = _curved_residual_problem(2.0)
    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        joint,
        residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.5,
            maximum_trust_radius=1.0,
        ),
    )

    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_OBJECTIVE
    )
    assert result.retryable_rejections == 1
    assert result.trust_radius == 0.125


def test_singular_trial_gram_is_retryable_correction_certificate_failure() -> None:
    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.reshape(values[1] * (values[0] - 0.75), (1,))

    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=1.0,
        ),
    )

    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_CORRECTION_CERTIFICATE
    )
    assert result.trust_radius == 0.0625


def test_finite_trial_gram_factorization_over_limit_retries_immutably() -> None:
    linear_tolerance = 1.0e-17

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        coupling = 0.04 * (1.0 - values[0])
        return 0.5 * values[0] ** 2, jnp.asarray(
            [
                values[1] + coupling * values[2] + (values[0] - 1.0) ** 2,
                values[2],
            ]
        )

    initial = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=1.0,
            linear_residual_tolerance=linear_tolerance,
            maximum_correction_step_ratio=1.0,
        ),
    )

    factorization_residual = result.history.trial_gram_factorization_relative_residual[
        0
    ]
    assert jnp.isfinite(factorization_residual)
    assert factorization_residual > linear_tolerance
    assert result.history.correction_relative_residual[0] <= linear_tolerance
    assert result.history.correction_forward_error_bound[0] < 1.0e-7
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_CORRECTION_CERTIFICATE
    )
    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.trust_radius == 0.0625


def test_finite_trial_gram_solve_residual_over_limit_fails_certificate() -> None:
    tolerance = 1.0e-10
    solve_residual = jnp.nextafter(
        jnp.asarray(tolerance, dtype=jnp.float64),
        jnp.asarray(jnp.inf, dtype=jnp.float64),
    )

    certified = jax.jit(_trial_correction_certified)(
        jnp.asarray(True),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        solve_residual,
        jnp.asarray(0.0, dtype=jnp.float64),
        linear_residual_tolerance=tolerance,
        forward_error_tolerance=1.0e-7,
    )

    assert jnp.isfinite(solve_residual)
    assert solve_residual > tolerance
    assert not certified


def test_failed_corrected_feasibility_is_retryable_and_preserves_state() -> None:
    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.reshape(values[1] - values[0] ** 2, (1,))

    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.1,
            maximum_trust_radius=1.0,
        ),
    )

    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_FEASIBILITY
    )
    assert result.trust_radius == 0.025


def test_iterative_normal_retraction_accepts_a_once_infeasible_trial() -> None:
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    one_correction = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.01,
            maximum_trust_radius=1.0,
            corrected_feasibility_tolerance=1.0e-12,
        ),
    )
    two_corrections = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.01,
            maximum_trust_radius=1.0,
            corrected_feasibility_tolerance=1.0e-12,
            maximum_nonlinear_corrections=2,
        ),
    )

    assert one_correction.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_FEASIBILITY
    )
    assert one_correction.history.nonlinear_corrections[0] == 1
    assert one_correction.history.candidate_feasibility_inf[0] > 1.0e-12
    assert two_corrections.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.ACCEPTED
    )
    assert two_corrections.history.nonlinear_corrections[0] == 2
    assert two_corrections.history.candidate_feasibility_inf[0] <= 1.0e-12
    assert two_corrections.history.correction_norm[0] > 0.0
    assert two_corrections.history.correction_step_ratio[0] <= 1.0e-3
    assert two_corrections.history.maximum_individual_correction_step_ratio[0] <= 1.0e-3
    assert two_corrections.history.correction_path_step_ratio[0] <= 2.0e-3
    assert (
        two_corrections.history.correction_path_step_ratio[0]
        >= two_corrections.history.correction_step_ratio[0]
    )


def test_correction_path_bounds_reject_hidden_opposite_corrections() -> None:
    denominator = jnp.asarray(1.0, dtype=jnp.float64)
    first_correction = jnp.asarray([1.1e-3], dtype=jnp.float64)
    second_correction = -first_correction
    net_ratio = jnp.linalg.norm(first_correction + second_correction) / denominator
    maximum_individual_ratio = (
        jnp.maximum(
            jnp.linalg.norm(first_correction),
            jnp.linalg.norm(second_correction),
        )
        / denominator
    )
    path_ratio = (
        jnp.linalg.norm(first_correction) + jnp.linalg.norm(second_correction)
    ) / denominator

    assert net_ratio == 0.0
    assert not _correction_path_bounds_valid(
        maximum_individual_ratio,
        path_ratio,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float64),
        correction_ratio_limit=jnp.asarray(1.0e-3, dtype=jnp.float64),
        corrected_radius_limit=jnp.asarray(1.0 + 1.0e-6, dtype=jnp.float64),
    )


def test_correction_path_bounds_are_closed_at_each_exact_limit() -> None:
    individual_limit = jnp.asarray(1.0e-3, dtype=jnp.float64)
    path_limit = jnp.asarray(2.0e-3, dtype=jnp.float64)
    corrected_radius_limit = jnp.asarray(1.0 + 1.0e-6, dtype=jnp.float64)
    count = jnp.asarray(2, dtype=jnp.int32)

    assert _correction_path_bounds_valid(
        individual_limit,
        path_limit,
        count,
        corrected_radius_limit,
        correction_ratio_limit=individual_limit,
        corrected_radius_limit=corrected_radius_limit,
    )
    assert not _correction_path_bounds_valid(
        jnp.nextafter(individual_limit, jnp.asarray(jnp.inf, dtype=jnp.float64)),
        path_limit,
        count,
        corrected_radius_limit,
        correction_ratio_limit=individual_limit,
        corrected_radius_limit=corrected_radius_limit,
    )
    assert not _correction_path_bounds_valid(
        individual_limit,
        jnp.nextafter(path_limit, jnp.asarray(jnp.inf, dtype=jnp.float64)),
        count,
        corrected_radius_limit,
        correction_ratio_limit=individual_limit,
        corrected_radius_limit=corrected_radius_limit,
    )


def test_iterative_normal_retraction_stops_after_feasibility_is_achieved() -> None:
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    common_options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=1,
        maximum_attempts=1,
        initial_trust_radius=0.01,
        maximum_trust_radius=1.0,
        corrected_feasibility_tolerance=1.0e-12,
        maximum_nonlinear_corrections=2,
    )
    two_corrections = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=common_options,
    )
    three_corrections = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=replace(common_options, maximum_nonlinear_corrections=3),
    )

    assert two_corrections.history.nonlinear_corrections[0] == 2
    assert three_corrections.history.nonlinear_corrections[0] == 2
    for two_leaf, three_leaf in zip(
        jax.tree.leaves(two_corrections),
        jax.tree.leaves(three_corrections),
        strict=True,
    ):
        np.testing.assert_array_equal(three_leaf, two_leaf)


def test_later_nonfinite_correction_certificate_rejects_immutably() -> None:
    @jax.custom_jvp
    def constraint(values: jax.Array) -> jax.Array:
        return values[1] - values[0] ** 2

    @constraint.defjvp
    def constraint_jvp(
        primals: tuple[jax.Array], tangents: tuple[jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        (values,), (values_dot,) = primals, tangents
        primal = constraint(values)
        regular_tangent = values_dot[1] - 2.0 * values[0] * values_dot[0]
        later_correction = (jnp.abs(primal) > 0.0) & (jnp.abs(primal) < 1.0e-8)
        tangent = jnp.where(later_correction, jnp.nan, regular_tangent)
        return primal, tangent

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.reshape(constraint(values), (1,))

    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    result = run_projected_gauss_newton_trust_region(
        joint,
        _linear_objective_residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.01,
            maximum_trust_radius=1.0,
            corrected_feasibility_tolerance=1.0e-12,
            maximum_nonlinear_corrections=3,
        ),
    )

    np.testing.assert_array_equal(result.optimizer_coordinates, initial)
    assert result.accepted_steps == 0
    assert result.retryable_rejections == 1
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_CORRECTION_CERTIFICATE
    )
    assert result.history.nonlinear_corrections[0] == 2
    assert not np.isfinite(result.history.correction_forward_error_bound[0])
    assert not np.isfinite(result.history.maximum_individual_correction_step_ratio[0])
    assert not np.isfinite(result.history.correction_path_step_ratio[0])


def test_default_normal_retraction_matches_explicit_single_correction() -> None:
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    common_options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=1,
        maximum_attempts=1,
        initial_trust_radius=0.1,
        maximum_trust_radius=1.0,
    )
    implicit_single = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=common_options,
    )
    explicit_single = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=replace(common_options, maximum_nonlinear_corrections=1),
    )
    for implicit_leaf, explicit_leaf in zip(
        jax.tree.leaves(implicit_single),
        jax.tree.leaves(explicit_single),
        strict=True,
    ):
        np.testing.assert_array_equal(explicit_leaf, implicit_leaf)
    np.testing.assert_array_equal(
        implicit_single.history.maximum_individual_correction_step_ratio,
        implicit_single.history.correction_step_ratio,
    )
    np.testing.assert_array_equal(
        implicit_single.history.correction_path_step_ratio,
        implicit_single.history.correction_step_ratio,
    )


def test_disabled_step_bound_safeguard_is_all_leaf_exact_for_max1() -> None:
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=1,
        maximum_attempts=1,
        initial_trust_radius=0.1,
        maximum_trust_radius=1.0,
        maximum_nonlinear_corrections=1,
    )
    implicit_disabled = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=options,
    )
    explicit_disabled = run_projected_gauss_newton_trust_region(
        _nonlinear_retraction_joint,
        _linear_objective_residual,
        initial,
        options=replace(options, enable_step_bound_safeguard=False),
    )

    for implicit_leaf, explicit_leaf in zip(
        jax.tree.leaves(implicit_disabled),
        jax.tree.leaves(explicit_disabled),
        strict=True,
    ):
        np.testing.assert_array_equal(explicit_leaf, implicit_leaf)
    assert implicit_disabled.history.subtrial_count[0] == 0
    assert implicit_disabled.history.selected_subtrial_index[0] == -1
    np.testing.assert_array_equal(
        implicit_disabled.history.subtrial_outcome[0],
        int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
    )


def test_interior_termination_and_final_certificates_pass() -> None:
    result = run_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=2.0,
            maximum_trust_radius=2.0,
        ),
    )

    assert result.history.steihaug_termination[0] == int(
        ProjectedSteihaugTermination.INTERIOR_CONVERGED
    )
    assert not result.history.steihaug_hit_boundary[0]
    assert (
        result.history.steihaug_final_projected_residual_norm[0]
        <= result.history.steihaug_projected_residual_target[0]
    )
    assert result.history.current_projection_tangency_relative_residual[0] <= 1e-10
    assert result.history.current_projection_solve_relative_residual[0] <= 1e-10
    assert result.history.current_projection_forward_error_bound[0] < 1e-7
    assert (
        result.history.steihaug_residual_projection_tangency_relative_residual[0]
        <= 1e-10
    )
    assert (
        result.history.steihaug_residual_projection_solve_relative_residual[0] <= 1e-10
    )
    assert result.history.steihaug_residual_projection_forward_error_bound[0] < 1e-7
    assert result.final_certificate.certified
    assert result.usable


def test_exact_null_curvature_termination_is_valid_but_mechanism_is_false() -> None:
    perturbation = 5.0e-11

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 + perturbation * values[0], jnp.reshape(values[1], (1,))

    def constant_residual(values: jax.Array) -> jax.Array:
        del values
        return jnp.ones((1,), dtype=jnp.float64)

    result = run_projected_gauss_newton_trust_region(
        joint,
        constant_residual,
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=1.0,
            projected_residual_tolerance=1.0e-12,
        ),
    )

    assert not result.fatal
    assert result.history.steihaug_termination[0] == int(
        ProjectedSteihaugTermination.NONPOSITIVE_CURVATURE
    )
    assert result.history.steihaug_hit_boundary[0]
    assert result.history.terminal_normalized_curvature[0] == 0.0
    assert result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_OBJECTIVE
    )
    assert not result.mechanism_exercised
    assert result.final_certificate.certified


def test_quality_latch_stops_on_first_corrected_accepted_candidate() -> None:
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=3,
        maximum_attempts=4,
        initial_trust_radius=0.25,
        maximum_trust_radius=1.0,
    )

    def quality(candidate: ProjectedGaussNewtonAcceptedState) -> jax.Array:
        return candidate.objective <= 0.3

    loop_result = run_projected_gauss_newton_trust_region_loop(
        _linear_equality_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=options,
        accepted_state_quality_predicate=quality,
    )

    assert loop_result.device_quality_candidate_reached
    assert loop_result.status == int(
        ProjectedGaussNewtonStatus.DEVICE_QUALITY_CANDIDATE
    )
    assert loop_result.first_quality_attempt == 1
    assert loop_result.first_quality_accepted_step == 1
    assert loop_result.attempts == 1
    assert loop_result.accepted_steps == 1
    assert not loop_result.bounded_complete
    np.testing.assert_allclose(
        loop_result.optimizer_coordinates, [0.75, 0.0], rtol=0.0, atol=1e-15
    )
    assert loop_result.accepted_optimizer_coordinates.shape == (4, 2)
    np.testing.assert_allclose(
        loop_result.accepted_optimizer_coordinates[:2],
        [[1.0, 0.0], [0.75, 0.0]],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        loop_result.accepted_state_mask, [True, True, False, False]
    )
    assert loop_result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.ACCEPTED
    )
    np.testing.assert_array_equal(
        loop_result.history.outcome[1:],
        int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
    )
    finalized = finalize_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        _linear_objective_residual,
        loop_result,
        options=options,
    )
    assert finalized.device_quality_candidate_reached
    assert finalized.first_quality_attempt == 1
    assert finalized.first_quality_accepted_step == 1
    assert not finalized.bounded_complete
    assert finalized.usable


def test_quality_latch_can_first_hit_on_last_accepted_step() -> None:
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=2,
        maximum_attempts=3,
        initial_trust_radius=0.25,
        maximum_trust_radius=1.0,
    )

    def quality(candidate: ProjectedGaussNewtonAcceptedState) -> jax.Array:
        return candidate.objective <= 0.04

    loop_result = run_projected_gauss_newton_trust_region_loop(
        _linear_equality_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=options,
        accepted_state_quality_predicate=quality,
    )

    assert loop_result.device_quality_candidate_reached
    assert loop_result.first_quality_attempt == 2
    assert loop_result.first_quality_accepted_step == 2
    assert loop_result.accepted_steps == 2
    assert loop_result.bounded_complete
    assert loop_result.status == int(
        ProjectedGaussNewtonStatus.DEVICE_QUALITY_CANDIDATE
    )
    assert loop_result.history.outcome[2] == int(
        ProjectedGaussNewtonAttemptOutcome.INACTIVE
    )
    np.testing.assert_allclose(
        loop_result.accepted_optimizer_coordinates,
        [[1.0, 0.0], [0.75, 0.0], [0.25, 0.0]],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(loop_result.accepted_state_mask, [True, True, True])


def test_rejected_candidate_cannot_trigger_quality_latch() -> None:
    joint, residual = _curved_residual_problem(2.0)

    def always_quality(_candidate: ProjectedGaussNewtonAcceptedState) -> jax.Array:
        return jnp.asarray(True)

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    loop_result = run_projected_gauss_newton_trust_region_loop(
        joint,
        residual,
        initial,
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.5,
            maximum_trust_radius=1.0,
        ),
        accepted_state_quality_predicate=always_quality,
    )

    assert not loop_result.device_quality_candidate_reached
    assert loop_result.first_quality_attempt == 0
    assert loop_result.first_quality_accepted_step == 0
    assert loop_result.status == int(ProjectedGaussNewtonStatus.ATTEMPT_LIMIT)
    assert loop_result.history.outcome[0] == int(
        ProjectedGaussNewtonAttemptOutcome.RETRY_OBJECTIVE
    )
    np.testing.assert_array_equal(loop_result.optimizer_coordinates, initial)
    np.testing.assert_array_equal(loop_result.accepted_state_mask, [True, False])
    np.testing.assert_array_equal(
        loop_result.accepted_optimizer_coordinates[0], initial
    )
    np.testing.assert_array_equal(
        loop_result.accepted_optimizer_coordinates[1], jnp.zeros_like(initial)
    )


def test_quality_predicate_observes_corrected_candidate() -> None:
    def nonlinear_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.reshape(values[1] - values[0] ** 2, (1,))

    def corrected_quality(candidate: ProjectedGaussNewtonAcceptedState) -> jax.Array:
        return (
            jnp.all(jnp.isfinite(candidate.optimizer_coordinates))
            & jnp.all(jnp.isfinite(candidate.constraints))
            & (candidate.scaled_feasibility_inf <= 1.0e-10)
        )

    loop_result = run_projected_gauss_newton_trust_region_loop(
        nonlinear_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=1.0e-5,
            minimum_trust_radius=1.0e-8,
            maximum_trust_radius=1.0e-3,
        ),
        accepted_state_quality_predicate=corrected_quality,
    )

    assert loop_result.device_quality_candidate_reached
    assert loop_result.history.correction_norm[0] > 0.0
    assert loop_result.history.candidate_feasibility_inf[0] <= 1.0e-10
    np.testing.assert_array_equal(
        loop_result.accepted_optimizer_coordinates[1],
        loop_result.optimizer_coordinates,
    )


def test_fatal_status_has_precedence_over_quality() -> None:
    def always_quality(_candidate: ProjectedGaussNewtonAcceptedState) -> jax.Array:
        return jnp.asarray(True)

    loop_result = run_projected_gauss_newton_trust_region_loop(
        _linear_equality_joint,
        lambda values: jnp.reshape(2.0 * values[0], (1,)),
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
        ),
        accepted_state_quality_predicate=always_quality,
    )

    assert loop_result.fatal
    assert loop_result.status == int(ProjectedGaussNewtonStatus.FATAL_CURRENT_STATE)
    assert not loop_result.device_quality_candidate_reached
    assert loop_result.first_quality_attempt == 0


def test_split_loop_finalization_matches_legacy_wrapper() -> None:
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=2,
        maximum_attempts=2,
        initial_trust_radius=0.25,
        maximum_trust_radius=1.0,
    )
    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    legacy = run_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        _linear_objective_residual,
        initial,
        options=options,
    )
    loop_result = run_projected_gauss_newton_trust_region_loop(
        _linear_equality_joint,
        _linear_objective_residual,
        initial,
        options=options,
    )
    jax.block_until_ready(loop_result)
    assert not hasattr(loop_result, "final_certificate")
    split = finalize_projected_gauss_newton_trust_region(
        _linear_equality_joint,
        _linear_objective_residual,
        loop_result,
        options=options,
    )

    for legacy_leaf, split_leaf in zip(
        jax.tree.leaves(legacy), jax.tree.leaves(split), strict=True
    ):
        np.testing.assert_array_equal(legacy_leaf, split_leaf)
    assert not split.device_quality_candidate_reached
    assert split.status == int(ProjectedGaussNewtonStatus.BOUNDED_COMPLETE)
    assert split.bounded_complete
    assert split.usable


def test_quality_loop_jits_without_callbacks_and_leaves_tail_inactive() -> None:
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=3,
        maximum_attempts=4,
        initial_trust_radius=0.25,
        maximum_trust_radius=1.0,
    )

    def run(initial: jax.Array):
        def quality(candidate: ProjectedGaussNewtonAcceptedState) -> jax.Array:
            return candidate.objective <= 0.3

        return run_projected_gauss_newton_trust_region_loop(
            _linear_equality_joint,
            _linear_objective_residual,
            initial,
            options=options,
            accepted_state_quality_predicate=quality,
        )

    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    executable = jax.jit(run).lower(initial).compile()
    loop_result = executable(initial)

    assert loop_result.attempts == 1
    np.testing.assert_array_equal(
        loop_result.history.outcome[1:],
        int(ProjectedGaussNewtonAttemptOutcome.INACTIVE),
    )
    compiled_text = executable.runtime_executable().hlo_modules()[0].to_string()
    assert "host_callback" not in compiled_text
    assert "io_callback" not in compiled_text


def test_disabled_gntr_annotations_never_enter_named_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_named_scope(*_args, **_kwargs):
        pytest.fail("disabled GNTR annotations must not enter jax.named_scope")

    monkeypatch.setattr(trace_annotations.jax, "named_scope", unexpected_named_scope)
    result = run_projected_gauss_newton_trust_region_loop(
        _linear_equality_joint,
        _linear_objective_residual,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        options=ProjectedGaussNewtonOptions(
            maximum_accepted_steps=1,
            maximum_attempts=1,
            initial_trust_radius=0.25,
            maximum_trust_radius=1.0,
        ),
    )

    assert result.accepted_steps == 1


def test_enabled_gntr_annotations_preserve_every_numerical_leaf_exactly() -> None:
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=2,
        maximum_attempts=3,
        initial_trust_radius=0.25,
        maximum_trust_radius=1.0,
    )

    def run(initial: jax.Array) -> ProjectedGaussNewtonLoopResult:
        return run_projected_gauss_newton_trust_region_loop(
            _linear_equality_joint,
            _linear_objective_residual,
            initial,
            options=options,
        )

    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    disabled = jax.jit(run)(initial)
    jax.clear_caches()
    with trace_session():
        enabled = jax.jit(run)(initial)

    for disabled_leaf, enabled_leaf in zip(
        jax.tree.leaves(disabled), jax.tree.leaves(enabled), strict=True
    ):
        np.testing.assert_array_equal(enabled_leaf, disabled_leaf)


def test_production_shape_normalized_ir_is_annotation_invariant() -> None:
    assert len(GNTR_DIAGNOSTIC_PHASES) == 7
    assert len(GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256) == 64
    state_size = 716
    equality_size = 255
    residual_size = 2110
    equality_matrix = jnp.pad(
        jnp.eye(equality_size, dtype=jnp.float64),
        ((0, 0), (0, state_size - equality_size)),
    )
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=256,
        maximum_attempts=300,
    )

    def run(initial: jax.Array) -> ProjectedGaussNewtonLoopResult:
        def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            return 0.5 * jnp.vdot(values, values), equality_matrix @ values

        def residual(values: jax.Array) -> jax.Array:
            return jnp.pad(values, (0, residual_size - state_size))

        return run_projected_gauss_newton_trust_region_loop(
            joint,
            residual,
            initial,
            options=options,
        )

    initial = jnp.zeros((state_size,), dtype=jnp.float64)
    disabled_ir = normalized_jax_ir(run, initial)
    jax.clear_caches()
    with trace_session():
        enabled_ir = normalized_jax_ir(run, initial)
        lowered = jax.jit(run).lower(initial)

    assert enabled_ir == disabled_ir
    stablehlo_with_debug_info = lowered.compiler_ir(
        dialect="stablehlo"
    ).operation.get_asm(enable_debug_info=True, pretty_debug_info=True)
    for phase in GNTR_DIAGNOSTIC_PHASES:
        assert phase.value in stablehlo_with_debug_info
    acceptance_scope = GNTR_DIAGNOSTIC_PHASES[-1].value
    for owned_operation in (
        "gt",
        "jit(clip)",
        "select_n",
        "add",
        "cond/branch_1_fun/scatter",
    ):
        assert f"{acceptance_scope}/{owned_operation}" in stablehlo_with_debug_info
