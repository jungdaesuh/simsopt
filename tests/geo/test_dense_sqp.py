from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers import dense_sqp as dense_sqp_module
from simsopt_jax.geo.optimizers.dense_sqp import (
    DenseSQPOptions,
    DenseSQPStatus,
    materialize_joint_vjp_rows,
    powell_damped_bfgs_update,
    prepare_dense_sqp,
    solve_dense_sqp_kkt,
)


def _quadratic_joint(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
    objective = 0.5 * jnp.dot(coordinates, coordinates)
    constraints = jnp.asarray(
        [coordinates[0] + coordinates[1] - 1.0], dtype=coordinates.dtype
    )
    return objective, constraints


@pytest.mark.parametrize("batch_width", (1, 2, 3, 8))
def test_joint_vjp_rows_match_independent_derivatives_with_exact_tail(
    batch_width: int,
) -> None:
    coordinates = jnp.linspace(-0.4, 0.7, 5, dtype=jnp.float64)

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = jnp.sum(jnp.sin(values) + 0.25 * values**2)
        constraints = jnp.asarray(
            [values[0] * values[2], values[1] + values[3] ** 2],
            dtype=values.dtype,
        )
        return objective, constraints

    actual = jax.jit(
        lambda values: materialize_joint_vjp_rows(
            joint, values, batch_width=batch_width
        )
    )(coordinates)
    expected_gradient = jax.grad(lambda values: joint(values)[0])(coordinates)
    expected_jacobian = jax.jacrev(lambda values: joint(values)[1])(coordinates)

    assert actual.joint_rows.shape == (3, 5)
    np.testing.assert_allclose(actual.objective_gradient, expected_gradient)
    np.testing.assert_allclose(actual.constraint_jacobian, expected_jacobian)
    np.testing.assert_allclose(
        actual.joint_rows,
        jnp.vstack((expected_gradient, expected_jacobian)),
    )


def test_joint_vjp_rows_executes_one_primal_and_has_no_host_callback() -> None:
    primal_calls = 0

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        nonlocal primal_calls
        primal_calls += 1
        return jnp.sum(values**2), jnp.sin(values[:2])

    coordinates = jnp.linspace(-0.2, 0.4, 5, dtype=jnp.float64)
    materialize_joint_vjp_rows(joint, coordinates, batch_width=2)
    assert primal_calls == 1

    jaxpr = jax.make_jaxpr(
        lambda values: materialize_joint_vjp_rows(joint, values, batch_width=2)
    )(coordinates)
    primitive_names = {equation.primitive.name for equation in jaxpr.jaxpr.eqns}
    assert "debug_callback" not in primitive_names
    assert "io_callback" not in primitive_names


def test_dense_kkt_step_satisfies_reconstructed_system() -> None:
    bfgs = jnp.asarray([[3.0, 0.4], [0.4, 2.0]], dtype=jnp.float64)
    jacobian = jnp.asarray([[1.0, -2.0]], dtype=jnp.float64)
    dual_residual = jnp.asarray([0.7, -0.3], dtype=jnp.float64)
    constraints = jnp.asarray([0.2], dtype=jnp.float64)

    step = jax.jit(solve_dense_sqp_kkt)(bfgs, jacobian, dual_residual, constraints)
    kkt = np.block(
        [
            [np.asarray(bfgs), np.asarray(jacobian.T)],
            [np.asarray(jacobian), np.zeros((1, 1))],
        ]
    )
    expected = np.linalg.solve(
        kkt, -np.concatenate((np.asarray(dual_residual), np.asarray(constraints)))
    )

    assert bool(step.valid)
    np.testing.assert_allclose(
        np.concatenate((step.primal_step, step.multiplier_step)),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert float(step.kkt_relative_residual) <= 1.0e-10
    expected_reciprocal_condition = 1.0 / np.linalg.cond(kkt, p=2)
    solution = np.concatenate(
        (np.asarray(step.primal_step), np.asarray(step.multiplier_step))
    )
    right_hand_side = -np.concatenate(
        (np.asarray(dual_residual), np.asarray(constraints))
    )
    expected_scaled_residual = np.linalg.norm(
        kkt @ solution - right_hand_side, ord=2
    ) / (np.linalg.norm(kkt, ord=2) * np.linalg.norm(solution, ord=2))
    assert float(step.kkt_reciprocal_condition) == pytest.approx(
        expected_reciprocal_condition, rel=1.0e-13
    )
    assert float(step.kkt_solution_scaled_residual) == pytest.approx(
        expected_scaled_residual, abs=1.0e-16
    )
    expected_forward_error_bound = expected_scaled_residual / (
        expected_reciprocal_condition - expected_scaled_residual
    )
    assert float(step.kkt_forward_error_bound) == pytest.approx(
        expected_forward_error_bound, rel=1.0e-13
    )
    assert float(step.schur_relative_residual) <= 1.0e-10
    assert float(step.bfgs_cholesky_relative_pivot) > 0.0
    assert float(step.schur_cholesky_relative_pivot) > 0.0


def test_dense_kkt_regularizes_spd_failure_and_fails_closed_on_rank_loss() -> None:
    regularized = solve_dense_sqp_kkt(
        jnp.diag(jnp.asarray([0.0, 1.0], dtype=jnp.float64)),
        jnp.asarray([[1.0, 0.0]], dtype=jnp.float64),
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        regularization_ladder=(0.0, 1.0e-6),
    )
    assert bool(regularized.valid)
    assert float(regularized.selected_regularization) == pytest.approx(1.0e-6)
    assert int(regularized.regularization_candidates_tested) == 2
    assert float(regularized.bfgs_cholesky_relative_pivot) > 1.4901161193847656e-08
    assert float(regularized.schur_cholesky_relative_pivot) > 1.4901161193847656e-08

    rank_deficient = solve_dense_sqp_kkt(
        jnp.eye(2, dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=jnp.float64),
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.ones((2,), dtype=jnp.float64),
        regularization_ladder=(0.0, 1.0e-6),
    )
    assert not bool(rank_deficient.valid)
    assert int(rank_deficient.regularization_candidates_tested) == 2
    deficient_pivot = float(rank_deficient.schur_cholesky_relative_pivot)
    assert not np.isfinite(deficient_pivot) or deficient_pivot <= 1.4901161193847656e-08


def test_full_kkt_forward_error_gate_accepts_small_rho_with_small_error() -> None:
    bfgs = jnp.eye(2, dtype=jnp.float64)
    jacobian = jnp.diag(jnp.asarray([1.0, 0.02], dtype=jnp.float64))
    dual_residual = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    constraints = jnp.asarray([0.1, -0.2], dtype=jnp.float64)

    step = solve_dense_sqp_kkt(
        bfgs,
        jacobian,
        dual_residual,
        constraints,
        regularization_ladder=(0.0,),
    )
    kkt = np.block(
        [
            [np.eye(2), np.asarray(jacobian.T)],
            [np.asarray(jacobian), np.zeros((2, 2))],
        ]
    )

    assert float(step.bfgs_cholesky_relative_pivot) == pytest.approx(1.0)
    assert float(step.schur_cholesky_relative_pivot) == pytest.approx(0.02)
    assert float(step.kkt_reciprocal_condition) == pytest.approx(
        1.0 / np.linalg.cond(kkt, p=2), rel=1.0e-12
    )
    assert float(step.kkt_reciprocal_condition) == pytest.approx(
        0.0002471147890910887, rel=1.0e-12
    )
    assert float(step.kkt_forward_error_bound) < 1.0e-7
    assert bool(step.valid)


def test_full_kkt_certificate_is_jittable_without_host_callbacks() -> None:
    bfgs = jnp.asarray([[2.0, 0.1], [0.1, 1.5]], dtype=jnp.float64)
    jacobian = jnp.asarray([[1.0, -1.0]], dtype=jnp.float64)
    dual_residual = jnp.asarray([0.2, -0.4], dtype=jnp.float64)
    constraints = jnp.asarray([0.3], dtype=jnp.float64)
    solve = jax.jit(solve_dense_sqp_kkt)

    result = solve(bfgs, jacobian, dual_residual, constraints)
    assert bool(result.valid)
    jaxpr_text = str(
        jax.make_jaxpr(solve_dense_sqp_kkt)(bfgs, jacobian, dual_residual, constraints)
    )
    assert "debug_callback" not in jaxpr_text
    assert "io_callback" not in jaxpr_text


def test_full_kkt_scaled_residual_zero_denominator_is_safe() -> None:
    step = solve_dense_sqp_kkt(
        jnp.eye(2, dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0]], dtype=jnp.float64),
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.zeros((1,), dtype=jnp.float64),
        regularization_ladder=(0.0,),
    )

    assert bool(step.valid)
    assert float(step.kkt_solution_scaled_residual) == 0.0
    np.testing.assert_array_equal(step.primal_step, jnp.zeros((2,)))
    np.testing.assert_array_equal(step.multiplier_step, jnp.zeros((1,)))


def test_powell_damped_bfgs_preserves_symmetry_and_resets_invalid_update() -> None:
    bfgs = jnp.asarray([[2.0, 0.2], [0.2, 1.0]], dtype=jnp.float64)
    step = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    adverse_difference = jnp.asarray([-1.0, 0.25], dtype=jnp.float64)

    update = powell_damped_bfgs_update(bfgs, step, adverse_difference)
    assert not bool(update.reset)
    assert float(update.theta) < 1.0
    np.testing.assert_allclose(update.matrix, update.matrix.T)
    assert np.min(np.linalg.eigvalsh(np.asarray(update.matrix))) > 0.0

    reset = powell_damped_bfgs_update(bfgs, jnp.zeros_like(step), adverse_difference)
    assert bool(reset.reset)
    np.testing.assert_array_equal(reset.matrix, np.eye(2))


def test_prepared_dense_sqp_reaches_independent_quadratic_primal_dual_solution() -> (
    None
):
    initial = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    prepared = prepare_dense_sqp(
        _quadratic_joint,
        initial,
        options=DenseSQPOptions(maximum_iterations=10),
    )
    result = prepared.run(initial)

    assert int(result.status) == int(DenseSQPStatus.CONVERGED)
    assert bool(result.converged)
    assert not bool(result.fatal)
    assert bool(result.all_accepted_states_finite)
    assert bool(result.all_finite)
    assert float(result.final_bfgs_cholesky_relative_pivot) > 0.0
    assert float(result.final_schur_cholesky_relative_pivot) > 0.0
    np.testing.assert_allclose(result.optimizer_coordinates, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.multipliers, [-0.5], atol=1e-12)
    np.testing.assert_allclose(result.constraints, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.stationarity, 0.0, atol=1e-12)
    assert int(result.iterations) == 1
    assert int(result.derivative_builds) == 2
    assert int(result.joint_evaluations) == (
        int(result.derivative_builds) + int(result.line_search_evaluations)
    )


def test_prepared_dense_sqp_converges_on_nonlinear_equality() -> None:
    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = 0.5 * (values[0] - 2.0) ** 2
        constraints = jnp.asarray([values[0] ** 2 - 1.0], dtype=values.dtype)
        return objective, constraints

    initial = jnp.asarray([1.5], dtype=jnp.float64)
    result = prepare_dense_sqp(
        joint,
        initial,
        options=DenseSQPOptions(maximum_iterations=20, objective_maximum=1.0),
    ).run(initial)

    assert int(result.status) == int(DenseSQPStatus.CONVERGED)
    np.testing.assert_allclose(result.optimizer_coordinates, [1.0], atol=1e-9)
    np.testing.assert_allclose(result.multipliers, [0.5], atol=1e-8)
    assert abs(float(result.constraints[0])) <= 1.0e-10
    assert abs(float(result.stationarity[0])) <= 1.0e-7


def test_dense_sqp_restores_nonlinear_feasibility_and_records_telemetry() -> None:
    def circle_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = 0.5 * (values[1] - 1.0) ** 2
        constraints = jnp.asarray(
            [values[0] ** 2 + values[1] ** 2 - 1.0], dtype=values.dtype
        )
        return objective, constraints

    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    result = prepare_dense_sqp(
        circle_joint,
        initial,
        options=DenseSQPOptions(maximum_iterations=3),
    ).run(initial)

    iterations = int(result.iterations)
    assert iterations == 3
    telemetry = result.convergence_telemetry
    assert bool(jnp.all(telemetry.restoration_applied[:iterations] == 1))
    assert bool(
        jnp.all(result.history.feasibility_infinity_norm[:iterations] <= 1.0e-10)
    )
    assert bool(jnp.all(jnp.isfinite(telemetry.merit[:iterations])))
    assert bool(jnp.all(jnp.isfinite(telemetry.penalty[:iterations])))
    assert bool(jnp.all(telemetry.penalty[:iterations] > 0.0))
    assert bool(
        jnp.all(jnp.isfinite(telemetry.multiplier_update_infinity_norm[:iterations]))
    )
    assert bool(jnp.all(telemetry.bfgs_reset[:iterations] >= 0))
    assert int(result.line_search_evaluations) == 54
    assert int(result.joint_evaluations) == (
        int(result.derivative_builds) + int(result.line_search_evaluations)
    )


def test_dense_sqp_fails_closed_when_normal_restoration_factor_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_normal_factor(
        constraint_jacobian: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        size = constraint_jacobian.shape[0]
        return (
            jnp.full((size, size), jnp.nan, dtype=constraint_jacobian.dtype),
            jnp.asarray(False),
        )

    monkeypatch.setattr(
        dense_sqp_module,
        "_normal_restoration_factor",
        invalid_normal_factor,
    )

    def circle_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = 0.5 * (values[1] - 1.0) ** 2
        constraints = jnp.asarray(
            [values[0] ** 2 + values[1] ** 2 - 1.0], dtype=values.dtype
        )
        return objective, constraints

    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    result = prepare_dense_sqp(circle_joint, initial).run(initial)

    assert int(result.status) == int(DenseSQPStatus.GLOBALIZATION_FAILED)
    assert int(result.iterations) == 0
    assert bool(result.fatal)
    assert int(result.restoration_numerical_failures) > 0
    assert int(result.rejected_nonfinite_trials) > 0
    assert not bool(result.all_finite)
    np.testing.assert_array_equal(result.optimizer_coordinates, initial)


def test_dense_sqp_rejects_nonfinite_restoration_before_masked_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overflowing_normal_factor(
        constraint_jacobian: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        size = constraint_jacobian.shape[0]
        return (
            jnp.eye(size, dtype=constraint_jacobian.dtype)
            * jnp.asarray(1.0e-300, dtype=constraint_jacobian.dtype),
            jnp.asarray(True),
        )

    monkeypatch.setattr(
        dense_sqp_module,
        "_normal_restoration_factor",
        overflowing_normal_factor,
    )

    def masked_circle_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        finite = jnp.all(jnp.isfinite(values))
        safe_values = jnp.where(finite, values, jnp.zeros_like(values))
        objective = 0.5 * (safe_values[1] - 1.0) ** 2
        constraints = jnp.asarray(
            [safe_values[0] ** 2 + safe_values[1] ** 2 - 1.0],
            dtype=values.dtype,
        )
        return objective, constraints

    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    result = prepare_dense_sqp(masked_circle_joint, initial).run(initial)

    assert int(result.status) == int(DenseSQPStatus.GLOBALIZATION_FAILED)
    assert int(result.iterations) == 0
    assert int(result.restoration_numerical_failures) > 0
    assert int(result.rejected_nonfinite_trials) > 0
    assert not bool(result.all_finite)
    np.testing.assert_array_equal(result.optimizer_coordinates, initial)


def test_prepared_dense_sqp_distinguishes_objective_and_budget_terminations() -> None:
    def quality_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.asarray(2.0, dtype=values.dtype), values[:1]

    initial = jnp.zeros((1,), dtype=jnp.float64)
    quality = prepare_dense_sqp(
        quality_joint,
        initial,
        options=DenseSQPOptions(objective_maximum=1.0),
    ).run(initial)
    assert int(quality.status) == int(DenseSQPStatus.OBJECTIVE_QUALITY_REJECTED)
    assert not bool(quality.converged)
    assert not bool(quality.fatal)
    assert int(quality.kkt_solves) == 0
    assert bool(quality.all_finite)

    def nonlinear_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * values[0] ** 2, jnp.asarray(
            [values[0] ** 2 - 1.0], dtype=values.dtype
        )

    iteration_limited = prepare_dense_sqp(
        nonlinear_joint,
        jnp.asarray([2.0], dtype=jnp.float64),
        options=DenseSQPOptions(maximum_iterations=1),
    ).run(jnp.asarray([2.0], dtype=jnp.float64))
    assert int(iteration_limited.status) == int(DenseSQPStatus.ITERATION_LIMIT)

    evaluation_limited = prepare_dense_sqp(
        _quadratic_joint,
        jnp.asarray([2.0, -1.0], dtype=jnp.float64),
        options=DenseSQPOptions(maximum_joint_evaluations=1),
    ).run(jnp.asarray([2.0, -1.0], dtype=jnp.float64))
    assert int(evaluation_limited.status) == int(DenseSQPStatus.EVALUATION_LIMIT)


def test_prepared_dense_sqp_rejects_nonfinite_trials_and_fails_globalization() -> None:
    def nonfinite_trials(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = jnp.where(
            values[0] == 0.0,
            jnp.asarray(0.0, dtype=values.dtype),
            jnp.asarray(jnp.nan, dtype=values.dtype),
        )
        return objective, values - 1.0

    initial = jnp.zeros((1,), dtype=jnp.float64)
    result = prepare_dense_sqp(nonfinite_trials, initial).run(initial)

    assert int(result.status) == int(DenseSQPStatus.GLOBALIZATION_FAILED)
    assert bool(result.fatal)
    assert int(result.rejected_nonfinite_trials) == 22
    np.testing.assert_array_equal(result.optimizer_coordinates, initial)


def test_forward_error_gate_keeps_well_conditioned_retry_on_globalization_path() -> (
    None
):
    def nonfinite_trials(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = jnp.where(
            values[0] == 0.0,
            jnp.asarray(0.0, dtype=values.dtype),
            jnp.asarray(jnp.nan, dtype=values.dtype),
        )
        return objective, values - 1.0

    initial = jnp.zeros((1,), dtype=jnp.float64)
    result = prepare_dense_sqp(
        nonfinite_trials,
        initial,
        options=DenseSQPOptions(
            initial_bfgs_identity_scale=0.1,
            kkt_forward_error_tolerance=1.0e-7,
        ),
    ).run(initial)

    assert int(result.status) == int(DenseSQPStatus.GLOBALIZATION_FAILED)
    assert int(result.kkt_solves) == 2
    assert int(result.line_search_evaluations) == 22
    assert int(result.rejected_nonfinite_trials) == 22
    assert int(result.joint_evaluations) == 23
    assert int(result.regularization_candidates_tested) == 2
    np.testing.assert_array_equal(result.optimizer_coordinates, initial)

    budget_limited = prepare_dense_sqp(
        nonfinite_trials,
        initial,
        options=DenseSQPOptions(
            maximum_joint_evaluations=5,
            initial_bfgs_identity_scale=0.1,
            kkt_forward_error_tolerance=1.0e-7,
        ),
    ).run(initial)
    assert int(budget_limited.status) == int(DenseSQPStatus.EVALUATION_LIMIT)
    assert int(budget_limited.kkt_solves) == 1
    assert int(budget_limited.line_search_evaluations) == 3
    assert int(budget_limited.rejected_nonfinite_trials) == 3
    assert int(budget_limited.joint_evaluations) == 4


def test_nan_initial_coordinates_cannot_report_converged() -> None:
    def constant_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray(0.0, dtype=values.dtype),
            jnp.zeros((1,), dtype=values.dtype),
        )

    initial = jnp.asarray([jnp.nan], dtype=jnp.float64)
    result = prepare_dense_sqp(constant_joint, initial).run(initial)

    assert int(result.status) == int(DenseSQPStatus.RANK_DEFICIENT_OR_UNSTABLE_KKT)
    assert not bool(result.converged)
    assert bool(result.fatal)
    assert not bool(result.all_accepted_states_finite)
    assert not bool(result.all_finite)


def test_second_consecutive_bfgs_reset_has_priority_over_successful_kkt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def force_identity_reset(
        bfgs_matrix: jax.Array,
        step: jax.Array,
        lagrangian_gradient_difference: jax.Array,
        *,
        curvature_fraction: float = 0.2,
    ) -> dense_sqp_module.PowellBFGSUpdate:
        del bfgs_matrix, lagrangian_gradient_difference, curvature_fraction
        return dense_sqp_module.PowellBFGSUpdate(
            matrix=jnp.eye(step.shape[0], dtype=step.dtype),
            reset=jnp.asarray(True),
            theta=jnp.asarray(jnp.nan, dtype=step.dtype),
            all_finite=jnp.asarray(True),
        )

    monkeypatch.setattr(
        dense_sqp_module, "powell_damped_bfgs_update", force_identity_reset
    )

    def nonlinear_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * (values[0] - 2.0) ** 2, jnp.asarray(
            [values[0] ** 2 - 1.0], dtype=values.dtype
        )

    initial = jnp.asarray([1.5], dtype=jnp.float64)
    result = dense_sqp_module.prepare_dense_sqp(
        nonlinear_joint,
        initial,
        options=DenseSQPOptions(maximum_consecutive_bfgs_resets=2),
    ).run(initial)

    assert int(result.status) == int(DenseSQPStatus.BFGS_UPDATE_FAILED)
    assert int(result.iterations) == 2
    assert int(result.kkt_solves) == 2
    assert int(result.bfgs_resets) == 2
    assert bool(result.fatal)


def test_rejected_nonfinite_trials_do_not_poison_eventual_finite_success() -> None:
    def bounded_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = jnp.where(
            jnp.abs(values[0]) > 2.0,
            jnp.asarray(jnp.nan, dtype=values.dtype),
            jnp.asarray(0.0, dtype=values.dtype),
        )
        return objective, jnp.asarray([values[0] ** 2 - 1.0], dtype=values.dtype)

    initial = jnp.asarray([0.1], dtype=jnp.float64)
    result = prepare_dense_sqp(
        bounded_joint,
        initial,
        options=DenseSQPOptions(maximum_iterations=20),
    ).run(initial)

    assert int(result.status) == int(DenseSQPStatus.CONVERGED)
    assert int(result.rejected_nonfinite_trials) >= 2
    assert bool(result.all_accepted_states_finite)
    assert bool(result.all_finite)
    np.testing.assert_allclose(result.optimizer_coordinates, [1.0], atol=1.0e-8)


def test_prepared_dense_sqp_latches_nonfinite_derivative_and_bfgs_failure() -> None:
    @jax.custom_jvp
    def poisoned_constant(values: jax.Array) -> jax.Array:
        return jnp.asarray(0.0, dtype=values.dtype)

    @poisoned_constant.defjvp
    def poisoned_constant_jvp(primals, tangents):
        (values,) = primals
        (tangent,) = tangents
        multiplier = jnp.where(
            values[0] == 0.0,
            jnp.asarray(0.0, dtype=values.dtype),
            jnp.asarray(jnp.nan, dtype=values.dtype),
        )
        return poisoned_constant(values), multiplier * jnp.sum(tangent)

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return poisoned_constant(values), values - 1.0

    initial = jnp.zeros((1,), dtype=jnp.float64)
    result = prepare_dense_sqp(
        joint,
        initial,
        options=DenseSQPOptions(maximum_consecutive_bfgs_resets=1),
    ).run(initial)

    assert int(result.status) == int(DenseSQPStatus.BFGS_UPDATE_FAILED)
    assert bool(result.fatal)
    assert int(result.bfgs_resets) == 1
    assert not bool(result.all_accepted_states_finite)
    assert not bool(result.all_finite)
