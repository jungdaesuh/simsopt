from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.filter_trust_region_sqp import (
    FilterTrustRegionSQPOptions,
    FilterTrustRegionSQPStatus,
    prepare_filter_trust_region_sqp,
)


def _quadratic_equality(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
    return 0.5 * jnp.dot(coordinates, coordinates), jnp.asarray(
        [coordinates[0] + coordinates[1] - 1.0], dtype=coordinates.dtype
    )


def test_linear_equality_converges_with_tangent_and_projection_certificates() -> None:
    initial = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(
        _quadratic_equality,
        initial,
        options=FilterTrustRegionSQPOptions(maximum_iterations=4),
    ).run(initial)

    assert result.status == int(FilterTrustRegionSQPStatus.CONVERGED)
    assert result.converged
    np.testing.assert_allclose(result.optimizer_coordinates, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.constraints, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.stationarity, 0.0, atol=1e-12)
    assert result.history.normal_step_norm[0] == 0.0
    assert result.history.tangential_step_norm[0] > 0.0
    assert result.final_tangency_relative_residual <= 1.0e-10
    assert result.final_multiplier_projection_relative_residual <= 1.0e-10
    assert result.final_multiplier_projection_forward_error_bound < 1.0e-7


def test_feasibility_step_uses_normal_dogleg_and_filter_acceptance() -> None:
    def joint(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 100.0 * coordinates[0], jnp.asarray(
            [coordinates[0] - 1.0], dtype=coordinates.dtype
        )

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(
        joint,
        initial,
        options=FilterTrustRegionSQPOptions(maximum_iterations=1),
    ).run(initial)

    assert result.accepted_iterations == 1
    assert result.history.accepted[0] == 1
    assert result.history.filter_accepted[0] == 1
    assert result.history.objective_type[0] == 0
    assert result.history.normal_step_norm[0] > 0.0
    assert result.history.tangency_relative_residual[0] <= 1.0e-10
    assert jnp.abs(result.constraints[0]) < 1.0


def test_trust_radius_bounds_combined_step() -> None:
    initial = jnp.asarray([3.0, -2.0], dtype=jnp.float64)
    radius = 0.125
    result = prepare_filter_trust_region_sqp(
        _quadratic_equality,
        initial,
        options=FilterTrustRegionSQPOptions(
            maximum_iterations=1,
            initial_trust_radius=radius,
            maximum_trust_radius=radius,
        ),
    ).run(initial)

    assert result.accepted_iterations == 1
    assert result.history.combined_step_norm[0] <= radius * (1.0 + 1.0e-12)
    assert result.history.normal_step_norm[0] <= 0.8 * radius * (1.0 + 1.0e-12)


def test_accepted_low_ratio_step_shrinks_radius() -> None:
    def mismatched_quadratic(
        coordinates: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        objective = 0.5 * 1.8 * coordinates[0] ** 2
        return objective, jnp.asarray([coordinates[1]], dtype=coordinates.dtype)

    initial = jnp.asarray([0.1, 0.0], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(
        mismatched_quadratic,
        initial,
        options=FilterTrustRegionSQPOptions(maximum_iterations=1),
    ).run(initial)

    assert result.accepted_iterations == 1
    assert 0.1 <= result.history.reduction_ratio[0] < 0.25
    assert result.radius == pytest.approx(0.25)


def test_nonlinear_constraint_projects_dual_at_accepted_state() -> None:
    def circle_joint(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = 0.5 * ((coordinates[0] - 1.0) ** 2 + (coordinates[1] - 0.25) ** 2)
        constraints = jnp.asarray(
            [coordinates[0] ** 2 + coordinates[1] ** 2 - 1.0],
            dtype=coordinates.dtype,
        )
        return objective, constraints

    initial = jnp.asarray([0.8, 0.8], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(
        circle_joint,
        initial,
        options=FilterTrustRegionSQPOptions(maximum_iterations=6),
    ).run(initial)

    assert result.accepted_iterations >= 1
    expected_stationarity = (
        result.objective_gradient + result.constraint_jacobian.T @ result.multipliers
    )
    np.testing.assert_allclose(result.stationarity, expected_stationarity, atol=1.0e-17)
    assert result.final_multiplier_projection_relative_residual <= 1.0e-10
    assert result.all_accepted_states_finite


def test_rank_deficient_gram_fails_closed() -> None:
    def dependent_constraints(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        constraint = coordinates[0] + coordinates[1] - 1.0
        return jnp.dot(coordinates, coordinates), jnp.asarray(
            [constraint, 2.0 * constraint], dtype=coordinates.dtype
        )

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(
        dependent_constraints,
        initial,
        options=FilterTrustRegionSQPOptions(maximum_iterations=2),
    ).run(initial)

    assert result.status == int(FilterTrustRegionSQPStatus.RANK_DEFICIENT_GRAM)
    assert result.fatal
    assert result.accepted_iterations == 0
    assert result.failure_counters.factor == 1
    np.testing.assert_array_equal(result.optimizer_coordinates, initial)


def test_zero_step_convergence_still_certifies_gram_and_projection() -> None:
    initial = jnp.asarray([0.5, 0.5], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(_quadratic_equality, initial).run(
        initial, jnp.asarray([-0.5], dtype=jnp.float64)
    )

    assert result.converged
    assert result.iterations == 0
    assert result.derivative_builds == 1
    assert result.final_normal_relative_residual <= 1.0e-10
    assert result.final_normal_forward_error_bound < 1.0e-7
    assert result.final_multiplier_projection_relative_residual <= 1.0e-10
    assert result.final_multiplier_projection_forward_error_bound < 1.0e-7


def test_zero_step_rank_deficiency_cannot_report_convergence() -> None:
    def flat_dependent(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        zero = coordinates[0] - coordinates[0]
        return zero, jnp.asarray([zero, 2.0 * zero], dtype=coordinates.dtype)

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(flat_dependent, initial).run(initial)

    assert not result.converged
    assert result.status == int(FilterTrustRegionSQPStatus.RANK_DEFICIENT_GRAM)
    assert result.failure_counters.factor == 1


def test_nonfinite_candidates_are_rejected_and_counted() -> None:
    def finite_only_at_initial(
        coordinates: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        moved = jnp.any(coordinates != 0.0)
        objective = jnp.where(moved, jnp.nan, 0.0)
        constraints = jnp.asarray([coordinates[0] - 1.0], dtype=coordinates.dtype)
        return objective, constraints

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    result = prepare_filter_trust_region_sqp(
        finite_only_at_initial,
        initial,
        options=FilterTrustRegionSQPOptions(maximum_iterations=1),
    ).run(initial)

    assert result.accepted_iterations == 0
    assert result.failure_counters.nonfinite == 1
    assert result.status == int(FilterTrustRegionSQPStatus.ITERATION_LIMIT)
    np.testing.assert_array_equal(result.optimizer_coordinates, initial)


def test_history_shapes_and_exact_logical_evaluation_accounting() -> None:
    maximum_iterations = 3
    initial = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    prepared = prepare_filter_trust_region_sqp(
        _quadratic_equality,
        initial,
        options=FilterTrustRegionSQPOptions(maximum_iterations=maximum_iterations),
    )
    result = prepared.run(initial)

    for history_array in result.history:
        assert history_array.shape == (maximum_iterations,)
    assert result.joint_evaluations == (
        1 + result.iterations + result.accepted_iterations
    )
    assert result.derivative_builds == 1 + result.accepted_iterations
    compiled_text = (
        prepared._run_prepared.runtime_executable().hlo_modules()[0].to_string()
    )
    assert "host_callback" not in compiled_text
    assert "io_callback" not in compiled_text


@pytest.mark.parametrize(
    "options",
    (
        FilterTrustRegionSQPOptions(maximum_iterations=0),
        FilterTrustRegionSQPOptions(initial_trust_radius=0.0),
        FilterTrustRegionSQPOptions(normal_radius_fraction=1.0),
        FilterTrustRegionSQPOptions(acceptance_ratio=0.9, expansion_ratio=0.8),
        FilterTrustRegionSQPOptions(maximum_tangential_cg_iterations=0),
    ),
)
def test_invalid_options_fail_before_compilation(
    options: FilterTrustRegionSQPOptions,
) -> None:
    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    with pytest.raises(ValueError):
        prepare_filter_trust_region_sqp(_quadratic_equality, initial, options=options)
