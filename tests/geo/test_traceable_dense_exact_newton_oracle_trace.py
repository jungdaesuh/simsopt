from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers import optimizer as _optimizer


def _affine_residual(values: jax.Array) -> jax.Array:
    matrix = jnp.asarray([[1.5, -0.25], [0.75, 2.0]], dtype=values.dtype)
    target = jnp.asarray([0.5, -1.25], dtype=values.dtype)
    return matrix @ values - target


def _identity_residual(values: jax.Array) -> jax.Array:
    return values


def _cubic_residual(values: jax.Array) -> jax.Array:
    return values**3 - 1.0


def test_c0_oracle_replay_is_separate_fixed_shape_and_source_equivalent() -> None:
    maxiter = 3
    tol = 1.0e-12
    initial = jax.device_put(np.asarray([0.25, -0.5], dtype=np.float64))
    production_runner = _optimizer._make_traceable_exact_newton_runner(
        _affine_residual,
        maxiter,
        tol,
        False,
    )
    oracle_runner = _optimizer._make_traceable_exact_newton_c0_oracle_runner(
        _affine_residual,
        maxiter,
        tol,
    )

    assert production_runner is not oracle_runner
    with jax.transfer_guard("disallow"):
        production = production_runner(initial, ())
        oracle = oracle_runner(initial, ())
        jax.block_until_ready((production, oracle))

    assert isinstance(production, dict)
    assert set(production) == {
        "x",
        "residual",
        "nit",
        "success",
        "exact_newton_linear_residual_rel",
        "exact_refinement_correction_rel",
    }
    assert isinstance(oracle, _optimizer._ExactNewtonC0OracleResult)
    np.testing.assert_array_equal(np.asarray(oracle.state), np.asarray(production["x"]))
    np.testing.assert_array_equal(
        np.asarray(oracle.residual),
        np.asarray(production["residual"]),
    )
    assert int(oracle.nit) == int(production["nit"])
    assert bool(oracle.success) == bool(production["success"])

    trace = oracle.trace
    active = np.asarray(trace.active)
    attempts = int(oracle.linear_solve_attempt_count)
    np.testing.assert_array_equal(active, np.arange(2 * maxiter) < attempts)
    assert trace.state_before.shape == (2 * maxiter, 2)
    assert trace.update.shape == trace.state_before.shape
    assert trace.state_after.shape == trace.state_before.shape
    for index in range(attempts):
        before = np.asarray(trace.state_before[index])
        update = np.asarray(trace.update[index])
        after = np.asarray(trace.state_after[index])
        np.testing.assert_allclose(after, before - update, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            float(trace.merit_before[index]),
            np.linalg.norm(np.asarray(_affine_residual(jnp.asarray(before)))),
            rtol=0.0,
            atol=1.0e-15,
        )
        if bool(trace.merit_after_assessed[index]):
            np.testing.assert_allclose(
                float(trace.merit_after[index]),
                np.linalg.norm(np.asarray(_affine_residual(jnp.asarray(after)))),
                rtol=0.0,
                atol=1.0e-15,
            )
        if index:
            np.testing.assert_array_equal(
                np.asarray(trace.state_before[index]),
                np.asarray(trace.state_after[index - 1]),
            )

    assert int(oracle.accepted_update_count) == int(np.sum(trace.accepted[active]))
    assert int(oracle.residual_evaluation_count) == 1 + int(
        np.sum(np.asarray(trace.backtracking_iterations)[active])
    )
    last = attempts - 1
    assert int(trace.residual_evaluation_count[last]) == int(
        oracle.residual_evaluation_count
    )
    assert int(trace.linear_solve_attempt_count[last]) == attempts
    assert int(trace.accepted_update_count[last]) == int(oracle.accepted_update_count)
    expected_jacobian = np.asarray(
        [[1.5, -0.25], [0.75, 2.0]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(np.asarray(oracle.jacobian), expected_jacobian)
    np.testing.assert_allclose(
        float(oracle.norm),
        np.linalg.norm(np.asarray(oracle.residual)),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_c0_oracle_converged_initial_state_has_inactive_fixed_trace() -> None:
    runner = _optimizer._make_traceable_exact_newton_c0_oracle_runner(
        _identity_residual,
        0,
        1.0e-12,
    )
    initial = jax.device_put(np.zeros(2, dtype=np.float64))

    with jax.transfer_guard("disallow"):
        result = runner(initial, ())
        jax.block_until_ready(result)

    assert bool(result.success)
    assert int(result.linear_solve_attempt_count) == 0
    assert int(result.residual_evaluation_count) == 1
    np.testing.assert_array_equal(np.asarray(result.trace.active), [False])
    assert result.trace.state_before.shape == (1, 2)
    assert result.jacobian.shape == (2, 2)


def test_c2_oracle_exposes_existing_raw_trace_without_changing_production() -> None:
    maxiter = 2
    tol = 1.0e-12
    initial = jax.device_put(np.asarray([0.25, -0.5], dtype=np.float64))
    production_runner = _optimizer._make_traceable_dense_direct_exact_newton_c2_runner(
        _affine_residual,
        maxiter,
        tol,
    )
    oracle_runner = (
        _optimizer._make_traceable_dense_direct_exact_newton_c2_oracle_runner(
            _affine_residual,
            maxiter,
            tol,
        )
    )

    assert production_runner is not oracle_runner
    with jax.transfer_guard("disallow"):
        production = production_runner(initial, ())
        oracle = oracle_runner(initial, ())
        jax.block_until_ready((production, oracle))

    assert isinstance(production, _optimizer._NativeDenseExactNewtonC2Result)
    assert isinstance(oracle, _optimizer._DenseExactNewtonC2OracleResult)
    for production_leaf, oracle_leaf in zip(
        jax.tree.leaves(production),
        jax.tree.leaves(oracle.native),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(oracle_leaf),
            np.asarray(production_leaf),
        )
    native = oracle.native
    step = oracle.first_attempt
    assert bool(step.active)
    step_state = np.asarray(step.state)
    step_residual = np.asarray(step.residual)
    step_jacobian = np.asarray(step.jacobian)
    initial_solve = np.asarray(step.initial_solve)
    refinement_rhs = np.asarray(step.refinement_rhs)
    refinement_correction = np.asarray(step.refinement_correction)
    refined_direction = np.asarray(step.refined_direction)
    np.testing.assert_allclose(
        refinement_rhs,
        step_residual - step_jacobian @ initial_solve,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        refined_direction,
        initial_solve + refinement_correction,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(step.refined_residual),
        step_residual - step_jacobian @ refined_direction,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(np.asarray(step.correction_step), refined_direction)
    np.testing.assert_allclose(
        np.asarray(step.next_state),
        step_state - np.asarray(step.correction_step),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert native.applied_state_trace.shape == (maxiter + 1, 2)
    assert native.applied_state_trace_active.shape == (maxiter + 1,)
    assert native.assessed_norm_trace.shape == (maxiter + 1,)
    assert native.assessed_norm_trace_active.shape == (maxiter + 1,)
    assert int(native.applied_update_count) == int(native.linear_solve_attempt_count)
    assert bool(native.persist_solved_state) != bool(native.rollback_branch_taken)
    materializations = int(native.dense_materialization_count)
    assert int(oracle.exact_newton_variant_residual_evaluation_count) == (
        materializations
    )
    assert int(oracle.exact_newton_variant_dense_primal_traversal_count) == (
        materializations
    )
    assert int(oracle.exact_newton_variant_dense_tangent_batch_count) == (
        materializations
    )
    assert int(oracle.exact_newton_variant_dense_tangent_direction_count) == (
        2 * materializations
    )


def test_c2_oracle_counts_rollback_materialization_from_source_telemetry() -> None:
    runner = _optimizer._make_traceable_dense_direct_exact_newton_c2_oracle_runner(
        _cubic_residual,
        2,
        1.0e-12,
    )
    initial = jax.device_put(np.asarray([0.1], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        oracle = runner(initial, ())
        jax.block_until_ready(oracle)

    assert bool(oracle.native.rollback_branch_taken)
    assert int(oracle.native.rollback_recompute_count) == 1
    assert int(oracle.native.dense_materialization_count) == 4
    assert int(oracle.exact_newton_variant_residual_evaluation_count) == 4
    assert int(oracle.exact_newton_variant_dense_primal_traversal_count) == 4
    assert int(oracle.exact_newton_variant_dense_tangent_batch_count) == 4
    assert int(oracle.exact_newton_variant_dense_tangent_direction_count) == 4


def test_c2_oracle_first_attempt_is_fixed_shape_and_inactive_when_converged() -> None:
    runner = _optimizer._make_traceable_dense_direct_exact_newton_c2_oracle_runner(
        _identity_residual,
        2,
        1.0e-12,
    )
    initial = jax.device_put(np.zeros(2, dtype=np.float64))

    with jax.transfer_guard("disallow"):
        oracle = runner(initial, ())
        jax.block_until_ready(oracle)

    assert not bool(oracle.first_attempt.active)
    assert oracle.first_attempt.state.shape == (2,)
    assert oracle.first_attempt.jacobian.shape == (2, 2)
    assert np.all(np.isnan(np.asarray(oracle.first_attempt.state)))
