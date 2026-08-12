from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jax_core
from simsopt_jax.geo.optimizers import optimizer as _optimizer
from simsopt_jax.runtime.trace_annotations import PhaseId, trace_session


def _quadratic_residual(values: jax.Array) -> jax.Array:
    return jnp.asarray(
        [
            values[0] ** 2 + 0.5 * values[1] - 2.0,
            -0.25 * values[0] + values[1] ** 2 - 3.0,
        ],
        dtype=values.dtype,
    )


def _affine_residual(values: jax.Array) -> jax.Array:
    matrix = jnp.asarray([[1.5, -0.25], [0.75, 2.0]], dtype=values.dtype)
    target = jnp.asarray([0.5, -1.25], dtype=values.dtype)
    return matrix @ values - target


def _identity_residual(values: jax.Array) -> jax.Array:
    return values


def _cubic_residual(values: jax.Array) -> jax.Array:
    return values**3 - 1.0


def _jaxpr_name_stacks(value: object) -> list[str]:
    names: list[str] = []
    if isinstance(value, jax_core.ClosedJaxpr):
        return _jaxpr_name_stacks(value.jaxpr)
    if isinstance(value, jax_core.Jaxpr):
        for equation in value.eqns:
            name = str(equation.source_info.name_stack)
            if name:
                names.append(name)
            for parameter in equation.params.values():
                names.extend(_jaxpr_name_stacks(parameter))
        return names
    if isinstance(value, dict):
        for item in value.values():
            names.extend(_jaxpr_name_stacks(item))
        return names
    if isinstance(value, (tuple, list)):
        for item in value:
            names.extend(_jaxpr_name_stacks(item))
    return names


def test_dense_direction_scopes_are_opt_in_disjoint_and_numerically_inert() -> None:
    initial = jnp.asarray([1.25, 1.75], dtype=jnp.float64)

    def unannotated_run(values: jax.Array) -> _optimizer._DenseExactNewtonDirection:
        return _optimizer._dense_direct_exact_newton_direction(
            _quadratic_residual,
            values,
            tol=1.0e-12,
        )

    def annotated_run(values: jax.Array) -> _optimizer._DenseExactNewtonDirection:
        return _optimizer._dense_direct_exact_newton_direction(
            _quadratic_residual,
            values,
            tol=1.0e-12,
        )

    unannotated_result = unannotated_run(initial)
    unannotated_jaxpr = jax.make_jaxpr(unannotated_run)(initial)
    with trace_session():
        annotated_result = annotated_run(initial)
        annotated_jaxpr = jax.make_jaxpr(annotated_run)(initial)

    for unannotated_leaf, annotated_leaf in zip(
        jax.tree.leaves(unannotated_result),
        jax.tree.leaves(annotated_result),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(annotated_leaf),
            np.asarray(unannotated_leaf),
        )

    phases = (
        PhaseId.NEWTON_JACOBIAN_CONSTRUCTION,
        PhaseId.NEWTON_DENSE_MATERIALIZATION,
        PhaseId.NEWTON_LU_FACTOR,
        PhaseId.NEWTON_REFINEMENT,
    )
    unannotated_names = _jaxpr_name_stacks(unannotated_jaxpr)
    annotated_names = _jaxpr_name_stacks(annotated_jaxpr)
    assert all(
        phase.value not in name for phase in phases for name in unannotated_names
    )
    assert all(any(phase.value in name for name in annotated_names) for phase in phases)
    assert all(
        not (
            PhaseId.NEWTON_JACOBIAN_CONSTRUCTION.value in name
            and PhaseId.NEWTON_DENSE_MATERIALIZATION.value in name
        )
        for name in annotated_names
    )
    assert all(
        not (
            PhaseId.NEWTON_LU_FACTOR.value in name
            and PhaseId.NEWTON_REFINEMENT.value in name
        )
        for name in annotated_names
    )


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


def test_c1_oracle_is_separately_cached_and_preserves_production_result() -> None:
    maxiter = 8
    tol = 1.0e-12
    initial = jax.device_put(np.asarray([1.25, 1.75], dtype=np.float64))
    production_runner = _optimizer._make_traceable_dense_direct_exact_newton_c1_runner(
        _quadratic_residual,
        maxiter,
        tol,
    )
    oracle_runner = (
        _optimizer._make_traceable_dense_direct_exact_newton_c1_oracle_runner(
            _quadratic_residual,
            maxiter,
            tol,
        )
    )

    assert production_runner is not oracle_runner
    assert production_runner is (
        _optimizer._make_traceable_dense_direct_exact_newton_c1_runner(
            _quadratic_residual,
            maxiter,
            tol,
        )
    )
    assert oracle_runner is (
        _optimizer._make_traceable_dense_direct_exact_newton_c1_oracle_runner(
            _quadratic_residual,
            maxiter,
            tol,
        )
    )

    with jax.transfer_guard("disallow"):
        production = production_runner(initial, ())
        oracle = oracle_runner(initial, ())
        jax.block_until_ready((production, oracle))

    assert isinstance(production, dict)
    assert "trace" not in production
    assert isinstance(oracle, _optimizer._DenseExactNewtonC1OracleResult)
    for field in oracle._fields:
        if (
            field != "trace"
            and not field.startswith("exact_newton_variant_dense_")
            and field != "exact_newton_variant_residual_evaluation_count"
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(oracle, field)),
                np.asarray(production[field]),
            )


def test_c1_oracle_trace_matches_active_attempt_equations_and_counters() -> None:
    maxiter = 8
    initial = jax.device_put(np.asarray([1.25, 1.75], dtype=np.float64))
    runner = _optimizer._make_traceable_dense_direct_exact_newton_c1_oracle_runner(
        _quadratic_residual,
        maxiter,
        1.0e-12,
    )

    with jax.transfer_guard("disallow"):
        result = runner(initial, ())
        jax.block_until_ready(result)

    trace = result.trace
    active = np.asarray(trace.active)
    attempts = int(result.linear_solve_attempt_count)
    source_first = _optimizer._dense_direct_exact_newton_direction(
        _quadratic_residual,
        initial,
        tol=1.0e-12,
    )
    for traced, source in (
        (trace.jacobian[0], source_first.jacobian),
        (trace.initial_solve[0], source_first.initial_solve),
        (trace.refinement_rhs[0], source_first.refinement_rhs),
        (trace.refinement_correction[0], source_first.correction),
        (trace.refined_direction[0], source_first.direction),
        (trace.refined_residual[0], source_first.linear_residual),
    ):
        np.testing.assert_array_equal(np.asarray(traced), np.asarray(source))
    np.testing.assert_array_equal(
        active,
        np.arange(2 * maxiter) < attempts,
    )
    assert trace.state.shape == (2 * maxiter, 2)
    assert trace.residual.shape == trace.state.shape
    assert trace.refined_direction.shape == trace.state.shape
    assert trace.refinement_correction.shape == trace.state.shape
    assert trace.next_state.shape == trace.state.shape
    assert trace.next_residual.shape == trace.state.shape

    states = np.asarray(trace.state)[active]
    residuals = np.asarray(trace.residual)[active]
    jacobians = np.asarray(trace.jacobian)[active]
    norms = np.asarray(trace.norm)[active]
    initial_solves = np.asarray(trace.initial_solve)[active]
    refinement_rhs = np.asarray(trace.refinement_rhs)[active]
    directions = np.asarray(trace.refined_direction)[active]
    refinement_corrections = np.asarray(trace.refinement_correction)[active]
    refined_residuals = np.asarray(trace.refined_residual)[active]
    correction_steps = np.asarray(trace.correction_step)[active]
    next_states = np.asarray(trace.next_state)[active]
    next_residuals = np.asarray(trace.next_residual)[active]
    next_norms = np.asarray(trace.next_norm)[active]
    alphas = np.asarray(trace.backtracking_alpha)[active]
    for index in range(attempts):
        expected_residual = np.asarray(_quadratic_residual(jnp.asarray(states[index])))
        expected_next_residual = np.asarray(
            _quadratic_residual(jnp.asarray(next_states[index]))
        )
        np.testing.assert_allclose(
            residuals[index],
            expected_residual,
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            refinement_rhs[index],
            residuals[index] - jacobians[index] @ initial_solves[index],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            directions[index],
            initial_solves[index] + refinement_corrections[index],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            refined_residuals[index],
            residuals[index] - jacobians[index] @ directions[index],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            correction_steps[index],
            states[index] - next_states[index],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(norms[index], np.linalg.norm(expected_residual))
        np.testing.assert_allclose(
            next_states[index],
            states[index] - alphas[index] * directions[index],
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            correction_steps[index],
            alphas[index] * directions[index],
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            next_residuals[index],
            expected_next_residual,
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            next_norms[index],
            np.linalg.norm(expected_next_residual),
            rtol=1.0e-15,
            atol=1.0e-15,
        )

    assert np.all(np.asarray(trace.accepted)[active])
    assert np.all(np.asarray(trace.linear_success)[active])
    last = attempts - 1
    assert int(trace.dense_materialization_count[last]) == int(
        result.dense_materialization_count
    )
    assert int(trace.lu_factorization_count[last]) == int(result.lu_factorization_count)
    assert int(trace.lu_solve_count[last]) == int(result.lu_solve_count)
    assert int(trace.refinement_correction_count[last]) == int(
        result.refinement_correction_count
    )
    assert int(trace.backtracking_iteration_count[last]) == int(
        result.backtracking_iteration_count
    )
    assert int(np.sum(np.asarray(trace.backtracking_iterations)[active])) == int(
        result.backtracking_iteration_count
    )
    assert int(result.exact_newton_variant_residual_evaluation_count) == (
        1 + attempts + int(result.backtracking_iteration_count)
    )
    assert int(result.exact_newton_variant_dense_primal_traversal_count) == attempts
    assert int(result.exact_newton_variant_dense_tangent_batch_count) == attempts
    assert int(result.exact_newton_variant_dense_tangent_direction_count) == (
        2 * attempts
    )
    assert int(trace.residual_evaluation_count[last]) == int(
        result.exact_newton_variant_residual_evaluation_count
    )
    assert int(trace.dense_primal_traversal_count[last]) == int(
        result.exact_newton_variant_dense_primal_traversal_count
    )
    assert int(trace.dense_tangent_batch_count[last]) == int(
        result.exact_newton_variant_dense_tangent_batch_count
    )
    assert int(trace.dense_tangent_direction_count[last]) == int(
        result.exact_newton_variant_dense_tangent_direction_count
    )


def test_c1_oracle_uses_one_inactive_slot_when_no_attempt_is_possible() -> None:
    initial = jax.device_put(np.zeros(2, dtype=np.float64))
    runner = _optimizer._make_traceable_dense_direct_exact_newton_c1_oracle_runner(
        _identity_residual,
        0,
        1.0e-12,
    )

    with jax.transfer_guard("disallow"):
        result = runner(initial, ())
        jax.block_until_ready(result)

    assert bool(result.success)
    assert int(result.linear_solve_attempt_count) == 0
    assert int(result.exact_newton_variant_residual_evaluation_count) == 1
    np.testing.assert_array_equal(np.asarray(result.trace.active), [False])
    assert result.trace.state.shape == (1, 2)
    assert result.trace.jacobian.shape == (1, 2, 2)


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
