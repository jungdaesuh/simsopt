from __future__ import annotations

from collections.abc import Callable
from weakref import ref

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.boozer_residual import boozer_residual_vector
from simsopt_jax.geo.optimizers import optimizer as _optimizer
from simsopt_jax.runtime.jaxpr_closure import (
    closure_converted_array_function,
    device_put_closure_consts,
)


def _native_exact_newton_oracle(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    *,
    maxiter: int,
    tol: float,
) -> dict[str, object]:
    initial_x = initial.copy()
    x = initial.copy()
    residual = residual_fn(x)
    jacobian = jacobian_fn(x)
    initial_norm: float | None = None
    assessed_norm = 1.0e6
    applied_states = [x.copy()]
    assessed_norms: list[float] = []
    iteration = 0
    while iteration < maxiter:
        assessed_norm = float(np.linalg.norm(residual))
        assessed_norms.append(assessed_norm)
        if initial_norm is None:
            initial_norm = assessed_norm
        if assessed_norm <= tol:
            break
        direction = np.linalg.solve(jacobian, residual)
        direction += np.linalg.solve(
            jacobian,
            residual - jacobian @ direction,
        )
        x -= direction
        iteration += 1
        applied_states.append(x.copy())
        residual = residual_fn(x)
        jacobian = jacobian_fn(x)

    success = assessed_norm <= tol
    persist = success or (
        initial_norm is not None
        and np.isfinite(assessed_norm)
        and assessed_norm <= initial_norm
    )
    returned_x = x if persist else initial_x
    return {
        "x": returned_x,
        "residual": residual_fn(returned_x),
        "iteration": iteration,
        "success": success,
        "persist": persist,
        "rollback_branch_taken": not persist,
        "initial_norm": np.nan if initial_norm is None else initial_norm,
        "assessed_norm": assessed_norm,
        "applied_states": np.stack(applied_states),
        "assessed_norms": np.asarray(assessed_norms),
    }


def _affine_residual(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray([[1.5, -0.25], [0.75, 2.0]], dtype=np.float64)
    target = np.asarray([0.5, -1.25], dtype=np.float64)
    return matrix @ values - target


def _affine_jacobian(_values: np.ndarray) -> np.ndarray:
    return np.asarray([[1.5, -0.25], [0.75, 2.0]], dtype=np.float64)


def test_c2_one_step_boozer_oracle_preserves_native_maxiter_stop_order() -> None:
    magnetic_field_host = np.asarray([[[1.2, -0.4, 0.7]]], dtype=np.float64)
    xphi_host = np.asarray([[[0.3, -0.5, 0.1]]], dtype=np.float64)
    xtheta_host = np.asarray([[[-0.2, 0.8, 0.4]]], dtype=np.float64)
    field_squared = np.sum(magnetic_field_host**2)
    native_jacobian = np.column_stack(
        (
            -field_squared * xtheta_host.ravel()[:2],
            magnetic_field_host.ravel()[:2],
        )
    )

    def native_residual(values: np.ndarray) -> np.ndarray:
        iota, current_parameter = values
        return current_parameter * magnetic_field_host.ravel()[:2] - (
            field_squared * (xphi_host.ravel()[:2] + iota * xtheta_host.ravel()[:2])
        )

    magnetic_field = jax.device_put(magnetic_field_host)
    xphi = jax.device_put(xphi_host)
    xtheta = jax.device_put(xtheta_host)

    def residual(values: jax.Array) -> jax.Array:
        iota, current_parameter = values
        return boozer_residual_vector(
            current_parameter,
            iota,
            magnetic_field,
            xphi,
            xtheta,
        )[:2]

    initial_host = np.asarray([0.15, 0.9], dtype=np.float64)
    initial = jax.device_put(initial_host)
    expected = _native_exact_newton_oracle(
        native_residual,
        lambda _values: native_jacobian,
        initial_host,
        maxiter=1,
        tol=1.0e-12,
    )
    converted_residual, residual_consts = closure_converted_array_function(
        residual,
        initial,
    )
    residual_consts = device_put_closure_consts(residual_consts, initial)

    def residual_with_explicit_closure(
        values: jax.Array,
        *current_consts: jax.Array,
    ) -> jax.Array:
        return converted_residual(values, current_consts)

    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual_with_explicit_closure),
        1,
        1.0e-12,
    )

    with jax.transfer_guard("disallow"):
        actual = runner(initial, residual_consts)
        jax.block_until_ready(actual)

    np.testing.assert_allclose(
        np.asarray(actual.x),
        expected["x"],
        rtol=0.0,
        atol=2.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        np.asarray(actual.residual),
        expected["residual"],
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        np.asarray(actual.returned_jacobian),
        native_jacobian,
        rtol=0.0,
        atol=2.0 * np.finfo(np.float64).eps,
    )
    assert int(actual.iteration_count) == expected["iteration"] == 1
    assert int(actual.applied_update_count) == 1
    assert not bool(actual.success)
    assert bool(actual.persist_solved_state)
    assert bool(actual.native_persist_predicate)
    assert not bool(actual.rollback_branch_taken)
    assert int(actual.stop_reason_code) == _optimizer._C2_STOP_REASON_MAXITER
    assert float(actual.assessed_norm) == expected["assessed_norm"]
    assert float(actual.returned_norm) <= 4.0 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        np.asarray(actual.applied_state_trace)[:2],
        expected["applied_states"],
        rtol=0.0,
        atol=2.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.applied_state_trace_active),
        np.asarray([True, True]),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.assessed_norm_trace_active),
        np.asarray([True, False]),
    )
    assert int(actual.linear_solve_attempt_count) == 1
    assert int(actual.dense_materialization_count) == 2
    assert int(actual.lu_factorization_count) == 1
    assert int(actual.lu_solve_count) == 12
    assert int(actual.refinement_correction_count) == 1
    assert int(actual.rollback_recompute_count) == 0
    assert "jacobian" not in actual._fields
    assert "lu" not in actual._fields
    assert "pivots" not in actual._fields


def test_c2_one_slot_persists_a_finite_worsened_final_native_update() -> None:
    def native_residual(values: np.ndarray) -> np.ndarray:
        return values**3 - 1.0

    def native_jacobian(values: np.ndarray) -> np.ndarray:
        return np.diag(3.0 * values**2)

    def residual(values: jax.Array) -> jax.Array:
        return values**3 - 1.0

    initial = np.asarray([0.1], dtype=np.float64)
    expected = _native_exact_newton_oracle(
        native_residual,
        native_jacobian,
        initial,
        maxiter=1,
        tol=1.0e-12,
    )
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual),
        1,
        1.0e-12,
    )

    actual = runner(jnp.asarray(initial), ())

    assert int(actual.iteration_count) == 1
    assert int(actual.applied_update_count) == 1
    assert not bool(actual.success)
    assert not bool(actual.numerical_failure)
    assert bool(actual.native_persist_predicate)
    assert bool(actual.persist_solved_state)
    assert not bool(actual.rollback_branch_taken)
    assert float(actual.returned_norm) > float(actual.initial_norm)
    np.testing.assert_allclose(
        np.asarray(actual.x),
        expected["x"],
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(actual.returned_jacobian),
        native_jacobian(np.asarray(actual.x)),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert int(actual.dense_materialization_count) == 2
    assert int(actual.rollback_recompute_count) == 0


def test_c2_assesses_converged_updated_state_in_the_next_native_loop_slot() -> None:
    initial = np.asarray([1.25, 0.5], dtype=np.float64)
    expected = _native_exact_newton_oracle(
        _affine_residual,
        _affine_jacobian,
        initial,
        maxiter=2,
        tol=1.0e-12,
    )

    def residual(values: jax.Array) -> jax.Array:
        matrix = jnp.asarray(_affine_jacobian(initial), dtype=values.dtype)
        target = jnp.asarray([0.5, -1.25], dtype=values.dtype)
        return matrix @ values - target

    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual),
        2,
        1.0e-12,
    )
    actual = runner(jnp.asarray(initial), ())

    assert int(actual.iteration_count) == expected["iteration"] == 1
    assert int(actual.applied_update_count) == 1
    assert bool(actual.success)
    assert bool(actual.persist_solved_state)
    assert int(actual.stop_reason_code) == _optimizer._C2_STOP_REASON_CONVERGED
    np.testing.assert_allclose(np.asarray(actual.x), expected["x"], atol=1.0e-15)
    np.testing.assert_allclose(
        np.asarray(actual.assessed_norm_trace)[:2],
        expected["assessed_norms"],
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.assessed_norm_trace_active),
        np.asarray([True, True, False]),
    )
    assert int(actual.linear_solve_attempt_count) == 1
    assert int(actual.dense_materialization_count) == 2
    assert int(actual.lu_factorization_count) == 1
    assert int(actual.rollback_recompute_count) == 0


def test_c2_rolls_back_after_native_assessed_norm_worsens() -> None:
    def native_residual(values: np.ndarray) -> np.ndarray:
        return values**3 - 1.0

    def native_jacobian(values: np.ndarray) -> np.ndarray:
        return np.diag(3.0 * values**2)

    def residual(values: jax.Array) -> jax.Array:
        return values**3 - 1.0

    initial = np.asarray([0.1], dtype=np.float64)
    expected = _native_exact_newton_oracle(
        native_residual,
        native_jacobian,
        initial,
        maxiter=2,
        tol=1.0e-12,
    )
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual),
        2,
        1.0e-12,
    )

    actual = runner(jnp.asarray(initial), ())

    assert int(actual.iteration_count) == expected["iteration"] == 2
    assert int(actual.applied_update_count) == 2
    assert not bool(actual.success)
    assert not bool(actual.persist_solved_state)
    assert bool(actual.rollback_branch_taken)
    assert int(actual.stop_reason_code) == _optimizer._C2_STOP_REASON_MAXITER
    np.testing.assert_array_equal(np.asarray(actual.x), initial)
    np.testing.assert_array_equal(
        np.asarray(actual.residual),
        native_residual(initial),
    )
    np.testing.assert_allclose(
        np.asarray(actual.returned_jacobian),
        native_jacobian(initial),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        np.asarray(actual.applied_state_trace)[:3],
        expected["applied_states"],
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(actual.assessed_norm_trace)[:2],
        expected["assessed_norms"],
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert float(actual.assessed_norm) > float(actual.initial_norm)
    assert int(actual.linear_solve_attempt_count) == 2
    assert int(actual.dense_materialization_count) == 4
    assert int(actual.lu_factorization_count) == 2
    assert int(actual.lu_solve_count) == 24
    assert int(actual.refinement_correction_count) == 2
    assert int(actual.rollback_recompute_count) == 1


def test_c2_singular_jacobian_fails_closed_without_an_applied_state() -> None:
    def residual(values: jax.Array) -> jax.Array:
        total = values[0] + values[1] - 1.0
        return jnp.asarray([total, 2.0 * total], dtype=values.dtype)

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual),
        3,
        1.0e-12,
    )

    actual = runner(initial, ())

    assert not bool(actual.success)
    assert bool(actual.numerical_failure)
    assert not bool(actual.persist_solved_state)
    assert bool(actual.rollback_branch_taken)
    assert int(actual.stop_reason_code) == (
        _optimizer._C2_STOP_REASON_NUMERICAL_FAILURE
    )
    assert int(actual.iteration_count) == 0
    assert int(actual.applied_update_count) == 0
    assert int(actual.linear_solve_attempt_count) == 1
    assert int(actual.dense_materialization_count) == 2
    assert int(actual.rollback_recompute_count) == 1
    np.testing.assert_array_equal(np.asarray(actual.x), np.asarray(initial))
    np.testing.assert_allclose(
        np.asarray(actual.returned_jacobian),
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_c2_nonfinite_applied_state_is_traced_then_rolled_back_fail_closed() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return jnp.sqrt(values)

    initial = jnp.asarray([1.0], dtype=jnp.float64)
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual),
        1,
        1.0e-12,
    )

    actual = runner(initial, ())

    assert bool(actual.numerical_failure)
    assert bool(actual.native_persist_predicate)
    assert not bool(actual.persist_solved_state)
    assert bool(actual.rollback_branch_taken)
    assert int(actual.stop_reason_code) == (
        _optimizer._C2_STOP_REASON_NUMERICAL_FAILURE
    )
    assert int(actual.iteration_count) == 1
    assert int(actual.applied_update_count) == 1
    np.testing.assert_array_equal(np.asarray(actual.x), np.asarray(initial))
    np.testing.assert_allclose(
        np.asarray(actual.returned_jacobian),
        np.asarray([[0.5]]),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.applied_state_trace_active),
        np.asarray([True, True]),
    )
    np.testing.assert_allclose(
        np.asarray(actual.applied_state_trace[1]),
        np.asarray([-1.0]),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    assert int(actual.linear_solve_attempt_count) == 1
    assert int(actual.dense_materialization_count) == 3
    assert int(actual.rollback_recompute_count) == 1


def test_c2_initially_converged_state_stops_without_factoring() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return values - jnp.asarray([1.0, -2.0], dtype=values.dtype)

    initial = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual),
        3,
        1.0e-12,
    )

    actual = runner(initial, ())

    assert bool(actual.success)
    assert bool(actual.persist_solved_state)
    assert int(actual.stop_reason_code) == _optimizer._C2_STOP_REASON_CONVERGED
    assert int(actual.iteration_count) == 0
    assert int(actual.applied_update_count) == 0
    assert int(actual.linear_solve_attempt_count) == 0
    assert int(actual.dense_materialization_count) == 1
    assert int(actual.lu_factorization_count) == 0
    assert int(actual.lu_solve_count) == 0
    assert int(actual.refinement_correction_count) == 0
    assert int(actual.rollback_recompute_count) == 0
    np.testing.assert_array_equal(np.asarray(actual.x), np.asarray(initial))
