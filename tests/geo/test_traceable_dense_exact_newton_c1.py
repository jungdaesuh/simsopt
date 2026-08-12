from __future__ import annotations

from weakref import ref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.boozer_residual import boozer_residual_vector
from simsopt_jax.geo.optimizers import optimizer as _optimizer
from simsopt_jax.runtime.jaxpr_closure import (
    closure_converted_array_function,
    device_put_closure_consts,
)


def _quadratic_residual(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(
        [
            values[0] ** 2 + 0.5 * values[1] - 2.0,
            -0.25 * values[0] + values[1] ** 2 - 3.0,
        ],
        dtype=values.dtype,
    )


def _quadratic_jacobian(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [2.0 * values[0], 0.5],
            [-0.25, 2.0 * values[1]],
        ],
        dtype=np.float64,
    )


def _native_dense_direction(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(_quadratic_residual(jnp.asarray(values)))
    jacobian = _quadratic_jacobian(values)
    direction = np.linalg.solve(jacobian, residual)
    correction = np.linalg.solve(
        jacobian,
        residual - jacobian @ direction,
    )
    return direction + correction, correction


def _native_backtracking_trajectory(
    initial: np.ndarray,
    *,
    maxiter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    values = initial.copy()
    residual = np.asarray(_quadratic_residual(jnp.asarray(values)))
    norm = np.linalg.norm(residual)
    accepted_iterations = 0
    backtracking_iterations = 0
    while accepted_iterations < maxiter and norm > tol:
        direction, _ = _native_dense_direction(values)
        accepted = False
        alpha = 1.0
        for _ in range(_optimizer._NEWTON_BACKTRACKING_MAX_STEPS):
            candidate = values - alpha * direction
            candidate_residual = np.asarray(_quadratic_residual(jnp.asarray(candidate)))
            candidate_norm = np.linalg.norm(candidate_residual)
            backtracking_iterations += 1
            if (
                np.all(np.isfinite(candidate))
                and np.all(np.isfinite(candidate_residual))
                and np.isfinite(candidate_norm)
                and candidate_norm <= norm
            ):
                values = candidate
                residual = candidate_residual
                norm = candidate_norm
                accepted = True
                accepted_iterations += 1
                break
            alpha *= 0.5
        if not accepted:
            break
    return values, residual, accepted_iterations, backtracking_iterations


def test_c1_one_step_matches_native_dense_refined_newton_equations() -> None:
    state = np.asarray([1.25, 1.75], dtype=np.float64)
    expected_direction, expected_correction = _native_dense_direction(state)
    expected_residual = np.asarray(_quadratic_residual(jnp.asarray(state)))
    expected_jacobian = _quadratic_jacobian(state)

    actual = _optimizer._dense_direct_exact_newton_direction(
        _quadratic_residual,
        jnp.asarray(state),
        tol=1.0e-12,
    )

    np.testing.assert_array_equal(np.asarray(actual.residual), expected_residual)
    np.testing.assert_array_equal(np.asarray(actual.jacobian), expected_jacobian)
    np.testing.assert_allclose(
        np.asarray(actual.direction),
        expected_direction,
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(actual.correction),
        expected_correction,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.linear_residual),
        expected_residual - expected_jacobian @ expected_direction,
    )
    assert bool(actual.status.success)
    assert int(actual.status.dense_materialization_count) == 1
    assert int(actual.status.lu_factorization_count) == 1
    assert int(actual.status.lu_solve_count) == 12
    assert int(actual.status.refinement_correction_count) == 1


def test_c1_one_step_matches_native_boozer_parameter_equations_under_strict_transfer() -> (
    None
):
    """Exercise the production Boozer residual primitive with device closures."""

    magnetic_field_host = np.asarray([[[1.2, -0.4, 0.7]]], dtype=np.float64)
    xphi_host = np.asarray([[[0.3, -0.5, 0.1]]], dtype=np.float64)
    xtheta_host = np.asarray([[[-0.2, 0.8, 0.4]]], dtype=np.float64)
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
    field_squared = np.sum(magnetic_field_host**2)
    expected_residual = initial_host[1] * magnetic_field_host.ravel()[
        :2
    ] - field_squared * (
        xphi_host.ravel()[:2] + initial_host[0] * xtheta_host.ravel()[:2]
    )
    expected_jacobian = np.column_stack(
        (
            -field_squared * xtheta_host.ravel()[:2],
            magnetic_field_host.ravel()[:2],
        )
    )
    expected_x = initial_host - np.linalg.solve(
        expected_jacobian,
        expected_residual,
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

    runner = _optimizer._build_traceable_dense_direct_exact_newton_c1_runner(
        ref(residual_with_explicit_closure),
        1,
        1.0e-12,
    )

    with jax.transfer_guard("disallow"):
        result = runner(initial, residual_consts)
        jax.block_until_ready(result)

    assert int(result["nit"]) == 1
    assert bool(result["success"])
    np.testing.assert_allclose(
        np.asarray(result["x"]),
        expected_x,
        rtol=0.0,
        atol=2.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        np.asarray(result["residual"]),
        np.zeros(2, dtype=np.float64),
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )


def test_c1_short_trajectory_matches_native_scaling_and_backtracking() -> None:
    initial = np.asarray([1.25, 1.75], dtype=np.float64)
    tol = 1.0e-12
    expected_x, expected_residual, expected_nit, expected_backtracking = (
        _native_backtracking_trajectory(initial, maxiter=8, tol=tol)
    )
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c1_runner(
        ref(_quadratic_residual),
        8,
        tol,
    )

    result = runner(jnp.asarray(initial), ())

    assert bool(result["success"])
    assert int(result["nit"]) == expected_nit
    assert int(result["backtracking_iteration_count"]) == expected_backtracking
    np.testing.assert_allclose(
        np.asarray(result["x"]),
        expected_x,
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(result["residual"]),
        expected_residual,
        rtol=0.0,
        atol=1.0e-15,
    )
    attempts = int(result["linear_solve_attempt_count"])
    assert attempts == expected_nit
    assert int(result["dense_materialization_count"]) == attempts
    assert int(result["lu_factorization_count"]) == attempts
    assert int(result["lu_solve_count"]) == 12 * attempts
    assert int(result["refinement_correction_count"]) == attempts
    assert "jacobian" not in result
    assert "lu" not in result
    assert "pivots" not in result


def test_c1_backtracks_a_finite_oversized_dense_newton_step() -> None:
    def residual(values: jnp.ndarray) -> jnp.ndarray:
        return values**3 - 1.0

    runner = _optimizer._build_traceable_dense_direct_exact_newton_c1_runner(
        ref(residual),
        1,
        1.0e-12,
    )

    result = runner(jnp.asarray([0.1], dtype=jnp.float64), ())

    assert int(result["nit"]) == 1
    assert int(result["backtracking_iteration_count"]) > 1
    assert np.linalg.norm(np.asarray(result["residual"])) < 0.999


def test_c1_retries_once_at_strict_cap_after_loose_direction_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def residual(values: jnp.ndarray) -> jnp.ndarray:
        return values - 1.0

    def reject_step(_residual_eval, x, _dx, residual_value, current_norm):
        return {
            "iteration": jnp.asarray(1, dtype=jnp.int32),
            "alpha": jnp.asarray(0.5, dtype=x.dtype),
            "x": x,
            "residual": residual_value,
            "norm": current_norm,
            "accepted": jnp.asarray(False),
        }

    monkeypatch.setattr(_optimizer, "_backtracking_residual_step", reject_step)
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c1_runner(
        ref(residual),
        4,
        1.0e-12,
    )
    initial = jnp.asarray([0.0], dtype=jnp.float64)

    result = runner(initial, ())

    assert not bool(result["success"])
    assert bool(result["stalled"])
    assert int(result["nit"]) == 0
    assert int(result["linear_solve_attempt_count"]) == 2
    np.testing.assert_array_equal(np.asarray(result["x"]), np.asarray(initial))


def test_c1_singular_jacobian_fails_closed_and_retains_incumbent() -> None:
    def singular_residual(values: jnp.ndarray) -> jnp.ndarray:
        total = values[0] + values[1] - 1.0
        return jnp.asarray([total, 2.0 * total], dtype=values.dtype)

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    direction = _optimizer._dense_direct_exact_newton_direction(
        singular_residual,
        initial,
        tol=1.0e-12,
    )
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c1_runner(
        ref(singular_residual),
        4,
        1.0e-12,
    )

    result = runner(initial, ())

    assert not bool(direction.status.success)
    assert np.all(np.isnan(np.asarray(direction.direction)))
    assert not bool(result["success"])
    assert bool(result["stalled"])
    assert int(result["nit"]) == 0
    np.testing.assert_array_equal(np.asarray(result["x"]), np.asarray(initial))


def test_c1_nonfinite_initial_residual_fails_before_dense_materialization() -> None:
    def nonfinite_residual(values: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(values, jnp.nan)

    initial = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c1_runner(
        ref(nonfinite_residual),
        4,
        1.0e-12,
    )

    result = runner(initial, ())

    assert not bool(result["success"])
    assert int(result["linear_solve_attempt_count"]) == 0
    assert int(result["dense_materialization_count"]) == 0
    assert int(result["lu_factorization_count"]) == 0
    np.testing.assert_array_equal(np.asarray(result["x"]), np.asarray(initial))
