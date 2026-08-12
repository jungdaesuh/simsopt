"""Observer-effect tests for production exact-Newton execution counts."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from simsopt_jax.geo.optimizers import optimizer as _optimizer


def _linear_residual(x):
    return x - jnp.asarray([1.0], dtype=x.dtype)


def _solve():
    return _optimizer.newton_exact_traceable(
        _linear_residual,
        jnp.asarray([0.0], dtype=jnp.float64),
        maxiter=2,
        tol=1.0e-12,
    )


def _assert_common_result_equal(left, right) -> None:
    common_keys = left.keys() & right.keys()
    for key in common_keys:
        left_value = left[key]
        right_value = right[key]
        if left_value is None:
            assert right_value is None
        else:
            np.testing.assert_array_equal(
                np.asarray(left_value),
                np.asarray(right_value),
                err_msg=key,
            )


def test_exact_execution_observer_is_absent_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        raising=False,
    )

    def forbidden_callback(*_args, **_kwargs):
        raise AssertionError("disabled exact Newton must not stage host callbacks")

    monkeypatch.setattr(_optimizer.jax.debug, "callback", forbidden_callback)
    result = _solve()

    assert bool(result["success"])
    assert "exact_newton_residual_evaluation_count" not in result
    assert "exact_newton_linear_operator_application_count" not in result
    assert "exact_newton_execution_observer_bearing" not in result
    assert "exact_newton_execution_counter_token" not in result


def test_exact_execution_observer_counts_actual_production_calls(monkeypatch) -> None:
    monkeypatch.setenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        "1",
    )

    result = _solve()

    residual_evaluations = int(result["exact_newton_residual_evaluation_count"])
    operator_applications = int(
        result["exact_newton_linear_operator_application_count"]
    )
    assert bool(result["success"])
    assert int(result["nit"]) == 1
    assert bool(result["exact_newton_execution_observer_bearing"])
    assert operator_applications > 0
    # One explicit initial residual and one accepted-candidate residual surround
    # the observed JVP primal executions for this one-step linear solve.
    assert residual_evaluations == operator_applications + 2


def test_exact_execution_observer_does_not_stage_debug_callback(monkeypatch) -> None:
    monkeypatch.setenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        "1",
    )

    def forbidden_callback(*_args, **_kwargs):
        raise AssertionError("exact Newton execution telemetry must remain on device")

    monkeypatch.setattr(_optimizer.jax.debug, "callback", forbidden_callback)
    result = _solve()

    assert bool(result["success"])
    assert int(result["exact_newton_linear_operator_application_count"]) > 0


def test_exact_execution_observer_preserves_solver_result(monkeypatch) -> None:
    monkeypatch.delenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        raising=False,
    )
    unobserved = _solve()
    monkeypatch.setenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        "true",
    )
    observed = _solve()

    _assert_common_result_equal(observed, unobserved)


def test_exact_execution_observer_respects_strict_transfer_guard(monkeypatch) -> None:
    monkeypatch.setenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        "1",
    )
    initial = jnp.asarray([0.0], dtype=jnp.float64)

    with jax.transfer_guard("disallow"):
        result = _optimizer.newton_exact_traceable(
            _linear_residual,
            initial,
            maxiter=2,
            tol=1.0e-12,
        )

    assert bool(result["success"])
    assert int(result["exact_newton_residual_evaluation_count"]) > 0
    assert int(result["exact_newton_linear_operator_application_count"]) > 0


def test_exact_execution_observer_is_fixed_shape_under_outer_jit(monkeypatch) -> None:
    monkeypatch.setenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        "on",
    )

    @jax.jit
    def solve(x):
        return _optimizer.newton_exact_traceable(
            _linear_residual,
            x,
            maxiter=2,
            tol=1.0e-12,
        )

    result = solve(jnp.asarray([0.0], dtype=jnp.float64))
    assert bool(result["exact_newton_execution_observer_bearing"])
    residual_count = result["exact_newton_residual_evaluation_count"]
    operator_count = result["exact_newton_linear_operator_application_count"]
    assert residual_count.shape == ()
    assert operator_count.shape == ()
    assert residual_count.dtype == jnp.int32
    assert operator_count.dtype == jnp.int32
    assert int(operator_count) > 0
    assert int(residual_count) == int(operator_count) + 2
    assert "exact_newton_execution_counter_token" not in result


def test_exact_execution_observer_counts_refinement_operator_work(monkeypatch) -> None:
    monkeypatch.setenv(
        _optimizer._TRACEABLE_EXACT_NEWTON_EXECUTION_COUNT_ENV,
        "1",
    )
    dimension = 65
    diagonal = jnp.logspace(0.0, 6.0, dimension, dtype=jnp.float64)
    target = jnp.ones(dimension, dtype=jnp.float64)

    def residual(x):
        return diagonal * x - target

    result = _optimizer.newton_exact_traceable(
        residual,
        jnp.zeros(dimension, dtype=jnp.float64),
        maxiter=1,
        tol=1.0e-12,
    )

    assert bool(result["success"])
    assert float(result["exact_refinement_correction_rel"]) > 0.0
    operator_count = int(result["exact_newton_linear_operator_application_count"])
    residual_count = int(result["exact_newton_residual_evaluation_count"])
    restart, maxiter = _optimizer._exact_newton_gmres_iteration_limits(dimension)
    single_solve_application_budget = 2 + maxiter * (restart + 1)
    assert operator_count > single_solve_application_budget
    assert residual_count > operator_count
