from __future__ import annotations

from weakref import ref

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from simsopt_jax.geo.optimizers import optimizer as _optimizer


def _run_c2(
    residual_fn,
    initial: np.ndarray,
    *,
    maxiter: int,
    tol: float,
    value_jacobian_fn=None,
):
    runner = _optimizer._build_traceable_dense_direct_exact_newton_c2_runner(
        ref(residual_fn),
        maxiter,
        tol,
        value_jacobian_fn=value_jacobian_fn,
    )
    return runner(jnp.asarray(initial, dtype=jnp.float64), ())


def _nonlinear_pair(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0, x1 = values
    one = jnp.asarray(1.0, dtype=values.dtype)
    residual = jnp.stack((x0**2 + x1 - one, x0 + x1**2 - one))
    jacobian = jnp.stack(
        (
            jnp.stack((2.0 * x0, one)),
            jnp.stack((one, 2.0 * x1)),
        )
    )
    return residual, jacobian


def test_c2_analytic_pair_matches_ad_on_a_nonlinear_system() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return _nonlinear_pair(values)[0]

    initial = np.asarray([0.8, 0.6], dtype=np.float64)
    ad_result = _run_c2(
        residual,
        initial,
        maxiter=8,
        tol=1.0e-12,
    )
    analytic_result = _run_c2(
        residual,
        initial,
        maxiter=8,
        tol=1.0e-12,
        value_jacobian_fn=_nonlinear_pair,
    )

    np.testing.assert_allclose(
        np.asarray(analytic_result.x),
        np.asarray(ad_result.x),
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(analytic_result.residual),
        np.asarray(ad_result.residual),
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(analytic_result.returned_jacobian),
        np.asarray(_nonlinear_pair(analytic_result.x)[1]),
        rtol=0.0,
        atol=2.0e-12,
    )
    assert bool(analytic_result.success)
    assert int(analytic_result.iteration_count) == int(ad_result.iteration_count)
    assert int(analytic_result.applied_update_count) == int(ad_result.applied_update_count)
    assert int(analytic_result.dense_materialization_count) == (
        int(analytic_result.applied_update_count) + 1
    )


def test_c2_analytic_pair_stops_initially_converged_without_a_linear_solve() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return values - jnp.asarray([1.0, -2.0], dtype=values.dtype)

    def value_jacobian(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return residual(values), jnp.eye(values.shape[0], dtype=values.dtype)

    initial = np.asarray([1.0, -2.0], dtype=np.float64)
    actual = _run_c2(
        residual,
        initial,
        maxiter=3,
        tol=1.0e-12,
        value_jacobian_fn=value_jacobian,
    )

    assert bool(actual.success)
    assert int(actual.iteration_count) == 0
    assert int(actual.applied_update_count) == 0
    assert int(actual.linear_solve_attempt_count) == 0
    assert int(actual.dense_materialization_count) == 1
    assert int(actual.rollback_recompute_count) == 0
    np.testing.assert_array_equal(np.asarray(actual.x), initial)
    np.testing.assert_array_equal(
        np.asarray(actual.returned_jacobian),
        np.eye(2, dtype=np.float64),
    )


def test_c2_analytic_pair_rebuilds_jacobian_at_changed_returned_state() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return values**3 - 1.0

    def value_jacobian(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return residual(values), jnp.diag(3.0 * values**2)

    initial = np.asarray([0.8], dtype=np.float64)
    actual = _run_c2(
        residual,
        initial,
        maxiter=1,
        tol=1.0e-12,
        value_jacobian_fn=value_jacobian,
    )

    assert not bool(actual.success)
    assert bool(actual.persist_solved_state)
    assert not bool(actual.rollback_branch_taken)
    assert int(actual.applied_update_count) == 1
    assert int(actual.dense_materialization_count) == 2
    np.testing.assert_allclose(
        np.asarray(actual.returned_jacobian),
        np.asarray([[3.0 * np.asarray(actual.x)[0] ** 2]]),
        rtol=0.0,
        atol=2.0e-12,
    )
    assert not np.array_equal(np.asarray(actual.x), initial)


def test_c2_analytic_pair_failure_rolls_back_and_rebuilds_initial_pair() -> None:
    def residual(values: jax.Array) -> jax.Array:
        total = values[0] + values[1] - 1.0
        return jnp.asarray([total, 2.0 * total], dtype=values.dtype)

    def value_jacobian(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return residual(values), jnp.asarray(
            [[1.0, 1.0], [2.0, 2.0]],
            dtype=values.dtype,
        )

    initial = np.asarray([0.0, 0.0], dtype=np.float64)
    actual = _run_c2(
        residual,
        initial,
        maxiter=3,
        tol=1.0e-12,
        value_jacobian_fn=value_jacobian,
    )

    assert bool(actual.numerical_failure)
    assert not bool(actual.persist_solved_state)
    assert bool(actual.rollback_branch_taken)
    assert int(actual.iteration_count) == 0
    assert int(actual.applied_update_count) == 0
    assert int(actual.linear_solve_attempt_count) == 1
    assert int(actual.dense_materialization_count) == 2
    assert int(actual.rollback_recompute_count) == 1
    np.testing.assert_array_equal(np.asarray(actual.x), initial)
    np.testing.assert_allclose(
        np.asarray(actual.residual),
        np.asarray([-1.0, -2.0]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.returned_jacobian),
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
    )


def test_c2_analytic_pair_does_not_trace_the_residual_ad_materializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_residual(_values: jax.Array) -> jax.Array:
        raise AssertionError("injected C2 must not evaluate the residual callback")

    def value_jacobian(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            values - jnp.asarray([1.0], dtype=values.dtype),
            jnp.ones((1, 1), dtype=values.dtype),
        )

    def forbidden_ad_materializer(*_args: object, **_kwargs: object):
        raise AssertionError("injected C2 must not materialize a basis AD Jacobian")

    monkeypatch.setattr(
        _optimizer,
        "_linearize_and_materialize_dense_square_jacobian",
        forbidden_ad_materializer,
    )
    monkeypatch.setattr(jax, "jvp", forbidden_ad_materializer)
    actual = _run_c2(
        forbidden_residual,
        np.asarray([0.0], dtype=np.float64),
        maxiter=1,
        tol=1.0e-12,
        value_jacobian_fn=value_jacobian,
    )

    np.testing.assert_allclose(np.asarray(actual.x), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(actual.returned_jacobian), np.asarray([[1.0]]))
    assert int(actual.dense_materialization_count) == 2


def test_c2_analytic_telemetry_uses_value_and_jacobian_assembler_code() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return values - 1.0

    def value_jacobian(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return residual(values), jnp.eye(values.shape[0], dtype=values.dtype)

    materialization = _optimizer._materialize_traceable_dense_exact_newton_c2_state(
        residual,
        jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        (),
        value_jacobian_fn=value_jacobian,
    )

    telemetry = materialization.telemetry
    assert int(telemetry.assembler_code) == int(
        _optimizer._DenseJacobianAssembler.VALUE_AND_JACOBIAN
    )
    assert int(telemetry.residual_evaluation_count) == 1
    assert int(telemetry.primal_traversal_count) == 1
    assert int(telemetry.tangent_batch_count) == 0
    assert int(telemetry.tangent_direction_count) == 0
    assert int(telemetry.batch_width) == 0
    assert int(telemetry.tail_width) == 0


def test_c2_analytic_cache_isolated_by_pair_provider_identity() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return values - 1.0

    def provider_unit(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return residual(values), jnp.ones((1, 1), dtype=values.dtype)

    def provider_two(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return residual(values), jnp.asarray([[2.0]], dtype=values.dtype)

    runner_unit = _optimizer._make_traceable_dense_direct_exact_newton_c2_runner(
        residual,
        1,
        1.0e-12,
        value_jacobian_fn=provider_unit,
    )
    runner_unit_again = (
        _optimizer._make_traceable_dense_direct_exact_newton_c2_runner(
            residual,
            1,
            1.0e-12,
            value_jacobian_fn=provider_unit,
        )
    )
    runner_two = _optimizer._make_traceable_dense_direct_exact_newton_c2_runner(
        residual,
        1,
        1.0e-12,
        value_jacobian_fn=provider_two,
    )

    assert runner_unit is runner_unit_again
    assert runner_unit is not runner_two
    unit_result = runner_unit(jnp.asarray([0.0], dtype=jnp.float64), ())
    two_result = runner_two(jnp.asarray([0.0], dtype=jnp.float64), ())
    np.testing.assert_allclose(np.asarray(unit_result.x), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(two_result.x), np.asarray([0.5]))
    np.testing.assert_allclose(
        np.asarray(unit_result.returned_jacobian),
        np.asarray([[1.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(two_result.returned_jacobian),
        np.asarray([[2.0]]),
    )
