"""Analytic final-state residual/Jacobian injection for the fused payload."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax_adapters.geo import surface_objectives_traceable as _traceable

jax.config.update("jax_enable_x64", True)

_TARGET = np.asarray([1.2, -0.4], dtype=np.float64)
_STATE_QUADRATIC = np.asarray([[1.7, 0.15], [0.15, 1.3]], dtype=np.float64)
_STATE_LINEAR = np.asarray([-0.25, 0.4], dtype=np.float64)
_DYNAMIC_QUADRATIC = np.asarray([[1.4, 0.1], [0.1, 1.8]], dtype=np.float64)
_DYNAMIC_LINEAR = np.asarray([0.2, -0.35], dtype=np.float64)


def _dynamic_inputs_from_dofs(coil_dofs):
    return (
        jnp.stack(
            (
                jnp.sin(coil_dofs[0]) + 0.1 * coil_dofs[1] ** 2,
                jnp.exp(0.2 * coil_dofs[1]) + 0.25 * coil_dofs[0] * coil_dofs[1],
            )
        ),
    )


def _exact_residual(state, _coil_dofs, dynamic_inputs, configuration):
    target0, target1 = configuration
    (dynamic,) = dynamic_inputs
    x0, x1 = state
    return jnp.stack(
        (
            x0 + 0.4 * dynamic[0] - target0,
            0.3 * x0 + 0.2 * x0**2 + 1.1 * x1 + 0.7 * dynamic[1] - target1,
        )
    )


def _exact_value_jacobian(state, coil_dofs, dynamic_inputs, configuration):
    return _exact_residual(state, coil_dofs, dynamic_inputs, configuration), jnp.stack(
        (
            jnp.asarray([1.0, 0.0], dtype=state.dtype),
            jnp.asarray([0.3 + 0.4 * state[0], 1.1], dtype=state.dtype),
        )
    )


def _returned_state(coil_dofs, dynamic_inputs, configuration, *, solve_success):
    target0, target1 = configuration
    (dynamic,) = dynamic_inputs
    x0 = target0 - 0.4 * dynamic[0]
    x1 = (target1 - 0.3 * x0 - 0.2 * x0**2 - 0.7 * dynamic[1]) / 1.1
    return _traceable._TraceableExactReturnedState(
        solved_state=jnp.stack((x0, x1)),
        solve_success=jnp.asarray(solve_success),
    )


def _scalar_objective(state, _coil_dofs, dynamic_inputs, _configuration):
    (dynamic,) = dynamic_inputs
    state_quadratic = jnp.asarray(_STATE_QUADRATIC, dtype=state.dtype)
    state_linear = jnp.asarray(_STATE_LINEAR, dtype=state.dtype)
    dynamic_quadratic = jnp.asarray(_DYNAMIC_QUADRATIC, dtype=state.dtype)
    dynamic_linear = jnp.asarray(_DYNAMIC_LINEAR, dtype=state.dtype)
    return (
        0.5 * state @ state_quadratic @ state
        + state_linear @ state
        + 0.5 * dynamic @ dynamic_quadratic @ dynamic
        + dynamic_linear @ dynamic
    )


def _configuration(device):
    return tuple(jax.device_put(value, device) for value in _TARGET)


def _build_evaluator(
    device,
    *,
    exact_value_jacobian_fn: Callable[
        [jax.Array, jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]],
        tuple[jax.Array, jax.Array],
    ]
    | None,
    solve_success: bool = True,
):
    configuration = _configuration(device)

    def returned_state(coil_dofs, dynamic_inputs, residual_configuration):
        return _returned_state(
            coil_dofs,
            dynamic_inputs,
            residual_configuration,
            solve_success=solve_success,
        )

    return _traceable._build_traceable_exact_payload_fused_value_and_gradient(
        returned_state_from_dofs=returned_state,
        coil_dynamic_inputs_from_dofs=_dynamic_inputs_from_dofs,
        scalar_objective_fn=_scalar_objective,
        exact_residual_fn=_exact_residual,
        exact_value_jacobian_fn=exact_value_jacobian_fn,
        residual_configuration=configuration,
        producer_residual_tol=1.0e-12,
        linear_solve_tol=1.0e-12,
    )


def _numpy_dynamic(coil_dofs):
    return np.asarray(
        [
            np.sin(coil_dofs[0]) + 0.1 * coil_dofs[1] ** 2,
            np.exp(0.2 * coil_dofs[1]) + 0.25 * coil_dofs[0] * coil_dofs[1],
        ],
        dtype=np.float64,
    )


def _numpy_dynamic_jacobian(coil_dofs):
    return np.asarray(
        [
            [np.cos(coil_dofs[0]), 0.2 * coil_dofs[1]],
            [
                0.25 * coil_dofs[1],
                0.2 * np.exp(0.2 * coil_dofs[1]) + 0.25 * coil_dofs[0],
            ],
        ],
        dtype=np.float64,
    )


def _numpy_state(coil_dofs):
    dynamic = _numpy_dynamic(coil_dofs)
    x0 = _TARGET[0] - 0.4 * dynamic[0]
    x1 = (_TARGET[1] - 0.3 * x0 - 0.2 * x0**2 - 0.7 * dynamic[1]) / 1.1
    return np.asarray([x0, x1], dtype=np.float64)


def _numpy_value(coil_dofs):
    state = _numpy_state(coil_dofs)
    dynamic = _numpy_dynamic(coil_dofs)
    return float(
        0.5 * state @ _STATE_QUADRATIC @ state
        + _STATE_LINEAR @ state
        + 0.5 * dynamic @ _DYNAMIC_QUADRATIC @ dynamic
        + _DYNAMIC_LINEAR @ dynamic
    )


def _numpy_implicit_gradient(coil_dofs):
    state = _numpy_state(coil_dofs)
    dynamic = _numpy_dynamic(coil_dofs)
    state_jacobian = np.asarray(
        [[1.0, 0.0], [0.3 + 0.4 * state[0], 1.1]],
        dtype=np.float64,
    )
    dynamic_residual_jacobian = np.asarray(
        [[0.4, 0.0], [0.0, 0.7]],
        dtype=np.float64,
    )
    state_gradient = _STATE_QUADRATIC @ state + _STATE_LINEAR
    dynamic_gradient = _DYNAMIC_QUADRATIC @ dynamic + _DYNAMIC_LINEAR
    adjoint = np.linalg.solve(state_jacobian.T, state_gradient)
    total_dynamic_gradient = dynamic_gradient - dynamic_residual_jacobian.T @ adjoint
    return _numpy_dynamic_jacobian(coil_dofs).T @ total_dynamic_gradient


def test_analytic_pair_injection_matches_implicit_gradient_and_uses_returned_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = jax.devices("cpu")[0]
    coil_values = np.asarray([0.6, -0.35], dtype=np.float64)
    coil_dofs = jax.device_put(coil_values, device)
    provider_states: list[np.ndarray] = []

    def provider(state, coil, dynamic_inputs, configuration):
        jax.debug.callback(
            lambda value: provider_states.append(np.asarray(value)),
            state,
            ordered=True,
        )
        return _exact_value_jacobian(state, coil, dynamic_inputs, configuration)

    def forbidden_ad_materializer(*_args, **_kwargs):
        raise AssertionError(
            "analytic payload must not materialize a basis AD Jacobian"
        )

    monkeypatch.setattr(
        _traceable._linear_solve,
        "_linearize_and_materialize_dense_square_jacobian",
        forbidden_ad_materializer,
    )
    evaluator = _build_evaluator(
        device,
        exact_value_jacobian_fn=provider,
    )
    result = evaluator(coil_dofs)
    jax.block_until_ready(result)

    expected_gradient = _numpy_implicit_gradient(coil_values)
    epsilon = 1.0e-6
    finite_difference = np.asarray(
        [
            (
                _numpy_value(coil_values + epsilon * np.eye(2)[index])
                - _numpy_value(coil_values - epsilon * np.eye(2)[index])
            )
            / (2.0 * epsilon)
            for index in range(2)
        ]
    )

    assert bool(result.status.success)
    assert bool(result.status.live_residual_matches_payload)
    assert len(provider_states) == 1
    np.testing.assert_allclose(
        provider_states[0],
        _numpy_state(coil_values),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(result.gradient),
        expected_gradient,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(result.gradient),
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )
    assert int(result.evidence.linearization_primal_traversal_count) == 1


def test_analytic_pair_injection_preserves_failed_producer_nan_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = jax.devices("cpu")[0]

    def forbidden_ad_materializer(*_args, **_kwargs):
        raise AssertionError(
            "analytic payload must not materialize a basis AD Jacobian"
        )

    monkeypatch.setattr(
        _traceable._linear_solve,
        "_linearize_and_materialize_dense_square_jacobian",
        forbidden_ad_materializer,
    )
    evaluator = _build_evaluator(
        device,
        exact_value_jacobian_fn=_exact_value_jacobian,
        solve_success=False,
    )
    result = evaluator(jnp.asarray([0.6, -0.35], dtype=jnp.float64))
    jax.block_until_ready(result)

    assert not bool(result.status.returned_state_solve_success)
    assert not bool(result.status.producer_solve_success)
    assert not bool(result.status.success)
    assert np.isnan(np.asarray(result.value))
    assert np.isnan(np.asarray(result.gradient)).all()
    assert np.isnan(np.asarray(result.evidence.adjoint.adjoint_output)).all()
