"""Closed exact-linearization producer and fused-adjoint contract."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest
from simsopt_jax.geo.optimizers import linear_solve as _linear_solve
from simsopt_jax.runtime.trace_annotations import PhaseId, trace_session
from simsopt_jax_adapters.geo import surface_objectives_traceable as _traceable

_JACOBIAN = np.asarray(
    [[3.0, -1.0, 0.5], [2.0, 4.0, -0.25], [0.0, 1.5, 2.5]],
    dtype=np.float64,
)
_DYNAMIC_TO_RESIDUAL = np.asarray(
    [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
    dtype=np.float64,
)
_RESIDUAL_OFFSET = np.asarray([0.75, -1.25, 2.0], dtype=np.float64)
_STATE_QUADRATIC = np.asarray(
    [[2.0, 0.25, -0.1], [0.25, 1.5, 0.2], [-0.1, 0.2, 1.25]],
    dtype=np.float64,
)
_STATE_LINEAR = np.asarray([0.4, -0.6, 0.25], dtype=np.float64)
_DYNAMIC_QUADRATIC = np.asarray([[1.75, -0.2], [-0.2, 2.25]], dtype=np.float64)
_DYNAMIC_LINEAR = np.asarray([-0.35, 0.8], dtype=np.float64)
_OFF_STATIONARY_DELTA = np.asarray([2.0e-12, -1.0e-12, 1.5e-12])
_OLD_COILS = np.asarray([-0.8, 0.45], dtype=np.float64)


def _configuration(device):
    return tuple(
        jax.device_put(value, device)
        for value in (
            _JACOBIAN,
            _DYNAMIC_TO_RESIDUAL,
            _RESIDUAL_OFFSET,
            _STATE_QUADRATIC,
            _STATE_LINEAR,
            _DYNAMIC_QUADRATIC,
            _DYNAMIC_LINEAR,
        )
    )


def _ill_conditioned_configuration(device):
    jacobian = np.diag(np.asarray([1.0, 2.0, 1.0e-14], dtype=np.float64))
    dynamic_to_residual = _DYNAMIC_TO_RESIDUAL.copy()
    dynamic_to_residual[2, :] = 0.0
    offset = _RESIDUAL_OFFSET.copy()
    offset[2] = 0.0
    return tuple(
        jax.device_put(value, device)
        for value in (
            jacobian,
            dynamic_to_residual,
            offset,
            _STATE_QUADRATIC,
            _STATE_LINEAR,
            _DYNAMIC_QUADRATIC,
            _DYNAMIC_LINEAR,
        )
    )


def _dynamic_inputs_from_dofs(coil_dofs):
    return (
        jnp.stack(
            (
                jnp.sin(coil_dofs[0]) + 0.1 * coil_dofs[1] ** 2,
                jnp.exp(0.2 * coil_dofs[1]) + coil_dofs[0] * coil_dofs[1],
            )
        ),
    )


def _exact_residual(state, _coil_dofs, dynamic_inputs, configuration):
    jacobian, dynamic_to_residual, offset, *_ = configuration
    (geometry,) = dynamic_inputs
    return jacobian @ state + dynamic_to_residual @ geometry - offset


def _scalar_objective(state, _coil_dofs, dynamic_inputs, configuration):
    _, _, _, state_quadratic, state_linear, dynamic_quadratic, dynamic_linear = (
        configuration
    )
    (geometry,) = dynamic_inputs
    return (
        0.5 * state @ state_quadratic @ state
        + state_linear @ state
        + 0.5 * geometry @ dynamic_quadratic @ geometry
        + dynamic_linear @ geometry
    )


def _zero_rhs_objective(_state, _coil_dofs, dynamic_inputs, configuration):
    *_, dynamic_quadratic, dynamic_linear = configuration
    (geometry,) = dynamic_inputs
    return 0.5 * geometry @ dynamic_quadratic @ geometry + dynamic_linear @ geometry


def _zero_weight_objective(state, _coil_dofs, _dynamic_inputs, _configuration):
    return jnp.zeros_like(jnp.sum(state))


def _returned_state(mode: str = "exact"):
    old_geometry = _numpy_dynamic(_OLD_COILS)

    def returned_state(coil_dofs, dynamic_inputs, configuration):
        del coil_dofs
        jacobian, dynamic_to_residual, offset, *_ = configuration
        (geometry,) = dynamic_inputs
        selected_geometry = (
            jnp.asarray(old_geometry) if mode == "old_state" else geometry
        )
        solved_state = jnp.linalg.solve(
            jacobian,
            offset - dynamic_to_residual @ selected_geometry,
        )
        if mode == "off_stationary":
            solved_state = solved_state + jnp.asarray(_OFF_STATIONARY_DELTA)
        return _traceable._TraceableExactReturnedState(
            solved_state=solved_state,
            solve_success=jnp.asarray(mode != "failed_status"),
        )

    return returned_state


def _closed_evaluator(
    device,
    *,
    mode: str = "exact",
    objective_fn=_scalar_objective,
    producer_residual_tol: float = 1.0e-10,
    configuration=None,
):
    return _traceable._build_traceable_exact_payload_fused_value_and_gradient(
        returned_state_from_dofs=_returned_state(mode),
        coil_dynamic_inputs_from_dofs=_dynamic_inputs_from_dofs,
        scalar_objective_fn=objective_fn,
        exact_residual_fn=_exact_residual,
        residual_configuration=(
            _configuration(device) if configuration is None else configuration
        ),
        producer_residual_tol=producer_residual_tol,
        linear_solve_tol=1.0e-12,
    )


def _numpy_dynamic(coil_dofs):
    return np.asarray(
        [
            np.sin(coil_dofs[0]) + 0.1 * coil_dofs[1] ** 2,
            np.exp(0.2 * coil_dofs[1]) + coil_dofs[0] * coil_dofs[1],
        ]
    )


def _numpy_dynamic_jacobian(coil_dofs):
    return np.asarray(
        [
            [np.cos(coil_dofs[0]), 0.2 * coil_dofs[1]],
            [coil_dofs[1], 0.2 * np.exp(0.2 * coil_dofs[1]) + coil_dofs[0]],
        ]
    )


def _numpy_value(coil_dofs, *, off_stationary=False):
    geometry = _numpy_dynamic(coil_dofs)
    state = np.linalg.solve(
        _JACOBIAN,
        _RESIDUAL_OFFSET - _DYNAMIC_TO_RESIDUAL @ geometry,
    )
    if off_stationary:
        state = state + _OFF_STATIONARY_DELTA
    return float(
        0.5 * state @ _STATE_QUADRATIC @ state
        + _STATE_LINEAR @ state
        + 0.5 * geometry @ _DYNAMIC_QUADRATIC @ geometry
        + _DYNAMIC_LINEAR @ geometry
    )


def _numpy_implicit_gradient(coil_dofs, *, off_stationary=False):
    geometry = _numpy_dynamic(coil_dofs)
    state = np.linalg.solve(
        _JACOBIAN,
        _RESIDUAL_OFFSET - _DYNAMIC_TO_RESIDUAL @ geometry,
    )
    if off_stationary:
        state = state + _OFF_STATIONARY_DELTA
    adjoint = np.linalg.solve(
        _JACOBIAN.T,
        _STATE_QUADRATIC @ state + _STATE_LINEAR,
    )
    geometry_gradient = (
        _DYNAMIC_QUADRATIC @ geometry
        + _DYNAMIC_LINEAR
        - _DYNAMIC_TO_RESIDUAL.T @ adjoint
    )
    return _numpy_dynamic_jacobian(coil_dofs).T @ geometry_gradient


@pytest.mark.parametrize("mode", ("exact", "off_stationary"))
def test_closed_graph_rebuilds_nonlinear_dynamic_inputs_and_matches_implicit_gradient(
    mode,
):
    device = jax.devices("cpu")[0]
    coil_values = np.asarray([0.6, -0.35], dtype=np.float64)
    coil_dofs = jax.device_put(coil_values, device)
    evaluator = _closed_evaluator(device, mode=mode)
    compiled = evaluator.lower(coil_dofs).compile()

    with jax.transfer_guard("disallow"):
        result = compiled(coil_dofs)
        jax.block_until_ready(result)

    expected = _numpy_implicit_gradient(
        coil_values,
        off_stationary=mode == "off_stationary",
    )
    epsilon = 1.0e-6
    finite_difference = np.asarray(
        [
            (
                _numpy_value(
                    coil_values + epsilon * np.eye(2)[index],
                    off_stationary=mode == "off_stationary",
                )
                - _numpy_value(
                    coil_values - epsilon * np.eye(2)[index],
                    off_stationary=mode == "off_stationary",
                )
            )
            / (2.0 * epsilon)
            for index in range(2)
        ]
    )

    assert bool(result.status.success)
    assert bool(result.status.dynamic_inputs_match)
    assert bool(result.status.live_residual_matches_payload)
    assert bool(result.status.factorization_valid)
    np.testing.assert_allclose(np.asarray(result.gradient), expected, rtol=2.0e-13)
    np.testing.assert_allclose(
        np.asarray(result.gradient), finite_difference, rtol=2.0e-8, atol=2.0e-9
    )
    np.testing.assert_array_equal(
        np.asarray(result.evidence.producer_residual),
        np.asarray(result.evidence.live_residual),
    )
    consumer = result.evidence.consumer_reuse_counts
    full = result.evidence.full_graph_counts
    assert int(consumer.dense_materialization_count) == 0
    assert int(consumer.lu_factorization_count) == 0
    assert int(consumer.lu_solve_count) == 12
    assert int(consumer.refinement_correction_count) == 1
    assert int(full.dense_materialization_count) == 1
    assert int(full.lu_factorization_count) == 1
    assert int(full.lu_solve_count) == 12
    assert int(full.refinement_correction_count) == 1
    assert int(result.evidence.factorization_reconstruction_count) == 1
    assert int(result.evidence.linearization_primal_traversal_count) == 1


def test_closed_fused_gradient_matches_existing_split_two_pullback(
    monkeypatch,
):
    device = jax.devices("cpu")[0]
    configuration = _configuration(device)
    coil_dofs = jax.device_put(np.asarray([0.6, -0.35]), device)
    dynamic_inputs = _dynamic_inputs_from_dofs(coil_dofs)
    returned_state = _returned_state()(coil_dofs, dynamic_inputs, configuration)

    def split_scalar_objective(
        state,
        current_coils,
        current_dynamic_inputs,
        *,
        objective_kwargs,
    ):
        return _scalar_objective(
            state,
            current_coils,
            current_dynamic_inputs,
            objective_kwargs["configuration"],
        )

    def solve_linearization(
        _booz_jax,
        _solved_state,
        rhs,
        _current_dynamic_inputs,
        objective_kwargs,
        *,
        transpose,
        **_kwargs,
    ):
        jacobian = objective_kwargs["configuration"][0]
        matrix = jacobian.T if transpose else jacobian
        solution = jnp.linalg.solve(matrix, rhs)
        residual = rhs - matrix @ solution
        status = _linear_solve._linear_solve_status(
            solution,
            residual,
            rhs,
            tol=1.0e-12,
            iterations=jnp.asarray(0, dtype=jnp.int32),
        )._replace(success=jnp.asarray(True))
        return solution, status

    monkeypatch.setattr(
        _traceable,
        "_traceable_solve_linearization",
        solve_linearization,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_directional_exact_residual",
        lambda state, tangent, current_dynamic_inputs, objective_kwargs: jnp.vdot(
            tangent,
            _exact_residual(
                state,
                jnp.zeros_like(coil_dofs),
                current_dynamic_inputs,
                objective_kwargs["configuration"],
            ),
        ),
    )
    split = _traceable._traceable_objective_gradient_parts(
        object(),
        _dynamic_inputs_from_dofs,
        coil_dofs=coil_dofs,
        solved_x=returned_state.solved_state,
        solved_linear_solve_factors=None,
        linearization_kind="exact_jacobian",
        linear_solve_tol=1.0e-12,
        linear_solve_stab=0.0,
        objective_kwargs={"configuration": configuration},
        scalar_objective_fn=split_scalar_objective,
    )
    fused = _closed_evaluator(
        device,
        configuration=configuration,
    )(coil_dofs)

    assert bool(split[3])
    assert bool(fused.status.success)
    np.testing.assert_allclose(
        np.asarray(fused.gradient),
        np.asarray(split[2]),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


@pytest.mark.parametrize("seed", (1409, 3253, 7919))
def test_closed_gradient_matches_seeded_random_directional_derivative(seed):
    device = jax.devices("cpu")[0]
    rng = np.random.default_rng(seed)
    coil_values = rng.uniform(-0.7, 0.7, size=2)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    coil_dofs = jax.device_put(coil_values, device)
    evaluator = _closed_evaluator(device)
    result = evaluator(coil_dofs)
    epsilon = 1.0e-6
    plus = evaluator(jax.device_put(coil_values + epsilon * direction, device))
    minus = evaluator(jax.device_put(coil_values - epsilon * direction, device))
    finite_difference = (float(plus.value) - float(minus.value)) / (2.0 * epsilon)

    assert bool(result.status.success)
    np.testing.assert_allclose(
        np.vdot(np.asarray(result.gradient), direction),
        finite_difference,
        rtol=3.0e-8,
        atol=3.0e-9,
    )


@pytest.mark.parametrize("mode", ("old_state", "failed_status"))
def test_closed_graph_masks_stale_state_and_failed_producer(mode):
    device = jax.devices("cpu")[0]
    coil_dofs = jax.device_put(np.asarray([0.6, -0.35]), device)
    result = _closed_evaluator(device, mode=mode)(coil_dofs)

    assert not bool(result.status.success)
    assert not bool(result.status.producer_solve_success)
    assert np.isnan(np.asarray(result.value))
    assert np.isnan(np.asarray(result.gradient)).all()
    assert np.isnan(np.asarray(result.evidence.adjoint.adjoint_output)).all()
    assert int(result.evidence.consumer_reuse_counts.lu_solve_count) == 0
    assert int(result.evidence.full_graph_counts.dense_materialization_count) == 1
    assert int(result.evidence.full_graph_counts.lu_factorization_count) == 1
    assert int(result.evidence.factorization_reconstruction_count) == 1


def test_closed_graph_zero_rhs_preserves_exact_fast_path_and_dynamic_gradient():
    device = jax.devices("cpu")[0]
    coil_values = np.asarray([0.6, -0.35], dtype=np.float64)
    coil_dofs = jax.device_put(coil_values, device)
    result = _closed_evaluator(device, objective_fn=_zero_rhs_objective)(coil_dofs)
    geometry = _numpy_dynamic(coil_values)
    expected = _numpy_dynamic_jacobian(coil_values).T @ (
        _DYNAMIC_QUADRATIC @ geometry + _DYNAMIC_LINEAR
    )

    assert bool(result.status.success)
    np.testing.assert_allclose(np.asarray(result.gradient), expected, rtol=2.0e-13)
    np.testing.assert_array_equal(
        np.asarray(result.evidence.adjoint.adjoint_output), np.zeros(3)
    )
    assert int(result.evidence.consumer_reuse_counts.lu_solve_count) == 0
    assert int(result.evidence.consumer_reuse_counts.refinement_correction_count) == 0
    assert int(result.evidence.consumer_reuse_counts.adjoint_execution_count) == 0
    assert int(result.evidence.full_graph_counts.dense_materialization_count) == 1
    assert int(result.evidence.full_graph_counts.lu_factorization_count) == 1


def test_zero_weight_inactive_objective_returns_exact_zero_without_adjoint():
    device = jax.devices("cpu")[0]
    coil_dofs = jax.device_put(np.asarray([0.6, -0.35]), device)
    result = _closed_evaluator(device, objective_fn=_zero_weight_objective)(coil_dofs)

    assert bool(result.status.success)
    assert float(result.value) == 0.0
    np.testing.assert_array_equal(np.asarray(result.gradient), np.zeros(2))
    assert int(result.evidence.consumer_reuse_counts.lu_solve_count) == 0
    assert int(result.evidence.consumer_reuse_counts.adjoint_execution_count) == 0
    assert int(result.evidence.full_graph_counts.dense_materialization_count) == 1
    assert int(result.evidence.full_graph_counts.lu_factorization_count) == 1


def test_valid_producer_masks_forced_adjoint_numerical_rejection():
    device = jax.devices("cpu")[0]
    configuration = _ill_conditioned_configuration(device)
    coil_dofs = jax.device_put(np.asarray([0.6, -0.35]), device)
    result = _closed_evaluator(
        device,
        configuration=configuration,
    )(coil_dofs)

    assert bool(result.status.returned_state_solve_success)
    assert bool(result.status.producer_solve_success)
    assert bool(result.status.returned_state_residual_success)
    assert bool(result.status.payload_validation_success)
    assert bool(result.status.factorization_valid)
    assert not bool(result.status.adjoint_solve_success)
    assert not bool(result.status.success)
    assert np.isnan(np.asarray(result.value))
    assert np.isnan(np.asarray(result.gradient)).all()
    assert np.isnan(np.asarray(result.evidence.adjoint.adjoint_output)).all()
    assert int(result.evidence.consumer_reuse_counts.lu_factorization_count) == 0
    assert int(result.evidence.consumer_reuse_counts.lu_solve_count) == 12
    assert int(result.evidence.consumer_reuse_counts.refinement_correction_count) == 1
    assert int(result.evidence.full_graph_counts.dense_materialization_count) == 1
    assert int(result.evidence.full_graph_counts.lu_factorization_count) == 1


def test_closed_builder_stablehlo_exposes_dense_matrix_and_lu_factor_paths():
    device = jax.devices("cpu")[0]
    coil_dofs = jax.device_put(np.asarray([0.6, -0.35]), device)
    evaluator = _closed_evaluator(device)

    with trace_session():
        lowered = evaluator.lower(coil_dofs)
    stablehlo = lowered.compiler_ir(dialect="stablehlo").operation.get_asm(
        enable_debug_info=True,
        pretty_debug_info=True,
    )

    assert PhaseId.ADJOINT_DENSE_MATRIX.value in stablehlo
    assert PhaseId.ADJOINT_LU_FACTOR.value in stablehlo


def test_closed_runtime_accepts_only_coil_dofs():
    device = jax.devices("cpu")[0]
    evaluator = _closed_evaluator(device)
    builder_parameters = inspect.signature(
        _traceable._build_traceable_exact_payload_fused_value_and_gradient
    ).parameters
    runtime_parameters = inspect.signature(evaluator).parameters
    result_fields = _traceable._TraceableExactPayloadFusedResult._fields

    assert "producer" not in builder_parameters
    assert tuple(runtime_parameters) == ("coil_dofs",)
    assert tuple(result_fields) == ("value", "gradient", "status", "evidence")
    assert all(
        forbidden not in result_fields
        for forbidden in ("payload", "jacobian", "lu", "pivots", "solved_state")
    )
    coil_dofs = jnp.asarray([0.6, -0.35], dtype=jnp.float64)
    result = evaluator(coil_dofs)
    assert not any(leaf.ndim == 2 for leaf in jax.tree.leaves(result))
    with pytest.raises(TypeError):
        evaluator(coil_dofs, jnp.asarray(_OLD_COILS))
    with pytest.raises(TypeError):
        evaluator(coil_dofs=coil_dofs, residual_configuration=())
    with pytest.raises(TypeError):
        evaluator(coil_dofs=coil_dofs, solved_state=jnp.zeros(3))
    with pytest.raises(TypeError):
        evaluator(coil_dofs=coil_dofs, payload=object())


def test_consumer_rejects_self_consistent_stale_dynamic_snapshot():
    device = jax.devices("cpu")[0]
    coil_dofs = jax.device_put(np.asarray([0.6, -0.35]), device)
    configuration = _configuration(device)
    stale_dynamic_inputs = _dynamic_inputs_from_dofs(jax.device_put(_OLD_COILS, device))
    jacobian = configuration[0]
    state = jnp.linalg.solve(
        jacobian,
        configuration[2] - configuration[1] @ stale_dynamic_inputs[0],
    )
    inputs = _linear_solve._build_exact_final_linearization_inputs(
        solved_state=state,
        coil_dofs=coil_dofs,
        coil_dynamic_inputs=stale_dynamic_inputs,
        residual_configuration=configuration,
    )
    stale_residual = _exact_residual(
        state,
        coil_dofs,
        stale_dynamic_inputs,
        configuration,
    )
    payload = _linear_solve._build_exact_final_linearization(
        inputs,
        residual=stale_residual,
        jacobian=jacobian,
        lu_piv=jsp_linalg.lu_factor(jacobian),
        producer_solve_success=jnp.asarray(True),
    )

    consumed = _traceable._traceable_exact_payload_fused_value_and_gradient(
        payload,
        coil_dynamic_inputs_from_dofs=_dynamic_inputs_from_dofs,
        scalar_objective_fn=_scalar_objective,
        exact_residual_fn=_exact_residual,
        producer_residual_tol=1.0e-10,
        linear_solve_tol=1.0e-12,
    )

    assert not bool(consumed.success)
    assert not bool(consumed.dynamic_inputs_match)
    assert not bool(consumed.live_residual_matches_payload)
    assert np.isnan(np.asarray(consumed.value))
    assert np.isnan(np.asarray(consumed.gradient)).all()


def test_closed_graph_compiles_and_executes_with_global_transfer_guard():
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax_adapters.geo import surface_objectives_traceable as traceable

device = jax.devices("cpu")[0]
A = jax.device_put(np.asarray([[2.0, 0.5], [-0.25, 1.5]], dtype=np.float64), device)
B = jax.device_put(np.asarray([[1.0], [-0.5]], dtype=np.float64), device)
b = jax.device_put(np.asarray([0.75, -1.0], dtype=np.float64), device)
configuration = (A, B, b)

def dynamic(c):
    return (jnp.sin(c[:1]),)

def residual(x, _c, dynamic_inputs, config):
    matrix, coupling, offset = config
    return matrix @ x + coupling @ dynamic_inputs[0] - offset

def objective(x, _c, dynamic_inputs, _config):
    return 0.5 * jnp.vdot(x, x) + jnp.sum(dynamic_inputs[0] ** 2)

def returned(_c, dynamic_inputs, config):
    matrix, coupling, offset = config
    x = jnp.linalg.solve(matrix, offset - coupling @ dynamic_inputs[0])
    return traceable._TraceableExactReturnedState(x, jnp.asarray(True))

evaluate = traceable._build_traceable_exact_payload_fused_value_and_gradient(
    returned_state_from_dofs=returned,
    coil_dynamic_inputs_from_dofs=dynamic,
    scalar_objective_fn=objective,
    exact_residual_fn=residual,
    residual_configuration=configuration,
    producer_residual_tol=1.0e-12,
    linear_solve_tol=1.0e-12,
)
coils = jax.device_put(np.asarray([0.2], dtype=np.float64), device)
compiled = evaluate.lower(coils).compile()
result = compiled(coils)
jax.block_until_ready(result)
assert bool(jax.device_get(result.status.success))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cpu",
            "JAX_TRANSFER_GUARD": "disallow",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
