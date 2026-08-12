from __future__ import annotations

import inspect
from copy import copy
from dataclasses import replace

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest
from simsopt_jax.geo.optimizers import linear_solve as _linear_solve
from simsopt_jax.geo.optimizers.exact_final_linearization import (
    _ExactFinalLinearization,
    _ExactFinalLinearizationInputs,
)

_JACOBIAN = np.asarray(
    [
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [1.0, 0.0, 6.0],
    ],
    dtype=np.float64,
)


def _unsafe_substitute(value, **changes):
    """Simulate corrupted in-memory fields while bypassing the mint contract."""

    substituted = copy(value)
    for name, replacement in changes.items():
        object.__setattr__(substituted, name, replacement)
    return substituted


def _payload_inputs():
    return _linear_solve._build_exact_final_linearization_inputs(
        solved_state=jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float64),
        coil_dofs=jnp.asarray([1.5, -2.0], dtype=jnp.float64),
        coil_dynamic_inputs=(
            jnp.asarray([0.1, 0.2], dtype=jnp.float64),
            jnp.asarray([3.0], dtype=jnp.float64),
        ),
        residual_configuration=(
            jnp.asarray([5, 4, 2], dtype=jnp.int32),
            jnp.asarray([0.5, 1.25], dtype=jnp.float64),
        ),
    )


def _payload(*, jacobian: jax.Array | None = None, producer_success=True):
    if jacobian is None:
        jacobian = jnp.asarray(_JACOBIAN)
    residual = jacobian @ _payload_inputs().solved_state
    lu_piv = jsp_linalg.lu_factor(jacobian)
    return _linear_solve._build_exact_final_linearization(
        _payload_inputs(),
        residual=residual,
        jacobian=jacobian,
        lu_piv=lu_piv,
        producer_solve_success=jnp.asarray(producer_success),
    )


def test_orientation_aware_condition_estimate_reuses_j_factors() -> None:
    jacobian = jnp.asarray(_JACOBIAN)
    lu_piv = jsp_linalg.lu_factor(jacobian)

    forward, forward_factors, forward_solves = (
        _linear_solve._dense_matrix_condition_estimate_with_telemetry(
            jacobian,
            lu_piv=lu_piv,
        )
    )
    transpose, transpose_factors, transpose_solves = (
        _linear_solve._dense_matrix_condition_estimate_with_telemetry(
            jacobian,
            lu_piv=lu_piv,
            transpose_operator=True,
        )
    )

    expected_forward = np.linalg.cond(_JACOBIAN, 1)
    expected_transpose = np.linalg.cond(_JACOBIAN.T, 1)
    assert expected_forward != expected_transpose
    np.testing.assert_allclose(float(forward), expected_forward, rtol=1.0e-14)
    np.testing.assert_allclose(float(transpose), expected_transpose, rtol=1.0e-14)
    assert int(forward_factors) == 0
    assert int(transpose_factors) == 0
    assert int(forward_solves) == 10
    assert int(transpose_solves) == 10


def test_plu_factorization_residual_reconstructs_j_and_detects_corruption() -> None:
    jacobian = jnp.asarray(_JACOBIAN)
    lu, pivots = jsp_linalg.lu_factor(jacobian)

    valid_residual = _linear_solve._normalized_plu_factorization_residual(
        jacobian,
        (lu, pivots),
    )
    corrupt_residual = _linear_solve._normalized_plu_factorization_residual(
        jacobian,
        (lu.at[0, 0].add(0.5), pivots),
    )

    assert float(valid_residual) <= np.finfo(np.float64).eps
    assert bool(
        _linear_solve._plu_factorization_residual_success(
            jacobian,
            valid_residual,
        )
    )
    assert float(corrupt_residual) > 1.0e-3
    assert not bool(
        _linear_solve._plu_factorization_residual_success(
            jacobian,
            corrupt_residual,
        )
    )


def test_retained_j_transpose_solve_matches_native_refinement_and_telemetry() -> None:
    payload = _payload()
    rhs = jnp.asarray([0.75, -1.0, 2.5], dtype=jnp.float64)
    native_solution = np.linalg.solve(_JACOBIAN.T, np.asarray(rhs))
    native_correction = np.linalg.solve(
        _JACOBIAN.T,
        np.asarray(rhs) - _JACOBIAN.T @ native_solution,
    )

    result = _linear_solve._solve_retained_jacobian_transpose_adjoint(
        payload,
        rhs,
        tol=1.0e-12,
    )

    assert bool(result.payload_validation.success)
    assert int(result.payload_validation.factorization_reconstruction_count) == 1
    assert bool(result.status.success)
    np.testing.assert_allclose(
        np.asarray(result.solution),
        native_solution + native_correction,
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(result.residual),
        np.asarray(rhs) - _JACOBIAN.T @ np.asarray(result.solution),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        float(result.condition_estimate),
        np.linalg.cond(_JACOBIAN.T, 1),
        rtol=1.0e-14,
    )
    assert int(result.status.dense_materialization_count) == 0
    assert int(result.status.lu_factorization_count) == 0
    assert int(result.status.lu_solve_count) == 12
    assert int(result.status.refinement_correction_count) == 1


def test_exact_payload_is_device_pytree_and_consumer_has_no_state_coil_overrides() -> (
    None
):
    payload = _payload()
    leaves = jax.tree_util.tree_leaves(payload)
    parameters = inspect.signature(
        _linear_solve._solve_retained_jacobian_transpose_adjoint
    ).parameters

    assert isinstance(payload, _ExactFinalLinearization)
    assert isinstance(payload.inputs, _ExactFinalLinearizationInputs)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert tuple(parameters) == ("payload", "rhs", "tol")
    with pytest.raises(TypeError):
        _linear_solve._solve_retained_jacobian_transpose_adjoint(
            payload,
            jnp.ones(3, dtype=jnp.float64),
            tol=1.0e-12,
            solved_state=jnp.zeros(3, dtype=jnp.float64),
        )
    with pytest.raises(RuntimeError, match="must be minted atomically"):
        _ExactFinalLinearizationInputs(
            solved_state=jnp.zeros(3),
            coil_dofs=jnp.zeros(2),
            coil_dynamic_inputs=(),
            residual_configuration=(),
        )
    with pytest.raises(RuntimeError, match="must be minted by their producer"):
        replace(payload, residual=payload.residual + 1.0)


@pytest.mark.parametrize(
    "replacement",
    (
        "solved_state",
        "coil_dofs",
        "coil_dynamic_inputs",
        "residual_configuration",
        "residual",
        "jacobian",
        "orientation",
    ),
)
def test_stale_or_replaced_payload_identity_fails_closed(replacement: str) -> None:
    payload = _payload()
    inputs = payload.inputs
    if replacement == "solved_state":
        payload = _unsafe_substitute(
            payload,
            inputs=_unsafe_substitute(
                inputs,
                solved_state=inputs.solved_state + 0.125,
            ),
        )
    elif replacement == "coil_dofs":
        payload = _unsafe_substitute(
            payload,
            inputs=_unsafe_substitute(
                inputs,
                coil_dofs=inputs.coil_dofs - 0.25,
            ),
        )
    elif replacement == "coil_dynamic_inputs":
        payload = _unsafe_substitute(
            payload,
            inputs=_unsafe_substitute(
                inputs,
                coil_dynamic_inputs=(
                    inputs.coil_dynamic_inputs[0] + 0.5,
                    inputs.coil_dynamic_inputs[1],
                ),
            ),
        )
    elif replacement == "residual_configuration":
        payload = _unsafe_substitute(
            payload,
            inputs=_unsafe_substitute(
                inputs,
                residual_configuration=(
                    inputs.residual_configuration[0],
                    inputs.residual_configuration[1] + 0.75,
                ),
            ),
        )
    elif replacement == "residual":
        payload = _unsafe_substitute(payload, residual=payload.residual + 0.25)
    elif replacement == "jacobian":
        payload = _unsafe_substitute(
            payload,
            jacobian=payload.jacobian.at[0, 1].add(0.5),
        )
    else:
        payload = _unsafe_substitute(
            payload,
            orientation_code=payload.orientation_code + 1,
        )

    result = _linear_solve._solve_retained_jacobian_transpose_adjoint(
        payload,
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64),
        tol=1.0e-12,
    )

    assert not bool(result.payload_validation.success)
    assert not bool(result.status.success)
    assert np.all(np.isnan(np.asarray(result.solution)))
    assert int(result.status.lu_factorization_count) == 0
    assert int(result.status.lu_solve_count) == 0


def test_corrupt_factors_and_failed_producer_status_are_rejected() -> None:
    payload = _payload()
    corrupt = _unsafe_substitute(payload, lu=payload.lu.at[0, 0].add(0.5))
    wrong_dtype = _unsafe_substitute(payload, lu=payload.lu.astype(jnp.float32))
    wrong_pivot_dtype = _unsafe_substitute(
        payload,
        pivots=payload.pivots.astype(jnp.int64),
    )
    replaced_float32_payload = _unsafe_substitute(
        payload,
        residual=payload.residual.astype(jnp.float32),
        jacobian=payload.jacobian.astype(jnp.float32),
        lu=payload.lu.astype(jnp.float32),
    )
    failed_producer = _unsafe_substitute(
        payload,
        producer_solve_success=jnp.asarray(False),
    )
    rhs = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float64)

    corrupt_result = _linear_solve._solve_retained_jacobian_transpose_adjoint(
        corrupt,
        rhs,
        tol=1.0e-12,
    )
    failed_result = _linear_solve._solve_retained_jacobian_transpose_adjoint(
        failed_producer,
        rhs,
        tol=1.0e-12,
    )
    wrong_dtype_validation = _linear_solve._validate_exact_final_linearization(
        wrong_dtype
    )
    replaced_float32_validation = _linear_solve._validate_exact_final_linearization(
        replaced_float32_payload
    )
    wrong_pivot_validation = _linear_solve._validate_exact_final_linearization(
        wrong_pivot_dtype
    )

    assert bool(corrupt_result.payload_validation.identity_valid)
    assert not bool(corrupt_result.payload_validation.factorization_valid)
    assert (
        int(corrupt_result.payload_validation.factorization_reconstruction_count) == 1
    )
    assert not bool(corrupt_result.status.success)
    assert not bool(wrong_dtype_validation.factor_metadata_valid)
    assert not bool(wrong_dtype_validation.success)
    assert not bool(replaced_float32_validation.identity_valid)
    assert not bool(replaced_float32_validation.success)
    assert not bool(wrong_pivot_validation.factor_metadata_valid)
    assert not bool(wrong_pivot_validation.success)
    assert not bool(failed_result.payload_validation.success)
    assert not bool(failed_result.status.success)


def test_exact_identity_rejects_reduction_collision_and_false_to_true_status() -> None:
    inputs = _linear_solve._build_exact_final_linearization_inputs(
        solved_state=jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float64),
        coil_dofs=jnp.asarray([1.5, -2.0], dtype=jnp.float64),
        coil_dynamic_inputs=(jnp.asarray([10.0, 10.0, 10.0, 10.0], dtype=jnp.float64),),
        residual_configuration=(jnp.asarray([5, 4, 2], dtype=jnp.int32),),
    )
    jacobian = jnp.asarray(_JACOBIAN)
    payload = _linear_solve._build_exact_final_linearization(
        inputs,
        residual=jacobian @ inputs.solved_state,
        jacobian=jacobian,
        lu_piv=jsp_linalg.lu_factor(jacobian),
        producer_solve_success=jnp.asarray(True),
    )
    colliding_reduction_inputs = _unsafe_substitute(
        inputs,
        coil_dynamic_inputs=(jnp.asarray([9.0, 13.0, 7.0, 11.0], dtype=jnp.float64),),
    )
    stale_payload = _unsafe_substitute(payload, inputs=colliding_reduction_inputs)
    built_failed = _payload(producer_success=False)
    forged_success = _unsafe_substitute(
        built_failed,
        producer_solve_success=jnp.asarray(True),
    )

    stale_validation = _linear_solve._validate_exact_final_linearization(stale_payload)
    forged_validation = _linear_solve._validate_exact_final_linearization(
        forged_success
    )

    assert not bool(stale_validation.identity_valid)
    assert not bool(stale_validation.success)
    assert not bool(forged_validation.identity_valid)
    assert not bool(forged_validation.success)


def test_retained_transpose_adjoint_is_strict_transfer_clean() -> None:
    device = jax.devices("cpu")[0]
    jacobian = jax.device_put(_JACOBIAN, device)
    solved_state = jax.device_put(np.asarray([0.25, -0.5, 0.75]), device)
    coil_dofs = jax.device_put(np.asarray([1.5, -2.0]), device)
    dynamic = jax.device_put(np.asarray([0.1, 0.2]), device)
    configuration = jax.device_put(np.asarray([5, 4, 2], dtype=np.int32), device)
    rhs = jax.device_put(np.asarray([0.75, -1.0, 2.5]), device)

    @jax.jit
    def produce_and_consume(
        current_jacobian,
        current_state,
        current_coils,
        current_dynamic,
        current_configuration,
        current_rhs,
    ):
        inputs = _linear_solve._build_exact_final_linearization_inputs(
            solved_state=current_state,
            coil_dofs=current_coils,
            coil_dynamic_inputs=(current_dynamic,),
            residual_configuration=(current_configuration,),
        )
        payload = _linear_solve._build_exact_final_linearization(
            inputs,
            residual=current_jacobian @ current_state,
            jacobian=current_jacobian,
            lu_piv=jsp_linalg.lu_factor(current_jacobian),
            producer_solve_success=jnp.asarray(True),
        )
        return _linear_solve._solve_retained_jacobian_transpose_adjoint(
            payload,
            current_rhs,
            tol=1.0e-12,
        )

    with jax.transfer_guard("disallow"):
        result = produce_and_consume(
            jacobian,
            solved_state,
            coil_dofs,
            dynamic,
            configuration,
            rhs,
        )
        jax.block_until_ready(result)

    assert bool(result.status.success)
    assert int(result.status.lu_factorization_count) == 0
    assert int(result.status.lu_solve_count) == 12


def test_zero_rhs_preserves_validated_payload_fast_path() -> None:
    result = _linear_solve._solve_retained_jacobian_transpose_adjoint(
        _payload(),
        jnp.zeros(3, dtype=jnp.float64),
        tol=1.0e-12,
    )

    assert bool(result.payload_validation.success)
    assert bool(result.status.success)
    np.testing.assert_array_equal(np.asarray(result.solution), np.zeros(3))
    assert int(result.status.lu_factorization_count) == 0
    assert int(result.status.lu_solve_count) == 0
    assert int(result.status.refinement_correction_count) == 0
