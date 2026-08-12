from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers import linear_solve as _linear_solve


def _nonsymmetric_residual(values: jax.Array) -> jax.Array:
    dimension = values.shape[0]
    diagonal = jnp.arange(1, dimension + 1, dtype=values.dtype)
    return (
        diagonal * values
        + jnp.asarray(0.125, dtype=values.dtype) * values**2
        + jnp.asarray(0.25, dtype=values.dtype) * jnp.roll(values, 1)
    )


@pytest.mark.parametrize("dimension", (7, 8, 15, 16, 255, 256))
def test_linearize_once_dense_jacobian_matches_forward_oracle_with_exact_tail(
    dimension: int,
) -> None:
    state = jnp.linspace(-0.5, 0.75, dimension, dtype=jnp.float64)

    actual = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        _nonsymmetric_residual,
        state,
    )
    expected = jax.jacfwd(_nonsymmetric_residual)(state)
    checkpointed = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        _nonsymmetric_residual,
        state,
        assembler=_linear_solve._DenseJacobianAssembler.CHECKPOINTED_BATCHED_JVP,
        batch_width=8,
    )

    assert actual.jacobian.shape == (dimension, dimension)
    assert actual.jacobian.dtype == jnp.float64
    np.testing.assert_array_equal(
        np.asarray(actual.residual),
        np.asarray(_nonsymmetric_residual(state)),
    )
    np.testing.assert_array_equal(np.asarray(actual.jacobian), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(actual.jacobian),
        np.asarray(checkpointed.jacobian),
    )
    expected_batches = math.ceil(dimension / 8)
    assert int(checkpointed.telemetry.assembler_code) == int(
        _linear_solve._DenseJacobianAssembler.CHECKPOINTED_BATCHED_JVP
    )
    assert int(checkpointed.telemetry.residual_evaluation_count) == 1
    assert int(checkpointed.telemetry.primal_traversal_count) == 1 + expected_batches
    assert int(checkpointed.telemetry.tangent_batch_count) == expected_batches
    assert int(checkpointed.telemetry.tangent_direction_count) == dimension
    assert int(checkpointed.telemetry.batch_width) == 8
    assert int(checkpointed.telemetry.tail_width) == dimension % 8


@pytest.mark.parametrize("dimension", (7, 8, 15, 16, 255, 256))
def test_linearized_operator_visits_each_exact_basis_direction_once(
    dimension: int,
) -> None:
    diagonal = jnp.arange(1, dimension + 1, dtype=jnp.float64)
    observed_basis: list[np.ndarray] = []

    def observed_operator(tangent: jax.Array) -> jax.Array:
        jax.debug.callback(
            lambda value: observed_basis.append(np.asarray(value).copy()),
            tangent,
        )
        return diagonal * tangent

    actual = _linear_solve._dense_square_operator_matrix(
        observed_operator,
        diagonal,
        matrix_dtype=jnp.float64,
        sweep_dtype=jnp.float64,
    )
    jax.block_until_ready(actual)

    assert len(observed_basis) == dimension
    observed = np.stack(observed_basis)
    np.testing.assert_array_equal(
        np.sort(np.argmax(observed, axis=1)),
        np.arange(dimension),
    )
    np.testing.assert_array_equal(np.sum(observed, axis=1), np.ones(dimension))
    assert np.all((observed == 0.0) | (observed == 1.0))


def test_linearize_retains_one_plain_residual_pushforward_across_tangent_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dimension = 15
    state = jnp.linspace(-0.5, 0.75, dimension, dtype=jnp.float64)
    linearize_primal_calls: list[np.ndarray] = []
    checkpointed_primal_calls: list[np.ndarray] = []
    linearized_tangent_calls: list[np.ndarray] = []

    def linearized_residual(values: jax.Array) -> jax.Array:
        jax.debug.callback(
            lambda value: linearize_primal_calls.append(np.asarray(value).copy()),
            values,
        )
        return _nonsymmetric_residual(values)

    def checkpointed_residual(values: jax.Array) -> jax.Array:
        jax.debug.callback(
            lambda value: checkpointed_primal_calls.append(np.asarray(value).copy()),
            values,
        )
        return _nonsymmetric_residual(values)

    original_materialize = _linear_solve._dense_square_operator_matrix

    def observed_materialize(matvec, rhs, **kwargs):
        def observed_matvec(tangent: jax.Array) -> jax.Array:
            jax.debug.callback(
                lambda value: linearized_tangent_calls.append(np.asarray(value).copy()),
                tangent,
            )
            return matvec(tangent)

        return original_materialize(observed_matvec, rhs, **kwargs)

    monkeypatch.setattr(
        _linear_solve,
        "_dense_square_operator_matrix",
        observed_materialize,
    )
    linearized = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        linearized_residual,
        state,
    )
    jax.block_until_ready(linearized)
    jax.effects_barrier()
    monkeypatch.setattr(
        _linear_solve,
        "_dense_square_operator_matrix",
        original_materialize,
    )
    checkpointed = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        checkpointed_residual,
        state,
        assembler=_linear_solve._DenseJacobianAssembler.CHECKPOINTED_BATCHED_JVP,
        batch_width=8,
    )
    jax.block_until_ready(checkpointed)
    jax.effects_barrier()

    np.testing.assert_array_equal(
        np.asarray(linearized.jacobian),
        np.asarray(checkpointed.jacobian),
    )
    assert len(linearize_primal_calls) == 1
    assert len(linearized_tangent_calls) == dimension
    assert len(checkpointed_primal_calls) == 1 + math.ceil(dimension / 8)


def test_checkpoint_rematerialization_is_not_reported_as_one_residual_execution() -> (
    None
):
    dimension = 15
    state = jnp.linspace(-0.5, 0.75, dimension, dtype=jnp.float64)
    residual_executions: list[np.ndarray] = []

    def observed_residual(values: jax.Array) -> jax.Array:
        jax.debug.callback(
            lambda value: residual_executions.append(np.asarray(value).copy()),
            values,
        )
        return _nonsymmetric_residual(values)

    checkpointed_residual = jax.checkpoint(
        observed_residual,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=False,
    )
    actual = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        checkpointed_residual,
        state,
    )
    jax.block_until_ready(actual)
    jax.effects_barrier()

    assert len(residual_executions) > 1
    assert actual._fields == ("residual", "jacobian", "telemetry")


@pytest.mark.parametrize(
    "assembler",
    (
        _linear_solve._DenseJacobianAssembler.LINEARIZE_ONCE,
        _linear_solve._DenseJacobianAssembler.CHECKPOINTED_BATCHED_JVP,
    ),
)
def test_dense_jacobian_assemblers_are_jittable_and_strict_transfer_clean(
    assembler: _linear_solve._DenseJacobianAssembler,
) -> None:
    dimension = 15
    state = jax.device_put(
        np.linspace(-0.5, 0.75, dimension, dtype=np.float64),
    )
    materialize = jax.jit(
        lambda values: _linear_solve._linearize_and_materialize_dense_square_jacobian(
            _nonsymmetric_residual,
            values,
            assembler=assembler,
            batch_width=8,
        )
    )

    with jax.transfer_guard("disallow"):
        actual = materialize(state)
        jax.block_until_ready(actual)

    expected = jax.jacfwd(_nonsymmetric_residual)(state)
    np.testing.assert_array_equal(
        np.asarray(actual.jacobian),
        np.asarray(expected),
    )


def test_dense_jacobian_assembler_default_remains_linearize_once() -> None:
    dimension = 15
    state = jnp.linspace(-0.5, 0.75, dimension, dtype=jnp.float64)

    actual = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        _nonsymmetric_residual,
        state,
        batch_width=8,
    )

    assert int(actual.telemetry.assembler_code) == int(
        _linear_solve._DenseJacobianAssembler.LINEARIZE_ONCE
    )
    assert int(actual.telemetry.residual_evaluation_count) == 1
    assert int(actual.telemetry.primal_traversal_count) == 1
    assert int(actual.telemetry.tangent_batch_count) == 2
    assert int(actual.telemetry.tangent_direction_count) == dimension
    assert int(actual.telemetry.batch_width) == 8
    assert int(actual.telemetry.tail_width) == 7


@pytest.mark.parametrize("batch_width", (0, -1))
def test_dense_jacobian_assembler_rejects_nonpositive_batch_width(
    batch_width: int,
) -> None:
    state = jnp.ones(7, dtype=jnp.float64)

    with pytest.raises(ValueError, match="batch_width must be positive"):
        _linear_solve._linearize_and_materialize_dense_square_jacobian(
            _nonsymmetric_residual,
            state,
            batch_width=batch_width,
        )


@pytest.mark.parametrize("invalid_assembler", ("linearize_once", 0, None))
def test_dense_jacobian_assembler_rejects_untyped_selector(
    invalid_assembler: object,
) -> None:
    state = jnp.ones(7, dtype=jnp.float64)

    with pytest.raises(TypeError, match="assembler must be a _DenseJacobianAssembler"):
        _linear_solve._linearize_and_materialize_dense_square_jacobian(
            _nonsymmetric_residual,
            state,
            assembler=invalid_assembler,
        )


def test_linearize_once_rebuilds_the_fixed_state_jacobian_for_changed_state() -> None:
    first_state = jnp.linspace(-0.5, 0.75, 15, dtype=jnp.float64)
    second_state = first_state + jnp.asarray(0.125, dtype=jnp.float64)

    first = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        _nonsymmetric_residual,
        first_state,
    )
    second = _linear_solve._linearize_and_materialize_dense_square_jacobian(
        _nonsymmetric_residual,
        second_state,
    )

    np.testing.assert_array_equal(
        np.asarray(second.jacobian),
        np.asarray(jax.jacfwd(_nonsymmetric_residual)(second_state)),
    )
    assert not np.array_equal(np.asarray(first.jacobian), np.asarray(second.jacobian))
