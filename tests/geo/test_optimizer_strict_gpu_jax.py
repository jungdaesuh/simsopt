"""Strict-CUDA coverage for dense optimizer placement contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_strict_parity_backend, parity_default_device
from simsopt_jax.geo.optimizers import optimizer as _optimizer


def _gpu_device() -> jax.Device:
    return next(device for device in jax.devices() if device.platform == "gpu")


def test_dense_hessian_solve_colocates_cpu_state_with_gpu_rhs(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    dimension = _optimizer.dense_operator_chunk_batch_size() + 1
    cpu = jax.devices("cpu")[0]
    gpu = _gpu_device()
    diagonal_values = np.arange(1.0, dimension + 1.0, dtype=np.float64)
    diagonal = jax.device_put(diagonal_values, gpu)
    state = jax.device_put(np.ones(dimension, dtype=np.float64), cpu)
    rhs = jax.device_put(np.ones(dimension, dtype=np.float64), gpu)
    monkeypatch.setattr(_optimizer, "_ADJOINT_LINEAR_SOLVER", "dense")

    def objective(candidate):
        return jnp.sum(diagonal * candidate * candidate)

    with parity_default_device("gpu"), jax.transfer_guard("disallow"):
        solution, status = _optimizer._solve_hessian_least_squares_system_with_status(
            objective,
            state,
            rhs,
            stab=0.0,
            tol=1.0e-12,
        )
        jax.block_until_ready((solution, status))

    assert bool(status.success)
    np.testing.assert_allclose(
        np.asarray(solution),
        1.0 / (2.0 * diagonal_values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dense_hessian_solve_colocates_cpu_objective_closure_with_gpu_rhs(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    dimension = _optimizer.dense_operator_chunk_batch_size() + 1
    cpu = jax.devices("cpu")[0]
    gpu = _gpu_device()
    closure_weights = jax.device_put(
        np.arange(1.0, 12.0, dtype=np.float64),
        cpu,
    )
    state = jax.device_put(np.ones(dimension, dtype=np.float64), gpu)
    rhs = jax.device_put(np.ones(dimension, dtype=np.float64), gpu)
    monkeypatch.setattr(_optimizer, "_ADJOINT_LINEAR_SOLVER", "dense")

    def objective(candidate):
        return jnp.sum(closure_weights) * jnp.sum(candidate * candidate)

    with parity_default_device("gpu"), jax.transfer_guard("disallow"):
        solution, status = _optimizer._solve_hessian_least_squares_system_with_status(
            objective,
            state,
            rhs,
            stab=0.0,
            tol=1.0e-12,
        )
        jax.block_until_ready((solution, status))

    assert bool(status.success)
    expected = np.full(
        dimension,
        1.0 / (2.0 * np.sum(np.arange(1.0, 12.0, dtype=np.float64))),
    )
    np.testing.assert_allclose(
        np.asarray(solution),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dense_condition_estimate_aligns_cached_cpu_factors_with_gpu_result(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    cpu = jax.devices("cpu")[0]
    gpu = _gpu_device()
    diagonal = np.geomspace(1.0, 1.0e-5, 32)
    matrix = jax.device_put(jnp.diag(jnp.asarray(diagonal)), gpu)
    lu_piv_gpu = _optimizer.jsp_linalg.lu_factor(matrix)
    lu_piv_cpu = tuple(jax.device_put(factor, cpu) for factor in lu_piv_gpu)

    with parity_default_device("gpu"), jax.transfer_guard("disallow"):
        estimate = _optimizer._dense_matrix_condition_estimate(
            matrix,
            lu_piv=lu_piv_cpu,
        )
        jax.block_until_ready(estimate)

    assert all(factor.devices() == {cpu} for factor in lu_piv_cpu)
    assert estimate.devices() == {gpu}
    assert float(np.asarray(estimate)) == pytest.approx(1.0e5, rel=1.0e-10)
