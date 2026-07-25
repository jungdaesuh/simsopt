"""Strict-CUDA coverage for dense optimizer placement contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_strict_parity_backend, parity_default_device
from simsopt_jax.backend import invalidate_backend_cache
from simsopt_jax.geo.optimizers import optimizer as _optimizer
from simsopt_jax.geo.optimizers.private import _bfgs as _private_bfgs
from simsopt_jax.geo.optimizers.private import _lbfgs as _private_lbfgs


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


def test_traceable_dense_ir_polish_stays_on_gpu_under_strict_transfer_guard(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    gpu = _gpu_device()
    x0 = jax.device_put(
        np.full(5, 2.0e-5 / np.sqrt(5.0), dtype=np.float64),
        gpu,
    )

    def objective(candidate):
        return jnp.dot(candidate, candidate)

    with parity_default_device("gpu"), jax.transfer_guard("disallow"):
        result = _optimizer.newton_polish_traceable(
            objective,
            x0,
            maxiter=4,
            tol=1.0e-6,
            stab=0.0,
            materialize_hessian=False,
            linear_solver="hybrid_final_dense_ir",
        )
        jax.block_until_ready(result)

    active = np.asarray(result["newton_trace_active"], dtype=bool)
    backend_codes = np.asarray(result["newton_trace_linear_solve_backend_code"])[active]
    dense_ir_code = _optimizer._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
        _optimizer._TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR
    ]
    assert result["x"].devices() == {gpu}
    assert bool(result["success"])
    np.testing.assert_array_equal(
        backend_codes,
        np.asarray([dense_ir_code], dtype=np.int32),
    )


def test_private_quasi_newton_compute_state_stays_fp32_on_gpu(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    monkeypatch.setenv("SIMSOPT_PRECISION", "mixed")
    invalidate_backend_cache()
    gpu = _gpu_device()
    x0 = jax.device_put(np.asarray([1.0, -2.0], dtype=np.float64), gpu)

    def objective(candidate):
        one = jnp.exp(jnp.sum(candidate - candidate))
        half = one / (one + one)
        return half * jnp.dot(candidate, candidate)

    with parity_default_device("gpu"), jax.transfer_guard("disallow"):
        bfgs_state = _private_bfgs._minimize_bfgs_private(
            objective,
            x0,
            maxiter=10,
            gtol=1.0e-5,
            x_dtype=jnp.float32,
        )
        lbfgs_state = _private_lbfgs._minimize_lbfgs_private(
            objective,
            x0,
            maxiter=10,
            gtol=1.0e-5,
            maxcor=5,
            x_dtype=jnp.float32,
        )
        jax.block_until_ready((bfgs_state, lbfgs_state))

    assert bfgs_state.x_k.dtype == jnp.float32
    assert bfgs_state.g_k.dtype == jnp.float32
    assert bfgs_state.H_k.dtype == jnp.float32
    assert lbfgs_state.x_k.dtype == jnp.float32
    assert lbfgs_state.g_k.dtype == jnp.float32
    assert bfgs_state.x_k.devices() == {gpu}
    assert lbfgs_state.x_k.devices() == {gpu}
