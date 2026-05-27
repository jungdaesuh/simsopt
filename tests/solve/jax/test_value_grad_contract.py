import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsopt.config as simsopt_config
import simsopt.solve.jax._dispatch as dispatch
from simsopt.solve.jax import (
    Driver,
    OptimistixLBFGSOptions,
    OptimistixLMOptions,
    OptaxAdamOptions,
    OptaxLBFGSOptions,
    ScipyBFGSOptions,
    SimsoptBFGSOptions,
    least_squares,
    minimize,
)


def test_scipy_driver_passes_host_numpy_array_to_value_grad():
    seen_types = []

    def value_and_grad(x):
        seen_types.append(type(x))
        residual = x - np.array([1.0, -2.0])
        return float(np.dot(residual, residual)), 2.0 * residual

    result = minimize(
        value_and_grad,
        np.array([0.0, 0.0]),
        driver=Driver.SCIPY_BFGS,
        options=ScipyBFGSOptions(maxiter=5),
    )

    assert result.success
    assert np.allclose(result.x, np.array([1.0, -2.0]))
    assert seen_types and set(seen_types) == {np.ndarray}


def test_optax_driver_passes_jax_array_to_value_grad():
    seen_jax_array = []

    def value_and_grad(x):
        seen_jax_array.append(isinstance(x, jax.Array))
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    result = minimize(
        value_and_grad,
        jnp.array([0.0, 0.0]),
        driver=Driver.OPTAX_ADAM,
        options=OptaxAdamOptions(maxiter=2, learning_rate=0.1),
    )

    assert result.driver is Driver.OPTAX_ADAM
    assert all(seen_jax_array)


def test_optax_adam_accepts_eager_host_materializing_value_grad():
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    target = np.asarray([1.0, -2.0], dtype=dtype)
    seen_jax_array = []

    def value_and_grad(x):
        seen_jax_array.append(isinstance(x, jax.Array))
        residual = np.asarray(x, dtype=dtype) - target
        return (
            np.asarray(np.vdot(residual, residual), dtype=dtype),
            np.asarray(2.0 * residual, dtype=dtype),
        )

    result = minimize(
        value_and_grad,
        jax.device_put(np.asarray([0.0, 0.0], dtype=dtype)),
        driver=Driver.OPTAX_ADAM,
        options=OptaxAdamOptions(maxiter=1, learning_rate=0.1),
    )

    assert result.driver is Driver.OPTAX_ADAM
    assert all(seen_jax_array)


def test_optax_lbfgs_runs_full_value_grad_driver():
    target = jnp.array([1.0, -2.0], dtype=jnp.float64)
    seen_jax_array = []

    def value_and_grad(x):
        seen_jax_array.append(isinstance(x, jax.Array))
        residual = x - target
        return jnp.vdot(residual, residual), 2.0 * residual

    result = minimize(
        value_and_grad,
        jnp.array([0.0, 0.0], dtype=jnp.float64),
        driver=Driver.OPTAX_LBFGS,
        options=OptaxLBFGSOptions(
            maxiter=12,
            gtol=1e-8,
            memory_size=3,
            max_linesearch_steps=10,
        ),
    )

    assert result.driver is Driver.OPTAX_LBFGS
    assert result.success is True
    assert all(seen_jax_array)
    np.testing.assert_allclose(result.x, np.asarray(target), rtol=1e-7, atol=1e-7)


def test_optax_lbfgs_uses_explicit_vjp_for_io_callback_value_grad():
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    target = np.asarray([1.0, -2.0], dtype=dtype)
    value_spec = jax.ShapeDtypeStruct((), jnp.dtype(dtype))
    grad_spec = jax.ShapeDtypeStruct(target.shape, jnp.dtype(dtype))

    def host_value_and_grad(x_host):
        residual = np.asarray(x_host, dtype=dtype) - target
        return (
            np.asarray(np.vdot(residual, residual), dtype=dtype),
            np.asarray(2.0 * residual, dtype=dtype),
        )

    def value_and_grad(x):
        return jax.experimental.io_callback(
            host_value_and_grad,
            (value_spec, grad_spec),
            x,
            ordered=True,
        )

    result = minimize(
        value_and_grad,
        jax.device_put(np.asarray([0.0, 0.0], dtype=dtype)),
        driver=Driver.OPTAX_LBFGS,
        options=OptaxLBFGSOptions(
            maxiter=2,
            gtol=1e-8,
            memory_size=3,
            max_linesearch_steps=5,
        ),
    )

    assert result.driver is Driver.OPTAX_LBFGS
    assert result.x.shape == target.shape


def test_optax_lbfgs_uses_quasi_newton_line_search_initial_guess(monkeypatch):
    observed = {}
    real_scale_by_zoom_linesearch = dispatch.optax.scale_by_zoom_linesearch

    def recording_scale_by_zoom_linesearch(*args, **kwargs):
        observed.update(kwargs)
        return real_scale_by_zoom_linesearch(*args, **kwargs)

    monkeypatch.setattr(
        dispatch.optax,
        "scale_by_zoom_linesearch",
        recording_scale_by_zoom_linesearch,
    )

    minimize(
        lambda x: (jnp.vdot(x, x), 2.0 * x),
        jnp.array([1.0, -2.0], dtype=jnp.float64),
        driver=Driver.OPTAX_LBFGS,
        options=OptaxLBFGSOptions(maxiter=0, max_linesearch_steps=7),
    )

    assert observed["max_linesearch_steps"] == 7
    assert observed["initial_guess_strategy"] == "one"


def test_optax_numpy_x0_uses_explicit_device_put_under_strict_transfer_guard():
    target = jax.device_put(np.asarray([1.0, -2.0], dtype=np.float32))
    two = jax.device_put(np.asarray(2.0, dtype=np.float32))
    seen_jax_array = []

    def value_and_grad(x):
        seen_jax_array.append(isinstance(x, jax.Array))
        residual = x - target
        return jnp.vdot(residual, residual), two * residual

    with jax.transfer_guard_host_to_device("disallow"):
        result = minimize(
            value_and_grad,
            np.asarray([0.0, 0.0], dtype=np.float32),
            driver=Driver.OPTAX_ADAM,
            options=OptaxAdamOptions(maxiter=1, learning_rate=0.1),
        )

    assert result.driver is Driver.OPTAX_ADAM
    assert all(seen_jax_array)


def test_optax_lbfgs_gpu_closure_constants_run_under_strict_transfer_guard():
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CUDA device required for device-to-host transfer-guard proof.")
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    x0 = jax.device_put(np.asarray([0.0, 0.0], dtype=dtype), device)
    target = jax.device_put(np.asarray([1.0, -2.0], dtype=dtype), device)
    active = jax.device_put(np.asarray(True), device)
    two = jax.device_put(np.asarray(2.0, dtype=dtype), device)

    def value_and_grad(x):
        residual = jnp.where(active, x - target, x)
        return jnp.vdot(residual, residual), two * residual

    with jax.transfer_guard("disallow"):
        result = minimize(
            value_and_grad,
            x0,
            driver=Driver.OPTAX_LBFGS,
            options=OptaxLBFGSOptions(maxiter=1, max_linesearch_steps=3),
        )

    assert result.driver is Driver.OPTAX_LBFGS
    assert result.x.shape == (2,)


def test_optimistix_numpy_x0_uses_explicit_device_put(monkeypatch):
    target = jax.device_put(np.asarray([1.0, -2.0], dtype=np.float64))
    x0 = np.asarray([0.0, 0.0], dtype=np.float64)
    staged_numpy_inputs = []

    real_device_put = dispatch.jax.device_put

    def recording_device_put(value, *args, **kwargs):
        if isinstance(value, np.ndarray):
            staged_numpy_inputs.append(value)
        return real_device_put(value, *args, **kwargs)

    def residual(x):
        return x - target

    monkeypatch.setattr(dispatch.jax, "device_put", recording_device_put)
    result = least_squares(
        residual,
        x0,
        driver=Driver.OPTIMISTIX_LM,
        options=OptimistixLMOptions(maxiter=1),
    )

    assert result.driver is Driver.OPTIMISTIX_LM
    assert result.residual is not None
    assert staged_numpy_inputs[0] is x0


def test_optimistix_lbfgs_runs_full_value_grad_driver():
    target = jax.device_put(np.asarray([1.0, -2.0], dtype=np.float64))
    seen_jax_array = []

    def value_and_grad(x):
        seen_jax_array.append(isinstance(x, jax.Array))
        residual = x - target
        return jnp.vdot(residual, residual), 2.0 * residual

    result = minimize(
        value_and_grad,
        np.asarray([0.0, 0.0], dtype=np.float64),
        driver=Driver.OPTIMISTIX_LBFGS,
        options=OptimistixLBFGSOptions(maxiter=12, tol=1e-8, history_length=3),
    )

    assert result.driver is Driver.OPTIMISTIX_LBFGS
    assert result.success is True
    assert all(seen_jax_array)
    np.testing.assert_allclose(result.x, np.asarray(target), rtol=1e-7, atol=1e-7)


def test_optimistix_lbfgs_uses_supplied_gradient_contract():
    target = jax.device_put(np.asarray([1.0, -2.0], dtype=np.float64))

    def value_and_grad(x):
        residual = x - target
        return jax.lax.stop_gradient(jnp.vdot(residual, residual)), 2.0 * residual

    result = minimize(
        value_and_grad,
        np.asarray([0.0, 0.0], dtype=np.float64),
        driver=Driver.OPTIMISTIX_LBFGS,
        options=OptimistixLBFGSOptions(maxiter=12, tol=1e-8, history_length=3),
    )

    assert result.driver is Driver.OPTIMISTIX_LBFGS
    assert result.success is True
    np.testing.assert_allclose(result.x, np.asarray(target), rtol=1e-7, atol=1e-7)


def test_optimistix_driver_runs_under_strict_host_to_device_transfer_guard():
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    x0 = jax.device_put(np.asarray([1.0, -2.0], dtype=dtype))

    with jax.transfer_guard_host_to_device("disallow"):
        result = least_squares(
            lambda x: x,
            x0,
            driver=Driver.OPTIMISTIX_LM,
            options=OptimistixLMOptions(
                maxiter=1,
                materialize_dense_linearization=False,
            ),
        )

    assert result.driver is Driver.OPTIMISTIX_LM
    assert result.x.shape == (2,)


def test_optimistix_driver_runs_under_strict_transfer_guard():
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    x0 = jax.device_put(np.asarray([1.0, -2.0], dtype=dtype))

    with jax.transfer_guard("disallow"):
        result = least_squares(
            lambda x: x,
            x0,
            driver=Driver.OPTIMISTIX_LM,
            options=OptimistixLMOptions(
                maxiter=1,
                materialize_dense_linearization=False,
            ),
        )

    assert result.driver is Driver.OPTIMISTIX_LM
    assert result.x.shape == (2,)


def test_optimistix_lbfgs_runs_under_strict_host_to_device_transfer_guard():
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    x0 = jax.device_put(np.asarray([1.0, -2.0], dtype=dtype))
    two = jax.device_put(np.asarray(2.0, dtype=dtype))

    with jax.transfer_guard_host_to_device("disallow"):
        result = minimize(
            lambda x: (jnp.vdot(x, x), two * x),
            x0,
            driver=Driver.OPTIMISTIX_LBFGS,
            options=OptimistixLBFGSOptions(maxiter=1),
        )

    assert result.driver is Driver.OPTIMISTIX_LBFGS
    assert result.x.shape == (2,)


@pytest.mark.xfail(
    raises=jax.errors.JaxRuntimeError,
    strict=True,
    reason=(
        "Optimistix/Equinox scalar predicate handling is not CUDA "
        "device-to-host transfer clean under full jax.transfer_guard('disallow')."
    ),
)
def test_optimistix_lbfgs_gpu_full_transfer_guard_upstream_predicate_xfail():
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CUDA device required for full strict-transfer predicate proof.")
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    x0 = jax.device_put(np.asarray([1.0, -2.0], dtype=dtype), device)
    two = jax.device_put(np.asarray(2.0, dtype=dtype), device)

    with jax.transfer_guard("disallow"):
        minimize(
            lambda x: (jnp.vdot(x, x), two * x),
            x0,
            driver=Driver.OPTIMISTIX_LBFGS,
            options=OptimistixLBFGSOptions(maxiter=1),
        )


def test_optimistix_metadata_uses_host_to_device_transfer_guard(monkeypatch):
    events = []

    def broad_transfer_guard(_level):
        raise AssertionError("optimistix metadata must not use broad transfer_guard")

    def device_to_host_transfer_guard(_level):
        raise AssertionError(
            "optimistix metadata must not allow device-to-host broadly"
        )

    @contextlib.contextmanager
    def host_to_device_transfer_guard(level):
        events.append(level)
        yield

    monkeypatch.setattr(dispatch.jax, "transfer_guard", broad_transfer_guard)
    monkeypatch.setattr(
        dispatch.jax,
        "transfer_guard_device_to_host",
        device_to_host_transfer_guard,
    )
    monkeypatch.setattr(
        dispatch.jax,
        "transfer_guard_host_to_device",
        host_to_device_transfer_guard,
    )

    successful, result_text, result_message = dispatch._optimistix_result_metadata(
        dispatch.optx.RESULTS.successful,
    )

    assert successful is True
    assert result_text
    assert result_message == ""
    assert events == ["allow"]


def test_optimistix_success_uses_zero_status():
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    result = least_squares(
        lambda x: x,
        jax.device_put(np.asarray([1.0, -2.0], dtype=dtype)),
        driver=Driver.OPTIMISTIX_LM,
        options=OptimistixLMOptions(
            maxiter=10,
            materialize_dense_linearization=False,
        ),
    )

    assert result.success is True
    assert result.status == 0


def test_optax_callback_event_matches_updated_parameter_state():
    events = []

    def value_and_grad(x):
        return jnp.vdot(x, x), 2.0 * x

    x0 = jnp.array([1.0, -2.0])
    result = minimize(
        value_and_grad,
        x0,
        driver=Driver.OPTAX_ADAM,
        options=OptaxAdamOptions(maxiter=1, learning_rate=0.1),
        callback=events.append,
    )

    assert result.driver is Driver.OPTAX_ADAM
    assert len(events) == 1
    event = events[0]
    assert event.driver is Driver.OPTAX_ADAM
    assert event.wallclock_s >= 0.0
    assert not np.allclose(event.x, np.asarray(x0))
    assert np.isclose(event.fun, float(np.dot(event.x, event.x)))
    assert np.isclose(event.grad_norm_inf, float(np.max(np.abs(2.0 * event.x))))


def test_simsopt_bfgs_uses_explicit_value_grad_under_strict_transfer_guard():
    previous_backend = simsopt_config.get_backend_config()
    previous_transfer_guard = jax.config.jax_transfer_guard
    try:
        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )
        half = jax.device_put(np.asarray(0.5, dtype=np.float64))

        def value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return half * jnp.dot(x, x), x

        result = minimize(
            value_and_grad,
            jnp.asarray(np.array([1.0, -2.0], dtype=np.float64)),
            driver=Driver.SIMSOPT_BFGS,
            options=SimsoptBFGSOptions(maxiter=5),
        )

        assert result.success is True
        assert result.fun < 1e-24
    finally:
        simsopt_config.set_backend(
            previous_backend.mode,
            strict=previous_backend.strict,
            debug_nans=previous_backend.debug_nans,
            disable_jit=previous_backend.disable_jit,
            transfer_guard=previous_backend.transfer_guard,
            compilation_cache_dir=previous_backend.compilation_cache_dir,
            xla_gpu_preallocate=previous_backend.xla_gpu_preallocate,
            xla_gpu_mem_fraction=previous_backend.xla_gpu_mem_fraction,
            xla_gpu_allocator=previous_backend.xla_gpu_allocator,
            tf_gpu_allocator=previous_backend.tf_gpu_allocator,
            configure_runtime=False,
        )
        jax.config.update("jax_transfer_guard", previous_transfer_guard)
