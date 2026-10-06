from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    enable_strict_parity_backend,
)

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.solve.dispatch import minimize
from simsopt_jax.solve import (
    Driver,
    ScipyBFGSOptions,
    SimsoptBFGSOptions,
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


def test_simsopt_bfgs_uses_explicit_value_grad_under_strict_transfer_guard(
    monkeypatch, request
):
    # Strict parity on this process's own JAX platform, with the guard carried
    # by the backend config: a CPU parity mode is refused in a GPU-default
    # process, and the jax_test_support helper pins the GPU determinism flag the
    # strict GPU mode requires.
    monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "disallow")
    enable_strict_parity_backend(
        monkeypatch,
        request,
        "gpu" if jax.default_backend() == "gpu" else "cpu",
    )
    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def value_and_grad(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return half * jnp.dot(x, x), x

    result = minimize(
        value_and_grad,
        jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
        driver=Driver.SIMSOPT_BFGS,
        options=SimsoptBFGSOptions(maxiter=5),
    )

    assert result.success is True
    assert result.fun < 1e-24
