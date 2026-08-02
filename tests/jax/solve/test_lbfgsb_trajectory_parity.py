"""Accepted-iterate parity for SIMSOPT's stepwise JAX L-BFGS-B."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize
from simsopt_jax.geo.optimizers.optimizer import target_minimize
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.geo.optimizers.private._lbfgs import (
    _minimize_lbfgs_private_value_and_grad,
)

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="Stepwise L-BFGS-B is validated on the pinned JAX runtime.",
    ),
]


def _scipy_rosenbrock(x):
    x = np.asarray(x, dtype=np.float64)
    value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    gradient = np.asarray(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=np.float64,
    )
    return np.float64(value), gradient


def _jax_rosenbrock(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def test_lbfgs_ondevice_accepted_trajectory_matches_scipy() -> None:
    native_trace = []

    def record_native_iterate(x):
        value, gradient = _scipy_rosenbrock(x)
        native_trace.append((np.array(x, copy=True), value, gradient))

    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    options = {
        "maxiter": 10,
        "maxcor": 10,
        "ftol": 1.0e-12,
        "gtol": 1.0e-12,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _scipy_rosenbrock,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
        callback=record_native_iterate,
    )
    result = target_minimize(
        _jax_rosenbrock,
        jnp.asarray(x0, dtype=jnp.float64),
        method="lbfgs-ondevice",
        tol=options["gtol"],
        maxiter=options["maxiter"],
        options={
            "maxcor": options["maxcor"],
            "ftol": options["ftol"],
            "maxls": options["maxls"],
            "record_optimizer_state_trace": True,
        },
    )

    assert len(result.optimizer_state_trace) == len(native_trace)
    for actual, (expected_x, expected_value, expected_gradient) in zip(
        result.optimizer_state_trace,
        native_trace,
    ):
        np.testing.assert_allclose(
            np.asarray(actual["x"]), expected_x, rtol=2.0e-12, atol=2.0e-14
        )
        np.testing.assert_allclose(
            actual["fun"], expected_value, rtol=2.0e-12, atol=2.0e-14
        )
        np.testing.assert_allclose(
            np.asarray(actual["jac"]),
            expected_gradient,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
    assert result.nit == scipy_result.nit
    assert result.nfev == scipy_result.nfev
    assert result.njev == scipy_result.njev


def test_lbfgs_fp64_rosenbrock_freezes_accepted_steps() -> None:
    def value_and_grad(x):
        value = _jax_rosenbrock(x)
        gradient = jnp.asarray(
            [
                -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] ** 2),
            ],
            dtype=jnp.float64,
        )
        return value, gradient

    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    result = _minimize_lbfgs_private_value_and_grad(
        value_and_grad,
        x0,
        maxiter=3,
        maxcor=3,
        ftol=0.0,
        gtol=1.0e-12,
        maxls=20,
        record_optimizer_state_trace=True,
    )
    expected_x = np.asarray(
        [
            [-1.0174097957038217, 1.0745266139984402],
            [-1.0270197526043319, 1.068358702632824],
            [-1.0285358469293884, 1.065342136730525],
        ],
        dtype=np.float64,
    )
    expected_step_norms = np.asarray(
        [0.19721409406782417, 0.011419036835204578, 0.003376123789996811],
        dtype=np.float64,
    )
    actual_x = np.asarray([entry["x"] for entry in result.optimizer_state_trace])
    np.testing.assert_allclose(actual_x, expected_x, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        np.linalg.norm(
            np.diff(np.vstack((np.asarray(x0), expected_x)), axis=0), axis=1
        ),
        expected_step_norms,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(result.f_k, 4.120517097042244, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        result.g_k,
        np.asarray([-0.9895053663875037, 1.4912296623541357]),
        rtol=0.0,
        atol=2e-13,
    )
    assert int(result.k) == 3
    assert int(result.nfev) == 5
    assert int(result.ngev) == 5
    assert int(result.status) == 1
    assert int(result.ls_status) == 0
    np.testing.assert_array_equal(np.asarray(result.task), np.asarray([5, 504]))


def test_lbfgs_ondevice_preserves_deferred_maxfun_stop() -> None:
    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    options = {
        "maxiter": 10,
        "maxfun": 3,
        "maxcor": 10,
        "ftol": 0.0,
        "gtol": 1.0e-12,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _scipy_rosenbrock,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    result = target_minimize(
        _jax_rosenbrock,
        jnp.asarray(x0, dtype=jnp.float64),
        method="lbfgs-ondevice",
        tol=options["gtol"],
        maxiter=options["maxiter"],
        options={
            "maxcor": options["maxcor"],
            "maxfun": options["maxfun"],
            "ftol": options["ftol"],
            "maxls": options["maxls"],
        },
    )

    assert result.nit == scipy_result.nit
    assert result.nfev == scipy_result.nfev
    assert result.njev == scipy_result.njev
    assert result.status == scipy_result.status
    np.testing.assert_allclose(result.fun, scipy_result.fun, rtol=2e-12, atol=2e-14)


def test_lbfgs_ondevice_observer_does_not_change_solution() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    options = {
        "maxcor": 10,
        "ftol": 0.0,
        "maxls": 20,
    }
    plain = target_minimize(
        _jax_rosenbrock,
        x0,
        method="lbfgs-ondevice",
        tol=1.0e-12,
        maxiter=5,
        options=options,
    )
    callback_points = []
    observed = target_minimize(
        _jax_rosenbrock,
        x0,
        method="lbfgs-ondevice",
        tol=1.0e-12,
        maxiter=5,
        options=options,
        callback=lambda x: callback_points.append(np.asarray(x)),
    )

    np.testing.assert_allclose(observed.x, plain.x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed.fun, plain.fun, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed.jac, plain.jac, rtol=0.0, atol=0.0)
    assert observed.nit == plain.nit
    assert observed.nfev == plain.nfev
    assert observed.njev == plain.njev
    assert plain.optimizer_state_trace == ()
    assert observed.optimizer_state_trace == ()
    assert len(callback_points) == observed.nit
