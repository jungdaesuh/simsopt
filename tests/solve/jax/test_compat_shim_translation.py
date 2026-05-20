import warnings

import jax.numpy as jnp
import numpy as np

from simsopt.geo.optimizer_jax import jax_least_squares, jax_minimize
from simsopt.solve.jax import (
    Driver,
    OptimistixLMOptions,
    ScipyBFGSOptions,
    ScipyLBFGSBOptions,
    SimsoptAdamHostOptions,
    SimsoptAdamOptions,
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
    SimsoptLMGMRESHostOptions,
    SimsoptLMGMRESOptions,
    SimsoptLMQROptions,
    SimsoptTraceLBFGSOptions,
    least_squares,
    minimize,
)


def _assert_same_core_result(old, new):
    assert old.success == new.success
    assert old.status == new.status
    np.testing.assert_allclose(np.asarray(old.x), new.x, rtol=0, atol=0)
    np.testing.assert_allclose(float(old.fun), new.fun, rtol=0, atol=0)


def test_old_lbfgs_call_matches_new_scipy_lbfgsb_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="lbfgs",
            value_and_grad=True,
            maxiter=20,
        )
    new = minimize(
        value_and_grad,
        np.array([0.0, 0.0]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_bfgs_call_matches_new_scipy_bfgs_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="bfgs",
            value_and_grad=True,
            maxiter=20,
        )
    new = minimize(
        value_and_grad,
        np.array([0.0, 0.0]),
        driver=Driver.SCIPY_BFGS,
        options=ScipyBFGSOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_lbfgs_trace_call_matches_new_trace_bridge_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="lbfgs-trace",
            value_and_grad=True,
            maxiter=20,
        )
    new = minimize(
        value_and_grad,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_TRACE_LBFGS,
        options=SimsoptTraceLBFGSOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_scipy_jax_call_matches_new_scipy_lbfgsb_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="lbfgs-scipy-jax",
            value_and_grad=True,
            maxiter=20,
        )
    new = minimize(
        value_and_grad,
        np.array([0.0, 0.0]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_scipy_jax_fullgraph_call_matches_new_scipy_lbfgsb_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="lbfgs-scipy-jax-fullgraph",
            value_and_grad=True,
            maxiter=20,
        )
    new = minimize(
        value_and_grad,
        np.array([0.0, 0.0]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_lbfgs_ondevice_call_matches_new_simsopt_lbfgsb_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="lbfgs-ondevice",
            value_and_grad=True,
            maxiter=1,
        )
    new = minimize(
        value_and_grad,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_LBFGSB,
        options=SimsoptLBFGSBOptions(maxiter=1),
    )

    _assert_same_core_result(old, new)


def test_old_bfgs_ondevice_call_matches_new_simsopt_bfgs_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="bfgs-ondevice",
            value_and_grad=True,
            maxiter=20,
        )
    new = minimize(
        value_and_grad,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_BFGS,
        options=SimsoptBFGSOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_adam_call_matches_new_host_bridge_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="adam",
            value_and_grad=True,
            maxiter=3,
            tol=0.0,
            options={"step_size": 0.1},
        )
    new = minimize(
        value_and_grad,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_ADAM_HOST,
        options=SimsoptAdamHostOptions(maxiter=3, learning_rate=0.1),
    )

    _assert_same_core_result(old, new)


def test_old_adam_ondevice_call_matches_new_target_bridge_driver():
    def value_and_grad(x):
        residual = x - jnp.array([1.0, -2.0])
        return jnp.vdot(residual, residual), 2.0 * residual

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_minimize(
            value_and_grad,
            jnp.array([0.0, 0.0]),
            method="adam-ondevice",
            value_and_grad=True,
            maxiter=3,
            tol=0.0,
            options={"step_size": 0.1},
        )
    new = minimize(
        value_and_grad,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_ADAM,
        options=SimsoptAdamOptions(maxiter=3, learning_rate=0.1),
    )

    _assert_same_core_result(old, new)


def test_old_lm_call_matches_new_gmres_host_bridge_driver():
    def residual(x):
        return x - jnp.array([1.0, -2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_least_squares(
            residual,
            jnp.array([0.0, 0.0]),
            method="lm",
            maxiter=20,
        )
    new = least_squares(
        residual,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_LM_GMRES_HOST,
        options=SimsoptLMGMRESHostOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_lm_ondevice_call_matches_new_gmres_target_bridge_driver():
    def residual(x):
        return x - jnp.array([1.0, -2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_least_squares(
            residual,
            jnp.array([0.0, 0.0]),
            method="lm-ondevice",
            maxiter=20,
        )
    new = least_squares(
        residual,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_lm_minpack_ondevice_call_matches_new_qr_bridge_driver():
    def residual(x):
        return x - jnp.array([1.0, -2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_least_squares(
            residual,
            jnp.array([0.0, 0.0]),
            method="lm-minpack-ondevice",
            maxiter=20,
        )
    new = least_squares(
        residual,
        jnp.array([0.0, 0.0]),
        driver=Driver.SIMSOPT_LM_QR,
        options=SimsoptLMQROptions(maxiter=20),
    )

    _assert_same_core_result(old, new)


def test_old_optimistix_lm_ondevice_call_matches_new_optimistix_driver():
    def residual(x):
        return x - jnp.array([1.0, -2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = jax_least_squares(
            residual,
            jnp.array([0.0, 0.0]),
            method="optimistix-lm-ondevice",
            maxiter=20,
        )
    new = least_squares(
        residual,
        jnp.array([0.0, 0.0]),
        driver=Driver.OPTIMISTIX_LM,
        options=OptimistixLMOptions(
            maxiter=20,
            materialize_dense_linearization=True,
        ),
    )

    _assert_same_core_result(old, new)
