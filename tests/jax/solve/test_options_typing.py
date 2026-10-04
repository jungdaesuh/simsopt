from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import replace

import numpy as np
import pytest
import jax.numpy as jnp

from simsopt_jax.solve.dispatch import least_squares, minimize
from simsopt_jax.solve import (
    Driver,
    ScipyBFGSOptions,
    ScipyLBFGSBOptions,
    ScipyLMOptions,
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
    SimsoptLMQROptions,
)
from simsopt_jax.solve.contracts import OptionsBase


def _value_and_grad(x):
    return 0.0, np.zeros_like(np.asarray(x, dtype=float))


def _residual(x):
    return np.asarray(x, dtype=float)


def test_minimize_rejects_options_for_the_wrong_driver():
    with pytest.raises(TypeError, match="requires options of type ScipyLBFGSBOptions"):
        minimize(
            _value_and_grad,
            np.zeros(2),
            driver=Driver.SCIPY_LBFGSB,
            options=SimsoptBFGSOptions(maxiter=1),
        )


def test_minimize_rejects_least_squares_driver():
    with pytest.raises(ValueError, match="not valid here"):
        minimize(_value_and_grad, np.zeros(2), driver=Driver.SCIPY_LM)


def test_least_squares_rejects_minimize_options():
    with pytest.raises(TypeError, match="requires options of type"):
        least_squares(
            _residual,
            np.zeros(2),
            driver=Driver.SCIPY_LM,
            options=ScipyLBFGSBOptions(maxiter=1),
        )


def test_least_squares_rejects_the_other_least_squares_drivers_options():
    with pytest.raises(
        TypeError,
        match="requires options of type SimsoptLMQROptions, got ScipyLMOptions",
    ):
        least_squares(
            _residual,
            np.zeros(2),
            driver=Driver.SIMSOPT_LM_QR,
            options=ScipyLMOptions(),
        )


_OPTION_FIELDS = (
    (
        Driver.SCIPY_LBFGSB,
        ScipyLBFGSBOptions(),
        ("maxiter", "maxfun"),
        ("maxcor", "maxls"),
        ("gtol", "ftol"),
    ),
    (Driver.SCIPY_BFGS, ScipyBFGSOptions(), ("maxiter",), (), ("gtol", "xrtol")),
    (Driver.SCIPY_LM, ScipyLMOptions(), (), ("max_nfev",), ("ftol", "xtol", "gtol")),
    (
        Driver.SIMSOPT_LBFGSB,
        SimsoptLBFGSBOptions(),
        ("maxiter", "maxfun"),
        ("maxcor", "maxls"),
        ("gtol", "ftol"),
    ),
    (
        Driver.SIMSOPT_BFGS,
        SimsoptBFGSOptions(),
        ("maxiter",),
        ("line_search_max_steps",),
        ("gtol", "xrtol"),
    ),
    (
        Driver.SIMSOPT_LM_QR,
        SimsoptLMQROptions(),
        (),
        ("maxiter", "max_dense_linearization_bytes"),
        ("ftol", "xtol", "gtol"),
    ),
)


def _dispatch_with_unevaluated_objective(driver: Driver, options: OptionsBase):
    def unevaluated(_x):
        pytest.fail("invalid options must be rejected before evaluating the objective")

    if driver in (Driver.SCIPY_LM, Driver.SIMSOPT_LM_QR):
        return least_squares(unevaluated, np.zeros(2), driver=driver, options=options)
    return minimize(unevaluated, np.zeros(2), driver=driver, options=options)


@pytest.mark.parametrize("value", (0, -1, 1.5, float("inf"), float("nan"), True))
@pytest.mark.parametrize(
    "driver,options,name",
    [
        (driver, options, name)
        for driver, options, _, budgets, _ in _OPTION_FIELDS
        for name in budgets
    ],
)
def test_dispatch_rejects_invalid_positive_budgets_before_evaluation(driver, options, name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        _dispatch_with_unevaluated_objective(driver, replace(options, **{name: value}))


@pytest.mark.parametrize("value", (-1, 1.5, float("inf"), float("nan"), True))
@pytest.mark.parametrize(
    "driver,options,name",
    [
        (driver, options, name)
        for driver, options, budgets, _, _ in _OPTION_FIELDS
        for name in budgets
    ],
)
def test_dispatch_rejects_invalid_nonnegative_budgets_before_evaluation(driver, options, name, value):
    with pytest.raises(ValueError, match=f"{name} must be a non-negative integer"):
        _dispatch_with_unevaluated_objective(driver, replace(options, **{name: value}))


@pytest.mark.parametrize("driver,options,nfev", (
    (Driver.SCIPY_BFGS, ScipyBFGSOptions(maxiter=0), 1),
    (Driver.SIMSOPT_BFGS, SimsoptBFGSOptions(maxiter=0), 2),
))
def test_bfgs_zero_iterations_returns_initial_state(driver, options, nfev):
    def value_and_grad(x):
        return jnp.sum(x * x), 2.0 * x

    initial = np.asarray([2.0, -1.0])
    result = minimize(value_and_grad, initial, driver=driver, options=options)
    np.testing.assert_array_equal(result.x, initial)
    np.testing.assert_array_equal(result.fun, 5.0)
    np.testing.assert_array_equal(result.jac, 2.0 * initial)
    assert result.nit == 0
    assert result.nfev == nfev
    assert result.status == 1


@pytest.mark.parametrize("name", ("maxiter", "maxfun"))
@pytest.mark.parametrize("driver,options", (
    (Driver.SCIPY_LBFGSB, ScipyLBFGSBOptions()),
    (Driver.SIMSOPT_LBFGSB, SimsoptLBFGSBOptions()),
))
def test_lbfgsb_zero_budgets_preserve_checks_after_first_step(driver, options, name):
    def value_and_grad(x):
        return jnp.sum(x * x), 2.0 * x

    result = minimize(
        value_and_grad, np.asarray([2.0]), driver=driver,
        options=replace(options, **{name: 0}),
    )
    np.testing.assert_array_equal(result.x, [1.0])
    np.testing.assert_array_equal(result.fun, 1.0)
    assert result.nit == 1
    assert result.nfev == 2
    assert result.status == 1


@pytest.mark.parametrize("value", (-1.0, float("inf"), float("-inf"), float("nan")))
@pytest.mark.parametrize(
    "driver,options,name",
    [
        (driver, options, name)
        for driver, options, _, _, tolerances in _OPTION_FIELDS
        for name in tolerances
    ],
)
def test_dispatch_rejects_invalid_tolerances_before_evaluation(driver, options, name, value):
    with pytest.raises(ValueError, match=f"{name} must be finite and non-negative"):
        _dispatch_with_unevaluated_objective(driver, replace(options, **{name: value}))


@pytest.mark.parametrize("name", ("ftol", "xtol", "gtol"))
@pytest.mark.parametrize("value", (0.0, np.finfo(float).eps))
def test_scipy_lm_rejects_tolerances_at_or_below_machine_epsilon(name, value):
    with pytest.raises(ValueError, match=f"{name} must be greater than machine epsilon"):
        _dispatch_with_unevaluated_objective(
            Driver.SCIPY_LM, replace(ScipyLMOptions(), **{name: value})
        )


@pytest.mark.parametrize("driver,options,nonnegative_budgets,positive_budgets,tolerances", _OPTION_FIELDS)
def test_valid_options_preserve_driver_specific_values(driver, options, nonnegative_budgets, positive_budgets, tolerances):
    budgets = nonnegative_budgets + positive_budgets
    updated = options
    for name in budgets:
        updated = replace(updated, **{name: np.int64(1)})
    if driver != Driver.SCIPY_LM:
        for name in tolerances:
            updated = replace(updated, **{name: 0.0})
    updated.validate()
    for name in budgets:
        assert getattr(updated, name) == 1
    if driver != Driver.SCIPY_LM:
        for name in tolerances:
            assert getattr(updated, name) == 0.0


def test_simsopt_lm_preserves_optional_tolerance_and_dense_budget():
    options = SimsoptLMQROptions(gtol=None, max_dense_linearization_bytes=None)
    options.validate()
    assert options.gtol is None
    assert options.max_dense_linearization_bytes is None


@pytest.mark.parametrize("norm", (float("inf"), float("-inf"), -1.0, 1.0, 2.0))
def test_scipy_bfgs_preserves_supported_vector_norm_orders(norm):
    options = ScipyBFGSOptions(norm=norm)
    options.validate()
    assert options.norm == norm


@pytest.mark.parametrize("norm", (float("nan"), 0.0))
def test_scipy_bfgs_rejects_invalid_norm_before_evaluation(norm):
    with pytest.raises(ValueError, match="norm must be a nonzero real vector norm order"):
        _dispatch_with_unevaluated_objective(
            Driver.SCIPY_BFGS, ScipyBFGSOptions(norm=norm)
        )


def test_scipy_lbfgsb_rejects_nonboolean_restart_policy_before_evaluation():
    with pytest.raises(ValueError, match="restart_after_nonwolfe_stop must be a bool"):
        _dispatch_with_unevaluated_objective(
            Driver.SCIPY_LBFGSB,
            replace(ScipyLBFGSBOptions(), **{"restart_after_nonwolfe_stop": 1}),
        )
