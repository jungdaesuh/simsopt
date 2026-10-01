from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
import pytest

from simsopt_jax.solve.dispatch import least_squares, minimize
from simsopt_jax.solve import (
    Driver,
    ScipyLBFGSBOptions,
    ScipyLMOptions,
    SimsoptBFGSOptions,
)


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
