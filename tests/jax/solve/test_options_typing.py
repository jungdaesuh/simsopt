import numpy as np
import pytest

from simsopt_jax.solve.dispatch import least_squares, minimize
from simsopt_jax.solve import (
    Driver,
    OptaxAdamOptions,
    ScipyLBFGSBOptions,
    SimsoptAdamOptions,
    SimsoptLMGMRESOptions,
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
            options=OptaxAdamOptions(maxiter=1),
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


def test_minimize_rejects_target_subclass_options_for_host_driver():
    with pytest.raises(
        TypeError,
        match="requires options of type SimsoptAdamHostOptions, got SimsoptAdamOptions",
    ):
        minimize(
            _value_and_grad,
            np.zeros(2),
            driver=Driver.SIMSOPT_ADAM_HOST,
            options=SimsoptAdamOptions(maxiter=1),
        )


def test_least_squares_rejects_target_subclass_options_for_host_driver():
    with pytest.raises(
        TypeError,
        match=(
            "requires options of type SimsoptLMGMRESHostOptions, "
            "got SimsoptLMGMRESOptions"
        ),
    ):
        least_squares(
            _residual,
            np.zeros(2),
            driver=Driver.SIMSOPT_LM_GMRES_HOST,
            options=SimsoptLMGMRESOptions(maxiter=1),
        )
