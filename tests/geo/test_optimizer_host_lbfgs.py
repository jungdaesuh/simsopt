"""Focused tests for the pure host L-BFGS implementation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import simsopt_jax.geo.optimizer_host_lbfgs as _host_lbfgs


def _rosenbrock_value_and_grad(x):
    value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    gradient = np.asarray(
        (
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        )
    )
    return float(value), gradient


def test_more_thuente_trial_observer_records_each_evaluated_trial():
    x0 = np.asarray((-1.2, 1.0), dtype=np.float64)
    value0, gradient0 = _rosenbrock_value_and_grad(x0)
    direction = -gradient0
    evaluated_parameters = []
    observed_trials = []

    def recording_value_and_grad(parameters):
        evaluated_parameters.append(np.asarray(parameters).copy())
        return _rosenbrock_value_and_grad(parameters)

    result = _host_lbfgs.line_search_value_and_grad_more_thuente_host(
        recording_value_and_grad,
        x0,
        direction,
        value0,
        gradient0,
        value0 + np.linalg.norm(gradient0) / 2.0,
        trial_observer=observed_trials.append,
    )

    assert len(observed_trials) == result.nfev == len(evaluated_parameters) == 2
    assert [trial.trial_ordinal for trial in observed_trials] == [1, 2]
    for trial, parameters in zip(observed_trials, evaluated_parameters):
        np.testing.assert_array_equal(parameters, x0 + trial.alpha * direction)
        objective, gradient = _rosenbrock_value_and_grad(parameters)
        directional_derivative = float(np.dot(gradient, direction))
        assert trial.objective == objective
        assert trial.directional_derivative == directional_derivative
        assert trial.gradient_finite
        assert trial.armijo_margin == pytest.approx(
            objective - (value0 + 1.0e-4 * trial.alpha * np.dot(gradient0, direction))
        )
        assert trial.curvature_margin == pytest.approx(
            abs(directional_derivative) - 0.9 * -np.dot(gradient0, direction)
        )

    with pytest.raises(FrozenInstanceError):
        observed_trials[0].alpha = 1.0


def test_more_thuente_observer_receives_terminating_nonfinite_gradient_trial():
    x0 = np.asarray((1.0,), dtype=np.float64)
    gradient0 = np.asarray((1.0,), dtype=np.float64)
    observed_trials = []

    def nonfinite_trial(_parameters):
        return 0.5, np.asarray((np.nan,), dtype=np.float64)

    result = _host_lbfgs.line_search_value_and_grad_more_thuente_host(
        nonfinite_trial,
        x0,
        -gradient0,
        1.0,
        gradient0,
        trial_observer=observed_trials.append,
    )

    assert result.failed
    assert result.nfev == 1
    assert len(observed_trials) == 1
    trial = observed_trials[0]
    assert trial.trial_ordinal == 1
    assert not trial.gradient_finite
    assert np.isnan(trial.directional_derivative)
    assert np.isnan(trial.armijo_margin)
    assert np.isnan(trial.curvature_margin)


def test_more_thuente_without_observer_does_not_construct_trial_records(monkeypatch):
    def fail_if_constructed(**_fields):
        raise AssertionError(
            "default line search constructed a diagnostic trial record"
        )

    monkeypatch.setattr(_host_lbfgs, "HostLineSearchTrial", fail_if_constructed)
    x0 = np.asarray((-1.2, 1.0), dtype=np.float64)
    value0, gradient0 = _rosenbrock_value_and_grad(x0)

    result = _host_lbfgs.line_search_value_and_grad_more_thuente_host(
        _rosenbrock_value_and_grad,
        x0,
        -gradient0,
        value0,
        gradient0,
        value0 + np.linalg.norm(gradient0) / 2.0,
    )

    assert not result.failed
    assert result.nfev == 2
    assert result.a_k == pytest.approx(0.0008486542812509528, rel=1.0e-15)
