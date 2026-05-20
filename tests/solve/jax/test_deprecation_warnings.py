import logging
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import simsopt.geo.optimizer_jax as legacy_optimizer


_DEPRECATION_LOGGER_NAME = "simsopt.solve.jax.deprecation"


def _fake_result():
    return OptimizeResult(
        x=np.zeros(2),
        fun=0.0,
        jac=np.zeros(2),
        nit=0,
        nfev=1,
        njev=1,
        status=0,
        success=True,
        message="ok",
    )


def test_old_api_warns_once_per_callsite_but_logs_every_call(monkeypatch, caplog):
    monkeypatch.setattr(
        legacy_optimizer,
        "_jax_minimize_legacy",
        lambda *_args, **_kwargs: _fake_result(),
    )
    with legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITE_LOCK:
        legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITES.clear()

    caplog.set_level(logging.INFO, logger=_DEPRECATION_LOGGER_NAME)

    def call_same_line_twice():
        for _ in range(2):
            legacy_optimizer.jax_minimize(lambda x: 0.0, np.zeros(2), method="lbfgs")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        call_same_line_twice()

    assert len(caught) == 1
    warning = caught[0]
    assert warning.filename == __file__
    assert "method='lbfgs' -> driver='scipy_lbfgsb'" in str(warning.message)
    assert len(caplog.records) == 2
    assert {record.translated_driver for record in caplog.records} == {"scipy_lbfgsb"}


def test_old_api_rejects_unknown_method_before_warning_or_log(caplog):
    caplog.set_level(logging.INFO, logger=_DEPRECATION_LOGGER_NAME)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        with pytest.raises(ValueError, match="Unknown method 'not-a-method'"):
            legacy_optimizer.jax_minimize(
                lambda x: 0.0,
                np.zeros(2),
                method="not-a-method",
            )

    assert caught == []
    assert caplog.records == []
