import logging
import warnings

import numpy as np
from scipy.optimize import OptimizeResult

import simsopt.geo.optimizer_jax as legacy_optimizer


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


def test_old_lm_shim_maps_to_gmres_host_not_scipy_lm(monkeypatch, caplog):
    monkeypatch.setattr(
        legacy_optimizer,
        "_jax_least_squares_legacy",
        lambda *_args, **_kwargs: _fake_result(),
    )
    with legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITE_LOCK:
        legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITES.clear()

    caplog.set_level(logging.INFO, logger="simsopt.solve.jax.deprecation")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_optimizer.jax_least_squares(
            lambda x: x,
            np.zeros(2),
            method="lm",
        )

    assert len(caught) == 1
    assert "driver='simsopt_lm_gmres_host'" in str(caught[0].message)
    assert caplog.records[0].translated_driver == "simsopt_lm_gmres_host"


def test_old_adam_shim_maps_to_bridge_not_optax(monkeypatch, caplog):
    monkeypatch.setattr(
        legacy_optimizer,
        "_jax_minimize_legacy",
        lambda *_args, **_kwargs: _fake_result(),
    )
    with legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITE_LOCK:
        legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITES.clear()

    caplog.set_level(logging.INFO, logger="simsopt.solve.jax.deprecation")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_optimizer.jax_minimize(
            lambda x: float(np.dot(x, x)),
            np.zeros(2),
            method="adam",
        )

    assert len(caught) == 1
    assert "driver='simsopt_adam_host'" in str(caught[0].message)
    assert caplog.records[0].translated_driver == "simsopt_adam_host"
