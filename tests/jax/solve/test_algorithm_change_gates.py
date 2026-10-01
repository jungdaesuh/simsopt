import logging
import warnings

import numpy as np
from scipy.optimize import OptimizeResult

import simsopt_jax.geo.optimizers.optimizer as legacy_optimizer


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


def test_old_lm_shim_maps_to_lm_qr_not_scipy_lm(monkeypatch, caplog):
    monkeypatch.setattr(
        legacy_optimizer,
        "target_least_squares",
        lambda *_args, **_kwargs: _fake_result(),
    )
    with legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITE_LOCK:
        legacy_optimizer._DEPRECATED_SOLVE_JAX_CALLSITES.clear()

    caplog.set_level(logging.INFO, logger="simsopt_jax.solve.deprecation")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_optimizer.jax_least_squares(
            lambda x: x,
            np.zeros(2),
            method="lm-minpack-ondevice",
        )

    assert len(caught) == 1
    assert "driver='simsopt_lm_qr'" in str(caught[0].message)
    assert caplog.records[0].translated_driver == "simsopt_lm_qr"
