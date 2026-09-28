import numpy as np
import pytest

from simsopt_jax.solve import (
    STATUS_CODES,
    Driver,
    OptimizerResult,
    ScipyLBFGSBOptions,
    fingerprint_optimizer_result,
)


def test_status_codes_cover_every_driver():
    assert set(STATUS_CODES) == set(Driver)


def test_optimizer_result_is_not_hashable_but_has_stable_fingerprint():
    result = OptimizerResult(
        x=np.array([1.0, 2.0], dtype=np.float64),
        fun=3.0,
        jac=np.array([0.0, 0.0]),
        nit=2,
        nfev=3,
        njev=3,
        status=0,
        success=True,
        message="ok",
        driver=Driver.SCIPY_LBFGSB,
        options_used=ScipyLBFGSBOptions(),
        wallclock_s=0.1,
    )

    with pytest.raises(TypeError):
        hash(result)
    assert result.restart_log == (), "no restart unless a driver records one"
    fingerprint = fingerprint_optimizer_result(result)
    assert fingerprint.driver is Driver.SCIPY_LBFGSB
    assert fingerprint.x_shape == (2,)
    assert fingerprint.x_dtype == "<f8"
    assert len(fingerprint.x_digest_blake2b) == 32
