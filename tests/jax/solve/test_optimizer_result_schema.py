from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
import pytest

from simsopt_jax.solve import (
    STATUS_CODES,
    Driver,
    OptimizerResult,
    ScipyLBFGSBOptions,
)


def test_status_codes_cover_every_driver():
    assert set(STATUS_CODES) == set(Driver)


def test_optimizer_result_is_not_hashable():
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
