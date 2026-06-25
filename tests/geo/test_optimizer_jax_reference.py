import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import simsopt_jax.geo.optimizers.reference as _opt_ref


@pytest.mark.parametrize(
    ("x", "fun", "jac"),
    (
        (np.asarray([np.nan, 1.0], dtype=np.float64), 3.4, np.asarray([0.0, 0.0])),
        (np.asarray([0.5, 1.0], dtype=np.float64), np.nan, np.asarray([0.0, 0.0])),
        (
            np.asarray([0.5, 1.0], dtype=np.float64),
            3.4,
            np.asarray([np.nan, 0.0], dtype=np.float64),
        ),
    ),
)
def test_normalize_scipy_result_marks_nonfinite_state_failure(x, fun, jac):
    result = OptimizeResult(
        x=x,
        fun=fun,
        jac=jac,
        nit=1,
        nfev=3,
        njev=3,
        success=True,
        status=0,
        message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    )

    normalized = _opt_ref._normalize_scipy_result(result, x_dtype=np.float32)

    assert normalized.success is False
    assert normalized.status == 6
    assert "non-finite objective, iterate, or gradient" in normalized.message.lower()


def test_normalize_scipy_result_preserves_finite_success():
    result = OptimizeResult(
        x=np.asarray([0.5, 1.0], dtype=np.float64),
        fun=3.4,
        jac=np.asarray([0.1, 0.0], dtype=np.float64),
        nit=1,
        nfev=3,
        njev=3,
        success=True,
        status=0,
        message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    )

    normalized = _opt_ref._normalize_scipy_result(result, x_dtype=np.float32)

    assert normalized.success is True
    assert normalized.status == 0
    assert normalized.message == "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
