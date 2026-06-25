import numpy as np
import pytest
import jax.numpy as jnp

from simsopt_jax.solve.dispatch import least_squares
from simsopt_jax.solve import (
    STATUS_CODES,
    Driver,
    OptimistixLMOptions,
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
    fingerprint = fingerprint_optimizer_result(result)
    assert fingerprint.driver is Driver.SCIPY_LBFGSB
    assert fingerprint.x_shape == (2,)
    assert fingerprint.x_dtype == "<f8"
    assert len(fingerprint.x_digest_blake2b) == 32


def test_optimistix_default_does_not_materialize_dense_linearization():
    pytest.importorskip("optimistix")
    pytest.importorskip("lineax")

    def residual(x):
        return x - jnp.array([1.0, -2.0])

    result = least_squares(
        residual,
        jnp.array([0.0, 0.0]),
        driver=Driver.OPTIMISTIX_LM,
        options=OptimistixLMOptions(maxiter=3),
    )

    assert result.residual is not None
    assert result.jac is not None
    assert result.residual_jacobian is None
    assert result.hessian is None


def test_optimistix_dense_linearization_honors_byte_budget():
    pytest.importorskip("optimistix")
    pytest.importorskip("lineax")

    def residual(x):
        return x - jnp.array([1.0, -2.0])

    with pytest.raises(ValueError, match="max_dense_linearization_bytes"):
        least_squares(
            residual,
            jnp.array([0.0, 0.0]),
            driver=Driver.OPTIMISTIX_LM,
            options=OptimistixLMOptions(
                maxiter=1,
                materialize_dense_linearization=True,
                max_dense_linearization_bytes=1,
            ),
        )
