import numpy as np
import pytest
import jax.numpy as jnp

from scipy.optimize import least_squares

from simsopt.geo import optimizer_jax as _opt


def _rosenbrock_residual_np(x):
    return np.asarray([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _rosenbrock_jacobian_np(x):
    return np.asarray([[-20.0 * x[0], 10.0], [-1.0, 0.0]])


def _rosenbrock_residual_jax(x):
    return jnp.asarray([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _linear_least_squares_problem():
    matrix = jnp.asarray(
        [[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    return matrix, rhs


def test_target_lm_minpack_qr_lane_matches_scipy_lm_solution():
    x0 = np.asarray([-2.0, 1.0], dtype=np.float64)
    scipy_result = least_squares(
        _rosenbrock_residual_np,
        x0,
        jac=_rosenbrock_jacobian_np,
        method="lm",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=200,
    )

    jax_result = _opt.target_least_squares(
        _rosenbrock_residual_jax,
        jnp.asarray(x0),
        method="lm-minpack-ondevice",
        tol=1.0e-12,
        maxiter=200,
        options={"ftol": 1.0e-12, "xtol": 1.0e-12, "gtol": 1.0e-12},
    )

    assert jax_result.success
    np.testing.assert_allclose(jax_result.x, scipy_result.x, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(
        jax_result.residual,
        scipy_result.fun,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert jax_result.residual_jacobian.shape == (2, 2)
    assert jax_result.hessian.shape == (2, 2)


def test_target_lm_minpack_qr_lane_uses_pivoted_qr_on_overdetermined_system():
    matrix = jnp.asarray(
        [[1.0, 2.0], [3.0, -1.0], [2.0, 0.5], [-1.0, 4.0]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)
    expected_x, *_ = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)

    result = _opt.target_least_squares(
        lambda x: matrix @ x - rhs,
        jnp.zeros(2, dtype=jnp.float64),
        method="lm-minpack-ondevice",
        tol=1.0e-12,
        maxiter=50,
        options={"ftol": 1.0e-12, "xtol": 1.0e-12, "gtol": 1.0e-12},
    )

    assert result.success
    np.testing.assert_allclose(result.x, expected_x, rtol=1.0e-10, atol=1.0e-10)
    assert result.info in {0, 1, 2, 3, 4}


def test_target_lm_minpack_qr_lane_always_reports_required_dense_artifacts():
    matrix, rhs = _linear_least_squares_problem()

    result = _opt.target_least_squares(
        lambda x: matrix @ x - rhs,
        jnp.zeros(2, dtype=jnp.float64),
        method="lm-minpack-ondevice",
        tol=1.0e-12,
        maxiter=50,
        options={
            "materialize_dense_linearization": False,
            "ftol": 1.0e-12,
            "xtol": 1.0e-12,
            "gtol": 1.0e-12,
        },
    )

    assert result.dense_linearization_materialized is True
    np.testing.assert_allclose(result.residual_jacobian, matrix)
    np.testing.assert_allclose(result.hessian, matrix.T @ matrix)


def test_target_lm_minpack_qr_lane_enforces_dense_linearization_byte_cap():
    matrix, rhs = _linear_least_squares_problem()

    with pytest.raises(
        MemoryError,
        match="dense QR solve requires.*max_dense_linearization_bytes=1",
    ):
        _opt.target_least_squares(
            lambda x: matrix @ x - rhs,
            jnp.zeros(2, dtype=jnp.float64),
            method="lm-minpack-ondevice",
            tol=1.0e-12,
            maxiter=50,
            options={"max_dense_linearization_bytes": 1},
        )
