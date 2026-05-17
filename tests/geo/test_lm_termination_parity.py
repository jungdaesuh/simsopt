import numpy as np
import jax.numpy as jnp

from simsopt.geo import optimizer_jax as _opt


def _info(
    *,
    actual_reduction,
    predicted_reduction,
    delta,
    x_norm=1.0,
    nit=1,
    maxiter=10,
    ftol=0.0,
    xtol=0.0,
    epsmch=1.0e-12,
):
    value = _opt._matrix_free_lm_info(
        actual_reduction=jnp.asarray(actual_reduction, dtype=jnp.float64),
        predicted_reduction=jnp.asarray(predicted_reduction, dtype=jnp.float64),
        cost=jnp.asarray(1.0, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        x_norm=jnp.asarray(x_norm, dtype=jnp.float64),
        nit=jnp.asarray(nit, dtype=jnp.int32),
        maxiter=jnp.asarray(maxiter, dtype=jnp.int32),
        ftol=jnp.asarray(ftol, dtype=jnp.float64),
        xtol=jnp.asarray(xtol, dtype=jnp.float64),
        epsmch=jnp.asarray(epsmch, dtype=jnp.float64),
    )
    return int(np.asarray(value))


def test_matrix_free_info_subset_matches_minpack_ordering():
    assert (
        _info(
            actual_reduction=1.0e-12,
            predicted_reduction=1.0e-12,
            delta=1.0,
            ftol=1.0e-8,
        )
        == 1
    )
    assert (
        _info(
            actual_reduction=0.5,
            predicted_reduction=0.5,
            delta=1.0e-12,
            xtol=1.0e-8,
        )
        == 2
    )
    assert (
        _info(
            actual_reduction=1.0e-12,
            predicted_reduction=1.0e-12,
            delta=1.0e-12,
            ftol=1.0e-8,
            xtol=1.0e-8,
        )
        == 3
    )
    assert (
        _info(
            actual_reduction=0.5,
            predicted_reduction=0.5,
            delta=1.0,
            nit=5,
            maxiter=5,
        )
        == 5
    )
    assert (
        _info(
            actual_reduction=5.0e-13,
            predicted_reduction=5.0e-13,
            delta=1.0,
            ftol=1.0e-15,
        )
        == 6
    )
    assert (
        _info(
            actual_reduction=0.5,
            predicted_reduction=0.5,
            delta=5.0e-13,
            xtol=1.0e-15,
        )
        == 7
    )


def test_matrix_free_info_does_not_accept_rejected_uphill_tiny_reduction():
    assert (
        _info(
            actual_reduction=-1.0e-12,
            predicted_reduction=1.0e-12,
            delta=1.0,
            ftol=1.0e-8,
        )
        == 0
    )


def test_levenberg_marquardt_surfaces_ftol_info():
    A = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=jnp.float64)
    b = jnp.array([5.0, 7.0], dtype=jnp.float64)

    result = _opt.levenberg_marquardt(
        lambda x: A @ x - b,
        jnp.zeros(2, dtype=jnp.float64),
        maxiter=25,
        tol=0.0,
        ftol=1.0,
        xtol=0.0,
    )

    assert result["success"]
    assert result["info"] == 1
    assert result["status"] == 1


def test_levenberg_marquardt_surfaces_xtol_info():
    A = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=jnp.float64)
    b = jnp.array([5.0, 7.0], dtype=jnp.float64)

    result = _opt.levenberg_marquardt(
        lambda x: A @ x - b,
        jnp.zeros(2, dtype=jnp.float64),
        maxiter=25,
        tol=0.0,
        ftol=0.0,
        xtol=10.0,
    )

    assert result["success"]
    assert result["info"] == 2
    assert result["status"] == 1


def test_levenberg_marquardt_uses_explicit_gtol_gradient_gate():
    x0 = jnp.asarray([0.0], dtype=jnp.float64)
    residual_fn = lambda x: x - jnp.asarray([1.0], dtype=jnp.float64)

    high_gtol = _opt.levenberg_marquardt(
        residual_fn,
        x0,
        maxiter=0,
        tol=0.0,
        gtol=10.0,
    )
    strict_gtol = _opt.levenberg_marquardt(
        residual_fn,
        x0,
        maxiter=0,
        tol=0.0,
        gtol=0.0,
    )

    assert high_gtol["success"]
    assert high_gtol["nit"] == 0
    assert not strict_gtol["success"]
