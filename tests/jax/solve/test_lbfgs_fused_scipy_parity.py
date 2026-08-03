"""Direct SciPy parity contracts for the fused-stepwise L-BFGS-B route."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize
from simsopt_jax.geo.optimizers.private import prepare_lbfgs_private
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.geo.optimizers.private._lbfgs import (
    _LBFGS_RUN_MODE_FUSED_STEPWISE,
)

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="Fused L-BFGS-B is validated on the pinned JAX runtime.",
    ),
]

# SciPy's Fortran line search and the fused JAX loop use the same transitions,
# but can reassociate a few FP64 operations.  These are the endpoint tolerances
# established by the existing fused-vs-stepwise contract.
_X_ATOL = 1.0e-11
_VALUE_RTOL = 2.0e-12
_VALUE_ATOL = 2.0e-14
_GRAD_RTOL = 2.0e-12
_GRAD_ATOL = 2.0e-12


def _rosenbrock_jax(x: jax.Array) -> jax.Array:
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def _rosenbrock_scipy(x: np.ndarray) -> tuple[np.float64, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    gradient = np.asarray(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=np.float64,
    )
    return np.float64(value), gradient


def _quadratic_jax(x: jax.Array) -> jax.Array:
    target = jnp.asarray([1.0, -2.0], dtype=x.dtype)
    return jnp.sum((x - target) ** 2)


def _quadratic_scipy(x: np.ndarray) -> tuple[np.float64, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    target = np.asarray([1.0, -2.0], dtype=np.float64)
    delta = x - target
    return np.float64(np.sum(delta**2)), 2.0 * delta


def _run_fused(
    objective: Callable[[jax.Array], jax.Array],
    x0: Sequence[float],
    *,
    maxiter: int,
    maxfun: int | None,
    maxcor: int = 10,
    ftol: float = 0.0,
    gtol: float = 1.0e-12,
    maxls: int = 20,
):
    x0_jax = jnp.asarray(x0, dtype=jnp.float64)
    prepared = prepare_lbfgs_private(
        objective,
        x0_jax,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
        run_mode=_LBFGS_RUN_MODE_FUSED_STEPWISE,
    )
    assert prepared.fused_solve is not None
    return prepared.run(x0_jax, maxiter=maxiter, maxfun=maxfun)


def _assert_direct_parity(fused, scipy_result: optimize.OptimizeResult) -> None:
    assert int(np.asarray(fused.status)) == int(scipy_result.status)
    assert bool(np.asarray(fused.converged)) is bool(scipy_result.success)
    assert bool(np.asarray(fused.failed)) is (not bool(scipy_result.success))
    assert int(np.asarray(fused.k)) == int(scipy_result.nit)
    assert int(np.asarray(fused.nfev)) == int(scipy_result.nfev)
    assert int(np.asarray(fused.ngev)) == int(scipy_result.njev)
    np.testing.assert_allclose(
        np.asarray(fused.x_k),
        np.asarray(scipy_result.x),
        rtol=0.0,
        atol=_X_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(fused.f_k),
        np.asarray(scipy_result.fun),
        rtol=_VALUE_RTOL,
        atol=_VALUE_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(fused.g_k),
        np.asarray(scipy_result.jac),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )


def test_fused_matches_scipy_with_active_solution_bounds() -> None:
    x0 = np.asarray([2.0, -1.0], dtype=np.float64)
    # The prepared fused API currently has no bounds argument.  These bounds
    # are active at the same unconstrained minimizer, so this is endpoint
    # parity at an active SciPy bound, not a claim of fused bound projection.
    bounds = ((1.0, 2.0), (-3.0, -1.0))
    options = {
        "maxiter": 50,
        "maxcor": 10,
        "ftol": 0.0,
        "gtol": 1.0e-12,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _quadratic_scipy,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=options,
    )
    fused = _run_fused(_quadratic_jax, x0, maxfun=None, **options)

    _assert_direct_parity(fused, scipy_result)
    assert scipy_result.x[0] == pytest.approx(bounds[0][0], abs=0.0)
    assert scipy_result.x[1] > bounds[1][0]


def test_fused_matches_scipy_when_maxfun_exhausts_mid_line_search() -> None:
    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    options = {
        "maxiter": 10,
        "maxfun": 1,
        "maxcor": 10,
        "ftol": 0.0,
        "gtol": 1.0e-12,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _rosenbrock_scipy,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    fused = _run_fused(_rosenbrock_jax, x0, **options)

    _assert_direct_parity(fused, scipy_result)
    assert scipy_result.status == 1
    assert not scipy_result.success
    assert scipy_result.nfev > options["maxfun"]
    assert "EVALUATIONS EXCEEDS LIMIT" in str(scipy_result.message)


def test_fused_matches_scipy_with_zero_maxiter_budget() -> None:
    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    options = {
        "maxiter": 0,
        "maxfun": 15000,
        "maxcor": 10,
        "ftol": 0.0,
        "gtol": 1.0e-12,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _rosenbrock_scipy,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    fused = _run_fused(_rosenbrock_jax, x0, **options)

    _assert_direct_parity(fused, scipy_result)
    assert scipy_result.status == 1
    assert "ITERATIONS REACHED LIMIT" in str(scipy_result.message)


@pytest.mark.parametrize("maxfun", [0, 1])
def test_fused_matches_scipy_with_zero_or_near_zero_maxfun(maxfun: int) -> None:
    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    options = {
        "maxiter": 10,
        "maxfun": maxfun,
        "maxcor": 10,
        "ftol": 0.0,
        "gtol": 1.0e-12,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _rosenbrock_scipy,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    fused = _run_fused(_rosenbrock_jax, x0, **options)

    _assert_direct_parity(fused, scipy_result)
    assert scipy_result.status == 1
    assert scipy_result.nfev > maxfun


def test_fused_matches_scipy_unconstrained_baseline() -> None:
    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    options = {
        "maxiter": 60,
        "maxcor": 10,
        "ftol": 1.0e-12,
        "gtol": 1.0e-10,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _rosenbrock_scipy,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    fused = _run_fused(_rosenbrock_jax, x0, maxfun=None, **options)

    _assert_direct_parity(fused, scipy_result)
    assert scipy_result.success


def test_fused_matches_scipy_at_stationary_point_with_zero_budgets() -> None:
    x0 = np.asarray([1.0, -2.0], dtype=np.float64)
    options = {
        "maxiter": 0,
        "maxfun": 0,
        "maxcor": 10,
        "ftol": 0.0,
        "gtol": 1.0e-12,
        "maxls": 20,
    }
    scipy_result = optimize.minimize(
        _quadratic_scipy,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    fused = _run_fused(_quadratic_jax, x0, **options)

    _assert_direct_parity(fused, scipy_result)
    assert scipy_result.success
    assert scipy_result.nit == 0
    assert scipy_result.nfev == 1
