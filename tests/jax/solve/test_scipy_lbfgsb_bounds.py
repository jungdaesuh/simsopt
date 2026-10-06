"""Typed box bounds on the SciPy L-BFGS-B route of ``simsopt_jax.solve``."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from simsopt_jax.solve import Driver, ScipyBFGSOptions, ScipyBounds, ScipyLBFGSBOptions
from simsopt_jax.solve.dispatch import minimize

_CENTER = np.array([3.0, -2.0, 0.25, 7.0])
_CURVATURE = np.array([1.0, 4.0, 0.5, 2.0])
_OPTIONS = ScipyLBFGSBOptions(maxiter=60, maxcor=10, ftol=1e-15, gtol=1e-13)


def _value_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
    residual = np.asarray(x, dtype=float) - _CENTER
    return 0.5 * float(np.sum(_CURVATURE * residual**2)), _CURVATURE * residual


def _direct_scipy(x0: np.ndarray, bounds: list[tuple[float, float]] | None):
    return scipy_minimize(
        _value_and_gradient,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": _OPTIONS.maxiter,
            "maxfun": _OPTIONS.maxfun,
            "gtol": _OPTIONS.gtol,
            "ftol": _OPTIONS.ftol,
            "maxcor": _OPTIONS.maxcor,
            "maxls": _OPTIONS.maxls,
        },
    )


def test_bounded_route_is_scipy_given_the_same_box_bit_for_bit() -> None:
    """The route hands SciPy the box unchanged: same endpoint, counts and message."""
    box = ScipyBounds(
        lower=(-1.0, -np.inf, 0.5, 0.0),
        upper=(1.0, np.inf, 2.0, 6.5),
    )
    x0 = np.zeros(4)

    routed = minimize(
        _value_and_gradient,
        x0,
        driver=Driver.SCIPY_LBFGSB,
        options=_OPTIONS,
        bounds=box,
    )
    direct = _direct_scipy(x0, box.scipy_bounds())

    assert np.array_equal(routed.x, direct.x), (routed.x, direct.x)
    assert (routed.nit, routed.nfev, routed.status, routed.message) == (
        direct.nit,
        direct.nfev,
        direct.status,
        direct.message,
    )


def test_bounds_hold_the_endpoint_on_the_active_faces() -> None:
    """Coordinates whose unconstrained minimum lies outside the box end on its face."""
    box = ScipyBounds(
        lower=(-1.0, -np.inf, 0.5, 0.0),
        upper=(1.0, np.inf, 2.0, 6.5),
    )
    unbounded = minimize(
        _value_and_gradient, np.zeros(4), driver=Driver.SCIPY_LBFGSB, options=_OPTIONS
    )
    bounded = minimize(
        _value_and_gradient,
        np.zeros(4),
        driver=Driver.SCIPY_LBFGSB,
        options=_OPTIONS,
        bounds=box,
    )

    np.testing.assert_allclose(unbounded.x, _CENTER, rtol=0, atol=1e-10)
    assert bounded.x[0] == 1.0, "center 3.0 lies above the upper face 1.0"
    assert bounded.x[2] == 0.5, "center 0.25 lies below the lower face 0.5"
    assert bounded.x[3] == 6.5, "center 7.0 lies above the upper face 6.5"
    np.testing.assert_allclose(bounded.x[1], _CENTER[1], rtol=0, atol=1e-10)


def test_all_infinite_box_is_the_unbounded_problem() -> None:
    """``(-inf, inf)`` on every coordinate is how SciPy reads "no bound"."""
    free = ScipyBounds.from_arrays(np.full(4, -np.inf), np.full(4, np.inf))
    boxed = minimize(
        _value_and_gradient,
        np.zeros(4),
        driver=Driver.SCIPY_LBFGSB,
        options=_OPTIONS,
        bounds=free,
    )
    unbounded = minimize(
        _value_and_gradient, np.zeros(4), driver=Driver.SCIPY_LBFGSB, options=_OPTIONS
    )

    assert np.array_equal(boxed.x, unbounded.x)
    assert (boxed.nit, boxed.nfev) == (unbounded.nit, unbounded.nfev)


def test_bounded_route_over_a_device_objective_keeps_the_box() -> None:
    """A device ``x0`` routes each evaluation through the device and still respects the box."""
    center = jnp.asarray(_CENTER)
    curvature = jnp.asarray(_CURVATURE)

    @jax.jit
    def device_value_and_gradient(x):
        return jax.value_and_grad(
            lambda y: 0.5 * jnp.sum(curvature * (y - center) ** 2)
        )(x)

    box = ScipyBounds(lower=(-1.0, -1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0, 1.0))
    result = minimize(
        device_value_and_gradient,
        jax.device_put(jnp.zeros(4)),
        driver=Driver.SCIPY_LBFGSB,
        options=_OPTIONS,
        bounds=box,
    )

    np.testing.assert_array_equal(result.x[[0, 1, 3]], np.array([1.0, -1.0, 1.0]))
    np.testing.assert_allclose(result.x[2], 0.25, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    ("driver", "options"),
    [
        (Driver.SCIPY_BFGS, ScipyBFGSOptions(maxiter=5)),
        (Driver.SIMSOPT_LBFGSB, None),
    ],
)
def test_bounds_are_refused_by_drivers_that_would_ignore_them(driver, options) -> None:
    box = ScipyBounds(lower=(-1.0,) * 4, upper=(1.0,) * 4)

    with pytest.raises(ValueError, match="bounds are enforced only by"):
        minimize(
            _value_and_gradient,
            np.zeros(4),
            driver=driver,
            options=options,
            bounds=box,
        )


def test_scipy_bounds_is_an_immutable_validated_box() -> None:
    box = ScipyBounds.from_arrays(
        np.array([-16000.0 / 1.0e4, -np.inf]), [16000.0 / 1.0e4, np.inf]
    )

    assert box.lower == (-1.6, -np.inf)
    assert box.upper == (1.6, np.inf)
    assert all(type(value) is float for value in box.lower + box.upper)
    assert box.scipy_bounds() == [(-1.6, 1.6), (-np.inf, np.inf)]
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.lower = (0.0, 0.0)  # type: ignore[misc]
    with pytest.raises(ValueError, match="one upper per lower"):
        ScipyBounds(lower=(0.0, 0.0), upper=(1.0,))
    with pytest.raises(ValueError, match="lower bound must be <="):
        ScipyBounds(lower=(2.0,), upper=(1.0,))
    with pytest.raises(ValueError, match="lower bound must be <="):
        ScipyBounds(lower=(np.nan,), upper=(1.0,))
    with pytest.raises(ValueError, match="admits no finite point"):
        ScipyBounds(lower=(np.inf,), upper=(np.inf,))
    with pytest.raises(ValueError, match="admits no finite point"):
        ScipyBounds(lower=(-np.inf,), upper=(-np.inf,))
