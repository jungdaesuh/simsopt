"""Strict host-ownership regressions for native curve geometry."""

from __future__ import annotations

import jax
import numpy as np
from typing import Type, Union

from simsopt.geo.curve import kappa_pure
from simsopt.geo.curveobjectives import CurveLength
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier


CurveType = Union[Type[CurveXYZFourier], Type[JaxCurveXYZFourier]]


def _unit_circle(curve_type: CurveType) -> Union[CurveXYZFourier, JaxCurveXYZFourier]:
    curve = curve_type(32, order=2)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    return curve


def test_native_curve_length_stays_host_owned_under_strict_transfer_guard() -> None:
    curve = _unit_circle(CurveXYZFourier)
    objective = CurveLength(curve)
    incremental_arclength = curve.incremental_arclength()
    expected_value = np.mean(incremental_arclength)
    expected_derivative = curve.dincremental_arclength_by_dcoeff_vjp(
        np.full_like(incremental_arclength, 1.0 / incremental_arclength.size)
    )(curve)

    with jax.transfer_guard("disallow"):
        value = objective.J()
        derivative = objective.dJ()

    assert isinstance(value, np.floating)
    np.testing.assert_array_equal(value, expected_value)
    np.testing.assert_allclose(
        derivative, expected_derivative, rtol=1.0e-14, atol=1.0e-14
    )


def test_native_curvature_stays_host_owned_without_changing_jax_curve() -> None:
    native_curve = _unit_circle(CurveXYZFourier)
    jax_curve = _unit_circle(JaxCurveXYZFourier)
    expected_native = (
        np.linalg.norm(
            np.cross(native_curve.gammadash(), native_curve.gammadashdash()),
            axis=1,
        )
        / np.linalg.norm(native_curve.gammadash(), axis=1) ** 3
    )

    with jax.transfer_guard("disallow"):
        native_curvature = native_curve.kappa().copy()

    expected_jax = np.asarray(
        kappa_pure(jax_curve.gammadash(), jax_curve.gammadashdash())
    )
    np.testing.assert_allclose(
        native_curvature, expected_native, rtol=1.0e-15, atol=0.0
    )
    np.testing.assert_allclose(jax_curve.kappa(), expected_jax, rtol=1.0e-15, atol=0.0)
