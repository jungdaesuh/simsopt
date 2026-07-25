"""Public derivative ownership for curve-surface distance objectives."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import pytest

from simsopt.geo.curveobjectives import CurveSurfaceDistance
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt._core.optimizable import Optimizable
from simsopt_jax_adapters.geo.curve_objectives import CurveSurfaceDistanceJAX


CurveSurfaceObjectiveFactory = Callable[
    [List[CurveXYZFourier], SurfaceRZFourier, float],
    Union[CurveSurfaceDistance, CurveSurfaceDistanceJAX],
]


def _curve() -> CurveXYZFourier:
    curve = CurveXYZFourier(64, order=2)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("zs(2)", 0.1)
    return curve


def _surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        nphi=16,
        ntheta=16,
        ntor=0,
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.2)
    surface.set("zs(1,0)", 0.2)
    return surface


@pytest.mark.parametrize(
    "objective_factory",
    (CurveSurfaceDistance, CurveSurfaceDistanceJAX),
)
def test_curve_surface_distance_owns_and_differentiates_surface(
    objective_factory: CurveSurfaceObjectiveFactory,
) -> None:
    curve = _curve()
    surface = _surface()
    objective = objective_factory([curve], surface, 2.0)

    assert tuple(objective.parents) == (curve, surface)
    partials = objective.dJ(partials=True)
    surface_partial = np.asarray(partials(surface), dtype=np.float64)
    curve_partial = np.asarray(partials(curve), dtype=np.float64)
    assert surface_partial.shape == surface.x.shape
    assert curve_partial.shape == curve.x.shape

    surface_anchor = surface.x.copy()
    surface_direction = np.zeros_like(surface_anchor)
    surface_direction[0] = 1.0
    step = 1.0e-6
    surface.x = surface_anchor + step * surface_direction
    plus = float(objective.J())
    surface.x = surface_anchor - step * surface_direction
    minus = float(objective.J())
    surface.x = surface_anchor
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = float(surface_partial @ surface_direction)

    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-5, atol=2.0e-7)

    curve_anchor = curve.x.copy()
    curve_direction = np.zeros_like(curve_anchor)
    curve_direction[1] = 1.0
    curve.x = curve_anchor + step * curve_direction
    plus = float(objective.J())
    curve.x = curve_anchor - step * curve_direction
    minus = float(objective.J())
    curve.x = curve_anchor
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = float(curve_partial @ curve_direction)

    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-5, atol=2.0e-7)


class _SurfacePointCloud(Optimizable):
    def __init__(self, gamma: np.ndarray) -> None:
        self._gamma = gamma
        super().__init__()

    def gamma(self) -> np.ndarray:
        return self._gamma

    def normal(self) -> np.ndarray:
        return np.ones_like(self._gamma)


def test_curve_surface_distance_shortest_distance_respects_downsample() -> None:
    curve = _curve()
    surface_gamma = np.full((2, 2, 3), 100.0)
    surface_gamma[1, 1, :] = curve.gamma()[1] + np.array([1.0e-3, 0.0, 0.0])
    surface = _SurfacePointCloud(surface_gamma)

    full_resolution_distance = CurveSurfaceDistanceJAX(
        [curve],
        surface,
        minimum_distance=0.01,
    ).shortest_distance()
    downsampled_native = CurveSurfaceDistance(
        [curve],
        surface,
        minimum_distance=0.01,
        downsample=2,
    ).shortest_distance()
    downsampled_jax = CurveSurfaceDistanceJAX(
        [curve],
        surface,
        minimum_distance=0.01,
        downsample=2,
    ).shortest_distance()

    assert full_resolution_distance < 0.01
    assert downsampled_jax > 1.0
    assert downsampled_jax == pytest.approx(downsampled_native)


def test_curve_surface_distance_contract_names_both_derivative_owners() -> None:
    native_doc = inspect.getdoc(CurveSurfaceDistance.dJ)
    jax_doc = inspect.getdoc(CurveSurfaceDistanceJAX.dJ)
    docs_text = (
        Path(__file__).resolve().parents[2] / "docs" / "source" / "geo.rst"
    ).read_text(encoding="utf-8")

    assert native_doc is not None
    assert jax_doc is not None
    assert "curve and surface DOFs" in native_doc
    assert "curve and surface DOFs" in jax_doc
    assert "derivatives for both curve and surface DOFs" in docs_text
