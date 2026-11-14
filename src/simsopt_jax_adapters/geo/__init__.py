"""Geometry-object adapters for ``simsopt_jax``."""

from .curve_objectives import (
    JaxCurveCurveDistance,
    JaxCurveLength,
    JaxCurveSurfaceDistance,
    JaxLpCurveCurvature,
    JaxMeanSquaredCurvature,
)
from .surface_specs import surface_spec_from_surface

__all__ = (
    "JaxCurveCurveDistance",
    "JaxCurveLength",
    "JaxCurveSurfaceDistance",
    "JaxLpCurveCurvature",
    "JaxMeanSquaredCurvature",
    "surface_spec_from_surface",
)
