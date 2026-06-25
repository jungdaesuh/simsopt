"""Adapter-owned conversion from legacy curve objects to immutable JAX specs."""

from __future__ import annotations

import numpy as np

from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveperturbed import CurvePerturbed
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import (
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
    FramedCurveSurfaceTangent,
    ZeroRotation,
)
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax.core import (
    curve_spec_from_curve as _pure_curve_spec_from_curve,
    make_curve_cwsfourier_rz_spec,
    make_curve_filament_spec,
    make_curve_helical_spec,
    make_curve_perturbed_spec,
    make_curve_planarfourier_spec,
    make_curve_rzfourier_spec,
    make_curve_xyzfourier_spec,
    make_frame_rotation_spec,
    make_zero_rotation_spec,
)
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs
from simsopt_jax_adapters.geo.curve_contract import _optimizable_dof_map_spec

__all__ = [
    "adapter_curve_dof_mode",
    "curve_spec_from_adapter_curve",
    "supports_adapter_curve_spec",
]

_SURF_TYPE_RZ_FOURIER = "RZ_Fourier"


def supports_adapter_curve_spec(curve: object) -> bool:
    return (
        isinstance(
            curve,
            (
                CurveXYZFourier,
                CurveHelical,
                CurvePlanarFourier,
                CurveRZFourier,
                CurvePerturbed,
                CurveFilament,
            ),
        )
        or _is_cws_rz_fourier_curve(curve)
        or callable(getattr(curve, "to_spec", None))
    )


def adapter_curve_dof_mode(curve: object) -> str:
    if isinstance(curve, (CurvePerturbed, CurveFilament)):
        return "full"
    return getattr(curve, "_jax_curve_dof_mode", "local")


def curve_spec_from_adapter_curve(curve):
    if isinstance(curve, CurveXYZFourier):
        return make_curve_xyzfourier_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
        )
    if isinstance(curve, CurveHelical):
        return make_curve_helical_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
            m=curve.m,
            ell=curve.ell,
            R0=curve.R0,
            r=curve.r,
        )
    if isinstance(curve, CurvePlanarFourier):
        return make_curve_planarfourier_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
        )
    if isinstance(curve, CurveRZFourier):
        return make_curve_rzfourier_spec(
            dofs=curve.get_dofs(),
            quadpoints=curve.quadpoints,
            order=curve.order,
            nfp=curve.nfp,
            stellsym=curve.stellsym,
        )
    if isinstance(curve, CurvePerturbed):
        return _curve_perturbed_spec_from_curve(curve)
    if isinstance(curve, CurveFilament):
        return _curve_filament_spec_from_curve(curve)
    if _is_cws_rz_fourier_curve(curve):
        return _curve_cws_rz_fourier_spec_from_curve(curve)

    to_spec = getattr(curve, "to_spec", None)
    if callable(to_spec):
        return to_spec()

    return _pure_curve_spec_from_curve(curve)


def _curve_perturbed_spec_from_curve(curve: CurvePerturbed):
    sample_gamma = curve.sample[0]
    sample_gammadash = curve.sample[1]
    sample_gammadashdash = (
        curve.sample[2] if len(curve.sample._sample) > 2 else np.zeros_like(sample_gamma)
    )
    sample_gammadashdashdash = (
        curve.sample[3] if len(curve.sample._sample) > 3 else np.zeros_like(sample_gamma)
    )

    return make_curve_perturbed_spec(
        dofs=curve.full_x,
        quadpoints=curve.quadpoints,
        base_curve=curve_spec_from_adapter_curve(curve.curve),
        base_curve_map=_optimizable_dof_map_spec(curve, curve.curve),
        sample_gamma=sample_gamma,
        sample_gammadash=sample_gammadash,
        sample_gammadashdash=sample_gammadashdash,
        sample_gammadashdashdash=sample_gammadashdashdash,
    )


def _curve_filament_spec_from_curve(curve: CurveFilament):
    frame_kind, surface_major_radius, surface_midplane_z = _frame_spec_from_framed_curve(
        curve.framedcurve
    )
    return make_curve_filament_spec(
        dofs=curve.full_x,
        quadpoints=curve.quadpoints,
        base_curve=curve_spec_from_adapter_curve(curve.curve),
        base_curve_map=_optimizable_dof_map_spec(curve, curve.curve),
        rotation=_rotation_spec_from_curve(curve.rotation, curve.curve.quadpoints),
        rotation_map=_optimizable_dof_map_spec(curve, curve.rotation),
        frame_kind=frame_kind,
        dn=curve.dn,
        db=curve.db,
        surface_major_radius=surface_major_radius,
        surface_midplane_z=surface_midplane_z,
    )


def _frame_spec_from_framed_curve(framed_curve):
    if isinstance(framed_curve, FramedCurveCentroid):
        return "centroid", None, None
    if isinstance(framed_curve, FramedCurveFrenet):
        return "frenet", None, None
    if isinstance(framed_curve, FramedCurveSurfaceTangent):
        return (
            "surface_tangent",
            framed_curve.major_radius,
            framed_curve.midplane_z,
        )
    raise NotImplementedError(
        "CurveFilament JAX spec conversion supports FramedCurveCentroid, "
        "FramedCurveFrenet, and FramedCurveSurfaceTangent, got "
        f"{type(framed_curve).__name__}."
    )


def _rotation_spec_from_curve(rotation, quadpoints):
    if isinstance(rotation, ZeroRotation):
        return make_zero_rotation_spec(quadpoints=quadpoints)
    if isinstance(rotation, FrameRotation):
        return make_frame_rotation_spec(
            dofs=rotation.full_x,
            quadpoints=quadpoints,
            order=rotation.order,
            scale=rotation.scale,
        )
    raise NotImplementedError(
        "CurveFilament JAX spec conversion supports FrameRotation and "
        f"ZeroRotation, got {type(rotation).__name__}."
    )


def _is_cws_rz_fourier_curve(curve: object) -> bool:
    surface = getattr(curve, "surf", None)
    return (
        surface is not None
        and getattr(curve, "surf_type", None) == _SURF_TYPE_RZ_FOURIER
        and isinstance(surface, SurfaceRZFourier)
    )


def _curve_cws_rz_fourier_spec_from_curve(curve):
    surface = curve.surf
    return make_curve_cwsfourier_rz_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        surface=surface_rz_fourier_spec_from_dofs(
            surface.get_dofs(),
            quadpoints_phi=surface.quadpoints_phi,
            quadpoints_theta=surface.quadpoints_theta,
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
        ),
        order=curve.order,
        G=getattr(curve, "G", 0.0),
        H=getattr(curve, "H", 0.0),
    )
