"""Pure curve-geometry helpers that operate on immutable specs."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from ..geo.curve import gamma_curve_on_surface
from ..geo.framedcurve import (
    jaxrotation_pure,
    jaxrotationdash_pure,
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotated_frenet_frame,
    rotated_frenet_frame_dash,
)
from ..geo.curvehelical import curve_helical_pure
from ..geo.curveplanarfourier import curveplanarfourier_pure
from ..geo.curverzfourier import curverzfourier_pure
from ..geo.curvexyzfourier import (
    jaxfouriercurve_geometry_pure,
    jaxfouriercurve_pure,
)
from ._math_utils import as_runtime_float64 as _as_runtime_float64
from .specs import (
    CurveCWSFourierRZSpec,
    CurveFilamentSpec,
    CurveHelicalSpec,
    CurvePlanarFourierSpec,
    CurvePerturbedSpec,
    CurveRZFourierSpec,
    CurveSpec,
    CurveXYZFourierSpec,
    OptimizableDofMapSpec,
    RotationSpec,
    ZeroRotationSpec,
    curve_spec_kind,
    make_curve_cwsfourier_rz_spec,
)

_SURF_TYPE_RZ_FOURIER = "RZ_Fourier"


def _as_explicit_float64(value, *, reference=None) -> jax.Array:
    if reference is not None:
        return _as_runtime_float64(value, reference=reference)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return jax.device_put(np.asarray(value, dtype=np.float64))
    return jnp.asarray(value, dtype=jnp.float64)


def _explicit_scalar(value: float, *, reference=None) -> jax.Array:
    return _as_explicit_float64(value, reference=reference)


def _ones_like_float64(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_explicit_scalar(1.0, reference=array), array.shape)


def _zeros_like_float64(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_explicit_scalar(0.0, reference=array), array.shape)


def curve_spec_from_curve(curve):
    to_spec = getattr(curve, "to_spec", None)
    if callable(to_spec):
        return to_spec()

    surface = getattr(curve, "surf", None)
    if surface is None:
        raise NotImplementedError(
            f"Curve type {type(curve).__name__} does not expose an immutable JAX spec."
        )

    surface_spec_fn = getattr(surface, "surface_spec", None)
    if not callable(surface_spec_fn):
        raise NotImplementedError(
            f"Surface type {type(surface).__name__} does not expose surface_spec()."
        )

    if getattr(curve, "surf_type", None) != _SURF_TYPE_RZ_FOURIER:
        raise NotImplementedError(
            "CWS spec generation requires surf_type='RZ_Fourier', "
            f"got {getattr(curve, 'surf_type', None)!r}."
        )

    return make_curve_cwsfourier_rz_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        surface=surface_spec_fn(),
        order=curve.order,
        G=getattr(curve, "G", 0.0),
        H=getattr(curve, "H", 0.0),
    )


def _curve_gamma_kernel(spec: CurveSpec, dofs=None):
    curve_dofs = spec.dofs if dofs is None else _as_explicit_float64(dofs)
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "xyz_fourier":
        spec = cast(CurveXYZFourierSpec, spec)
        return lambda quadpoints: jaxfouriercurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if spec_kind == "rz_fourier":
        spec = cast(CurveRZFourierSpec, spec)
        return lambda quadpoints: curverzfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
        )
    if spec_kind == "planar_fourier":
        spec = cast(CurvePlanarFourierSpec, spec)
        return lambda quadpoints: curveplanarfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if spec_kind == "helical":
        spec = cast(CurveHelicalSpec, spec)
        return lambda quadpoints: curve_helical_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.m,
            spec.ell,
            spec.R0,
            spec.r,
        )
    if spec_kind == "cws_fourier_rz":
        spec = cast(CurveCWSFourierRZSpec, spec)
        surface_dofs = spec.surface_dofs()
        return lambda quadpoints: gamma_curve_on_surface(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.G,
            spec.H,
            surface_dofs,
            _SURF_TYPE_RZ_FOURIER,
            spec.surface.mpol,
            spec.surface.ntor,
            spec.surface.nfp,
            spec.surface.stellsym,
        )
    raise TypeError(
        "curve_gamma_kernel only supports direct curve specs, "
        f"got {type(spec).__name__}."
    )


def _curve_quadpoints(spec: CurveSpec, *, reference):
    quadpoints = _as_explicit_float64(spec.quadpoints, reference=reference)
    return quadpoints, _ones_like_float64(quadpoints)


def _curve_geometry_terms_from_kernel(gamma_kernel, quadpoints, tangents, *, order):
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    if order == 1:
        return gamma, gammadash

    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    if order == 2:
        return gamma, gammadash, gammadashdash

    gammadashdash_kernel = lambda qp: jax.jvp(
        gammadash_kernel,
        (qp,),
        (tangents,),
    )[1]
    _, gammadashdashdash = jax.jvp(
        gammadashdash_kernel,
        (quadpoints,),
        (tangents,),
    )
    return gamma, gammadash, gammadashdash, gammadashdashdash


def _direct_curve_geometry_terms(spec: CurveSpec, dofs, *, order):
    if curve_spec_kind(spec) != "xyz_fourier":
        return None
    spec = cast(CurveXYZFourierSpec, spec)
    curve_dofs = spec.dofs if dofs is None else dofs
    geometry = jaxfouriercurve_geometry_pure(
        curve_dofs,
        spec.quadpoints,
        spec.order,
    )
    return geometry[: order + 1]


def _mapped_full_dofs(map_spec: OptimizableDofMapSpec, owner_dofs):
    mapped = _as_explicit_float64(map_spec.template_full_dofs, reference=owner_dofs)
    owner_dofs = _as_explicit_float64(owner_dofs, reference=owner_dofs)
    for owner_start, owner_end, target_start, target_end in map_spec.owner_segments:
        segment = jax.lax.slice_in_dim(owner_dofs, owner_start, owner_end, axis=0)
        mapped = mapped.at[target_start:target_end].set(
            segment
        )
    return mapped


def _mapped_input_dofs(map_spec: OptimizableDofMapSpec, owner_dofs):
    mapped_full = _mapped_full_dofs(map_spec, owner_dofs)
    if map_spec.input_mode == "full":
        return mapped_full
    return jax.lax.slice_in_dim(
        mapped_full,
        map_spec.input_start,
        map_spec.input_end,
        axis=0,
    )


def optimizable_full_dofs_from_map_spec(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
):
    return _mapped_full_dofs(map_spec, owner_dofs)


def optimizable_input_dofs_from_map_spec(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
):
    return _mapped_input_dofs(map_spec, owner_dofs)


def _rotation_alpha_and_dash_from_dofs(
    rotation_spec: RotationSpec,
    rotation_map: OptimizableDofMapSpec,
    owner_dofs,
):
    quadpoints = _as_explicit_float64(rotation_spec.quadpoints, reference=owner_dofs)
    if isinstance(rotation_spec, ZeroRotationSpec):
        zeros = _zeros_like_float64(quadpoints)
        return zeros, zeros

    rotation_dofs = _mapped_input_dofs(rotation_map, owner_dofs)
    rotation_scale = _explicit_scalar(rotation_spec.scale, reference=owner_dofs)
    return (
        rotation_scale
        * jaxrotation_pure(rotation_dofs, quadpoints, rotation_spec.order),
        rotation_scale
        * jaxrotationdash_pure(rotation_dofs, quadpoints, rotation_spec.order),
    )


def _curve_geometry_with_third_derivative_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash, gammadashdash, gammadashdashdash) in one pass."""
    if isinstance(spec, CurvePerturbedSpec):
        base_geometry = _curve_geometry_with_third_derivative_from_dofs(
            spec.base_curve,
            _curve_perturbed_base_dofs(spec, dofs),
        )
        return _add_curve_perturbation(spec, *base_geometry)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=3)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=3,
    )


def _curve_perturbed_base_dofs(spec: CurvePerturbedSpec, dofs):
    return _mapped_input_dofs(spec.base_curve_map, dofs)


def _add_curve_perturbation(spec: CurvePerturbedSpec, *geometry_terms):
    sample_terms = (
        spec.sample_gamma,
        spec.sample_gammadash,
        spec.sample_gammadashdash,
        spec.sample_gammadashdashdash,
    )
    return tuple(
        geometry_term + sample_term
        for geometry_term, sample_term in zip(geometry_terms, sample_terms)
    )


def _curve_perturbed_gamma_and_dash_from_dofs(spec: CurvePerturbedSpec, dofs):
    base_geometry = curve_gamma_and_dash_from_dofs(
        spec.base_curve,
        _curve_perturbed_base_dofs(spec, dofs),
    )
    return _add_curve_perturbation(spec, *base_geometry)


def _curve_perturbed_geometry_from_dofs(spec: CurvePerturbedSpec, dofs):
    base_geometry = curve_geometry_from_dofs(
        spec.base_curve,
        _curve_perturbed_base_dofs(spec, dofs),
    )
    return _add_curve_perturbation(spec, *base_geometry)


def _curve_spec_with_quadpoints(spec: CurveSpec, quadpoints):
    quadpoints_jax = _as_explicit_float64(quadpoints)
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return replace(
            spec,
            quadpoints=quadpoints_jax,
            base_curve=_curve_spec_with_quadpoints(spec.base_curve, quadpoints_jax),
        )
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return replace(
            spec,
            quadpoints=quadpoints_jax,
            base_curve=_curve_spec_with_quadpoints(spec.base_curve, quadpoints_jax),
            rotation=replace(spec.rotation, quadpoints=quadpoints_jax),
        )
    return replace(spec, quadpoints=quadpoints_jax)


def _curve_filament_geometry_from_dofs(spec: CurveFilamentSpec, dofs):
    def gamma_kernel(qp):
        quad_spec = _curve_spec_with_quadpoints(spec, qp)
        base_dofs = _mapped_input_dofs(quad_spec.base_curve_map, dofs)
        alpha, _alphadash = _rotation_alpha_and_dash_from_dofs(
            quad_spec.rotation,
            quad_spec.rotation_map,
            dofs,
        )
        gamma, gammadash = curve_gamma_and_dash_from_dofs(
            quad_spec.base_curve, base_dofs
        )
        if quad_spec.frame_kind == "frenet":
            _gamma, _gammadash, gammadashdash = curve_geometry_from_dofs(
                quad_spec.base_curve,
                base_dofs,
            )
            _tangent, normal, binormal = rotated_frenet_frame(
                gamma,
                gammadash,
                gammadashdash,
                alpha,
            )
        else:
            _tangent, normal, binormal = rotated_centroid_frame(
                gamma,
                gammadash,
                alpha,
            )
        return gamma + quad_spec.dn * normal + quad_spec.db * binormal

    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    return gamma, gammadash, gammadashdash


def _curve_filament_gamma_and_dash_from_dofs(spec: CurveFilamentSpec, dofs):
    base_dofs = _mapped_input_dofs(spec.base_curve_map, dofs)
    alpha, alphadash = _rotation_alpha_and_dash_from_dofs(
        spec.rotation,
        spec.rotation_map,
        dofs,
    )

    if spec.frame_kind == "frenet":
        gamma, gammadash, gammadashdash, gammadashdashdash = (
            _curve_geometry_with_third_derivative_from_dofs(spec.base_curve, base_dofs)
        )
        _tangent, normal, binormal = rotated_frenet_frame(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
        )
        tangent_dash, normal_dash, binormal_dash = rotated_frenet_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
        )
    else:
        gamma, gammadash, gammadashdash = curve_geometry_from_dofs(
            spec.base_curve, base_dofs
        )
        _tangent, normal, binormal = rotated_centroid_frame(
            gamma,
            gammadash,
            alpha,
        )
        _tangent_dash, normal_dash, binormal_dash = rotated_centroid_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
        )
    return (
        gamma + spec.dn * normal + spec.db * binormal,
        gammadash + spec.dn * normal_dash + spec.db * binormal_dash,
    )


def curve_spec_with_dofs(spec: CurveSpec, dofs):
    return replace(spec, dofs=_as_explicit_float64(dofs))


def curve_gamma_and_dash_from_spec(spec: CurveSpec):
    return curve_gamma_and_dash_from_dofs(spec, spec.dofs)


def curve_gamma_and_dash_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash) from a single kernel build and JVP call."""
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return _curve_perturbed_gamma_and_dash_from_dofs(spec, dofs)
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return _curve_filament_gamma_and_dash_from_dofs(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=1)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=1,
    )


def curve_geometry_from_spec(spec: CurveSpec):
    return curve_geometry_from_dofs(spec, spec.dofs)


def curve_geometry_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash, gammadashdash) from a single kernel build."""
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return _curve_perturbed_geometry_from_dofs(spec, dofs)
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return _curve_filament_geometry_from_dofs(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=2)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=2,
    )


def _curve_cws_gamma_and_dash_from_parts(
    spec: CurveCWSFourierRZSpec,
    curve_dofs,
    surface_dofs,
):
    quadpoints, tangents = _curve_quadpoints(spec, reference=curve_dofs)

    def gamma_kernel(qp):
        return gamma_curve_on_surface(
            curve_dofs,
            qp,
            spec.order,
            spec.G,
            spec.H,
            surface_dofs,
            _SURF_TYPE_RZ_FOURIER,
            spec.surface.mpol,
            spec.surface.ntor,
            spec.surface.nfp,
            spec.surface.stellsym,
        )

    return jax.jvp(gamma_kernel, (quadpoints,), (tangents,))


def curve_pullback_from_spec(spec: CurveSpec, dg, dgd):
    return curve_pullback_from_dofs(spec, spec.dofs, dg, dgd)


def curve_pullback_from_dofs(spec: CurveSpec, dofs, dg, dgd):
    """Return coefficient and optional surface cotangents for one curve spec."""
    curve_dofs = _as_explicit_float64(dofs)
    dg_jax = _as_explicit_float64(dg)
    dgd_jax = _as_explicit_float64(dgd)

    if curve_spec_kind(spec) == "cws_fourier_rz":
        spec = cast(CurveCWSFourierRZSpec, spec)
        surface_dofs = spec.surface_dofs()

        def outputs(curve_x, surface_x):
            return _curve_cws_gamma_and_dash_from_parts(spec, curve_x, surface_x)

        _, pullback = jax.vjp(outputs, curve_dofs, surface_dofs)
        coeff_cotangent, surface_cotangent = pullback((dg_jax, dgd_jax))
        return coeff_cotangent, surface_cotangent

    def outputs(curve_x):
        return curve_gamma_and_dash_from_dofs(spec, curve_x)

    _, pullback = jax.vjp(outputs, curve_dofs)
    (coeff_cotangent,) = pullback((dg_jax, dgd_jax))
    return coeff_cotangent, None
