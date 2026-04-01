"""Pure curve-geometry helpers that operate on immutable specs."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp

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
from ..geo.curvexyzfourier import jaxfouriercurve_pure
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
    curve_dofs = spec.dofs if dofs is None else jnp.asarray(dofs, dtype=jnp.float64)
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


def _curve_quadpoints(spec: CurveSpec):
    quadpoints = jnp.asarray(spec.quadpoints, dtype=jnp.float64)
    return quadpoints, jnp.ones_like(quadpoints)


def _mapped_full_dofs(map_spec: OptimizableDofMapSpec, owner_dofs):
    mapped = jnp.asarray(map_spec.template_full_dofs, dtype=jnp.float64)
    owner_dofs = jnp.asarray(owner_dofs, dtype=jnp.float64)
    for owner_start, owner_end, target_start, target_end in map_spec.owner_segments:
        mapped = mapped.at[target_start:target_end].set(
            owner_dofs[owner_start:owner_end]
        )
    return mapped


def _mapped_input_dofs(map_spec: OptimizableDofMapSpec, owner_dofs):
    mapped_full = _mapped_full_dofs(map_spec, owner_dofs)
    if map_spec.input_mode == "full":
        return mapped_full
    return mapped_full[map_spec.input_start : map_spec.input_end]


def _rotation_alpha_and_dash_from_dofs(
    rotation_spec: RotationSpec,
    rotation_map: OptimizableDofMapSpec,
    owner_dofs,
):
    quadpoints = jnp.asarray(rotation_spec.quadpoints, dtype=jnp.float64)
    if isinstance(rotation_spec, ZeroRotationSpec):
        zeros = jnp.zeros_like(quadpoints)
        return zeros, zeros

    rotation_dofs = _mapped_input_dofs(rotation_map, owner_dofs)
    return (
        rotation_spec.scale
        * jaxrotation_pure(rotation_dofs, quadpoints, rotation_spec.order),
        rotation_spec.scale
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
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec)
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
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
    quadpoints_jax = jnp.asarray(quadpoints, dtype=jnp.float64)
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

    quadpoints, tangents = _curve_quadpoints(spec)
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
    return replace(spec, dofs=jnp.asarray(dofs, dtype=jnp.float64))


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
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec)
    return jax.jvp(gamma_kernel, (quadpoints,), (tangents,))


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
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec)
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    return gamma, gammadash, gammadashdash


def _curve_cws_gamma_and_dash_from_parts(
    spec: CurveCWSFourierRZSpec,
    curve_dofs,
    surface_dofs,
):
    quadpoints, tangents = _curve_quadpoints(spec)

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
    curve_dofs = jnp.asarray(dofs, dtype=jnp.float64)
    dg_jax = jnp.asarray(dg, dtype=jnp.float64)
    dgd_jax = jnp.asarray(dgd, dtype=jnp.float64)

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
