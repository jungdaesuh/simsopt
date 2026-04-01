"""Pure curve-geometry helpers that operate on immutable specs."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from ..geo.curve import gamma_curve_on_surface
from ..geo.curverzfourier import curverzfourier_pure
from ..geo.curvexyzfourier import jaxfouriercurve_pure
from .specs import (
    CurveCWSFourierRZSpec,
    CurveRZFourierSpec,
    CurveSpec,
    CurveXYZFourierSpec,
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
    if isinstance(spec, CurveXYZFourierSpec):
        return lambda quadpoints: jaxfouriercurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if isinstance(spec, CurveRZFourierSpec):
        return lambda quadpoints: curverzfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
        )
    if isinstance(spec, CurveCWSFourierRZSpec):
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
    raise TypeError(f"Unsupported curve spec type: {type(spec).__name__}")


def _curve_quadpoints(spec: CurveSpec):
    quadpoints = jnp.asarray(spec.quadpoints, dtype=jnp.float64)
    return quadpoints, jnp.ones_like(quadpoints)


def curve_spec_with_dofs(spec: CurveSpec, dofs):
    return replace(spec, dofs=jnp.asarray(dofs, dtype=jnp.float64))


def curve_gamma_and_dash_from_spec(spec: CurveSpec):
    return curve_gamma_and_dash_from_dofs(spec, spec.dofs)


def curve_gamma_and_dash_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash) from a single kernel build and JVP call."""
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec)
    return jax.jvp(gamma_kernel, (quadpoints,), (tangents,))


def curve_geometry_from_spec(spec: CurveSpec):
    return curve_geometry_from_dofs(spec, spec.dofs)


def curve_geometry_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash, gammadashdash) from a single kernel build."""
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec)
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    return gamma, gammadash, gammadashdash
