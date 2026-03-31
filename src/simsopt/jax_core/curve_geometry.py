"""Pure curve-geometry helpers that operate on immutable specs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geo.curverzfourier import curverzfourier_pure
from ..geo.curvexyzfourier import jaxfouriercurve_pure
from .specs import CurveRZFourierSpec, CurveSpec, CurveXYZFourierSpec


def _curve_gamma_kernel(spec: CurveSpec):
    if isinstance(spec, CurveXYZFourierSpec):
        return lambda quadpoints: jaxfouriercurve_pure(
            spec.dofs,
            quadpoints,
            spec.order,
        )
    if isinstance(spec, CurveRZFourierSpec):
        return lambda quadpoints: curverzfourier_pure(
            spec.dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
        )
    raise TypeError(f"Unsupported curve spec type: {type(spec).__name__}")


def _curve_quadpoints(spec: CurveSpec):
    quadpoints = jnp.asarray(spec.quadpoints, dtype=jnp.float64)
    return quadpoints, jnp.ones_like(quadpoints)


def curve_gamma_from_spec(spec: CurveSpec):
    gamma_kernel = _curve_gamma_kernel(spec)
    quadpoints, _ = _curve_quadpoints(spec)
    return gamma_kernel(quadpoints)


def curve_gammadash_from_spec(spec: CurveSpec):
    gamma_kernel = _curve_gamma_kernel(spec)
    quadpoints, quadpoint_tangents = _curve_quadpoints(spec)
    return jax.jvp(
        gamma_kernel,
        (quadpoints,),
        (quadpoint_tangents,),
    )[1]
