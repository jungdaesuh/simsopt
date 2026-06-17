from __future__ import annotations

import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit


@jit
def _lp_abs_hinge_pure(values, gammadash, p, threshold):
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1.0 / p) * jnp.mean(
        jnp.maximum(jnp.abs(values) - threshold, 0.0) ** p * arc_length
    )


class CurveSurfaceGeodesicCurvature(Optimizable):
    r"""Fold objective for a curve constrained to a winding surface.

    ``FramedCurveSurfaceTangent.frame_binormal_curvature()`` is
    :math:`\hat t'(s) \cdot \hat b`, where the unrotated surface-tangent
    binormal lies in the winding-surface tangent plane and perpendicular to the
    curve tangent. This is the signed geodesic curvature of the centerline with
    respect to that surface frame. The objective penalizes
    ``max(abs(kappa_g) - threshold, 0)^p`` with the same arc-length weighting as
    ``LpCurveCurvature``.
    """

    def __init__(self, framedcurve, p=2, threshold=0.0):
        self.framedcurve = framedcurve
        self.curve = framedcurve.curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[framedcurve])
        self.J_jax = jit(
            lambda geodesic_curvature, gammadash: _lp_abs_hinge_pure(
                geodesic_curvature,
                gammadash,
                p,
                threshold,
            )
        )
        self.grad0 = jit(
            lambda geodesic_curvature, gammadash: grad(self.J_jax, argnums=0)(
                geodesic_curvature,
                gammadash,
            )
        )
        self.grad1 = jit(
            lambda geodesic_curvature, gammadash: grad(self.J_jax, argnums=1)(
                geodesic_curvature,
                gammadash,
            )
        )

    def geodesic_curvature(self):
        return np.asarray(self.framedcurve.frame_binormal_curvature(), dtype=float)

    def max_abs_geodesic_curvature(self):
        values = self.geodesic_curvature()
        return float(np.max(np.abs(values)))

    def J(self):
        return float(
            self.J_jax(
                self.framedcurve.frame_binormal_curvature(),
                self.curve.gammadash(),
            )
        )

    @derivative_dec
    def dJ(self):
        geodesic_curvature = self.framedcurve.frame_binormal_curvature()
        gammadash = self.curve.gammadash()
        grad0 = self.grad0(geodesic_curvature, gammadash)
        grad1 = self.grad1(geodesic_curvature, gammadash)
        return self.framedcurve.dframe_binormal_curvature_by_dcoeff_vjp(grad0) + (
            self.curve.dgammadash_by_dcoeff_vjp(grad1)
        )

    return_fn_map = {"J": J, "dJ": dJ}
