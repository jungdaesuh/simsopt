from __future__ import annotations

import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit

from banana_opt.hardware_contracts import (
    TYPE_KK_INNER_RADIUS_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
)
from banana_opt.stage2_geometry import rotation_aware_projected_half_extent_m


@jit
def _lp_abs_hinge_pure(values, gammadash, p, threshold):
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1.0 / p) * jnp.mean(
        jnp.maximum(jnp.abs(values) - threshold, 0.0) ** p * arc_length
    )


class CurveSurfaceGeodesicCurvature(Optimizable):
    r"""Fold objective for a curve constrained to a winding surface frame.

    ``FramedCurveSurfaceTangent.frame_binormal_curvature()`` is
    :math:`\hat t'(s) \cdot \hat b`, where the unrotated surface-tangent
    binormal lies in the winding-surface tangent plane and perpendicular to the
    curve tangent. This is the signed geodesic curvature of the centerline with
    respect to that surface frame. The objective penalizes
    ``max(abs(kappa_g) - threshold, 0)^p`` with the same arc-length weighting as
    ``LpCurveCurvature``.

    If the supplied frame carries a nonzero material rotation, the same
    ``frame_binormal_curvature`` component is a material-frame binormal
    curvature, not the winding surface's geodesic curvature.
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
        self.grad_primal = jit(
            lambda geodesic_curvature, gammadash: grad(self.J_jax, argnums=(0, 1))(
                geodesic_curvature,
                gammadash,
            )
        )

    def geodesic_curvature(self):
        return np.asarray(self.framedcurve.frame_binormal_curvature(), dtype=float)

    def max_abs_geodesic_curvature(self):
        values = self.geodesic_curvature()
        return float(np.max(np.abs(values)))

    def frame_binormal_curvature(self):
        return self.geodesic_curvature()

    def max_abs_frame_binormal_curvature(self):
        return self.max_abs_geodesic_curvature()

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
        grad0, grad1 = self.grad_primal(geodesic_curvature, gammadash)
        return self.framedcurve.dframe_binormal_curvature_by_dcoeff_vjp(grad0) + (
            self.curve.dgammadash_by_dcoeff_vjp(grad1)
        )

    return_fn_map = {"J": J, "dJ": dJ}


@jit
def _normalized_curve_curvature_hinge_pure(kappa, gammadash, p, threshold):
    """Dimensionless arclength-weighted curvature hinge.

    Uses ``relu(|kappa| / threshold - 1)^p`` instead of
    ``relu(|kappa| - threshold)^p`` so the penalty scale is not tied to the
    numerical units of the curvature cap. The average is normalized by total
    arclength, making the value comparable across nearby discretizations.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(jnp.abs(kappa) / threshold - 1.0, 0.0) ** p
    return (1.0 / p) * jnp.sum(excess * arc_length) / jnp.sum(arc_length)


class NormalizedCurveCurvatureHinge(Optimizable):
    r"""Dimensionless normalized centerline curvature hinge.

    ``J = (1 / (p L)) * integral relu(|kappa| / kappa_max - 1)^p ds``.
    This is the normalized counterpart to SIMSOPT's raw-units
    ``LpCurveCurvature`` and is intended for p-continuation experiments where
    changing ``p`` should alter peak focus without also changing the basic units
    of the curvature objective.
    """

    def __init__(self, curve, p=4, threshold=1.0):
        if p < 1:
            raise ValueError("p must be >= 1")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.curve = curve
        self.p = int(p)
        self.threshold = float(threshold)
        super().__init__(depends_on=[curve])
        self.J_jax = jit(
            lambda kappa, gammadash: _normalized_curve_curvature_hinge_pure(
                kappa, gammadash, self.p, self.threshold
            )
        )
        self.grad_primal = jit(
            lambda kappa, gammadash: grad(self.J_jax, argnums=(0, 1))(
                kappa, gammadash
            )
        )

    def J(self):
        return float(self.J_jax(self.curve.kappa(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        kappa = self.curve.kappa()
        gammadash = self.curve.gammadash()
        grad_kappa, grad_gammadash = self.grad_primal(kappa, gammadash)
        return self.curve.dkappa_by_dcoeff_vjp(
            grad_kappa
        ) + self.curve.dgammadash_by_dcoeff_vjp(grad_gammadash)

    return_fn_map = {"J": J, "dJ": dJ}


@jit
def _projected_reach_pure(gammadash, gammadashdash, frame_n, frame_b, half_n, half_b):
    """Per-quadpoint outer-channel reach projected into the centerline bend plane.

    Differentiable JAX twin of ``rotation_aware_projected_half_extent_m``: the bend
    direction is the curvature vector (``gammadashdash`` projected perpendicular to
    the tangent), and the reach is the rectangle support function
    ``half_n*|n| + half_b*|b|`` (worst inner-corner half-extent) in the bend
    direction -- the exact non-fold quantity. The numpy SSOT supplies the J VALUE;
    this pure form supplies the gradient and is pinned byte-for-byte to the numpy
    function by the gradient-check test. (``jnp.abs`` has a benign subgradient kink
    at projection 0, where the relu hinge is already non-smooth.)
    """
    tangent_norm = jnp.linalg.norm(gammadash, axis=1)
    tangent = gammadash / tangent_norm[:, None]
    curvature_vector = (
        gammadashdash - jnp.sum(gammadashdash * tangent, axis=1)[:, None] * tangent
    )
    bend_norm = jnp.linalg.norm(curvature_vector, axis=1)
    safe_bend_norm = jnp.where(bend_norm > 0.0, bend_norm, 1.0)
    bend_direction = jnp.where(
        (bend_norm > 0.0)[:, None],
        curvature_vector / safe_bend_norm[:, None],
        jnp.zeros_like(curvature_vector),
    )
    n_component = jnp.sum(bend_direction * frame_n, axis=1)
    b_component = jnp.sum(bend_direction * frame_b, axis=1)
    return half_n * jnp.abs(n_component) + half_b * jnp.abs(b_component)


@jit
def _rotation_aware_excess_pure(
    kappa, gammadash, gammadashdash, frame_n, frame_b, p, margin, half_n, half_b
):
    """Arclength-weighted p-norm hinge of ``|kappa| - cap`` with a per-point cap.

    ``cap(theta) = 1 / (margin + reach(theta; alpha))`` where ``reach`` is the
    live rotated outer-channel projection. Implements the plan's
    ``J = (1/L) * sum_i relu(|kappa_i| - cap_i)^p * |gammadash_i|``,
    ``L = sum_i |gammadash_i|``.
    """
    reach = _projected_reach_pure(
        gammadash, gammadashdash, frame_n, frame_b, half_n, half_b
    )
    cap = 1.0 / (margin + reach)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(jnp.abs(kappa) - cap, 0.0) ** p
    return jnp.sum(excess * arc_length) / jnp.sum(arc_length)


class RotationAwareCurvatureExcessPenalty(Optimizable):
    r"""Per-point rotation-aware centerline curvature-excess penalty on a pack.

    ``J = (1/L) * sum relu(|kappa(theta)| - cap(theta))^p * |gammadash(theta)|`` with
    ``cap(theta) = 1/(margin + reach(theta; alpha))`` and ``reach`` the LIVE rotated
    outer-channel half-extent projected into the centerline bend plane (SSOT:
    ``rotation_aware_projected_half_extent_m``). ``dJ`` carries both gradient paths:
    through ``|kappa|`` to the ``banana_curve`` dofs and through ``cap -> reach ->
    framedcurve.rotated_frame()`` to the ``FrameRotation`` alpha dofs.
    """

    def __init__(
        self,
        framedcurve,
        banana_curve,
        finite_build,
        margin=TYPE_KK_INNER_RADIUS_MARGIN_M,
        p=6,
        half_n_m=TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
        half_b_m=TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
    ):
        self.framedcurve = framedcurve
        self.banana_curve = banana_curve
        self.finite_build = finite_build
        self.margin = float(margin)
        self.p = p
        self.half_n_m = float(half_n_m)
        self.half_b_m = float(half_b_m)
        super().__init__(depends_on=[framedcurve, banana_curve])
        self.J_jax = jit(
            lambda kappa, gammadash, gammadashdash, frame_n, frame_b: (
                _rotation_aware_excess_pure(
                    kappa,
                    gammadash,
                    gammadashdash,
                    frame_n,
                    frame_b,
                    p,
                    self.margin,
                    self.half_n_m,
                    self.half_b_m,
                )
            )
        )
        self.grad_primal = jit(
            lambda kappa, gammadash, gammadashdash, frame_n, frame_b: grad(
                self.J_jax, argnums=(0, 1, 2, 3, 4)
            )(kappa, gammadash, gammadashdash, frame_n, frame_b)
        )

    def _frame_axes(self):
        _, normal_axis, binormal_axis = (
            np.asarray(axis, dtype=float) for axis in self.framedcurve.rotated_frame()
        )
        return normal_axis, binormal_axis

    def rotation_aware_cap_inv_m(self):
        """Realized per-quadpoint cap ``1/(margin + reach(alpha(theta)))`` (1/m)."""
        reach = rotation_aware_projected_half_extent_m(
            self.finite_build, self.banana_curve, self.framedcurve
        )
        return 1.0 / (self.margin + np.asarray(reach, dtype=float))

    def J(self):
        normal_axis, binormal_axis = self._frame_axes()
        return float(
            self.J_jax(
                self.banana_curve.kappa(),
                self.banana_curve.gammadash(),
                self.banana_curve.gammadashdash(),
                normal_axis,
                binormal_axis,
            )
        )

    @derivative_dec
    def dJ(self):
        kappa = self.banana_curve.kappa()
        gammadash = self.banana_curve.gammadash()
        gammadashdash = self.banana_curve.gammadashdash()
        normal_axis, binormal_axis = self._frame_axes()
        args = (kappa, gammadash, gammadashdash, normal_axis, binormal_axis)
        g_kappa, g_gammadash, g_gammadashdash, g_frame_n, g_frame_b = (
            self.grad_primal(*args)
        )
        # (i) |kappa| path: kappa + arclength/bend-direction (gammadash, gammadashdash)
        # on the banana centerline dofs.
        curve_path = (
            self.banana_curve.dkappa_by_dcoeff_vjp(g_kappa)
            + self.banana_curve.dgammadash_by_dcoeff_vjp(g_gammadash)
            + self.banana_curve.dgammadashdash_by_dcoeff_vjp(g_gammadashdash)
        )
        # (ii) cap -> reach -> rotated_frame() path: cotangents on the normal and
        # binormal axes routed through the framedcurve VJP to the FrameRotation alpha
        # (and shared curve) dofs. The tangent-axis cotangent is zero.
        frame_path = self.framedcurve.rotated_frame_dcoeff_vjp(
            np.zeros_like(g_frame_n), g_frame_n, g_frame_b
        )
        return curve_path + frame_path

    return_fn_map = {"J": J, "dJ": dJ}
