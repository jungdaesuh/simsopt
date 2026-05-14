import numpy as np
from jax import vjp
import jax.numpy as jnp

from .framedcurve import (
    FramedCurve,
    FrameRotation,
    ZeroRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotated_frenet_frame,
    rotated_frenet_frame_dash,
)
from .curve import (
    _curve_jax_eval_from_arg,
    _curve_jax_arg_from_full_dofs,
    _optimizable_dof_map_spec,
    _optimizable_local_full_dofs_from_full_dofs,
)
from .jit import jit

"""
The functions and classes in this model are used to deal with multifilament
approximation of finite build coils.
"""

__all__ = ["create_multifilament_grid", "CurveFilament"]


class CurveFilament(FramedCurve):
    def __init__(self, framedcurve, dn, db):
        """
        Given a FramedCurve, defining a normal and
        binormal vector, create a grid of curves by shifting
        along the normal and binormal vector.

        The idea is explained well in Figure 1 in the reference:

        Singh et al, "Optimization of finite-build stellarator coils",
        Journal of Plasma Physics 86 (2020),
        doi:10.1017/S0022377820000756.

        Args:
            curve: the underlying curve
            dn: how far to move in normal direction
            db: how far to move in binormal direction
            rotation: angle along the curve to rotate the frame.
        """
        self.curve = framedcurve.curve
        self.dn = dn
        self.db = db
        self.rotation = framedcurve.rotation
        self.framedcurve = framedcurve
        FramedCurve.__init__(self, self.curve, self.rotation)
        self._jax_curve_dof_mode = "full"

        if self._supports_jax_finitebuild_geometry():
            self.gamma_jax = jit(self._gamma_jax_from_full_dofs)
            self.dgamma_by_dcoeff_vjp_jax = jit(
                lambda dofs, cotangent: vjp(self.gamma_jax, dofs)[1](cotangent)[0]
            )
            self.gammadash_jax = jit(self._gammadash_jax_from_full_dofs)
            self.dgammadash_by_dcoeff_vjp_jax = jit(
                lambda dofs, cotangent: vjp(self.gammadash_jax, dofs)[1](cotangent)[0]
            )

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        t, n, b = self.framedcurve.rotated_frame()
        gamma[:] = self.curve.gamma() + self.dn * n + self.db * b

    def gammadash_impl(self, gammadash):
        td, nd, bd = self.framedcurve.rotated_frame_dash()
        gammadash[:] = self.curve.gammadash() + self.dn * nd + self.db * bd

    def dgamma_by_dcoeff_vjp(self, v):
        zero = np.zeros_like(v)
        return self.curve.dgamma_by_dcoeff_vjp(
            v
        ) + self.framedcurve.rotated_frame_dcoeff_vjp(zero, self.dn * v, self.db * v)

    def dgammadash_by_dcoeff_vjp(self, v):
        zero = np.zeros_like(v)
        return self.curve.dgammadash_by_dcoeff_vjp(
            v
        ) + self.framedcurve.rotated_frame_dash_dcoeff_vjp(
            zero, self.dn * v, self.db * v
        )

    def _rotation_jax_values(self, dofs):
        points = jnp.asarray(self.curve.quadpoints, dtype=jnp.float64)
        if (
            isinstance(self.rotation, ZeroRotation)
            or self.rotation.local_full_dof_size == 0
        ):
            zeros = jnp.zeros_like(points)
            return zeros, zeros

        rotation_dofs = _optimizable_local_full_dofs_from_full_dofs(
            self,
            self.rotation,
            dofs,
        )
        alpha = self.rotation.scale * self.rotation.jax_alpha(rotation_dofs, points)
        alphadash = self.rotation.scale * self.rotation.jax_alphadash(
            rotation_dofs,
            points,
        )
        return alpha, alphadash

    def _curve_jax_geometry(self, dofs, surf_dofs=None):
        curve_dofs = _curve_jax_arg_from_full_dofs(self, self.curve, dofs)
        gamma = _curve_jax_eval_from_arg(
            self.curve,
            "gamma_jax",
            curve_dofs,
            surf_dofs=surf_dofs,
        )
        gammadash = _curve_jax_eval_from_arg(
            self.curve,
            "gammadash_jax",
            curve_dofs,
            surf_dofs=surf_dofs,
        )
        return curve_dofs, gamma, gammadash

    def _supports_jax_finitebuild_geometry(self):
        if not (
            hasattr(self.curve, "gamma_jax")
            and hasattr(self.curve, "gammadash_jax")
            and hasattr(self.curve, "gammadashdash_jax")
        ):
            return False
        if isinstance(self.framedcurve, FramedCurveFrenet):
            return hasattr(self.curve, "gammadashdashdash_jax")
        return isinstance(self.framedcurve, FramedCurveCentroid)

    def _gamma_jax_from_full_dofs(self, dofs):
        curve_dofs, gamma, gammadash = self._curve_jax_geometry(dofs)
        alpha, _alphadash = self._rotation_jax_values(dofs)

        if isinstance(self.framedcurve, FramedCurveFrenet):
            gammadashdash = _curve_jax_eval_from_arg(
                self.curve,
                "gammadashdash_jax",
                curve_dofs,
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

        return gamma + self.dn * normal + self.db * binormal

    def _gammadash_jax_from_full_dofs(self, dofs):
        curve_dofs, gamma, gammadash = self._curve_jax_geometry(dofs)
        alpha, alphadash = self._rotation_jax_values(dofs)

        if isinstance(self.framedcurve, FramedCurveFrenet):
            gammadashdash = _curve_jax_eval_from_arg(
                self.curve,
                "gammadashdash_jax",
                curve_dofs,
            )
            gammadashdashdash = _curve_jax_eval_from_arg(
                self.curve,
                "gammadashdashdash_jax",
                curve_dofs,
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
            gammadashdash = _curve_jax_eval_from_arg(
                self.curve,
                "gammadashdash_jax",
                curve_dofs,
            )
            tangent_dash, normal_dash, binormal_dash = rotated_centroid_frame_dash(
                gamma,
                gammadash,
                gammadashdash,
                alpha,
                alphadash,
            )

        return gammadash + self.dn * normal_dash + self.db * binormal_dash

    def to_spec(self):
        """Build an immutable JAX geometry spec from the current wrapper state."""
        from ..jax_core import (
            curve_spec_from_curve,
            make_curve_filament_spec,
            make_frame_rotation_spec,
            make_zero_rotation_spec,
        )

        if isinstance(self.rotation, ZeroRotation):
            rotation_spec = make_zero_rotation_spec(quadpoints=self.curve.quadpoints)
        else:
            rotation_spec = make_frame_rotation_spec(
                dofs=self.rotation.full_x,
                quadpoints=self.curve.quadpoints,
                order=self.rotation.order,
                scale=self.rotation.scale,
            )
        frame_kind = (
            "frenet" if isinstance(self.framedcurve, FramedCurveFrenet) else "centroid"
        )
        return make_curve_filament_spec(
            dofs=self.full_x,
            quadpoints=self.quadpoints,
            base_curve=curve_spec_from_curve(self.curve),
            base_curve_map=_optimizable_dof_map_spec(self, self.curve),
            rotation=rotation_spec,
            rotation_map=_optimizable_dof_map_spec(self, self.rotation),
            frame_kind=frame_kind,
            dn=self.dn,
            db=self.db,
        )


def create_multifilament_grid(
    curve,
    numfilaments_n,
    numfilaments_b,
    gapsize_n,
    gapsize_b,
    rotation_order=None,
    rotation_scaling=None,
    frame="centroid",
):
    """
    Create a regular grid of ``numfilaments_n * numfilaments_b`` many
    filaments to approximate a finite-build coil.

    Note that "normal" and "binormal" in the function arguments here
    refer to either the Frenet frame or the "coil centroid
    frame" defined by Singh et al., before rotation.

    Args:
        curve: The underlying curve.
        numfilaments_n: number of filaments in normal direction.
        numfilaments_b: number of filaments in bi-normal direction.
        gapsize_n: gap between filaments in normal direction.
        gapsize_b: gap between filaments in bi-normal direction.
        rotation_order: Fourier order (maximum mode number) to use in the expression for the rotation
                        of the filament pack. ``None`` means that the rotation is not optimized.
        rotation_scaling: scaling for the rotation degrees of freedom. good
                           scaling improves the convergence of first order optimization
                           algorithms. If ``None``, then the default of ``1 / max(gapsize_n, gapsize_b)``
                           is used.
        frame: orthonormal frame to define normal and binormal before rotation (either 'centroid' or 'frenet')
    """
    assert frame in ["centroid", "frenet"]
    if numfilaments_n % 2 == 1:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n // 2
    else:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n / 2 + 0.5
    shifts_n = shifts_n * gapsize_n
    if numfilaments_b % 2 == 1:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b // 2
    else:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b / 2 + 0.5
    shifts_b = shifts_b * gapsize_b

    if rotation_scaling is None:
        rotation_scaling = 1 / max(gapsize_n, gapsize_b)
    if rotation_order is None:
        rotation = ZeroRotation(curve.quadpoints)
    else:
        rotation = FrameRotation(
            curve.quadpoints, rotation_order, scale=rotation_scaling
        )
    if frame == "frenet":
        framedcurve = FramedCurveFrenet(curve, rotation)
    else:
        framedcurve = FramedCurveCentroid(curve, rotation)

    filaments = []
    for i in range(numfilaments_n):
        for j in range(numfilaments_b):
            filaments.append(CurveFilament(framedcurve, shifts_n[i], shifts_b[j]))
    return filaments
