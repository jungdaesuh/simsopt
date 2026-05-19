import numpy as np
import jax.numpy as jnp
from jax import vjp

from ._simsoptpp import sopp_namespace

sopp = sopp_namespace("Curve")
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative
from ..jax_core.framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotated_frenet_frame,
    rotated_frenet_frame_dash,
    rotation_alpha as jaxrotation_pure,
    rotation_alphadash as jaxrotationdash_pure,
    rotation_dcoeff,
    rotationdash_dcoeff,
)
from .curve import Curve, _as_runtime_jax_float64
from .jit import jit

__all__ = [
    "FramedCurve",
    "FramedCurveFrenet",
    "FramedCurveCentroid",
    "FrameRotation",
    "ZeroRotation",
    "FramedCurve",
]


_as_jax_float64_array = _as_runtime_jax_float64


def _jax_float64_tuple(*values):
    return tuple(_as_jax_float64_array(value) for value in values)


_as_jax_float64_arrays = _jax_float64_tuple


class FramedCurve(sopp.Curve, Curve):
    def __init__(self, curve, rotation=None):
        """
        A FramedCurve defines an orthonormal basis around a Curve,
        where one basis is taken to be the tangent along the Curve.
        The frame is defined with respect to a reference frame,
        either centroid or frenet. A rotation angle defines the rotation
        with respect to this reference frame.
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        deps = [curve]
        if rotation is not None:
            deps.append(rotation)
        if rotation is None:
            rotation = ZeroRotation(curve.quadpoints)
        self.rotation = rotation
        Curve.__init__(self, depends_on=deps)

    def _frame_twist_inputs(self):
        gammadash = _as_jax_float64_array(self.curve.gammadash())
        t, n, _ = self.rotated_frame()
        _, ndash, _ = self.rotated_frame_dash()
        return gammadash, t, n, ndash

    def frame_twist(self):
        """
        Evaluates the total twist (https://en.wikipedia.org/wiki/Twist_(mathematics))
        of the given frame, quantifying the winding of the normal about the given
        curve.
        """
        gammadash, t, n, ndash = self._frame_twist_inputs()
        return _frame_twist_eval(gammadash, t, n, ndash)

    def dframe_twist_by_dcoeff_vjp(self, v):
        gammadash, t, n, ndash = self._frame_twist_inputs()
        grad0, grad1, grad2 = _frame_twist_vjps(
            gammadash,
            t,
            n,
            ndash,
            _as_jax_float64_array(v),
        )
        zeros = jnp.zeros_like(grad0)
        return self.rotated_frame_dcoeff_vjp(
            grad0, grad1, zeros
        ) + self.rotated_frame_dash_dcoeff_vjp(zeros, grad2, zeros)


class FramedCurveFrenet(FramedCurve):
    r"""
    Given a curve, one defines a reference frame using the Frenet normal and
    binormal vectors:

    tangent = dr/dl

    normal = (dtangent/dl)/||dtangent/dl||

    binormal = tangent x normal

    In addition, we specify an angle along the curve that
    defines the rotation with respect to this reference frame. This defines the
    frame :math:`(\hat{\textbf{t}}, \hat{\textbf{n}}, \hat{\textbf{b}})`.
    """

    def __init__(self, curve, rotation=None):
        FramedCurve.__init__(self, curve, rotation)

    def _frame_scalar_inputs(self):
        gamma = _as_jax_float64_array(self.curve.gamma())
        d1gamma = _as_jax_float64_array(self.curve.gammadash())
        d2gamma = _as_jax_float64_array(self.curve.gammadashdash())
        d3gamma = _as_jax_float64_array(self.curve.gammadashdashdash())
        alpha = _as_jax_float64_array(self.rotation.alpha(self.curve.quadpoints))
        alphadash = _as_jax_float64_array(
            self.rotation.alphadash(self.curve.quadpoints)
        )
        return gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash

    def rotated_frame(self):
        r"""
        Returns the frame :math:`(\hat{\textbf{t}}, \hat{\textbf{n}}, \hat{\textbf{b}})`, which is rotated
        with respect to the reference Frenet frame by the rotation.
        """
        return rotated_frenet_frame(
            _as_jax_float64_array(self.curve.gamma()),
            _as_jax_float64_array(self.curve.gammadash()),
            _as_jax_float64_array(self.curve.gammadashdash()),
            _as_jax_float64_array(self.rotation.alpha(self.curve.quadpoints)),
        )

    def rotated_frame_dash(self):
        r"""
        Returns the derivative of the frame with respect to the parameterization of the curve,
        :math:`(\hat{\textbf{t}}'(\phi), \hat{\textbf{n}}'(\phi), \hat{\textbf{b}})'(\phi)`.
        The frame is obtained by rotating with respect to the reference Frenet frame by the rotation.
        """
        return rotated_frenet_frame_dash(
            _as_jax_float64_array(self.curve.gamma()),
            _as_jax_float64_array(self.curve.gammadash()),
            _as_jax_float64_array(self.curve.gammadashdash()),
            _as_jax_float64_array(self.curve.gammadashdashdash()),
            _as_jax_float64_array(self.rotation.alpha(self.curve.quadpoints)),
            _as_jax_float64_array(self.rotation.alphadash(self.curve.quadpoints)),
        )

    def frame_torsion(self):
        r"""
        Returns the frame torsion, :math:`\hat{\textbf{n}}'(l) \cdot \hat{\textbf{b}}`,
        along the curve.
        """
        gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash = self._frame_scalar_inputs()
        return _frenet_torsion_eval(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash)

    def frame_binormal_curvature(self):
        r"""
        Returns the frame binormal curvature, :math:`\hat{\textbf{t}}'(l) \cdot \hat{\textbf{b}}`,
        where prime indicates derivative with respect to curve length.
        """
        gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash = self._frame_scalar_inputs()
        return _frenet_binorm_eval(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash)

    def dframe_torsion_by_dcoeff_vjp(self, v):
        """
        VJP function for derivatives of the frame torsion with respect to the
        curve and rotation dofs.
        """
        gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash = self._frame_scalar_inputs()
        grad0, grad1, grad2, grad3, grad4, grad5 = _frenet_torsion_vjps(
            gamma,
            d1gamma,
            d2gamma,
            d3gamma,
            alpha,
            alphadash,
            _as_jax_float64_array(v),
        )

        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(grad3)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def dframe_binormal_curvature_by_dcoeff_vjp(self, v):
        """
        VJP function for the derivatives of the binormal curvature with respect to the curve
        and rotation dofs.
        """
        gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash = self._frame_scalar_inputs()
        grad0, grad1, grad2, grad3, grad4, grad5 = _frenet_binorm_vjps(
            gamma,
            d1gamma,
            d2gamma,
            d3gamma,
            alpha,
            alphadash,
            _as_jax_float64_array(v),
        )

        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(grad3)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def rotated_frame_dcoeff_vjp(self, v0, v1, v2):
        r"""
        VJP function for the derivatives of the frame
        :math:`(\hat{\textbf{t}}, \hat{\textbf{n}}, \hat{\textbf{b}})`,
        with respect to the curve and rotation dofs.
        """
        g, gd, gdd, _, a, _ = self._frame_scalar_inputs()
        v0, v1, v2 = _as_jax_float64_arrays(v0, v1, v2)

        vjp0 = rotated_frenet_frame_dcoeff_vjp0(g, gd, gdd, a, (v0, v1, v2))
        vjp1 = rotated_frenet_frame_dcoeff_vjp1(g, gd, gdd, a, (v0, v1, v2))
        vjp2 = rotated_frenet_frame_dcoeff_vjp2(g, gd, gdd, a, (v0, v1, v2))
        vjp3 = rotated_frenet_frame_dcoeff_vjp3(g, gd, gdd, a, (v0, v1, v2))

        return (
            self.curve.dgamma_by_dcoeff_vjp(vjp0)
            + self.curve.dgammadash_by_dcoeff_vjp(vjp1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(vjp2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, vjp3)
        )

    def rotated_frame_dash_dcoeff_vjp(self, v0, v1, v2):
        r"""
        VJP function for the derivatives of the frame parameter derivatives,
        :math:`(\hat{\textbf{t}}'(\phi), \hat{\textbf{n}}'(\phi), \hat{\textbf{b}}'(\phi))`,
        with respect to the curve and rotation dofs.
        """
        g, gd, gdd, gddd, a, ad = self._frame_scalar_inputs()
        v0, v1, v2 = _as_jax_float64_arrays(v0, v1, v2)
        vjp0 = rotated_frenet_frame_dash_dcoeff_vjp0(
            g, gd, gdd, gddd, a, ad, (v0, v1, v2)
        )
        vjp1 = rotated_frenet_frame_dash_dcoeff_vjp1(
            g, gd, gdd, gddd, a, ad, (v0, v1, v2)
        )
        vjp2 = rotated_frenet_frame_dash_dcoeff_vjp2(
            g, gd, gdd, gddd, a, ad, (v0, v1, v2)
        )
        vjp3 = rotated_frenet_frame_dash_dcoeff_vjp3(
            g, gd, gdd, gddd, a, ad, (v0, v1, v2)
        )
        vjp4 = rotated_frenet_frame_dash_dcoeff_vjp4(
            g, gd, gdd, gddd, a, ad, (v0, v1, v2)
        )
        vjp5 = rotated_frenet_frame_dash_dcoeff_vjp5(
            g, gd, gdd, gddd, a, ad, (v0, v1, v2)
        )

        return (
            self.curve.dgamma_by_dcoeff_vjp(vjp0)
            + self.curve.dgammadash_by_dcoeff_vjp(vjp1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(vjp2)
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(vjp3)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, vjp4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, vjp5)
        )


class FramedCurveCentroid(FramedCurve):
    """
    Implementation of the centroid frame introduced in
    Singh et al, "Optimization of finite-build stellarator coils",
    Journal of Plasma Physics 86 (2020),
    doi:10.1017/S0022377820000756.
    Given a curve, one defines a reference frame using the normal and
    binormal vector based on the centoid of the coil. In addition, we specify an
    angle along the curve that defines the rotation with respect to this
    reference frame.

    The idea is explained well in Figure 1 in the reference above.
    """

    def __init__(self, curve, rotation=None):
        FramedCurve.__init__(self, curve, rotation)

    def _frame_scalar_inputs(self):
        gamma = _as_jax_float64_array(self.curve.gamma())
        d1gamma = _as_jax_float64_array(self.curve.gammadash())
        d2gamma = _as_jax_float64_array(self.curve.gammadashdash())
        alpha = _as_jax_float64_array(self.rotation.alpha(self.curve.quadpoints))
        alphadash = _as_jax_float64_array(
            self.rotation.alphadash(self.curve.quadpoints)
        )
        return gamma, d1gamma, d2gamma, alpha, alphadash

    def frame_torsion(self):
        r"""
        Returns the frame torsion, :math:`\hat{\textbf{n}}'(l) \cdot \hat{\textbf{b}}`,
        along the curve.
        """
        gamma, d1gamma, d2gamma, alpha, alphadash = self._frame_scalar_inputs()
        return _centroid_torsion_eval(gamma, d1gamma, d2gamma, alpha, alphadash)

    def dframe_torsion_by_dcoeff_vjp(self, v):
        """
        VJP function for derivatives of the frame torsion with respect to the
        curve and rotation dofs.
        """
        gamma, d1gamma, d2gamma, alpha, alphadash = self._frame_scalar_inputs()
        grad0, grad1, grad2, grad4, grad5 = _centroid_torsion_vjps(
            gamma,
            d1gamma,
            d2gamma,
            alpha,
            alphadash,
            _as_jax_float64_array(v),
        )

        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def frame_binormal_curvature(self):
        r"""
        Returns the frame binormal curvature, :math:`\hat{\textbf{t}}'(l) \cdot \hat{\textbf{b}}`,
        where prime indicates derivative with respect to curve length.
        """
        gamma, d1gamma, d2gamma, alpha, alphadash = self._frame_scalar_inputs()
        return _centroid_binorm_eval(gamma, d1gamma, d2gamma, alpha, alphadash)

    def rotated_frame(self):
        r"""
        Returns the frame :math:`(\hat{\textbf{t}}, \hat{\textbf{n}}, \hat{\textbf{b}})`, which is rotated
        with respect to the reference centroid frame by the rotation.
        """
        return rotated_centroid_frame(
            _as_jax_float64_array(self.curve.gamma()),
            _as_jax_float64_array(self.curve.gammadash()),
            _as_jax_float64_array(self.rotation.alpha(self.curve.quadpoints)),
        )

    def rotated_frame_dash(self):
        r"""
        Returns the derivative of the frame with respect to the parameterization of the curve,
        :math:`(\hat{\textbf{t}}'(\phi), \hat{\textbf{n}}'(\phi), \hat{\textbf{b}})'(\phi)`.
        The frame is obtained by rotating with respect to the reference centroid frame by the rotation.
        """
        return rotated_centroid_frame_dash(
            _as_jax_float64_array(self.curve.gamma()),
            _as_jax_float64_array(self.curve.gammadash()),
            _as_jax_float64_array(self.curve.gammadashdash()),
            _as_jax_float64_array(self.rotation.alpha(self.curve.quadpoints)),
            _as_jax_float64_array(self.rotation.alphadash(self.curve.quadpoints)),
        )

    def dframe_binormal_curvature_by_dcoeff_vjp(self, v):
        """
        VJP function for the derivatives of the binormal curvature with respect to the curve
        and rotation dofs.
        """
        gamma, d1gamma, d2gamma, alpha, alphadash = self._frame_scalar_inputs()
        grad0, grad1, grad2, grad4, grad5 = _centroid_binorm_vjps(
            gamma,
            d1gamma,
            d2gamma,
            alpha,
            alphadash,
            _as_jax_float64_array(v),
        )

        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def rotated_frame_dcoeff_vjp(self, v0, v1, v2):
        r"""
        VJP function for the derivatives of the frame
        :math:`(\hat{\textbf{t}}, \hat{\textbf{n}}, \hat{\textbf{b}})`,
        with respect to the curve and rotation dofs.
        """
        g, gd, _, a, _ = self._frame_scalar_inputs()
        v0, v1, v2 = _as_jax_float64_arrays(v0, v1, v2)

        vjp0 = rotated_centroid_frame_dcoeff_vjp0(g, gd, a, (v0, v1, v2))
        vjp1 = rotated_centroid_frame_dcoeff_vjp1(g, gd, a, (v0, v1, v2))
        vjp2 = rotated_centroid_frame_dcoeff_vjp3(g, gd, a, (v0, v1, v2))

        return (
            self.curve.dgamma_by_dcoeff_vjp(vjp0)
            + self.curve.dgammadash_by_dcoeff_vjp(vjp1)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, vjp2)
        )

    def rotated_frame_dash_dcoeff_vjp(self, v0, v1, v2):
        r"""
        VJP function for the derivatives of the frame parameter derivatives,
        :math:`(\hat{\textbf{t}}'(\phi), \hat{\textbf{n}}'(\phi), \hat{\textbf{b}}'(\phi))`,
        with respect to the curve and rotation dofs.
        """
        g, gd, gdd, a, ad = self._frame_scalar_inputs()
        v0, v1, v2 = _as_jax_float64_arrays(v0, v1, v2)
        vjp0 = rotated_centroid_frame_dash_dcoeff_vjp0(g, gd, gdd, a, ad, (v0, v1, v2))
        vjp1 = rotated_centroid_frame_dash_dcoeff_vjp1(g, gd, gdd, a, ad, (v0, v1, v2))
        vjp2 = rotated_centroid_frame_dash_dcoeff_vjp2(g, gd, gdd, a, ad, (v0, v1, v2))
        vjp4 = rotated_centroid_frame_dash_dcoeff_vjp4(g, gd, gdd, a, ad, (v0, v1, v2))
        vjp5 = rotated_centroid_frame_dash_dcoeff_vjp5(g, gd, gdd, a, ad, (v0, v1, v2))

        return (
            self.curve.dgamma_by_dcoeff_vjp(vjp0)
            + self.curve.dgammadash_by_dcoeff_vjp(vjp1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(vjp2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, vjp4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, vjp5)
        )


class FrameRotation(Optimizable):
    def __init__(self, quadpoints, order, scale=1.0, dofs=None):
        """
        Defines the rotation angle with respect to a reference orthonormal
        frame (either frenet or centroid). For example, can be used to
        define the rotation of a multifilament pack; alpha in Figure 1 of
        Singh et al, "Optimization of finite-build stellarator coils",
        Journal of Plasma Physics 86 (2020),
        doi:10.1017/S0022377820000756
        """
        self.order = order
        if dofs is None:
            super().__init__(x0=np.zeros((2 * order + 1,)))
        else:
            super().__init__(dofs=dofs)
        self.quadpoints = quadpoints
        self.scale = scale
        self.jac = rotation_dcoeff(quadpoints, order)
        self.jacdash = rotationdash_dcoeff(quadpoints, order)

    def jax_alpha(self, dofs, points):
        return _rotation_eval(dofs, points, self.order)

    def jax_alphadash(self, dofs, points):
        return _rotationdash_eval(dofs, points, self.order)

    def _rotation_value(self, evaluator, quadpoints):
        scale = _as_jax_float64_array(self.scale)
        return scale * evaluator(
            _as_jax_float64_array(self._dofs.full_x),
            _as_jax_float64_array(quadpoints),
        )

    def alpha(self, quadpoints):
        return self._rotation_value(self.jax_alpha, quadpoints)

    def alphadash(self, quadpoints):
        return self._rotation_value(self.jax_alphadash, quadpoints)

    def dalpha_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({self: self.scale * sopp.vjp(v, self.jac)})

    def dalphadash_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({self: self.scale * sopp.vjp(v, self.jacdash)})


class ZeroRotation(Optimizable):
    def __init__(self, quadpoints):
        """
        Dummy class that just returns zero for the rotation angle. Equivalent to using

        .. code-block:: python

            rot = FrameRotation(...)
            rot.fix_all()

        """
        super().__init__()
        self.zero = np.zeros((quadpoints.size,))

    def alpha(self, quadpoints):
        return self.zero

    def alphadash(self, quadpoints):
        return self.zero

    def dalpha_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({})

    def dalphadash_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({})


rotated_centroid_frame_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda g: rotated_centroid_frame(g, gammadash, alpha), gamma
    )[1](v)[0]
)

rotated_centroid_frame_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda gd: rotated_centroid_frame(gamma, gd, alpha), gammadash
    )[1](v)[0]
)

rotated_centroid_frame_dcoeff_vjp3 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda a: rotated_centroid_frame(gamma, gammadash, a), alpha
    )[1](v)[0]
)

rotated_centroid_frame_dash_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda g: rotated_centroid_frame_dash(
            g, gammadash, gammadashdash, alpha, alphadash
        ),
        gamma,
    )[1](v)[0]
)

rotated_centroid_frame_dash_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda gd: rotated_centroid_frame_dash(
            gamma, gd, gammadashdash, alpha, alphadash
        ),
        gammadash,
    )[1](v)[0]
)

rotated_centroid_frame_dash_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda gdd: rotated_centroid_frame_dash(
            gamma, gammadash, gdd, alpha, alphadash
        ),
        gammadashdash,
    )[1](v)[0]
)

rotated_centroid_frame_dash_dcoeff_vjp4 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda a: rotated_centroid_frame_dash(
            gamma, gammadash, gammadashdash, a, alphadash
        ),
        alpha,
    )[1](v)[0]
)

rotated_centroid_frame_dash_dcoeff_vjp5 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda ad: rotated_centroid_frame_dash(
            gamma, gammadash, gammadashdash, alpha, ad
        ),
        alphadash,
    )[1](v)[0]
)


rotated_frenet_frame_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda g: rotated_frenet_frame(g, gammadash, gammadashdash, alpha), gamma
    )[1](v)[0]
)

rotated_frenet_frame_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda gd: rotated_frenet_frame(gamma, gd, gammadashdash, alpha), gammadash
    )[1](v)[0]
)

rotated_frenet_frame_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda gdd: rotated_frenet_frame(gamma, gammadash, gdd, alpha), gammadashdash
    )[1](v)[0]
)

rotated_frenet_frame_dcoeff_vjp3 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda a: rotated_frenet_frame(gamma, gammadash, gammadashdash, a), alpha
    )[1](v)[0]
)

rotated_frenet_frame_dash_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda g: rotated_frenet_frame_dash(
            g, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
        ),
        gamma,
    )[1](v)[0]
)

rotated_frenet_frame_dash_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda gd: rotated_frenet_frame_dash(
            gamma, gd, gammadashdash, gammadashdashdash, alpha, alphadash
        ),
        gammadash,
    )[1](v)[0]
)

rotated_frenet_frame_dash_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda gdd: rotated_frenet_frame_dash(
            gamma, gammadash, gdd, gammadashdashdash, alpha, alphadash
        ),
        gammadashdash,
    )[1](v)[0]
)

rotated_frenet_frame_dash_dcoeff_vjp3 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda gddd: rotated_frenet_frame_dash(
            gamma, gammadash, gammadashdash, gddd, alpha, alphadash
        ),
        gammadashdashdash,
    )[1](v)[0]
)

rotated_frenet_frame_dash_dcoeff_vjp4 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda a: rotated_frenet_frame_dash(
            gamma, gammadash, gammadashdash, gammadashdashdash, a, alphadash
        ),
        alpha,
    )[1](v)[0]
)

rotated_frenet_frame_dash_dcoeff_vjp5 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda ad: rotated_frenet_frame_dash(
            gamma, gammadash, gammadashdash, gammadashdashdash, alpha, ad
        ),
        alphadash,
    )[1](v)[0]
)


@jit
def _frame_twist_eval(gammadash, t, n, ndash):
    return frame_twist_pure(gammadash, t, n, ndash)


@jit
def _frame_twist_vjps(gammadash, t, n, ndash, v):
    return (
        vjp(lambda g: _frame_twist_eval(g, t, n, ndash), gammadash)[1](v)[0],
        vjp(lambda g: _frame_twist_eval(gammadash, t, g, ndash), n)[1](v)[0],
        vjp(lambda g: _frame_twist_eval(gammadash, t, n, g), ndash)[1](v)[0],
    )


@jit
def _frenet_torsion_eval(
    gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
):
    return torsion_pure_frenet(
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
    )


@jit
def _frenet_torsion_vjps(
    gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v
):
    return (
        vjp(
            lambda g: _frenet_torsion_eval(
                g, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
            ),
            gamma,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_torsion_eval(
                gamma, g, gammadashdash, gammadashdashdash, alpha, alphadash
            ),
            gammadash,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_torsion_eval(
                gamma, gammadash, g, gammadashdashdash, alpha, alphadash
            ),
            gammadashdash,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_torsion_eval(
                gamma, gammadash, gammadashdash, g, alpha, alphadash
            ),
            gammadashdashdash,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_torsion_eval(
                gamma, gammadash, gammadashdash, gammadashdashdash, g, alphadash
            ),
            alpha,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_torsion_eval(
                gamma, gammadash, gammadashdash, gammadashdashdash, alpha, g
            ),
            alphadash,
        )[1](v)[0],
    )


@jit
def _frenet_binorm_eval(
    gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
):
    return binormal_curvature_pure_frenet(
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
    )


@jit
def _frenet_binorm_vjps(
    gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v
):
    return (
        vjp(
            lambda g: _frenet_binorm_eval(
                g, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
            ),
            gamma,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_binorm_eval(
                gamma, g, gammadashdash, gammadashdashdash, alpha, alphadash
            ),
            gammadash,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_binorm_eval(
                gamma, gammadash, g, gammadashdashdash, alpha, alphadash
            ),
            gammadashdash,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_binorm_eval(
                gamma, gammadash, gammadashdash, g, alpha, alphadash
            ),
            gammadashdashdash,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_binorm_eval(
                gamma, gammadash, gammadashdash, gammadashdashdash, g, alphadash
            ),
            alpha,
        )[1](v)[0],
        vjp(
            lambda g: _frenet_binorm_eval(
                gamma, gammadash, gammadashdash, gammadashdashdash, alpha, g
            ),
            alphadash,
        )[1](v)[0],
    )


@jit
def _centroid_torsion_eval(gamma, gammadash, gammadashdash, alpha, alphadash):
    return torsion_pure_centroid(gamma, gammadash, gammadashdash, alpha, alphadash)


@jit
def _centroid_torsion_vjps(gamma, gammadash, gammadashdash, alpha, alphadash, v):
    return (
        vjp(
            lambda g: _centroid_torsion_eval(
                g, gammadash, gammadashdash, alpha, alphadash
            ),
            gamma,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_torsion_eval(gamma, g, gammadashdash, alpha, alphadash),
            gammadash,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_torsion_eval(gamma, gammadash, g, alpha, alphadash),
            gammadashdash,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_torsion_eval(
                gamma, gammadash, gammadashdash, g, alphadash
            ),
            alpha,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_torsion_eval(gamma, gammadash, gammadashdash, alpha, g),
            alphadash,
        )[1](v)[0],
    )


@jit
def _centroid_binorm_eval(gamma, gammadash, gammadashdash, alpha, alphadash):
    return binormal_curvature_pure_centroid(
        gamma, gammadash, gammadashdash, alpha, alphadash
    )


@jit
def _centroid_binorm_vjps(gamma, gammadash, gammadashdash, alpha, alphadash, v):
    return (
        vjp(
            lambda g: _centroid_binorm_eval(
                g, gammadash, gammadashdash, alpha, alphadash
            ),
            gamma,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_binorm_eval(gamma, g, gammadashdash, alpha, alphadash),
            gammadash,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_binorm_eval(gamma, gammadash, g, alpha, alphadash),
            gammadashdash,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_binorm_eval(
                gamma, gammadash, gammadashdash, g, alphadash
            ),
            alpha,
        )[1](v)[0],
        vjp(
            lambda g: _centroid_binorm_eval(gamma, gammadash, gammadashdash, alpha, g),
            alphadash,
        )[1](v)[0],
    )


_rotation_eval = jaxrotation_pure
_rotationdash_eval = jaxrotationdash_pure


def inner(a, b):
    """Inner product for arrays of shape (N, 3)"""
    return jnp.sum(a * b, axis=1)


def torsion_pure_frenet(
    gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
):
    _, _, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    _, ndash, _ = rotated_frenet_frame_dash(
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
    )

    ndash = ndash / jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(ndash, b)


def binormal_curvature_pure_frenet(
    gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
):
    _, _, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    tdash, _, _ = rotated_frenet_frame_dash(
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash
    )

    tdash = tdash / jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(tdash, b)


def torsion_pure_centroid(gamma, gammadash, gammadashdash, alpha, alphadash):
    _, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
    _, ndash, _ = rotated_centroid_frame_dash(
        gamma, gammadash, gammadashdash, alpha, alphadash
    )

    ndash = ndash / jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(ndash, b)


def binormal_curvature_pure_centroid(gamma, gammadash, gammadashdash, alpha, alphadash):
    _, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
    tdash, _, _ = rotated_centroid_frame_dash(
        gamma, gammadash, gammadashdash, alpha, alphadash
    )

    tdash = tdash / jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(tdash, b)


def frame_twist_pure(gammadash, t, n, ndash):
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    T = (
        n[:, 0] * (ndash[:, 1] * t[:, 2] - ndash[:, 2] * t[:, 1])
        + n[:, 1] * (ndash[:, 2] * t[:, 0] - ndash[:, 0] * t[:, 2])
        + n[:, 2] * (ndash[:, 0] * t[:, 1] - ndash[:, 1] * t[:, 0])
    )
    return T / (2 * jnp.pi * arc_length)
