"""JAX-backed Optimizable wrappers for framed curves and rotations.

Wave R4 item 18: provide ``Optimizable`` adapters that mirror the public
API of :class:`simsopt.geo.framedcurve.FramedCurveFrenet`,
:class:`FramedCurveCentroid`, :class:`FrameRotation`, and
:class:`ZeroRotation`, while routing hot paths through the JAX kernels
in :mod:`simsopt.jax_core.framedcurve`.

These wrappers follow the same adapter pattern used by
``BiotSavartJAX``: the CPU-derived parent curve participates in the
``Optimizable`` dependency graph for DOF orchestration, while frame
evaluation, derivative, and VJP work is delegated to the immutable
JAX kernels. The wrappers do **not** subclass ``sopp.Curve`` — they
sit alongside the C++/CPU framed-curve classes as a parallel JAX
implementation.

Conventions
-----------
* ``FrameRotationJAX`` / ``ZeroRotationJAX`` carry the same DOFs as their
  upstream counterparts so a JAX-backed framed curve can be swapped into
  an existing pipeline that expects the ``FrameRotation`` interface.
* ``FramedCurveFrenetJAX`` / ``FramedCurveCentroidJAX`` mirror the public
  ``rotated_frame``, ``rotated_frame_dash``, ``frame_torsion``, and
  ``frame_binormal_curvature`` methods.
* All host-facing arrays are converted through
  :func:`simsopt.jax_core._math_utils.as_jax_float64`; outputs of the
  JAX kernels stay as ``jax.Array`` so downstream JAX consumers do not
  trigger implicit host transfers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._core.derivative import Derivative
from .._core.optimizable import Optimizable
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotated_frenet_frame,
    rotated_frenet_frame_dash,
    rotation_alpha,
    rotation_alphadash,
    rotation_dcoeff,
    rotationdash_dcoeff,
)


__all__ = [
    "FrameRotationJAX",
    "ZeroRotationJAX",
    "FramedCurveCentroidJAX",
    "FramedCurveFrenetJAX",
]


def _inner(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.sum(a * b, axis=1)


def _frame_twist(
    gammadash: jax.Array,
    t: jax.Array,
    n: jax.Array,
    ndash: jax.Array,
) -> jax.Array:
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    triple_product = _inner(n, jnp.cross(ndash, t))
    return triple_product / (2.0 * jnp.pi * arc_length)


def _frame_twist_vjps(
    gammadash: jax.Array,
    t: jax.Array,
    n: jax.Array,
    ndash: jax.Array,
    v: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    cotangent = _as_jax_float64(v)
    _, pullback = jax.vjp(_frame_twist, gammadash, t, n, ndash)
    return pullback(cotangent)


def _cotangent3(v0: object, v1: object, v2: object) -> tuple[jax.Array, ...]:
    return (
        _as_jax_float64(v0),
        _as_jax_float64(v1),
        _as_jax_float64(v2),
    )


def _torsion_centroid(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
) -> jax.Array:
    _t, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
    _td, ndash, _bd = rotated_centroid_frame_dash(
        gamma, gammadash, gammadashdash, alpha, alphadash
    )
    arc_length = jnp.linalg.norm(gammadash, axis=1)[:, None]
    return _inner(ndash / arc_length, b)


def _binormal_curvature_centroid(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
) -> jax.Array:
    _t, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
    tdash, _nd, _bd = rotated_centroid_frame_dash(
        gamma, gammadash, gammadashdash, alpha, alphadash
    )
    arc_length = jnp.linalg.norm(gammadash, axis=1)[:, None]
    return _inner(tdash / arc_length, b)


def _torsion_frenet(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    gammadashdashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
) -> jax.Array:
    _t, _, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    _td, ndash, _bd = rotated_frenet_frame_dash(
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        alpha,
        alphadash,
    )
    arc_length = jnp.linalg.norm(gammadash, axis=1)[:, None]
    return _inner(ndash / arc_length, b)


def _binormal_curvature_frenet(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    gammadashdashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
) -> jax.Array:
    _t, _, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    tdash, _nd, _bd = rotated_frenet_frame_dash(
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        alpha,
        alphadash,
    )
    arc_length = jnp.linalg.norm(gammadash, axis=1)[:, None]
    return _inner(tdash / arc_length, b)


def _centroid_torsion_vjps(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
    v: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    cotangent = _as_jax_float64(v)
    _, pullback = jax.vjp(
        _torsion_centroid,
        gamma,
        gammadash,
        gammadashdash,
        alpha,
        alphadash,
    )
    return pullback(cotangent)


def _centroid_binormal_curvature_vjps(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
    v: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    cotangent = _as_jax_float64(v)
    _, pullback = jax.vjp(
        _binormal_curvature_centroid,
        gamma,
        gammadash,
        gammadashdash,
        alpha,
        alphadash,
    )
    return pullback(cotangent)


def _frenet_torsion_vjps(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    gammadashdashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
    v: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    cotangent = _as_jax_float64(v)
    _, pullback = jax.vjp(
        _torsion_frenet,
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        alpha,
        alphadash,
    )
    return pullback(cotangent)


def _frenet_binormal_curvature_vjps(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    gammadashdashdash: jax.Array,
    alpha: jax.Array,
    alphadash: jax.Array,
    v: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    cotangent = _as_jax_float64(v)
    _, pullback = jax.vjp(
        _binormal_curvature_frenet,
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        alpha,
        alphadash,
    )
    return pullback(cotangent)


class FrameRotationJAX(Optimizable):
    """Optimizable Fourier rotation backed by JAX kernels.

    Public API mirrors :class:`simsopt.geo.framedcurve.FrameRotation`:
    ``alpha(quadpoints)``, ``alphadash(quadpoints)``,
    ``dalpha_by_dcoeff_vjp``, and ``dalphadash_by_dcoeff_vjp``.
    The host-side Jacobian helpers ``rotation_dcoeff`` /
    ``rotationdash_dcoeff`` are reused so the VJP path stays compatible
    with the C++ ``sopp.vjp`` accumulator.
    """

    def __init__(
        self,
        quadpoints: object,
        order: int,
        scale: float = 1.0,
        dofs: object | None = None,
    ) -> None:
        self.order = int(order)
        self.scale = float(scale)
        if dofs is None:
            super().__init__(x0=np.zeros((2 * self.order + 1,)))
        else:
            super().__init__(dofs=dofs)
        self.quadpoints = quadpoints
        self.jac = rotation_dcoeff(quadpoints, self.order)
        self.jacdash = rotationdash_dcoeff(quadpoints, self.order)

    def _rotation_value(
        self,
        evaluator,
        quadpoints: object,
    ) -> jax.Array:
        return _as_jax_float64(self.scale) * evaluator(
            _as_jax_float64(self._dofs.full_x),
            _as_jax_float64(quadpoints),
            self.order,
        )

    def jax_alpha(self, dofs: object, points: object) -> jax.Array:
        return rotation_alpha(
            _as_jax_float64(dofs),
            _as_jax_float64(points),
            self.order,
        )

    def jax_alphadash(self, dofs: object, points: object) -> jax.Array:
        return rotation_alphadash(
            _as_jax_float64(dofs),
            _as_jax_float64(points),
            self.order,
        )

    def alpha(self, quadpoints: object) -> jax.Array:
        return self._rotation_value(rotation_alpha, quadpoints)

    def alphadash(self, quadpoints: object) -> jax.Array:
        return self._rotation_value(rotation_alphadash, quadpoints)

    def dalpha_by_dcoeff_vjp(self, quadpoints: object, v: object) -> Derivative:
        del quadpoints  # Jacobian is precomputed for ``self.quadpoints``.
        gradient = (
            self.scale
            * np.asarray(self.jac, dtype=np.float64).T
            @ np.asarray(v, dtype=np.float64)
        )
        return Derivative({self: gradient})

    def dalphadash_by_dcoeff_vjp(self, quadpoints: object, v: object) -> Derivative:
        del quadpoints
        gradient = (
            self.scale
            * np.asarray(self.jacdash, dtype=np.float64).T
            @ np.asarray(v, dtype=np.float64)
        )
        return Derivative({self: gradient})


class ZeroRotationJAX(Optimizable):
    """Optimizable zero-rotation stub matching :class:`ZeroRotation`."""

    def __init__(self, quadpoints: object) -> None:
        super().__init__()
        quad_array = np.asarray(quadpoints, dtype=np.float64)
        self.quadpoints = quad_array
        self._zero = jnp.zeros(int(quad_array.size), dtype=jnp.float64)

    def alpha(self, quadpoints: object) -> jax.Array:
        del quadpoints
        return self._zero

    def alphadash(self, quadpoints: object) -> jax.Array:
        del quadpoints
        return self._zero

    def dalpha_by_dcoeff_vjp(self, quadpoints: object, v: object) -> Derivative:
        del quadpoints, v
        return Derivative({})

    def dalphadash_by_dcoeff_vjp(self, quadpoints: object, v: object) -> Derivative:
        del quadpoints, v
        return Derivative({})


class _FramedCurveJAXBase(Optimizable):
    """Shared infrastructure for the JAX framed-curve wrappers."""

    def __init__(self, curve, rotation) -> None:
        self.curve = curve
        if rotation is None:
            rotation = ZeroRotationJAX(curve.quadpoints)
        self.rotation = rotation
        deps = [curve, rotation]
        Optimizable.__init__(self, x0=np.asarray([], dtype=np.float64), depends_on=deps)

    @property
    def quadpoints(self):
        return self.curve.quadpoints

    def _alpha(self) -> jax.Array:
        return _as_jax_float64(self.rotation.alpha(self.curve.quadpoints))

    def _alphadash(self) -> jax.Array:
        return _as_jax_float64(self.rotation.alphadash(self.curve.quadpoints))

    def _frame_twist_inputs(self):
        gammadash = _as_jax_float64(self.curve.gammadash())
        t, n, _b = self.rotated_frame()
        _tdash, ndash, _bdash = self.rotated_frame_dash()
        return gammadash, t, n, ndash

    def frame_twist(self) -> jax.Array:
        return _frame_twist(*self._frame_twist_inputs())

    def dframe_twist_by_dcoeff_vjp(self, v: object) -> Derivative:
        gammadash, t, n, ndash = self._frame_twist_inputs()
        grad0, grad1, grad2, grad3 = _frame_twist_vjps(gammadash, t, n, ndash, v)
        zeros = jnp.zeros_like(grad0)
        return (
            self.curve.dgammadash_by_dcoeff_vjp(grad0)
            + self.rotated_frame_dcoeff_vjp(grad1, grad2, zeros)
            + self.rotated_frame_dash_dcoeff_vjp(zeros, grad3, zeros)
        )


class FramedCurveFrenetJAX(_FramedCurveJAXBase):
    """JAX-backed Frenet frame wrapper.

    Mirrors the public surface of
    :class:`simsopt.geo.framedcurve.FramedCurveFrenet`:
    ``rotated_frame``, ``rotated_frame_dash``, ``frame_torsion``,
    ``frame_binormal_curvature``. The DOF graph is composed via the
    underlying ``curve`` and ``rotation`` Optimizables; this wrapper
    holds no DOFs of its own.
    """

    def _scalar_inputs(self):
        return (
            _as_jax_float64(self.curve.gamma()),
            _as_jax_float64(self.curve.gammadash()),
            _as_jax_float64(self.curve.gammadashdash()),
            _as_jax_float64(self.curve.gammadashdashdash()),
            self._alpha(),
            self._alphadash(),
        )

    def rotated_frame(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        gamma, gammadash, gammadashdash, _gddd, alpha, _ad = self._scalar_inputs()
        return rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)

    def rotated_frame_dash(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash = (
            self._scalar_inputs()
        )
        return rotated_frenet_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
        )

    def frame_torsion(self) -> jax.Array:
        return _torsion_frenet(*self._scalar_inputs())

    def frame_binormal_curvature(self) -> jax.Array:
        return _binormal_curvature_frenet(*self._scalar_inputs())

    def dframe_torsion_by_dcoeff_vjp(self, v: object) -> Derivative:
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash = (
            self._scalar_inputs()
        )
        grad0, grad1, grad2, grad3, grad4, grad5 = _frenet_torsion_vjps(
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
            v,
        )
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(grad3)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def dframe_binormal_curvature_by_dcoeff_vjp(self, v: object) -> Derivative:
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash = (
            self._scalar_inputs()
        )
        grad0, grad1, grad2, grad3, grad4, grad5 = _frenet_binormal_curvature_vjps(
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
            v,
        )
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(grad3)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def rotated_frame_dcoeff_vjp(
        self,
        v0: object,
        v1: object,
        v2: object,
    ) -> Derivative:
        gamma, gammadash, gammadashdash, _gammadashdashdash, alpha, _alphadash = (
            self._scalar_inputs()
        )
        grad0, grad1, grad2, grad3 = jax.vjp(
            rotated_frenet_frame,
            gamma,
            gammadash,
            gammadashdash,
            alpha,
        )[1](_cotangent3(v0, v1, v2))
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad3)
        )

    def rotated_frame_dash_dcoeff_vjp(
        self,
        v0: object,
        v1: object,
        v2: object,
    ) -> Derivative:
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash = (
            self._scalar_inputs()
        )
        grad0, grad1, grad2, grad3, grad4, grad5 = jax.vjp(
            rotated_frenet_frame_dash,
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
        )[1](_cotangent3(v0, v1, v2))
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(grad3)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )


class FramedCurveCentroidJAX(_FramedCurveJAXBase):
    """JAX-backed centroid frame wrapper.

    Mirrors the public surface of
    :class:`simsopt.geo.framedcurve.FramedCurveCentroid`:
    ``rotated_frame``, ``rotated_frame_dash``, ``frame_torsion``,
    ``frame_binormal_curvature``.
    """

    def _scalar_inputs(self):
        return (
            _as_jax_float64(self.curve.gamma()),
            _as_jax_float64(self.curve.gammadash()),
            _as_jax_float64(self.curve.gammadashdash()),
            self._alpha(),
            self._alphadash(),
        )

    def rotated_frame(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        gamma, gammadash, _gdd, alpha, _ad = self._scalar_inputs()
        return rotated_centroid_frame(gamma, gammadash, alpha)

    def rotated_frame_dash(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        gamma, gammadash, gammadashdash, alpha, alphadash = self._scalar_inputs()
        return rotated_centroid_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
        )

    def frame_torsion(self) -> jax.Array:
        return _torsion_centroid(*self._scalar_inputs())

    def frame_binormal_curvature(self) -> jax.Array:
        return _binormal_curvature_centroid(*self._scalar_inputs())

    def dframe_torsion_by_dcoeff_vjp(self, v: object) -> Derivative:
        gamma, gammadash, gammadashdash, alpha, alphadash = self._scalar_inputs()
        grad0, grad1, grad2, grad4, grad5 = _centroid_torsion_vjps(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
            v,
        )
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def dframe_binormal_curvature_by_dcoeff_vjp(self, v: object) -> Derivative:
        gamma, gammadash, gammadashdash, alpha, alphadash = self._scalar_inputs()
        grad0, grad1, grad2, grad4, grad5 = _centroid_binormal_curvature_vjps(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
            v,
        )
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )

    def rotated_frame_dcoeff_vjp(
        self,
        v0: object,
        v1: object,
        v2: object,
    ) -> Derivative:
        gamma, gammadash, _gammadashdash, alpha, _alphadash = self._scalar_inputs()
        grad0, grad1, grad2 = jax.vjp(
            rotated_centroid_frame,
            gamma,
            gammadash,
            alpha,
        )[1](_cotangent3(v0, v1, v2))
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad2)
        )

    def rotated_frame_dash_dcoeff_vjp(
        self,
        v0: object,
        v1: object,
        v2: object,
    ) -> Derivative:
        gamma, gammadash, gammadashdash, alpha, alphadash = self._scalar_inputs()
        grad0, grad1, grad2, grad4, grad5 = jax.vjp(
            rotated_centroid_frame_dash,
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
        )[1](_cotangent3(v0, v1, v2))
        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4)
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)
        )
