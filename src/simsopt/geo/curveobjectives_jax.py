"""Public JAX-backed curve objective wrappers.

The legacy :mod:`simsopt.geo.curveobjectives` classes remain the CPU/C++
compatibility surface. These wrappers expose the same scalar objective
contracts while keeping the value/gradient kernels in JAX and avoiding the
C++ point-cloud candidate cullers for distance penalties.
"""

from __future__ import annotations

import numpy as np

from .._core.derivative import Derivative, derivative_dec
from .._core.optimizable import Optimizable
from ..jax_core.curve_geometry import pair_linking_number_pure
from .curveobjectives import (
    Lp_curvature_pure,
    _add_curve_vjp,
    _as_jax_float64,
    _as_numpy_float64,
    _cc_distance_barrier_grad,
    _cc_distance_grad,
    _cs_distance_grad,
    _curvature_barrier_grad,
    _curve_jax_position_and_tangent,
    _curve_length_grad,
    _curve_msc_grad,
    _curve_pair_minimum_distance,
    _curve_surface_geometry_snapshot,
    _curve_vjp_buffers,
    _lp_curve_curvature_grad,
    _sum_curve_vjp_contributions,
    cc_distance_barrier_pure,
    cc_distance_pure,
    cs_distance_pure,
    curvature_barrier_pure,
    curve_length_pure,
    curve_msc_pure,
)

__all__ = [
    "CurveCurveDistanceBarrierJAX",
    "CurveCurveDistanceJAX",
    "CurveLengthJAX",
    "CurveSurfaceDistanceJAX",
    "LpCurveCurvatureBarrierJAX",
    "LpCurveCurvatureJAX",
    "LinkingNumberJAX",
    "MeanSquaredCurvatureJAX",
]


class CurveLengthJAX(Optimizable):
    """JAX-backed mirror of :class:`~simsopt.geo.CurveLength`."""

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        return curve_length_pure(_as_jax_float64(self.curve.incremental_arclength()))

    @derivative_dec
    def dJ(self):
        arc = _as_jax_float64(self.curve.incremental_arclength())
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            _as_numpy_float64(_curve_length_grad(arc))
        )

    return_fn_map = {"J": J, "dJ": dJ}


class LpCurveCurvatureJAX(Optimizable):
    """JAX-backed mirror of :class:`~simsopt.geo.LpCurveCurvature`."""

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])

    def J(self):
        return Lp_curvature_pure(
            _as_jax_float64(self.curve.kappa()),
            _as_jax_float64(self.curve.gammadash()),
            _as_jax_float64(self.p),
            _as_jax_float64(self.threshold),
        )

    @derivative_dec
    def dJ(self):
        kappa = _as_jax_float64(self.curve.kappa())
        gammadash = _as_jax_float64(self.curve.gammadash())
        grad_kappa, grad_gammadash = _lp_curve_curvature_grad(
            kappa,
            gammadash,
            _as_jax_float64(self.p),
            _as_jax_float64(self.threshold),
        )
        return self.curve.dkappa_by_dcoeff_vjp(
            _as_numpy_float64(grad_kappa)
        ) + self.curve.dgammadash_by_dcoeff_vjp(_as_numpy_float64(grad_gammadash))

    return_fn_map = {"J": J, "dJ": dJ}


class LpCurveCurvatureBarrierJAX(Optimizable):
    """JAX-backed mirror of :class:`~simsopt.geo.LpCurveCurvatureBarrier`."""

    def __init__(self, curve, threshold):
        self.curve = curve
        self.threshold = threshold
        super().__init__(depends_on=[curve])

    def J(self):
        return curvature_barrier_pure(
            _as_jax_float64(self.curve.kappa()),
            _as_jax_float64(self.curve.gammadash()),
            _as_jax_float64(self.threshold),
        )

    @derivative_dec
    def dJ(self):
        kappa = _as_jax_float64(self.curve.kappa())
        gammadash = _as_jax_float64(self.curve.gammadash())
        grad_kappa, grad_gammadash = _curvature_barrier_grad(
            kappa,
            gammadash,
            _as_jax_float64(self.threshold),
        )
        return self.curve.dkappa_by_dcoeff_vjp(
            _as_numpy_float64(grad_kappa)
        ) + self.curve.dgammadash_by_dcoeff_vjp(_as_numpy_float64(grad_gammadash))

    return_fn_map = {"J": J, "dJ": dJ}


class MeanSquaredCurvatureJAX(Optimizable):
    """JAX-backed mirror of :class:`~simsopt.geo.MeanSquaredCurvature`."""

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        return curve_msc_pure(
            _as_jax_float64(self.curve.kappa()),
            _as_jax_float64(self.curve.gammadash()),
        )

    @derivative_dec
    def dJ(self):
        kappa = _as_jax_float64(self.curve.kappa())
        gammadash = _as_jax_float64(self.curve.gammadash())
        grad_kappa, grad_gammadash = _curve_msc_grad(kappa, gammadash)
        return self.curve.dkappa_by_dcoeff_vjp(
            _as_numpy_float64(grad_kappa)
        ) + self.curve.dgammadash_by_dcoeff_vjp(_as_numpy_float64(grad_gammadash))

    return_fn_map = {"J": J, "dJ": dJ}


class _CurveCurveDistanceJAXBase(Optimizable):
    def __init__(self, curves, minimum_distance, num_basecurves=None, downsample=1):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self.num_basecurves = num_basecurves or len(curves)
        self.downsample = downsample
        super().__init__(depends_on=curves)

    def _iter_curve_pair_indices(self):
        for i in range(len(self.curves)):
            for j in range(min(i, self.num_basecurves)):
                yield i, j

    def _pair_data(self, i, j):
        gamma1, gammadash1 = _curve_jax_position_and_tangent(self.curves[i])
        gamma2, gammadash2 = _curve_jax_position_and_tangent(self.curves[j])
        if self.downsample != 1:
            gamma1 = gamma1[:: self.downsample]
            gammadash1 = gammadash1[:: self.downsample]
            gamma2 = gamma2[:: self.downsample]
            gammadash2 = gammadash2[:: self.downsample]
        return gamma1, gammadash1, gamma2, gammadash2

    def shortest_distance(self):
        return min(
            _curve_pair_minimum_distance(self.curves, i, j, self.downsample)
            for i, j in self._iter_curve_pair_indices()
        )


class CurveCurveDistanceJAX(_CurveCurveDistanceJAXBase):
    """JAX-backed curve-curve distance penalty without C++ candidate culling."""

    def J(self):
        res = _as_jax_float64(0.0)
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for i, j in self._iter_curve_pair_indices():
            res += cc_distance_pure(*self._pair_data(i, j), minimum_distance)
        return res

    @derivative_dec
    def dJ(self):
        dgamma_buffers, dgammadash_buffers = _curve_vjp_buffers(self.curves)
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for i, j in self._iter_curve_pair_indices():
            grad0, grad1, grad2, grad3 = _cc_distance_grad(
                *self._pair_data(i, j),
                minimum_distance,
            )
            _add_curve_vjp(dgamma_buffers[i], _as_numpy_float64(grad0), self.downsample)
            _add_curve_vjp(
                dgammadash_buffers[i],
                _as_numpy_float64(grad1),
                self.downsample,
            )
            _add_curve_vjp(dgamma_buffers[j], _as_numpy_float64(grad2), self.downsample)
            _add_curve_vjp(
                dgammadash_buffers[j],
                _as_numpy_float64(grad3),
                self.downsample,
            )
        return _sum_curve_vjp_contributions(
            self.curves,
            dgamma_buffers,
            dgammadash_buffers,
        )

    return_fn_map = {"J": J, "dJ": dJ}


class CurveCurveDistanceBarrierJAX(_CurveCurveDistanceJAXBase):
    """JAX-backed curve-curve strict distance barrier."""

    def __init__(self, curves, minimum_distance, num_basecurves=None):
        super().__init__(curves, minimum_distance, num_basecurves=num_basecurves)

    def J(self):
        res = _as_jax_float64(0.0)
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for i, j in self._iter_curve_pair_indices():
            res += cc_distance_barrier_pure(*self._pair_data(i, j), minimum_distance)
        return res

    @derivative_dec
    def dJ(self):
        dgamma_buffers, dgammadash_buffers = _curve_vjp_buffers(self.curves)
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for i, j in self._iter_curve_pair_indices():
            grad0, grad1, grad2, grad3 = _cc_distance_barrier_grad(
                *self._pair_data(i, j),
                minimum_distance,
            )
            dgamma_buffers[i] += _as_numpy_float64(grad0)
            dgammadash_buffers[i] += _as_numpy_float64(grad1)
            dgamma_buffers[j] += _as_numpy_float64(grad2)
            dgammadash_buffers[j] += _as_numpy_float64(grad3)
        return _sum_curve_vjp_contributions(
            self.curves,
            dgamma_buffers,
            dgammadash_buffers,
        )

    return_fn_map = {"J": J, "dJ": dJ}


class CurveSurfaceDistanceJAX(Optimizable):
    """JAX-backed curve-surface distance penalty without C++ candidate culling."""

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance
        super().__init__(depends_on=curves)

    def _evaluation_geometry(self):
        curve_positions, curve_tangents, surface_gamma, surface_normals = (
            _curve_surface_geometry_snapshot(self.curves, self.surface)
        )
        return (
            curve_positions,
            curve_tangents,
            _as_jax_float64(surface_gamma),
            _as_jax_float64(surface_normals),
        )

    def shortest_distance(self):
        surface_points = np.asarray(self.surface.gamma(), dtype=np.float64).reshape(
            (-1, 3)
        )
        return min(
            float(
                np.min(
                    np.linalg.norm(
                        np.asarray(curve.gamma(), dtype=np.float64)[:, None, :]
                        - surface_points[None, :, :],
                        axis=-1,
                    )
                )
            )
            for curve in self.curves
        )

    def J(self):
        curve_positions, curve_tangents, surface_gamma, surface_normals = (
            self._evaluation_geometry()
        )
        res = _as_jax_float64(0.0)
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for gamma, gammadash in zip(curve_positions, curve_tangents):
            res += cs_distance_pure(
                _as_jax_float64(gamma),
                _as_jax_float64(gammadash),
                surface_gamma,
                surface_normals,
                minimum_distance,
            )
        return res

    @derivative_dec
    def dJ(self):
        curve_positions, curve_tangents, surface_gamma, surface_normals = (
            self._evaluation_geometry()
        )
        dgamma_buffers, dgammadash_buffers = _curve_vjp_buffers(self.curves)
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for index, (gamma, gammadash) in enumerate(
            zip(curve_positions, curve_tangents)
        ):
            grad_gamma, grad_gammadash = _cs_distance_grad(
                _as_jax_float64(gamma),
                _as_jax_float64(gammadash),
                surface_gamma,
                surface_normals,
                minimum_distance,
            )
            dgamma_buffers[index] += _as_numpy_float64(grad_gamma)
            dgammadash_buffers[index] += _as_numpy_float64(grad_gammadash)
        return _sum_curve_vjp_contributions(
            self.curves,
            dgamma_buffers,
            dgammadash_buffers,
        )

    return_fn_map = {"J": J, "dJ": dJ}


class LinkingNumberJAX(Optimizable):
    """JAX-backed mirror of :class:`~simsopt.geo.LinkingNumber`."""

    def __init__(self, curves, downsample=1):
        self.curves = curves
        for curve in curves:
            assert np.mod(len(curve.quadpoints), downsample) == 0, (
                f"Downsample {downsample} does not divide the number of quadpoints "
                f"{len(curve.quadpoints)}."
            )
        self.downsample = downsample
        self.dphis = np.array(
            [(c.quadpoints[1] - c.quadpoints[0]) * downsample for c in self.curves]
        )
        super().__init__(depends_on=curves)

    def J(self):
        total = _as_jax_float64(0.0)
        for p in range(1, len(self.curves)):
            gamma_p, gammadash_p = _curve_jax_position_and_tangent(self.curves[p])
            if self.downsample != 1:
                gamma_p = gamma_p[:: self.downsample]
                gammadash_p = gammadash_p[:: self.downsample]
            dphi_p = _as_jax_float64(self.dphis[p])
            for q in range(p):
                gamma_q, gammadash_q = _curve_jax_position_and_tangent(self.curves[q])
                if self.downsample != 1:
                    gamma_q = gamma_q[:: self.downsample]
                    gammadash_q = gammadash_q[:: self.downsample]
                total += _as_jax_float64(
                    pair_linking_number_pure(
                        gamma_p,
                        gammadash_p,
                        gamma_q,
                        gammadash_q,
                        dphi_p,
                        _as_jax_float64(self.dphis[q]),
                    )
                )
        return total

    @derivative_dec
    def dJ(self):
        return Derivative({})

    return_fn_map = {"J": J, "dJ": dJ}
