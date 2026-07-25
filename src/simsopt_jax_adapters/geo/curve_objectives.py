"""Public JAX-backed curve objective wrappers.

The legacy :mod:`simsopt.geo.curveobjectives` classes remain the CPU/C++
compatibility surface. These wrappers expose the same scalar objective
contracts while keeping the value/gradient kernels in JAX and avoiding the
C++ point-cloud candidate cullers for distance penalties.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, lax
from scipy.spatial.distance import cdist

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.geo._curve_surface_distance_owners import (
    curve_surface_distance_owners,
)
from simsopt_jax.core._math_utils import (
    as_jax_float64 as _as_jax_float64,
)
from simsopt_jax.core.curve_geometry import pair_linking_number_pure
from simsopt_jax.core.curve_kernels import (
    curve_curve_distance_penalty_pure,
    curve_surface_distance_penalty_pure,
    curve_length_from_incremental_arclength_pure,
    curvature_p_norm_from_kappa_pure,
)
from simsopt_jax.geo._pairwise_reductions import (
    _chunk_rows,
    _masked_pairwise_distances,
    _pairwise_distances,
    _resolve_pairwise_penalty_chunk_size,
    _use_dense_pairwise_path,
)
from simsopt_jax.runtime.host_boundary import (
    host_float as _host_float,
    host_float64 as _as_numpy_float64,
)

jit = jax.jit

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


@jit
def curve_length_pure(l):
    return curve_length_from_incremental_arclength_pure(l)


@jit
def _curve_length_grad(l):
    return grad(curve_length_pure)(l)


def _curve_jax_position_and_tangent(curve):
    return _as_jax_float64(curve.gamma()), _as_jax_float64(curve.gammadash())


def _curve_position_samples(curve, downsample=1):
    gamma = curve.gamma()
    return gamma if downsample == 1 else gamma[::downsample]


def _curve_pair_minimum_distance(curves, i, j, downsample=1):
    return np.min(
        cdist(
            _curve_position_samples(curves[i], downsample),
            _curve_position_samples(curves[j], downsample),
        )
    )


def _curve_numpy_position_and_tangent(curve):
    return _as_numpy_float64(curve.gamma()).copy(), _as_numpy_float64(
        curve.gammadash()
    ).copy()


def _curve_surface_geometry_snapshot(curves, surface):
    curve_positions = []
    curve_tangents = []
    for curve in curves:
        gamma, gammadash = _curve_numpy_position_and_tangent(curve)
        curve_positions.append(gamma)
        curve_tangents.append(gammadash)
    surface_gamma = _as_numpy_float64(surface.gamma().reshape((-1, 3))).copy()
    surface_normals = _as_numpy_float64(surface.normal().reshape((-1, 3))).copy()
    return curve_positions, curve_tangents, surface_gamma, surface_normals


def _add_curve_vjp(buffer, values, downsample):
    if downsample == 1:
        buffer += values
    else:
        buffer[::downsample] += values


def _curve_vjp_buffers(curves):
    return [np.zeros_like(c.gamma()) for c in curves], [
        np.zeros_like(c.gammadash()) for c in curves
    ]


def _sum_curve_vjp_contributions(curves, dgamma_vjps, dgammadash_vjps):
    return sum(
        curve.dgamma_by_dcoeff_vjp(dgamma_vjp)
        + curve.dgammadash_by_dcoeff_vjp(dgammadash_vjp)
        for curve, dgamma_vjp, dgammadash_vjp in zip(
            curves, dgamma_vjps, dgammadash_vjps
        )
    )


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    return curvature_p_norm_from_kappa_pure(kappa, gammadash, p, desired_kappa)


@jit
def _lp_curve_curvature_grad(kappa, gammadash, p, threshold):
    return grad(Lp_curvature_pure, argnums=(0, 1))(kappa, gammadash, p, threshold)


@jit
def curvature_barrier_pure(kappa, gammadash, threshold):
    threshold_jax = _as_jax_float64(threshold)
    two = _as_jax_float64(2.0)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    feasible = kappa < threshold_jax
    safe_ratio = jnp.where(feasible, kappa / threshold_jax, two)
    barrier = -jnp.log1p(-safe_ratio)
    barrier = jnp.where(feasible, barrier, jnp.inf)
    return jnp.mean(barrier * arc_length)


@jit
def _curvature_barrier_grad(kappa, gammadash, threshold):
    return grad(curvature_barrier_pure, argnums=(0, 1))(kappa, gammadash, threshold)


def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    return curve_curve_distance_penalty_pure(gamma1, l1, gamma2, l2, minimum_distance)


def cc_distance_barrier_pure(gamma1, l1, gamma2, l2, minimum_distance):
    gamma1 = _as_jax_float64(gamma1)
    l1 = _as_jax_float64(l1)
    gamma2 = _as_jax_float64(gamma2)
    l2 = _as_jax_float64(l2)
    minimum_distance_jax = _as_jax_float64(minimum_distance)
    half = _as_jax_float64(0.5)
    zero = _as_jax_float64(0.0)
    row_count = int(gamma1.shape[0])
    col_count = int(gamma2.shape[0])
    if row_count == 0 or col_count == 0:
        return zero
    normalization = _as_jax_float64(row_count * col_count)

    arc_length_1 = jnp.linalg.norm(l1, axis=1)
    arc_length_2 = jnp.linalg.norm(l2, axis=1)
    chunk_size = _resolve_pairwise_penalty_chunk_size()
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        dists = _pairwise_distances(gamma1, gamma2)
        alen = arc_length_1[:, None] * arc_length_2[None, :]
        feasible = dists > minimum_distance_jax
        safe_ratio = jnp.where(feasible, minimum_distance_jax / dists, half)
        barrier = -jnp.log1p(-safe_ratio)
        barrier = jnp.where(feasible, barrier, jnp.inf)
        return jnp.sum(alen * barrier) / normalization

    gamma1_chunks, gamma1_masks = _chunk_rows(gamma1, chunk_size)
    gamma2_chunks, gamma2_masks = _chunk_rows(gamma2, chunk_size)
    arc_length_1_chunks, _ = _chunk_rows(arc_length_1, chunk_size)
    arc_length_2_chunks, _ = _chunk_rows(arc_length_2, chunk_size)

    def _scan_gamma1_chunks(carry, gamma1_inputs):
        total, feasible_all = carry
        gamma1_chunk, arc_length_1_chunk, gamma1_mask = gamma1_inputs

        def _scan_gamma2_chunks(inner_carry, gamma2_inputs):
            inner_total, inner_feasible = inner_carry
            gamma2_chunk, arc_length_2_chunk, gamma2_mask = gamma2_inputs
            valid = gamma1_mask[:, None] & gamma2_mask[None, :]
            dists = _masked_pairwise_distances(
                gamma1_chunk,
                gamma2_chunk,
                valid,
                minimum_distance_jax,
            )
            feasible = jnp.logical_or(~valid, dists > minimum_distance_jax)
            safe_dists = jnp.where(valid, dists, minimum_distance_jax)
            safe_ratio = jnp.where(
                valid,
                jnp.where(feasible, minimum_distance_jax / safe_dists, half),
                zero,
            )
            barrier = -jnp.log1p(-safe_ratio)
            alen = arc_length_1_chunk[:, None] * arc_length_2_chunk[None, :]
            block_total = jnp.sum(jnp.where(valid, alen * barrier, zero))
            return (inner_total + block_total, inner_feasible & jnp.all(feasible)), None

        (total, feasible_all), _ = lax.scan(
            jax.checkpoint(_scan_gamma2_chunks),
            (total, feasible_all),
            (gamma2_chunks, arc_length_2_chunks, gamma2_masks),
        )
        return (total, feasible_all), None

    (total, feasible_all), _ = lax.scan(
        _scan_gamma1_chunks,
        (zero, jnp.asarray(True)),
        (gamma1_chunks, arc_length_1_chunks, gamma1_masks),
    )
    return jnp.where(feasible_all, total / normalization, jnp.inf)


@jit
def _cc_distance_barrier_grad(gamma1, l1, gamma2, l2, minimum_distance):
    return grad(cc_distance_barrier_pure, argnums=(0, 1, 2, 3))(
        gamma1,
        l1,
        gamma2,
        l2,
        minimum_distance,
    )


@jit
def _cc_distance_grad(gamma1, l1, gamma2, l2, minimum_distance):
    return grad(cc_distance_pure, argnums=(0, 1, 2, 3))(
        gamma1,
        l1,
        gamma2,
        l2,
        minimum_distance,
    )


def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    return curve_surface_distance_penalty_pure(
        gammac,
        lc,
        gammas,
        ns,
        minimum_distance,
    )


@jit
def _cs_distance_grad(gammac, lc, gammas, ns, minimum_distance):
    return grad(cs_distance_pure, argnums=(0, 1, 2, 3))(
        gammac,
        lc,
        gammas,
        ns,
        minimum_distance,
    )


@jit
def curve_msc_pure(kappa, gammadash):
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length) / jnp.mean(arc_length)


@jit
def _curve_msc_grad(kappa, gammadash):
    return grad(curve_msc_pure, argnums=(0, 1))(kappa, gammadash)


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
        return _host_float(
            Lp_curvature_pure(
                _as_jax_float64(self.curve.kappa()),
                _as_jax_float64(self.curve.gammadash()),
                _as_jax_float64(self.p),
                _as_jax_float64(self.threshold),
            )
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
        return _host_float(res)

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

    def __init__(self, curves, surface, minimum_distance, downsample=1):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance
        self.downsample = downsample
        super().__init__(depends_on=curve_surface_distance_owners(curves, surface))

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
        surface_points = np.asarray(
            self.surface.gamma()[::self.downsample, ::self.downsample, :],
            dtype=np.float64,
        ).reshape((-1, 3))
        return min(
            float(
                np.min(
                    np.linalg.norm(
                        np.asarray(
                            curve.gamma()[::self.downsample, :],
                            dtype=np.float64,
                        )[:, None, :]
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
        return _host_float(res)

    @derivative_dec
    def dJ(self):
        """Return the derivative with respect to curve and surface DOFs."""
        curve_positions, curve_tangents, surface_gamma, surface_normals = (
            self._evaluation_geometry()
        )
        dgamma_buffers, dgammadash_buffers = _curve_vjp_buffers(self.curves)
        surface_gamma_vjp = np.zeros_like(self.surface.gamma())
        surface_normal_vjp = np.zeros_like(self.surface.normal())
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for index, (gamma, gammadash) in enumerate(
            zip(curve_positions, curve_tangents)
        ):
            grad_gamma, grad_gammadash, grad_surface_gamma, grad_surface_normal = (
                _cs_distance_grad(
                    _as_jax_float64(gamma),
                    _as_jax_float64(gammadash),
                    surface_gamma,
                    surface_normals,
                    minimum_distance,
                )
            )
            dgamma_buffers[index] += _as_numpy_float64(grad_gamma)
            dgammadash_buffers[index] += _as_numpy_float64(grad_gammadash)
            surface_gamma_vjp += _as_numpy_float64(grad_surface_gamma).reshape(
                surface_gamma_vjp.shape
            )
            surface_normal_vjp += _as_numpy_float64(grad_surface_normal).reshape(
                surface_normal_vjp.shape
            )
        curve_derivative = _sum_curve_vjp_contributions(
            self.curves,
            dgamma_buffers,
            dgammadash_buffers,
        )
        surface_derivative = Derivative(
            {
                self.surface: (
                    self.surface.dgamma_by_dcoeff_vjp(surface_gamma_vjp)
                    + self.surface.dnormal_by_dcoeff_vjp(surface_normal_vjp)
                )
            }
        )
        return curve_derivative + surface_derivative

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
