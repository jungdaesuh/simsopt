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
from simsopt_jax.core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    scalar_like as _scalar_like,
)
from simsopt_jax.core.curve_geometry import pair_linking_number_pure
from simsopt_jax.geo._pairwise_reductions import (
    _chunk_rows,
    _chunk_rows_with_valid_weights,
    _masked_pairwise_distances,
    _pairwise_distances,
    _resolve_pairwise_penalty_chunk_size,
    _use_dense_pairwise_path,
)
from simsopt_jax.runtime.host_boundary import host_float64 as _as_numpy_float64

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
    return jnp.mean(l)


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
    p_jax = _scalar_like(kappa, p)
    desired_kappa_jax = _scalar_like(kappa, desired_kappa)
    zero = _scalar_like(kappa, 0.0)
    one = _scalar_like(kappa, 1.0)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(kappa - desired_kappa_jax, zero)
    return (one / p_jax) * jnp.mean((excess**p_jax) * arc_length)


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
    gamma1 = _as_jax_float64(gamma1)
    l1 = _as_jax_float64(l1)
    gamma2 = _as_jax_float64(gamma2)
    l2 = _as_jax_float64(l2)
    minimum_distance_jax = _scalar_like(gamma1, minimum_distance)
    zero = _scalar_like(gamma1, 0.0)
    row_count = int(gamma1.shape[0])
    col_count = int(gamma2.shape[0])
    if row_count == 0 or col_count == 0:
        return zero
    normalization = _scalar_like(gamma1, row_count * col_count)

    arc_length_1 = jnp.linalg.norm(l1, axis=1)
    arc_length_2 = jnp.linalg.norm(l2, axis=1)
    chunk_size = _resolve_pairwise_penalty_chunk_size()
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        dists = _pairwise_distances(gamma1, gamma2)
        alen = arc_length_1[:, None] * arc_length_2[None, :]
        excess = jnp.maximum(minimum_distance_jax - dists, zero)
        return jnp.sum(alen * jnp.square(excess)) / normalization

    gamma1_chunks, gamma1_masks = _chunk_rows(gamma1, chunk_size)
    gamma2_chunks, gamma2_masks = _chunk_rows(gamma2, chunk_size)
    arc_length_1_chunks, _ = _chunk_rows(arc_length_1, chunk_size)
    arc_length_2_chunks, _ = _chunk_rows(arc_length_2, chunk_size)

    def _scan_gamma1_chunks(total, gamma1_inputs):
        gamma1_chunk, arc_length_1_chunk, gamma1_mask = gamma1_inputs

        def _scan_gamma2_chunks(row_total, gamma2_inputs):
            gamma2_chunk, arc_length_2_chunk, gamma2_mask = gamma2_inputs
            valid = gamma1_mask[:, None] & gamma2_mask[None, :]
            dists = _masked_pairwise_distances(
                gamma1_chunk,
                gamma2_chunk,
                valid,
                minimum_distance_jax,
            )
            alen = arc_length_1_chunk[:, None] * arc_length_2_chunk[None, :]
            safe_dists = jnp.where(valid, dists, minimum_distance_jax)
            diff = minimum_distance_jax - safe_dists
            excess = jnp.where(diff > zero, diff, zero)
            block_total = jnp.sum(jnp.where(valid, alen * jnp.square(excess), zero))
            return row_total + block_total, None

        total, _ = lax.scan(
            jax.checkpoint(_scan_gamma2_chunks),
            total,
            (gamma2_chunks, arc_length_2_chunks, gamma2_masks),
        )
        return total, None

    total, _ = lax.scan(
        _scan_gamma1_chunks,
        zero,
        (gamma1_chunks, arc_length_1_chunks, gamma1_masks),
    )
    return total / normalization


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
    gammac = _as_jax_float64(gammac)
    lc = _as_jax_float64(lc)
    gammas = _as_jax_float64(gammas)
    ns = _as_jax_float64(ns)
    minimum_distance_jax = _scalar_like(gammac, minimum_distance)
    zero = minimum_distance_jax - minimum_distance_jax
    one = _scalar_like(gammac, 1.0)
    row_count = int(gammac.shape[0])
    col_count = int(gammas.shape[0])
    if row_count == 0 or col_count == 0:
        return zero

    curve_weights = jnp.linalg.norm(lc, axis=1)
    surface_weights = jnp.linalg.norm(ns, axis=1)
    chunk_size = _resolve_pairwise_penalty_chunk_size()
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        dists = _pairwise_distances(gammac, gammas)
        integralweight = curve_weights[:, None] * surface_weights[None, :]
        diff = minimum_distance_jax - dists
        excess = jnp.where(diff > zero, diff, zero)
        normalization = jnp.sum(jnp.broadcast_to(one, dists.shape))
        return jnp.sum(integralweight * jnp.square(excess)) / normalization

    def _chunk_with_weights(array):
        return _chunk_rows_with_valid_weights(array, chunk_size, one, zero)

    gammac_chunks, gammac_masks = _chunk_with_weights(gammac)
    gammas_chunks, gammas_masks = _chunk_with_weights(gammas)
    curve_weight_chunks, _ = _chunk_with_weights(curve_weights)
    surface_weight_chunks, _ = _chunk_with_weights(surface_weights)

    def _scan_curve_chunks(carry, curve_inputs):
        total, normalization = carry
        gammac_chunk, curve_weight_chunk, gammac_mask = curve_inputs

        def _scan_surface_chunks(inner_carry, surface_inputs):
            row_total, row_normalization = inner_carry
            gammas_chunk, surface_weight_chunk, gammas_mask = surface_inputs
            valid_weight = gammac_mask[:, None] * gammas_mask[None, :]
            valid = valid_weight > zero
            dists = _masked_pairwise_distances(
                gammac_chunk,
                gammas_chunk,
                valid,
                minimum_distance_jax,
            )
            integralweight = curve_weight_chunk[:, None] * surface_weight_chunk[None, :]
            safe_dists = jnp.where(valid, dists, minimum_distance_jax)
            diff = minimum_distance_jax - safe_dists
            excess = jnp.where(diff > zero, diff, zero)
            block_total = jnp.sum(
                jnp.where(valid, integralweight * jnp.square(excess), zero)
            )
            block_normalization = jnp.sum(valid_weight)
            return (
                row_total + block_total,
                row_normalization + block_normalization,
            ), None

        (total, normalization), _ = lax.scan(
            jax.checkpoint(_scan_surface_chunks),
            (total, normalization),
            (gammas_chunks, surface_weight_chunks, gammas_masks),
        )
        return (total, normalization), None

    (total, normalization), _ = lax.scan(
        _scan_curve_chunks,
        (zero, zero),
        (gammac_chunks, curve_weight_chunks, gammac_masks),
    )
    return total / normalization


@jit
def _cs_distance_grad(gammac, lc, gammas, ns, minimum_distance):
    return grad(cs_distance_pure, argnums=(0, 1))(
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
