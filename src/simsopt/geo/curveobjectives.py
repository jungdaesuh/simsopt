from deprecated import deprecated

import numpy as np
from jax import grad, vjp, lax
import jax.numpy as jnp
import jax

from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
from .._core.jax_host_boundary import host_array as _host_array
from ..backend.runtime import is_jax_backend, raise_if_target_lane_bypass
from ..jax_core._math_utils import as_jax_float64 as _runtime_as_jax_float64
from ..jax_core._math_utils import as_jax_int32 as _runtime_as_jax_int32
from ..jax_core._math_utils import as_runtime_float64 as _runtime_as_runtime_float64
from ._pairwise_reductions import (
    _chunk_rows,
    _chunk_rows_with_valid_weights,
    _masked_pairwise_distances,
    _pairwise_distances,
    _pairwise_rowwise_min_distance,
    _pairwise_rowwise_pnorm_distance,
    _resolve_pairwise_penalty_chunk_size,
    _use_dense_pairwise_path,
    pairwise_min_distance_pure,
)
from ._simsoptpp import sopp_namespace
from .framedcurve import FramedCurveCentroid

sopp = sopp_namespace(
    "get_pointclouds_closer_than_threshold_within_collection",
    "get_pointclouds_closer_than_threshold_between_two_collections",
    "compute_linking_number",
    kind="function",
)

__all__ = [
    "CurveLength",
    "LpCurveCurvature",
    "LpCurveCurvatureBarrier",
    "LpCurveTorsion",
    "CurveCurveDistance",
    "CurveCurveDistanceBarrier",
    "CurveSurfaceDistance",
    "ArclengthVariation",
    "MeanSquaredCurvature",
    "LinkingNumber",
    "FramedCurveTwist",
    "MinCurveCurveDistance",
    "pairwise_min_distance_pure",
]


@jit
def curve_length_pure(l):
    """
    This function is used in a Python+Jax implementation of the curve length formula.
    """
    return jnp.mean(l)


@jit
def _curve_length_grad(l):
    return grad(curve_length_pure)(l)


def _as_jax_float64(value):
    return _runtime_as_jax_float64(value)


def _as_jax_int32(value):
    return _runtime_as_jax_int32(value)


def _scalar_like(reference, value):
    return _runtime_as_runtime_float64(value, reference=reference)


def _as_numpy_float64(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    return _host_array(value, dtype=np.float64)


def _curve_jax_position_and_tangent(curve):
    return _as_jax_float64(curve.gamma()), _as_jax_float64(curve.gammadash())


def _curve_position_samples(curve, downsample=1):
    gamma = curve.gamma()
    return gamma if downsample == 1 else gamma[::downsample]


def _curve_pair_minimum_distance(curves, i, j, downsample=1):
    from scipy.spatial.distance import cdist

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


def _curve_pair_jax_data(curves, i, j, downsample=1):
    gamma1, l1 = _curve_jax_position_and_tangent(curves[i])
    gamma2, l2 = _curve_jax_position_and_tangent(curves[j])
    if downsample != 1:
        gamma1 = gamma1[::downsample]
        l1 = l1[::downsample]
        gamma2 = gamma2[::downsample]
        l2 = l2[::downsample]
    return gamma1, l1, gamma2, l2


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


class CurveLength(Optimizable):
    r"""
    CurveLength is a class that computes the length of a curve, i.e.

    .. math::
        J = \int_{\text{curve}}~dl.

    """

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return curve_length_pure(_as_jax_float64(self.curve.incremental_arclength()))

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            _as_numpy_float64(
                _curve_length_grad(_as_jax_float64(self.curve.incremental_arclength()))
            )
        )

    return_fn_map = {"J": J, "dJ": dJ}


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    """
    This function is used in a Python+Jax implementation of the curvature penalty term.
    """
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
    """
    A strict interior-point barrier used to optimize against the engineering
    curvature limit.

    The production/reporting contract may treat the configured limit as
    inclusive (``kappa <= threshold``), but the barrier itself stays strict so
    its value and gradient remain the expected log-barrier surrogate inside the
    feasible region. The value is therefore finite only when every sampled
    curvature stays strictly below ``threshold``.
    """
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


class LpCurveCurvature(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the curve's curvature, and penalizes where the local curve curvature exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\kappa - \kappa_0, 0)^p ~dl

    where :math:`\kappa_0` is a threshold curvature, given by the argument ``threshold``.
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        p = _as_jax_float64(self.p)
        threshold = _as_jax_float64(self.threshold)
        return Lp_curvature_pure(
            _as_jax_float64(self.curve.kappa()),
            _as_jax_float64(self.curve.gammadash()),
            p,
            threshold,
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        kappa = _as_jax_float64(self.curve.kappa())
        gammadash = _as_jax_float64(self.curve.gammadash())
        p = _as_jax_float64(self.p)
        threshold = _as_jax_float64(self.threshold)
        grad0, grad1 = _lp_curve_curvature_grad(
            kappa,
            gammadash,
            p,
            threshold,
        )
        grad0 = _as_numpy_float64(grad0)
        grad1 = _as_numpy_float64(grad1)
        return self.curve.dkappa_by_dcoeff_vjp(
            grad0
        ) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {"J": J, "dJ": dJ}


class LpCurveCurvatureBarrier(Optimizable):
    r"""
    Strict interior-point barrier for enforcing a maximum curve curvature:

    .. math::
        J = \int_{\text{curve}}
        -\log\left(1 - \frac{\kappa}{\kappa_0}\right) ~dl

    The barrier is finite only when every sampled curvature stays strictly
    below ``threshold`` and tends to ``+\infty`` as the threshold is
    approached from below.
    """

    def __init__(self, curve, threshold):
        self.curve = curve
        self.threshold = threshold
        super().__init__(depends_on=[curve])

    def J(self):
        kappa = _as_jax_float64(self.curve.kappa())
        gammadash = _as_jax_float64(self.curve.gammadash())
        threshold = _as_jax_float64(self.threshold)
        return curvature_barrier_pure(
            kappa,
            gammadash,
            threshold,
        )

    @derivative_dec
    def dJ(self):
        kappa = _as_jax_float64(self.curve.kappa())
        gammadash = _as_jax_float64(self.curve.gammadash())
        threshold = _as_jax_float64(self.threshold)
        grad0, grad1 = _curvature_barrier_grad(
            kappa,
            gammadash,
            threshold,
        )
        grad0 = _as_numpy_float64(grad0)
        grad1 = _as_numpy_float64(grad1)
        return self.curve.dkappa_by_dcoeff_vjp(
            grad0
        ) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {"J": J, "dJ": dJ}


@jit
def Lp_torsion_pure(torsion, gammadash, p, threshold):
    """
    This function is used in a Python+Jax implementation of the formula for the torsion penalty term.
    """
    p_jax = _as_jax_float64(p)
    threshold_jax = _as_jax_float64(threshold)
    zero = _as_jax_float64(0.0)
    one = _as_jax_float64(1.0)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    # jax.debug.print("arc_length: {arc_length}",arc_length=arc_length)
    # jax.debug.print('p: {p}',p=p)
    # jax.debug.print('threshold: {threshold}',threshold=threshold)
    # jax.debug.print('binorm: {binorm}',binorm=torsion)
    # jax.debug.print('integrand: {integrand}',integrand=jnp.maximum(jnp.abs(torsion)-threshold, 0)**p)
    excess = jnp.maximum(jnp.abs(torsion) - threshold_jax, zero)
    return (one / p_jax) * jnp.mean((excess**p_jax) * arc_length)


@jit
def _lp_curve_torsion_grad(torsion, gammadash, p, threshold):
    return grad(Lp_torsion_pure, argnums=(0, 1))(torsion, gammadash, p, threshold)


class LpCurveTorsion(Optimizable):
    r"""
    LpCurveTorsion is a class that computes a penalty term based on the :math:`L_p` norm
    of the curve's torsion:

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(|\tau|-\tau_0, 0)^p ~dl.

    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        torsion = _as_jax_float64(self.curve.torsion())
        gammadash = _as_jax_float64(self.curve.gammadash())
        p = _as_jax_float64(self.p)
        threshold = _as_jax_float64(self.threshold)
        return Lp_torsion_pure(
            torsion,
            gammadash,
            p,
            threshold,
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        torsion = _as_jax_float64(self.curve.torsion())
        gammadash = _as_jax_float64(self.curve.gammadash())
        p = _as_jax_float64(self.p)
        threshold = _as_jax_float64(self.threshold)
        grad0, grad1 = _lp_curve_torsion_grad(
            torsion,
            gammadash,
            p,
            threshold,
        )
        grad0 = _as_numpy_float64(grad0)
        grad1 = _as_numpy_float64(grad1)
        return self.curve.dtorsion_by_dcoeff_vjp(
            grad0
        ) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {"J": J, "dJ": dJ}


def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula.
    """
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
    """
    A true interior-point barrier for the pointwise coil-coil separation
    constraint. The value is finite only when every sampled point-pair distance
    stays strictly above ``minimum_distance`` and diverges at the constraint
    boundary.
    """
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


class CurveCurveDistanceBarrier(Optimizable):
    r"""
    ``CurveCurveDistanceBarrier`` is a strict interior-point barrier for
    enforcing a minimum coil-coil separation:

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} b_{i,j}

    where

    .. math::
        b_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j}
        -\log\left(1 - \frac{d_{\min}}{\| \mathbf{r}_i - \mathbf{r}_j \|_2}\right)
        ~dl_j ~dl_i.

    The barrier is finite only when every sampled point-pair distance exceeds
    ``minimum_distance`` and tends to ``+\infty`` as the true sampled minimum
    distance approaches the threshold from above.
    """

    def __init__(self, curves, minimum_distance, num_basecurves=None):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self.num_basecurves = num_basecurves or len(curves)
        super().__init__(depends_on=curves)

    def _iter_curve_pair_indices(self):
        for i in range(len(self.curves)):
            for j in range(min(i, self.num_basecurves)):
                yield i, j

    def shortest_distance(self):
        from scipy.spatial.distance import cdist

        return min(
            np.min(cdist(self.curves[i].gamma(), self.curves[j].gamma()))
            for i, j in self._iter_curve_pair_indices()
        )

    def J(self):
        """
        This returns the value of the quantity.
        """
        res = _as_jax_float64(0.0)
        for i, j in self._iter_curve_pair_indices():
            gamma1, l1, gamma2, l2 = _curve_pair_jax_data(self.curves, i, j)
            res += cc_distance_barrier_pure(
                gamma1,
                l1,
                gamma2,
                l2,
                self.minimum_distance,
            )
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dgamma_by_dcoeff_vjp_vecs, dgammadash_by_dcoeff_vjp_vecs = _curve_vjp_buffers(
            self.curves
        )

        for i, j in self._iter_curve_pair_indices():
            gamma1, l1, gamma2, l2 = _curve_pair_jax_data(self.curves, i, j)
            minimum_distance = _as_jax_float64(self.minimum_distance)
            grad0, grad1, grad2, grad3 = _cc_distance_barrier_grad(
                gamma1,
                l1,
                gamma2,
                l2,
                minimum_distance,
            )
            dgamma_by_dcoeff_vjp_vecs[i] += _as_numpy_float64(grad0)
            dgammadash_by_dcoeff_vjp_vecs[i] += _as_numpy_float64(grad1)
            dgamma_by_dcoeff_vjp_vecs[j] += _as_numpy_float64(grad2)
            dgammadash_by_dcoeff_vjp_vecs[j] += _as_numpy_float64(grad3)

        return _sum_curve_vjp_contributions(
            self.curves,
            dgamma_by_dcoeff_vjp_vecs,
            dgammadash_by_dcoeff_vjp_vecs,
        )

    return_fn_map = {"J": J, "dJ": dJ}


class CurveCurveDistance(Optimizable):
    r"""
    CurveCurveDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} d_{i,j}

    where 

    .. math::
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coils :math:`i` and :math:`j`, respectively.
    :math:`d_\min` is a desired threshold minimum intercoil distance.  This penalty term is zero when the points on coil :math:`i` and 
    coil :math:`j` lie more than :math:`d_\min` away from one another, for :math:`i, j \in \{1, \cdots, \text{num_coils}\}`

    If num_basecurves is passed, then the code only computes the distance to
    the first `num_basecurves` many curves, which is useful when the coils
    satisfy symmetries that can be exploited.

    """

    def __init__(self, curves, minimum_distance, num_basecurves=None, downsample=1):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self.candidates = None
        self.num_basecurves = num_basecurves or len(curves)
        self.downsample = downsample
        super().__init__(depends_on=curves)

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        raise_if_target_lane_bypass("CurveCurveDistance.compute_candidates")
        if self.candidates is None:
            point_clouds = [
                _curve_position_samples(c, self.downsample) for c in self.curves
            ]
            if is_jax_backend():
                from ._distance_jax import get_close_candidates_within_collection

                candidates = get_close_candidates_within_collection(
                    point_clouds,
                    self.minimum_distance,
                    self.num_basecurves,
                )
            else:
                candidates = (
                    sopp.get_pointclouds_closer_than_threshold_within_collection(
                        point_clouds,
                        self.minimum_distance,
                        self.num_basecurves,
                    )
                )
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        return min(
            [self.minimum_distance]
            + [
                _curve_pair_minimum_distance(self.curves, i, j, self.downsample)
                for i, j in self.candidates
            ]
        )

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()

        return min(
            [
                _curve_pair_minimum_distance(self.curves, i, j, self.downsample)
                for i in range(len(self.curves))
                for j in range(i)
            ]
        )

    def J(self):
        """
        This returns the value of the quantity.
        """
        raise_if_target_lane_bypass("CurveCurveDistance.J")
        self.compute_candidates()
        res = _as_jax_float64(0.0)
        for i, j in self.candidates:
            gamma1, l1, gamma2, l2 = _curve_pair_jax_data(
                self.curves, i, j, self.downsample
            )
            res += cc_distance_pure(gamma1, l1, gamma2, l2, self.minimum_distance)

        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        raise_if_target_lane_bypass("CurveCurveDistance.dJ")
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs, dgammadash_by_dcoeff_vjp_vecs = _curve_vjp_buffers(
            self.curves
        )

        for i, j in self.candidates:
            gamma1, l1, gamma2, l2 = _curve_pair_jax_data(
                self.curves, i, j, self.downsample
            )
            minimum_distance = _as_jax_float64(self.minimum_distance)
            grad0, grad1, grad2, grad3 = _cc_distance_grad(
                gamma1,
                l1,
                gamma2,
                l2,
                minimum_distance,
            )
            _add_curve_vjp(
                dgamma_by_dcoeff_vjp_vecs[i],
                _as_numpy_float64(grad0),
                self.downsample,
            )
            _add_curve_vjp(
                dgammadash_by_dcoeff_vjp_vecs[i],
                _as_numpy_float64(grad1),
                self.downsample,
            )
            _add_curve_vjp(
                dgamma_by_dcoeff_vjp_vecs[j],
                _as_numpy_float64(grad2),
                self.downsample,
            )
            _add_curve_vjp(
                dgammadash_by_dcoeff_vjp_vecs[j],
                _as_numpy_float64(grad3),
                self.downsample,
            )

        return _sum_curve_vjp_contributions(
            self.curves,
            dgamma_by_dcoeff_vjp_vecs,
            dgammadash_by_dcoeff_vjp_vecs,
        )

    return_fn_map = {"J": J, "dJ": dJ}


def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    gammac = _as_jax_float64(gammac)
    lc = _as_jax_float64(lc)
    gammas = _as_jax_float64(gammas)
    ns = _as_jax_float64(ns)
    minimum_distance_jax = jnp.asarray(minimum_distance, dtype=gammac.dtype)
    zero = minimum_distance_jax - minimum_distance_jax
    one = minimum_distance_jax / minimum_distance_jax
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


class CurveSurfaceDistance(Optimizable):
    r"""
    CurveSurfaceDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} d_{i}

    where

    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. :math:`d_\min` is a desired threshold
    minimum coil-to-surface distance.  This penalty term is zero when the
    points on all coils :math:`i` and on the surface lie more than
    :math:`d_\min` away from one another.

    """

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance
        self.candidates = None
        super().__init__(
            depends_on=curves
        )  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self, curve_positions=None, surface_gamma=None):
        raise_if_target_lane_bypass("CurveSurfaceDistance.compute_candidates")
        if self.candidates is None:
            if curve_positions is None or surface_gamma is None:
                curve_positions, _, surface_gamma, _ = _curve_surface_geometry_snapshot(
                    self.curves, self.surface
                )
            if is_jax_backend():
                from ._distance_jax import get_close_candidates_between_collections

                candidates = get_close_candidates_between_collections(
                    curve_positions,
                    [surface_gamma],
                    self.minimum_distance,
                )
            else:
                candidates = (
                    sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                        curve_positions,
                        [surface_gamma],
                        self.minimum_distance,
                    )
                )
            self.candidates = candidates

    def _evaluation_geometry(self):
        curve_positions, curve_tangents, surface_gamma, surface_normals = (
            _curve_surface_geometry_snapshot(self.curves, self.surface)
        )
        self.compute_candidates(
            curve_positions=curve_positions, surface_gamma=surface_gamma
        )
        return (
            curve_positions,
            curve_tangents,
            _as_jax_float64(surface_gamma),
            _as_jax_float64(surface_normals),
        )

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist

        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min(
            [self.minimum_distance]
            + [
                np.min(cdist(self.curves[i].gamma(), xyz_surf))
                for i, _ in self.candidates
            ]
        )

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist

        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min(
            [
                np.min(cdist(self.curves[i].gamma(), xyz_surf))
                for i in range(len(self.curves))
            ]
        )

    def J(self):
        """
        This returns the value of the quantity.
        """
        raise_if_target_lane_bypass("CurveSurfaceDistance.J")
        curve_positions, curve_tangents, gammas, normals = self._evaluation_geometry()
        res = _as_jax_float64(0.0)
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for i, _ in self.candidates:
            gammac = _as_jax_float64(curve_positions[i])
            lc = _as_jax_float64(curve_tangents[i])
            res += cs_distance_pure(gammac, lc, gammas, normals, minimum_distance)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        raise_if_target_lane_bypass("CurveSurfaceDistance.dJ")
        curve_positions, curve_tangents, gammas, normals = self._evaluation_geometry()
        dgamma_by_dcoeff_vjp_vecs, dgammadash_by_dcoeff_vjp_vecs = _curve_vjp_buffers(
            self.curves
        )
        minimum_distance = _as_jax_float64(self.minimum_distance)
        for i, _ in self.candidates:
            gammac = _as_jax_float64(curve_positions[i])
            lc = _as_jax_float64(curve_tangents[i])
            grad0, grad1 = _cs_distance_grad(
                gammac,
                lc,
                gammas,
                normals,
                minimum_distance,
            )
            dgamma_by_dcoeff_vjp_vecs[i] += _as_numpy_float64(grad0)
            dgammadash_by_dcoeff_vjp_vecs[i] += _as_numpy_float64(grad1)
        return _sum_curve_vjp_contributions(
            self.curves,
            dgamma_by_dcoeff_vjp_vecs,
            dgammadash_by_dcoeff_vjp_vecs,
        )

    return_fn_map = {"J": J, "dJ": dJ}


@jit
def curve_arclengthvariation_pure(l, mat):
    """
    This function is used in a Python+Jax implementation of the curve arclength variation.
    """
    return jnp.var(mat @ l)


@jit
def _curve_arclengthvariation_grad(l, mat):
    return grad(curve_arclengthvariation_pure, argnums=0)(l, mat)


class ArclengthVariation(Optimizable):
    def __init__(self, curve, nintervals="full"):
        r"""
        This class penalizes variation of the arclength along a curve.
        The idea of this class is to avoid ill-posedness of curve objectives due to
        non-uniqueness of the underlying parametrization. Essentially we want to
        achieve constant arclength along the curve. Since we can not expect
        perfectly constant arclength along the entire curve, this class has
        some support to relax this notion. Consider a partition of the :math:`[0, 1]`
        interval into intervals :math:`\{I_i\}_{i=1}^L`, and tenote the average incremental arclength
        on interval :math:`I_i` by :math:`\ell_i`. This objective then penalises the variance

        .. math::
            J = \mathrm{Var}(\ell_i)

        it remains to choose the number of intervals :math:`L` that :math:`[0, 1]` is split into.
        If ``nintervals="full"``, then the number of intervals :math:`L` is equal to the number of quadrature
        points of the curve. If ``nintervals="partial"``, then the argument is as follows:

        A curve in 3d space is defined uniquely by an initial point, an initial
        direction, and the arclength, curvature, and torsion along the curve. For a
        :mod:`simsopt.geo.curvexyzfourier.CurveXYZFourier`, the intuition is now as
        follows: assuming that the curve has order :math:`p`, that means we have
        :math:`3*(2p+1)` degrees of freedom in total. Assuming that three each are
        required for both the initial position and direction, :math:`6p-3` are left
        over for curvature, torsion, and arclength. We want to fix the arclength,
        so we can afford :math:`2p-1` constraints, which corresponds to
        :math:`L=2p`.

        Finally, the user can also provide an integer value for `nintervals`
        and thus specify the number of intervals directly.
        """
        super().__init__(depends_on=[curve])

        assert nintervals in ["full", "partial"] or (
            isinstance(nintervals, int) and 0 < nintervals <= curve.gamma().shape[0]
        )
        self.curve = curve
        nquadpoints = len(curve.quadpoints)
        if nintervals == "full":
            nintervals = curve.gamma().shape[0]
        elif nintervals == "partial":
            from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier

            if isinstance(curve, CurveXYZFourier) or isinstance(
                curve, JaxCurveXYZFourier
            ):
                nintervals = 2 * curve.order
            else:
                raise RuntimeError(
                    "Please provide a value other than `partial` for `nintervals`. We only have a default for `CurveXYZFourier` and `JaxCurveXYZFourier`."
                )

        self.nintervals = nintervals
        indices = np.floor(
            np.linspace(0, nquadpoints, nintervals + 1, endpoint=True)
        ).astype(int)
        mat = np.zeros((nintervals, nquadpoints))
        for i in range(nintervals):
            mat[i, indices[i] : indices[i + 1]] = 1 / (indices[i + 1] - indices[i])
        self.mat = mat

    def J(self):
        return float(
            curve_arclengthvariation_pure(self.curve.incremental_arclength(), self.mat)
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            _curve_arclengthvariation_grad(
                self.curve.incremental_arclength(),
                self.mat,
            )
        )

    return_fn_map = {"J": J, "dJ": dJ}


@jit
def curve_msc_pure(kappa, gammadash):
    """
    This function is used in a Python+Jax implementation of the mean squared curvature objective.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length) / jnp.mean(arc_length)


@jit
def _curve_msc_grad(kappa, gammadash):
    return grad(curve_msc_pure, argnums=(0, 1))(kappa, gammadash)


class MeanSquaredCurvature(Optimizable):
    def __init__(self, curve):
        r"""
        Compute the mean of the squared curvature of a curve.

        .. math::
            J = (1/L) \int_{\text{curve}} \kappa^2 ~dl

        where :math:`L` is the curve length, :math:`\ell` is the incremental
        arclength, and :math:`\kappa` is the curvature.

        Args:
            curve: the curve of which the curvature should be computed.
        """
        super().__init__(depends_on=[curve])
        self.curve = curve

    def J(self):
        return float(curve_msc_pure(self.curve.kappa(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        grad0, grad1 = _curve_msc_grad(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(
            grad0
        ) + self.curve.dgammadash_by_dcoeff_vjp(grad1)


@deprecated(
    "`MinimumDistance` has been deprecated and will be removed. Please use `CurveCurveDistance` instead."
)
class MinimumDistance(CurveCurveDistance):
    pass


class LinkingNumber(Optimizable):
    def __init__(self, curves, downsample=1):
        Optimizable.__init__(self, depends_on=curves)
        self.curves = curves
        for curve in curves:
            assert np.mod(len(curve.quadpoints), downsample) == 0, (
                f"Downsample {downsample} does not divide the number of quadpoints {len(curve.quadpoints)}."
            )

        self.downsample = downsample
        self.dphis = np.array(
            [(c.quadpoints[1] - c.quadpoints[0]) * downsample for c in self.curves]
        )

        r"""
        Compute the Gauss linking number of a set of curves, i.e. whether the curves
        are interlocked or not.

        The value is an integer, >= 1 if the curves are interlocked, 0 if not. For each pair
        of curves, the contribution to the linking number is
        
        .. math::
            Link(c_1, c_2) = \frac{1}{4\pi} \left| \oint_{c_1}\oint_{c_2}\frac{\textbf{r}_1 - \textbf{r}_2}{|\textbf{r}_1 - \textbf{r}_2|^3} (d\textbf{r}_1 \times d\textbf{r}_2) \right|
            
        where :math:`c_1` is the first curve, :math:`c_2` is the second curve,
        :math:`\textbf{r}_1` is the position vector along the first curve, and
        :math:`\textbf{r}_2` is the position vector along the second curve.

        Args:
            curves: the set of curves for which the linking number should be computed.
            downsample: integer factor by which to downsample the quadrature
                points when computing the linking number. Setting this parameter to
                a value larger than 1 will speed up the calculation, which may
                be useful if the set of coils is large, though it may introduce
                inaccuracy if ``downsample`` is set too large.
        """

    def J(self):
        return sopp.compute_linking_number(
            [c.gamma() for c in self.curves],
            [c.gammadash() for c in self.curves],
            self.dphis,
            self.downsample,
        )

    @derivative_dec
    def dJ(self):
        return Derivative({})


@jit
def frametwist_pure(n1, n2, b1, b2, b1dash, n2dash):
    dot1 = b1dash[:, 0] * n2[:, 0] + b1dash[:, 1] * n2[:, 1] + b1dash[:, 2] * n2[:, 2]
    dot2 = n2dash[:, 0] * b1[:, 0] + n2dash[:, 1] * b1[:, 1] + n2dash[:, 2] * b1[:, 2]
    dot3 = n1[:, 0] * n2[:, 0] + n1[:, 1] * n2[:, 1] + n1[:, 2] * n2[:, 2]
    dot4 = n1[:, 0] * b2[:, 0] + n1[:, 1] * b2[:, 1] + n1[:, 2] * b2[:, 2]
    size = jnp.size(n1[:, 0])
    dphi = 1 / (size)
    data0 = jnp.arctan2(-dot4[0], dot3[0])
    data = jnp.zeros((size,))
    integrand = dphi * 0.5 * (dot1 + dot2) / dot3
    data = data.at[0].set(data0)

    def body_fun(i, val):
        val = val.at[i].set(val[i - 1] + (integrand[i - 1] + integrand[i]))
        return val

    data = lax.fori_loop(1, len(n1[:, 0]), body_fun, data)
    data = data.at[-1].set(data[-2] + (integrand[-1] + integrand[1]))
    return data


@jit
def frametwist_net_pure(frametwist):
    return frametwist[-1] - frametwist[0]


@jit
def frametwist_range_pure(frametwist):
    return jnp.max(frametwist) - jnp.min(frametwist)


@jit
def frametwist_max_pure(frametwist):
    return jnp.max(jnp.abs(frametwist))


@jit
def frametwist_lp_pure(frametwist, gammadash, p):
    p_jax = _as_jax_float64(p)
    one = _as_jax_float64(1.0)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (jnp.mean(frametwist**p_jax * arc_length) / jnp.mean(arc_length)) ** (
        one / p_jax
    )


@jit
def _frametwist_lp_grad(frametwist, gammadash, p):
    return grad(frametwist_lp_pure, argnums=(0, 1))(frametwist, gammadash, p)


@jit
def _frametwist_vjp(n1, n2, b1, b2, b1dash, n2dash, v):
    return vjp(frametwist_pure, n1, n2, b1, b2, b1dash, n2dash)[1](v)


class FramedCurveTwist(Optimizable):
    def __init__(self, framedcurve, f="lp", p=2):
        r"""
        Computes the maximum relative twist angle between the framedcurve
        and the centroid frame. If the frame is evaluated with respect to
        the centroid frame, the frame twist is equivalent to the rotation,
        alpha. If not, then the net rotation is evaluated by integrating
        along the curve. The frame rotation can be used within the context
        of HTS strain optimization to avoid 180-degree or 360-degree turns,
        which can be challenging to wind.

        Args:
            framedcurve: A FramedCurve from which the twist angle is evaluated

        """
        Optimizable.__init__(self, depends_on=[framedcurve])
        assert f in ["net", "range", "lp", "max"]
        self.f = f
        self.p = p
        self.framedcurve = framedcurve
        self.framedcurve_centroid = FramedCurveCentroid(framedcurve.curve)
        self.framedcurve_centroid.rotation.fix_all()

    def angle_profile(self, endpoint=False):
        """
        Returns the value of alpha
        """
        _, n1, b1 = self.framedcurve.rotated_frame()
        _, n2, b2 = self.framedcurve_centroid.rotated_frame()
        _, n1dash, b1dash = self.framedcurve.rotated_frame_dash()
        _, n2dash, b2dash = self.framedcurve_centroid.rotated_frame_dash()
        if endpoint:
            n1 = np.concatenate((n1, n1[0:1, :]))
            n2 = np.concatenate((n2, n2[0:1, :]))
            b1 = np.concatenate((b1, b1[0:1, :]))
            b2 = np.concatenate((b2, b2[0:1, :]))
            b1dash = np.concatenate((b1dash, b1dash[0:1, :]), axis=0)
            n2dash = np.concatenate((n2dash, n2dash[0:1, :]), axis=0)
        return frametwist_pure(n1, n2, b1, b2, b1dash, n2dash)

    def J(self, f=None, p=None):
        if f is None:
            f = self.f
        else:
            assert f in ["net", "range", "lp", "max"]
        data = self.angle_profile()
        if f == "net":
            return frametwist_net_pure(data)
        if f == "range":
            return frametwist_range_pure(data)
        if f == "max":
            return frametwist_max_pure(data)
        if f == "lp":
            if p is None:
                p = self.p
            gammadash = _as_jax_float64(self.framedcurve.curve.gammadash())
            return frametwist_lp_pure(
                _as_jax_float64(data),
                gammadash,
                _as_jax_float64(p),
            )
        raise Exception("incorrect wrapping function f provided")

    @derivative_dec
    def dJ(self):
        if self.f == "lp":
            data = self.angle_profile(endpoint=False)
            gammadash = _as_jax_float64(self.framedcurve.curve.gammadash())
            grad0, grad1 = _frametwist_lp_grad(
                _as_jax_float64(data),
                gammadash,
                _as_jax_float64(self.p),
            )
        else:
            return Derivative({})
        _, n1, b1 = self.framedcurve.rotated_frame()
        _, n2, b2 = self.framedcurve_centroid.rotated_frame()
        _, _, b1dash = self.framedcurve.rotated_frame_dash()
        _, n2dash, _ = self.framedcurve_centroid.rotated_frame_dash()

        vjp0, vjp1, vjp2, vjp3, vjp4, vjp5 = _frametwist_vjp(
            _as_jax_float64(n1),
            _as_jax_float64(n2),
            _as_jax_float64(b1),
            _as_jax_float64(b2),
            _as_jax_float64(b1dash),
            _as_jax_float64(n2dash),
            grad0,
        )
        vjp0 = _as_numpy_float64(vjp0)
        vjp1 = _as_numpy_float64(vjp1)
        vjp2 = _as_numpy_float64(vjp2)
        vjp3 = _as_numpy_float64(vjp3)
        vjp4 = _as_numpy_float64(vjp4)
        vjp5 = _as_numpy_float64(vjp5)
        zero = np.zeros_like(vjp0)

        grad = (
            self.framedcurve.rotated_frame_dcoeff_vjp(zero, vjp0, vjp2)
            + self.framedcurve.rotated_frame_dash_dcoeff_vjp(zero, zero, vjp4)
            + self.framedcurve_centroid.rotated_frame_dcoeff_vjp(zero, vjp1, vjp3)
            + self.framedcurve_centroid.rotated_frame_dash_dcoeff_vjp(zero, vjp5, zero)
        )
        if self.f == "lp":
            grad += self.framedcurve.curve.dgammadash_by_dcoeff_vjp(grad1)

        return grad


def max_distance_pure(g1, g2, dmax, p):
    """
    This returns 0 if all points of g1 have at least one point of g2 at a distance smaller or equal to dmax
    Otherwise, returns the sum of |g2-g1_i|-dmax where only points further than dmax are considered.
    The minimum distance between a point g1_i and g2 is obtained using the p-norm, with p < -1.
    """
    mindists = _pairwise_rowwise_pnorm_distance(g1, g2, p)

    # We now evaluate if any of mindists is larger than dmax. If yes, we add the value of (mindists[i]-dmax)**2 to the output.
    # We normalize by the number of quadrature points along the first curve g1.
    return jnp.sum(jnp.square(jnp.maximum(mindists - dmax, 0))) / g1.shape[0]


@jit
def _max_distance_grad(g1, g2, dmax, p):
    return grad(max_distance_pure, argnums=(0, 1))(g1, g2, dmax, p)


class MinCurveCurveDistance(Optimizable):
    """
    This class can be used to constrain a curve to remain close
    to another curve.
    """

    def __init__(self, curve1, curve2, maximum_distance, p=-10):
        self.curve1 = curve1
        self.curve2 = curve2
        self.maximum_distance = maximum_distance
        self.p = p

        Optimizable.__init__(self, depends_on=[curve1, curve2])

    def max_distance(self):
        """
        returns the max distance between curve1 and curve2
        """
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()
        return jnp.max(_pairwise_rowwise_min_distance(g1, g2))

    def min_dists(self):
        """
        returns the an array of the minimum distance between curve1 and curve2
        """
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()
        return _pairwise_rowwise_min_distance(g1, g2)

    def min_dists_p(self):
        """
        returns the an array of the minimum distance between curve1 and curve2 (approximated w/ p norm)
        """
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()
        return _pairwise_rowwise_pnorm_distance(g1, g2, self.p)

    def J(self):
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()

        return max_distance_pure(g1, g2, self.maximum_distance, self.p)

    @derivative_dec
    def dJ(self):
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()

        grad0, grad1 = _max_distance_grad(g1, g2, self.maximum_distance, self.p)

        return self.curve1.dgamma_by_dcoeff_vjp(
            grad0
        ) + self.curve2.dgamma_by_dcoeff_vjp(grad1)
