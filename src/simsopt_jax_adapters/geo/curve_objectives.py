"""JAX coil-geometry penalties as drop-in native Optimizables.

Each class mirrors the native objective named by removing the ``Jax`` prefix in
:mod:`simsopt.geo.curveobjectives`: same constructor arguments, value,
dependencies, ``Derivative`` (fixed and free partials of the curve DOFs) and
``shortest_distance``. The curves evaluate their own geometry
(``gamma``/``gammadash``/``gammadashdash``) and coefficient VJPs on the host;
the geometry moves to the active JAX device and the gradients back through
explicit transfers, and the penalty with its gradient runs as one jitted JAX
program. Curvature is computed from ``gammadash`` and ``gammadashdash`` in that
program rather than by the curve. Distance penalties use the native C++
candidate search, then evaluate every point pair within each selected pair;
only selected pairs enter the point-pair distance computation, gathered into
batches padded to power-of-two lengths. Placing and differentiating the full
curve geometry still costs O(curves x quadrature points); without a candidate
no penalty program runs and no geometry is placed.
For C++ curves (and their rotated copies) these paths make no
implicit transfer; JAX-backed native curves (``JaxCurve`` subclasses) still
compute their own geometry and VJPs with implicit transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
import simsoptpp as sopp
from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt_jax.backend.jit import jit
from simsopt_jax.backend.runtime import get_backend_policy, get_runtime_jax_device
from simsopt_jax.core._math_utils import as_jax_array as _as_jax_array
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.core._math_utils import as_jax_int32 as _as_jax_int32
from simsopt_jax.core.curve_kernels import (
    curvature_p_norm_from_kappa_pure,
    curve_curve_distance_penalty_pure,
    curve_length_from_incremental_arclength_pure,
    curve_surface_distance_penalty_pure,
    kappa_pure,
    mean_squared_curvature_pure,
)
from simsopt_jax.runtime.host_boundary import host_array, host_tree

__all__ = [
    "JaxCurveCurveDistance",
    "JaxCurveLength",
    "JaxCurveSurfaceDistance",
    "JaxLpCurveCurvature",
    "JaxMeanSquaredCurvature",
]


def _host_float(value) -> float:
    return float(host_array(value, dtype=np.float64))


def _length_from_tangent(gammadash):
    return curve_length_from_incremental_arclength_pure(jnp.linalg.norm(gammadash, axis=1))


def _lp_curvature_from_geometry(gammadash, gammadashdash, p, threshold):
    return curvature_p_norm_from_kappa_pure(
        kappa_pure(gammadash, gammadashdash), gammadash, p, threshold
    )


def _mean_squared_curvature_from_geometry(gammadash, gammadashdash):
    return mean_squared_curvature_pure(kappa_pure(gammadash, gammadashdash), gammadash)


_curve_length = jit(_length_from_tangent)
_curve_length_grad = jit(jax.grad(_length_from_tangent))
_lp_curvature = jit(_lp_curvature_from_geometry)
_lp_curvature_grad = jit(jax.grad(_lp_curvature_from_geometry, argnums=(0, 1)))
_mean_squared_curvature = jit(_mean_squared_curvature_from_geometry)
_mean_squared_curvature_grad = jit(
    jax.grad(_mean_squared_curvature_from_geometry, argnums=(0, 1))
)


def _curve_position_samples(curve, downsample=1):
    gamma = curve.gamma()
    return gamma if downsample == 1 else gamma[::downsample]


def _sum_curve_vjp_contributions(curves, dgamma_vjps, dgammadash_vjps):
    return sum(
        curve.dgamma_by_dcoeff_vjp(dgamma_vjp)
        + curve.dgammadash_by_dcoeff_vjp(dgammadash_vjp)
        for curve, dgamma_vjp, dgammadash_vjp in zip(
            curves, dgamma_vjps, dgammadash_vjps
        )
    )


def _tangents(curve):
    """``gammadash`` and ``gammadashdash`` of a native curve, explicitly placed."""
    return _as_jax_float64(curve.gammadash()), _as_jax_float64(curve.gammadashdash())


def _tangent_derivative(curve, grad_gammadash, grad_gammadashdash):
    return curve.dgammadash_by_dcoeff_vjp(
        host_array(grad_gammadash, dtype=np.float64)
    ) + curve.dgammadashdash_by_dcoeff_vjp(host_array(grad_gammadashdash, dtype=np.float64))


@lru_cache(maxsize=64)
def _quadrature_classes(sample_counts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Curve indices grouped by sample count, classes and members in curve order."""
    members_by_count: dict[int, list[int]] = {}
    for index, count in enumerate(sample_counts):
        members_by_count.setdefault(count, []).append(index)
    return tuple(tuple(members) for members in members_by_count.values())


def _class_stacked_geometry(curves, class_members):
    """Return device stacks of full-grid ``gamma``/``gammadash`` per quadrature class."""

    def _stack(samples):
        return tuple(
            _as_jax_float64(
                np.stack([host_array(samples(curves[index]), dtype=np.float64) for index in members])
            )
            for members in class_members
        )

    return (
        _stack(lambda curve: curve.gamma()),
        _stack(lambda curve: curve.gammadash()),
    )


def _array_snapshot_key(values) -> tuple[tuple[int, ...], str, bytes]:
    array = host_array(values)
    return array.shape, array.dtype.str, array.tobytes()


def _distance_candidate_key(curves, minimum_distance) -> tuple[object, ...]:
    """Exact native candidate inputs, without resolving a backend or placing arrays."""
    return (
        _array_snapshot_key(minimum_distance),
        tuple(
            (_array_snapshot_key(curve.gamma()), _array_snapshot_key(curve.quadpoints))
            for curve in curves
        ),
    )


def _distance_operand_key(curves, candidates: _DistanceCandidates) -> tuple[object, ...]:
    """Invalidate placed operands when candidates, geometry, policy or default-device scope change.

    Backend dtypes remains the placement owner; recording the scope prevents
    cross-scope reuse without duplicating its device-selection rules.
    """
    return candidates.key + (
        candidates.candidates,
        get_backend_policy(), get_runtime_jax_device(), jax.config.values["jax_default_device"],
        tuple(_array_snapshot_key(curve.gammadash()) for curve in curves),
    )


def _class_geometry_derivative(curves, class_members, class_dgammas, class_dgammadashes):
    """Pull full-grid per-class geometry cotangents back to the curves' coefficient derivatives.

    Every curve is one row of exactly one class, so the rows only need unstacking.
    """
    class_dgammas, class_dgammadashes = host_tree(
        (class_dgammas, class_dgammadashes), dtype=np.float64
    )
    rows = {
        index: (dgammas[row], dgammadashes[row])
        for members, dgammas, dgammadashes in zip(
            class_members, class_dgammas, class_dgammadashes, strict=True
        )
        for row, index in enumerate(members)
    }
    return _sum_curve_vjp_contributions(
        curves,
        [rows[index][0] for index in range(len(curves))],
        [rows[index][1] for index in range(len(curves))],
    )


class JaxCurveLength(Optimizable):
    r"""JAX version of :class:`simsopt.geo.CurveLength`: the length of a curve,

    .. math::
        J = \int_{\text{curve}}~dl = \int_0^1 |\Gamma'(t)|~dt,

    where :math:`\Gamma(t)` is the curve, evaluated as the mean of :math:`|\Gamma'|` over its
    quadrature points.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
    """

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m.
        """
        return _host_float(_curve_length(_as_jax_float64(self.curve.gammadash())))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return self.curve.dgammadash_by_dcoeff_vjp(
            host_array(
                _curve_length_grad(_as_jax_float64(self.curve.gammadash())),
                dtype=np.float64,
            )
        )

    return_fn_map = {"J": J, "dJ": dJ}


class JaxLpCurveCurvature(Optimizable):
    r"""JAX version of :class:`simsopt.geo.LpCurveCurvature`: penalize curvature above a threshold,

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(\kappa - \kappa_0, 0)^p ~dl,

    where :math:`\kappa = |\Gamma' \times \Gamma''| / |\Gamma'|^3` is the curvature and
    :math:`\kappa_0` is ``threshold``. No :math:`p`-th root is taken.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
        p (float): dimensionless exponent.
        threshold (float): curvature threshold :math:`\kappa_0` in 1/m; default 0.0.
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])

    def _parameters(self):
        return _as_jax_float64(self.p), _as_jax_float64(self.threshold)

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m^(1-p).
        """
        return _host_float(_lp_curvature(*_tangents(self.curve), *self._parameters()))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return _tangent_derivative(
            self.curve, *_lp_curvature_grad(*_tangents(self.curve), *self._parameters())
        )

    return_fn_map = {"J": J, "dJ": dJ}


class JaxMeanSquaredCurvature(Optimizable):
    r"""JAX version of :class:`simsopt.geo.MeanSquaredCurvature`: the mean squared curvature,

    .. math::
        J = \frac{1}{L} \int_{\text{curve}} \kappa^2 ~dl,

    where :math:`L` is the curve length and
    :math:`\kappa = |\Gamma' \times \Gamma''| / |\Gamma'|^3` is the curvature.

    Args:
        curve (Curve): curve with geometry in m, parameterized over [0, 1).
    """

    def __init__(self, curve):
        self.curve = curve
        super().__init__(depends_on=[curve])

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in 1/m^2.
        """
        return _host_float(_mean_squared_curvature(*_tangents(self.curve)))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        return _tangent_derivative(
            self.curve, *_mean_squared_curvature_grad(*_tangents(self.curve))
        )

    return_fn_map = {"J": J, "dJ": dJ}


@dataclass(frozen=True)
class _DistanceCandidates:
    """Host-only native candidate data, independent of JAX operand placement.

    ``candidates`` is the native search's result for the state ``key``, or the
    pairs a caller assigned at that state.
    """

    key: tuple[object, ...]
    candidates: tuple[tuple[int, int], ...]


def _assigned_candidates(key: tuple[object, ...], pairs) -> _DistanceCandidates | None:
    """Native's ``candidates`` assignment at the state ``key``: None reruns the search."""
    return None if pairs is None else _DistanceCandidates(key, tuple(tuple(pair) for pair in pairs))


@dataclass(frozen=True)
class _DistanceOperands:
    """One exact-state set of placed operands for the candidates of one search.

    ``batch_classes`` is the static (hashable) part, one entry per batch in
    sorted order: the quadrature classes its rows index. Rows are dynamic and
    batch lengths are powers of two, so the compiled programs depend on the
    candidates only through which class pairs occur and each batch's
    power-of-two length: with one quadrature class, O(log(candidates)) programs.
    """

    key: tuple[object, ...]
    class_members: tuple[tuple[int, ...], ...]
    batch_classes: tuple[object, ...]
    operands: tuple[object, ...]


def _curve_pairs(num_curves: int, num_basecurves: int):
    """The native ``CurveCurveDistance`` pairs ``(i, j)``, ``j < min(i, num_basecurves)``."""
    return tuple(
        (i, j) for i in range(num_curves) for j in range(min(i, num_basecurves))
    )


def _flags(values) -> jax.Array:
    return _as_jax_array(np.asarray(values, dtype=bool), dtype=jnp.bool_)


def _class_positions(class_members) -> dict[int, tuple[int, int]]:
    """Map each curve index to its (quadrature class, row in the class stack)."""
    return {
        index: (cls, row)
        for cls, members in enumerate(class_members)
        for row, index in enumerate(members)
    }


def _padded_candidate_batches(groups):
    """Place grouped candidate rows as padded index columns and validity flags.

    A group of ``n`` rows is padded to ``2**ceil(log2(n))`` rows that repeat
    row 0 and are flagged invalid, so the penalty kernel gives them zero value
    and, for finite geometry, zero gradient. Padding evaluates up to about
    twice the ``n`` rows. ``simsoptpp`` returns candidates in no fixed order;
    callers group them sorted, so equal candidate sets give bitwise-equal sums.
    Batches follow the sorted group keys, so candidate sets with the same keys
    and padded lengths share one static key and one compiled program.

    Args:
        groups (dict): maps a sortable batch key to a non-empty list of row
            tuples of one length ``k``.

    Returns:
        tuple: ``(keys, batches)``: the sorted group keys, and per group ``k``
            int32 index arrays and one bool validity array, each of shape
            (padded length,).
    """
    keys = tuple(sorted(groups))
    batches = []
    for rows in (groups[key] for key in keys):
        padded = np.zeros((1 << (len(rows) - 1).bit_length(), len(rows[0])), dtype=np.int32)
        padded[: len(rows)] = rows
        batches.append((
            *(_as_jax_int32(column) for column in padded.T),
            _flags(np.arange(padded.shape[0]) < len(rows)),
        ))
    return keys, tuple(batches)


def _zero_curve_derivative(curves):
    """The derivative with no candidate: every curve's coefficient VJP of zero, as native."""
    return _sum_curve_vjp_contributions(
        curves,
        [np.zeros_like(curve.gamma()) for curve in curves],
        [np.zeros_like(curve.gammadash()) for curve in curves],
    )


def _curve_pair_penalty_total(
    class_gammas, class_gammadashes, batches, minimum_distance, batch_classes, downsample
):
    """Sum the curve-curve penalty over the candidate pairs, one vmap per batch.

    The class stacks hold full quadrature grids; every ``downsample``-th point
    enters the penalty, so the gradient is on the full grids. Batch ``b``
    pairs rows ``batches[b][0]`` of class ``batch_classes[b][0]`` with rows
    ``batches[b][1]`` of class ``batch_classes[b][1]``; ``batches[b][2]`` is
    false on padding. Summing batch totals reassociates native's running sum
    over pairs (about 1e-16 relative).
    """
    class_gammas = tuple(gammas[:, ::downsample] for gammas in class_gammas)
    class_gammadashes = tuple(gammadashes[:, ::downsample] for gammadashes in class_gammadashes)
    batched_kernel = jax.vmap(
        curve_curve_distance_penalty_pure, in_axes=(0, 0, 0, 0, None, 0)
    )
    total = jnp.zeros((), dtype=jnp.float64)
    for (first_class, second_class), (first_rows, second_rows, valid) in zip(
        batch_classes, batches, strict=True
    ):
        # The gather's gradient is a scatter-add: GPU dJ is bitwise reproducible
        # only under --xla_gpu_exclude_nondeterministic_ops=true.
        total = total + jnp.sum(batched_kernel(
            class_gammas[first_class][first_rows],
            class_gammadashes[first_class][first_rows],
            class_gammas[second_class][second_rows],
            class_gammadashes[second_class][second_rows],
            minimum_distance,
            valid,
        ))
    return total


_curve_pair_penalty = jit(_curve_pair_penalty_total, static_argnames=("batch_classes", "downsample"))
_curve_pair_penalty_grad = jit(
    jax.grad(_curve_pair_penalty_total, argnums=(0, 1)),
    static_argnames=("batch_classes", "downsample"),
)


class JaxCurveCurveDistance(Optimizable):
    r"""JAX version of :class:`simsopt.geo.CurveCurveDistance`: penalize close curve pairs,

    .. math::
        J = \sum_{i} \sum_{j < \min(i, N_b)} d_{i,j},
        \qquad
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j}
            \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i,

    where :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on curves :math:`i` and :math:`j`,
    :math:`d_{\min}` is ``minimum_distance`` and :math:`N_b` is ``num_basecurves``. Each double
    integral is the mean over every ``downsample``-th quadrature point of both curves,

    .. math::
        d_{i,j} \approx \frac{1}{n_i n_j} \sum_{k, l} |\Gamma_i'(t_k)| |\Gamma_j'(t_l)|
            \max(0, d_{\min} - \| \Gamma_i(t_k) - \Gamma_j(t_l) \|_2)^2,

    with :math:`n_i`, :math:`n_j` the numbers of sampled points. As in the native class, only
    pairs the native candidate search finds within :math:`d_{\min}` are evaluated; the others
    contribute zero. Only candidate pairs enter the point-pair distance computation, so its
    O(:math:`n_i n_j`) work and memory scale with the number of candidates (padded to a power
    of two per pair of quadrature counts). The placed geometry, its transfers and the gradient
    pull-back cover every curve, O(curves x quadrature points). Without a candidate the adapter
    places nothing and runs no penalty program; JAX-backed curves still evaluate their own
    geometry and VJPs. ``candidates``,
    ``compute_candidates()`` and ``shortest_distance_among_candidates()`` are native's; the
    search is cached per exact geometry and settings, so ``candidates`` is always current
    unless assigned (see ``candidates``).

    Args:
        curves (list[Curve]): all curves, including symmetry copies.
        minimum_distance (float): separation threshold :math:`d_{\min}` in m.
        num_basecurves (int or None): leading base-curve count :math:`N_b`; default None. None or
            zero selects all curves, as native does.
        downsample (int): quadrature stride for both candidate search and penalty; default 1.
            As in native, a stride that does not divide a curve's quadrature count is accepted
            and samples that curve nonuniformly.
    """

    def __init__(self, curves, minimum_distance, num_basecurves=None, downsample=1):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self.num_basecurves = num_basecurves or len(curves)
        self.downsample = downsample
        self._candidate_cache: _DistanceCandidates | None = None
        self._distance_snapshot: _DistanceOperands | None = None
        super().__init__(depends_on=curves)

    def _samples(self):
        return [_curve_position_samples(curve, self.downsample) for curve in self.curves]

    def _candidates(self, samples):
        """The native candidate pairs: ``simsoptpp``'s own search on the native samples."""
        return sopp.get_pointclouds_closer_than_threshold_within_collection(
            samples, self.minimum_distance, self.num_basecurves
        )

    def _candidate_key(self) -> tuple[object, ...]:
        return _distance_candidate_key(self.curves, self.minimum_distance) + (
            self.num_basecurves, self.downsample,
        )

    def _candidate_snapshot(self) -> _DistanceCandidates:
        key = self._candidate_key()
        cached = self._candidate_cache
        if cached is not None and cached.key == key:
            return cached
        candidates = tuple(tuple(pair) for pair in self._candidates(self._samples()))
        snapshot = _DistanceCandidates(key, candidates)
        self._candidate_cache = snapshot
        return snapshot

    def _snapshot(self) -> _DistanceOperands | None:
        """Placed operands of the candidate pairs, or None without a candidate.

        Without a candidate the operands of an earlier state are released too.
        """
        candidate_snapshot = self._candidate_snapshot()
        if not candidate_snapshot.candidates:
            self._distance_snapshot = None
            return None
        key = _distance_operand_key(self.curves, candidate_snapshot)
        cached = self._distance_snapshot
        if cached is not None and cached.key == key:
            return cached
        class_members = _quadrature_classes(tuple(int(curve.gamma().shape[0]) for curve in self.curves))
        positions = _class_positions(class_members)
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for first, second in sorted(candidate_snapshot.candidates):
            (first_class, first_row), (second_class, second_row) = positions[first], positions[second]
            groups.setdefault((first_class, second_class), []).append((first_row, second_row))
        batch_classes, batches = _padded_candidate_batches(groups)
        class_gammas, class_gammadashes = _class_stacked_geometry(self.curves, class_members)
        snapshot = _DistanceOperands(
            key, class_members, batch_classes,
            (class_gammas, class_gammadashes, batches, _as_jax_float64(self.minimum_distance)),
        )
        self._distance_snapshot = snapshot
        return snapshot

    @property
    def candidates(self):
        """The candidate pairs at the current state, searched when not yet known.

        Assigning a list of pairs makes ``J``, ``dJ`` and the distance queries use it, as
        in native, until the curve geometry or the settings change; assigning None
        reruns the search.

        Returns:
            list[tuple[int, int]]: a copy of the curve index pairs ``(i, j)`` with some
                sampled points closer than ``minimum_distance``, in no fixed order.
        """
        return list(self._candidate_snapshot().candidates)

    @candidates.setter
    def candidates(self, pairs):
        self._candidate_cache = _assigned_candidates(self._candidate_key(), pairs)

    def compute_candidates(self):
        """Run the native candidate search unless its result for the current state is cached."""
        self._candidate_snapshot()

    def shortest_distance_among_candidates(self):
        """Find the native sampled shortest distance over the candidate pairs.

        Returns:
            float: the smaller of ``minimum_distance`` and the candidate pairs' shortest
                sampled distance, in m.
        """
        samples = self._samples()
        return min([self.minimum_distance] + [
            np.min(cdist(samples[i], samples[j])) for i, j in self._candidate_snapshot().candidates
        ])

    def shortest_distance(self):
        """Find the native sampled shortest distance.

        With candidates, return the minimum over candidates and the threshold;
        without candidates, search all curve pairs.

        Returns:
            float: sampled distance in m.
        """
        if self._candidate_snapshot().candidates:
            return self.shortest_distance_among_candidates()
        samples = self._samples()
        return min(
            np.min(cdist(samples[i], samples[j])) for i, j in _curve_pairs(len(samples), len(samples))
        )

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m^4.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return 0.0
        return _host_float(_curve_pair_penalty(
            *snapshot.operands, batch_classes=snapshot.batch_classes, downsample=self.downsample,
        ))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return _zero_curve_derivative(self.curves)
        class_dgammas, class_dgammadashes = _curve_pair_penalty_grad(
            *snapshot.operands, batch_classes=snapshot.batch_classes, downsample=self.downsample,
        )
        return _class_geometry_derivative(
            self.curves, snapshot.class_members, class_dgammas, class_dgammadashes
        )

    return_fn_map = {"J": J, "dJ": dJ}


def _curve_surface_penalty_total(
    class_gammas, class_gammadashes, batches, minimum_distance, surface_gamma, surface_normal,
    batch_classes,
):
    """Sum the curve-surface penalty over the candidate curves, one vmap per batch.

    Batch ``b`` holds rows ``batches[b][0]`` of class ``batch_classes[b]``;
    ``batches[b][1]`` is false on padding.
    """
    batched_kernel = jax.vmap(
        curve_surface_distance_penalty_pure, in_axes=(0, 0, None, None, None, 0)
    )
    total = jnp.zeros((), dtype=jnp.float64)
    for cls, (rows, valid) in zip(batch_classes, batches, strict=True):
        total = total + jnp.sum(batched_kernel(
            class_gammas[cls][rows], class_gammadashes[cls][rows],
            surface_gamma, surface_normal, minimum_distance, valid,
        ))
    return total


_curve_surface_penalty = jit(_curve_surface_penalty_total, static_argnames=("batch_classes",))
_curve_surface_penalty_grad = jit(
    jax.grad(_curve_surface_penalty_total, argnums=(0, 1)), static_argnames=("batch_classes",)
)


class JaxCurveSurfaceDistance(Optimizable):
    r"""JAX version of :class:`simsopt.geo.CurveSurfaceDistance`: penalize curves near a surface,

    .. math::
        J = \sum_{i} d_{i},
        \qquad
        d_{i} = \int_{\text{curve}_i} \int_{S}
            \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds,

    where :math:`\mathbf{r}_i` and :math:`\mathbf{s}` are points on curve :math:`i` and the surface
    and :math:`d_{\min}` is ``minimum_distance``. The surface element uses the unnormalized
    normal :math:`\mathbf{N}`, :math:`ds = |\mathbf{N}|~d\varphi~d\theta`, and each integral is
    the mean over the quadrature points. As in the native class, only curves the native
    candidate search finds within :math:`d_{\min}` are evaluated: only they enter the
    point-pair distance computation, while the placed curve geometry, its transfers and the
    gradient pull-back cover every curve. Without a candidate the adapter places nothing and
    runs no penalty program; JAX-backed curves still evaluate their own geometry and VJPs.
    ``candidates``, ``compute_candidates()`` and
    ``shortest_distance_among_candidates()`` are native's; the search is cached per exact
    geometry, so ``candidates`` is always current unless assigned (see ``candidates``).

    The surface is not a dependency: only curve DOFs receive derivatives, and the surface
    geometry is read at each evaluation.

    Args:
        curves (list[Curve]): all curves to penalize.
        surface (Surface): surface with positions in m and unnormalized normals in m^2.
        minimum_distance (float): separation threshold :math:`d_{\min}` in m.
    """

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance
        self._candidate_cache: _DistanceCandidates | None = None
        self._distance_snapshot: _DistanceOperands | None = None
        super().__init__(depends_on=curves)

    def _candidates(self, surface_points):
        """The native candidate curves: ``simsoptpp``'s own search on the native samples."""
        return sopp.get_pointclouds_closer_than_threshold_between_two_collections(
            [curve.gamma() for curve in self.curves], [surface_points], self.minimum_distance
        )

    def _candidate_key(self) -> tuple[object, ...]:
        return _distance_candidate_key(self.curves, self.minimum_distance) + (
            _array_snapshot_key(self.surface.gamma()),
            _array_snapshot_key(self.surface.quadpoints_phi),
            _array_snapshot_key(self.surface.quadpoints_theta),
        )

    def _candidate_snapshot(self) -> _DistanceCandidates:
        key = self._candidate_key()
        cached = self._candidate_cache
        if cached is not None and cached.key == key:
            return cached
        surface_points = self.surface.gamma().reshape((-1, 3))
        candidates = tuple(tuple(pair) for pair in self._candidates(surface_points))
        snapshot = _DistanceCandidates(key, candidates)
        self._candidate_cache = snapshot
        return snapshot

    def _snapshot(self) -> _DistanceOperands | None:
        """Placed operands of the candidate curves, or None without a candidate.

        Without a candidate the operands of an earlier state are released too.
        """
        candidate_snapshot = self._candidate_snapshot()
        if not candidate_snapshot.candidates:
            self._distance_snapshot = None
            return None
        key = _distance_operand_key(self.curves, candidate_snapshot) + (
            _array_snapshot_key(self.surface.normal()),
        )
        cached = self._distance_snapshot
        if cached is not None and cached.key == key:
            return cached
        class_members = _quadrature_classes(tuple(int(curve.gamma().shape[0]) for curve in self.curves))
        positions = _class_positions(class_members)
        groups: dict[int, list[tuple[int]]] = {}
        for index, _ in sorted(candidate_snapshot.candidates):
            cls, row = positions[index]
            groups.setdefault(cls, []).append((row,))
        batch_classes, batches = _padded_candidate_batches(groups)
        class_gammas, class_gammadashes = _class_stacked_geometry(self.curves, class_members)
        operands = (
            class_gammas,
            class_gammadashes,
            batches,
            _as_jax_float64(self.minimum_distance),
            _as_jax_float64(self.surface.gamma().reshape((-1, 3))),
            _as_jax_float64(self.surface.normal().reshape((-1, 3))),
        )
        snapshot = _DistanceOperands(key, class_members, batch_classes, operands)
        self._distance_snapshot = snapshot
        return snapshot

    @property
    def candidates(self):
        """The candidate curves at the current state, searched when not yet known.

        Assigning a list of pairs ``(i, 0)`` makes ``J``, ``dJ`` and the distance queries
        use it, as in native, until the curve or surface geometry or the threshold
        change; assigning None reruns the search.

        Returns:
            list[tuple[int, int]]: a copy of the pairs ``(i, 0)`` of curve ``i`` with some
                sampled point closer than ``minimum_distance`` to the surface, in no fixed
                order.
        """
        return list(self._candidate_snapshot().candidates)

    @candidates.setter
    def candidates(self, pairs):
        self._candidate_cache = _assigned_candidates(self._candidate_key(), pairs)

    def compute_candidates(self):
        """Run the native candidate search unless its result for the current state is cached."""
        self._candidate_snapshot()

    def shortest_distance_among_candidates(self):
        """Find the native sampled shortest distance over the candidate curves.

        Returns:
            float: the smaller of ``minimum_distance`` and the candidate curves' shortest
                sampled distance to the surface, in m.
        """
        surface_points = self.surface.gamma().reshape((-1, 3))
        return min([self.minimum_distance] + [
            np.min(cdist(self.curves[i].gamma(), surface_points))
            for i, _ in self._candidate_snapshot().candidates
        ])

    def shortest_distance(self):
        """Find the native sampled shortest distance.

        With candidates, return the minimum over candidates and the threshold;
        without candidates, search all curves.

        Returns:
            float: sampled distance in m.
        """
        if self._candidate_snapshot().candidates:
            return self.shortest_distance_among_candidates()
        surface_points = self.surface.gamma().reshape((-1, 3))
        return min(np.min(cdist(curve.gamma(), surface_points)) for curve in self.curves)

    def J(self):
        """Evaluate the native objective formula on the active JAX device.

        Returns:
            float: objective value in m^5.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return 0.0
        return _host_float(_curve_surface_penalty(*snapshot.operands, batch_classes=snapshot.batch_classes))

    @derivative_dec
    def dJ(self):
        """Differentiate with respect to curve DOFs, accumulating shared DOFs.

        The derivative decorator selects free DOFs by default; partials=True
        retains fixed DOF partials.

        Returns:
            Array of shape (ndofs,): free-DOF gradient, or Derivative for partials=True; units are objective units per native DOF unit.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return _zero_curve_derivative(self.curves)
        class_dgammas, class_dgammadashes = _curve_surface_penalty_grad(
            *snapshot.operands, batch_classes=snapshot.batch_classes
        )
        return _class_geometry_derivative(
            self.curves, snapshot.class_members, class_dgammas, class_dgammadashes
        )

    return_fn_map = {"J": J, "dJ": dJ}
