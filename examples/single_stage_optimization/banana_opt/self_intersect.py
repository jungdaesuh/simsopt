"""
Self-intersection penalty (``CurveSelfIntersect``).

Motivation
----------
Banana coils are parameterised as ``CurveCWSFourier`` curves on an
axisymmetric winding surface and optimised with Fourier order as a DOF.
At ``order=2`` the shape space is narrow enough that self-intersecting
("figure-8") geometry is uncommon; from ``order>=3`` onward these
topologies appear routinely and the post-hoc ``is_self_intersecting``
detector in ``banana_coil_solver.py`` flags a non-trivial fraction of
optimizer outputs as invalid. Higher order is desired for a better fit
to the plasma surface; the blocker is a differentiable penalty that
discourages self-intersection during optimisation instead of only
catching it after the fact.

This module provides two related penalties: a legacy topology guard and a
physical self-envelope distance guard. Both use a quadratic-hinge
self-distance penalty with the same mathematical structure as
:class:`simsopt.geo.curveobjectives.CurveCurveDistance`, but with a
single curve and a periodic neighbour-exclusion mask.

Mathematical form
-----------------
Given a closed curve :math:`\\gamma : [0, 1) \\to \\mathbb{R}^3`
discretised on ``N`` quadpoints, the objective is

.. math::
    J = \\frac{C}{2} \\sum_{i, j}
        M_{ij}\\,
        \\bigl\\lVert \\gamma'_i \\bigr\\rVert\\,
        \\bigl\\lVert \\gamma'_j \\bigr\\rVert\\,
        \\max\\!\\bigl(d_{\\min} - \\lVert \\gamma_i - \\gamma_j \\rVert,\\; 0\\bigr)^2,

where :math:`M_{ij} \\in \\{0, 1\\}` is a static index mask that
zeroes out the diagonal and the quadpoints within ``neighbor_skip``
steps of it, wrapping periodically. The factor ``1/2`` compensates
for symmetric double counting. The prefactor :math:`C` is selected
by the ``normalize`` kwarg: ``normalize=True`` sets
:math:`C = 1/N^2` (quadpoint-count invariant but scales the penalty
by :math:`1/N^2` at fixed threshold/violation, which at typical
``N ~ 500`` crushes the term by :math:`\\sim 10^{-6}` relative to
comparable CurveCurveDistance-style penalties), while the default
``normalize=False`` uses :math:`C = 1` so the raw pairwise sum is
returned.

Design notes
------------
* The mask is built once at construction from quadpoint indices and
  held as a constant JAX array. This keeps the JIT trace simple and
  avoids spurious gradient contributions.
* Quadpoint spacing is uniform in curve parameter ``s``, not arc
  length. At high Fourier order, a fixed ``neighbor_skip`` therefore
  corresponds to a variable arc-length window. For the banana-coil
  application this is acceptable because ``neighbor_skip`` is meant
  to suppress adjacency artifacts, not to define a physical minimum
  separation; ``minimum_distance`` carries the physical meaning.
* The ``max(.,0)^2`` hinge is :math:`C^1` at the threshold, so
  :math:`J` is a smooth function of the DOFs (this is what
  ``CurveCurveDistance`` does).
"""

import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit


def _self_distance_pure(gamma, gammadash, minimum_distance, mask, normalize):
    """Self curve-curve distance penalty, JAX-pure.

    Parameters
    ----------
    gamma : (N, 3) array
        Curve points in Cartesian coordinates.
    gammadash : (N, 3) array
        Curve tangents d(gamma)/ds.
    minimum_distance : float
        Activation threshold :math:`d_{\\min}` on the pairwise
        distance :math:`\\lVert \\gamma_i - \\gamma_j \\rVert`.
    mask : (N, N) array of {0, 1}
        Precomputed constant mask zeroing out the diagonal and the
        within-``neighbor_skip`` band (with periodic wrap).
    normalize : bool
        When ``True``, divide by :math:`N^2` (quadpoint-count invariant
        but scales the penalty down by :math:`1/N^2` at fixed
        threshold/violation). When ``False``, return the raw pairwise
        sum.
    """
    # Squared pairwise distance. The diagonal is identically 0 in exact
    # arithmetic; jnp.sqrt(0) has an infinite subgradient, and even
    # though the mask multiplies those entries by 0 in the forward pass,
    # the VJP would still propagate NaN through the 0*inf product. The
    # standard "double-where" pattern below protects the backward pass
    # on any pair where d^2 = 0 without biasing the forward value.
    dist_sq = jnp.sum((gamma[:, None, :] - gamma[None, :, :]) ** 2, axis=2)
    safe = jnp.where(dist_sq > 0.0, dist_sq, 1.0)
    dists = jnp.where(dist_sq > 0.0, jnp.sqrt(safe), 0.0)
    alen = (jnp.linalg.norm(gammadash, axis=1)[:, None]
            * jnp.linalg.norm(gammadash, axis=1)[None, :])
    viol = jnp.maximum(minimum_distance - dists, 0.0) ** 2
    # 0.5 removes the symmetric double count.
    total = 0.5 * jnp.sum(mask * alen * viol)
    if normalize:
        return total / (gamma.shape[0] ** 2)
    return total


def _periodic_arc_positions(gamma):
    segment_lengths = jnp.linalg.norm(jnp.roll(gamma, -1, axis=0) - gamma, axis=1)
    return (
        jnp.concatenate((jnp.asarray([0.0], dtype=gamma.dtype), jnp.cumsum(segment_lengths[:-1]))),
        jnp.sum(segment_lengths),
    )


def _periodic_arc_distance_mask(gamma, self_window_m):
    arc_positions, perimeter = _periodic_arc_positions(gamma)
    arc_delta = jnp.abs(arc_positions[:, None] - arc_positions[None, :])
    periodic_arc_delta = jnp.minimum(arc_delta, perimeter - arc_delta)
    return (periodic_arc_delta > self_window_m).astype(gamma.dtype)


def _self_distance_window_pure(
    gamma,
    gammadash,
    minimum_distance,
    self_window_m,
    normalize,
):
    mask = _periodic_arc_distance_mask(gamma, self_window_m)
    return _self_distance_pure(gamma, gammadash, minimum_distance, mask, normalize)


def _global_radius_values_pure(gamma, gammadash):
    speed = jnp.linalg.norm(gammadash, axis=1)
    safe_speed = jnp.where(speed > 0.0, speed, 1.0)
    tangent = gammadash / safe_speed[:, None]
    delta = gamma[None, :, :] - gamma[:, None, :]
    delta_sq = jnp.sum(delta * delta, axis=2)
    tangent_projection = jnp.sum(delta * tangent[:, None, :], axis=2)
    normal_component = delta - tangent_projection[:, :, None] * tangent[:, None, :]
    normal_sq = jnp.sum(normal_component * normal_component, axis=2)
    safe_normal = jnp.sqrt(jnp.where(normal_sq > 0.0, normal_sq, 1.0))
    radius = delta_sq / (2.0 * safe_normal)
    valid = (delta_sq > 0.0) & (normal_sq > 0.0)
    return jnp.where(valid, radius, jnp.inf)


def _global_radius_pure(gamma, gammadash, radius_floor, normalize):
    radii = _global_radius_values_pure(gamma, gammadash)
    speed = jnp.linalg.norm(gammadash, axis=1)
    weights = speed[:, None] * speed[None, :]
    violation = jnp.maximum(radius_floor - radii, 0.0) ** 2
    total = jnp.sum(weights * violation)
    if normalize:
        return total / (gamma.shape[0] ** 2)
    return total


def _periodic_arc_delta_np(gamma):
    segment_lengths = np.linalg.norm(np.roll(gamma, -1, axis=0) - gamma, axis=1)
    arc_positions = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    perimeter = float(np.sum(segment_lengths))
    arc_delta = np.abs(arc_positions[:, None] - arc_positions[None, :])
    return np.minimum(arc_delta, perimeter - arc_delta)


def _periodic_arc_distance_mask_np(gamma, self_window_m):
    return (_periodic_arc_delta_np(gamma) > self_window_m).astype(np.float64)


def doubly_critical_self_distance_np(gamma, *, min_arc_separation_m=0.005):
    """Discrete doubly-critical self-distance (Gonzalez-Maddocks distance term).

    Gonzalez & Maddocks (PNAS 96:4769) decompose the global radius of
    curvature of a closed curve as

    .. math::
        \\Delta[\\gamma] = \\min\\bigl(1/\\kappa_{\\max},\\;
                                       d_{\\mathrm{dc}}/2\\bigr),

    where :math:`d_{\\mathrm{dc}}` is the *doubly-critical* self-distance:
    the minimum pairwise distance over pairs whose connecting chord is
    perpendicular to the tangent at **both** endpoints (an interior local
    minimum of the pairwise distance in both curve parameters). The
    windowed-hinge minimum (``CurveSelfDistance.shortest_self_distance``)
    is NOT this quantity: its binding pair can sit at the arc-window edge
    with an oblique chord, in which case ``d/2`` of that pair is strictly
    below the thickness and the identity appears to fail.

    Parameters
    ----------
    gamma : (N, 3) array
        Closed-curve points (no duplicated closure point).
    min_arc_separation_m : float, optional
        Periodic arc-length guard around the diagonal. Pairs closer in
        arc than this are excluded so that discretisation noise on the
        trivially-critical diagonal valley cannot register as a
        doubly-critical pair. This is a numerical guard, not a physical
        window; it should stay tiny relative to any feature of interest.

    Returns
    -------
    (distance_m, index_i, index_j, arc_separation_m) : tuple
        Minimum doubly-critical distance with its binding pair. When the
        curve has no doubly-critical pair (e.g. a circle sampled exactly),
        returns ``(inf, -1, -1, inf)`` and the thickness identity reduces
        to the local-curvature branch.
    """
    points = np.asarray(gamma, dtype=float)
    n = points.shape[0]
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    arc_delta = _periodic_arc_delta_np(points)
    local_min = (
        (dist <= np.roll(dist, 1, axis=0))
        & (dist <= np.roll(dist, -1, axis=0))
        & (dist <= np.roll(dist, 1, axis=1))
        & (dist <= np.roll(dist, -1, axis=1))
        & (arc_delta > min_arc_separation_m)
    )
    if not np.any(local_min):
        return (np.inf, -1, -1, np.inf)
    masked = np.where(local_min, dist, np.inf)
    flat_index = int(np.argmin(masked))
    index_i, index_j = divmod(flat_index, n)
    return (
        float(dist[index_i, index_j]),
        index_i,
        index_j,
        float(arc_delta[index_i, index_j]),
    )


def _global_radius_values_np(gamma, gammadash):
    speed = np.linalg.norm(gammadash, axis=1)
    safe_speed = np.where(speed > 0.0, speed, 1.0)
    tangent = gammadash / safe_speed[:, None]
    delta = gamma[None, :, :] - gamma[:, None, :]
    delta_sq = np.sum(delta * delta, axis=2)
    tangent_projection = np.sum(delta * tangent[:, None, :], axis=2)
    normal_component = delta - tangent_projection[:, :, None] * tangent[:, None, :]
    normal_sq = np.sum(normal_component * normal_component, axis=2)
    safe_normal = np.sqrt(np.where(normal_sq > 0.0, normal_sq, 1.0))
    radius = delta_sq / (2.0 * safe_normal)
    valid = (delta_sq > 0.0) & (normal_sq > 0.0)
    return np.where(valid, radius, np.inf)


class CurveSelfIntersect(Optimizable):
    r"""
    Penalty that steers a curve away from self-intersecting ("figure-8")
    topology by penalising non-neighbouring quadpoints that come within
    ``minimum_distance`` of each other.

    The implementation mechanism is a self curve-curve distance hinge
    (the single-curve analogue of
    :class:`simsopt.geo.curveobjectives.CurveCurveDistance`), but the
    design intent is self-intersection prevention: the penalty fires
    well before a true crossing forms, giving the optimiser a smooth
    gradient pushing distant-in-parameter points apart. Without it,
    self-intersecting geometry at higher-order banana coils is caught
    only post hoc by ``banana_coil_solver.is_self_intersecting``.

    .. math::
        J = \frac{C}{2} \sum_{i, j}
            M_{ij}\,
            \lVert \gamma'_i \rVert\,
            \lVert \gamma'_j \rVert\,
            \max\!\bigl(d_{\min} - \lVert \gamma_i - \gamma_j \rVert,\; 0\bigr)^2

    with :math:`M_{ij}` the periodic neighbour-exclusion mask
    described below and :math:`C = 1/N^2` when ``normalize=True`` or
    :math:`C = 1` otherwise.

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to penalise. Must expose ``gamma()``, ``gammadash()``,
        ``dgamma_by_dcoeff_vjp`` and ``dgammadash_by_dcoeff_vjp``
        (e.g. ``CurveXYZFourier`` or ``CurveCWSFourierCPP``).
    minimum_distance : float
        Activation threshold :math:`d_{\min}`. Pairs of quadpoints
        separated by less than this distance contribute a penalty.
    neighbor_skip : int, optional
        Number of nearest-index-neighbour quadpoints to exclude on
        each side (wrapping periodically). Must satisfy
        ``0 <= neighbor_skip < N/2``. Default 3 matches
        ``banana_coil_solver.is_self_intersecting``.
    normalize : bool, optional
        When ``True``, include the :math:`1/N^2` prefactor, making the
        penalty magnitude approximately invariant to the quadpoint
        count but also shrinking it by :math:`1/N^2` relative to the
        raw pairwise sum. When ``False`` (default), return the raw
        sum, so the penalty has the same dimensional scaling as other
        SIMSOPT pairwise distance objectives (e.g.,
        ``CurveCurveDistance``, whose normalization comes from the
        integral measure, not an explicit :math:`1/N^2`).

    Notes
    -----
    * The mask is static (constructed once from quadpoint indices).
    * ``neighbor_skip`` counts indices, not arc length; with
      non-uniform arc-length spacing at higher Fourier order this
      means a variable arc-length exclusion window. The activation
      threshold ``minimum_distance`` is what carries physical
      meaning.
    """

    def __init__(self, curve, minimum_distance, neighbor_skip=3,
                 normalize=False):
        self.curve = curve
        self.minimum_distance = minimum_distance
        self.neighbor_skip = neighbor_skip
        self.normalize = normalize

        N = len(curve.quadpoints)
        if not (0 <= neighbor_skip < N // 2):
            raise ValueError(
                f"neighbor_skip={neighbor_skip} must satisfy "
                f"0 <= neighbor_skip < N/2 = {N // 2}.")
        idx = np.arange(N)
        d = np.abs(idx[:, None] - idx[None, :])
        d = np.minimum(d, N - d)                    # periodic wrap
        mask_np = (d > neighbor_skip).astype(np.float64)
        self._mask_np = mask_np                     # kept for diagnostics
        self._mask = jnp.asarray(mask_np)

        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _self_distance_pure(
            g, gd, minimum_distance, self._mask, normalize))
        self.dJ_dprimal = jit(lambda g, gd: grad(self.J_jax, argnums=(0, 1))(g, gd))

    # ── Diagnostics ────────────────────────────────────────────────────
    def shortest_self_distance(self):
        """Return the minimum pairwise distance over non-masked pairs.

        Pure NumPy; cheap to call from a driver's diagnostics CSV.
        Returns ``+inf`` if the mask is all zeros (degenerate).
        """
        g = self.curve.gamma()
        diff = g[:, None, :] - g[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        d = np.where(self._mask_np > 0, d, np.inf)
        return float(np.min(d))

    # ── Optimizable API ────────────────────────────────────────────────
    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        dJ_dgamma, dJ_dgammadash = self.dJ_dprimal(g, gd)
        return (self.curve.dgamma_by_dcoeff_vjp(
                    np.asarray(dJ_dgamma))
                + self.curve.dgammadash_by_dcoeff_vjp(
                    np.asarray(dJ_dgammadash)))

    return_fn_map = {'J': J, 'dJ': dJ}


class CurveSelfDistance(Optimizable):
    r"""
    Physical self-envelope spacing penalty for a single closed curve.

    This objective uses the same hinge as ``CurveSelfIntersect``, but its
    neighbour exclusion window is specified in live curve arc length rather
    than in quadpoint indices. It is intended for finite-build channel
    clearance: a banana hairpin with two long legs is allowed to bring
    adjacent-in-arc points close near the turn, while nonlocal leg-to-leg
    contacts below ``minimum_distance`` are penalised.

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to penalise.
    minimum_distance : float
        Activation threshold in metres.
    self_window_m : float, optional
        Periodic arc-length exclusion window in metres. Pairs whose shortest
        curve-arc separation is at or below this window are masked out.
    sampling_margin_m : float, optional
        Conservative additive buffer on the point-pair activation threshold.
        Use this when a point-sampled hinge must imply a segment-segment screen
        pass at the exported sampling density. Diagnostics still report the
        measured point-pair distance, not the buffered threshold.
    normalize : bool, optional
        Matches ``CurveSelfIntersect``.
    """

    def __init__(
        self,
        curve,
        minimum_distance,
        *,
        self_window_m=0.060,
        sampling_margin_m=0.0,
        normalize=False,
    ):
        self.curve = curve
        self.minimum_distance = minimum_distance
        self.sampling_margin_m = sampling_margin_m
        self.activation_distance = minimum_distance + sampling_margin_m
        self.self_window_m = self_window_m
        self.normalize = normalize

        super().__init__(depends_on=[curve])
        self.J_jax = jit(
            lambda g, gd: _self_distance_window_pure(
                g,
                gd,
                self.activation_distance,
                self_window_m,
                normalize,
            )
        )
        self.dJ_dprimal = jit(lambda g, gd: grad(self.J_jax, argnums=(0, 1))(g, gd))

    def shortest_self_distance(self):
        """Return the minimum pairwise distance outside the live arc window."""
        g = self.curve.gamma()
        diff = g[:, None, :] - g[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        mask = _periodic_arc_distance_mask_np(g, self.self_window_m)
        d = np.where(mask > 0, d, np.inf)
        return float(np.min(d))

    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        dJ_dgamma, dJ_dgammadash = self.dJ_dprimal(g, gd)
        return (self.curve.dgamma_by_dcoeff_vjp(
                    np.asarray(dJ_dgamma))
                + self.curve.dgammadash_by_dcoeff_vjp(
                    np.asarray(dJ_dgammadash)))

    return_fn_map = {'J': J, 'dJ': dJ}


class CurveGlobalRadiusOfCurvature(Optimizable):
    r"""
    Global-radius self-envelope objective for a single curve.

    For every ordered point pair ``(i, j)``, this objective evaluates the
    point-tangent radius

    .. math::
        \rho_{ij} =
        \frac{\|\gamma_j-\gamma_i\|^2}
             {2\,\mathrm{dist}(\gamma_j, \gamma_i + \mathbb{R} t_i)}

    and penalises ``rho_ij < radius_floor``. Unlike
    :class:`CurveSelfDistance`, no local arc window is needed: adjacent points
    limit to the local curvature radius while doubly-critical distant pairs
    limit to half the leg-to-leg distance.
    """

    def __init__(
        self,
        curve,
        radius_floor,
        *,
        normalize=False,
    ):
        self.curve = curve
        self.radius_floor = radius_floor
        self.normalize = normalize

        super().__init__(depends_on=[curve])
        self.J_jax = jit(
            lambda g, gd: _global_radius_pure(
                g,
                gd,
                radius_floor,
                normalize,
            )
        )
        self.dJ_dprimal = jit(lambda g, gd: grad(self.J_jax, argnums=(0, 1))(g, gd))

    def shortest_groc(self):
        """Return the minimum point-tangent global radius in metres."""
        radii = _global_radius_values_np(self.curve.gamma(), self.curve.gammadash())
        return float(np.min(radii))

    def shortest_self_distance(self):
        """Return the equivalent doubly-critical distance floor diagnostic."""
        return 2.0 * self.shortest_groc()

    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        dJ_dgamma, dJ_dgammadash = self.dJ_dprimal(g, gd)
        return (self.curve.dgamma_by_dcoeff_vjp(
                    np.asarray(dJ_dgamma))
                + self.curve.dgammadash_by_dcoeff_vjp(
                    np.asarray(dJ_dgammadash)))

    return_fn_map = {'J': J, 'dJ': dJ}
