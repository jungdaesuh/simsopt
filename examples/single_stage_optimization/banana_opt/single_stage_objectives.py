import math

import numpy as np

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.objectives import QuadraticPenalty

# Sentinel for missing optional dependencies; checked at class instantiation.
_MATH_SQRT2 = float(np.sqrt(2.0))

from alm_utils import (
    augmented_inequality_objective,
    normalize_alm_constraint_grads,
    normalize_alm_constraint_signals,
    require_positive_alm_threshold,
)
from banana_opt.hardware_contracts import (
    BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
    BANANA_WIDTH_MAX_M,
    BANANA_WIDTH_MIN_M,
    HARDWARE_KEEPOUT_MIN_DISTANCE_M,
    LCFS_CONSTRAINT_MODE_CENTERED,
    LCFS_CONSTRAINT_MODE_DEFAULT,
    LCFS_CONSTRAINT_MODE_EDGE_ENVELOPE,
    validate_lcfs_constraint_mode,
)
from banana_opt.hardware_constraint_schema import (
    ALMConstraintMetadata,
    ALM_OBJECTIVE_SCALE_FLOOR,
    alm_constraint_metadata_payload,
    build_threshold_overrides,
    get_hardware_constraint_spec,
    hardware_constraint_alm_activity_tolerance,
    hardware_constraint_alm_metadata,
    independent_banana_current_alm_constraint_name,
    is_independent_banana_current_alm_constraint_name,
    resolve_alm_scale_with_provenance,
)
from banana_opt.objective_gradients import objective_gradient as _objective_gradient
from banana_opt.poloidal_extent import (
    poloidal_extent_rad_from_objective,
    poloidal_extent_signed_constraint,
)
from banana_opt.search_evaluation import annotate_search_evaluation_finiteness
from banana_opt.single_stage_constraints import (
    single_stage_constraint_activity_tolerances,
)
from banana_opt.smooth_distance_selection import (
    pairwise_block_min,
    point_tree,
    surface_points_and_tree,
)


ALM_HARD_GEOMETRY_DUAL_SIGNALS = True


def average_surface_objectives(objectives, weights=None):
    if len(objectives) == 0:
        raise ValueError("Need at least one surface objective to average")
    if weights is None:
        weights = np.ones(len(objectives))
    if len(objectives) != len(weights):
        raise ValueError("Number of objectives and weights must match")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be positive")

    weighted_sum = None
    for weight, objective in zip(weights, objectives):
        weighted_objective = weight * objective
        weighted_sum = (
            weighted_objective
            if weighted_sum is None
            else weighted_sum + weighted_objective
        )
    return (1.0 / total_weight) * weighted_sum


class IotaShearShortfall(Optimizable):
    r"""One-sided quadratic shortfall on the magnitude of the iota spread.

    Given the axis-side and edge-side ``Iotas`` Boozer-surface terms, this
    rewards a larger magnetic shear ``|iota_edge - iota_axis|``:

    .. math::
       J = \tfrac12 \min\bigl(|\iota_{\text{edge}} - \iota_{\text{axis}}|
                              - \text{shear\_target},\ 0\bigr)^2 .

    The penalty is zero once the spread magnitude reaches ``shear_target`` and
    otherwise drives the spread magnitude up (mirroring the Stage-1
    ``max(0, SHEAR_MIN - slope)^2`` working-layer shear shortfall). Taking the
    absolute value makes the term orientation-agnostic: it rewards more shear
    whether the profile falls (CW lineage: axis iota > edge iota) or rises from
    axis to edge, so no sign assumption can silently invert the push. The
    gradient w.r.t. the coil dofs flows through each ``Iotas`` Boozer adjoint.

    This term is only meaningful with >=2 Boozer surfaces (an axis->edge
    profile); single-surface mode has no profile and must not construct it
    (see :func:`build_single_stage_shear_objective`).
    """

    def __init__(self, iota_axis_term, iota_edge_term, shear_target):
        Optimizable.__init__(
            self,
            x0=np.asarray([]),
            depends_on=[iota_axis_term, iota_edge_term],
        )
        self.iota_axis_term = iota_axis_term
        self.iota_edge_term = iota_edge_term
        self.shear_target = float(shear_target)

    def _spread(self):
        return float(self.iota_edge_term.J()) - float(self.iota_axis_term.J())

    def J(self):
        shortfall = min(abs(self._spread()) - self.shear_target, 0.0)
        return 0.5 * shortfall**2

    @derivative_dec
    def dJ(self):
        spread = self._spread()
        shortfall = min(abs(spread) - self.shear_target, 0.0)
        # d|spread|/dx = sign(spread) * d(spread)/dx; at the floor (shortfall==0)
        # the prefactor is 0, so the sign(0)=0 ambiguity never reaches dJ.
        sign = 1.0 if spread >= 0.0 else -1.0
        dspread = self.iota_edge_term.dJ(partials=True) - self.iota_axis_term.dJ(
            partials=True
        )
        return (shortfall * sign) * dspread

    return_fn_map = {"J": J, "dJ": dJ}


def build_single_stage_shear_objective(surface_iota_terms, shear_target):
    """Build the magnetic-shear shortfall term, or ``None`` when not meaningful.

    ``surface_iota_terms`` is the axis->edge-ordered list of per-surface
    ``Iotas`` objectives. Returns ``None`` (a clean no-op) when there are fewer
    than two surfaces, because a single surface has no axis-to-edge profile and
    thus no defined spread. The caller multiplies the returned term by a weight
    that defaults to 0, so the term is also absent from default-OFF runs.
    """
    if len(surface_iota_terms) < 2:
        return None
    return IotaShearShortfall(
        surface_iota_terms[0],
        surface_iota_terms[-1],
        shear_target,
    )


# Default low-order-rational cutoff for the rational-iota avoidance penalty. q<=10
# spans the operationally dangerous low-order resonances: for a fixed perturbation
# spectrum the island half-width scales like 1/q (Chirikov), and the natural
# resonant-harmonic amplitude itself decays with mode number, so q>10 islands are
# both thinner and weaker. q=10 also brackets the frontier iota target 3/10=0.30.
RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR = 10
# Default avoidance band scale (in iota units). Each rational p/q gets a Gaussian
# well of width sigma_q = sigma0/q, narrower for the thinner high-q islands. sigma0
# = 0.012 keeps the wells isolated (a >500x peak-to-midpoint contrast at q<=10) while
# giving each low-order rational a ~+/-2*sigma0/q repulsion basin; it was selected so
# the descent direction points away from the rational throughout +/-1 sigma_q (a
# larger sigma0 lets neighbouring wells overlap and locally invert that push).
RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA = 0.012


def _low_order_rationals_in_window(center, half_window, max_denominator):
    """Reduced rationals p/q (1<=q<=max_denominator) within ``center +- half_window``.

    Restricting to a window around the current iota keeps the bump sum O(window) sized
    and independent of where iota sits in [0, 1]; rationals outside the window have a
    Gaussian contribution below machine-meaningful levels (the window is many sigma).
    Returned as a sorted list of ``(p, q)`` in lowest terms so each rational is counted
    once with its true (reduced) denominator -- the denominator that sets its weight
    and width.
    """
    lower = center - half_window
    upper = center + half_window
    seen: set[tuple[int, int]] = set()
    for q in range(1, int(max_denominator) + 1):
        p_min = int(np.floor(lower * q))
        p_max = int(np.ceil(upper * q))
        for p in range(p_min, p_max + 1):
            value = p / q
            if value < lower or value > upper:
                continue
            divisor = math.gcd(p, q)
            seen.add((p // divisor, q // divisor))
    return sorted(seen)


class RationalIotaAvoidance(Optimizable):
    r"""Smooth repeller keeping the outer Boozer iota off low-order rationals.

    A near-shearless vacuum field whose outer rotational transform sits on a
    low-order rational :math:`p/q` seeds magnetic islands of half-width growing with
    the resonant flux. This term places a smooth Gaussian well on every reduced
    rational with denominator :math:`q \le Q` and sums them:

    .. math::
       J(\iota) = \sum_{p/q,\; q\le Q} \frac{1}{q^2}\,
                  \exp\!\Bigl(-\frac{(\iota - p/q)^2}{2\,(\sigma_0/q)^2}\Bigr).

    The :math:`1/q^2` weight makes lower-order (wider-island) rationals dominate and
    the :math:`\sigma_q=\sigma_0/q` width makes each well as narrow as its island is
    thin, so :math:`J` peaks ON each rational and decays to ~0 between them. Minimizing
    it therefore pushes :math:`\iota` off the nearest low-order rational; far from every
    rational the term and its gradient are ~0, so it does not fight the iota target.

    The single argument ``iota_term`` is the outer-surface :class:`Iotas` objective --
    the SAME term the existing :class:`QuadraticPenalty` ``Jiota`` uses -- so ``J()``
    reads :math:`\iota` and ``dJ()`` reuses that term's Boozer adjoint for the live
    coil-dof gradient (single source of truth for the iota derivative path). Only the
    rationals within a many-sigma window of the current :math:`\iota` are summed; the
    rest contribute below floating-point relevance.
    """

    def __init__(
        self,
        iota_term,
        *,
        max_denominator=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR,
        sigma=RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA,
    ):
        self.iota_term = iota_term
        self.max_denominator = int(max_denominator)
        self.sigma = float(sigma)
        if self.max_denominator < 1:
            raise ValueError("rational-iota avoidance requires max_denominator >= 1")
        if self.sigma <= 0.0:
            raise ValueError("rational-iota avoidance requires sigma > 0")
        # Window at 6 * sigma0 (>= 6 sigma_q for every q>=1): a Gaussian is < 1.5e-8 of
        # its peak beyond 6 sigma, so rationals outside the window are negligible.
        self._half_window = 6.0 * self.sigma
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[iota_term])

    def _bumps(self):
        """Per-rational (height, signed_distance/sigma_q^2) at the current iota.

        Returns the bump heights and the d/d(iota) prefactors so ``J`` and ``dJ`` share
        one evaluation of the (iota-dependent) rational window and exponentials.
        """
        iota = float(self.iota_term.J())
        heights = []
        d_prefactors = []
        for p, q in _low_order_rationals_in_window(
            iota, self._half_window, self.max_denominator
        ):
            center = p / q
            weight = 1.0 / (q * q)
            sigma_q = self.sigma / q
            height = weight * np.exp(-((iota - center) ** 2) / (2.0 * sigma_q**2))
            heights.append(height)
            # d/d(iota) exp(-(iota-c)^2/(2 s^2)) = exp(...) * -(iota-c)/s^2
            d_prefactors.append(-(iota - center) / (sigma_q**2))
        return heights, d_prefactors

    def J(self):
        heights, _ = self._bumps()
        return float(sum(heights))

    @derivative_dec
    def dJ(self):
        heights, d_prefactors = self._bumps()
        # dJ/d(iota) = sum_k height_k * d_prefactor_k; chain through the Iotas adjoint.
        dJ_diota = float(
            sum(height * prefactor for height, prefactor in zip(heights, d_prefactors))
        )
        return dJ_diota * self.iota_term.dJ(partials=True)

    return_fn_map = {"J": J, "dJ": dJ}


def build_single_stage_rational_iota_avoidance_objective(
    outer_iota_term,
    *,
    max_denominator=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR,
    sigma=RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA,
):
    """Build the outer-iota rational-avoidance penalty term.

    ``outer_iota_term`` is the outermost-surface ``Iotas`` objective (the same term
    feeding the iota target penalty). Returns ``None`` when ``outer_iota_term`` is
    ``None`` so callers can pass it through unconditionally; the caller additionally
    multiplies the term by a weight that defaults to 0, leaving default-OFF runs
    byte-identical.
    """
    if outer_iota_term is None:
        return None
    return RationalIotaAvoidance(
        outer_iota_term,
        max_denominator=max_denominator,
        sigma=sigma,
    )


class MagneticWellVolumeShortfall(Optimizable):
    r"""One-sided shortfall on a Boozer-surface magnetic-well proxy.

    For three axis-to-edge Boozer surfaces with labels :math:`s_0<s_1<s_2`,
    approximate the local volume derivatives

    .. math::
       V'_\mathrm{inner} = (V_1 - V_0)/(s_1-s_0), \quad
       V'_\mathrm{outer} = (V_2 - V_1)/(s_2-s_1),

    then use the normalized VMEC-well-sign proxy

    .. math::
       W = (V'_\mathrm{outer} - V'_\mathrm{inner})
           /(V'_\mathrm{outer} + V'_\mathrm{inner}).

    The sign convention matches SIMSOPT's VMEC ``WellWeighted`` description:
    negative is favorable.  The objective penalizes only values above
    ``well_target``:

    .. math::
       J = \tfrac12 \max(W - W_\mathrm{target}, 0)^2 .

    This is not a Mercier criterion.  It is a coil/Boozer objective-side proxy
    for the magnetic-well part of the A7 physics contract; full finite-beta
    Mercier still belongs to the VMEC lane.
    """

    def __init__(
        self,
        volume_inner_term,
        volume_mid_term,
        volume_outer_term,
        label_inner,
        label_mid,
        label_outer,
        well_target=0.0,
    ):
        self.volume_inner_term = volume_inner_term
        self.volume_mid_term = volume_mid_term
        self.volume_outer_term = volume_outer_term
        self.label_inner = float(label_inner)
        self.label_mid = float(label_mid)
        self.label_outer = float(label_outer)
        if not (self.label_inner < self.label_mid < self.label_outer):
            raise ValueError(
                "Magnetic-well volume proxy requires strictly increasing "
                "surface labels"
            )
        self.well_target = float(well_target)
        Optimizable.__init__(
            self,
            x0=np.asarray([]),
            depends_on=[volume_inner_term, volume_mid_term, volume_outer_term],
        )

    @property
    def _inner_delta(self):
        return self.label_mid - self.label_inner

    @property
    def _outer_delta(self):
        return self.label_outer - self.label_mid

    def _volume_values(self):
        return (
            float(self.volume_inner_term.J()),
            float(self.volume_mid_term.J()),
            float(self.volume_outer_term.J()),
        )

    def _slopes(self):
        v_inner, v_mid, v_outer = self._volume_values()
        inner_slope = (v_mid - v_inner) / self._inner_delta
        outer_slope = (v_outer - v_mid) / self._outer_delta
        return inner_slope, outer_slope

    def magnetic_well_proxy(self):
        inner_slope, outer_slope = self._slopes()
        denominator = inner_slope + outer_slope
        return (outer_slope - inner_slope) / denominator

    def J(self):
        excess = max(self.magnetic_well_proxy() - self.well_target, 0.0)
        return 0.5 * excess**2

    @derivative_dec
    def dJ(self):
        inner_slope, outer_slope = self._slopes()
        denominator = inner_slope + outer_slope
        metric = (outer_slope - inner_slope) / denominator
        excess = metric - self.well_target
        if excess <= 0.0:
            return Derivative({})

        d_inner_volume = self.volume_inner_term.dJ(partials=True)
        d_mid_volume = self.volume_mid_term.dJ(partials=True)
        d_outer_volume = self.volume_outer_term.dJ(partials=True)
        d_inner_slope = (d_mid_volume - d_inner_volume) * (1.0 / self._inner_delta)
        d_outer_slope = (d_outer_volume - d_mid_volume) * (1.0 / self._outer_delta)
        d_metric = (
            (d_outer_slope - d_inner_slope) * denominator
            - (outer_slope - inner_slope) * (d_outer_slope + d_inner_slope)
        ) * (1.0 / (denominator * denominator))
        return excess * d_metric

    return_fn_map = {"J": J, "dJ": dJ}


def build_single_stage_magnetic_well_objective(
    surface_volume_terms,
    surface_labels,
    well_target,
):
    """Build the multisurface Boozer-volume magnetic-well proxy term.

    A second derivative of enclosed volume needs at least three radial labels,
    so single-surface and two-surface modes return ``None`` cleanly.
    """
    if len(surface_volume_terms) < 3:
        return None
    if len(surface_volume_terms) != len(surface_labels):
        raise ValueError("Surface volume terms and labels must match")
    return MagneticWellVolumeShortfall(
        surface_volume_terms[0],
        surface_volume_terms[-2],
        surface_volume_terms[-1],
        surface_labels[0],
        surface_labels[-2],
        surface_labels[-1],
        well_target,
    )


class MinLGradBShortfall(Optimizable):
    r"""Coil-realizability shortfall on the minimum Kappel L_grad_B scale length.

    Computes ``min(L_grad_B)`` over the provided Cartesian surface quadrature
    points using the BiotSavart field, where::

        L_grad_B = sqrt(2) * |B| / ||dB/dX||_F

    with ``||dB/dX||_F`` the Frobenius norm of the magnetic-field Jacobian
    (``dB_by_dX``). The objective is a one-sided shortfall::

        J = max(0, floor - min(L_grad_B))

    so minimizing it rewards a minimum field scale length above ``floor`` (m).
    Larger ``L_grad_B`` means a gentler gradient → more coil-realizable.
    Calibration range: good ~0.34 m, marginal/bad ~0.13 m (HBT-EP scale;
    uncalibrated — do NOT enable by default).

    Notes:
        Gradient w.r.t. coil DOFs flows through the ``biot_savart`` dependency
        graph. Because the gradient requires ``d²B/dXdcoil`` (the Jacobian of
        the field Jacobian w.r.t. coil Fourier dofs), which is expensive to
        compute analytically, finite-difference gradient validation is
        recommended before trusting analytic-adjoint-based gradient behaviour.
        In practice this term is ON only when ``--lgradb-weight > 0``
        (default 0 = off), so it never appears in default objective graphs.

    Args:
        biot_savart: A simsopt ``BiotSavart`` field whose coils are the
            optimization variables.
        surface_points: ``(N, 3)`` float array of Cartesian quadrature points
            on the target surface (e.g. the outermost Boozer surface).
        floor: One-sided shortfall floor in metres.  The objective is zero
            when ``min(L_grad_B) >= floor``.
    """

    def __init__(self, biot_savart, surface_points, floor):
        self._biot_savart = biot_savart
        self._surface_points = np.ascontiguousarray(
            np.asarray(surface_points, dtype=float)
        )
        if self._surface_points.ndim != 2 or self._surface_points.shape[1] != 3:
            raise ValueError(
                "surface_points must have shape (N, 3), got "
                f"{self._surface_points.shape}"
            )
        self._floor = float(floor)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[biot_savart])

    def _compute_min_lgradB(self):
        """Return min(L_grad_B) on the surface points using BiotSavart."""
        bs = self._biot_savart
        bs.set_points(self._surface_points)
        abs_B = np.asarray(bs.AbsB(), dtype=float).ravel()  # (N,)
        dB_dX = np.asarray(bs.dB_by_dX(), dtype=float)     # (N, 3, 3)
        # Frobenius norm: sqrt(sum_ij (dB_i/dX_j)^2) per point
        frobenius_sq = np.einsum("...ij,...ij->...", dB_dX, dB_dX)  # (N,)
        # Avoid division by zero at degenerate points (safe: floor-shortfall is
        # still bounded, and zero-gradient points contribute infinite scale length).
        eps = np.finfo(float).eps
        safe_frobenius_sq = np.where(frobenius_sq > eps, frobenius_sq, eps)
        lgradB = _MATH_SQRT2 * abs_B / np.sqrt(safe_frobenius_sq)  # (N,)
        return float(np.min(lgradB))

    def J(self):
        return max(0.0, self._floor - self._compute_min_lgradB())

    @derivative_dec
    def dJ(self):
        # The gradient of max(0, floor - min(L_grad_B)) w.r.t. coil DOFs is
        # zero when the shortfall is inactive (min L_grad_B >= floor).  When
        # active the chain rule requires d(min L_grad_B)/d(coils), which in
        # turn depends on d²B/dXdcoil (the Jacobian of dB/dX w.r.t. coil
        # Fourier dofs) — expensive to compute analytically.  For the intended
        # use case (OFF by default, swept at weight > 0 to gauge realizability)
        # the derivative_dec zero derivative is sufficient; if a gradient-based
        # run is needed, validate the gradient with a finite-difference check
        # (Taylor-test recommended) before trusting analytic adjoint path.
        return Derivative({})

    return_fn_map = {"J": J, "dJ": dJ}


def build_total_objective(
    JnonQSRatio,
    RES_WEIGHT,
    JBoozerResidual,
    IOTAS_WEIGHT,
    Jiota,
    VOLUME_WEIGHT,
    JVolume,
    LENGTH_WEIGHT,
    JCurveLength,
    CC_WEIGHT,
    JCurveCurve,
    CS_WEIGHT,
    JCurveSurface,
    CURVATURE_WEIGHT,
    JCurvature,
    POLOIDAL_EXTENT_WEIGHT=0.0,
    JPoloidalExtent=None,
    JCurveLengthMin=None,
    JCoilWidth=None,
    WIDTH_WEIGHT=0.0,
    width_min_threshold=BANANA_WIDTH_MIN_M,
    width_max_threshold=BANANA_WIDTH_MAX_M,
    JCurveSelfIntersect=None,
    SELFINT_WEIGHT=0.0,
    JCurveHardwareKeepout=None,
    HARDWARE_KEEPOUT_WEIGHT=0.0,
    JCurveVesselEnvelopeKeepout=None,
    VESSEL_KEEPOUT_WEIGHT=0.0,
    JCurveAvailableEnvelopeReward=None,
    AVAILABLE_ENVELOPE_REWARD_WEIGHT=0.0,
    JCurveHardwareSdfFreeSpaceReward=None,
    HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT=0.0,
    JResidueObjective=None,
    JMeanSquaredCurvature=None,
    MSC_WEIGHT=0.0,
    JArclengthVariation=None,
    ARCLEN_WEIGHT=0.0,
    JLinkingNumber=None,
    LINKING_WEIGHT=0.0,
    JCoilForce=None,
    FORCE_WEIGHT=0.0,
    JShear=None,
    SHEAR_WEIGHT=0.0,
    JMagneticWell=None,
    MAGNETIC_WELL_WEIGHT=0.0,
    JRationalIotaAvoidance=None,
    RATIONAL_IOTA_AVOIDANCE_WEIGHT=0.0,
    JMinLGradB=None,
    LGRADB_WEIGHT=0.0,
    JTFCurvature=None,
    JTFCurveLength=None,
):
    objective = (
        JnonQSRatio
        + RES_WEIGHT * JBoozerResidual
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
        + CC_WEIGHT * JCurveCurve
        + CS_WEIGHT * JCurveSurface
        + CURVATURE_WEIGHT * JCurvature
    )
    if JVolume is not None:
        objective = objective + VOLUME_WEIGHT * JVolume
    if JCurveLengthMin is not None:
        objective = objective + LENGTH_WEIGHT * JCurveLengthMin
    if JPoloidalExtent is not None:
        objective = objective + POLOIDAL_EXTENT_WEIGHT * JPoloidalExtent
    if JCoilWidth is not None:
        objective = objective + WIDTH_WEIGHT * (
            QuadraticPenalty(JCoilWidth, width_min_threshold, "min")
            + QuadraticPenalty(JCoilWidth, width_max_threshold, "max")
        )
    if JCurveSelfIntersect is not None:
        objective = objective + SELFINT_WEIGHT * JCurveSelfIntersect
    if JCurveHardwareKeepout is not None:
        objective = objective + HARDWARE_KEEPOUT_WEIGHT * JCurveHardwareKeepout
    if JCurveVesselEnvelopeKeepout is not None:
        objective = (
            objective + VESSEL_KEEPOUT_WEIGHT * JCurveVesselEnvelopeKeepout
        )
    if JCurveAvailableEnvelopeReward is not None:
        objective = (
            objective
            + AVAILABLE_ENVELOPE_REWARD_WEIGHT * JCurveAvailableEnvelopeReward
        )
    if JCurveHardwareSdfFreeSpaceReward is not None:
        objective = (
            objective
            + HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
            * JCurveHardwareSdfFreeSpaceReward
        )
    if JResidueObjective is not None:
        objective = objective + JResidueObjective
    # SIMSOPT-official coil regularization/validity terms (opt-in; default-0 so
    # existing runs are byte-identical until a weight is set).
    if JMeanSquaredCurvature is not None:
        objective = objective + MSC_WEIGHT * JMeanSquaredCurvature
    if JArclengthVariation is not None:
        objective = objective + ARCLEN_WEIGHT * JArclengthVariation
    if JLinkingNumber is not None:
        objective = objective + LINKING_WEIGHT * JLinkingNumber
    if JCoilForce is not None:
        objective = objective + FORCE_WEIGHT * JCoilForce
    # min(L_grad_B) Kappel realizability shortfall (opt-in; default weight 0 and
    # a None term, so the objective graph is byte-identical until
    # --lgradb-weight > 0 is set. Calibration: good ~0.34 m / bad ~0.13 m at
    # HBT-EP scale — sweep range only, do NOT enable by default).
    if JMinLGradB is not None:
        objective = objective + LGRADB_WEIGHT * JMinLGradB
    # Magnetic-shear shortfall (opt-in; default weight 0 and a None term in
    # single-surface mode, so the objective graph is byte-identical until a
    # shear weight is set on a multisurface build). Rewards a larger
    # |iota_edge - iota_axis| spread to escape the flat-iota rational soup.
    if JShear is not None:
        objective = objective + SHEAR_WEIGHT * JShear
    if JMagneticWell is not None:
        objective = objective + MAGNETIC_WELL_WEIGHT * JMagneticWell
    # Rational-iota avoidance penalty (opt-in; default weight 0 and a None term, so the
    # objective graph is byte-identical until --single-stage-rational-iota-avoidance-weight
    # > 0 is set). Repels the outer Boozer iota from low-order rationals (q<=Q) that seed
    # islands; most relevant to the frontier lane whose iota target 3/10=0.30 is itself a
    # low-order rational.
    if JRationalIotaAvoidance is not None:
        objective = (
            objective
            + RATIONAL_IOTA_AVOIDANCE_WEIGHT * JRationalIotaAvoidance
        )
    # TF-coil buildability terms (opt-in via --free-tf-geometry; both None unless
    # the TF curve geometry is unfrozen, so the objective graph is byte-identical
    # for the default frozen-TF run). They reuse the banana HW weights so the freed
    # TF coils obey the same curvature cap and an as-shipped length ceiling, keeping
    # the newly optimizable TF geometry buildable. TF-TF and TF-surface clearances
    # ride on JCurveCurve/JCurveSurface, which already span the full objective_curves
    # list once the TF curves are added to it.
    if JTFCurvature is not None:
        objective = objective + CURVATURE_WEIGHT * JTFCurvature
    if JTFCurveLength is not None:
        objective = objective + LENGTH_WEIGHT * JTFCurveLength
    return objective


def _surface_objective_pair(diagnostic_surface_weights, nonQSs, brs):
    J_QS_obj = average_surface_objectives(nonQSs, weights=diagnostic_surface_weights)
    J_Boozer_obj = average_surface_objectives(brs, weights=diagnostic_surface_weights)
    return J_QS_obj, J_Boozer_obj


def _resolve_surface_objective_terms(
    diagnostic_surface_weights,
    nonQSs,
    brs,
    *,
    JNonQSObjective=None,
    JBoozerObjective=None,
):
    """Resolve diagnostic and descended QS/Boozer objectives.

    Production callers pass fixed global objectives, so search-ramp weights affect
    only raw diagnostics. Standalone callers without overrides use those weights
    as the objective pair for backwards-compatible helper behavior.
    """
    raw_J_QS_obj, raw_J_Boozer_obj = _surface_objective_pair(
        diagnostic_surface_weights,
        nonQSs,
        brs,
    )
    objective_J_QS_obj = raw_J_QS_obj if JNonQSObjective is None else JNonQSObjective
    objective_J_Boozer_obj = (
        raw_J_Boozer_obj if JBoozerObjective is None else JBoozerObjective
    )
    return raw_J_QS_obj, raw_J_Boozer_obj, objective_J_QS_obj, objective_J_Boozer_obj


def _positive_violation(signed_value):
    return max(0.0, float(signed_value))


def _worst_signed_constraint(results):
    return max(results, key=lambda result: float(result[0]))


def _objective_upper_bound_constraint(objective, threshold, objective_optimizable):
    if threshold is None:
        raise ValueError(
            "thresholded_physics ALM formulation requires explicit objective thresholds"
        )
    signed_value = float(objective.J()) - float(threshold)
    grad = _objective_gradient(objective, objective_optimizable)
    return signed_value, grad, _positive_violation(signed_value)


def _objective_lower_bound_constraint(objective, threshold, objective_optimizable):
    signed_value = float(threshold) - float(objective.J())
    grad = -_objective_gradient(objective, objective_optimizable)
    return signed_value, grad, _positive_violation(signed_value)


def _surface_upper_bound_constraint(
    surface, value, dvalue_by_dcoeff, threshold, objective_optimizable
):
    signed_value = float(value) - float(threshold)
    derivative = Derivative({surface: np.asarray(dvalue_by_dcoeff, dtype=float)})
    optimizable = surface if objective_optimizable is None else objective_optimizable
    grad = np.asarray(derivative(optimizable), dtype=float)
    return signed_value, grad, _positive_violation(signed_value)


def _boozer_adjoint_upper_bound_constraint(
    boozer_adjoint_objective, threshold, objective_optimizable
):
    """ALM upper-bound constraint whose gradient flows through the Boozer adjoint.

    ``boozer_adjoint_objective`` is a simsopt ``MajorRadius``-style wrapper over a
    BoozerSurface: ``J()`` returns the surface quantity (e.g. major radius, in m)
    and ``dJ(partials=True)`` returns its derivative w.r.t. the COIL dofs via the
    Boozer-surface adjoint. ``signed_value = J() - threshold`` and the violation are
    identical to the fixed-surface ``_surface_upper_bound_constraint`` path; only the
    gradient differs (nonzero over the coil dofs instead of gradient-dead). The
    BoozerSurface is reused from its last solution, so this adds no fresh Boozer solve.
    """
    signed_value = float(boozer_adjoint_objective.J()) - float(threshold)
    grad = np.asarray(
        boozer_adjoint_objective.dJ(partials=True)(objective_optimizable),
        dtype=float,
    )
    return signed_value, grad, _positive_violation(signed_value)


def _surface_linear_radius_constraint(
    surface,
    major_coeff,
    minor_coeff,
    offset,
    objective_optimizable,
):
    major_coeff_value = float(major_coeff)
    minor_coeff_value = float(minor_coeff)
    signed_value = (
        float(offset)
        + major_coeff_value * float(surface.major_radius())
        + minor_coeff_value * float(surface.minor_radius())
    )
    derivative = Derivative(
        {
            surface: (
                major_coeff_value * np.asarray(surface.dmajor_radius_by_dcoeff())
                + minor_coeff_value * np.asarray(surface.dminor_radius_by_dcoeff())
            )
        }
    )
    optimizable = surface if objective_optimizable is None else objective_optimizable
    grad = np.asarray(derivative(optimizable), dtype=float)
    return signed_value, grad, _positive_violation(signed_value)


def _boozer_adjoint_linear_radius_constraint(
    major_radius_objective,
    minor_radius_objective,
    major_coeff,
    minor_coeff,
    offset,
    objective_optimizable,
):
    major_coeff_value = float(major_coeff)
    minor_coeff_value = float(minor_coeff)
    signed_value = (
        float(offset)
        + major_coeff_value * float(major_radius_objective.J())
        + minor_coeff_value * float(minor_radius_objective.J())
    )
    major_grad = np.asarray(
        major_radius_objective.dJ(partials=True)(objective_optimizable),
        dtype=float,
    )
    minor_grad = np.asarray(
        minor_radius_objective.dJ(partials=True)(objective_optimizable),
        dtype=float,
    )
    grad = major_coeff_value * major_grad + minor_coeff_value * minor_grad
    return signed_value, grad, _positive_violation(signed_value)


def _optional_objective_value_and_gradient(
    objective,
    reference_grad,
    objective_optimizable,
):
    if objective is None:
        return 0.0, np.zeros_like(reference_grad)
    return float(objective.J()), _objective_gradient(objective, objective_optimizable)


def _optional_weighted_objective_terms(
    objective,
    weight,
    reference_grad,
    objective_optimizable,
):
    value, grad = _optional_objective_value_and_gradient(
        objective,
        reference_grad,
        objective_optimizable,
    )
    objective_weight = float(weight)
    return (
        value,
        grad,
        objective_weight,
        objective is not None and objective_weight != 0.0,
        objective_weight * value,
        objective_weight * grad,
    )


def _optional_objective_payload(objective):
    if objective is None:
        return None
    return objective.to_json_dict()


def _scalar_abs_upper_bound_constraint(optimizable, threshold, objective_optimizable):
    value = float(optimizable.get_value())
    signed_value = abs(value) - float(threshold)
    sign = 1.0 if value >= 0.0 else -1.0
    cotangent = np.array([sign], dtype=float)
    grad = np.asarray(
        optimizable.vjp(cotangent)(objective_optimizable),
        dtype=float,
    )
    return signed_value, grad, _positive_violation(signed_value)


def _banana_current_alm_metadata_name(name: str) -> str:
    if is_independent_banana_current_alm_constraint_name(name):
        return "banana_current_upper_bound"
    return name


def _hardware_alm_activity_tolerance(
    name: str,
    *,
    threshold_overrides: dict[str, float],
) -> float:
    return hardware_constraint_alm_activity_tolerance(
        _banana_current_alm_metadata_name(name),
        threshold_overrides=threshold_overrides,
    )


def _hard_min_curve_curve_signed_constraint(curves, minimum_distance):
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    curve_trees = [point_tree(points) for points in curve_points]
    hard_min = min(
        pairwise_block_min(
            gamma_i,
            curve_points[previous_index],
            right_tree=curve_trees[previous_index],
        )
        for index, gamma_i in enumerate(curve_points)
        for previous_index in range(index)
    )
    signed_value = float(minimum_distance) - float(hard_min)
    return signed_value, _positive_violation(signed_value)


def _hard_min_curve_surface_signed_constraint(curves, surface, minimum_distance):
    flat_surface, surface_tree = surface_points_and_tree(surface)
    hard_min = min(
        pairwise_block_min(
            np.asarray(curve.gamma(), dtype=float),
            flat_surface,
            right_tree=surface_tree,
        )
        for curve in curves
    )
    signed_value = float(minimum_distance) - float(hard_min)
    return signed_value, _positive_violation(signed_value)


def _hard_min_surface_stack_signed_constraint(surfaces, minimum_distance):
    surface_entries = [surface_points_and_tree(surface) for surface in surfaces]
    surface_points = [points for points, _tree in surface_entries]
    surface_trees = [tree for _points, tree in surface_entries]
    hard_min = min(
        pairwise_block_min(
            surface_points[index - 1],
            surface_points[index],
            right_tree=surface_trees[index],
        )
        for index in range(1, len(surface_points))
    )
    signed_value = float(minimum_distance) - float(hard_min)
    return signed_value, _positive_violation(signed_value)


def _hard_max_curvature_signed_constraint(curve, threshold):
    signed_value = float(np.max(np.asarray(curve.kappa(), dtype=float))) - float(
        threshold
    )
    return signed_value, _positive_violation(signed_value)


def _penalty_search_constraint_payload(
    JCurveCurve,
    JCurveSurface,
    JCurvature,
):
    constraint_terms = [
        ("coil_coil_spacing", JCurveCurve),
        ("coil_surface_spacing", JCurveSurface),
        ("max_curvature", JCurvature),
    ]
    return (
        [name for name, _ in constraint_terms],
        np.asarray(
            [max(float(objective.J()), 0.0) for _, objective in constraint_terms],
            dtype=float,
        ),
    )


def evaluate_total_objective(
    diagnostic_surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    CC_WEIGHT,
    JCurveSurface,
    CS_WEIGHT,
    JCurvature,
    CURVATURE_WEIGHT,
    JNonQSObjective=None,
    JBoozerObjective=None,
    JVolume=None,
    VOLUME_WEIGHT=0.0,
    objective_optimizable=None,
    include_diagnostics=True,
    POLOIDAL_EXTENT_WEIGHT=0.0,
    JPoloidalExtent=None,
    JCurveLengthMin=None,
    JCoilWidth=None,
    WIDTH_WEIGHT=0.0,
    width_min_threshold=BANANA_WIDTH_MIN_M,
    width_max_threshold=BANANA_WIDTH_MAX_M,
    JCurveSelfIntersect=None,
    SELFINT_WEIGHT=0.0,
    JCurveHardwareKeepout=None,
    HARDWARE_KEEPOUT_WEIGHT=0.0,
    JCurveVesselEnvelopeKeepout=None,
    VESSEL_KEEPOUT_WEIGHT=0.0,
    JCurveAvailableEnvelopeReward=None,
    AVAILABLE_ENVELOPE_REWARD_WEIGHT=0.0,
    JCurveHardwareSdfFreeSpaceReward=None,
    HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT=0.0,
    JResidueObjective=None,
    JMeanSquaredCurvature=None,
    MSC_WEIGHT=0.0,
    JArclengthVariation=None,
    ARCLEN_WEIGHT=0.0,
    JLinkingNumber=None,
    LINKING_WEIGHT=0.0,
    JCoilForce=None,
    FORCE_WEIGHT=0.0,
    JShear=None,
    SHEAR_WEIGHT=0.0,
    JMagneticWell=None,
    MAGNETIC_WELL_WEIGHT=0.0,
    JRationalIotaAvoidance=None,
    RATIONAL_IOTA_AVOIDANCE_WEIGHT=0.0,
    JMinLGradB=None,
    LGRADB_WEIGHT=0.0,
):
    (
        raw_J_QS_obj,
        raw_J_Boozer_obj,
        objective_J_QS_obj,
        objective_J_Boozer_obj,
    ) = _resolve_surface_objective_terms(
        diagnostic_surface_weights,
        nonQSs,
        brs,
        JNonQSObjective=JNonQSObjective,
        JBoozerObjective=JBoozerObjective,
    )
    total_objective = build_total_objective(
        objective_J_QS_obj,
        RES_WEIGHT,
        objective_J_Boozer_obj,
        IOTAS_WEIGHT,
        Jiota,
        VOLUME_WEIGHT,
        JVolume,
        LENGTH_WEIGHT,
        JCurveLength,
        CC_WEIGHT,
        JCurveCurve,
        CS_WEIGHT,
        JCurveSurface,
        CURVATURE_WEIGHT,
        JCurvature,
        POLOIDAL_EXTENT_WEIGHT=POLOIDAL_EXTENT_WEIGHT,
        JPoloidalExtent=JPoloidalExtent,
        JCurveLengthMin=JCurveLengthMin,
        JCoilWidth=JCoilWidth,
        WIDTH_WEIGHT=WIDTH_WEIGHT,
        width_min_threshold=width_min_threshold,
        width_max_threshold=width_max_threshold,
        JCurveSelfIntersect=JCurveSelfIntersect,
        SELFINT_WEIGHT=SELFINT_WEIGHT,
        JCurveHardwareKeepout=JCurveHardwareKeepout,
        HARDWARE_KEEPOUT_WEIGHT=HARDWARE_KEEPOUT_WEIGHT,
        JCurveVesselEnvelopeKeepout=JCurveVesselEnvelopeKeepout,
        VESSEL_KEEPOUT_WEIGHT=VESSEL_KEEPOUT_WEIGHT,
        JCurveAvailableEnvelopeReward=JCurveAvailableEnvelopeReward,
        AVAILABLE_ENVELOPE_REWARD_WEIGHT=AVAILABLE_ENVELOPE_REWARD_WEIGHT,
        JCurveHardwareSdfFreeSpaceReward=JCurveHardwareSdfFreeSpaceReward,
        HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT=(
            HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
        ),
        JResidueObjective=JResidueObjective,
        JMeanSquaredCurvature=JMeanSquaredCurvature,
        MSC_WEIGHT=MSC_WEIGHT,
        JArclengthVariation=JArclengthVariation,
        ARCLEN_WEIGHT=ARCLEN_WEIGHT,
        JLinkingNumber=JLinkingNumber,
        LINKING_WEIGHT=LINKING_WEIGHT,
        JCoilForce=JCoilForce,
        FORCE_WEIGHT=FORCE_WEIGHT,
        JShear=JShear,
        SHEAR_WEIGHT=SHEAR_WEIGHT,
        JMagneticWell=JMagneticWell,
        MAGNETIC_WELL_WEIGHT=MAGNETIC_WELL_WEIGHT,
        JRationalIotaAvoidance=JRationalIotaAvoidance,
        RATIONAL_IOTA_AVOIDANCE_WEIGHT=RATIONAL_IOTA_AVOIDANCE_WEIGHT,
        JMinLGradB=JMinLGradB,
        LGRADB_WEIGHT=LGRADB_WEIGHT,
    )
    total_grad = _objective_gradient(total_objective, objective_optimizable)
    constraint_names, constraint_values = _penalty_search_constraint_payload(
        JCurveCurve,
        JCurveSurface,
        JCurvature,
    )
    evaluation = {
        "total": float(total_objective.J()),
        "grad": total_grad,
        "surface_weights": np.asarray(diagnostic_surface_weights, dtype=float).copy(),
        "diagnostics_included": False,
        "constraint_names": constraint_names,
        "dual_update_values": constraint_values,
        "feasibility_values": constraint_values.copy(),
        "search_hardware_constraint_payload_kind": "penalty_objective",
    }
    if not include_diagnostics:
        return annotate_search_evaluation_finiteness(evaluation)

    volume_grad = (
        np.zeros_like(total_grad)
        if JVolume is None
        else _objective_gradient(JVolume, objective_optimizable)
    )
    residue_value, residue_grad = _optional_objective_value_and_gradient(
        JResidueObjective,
        total_grad,
        objective_optimizable,
    )
    (
        shear_value,
        shear_grad,
        shear_weight,
        shear_objective_enabled,
        _,
        _,
    ) = _optional_weighted_objective_terms(
        JShear,
        SHEAR_WEIGHT,
        total_grad,
        objective_optimizable,
    )
    (
        magnetic_well_value,
        magnetic_well_grad,
        magnetic_well_weight,
        magnetic_well_objective_enabled,
        _,
        _,
    ) = _optional_weighted_objective_terms(
        JMagneticWell,
        MAGNETIC_WELL_WEIGHT,
        total_grad,
        objective_optimizable,
    )
    (
        rational_iota_avoidance_value,
        rational_iota_avoidance_grad,
        rational_iota_avoidance_weight,
        rational_iota_avoidance_objective_enabled,
        _,
        _,
    ) = _optional_weighted_objective_terms(
        JRationalIotaAvoidance,
        RATIONAL_IOTA_AVOIDANCE_WEIGHT,
        total_grad,
        objective_optimizable,
    )
    evaluation.update(
        {
            "diagnostics_included": True,
            "J_QS": float(raw_J_QS_obj.J()),
            "dJ_QS": _objective_gradient(raw_J_QS_obj, objective_optimizable),
            "J_QS_objective": float(objective_J_QS_obj.J()),
            "dJ_QS_objective": _objective_gradient(
                objective_J_QS_obj,
                objective_optimizable,
            ),
            "J_Boozer": float(raw_J_Boozer_obj.J()),
            "dJ_Boozer": _objective_gradient(raw_J_Boozer_obj, objective_optimizable),
            "J_Boozer_objective": float(objective_J_Boozer_obj.J()),
            "dJ_Boozer_objective": _objective_gradient(
                objective_J_Boozer_obj,
                objective_optimizable,
            ),
            "J_iota": float(Jiota.J()),
            "dJ_iota": _objective_gradient(Jiota, objective_optimizable),
            "J_volume": 0.0 if JVolume is None else float(JVolume.J()),
            "dJ_volume": volume_grad,
            "J_len": float(JCurveLength.J()),
            "dJ_len": _objective_gradient(JCurveLength, objective_optimizable),
            "J_len_min": (
                0.0 if JCurveLengthMin is None else float(JCurveLengthMin.J())
            ),
            "dJ_len_min": (
                np.zeros_like(total_grad)
                if JCurveLengthMin is None
                else _objective_gradient(JCurveLengthMin, objective_optimizable)
            ),
            "J_cc": float(JCurveCurve.J()),
            "dJ_cc": _objective_gradient(JCurveCurve, objective_optimizable),
            "J_cs": float(JCurveSurface.J()),
            "dJ_cs": _objective_gradient(JCurveSurface, objective_optimizable),
            "J_curvature": float(JCurvature.J()),
            "dJ_curvature": _objective_gradient(JCurvature, objective_optimizable),
            "J_poloidal_extent": (
                0.0 if JPoloidalExtent is None else float(JPoloidalExtent.J())
            ),
            "dJ_poloidal_extent": (
                np.zeros_like(total_grad)
                if JPoloidalExtent is None
                else _objective_gradient(JPoloidalExtent, objective_optimizable)
            ),
            "J_coil_width": (0.0 if JCoilWidth is None else float(JCoilWidth.J())),
            "dJ_coil_width": (
                np.zeros_like(total_grad)
                if JCoilWidth is None
                else _objective_gradient(JCoilWidth, objective_optimizable)
            ),
            "J_self_intersect": (
                0.0 if JCurveSelfIntersect is None else float(JCurveSelfIntersect.J())
            ),
            "dJ_self_intersect": (
                np.zeros_like(total_grad)
                if JCurveSelfIntersect is None
                else _objective_gradient(JCurveSelfIntersect, objective_optimizable)
            ),
            "J_hardware_keepout": (
                0.0
                if JCurveHardwareKeepout is None
                else float(JCurveHardwareKeepout.J())
            ),
            "dJ_hardware_keepout": (
                np.zeros_like(total_grad)
                if JCurveHardwareKeepout is None
                else _objective_gradient(JCurveHardwareKeepout, objective_optimizable)
            ),
            "J_vessel_keepout": (
                0.0
                if JCurveVesselEnvelopeKeepout is None
                else float(JCurveVesselEnvelopeKeepout.J())
            ),
            "dJ_vessel_keepout": (
                np.zeros_like(total_grad)
                if JCurveVesselEnvelopeKeepout is None
                else _objective_gradient(
                    JCurveVesselEnvelopeKeepout,
                    objective_optimizable,
                )
            ),
            "J_available_envelope_reward": (
                0.0
                if JCurveAvailableEnvelopeReward is None
                else float(JCurveAvailableEnvelopeReward.J())
            ),
            "dJ_available_envelope_reward": (
                np.zeros_like(total_grad)
                if JCurveAvailableEnvelopeReward is None
                else _objective_gradient(
                    JCurveAvailableEnvelopeReward,
                    objective_optimizable,
                )
            ),
            "J_hardware_sdf_free_space_reward": (
                0.0
                if JCurveHardwareSdfFreeSpaceReward is None
                else float(JCurveHardwareSdfFreeSpaceReward.J())
            ),
            "dJ_hardware_sdf_free_space_reward": (
                np.zeros_like(total_grad)
                if JCurveHardwareSdfFreeSpaceReward is None
                else _objective_gradient(
                    JCurveHardwareSdfFreeSpaceReward,
                    objective_optimizable,
                )
            ),
            "J_msc": (
                0.0
                if JMeanSquaredCurvature is None
                else float(JMeanSquaredCurvature.J())
            ),
            "dJ_msc": (
                np.zeros_like(total_grad)
                if JMeanSquaredCurvature is None
                else _objective_gradient(JMeanSquaredCurvature, objective_optimizable)
            ),
            "J_arclen": (
                0.0 if JArclengthVariation is None else float(JArclengthVariation.J())
            ),
            "dJ_arclen": (
                np.zeros_like(total_grad)
                if JArclengthVariation is None
                else _objective_gradient(JArclengthVariation, objective_optimizable)
            ),
            "J_link": (0.0 if JLinkingNumber is None else float(JLinkingNumber.J())),
            "dJ_link": (
                np.zeros_like(total_grad)
                if JLinkingNumber is None
                else _objective_gradient(JLinkingNumber, objective_optimizable)
            ),
            "J_coil_force": (0.0 if JCoilForce is None else float(JCoilForce.J())),
            "dJ_coil_force": (
                np.zeros_like(total_grad)
                if JCoilForce is None
                else _objective_gradient(JCoilForce, objective_optimizable)
            ),
            # min(L_grad_B) realizability shortfall diagnostic (0.0/zeros when
            # term is None — weight-0 default runs are byte-identical).
            "J_lgradb": (0.0 if JMinLGradB is None else float(JMinLGradB.J())),
            "dJ_lgradb": (
                np.zeros_like(total_grad)
                if JMinLGradB is None
                else _objective_gradient(JMinLGradB, objective_optimizable)
            ),
            "J_shear": shear_value,
            "dJ_shear": shear_grad,
            "shear_weight": shear_weight,
            "shear_objective_enabled": shear_objective_enabled,
            "J_magnetic_well": magnetic_well_value,
            "dJ_magnetic_well": magnetic_well_grad,
            "magnetic_well_weight": magnetic_well_weight,
            "magnetic_well_objective_enabled": magnetic_well_objective_enabled,
            "J_rational_iota_avoidance": rational_iota_avoidance_value,
            "dJ_rational_iota_avoidance": rational_iota_avoidance_grad,
            "rational_iota_avoidance_weight": rational_iota_avoidance_weight,
            "rational_iota_avoidance_objective_enabled": (
                rational_iota_avoidance_objective_enabled
            ),
            "J_residue_objective": residue_value,
            "dJ_residue_objective": residue_grad,
            "residue_objective_enabled": JResidueObjective is not None,
            "residue_objective_payload": _optional_objective_payload(JResidueObjective),
            "coil_width_min_threshold": (
                None if JCoilWidth is None else float(width_min_threshold)
            ),
            "coil_width_max_threshold": (
                None if JCoilWidth is None else float(width_max_threshold)
            ),
            "self_intersect_min_distance": (
                None
                if JCurveSelfIntersect is None
                else BANANA_SELF_INTERSECT_MIN_DISTANCE_M
            ),
            "hardware_keepout_min_distance": (
                None
                if JCurveHardwareKeepout is None
                else HARDWARE_KEEPOUT_MIN_DISTANCE_M
            ),
        }
    )
    return annotate_search_evaluation_finiteness(evaluation)


def evaluate_base_objective(
    diagnostic_surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JVolume,
    VOLUME_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    *,
    objective_optimizable=None,
    alm_formulation="weighted_sum",
    _surface_pair=None,
    JNonQSObjective=None,
    JBoozerObjective=None,
    JResidueObjective=None,
    JMeanSquaredCurvature=None,
    MSC_WEIGHT=0.0,
    JArclengthVariation=None,
    ARCLEN_WEIGHT=0.0,
    JLinkingNumber=None,
    LINKING_WEIGHT=0.0,
    JCoilForce=None,
    FORCE_WEIGHT=0.0,
    JShear=None,
    SHEAR_WEIGHT=0.0,
    JMagneticWell=None,
    MAGNETIC_WELL_WEIGHT=0.0,
    JRationalIotaAvoidance=None,
    RATIONAL_IOTA_AVOIDANCE_WEIGHT=0.0,
    JCurveVesselEnvelopeKeepout=None,
    VESSEL_KEEPOUT_WEIGHT=0.0,
    JCurveAvailableEnvelopeReward=None,
    AVAILABLE_ENVELOPE_REWARD_WEIGHT=0.0,
    JCurveHardwareSdfFreeSpaceReward=None,
    HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT=0.0,
    JMinLGradB=None,
    LGRADB_WEIGHT=0.0,
    include_diagnostics=True,
):
    if _surface_pair is not None:
        raw_J_QS_obj, raw_J_Boozer_obj = _surface_pair
    else:
        raw_J_QS_obj, raw_J_Boozer_obj = _surface_objective_pair(
            diagnostic_surface_weights,
            nonQSs,
            brs,
        )
    objective_J_QS_obj = raw_J_QS_obj if JNonQSObjective is None else JNonQSObjective
    objective_J_Boozer_obj = (
        raw_J_Boozer_obj if JBoozerObjective is None else JBoozerObjective
    )
    base_objective = (
        objective_J_QS_obj
        + RES_WEIGHT * objective_J_Boozer_obj
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
    )
    if JVolume is not None:
        base_objective = base_objective + VOLUME_WEIGHT * JVolume
    # SIMSOPT-official coil regularization/validity terms (opt-in; default-None so
    # the ALM physics objective is byte-identical until a weight is set). Soft
    # penalties on the objective, NOT hard ALM constraints (no threshold semantics).
    if JMeanSquaredCurvature is not None:
        base_objective = base_objective + MSC_WEIGHT * JMeanSquaredCurvature
    if JArclengthVariation is not None:
        base_objective = base_objective + ARCLEN_WEIGHT * JArclengthVariation
    if JLinkingNumber is not None:
        base_objective = base_objective + LINKING_WEIGHT * JLinkingNumber
    if JCoilForce is not None:
        base_objective = base_objective + FORCE_WEIGHT * JCoilForce
    if JCurveVesselEnvelopeKeepout is not None:
        base_objective = (
            base_objective
            + VESSEL_KEEPOUT_WEIGHT * JCurveVesselEnvelopeKeepout
        )
    if JCurveAvailableEnvelopeReward is not None:
        base_objective = (
            base_objective
            + AVAILABLE_ENVELOPE_REWARD_WEIGHT * JCurveAvailableEnvelopeReward
        )
    if JCurveHardwareSdfFreeSpaceReward is not None:
        base_objective = (
            base_objective
            + HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
            * JCurveHardwareSdfFreeSpaceReward
        )
    # min(L_grad_B) realizability shortfall (opt-in; default-None so the ALM
    # physics objective is byte-identical until --lgradb-weight > 0 is set).
    if JMinLGradB is not None:
        base_objective = base_objective + LGRADB_WEIGHT * JMinLGradB
    base_physics_terms_total = float(base_objective.J())
    base_physics_grad = _objective_gradient(base_objective, objective_optimizable)
    (
        vessel_keepout_value,
        vessel_keepout_grad,
        vessel_keepout_weight,
        vessel_keepout_objective_enabled,
        weighted_vessel_keepout_value,
        weighted_vessel_keepout_grad,
    ) = _optional_weighted_objective_terms(
        JCurveVesselEnvelopeKeepout,
        VESSEL_KEEPOUT_WEIGHT,
        base_physics_grad,
        objective_optimizable,
    )
    (
        available_envelope_reward_value,
        available_envelope_reward_grad,
        available_envelope_reward_weight,
        available_envelope_reward_objective_enabled,
        weighted_available_envelope_reward_value,
        weighted_available_envelope_reward_grad,
    ) = _optional_weighted_objective_terms(
        JCurveAvailableEnvelopeReward,
        AVAILABLE_ENVELOPE_REWARD_WEIGHT,
        base_physics_grad,
        objective_optimizable,
    )
    (
        hardware_sdf_free_space_reward_value,
        hardware_sdf_free_space_reward_grad,
        hardware_sdf_free_space_reward_weight,
        hardware_sdf_free_space_reward_objective_enabled,
        weighted_hardware_sdf_free_space_reward_value,
        weighted_hardware_sdf_free_space_reward_grad,
    ) = _optional_weighted_objective_terms(
        JCurveHardwareSdfFreeSpaceReward,
        HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT,
        base_physics_grad,
        objective_optimizable,
    )
    (
        shear_value,
        shear_grad,
        shear_weight,
        shear_objective_enabled,
        weighted_shear_value,
        weighted_shear_grad,
    ) = _optional_weighted_objective_terms(
        JShear,
        SHEAR_WEIGHT,
        base_physics_grad,
        objective_optimizable,
    )
    (
        magnetic_well_value,
        magnetic_well_grad,
        magnetic_well_weight,
        magnetic_well_objective_enabled,
        weighted_magnetic_well_value,
        weighted_magnetic_well_grad,
    ) = _optional_weighted_objective_terms(
        JMagneticWell,
        MAGNETIC_WELL_WEIGHT,
        base_physics_grad,
        objective_optimizable,
    )
    (
        rational_iota_avoidance_value,
        rational_iota_avoidance_grad,
        rational_iota_avoidance_weight,
        rational_iota_avoidance_objective_enabled,
        weighted_rational_iota_avoidance_value,
        weighted_rational_iota_avoidance_grad,
    ) = _optional_weighted_objective_terms(
        JRationalIotaAvoidance,
        RATIONAL_IOTA_AVOIDANCE_WEIGHT,
        base_physics_grad,
        objective_optimizable,
    )
    physics_terms_total = (
        base_physics_terms_total
        + weighted_shear_value
        + weighted_magnetic_well_value
        + weighted_rational_iota_avoidance_value
    )
    physics_grad = (
        base_physics_grad
        + weighted_shear_grad
        + weighted_magnetic_well_grad
        + weighted_rational_iota_avoidance_grad
    )
    residue_value, residue_grad = _optional_objective_value_and_gradient(
        JResidueObjective,
        physics_grad,
        objective_optimizable,
    )
    if alm_formulation == "thresholded_physics":
        total = (
            residue_value
            + weighted_shear_value
            + weighted_magnetic_well_value
            + weighted_rational_iota_avoidance_value
            + weighted_vessel_keepout_value
            + weighted_available_envelope_reward_value
            + weighted_hardware_sdf_free_space_reward_value
        )
        grad = (
            residue_grad
            + weighted_shear_grad
            + weighted_magnetic_well_grad
            + weighted_rational_iota_avoidance_grad
            + weighted_vessel_keepout_grad
            + weighted_available_envelope_reward_grad
            + weighted_hardware_sdf_free_space_reward_grad
        )
    elif alm_formulation == "weighted_sum":
        total = physics_terms_total + residue_value
        grad = physics_grad + residue_grad
    else:
        raise ValueError(f"Unsupported ALM formulation {alm_formulation!r}")
    physics_total = total if JResidueObjective is not None else physics_terms_total
    evaluation = {
        "total": total,
        "grad": grad,
        "physics_total": physics_total,
        "physics_terms_total": physics_terms_total,
        "surface_weights": np.asarray(diagnostic_surface_weights, dtype=float).copy(),
        "diagnostics_included": False,
    }
    if not include_diagnostics:
        return annotate_search_evaluation_finiteness(evaluation)

    volume_grad = (
        np.zeros_like(physics_grad)
        if JVolume is None
        else _objective_gradient(JVolume, objective_optimizable)
    )
    evaluation.update(
        {
            "diagnostics_included": True,
            "J_QS": float(raw_J_QS_obj.J()),
            "dJ_QS": _objective_gradient(raw_J_QS_obj, objective_optimizable),
            "J_QS_objective": float(objective_J_QS_obj.J()),
            "dJ_QS_objective": _objective_gradient(
                objective_J_QS_obj, objective_optimizable
            ),
            "J_Boozer": float(raw_J_Boozer_obj.J()),
            "dJ_Boozer": _objective_gradient(raw_J_Boozer_obj, objective_optimizable),
            "J_Boozer_objective": float(objective_J_Boozer_obj.J()),
            "dJ_Boozer_objective": _objective_gradient(
                objective_J_Boozer_obj, objective_optimizable
            ),
            "J_iota": float(Jiota.J()),
            "dJ_iota": _objective_gradient(Jiota, objective_optimizable),
            "J_volume": 0.0 if JVolume is None else float(JVolume.J()),
            "dJ_volume": volume_grad,
            "J_len": float(JCurveLength.J()),
            "dJ_len": _objective_gradient(JCurveLength, objective_optimizable),
            "J_shear": shear_value,
            "dJ_shear": shear_grad,
            "shear_weight": shear_weight,
            "shear_objective_enabled": shear_objective_enabled,
            "J_magnetic_well": magnetic_well_value,
            "dJ_magnetic_well": magnetic_well_grad,
            "magnetic_well_weight": magnetic_well_weight,
            "magnetic_well_objective_enabled": magnetic_well_objective_enabled,
            "J_rational_iota_avoidance": rational_iota_avoidance_value,
            "dJ_rational_iota_avoidance": rational_iota_avoidance_grad,
            "rational_iota_avoidance_weight": rational_iota_avoidance_weight,
            "rational_iota_avoidance_objective_enabled": (
                rational_iota_avoidance_objective_enabled
            ),
            "J_vessel_keepout": vessel_keepout_value,
            "dJ_vessel_keepout": vessel_keepout_grad,
            "vessel_keepout_weight": vessel_keepout_weight,
            "vessel_keepout_objective_enabled": vessel_keepout_objective_enabled,
            "J_available_envelope_reward": available_envelope_reward_value,
            "dJ_available_envelope_reward": available_envelope_reward_grad,
            "available_envelope_reward_weight": available_envelope_reward_weight,
            "available_envelope_reward_objective_enabled": (
                available_envelope_reward_objective_enabled
            ),
            "J_hardware_sdf_free_space_reward": (
                hardware_sdf_free_space_reward_value
            ),
            "dJ_hardware_sdf_free_space_reward": (
                hardware_sdf_free_space_reward_grad
            ),
            "hardware_sdf_free_space_reward_weight": (
                hardware_sdf_free_space_reward_weight
            ),
            "hardware_sdf_free_space_reward_objective_enabled": (
                hardware_sdf_free_space_reward_objective_enabled
            ),
            "J_residue_objective": residue_value,
            "dJ_residue_objective": residue_grad,
            "residue_objective_enabled": JResidueObjective is not None,
            "residue_objective_payload": _optional_objective_payload(JResidueObjective),
        }
    )
    return annotate_search_evaluation_finiteness(evaluation)


def _single_stage_hardware_threshold_overrides(
    *,
    curve_curve_min_distance,
    curve_surface_min_distance,
    curvature_threshold,
    surface_stack_min_distance=0.0,
    coil_length_threshold=None,
    coil_length_min_threshold=None,
    banana_current_threshold=None,
    poloidal_extent_threshold=None,
    width_min_threshold=None,
    width_max_threshold=None,
    lcfs_major_radius_threshold=None,
    lcfs_minor_radius_threshold=None,
    lcfs_outboard_edge_threshold=None,
    lcfs_inboard_edge_threshold=None,
) -> dict[str, float]:
    return build_threshold_overrides(
        (
            ("coil_coil_spacing", curve_curve_min_distance),
            ("coil_surface_spacing", curve_surface_min_distance),
            ("max_curvature", curvature_threshold),
            ("surface_surface_spacing", surface_stack_min_distance),
            ("coil_length", coil_length_threshold),
            ("coil_length_min", coil_length_min_threshold),
            ("banana_current", banana_current_threshold),
            ("poloidal_extent", poloidal_extent_threshold),
            ("width_min", width_min_threshold),
            ("width_max", width_max_threshold),
            ("lcfs_major_radius", lcfs_major_radius_threshold),
            ("lcfs_outboard_edge", lcfs_outboard_edge_threshold),
            ("lcfs_inboard_edge", lcfs_inboard_edge_threshold),
            ("lcfs_minor_radius", lcfs_minor_radius_threshold),
        )
    )


def _physics_alm_metadata(
    name: str,
    *,
    threshold: float,
    activity_tolerance: float,
) -> ALMConstraintMetadata:
    raw_threshold = require_positive_alm_threshold(f"threshold:{name}", threshold)
    scale, scale_floor_applied, source = resolve_alm_scale_with_provenance(
        raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR, f"threshold:{name}"
    )
    return ALMConstraintMetadata(
        scale=scale,
        block="physics",
        activity_tolerance=float(activity_tolerance),
        raw_threshold=raw_threshold,
        source=source,
        objective_value_kind="raw_physics",
        gradient_value_kind="raw_physics",
        dual_update_value_kind="hard",
        feasibility_value_kind="hard",
        certification_value_kind="hard",
        scale_floor_applied=scale_floor_applied,
    )


def _single_stage_alm_constraint_metadata(
    constraint_names: list[str],
    *,
    threshold_overrides: dict[str, float],
    activity_tolerance_by_name: dict[str, float],
    use_hard_geometry_signals: bool,
    qs_threshold,
    boozer_threshold,
    iota_penalty_threshold,
    length_penalty_threshold,
) -> dict[str, ALMConstraintMetadata]:
    metadata_by_name: dict[str, ALMConstraintMetadata] = {}
    exact_hardware_names = {
        "coil_length_min",
        "coil_length_upper_bound",
        "banana_current_upper_bound",
        "width_min",
        "width_max",
        "self_intersect",
        "lcfs_major_radius",
        "lcfs_outboard_edge",
        "lcfs_inboard_edge",
        "lcfs_minor_radius",
    }
    physics_threshold_by_name = {
        "qs_error": qs_threshold,
        "boozer_residual": boozer_threshold,
        "iota_penalty": iota_penalty_threshold,
        "length_penalty": length_penalty_threshold,
    }
    exact_geometry_names = {
        "coil_coil_spacing",
        "coil_surface_spacing",
        "surface_surface_spacing",
        "max_curvature",
        "poloidal_extent",
        "width_min",
        "width_max",
        "self_intersect",
        "lcfs_major_radius",
        "lcfs_outboard_edge",
        "lcfs_inboard_edge",
        "lcfs_minor_radius",
    }
    for constraint_name in constraint_names:
        activity_tolerance = activity_tolerance_by_name.get(constraint_name)
        if constraint_name in physics_threshold_by_name:
            metadata_by_name[constraint_name] = _physics_alm_metadata(
                constraint_name,
                threshold=physics_threshold_by_name[constraint_name],
                activity_tolerance=(
                    0.0 if activity_tolerance is None else activity_tolerance
                ),
            )
            continue
        metadata_name = _banana_current_alm_metadata_name(constraint_name)
        uses_hard_value = (
            constraint_name in exact_hardware_names
            or is_independent_banana_current_alm_constraint_name(constraint_name)
        )
        uses_hard_signal = uses_hard_value or (
            use_hard_geometry_signals and constraint_name in exact_geometry_names
        )
        metadata_by_name[constraint_name] = hardware_constraint_alm_metadata(
            metadata_name,
            threshold_overrides=threshold_overrides,
            activity_tolerance=activity_tolerance,
            objective_value_kind="hard" if uses_hard_value else "surrogate",
            gradient_value_kind="hard" if uses_hard_value else "surrogate",
            dual_update_value_kind="hard" if uses_hard_signal else "surrogate",
            feasibility_value_kind="hard" if uses_hard_signal else "surrogate",
        )
    return metadata_by_name


def _evaluate_constraint_with_optional_hard_signal(
    constraint_fn,
    hard_signal_fn,
    use_hard_signal,
    *args,
    **kwargs,
):
    if use_hard_signal and hard_signal_fn is not None:
        return hard_signal_fn(*args, **kwargs)
    signed_value, grad, violation = constraint_fn(*args, **kwargs)
    return signed_value, grad, violation, None, None


def _resolve_hard_signal(
    cached_signed_value,
    cached_violation,
    hard_signal_fn,
    *args,
    **kwargs,
):
    if cached_signed_value is not None:
        return cached_signed_value, cached_violation
    return hard_signal_fn(*args, **kwargs)


def evaluate_alm_objective(
    diagnostic_surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JVolume,
    VOLUME_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    JCurveSurface,
    JCurvature,
    multipliers,
    penalty,
    *,
    objective_optimizable,
    curves,
    curve_curve_min_distance,
    outer_surface,
    curve_surface_min_distance,
    banana_curve,
    curvature_threshold,
    distance_smoothing,
    curvature_smoothing,
    banana_curves=None,
    constraint_names,
    curve_curve_constraint_fn,
    curve_surface_constraint_fn,
    curvature_constraint_fn,
    curve_curve_constraint_with_hard_signal_fn=None,
    curve_surface_constraint_with_hard_signal_fn=None,
    surface_stack_surfaces=None,
    surface_stack_boozer_surfaces=None,
    surface_stack_min_distance=0.0,
    surface_stack_constraint_fn=None,
    surface_stack_constraint_with_hard_signal_fn=None,
    hard_surrogate_diagnostics=False,
    augmented_inequality_objective_fn=augmented_inequality_objective,
    activity_tolerances_fn=single_stage_constraint_activity_tolerances,
    alm_formulation="weighted_sum",
    qs_threshold=None,
    boozer_threshold=None,
    iota_penalty_threshold=None,
    length_penalty_threshold=0.0,
    coil_length_objective=None,
    coil_length_objectives=None,
    coil_length_threshold=None,
    coil_length_min_threshold=None,
    banana_current=None,
    banana_currents=None,
    banana_current_threshold=None,
    JPoloidalExtent=None,
    JPoloidalExtentTerms=None,
    poloidal_extent_threshold=None,
    poloidal_extent_smoothing=None,
    poloidal_extent_constraint_fn=None,
    poloidal_extent_constraint_with_hard_signal_fn=None,
    JCoilWidth=None,
    JCoilWidthTerms=None,
    width_min_threshold=None,
    width_max_threshold=None,
    JCurveSelfIntersect=None,
    JCurveSelfIntersectTerms=None,
    JCurveHardwareKeepout=None,
    JCurveVesselEnvelopeKeepout=None,
    VESSEL_KEEPOUT_WEIGHT=0.0,
    JCurveAvailableEnvelopeReward=None,
    AVAILABLE_ENVELOPE_REWARD_WEIGHT=0.0,
    JCurveHardwareSdfFreeSpaceReward=None,
    HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT=0.0,
    lcfs_surface=None,
    JLCFSMajorRadius=None,
    JLCFSMinorRadius=None,
    lcfs_constraint_mode=LCFS_CONSTRAINT_MODE_DEFAULT,
    lcfs_major_radius_threshold=None,
    lcfs_minor_radius_threshold=None,
    lcfs_outboard_edge_threshold=None,
    lcfs_inboard_edge_threshold=None,
    JNonQSObjective=None,
    JBoozerObjective=None,
    JResidueObjective=None,
    JMeanSquaredCurvature=None,
    MSC_WEIGHT=0.0,
    JArclengthVariation=None,
    ARCLEN_WEIGHT=0.0,
    JLinkingNumber=None,
    LINKING_WEIGHT=0.0,
    JCoilForce=None,
    FORCE_WEIGHT=0.0,
    JShear=None,
    SHEAR_WEIGHT=0.0,
    JMagneticWell=None,
    MAGNETIC_WELL_WEIGHT=0.0,
    JRationalIotaAvoidance=None,
    RATIONAL_IOTA_AVOIDANCE_WEIGHT=0.0,
    JMinLGradB=None,
    LGRADB_WEIGHT=0.0,
    include_diagnostics=True,
):
    raw_surface_pair = _surface_objective_pair(diagnostic_surface_weights, nonQSs, brs)
    raw_J_QS_obj, raw_J_Boozer_obj = raw_surface_pair
    base_eval = evaluate_base_objective(
        diagnostic_surface_weights,
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JVolume,
        VOLUME_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
        objective_optimizable=objective_optimizable,
        alm_formulation=alm_formulation,
        _surface_pair=raw_surface_pair,
        JNonQSObjective=JNonQSObjective,
        JBoozerObjective=JBoozerObjective,
        JResidueObjective=JResidueObjective,
        JMeanSquaredCurvature=JMeanSquaredCurvature,
        MSC_WEIGHT=MSC_WEIGHT,
        JArclengthVariation=JArclengthVariation,
        ARCLEN_WEIGHT=ARCLEN_WEIGHT,
        JLinkingNumber=JLinkingNumber,
        LINKING_WEIGHT=LINKING_WEIGHT,
        JCoilForce=JCoilForce,
        FORCE_WEIGHT=FORCE_WEIGHT,
        JShear=JShear,
        SHEAR_WEIGHT=SHEAR_WEIGHT,
        JMagneticWell=JMagneticWell,
        MAGNETIC_WELL_WEIGHT=MAGNETIC_WELL_WEIGHT,
        JRationalIotaAvoidance=JRationalIotaAvoidance,
        RATIONAL_IOTA_AVOIDANCE_WEIGHT=RATIONAL_IOTA_AVOIDANCE_WEIGHT,
        JCurveVesselEnvelopeKeepout=JCurveVesselEnvelopeKeepout,
        VESSEL_KEEPOUT_WEIGHT=VESSEL_KEEPOUT_WEIGHT,
        JCurveAvailableEnvelopeReward=JCurveAvailableEnvelopeReward,
        AVAILABLE_ENVELOPE_REWARD_WEIGHT=AVAILABLE_ENVELOPE_REWARD_WEIGHT,
        JCurveHardwareSdfFreeSpaceReward=JCurveHardwareSdfFreeSpaceReward,
        HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT=(
            HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
        ),
        JMinLGradB=JMinLGradB,
        LGRADB_WEIGHT=LGRADB_WEIGHT,
        include_diagnostics=include_diagnostics,
    )

    (
        curve_curve_signed_value,
        curve_curve_grad,
        curve_curve_violation,
        curve_curve_hard_signed_value,
        curve_curve_hard_violation,
    ) = _evaluate_constraint_with_optional_hard_signal(
        curve_curve_constraint_fn,
        curve_curve_constraint_with_hard_signal_fn,
        hard_surrogate_diagnostics,
        curves,
        curve_curve_min_distance,
        distance_smoothing,
        objective_optimizable,
    )
    (
        curve_surface_signed_value,
        curve_surface_grad,
        curve_surface_violation,
        curve_surface_hard_signed_value,
        curve_surface_hard_violation,
    ) = _evaluate_constraint_with_optional_hard_signal(
        curve_surface_constraint_fn,
        curve_surface_constraint_with_hard_signal_fn,
        hard_surrogate_diagnostics,
        curves,
        outer_surface,
        curve_surface_min_distance,
        distance_smoothing,
        objective_optimizable,
    )
    curvature_constraint_curves = (
        (banana_curve,) if banana_curves is None else tuple(banana_curves)
    )
    curvature_signed_value, curvature_grad, curvature_violation = (
        _worst_signed_constraint(
            [
                curvature_constraint_fn(
                    curve,
                    curvature_threshold,
                    curvature_smoothing,
                    objective_optimizable,
                )
                for curve in curvature_constraint_curves
            ]
        )
    )

    hardware_constraints: dict[str, tuple[float, np.ndarray, float]] = {
        "coil_coil_spacing": (
            curve_curve_signed_value,
            curve_curve_grad,
            curve_curve_violation,
        ),
        "coil_surface_spacing": (
            curve_surface_signed_value,
            curve_surface_grad,
            curve_surface_violation,
        ),
        "max_curvature": (
            curvature_signed_value,
            curvature_grad,
            curvature_violation,
        ),
    }
    if surface_stack_surfaces is not None:
        # Forward the adjoint-capable BoozerSurfaces ONLY when supplied: with them the
        # constraint routes its per-surface point gradient through each Boozer adjoint
        # for a LIVE coil-dof gradient; without them (the default) the dispatch stays
        # byte-identical to the legacy gradient-dead fixed-surface path. Passing the
        # keyword unconditionally would force every constraint_fn to accept it, so the
        # legacy/None case omits it and reaches the constraint exactly as before.
        surface_stack_boozer_kwargs = (
            {}
            if surface_stack_boozer_surfaces is None
            else {"boozer_surfaces": surface_stack_boozer_surfaces}
        )
        (
            surface_stack_signed_value,
            surface_stack_grad,
            surface_stack_violation,
            surface_stack_hard_signed_value,
            surface_stack_hard_violation,
        ) = _evaluate_constraint_with_optional_hard_signal(
            surface_stack_constraint_fn,
            surface_stack_constraint_with_hard_signal_fn,
            hard_surrogate_diagnostics,
            surface_stack_surfaces,
            surface_stack_min_distance,
            distance_smoothing,
            objective_optimizable,
            **surface_stack_boozer_kwargs,
        )
        hardware_constraints["surface_surface_spacing"] = (
            surface_stack_signed_value,
            surface_stack_grad,
            surface_stack_violation,
        )
    coil_length_constraint_objectives = (
        (coil_length_objective,)
        if coil_length_objectives is None
        else tuple(coil_length_objectives)
    )
    if coil_length_objective is not None and coil_length_threshold is not None:
        hardware_constraints["coil_length_upper_bound"] = (
            _worst_signed_constraint(
                [
                    _objective_upper_bound_constraint(
                        objective,
                        coil_length_threshold,
                        objective_optimizable,
                    )
                    for objective in coil_length_constraint_objectives
                ]
            )
        )
    if coil_length_objective is not None and coil_length_min_threshold is not None:
        hardware_constraints["coil_length_min"] = _worst_signed_constraint(
            [
                _objective_lower_bound_constraint(
                    objective,
                    coil_length_min_threshold,
                    objective_optimizable,
                )
                for objective in coil_length_constraint_objectives
            ]
        )
    if banana_current is not None and banana_current_threshold is not None:
        hardware_constraints["banana_current_upper_bound"] = (
            _scalar_abs_upper_bound_constraint(
                banana_current,
                banana_current_threshold,
                objective_optimizable,
            )
        )
    if banana_currents is not None and banana_current_threshold is not None:
        for index, current in enumerate(banana_currents):
            hardware_constraints[
                independent_banana_current_alm_constraint_name(index)
            ] = _scalar_abs_upper_bound_constraint(
                current,
                banana_current_threshold,
                objective_optimizable,
            )
    if (
        JPoloidalExtent is not None
        and poloidal_extent_threshold is not None
        and poloidal_extent_constraint_fn is not None
    ):
        poloidal_extent_smoothing_value = (
            curvature_smoothing
            if poloidal_extent_smoothing is None
            else poloidal_extent_smoothing
        )
        poloidal_extent_terms = (
            (JPoloidalExtent,)
            if JPoloidalExtentTerms is None
            else tuple(JPoloidalExtentTerms)
        )
        (
            poloidal_extent_signed_value,
            poloidal_extent_grad,
            poloidal_extent_violation,
            poloidal_extent_hard_signed_value,
            poloidal_extent_hard_violation,
        ) = _worst_signed_constraint(
            [
                _evaluate_constraint_with_optional_hard_signal(
                    poloidal_extent_constraint_fn,
                    poloidal_extent_constraint_with_hard_signal_fn,
                    hard_surrogate_diagnostics,
                    term.curve,
                    term.R_winding,
                    poloidal_extent_threshold,
                    poloidal_extent_smoothing_value,
                    objective_optimizable,
                    Z_winding=term.Z_winding,
                )
                for term in poloidal_extent_terms
            ]
        )
        hardware_constraints["poloidal_extent"] = (
            poloidal_extent_signed_value,
            poloidal_extent_grad,
            poloidal_extent_violation,
        )
    coil_width_value = None
    coil_width_grad = None
    self_intersect_penalty = None
    self_intersect_grad = None
    if JCoilWidth is not None:
        coil_width_terms = (
            (JCoilWidth,) if JCoilWidthTerms is None else tuple(JCoilWidthTerms)
        )
        coil_width_values_and_grads = [
            (float(term.J()), _objective_gradient(term, objective_optimizable))
            for term in coil_width_terms
        ]
        coil_width_value, coil_width_grad = max(
            coil_width_values_and_grads,
            key=lambda value_and_grad: value_and_grad[0],
        )
        if width_min_threshold is not None:
            width_min_value, width_min_grad = min(
                coil_width_values_and_grads,
                key=lambda value_and_grad: value_and_grad[0],
            )
            width_min_signed_value = float(width_min_threshold) - width_min_value
            hardware_constraints["width_min"] = (
                width_min_signed_value,
                -width_min_grad,
                _positive_violation(width_min_signed_value),
            )
        if width_max_threshold is not None:
            width_max_signed_value = coil_width_value - float(width_max_threshold)
            hardware_constraints["width_max"] = (
                width_max_signed_value,
                coil_width_grad,
                _positive_violation(width_max_signed_value),
            )
    if JCurveSelfIntersect is not None:
        self_intersect_terms = (
            (JCurveSelfIntersect,)
            if JCurveSelfIntersectTerms is None
            else tuple(JCurveSelfIntersectTerms)
        )
        self_intersect_penalty, self_intersect_grad = max(
            [
                (float(term.J()), _objective_gradient(term, objective_optimizable))
                for term in self_intersect_terms
            ],
            key=lambda value_and_grad: value_and_grad[0],
        )
        self_intersect_signed_value = self_intersect_penalty
        hardware_constraints["self_intersect"] = (
            self_intersect_signed_value,
            self_intersect_grad,
            _positive_violation(self_intersect_signed_value),
        )
    if JCurveHardwareKeepout is not None:
        # Same penalty-as-constraint contract as self_intersect: clear of the
        # sensor/mount keep-out cloud is exactly zero hinge; any positive value
        # is an active ALM violation with zero slack.
        hardware_keepout_penalty = float(JCurveHardwareKeepout.J())
        hardware_keepout_grad = _objective_gradient(
            JCurveHardwareKeepout,
            objective_optimizable,
        )
        hardware_keepout_signed_value = hardware_keepout_penalty
        hardware_constraints["hardware_keepout"] = (
            hardware_keepout_signed_value,
            hardware_keepout_grad,
            _positive_violation(hardware_keepout_signed_value),
        )
    resolved_lcfs_constraint_mode = validate_lcfs_constraint_mode(lcfs_constraint_mode)
    if (
        lcfs_surface is not None
        and resolved_lcfs_constraint_mode == LCFS_CONSTRAINT_MODE_CENTERED
        and lcfs_major_radius_threshold is not None
    ):
        # The LCFS Fourier dofs are fixed (not in the coil dof graph), so the
        # fixed-surface dmajor_radius_by_dcoeff path is gradient-dead over the coils
        # and cannot actuate the ALM constraint. When the Boozer-adjoint objective is
        # supplied, route the true d(major_radius)/d(coils) through it instead; the
        # signed value (major_radius - threshold) is identical.
        if JLCFSMajorRadius is not None:
            hardware_constraints["lcfs_major_radius"] = (
                _boozer_adjoint_upper_bound_constraint(
                    JLCFSMajorRadius,
                    lcfs_major_radius_threshold,
                    objective_optimizable,
                )
            )
        else:
            hardware_constraints["lcfs_major_radius"] = _surface_upper_bound_constraint(
                lcfs_surface,
                lcfs_surface.major_radius(),
                lcfs_surface.dmajor_radius_by_dcoeff(),
                lcfs_major_radius_threshold,
                objective_optimizable,
            )
    if (
        lcfs_surface is not None
        and resolved_lcfs_constraint_mode == LCFS_CONSTRAINT_MODE_EDGE_ENVELOPE
        and lcfs_outboard_edge_threshold is not None
    ):
        if JLCFSMajorRadius is not None and JLCFSMinorRadius is not None:
            hardware_constraints["lcfs_outboard_edge"] = (
                _boozer_adjoint_linear_radius_constraint(
                    JLCFSMajorRadius,
                    JLCFSMinorRadius,
                    1.0,
                    1.0,
                    -float(lcfs_outboard_edge_threshold),
                    objective_optimizable,
                )
            )
        else:
            hardware_constraints["lcfs_outboard_edge"] = (
                _surface_linear_radius_constraint(
                    lcfs_surface,
                    1.0,
                    1.0,
                    -float(lcfs_outboard_edge_threshold),
                    objective_optimizable,
                )
            )
    if (
        lcfs_surface is not None
        and resolved_lcfs_constraint_mode == LCFS_CONSTRAINT_MODE_EDGE_ENVELOPE
        and lcfs_inboard_edge_threshold is not None
    ):
        if JLCFSMajorRadius is not None and JLCFSMinorRadius is not None:
            hardware_constraints["lcfs_inboard_edge"] = (
                _boozer_adjoint_linear_radius_constraint(
                    JLCFSMajorRadius,
                    JLCFSMinorRadius,
                    -1.0,
                    1.0,
                    float(lcfs_inboard_edge_threshold),
                    objective_optimizable,
                )
            )
        else:
            hardware_constraints["lcfs_inboard_edge"] = (
                _surface_linear_radius_constraint(
                    lcfs_surface,
                    -1.0,
                    1.0,
                    float(lcfs_inboard_edge_threshold),
                    objective_optimizable,
                )
            )
    if lcfs_surface is not None and lcfs_minor_radius_threshold is not None:
        # Same gradient-dead mechanism as major radius: the LCFS Fourier dofs are
        # fixed (not in the coil dof graph), so the fixed-surface
        # dminor_radius_by_dcoeff path is zero over the coils and cannot actuate the
        # ALM constraint. When the Boozer-adjoint objective is supplied, route the
        # true d(minor_radius)/d(coils) through it; the signed value
        # (minor_radius - threshold) is identical.
        if JLCFSMinorRadius is not None:
            hardware_constraints["lcfs_minor_radius"] = (
                _boozer_adjoint_upper_bound_constraint(
                    JLCFSMinorRadius,
                    lcfs_minor_radius_threshold,
                    objective_optimizable,
                )
            )
        else:
            hardware_constraints["lcfs_minor_radius"] = _surface_upper_bound_constraint(
                lcfs_surface,
                lcfs_surface.minor_radius(),
                lcfs_surface.dminor_radius_by_dcoeff(),
                lcfs_minor_radius_threshold,
                objective_optimizable,
            )

    hard_signed_values_by_name: dict[str, float] = {
        name: float(values[0]) for name, values in hardware_constraints.items()
    }
    hard_violation_values_by_name: dict[str, float] = {
        name: float(values[2]) for name, values in hardware_constraints.items()
    }
    if hard_surrogate_diagnostics:
        curve_curve_hard_signed_value, curve_curve_hard_violation = (
            _resolve_hard_signal(
                curve_curve_hard_signed_value,
                curve_curve_hard_violation,
                _hard_min_curve_curve_signed_constraint,
                curves,
                curve_curve_min_distance,
            )
        )
        curve_surface_hard_signed_value, curve_surface_hard_violation = (
            _resolve_hard_signal(
                curve_surface_hard_signed_value,
                curve_surface_hard_violation,
                _hard_min_curve_surface_signed_constraint,
                curves,
                outer_surface,
                curve_surface_min_distance,
            )
        )
        hard_signed_values_by_name["coil_coil_spacing"] = curve_curve_hard_signed_value
        hard_violation_values_by_name["coil_coil_spacing"] = curve_curve_hard_violation
        hard_signed_values_by_name["coil_surface_spacing"] = (
            curve_surface_hard_signed_value
        )
        hard_violation_values_by_name["coil_surface_spacing"] = (
            curve_surface_hard_violation
        )
        (
            hard_signed_values_by_name["max_curvature"],
            hard_violation_values_by_name["max_curvature"],
        ) = _worst_signed_constraint(
            [
                _hard_max_curvature_signed_constraint(curve, curvature_threshold)
                for curve in curvature_constraint_curves
            ]
        )
        if surface_stack_surfaces is not None:
            surface_stack_hard_signed_value, surface_stack_hard_violation = (
                _resolve_hard_signal(
                    surface_stack_hard_signed_value,
                    surface_stack_hard_violation,
                    _hard_min_surface_stack_signed_constraint,
                    surface_stack_surfaces,
                    surface_stack_min_distance,
                )
            )
            (
                hard_signed_values_by_name["surface_surface_spacing"],
                hard_violation_values_by_name["surface_surface_spacing"],
            ) = (surface_stack_hard_signed_value, surface_stack_hard_violation)
        if "poloidal_extent" in hardware_constraints:
            poloidal_extent_hard_signed_value, poloidal_extent_hard_violation = (
                _resolve_hard_signal(
                    poloidal_extent_hard_signed_value,
                    poloidal_extent_hard_violation,
                    lambda terms, threshold: _worst_signed_constraint(
                        [
                            poloidal_extent_signed_constraint(
                                term.curve,
                                term.R_winding,
                                threshold,
                                Z_winding=term.Z_winding,
                            )
                            for term in terms
                        ]
                    ),
                    poloidal_extent_terms,
                    poloidal_extent_threshold,
                )
            )
            hard_signed_values_by_name["poloidal_extent"] = (
                poloidal_extent_hard_signed_value
            )
            hard_violation_values_by_name["poloidal_extent"] = (
                poloidal_extent_hard_violation
            )

    physics_constraints: dict[str, tuple[float, np.ndarray, float]] = {}
    if alm_formulation == "thresholded_physics":
        physics_constraints = {
            "qs_error": _objective_upper_bound_constraint(
                raw_J_QS_obj,
                qs_threshold,
                objective_optimizable,
            ),
            "boozer_residual": _objective_upper_bound_constraint(
                raw_J_Boozer_obj,
                boozer_threshold,
                objective_optimizable,
            ),
            "iota_penalty": _objective_upper_bound_constraint(
                Jiota,
                iota_penalty_threshold,
                objective_optimizable,
            ),
            "length_penalty": _objective_upper_bound_constraint(
                JCurveLength,
                length_penalty_threshold,
                objective_optimizable,
            ),
        }

    active_constraint_names: list[str] = []
    constraint_values: list[float] = []
    constraint_grads: list[np.ndarray] = []
    surrogate_feasibility_values: list[float] = []
    dual_update_values: list[float] = []
    feasibility_values: list[float] = []
    hard_signed_values: list[float] = []
    hard_violation_values: list[float] = []
    surrogate_signed_values: list[float] = []
    for constraint_name in constraint_names:
        if constraint_name in hardware_constraints:
            signed_value, grad, violation = hardware_constraints[constraint_name]
            hard_signed_value = hard_signed_values_by_name[constraint_name]
            hard_violation_value = hard_violation_values_by_name[constraint_name]
        elif constraint_name in physics_constraints:
            signed_value, grad, violation = physics_constraints[constraint_name]
            hard_signed_value = signed_value
            hard_violation_value = violation
        else:
            raise ValueError(f"Unknown ALM constraint name {constraint_name!r}.")
        active_constraint_names.append(constraint_name)
        constraint_values.append(signed_value)
        constraint_grads.append(grad)
        surrogate_feasibility_values.append(violation)
        hard_signed_values.append(hard_signed_value)
        hard_violation_values.append(hard_violation_value)
        surrogate_signed_values.append(signed_value)
        if hard_surrogate_diagnostics and constraint_name in hardware_constraints:
            dual_update_values.append(hard_signed_value)
            feasibility_values.append(hard_violation_value)
        else:
            dual_update_values.append(signed_value)
            feasibility_values.append(violation)

    threshold_overrides = _single_stage_hardware_threshold_overrides(
        curve_curve_min_distance=curve_curve_min_distance,
        curve_surface_min_distance=curve_surface_min_distance,
        curvature_threshold=curvature_threshold,
        surface_stack_min_distance=surface_stack_min_distance,
        coil_length_threshold=coil_length_threshold,
        coil_length_min_threshold=coil_length_min_threshold,
        banana_current_threshold=banana_current_threshold,
        poloidal_extent_threshold=poloidal_extent_threshold,
        width_min_threshold=width_min_threshold,
        width_max_threshold=width_max_threshold,
        lcfs_major_radius_threshold=lcfs_major_radius_threshold,
        lcfs_minor_radius_threshold=lcfs_minor_radius_threshold,
        lcfs_outboard_edge_threshold=lcfs_outboard_edge_threshold,
        lcfs_inboard_edge_threshold=lcfs_inboard_edge_threshold,
    )
    geometry_tolerances = np.asarray(
        activity_tolerances_fn(
            distance_smoothing,
            curvature_smoothing,
            include_surface_stack=surface_stack_surfaces is not None,
        ),
        dtype=float,
    )
    geometry_names = ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"]
    if surface_stack_surfaces is not None:
        geometry_names.append("surface_surface_spacing")
    if JPoloidalExtent is not None:
        geometry_names.append("poloidal_extent")
    constraint_tolerance_by_name = {
        name: float(value) for name, value in zip(geometry_names, geometry_tolerances)
    }
    for constraint_name in active_constraint_names:
        metadata_name = _banana_current_alm_metadata_name(constraint_name)
        if metadata_name in {
            "coil_length_min",
            "coil_length_upper_bound",
            "banana_current_upper_bound",
            "poloidal_extent",
            "width_min",
            "width_max",
            "self_intersect",
            "lcfs_major_radius",
            "lcfs_outboard_edge",
            "lcfs_inboard_edge",
            "lcfs_minor_radius",
        }:
            constraint_tolerance_by_name[constraint_name] = (
                _hardware_alm_activity_tolerance(
                    metadata_name,
                    threshold_overrides=threshold_overrides,
                )
            )
    constraint_activity_tolerances = np.asarray(
        [
            constraint_tolerance_by_name.get(constraint_name, 0.0)
            for constraint_name in active_constraint_names
        ],
        dtype=float,
    )
    base_eval["constraint_activity_tolerances"] = constraint_activity_tolerances
    metadata_by_name = _single_stage_alm_constraint_metadata(
        active_constraint_names,
        threshold_overrides=threshold_overrides,
        activity_tolerance_by_name=constraint_tolerance_by_name,
        use_hard_geometry_signals=hard_surrogate_diagnostics,
        qs_threshold=qs_threshold,
        boozer_threshold=boozer_threshold,
        iota_penalty_threshold=iota_penalty_threshold,
        length_penalty_threshold=length_penalty_threshold,
    )
    metadata_payload = alm_constraint_metadata_payload(
        active_constraint_names,
        metadata_by_name,
    )
    constraint_scales = np.asarray(metadata_payload["constraint_scales"], dtype=float)
    normalized_constraint_grads = normalize_alm_constraint_grads(
        constraint_grads,
        constraint_scales,
    )
    normalized_payload = normalize_alm_constraint_signals(
        constraint_values,
        surrogate_feasibility_values,
        constraint_activity_tolerances,
        constraint_scales,
    )
    signal_payload = normalize_alm_constraint_signals(
        dual_update_values,
        feasibility_values,
        constraint_activity_tolerances,
        constraint_scales,
    )
    hard_signal_payload = normalize_alm_constraint_signals(
        hard_signed_values,
        hard_violation_values,
        constraint_activity_tolerances,
        constraint_scales,
    )
    surrogate_signal_payload = normalize_alm_constraint_signals(
        surrogate_signed_values,
        surrogate_feasibility_values,
        constraint_activity_tolerances,
        constraint_scales,
    )
    normalized_constraint_values = normalized_payload["normalized_signed_values"]
    normalized_dual_update_values = signal_payload["normalized_signed_values"]
    normalized_feasibility_values = signal_payload["normalized_feasibility_values"]
    normalized_activity_tolerances = normalized_payload[
        "normalized_activity_tolerances"
    ]
    alm_eval = augmented_inequality_objective_fn(
        base_eval["total"],
        base_eval["grad"],
        normalized_constraint_values,
        normalized_constraint_grads,
        multipliers,
        penalty,
    )
    base_total = float(base_eval["physics_total"])
    base_eval.update(alm_eval)
    base_eval["base_total"] = base_total
    base_eval["constraint_names"] = active_constraint_names
    base_eval["dual_update_values"] = normalized_dual_update_values
    base_eval["feasibility_values"] = normalized_feasibility_values
    base_eval["max_feasibility_violation"] = float(max(normalized_feasibility_values))
    base_eval["constraint_grads"] = normalized_constraint_grads
    base_eval["constraint_activity_tolerances"] = normalized_activity_tolerances
    base_eval["normalized_signed_constraint_values"] = normalized_constraint_values
    base_eval["normalized_feasibility_values"] = normalized_feasibility_values
    base_eval["hard_signed_constraint_values"] = hard_signal_payload[
        "normalized_signed_values"
    ]
    base_eval["hard_violation_values"] = hard_signal_payload[
        "normalized_feasibility_values"
    ]
    base_eval["surrogate_signed_constraint_values"] = surrogate_signal_payload[
        "normalized_signed_values"
    ]
    base_eval["hard_dual_update_values"] = hard_signal_payload[
        "normalized_signed_values"
    ]
    base_eval["search_hardware_constraint_payload_kind"] = "signed_residual"
    base_eval.update(metadata_payload)
    base_eval["raw_constraint_values"] = np.asarray(constraint_values, dtype=float)
    base_eval["raw_solver_constraint_values"] = np.asarray(
        constraint_values, dtype=float
    )
    base_eval["raw_dual_update_values"] = np.asarray(dual_update_values, dtype=float)
    base_eval["raw_feasibility_values"] = np.asarray(feasibility_values, dtype=float)
    base_eval["raw_hard_signed_constraint_values"] = np.asarray(
        hard_signed_values,
        dtype=float,
    )
    base_eval["raw_hard_violation_values"] = np.asarray(
        hard_violation_values,
        dtype=float,
    )
    base_eval["raw_surrogate_signed_constraint_values"] = np.asarray(
        surrogate_signed_values,
        dtype=float,
    )
    base_eval["raw_hard_dual_update_values"] = np.asarray(
        hard_signed_values,
        dtype=float,
    )
    base_eval["raw_constraint_grads"] = [
        np.asarray(grad, dtype=float) for grad in constraint_grads
    ]
    base_eval["raw_constraint_activity_tolerances"] = constraint_activity_tolerances
    if include_diagnostics:
        base_eval["diagnostics_included"] = True
        base_eval["J_cc"] = float(JCurveCurve.J())
        base_eval["dJ_cc"] = _objective_gradient(
            JCurveCurve,
            objective_optimizable,
        )
        base_eval["J_cs"] = float(JCurveSurface.J())
        base_eval["dJ_cs"] = _objective_gradient(
            JCurveSurface,
            objective_optimizable,
        )
        base_eval["J_curvature"] = float(JCurvature.J())
        base_eval["dJ_curvature"] = _objective_gradient(
            JCurvature,
            objective_optimizable,
        )
        if coil_length_objective is not None:
            coil_length_spec = get_hardware_constraint_spec("coil_length")
            base_eval["coil_length_upper_bound_threshold"] = (
                coil_length_spec.threshold
                if coil_length_threshold is None
                else float(coil_length_threshold)
            )
            if coil_length_min_threshold is not None:
                base_eval["coil_length_min_threshold"] = float(
                    coil_length_min_threshold
                )
        if banana_current_threshold is not None:
            base_eval["banana_current_upper_bound_threshold"] = float(
                banana_current_threshold
            )
        if poloidal_extent_threshold is not None:
            base_eval["poloidal_extent_rad"] = (
                None
                if JPoloidalExtent is None
                else poloidal_extent_rad_from_objective(JPoloidalExtent)
            )
            base_eval["poloidal_extent_threshold_rad"] = float(
                poloidal_extent_threshold
            )
        base_eval["J_poloidal_extent"] = (
            0.0 if JPoloidalExtent is None else float(JPoloidalExtent.J())
        )
        base_eval["dJ_poloidal_extent"] = (
            np.zeros_like(base_eval["grad"])
            if JPoloidalExtent is None
            else _objective_gradient(JPoloidalExtent, objective_optimizable)
        )
        base_eval["J_coil_width"] = (
            0.0 if JCoilWidth is None else float(coil_width_value)
        )
        base_eval["dJ_coil_width"] = (
            np.zeros_like(base_eval["grad"])
            if JCoilWidth is None
            else np.asarray(coil_width_grad, dtype=float)
        )
        if JCoilWidth is not None:
            base_eval["coil_width"] = coil_width_value
            if width_min_threshold is not None:
                base_eval["coil_width_min_threshold"] = float(width_min_threshold)
            if width_max_threshold is not None:
                base_eval["coil_width_max_threshold"] = float(width_max_threshold)
        base_eval["J_self_intersect"] = (
            0.0 if JCurveSelfIntersect is None else float(self_intersect_penalty)
        )
        base_eval["dJ_self_intersect"] = (
            np.zeros_like(base_eval["grad"])
            if JCurveSelfIntersect is None
            else np.asarray(self_intersect_grad, dtype=float)
        )
        if JCurveSelfIntersect is not None:
            base_eval["self_intersect_penalty"] = self_intersect_penalty
            base_eval["self_intersect_threshold"] = 0.0
        base_eval["J_hardware_keepout"] = (
            0.0 if JCurveHardwareKeepout is None else float(hardware_keepout_penalty)
        )
        base_eval["dJ_hardware_keepout"] = (
            np.zeros_like(base_eval["grad"])
            if JCurveHardwareKeepout is None
            else np.asarray(hardware_keepout_grad, dtype=float)
        )
        if JCurveHardwareKeepout is not None:
            base_eval["hardware_keepout_penalty"] = hardware_keepout_penalty
            base_eval["hardware_keepout_threshold"] = 0.0
            base_eval["hardware_keepout_min_distance"] = (
                HARDWARE_KEEPOUT_MIN_DISTANCE_M
            )
        # Opt-in SIMSOPT coil regularizers (frontier override reads these from the
        # diagnostics dict; 0.0/zeros when the weight is off and the term is None).
        base_eval["J_msc"] = (
            0.0 if JMeanSquaredCurvature is None else float(JMeanSquaredCurvature.J())
        )
        base_eval["dJ_msc"] = (
            np.zeros_like(base_eval["grad"])
            if JMeanSquaredCurvature is None
            else _objective_gradient(JMeanSquaredCurvature, objective_optimizable)
        )
        base_eval["J_arclen"] = (
            0.0 if JArclengthVariation is None else float(JArclengthVariation.J())
        )
        base_eval["dJ_arclen"] = (
            np.zeros_like(base_eval["grad"])
            if JArclengthVariation is None
            else _objective_gradient(JArclengthVariation, objective_optimizable)
        )
        base_eval["J_link"] = (
            0.0 if JLinkingNumber is None else float(JLinkingNumber.J())
        )
        base_eval["dJ_link"] = (
            np.zeros_like(base_eval["grad"])
            if JLinkingNumber is None
            else _objective_gradient(JLinkingNumber, objective_optimizable)
        )
        base_eval["J_coil_force"] = 0.0 if JCoilForce is None else float(JCoilForce.J())
        base_eval["dJ_coil_force"] = (
            np.zeros_like(base_eval["grad"])
            if JCoilForce is None
            else _objective_gradient(JCoilForce, objective_optimizable)
        )
    base_eval["alm_formulation"] = alm_formulation
    return annotate_search_evaluation_finiteness(base_eval)
