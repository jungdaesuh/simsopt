"""Realized rotational-transform profile ι(r) of a vacuum field, and selection
of in-domain Greene rational targets from it.

The Greene residue driver needs two facts answered *before* it tries to find a
periodic orbit (doc ``kam_correct_implementation_research_2026-05-27.md`` §3.2):
which p/q rationals actually occur inside the traced radial band, and at what
midplane radial label each one sits. Without them, a target whose ι is out of
domain has no periodic orbit to find, and Newton slides onto the trivial
magnetic-axis fixed point (winding ≈ 0) or stalls.

This module measures ι(r) by tracing the BiotSavart field directly, in the
*same* full-torus winding convention the residue probe uses
(``integrate_full_torus_return_map`` + ``PoincareChart`` unwrapped angle). A p/q
island chain sits where ι ≡ p/q (mod 1): the full-torus poloidal winding carries
an integer geometric/section offset that the residue map removes, so a field
rotating at, e.g., nfp + p/q is a p/q resonance. From the profile it (a)
re-centers the chart on the located magnetic axis, (b) places each target's
``radial_label``/``radial_window`` on its innermost realized ι ≡ p/q (mod 1)
resonance, and (c) reports any requested rational with no realized branch in the
traced band — instead of copying labels/windows verbatim from a manifest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import ceil, floor, gcd, isfinite, pi

import numpy as np

from simsopt.field.magnetic_axis_helpers import locate_magnetic_axis_point

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    FieldlineIntegratorOptions,
    MagneticFieldLike,
    integrate_full_torus_return_map,
)
from .poincare_chart import PoincareChart
from .rational_target import RationalTarget


# Toroidal turns traced per radial seed when estimating ι(r). The vacuum
# field-line return map is smooth, so the endpoint winding over a few tens of
# turns resolves ι to well below the rational spacings of interest.
DEFAULT_IOTA_PROFILE_TORUS_TURNS = 40

# R/Z search bound (metres) around the supplied axis guess for the magnetic-axis
# locator. A search bound, not a model parameter; override per device scale.
DEFAULT_AXIS_SEARCH_HALF_WIDTH = 0.05

# Largest rational denominator this module is asked to keep distinct. The WBA
# island scan enumerates up to q=24 (kam_birkhoff.DEFAULT_WBA_RATIONAL_MAX_DENOMINATOR);
# kept as the module's own resolution-band SSOT so the match tolerance below is
# derived rather than guessed, without coupling to that module's import.
MATCH_TOLERANCE_REFERENCE_DENOMINATOR = 24

# Absolute |ι − p/q| at which a traced sample counts as sitting on the rational
# (e.g. a flat profile pinned to a rational surface). Derived as a small fraction
# of the worst-case spacing between distinct reduced fractions: two Farey
# neighbours with denominators ≤ Q are at least 1/(Q·(Q−1)) apart (e.g. for
# Q=24 that is 1.81e-3). At 1/1000 of that worst-case gap a sample can sit on
# its own rational without ever being mistaken for an adjacent one, while still
# admitting realistic return-map winding error (ι resolved to well below 1e-4
# over tens of turns). The 1/1000 margin is the safety factor between "measured
# ι noise" and "adjacent-rational confusion", the two scales this gate separates.
_MATCH_TOLERANCE_GAP_SAFETY_FACTOR = 1.0e-3
DEFAULT_IOTA_MATCH_TOLERANCE = _MATCH_TOLERANCE_GAP_SAFETY_FACTOR / (
    MATCH_TOLERANCE_REFERENCE_DENOMINATOR
    * (MATCH_TOLERANCE_REFERENCE_DENOMINATOR - 1)
)

# Reasons attached to a freezing resolution (the contract a caller reports).
FREEZE_IN_DOMAIN = "iota_crossing_in_profile"
FREEZE_NO_CROSSING = "no_resonant_branch_in_iota_band"
FREEZE_EMPTY_PROFILE = "iota_profile_has_no_valid_samples"

# Reason a single radial seed produced no ι (excluded from the usable profile).
IOTA_SAMPLE_TRACE_FAILED = "fieldline_trace_failed"


def magnetic_axis_from_guess(
    field: MagneticFieldLike,
    *,
    nfp: int,
    axis_r_guess: float,
    axis_z_guess: float = 0.0,
    phi0: float = 0.0,
    search_half_width: float = DEFAULT_AXIS_SEARCH_HALF_WIDTH,
    min_bphi_over_b: float = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS.min_bphi_over_b,
) -> tuple[float, float]:
    """Return the magnetic-axis (R, Z) as the return-map fixed point near a guess.

    Thin policy over ``locate_magnetic_axis_point`` that derives R/Z search
    bounds from ``search_half_width`` (clamping R away from the coordinate
    singularity) so callers pass only a guess and nfp. Raises if the locator
    cannot converge — a wrong axis silently biases every downstream ι and window.
    """

    guess_r = float(axis_r_guess)
    guess_z = float(axis_z_guess)
    half_width = float(search_half_width)
    if not isfinite(guess_r) or guess_r <= 0.0:
        raise ValueError("magnetic-axis guess requires finite R > 0")
    if not isfinite(guess_z):
        raise ValueError("magnetic-axis guess requires finite Z")
    if not isfinite(half_width) or half_width <= 0.0:
        raise ValueError("magnetic-axis search_half_width must be finite and positive")
    r_lower = max(float(np.finfo(float).eps), guess_r - half_width)
    point = locate_magnetic_axis_point(
        field,
        (guess_r, guess_z),
        nfp=int(nfp),
        phi0=float(phi0),
        r_bounds=(r_lower, guess_r + half_width),
        z_bounds=(guess_z - half_width, guess_z + half_width),
        min_bphi_over_b=float(min_bphi_over_b),
    )
    return (float(point["r"]), float(point["z"]))


def linear_radial_labels(
    *,
    lower: float,
    upper: float,
    count: int,
) -> tuple[float, ...]:
    """Return ``count`` midplane radial labels evenly spaced over (lower, upper].

    Both bounds must be positive: a label of 0 is the magnetic axis, where ι is
    undefined and the field line does not wind.
    """

    lower_label = float(lower)
    upper_label = float(upper)
    sample_count = int(count)
    if sample_count < 2:
        raise ValueError("iota profile requires at least two radial labels")
    if not isfinite(lower_label) or not isfinite(upper_label):
        raise ValueError("iota profile radial bounds must be finite")
    if lower_label <= 0.0 or upper_label <= lower_label:
        raise ValueError("iota profile radial bounds must satisfy 0 < lower < upper")
    return tuple(
        float(label) for label in np.linspace(lower_label, upper_label, sample_count)
    )


@dataclass(frozen=True, slots=True)
class IotaProfileSample:
    """One traced seed: its midplane radial label and realized ι (or why not).

    ``iota`` is the rotation number (poloidal turns per toroidal turn) measured
    from the full-torus return map; ``None`` when the trace aborted (e.g. low
    |B_φ|), in which case ``reason`` explains and the sample is excluded from
    rational location. ``radial_label`` is in PoincareChart label units.
    """

    radial_label: float
    iota: float | None
    toroidal_turns: int
    min_bphi_over_b: float | None
    reason: str


def sample_iota_profile(
    field: MagneticFieldLike,
    *,
    axis_r: float,
    axis_z: float,
    radial_labels: Sequence[float],
    toroidal_turns: int = DEFAULT_IOTA_PROFILE_TORUS_TURNS,
    poloidal_orientation: int = 1,
    radial_label_scale: float = 1.0,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> "IotaProfile":
    """Trace each midplane seed and return the realized ι(r) profile.

    Seeds are placed at ``(axis_r + radial_label_scale * label, axis_z)`` on the
    φ=phi0 plane and integrated ``toroidal_turns`` full toroidal turns; ι is the
    net unwrapped poloidal angle over 2π·turns, i.e. the same winding the residue
    probe q-step-reduces to p. Samples are returned sorted by radial label;
    failed traces are retained with ``iota=None`` so the profile stays auditable.
    """

    turns = int(toroidal_turns)
    if turns <= 0:
        raise ValueError("iota profile toroidal_turns must be positive")
    chart = PoincareChart(
        axis_r=float(axis_r),
        axis_z=float(axis_z),
        poloidal_orientation=int(poloidal_orientation),
        radial_label_scale=float(radial_label_scale),
    )
    unique_labels = _sorted_unique_positive_labels(radial_labels)
    samples: list[IotaProfileSample] = []
    for label in unique_labels:
        seed = (
            chart.axis_r + chart.radial_label_scale * label,
            chart.axis_z,
        )
        try:
            result = integrate_full_torus_return_map(
                field,
                seed,
                chart=chart,
                torus_turns=turns,
                options=integrator_options,
            )
        except RuntimeError:
            samples.append(
                IotaProfileSample(
                    radial_label=label,
                    iota=None,
                    toroidal_turns=turns,
                    min_bphi_over_b=None,
                    reason=IOTA_SAMPLE_TRACE_FAILED,
                )
            )
            continue
        total_poloidal_turns = (
            float(result.unwrapped_theta[-1]) - float(result.unwrapped_theta[0])
        ) / (2.0 * pi)
        samples.append(
            IotaProfileSample(
                radial_label=label,
                iota=total_poloidal_turns / float(turns),
                toroidal_turns=turns,
                min_bphi_over_b=float(result.min_bphi_over_b),
                reason="",
            )
        )
    return IotaProfile(
        axis_r=float(chart.axis_r),
        axis_z=float(chart.axis_z),
        samples=tuple(samples),
    )


def _sorted_unique_positive_labels(
    radial_labels: Sequence[float],
) -> tuple[float, ...]:
    labels = [float(label) for label in radial_labels]
    if len(labels) == 0:
        raise ValueError("iota profile requires at least one radial label")
    for label in labels:
        if not isfinite(label) or label <= 0.0:
            raise ValueError("iota profile radial labels must be finite and > 0")
    return tuple(sorted(set(labels)))


@dataclass(frozen=True, slots=True)
class RationalCrossing:
    """Where a p/q resonance (ι ≡ p/q mod 1) is realized in a profile.

    ``iota`` is the realized rotational transform there (the branch ``n + p/q``,
    which may exceed 1); ``radial_label`` is the (linearly interpolated) label at
    which it occurs; ``bracket_lower_label``/``bracket_upper_label`` are the
    adjacent traced samples enclosing it — a data-derived radial window
    guaranteed to contain the resonance. ``local_shear`` is the finite-difference
    dι/dlabel across that bracket (signed): its sign tells the caller whether ι
    is rising or falling through the resonance, so multiple crossings of the same
    p/q on a non-monotonic ι(r) are distinguishable rather than silently merged.
    """

    p: int
    q: int
    iota: float
    radial_label: float
    bracket_lower_label: float
    bracket_upper_label: float
    local_shear: float


@dataclass(frozen=True, slots=True)
class IotaProfile:
    """Realized ι as a function of midplane radial label for one field.

    Owns the rotation-number-vs-radius knowledge: the magnetic axis the labels
    are measured from, the traced samples, and the logic to locate where a given
    resonance is realized. A p/q resonance sits where ι ≡ p/q (mod 1); the
    innermost realized integer branch in increasing radial label is returned.
    """

    axis_r: float
    axis_z: float
    samples: tuple[IotaProfileSample, ...]

    def valid_samples(self) -> tuple[IotaProfileSample, ...]:
        return tuple(
            sample
            for sample in self.samples
            if sample.iota is not None and isfinite(sample.iota)
        )

    def iota_bounds(self) -> tuple[float, float] | None:
        valid = self.valid_samples()
        if len(valid) == 0:
            return None
        iotas = [float(sample.iota) for sample in valid]
        return (min(iotas), max(iotas))

    def all_rational_crossings(
        self,
        p: int,
        q: int,
        *,
        match_tolerance: float = DEFAULT_IOTA_MATCH_TOLERANCE,
    ) -> tuple[RationalCrossing, ...]:
        """Return every realized p/q crossing in the profile, sorted by label.

        A p/q island chain sits where ι ≡ p/q (mod 1): the full-torus poloidal
        winding about the axis carries an integer geometric/section offset that
        the residue map removes (e.g. a field rotating at nfp + p/q is a p/q
        resonance). So every integer branch ``n + p/q`` inside the traced band is
        a candidate. Within each branch this scans **all** adjacent-sample
        brackets, not just the first: a non-monotonic ι(r) can cross the same
        rational more than once (distinct island chains), and each crossing is a
        real resonance the caller must see rather than have silently collapsed to
        one. Each returned ``RationalCrossing`` carries the signed ``local_shear``
        across its bracket. A sample within ``match_tolerance`` of the branch is
        treated as on-resonance; otherwise the sign change is interpolated.
        """

        valid = self.valid_samples()
        bounds = self.iota_bounds()
        if len(valid) == 0 or bounds is None:
            return ()
        minimum_iota, maximum_iota = bounds
        base_fraction = (float(p) / float(q)) % 1.0
        tolerance = float(match_tolerance)
        lowest_branch = int(ceil(minimum_iota - base_fraction - tolerance))
        highest_branch = int(floor(maximum_iota - base_fraction + tolerance))
        crossings: list[RationalCrossing] = []
        for branch in range(lowest_branch, highest_branch + 1):
            crossings.extend(
                self._crossings_for_transform_value(
                    p,
                    q,
                    base_fraction + float(branch),
                    tolerance,
                )
            )
        return tuple(sorted(crossings, key=lambda crossing: crossing.radial_label))

    def locate_rational(
        self,
        p: int,
        q: int,
        *,
        match_tolerance: float = DEFAULT_IOTA_MATCH_TOLERANCE,
    ) -> RationalCrossing | None:
        """Return the innermost realized p/q crossing, or None if out of domain.

        Convenience over :meth:`all_rational_crossings` that selects the crossing
        at the lowest radial label (the innermost island chain). When ι(r) is
        non-monotonic and a rational is crossed more than once, the outer
        crossings are *not* spurious — callers that must see them use
        ``all_rational_crossings``; this method only narrows the full set to the
        innermost, it never hides crossings it was unaware of.
        """

        crossings = self.all_rational_crossings(
            p, q, match_tolerance=match_tolerance
        )
        if len(crossings) == 0:
            return None
        return crossings[0]

    def _crossings_for_transform_value(
        self,
        p: int,
        q: int,
        transform_value: float,
        match_tolerance: float,
    ) -> tuple[RationalCrossing, ...]:
        """Collect every bracket of ``valid_samples`` that realizes ``transform_value``.

        On-resonance samples (within ``match_tolerance``) and strict sign-change
        brackets are both reported; a sample that is itself within tolerance is
        attributed once to its own bracket and excluded from the sign-change scan
        of its neighbouring pairs so the same crossing is not counted twice.
        """

        valid = self.valid_samples()
        crossings: list[RationalCrossing] = []
        on_resonance = [
            abs(float(sample.iota) - transform_value) <= match_tolerance
            for sample in valid
        ]
        for index, sample in enumerate(valid):
            if not on_resonance[index]:
                continue
            lower = valid[max(0, index - 1)]
            upper = valid[min(len(valid) - 1, index + 1)]
            crossings.append(
                self._make_crossing(
                    p,
                    q,
                    transform_value,
                    radial_label=float(sample.radial_label),
                    lower=lower,
                    upper=upper,
                )
            )
        for index in range(len(valid) - 1):
            if on_resonance[index] or on_resonance[index + 1]:
                continue
            lower = valid[index]
            upper = valid[index + 1]
            lower_offset = float(lower.iota) - transform_value
            upper_offset = float(upper.iota) - transform_value
            if lower_offset * upper_offset >= 0.0:
                continue
            fraction = lower_offset / (lower_offset - upper_offset)
            label = lower.radial_label + fraction * (
                upper.radial_label - lower.radial_label
            )
            crossings.append(
                self._make_crossing(
                    p,
                    q,
                    transform_value,
                    radial_label=float(label),
                    lower=lower,
                    upper=upper,
                )
            )
        return tuple(crossings)

    @staticmethod
    def _make_crossing(
        p: int,
        q: int,
        transform_value: float,
        *,
        radial_label: float,
        lower: IotaProfileSample,
        upper: IotaProfileSample,
    ) -> RationalCrossing:
        label_span = float(upper.radial_label) - float(lower.radial_label)
        local_shear = (
            0.0
            if label_span == 0.0
            else (float(upper.iota) - float(lower.iota)) / label_span
        )
        return RationalCrossing(
            p=int(p),
            q=int(q),
            iota=float(transform_value),
            radial_label=float(radial_label),
            bracket_lower_label=float(lower.radial_label),
            bracket_upper_label=float(upper.radial_label),
            local_shear=float(local_shear),
        )

    def in_domain_rationals(
        self,
        *,
        max_denominator: int,
    ) -> tuple[RationalCrossing, ...]:
        """Return every reduced proper fraction p/q (2 ≤ q ≤ max_denominator,
        1 ≤ p < q) whose resonance ι ≡ p/q (mod 1) is realized, by radial label.
        """

        if self.iota_bounds() is None:
            return ()
        crossings: list[RationalCrossing] = []
        for q in range(2, int(max_denominator) + 1):
            for p in range(1, q):
                if gcd(p, q) != 1:
                    continue
                crossing = self.locate_rational(p, q)
                if crossing is not None:
                    crossings.append(crossing)
        return tuple(sorted(crossings, key=lambda item: item.radial_label))


@dataclass(frozen=True, slots=True)
class BandShearLoad:
    """How a realized ι(r) profile sweeps the low-order rational ladder in a band.

    Read off an existing :class:`IotaProfile` (no tracing). ``max_abs_shear`` is the
    largest |Δι/Δlabel| over band-adjacent samples (None when fewer than two samples
    fall in the band, so there is no pair to difference); ``low_order_count`` is the
    number of in-band realized low-order (q ≤ cutoff) crossings, counted per bracket
    (a conservative upper bound on distinct island chains — a profile pinned flat on
    one rational contributes one count per on-resonance sample);
    ``min_low_order_distance`` is how close the nearest in-band sample sits to any
    such rational (None when the band holds no samples). Lower shear and count and
    larger distance mean a flatter profile parked away from the dangerous resonances.
    """

    max_abs_shear: float | None
    low_order_count: int
    min_low_order_distance: float | None
    in_band_sample_count: int


def band_shear_load(
    profile: IotaProfile,
    *,
    band: tuple[float, float],
    max_low_order_denominator: int,
) -> BandShearLoad:
    """Summarize how ``profile`` sweeps low-order rationals inside a closed ι band.

    ``band`` is the physical operating window ``(ι_lo, ι_hi)``; samples outside it
    are ignored because the full-torus winding carries an integer charting offset
    near the axis (an inner sample can read ι≈1.x) that is not physical edge shear.
    Low-order resonances are reduced fractions p/q with 2 ≤ q ≤
    ``max_low_order_denominator``. Crossings are taken from
    :meth:`IotaProfile.all_rational_crossings` (every integer branch) and kept only
    when the realized branch value ``crossing.iota`` falls in the band, so a near-axis
    n+p/q branch outside the band is excluded rather than miscounted.
    """

    band_lower, band_upper = float(band[0]), float(band[1])
    if not (isfinite(band_lower) and isfinite(band_upper)) or band_lower >= band_upper:
        raise ValueError("band must satisfy finite lower < upper")
    max_q = int(max_low_order_denominator)
    if max_q < 2:
        raise ValueError("max_low_order_denominator must be at least 2")

    # Both the shear difference and all_rational_crossings' sign-change brackets
    # presuppose radially-ordered samples (so "adjacent" means radially adjacent).
    # sample_iota_profile already emits sorted samples; sort defensively here so a
    # hand-built profile still yields order-independent shear and crossing counts.
    ordered = IotaProfile(
        axis_r=profile.axis_r,
        axis_z=profile.axis_z,
        samples=tuple(sorted(profile.samples, key=lambda sample: sample.radial_label)),
    )

    in_band = tuple(
        sample
        for sample in ordered.valid_samples()
        if band_lower <= float(sample.iota) <= band_upper
    )

    max_abs_shear: float | None = None
    for lower, upper in zip(in_band, in_band[1:]):
        label_span = float(upper.radial_label) - float(lower.radial_label)
        if label_span == 0.0:
            continue
        shear = abs(float(upper.iota) - float(lower.iota)) / label_span
        max_abs_shear = shear if max_abs_shear is None else max(max_abs_shear, shear)

    low_order_count = 0
    for q in range(2, max_q + 1):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            for crossing in ordered.all_rational_crossings(p, q):
                if band_lower <= float(crossing.iota) <= band_upper:
                    low_order_count += 1

    low_order_fractions = tuple(
        p / q
        for q in range(2, max_q + 1)
        for p in range(1, q)
        if gcd(p, q) == 1
    )
    min_low_order_distance: float | None = None
    for sample in in_band:
        iota = float(sample.iota)
        # Distance to the nearest n+p/q resonance on ANY integer branch, so the
        # measure is correct even when the band sits on a branch other than n=0.
        nearest = min(
            abs(iota - (round(iota - fraction) + fraction))
            for fraction in low_order_fractions
        )
        min_low_order_distance = (
            nearest
            if min_low_order_distance is None
            else min(min_low_order_distance, nearest)
        )

    return BandShearLoad(
        max_abs_shear=max_abs_shear,
        low_order_count=low_order_count,
        min_low_order_distance=min_low_order_distance,
        in_band_sample_count=len(in_band),
    )


@dataclass(frozen=True, slots=True)
class FrozenRationalTarget:
    """Resolution of one requested rational against a measured ι profile.

    ``in_domain`` is True iff the profile realizes ι = p/q; then ``target`` is the
    requested target with ``radial_label`` set to the crossing and
    ``radial_window`` set to the enclosing traced bracket, ready for the probe.
    Otherwise ``target`` is None and ``reason`` says why (below/above/no crossing).
    """

    requested: RationalTarget
    in_domain: bool
    target: RationalTarget | None
    crossing: RationalCrossing | None
    reason: str


def freeze_rational_targets(
    profile: IotaProfile,
    requested_targets: Sequence[RationalTarget],
) -> tuple[FrozenRationalTarget, ...]:
    """Place each requested target on its realized ι = p/q crossing.

    For an in-domain target the returned target keeps every requested field
    (p, q, weight, branches, phi0, nfp, Fourier metadata) and only gains the
    profile-derived ``radial_label`` and ``radial_window`` (the enclosing
    bracket). Out-of-domain requests are returned unplaced with a reason, so the
    caller reports them rather than driving Newton onto a nonexistent resonance.
    """

    bounds = profile.iota_bounds()
    resolutions: list[FrozenRationalTarget] = []
    for requested in requested_targets:
        crossing = profile.locate_rational(requested.p, requested.q)
        if crossing is not None:
            frozen_target = replace(
                requested,
                radial_label=crossing.radial_label,
                radial_window=(
                    crossing.bracket_lower_label,
                    crossing.bracket_upper_label,
                ),
            )
            resolutions.append(
                FrozenRationalTarget(
                    requested=requested,
                    in_domain=True,
                    target=frozen_target,
                    crossing=crossing,
                    reason=FREEZE_IN_DOMAIN,
                )
            )
            continue
        resolutions.append(
            FrozenRationalTarget(
                requested=requested,
                in_domain=False,
                target=None,
                crossing=None,
                reason=(FREEZE_EMPTY_PROFILE if bounds is None else FREEZE_NO_CROSSING),
            )
        )
    return tuple(resolutions)


@dataclass(frozen=True, slots=True)
class ProbeFreezing:
    """Probe inputs derived from a field's realized ι profile (doc §3.2).

    ``chart`` is re-centered on the located magnetic axis; ``in_domain_targets``
    are the frozen targets whose ι is realized (ready for ``run_residue_probe``);
    ``frozen`` carries the per-request resolution including out-of-domain ones.
    """

    chart: PoincareChart
    profile: IotaProfile
    frozen: tuple[FrozenRationalTarget, ...]
    in_domain_targets: tuple[RationalTarget, ...]


def freeze_probe_inputs(
    field: MagneticFieldLike,
    *,
    requested_targets: Sequence[RationalTarget],
    axis_r_guess: float,
    axis_z_guess: float = 0.0,
    nfp: int,
    locate_axis: bool,
    radial_labels: Sequence[float],
    toroidal_turns: int = DEFAULT_IOTA_PROFILE_TORUS_TURNS,
    poloidal_orientation: int = 1,
    radial_label_scale: float = 1.0,
    axis_search_half_width: float = DEFAULT_AXIS_SEARCH_HALF_WIDTH,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> ProbeFreezing:
    """Derive a re-centered chart and in-domain targets from a field's ι profile.

    When ``locate_axis`` is set the chart axis is the magnetic-axis fixed point
    near the guess (correcting a stale manifest axis); otherwise the guess is
    used as-is. The ι profile is then traced about that axis and the requested
    rationals are frozen onto their realized crossings.
    """

    if locate_axis:
        axis_r, axis_z = magnetic_axis_from_guess(
            field,
            nfp=int(nfp),
            axis_r_guess=float(axis_r_guess),
            axis_z_guess=float(axis_z_guess),
            search_half_width=float(axis_search_half_width),
            min_bphi_over_b=float(integrator_options.min_bphi_over_b),
        )
    else:
        axis_r, axis_z = float(axis_r_guess), float(axis_z_guess)
    profile = sample_iota_profile(
        field,
        axis_r=axis_r,
        axis_z=axis_z,
        radial_labels=radial_labels,
        toroidal_turns=int(toroidal_turns),
        poloidal_orientation=int(poloidal_orientation),
        radial_label_scale=float(radial_label_scale),
        integrator_options=integrator_options,
    )
    frozen = freeze_rational_targets(profile, requested_targets)
    chart = PoincareChart(
        axis_r=axis_r,
        axis_z=axis_z,
        poloidal_orientation=int(poloidal_orientation),
        radial_label_scale=float(radial_label_scale),
    )
    in_domain_targets = tuple(
        resolution.target
        for resolution in frozen
        if resolution.in_domain and resolution.target is not None
    )
    return ProbeFreezing(
        chart=chart,
        profile=profile,
        frozen=frozen,
        in_domain_targets=in_domain_targets,
    )


def iota_profile_payload(profile: IotaProfile) -> dict[str, object]:
    bounds = profile.iota_bounds()
    return {
        "axis_r": float(profile.axis_r),
        "axis_z": float(profile.axis_z),
        "iota_min": None if bounds is None else float(bounds[0]),
        "iota_max": None if bounds is None else float(bounds[1]),
        "samples": [
            {
                "radial_label": float(sample.radial_label),
                "iota": None if sample.iota is None else float(sample.iota),
                "toroidal_turns": int(sample.toroidal_turns),
                "min_bphi_over_b": (
                    None
                    if sample.min_bphi_over_b is None
                    else float(sample.min_bphi_over_b)
                ),
                "reason": sample.reason,
            }
            for sample in profile.samples
        ],
    }


def frozen_target_payload(resolution: FrozenRationalTarget) -> dict[str, object]:
    crossing = resolution.crossing
    return {
        "p": int(resolution.requested.p),
        "q": int(resolution.requested.q),
        "requested_iota": float(resolution.requested.iota_float),
        "in_domain": bool(resolution.in_domain),
        "reason": resolution.reason,
        "radial_label": None if crossing is None else float(crossing.radial_label),
        "radial_window": (
            None
            if crossing is None
            else [
                float(crossing.bracket_lower_label),
                float(crossing.bracket_upper_label),
            ]
        ),
        "local_shear": None if crossing is None else float(crossing.local_shear),
    }


def probe_freezing_payload(freezing: ProbeFreezing) -> dict[str, object]:
    return {
        "axis_r": float(freezing.chart.axis_r),
        "axis_z": float(freezing.chart.axis_z),
        "iota_profile": iota_profile_payload(freezing.profile),
        "frozen_targets": [
            frozen_target_payload(resolution) for resolution in freezing.frozen
        ],
        "in_domain_target_count": int(len(freezing.in_domain_targets)),
    }
