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

# Absolute |ι − p/q| at which a traced sample counts as sitting on the rational
# (e.g. a flat profile pinned to a rational surface). Far below the spacing of
# low-order rationals, so it never conflates neighbouring p/q.
DEFAULT_IOTA_MATCH_TOLERANCE = 1.0e-6

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
    guaranteed to contain the resonance.
    """

    p: int
    q: int
    iota: float
    radial_label: float
    bracket_lower_label: float
    bracket_upper_label: float


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

    def locate_rational(
        self,
        p: int,
        q: int,
        *,
        match_tolerance: float = DEFAULT_IOTA_MATCH_TOLERANCE,
    ) -> RationalCrossing | None:
        """Return where the p/q resonance is realized, or None if out of domain.

        A p/q island chain sits where ι ≡ p/q (mod 1): the full-torus poloidal
        winding about the axis carries an integer geometric/section offset that
        the residue map removes (e.g. a field rotating at nfp + p/q is a p/q
        resonance). So every integer branch ``n + p/q`` inside the traced band is
        a candidate; the innermost (lowest radial label) realized branch is
        returned. ``RationalCrossing.iota`` records the realized transform there
        (which may sit outside [0, 1)). Within a branch, a sample within
        ``match_tolerance`` is treated as on-resonance, else the sign change is
        interpolated.
        """

        valid = self.valid_samples()
        if len(valid) == 0:
            return None
        bounds = self.iota_bounds()
        if bounds is None:
            return None
        minimum_iota, maximum_iota = bounds
        base_fraction = (float(p) / float(q)) % 1.0
        best: RationalCrossing | None = None
        tolerance = float(match_tolerance)
        lowest_branch = int(ceil(minimum_iota - base_fraction - tolerance))
        highest_branch = int(floor(maximum_iota - base_fraction + tolerance))
        for branch in range(lowest_branch, highest_branch + 1):
            crossing = self._locate_transform_value(
                p,
                q,
                base_fraction + float(branch),
                match_tolerance,
            )
            if crossing is not None and (
                best is None or crossing.radial_label < best.radial_label
            ):
                best = crossing
        return best

    def _locate_transform_value(
        self,
        p: int,
        q: int,
        transform_value: float,
        match_tolerance: float,
    ) -> RationalCrossing | None:
        valid = self.valid_samples()
        nearest_index = min(
            range(len(valid)),
            key=lambda index: abs(float(valid[index].iota) - transform_value),
        )
        if abs(float(valid[nearest_index].iota) - transform_value) <= float(
            match_tolerance
        ):
            lower = valid[max(0, nearest_index - 1)]
            upper = valid[min(len(valid) - 1, nearest_index + 1)]
            return RationalCrossing(
                p=int(p),
                q=int(q),
                iota=float(transform_value),
                radial_label=float(valid[nearest_index].radial_label),
                bracket_lower_label=float(lower.radial_label),
                bracket_upper_label=float(upper.radial_label),
            )
        for lower, upper in zip(valid, valid[1:], strict=False):
            lower_offset = float(lower.iota) - transform_value
            upper_offset = float(upper.iota) - transform_value
            if lower_offset * upper_offset < 0.0:
                fraction = lower_offset / (lower_offset - upper_offset)
                label = lower.radial_label + fraction * (
                    upper.radial_label - lower.radial_label
                )
                return RationalCrossing(
                    p=int(p),
                    q=int(q),
                    iota=float(transform_value),
                    radial_label=float(label),
                    bracket_lower_label=float(lower.radial_label),
                    bracket_upper_label=float(upper.radial_label),
                )
        return None

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
