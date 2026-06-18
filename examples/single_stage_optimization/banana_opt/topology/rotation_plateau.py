"""Radial ω-plateau discriminator that rescues nested tori the single-surface
WBA classifier conservatively flagged as island chains.

Interface: consume a radially-ordered sequence of per-seed
``(radial_label, omega, single_surface_class, nearest_rational_value)`` -- where
``omega`` is the WBA rotation number (mod 1) and ``single_surface_class`` is the
single-surface label -- and return, per seed, a reconciled class + a reason:
CONFIRM island when ω is a flat phase-locked plateau pinned at p/q across the
radial band (both neighbours on-resonance, within the pinning tolerance,
|dω/dr|≈0); RECLASSIFY nested torus only when ω passes monotonically *through*
p/q with GENUINE radial shear (neighbours straddle the rational on opposite
sides AND at least one neighbour offset is a full guard-band margin -- k·tol --
clear of the pinning bar); else CONSERVATIVE fallback = keep the island label.
The conservative branch deliberately covers the ambiguous guard band in which
neighbours straddle p/q but only marginally (every offset in [tol, k·tol]): too
sheared to be a clean plateau, too thin to be trusted shear -- so the fail-safe
keeps the island rather than fabricate a torus from a few-ULP scatter.

Physics. A single field line resolves ω at one minor radius, so a genuine nested
torus sitting *at* a low-order rational is indistinguishable, on that one
surface, from a p/q island chain. The distinguishing feature is radial: a true
island is phase-locked, so ω is flat (dω/dr ≈ 0) across a band of neighbouring
radii all sharing the island's rational; a nested family has finite shear, so ω
varies monotonically with radius and merely *passes through* the rational. This
module makes that radial decision and rewrites the per-seed class accordingly.

This mirrors the signed local-shear idea in
``iota_profile.RationalCrossing.local_shear`` (dι/dlabel across an adjacent
bracket), but the quantity here is the WBA rotation number ω, which is reduced
mod 1, not the full-winding Boozer/return-map ι. Mod-1 ω differences are
therefore taken on the wrapped circle (signed offsets in (-0.5, 0.5]), where
ι differencing in ``iota_profile`` is plain subtraction of an unreduced winding.
The two rotation numbers are kept deliberately distinct.

Pure numpy: no simsopt import, so the discriminator is cheap to unit-test on
constructed (radial_label, ω) profiles exactly like the WBA core.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


# Single-surface class labels this discriminator reconciles. Kept as module-local
# string constants so the pure discriminator does not import the WBA core; the
# wiring site (topology_scorer) is responsible for passing the matching labels
# (asserted equal to kam_birkhoff.KAM_CLASS_* by the contract test).
CLASS_ISLAND_CHAIN = "island_chain"
CLASS_INVARIANT_TORUS = "invariant_torus"

# Reconciliation verdicts attached to each reconciled seed.
RECONCILE_CONFIRMED_ISLAND_PLATEAU = "confirmed_island_radial_omega_plateau"
RECONCILE_RECLASSIFIED_NESTED_TORUS = "reclassified_nested_torus_radial_omega_shear"
RECONCILE_KEPT_NON_ISLAND = "non_island_class_unchanged"
RECONCILE_KEPT_ISLAND_NO_RATIONAL = "kept_island_no_nearest_rational"
RECONCILE_KEPT_ISLAND_INSUFFICIENT_RADIAL = (
    "kept_island_insufficient_radial_neighbours"
)
RECONCILE_KEPT_ISLAND_AMBIGUOUS_BAND = "kept_island_ambiguous_radial_band"

# |ω − p/q| (on the wrapped circle) at or below which a neighbouring seed counts
# as sitting *on* the same rational. A flat plateau requires both radial
# neighbours pinned this close. Set equal to the single-surface classifier's own
# exact-rational tolerance (kam_birkhoff.DEFAULT_WBA_EXACT_RATIONAL_TOLERANCE =
# 1e-8): the seed under test was flagged island precisely because its ω was
# within that tolerance of p/q, so a neighbour is "on the same plateau" only when
# it meets the identical pinning bar. Replicated here as a module constant rather
# than imported so the discriminator stays simsopt-free; the contract test
# asserts the two are equal so they cannot drift.
DEFAULT_PLATEAU_RATIONAL_TOLERANCE = 1.0e-8

# Guard-band factor between the plateau-pinning tolerance and the straddle
# RECLASSIFY threshold. The CONFIRM-island plateau branch pins both neighbours
# within ``plateau_rational_tolerance`` (tol); the RECLASSIFY-to-nested-torus
# straddle branch must, conversely, see *unambiguous* radial shear -- at least
# one neighbour offset >= k * tol -- before it overrides the conservative
# single-surface island verdict. Without this floor the two branches are not
# complementary: a genuine phase-locked island whose two surviving lines scatter
# a few * tol onto OPPOSITE sides of p/q (center seed ~ p/q, so monotone holds)
# fails the plateau test yet trips the straddle test and is silently fabricated
# into a torus. The intervening band (both offsets in [tol, k*tol]) is ambiguous
# radial evidence -> stay CONSERVATIVE = keep island, never invent a torus.
#
# k = 10 places a clean decade between "pinned on the rational" (<= tol) and
# "unambiguously sheared off it" (>= k * tol). A full decade is justified: it is
# many orders of magnitude clear of the worst-case ULP asymmetry of the mod-1
# wrap in ``_signed_wrapped_offset`` (the (raw + 0.5) % 1.0 - 0.5 reduction makes
# |offset(p/q +- tol, p/q)| differ from tol by only ~1e-16 relative), so an
# exactly-tol neighbour can never bleed across the floor; and it matches the
# order-of-magnitude "on- vs off-resonance" separation the WBA pinning already
# relies on. Asserted positive and > 1 so the guard band cannot collapse.
DEFAULT_PLATEAU_SHEAR_MARGIN = 10.0
assert DEFAULT_PLATEAU_SHEAR_MARGIN > 1.0


def _signed_wrapped_offset(omega: float, rational_value: float) -> float:
    """Return ω − p/q on the wrapped circle, signed, in (-0.5, 0.5].

    ω is reduced mod 1, so a rational near an integer boundary (e.g. p/q = 0/1 ≡
    1.0) is approached from ω ≈ 0.9999 (below) or ω ≈ 0.0001 (above); a plain
    subtraction would read those as ~1 apart. The wrapped signed difference keeps
    the sign that says which radial side of the resonance the seed sits on, which
    is exactly what the straddle test below keys on.
    """

    raw = float(omega) - float(rational_value)
    return (raw + 0.5) % 1.0 - 0.5


@dataclass(frozen=True, slots=True)
class RadialOmegaSeed:
    """One WBA seed positioned radially for the plateau discriminator.

    ``radial_label`` is the seed's launch radius / mean minor radius (the same
    label topology_scorer seeds the midplane sweep on). ``omega`` is the WBA
    rotation number (mod 1). ``single_surface_class`` is the single-surface WBA
    label. ``nearest_rational_value`` is the p/q (as a float in [0, 1)) the
    single-surface classifier locked the island onto -- ``None`` for a seed that
    has no rational (e.g. an invariant_torus or a seed the classifier could not
    pin), in which case an island-flagged seed cannot be radially reconciled.
    """

    radial_label: float
    omega: float
    single_surface_class: str
    nearest_rational_value: float | None


@dataclass(frozen=True, slots=True)
class ReconciledSeed:
    """Reconciled verdict for one seed.

    ``reconciled_class`` is the post-discriminator class (island_chain unless a
    seed was rescued to invariant_torus); ``reason`` is the verdict constant;
    ``local_omega_shear`` is the signed dω/dr across the seed's radial neighbours
    (``None`` when no interior band was available), mirroring iota_profile's
    signed local_shear as the evidence behind the verdict.
    """

    radial_label: float
    omega: float
    single_surface_class: str
    reconciled_class: str
    reason: str
    local_omega_shear: float | None


def reconcile_radial_omega_plateau(
    seeds: Sequence[RadialOmegaSeed],
    *,
    plateau_rational_tolerance: float = DEFAULT_PLATEAU_RATIONAL_TOLERANCE,
    shear_margin: float = DEFAULT_PLATEAU_SHEAR_MARGIN,
) -> tuple[ReconciledSeed, ...]:
    """Reconcile single-surface WBA classes using the radial ω structure.

    Sorts seeds by ``radial_label`` and, for each ``island_chain``-flagged seed,
    inspects its immediate radial neighbours (the next seed inward and outward):

    * **RECLASSIFY nested torus** when the neighbours straddle the seed's
      rational -- their signed wrapped offsets ω − p/q have opposite signs, ω is
      monotone across the band, AND at least one neighbour is a full guard-band
      margin (``shear_margin`` × ``plateau_rational_tolerance``) clear of the
      pinning bar -- i.e. ω passes *through* p/q with GENUINE shear, the
      signature of a sheared nested family rather than a phase-locked island.
    * **CONFIRM island** when both neighbours are themselves pinned within
      ``plateau_rational_tolerance`` of the same p/q -- a flat ω plateau, dω/dr ≈
      0, the signature of a phase-locked island chain.
    * **KEEP island (conservative)** otherwise: a seed with no nearest rational,
      an edge seed lacking an interior radial band, or an ambiguous band (same-
      side neighbours, non-monotone ω, or a straddle whose neighbour offsets all
      sit in the [tol, k·tol] guard band, too marginal to be trusted shear). The
      discriminator never invents a torus on thin radial evidence -- the fail-safe
      is to leave the conservative single-surface island verdict in place.

    Non-island seeds pass through unchanged. The result is in the input order
    (not the internal radial sort), so the caller can zip it back onto its
    per-seed payload positionally.
    """

    tolerance = float(plateau_rational_tolerance)
    if tolerance < 0.0:
        raise ValueError("plateau_rational_tolerance must be non-negative")
    margin = float(shear_margin)
    if margin <= 1.0:
        # The guard band must be strictly wider than the pinning bar, else the
        # CONFIRM-plateau and RECLASSIFY-straddle branches stop being
        # complementary and the fail-open band reopens.
        raise ValueError("shear_margin must be greater than 1")
    shear_threshold = margin * tolerance
    ordered = sorted(
        range(len(seeds)),
        key=lambda position: float(seeds[position].radial_label),
    )
    # rank[input_position] -> index within the radially-sorted ordering.
    rank = {position: order for order, position in enumerate(ordered)}
    results: list[ReconciledSeed | None] = [None] * len(seeds)
    for input_position, seed in enumerate(seeds):
        if seed.single_surface_class != CLASS_ISLAND_CHAIN:
            results[input_position] = ReconciledSeed(
                radial_label=float(seed.radial_label),
                omega=float(seed.omega),
                single_surface_class=seed.single_surface_class,
                reconciled_class=seed.single_surface_class,
                reason=RECONCILE_KEPT_NON_ISLAND,
                local_omega_shear=None,
            )
            continue
        results[input_position] = _reconcile_island_seed(
            seed,
            ordered=ordered,
            seeds=seeds,
            sorted_index=rank[input_position],
            tolerance=tolerance,
            shear_threshold=shear_threshold,
        )
    return tuple(result for result in results if result is not None)


def _reconcile_island_seed(
    seed: RadialOmegaSeed,
    *,
    ordered: Sequence[int],
    seeds: Sequence[RadialOmegaSeed],
    sorted_index: int,
    tolerance: float,
    shear_threshold: float,
) -> ReconciledSeed:
    rational_value = seed.nearest_rational_value
    if rational_value is None:
        return _kept_island(seed, RECONCILE_KEPT_ISLAND_NO_RATIONAL, shear=None)

    # An interior seed needs a neighbour on each radial side to define a band.
    if sorted_index == 0 or sorted_index == len(ordered) - 1:
        return _kept_island(
            seed, RECONCILE_KEPT_ISLAND_INSUFFICIENT_RADIAL, shear=None
        )
    lower = seeds[ordered[sorted_index - 1]]
    upper = seeds[ordered[sorted_index + 1]]
    label_span = float(upper.radial_label) - float(lower.radial_label)
    if label_span <= 0.0:
        # Degenerate / duplicate radial labels: no resolvable band -> conservative.
        return _kept_island(
            seed, RECONCILE_KEPT_ISLAND_INSUFFICIENT_RADIAL, shear=None
        )

    lower_offset = _signed_wrapped_offset(lower.omega, float(rational_value))
    upper_offset = _signed_wrapped_offset(upper.omega, float(rational_value))
    seed_offset = _signed_wrapped_offset(seed.omega, float(rational_value))
    # Signed dω/dr across the band, on the wrapped circle: the difference of the
    # neighbours' signed offsets equals the wrapped ω difference, divided by the
    # radial span. Mirrors iota_profile's local_shear, on ω instead of ι.
    local_shear = (upper_offset - lower_offset) / label_span

    # Flat phase-locked plateau: both neighbours pinned on the same rational.
    if abs(lower_offset) <= tolerance and abs(upper_offset) <= tolerance:
        return ReconciledSeed(
            radial_label=float(seed.radial_label),
            omega=float(seed.omega),
            single_surface_class=seed.single_surface_class,
            reconciled_class=CLASS_ISLAND_CHAIN,
            reason=RECONCILE_CONFIRMED_ISLAND_PLATEAU,
            local_omega_shear=float(local_shear),
        )

    # Monotone GENUINE shear through the rational: neighbours on opposite radial
    # sides of p/q, ω monotone across the band, AND at least one neighbour an
    # unambiguous guard-band margin clear of the pinning bar -> a sheared nested
    # torus passing through the resonance, not a phase-locked island. The
    # ``genuine_shear`` floor is what makes this branch complementary to the
    # CONFIRM-plateau branch above: a marginal straddle whose offsets all sit in
    # the [tolerance, shear_threshold] guard band is too thin to trust and falls
    # through to the conservative keep-island fallback rather than fabricating a
    # torus from a few-ULP scatter around p/q.
    straddles = lower_offset * upper_offset < 0.0
    monotone = (lower_offset < seed_offset < upper_offset) or (
        lower_offset > seed_offset > upper_offset
    )
    genuine_shear = max(abs(lower_offset), abs(upper_offset)) >= shear_threshold
    if straddles and monotone and genuine_shear:
        return ReconciledSeed(
            radial_label=float(seed.radial_label),
            omega=float(seed.omega),
            single_surface_class=seed.single_surface_class,
            reconciled_class=CLASS_INVARIANT_TORUS,
            reason=RECONCILE_RECLASSIFIED_NESTED_TORUS,
            local_omega_shear=float(local_shear),
        )

    return _kept_island(
        seed, RECONCILE_KEPT_ISLAND_AMBIGUOUS_BAND, shear=float(local_shear)
    )


def _kept_island(
    seed: RadialOmegaSeed,
    reason: str,
    *,
    shear: float | None,
) -> ReconciledSeed:
    return ReconciledSeed(
        radial_label=float(seed.radial_label),
        omega=float(seed.omega),
        single_surface_class=seed.single_surface_class,
        reconciled_class=CLASS_ISLAND_CHAIN,
        reason=reason,
        local_omega_shear=shear,
    )
