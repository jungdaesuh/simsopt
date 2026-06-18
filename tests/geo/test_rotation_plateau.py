"""Radial ω-plateau discriminator contract.

The single-surface WBA classifier conservatively flags a seed ``island_chain``
whenever its rotation number ω is within 1e-8 of a low-order rational, which
mislabels a nested torus that legitimately sits at that rational. The radial
ω-plateau discriminator reconciles this across neighbouring seeds at different
minor radii:

* a flat ω plateau pinned at p/q across the radial band -> CONFIRM island,
* a monotone ω passing *through* p/q with finite shear -> RECLASSIFY nested torus,
* an isolated seed with no radial neighbours -> keep island (fail-safe).

These are asserted on constructed (radial_label, ω) profiles -- no field trace --
and end-to-end through topology_scorer.invariant_torus_classification, where a
reclassification must drop the island count the strict verdict consumes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.topology.kam_birkhoff import (  # noqa: E402
    DEFAULT_WBA_EXACT_RATIONAL_TOLERANCE,
    KAM_CLASS_CHAOTIC,
    KAM_CLASS_INVARIANT_TORUS,
    KAM_CLASS_ISLAND_CHAIN,
)
from banana_opt.topology.rotation_plateau import (  # noqa: E402
    CLASS_INVARIANT_TORUS,
    CLASS_ISLAND_CHAIN,
    DEFAULT_PLATEAU_RATIONAL_TOLERANCE,
    DEFAULT_PLATEAU_SHEAR_MARGIN,
    RECONCILE_CONFIRMED_ISLAND_PLATEAU,
    RECONCILE_KEPT_ISLAND_AMBIGUOUS_BAND,
    RECONCILE_KEPT_ISLAND_INSUFFICIENT_RADIAL,
    RECONCILE_KEPT_ISLAND_NO_RATIONAL,
    RECONCILE_KEPT_NON_ISLAND,
    RECONCILE_RECLASSIFIED_NESTED_TORUS,
    RadialOmegaSeed,
    _signed_wrapped_offset,
    reconcile_radial_omega_plateau,
)

import topology_scorer  # noqa: E402


# A low-order rational the seeds resonate with (1/3); within the WBA q<=24 band.
P_OVER_Q = 1.0 / 3.0


def _island_seed(radial_label, omega, *, rational=P_OVER_Q):
    return RadialOmegaSeed(
        radial_label=float(radial_label),
        omega=float(omega),
        single_surface_class=CLASS_ISLAND_CHAIN,
        nearest_rational_value=None if rational is None else float(rational),
    )


def _by_label(reconciled):
    return {round(item.radial_label, 12): item for item in reconciled}


def test_class_labels_match_kam_birkhoff_ssot():
    # The pure discriminator carries its own label strings to stay simsopt-free;
    # they must equal the kam_birkhoff SSOT the WBA payload is keyed on.
    assert CLASS_ISLAND_CHAIN == KAM_CLASS_ISLAND_CHAIN
    assert CLASS_INVARIANT_TORUS == KAM_CLASS_INVARIANT_TORUS
    # The plateau tolerance is the classifier's own exact-rational pinning bar.
    assert DEFAULT_PLATEAU_RATIONAL_TOLERANCE == DEFAULT_WBA_EXACT_RATIONAL_TOLERANCE
    # The shear margin must be strictly wider than the pinning bar so the
    # CONFIRM-plateau and RECLASSIFY-straddle branches stay complementary (no
    # fail-open band where a marginal straddle fabricates a torus).
    assert DEFAULT_PLATEAU_SHEAR_MARGIN > 1.0


def test_signed_wrapped_offset_handles_integer_boundary():
    # p/q = 0 (== 1.0 mod 1): ω just below the boundary is a small NEGATIVE
    # offset, just above is a small POSITIVE offset -- not a ~1.0 jump.
    assert _signed_wrapped_offset(0.999_999, 0.0) < 0.0
    assert abs(_signed_wrapped_offset(0.999_999, 0.0)) < 1.0e-5
    assert _signed_wrapped_offset(0.000_001, 0.0) > 0.0
    assert abs(_signed_wrapped_offset(0.000_001, 0.0)) < 1.0e-5
    # Interior rational behaves like a plain signed difference.
    assert _signed_wrapped_offset(P_OVER_Q + 1.0e-3, P_OVER_Q) > 0.0
    assert _signed_wrapped_offset(P_OVER_Q - 1.0e-3, P_OVER_Q) < 0.0


def test_flat_plateau_confirms_island():
    # (a) Three neighbouring seeds all pinned at p/q within tolerance: a flat
    # phase-locked plateau -> the central island is CONFIRMED island.
    eps = DEFAULT_PLATEAU_RATIONAL_TOLERANCE / 10.0
    seeds = [
        _island_seed(0.10, P_OVER_Q - eps),
        _island_seed(0.20, P_OVER_Q + eps),
        _island_seed(0.30, P_OVER_Q - eps),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.reconciled_class == CLASS_ISLAND_CHAIN
    assert center.reason == RECONCILE_CONFIRMED_ISLAND_PLATEAU
    assert center.local_omega_shear is not None
    assert abs(center.local_omega_shear) <= 2.0 * eps / 0.20


def test_monotone_shear_reclassifies_nested_torus():
    # (b) ω rises monotonically through p/q; the central seed sits within 1e-8 of
    # p/q (the single-surface island flag) but its neighbours straddle the
    # rational on opposite sides -> RECLASSIFIED nested torus.
    seeds = [
        _island_seed(0.10, P_OVER_Q - 5.0e-3),
        _island_seed(0.20, P_OVER_Q + 1.0e-9),  # within 1e-8 -> island-flagged
        _island_seed(0.30, P_OVER_Q + 5.0e-3),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.single_surface_class == CLASS_ISLAND_CHAIN
    assert center.reconciled_class == CLASS_INVARIANT_TORUS
    assert center.reason == RECONCILE_RECLASSIFIED_NESTED_TORUS
    # Signed shear is positive (ω rising with radius) and finite.
    assert center.local_omega_shear > 0.0
    # The two edge seeds are not interior -> stay island (conservative).
    assert reconciled[round(0.10, 12)].reconciled_class == CLASS_ISLAND_CHAIN
    assert reconciled[round(0.30, 12)].reconciled_class == CLASS_ISLAND_CHAIN


def test_descending_shear_also_reclassifies():
    # Monotone shear in the other direction (ω falling through p/q) is equally a
    # nested torus: the straddle/monotone test is sign-agnostic.
    seeds = [
        _island_seed(0.10, P_OVER_Q + 5.0e-3),
        _island_seed(0.20, P_OVER_Q - 1.0e-9),
        _island_seed(0.30, P_OVER_Q - 5.0e-3),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.reconciled_class == CLASS_INVARIANT_TORUS
    assert center.local_omega_shear < 0.0


def test_isolated_seed_is_not_rescued():
    # (c) A single island seed at p/q with no radial neighbours: insufficient
    # radial evidence -> stay island (fail-safe), never fabricate a torus.
    seeds = [_island_seed(0.20, P_OVER_Q + 1.0e-9)]
    reconciled = reconcile_radial_omega_plateau(seeds)
    assert len(reconciled) == 1
    assert reconciled[0].reconciled_class == CLASS_ISLAND_CHAIN
    assert reconciled[0].reason == RECONCILE_KEPT_ISLAND_INSUFFICIENT_RADIAL
    assert reconciled[0].local_omega_shear is None


def test_edge_island_with_one_sided_neighbours_is_not_rescued():
    # An island at the inner edge has only an outward neighbour -> no interior
    # band -> conservative keep island, even if the outward trend looks sheared.
    seeds = [
        _island_seed(0.10, P_OVER_Q + 1.0e-9),  # inner edge, island-flagged
        _island_seed(0.20, P_OVER_Q + 5.0e-3),
        _island_seed(0.30, P_OVER_Q + 1.0e-2),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    edge = reconciled[round(0.10, 12)]
    assert edge.reconciled_class == CLASS_ISLAND_CHAIN
    assert edge.reason == RECONCILE_KEPT_ISLAND_INSUFFICIENT_RADIAL


def test_same_side_neighbours_are_ambiguous_and_kept_island():
    # Both neighbours on the SAME side of p/q (no straddle) and not a plateau:
    # ambiguous radial band -> conservative keep island.
    seeds = [
        _island_seed(0.10, P_OVER_Q + 2.0e-3),
        _island_seed(0.20, P_OVER_Q + 1.0e-9),
        _island_seed(0.30, P_OVER_Q + 4.0e-3),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.reconciled_class == CLASS_ISLAND_CHAIN
    assert center.reason == RECONCILE_KEPT_ISLAND_AMBIGUOUS_BAND
    # Shear evidence is still surfaced even when the verdict is conservative.
    assert center.local_omega_shear is not None


def test_marginal_straddle_within_guard_band_is_kept_island():
    # REGRESSION (fail-open): a genuine phase-locked island whose two surviving
    # radial neighbours scatter onto OPPOSITE sides of p/q by only ~2x the
    # pinning tolerance (center seed ~ p/q, so the straddle is monotone). This
    # is NOT shear -- it is few-ULP scatter around the resonance -- yet it
    # straddles. The old code (no guard band) RECLASSIFIED it to invariant_torus,
    # fabricating a torus from a real island. With the guard band the offsets all
    # sit in [tol, k*tol], so the verdict must stay conservative island_chain.
    eps = 2.0 * DEFAULT_PLATEAU_RATIONAL_TOLERANCE
    assert eps < DEFAULT_PLATEAU_SHEAR_MARGIN * DEFAULT_PLATEAU_RATIONAL_TOLERANCE
    seeds = [
        _island_seed(0.10, P_OVER_Q - eps),
        _island_seed(0.20, P_OVER_Q),
        _island_seed(0.30, P_OVER_Q + eps),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.reconciled_class == CLASS_ISLAND_CHAIN
    assert center.reason == RECONCILE_KEPT_ISLAND_AMBIGUOUS_BAND


def test_exact_ulp_edge_straddle_is_kept_island():
    # REGRESSION (fail-open ULP edge): neighbours exactly at p/q +- tol on the
    # exactly-representable rational 1/2. The mod-1 wrap in _signed_wrapped_offset
    # ((raw + 0.5) % 1.0 - 0.5) makes |offset(1/2 + tol, 1/2)| exceed tol by one
    # ULP while |offset(1/2 - tol, 1/2)| stays within it, so the plateau branch
    # fails OPEN and -- on the old code -- the asymmetric straddle fired,
    # fabricating a torus right at the pinning bar. The guard band absorbs this:
    # offsets ~ tol are far below k*tol -> keep island. (1/2 is used rather than
    # 1/3 because 1/3 is not exactly representable, which rounds both +-tol
    # neighbours back inside tol and lands them on the plateau branch instead.)
    half = 0.5
    tol = DEFAULT_PLATEAU_RATIONAL_TOLERANCE
    # Sanity: this profile really does evade the plateau branch and trip the
    # straddle on the old logic (one offset within tol, one one ULP over).
    assert abs(_signed_wrapped_offset(half - tol, half)) <= tol
    assert abs(_signed_wrapped_offset(half + tol, half)) > tol
    seeds = [
        _island_seed(0.10, half - tol, rational=half),
        _island_seed(0.20, half, rational=half),
        _island_seed(0.30, half + tol, rational=half),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.reconciled_class == CLASS_ISLAND_CHAIN
    assert center.reason == RECONCILE_KEPT_ISLAND_AMBIGUOUS_BAND


def test_genuine_shear_above_guard_band_still_reclassifies():
    # The guard band must not suppress a real nested torus: one neighbour a full
    # guard-band margin (k*tol) clear of p/q on its side is unambiguous shear and
    # must still RECLASSIFY. Pin the threshold-boundary behaviour (>= is
    # inclusive) so the floor cannot silently swallow a true torus.
    threshold = DEFAULT_PLATEAU_SHEAR_MARGIN * DEFAULT_PLATEAU_RATIONAL_TOLERANCE
    seeds = [
        _island_seed(0.10, P_OVER_Q - threshold),
        _island_seed(0.20, P_OVER_Q),
        _island_seed(0.30, P_OVER_Q + DEFAULT_PLATEAU_RATIONAL_TOLERANCE / 10.0),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.reconciled_class == CLASS_INVARIANT_TORUS
    assert center.reason == RECONCILE_RECLASSIFIED_NESTED_TORUS


def test_shear_margin_at_or_below_one_is_rejected():
    # A guard band <= the pinning bar reopens the fail-open band, so it is
    # rejected rather than silently degrading to the old non-complementary logic.
    seeds = [_island_seed(0.20, P_OVER_Q)]
    for bad in (1.0, 0.5, 0.0):
        try:
            reconcile_radial_omega_plateau(seeds, shear_margin=bad)
        except ValueError:
            continue
        raise AssertionError(f"shear_margin={bad} should have raised ValueError")


def test_island_without_rational_is_kept():
    # An island-flagged seed with no pinned rational cannot be radially
    # reconciled -> conservative keep island.
    seeds = [
        _island_seed(0.10, P_OVER_Q - 5.0e-3),
        _island_seed(0.20, P_OVER_Q, rational=None),
        _island_seed(0.30, P_OVER_Q + 5.0e-3),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    center = reconciled[round(0.20, 12)]
    assert center.reconciled_class == CLASS_ISLAND_CHAIN
    assert center.reason == RECONCILE_KEPT_ISLAND_NO_RATIONAL
    assert center.local_omega_shear is None


def test_non_island_seeds_pass_through_unchanged():
    seeds = [
        RadialOmegaSeed(0.10, 0.21, KAM_CLASS_INVARIANT_TORUS, None),
        RadialOmegaSeed(0.20, 0.40, KAM_CLASS_CHAOTIC, None),
    ]
    reconciled = _by_label(reconcile_radial_omega_plateau(seeds))
    torus = reconciled[round(0.10, 12)]
    chaotic = reconciled[round(0.20, 12)]
    assert torus.reconciled_class == KAM_CLASS_INVARIANT_TORUS
    assert torus.reason == RECONCILE_KEPT_NON_ISLAND
    assert chaotic.reconciled_class == KAM_CLASS_CHAOTIC
    assert chaotic.reason == RECONCILE_KEPT_NON_ISLAND


def test_result_preserves_input_order():
    # Deliberately unsorted radial labels: result must come back in input order
    # so the caller can zip it positionally onto the per-seed payload.
    seeds = [
        _island_seed(0.30, P_OVER_Q + 5.0e-3),
        _island_seed(0.10, P_OVER_Q - 5.0e-3),
        _island_seed(0.20, P_OVER_Q + 1.0e-9),
    ]
    reconciled = reconcile_radial_omega_plateau(seeds)
    assert [item.radial_label for item in reconciled] == [0.30, 0.10, 0.20]
    # The middle input (radial 0.20) is the interior seed that gets rescued.
    assert reconciled[2].reconciled_class == CLASS_INVARIANT_TORUS


# ---------------------------------------------------------------------------
# End-to-end through topology_scorer.invariant_torus_classification
# ---------------------------------------------------------------------------


def _quasiperiodic_hits(*, axis_r, axis_z, rotation_number, count, minor=0.08):
    theta = np.arange(count, dtype=float) * 2.0 * math.pi * float(rotation_number)
    return np.column_stack(
        (
            np.arange(theta.size, dtype=float),
            np.zeros(theta.size, dtype=float),
            float(axis_r) + minor * np.cos(theta),
            np.zeros(theta.size, dtype=float),
            float(axis_z) + minor * np.sin(theta),
        )
    )


class _RingSurface:
    nfp = 1


def test_invariant_torus_classification_reclassifies_sheared_island_end_to_end():
    # Three classifiable field lines whose WBA rotation numbers rise
    # monotonically through 1/3, the middle one landing exactly on it (the
    # single-surface island flag). With radial labels supplied, the discriminator
    # must rescue the middle line -> island_chain_count drops to 0 and the
    # invariant-torus fraction rises to 1.0; without labels it stays an island.
    axis_r, axis_z = 1.05, 0.0
    axis_point = {"r": axis_r, "z": axis_z, "source": "test_magnetic_axis"}
    inner = _quasiperiodic_hits(
        axis_r=axis_r, axis_z=axis_z, rotation_number=P_OVER_Q - 3.0e-3, count=300
    )
    middle = _quasiperiodic_hits(
        axis_r=axis_r, axis_z=axis_z, rotation_number=P_OVER_Q, count=300
    )
    outer = _quasiperiodic_hits(
        axis_r=axis_r, axis_z=axis_z, rotation_number=P_OVER_Q + 3.0e-3, count=300
    )
    fieldlines = [inner, middle, outer]
    labels = [0.04, 0.06, 0.08]

    without_labels = topology_scorer.invariant_torus_classification(
        fieldlines, _RingSurface(), axis_point=axis_point
    )
    # The middle line is single-surface-flagged island (ω == 1/3 exactly).
    assert without_labels["wba_classification_counts"][KAM_CLASS_ISLAND_CHAIN] == 1

    with_labels = topology_scorer.invariant_torus_classification(
        fieldlines, _RingSurface(), axis_point=axis_point, seed_radial_labels=labels
    )
    counts = with_labels["wba_classification_counts"]
    assert counts[KAM_CLASS_ISLAND_CHAIN] == 0
    assert counts[KAM_CLASS_INVARIANT_TORUS] == 3
    assert with_labels["invariant_torus_fraction"] == 1.0
    # The rescued seed records the reconciliation in its reason audit trail.
    rescued = [
        item
        for item in with_labels["wba_seed_classifications"]
        if "radial_omega_plateau" in item["reason"]
    ]
    assert len(rescued) == 1
    assert RECONCILE_RECLASSIFIED_NESTED_TORUS in rescued[0]["reason"]


def test_invariant_torus_classification_confirms_island_plateau_end_to_end():
    # Three field lines all phase-locked at 1/3 across the radial band: a flat ω
    # plateau -> the discriminator CONFIRMS all three island, never rescues them.
    axis_r, axis_z = 1.05, 0.0
    axis_point = {"r": axis_r, "z": axis_z, "source": "test_magnetic_axis"}
    fieldlines = [
        _quasiperiodic_hits(
            axis_r=axis_r, axis_z=axis_z, rotation_number=P_OVER_Q, count=300
        )
        for _ in range(3)
    ]
    labels = [0.04, 0.06, 0.08]

    result = topology_scorer.invariant_torus_classification(
        fieldlines, _RingSurface(), axis_point=axis_point, seed_radial_labels=labels
    )
    counts = result["wba_classification_counts"]
    assert counts[KAM_CLASS_ISLAND_CHAIN] == 3
    assert counts[KAM_CLASS_INVARIANT_TORUS] == 0
    confirmed = [
        item
        for item in result["wba_seed_classifications"]
        if RECONCILE_CONFIRMED_ISLAND_PLATEAU in item["reason"]
    ]
    # The interior seed (middle radius) is the one with a full radial band.
    assert len(confirmed) == 1
