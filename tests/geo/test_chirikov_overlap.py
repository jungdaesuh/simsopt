"""Tests for the Chirikov island-overlap parameter.

The load-bearing check is the standard-map calibration: the overlap formula must
return ``s == 1`` exactly when two unit-spaced resonances' half-widths sum to the
spacing, and scale linearly with width / inversely with spacing. The nontwist
regression pins the shearless-band routing: the pendulum ``1/sqrt(shear)`` must be
abandoned for a finite measured O/X separation rather than diverging.
"""

from __future__ import annotations

import pytest

from examples.single_stage_optimization.banana_opt.topology.chirikov_overlap import (
    CHIRIKOV_OVERLAP_THRESHOLD,
    PENDULUM_WIDTH_COEFFICIENT,
    ChirikovOverlap,
    chirikov_overlap_from_crossings,
    chirikov_overlap_pairwise,
    chirikov_overlap_scalar,
    island_full_width,
    pendulum_full_width,
)
from examples.single_stage_optimization.banana_opt.topology.iota_profile import (
    RationalCrossing,
)


# --------------------------------------------------------------------------- #
# Standard-map calibration of the overlap formula (the load-bearing check).
# --------------------------------------------------------------------------- #
def test_overlap_is_one_when_half_widths_sum_to_spacing():
    # Two resonances unit-spaced; each FULL width 1.0 => half-widths 0.5 each =>
    # half-widths sum exactly to the spacing => s == 1 (classical overlap onset).
    result = chirikov_overlap_scalar([0.0, 1.0], [1.0, 1.0])
    assert result.scalar == pytest.approx(1.0)
    assert result.overlapping
    assert result.scalar >= CHIRIKOV_OVERLAP_THRESHOLD


def test_analytic_pendulum_half_widths_summing_to_spacing_cross_one():
    # The doc's headline calibration end-to-end: drive the ANALYTIC pendulum width
    # (not a literal) into the overlap onset, so a regression in the pendulum
    # coefficient or the A/|s'| form is caught at s == 1 (the previous onset test
    # fed a literal width and would miss such a regression).
    # Two symmetric pendulum islands, spacing 1.0, each A=0.0625, s'=1.0 =>
    # W = 4*sqrt(0.0625/1.0) = 1.0 (half-width 0.5); 0.5 + 0.5 = 1.0 = spacing => s == 1.
    amplitude, shear, spacing = 0.0625, 1.0, 1.0
    width = pendulum_full_width(amplitude, shear)
    assert width == pytest.approx(1.0)
    assert chirikov_overlap_scalar([0.0, spacing], [width, width]).scalar == pytest.approx(1.0)
    # Bracket the crossing through the analytic machinery: +/-10% island width via
    # shear * 1.1**2 (narrower, s<1) and shear / 1.1**2 (wider, s>1).
    width_below = pendulum_full_width(amplitude, shear * 1.21)
    width_above = pendulum_full_width(amplitude, shear / 1.21)
    assert chirikov_overlap_scalar([0.0, spacing], [width_below, width_below]).scalar < 1.0
    assert chirikov_overlap_scalar([0.0, spacing], [width_above, width_above]).scalar > 1.0


def test_overlap_scales_linearly_with_width_and_inversely_with_spacing():
    # Half the widths => half the overlap.
    assert chirikov_overlap_scalar([0.0, 1.0], [0.5, 0.5]).scalar == pytest.approx(0.5)
    # Double the spacing => half the overlap.
    assert chirikov_overlap_scalar([0.0, 2.0], [1.0, 1.0]).scalar == pytest.approx(0.5)
    # Asymmetric widths: (0.5*0.8 + 0.5*0.2)/1.0 = 0.5.
    assert chirikov_overlap_scalar([0.0, 1.0], [0.8, 0.2]).scalar == pytest.approx(0.5)


def test_overlap_below_one_does_not_flag_overlapping():
    result = chirikov_overlap_scalar([0.0, 1.0], [0.4, 0.4])
    assert result.scalar == pytest.approx(0.4)
    assert not result.overlapping


def test_pairwise_uses_worst_neighbour_and_reports_geometry():
    # Three resonances; the tight inner gap overlaps, the wide outer one does not.
    pairs = chirikov_overlap_pairwise([0.0, 1.0, 4.0], [1.2, 1.2, 1.2])
    assert len(pairs) == 2
    assert pairs[0].overlap == pytest.approx(1.2)  # (0.6+0.6)/1.0
    assert pairs[1].overlap == pytest.approx(0.4)  # (0.6+0.6)/3.0
    # max aggregate picks the overlapping inner pair.
    assert chirikov_overlap_scalar([0.0, 1.0, 4.0], [1.2, 1.2, 1.2]).scalar == pytest.approx(1.2)


def test_positions_are_sorted_before_pairing():
    # Unsorted input must give the same ladder as sorted input.
    a = chirikov_overlap_scalar([4.0, 0.0, 1.0], [1.2, 1.2, 1.2]).scalar
    b = chirikov_overlap_scalar([0.0, 1.0, 4.0], [1.2, 1.2, 1.2]).scalar
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# Aggregation: soft-max (objective) upper-bounds the hard max (diagnostic).
# --------------------------------------------------------------------------- #
def test_softmax_upper_bounds_max_and_converges():
    positions = [0.0, 1.0, 4.0]
    widths = [1.2, 1.2, 1.2]
    hard = chirikov_overlap_scalar(positions, widths, aggregate="max").scalar
    soft_loose = chirikov_overlap_scalar(positions, widths, aggregate="softmax", lse_beta=5.0).scalar
    soft_tight = chirikov_overlap_scalar(positions, widths, aggregate="softmax", lse_beta=500.0).scalar
    assert soft_loose >= hard  # LSE is an upper bound on the max
    assert soft_tight >= hard
    assert abs(soft_tight - hard) < abs(soft_loose - hard)  # tighter beta -> closer
    assert soft_tight == pytest.approx(hard, abs=1e-2)


def test_unknown_aggregate_raises():
    with pytest.raises(ValueError, match="aggregate"):
        chirikov_overlap_scalar([0.0, 1.0], [1.0, 1.0], aggregate="median")


# --------------------------------------------------------------------------- #
# Edge cases of the pure core.
# --------------------------------------------------------------------------- #
def test_fewer_than_two_resonances_is_zero_overlap():
    assert chirikov_overlap_scalar([0.5], [1.0]).scalar == 0.0
    assert chirikov_overlap_scalar([], []).scalar == 0.0
    assert chirikov_overlap_pairwise([0.5], [1.0]) == ()


def test_coincident_resonances_raise():
    with pytest.raises(ValueError, match="distinct resonance labels"):
        chirikov_overlap_pairwise([1.0, 1.0], [0.5, 0.5])


def test_non_finite_and_negative_width_raise():
    with pytest.raises(ValueError, match="finite"):
        chirikov_overlap_pairwise([0.0, float("inf")], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        chirikov_overlap_pairwise([0.0, 1.0], [0.5, -0.1])
    with pytest.raises(ValueError, match="equal length"):
        chirikov_overlap_pairwise([0.0, 1.0, 2.0], [0.5, 0.5])


# --------------------------------------------------------------------------- #
# Pendulum width and the nontwist (shearless) routing.
# --------------------------------------------------------------------------- #
def test_pendulum_full_width_matches_closed_form():
    # W = C*sqrt(|A|/|s'|); C=4, A=1, s'=1 -> 4.0; A=1, s'=4 -> 4*sqrt(1/4)=2.0.
    assert pendulum_full_width(1.0, 1.0) == pytest.approx(PENDULUM_WIDTH_COEFFICIENT)
    assert pendulum_full_width(1.0, 4.0) == pytest.approx(2.0)
    # Sign of shear does not matter (uses |shear|).
    assert pendulum_full_width(1.0, -4.0) == pytest.approx(2.0)


def test_pendulum_width_diverges_to_infinity_as_shear_shrinks():
    # The thing the nontwist routing exists to avoid: 1/sqrt(shear) blow-up.
    assert pendulum_full_width(1.0, 1e-2) < pendulum_full_width(1.0, 1e-4)
    with pytest.raises(ZeroDivisionError):
        pendulum_full_width(1.0, 0.0)


def test_island_full_width_uses_pendulum_in_twist_band():
    w = island_full_width(shear=1.0, amplitude=1.0, shear_floor=1e-3)
    assert w == pytest.approx(PENDULUM_WIDTH_COEFFICIENT)


def test_island_full_width_routes_shearless_to_measured_separation():
    # |shear| below the floor: the pendulum would diverge; the measured O/X
    # separation is used instead and stays finite.
    w = island_full_width(shear=1e-9, amplitude=1.0, measured_full_width=0.3, shear_floor=1e-3)
    assert w == pytest.approx(0.3)


def test_island_full_width_shearless_without_measurement_raises():
    # Never fabricate a width in the shearless band.
    with pytest.raises(ValueError, match="shearless band"):
        island_full_width(shear=0.0, amplitude=1.0, shear_floor=1e-3)


def test_island_full_width_measured_overrides_pendulum_in_twist():
    w = island_full_width(shear=1.0, amplitude=1.0, measured_full_width=0.7, shear_floor=1e-3)
    assert w == pytest.approx(0.7)


def test_island_full_width_twist_without_amplitude_or_measurement_raises():
    with pytest.raises(ValueError, match="twist band"):
        island_full_width(shear=1.0, shear_floor=1e-3)


# --------------------------------------------------------------------------- #
# Field-level adapter over RationalCrossing objects.
# --------------------------------------------------------------------------- #
def _crossing(p, q, label, shear):
    return RationalCrossing(
        p=p,
        q=q,
        iota=p / q,
        radial_label=label,
        bracket_lower_label=label - 0.01,
        bracket_upper_label=label + 0.01,
        local_shear=shear,
    )


def test_overlap_from_crossings_composes_positions_and_measured_widths():
    crossings = [
        _crossing(1, 5, 0.0, 0.2),
        _crossing(1, 6, 1.0, 0.2),
        _crossing(1, 7, 4.0, 0.2),
    ]
    measured = [1.2, 1.2, 1.2]
    result = chirikov_overlap_from_crossings(crossings, measured)
    assert isinstance(result, ChirikovOverlap)
    # Same ladder as the pure-core three-resonance case.
    assert result.scalar == pytest.approx(1.2)
    assert result.pairs[0].inner_label == pytest.approx(0.0)
    assert result.pairs[0].outer_label == pytest.approx(1.0)


def test_overlap_from_crossings_requires_measurement_in_shearless_band():
    # A shearless crossing (|local_shear| < floor) with NO measured width must
    # raise -- the pendulum cannot rescue it, and we never fabricate one.
    crossings = [_crossing(1, 5, 0.0, 1e-9), _crossing(1, 6, 1.0, 0.2)]
    with pytest.raises(ValueError, match="shearless band"):
        chirikov_overlap_from_crossings(crossings, [None, 1.0], shear_floor=1e-3)
    # A non-finite measurement is likewise rejected (just with the finiteness msg).
    with pytest.raises(ValueError, match="finite"):
        chirikov_overlap_from_crossings(crossings, [float("nan"), 1.0], shear_floor=1e-3)


def test_overlap_from_crossings_length_mismatch_raises():
    crossings = [_crossing(1, 5, 0.0, 0.2)]
    with pytest.raises(ValueError, match="match in length"):
        chirikov_overlap_from_crossings(crossings, [1.0, 2.0])
