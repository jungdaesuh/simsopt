"""Behavioral tests for the realized-iota-profile target/window/axis selection.

These assert observable selection behavior against closed-form iota profiles:
- iota(r) is measured correctly from a known radial-shear field,
- a rational p/q is located at the radial label where iota actually equals p/q,
- a rational whose iota is out of the traced band is reported out-of-domain
  (never placed), and
- freezing re-centers the chart on the located magnetic axis.

The synthetic CircularTransformField (constant iota) and RadialShearTransformField
(iota = reference_iota + shear*(minor_radius - reference_minor_radius)) are the
suite's canonical analytic fields, reused so the profile is checked against the
same winding convention the residue probe assumes.
"""

from __future__ import annotations

import pytest

from geo.test_greene_residue_conventions import (
    CircularTransformField,
    RadialShearTransformField,
)

from examples.single_stage_optimization.banana_opt.topology.fieldline_map import (
    FieldlineIntegratorOptions,
)
from examples.single_stage_optimization.banana_opt.topology.iota_profile import (
    FREEZE_IN_DOMAIN,
    FREEZE_NO_CROSSING,
    freeze_probe_inputs,
    freeze_rational_targets,
    linear_radial_labels,
    sample_iota_profile,
)
from examples.single_stage_optimization.banana_opt.topology.rational_target import (
    GREENE_BRANCH_O,
    RationalTarget,
)


_INTEGRATOR_OPTIONS = FieldlineIntegratorOptions(
    rtol=1.0e-10,
    atol=1.0e-12,
    max_step=0.025,
    samples_per_full_torus=64,
)
_PROFILE_TURNS = 8
_REFERENCE_IOTA = 0.25
_REFERENCE_MINOR_RADIUS = 0.2
_SHEAR = 0.8


def _radial_shear_field() -> RadialShearTransformField:
    return RadialShearTransformField(
        axis_r=1.0,
        axis_z=0.0,
        reference_iota=_REFERENCE_IOTA,
        reference_minor_radius=_REFERENCE_MINOR_RADIUS,
        shear=_SHEAR,
    )


def _closed_form_iota(radial_label: float) -> float:
    return _REFERENCE_IOTA + _SHEAR * (radial_label - _REFERENCE_MINOR_RADIUS)


def _closed_form_crossing_label(target_iota: float) -> float:
    return _REFERENCE_MINOR_RADIUS + (target_iota - _REFERENCE_IOTA) / _SHEAR


def _profile_over(lower: float, upper: float, count: int):
    return sample_iota_profile(
        _radial_shear_field(),
        axis_r=1.0,
        axis_z=0.0,
        radial_labels=linear_radial_labels(lower=lower, upper=upper, count=count),
        toroidal_turns=_PROFILE_TURNS,
        integrator_options=_INTEGRATOR_OPTIONS,
    )


def test_sample_iota_profile_matches_radial_shear_closed_form():
    profile = _profile_over(0.10, 0.40, 7)
    assert len(profile.valid_samples()) == 7
    for sample in profile.valid_samples():
        assert sample.iota == pytest.approx(
            _closed_form_iota(sample.radial_label), abs=1.0e-6
        )
    minimum_iota, maximum_iota = profile.iota_bounds()
    assert minimum_iota == pytest.approx(_closed_form_iota(0.10), abs=1.0e-6)
    assert maximum_iota == pytest.approx(_closed_form_iota(0.40), abs=1.0e-6)


def test_locate_rational_returns_interpolated_crossing():
    profile = _profile_over(0.10, 0.40, 7)
    # 1/4 = 0.25 falls on the analytic crossing at the reference minor radius 0.2.
    crossing_quarter = profile.locate_rational(1, 4)
    assert crossing_quarter is not None
    assert crossing_quarter.radial_label == pytest.approx(0.2, abs=1.0e-4)
    # 3/10 = 0.30 falls strictly between samples and must be interpolated.
    crossing_three_tenths = profile.locate_rational(3, 10)
    assert crossing_three_tenths is not None
    assert crossing_three_tenths.radial_label == pytest.approx(
        _closed_form_crossing_label(0.3), abs=2.0e-3
    )
    assert (
        crossing_three_tenths.bracket_lower_label
        < crossing_three_tenths.radial_label
        < crossing_three_tenths.bracket_upper_label
    )


def test_locate_rational_out_of_domain_returns_none():
    profile = _profile_over(0.10, 0.40, 7)
    minimum_iota, maximum_iota = profile.iota_bounds()
    assert 0.5 > maximum_iota
    assert 0.1 < minimum_iota
    assert profile.locate_rational(1, 2) is None  # iota 0.5 above the band
    assert profile.locate_rational(1, 10) is None  # iota 0.1 below the band


def test_in_domain_rationals_enumerates_only_realized_fractions():
    profile = _profile_over(0.10, 0.40, 9)
    minimum_iota, maximum_iota = profile.iota_bounds()
    crossings = profile.in_domain_rationals(max_denominator=10)
    assert len(crossings) > 0
    fractions = {(crossing.p, crossing.q) for crossing in crossings}
    assert (1, 4) in fractions  # 0.25 is in band
    assert (1, 2) not in fractions  # 0.5 is above band
    labels = [crossing.radial_label for crossing in crossings]
    assert labels == sorted(labels)
    for crossing in crossings:
        assert minimum_iota <= crossing.iota <= maximum_iota


def test_freeze_rational_targets_places_window_on_crossing():
    profile = _profile_over(0.10, 0.40, 7)
    in_domain_request = RationalTarget(p=1, q=4, branches=(GREENE_BRANCH_O,))
    out_of_domain_request = RationalTarget(p=1, q=2, branches=(GREENE_BRANCH_O,))

    frozen = freeze_rational_targets(
        profile, (in_domain_request, out_of_domain_request)
    )

    placed, rejected = frozen
    assert placed.in_domain is True
    assert placed.reason == FREEZE_IN_DOMAIN
    assert placed.target is not None
    assert placed.target.radial_label == pytest.approx(0.2, abs=1.0e-4)
    lower, upper = placed.target.radial_window
    assert lower <= placed.target.radial_label <= upper
    # Requested fields other than label/window are preserved.
    assert placed.target.p == 1 and placed.target.q == 4
    assert placed.target.branches == (GREENE_BRANCH_O,)

    assert rejected.in_domain is False
    assert rejected.target is None
    assert rejected.reason == FREEZE_NO_CROSSING


def test_freeze_rational_targets_flags_out_of_band():
    profile = _profile_over(0.10, 0.40, 7)
    frozen = freeze_rational_targets(
        profile, (RationalTarget(p=1, q=10, branches=(GREENE_BRANCH_O,)),)
    )
    assert frozen[0].in_domain is False
    assert frozen[0].reason == FREEZE_NO_CROSSING


def test_locate_rational_matches_resonance_in_higher_integer_branch():
    # A field rotating at iota = 1 + 1/4 = 1.25 realizes the 1/4 resonance in
    # the n=1 branch (iota ≡ 1/4 mod 1), not at 0.25: the integer geometric
    # offset is what the residue map removes (residue convention SSOT).
    field = CircularTransformField(
        axis_r=1.0, axis_z=0.0, iota=0.25, geometric_winding_offset=1.0
    )
    profile = sample_iota_profile(
        field,
        axis_r=1.0,
        axis_z=0.0,
        radial_labels=linear_radial_labels(lower=0.05, upper=0.20, count=4),
        toroidal_turns=_PROFILE_TURNS,
        integrator_options=_INTEGRATOR_OPTIONS,
    )
    minimum_iota, maximum_iota = profile.iota_bounds()
    assert minimum_iota == pytest.approx(1.25, abs=1.0e-4)

    crossing = profile.locate_rational(1, 4)
    assert crossing is not None
    assert crossing.iota == pytest.approx(1.25, abs=1.0e-4)
    # 1/4 with no in-band branch (the 0.25 and 2.25 branches are out) -> the
    # n=1 branch at 1.25 is selected, proving mod-1 resonance matching.
    assert profile.locate_rational(1, 3) is None  # 1/3,4/3 both out of [1.25] band


def test_locate_rational_matches_resonance_in_lower_integer_branch():
    # A field rotating at iota = -1 + 1/4 = -0.75 realizes the 1/4
    # resonance in the n=-1 branch. This is the negative-iota counterpart of
    # the higher-branch test above.
    field = CircularTransformField(
        axis_r=1.0, axis_z=0.0, iota=0.25, geometric_winding_offset=-1.0
    )
    profile = sample_iota_profile(
        field,
        axis_r=1.0,
        axis_z=0.0,
        radial_labels=linear_radial_labels(lower=0.05, upper=0.20, count=4),
        toroidal_turns=_PROFILE_TURNS,
        integrator_options=_INTEGRATOR_OPTIONS,
    )
    minimum_iota, maximum_iota = profile.iota_bounds()
    assert minimum_iota == pytest.approx(-0.75, abs=1.0e-4)
    assert maximum_iota == pytest.approx(-0.75, abs=1.0e-4)

    crossing = profile.locate_rational(1, 4)
    assert crossing is not None
    assert crossing.iota == pytest.approx(-0.75, abs=1.0e-4)

    frozen = freeze_rational_targets(
        profile, (RationalTarget(p=-3, q=4, branches=(GREENE_BRANCH_O,)),)
    )
    assert frozen[0].in_domain is True
    assert frozen[0].target is not None
    assert frozen[0].target.radial_label == pytest.approx(
        crossing.radial_label, abs=1.0e-12
    )


def test_freeze_probe_inputs_recenters_chart_on_located_axis():
    true_axis_r = 1.0
    field = CircularTransformField(axis_r=true_axis_r, axis_z=0.0, iota=0.25)

    freezing = freeze_probe_inputs(
        field,
        requested_targets=(RationalTarget(p=1, q=4, branches=(GREENE_BRANCH_O,)),),
        axis_r_guess=true_axis_r + 0.012,  # stale manifest axis, off by ~0.01
        axis_z_guess=0.0,
        nfp=1,
        locate_axis=True,
        radial_labels=linear_radial_labels(lower=0.05, upper=0.20, count=5),
        toroidal_turns=_PROFILE_TURNS,
        integrator_options=_INTEGRATOR_OPTIONS,
    )

    assert freezing.chart.axis_r == pytest.approx(true_axis_r, abs=1.0e-3)
    # Constant-iota field realizes 1/4 everywhere -> the target is in domain.
    assert len(freezing.in_domain_targets) == 1
    assert freezing.in_domain_targets[0].p == 1
    assert freezing.in_domain_targets[0].q == 4


def test_freeze_probe_inputs_excludes_out_of_domain_target():
    # Constant iota 0.25 band: 1/4 in domain, 1/7 (~0.143) out of domain.
    field = CircularTransformField(axis_r=1.0, axis_z=0.0, iota=0.25)

    freezing = freeze_probe_inputs(
        field,
        requested_targets=(
            RationalTarget(p=1, q=4, branches=(GREENE_BRANCH_O,)),
            RationalTarget(p=1, q=7, branches=(GREENE_BRANCH_O,)),
        ),
        axis_r_guess=1.0,
        nfp=1,
        locate_axis=False,
        radial_labels=linear_radial_labels(lower=0.05, upper=0.20, count=4),
        toroidal_turns=_PROFILE_TURNS,
        integrator_options=_INTEGRATOR_OPTIONS,
    )

    in_domain = {(target.p, target.q) for target in freezing.in_domain_targets}
    assert in_domain == {(1, 4)}
    resolutions = {
        (item.requested.p, item.requested.q): item for item in freezing.frozen
    }
    assert resolutions[(1, 7)].in_domain is False
    assert resolutions[(1, 7)].reason == FREEZE_NO_CROSSING


def test_sample_iota_profile_rejects_nonpositive_radial_label():
    with pytest.raises(ValueError, match="radial labels must be finite and > 0"):
        sample_iota_profile(
            _radial_shear_field(),
            axis_r=1.0,
            axis_z=0.0,
            radial_labels=(0.0, 0.1),
            toroidal_turns=_PROFILE_TURNS,
            integrator_options=_INTEGRATOR_OPTIONS,
        )
