"""Continuation-ladder resolution sequencing, including the optional
intermediate rungs that let the donor step up in smaller increments
(e.g. 2->4->6->8->10 instead of the default 6->final jump)."""
from __future__ import annotations

from examples.single_stage_optimization.SINGLE_STAGE import (
    run_single_stage_continuation as cont,
)

_COMMON = dict(
    final_nphi=64,
    final_ntheta=32,
    final_maxiter=300,
    coarse_maxiter=1,
    medium_maxiter=1,
    prefinal_maxiter=2,
)


def _mpols(**kwargs):
    return [stage.mpol for stage in cont.build_default_continuation_stages(**kwargs)]


def test_default_ladder_is_unchanged():
    # No intermediate rungs => historical 2,4,6,final ladder, byte-for-byte.
    assert _mpols(final_mpol=12, final_ntor=12, **_COMMON) == [2, 4, 6, 12]
    assert _mpols(final_mpol=10, final_ntor=10, **_COMMON) == [2, 4, 6, 10]


def test_intermediate_rung_inserts_between_prefinal_and_final():
    assert _mpols(
        final_mpol=10, final_ntor=10, intermediate_rungs=(8,), **_COMMON
    ) == [2, 4, 6, 8, 10]


def test_out_of_band_intermediate_rungs_are_ignored():
    # 4 is at/below the prefinal band, 12 is above the final; both dropped, and
    # the ladder stays monotonically increasing with the in-band rung (8) kept.
    assert _mpols(
        final_mpol=10, final_ntor=10, intermediate_rungs=(4, 8, 12), **_COMMON
    ) == [2, 4, 6, 8, 10]


def test_intermediate_rung_inherits_prefinal_fast_trial_overrides():
    stages = cont.build_default_continuation_stages(
        final_mpol=10, final_ntor=10, intermediate_rungs=(8,), **_COMMON
    )
    rung = next(stage for stage in stages if stage.mpol == 8)
    prefinal = cont._VALIDATED_FAST_TRIAL_STAGE_OVERRIDES["prefinal"]
    assert rung.minimal_artifacts is prefinal["minimal_artifacts"]
    assert rung.target_lane_boozer_bfgs_maxiter == prefinal["target_lane_boozer_bfgs_maxiter"]
    assert rung.ntor == 8
    assert rung.nphi == _COMMON["final_nphi"]
