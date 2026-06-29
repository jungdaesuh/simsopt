"""Continuation-ladder resolution sequencing, including the optional
intermediate rungs that let the donor step up in smaller increments
(e.g. 2->4->6->8->10 instead of the default 6->final jump)."""

from __future__ import annotations

import json

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
_EXPLICIT_COMMON = {
    key: value
    for key, value in _COMMON.items()
    if key not in {"coarse_maxiter", "medium_maxiter"}
}


def _mpols(**kwargs):
    return [stage.mpol for stage in cont.build_default_continuation_stages(**kwargs)]


def test_default_ladder_is_unchanged():
    # No intermediate rungs => historical 2,4,6,final ladder, byte-for-byte.
    assert _mpols(final_mpol=12, final_ntor=12, **_COMMON) == [2, 4, 6, 12]
    assert _mpols(final_mpol=10, final_ntor=10, **_COMMON) == [2, 4, 6, 10]


def test_intermediate_rung_inserts_between_prefinal_and_final():
    assert _mpols(final_mpol=10, final_ntor=10, intermediate_rungs=(8,), **_COMMON) == [
        2,
        4,
        6,
        8,
        10,
    ]


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
    assert (
        rung.target_lane_boozer_bfgs_maxiter
        == prefinal["target_lane_boozer_bfgs_maxiter"]
    )
    assert rung.ntor == 8
    assert rung.nphi == _COMMON["final_nphi"]


def test_explicit_stage_rungs_replace_default_ladder():
    stages = cont.build_explicit_continuation_stages(
        stage_rungs=(10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36),
        final_mpol=36,
        final_ntor=36,
        **_EXPLICIT_COMMON,
    )
    assert [stage.mpol for stage in stages] == [
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
    ]
    assert [stage.ntor for stage in stages] == [
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
    ]
    assert stages[0].name == "rung-mpol10"
    assert stages[-1].name == "final"


def test_explicit_stage_rungs_reject_final_mismatch():
    try:
        cont.build_explicit_continuation_stages(
            stage_rungs=(10, 12),
            final_mpol=14,
            final_ntor=14,
            **_EXPLICIT_COMMON,
        )
    except ValueError as exc:
        assert "must match --mpol" in str(exc)
    else:
        raise AssertionError("expected final-rung mismatch to fail")


def test_warm_start_contract_overrides_allow_partial_contract(tmp_path):
    warm_start = tmp_path / "seed"
    warm_start.mkdir()
    (warm_start / "results.json").write_text(
        json.dumps(
            {
                "TARGET_VOLUME": 0.04996266057182704,
                "CURVATURE_THRESHOLD": 43.311417755948824,
            }
        ),
        encoding="utf-8",
    )

    assert cont.load_warm_start_contract_overrides(warm_start) == {
        "--vol-target": 0.04996266057182704,
        "--curvature-threshold": 43.311417755948824,
    }


def test_mismatched_runtime_seed_spec_is_reprojected_without_host_artifacts(tmp_path):
    warm_start = tmp_path / "target_lane_seed"
    warm_start.mkdir()
    runtime_seed_spec = warm_start / cont._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME
    runtime_seed_spec.write_text(
        json.dumps(
            {
                "surface": {"mpol": 6, "ntor": 6},
                "quadrature": {"nphi": 64, "ntheta": 32},
            }
        ),
        encoding="utf-8",
    )
    stage = cont.ContinuationStage(
        name="rung-mpol8",
        mpol=8,
        ntor=8,
        nphi=64,
        ntheta=32,
        maxiter=1,
    )
    output_spec = tmp_path / "next" / cont._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME

    assert cont.resolve_optional_stage_seed_path(warm_start) is None
    assert cont.existing_stage_jax_runtime_seed_spec_path(warm_start, stage) is None
    assert (
        cont.reproject_stage_jax_runtime_seed_source_path(warm_start, stage)
        == runtime_seed_spec
    )

    command = cont.build_stage_jax_runtime_seed_source_command(
        python_executable="python",
        passthrough_args=[],
        stage=stage,
        jax_runtime_seed_source=runtime_seed_spec,
        jax_runtime_seed_spec_path=output_spec,
    )

    assert "--jax-runtime-seed-source" in command
    assert str(runtime_seed_spec) in command
    assert "--warm-start-run-dir" not in command


def test_matching_runtime_seed_spec_is_reused_without_reprojection(tmp_path):
    warm_start = tmp_path / "target_lane_seed"
    warm_start.mkdir()
    runtime_seed_spec = warm_start / cont._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME
    runtime_seed_spec.write_text(
        json.dumps(
            {
                "surface": {"mpol": 8, "ntor": 8},
                "quadrature": {"nphi": 64, "ntheta": 32},
            }
        ),
        encoding="utf-8",
    )
    stage = cont.ContinuationStage(
        name="rung-mpol8",
        mpol=8,
        ntor=8,
        nphi=64,
        ntheta=32,
        maxiter=1,
    )

    assert cont.existing_stage_jax_runtime_seed_spec_path(warm_start, stage) == (
        runtime_seed_spec
    )
    assert cont.reproject_stage_jax_runtime_seed_source_path(warm_start, stage) is None
