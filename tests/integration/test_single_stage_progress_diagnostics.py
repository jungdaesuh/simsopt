"""Progress-diagnostic contracts for single-stage target-lane runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

from examples.single_stage_optimization.SINGLE_STAGE import (
    single_stage_banana_example as single_stage_example,
)
from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
    HIGH_MPOL_OUTER_FTOL_FLOOR,
    _single_stage_hardware_status_progress_fields,
    _summarize_k1_forward_result_for_progress,
    build_event_progress_recorder,
    load_single_stage_objective_evaluation_replay_events,
    parse_args,
    resolve_single_stage_outer_ftol,
    should_evaluate_pending_target_lane_initial_objective,
)


def test_hardware_status_progress_fields_preserve_violation_details():
    hardware_status = {
        "success": False,
        "violation_keys": ["max_curvature"],
        "violations": ["max_curvature 41.0 exceeds threshold 40.0"],
        "threshold_margins": {
            "curve_curve_min_dist": 0.01,
            "curve_surface_min_dist": 0.02,
            "surface_vessel_min_dist": 0.03,
            "max_curvature": -1.0,
        },
        "curve_curve_min_dist": 0.11,
        "cc_dist": 0.10,
        "curve_surface_min_dist": 0.12,
        "cs_dist": 0.10,
        "surface_vessel_min_dist": 0.13,
        "ss_dist": 0.10,
        "max_curvature": 41.0,
        "curvature_threshold": 40.0,
        "finite_flags": {"max_curvature": True},
        "threshold_flags": {"max_curvature": False},
    }

    fields = _single_stage_hardware_status_progress_fields(hardware_status)

    assert fields == {
        "success": False,
        "violation_keys": ["max_curvature"],
        "violations": ["max_curvature 41.0 exceeds threshold 40.0"],
        "threshold_margins": {
            "curve_curve_min_dist": 0.01,
            "curve_surface_min_dist": 0.02,
            "surface_vessel_min_dist": 0.03,
            "max_curvature": -1.0,
        },
        "curve_curve_min_dist": 0.11,
        "cc_dist": 0.10,
        "curve_surface_min_dist": 0.12,
        "cs_dist": 0.10,
        "surface_vessel_min_dist": 0.13,
        "ss_dist": 0.10,
        "max_curvature": 41.0,
        "curvature_threshold": 40.0,
    }


def test_reporting_snapshot_diagnostic_skips_pending_initial_value_grad():
    run_dict = {"initial_objective_pending": True}

    assert not should_evaluate_pending_target_lane_initial_objective(
        run_dict,
        diagnose_target_lane_reporting_snapshot=True,
    )
    assert should_evaluate_pending_target_lane_initial_objective(
        run_dict,
        diagnose_target_lane_reporting_snapshot=False,
    )
    assert not should_evaluate_pending_target_lane_initial_objective(
        {"initial_objective_pending": False},
        diagnose_target_lane_reporting_snapshot=True,
    )


def _parse_child_args(*extra: str):
    argv = ["single_stage_banana_example.py", *extra]
    with mock.patch.object(sys, "argv", argv):
        return parse_args()


def test_cli_exposes_reporting_snapshot_diagnostic():
    args = _parse_child_args("--diagnose-target-lane-reporting-snapshot")

    assert args.diagnose_target_lane_reporting_snapshot is True


def test_cli_k1_subtimers_do_not_enable_full_target_lane_profile():
    args = _parse_child_args(
        "--backend",
        "jax",
        "--optimizer-backend",
        "scipy-jax-decomposed",
        "--trace-target-lane-k1-subtimers",
    )

    assert args.trace_target_lane_k1_subtimers is True
    assert args.profile_target_lane is False


@pytest.mark.parametrize("mpol", [14, 15, 16, 17, 18, 24])
def test_default_outer_ftol_respects_high_mpol_noise_floor(mpol):
    assert resolve_single_stage_outer_ftol(mpol) >= HIGH_MPOL_OUTER_FTOL_FLOOR


def test_default_outer_ftol_preserves_low_mpol_fallback():
    assert resolve_single_stage_outer_ftol(2) == pytest.approx(1e-5)


def test_explicit_outer_ftol_remains_user_override():
    assert resolve_single_stage_outer_ftol(18, 1e-12) == pytest.approx(1e-12)


def test_reporting_snapshot_diagnostic_rejects_other_skip_modes():
    with mock.patch.object(
        sys,
        "argv",
        [
            "single_stage_banana_example.py",
            "--diagnose-target-lane-reporting-snapshot",
            "--profile-target-lane-only",
        ],
    ):
        with pytest.raises(ValueError, match="cannot be combined"):
            parse_args()


def test_event_progress_recorder_uses_ndjson_sidecar(tmp_path: Path):
    progress_json = tmp_path / "outer_optimizer_progress.json"
    record_event = build_event_progress_recorder(progress_json)

    for index in range(5):
        record_event(
            "objective_evaluation",
            objective={"value": float(index)},
            candidate={"dofs": [float(index)]},
        )

    payload = json.loads(progress_json.read_text(encoding="utf-8"))
    sidecar_path = tmp_path / payload["events_path"]
    sidecar_events = [
        json.loads(line)
        for line in sidecar_path.read_text(encoding="utf-8").splitlines()
    ]

    assert payload["events_format"] == "ndjson"
    assert payload["events"] == []
    assert payload["event_count"] == 5
    assert payload["latest_event"]["event_index"] == 4
    assert [event["event_index"] for event in sidecar_events] == [0, 1, 2, 3, 4]
    replay_events = load_single_stage_objective_evaluation_replay_events(progress_json)
    assert [event["candidate"]["dofs"] for event in replay_events] == [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]


def test_k1_progress_summary_batches_scalar_materialization(monkeypatch):
    calls = []

    def fake_device_get(value):
        calls.append(value)
        return value

    monkeypatch.setattr(single_stage_example.jax, "device_get", fake_device_get)

    fields = _summarize_k1_forward_result_for_progress(
        {
            "primal_success": True,
            "primal_success_present": True,
            "newton_iter": 3,
            "newton_iter_present": True,
            "final_gradient_norm": np.inf,
            "final_gradient_norm_present": True,
        }
    )

    assert len(calls) == 1
    assert set(calls[0]) == {
        "primal_success",
        "primal_success_present",
        "newton_iter",
        "newton_iter_present",
        "final_gradient_norm",
        "final_gradient_norm_present",
    }
    assert fields["primal_success"] is True
    assert fields["newton_iter"] == 3
    assert fields["final_gradient_norm_finite"] is False
    assert fields["final_gradient_norm_classification"] == "+inf"


def test_k1_progress_summary_accepts_sparse_absent_optional_fields():
    fields = _summarize_k1_forward_result_for_progress(
        {
            "newton_matvec_counter_token_present": False,
            "newton_attempted_iterations_present": False,
        }
    )

    assert fields == {}
