"""Progress-diagnostic contracts for single-stage target-lane runs."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
    _single_stage_hardware_status_progress_fields,
    parse_args,
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
