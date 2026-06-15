"""Progress-diagnostic contracts for single-stage target-lane runs."""

from __future__ import annotations

from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
    _single_stage_hardware_status_progress_fields,
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
