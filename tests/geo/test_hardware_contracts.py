from __future__ import annotations

import math
import sys
from pathlib import Path

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt import hardware_contracts as hc  # noqa: E402


def test_type_kk_outer_channel_dimensions_and_keepout_threshold() -> None:
    assert hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_M == 1.818 * hc.INCH_TO_M
    assert hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_M == 0.640 * hc.INCH_TO_M
    assert hc.TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M == (
        hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_M / 2.0
    )
    assert hc.TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M == (
        hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_M / 2.0
    )

    corner_reach = math.hypot(
        hc.TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
        hc.TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    )
    assert hc.TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M == corner_reach
    assert hc.HARDWARE_KEEPOUT_MIN_DISTANCE_M == (
        corner_reach + hc.HARDWARE_KEEPOUT_SAFETY_MARGIN_M
    )
    assert math.isclose(hc.HARDWARE_KEEPOUT_MIN_DISTANCE_M, 0.029477496480645238)


def test_type_kk_conductor_pack_and_finite_build_defaults_are_separate() -> None:
    assert hc.TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M == 1.568 * hc.INCH_TO_M
    assert hc.TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M == 0.390 * hc.INCH_TO_M
    assert hc.TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M != (
        hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_M
    )
    assert hc.TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M != (
        hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_M
    )

    assert hc.TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N == 2
    assert hc.TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B == 7
    assert hc.TYPE_KK_FINITE_BUILD_FILAMENT_COUNT == 14
    assert hc.TYPE_KK_FINITE_BUILD_GAPSIZE_N_M == (
        hc.TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M
    )
    assert hc.TYPE_KK_FINITE_BUILD_GAPSIZE_B_M == (
        hc.TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M / 6.0
    )
    assert hc.TYPE_KK_FORCE_REGULARIZATION_CONDUCTOR_RADIUS_M == (
        0.5
        * min(hc.TYPE_KK_FINITE_BUILD_GAPSIZE_N_M, hc.TYPE_KK_FINITE_BUILD_GAPSIZE_B_M)
    )


def test_single_stage_engineering_defaults_are_production_on() -> None:
    assert hc.SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT_DEFAULT > 0.0
    assert hc.SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT_DEFAULT > 0.0
    assert hc.SINGLE_STAGE_COIL_FORCE_WEIGHT_DEFAULT > 0.0
    assert hc.SINGLE_STAGE_COIL_FORCE_CONDUCTOR_RADIUS_DEFAULT_M == (
        hc.TYPE_KK_FORCE_REGULARIZATION_CONDUCTOR_RADIUS_M
    )
    assert hc.SINGLE_STAGE_COIL_FORCE_CONDUCTOR_RADIUS_DEFAULT_M > 0.0


def test_legacy_pack_aliases_no_longer_point_to_old_placeholder() -> None:
    assert hc.BANANA_PACK_WIDTH_BINORMAL_M == hc.TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M
    assert hc.BANANA_PACK_DEPTH_NORMAL_M == hc.TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M
    assert hc.BANANA_PACK_WIDTH_BINORMAL_M != 0.029
    assert hc.BANANA_PACK_DEPTH_NORMAL_M != 0.020


def test_lcfs_edge_envelope_limits_derive_from_winding_surface_clearance() -> None:
    assert hc.LCFS_OUTBOARD_RADIUS_MAX_M == (
        hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
        + hc.BANANA_WINDING_MINOR_RADIUS_M
        - hc.COIL_PLASMA_MIN_DIST_M
    )
    assert hc.LCFS_INBOARD_RADIUS_MIN_M == (
        hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
        - hc.BANANA_WINDING_MINOR_RADIUS_M
        + hc.COIL_PLASMA_MIN_DIST_M
    )
    assert math.isclose(hc.lcfs_outboard_edge_radius_m(0.921401, 0.113599), 1.035)
    assert math.isclose(hc.lcfs_inboard_edge_radius_m(0.921401, 0.113599), 0.807802)


def test_lcfs_constraint_mode_validation_keeps_centered_default() -> None:
    assert hc.LCFS_CONSTRAINT_MODE_DEFAULT == hc.LCFS_CONSTRAINT_MODE_CENTERED
    assert hc.validate_lcfs_constraint_mode("centered") == "centered"
    assert hc.validate_lcfs_constraint_mode("edge_envelope") == "edge_envelope"


def test_lcfs_edge_envelope_allows_offcenter_center_when_edges_fit() -> None:
    hc.validate_lcfs_edge_envelope(0.921401, 0.113599)

    try:
        hc.validate_target_lcfs_major_radius(0.921401)
    except ValueError as exc:
        assert "target LCFS major radius" in str(exc)
    else:
        raise AssertionError("centered LCFS major-radius validator accepted oversize R")


def test_lcfs_edge_envelope_rejects_nonfinite_or_nonpositive_major_radius() -> None:
    for invalid_major_radius in (float("nan"), float("inf"), 0.0, -0.1):
        try:
            hc.validate_lcfs_edge_envelope(invalid_major_radius, 0.113599)
        except ValueError as exc:
            assert "finite and positive" in str(exc)
        else:
            raise AssertionError(
                "edge-envelope LCFS validator accepted invalid major radius "
                f"{invalid_major_radius!r}"
            )
