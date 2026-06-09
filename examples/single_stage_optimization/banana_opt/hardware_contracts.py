from __future__ import annotations

import math
import os

TF_CURRENT_HARD_LIMIT_A = 8.0e4
# Jeff's HBT convention: clockwise toroidal field corresponds to negative
# current in this coil parameterization.
TF_CURRENT_CW_DEFAULT_A = -TF_CURRENT_HARD_LIMIT_A
BANANA_CURRENT_HARD_LIMIT_A = 1.6e4

# Preferred engineering target leaves buffer under the absolute hardware ceiling.
COIL_LENGTH_TARGET_M = 1.9
COIL_LENGTH_MIN_FRACTION = 0.5
COIL_LENGTH_MIN_TARGET_M = COIL_LENGTH_TARGET_M * COIL_LENGTH_MIN_FRACTION
COIL_LENGTH_HARD_LIMIT_M = 2.0
COIL_COIL_MIN_DIST_M = 0.0462
COIL_PLASMA_MIN_DIST_M = 0.010
# Type KK banana-coil hardware contract. The outer channel is the solid object
# that collides with hardware; the conductor pack is the current-carrying
# magnetic finite-build footprint. Keep the two rectangles separate.
INCH_TO_M = 0.0254
TYPE_KK_HARDWARE_CONTRACT_ID = "type_kk_outer_channel_2026_06_09"
TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_IN = 1.818
TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_IN = 0.640
TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_M = (
    TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_IN * INCH_TO_M
)
TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_M = (
    TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_IN * INCH_TO_M
)
TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM = (
    TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_M * 1000.0
)
TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM = (
    TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_M * 1000.0
)
TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M = (
    TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_M / 2.0
)
TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M = (
    TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_M / 2.0
)
TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M = math.hypot(
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
)
HARDWARE_KEEPOUT_SAFETY_MARGIN_M = 0.005
# Minimum allowed distance from the banana coil CENTERLINE to the fixed
# in-vessel hardware point cloud. The swept-box keepout uses the Type KK outer
# channel half-extents directly; this legacy centerline threshold preserves the
# 5 mm safety margin for point-cloud and JSON contract checks.
HARDWARE_KEEPOUT_MIN_DISTANCE_M = (
    TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M + HARDWARE_KEEPOUT_SAFETY_MARGIN_M
)
# Diagnostic reference for the LCFS-to-vessel SurfaceSurfaceDistance metric.
# wh_notes.md does not define this as an engineering acceptance floor.
PLASMA_VESSEL_MIN_DIST_M = 0.04
MAX_CURVATURE_INV_M = 100.0

VACUUM_VESSEL_MAJOR_RADIUS_M = 0.976
VACUUM_VESSEL_MINOR_RADIUS_M = 0.222
BANANA_WINDING_SURFACE_MAJOR_RADIUS_M = 0.903
BANANA_WINDING_MINOR_RADIUS_M = 0.142
BANANA_WINDING_CHANNEL_ORIENTATION = "toroidal_surface_tangent"
TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_IN = 1.568
TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_IN = 0.390
TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M = (
    TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_IN * INCH_TO_M
)
TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M = (
    TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_IN * INCH_TO_M
)
TYPE_KK_CONDUCTOR_PACK_HALF_WIDTH_BINORMAL_M = (
    TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M / 2.0
)
TYPE_KK_CONDUCTOR_PACK_HALF_DEPTH_NORMAL_M = (
    TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M / 2.0
)
TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N = 2
TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B = 7
TYPE_KK_FINITE_BUILD_FILAMENT_COUNT = (
    TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N * TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B
)
TYPE_KK_FINITE_BUILD_GAPSIZE_N_M = TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M
TYPE_KK_FINITE_BUILD_GAPSIZE_B_M = (
    TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M
    / (TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B - 1)
)
# Legacy compatibility aliases for magnetic finite-build callers. They now point
# to the Type KK conductor pack, not the outer clearance channel or retired
# placeholder dimensions.
BANANA_PACK_WIDTH_BINORMAL_M = TYPE_KK_CONDUCTOR_PACK_WIDTH_BINORMAL_M
BANANA_PACK_DEPTH_NORMAL_M = TYPE_KK_CONDUCTOR_PACK_DEPTH_NORMAL_M
BANANA_PACK_HALF_BUILD_BINORMAL_M = TYPE_KK_CONDUCTOR_PACK_HALF_WIDTH_BINORMAL_M
BANANA_PACK_HALF_BUILD_NORMAL_M = TYPE_KK_CONDUCTOR_PACK_HALF_DEPTH_NORMAL_M
# Port-fit bounds on ProjectedEllipseWidth.J() (short-axis diameter of the
# best-fit ellipse to the coil curve projected onto the winding surface).
# Max 0.197 m is the short-width limit for the banana coil through the vessel
# port. The 0.17 m value was the previous allowance and is no longer the hard
# engineering limit.
# Min 0.1 m prevents the optimizer from collapsing the coil to a degenerate
# zero-width shape. Matches the current HBT hardware contract.
BANANA_WIDTH_MIN_M = 0.1
BANANA_WIDTH_MAX_M = 0.197
BANANA_SELF_INTERSECT_ALM_SCALE = 1.0
BANANA_HARDWARE_KEEPOUT_ALM_SCALE = 1.0
# Minimum allowed self-distance for the banana coil curve. The external
# jhalpern30 driver activates `CurveSelfIntersect` at 1/CURVATURE_THRESHOLD,
# which matches our reciprocal of the maximum allowed curvature.
BANANA_SELF_INTERSECT_MIN_DISTANCE_M = 1.0 / MAX_CURVATURE_INV_M
# Neighbor-skip factor for CurveSelfIntersect: at runtime, neighbor_skip is
# computed as int(BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR * curve.order).
# 1.5x order matches the external driver convention so the mask excludes
# nearest curve-parameter neighbors that are trivially close but not
# topologically self-intersecting.
BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR = 1.5
# Initial scalarization weights for the new geometric parity terms. These are
# intentionally 100x below the external driver's 1e2 width/self weights and
# match this repo's calibration. Final calibration is validation work.
STAGE2_WIDTH_WEIGHT_DEFAULT = 1.0
STAGE2_SELF_INTERSECT_WEIGHT_DEFAULT = 1.0
STAGE2_POLOIDAL_WEIGHT_DEFAULT = 1.0
SINGLE_STAGE_WIDTH_WEIGHT_DEFAULT = 1.0
SINGLE_STAGE_SELF_INTERSECT_WEIGHT_DEFAULT = 1.0
# Hardware keep-out is opt-in (default-OFF): existing runs stay byte-identical
# until a weight is set AND a keep-out point cloud path is provided.
SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT_DEFAULT = 0.0
POLOIDAL_EXTENT_IDEAL_HALF_WIDTH_RAD = 70.0 * math.pi / 180.0
POLOIDAL_EXTENT_HALF_WIDTH_RAD = 87.0 * math.pi / 180.0
SINGLE_STAGE_POLOIDAL_WEIGHT_DEFAULT = 1.0
POLOIDAL_EXTENT_WEIGHT = SINGLE_STAGE_POLOIDAL_WEIGHT_DEFAULT
LCFS_CLEARANCE_REFERENCE_MAJOR_RADIUS_M = BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M = (
    BANANA_WINDING_MINOR_RADIUS_M - COIL_PLASMA_MIN_DIST_M
)

TARGET_LCFS_MAX_MAJOR_RADIUS_M = BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
TARGET_LCFS_MAX_MINOR_RADIUS_M = BANANA_WINDING_MINOR_RADIUS_M - COIL_PLASMA_MIN_DIST_M
LCFS_RADIUS_ABS_TOL_M = 1.0e-12


def fixed_stage2_clearance_contract() -> dict[str, float]:
    return {
        "COIL_PLASMA_MIN_DIST_M": COIL_PLASMA_MIN_DIST_M,
    }


def stage2_artifact_hardware_contract(length_target_m: float) -> dict[str, float]:
    length_target = float(length_target_m)
    return {
        **fixed_stage2_clearance_contract(),
        "LENGTH_TARGET": length_target,
        "LENGTH_MIN_TARGET": COIL_LENGTH_MIN_FRACTION * length_target,
        "WIDTH_MIN_THRESHOLD": BANANA_WIDTH_MIN_M,
        "WIDTH_MAX_THRESHOLD": BANANA_WIDTH_MAX_M,
        "SELF_INTERSECT_THRESHOLD": 0.0,
        "SELF_INTERSECT_MIN_DISTANCE": BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
    }


def fixed_stage2_artifact_hardware_contract() -> dict[str, float]:
    return stage2_artifact_hardware_contract(COIL_LENGTH_TARGET_M)


def validate_tf_current_limit(tf_current_A: float) -> float:
    current = float(tf_current_A)
    if not (-TF_CURRENT_HARD_LIMIT_A <= current < 0.0):
        raise ValueError(
            "TF coil current must be negative for the CW toroidal-field "
            f"convention and no smaller than {-TF_CURRENT_HARD_LIMIT_A:.0f} A."
        )
    return current


def is_coil_length_target_offspec(length_target_m: float) -> bool:
    return float(length_target_m) > COIL_LENGTH_HARD_LIMIT_M


def validate_coil_length_target(
    length_target_m: float,
    *,
    accept_offspec_coil_length: bool = False,
    field_name: str = "--length-target",
) -> float:
    length_target = float(length_target_m)
    if is_coil_length_target_offspec(length_target) and not accept_offspec_coil_length:
        raise ValueError(f"{field_name} must be <= {COIL_LENGTH_HARD_LIMIT_M:.3f} m.")
    return length_target


def validate_banana_winding_surface_radius(banana_surf_radius: float) -> float:
    radius = float(banana_surf_radius)
    if not (0.0 < radius < VACUUM_VESSEL_MINOR_RADIUS_M):
        raise ValueError(
            "Banana winding-surface radius must stay strictly inside the vacuum vessel "
            f"minor radius {VACUUM_VESSEL_MINOR_RADIUS_M:.3f} m."
        )
    return radius


def validate_target_lcfs_major_radius(target_major_radius_m: float) -> float:
    radius = float(target_major_radius_m)
    if not (0.0 < radius <= TARGET_LCFS_MAX_MAJOR_RADIUS_M + LCFS_RADIUS_ABS_TOL_M):
        raise ValueError(
            "Requested target LCFS major radius must lie in "
            f"(0, {TARGET_LCFS_MAX_MAJOR_RADIUS_M:.3f}] m."
        )
    return radius


def validate_target_lcfs_minor_radius(target_minor_radius_m: float) -> float:
    radius = float(target_minor_radius_m)
    if not (0.0 < radius <= TARGET_LCFS_MAX_MINOR_RADIUS_M + LCFS_RADIUS_ABS_TOL_M):
        raise ValueError(
            "Requested target LCFS minor radius must lie in "
            f"(0, {TARGET_LCFS_MAX_MINOR_RADIUS_M:.3f}] m."
        )
    return radius


_MAJOR_RADIUS_TOL_M = 1.0e-12


def is_major_radius_offspec(major_radius: float) -> bool:
    return abs(float(major_radius) - VACUUM_VESSEL_MAJOR_RADIUS_M) > _MAJOR_RADIUS_TOL_M


def validate_major_radius(major_radius: float) -> float:
    radius = float(major_radius)
    if not is_major_radius_offspec(radius):
        return radius
    raise ValueError(
        f"--major-radius must match the vacuum-vessel major radius "
        f"{VACUUM_VESSEL_MAJOR_RADIUS_M:.3f} m (got {radius:.6f}). "
        "Off-spec R0 produces coils that do not fit the HBT-EP vacuum vessel."
    )


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
