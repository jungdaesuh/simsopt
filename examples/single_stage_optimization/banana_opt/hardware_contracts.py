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
COIL_COIL_MIN_DIST_M = 0.05
COIL_PLASMA_MIN_DIST_M = 0.015
PLASMA_VESSEL_MIN_DIST_M = 0.04
MAX_CURVATURE_INV_M = 100.0

VACUUM_VESSEL_MAJOR_RADIUS_M = 0.976
VACUUM_VESSEL_MINOR_RADIUS_M = 0.222
BANANA_WINDING_SURFACE_MAJOR_RADIUS_M = VACUUM_VESSEL_MAJOR_RADIUS_M
BANANA_WINDING_MINOR_RADIUS_M = 0.21
# Port-fit bounds on ProjectedEllipseWidth.J() (short-axis diameter of the
# best-fit ellipse to the coil curve projected onto the winding surface).
# Max 0.17 m is the bare-curve allowance for a 0.30 m port aperture; the
# remaining ~0.13 m budgets coil pack, insulation, supports, and tolerance.
# Min 0.05 m prevents the optimizer from collapsing the coil to a degenerate
# zero-width shape.
BANANA_WIDTH_MIN_M = 0.05
BANANA_WIDTH_MAX_M = 0.17
BANANA_SELF_INTERSECT_ALM_SCALE = 1.0
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
SINGLE_STAGE_WIDTH_WEIGHT_DEFAULT = 1.0
SINGLE_STAGE_SELF_INTERSECT_WEIGHT_DEFAULT = 1.0
POLOIDAL_EXTENT_HALF_WIDTH_RAD = 45.0 * math.pi / 180.0
POLOIDAL_EXTENT_WEIGHT = 1.0
LCFS_CLEARANCE_REFERENCE_MAJOR_RADIUS_M = VACUUM_VESSEL_MAJOR_RADIUS_M
LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M = (
    VACUUM_VESSEL_MINOR_RADIUS_M - PLASMA_VESSEL_MIN_DIST_M
)

TARGET_LCFS_MAX_MAJOR_RADIUS_M = 0.92
TARGET_LCFS_MAX_MINOR_RADIUS_M = 0.15


def fixed_stage2_clearance_contract() -> dict[str, float]:
    return {
        "COIL_PLASMA_MIN_DIST_M": COIL_PLASMA_MIN_DIST_M,
        "PLASMA_VESSEL_MIN_DIST_M": PLASMA_VESSEL_MIN_DIST_M,
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
    if not (0.0 < radius <= TARGET_LCFS_MAX_MAJOR_RADIUS_M):
        raise ValueError(
            "Requested target LCFS major radius must lie in "
            f"(0, {TARGET_LCFS_MAX_MAJOR_RADIUS_M:.3f}] m."
        )
    return radius


def validate_target_lcfs_minor_radius(target_minor_radius_m: float) -> float:
    radius = float(target_minor_radius_m)
    if not (0.0 < radius <= TARGET_LCFS_MAX_MINOR_RADIUS_M):
        raise ValueError(
            "Requested target LCFS minor radius must lie in "
            f"(0, {TARGET_LCFS_MAX_MINOR_RADIUS_M:.3f}] m."
        )
    return radius


_MAJOR_RADIUS_TOL_M = 1.0e-12
_PLASMA_VESSEL_CLEARANCE_TOL_M = 1.0e-9


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


def is_plasma_vessel_clearance_offspec(
    plasma_vessel_min_dist_m: float,
    *,
    threshold: float = PLASMA_VESSEL_MIN_DIST_M,
) -> bool:
    clearance = float(plasma_vessel_min_dist_m)
    clearance_threshold = float(threshold)
    return clearance < clearance_threshold - _PLASMA_VESSEL_CLEARANCE_TOL_M


def validate_plasma_vessel_clearance(
    plasma_vessel_min_dist_m: float,
    *,
    threshold: float = PLASMA_VESSEL_MIN_DIST_M,
) -> float:
    clearance = float(plasma_vessel_min_dist_m)
    clearance_threshold = float(threshold)
    if not is_plasma_vessel_clearance_offspec(
        clearance,
        threshold=clearance_threshold,
    ):
        return clearance
    raise ValueError(
        "LCFS-to-vessel clearance violates the HBT-EP hardware contract "
        f"({clearance:.6f} m < {clearance_threshold:.6f} m). "
        "Use the direct LCFS-to-vessel spacing metric for fit validation, not "
        "a proxy envelope."
    )


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
