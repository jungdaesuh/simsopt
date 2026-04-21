from __future__ import annotations

import os

TF_CURRENT_HARD_LIMIT_A = 8.0e4
BANANA_CURRENT_HARD_LIMIT_A = 1.6e4

COIL_LENGTH_TARGET_M = 1.7
COIL_COIL_MIN_DIST_M = 0.05
COIL_PLASMA_MIN_DIST_M = 0.015
PLASMA_VESSEL_MIN_DIST_M = 0.04
MAX_CURVATURE_INV_M = 100.0

VACUUM_VESSEL_MAJOR_RADIUS_M = 0.976
VACUUM_VESSEL_MINOR_RADIUS_M = 0.222
BANANA_WINDING_SURFACE_MAJOR_RADIUS_M = VACUUM_VESSEL_MAJOR_RADIUS_M
BANANA_WINDING_MINOR_RADIUS_M = 0.21
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


def fixed_stage2_artifact_hardware_contract() -> dict[str, float]:
    return {
        **fixed_stage2_clearance_contract(),
        "LENGTH_TARGET": COIL_LENGTH_TARGET_M,
    }


def validate_tf_current_limit(tf_current_A: float) -> float:
    current = float(tf_current_A)
    if not (0.0 < current <= TF_CURRENT_HARD_LIMIT_A):
        raise ValueError(
            f"TF coil current must be in the interval (0, {TF_CURRENT_HARD_LIMIT_A:.0f}] A."
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


def validate_major_radius(major_radius: float, *, accept_offspec: bool = False) -> float:
    radius = float(major_radius)
    if not is_major_radius_offspec(radius):
        return radius
    if accept_offspec:
        return radius
    raise ValueError(
        f"--major-radius must match the vacuum-vessel major radius "
        f"{VACUUM_VESSEL_MAJOR_RADIUS_M:.3f} m (got {radius:.6f}). "
        "Off-spec R0 was accepted historically but produces coils that do not fit "
        "the HBT-EP vacuum vessel. Pass --accept-offspec-r0-seed to reproduce "
        "historical artifacts on off-spec geometry."
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
    accept_offspec: bool = False,
    threshold: float = PLASMA_VESSEL_MIN_DIST_M,
) -> float:
    clearance = float(plasma_vessel_min_dist_m)
    clearance_threshold = float(threshold)
    if not is_plasma_vessel_clearance_offspec(
        clearance,
        threshold=clearance_threshold,
    ):
        return clearance
    if accept_offspec:
        return clearance
    raise ValueError(
        "LCFS-to-vessel clearance violates the HBT-EP hardware contract "
        f"({clearance:.6f} m < {clearance_threshold:.6f} m). "
        "Use the direct LCFS-to-vessel spacing metric for fit validation, not "
        "a proxy envelope. Set "
        f"{ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE_ENV} to reproduce historical "
        "artifacts on off-spec plasma geometry."
    )


ACCEPT_OFFSPEC_R0_SEED_ENV = "ACCEPT_OFFSPEC_R0_SEED"
ACCEPT_OFFSPEC_R0_SEED_HELP = (
    "Allow --major-radius to deviate from the vacuum-vessel contract. "
    "Use only to reproduce historical artifacts; produces coils that do "
    "not fit HBT-EP."
)

ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE_ENV = (
    "ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE"
)
ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE_ENV_HELP = (
    "Allow the LCFS-to-vessel clearance to fall below the fixed HBT-EP "
    f"threshold of {PLASMA_VESSEL_MIN_DIST_M:.3f} m. Use only to reproduce "
    "historical artifacts on off-spec plasma geometry."
)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
