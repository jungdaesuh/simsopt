from __future__ import annotations

TF_CURRENT_HARD_LIMIT_A = 8.0e4
BANANA_CURRENT_HARD_LIMIT_A = 1.6e4

COIL_LENGTH_TARGET_M = 1.7
COIL_COIL_MIN_DIST_M = 0.05
COIL_PLASMA_MIN_DIST_M = 0.015
PLASMA_VESSEL_MIN_DIST_M = 0.04
MAX_CURVATURE_INV_M = 100.0

VACUUM_VESSEL_MAJOR_RADIUS_M = 0.976
VACUUM_VESSEL_MINOR_RADIUS_M = 0.222
BANANA_WINDING_MINOR_RADIUS_M = 0.21


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


def validate_major_radius(major_radius: float, *, accept_offspec: bool = False) -> float:
    radius = float(major_radius)
    if abs(radius - VACUUM_VESSEL_MAJOR_RADIUS_M) <= 1.0e-12:
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
