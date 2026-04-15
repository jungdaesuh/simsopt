from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    validate_tf_current_limit,
)


MU0_OVER_2PI = 2.0e-7

CurrentInputSource = Literal["physical_A", "raw_boozer_I", "default_zero"]
FiniteCurrentMode = Literal["boozer_surrogate", "disabled"]
EffectiveCurrentMode = Literal["vacuum", "boozer_surrogate"]
CURRENT_MODE_ZERO_TOL = 1e-12

__all__ = [
    "BANANA_CURRENT_HARD_LIMIT_A",
    "CURRENT_MODE_ZERO_TOL",
    "CurrentInputSource",
    "EffectiveCurrentMode",
    "FiniteCurrentMode",
    "MU0_OVER_2PI",
    "PlasmaCurrentSettings",
    "apply_banana_current_upper_bound",
    "banana_current_exceeds_limit",
    "boozer_I_to_physical_current_A",
    "infer_uniform_coil_current_A",
    "physical_current_to_boozer_I",
    "resolve_effective_current_mode",
    "resolve_loaded_tf_current_A",
    "resolve_plasma_current_settings",
    "unwrap_current_optimizable",
]


@dataclass(frozen=True)
class PlasmaCurrentSettings:
    boozer_I: float
    plasma_current_A: float
    input_source: CurrentInputSource
    mode: FiniteCurrentMode
    effective_mode: EffectiveCurrentMode


def physical_current_to_boozer_I(plasma_current_A: float) -> float:
    return MU0_OVER_2PI * float(plasma_current_A)


def boozer_I_to_physical_current_A(boozer_I: float) -> float:
    return float(boozer_I) / MU0_OVER_2PI


def resolve_effective_current_mode(boozer_I: float) -> EffectiveCurrentMode:
    if abs(float(boozer_I)) <= CURRENT_MODE_ZERO_TOL:
        return "vacuum"
    return "boozer_surrogate"


def unwrap_current_optimizable(current):
    scale = 1.0
    current_optimizable = current
    while hasattr(current_optimizable, "current_to_scale") and hasattr(
        current_optimizable,
        "scale",
    ):
        scale *= float(current_optimizable.scale)
        current_optimizable = current_optimizable.current_to_scale
    if not hasattr(current_optimizable, "local_lower_bounds") or not hasattr(
        current_optimizable,
        "local_upper_bounds",
    ):
        raise TypeError("Current does not expose local bounds.")
    return current_optimizable, scale


def apply_banana_current_upper_bound(current, banana_current_max_A):
    current_optimizable, scale = unwrap_current_optimizable(current)
    if scale == 0.0:
        raise ValueError("Banana current scale must be non-zero to apply a bound.")
    lower_bounds = np.asarray(current_optimizable.local_lower_bounds, dtype=float).copy()
    upper_bounds = np.asarray(current_optimizable.local_upper_bounds, dtype=float).copy()
    scaled_magnitude_bound = float(banana_current_max_A) / abs(scale)
    lower_bounds[0] = max(lower_bounds[0], -scaled_magnitude_bound)
    upper_bounds[0] = min(upper_bounds[0], scaled_magnitude_bound)
    current_optimizable.local_lower_bounds = lower_bounds
    current_optimizable.local_upper_bounds = upper_bounds


def banana_current_exceeds_limit(current_A: float, banana_current_max_A: float) -> bool:
    return abs(float(current_A)) > float(banana_current_max_A)


def infer_uniform_coil_current_A(coils) -> float | None:
    if not coils:
        return None
    coil_currents = np.asarray([coil.current.get_value() for coil in coils], dtype=float)
    if np.allclose(coil_currents, coil_currents[0], rtol=0.0, atol=1.0e-12):
        return float(coil_currents[0])
    return None


def resolve_loaded_tf_current_A(recorded_tf_current_A, tf_coils) -> float:
    realized_tf_current_A = infer_uniform_coil_current_A(tf_coils)
    if realized_tf_current_A is None:
        raise ValueError(
            "Loaded Stage 2 TF coils do not share a uniform fixed current; cannot "
            "validate the seed current contract."
        )
    if recorded_tf_current_A is not None and not np.isclose(
        realized_tf_current_A,
        float(recorded_tf_current_A),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Loaded Stage 2 TF coil current "
            f"{realized_tf_current_A:.6f} A does not match the artifact metadata "
            f"TF_CURRENT_A={float(recorded_tf_current_A):.6f} A."
        )
    return validate_tf_current_limit(realized_tf_current_A)


def resolve_plasma_current_settings(
    *,
    raw_boozer_I: float | None,
    plasma_current_A: float | None,
) -> PlasmaCurrentSettings:
    if plasma_current_A is not None:
        if raw_boozer_I is not None:
            raise ValueError("Cannot use --plasma-current-A together with --boozer-I")
        resolved_boozer_I = physical_current_to_boozer_I(plasma_current_A)
        return PlasmaCurrentSettings(
            boozer_I=resolved_boozer_I,
            plasma_current_A=float(plasma_current_A),
            input_source="physical_A",
            mode="boozer_surrogate",
            effective_mode=resolve_effective_current_mode(resolved_boozer_I),
        )
    if raw_boozer_I is not None:
        resolved_boozer_I = float(raw_boozer_I)
        return PlasmaCurrentSettings(
            boozer_I=resolved_boozer_I,
            plasma_current_A=boozer_I_to_physical_current_A(raw_boozer_I),
            input_source="raw_boozer_I",
            mode="boozer_surrogate",
            effective_mode=resolve_effective_current_mode(resolved_boozer_I),
        )

    return PlasmaCurrentSettings(
        boozer_I=0.0,
        plasma_current_A=0.0,
        input_source="default_zero",
        mode="disabled",
        effective_mode="vacuum",
    )
