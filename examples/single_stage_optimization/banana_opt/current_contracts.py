from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MU0_OVER_2PI = 2.0e-7

CurrentInputSource = Literal["physical_A", "raw_boozer_I", "default_zero"]
FiniteCurrentMode = Literal["boozer_surrogate", "disabled"]
EffectiveCurrentMode = Literal["vacuum", "boozer_surrogate"]
CURRENT_MODE_ZERO_TOL = 1e-12
BANANA_CURRENT_HARD_LIMIT_A = 16000.0


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
