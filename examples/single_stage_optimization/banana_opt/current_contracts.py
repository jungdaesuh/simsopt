from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MU0_OVER_2PI = 2.0e-7

CurrentInputSource = Literal["physical_A", "raw_boozer_I", "default_zero"]
FiniteCurrentMode = Literal["boozer_surrogate", "disabled"]


@dataclass(frozen=True)
class PlasmaCurrentSettings:
    boozer_I: float
    plasma_current_A: float
    input_source: CurrentInputSource
    mode: FiniteCurrentMode


def physical_current_to_boozer_I(plasma_current_A: float) -> float:
    return MU0_OVER_2PI * float(plasma_current_A)


def boozer_I_to_physical_current_A(boozer_I: float) -> float:
    return float(boozer_I) / MU0_OVER_2PI


def resolve_plasma_current_settings(
    *,
    raw_boozer_I: float | None,
    plasma_current_A: float | None,
) -> PlasmaCurrentSettings:
    if plasma_current_A is not None:
        if raw_boozer_I is not None:
            raise ValueError("Cannot use --plasma-current-A together with --boozer-I")
        return PlasmaCurrentSettings(
            boozer_I=physical_current_to_boozer_I(plasma_current_A),
            plasma_current_A=float(plasma_current_A),
            input_source="physical_A",
            mode="boozer_surrogate",
        )

    if raw_boozer_I is not None:
        return PlasmaCurrentSettings(
            boozer_I=float(raw_boozer_I),
            plasma_current_A=boozer_I_to_physical_current_A(raw_boozer_I),
            input_source="raw_boozer_I",
            mode="boozer_surrogate",
        )

    return PlasmaCurrentSettings(
        boozer_I=0.0,
        plasma_current_A=0.0,
        input_source="default_zero",
        mode="disabled",
    )
