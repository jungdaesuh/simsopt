from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol, Sequence

from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, Current

from .current_contracts import apply_banana_current_upper_bound, banana_current_exceeds_limit
from .stage2_single_stage_handoff import Stage2CoilPartitions


BananaCurrentMode = Literal["shared", "independent"]
BANANA_CURRENT_MODE_SHARED: BananaCurrentMode = "shared"
BANANA_CURRENT_MODE_INDEPENDENT: BananaCurrentMode = "independent"
BANANA_CURRENT_CONTROL_METRIC_MAX_ABS = "max_abs"

__all__ = [
    "BANANA_CURRENT_CONTROL_METRIC_MAX_ABS",
    "BANANA_CURRENT_MODE_INDEPENDENT",
    "BANANA_CURRENT_MODE_SHARED",
    "BananaCurrentMode",
    "SingleStageBananaCurrentState",
    "apply_single_stage_penalty_banana_current_bounds",
    "build_single_stage_banana_current_state",
    "build_single_stage_banana_current_payload_fields",
    "resolve_single_stage_banana_current_state",
]


class CurrentLike(Protocol):
    def get_value(self) -> float: ...


@dataclass(frozen=True)
class SingleStageBananaCurrentState:
    mode: BananaCurrentMode
    currents: tuple[CurrentLike, ...]
    seed_currents_A: tuple[float, ...]

    def current_values_A(self) -> tuple[float, ...]:
        return tuple(float(current.get_value()) for current in self.currents)

    def control_current_A(self) -> float | None:
        return max((abs(current_A) for current_A in self.current_values_A()), default=None)

    def representative_current_A(self) -> float | None:
        if self.mode != BANANA_CURRENT_MODE_SHARED or not self.currents:
            return None
        return float(self.currents[0].get_value())

    def compatibility_current_A(self) -> float | None:
        representative_current_A = self.representative_current_A()
        if representative_current_A is not None:
            return representative_current_A
        return self.control_current_A()

    def num_currents(self) -> int:
        return len(self.currents)

    def num_control_currents(self) -> int:
        if not self.currents:
            return 0
        if self.mode == BANANA_CURRENT_MODE_SHARED:
            return 1
        return len(self.currents)


def _validated_banana_current_mode(mode: str) -> BananaCurrentMode:
    if mode == BANANA_CURRENT_MODE_SHARED:
        return BANANA_CURRENT_MODE_SHARED
    if mode == BANANA_CURRENT_MODE_INDEPENDENT:
        return BANANA_CURRENT_MODE_INDEPENDENT
    raise ValueError(f"Unsupported single-stage banana current mode {mode!r}.")


def _seed_currents_from_coils(banana_coils: Sequence[Coil]) -> tuple[float, ...]:
    return tuple(float(coil.current.get_value()) for coil in banana_coils)


def build_single_stage_banana_current_state(
    banana_coils: Sequence[Coil],
    *,
    mode: BananaCurrentMode,
    seed_currents_A: Sequence[float] | None = None,
) -> SingleStageBananaCurrentState:
    resolved_seed_currents_A = (
        _seed_currents_from_coils(banana_coils)
        if seed_currents_A is None
        else tuple(float(current_A) for current_A in seed_currents_A)
    )
    return SingleStageBananaCurrentState(
        mode=_validated_banana_current_mode(str(mode)),
        currents=tuple(coil.current for coil in banana_coils),
        seed_currents_A=resolved_seed_currents_A,
    )


def _build_independent_banana_coils(
    banana_coils: Sequence[Coil],
) -> tuple[tuple[Coil, ...], tuple[float, ...]]:
    seed_currents_A = _seed_currents_from_coils(banana_coils)
    return (
        tuple(
            Coil(coil.curve, Current(seed_current_A))
            for coil, seed_current_A in zip(banana_coils, seed_currents_A)
        ),
        seed_currents_A,
    )


def resolve_single_stage_banana_current_state(
    biot_savart: BiotSavart,
    coil_partitions: Stage2CoilPartitions,
    *,
    mode: BananaCurrentMode,
) -> tuple[BiotSavart, Stage2CoilPartitions, SingleStageBananaCurrentState]:
    resolved_mode = _validated_banana_current_mode(str(mode))
    banana_coils = tuple(coil_partitions.banana_coils)
    if resolved_mode == BANANA_CURRENT_MODE_SHARED:
        return (
            biot_savart,
            coil_partitions,
            build_single_stage_banana_current_state(
                banana_coils,
                mode=resolved_mode,
            ),
        )

    rebuilt_banana_coils, seed_currents_A = _build_independent_banana_coils(
        banana_coils
    )
    rebuilt_banana_by_original_id = {
        id(original_coil): rebuilt_coil
        for original_coil, rebuilt_coil in zip(banana_coils, rebuilt_banana_coils)
    }
    biot_savart_coil_ids = {id(coil) for coil in biot_savart.coils}
    missing_banana_coil_ids = set(rebuilt_banana_by_original_id) - biot_savart_coil_ids
    if missing_banana_coil_ids:
        raise ValueError(
            "Stage 2 banana coils are not all present in the Biot-Savart coil list; "
            "cannot rebuild independent-mode banana currents without losing the "
            "original coil ordering."
        )
    rebuilt_coils = [
        rebuilt_banana_by_original_id.get(id(coil), coil) for coil in biot_savart.coils
    ]
    rebuilt_biot_savart = BiotSavart(rebuilt_coils)
    rebuilt_partitions = replace(
        coil_partitions,
        banana_coils=rebuilt_banana_coils,
        num_banana_coils=len(rebuilt_banana_coils),
    )
    current_state = build_single_stage_banana_current_state(
        rebuilt_banana_coils,
        mode=resolved_mode,
        seed_currents_A=seed_currents_A,
    )
    return rebuilt_biot_savart, rebuilt_partitions, current_state


def apply_single_stage_penalty_banana_current_bounds(
    state: SingleStageBananaCurrentState,
    *,
    banana_current_max_A: float,
    validate_seed: bool,
    seed_context: str,
) -> None:
    for current, seed_current_A in zip(state.currents, state.seed_currents_A):
        if validate_seed and banana_current_exceeds_limit(
            seed_current_A,
            banana_current_max_A,
        ):
            raise ValueError(
                f"{seed_context} banana_current={float(seed_current_A):.6f} exceeds the "
                f"traversal-forbidden penalty box bound {float(banana_current_max_A):.6f}."
            )
        apply_banana_current_upper_bound(current, banana_current_max_A)


def build_single_stage_banana_current_payload_fields(
    state: SingleStageBananaCurrentState | None,
    *,
    prefix: str = "",
) -> dict[str, object]:
    if state is None:
        mode = None
        current_values_A: list[float] | None = None
        max_abs_A: float | None = None
        control_metric: str | None = None
        num_controls: int | None = None
        scalar_current_A: float | None = None
    else:
        values = state.current_values_A()
        max_abs = max((abs(value) for value in values), default=None)
        representative = state.representative_current_A()
        mode = state.mode
        current_values_A = list(values)
        max_abs_A = None if max_abs is None else float(max_abs)
        control_metric = (
            None if max_abs is None else BANANA_CURRENT_CONTROL_METRIC_MAX_ABS
        )
        num_controls = state.num_control_currents()
        scalar = representative if representative is not None else max_abs
        scalar_current_A = None if scalar is None else float(scalar)
    return {
        f"{prefix}BANANA_CURRENT_MODE": mode,
        f"{prefix}BANANA_CURRENTS_A": current_values_A,
        f"{prefix}BANANA_CURRENT_MAX_ABS_A": max_abs_A,
        f"{prefix}BANANA_CURRENT_CONTROL_METRIC": control_metric,
        f"{prefix}BANANA_NUM_CURRENT_CONTROLS": num_controls,
        f"{prefix}BANANA_CURRENT_A": scalar_current_A,
    }
