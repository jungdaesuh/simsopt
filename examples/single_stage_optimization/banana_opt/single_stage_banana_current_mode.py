from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol, Sequence

from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent

from .current_contracts import (
    apply_banana_current_upper_bound,
    banana_current_exceeds_limit,
    unwrap_current_optimizable,
)
from .stage2_single_stage_handoff import Stage2CoilPartitions


BananaCurrentMode = Literal["shared", "independent"]
BananaCurrentCoordinateScaling = Literal["none", "seed-relative"]
BANANA_CURRENT_MODE_SHARED: BananaCurrentMode = "shared"
BANANA_CURRENT_MODE_INDEPENDENT: BananaCurrentMode = "independent"
BANANA_CURRENT_COORDINATE_SCALING_NONE: BananaCurrentCoordinateScaling = "none"
BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE: BananaCurrentCoordinateScaling = (
    "seed-relative"
)
BANANA_CURRENT_CONTROL_METRIC_MAX_ABS = "max_abs"

__all__ = [
    "BananaCurrentCoordinateSpec",
    "BANANA_CURRENT_CONTROL_METRIC_MAX_ABS",
    "BANANA_CURRENT_COORDINATE_SCALING_NONE",
    "BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE",
    "BANANA_CURRENT_MODE_INDEPENDENT",
    "BANANA_CURRENT_MODE_SHARED",
    "BananaCurrentCoordinateScaling",
    "BananaCurrentMode",
    "SingleStageBananaCurrentState",
    "apply_single_stage_penalty_banana_current_bounds",
    "build_single_stage_banana_current_state",
    "build_single_stage_banana_current_payload_fields",
    "resolve_single_stage_banana_current_state",
    "resolve_banana_current_coordinate_spec",
]


class CurrentLike(Protocol):
    def get_value(self) -> float: ...


@dataclass(frozen=True)
class BananaCurrentCoordinateSpec:
    dof_names: tuple[str, ...]
    indices: tuple[int, ...]
    scale_factors_A: tuple[float, ...]


@dataclass(frozen=True)
class SingleStageBananaCurrentState:
    mode: BananaCurrentMode
    currents: tuple[CurrentLike, ...]
    seed_currents_A: tuple[float, ...]
    coordinate_scaling: BananaCurrentCoordinateScaling = (
        BANANA_CURRENT_COORDINATE_SCALING_NONE
    )
    current_coordinate_scale_factors_A: tuple[float, ...] = ()

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

    def optimizer_coordinate_values(self) -> tuple[float, ...]:
        values: list[float] = []
        seen_names: set[str] = set()
        for current in self.currents:
            dof_names = getattr(current, "dof_names", ())
            if len(dof_names) == 0:
                continue
            current_optimizable, _ = unwrap_current_optimizable(current)
            for dof_name in dof_names:
                resolved_name = str(dof_name)
                if resolved_name in seen_names:
                    continue
                seen_names.add(resolved_name)
                values.append(float(current_optimizable.get_value()))
        if values:
            return tuple(values)
        if self.mode == BANANA_CURRENT_MODE_SHARED:
            representative_current_A = self.representative_current_A()
            if representative_current_A is None:
                return ()
            return (representative_current_A,)
        return self.current_values_A()

    def coordinate_scale_factors_A(self) -> tuple[float, ...]:
        if len(self.current_coordinate_scale_factors_A) == self.num_control_currents():
            return self.current_coordinate_scale_factors_A
        scale_factors_A = tuple(
            scale for _, scale in _unique_current_control_entries(self)
        )
        if scale_factors_A:
            return scale_factors_A
        return tuple(1.0 for _ in range(self.num_control_currents()))


def _validated_banana_current_mode(mode: str) -> BananaCurrentMode:
    if mode == BANANA_CURRENT_MODE_SHARED:
        return BANANA_CURRENT_MODE_SHARED
    if mode == BANANA_CURRENT_MODE_INDEPENDENT:
        return BANANA_CURRENT_MODE_INDEPENDENT
    raise ValueError(f"Unsupported single-stage banana current mode {mode!r}.")


def _validated_banana_current_coordinate_scaling(
    coordinate_scaling: str,
) -> BananaCurrentCoordinateScaling:
    if coordinate_scaling == BANANA_CURRENT_COORDINATE_SCALING_NONE:
        return BANANA_CURRENT_COORDINATE_SCALING_NONE
    if coordinate_scaling == BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE:
        return BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE
    raise ValueError(
        f"Unsupported banana-current coordinate scaling {coordinate_scaling!r}."
    )


def _seed_currents_from_coils(banana_coils: Sequence[Coil]) -> tuple[float, ...]:
    return tuple(float(coil.current.get_value()) for coil in banana_coils)


def build_single_stage_banana_current_state(
    banana_coils: Sequence[Coil],
    *,
    mode: BananaCurrentMode,
    seed_currents_A: Sequence[float] | None = None,
    coordinate_scaling: BananaCurrentCoordinateScaling = (
        BANANA_CURRENT_COORDINATE_SCALING_NONE
    ),
) -> SingleStageBananaCurrentState:
    resolved_seed_currents_A = (
        _seed_currents_from_coils(banana_coils)
        if seed_currents_A is None
        else tuple(float(current_A) for current_A in seed_currents_A)
    )
    resolved_coordinate_scaling = _validated_banana_current_coordinate_scaling(
        str(coordinate_scaling)
    )
    currents = tuple(coil.current for coil in banana_coils)
    return SingleStageBananaCurrentState(
        mode=_validated_banana_current_mode(str(mode)),
        currents=currents,
        seed_currents_A=resolved_seed_currents_A,
        coordinate_scaling=resolved_coordinate_scaling,
        current_coordinate_scale_factors_A=tuple(
            scale for _, scale in _unique_current_control_entries_from_currents(currents)
        ),
    )


def _current_coordinate_scale_factor_A(current: CurrentLike) -> float:
    _, scale = unwrap_current_optimizable(current)
    return float(scale)


def _unique_current_control_entries(
    state: SingleStageBananaCurrentState,
) -> tuple[tuple[str, float], ...]:
    return _unique_current_control_entries_from_currents(state.currents)


def _unique_current_control_entries_from_currents(
    currents: Sequence[CurrentLike],
) -> tuple[tuple[str, float], ...]:
    ordered_entries: list[tuple[str, float]] = []
    seen_names: set[str] = set()
    for current in currents:
        dof_names = getattr(current, "dof_names", ())
        if len(dof_names) == 0:
            continue
        scale_factor_A = _current_coordinate_scale_factor_A(current)
        for dof_name in dof_names:
            resolved_name = str(dof_name)
            if resolved_name in seen_names:
                continue
            seen_names.add(resolved_name)
            ordered_entries.append((resolved_name, scale_factor_A))
    return tuple(ordered_entries)


def _unique_current_control_dof_names(
    state: SingleStageBananaCurrentState,
) -> tuple[str, ...]:
    return tuple(dof_name for dof_name, _ in _unique_current_control_entries(state))


def resolve_banana_current_coordinate_spec(
    objective_optimizable,
    state: SingleStageBananaCurrentState,
) -> BananaCurrentCoordinateSpec:
    current_control_entries = _unique_current_control_entries(state)
    current_dof_names = tuple(dof_name for dof_name, _ in current_control_entries)
    expected_control_count = int(state.num_control_currents())
    if len(current_dof_names) != expected_control_count:
        raise ValueError(
            "Banana current diagnostics expected "
            f"{expected_control_count} free current controls but found "
            f"{len(current_dof_names)} current DOF names: {current_dof_names!r}."
        )

    objective_dof_names = tuple(str(name) for name in objective_optimizable.dof_names)
    objective_index_by_name: dict[str, int] = {}
    duplicate_names: list[str] = []
    for index, dof_name in enumerate(objective_dof_names):
        if dof_name in objective_index_by_name:
            duplicate_names.append(dof_name)
            continue
        objective_index_by_name[dof_name] = index
    if duplicate_names:
        raise ValueError(
            "Banana current diagnostics require unique optimizer DOF names, but "
            f"found duplicates: {duplicate_names!r}."
        )

    missing_names = [
        dof_name for dof_name in current_dof_names if dof_name not in objective_index_by_name
    ]
    if missing_names:
        raise ValueError(
            "Banana current diagnostics could not locate current DOFs in the "
            "optimizer coordinate vector. Missing names: "
            f"{missing_names!r}."
        )

    return BananaCurrentCoordinateSpec(
        dof_names=current_dof_names,
        indices=tuple(objective_index_by_name[dof_name] for dof_name in current_dof_names),
        scale_factors_A=tuple(scale for _, scale in current_control_entries),
    )


def _scaled_current_for_optimizer_coordinate(
    seed_current_A: float,
    coordinate_scaling: BananaCurrentCoordinateScaling,
) -> CurrentLike:
    if coordinate_scaling == BANANA_CURRENT_COORDINATE_SCALING_NONE:
        return Current(seed_current_A)
    if coordinate_scaling == BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE:
        scale_A = abs(float(seed_current_A))
        if scale_A == 0.0:
            raise ValueError(
                "Seed-relative banana-current coordinate scaling requires "
                "non-zero loaded banana currents."
            )
        return ScaledCurrent(Current(float(seed_current_A) / scale_A), scale_A)
    raise AssertionError(f"Unhandled coordinate scaling {coordinate_scaling!r}.")


def _build_independent_banana_coils(
    banana_coils: Sequence[Coil],
    *,
    coordinate_scaling: BananaCurrentCoordinateScaling,
) -> tuple[tuple[Coil, ...], tuple[float, ...]]:
    seed_currents_A = _seed_currents_from_coils(banana_coils)
    return (
        tuple(
            Coil(
                coil.curve,
                _scaled_current_for_optimizer_coordinate(
                    seed_current_A,
                    coordinate_scaling,
                ),
            )
            for coil, seed_current_A in zip(banana_coils, seed_currents_A)
        ),
        seed_currents_A,
    )


def resolve_single_stage_banana_current_state(
    biot_savart: BiotSavart,
    coil_partitions: Stage2CoilPartitions,
    *,
    mode: BananaCurrentMode,
    coordinate_scaling: BananaCurrentCoordinateScaling = (
        BANANA_CURRENT_COORDINATE_SCALING_NONE
    ),
) -> tuple[BiotSavart, Stage2CoilPartitions, SingleStageBananaCurrentState]:
    resolved_mode = _validated_banana_current_mode(str(mode))
    resolved_coordinate_scaling = _validated_banana_current_coordinate_scaling(
        str(coordinate_scaling)
    )
    banana_coils = tuple(coil_partitions.banana_coils)
    if resolved_mode == BANANA_CURRENT_MODE_SHARED:
        if resolved_coordinate_scaling != BANANA_CURRENT_COORDINATE_SCALING_NONE:
            raise ValueError(
                "Banana-current coordinate scaling is only supported for "
                "independent banana-current mode."
            )
        return (
            biot_savart,
            coil_partitions,
            build_single_stage_banana_current_state(
                banana_coils,
                mode=resolved_mode,
            ),
        )

    rebuilt_banana_coils, seed_currents_A = _build_independent_banana_coils(
        banana_coils,
        coordinate_scaling=resolved_coordinate_scaling,
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
        coordinate_scaling=resolved_coordinate_scaling,
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
        coordinate_scaling = None
        current_values_A: list[float] | None = None
        optimizer_coordinates: list[float] | None = None
        scale_factors_A: list[float] | None = None
        max_abs_A: float | None = None
        control_metric: str | None = None
        num_controls: int | None = None
        scalar_current_A: float | None = None
    else:
        values = state.current_values_A()
        max_abs = max((abs(value) for value in values), default=None)
        representative = state.representative_current_A()
        mode = state.mode
        coordinate_scaling = state.coordinate_scaling
        current_values_A = list(values)
        optimizer_coordinates = list(state.optimizer_coordinate_values())
        scale_factors_A = list(state.coordinate_scale_factors_A())
        max_abs_A = None if max_abs is None else float(max_abs)
        control_metric = (
            None if max_abs is None else BANANA_CURRENT_CONTROL_METRIC_MAX_ABS
        )
        num_controls = state.num_control_currents()
        scalar = representative if representative is not None else max_abs
        scalar_current_A = None if scalar is None else float(scalar)
    return {
        f"{prefix}BANANA_CURRENT_MODE": mode,
        f"{prefix}BANANA_CURRENT_COORDINATE_SCALING": coordinate_scaling,
        f"{prefix}BANANA_CURRENTS_A": current_values_A,
        f"{prefix}BANANA_CURRENT_OPTIMIZER_COORDINATES": optimizer_coordinates,
        f"{prefix}BANANA_CURRENT_COORDINATE_SCALE_FACTORS_A": scale_factors_A,
        f"{prefix}BANANA_CURRENT_MAX_ABS_A": max_abs_A,
        f"{prefix}BANANA_CURRENT_CONTROL_METRIC": control_metric,
        f"{prefix}BANANA_NUM_CURRENT_CONTROLS": num_controls,
        f"{prefix}BANANA_CURRENT_A": scalar_current_A,
    }
