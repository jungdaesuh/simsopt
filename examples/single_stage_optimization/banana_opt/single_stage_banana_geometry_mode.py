from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from typing import Literal, Sequence

import numpy as np

from simsopt.field import BiotSavart
from simsopt.field.coil import Coil
from simsopt.geo import CurveCWSFourierCPP, RotatedCurve

from .stage2_single_stage_handoff import Stage2CoilPartitions


BananaGeometryMode = Literal["shared_symmetry", "materialized_cws"]
BANANA_GEOMETRY_MODE_SHARED_SYMMETRY: BananaGeometryMode = "shared_symmetry"
BANANA_GEOMETRY_MODE_MATERIALIZED_CWS: BananaGeometryMode = "materialized_cws"

__all__ = [
    "BANANA_GEOMETRY_MODE_MATERIALIZED_CWS",
    "BANANA_GEOMETRY_MODE_SHARED_SYMMETRY",
    "BananaGeometryMode",
    "SingleStageBananaGeometryState",
    "materialize_cws_symmetry_curve",
    "resolve_single_stage_banana_geometry_state",
]


@dataclass(frozen=True)
class SingleStageBananaGeometryState:
    mode: BananaGeometryMode
    num_banana_curves: int
    num_independent_banana_curves: int
    max_materialization_error_m: float | None = None

    def payload_fields(self, *, prefix: str = "") -> dict[str, object]:
        return {
            f"{prefix}SINGLE_STAGE_BANANA_GEOMETRY_MODE": self.mode,
            f"{prefix}SINGLE_STAGE_NUM_BANANA_GEOMETRY_CURVES": int(
                self.num_banana_curves
            ),
            f"{prefix}SINGLE_STAGE_NUM_INDEPENDENT_BANANA_GEOMETRY_CURVES": int(
                self.num_independent_banana_curves
            ),
            f"{prefix}SINGLE_STAGE_BANANA_GEOMETRY_MATERIALIZATION_ERROR_M": (
                None
                if self.max_materialization_error_m is None
                else float(self.max_materialization_error_m)
            ),
        }


def _validated_banana_geometry_mode(mode: str) -> BananaGeometryMode:
    if mode == BANANA_GEOMETRY_MODE_SHARED_SYMMETRY:
        return BANANA_GEOMETRY_MODE_SHARED_SYMMETRY
    if mode == BANANA_GEOMETRY_MODE_MATERIALIZED_CWS:
        return BANANA_GEOMETRY_MODE_MATERIALIZED_CWS
    raise ValueError(f"Unsupported single-stage banana geometry mode {mode!r}.")


def _clone_cws_curve(curve: CurveCWSFourierCPP) -> CurveCWSFourierCPP:
    clone = CurveCWSFourierCPP(
        np.asarray(curve.quadpoints, dtype=float).copy(),
        int(curve.order),
        curve.surf,
        G=curve.G,
        H=curve.H,
    )
    clone.set_dofs(np.asarray(curve.get_dofs(), dtype=float).copy())
    return clone


def materialize_cws_symmetry_curve(curve) -> CurveCWSFourierCPP:
    """Return an independent CWS curve matching a CWS symmetry copy.

    ``coils_via_symmetries`` represents banana copies as ``RotatedCurve``
    wrappers around one CWS master. This function folds the same rotation and
    stellarator reflection into the CWS ``(phi, theta)`` coefficients so each
    banana route can be optimized independently while staying on the winding
    surface.
    """
    if isinstance(curve, CurveCWSFourierCPP):
        return _clone_cws_curve(curve)
    if not isinstance(curve, RotatedCurve):
        raise TypeError(
            "materialized_cws banana geometry requires CurveCWSFourierCPP or "
            f"CWS-backed RotatedCurve inputs, got {type(curve).__name__}."
        )
    source_curve = curve.curve
    if not isinstance(source_curve, CurveCWSFourierCPP):
        raise TypeError(
            "materialized_cws banana geometry requires RotatedCurve inputs to "
            "wrap CurveCWSFourierCPP masters."
        )

    dofs = np.asarray(source_curve.get_dofs(), dtype=float).copy()
    toroidal_phase_turns = float(curve._phi) / (2.0 * pi)
    if curve.flip:
        dofs = -dofs
        dofs[0] -= toroidal_phase_turns
        G = -source_curve.G
        H = -source_curve.H
    else:
        dofs[0] += toroidal_phase_turns
        G = source_curve.G
        H = source_curve.H

    materialized = CurveCWSFourierCPP(
        np.asarray(source_curve.quadpoints, dtype=float).copy(),
        int(source_curve.order),
        source_curve.surf,
        G=G,
        H=H,
    )
    materialized.set_dofs(dofs)
    return materialized


def _build_materialized_banana_coils(
    banana_coils: Sequence[Coil],
) -> tuple[tuple[Coil, ...], float]:
    rebuilt_coils: list[Coil] = []
    materialization_errors: list[float] = []
    for coil in banana_coils:
        materialized_curve = materialize_cws_symmetry_curve(coil.curve)
        materialization_errors.append(
            float(np.max(np.abs(coil.curve.gamma() - materialized_curve.gamma())))
        )
        rebuilt_coils.append(Coil(materialized_curve, coil.current))
    return tuple(rebuilt_coils), max(materialization_errors, default=0.0)


def resolve_single_stage_banana_geometry_state(
    biot_savart: BiotSavart,
    coil_partitions: Stage2CoilPartitions,
    *,
    mode: BananaGeometryMode,
) -> tuple[BiotSavart, Stage2CoilPartitions, SingleStageBananaGeometryState]:
    resolved_mode = _validated_banana_geometry_mode(str(mode))
    banana_coils = tuple(coil_partitions.banana_coils)
    if resolved_mode == BANANA_GEOMETRY_MODE_SHARED_SYMMETRY:
        state = SingleStageBananaGeometryState(
            mode=resolved_mode,
            num_banana_curves=len(banana_coils),
            num_independent_banana_curves=1 if banana_coils else 0,
        )
        return biot_savart, coil_partitions, state

    rebuilt_banana_coils, max_error_m = _build_materialized_banana_coils(
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
            "cannot materialize banana geometry without preserving coil ordering."
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
    state = SingleStageBananaGeometryState(
        mode=resolved_mode,
        num_banana_curves=len(rebuilt_banana_coils),
        num_independent_banana_curves=len(rebuilt_banana_coils),
        max_materialization_error_m=max_error_m,
    )
    return rebuilt_biot_savart, rebuilt_partitions, state
