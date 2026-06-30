from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from math import pi
from typing import Literal, Sequence

import numpy as np

from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, coils_to_makegrid
from simsopt.geo import CurveCWSFourierCPP, CurveXYZFourier, RotatedCurve

from .stage2_single_stage_handoff import Stage2CoilPartitions


BananaGeometryMode = Literal["shared_symmetry", "materialized_cws", "free_xyz"]
BANANA_GEOMETRY_MODE_SHARED_SYMMETRY: BananaGeometryMode = "shared_symmetry"
BANANA_GEOMETRY_MODE_MATERIALIZED_CWS: BananaGeometryMode = "materialized_cws"
BANANA_GEOMETRY_MODE_FREE_XYZ: BananaGeometryMode = "free_xyz"

# Points-per-period for the Cartesian Fourier resampling of a converted banana
# curve. The peak-curvature U-turn is fully resolved once the quadrature grid is
# dense relative to the bend (see 2026-06-21 free_xyz existence test: the source
# CurveCWSFourierCPP under-sampled its own peak on its 128-point default grid;
# order*FREE_XYZ_PPP quadpoints recover the true dense-grid kappa). 20 matches
# the simsopt CurveXYZFourier.load_curves_from_makegrid_file default.
FREE_XYZ_PPP = 20

__all__ = [
    "BANANA_GEOMETRY_MODE_FREE_XYZ",
    "BANANA_GEOMETRY_MODE_MATERIALIZED_CWS",
    "BANANA_GEOMETRY_MODE_SHARED_SYMMETRY",
    "BananaGeometryMode",
    "SingleStageBananaGeometryState",
    "convert_single_stage_banana_family_to_xyz",
    "materialize_cws_symmetry_curve",
    "resolve_single_stage_banana_geometry_state",
]


@dataclass(frozen=True)
class SingleStageBananaGeometryState:
    mode: BananaGeometryMode
    num_banana_curves: int
    num_independent_banana_curves: int
    max_materialization_error_m: float | None = None
    free_xyz_order: int | None = None

    def payload_fields(self, *, prefix: str = "") -> dict[str, object]:
        fields: dict[str, object] = {
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
        # Only the free_xyz mode carries a Cartesian Fourier order; omit the key for
        # the CWS modes so their results payload stays byte-identical.
        if self.free_xyz_order is not None:
            fields[f"{prefix}SINGLE_STAGE_BANANA_FREE_XYZ_ORDER"] = int(
                self.free_xyz_order
            )
        return fields


def _validated_banana_geometry_mode(mode: str) -> BananaGeometryMode:
    if mode == BANANA_GEOMETRY_MODE_SHARED_SYMMETRY:
        return BANANA_GEOMETRY_MODE_SHARED_SYMMETRY
    if mode == BANANA_GEOMETRY_MODE_MATERIALIZED_CWS:
        return BANANA_GEOMETRY_MODE_MATERIALIZED_CWS
    if mode == BANANA_GEOMETRY_MODE_FREE_XYZ:
        return BANANA_GEOMETRY_MODE_FREE_XYZ
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


def _rebuild_biot_savart_with_banana_coils(
    biot_savart: BiotSavart,
    original_banana_coils: Sequence[Coil],
    rebuilt_banana_coils: Sequence[Coil],
) -> BiotSavart:
    """Return a new ``BiotSavart`` with each original banana coil swapped for its
    rebuilt counterpart, preserving the loaded coil ordering (TF / banana / proxy
    / VF groups stay in place). The non-banana coils are reused by identity so the
    field sources are byte-identical. Raises when a rebuilt banana coil is absent
    from ``biot_savart.coils`` -- silently dropping it would reshape the coil-groups
    manifest the Stage 2 handoff depends on.
    """
    rebuilt_banana_by_original_id = {
        id(original_coil): rebuilt_coil
        for original_coil, rebuilt_coil in zip(
            original_banana_coils, rebuilt_banana_coils
        )
    }
    biot_savart_coil_ids = {id(coil) for coil in biot_savart.coils}
    missing_banana_coil_ids = set(rebuilt_banana_by_original_id) - biot_savart_coil_ids
    if missing_banana_coil_ids:
        raise ValueError(
            "Stage 2 banana coils are not all present in the Biot-Savart coil list; "
            "cannot rebuild banana geometry without preserving coil ordering."
        )
    rebuilt_coils = [
        rebuilt_banana_by_original_id.get(id(coil), coil) for coil in biot_savart.coils
    ]
    return BiotSavart(rebuilt_coils)


def resolve_single_stage_banana_geometry_state(
    biot_savart: BiotSavart,
    coil_partitions: Stage2CoilPartitions,
    *,
    mode: BananaGeometryMode,
) -> tuple[BiotSavart, Stage2CoilPartitions, SingleStageBananaGeometryState]:
    resolved_mode = _validated_banana_geometry_mode(str(mode))
    if resolved_mode == BANANA_GEOMETRY_MODE_FREE_XYZ:
        raise ValueError(
            "free_xyz banana geometry is resolved by "
            "convert_single_stage_banana_family_to_xyz, not "
            "resolve_single_stage_banana_geometry_state."
        )
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
    rebuilt_biot_savart = _rebuild_biot_savart_with_banana_coils(
        biot_savart, banana_coils, rebuilt_banana_coils
    )
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


def _convert_banana_coil_to_xyz(
    coil: Coil,
    *,
    order: int,
    ppp: int,
    tmpdir: str,
    index: int,
) -> tuple[Coil, float]:
    """Convert one banana ``Coil`` into a free ``CurveXYZFourier`` coil.

    The curve geometry is rendered to a MAKEGRID filament file (one curve, no
    symmetry expansion) and re-read through an FFT into a Cartesian Fourier
    series, so any banana curve type -- CWS master or symmetry wrapper -- becomes
    an independent 3-D curve with every x/y/z Fourier coefficient free. Returns
    the rebuilt coil (current preserved) and the max |gamma| reproduction error in
    metres between the source curve and the converted curve, sampled on the source
    curve's quadrature grid.
    """
    makegrid_path = os.path.join(tmpdir, f"banana_xyz_{index}.txt")
    coils_to_makegrid(makegrid_path, [coil.curve], [coil.current], nfp=1, stellsym=False)
    xyz_curve = CurveXYZFourier.load_curves_from_makegrid_file(
        makegrid_path, order=int(order), ppp=int(ppp)
    )[0]
    xyz_curve.unfix_all()
    source_gamma = np.asarray(coil.curve.gamma(), dtype=float)
    xyz_on_source_grid = CurveXYZFourier(
        np.asarray(coil.curve.quadpoints, dtype=float).copy(), int(order)
    )
    xyz_on_source_grid.x = np.asarray(xyz_curve.x, dtype=float).copy()
    reproduction_error_m = float(
        np.max(np.abs(source_gamma - np.asarray(xyz_on_source_grid.gamma(), dtype=float)))
    )
    return Coil(xyz_curve, coil.current), reproduction_error_m


def convert_single_stage_banana_family_to_xyz(
    biot_savart: BiotSavart,
    coil_partitions: Stage2CoilPartitions,
    *,
    order: int,
    ppp: int = FREE_XYZ_PPP,
) -> tuple[BiotSavart, Stage2CoilPartitions, SingleStageBananaGeometryState]:
    """Resolve the ``free_xyz`` banana geometry mode.

    Replaces every surface-locked banana coil with a free ``CurveXYZFourier``
    fit (one independent 3-D curve per loaded banana coil) and rebuilds the
    Biot-Savart object preserving coil ordering, mirroring the id-preserving
    rebuild in :func:`resolve_single_stage_banana_geometry_state`. This is the
    over-parameterized geometry the optimizer needs to relax the banana U-turn
    curvature off the winding surface. ``order`` is the Cartesian Fourier series
    order of each converted curve.
    """
    banana_coils = tuple(coil_partitions.banana_coils)
    with tempfile.TemporaryDirectory() as tmpdir:
        converted = [
            _convert_banana_coil_to_xyz(
                coil, order=int(order), ppp=int(ppp), tmpdir=tmpdir, index=index
            )
            for index, coil in enumerate(banana_coils)
        ]
    rebuilt_banana_coils = tuple(coil for coil, _ in converted)
    max_error_m = max((error for _, error in converted), default=0.0)
    rebuilt_biot_savart = _rebuild_biot_savart_with_banana_coils(
        biot_savart, banana_coils, rebuilt_banana_coils
    )
    rebuilt_partitions = replace(
        coil_partitions,
        banana_coils=rebuilt_banana_coils,
        num_banana_coils=len(rebuilt_banana_coils),
    )
    state = SingleStageBananaGeometryState(
        mode=BANANA_GEOMETRY_MODE_FREE_XYZ,
        num_banana_curves=len(rebuilt_banana_coils),
        num_independent_banana_curves=len(rebuilt_banana_coils),
        max_materialization_error_m=max_error_m,
        free_xyz_order=int(order),
    )
    return rebuilt_biot_savart, rebuilt_partitions, state
