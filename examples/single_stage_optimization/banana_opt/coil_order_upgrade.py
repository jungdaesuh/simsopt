from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from simsopt.field import BiotSavart, Coil, coils_via_symmetries
from simsopt.field.coil import CurrentBase
from simsopt.geo import CurveCWSFourierCPP
from simsopt.geo.finitebuild import CurveFilament


def _resize_mode(mode: np.ndarray, target_size: int) -> np.ndarray:
    resized = np.zeros(target_size, dtype=float)
    overlap = min(mode.size, target_size)
    resized[:overlap] = np.asarray(mode, dtype=float)[:overlap]
    return resized


def upgrade_cws_order(
    curve: CurveCWSFourierCPP,
    new_order: int,
) -> CurveCWSFourierCPP:
    """Clone a CWS Fourier curve at a new order without mutating the source.

    Modes and per-DOF fix status are re-bound by semantic name
    (``phic(k)``, ``phis(k)``, ``thetac(k)``, ``thetas(k)``) rather than by
    flat-vector position. This matters as soon as ``new_order != old_order``:
    the layout groups DOFs as ``[phic, phis, thetac, thetas]`` with block
    sizes ``[O+1, O, O+1, O]``, so changing the order shifts the starting
    index of every block after ``phic``.
    """
    old_order = int(curve.order)
    target_order = int(new_order)
    if target_order < 0:
        raise ValueError("new_order must be non-negative.")
    if target_order < old_order:
        warnings.warn(
            "Truncating CurveCWSFourierCPP modes from "
            f"order={old_order} to order={target_order}.",
            RuntimeWarning,
            stacklevel=2,
        )

    upgraded_curve = CurveCWSFourierCPP(
        np.array(curve.quadpoints, copy=True),
        order=target_order,
        surf=curve.surf,
        G=int(curve.G),
        H=int(curve.H),
    )
    resized_modes = (
        _resize_mode(np.asarray(curve.modes[0], dtype=float), target_order + 1),
        _resize_mode(np.asarray(curve.modes[1], dtype=float), target_order),
        _resize_mode(np.asarray(curve.modes[2], dtype=float), target_order + 1),
        _resize_mode(np.asarray(curve.modes[3], dtype=float), target_order),
    )
    upgraded_curve.set_dofs(np.concatenate(resized_modes))

    source_full_names = list(curve.local_full_dof_names)
    source_free_status = np.asarray(curve.dofs.free_status, dtype=bool)
    fixed_source_names = {
        name
        for name, is_free in zip(
            source_full_names, source_free_status, strict=True
        )
        if not is_free
    }
    upgraded_full_names = set(upgraded_curve.local_full_dof_names)
    for name in fixed_source_names & upgraded_full_names:
        upgraded_curve.fix(name)
    return upgraded_curve


def _master_cws_curve(curve) -> CurveCWSFourierCPP | None:
    """Return the underlying ``CurveCWSFourierCPP`` master of a banana coil curve.

    A thin banana coil's curve IS the CWS master. A finite-build filament wraps it
    in a ``CurveFilament`` whose ``.curve`` attribute is that same master (the pack
    is built from one shared centerline), so unwrap one level. Returns ``None`` when
    neither holds.
    """
    if isinstance(curve, CurveCWSFourierCPP):
        return curve
    inner = getattr(curve, "curve", None)
    if isinstance(inner, CurveCWSFourierCPP):
        return inner
    return None


def _net_banana_current(coil: Coil) -> CurrentBase:
    """The NET banana current optimizable for a banana coil.

    A thin banana coil's ``coil.current`` IS the net current. A finite-build pack
    filament's ``coil.current`` is the per-filament ``ScaledCurrent`` (net/nfilaments)
    whose ``current_to_scale`` is the single shared NET optimizable driving the whole
    pack; recover that net so the upgraded family carries the seed's net current (not
    the filament scale), mirroring how the example unwraps it.
    """
    if isinstance(coil.curve, CurveFilament):
        return coil.current.current_to_scale
    return coil.current


def _resolve_master_banana_seed(
    banana_coils: Sequence[Coil],
) -> tuple[CurveCWSFourierCPP, CurrentBase]:
    for coil in banana_coils:
        master_curve = _master_cws_curve(coil.curve)
        if master_curve is not None:
            return master_curve, _net_banana_current(coil)
    raise ValueError(
        "Loaded banana coils do not contain a CurveCWSFourierCPP master curve."
    )


def realized_cws_winding_radii(
    banana_coils: Sequence[Coil],
) -> tuple[float, float] | None:
    """Return ``(major_radius_m, minor_radius_m)`` of the master CWS curve's
    EMBEDDED winding torus, or ``None`` when the lineage has no CWS master.

    Warm resumes optimize on the torus serialized inside the seed's
    ``CurveCWSFourierCPP`` — NOT on the ``--banana-surf-major-radius``
    reference torus (plot/cold-init only) and NOT necessarily on the
    ``BANANA_WINDING_SURFACE_MAJOR_RADIUS_M`` spec constant (the M-family
    lineage is embedded on 0.976/0.21; 2026-06-10 laneR0920 finding). Results
    artifacts must record the embedded torus so downstream consumers (seed
    remaps, hardware audits, ledgers) see the geometry the coils actually
    live on.
    """
    try:
        master_curve, _ = _resolve_master_banana_seed(banana_coils)
    except ValueError:
        return None
    surf = master_curve.surf
    if not hasattr(surf, "get_rc"):
        return None
    return float(surf.get_rc(0, 0)), float(surf.get_rc(1, 0))


def upgrade_loaded_seed_biot_savart_order(
    bs: BiotSavart,
    *,
    banana_coils: Sequence[Coil],
    tf_coils: Sequence[Coil],
    proxy_coils: Sequence[Coil],
    vf_coils: Sequence[Coil],
    new_order: int,
) -> tuple[BiotSavart, CurveCWSFourierCPP, list[Coil]]:
    master_curve, master_current = _resolve_master_banana_seed(banana_coils)
    upgraded_curve = upgrade_cws_order(master_curve, new_order)
    upgraded_banana_coils = list(
        coils_via_symmetries(
            [upgraded_curve],
            [master_current],
            upgraded_curve.surf.nfp,
            upgraded_curve.surf.stellsym,
        )
    )
    if len(upgraded_banana_coils) != len(banana_coils):
        raise ValueError(
            "Upgraded banana symmetry family has "
            f"{len(upgraded_banana_coils)} coils but the loaded seed "
            f"contains {len(banana_coils)}; refusing to silently reshape the "
            "Stage 2 coil-groups manifest."
        )
    upgraded_bs = BiotSavart(
        [*tf_coils, *upgraded_banana_coils, *proxy_coils, *vf_coils]
    )
    upgraded_bs.set_points(np.array(bs.get_points_cart_ref(), copy=True))
    return upgraded_bs, upgraded_curve, upgraded_banana_coils
