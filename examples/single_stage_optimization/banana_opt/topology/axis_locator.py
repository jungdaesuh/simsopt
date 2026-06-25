"""Robust per-toroidal-plane magnetic-axis locator for coil (Biot-Savart) fields.

ROOT CAUSE this module exists to fix
------------------------------------
``locate_magnetic_axis_point`` finds the magnetic axis as the root of the field-line
return-map residual ``P(R, Z) - (R, Z)`` via ``scipy.least_squares``. On a strongly-shaped,
low-iota coil field the residual surface has a SECOND, spurious flat local minimum well
OUTBOARD of the true (elliptic) axis -- often the hyperbolic X-point's basin. Two failure
modes follow, both observed on real candidates:

  * Seeding the solve from the boundary major radius R0 drops the optimizer straight into
    the spurious basin: it terminates on a flat gradient at a non-axis point with residual
    ~7.5e-4 .. 3e-2 >> the 1e-6 acceptance tolerance.
  * Marching a SINGLE warm-started guess plane-to-plane (as the previous ``build_axis_model``
    did) compounds this: once one plane lands off-axis, every later plane inherits the bad
    seed, and -- with NO strict ``residual_tolerance`` forwarded -- the locator SILENTLY
    returns the ~0.03-residual non-axis point instead of failing, corrupting the direction
    field downstream.

The true axis sits INBOARD of R0 on these shaped fields and is reached from essentially any
guess at or inboard of R0, where it converges to residual 1e-13 .. 1e-16 (~13 orders below
tolerance). The fix is therefore a better GUESS plus a STRICT tolerance, never a looser one.

THE FIX
-------
At EACH plane run a small INBOARD-biased multi-start and keep only starts that converge
below the strict (unchanged) acceptance tolerance, returning the min-residual point:

  * Plane 0 (phi=0): an inboard radial x vertical grid bracketing the axis, biased inboard
    of the guess radius (the true-axis side).
  * Plane k>0: a LOCAL grid centered on plane (k-1)'s converged axis (the axis curve is
    continuous, so the previous plane is the best center), with a few inboard-and-around
    offsets; if NO local start converges (a large per-plane drift), fall back to the full
    inboard grid so the march cannot be stranded.

The strict ``residual_tolerance`` is forwarded UNCHANGED to ``locate_magnetic_axis_point``,
so a genuinely non-converged plane FAILS CLOSED (raises ``MagneticAxisNotLocatedError``)
instead of silently returning a non-axis point.

This module imports only ``numpy``, ``math``, and ``simsopt.field.magnetic_axis_helpers``;
it takes the geometric seed (a guess point + cylindrical bounds) explicitly rather than a
surface, so it stays decoupled from any particular surface representation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from simsopt.field.magnetic_axis_helpers import (
    MagneticAxisNotLocatedError,
    locate_magnetic_axis_point,
)

# Result-record key naming the multistart provenance (stable; recorded in diagnostics).
AXIS_CURVE_SOURCE = "magnetic_axis_curve_fieldline_fixed_point_multistart"


def _solve_plane(
    field,
    phi: float,
    centers: tuple[tuple[float, float], ...],
    *,
    nfp: int,
    r_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    residual_tolerance: float,
) -> Optional[dict]:
    """Try every (r, z) center at one toroidal plane; return the min-residual converged
    point (strictly below ``residual_tolerance``), or ``None`` if no center converges.

    Centers are clamped strictly inside ``r_bounds``/``z_bounds`` so ``least_squares`` never
    rejects an initial guess on the boundary. Convergence is judged solely by the locator's
    own strict acceptance (a non-converging start raises and is skipped); we never relax it.
    """
    eps = 1.0e-9
    lo_r, hi_r = r_bounds[0] + eps, r_bounds[1] - eps
    lo_z, hi_z = z_bounds[0] + eps, z_bounds[1] - eps
    converged: list[dict] = []
    for r_g, z_g in centers:
        r_clamped = min(max(float(r_g), lo_r), hi_r)
        z_clamped = min(max(float(z_g), lo_z), hi_z)
        try:
            point = locate_magnetic_axis_point(
                field,
                np.asarray([r_clamped, z_clamped], dtype=float),
                nfp=int(nfp),
                phi0=float(phi),
                r_bounds=r_bounds,
                z_bounds=z_bounds,
                residual_tolerance=float(residual_tolerance),
            )
        except MagneticAxisNotLocatedError:
            continue
        converged.append(point)
    if not converged:
        return None
    best = min(converged, key=lambda p: p["normalized_return_residual"])
    best["_n_converged"] = len(converged)
    best["_n_starts"] = len(centers)
    return best


def _inboard_global_centers(
    r_guess: float,
    z_guess: float,
    r_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    *,
    minor_scale: float,
    n_r: int,
    inboard_fraction: float,
    z_grid_fractions: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    """Inboard-biased radial x vertical start grid bracketing the axis (plane 0 + fallback).

    The radial span runs from ``inboard_fraction * r_guess`` up to ``r_guess`` (clamped inside
    ``r_bounds``): biased inboard because the true axis lies inboard of the boundary/guess
    radius on shaped fields. Vertical starts are ``z_guess + frac * minor_scale`` for each
    ``frac`` (handles up/down-shifted axes). ``minor_scale`` is the minor-radius length scale
    used to size the vertical offsets.
    """
    eps = 1.0e-9
    lower_r, upper_r = r_bounds
    r_lo = max(lower_r + eps, inboard_fraction * r_guess)
    r_hi = min(upper_r - eps, r_guess)
    if not r_lo < r_hi:
        r_lo, r_hi = lower_r + eps, upper_r - eps
    radial = np.linspace(r_lo, r_hi, int(n_r))
    lo_z, hi_z = z_bounds[0] + eps, z_bounds[1] - eps
    vertical = [
        min(max(z_guess + frac * minor_scale, lo_z), hi_z) for frac in z_grid_fractions
    ]
    return tuple((float(r), float(z)) for z in vertical for r in radial)


def locate_axis_curve(
    field,
    *,
    nfp: int,
    n_phi_planes: int,
    r_guess: float,
    z_guess: float = 0.0,
    r_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    residual_tolerance: float = 1.0e-6,
    n_r: int = 13,
    inboard_fraction: float = 0.78,
    z_grid_fractions: tuple[float, ...] = (0.0, 0.25, -0.25, 0.5, -0.5),
    local_r_fractions: tuple[float, ...] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05),
    local_z_fractions: tuple[float, ...] = (0.0, 0.05, -0.05, 0.10, -0.10),
) -> list[dict]:
    """Locate the magnetic axis at each of ``n_phi_planes`` equally-spaced toroidal planes.

    Robust per-plane inboard-multistart (see module docstring): plane 0 uses an inboard
    global grid around ``(r_guess, z_guess)``; each later plane uses a LOCAL grid centered on
    the previous plane's axis with an inboard-biased global fallback. The strict
    ``residual_tolerance`` is forwarded unchanged, so any plane that cannot reach an actual
    axis fails closed.

    Args:
        field: simsopt ``MagneticField`` (e.g. ``BiotSavart``).
        nfp: field-period count, forwarded to the locator as device metadata/validation. Each
            plane closes the return map over a FULL toroidal turn (an optimized coil field
            breaks exact nfp symmetry), so nfp is not a map shortener.
        n_phi_planes: number of equally-spaced planes ``phi_i = (i / n_phi_planes) * 2*pi``.
        r_guess, z_guess: cylindrical seed for plane 0 (typically the phi=0 axis from a prior
            robust locate). The plane-0 grid is biased inboard of ``r_guess``.
        r_bounds, z_bounds: cylindrical (R, Z) box the axis must lie in; passed to the locator
            and used to size and clamp the start grids. ``minor_scale`` (the vertical-offset
            length scale) is taken as half the R-bound width.
        residual_tolerance: strict acceptance tolerance, forwarded unchanged to
            ``locate_magnetic_axis_point``.
        n_r, inboard_fraction, z_grid_fractions: plane-0 / fallback global inboard grid.
        local_r_fractions, local_z_fractions: per-plane LOCAL grid offsets, as fractions of
            ``minor_scale``, around the previous plane's axis.

    Returns:
        list of ``n_phi_planes`` dicts, one per plane, each with ``phi``, ``r``, ``z``,
        ``nfp``, ``normalized_return_residual``, ``residual_accept_tolerance``,
        ``optimizer_success``, ``source`` (== ``AXIS_CURVE_SOURCE``), and diagnostics
        ``n_converged``, ``n_starts``, ``used_fallback``.

    Raises:
        MagneticAxisNotLocatedError: if ANY plane has no start converging below
            ``residual_tolerance`` (fail-closed; never returns a non-axis point).
    """
    lower_r, upper_r = float(r_bounds[0]), float(r_bounds[1])
    lower_z, upper_z = float(z_bounds[0]), float(z_bounds[1])
    if not (lower_r > 0.0 and upper_r > lower_r):
        raise ValueError(f"r_bounds invalid: {r_bounds!r} (need 0 < lo < hi)")
    if not (upper_z > lower_z):
        raise ValueError(f"z_bounds invalid: {z_bounds!r} (need lo < hi)")
    # Vertical-offset length scale: half the radial-bound width is the minor-radius scale the
    # bounds were built from (build_axis_model passes r_bounds = r_guess +/- 2a, so this is a).
    minor_scale = 0.25 * (upper_r - lower_r)

    global_centers = _inboard_global_centers(
        float(r_guess),
        float(z_guess),
        (lower_r, upper_r),
        (lower_z, upper_z),
        minor_scale=minor_scale,
        n_r=n_r,
        inboard_fraction=inboard_fraction,
        z_grid_fractions=z_grid_fractions,
    )

    two_pi = 2.0 * math.pi
    phis = [(i / int(n_phi_planes)) * two_pi for i in range(int(n_phi_planes))]

    results: list[dict] = []
    prev_r: Optional[float] = None
    prev_z: Optional[float] = None
    for idx, phi in enumerate(phis):
        used_fallback = False
        if idx == 0 or prev_r is None:
            centers = global_centers
        else:
            centers = tuple(
                (prev_r + dr * minor_scale, prev_z + dz * minor_scale)
                for dz in local_z_fractions
                for dr in local_r_fractions
            )
        best = _solve_plane(
            field, phi, centers, nfp=nfp,
            r_bounds=(lower_r, upper_r), z_bounds=(lower_z, upper_z),
            residual_tolerance=residual_tolerance,
        )
        if best is None and idx > 0:
            # Local cluster missed (large drift): retry on the global inboard grid so a big
            # per-plane jump cannot strand the march.
            used_fallback = True
            best = _solve_plane(
                field, phi, global_centers, nfp=nfp,
                r_bounds=(lower_r, upper_r), z_bounds=(lower_z, upper_z),
                residual_tolerance=residual_tolerance,
            )
        if best is None:
            raise MagneticAxisNotLocatedError(
                f"per-plane axis locate failed at plane {idx} (phi={phi:.6f}): no start "
                f"converged below {residual_tolerance:g}"
            )
        rec = {
            "phi": float(best["phi"]),
            "r": float(best["r"]),
            "z": float(best["z"]),
            "nfp": int(best["nfp"]),
            "normalized_return_residual": float(best["normalized_return_residual"]),
            "residual_accept_tolerance": float(residual_tolerance),
            "optimizer_success": bool(best["optimizer_success"]),
            "source": AXIS_CURVE_SOURCE,
            "n_converged": int(best["_n_converged"]),
            "n_starts": int(best["_n_starts"]),
            "used_fallback": bool(used_fallback),
        }
        results.append(rec)
        prev_r, prev_z = rec["r"], rec["z"]
    return results
