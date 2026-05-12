"""Smoke evaluation of colleague artifacts at HEAD.

Loads each of the 4 colleague artifacts via the finite-I bridge loader and
prints every quantity that the snapshot generator will freeze. Writes nothing
to disk. Used for ad-hoc diagnostic; the durable regression panel lives in
_generate_colleague_snapshots.py + test_colleague_artifact.py.

Run:
    OMP_NUM_THREADS=1 python tests/regression/_smoke_colleague_artifacts.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _helpers as H  # noqa: E402

import simsoptpp as sopp  # noqa: E402
from simsopt.geo.curveobjectives import CurveCurveDistance  # noqa: E402
from banana_opt.boozer_finite_current import (  # noqa: E402
    boozer_surface_residual_finite_I,
)
from banana_opt.json_compat import load_boozer_finite_i  # noqa: E402


def _evaluate_one(name: str, path: Path) -> None:
    print()
    print("=" * 72)
    print(f"ARTIFACT: {name}  ({path})")
    print("=" * 72)

    obj = load_boozer_finite_i(str(path))
    surface = obj.surface
    bs = obj.biotsavart
    I = obj.I
    bs.clear_cached_properties()

    print(f"  type             : {type(obj).__name__}")
    print(f"  I                : {I:.10e}")
    print(f"  surface          : {type(surface).__name__}, gamma.shape = {surface.gamma().shape}")
    print(f"  biotsavart coils : {len(bs.coils)}")

    # Surface geometry
    gamma = surface.gamma()
    normal = surface.normal()
    print("\n  --- surface geometry ---")
    print(f"    gamma   sha={H.sha_short(gamma)}  |gamma|_max={np.abs(gamma).max():.10e}")
    print(f"    normal  sha={H.sha_short(normal)}  |normal|_max={np.abs(normal).max():.10e}")

    # Volume
    print("\n  --- volume ---")
    print(f"    Volume.J() = {obj.label.J():.15e}")

    # BiotSavart at fixed eval points
    pts = H.eval_points(surface, seed=H.EVAL_POINTS_SEED, n=H.EVAL_POINTS_N)
    bs.set_points(pts)
    B = bs.B().copy()
    dB = bs.dB_by_dX().copy()
    print(f"\n  --- biot-savart @ {H.EVAL_POINTS_N} eval points (seed={H.EVAL_POINTS_SEED}) ---")
    print(f"    eval_pts sha={H.sha_short(pts)}")
    print(f"    B        sha={H.sha_short(B)}  |B|_max={np.linalg.norm(B, axis=1).max():.10e}")
    print(f"    dB/dX    sha={H.sha_short(dB)}  |dB|_max={np.abs(dB).max():.10e}")

    # First coil
    curve = bs.coils[0].curve
    cgamma = curve.gamma()
    cdg = curve.dgamma_by_dcoeff()
    print(f"\n  --- coil0 ({type(curve).__name__}) ---")
    print(f"    gamma     sha={H.sha_short(cgamma)}  shape={cgamma.shape}")
    print(f"    dgamma/dc sha={H.sha_short(cdg)}  shape={cdg.shape}")

    # Coil-coil distance
    curves = [c.curve for c in bs.coils]
    ccd = CurveCurveDistance(
        curves, minimum_distance=H.CCD_MIN_DISTANCE, downsample=H.CCD_DOWNSAMPLE
    )
    print("\n  --- coil-coil distance ---")
    print(f"    CurveCurveDistance(min={H.CCD_MIN_DISTANCE}, downsample={H.CCD_DOWNSAMPLE}).J() "
          f"= {ccd.J():.10e}")

    # Path-B Boozer kernel
    iota = H.BOOZER_KERNEL_IOTA
    sum_abs_I = sum(abs(c.current.get_value()) for c in bs.coils)
    G0 = H.MU0 * sum_abs_I
    G_eff = G0 + iota * I

    residual = boozer_surface_residual_finite_I(
        surface, iota, G0, bs, derivatives=0, weight_inv_modB=False, I=I
    )[0]
    wrapper_val = 0.5 * float(np.sum(residual ** 2))

    x = surface.gamma()
    bs.set_points(x.reshape(-1, 3).copy())
    B_at_surf = bs.B().reshape(x.shape)
    kernel_val = float(sopp.boozer_residual(
        G_eff, iota, surface.gammadash1(), surface.gammadash2(), B_at_surf, False
    ))
    kernel_at_G0 = float(sopp.boozer_residual(
        G0, iota, surface.gammadash1(), surface.gammadash2(), B_at_surf, False
    ))
    print(f"\n  --- Path-B Boozer kernel @ (iota={iota}, G=μ₀·Σ|I_coil|) ---")
    print(f"    Σ|I_coil|           = {sum_abs_I:.6e} A")
    print(f"    G_eff = G0 + iota*I = {G_eff:.10e}")
    print(f"    wrapper 0.5*||r||²  = {wrapper_val:.15e}")
    print(f"    raw kernel (G_eff)  = {kernel_val:.15e}")
    print(f"    raw kernel (G0)     = {kernel_at_G0:.15e}")
    print(f"    |wrapper - kernel|  = {abs(wrapper_val - kernel_val):.6e}")
    print(f"    finite-I non-trivial: {not np.isclose(kernel_val, kernel_at_G0)}")

    # Linearity probe — colleague's leaves are all fixed, so unfix-then-scale-then-restore.
    probe_pts = H.eval_points(surface, seed=H.LINEARITY_PROBE_SEED, n=H.LINEARITY_PROBE_N)
    bs.clear_cached_properties()
    bs.set_points(probe_pts)
    B0 = bs.B().copy()

    leaves, restore = H.scale_leaf_currents_in_memory(bs, 2.0)
    bs.clear_cached_properties()
    bs.set_points(probe_pts)
    B1 = bs.B().copy()
    restore()

    diff = float(np.max(np.abs(B1 - 2.0 * B0)))
    rel = diff / max(float(np.abs(B0).max()), 1e-30)
    print("\n  --- in-memory BS linearity (×2 on unique leaf Current DOFs) ---")
    print(f"    unique leaf currents : {len(leaves)}")
    print(f"    max |B1 - 2*B0|      : {diff:.6e}")
    print(f"    relative             : {rel:.6e}")
    print(f"    passes (bit-equal)   : {diff == 0.0}")


def main() -> int:
    env = H.env_summary()
    for k, v in env.items():
        print(f"{k:15s}: {v}")
    n_fail = 0
    for name, path in H.ARTIFACTS.items():
        try:
            _evaluate_one(name, path)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"\n!!! ARTIFACT {name} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print()
    print("=" * 72)
    print(f"SUMMARY: {len(H.ARTIFACTS) - n_fail}/{len(H.ARTIFACTS)} artifacts evaluated cleanly.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
