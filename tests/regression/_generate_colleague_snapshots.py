"""Generate snapshot JSON files for the colleague-artifact regression panel.

For each of the 4 colleague artifacts, evaluate the math-layer invariants
defined in docs/regression_panel_colleague_artifacts_2026-05-11.md §6.1 and
write one snapshot JSON per artifact under
``tests/regression/colleague_artifact_snapshots/``.

Snapshots are platform-pinned to the machine they were generated on (Darwin
ARM64 / Accelerate by default — see §6.4). Re-run only when intentional math
changes ship, with explicit reviewer sign-off and a justification entry.

Run:
    OMP_NUM_THREADS=1 python tests/regression/_generate_colleague_snapshots.py
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
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

_SNAPSHOT_FIRST_N = 10  # number of leading entries to retain for diagnostic readability


def _flat_prefix(arr, n: int = _SNAPSHOT_FIRST_N) -> list[float]:
    flat = np.ascontiguousarray(arr).reshape(-1)
    return [float(x) for x in flat[:n]]


def _build_snapshot(name: str, path: Path) -> dict:
    obj = load_boozer_finite_i(str(path))
    surface = obj.surface
    bs = obj.biotsavart
    I = obj.I
    bs.clear_cached_properties()

    snap: dict = {
        "_meta": {
            "artifact_file": path.name,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            **H.env_summary(),
            "format_version": 1,
        },
        "I": float(I),
        "boozer_surface_class": type(obj).__name__,
    }

    # Surface geometry
    gamma = surface.gamma()
    normal = surface.normal()
    snap["surface_geometry"] = {
        "type": type(surface).__name__,
        "gamma_shape": list(gamma.shape),
        "gamma_sha256": H.sha_full(gamma),
        "gamma_sample_first10_flat": _flat_prefix(gamma),
        "normal_sha256": H.sha_full(normal),
        "normal_sample_first10_flat": _flat_prefix(normal),
    }

    # Volume
    snap["volume"] = float(obj.label.J())

    # BiotSavart at fixed eval points
    pts = H.eval_points(surface, seed=H.EVAL_POINTS_SEED, n=H.EVAL_POINTS_N)
    bs.set_points(pts)
    B = bs.B().copy()
    dB = bs.dB_by_dX().copy()
    snap["biot_savart_eval"] = {
        "eval_points_seed": H.EVAL_POINTS_SEED,
        "n_eval_points": H.EVAL_POINTS_N,
        "eval_points_sha256": H.sha_full(pts),
        "B_sha256": H.sha_full(B),
        "B_sample_first10_flat": _flat_prefix(B),
        "dB_sha256": H.sha_full(dB),
        "dB_sample_first10_flat": _flat_prefix(dB),
    }

    # First coil geometry
    curve = bs.coils[0].curve
    cgamma = curve.gamma()
    cdg = curve.dgamma_by_dcoeff()
    snap["coil0_geometry"] = {
        "type": type(curve).__name__,
        "gamma_shape": list(cgamma.shape),
        "gamma_sha256": H.sha_full(cgamma),
        "gamma_sample_first10_flat": _flat_prefix(cgamma),
        "dgamma_dcoeff_shape": list(cdg.shape),
        "dgamma_dcoeff_sha256": H.sha_full(cdg),
    }

    # CurveCurveDistance
    curves = [c.curve for c in bs.coils]
    ccd = CurveCurveDistance(
        curves, minimum_distance=H.CCD_MIN_DISTANCE, downsample=H.CCD_DOWNSAMPLE
    )
    snap["curve_curve_distance"] = {
        "minimum_distance": H.CCD_MIN_DISTANCE,
        "downsample": H.CCD_DOWNSAMPLE,
        "value": float(ccd.J()),
    }

    # Path-B Boozer kernel — wrapper-vs-raw kernel with G_eff = G0 + iota*I
    iota = H.BOOZER_KERNEL_IOTA
    sum_abs_I = float(sum(abs(c.current.get_value()) for c in bs.coils))
    G0 = H.MU0 * sum_abs_I
    G_eff = G0 + iota * float(I)

    residual = boozer_surface_residual_finite_I(
        surface, iota, G0, bs, derivatives=0, weight_inv_modB=False, I=float(I)
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

    snap["boozer_kernel_path_b"] = {
        "iota": iota,
        "G0_formula": "MU0 * sum(|coil.current.get_value()|)",
        "sum_abs_I_coil": sum_abs_I,
        "G0": G0,
        "G_eff": G_eff,
        "wrapper_residual_sqsum_half": wrapper_val,
        "raw_kernel_value": kernel_val,
        "raw_kernel_value_at_G0": kernel_at_G0,
        "weight_inv_modB": False,
    }

    return snap


def main() -> int:
    H.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, path in H.ARTIFACTS.items():
        if not path.exists():
            print(f"SKIP {name}: artifact missing at {path}")
            continue
        snap = _build_snapshot(name, path)
        out_path = H.SNAPSHOT_DIR / f"bsurf_opt_{name}kA.snapshot.json"
        with open(out_path, "w") as f:
            json.dump(snap, f, indent=2, sort_keys=False)
            f.write("\n")
        print(f"wrote {out_path}  (I = {snap['I']:.6e})")
        written += 1
    print(f"\n{written}/{len(H.ARTIFACTS)} snapshots written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
