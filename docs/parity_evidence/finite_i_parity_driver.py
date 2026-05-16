"""Finite-I parity driver — produces deterministic JSON for either the
pre-revert `BoozerSurface(I=...)` API or the HEAD `BoozerSurfaceFiniteI(I=...)`
wrapper API.

Run identically in both envs. Output: parity_<flavor>_<lane>.json with the
solved-state observables.
"""
from __future__ import annotations

import json
import os
import sys
import argparse
import hashlib

import numpy as np


def detect_flavor():
    """HEAD if BoozerSurfaceFiniteI is importable; pre-revert otherwise."""
    try:
        from banana_opt.boozer_finite_current import BoozerSurfaceFiniteI  # noqa: F401
        return "HEAD"
    except Exception:
        from simsopt.geo.boozersurface import BoozerSurface
        import inspect
        sig = inspect.signature(BoozerSurface.__init__)
        if "I" in sig.parameters:
            return "prerevert"
        raise RuntimeError(
            "Neither HEAD wrapper nor pre-revert I= kwarg available — wrong env"
        )


def build_fixture():
    """Build the deterministic fixture: NCSX coils + small area surface."""
    from simsopt.geo import SurfaceXYZTensorFourier
    from simsopt.geo.surfaceobjectives import Area
    from simsopt.configs.zoo import get_ncsx_data
    from simsopt.field.coil import coils_via_symmetries
    from simsopt.field.biotsavart import BiotSavart

    curves, currents, ma = get_ncsx_data()
    coils = coils_via_symmetries(curves, currents, 3, True)
    bs = BiotSavart(coils)

    current_sum = sum(abs(c.current.get_value()) for c in coils)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))

    mpol = 3
    ntor = 3
    phis = np.linspace(0, 1 / 3, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0, 1, 2 * mpol + 1, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=True, nfp=3,
        quadpoints_phi=phis, quadpoints_theta=thetas,
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)

    label = Area(surface)
    target_label = float(label.J())

    return {
        "bs": bs,
        "surface": surface,
        "label": label,
        "target_label": target_label,
        "G0": G0,
        "iota0": 0.4,  # NCSX rotational transform-like seed
    }


def fixture_hash(fixture):
    """Deterministic hash of the fixture inputs — proves both envs see the same seed."""
    payload = {
        "biotsavart_x": fixture["bs"].x.tolist(),
        "surface_x_init": fixture["surface"].x.tolist(),
        "target_label": fixture["target_label"],
        "G0": fixture["G0"],
        "iota0": fixture["iota0"],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def build_boozer_surface(fixture, *, mode, current_I):
    """Build BoozerSurface (pre-revert) or BoozerSurfaceFiniteI (HEAD).

    mode: "ls" (constraint_weight=100.) or "exact" (constraint_weight=None).
    """
    constraint_weight = 100.0 if mode == "ls" else None
    options = {"weight_inv_modB": False, "verbose": False}

    flavor = detect_flavor()
    if flavor == "HEAD":
        from banana_opt.boozer_finite_current import BoozerSurfaceFiniteI
        bsurf = BoozerSurfaceFiniteI(
            fixture["bs"],
            fixture["surface"],
            fixture["label"],
            fixture["target_label"],
            constraint_weight=constraint_weight,
            options=options,
            I=current_I,
        )
    else:
        from simsopt.geo.boozersurface import BoozerSurface
        bsurf = BoozerSurface(
            fixture["bs"],
            fixture["surface"],
            fixture["label"],
            fixture["target_label"],
            constraint_weight=constraint_weight,
            options=options,
            I=current_I,
        )
    return bsurf


def run_lane(*, mode, current_I, output_path):
    fixture = build_fixture()
    fh = fixture_hash(fixture)

    bsurf = build_boozer_surface(fixture, mode=mode, current_I=current_I)
    try:
        res = bsurf.run_code(fixture["iota0"], fixture["G0"])
        solve_error = None
    except Exception as exc:
        res = getattr(bsurf, "res", {}) or {}
        solve_error = f"{type(exc).__name__}: {exc}"

    surface_x = np.asarray(bsurf.surface.x, dtype=np.float64)

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    iota = _to_float(res.get("iota"))
    G = _to_float(res.get("G"))

    output = {
        "flavor": detect_flavor(),
        "mode": mode,
        "current_I": current_I,
        "fixture_hash": fh,
        "iota0_seed": fixture["iota0"],
        "G0_seed": fixture["G0"],
        "target_label": fixture["target_label"],
        "success": bool(res.get("success", False)),
        "solve_error": solve_error,
        "surface_x": surface_x.tolist(),
        "surface_x_norm": float(np.linalg.norm(surface_x)),
        "iota": iota,
        "G": G,
        "type": res.get("type", "unknown"),
    }
    if "I" in res:
        output["res_I"] = _to_float(res["I"])

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    iota_str = f"{iota:.15e}" if iota is not None else "None"
    G_str = f"{G:.15e}" if G is not None else "None"
    err_str = f" err={solve_error}" if solve_error else ""
    print(
        f"  wrote {output_path}: success={output['success']} "
        f"iota={iota_str} G={G_str}{err_str}"
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ls", "exact"], required=True)
    parser.add_argument("--current-I", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_lane(mode=args.mode, current_I=args.current_I, output_path=args.output)


if __name__ == "__main__":
    main()
