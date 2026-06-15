"""Offline REGCOIL/NESCOIL floor diagnostic driver.

Given one or more finished single-stage run artifact directories, computes the
unrestricted current-potential lower bound on the quadratic flux achievable by ANY
divergence-free surface current on the winding surface, recomputes the achieved
flux of the run's banana coils on the IDENTICAL scaled plasma grid + reduction, and
writes a JSON sidecar decomposing the residual into a winding-surface-physics floor
and a coil-parameterization deficit (an upper bound, since the single-shared-current
banana family cannot reach the unrestricted floor).

Reads the run's ``results.json`` (UPPERCASE keys) for the plasma wout path, toroidal
flux label, LCFS scaling, winding-surface radii, and net TF current; each is
overridable on the CLI. The achieved flux is RECOMPUTED from ``biot_savart_opt.json``
(results.json has no field-objective key) so achieved and floor share the exact same
``SquaredFlux(definition="quadratic flux")`` reduction on the same surface object.

Mirrors the structure of run_residue_probe.py: local-simsopt import boilerplate,
``parse_args``, ``evaluate_output_dir``, and a ``main`` that writes a JSON sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

from simsopt import load  # noqa: E402
from simsopt.geo import SurfaceRZFourier  # noqa: E402
from simsopt.geo.surface import Surface  # noqa: E402
from simsopt.objectives import SquaredFlux  # noqa: E402

from banana_opt.hardware_contracts import (  # noqa: E402
    BANANA_WINDING_MINOR_RADIUS_M,
    BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
)
from banana_opt.regcoil_floor import (  # noqa: E402
    build_current_potential_basis,
    compute_regcoil_floor,
    net_poloidal_current_amperes,
    poloidal_current_constant_tesla_meter,
    surface_quadrature,
)
from banana_opt.stage2_geometry import load_plasma_geometry  # noqa: E402

SCHEMA_VERSION = "regcoil_floor_runner_v1"

# Hardware-pinned winding-surface minor radius (m); sensor-array derived, "not a
# free knob" per the hardware contracts (SSOT: BANANA_WINDING_MINOR_RADIUS_M). The
# artifact under audit realized 0.143 at the rescinded 06-11 retarget, so
# --winding-minor-radius lets a faithful comparison pass the artifact's realized
# radius while the default stays on spec.
DEFAULT_WINDING_MINOR_RADIUS_M = BANANA_WINDING_MINOR_RADIUS_M

# Below this floor/achieved ratio the unregularized current sheet has nulled B.n to
# (near) machine zero — a trivial unrestricted-sheet limit, not a surface-quality
# discriminator. See DEGENERATE_FLOOR_WARNING / floor_regime emission.
DEGENERATE_FLOOR_RATIO_THRESHOLD = 1.0e-3

DEGENERATE_FLOOR_WARNING = (
    "Unregularized (lambda=0) floor is at the trivial unrestricted-current-sheet "
    "limit: an N-mode single-valued current potential with no current-complexity "
    "bound nulls B.n to ~machine zero, so floor_field_objective ~ 0 by "
    "construction. This floor is NOT a surface-quality discriminator and the "
    "achieved/floor ratio is uninformative: the achieved-vs-floor deficit is "
    "dominated by the coil PARAMETERIZATION (single-shared-current banana family), "
    "not the winding surface. To obtain the surface-quality signal, run a "
    "current-complexity-bounded (lambda > 0) sweep (see --lambda-sweep) and read "
    "the regularized floor / Pareto knee."
)

CAVEATS = (
    "REGCOIL floor is an UNRESTRICTED-surface-current lower bound; the "
    "single-shared-current banana family cannot reach it, so "
    "parameterization_deficit_upper_bound (achieved - floor) is an UPPER bound on "
    "the coil-parameterization deficit, not the realizable gap."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the offline REGCOIL/NESCOIL quadratic-flux floor on the "
            "winding surface and decompose the achieved/floor residual for one or "
            "more single-stage output directories."
        )
    )
    parser.add_argument(
        "output_dirs",
        nargs="+",
        help="Run artifact directories containing results.json and biot_savart_opt.json.",
    )
    parser.add_argument(
        "--report-path",
        default="regcoil_floor_report.json",
        help="Path to the JSON report to write.",
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=None,
        help="Normalized toroidal flux label s of the working surface "
        "(default: results.json TOROIDAL_FLUX, else 0.24).",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=255,
        help="Plasma-surface toroidal quadrature points (matches the solver default).",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=64,
        help="Plasma-surface poloidal quadrature points (matches the solver default).",
    )
    parser.add_argument(
        "--winding-major-radius",
        type=float,
        default=None,
        help="Winding-surface major radius R0 (m) "
        "(default: results.json BANANA_WINDING_SURFACE_MAJOR_RADIUS_M, else the "
        f"hardware-contract spec {BANANA_WINDING_SURFACE_MAJOR_RADIUS_M}).",
    )
    parser.add_argument(
        "--winding-minor-radius",
        type=float,
        default=DEFAULT_WINDING_MINOR_RADIUS_M,
        help="Winding-surface minor radius a (m). Defaults to the hardware spec "
        f"{DEFAULT_WINDING_MINOR_RADIUS_M}; pass the artifact's realized "
        "banana_surf_radius (e.g. 0.143) for a faithful same-surface comparison.",
    )
    parser.add_argument(
        "--winding-nphi",
        type=int,
        default=128,
        help="Winding-surface toroidal quadrature points (Biot-Savart source grid).",
    )
    parser.add_argument(
        "--winding-ntheta",
        type=int,
        default=64,
        help="Winding-surface poloidal quadrature points (Biot-Savart source grid).",
    )
    parser.add_argument(
        "--regcoil-mpol",
        type=int,
        default=8,
        help="Single-valued current-potential poloidal mode cutoff.",
    )
    parser.add_argument(
        "--regcoil-ntor",
        type=int,
        default=8,
        help="Single-valued current-potential toroidal mode cutoff (per field period).",
    )
    parser.add_argument(
        "--lambda",
        dest="tikhonov_lambda",
        type=float,
        default=0.0,
        help="Tikhonov regularization on the current-potential coefficients. "
        "0 (default) gives the true (degenerate) unregularized floor.",
    )
    parser.add_argument(
        "--lambda-sweep",
        dest="lambda_sweep",
        type=str,
        default=None,
        help="Comma-separated Tikhonov lambda values (e.g. '0,1e-12,1e-10,1e-8') to "
        "evaluate the current-complexity-bounded floor at. When given, emits a "
        "lambda_sweep array (lambda, floor, max_abs_coefficient, cond) so the "
        "regularized-floor / Pareto-knee surface-quality signal is obtainable.",
    )
    parser.add_argument(
        "--toroidal-current-amperes",
        type=float,
        default=None,
        help="Net secular toroidal current I (A). Default: results.json "
        "PROXY_PLASMA_CURRENT_A (0 for the vacuum floor).",
    )
    return parser.parse_args()


def _parse_lambda_sweep(spec: str) -> list[float]:
    """Parse a comma-separated lambda-sweep spec into a list of nonneg floats.

    Fails loud on a malformed token or a negative value (genuinely invalid input);
    a single value is a one-point sweep.
    """
    values = [float(token) for token in spec.split(",") if token.strip() != ""]
    if not values:
        raise ValueError(f"--lambda-sweep {spec!r} parsed to no values.")
    if any(value < 0.0 for value in values):
        raise ValueError(f"--lambda-sweep values must be >= 0; got {values}.")
    return values


def _compute_lambda_sweep(
    *,
    plasma_quad,
    winding_quad,
    basis,
    net_poloidal_current_amperes: float,
    net_toroidal_current_amperes: float,
    lambdas: list[float],
) -> list[dict[str, float]]:
    """Floor / max|coefficient| / cond at each Tikhonov lambda.

    Emits the current-complexity-bounded floor curve so the regularized-floor /
    Pareto-knee surface-quality signal is obtainable. No knee auto-detection (KISS);
    just the raw sweep. ``inductance_cond`` is lambda-independent (a property of the
    unregularized operator) but is echoed per row for self-contained readability.
    """
    sweep: list[dict[str, float]] = []
    for tikhonov_lambda in lambdas:
        result = compute_regcoil_floor(
            plasma_quad=plasma_quad,
            winding_quad=winding_quad,
            basis=basis,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=net_toroidal_current_amperes,
            tikhonov_lambda=tikhonov_lambda,
        )
        max_abs_coefficient = (
            float(np.max(np.abs(result.coefficients)))
            if result.coefficients.size
            else 0.0
        )
        sweep.append(
            {
                "lambda": float(tikhonov_lambda),
                "floor": float(result.floor_field_objective),
                "max_abs_coefficient": max_abs_coefficient,
                "cond": float(result.inductance_cond),
            }
        )
    return sweep


def _read_results(output_dir: Path) -> dict:
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json in {output_dir}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _resolve_wout_path(results: dict, output_dir: Path) -> Path:
    wout = results.get("PLASMA_SURF_PATH") or results.get("PLASMA_SURF_FILENAME")
    if wout is None:
        raise KeyError(
            f"results.json in {output_dir} has no PLASMA_SURF_PATH / PLASMA_SURF_FILENAME"
        )
    wout_path = Path(wout)
    if not wout_path.exists():
        raise FileNotFoundError(f"Plasma wout file does not exist: {wout_path}")
    return wout_path


def _build_winding_surface(
    *, nfp: int, stellsym: bool, major_radius: float, minor_radius: float, nphi: int, ntheta: int
) -> SurfaceRZFourier:
    if minor_radius <= 0.0 or major_radius <= minor_radius:
        raise ValueError(
            f"Winding surface must have 0 < minor ({minor_radius}) < major ({major_radius})."
        )
    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=nphi, ntheta=ntheta, range="full torus", nfp=nfp
    )
    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    surface.set_rc(0, 0, major_radius)
    surface.set_rc(1, 0, minor_radius)
    surface.set_zs(1, 0, minor_radius)
    return surface


def evaluate_output_dir(output_dir: str | Path, *, args: argparse.Namespace) -> dict[str, object]:
    output_path = Path(output_dir).resolve()
    results = _read_results(output_path)
    wout_path = _resolve_wout_path(results, output_path)

    toroidal_flux = (
        float(args.toroidal_flux)
        if args.toroidal_flux is not None
        else float(results.get("TOROIDAL_FLUX", 0.24))
    )
    scaling_mode = str(results.get("STAGE2_PLASMA_SCALING_MODE", "lcfs"))
    if scaling_mode != "lcfs":
        raise ValueError(
            f"Only the 'lcfs' plasma scaling mode is supported here; "
            f"{output_path} reports STAGE2_PLASMA_SCALING_MODE={scaling_mode!r}."
        )
    target_lcfs_major_radius = float(results["FINAL_LCFS_MAJOR_RADIUS_M"])
    nfp = int(results.get("NFP", 5))
    winding_major = (
        float(args.winding_major_radius)
        if args.winding_major_radius is not None
        else float(
            results.get(
                "BANANA_WINDING_SURFACE_MAJOR_RADIUS_M",
                BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            )
        )
    )
    winding_minor = float(args.winding_minor_radius)
    num_tf_coils = int(results["NUM_TF_COILS"])
    tf_current_amperes = float(results["TF_CURRENT_A"])
    toroidal_current_amperes = (
        float(args.toroidal_current_amperes)
        if args.toroidal_current_amperes is not None
        else float(results.get("PROXY_PLASMA_CURRENT_A", 0.0))
    )

    # Build the scaled working (plasma) surface exactly as the solver did (lcfs mode).
    plasma_geometry = load_plasma_geometry(
        target_lcfs_major_radius,
        toroidal_flux,
        str(wout_path),
        args.nphi,
        args.ntheta,
    )
    working_surface = plasma_geometry.working_surface

    # Achieved flux: recompute on the IDENTICAL surface object + reduction.
    biot_savart_path = output_path / "biot_savart_opt.json"
    if not biot_savart_path.exists():
        raise FileNotFoundError(f"Missing biot_savart_opt.json in {output_path}")
    biot_savart = load(biot_savart_path)
    achieved_objective = float(SquaredFlux(working_surface, biot_savart).J())
    field_error_check = _area_weighted_field_error(working_surface, biot_savart)

    # Winding surface + REGCOIL floor.
    winding_surface = _build_winding_surface(
        nfp=nfp,
        stellsym=bool(working_surface.stellsym),
        major_radius=winding_major,
        minor_radius=winding_minor,
        nphi=args.winding_nphi,
        ntheta=args.winding_ntheta,
    )
    plasma_quad = surface_quadrature(working_surface)
    winding_quad = surface_quadrature(winding_surface)
    basis = build_current_potential_basis(
        winding_quad,
        mpol=args.regcoil_mpol,
        ntor=args.regcoil_ntor,
        nfp=nfp,
        stellsym=bool(working_surface.stellsym),
    )
    g_amperes = net_poloidal_current_amperes(num_tf_coils, tf_current_amperes)
    floor = compute_regcoil_floor(
        plasma_quad=plasma_quad,
        winding_quad=winding_quad,
        basis=basis,
        net_poloidal_current_amperes=g_amperes,
        net_toroidal_current_amperes=toroidal_current_amperes,
        tikhonov_lambda=float(args.tikhonov_lambda),
    )

    if floor.floor_field_objective > achieved_objective * (1.0 + 1.0e-6):
        raise ValueError(
            "REGCOIL floor exceeds the achieved field objective "
            f"(floor={floor.floor_field_objective:.6e} > achieved={achieved_objective:.6e}); "
            "the floor is a lower bound, so this signals a grid/units/surface bug, not physics."
        )

    # F1/F2: honest degenerate-floor handling. A zero (or near-zero relative to
    # achieved) unregularized floor is the trivial unrestricted-sheet limit, not a
    # surface-quality signal. Avoid divide-by-zero on a physically-valid zero floor.
    if achieved_objective <= 0.0:
        raise ValueError(
            f"Achieved field objective is non-positive ({achieved_objective:.6e}); "
            "cannot form an achieved/floor decomposition."
        )
    floor_value = floor.floor_field_objective
    if floor_value == 0.0:
        achieved_over_floor_ratio: float | None = None
    else:
        achieved_over_floor_ratio = achieved_objective / floor_value
    degenerate = floor_value < achieved_objective * DEGENERATE_FLOOR_RATIO_THRESHOLD
    floor_regime = (
        "unregularized_overfitting_limit" if degenerate else "regularized"
    )

    lambda_sweep = None
    if args.lambda_sweep is not None:
        lambda_sweep = _compute_lambda_sweep(
            plasma_quad=plasma_quad,
            winding_quad=winding_quad,
            basis=basis,
            net_poloidal_current_amperes=g_amperes,
            net_toroidal_current_amperes=toroidal_current_amperes,
            lambdas=_parse_lambda_sweep(args.lambda_sweep),
        )

    g_poloidal_tm = poloidal_current_constant_tesla_meter(num_tf_coils, tf_current_amperes)
    regcoil_block: dict[str, object] = {
        "mpol": int(args.regcoil_mpol),
        "ntor": int(args.regcoil_ntor),
        "n_basis": int(floor.n_basis),
        "lambda": float(args.tikhonov_lambda),
        "floor_field_objective": floor_value,
        "secular_only_field_objective": floor.secular_field_objective,
        "inductance_cond": floor.inductance_cond,
        "inductance_rank": int(floor.inductance_rank),
        "floor_regime": floor_regime,
    }
    if degenerate:
        regcoil_block["warning"] = DEGENERATE_FLOOR_WARNING
    if lambda_sweep is not None:
        regcoil_block["lambda_sweep"] = lambda_sweep
    return {
        "output_dir": str(output_path),
        "winding_surface": {
            "major": winding_major,
            "minor": winding_minor,
            "nfp": nfp,
            "stellsym": bool(winding_surface.stellsym),
            "nphi": int(args.winding_nphi),
            "ntheta": int(args.winding_ntheta),
        },
        "plasma_surface": {
            "s": toroidal_flux,
            "scaling_mode": scaling_mode,
            "scale_factor": float(plasma_geometry.scale_factor),
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "wout": str(wout_path),
            "working_major_radius_m": float(working_surface.major_radius()),
            "working_minor_radius_m": float(working_surface.minor_radius()),
        },
        "net_currents": {
            "G_poloidal_Tm": g_poloidal_tm,
            "G_poloidal_amperes": g_amperes,
            "I_toroidal_A": toroidal_current_amperes,
            "tf_current_A": tf_current_amperes,
            "num_tf_coils": num_tf_coils,
            "proxy_plasma_current_A": float(results.get("PROXY_PLASMA_CURRENT_A", 0.0)),
        },
        "regcoil": regcoil_block,
        "achieved": {
            "field_objective_recomputed": achieved_objective,
            "field_error_check": field_error_check,
        },
        "decomposition": {
            "achieved_over_floor_ratio": achieved_over_floor_ratio,
            "physics_floor": floor_value,
            "parameterization_deficit_upper_bound": achieved_objective - floor_value,
        },
        "caveats": CAVEATS,
    }


def _area_weighted_field_error(surface, biot_savart) -> float:
    """Area-weighted mean |B.n|/|B| on the surface grid.

    Reproduces the persisted ``results.json["FIELD_ERROR"]`` (see
    banana_opt plotting norm_field_plot): sum(|B.n|/|B| * |n|) / sum(|n|). Used as
    a sanity cross-check that the working surface + scaling were rebuilt correctly.
    """
    normal = surface.normal()
    abs_normal = np.linalg.norm(normal, axis=2)
    unit_normal = normal / abs_normal[..., None]
    biot_savart.set_points(surface.gamma().reshape((-1, 3)))
    b_field = biot_savart.B().reshape(normal.shape)
    bn = np.sum(b_field * unit_normal, axis=2)
    mod_b = np.linalg.norm(b_field, axis=2)
    return float(np.sum((np.abs(bn) / mod_b) * abs_normal) / np.sum(abs_normal))


def main() -> None:
    args = parse_args()
    cases = [evaluate_output_dir(output_dir, args=args) for output_dir in args.output_dirs]
    report = {
        "schema_version": SCHEMA_VERSION,
        "cases": cases,
    }
    report_path = Path(args.report_path).resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote REGCOIL floor report to {report_path}")


if __name__ == "__main__":
    main()
