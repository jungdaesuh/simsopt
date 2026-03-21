"""Tier 3 real single-stage init parity probe on a fixed Columbia seed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    find_single_file,
    load_json,
    max_pointwise_geometry_drift,
    preparse_platform,
    print_provenance,
    relative_error,
    repo_pythonpath_env,
    run_python_script,
    write_json,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_OPTIMIZER_BACKEND,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)


IOTA_ABS_TOL = 1e-3
VOLUME_REL_TOL = 1e-6
FIELD_ERROR_REL_TOL = 1e-4
SURFACE_GEOMETRY_REL_TOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real single-stage init path on CPU vs JAX and compare outcomes."
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured comparison results.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the real single-stage fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=str(DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains VMEC equilibrium files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Explicit equilibrium path override.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Path to the fixed Stage 2 seed biot_savart_opt.json fixture.",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=DEFAULT_SMOKE_NPHI,
        help="Surface toroidal grid points.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=DEFAULT_SMOKE_NTHETA,
        help="Surface poloidal grid points.",
    )
    parser.add_argument(
        "--mpol",
        type=int,
        default=DEFAULT_SMOKE_MPOL,
        help="Surface poloidal mode count.",
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=DEFAULT_SMOKE_NTOR,
        help="Surface toroidal mode count.",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=DEFAULT_VOL_TARGET,
        help="Single-stage target volume.",
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        default=DEFAULT_IOTA_TARGET,
        help="Single-stage target iota.",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default=DEFAULT_OPTIMIZER_BACKEND,
        help="JAX Boozer optimizer backend for the init probe.",
    )
    return parser.parse_args()


def _single_stage_script_path() -> Path:
    return (
        REPO_ROOT
        / "examples"
        / "single_stage_optimization"
        / "SINGLE_STAGE"
        / "single_stage_banana_example.py"
    )


def _run_single_stage_case(
    args: argparse.Namespace,
    backend: str,
    *,
    platform: str,
) -> dict[str, Any]:
    script_path = _single_stage_script_path()
    with tempfile.TemporaryDirectory(prefix=f"single-stage-init-{backend}-") as temp_dir:
        output_root = Path(temp_dir) / "outputs"
        command = [
            "--backend",
            backend,
            "--init-only",
            "--output-root",
            str(output_root),
            "--plasma-surf-filename",
            args.plasma_surf_filename,
            "--stage2-bs-path",
            args.stage2_bs_path,
            "--nphi",
            str(args.nphi),
            "--ntheta",
            str(args.ntheta),
            "--mpol",
            str(args.mpol),
            "--ntor",
            str(args.ntor),
            "--vol-target",
            str(args.vol_target),
            "--iota-target",
            str(args.iota_target),
        ]
        if backend == "jax":
            command.extend(["--optimizer-backend", args.optimizer_backend])
        if args.equilibrium_path:
            command.extend(["--equilibrium-path", args.equilibrium_path])
        else:
            command.extend(["--equilibria-dir", args.equilibria_dir])

        start = time.perf_counter()
        result = run_python_script(
            script_path,
            command,
            env=repo_pythonpath_env(platform=platform if backend == "jax" else "cpu"),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        elapsed_s = time.perf_counter() - start

        results_json = find_single_file(output_root, "results.json")
        surf_json = find_single_file(output_root, "surf_init.json")
        results = dict(load_json(results_json))
        surface_gamma, surface_self_intersecting = _load_surface_artifacts(
            str(surf_json)
        )
        results["SELF_INTERSECTING"] = surface_self_intersecting
        return {
            "results": results,
            "surface_gamma": surface_gamma,
            "elapsed_s": float(elapsed_s),
        }


def _load_surface_artifacts(surface_json_path: str) -> tuple[np.ndarray, bool]:
    from simsopt._core.optimizable import load

    surface = load(surface_json_path)
    return (
        np.asarray(surface.gamma(), dtype=float),
        bool(surface.is_self_intersecting()),
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def evaluate_single_stage_init_parity(
    cpu_results: dict[str, Any],
    jax_results: dict[str, Any],
    *,
    max_surface_geometry_abs: float,
    max_surface_geometry_rel: float,
) -> tuple[dict[str, Any], list[str]]:
    comparison = {
        "final_iota_abs_diff": abs(
            float(jax_results["FINAL_IOTA"]) - float(cpu_results["FINAL_IOTA"])
        ),
        "final_volume_rel_diff": relative_error(
            float(jax_results["FINAL_VOLUME"]),
            float(cpu_results["FINAL_VOLUME"]),
        ),
        "field_error_rel_diff": relative_error(
            float(jax_results["FIELD_ERROR"]),
            float(cpu_results["FIELD_ERROR"]),
        ),
        "max_curvature_rel_diff": relative_error(
            float(jax_results["MAX_CURVATURE"]),
            float(cpu_results["MAX_CURVATURE"]),
        ),
        "max_surface_pointwise_abs": max_surface_geometry_abs,
        "max_surface_pointwise_rel": max_surface_geometry_rel,
        "cpu_self_intersecting": bool(cpu_results["SELF_INTERSECTING"]),
        "jax_self_intersecting": bool(jax_results["SELF_INTERSECTING"]),
    }

    failures: list[str] = []
    if comparison["final_iota_abs_diff"] >= IOTA_ABS_TOL:
        failures.append(
            f"Final iota disagreement too large: {comparison['final_iota_abs_diff']:.2e}"
        )
    if comparison["final_volume_rel_diff"] >= VOLUME_REL_TOL:
        failures.append(
            "Final volume relative difference too large: "
            f"{comparison['final_volume_rel_diff']:.2e}"
        )
    if comparison["field_error_rel_diff"] >= FIELD_ERROR_REL_TOL:
        failures.append(
            "Final field error relative difference too large: "
            f"{comparison['field_error_rel_diff']:.2e}"
        )
    if comparison["max_surface_pointwise_rel"] >= SURFACE_GEOMETRY_REL_TOL:
        failures.append(
            "Initial Boozer surface geometry drift too large: "
            f"{comparison['max_surface_pointwise_rel']:.2e} relative"
        )
    if comparison["cpu_self_intersecting"]:
        failures.append("CPU single-stage init produced a self-intersecting surface.")
    if comparison["jax_self_intersecting"]:
        failures.append("JAX single-stage init produced a self-intersecting surface.")
    return comparison, failures


def main() -> None:
    args = parse_args()
    bootstrap_local_simsopt()
    stage2_bs_path = Path(args.stage2_bs_path)
    if not stage2_bs_path.exists():
        raise RuntimeError(f"Stage 2 seed fixture does not exist: {stage2_bs_path}")
    stage2_results_path = stage2_bs_path.with_name("results.json")
    if not stage2_results_path.exists():
        raise RuntimeError(f"Stage 2 seed results.json does not exist: {stage2_results_path}")

    provenance = build_provenance(
        jax,
        jaxlib,
        title="Single-stage init parity",
        extra={
            "fixture": "real-single-stage-init",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": _display_path(stage2_bs_path),
            "optimizer_backend": args.optimizer_backend,
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "iota_abs_tol": IOTA_ABS_TOL,
            "volume_rel_tol": VOLUME_REL_TOL,
            "field_error_rel_tol": FIELD_ERROR_REL_TOL,
            "surface_geometry_rel_tol": SURFACE_GEOMETRY_REL_TOL,
        },
    )
    print_provenance(provenance)

    cpu_case = _run_single_stage_case(args, "cpu", platform="cpu")
    jax_case = _run_single_stage_case(args, "jax", platform=args.platform)

    cpu_results = cpu_case["results"]
    jax_results = jax_case["results"]
    max_geom_abs, max_geom_rel = max_pointwise_geometry_drift(
        jax_case["surface_gamma"],
        cpu_case["surface_gamma"],
    )
    comparison, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=max_geom_abs,
        max_surface_geometry_rel=max_geom_rel,
    )

    print(
        "CPU vs JAX: "
        f"|iota diff|={comparison['final_iota_abs_diff']:.2e}, "
        f"volume rel_diff={comparison['final_volume_rel_diff']:.2e}, "
        f"field error rel_diff={comparison['field_error_rel_diff']:.2e}, "
        f"surface rel_diff={comparison['max_surface_pointwise_rel']:.2e}"
    )

    payload = {
        "provenance": provenance,
        "cpu_results": cpu_results,
        "jax_results": jax_results,
        "comparison": comparison,
        "timings": {
            "cpu_elapsed_s": float(cpu_case["elapsed_s"]),
            "jax_elapsed_s": float(jax_case["elapsed_s"]),
        },
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.output_json, payload)
    if failures:
        print("SINGLE-STAGE INIT PARITY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("SINGLE-STAGE INIT PARITY PASSED")


if __name__ == "__main__":
    main()
