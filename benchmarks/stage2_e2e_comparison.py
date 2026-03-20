"""Tier 2 Stage 2 end-to-end optimization comparison probe."""

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
    build_provenance,
    find_single_file,
    load_json,
    max_pointwise_geometry_drift,
    preparse_platform,
    print_provenance,
    relative_error,
    repo_pythonpath_env,
    run_python_script,
    short_run_geometry_rel_tolerance,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)


FINAL_OBJECTIVE_REL_TOL = 1e-4
FIELD_ERROR_REL_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short Stage 2 optimization on CPU vs JAX and compare outcomes."
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument("--nphi", type=int, default=255, help="Surface toroidal grid points.")
    parser.add_argument("--ntheta", type=int, default=64, help="Surface poloidal grid points.")
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="Short but meaningful optimizer iteration budget.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured comparison results.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default="wout_nfp22ginsburg_000_014417_iota15.nc",
        help="VMEC equilibrium filename for the real Stage 2 fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=str(REPO_ROOT.parent / "DATABASE" / "EQUILIBRIA"),
        help="Directory that contains VMEC equilibrium files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Explicit equilibrium path override.",
    )
    parser.add_argument(
        "--geometry-rel-tol",
        type=float,
        default=None,
        help="Override the final banana-coil geometry relative tolerance.",
    )
    return parser.parse_args()


def _stage2_script_path() -> Path:
    return (
        REPO_ROOT
        / "examples"
        / "single_stage_optimization"
        / "STAGE_2"
        / "banana_coil_solver.py"
    )


def _run_stage2_case(args: argparse.Namespace, backend: str, *, platform: str) -> dict:
    script_path = _stage2_script_path()
    with tempfile.TemporaryDirectory(prefix=f"stage2-e2e-{backend}-") as temp_dir:
        trajectory_json = str(Path(temp_dir) / f"{backend}_trajectory.json")
        output_root = str(Path(temp_dir) / "outputs")

        command = [
            "--backend",
            backend,
            "--skip-postprocess",
            "--trajectory-json",
            trajectory_json,
            "--output-root",
            output_root,
            "--nphi",
            str(args.nphi),
            "--ntheta",
            str(args.ntheta),
            "--maxiter",
            str(args.maxiter),
        ]
        if args.equilibrium_path:
            command.extend(["--equilibrium-path", args.equilibrium_path])
        else:
            command.extend(
                [
                    "--plasma-surf-filename",
                    args.plasma_surf_filename,
                    "--equilibria-dir",
                    args.equilibria_dir,
                ]
            )

        start = time.perf_counter()
        result = run_python_script(
            script_path,
            command,
            env=repo_pythonpath_env(
                platform=platform if backend == "jax" else "cpu"
            ),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
        )
        elapsed_s = time.perf_counter() - start
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        results_json = find_single_file(output_root, "results.json")
        trajectory_payload = load_json(trajectory_json)
        return {
            "results": load_json(results_json),
            "trajectory": trajectory_payload["evaluations"],
            "elapsed_s": float(elapsed_s),
        }


def _trajectory_is_finite(trajectory: list[dict]) -> bool:
    for entry in trajectory:
        values = (
            entry["J"],
            entry["Jf"],
            entry["mean_abs_relBfinal_norm"],
            entry["curve_length"],
            entry["coil_coil_distance"],
            entry["curvature"],
            entry["grad_norm"],
        )
        if not np.all(np.isfinite(values)):
            return False
    return True


def _trajectory_improves(trajectory: list[dict]) -> bool:
    if not trajectory:
        return False
    return float(trajectory[-1]["J"]) <= float(trajectory[0]["J"])


def _max_geometry_deviation(cpu_results: dict, jax_results: dict) -> tuple[float, float]:
    cpu_gamma = np.asarray(cpu_results["FINAL_BANANA_GAMMA"], dtype=float)
    jax_gamma = np.asarray(jax_results["FINAL_BANANA_GAMMA"], dtype=float)
    return max_pointwise_geometry_drift(jax_gamma, cpu_gamma)


def evaluate_stage2_e2e_comparison(comparison: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if float(comparison["final_objective_rel_diff"]) >= FINAL_OBJECTIVE_REL_TOL:
        failures.append(
            "Final objective relative difference too large: "
            f"{float(comparison['final_objective_rel_diff']):.2e}"
        )
    if float(comparison["field_error_rel_diff"]) >= FIELD_ERROR_REL_TOL:
        failures.append(
            "Final field error relative difference too large: "
            f"{float(comparison['field_error_rel_diff']):.2e}"
        )
    if float(comparison["max_geometry_pointwise_rel"]) >= float(comparison["geometry_rel_tol"]):
        failures.append(
            "Final banana-coil geometry drift too large: "
            f"{float(comparison['max_geometry_pointwise_rel']):.2e} "
            f"relative (tol={float(comparison['geometry_rel_tol']):.2e})"
        )
    if not bool(comparison["cpu_trajectory_finite"]):
        failures.append("CPU trajectory contains NaN/inf.")
    if not bool(comparison["jax_trajectory_finite"]):
        failures.append("JAX trajectory contains NaN/inf.")
    if not bool(comparison["cpu_trajectory_improves"]):
        failures.append("CPU trajectory did not improve final objective.")
    if not bool(comparison["jax_trajectory_improves"]):
        failures.append("JAX trajectory did not improve final objective.")
    return failures


def build_stage2_e2e_payload(
    provenance: dict[str, Any],
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
    *,
    geometry_rel_tol: float,
) -> dict[str, Any]:
    cpu_results = cpu_case["results"]
    jax_results = jax_case["results"]
    cpu_trajectory = cpu_case["trajectory"]
    jax_trajectory = jax_case["trajectory"]

    max_geom_abs, max_geom_rel = _max_geometry_deviation(cpu_results, jax_results)
    final_objective_rel_diff = relative_error(
        float(jax_results["FINAL_OBJECTIVE"]),
        float(cpu_results["FINAL_OBJECTIVE"]),
    )

    comparison = {
        "final_objective_rel_diff": final_objective_rel_diff,
        "field_error_rel_diff": relative_error(
            float(jax_results["FIELD_ERROR"]),
            float(cpu_results["FIELD_ERROR"]),
        ),
        "field_error_rel_tol": FIELD_ERROR_REL_TOL,
        "max_geometry_pointwise_abs": max_geom_abs,
        "max_geometry_pointwise_rel": max_geom_rel,
        "geometry_rel_tol": geometry_rel_tol,
        "cpu_iterations": int(cpu_results["iterations"]),
        "jax_iterations": int(jax_results["iterations"]),
        "cpu_elapsed_s": float(cpu_case["elapsed_s"]),
        "jax_elapsed_s": float(jax_case["elapsed_s"]),
        "cpu_trajectory_len": len(cpu_trajectory),
        "jax_trajectory_len": len(jax_trajectory),
        "cpu_trajectory_finite": _trajectory_is_finite(cpu_trajectory),
        "jax_trajectory_finite": _trajectory_is_finite(jax_trajectory),
        "cpu_trajectory_improves": _trajectory_improves(cpu_trajectory),
        "jax_trajectory_improves": _trajectory_improves(jax_trajectory),
    }
    failures = evaluate_stage2_e2e_comparison(comparison)
    return {
        "provenance": provenance,
        "cpu_results": cpu_results,
        "jax_results": jax_results,
        "cpu_trajectory": cpu_trajectory,
        "jax_trajectory": jax_trajectory,
        "comparison": comparison,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    args = parse_args()
    geometry_rel_tol = short_run_geometry_rel_tolerance(
        args.maxiter,
        explicit_tol=args.geometry_rel_tol,
    )
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Stage 2 end-to-end comparison",
        extra={
            "fixture": "real-stage2",
            "platform_request": args.platform,
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "maxiter": int(args.maxiter),
            "geometry_rel_tol": geometry_rel_tol,
        },
    )
    print_provenance(provenance)

    cpu_case = _run_stage2_case(args, "cpu", platform="auto")
    jax_case = _run_stage2_case(args, "jax", platform=args.platform)

    payload = build_stage2_e2e_payload(
        provenance,
        cpu_case,
        jax_case,
        geometry_rel_tol=geometry_rel_tol,
    )
    comparison = payload["comparison"]
    failures = payload["failures"]

    print(
        "CPU vs JAX: "
        f"final objective rel_diff={comparison['final_objective_rel_diff']:.2e}, "
        f"field error rel_diff={comparison['field_error_rel_diff']:.2e}, "
        f"geometry rel_diff={comparison['max_geometry_pointwise_rel']:.2e}"
    )

    write_json(args.output_json, payload)
    if failures:
        print("STAGE 2 E2E COMPARISON FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("STAGE 2 E2E COMPARISON PASSED")


if __name__ == "__main__":
    main()
