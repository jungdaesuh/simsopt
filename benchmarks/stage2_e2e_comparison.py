"""Tier 2 Stage 2 end-to-end optimization comparison probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

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
    preparse_platform,
    print_provenance,
    relative_error,
    repo_pythonpath_env,
    run_python_script,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)


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

        result = run_python_script(
            script_path,
            command,
            env=repo_pythonpath_env(
                platform=platform if backend == "jax" else "cpu"
            ),
            cwd=REPO_ROOT,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        results_json = find_single_file(output_root, "results.json")
        trajectory_payload = load_json(trajectory_json)
        return {
            "results": load_json(results_json),
            "trajectory": trajectory_payload["evaluations"],
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
    pointwise = np.linalg.norm(jax_gamma - cpu_gamma, axis=1)
    curve_scale = max(float(np.max(np.linalg.norm(cpu_gamma, axis=1))), 1e-30)
    return float(np.max(pointwise)), float(np.max(pointwise) / curve_scale)


def main() -> None:
    args = parse_args()
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
        },
    )
    print_provenance(provenance)

    cpu_case = _run_stage2_case(args, "cpu", platform="auto")
    jax_case = _run_stage2_case(args, "jax", platform=args.platform)

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
        "max_geometry_pointwise_abs": max_geom_abs,
        "max_geometry_pointwise_rel": max_geom_rel,
        "cpu_iterations": int(cpu_results["iterations"]),
        "jax_iterations": int(jax_results["iterations"]),
        "cpu_trajectory_len": len(cpu_trajectory),
        "jax_trajectory_len": len(jax_trajectory),
        "cpu_trajectory_finite": _trajectory_is_finite(cpu_trajectory),
        "jax_trajectory_finite": _trajectory_is_finite(jax_trajectory),
        "cpu_trajectory_improves": _trajectory_improves(cpu_trajectory),
        "jax_trajectory_improves": _trajectory_improves(jax_trajectory),
    }

    failures: list[str] = []
    if final_objective_rel_diff >= 1e-4:
        failures.append(
            f"Final objective relative difference too large: {final_objective_rel_diff:.2e}"
        )
    if max_geom_rel >= 1e-6:
        failures.append(
            f"Final banana-coil geometry drift too large: {max_geom_rel:.2e} relative"
        )
    if not comparison["cpu_trajectory_finite"]:
        failures.append("CPU trajectory contains NaN/inf.")
    if not comparison["jax_trajectory_finite"]:
        failures.append("JAX trajectory contains NaN/inf.")
    if not comparison["cpu_trajectory_improves"]:
        failures.append("CPU trajectory did not improve final objective.")
    if not comparison["jax_trajectory_improves"]:
        failures.append("JAX trajectory did not improve final objective.")

    print(
        "CPU vs JAX: "
        f"final objective rel_diff={comparison['final_objective_rel_diff']:.2e}, "
        f"field error rel_diff={comparison['field_error_rel_diff']:.2e}, "
        f"geometry rel_diff={comparison['max_geometry_pointwise_rel']:.2e}"
    )

    payload = {
        "provenance": provenance,
        "cpu_results": cpu_results,
        "jax_results": jax_results,
        "comparison": comparison,
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.output_json, payload)
    if failures:
        print("STAGE 2 E2E COMPARISON FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("STAGE 2 E2E COMPARISON PASSED")


if __name__ == "__main__":
    main()
