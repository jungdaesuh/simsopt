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
    apply_compilation_cache_policy,
    apply_requested_platform,
    build_provenance,
    describe_compile_behavior,
    find_single_file,
    load_json,
    max_pointwise_geometry_drift,
    optimizer_drift_tolerances,
    preparse_platform,
    print_provenance,
    require_x64_runtime,
    relative_error,
    resolve_probe_lane,
    repo_pythonpath_env,
    run_python_script,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_compilation_cache_policy()

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Stage 2 end-to-end comparison")


_TIER2_BASE_TOLERANCES = optimizer_drift_tolerances("tier2_stage2_e2e")
FINAL_OBJECTIVE_REL_TOL = _TIER2_BASE_TOLERANCES["final_objective_rel_tol"]
FIELD_ERROR_REL_TOL = _TIER2_BASE_TOLERANCES["field_error_rel_tol"]


def _objective_not_worse(jax_value: float, cpu_value: float) -> bool:
    return float(jax_value) <= float(cpu_value) * (1.0 + FINAL_OBJECTIVE_REL_TOL)


def _field_error_not_worse(jax_value: float, cpu_value: float) -> bool:
    return float(jax_value) <= float(cpu_value) * (1.0 + FIELD_ERROR_REL_TOL)


def _build_ondevice_stage2_metrics(
    cpu_results: dict[str, Any],
    jax_results: dict[str, Any],
) -> dict[str, Any]:
    cpu_final_objective = float(cpu_results["FINAL_OBJECTIVE"])
    jax_final_objective = float(jax_results["FINAL_OBJECTIVE"])
    cpu_field_error = float(cpu_results["FIELD_ERROR"])
    jax_field_error = float(jax_results["FIELD_ERROR"])
    length_target = float(jax_results["LENGTH_TARGET"])
    jax_final_curve_length = float(jax_results["FINAL_CURVE_LENGTH"])
    cc_threshold = float(jax_results["CC_THRESHOLD"])
    jax_final_cc_distance = float(jax_results["FINAL_CC_DISTANCE"])
    curvature_threshold = float(jax_results["CURVATURE_THRESHOLD"])
    cpu_max_curvature = float(cpu_results["MAX_CURVATURE"])
    jax_max_curvature = float(jax_results["MAX_CURVATURE"])
    return {
        "cpu_final_objective": cpu_final_objective,
        "jax_final_objective": jax_final_objective,
        "jax_objective_not_worse_than_cpu": _objective_not_worse(
            jax_final_objective,
            cpu_final_objective,
        ),
        "cpu_field_error": cpu_field_error,
        "jax_field_error": jax_field_error,
        "jax_field_error_not_worse_than_cpu": _field_error_not_worse(
            jax_field_error,
            cpu_field_error,
        ),
        "length_target": length_target,
        "jax_final_curve_length": jax_final_curve_length,
        "jax_curve_length_within_target": jax_final_curve_length <= length_target,
        "cc_threshold": cc_threshold,
        "jax_final_cc_distance": jax_final_cc_distance,
        "jax_cc_distance_within_threshold": jax_final_cc_distance >= cc_threshold,
        "curvature_threshold": curvature_threshold,
        "cpu_max_curvature": cpu_max_curvature,
        "jax_max_curvature": jax_max_curvature,
        "jax_curvature_not_worse_than_cpu": jax_max_curvature
        <= max(cpu_max_curvature, curvature_threshold),
        "jax_self_intersecting": bool(jax_results["SELF_INTERSECTING"]),
    }


def _build_jax_stage2_timings(
    jax_case: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    jax_outer_elapsed_s = float(jax_case["elapsed_s"])
    jax_primary_elapsed_s = jax_outer_elapsed_s
    optimizer_timings = jax_case.get("optimizer_timings")
    if optimizer_timings is not None and "warm_run_s" in optimizer_timings:
        jax_primary_elapsed_s = max(
            jax_outer_elapsed_s - float(optimizer_timings["warm_run_s"]),
            0.0,
        )
    timings = {
        "jax_outer_elapsed_s": jax_outer_elapsed_s,
        "jax_primary_elapsed_s": jax_primary_elapsed_s,
    }
    if optimizer_timings is None:
        return jax_primary_elapsed_s, timings
    timings["jax_optimizer_cold_run_s"] = float(optimizer_timings["cold_run_s"])
    if "warm_run_s" in optimizer_timings:
        timings["jax_optimizer_warm_run_s"] = float(optimizer_timings["warm_run_s"])
    if "compile_overhead_s" in optimizer_timings:
        timings["jax_optimizer_compile_overhead_s"] = float(
            optimizer_timings["compile_overhead_s"]
        )
    return jax_primary_elapsed_s, timings


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
        "--optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default="scipy",
        help="Stage 2 optimizer backend for the JAX lane.",
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
        if backend == "jax":
            command.extend(["--optimizer-backend", args.optimizer_backend])
            if args.optimizer_backend == "ondevice":
                command.append("--record-warm-timings")
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
            stream_output=True,
        )
        elapsed_s = time.perf_counter() - start

        results_json = find_single_file(output_root, "results.json")
        results_payload = load_json(results_json)
        trajectory_payload = load_json(trajectory_json)
        return {
            "results": results_payload,
            "trajectory": trajectory_payload["evaluations"],
            "elapsed_s": float(elapsed_s),
            "optimizer_timings": results_payload.get("OPTIMIZER_TIMINGS"),
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


def _append_stage2_ondevice_failures(
    failures: list[str],
    comparison: dict[str, Any],
) -> None:
    checks = [
        (
            not bool(comparison["jax_objective_not_worse_than_cpu"]),
            lambda: (
                "Final objective is worse than the CPU reference beyond tolerance: "
                f"jax={float(comparison['jax_final_objective']):.6e}, "
                f"cpu={float(comparison['cpu_final_objective']):.6e}"
            ),
        ),
        (
            not bool(comparison["jax_field_error_not_worse_than_cpu"]),
            lambda: (
                "Final field error is worse than the CPU reference beyond tolerance: "
                f"jax={float(comparison['jax_field_error']):.6e}, "
                f"cpu={float(comparison['cpu_field_error']):.6e}"
            ),
        ),
        (
            not bool(comparison["jax_curve_length_within_target"]),
            lambda: (
                "Final banana-coil length violates the configured target: "
                f"{float(comparison['jax_final_curve_length']):.6e} > "
                f"{float(comparison['length_target']):.6e}"
            ),
        ),
        (
            not bool(comparison["jax_cc_distance_within_threshold"]),
            lambda: (
                "Final banana-coil distance violates the configured threshold: "
                f"{float(comparison['jax_final_cc_distance']):.6e} < "
                f"{float(comparison['cc_threshold']):.6e}"
            ),
        ),
        (
            not bool(comparison["jax_curvature_not_worse_than_cpu"]),
            lambda: (
                "Final banana-coil curvature is worse than the CPU reference envelope: "
                f"jax={float(comparison['jax_max_curvature']):.6e}, "
                f"cpu={float(comparison['cpu_max_curvature']):.6e}, "
                f"threshold={float(comparison['curvature_threshold']):.6e}"
            ),
        ),
        (
            bool(comparison["jax_self_intersecting"]),
            lambda: "Final banana coil is self-intersecting on the ondevice lane.",
        ),
    ]
    for should_fail, message_factory in checks:
        if should_fail:
            failures.append(message_factory())


def evaluate_stage2_e2e_comparison(comparison: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    optimizer_backend = comparison.get("optimizer_backend", "scipy")
    if optimizer_backend == "ondevice":
        _append_stage2_ondevice_failures(failures, comparison)
    else:
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
    jax_primary_elapsed_s, jax_timings = _build_jax_stage2_timings(jax_case)

    max_geom_abs, max_geom_rel = _max_geometry_deviation(cpu_results, jax_results)
    ondevice_metrics = _build_ondevice_stage2_metrics(cpu_results, jax_results)
    final_objective_rel_diff = relative_error(
        ondevice_metrics["jax_final_objective"],
        ondevice_metrics["cpu_final_objective"],
    )

    comparison = {
        "optimizer_backend": str(jax_results.get("optimizer_backend", "scipy")),
        "final_objective_rel_diff": final_objective_rel_diff,
        "field_error_rel_diff": relative_error(
            ondevice_metrics["jax_field_error"],
            ondevice_metrics["cpu_field_error"],
        ),
        "field_error_rel_tol": FIELD_ERROR_REL_TOL,
        "max_geometry_pointwise_abs": max_geom_abs,
        "max_geometry_pointwise_rel": max_geom_rel,
        "geometry_rel_tol": geometry_rel_tol,
        "cpu_iterations": int(cpu_results["iterations"]),
        "jax_iterations": int(jax_results["iterations"]),
        "cpu_elapsed_s": float(cpu_case["elapsed_s"]),
        "jax_elapsed_s": jax_primary_elapsed_s,
        "cpu_trajectory_len": len(cpu_trajectory),
        "jax_trajectory_len": len(jax_trajectory),
        "cpu_trajectory_finite": _trajectory_is_finite(cpu_trajectory),
        "jax_trajectory_finite": _trajectory_is_finite(jax_trajectory),
        "cpu_trajectory_improves": _trajectory_improves(cpu_trajectory),
        "jax_trajectory_improves": _trajectory_improves(jax_trajectory),
        **ondevice_metrics,
    }
    failures = evaluate_stage2_e2e_comparison(comparison)
    timings = {
        "cpu_outer_elapsed_s": float(cpu_case["elapsed_s"]),
        **jax_timings,
    }
    return {
        "provenance": provenance,
        "cpu_results": cpu_results,
        "jax_results": jax_results,
        "cpu_trajectory": cpu_trajectory,
        "jax_trajectory": jax_trajectory,
        "timings": timings,
        "comparison": comparison,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    args = parse_args()
    geometry_rel_tol = (
        float(args.geometry_rel_tol)
        if args.geometry_rel_tol is not None
        else optimizer_drift_tolerances(
            "tier2_stage2_e2e",
            maxiter=args.maxiter,
        )["geometry_rel_tol"]
    )
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Stage 2 end-to-end comparison",
        extra={
            "lane": resolve_probe_lane(optimizer_backend=args.optimizer_backend),
            "fixture": "real-stage2",
            "platform_request": args.platform,
            "optimizer_backend": args.optimizer_backend,
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "maxiter": int(args.maxiter),
            "geometry_rel_tol": geometry_rel_tol,
            "compile_behavior": describe_compile_behavior(uses_subprocesses=True),
            "optimizer_drift_tolerances": optimizer_drift_tolerances(
                "tier2_stage2_e2e",
                maxiter=args.maxiter,
            ),
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
    if "jax_optimizer_warm_run_s" in payload["timings"]:
        print(
            "JAX ondevice optimizer timings: "
            f"cold={payload['timings']['jax_optimizer_cold_run_s']:.2f}s, "
            f"warm={payload['timings']['jax_optimizer_warm_run_s']:.2f}s, "
            "compile_overhead~="
            f"{payload['timings']['jax_optimizer_compile_overhead_s']:.2f}s"
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
