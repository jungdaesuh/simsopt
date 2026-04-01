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
_TIER1_BASE_TOLERANCES = optimizer_drift_tolerances("tier1_stage2_value_gradient")
MATCHED_OBJECTIVE_REL_TOL = _TIER1_BASE_TOLERANCES["objective_rel_tol"]
MATCHED_GRADIENT_RTOL = _TIER1_BASE_TOLERANCES["gradient_rtol"]
MATCHED_GRADIENT_ATOL = _TIER1_BASE_TOLERANCES["gradient_atol"]
STAGE2_MATCHED_GRADIENT_ATOL = max(MATCHED_GRADIENT_ATOL, 5e-12)
MATCHED_FIELD_REL_TOL = 1e-10
STAGE2_CURVATURE_BARRIER_EDGE_MARGIN = 1e-6
STAGE2_CURVATURE_BARRIER_EDGE_GRADIENT_RTOL = 1e-5
STAGE2_CURVATURE_BARRIER_EDGE_OBJECTIVE_REL_TOL = 5e-4
STAGE2_CURVATURE_BARRIER_TERM_NAME = "curvature_barrier"

_CPU_ONDEVICE_ENDPOINT_LANE = ("jax", "cpu", "cpu-ondevice")
_CPU_REFERENCE_ENDPOINT_LANE = ("cpu", "auto", "cpu-reference")


def _resolve_stage2_endpoint_cpu_lane(
    optimizer_backend: str,
) -> tuple[str, str, str]:
    """Return the CPU-side endpoint lane for the requested Stage 2 comparison."""
    if optimizer_backend == "ondevice":
        return _CPU_ONDEVICE_ENDPOINT_LANE
    return _CPU_REFERENCE_ENDPOINT_LANE


def _cpu_endpoint_lane_label(cpu_lane_kind: str) -> str:
    if cpu_lane_kind == _CPU_ONDEVICE_ENDPOINT_LANE[2]:
        return "CPU ondevice lane"
    return "CPU reference lane"


def _objective_not_worse(
    jax_value: float,
    cpu_value: float,
    *,
    rel_tol: float = FINAL_OBJECTIVE_REL_TOL,
) -> bool:
    return float(jax_value) <= float(cpu_value) * (1.0 + float(rel_tol))


def _curvature_margin(curvature: float, threshold: float) -> float:
    return float(threshold) - float(curvature)


def _curvature_within_threshold(curvature: float, threshold: float) -> bool:
    return float(curvature) <= float(threshold)


def _curvature_barrier_edge_active(
    curvature: float,
    threshold: float,
) -> bool:
    margin = _curvature_margin(curvature, threshold)
    return 0.0 <= margin <= STAGE2_CURVATURE_BARRIER_EDGE_MARGIN


def _stage2_final_objective_rel_tol(
    *,
    base_rel_tol: float,
    cpu_max_curvature: float,
    jax_max_curvature: float,
    curvature_threshold: float,
) -> float:
    if _curvature_barrier_edge_active(
        cpu_max_curvature,
        curvature_threshold,
    ) and _curvature_barrier_edge_active(
        jax_max_curvature,
        curvature_threshold,
    ):
        return max(base_rel_tol, STAGE2_CURVATURE_BARRIER_EDGE_OBJECTIVE_REL_TOL)
    return base_rel_tol


def _field_error_not_worse(jax_value: float, cpu_value: float) -> bool:
    return float(jax_value) <= float(cpu_value) * (1.0 + FIELD_ERROR_REL_TOL)


def _build_ondevice_stage2_metrics(
    cpu_results: dict[str, Any],
    jax_results: dict[str, Any],
    *,
    final_objective_rel_tol: float,
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
    final_objective_rel_tol = _stage2_final_objective_rel_tol(
        base_rel_tol=final_objective_rel_tol,
        cpu_max_curvature=cpu_max_curvature,
        jax_max_curvature=jax_max_curvature,
        curvature_threshold=curvature_threshold,
    )
    return {
        "cpu_final_objective": cpu_final_objective,
        "jax_final_objective": jax_final_objective,
        "jax_objective_not_worse_than_cpu": _objective_not_worse(
            jax_final_objective,
            cpu_final_objective,
            rel_tol=final_objective_rel_tol,
        ),
        "final_objective_rel_tol": final_objective_rel_tol,
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
        "cpu_curvature_margin": _curvature_margin(
            cpu_max_curvature,
            curvature_threshold,
        ),
        "jax_curvature_margin": _curvature_margin(
            jax_max_curvature,
            curvature_threshold,
        ),
        "cpu_curvature_barrier_edge_active": _curvature_barrier_edge_active(
            cpu_max_curvature,
            curvature_threshold,
        ),
        "jax_curvature_barrier_edge_active": _curvature_barrier_edge_active(
            jax_max_curvature,
            curvature_threshold,
        ),
        "jax_curvature_not_worse_than_cpu": _curvature_within_threshold(
            jax_max_curvature,
            max(cpu_max_curvature, curvature_threshold),
        ),
        "jax_self_intersecting": bool(jax_results["SELF_INTERSECTING"]),
    }


def _build_jax_stage2_timings(
    jax_case: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    jax_outer_elapsed_s = float(jax_case["elapsed_s"])
    jax_primary_elapsed_s = jax_outer_elapsed_s
    optimizer_timings = jax_case.get("optimizer_timings")
    if optimizer_timings is not None and "cold_run_s" in optimizer_timings:
        jax_primary_elapsed_s = float(optimizer_timings["cold_run_s"])
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
    parser.add_argument(
        "--nphi", type=int, default=255, help="Surface toroidal grid points."
    )
    parser.add_argument(
        "--ntheta", type=int, default=64, help="Surface poloidal grid points."
    )
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
    effective_platform = platform if backend == "jax" else "cpu"
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
        run_python_script(
            script_path,
            command,
            env=repo_pythonpath_env(
                platform=effective_platform,
                disable_compilation_cache=(effective_platform == "cpu"),
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


def _run_stage2_probe(
    args: argparse.Namespace,
    backend: str,
    *,
    platform: str,
    dofs: list[float],
) -> dict[str, Any]:
    script_path = _stage2_script_path()
    effective_platform = platform if backend == "jax" else "cpu"
    with tempfile.TemporaryDirectory(prefix=f"stage2-probe-{backend}-") as temp_dir:
        export_json = Path(temp_dir) / f"{backend}_probe.json"
        dofs_json = Path(temp_dir) / "override_dofs.json"
        write_json(dofs_json, list(dofs))
        command = [
            "--backend",
            backend,
            "--probe-only",
            "--skip-postprocess",
            "--export-objective-json",
            str(export_json),
            "--override-dofs-json",
            str(dofs_json),
            "--nphi",
            str(args.nphi),
            "--ntheta",
            str(args.ntheta),
            "--optimizer-backend",
            args.optimizer_backend,
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
        run_python_script(
            script_path,
            command,
            env=repo_pythonpath_env(
                platform=effective_platform,
                disable_compilation_cache=(effective_platform == "cpu"),
            ),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        return load_json(export_json)


def _run_stage2_matched_state_probes(
    args: argparse.Namespace,
    *,
    cpu_backend: str,
    cpu_platform: str,
    jax_platform: str,
    dofs: list[float],
) -> dict[str, dict[str, Any]]:
    return {
        "cpu": _run_stage2_probe(
            args,
            cpu_backend,
            platform=cpu_platform,
            dofs=dofs,
        ),
        "jax": _run_stage2_probe(
            args,
            "jax",
            platform=jax_platform,
            dofs=dofs,
        ),
    }


def _trajectory_is_finite(trajectory: list[dict]) -> bool:
    for entry in trajectory:
        barrier_rejection = bool(entry.get("distance_constraint_violated", False))
        objective_value = float(entry["J"])
        if barrier_rejection and np.isposinf(objective_value):
            values = (
                entry["Jf"],
                entry["mean_abs_relBfinal_norm"],
                entry["curve_length"],
                entry["coil_coil_distance"],
                entry["curvature"],
            )
        else:
            values = (
                objective_value,
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


def _max_geometry_deviation(
    cpu_results: dict, jax_results: dict
) -> tuple[float, float]:
    cpu_gamma = np.asarray(cpu_results["FINAL_BANANA_GAMMA"], dtype=float)
    jax_gamma = np.asarray(jax_results["FINAL_BANANA_GAMMA"], dtype=float)
    return max_pointwise_geometry_drift(jax_gamma, cpu_gamma)


def _build_gradient_term_metrics(
    cpu_terms: dict[str, Any],
    jax_terms: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    metrics: dict[str, Any] = {}
    worst_term: dict[str, Any] | None = None
    for name in cpu_terms:
        if name not in jax_terms:
            continue
        term_metric = {
            "objective_rel_diff": relative_error(
                float(jax_terms[name]["J"]),
                float(cpu_terms[name]["J"]),
            ),
            **_build_gradient_parity_metrics(
                np.asarray(cpu_terms[name]["dJ"], dtype=float),
                np.asarray(jax_terms[name]["dJ"], dtype=float),
            ),
        }
        metrics[name] = term_metric
        if worst_term is None or float(term_metric["gradient_l2_rel_diff"]) > float(
            worst_term["gradient_l2_rel_diff"]
        ):
            worst_term = {
                "name": name,
                **term_metric,
            }
    return metrics, worst_term


def _build_gradient_parity_metrics(
    cpu_grad: np.ndarray,
    jax_grad: np.ndarray,
) -> dict[str, Any]:
    gradient_l2_rel_diff = float(
        np.linalg.norm(jax_grad - cpu_grad) / (np.linalg.norm(cpu_grad) + 1e-30)
    )
    gradient_max_abs_diff = float(np.max(np.abs(jax_grad - cpu_grad)))
    gradient_componentwise_allclose = bool(
        np.allclose(
            jax_grad,
            cpu_grad,
            rtol=MATCHED_GRADIENT_RTOL,
            atol=STAGE2_MATCHED_GRADIENT_ATOL,
        )
    )
    gradient_scaled_atol = max(
        STAGE2_MATCHED_GRADIENT_ATOL,
        MATCHED_GRADIENT_RTOL * max(1.0, float(np.max(np.abs(cpu_grad)))),
    )
    gradient_global_scale_match = bool(
        gradient_l2_rel_diff <= MATCHED_GRADIENT_RTOL
        and gradient_max_abs_diff <= gradient_scaled_atol
    )
    return {
        "gradient_allclose": bool(
            gradient_componentwise_allclose or gradient_global_scale_match
        ),
        "gradient_componentwise_allclose": gradient_componentwise_allclose,
        "gradient_global_scale_match": gradient_global_scale_match,
        "gradient_l2_rel_diff": gradient_l2_rel_diff,
        "gradient_max_abs_diff": gradient_max_abs_diff,
        "gradient_scaled_atol": float(gradient_scaled_atol),
    }


def _build_matched_state_metrics(
    cpu_probe: dict[str, Any],
    jax_probe: dict[str, Any],
) -> dict[str, Any]:
    cpu_composite = cpu_probe["composite"]
    jax_composite = jax_probe["composite"]
    curvature_threshold = float(cpu_probe["curvature_threshold"])
    jax_curvature = float(jax_composite["curvature"])
    gradient_metrics = _build_gradient_parity_metrics(
        np.asarray(cpu_composite["dJ"], dtype=float),
        np.asarray(jax_composite["dJ"], dtype=float),
    )
    gradient_terms: dict[str, Any] = {}
    worst_gradient_term = None
    cpu_terms = cpu_composite.get("terms")
    jax_terms = jax_composite.get("terms")
    if isinstance(cpu_terms, dict) and isinstance(jax_terms, dict):
        gradient_terms, worst_gradient_term = _build_gradient_term_metrics(
            cpu_terms,
            jax_terms,
        )
    return {
        "objective_rel_diff": relative_error(
            float(jax_composite["J"]),
            float(cpu_composite["J"]),
        ),
        "field_error_rel_diff": relative_error(
            float(jax_composite["mean_abs_relBfinal_norm"]),
            float(cpu_composite["mean_abs_relBfinal_norm"]),
        ),
        "curvature": jax_curvature,
        "curvature_threshold": curvature_threshold,
        "curvature_margin": _curvature_margin(
            jax_curvature,
            curvature_threshold,
        ),
        "curvature_barrier_edge_active": _curvature_barrier_edge_active(
            jax_curvature,
            curvature_threshold,
        ),
        **gradient_metrics,
        "gradient_terms": gradient_terms,
        "worst_gradient_term": worst_gradient_term,
    }


def _matched_gradient_barrier_edge_portable(state: dict[str, Any]) -> bool:
    if not bool(state.get("curvature_barrier_edge_active", False)):
        return False
    worst_term = state.get("worst_gradient_term")
    if not isinstance(worst_term, dict):
        return False
    if str(worst_term.get("name")) != STAGE2_CURVATURE_BARRIER_TERM_NAME:
        return False
    return float(worst_term["gradient_l2_rel_diff"]) <= (
        STAGE2_CURVATURE_BARRIER_EDGE_GRADIENT_RTOL
    )


def _matched_gradient_failure_message(
    state: dict[str, Any],
    *,
    state_label: str,
) -> str:
    worst_term = state.get("worst_gradient_term")
    margin_suffix = ""
    if "curvature_margin" in state:
        margin_suffix = f", curvature_margin={float(state['curvature_margin']):.2e}"
    if isinstance(worst_term, dict):
        return (
            f"Matched {state_label}-final gradient parity failed gate "
            f"(worst term: {str(worst_term['name'])}, "
            f"rel_diff={float(worst_term['gradient_l2_rel_diff']):.2e}, "
            f"max_abs_diff={float(worst_term['gradient_max_abs_diff']):.2e}, "
            f"scaled_atol={float(worst_term['gradient_scaled_atol']):.2e}"
            f"{margin_suffix})."
        )
    return f"Matched {state_label}-final gradient parity failed gate."


def _append_geometry_gate_failure(
    failures: list[str],
    *,
    geometry_rel_diff: float,
    geometry_rel_tol: float | None,
) -> None:
    if geometry_rel_tol is None:
        return
    if float(geometry_rel_diff) < float(geometry_rel_tol):
        return
    failures.append(
        "Final banana-coil geometry drift too large: "
        f"{float(geometry_rel_diff):.2e} "
        f"relative (tol={float(geometry_rel_tol):.2e})"
    )


def _append_matched_state_failures(
    failures: list[str],
    *,
    cpu_state: dict[str, Any],
    jax_state: dict[str, Any],
) -> None:
    if float(cpu_state["objective_rel_diff"]) >= MATCHED_OBJECTIVE_REL_TOL:
        failures.append(
            "Matched CPU-final objective parity too large: "
            f"{float(cpu_state['objective_rel_diff']):.2e}"
        )
    if float(jax_state["objective_rel_diff"]) >= MATCHED_OBJECTIVE_REL_TOL:
        failures.append(
            "Matched JAX-final objective parity too large: "
            f"{float(jax_state['objective_rel_diff']):.2e}"
        )
    if float(cpu_state["field_error_rel_diff"]) >= MATCHED_FIELD_REL_TOL:
        failures.append(
            "Matched CPU-final field diagnostic parity too large: "
            f"{float(cpu_state['field_error_rel_diff']):.2e}"
        )
    if float(jax_state["field_error_rel_diff"]) >= MATCHED_FIELD_REL_TOL:
        failures.append(
            "Matched JAX-final field diagnostic parity too large: "
            f"{float(jax_state['field_error_rel_diff']):.2e}"
        )
    if not bool(cpu_state["gradient_allclose"]):
        if not _matched_gradient_barrier_edge_portable(cpu_state):
            failures.append(
                _matched_gradient_failure_message(
                    cpu_state,
                    state_label="CPU",
                )
            )
    if not bool(jax_state["gradient_allclose"]):
        if not _matched_gradient_barrier_edge_portable(jax_state):
            failures.append(
                _matched_gradient_failure_message(
                    jax_state,
                    state_label="JAX",
                )
            )


def _append_stage2_ondevice_failures(
    failures: list[str],
    comparison: dict[str, Any],
) -> None:
    cpu_lane_label = str(comparison.get("cpu_lane_label", "CPU lane"))
    checks = [
        (
            not bool(comparison["jax_objective_not_worse_than_cpu"]),
            lambda: (
                f"Final objective is worse than the {cpu_lane_label} beyond tolerance: "
                f"jax={float(comparison['jax_final_objective']):.6e}, "
                f"cpu={float(comparison['cpu_final_objective']):.6e}, "
                "rel_tol="
                f"{float(comparison['final_objective_rel_tol']):.2e}"
            ),
        ),
        (
            not bool(comparison["jax_field_error_not_worse_than_cpu"]),
            lambda: (
                f"Final field error is worse than the {cpu_lane_label} beyond tolerance: "
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
                f"Final banana-coil curvature is worse than the {cpu_lane_label} envelope: "
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
    cpu_state = comparison["matched_cpu_state"]
    jax_state = comparison["matched_jax_state"]
    _append_matched_state_failures(
        failures,
        cpu_state=cpu_state,
        jax_state=jax_state,
    )
    if optimizer_backend == "ondevice":
        _append_stage2_ondevice_failures(failures, comparison)
    _append_geometry_gate_failure(
        failures,
        geometry_rel_diff=float(comparison["max_geometry_pointwise_rel"]),
        geometry_rel_tol=comparison["geometry_rel_tol"],
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
    cpu_final_state_probes: dict[str, dict[str, Any]],
    jax_final_state_probes: dict[str, dict[str, Any]],
    *,
    cpu_lane_kind: str,
    final_objective_rel_tol: float,
    geometry_rel_tol: float | None,
) -> dict[str, Any]:
    cpu_results = cpu_case["results"]
    jax_results = jax_case["results"]
    cpu_trajectory = cpu_case["trajectory"]
    jax_trajectory = jax_case["trajectory"]
    jax_primary_elapsed_s, jax_timings = _build_jax_stage2_timings(jax_case)

    max_geom_abs, max_geom_rel = _max_geometry_deviation(cpu_results, jax_results)
    ondevice_metrics = _build_ondevice_stage2_metrics(
        cpu_results,
        jax_results,
        final_objective_rel_tol=final_objective_rel_tol,
    )
    final_objective_rel_diff = relative_error(
        ondevice_metrics["jax_final_objective"],
        ondevice_metrics["cpu_final_objective"],
    )

    cpu_elapsed_s = float(cpu_case["elapsed_s"])
    cpu_iterations = int(cpu_results["iterations"])
    jax_iterations = int(jax_results["iterations"])
    jax_optimizer_backend = str(jax_results.get("optimizer_backend", "scipy"))
    comparison = {
        "optimizer_backend": jax_optimizer_backend,
        "cpu_lane_kind": cpu_lane_kind,
        "cpu_lane_label": _cpu_endpoint_lane_label(cpu_lane_kind),
        "final_objective_rel_diff": final_objective_rel_diff,
        "field_error_rel_diff": relative_error(
            ondevice_metrics["jax_field_error"],
            ondevice_metrics["cpu_field_error"],
        ),
        "field_error_rel_tol": FIELD_ERROR_REL_TOL,
        "max_geometry_pointwise_abs": max_geom_abs,
        "max_geometry_pointwise_rel": max_geom_rel,
        "geometry_rel_tol": geometry_rel_tol,
        "cpu_iterations": cpu_iterations,
        "jax_iterations": jax_iterations,
        "cpu_elapsed_s": cpu_elapsed_s,
        "jax_elapsed_s": jax_primary_elapsed_s,
        "cpu_trajectory_len": len(cpu_trajectory),
        "jax_trajectory_len": len(jax_trajectory),
        "cpu_trajectory_finite": _trajectory_is_finite(cpu_trajectory),
        "jax_trajectory_finite": _trajectory_is_finite(jax_trajectory),
        "cpu_trajectory_improves": _trajectory_improves(cpu_trajectory),
        "jax_trajectory_improves": _trajectory_improves(jax_trajectory),
        "matched_cpu_state": _build_matched_state_metrics(
            cpu_final_state_probes["cpu"],
            cpu_final_state_probes["jax"],
        ),
        "matched_jax_state": _build_matched_state_metrics(
            jax_final_state_probes["cpu"],
            jax_final_state_probes["jax"],
        ),
        **ondevice_metrics,
    }
    failures = evaluate_stage2_e2e_comparison(comparison)
    timings = {
        "cpu_outer_elapsed_s": cpu_elapsed_s,
        **jax_timings,
    }
    status = "passed" if not failures else "failed"
    cpu_summary = {"elapsed_s": cpu_elapsed_s, "iterations": cpu_iterations}
    jax_summary = {
        "elapsed_s": jax_primary_elapsed_s,
        "iterations": jax_iterations,
        "optimizer_backend": jax_optimizer_backend,
    }
    return {
        "status": status,
        "provenance": provenance,
        "cpu": cpu_summary,
        "jax": jax_summary,
        "ondevice_metrics": ondevice_metrics,
        "cpu_results": cpu_results,
        "jax_results": jax_results,
        "cpu_trajectory": cpu_trajectory,
        "jax_trajectory": jax_trajectory,
        "matched_state_probes": {
            "cpu_final_state": cpu_final_state_probes,
            "jax_final_state": jax_final_state_probes,
        },
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
    final_objective_rel_tol = float(
        optimizer_drift_tolerances(
            "tier2_stage2_e2e",
            maxiter=args.maxiter,
        )["final_objective_rel_tol"]
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
            "final_objective_rel_tol": final_objective_rel_tol,
            "compile_behavior": describe_compile_behavior(uses_subprocesses=True),
            "optimizer_drift_tolerances": optimizer_drift_tolerances(
                "tier2_stage2_e2e",
                maxiter=args.maxiter,
            ),
        },
    )
    cpu_backend, cpu_platform, cpu_lane_kind = _resolve_stage2_endpoint_cpu_lane(
        args.optimizer_backend
    )
    provenance["cpu_endpoint_lane"] = {
        "backend": cpu_backend,
        "platform": cpu_platform,
        "kind": cpu_lane_kind,
    }
    print_provenance(provenance)
    cpu_case = _run_stage2_case(args, cpu_backend, platform=cpu_platform)
    jax_case = _run_stage2_case(args, "jax", platform=args.platform)
    cpu_final_dofs = cpu_case["results"]["FINAL_DOFS"]
    jax_final_dofs = jax_case["results"]["FINAL_DOFS"]
    cpu_final_state_probes = _run_stage2_matched_state_probes(
        args,
        cpu_backend=cpu_backend,
        cpu_platform=cpu_platform,
        jax_platform=args.platform,
        dofs=cpu_final_dofs,
    )
    jax_final_state_probes = _run_stage2_matched_state_probes(
        args,
        cpu_backend=cpu_backend,
        cpu_platform=cpu_platform,
        jax_platform=args.platform,
        dofs=jax_final_dofs,
    )

    payload = build_stage2_e2e_payload(
        provenance,
        cpu_case,
        jax_case,
        cpu_final_state_probes,
        jax_final_state_probes,
        cpu_lane_kind=cpu_lane_kind,
        final_objective_rel_tol=final_objective_rel_tol,
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
    matched_jax_worst_term = comparison["matched_jax_state"].get("worst_gradient_term")
    if isinstance(matched_jax_worst_term, dict):
        print(
            "Matched JAX-final worst gradient term: "
            f"{matched_jax_worst_term['name']} "
            f"(rel_diff={float(matched_jax_worst_term['gradient_l2_rel_diff']):.2e}, "
            "max_abs_diff="
            f"{float(matched_jax_worst_term['gradient_max_abs_diff']):.2e})"
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
