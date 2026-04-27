"""Build a lane-aware single-stage parity matrix from run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (  # noqa: E402
    load_json,
    optimizer_drift_tolerances,
    parity_ladder_tolerances,
    relative_error,
    write_json,
)


LANE_CPU_SCIPY = "cpu_scipy"
LANE_JAX_CPU = "jax_cpu"
LANE_H100_GPU = "h100_gpu"
SAME_STATE_INITIAL_PREFIX = "INITIAL_"
DERIVED_BOOZER_METRICS = frozenset(
    {
        "INITIAL_IOTA",
        "FINAL_IOTA",
        "INITIAL_FIELD_ERROR",
        "FIELD_ERROR",
    }
)
METRIC_COMPARISON_KEYS = (
    "INITIAL_VOLUME",
    "INITIAL_IOTA",
    "INITIAL_FIELD_ERROR",
    "INITIAL_MAX_CURVATURE",
    "FINAL_VOLUME",
    "FINAL_IOTA",
    "FIELD_ERROR",
    "MAX_CURVATURE",
    "CURVE_CURVE_MIN_DIST",
    "CURVE_SURFACE_MIN_DIST",
    "SURFACE_VESSEL_MIN_DIST",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify C++ CPU, JAX CPU, and GPU single-stage parity from "
            "merged parity-report and optional optimizer-progress artifacts."
        )
    )
    parser.add_argument(
        "--parity-report-json",
        required=True,
        help="Merged single-stage parity report JSON.",
    )
    parser.add_argument(
        "--cpu-progress-json",
        default=None,
        help="Optional CPU outer_optimizer_progress.json with optimizer-state trace.",
    )
    parser.add_argument(
        "--jax-cpu-progress-json",
        default=None,
        help="Optional JAX CPU outer_optimizer_progress.json with optimizer-state trace.",
    )
    parser.add_argument(
        "--gpu-progress-json",
        default=None,
        help="Optional GPU outer_optimizer_progress.json with optimizer-state trace.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the parity matrix JSON.",
    )
    return parser.parse_args()


def _status(passed: bool) -> str:
    return "pass" if passed else "drift"


def _aggregate_status(statuses: list[str] | set[str]) -> str:
    if "blocked" in statuses:
        return "blocked"
    if "drift" in statuses:
        return "drift"
    return "pass"


def _metric_tolerances(metric_name: str) -> tuple[float, float]:
    tolerances = parity_ladder_tolerances("branch-stable-resolve")
    if metric_name in DERIVED_BOOZER_METRICS:
        return (
            float(tolerances["derived_value_rtol"]),
            float(tolerances["derived_value_atol"]),
        )
    return (
        float(tolerances["core_value_rtol"]),
        float(tolerances["core_value_atol"]),
    )


def _allclose_delta(lhs: float, rhs: float, *, rtol: float, atol: float) -> dict[str, Any]:
    abs_delta = float(abs(lhs - rhs))
    rel_delta = float(relative_error(lhs, rhs))
    passed = bool(np.isclose(lhs, rhs, rtol=rtol, atol=atol))
    return {
        "status": _status(passed),
        "lhs": float(lhs),
        "rhs": float(rhs),
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
        "rtol": float(rtol),
        "atol": float(atol),
    }


def _vectors_close(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> bool:
    return bool(
        lhs.shape == rhs.shape
        and np.allclose(lhs, rhs, rtol=rtol, atol=atol)
    )


def _merged_metric_comparisons(
    metrics: dict[str, Any],
    *,
    lhs_lane: str,
    rhs_lane: str,
    initial_only: bool,
) -> dict[str, Any]:
    comparisons = {}
    for metric_name in METRIC_COMPARISON_KEYS:
        if initial_only and not metric_name.startswith(SAME_STATE_INITIAL_PREFIX):
            continue
        metric = metrics.get(metric_name)
        if metric is None:
            continue
        values = metric["values"]
        if lhs_lane not in values or rhs_lane not in values:
            continue
        rtol, atol = _metric_tolerances(metric_name)
        comparisons[metric_name] = _allclose_delta(
            float(values[lhs_lane]),
            float(values[rhs_lane]),
            rtol=rtol,
            atol=atol,
        )
    return comparisons


def _comparison_summary(comparisons: dict[str, Any]) -> dict[str, Any]:
    if not comparisons:
        return {
            "status": "blocked",
            "reason": "no comparable metrics found",
            "drifted_metrics": [],
            "metric_count": 0,
            "metrics": {},
        }
    drifted = [
        metric_name
        for metric_name, comparison in comparisons.items()
        if comparison["status"] != "pass"
    ]
    return {
        "status": "pass" if not drifted else "drift",
        "drifted_metrics": drifted,
        "metric_count": int(len(comparisons)),
        "metrics": comparisons,
    }


def _termination_values(metrics: dict[str, Any]) -> dict[str, str]:
    termination = metrics.get("TERMINATION_MESSAGE", {})
    return dict(termination.get("values", {}))


def _same_state_value_grad_summary(report: dict[str, Any]) -> dict[str, Any]:
    value_grad = report.get("jax_cpu_vs_h100_value_grad")
    if value_grad is None:
        return {
            "status": "blocked",
            "reason": "jax_cpu_vs_h100_value_grad artifact is missing",
        }
    tolerances = parity_ladder_tolerances("gpu-runtime")
    objective_rtol = float(tolerances["same_state_forward_rtol"])
    objective_atol = float(tolerances["same_state_forward_atol"])
    gradient_rtol = float(tolerances["same_state_gradient_rtol"])
    gradient_atol = float(tolerances["same_state_gradient_atol"])
    objective_abs = abs(float(value_grad["objective_abs_delta"]))
    objective_rel = abs(float(value_grad["objective_rel_delta"]))
    grad_abs = abs(float(value_grad["grad_max_abs_delta"]))
    objective_passed = (
        objective_abs <= objective_atol
        or objective_rel <= objective_rtol
    )
    grad_passed = bool(value_grad["grad_allclose_rtol_1e-10_atol_1e-12"]) and (
        grad_abs <= gradient_atol
        or grad_abs
        <= gradient_rtol * max(abs(float(value_grad["jax_cpu_grad_inf_norm"])), 1.0)
    )
    return {
        "status": _status(bool(objective_passed and grad_passed)),
        "objective_abs_delta": objective_abs,
        "objective_rel_delta": objective_rel,
        "grad_max_abs_delta": grad_abs,
        "tolerances": {
            "objective_rtol": objective_rtol,
            "objective_atol": objective_atol,
            "gradient_rtol": gradient_rtol,
            "gradient_atol": gradient_atol,
        },
    }


def _load_optimizer_state_trace(progress_json: str | None) -> list[dict[str, Any]]:
    if progress_json is None:
        return []
    payload = load_json(progress_json)
    events = payload["events"]
    for event in reversed(events):
        result = event.get("result")
        if not result:
            continue
        trace = result.get("optimizer_state_trace", [])
        if trace:
            return list(trace)
    return []


def _summary_vector_values(entry: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(entry[key]["values"], dtype=np.float64)


def _summary_scalar_value(entry: dict[str, Any], key: str) -> float:
    return float(entry[key]["value"])


def _compare_optimizer_state_trace_pair(
    lhs_trace: list[dict[str, Any]],
    rhs_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    if not lhs_trace or not rhs_trace:
        return {
            "status": "blocked",
            "reason": "matched optimizer_state_trace entries are missing",
        }
    lhs = lhs_trace[0]
    rhs = rhs_trace[0]
    tolerances = optimizer_drift_tolerances("optimizer_state_parity")
    lhs_trial_x = _summary_vector_values(lhs, "trial_x")
    rhs_trial_x = _summary_vector_values(rhs, "trial_x")
    lhs_direction = _summary_vector_values(lhs, "search_direction")
    rhs_direction = _summary_vector_values(rhs, "search_direction")
    lhs_trial_jac = _summary_vector_values(lhs, "trial_jac")
    rhs_trial_jac = _summary_vector_values(rhs, "trial_jac")
    x_rtol = float(tolerances["x_rtol"])
    x_atol = float(tolerances["x_atol"])
    gradient_rtol = float(tolerances["gradient_rtol"])
    gradient_atol = float(tolerances["gradient_atol"])
    x_close = _vectors_close(
        lhs_trial_x,
        rhs_trial_x,
        rtol=x_rtol,
        atol=x_atol,
    )
    search_direction_close = _vectors_close(
        lhs_direction,
        rhs_direction,
        rtol=x_rtol,
        atol=x_atol,
    )
    grad_close = _vectors_close(
        lhs_trial_jac,
        rhs_trial_jac,
        rtol=gradient_rtol,
        atol=gradient_atol,
    )
    objective = _allclose_delta(
        _summary_scalar_value(lhs, "trial_fun"),
        _summary_scalar_value(rhs, "trial_fun"),
        rtol=float(tolerances["objective_rel_tol"]),
        atol=0.0,
    )
    step_scales = [
        _summary_scalar_value(lhs, "step_scale"),
        _summary_scalar_value(rhs, "step_scale"),
    ]
    step_scale_close = bool(
        np.isclose(
            step_scales[0],
            step_scales[1],
            rtol=x_rtol,
            atol=x_atol,
        )
    )
    jac_inf_delta = abs(
        _summary_scalar_value(lhs, "trial_jac_inf_norm")
        - _summary_scalar_value(rhs, "trial_jac_inf_norm")
    )
    jac_inf_close = jac_inf_delta <= float(tolerances["jac_norm_inf_abs_tol"])
    line_search_statuses = [
        int(lhs["line_search_status"]),
        int(rhs["line_search_status"]),
    ]
    line_search_status_close = line_search_statuses[0] == line_search_statuses[1]
    return {
        "status": _status(
            bool(
                x_close
                and search_direction_close
                and grad_close
                and objective["status"] == "pass"
                and step_scale_close
                and jac_inf_close
                and line_search_status_close
            )
        ),
        "x_close": x_close,
        "search_direction_close": search_direction_close,
        "gradient_close": grad_close,
        "objective": objective,
        "step_scale_close": step_scale_close,
        "jac_inf_norm_abs_delta": float(jac_inf_delta),
        "line_search_status_close": line_search_status_close,
        "line_search_statuses": line_search_statuses,
        "step_scales": step_scales,
        "tolerances": tolerances,
    }


def build_single_stage_parity_matrix(
    report: dict[str, Any],
    *,
    cpu_progress_json: str | None = None,
    jax_cpu_progress_json: str | None = None,
    gpu_progress_json: str | None = None,
) -> dict[str, Any]:
    metrics = report["same_seed_no_optimizer_metrics"]
    terminations = _termination_values(metrics)
    cpu_jax_metrics = _comparison_summary(
        _merged_metric_comparisons(
            metrics,
            lhs_lane=LANE_JAX_CPU,
            rhs_lane=LANE_CPU_SCIPY,
            initial_only=False,
        )
    )
    jax_gpu_initial_metrics = _comparison_summary(
        _merged_metric_comparisons(
            metrics,
            lhs_lane=LANE_H100_GPU,
            rhs_lane=LANE_JAX_CPU,
            initial_only=True,
        )
    )
    cpu_trace = _load_optimizer_state_trace(cpu_progress_json)
    jax_cpu_trace = _load_optimizer_state_trace(jax_cpu_progress_json)
    gpu_trace = _load_optimizer_state_trace(gpu_progress_json)
    optimizer_trace_pairs = {
        "jax_cpu_vs_h100_gpu": _compare_optimizer_state_trace_pair(
            jax_cpu_trace,
            gpu_trace,
        ),
    }
    if cpu_progress_json is not None:
        optimizer_trace_pairs["cpu_scipy_vs_jax_cpu"] = (
            _compare_optimizer_state_trace_pair(
                cpu_trace,
                jax_cpu_trace,
            )
        )
    pair_statuses = [pair["status"] for pair in optimizer_trace_pairs.values()]
    optimizer_trace_status = _aggregate_status(pair_statuses)
    full_trajectory_status = _aggregate_status(pair_statuses)
    full_trajectory_reasons = []
    for pair_name, pair in optimizer_trace_pairs.items():
        if pair["status"] != "pass":
            full_trajectory_reasons.append(f"{pair_name}: {pair['status']}")
    target_lane_terminations = [
        terminations[lane]
        for lane in (LANE_JAX_CPU, LANE_H100_GPU)
        if lane in terminations
    ]
    if len(set(target_lane_terminations)) > 1:
        full_trajectory_status = "blocked"
        full_trajectory_reasons.append("target-lane termination modes differ")
    comparisons = {
        "jax_cpu_vs_h100_same_state_value_grad": _same_state_value_grad_summary(report),
        "cpu_scipy_vs_jax_cpu_same_seed_metrics": cpu_jax_metrics,
        "jax_cpu_vs_h100_initial_metrics": jax_gpu_initial_metrics,
        "optimizer_state_trace_pairs": {
            "status": optimizer_trace_status,
            "pairs": optimizer_trace_pairs,
        },
        "full_trajectory_parity": {
            "status": full_trajectory_status,
            "reasons": full_trajectory_reasons,
            "termination_messages": terminations,
        },
    }
    blocking = [
        name
        for name, comparison in comparisons.items()
        if comparison["status"] != "pass"
    ]
    return {
        "comparisons": comparisons,
        "blocking_comparisons": blocking,
        "passed": not blocking,
    }


def main() -> None:
    args = parse_args()
    report = load_json(args.parity_report_json)
    payload = build_single_stage_parity_matrix(
        report,
        cpu_progress_json=args.cpu_progress_json,
        jax_cpu_progress_json=args.jax_cpu_progress_json,
        gpu_progress_json=args.gpu_progress_json,
    )
    write_json(args.output_json, payload)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
