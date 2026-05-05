"""Build a lane-aware single-stage parity matrix from run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (  # noqa: E402
    evaluate_grouped_adjoint_memory_budget,
    evaluate_tier5_performance_budget,
    grouped_adjoint_memory_budget,
    load_json,
    optimizer_drift_tolerances,
    parity_ladder_tolerances,
    relative_error,
    tier5_performance_budget,
    write_json,
)


LANE_CPP_CPU = "cpp_cpu"
LANE_CPU_SCIPY = "cpu_scipy"
LANE_CPU_CPP_TRACE = "cpu_cpp_trace"
LANE_JAX_CPU = "jax_cpu"
LANE_JAX_GPU = "jax_gpu"
LANE_H100_GPU = "h100_gpu"
SAME_STATE_INITIAL_PREFIX = "INITIAL_"
RELEASE_GATE_BUCKETS = (
    "fixed_state_physics_parity",
    "coordinate_mapping_parity",
    "full_run_artifact_contract",
    "optimizer_public_behavior_parity",
    "final_metric_envelope",
    "performance_memory_report",
)
FIRST_DIVERGENCE_STAGES = frozenset(
    {
        "fixed_state_physics",
        "coordinate_mapping",
        "run_contract",
        "initial_gradient",
        "line_search",
        "termination",
        "final_sync",
    }
)
REQUIRED_FIXED_STATE_LANES = (LANE_CPP_CPU, LANE_JAX_CPU, LANE_JAX_GPU)
REQUIRED_FIXED_STATE_COMPARISONS = (
    "cpp_cpu_vs_jax_cpu",
    "cpp_cpu_vs_jax_gpu",
    "jax_cpu_vs_jax_gpu",
)
REQUIRED_FIXED_STATE_HASH_KEYS = (
    "equilibrium_hash",
    "runtime_seed_spec_hash",
    "biot_savart_json_hash",
    "objective_configuration_hash",
    "active_dof_mask_hash",
    "fixed_dof_mask_hash",
    "frozen_dof_mask_hash",
)
FIXED_STATE_HASH_EQUALITY_KEYS = (
    "equilibrium_hash",
    "runtime_seed_spec_hash",
    "biot_savart_json_hash",
    "objective_configuration_hash",
    "active_dof_mask_hash",
)
FULL_RUN_CONTRACT_LANES = (LANE_CPU_SCIPY, LANE_JAX_CPU, LANE_JAX_GPU)
FULL_RUN_CONTRACT_EQUALITY_KEYS = (
    "runtime_seed_spec_hash",
    "objective_configuration_hash",
    "run_family_id",
)
FULL_RUN_OBJECTIVE_CONFIG_KEYS = (
    "CONSTRAINT_METHOD",
    "CONSTRAINT_WEIGHT",
    "ALM_FORMULATION",
    "TARGET_VOLUME",
    "TARGET_IOTA",
    "NON_QS_WEIGHT",
    "RES_WEIGHT",
    "IOTAS_WEIGHT",
    "LENGTH_WEIGHT",
    "LENGTH_TARGET",
    "CC_DIST",
    "CC_WEIGHT",
    "CS_DIST",
    "CS_WEIGHT",
    "SS_DIST",
    "SURF_DIST_WEIGHT",
    "CURVATURE_THRESHOLD",
    "CURVATURE_WEIGHT",
    "BANANA_CURRENT_MAX_A",
    "STAGE2_TF_CURRENT_A",
    "STAGE2_TF_CURRENT_LIMIT_ENFORCED",
    "TF_CURRENT_LIMIT_A",
    "COIL_VESSEL_MIN_DIST_M",
)
REQUIRED_ASSEMBLED_LANE_OUTPUT_KEYS = (
    "total_objective",
    "objective_components",
    "full_optimizer_basis_gradient",
    "gradient_inf_norm",
    "gradient_l2_norm",
    "field_error",
    "iota",
    "volume",
    "max_curvature",
    "coil_coil_min_distance",
    "coil_plasma_min_distance",
    "plasma_vessel_min_distance",
    "self_intersection",
    "hardware_constraints",
)
REQUIRED_OPERATOR_LANE_OUTPUT_KEYS = (
    "biot_savart_B",
    "surface_gamma",
    "integral_BdotN",
    "boozer_residual_vector",
    "boozer_residual_norm",
    "boozer_residual_max_norm",
    "first_derivative_kernel_samples",
    "boozer_residual_jacobian_metadata",
    "boozer_jvp",
    "boozer_vjp",
    "boozer_adjoint_solve",
)
COORDINATE_MAPPING_REQUIRED_SECTIONS = (
    "inputs",
    "mapping",
    "active_indices",
    "frozen_indices",
    "state_reconstruction",
    "gradient_projection",
    "finite_difference_checks",
)
CUDA_REQUIRED_PROVENANCE_KEYS = (
    "cuda_runtime_version",
    "cuda_driver_version",
    "nvcc_version",
    "matmul_precision",
)
PERFORMANCE_REQUIRED_PROVENANCE_KEYS = (
    "repo_sha",
    "jax",
    "jaxlib",
    "backend",
    "devices",
    "x64_enabled",
    "peak_rss_mb",
    "xla_flags",
    "compilation_cache_policy",
)
PERFORMANCE_REQUIRED_TIMING_KEYS = ("compile_time_s", "run_time_s")
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
        "--fixed-state-parity-json",
        default=None,
        help=(
            "Optional release-gate fixed-state cpp_cpu/jax_cpu/jax_gpu "
            "parity artifact."
        ),
    )
    parser.add_argument(
        "--coordinate-mapping-json",
        default=None,
        help=(
            "Optional release-gate JF.x to bs.x coordinate-mapping proof. "
            "Required whenever --fixed-state-parity-json is provided."
        ),
    )
    parser.add_argument(
        "--cpu-progress-json",
        default=None,
        help=(
            "Optional CPU/SciPy outer_optimizer_progress.json. If it contains "
            "target-style optimizer_state_trace entries, those are compared as "
            "diagnostic traces; otherwise it contributes termination data only."
        ),
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
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional path to write a human-readable release-gate Markdown report.",
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


def _comparison_status(value: dict[str, Any] | None) -> str:
    if not value:
        return "blocked"
    return str(value.get("status", "blocked"))


def _status_from_failures(
    *,
    blocked: list[str],
    drifted: list[str],
) -> str:
    if blocked:
        return "blocked"
    if drifted:
        return "drift"
    return "pass"


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    value_grad = report.get("jax_cpu_vs_jax_gpu_value_grad")
    if value_grad is None:
        value_grad = report.get("jax_cpu_vs_h100_value_grad")
    if value_grad is None:
        return {
            "status": "blocked",
            "reason": "jax_cpu_vs_jax_gpu_value_grad artifact is missing",
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


def _load_optimizer_result(progress_json: str | None) -> dict[str, Any] | None:
    if progress_json is None:
        return None
    payload = load_json(progress_json)
    events = payload["events"]
    for event in reversed(events):
        result = event.get("result")
        if not result:
            continue
        return dict(result)
    return None


def _load_optimizer_state_trace(progress_json: str | None) -> list[dict[str, Any]]:
    result = _load_optimizer_result(progress_json)
    if result is None:
        return []
    trace = result.get("optimizer_state_trace", [])
    if trace:
        return list(trace)
    return []


def _progress_termination_values(
    report_terminations: dict[str, str],
    *,
    cpu_progress_json: str | None,
    jax_cpu_progress_json: str | None,
    gpu_progress_json: str | None,
    gpu_lane: str = LANE_H100_GPU,
) -> dict[str, str]:
    terminations = dict(report_terminations)
    for lane, progress_json in (
        (LANE_CPU_SCIPY, cpu_progress_json),
        (LANE_JAX_CPU, jax_cpu_progress_json),
        (gpu_lane, gpu_progress_json),
    ):
        result = _load_optimizer_result(progress_json)
        if result is None:
            continue
        message = result.get("message")
        if message is not None:
            terminations[lane] = str(message)
    return terminations


def _summary_vector_values(entry: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(entry[key]["values"], dtype=np.float64)


def _summary_scalar_value(entry: dict[str, Any], key: str) -> float:
    return float(entry[key]["value"])


def _compare_optimizer_state_trace_entry(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    iteration_index: int,
) -> dict[str, Any]:
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
    iteration_values = [int(lhs["iteration"]), int(rhs["iteration"])]
    iteration_close = iteration_values[0] == iteration_values[1]
    return {
        "status": _status(
            bool(
                iteration_close
                and x_close
                and search_direction_close
                and grad_close
                and objective["status"] == "pass"
                and step_scale_close
                and jac_inf_close
                and line_search_status_close
            )
        ),
        "iteration_index": int(iteration_index),
        "iterations": iteration_values,
        "iteration_close": iteration_close,
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


def _compare_optimizer_state_trace_pair(
    lhs_trace: list[dict[str, Any]],
    rhs_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    if not lhs_trace or not rhs_trace:
        return {
            "status": "blocked",
            "reason": "matched optimizer_state_trace entries are missing",
        }
    if len(lhs_trace) != len(rhs_trace):
        return {
            "status": "drift",
            "reason": "optimizer_state_trace lengths differ",
            "lhs_entry_count": int(len(lhs_trace)),
            "rhs_entry_count": int(len(rhs_trace)),
        }
    entry_comparisons = [
        _compare_optimizer_state_trace_entry(
            lhs,
            rhs,
            iteration_index=index,
        )
        for index, (lhs, rhs) in enumerate(zip(lhs_trace, rhs_trace, strict=True))
    ]
    first_mismatch = next(
        (
            entry
            for entry in entry_comparisons
            if entry["status"] != "pass"
        ),
        None,
    )
    return {
        "status": "pass" if first_mismatch is None else "drift",
        "entry_count": int(len(entry_comparisons)),
        "first_mismatch": first_mismatch,
        "entries": entry_comparisons,
    }


def _legacy_h100_release_keys(report: dict[str, Any]) -> list[str]:
    legacy_keys = []
    if "jax_cpu_vs_h100_value_grad" in report:
        legacy_keys.append("jax_cpu_vs_h100_value_grad")
    metrics = report.get("same_seed_no_optimizer_metrics", {})
    for metric_name, metric in metrics.items():
        values = metric.get("values", {}) if isinstance(metric, dict) else {}
        if LANE_H100_GPU in values:
            legacy_keys.append(f"same_seed_no_optimizer_metrics.{metric_name}.h100_gpu")
    return legacy_keys


def _release_gate_same_state_value_grad_summary(report: dict[str, Any]) -> dict[str, Any]:
    legacy_keys = _legacy_h100_release_keys(report)
    if legacy_keys:
        return {
            "status": "blocked",
            "reason": "release-gate mode rejects legacy h100 lane keys",
            "legacy_keys": legacy_keys,
        }
    value_grad = report.get("jax_cpu_vs_jax_gpu_value_grad")
    if value_grad is None:
        return {
            "status": "blocked",
            "reason": "jax_cpu_vs_jax_gpu_value_grad artifact is missing",
        }
    return _same_state_value_grad_summary(report)


def _blocked_metric_summary(
    reason: str,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "blocked",
        "reason": reason,
        "drifted_metrics": [],
        "metric_count": 0,
        "metrics": {},
    }
    if contract is not None:
        payload["contract"] = contract
    return payload


def _full_run_contract_source(report: dict[str, Any]) -> dict[str, Any]:
    for key in ("full_run_artifact_contract", "full_run_contracts", "lane_contracts"):
        value = report.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _result_path_from_artifact_path(artifact_path: Any) -> Path | None:
    if not isinstance(artifact_path, str) or not artifact_path:
        return None
    path = Path(artifact_path)
    if path.name == "results.json":
        return path
    return path / "results.json"


def _objective_config_hash_from_results(result: dict[str, Any]) -> tuple[str | None, list[str]]:
    missing = [key for key in FULL_RUN_OBJECTIVE_CONFIG_KEYS if key not in result]
    if "objective_configuration_hash" in result:
        return str(result["objective_configuration_hash"]), missing
    if missing:
        return None, missing
    return (
        _json_hash({key: result[key] for key in FULL_RUN_OBJECTIVE_CONFIG_KEYS}),
        [],
    )


def _lane_runtime_seed_spec_hash(
    result: dict[str, Any],
    lane_dir: Path | None,
) -> str | None:
    if "runtime_seed_spec_hash" in result:
        return str(result["runtime_seed_spec_hash"])
    if lane_dir is None:
        return None
    return _file_sha256(lane_dir / "single_stage_jax_runtime_spec.json")


def _lane_run_family_id(
    result: dict[str, Any],
    contract_root: dict[str, Any],
) -> str | None:
    provenance = result.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    for value in (
        result.get("run_family_id"),
        result.get("RUN_FAMILY_ID"),
        provenance.get("run_family_id"),
        contract_root.get("run_family_id"),
    ):
        if value is not None:
            return str(value)
    return None


def _full_run_lane_contract(
    report: dict[str, Any],
    lane: str,
    *,
    progress_json: str | None,
) -> dict[str, Any]:
    contract_root = _full_run_contract_source(report)
    direct_lanes = contract_root.get("lanes", {})
    direct_contract = (
        dict(direct_lanes[lane])
        if isinstance(direct_lanes, dict) and isinstance(direct_lanes.get(lane), dict)
        else None
    )
    if direct_contract is not None:
        if "run_family_id" not in direct_contract and "run_family_id" in contract_root:
            direct_contract["run_family_id"] = contract_root["run_family_id"]
        if progress_json is not None:
            direct_contract["progress_json"] = progress_json
        return direct_contract

    lane_sources = report.get("lanes", {})
    artifact_path = lane_sources.get(lane) if isinstance(lane_sources, dict) else None
    result_path = _result_path_from_artifact_path(artifact_path)
    if result_path is None or not result_path.is_file():
        return {
            "artifact_path": artifact_path,
            "results_json": None if result_path is None else str(result_path),
            "progress_json": progress_json,
        }

    result = load_json(str(result_path))
    lane_dir = result_path.parent
    objective_hash, missing_objective_keys = _objective_config_hash_from_results(result)
    return {
        "artifact_path": artifact_path,
        "results_json": str(result_path),
        "progress_json": progress_json,
        "runtime_seed_spec_hash": _lane_runtime_seed_spec_hash(result, lane_dir),
        "objective_configuration_hash": objective_hash,
        "missing_objective_config_keys": missing_objective_keys,
        "run_family_id": _lane_run_family_id(result, contract_root),
        "init_only": result.get("init_only"),
        "generated_at_utc": (
            result.get("provenance", {}).get("generated_at_utc")
            if isinstance(result.get("provenance"), dict)
            else None
        ),
        "repo_sha": (
            result.get("provenance", {}).get("repo_sha")
            if isinstance(result.get("provenance"), dict)
            else None
        ),
    }


def _full_run_metric_lane_failures(
    metrics: dict[str, Any],
    *,
    required_lanes: tuple[str, ...],
) -> list[str]:
    failures = []
    required = set(required_lanes)
    for metric_name, metric in metrics.items():
        values = metric.get("values", {}) if isinstance(metric, dict) else {}
        if not isinstance(values, dict) or not values:
            continue
        lanes = set(values)
        if lanes != required:
            failures.append(
                f"{metric_name}: expected canonical lanes "
                f"{', '.join(required_lanes)}, got {', '.join(sorted(lanes))}"
            )
    return failures


def _full_run_artifact_contract_bucket(
    report: dict[str, Any],
    *,
    cpu_progress_json: str | None,
    jax_cpu_progress_json: str | None,
    gpu_progress_json: str | None,
) -> dict[str, Any]:
    required_lanes = FULL_RUN_CONTRACT_LANES
    progress_jsons = {
        LANE_CPU_SCIPY: cpu_progress_json,
        LANE_JAX_CPU: jax_cpu_progress_json,
        LANE_JAX_GPU: gpu_progress_json,
    }
    lane_contracts = {
        lane: _full_run_lane_contract(
            report,
            lane,
            progress_json=progress_jsons[lane],
        )
        for lane in required_lanes
    }
    failures = _full_run_metric_lane_failures(
        report.get("same_seed_no_optimizer_metrics", {}),
        required_lanes=required_lanes,
    )
    for lane, contract in lane_contracts.items():
        if contract.get("results_json") is None and not {
            "runtime_seed_spec_hash",
            "objective_configuration_hash",
            "run_family_id",
        }.issubset(contract):
            failures.append(f"{lane}: full-run lane contract is missing")
        progress_json = contract.get("progress_json")
        if progress_json is None:
            failures.append(f"{lane}: progress JSON is missing")
        elif not Path(str(progress_json)).is_file():
            failures.append(f"{lane}: progress JSON does not exist at {progress_json}")
        missing_objective_keys = contract.get("missing_objective_config_keys", [])
        if missing_objective_keys:
            failures.append(
                f"{lane}: missing objective config keys "
                f"{', '.join(str(key) for key in missing_objective_keys)}"
            )
    cpu_init_only = lane_contracts[LANE_CPU_SCIPY].get("init_only")
    if cpu_init_only is not False:
        failures.append("cpu_scipy: lane must be a full optimizer run, not init_only")
    checked_hashes: dict[str, dict[str, Any]] = {}
    for hash_name in FULL_RUN_CONTRACT_EQUALITY_KEYS:
        lane_values = {
            lane: lane_contracts[lane].get(hash_name) for lane in required_lanes
        }
        checked_hashes[hash_name] = lane_values
        missing = [lane for lane, value in lane_values.items() if value is None]
        if missing:
            failures.append(
                f"{hash_name}: missing value for {', '.join(missing)}"
            )
            continue
        if len(set(lane_values.values())) != 1:
            failures.append(f"{hash_name}: lane values differ")
    return {
        "status": "pass" if not failures else "blocked",
        "lanes": lane_contracts,
        "checked_hashes": checked_hashes,
        "failures": failures,
    }


def _fixed_state_comparison(
    comparisons: dict[str, Any],
    comparison_name: str,
) -> dict[str, Any]:
    comparison = comparisons.get(comparison_name)
    if comparison is None:
        comparison = comparisons.get(f"{comparison_name}_fixed_state")
    if comparison is None:
        return {
            "status": "blocked",
            "reason": f"required comparison {comparison_name!r} is missing",
        }
    return dict(comparison)


def _hash_values_for_lane(lane_payload: dict[str, Any], hash_name: str) -> Any:
    hashes = lane_payload.get("hashes", {})
    if isinstance(hashes, dict) and hash_name in hashes:
        return hashes[hash_name]
    return lane_payload.get(hash_name)


def _fixed_state_hash_equality(lanes: dict[str, Any]) -> dict[str, Any]:
    failures = []
    checked: dict[str, dict[str, Any]] = {}
    for hash_name in FIXED_STATE_HASH_EQUALITY_KEYS:
        lane_values = {
            lane: _hash_values_for_lane(dict(lanes.get(lane, {})), hash_name)
            for lane in REQUIRED_FIXED_STATE_LANES
        }
        checked[hash_name] = lane_values
        missing = [lane for lane, value in lane_values.items() if value is None]
        if missing:
            failures.append(
                {
                    "hash": hash_name,
                    "reason": "missing hash value",
                    "lanes": missing,
                    "values": lane_values,
                }
            )
            continue
        if len(set(lane_values.values())) != 1:
            failures.append(
                {
                    "hash": hash_name,
                    "reason": "hash values differ",
                    "values": lane_values,
                }
            )
    return {
        "status": "pass" if not failures else "blocked",
        "checked_hashes": checked,
        "failures": failures,
    }


def _fixed_state_required_lane_output_failures(
    lane: str,
    lane_payload: dict[str, Any],
) -> list[str]:
    failures = []
    hashes = lane_payload.get("hashes", {})
    if not isinstance(hashes, dict):
        hashes = {}
    for hash_name in REQUIRED_FIXED_STATE_HASH_KEYS:
        if hash_name not in hashes and hash_name not in lane_payload:
            failures.append(f"{lane}: missing required hash {hash_name}")
    assembled = lane_payload.get("assembled_outputs", {})
    if not isinstance(assembled, dict):
        assembled = {}
    operators = lane_payload.get("operator_outputs", {})
    if not isinstance(operators, dict):
        operators = {}
    for output_name in REQUIRED_ASSEMBLED_LANE_OUTPUT_KEYS:
        if output_name not in assembled:
            failures.append(f"{lane}: missing assembled output {output_name}")
    for output_name in REQUIRED_OPERATOR_LANE_OUTPUT_KEYS:
        if output_name not in operators:
            failures.append(f"{lane}: missing operator output {output_name}")
    return failures


def _fixed_state_bucket(fixed_state_artifact: dict[str, Any] | None) -> dict[str, Any]:
    if fixed_state_artifact is None:
        return {
            "status": "blocked",
            "reason": "fixed-state parity artifact is missing",
            "comparisons": {},
            "failures": ["fixed-state parity artifact is missing"],
        }
    lanes = fixed_state_artifact.get("lanes", {})
    comparisons = fixed_state_artifact.get("comparisons", {})
    missing_lanes = [
        lane
        for lane in REQUIRED_FIXED_STATE_LANES
        if lane not in lanes
    ]
    comparison_payloads = {
        f"{name}_fixed_state": _fixed_state_comparison(comparisons, name)
        for name in REQUIRED_FIXED_STATE_COMPARISONS
    }
    comparison_statuses = {
        name: _comparison_status(comparison)
        for name, comparison in comparison_payloads.items()
    }
    hash_equality = (
        _fixed_state_hash_equality(dict(lanes))
        if not missing_lanes
        else {
            "status": "blocked",
            "checked_hashes": {},
            "failures": [
                {
                    "reason": "required fixed-state lanes are missing",
                    "lanes": missing_lanes,
                }
            ],
        }
    )
    blocked = []
    drifted = []
    if missing_lanes:
        blocked.append(f"missing fixed-state lanes: {', '.join(missing_lanes)}")
    for lane in REQUIRED_FIXED_STATE_LANES:
        lane_payload = lanes.get(lane)
        if isinstance(lane_payload, dict):
            blocked.extend(
                _fixed_state_required_lane_output_failures(
                    lane,
                    lane_payload,
                )
            )
    if hash_equality["status"] != "pass":
        blocked.append("fixed-state hash equality gate failed")
    for name, status in comparison_statuses.items():
        if status == "blocked":
            blocked.append(f"{name}: blocked")
        elif status != "pass":
            drifted.append(f"{name}: {status}")
    if bool(fixed_state_artifact.get("passed", True)) is False:
        drifted.extend(str(failure) for failure in fixed_state_artifact.get("failures", []))
    return {
        "status": _status_from_failures(blocked=blocked, drifted=drifted),
        "comparisons": comparison_payloads,
        "hash_equality": hash_equality,
        "missing_lanes": missing_lanes,
        "failures": [*blocked, *drifted],
    }


def _section_status(section: Any) -> str:
    if isinstance(section, dict):
        return str(section.get("status", "pass"))
    if isinstance(section, list):
        statuses = [
            str(item.get("status", "pass"))
            for item in section
            if isinstance(item, dict)
        ]
        return _aggregate_status(statuses or ["pass"])
    return "pass" if section is not None else "blocked"


def _coordinate_mapping_bucket(
    coordinate_mapping_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if coordinate_mapping_artifact is None:
        return {
            "status": "blocked",
            "reason": "coordinate mapping proof is missing",
            "failures": ["coordinate mapping proof is missing"],
        }
    missing_sections = [
        section
        for section in COORDINATE_MAPPING_REQUIRED_SECTIONS
        if section not in coordinate_mapping_artifact
    ]
    section_statuses = {
        section: _section_status(coordinate_mapping_artifact.get(section))
        for section in COORDINATE_MAPPING_REQUIRED_SECTIONS
        if section in coordinate_mapping_artifact
    }
    blocked = [
        f"missing coordinate mapping section: {section}"
        for section in missing_sections
    ]
    drifted = [
        f"{section}: {status}"
        for section, status in section_statuses.items()
        if status not in {"pass", "blocked"}
    ]
    blocked.extend(
        f"{section}: blocked"
        for section, status in section_statuses.items()
        if status == "blocked"
    )
    top_status = str(coordinate_mapping_artifact.get("status", "blocked"))
    if top_status == "blocked":
        blocked.append("coordinate mapping artifact status is blocked")
    elif top_status != "pass":
        drifted.append(f"coordinate mapping artifact status is {top_status}")
    drifted.extend(str(failure) for failure in coordinate_mapping_artifact.get("failures", []))
    return {
        "status": _status_from_failures(blocked=blocked, drifted=drifted),
        "section_statuses": section_statuses,
        "missing_sections": missing_sections,
        "failures": [*blocked, *drifted],
    }


def _lane_timings(lane_payload: dict[str, Any]) -> dict[str, Any]:
    timings = lane_payload.get("timings", {})
    return dict(timings) if isinstance(timings, dict) else {}


def _lane_provenance(lane_payload: dict[str, Any]) -> dict[str, Any]:
    provenance = lane_payload.get("provenance", {})
    return dict(provenance) if isinstance(provenance, dict) else {}


def _lane_memory_metrics(provenance: dict[str, Any]) -> dict[str, Any]:
    peak_gpu_memory_mb = provenance.get("peak_gpu_memory_mb")
    if peak_gpu_memory_mb is None:
        peak_gpu_memory_mb = provenance.get("gpu_memory_mb")
    return {
        "peak_rss_mb": provenance.get("peak_rss_mb"),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _timing_value(timings: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(timings.get(key))
        if value is not None:
            return value
    return None


def _performance_summary_from_fixed_state(
    fixed_state_artifact: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    performance_summary = fixed_state_artifact.get("performance_summary_by_name")
    if isinstance(performance_summary, dict):
        return performance_summary, "fixed_state.performance_summary_by_name"
    performance = fixed_state_artifact.get("performance", {})
    if isinstance(performance, dict):
        nested_summary = performance.get("summary_by_name")
        if isinstance(nested_summary, dict):
            return nested_summary, "fixed_state.performance.summary_by_name"
    return None, None


def _performance_summary_from_report_timings(
    report: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    timings = report.get("timings", {})
    if not isinstance(timings, dict):
        return None, None
    cpu_timings = timings.get(LANE_CPU_SCIPY)
    gpu_timings = timings.get(LANE_JAX_GPU)
    if not isinstance(cpu_timings, dict) or not isinstance(gpu_timings, dict):
        return None, None

    cpu_elapsed_s = _timing_value(
        cpu_timings,
        "outer_optimizer_s",
        "script_total_s",
    )
    lane_outer_elapsed_s = _timing_value(
        gpu_timings,
        "outer_optimizer_s",
        "script_total_s",
    )
    lane_warm_elapsed_s = _timing_value(
        gpu_timings,
        "jax_optimizer_warm_run_s",
        "optimizer_warm_run_s",
        "warm_run_s",
    )
    explicit_compile_overhead_s = _timing_value(
        gpu_timings,
        "jax_optimizer_compile_overhead_s",
        "optimizer_compile_overhead_s",
        "compile_overhead_s",
    )
    outer_s = _timing_value(gpu_timings, "outer_optimizer_s")
    main_s = _timing_value(gpu_timings, "outer_optimizer_main_s")
    derived_compile_overhead_s = (
        outer_s - main_s
        if explicit_compile_overhead_s is None
        and outer_s is not None
        and main_s is not None
        else explicit_compile_overhead_s
    )
    if cpu_elapsed_s is None or lane_outer_elapsed_s is None:
        return None, None
    return (
        {
            "tier2_stage2_e2e": {
                "outer_speedup_vs_cpu": cpu_elapsed_s / lane_outer_elapsed_s,
                "warm_speedup_vs_cpu": (
                    cpu_elapsed_s / lane_warm_elapsed_s
                    if lane_warm_elapsed_s is not None
                    else None
                ),
                "lane_compile_overhead_s": derived_compile_overhead_s,
                "cpu_elapsed_s": cpu_elapsed_s,
                "lane_outer_elapsed_s": lane_outer_elapsed_s,
                "lane_warm_elapsed_s": lane_warm_elapsed_s,
            }
        },
        "parity_report.timings",
    )


def _performance_memory_bucket(
    fixed_state_artifact: dict[str, Any] | None,
    report: dict[str, Any],
) -> dict[str, Any]:
    if fixed_state_artifact is None:
        return {
            "status": "blocked",
            "failures": ["fixed-state artifact is required for performance/memory report"],
        }
    lanes = fixed_state_artifact.get("lanes", {})
    blocked = []
    drifted = []
    lane_reports: dict[str, Any] = {}
    for lane in REQUIRED_FIXED_STATE_LANES:
        lane_payload = lanes.get(lane)
        if not isinstance(lane_payload, dict):
            blocked.append(f"{lane}: lane payload is missing")
            continue
        provenance = _lane_provenance(lane_payload)
        timings = _lane_timings(lane_payload)
        missing_provenance = [
            key
            for key in PERFORMANCE_REQUIRED_PROVENANCE_KEYS
            if key not in provenance
        ]
        missing_timings = [
            key
            for key in PERFORMANCE_REQUIRED_TIMING_KEYS
            if key not in timings
        ]
        if missing_provenance:
            blocked.append(
                f"{lane}: missing provenance keys {', '.join(missing_provenance)}"
            )
        if missing_timings:
            blocked.append(f"{lane}: missing timing keys {', '.join(missing_timings)}")
        if provenance.get("x64_enabled") is not True:
            blocked.append(f"{lane}: x64_enabled is not true")
        platform = "cuda" if lane == LANE_JAX_GPU else "cpu"
        if lane == LANE_JAX_GPU:
            backend = str(provenance.get("backend", "")).lower()
            if backend not in {"cuda", "gpu"}:
                blocked.append(f"{lane}: CUDA lane did not prove CUDA backend execution")
            if not str(provenance.get("cuda_visible_devices") or "").strip():
                blocked.append(f"{lane}: CUDA_VISIBLE_DEVICES provenance is missing")
            if not provenance.get("nvidia_smi_gpus"):
                blocked.append(f"{lane}: NVIDIA GPU facts are missing")
            missing_cuda_provenance = [
                key for key in CUDA_REQUIRED_PROVENANCE_KEYS if not provenance.get(key)
            ]
            if missing_cuda_provenance:
                blocked.append(
                    f"{lane}: missing CUDA provenance keys "
                    f"{', '.join(missing_cuda_provenance)}"
                )
            xla_flags = str(provenance.get("xla_flags") or "")
            if "--xla_gpu_deterministic_ops=true" not in xla_flags.split():
                blocked.append(f"{lane}: deterministic GPU XLA flag is missing")
            if _lane_memory_metrics(provenance)["peak_gpu_memory_mb"] is None:
                blocked.append(f"{lane}: GPU memory high-water is missing")
        memory_metrics = _lane_memory_metrics(provenance)
        memory_failures = evaluate_grouped_adjoint_memory_budget(
            memory_metrics,
            grouped_adjoint_memory_budget(
                fixture="real_single_stage_init",
                platform=platform,
            ),
        )
        drifted.extend(f"{lane}: {failure}" for failure in memory_failures)
        lane_reports[lane] = {
            "provenance": provenance,
            "timings": timings,
            "memory_metrics": memory_metrics,
            "memory_failures": memory_failures,
        }
    performance_summary, performance_summary_source = _performance_summary_from_fixed_state(
        fixed_state_artifact
    )
    if performance_summary is None:
        performance_summary, performance_summary_source = (
            _performance_summary_from_report_timings(report)
        )
    if performance_summary is None:
        blocked.append("performance summary is missing")
        performance_failures = []
    else:
        performance_failures = evaluate_tier5_performance_budget(
            performance_summary,
            tier5_performance_budget(profile="stable_hardware_weekly"),
        )
        drifted.extend(performance_failures)
    return {
        "status": _status_from_failures(blocked=blocked, drifted=drifted),
        "lanes": lane_reports,
        "performance_summary_by_name": performance_summary,
        "performance_summary_source": performance_summary_source,
        "performance_failures": performance_failures,
        "failures": [*blocked, *drifted],
    }


def _metric_pair_summary(
    metrics: dict[str, Any],
    *,
    lhs_lane: str,
    rhs_lane: str,
    initial_only: bool = False,
) -> dict[str, Any]:
    return _comparison_summary(
        _merged_metric_comparisons(
            metrics,
            lhs_lane=lhs_lane,
            rhs_lane=rhs_lane,
            initial_only=initial_only,
        )
    )


def _comparison_failure_reasons(comparisons: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    for comparison_name, comparison in comparisons.items():
        status = comparison.get("status")
        if status == "pass":
            continue
        drifted_metrics = comparison.get("drifted_metrics")
        if drifted_metrics:
            failures.append(
                f"{comparison_name}: drifted metrics {', '.join(drifted_metrics)}"
            )
            continue
        reason = comparison.get("reason")
        failures.append(
            f"{comparison_name}: {reason}"
            if reason is not None
            else f"{comparison_name}: {status}"
        )
    return failures


def _optimizer_public_behavior_bucket(
    *,
    cpu_jax_metrics: dict[str, Any],
    cpu_gpu_metrics: dict[str, Any],
    full_trajectory: dict[str, Any],
) -> dict[str, Any]:
    statuses = [
        cpu_jax_metrics["status"],
        cpu_gpu_metrics["status"],
        full_trajectory["status"],
    ]
    comparisons = {
        "cpu_scipy_vs_jax_cpu_public_behavior": cpu_jax_metrics,
        "cpu_scipy_vs_jax_gpu_public_behavior": cpu_gpu_metrics,
        "trajectory_termination_and_trace": full_trajectory,
    }
    return {
        "status": _aggregate_status(statuses),
        "comparisons": comparisons,
        "failures": _comparison_failure_reasons(comparisons),
    }


def _final_metric_envelope_bucket(
    *,
    cpu_jax_metrics: dict[str, Any],
    cpu_gpu_metrics: dict[str, Any],
    jax_cpu_gpu_metrics: dict[str, Any],
) -> dict[str, Any]:
    statuses = [
        cpu_jax_metrics["status"],
        cpu_gpu_metrics["status"],
        jax_cpu_gpu_metrics["status"],
    ]
    comparisons = {
        "cpu_scipy_vs_jax_cpu_final_metrics": cpu_jax_metrics,
        "cpu_scipy_vs_jax_gpu_final_metrics": cpu_gpu_metrics,
        "jax_cpu_vs_jax_gpu_initial_metrics": jax_cpu_gpu_metrics,
    }
    return {
        "status": _aggregate_status(statuses),
        "comparisons": comparisons,
        "failures": _comparison_failure_reasons(comparisons),
    }


def _first_divergence_from_buckets(
    buckets: dict[str, dict[str, Any]],
    *,
    full_trajectory_reasons: list[str],
) -> dict[str, Any] | None:
    if buckets["fixed_state_physics_parity"]["status"] != "pass":
        stage = "fixed_state_physics"
        evidence = buckets["fixed_state_physics_parity"]["failures"]
        metric = "fixed_state_physics_parity"
    elif buckets["coordinate_mapping_parity"]["status"] != "pass":
        stage = "coordinate_mapping"
        evidence = buckets["coordinate_mapping_parity"]["failures"]
        metric = "coordinate_mapping_parity"
    elif buckets["full_run_artifact_contract"]["status"] != "pass":
        stage = "run_contract"
        evidence = buckets["full_run_artifact_contract"]["failures"]
        metric = "full_run_artifact_contract"
    elif any("optimizer_state_trace" in reason for reason in full_trajectory_reasons):
        stage = "line_search"
        evidence = full_trajectory_reasons
        metric = "optimizer_state_trace"
    elif any("termination" in reason for reason in full_trajectory_reasons):
        stage = "termination"
        evidence = full_trajectory_reasons
        metric = "termination"
    elif buckets["final_metric_envelope"]["status"] != "pass":
        stage = "final_sync"
        evidence = buckets["final_metric_envelope"]
        metric = "final_metric_envelope"
    elif buckets["optimizer_public_behavior_parity"]["status"] != "pass":
        stage = "initial_gradient"
        evidence = buckets["optimizer_public_behavior_parity"]
        metric = "optimizer_public_behavior_parity"
    else:
        return None
    if stage not in FIRST_DIVERGENCE_STAGES:
        raise ValueError(f"Unknown first_divergence stage {stage!r}.")
    return {
        "stage": stage,
        "lane_pair": None,
        "metric": metric,
        "evidence": evidence,
        "explanation": f"Release gate first diverged at {stage}.",
    }


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return "`" + str(value).replace("|", "\\|") + "`"
    return str(value).replace("|", "\\|")


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return lines


def _comparison_delta_rows(matrix: dict[str, Any]) -> list[tuple[Any, ...]]:
    fixed_state = matrix.get("buckets", {}).get("fixed_state_physics_parity", {})
    comparisons = fixed_state.get("comparisons", {})
    rows = []
    for name, comparison in comparisons.items():
        if isinstance(comparison, dict):
            rows.append(
                (
                    name,
                    comparison.get("status"),
                    comparison.get("objective_abs_delta"),
                    comparison.get("objective_rel_delta"),
                    comparison.get("grad_max_abs_delta"),
                )
            )
    return rows


def _lane_runtime_rows(matrix: dict[str, Any]) -> list[tuple[Any, ...]]:
    fixed_state = matrix.get("fixed_state_artifact", {})
    lanes = fixed_state.get("lanes", {}) if isinstance(fixed_state, dict) else {}
    rows = []
    for lane_name, lane in lanes.items():
        if not isinstance(lane, dict):
            continue
        provenance = lane.get("provenance", {})
        timings = lane.get("timings", {})
        if not isinstance(provenance, dict):
            provenance = {}
        if not isinstance(timings, dict):
            timings = {}
        rows.append(
            (
                lane_name,
                lane.get("status"),
                provenance.get("backend"),
                provenance.get("jax"),
                provenance.get("jaxlib"),
                timings.get("compile_time_s"),
                timings.get("run_time_s"),
                provenance.get("peak_rss_mb"),
                provenance.get("peak_gpu_memory_mb") or provenance.get("gpu_memory_mb"),
            )
        )
    return rows


def _metric_rows(bucket: dict[str, Any]) -> list[tuple[Any, ...]]:
    comparisons = bucket.get("comparisons", {}) if isinstance(bucket, dict) else {}
    rows = []
    for comparison_name, comparison in comparisons.items():
        if not isinstance(comparison, dict):
            continue
        metrics = comparison.get("metrics", {})
        if not metrics:
            rows.append(
                (
                    comparison_name,
                    comparison.get("status"),
                    comparison.get("reason"),
                    "",
                    "",
                    "",
                )
            )
            continue
        for metric_name, metric in metrics.items():
            if isinstance(metric, dict):
                rows.append(
                    (
                        comparison_name,
                        metric_name,
                        metric.get("status"),
                        metric.get("abs_delta"),
                        metric.get("rel_delta"),
                        metric.get("rtol"),
                    )
                )
    return rows


def _memory_budget_rows(matrix: dict[str, Any]) -> list[tuple[Any, ...]]:
    performance = matrix.get("buckets", {}).get("performance_memory_report", {})
    lane_reports = performance.get("lanes", {}) if isinstance(performance, dict) else {}
    performance_failures = (
        performance.get("failures", []) if isinstance(performance, dict) else []
    )
    fixed_state = matrix.get("fixed_state_artifact", {})
    fixed_lanes = fixed_state.get("lanes", {}) if isinstance(fixed_state, dict) else {}
    rows = []
    for lane_name, report in lane_reports.items():
        if not isinstance(report, dict):
            continue
        metrics = report.get("memory_metrics", {})
        lane_payload = fixed_lanes.get(lane_name, {})
        lane_status = (
            lane_payload.get("status") if isinstance(lane_payload, dict) else None
        )
        failures = [
            *report.get("memory_failures", []),
            *(
                failure
                for failure in performance_failures
                if str(failure).startswith(f"{lane_name}:")
            ),
        ]
        rows.append(
            (
                lane_name,
                lane_status if lane_status != "pass" else "pass" if not failures else "fail",
                metrics.get("peak_rss_mb") if isinstance(metrics, dict) else None,
                metrics.get("peak_gpu_memory_mb") if isinstance(metrics, dict) else None,
                "; ".join(str(failure) for failure in failures),
            )
        )
    return rows


def _performance_budget_rows(matrix: dict[str, Any]) -> list[tuple[Any, ...]]:
    performance = matrix.get("buckets", {}).get("performance_memory_report", {})
    if not isinstance(performance, dict):
        return []
    failures = performance.get("performance_failures", [])
    if failures:
        return [("performance_budget", "fail", failure) for failure in failures]
    if performance.get("status") == "pass":
        return [("performance_budget", "pass", "")]
    return [("performance_budget", "blocked", "performance summary is missing")]


def _device_version_rows(matrix: dict[str, Any]) -> list[tuple[Any, ...]]:
    fixed_state = matrix.get("fixed_state_artifact", {})
    lanes = fixed_state.get("lanes", {}) if isinstance(fixed_state, dict) else {}
    rows = []
    for lane_name, lane in lanes.items():
        if not isinstance(lane, dict):
            continue
        provenance = lane.get("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
        rows.append(
            (
                lane_name,
                provenance.get("backend"),
                provenance.get("devices"),
                provenance.get("x64_enabled"),
                provenance.get("matmul_precision"),
                provenance.get("jax"),
                provenance.get("jaxlib"),
                provenance.get("cuda_runtime_version"),
                provenance.get("cuda_driver_version"),
                provenance.get("nvcc_version"),
                provenance.get("xla_flags"),
                provenance.get("cuda_visible_devices"),
                provenance.get("nvidia_smi_gpus"),
            )
        )
    return rows


def _git_status_lines(matrix: dict[str, Any]) -> list[str]:
    fixed_state = matrix.get("fixed_state_artifact", {})
    lanes = fixed_state.get("lanes", {}) if isinstance(fixed_state, dict) else {}
    for lane in lanes.values():
        if not isinstance(lane, dict):
            continue
        provenance = lane.get("provenance", {})
        if not isinstance(provenance, dict):
            continue
        dirty = provenance.get("dirty_worktree")
        lines = [
            f"- git SHA: `{provenance.get('repo_sha')}`",
        ]
        if isinstance(dirty, dict):
            lines.append(f"- dirty: `{dirty.get('is_dirty')}`")
            lines.append(f"- dirty entry count: `{dirty.get('entry_count')}`")
        return lines
    return ["- git SHA: unknown", "- dirty: unknown"]


def _release_reason_lines(matrix: dict[str, Any]) -> list[str]:
    buckets = matrix.get("buckets", {})
    blocking_buckets = matrix.get("blocking_buckets", [])
    if not blocking_buckets:
        return ["- all release-gate buckets passed."]
    lines = []
    for bucket_name in blocking_buckets:
        bucket = buckets.get(bucket_name, {})
        failures = bucket.get("failures", []) if isinstance(bucket, dict) else []
        status = bucket.get("status") if isinstance(bucket, dict) else "blocked"
        verb = "failed" if status == "drift" else "is blocked"
        reason = str(failures[0]).rstrip(".") if failures else "status is not pass"
        lines.append(f"- {bucket_name} {verb} because {reason}.")
    if "fixed_state_physics_parity" in blocking_buckets:
        lines.append("- JAX CPU vs GPU parity is not enough for release.")
    return lines


def build_release_gate_markdown_report(matrix: dict[str, Any]) -> str:
    verdict = "PASS" if matrix.get("release_gate_passed") else "FAIL"
    lines = [
        f"# Single-Stage Release Gate: {verdict}",
        "",
        f"Release gate: {verdict}",
        "",
        "## Reason",
        *_release_reason_lines(matrix),
        "",
    ]
    buckets = matrix.get("buckets", {})
    if isinstance(buckets, dict):
        lines.extend(
            [
                "## Bucket Status",
                *_markdown_table(
                    ("bucket", "status", "failure_count"),
                    [
                        (
                            bucket_name,
                            bucket.get("status") if isinstance(bucket, dict) else "",
                            len(bucket.get("failures", []))
                            if isinstance(bucket, dict)
                            else "",
                        )
                        for bucket_name, bucket in buckets.items()
                    ],
                ),
                "",
            ]
        )
    blocking = matrix.get("blocking_comparisons", [])
    lines.extend(
        [
            "## Blocking Comparisons",
            *(f"- {item}" for item in blocking),
            *([] if blocking else ["- none"]),
            "",
        ]
    )
    fixed_rows = _comparison_delta_rows(matrix)
    lines.extend(["## Fixed-State Deltas"])
    lines.extend(
        _markdown_table(
            (
                "comparison",
                "status",
                "objective_abs_delta",
                "objective_rel_delta",
                "grad_max_abs_delta",
            ),
            fixed_rows,
        )
        if fixed_rows
        else ["- no fixed-state comparisons recorded."]
    )
    coordinate = buckets.get("coordinate_mapping_parity", {})
    public_rows = _metric_rows(buckets.get("optimizer_public_behavior_parity", {}))
    final_rows = _metric_rows(buckets.get("final_metric_envelope", {}))
    lines.extend(
        [
            "",
            "## Coordinate Mapping",
            f"- status: {coordinate.get('status') if isinstance(coordinate, dict) else ''}",
            "",
            "## Full-Run Public Behavior Deltas",
        ]
    )
    lines.extend(
        _markdown_table(
            ("comparison", "metric", "status", "abs_delta", "rel_delta", "rtol"),
            public_rows,
        )
        if public_rows
        else ["- no public behavior deltas recorded."]
    )
    lines.extend(
        [
            "",
            "## Final Metric Envelope",
        ]
    )
    lines.extend(
        _markdown_table(
            ("comparison", "metric", "status", "abs_delta", "rel_delta", "rtol"),
            final_rows,
        )
        if final_rows
        else ["- no final metric deltas recorded."]
    )
    lines.extend(
        [
            "",
            "## First Divergence",
            f"- `{matrix.get('first_divergence')}`",
            "",
            "## Runtime Table",
        ]
    )
    runtime_rows = _lane_runtime_rows(matrix)
    lines.extend(
        _markdown_table(
            (
                "lane",
                "status",
                "backend",
                "jax",
                "jaxlib",
                "compile_s",
                "run_s",
                "peak_rss_mb",
                "peak_gpu_mb",
            ),
            runtime_rows,
        )
        if runtime_rows
        else ["- no fixed-state lane runtime table recorded."]
    )
    memory_rows = _memory_budget_rows(matrix)
    performance_rows = _performance_budget_rows(matrix)
    device_rows = _device_version_rows(matrix)
    lines.extend(
        [
            "",
            "## Memory Table",
            *(
                _markdown_table(
                    ("lane", "status", "peak_rss_mb", "peak_gpu_mb", "failures"),
                    memory_rows,
                )
                if memory_rows
                else ["- no memory metrics recorded."]
            ),
            "",
            "## Memory And Performance Budgets",
            *(
                _markdown_table(("budget", "status", "failure"), performance_rows)
                if performance_rows
                else ["- no budget status recorded."]
            ),
            "",
            "## Device And Version Table",
            *(
                _markdown_table(
                    (
                        "lane",
                        "backend",
                        "devices",
                        "x64",
                        "matmul_precision",
                        "jax",
                        "jaxlib",
                        "cuda_runtime",
                        "cuda_driver",
                        "nvcc",
                        "xla_flags",
                        "cuda_visible_devices",
                        "nvidia_smi_gpus",
                    ),
                    device_rows,
                )
                if device_rows
                else ["- no device/version data recorded."]
            ),
            "",
            "## Git Status",
            *_git_status_lines(matrix),
        ]
    )
    artifact_paths = matrix.get("artifact_paths", {})
    lines.extend(["", "## Artifacts"])
    lines.extend(
        [f"- {name}: `{path}`" for name, path in artifact_paths.items()]
        if artifact_paths
        else ["- none recorded"]
    )
    lines.extend(["", "## Command", "- `" + " ".join(sys.argv) + "`", ""])
    return "\n".join(lines)


def build_single_stage_parity_matrix(
    report: dict[str, Any],
    *,
    cpu_progress_json: str | None = None,
    jax_cpu_progress_json: str | None = None,
    gpu_progress_json: str | None = None,
    fixed_state_artifact: dict[str, Any] | None = None,
    coordinate_mapping_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    release_gate_mode = bool(
        fixed_state_artifact is not None or coordinate_mapping_artifact is not None
    )
    gpu_lane = LANE_JAX_GPU if release_gate_mode else LANE_H100_GPU
    metrics = report["same_seed_no_optimizer_metrics"]
    full_run_contract = (
        _full_run_artifact_contract_bucket(
            report,
            cpu_progress_json=cpu_progress_json,
            jax_cpu_progress_json=jax_cpu_progress_json,
            gpu_progress_json=gpu_progress_json,
        )
        if release_gate_mode
        else None
    )
    terminations = _progress_termination_values(
        _termination_values(metrics),
        cpu_progress_json=cpu_progress_json,
        jax_cpu_progress_json=jax_cpu_progress_json,
        gpu_progress_json=gpu_progress_json,
        gpu_lane=gpu_lane,
    )
    full_run_contract_passed = (
        full_run_contract is None or full_run_contract["status"] == "pass"
    )
    if full_run_contract_passed:
        cpu_jax_metrics = _metric_pair_summary(
            metrics,
            lhs_lane=LANE_JAX_CPU,
            rhs_lane=LANE_CPU_SCIPY,
            initial_only=False,
        )
        cpu_gpu_metrics = _metric_pair_summary(
            metrics,
            lhs_lane=gpu_lane,
            rhs_lane=LANE_CPU_SCIPY,
            initial_only=False,
        )
        jax_gpu_initial_metrics = _metric_pair_summary(
            metrics,
            lhs_lane=gpu_lane,
            rhs_lane=LANE_JAX_CPU,
            initial_only=True,
        )
        cpu_trace = _load_optimizer_state_trace(cpu_progress_json)
        jax_cpu_trace = _load_optimizer_state_trace(jax_cpu_progress_json)
        gpu_trace = _load_optimizer_state_trace(gpu_progress_json)
        optimizer_trace_pairs = {
            f"jax_cpu_vs_{gpu_lane}": _compare_optimizer_state_trace_pair(
                jax_cpu_trace,
                gpu_trace,
            )
        }
        if cpu_trace and jax_cpu_trace:
            optimizer_trace_pairs["cpu_cpp_trace_vs_jax_cpu"] = (
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
        trajectory_lanes = [LANE_JAX_CPU, gpu_lane]
        if cpu_progress_json is not None:
            trajectory_lanes.append(LANE_CPU_SCIPY)
        trajectory_terminations = [
            terminations[lane]
            for lane in trajectory_lanes
            if lane in terminations
        ]
        if len(set(trajectory_terminations)) > 1:
            full_trajectory_status = "blocked"
            full_trajectory_reasons.append("trajectory termination modes differ")
    else:
        reason = "full-run artifact contract failed"
        cpu_jax_metrics = _blocked_metric_summary(reason, full_run_contract)
        cpu_gpu_metrics = _blocked_metric_summary(reason, full_run_contract)
        jax_gpu_initial_metrics = _blocked_metric_summary(reason, full_run_contract)
        optimizer_trace_pairs = {}
        optimizer_trace_status = "blocked"
        full_trajectory_status = "blocked"
        full_trajectory_reasons = [reason]
    full_trajectory = {
        "status": full_trajectory_status,
        "reasons": full_trajectory_reasons,
        "termination_messages": terminations,
    }
    same_state_key = (
        "jax_cpu_vs_jax_gpu_same_state_value_grad"
        if release_gate_mode
        else "jax_cpu_vs_h100_same_state_value_grad"
    )
    same_state_summary = (
        _release_gate_same_state_value_grad_summary(report)
        if release_gate_mode
        else _same_state_value_grad_summary(report)
    )
    comparisons = {
        same_state_key: same_state_summary,
        "cpu_scipy_vs_jax_cpu_same_seed_metrics": cpu_jax_metrics,
        f"jax_cpu_vs_{gpu_lane}_initial_metrics": jax_gpu_initial_metrics,
        "optimizer_state_trace_pairs": {
            "status": optimizer_trace_status,
            "pairs": optimizer_trace_pairs,
        },
        "full_trajectory_parity": full_trajectory,
    }
    blocking = [
        name
        for name, comparison in comparisons.items()
        if comparison["status"] != "pass"
    ]
    payload = {
        "comparisons": comparisons,
        "blocking_comparisons": blocking,
        "passed": not blocking,
    }
    if not release_gate_mode:
        return payload

    buckets = {
        "fixed_state_physics_parity": _fixed_state_bucket(fixed_state_artifact),
        "coordinate_mapping_parity": _coordinate_mapping_bucket(
            coordinate_mapping_artifact
        ),
        "full_run_artifact_contract": full_run_contract,
        "optimizer_public_behavior_parity": _optimizer_public_behavior_bucket(
            cpu_jax_metrics=cpu_jax_metrics,
            cpu_gpu_metrics=cpu_gpu_metrics,
            full_trajectory=full_trajectory,
        ),
        "final_metric_envelope": _final_metric_envelope_bucket(
            cpu_jax_metrics=cpu_jax_metrics,
            cpu_gpu_metrics=cpu_gpu_metrics,
            jax_cpu_gpu_metrics=jax_gpu_initial_metrics,
        ),
        "performance_memory_report": _performance_memory_bucket(
            fixed_state_artifact,
            report,
        ),
    }
    blocking_buckets = [
        bucket_name
        for bucket_name in RELEASE_GATE_BUCKETS
        if buckets[bucket_name]["status"] != "pass"
    ]
    first_divergence = _first_divergence_from_buckets(
        buckets,
        full_trajectory_reasons=full_trajectory_reasons,
    )
    payload.update(
        {
            "release_gate_mode": True,
            "buckets": buckets,
            "blocking_buckets": blocking_buckets,
            "release_gate_passed": not blocking_buckets,
            "first_divergence": first_divergence,
            "passed": not blocking and not blocking_buckets,
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    if args.fixed_state_parity_json and not args.coordinate_mapping_json:
        raise SystemExit(
            "--coordinate-mapping-json is required with --fixed-state-parity-json"
        )
    report = load_json(args.parity_report_json)
    fixed_state_artifact = (
        load_json(args.fixed_state_parity_json)
        if args.fixed_state_parity_json is not None
        else None
    )
    coordinate_mapping_artifact = (
        load_json(args.coordinate_mapping_json)
        if args.coordinate_mapping_json is not None
        else None
    )
    payload = build_single_stage_parity_matrix(
        report,
        cpu_progress_json=args.cpu_progress_json,
        jax_cpu_progress_json=args.jax_cpu_progress_json,
        gpu_progress_json=args.gpu_progress_json,
        fixed_state_artifact=fixed_state_artifact,
        coordinate_mapping_artifact=coordinate_mapping_artifact,
    )
    payload["artifact_paths"] = {
        "parity_report_json": args.parity_report_json,
        "fixed_state_parity_json": args.fixed_state_parity_json,
        "coordinate_mapping_json": args.coordinate_mapping_json,
        "cpu_progress_json": args.cpu_progress_json,
        "jax_cpu_progress_json": args.jax_cpu_progress_json,
        "gpu_progress_json": args.gpu_progress_json,
        "output_json": args.output_json,
        "output_md": args.output_md,
    }
    if fixed_state_artifact is not None:
        payload["fixed_state_artifact"] = fixed_state_artifact
    write_json(args.output_json, payload)
    if args.output_md is not None:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_release_gate_markdown_report(payload))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
