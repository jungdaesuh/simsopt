"""Reduced convergence probe for the target single-stage outer optimizer path."""

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

from benchmarks.single_stage_init_parity import (
    _collect_partial_single_stage_case_artifacts,
    _prefix_phase_timings,
    _run_single_stage_case,
    resolve_target_lane_compile_diagnostics,
)
from benchmarks.single_stage_backend_routing import (
    resolve_boozer_least_squares_algorithm,
    resolve_boozer_optimizer_backend,
    resolve_boozer_optimizer_method,
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
    SMOKE_TEST_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
)
from benchmarks.validation_ladder_common import (
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    maybe_initialize_distributed_runtime,
    preparse_platform,
    print_provenance,
    require_x64_runtime,
    resolve_probe_lane,
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    single_stage_proof_contract,
    load_json,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_benchmark_compilation_cache_policy(
    "single_stage_outer_loop_probe",
    requested_platform=REQUESTED_PLATFORM,
)

import jax
import jaxlib

maybe_initialize_distributed_runtime()
jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Single-stage outer-loop probe")

LADDER_RUNG = TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG
_OUTER_LOOP_PROOF_CONTRACT = single_stage_proof_contract(LADDER_RUNG)
TARGET_OUTER_OPTIMIZER_METHOD = str(
    _OUTER_LOOP_PROOF_CONTRACT["required_outer_optimizer_method"]
)
DEFAULT_OUTER_PROOF_MAXITER = int(_OUTER_LOOP_PROOF_CONTRACT["default_maxiter"])
_MIN_ACCEPTED_ITERATIONS = int(_OUTER_LOOP_PROOF_CONTRACT["min_iterations"])
_REQUIRE_OBJECTIVE_DECREASE = bool(
    _OUTER_LOOP_PROOF_CONTRACT.get("require_objective_decrease", False)
)
_REQUIRED_RESULT_KEYS = tuple(_OUTER_LOOP_PROOF_CONTRACT["required_result_keys"])
_PHASE1_INITIAL_TRACE_REGION = "single_stage.outer_optimizer_initial_phase"
_PHASE2_TRACE_REGION = "single_stage.outer_optimizer"
_HOST_JAX_OUTER_OPTIMIZER_METHOD = "lbfgs"
_OUTER_LOOP_PROBE_OPTIMIZER_BACKENDS = (DEFAULT_OPTIMIZER_BACKEND, "host-jax")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced real single-stage target lane long enough to prove "
            "the outer optimizer takes a real multi-step descent path without "
            "entering SciPy."
        )
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
        help="Path to write structured probe results.",
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
        default=str(SMOKE_TEST_STAGE2_BS_PATH),
        help="Path to the fixed Stage 2 seed biot_savart_opt.json fixture.",
    )
    parser.add_argument(
        "--jax-runtime-seed-spec",
        default=None,
        help=(
            "Immutable single-stage JAX runtime seed spec matching the requested "
            "surface/quadrature resolution."
        ),
    )
    parser.add_argument(
        "--reuse-jax-runtime-seed-solve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --jax-runtime-seed-spec is supplied for an outer run, install "
            "the seed's resolved surface/iota/G state before the first objective "
            "evaluation instead of replaying startup Boozer initialization."
        ),
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
        choices=_OUTER_LOOP_PROBE_OPTIMIZER_BACKENDS,
        default=DEFAULT_OPTIMIZER_BACKEND,
        help="JAX optimizer backend for the outer-loop proof.",
    )
    parser.add_argument(
        "--boozer-optimizer-backend",
        choices=("host-jax", "ondevice", "scipy"),
        default=None,
        help=(
            "Optional override for the inner JAX Boozer LS backend. "
            "Defaults to the backend resolved from the selected outer lane."
        ),
    )
    parser.add_argument(
        "--boozer-least-squares-algorithm",
        choices=("quasi-newton", "lm", "lm-minpack"),
        default=None,
        help=(
            "Optional inner Boozer LS algorithm override forwarded to the "
            "single-stage runner. Use 'lm' to exercise the residual/Jacobian "
            "kernel path under host-jax."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_OUTER_PROOF_MAXITER,
        help="Single-stage outer-loop iteration budget for the proof rung.",
    )
    parser.add_argument(
        "--outer-maxls",
        type=int,
        default=20,
        help=(
            "Strong-Wolfe line-search budget forwarded to the single-stage "
            "outer optimizer."
        ),
    )
    parser.add_argument(
        "--single-stage-case-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional wall-clock timeout for the single-stage child process. "
            "A positive value writes a structured failed probe JSON with partial "
            "artifact paths when the child cannot finish locally."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-bfgs-tol",
        type=float,
        default=None,
        help="Optional temporary Boozer LS tolerance override for target-lane trials.",
    )
    parser.add_argument(
        "--target-lane-boozer-bfgs-maxiter",
        type=int,
        default=None,
        help="Optional temporary Boozer LS iteration cap for target-lane trials.",
    )
    parser.add_argument(
        "--target-lane-boozer-newton-tol",
        type=float,
        default=None,
        help=(
            "Optional temporary Boozer Newton tolerance override for target-lane "
            "trials."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-newton-maxiter",
        type=int,
        default=None,
        help=("Optional temporary Boozer Newton iteration cap for target-lane trials."),
    )
    parser.add_argument(
        "--profile-target-lane",
        action="store_true",
        help="Record target-lane objective profiling breakdowns in the probe payload.",
    )
    parser.add_argument(
        "--profile-target-lane-only",
        action="store_true",
        help=(
            "Build and profile the target-lane runtime bundle without running the "
            "outer optimizer."
        ),
    )
    parser.add_argument(
        "--profile-target-lane-batch-size",
        type=int,
        default=1,
        help=(
            "When profiling the target lane, also record the batched seed-evaluation "
            "path over this many nearby deterministic seed points."
        ),
    )
    parser.add_argument(
        "--jax-profile-dir",
        default=None,
        help=(
            "Optional JAX/XProf trace output directory threaded through to the "
            "single-stage example subprocess."
        ),
    )
    parser.add_argument(
        "--diagnose-target-lane-scaled-phase1",
        action="store_true",
        help=(
            "Run the scaled phase-1 target-lane diagnosis instead of the full "
            "outer-loop proof so the first failing/stalling region can be "
            "localized."
        ),
    )
    parser.add_argument(
        "--diagnostic-callbacks",
        dest="record_target_lane_invalid_state_events",
        action="store_true",
        help=(
            "Opt in to target-lane host callbacks so proof failures carry a "
            "structured first-event postmortem. On explicit CUDA-only runs "
            "this also keeps a CPU lane available for JAX host callbacks."
        ),
    )
    parser.add_argument(
        "--record-target-lane-invalid-state-events",
        dest="record_target_lane_invalid_state_events",
        action="store_true",
        help="Deprecated alias for --diagnostic-callbacks.",
    )
    parser.add_argument(
        "--enable-compile-diagnostics",
        action="store_true",
        help=(
            "Enable scoped JAX compile/cache-miss diagnostics inside the real "
            "single-stage target-lane or host-jax section."
        ),
    )
    parser.add_argument(
        "--record-objective-evaluation-trace",
        action="store_true",
        help=(
            "Record every outer objective evaluation to outer_optimizer_progress.json."
        ),
    )
    parser.add_argument(
        "--replay-objective-evaluation-trace",
        default=None,
        help=(
            "Evaluate only the objective candidates recorded in an existing "
            "outer_optimizer_progress.json trace."
        ),
    )
    parser.add_argument(
        "--enable-phase0-noise-calibration-gate",
        action="store_true",
        help=(
            "Compare a baseline fixed-candidate trace with this run's replay-grade "
            "objective-evaluation trace for Phase-0 delta-J calibration."
        ),
    )
    parser.add_argument(
        "--phase0-noise-baseline-progress-json",
        default=None,
        help=(
            "Baseline outer_optimizer_progress.json for Phase-0 noise calibration. "
            "Defaults to --replay-objective-evaluation-trace."
        ),
    )
    parser.add_argument(
        "--phase0-noise-baseline-newton-tol",
        type=float,
        default=1.0e-11,
        help="Expected Boozer Newton tolerance in the baseline Phase-0 trace.",
    )
    parser.add_argument(
        "--phase0-noise-tightened-newton-tol",
        type=float,
        default=1.0e-13,
        help="Expected Boozer Newton tolerance in this run's Phase-0 trace.",
    )
    parser.add_argument(
        "--phase0-noise-objective-evaluation-index",
        type=int,
        default=0,
        help="Objective-evaluation event index to compare for Phase-0 calibration.",
    )
    parser.add_argument(
        "--enable-phase3-gradient-proof-gate",
        action="store_true",
        help=(
            "Evaluate the Phase-3 accepted-step gradient proof gate from "
            "outer_optimizer_progress.json."
        ),
    )
    parser.add_argument(
        "--phase3-constraint-margin-abs-tol",
        type=float,
        default=None,
        help=(
            "Optional maximum absolute hardware threshold margin for treating a "
            "Phase-3 accepted-step comparison as constraint-marginal."
        ),
    )
    parser.add_argument(
        "--enable-host-jax-memory-gate",
        action="store_true",
        help=(
            "Evaluate the host-jax post-warm objective-evaluation memory no-growth gate."
        ),
    )
    parser.add_argument(
        "--host-jax-memory-warmup-evaluations",
        type=int,
        default=1,
        help="Objective-evaluation snapshots to exclude before memory growth checks.",
    )
    parser.add_argument(
        "--max-host-jax-steady-rss-growth-mb",
        type=float,
        default=64.0,
        help="Allowed post-warm host RSS high-water growth in MB.",
    )
    parser.add_argument(
        "--max-host-jax-steady-gpu-growth-mb",
        type=float,
        default=64.0,
        help="Allowed post-warm nvidia-smi GPU memory growth in MB.",
    )
    parser.add_argument(
        "--deterministic-gpu-reductions",
        action="store_true",
        help=(
            "Append --xla_gpu_exclude_nondeterministic_ops=true for the CUDA subprocess as "
            "a phase-1 A/B experiment."
        ),
    )
    parser.add_argument(
        "--experimental-target-lane-value-and-grad",
        action="store_true",
        help=(
            "Legacy compatibility flag. The explicit ondevice target lane now "
            "uses the fused runtime-bundle (value, grad) contract by default."
        ),
    )
    return parser.parse_args()


def _finite_result_keys(results: dict[str, Any]) -> dict[str, bool]:
    return {
        key: bool(np.isfinite(float(results.get(key, np.nan))))
        for key in _REQUIRED_RESULT_KEYS
    }


def _finite_optional_scalar(value: object) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _expected_outer_optimizer_method(optimizer_backend: str) -> str:
    if optimizer_backend == "host-jax":
        return _HOST_JAX_OUTER_OPTIMIZER_METHOD
    return TARGET_OUTER_OPTIMIZER_METHOD


def _objective_evaluation_events(progress_json: str | Path) -> list[dict[str, Any]]:
    payload = load_json(progress_json)
    return [
        event
        for event in payload.get("events", [])
        if isinstance(event, dict) and event.get("label") == "objective_evaluation"
    ]


def _objective_evaluation_events_with_failures(
    progress_json: str | Path,
    *,
    context: str = "Objective-evaluation gate",
) -> tuple[list[dict[str, Any]], list[str]]:
    progress_path = Path(progress_json)
    if not progress_path.is_file():
        return [], [
            f"{context} requires objective-evaluation progress trace, "
            f"but it was not written: {progress_path}."
        ]
    return _objective_evaluation_events(progress_path), []


def _event_objective_value(event: dict[str, Any]) -> float | None:
    objective = event.get("objective")
    if isinstance(objective, dict):
        return _finite_optional_scalar(objective.get("value"))
    return _finite_optional_scalar(objective)


def _event_objective_finite(event: dict[str, Any]) -> bool | None:
    objective = event.get("objective")
    if isinstance(objective, dict) and objective.get("finite") is not None:
        return bool(objective["finite"])
    return _event_objective_value(event) is not None


def _event_optimizer_gradient_summary(
    event: dict[str, Any],
) -> dict[str, float | bool | None]:
    gradient = event.get("optimizer_gradient")
    if not isinstance(gradient, dict):
        gradient = event.get("native_gradient")
    if not isinstance(gradient, dict):
        return {"finite": None, "inf_norm": None}
    finite = gradient.get("all_finite")
    inf_norm = _finite_optional_scalar(gradient.get("inf_norm"))
    if finite is None:
        finite = inf_norm is not None
    return {"finite": bool(finite), "inf_norm": inf_norm}


def _trace_result_evidence(payload: dict[str, Any]) -> dict[str, Any] | None:
    for event in reversed(payload.get("events", [])):
        if not isinstance(event, dict):
            continue
        result = event.get("result")
        if not isinstance(result, dict):
            continue
        if not any(key in result for key in ("message", "status", "ls_status", "task")):
            continue
        return {
            "event_index": event.get("event_index"),
            "label": event.get("label"),
            "message": result.get("message"),
            "status": result.get("status"),
            "ls_status": result.get("ls_status"),
            "nfev": result.get("nfev"),
            "njev": result.get("njev"),
            "success": result.get("success"),
            "task": result.get("task"),
        }
    return None


def _termination_indicates_line_search_failure(
    evidence: dict[str, Any] | None,
) -> bool:
    if evidence is None:
        return False
    message = str(evidence.get("message", "")).upper()
    task = str(evidence.get("task", "")).upper()
    return "ABNORMAL_TERMINATION_IN_LNSRCH" in message or (
        "ABNORMAL_TERMINATION_IN_LNSRCH" in task
    )


def evaluate_phase0_line_search_trace(
    *,
    progress_json: str | Path,
    flat_objective_abs_tol: float = 1.0e-10,
    flat_objective_rel_tol: float = 1.0e-8,
) -> tuple[dict[str, Any], list[str]]:
    """Classify a recorded outer objective-evaluation trace for Phase 0."""
    progress_path = Path(progress_json)
    failures: list[str] = []
    if not progress_path.is_file():
        summary = {
            "progress_json": str(progress_path),
            "classification": "missing_trace",
            "event_count": 0,
            "objective_evaluation_count": 0,
            "passed": False,
        }
        return summary, [f"Missing outer optimizer progress trace: {progress_path}."]

    payload = load_json(progress_path)
    events = _objective_evaluation_events(progress_path)
    objective_values = [_event_objective_value(event) for event in events]
    objective_finite = [_event_objective_finite(event) for event in events]
    gradient_summaries = [_event_optimizer_gradient_summary(event) for event in events]
    gradient_finite = [summary["finite"] for summary in gradient_summaries]
    gradient_inf_norms = [summary["inf_norm"] for summary in gradient_summaries]
    nonfinite_indices = [
        index
        for index, (objective_ok, gradient_ok) in enumerate(
            zip(objective_finite, gradient_finite, strict=True)
        )
        if objective_ok is False or gradient_ok is False
    ]
    missing_gradient_indices = [
        index for index, finite in enumerate(gradient_finite) if finite is None
    ]
    finite_values = [value for value in objective_values if value is not None]
    objective_span = (
        None if not finite_values else float(max(finite_values) - min(finite_values))
    )
    objective_scale = (
        1.0
        if not finite_values
        else max(1.0, max(abs(value) for value in finite_values))
    )
    flat_threshold = (
        float(flat_objective_abs_tol) + float(flat_objective_rel_tol) * objective_scale
    )
    objective_flat = (
        None if objective_span is None else bool(objective_span <= flat_threshold)
    )
    objective_deltas = [
        float(rhs - lhs)
        for lhs, rhs in zip(finite_values, finite_values[1:], strict=False)
    ]
    descent_detected = any(delta < -flat_threshold for delta in objective_deltas)
    termination_evidence = _trace_result_evidence(payload)
    line_search_failure = _termination_indicates_line_search_failure(
        termination_evidence
    )

    if not events:
        classification = "missing_objective_evaluations"
        failures.append("Progress trace contains no objective_evaluation events.")
    elif nonfinite_indices:
        classification = "nonfinite_objective_or_gradient"
        failures.append(
            "Progress trace contains nonfinite objective or optimizer gradient events."
        )
    elif missing_gradient_indices:
        classification = "incomplete_objective_gradient_trace"
        failures.append("Objective-evaluation trace is missing optimizer gradients.")
    elif line_search_failure and objective_flat:
        classification = "finite_plateau_line_search_failure"
    elif line_search_failure:
        classification = "finite_line_search_failure"
    elif objective_flat:
        classification = "finite_flat_plateau"
    elif descent_detected:
        classification = "finite_descent"
    else:
        classification = "finite_unclassified"

    summary = {
        "progress_json": str(progress_path),
        "classification": classification,
        "event_count": int(payload.get("event_count", len(payload.get("events", [])))),
        "objective_evaluation_count": int(len(events)),
        "objective_values": objective_values,
        "objective_finite": objective_finite,
        "optimizer_gradient_finite": gradient_finite,
        "optimizer_gradient_inf_norms": gradient_inf_norms,
        "nonfinite_event_indices": nonfinite_indices,
        "missing_gradient_event_indices": missing_gradient_indices,
        "objective_span": objective_span,
        "objective_flat_threshold": flat_threshold,
        "objective_flat": objective_flat,
        "objective_deltas": objective_deltas,
        "descent_detected": bool(descent_detected),
        "termination_evidence": termination_evidence,
        "line_search_failure_evidence": bool(line_search_failure),
    }
    summary["passed"] = not failures and classification not in (
        "missing_trace",
        "missing_objective_evaluations",
        "incomplete_objective_gradient_trace",
        "nonfinite_objective_or_gradient",
    )
    return summary, failures


def _event_candidate_values(event: dict[str, Any]) -> np.ndarray | None:
    candidate = event.get("candidate_optimizer_dofs")
    if not isinstance(candidate, dict):
        return None
    values = candidate.get("values")
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _event_gradient_values(event: dict[str, Any]) -> np.ndarray | None:
    gradient = event.get("optimizer_gradient")
    if not isinstance(gradient, dict):
        gradient = event.get("native_gradient")
    if not isinstance(gradient, dict):
        return None
    values = gradient.get("values")
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _event_int_field(event: dict[str, Any], key: str) -> int | None:
    value = event.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _event_hardware_margin_abs_min(event: dict[str, Any]) -> float | None:
    hardware_status = event.get("hardware_status")
    if not isinstance(hardware_status, dict):
        return None
    threshold_margins = hardware_status.get("threshold_margins")
    if not isinstance(threshold_margins, dict):
        return None
    margins = [_finite_optional_scalar(value) for value in threshold_margins.values()]
    finite_margins = [value for value in margins if value is not None]
    if not finite_margins:
        return None
    return float(min(abs(value) for value in finite_margins))


def _accepted_step_gradient_record(
    event: dict[str, Any],
) -> dict[str, Any] | None:
    target = _event_int_field(event, "accepted_iteration_target")
    candidate = _event_candidate_values(event)
    gradient = _event_gradient_values(event)
    gradient_summary = _event_optimizer_gradient_summary(event)
    if (
        target is None
        or candidate is None
        or gradient is None
        or _event_objective_finite(event) is False
        or gradient_summary["finite"] is False
        or event.get("native_gradient_used") is not True
    ):
        return None
    return {
        "accepted_iteration_target": target,
        "event_index": _event_int_field(event, "event_index"),
        "line_search_evaluation": _event_int_field(event, "line_search_evaluation"),
        "candidate": candidate,
        "gradient": gradient,
        "hardware_margin_abs_min": _event_hardware_margin_abs_min(event),
    }


def _gradient_inf_norm_delta(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    if lhs.shape != rhs.shape:
        return None
    return float(np.linalg.norm(lhs - rhs, ord=np.inf))


def _gradient_diff_threshold(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    abs_tol: float,
    rel_tol: float,
) -> float:
    scale = max(
        1.0,
        float(np.linalg.norm(lhs, ord=np.inf)),
        float(np.linalg.norm(rhs, ord=np.inf)),
    )
    return float(abs_tol) + float(rel_tol) * scale


def evaluate_phase3_accepted_step_gradient_trace(
    *,
    progress_json: str | Path,
    gradient_abs_tol: float = 1.0e-10,
    gradient_rel_tol: float = 1.0e-8,
    constraint_margin_abs_tol: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Verify accepted-step gradients are point-dependent in a progress trace."""
    progress_path = Path(progress_json)
    if not progress_path.is_file():
        summary = {
            "progress_json": str(progress_path),
            "event_count": 0,
            "objective_evaluation_count": 0,
            "accepted_step_group_count": 0,
            "passed": False,
        }
        return summary, [f"Missing outer optimizer progress trace: {progress_path}."]

    payload = load_json(progress_path)
    events = _objective_evaluation_events(progress_path)
    grouped_events: dict[int, list[dict[str, Any]]] = {}
    missing_target_count = 0
    for event in events:
        target = _event_int_field(event, "accepted_iteration_target")
        if target is None:
            missing_target_count += 1
            continue
        grouped_events.setdefault(target, []).append(event)

    records: list[dict[str, Any]] = []
    missing_replay_grade_targets: list[int] = []
    for target, group_events in grouped_events.items():
        group_record = None
        for event in reversed(group_events):
            group_record = _accepted_step_gradient_record(event)
            if group_record is not None:
                break
        if group_record is None:
            missing_replay_grade_targets.append(target)
        else:
            records.append(group_record)

    comparisons: list[dict[str, Any]] = []
    point_dependent_gradient_evidence = False
    constraint_marginal_evidence: bool | None = (
        None if constraint_margin_abs_tol is None else False
    )
    constraint_marginal_point_dependent_gradient_evidence: bool | None = (
        None if constraint_margin_abs_tol is None else False
    )
    failures: list[str] = []
    if not events:
        failures.append("Progress trace contains no objective_evaluation events.")
    if len(records) < 2:
        failures.append(
            "Phase-3 gradient proof requires at least two replay-grade accepted-step "
            "groups with candidate DOFs and optimizer-gradient values."
        )

    if len(records) >= 2:
        baseline = records[0]
        baseline_candidate = baseline["candidate"]
        baseline_gradient = baseline["gradient"]
        for record in records[1:]:
            candidate_delta_norm = float(
                np.linalg.norm(record["candidate"] - baseline_candidate)
            )
            gradient_delta_inf_norm = _gradient_inf_norm_delta(
                record["gradient"],
                baseline_gradient,
            )
            threshold = _gradient_diff_threshold(
                record["gradient"],
                baseline_gradient,
                abs_tol=gradient_abs_tol,
                rel_tol=gradient_rel_tol,
            )
            gradient_differs = (
                False
                if gradient_delta_inf_norm is None
                else bool(gradient_delta_inf_norm > threshold)
            )
            candidate_moved = bool(candidate_delta_norm > 0.0)
            hardware_margin_abs_min = record["hardware_margin_abs_min"]
            comparison = {
                "accepted_iteration_target": record["accepted_iteration_target"],
                "event_index": record["event_index"],
                "line_search_evaluation": record["line_search_evaluation"],
                "candidate_delta_norm_from_baseline": candidate_delta_norm,
                "gradient_inf_norm_delta_from_baseline": gradient_delta_inf_norm,
                "gradient_diff_threshold": threshold,
                "candidate_moved_from_baseline": candidate_moved,
                "gradient_differs_from_baseline": gradient_differs,
                "hardware_margin_abs_min": hardware_margin_abs_min,
            }
            comparisons.append(comparison)
            point_dependent_gradient_evidence = bool(
                point_dependent_gradient_evidence
                or (candidate_moved and gradient_differs)
            )
            if (
                constraint_margin_abs_tol is not None
                and hardware_margin_abs_min is not None
                and hardware_margin_abs_min <= constraint_margin_abs_tol
            ):
                constraint_marginal_evidence = True
                if candidate_moved and gradient_differs:
                    constraint_marginal_point_dependent_gradient_evidence = True
        if not any(item["candidate_moved_from_baseline"] for item in comparisons):
            failures.append(
                "Phase-3 gradient proof found no accepted step that moved from the "
                "baseline candidate."
            )
        if not point_dependent_gradient_evidence:
            failures.append(
                "Phase-3 gradient proof accepted-step gradients match the frozen "
                "baseline gradient within tolerance."
            )
        if (
            constraint_margin_abs_tol is not None
            and not constraint_marginal_point_dependent_gradient_evidence
        ):
            failures.append(
                "Phase-3 gradient proof did not observe a point-dependent accepted "
                "gradient at the requested hardware constraint margin."
            )

    summary = {
        "progress_json": str(progress_path),
        "event_count": int(payload.get("event_count", len(payload.get("events", [])))),
        "objective_evaluation_count": int(len(events)),
        "accepted_step_group_count": int(len(grouped_events)),
        "replay_grade_group_count": int(len(records)),
        "missing_accepted_iteration_target_count": int(missing_target_count),
        "missing_replay_grade_targets": missing_replay_grade_targets,
        "gradient_abs_tol": float(gradient_abs_tol),
        "gradient_rel_tol": float(gradient_rel_tol),
        "constraint_margin_abs_tol": (
            None
            if constraint_margin_abs_tol is None
            else float(constraint_margin_abs_tol)
        ),
        "baseline_accepted_iteration_target": (
            None if not records else records[0]["accepted_iteration_target"]
        ),
        "comparisons": comparisons,
        "point_dependent_gradient_evidence": bool(point_dependent_gradient_evidence),
        "constraint_marginal_evidence": constraint_marginal_evidence,
        "constraint_marginal_point_dependent_gradient_evidence": (
            constraint_marginal_point_dependent_gradient_evidence
        ),
    }
    summary["passed"] = not failures
    return summary, failures


def _event_boozer_newton_tol(event: dict[str, Any]) -> float | None:
    metadata = event.get("boozer_solver_metadata")
    if not isinstance(metadata, dict):
        return None
    return _finite_optional_scalar(metadata.get("newton_tol"))


def _newton_tol_matches_recorded(
    recorded_newton_tol: float | None,
    requested_newton_tol: float,
) -> bool | None:
    if recorded_newton_tol is None:
        return None
    return bool(
        np.isclose(
            recorded_newton_tol,
            float(requested_newton_tol),
            rtol=1.0e-12,
            atol=0.0,
        )
    )


def evaluate_phase0_noise_calibration_pair(
    *,
    baseline_progress_json: str | Path,
    tightened_progress_json: str | Path,
    baseline_newton_tol: float,
    tightened_newton_tol: float,
    objective_evaluation_index: int = 0,
) -> tuple[dict[str, Any], list[str]]:
    """Compare two fixed-candidate traces to estimate the seed's J noise."""
    baseline_events, baseline_failures = _objective_evaluation_events_with_failures(
        baseline_progress_json,
        context="Phase-0 noise calibration baseline trace",
    )
    tightened_events, tightened_failures = _objective_evaluation_events_with_failures(
        tightened_progress_json,
        context="Phase-0 noise calibration tightened trace",
    )
    failures = [*baseline_failures, *tightened_failures]
    index = int(objective_evaluation_index)
    if index < 0:
        failures.append(
            "Noise calibration objective_evaluation_index must be non-negative."
        )
        baseline_event = None
        tightened_event = None
    else:
        baseline_event = (
            baseline_events[index] if len(baseline_events) > index else None
        )
        tightened_event = (
            tightened_events[index] if len(tightened_events) > index else None
        )
    baseline_objective = (
        None if baseline_event is None else _event_objective_value(baseline_event)
    )
    tightened_objective = (
        None if tightened_event is None else _event_objective_value(tightened_event)
    )
    baseline_candidate = (
        None if baseline_event is None else _event_candidate_values(baseline_event)
    )
    tightened_candidate = (
        None if tightened_event is None else _event_candidate_values(tightened_event)
    )
    baseline_recorded_newton_tol = (
        None if baseline_event is None else _event_boozer_newton_tol(baseline_event)
    )
    tightened_recorded_newton_tol = (
        None if tightened_event is None else _event_boozer_newton_tol(tightened_event)
    )
    baseline_newton_tol_matches = _newton_tol_matches_recorded(
        baseline_recorded_newton_tol,
        baseline_newton_tol,
    )
    tightened_newton_tol_matches = _newton_tol_matches_recorded(
        tightened_recorded_newton_tol,
        tightened_newton_tol,
    )
    candidate_match = (
        None
        if baseline_candidate is None or tightened_candidate is None
        else bool(np.array_equal(baseline_candidate, tightened_candidate))
    )
    if baseline_event is None or tightened_event is None:
        failures.append(
            "Noise calibration requires the selected objective_evaluation event "
            "in both traces."
        )
    if baseline_objective is None or tightened_objective is None:
        failures.append("Noise calibration requires finite objective values.")
    if candidate_match is None:
        failures.append("Noise calibration requires replay-grade candidate values.")
    elif not candidate_match:
        failures.append("Noise calibration traces do not hold candidate DOFs fixed.")
    if baseline_newton_tol_matches is None:
        failures.append("Baseline trace is missing boozer_solver_metadata.newton_tol.")
    elif not baseline_newton_tol_matches:
        failures.append("Baseline trace newton_tol does not match the requested value.")
    if tightened_newton_tol_matches is None:
        failures.append("Tightened trace is missing boozer_solver_metadata.newton_tol.")
    elif not tightened_newton_tol_matches:
        failures.append(
            "Tightened trace newton_tol does not match the requested value."
        )
    objective_abs_delta = (
        None
        if baseline_objective is None or tightened_objective is None
        else float(abs(tightened_objective - baseline_objective))
    )
    objective_rel_delta = (
        None
        if objective_abs_delta is None or baseline_objective is None
        else float(objective_abs_delta / max(abs(baseline_objective), 1.0e-30))
    )
    summary = {
        "baseline_progress_json": str(baseline_progress_json),
        "tightened_progress_json": str(tightened_progress_json),
        "baseline_newton_tol": float(baseline_newton_tol),
        "tightened_newton_tol": float(tightened_newton_tol),
        "baseline_recorded_newton_tol": baseline_recorded_newton_tol,
        "tightened_recorded_newton_tol": tightened_recorded_newton_tol,
        "baseline_newton_tol_matches": baseline_newton_tol_matches,
        "tightened_newton_tol_matches": tightened_newton_tol_matches,
        "objective_evaluation_index": index,
        "baseline_objective": baseline_objective,
        "tightened_objective": tightened_objective,
        "objective_abs_delta": objective_abs_delta,
        "objective_rel_delta": objective_rel_delta,
        "candidate_dofs_match": candidate_match,
        "baseline_objective_evaluation_count": int(len(baseline_events)),
        "tightened_objective_evaluation_count": int(len(tightened_events)),
        "passed": not failures,
    }
    return summary, failures


def _memory_snapshot_from_event(event: dict[str, Any]) -> dict[str, float | None]:
    snapshot = event.get("memory_snapshot")
    if not isinstance(snapshot, dict):
        return {"rss_mb": None, "gpu_memory_mb": None}
    rss_mb = _finite_optional_scalar(snapshot.get("rss_mb"))
    gpu_memory = snapshot.get("gpu_memory_mb")
    gpu_memory_mb = None if gpu_memory is None else _finite_optional_scalar(gpu_memory)
    return {"rss_mb": rss_mb, "gpu_memory_mb": gpu_memory_mb}


def _compile_snapshot_from_event(event: dict[str, Any]) -> dict[str, int | None]:
    snapshot = event.get("compile_diagnostics_snapshot")
    if not isinstance(snapshot, dict):
        return {"compile_event_count": None, "cache_miss_count": None}
    compile_events = snapshot.get("compile_event_count")
    cache_misses = snapshot.get("cache_miss_count")
    return {
        "compile_event_count": None if compile_events is None else int(compile_events),
        "cache_miss_count": None if cache_misses is None else int(cache_misses),
    }


def _steady_peak_growth(
    values: list[float | None],
) -> tuple[float | None, float | None]:
    if len(values) < 2 or values[0] is None or any(value is None for value in values):
        return None, None
    baseline = float(values[0])
    peak = max(float(value) for value in values if value is not None)
    return peak, float(peak - baseline)


def _steady_counter_growth(values: list[int | None]) -> tuple[int | None, int | None]:
    if len(values) < 2 or values[0] is None or any(value is None for value in values):
        return None, None
    baseline = int(values[0])
    peak = max(int(value) for value in values if value is not None)
    return peak, int(peak - baseline)


def evaluate_host_jax_compile_gate(
    *,
    progress_json: str | Path,
    warmup_evaluations: int,
) -> tuple[dict[str, Any], list[str]]:
    events, failures = _objective_evaluation_events_with_failures(
        progress_json,
        context="Host-jax compile gate",
    )
    snapshots = [_compile_snapshot_from_event(event) for event in events]
    warmup_count = int(warmup_evaluations)
    steady_snapshots = snapshots[warmup_count:]
    compile_counts = [snapshot["compile_event_count"] for snapshot in steady_snapshots]
    cache_miss_counts = [snapshot["cache_miss_count"] for snapshot in steady_snapshots]
    peak_compile_count, compile_count_growth = _steady_counter_growth(compile_counts)
    peak_cache_miss_count, cache_miss_count_growth = _steady_counter_growth(
        cache_miss_counts
    )
    summary: dict[str, Any] = {
        "progress_json": str(progress_json),
        "event_count": int(len(events)),
        "snapshot_count": int(len(snapshots)),
        "warmup_evaluations": warmup_count,
        "steady_snapshot_count": int(len(steady_snapshots)),
        "peak_steady_compile_event_count": peak_compile_count,
        "peak_steady_cache_miss_count": peak_cache_miss_count,
        "steady_compile_event_count_growth": compile_count_growth,
        "steady_cache_miss_count_growth": cache_miss_count_growth,
        "first_steady_snapshot": None if not steady_snapshots else steady_snapshots[0],
        "last_steady_snapshot": None if not steady_snapshots else steady_snapshots[-1],
    }
    if len(steady_snapshots) < 2:
        failures.append(
            "Host-jax compile gate requires at least two post-warm objective "
            "evaluation compile snapshots."
        )
    if any(value is None for value in compile_counts):
        failures.append("Host-jax compile gate found missing compile counters.")
    if any(value is None for value in cache_miss_counts):
        failures.append("Host-jax compile gate found missing cache-miss counters.")
    if compile_count_growth is not None and compile_count_growth > 0:
        failures.append(
            "Host-jax compile gate recorded post-warm compile event growth: "
            f"{compile_count_growth}."
        )
    if cache_miss_count_growth is not None and cache_miss_count_growth > 0:
        failures.append(
            "Host-jax compile gate recorded post-warm cache-miss growth: "
            f"{cache_miss_count_growth}."
        )
    summary["passed"] = not failures
    return summary, failures


def evaluate_host_jax_memory_gate(
    *,
    progress_json: str | Path,
    warmup_evaluations: int,
    max_steady_rss_growth_mb: float,
    max_steady_gpu_growth_mb: float,
    require_gpu_memory: bool,
) -> tuple[dict[str, Any], list[str]]:
    events, failures = _objective_evaluation_events_with_failures(
        progress_json,
        context="Host-jax memory gate",
    )
    snapshots = [_memory_snapshot_from_event(event) for event in events]
    warmup_count = int(warmup_evaluations)
    steady_snapshots = snapshots[warmup_count:]
    rss_values = [snapshot["rss_mb"] for snapshot in steady_snapshots]
    gpu_values = [snapshot["gpu_memory_mb"] for snapshot in steady_snapshots]
    peak_rss_mb, rss_growth_mb = _steady_peak_growth(rss_values)
    peak_gpu_memory_mb, gpu_growth_mb = _steady_peak_growth(gpu_values)
    summary: dict[str, Any] = {
        "progress_json": str(progress_json),
        "event_count": int(len(events)),
        "snapshot_count": int(len(snapshots)),
        "warmup_evaluations": warmup_count,
        "steady_snapshot_count": int(len(steady_snapshots)),
        "peak_steady_rss_mb": peak_rss_mb,
        "peak_steady_gpu_memory_mb": peak_gpu_memory_mb,
        "peak_steady_rss_growth_mb": rss_growth_mb,
        "peak_steady_gpu_memory_growth_mb": gpu_growth_mb,
        "max_steady_rss_growth_mb": float(max_steady_rss_growth_mb),
        "max_steady_gpu_growth_mb": float(max_steady_gpu_growth_mb),
        "require_gpu_memory": bool(require_gpu_memory),
        "first_steady_snapshot": None if not steady_snapshots else steady_snapshots[0],
        "last_steady_snapshot": None if not steady_snapshots else steady_snapshots[-1],
    }
    if len(steady_snapshots) < 2:
        failures.append(
            "Host-jax memory gate requires at least two post-warm objective "
            "evaluation snapshots."
        )
    if any(value is None for value in rss_values):
        failures.append("Host-jax memory gate found missing host RSS snapshots.")
    if require_gpu_memory and any(value is None for value in gpu_values):
        failures.append("Host-jax memory gate found missing GPU memory snapshots.")
    if rss_growth_mb is not None and rss_growth_mb > max_steady_rss_growth_mb:
        failures.append(
            "Host-jax memory gate exceeded post-warm peak RSS growth tolerance: "
            f"{rss_growth_mb:.2f} MB > {max_steady_rss_growth_mb:.2f} MB."
        )
    if gpu_growth_mb is not None and gpu_growth_mb > max_steady_gpu_growth_mb:
        failures.append(
            "Host-jax memory gate exceeded post-warm peak GPU memory growth tolerance: "
            f"{gpu_growth_mb:.2f} MB > {max_steady_gpu_growth_mb:.2f} MB."
        )
    summary["passed"] = not failures
    return summary, failures


def _diagnostic_trace_region_for_phase(phase: object) -> str | None:
    if phase == "phase1":
        return _PHASE1_INITIAL_TRACE_REGION
    if phase == "phase2":
        return _PHASE2_TRACE_REGION
    return None


def _phase1_compile_behavior(
    *,
    compile_diagnostics_requested: bool,
    compile_diagnostics_enabled: bool,
    compile_diagnostics_disable_reason: str | None,
) -> dict[str, Any]:
    diagnostics_enabled = bool(compile_diagnostics_enabled)
    return {
        "diagnostics_requested": bool(compile_diagnostics_requested),
        "diagnostics_enabled": diagnostics_enabled,
        "jax_log_compiles": diagnostics_enabled,
        "jax_explain_cache_misses": diagnostics_enabled,
        "cache_reuse_evidence_valid": diagnostics_enabled,
        "disabled_reason": compile_diagnostics_disable_reason,
    }


def _result_int_or_default(
    results: dict[str, object],
    key: str,
    default: int,
) -> int:
    value = results.get(key)
    if value is None:
        return default
    return int(value)


def build_phase1_diagnostic_note(
    results: dict[str, Any],
    *,
    failures: list[str],
    compile_diagnostics_requested: bool,
    compile_diagnostics_enabled: bool,
    compile_diagnostics_disable_reason: str | None,
    deterministic_gpu_reductions: bool,
) -> dict[str, Any]:
    note: dict[str, Any] = {
        "reproduced": bool(failures),
        "trace_dir": results.get("JAX_PROFILE_DIR"),
        "termination_message": results.get("TERMINATION_MESSAGE"),
        "initial_phase_iterations": _result_int_or_default(
            results,
            "INITIAL_PHASE_ITERATIONS",
            0,
        ),
        "total_iterations": _result_int_or_default(results, "iterations", 0),
        "target_lane_profile_recorded": "TARGET_LANE_PROFILE" in results,
        "compile_behavior": _phase1_compile_behavior(
            compile_diagnostics_requested=compile_diagnostics_requested,
            compile_diagnostics_enabled=compile_diagnostics_enabled,
            compile_diagnostics_disable_reason=compile_diagnostics_disable_reason,
        ),
        "deterministic_gpu_reductions": bool(deterministic_gpu_reductions),
        "first_bad_region": None,
        "first_bad_region_source": None,
        "first_bad_region_detail": None,
    }
    scaled_phase1 = results.get("TARGET_LANE_SCALED_PHASE1_DIAGNOSIS")
    if isinstance(scaled_phase1, dict):
        first_nonfinite_stage = scaled_phase1.get("first_nonfinite_stage")
        if isinstance(first_nonfinite_stage, str):
            note["first_bad_region"] = _PHASE1_INITIAL_TRACE_REGION
            note["first_bad_region_source"] = (
                "TARGET_LANE_SCALED_PHASE1_DIAGNOSIS.first_nonfinite_stage"
            )
            note["first_bad_region_detail"] = first_nonfinite_stage
            return note
    invalid_state = results.get("TARGET_LANE_INVALID_STATE_DIAGNOSIS")
    if isinstance(invalid_state, dict):
        events = invalid_state.get("events")
        if isinstance(events, list) and events:
            first_event = events[0]
            phase = first_event.get("phase")
            note["first_bad_region"] = _diagnostic_trace_region_for_phase(phase) or str(
                phase
            )
            note["first_bad_region_source"] = (
                "TARGET_LANE_INVALID_STATE_DIAGNOSIS.events[0]"
            )
            note["first_bad_region_detail"] = {
                "phase": phase,
                "iteration": first_event.get("iteration"),
                "line_search_failed": first_event.get("line_search_failed"),
                "nonfinite_step": first_event.get("nonfinite_step"),
                "stalled_step": first_event.get("stalled_step"),
                "valid_curvature": first_event.get("valid_curvature"),
                "ls_status": first_event.get("ls_status"),
            }
            return note
    if failures:
        note["first_bad_region"] = _PHASE2_TRACE_REGION
        note["first_bad_region_source"] = "probe_failures[0]"
        note["first_bad_region_detail"] = failures[0]
    return note


def evaluate_single_stage_outer_loop_probe(
    results: dict[str, Any],
    *,
    expected_outer_optimizer_method: str = TARGET_OUTER_OPTIMIZER_METHOD,
    expected_boozer_optimizer_backend: str | None = None,
    expected_boozer_optimizer_method: str | None = None,
    require_accepted_step: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    summary = {
        "rung": LADDER_RUNG,
        "iterations": _result_int_or_default(results, "iterations", 0),
        "boozer_optimizer_backend": results.get("boozer_optimizer_backend"),
        "boozer_optimizer_method": results.get("boozer_optimizer_method"),
        "outer_optimizer_method": str(results.get("outer_optimizer_method", "")),
        "self_intersecting": bool(results.get("SELF_INTERSECTING", False)),
        "self_intersection_check_available": bool(
            results.get("SELF_INTERSECTION_CHECK_AVAILABLE", True)
        ),
        "finite_result_keys": _finite_result_keys(results),
        "initial_objective": _finite_optional_scalar(results.get("INITIAL_OBJECTIVE")),
        "final_objective": _finite_optional_scalar(results.get("FINAL_OBJECTIVE")),
    }
    summary["objective_decrease"] = (
        None
        if summary["initial_objective"] is None or summary["final_objective"] is None
        else float(summary["initial_objective"] - summary["final_objective"])
    )
    summary["objective_decreased"] = (
        None
        if summary["objective_decrease"] is None
        else bool(summary["objective_decrease"] > 0.0)
    )

    failures: list[str] = []
    if require_accepted_step and summary["iterations"] < _MIN_ACCEPTED_ITERATIONS:
        failures.append(
            "Single-stage outer-loop probe did not complete the required "
            f"{_MIN_ACCEPTED_ITERATIONS} accepted optimizer iterations."
        )
    if summary["outer_optimizer_method"] != expected_outer_optimizer_method:
        failures.append(
            "Single-stage outer-loop probe did not use the target "
            f"{expected_outer_optimizer_method} method."
        )
    if (
        expected_boozer_optimizer_backend is not None
        and summary["boozer_optimizer_backend"] != expected_boozer_optimizer_backend
    ):
        failures.append(
            "Single-stage outer-loop probe did not use the requested inner "
            f"Boozer backend {expected_boozer_optimizer_backend!r}."
        )
    if (
        expected_boozer_optimizer_method is not None
        and summary["boozer_optimizer_method"] != expected_boozer_optimizer_method
    ):
        failures.append(
            "Single-stage outer-loop probe did not use the requested inner "
            f"Boozer optimizer method {expected_boozer_optimizer_method!r}."
        )
    if summary["self_intersecting"]:
        failures.append(
            "Single-stage outer-loop probe produced a self-intersecting surface."
        )
    if require_accepted_step and _REQUIRE_OBJECTIVE_DECREASE:
        if summary["initial_objective"] is None:
            failures.append(
                "Single-stage outer-loop probe did not report a finite initial objective."
            )
        elif summary["final_objective"] is None:
            failures.append(
                "Single-stage outer-loop probe did not report a finite final objective."
            )
        elif not summary["objective_decreased"]:
            failures.append(
                "Single-stage outer-loop probe did not decrease the objective."
            )
    for key, is_finite in summary["finite_result_keys"].items():
        if not is_finite:
            failures.append(
                f"Single-stage outer-loop probe produced a non-finite {key}."
            )
    return summary, failures


def _phase0_trace_from_partial_artifacts(
    artifacts: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    progress_paths = artifacts.get("outer_optimizer_progress_json")
    if not isinstance(progress_paths, list) or len(progress_paths) != 1:
        return None, []
    return evaluate_phase0_line_search_trace(progress_json=progress_paths[0])


def _phase0_noise_baseline_progress_json(args: argparse.Namespace) -> Path:
    baseline = (
        args.phase0_noise_baseline_progress_json
        if args.phase0_noise_baseline_progress_json is not None
        else args.replay_objective_evaluation_trace
    )
    if baseline is None:
        raise ValueError(
            "--enable-phase0-noise-calibration-gate requires either "
            "--phase0-noise-baseline-progress-json or "
            "--replay-objective-evaluation-trace."
        )
    return Path(baseline)


def _phase0_noise_replay_progress_json(args: argparse.Namespace) -> Path:
    if args.replay_objective_evaluation_trace is not None:
        return Path(args.replay_objective_evaluation_trace)
    return _phase0_noise_baseline_progress_json(args)


def write_outer_loop_case_execution_failure_json(
    output_json: str | Path,
    *,
    provenance: dict[str, Any],
    case_output_root: Path,
    error: RuntimeError,
) -> None:
    """Write a structured failed probe payload with partial child artifacts."""
    artifacts = _collect_partial_single_stage_case_artifacts(case_output_root)
    phase0_line_search_trace, phase0_trace_failures = (
        _phase0_trace_from_partial_artifacts(artifacts)
    )
    failure = f"Single-stage outer-loop child failed: {type(error).__name__}: {error}"
    payload: dict[str, Any] = {
        "rung": LADDER_RUNG,
        "provenance": provenance,
        "status": "case-execution-failed",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "artifacts": artifacts,
        "failures": [failure, *phase0_trace_failures],
        "passed": False,
    }
    if phase0_line_search_trace is not None:
        payload["phase0_line_search_trace"] = phase0_line_search_trace
    write_json(output_json, payload)


def main() -> None:
    args = parse_args()
    args.disable_target_lane_success_filter = True
    if args.enable_host_jax_memory_gate:
        if args.optimizer_backend != "host-jax":
            raise ValueError(
                "--enable-host-jax-memory-gate requires --optimizer-backend host-jax."
            )
        args.record_objective_evaluation_trace = True
    if args.enable_phase0_noise_calibration_gate:
        args.replay_objective_evaluation_trace = str(
            _phase0_noise_replay_progress_json(args)
        )
        args.record_objective_evaluation_trace = True
    if args.enable_phase3_gradient_proof_gate:
        args.record_objective_evaluation_trace = True
    (
        compile_diagnostics_enabled,
        compile_diagnostics_disable_reason,
    ) = resolve_target_lane_compile_diagnostics(
        enable_compile_diagnostics=args.enable_compile_diagnostics,
        diagnose_target_lane_scaled_phase1=args.diagnose_target_lane_scaled_phase1,
        record_target_lane_invalid_state_events=(
            args.record_target_lane_invalid_state_events
        ),
    )
    host_jax_compile_gate_enabled = bool(
        args.optimizer_backend == "host-jax" and compile_diagnostics_enabled
    )
    if host_jax_compile_gate_enabled:
        args.record_objective_evaluation_trace = True
    bootstrap_local_simsopt()
    resolved_boozer_optimizer_backend = resolve_boozer_optimizer_backend(
        args.optimizer_backend,
        args.boozer_optimizer_backend,
    )
    expected_outer_optimizer_method = _expected_outer_optimizer_method(
        args.optimizer_backend
    )
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Single-stage outer-loop probe",
        extra={
            "lane": resolve_probe_lane(optimizer_backend=args.optimizer_backend),
            "ladder_rung": LADDER_RUNG,
            "fixture": "real-single-stage-init",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": str(Path(args.stage2_bs_path)),
            "jax_runtime_seed_spec": (
                None
                if args.jax_runtime_seed_spec is None
                else str(Path(args.jax_runtime_seed_spec))
            ),
            "reuse_jax_runtime_seed_solve": bool(args.reuse_jax_runtime_seed_solve),
            "optimizer_backend": args.optimizer_backend,
            "boozer_optimizer_backend": resolved_boozer_optimizer_backend,
            "boozer_optimizer_backend_requested": args.boozer_optimizer_backend,
            "boozer_least_squares_algorithm": args.boozer_least_squares_algorithm,
            "target_lane_boozer_newton_tol": args.target_lane_boozer_newton_tol,
            "target_lane_boozer_newton_maxiter": (
                args.target_lane_boozer_newton_maxiter
            ),
            "outer_maxiter": int(args.maxiter),
            "outer_maxls": int(args.outer_maxls),
            "single_stage_case_timeout_seconds": float(
                args.single_stage_case_timeout_seconds
            ),
            "profile_target_lane_batch_size": int(args.profile_target_lane_batch_size),
            "diagnose_target_lane_scaled_phase1": bool(
                args.diagnose_target_lane_scaled_phase1
            ),
            "record_target_lane_invalid_state_events": bool(
                args.record_target_lane_invalid_state_events
            ),
            "record_objective_evaluation_trace": bool(
                args.record_objective_evaluation_trace
            ),
            "replay_objective_evaluation_trace": args.replay_objective_evaluation_trace,
            "phase0_noise_calibration_gate_enabled": bool(
                args.enable_phase0_noise_calibration_gate
            ),
            "phase0_noise_baseline_progress_json": (
                args.phase0_noise_baseline_progress_json
            ),
            "phase0_noise_baseline_newton_tol": float(
                args.phase0_noise_baseline_newton_tol
            ),
            "phase0_noise_tightened_newton_tol": float(
                args.phase0_noise_tightened_newton_tol
            ),
            "phase0_noise_objective_evaluation_index": int(
                args.phase0_noise_objective_evaluation_index
            ),
            "phase3_gradient_proof_gate_enabled": bool(
                args.enable_phase3_gradient_proof_gate
            ),
            "phase3_constraint_margin_abs_tol": args.phase3_constraint_margin_abs_tol,
            "host_jax_memory_gate_requested": bool(args.enable_host_jax_memory_gate),
            "host_jax_compile_gate_enabled": bool(host_jax_compile_gate_enabled),
            "compile_diagnostics_requested": bool(args.enable_compile_diagnostics),
            "compile_diagnostics_enabled": bool(compile_diagnostics_enabled),
            "compile_diagnostics_disable_reason": compile_diagnostics_disable_reason,
            "deterministic_gpu_reductions": bool(args.deterministic_gpu_reductions),
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "compile_behavior": describe_compile_behavior(uses_subprocesses=True),
        },
    )
    print_provenance(provenance)

    case_output_root = Path(args.output_json).parent / "single_stage_case_outputs"
    case_output_root.mkdir(parents=True, exist_ok=True)
    try:
        case = _run_single_stage_case(
            args,
            "jax",
            platform=args.platform,
            benchmark_mode=True,
            load_surface_gamma=False,
            profile_target_lane=args.profile_target_lane,
            profile_target_lane_only=args.profile_target_lane_only,
            diagnose_target_lane_scaled_phase1=args.diagnose_target_lane_scaled_phase1,
            record_target_lane_invalid_state_events=(
                args.record_target_lane_invalid_state_events
            ),
            experimental_target_lane_value_and_grad=(
                args.experimental_target_lane_value_and_grad
            ),
            enable_compile_diagnostics=compile_diagnostics_enabled,
            deterministic_gpu_reductions=args.deterministic_gpu_reductions,
            jax_runtime_seed_spec=(
                None
                if args.jax_runtime_seed_spec is None
                else Path(args.jax_runtime_seed_spec)
            ),
            replay_objective_evaluation_trace=(
                None
                if args.replay_objective_evaluation_trace is None
                else Path(args.replay_objective_evaluation_trace)
            ),
            output_root=case_output_root,
        )
    except RuntimeError as exc:
        write_outer_loop_case_execution_failure_json(
            args.output_json,
            provenance=provenance,
            case_output_root=case_output_root,
            error=exc,
        )
        print("SINGLE-STAGE OUTER-LOOP PROBE FAILED")
        print(f"  - Single-stage outer-loop child failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    replay_measurement_run = args.replay_objective_evaluation_trace is not None
    summary, failures = evaluate_single_stage_outer_loop_probe(
        case["results"],
        expected_outer_optimizer_method=expected_outer_optimizer_method,
        expected_boozer_optimizer_backend=resolved_boozer_optimizer_backend,
        expected_boozer_optimizer_method=resolve_boozer_optimizer_method(
            resolved_boozer_optimizer_backend,
            least_squares_algorithm=resolve_boozer_least_squares_algorithm(
                resolved_boozer_optimizer_backend,
                args.boozer_least_squares_algorithm,
            ),
        ),
        require_accepted_step=not (
            args.profile_target_lane_only
            or args.diagnose_target_lane_scaled_phase1
            or replay_measurement_run
        ),
    )
    memory_gate = None
    compile_gate = None
    phase0_noise_calibration = None
    phase3_gradient_trace = None
    if args.enable_host_jax_memory_gate:
        memory_gate, memory_gate_failures = evaluate_host_jax_memory_gate(
            progress_json=case["outer_optimizer_progress_json"],
            warmup_evaluations=args.host_jax_memory_warmup_evaluations,
            max_steady_rss_growth_mb=args.max_host_jax_steady_rss_growth_mb,
            max_steady_gpu_growth_mb=args.max_host_jax_steady_gpu_growth_mb,
            require_gpu_memory=args.platform == "cuda",
        )
        failures.extend(memory_gate_failures)
    if host_jax_compile_gate_enabled:
        compile_gate, compile_gate_failures = evaluate_host_jax_compile_gate(
            progress_json=case["outer_optimizer_progress_json"],
            warmup_evaluations=args.host_jax_memory_warmup_evaluations,
        )
        failures.extend(compile_gate_failures)
    phase0_line_search_trace = None
    if args.record_objective_evaluation_trace:
        phase0_line_search_trace, phase0_trace_failures = (
            evaluate_phase0_line_search_trace(
                progress_json=case["outer_optimizer_progress_json"]
            )
        )
        failures.extend(phase0_trace_failures)
    if args.enable_phase0_noise_calibration_gate:
        phase0_noise_calibration, phase0_noise_failures = (
            evaluate_phase0_noise_calibration_pair(
                baseline_progress_json=_phase0_noise_baseline_progress_json(args),
                tightened_progress_json=case["outer_optimizer_progress_json"],
                baseline_newton_tol=args.phase0_noise_baseline_newton_tol,
                tightened_newton_tol=args.phase0_noise_tightened_newton_tol,
                objective_evaluation_index=(
                    args.phase0_noise_objective_evaluation_index
                ),
            )
        )
        failures.extend(phase0_noise_failures)
    if args.enable_phase3_gradient_proof_gate:
        phase3_gradient_trace, phase3_gradient_failures = (
            evaluate_phase3_accepted_step_gradient_trace(
                progress_json=case["outer_optimizer_progress_json"],
                constraint_margin_abs_tol=args.phase3_constraint_margin_abs_tol,
            )
        )
        failures.extend(phase3_gradient_failures)
    measurement_passed = not failures
    convergence_passed = bool(measurement_passed and not replay_measurement_run)
    payload = {
        "rung": LADDER_RUNG,
        "provenance": provenance,
        "status": (
            "replay-measurement-passed"
            if replay_measurement_run and measurement_passed
            else "replay-measurement-failed"
            if replay_measurement_run
            else "convergence-probe-passed"
            if convergence_passed
            else "convergence-probe-failed"
        ),
        "results": case["results"],
        "probe": summary,
        "timings": {
            "jax_elapsed_s": float(case["elapsed_s"]),
            "jax_outer_elapsed_s": float(case["elapsed_s"]),
            **_prefix_phase_timings("jax", case["phase_timings"]),
        },
        "failures": failures,
        "measurement_passed": bool(measurement_passed),
        "convergence_passed": convergence_passed,
        "passed": convergence_passed,
        "replay_measurement_run": bool(replay_measurement_run),
    }
    if memory_gate is not None:
        payload["host_jax_memory_gate"] = memory_gate
    if compile_gate is not None:
        payload["host_jax_compile_gate"] = compile_gate
    if phase0_line_search_trace is not None:
        payload["phase0_line_search_trace"] = phase0_line_search_trace
    if phase0_noise_calibration is not None:
        payload["phase0_noise_calibration"] = phase0_noise_calibration
    if phase3_gradient_trace is not None:
        payload["phase3_accepted_step_gradient_trace"] = phase3_gradient_trace
    if "TARGET_LANE_PROFILE" in case["results"]:
        payload["target_lane_profile"] = case["results"]["TARGET_LANE_PROFILE"]
    if "JAX_COMPILE_DIAGNOSTICS" in case["results"]:
        payload["compile_diagnostics"] = case["results"]["JAX_COMPILE_DIAGNOSTICS"]
    if (
        args.jax_profile_dir
        or args.enable_compile_diagnostics
        or args.deterministic_gpu_reductions
        or args.enable_host_jax_memory_gate
        or args.record_target_lane_invalid_state_events
        or args.diagnose_target_lane_scaled_phase1
        or "TARGET_LANE_INVALID_STATE_DIAGNOSIS" in case["results"]
        or "TARGET_LANE_SCALED_PHASE1_DIAGNOSIS" in case["results"]
    ):
        payload["phase1_diagnostic_note"] = build_phase1_diagnostic_note(
            case["results"],
            failures=failures,
            compile_diagnostics_requested=args.enable_compile_diagnostics,
            compile_diagnostics_enabled=compile_diagnostics_enabled,
            compile_diagnostics_disable_reason=compile_diagnostics_disable_reason,
            deterministic_gpu_reductions=args.deterministic_gpu_reductions,
        )
    write_json(args.output_json, payload)
    if failures:
        print("SINGLE-STAGE OUTER-LOOP PROBE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    if replay_measurement_run:
        print("SINGLE-STAGE OUTER-LOOP PROBE MEASUREMENT PASSED")
        return
    print("SINGLE-STAGE OUTER-LOOP PROBE PASSED")


if __name__ == "__main__":
    main()
