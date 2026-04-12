from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_STAGE_SCRIPT = SCRIPT_DIR / "single_stage_banana_example.py"
DEFAULT_CONTINUATION_OUTPUT_ROOT = SCRIPT_DIR / "continuation_outputs"


@dataclass(frozen=True)
class ContinuationStage:
    name: str
    mpol: int
    ntor: int
    nphi: int
    ntheta: int
    maxiter: int
    minimal_artifacts: bool = False
    outer_maxls: int | None = None
    maxcor: int | None = None
    initial_step_scale: float | None = None
    initial_step_maxiter: int | None = None
    target_lane_boozer_bfgs_tol: float | None = None
    target_lane_boozer_bfgs_maxiter: int | None = None


@dataclass(frozen=True)
class ContinuationRunOutcome:
    run_root: Path
    summary_path: Path | None
    summary: dict[str, object] | None
    report_path: Path | None
    report: dict[str, object] | None
    exit_code: int


CONTINUATION_TRIAL_POLICY_NONE = "none"
CONTINUATION_TRIAL_POLICY_VALIDATED_FAST = "validated-fast"
CONTINUATION_TRIAL_POLICY_CHOICES = (
    CONTINUATION_TRIAL_POLICY_NONE,
    CONTINUATION_TRIAL_POLICY_VALIDATED_FAST,
)

_VALIDATED_FAST_TRIAL_STAGE_OVERRIDES = {
    "coarse": {
        "minimal_artifacts": True,
        "outer_maxls": 2,
        "maxcor": 10,
        "initial_step_scale": 0.1,
        "initial_step_maxiter": 0,
        "target_lane_boozer_bfgs_tol": 1e-5,
        "target_lane_boozer_bfgs_maxiter": 24,
    },
    "medium": {
        "minimal_artifacts": True,
        "outer_maxls": 4,
        "maxcor": 12,
        "initial_step_scale": 0.25,
        "initial_step_maxiter": 1,
        "target_lane_boozer_bfgs_tol": 3e-6,
        "target_lane_boozer_bfgs_maxiter": 32,
    },
    "prefinal": {
        "minimal_artifacts": True,
        "outer_maxls": 6,
        "maxcor": 16,
        "initial_step_scale": 0.5,
        "initial_step_maxiter": 1,
        "target_lane_boozer_bfgs_tol": 1e-6,
        "target_lane_boozer_bfgs_maxiter": 48,
    },
    "final": {},
}


_OVERRIDDEN_VALUE_FLAGS = frozenset(
    {
        "--output-root",
        "--stage2-bs-path",
        "--warm-start-run-dir",
        "--mpol",
        "--ntor",
        "--nphi",
        "--ntheta",
        "--maxiter",
        "--initial-step-scale",
        "--initial-step-maxiter",
    }
)

_STAGE_RESULT_SUMMARY_KEYS = (
    "TARGET_IOTA",
    "FINAL_IOTA",
    "FINAL_NON_QS",
    "FINAL_BOOZER_RESIDUAL",
    "TARGET_VOLUME",
    "FINAL_VOLUME",
    "FINAL_G",
    "FIELD_ERROR",
    "MAX_CURVATURE",
    "CURVE_CURVE_MIN_DIST",
    "CURVE_SURFACE_MIN_DIST",
    "SURFACE_VESSEL_MIN_DIST",
    "CC_DIST",
    "CS_DIST",
    "SS_DIST",
    "CURVATURE_THRESHOLD",
    "HARDWARE_CONSTRAINTS_OK",
    "HARDWARE_CONSTRAINT_VIOLATIONS",
    "INITIAL_PHASE_ITERATIONS",
    "iterations",
    "OPTIMIZER_SUCCESS",
    "OPTIMIZER_STATUS",
    "OPTIMIZER_NFEV",
    "OPTIMIZER_NJEV",
    "OPTIMIZER_LS_STATUS",
    "OPTIMIZER_FUN",
    "OPTIMIZER_FUN_FINITE",
    "OPTIMIZER_JAC_FINITE",
    "OPTIMIZER_JAC_INF_NORM",
    "OPTIMIZER_X_FINITE",
    "OPTIMIZER_INVALID_STATE",
    "TERMINATION_MESSAGE",
    "JAX_PROFILE_DIR",
    "INITIAL_VOLUME",
    "INITIAL_IOTA",
    "INITIAL_FIELD_ERROR",
    "INITIAL_MAX_CURVATURE",
    "TIMINGS",
)
_REQUIRED_STAGE_ARTIFACT_FILENAMES = (
    "results.json",
    "biot_savart_opt.json",
    "surf_opt.json",
)
_REQUIRED_NONFINAL_PROMOTION_FINITE_KEYS = ("FINAL_IOTA", "FINAL_G", "FIELD_ERROR")
_REQUIRED_FINAL_FINITE_KEYS = ("FINAL_IOTA", "FINAL_G", "FIELD_ERROR")
_CONTINUATION_SCHEMA_VERSION = 2
_CONTINUATION_CAMPAIGN_SCHEMA_VERSION = 1
_NONFINITE_OPTIMIZER_MESSAGE_FRAGMENT = "non-finite objective or gradient"
_NONFINAL_PROGRESS_SIGNAL_PAIRS = (
    ("FINAL_IOTA", "INITIAL_IOTA"),
    ("FIELD_ERROR", "INITIAL_FIELD_ERROR"),
    ("MAX_CURVATURE", "INITIAL_MAX_CURVATURE"),
    ("FINAL_VOLUME", "INITIAL_VOLUME"),
)
_CAMPAIGN_STATUS_RANK = {
    "research_grade": 0,
    "eligible": 1,
    "salvageable": 2,
    "rejected": 3,
}
_TARGET_LANE_PROFILE_COMPONENT_KEYS = (
    "forward_result",
    "forward_value",
    "warmstart_predict",
    "inner_solve",
    "surface_geometry",
    "field_eval",
    "solved_total_objective",
    "solved_total_gradient",
    "value_and_grad_pipeline",
)


def build_default_continuation_stages(
    *,
    final_mpol: int,
    final_ntor: int,
    final_nphi: int,
    final_ntheta: int,
    final_maxiter: int,
    coarse_maxiter: int,
    medium_maxiter: int,
    prefinal_maxiter: int,
    trial_policy: str = CONTINUATION_TRIAL_POLICY_VALIDATED_FAST,
) -> list[ContinuationStage]:
    candidate_specs = [
        ContinuationStage(
            "coarse",
            min(final_mpol, 2),
            min(final_ntor, 2),
            min(final_nphi, 31),
            min(final_ntheta, 16),
            coarse_maxiter,
        ),
        ContinuationStage(
            "medium",
            min(final_mpol, 4),
            min(final_ntor, 4),
            min(final_nphi, 63),
            min(final_ntheta, 32),
            medium_maxiter,
        ),
        ContinuationStage(
            "prefinal",
            min(final_mpol, 6),
            min(final_ntor, 6),
            min(final_nphi, 127),
            min(final_ntheta, 48),
            prefinal_maxiter,
        ),
        ContinuationStage(
            "final",
            final_mpol,
            final_ntor,
            final_nphi,
            final_ntheta,
            final_maxiter,
        ),
    ]
    if trial_policy == CONTINUATION_TRIAL_POLICY_VALIDATED_FAST:
        candidate_specs = [
            ContinuationStage(
                **(stage.__dict__ | _VALIDATED_FAST_TRIAL_STAGE_OVERRIDES[stage.name])
            )
            for stage in candidate_specs
        ]
    elif trial_policy != CONTINUATION_TRIAL_POLICY_NONE:
        raise ValueError(f"Unsupported continuation trial policy: {trial_policy}")

    last_index_by_shape = {
        (stage.mpol, stage.ntor, stage.nphi, stage.ntheta): index
        for index, stage in enumerate(candidate_specs)
    }
    return [
        stage
        for index, stage in enumerate(candidate_specs)
        if last_index_by_shape[(stage.mpol, stage.ntor, stage.nphi, stage.ntheta)]
        == index
    ]


def strip_overridden_passthrough_args(args: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if any(token.startswith(f"{flag}=") for flag in _OVERRIDDEN_VALUE_FLAGS):
            index += 1
            continue
        if token in _OVERRIDDEN_VALUE_FLAGS:
            index += 2
            continue
        stripped.append(token)
        index += 1
    return stripped


def resolve_stage_seed_path(run_dir: Path) -> Path:
    seed_path = run_dir / "biot_savart_opt.json"
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Continuation stage seed file not found: {seed_path}"
        )
    return seed_path


def find_single_stage_run_dir(stage_output_root: Path) -> Path:
    results_paths = sorted(stage_output_root.rglob("results.json"))
    if len(results_paths) != 1:
        raise RuntimeError(
            "Expected exactly one single-stage results.json under "
            f"{stage_output_root}, found {len(results_paths)}."
        )
    return results_paths[0].parent


def load_single_stage_results(run_dir: Path) -> dict[str, object]:
    results_path = run_dir / "results.json"
    with open(results_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def summarize_single_stage_results(results: dict[str, object]) -> dict[str, object]:
    return {key: results.get(key) for key in _STAGE_RESULT_SUMMARY_KEYS}


def detect_stage_artifacts(run_dir: Path) -> dict[str, object]:
    artifact_paths = {
        filename: run_dir / filename for filename in _REQUIRED_STAGE_ARTIFACT_FILENAMES
    }
    return {
        "run_dir": str(run_dir),
        "files": {
            filename: {
                "path": str(path),
                "exists": path.exists(),
            }
            for filename, path in artifact_paths.items()
        },
    }


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        finite_value = float(value)
        if math.isfinite(finite_value):
            return finite_value
    return None


def _stage_total_time_s(results: dict[str, object]) -> float | None:
    timings = results.get("TIMINGS")
    if not isinstance(timings, dict):
        return None
    return _finite_float(timings.get("script_total_s"))


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        finite_value = float(value)
        if math.isfinite(finite_value) and finite_value.is_integer():
            return int(finite_value)
    return None


def _timings_dict(results: dict[str, object]) -> dict[str, object]:
    timings = results.get("TIMINGS")
    if not isinstance(timings, dict):
        return {}
    return timings


def _timing_metric_s(results: dict[str, object], key: str) -> float | None:
    return _finite_float(_timings_dict(results).get(key))


def _accumulate_float(
    totals: dict[str, float],
    key: str,
    profiling: dict[str, object],
    source_key: str | None = None,
) -> None:
    """Accumulate a finite float from *profiling* into *totals[key]*."""
    value = _finite_float(profiling.get(source_key or key))
    if value is not None:
        totals[key] = totals.get(key, 0.0) + value


def _accumulate_int(
    totals: dict[str, int],
    key: str,
    profiling: dict[str, object],
    source_key: str | None = None,
) -> None:
    """Accumulate a safe int from *profiling* into *totals[key]*."""
    value = _safe_int(profiling.get(source_key or key))
    if value is not None:
        totals[key] = totals.get(key, 0) + value


def _ratio_or_none(
    numerator: int | float | None,
    denominator: int | float | None,
) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _profile_dir_value(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _summarize_profiled_callable(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}

    summary: dict[str, object] = {}
    compile_overhead_s = _finite_float(payload.get("compile_overhead_s"))
    if compile_overhead_s is not None:
        summary["compile_overhead_s"] = compile_overhead_s

    first = payload.get("first")
    if isinstance(first, dict):
        first_total_s = _finite_float(first.get("total_s"))
        if first_total_s is not None:
            summary["first_total_s"] = first_total_s

    warm = payload.get("warm")
    if isinstance(warm, dict):
        warm_total_s = _finite_float(warm.get("total_s"))
        if warm_total_s is not None:
            summary["warm_total_s"] = warm_total_s

    return summary


def _summarize_target_lane_profile(results: dict[str, object]) -> dict[str, object]:
    raw_profile = results.get("TARGET_LANE_PROFILE")
    if not isinstance(raw_profile, dict):
        return {}

    summary: dict[str, object] = {}
    solve_success = raw_profile.get("solve_success")
    if isinstance(solve_success, bool):
        summary["solve_success"] = solve_success

    component_summaries: dict[str, object] = {}
    for key in _TARGET_LANE_PROFILE_COMPONENT_KEYS:
        component_summary = _summarize_profiled_callable(raw_profile.get(key))
        if component_summary:
            component_summaries[key] = component_summary
    if component_summaries:
        summary["components"] = component_summaries

    value_and_grad_summary = component_summaries.get("value_and_grad_pipeline")
    if isinstance(value_and_grad_summary, dict):
        value_and_grad_compile_overhead_s = _finite_float(
            value_and_grad_summary.get("compile_overhead_s")
        )
        if value_and_grad_compile_overhead_s is not None:
            summary["value_and_grad_compile_overhead_s"] = (
                value_and_grad_compile_overhead_s
            )

    inner_solve_summary = component_summaries.get("inner_solve")
    if isinstance(inner_solve_summary, dict):
        inner_solve_compile_overhead_s = _finite_float(
            inner_solve_summary.get("compile_overhead_s")
        )
        if inner_solve_compile_overhead_s is not None:
            summary["inner_solve_compile_overhead_s"] = inner_solve_compile_overhead_s

    batched_seed_profile = raw_profile.get("batched_seed_profile")
    if isinstance(batched_seed_profile, dict):
        batched_summary: dict[str, object] = {}
        batch_size = _safe_int(batched_seed_profile.get("batch_size"))
        if batch_size is not None:
            batched_summary["batch_size"] = batch_size
        for key in (
            "first_total_s_per_seed",
            "warm_total_s_per_seed",
            "max_value_abs",
            "max_gradient_inf_norm",
        ):
            finite_value = _finite_float(batched_seed_profile.get(key))
            if finite_value is not None:
                batched_summary[key] = finite_value
        for key in ("all_values_finite", "all_gradients_finite"):
            maybe_bool = batched_seed_profile.get(key)
            if isinstance(maybe_bool, bool):
                batched_summary[key] = maybe_bool
        batched_pipeline = _summarize_profiled_callable(
            batched_seed_profile.get("value_and_grad_pipeline")
        )
        if batched_pipeline:
            batched_summary["value_and_grad_pipeline"] = batched_pipeline
        if batched_summary:
            summary["batched_seed_profile"] = batched_summary

    return summary


def build_stage_profiling_summary(
    results: dict[str, object],
    *,
    jax_profile_dir: str | None,
) -> dict[str, object]:
    accepted_step_count = _safe_int(results.get("iterations"))
    objective_eval_count = _safe_int(results.get("OPTIMIZER_NFEV"))
    gradient_eval_count = _safe_int(results.get("OPTIMIZER_NJEV"))
    profiling: dict[str, object] = {
        "jax_profile_dir": jax_profile_dir,
        "script_total_s": _timing_metric_s(results, "script_total_s"),
        "outer_optimizer_s": _timing_metric_s(results, "outer_optimizer_s"),
        "outer_optimizer_initial_phase_s": _timing_metric_s(
            results, "outer_optimizer_initial_phase_s"
        ),
        "outer_optimizer_main_s": _timing_metric_s(results, "outer_optimizer_main_s"),
        "target_lane_bundle_setup_s": _timing_metric_s(
            results, "target_lane_bundle_setup_s"
        ),
        "target_lane_profile_only_s": _timing_metric_s(
            results, "target_lane_profile_only_s"
        ),
        "accepted_step_count": accepted_step_count,
        "initial_phase_iterations": _safe_int(results.get("INITIAL_PHASE_ITERATIONS")),
        "objective_eval_count": objective_eval_count,
        "gradient_eval_count": gradient_eval_count,
        "objective_evals_per_accepted_step": _ratio_or_none(
            objective_eval_count,
            accepted_step_count,
        ),
    }
    target_lane_profile = _summarize_target_lane_profile(results)
    if target_lane_profile:
        profiling["target_lane_profile"] = target_lane_profile
    return profiling


def _aggregate_stage_profiling(stage_reports: list[dict[str, object]]) -> dict[str, object]:
    profile_count = 0
    target_lane_profile_count = 0
    floats: dict[str, float] = {}
    ints: dict[str, int] = {}
    stage_summaries: dict[str, object] = {}

    for stage_report in stage_reports:
        profiling = stage_report.get("profiling")
        if not isinstance(profiling, dict):
            continue
        profile_count += 1
        stage_name = stage_report.get("name")
        if isinstance(stage_name, str):
            stage_summaries[stage_name] = profiling

        _accumulate_float(floats, "total_stage_script_time_s", profiling, source_key="script_total_s")
        _accumulate_float(floats, "total_outer_optimizer_s", profiling, source_key="outer_optimizer_s")
        _accumulate_float(floats, "total_outer_optimizer_initial_phase_s", profiling, source_key="outer_optimizer_initial_phase_s")
        _accumulate_float(floats, "total_outer_optimizer_main_s", profiling, source_key="outer_optimizer_main_s")
        _accumulate_float(floats, "total_target_lane_bundle_setup_s", profiling, source_key="target_lane_bundle_setup_s")
        _accumulate_int(ints, "total_accepted_step_count", profiling, source_key="accepted_step_count")
        _accumulate_int(ints, "total_objective_eval_count", profiling, source_key="objective_eval_count")
        _accumulate_int(ints, "total_gradient_eval_count", profiling, source_key="gradient_eval_count")

        target_lane_profile = profiling.get("target_lane_profile")
        if not isinstance(target_lane_profile, dict):
            continue
        target_lane_profile_count += 1
        _accumulate_float(floats, "total_value_and_grad_compile_overhead_s", target_lane_profile, source_key="value_and_grad_compile_overhead_s")
        _accumulate_float(floats, "total_inner_solve_compile_overhead_s", target_lane_profile, source_key="inner_solve_compile_overhead_s")

    return {
        "profiled_stage_count": profile_count,
        "target_lane_profiled_stage_count": target_lane_profile_count,
        "total_stage_script_time_s": floats.get("total_stage_script_time_s"),
        "total_outer_optimizer_s": floats.get("total_outer_optimizer_s"),
        "total_outer_optimizer_initial_phase_s": floats.get("total_outer_optimizer_initial_phase_s"),
        "total_outer_optimizer_main_s": floats.get("total_outer_optimizer_main_s"),
        "total_target_lane_bundle_setup_s": floats.get("total_target_lane_bundle_setup_s"),
        "total_accepted_step_count": ints.get("total_accepted_step_count"),
        "total_objective_eval_count": ints.get("total_objective_eval_count"),
        "total_gradient_eval_count": ints.get("total_gradient_eval_count"),
        "objective_evals_per_accepted_step": _ratio_or_none(
            ints.get("total_objective_eval_count"),
            ints.get("total_accepted_step_count"),
        ),
        "total_value_and_grad_compile_overhead_s": floats.get("total_value_and_grad_compile_overhead_s"),
        "total_inner_solve_compile_overhead_s": floats.get("total_inner_solve_compile_overhead_s"),
        "stages": stage_summaries,
    }


def _aggregate_campaign_profiling(
    donor_records: list[dict[str, object]],
) -> dict[str, object]:
    profiled_candidate_count = 0
    floats: dict[str, float] = {}
    ints: dict[str, int] = {}

    for record in donor_records:
        profiling = record.get("profiling")
        if not isinstance(profiling, dict):
            continue
        profiled_candidate_count += 1

        _accumulate_float(floats, "total_stage_script_time_s", profiling)
        _accumulate_float(floats, "total_outer_optimizer_s", profiling)
        _accumulate_float(floats, "total_outer_optimizer_initial_phase_s", profiling)
        _accumulate_float(floats, "total_outer_optimizer_main_s", profiling)
        _accumulate_float(floats, "total_target_lane_bundle_setup_s", profiling)
        _accumulate_int(ints, "total_accepted_step_count", profiling)
        _accumulate_int(ints, "total_objective_eval_count", profiling)
        _accumulate_int(ints, "total_gradient_eval_count", profiling)
        _accumulate_float(floats, "total_value_and_grad_compile_overhead_s", profiling)

    return {
        "profiled_candidate_count": profiled_candidate_count,
        "total_stage_script_time_s": floats.get("total_stage_script_time_s"),
        "total_outer_optimizer_s": floats.get("total_outer_optimizer_s"),
        "total_outer_optimizer_initial_phase_s": floats.get("total_outer_optimizer_initial_phase_s"),
        "total_outer_optimizer_main_s": floats.get("total_outer_optimizer_main_s"),
        "total_target_lane_bundle_setup_s": floats.get("total_target_lane_bundle_setup_s"),
        "total_accepted_step_count": ints.get("total_accepted_step_count"),
        "total_objective_eval_count": ints.get("total_objective_eval_count"),
        "total_gradient_eval_count": ints.get("total_gradient_eval_count"),
        "objective_evals_per_accepted_step": _ratio_or_none(
            ints.get("total_objective_eval_count"),
            ints.get("total_accepted_step_count"),
        ),
        "total_value_and_grad_compile_overhead_s": floats.get("total_value_and_grad_compile_overhead_s"),
    }


def _build_campaign_branch_decision(
    *,
    profiling: dict[str, object],
    passed_candidate_count: int,
    research_grade_candidate_count: int,
    trial_policy: str,
) -> dict[str, object]:
    profiled_candidate_count = _safe_int(profiling.get("profiled_candidate_count"))
    total_stage_script_time_s = _finite_float(profiling.get("total_stage_script_time_s"))
    total_outer_optimizer_s = _finite_float(profiling.get("total_outer_optimizer_s"))
    total_outer_optimizer_initial_phase_s = _finite_float(
        profiling.get("total_outer_optimizer_initial_phase_s")
    )
    total_outer_optimizer_main_s = _finite_float(
        profiling.get("total_outer_optimizer_main_s")
    )
    total_target_lane_bundle_setup_s = _finite_float(
        profiling.get("total_target_lane_bundle_setup_s")
    )
    total_accepted_step_count = _safe_int(profiling.get("total_accepted_step_count"))
    total_objective_eval_count = _safe_int(profiling.get("total_objective_eval_count"))
    objective_evals_per_accepted_step = _finite_float(
        profiling.get("objective_evals_per_accepted_step")
    )
    signals = {
        "profiled_candidate_count": profiled_candidate_count,
        "passed_candidate_count": passed_candidate_count,
        "research_grade_candidate_count": research_grade_candidate_count,
        "total_stage_script_time_s": total_stage_script_time_s,
        "total_outer_optimizer_s": total_outer_optimizer_s,
        "total_outer_optimizer_initial_phase_s": total_outer_optimizer_initial_phase_s,
        "total_outer_optimizer_main_s": total_outer_optimizer_main_s,
        "total_target_lane_bundle_setup_s": total_target_lane_bundle_setup_s,
        "total_accepted_step_count": total_accepted_step_count,
        "total_objective_eval_count": total_objective_eval_count,
        "objective_evals_per_accepted_step": objective_evals_per_accepted_step,
    }
    if not profiled_candidate_count:
        return {
            "category": "insufficient_signal",
            "rationale": ["No profiled donor records were available."],
            "recommended_actions": [
                "Re-run the continuation campaign with --jax-profile-dir enabled."
            ],
            "signals": signals,
        }

    if (
        total_objective_eval_count is not None
        and total_objective_eval_count > 0
        and total_accepted_step_count == 0
    ):
        rationale = [
            "Profiled donors consumed objective evaluations without any accepted outer-loop progress.",
        ]
        if (
            total_outer_optimizer_s is not None
            and total_target_lane_bundle_setup_s is not None
        ):
            rationale.append(
                "Outer-optimizer time dominated target-lane bundle setup time."
            )
        recommended_actions = [
            "Reduce non-final outer-loop reevaluation before tuning hardware or XLA flags.",
        ]
        if trial_policy == CONTINUATION_TRIAL_POLICY_VALIDATED_FAST:
            if (
                total_outer_optimizer_initial_phase_s is not None
                and total_outer_optimizer_initial_phase_s > 0.0
            ):
                recommended_actions.insert(
                    0,
                    "Skip the validated-fast coarse scaled initial outer phase and re-profile the same donor set.",
                )
            else:
                rationale.append(
                    "The validated-fast coarse scaled phase is already absent, so the remaining waste is in the main outer loop."
                )
                recommended_actions.insert(
                    0,
                    "Tighten the validated-fast non-final outer line-search budget and re-profile the same donor set.",
                )
        return {
            "category": "reevaluation_or_host_stall_dominated",
            "rationale": rationale,
            "recommended_actions": recommended_actions,
            "signals": signals,
        }

    if passed_candidate_count == 0 and total_accepted_step_count not in (None, 0):
        return {
            "category": "donor_quality_dominated",
            "rationale": [
                "The campaign recorded accepted continuation progress but no donor passed the final continuation contract."
            ],
            "recommended_actions": [
                "Implement donor ranking and seed-selection policy before expanding schedules."
            ],
            "signals": signals,
        }

    if passed_candidate_count > 0:
        return {
            "category": "campaign_ready_for_convergence",
            "rationale": [
                "The campaign already produced continuation-valid donors, so the next bottleneck is no longer basic throughput triage."
            ],
            "recommended_actions": [
                "Run longer multi-donor convergence campaigns and rank final candidates on physics and hardware gates."
            ],
            "signals": signals,
        }

    return {
        "category": "insufficient_signal",
        "rationale": [
            "The available profiling metrics do not yet separate reevaluation, device-throughput, and donor-quality limits cleanly."
        ],
        "recommended_actions": [
            "Inspect the per-donor continuation profiling reports and rerun with a tighter donor set if needed."
        ],
        "signals": signals,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _termination_message_indicates_invalid_state(message: object) -> bool:
    return isinstance(message, str) and (
        _NONFINITE_OPTIMIZER_MESSAGE_FRAGMENT in message.lower()
    )


def _nonfinal_stage_recorded_progress(metrics: dict[str, object]) -> bool:
    iterations = metrics.get("iterations")
    if isinstance(iterations, int):
        if iterations > 0:
            return True
    elif isinstance(iterations, float):
        if math.isfinite(iterations) and int(iterations) > 0:
            return True

    for final_key, initial_key in _NONFINAL_PROGRESS_SIGNAL_PAIRS:
        final_value = _finite_float(metrics.get(final_key))
        initial_value = _finite_float(metrics.get(initial_key))
        if final_value is None or initial_value is None:
            continue
        if not math.isclose(
            final_value,
            initial_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return True
    return False


def collect_stage_run_snapshot(stage_output_root: Path) -> dict[str, object]:
    results_paths = sorted(stage_output_root.rglob("results.json"))
    snapshot: dict[str, object] = {
        "stage_output_root": str(stage_output_root),
        "results_json_count": len(results_paths),
    }
    if results_paths:
        snapshot["results_json_paths"] = [str(path) for path in results_paths]
    if len(results_paths) != 1:
        return snapshot

    run_dir = results_paths[0].parent
    snapshot["run_dir"] = str(run_dir)
    snapshot["artifacts"] = detect_stage_artifacts(run_dir)
    try:
        loaded_results = load_single_stage_results(run_dir)
        snapshot["results"] = summarize_single_stage_results(loaded_results)
        snapshot["profiling"] = build_stage_profiling_summary(
            loaded_results,
            jax_profile_dir=_profile_dir_value(loaded_results.get("JAX_PROFILE_DIR")),
        )
    except Exception as exc:
        snapshot["results_load_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def classify_existing_stage_output(
    stage_output_root: Path,
    snapshot: dict[str, object],
) -> str:
    results_json_count = snapshot.get("results_json_count")
    if isinstance(results_json_count, int):
        if results_json_count == 1:
            if isinstance(snapshot.get("results_load_error"), str):
                return "completed_unreadable"
            return "completed"
        if results_json_count > 1:
            return "ambiguous_existing"
    try:
        has_existing_content = any(stage_output_root.iterdir())
    except FileNotFoundError:
        has_existing_content = False
    return "incomplete_existing" if has_existing_content else "not_started"


def inspect_existing_stage_output(stage_output_root: Path) -> tuple[str, dict[str, object]]:
    snapshot = collect_stage_run_snapshot(stage_output_root)
    return classify_existing_stage_output(stage_output_root, snapshot), snapshot


def build_stage_output_root(
    run_root: Path,
    *,
    stage_index: int,
    stage: ContinuationStage,
) -> Path:
    return run_root / f"stage-{stage_index:02d}-{stage.name}"


def resolve_jax_profile_root(
    *,
    run_root: Path,
    requested_profile_dir: str | None,
) -> Path | None:
    if requested_profile_dir is None:
        return None
    requested_path = Path(requested_profile_dir).expanduser()
    if requested_path.is_absolute():
        return requested_path.resolve()
    return (run_root / requested_path).resolve()


def build_stage_jax_profile_dir(
    jax_profile_root: Path | None,
    *,
    stage_index: int,
    stage: ContinuationStage,
) -> Path | None:
    if jax_profile_root is None:
        return None
    return jax_profile_root / f"stage-{stage_index:02d}-{stage.name}"


def build_stage_record(
    *,
    stage: ContinuationStage,
    stage_output_root: Path,
    stage2_seed_path: Path | None,
    warm_start_run_dir: Path | None,
    jax_profile_dir: Path | None,
    command: list[str],
) -> dict[str, object]:
    return {
        "name": stage.name,
        "shape": {
            "mpol": stage.mpol,
            "ntor": stage.ntor,
            "nphi": stage.nphi,
            "ntheta": stage.ntheta,
        },
        "maxiter": stage.maxiter,
        "minimal_artifacts": stage.minimal_artifacts,
        "outer_maxls": stage.outer_maxls,
        "maxcor": stage.maxcor,
        "initial_step_scale": stage.initial_step_scale,
        "initial_step_maxiter": stage.initial_step_maxiter,
        "target_lane_boozer_bfgs_tol": stage.target_lane_boozer_bfgs_tol,
        "target_lane_boozer_bfgs_maxiter": stage.target_lane_boozer_bfgs_maxiter,
        "stage2_seed_path": None
        if stage2_seed_path is None
        else str(stage2_seed_path),
        "stage_output_root": str(stage_output_root),
        "warm_start_run_dir": None
        if warm_start_run_dir is None
        else str(warm_start_run_dir),
        "jax_profile_dir": None
        if jax_profile_dir is None
        else str(jax_profile_dir),
        "command": command,
    }


def annotate_existing_stage_jax_profile_dir(
    stage_record: dict[str, object],
    *,
    stage_jax_profile_dir: Path | None,
) -> None:
    planned_stage_profile_dir = (
        None if stage_jax_profile_dir is None else str(stage_jax_profile_dir)
    )
    if stage_jax_profile_dir is not None and stage_jax_profile_dir.exists():
        stage_record["jax_profile_dir"] = planned_stage_profile_dir
        return
    results_value = stage_record.get("results")
    if isinstance(results_value, dict):
        existing_results_profile_dir = _profile_dir_value(
            results_value.get("JAX_PROFILE_DIR")
        )
        if existing_results_profile_dir is not None:
            stage_record["jax_profile_dir"] = existing_results_profile_dir
            return
    existing_stage_profile_dir = _profile_dir_value(stage_record.get("jax_profile_dir"))
    if (
        existing_stage_profile_dir is not None
        and existing_stage_profile_dir != planned_stage_profile_dir
    ):
        stage_record["jax_profile_dir"] = existing_stage_profile_dir
        return
    stage_record["jax_profile_dir"] = None


_RERUN_STAGE_OUTCOME_KEYS = (
    "artifacts",
    "completed_at_utc",
    "failure_kind",
    "failure_message",
    "results",
    "results_json_count",
    "results_json_paths",
    "results_load_error",
    "run_dir",
    "stage_contract",
    "started_at_utc",
    "status",
    "subprocess_returncode",
)


def reset_stage_record_for_rerun(stage_record: dict[str, object]) -> None:
    for key in _RERUN_STAGE_OUTCOME_KEYS:
        stage_record.pop(key, None)
    stage_record["reused_existing_run"] = False


def _threshold_verdict(value: float | None, threshold: float | None) -> bool | None:
    if value is None or threshold is None:
        return None
    return value <= threshold


def _build_research_verdicts(
    final_stage_report: dict[str, object] | None,
    *,
    max_final_field_error: float | None,
    max_final_abs_iota_error: float | None,
    max_final_non_qs: float | None,
) -> dict[str, object]:
    if not isinstance(final_stage_report, dict):
        return {
            "full_convergence": False,
            "hardware_feasible_final_coils": False,
            "acceptable_final_field_error": None,
            "acceptable_iota_target": None,
            "acceptable_non_qs_behavior": None,
            "physics_gate_pass": None,
            "research_grade_ready": False,
            "evidence_gaps": ["final stage report is unavailable"],
        }

    metrics = final_stage_report.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    field_error = _finite_float(metrics.get("FIELD_ERROR"))
    abs_iota_error = _finite_float(final_stage_report.get("abs_iota_error"))
    final_non_qs = _finite_float(metrics.get("FINAL_NON_QS"))
    optimizer_success = metrics.get("OPTIMIZER_SUCCESS") is True
    hardware_ok = metrics.get("HARDWARE_CONSTRAINTS_OK") is True
    field_ok = _threshold_verdict(field_error, max_final_field_error)
    iota_ok = _threshold_verdict(abs_iota_error, max_final_abs_iota_error)
    non_qs_ok = _threshold_verdict(final_non_qs, max_final_non_qs)

    evidence_gaps: list[str] = []
    if max_final_field_error is None:
        evidence_gaps.append("max_final_field_error threshold is unset")
    elif field_error is None:
        evidence_gaps.append("final FIELD_ERROR metric is missing or non-finite")
    if max_final_abs_iota_error is None:
        evidence_gaps.append("max_final_abs_iota_error threshold is unset")
    elif abs_iota_error is None:
        evidence_gaps.append("final |FINAL_IOTA - TARGET_IOTA| is unavailable")
    if max_final_non_qs is None:
        evidence_gaps.append("max_final_non_qs threshold is unset")
    elif final_non_qs is None:
        evidence_gaps.append("final FINAL_NON_QS metric is missing or non-finite")

    physics_gate_inputs = [verdict for verdict in (field_ok, iota_ok, non_qs_ok) if verdict is not None]
    physics_gate_pass = None if not physics_gate_inputs else all(physics_gate_inputs)
    research_grade_ready = (
        optimizer_success
        and hardware_ok
        and physics_gate_pass is True
    )
    return {
        "full_convergence": optimizer_success,
        "hardware_feasible_final_coils": hardware_ok,
        "acceptable_final_field_error": field_ok,
        "acceptable_iota_target": iota_ok,
        "acceptable_non_qs_behavior": non_qs_ok,
        "physics_gate_pass": physics_gate_pass,
        "research_grade_ready": research_grade_ready,
        "evidence_gaps": evidence_gaps,
    }


def evaluate_continuation_stage(
    stage_record: dict[str, object],
    *,
    is_final_stage: bool,
    max_final_field_error: float | None,
    max_final_abs_iota_error: float | None,
    max_final_non_qs: float | None,
) -> dict[str, object]:
    stage_name = str(stage_record.get("name", "unknown"))
    run_dir_value = stage_record.get("run_dir")
    results = stage_record.get("results")
    execution_status = str(stage_record.get("status", "unknown"))
    failures: list[str] = []
    warnings: list[str] = []
    completed = execution_status == "completed" and isinstance(run_dir_value, str) and bool(run_dir_value)
    artifact_report: dict[str, object] | None = None
    if completed:
        artifact_report = detect_stage_artifacts(Path(run_dir_value))
        missing_artifacts = [
            filename
            for filename, payload in artifact_report["files"].items()
            if not payload["exists"]
        ]
        if missing_artifacts:
            failures.append(
                "missing required stage artifacts: " + ", ".join(missing_artifacts)
            )
    else:
        failures.append(f"stage status is {execution_status!r}, not 'completed'")
        if isinstance(run_dir_value, str) and bool(run_dir_value):
            warnings.append("stage produced a run_dir but did not finish cleanly")

    if not isinstance(results, dict):
        failures.append("stage did not record a results snapshot")
        results = {}
    results_load_error = stage_record.get("results_load_error")
    if isinstance(results_load_error, str):
        failures.append(f"stage results could not be loaded: {results_load_error}")

    metrics = summarize_single_stage_results(results)
    resolved_jax_profile_dir = _profile_dir_value(stage_record.get("jax_profile_dir"))
    if resolved_jax_profile_dir is None:
        resolved_jax_profile_dir = _profile_dir_value(results.get("JAX_PROFILE_DIR"))
    stage_profiling = stage_record.get("profiling")
    if isinstance(stage_profiling, dict):
        stage_profiling = dict(stage_profiling)
        if resolved_jax_profile_dir is not None:
            stage_profiling["jax_profile_dir"] = resolved_jax_profile_dir
    else:
        stage_profiling = build_stage_profiling_summary(
            results,
            jax_profile_dir=resolved_jax_profile_dir,
        )
    stage_total_time_s = _stage_total_time_s(results)
    if completed and stage_total_time_s is None:
        warnings.append("stage TIMINGS.script_total_s is missing or non-finite")
    invalid_state_signals = (
        metrics.get("OPTIMIZER_INVALID_STATE") is True
        or metrics.get("OPTIMIZER_FUN_FINITE") is False
        or metrics.get("OPTIMIZER_JAC_FINITE") is False
        or metrics.get("OPTIMIZER_X_FINITE") is False
        or _termination_message_indicates_invalid_state(
            metrics.get("TERMINATION_MESSAGE")
        )
    )
    accepted_progress = _nonfinal_stage_recorded_progress(metrics)

    if is_final_stage:
        if metrics.get("OPTIMIZER_SUCCESS") is not True:
            failures.append("final stage OPTIMIZER_SUCCESS is not true")
        if metrics.get("HARDWARE_CONSTRAINTS_OK") is not True:
            failures.append("final stage HARDWARE_CONSTRAINTS_OK is not true")
        if invalid_state_signals:
            failures.append("final stage optimizer ended in an invalid state")
        for key in _REQUIRED_FINAL_FINITE_KEYS:
            if _finite_float(metrics.get(key)) is None:
                failures.append(f"final stage {key} is missing or non-finite")
        final_field_error = _finite_float(metrics.get("FIELD_ERROR"))
        if (
            max_final_field_error is not None
            and final_field_error is not None
            and final_field_error > max_final_field_error
        ):
            failures.append(
                "final stage FIELD_ERROR "
                f"{final_field_error:.6g} exceeds threshold {max_final_field_error:.6g}"
            )
        target_iota = _finite_float(metrics.get("TARGET_IOTA"))
        final_iota = _finite_float(metrics.get("FINAL_IOTA"))
        abs_iota_error = (
            None
            if target_iota is None or final_iota is None
            else abs(final_iota - target_iota)
        )
        if max_final_abs_iota_error is not None:
            if abs_iota_error is None:
                failures.append(
                    "final stage TARGET_IOTA or FINAL_IOTA is missing or non-finite"
                )
            elif abs_iota_error > max_final_abs_iota_error:
                failures.append(
                    "final stage |FINAL_IOTA - TARGET_IOTA| "
                    f"{abs_iota_error:.6g} exceeds threshold "
                    f"{max_final_abs_iota_error:.6g}"
                )
        final_non_qs = _finite_float(metrics.get("FINAL_NON_QS"))
        if max_final_non_qs is not None:
            if final_non_qs is None:
                failures.append("final stage FINAL_NON_QS is missing or non-finite")
            elif final_non_qs > max_final_non_qs:
                failures.append(
                    "final stage FINAL_NON_QS "
                    f"{final_non_qs:.6g} exceeds threshold "
                    f"{max_final_non_qs:.6g}"
                )
    else:
        if metrics.get("HARDWARE_CONSTRAINTS_OK") is not True:
            failures.append("non-final stage HARDWARE_CONSTRAINTS_OK is not true")
        if invalid_state_signals:
            failures.append("non-final stage optimizer ended in an invalid state")
        for key in _REQUIRED_NONFINAL_PROMOTION_FINITE_KEYS:
            if _finite_float(metrics.get(key)) is None:
                failures.append(f"non-final stage {key} is missing or non-finite")
        if not accepted_progress:
            failures.append("non-final stage recorded no accepted optimizer progress")
        elif metrics.get("OPTIMIZER_SUCCESS") is not True:
            warnings.append(
                "non-final stage terminated without OPTIMIZER_SUCCESS but "
                "recorded accepted progress"
            )
        abs_iota_error = None

    salvage_status = "none"
    if isinstance(run_dir_value, str) and bool(run_dir_value):
        salvage_status = "run_dir_only" if not isinstance(results, dict) or not results else "results_available"
    if completed and not failures:
        salvage_status = "complete"

    return {
        "name": stage_name,
        "status": execution_status,
        "completed": completed,
        "salvage_status": salvage_status,
        "artifacts": artifact_report,
        "metrics": metrics,
        "profiling": stage_profiling,
        "accepted_progress": None if is_final_stage else accepted_progress,
        "script_total_s": stage_total_time_s,
        "abs_iota_error": abs_iota_error,
        "subprocess_returncode": stage_record.get("subprocess_returncode"),
        "failures": failures,
        "warnings": warnings,
        "passed": not failures,
    }


def build_continuation_validation_report(
    summary: dict[str, object],
    *,
    max_final_field_error: float | None,
    max_final_abs_iota_error: float | None,
    max_final_non_qs: float | None,
) -> dict[str, object]:
    stage_records = summary.get("stages")
    if not isinstance(stage_records, list):
        stage_records = []
    stage_reports: list[dict[str, object]] = []
    failures: list[str] = []
    warnings: list[str] = []
    completed_stage_count = 0
    completed_field_errors: list[float] = []
    total_stage_script_time_s = 0.0
    total_stage_script_time_available = False

    for index, stage_record in enumerate(stage_records):
        if not isinstance(stage_record, dict):
            failures.append(f"stage #{index + 1} summary is not a mapping")
            continue
        stage_report = evaluate_continuation_stage(
            stage_record,
            is_final_stage=index == len(stage_records) - 1,
            max_final_field_error=max_final_field_error,
            max_final_abs_iota_error=max_final_abs_iota_error,
            max_final_non_qs=max_final_non_qs,
        )
        stage_reports.append(stage_report)
        if stage_report["completed"]:
            completed_stage_count += 1
        script_total_s = stage_report["script_total_s"]
        if script_total_s is not None:
            total_stage_script_time_s += script_total_s
            total_stage_script_time_available = True
        field_error = _finite_float(stage_report["metrics"].get("FIELD_ERROR"))
        if field_error is not None:
            completed_field_errors.append(field_error)
        failures.extend(
            f"{stage_report['name']}: {message}" for message in stage_report["failures"]
        )
        warnings.extend(
            f"{stage_report['name']}: {message}" for message in stage_report["warnings"]
        )

    if not stage_reports:
        failures.append("continuation summary did not contain any stages")

    field_error_improvement = None
    if len(completed_field_errors) >= 2:
        field_error_improvement = completed_field_errors[0] - completed_field_errors[-1]

    final_stage_report = stage_reports[-1] if stage_reports else None
    research_verdicts = _build_research_verdicts(
        final_stage_report,
        max_final_field_error=max_final_field_error,
        max_final_abs_iota_error=max_final_abs_iota_error,
        max_final_non_qs=max_final_non_qs,
    )
    profiling = _aggregate_stage_profiling(stage_reports)
    return {
        "schema_version": _CONTINUATION_SCHEMA_VERSION,
        "run_root": summary.get("run_root"),
        "planned_stage_count": len(stage_records),
        "completed_stage_count": completed_stage_count,
        "validation_config": {
            "max_final_field_error": max_final_field_error,
            "max_final_abs_iota_error": max_final_abs_iota_error,
            "max_final_non_qs": max_final_non_qs,
            "required_stage_artifacts": list(_REQUIRED_STAGE_ARTIFACT_FILENAMES),
            "required_nonfinal_promotion_finite_keys": list(
                _REQUIRED_NONFINAL_PROMOTION_FINITE_KEYS
            ),
            "required_final_finite_keys": list(_REQUIRED_FINAL_FINITE_KEYS),
        },
        "final_stage": final_stage_report,
        "research_verdicts": research_verdicts,
        "aggregate": {
            "field_error_improvement_vs_first_completed": field_error_improvement,
            "total_stage_script_time_s": (
                total_stage_script_time_s if total_stage_script_time_available else None
            ),
        },
        "profiling": profiling,
        "stage_reports": stage_reports,
        "failures": failures,
        "warnings": warnings,
        "passed": not failures,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def write_text(path: Path, payload: str) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(payload)


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def load_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def _slugify_path_component(value: str) -> str:
    slug_chars: list[str] = []
    last_was_dash = False
    for char in value.lower():
        if char.isalnum():
            slug_chars.append(char)
            last_was_dash = False
            continue
        if not last_was_dash:
            slug_chars.append("-")
            last_was_dash = True
    slug = "".join(slug_chars).strip("-")
    return slug or "donor"


def _require_campaign_donor_run_dir(path_value: str) -> Path:
    run_dir = Path(path_value).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Campaign donor run directory does not exist: {run_dir}")
    missing = [
        filename
        for filename in _REQUIRED_STAGE_ARTIFACT_FILENAMES
        if not (run_dir / filename).exists()
    ]
    if missing:
        raise SystemExit(
            "Campaign donor run directory is missing required artifacts: "
            + ", ".join(str(run_dir / filename) for filename in missing)
        )
    return run_dir


def _continuation_report_status(report: dict[str, object]) -> tuple[str, bool]:
    final_stage = report.get("final_stage")
    research_verdicts = report.get("research_verdicts")
    if not isinstance(final_stage, dict):
        return "rejected", False
    passed = report.get("passed") is True
    research_grade_ready = (
        isinstance(research_verdicts, dict)
        and research_verdicts.get("research_grade_ready") is True
    )
    if passed and research_grade_ready:
        return "research_grade", True
    if passed:
        return "eligible", False
    if final_stage:
        return "salvageable", False
    return "rejected", False


def build_campaign_candidate_record(
    *,
    donor_index: int,
    donor_label: str,
    donor_run_dir: Path,
    outcome: ContinuationRunOutcome,
) -> dict[str, object]:
    if not isinstance(outcome.report, dict):
        status = "rejected"
        research_grade = False
        failures = ["continuation validation report is unavailable"]
        warnings: list[str] = []
        final_stage: dict[str, object] = {}
    else:
        status, research_grade = _continuation_report_status(outcome.report)
        failures = list(outcome.report.get("failures", []))
        warnings = list(outcome.report.get("warnings", []))
        final_stage = outcome.report.get("final_stage")
        if not isinstance(final_stage, dict):
            final_stage = {}

    metrics = final_stage.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    field_error = _finite_float(metrics.get("FIELD_ERROR"))
    abs_iota_error = _finite_float(final_stage.get("abs_iota_error"))
    final_non_qs = _finite_float(metrics.get("FINAL_NON_QS"))

    return {
        "donor_index": donor_index,
        "donor_label": donor_label,
        "donor_run_dir": str(donor_run_dir),
        "run_root": str(outcome.run_root),
        "summary_path": None
        if outcome.summary_path is None
        else str(outcome.summary_path),
        "validation_path": None
        if outcome.report_path is None
        else str(outcome.report_path),
        "exit_code": outcome.exit_code,
        "status": status,
        "research_grade": research_grade,
        "passed": isinstance(outcome.report, dict)
        and outcome.report.get("passed") is True,
        "research_verdicts": {}
        if not isinstance(outcome.report, dict)
        else outcome.report.get("research_verdicts", {}),
        "profiling": {}
        if not isinstance(outcome.report, dict)
        else outcome.report.get("profiling", {}),
        "metrics": {
            "FIELD_ERROR": field_error,
            "FINAL_IOTA": _finite_float(metrics.get("FINAL_IOTA")),
            "FINAL_NON_QS": final_non_qs,
            "FINAL_BOOZER_RESIDUAL": _finite_float(
                metrics.get("FINAL_BOOZER_RESIDUAL")
            ),
        },
        "ranking": {
            "status_rank": _CAMPAIGN_STATUS_RANK[status],
            "sort_key": [
                _CAMPAIGN_STATUS_RANK[status],
                math.inf if field_error is None else field_error,
                math.inf if abs_iota_error is None else abs_iota_error,
                math.inf if final_non_qs is None else final_non_qs,
            ],
        },
        "failures": failures,
        "warnings": warnings,
    }


def build_continuation_campaign_summary(
    *,
    campaign_root: Path,
    run_id: str,
    donor_records: list[dict[str, object]],
    passthrough_args: list[str],
    trial_policy: str,
    validation_thresholds: dict[str, object],
) -> dict[str, object]:
    sorted_records = sorted(
        donor_records,
        key=lambda record: tuple(record["ranking"]["sort_key"]),
    )
    passed_candidate_count = sum(
        record["passed"] is True for record in sorted_records
    )
    research_grade_candidate_count = sum(
        record["research_grade"] is True for record in sorted_records
    )
    status_counts = {
        status: sum(record["status"] == status for record in sorted_records)
        for status in _CAMPAIGN_STATUS_RANK
    }
    profiling = _aggregate_campaign_profiling(sorted_records)
    return {
        "schema_version": _CONTINUATION_CAMPAIGN_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "campaign_root": str(campaign_root),
        "donor_count": len(sorted_records),
        "candidate_count": len(sorted_records),
        "best_candidate": None if not sorted_records else sorted_records[0],
        "status_counts": status_counts,
        "passed_candidate_count": passed_candidate_count,
        "research_grade_candidate_count": research_grade_candidate_count,
        "passthrough_args": passthrough_args,
        "trial_policy": trial_policy,
        "backend": resolve_passthrough_backend(passthrough_args),
        "optimizer_backend": resolve_passthrough_optimizer_backend(passthrough_args),
        "validation_thresholds": validation_thresholds,
        "profiling": profiling,
        "branch_decision": _build_campaign_branch_decision(
            profiling=profiling,
            passed_candidate_count=passed_candidate_count,
            research_grade_candidate_count=research_grade_candidate_count,
            trial_policy=trial_policy,
        ),
        "reports": sorted_records,
    }


def passthrough_has_flag(args: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in args)


def passthrough_value(args: list[str], flag: str) -> str | None:
    for index, token in enumerate(args):
        if token == flag:
            if index + 1 >= len(args):
                raise ValueError(f"Missing value for passthrough flag {flag}.")
            return args[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def resolve_passthrough_backend(args: list[str]) -> str:
    backend = passthrough_value(args, "--backend")
    if backend is None:
        return os.environ.get("SIMSOPT_BACKEND", "cpu")
    return backend


def resolve_passthrough_optimizer_backend(args: list[str]) -> str:
    optimizer_backend = passthrough_value(args, "--optimizer-backend")
    if optimizer_backend is not None:
        return optimizer_backend
    optimizer_backend = os.environ.get("OPTIMIZER_BACKEND")
    if optimizer_backend is not None:
        return optimizer_backend
    return "ondevice" if resolve_passthrough_backend(args) == "jax" else "scipy"


def continuation_uses_target_lane_fast_trials(args: list[str]) -> bool:
    return (
        resolve_passthrough_backend(args) == "jax"
        and resolve_passthrough_optimizer_backend(args) == "ondevice"
        and not passthrough_has_flag(args, "--benchmark-mode")
    )


def append_optional_value_flag(
    command: list[str],
    passthrough_args: list[str],
    *,
    flag: str,
    value: object | None,
) -> None:
    if value is None or passthrough_has_flag(passthrough_args, flag):
        return
    command.extend([flag, str(value)])


def resolve_initial_stage_inputs(
    *,
    initial_stage2_bs_path: str | None,
    initial_warm_start_run_dir: str | None,
) -> tuple[Path | None, Path | None]:
    warm_start_run_dir = (
        None
        if initial_warm_start_run_dir is None
        else Path(initial_warm_start_run_dir).expanduser().resolve()
    )
    if initial_stage2_bs_path is not None:
        stage2_seed_path = Path(initial_stage2_bs_path).expanduser().resolve()
    elif warm_start_run_dir is not None:
        stage2_seed_path = resolve_stage_seed_path(warm_start_run_dir)
    else:
        stage2_seed_path = None
    return stage2_seed_path, warm_start_run_dir


def build_stage_command(
    *,
    python_executable: str,
    passthrough_args: list[str],
    stage: ContinuationStage,
    stage_output_root: Path,
    stage2_seed_path: Path | None,
    warm_start_run_dir: Path | None,
    jax_profile_dir: Path | None,
    use_target_lane_fast_trials: bool,
) -> list[str]:
    command = [
        python_executable,
        str(SINGLE_STAGE_SCRIPT),
        "--output-root",
        str(stage_output_root),
        "--mpol",
        str(stage.mpol),
        "--ntor",
        str(stage.ntor),
        "--nphi",
        str(stage.nphi),
        "--ntheta",
        str(stage.ntheta),
        "--maxiter",
        str(stage.maxiter),
    ]
    append_optional_value_flag(
        command,
        passthrough_args,
        flag="--stage2-bs-path",
        value=stage2_seed_path,
    )
    append_optional_value_flag(
        command,
        passthrough_args,
        flag="--warm-start-run-dir",
        value=warm_start_run_dir,
    )
    append_optional_value_flag(
        command,
        passthrough_args,
        flag="--jax-profile-dir",
        value=jax_profile_dir,
    )
    if stage.minimal_artifacts and not passthrough_has_flag(
        passthrough_args, "--minimal-artifacts"
    ):
        command.append("--minimal-artifacts")
    if use_target_lane_fast_trials:
        append_optional_value_flag(
            command,
            passthrough_args,
            flag="--outer-maxls",
            value=stage.outer_maxls,
        )
        append_optional_value_flag(
            command,
            passthrough_args,
            flag="--maxcor",
            value=stage.maxcor,
        )
        append_optional_value_flag(
            command,
            passthrough_args,
            flag="--initial-step-scale",
            value=stage.initial_step_scale,
        )
        append_optional_value_flag(
            command,
            passthrough_args,
            flag="--initial-step-maxiter",
            value=stage.initial_step_maxiter,
        )
        append_optional_value_flag(
            command,
            passthrough_args,
            flag="--target-lane-boozer-bfgs-tol",
            value=stage.target_lane_boozer_bfgs_tol,
        )
        append_optional_value_flag(
            command,
            passthrough_args,
            flag="--target-lane-boozer-bfgs-maxiter",
            value=stage.target_lane_boozer_bfgs_maxiter,
        )
    command.extend(passthrough_args)
    return command


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-stage optimization through a coarse-to-fine continuation "
            "schedule, chaining previous single-stage outputs as seeds."
        )
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_CONTINUATION_OUTPUT_ROOT),
        help="Directory where the continuation run family will be written.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run identifier. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--campaign-donor-run-dir",
        action="append",
        default=[],
        help=(
            "Optional donor run directory to include in a multi-donor continuation "
            "campaign. May be passed multiple times. When set, the runner creates "
            "one continuation sub-run per donor under campaign-<run-id>."
        ),
    )
    parser.add_argument(
        "--campaign-output-json",
        default=None,
        help=(
            "Optional explicit output path for a multi-donor campaign summary. "
            "Defaults to campaign_summary.json inside the campaign root."
        ),
    )
    parser.add_argument(
        "--jax-profile-dir",
        default=None,
        help=(
            "Optional continuation-level JAX/XProf trace root. Relative paths are "
            "resolved under the continuation run root, and each stage writes to its "
            "own stage-XX-<name> subdirectory."
        ),
    )
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--nphi", type=int, default=255)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--coarse-maxiter", type=int, default=1)
    parser.add_argument("--medium-maxiter", type=int, default=1)
    parser.add_argument("--prefinal-maxiter", type=int, default=2)
    parser.add_argument(
        "--trial-policy",
        choices=CONTINUATION_TRIAL_POLICY_CHOICES,
        default=CONTINUATION_TRIAL_POLICY_VALIDATED_FAST,
        help=(
            "Per-stage target-lane trial-budget policy for non-final JAX/ondevice "
            "stages. 'validated-fast' trims restart/artifact overhead and some "
            "inner trial budgets while still requiring usable accepted-progress "
            "warm starts before promotion."
        ),
    )
    parser.add_argument(
        "--initial-stage2-bs-path",
        default=None,
        help=(
            "Optional initial BiotSavart seed for the first continuation stage. "
            "If omitted, the single-stage driver uses its normal Stage 2 seed "
            "resolution unless --initial-warm-start-run-dir supplies a prior run."
        ),
    )
    parser.add_argument(
        "--initial-warm-start-run-dir",
        default=None,
        help=(
            "Optional prior single-stage run directory to use as the first-stage "
            "surface/iota/G warm start. When --initial-stage2-bs-path is omitted, "
            "its biot_savart_opt.json also becomes the first-stage coil seed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived stage commands without executing them.",
    )
    parser.add_argument(
        "--validate-summary-json",
        default=None,
        help=(
            "Validate an existing continuation_summary.json without re-running "
            "any stages."
        ),
    )
    parser.add_argument(
        "--summarize-run-root",
        default=None,
        help=(
            "Reconstruct continuation_summary.json and continuation_validation.json "
            "from an existing continuation run root without re-running stages."
        ),
    )
    parser.add_argument(
        "--resume-run-root",
        default=None,
        help=(
            "Resume an existing continuation run root by reusing completed stages "
            "and restarting from the first incomplete stage."
        ),
    )
    parser.add_argument(
        "--validation-output-json",
        default=None,
        help=(
            "Optional explicit output path for the validation report. Defaults to "
            "continuation_validation.json next to the summary."
        ),
    )
    parser.add_argument(
        "--max-final-field-error",
        type=float,
        default=None,
        help=(
            "Optional maximum allowed final-stage FIELD_ERROR for continuation "
            "validation."
        ),
    )
    parser.add_argument(
        "--max-final-abs-iota-error",
        type=float,
        default=None,
        help=(
            "Optional maximum allowed |FINAL_IOTA - TARGET_IOTA| for final-stage "
            "continuation validation."
        ),
    )
    parser.add_argument(
        "--max-final-non-qs",
        type=float,
        default=None,
        help=(
            "Optional maximum allowed FINAL_NON_QS for final-stage continuation "
            "validation."
        ),
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help=(
            "Exit with status 1 when the generated continuation validation report "
            "fails."
        ),
    )
    args, passthrough = parser.parse_known_args(argv)
    return args, strip_overridden_passthrough_args(passthrough)


def resolve_validation_output_path(
    *,
    summary_path: Path,
    requested_output_path: str | None,
) -> Path:
    if requested_output_path is None:
        return summary_path.with_name("continuation_validation.json")
    return Path(requested_output_path).expanduser().resolve()


def resolve_campaign_output_path(
    *,
    campaign_root: Path,
    requested_output_path: str | None,
) -> Path:
    if requested_output_path is None:
        return campaign_root / "campaign_summary.json"
    return Path(requested_output_path).expanduser().resolve()


def resolve_single_profiling_report_path(*, summary_path: Path) -> Path:
    return summary_path.with_name("continuation_profiling_report.md")


def resolve_campaign_profiling_report_path(*, campaign_root: Path) -> Path:
    return campaign_root / "campaign_profiling_report.md"


def _format_report_float(value: object) -> str:
    finite_value = _finite_float(value)
    if finite_value is None:
        return "n/a"
    return f"{finite_value:.6g}"


def _format_report_int(value: object) -> str:
    int_value = _safe_int(value)
    if int_value is None:
        return "n/a"
    return str(int_value)


def _append_report_metric(
    lines: list[str],
    label: str,
    value: object,
    *,
    formatter=_format_report_float,
) -> None:
    formatted = formatter(value)
    if formatted == "n/a":
        return
    lines.append(f"- {label}: {formatted}")


def build_continuation_profiling_report_markdown(report: dict[str, object]) -> str:
    profiling = report.get("profiling")
    if not isinstance(profiling, dict):
        profiling = {}
    stage_reports = report.get("stage_reports")
    if not isinstance(stage_reports, list):
        stage_reports = []

    lines = [
        "# Continuation Profiling Report",
        "",
        f"- Run root: {report.get('run_root', 'n/a')}",
        f"- Generated at UTC: {_utc_now_iso()}",
        f"- Validation passed: {bool(report.get('passed') is True)}",
        "",
        "## Aggregate",
    ]
    _append_report_metric(
        lines,
        "Profiled stages",
        profiling.get("profiled_stage_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(
        lines,
        "Target-lane profiled stages",
        profiling.get("target_lane_profiled_stage_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(lines, "Total stage wall time (s)", profiling.get("total_stage_script_time_s"))
    _append_report_metric(lines, "Total outer optimizer time (s)", profiling.get("total_outer_optimizer_s"))
    _append_report_metric(
        lines,
        "Total initial outer phase time (s)",
        profiling.get("total_outer_optimizer_initial_phase_s"),
    )
    _append_report_metric(
        lines,
        "Total main outer phase time (s)",
        profiling.get("total_outer_optimizer_main_s"),
    )
    _append_report_metric(
        lines,
        "Total target-lane bundle setup time (s)",
        profiling.get("total_target_lane_bundle_setup_s"),
    )
    _append_report_metric(
        lines,
        "Total accepted steps",
        profiling.get("total_accepted_step_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(
        lines,
        "Total objective evaluations",
        profiling.get("total_objective_eval_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(
        lines,
        "Objective evaluations per accepted step",
        profiling.get("objective_evals_per_accepted_step"),
    )
    _append_report_metric(
        lines,
        "Total gradient evaluations",
        profiling.get("total_gradient_eval_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(
        lines,
        "Total value-and-grad compile overhead (s)",
        profiling.get("total_value_and_grad_compile_overhead_s"),
    )
    _append_report_metric(
        lines,
        "Total inner-solve compile overhead (s)",
        profiling.get("total_inner_solve_compile_overhead_s"),
    )

    for stage_report in stage_reports:
        if not isinstance(stage_report, dict):
            continue
        stage_name = str(stage_report.get("name", "unknown"))
        stage_profiling = stage_report.get("profiling")
        if not isinstance(stage_profiling, dict):
            stage_profiling = {}
        target_lane_profile = stage_profiling.get("target_lane_profile")
        if not isinstance(target_lane_profile, dict):
            target_lane_profile = {}
        lines.extend(
            [
                "",
                f"## Stage `{stage_name}`",
                f"- Status: {stage_report.get('status', 'unknown')}",
                f"- Passed: {bool(stage_report.get('passed') is True)}",
            ]
        )
        jax_profile_dir = _profile_dir_value(stage_profiling.get("jax_profile_dir"))
        if jax_profile_dir is not None:
            lines.append(f"- JAX profile dir: `{jax_profile_dir}`")
        _append_report_metric(lines, "Stage wall time (s)", stage_profiling.get("script_total_s"))
        _append_report_metric(lines, "Outer optimizer time (s)", stage_profiling.get("outer_optimizer_s"))
        _append_report_metric(
            lines,
            "Initial outer phase time (s)",
            stage_profiling.get("outer_optimizer_initial_phase_s"),
        )
        _append_report_metric(
            lines,
            "Main outer phase time (s)",
            stage_profiling.get("outer_optimizer_main_s"),
        )
        _append_report_metric(
            lines,
            "Target-lane bundle setup time (s)",
            stage_profiling.get("target_lane_bundle_setup_s"),
        )
        _append_report_metric(
            lines,
            "Accepted steps",
            stage_profiling.get("accepted_step_count"),
            formatter=_format_report_int,
        )
        _append_report_metric(
            lines,
            "Initial-phase iterations",
            stage_profiling.get("initial_phase_iterations"),
            formatter=_format_report_int,
        )
        _append_report_metric(
            lines,
            "Objective evaluations",
            stage_profiling.get("objective_eval_count"),
            formatter=_format_report_int,
        )
        _append_report_metric(
            lines,
            "Objective evaluations per accepted step",
            stage_profiling.get("objective_evals_per_accepted_step"),
        )
        _append_report_metric(
            lines,
            "Gradient evaluations",
            stage_profiling.get("gradient_eval_count"),
            formatter=_format_report_int,
        )
        _append_report_metric(
            lines,
            "Value-and-grad compile overhead (s)",
            target_lane_profile.get("value_and_grad_compile_overhead_s"),
        )
        _append_report_metric(
            lines,
            "Inner-solve compile overhead (s)",
            target_lane_profile.get("inner_solve_compile_overhead_s"),
        )
    lines.append("")
    return "\n".join(lines)


def build_campaign_profiling_report_markdown(summary: dict[str, object]) -> str:
    profiling = summary.get("profiling")
    if not isinstance(profiling, dict):
        profiling = {}
    branch_decision = summary.get("branch_decision")
    if not isinstance(branch_decision, dict):
        branch_decision = {}
    reports = summary.get("reports")
    if not isinstance(reports, list):
        reports = []

    lines = [
        "# Campaign Profiling Report",
        "",
        f"- Campaign root: {summary.get('campaign_root', 'n/a')}",
        f"- Run ID: {summary.get('run_id', 'n/a')}",
        f"- Generated at UTC: {_utc_now_iso()}",
        "",
        "## Aggregate",
    ]
    _append_report_metric(
        lines,
        "Profiled candidates",
        profiling.get("profiled_candidate_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(lines, "Total stage wall time (s)", profiling.get("total_stage_script_time_s"))
    _append_report_metric(
        lines,
        "Total outer optimizer time (s)",
        profiling.get("total_outer_optimizer_s"),
    )
    _append_report_metric(
        lines,
        "Total initial outer phase time (s)",
        profiling.get("total_outer_optimizer_initial_phase_s"),
    )
    _append_report_metric(
        lines,
        "Total main outer phase time (s)",
        profiling.get("total_outer_optimizer_main_s"),
    )
    _append_report_metric(
        lines,
        "Total target-lane bundle setup time (s)",
        profiling.get("total_target_lane_bundle_setup_s"),
    )
    _append_report_metric(
        lines,
        "Total accepted steps",
        profiling.get("total_accepted_step_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(
        lines,
        "Total objective evaluations",
        profiling.get("total_objective_eval_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(
        lines,
        "Objective evaluations per accepted step",
        profiling.get("objective_evals_per_accepted_step"),
    )
    _append_report_metric(
        lines,
        "Total gradient evaluations",
        profiling.get("total_gradient_eval_count"),
        formatter=_format_report_int,
    )
    _append_report_metric(
        lines,
        "Total value-and-grad compile overhead (s)",
        profiling.get("total_value_and_grad_compile_overhead_s"),
    )

    if branch_decision:
        lines.extend(
            [
                "",
                "## Branch Decision",
                f"- Category: {branch_decision.get('category', 'n/a')}",
            ]
        )
        for rationale in branch_decision.get("rationale", []):
            lines.append(f"- Rationale: {rationale}")
        for action in branch_decision.get("recommended_actions", []):
            lines.append(f"- Next action: {action}")

    for candidate in reports:
        if not isinstance(candidate, dict):
            continue
        candidate_profiling = candidate.get("profiling")
        if not isinstance(candidate_profiling, dict):
            candidate_profiling = {}
        lines.extend(
            [
                "",
                f"## Donor `{candidate.get('donor_label', 'unknown')}`",
                f"- Status: {candidate.get('status', 'unknown')}",
                f"- Research grade: {bool(candidate.get('research_grade') is True)}",
                f"- Run root: {candidate.get('run_root', 'n/a')}",
            ]
        )
        _append_report_metric(
            lines,
            "Total stage wall time (s)",
            candidate_profiling.get("total_stage_script_time_s"),
        )
        _append_report_metric(
            lines,
            "Total outer optimizer time (s)",
            candidate_profiling.get("total_outer_optimizer_s"),
        )
        _append_report_metric(
            lines,
            "Total initial outer phase time (s)",
            candidate_profiling.get("total_outer_optimizer_initial_phase_s"),
        )
        _append_report_metric(
            lines,
            "Total main outer phase time (s)",
            candidate_profiling.get("total_outer_optimizer_main_s"),
        )
        _append_report_metric(
            lines,
            "Total accepted steps",
            candidate_profiling.get("total_accepted_step_count"),
            formatter=_format_report_int,
        )
        _append_report_metric(
            lines,
            "Total objective evaluations",
            candidate_profiling.get("total_objective_eval_count"),
            formatter=_format_report_int,
        )
        _append_report_metric(
            lines,
            "Objective evaluations per accepted step",
            candidate_profiling.get("objective_evals_per_accepted_step"),
        )
        _append_report_metric(
            lines,
            "Total gradient evaluations",
            candidate_profiling.get("total_gradient_eval_count"),
            formatter=_format_report_int,
        )
        _append_report_metric(
            lines,
            "Total value-and-grad compile overhead (s)",
            candidate_profiling.get("total_value_and_grad_compile_overhead_s"),
        )
    lines.append("")
    return "\n".join(lines)


def write_continuation_profiling_report(
    report: dict[str, object],
    *,
    summary_path: Path,
) -> Path:
    report_path = resolve_single_profiling_report_path(summary_path=summary_path)
    write_text(report_path, build_continuation_profiling_report_markdown(report))
    return report_path


def write_campaign_profiling_report(
    summary: dict[str, object],
    *,
    campaign_root: Path,
) -> Path:
    report_path = resolve_campaign_profiling_report_path(campaign_root=campaign_root)
    write_text(report_path, build_campaign_profiling_report_markdown(summary))
    return report_path


def write_continuation_validation_report(
    summary: dict[str, object],
    *,
    summary_path: Path,
    validation_output_path: str | None,
    max_final_field_error: float | None,
    max_final_abs_iota_error: float | None,
    max_final_non_qs: float | None,
) -> tuple[Path, dict[str, object]]:
    report = build_continuation_validation_report(
        summary,
        max_final_field_error=max_final_field_error,
        max_final_abs_iota_error=max_final_abs_iota_error,
        max_final_non_qs=max_final_non_qs,
    )
    report_path = resolve_validation_output_path(
        summary_path=summary_path,
        requested_output_path=validation_output_path,
    )
    write_json(report_path, report)
    return report_path, report


def persist_continuation_summary(
    summary: dict[str, object],
    *,
    summary_path: Path,
) -> None:
    summary["updated_at_utc"] = _utc_now_iso()
    write_json(summary_path, summary)


def run_single_continuation_with_args(
    args: argparse.Namespace,
    passthrough_args: list[str],
    *,
    forced_run_root: Path | None = None,
    forced_run_mode: str | None = None,
    forced_initial_stage2_seed_path: Path | None = None,
    forced_initial_warm_start_run_dir: Path | None = None,
    forced_jax_profile_root: Path | None = None,
) -> ContinuationRunOutcome:
    mode_flag_count = sum(
        value is not None
        for value in (
            args.validate_summary_json,
            args.summarize_run_root,
            args.resume_run_root,
        )
    )
    if forced_run_mode is None and mode_flag_count > 1:
        raise SystemExit(
            "Pass at most one of --validate-summary-json, --summarize-run-root, "
            "or --resume-run-root."
        )
    if forced_run_mode is None and args.validate_summary_json is not None:
        summary_path = Path(args.validate_summary_json).expanduser().resolve()
        summary = load_json(summary_path)
        report_path, report = write_continuation_validation_report(
            summary,
            summary_path=summary_path,
            validation_output_path=args.validation_output_json,
            max_final_field_error=args.max_final_field_error,
            max_final_abs_iota_error=args.max_final_abs_iota_error,
            max_final_non_qs=args.max_final_non_qs,
        )
        print(f"Wrote continuation validation report to {report_path}")
        profiling_report_path = write_continuation_profiling_report(
            report,
            summary_path=summary_path,
        )
        print(f"Wrote continuation profiling report to {profiling_report_path}")
        return ContinuationRunOutcome(
            run_root=summary_path.parent,
            summary_path=summary_path,
            summary=summary,
            report_path=report_path,
            report=report,
            exit_code=1 if args.strict_validation and not report["passed"] else 0,
        )

    run_mode = forced_run_mode or "new"
    if forced_run_root is not None:
        run_root = forced_run_root.resolve()
        if not args.dry_run:
            run_root.mkdir(parents=True, exist_ok=True)
    elif args.summarize_run_root is not None:
        run_mode = "summarize"
        run_root = Path(args.summarize_run_root).expanduser().resolve()
        if not run_root.exists():
            raise SystemExit(f"Existing continuation run root does not exist: {run_root}")
    elif args.resume_run_root is not None:
        run_mode = "resume"
        run_root = Path(args.resume_run_root).expanduser().resolve()
        if not run_root.exists():
            raise SystemExit(f"Existing continuation run root does not exist: {run_root}")
    else:
        run_id = (
            args.run_id
            if args.run_id is not None
            else time.strftime("%Y%m%d-%H%M%S")
        )
        run_root = Path(args.output_root).resolve() / f"continuation-{run_id}"
        if not args.dry_run:
            run_root.mkdir(parents=True, exist_ok=True)

    use_target_lane_fast_trials = continuation_uses_target_lane_fast_trials(
        passthrough_args
    )
    stages = build_default_continuation_stages(
        final_mpol=args.mpol,
        final_ntor=args.ntor,
        final_nphi=args.nphi,
        final_ntheta=args.ntheta,
        final_maxiter=args.maxiter,
        coarse_maxiter=args.coarse_maxiter,
        medium_maxiter=args.medium_maxiter,
        prefinal_maxiter=args.prefinal_maxiter,
        trial_policy=args.trial_policy,
    )
    initial_stage2_seed_path, initial_warm_start_run_dir = (
        resolve_initial_stage_inputs(
            initial_stage2_bs_path=args.initial_stage2_bs_path,
            initial_warm_start_run_dir=args.initial_warm_start_run_dir,
        )
    )
    if forced_initial_stage2_seed_path is not None:
        initial_stage2_seed_path = forced_initial_stage2_seed_path.resolve()
    if forced_initial_warm_start_run_dir is not None:
        initial_warm_start_run_dir = forced_initial_warm_start_run_dir.resolve()
    jax_profile_root = (
        forced_jax_profile_root.resolve()
        if forced_jax_profile_root is not None
        else resolve_jax_profile_root(
            run_root=run_root,
            requested_profile_dir=args.jax_profile_dir,
        )
    )

    summary: dict[str, object] = {
        "schema_version": _CONTINUATION_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "run_mode": run_mode,
        "run_root": str(run_root),
        "jax_profile_dir": None
        if jax_profile_root is None
        else str(jax_profile_root),
        "stages": [],
        "passthrough_args": passthrough_args,
        "trial_policy": args.trial_policy,
        "use_target_lane_fast_trials": use_target_lane_fast_trials,
        "backend": resolve_passthrough_backend(passthrough_args),
        "optimizer_backend": resolve_passthrough_optimizer_backend(passthrough_args),
        "strict_validation": bool(args.strict_validation),
        "validation_thresholds": {
            "max_final_field_error": args.max_final_field_error,
            "max_final_abs_iota_error": args.max_final_abs_iota_error,
            "max_final_non_qs": args.max_final_non_qs,
        },
        "initial_stage2_bs_path": None
        if initial_stage2_seed_path is None
        else str(initial_stage2_seed_path),
        "initial_warm_start_run_dir": None
        if initial_warm_start_run_dir is None
        else str(initial_warm_start_run_dir),
        "summarize_run_root": None
        if args.summarize_run_root is None
        else str(Path(args.summarize_run_root).expanduser().resolve()),
        "resume_run_root": None
        if args.resume_run_root is None
        else str(Path(args.resume_run_root).expanduser().resolve()),
    }
    summary_path = run_root / "continuation_summary.json"
    report_path = resolve_validation_output_path(
        summary_path=summary_path,
        requested_output_path=args.validation_output_json,
    )
    if not args.dry_run:
        remove_if_exists(report_path)
        persist_continuation_summary(summary, summary_path=summary_path)
    previous_run_dir: Path | None = None
    stage_failure_exit_code: int | None = None

    for stage_index, stage in enumerate(stages, start=1):
        is_final_stage = stage_index == len(stages)
        stage_output_root = build_stage_output_root(
            run_root,
            stage_index=stage_index,
            stage=stage,
        )
        stage_jax_profile_dir = build_stage_jax_profile_dir(
            jax_profile_root,
            stage_index=stage_index,
            stage=stage,
        )
        if previous_run_dir is None:
            stage2_seed_path = initial_stage2_seed_path
            warm_start_run_dir = initial_warm_start_run_dir
        else:
            stage2_seed_path = resolve_stage_seed_path(previous_run_dir)
            warm_start_run_dir = previous_run_dir
        command = build_stage_command(
            python_executable=sys.executable,
            passthrough_args=passthrough_args,
            stage=stage,
            stage_output_root=stage_output_root,
            stage2_seed_path=stage2_seed_path,
            warm_start_run_dir=warm_start_run_dir,
            jax_profile_dir=stage_jax_profile_dir,
            use_target_lane_fast_trials=use_target_lane_fast_trials,
        )
        stage_record = build_stage_record(
            stage=stage,
            stage_output_root=stage_output_root,
            stage2_seed_path=stage2_seed_path,
            warm_start_run_dir=warm_start_run_dir,
            jax_profile_dir=stage_jax_profile_dir,
            command=command,
        )
        if run_mode in {"summarize", "resume"}:
            existing_status, existing_snapshot = inspect_existing_stage_output(
                stage_output_root
            )
            if existing_status == "completed":
                stage_record["status"] = "completed"
                stage_record["subprocess_returncode"] = 0
                stage_record["reused_existing_run"] = True
                stage_record.update(existing_snapshot)
                annotate_existing_stage_jax_profile_dir(
                    stage_record,
                    stage_jax_profile_dir=stage_jax_profile_dir,
                )
                stage_gate = evaluate_continuation_stage(
                    stage_record,
                    is_final_stage=is_final_stage,
                    max_final_field_error=args.max_final_field_error,
                    max_final_abs_iota_error=args.max_final_abs_iota_error,
                    max_final_non_qs=args.max_final_non_qs,
                )
                stage_record["stage_contract"] = stage_gate
                if run_mode == "resume" and (not is_final_stage) and (not stage_gate["passed"]):
                    stage_record["reused_existing_run"] = False
                    stage_record["resume_replaces_existing_status"] = existing_status
                    stage_record["preexisting_stage_snapshot"] = existing_snapshot
                    stage_record["preexisting_stage_validation"] = stage_gate
                else:
                    if run_mode == "summarize":
                        stage_record["reused_existing_run"] = True
                    summary["stages"].append(stage_record)
                    if not args.dry_run:
                        persist_continuation_summary(summary, summary_path=summary_path)
                    run_dir_value = stage_record.get("run_dir")
                    if isinstance(run_dir_value, str) and bool(run_dir_value):
                        previous_run_dir = Path(run_dir_value)
                    continue
            if run_mode == "summarize":
                stage_record["status"] = existing_status
                stage_record["reused_existing_run"] = existing_status != "not_started"
                stage_record.update(existing_snapshot)
                annotate_existing_stage_jax_profile_dir(
                    stage_record,
                    stage_jax_profile_dir=stage_jax_profile_dir,
                )
                summary["stages"].append(stage_record)
                if not args.dry_run:
                    persist_continuation_summary(summary, summary_path=summary_path)
                continue
            if existing_status != "not_started":
                stage_record["resume_replaces_existing_status"] = existing_status
                stage_record["preexisting_stage_snapshot"] = existing_snapshot
                reset_stage_record_for_rerun(stage_record)
        stage_record["status"] = "planned" if not args.dry_run else "dry_run"
        summary["stages"].append(stage_record)
        if not args.dry_run:
            persist_continuation_summary(summary, summary_path=summary_path)
        if args.dry_run:
            continue
        if (
            run_mode == "resume"
            and stage_record.get("resume_replaces_existing_status") is not None
            and stage_output_root.exists()
        ):
            shutil.rmtree(stage_output_root)
        if stage_jax_profile_dir is not None and stage_jax_profile_dir.exists():
            shutil.rmtree(stage_jax_profile_dir)
        stage_output_root.mkdir(parents=True, exist_ok=True)
        stage_record["started_at_utc"] = _utc_now_iso()
        stage_record["status"] = "running"
        persist_continuation_summary(summary, summary_path=summary_path)
        try:
            subprocess.run(command, check=True)
            stage_record["status"] = "completed"
            stage_record["subprocess_returncode"] = 0
        except subprocess.CalledProcessError as exc:
            stage_record["status"] = "subprocess_failed"
            stage_record["subprocess_returncode"] = exc.returncode
            stage_record["failure_kind"] = "subprocess_error"
            stage_record["failure_message"] = f"{type(exc).__name__}: {exc}"
            stage_failure_exit_code = exc.returncode or 1
        except Exception as exc:
            stage_record["status"] = "runner_error"
            stage_record["subprocess_returncode"] = None
            stage_record["failure_kind"] = "runner_error"
            stage_record["failure_message"] = f"{type(exc).__name__}: {exc}"
            stage_failure_exit_code = 1
        finally:
            stage_record["completed_at_utc"] = _utc_now_iso()
            stage_record.update(collect_stage_run_snapshot(stage_output_root))

        if stage_record.get("status") == "completed" and not is_final_stage:
            stage_gate = evaluate_continuation_stage(
                stage_record,
                is_final_stage=False,
                max_final_field_error=args.max_final_field_error,
                max_final_abs_iota_error=args.max_final_abs_iota_error,
                max_final_non_qs=args.max_final_non_qs,
            )
            stage_record["stage_contract"] = stage_gate
            if not stage_gate["passed"]:
                stage_record["failure_kind"] = "stage_contract_failed"
                stage_record["failure_message"] = "; ".join(stage_gate["failures"])
                stage_failure_exit_code = 1

        run_dir_value = stage_record.get("run_dir")
        if (
            stage_record.get("status") == "completed"
            and (
                is_final_stage
                or stage_record.get("failure_kind") != "stage_contract_failed"
            )
            and isinstance(run_dir_value, str)
            and bool(run_dir_value)
        ):
            previous_run_dir = Path(run_dir_value)
        else:
            break

    summary["completed_at_utc"] = _utc_now_iso()
    if not args.dry_run:
        persist_continuation_summary(summary, summary_path=summary_path)
        print(f"Wrote continuation summary to {summary_path}")
    if args.dry_run:
        return ContinuationRunOutcome(
            run_root=run_root,
            summary_path=summary_path,
            summary=summary,
            report_path=None,
            report=None,
            exit_code=0,
        )
    report_path, report = write_continuation_validation_report(
        summary,
        summary_path=summary_path,
        validation_output_path=args.validation_output_json,
        max_final_field_error=args.max_final_field_error,
        max_final_abs_iota_error=args.max_final_abs_iota_error,
        max_final_non_qs=args.max_final_non_qs,
    )
    print(f"Wrote continuation validation report to {report_path}")
    profiling_report_path = write_continuation_profiling_report(
        report,
        summary_path=summary_path,
    )
    print(f"Wrote continuation profiling report to {profiling_report_path}")
    exit_code = 0
    if stage_failure_exit_code is not None:
        exit_code = stage_failure_exit_code
    elif args.strict_validation and not report["passed"]:
        exit_code = 1
    return ContinuationRunOutcome(
        run_root=run_root,
        summary_path=summary_path,
        summary=summary,
        report_path=report_path,
        report=report,
        exit_code=exit_code,
    )


def run_continuation_campaign_with_args(
    args: argparse.Namespace,
    passthrough_args: list[str],
) -> ContinuationRunOutcome:
    if any(
        value is not None
        for value in (
            args.validate_summary_json,
            args.summarize_run_root,
            args.resume_run_root,
        )
    ):
        raise SystemExit(
            "Campaign mode does not support --validate-summary-json, "
            "--summarize-run-root, or --resume-run-root."
        )
    donor_run_dirs = [
        _require_campaign_donor_run_dir(path_value)
        for path_value in args.campaign_donor_run_dir
    ]
    if not donor_run_dirs:
        raise SystemExit(
            "Campaign mode requires at least one --campaign-donor-run-dir."
        )
    run_id = args.run_id if args.run_id is not None else time.strftime("%Y%m%d-%H%M%S")
    campaign_root = Path(args.output_root).resolve() / f"campaign-{run_id}"
    if not args.dry_run:
        campaign_root.mkdir(parents=True, exist_ok=True)
    campaign_jax_profile_root = resolve_jax_profile_root(
        run_root=campaign_root,
        requested_profile_dir=args.jax_profile_dir,
    )
    donor_records: list[dict[str, object]] = []
    for donor_index, donor_run_dir in enumerate(donor_run_dirs, start=1):
        donor_label = (
            f"{donor_index:02d}-"
            f"{_slugify_path_component(donor_run_dir.parent.name)}-"
            f"{_slugify_path_component(donor_run_dir.name)}"
        )
        donor_run_root = campaign_root / f"donor-{donor_label}"
        donor_jax_profile_root = (
            None
            if campaign_jax_profile_root is None
            else campaign_jax_profile_root / donor_label
        )
        outcome = run_single_continuation_with_args(
            args,
            passthrough_args,
            forced_run_root=donor_run_root,
            forced_run_mode="new",
            forced_initial_stage2_seed_path=resolve_stage_seed_path(donor_run_dir),
            forced_initial_warm_start_run_dir=donor_run_dir,
            forced_jax_profile_root=donor_jax_profile_root,
        )
        donor_records.append(
            build_campaign_candidate_record(
                donor_index=donor_index,
                donor_label=donor_label,
                donor_run_dir=donor_run_dir,
                outcome=outcome,
            )
        )

    summary = build_continuation_campaign_summary(
        campaign_root=campaign_root,
        run_id=run_id,
        donor_records=donor_records,
        passthrough_args=passthrough_args,
        trial_policy=args.trial_policy,
        validation_thresholds={
            "max_final_field_error": args.max_final_field_error,
            "max_final_abs_iota_error": args.max_final_abs_iota_error,
            "max_final_non_qs": args.max_final_non_qs,
        },
    )
    summary_path = resolve_campaign_output_path(
        campaign_root=campaign_root,
        requested_output_path=args.campaign_output_json,
    )
    if not args.dry_run:
        write_json(summary_path, summary)
        print(f"Wrote continuation campaign summary to {summary_path}")
        profiling_report_path = write_campaign_profiling_report(
            summary,
            campaign_root=campaign_root,
        )
        print(f"Wrote continuation campaign profiling report to {profiling_report_path}")

    best_candidate = summary.get("best_candidate")
    exit_code = 0
    if not donor_records:
        exit_code = 1
    elif args.strict_validation and not (
        isinstance(best_candidate, dict) and best_candidate.get("passed") is True
    ):
        exit_code = 1
    elif all(record.get("exit_code") not in (0, None) for record in donor_records):
        exit_code = 1

    return ContinuationRunOutcome(
        run_root=campaign_root,
        summary_path=summary_path,
        summary=summary,
        report_path=None,
        report=None,
        exit_code=exit_code,
    )


def main(argv: list[str] | None = None) -> None:
    args, passthrough_args = parse_args(argv)
    if args.campaign_donor_run_dir:
        outcome = run_continuation_campaign_with_args(args, passthrough_args)
    else:
        outcome = run_single_continuation_with_args(args, passthrough_args)
    if outcome.exit_code != 0:
        raise SystemExit(outcome.exit_code)


if __name__ == "__main__":
    main()
