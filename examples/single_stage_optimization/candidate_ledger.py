from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "STAGE_2"))
sys.path.insert(0, str(SCRIPT_DIR / "SINGLE_STAGE"))

from run_single_stage_continuation import (  # noqa: E402
    build_continuation_validation_report,
    load_json as load_continuation_json,
)
from stage2_seed_report import (  # noqa: E402
    build_stage2_seed_catalog,
)


_LEDGER_SCHEMA_VERSION = 1
_SINGLE_STAGE_STATUS_RANK = {
    "research_grade": 0,
    "eligible": 1,
    "salvageable": 2,
    "rejected": 3,
}
_SINGLE_STAGE_SCHEDULE_STAGE_KEYS = (
    "name",
    "maxiter",
    "minimal_artifacts",
    "outer_maxls",
    "maxcor",
    "initial_step_scale",
    "initial_step_maxiter",
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        float_value = float(value)
        if math.isfinite(float_value) and float_value.is_integer():
            return int(float_value)
    return None


def _ratio_or_none(
    numerator: int | float | None,
    denominator: int | float | None,
) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _path_or_none(value: object) -> Path | None:
    if isinstance(value, str) and value:
        return Path(value).expanduser().resolve()
    return None


def _hardware_margin(
    measured_value: float | None,
    threshold: float | None,
) -> float | None:
    if measured_value is None or threshold is None:
        return None
    return float(measured_value - threshold)


def _curvature_margin(
    max_curvature: float | None,
    threshold: float | None,
) -> float | None:
    if max_curvature is None or threshold is None:
        return None
    return float(threshold - max_curvature)


def _min_available(values: list[float | None]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return min(finite_values)


def _extract_continuation_schedule(
    summary: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(summary, dict):
        return {}
    stages_value = summary.get("stages")
    schedule_stages: list[dict[str, object]] = []
    if isinstance(stages_value, list):
        for stage_value in stages_value:
            if not isinstance(stage_value, dict):
                continue
            stage_summary = {
                key: stage_value.get(key) for key in _SINGLE_STAGE_SCHEDULE_STAGE_KEYS
            }
            shape_value = stage_value.get("shape")
            if isinstance(shape_value, dict):
                stage_summary["shape"] = {
                    key: shape_value.get(key)
                    for key in ("mpol", "ntor", "nphi", "ntheta")
                }
            schedule_stages.append(stage_summary)
    return {
        "run_mode": summary.get("run_mode"),
        "trial_policy": summary.get("trial_policy"),
        "backend": summary.get("backend"),
        "optimizer_backend": summary.get("optimizer_backend"),
        "use_target_lane_fast_trials": summary.get("use_target_lane_fast_trials"),
        "stages": schedule_stages,
    }


def build_campaign_context_map(scan_root: Path) -> dict[str, dict[str, object]]:
    contexts: dict[str, dict[str, object]] = {}
    for campaign_summary_path in scan_root.rglob("campaign_summary.json"):
        campaign_summary = load_continuation_json(campaign_summary_path)
        reports = campaign_summary.get("reports")
        if not isinstance(reports, list):
            continue
        best_candidate = campaign_summary.get("best_candidate")
        best_run_root = None
        if isinstance(best_candidate, dict):
            best_run_root_path = _path_or_none(best_candidate.get("run_root"))
            if best_run_root_path is not None:
                best_run_root = str(best_run_root_path)
        for candidate_rank, report in enumerate(reports, start=1):
            if not isinstance(report, dict):
                continue
            run_root_path = _path_or_none(report.get("run_root"))
            if run_root_path is None:
                continue
            run_root_key = str(run_root_path)
            contexts[run_root_key] = {
                "campaign_root": str(campaign_summary_path.parent.resolve()),
                "campaign_summary_path": str(campaign_summary_path.resolve()),
                "run_id": campaign_summary.get("run_id"),
                "trial_policy": campaign_summary.get("trial_policy"),
                "backend": campaign_summary.get("backend"),
                "optimizer_backend": campaign_summary.get("optimizer_backend"),
                "candidate_rank": candidate_rank,
                "candidate_count": campaign_summary.get("candidate_count"),
                "status_counts": campaign_summary.get("status_counts"),
                "best_candidate": run_root_key == best_run_root,
                "donor_index": report.get("donor_index"),
                "donor_label": report.get("donor_label"),
                "status": report.get("status"),
                "research_grade": report.get("research_grade"),
            }
    return contexts


def _single_stage_best_candidate_reason(
    reports: list[dict[str, object]],
) -> dict[str, object] | None:
    if not reports:
        return None
    best_report = reports[0]
    best_metrics = best_report.get("metrics")
    if not isinstance(best_metrics, dict):
        best_metrics = {}
    best_ranking = best_report.get("ranking")
    if not isinstance(best_ranking, dict):
        best_ranking = {}
    best_margins = best_report.get("margins")
    if not isinstance(best_margins, dict):
        best_margins = {}
    reasons: list[str] = [
        f"best status rank ({best_report.get('status')})",
    ]
    evidence = {
        "status_rank": best_ranking.get("status_rank"),
        "field_error": best_metrics.get("FIELD_ERROR"),
        "abs_iota_error": best_metrics.get("ABS_IOTA_ERROR"),
        "final_non_qs": best_metrics.get("FINAL_NON_QS"),
        "minimum_hardware_margin": best_margins.get("minimum_hardware_margin"),
        "objective_evals_per_accepted_step": best_metrics.get(
            "OBJECTIVE_EVALS_PER_ACCEPTED_STEP"
        ),
    }
    for key, label, prefer_smaller in (
        ("FIELD_ERROR", "lowest field error", True),
        ("ABS_IOTA_ERROR", "closest iota target", True),
        ("FINAL_NON_QS", "lowest non-QS", True),
        ("FINAL_BOOZER_RESIDUAL", "lowest Boozer residual", True),
        ("OBJECTIVE_EVALS_PER_ACCEPTED_STEP", "best objective-eval efficiency", True),
        ("minimum_hardware_margin", "largest hardware margin", False),
    ):
        best_value = (
            best_metrics.get(key)
            if key != "minimum_hardware_margin"
            else best_margins.get(key)
        )
        finite_best_value = _finite_float(best_value)
        if finite_best_value is None:
            continue
        candidate_values: list[float] = []
        for report in reports:
            report_metrics = report.get("metrics")
            if not isinstance(report_metrics, dict):
                report_metrics = {}
            report_margins = report.get("margins")
            if not isinstance(report_margins, dict):
                report_margins = {}
            maybe_value = (
                report_metrics.get(key)
                if key != "minimum_hardware_margin"
                else report_margins.get(key)
            )
            finite_value = _finite_float(maybe_value)
            if finite_value is not None:
                candidate_values.append(finite_value)
        if not candidate_values:
            continue
        target_value = min(candidate_values) if prefer_smaller else max(candidate_values)
        if math.isclose(
            finite_best_value,
            target_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            reasons.append(label)
    return {
        "run_root": best_report.get("run_root"),
        "reasons": reasons,
        "evidence": evidence,
    }


def _stage2_best_candidate_reason(
    reports: list[dict[str, object]],
) -> dict[str, object] | None:
    if not reports:
        return None
    best_report = reports[0]
    best_metrics = best_report.get("metrics")
    if not isinstance(best_metrics, dict):
        best_metrics = {}
    best_margins = best_report.get("margins")
    if not isinstance(best_margins, dict):
        best_margins = {}
    return {
        "run_dir": best_report.get("run_dir"),
        "reasons": [
            f"best status rank ({best_report.get('status')})",
            "lowest field error",
        ],
        "evidence": {
            "field_error": best_metrics.get("FIELD_ERROR"),
            "final_objective": best_metrics.get("FINAL_OBJECTIVE"),
            "coil_coil_margin": best_margins.get("coil_coil_margin"),
            "curvature_margin": best_margins.get("curvature_margin"),
        },
    }


def _single_stage_status(report: dict[str, object]) -> tuple[str, bool]:
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


def evaluate_single_stage_validation(
    run_root: Path,
    report: dict[str, object],
    *,
    summary: dict[str, object] | None,
    campaign_context: dict[str, object] | None,
) -> dict[str, object]:
    final_stage = report.get("final_stage")
    if not isinstance(final_stage, dict):
        final_stage = {}
    metrics = final_stage.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    status, research_grade = _single_stage_status(report)
    field_error = _finite_float(metrics.get("FIELD_ERROR"))
    abs_iota_error = _finite_float(final_stage.get("abs_iota_error"))
    final_non_qs = _finite_float(metrics.get("FINAL_NON_QS"))
    final_boozer_residual = _finite_float(metrics.get("FINAL_BOOZER_RESIDUAL"))
    max_curvature = _finite_float(metrics.get("MAX_CURVATURE"))
    curve_curve_min_dist = _finite_float(metrics.get("CURVE_CURVE_MIN_DIST"))
    curve_surface_min_dist = _finite_float(metrics.get("CURVE_SURFACE_MIN_DIST"))
    surface_vessel_min_dist = _finite_float(metrics.get("SURFACE_VESSEL_MIN_DIST"))
    cc_dist = _finite_float(metrics.get("CC_DIST"))
    cs_dist = _finite_float(metrics.get("CS_DIST"))
    ss_dist = _finite_float(metrics.get("SS_DIST"))
    curvature_threshold = _finite_float(metrics.get("CURVATURE_THRESHOLD"))
    curve_curve_margin = _hardware_margin(curve_curve_min_dist, cc_dist)
    curve_surface_margin = _hardware_margin(curve_surface_min_dist, cs_dist)
    surface_vessel_margin = _hardware_margin(surface_vessel_min_dist, ss_dist)
    curvature_margin = _curvature_margin(max_curvature, curvature_threshold)
    minimum_hardware_margin = _min_available(
        [
            curve_curve_margin,
            curve_surface_margin,
            surface_vessel_margin,
            curvature_margin,
        ]
    )
    profiling = report.get("profiling")
    if not isinstance(profiling, dict):
        profiling = {}
    accepted_step_count = _safe_int(profiling.get("total_accepted_step_count"))
    objective_eval_count = _safe_int(profiling.get("total_objective_eval_count"))
    research_usable = status in {"research_grade", "eligible"}
    report_payload = {
        "run_root": str(run_root),
        "status": status,
        "research_grade": research_grade,
        "research_usable": research_usable,
        "passed": report.get("passed") is True,
        "research_verdicts": report.get("research_verdicts", {}),
        "campaign": {} if campaign_context is None else campaign_context,
        "continuation": _extract_continuation_schedule(summary),
        "profiling": profiling,
        "metrics": {
            "FIELD_ERROR": field_error,
            "ABS_IOTA_ERROR": abs_iota_error,
            "FINAL_IOTA": _finite_float(metrics.get("FINAL_IOTA")),
            "FINAL_NON_QS": final_non_qs,
            "FINAL_BOOZER_RESIDUAL": final_boozer_residual,
            "MAX_CURVATURE": max_curvature,
            "CURVE_CURVE_MIN_DIST": curve_curve_min_dist,
            "CURVE_SURFACE_MIN_DIST": curve_surface_min_dist,
            "SURFACE_VESSEL_MIN_DIST": surface_vessel_min_dist,
            "OBJECTIVE_EVAL_COUNT": objective_eval_count,
            "ACCEPTED_STEP_COUNT": accepted_step_count,
            "OBJECTIVE_EVALS_PER_ACCEPTED_STEP": _ratio_or_none(
                objective_eval_count,
                accepted_step_count,
            ),
        },
        "margins": {
            "curve_curve_margin": curve_curve_margin,
            "curve_surface_margin": curve_surface_margin,
            "surface_vessel_margin": surface_vessel_margin,
            "curvature_margin": curvature_margin,
            "minimum_hardware_margin": minimum_hardware_margin,
        },
        "ranking": {
            "status_rank": _SINGLE_STAGE_STATUS_RANK[status],
            "sort_key": [
                _SINGLE_STAGE_STATUS_RANK[status],
                math.inf if field_error is None else field_error,
                math.inf if abs_iota_error is None else abs_iota_error,
                math.inf if final_non_qs is None else final_non_qs,
                math.inf if final_boozer_residual is None else final_boozer_residual,
                math.inf if minimum_hardware_margin is None else -minimum_hardware_margin,
            ],
        },
        "failures": list(report.get("failures", [])),
        "warnings": list(report.get("warnings", [])),
    }
    return report_payload


def load_or_build_continuation_validation(
    run_root: Path,
    *,
    max_final_field_error: float | None,
    max_final_abs_iota_error: float | None,
    max_final_non_qs: float | None,
) -> dict[str, object] | None:
    validation_path = run_root / "continuation_validation.json"
    if validation_path.exists():
        return load_continuation_json(validation_path)
    summary_path = run_root / "continuation_summary.json"
    if not summary_path.exists():
        return None
    summary = load_continuation_json(summary_path)
    return build_continuation_validation_report(
        summary,
        max_final_field_error=max_final_field_error,
        max_final_abs_iota_error=max_final_abs_iota_error,
        max_final_non_qs=max_final_non_qs,
    )


def load_continuation_summary(run_root: Path) -> dict[str, object] | None:
    summary_path = run_root / "continuation_summary.json"
    if not summary_path.exists():
        return None
    return load_continuation_json(summary_path)


def find_continuation_run_roots(scan_root: Path) -> list[Path]:
    roots = {path.parent for path in scan_root.rglob("continuation_validation.json")}
    roots.update({path.parent for path in scan_root.rglob("continuation_summary.json")})
    return sorted(roots)


def build_candidate_ledger(
    *,
    stage2_root: Path,
    single_stage_root: Path,
    stage2_max_field_error: float | None,
    single_stage_max_final_field_error: float | None,
    single_stage_max_final_abs_iota_error: float | None,
    single_stage_max_final_non_qs: float | None,
) -> dict[str, object]:
    stage2_catalog = build_stage2_seed_catalog(
        stage2_root,
        max_field_error=stage2_max_field_error,
    )
    campaign_context_map = build_campaign_context_map(single_stage_root)
    single_stage_reports = []
    for run_root in find_continuation_run_roots(single_stage_root):
        summary = load_continuation_summary(run_root)
        report = load_or_build_continuation_validation(
            run_root,
            max_final_field_error=single_stage_max_final_field_error,
            max_final_abs_iota_error=single_stage_max_final_abs_iota_error,
            max_final_non_qs=single_stage_max_final_non_qs,
        )
        if report is None:
            continue
        single_stage_reports.append(
            evaluate_single_stage_validation(
                run_root,
                report,
                summary=summary,
                campaign_context=campaign_context_map.get(str(run_root.resolve())),
            )
        )
    single_stage_reports.sort(key=lambda report: tuple(report["ranking"]["sort_key"]))
    single_stage_status_counts = {
        status: sum(report["status"] == status for report in single_stage_reports)
        for status in _SINGLE_STAGE_STATUS_RANK
    }
    return {
        "schema_version": _LEDGER_SCHEMA_VERSION,
        "stage2": stage2_catalog,
        "single_stage": {
            "candidate_count": len(single_stage_reports),
            "research_usable_count": sum(
                report["research_usable"] is True for report in single_stage_reports
            ),
            "status_counts": single_stage_status_counts,
            "best_candidate": (
                None if not single_stage_reports else single_stage_reports[0]
            ),
            "best_candidate_reason": _single_stage_best_candidate_reason(
                single_stage_reports
            ),
            "reports": single_stage_reports,
        },
        "cross_workflow_summary": {
            "best_stage2_seed_reason": _stage2_best_candidate_reason(
                stage2_catalog.get("reports", [])
                if isinstance(stage2_catalog.get("reports"), list)
                else []
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ranked cross-workflow candidate ledger from Stage 2 seed "
            "outputs and single-stage continuation validation reports."
        )
    )
    parser.add_argument("--stage2-root", required=True)
    parser.add_argument("--single-stage-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stage2-max-field-error", type=float, default=None)
    parser.add_argument("--single-stage-max-final-field-error", type=float, default=None)
    parser.add_argument(
        "--single-stage-max-final-abs-iota-error",
        type=float,
        default=None,
    )
    parser.add_argument("--single-stage-max-final-non-qs", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_candidate_ledger(
        stage2_root=Path(args.stage2_root).expanduser().resolve(),
        single_stage_root=Path(args.single_stage_root).expanduser().resolve(),
        stage2_max_field_error=args.stage2_max_field_error,
        single_stage_max_final_field_error=args.single_stage_max_final_field_error,
        single_stage_max_final_abs_iota_error=args.single_stage_max_final_abs_iota_error,
        single_stage_max_final_non_qs=args.single_stage_max_final_non_qs,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    write_json(output_path, payload)
    print(f"Wrote candidate ledger to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
