from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


_STAGE2_SEED_REPORT_SCHEMA_VERSION = 1
_REQUIRED_RESULTS_FILENAMES = (
    "results.json",
    "biot_savart_opt.json",
    "surf_opt.json",
)
_OPTIONAL_LEGACY_FILENAMES = (
    "surf_opt.vts",
    "curves_opt.vtu",
    "CrossSectionPlot.png",
    "NormFieldPlot.png",
    "MagFieldPlot.png",
    "VV.vts",
)
_FINITE_METRIC_KEYS = (
    "FIELD_ERROR",
    "FINAL_CURVE_LENGTH",
    "FINAL_CC_DISTANCE",
    "MAX_CURVATURE",
    "FINAL_OBJECTIVE",
)
_STATUS_RANK = {
    "research_grade": 0,
    "eligible": 1,
    "salvageable": 2,
    "rejected": 3,
}


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def load_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def detect_stage2_seed_artifacts(run_dir: Path) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "required": {
            filename: {
                "path": str(run_dir / filename),
                "exists": (run_dir / filename).exists(),
            }
            for filename in _REQUIRED_RESULTS_FILENAMES
        },
        "legacy_optional": {
            filename: {
                "path": str(run_dir / filename),
                "exists": (run_dir / filename).exists(),
            }
            for filename in _OPTIONAL_LEGACY_FILENAMES
        },
    }


def _all_required_artifacts_present(artifacts: dict[str, object]) -> bool:
    required = artifacts.get("required", {})
    if not isinstance(required, dict):
        return False
    return all(
        isinstance(entry, dict) and bool(entry.get("exists"))
        for entry in required.values()
    )


def _finite_metric(results: dict[str, object], key: str) -> float | None:
    return _finite_float(results.get(key))


def _rejected_stage2_seed_candidate(
    run_dir: Path,
    *,
    artifacts: dict[str, object],
    failures: list[str],
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "status": "rejected",
        "downstream_eligible": False,
        "research_grade": False,
        "artifacts": artifacts,
        "metrics": {},
        "margins": {},
        "ranking": {
            "status_rank": _STATUS_RANK["rejected"],
            "sort_key": [_STATUS_RANK["rejected"], math.inf, math.inf, math.inf],
        },
        "failures": failures,
        "warnings": [] if warnings is None else warnings,
    }


def evaluate_stage2_seed_candidate(
    run_dir: Path,
    *,
    max_field_error: float | None = None,
) -> dict[str, object]:
    artifacts = detect_stage2_seed_artifacts(run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return _rejected_stage2_seed_candidate(
            run_dir,
            artifacts=artifacts,
            failures=["results.json is missing"],
        )

    try:
        results = load_json(results_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _rejected_stage2_seed_candidate(
            run_dir,
            artifacts=artifacts,
            failures=[f"results.json is unreadable: {type(exc).__name__}: {exc}"],
        )
    failures: list[str] = []
    warnings: list[str] = []

    field_error = _finite_metric(results, "FIELD_ERROR")
    final_curve_length = _finite_metric(results, "FINAL_CURVE_LENGTH")
    final_cc_distance = _finite_metric(results, "FINAL_CC_DISTANCE")
    max_curvature = _finite_metric(results, "MAX_CURVATURE")
    final_objective = _finite_metric(results, "FINAL_OBJECTIVE")
    length_target = _finite_metric(results, "LENGTH_TARGET")
    cc_threshold = _finite_metric(results, "CC_THRESHOLD")
    curvature_threshold = _finite_metric(results, "CURVATURE_THRESHOLD")

    for key in _FINITE_METRIC_KEYS:
        if _finite_metric(results, key) is None:
            failures.append(f"{key} is missing or non-finite")

    required_artifacts_present = _all_required_artifacts_present(artifacts)
    if not required_artifacts_present:
        failures.append("restart artifacts are incomplete")
        if artifacts["legacy_optional"]["surf_opt.vts"]["exists"]:
            warnings.append(
                "legacy surf_opt.vts exists but restartable surf_opt.json is missing"
            )

    hardware_ok = results.get("HARDWARE_CONSTRAINTS_OK") is True
    if not hardware_ok:
        failures.append("hardware constraints did not pass")
    self_intersecting = results.get("SELF_INTERSECTING") is True
    if self_intersecting:
        failures.append("banana curve is self-intersecting")

    if max_field_error is not None:
        if field_error is None:
            failures.append("FIELD_ERROR is missing or non-finite")
        elif field_error > max_field_error:
            failures.append(
                f"FIELD_ERROR {field_error:.6g} exceeds threshold {max_field_error:.6g}"
            )

    if results.get("OPTIMIZER_SUCCESS") is not True:
        warnings.append(
            f"optimizer did not report success: {results.get('TERMINATION_MESSAGE')}"
        )

    hardware_violations = results.get("HARDWARE_CONSTRAINT_VIOLATIONS")
    if isinstance(hardware_violations, list) and hardware_violations:
        warnings.extend(
            f"hardware_violation: {message}" for message in hardware_violations
        )

    length_margin = (
        None
        if final_curve_length is None or length_target is None
        else float(length_target - final_curve_length)
    )
    coil_coil_margin = (
        None
        if final_cc_distance is None or cc_threshold is None
        else float(final_cc_distance - cc_threshold)
    )
    curvature_margin = (
        None
        if max_curvature is None or curvature_threshold is None
        else float(curvature_threshold - max_curvature)
    )

    downstream_eligible = not failures
    research_grade = downstream_eligible and not warnings
    finite_metrics_available = all(
        _finite_metric(results, key) is not None for key in _FINITE_METRIC_KEYS
    )
    if research_grade:
        status = "research_grade"
    elif downstream_eligible:
        status = "eligible"
    elif finite_metrics_available:
        status = "salvageable"
    else:
        status = "rejected"

    sort_key = [
        _STATUS_RANK[status],
        math.inf if field_error is None else field_error,
        math.inf if coil_coil_margin is None else -coil_coil_margin,
        math.inf if curvature_margin is None else -curvature_margin,
        math.inf if final_objective is None else final_objective,
    ]
    return {
        "run_dir": str(run_dir),
        "status": status,
        "downstream_eligible": downstream_eligible,
        "research_grade": research_grade,
        "artifacts": artifacts,
        "metrics": {
            "FIELD_ERROR": field_error,
            "FINAL_OBJECTIVE": final_objective,
            "FINAL_CURVE_LENGTH": final_curve_length,
            "FINAL_CC_DISTANCE": final_cc_distance,
            "MAX_CURVATURE": max_curvature,
            "LENGTH_TARGET": length_target,
            "CC_THRESHOLD": cc_threshold,
            "CURVATURE_THRESHOLD": curvature_threshold,
            "OPTIMIZER_SUCCESS": results.get("OPTIMIZER_SUCCESS"),
            "TERMINATION_MESSAGE": results.get("TERMINATION_MESSAGE"),
            "HARDWARE_CONSTRAINTS_OK": results.get("HARDWARE_CONSTRAINTS_OK"),
        },
        "margins": {
            "length_margin": length_margin,
            "coil_coil_margin": coil_coil_margin,
            "curvature_margin": curvature_margin,
        },
        "ranking": {
            "status_rank": _STATUS_RANK[status],
            "sort_key": sort_key,
        },
        "failures": failures,
        "warnings": warnings,
    }


def find_stage2_run_dirs(scan_root: Path) -> list[Path]:
    return sorted({path.parent for path in scan_root.rglob("results.json")})


def build_stage2_seed_catalog(
    scan_root: Path,
    *,
    max_field_error: float | None = None,
) -> dict[str, object]:
    run_dirs = find_stage2_run_dirs(scan_root)
    reports = [
        evaluate_stage2_seed_candidate(run_dir, max_field_error=max_field_error)
        for run_dir in run_dirs
    ]
    reports.sort(key=lambda report: tuple(report["ranking"]["sort_key"]))
    best_candidate = reports[0] if reports else None
    eligible_count = sum(1 for report in reports if report["downstream_eligible"])
    research_grade_count = sum(1 for report in reports if report["research_grade"])
    return {
        "schema_version": _STAGE2_SEED_REPORT_SCHEMA_VERSION,
        "scan_root": str(scan_root),
        "validation_config": {
            "max_field_error": max_field_error,
            "required_restart_artifacts": list(_REQUIRED_RESULTS_FILENAMES),
        },
        "candidate_count": len(reports),
        "eligible_count": eligible_count,
        "research_grade_count": research_grade_count,
        "best_candidate": best_candidate,
        "reports": reports,
        "passed": eligible_count > 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Stage 2 output folders, validate donor-seed restartability, "
            "and emit a ranked candidate catalog for downstream single-stage use."
        )
    )
    parser.add_argument(
        "--scan-root",
        required=True,
        help="Root directory to scan recursively for Stage 2 results.json files.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Destination path for the ranked Stage 2 seed catalog JSON.",
    )
    parser.add_argument(
        "--max-field-error",
        type=float,
        default=None,
        help="Optional hard field-error threshold for downstream seed eligibility.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit with status 1 when no downstream-eligible candidate is found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_root = Path(args.scan_root).expanduser().resolve()
    catalog = build_stage2_seed_catalog(
        scan_root,
        max_field_error=args.max_field_error,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    write_json(output_path, catalog)
    print(f"Wrote Stage 2 seed catalog to {output_path}")
    if args.require_pass and not catalog["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
