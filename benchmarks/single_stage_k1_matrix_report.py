#!/usr/bin/env python3
"""Summarize K1 matrix artifacts into one machine-checkable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks import single_stage_k1_progress_assertions as k1_assertions


K1_RETURNED_LABEL = "target_lane_decomposed_k1_forward_returned"
TRACE_REUSED_SUFFIX = "_forward_result_reused"
TRACE_STARTED_SUFFIX = "_forward_result_started"
TRACE_PREFIX = "target_lane_optimizer_objective_eval_"
REPORTING_STARTED_LABEL = "target_lane_reporting_forward_result_started"
REPORTING_RETURNED_LABEL = "target_lane_reporting_forward_result_returned"
PRE_NEWTON_FIELDS = (
    "pre_newton_iter",
    "pre_newton_nfev",
    "pre_newton_ngev",
    "pre_newton_line_search_status",
)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_exit_code(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if text == "":
        raise ValueError(f"Exit-code file is empty: {path}")
    return int(text)


def _event_label(event: dict[str, object]) -> str:
    value = event.get("label")
    if not isinstance(value, str):
        raise ValueError(f"Progress event has non-string label: {event}")
    return value


def _event_bool_or_none(event: dict[str, object], key: str) -> bool | None:
    value = event.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"Progress event {_event_label(event)!r} has non-bool {key}")
    return value


def _event_int_or_none(event: dict[str, object], key: str) -> int | None:
    value = event.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Progress event {_event_label(event)!r} has non-int {key}")
    return value


def _find_progress_json(case_dir: Path) -> Path | None:
    case_roots = [case_dir / "cases"]
    progress_files: list[Path] = []
    for case_root in case_roots:
        if case_root.exists():
            progress_files.extend(sorted(case_root.rglob("outer_optimizer_progress.json")))
    if not progress_files:
        return None
    if len(progress_files) > 1:
        raise ValueError(
            f"Expected one outer_optimizer_progress.json under {case_dir}, "
            f"found {len(progress_files)}"
        )
    return progress_files[0]


def _trace_reuse_counts(events: list[dict[str, object]]) -> dict[str, int]:
    reused = 0
    recomputed = 0
    for event in events:
        label = _event_label(event)
        if not label.startswith(TRACE_PREFIX):
            continue
        if label.endswith(TRACE_REUSED_SUFFIX):
            reused += 1
        elif label.endswith(TRACE_STARTED_SUFFIX):
            recomputed += 1
    return {"reused": reused, "recomputed": recomputed}


def _reporting_reuse(events: list[dict[str, object]]) -> dict[str, object]:
    reporting_events = {
        _event_label(event): event
        for event in events
        if _event_label(event) in {REPORTING_STARTED_LABEL, REPORTING_RETURNED_LABEL}
    }
    started_event = reporting_events.get(REPORTING_STARTED_LABEL)
    returned_event = reporting_events.get(REPORTING_RETURNED_LABEL)
    started_reused = (
        None if started_event is None else _event_bool_or_none(started_event, "reused")
    )
    returned_reused = (
        None if returned_event is None else _event_bool_or_none(returned_event, "reused")
    )
    return {
        "started_present": started_event is not None,
        "returned_present": returned_event is not None,
        "started_reused": started_reused,
        "returned_reused": returned_reused,
        "both_reused": started_reused is True and returned_reused is True,
    }


def _progress_report(progress_json: Path) -> dict[str, object]:
    events = k1_assertions.load_events(progress_json)
    summary = k1_assertions.summarize(events)
    k1_events = [event for event in events if _event_label(event) == K1_RETURNED_LABEL]
    newton_iters = [
        value
        for event in k1_events
        for value in [_event_int_or_none(event, "newton_iter")]
        if value is not None
    ]
    newton_skipped = [
        value
        for event in k1_events
        for value in [_event_bool_or_none(event, "newton_polish_skipped")]
        if value is not None
    ]
    pre_newton_metrics_count = sum(
        1 for event in k1_events if all(field in event for field in PRE_NEWTON_FIELDS)
    )
    return {
        **summary,
        "path": str(progress_json),
        "k1_newton_iter_max": max(newton_iters) if newton_iters else None,
        "k1_newton_polish_skipped_all": (
            all(newton_skipped) if newton_skipped else None
        ),
        "k1_pre_newton_metrics_present_count": pre_newton_metrics_count,
        "trace_forward_result": _trace_reuse_counts(events),
        "final_sync_forward_result": _reporting_reuse(events),
    }


def summarize_case(case_dir: Path) -> dict[str, object]:
    issues: list[str] = []
    manifest_path = case_dir / "case_manifest.json"
    exit_code = _read_exit_code(case_dir / "exit_status.txt")
    assertion_exit_code = _read_exit_code(
        case_dir / "k1_progress_assertions_exit_status.txt"
    )
    summary_path = case_dir / "summary.json"
    progress_json = _find_progress_json(case_dir)

    manifest = _load_json_object(manifest_path) if manifest_path.exists() else None
    if manifest is None:
        issues.append("missing case_manifest.json")
    if exit_code is None and not (case_dir / "skip_reason.txt").exists():
        issues.append("missing exit_status.txt")
    if exit_code not in {None, 0}:
        issues.append(f"case exited nonzero: {exit_code}")
    if assertion_exit_code not in {None, 0}:
        issues.append(f"k1 progress assertions exited nonzero: {assertion_exit_code}")
    if progress_json is None and exit_code == 0:
        issues.append("missing outer_optimizer_progress.json")

    return {
        "label": case_dir.name,
        "case_dir": str(case_dir),
        "manifest": manifest,
        "exit_code": exit_code,
        "assertion_exit_code": assertion_exit_code,
        "skipped": (case_dir / "skip_reason.txt").exists(),
        "summary_present": summary_path.exists(),
        "progress": None if progress_json is None else _progress_report(progress_json),
        "issues": issues,
    }


def build_report(artifact_dir: Path) -> dict[str, object]:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
    case_dirs = sorted(
        path
        for path in artifact_dir.iterdir()
        if path.is_dir() and (path / "case_manifest.json").exists()
    )
    cases = [summarize_case(case_dir) for case_dir in case_dirs]
    matrix_exit_summary_path = artifact_dir / "matrix_exit_summary.json"
    matrix_exit_summary = (
        _load_json_object(matrix_exit_summary_path)
        if matrix_exit_summary_path.exists()
        else None
    )
    failed_cases = [
        str(case["label"])
        for case in cases
        if case["exit_code"] not in {None, 0}
        or case["assertion_exit_code"] not in {None, 0}
        or bool(case["issues"])
    ]
    return {
        "artifact_dir": str(artifact_dir),
        "case_count": len(cases),
        "failed_cases": failed_cases,
        "matrix_exit_summary": matrix_exit_summary,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = build_report(args.artifact_dir)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
