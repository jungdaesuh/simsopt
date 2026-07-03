"""Tests for the K1 matrix artifact report helper."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks import single_stage_k1_matrix_report as matrix_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_case(
    artifact_dir: Path,
    label: str,
    *,
    exit_code: int | None,
    assertion_exit_code: int | None = None,
    progress_payload: dict[str, object] | None = None,
) -> Path:
    case_dir = artifact_dir / label
    case_dir.mkdir(parents=True)
    _write_json(case_dir / "case_manifest.json", {"label": label})
    if exit_code is not None:
        (case_dir / "exit_status.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    if assertion_exit_code is not None:
        (case_dir / "k1_progress_assertions_exit_status.txt").write_text(
            f"{assertion_exit_code}\n",
            encoding="utf-8",
        )
    if progress_payload is not None:
        _write_json(
            case_dir / "cases" / "mpol=10-ntor=10-test" / "outer_optimizer_progress.json",
            progress_payload,
        )
    return case_dir


def _write_case_progress(
    case_dir: Path,
    output_label: str,
    progress_payload: dict[str, object],
) -> None:
    _write_json(
        case_dir
        / "cases"
        / output_label
        / "mpol=10-ntor=10-test"
        / "outer_optimizer_progress.json",
        progress_payload,
    )


def _write_case_progress_ndjson(
    case_dir: Path,
    output_label: str,
    progress_events: list[dict[str, object]],
) -> None:
    progress_json = (
        case_dir
        / "cases"
        / output_label
        / "mpol=10-ntor=10-test"
        / "outer_optimizer_progress.json"
    )
    events_path = progress_json.with_name(f"{progress_json.name}.events.ndjson")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in progress_events),
        encoding="utf-8",
    )
    _write_json(
        progress_json,
        {
            "current_event": progress_events[-1]["label"],
            "event_count": len(progress_events),
            "events_format": "ndjson",
            "events_path": events_path.name,
            "events": [],
        },
    )


def _progress_payload() -> dict[str, object]:
    return {
        "events": [
            {
                "label": "objective_evaluation",
                "event_index": 1,
            },
            {
                "label": "target_lane_decomposed_k1_forward_returned",
                "event_index": 2,
                "evaluation_index": 1,
                "newton_polish_skipped": True,
                "newton_iter": 0,
                "pre_newton_iter": 4,
                "pre_newton_nfev": 8,
                "pre_newton_ngev": 5,
                "pre_newton_line_search_status": 0,
            },
            {
                "label": (
                    "target_lane_optimizer_objective_eval_1_forward_result_reused"
                ),
                "line_search_evaluation": 1,
            },
            {
                "label": "target_lane_reporting_forward_result_started",
                "reused": True,
            },
            {
                "label": "target_lane_reporting_forward_result_returned",
                "reused": True,
            },
        ]
    }


def test_matrix_report_summarizes_successful_case(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_case(
        artifact_dir,
        "dense_skip_chunk8",
        exit_code=0,
        assertion_exit_code=0,
        progress_payload=_progress_payload(),
    )
    _write_json(artifact_dir / "matrix_exit_summary.json", {"dense_skip_chunk8": 0})

    report = matrix_report.build_report(artifact_dir)

    assert report["case_count"] == 1
    assert report["failed_cases"] == []
    case = report["cases"][0]
    assert case["label"] == "dense_skip_chunk8"
    assert case["issues"] == []
    progress = case["progress"]
    assert progress["objective_evaluation_count"] == 1
    assert progress["k1_forward_returned_count"] == 1
    assert progress["k1_newton_iter_max"] == 0
    assert progress["k1_newton_polish_skipped_all"] is True
    assert progress["k1_pre_newton_metrics_present_count"] == 1
    assert progress["trace_forward_result"] == {"reused": 1, "recomputed": 0}
    assert progress["final_sync_forward_result"]["both_reused"] is True
    assert case["progresses"] == [progress]


def test_matrix_report_summarizes_reference_and_target_progress_files(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    case_dir = _write_case(
        artifact_dir,
        "dense_skip_chunk8",
        exit_code=0,
        assertion_exit_code=0,
    )
    _write_case_progress(case_dir, "reference_outputs", _progress_payload())
    _write_case_progress(case_dir, "target_outputs", _progress_payload())
    _write_json(artifact_dir / "matrix_exit_summary.json", {"dense_skip_chunk8": 0})

    report = matrix_report.build_report(artifact_dir)

    assert report["failed_cases"] == []
    case = report["cases"][0]
    assert case["issues"] == []
    assert case["progress"]["label"] == "target_outputs"
    assert [progress["label"] for progress in case["progresses"]] == [
        "reference_outputs",
        "target_outputs",
    ]


def test_matrix_report_reads_ndjson_progress_sidecar(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    case_dir = _write_case(
        artifact_dir,
        "dense_skip_chunk8",
        exit_code=0,
        assertion_exit_code=0,
    )
    _write_case_progress_ndjson(
        case_dir,
        "target_outputs",
        list(_progress_payload()["events"]),
    )
    _write_json(artifact_dir / "matrix_exit_summary.json", {"dense_skip_chunk8": 0})

    report = matrix_report.build_report(artifact_dir)

    assert report["failed_cases"] == []
    progress = report["cases"][0]["progress"]
    assert progress["objective_evaluation_count"] == 1
    assert progress["k1_forward_returned_count"] == 1
    assert progress["trace_forward_result"] == {"reused": 1, "recomputed": 0}


def test_matrix_report_records_failed_and_missing_progress_cases(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_case(artifact_dir, "failed_case", exit_code=7)
    _write_case(artifact_dir, "missing_progress", exit_code=0)

    report = matrix_report.build_report(artifact_dir)

    assert report["case_count"] == 2
    assert report["failed_cases"] == ["failed_case", "missing_progress"]
    failed_case, missing_progress = report["cases"]
    assert failed_case["issues"] == ["case exited nonzero: 7"]
    assert missing_progress["issues"] == ["missing outer_optimizer_progress.json"]


def test_matrix_report_ignores_skipped_case_without_manifest(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    skipped_dir = artifact_dir / "dense_run_chunk16"
    skipped_dir.mkdir(parents=True)
    (skipped_dir / "skip_reason.txt").write_text(
        "MATRIX_CASES=dense_skip_chunk8\n",
        encoding="utf-8",
    )

    report = matrix_report.build_report(artifact_dir)

    assert report["case_count"] == 0
    assert report["failed_cases"] == []
