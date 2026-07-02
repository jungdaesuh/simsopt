#!/usr/bin/env python3
"""Validate single-stage K1 progress events for the GPU perf-gap plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class ProgressAssertionError(AssertionError):
    """Progress telemetry does not satisfy the requested K1 contract."""


def load_events(progress_json: Path) -> list[dict[str, object]]:
    payload = json.loads(progress_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProgressAssertionError(f"Progress JSON is not an object: {progress_json}")
    events = payload.get("events")
    if not isinstance(events, list):
        raise ProgressAssertionError(f"Progress JSON has no events list: {progress_json}")
    normalized = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ProgressAssertionError(
                f"Progress event {index} is not an object: {progress_json}"
            )
        normalized.append(event)
    return normalized


def _label(event: dict[str, object]) -> str:
    value = event.get("label")
    if not isinstance(value, str):
        raise ProgressAssertionError(f"Progress event has non-string label: {event}")
    return value


def _event_int(event: dict[str, object], key: str) -> int:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgressAssertionError(f"Event {_label(event)!r} lacks integer {key}.")
    return int(value)


def _event_bool(event: dict[str, object], key: str) -> bool:
    value = event.get(key)
    if not isinstance(value, bool):
        raise ProgressAssertionError(f"Event {_label(event)!r} lacks boolean {key}.")
    return bool(value)


def require_k1_per_objective(events: list[dict[str, object]]) -> None:
    objective_events = [
        event for event in events if _label(event) == "objective_evaluation"
    ]
    k1_returned = [
        event
        for event in events
        if _label(event) == "target_lane_decomposed_k1_forward_returned"
    ]
    if not objective_events:
        raise ProgressAssertionError("No objective_evaluation events were recorded.")
    if len(k1_returned) != len(objective_events):
        raise ProgressAssertionError(
            "Expected exactly one target_lane_decomposed_k1_forward_returned "
            "event per objective_evaluation; "
            f"got k1={len(k1_returned)} objective={len(objective_events)}."
        )
    indices = [_event_int(event, "evaluation_index") for event in k1_returned]
    if len(set(indices)) != len(indices):
        raise ProgressAssertionError(
            f"K1 returned events have duplicate evaluation_index values: {indices}"
        )


def require_k1_forward_events(events: list[dict[str, object]]) -> None:
    k1_returned = [
        event
        for event in events
        if _label(event) == "target_lane_decomposed_k1_forward_returned"
    ]
    if not k1_returned:
        raise ProgressAssertionError("No K1 returned events were recorded.")
    indices = [_event_int(event, "evaluation_index") for event in k1_returned]
    if len(set(indices)) != len(indices):
        raise ProgressAssertionError(
            f"K1 returned events have duplicate evaluation_index values: {indices}"
        )


def require_newton_skip(
    events: list[dict[str, object]],
    *,
    expected: bool,
) -> None:
    k1_returned = [
        event
        for event in events
        if _label(event) == "target_lane_decomposed_k1_forward_returned"
    ]
    if not k1_returned:
        raise ProgressAssertionError("No K1 returned events were recorded.")
    mismatches = [
        _event_int(event, "evaluation_index")
        for event in k1_returned
        if _event_bool(event, "newton_polish_skipped") is not expected
    ]
    if mismatches:
        raise ProgressAssertionError(
            f"K1 events with unexpected newton_polish_skipped={not expected}: "
            f"{mismatches}"
        )
    if expected:
        nonzero_newton = [
            _event_int(event, "evaluation_index")
            for event in k1_returned
            if _event_int(event, "newton_iter") != 0
        ]
        if nonzero_newton:
            raise ProgressAssertionError(
                "Trial Newton polish was marked skipped, but newton_iter was "
                f"nonzero for evaluations: {nonzero_newton}"
            )


def require_trace_forward_reuse(events: list[dict[str, object]]) -> None:
    reused_by_line_search: dict[int, int] = {}
    started_by_line_search: dict[int, int] = {}
    for event in events:
        label = _label(event)
        if not (
            label.startswith("target_lane_optimizer_objective_eval_")
            and (
                label.endswith("_forward_result_reused")
                or label.endswith("_forward_result_started")
            )
        ):
            continue
        line_search_evaluation = _event_int(event, "line_search_evaluation")
        if label.endswith("_forward_result_reused"):
            reused_by_line_search[line_search_evaluation] = (
                reused_by_line_search.get(line_search_evaluation, 0) + 1
            )
        else:
            started_by_line_search[line_search_evaluation] = (
                started_by_line_search.get(line_search_evaluation, 0) + 1
            )
    if not reused_by_line_search:
        raise ProgressAssertionError(
            "No target_lane_optimizer_objective_eval_*_forward_result_reused "
            "events were recorded."
        )
    overlap = sorted(set(reused_by_line_search) & set(started_by_line_search))
    if overlap:
        raise ProgressAssertionError(
            "Forward result was both reused and recomputed for line-search "
            f"evaluations: {overlap}"
        )


def require_final_sync_reuse(events: list[dict[str, object]]) -> None:
    required_labels = {
        "target_lane_reporting_forward_result_started",
        "target_lane_reporting_forward_result_returned",
    }
    sync_events = [event for event in events if _label(event) in required_labels]
    seen_labels = {_label(event) for event in sync_events}
    missing_labels = sorted(required_labels - seen_labels)
    if missing_labels:
        raise ProgressAssertionError(
            "Missing target_lane_reporting_forward_result_* events: "
            f"{missing_labels}"
        )
    not_reused = [
        _label(event) for event in sync_events if not _event_bool(event, "reused")
    ]
    if not_reused:
        raise ProgressAssertionError(
            f"Final reporting forward result did not reuse cache: {not_reused}"
        )


def summarize(events: list[dict[str, object]]) -> dict[str, int]:
    labels = [_label(event) for event in events]
    return {
        "event_count": len(events),
        "objective_evaluation_count": labels.count("objective_evaluation"),
        "k1_forward_returned_count": labels.count(
            "target_lane_decomposed_k1_forward_returned"
        ),
        "reporting_forward_started_count": labels.count(
            "target_lane_reporting_forward_result_started"
        ),
        "reporting_forward_returned_count": labels.count(
            "target_lane_reporting_forward_result_returned"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--require-k1-forward-events", action="store_true")
    parser.add_argument("--require-k1-per-objective", action="store_true")
    parser.add_argument(
        "--expect-newton-skip",
        choices=("true", "false"),
        default=None,
    )
    parser.add_argument("--require-trace-forward-reuse", action="store_true")
    parser.add_argument("--require-final-sync-reuse", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    events = load_events(args.progress_json)
    if args.require_k1_forward_events:
        require_k1_forward_events(events)
    if args.require_k1_per_objective:
        require_k1_per_objective(events)
    if args.expect_newton_skip is not None:
        require_newton_skip(events, expected=args.expect_newton_skip == "true")
    if args.require_trace_forward_reuse:
        require_trace_forward_reuse(events)
    if args.require_final_sync_reuse:
        require_final_sync_reuse(events)

    payload = summarize(events)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
