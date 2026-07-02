"""Tests for single-stage K1 progress assertion helpers."""

import pytest

from benchmarks import single_stage_k1_progress_assertions as k1_assertions


def _objective_event(index: int) -> dict[str, object]:
    return {
        "label": "objective_evaluation",
        "event_index": index,
    }


def _k1_event(
    index: int,
    *,
    skipped: bool,
    newton_iter: int,
) -> dict[str, object]:
    return {
        "label": "target_lane_decomposed_k1_forward_returned",
        "event_index": index,
        "evaluation_index": index,
        "newton_polish_skipped": skipped,
        "newton_iter": newton_iter,
    }


def test_k1_assertions_accept_one_k1_per_objective_with_skipped_newton() -> None:
    events = [
        _objective_event(1),
        _k1_event(1, skipped=True, newton_iter=0),
        _objective_event(2),
        _k1_event(2, skipped=True, newton_iter=0),
    ]

    k1_assertions.require_k1_per_objective(events)
    k1_assertions.require_newton_skip(events, expected=True)


def test_k1_assertions_accept_k1_forward_events_without_objective_trace() -> None:
    events = [
        _k1_event(1, skipped=True, newton_iter=0),
        _k1_event(2, skipped=True, newton_iter=0),
    ]

    k1_assertions.require_k1_forward_events(events)
    k1_assertions.require_newton_skip(events, expected=True)


def test_k1_assertions_reject_no_k1_forward_events() -> None:
    with pytest.raises(k1_assertions.ProgressAssertionError, match="No K1"):
        k1_assertions.require_k1_forward_events([_objective_event(1)])


def test_k1_assertions_reject_duplicate_k1_for_one_objective() -> None:
    events = [
        _objective_event(1),
        _k1_event(1, skipped=True, newton_iter=0),
        _k1_event(2, skipped=True, newton_iter=0),
    ]

    with pytest.raises(k1_assertions.ProgressAssertionError, match="per objective"):
        k1_assertions.require_k1_per_objective(events)


def test_k1_assertions_reject_skipped_newton_with_nonzero_iter() -> None:
    events = [
        _objective_event(1),
        _k1_event(1, skipped=True, newton_iter=3),
    ]

    with pytest.raises(k1_assertions.ProgressAssertionError, match="nonzero"):
        k1_assertions.require_newton_skip(events, expected=True)


def test_k1_assertions_accept_trace_reuse_without_same_eval_recompute() -> None:
    events = [
        {
            "label": "target_lane_optimizer_objective_eval_1_forward_result_reused",
            "line_search_evaluation": 1,
        }
    ]

    k1_assertions.require_trace_forward_reuse(events)


def test_k1_assertions_reject_trace_reuse_and_recompute_same_eval() -> None:
    events = [
        {
            "label": "target_lane_optimizer_objective_eval_1_forward_result_reused",
            "line_search_evaluation": 1,
        },
        {
            "label": "target_lane_optimizer_objective_eval_1_forward_result_started",
            "line_search_evaluation": 1,
        },
    ]

    with pytest.raises(k1_assertions.ProgressAssertionError, match="both reused"):
        k1_assertions.require_trace_forward_reuse(events)


def test_k1_assertions_accept_final_sync_reuse() -> None:
    events = [
        {
            "label": "target_lane_reporting_forward_result_started",
            "reused": True,
        },
        {
            "label": "target_lane_reporting_forward_result_returned",
            "reused": True,
        },
    ]

    k1_assertions.require_final_sync_reuse(events)


def test_k1_assertions_reject_incomplete_final_sync_reuse() -> None:
    events = [
        {
            "label": "target_lane_reporting_forward_result_started",
            "reused": True,
        }
    ]

    with pytest.raises(k1_assertions.ProgressAssertionError, match="Missing"):
        k1_assertions.require_final_sync_reuse(events)
