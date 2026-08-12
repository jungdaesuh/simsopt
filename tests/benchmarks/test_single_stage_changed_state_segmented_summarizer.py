from __future__ import annotations

from dataclasses import replace

import pytest
from benchmarks.single_stage_changed_state_gpu_timeline_receipt import (
    evaluation_ids_sha256,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    SEGMENTED_SUMMARY_SCHEMA_ID,
    Interval,
    TraceSummaryError,
    combine_segmented_trace_summaries,
    summarize_segmented_trace_document,
)
from simsopt_jax.runtime.trace_annotations import (
    EvaluationKind,
    EvaluationTraceContext,
    HostEvent,
    HostEventRecord,
    PhaseId,
)

_HOST_PID = 1
_DEVICE_PID = 2


def _metadata(pid: int, name: str) -> dict[str, object]:
    return {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}}


def _span(
    name: str,
    timestamp_us: float,
    duration_us: float,
    *,
    pid: int = _HOST_PID,
    args: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "ph": "X",
        "pid": pid,
        "tid": 1,
        "ts": timestamp_us,
        "dur": duration_us,
        "name": name,
        "args": {} if args is None else args,
    }


def _lifecycle_args(
    evaluation_id: str, parameter: str, iteration: int
) -> dict[str, object]:
    return {
        "evaluation_id": evaluation_id,
        "parameter_sha256": parameter,
        "evaluation_kind": EvaluationKind.TRIAL.value,
        "outer_iteration_id": str(iteration),
    }


def _host_group(
    evaluation_id: str,
    parameter: str,
    kind: EvaluationKind,
    iteration: int | None,
    timestamps_ns: tuple[int, int, int],
) -> tuple[HostEventRecord, ...]:
    context = EvaluationTraceContext(evaluation_id, parameter, kind, iteration)
    return tuple(
        HostEventRecord(index, event, timestamps_ns[index], context, ())
        for index, event in enumerate(HostEvent)
    )


def _renumber(records: tuple[HostEventRecord, ...]) -> tuple[HostEventRecord, ...]:
    return tuple(
        HostEventRecord(
            sequence,
            record.event,
            record.timestamp_ns,
            record.evaluation,
            record.attributes,
        )
        for sequence, record in enumerate(records)
    )


def _fixture() -> tuple[dict[str, object], tuple[HostEventRecord, ...]]:
    trial_a = _lifecycle_args("trial-1a", "b" * 64, 1)
    trial_b = _lifecycle_args("trial-1b", "c" * 64, 1)
    events = [
        _metadata(_DEVICE_PID, "/device:GPU:0"),
        _metadata(_HOST_PID, "/host:CPU"),
        _span("optimizer.accepted_iteration", 1000.0, 500.0, args={"step_num": "1"}),
        *(
            _span(f"optimizer.lifecycle.{event.value}", timestamp, 0.001, args=trial_a)
            for event, timestamp in zip(
                HostEvent, (1040.0, 1100.0, 1200.0), strict=True
            )
        ),
        *(
            _span(f"optimizer.lifecycle.{event.value}", timestamp, 0.001, args=trial_b)
            for event, timestamp in zip(
                HostEvent, (1250.0, 1350.0, 1450.0), strict=True
            )
        ),
        {},
    ]
    host_events = _renumber(
        (
            *_host_group(
                "initial",
                "a" * 64,
                EvaluationKind.INITIAL,
                None,
                (5_800_000, 5_900_000, 5_950_000),
            ),
            *_host_group(
                "trial-1a",
                "b" * 64,
                EvaluationKind.TRIAL,
                1,
                (6_040_000, 6_100_000, 6_200_000),
            ),
            *_host_group(
                "trial-1b",
                "c" * 64,
                EvaluationKind.TRIAL,
                1,
                (6_250_000, 6_350_000, 6_450_000),
            ),
            *_host_group(
                "trial-2",
                "d" * 64,
                EvaluationKind.TRIAL,
                2,
                (6_600_000, 6_700_000, 6_800_000),
            ),
        )
    )
    return {
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": True},
        "traceEvents": events,
    }, host_events


def _summary():
    document, host_events = _fixture()
    return summarize_segmented_trace_document(
        document,
        host_events,
        child_id="profile-0",
        sample_id="iteration-01",
        accepted_iteration=1,
        profiler_boundary_pauses=(Interval(5_970_000, 6_000_000),),
    )


def test_segment_uses_full_host_audit_and_subtracts_exact_boundary_pause() -> None:
    summary = _summary()

    assert summary.accepted_iteration == 1
    assert summary.raw_active_ns == 140_000
    assert summary.profiler_boundary_pause_ns == 30_000
    assert summary.iteration.host_control_gap_ns == 110_000
    assert summary.iteration.active_ns == 110_000
    assert summary.segment_evaluation_ids_sha256 == evaluation_ids_sha256(
        ("trial-1a", "trial-1b")
    )
    assert summary.raw_active_ns == (
        summary.iteration.active_ns + summary.profiler_boundary_pause_ns
    )


def test_segment_rejects_adjacent_iteration_lifecycle_leakage() -> None:
    document, host_events = _fixture()
    events = document["traceEvents"]
    assert isinstance(events, list)
    args = _lifecycle_args("trial-2", "d" * 64, 2)
    events[-1:-1] = [
        _span(f"optimizer.lifecycle.{event.value}", timestamp, 0.001, args=args)
        for event, timestamp in zip(HostEvent, (1460.0, 1470.0, 1480.0), strict=True)
    ]

    with pytest.raises(TraceSummaryError, match="adjacent or non-trial"):
        summarize_segmented_trace_document(
            document,
            host_events,
            child_id="profile-0",
            sample_id="iteration-01",
            accepted_iteration=1,
        )


def test_segment_rejects_missing_target_evaluation_lifecycle() -> None:
    document, host_events = _fixture()
    events = document["traceEvents"]
    assert isinstance(events, list)
    events[:] = [
        event
        for event in events
        if not (
            isinstance(event.get("args"), dict)
            and event["args"].get("evaluation_id") == "trial-1b"
        )
    ]

    with pytest.raises(TraceSummaryError, match="bijective ordered match"):
        summarize_segmented_trace_document(
            document,
            host_events,
            child_id="profile-0",
            sample_id="iteration-01",
            accepted_iteration=1,
        )


@pytest.mark.parametrize("step_num", ["2", None])
def test_segment_rejects_wrong_or_missing_target_step(step_num: str | None) -> None:
    document, host_events = _fixture()
    events = document["traceEvents"]
    assert isinstance(events, list)
    step = next(
        event for event in events if event.get("name") == "optimizer.accepted_iteration"
    )
    args = step["args"]
    assert isinstance(args, dict)
    if step_num is None:
        events.insert(-1, dict(step))
    else:
        args["step_num"] = step_num

    with pytest.raises(TraceSummaryError, match="iteration"):
        summarize_segmented_trace_document(
            document,
            host_events,
            child_id="profile-0",
            sample_id="iteration-01",
            accepted_iteration=1,
        )


def test_segment_rejects_device_interval_outside_target_step() -> None:
    document, host_events = _fixture()
    events = document["traceEvents"]
    assert isinstance(events, list)
    events.insert(
        -1,
        _span(
            "outside",
            1510.0,
            10.0,
            pid=_DEVICE_PID,
            args={
                "context_id": "1",
                "correlation_id": "1",
                "hlo_module": "jit(f)",
                "hlo_op": PhaseId.NEWTON_RESIDUAL_JVP.value,
                "kernel_details": "regs:16",
                "scope_range_id": "1",
                "tf_op": PhaseId.NEWTON_RESIDUAL_JVP.value,
            },
        ),
    )

    with pytest.raises(TraceSummaryError, match="outside the target step"):
        summarize_segmented_trace_document(
            document,
            host_events,
            child_id="profile-0",
            sample_id="iteration-01",
            accepted_iteration=1,
        )


def test_segment_rejects_overlapping_or_non_gap_boundary_pauses() -> None:
    document, host_events = _fixture()
    with pytest.raises(TraceSummaryError, match="intervals overlap"):
        summarize_segmented_trace_document(
            document,
            host_events,
            child_id="profile-0",
            sample_id="iteration-01",
            accepted_iteration=1,
            profiler_boundary_pauses=(
                Interval(5_970_000, 6_010_000),
                Interval(5_990_000, 6_020_000),
            ),
        )
    with pytest.raises(TraceSummaryError, match="contained in exactly one"):
        summarize_segmented_trace_document(
            document,
            host_events,
            child_id="profile-0",
            sample_id="iteration-01",
            accepted_iteration=1,
            profiler_boundary_pauses=(Interval(6_100_000, 6_110_000),),
        )


def test_combine_requires_exact_targets_and_sums_independent_durations() -> None:
    first = _summary()
    segments = tuple(
        replace(
            first,
            accepted_iteration=iteration,
            sample_id=f"iteration-{iteration:02d}",
            iteration=replace(first.iteration, iteration=iteration),
        )
        for iteration in range(1, 8)
    )

    combined = combine_segmented_trace_summaries(segments)

    assert combined.schema_id == SEGMENTED_SUMMARY_SCHEMA_ID
    assert tuple(item.iteration for item in combined.iterations) == tuple(range(1, 8))
    assert combined.device_active_ns == 7 * first.device_active_ns
    assert sum(item.active_ns for item in combined.iterations) == 7 * 110_000
    with pytest.raises(TraceSummaryError, match="expected"):
        combine_segmented_trace_summaries(segments[:-1])
    with pytest.raises(TraceSummaryError, match="duplicate"):
        combine_segmented_trace_summaries((*segments[:-1], segments[0]))
    with pytest.raises(TraceSummaryError, match="payload differs"):
        combine_segmented_trace_summaries(
            (
                replace(
                    segments[0], iteration=replace(segments[0].iteration, iteration=2)
                ),
                *segments[1:],
            )
        )
