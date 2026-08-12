"""Deterministically summarize a JAX changed-state GPU profiler trace.

The parser owns the supported JAX 0.10.0 Chrome-trace representation. Device
intervals stay in the profiler clock, while exclusive host control gaps stay in
``perf_counter_ns``. The two are combined only after lifecycle points establish
a bijective causal correlation; their timestamps are never directly unioned.
"""

from __future__ import annotations

import gzip
import heapq
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Final

from simsopt_jax.runtime.trace_annotations import (
    EvaluationKind,
    HostEvent,
    HostEventRecord,
    PhaseId,
)

from benchmarks.single_stage_changed_state_gpu_timeline_receipt import (
    evaluation_ids_sha256,
)

SUMMARY_SCHEMA_ID: Final = "single-stage-changed-state-gpu-timeline-summary-v1"
SEGMENT_SUMMARY_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-segment-summary-v2"
)
SEGMENTED_SUMMARY_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-summary-v2"
)
TRACE_SCHEMA_ID: Final = "jax-profiler-chrome-trace-jax-0.10.0-v1"
_ACCEPTED_ITERATION_EVENT: Final = "optimizer.accepted_iteration"
_LIFECYCLE_EVENT_PREFIX: Final = "optimizer.lifecycle."
_DEVICE_PROCESS_PATTERN: Final = re.compile(r"/device:GPU:[0-9]+\Z")
_MEMCPY_DETAILS_PATTERN: Final = re.compile(
    r"kind_src:(?P<src>[^ ]+) kind_dst:(?P<dst>[^ ]+) "
    r"size:(?P<size>[0-9]+) dest:(?P<dest>[0-9]+) async:(?P<async>[01])\Z"
)
_DRILLDOWN_PHASES: Final = frozenset(
    {PhaseId.BIOTSAVART_FORWARD, PhaseId.BIOTSAVART_VJP}
)
_HOST_TRANSFER_PHASES: Final = frozenset(
    {PhaseId.HOST_H2D_SUBMIT, PhaseId.HOST_D2H_MATERIALIZE}
)
_HOST_TRANSFER_PHASE_VALUES: Final = frozenset(
    phase.value for phase in _HOST_TRANSFER_PHASES
)
_NEWTON_ADJOINT_PHASES: Final = frozenset(
    phase
    for phase in PhaseId
    if phase.value.startswith("newton.") or phase.value.startswith("adjoint.")
)
_REQUIRED_DEVICE_PHASES: Final = frozenset(
    {
        PhaseId.HOST_H2D_SUBMIT,
        PhaseId.HOST_D2H_MATERIALIZE,
        PhaseId.NEWTON_RESIDUAL_JVP,
        PhaseId.NEWTON_LINEAR_SOLVE,
        PhaseId.ADJOINT_OUTER_VJP_RHS,
        PhaseId.ADJOINT_DENSE_MATRIX,
        PhaseId.ADJOINT_LU_FACTOR,
        PhaseId.ADJOINT_LU_SOLVE,
        PhaseId.ADJOINT_REFINEMENT,
        PhaseId.ADJOINT_IMPLICIT_COIL_VJP,
        PhaseId.BIOTSAVART_FORWARD,
        PhaseId.BIOTSAVART_VJP,
    }
)
_PHASE_BY_VALUE: Final = {phase.value: phase for phase in PhaseId}
_HLO_MODULE_PHASE: Final = {
    "jit_biotsavart_forward": PhaseId.BIOTSAVART_FORWARD,
    "jit_biotsavart_vjp": PhaseId.BIOTSAVART_VJP,
}
_TIMELINE_NAMESPACES: Final = (
    "host.",
    "optimizer.",
    "newton.",
    "adjoint.",
    "biotsavart.",
)
_LIFECYCLE_EVENT_NAMES: Final = frozenset(
    f"{_LIFECYCLE_EVENT_PREFIX}{event.value}" for event in HostEvent
)
_ALLOWED_TIMELINE_EVENT_NAMES: Final = frozenset(
    {*_PHASE_BY_VALUE, _ACCEPTED_ITERATION_EVENT, *_LIFECYCLE_EVENT_NAMES}
)
_METADATA_EVENT_NAMES: Final = frozenset(
    {"process_name", "process_sort_index", "thread_name", "thread_sort_index"}
)
_COMPILATION_EVENT_NAMES: Final = frozenset(
    {
        "$compiler.py:308 backend_compile_and_load",
        "$pxla.py:1216 compile",
        "CompileModuleToLlvmIr",
        "CompileSingleModule",
        "CompileToBackendResult",
        "PJRT_Client_Compile",
    }
)
_COMPILE_AND_LOAD_EVENT_PATTERN: Final = re.compile(
    r"PjRtCApiClient::CompileAndLoad\([^()]+\)\Z"
)
_TIMELINE_LIKE_SEGMENT_PATTERN: Final = re.compile(
    r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_.]*\Z"
)
_KERNEL_REQUIRED_ARGUMENTS: Final = frozenset(
    {
        "context_id",
        "correlation_id",
        "hlo_module",
        "hlo_op",
        "kernel_details",
        "scope_range_id",
        "tf_op",
    }
)
_MEMCPY_REQUIRED_ARGUMENTS: Final = frozenset(
    {"context_id", "correlation_id", "memcpy_details"}
)
_CLOCK_RATE_RELATIVE_TOLERANCE: Final = Decimal("0.01")
_CLOCK_ANCHOR_RESIDUAL_FLOOR_NS: Final = 100_000
_ADJOINT_SEMANTIC_COUNT_FIELDS: Final = (
    "dense_materializations",
    "lu_factorizations",
    "lu_solves",
    "refinement_corrections",
    "adjoint_executions",
)


class TraceSummaryError(ValueError):
    """Fail-closed trace or lifecycle-record rejection with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, order=True)
class Interval:
    """A half-open interval expressed in integer nanoseconds."""

    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise TraceSummaryError(
                "invalid_interval",
                f"expected 0 <= start < end, got [{self.start_ns}, {self.end_ns})",
            )

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def contains(self, other: Interval) -> bool:
        return self.start_ns <= other.start_ns and other.end_ns <= self.end_ns

    def overlaps(self, other: Interval) -> bool:
        return self.start_ns < other.end_ns and other.start_ns < self.end_ns


@dataclass(frozen=True)
class ScopeAttribution:
    """Deepest unique phase and its non-drill-down critical-path category."""

    phase: PhaseId | None
    critical_phase: PhaseId | None
    ambiguous: bool


@dataclass(frozen=True)
class _TraceSpan:
    interval: Interval
    name: str
    args: Mapping[str, object]
    pid: int
    tid: int


@dataclass(frozen=True)
class _DeviceInterval:
    interval: Interval
    kind: str
    attribution: ScopeAttribution


@dataclass(frozen=True)
class _LifecyclePoint:
    event: HostEvent
    timestamp_ns: int
    evaluation_id: str
    parameter_sha256: str
    evaluation_kind: EvaluationKind
    outer_iteration_id: int | None
    annotation_duration_ns: int


@dataclass(frozen=True)
class _SemanticIterationCounts:
    counts: tuple[tuple[str, int], ...]
    available: bool


@dataclass(frozen=True)
class IterationTraceSummary:
    """One accepted iteration with separate host and device diagnostics."""

    iteration: int
    host_boundary_ns: int
    host_control_gap_ns: int
    cuda_memcpy_ns: int
    newton_adjoint_ns: int
    other_attributed_ns: int
    unattributed_ns: int
    device_active_ns: int
    active_ns: int
    required_phase_families_present: bool
    missing_required_phases: tuple[str, ...]
    semantic_counts_available: bool
    phase_active_ns: tuple[tuple[str, int], ...]
    raw_phase_duration_ns: tuple[tuple[str, int], ...]
    semantic_solver_counts: tuple[tuple[str, int], ...]
    device_interval_counts: tuple[tuple[str, int], ...]
    device_kernel_group_counts: tuple[tuple[str, int], ...]
    device_overlap_ns: int

    def to_json(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "host_boundary_ns": self.host_boundary_ns,
            "host_control_gap_ns": self.host_control_gap_ns,
            "cuda_memcpy_ns": self.cuda_memcpy_ns,
            "newton_adjoint_ns": self.newton_adjoint_ns,
            "other_attributed_ns": self.other_attributed_ns,
            "unattributed_ns": self.unattributed_ns,
            "device_active_ns": self.device_active_ns,
            "active_ns": self.active_ns,
            "required_phase_families_present": self.required_phase_families_present,
            "missing_required_phases": list(self.missing_required_phases),
            "semantic_counts_available": self.semantic_counts_available,
            "phase_active_ns": dict(self.phase_active_ns),
            "raw_phase_duration_ns": dict(self.raw_phase_duration_ns),
            "semantic_solver_counts": dict(self.semantic_solver_counts),
            "device_interval_counts": dict(self.device_interval_counts),
            "device_kernel_group_counts": dict(self.device_kernel_group_counts),
            "device_overlap_ns": self.device_overlap_ns,
        }


@dataclass(frozen=True)
class TraceSummary:
    """Validated structural summary recomputable from raw trace evidence."""

    schema_id: str
    trace_schema_id: str
    child_id: str
    trace_schema_valid: bool
    clock_correlation_valid: bool
    required_phase_families_present: bool
    semantic_counts_available: bool
    diagnostics: tuple[str, ...]
    iterations: tuple[IterationTraceSummary, ...]
    raw_nested_device_ns: int
    device_active_ns: int
    device_overlap_ns: int
    unattributed_device_ns: int
    out_of_window_device_ns: int

    def to_json(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "trace_schema_id": self.trace_schema_id,
            "child_id": self.child_id,
            "trace_schema_valid": self.trace_schema_valid,
            "clock_correlation_valid": self.clock_correlation_valid,
            "required_phase_families_present": self.required_phase_families_present,
            "semantic_counts_available": self.semantic_counts_available,
            "diagnostics": list(self.diagnostics),
            "iterations": [iteration.to_json() for iteration in self.iterations],
            "raw_nested_device_ns": self.raw_nested_device_ns,
            "device_active_ns": self.device_active_ns,
            "device_overlap_ns": self.device_overlap_ns,
            "unattributed_device_ns": self.unattributed_device_ns,
            "out_of_window_device_ns": self.out_of_window_device_ns,
        }


@dataclass(frozen=True)
class SegmentedIterationTraceSummary:
    """One independently clock-correlated accepted-iteration trace segment."""

    schema_id: str
    trace_schema_id: str
    child_id: str
    sample_id: str
    accepted_iteration: int
    segment_evaluation_ids_sha256: str
    trace_schema_valid: bool
    clock_correlation_valid: bool
    diagnostics: tuple[str, ...]
    iteration: IterationTraceSummary
    profiler_boundary_pause_ns: int
    raw_active_ns: int
    raw_nested_device_ns: int
    device_active_ns: int
    device_overlap_ns: int
    unattributed_device_ns: int

    def to_json(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "trace_schema_id": self.trace_schema_id,
            "child_id": self.child_id,
            "sample_id": self.sample_id,
            "accepted_iteration": self.accepted_iteration,
            "segment_evaluation_ids_sha256": self.segment_evaluation_ids_sha256,
            "trace_schema_valid": self.trace_schema_valid,
            "clock_correlation_valid": self.clock_correlation_valid,
            "diagnostics": list(self.diagnostics),
            "iteration": self.iteration.to_json(),
            "profiler_boundary_pause_ns": self.profiler_boundary_pause_ns,
            "raw_active_ns": self.raw_active_ns,
            "raw_nested_device_ns": self.raw_nested_device_ns,
            "device_active_ns": self.device_active_ns,
            "device_overlap_ns": self.device_overlap_ns,
            "unattributed_device_ns": self.unattributed_device_ns,
        }


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TraceSummaryError("unknown_trace_schema", f"{label} must be an object")
    return value


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TraceSummaryError("unknown_trace_schema", f"{label} must be an array")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceSummaryError("unknown_trace_schema", f"{label} must be a string")
    return value


def _require_complete_event_name(value: object, label: str) -> str:
    """Return a Chrome complete-event name, which JAX may emit as empty."""

    if not isinstance(value, str):
        raise TraceSummaryError("unknown_trace_schema", f"{label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceSummaryError("unknown_trace_schema", f"{label} must be an integer")
    return value


def _trace_time_ns(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceSummaryError("unknown_trace_schema", f"{label} must be numeric")
    decimal_ns = Decimal(str(value)) * 1000
    if decimal_ns != decimal_ns.to_integral_value():
        raise TraceSummaryError(
            "unknown_trace_schema", f"{label} exceeds nanosecond trace precision"
        )
    result = int(decimal_ns)
    if result < 0:
        raise TraceSummaryError("unknown_trace_schema", f"{label} must be nonnegative")
    return result


def union_intervals(intervals: Sequence[Interval]) -> tuple[tuple[Interval, ...], int]:
    """Return the canonical interval union and its total duration."""

    if not intervals:
        return (), 0
    ordered = sorted(intervals)
    merged: list[Interval] = []
    start_ns = ordered[0].start_ns
    end_ns = ordered[0].end_ns
    for interval in ordered[1:]:
        if interval.start_ns <= end_ns:
            end_ns = max(end_ns, interval.end_ns)
            continue
        merged.append(Interval(start_ns, end_ns))
        start_ns = interval.start_ns
        end_ns = interval.end_ns
    merged.append(Interval(start_ns, end_ns))
    return tuple(merged), sum(interval.duration_ns for interval in merged)


def unique_deepest_scope(
    scope_paths: Sequence[Sequence[PhaseId]],
) -> ScopeAttribution:
    """Resolve one deepest phase; equal-depth multi-owner scopes are ambiguous."""

    candidates: list[tuple[int, tuple[PhaseId, ...]]] = []
    for scope_path in scope_paths:
        normalized = tuple(scope_path)
        if normalized:
            candidates.append((len(normalized), normalized))
    if not candidates:
        return ScopeAttribution(None, None, False)
    deepest_depth = max(depth for depth, _ in candidates)
    deepest_paths = tuple(path for depth, path in candidates if depth == deepest_depth)
    deepest_phases = frozenset(path[-1] for path in deepest_paths)
    if len(deepest_phases) != 1:
        return ScopeAttribution(None, None, True)
    phase = next(iter(deepest_phases))
    critical_phases = frozenset(
        next((item for item in reversed(path) if item not in _DRILLDOWN_PHASES), None)
        for path in deepest_paths
    )
    if len(critical_phases) != 1:
        return ScopeAttribution(None, None, True)
    return ScopeAttribution(phase, next(iter(critical_phases)), False)


def _phase_path_from_name(name: str) -> tuple[PhaseId, ...]:
    _validate_timeline_segments(name)
    return tuple(
        _PHASE_BY_VALUE[segment]
        for segment in name.split("/")
        if segment in _PHASE_BY_VALUE
    )


def _validate_timeline_segments(value: str) -> None:
    for segment in value.split("/"):
        is_unknown_namespace = segment.startswith(_TIMELINE_NAMESPACES)
        is_unknown_phase_shape = (
            _TIMELINE_LIKE_SEGMENT_PATTERN.fullmatch(segment) is not None
        )
        if (
            is_unknown_namespace or is_unknown_phase_shape
        ) and segment not in _ALLOWED_TIMELINE_EVENT_NAMES:
            raise TraceSummaryError(
                "unknown_phase_id", f"unsupported timeline phase segment {segment!r}"
            )


def _scope_paths_from_span(span: _TraceSpan) -> tuple[tuple[PhaseId, ...], ...]:
    metadata_values = [span.name]
    for key in ("name", "hlo_op", "hlo_module", "tf_op"):
        value = span.args.get(key)
        if isinstance(value, str):
            metadata_values.append(value)
    paths = {_phase_path_from_name(value) for value in metadata_values}
    hlo_module = span.args.get("hlo_module")
    if isinstance(hlo_module, str) and hlo_module in _HLO_MODULE_PHASE:
        paths.add((_HLO_MODULE_PHASE[hlo_module],))
    return tuple(
        sorted(
            (path for path in paths if path),
            key=lambda path: tuple(item.value for item in path),
        )
    )


def _scope_range_attributions(
    spans: Sequence[_TraceSpan],
) -> Mapping[str, ScopeAttribution]:
    """Resolve directly labeled profiler observations by ownership range.

    Range inheritance is allowed only when all direct labels select one deepest
    owner. An ambiguous range is corrupt evidence, not an attribution hint.
    """

    paths_by_range: dict[str, list[tuple[PhaseId, ...]]] = {}
    for span in spans:
        if "scope_range_id" not in span.args:
            continue
        scope_range_id = _require_string(
            span.args.get("scope_range_id"), "trace event args.scope_range_id"
        )
        direct_scope_name = span.args.get("name")
        if isinstance(direct_scope_name, str):
            direct_path = _phase_path_from_name(direct_scope_name)
            if direct_path:
                paths_by_range.setdefault(scope_range_id, []).append(direct_path)

    attributions: dict[str, ScopeAttribution] = {}
    for scope_range_id, direct_paths in paths_by_range.items():
        deepest_path = max(direct_paths, key=len)
        paths_disagree = any(
            deepest_path[: len(direct_path)] != direct_path
            for direct_path in direct_paths
        )
        attribution = unique_deepest_scope(direct_paths)
        if paths_disagree or attribution.ambiguous:
            raise TraceSummaryError(
                "scope_range_attribution_invalid",
                f"scope_range_id {scope_range_id!r} has conflicting direct labels",
            )
        if attribution.phase is not None:
            attributions[scope_range_id] = attribution
    return attributions


def _inherit_scope_range_attribution(
    span: _TraceSpan,
    direct: ScopeAttribution,
    attributions_by_range: Mapping[str, ScopeAttribution],
) -> ScopeAttribution:
    """Fill an unlabeled event from its exact profiler ownership range."""

    if direct.phase is not None or direct.ambiguous:
        return direct
    if "scope_range_id" not in span.args:
        return direct
    scope_range_id = _require_string(
        span.args.get("scope_range_id"), "device event args.scope_range_id"
    )
    return attributions_by_range.get(scope_range_id, direct)


def _parse_complete_span(event: Mapping[str, object], index: int) -> _TraceSpan:
    base_keys = {"ph", "pid", "tid", "ts", "dur", "name"}
    if set(event) not in (base_keys, base_keys | {"args"}):
        raise TraceSummaryError(
            "unknown_trace_schema",
            f"traceEvents[{index}] complete-event keys do not match JAX 0.10.0",
        )
    if event.get("ph") != "X":
        raise TraceSummaryError(
            "unknown_trace_schema", f"traceEvents[{index}] is not a complete event"
        )
    start_ns = _trace_time_ns(event.get("ts"), f"traceEvents[{index}].ts")
    duration_ns = _trace_time_ns(event.get("dur"), f"traceEvents[{index}].dur")
    if duration_ns <= 0:
        raise TraceSummaryError(
            "unknown_trace_schema", f"traceEvents[{index}].dur must be positive"
        )
    return _TraceSpan(
        interval=Interval(start_ns, start_ns + duration_ns),
        name=_require_complete_event_name(
            event.get("name"), f"traceEvents[{index}].name"
        ),
        args=_require_mapping(event.get("args", {}), f"traceEvents[{index}].args"),
        pid=_require_int(event.get("pid"), f"traceEvents[{index}].pid"),
        tid=_require_int(event.get("tid"), f"traceEvents[{index}].tid"),
    )


def _parse_trace_document(
    document: Mapping[str, object],
) -> tuple[tuple[_TraceSpan, ...], frozenset[int]]:
    if set(document) != {"displayTimeUnit", "metadata", "traceEvents"}:
        raise TraceSummaryError(
            "unknown_trace_schema",
            "top-level keys must be displayTimeUnit, metadata, and traceEvents",
        )
    if document["displayTimeUnit"] != "ns":
        raise TraceSummaryError("unknown_trace_schema", "displayTimeUnit must be ns")
    metadata = _require_mapping(document["metadata"], "metadata")
    if metadata != {"highres-ticks": True}:
        raise TraceSummaryError(
            "unknown_trace_schema", "metadata must declare highres-ticks=true"
        )
    raw_events = _require_sequence(document["traceEvents"], "traceEvents")
    process_names: dict[int, str] = {}
    complete_spans: list[_TraceSpan] = []
    for index, raw_event in enumerate(raw_events):
        event = _require_mapping(raw_event, f"traceEvents[{index}]")
        if not event:
            if index != len(raw_events) - 1:
                raise TraceSummaryError(
                    "unknown_trace_schema", "only the final trace event may be empty"
                )
            continue
        phase = event.get("ph")
        if phase == "M":
            name = event.get("name")
            if name not in _METADATA_EVENT_NAMES:
                raise TraceSummaryError(
                    "unknown_trace_schema", f"unsupported metadata event {name!r}"
                )
            required_keys = {"ph", "pid", "name", "args"}
            if name in {"thread_name", "thread_sort_index"}:
                required_keys.add("tid")
            if set(event) != required_keys:
                raise TraceSummaryError(
                    "unknown_trace_schema",
                    f"traceEvents[{index}] metadata keys do not match JAX 0.10.0",
                )
            if event.get("name") == "process_name":
                pid = _require_int(event.get("pid"), f"traceEvents[{index}].pid")
                args = _require_mapping(event.get("args"), f"traceEvents[{index}].args")
                process_names[pid] = _require_string(
                    args.get("name"), f"traceEvents[{index}].args.name"
                )
            continue
        if phase == "X":
            complete_spans.append(_parse_complete_span(event, index))
            continue
        raise TraceSummaryError(
            "unknown_trace_schema", f"unsupported trace event phase {phase!r}"
        )
    device_pids = frozenset(
        pid
        for pid, name in process_names.items()
        if _DEVICE_PROCESS_PATTERN.fullmatch(name)
    )
    if len(device_pids) != 1:
        raise TraceSummaryError(
            "unknown_trace_schema",
            f"expected exactly one CUDA device process, found {len(device_pids)}",
        )
    if "/host:CPU" not in process_names.values():
        raise TraceSummaryError("unknown_trace_schema", "missing /host:CPU process")
    return tuple(complete_spans), device_pids


def _is_compilation_event(name: str) -> bool:
    return (
        name in _COMPILATION_EVENT_NAMES
        or _COMPILE_AND_LOAD_EVENT_PATTERN.fullmatch(name) is not None
    )


def _reject_compilation_inside_measurement(
    spans: Sequence[_TraceSpan], trace_points: Sequence[_LifecyclePoint]
) -> None:
    initial_entries = tuple(
        point
        for point in trace_points
        if point.evaluation_kind is EvaluationKind.INITIAL
        and point.event is HostEvent.EVALUATOR_ENTRY
    )
    trial_returns = tuple(
        point
        for point in trace_points
        if point.evaluation_kind is EvaluationKind.TRIAL
        and point.event is HostEvent.EVALUATOR_RETURN
    )
    if len(initial_entries) != 1 or not trial_returns:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            "compile gate requires one initial entry and at least one trial return",
        )
    measurement_envelope = Interval(
        initial_entries[0].timestamp_ns,
        trial_returns[-1].timestamp_ns + trial_returns[-1].annotation_duration_ns,
    )
    for span in spans:
        if not _is_compilation_event(span.name):
            continue
        if measurement_envelope.overlaps(span.interval):
            raise TraceSummaryError(
                "compilation_in_measurement",
                f"compilation event {span.name!r} overlaps the initial/trial "
                "measurement envelope",
            )


def _accepted_iteration_spans(
    spans: Sequence[_TraceSpan], expected_iterations: int
) -> dict[int, _TraceSpan]:
    accepted: dict[int, _TraceSpan] = {}
    for span in spans:
        if span.name != _ACCEPTED_ITERATION_EVENT:
            continue
        step_num = span.args.get("step_num")
        if not isinstance(step_num, str) or not step_num.isdecimal():
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                "accepted iteration has invalid step_num",
            )
        iteration = int(step_num)
        if iteration in accepted:
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                f"duplicate accepted iteration envelope {iteration}",
            )
        accepted[iteration] = span
    required = set(range(1, expected_iterations + 1))
    if set(accepted) != required:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            f"accepted iteration IDs are {sorted(accepted)}, expected {sorted(required)}",
        )
    ordered = [accepted[iteration].interval for iteration in sorted(accepted)]
    if any(left.overlaps(right) for left, right in zip(ordered, ordered[1:])):
        raise TraceSummaryError(
            "iteration_correlation_invalid", "accepted iteration envelopes overlap"
        )
    return accepted


def _host_transfer_spans(spans: Sequence[_TraceSpan]) -> tuple[_TraceSpan, ...]:
    return tuple(span for span in spans if span.name in _HOST_TRANSFER_PHASE_VALUES)


def _containing_transfer_paths(
    interval: Interval, transfer_spans: Sequence[_TraceSpan]
) -> tuple[tuple[PhaseId, ...], ...]:
    containing = [span for span in transfer_spans if span.interval.contains(interval)]
    paths: list[tuple[PhaseId, ...]] = []
    for candidate in containing:
        outer = [
            span
            for span in containing
            if span is not candidate and span.interval.contains(candidate.interval)
        ]
        path = tuple(
            _PHASE_BY_VALUE[span.name]
            for span in sorted(
                (*outer, candidate),
                key=lambda span: (-span.interval.duration_ns, span.interval.start_ns),
            )
        )
        paths.append(path)
    return tuple(paths)


def _transfer_paths_by_correlation(
    spans: Sequence[_TraceSpan],
    device_pids: frozenset[int],
    transfer_spans: Sequence[_TraceSpan],
) -> Mapping[str, tuple[tuple[PhaseId, ...], ...]]:
    paths_by_correlation: dict[str, list[tuple[PhaseId, ...]]] = {}
    for span in spans:
        if span.pid in device_pids or "memcpy_details" not in span.args:
            continue
        correlation_id = _require_string(
            span.args.get("correlation_id"), "host memcpy correlation_id"
        )
        containing_paths = _containing_transfer_paths(span.interval, transfer_spans)
        if containing_paths:
            paths_by_correlation.setdefault(correlation_id, []).extend(containing_paths)
    return {
        correlation_id: tuple(paths)
        for correlation_id, paths in paths_by_correlation.items()
    }


def _device_intervals(
    spans: Sequence[_TraceSpan],
    device_pids: frozenset[int],
    transfer_spans: Sequence[_TraceSpan],
) -> tuple[_DeviceInterval, ...]:
    attributions_by_range = _scope_range_attributions(spans)
    correlated_transfer_paths = _transfer_paths_by_correlation(
        spans, device_pids, transfer_spans
    )
    intervals: list[_DeviceInterval] = []
    for span in spans:
        if span.pid not in device_pids:
            continue
        has_kernel = "kernel_details" in span.args
        has_memcpy = "memcpy_details" in span.args
        if has_kernel and has_memcpy:
            raise TraceSummaryError(
                "unknown_trace_schema",
                "device event declares kernel and memcpy details",
            )
        if has_kernel:
            missing_arguments = _KERNEL_REQUIRED_ARGUMENTS - span.args.keys()
            if missing_arguments:
                raise TraceSummaryError(
                    "unknown_trace_schema",
                    f"kernel event misses JAX 0.10.0 arguments {sorted(missing_arguments)}",
                )
            attribution = _inherit_scope_range_attribution(
                span,
                unique_deepest_scope(_scope_paths_from_span(span)),
                attributions_by_range,
            )
            kind = "kernel"
        elif has_memcpy:
            missing_arguments = _MEMCPY_REQUIRED_ARGUMENTS - span.args.keys()
            if missing_arguments:
                raise TraceSummaryError(
                    "unknown_trace_schema",
                    f"memcpy event misses JAX 0.10.0 arguments {sorted(missing_arguments)}",
                )
            details = _require_string(
                span.args.get("memcpy_details"), "device event args.memcpy_details"
            )
            if _MEMCPY_DETAILS_PATTERN.fullmatch(details) is None:
                raise TraceSummaryError(
                    "unknown_trace_schema",
                    "unsupported CUDA memcpy_details representation",
                )
            correlation_id = _require_string(
                span.args.get("correlation_id"), "device memcpy correlation_id"
            )
            scope_paths = tuple(
                {
                    *_scope_paths_from_span(span),
                    *correlated_transfer_paths.get(correlation_id, ()),
                    *_containing_transfer_paths(span.interval, transfer_spans),
                }
            )
            attribution = _inherit_scope_range_attribution(
                span,
                unique_deepest_scope(scope_paths),
                attributions_by_range,
            )
            kind = "memcpy"
        else:
            attribution = ScopeAttribution(None, None, False)
            kind = "unknown_device"
        intervals.append(_DeviceInterval(span.interval, kind, attribution))
    return tuple(intervals)


def _trace_lifecycle_points(spans: Sequence[_TraceSpan]) -> tuple[_LifecyclePoint, ...]:
    points: list[_LifecyclePoint] = []
    valid_events = {event.value: event for event in HostEvent}
    for span in spans:
        if not span.name.startswith(_LIFECYCLE_EVENT_PREFIX):
            continue
        event_name = span.name.removeprefix(_LIFECYCLE_EVENT_PREFIX)
        event = valid_events.get(event_name)
        if event is None:
            raise TraceSummaryError(
                "host_correlation_invalid",
                f"unknown lifecycle event {span.name!r}",
            )
        kind_value = _require_string(
            span.args.get("evaluation_kind"), "evaluation_kind"
        )
        try:
            kind = EvaluationKind(kind_value)
        except ValueError as error:
            raise TraceSummaryError(
                "host_correlation_invalid", f"unknown evaluation kind {kind_value!r}"
            ) from error
        iteration_value = span.args.get("outer_iteration_id")
        iteration = None
        if iteration_value is not None:
            if not isinstance(iteration_value, str) or not iteration_value.isdecimal():
                raise TraceSummaryError(
                    "host_correlation_invalid", "outer_iteration_id must be decimal"
                )
            iteration = int(iteration_value)
        points.append(
            _LifecyclePoint(
                event=event,
                timestamp_ns=span.interval.start_ns,
                evaluation_id=_require_string(
                    span.args.get("evaluation_id"), "evaluation_id"
                ),
                parameter_sha256=_require_string(
                    span.args.get("parameter_sha256"), "parameter_sha256"
                ),
                evaluation_kind=kind,
                outer_iteration_id=iteration,
                annotation_duration_ns=span.interval.duration_ns,
            )
        )
    return tuple(sorted(points, key=lambda point: point.timestamp_ns))


def _host_lifecycle_points(
    host_events: Sequence[HostEventRecord],
) -> tuple[_LifecyclePoint, ...]:
    if tuple(record.sequence for record in host_events) != tuple(
        range(len(host_events))
    ):
        raise TraceSummaryError(
            "host_correlation_invalid",
            "host lifecycle sequence IDs must be contiguous and unique",
        )
    points = tuple(
        _LifecyclePoint(
            event=record.event,
            timestamp_ns=record.timestamp_ns,
            evaluation_id=record.evaluation.evaluation_id,
            parameter_sha256=record.evaluation.parameter_sha256,
            evaluation_kind=record.evaluation.kind,
            outer_iteration_id=record.evaluation.outer_iteration_id,
            annotation_duration_ns=0,
        )
        for record in host_events
    )
    if any(
        left.timestamp_ns >= right.timestamp_ns
        for left, right in zip(points, points[1:])
    ):
        raise TraceSummaryError(
            "host_correlation_invalid",
            "host lifecycle timestamps must be strictly increasing",
        )
    return points


def _point_identity(point: _LifecyclePoint) -> tuple[object, ...]:
    return (
        point.event,
        point.evaluation_id,
        point.parameter_sha256,
        point.evaluation_kind,
        point.outer_iteration_id,
    )


def _validate_lifecycle_points(
    host_points: Sequence[_LifecyclePoint], trace_points: Sequence[_LifecyclePoint]
) -> None:
    if not host_points:
        raise TraceSummaryError("host_correlation_invalid", "host lifecycle is empty")
    host_identities = tuple(_point_identity(point) for point in host_points)
    trace_identities = tuple(_point_identity(point) for point in trace_points)
    if host_identities != trace_identities:
        raise TraceSummaryError(
            "host_correlation_invalid",
            "host and profiler lifecycle points are not a bijective ordered match",
        )
    by_evaluation: dict[str, list[_LifecyclePoint]] = {}
    for point in host_points:
        by_evaluation.setdefault(point.evaluation_id, []).append(point)
    expected_events = tuple(HostEvent)
    for evaluation_id, points in by_evaluation.items():
        observed_events = tuple(point.event for point in points)
        if observed_events != expected_events:
            raise TraceSummaryError(
                "host_correlation_invalid",
                f"evaluation {evaluation_id!r} events are {observed_events}, "
                f"expected {expected_events}",
            )
        if len({_point_identity(point)[1:] for point in points}) != 1:
            raise TraceSummaryError(
                "host_correlation_invalid",
                f"evaluation {evaluation_id!r} changes identity across events",
            )


def _validate_clock_anchors(
    host_points: Sequence[_LifecyclePoint], trace_points: Sequence[_LifecyclePoint]
) -> None:
    if len(host_points) < 2:
        raise TraceSummaryError(
            "clock_correlation_invalid", "at least two lifecycle anchors are required"
        )
    host_span_ns = host_points[-1].timestamp_ns - host_points[0].timestamp_ns
    trace_span_ns = trace_points[-1].timestamp_ns - trace_points[0].timestamp_ns
    if host_span_ns <= 0 or trace_span_ns <= 0:
        raise TraceSummaryError(
            "clock_correlation_invalid", "lifecycle anchor clocks do not advance"
        )
    clock_rate = Decimal(host_span_ns) / Decimal(trace_span_ns)
    if abs(clock_rate - Decimal(1)) > _CLOCK_RATE_RELATIVE_TOLERANCE:
        raise TraceSummaryError(
            "clock_correlation_invalid",
            f"host/profiler anchor clock-rate ratio {clock_rate} exceeds tolerance",
        )
    host_origin = host_points[0].timestamp_ns
    trace_origin = trace_points[0].timestamp_ns
    residual_tolerance_ns = max(
        _CLOCK_ANCHOR_RESIDUAL_FLOOR_NS,
        max(point.annotation_duration_ns for point in trace_points) * 10,
    )
    for host_point, trace_point in zip(host_points, trace_points, strict=True):
        predicted_host_ns = host_origin + int(
            Decimal(trace_point.timestamp_ns - trace_origin) * clock_rate
        )
        if abs(host_point.timestamp_ns - predicted_host_ns) > residual_tolerance_ns:
            raise TraceSummaryError(
                "clock_correlation_invalid",
                f"lifecycle anchor {host_point.evaluation_id}:{host_point.event.value} "
                "exceeds affine residual tolerance",
            )


def _validate_lifecycle_iteration_envelopes(
    trace_points: Sequence[_LifecyclePoint], accepted: Mapping[int, _TraceSpan]
) -> None:
    for point in trace_points:
        containing = tuple(
            iteration
            for iteration, envelope in accepted.items()
            if envelope.interval.start_ns
            <= point.timestamp_ns
            < envelope.interval.end_ns
        )
        if point.evaluation_kind is EvaluationKind.TRIAL:
            if point.outer_iteration_id is None or containing != (
                point.outer_iteration_id,
            ):
                raise TraceSummaryError(
                    "iteration_correlation_invalid",
                    f"trial {point.evaluation_id!r} lifecycle point is outside its "
                    "accepted-iteration envelope",
                )
        elif containing:
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                f"{point.evaluation_kind.value} lifecycle point appears inside an "
                "accepted-iteration envelope",
            )


def _validate_host_gaps_exclude_device_work(
    trace_points: Sequence[_LifecyclePoint],
    device_intervals: Sequence[_DeviceInterval],
) -> None:
    evaluations = [
        trace_points[offset : offset + len(HostEvent)]
        for offset in range(0, len(trace_points), len(HostEvent))
    ]
    for previous, following in zip(evaluations, evaluations[1:]):
        next_entry = following[0]
        if next_entry.evaluation_kind is EvaluationKind.FINAL_REPORTING:
            continue
        trace_gap = Interval(previous[-1].timestamp_ns, next_entry.timestamp_ns)
        if any(item.interval.overlaps(trace_gap) for item in device_intervals):
            raise TraceSummaryError(
                "clock_correlation_invalid",
                "CUDA work overlaps a correlated host-control gap",
            )


def _host_control_gaps(
    host_points: Sequence[_LifecyclePoint], expected_iterations: int
) -> dict[int, tuple[Interval, ...]]:
    evaluations: list[tuple[_LifecyclePoint, _LifecyclePoint]] = []
    for offset in range(0, len(host_points), len(HostEvent)):
        group = host_points[offset : offset + len(HostEvent)]
        if len(group) != len(HostEvent):
            raise TraceSummaryError(
                "host_correlation_invalid", "incomplete lifecycle point group"
            )
        evaluations.append((group[0], group[-1]))
    gaps: dict[int, list[Interval]] = {
        iteration: [] for iteration in range(1, expected_iterations + 1)
    }
    for (_, previous_return), (next_entry, _) in zip(evaluations, evaluations[1:]):
        if next_entry.evaluation_kind is EvaluationKind.FINAL_REPORTING:
            continue
        iteration = next_entry.outer_iteration_id
        if next_entry.evaluation_kind is not EvaluationKind.TRIAL or iteration is None:
            raise TraceSummaryError(
                "host_correlation_invalid",
                "every non-initial, non-final evaluation must be an iteration-bound trial",
            )
        if iteration not in gaps:
            raise TraceSummaryError(
                "host_correlation_invalid", f"trial references iteration {iteration}"
            )
        if next_entry.timestamp_ns <= previous_return.timestamp_ns:
            raise TraceSummaryError(
                "host_correlation_invalid",
                "evaluator return/entry gap is reversed or empty",
            )
        gaps[iteration].append(
            Interval(previous_return.timestamp_ns, next_entry.timestamp_ns)
        )
    return {iteration: tuple(intervals) for iteration, intervals in gaps.items()}


def _semantic_execution_counts(
    host_points: Sequence[_LifecyclePoint],
    evaluation_documents: Sequence[Mapping[str, object]] | None,
    expected_iterations: int,
    *,
    iteration_ids: Sequence[int] | None = None,
) -> Mapping[int, _SemanticIterationCounts]:
    target_iterations = tuple(
        range(1, expected_iterations + 1) if iteration_ids is None else iteration_ids
    )
    unavailable = {
        iteration: _SemanticIterationCounts((), False)
        for iteration in target_iterations
    }
    if evaluation_documents is None:
        return unavailable
    contexts = {
        point.evaluation_id: point
        for point in host_points
        if point.event is HostEvent.EVALUATOR_ENTRY
    }
    documents_by_id: dict[str, Mapping[str, object]] = {}
    for document in evaluation_documents:
        evaluation_id = _require_string(
            document.get("evaluation_id"), "evaluation execution-count ID"
        )
        if evaluation_id in documents_by_id:
            raise TraceSummaryError(
                "semantic_count_correlation_invalid",
                f"duplicate evaluation document {evaluation_id!r}",
            )
        documents_by_id[evaluation_id] = document
    if set(documents_by_id) != set(contexts):
        raise TraceSummaryError(
            "semantic_count_correlation_invalid",
            "evaluation documents do not bijectively match lifecycle evaluations",
        )
    aggregated: dict[int, dict[str, int]] = {
        iteration: {} for iteration in target_iterations
    }
    available = {iteration: True for iteration in aggregated}
    trial_counts = {iteration: 0 for iteration in aggregated}
    for evaluation_id, context in contexts.items():
        document = documents_by_id[evaluation_id]
        if document.get("lifecycle") != context.evaluation_kind.value:
            raise TraceSummaryError(
                "semantic_count_correlation_invalid",
                f"evaluation {evaluation_id!r} lifecycle differs from trace evidence",
            )
        if context.evaluation_kind is not EvaluationKind.TRIAL:
            continue
        iteration = context.outer_iteration_id
        if iteration not in aggregated or document.get("iteration") != iteration:
            raise TraceSummaryError(
                "semantic_count_correlation_invalid",
                f"evaluation {evaluation_id!r} iteration differs from trace evidence",
            )
        trial_counts[iteration] += 1
        inner = _require_mapping(
            document.get("inner_evidence"), f"{evaluation_id}.inner_evidence"
        )
        adjoint = _require_mapping(
            document.get("adjoint_evidence"), f"{evaluation_id}.adjoint_evidence"
        )
        count_sources: tuple[tuple[str, Mapping[str, object]], ...] = (
            ("newton_iterations", inner),
            *((field, adjoint) for field in _ADJOINT_SEMANTIC_COUNT_FIELDS),
        )
        for count_id, source in count_sources:
            value = source.get(count_id)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                available[iteration] = False
                continue
            aggregate = aggregated[iteration]
            aggregate[count_id] = aggregate.get(count_id, 0) + value
    return {
        iteration: _SemanticIterationCounts(
            counts=tuple(sorted(aggregated[iteration].items())),
            available=available[iteration] and trial_counts[iteration] > 0,
        )
        for iteration in aggregated
    }


def _critical_category(interval: _DeviceInterval) -> str:
    phase = interval.attribution.critical_phase
    if interval.attribution.ambiguous or phase is None:
        return "unattributed"
    if phase in _HOST_TRANSFER_PHASES:
        return "host_boundary"
    if phase in _NEWTON_ADJOINT_PHASES:
        return "newton_adjoint"
    return "other_attributed"


def _exclusive_category_durations(
    intervals: Sequence[_DeviceInterval],
) -> tuple[dict[str, int], int]:
    """Union device time by category, making cross-category overlap unattributed."""

    category_names = (
        "host_boundary",
        "newton_adjoint",
        "other_attributed",
        "unattributed",
    )
    durations = dict.fromkeys(category_names, 0)
    if not intervals:
        return durations, 0

    category_unions: dict[str, list[Interval]] = {
        category: [] for category in category_names
    }
    for item in intervals:
        category_unions[_critical_category(item)].append(item.interval)
    for category, category_intervals in category_unions.items():
        category_unions[category] = _coalesce_intervals_in_place(category_intervals)

    attributed_categories = category_names[:-1]
    for category in attributed_categories:
        blockers = _union_sorted_interval_sequences(
            *(category_unions[other] for other in category_names if other != category)
        )
        durations[category] = _interval_difference_duration(
            category_unions[category], blockers
        )
    all_device_intervals = _union_sorted_interval_sequences(
        *(category_unions[category] for category in category_names)
    )
    total_active_ns = sum(interval.duration_ns for interval in all_device_intervals)
    durations["unattributed"] = total_active_ns - sum(
        durations[category] for category in attributed_categories
    )
    return durations, total_active_ns


def _coalesce_intervals_in_place(intervals: list[Interval]) -> list[Interval]:
    """Sort and union one category using only its existing reference array."""

    if not intervals:
        return intervals
    intervals.sort()
    output_index = 0
    current = intervals[0]
    for candidate in intervals[1:]:
        if candidate.start_ns <= current.end_ns:
            if candidate.end_ns > current.end_ns:
                current = Interval(current.start_ns, candidate.end_ns)
            continue
        intervals[output_index] = current
        output_index += 1
        current = candidate
    intervals[output_index] = current
    del intervals[output_index + 1 :]
    return intervals


def _union_sorted_interval_sequences(
    *sequences: Sequence[Interval],
) -> list[Interval]:
    """Union already-sorted interval sequences without concatenating them."""

    union: list[Interval] = []
    for candidate in heapq.merge(*sequences):
        if not union or candidate.start_ns > union[-1].end_ns:
            union.append(candidate)
            continue
        if candidate.end_ns > union[-1].end_ns:
            union[-1] = Interval(union[-1].start_ns, candidate.end_ns)
    return union


def _interval_difference_duration(
    source: Sequence[Interval], blockers: Sequence[Interval]
) -> int:
    """Measure half-open source intervals not covered by sorted blockers."""

    duration_ns = 0
    blocker_index = 0
    for interval in source:
        cursor_ns = interval.start_ns
        while (
            blocker_index < len(blockers)
            and blockers[blocker_index].end_ns <= cursor_ns
        ):
            blocker_index += 1
        scan_index = blocker_index
        while (
            scan_index < len(blockers)
            and blockers[scan_index].start_ns < interval.end_ns
        ):
            blocker = blockers[scan_index]
            if blocker.start_ns > cursor_ns:
                duration_ns += blocker.start_ns - cursor_ns
            cursor_ns = max(cursor_ns, blocker.end_ns)
            if cursor_ns >= interval.end_ns:
                break
            scan_index += 1
        if cursor_ns < interval.end_ns:
            duration_ns += interval.end_ns - cursor_ns
    return duration_ns


def _summarize_iteration(
    iteration: int,
    envelope: Interval,
    device_intervals: Sequence[_DeviceInterval],
    host_gaps: Sequence[Interval],
    semantic_counts: _SemanticIterationCounts,
) -> IterationTraceSummary:
    contained = tuple(
        item for item in device_intervals if envelope.contains(item.interval)
    )
    category_ns, device_active_ns = _exclusive_category_durations(contained)
    raw_nested_ns = sum(item.interval.duration_ns for item in contained)
    phase_intervals: dict[PhaseId, list[Interval]] = {}
    phase_raw_ns: dict[PhaseId, int] = {}
    phase_counts: dict[PhaseId, int] = {}
    phase_kernel_counts: dict[PhaseId, int] = {}
    for item in contained:
        phase = item.attribution.phase
        if phase is None or item.attribution.ambiguous:
            continue
        phase_intervals.setdefault(phase, []).append(item.interval)
        phase_raw_ns[phase] = phase_raw_ns.get(phase, 0) + item.interval.duration_ns
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if item.kind == "kernel":
            phase_kernel_counts[phase] = phase_kernel_counts.get(phase, 0) + 1
    phase_active_ns = {
        phase.value: union_intervals(intervals)[1]
        for phase, intervals in phase_intervals.items()
    }
    host_control_gap_ns = sum(interval.duration_ns for interval in host_gaps)
    cuda_memcpy_ns = sum(
        duration
        for phase, duration in phase_active_ns.items()
        if phase in _HOST_TRANSFER_PHASE_VALUES
    )
    host_boundary_ns = host_control_gap_ns + category_ns["host_boundary"]
    device_interval_counts = {
        phase.value: count for phase, count in phase_counts.items()
    }
    observed_phases = frozenset(phase_counts)
    missing_required_phases = tuple(
        sorted(phase.value for phase in _REQUIRED_DEVICE_PHASES - observed_phases)
    )
    if not host_gaps:
        missing_required_phases += (PhaseId.HOST_LINE_SEARCH_CONTROL.value,)
    device_kernel_group_counts = tuple(
        sorted(
            (phase.value, phase_kernel_counts.get(phase, 0))
            for phase in _DRILLDOWN_PHASES
        )
    )
    biotsavart_groups_available = all(
        count > 0 for _, count in device_kernel_group_counts
    )
    return IterationTraceSummary(
        iteration=iteration,
        host_boundary_ns=host_boundary_ns,
        host_control_gap_ns=host_control_gap_ns,
        cuda_memcpy_ns=cuda_memcpy_ns,
        newton_adjoint_ns=category_ns["newton_adjoint"],
        other_attributed_ns=category_ns["other_attributed"],
        unattributed_ns=category_ns["unattributed"],
        device_active_ns=device_active_ns,
        active_ns=host_control_gap_ns + device_active_ns,
        required_phase_families_present=not missing_required_phases,
        missing_required_phases=missing_required_phases,
        semantic_counts_available=(
            semantic_counts.available and biotsavart_groups_available
        ),
        phase_active_ns=tuple(sorted(phase_active_ns.items())),
        raw_phase_duration_ns=tuple(
            sorted((phase.value, duration) for phase, duration in phase_raw_ns.items())
        ),
        semantic_solver_counts=semantic_counts.counts,
        device_interval_counts=tuple(sorted(device_interval_counts.items())),
        device_kernel_group_counts=device_kernel_group_counts,
        device_overlap_ns=raw_nested_ns - device_active_ns,
    )


def summarize_trace_document(
    document: Mapping[str, object],
    host_events: Sequence[HostEventRecord],
    *,
    child_id: str,
    expected_iterations: int = 7,
    evaluation_documents: Sequence[Mapping[str, object]] | None = None,
) -> TraceSummary:
    """Validate and summarize one raw profiler document plus host lifecycle."""

    if not child_id:
        raise TraceSummaryError("invalid_child_id", "child_id must be nonempty")
    if expected_iterations <= 0:
        raise TraceSummaryError(
            "iteration_correlation_invalid", "expected_iterations must be positive"
        )
    spans, device_pids = _parse_trace_document(document)
    accepted = _accepted_iteration_spans(spans, expected_iterations)
    trace_points = _trace_lifecycle_points(spans)
    _reject_compilation_inside_measurement(spans, trace_points)
    host_points = _host_lifecycle_points(host_events)
    _validate_lifecycle_points(host_points, trace_points)
    _validate_clock_anchors(host_points, trace_points)
    _validate_lifecycle_iteration_envelopes(trace_points, accepted)
    host_gaps = _host_control_gaps(host_points, expected_iterations)
    semantic_counts = _semantic_execution_counts(
        host_points, evaluation_documents, expected_iterations
    )
    transfer_spans = _host_transfer_spans(spans)
    device_intervals = _device_intervals(spans, device_pids, transfer_spans)
    _validate_host_gaps_exclude_device_work(trace_points, device_intervals)

    for item in device_intervals:
        overlapping_iterations = [
            iteration
            for iteration, span in accepted.items()
            if span.interval.overlaps(item.interval)
        ]
        if overlapping_iterations and not any(
            accepted[iteration].interval.contains(item.interval)
            for iteration in overlapping_iterations
        ):
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                "device interval crosses an accepted-iteration boundary",
            )

    iterations = tuple(
        _summarize_iteration(
            iteration,
            accepted[iteration].interval,
            device_intervals,
            host_gaps[iteration],
            semantic_counts[iteration],
        )
        for iteration in range(1, expected_iterations + 1)
    )
    in_window = tuple(
        item
        for item in device_intervals
        if any(span.interval.contains(item.interval) for span in accepted.values())
    )
    out_of_window = tuple(item for item in device_intervals if item not in in_window)
    _, total_device_active_ns = union_intervals(
        tuple(item.interval for item in device_intervals)
    )
    _, out_of_window_device_ns = union_intervals(
        tuple(item.interval for item in out_of_window)
    )
    raw_nested_device_ns = sum(item.interval.duration_ns for item in device_intervals)
    required_present = all(
        iteration.required_phase_families_present for iteration in iterations
    )
    semantic_counts_available = all(
        iteration.semantic_counts_available for iteration in iterations
    )
    diagnostics = (
        "trace_time_values_are_chrome_microseconds_converted_to_integer_ns",
        "host_perf_counter_and_profiler_timestamps_are_not_directly_unioned",
        "clock_correlation_policy=affine_bijective_lifecycle_anchors_v1",
        f"semantic_execution_counts_available={str(semantic_counts_available).lower()}",
        *(
            f"iteration={iteration.iteration}:missing_required_phase={phase}"
            for iteration in iterations
            for phase in iteration.missing_required_phases
        ),
    )
    unattributed_device_ns = sum(iteration.unattributed_ns for iteration in iterations)
    return TraceSummary(
        schema_id=SUMMARY_SCHEMA_ID,
        trace_schema_id=TRACE_SCHEMA_ID,
        child_id=child_id,
        trace_schema_valid=True,
        clock_correlation_valid=True,
        required_phase_families_present=required_present,
        semantic_counts_available=semantic_counts_available,
        diagnostics=diagnostics,
        iterations=iterations,
        raw_nested_device_ns=raw_nested_device_ns,
        device_active_ns=total_device_active_ns,
        device_overlap_ns=raw_nested_device_ns - total_device_active_ns,
        unattributed_device_ns=unattributed_device_ns,
        out_of_window_device_ns=out_of_window_device_ns,
    )


def _target_iteration_span(
    spans: Sequence[_TraceSpan], target_iteration: int
) -> _TraceSpan:
    step_spans = tuple(span for span in spans if span.name == _ACCEPTED_ITERATION_EVENT)
    if len(step_spans) != 1:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            f"segment must contain exactly one accepted-iteration envelope, found "
            f"{len(step_spans)}",
        )
    span = step_spans[0]
    step_num = span.args.get("step_num")
    if not isinstance(step_num, str) or not step_num.isdecimal():
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            "accepted iteration has invalid step_num",
        )
    if int(step_num) != target_iteration:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            f"segment step_num={step_num}, expected {target_iteration}",
        )
    return span


def _target_host_points(
    host_points: Sequence[_LifecyclePoint], target_iteration: int
) -> tuple[_LifecyclePoint, ...]:
    target = tuple(
        point
        for point in host_points
        if point.evaluation_kind is EvaluationKind.TRIAL
        and point.outer_iteration_id == target_iteration
    )
    if not target:
        raise TraceSummaryError(
            "host_correlation_invalid",
            f"full host audit has no trial evaluation for iteration {target_iteration}",
        )
    return target


def _validate_target_lifecycle_subset(
    full_host_points: Sequence[_LifecyclePoint],
    trace_points: Sequence[_LifecyclePoint],
    target_iteration: int,
) -> tuple[_LifecyclePoint, ...]:
    target_host_points = _target_host_points(full_host_points, target_iteration)
    for point in trace_points:
        if (
            point.evaluation_kind is not EvaluationKind.TRIAL
            or point.outer_iteration_id != target_iteration
        ):
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                "segment lifecycle contains an adjacent or non-trial evaluation",
            )
    _validate_lifecycle_points(target_host_points, trace_points)
    return target_host_points


def _target_pause_duration(
    host_gaps: Sequence[Interval], profiler_boundary_pauses: Sequence[Interval]
) -> int:
    ordered_pauses = tuple(sorted(profiler_boundary_pauses))
    if any(
        left.overlaps(right) for left, right in zip(ordered_pauses, ordered_pauses[1:])
    ):
        raise TraceSummaryError(
            "profiler_boundary_pause_invalid",
            "profiler boundary-pause intervals overlap",
        )
    for pause in ordered_pauses:
        containing_gaps = tuple(gap for gap in host_gaps if gap.contains(pause))
        if len(containing_gaps) != 1:
            raise TraceSummaryError(
                "profiler_boundary_pause_invalid",
                "each profiler boundary pause must be contained in exactly one "
                "target host-control gap",
            )
    pause_ns = sum(pause.duration_ns for pause in ordered_pauses)
    raw_host_gap_ns = sum(gap.duration_ns for gap in host_gaps)
    if pause_ns > raw_host_gap_ns:
        raise TraceSummaryError(
            "profiler_boundary_pause_invalid",
            "profiler boundary pauses exceed target host-control gaps",
        )
    return pause_ns


def _evaluation_ids_sha256(points: Sequence[_LifecyclePoint]) -> str:
    evaluation_ids = tuple(
        point.evaluation_id
        for point in points
        if point.event is HostEvent.EVALUATOR_ENTRY
    )
    return evaluation_ids_sha256(evaluation_ids)


def summarize_segmented_trace_document(
    document: Mapping[str, object],
    host_events: Sequence[HostEventRecord],
    *,
    child_id: str,
    sample_id: str,
    accepted_iteration: int,
    profiler_boundary_pauses: Sequence[Interval] = (),
    evaluation_documents: Sequence[Mapping[str, object]] | None = None,
) -> SegmentedIterationTraceSummary:
    """Summarize one raw trace that owns exactly one accepted iteration.

    ``host_events`` is the full child audit. Profiler lifecycle events must be
    the exact target-iteration subset, and pauses use the host audit clock.
    Device timestamps remain local to this trace and are never joined to peers.
    """

    if not child_id or not sample_id:
        raise TraceSummaryError(
            "invalid_child_id", "child_id and sample_id must be nonempty"
        )
    if accepted_iteration <= 0:
        raise TraceSummaryError(
            "iteration_correlation_invalid", "accepted_iteration must be positive"
        )
    expected_sample_id = f"iteration-{accepted_iteration:02d}"
    if sample_id != expected_sample_id:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            f"sample_id is {sample_id!r}, expected {expected_sample_id!r}",
        )
    spans, device_pids = _parse_trace_document(document)
    step_span = _target_iteration_span(spans, accepted_iteration)
    trace_points = _trace_lifecycle_points(spans)
    full_host_points = _host_lifecycle_points(host_events)
    target_host_points = _validate_target_lifecycle_subset(
        full_host_points, trace_points, accepted_iteration
    )
    _validate_clock_anchors(target_host_points, trace_points)
    _validate_lifecycle_iteration_envelopes(
        trace_points, {accepted_iteration: step_span}
    )
    for span in spans:
        if _is_compilation_event(span.name) and step_span.interval.overlaps(
            span.interval
        ):
            raise TraceSummaryError(
                "compilation_in_measurement",
                f"compilation event {span.name!r} overlaps the target step",
            )

    host_iteration_ids = tuple(
        point.outer_iteration_id
        for point in full_host_points
        if point.evaluation_kind is EvaluationKind.TRIAL
        and point.outer_iteration_id is not None
    )
    all_host_gaps = _host_control_gaps(full_host_points, max(host_iteration_ids))
    target_host_gaps = all_host_gaps[accepted_iteration]
    pause_ns = _target_pause_duration(target_host_gaps, profiler_boundary_pauses)
    filtered_documents = None
    if evaluation_documents is not None:
        target_ids = frozenset(
            point.evaluation_id
            for point in target_host_points
            if point.event is HostEvent.EVALUATOR_ENTRY
        )
        filtered_documents = tuple(
            document
            for document in evaluation_documents
            if document.get("evaluation_id") in target_ids
        )
    semantic_counts = _semantic_execution_counts(
        target_host_points,
        filtered_documents,
        1,
        iteration_ids=(accepted_iteration,),
    )[accepted_iteration]
    transfer_spans = _host_transfer_spans(spans)
    device_intervals = _device_intervals(spans, device_pids, transfer_spans)
    if any(not step_span.interval.contains(item.interval) for item in device_intervals):
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            "segmented device interval lies outside the target step envelope",
        )
    _validate_host_gaps_exclude_device_work(trace_points, device_intervals)
    raw_iteration = _summarize_iteration(
        accepted_iteration,
        step_span.interval,
        device_intervals,
        target_host_gaps,
        semantic_counts,
    )
    iteration = replace(
        raw_iteration,
        host_boundary_ns=raw_iteration.host_boundary_ns - pause_ns,
        host_control_gap_ns=raw_iteration.host_control_gap_ns - pause_ns,
        active_ns=raw_iteration.active_ns - pause_ns,
    )
    raw_nested_device_ns = sum(item.interval.duration_ns for item in device_intervals)
    diagnostics = (
        "segmented_profiler_policy=one_raw_trace_per_accepted_iteration_v2",
        "host_perf_counter_and_profiler_timestamps_are_not_directly_unioned",
        "clock_correlation_policy=per_segment_affine_bijective_lifecycle_anchors_v2",
        f"profiler_boundary_pause_ns={pause_ns}",
        *(
            f"missing_required_phase={phase}"
            for phase in iteration.missing_required_phases
        ),
    )
    return SegmentedIterationTraceSummary(
        schema_id=SEGMENT_SUMMARY_SCHEMA_ID,
        trace_schema_id=TRACE_SCHEMA_ID,
        child_id=child_id,
        sample_id=sample_id,
        accepted_iteration=accepted_iteration,
        segment_evaluation_ids_sha256=_evaluation_ids_sha256(target_host_points),
        trace_schema_valid=True,
        clock_correlation_valid=True,
        diagnostics=diagnostics,
        iteration=iteration,
        profiler_boundary_pause_ns=pause_ns,
        raw_active_ns=raw_iteration.active_ns,
        raw_nested_device_ns=raw_nested_device_ns,
        device_active_ns=iteration.device_active_ns,
        device_overlap_ns=iteration.device_overlap_ns,
        unattributed_device_ns=iteration.unattributed_ns,
    )


def combine_segmented_trace_summaries(
    segments: Sequence[SegmentedIterationTraceSummary],
    *,
    expected_iterations: int = 7,
) -> TraceSummary:
    """Combine independent segment durations without combining their clocks."""

    if expected_iterations != 7:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            "segmented-v2 requires the exact accepted-iteration targets 1..7",
        )
    by_iteration: dict[int, SegmentedIterationTraceSummary] = {}
    for segment in segments:
        if segment.schema_id != SEGMENT_SUMMARY_SCHEMA_ID:
            raise TraceSummaryError(
                "unknown_trace_schema", "segment summary schema is not v2"
            )
        if segment.trace_schema_id != TRACE_SCHEMA_ID:
            raise TraceSummaryError(
                "unknown_trace_schema", "segment trace schema differs from v2"
            )
        if segment.iteration.iteration != segment.accepted_iteration:
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                "segment iteration payload differs from accepted_iteration",
            )
        if (
            segment.raw_active_ns
            != segment.iteration.active_ns + segment.profiler_boundary_pause_ns
            or segment.device_active_ns != segment.iteration.device_active_ns
            or segment.device_overlap_ns != segment.iteration.device_overlap_ns
            or segment.unattributed_device_ns != segment.iteration.unattributed_ns
        ):
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                "segment aggregate fields differ from its iteration payload",
            )
        if segment.accepted_iteration in by_iteration:
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                f"duplicate segment iteration {segment.accepted_iteration}",
            )
        by_iteration[segment.accepted_iteration] = segment
    required = set(range(1, 8))
    if set(by_iteration) != required:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            f"segment targets are {sorted(by_iteration)}, expected {sorted(required)}",
        )
    child_ids = {segment.child_id for segment in segments}
    if len(child_ids) != 1:
        raise TraceSummaryError(
            "iteration_correlation_invalid",
            "all segments must have one child_id",
        )
    ordered = tuple(by_iteration[iteration] for iteration in range(1, 8))
    for segment in ordered:
        expected_sample_id = f"iteration-{segment.accepted_iteration:02d}"
        if segment.sample_id != expected_sample_id:
            raise TraceSummaryError(
                "iteration_correlation_invalid",
                f"segment {segment.accepted_iteration} sample_id is "
                f"{segment.sample_id!r}, expected {expected_sample_id!r}",
            )
    iterations = tuple(segment.iteration for segment in ordered)
    required_present = all(
        iteration.required_phase_families_present for iteration in iterations
    )
    semantic_available = all(
        iteration.semantic_counts_available for iteration in iterations
    )
    raw_nested_device_ns = sum(segment.raw_nested_device_ns for segment in ordered)
    device_active_ns = sum(segment.device_active_ns for segment in ordered)
    return TraceSummary(
        schema_id=SEGMENTED_SUMMARY_SCHEMA_ID,
        trace_schema_id=TRACE_SCHEMA_ID,
        child_id=next(iter(child_ids)),
        trace_schema_valid=all(segment.trace_schema_valid for segment in ordered),
        clock_correlation_valid=all(
            segment.clock_correlation_valid for segment in ordered
        ),
        required_phase_families_present=required_present,
        semantic_counts_available=semantic_available,
        diagnostics=(
            "segmented_profiler_policy=seven_independent_target_clocks_v2",
            "segment_timestamps_are_never_concatenated_or_unioned",
            f"profiler_boundary_pause_ns={sum(segment.profiler_boundary_pause_ns for segment in ordered)}",
            *(
                f"iteration={iteration.iteration}:missing_required_phase={phase}"
                for iteration in iterations
                for phase in iteration.missing_required_phases
            ),
        ),
        iterations=iterations,
        raw_nested_device_ns=raw_nested_device_ns,
        device_active_ns=device_active_ns,
        device_overlap_ns=sum(segment.device_overlap_ns for segment in ordered),
        unattributed_device_ns=sum(
            segment.unattributed_device_ns for segment in ordered
        ),
        out_of_window_device_ns=0,
    )


def load_trace_document(path: Path) -> Mapping[str, object]:
    """Load one exact JSON or gzip-compressed JSON profiler export."""

    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    else:
        raise TraceSummaryError(
            "unknown_trace_schema", "trace path must end in .json or .json.gz"
        )
    return _require_mapping(document, "trace document")


def summarize_segmented_trace(
    trace_path: Path,
    host_events: Sequence[HostEventRecord],
    *,
    child_id: str,
    sample_id: str,
    accepted_iteration: int,
    profiler_boundary_pauses: Sequence[Interval] = (),
    evaluation_documents: Sequence[Mapping[str, object]] | None = None,
) -> SegmentedIterationTraceSummary:
    """Load and independently summarize one accepted-iteration trace segment."""

    return summarize_segmented_trace_document(
        load_trace_document(trace_path),
        host_events,
        child_id=child_id,
        sample_id=sample_id,
        accepted_iteration=accepted_iteration,
        profiler_boundary_pauses=profiler_boundary_pauses,
        evaluation_documents=evaluation_documents,
    )


def summarize_trace(
    trace_path: Path,
    host_events: Sequence[HostEventRecord],
    *,
    child_id: str,
    expected_iterations: int = 7,
    evaluation_documents: Sequence[Mapping[str, object]] | None = None,
) -> TraceSummary:
    """Load and summarize a profiler trace without trusting saved summaries."""

    return summarize_trace_document(
        load_trace_document(trace_path),
        host_events,
        child_id=child_id,
        expected_iterations=expected_iterations,
        evaluation_documents=evaluation_documents,
    )


__all__ = (
    "SEGMENTED_SUMMARY_SCHEMA_ID",
    "SEGMENT_SUMMARY_SCHEMA_ID",
    "SUMMARY_SCHEMA_ID",
    "TRACE_SCHEMA_ID",
    "Interval",
    "IterationTraceSummary",
    "SegmentedIterationTraceSummary",
    "TraceSummary",
    "TraceSummaryError",
    "combine_segmented_trace_summaries",
    "load_trace_document",
    "summarize_segmented_trace",
    "summarize_segmented_trace_document",
    "summarize_trace",
    "summarize_trace_document",
    "union_intervals",
    "unique_deepest_scope",
)
