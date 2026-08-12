"""Profile the replay performed after one isolated C0 timed evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from simsopt_jax.runtime.trace_annotations import (
    EvaluationKind,
    HostEvent,
    evaluation_context,
    trace_session,
)

from benchmarks.process_gpu_monitor import ProcessGpuMemoryResult
from benchmarks.single_stage_changed_state_profiler_policy import (
    PROFILED_PROFILER_POLICY,
    build_jax_profiler_options,
)
from benchmarks.single_stage_compute_graph_c0_evaluator import (
    CAPTURE_SCHEMA_ID,
    CaptureEvidence,
    ChildRequest,
    EvaluationResult,
    _validate_result,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    HLO_MODULE_SET_IDENTITY_SOURCE,
    SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    canonical_hlo_module_set_identity,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    TRACE_SCHEMA_ID,
    Interval,
    TraceSummaryError,
    _device_intervals,
    _host_transfer_spans,
    _is_compilation_event,
    _parse_trace_document,
    _trace_lifecycle_points,
    load_trace_document,
)

IDENTITY_ANCHOR_SCHEMA_ID: Final = (
    "single-stage-compute-graph-c0-hlo-module-set-identity-anchor-v2"
)
_PJRT_PRIMARY_PREFIX: Final = "CommonPjRtLoadedExecutable::Execute ("
_PJRT_PRIMARY_EXACT: Final = "CommonPjRtLoadedExecutable::Execute"
_PJRT_FALLBACK_EXACT: Final = "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"


class C0CaptureError(RuntimeError):
    """The raw capture cannot support the required evidence claims."""


@dataclass(frozen=True, slots=True)
class TraceCaptureFacts:
    """Facts extracted from one exact evaluation trace envelope."""

    hlo_module_set_identity: str
    hlo_modules: tuple[str, ...]
    pjrt_execute_count: int
    kernel_launch_count: int


class PreparedEvaluation(Protocol):
    """Canonical evaluator surface used for warmup and capture."""

    def evaluate_once(self) -> EvaluationResult: ...


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, document: Mapping[str, object]) -> None:
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def _single_trace_path(trace_root: Path) -> Path:
    matches = tuple(sorted(trace_root.rglob("*.trace.json.gz")))
    if len(matches) != 1:
        raise C0CaptureError(
            f"expected exactly one JAX Chrome trace, found {len(matches)}"
        )
    return matches[0]


def _evaluation_envelope(spans: Sequence[object], parameter_sha256: str) -> Interval:
    points = _trace_lifecycle_points(spans)
    if tuple(point.event for point in points) != tuple(HostEvent):
        raise C0CaptureError(
            "trace must contain exactly ENTRY, READY, RETURN lifecycle points"
        )
    if any(
        point.evaluation_id != parameter_sha256
        or point.parameter_sha256 != parameter_sha256
        or point.evaluation_kind is not EvaluationKind.TRIAL
        for point in points
    ):
        raise C0CaptureError("trace lifecycle is not bound to the frozen candidate")
    return Interval(
        points[0].timestamp_ns,
        points[-1].timestamp_ns + points[-1].annotation_duration_ns,
    )


def _canonical_pjrt_execute_count(spans: Sequence[object], envelope: Interval) -> int:
    contained = tuple(span for span in spans if envelope.contains(span.interval))
    primary = tuple(
        span
        for span in contained
        if span.name == _PJRT_PRIMARY_EXACT
        or span.name.startswith(_PJRT_PRIMARY_PREFIX)
    )
    fallback = tuple(span for span in contained if span.name == _PJRT_FALLBACK_EXACT)
    selected = primary if primary else fallback
    if not selected:
        raise C0CaptureError("canonical PJRT execute events are unavailable")
    return len(selected)


def summarize_c0_trace(
    document: Mapping[str, object], parameter_sha256: str
) -> TraceCaptureFacts:
    """Extract non-duplicated PJRT, kernel, and HLO-module facts."""

    try:
        spans, device_pids = _parse_trace_document(document)
        envelope = _evaluation_envelope(spans, parameter_sha256)
        compilation = tuple(
            span
            for span in spans
            if envelope.overlaps(span.interval) and _is_compilation_event(span.name)
        )
        if compilation:
            raise C0CaptureError("compilation occurred inside the profiled envelope")
        device_intervals = _device_intervals(
            spans,
            device_pids,
            _host_transfer_spans(spans),
        )
    except TraceSummaryError as error:
        raise C0CaptureError(str(error)) from error
    kernel_spans = tuple(
        span
        for span in spans
        if span.pid in device_pids
        and envelope.contains(span.interval)
        and "kernel_details" in span.args
    )
    parsed_kernel_count = sum(
        interval.kind == "kernel" and envelope.contains(interval.interval)
        for interval in device_intervals
    )
    if not kernel_spans or parsed_kernel_count != len(kernel_spans):
        raise C0CaptureError(
            "CUDA kernel-detail events are unavailable or inconsistent"
        )
    hlo_modules = tuple(sorted({str(span.args["hlo_module"]) for span in kernel_spans}))
    if any(not module for module in hlo_modules):
        raise C0CaptureError("kernel HLO module identity is empty")
    identity = canonical_hlo_module_set_identity(hlo_modules)
    return TraceCaptureFacts(
        hlo_module_set_identity=identity,
        hlo_modules=hlo_modules,
        pjrt_execute_count=_canonical_pjrt_execute_count(spans, envelope),
        kernel_launch_count=len(kernel_spans),
    )


def _bind_identity_anchor(
    path: Path, request: ChildRequest, hlo_module_set_identity: str
) -> None:
    document = {
        "schema_id": IDENTITY_ANCHOR_SCHEMA_ID,
        "hlo_module_set_identity": hlo_module_set_identity,
        "hlo_module_set_identity_source": HLO_MODULE_SET_IDENTITY_SOURCE,
    }
    if request.mode == "profile":
        _write_exclusive(path, document)
        return
    try:
        anchor = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise C0CaptureError("HLO identity anchor is not valid JSON") from error
    if anchor != document:
        raise C0CaptureError("warm HLO module-set identity differs from first capture")


def build_capture_document(
    request: ChildRequest,
    parameter_sha256: str,
    facts: TraceCaptureFacts,
    memory: ProcessGpuMemoryResult,
) -> dict[str, object]:
    """Build the evaluator-compatible capture receipt with honest provenance."""

    if facts.hlo_module_set_identity != canonical_hlo_module_set_identity(
        facts.hlo_modules
    ):
        raise C0CaptureError("HLO module-set identity does not match trace modules")
    if facts.pjrt_execute_count <= 0 or facts.kernel_launch_count <= 0:
        raise C0CaptureError("trace execution and kernel counts must be positive")
    if not memory.samples or memory.peak_used_memory_mib != max(
        sample.used_memory_mib for sample in memory.samples
    ):
        raise C0CaptureError("sampled process GPU-memory peak is inconsistent")
    peak_bytes = memory.peak_used_memory_mib * 1024 * 1024
    if peak_bytes <= 0:
        raise C0CaptureError("sampled process GPU-memory peak must be positive")
    document: dict[str, object] = {
        "schema_id": CAPTURE_SCHEMA_ID,
        "trace_schema_id": TRACE_SCHEMA_ID,
        "mode": request.mode,
        "sample_index": request.sample_index,
        "parameter_sha256": parameter_sha256,
        "sampled_process_gpu_memory_peak_bytes": peak_bytes,
        "sampled_process_gpu_memory_source": SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
        "gpu_uuid": memory.gpu_uuid,
        "gpu_memory_sample_count": len(memory.samples),
        "hlo_module_set_identity": facts.hlo_module_set_identity,
        "hlo_module_set_identity_source": HLO_MODULE_SET_IDENTITY_SOURCE,
        "hlo_modules": list(facts.hlo_modules),
        "pjrt_execute_count": None,
        "kernel_launch_count": None,
    }
    if request.mode == "profile":
        document["pjrt_execute_count"] = facts.pjrt_execute_count
        document["kernel_launch_count"] = facts.kernel_launch_count
    return document


def build_capture_evidence(
    request: ChildRequest,
    parameter_sha256: str,
    facts: TraceCaptureFacts,
    memory: ProcessGpuMemoryResult,
) -> CaptureEvidence:
    """Return the typed subset consumed by the combined child observation."""

    document = build_capture_document(request, parameter_sha256, facts, memory)
    return CaptureEvidence(
        mode=request.mode,
        sample_index=request.sample_index,
        parameter_sha256=parameter_sha256,
        sampled_process_gpu_memory_peak_bytes=int(
            document["sampled_process_gpu_memory_peak_bytes"]
        ),
        sampled_process_gpu_memory_source=SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
        hlo_module_set_identity=facts.hlo_module_set_identity,
        hlo_module_set_identity_source=HLO_MODULE_SET_IDENTITY_SOURCE,
        pjrt_execute_count=(
            facts.pjrt_execute_count if request.mode == "profile" else None
        ),
        kernel_launch_count=(
            facts.kernel_launch_count if request.mode == "profile" else None
        ),
    )


def capture_profiled_replay(
    prepared: PreparedEvaluation,
    *,
    parameter_sha256: str,
    trace_root: Path,
) -> tuple[EvaluationResult, TraceCaptureFacts]:
    """Profile exactly one replay and derive its trace-backed execution facts."""

    import jax

    trace_root.mkdir(parents=True, exist_ok=False)
    with trace_session(), evaluation_context(
        parameter_sha256,
        parameter_sha256,
        EvaluationKind.TRIAL,
    ):
        jax.profiler.start_trace(
            str(trace_root),
            profiler_options=build_jax_profiler_options(
                jax.profiler.ProfileOptions,
                PROFILED_PROFILER_POLICY,
            ),
        )
        try:
            result = prepared.evaluate_once()
        finally:
            jax.profiler.stop_trace()
    _validate_result(result)
    document = load_trace_document(_single_trace_path(trace_root))
    return result, summarize_c0_trace(document, parameter_sha256)
