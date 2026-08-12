"""Strict Phase 0 profile evidence derived from one JAX Chrome trace."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from benchmarks.single_stage_compute_graph_c0_capture import (
    C0CaptureError,
    _canonical_pjrt_execute_count,
    _evaluation_envelope,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    _ALLOWED_PROFILE_PHASES,
    A100_LANE_ID,
    HLO_MODULE_SET_IDENTITY_SOURCE,
    MINIMUM_ATTRIBUTION_COVERAGE,
    RTX_LANE_ID,
    LaneId,
    canonical_hlo_module_set_identity,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    canonical_json_bytes as _phase0_canonical_json_bytes,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    TRACE_SCHEMA_ID,
    Interval,
    TraceSummaryError,
    _device_intervals,
    _host_transfer_spans,
    _is_compilation_event,
    _parse_trace_document,
    load_trace_document,
    union_intervals,
)

_PJRT_PRIMARY_PREFIX = "CommonPjRtLoadedExecutable::Execute ("
_PJRT_PRIMARY_EXACT = "CommonPjRtLoadedExecutable::Execute"
_PJRT_FALLBACK_EXACT = "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"
_CAPTURED_EXECUTION_MODES = frozenset(
    {"command_buffer", "command-buffer", "cuda_graph", "cuda_graph_launch"}
)
_UNCAPTURED_EXECUTION_MODES = frozenset({"direct", "uncaptured"})
_DEFAULT_EXECUTION_MODES = frozenset({"default"})
_SUPPORTED_EXECUTION_MODES = (
    _CAPTURED_EXECUTION_MODES | _UNCAPTURED_EXECUTION_MODES | _DEFAULT_EXECUTION_MODES
)
_COMMAND_BUFFER_EXECUTE_EXACT = "command_buffer::execute"
_CUDA_GRAPH_LAUNCH_PREFIX = "cuGraphLaunch (CudaGraph:"
_CUDA_GRAPH_LAUNCH_SUFFIX = ")"
_CUDA_GRAPH_ID_ARGUMENT = "cuda_graph_id"
_CUDA_GRAPH_NODE_ID_ARGUMENT = "cuda_graph_node_id"
PROFILE_EVIDENCE_SCHEMA_ID: Final = "single-stage-compute-graph-profile-evidence-v1"
_SHA256_HEX_LENGTH: Final = 64
_LOWER_HEX: Final = frozenset("0123456789abcdef")


class ComputeGraphProfileError(RuntimeError):
    """The trace cannot support a required Phase 0 profile claim."""


class Phase0ReceiptProfileMismatch(ComputeGraphProfileError):
    """Current receipt fields cannot honestly encode the available trace facts."""


@dataclass(frozen=True, slots=True)
class CommandBufferEvidence:
    """Trace-only command-buffer diagnostics; resolved XLA flags come externally."""

    observed_pjrt_execution_modes: tuple[str, ...]
    resolved_xla_configuration: None
    observed_capture_participation: bool
    command_buffer_execute_count: int
    graph_api_launch_count: int
    graph_device_activity_count: int
    graph_kernel_activity_count: int
    graph_memcpy_activity_count: int
    graph_other_device_activity_count: int
    direct_device_activity_count: int
    direct_kernel_activity_count: int
    direct_memcpy_activity_count: int
    direct_other_device_activity_count: int
    graph_device_union_ns: int
    direct_device_union_ns: int
    graph_direct_overlap_ns: int
    classified_device_union_ns: int

    def to_phase0_json(self) -> dict[str, object]:
        return {
            "observed_pjrt_execution_modes": list(self.observed_pjrt_execution_modes),
            "resolved_xla_configuration": self.resolved_xla_configuration,
            "observed_capture_participation": self.observed_capture_participation,
            "command_buffer_execute_count": self.command_buffer_execute_count,
            "graph_api_launch_count": self.graph_api_launch_count,
            "graph_device_activity_count": self.graph_device_activity_count,
            "graph_kernel_activity_count": self.graph_kernel_activity_count,
            "graph_memcpy_activity_count": self.graph_memcpy_activity_count,
            "graph_other_device_activity_count": (
                self.graph_other_device_activity_count
            ),
            "direct_device_activity_count": self.direct_device_activity_count,
            "direct_kernel_activity_count": self.direct_kernel_activity_count,
            "direct_memcpy_activity_count": self.direct_memcpy_activity_count,
            "direct_other_device_activity_count": (
                self.direct_other_device_activity_count
            ),
            "graph_device_union_ns": self.graph_device_union_ns,
            "direct_device_union_ns": self.direct_device_union_ns,
            "graph_direct_overlap_ns": self.graph_direct_overlap_ns,
            "classified_device_union_ns": self.classified_device_union_ns,
            "ab_control": None,
        }


@dataclass(frozen=True, slots=True)
class ComputeGraphProfile:
    """Recomputed profile and optional command-buffer evidence for one evaluation."""

    evaluation_envelope_ns: int
    device_active_ns: int
    phase_interval_unions: tuple[tuple[str, tuple[Interval, ...]], ...]
    attributed_union_ns: int
    unattributed_ns: int
    attribution_coverage: float
    pjrt_execute_count: int
    kernel_launch_count: int
    kernel_duration_ns: tuple[int, ...]
    inter_launch_gap_ns: int
    hlo_module_set_identity: str
    hlo_module_set_identity_source: str
    device_active_share: float
    inter_launch_gap_share: float
    command_buffer: CommandBufferEvidence | None
    command_buffer_unavailable_reason: str | None

    def profile_phase0_json(self) -> dict[str, object]:
        return {
            "evaluation_envelope_ns": self.evaluation_envelope_ns,
            "device_active_ns": self.device_active_ns,
            "phase_interval_unions": [
                {
                    "phase_id": phase_id,
                    "intervals": [
                        [interval.start_ns, interval.end_ns] for interval in intervals
                    ],
                }
                for phase_id, intervals in self.phase_interval_unions
            ],
            "attributed_union_ns": self.attributed_union_ns,
            "unattributed_ns": self.unattributed_ns,
            "attribution_coverage": self.attribution_coverage,
            "pjrt_execute_count": self.pjrt_execute_count,
            "kernel_launch_count": self.kernel_launch_count,
            "kernel_duration_ns": list(self.kernel_duration_ns),
            "inter_launch_gap_ns": self.inter_launch_gap_ns,
            "hlo_module_set_identity": self.hlo_module_set_identity,
            "hlo_module_set_identity_source": self.hlo_module_set_identity_source,
        }

    def phase0_documents(self) -> dict[str, object]:
        """Return exact current receipt fields, or fail when capture is unknowable."""

        if self.command_buffer is None:
            raise Phase0ReceiptProfileMismatch(
                "measurement.command_buffer requires observed boolean/count/duration "
                "evidence, but this trace cannot classify command-buffer launches: "
                f"{self.command_buffer_unavailable_reason}"
            )
        return {
            "profile": self.profile_phase0_json(),
            "command_buffer": self.command_buffer.to_phase0_json(),
        }

    def to_json(self) -> dict[str, object]:
        """Return a diagnostic document without inventing receipt evidence."""

        return {
            "profile": self.profile_phase0_json(),
            "device_active_share": self.device_active_share,
            "inter_launch_gap_share": self.inter_launch_gap_share,
            "command_buffer": (
                None
                if self.command_buffer is None
                else self.command_buffer.to_phase0_json()
            ),
            "command_buffer_unavailable_reason": self.command_buffer_unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class ProfileEvidenceIdentity:
    """Runner-owned identities that make one profile artifact non-transferable."""

    candidate_sha256: str
    specimen_sha256: str
    input_bundle_sha256: str
    source_sha256: str
    runtime_identity_sha256: str
    lane_id: LaneId
    gpu_uuid: str
    gate_checkpoint_sha256: str
    warm_checkpoint_sha256: str
    warm_p50_ns: float

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "specimen_sha256": self.specimen_sha256,
            "input_bundle_sha256": self.input_bundle_sha256,
            "source_sha256": self.source_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "lane_id": self.lane_id,
            "gpu_uuid": self.gpu_uuid,
            "gate_checkpoint_sha256": self.gate_checkpoint_sha256,
            "warm_checkpoint_sha256": self.warm_checkpoint_sha256,
            "warm_p50_ns": self.warm_p50_ns,
        }


@dataclass(frozen=True, slots=True)
class ProfileTraceBinding:
    """Repository-relative raw trace path and its exact byte identity."""

    path: str
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema_id": TRACE_SCHEMA_ID,
        }


@dataclass(frozen=True, slots=True)
class ComputeGraphProfileEvidence:
    """Self-validating wrapper for staged ingestion by the C0 runner."""

    identity: ProfileEvidenceIdentity
    trace: ProfileTraceBinding
    profile: ComputeGraphProfile

    def to_json(self) -> dict[str, object]:
        if self.profile.command_buffer is None:
            classification: dict[str, object] = {
                "state": "unavailable",
                "reason": self.profile.command_buffer_unavailable_reason,
            }
        else:
            classification = {
                "state": "available",
                "evidence": self.profile.command_buffer.to_phase0_json(),
            }
        return {
            "schema_id": PROFILE_EVIDENCE_SCHEMA_ID,
            "identity": self.identity.to_json(),
            "trace": self.trace.to_json(),
            "profile": self.profile.profile_phase0_json(),
            "diagnostics": {
                "device_active_share": self.profile.device_active_share,
                "inter_launch_gap_share": self.profile.inter_launch_gap_share,
                "command_buffer_classification": classification,
            },
        }

    def phase0_documents(self) -> dict[str, object]:
        """Expose only receipt-owned fields after command-buffer classification."""

        return self.profile.phase0_documents()


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    """Return deterministic, finite JSON suitable for hashing and receipts."""

    return _phase0_canonical_json_bytes(document)


def _sha256_text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ComputeGraphProfileError(f"{context} must be a lowercase SHA-256")
    return value


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ComputeGraphProfileError(f"{context} must be a nonempty exact string")
    return value


def _warm_p50(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise ComputeGraphProfileError("warm_p50_ns must be a finite positive float")
    if not math.isfinite(value) or value <= 0.0:
        raise ComputeGraphProfileError("warm_p50_ns must be a finite positive float")
    return value


def _lane_id(value: object) -> LaneId:
    if value == RTX_LANE_ID:
        return RTX_LANE_ID
    if value == A100_LANE_ID:
        return A100_LANE_ID
    raise ComputeGraphProfileError(
        f"lane_id must be {RTX_LANE_ID!r} or {A100_LANE_ID!r}"
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_relative_path(trace_path: Path, artifact_root: Path) -> str:
    try:
        root = artifact_root.resolve(strict=True)
        trace = trace_path.resolve(strict=True)
    except OSError as error:
        raise ComputeGraphProfileError(
            "artifact_root and trace_path must exist"
        ) from error
    if not trace.is_file():
        raise ComputeGraphProfileError("trace_path must be a regular file")
    try:
        relative = trace.relative_to(root)
    except ValueError as error:
        raise ComputeGraphProfileError(
            "trace_path must be contained by artifact_root"
        ) from error
    relative_path = PurePosixPath(relative.as_posix())
    if (
        not relative_path.parts
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ComputeGraphProfileError("trace path must be safe and relative")
    return relative_path.as_posix()


def _build_profile_evidence(
    *,
    trace_path: Path,
    artifact_root: Path,
    candidate_sha256: str,
    specimen_sha256: str,
    input_bundle_sha256: str,
    source_sha256: str,
    runtime_identity_sha256: str,
    lane_id: LaneId,
    gpu_uuid: str,
    gate_checkpoint_sha256: str,
    warm_checkpoint_sha256: str,
    warm_p50_ns: float,
    enforce_minimum_attribution_coverage: bool,
) -> ComputeGraphProfileEvidence:
    identity = ProfileEvidenceIdentity(
        candidate_sha256=_sha256_text(candidate_sha256, "candidate_sha256"),
        specimen_sha256=_sha256_text(specimen_sha256, "specimen_sha256"),
        input_bundle_sha256=_sha256_text(input_bundle_sha256, "input_bundle_sha256"),
        source_sha256=_sha256_text(source_sha256, "source_sha256"),
        runtime_identity_sha256=_sha256_text(
            runtime_identity_sha256, "runtime_identity_sha256"
        ),
        lane_id=_lane_id(lane_id),
        gpu_uuid=_nonempty_string(gpu_uuid, "gpu_uuid"),
        gate_checkpoint_sha256=_sha256_text(
            gate_checkpoint_sha256, "gate_checkpoint_sha256"
        ),
        warm_checkpoint_sha256=_sha256_text(
            warm_checkpoint_sha256, "warm_checkpoint_sha256"
        ),
        warm_p50_ns=_warm_p50(warm_p50_ns),
    )
    relative_trace = _trace_relative_path(trace_path, artifact_root)
    resolved_trace = artifact_root.resolve(strict=True) / relative_trace
    trace = ProfileTraceBinding(
        path=relative_trace,
        sha256=_sha256_path(resolved_trace),
    )
    profile = _summarize_compute_graph_profile(
        load_trace_document(resolved_trace),
        identity.candidate_sha256,
        enforce_minimum_attribution_coverage=enforce_minimum_attribution_coverage,
    )
    return ComputeGraphProfileEvidence(identity=identity, trace=trace, profile=profile)


def build_profile_evidence(
    *,
    trace_path: Path,
    artifact_root: Path,
    candidate_sha256: str,
    specimen_sha256: str,
    input_bundle_sha256: str,
    source_sha256: str,
    runtime_identity_sha256: str,
    lane_id: LaneId,
    gpu_uuid: str,
    gate_checkpoint_sha256: str,
    warm_checkpoint_sha256: str,
    warm_p50_ns: float,
) -> ComputeGraphProfileEvidence:
    """Bind one authoritative profile, retaining its strict 90% coverage gate."""

    return _build_profile_evidence(
        trace_path=trace_path,
        artifact_root=artifact_root,
        candidate_sha256=candidate_sha256,
        specimen_sha256=specimen_sha256,
        input_bundle_sha256=input_bundle_sha256,
        source_sha256=source_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        lane_id=lane_id,
        gpu_uuid=gpu_uuid,
        gate_checkpoint_sha256=gate_checkpoint_sha256,
        warm_checkpoint_sha256=warm_checkpoint_sha256,
        warm_p50_ns=warm_p50_ns,
        enforce_minimum_attribution_coverage=True,
    )


def build_attribution_control_profile_evidence(
    *,
    trace_path: Path,
    artifact_root: Path,
    candidate_sha256: str,
    specimen_sha256: str,
    input_bundle_sha256: str,
    source_sha256: str,
    runtime_identity_sha256: str,
    lane_id: LaneId,
    gpu_uuid: str,
    gate_checkpoint_sha256: str,
    warm_checkpoint_sha256: str,
    warm_p50_ns: float,
) -> ComputeGraphProfileEvidence:
    """Bind a non-timing control profile without granting receipt authority."""

    return _build_profile_evidence(
        trace_path=trace_path,
        artifact_root=artifact_root,
        candidate_sha256=candidate_sha256,
        specimen_sha256=specimen_sha256,
        input_bundle_sha256=input_bundle_sha256,
        source_sha256=source_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        lane_id=lane_id,
        gpu_uuid=gpu_uuid,
        gate_checkpoint_sha256=gate_checkpoint_sha256,
        warm_checkpoint_sha256=warm_checkpoint_sha256,
        warm_p50_ns=warm_p50_ns,
        enforce_minimum_attribution_coverage=False,
    )


def write_profile_evidence(path: Path, evidence: ComputeGraphProfileEvidence) -> None:
    """Exclusively persist one canonical profile artifact."""

    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(evidence.to_json()))
        stream.flush()
        os.fsync(stream.fileno())


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ComputeGraphProfileError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ComputeGraphProfileError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def parse_profile_evidence(
    path: Path, *, expected_identity: ProfileEvidenceIdentity
) -> ComputeGraphProfileEvidence:
    """Recompute one artifact and require its runner-owned expected identity."""

    raw_bytes = path.read_bytes()
    try:
        value = json.loads(
            raw_bytes,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ComputeGraphProfileError(
                    f"profile evidence contains non-finite constant {constant}"
                )
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ComputeGraphProfileError(
            "profile evidence is not valid UTF-8 JSON"
        ) from error
    document = _mapping(value, "profile evidence")
    if raw_bytes != canonical_json_bytes(document):
        raise ComputeGraphProfileError("profile evidence bytes are not canonical JSON")
    _exact_keys(
        document,
        frozenset({"schema_id", "identity", "trace", "profile", "diagnostics"}),
        "profile evidence",
    )
    if document["schema_id"] != PROFILE_EVIDENCE_SCHEMA_ID:
        raise ComputeGraphProfileError("profile evidence schema_id is unsupported")
    identity = _mapping(document["identity"], "profile evidence identity")
    _exact_keys(
        identity,
        frozenset(
            {
                "candidate_sha256",
                "specimen_sha256",
                "input_bundle_sha256",
                "source_sha256",
                "runtime_identity_sha256",
                "lane_id",
                "gpu_uuid",
                "gate_checkpoint_sha256",
                "warm_checkpoint_sha256",
                "warm_p50_ns",
            }
        ),
        "profile evidence identity",
    )
    trace = _mapping(document["trace"], "profile evidence trace")
    _exact_keys(
        trace,
        frozenset({"path", "sha256", "schema_id"}),
        "profile evidence trace",
    )
    trace_relative = PurePosixPath(
        _nonempty_string(trace["path"], "profile evidence trace.path")
    )
    if trace_relative.is_absolute() or ".." in trace_relative.parts:
        raise ComputeGraphProfileError(
            "profile evidence trace.path must be safe and relative"
        )
    trace_path = path.parent / Path(*trace_relative.parts)
    expected = build_profile_evidence(
        trace_path=trace_path,
        artifact_root=path.parent,
        candidate_sha256=_sha256_text(
            identity["candidate_sha256"], "identity.candidate_sha256"
        ),
        specimen_sha256=_sha256_text(
            identity["specimen_sha256"], "identity.specimen_sha256"
        ),
        input_bundle_sha256=_sha256_text(
            identity["input_bundle_sha256"], "identity.input_bundle_sha256"
        ),
        source_sha256=_sha256_text(identity["source_sha256"], "identity.source_sha256"),
        runtime_identity_sha256=_sha256_text(
            identity["runtime_identity_sha256"], "identity.runtime_identity_sha256"
        ),
        lane_id=_lane_id(identity["lane_id"]),
        gpu_uuid=_nonempty_string(identity["gpu_uuid"], "identity.gpu_uuid"),
        gate_checkpoint_sha256=_sha256_text(
            identity["gate_checkpoint_sha256"], "identity.gate_checkpoint_sha256"
        ),
        warm_checkpoint_sha256=_sha256_text(
            identity["warm_checkpoint_sha256"], "identity.warm_checkpoint_sha256"
        ),
        warm_p50_ns=_warm_p50(identity["warm_p50_ns"]),
    )
    if expected.identity != expected_identity:
        raise ComputeGraphProfileError(
            "profile evidence identity differs from runner-owned expected identity"
        )
    _sha256_text(trace["sha256"], "trace.sha256")
    if document != expected.to_json():
        raise ComputeGraphProfileError(
            "profile evidence differs from trace-recomputed canonical evidence"
        )
    return expected


def _canonical_pjrt_spans(
    spans: Sequence[object], envelope: Interval
) -> tuple[object, ...]:
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
        raise ComputeGraphProfileError("canonical PJRT execute events are unavailable")
    return selected


def _coalesced_kernel_intervals(kernel_spans: Sequence[object]) -> tuple[Interval, ...]:
    return union_intervals(tuple(span.interval for span in kernel_spans))[0]


def _inter_launch_gap_ns(
    kernel_spans: Sequence[object], device_intervals: Sequence[object]
) -> int:
    """Return kernel-to-kernel idle time disjoint from every device interval."""

    kernels = _coalesced_kernel_intervals(kernel_spans)
    if len(kernels) < 2:
        return 0
    gap_envelope = Interval(kernels[0].end_ns, kernels[-1].start_ns)
    active_intervals = union_intervals(
        tuple(item.interval for item in device_intervals)
    )[0]
    active_in_gap_envelope_ns = sum(
        max(
            0,
            min(gap_envelope.end_ns, active.end_ns)
            - max(gap_envelope.start_ns, active.start_ns),
        )
        for active in active_intervals
    )
    return gap_envelope.duration_ns - active_in_gap_envelope_ns


def _disjoint_phase_intervals(
    device_intervals: Sequence[object],
) -> tuple[tuple[str, tuple[Interval, ...]], ...]:
    events: list[tuple[int, int, str | None]] = []
    for item in device_intervals:
        attribution = item.attribution
        phase = attribution.phase
        phase_id = (
            phase.value
            if not attribution.ambiguous
            and phase is not None
            and phase.value in _ALLOWED_PROFILE_PHASES
            else None
        )
        events.append((item.interval.start_ns, 1, phase_id))
        events.append((item.interval.end_ns, -1, phase_id))
    events.sort(key=lambda event: event[0])

    fragments: dict[str, list[Interval]] = {}
    active_count = 0
    unattributed_count = 0
    phase_counts: dict[str, int] = {}
    event_index = 0
    previous_ns: int | None = None
    while event_index < len(events):
        timestamp_ns = events[event_index][0]
        if (
            previous_ns is not None
            and previous_ns < timestamp_ns
            and active_count > 0
            and unattributed_count == 0
            and len(phase_counts) == 1
        ):
            phase_id = next(iter(phase_counts))
            phase_fragments = fragments.setdefault(phase_id, [])
            if phase_fragments and phase_fragments[-1].end_ns == previous_ns:
                phase_fragments[-1] = Interval(
                    phase_fragments[-1].start_ns, timestamp_ns
                )
            else:
                phase_fragments.append(Interval(previous_ns, timestamp_ns))
        while event_index < len(events) and events[event_index][0] == timestamp_ns:
            _timestamp_ns, delta, phase_id = events[event_index]
            active_count += delta
            if phase_id is None:
                unattributed_count += delta
            else:
                updated = phase_counts.get(phase_id, 0) + delta
                if updated == 0:
                    del phase_counts[phase_id]
                else:
                    phase_counts[phase_id] = updated
            event_index += 1
        previous_ns = timestamp_ns
    return tuple(
        (phase_id, tuple(intervals))
        for phase_id, intervals in sorted(fragments.items())
    )


def _command_buffer_evidence(
    pjrt_spans: Sequence[object],
    kernel_spans: Sequence[object],
    device_spans: Sequence[object],
    spans: Sequence[object],
    envelope: Interval,
) -> tuple[CommandBufferEvidence | None, str | None]:
    module_modes: dict[str, str] = {}
    for span in pjrt_spans:
        mode = span.args.get("execution_mode")
        if not isinstance(mode, str):
            return None, "canonical PJRT execute lacks string execution_mode"
        normalized = mode.strip().lower()
        if normalized not in _SUPPORTED_EXECUTION_MODES:
            return None, f"unsupported canonical PJRT execution_mode={mode!r}"
        module = span.args.get("name")
        if not isinstance(module, str) or not module:
            return None, "canonical PJRT execute lacks a nonempty module name"
        previous = module_modes.setdefault(module, normalized)
        if previous != normalized:
            return None, f"HLO module {module!r} has mixed execution policies"

    command_buffer_executes = tuple(
        span
        for span in spans
        if span.name == _COMMAND_BUFFER_EXECUTE_EXACT
        and envelope.contains(span.interval)
    )
    graph_launches = tuple(
        span
        for span in spans
        if span.name.startswith(_CUDA_GRAPH_LAUNCH_PREFIX)
        and span.name.endswith(_CUDA_GRAPH_LAUNCH_SUFFIX)
        and envelope.contains(span.interval)
    )
    executes_by_thread: dict[tuple[int, int], list[object]] = {}
    launches_by_thread: dict[tuple[int, int], list[object]] = {}
    for execute in command_buffer_executes:
        executes_by_thread.setdefault((execute.pid, execute.tid), []).append(execute)
    for launch in graph_launches:
        launches_by_thread.setdefault((launch.pid, launch.tid), []).append(launch)
    if executes_by_thread.keys() != launches_by_thread.keys():
        return None, "command_buffer::execute and cuGraphLaunch threads contradict"

    launch_keys: set[tuple[str, str]] = set()
    for thread, thread_executes in executes_by_thread.items():
        thread_launches = launches_by_thread[thread]
        if len(thread_executes) != len(thread_launches):
            return None, "command_buffer::execute and cuGraphLaunch counts contradict"
        for launch in thread_launches:
            graph_id = launch.args.get(_CUDA_GRAPH_ID_ARGUMENT)
            correlation_id = launch.args.get("correlation_id")
            if (
                not isinstance(graph_id, str)
                or not graph_id.isdecimal()
                or int(graph_id) <= 0
            ):
                return None, "cuGraphLaunch lacks a positive decimal cuda_graph_id"
            if not isinstance(correlation_id, str) or not correlation_id:
                return None, "cuGraphLaunch lacks a nonempty correlation_id"
            name_graph_id = launch.name[
                len(_CUDA_GRAPH_LAUNCH_PREFIX) : -len(_CUDA_GRAPH_LAUNCH_SUFFIX)
            ]
            if name_graph_id != graph_id:
                return None, "cuGraphLaunch name and cuda_graph_id contradict"
            launch_key = (graph_id, correlation_id)
            if launch_key in launch_keys:
                return None, "duplicate cuGraphLaunch graph/correlation identity"
            launch_keys.add(launch_key)

    for kernel in kernel_spans:
        module = kernel.args.get("hlo_module")
        if not isinstance(module, str) or module not in module_modes:
            return None, "kernel HLO module is not bound to a classified PJRT execute"

    graph_activities: list[Interval] = []
    direct_activities: list[Interval] = []
    graph_activity_launch_keys: set[tuple[str, str]] = set()
    graph_kernel_count = 0
    graph_memcpy_count = 0
    graph_other_count = 0
    direct_kernel_count = 0
    direct_memcpy_count = 0
    direct_other_count = 0
    for activity in device_spans:
        has_kernel = "kernel_details" in activity.args
        has_memcpy = "memcpy_details" in activity.args
        has_graph_id = _CUDA_GRAPH_ID_ARGUMENT in activity.args
        has_node_id = _CUDA_GRAPH_NODE_ID_ARGUMENT in activity.args
        if has_graph_id != has_node_id:
            return None, "device graph metadata requires both graph and node IDs"
        if not has_graph_id:
            direct_activities.append(activity.interval)
            if has_kernel:
                direct_kernel_count += 1
            elif has_memcpy:
                direct_memcpy_count += 1
            else:
                direct_other_count += 1
            continue
        graph_id = activity.args[_CUDA_GRAPH_ID_ARGUMENT]
        node_id = activity.args[_CUDA_GRAPH_NODE_ID_ARGUMENT]
        correlation_id = activity.args.get("correlation_id")
        if (
            not isinstance(graph_id, str)
            or not graph_id.isdecimal()
            or int(graph_id) <= 0
            or not isinstance(node_id, str)
            or not node_id.isdecimal()
            or int(node_id) <= 0
        ):
            return None, "device graph and node IDs must be positive decimal strings"
        if not isinstance(correlation_id, str) or not correlation_id:
            return None, "graph device activity lacks a nonempty correlation_id"
        launch_key = (graph_id, correlation_id)
        if launch_key not in launch_keys:
            return None, "device graph metadata is not bound to a cuGraphLaunch"
        graph_activity_launch_keys.add(launch_key)
        graph_activities.append(activity.interval)
        if has_kernel:
            graph_kernel_count += 1
        elif has_memcpy:
            graph_memcpy_count += 1
        else:
            graph_other_count += 1
    if graph_activity_launch_keys != launch_keys:
        return None, "every cuGraphLaunch must bind at least one graph device activity"

    graph_ns = union_intervals(tuple(graph_activities))[1]
    direct_ns = union_intervals(tuple(direct_activities))[1]
    classified_union_ns = union_intervals((*graph_activities, *direct_activities))[1]
    overlap_ns = graph_ns + direct_ns - classified_union_ns
    return (
        CommandBufferEvidence(
            observed_pjrt_execution_modes=tuple(sorted(set(module_modes.values()))),
            resolved_xla_configuration=None,
            observed_capture_participation=bool(graph_launches),
            command_buffer_execute_count=len(command_buffer_executes),
            graph_api_launch_count=len(graph_launches),
            graph_device_activity_count=len(graph_activities),
            graph_kernel_activity_count=graph_kernel_count,
            graph_memcpy_activity_count=graph_memcpy_count,
            graph_other_device_activity_count=graph_other_count,
            direct_device_activity_count=len(direct_activities),
            direct_kernel_activity_count=direct_kernel_count,
            direct_memcpy_activity_count=direct_memcpy_count,
            direct_other_device_activity_count=direct_other_count,
            graph_device_union_ns=graph_ns,
            direct_device_union_ns=direct_ns,
            graph_direct_overlap_ns=overlap_ns,
            classified_device_union_ns=classified_union_ns,
        ),
        None,
    )


def _summarize_compute_graph_profile(
    document: Mapping[str, object],
    parameter_sha256: str,
    *,
    enforce_minimum_attribution_coverage: bool,
) -> ComputeGraphProfile:
    try:
        spans, device_pids = _parse_trace_document(document)
        envelope = _evaluation_envelope(spans, parameter_sha256)
        if any(
            envelope.overlaps(span.interval) and _is_compilation_event(span.name)
            for span in spans
        ):
            raise ComputeGraphProfileError(
                "compilation occurred inside the exact evaluation envelope"
            )
        all_device_intervals = _device_intervals(
            spans, device_pids, _host_transfer_spans(spans)
        )
    except (C0CaptureError, TraceSummaryError) as error:
        raise ComputeGraphProfileError(str(error)) from error

    device_intervals = tuple(
        item for item in all_device_intervals if envelope.contains(item.interval)
    )
    device_spans = tuple(
        span
        for span in spans
        if span.pid in device_pids and envelope.contains(span.interval)
    )
    if not device_intervals:
        raise ComputeGraphProfileError(
            "exact evaluation envelope contains no CUDA device intervals"
        )
    if len(device_spans) != len(device_intervals):
        raise ComputeGraphProfileError(
            "strict CUDA parser and raw device event counts disagree"
        )
    kernel_spans = tuple(span for span in device_spans if "kernel_details" in span.args)
    if not kernel_spans:
        raise ComputeGraphProfileError(
            "exact evaluation envelope contains no CUDA kernel-detail events"
        )
    parsed_kernel_count = sum(item.kind == "kernel" for item in device_intervals)
    if parsed_kernel_count != len(kernel_spans):
        raise ComputeGraphProfileError(
            "strict CUDA parser and raw kernel-detail event counts disagree"
        )

    phase_intervals = _disjoint_phase_intervals(device_intervals)
    attributed_intervals = tuple(
        interval for _phase_id, intervals in phase_intervals for interval in intervals
    )
    attributed_ns = union_intervals(attributed_intervals)[1]
    device_active_ns = union_intervals(
        tuple(item.interval for item in device_intervals)
    )[1]
    coverage = attributed_ns / device_active_ns
    if enforce_minimum_attribution_coverage and coverage < MINIMUM_ATTRIBUTION_COVERAGE:
        raise ComputeGraphProfileError(
            f"phase attribution coverage {coverage:.6f} is below "
            f"{MINIMUM_ATTRIBUTION_COVERAGE:.6f}"
        )

    pjrt_spans = _canonical_pjrt_spans(spans, envelope)
    pjrt_count = _canonical_pjrt_execute_count(spans, envelope)
    if pjrt_count != len(pjrt_spans):
        raise ComputeGraphProfileError("canonical PJRT execute count is inconsistent")
    command_buffer, unavailable_reason = _command_buffer_evidence(
        pjrt_spans, kernel_spans, device_spans, spans, envelope
    )
    if (
        command_buffer is not None
        and command_buffer.classified_device_union_ns != device_active_ns
    ):
        raise ComputeGraphProfileError(
            "command-buffer classification does not cover device-active union"
        )
    envelope_ns = envelope.duration_ns
    gap_ns = _inter_launch_gap_ns(kernel_spans, device_intervals)
    hlo_modules = tuple(sorted({str(span.args["hlo_module"]) for span in kernel_spans}))
    return ComputeGraphProfile(
        evaluation_envelope_ns=envelope_ns,
        device_active_ns=device_active_ns,
        phase_interval_unions=phase_intervals,
        attributed_union_ns=attributed_ns,
        unattributed_ns=device_active_ns - attributed_ns,
        attribution_coverage=coverage,
        pjrt_execute_count=pjrt_count,
        kernel_launch_count=len(kernel_spans),
        kernel_duration_ns=tuple(
            sorted(span.interval.duration_ns for span in kernel_spans)
        ),
        inter_launch_gap_ns=gap_ns,
        hlo_module_set_identity=canonical_hlo_module_set_identity(hlo_modules),
        hlo_module_set_identity_source=HLO_MODULE_SET_IDENTITY_SOURCE,
        device_active_share=device_active_ns / envelope_ns,
        inter_launch_gap_share=gap_ns / envelope_ns,
        command_buffer=command_buffer,
        command_buffer_unavailable_reason=unavailable_reason,
    )


def summarize_compute_graph_profile(
    document: Mapping[str, object], parameter_sha256: str
) -> ComputeGraphProfile:
    """Validate and summarize one authoritative candidate-evaluation envelope."""

    return _summarize_compute_graph_profile(
        document,
        parameter_sha256,
        enforce_minimum_attribution_coverage=True,
    )
