"""Independently validate a changed-state GPU timeline artifact.

The validator starts from manifested raw bytes, recomputes correlation,
numerical equivalence, interval unions, shares, medians, overhead, and the
terminal attribution verdict.  It never reads ``decision.json`` as input and
publishes exactly one fresh sibling validation result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
for _import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

import numpy as np
from simsopt_jax.parity_tolerances import PARITY_LADDER_TOLERANCES
from simsopt_jax.runtime.trace_annotations import (
    EvaluationKind,
    EvaluationTraceContext,
    HostEvent,
    HostEventRecord,
    PhaseId,
)

from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    TRACE_SCHEMA_ID,
    Interval,
    TraceSummaryError,
    combine_segmented_trace_summaries,
    summarize_segmented_trace_document,
)

ARTIFACT_SCHEMA_ID: Final = "single-stage-changed-state-gpu-timeline-segmented-v2"
CHILD_SCHEMA_ID: Final = "single-stage-changed-state-gpu-timeline-child-segmented-v2"
EVENT_SCHEMA_ID: Final = "single-stage-changed-state-gpu-timeline-event-segmented-v2"
OBSERVATION_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-observation-segmented-v2"
)
MANIFEST_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-manifest-segmented-v2"
)
VALIDATION_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-validation-segmented-v2"
)
EXPECTED_TRACE_SCHEMA_ID: Final = TRACE_SCHEMA_ID
EXPECTED_PHASE_SCHEMA_VERSION: Final = "single-stage-timeline-phases-v1"
PRODUCTION_ROUTE: Final = {
    "optimizer": "SIMSOPT_LBFGSB",
    "driver": "minimize_lbfgs_host_core",
    "line_search": "line_search_value_and_grad_host",
    "adjoint_route": "exact_jacobian_dense_fp64_lu",
}
EXPECTED_PHASE_IDS: Final = frozenset(
    {
        "host.h2d_submit",
        "host.line_search_control",
        "optimizer.lifecycle",
        "newton.warm_start",
        "newton.residual_jvp",
        "newton.linear_solve",
        "adjoint.outer_vjp_rhs",
        "adjoint.dense_matrix",
        "adjoint.lu_factor",
        "adjoint.lu_solve",
        "adjoint.refinement",
        "adjoint.implicit_coil_vjp",
        "biotsavart.forward",
        "biotsavart.vjp",
        "host.d2h_materialize",
    }
)
EXPECTED_MANIFEST_ROLES: Final = frozenset(
    {
        "artifact_metadata",
        "child_metadata",
        "host_device_events",
        "numerical_observations",
        "raw_trace",
        "trace_summary",
        "optimization_timing",
        "trajectory",
        "provenance",
        "diagnostic",
        "identity_preimages",
        "input_evidence",
        "preflight_evidence",
        "source_evidence",
    }
)
REQUIRED_PHASE_FAMILIES: Final = (
    frozenset({"host.h2d_submit", "host.d2h_materialize"}),
    frozenset({"host.line_search_control"}),
    frozenset({"newton.residual_jvp", "newton.linear_solve"}),
    frozenset(
        {
            "adjoint.outer_vjp_rhs",
            "adjoint.dense_matrix",
            "adjoint.lu_factor",
            "adjoint.lu_solve",
            "adjoint.refinement",
            "adjoint.implicit_coil_vjp",
        }
    ),
    frozenset({"biotsavart.forward", "biotsavart.vjp"}),
)
REQUIRED_ACCEPTED_ITERATIONS: Final = 7
PROFILE_CHILDREN: Final = 3
CONTROL_CHILDREN: Final = 3
EXPECTED_TRACE_VIEWER_MAX_EVENTS: Final = 67_108_864
EXPECTED_TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT: Final = (
    "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS"
)
EXPECTED_CUPTI_ACTIVITY_DROP_WARNING: Final = (
    "Already too many activity events, drop the buffer"
)
EXPECTED_PROFILED_PROFILER_POLICY: Final = {
    "enabled": True,
    "host_tracer_level": 1,
    "python_tracer_level": 0,
    "device_tracing": "jax_default",
    "trace_viewer_max_events": EXPECTED_TRACE_VIEWER_MAX_EVENTS,
    "advanced_configuration": {
        "gpu_max_activity_api_events": 33_554_432,
        "gpu_max_callback_api_events": 33_554_432,
    },
}
EXPECTED_CONTROL_PROFILER_POLICY: Final = {
    "enabled": False,
    "host_tracer_level": None,
    "python_tracer_level": None,
    "device_tracing": None,
    "trace_viewer_max_events": None,
    "advanced_configuration": {},
}
MAX_PROFILE_OVERHEAD: Final = 1.10
MAX_UNATTRIBUTED_SHARE: Final = 0.20
DOMINANCE_POOLED_SHARE: Final = 0.60
DOMINANCE_PROCESS_SHARE: Final = 0.50
DOMINANCE_LEAD: Final = 0.10
_GPU_RUNTIME_TOLERANCES: Final = PARITY_LADDER_TOLERANCES["gpu_runtime"]
NUMERICAL_RTOL: Final = float(_GPU_RUNTIME_TOLERANCES["same_state_forward_rtol"])
NUMERICAL_ATOL: Final = float(_GPU_RUNTIME_TOLERANCES["same_state_forward_atol"])
GRADIENT_RTOL: Final = float(_GPU_RUNTIME_TOLERANCES["same_state_gradient_rtol"])
GRADIENT_ATOL: Final = float(_GPU_RUNTIME_TOLERANCES["same_state_gradient_atol"])
_LOWER_HEX: Final = frozenset("0123456789abcdef")

Verdict = Literal[
    "HOST_BOUNDARY_DOMINANT",
    "NEWTON_ADJOINT_DOMINANT",
    "MIXED",
    "UNATTRIBUTABLE",
    "SCIENTIFIC_INVALID",
    "INTEGRITY_ERROR",
]


class IntegrityError(Exception):
    """Raw bytes are missing, malformed, drifted, or internally inconsistent."""


class AttributionError(Exception):
    """Valid raw evidence cannot support quantitative phase attribution."""


@dataclass(frozen=True)
class ManifestEntry:
    role: str
    relative_path: str
    size_bytes: int
    sha256: str
    source_state_sha256: str
    process_id: str
    evaluation_ids_sha256: str
    sample_id: str | None
    evaluation_id: str | None
    segment_evaluation_ids_sha256: str | None


@dataclass(frozen=True)
class ArtifactEvidence:
    entries: tuple[ManifestEntry, ...]
    files: Mapping[str, bytes]
    manifest_bytes: bytes


@dataclass(frozen=True)
class IdentityBindings:
    simsoptpp: Mapping[str, object]
    source_sha256_by_original_path: Mapping[str, str]


@dataclass(frozen=True)
class IterationShares:
    iteration: int
    host_boundary: float
    newton_adjoint: float
    other_attributed: float
    unattributed: float


@dataclass(frozen=True)
class ProfileEvidence:
    child_id: str
    pair_index: int
    active_wall_ns: int
    raw_wall_ns: int
    child_end_to_end_ns: int
    boundary_pause_ns: int
    process_host_median: float
    process_newton_median: float
    process_unattributed_median: float
    iteration_shares: tuple[IterationShares, ...]


@dataclass(frozen=True)
class ControlEvidence:
    child_id: str
    pair_index: int
    active_wall_ns: int
    raw_wall_ns: int
    child_end_to_end_ns: int


@dataclass(frozen=True)
class HostSpanEvidence:
    record_type: Literal["host_span", "optimizer_span"]
    sequence: int
    phase: PhaseId
    start_ns: int
    end_ns: int
    depth: int
    attributes: tuple[tuple[str, str | int | float | bool], ...]


def _fail(message: str) -> IntegrityError:
    return IntegrityError(message)


def _canonical_json_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_canonical_json(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise _fail(f"missing or non-regular JSON file: {path}")
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical_json_bytes(document):
        raise _fail(f"non-canonical JSON encoding: {path}")
    return document


def _load_canonical_json_bytes(raw: bytes, context: str) -> object:
    document = json.loads(raw)
    if raw != _canonical_json_bytes(document):
        raise _fail(f"non-canonical JSON encoding: {context}")
    return document


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise _fail(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise _fail(f"{context} must be a JSON array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{context} must be a non-empty string")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(character not in _LOWER_HEX for character in digest):
        raise _fail(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _evaluation_ids_sha256(evaluation_ids: Sequence[str]) -> str:
    if len(evaluation_ids) != len(set(evaluation_ids)) or any(
        not evaluation_id for evaluation_id in evaluation_ids
    ):
        raise _fail("evaluation IDs must be nonempty and unique")
    canonical_ids = sorted(evaluation_ids)
    return hashlib.sha256(_canonical_json_bytes(canonical_ids)).hexdigest()


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{context} must be an integer")
    return value


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(f"{context} must be finite")
    return number


def _parameter_vector_and_sha256(
    value: object, context: str
) -> tuple[tuple[float, ...], str]:
    vector = tuple(
        _finite_float(component, f"{context} component")
        for component in _sequence(value, context)
    )
    if not vector:
        raise _fail(f"{context} must be nonempty")
    content = struct.pack(f"<{len(vector)}d", *vector)
    return vector, hashlib.sha256(content).hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not value
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise _fail(f"unsafe manifest path: {value!r}")
    return path


def _live_worktree_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for arguments in (
        ("rev-parse", "HEAD"),
        ("status", "--porcelain=v1", "-z"),
        ("diff", "--binary", "HEAD"),
    ):
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise _fail(f"cannot recompute source state: git {' '.join(arguments)}")
        digest.update(len(completed.stdout).to_bytes(8, "little"))
        digest.update(completed.stdout)
    untracked = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if untracked.returncode != 0:
        raise _fail("cannot enumerate untracked source-state files")
    for relative_bytes in sorted(
        path for path in untracked.stdout.split(b"\0") if path
    ):
        relative_path = relative_bytes.decode("utf-8")
        path = repo_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise _fail(
                f"untracked source-state path is not a regular file: {relative_path}"
            )
        content = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "little"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _load_manifest(root: Path) -> ArtifactEvidence:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise _fail(f"missing or non-regular JSON file: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    document = _mapping(
        _load_canonical_json_bytes(manifest_bytes, "manifest.json"), "manifest"
    )
    if set(document) != {"schema_id", "artifact_schema_id", "entries"}:
        raise _fail("manifest top-level schema fields drifted")
    if document.get("schema_id") != MANIFEST_SCHEMA_ID:
        raise _fail(f"unknown manifest schema: {document.get('schema_id')!r}")
    if document.get("artifact_schema_id") != ARTIFACT_SCHEMA_ID:
        raise _fail("manifest artifact schema does not match timeline schema")
    raw_entries = _sequence(document.get("entries"), "manifest.entries")
    entries: list[ManifestEntry] = []
    paths: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"manifest.entries[{index}]")
        if set(entry) != {
            "role",
            "relative_path",
            "size_bytes",
            "sha256",
            "source_state_sha256",
            "process_id",
            "evaluation_ids_sha256",
            "sample_id",
            "evaluation_id",
            "segment_evaluation_ids_sha256",
        }:
            raise _fail(f"manifest.entries[{index}] schema fields drifted")
        relative_path = _string(entry.get("relative_path"), "relative_path")
        _safe_relative(relative_path)
        if relative_path in paths:
            raise _fail(f"duplicate manifest path: {relative_path}")
        paths.add(relative_path)
        role = _string(entry.get("role"), "role")
        if role not in EXPECTED_MANIFEST_ROLES:
            raise _fail(f"unknown manifest role: {role}")
        process_id = _string(entry.get("process_id"), "process_id")
        sample_id = (
            None
            if entry.get("sample_id") is None
            else _string(entry.get("sample_id"), "sample_id")
        )
        if process_id != "artifact" and sample_id is None:
            raise _fail(f"{relative_path}: child evidence does not bind sample_id")
        raw_segment_digest = entry.get("segment_evaluation_ids_sha256")
        if role in {"raw_trace", "trace_summary"}:
            _sha256(raw_segment_digest, "segment_evaluation_ids_sha256")
        elif raw_segment_digest is not None:
            raise _fail(
                f"{relative_path}: non-segment evidence has segment evaluation binding"
            )
        entries.append(
            ManifestEntry(
                role=role,
                relative_path=relative_path,
                size_bytes=_integer(entry.get("size_bytes"), "size_bytes"),
                sha256=_string(entry.get("sha256"), "sha256"),
                source_state_sha256=_string(
                    entry.get("source_state_sha256"), "source_state_sha256"
                ),
                process_id=process_id,
                evaluation_ids_sha256=_sha256(
                    entry.get("evaluation_ids_sha256"), "evaluation_ids_sha256"
                ),
                sample_id=sample_id,
                evaluation_id=(
                    None
                    if entry.get("evaluation_id") is None
                    else _string(entry.get("evaluation_id"), "evaluation_id")
                ),
                segment_evaluation_ids_sha256=(
                    None
                    if entry.get("segment_evaluation_ids_sha256") is None
                    else _sha256(
                        entry.get("segment_evaluation_ids_sha256"),
                        "segment_evaluation_ids_sha256",
                    )
                ),
            )
        )
    if not entries:
        raise _fail("manifest has no entries")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_paths = paths | {"manifest.json"}
    if actual_paths != expected_paths:
        raise _fail(
            "artifact file set differs from manifest: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    files: dict[str, bytes] = {}
    for entry in entries:
        path = root / entry.relative_path
        if path.is_symlink() or not path.is_file():
            raise _fail(f"manifest path is not a regular file: {entry.relative_path}")
        content = path.read_bytes()
        if len(content) != entry.size_bytes:
            raise _fail(f"size mismatch: {entry.relative_path}")
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            raise _fail(f"SHA256 mismatch: {entry.relative_path}")
        files[entry.relative_path] = content
    return ArtifactEvidence(tuple(entries), files, manifest_bytes)


def _entry_by_role(
    entries: tuple[ManifestEntry, ...], process_id: str, role: str
) -> ManifestEntry:
    matches = [
        entry
        for entry in entries
        if entry.process_id == process_id and entry.role == role
    ]
    if len(matches) != 1:
        raise _fail(
            f"{process_id}: expected exactly one {role!r} manifest entry, "
            f"found {len(matches)}"
        )
    return matches[0]


def _segment_entries_by_role(
    entries: tuple[ManifestEntry, ...], process_id: str, role: str, *, complete: bool
) -> tuple[ManifestEntry, ...]:
    """Return a paired prefix, requiring all seven for a complete child."""

    matches = tuple(
        entry
        for entry in entries
        if entry.process_id == process_id and entry.role == role
    )
    all_sample_ids = tuple(
        f"iteration-{iteration:02d}"
        for iteration in range(1, REQUIRED_ACCEPTED_ITERATIONS + 1)
    )
    by_sample_id = {entry.sample_id: entry for entry in matches}
    expected_sample_ids = all_sample_ids if complete else all_sample_ids[: len(matches)]
    if len(matches) != len(by_sample_id):
        raise _fail(f"{process_id}: {role} segment IDs differ from the frozen protocol")
    if set(by_sample_id) != set(expected_sample_ids):
        if complete and set(by_sample_id) < set(expected_sample_ids):
            raise AttributionError(f"{process_id}:missing_{role}_segment")
        raise _fail(f"{process_id}: {role} segment IDs differ from the frozen protocol")
    ordered = tuple(by_sample_id[sample_id] for sample_id in expected_sample_ids)
    expected_filename = "trace.json.gz" if role == "raw_trace" else "trace_summary.json"
    for sample_id, entry in zip(expected_sample_ids, ordered, strict=True):
        expected_path = (
            f"children/{process_id}/segments/{sample_id}/{expected_filename}"
        )
        if entry.relative_path != expected_path:
            raise _fail(
                f"{process_id}: {role} segment path differs from frozen protocol"
            )
        if entry.evaluation_id is None:
            raise _fail(f"{process_id}: {role} segment omits evaluation binding")
    return ordered


def _expected_profiler_policy(mode: object) -> Mapping[str, object]:
    if mode == "profiled":
        return EXPECTED_PROFILED_PROFILER_POLICY
    if mode == "control":
        return EXPECTED_CONTROL_PROFILER_POLICY
    raise _fail(f"unknown child mode for profiler policy: {mode!r}")


def _profiler_retention_evidence(
    evidence: ArtifactEvidence,
    entries: tuple[ManifestEntry, ...],
    scheduled: Mapping[str, object],
) -> bool:
    """Validate the manifested monitor and return whether CUPTI dropped buffers."""

    child_id = _string(scheduled.get("child_id"), "child_id")
    expected_relative_path = f"children/{child_id}/monitor.json"
    expected_diagnostic_paths = {
        expected_relative_path,
        f"children/{child_id}/stdout.log",
        f"children/{child_id}/stderr.log",
    }
    actual_diagnostic_paths = {
        entry.relative_path
        for entry in entries
        if entry.process_id == child_id and entry.role == "diagnostic"
    }
    if actual_diagnostic_paths != expected_diagnostic_paths:
        raise _fail(f"{child_id}: diagnostic file set differs from protocol")
    matches = tuple(
        entry
        for entry in entries
        if entry.process_id == child_id
        and entry.role == "diagnostic"
        and entry.relative_path == expected_relative_path
    )
    if len(matches) != 1:
        raise _fail(f"{child_id}: expected exactly one manifested monitor.json")
    monitor = _mapping(
        _load_canonical_json_bytes(
            evidence.files[expected_relative_path], expected_relative_path
        ),
        f"{child_id}.monitor",
    )
    retention = _mapping(
        monitor.get("profiler_retention"), f"{child_id}.profiler_retention"
    )
    if set(retention) != {
        "evidence_available",
        "activity_buffers_dropped",
        "warning",
    }:
        raise _fail(f"{child_id}: profiler retention schema drift")
    if scheduled.get("mode") == "profiled":
        dropped = retention.get("activity_buffers_dropped")
        if retention.get("evidence_available") is not True or not isinstance(
            dropped, bool
        ):
            raise _fail(f"{child_id}: profiled retention evidence is unavailable")
        expected_warning = EXPECTED_CUPTI_ACTIVITY_DROP_WARNING if dropped else None
        if retention.get("warning") != expected_warning:
            raise _fail(f"{child_id}: CUPTI drop warning contradicts retention state")
        raw_stderr = evidence.files[f"children/{child_id}/stderr.log"]
        warning_present = EXPECTED_CUPTI_ACTIVITY_DROP_WARNING.encode() in raw_stderr
        if warning_present is not dropped:
            raise _fail(f"{child_id}: raw stderr contradicts CUPTI retention state")
        return dropped
    if retention != {
        "evidence_available": False,
        "activity_buffers_dropped": None,
        "warning": None,
    }:
        raise _fail(f"{child_id}: control child claims profiler retention evidence")
    if (
        EXPECTED_CUPTI_ACTIVITY_DROP_WARNING.encode()
        in evidence.files[f"children/{child_id}/stderr.log"]
    ):
        raise _fail(f"{child_id}: unprofiled control emitted a CUPTI drop warning")
    return False


def _validate_identity_preimages(
    evidence: ArtifactEvidence, artifact: Mapping[str, object]
) -> IdentityBindings:
    empty_evaluation_digest = _evaluation_ids_sha256(())
    if any(
        entry.evaluation_ids_sha256 != empty_evaluation_digest
        for entry in evidence.entries
        if entry.process_id == "artifact"
    ):
        raise _fail("artifact-level evidence has nonempty evaluation binding")
    preflight_entry = _entry_by_role(evidence.entries, "artifact", "preflight_evidence")
    preflight = _mapping(
        _load_canonical_json_bytes(
            evidence.files[preflight_entry.relative_path],
            preflight_entry.relative_path,
        ),
        "preflight evidence",
    )
    expected_preflight_keys = {
        "schema_id",
        "state",
        "trace_schema_id",
        "required_scopes",
        "observed_evidence",
        "device_identity",
        "profiler_policy",
        "session_evidence",
        "failure_reason",
    }
    required_scopes = ("newton.residual_jvp", "adjoint.lu_solve")
    if (
        set(preflight) != expected_preflight_keys
        or preflight.get("schema_id") != "single-stage-changed-state-trace-preflight-v2"
        or preflight.get("state") != "pass"
        or preflight.get("trace_schema_id") != EXPECTED_TRACE_SCHEMA_ID
        or tuple(_sequence(preflight.get("required_scopes"), "required_scopes"))
        != required_scopes
        or preflight.get("failure_reason") is not None
        or preflight.get("profiler_policy") != EXPECTED_PROFILED_PROFILER_POLICY
        or _mapping(preflight.get("device_identity"), "preflight device")
        != {"name": artifact.get("device_name"), "uuid": artifact.get("device_uuid")}
    ):
        raise _fail("trace preflight schema, scope, state, or device drift")
    sessions = tuple(
        _mapping(value, f"preflight.session_evidence[{index}]")
        for index, value in enumerate(
            _sequence(preflight.get("session_evidence"), "session_evidence")
        )
    )
    if len(sessions) != 2:
        raise _fail("trace preflight does not contain two sequential sessions")
    session_observations: list[tuple[Mapping[str, object], ...]] = []
    for index, session in enumerate(sessions, start=1):
        if set(session) != {"session_id", "device_processes", "observed_evidence"}:
            raise _fail("trace preflight session evidence differs from protocol")
        if session.get("session_id") != f"session-{index:02d}" or session.get(
            "device_processes"
        ) != ["/device:GPU:0"]:
            raise _fail("trace preflight session identity differs from protocol")
        session_observations.append(
            tuple(
                _mapping(value, "preflight session observation")
                for value in _sequence(
                    session.get("observed_evidence"), "session observed_evidence"
                )
            )
        )
    aggregate_by_phase = {
        _string(observation.get("phase_id"), "preflight phase_id"): observation
        for observation in _sequence(
            preflight.get("observed_evidence"), "preflight observed_evidence"
        )
        if isinstance(observation, Mapping)
    }
    count_fields = (
        "device_kernel_intervals_containing_scope",
        "uniquely_attributed_device_kernel_intervals",
        "ambiguous_device_kernel_intervals",
    )
    for phase in required_scopes:
        per_session = []
        for observations in session_observations:
            matches = tuple(
                observation
                for observation in observations
                if observation.get("phase_id") == phase
            )
            if len(matches) != 1:
                raise _fail("preflight session phase evidence is incomplete")
            containing = _integer(matches[0].get(count_fields[0]), count_fields[0])
            unique = _integer(matches[0].get(count_fields[1]), count_fields[1])
            ambiguous = _integer(matches[0].get(count_fields[2]), count_fields[2])
            if containing < 1 or unique < 1 or unique > containing or ambiguous != 0:
                raise _fail("preflight session scope evidence is not attributable")
            per_session.append(matches[0])
        aggregate = aggregate_by_phase.get(phase)
        if aggregate is None or any(
            aggregate.get(field)
            != sum(_integer(item.get(field), field) for item in per_session)
            for field in count_fields
        ):
            raise _fail("preflight aggregate counts do not recompute from sessions")
    observed_preflight = tuple(
        _mapping(row, f"observed_evidence[{index}]")
        for index, row in enumerate(
            _sequence(preflight.get("observed_evidence"), "observed_evidence")
        )
    )
    if tuple(row.get("phase_id") for row in observed_preflight) != required_scopes:
        raise _fail("trace preflight observed scopes differ from required scopes")
    for row in observed_preflight:
        if set(row) != {
            "phase_id",
            "device_kernel_intervals_containing_scope",
            "uniquely_attributed_device_kernel_intervals",
            "ambiguous_device_kernel_intervals",
        }:
            raise _fail("trace preflight scope-survival evidence failed")
        containing_intervals = _integer(
            row.get("device_kernel_intervals_containing_scope"),
            "preflight containing intervals",
        )
        unique_intervals = _integer(
            row.get("uniquely_attributed_device_kernel_intervals"),
            "preflight unique intervals",
        )
        ambiguous_intervals = _integer(
            row.get("ambiguous_device_kernel_intervals"),
            "preflight ambiguous intervals",
        )
        if (
            containing_intervals <= 0
            or unique_intervals <= 0
            or unique_intervals > containing_intervals
            or ambiguous_intervals != 0
        ):
            raise _fail("trace preflight scope-survival evidence failed")
    identity_entry = _entry_by_role(evidence.entries, "artifact", "identity_preimages")
    identity = _mapping(
        _load_canonical_json_bytes(
            evidence.files[identity_entry.relative_path], identity_entry.relative_path
        ),
        "identity_preimages",
    )
    expected_identity_keys = {
        "schema_id",
        "input_fingerprint_payload",
        "configuration",
        "construction_fingerprint_payload",
        "runtime_policy_payload",
        "source_preimages",
        "simsoptpp",
    }
    if (
        set(identity) != expected_identity_keys
        or identity.get("schema_id")
        != "single-stage-changed-state-gpu-timeline-identity-preimages-v1"
    ):
        raise _fail("identity preimage schema drift")

    input_entries = tuple(
        entry
        for entry in evidence.entries
        if entry.process_id == "artifact" and entry.role == "input_evidence"
    )
    bundle_entries = tuple(
        entry
        for entry in input_entries
        if entry.relative_path == "inputs/input_bundle.json"
    )
    if len(bundle_entries) != 1:
        raise _fail("input evidence requires exactly one input bundle")
    bundle = _mapping(
        _load_canonical_json_bytes(
            evidence.files[bundle_entries[0].relative_path],
            bundle_entries[0].relative_path,
        ),
        "input_bundle",
    )
    expected_bundle_keys = {
        "schema_version",
        "case_id",
        "scale",
        "random_seed",
        "configuration",
        "configuration_fingerprint",
        "arrays",
        "input_fingerprint",
    }
    if set(bundle) != expected_bundle_keys or bundle.get("schema_version") != 2:
        raise _fail("input bundle schema drift")
    configuration = _mapping(bundle.get("configuration"), "configuration")
    configuration_sha256 = hashlib.sha256(
        _canonical_json_bytes(configuration)
    ).hexdigest()
    if (
        configuration_sha256 != bundle.get("configuration_fingerprint")
        or configuration_sha256 != artifact.get("configuration_sha256")
        or configuration != _mapping(identity.get("configuration"), "configuration")
    ):
        raise _fail("configuration identity does not recompute from raw bundle")

    raw_references = _mapping(bundle.get("arrays"), "input bundle arrays")
    references: dict[str, Mapping[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    expected_input_paths = {"inputs/input_bundle.json"}
    for name, raw_reference in sorted(raw_references.items()):
        if not isinstance(name, str) or not name:
            raise _fail("input array name must be nonempty")
        reference = _mapping(raw_reference, f"array reference {name}")
        if set(reference) != {"path", "dtype", "shape", "order", "sha256"}:
            raise _fail(f"array reference schema drift: {name}")
        relative = _string(reference.get("path"), f"{name}.path")
        _safe_relative(relative)
        evidence_path = f"inputs/{relative}"
        expected_input_paths.add(evidence_path)
        raw_array = evidence.files.get(evidence_path)
        if raw_array is None or hashlib.sha256(raw_array).hexdigest() != reference.get(
            "sha256"
        ):
            raise _fail(f"array evidence hash mismatch: {name}")
        array = np.load(io.BytesIO(raw_array), allow_pickle=False)
        shape = tuple(
            _integer(component, f"{name}.shape")
            for component in _sequence(reference.get("shape"), f"{name}.shape")
        )
        if (
            array.dtype.str != reference.get("dtype")
            or tuple(array.shape) != shape
            or reference.get("order") != "C"
            or not array.flags.c_contiguous
        ):
            raise _fail(f"array evidence metadata mismatch: {name}")
        references[name] = reference
        arrays[name] = array
    if {entry.relative_path for entry in input_entries} != expected_input_paths:
        raise _fail("input evidence file set differs from bundle references")

    input_payload = {
        "case_id": bundle.get("case_id"),
        "scale": bundle.get("scale"),
        "random_seed": bundle.get("random_seed"),
        "configuration_fingerprint": bundle.get("configuration_fingerprint"),
        "arrays": references,
    }
    input_sha256 = hashlib.sha256(_canonical_json_bytes(input_payload)).hexdigest()
    if (
        input_payload
        != _mapping(identity.get("input_fingerprint_payload"), "input payload")
        or input_sha256 != bundle.get("input_fingerprint")
        or input_sha256 != artifact.get("input_sha256")
    ):
        raise _fail("input identity does not recompute from raw evidence")

    if "surface_dofs" not in arrays or "coil_dofs" not in arrays:
        raise _fail("construction evidence omits surface or coil DOFs")
    raw_value_hashes = {
        name: hashlib.sha256(
            np.ascontiguousarray(arrays[name], dtype=np.dtype("<f8"))
            .reshape(-1)
            .tobytes(order="C")
        ).hexdigest()
        for name in ("surface_dofs", "coil_dofs")
    }
    construction_payload = {
        "case_id": bundle.get("case_id"),
        "scale": bundle.get("scale"),
        "random_seed": bundle.get("random_seed"),
        "applied_construction": {
            **raw_value_hashes,
            **configuration,
        },
    }
    if construction_payload != _mapping(
        identity.get("construction_fingerprint_payload"), "construction payload"
    ) or hashlib.sha256(_canonical_json_bytes(construction_payload)).hexdigest() != (
        artifact.get("construction_sha256")
    ):
        raise _fail("construction identity does not recompute from raw evidence")

    runtime_policy = _mapping(
        identity.get("runtime_policy_payload"), "runtime policy payload"
    )
    if runtime_policy.get("profiler_policy") != {
        "profiled": EXPECTED_PROFILED_PROFILER_POLICY,
        "control": EXPECTED_CONTROL_PROFILER_POLICY,
    }:
        raise _fail("runtime profiler policy differs from the frozen protocol")
    if hashlib.sha256(
        _canonical_json_bytes(runtime_policy)
    ).hexdigest() != artifact.get("runtime_policy_sha256"):
        raise _fail("runtime policy identity does not recompute from raw evidence")
    simsoptpp = _mapping(identity.get("simsoptpp"), "simsoptpp identity")
    if set(simsoptpp) != {"path", "sha256", "build_commit"}:
        raise _fail("simsoptpp identity schema drift")
    declared_path = Path(_string(simsoptpp.get("path"), "simsoptpp.path"))
    declared_sha256 = _sha256(simsoptpp.get("sha256"), "simsoptpp.sha256")
    raw_build_commit = simsoptpp.get("build_commit")
    declared_build_commit = (
        None
        if raw_build_commit is None
        else _string(raw_build_commit, "simsoptpp.build_commit")
    )
    module_spec = importlib.util.find_spec("simsoptpp")
    if module_spec is None or module_spec.origin is None:
        raise _fail("live simsoptpp extension is unavailable")
    live_path = Path(module_spec.origin).resolve()
    if (
        declared_path.resolve() != live_path
        or live_path.is_symlink()
        or not live_path.is_file()
        or hashlib.sha256(live_path.read_bytes()).hexdigest() != declared_sha256
    ):
        raise _fail("simsoptpp identity does not match live extension bytes")
    repo_root = Path(__file__).resolve().parents[1]
    live_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if declared_build_commit is None:
        if artifact.get("authoritative") is not False:
            raise _fail(
                "authoritative source provenance requires a simsoptpp build commit"
            )
    elif declared_build_commit != live_commit:
        raise _fail("simsoptpp build commit does not match live source commit")

    raw_source_preimages = _sequence(
        identity.get("source_preimages"), "source_preimages"
    )
    source_preimages = tuple(
        _mapping(value, f"source_preimages[{index}]")
        for index, value in enumerate(raw_source_preimages)
    )
    if not source_preimages or tuple(
        row.get("original_path") for row in source_preimages
    ) != tuple(sorted(row.get("original_path") for row in source_preimages)):
        raise _fail("source preimages must be nonempty and sorted by original path")
    source_entries = tuple(
        entry
        for entry in evidence.entries
        if entry.process_id == "artifact" and entry.role == "source_evidence"
    )
    source_paths: set[str] = set()
    source_sha256_by_original_path: dict[str, str] = {}
    for row in source_preimages:
        if set(row) != {"original_path", "manifest_path", "sha256", "blob_id"}:
            raise _fail("source preimage mapping schema drift")
        original_path = _string(row.get("original_path"), "original_path")
        if original_path in source_sha256_by_original_path:
            raise _fail(f"duplicate source preimage original path: {original_path}")
        manifest_path = _string(row.get("manifest_path"), "manifest_path")
        safe_manifest_path = _safe_relative(manifest_path)
        sha256 = _sha256(row.get("sha256"), "source preimage sha256")
        blob_id = _sha256(row.get("blob_id"), "source preimage blob_id")
        if blob_id != sha256 or safe_manifest_path.parts != (
            "source_preimages",
            blob_id,
        ):
            raise _fail("source preimage content-addressed path mismatch")
        raw_source = evidence.files.get(manifest_path)
        if raw_source is None or hashlib.sha256(raw_source).hexdigest() != sha256:
            raise _fail(f"source preimage byte mismatch: {original_path}")
        original = Path(original_path)
        if original.is_absolute():
            live_source = original.resolve()
            if live_source != live_path:
                raise _fail("only simsoptpp may use an absolute source preimage path")
        else:
            _safe_relative(original_path)
            live_source = (repo_root / original_path).resolve()
            if not live_source.is_relative_to(repo_root.resolve()):
                raise _fail(f"source preimage escapes repository: {original_path}")
        if (
            live_source.is_symlink()
            or not live_source.is_file()
            or live_source.read_bytes() != raw_source
        ):
            raise _fail(f"source preimage differs from live bytes: {original_path}")
        source_paths.add(manifest_path)
        source_sha256_by_original_path[original_path] = sha256
    if {entry.relative_path for entry in source_entries} != source_paths:
        raise _fail("manifested source evidence differs from source preimage mappings")
    if source_sha256_by_original_path.get(str(live_path)) != declared_sha256:
        raise _fail("simsoptpp binary is absent from source preimages")
    return IdentityBindings(simsoptpp, source_sha256_by_original_path)


def _load_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line:
            raise _fail(f"blank JSONL record: {path}:{line_number}")
        record = json.loads(raw_line)
        if raw_line + b"\n" != _canonical_json_bytes(record):
            raise _fail(f"non-canonical JSONL encoding: {path}:{line_number}")
        records.append(_mapping(record, f"{path}:{line_number}"))
    if not records:
        raise _fail(f"empty JSONL file: {path}")
    return tuple(records)


def _load_jsonl_bytes(raw: bytes, context: str) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line:
            raise _fail(f"blank JSONL record: {context}:{line_number}")
        record = json.loads(raw_line)
        if raw_line + b"\n" != _canonical_json_bytes(record):
            raise _fail(f"non-canonical JSONL encoding: {context}:{line_number}")
        records.append(_mapping(record, f"{context}:{line_number}"))
    if not records:
        raise _fail(f"empty JSONL file: {context}")
    return tuple(records)


def _load_trace_document_bytes(raw: bytes, relative_path: str) -> Mapping[str, object]:
    if relative_path.endswith(".json.gz"):
        decoded = gzip.decompress(raw)
    elif relative_path.endswith(".json"):
        decoded = raw
    else:
        raise _fail("trace path must end in .json or .json.gz")
    return _mapping(json.loads(decoded), relative_path)


def _load_host_evidence(
    raw: bytes, context: str, child_id: str
) -> tuple[tuple[HostEventRecord, ...], tuple[HostSpanEvidence, ...]]:
    records = _load_jsonl_bytes(raw, context)
    events: list[HostEventRecord] = []
    spans: list[HostSpanEvidence] = []
    lifecycle_keys = {
        "schema_id",
        "child_id",
        "record_type",
        "sequence",
        "event",
        "timestamp_ns",
        "evaluation_id",
        "parameter_sha256",
        "evaluation_kind",
        "outer_iteration_id",
        "attributes",
    }
    span_keys = {
        "schema_id",
        "child_id",
        "record_type",
        "sequence",
        "phase_id",
        "start_ns",
        "end_ns",
        "depth",
        "attributes",
    }
    for record in records:
        record_type = record.get("record_type")
        expected_keys = lifecycle_keys if record_type == "lifecycle" else span_keys
        if record_type not in {"lifecycle", "host_span", "optimizer_span"}:
            raise _fail(f"{child_id}: unknown host evidence record type")
        if set(record) != expected_keys:
            raise _fail(
                f"{child_id}: host evidence keys differ from schema: "
                f"{sorted(set(record) ^ expected_keys)}"
            )
        if record.get("schema_id") != EVENT_SCHEMA_ID:
            raise _fail(f"{child_id}: unknown host-event schema")
        if record.get("child_id") != child_id:
            raise _fail(f"{child_id}: host-event child ID mismatch")
        sequence = _integer(record.get("sequence"), "host evidence sequence")
        expected_sequence = len(events) if record_type == "lifecycle" else len(spans)
        if sequence != expected_sequence:
            raise _fail(
                f"{child_id}: {record_type} sequence is not contiguous from zero"
            )
        attributes = _mapping(record.get("attributes"), "host evidence attributes")
        normalized_attributes: list[tuple[str, str | int | float | bool]] = []
        for key, value in attributes.items():
            if not isinstance(key, str) or not isinstance(
                value, (str, int, float, bool)
            ):
                raise _fail(f"{child_id}: invalid host evidence attribute")
            normalized_attributes.append((key, value))
        if record_type in {"host_span", "optimizer_span"}:
            try:
                phase = PhaseId(_string(record.get("phase_id"), "phase_id"))
            except ValueError as error:
                raise _fail(f"{child_id}: unknown host span phase: {error}") from error
            permitted_phases = (
                {PhaseId.OPTIMIZER_LIFECYCLE}
                if record_type == "optimizer_span"
                else {
                    PhaseId.HOST_H2D_SUBMIT,
                    PhaseId.HOST_LINE_SEARCH_CONTROL,
                    PhaseId.HOST_D2H_MATERIALIZE,
                }
            )
            if phase not in permitted_phases:
                raise _fail(
                    f"{child_id}: phase {phase.value} is invalid for {record_type}"
                )
            start_ns = _integer(record.get("start_ns"), "host span start_ns")
            end_ns = _integer(record.get("end_ns"), "host span end_ns")
            depth = _integer(record.get("depth"), "host span depth")
            if start_ns < 0 or end_ns <= start_ns or depth < 0:
                raise _fail(f"{child_id}: reversed or invalid host span")
            spans.append(
                HostSpanEvidence(
                    record_type=record_type,
                    sequence=sequence,
                    phase=phase,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    depth=depth,
                    attributes=tuple(sorted(normalized_attributes)),
                )
            )
            continue
        try:
            event = HostEvent(_string(record.get("event"), "host event"))
            kind = EvaluationKind(
                _string(record.get("evaluation_kind"), "evaluation_kind")
            )
        except ValueError as error:
            raise _fail(f"{child_id}: unknown host lifecycle enum: {error}") from error
        outer_iteration = record.get("outer_iteration_id")
        if outer_iteration is not None:
            outer_iteration = _integer(outer_iteration, "outer_iteration_id")
        timestamp_ns = _integer(record.get("timestamp_ns"), "timestamp_ns")
        if timestamp_ns < 0:
            raise _fail(f"{child_id}: host-event timestamp must be nonnegative")
        events.append(
            HostEventRecord(
                sequence=sequence,
                event=event,
                timestamp_ns=timestamp_ns,
                evaluation=EvaluationTraceContext(
                    evaluation_id=_string(record.get("evaluation_id"), "evaluation_id"),
                    parameter_sha256=_string(
                        record.get("parameter_sha256"), "parameter_sha256"
                    ),
                    kind=kind,
                    outer_iteration_id=outer_iteration,
                ),
                attributes=tuple(sorted(normalized_attributes)),
            )
        )
    return tuple(events), tuple(spans)


def _validate_artifact_metadata(
    document: Mapping[str, object], entries: tuple[ManifestEntry, ...]
) -> tuple[Mapping[str, object], ...]:
    if document.get("schema_id") != ARTIFACT_SCHEMA_ID:
        raise _fail(f"unknown artifact schema: {document.get('schema_id')!r}")
    if document.get("artifact_id") != ARTIFACT_SCHEMA_ID:
        raise _fail("artifact_id does not match the timeline protocol")
    if document.get("trace_schema_id") != EXPECTED_TRACE_SCHEMA_ID:
        raise _fail(f"unknown trace schema: {document.get('trace_schema_id')!r}")
    if document.get("phase_schema_version") != EXPECTED_PHASE_SCHEMA_VERSION:
        raise _fail("phase schema version drift")
    if set(_sequence(document.get("phase_ids"), "phase_ids")) != EXPECTED_PHASE_IDS:
        raise _fail("artifact phase IDs differ from the frozen validator taxonomy")
    if document.get("route") != PRODUCTION_ROUTE:
        raise _fail("artifact declares a non-production optimizer or adjoint route")
    if document.get("scale") != "native_default" or document.get("precision") != "fp64":
        raise _fail("artifact must use native_default FP64")
    if document.get("accepted_iterations") != REQUIRED_ACCEPTED_ITERATIONS:
        raise _fail("artifact accepted-iteration budget is not seven")
    if not isinstance(document.get("authoritative"), bool):
        raise _fail("artifact authority flag must be boolean")
    if (
        document.get("jax_version") != "0.10.0"
        or document.get("jaxlib_version") != "0.10.0"
    ):
        raise _fail(
            "artifact JAX/jaxlib versions are incompatible with the exact "
            f"{EXPECTED_TRACE_SCHEMA_ID} parser"
        )
    source_state = _string(document.get("source_state_sha256"), "source_state")
    repo_root = Path(__file__).resolve().parents[1]
    if _live_worktree_sha256(repo_root) != source_state:
        raise _fail("live executed-source state differs from artifact source state")
    if any(entry.source_state_sha256 != source_state for entry in entries):
        raise _fail("manifest source state does not match artifact metadata")

    raw_schedule = _sequence(document.get("child_schedule"), "child_schedule")
    schedule = tuple(
        _mapping(raw_entry, f"child_schedule[{index}]")
        for index, raw_entry in enumerate(raw_schedule)
    )
    if len(schedule) != PROFILE_CHILDREN + CONTROL_CHILDREN:
        raise _fail("child schedule must contain exactly six children")
    expected_modes = ("profiled", "control") * PROFILE_CHILDREN
    if tuple(entry.get("mode") for entry in schedule) != expected_modes:
        raise _fail("child schedule is not alternating profiled/control")
    if tuple(entry.get("order_index") for entry in schedule) != tuple(range(6)):
        raise _fail("child order indices are not contiguous from zero")
    if len({_string(entry.get("child_id"), "child_id") for entry in schedule}) != 6:
        raise _fail("child IDs are not unique")
    scheduled_child_ids = {
        _string(entry.get("child_id"), "child_id") for entry in schedule
    }
    if {entry.process_id for entry in entries} != {"artifact", *scheduled_child_ids}:
        raise _fail("manifest process IDs differ from the child schedule")
    for pair_index in range(PROFILE_CHILDREN):
        pair = [entry for entry in schedule if entry.get("pair_index") == pair_index]
        if len(pair) != 2 or {entry.get("mode") for entry in pair} != {
            "profiled",
            "control",
        }:
            raise _fail(f"pair {pair_index} is not one profiled/control pair")
    common_roles = {
        "child_metadata",
        "host_device_events",
        "numerical_observations",
        "optimization_timing",
        "trajectory",
        "provenance",
    }
    for scheduled in schedule:
        child_id = _string(scheduled.get("child_id"), "child_id")
        roles = {entry.role for entry in entries if entry.process_id == child_id} - {
            "diagnostic"
        }
        if scheduled.get("mode") == "profiled":
            segment_roles = roles - common_roles
            expected_roles = common_roles | segment_roles
            if segment_roles not in (set(), {"raw_trace", "trace_summary"}):
                raise _fail(f"{child_id}: trace/summary segment roles are unpaired")
        else:
            expected_roles = common_roles
        if roles != expected_roles:
            raise _fail(f"{child_id}: manifest roles differ from child mode schema")
        if scheduled.get("mode") == "control" and any(
            entry.role in {"raw_trace", "trace_summary"}
            for entry in entries
            if entry.process_id == child_id
        ):
            raise _fail(f"{child_id}: control child publishes segment evidence")
    return schedule


def _validate_child_identity(
    artifact: Mapping[str, object],
    scheduled: Mapping[str, object],
    child: Mapping[str, object],
) -> None:
    if child.get("schema_id") != CHILD_SCHEMA_ID:
        raise _fail(f"unknown child schema for {scheduled.get('child_id')}")
    for field in ("child_id", "mode", "pair_index", "order_index"):
        if child.get(field) != scheduled.get(field):
            raise _fail(f"child {field} differs from artifact schedule")
    if child.get("route") != PRODUCTION_ROUTE:
        raise _fail(f"{child.get('child_id')}: runtime route is not production direct")
    identity_fields = (
        "source_state_sha256",
        "environment_sha256",
        "input_sha256",
        "configuration_sha256",
        "construction_sha256",
        "runtime_policy_sha256",
        "initial_parameters_sha256",
        "device_name",
        "device_uuid",
    )
    for field in identity_fields:
        if child.get(field) != artifact.get(field):
            raise _fail(f"{child.get('child_id')}: {field} differs from artifact")
    expected_profiler_policy = _expected_profiler_policy(scheduled.get("mode"))
    if child.get("profiler_policy") != expected_profiler_policy:
        raise _fail(f"{child.get('child_id')}: profiler policy differs from mode")
    _string(child.get("cache_sha256"), "cache_sha256")
    state = child.get("state")
    if state not in {"complete", "failed", "incomplete"}:
        raise _fail(f"{child.get('child_id')}: unknown child state")
    child_end_to_end_ns = _integer(
        child.get("child_end_to_end_ns"), "child_end_to_end_ns"
    )
    if child_end_to_end_ns <= 0:
        raise _fail(f"{child.get('child_id')}: child end-to-end time is invalid")
    raw_wall = child.get("optimizer_raw_wall_ns")
    active_wall = child.get("optimizer_active_wall_ns")
    boundary_pause = child.get("profiler_boundary_pause_total_ns")
    raw_boundary_records = _sequence(
        child.get("boundary_pause_records"), "boundary_pause_records"
    )
    if state == "complete":
        raw_wall_ns = _integer(raw_wall, "optimizer_raw_wall_ns")
        active_wall_ns = _integer(active_wall, "optimizer_active_wall_ns")
        boundary_pause_ns = _integer(boundary_pause, "profiler_boundary_pause_total_ns")
        if raw_wall_ns <= 0 or active_wall_ns <= 0 or boundary_pause_ns < 0:
            raise _fail(f"{child.get('child_id')}: optimizer timing is invalid")
        if raw_wall_ns != active_wall_ns + boundary_pause_ns:
            raise _fail(
                f"{child.get('child_id')}: raw wall is not active wall plus pauses"
            )
        if child_end_to_end_ns < raw_wall_ns:
            raise _fail(f"{child.get('child_id')}: end-to-end time precedes raw wall")
        expected_records = (
            2 * REQUIRED_ACCEPTED_ITERATIONS
            if scheduled.get("mode") == "profiled"
            else 0
        )
        if len(raw_boundary_records) != expected_records:
            raise _fail(
                f"{child.get('child_id')}: boundary pause record count is invalid"
            )
        if scheduled.get("mode") == "control" and boundary_pause_ns != 0:
            raise _fail(f"{child.get('child_id')}: control child declares pauses")
    else:
        if raw_wall is not None or active_wall is not None:
            raise _fail(
                f"{child.get('child_id')}: incomplete optimizer timing must be null"
            )
        if _integer(boundary_pause, "profiler_boundary_pause_total_ns") < 0:
            raise _fail(f"{child.get('child_id')}: failed child pause is invalid")
    failure_class = child.get("failure_class")
    failure_reason = child.get("failure_reason")
    first_failed_evaluation_id = child.get("first_failed_evaluation_id")
    if state == "complete":
        if any(
            value is not None
            for value in (
                failure_class,
                failure_reason,
                first_failed_evaluation_id,
            )
        ):
            raise _fail(f"{child.get('child_id')}: complete child declares failure")
        return
    if failure_class not in {"scientific", "trace", "integrity"}:
        raise _fail(f"{child.get('child_id')}: invalid failure_class")
    _string(failure_reason, "failure_reason")
    if failure_class == "scientific":
        _string(first_failed_evaluation_id, "first_failed_evaluation_id")
    elif first_failed_evaluation_id is not None:
        raise _fail(
            f"{child.get('child_id')}: non-scientific failure binds evaluation ID"
        )


def _validate_observations(
    child_id: str,
    document: Mapping[str, object],
    artifact: Mapping[str, object],
    child: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    if document.get("schema_id") != OBSERVATION_SCHEMA_ID:
        raise _fail(f"{child_id}: unknown observation schema")
    if document.get("child_id") != child_id:
        raise _fail(f"{child_id}: observation child ID mismatch")
    evaluations = tuple(
        _mapping(value, f"{child_id}.evaluations[{index}]")
        for index, value in enumerate(
            _sequence(document.get("evaluations"), f"{child_id}.evaluations")
        )
    )
    evaluation_ids = [
        _string(row.get("evaluation_id"), "evaluation_id") for row in evaluations
    ]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise _fail(f"{child_id}: duplicate evaluation ID")
    if tuple(row.get("evaluation_index") for row in evaluations) != tuple(
        range(len(evaluations))
    ):
        raise _fail(f"{child_id}: evaluation indices are not contiguous from zero")
    initial = [row for row in evaluations if row.get("lifecycle") == "initial"]
    final = [row for row in evaluations if row.get("lifecycle") == "final_reporting"]
    trials = [row for row in evaluations if row.get("lifecycle") == "trial"]
    if len(initial) != 1 or len(final) != 1:
        raise _fail(
            f"{child_id}: exactly one initial and final_reporting evaluation required"
        )
    if (
        evaluations[0].get("lifecycle") != "initial"
        or evaluations[-1].get("lifecycle") != "final_reporting"
        or any(row.get("lifecycle") != "trial" for row in evaluations[1:-1])
    ):
        raise _fail(
            f"{child_id}: lifecycle must be initial, only trials, final_reporting"
        )
    if any(row.get("accepted") is not None for row in (*initial, *final)):
        raise _fail(f"{child_id}: non-trial evaluations must not have a disposition")
    if any(not isinstance(row.get("accepted"), bool) for row in trials):
        raise _fail(
            f"{child_id}: every trial requires one accepted/rejected disposition"
        )
    trial_iterations = [
        _integer(row.get("iteration"), "trial iteration") for row in trials
    ]
    if any(
        iteration < 1 or iteration > REQUIRED_ACCEPTED_ITERATIONS
        for iteration in trial_iterations
    ) or any(
        current < previous
        for previous, current in zip(trial_iterations, trial_iterations[1:])
    ):
        raise _fail(f"{child_id}: trial iteration IDs are invalid or decrease")
    accepted = [row for row in trials if row.get("accepted") is True]
    if len(accepted) != REQUIRED_ACCEPTED_ITERATIONS:
        raise _fail(f"{child_id}: accepted trial count is not seven")
    if tuple(row.get("iteration") for row in accepted) != tuple(
        range(1, REQUIRED_ACCEPTED_ITERATIONS + 1)
    ):
        raise _fail(f"{child_id}: accepted iterations are not contiguous from one")
    expected_line_search_decisions = [
        {
            "evaluation_id": row.get("evaluation_id"),
            "accepted": row.get("accepted"),
            "iteration": row.get("iteration"),
        }
        for row in trials
    ]
    if child.get("line_search_decisions") != expected_line_search_decisions:
        raise _fail(f"{child_id}: line-search decisions do not recompute")
    if child.get("nit") != len(accepted):
        raise _fail(f"{child_id}: optimizer iteration count does not recompute")
    expected_objective_gradient_evaluations = 1 + len(trials) + 1
    if (
        child.get("nfev") != expected_objective_gradient_evaluations
        or child.get("njev") != expected_objective_gradient_evaluations
    ):
        raise _fail(f"{child_id}: objective/gradient counts do not recompute")
    trial_hashes: list[str] = []
    for row in evaluations:
        parameters, recomputed_parameter_sha256 = _parameter_vector_and_sha256(
            row.get("parameters"), "parameters"
        )
        if row.get("parameter_shape") != [len(parameters)]:
            raise _fail(f"{child_id}: parameter shape does not match raw parameters")
        parameter_sha256 = _string(row.get("parameter_sha256"), "parameter_sha256")
        if parameter_sha256 != recomputed_parameter_sha256:
            raise _fail(f"{child_id}: parameter SHA-256 does not match raw values")
        _finite_float(row.get("objective"), "objective")
        if row.get("dtype") != "float64":
            raise _fail(f"{child_id}: evaluation is not float64")
        gradient = _sequence(row.get("gradient"), "gradient")
        if not gradient:
            raise _fail(f"{child_id}: gradient is empty")
        for component in gradient:
            _finite_float(component, "gradient component")
        if row.get("gradient_shape") != [len(gradient)]:
            raise _fail(f"{child_id}: gradient shape does not match raw gradient")
        if len(parameters) != len(gradient):
            raise _fail(f"{child_id}: parameter and gradient lengths differ")
        if row.get("gradient_source") != "candidate":
            raise _fail(f"{child_id}: gradient is not candidate-sourced")
        if (
            row.get("inner_success") is not True
            or row.get("adjoint_success") is not True
            or row.get("values_finite") is not True
            or row.get("candidate_gradient_source") is not True
            or row.get("eligible") is not True
            or row.get("trajectory_valid") is not True
        ):
            raise _fail(f"{child_id}: evaluation scientific evidence is invalid")
        inner_evidence = _mapping(row.get("inner_evidence"), "inner_evidence")
        if set(inner_evidence) != {
            "residual_trace",
            "step_accepted_trace",
            "linear_solve_success_trace",
            "newton_iterations",
            "newton_attempted_iterations",
            "newton_trace_available",
        }:
            raise _fail(f"{child_id}: inner evidence schema drift")
        newton_iterations = _integer(
            inner_evidence.get("newton_iterations"), "newton_iterations"
        )
        if newton_iterations < 0:
            raise _fail(f"{child_id}: negative Newton iteration count")
        newton_trace_available = inner_evidence.get("newton_trace_available")
        if not isinstance(newton_trace_available, bool):
            raise _fail(f"{child_id}: Newton trace availability must be boolean")
        residual_trace = _sequence(
            inner_evidence.get("residual_trace"), "inner residual_trace"
        )
        for residual in residual_trace:
            _finite_float(residual, "inner residual")
        evidence_traces: dict[str, Sequence[object]] = {}
        for evidence_name in (
            "step_accepted_trace",
            "linear_solve_success_trace",
        ):
            evidence_trace = _sequence(inner_evidence.get(evidence_name), evidence_name)
            if any(not isinstance(value, bool) for value in evidence_trace):
                raise _fail(f"{child_id}: {evidence_name} schema/count mismatch")
            evidence_traces[evidence_name] = evidence_trace
        raw_attempted_iterations = inner_evidence.get("newton_attempted_iterations")
        if newton_trace_available:
            newton_attempted_iterations = _integer(
                raw_attempted_iterations,
                "newton_attempted_iterations",
            )
            if (
                newton_attempted_iterations < 0
                or newton_iterations > newton_attempted_iterations
                or len(residual_trace) != newton_attempted_iterations
                or any(
                    len(trace) != newton_attempted_iterations
                    for trace in evidence_traces.values()
                )
            ):
                raise _fail(f"{child_id}: invalid Newton attempted/trace counts")
            if sum(evidence_traces["step_accepted_trace"]) != newton_iterations:
                raise _fail(
                    f"{child_id}: accepted Newton step count does not recompute"
                )
        elif (
            raw_attempted_iterations is not None
            or residual_trace
            or any(evidence_traces.values())
        ):
            raise _fail(f"{child_id}: unavailable Newton trace must be exactly empty")
        adjoint_evidence = _mapping(row.get("adjoint_evidence"), "adjoint_evidence")
        if set(adjoint_evidence) != {
            "route",
            "output",
            "residual",
            "residual_relative",
            "dense_materializations",
            "lu_factorizations",
            "lu_solves",
            "refinement_corrections",
            "adjoint_executions",
        }:
            raise _fail(f"{child_id}: adjoint evidence schema drift")
        if adjoint_evidence.get("route") != "exact_jacobian_dense_fp64_lu":
            raise _fail(f"{child_id}: adjoint evidence route is not production direct")
        adjoint_output = tuple(
            _finite_float(value, "adjoint output")
            for value in _sequence(adjoint_evidence.get("output"), "adjoint output")
        )
        residual = _finite_float(adjoint_evidence.get("residual"), "adjoint residual")
        residual_relative = _finite_float(
            adjoint_evidence.get("residual_relative"), "relative adjoint residual"
        )
        if residual < 0.0 or residual_relative < 0.0:
            raise _fail(f"{child_id}: adjoint residual evidence is negative")
        adjoint_counts = {
            name: _integer(adjoint_evidence.get(name), name)
            for name in (
                "dense_materializations",
                "lu_factorizations",
                "lu_solves",
                "refinement_corrections",
                "adjoint_executions",
            )
        }
        if any(value < -1 for value in adjoint_counts.values()):
            raise _fail(f"{child_id}: invalid negative adjoint semantic count")
        counts_available = all(value >= 0 for value in adjoint_counts.values())
        if counts_available and adjoint_counts["adjoint_executions"] not in {0, 1}:
            raise _fail(f"{child_id}: invalid adjoint execution count")
        if counts_available and adjoint_counts["adjoint_executions"] == 0:
            if (
                any(adjoint_output)
                or residual != 0.0
                or residual_relative != 0.0
                or any(
                    adjoint_counts[name]
                    for name in (
                        "dense_materializations",
                        "lu_factorizations",
                        "lu_solves",
                        "refinement_corrections",
                    )
                )
            ):
                raise _fail(
                    f"{child_id}: zero-execution adjoint evidence is inconsistent"
                )
        elif counts_available and (
            adjoint_counts["dense_materializations"] < 1
            or adjoint_counts["lu_factorizations"] < 1
            or adjoint_counts["lu_solves"] < 1
        ):
            raise _fail(f"{child_id}: executed direct adjoint omits required counts")
        if row.get("lifecycle") == "trial":
            trial_hashes.append(
                _string(row.get("parameter_sha256"), "parameter_sha256")
            )
    if len(trial_hashes) != len(set(trial_hashes)):
        raise _fail(f"{child_id}: optimizer trial parameter hashes are not distinct")
    if final[0].get("parameter_sha256") != accepted[-1].get("parameter_sha256"):
        raise _fail(f"{child_id}: final_reporting is not bound to the final incumbent")
    if initial[0].get("parameter_sha256") != artifact.get("initial_parameters_sha256"):
        raise _fail(f"{child_id}: initial parameter values do not bind artifact")
    final_parameters, final_parameters_sha256 = _parameter_vector_and_sha256(
        child.get("final_parameters"), "child final_parameters"
    )
    if child.get("final_parameters_sha256") != final_parameters_sha256:
        raise _fail(f"{child_id}: child final parameter SHA-256 is invalid")
    if final_parameters_sha256 != final[0].get("parameter_sha256") or list(
        final_parameters
    ) != final[0].get("parameters"):
        raise _fail(f"{child_id}: child final parameters do not bind final reporting")
    final_observables = _mapping(final[0].get("observables"), "final observables")
    required_observables = {
        "objective",
        "iota",
        "volume",
        "non_qs_ratio",
        "boozer_residual",
    }
    if set(final_observables) != required_observables:
        raise _fail(f"{child_id}: final observable schema drift")
    for name, value in final_observables.items():
        _finite_float(value, f"final observable {name}")
    _close(final_observables["objective"], final[0].get("objective"), "objective")
    return evaluations


def _validate_boundary_pause_records(
    child: Mapping[str, object],
    evaluations: tuple[Mapping[str, object], ...],
    *,
    complete: bool,
) -> None:
    """Recompute profiler pause accounting and accepted-iteration bindings."""

    child_id = _string(child.get("child_id"), "child_id")
    records = tuple(
        _mapping(value, f"{child_id}.boundary_pause_records[{index}]")
        for index, value in enumerate(
            _sequence(child.get("boundary_pause_records"), "boundary_pause_records")
        )
    )
    expected_keys = {"iteration_id", "operation", "start_ns", "end_ns", "duration_ns"}
    accepted_by_iteration = {
        _integer(row.get("iteration"), "accepted iteration"): _string(
            row.get("evaluation_id"), "accepted evaluation_id"
        )
        for row in evaluations
        if row.get("lifecycle") == "trial" and row.get("accepted") is True
    }
    total_pause_ns = 0
    previous_end_ns: int | None = None
    for record_index, record in enumerate(records):
        if set(record) != expected_keys:
            raise _fail(f"{child_id}: boundary pause record schema drift")
        expected_iteration = record_index // 2 + 1
        expected_operation = "start" if record_index % 2 == 0 else "stop"
        if (
            record.get("iteration_id") != expected_iteration
            or record.get("operation") != expected_operation
            or (complete and expected_iteration not in accepted_by_iteration)
        ):
            raise _fail(f"{child_id}: boundary pause lifecycle binding differs")
        start_ns = _integer(record.get("start_ns"), "start_ns")
        end_ns = _integer(record.get("end_ns"), "end_ns")
        duration_ns = _integer(record.get("duration_ns"), "duration_ns")
        if start_ns < 0 or end_ns < start_ns or duration_ns != end_ns - start_ns:
            raise _fail(f"{child_id}: boundary pause timestamps are not monotonic")
        if previous_end_ns is not None and start_ns < previous_end_ns:
            raise _fail(f"{child_id}: profiler segment boundaries overlap")
        previous_end_ns = end_ns
        total_pause_ns += duration_ns
    expected_total = 0 if child.get("mode") == "control" else total_pause_ns
    if child.get("profiler_boundary_pause_total_ns") != expected_total:
        raise _fail(f"{child_id}: boundary pause total does not recompute")


def _validate_segment_manifest_bindings(
    entries: tuple[ManifestEntry, ...],
    child: Mapping[str, object],
    evaluations: tuple[Mapping[str, object], ...],
) -> tuple[tuple[ManifestEntry, ManifestEntry], ...]:
    """Bind each segment pair to exactly one accepted lifecycle evaluation."""

    child_id = _string(child.get("child_id"), "child_id")
    if child.get("mode") == "control":
        if any(
            entry.role in {"raw_trace", "trace_summary"}
            for entry in entries
            if entry.process_id == child_id
        ):
            raise _fail(f"{child_id}: control child publishes segment evidence")
        return ()
    complete = child.get("state") == "complete"
    raw_entries = _segment_entries_by_role(
        entries, child_id, "raw_trace", complete=complete
    )
    summary_entries = _segment_entries_by_role(
        entries, child_id, "trace_summary", complete=complete
    )
    if len(raw_entries) != len(summary_entries):
        raise _fail(f"{child_id}: raw/summary segment counts differ")
    accepted_by_iteration = {
        _integer(row.get("iteration"), "accepted iteration"): _string(
            row.get("evaluation_id"), "accepted evaluation_id"
        )
        for row in evaluations
        if row.get("lifecycle") == "trial" and row.get("accepted") is True
    }
    target_ids_by_iteration = {
        iteration: tuple(
            _string(row.get("evaluation_id"), "trial evaluation_id")
            for row in evaluations
            if row.get("lifecycle") == "trial" and row.get("iteration") == iteration
        )
        for iteration in range(1, REQUIRED_ACCEPTED_ITERATIONS + 1)
    }
    pairs: list[tuple[ManifestEntry, ManifestEntry]] = []
    for iteration, (raw_entry, summary_entry) in enumerate(
        zip(raw_entries, summary_entries, strict=True), start=1
    ):
        evaluation_id = accepted_by_iteration.get(iteration)
        if evaluation_id is None:
            raise _fail(f"{child_id}: segment has no accepted lifecycle evaluation")
        target_digest = _evaluation_ids_sha256(target_ids_by_iteration[iteration])
        if (
            raw_entry.sample_id != summary_entry.sample_id
            or raw_entry.evaluation_id != evaluation_id
            or summary_entry.evaluation_id != evaluation_id
            or raw_entry.evaluation_ids_sha256 != summary_entry.evaluation_ids_sha256
            or raw_entry.segment_evaluation_ids_sha256 != target_digest
            or summary_entry.segment_evaluation_ids_sha256 != target_digest
        ):
            raise _fail(f"{child_id}: segment manifest binding differs")
        pairs.append((raw_entry, summary_entry))
    return tuple(pairs)


def _validate_failed_child_raw_evidence(
    evidence: ArtifactEvidence,
    entries: tuple[ManifestEntry, ...],
    scheduled: Mapping[str, object],
    child: Mapping[str, object],
    observations: Mapping[str, object],
) -> None:
    child_id = _string(child.get("child_id"), "child_id")
    failure_reason = _string(child.get("failure_reason"), "failure_reason")
    if (
        set(observations)
        != {
            "schema_id",
            "child_id",
            "evaluations",
            "failure_reason",
            "first_failed_evaluation_id",
        }
        or observations.get("failure_reason") != failure_reason
    ):
        raise _fail(f"{child_id}: failed observation placeholder schema drift")
    if observations.get("first_failed_evaluation_id") != child.get(
        "first_failed_evaluation_id"
    ):
        raise _fail(f"{child_id}: failed observation identity mismatch")
    events_entry = _entry_by_role(entries, child_id, "host_device_events")
    raw_events = evidence.files[events_entry.relative_path]
    if scheduled.get("mode") == "profiled":
        diagnostics = _load_jsonl_bytes(raw_events, events_entry.relative_path)
        if not diagnostics or diagnostics[-1] != {
            "schema_id": EVENT_SCHEMA_ID,
            "child_id": child_id,
            "sequence": len(diagnostics) - 1,
            "event": "diagnostic_failure",
            "failure_reason": failure_reason,
        }:
            raise _fail(f"{child_id}: failed diagnostic event schema drift")
        if len(diagnostics) > 1:
            raw_audit = b"".join(
                _canonical_json_bytes(document) for document in diagnostics[:-1]
            )
            _load_host_evidence(raw_audit, events_entry.relative_path, child_id)
    elif raw_events:
        raise _fail(f"{child_id}: failed control event evidence must be empty")


def _first_scientific_failure(
    child_id: str, evaluations: tuple[Mapping[str, object], ...]
) -> tuple[str, tuple[str, ...]] | None:
    seen_ids: set[str] = set()
    for index, row in enumerate(evaluations):
        evaluation_id = _string(row.get("evaluation_id"), "evaluation_id")
        if evaluation_id in seen_ids:
            raise _fail(f"{child_id}: duplicate failed-run evaluation ID")
        seen_ids.add(evaluation_id)
        if row.get("evaluation_index") != index:
            raise _fail(f"{child_id}: failed-run evaluation indices are not contiguous")
        failures: list[str] = []
        if row.get("values_finite") is False:
            failures.append("nonfinite_values")
        if row.get("inner_success") is False:
            failures.append("inner_solve_failed")
        if row.get("adjoint_success") is False:
            failures.append("adjoint_failed")
        if row.get("trajectory_valid") is False:
            failures.append("trajectory_invalid")
        if row.get("candidate_gradient_source") is False:
            failures.append("noncandidate_gradient")
        if failures:
            return evaluation_id, tuple(failures)
    return None


def _load_raw_child_records(
    evidence: ArtifactEvidence,
    entries: tuple[ManifestEntry, ...],
    artifact: Mapping[str, object],
    child_id: str,
    child: Mapping[str, object],
    identity_bindings: IdentityBindings,
) -> tuple[int, tuple[Mapping[str, object], ...]]:
    timing_entry = _entry_by_role(entries, child_id, "optimization_timing")
    trajectory_entry = _entry_by_role(entries, child_id, "trajectory")
    provenance_entry = _entry_by_role(entries, child_id, "provenance")
    timing = _mapping(
        _load_canonical_json_bytes(
            evidence.files[timing_entry.relative_path], timing_entry.relative_path
        ),
        f"{child_id}.timing",
    )
    timing_available = (
        set(timing) == {"schema_version", "wall_seconds"}
        and timing.get("schema_version") == 1
    )
    if timing_available:
        wall_seconds = _finite_float(timing.get("wall_seconds"), "wall_seconds")
        if wall_seconds <= 0.0:
            raise _fail(f"{child_id}: optimization timing wall must be positive")
        wall_ns = round(wall_seconds * 1_000_000_000)
        if child.get("optimizer_raw_wall_ns") != wall_ns:
            raise _fail(f"{child_id}: child wall does not match raw timing sidecar")
    elif child.get("state") != "complete":
        unavailable_timing = {
            "schema_id": "single-stage-changed-state-gpu-timeline-timing-unavailable-v1",
            "failure_reason": timing.get("failure_reason"),
        }
        if timing != unavailable_timing:
            raise _fail(f"{child_id}: incomplete timing placeholder schema drift")
        _string(timing.get("failure_reason"), "timing failure_reason")
        if any(
            child.get(field) is not None
            for field in ("optimizer_raw_wall_ns", "optimizer_active_wall_ns")
        ):
            raise _fail(f"{child_id}: unavailable timing requires null optimizer time")
        wall_ns = 0
    else:
        raise _fail(f"{child_id}: optimization timing is not exact schema v1")

    provenance = _mapping(
        _load_canonical_json_bytes(
            evidence.files[provenance_entry.relative_path],
            provenance_entry.relative_path,
        ),
        f"{child_id}.provenance",
    )
    if (
        provenance.get("schema_id")
        != ("single-stage-changed-state-gpu-timeline-provenance-v1")
        or provenance.get("child_id") != child_id
    ):
        raise _fail(f"{child_id}: invalid raw provenance schema or child identity")
    if provenance.get("source_state_sha256") != artifact.get("source_state_sha256"):
        raise _fail(f"{child_id}: raw provenance source state mismatch")
    environment = _mapping(provenance.get("environment"), "provenance.environment")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise _fail(f"{child_id}: provenance environment must map strings to strings")
    environment_sha256 = hashlib.sha256(
        _canonical_json_bytes(dict(sorted(environment.items())))
    ).hexdigest()
    if environment_sha256 != artifact.get("environment_sha256"):
        raise _fail(f"{child_id}: environment hash does not recompute from provenance")
    if environment.get(EXPECTED_TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT) != str(
        EXPECTED_TRACE_VIEWER_MAX_EVENTS
    ):
        raise _fail(f"{child_id}: trace-viewer event capacity is not runtime-bound")
    if provenance.get("profiler_policy") != _expected_profiler_policy(
        child.get("mode")
    ):
        raise _fail(f"{child_id}: raw provenance profiler policy differs from mode")
    device = _mapping(provenance.get("device"), "provenance.device")
    if device != {
        "name": artifact.get("device_name"),
        "uuid": artifact.get("device_uuid"),
    }:
        raise _fail(f"{child_id}: raw device provenance differs from artifact")
    runtime = _mapping(provenance.get("runtime"), "provenance.runtime")
    expected_runtime = {
        "python_version": artifact.get("python_version"),
        "jax_version": artifact.get("jax_version"),
        "jaxlib_version": artifact.get("jaxlib_version"),
    }
    if runtime != expected_runtime:
        raise _fail(f"{child_id}: raw runtime provenance differs from artifact")
    if {
        "path": provenance.get("simsoptpp_path"),
        "sha256": provenance.get("simsoptpp_sha256"),
        "build_commit": provenance.get("simsoptpp_build_commit"),
    } != identity_bindings.simsoptpp:
        raise _fail(f"{child_id}: simsoptpp provenance differs from live identity")
    raw_executed_sources = _sequence(
        provenance.get("executed_sources"), "provenance.executed_sources"
    )
    if not raw_executed_sources:
        raise _fail(f"{child_id}: executed source provenance is empty")
    executed_paths: set[str] = set()
    for index, raw_source in enumerate(raw_executed_sources):
        source = _mapping(raw_source, f"executed_sources[{index}]")
        if set(source) != {"path", "sha256", "git_blob_id"}:
            raise _fail(f"{child_id}: executed source schema drift")
        path = _string(source.get("path"), "executed source path")
        sha256 = _sha256(source.get("sha256"), "executed source sha256")
        if path in executed_paths:
            raise _fail(f"{child_id}: duplicate executed source path")
        executed_paths.add(path)
        if identity_bindings.source_sha256_by_original_path.get(path) != sha256:
            raise _fail(f"{child_id}: executed source lacks exact-byte preimage")
        git_blob_id = source.get("git_blob_id")
        if git_blob_id is not None and (
            not isinstance(git_blob_id, str)
            or len(git_blob_id) not in {40, 64}
            or any(character not in _LOWER_HEX for character in git_blob_id)
        ):
            raise _fail(f"{child_id}: invalid executed source git blob ID")
        live_blob = subprocess.run(
            ("git", "ls-files", "-s", "--", path),
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        recomputed_git_blob_id = live_blob.split()[1] if live_blob else None
        if git_blob_id != recomputed_git_blob_id:
            raise _fail(f"{child_id}: executed source git blob ID mismatch")
    collection_scope = provenance.get("collection_scope")
    permitted_scopes = (
        {"child_postexecution"}
        if child.get("state") == "complete"
        else {
            "child_preexecution",
            "child_postexecution",
            "parent_prelaunch_after_child_termination",
        }
    )
    if collection_scope not in permitted_scopes:
        raise _fail(f"{child_id}: provenance collection scope is invalid for state")
    child_provenance = _mapping(child.get("provenance"), f"{child_id}.child.provenance")
    artifact_authority = artifact.get("authoritative")
    if (
        provenance.get("authoritative") is not artifact_authority
        or child_provenance.get("authoritative") is not artifact_authority
    ):
        raise _fail(f"{child_id}: artifact authority contradicts lane provenance")

    trajectory = (
        ()
        if not evidence.files[trajectory_entry.relative_path]
        else _load_jsonl_bytes(
            evidence.files[trajectory_entry.relative_path],
            trajectory_entry.relative_path,
        )
    )
    if child.get("state") != "complete" and not timing_available and trajectory:
        raise _fail(f"{child_id}: incomplete child trajectory must be empty")
    return wall_ns, trajectory


def _validate_trajectory(
    child_id: str,
    trajectory: tuple[Mapping[str, object], ...],
    evaluations: tuple[Mapping[str, object], ...],
    wall_ns: int,
) -> None:
    accepted = tuple(row for row in evaluations if row.get("accepted") is True)
    if len(trajectory) != REQUIRED_ACCEPTED_ITERATIONS:
        raise _fail(f"{child_id}: raw trajectory does not contain seven records")
    previous_wall = -1.0
    for iteration, (point, observation) in enumerate(
        zip(trajectory, accepted, strict=True), start=1
    ):
        if set(point) != {"iteration", "objective", "wall_seconds_from_start"}:
            raise _fail(f"{child_id}: trajectory record schema drift")
        if point.get("iteration") != iteration:
            raise _fail(f"{child_id}: trajectory iterations are not contiguous")
        _close(
            point.get("objective"), observation.get("objective"), "trajectory objective"
        )
        point_wall = _finite_float(
            point.get("wall_seconds_from_start"), "trajectory wall"
        )
        if point_wall < previous_wall or round(point_wall * 1_000_000_000) > wall_ns:
            raise _fail(f"{child_id}: trajectory wall is decreasing or exceeds timing")
        previous_wall = point_wall


def _close(left: object, right: object, context: str) -> None:
    left_number = _finite_float(left, context)
    right_number = _finite_float(right, context)
    tolerance = NUMERICAL_ATOL + NUMERICAL_RTOL * abs(right_number)
    if abs(left_number - right_number) > tolerance:
        raise _fail(f"profile/control numerical mismatch for {context}")


def _compare_pair(
    profile_child: Mapping[str, object],
    control_child: Mapping[str, object],
    profile_evaluations: tuple[Mapping[str, object], ...],
    control_evaluations: tuple[Mapping[str, object], ...],
) -> None:
    for field in ("nit", "nfev", "njev", "status", "line_search_decisions"):
        if profile_child.get(field) != control_child.get(field):
            raise _fail(f"profile/control mismatch for {field}")
    if len(profile_evaluations) != len(control_evaluations):
        raise _fail("profile/control evaluation counts differ")
    exact_fields = (
        "evaluation_index",
        "lifecycle",
        "accepted",
        "iteration",
        "parameter_sha256",
        "dtype",
        "gradient_shape",
        "gradient_source",
        "values_finite",
        "inner_success",
        "adjoint_success",
        "candidate_gradient_source",
        "eligible",
        "trajectory_valid",
        "inner_evidence",
        "adjoint_evidence",
    )
    for index, (profile_row, control_row) in enumerate(
        zip(profile_evaluations, control_evaluations, strict=True)
    ):
        for field in exact_fields:
            if field not in profile_row or field not in control_row:
                raise _fail(f"profile/control evaluation omits required {field}")
            if profile_row.get(field) != control_row.get(field):
                raise _fail(
                    f"profile/control mismatch for evaluation {index} field {field}"
                )
        if profile_row.get("parameter_sha256") != control_row.get("parameter_sha256"):
            raise _fail(
                f"profile/control parameter hash mismatch at evaluation {index}"
            )
        profile_parameters = _sequence(profile_row.get("parameters"), "parameters")
        control_parameters = _sequence(control_row.get("parameters"), "parameters")
        if len(profile_parameters) != len(control_parameters):
            raise _fail(
                f"profile/control parameter lengths differ at evaluation {index}"
            )
        for component_index, (profile_value, control_value) in enumerate(
            zip(profile_parameters, control_parameters, strict=True)
        ):
            _close(
                profile_value,
                control_value,
                f"parameters[{index}][{component_index}]",
            )
        _close(profile_row.get("objective"), control_row.get("objective"), "objective")
        profile_gradient = _sequence(profile_row.get("gradient"), "gradient")
        control_gradient = _sequence(control_row.get("gradient"), "gradient")
        if len(profile_gradient) != len(control_gradient):
            raise _fail(
                f"profile/control gradient lengths differ at evaluation {index}"
            )
        for component_index, (profile_value, control_value) in enumerate(
            zip(profile_gradient, control_gradient, strict=True)
        ):
            profile_number = _finite_float(profile_value, "gradient component")
            control_number = _finite_float(control_value, "gradient component")
            tolerance = GRADIENT_ATOL + GRADIENT_RTOL * abs(control_number)
            if abs(profile_number - control_number) > tolerance:
                raise _fail(
                    "profile/control gradient mismatch at evaluation "
                    f"{index} component {component_index}"
                )
        raw_profile_observables = profile_row.get("observables")
        raw_control_observables = control_row.get("observables")
        if raw_profile_observables is None or raw_control_observables is None:
            if (
                raw_profile_observables is not None
                or raw_control_observables is not None
            ):
                raise _fail("profile/control observable availability differs")
            continue
        profile_observables = _mapping(raw_profile_observables, "observables")
        control_observables = _mapping(raw_control_observables, "observables")
        if set(profile_observables) != set(control_observables):
            raise _fail("profile/control observable names differ")
        for name in profile_observables:
            profile_value = profile_observables[name]
            control_value = control_observables[name]
            if profile_value is None or control_value is None:
                if profile_value is not None or control_value is not None:
                    raise _fail(f"profile/control availability mismatch for {name}")
            else:
                _close(profile_value, control_value, name)
    if profile_child.get("endpoint_certificate") != control_child.get(
        "endpoint_certificate"
    ):
        raise _fail("profile/control endpoint certificates differ")


def _correlate_host_events_with_observations(
    child_id: str,
    host_events: tuple[HostEventRecord, ...],
    evaluations: tuple[Mapping[str, object], ...],
) -> bool:
    observation_by_id = {
        _string(row.get("evaluation_id"), "evaluation_id"): row for row in evaluations
    }
    event_evaluation_ids = {record.evaluation.evaluation_id for record in host_events}
    if event_evaluation_ids != set(observation_by_id):
        raise _fail(f"{child_id}: host-event/evaluation ID sets differ")
    event_groups: dict[str, list[HostEventRecord]] = {}
    for record in host_events:
        observed = observation_by_id[record.evaluation.evaluation_id]
        event_groups.setdefault(record.evaluation.evaluation_id, []).append(record)
        if record.evaluation.parameter_sha256 != observed.get("parameter_sha256"):
            raise _fail(f"{child_id}: host-event parameter correlation mismatch")
        if record.evaluation.kind.value != observed.get("lifecycle"):
            raise _fail(f"{child_id}: host-event lifecycle correlation mismatch")
        expected_iteration = (
            observed.get("iteration") if observed.get("lifecycle") == "trial" else None
        )
        if record.evaluation.outer_iteration_id != expected_iteration:
            raise _fail(f"{child_id}: host-event iteration correlation mismatch")
    return all(
        tuple(record.event for record in records) == tuple(HostEvent)
        and all(
            left.timestamp_ns < right.timestamp_ns
            for left, right in zip(records, records[1:])
        )
        for records in event_groups.values()
    ) and all(
        left.timestamp_ns < right.timestamp_ns
        for left, right in zip(host_events, host_events[1:])
    )


def _validate_host_spans(
    child_id: str,
    spans: tuple[HostSpanEvidence, ...],
    host_events: tuple[HostEventRecord, ...],
    evaluations: tuple[Mapping[str, object], ...],
) -> None:
    event_groups: dict[str, dict[HostEvent, HostEventRecord]] = {}
    for event in host_events:
        event_groups.setdefault(event.evaluation.evaluation_id, {})[event.event] = event
    observations = {
        _string(row.get("evaluation_id"), "evaluation_id"): row for row in evaluations
    }
    exclusive_spans = tuple(span for span in spans if span.record_type == "host_span")
    optimizer_spans = tuple(
        span for span in spans if span.record_type == "optimizer_span"
    )
    ordered = sorted(exclusive_spans, key=lambda span: (span.start_ns, span.end_ns))
    for left, right in zip(ordered, ordered[1:]):
        if right.start_ns < left.end_ns:
            raise _fail(f"{child_id}: exclusive host spans overlap")

    transfer_spans: dict[tuple[PhaseId, str], list[HostSpanEvidence]] = {}
    line_spans: dict[tuple[str, str], list[HostSpanEvidence]] = {}
    for span in exclusive_spans:
        attributes = dict(span.attributes)
        if span.phase in {
            PhaseId.HOST_H2D_SUBMIT,
            PhaseId.HOST_D2H_MATERIALIZE,
        }:
            evaluation_id = _string(
                attributes.get("evaluation_id"), "host span evaluation_id"
            )
            observation = observations.get(evaluation_id)
            if observation is None:
                raise _fail(f"{child_id}: transfer span has unknown evaluation")
            if attributes.get("parameter_sha256") != observation.get(
                "parameter_sha256"
            ) or attributes.get("evaluation_kind") != observation.get("lifecycle"):
                raise _fail(f"{child_id}: transfer span identity mismatch")
            expected_iteration = (
                observation.get("iteration")
                if observation.get("lifecycle") == "trial"
                else None
            )
            if attributes.get("outer_iteration_id") != expected_iteration:
                raise _fail(f"{child_id}: transfer span iteration mismatch")
            lifecycle = event_groups[evaluation_id]
            if span.phase is PhaseId.HOST_H2D_SUBMIT:
                lower = lifecycle[HostEvent.EVALUATOR_ENTRY].timestamp_ns
                upper = lifecycle[HostEvent.DEVICE_READY].timestamp_ns
            else:
                lower = lifecycle[HostEvent.DEVICE_READY].timestamp_ns
                upper = lifecycle[HostEvent.EVALUATOR_RETURN].timestamp_ns
            if span.start_ns < lower or span.end_ns > upper:
                raise _fail(f"{child_id}: transfer span lies outside lifecycle bounds")
            transfer_spans.setdefault((span.phase, evaluation_id), []).append(span)
            continue

        previous_id = _string(
            attributes.get("previous_evaluation_id"),
            "line-search previous_evaluation_id",
        )
        next_id = _string(
            attributes.get("next_evaluation_id"), "line-search next_evaluation_id"
        )
        if previous_id not in observations or next_id not in observations:
            raise _fail(f"{child_id}: line-search span has unknown evaluation")
        previous_index = evaluations.index(observations[previous_id])
        next_index = evaluations.index(observations[next_id])
        if next_index != previous_index + 1:
            raise _fail(
                f"{child_id}: line-search span does not join adjacent evaluations"
            )
        if observations[next_id].get("lifecycle") != "trial":
            raise _fail(f"{child_id}: line-search span must lead to a trial")
        lower = event_groups[previous_id][HostEvent.EVALUATOR_RETURN].timestamp_ns
        upper = event_groups[next_id][HostEvent.EVALUATOR_ENTRY].timestamp_ns
        if span.start_ns < lower or span.end_ns > upper:
            raise _fail(f"{child_id}: line-search span lies outside evaluator gap")
        line_spans.setdefault((previous_id, next_id), []).append(span)

    for evaluation_id in observations:
        for phase in (PhaseId.HOST_H2D_SUBMIT, PhaseId.HOST_D2H_MATERIALIZE):
            if len(transfer_spans.get((phase, evaluation_id), ())) != 1:
                raise _fail(
                    f"{child_id}: evaluation {evaluation_id} requires exactly one "
                    f"{phase.value} span"
                )
    expected_lines = {
        (
            _string(evaluations[index - 1].get("evaluation_id"), "evaluation_id"),
            _string(evaluations[index].get("evaluation_id"), "evaluation_id"),
        )
        for index in range(1, len(evaluations) - 1)
    }
    if set(line_spans) != expected_lines or any(
        len(records) != 1 for records in line_spans.values()
    ):
        raise _fail(f"{child_id}: line-search host spans are missing or duplicated")

    accepted_optimizer_iterations: list[int] = []
    for span in optimizer_spans:
        attributes = dict(span.attributes)
        if set(attributes) != {"accepted_iteration_id"}:
            raise _fail(f"{child_id}: optimizer lifecycle span attributes drift")
        accepted_iteration = _integer(
            attributes.get("accepted_iteration_id"), "accepted_iteration_id"
        )
        accepted_optimizer_iterations.append(accepted_iteration)
    expected_optimizer_iterations = list(range(1, REQUIRED_ACCEPTED_ITERATIONS + 1))
    if accepted_optimizer_iterations != expected_optimizer_iterations:
        raise _fail(
            f"{child_id}: optimizer lifecycle spans do not bind accepted iterations"
        )


def _terminal_result(root: Path, evidence: ArtifactEvidence) -> dict[str, object]:
    entries = evidence.entries
    artifact_entry = _entry_by_role(entries, "artifact", "artifact_metadata")
    if artifact_entry.relative_path != "artifact.json":
        raise _fail("artifact metadata manifest role must bind artifact.json")
    if artifact_entry.evaluation_ids_sha256 != _evaluation_ids_sha256(()):
        raise _fail("artifact metadata has an invalid aggregate evaluation binding")
    artifact = _mapping(
        _load_canonical_json_bytes(evidence.files["artifact.json"], "artifact.json"),
        "artifact",
    )
    schedule = _validate_artifact_metadata(artifact, entries)
    identity_bindings = _validate_identity_preimages(evidence, artifact)

    children: dict[str, Mapping[str, object]] = {}
    profiler_buffers_dropped: dict[str, bool] = {}
    evaluations_by_child: dict[str, tuple[Mapping[str, object], ...]] = {}
    host_events_by_child: dict[str, tuple[HostEventRecord, ...]] = {}
    host_spans_by_child: dict[str, tuple[HostSpanEvidence, ...]] = {}
    segment_entries_by_child: dict[
        str, tuple[tuple[ManifestEntry, ManifestEntry], ...]
    ] = {}
    failed_children: list[
        tuple[int, str, Mapping[str, object], tuple[Mapping[str, object], ...]]
    ] = []
    for scheduled in schedule:
        child_id = _string(scheduled.get("child_id"), "child_id")
        child_entry = _entry_by_role(entries, child_id, "child_metadata")
        observations_entry = _entry_by_role(entries, child_id, "numerical_observations")
        events_entry = _entry_by_role(entries, child_id, "host_device_events")
        child = _mapping(
            _load_canonical_json_bytes(
                evidence.files[child_entry.relative_path], child_entry.relative_path
            ),
            f"{child_id}.child",
        )
        _validate_child_identity(artifact, scheduled, child)
        profiler_buffers_dropped[child_id] = _profiler_retention_evidence(
            evidence, entries, scheduled
        )
        children[child_id] = child
        observations = _mapping(
            _load_canonical_json_bytes(
                evidence.files[observations_entry.relative_path],
                observations_entry.relative_path,
            ),
            f"{child_id}.observations",
        )
        evaluations = tuple(
            _mapping(value, f"{child_id}.evaluations")
            for value in _sequence(
                observations.get("evaluations"), f"{child_id}.evaluations"
            )
        )
        evaluation_ids_digest = _evaluation_ids_sha256(
            tuple(
                _string(evaluation.get("evaluation_id"), "evaluation_id")
                for evaluation in evaluations
            )
        )
        if any(
            entry.evaluation_ids_sha256 != evaluation_ids_digest
            for entry in entries
            if entry.process_id == child_id
        ):
            raise _fail(
                f"{child_id}: manifest aggregate evaluation binding does not "
                "match raw observations"
            )
        raw_wall_ns, trajectory = _load_raw_child_records(
            evidence,
            entries,
            artifact,
            child_id,
            child,
            identity_bindings,
        )
        segment_entries_by_child[child_id] = _validate_segment_manifest_bindings(
            entries, child, evaluations
        )
        state = child.get("state")
        if state != "complete":
            _validate_boundary_pause_records(child, evaluations, complete=False)
            _validate_failed_child_raw_evidence(
                evidence, entries, scheduled, child, observations
            )
            _first_scientific_failure(child_id, evaluations)
            failed_children.append(
                (
                    _integer(scheduled.get("order_index"), "order_index"),
                    child_id,
                    child,
                    evaluations,
                )
            )
            continue
        _validate_observations(child_id, observations, artifact, child)
        _validate_boundary_pause_records(child, evaluations, complete=True)
        _validate_trajectory(child_id, trajectory, evaluations, raw_wall_ns)
        evaluations_by_child[child_id] = evaluations
        if scheduled.get("mode") == "profiled":
            host_events, host_spans = _load_host_evidence(
                evidence.files[events_entry.relative_path],
                events_entry.relative_path,
                child_id,
            )
            lifecycle_complete = _correlate_host_events_with_observations(
                child_id, host_events, evaluations
            )
            if lifecycle_complete:
                _validate_host_spans(child_id, host_spans, host_events, evaluations)
            host_events_by_child[child_id] = host_events
            host_spans_by_child[child_id] = host_spans
        elif evidence.files[events_entry.relative_path]:
            raise _fail(f"{child_id}: unprofiled control host-event file is not empty")

    cache_ids = [
        _string(child.get("cache_sha256"), "cache_sha256")
        for child in children.values()
    ]
    if len(cache_ids) != len(set(cache_ids)):
        raise _fail("children must use distinct compilation caches")

    integrity_failures = [
        (order_index, child_id, child)
        for order_index, child_id, child, _ in failed_children
        if child.get("failure_class") == "integrity"
    ]
    if integrity_failures:
        _, child_id, child = min(integrity_failures)
        raise _fail(
            f"{child_id}: child-reported integrity failure: "
            f"{_string(child.get('failure_reason'), 'failure_reason')}"
        )

    scientific_failures: list[tuple[int, int, str, str, tuple[str, ...]]] = []
    for order_index, child_id, child, evaluations in failed_children:
        derived = _first_scientific_failure(child_id, evaluations)
        if child.get("failure_class") == "scientific":
            if derived is None:
                raise _fail(
                    f"{child_id}: declared scientific failure has no raw failing gate"
                )
            evaluation_id, failure_codes = derived
            if child.get("first_failed_evaluation_id") != evaluation_id:
                raise _fail(
                    f"{child_id}: child failure identity is not bound to raw evidence"
                )
            evaluation_index = next(
                index
                for index, evaluation in enumerate(evaluations)
                if evaluation.get("evaluation_id") == evaluation_id
            )
            scientific_failures.append(
                (
                    order_index,
                    evaluation_index,
                    child_id,
                    evaluation_id,
                    failure_codes,
                )
            )
        elif derived is not None:
            raise _fail(
                f"{child_id}: non-scientific failure contradicts raw scientific gate"
            )
    if scientific_failures:
        _, _, child_id, evaluation_id, failure_codes = min(scientific_failures)
        return {
            "schema_id": VALIDATION_SCHEMA_ID,
            "artifact_schema_id": ARTIFACT_SCHEMA_ID,
            "verdict": "SCIENTIFIC_INVALID",
            "valid": True,
            "failing_gates": [f"{child_id}:{evaluation_id}:{','.join(failure_codes)}"],
            "first_failed_evaluation_id": evaluation_id,
            "metrics": None,
        }
    if failed_children:
        _, child_id, _, _ = min(failed_children)
        return {
            "schema_id": VALIDATION_SCHEMA_ID,
            "artifact_schema_id": ARTIFACT_SCHEMA_ID,
            "verdict": "UNATTRIBUTABLE",
            "valid": True,
            "failing_gates": [f"{child_id}:incomplete_without_scientific_failure"],
            "metrics": None,
        }

    profiles: list[ProfileEvidence] = []
    controls: list[ControlEvidence] = []
    overheads: list[float] = []
    for pair_index in range(PROFILE_CHILDREN):
        pair_entries = [
            entry for entry in schedule if entry.get("pair_index") == pair_index
        ]
        profile_id = _string(
            next(
                entry for entry in pair_entries if entry.get("mode") == "profiled"
            ).get("child_id"),
            "profile child_id",
        )
        control_id = _string(
            next(entry for entry in pair_entries if entry.get("mode") == "control").get(
                "child_id"
            ),
            "control child_id",
        )
        profile_child = children[profile_id]
        control_child = children[control_id]
        _compare_pair(
            profile_child,
            control_child,
            evaluations_by_child[profile_id],
            evaluations_by_child[control_id],
        )
        segment_entries = segment_entries_by_child[profile_id]
        profile_active_wall = _integer(
            profile_child.get("optimizer_active_wall_ns"), "optimizer_active_wall_ns"
        )
        control_active_wall = _integer(
            control_child.get("optimizer_active_wall_ns"), "optimizer_active_wall_ns"
        )
        profile_raw_wall = _integer(
            profile_child.get("optimizer_raw_wall_ns"), "optimizer_raw_wall_ns"
        )
        control_raw_wall = _integer(
            control_child.get("optimizer_raw_wall_ns"), "optimizer_raw_wall_ns"
        )
        profile_end_to_end = _integer(
            profile_child.get("child_end_to_end_ns"), "child_end_to_end_ns"
        )
        control_end_to_end = _integer(
            control_child.get("child_end_to_end_ns"), "child_end_to_end_ns"
        )
        try:
            if profiler_buffers_dropped[profile_id]:
                raise AttributionError(
                    f"{profile_id}: CUPTI activity buffers were dropped"
                )
            pause_records = tuple(
                _mapping(value, "boundary pause record")
                for value in _sequence(
                    profile_child.get("boundary_pause_records"),
                    "boundary_pause_records",
                )
            )
            segment_summaries = []
            for iteration, (trace_entry, summary_entry) in enumerate(
                segment_entries, start=1
            ):
                sample_id = f"iteration-{iteration:02d}"
                trace_document = _load_trace_document_bytes(
                    evidence.files[trace_entry.relative_path],
                    trace_entry.relative_path,
                )
                segment_summary = summarize_segmented_trace_document(
                    trace_document,
                    host_events_by_child[profile_id],
                    child_id=profile_id,
                    sample_id=sample_id,
                    accepted_iteration=iteration,
                    profiler_boundary_pauses=tuple(
                        Interval(
                            _integer(record.get("start_ns"), "pause start_ns"),
                            _integer(record.get("end_ns"), "pause end_ns"),
                        )
                        for record in pause_records
                        if any(
                            span.phase is PhaseId.HOST_LINE_SEARCH_CONTROL
                            and dict(span.attributes).get("outer_iteration_id")
                            == iteration
                            and span.start_ns
                            <= _integer(record.get("start_ns"), "pause start_ns")
                            and _integer(record.get("end_ns"), "pause end_ns")
                            <= span.end_ns
                            for span in host_spans_by_child[profile_id]
                        )
                    ),
                    evaluation_documents=evaluations_by_child[profile_id],
                )
                saved_summary = _mapping(
                    _load_canonical_json_bytes(
                        evidence.files[summary_entry.relative_path],
                        summary_entry.relative_path,
                    ),
                    f"{profile_id}.{sample_id}.summary",
                )
                if saved_summary != segment_summary.to_json():
                    raise _fail(
                        f"{profile_id}: {sample_id} saved summary differs from raw "
                        "recomputation"
                    )
                if (
                    segment_summary.raw_active_ns
                    != segment_summary.iteration.active_ns
                    + segment_summary.profiler_boundary_pause_ns
                ):
                    raise _fail(
                        f"{profile_id}: {sample_id} boundary pause subtraction differs"
                    )
                if (
                    segment_summary.segment_evaluation_ids_sha256
                    != trace_entry.segment_evaluation_ids_sha256
                ):
                    raise _fail(
                        f"{profile_id}: {sample_id} trace lifecycle evaluation differs"
                    )
                segment_summaries.append(segment_summary)
            summary = combine_segmented_trace_summaries(segment_summaries)
            if (
                not summary.trace_schema_valid
                or not summary.clock_correlation_valid
                or not summary.required_phase_families_present
                or not summary.semantic_counts_available
            ):
                raise AttributionError(
                    f"{profile_id}: trace coverage/correlation gate failed: "
                    f"{summary.diagnostics}"
                )
            for iteration in summary.iterations:
                if (
                    not iteration.required_phase_families_present
                    or iteration.missing_required_phases
                    or not iteration.semantic_counts_available
                ):
                    raise AttributionError(
                        f"{profile_id}: iteration {iteration.iteration} missing "
                        "required phases or exact semantic counts: "
                        f"{iteration.missing_required_phases}"
                    )
            shares = tuple(
                IterationShares(
                    iteration=iteration.iteration,
                    host_boundary=iteration.host_boundary_ns / iteration.active_ns,
                    newton_adjoint=(iteration.newton_adjoint_ns / iteration.active_ns),
                    other_attributed=(
                        iteration.other_attributed_ns / iteration.active_ns
                    ),
                    unattributed=iteration.unattributed_ns / iteration.active_ns,
                )
                for iteration in summary.iterations
            )
        except TraceSummaryError as error:
            return {
                "schema_id": VALIDATION_SCHEMA_ID,
                "artifact_schema_id": ARTIFACT_SCHEMA_ID,
                "verdict": "UNATTRIBUTABLE",
                "valid": True,
                "failing_gates": [f"{error.code}:{error}"],
                "metrics": None,
            }
        except (AttributionError, ZeroDivisionError) as error:
            return {
                "schema_id": VALIDATION_SCHEMA_ID,
                "artifact_schema_id": ARTIFACT_SCHEMA_ID,
                "verdict": "UNATTRIBUTABLE",
                "valid": True,
                "failing_gates": [str(error)],
                "metrics": None,
            }
        profiles.append(
            ProfileEvidence(
                child_id=profile_id,
                pair_index=pair_index,
                active_wall_ns=profile_active_wall,
                raw_wall_ns=profile_raw_wall,
                child_end_to_end_ns=profile_end_to_end,
                boundary_pause_ns=_integer(
                    profile_child.get("profiler_boundary_pause_total_ns"),
                    "profiler_boundary_pause_total_ns",
                ),
                process_host_median=statistics.median(
                    share.host_boundary for share in shares
                ),
                process_newton_median=statistics.median(
                    share.newton_adjoint for share in shares
                ),
                process_unattributed_median=statistics.median(
                    share.unattributed for share in shares
                ),
                iteration_shares=shares,
            )
        )
        controls.append(
            ControlEvidence(
                control_id,
                pair_index,
                control_active_wall,
                control_raw_wall,
                control_end_to_end,
            )
        )
        overheads.append(profile_active_wall / control_active_wall)

    median_overhead = statistics.median(overheads)
    all_shares = [share for profile in profiles for share in profile.iteration_shares]
    pooled_host = statistics.median(share.host_boundary for share in all_shares)
    pooled_newton = statistics.median(share.newton_adjoint for share in all_shares)
    pooled_unattributed = statistics.median(share.unattributed for share in all_shares)
    attribution_failures: list[str] = []
    if median_overhead > MAX_PROFILE_OVERHEAD:
        attribution_failures.append("profiler_overhead_above_10_percent")
    if pooled_unattributed > MAX_UNATTRIBUTED_SHARE:
        attribution_failures.append("unattributed_share_above_20_percent")
    if any(
        profile.process_unattributed_median > MAX_UNATTRIBUTED_SHARE
        for profile in profiles
    ):
        attribution_failures.append("process_unattributed_share_above_20_percent")
    metrics = {
        "paired_profile_control_wall_ratios": overheads,
        "median_profile_overhead_ratio": median_overhead,
        "profile_optimizer_raw_wall_ns": [profile.raw_wall_ns for profile in profiles],
        "profile_optimizer_active_wall_ns": [
            profile.active_wall_ns for profile in profiles
        ],
        "profile_boundary_pause_ns": [
            profile.boundary_pause_ns for profile in profiles
        ],
        "profile_child_end_to_end_ns": [
            profile.child_end_to_end_ns for profile in profiles
        ],
        "control_optimizer_raw_wall_ns": [control.raw_wall_ns for control in controls],
        "control_optimizer_active_wall_ns": [
            control.active_wall_ns for control in controls
        ],
        "control_child_end_to_end_ns": [
            control.child_end_to_end_ns for control in controls
        ],
        "pooled_host_boundary_share": pooled_host,
        "pooled_newton_adjoint_share": pooled_newton,
        "pooled_unattributed_share": pooled_unattributed,
        "process_host_medians": [profile.process_host_median for profile in profiles],
        "process_newton_medians": [
            profile.process_newton_median for profile in profiles
        ],
        "process_unattributed_medians": [
            profile.process_unattributed_median for profile in profiles
        ],
        "iteration_shares": [
            {
                "child_id": profile.child_id,
                **share.__dict__,
            }
            for profile in profiles
            for share in profile.iteration_shares
        ],
    }
    if attribution_failures:
        return {
            "schema_id": VALIDATION_SCHEMA_ID,
            "artifact_schema_id": ARTIFACT_SCHEMA_ID,
            "verdict": "UNATTRIBUTABLE",
            "valid": True,
            "failing_gates": attribution_failures,
            "metrics": None,
        }

    host_dominant = (
        pooled_host >= DOMINANCE_POOLED_SHARE
        and all(
            profile.process_host_median >= DOMINANCE_PROCESS_SHARE
            for profile in profiles
        )
        and pooled_host - pooled_newton >= DOMINANCE_LEAD
    )
    newton_dominant = (
        pooled_newton >= DOMINANCE_POOLED_SHARE
        and all(
            profile.process_newton_median >= DOMINANCE_PROCESS_SHARE
            for profile in profiles
        )
        and pooled_newton - pooled_host >= DOMINANCE_LEAD
    )
    if host_dominant == newton_dominant and host_dominant:
        raise _fail("dominance partition is not mutually exclusive")
    verdict: Verdict
    if host_dominant:
        verdict = "HOST_BOUNDARY_DOMINANT"
    elif newton_dominant:
        verdict = "NEWTON_ADJOINT_DOMINANT"
    else:
        verdict = "MIXED"
    return {
        "schema_id": VALIDATION_SCHEMA_ID,
        "artifact_schema_id": ARTIFACT_SCHEMA_ID,
        "verdict": verdict,
        "valid": True,
        "failing_gates": [],
        "metrics": metrics,
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_validation(root: Path, result: Mapping[str, object]) -> Path:
    validation_root = root.parent / f"{root.name}.validation"
    if validation_root.exists():
        raise FileExistsError(f"validation result already exists: {validation_root}")
    validation_root.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(validation_root.parent)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{validation_root.name}.staging-", dir=validation_root.parent
        )
    )
    try:
        result_path = staging_root / "validation_result.json"
        with result_path.open("xb") as stream:
            stream.write(_canonical_json_bytes(result))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging_root)
        staging_root.rename(validation_root)
        _fsync_directory(validation_root.parent)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return validation_root / "validation_result.json"


def validate_and_publish(root: Path) -> tuple[Path, Mapping[str, object]]:
    """Validate raw evidence and exclusively publish the terminal result."""

    evidence: ArtifactEvidence | None = None
    try:
        evidence = _load_manifest(root)
        result = _terminal_result(root, evidence)
    except AttributionError as error:
        result = {
            "schema_id": VALIDATION_SCHEMA_ID,
            "artifact_schema_id": ARTIFACT_SCHEMA_ID,
            "verdict": "UNATTRIBUTABLE",
            "valid": True,
            "failing_gates": [str(error)],
            "metrics": None,
        }
    except (
        IntegrityError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        ZeroDivisionError,
    ) as error:
        result = {
            "schema_id": VALIDATION_SCHEMA_ID,
            "artifact_schema_id": ARTIFACT_SCHEMA_ID,
            "verdict": "INTEGRITY_ERROR",
            "valid": False,
            "failing_gates": [str(error)],
            "metrics": None,
        }

    def digest_if_regular(path: Path) -> str | None:
        if not path.is_file() or path.is_symlink():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    artifact_digest = (
        hashlib.sha256(evidence.files["artifact.json"]).hexdigest()
        if evidence is not None and "artifact.json" in evidence.files
        else digest_if_regular(root / "artifact.json")
    )
    manifest_digest = (
        hashlib.sha256(evidence.manifest_bytes).hexdigest()
        if evidence is not None
        else digest_if_regular(root / "manifest.json")
    )
    source_provenance_authoritative: bool | None = None
    if evidence is not None and "artifact.json" in evidence.files:
        artifact_document = json.loads(evidence.files["artifact.json"])
        declared_authority = artifact_document.get("authoritative")
        if isinstance(declared_authority, bool):
            source_provenance_authoritative = declared_authority
    attribution_verdicts = {
        "HOST_BOUNDARY_DOMINANT",
        "NEWTON_ADJOINT_DOMINANT",
        "MIXED",
    }
    engineering_branch_eligible = (
        result.get("valid") is True and result.get("verdict") in attribution_verdicts
    )
    bound_result = {
        **result,
        "source_provenance_authoritative": source_provenance_authoritative,
        "promotion_eligible": (
            engineering_branch_eligible and source_provenance_authoritative is True
        ),
        "engineering_branch_eligible": engineering_branch_eligible,
        "claim_ceiling": (
            "protocol_attribution"
            if source_provenance_authoritative is True
            else "diagnostic_attribution_only"
        ),
        "artifact_root": str(root.resolve(strict=False)),
        "artifact_json_sha256": artifact_digest,
        "manifest_json_sha256": manifest_digest,
        "validator_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "validated_utc": datetime.now(UTC).isoformat(),
    }
    return _publish_validation(root, bound_result), bound_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    arguments = parser.parse_args()
    try:
        result_path, result = validate_and_publish(arguments.artifact_root.expanduser())
    except (FileExistsError, OSError) as error:
        print(f"INTEGRITY_ERROR: validation publication failed: {error}")
        return 2
    print(f"validation_result: {result_path}")
    print(f"VERDICT: {result['verdict']}")
    return 2 if result["verdict"] == "INTEGRITY_ERROR" else 0


if __name__ == "__main__":
    sys.exit(main())
