"""Run the bounded changed-state GPU timeline protocol.

The parent process owns immutable scheduling, runtime provenance, resource
bounds, and receipt publication. Each fresh profiled child warms the exact
annotated graph, then collects seven sequential JAX sessions around the seven
accepted steps of one unchanged optimizer execution. Numerical failures are
published as diagnostic evidence; they are never replaced with timings.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypeVar

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
for _import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from examples.jax.parity.cases.native_boozerqa import (
    ChangedStateTimelineDisposition,
    ChangedStateTimelineObservation,
    ChangedStateTimelineRecord,
    _prepare_jax_variant_execution,
    changed_state_timeline_observation_sink,
)
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import (
    InputBundle,
    read_input_bundle,
)
from examples.jax.parity.measurement import MeasurementExecution
from examples.jax.parity.provenance import collect_lane_provenance
from simsopt.optimization_trajectory import read_optimization_window_timing
from simsopt.single_stage_boozer_vacuum import JAX_FAST_DRIVER_ID
from simsopt_jax.runtime.trace_annotations import (
    HostTraceAudit,
    SegmentedProfilerBoundaryAudit,
    segmented_profiler_boundaries,
    trace_session,
)

from benchmarks.run_jax_native_example_measurements import (
    _cpu_model,
    _gpu_concurrent_use_preflight,
    _gpu_identity,
    build_measurement_environment,
    execute_monitored_command,
)
from benchmarks.single_stage_changed_state_gpu_timeline_receipt import (
    ARTIFACT_SCHEMA_ID,
    CHILD_SCHEMA_ID,
    DIRECT_ADJOINT_ROUTE,
    EVENT_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    PRODUCTION_DRIVER,
    PRODUCTION_LINE_SEARCH,
    PRODUCTION_OPTIMIZER,
    REQUIRED_ACCEPTED_ITERATIONS,
    REQUIRED_CONTROL_CHILDREN,
    REQUIRED_PROFILE_CHILDREN,
    ChildScheduleEntry,
    ClaimFile,
    RouteIdentity,
    TimelineMetadata,
    canonical_json_bytes,
    evaluation_ids_sha256,
    write_timeline_receipt,
)
from benchmarks.single_stage_changed_state_profiler_policy import (
    CUPTI_ACTIVITY_DROP_WARNING,
    TRACE_VIEWER_MAX_EVENTS,
    TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT,
    ProfilerPolicy,
    build_jax_profiler_options,
    parse_profiler_policy,
    profiler_policy,
    profiler_policy_document,
)
from benchmarks.single_stage_changed_state_trace_preflight import (
    PREFLIGHT_SCHEMA_ID,
    DeviceIdentity,
    PreflightEvidenceError,
    validate_passing_preflight_evidence,
)

CASE_ID: Final = "native-single-stage-boozer-vacuum-optimization"
TRACE_PHASE_SCHEMA_VERSION: Final = "single-stage-timeline-phases-v1"
_EXACT_ADJOINT_ENVIRONMENT: Final = "SIMSOPT_EXACT_ADJOINT_DENSE_LU"
_DEFAULT_TIMEOUT_SECONDS: Final = 1_800.0
_DEFAULT_MAX_PROCESS_TREE_RSS_BYTES: Final = 96 * 1024**3
_DEFAULT_POLL_INTERVAL_SECONDS: Final = 0.05
_ARTIFACT_NAME_PREFIX: Final = ARTIFACT_SCHEMA_ID
_CLAIM_FILES: Final = {
    "child_metadata": "child.json",
    "host_device_events": "events.jsonl",
    "numerical_observations": "observations.json",
    "optimization_timing": "timing.json",
    "trajectory": "trajectory.jsonl",
    "provenance": "provenance.json",
}
_SEGMENTS_DIRECTORY: Final = "segments"
_CLAIMED_ENVIRONMENT_NAMES: Final = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "JAX_TRANSFER_GUARD",
        "MPI4PY_RC_INITIALIZE",
        "OMP_DYNAMIC",
        "OMP_NUM_THREADS",
        "OMP_SCHEDULE",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "SIMSOPT_BACKEND_MODE",
        "SIMSOPT_BACKEND_STRICT",
        "SIMSOPT_EXACT_ADJOINT_DENSE_LU",
        "SIMSOPT_JAX_TRANSFER_GUARD",
        "SIMSOPT_PRECISION",
        "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    }
)

ChildMode = Literal["profiled", "control"]
_ExecutionResult = TypeVar("_ExecutionResult")


class TimelineRunnerError(RuntimeError):
    """The bounded runner cannot publish trustworthy timeline evidence."""


class ScientificTimelineError(TimelineRunnerError):
    """One concrete changed-state evaluation failed numerical eligibility."""

    def __init__(self, message: str, evaluation_id: str) -> None:
        super().__init__(message)
        self.evaluation_id = evaluation_id


class TraceCollectionError(TimelineRunnerError):
    """Profiler capture or trace summarization failed independently of numerics."""


def _execute_segmented_profiled_measurement(
    execute: Callable[[], _ExecutionResult],
    *,
    start_segment: Callable[[int], None],
    stop_segment: Callable[[int], None],
) -> tuple[_ExecutionResult, HostTraceAudit, SegmentedProfilerBoundaryAudit]:
    """Execute once with seven optimizer-owned sequential profiler boundaries."""

    with trace_session() as audit, segmented_profiler_boundaries(
        start_segment, stop_segment
    ) as boundary_audit:
        result = execute()
    return result, audit, boundary_audit


@dataclass(frozen=True, slots=True)
class ChildSpec:
    """All child-varying paths plus the parent-frozen execution identity."""

    child_id: str
    mode: ChildMode
    pair_index: int
    order_index: int
    input_root: str
    output_root: str
    trace_root: str
    cache_sha256: str
    source_state_sha256: str
    environment_sha256: str
    input_sha256: str
    configuration_sha256: str
    construction_sha256: str
    runtime_policy_sha256: str
    initial_parameters_sha256: str
    device_name: str
    device_uuid: str
    environment: tuple[tuple[str, str], ...]
    phase_ids: tuple[str, ...]
    trace_schema_id: str
    profiler_policy: ProfilerPolicy


def child_schedule(
    profile_children: int = REQUIRED_PROFILE_CHILDREN,
    control_children: int = REQUIRED_CONTROL_CHILDREN,
) -> tuple[ChildScheduleEntry, ...]:
    """Return the frozen alternating three-pair execution order."""

    if (
        profile_children != REQUIRED_PROFILE_CHILDREN
        or control_children != REQUIRED_CONTROL_CHILDREN
    ):
        raise ValueError("timeline requires exactly three profiled and three controls")
    entries: list[ChildScheduleEntry] = []
    for pair_index in range(REQUIRED_PROFILE_CHILDREN):
        for mode in ("profiled", "control"):
            entries.append(
                ChildScheduleEntry(
                    child_id=f"{mode}-{pair_index}",
                    mode=mode,
                    pair_index=pair_index,
                    order_index=len(entries),
                )
            )
    return tuple(entries)


def validate_artifact_root(artifact_root: Path) -> Path:
    """Require a fresh diagnostic-specific, non-temporary publication root."""

    resolved = artifact_root.expanduser().resolve()
    if resolved.is_relative_to(Path("/tmp")):
        raise TimelineRunnerError("authoritative timeline root must not be under /tmp")
    if resolved.is_relative_to(_REPO_ROOT):
        raise TimelineRunnerError(
            "authoritative timeline root must be outside the repo"
        )
    if any(
        path.name == "campaign-20260804-frozen-r5"
        for path in (resolved, *resolved.parents)
    ):
        raise TimelineRunnerError(
            "timeline root must not be the frozen r5 root or any descendant"
        )
    if not (
        resolved.name == _ARTIFACT_NAME_PREFIX
        or resolved.name.startswith(f"{_ARTIFACT_NAME_PREFIX}-")
    ):
        raise TimelineRunnerError(
            f"artifact root name must identify {ARTIFACT_SCHEMA_ID}"
        )
    if resolved.exists():
        raise FileExistsError(f"artifact root already exists: {resolved}")
    return resolved


def build_child_command(
    python_executable: str, child_spec_path: Path
) -> tuple[str, ...]:
    """Build one fresh-process invocation of this runner's child mode."""

    return (
        python_executable,
        str(Path(__file__).resolve()),
        "--child-spec",
        str(child_spec_path.resolve()),
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8")).reshape(-1)
    return _sha256_bytes(canonical.tobytes(order="C"))


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, document: object) -> None:
    _write_bytes_exclusive(path, canonical_json_bytes(document))


def _write_jsonl_exclusive(path: Path, records: Sequence[object]) -> None:
    _write_bytes_exclusive(path, b"".join(canonical_json_bytes(row) for row in records))


def _load_canonical_json(path: Path) -> Mapping[str, object]:
    content = path.read_bytes()
    document = json.loads(content)
    if not isinstance(document, dict) or content != canonical_json_bytes(document):
        raise TimelineRunnerError(f"non-canonical child JSON: {path}")
    return document


def _child_spec_document(spec: ChildSpec) -> dict[str, object]:
    return {
        "schema_id": "single-stage-changed-state-child-spec-segmented-v2",
        **asdict(spec),
        "profiler_policy": _profiler_policy_document(spec.profiler_policy),
    }


def _profiler_policy_document(policy: ProfilerPolicy) -> dict[str, object]:
    return profiler_policy_document(policy)


def _parse_profiler_policy(document: object) -> ProfilerPolicy:
    try:
        return parse_profiler_policy(document)
    except ValueError as error:
        raise TimelineRunnerError(
            "child profiler policy differs from schema"
        ) from error


def _profiler_policy(mode: ChildMode) -> ProfilerPolicy:
    return profiler_policy(mode == "profiled")


def _jax_profiler_options(jax, policy: ProfilerPolicy):
    """Construct JAX options from the already receipt-bound child policy."""

    try:
        return build_jax_profiler_options(jax.profiler.ProfileOptions, policy)
    except ValueError as error:
        raise TimelineRunnerError(
            "profiled execution has unexpected profiler policy"
        ) from error


def _profiler_retention_document(
    mode: ChildMode, stderr_bytes: bytes
) -> dict[str, object]:
    activity_buffers_dropped = (
        CUPTI_ACTIVITY_DROP_WARNING.encode("utf-8") in stderr_bytes
        if mode == "profiled"
        else None
    )
    return {
        "evidence_available": mode == "profiled",
        "activity_buffers_dropped": activity_buffers_dropped,
        "warning": CUPTI_ACTIVITY_DROP_WARNING if activity_buffers_dropped else None,
    }


def _read_child_spec(path: Path) -> ChildSpec:
    document = _load_canonical_json(path)
    if document.get("schema_id") != (
        "single-stage-changed-state-child-spec-segmented-v2"
    ):
        raise TimelineRunnerError("unknown child specification schema")
    fields = {field.name for field in dataclasses.fields(ChildSpec)}
    if set(document) != fields | {"schema_id"}:
        raise TimelineRunnerError("child specification fields differ from schema")
    mode_value = document["mode"]
    if mode_value not in ("profiled", "control"):
        raise TimelineRunnerError("child mode must be profiled or control")
    return ChildSpec(
        child_id=str(document["child_id"]),
        mode=mode_value,
        pair_index=int(document["pair_index"]),
        order_index=int(document["order_index"]),
        input_root=str(document["input_root"]),
        output_root=str(document["output_root"]),
        trace_root=str(document["trace_root"]),
        cache_sha256=str(document["cache_sha256"]),
        source_state_sha256=str(document["source_state_sha256"]),
        environment_sha256=str(document["environment_sha256"]),
        input_sha256=str(document["input_sha256"]),
        configuration_sha256=str(document["configuration_sha256"]),
        construction_sha256=str(document["construction_sha256"]),
        runtime_policy_sha256=str(document["runtime_policy_sha256"]),
        initial_parameters_sha256=str(document["initial_parameters_sha256"]),
        device_name=str(document["device_name"]),
        device_uuid=str(document["device_uuid"]),
        environment=tuple(
            (str(item[0]), str(item[1])) for item in document["environment"]
        ),
        phase_ids=tuple(str(value) for value in document["phase_ids"]),
        trace_schema_id=str(document["trace_schema_id"]),
        profiler_policy=_parse_profiler_policy(document["profiler_policy"]),
    )


def _route_document() -> dict[str, str]:
    return {
        "optimizer": PRODUCTION_OPTIMIZER,
        "driver": PRODUCTION_DRIVER,
        "line_search": PRODUCTION_LINE_SEARCH,
        "adjoint_route": DIRECT_ADJOINT_ROUTE,
    }


def _claimed_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: environment[name]
        for name in sorted(_CLAIMED_ENVIRONMENT_NAMES)
        if name in environment
    }


def _assert_source_state(expected_sha256: str) -> None:
    if _timeline_source_state_sha256(_REPO_ROOT) != expected_sha256:
        raise TimelineRunnerError("source state changed after parent preflight")


def _timeline_source_state_sha256(repo_root: Path) -> str:
    """Hash HEAD, tracked changes, and exact nonignored untracked file bytes."""

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
            raise TimelineRunnerError(
                f"cannot compute source state: git {' '.join(arguments)}"
            )
        digest.update(len(completed.stdout).to_bytes(8, "little"))
        digest.update(completed.stdout)
    untracked = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if untracked.returncode != 0:
        raise TimelineRunnerError("cannot enumerate untracked source-state files")
    relative_paths = tuple(
        sorted(path for path in untracked.stdout.split(b"\0") if path)
    )
    for relative_bytes in relative_paths:
        relative_path = relative_bytes.decode("utf-8")
        path = repo_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise TimelineRunnerError(
                f"untracked source-state path is not a regular file: {relative_path}"
            )
        digest.update(len(relative_bytes).to_bytes(8, "little"))
        digest.update(relative_bytes)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _phase_ids() -> tuple[str, ...]:
    from simsopt_jax.runtime.trace_annotations import PhaseId

    return tuple(phase.value for phase in PhaseId)


def _endpoint_certificate(observation) -> dict[str, object]:
    values = observation.values
    return {
        "success": bool(np.asarray(values["final:endpoint_certificate_success"])),
        "initial_stationary": bool(
            np.asarray(values["final:endpoint_initial_stationary"])
        ),
        "terminal_stationary": bool(
            np.asarray(values["final:endpoint_terminal_stationary"])
        ),
        "constraints_satisfied": bool(
            np.asarray(values["final:endpoint_constraints_satisfied"])
        ),
        "outer_status": int(np.asarray(values["final:outer_solver_status"])),
    }


def _child_identity(spec: ChildSpec) -> dict[str, object]:
    return {
        "schema_id": CHILD_SCHEMA_ID,
        "child_id": spec.child_id,
        "mode": spec.mode,
        "pair_index": spec.pair_index,
        "order_index": spec.order_index,
        "route": _route_document(),
        "source_state_sha256": spec.source_state_sha256,
        "environment_sha256": spec.environment_sha256,
        "input_sha256": spec.input_sha256,
        "configuration_sha256": spec.configuration_sha256,
        "construction_sha256": spec.construction_sha256,
        "runtime_policy_sha256": spec.runtime_policy_sha256,
        "initial_parameters_sha256": spec.initial_parameters_sha256,
        "device_name": spec.device_name,
        "device_uuid": spec.device_uuid,
        "cache_sha256": spec.cache_sha256,
        "profiler_policy": _profiler_policy_document(spec.profiler_policy),
    }


def _failure_documents(
    spec: ChildSpec,
    *,
    failure_class: Literal["scientific", "trace", "integrity"],
    failure_reason: str,
    child_end_to_end_ns: int,
    provenance,
    evaluations: Sequence[Mapping[str, object]] = (),
    first_failed_evaluation_id: str | None = None,
    boundary_pause_records: Sequence[Mapping[str, object]] = (),
) -> tuple[dict[str, object], dict[str, object], tuple[dict[str, object], ...]]:
    child = {
        **_child_identity(spec),
        "state": "failed",
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "optimizer_raw_wall_ns": None,
        "profiler_boundary_pause_total_ns": sum(
            int(record["duration_ns"]) for record in boundary_pause_records
        ),
        "optimizer_active_wall_ns": None,
        "child_end_to_end_ns": child_end_to_end_ns,
        "boundary_pause_records": [dict(record) for record in boundary_pause_records],
        "nit": None,
        "nfev": None,
        "njev": None,
        "status": "not_completed",
        "line_search_decisions": [],
        "endpoint_certificate": None,
        "phase_ids": list(spec.phase_ids),
        "trace_schema_id": spec.trace_schema_id,
        "first_failed_evaluation_id": first_failed_evaluation_id,
        "provenance": _child_provenance_document(provenance),
    }
    observations = {
        "schema_id": OBSERVATION_SCHEMA_ID,
        "child_id": spec.child_id,
        "evaluations": list(evaluations),
        "failure_reason": failure_reason,
        "first_failed_evaluation_id": first_failed_evaluation_id,
    }
    events = (
        {
            "schema_id": EVENT_SCHEMA_ID,
            "child_id": spec.child_id,
            "sequence": 0,
            "event": "diagnostic_failure",
            "failure_reason": failure_reason,
        },
    )
    return child, observations, events


def _boundary_pause_documents(records) -> tuple[dict[str, object], ...]:
    """Serialize benchmark hook pauses without erasing a failed boundary."""

    return tuple(
        {
            "iteration_id": record.iteration_id,
            "operation": record.operation.value,
            "start_ns": record.start_ns,
            "end_ns": record.end_ns,
            "duration_ns": record.end_ns - record.start_ns,
        }
        for record in records
    )


def _segment_evaluation_ids(
    evaluations: Sequence[Mapping[str, object]], iteration_id: int
) -> tuple[str, ...]:
    """Return all trial identities executed inside one accepted-step segment."""

    evaluation_ids = tuple(
        str(row["evaluation_id"])
        for row in evaluations
        if row.get("lifecycle") == "trial" and row.get("iteration") == iteration_id
    )
    if not evaluation_ids:
        raise TimelineRunnerError(
            f"iteration {iteration_id} has no trial evaluation identities"
        )
    return evaluation_ids


def _validate_complete_boundary_records(records) -> None:
    observed = tuple(
        (record.iteration_id, record.operation.value) for record in records
    )
    expected = tuple(
        (iteration_id, operation)
        for iteration_id in range(1, REQUIRED_ACCEPTED_ITERATIONS + 1)
        for operation in ("start", "stop")
    )
    if observed != expected:
        raise TraceCollectionError(
            "profiled execution did not emit exact sequential start/stop boundaries"
        )


def _iteration_pause_intervals(records, span_records, iteration_id: int):
    """Return profiler pauses contained in this iteration's host-control gaps."""

    from simsopt_jax.runtime.trace_annotations import PhaseId

    from benchmarks.summarize_single_stage_changed_state_gpu_timeline import Interval

    gaps = tuple(
        record
        for record in span_records
        if record.phase is PhaseId.HOST_LINE_SEARCH_CONTROL
        and dict(record.attributes).get("outer_iteration_id") == iteration_id
    )
    return tuple(
        Interval(record.start_ns, record.end_ns)
        for record in records
        if any(
            gap.start_ns <= record.start_ns and record.end_ns <= gap.end_ns
            for gap in gaps
        )
    )


def _serialize_host_evidence(
    child_id: str, lifecycle_records, span_records
) -> tuple[dict[str, object], ...]:
    lifecycle = tuple(
        {
            "schema_id": EVENT_SCHEMA_ID,
            "child_id": child_id,
            "record_type": "lifecycle",
            "sequence": record.sequence,
            "event": record.event.value,
            "timestamp_ns": record.timestamp_ns,
            "evaluation_id": record.evaluation.evaluation_id,
            "parameter_sha256": record.evaluation.parameter_sha256,
            "evaluation_kind": record.evaluation.kind.value,
            "outer_iteration_id": record.evaluation.outer_iteration_id,
            "attributes": dict(record.attributes),
        }
        for record in lifecycle_records
    )
    spans = tuple(
        {
            "schema_id": EVENT_SCHEMA_ID,
            "child_id": child_id,
            "record_type": (
                "optimizer_span"
                if record.phase.value == "optimizer.lifecycle"
                else "host_span"
            ),
            "sequence": record.sequence,
            "phase_id": record.phase.value,
            "start_ns": record.start_ns,
            "end_ns": record.end_ns,
            "depth": record.depth,
            "attributes": dict(record.attributes),
        }
        for record in span_records
    )
    return lifecycle + spans


def _evaluation_documents(
    records: Sequence[ChangedStateTimelineRecord],
    *,
    require_dispositions: bool = True,
) -> tuple[dict[str, object], ...]:
    observations = tuple(
        record
        for record in records
        if isinstance(record, ChangedStateTimelineObservation)
    )
    dispositions = {
        record.evaluation_id: record
        for record in records
        if isinstance(record, ChangedStateTimelineDisposition)
    }
    rows: list[dict[str, object]] = []
    for evaluation_index, observation in enumerate(observations):
        disposition = dispositions.get(observation.evaluation_id)
        is_trial = observation.evaluation_kind == "trial"
        if require_dispositions and is_trial and disposition is None:
            raise TimelineRunnerError(
                f"trial {observation.evaluation_id} has no optimizer disposition"
            )
        accepted = (
            disposition.disposition == "accepted" if disposition is not None else None
        )
        iteration = (
            disposition.accepted_iteration_id
            if disposition is not None and accepted
            else observation.outer_iteration_id
        )
        rows.append(
            {
                "evaluation_id": observation.evaluation_id,
                "evaluation_index": evaluation_index,
                "lifecycle": observation.evaluation_kind,
                "iteration": iteration,
                "parameter_sha256": observation.parameter_sha256,
                "parameters": list(observation.parameters),
                "parameter_shape": list(observation.parameter_shape),
                "objective": observation.objective,
                "dtype": "float64",
                "gradient": list(observation.gradient),
                "gradient_shape": list(observation.gradient_shape),
                "gradient_source": observation.gradient_source,
                "values_finite": observation.values_finite,
                "inner_success": bool(
                    observation.forward_success and observation.primal_success
                ),
                "adjoint_success": observation.actual_adjoint_success,
                "candidate_gradient_source": observation.candidate_gradient_source,
                "eligible": observation.eligible,
                "trajectory_valid": True,
                "accepted": accepted,
                "inner_evidence": {
                    "residual_trace": list(observation.inner_residual_trace),
                    "step_accepted_trace": list(observation.newton_step_accepted_trace),
                    "linear_solve_success_trace": list(
                        observation.newton_linear_solve_success_trace
                    ),
                    "newton_iterations": observation.newton_iterations,
                    "newton_attempted_iterations": (
                        observation.newton_attempted_iterations
                    ),
                    "newton_trace_available": observation.newton_trace_available,
                },
                "adjoint_evidence": {
                    "route": observation.adjoint_route,
                    "output": list(observation.adjoint_output),
                    "residual": observation.adjoint_residual,
                    "residual_relative": observation.adjoint_residual_relative,
                    "dense_materializations": observation.dense_materializations,
                    "lu_factorizations": observation.lu_factorizations,
                    "lu_solves": observation.lu_solves,
                    "refinement_corrections": observation.refinement_corrections,
                    "adjoint_executions": observation.adjoint_executions,
                },
                "observables": dict(observation.observables),
            }
        )
    return tuple(rows)


def _line_search_decisions(
    evaluations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "evaluation_id": row["evaluation_id"],
            "accepted": row["accepted"],
            "iteration": row["iteration"],
        }
        for row in evaluations
        if row["lifecycle"] == "trial"
    ]


def _validate_newton_inner_evidence(inner_evidence: Mapping[str, object]) -> None:
    attempted = inner_evidence.get("newton_attempted_iterations")
    trace_available = inner_evidence.get("newton_trace_available")
    step_trace = inner_evidence.get("step_accepted_trace")
    linear_trace = inner_evidence.get("linear_solve_success_trace")
    residual_trace = inner_evidence.get("residual_trace")
    newton_iterations = inner_evidence.get("newton_iterations")
    if (
        isinstance(newton_iterations, bool)
        or not isinstance(newton_iterations, int)
        or newton_iterations < 0
    ):
        raise TimelineRunnerError("Newton iteration count is invalid")
    if not isinstance(trace_available, bool):
        raise TimelineRunnerError("Newton trace availability is invalid")
    if not all(
        isinstance(trace, list) for trace in (step_trace, linear_trace, residual_trace)
    ):
        raise TimelineRunnerError("Newton trace arrays must be JSON arrays")
    if trace_available:
        if (
            isinstance(attempted, bool)
            or not isinstance(attempted, int)
            or attempted < 0
        ):
            raise TimelineRunnerError(
                "available Newton trace has invalid attempted count"
            )
        if not (
            len(step_trace) == len(linear_trace) == len(residual_trace) == attempted
        ):
            raise TimelineRunnerError("available Newton trace lengths are inconsistent")
        if not all(isinstance(value, bool) for value in (*step_trace, *linear_trace)):
            raise TimelineRunnerError("Newton decision traces must contain booleans")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in residual_trace
        ):
            raise TimelineRunnerError("Newton residual trace must contain numbers")
        if sum(step_trace) != newton_iterations:
            raise TimelineRunnerError(
                "Newton accepted-step count differs from available trace"
            )
    elif attempted is not None or any((step_trace, linear_trace, residual_trace)):
        raise TimelineRunnerError(
            "unavailable Newton trace must have null count and empty arrays"
        )


def _validate_evaluations(evaluations: Sequence[Mapping[str, object]]) -> None:
    initial = [row for row in evaluations if row["lifecycle"] == "initial"]
    trials = [row for row in evaluations if row["lifecycle"] == "trial"]
    final = [row for row in evaluations if row["lifecycle"] == "final_reporting"]
    if len(initial) != 1 or len(final) != 1 or not trials:
        raise ScientificTimelineError(
            "timeline requires one initial, optimizer trials, and one final report",
            str(evaluations[-1]["evaluation_id"]) if evaluations else "unavailable",
        )
    if initial[0]["accepted"] is not None or final[0]["accepted"] is not None:
        raise ScientificTimelineError(
            "only optimizer trials may have dispositions",
            str(final[0]["evaluation_id"]),
        )
    for row in evaluations:
        parameters = np.asarray(row["parameters"], dtype=np.float64)
        gradient = np.asarray(row["gradient"], dtype=np.float64)
        if list(parameters.shape) != row["parameter_shape"]:
            raise TimelineRunnerError("parameter shape differs from serialized values")
        if list(gradient.shape) != row["gradient_shape"]:
            raise TimelineRunnerError("gradient shape differs from serialized values")
        if _array_sha256(parameters) != row["parameter_sha256"]:
            raise TimelineRunnerError("parameter SHA-256 differs from raw FP64 values")
        inner_evidence = row["inner_evidence"]
        if not isinstance(inner_evidence, dict):
            raise TimelineRunnerError("inner evidence is not a mapping")
        _validate_newton_inner_evidence(inner_evidence)
        residual_trace = inner_evidence["residual_trace"]
        adjoint_evidence = row["adjoint_evidence"]
        if not isinstance(adjoint_evidence, dict):
            raise TimelineRunnerError("adjoint evidence is not a mapping")
        if adjoint_evidence["route"] != DIRECT_ADJOINT_ROUTE:
            raise TimelineRunnerError("evaluation used a non-direct adjoint route")
        evidence_values = np.asarray(
            [
                *residual_trace,
                *adjoint_evidence["output"],
                adjoint_evidence["residual"],
                adjoint_evidence["residual_relative"],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(evidence_values)):
            raise ScientificTimelineError(
                "Newton or adjoint evidence contains non-finite values",
                str(row["evaluation_id"]),
            )
    accepted = [row for row in trials if row["accepted"] is True]
    if [row["iteration"] for row in accepted] != list(
        range(1, REQUIRED_ACCEPTED_ITERATIONS + 1)
    ):
        raise ScientificTimelineError(
            "accepted iteration IDs are not contiguous 1..7",
            str(trials[-1]["evaluation_id"]),
        )
    trial_hashes = [str(row["parameter_sha256"]) for row in trials]
    if len(trial_hashes) != len(set(trial_hashes)):
        raise ScientificTimelineError(
            "optimizer trial parameter hashes are not distinct",
            str(trials[-1]["evaluation_id"]),
        )
    if str(initial[0]["parameter_sha256"]) in trial_hashes:
        raise ScientificTimelineError(
            "an optimizer trial repeats the initial parameter state",
            str(
                next(
                    row["evaluation_id"]
                    for row in trials
                    if row["parameter_sha256"] == initial[0]["parameter_sha256"]
                )
            ),
        )
    if final[0]["parameter_sha256"] != accepted[-1]["parameter_sha256"]:
        raise ScientificTimelineError(
            "final reporting is not bound to the final incumbent",
            str(final[0]["evaluation_id"]),
        )
    failing_rows = [
        row
        for row in evaluations
        if not bool(row["values_finite"])
        or not bool(row["inner_success"])
        or not bool(row["adjoint_success"])
        or not bool(row["candidate_gradient_source"])
        or not bool(row["eligible"])
    ]
    if failing_rows:
        raise ScientificTimelineError(
            "evaluation failed FP64, inner, adjoint, or candidate-gradient evidence",
            str(failing_rows[0]["evaluation_id"]),
        )


def _find_trace_file(trace_root: Path) -> Path:
    matches = tuple(sorted(trace_root.rglob("*.trace.json.gz")))
    if len(matches) != 1:
        raise TimelineRunnerError(
            f"expected exactly one JAX Chrome trace, found {len(matches)}"
        )
    return matches[0]


def _validate_raw_trajectory(
    trajectory_path: Path, evaluations: Sequence[Mapping[str, object]]
) -> None:
    records = tuple(
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    )
    accepted = tuple(row for row in evaluations if row["accepted"] is True)
    if len(records) != REQUIRED_ACCEPTED_ITERATIONS or len(accepted) != len(records):
        raise ScientificTimelineError(
            "raw trajectory does not contain seven accepted iterations",
            str(evaluations[-1]["evaluation_id"]),
        )
    for iteration, (point, evaluation) in enumerate(
        zip(records, accepted, strict=True), start=1
    ):
        if not isinstance(point, dict) or point.get("iteration") != iteration:
            raise ScientificTimelineError(
                "raw trajectory iteration sequence is invalid",
                str(evaluation["evaluation_id"]),
            )
        if not np.isclose(
            float(point["objective"]),
            float(evaluation["objective"]),
            rtol=2e-8,
            atol=2e-12,
        ):
            raise ScientificTimelineError(
                "raw trajectory objective differs from accepted evaluation",
                str(evaluation["evaluation_id"]),
            )


def _child_provenance_document(provenance) -> dict[str, object]:
    return {
        "repository_commit": provenance.repository_commit,
        "repository_dirty": provenance.repository_dirty,
        "tracked_diff_sha256": provenance.tracked_diff_sha256,
        "untracked_files": list(provenance.untracked_files),
        "executed_sources": [asdict(item) for item in provenance.executed_sources],
        "python_version": provenance.python_version,
        "jax_version": provenance.jax_version,
        "lane_environment_policy": dict(provenance.lane_environment_policy),
        "jax_effective_transfer_guards": dict(provenance.jax_effective_transfer_guards),
        "devices": [asdict(item) for item in provenance.devices],
        "simsoptpp_path": provenance.simsoptpp_path,
        "simsoptpp_sha256": provenance.simsoptpp_sha256,
        "simsoptpp_build_commit": provenance.simsoptpp_build_commit,
        "authoritative": provenance.authoritative,
    }


def _require_runtime_provenance(provenance, *, context: str) -> None:
    if not provenance.executed_sources:
        raise TimelineRunnerError(f"{context} provenance omits executed sources")
    for field_name in ("simsoptpp_path", "simsoptpp_sha256"):
        value = getattr(provenance, field_name)
        if not isinstance(value, str) or not value:
            raise TimelineRunnerError(f"{context} provenance omits {field_name}")
    build_commit = provenance.simsoptpp_build_commit
    if build_commit is not None and (
        not isinstance(build_commit, str) or not build_commit
    ):
        raise TimelineRunnerError(
            f"{context} provenance has an invalid simsoptpp_build_commit"
        )
    if provenance.authoritative and build_commit is None:
        raise TimelineRunnerError(
            f"{context} authoritative provenance omits simsoptpp_build_commit"
        )


def _provenance_document(
    spec: ChildSpec,
    provenance,
    *,
    collection_scope: Literal[
        "child_preexecution",
        "child_postexecution",
        "parent_prelaunch_after_child_termination",
    ],
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_id": "single-stage-changed-state-gpu-timeline-provenance-v1",
        "child_id": spec.child_id,
        "collection_scope": collection_scope,
        "source_state_sha256": spec.source_state_sha256,
        "environment": dict(spec.environment),
        "device": {"name": spec.device_name, "uuid": spec.device_uuid},
        "runtime": {
            "python_version": platform.python_version(),
            "jax_version": importlib.metadata.version("jax"),
            "jaxlib_version": importlib.metadata.version("jaxlib"),
        },
        "profiler_policy": _profiler_policy_document(spec.profiler_policy),
    }
    document.update(_child_provenance_document(provenance))
    return document


def _child_execute(spec: ChildSpec) -> int:
    output_root = Path(spec.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    trace_root = Path(spec.trace_root)
    if spec.mode == "profiled":
        trace_root.mkdir(parents=True, exist_ok=False)
    started_ns = time.perf_counter_ns()
    timeline_records: list[ChangedStateTimelineRecord] = []
    serialized_evaluations: tuple[dict[str, object], ...] = ()
    boundary_audit = None
    audit = None
    segment_trace_paths: dict[int, Path] = {}
    provenance = collect_lane_provenance(
        _REPO_ROOT,
        measurement_synchronization="child preexecution exact-byte snapshot",
    )
    _require_runtime_provenance(provenance, context="child preexecution")
    try:
        import jax

        from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
            TRACE_SCHEMA_ID,
            TraceSummaryError,
            summarize_segmented_trace,
        )

        _assert_source_state(spec.source_state_sha256)
        if spec.profiler_policy != _profiler_policy(spec.mode):
            raise TimelineRunnerError(
                "child profiler policy differs from its execution mode"
            )
        if _claimed_environment(os.environ) != dict(spec.environment):
            raise TimelineRunnerError("child environment differs from parent policy")
        bundle, arrays = read_input_bundle(Path(spec.input_root))
        if bundle.input_fingerprint != spec.input_sha256:
            raise TimelineRunnerError("child input fingerprint differs from parent")
        warm_measurement = MeasurementExecution(
            trajectory_path=output_root / "warm_trajectory.jsonl",
            optimization_timing_path=output_root / "warm_timing.json",
        )
        prepared_execution = _prepare_jax_variant_execution(
            "jax-gpu", bundle, arrays, SPEC, warm_measurement
        )
        compilation_identity = prepared_execution.compilation_identity
        if spec.mode == "profiled":
            with trace_session(), changed_state_timeline_observation_sink(
                lambda _row: None
            ):
                warm_observation = prepared_execution.execute(warm_measurement)
        else:
            with changed_state_timeline_observation_sink(lambda _row: None):
                warm_observation = prepared_execution.execute(warm_measurement)
        jax.block_until_ready(tuple(warm_observation.values.values()))
        if prepared_execution.compilation_identity != compilation_identity:
            raise TimelineRunnerError(
                "warm execution changed compiled runtime identity"
            )

        measured = MeasurementExecution(
            trajectory_path=output_root / "trajectory.jsonl",
            optimization_timing_path=output_root / "timing.json",
        )
        if spec.mode == "profiled":

            def start_segment(iteration_id: int) -> None:
                segment_root = trace_root / f"iteration-{iteration_id:02d}"
                segment_root.mkdir()
                try:
                    jax.profiler.start_trace(
                        str(segment_root),
                        profiler_options=_jax_profiler_options(
                            jax, spec.profiler_policy
                        ),
                    )
                except RuntimeError as error:
                    raise TraceCollectionError(
                        f"iteration {iteration_id} profiler start failed: {error}"
                    ) from error

            def stop_segment(iteration_id: int) -> None:
                try:
                    jax.profiler.stop_trace()
                except RuntimeError as error:
                    raise TraceCollectionError(
                        f"iteration {iteration_id} profiler stop/export failed: {error}"
                    ) from error
                try:
                    segment_trace_paths[iteration_id] = _find_trace_file(
                        trace_root / f"iteration-{iteration_id:02d}"
                    )
                except TimelineRunnerError as error:
                    raise TraceCollectionError(
                        f"iteration {iteration_id} profiler export failed: {error}"
                    ) from error

            def execute_measured():
                with changed_state_timeline_observation_sink(timeline_records.append):
                    result = prepared_execution.execute(measured)
                jax.block_until_ready(tuple(result.values.values()))
                return result

            observation, audit, boundary_audit = (
                _execute_segmented_profiled_measurement(
                    execute_measured,
                    start_segment=start_segment,
                    stop_segment=stop_segment,
                )
            )
        else:
            with changed_state_timeline_observation_sink(timeline_records.append):
                observation = prepared_execution.execute(measured)
            jax.block_until_ready(tuple(observation.values.values()))
        if prepared_execution.compilation_identity != compilation_identity:
            raise TimelineRunnerError(
                "measured execution changed compiled runtime identity"
            )
        timing = read_optimization_window_timing(output_root / "timing.json")
        optimizer_raw_wall_ns = round(timing.wall_seconds * 1_000_000_000)
        serialized_evaluations = _evaluation_documents(timeline_records)
        _validate_evaluations(serialized_evaluations)
        try:
            _validate_raw_trajectory(
                output_root / "trajectory.jsonl", serialized_evaluations
            )
        except ScientificTimelineError as error:
            serialized_evaluations = tuple(
                {
                    **row,
                    "trajectory_valid": row["evaluation_id"] != error.evaluation_id,
                }
                for row in serialized_evaluations
            )
            raise
        evaluations = serialized_evaluations
        accepted = tuple(row for row in evaluations if row["accepted"] is True)
        if (
            observation.nit != REQUIRED_ACCEPTED_ITERATIONS
            or len(accepted) != REQUIRED_ACCEPTED_ITERATIONS
        ):
            raise ScientificTimelineError(
                "production route did not complete exactly seven accepted iterations",
                str(evaluations[-1]["evaluation_id"]),
            )
        if observation.driver != JAX_FAST_DRIVER_ID:
            raise TimelineRunnerError(
                f"runtime driver {observation.driver!r} is not production custom L-BFGS-B"
            )
        provenance = collect_lane_provenance(
            _REPO_ROOT,
            measurement_synchronization=(
                "jax.block_until_ready over changed-state timeline observation"
            ),
        )
        _require_runtime_provenance(provenance, context="child postexecution")
        _write_json_exclusive(
            output_root / _CLAIM_FILES["provenance"],
            _provenance_document(
                spec, provenance, collection_scope="child_postexecution"
            ),
        )
        observation_document = {
            "schema_id": OBSERVATION_SCHEMA_ID,
            "child_id": spec.child_id,
            "evaluations": list(evaluations),
        }
        if spec.mode == "profiled":
            if audit is None or boundary_audit is None:
                raise TimelineRunnerError(
                    "profiled child omitted its host or boundary trace audit"
                )
            boundary_records = boundary_audit.records()
            _validate_complete_boundary_records(boundary_records)
            if set(segment_trace_paths) != set(
                range(1, REQUIRED_ACCEPTED_ITERATIONS + 1)
            ):
                raise TraceCollectionError(
                    "profiled execution omitted a required segment export"
                )
            events = _serialize_host_evidence(
                spec.child_id, audit.events(), audit.records()
            )
            if not events:
                raise TimelineRunnerError("profiled child emitted no host lifecycle")
            for iteration_id in range(1, REQUIRED_ACCEPTED_ITERATIONS + 1):
                sample_id = f"iteration-{iteration_id:02d}"
                try:
                    summary = summarize_segmented_trace(
                        segment_trace_paths[iteration_id],
                        audit.events(),
                        child_id=spec.child_id,
                        sample_id=sample_id,
                        accepted_iteration=iteration_id,
                        profiler_boundary_pauses=_iteration_pause_intervals(
                            boundary_records, audit.records(), iteration_id
                        ),
                        evaluation_documents=evaluations,
                    )
                except TraceSummaryError as error:
                    raise TraceCollectionError(
                        f"{sample_id} trace summary failed: {error}"
                    ) from error
                if summary.trace_schema_id != spec.trace_schema_id:
                    raise TimelineRunnerError(
                        f"{sample_id} raw trace schema differs from child contract"
                    )
                segment_output = output_root / _SEGMENTS_DIRECTORY / sample_id
                _write_bytes_exclusive(
                    segment_output / "trace.json.gz",
                    segment_trace_paths[iteration_id].read_bytes(),
                )
                _write_json_exclusive(
                    segment_output / "trace_summary.json", summary.to_json()
                )
        else:
            events = ()
            boundary_records = ()
        boundary_pause_records = _boundary_pause_documents(boundary_records)
        profiler_boundary_pause_total_ns = sum(
            int(record["duration_ns"]) for record in boundary_pause_records
        )
        optimizer_active_wall_ns = (
            optimizer_raw_wall_ns - profiler_boundary_pause_total_ns
        )
        if optimizer_active_wall_ns <= 0:
            raise TimelineRunnerError(
                "profiler boundary pauses consume the optimizer measurement window"
            )
        child_document = {
            **_child_identity(spec),
            "state": "complete",
            "failure_class": None,
            "failure_reason": None,
            "optimizer_raw_wall_ns": optimizer_raw_wall_ns,
            "profiler_boundary_pause_total_ns": profiler_boundary_pause_total_ns,
            "optimizer_active_wall_ns": optimizer_active_wall_ns,
            "child_end_to_end_ns": time.perf_counter_ns() - started_ns,
            "boundary_pause_records": list(boundary_pause_records),
            "nit": observation.nit,
            "nfev": observation.nfev,
            "njev": observation.njev,
            "status": observation.raw_status,
            "line_search_decisions": _line_search_decisions(evaluations),
            "endpoint_certificate": _endpoint_certificate(observation),
            "phase_ids": list(spec.phase_ids),
            "trace_schema_id": TRACE_SCHEMA_ID,
            "final_parameters": evaluations[-1]["parameters"],
            "final_parameters_sha256": evaluations[-1]["parameter_sha256"],
            "provenance": _child_provenance_document(provenance),
        }
        _write_json_exclusive(
            output_root / _CLAIM_FILES["child_metadata"], child_document
        )
        _write_json_exclusive(
            output_root / _CLAIM_FILES["numerical_observations"],
            observation_document,
        )
        _write_jsonl_exclusive(output_root / _CLAIM_FILES["host_device_events"], events)
        return 0
    # The child process is the protocol's failure-serialization boundary.
    except Exception as error:  # noqa: BLE001
        child_end_to_end_ns = time.perf_counter_ns() - started_ns
        failure_reason = f"{type(error).__name__}: {error}"
        if isinstance(error, ScientificTimelineError):
            failure_class: Literal["scientific", "trace", "integrity"] = "scientific"
            first_failed_evaluation_id = error.evaluation_id
        elif isinstance(error, TraceCollectionError):
            failure_class = "trace"
            first_failed_evaluation_id = None
        else:
            failure_class = "integrity"
            first_failed_evaluation_id = None
        partial_evaluations = serialized_evaluations or _evaluation_documents(
            timeline_records, require_dispositions=False
        )
        boundary_pause_records = _boundary_pause_documents(
            boundary_audit.records() if boundary_audit is not None else ()
        )
        child, observations, events = _failure_documents(
            spec,
            failure_class=failure_class,
            failure_reason=failure_reason,
            child_end_to_end_ns=child_end_to_end_ns,
            provenance=provenance,
            evaluations=partial_evaluations,
            first_failed_evaluation_id=first_failed_evaluation_id,
            boundary_pause_records=boundary_pause_records,
        )
        if spec.mode == "profiled" and audit is not None:
            events = (
                *_serialize_host_evidence(
                    spec.child_id, audit.events(), audit.records()
                ),
                {**events[0], "sequence": len(audit.events()) + len(audit.records())},
            )
        for role, document in (
            ("child_metadata", child),
            ("numerical_observations", observations),
        ):
            path = output_root / _CLAIM_FILES[role]
            if not path.exists():
                _write_json_exclusive(path, document)
        events_path = output_root / _CLAIM_FILES["host_device_events"]
        if not events_path.exists():
            _write_jsonl_exclusive(
                events_path, events if spec.mode == "profiled" else ()
            )
        timing_path = output_root / _CLAIM_FILES["optimization_timing"]
        timing_path.unlink(missing_ok=True)
        _write_json_exclusive(
            timing_path,
            {
                "schema_id": (
                    "single-stage-changed-state-gpu-timeline-timing-unavailable-v1"
                ),
                "failure_reason": failure_reason,
            },
        )
        trajectory_path = output_root / _CLAIM_FILES["trajectory"]
        trajectory_path.unlink(missing_ok=True)
        _write_bytes_exclusive(trajectory_path, b"")
        provenance_path = output_root / _CLAIM_FILES["provenance"]
        if not provenance_path.exists():
            _write_json_exclusive(
                provenance_path,
                _provenance_document(
                    spec, provenance, collection_scope="child_preexecution"
                ),
            )
        return 0


def _frozen_r5_check(repo_root: Path) -> None:
    program = (
        "from pathlib import Path; "
        "from benchmarks.validate_single_stage_speed_claim import check_frozen_files; "
        "check_frozen_files(Path.cwd(), 'campaign-20260804-frozen-r5')"
    )
    completed = subprocess.run(
        (sys.executable, "-c", program),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise TimelineRunnerError(
            "frozen r5 byte check failed: " + completed.stderr.strip()
        )


def _bounded_input(workspace: Path) -> tuple[InputBundle, dict[str, np.ndarray]]:
    from examples.jax.parity.cases.native_boozerqa import create_variant_input

    bounded_spec = dataclasses.replace(
        SPEC, native_outer_maxiter=REQUIRED_ACCEPTED_ITERATIONS
    )
    input_root = workspace / "inputs"
    bundle = create_variant_input(input_root, "native_default", bounded_spec)
    loaded, arrays = read_input_bundle(input_root)
    if loaded != bundle:
        raise TimelineRunnerError("persisted bounded input differs from its contract")
    return bundle, arrays


def _construction_sha256(bundle: InputBundle, arrays: Mapping[str, np.ndarray]) -> str:
    return _sha256_bytes(
        canonical_json_bytes(_construction_fingerprint_payload(bundle, arrays))
    )


def _input_fingerprint_payload(bundle: InputBundle) -> dict[str, object]:
    return {
        "case_id": bundle.case_id,
        "scale": bundle.scale,
        "random_seed": bundle.random_seed,
        "configuration_fingerprint": bundle.configuration_fingerprint,
        "arrays": {
            name: asdict(reference) for name, reference in sorted(bundle.arrays.items())
        },
    }


def _construction_fingerprint_payload(
    bundle: InputBundle, arrays: Mapping[str, np.ndarray]
) -> dict[str, object]:
    return {
        "case_id": bundle.case_id,
        "scale": bundle.scale,
        "random_seed": bundle.random_seed,
        "applied_construction": {
            "surface_dofs": _array_sha256(arrays["surface_dofs"]),
            "coil_dofs": _array_sha256(arrays["coil_dofs"]),
            **bundle.configuration,
        },
    }


def _identity_preimages_document(
    bundle: InputBundle,
    arrays: Mapping[str, np.ndarray],
    runtime_policy_payload: Mapping[str, object],
    parent_provenance,
    source_preimages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_id": "single-stage-changed-state-gpu-timeline-identity-preimages-v1",
        "input_fingerprint_payload": _input_fingerprint_payload(bundle),
        "configuration": dict(bundle.configuration),
        "construction_fingerprint_payload": _construction_fingerprint_payload(
            bundle, arrays
        ),
        "runtime_policy_payload": dict(runtime_policy_payload),
        "source_preimages": [dict(entry) for entry in source_preimages],
        "simsoptpp": {
            "path": parent_provenance.simsoptpp_path,
            "sha256": parent_provenance.simsoptpp_sha256,
            "build_commit": parent_provenance.simsoptpp_build_commit,
        },
    }


def _validate_preflight_evidence(
    evidence: Mapping[str, object],
    *,
    trace_schema_id: str,
    device_name: str,
    device_uuid: str,
) -> None:
    try:
        validate_passing_preflight_evidence(
            evidence,
            trace_schema_id=trace_schema_id,
            device_identity=DeviceIdentity(name=device_name, uuid=device_uuid),
        )
    except PreflightEvidenceError as error:
        raise TimelineRunnerError(
            f"trace preflight evidence invalid: {error}"
        ) from error


def _source_preimages(
    schedule: Sequence[ChildScheduleEntry],
    workspace: Path,
    parent_provenance,
) -> tuple[tuple[dict[str, str], ...], dict[str, Path]]:
    sources: dict[str, tuple[str, Path]] = {}

    def register(original_path: str, expected_sha256: str, source_path: Path) -> None:
        resolved = source_path.resolve()
        if source_path.is_symlink() or not resolved.is_file():
            raise TimelineRunnerError(
                f"source preimage is not a regular non-symlink file: {original_path}"
            )
        observed_sha256 = _sha256_path(resolved)
        if observed_sha256 != expected_sha256:
            raise TimelineRunnerError(
                f"source preimage changed before publication: {original_path}"
            )
        previous = sources.get(original_path)
        value = (expected_sha256, resolved)
        if previous is not None and previous != value:
            raise TimelineRunnerError(
                f"source provenance disagrees across processes: {original_path}"
            )
        sources[original_path] = value

    def register_provenance(provenance: Mapping[str, object]) -> None:
        raw_sources = provenance.get("executed_sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise TimelineRunnerError(
                "lane provenance omits executed source identities"
            )
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                raise TimelineRunnerError("executed source identity is not an object")
            original_path = str(raw_source.get("path", ""))
            expected_sha256 = str(raw_source.get("sha256", ""))
            relative = Path(original_path)
            if (
                not original_path
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != original_path
            ):
                raise TimelineRunnerError(
                    f"unsafe executed source provenance path: {original_path!r}"
                )
            register(original_path, expected_sha256, _REPO_ROOT / relative)

        simsoptpp_path = provenance.get("simsoptpp_path")
        simsoptpp_sha256 = provenance.get("simsoptpp_sha256")
        if not isinstance(simsoptpp_path, str) or not isinstance(simsoptpp_sha256, str):
            raise TimelineRunnerError("lane provenance omits simsoptpp identity")
        register(simsoptpp_path, simsoptpp_sha256, Path(simsoptpp_path))

    register_provenance(
        {
            **_child_provenance_document(parent_provenance),
            "executed_sources": [
                asdict(source) for source in parent_provenance.executed_sources
            ],
        }
    )
    for entry in schedule:
        register_provenance(
            _load_canonical_json(
                workspace / "children" / entry.child_id / "provenance.json"
            )
        )

    blobs: dict[str, Path] = {}
    entries: list[dict[str, str]] = []
    for original_path, (sha256, source_path) in sorted(sources.items()):
        manifest_path = f"source_preimages/{sha256}"
        blobs.setdefault(sha256, source_path)
        entries.append(
            {
                "original_path": original_path,
                "manifest_path": manifest_path,
                "sha256": sha256,
                "blob_id": sha256,
            }
        )
    return tuple(entries), blobs


def _child_spec(
    entry: ChildScheduleEntry,
    *,
    workspace: Path,
    source_state_sha256: str,
    environment_sha256: str,
    input_sha256: str,
    configuration_sha256: str,
    construction_sha256: str,
    runtime_policy_sha256: str,
    initial_parameters_sha256: str,
    device_name: str,
    device_uuid: str,
    environment: Mapping[str, str],
    phase_ids: tuple[str, ...],
    trace_schema_id: str,
) -> ChildSpec:
    cache_root = workspace / "caches" / entry.child_id
    output_root = workspace / "children" / entry.child_id
    trace_root = workspace / "traces" / entry.child_id
    cache_sha256 = _sha256_bytes(
        canonical_json_bytes(
            {"child_id": entry.child_id, "cache_root": str(cache_root.resolve())}
        )
    )
    return ChildSpec(
        child_id=entry.child_id,
        mode=entry.mode,
        pair_index=entry.pair_index,
        order_index=entry.order_index,
        input_root=str((workspace / "inputs").resolve()),
        output_root=str(output_root.resolve()),
        trace_root=str(trace_root.resolve()),
        cache_sha256=cache_sha256,
        source_state_sha256=source_state_sha256,
        environment_sha256=environment_sha256,
        input_sha256=input_sha256,
        configuration_sha256=configuration_sha256,
        construction_sha256=construction_sha256,
        runtime_policy_sha256=runtime_policy_sha256,
        initial_parameters_sha256=initial_parameters_sha256,
        device_name=device_name,
        device_uuid=device_uuid,
        environment=tuple(sorted(environment.items())),
        phase_ids=phase_ids,
        trace_schema_id=trace_schema_id,
        profiler_policy=_profiler_policy(entry.mode),
    )


def _parent_failure_evidence(
    spec: ChildSpec, reason: str, wall_ns: int, parent_provenance
) -> None:
    output_root = Path(spec.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    child, observations, events = _failure_documents(
        spec,
        failure_class="trace",
        failure_reason=reason,
        child_end_to_end_ns=wall_ns,
        provenance=parent_provenance,
    )
    for path, document in (
        (output_root / "child.json", child),
        (output_root / "observations.json", observations),
    ):
        if not path.exists():
            _write_json_exclusive(path, document)
    events_path = output_root / "events.jsonl"
    if not events_path.exists():
        _write_jsonl_exclusive(events_path, events if spec.mode == "profiled" else ())
    timing_path = output_root / "timing.json"
    timing_path.unlink(missing_ok=True)
    _write_json_exclusive(
        timing_path,
        {
            "schema_id": (
                "single-stage-changed-state-gpu-timeline-timing-unavailable-v1"
            ),
            "failure_reason": reason,
        },
    )
    trajectory_path = output_root / "trajectory.jsonl"
    trajectory_path.unlink(missing_ok=True)
    _write_bytes_exclusive(trajectory_path, b"")
    provenance_path = output_root / "provenance.json"
    if not provenance_path.exists():
        _write_json_exclusive(
            provenance_path,
            _provenance_document(
                spec,
                parent_provenance,
                collection_scope="parent_prelaunch_after_child_termination",
            ),
        )


def _claim_files(
    schedule: Sequence[ChildScheduleEntry],
    workspace: Path,
    source_state_sha256: str,
    bundle: InputBundle,
    source_blob_paths: Mapping[str, Path],
) -> tuple[ClaimFile, ...]:
    files: list[ClaimFile] = []
    empty_evaluation_digest = evaluation_ids_sha256(())
    files.extend(
        (
            ClaimFile(
                role="identity_preimages",
                relative_path="identity_preimages.json",
                source_path=workspace / "identity_preimages.json",
                source_state_sha256=source_state_sha256,
                process_id="artifact",
                evaluation_ids_sha256=empty_evaluation_digest,
            ),
            ClaimFile(
                role="preflight_evidence",
                relative_path="preflight/preflight.json",
                source_path=workspace / "preflight" / "preflight.json",
                source_state_sha256=source_state_sha256,
                process_id="artifact",
                evaluation_ids_sha256=empty_evaluation_digest,
            ),
            ClaimFile(
                role="input_evidence",
                relative_path="inputs/input_bundle.json",
                source_path=workspace / "inputs" / "input_bundle.json",
                source_state_sha256=source_state_sha256,
                process_id="artifact",
                evaluation_ids_sha256=empty_evaluation_digest,
            ),
        )
    )
    for reference in bundle.arrays.values():
        files.append(
            ClaimFile(
                role="input_evidence",
                relative_path=f"inputs/{reference.path}",
                source_path=workspace / "inputs" / reference.path,
                source_state_sha256=source_state_sha256,
                process_id="artifact",
                evaluation_ids_sha256=empty_evaluation_digest,
            )
        )
    for blob_id, source_path in sorted(source_blob_paths.items()):
        files.append(
            ClaimFile(
                role="source_evidence",
                relative_path=f"source_preimages/{blob_id}",
                source_path=source_path,
                source_state_sha256=source_state_sha256,
                process_id="artifact",
                evaluation_ids_sha256=empty_evaluation_digest,
            )
        )
    for entry in schedule:
        child_root = workspace / "children" / entry.child_id
        observation_document = _load_canonical_json(child_root / "observations.json")
        raw_evaluations = observation_document.get("evaluations")
        if not isinstance(raw_evaluations, list):
            raise TimelineRunnerError(
                f"{entry.child_id}: observations omit evaluation records"
            )
        evaluation_ids = tuple(
            str(row["evaluation_id"])
            for row in raw_evaluations
            if isinstance(row, dict) and "evaluation_id" in row
        )
        if len(evaluation_ids) != len(raw_evaluations):
            raise TimelineRunnerError(
                f"{entry.child_id}: an observation omits evaluation_id"
            )
        evaluation_digest = evaluation_ids_sha256(evaluation_ids)
        child_document = _load_canonical_json(child_root / "child.json")
        roles = [
            "child_metadata",
            "host_device_events",
            "numerical_observations",
            "optimization_timing",
            "trajectory",
            "provenance",
        ]
        for role in roles:
            files.append(
                ClaimFile(
                    role=role,
                    relative_path=(f"children/{entry.child_id}/{_CLAIM_FILES[role]}"),
                    source_path=child_root / _CLAIM_FILES[role],
                    source_state_sha256=source_state_sha256,
                    process_id=entry.child_id,
                    evaluation_ids_sha256=evaluation_digest,
                    sample_id=entry.child_id,
                )
            )
        if entry.mode == "profiled" and child_document.get("state") == "complete":
            accepted = tuple(
                row
                for row in raw_evaluations
                if isinstance(row, dict) and row.get("accepted") is True
            )
            if len(accepted) != REQUIRED_ACCEPTED_ITERATIONS:
                raise TimelineRunnerError(
                    f"{entry.child_id}: complete profiled child lacks seven accepted evaluations"
                )
            for iteration_id, evaluation in enumerate(accepted, start=1):
                sample_id = f"iteration-{iteration_id:02d}"
                evaluation_id = str(evaluation["evaluation_id"])
                segment_evaluation_ids = _segment_evaluation_ids(
                    tuple(row for row in raw_evaluations if isinstance(row, dict)),
                    iteration_id,
                )
                segment_evaluation_digest = evaluation_ids_sha256(
                    segment_evaluation_ids
                )
                for role, filename in (
                    ("raw_trace", "trace.json.gz"),
                    ("trace_summary", "trace_summary.json"),
                ):
                    relative_path = (
                        f"children/{entry.child_id}/{_SEGMENTS_DIRECTORY}/"
                        f"{sample_id}/{filename}"
                    )
                    files.append(
                        ClaimFile(
                            role=role,
                            relative_path=relative_path,
                            source_path=child_root
                            / _SEGMENTS_DIRECTORY
                            / sample_id
                            / filename,
                            source_state_sha256=source_state_sha256,
                            process_id=entry.child_id,
                            evaluation_ids_sha256=evaluation_digest,
                            sample_id=sample_id,
                            evaluation_id=evaluation_id,
                            segment_evaluation_ids_sha256=(segment_evaluation_digest),
                        )
                    )
        for diagnostic_name in ("monitor.json", "stdout.log", "stderr.log"):
            files.append(
                ClaimFile(
                    role="diagnostic",
                    relative_path=f"children/{entry.child_id}/{diagnostic_name}",
                    source_path=(
                        workspace / "diagnostics" / entry.child_id / diagnostic_name
                    ),
                    source_state_sha256=source_state_sha256,
                    process_id=entry.child_id,
                    evaluation_ids_sha256=evaluation_digest,
                    sample_id=entry.child_id,
                )
            )
    return tuple(files)


def run_timeline_campaign(
    *,
    artifact_root: Path,
    python_executable: str,
    gpu_index: int,
    timeout_seconds: float,
    max_process_tree_rss_bytes: int,
    poll_interval_seconds: float,
    base_environment: Mapping[str, str] = os.environ,
) -> Path:
    """Execute six bounded children and atomically publish their raw evidence."""

    resolved_root = validate_artifact_root(artifact_root)
    if (
        timeout_seconds <= 0
        or max_process_tree_rss_bytes <= 0
        or poll_interval_seconds <= 0
    ):
        raise ValueError("timeout, RSS bound, and poll interval must be positive")
    _frozen_r5_check(_REPO_ROOT)
    gpu_preflight = _gpu_concurrent_use_preflight(gpu_index)
    phase_ids = _phase_ids()
    from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
        TRACE_SCHEMA_ID,
    )

    source_state_sha256 = _timeline_source_state_sha256(_REPO_ROOT)
    device_name, device_uuid, cuda_driver, cuda_runtime = _gpu_identity(gpu_index)
    resolved_root.parent.mkdir(parents=True, exist_ok=True)
    workspace = resolved_root.parent / (
        f".{resolved_root.name}.work-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"
    )
    workspace.mkdir()
    bundle, arrays = _bounded_input(workspace)
    parent_provenance = collect_lane_provenance(
        _REPO_ROOT,
        measurement_synchronization="parent prelaunch exact-byte preflight",
    )
    _require_runtime_provenance(parent_provenance, context="parent prelaunch")
    construction_sha256 = _construction_sha256(bundle, arrays)
    initial_parameters_sha256 = _array_sha256(arrays["coil_dofs"])
    environment = build_measurement_environment(
        "jax_gpu_fast",
        allocation_sensitive=False,
        base_environment=base_environment,
        gpu_index=gpu_index,
        repo_root=_REPO_ROOT,
    )
    environment[_EXACT_ADJOINT_ENVIRONMENT] = "1"
    environment[TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT] = str(TRACE_VIEWER_MAX_EVENTS)
    claimed_environment = _claimed_environment(environment)
    environment_sha256 = _sha256_bytes(canonical_json_bytes(claimed_environment))
    runtime_policy_payload = {
        "route": _route_document(),
        "accepted_iterations": REQUIRED_ACCEPTED_ITERATIONS,
        "profiler_policy": {
            mode: _profiler_policy_document(_profiler_policy(mode))
            for mode in ("profiled", "control")
        },
        "gpu_preflight": gpu_preflight,
        "timeout_seconds": timeout_seconds,
        "max_process_tree_rss_bytes": max_process_tree_rss_bytes,
        "poll_interval_seconds": poll_interval_seconds,
    }
    runtime_policy_sha256 = _sha256_bytes(canonical_json_bytes(runtime_policy_payload))
    preflight_root = workspace / "preflight"
    preflight_cache = workspace / "caches" / "preflight"
    preflight_cache.mkdir(parents=True)
    preflight_environment = dict(environment)
    preflight_environment["JAX_COMPILATION_CACHE_DIR"] = str(preflight_cache.resolve())
    preflight_environment["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    preflight_environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    preflight_logs = workspace / "logs"
    monitored_preflight = execute_monitored_command(
        command=(
            python_executable,
            str(
                _REPO_ROOT / "benchmarks/single_stage_changed_state_trace_preflight.py"
            ),
            "--output-root",
            str(preflight_root),
            "--device-name",
            device_name,
            "--device-uuid",
            device_uuid,
        ),
        environment=preflight_environment,
        cwd=_REPO_ROOT,
        stdout_path=preflight_logs / "preflight.stdout",
        stderr_path=preflight_logs / "preflight.stderr",
        device="gpu",
        gpu_index=gpu_index,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_process_tree_rss_bytes=max_process_tree_rss_bytes,
    )
    preflight_path = preflight_root / "preflight.json"
    if not preflight_path.is_file():
        preflight_root.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(
            preflight_path,
            {
                "schema_id": PREFLIGHT_SCHEMA_ID,
                "state": "failed",
                "trace_schema_id": TRACE_SCHEMA_ID,
                "required_scopes": [],
                "observed_evidence": [],
                "device_identity": {"name": device_name, "uuid": device_uuid},
                "profiler_policy": _profiler_policy_document(
                    _profiler_policy("profiled")
                ),
                "session_evidence": [],
                "failure_reason": (
                    "preflight process terminated before evidence publication: "
                    f"{monitored_preflight.termination}"
                ),
            },
        )
    preflight_evidence = _load_canonical_json(preflight_path)
    if monitored_preflight.returncode != 0:
        raise TimelineRunnerError(
            "trace-schema preflight process failed before production children: "
            f"{monitored_preflight.termination}"
        )
    _validate_preflight_evidence(
        preflight_evidence,
        trace_schema_id=TRACE_SCHEMA_ID,
        device_name=device_name,
        device_uuid=device_uuid,
    )
    _assert_source_state(source_state_sha256)
    schedule = child_schedule()
    for entry in schedule:
        spec = _child_spec(
            entry,
            workspace=workspace,
            source_state_sha256=source_state_sha256,
            environment_sha256=environment_sha256,
            input_sha256=bundle.input_fingerprint,
            configuration_sha256=bundle.configuration_fingerprint,
            construction_sha256=construction_sha256,
            runtime_policy_sha256=runtime_policy_sha256,
            initial_parameters_sha256=initial_parameters_sha256,
            device_name=device_name,
            device_uuid=device_uuid,
            environment=claimed_environment,
            phase_ids=phase_ids,
            trace_schema_id=TRACE_SCHEMA_ID,
        )
        cache_root = workspace / "caches" / entry.child_id
        cache_root.mkdir(parents=True)
        child_environment = dict(environment)
        child_environment["JAX_COMPILATION_CACHE_DIR"] = str(cache_root.resolve())
        child_environment["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
        child_environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        spec_root = workspace / "specs"
        spec_root.mkdir(exist_ok=True)
        spec_path = spec_root / f"{entry.child_id}.json"
        _write_json_exclusive(spec_path, _child_spec_document(spec))
        logs = workspace / "logs"
        stdout_path = logs / f"{entry.child_id}.stdout"
        stderr_path = logs / f"{entry.child_id}.stderr"
        monitored = execute_monitored_command(
            command=build_child_command(python_executable, spec_path),
            environment=child_environment,
            cwd=_REPO_ROOT,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            device="gpu",
            gpu_index=gpu_index,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
            max_process_tree_rss_bytes=max_process_tree_rss_bytes,
        )
        diagnostic_root = workspace / "diagnostics" / entry.child_id
        diagnostic_root.mkdir(parents=True)
        process_failure_reason = (
            None
            if monitored.returncode == 0
            else f"child process terminated: {monitored.termination}"
        )
        stderr_bytes = stderr_path.read_bytes()
        profiler_retention = _profiler_retention_document(entry.mode, stderr_bytes)
        _write_json_exclusive(
            diagnostic_root / "monitor.json",
            {
                **asdict(monitored),
                "timeout_seconds": timeout_seconds,
                "max_process_tree_rss_bytes": max_process_tree_rss_bytes,
                "poll_interval_seconds": poll_interval_seconds,
                "failure_reason": process_failure_reason,
                "profiler_retention": profiler_retention,
            },
        )
        _write_bytes_exclusive(diagnostic_root / "stdout.log", stdout_path.read_bytes())
        _write_bytes_exclusive(diagnostic_root / "stderr.log", stderr_bytes)
        if monitored.returncode != 0:
            _parent_failure_evidence(
                spec,
                str(process_failure_reason),
                round(monitored.wall_seconds * 1_000_000_000),
                parent_provenance,
            )
        _assert_source_state(source_state_sha256)
    _frozen_r5_check(_REPO_ROOT)

    source_preimages, source_blob_paths = _source_preimages(
        schedule, workspace, parent_provenance
    )
    _write_json_exclusive(
        workspace / "identity_preimages.json",
        _identity_preimages_document(
            bundle,
            arrays,
            runtime_policy_payload,
            parent_provenance,
            source_preimages,
        ),
    )

    trace_schema_ids = {
        str(
            _load_canonical_json(
                workspace / "children" / entry.child_id / "child.json"
            ).get("trace_schema_id", TRACE_SCHEMA_ID)
        )
        for entry in schedule
        if entry.mode == "profiled"
    }
    if len(trace_schema_ids) != 1:
        raise TimelineRunnerError("profile children disagree on trace schema identity")
    metadata = TimelineMetadata(
        artifact_id=ARTIFACT_SCHEMA_ID,
        created_utc=datetime.now(UTC).isoformat(),
        source_state_sha256=source_state_sha256,
        trace_schema_id=next(iter(trace_schema_ids)),
        phase_schema_version=TRACE_PHASE_SCHEMA_VERSION,
        phase_ids=phase_ids,
        hostname=platform.node(),
        device_name=device_name,
        device_uuid=device_uuid,
        python_version=platform.python_version(),
        jax_version=importlib.metadata.version("jax"),
        jaxlib_version=importlib.metadata.version("jaxlib"),
        cuda_runtime=cuda_runtime,
        cuda_driver=cuda_driver,
        cpu_identity=_cpu_model(),
        affinity=",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
        environment_sha256=environment_sha256,
        input_sha256=bundle.input_fingerprint,
        configuration_sha256=bundle.configuration_fingerprint,
        construction_sha256=construction_sha256,
        runtime_policy_sha256=runtime_policy_sha256,
        initial_parameters_sha256=initial_parameters_sha256,
        child_schedule=schedule,
        route=RouteIdentity(),
        authoritative=parent_provenance.authoritative,
    )
    return write_timeline_receipt(
        resolved_root,
        metadata,
        _claim_files(
            schedule,
            workspace,
            source_state_sha256,
            bundle,
            source_blob_paths,
        ),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=CASE_ID)
    parser.add_argument(
        "--scale", choices=("native_default",), default="native_default"
    )
    parser.add_argument("--accepted-iterations", type=int, default=7)
    parser.add_argument("--profile-children", type=int, default=3)
    parser.add_argument("--control-children", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--timeout-seconds", type=float, default=_DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--max-process-tree-rss-bytes",
        type=int,
        default=_DEFAULT_MAX_PROCESS_TREE_RSS_BYTES,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=_DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _argument_parser().parse_args(arguments)
    if options.child_spec is not None:
        return _child_execute(_read_child_spec(options.child_spec))
    if options.artifact_root is None:
        _argument_parser().error("--artifact-root is required")
    if options.case != CASE_ID:
        _argument_parser().error(f"--case must be {CASE_ID}")
    if options.accepted_iterations != REQUIRED_ACCEPTED_ITERATIONS:
        _argument_parser().error("--accepted-iterations must be 7")
    child_schedule(options.profile_children, options.control_children)
    artifact = run_timeline_campaign(
        artifact_root=options.artifact_root,
        python_executable=sys.executable,
        gpu_index=options.gpu_index,
        timeout_seconds=options.timeout_seconds,
        max_process_tree_rss_bytes=options.max_process_tree_rss_bytes,
        poll_interval_seconds=options.poll_interval_seconds,
    )
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CASE_ID",
    "ChildSpec",
    "TimelineRunnerError",
    "build_child_command",
    "child_schedule",
    "main",
    "run_timeline_campaign",
    "validate_artifact_root",
)
