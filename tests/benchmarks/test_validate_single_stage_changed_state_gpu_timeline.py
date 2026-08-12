"""Artifact, verdict, and tamper tests for changed-state timeline validation."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from benchmarks.single_stage_changed_state_gpu_timeline_receipt import (
    ARTIFACT_SCHEMA_ID,
    CHILD_SCHEMA_ID,
    EVENT_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    ChildScheduleEntry,
    ClaimFile,
    TimelineMetadata,
    canonical_json_bytes,
    evaluation_ids_sha256,
    write_timeline_receipt,
)
from benchmarks.single_stage_changed_state_trace_preflight import (
    DeviceIdentity,
    evaluate_trace_scope_survival,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    Interval,
    summarize_segmented_trace_document,
)
from benchmarks.validate_single_stage_changed_state_gpu_timeline import (
    EXPECTED_PHASE_IDS,
    EXPECTED_PHASE_SCHEMA_VERSION,
    EXPECTED_TRACE_SCHEMA_ID,
    _validate_observations,
    validate_and_publish,
)
from simsopt_jax.runtime.trace_annotations import (
    EvaluationKind,
    EvaluationTraceContext,
    HostEvent,
    HostEventRecord,
    PhaseId,
)


def _live_source_sha256() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for arguments in (
        ("rev-parse", "HEAD"),
        ("status", "--porcelain=v1", "-z"),
        ("diff", "--binary", "HEAD"),
    ):
        completed = subprocess.run(
            ("git", *arguments), cwd=repo_root, check=True, capture_output=True
        )
        digest.update(len(completed.stdout).to_bytes(8, "little"))
        digest.update(completed.stdout)
    untracked = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    for relative_bytes in sorted(
        path for path in untracked.stdout.split(b"\0") if path
    ):
        content = (repo_root / relative_bytes.decode("utf-8")).read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "little"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


SOURCE_SHA = _live_source_sha256()
TEST_ENVIRONMENT = {
    "JAX_ENABLE_X64": "true",
    "JAX_PLATFORM_NAME": "gpu",
    "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS": "67108864",
}
ENVIRONMENT_SHA = hashlib.sha256(
    canonical_json_bytes(dict(sorted(TEST_ENVIRONMENT.items())))
).hexdigest()
ROUTE = {
    "optimizer": "SIMSOPT_LBFGSB",
    "driver": "minimize_lbfgs_host_core",
    "line_search": "line_search_value_and_grad_host",
    "adjoint_route": "exact_jacobian_dense_fp64_lu",
}
TEST_CONFIGURATION = {"test_configuration": 1}
TEST_SURFACE_DOFS = np.asarray([1.0, 2.0], dtype=np.dtype("<f8"))
TEST_COIL_DOFS = np.asarray([3.0, 4.0], dtype=np.dtype("<f8"))


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(values),
        version=(2, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


TEST_ARRAY_BYTES = {
    "surface_dofs": _npy_bytes(TEST_SURFACE_DOFS),
    "coil_dofs": _npy_bytes(TEST_COIL_DOFS),
}
TEST_ARRAY_REFERENCES = {
    name: {
        "path": f"inputs/{name}.npy",
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "order": "C",
        "sha256": hashlib.sha256(TEST_ARRAY_BYTES[name]).hexdigest(),
    }
    for name, values in {
        "surface_dofs": TEST_SURFACE_DOFS,
        "coil_dofs": TEST_COIL_DOFS,
    }.items()
}
TEST_CONFIGURATION_SHA = hashlib.sha256(
    canonical_json_bytes(TEST_CONFIGURATION)
).hexdigest()
TEST_INPUT_PAYLOAD = {
    "case_id": "synthetic-native-default",
    "scale": "native_default",
    "random_seed": 0,
    "configuration_fingerprint": TEST_CONFIGURATION_SHA,
    "arrays": TEST_ARRAY_REFERENCES,
}
TEST_INPUT_SHA = hashlib.sha256(canonical_json_bytes(TEST_INPUT_PAYLOAD)).hexdigest()
TEST_CONSTRUCTION_PAYLOAD = {
    "case_id": "synthetic-native-default",
    "scale": "native_default",
    "random_seed": 0,
    "applied_construction": {
        "surface_dofs": hashlib.sha256(TEST_SURFACE_DOFS.tobytes()).hexdigest(),
        "coil_dofs": hashlib.sha256(TEST_COIL_DOFS.tobytes()).hexdigest(),
        **TEST_CONFIGURATION,
    },
}
TEST_CONSTRUCTION_SHA = hashlib.sha256(
    canonical_json_bytes(TEST_CONSTRUCTION_PAYLOAD)
).hexdigest()
TEST_RUNTIME_POLICY = {
    "route": ROUTE,
    "accepted_iterations": 7,
    "profiler_policy": {
        "profiled": {
            "enabled": True,
            "host_tracer_level": 1,
            "python_tracer_level": 0,
            "device_tracing": "jax_default",
            "trace_viewer_max_events": 67_108_864,
            "advanced_configuration": {
                "gpu_max_activity_api_events": 33_554_432,
                "gpu_max_callback_api_events": 33_554_432,
            },
        },
        "control": {
            "enabled": False,
            "host_tracer_level": None,
            "python_tracer_level": None,
            "device_tracing": None,
            "trace_viewer_max_events": None,
            "advanced_configuration": {},
        },
    },
    "gpu_preflight": "synthetic-pass",
    "timeout_seconds": 60.0,
    "max_process_tree_rss_bytes": 1_000_000,
    "poll_interval_seconds": 0.1,
}
TEST_RUNTIME_POLICY_SHA = hashlib.sha256(
    canonical_json_bytes(TEST_RUNTIME_POLICY)
).hexdigest()


def _producer_preflight_evidence(metadata: TimelineMetadata) -> dict[str, object]:
    def kernel_event(phase: PhaseId, timestamp_us: int) -> dict[str, object]:
        return {
            "ph": "X",
            "pid": 2,
            "tid": 7,
            "ts": timestamp_us,
            "dur": 1,
            "name": "canary_kernel",
            "args": {
                "context_id": "1",
                "correlation_id": str(timestamp_us),
                "hlo_module": "jit_canary",
                "hlo_op": f"jit_canary/{phase.value}/multiply",
                "kernel_details": (
                    "regs:16 static_shared:0 dynamic_shared:0 grid:1,1,1 block:32,1,1"
                ),
                "name": "canary_kernel",
                "scope_range_id": "1",
                "tf_op": f"jit_canary/{phase.value}/multiply",
            },
        }

    trace_document = {
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": True},
        "traceEvents": [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/host:CPU"},
            },
            {
                "ph": "M",
                "pid": 2,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            kernel_event(PhaseId.NEWTON_RESIDUAL_JVP, 10),
            kernel_event(PhaseId.ADJOINT_LU_SOLVE, 20),
            {},
        ],
    }
    evidence = {
        **evaluate_trace_scope_survival(
            trace_document,
            device_identity=DeviceIdentity(
                name=metadata.device_name, uuid=metadata.device_uuid
            ),
        ),
        "profiler_policy": TEST_RUNTIME_POLICY["profiler_policy"]["profiled"],
    }
    session_observed_evidence = evidence["observed_evidence"]
    evidence["session_evidence"] = [
        {
            "session_id": f"session-{index:02d}",
            "device_processes": ["/device:GPU:0"],
            "observed_evidence": session_observed_evidence,
        }
        for index in range(1, 3)
    ]
    evidence["observed_evidence"] = [
        {
            **observation,
            "device_kernel_intervals_containing_scope": 2
            * observation["device_kernel_intervals_containing_scope"],
            "uniquely_attributed_device_kernel_intervals": 2
            * observation["uniquely_attributed_device_kernel_intervals"],
            "ambiguous_device_kernel_intervals": 2
            * observation["ambiguous_device_kernel_intervals"],
        }
        for observation in session_observed_evidence
    ]
    return evidence


def _parameters(evaluation_index: int) -> list[float]:
    effective_index = 7 if evaluation_index == 8 else evaluation_index
    return [float(effective_index), float(effective_index) + 0.25]


def _parameter_sha256(evaluation_index: int) -> str:
    values = _parameters(evaluation_index)
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document))


def _write_jsonl(path: Path, documents: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(document) for document in documents))


def _schedule() -> tuple[ChildScheduleEntry, ...]:
    return tuple(
        ChildScheduleEntry(
            child_id=f"{mode}-{pair_index}",
            mode=mode,
            pair_index=pair_index,
            order_index=2 * pair_index + mode_index,
        )
        for pair_index in range(3)
        for mode_index, mode in enumerate(("profiled", "control"))
    )


def _metadata() -> TimelineMetadata:
    return TimelineMetadata(
        artifact_id=ARTIFACT_SCHEMA_ID,
        created_utc="2026-08-05T00:00:00+00:00",
        source_state_sha256=SOURCE_SHA,
        trace_schema_id=EXPECTED_TRACE_SCHEMA_ID,
        phase_schema_version=EXPECTED_PHASE_SCHEMA_VERSION,
        phase_ids=tuple(sorted(EXPECTED_PHASE_IDS)),
        hostname="test-host",
        device_name="synthetic GPU",
        device_uuid="GPU-test",
        python_version="3.12",
        jax_version="0.10.0",
        jaxlib_version="0.10.0",
        cuda_runtime="13.0",
        cuda_driver="580",
        cpu_identity="synthetic CPU",
        affinity="0-1",
        environment_sha256=ENVIRONMENT_SHA,
        input_sha256=TEST_INPUT_SHA,
        configuration_sha256=TEST_CONFIGURATION_SHA,
        construction_sha256=TEST_CONSTRUCTION_SHA,
        runtime_policy_sha256=TEST_RUNTIME_POLICY_SHA,
        initial_parameters_sha256=_parameter_sha256(0),
        child_schedule=_schedule(),
        authoritative=False,
    )


def _observations(child_id: str) -> dict[str, object]:
    evaluations: list[dict[str, object]] = []
    for evaluation_index in range(9):
        if evaluation_index == 0:
            lifecycle = "initial"
            accepted = None
            iteration = None
        elif evaluation_index == 8:
            lifecycle = "final_reporting"
            accepted = None
            iteration = None
        else:
            lifecycle = "trial"
            accepted = True
            iteration = evaluation_index
        evaluations.append(
            {
                "evaluation_id": f"evaluation-{evaluation_index}",
                "evaluation_index": evaluation_index,
                "lifecycle": lifecycle,
                "accepted": accepted,
                "iteration": iteration,
                "parameter_sha256": _parameter_sha256(evaluation_index),
                "parameters": _parameters(evaluation_index),
                "parameter_shape": [2],
                "objective": 10.0 - evaluation_index,
                "gradient": [0.5, -0.25],
                "gradient_source": "candidate",
                "dtype": "float64",
                "inner_success": True,
                "adjoint_success": True,
                "gradient_shape": [2],
                "values_finite": True,
                "candidate_gradient_source": True,
                "eligible": True,
                "trajectory_valid": True,
                "inner_evidence": {
                    "residual_trace": [1.0e-12],
                    "step_accepted_trace": [True],
                    "linear_solve_success_trace": [True],
                    "newton_iterations": 1,
                    "newton_attempted_iterations": 1,
                    "newton_trace_available": True,
                },
                "adjoint_evidence": {
                    "route": "exact_jacobian_dense_fp64_lu",
                    "output": [0.25, -0.5],
                    "residual": 1.0e-12,
                    "residual_relative": 1.0e-10,
                    "dense_materializations": 1,
                    "lu_factorizations": 1,
                    "lu_solves": 1,
                    "refinement_corrections": 0,
                    "adjoint_executions": 1,
                },
                "observables": (
                    {
                        "objective": 2.0,
                        "iota": -0.4,
                        "volume": 0.1,
                        "non_qs_ratio": 1.0e-6,
                        "boozer_residual": 1.0e-12,
                    }
                    if lifecycle == "final_reporting"
                    else None
                ),
            }
        )
    return {
        "schema_id": OBSERVATION_SCHEMA_ID,
        "child_id": child_id,
        "evaluations": evaluations,
    }


def _event(
    event_id: str,
    iteration: int,
    start_ns: int,
    end_ns: int,
    phase_id: str,
    *,
    kind: str,
    completion: bool = False,
    owners: list[str] | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_id": EVENT_SCHEMA_ID,
        "event_id": event_id,
        "iteration": iteration,
        "evaluation_id": f"evaluation-{iteration}",
        "parameter_sha256": f"{iteration + 1:x}" * 64,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "phase_id": phase_id,
        "kind": kind,
        "completion": completion,
    }
    if kind == "host_span":
        document["exclusive"] = True
        if phase_id == "host.line_search_control":
            document["previous_evaluation_id"] = f"evaluation-{iteration - 1}"
            document["next_evaluation_id"] = f"evaluation-{iteration}"
    else:
        document["owners"] = [phase_id] if owners is None else owners
    return document


def _events(
    profile_id: str,
    *,
    host_ns: int,
    newton_ns: int,
    other_ns: int,
    unattributed_ns: int = 0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    newton_phases = (
        "newton.residual_jvp",
        "newton.linear_solve",
        "adjoint.outer_vjp_rhs",
        "adjoint.dense_matrix",
        "adjoint.lu_factor",
        "adjoint.lu_solve",
        "adjoint.refinement",
        "adjoint.implicit_coil_vjp",
    )
    for iteration in range(1, 8):
        cursor = iteration * 10_000
        host_parts = (max(1, host_ns // 5), max(1, host_ns * 3 // 5))
        host_parts += (host_ns - sum(host_parts),)
        for phase, duration, completion in zip(
            (
                "host.h2d_submit",
                "host.line_search_control",
                "host.d2h_materialize",
            ),
            host_parts,
            (False, False, True),
            strict=True,
        ):
            records.append(
                _event(
                    f"{profile_id}-{iteration}-{len(records)}",
                    iteration,
                    cursor,
                    cursor + duration,
                    phase,
                    kind="host_span",
                    completion=completion,
                )
            )
            cursor += duration
        remaining = newton_ns
        for phase_index, phase in enumerate(newton_phases):
            phases_left = len(newton_phases) - phase_index
            duration = remaining // phases_left
            records.append(
                _event(
                    f"{profile_id}-{iteration}-{len(records)}",
                    iteration,
                    cursor,
                    cursor + duration,
                    phase,
                    kind="device_interval",
                )
            )
            cursor += duration
            remaining -= duration
        for phase, duration in (
            ("biotsavart.forward", other_ns // 2),
            ("biotsavart.vjp", other_ns - other_ns // 2),
        ):
            records.append(
                _event(
                    f"{profile_id}-{iteration}-{len(records)}",
                    iteration,
                    cursor,
                    cursor + duration,
                    phase,
                    kind="device_interval",
                )
            )
            cursor += duration
        if unattributed_ns:
            records.append(
                _event(
                    f"{profile_id}-{iteration}-{len(records)}",
                    iteration,
                    cursor,
                    cursor + unattributed_ns,
                    "optimizer.lifecycle",
                    kind="device_interval",
                    owners=[],
                )
            )
    return records


def _timeline_layout(
    *, host_gap_ns: int, device_ns: int, d2h_ns: int
) -> tuple[tuple[int, int, int], ...]:
    layout: list[tuple[int, int, int]] = [(100, 104, 108)]
    previous_return = 108
    for _ in range(7):
        entry = previous_return + host_gap_ns
        ready = entry + 1 + device_ns - d2h_ns
        returned = entry + 1 + device_ns + 1
        layout.append((entry, ready, returned))
        previous_return = returned
    final_entry = previous_return + 1
    layout.append((final_entry, final_entry + 4, final_entry + 8))
    return tuple(layout)


def _host_event_evidence(
    child_id: str, *, host_gap_ns: int, device_ns: int, d2h_ns: int
) -> tuple[list[dict[str, object]], tuple[HostEventRecord, ...]]:
    documents: list[dict[str, object]] = []
    records: list[HostEventRecord] = []
    sequence = 0
    layout = _timeline_layout(
        host_gap_ns=host_gap_ns, device_ns=device_ns, d2h_ns=d2h_ns
    )
    for evaluation_index, timestamps in enumerate(layout):
        if evaluation_index == 0:
            kind = EvaluationKind.INITIAL
            iteration = None
        elif evaluation_index == 8:
            kind = EvaluationKind.FINAL_REPORTING
            iteration = None
        else:
            kind = EvaluationKind.TRIAL
            iteration = evaluation_index
        parameter_sha256 = _parameter_sha256(evaluation_index)
        context = EvaluationTraceContext(
            evaluation_id=f"evaluation-{evaluation_index}",
            parameter_sha256=parameter_sha256,
            kind=kind,
            outer_iteration_id=iteration,
        )
        for event, timestamp_ns in zip(HostEvent, timestamps, strict=True):
            record = HostEventRecord(sequence, event, timestamp_ns, context, ())
            records.append(record)
            documents.append(
                {
                    "schema_id": EVENT_SCHEMA_ID,
                    "child_id": child_id,
                    "record_type": "lifecycle",
                    "sequence": sequence,
                    "event": event.value,
                    "timestamp_ns": timestamp_ns,
                    "evaluation_id": context.evaluation_id,
                    "parameter_sha256": context.parameter_sha256,
                    "evaluation_kind": context.kind.value,
                    "outer_iteration_id": context.outer_iteration_id,
                    "attributes": {},
                }
            )
            sequence += 1
    span_sequence = 0
    for evaluation_index, (entry, ready, returned) in enumerate(layout):
        kind = (
            "initial"
            if evaluation_index == 0
            else "final_reporting"
            if evaluation_index == 8
            else "trial"
        )
        attributes: dict[str, object] = {
            "evaluation_id": f"evaluation-{evaluation_index}",
            "evaluation_kind": kind,
            "parameter_sha256": _parameter_sha256(evaluation_index),
        }
        if kind == "trial":
            attributes["outer_iteration_id"] = evaluation_index
        for phase_id, start_ns, end_ns in (
            ("host.h2d_submit", entry + 1, entry + 2),
            ("host.d2h_materialize", ready, returned),
        ):
            documents.append(
                {
                    "schema_id": EVENT_SCHEMA_ID,
                    "child_id": child_id,
                    "record_type": "host_span",
                    "sequence": span_sequence,
                    "phase_id": phase_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "depth": 0,
                    "attributes": attributes,
                }
            )
            span_sequence += 1
        if 0 < evaluation_index < 8:
            previous_id = f"evaluation-{evaluation_index - 1}"
            documents.append(
                {
                    "schema_id": EVENT_SCHEMA_ID,
                    "child_id": child_id,
                    "record_type": "host_span",
                    "sequence": span_sequence,
                    "phase_id": "host.line_search_control",
                    "start_ns": layout[evaluation_index - 1][2],
                    "end_ns": entry,
                    "depth": 0,
                    "attributes": {
                        "previous_evaluation_id": previous_id,
                        "next_evaluation_id": f"evaluation-{evaluation_index}",
                        "outer_iteration_id": evaluation_index,
                    },
                }
            )
            span_sequence += 1
            documents.append(
                {
                    "schema_id": EVENT_SCHEMA_ID,
                    "child_id": child_id,
                    "record_type": "optimizer_span",
                    "sequence": span_sequence,
                    "phase_id": "optimizer.lifecycle",
                    "start_ns": returned - 1,
                    "end_ns": returned + 1,
                    "depth": 0,
                    "attributes": {"accepted_iteration_id": evaluation_index},
                }
            )
            span_sequence += 1
    return documents, tuple(records)


def _trace_document(
    *,
    host_ns: int,
    newton_ns: int,
    other_ns: int,
    unattributed_ns: int,
    target_iteration: int | None = None,
) -> dict[str, object]:
    trace_events: list[dict[str, object]] = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": 1,
            "args": {"name": "/host:CPU"},
        },
        {
            "ph": "M",
            "name": "process_name",
            "pid": 2,
            "args": {"name": "/device:GPU:0"},
        },
    ]
    transfer_ns = min(host_ns - 1, 10)
    h2d_ns = transfer_ns // 2
    d2h_ns = transfer_ns - h2d_ns
    device_ns = transfer_ns + newton_ns + other_ns + unattributed_ns
    _, lifecycle_records = _host_event_evidence(
        "trace-clock",
        host_gap_ns=host_ns - transfer_ns,
        device_ns=device_ns,
        d2h_ns=d2h_ns,
    )
    for record in lifecycle_records:
        if target_iteration is not None and (
            record.evaluation.kind is not EvaluationKind.TRIAL
            or record.evaluation.outer_iteration_id != target_iteration
        ):
            continue
        args = {
            "evaluation_id": record.evaluation.evaluation_id,
            "parameter_sha256": record.evaluation.parameter_sha256,
            "evaluation_kind": record.evaluation.kind.value,
        }
        if record.evaluation.outer_iteration_id is not None:
            args["outer_iteration_id"] = str(record.evaluation.outer_iteration_id)
        trace_events.append(
            {
                "ph": "X",
                "name": f"optimizer.lifecycle.{record.event.value}",
                "pid": 1,
                "tid": 1,
                "ts": (record.timestamp_ns + 10) / 1000,
                "dur": 0.001,
                "args": args,
            }
        )

    newton_phases = (
        "newton.residual_jvp",
        "newton.linear_solve",
        "adjoint.outer_vjp_rhs",
        "adjoint.dense_matrix",
        "adjoint.lu_factor",
        "adjoint.lu_solve",
        "adjoint.refinement",
        "adjoint.implicit_coil_vjp",
    )
    for iteration in range(1, 8):
        if target_iteration is not None and iteration != target_iteration:
            continue
        entry_ns = lifecycle_records[iteration * len(HostEvent)].timestamp_ns
        cursor = entry_ns + 11
        total_device_ns = device_ns
        trace_events.append(
            {
                "ph": "X",
                "name": "optimizer.accepted_iteration",
                "pid": 1,
                "tid": 1,
                "ts": (entry_ns + 10) / 1000,
                "dur": (total_device_ns + 3) / 1000,
                "args": {"step_num": str(iteration)},
            }
        )

        def append_device(
            phase: str,
            duration_ns: int,
            *,
            iteration_id: int = iteration,
            memcpy: bool = False,
        ) -> None:
            nonlocal cursor
            args: dict[str, object] = {"name": phase}
            if memcpy:
                trace_events.append(
                    {
                        "ph": "X",
                        "name": phase,
                        "pid": 1,
                        "tid": 1,
                        "ts": cursor / 1000,
                        "dur": duration_ns / 1000,
                        "args": {"synthetic_transfer": True},
                    }
                )
                args["context_id"] = "$$1"
                args["correlation_id"] = str(iteration_id)
                args["memcpy_details"] = (
                    "kind_src:Host kind_dst:Device size:8 dest:0 async:1"
                )
            else:
                args.update(
                    {
                        "context_id": "$$1",
                        "correlation_id": str(iteration_id),
                        "hlo_module": "jit_synthetic",
                        "hlo_op": phase,
                        "scope_range_id": "2",
                        "tf_op": "XlaModule:",
                    }
                )
                args["kernel_details"] = "synthetic"
            trace_events.append(
                {
                    "ph": "X",
                    "name": "synthetic device event",
                    "pid": 2,
                    "tid": 1,
                    "ts": cursor / 1000,
                    "dur": duration_ns / 1000,
                    "args": args,
                }
            )
            cursor += duration_ns

        append_device("host.h2d_submit", h2d_ns, memcpy=True)
        remaining = newton_ns
        for phase_index, phase in enumerate(newton_phases):
            phases_left = len(newton_phases) - phase_index
            duration = remaining // phases_left
            append_device(phase, duration)
            remaining -= duration
        append_device("optimizer.lifecycle/biotsavart.forward", other_ns // 2)
        append_device("optimizer.lifecycle/biotsavart.vjp", other_ns - other_ns // 2)
        if unattributed_ns:
            append_device("fusion_kernel", unattributed_ns)
        append_device("host.d2h_materialize", d2h_ns, memcpy=True)
    return {
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": True},
        "traceEvents": trace_events,
    }


def _child_document(
    metadata: TimelineMetadata,
    entry: ChildScheduleEntry,
    *,
    profile_wall_ns: int,
    control_wall_ns: int,
    state: str = "complete",
    failure_class: str | None = None,
    boundary_pause_records: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    boundary_pause_ns = sum(
        int(record["duration_ns"]) for record in boundary_pause_records
    )
    active_wall_ns = profile_wall_ns if entry.mode == "profiled" else control_wall_ns
    raw_wall_ns = active_wall_ns + boundary_pause_ns
    return {
        "schema_id": CHILD_SCHEMA_ID,
        "child_id": entry.child_id,
        "mode": entry.mode,
        "pair_index": entry.pair_index,
        "order_index": entry.order_index,
        "state": state,
        "failure_class": failure_class,
        "failure_reason": "synthetic failure" if failure_class else None,
        "first_failed_evaluation_id": (
            "evaluation-0" if failure_class == "scientific" else None
        ),
        "route": ROUTE,
        "source_state_sha256": metadata.source_state_sha256,
        "environment_sha256": metadata.environment_sha256,
        "input_sha256": metadata.input_sha256,
        "configuration_sha256": metadata.configuration_sha256,
        "construction_sha256": metadata.construction_sha256,
        "runtime_policy_sha256": metadata.runtime_policy_sha256,
        "initial_parameters_sha256": metadata.initial_parameters_sha256,
        "device_name": metadata.device_name,
        "device_uuid": metadata.device_uuid,
        "cache_sha256": f"{entry.order_index + 2:x}" * 64,
        "profiler_policy": TEST_RUNTIME_POLICY["profiler_policy"][entry.mode],
        "optimizer_raw_wall_ns": raw_wall_ns if state == "complete" else None,
        "profiler_boundary_pause_total_ns": (boundary_pause_ns),
        "optimizer_active_wall_ns": active_wall_ns if state == "complete" else None,
        "child_end_to_end_ns": raw_wall_ns,
        "boundary_pause_records": list(boundary_pause_records),
        "nit": 7,
        "nfev": 9,
        "njev": 9,
        "status": "iteration_limit",
        "line_search_decisions": [
            {
                "evaluation_id": f"evaluation-{iteration}",
                "accepted": True,
                "iteration": iteration,
            }
            for iteration in range(1, 8)
        ],
        "final_parameters": _parameters(8),
        "final_parameters_sha256": _parameter_sha256(8),
        "endpoint_certificate": {
            "success": False,
            "outer_status": 1,
            "constraints_satisfied": True,
        },
        "provenance": {"authoritative": True},
    }


def _build_artifact(
    tmp_path: Path,
    *,
    host_ns: int = 70,
    newton_ns: int = 20,
    other_ns: int = 10,
    unattributed_ns: int = 0,
    profile_wall_ns: int = 105,
    control_wall_ns: int = 100,
    failed_child: tuple[str, str] | None = None,
) -> Path:
    metadata = _metadata()
    inputs = tmp_path / "inputs"
    simsoptpp_spec = importlib.util.find_spec("simsoptpp")
    assert simsoptpp_spec is not None and simsoptpp_spec.origin is not None
    simsoptpp_path = Path(simsoptpp_spec.origin).resolve()
    simsoptpp_identity = {
        "path": str(simsoptpp_path),
        "sha256": hashlib.sha256(simsoptpp_path.read_bytes()).hexdigest(),
        "build_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    repo_root = Path(__file__).resolve().parents[2]
    test_source_relative = Path(__file__).resolve().relative_to(repo_root).as_posix()
    source_files = {
        test_source_relative: Path(__file__).resolve(),
        str(simsoptpp_path): simsoptpp_path,
    }
    source_preimages = []
    for original_path, source_path in sorted(source_files.items()):
        sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        source_preimages.append(
            {
                "original_path": original_path,
                "manifest_path": f"source_preimages/{sha256}",
                "sha256": sha256,
                "blob_id": sha256,
            }
        )
    identity_path = inputs / "identity_preimages.json"
    bundle_path = inputs / "input_bundle.json"
    preflight_path = inputs / "preflight.json"
    _write_json(
        identity_path,
        {
            "schema_id": (
                "single-stage-changed-state-gpu-timeline-identity-preimages-v1"
            ),
            "input_fingerprint_payload": TEST_INPUT_PAYLOAD,
            "configuration": TEST_CONFIGURATION,
            "construction_fingerprint_payload": TEST_CONSTRUCTION_PAYLOAD,
            "runtime_policy_payload": TEST_RUNTIME_POLICY,
            "source_preimages": source_preimages,
            "simsoptpp": simsoptpp_identity,
        },
    )
    _write_json(
        bundle_path,
        {
            "schema_version": 2,
            "case_id": TEST_INPUT_PAYLOAD["case_id"],
            "scale": TEST_INPUT_PAYLOAD["scale"],
            "random_seed": TEST_INPUT_PAYLOAD["random_seed"],
            "configuration": TEST_CONFIGURATION,
            "configuration_fingerprint": TEST_CONFIGURATION_SHA,
            "arrays": TEST_ARRAY_REFERENCES,
            "input_fingerprint": TEST_INPUT_SHA,
        },
    )
    _write_json(preflight_path, _producer_preflight_evidence(metadata))
    array_paths: list[Path] = []
    for name, raw_array in TEST_ARRAY_BYTES.items():
        path = inputs / "inputs" / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_array)
        array_paths.append(path)
    empty_evaluations = evaluation_ids_sha256(())
    claims: list[ClaimFile] = [
        ClaimFile(
            "identity_preimages",
            "identity_preimages.json",
            identity_path,
            SOURCE_SHA,
            "artifact",
            empty_evaluations,
        ),
        ClaimFile(
            "input_evidence",
            "inputs/input_bundle.json",
            bundle_path,
            SOURCE_SHA,
            "artifact",
            empty_evaluations,
        ),
        ClaimFile(
            "preflight_evidence",
            "preflight/preflight.json",
            preflight_path,
            SOURCE_SHA,
            "artifact",
            empty_evaluations,
        ),
        *[
            ClaimFile(
                "input_evidence",
                f"inputs/inputs/{path.stem}.npy",
                path,
                SOURCE_SHA,
                "artifact",
                empty_evaluations,
            )
            for path in array_paths
        ],
        *[
            ClaimFile(
                "source_evidence",
                source["manifest_path"],
                source_files[source["original_path"]],
                SOURCE_SHA,
                "artifact",
                empty_evaluations,
            )
            for source in source_preimages
        ],
    ]
    for entry in metadata.child_schedule:
        state = "complete"
        failure_class = None
        if failed_child is not None and entry.child_id == failed_child[0]:
            state = "failed"
            failure_class = failed_child[1]
        child_path = inputs / entry.child_id / "child.json"
        events_path = inputs / entry.child_id / "events.jsonl"
        observations_path = inputs / entry.child_id / "observations.json"
        timing_path = inputs / entry.child_id / "timing.json"
        trajectory_path = inputs / entry.child_id / "trajectory.jsonl"
        provenance_path = inputs / entry.child_id / "provenance.json"
        transfer_ns = min(host_ns - 1, 10)
        host_gap_ns = host_ns - transfer_ns
        d2h_ns = transfer_ns - transfer_ns // 2
        device_ns = transfer_ns + newton_ns + other_ns + unattributed_ns
        host_event_documents, host_event_records = _host_event_evidence(
            entry.child_id,
            host_gap_ns=host_gap_ns,
            device_ns=device_ns,
            d2h_ns=d2h_ns,
        )
        layout = _timeline_layout(
            host_gap_ns=host_gap_ns, device_ns=device_ns, d2h_ns=d2h_ns
        )
        boundary_pause_records = (
            tuple(
                {
                    "iteration_id": iteration,
                    "operation": operation,
                    "start_ns": layout[iteration - 1][2] + offset,
                    "end_ns": layout[iteration - 1][2] + offset + 1,
                    "duration_ns": 1,
                }
                for iteration in range(1, 8)
                for operation, offset in (("start", 1), ("stop", 3))
            )
            if entry.mode == "profiled" and state == "complete"
            else ()
        )
        _write_json(
            child_path,
            _child_document(
                metadata,
                entry,
                profile_wall_ns=profile_wall_ns,
                control_wall_ns=control_wall_ns,
                state=state,
                failure_class=failure_class,
                boundary_pause_records=boundary_pause_records,
            ),
        )
        _write_jsonl(
            events_path,
            (
                host_event_documents
                if entry.mode == "profiled" and state == "complete"
                else [
                    {
                        "schema_id": EVENT_SCHEMA_ID,
                        "child_id": entry.child_id,
                        "sequence": 0,
                        "event": "diagnostic_failure",
                        "failure_reason": "synthetic failure",
                    }
                ]
                if entry.mode == "profiled"
                else []
            ),
        )
        observation_document = _observations(entry.child_id)
        if failure_class == "scientific":
            observation_document["evaluations"] = [
                {
                    **observation_document["evaluations"][0],
                    "inner_success": False,
                    "trajectory_valid": False,
                },
                *observation_document["evaluations"][1:],
            ]
        if state != "complete":
            observation_document.update(
                {
                    "failure_reason": "synthetic failure",
                    "first_failed_evaluation_id": (
                        "evaluation-0" if failure_class == "scientific" else None
                    ),
                }
            )
        _write_json(observations_path, observation_document)
        evaluation_digest = evaluation_ids_sha256(
            tuple(
                str(evaluation["evaluation_id"])
                for evaluation in observation_document["evaluations"]
            )
        )
        _write_json(
            timing_path,
            (
                {
                    "schema_version": 1,
                    "wall_seconds": (
                        (
                            profile_wall_ns
                            if entry.mode == "profiled"
                            else control_wall_ns
                        )
                        + sum(
                            int(record["duration_ns"])
                            for record in boundary_pause_records
                        )
                    )
                    / 1_000_000_000,
                }
                if state == "complete"
                else {
                    "schema_id": (
                        "single-stage-changed-state-gpu-timeline-timing-unavailable-v1"
                    ),
                    "failure_reason": "synthetic failure",
                }
            ),
        )
        _write_jsonl(
            trajectory_path,
            []
            if state != "complete"
            else [
                {
                    "iteration": iteration,
                    "objective": 10.0 - iteration,
                    "wall_seconds_from_start": iteration
                    * (profile_wall_ns if entry.mode == "profiled" else control_wall_ns)
                    / 7
                    / 1_000_000_000,
                }
                for iteration in range(1, 8)
            ],
        )
        _write_json(
            provenance_path,
            {
                "schema_id": "single-stage-changed-state-gpu-timeline-provenance-v1",
                "child_id": entry.child_id,
                "collection_scope": (
                    "child_postexecution"
                    if state == "complete"
                    else "child_preexecution"
                ),
                "source_state_sha256": SOURCE_SHA,
                "environment": TEST_ENVIRONMENT,
                "device": {
                    "name": metadata.device_name,
                    "uuid": metadata.device_uuid,
                },
                "runtime": {
                    "python_version": metadata.python_version,
                    "jax_version": metadata.jax_version,
                    "jaxlib_version": metadata.jaxlib_version,
                },
                "profiler_policy": TEST_RUNTIME_POLICY["profiler_policy"][entry.mode],
                "authoritative": True,
                "executed_sources": [
                    {
                        "path": test_source_relative,
                        "sha256": next(
                            source["sha256"]
                            for source in source_preimages
                            if source["original_path"] == test_source_relative
                        ),
                        "git_blob_id": None,
                    }
                ],
                "simsoptpp_path": simsoptpp_identity["path"],
                "simsoptpp_sha256": simsoptpp_identity["sha256"],
                "simsoptpp_build_commit": simsoptpp_identity["build_commit"],
            },
        )
        monitor_path = inputs / entry.child_id / "monitor.json"
        stdout_path = inputs / entry.child_id / "stdout.log"
        stderr_path = inputs / entry.child_id / "stderr.log"
        _write_json(
            monitor_path,
            {
                "profiler_retention": (
                    {
                        "evidence_available": True,
                        "activity_buffers_dropped": False,
                        "warning": None,
                    }
                    if entry.mode == "profiled"
                    else {
                        "evidence_available": False,
                        "activity_buffers_dropped": None,
                        "warning": None,
                    }
                )
            },
        )
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        for role, path in (
            ("child_metadata", child_path),
            ("host_device_events", events_path),
            ("numerical_observations", observations_path),
            ("optimization_timing", timing_path),
            ("trajectory", trajectory_path),
            ("provenance", provenance_path),
        ):
            claims.append(
                ClaimFile(
                    role=role,
                    relative_path=f"process/{entry.child_id}/{path.name}",
                    source_path=path,
                    source_state_sha256=SOURCE_SHA,
                    process_id=entry.child_id,
                    evaluation_ids_sha256=evaluation_digest,
                    sample_id=entry.child_id,
                )
            )
        for path in (monitor_path, stdout_path, stderr_path):
            claims.append(
                ClaimFile(
                    role="diagnostic",
                    relative_path=f"children/{entry.child_id}/{path.name}",
                    source_path=path,
                    source_state_sha256=SOURCE_SHA,
                    process_id=entry.child_id,
                    evaluation_ids_sha256=evaluation_digest,
                    sample_id=entry.child_id,
                )
            )
        if entry.mode == "profiled" and state == "complete":
            for iteration in range(1, 8):
                sample_id = f"iteration-{iteration:02d}"
                segment_root = inputs / entry.child_id / "segments" / sample_id
                trace_path = segment_root / "trace.json.gz"
                summary_path = segment_root / "trace_summary.json"
                trace_document = _trace_document(
                    host_ns=host_ns,
                    newton_ns=newton_ns,
                    other_ns=other_ns,
                    unattributed_ns=unattributed_ns,
                    target_iteration=iteration,
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_bytes(
                    gzip.compress(canonical_json_bytes(trace_document), mtime=0)
                )
                segment_pauses = tuple(
                    Interval(int(record["start_ns"]), int(record["end_ns"]))
                    for record in boundary_pause_records
                    if record["iteration_id"] == iteration
                )
                summary = summarize_segmented_trace_document(
                    trace_document,
                    host_event_records,
                    child_id=entry.child_id,
                    sample_id=sample_id,
                    accepted_iteration=iteration,
                    profiler_boundary_pauses=segment_pauses,
                    evaluation_documents=observation_document["evaluations"],
                )
                _write_json(summary_path, summary.to_json())
                evaluation_id = f"evaluation-{iteration}"
                segment_evaluation_digest = evaluation_ids_sha256((evaluation_id,))
                claims.extend(
                    (
                        ClaimFile(
                            "raw_trace",
                            f"children/{entry.child_id}/segments/{sample_id}/trace.json.gz",
                            trace_path,
                            SOURCE_SHA,
                            entry.child_id,
                            evaluation_digest,
                            sample_id,
                            evaluation_id,
                            segment_evaluation_digest,
                        ),
                        ClaimFile(
                            "trace_summary",
                            f"children/{entry.child_id}/segments/{sample_id}/trace_summary.json",
                            summary_path,
                            SOURCE_SHA,
                            entry.child_id,
                            evaluation_digest,
                            sample_id,
                            evaluation_id,
                            segment_evaluation_digest,
                        ),
                    )
                )
    root = write_timeline_receipt(tmp_path / "artifact", metadata, tuple(claims))
    artifact_document = json.loads((root / "artifact.json").read_text())
    artifact_document["authoritative"] = True
    _write_json(root / "artifact.json", artifact_document)
    _rewrite_manifest_hash(root, "artifact.json")
    return root


def test_validator_cli_imports_without_pythonpath(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        (
            sys.executable,
            str(
                repo_root
                / "benchmarks/validate_single_stage_changed_state_gpu_timeline.py"
            ),
            "--help",
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_production_shape_rejected_trial_counts_all_objective_evaluations() -> None:
    child_id = "profiled-0"
    observations = _observations(child_id)
    evaluations = observations["evaluations"]
    assert isinstance(evaluations, list)
    final_reporting = evaluations.pop()
    rejected_trial = json.loads(json.dumps(evaluations[-1]))
    rejected_parameters = [8.0, 8.25]
    rejected_trial.update(
        {
            "evaluation_id": "evaluation-8",
            "evaluation_index": 8,
            "accepted": False,
            "iteration": 7,
            "parameter_sha256": hashlib.sha256(
                struct.pack("<2d", *rejected_parameters)
            ).hexdigest(),
            "parameters": rejected_parameters,
            "objective": 1.5,
        }
    )
    final_reporting.update(
        {
            "evaluation_id": "evaluation-9",
            "evaluation_index": 9,
        }
    )
    evaluations.extend((rejected_trial, final_reporting))
    trial_evaluations = [
        evaluation for evaluation in evaluations if evaluation["lifecycle"] == "trial"
    ]
    child = {
        "line_search_decisions": [
            {
                "evaluation_id": evaluation["evaluation_id"],
                "accepted": evaluation["accepted"],
                "iteration": evaluation["iteration"],
            }
            for evaluation in trial_evaluations
        ],
        "nit": 7,
        "nfev": 10,
        "njev": 10,
        "final_parameters": _parameters(7),
        "final_parameters_sha256": _parameter_sha256(7),
    }

    validated = _validate_observations(
        child_id,
        observations,
        {"initial_parameters_sha256": _parameter_sha256(0)},
        child,
    )

    assert len(validated) == 10


@pytest.mark.parametrize(
    ("host_ns", "newton_ns", "other_ns", "expected"),
    (
        (70, 20, 10, "HOST_BOUNDARY_DOMINANT"),
        (20, 70, 10, "NEWTON_ADJOINT_DOMINANT"),
        (45, 45, 10, "MIXED"),
    ),
)
def test_complete_evidence_reaches_each_attribution_verdict(
    tmp_path: Path,
    host_ns: int,
    newton_ns: int,
    other_ns: int,
    expected: str,
) -> None:
    root = _build_artifact(
        tmp_path, host_ns=host_ns, newton_ns=newton_ns, other_ns=other_ns
    )

    result_path, result = validate_and_publish(root)

    assert result["verdict"] == expected
    assert result["source_provenance_authoritative"] is True
    assert result["promotion_eligible"] is True
    assert result["engineering_branch_eligible"] is True
    assert result["claim_ceiling"] == "protocol_attribution"
    assert result_path == tmp_path / "artifact.validation" / "validation_result.json"
    assert not (root / "decision.json").exists()


def _set_source_authority_and_build_commit(
    root: Path,
    *,
    authoritative: bool,
    build_commit: str | None,
    mismatched_child: str | None = None,
) -> None:
    artifact_path = root / "artifact.json"
    artifact = json.loads(artifact_path.read_text())
    artifact_path.write_bytes(
        canonical_json_bytes({**artifact, "authoritative": authoritative})
    )
    _rewrite_manifest_hash(root, "artifact.json")
    identity_path = root / "identity_preimages.json"
    identity = json.loads(identity_path.read_text())
    identity_path.write_bytes(
        canonical_json_bytes(
            {
                **identity,
                "simsoptpp": {
                    **identity["simsoptpp"],
                    "build_commit": build_commit,
                },
            }
        )
    )
    _rewrite_manifest_hash(root, "identity_preimages.json")
    for child_id in (
        "profiled-0",
        "control-0",
        "profiled-1",
        "control-1",
        "profiled-2",
        "control-2",
    ):
        child_relative = f"process/{child_id}/child.json"
        child_path = root / child_relative
        child = json.loads(child_path.read_text())
        child_path.write_bytes(
            canonical_json_bytes(
                {
                    **child,
                    "provenance": {
                        **child["provenance"],
                        "authoritative": authoritative,
                    },
                }
            )
        )
        _rewrite_manifest_hash(root, child_relative)
        provenance_relative = f"process/{child_id}/provenance.json"
        provenance_path = root / provenance_relative
        provenance = json.loads(provenance_path.read_text())
        provenance_path.write_bytes(
            canonical_json_bytes(
                {
                    **provenance,
                    "authoritative": authoritative,
                    "simsoptpp_build_commit": (
                        "0" * 40 if child_id == mismatched_child else build_commit
                    ),
                }
            )
        )
        _rewrite_manifest_hash(root, provenance_relative)


def test_exact_byte_dirty_source_is_diagnostic_and_nonpromoting(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    _set_source_authority_and_build_commit(root, authoritative=False, build_commit=None)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "HOST_BOUNDARY_DOMINANT"
    assert result["source_provenance_authoritative"] is False
    assert result["promotion_eligible"] is False
    assert result["engineering_branch_eligible"] is True
    assert result["claim_ceiling"] == "diagnostic_attribution_only"


def test_authoritative_source_rejects_null_simsoptpp_build_commit(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    _set_source_authority_and_build_commit(root, authoritative=True, build_commit=None)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"
    assert result["promotion_eligible"] is False


def test_non_authoritative_source_rejects_child_build_commit_mismatch(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    _set_source_authority_and_build_commit(
        root,
        authoritative=False,
        build_commit=None,
        mismatched_child="control-2",
    )

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"
    assert result["promotion_eligible"] is False


def test_failed_child_cross_binds_null_build_commit_for_diagnostic_result(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path, failed_child=("profiled-0", "trace"))
    _set_source_authority_and_build_commit(root, authoritative=False, build_commit=None)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"
    assert result["source_provenance_authoritative"] is False
    assert result["promotion_eligible"] is False
    assert result["claim_ceiling"] == "diagnostic_attribution_only"


@pytest.mark.parametrize(
    "options",
    (
        {"profile_wall_ns": 120},
        {"host_ns": 20, "newton_ns": 20, "other_ns": 10, "unattributed_ns": 50},
        {"failed_child": ("profiled-0", "trace")},
    ),
)
def test_trace_or_overhead_failure_is_unattributable(
    tmp_path: Path, options: dict[str, object]
) -> None:
    root = _build_artifact(tmp_path, **options)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"
    assert result["metrics"] is None


def test_numerical_failure_is_scientific_invalid(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path, failed_child=("profiled-0", "scientific"))

    _, result = validate_and_publish(root)

    assert result["verdict"] == "SCIENTIFIC_INVALID"
    assert result["metrics"] is None


def test_child_reported_integrity_failure_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path, failed_child=("profiled-0", "integrity"))

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"
    assert result["valid"] is False


def test_early_failed_child_does_not_hide_later_malformed_evidence(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path, failed_child=("profiled-0", "trace"))
    relative_path = "process/control-2/provenance.json"
    path = root / relative_path
    document = json.loads(path.read_text())
    path.write_bytes(
        canonical_json_bytes(
            {**document, "environment": {**document["environment"], "BAD": "1"}}
        )
    )
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_later_scientific_failure_has_priority_over_earlier_trace_failure(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path, failed_child=("profiled-0", "trace"))
    child_relative = "process/control-2/child.json"
    child_path = root / child_relative
    child = json.loads(child_path.read_text())
    child_path.write_bytes(
        canonical_json_bytes(
            {
                **child,
                "state": "failed",
                "failure_class": "scientific",
                "failure_reason": "synthetic scientific failure",
                "first_failed_evaluation_id": "evaluation-0",
                "optimizer_raw_wall_ns": None,
                "optimizer_active_wall_ns": None,
                "profiler_boundary_pause_total_ns": 0,
                "boundary_pause_records": [],
            }
        )
    )
    _rewrite_manifest_hash(root, child_relative)
    observations_relative = "process/control-2/observations.json"
    observations_path = root / observations_relative
    observations = json.loads(observations_path.read_text())
    observations["evaluations"][0]["inner_success"] = False
    observations.update(
        {
            "failure_reason": "synthetic scientific failure",
            "first_failed_evaluation_id": "evaluation-0",
        }
    )
    observations_path.write_bytes(canonical_json_bytes(observations))
    _rewrite_manifest_hash(root, observations_relative)
    timing_relative = "process/control-2/timing.json"
    timing_path = root / timing_relative
    timing_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_id": (
                    "single-stage-changed-state-gpu-timeline-timing-unavailable-v1"
                ),
                "failure_reason": "synthetic scientific failure",
            }
        )
    )
    _rewrite_manifest_hash(root, timing_relative)
    trajectory_relative = "process/control-2/trajectory.jsonl"
    (root / trajectory_relative).write_bytes(b"")
    _rewrite_manifest_hash(root, trajectory_relative)
    provenance_relative = "process/control-2/provenance.json"
    provenance_path = root / provenance_relative
    provenance = json.loads(provenance_path.read_text())
    provenance_path.write_bytes(
        canonical_json_bytes({**provenance, "collection_scope": "child_preexecution"})
    )
    _rewrite_manifest_hash(root, provenance_relative)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "SCIENTIFIC_INVALID"
    assert result["first_failed_evaluation_id"] == "evaluation-0"


def _rewrite_manifest_hash(root: Path, relative_path: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    content = (root / relative_path).read_bytes()
    for entry in manifest["entries"]:
        if entry["relative_path"] == relative_path:
            entry["size_bytes"] = len(content)
            entry["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _mutate_newton_evidence(
    root: Path,
    child_ids: tuple[str, ...],
    mutate: object,
) -> None:
    for child_id in child_ids:
        relative_path = f"process/{child_id}/observations.json"
        path = root / relative_path
        document = json.loads(path.read_text())
        document["evaluations"] = [
            {
                **evaluation,
                "inner_evidence": mutate(dict(evaluation["inner_evidence"])),
            }
            for evaluation in document["evaluations"]
        ]
        path.write_bytes(canonical_json_bytes(document))
        _rewrite_manifest_hash(root, relative_path)


def _unavailable_newton_trace(evidence: dict[str, object]) -> dict[str, object]:
    return {
        **evidence,
        "newton_trace_available": False,
        "newton_attempted_iterations": None,
        "residual_trace": [],
        "step_accepted_trace": [],
        "linear_solve_success_trace": [],
    }


def test_unavailable_newton_trace_preserves_real_iteration_evidence(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    _mutate_newton_evidence(
        root,
        (
            "profiled-0",
            "control-0",
            "profiled-1",
            "control-1",
            "profiled-2",
            "control-2",
        ),
        _unavailable_newton_trace,
    )

    _, result = validate_and_publish(root)

    assert result["verdict"] == "HOST_BOUNDARY_DOMINANT"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda evidence: {
            **_unavailable_newton_trace(evidence),
            "newton_attempted_iterations": 0,
        },
        lambda evidence: {
            **_unavailable_newton_trace(evidence),
            "residual_trace": [1.0e-12],
        },
        lambda evidence: {**evidence, "newton_attempted_iterations": None},
        lambda evidence: {**evidence, "newton_attempted_iterations": 2},
        lambda evidence: {**evidence, "newton_iterations": 0},
    ),
)
def test_newton_trace_availability_count_or_length_tamper_is_integrity_error(
    tmp_path: Path, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    _mutate_newton_evidence(root, ("profiled-0",), mutate)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_profile_control_newton_trace_availability_mismatch_is_integrity_error(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    _mutate_newton_evidence(root, ("profiled-0",), _unavailable_newton_trace)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def _mutate_event_records(
    root: Path, mutate: object, *, child_id: str = "profiled-0"
) -> None:
    relative_path = f"process/{child_id}/events.jsonl"
    path = root / relative_path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutated = mutate(records)
    _write_jsonl(path, mutated)
    _rewrite_manifest_hash(root, relative_path)


def _host_records_from_documents(
    documents: list[dict[str, object]],
) -> tuple[HostEventRecord, ...]:
    return tuple(
        HostEventRecord(
            sequence=int(document["sequence"]),
            event=HostEvent(str(document["event"])),
            timestamp_ns=int(document["timestamp_ns"]),
            evaluation=EvaluationTraceContext(
                evaluation_id=str(document["evaluation_id"]),
                parameter_sha256=str(document["parameter_sha256"]),
                kind=EvaluationKind(str(document["evaluation_kind"])),
                outer_iteration_id=(
                    None
                    if document["outer_iteration_id"] is None
                    else int(document["outer_iteration_id"])
                ),
            ),
            attributes=(),
        )
        for document in documents
        if document.get("record_type") == "lifecycle"
    )


def _mutate_trace(
    root: Path,
    mutate: object,
    *,
    recompute_summary: bool,
    child_id: str = "profiled-0",
    iteration: int = 1,
) -> None:
    sample_id = f"iteration-{iteration:02d}"
    trace_relative = f"children/{child_id}/segments/{sample_id}/trace.json.gz"
    trace_path = root / trace_relative
    trace_document = json.loads(gzip.decompress(trace_path.read_bytes()))
    mutated = mutate(trace_document)
    trace_path.write_bytes(gzip.compress(canonical_json_bytes(mutated), mtime=0))
    _rewrite_manifest_hash(root, trace_relative)
    if recompute_summary:
        event_path = root / f"process/{child_id}/events.jsonl"
        event_documents = [
            json.loads(line) for line in event_path.read_text().splitlines()
        ]
        observations = json.loads(
            (root / f"process/{child_id}/observations.json").read_text()
        )
        child = json.loads((root / f"process/{child_id}/child.json").read_text())
        summary = summarize_segmented_trace_document(
            mutated,
            _host_records_from_documents(event_documents),
            child_id=child_id,
            sample_id=sample_id,
            accepted_iteration=iteration,
            profiler_boundary_pauses=tuple(
                Interval(int(record["start_ns"]), int(record["end_ns"]))
                for record in child["boundary_pause_records"]
                if record["iteration_id"] == iteration
            ),
            evaluation_documents=observations["evaluations"],
        )
        summary_relative = (
            f"children/{child_id}/segments/{sample_id}/trace_summary.json"
        )
        _write_json(root / summary_relative, summary.to_json())
        _rewrite_manifest_hash(root, summary_relative)


def _add_ambiguous_device_intervals(
    document: dict[str, object],
) -> dict[str, object]:
    accepted = {
        int(event["args"]["step_num"]): round(float(event["ts"]) * 1000)
        for event in document["traceEvents"]
        if event.get("name") == "optimizer.accepted_iteration"
    }
    for iteration in sorted(accepted):
        start_ns = accepted[iteration] + 5
        duration_ns = 60
        for phase in ("host.h2d_submit", "host.d2h_materialize"):
            document["traceEvents"].append(
                {
                    "ph": "X",
                    "name": phase,
                    "pid": 1,
                    "tid": 2,
                    "ts": (start_ns - 1) / 1000,
                    "dur": (duration_ns + 2) / 1000,
                    "args": {},
                }
            )
        document["traceEvents"].append(
            {
                "ph": "X",
                "name": "ambiguous memcpy",
                "pid": 2,
                "tid": 2,
                "ts": start_ns / 1000,
                "dur": duration_ns / 1000,
                "args": {
                    "context_id": "$$1",
                    "correlation_id": f"ambiguous-{iteration}",
                    "name": "fusion_kernel",
                    "memcpy_details": (
                        "kind_src:Host kind_dst:Device size:8 dest:0 async:1"
                    ),
                },
            }
        )
    return document


@pytest.mark.parametrize(
    ("relative_path", "mutate"),
    (
        (
            "artifact.json",
            lambda document: {**document, "schema_id": "unknown-schema"},
        ),
        (
            "artifact.json",
            lambda document: {
                **document,
                "route": {**document["route"], "adjoint_route": "operator_gmres"},
            },
        ),
        (
            "artifact.json",
            lambda document: {**document, "phase_ids": ["unknown.phase"]},
        ),
        (
            "artifact.json",
            lambda document: {**document, "trace_schema_id": "unknown-trace-v9"},
        ),
        (
            "process/profiled-0/child.json",
            lambda document: {**document, "device_uuid": "GPU-tampered"},
        ),
        (
            "process/profiled-0/child.json",
            lambda document: {**document, "source_state_sha256": "9" * 64},
        ),
        (
            "process/profiled-0/child.json",
            lambda document: {**document, "environment_sha256": "9" * 64},
        ),
        (
            "process/profiled-0/child.json",
            lambda document: {
                **document,
                "provenance": {**document["provenance"], "authoritative": False},
            },
        ),
        (
            "process/profiled-0/provenance.json",
            lambda document: {**document, "authoritative": False},
        ),
        (
            "process/control-0/child.json",
            lambda document: {**document, "cache_sha256": "2" * 64},
        ),
        (
            "process/profiled-0/observations.json",
            lambda document: {
                **document,
                "evaluations": [
                    *document["evaluations"][:-1],
                    {
                        **document["evaluations"][-1],
                        "parameter_sha256": "f" * 64,
                    },
                ],
            },
        ),
        (
            "process/profiled-0/observations.json",
            lambda document: {
                **document,
                "evaluations": [
                    document["evaluations"][0],
                    {**document["evaluations"][1], "objective": 999.0},
                    *document["evaluations"][2:],
                ],
            },
        ),
        (
            "children/profiled-0/segments/iteration-01/trace_summary.json",
            lambda document: {**document, "device_active_ns": 1},
        ),
    ),
)
def test_schema_route_identity_and_correlation_tampering_is_integrity_error(
    tmp_path: Path, relative_path: str, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    path = root / relative_path
    document = json.loads(path.read_text())
    path.write_bytes(canonical_json_bytes(mutate(document)))
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"
    assert not (root / "decision.json").exists()


def test_unmanifested_tamper_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    (root / "process" / "profiled-0" / "trace.json").write_text("tampered\n")

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


@pytest.mark.parametrize(
    ("relative_path", "mutate"),
    (
        (
            "process/profiled-0/timing.json",
            lambda document: {**document, "wall_seconds": 999.0},
        ),
        (
            "process/profiled-0/provenance.json",
            lambda document: {
                **document,
                "environment": {**document["environment"], "EXTRA": "tampered"},
            },
        ),
    ),
)
def test_raw_timing_or_provenance_tamper_is_integrity_error(
    tmp_path: Path, relative_path: str, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    path = root / relative_path
    path.write_bytes(canonical_json_bytes(mutate(json.loads(path.read_text()))))
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


@pytest.mark.parametrize(
    ("relative_path", "mutate"),
    (
        (
            "identity_preimages.json",
            lambda document: {
                **document,
                "configuration": {"test_configuration": 2},
            },
        ),
        (
            "identity_preimages.json",
            lambda document: {
                **document,
                "runtime_policy_payload": {
                    **document["runtime_policy_payload"],
                    "timeout_seconds": 61.0,
                },
            },
        ),
        (
            "identity_preimages.json",
            lambda document: {
                **document,
                "simsoptpp": {**document["simsoptpp"], "path": "/tmp/wrong.so"},
            },
        ),
        (
            "preflight/preflight.json",
            lambda document: {
                **document,
                "device_identity": {**document["device_identity"], "uuid": "wrong"},
            },
        ),
        (
            "preflight/preflight.json",
            lambda document: {
                **document,
                "observed_evidence": [
                    {
                        **document["observed_evidence"][0],
                        "ambiguous_device_kernel_intervals": 1,
                    },
                    *document["observed_evidence"][1:],
                ],
            },
        ),
        (
            "preflight/preflight.json",
            lambda document: {
                **document,
                "observed_evidence": [
                    {
                        **document["observed_evidence"][0],
                        "uniquely_attributed_device_kernel_intervals": 3,
                    },
                    *document["observed_evidence"][1:],
                ],
            },
        ),
        (
            "preflight/preflight.json",
            lambda document: {
                **document,
                "profiler_policy": {
                    **document["profiler_policy"],
                    "host_tracer_level": 0,
                },
            },
        ),
    ),
)
def test_identity_or_preflight_preimage_tamper_is_integrity_error(
    tmp_path: Path, relative_path: str, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    path = root / relative_path
    path.write_bytes(canonical_json_bytes(mutate(json.loads(path.read_text()))))
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


@pytest.mark.parametrize(
    ("relative_path", "mutate"),
    (
        (
            "identity_preimages.json",
            lambda document: {
                **document,
                "runtime_policy_payload": {
                    **document["runtime_policy_payload"],
                    "profiler_policy": {
                        **document["runtime_policy_payload"]["profiler_policy"],
                        "profiled": {
                            **document["runtime_policy_payload"]["profiler_policy"][
                                "profiled"
                            ],
                            "host_tracer_level": 0,
                        },
                    },
                },
            },
        ),
        (
            "identity_preimages.json",
            lambda document: {
                **document,
                "runtime_policy_payload": {
                    **document["runtime_policy_payload"],
                    "profiler_policy": {
                        **document["runtime_policy_payload"]["profiler_policy"],
                        "profiled": {
                            **document["runtime_policy_payload"]["profiler_policy"][
                                "profiled"
                            ],
                            "trace_viewer_max_events": 1_000_000,
                        },
                    },
                },
            },
        ),
        (
            "process/profiled-0/child.json",
            lambda document: {
                **document,
                "profiler_policy": {
                    **document["profiler_policy"],
                    "python_tracer_level": 1,
                },
            },
        ),
        (
            "preflight/preflight.json",
            lambda document: {
                **document,
                "profiler_policy": {
                    **document["profiler_policy"],
                    "device_tracing": "disabled",
                },
            },
        ),
        (
            "process/control-0/child.json",
            lambda document: {
                **document,
                "profiler_policy": {
                    **document["profiler_policy"],
                    "enabled": True,
                },
            },
        ),
        (
            "process/profiled-0/provenance.json",
            lambda document: {
                **document,
                "profiler_policy": {
                    **document["profiler_policy"],
                    "advanced_configuration": {
                        **document["profiler_policy"]["advanced_configuration"],
                        "gpu_max_activity_api_events": 1_000_000,
                    },
                },
            },
        ),
        (
            "process/control-0/provenance.json",
            lambda document: {
                **document,
                "environment": {
                    **document["environment"],
                    "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS": "1000000",
                },
            },
        ),
    ),
)
def test_profiler_policy_tamper_is_integrity_error(
    tmp_path: Path, relative_path: str, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    path = root / relative_path
    path.write_bytes(canonical_json_bytes(mutate(json.loads(path.read_text()))))
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_cupti_activity_drop_is_unattributable(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    child_id = "profiled-0"
    monitor_relative = f"children/{child_id}/monitor.json"
    stderr_relative = f"children/{child_id}/stderr.log"
    monitor_path = root / monitor_relative
    monitor = json.loads(monitor_path.read_text())
    monitor["profiler_retention"] = {
        "evidence_available": True,
        "activity_buffers_dropped": True,
        "warning": "Already too many activity events, drop the buffer",
    }
    monitor_path.write_bytes(canonical_json_bytes(monitor))
    (root / stderr_relative).write_bytes(
        b"Already too many activity events, drop the buffer\n"
    )
    _rewrite_manifest_hash(root, monitor_relative)
    _rewrite_manifest_hash(root, stderr_relative)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"
    assert result["valid"] is True
    assert result["failing_gates"] == [
        "profiled-0: CUPTI activity buffers were dropped"
    ]


def test_hidden_cupti_activity_drop_warning_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    relative_path = "children/profiled-0/stderr.log"
    (root / relative_path).write_bytes(
        b"Already too many activity events, drop the buffer\n"
    )
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_control_cannot_claim_profiler_retention(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    relative_path = "children/control-0/monitor.json"
    path = root / relative_path
    monitor = json.loads(path.read_text())
    monitor["profiler_retention"] = {
        "evidence_available": True,
        "activity_buffers_dropped": False,
        "warning": None,
    }
    path.write_bytes(canonical_json_bytes(monitor))
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_manifest_rehash_does_not_hide_input_array_byte_tamper(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    relative_path = "inputs/inputs/coil_dofs.npy"
    path = root / relative_path
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_raw_trajectory_tamper_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    relative_path = "process/profiled-0/trajectory.jsonl"
    path = root / relative_path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[2]["objective"] = 999.0
    _write_jsonl(path, records)
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_manifest_aggregate_evaluation_binding_tamper_is_integrity_error(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][1]["evaluation_ids_sha256"] = "0" * 64
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def _mutate_first_host_span(
    records: list[dict[str, object]],
    mutate: object,
) -> list[dict[str, object]]:
    span_index = next(
        index
        for index, record in enumerate(records)
        if record.get("record_type") == "host_span"
    )
    records[span_index] = mutate(records[span_index])
    return records


def _mutate_first_optimizer_span(
    records: list[dict[str, object]],
    mutate: object,
) -> list[dict[str, object]]:
    span_index = next(
        index
        for index, record in enumerate(records)
        if record.get("record_type") == "optimizer_span"
    )
    records[span_index] = mutate(records[span_index])
    return records


def _drop_first_host_span(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    dropped = False
    span_sequence = 0
    result: list[dict[str, object]] = []
    for record in records:
        if record.get("record_type") == "host_span" and not dropped:
            dropped = True
            continue
        if record.get("record_type") == "host_span":
            record = {**record, "sequence": span_sequence}
            span_sequence += 1
        result.append(record)
    return result


def _duplicate_first_host_span(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    first = next(
        record for record in records if record.get("record_type") == "host_span"
    )
    span_count = sum(record.get("record_type") == "host_span" for record in records)
    return [*records, {**first, "sequence": span_count}]


def _overlap_first_host_spans(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    span_indices = [
        index
        for index, record in enumerate(records)
        if record.get("record_type") == "host_span"
    ]
    first = records[span_indices[0]]
    second = records[span_indices[1]]
    records[span_indices[1]] = {
        **second,
        "start_ns": first["start_ns"],
        "end_ns": first["end_ns"],
    }
    return records


@pytest.mark.parametrize(
    "mutate",
    (
        _drop_first_host_span,
        _duplicate_first_host_span,
        _overlap_first_host_spans,
        lambda records: _mutate_first_host_span(
            records, lambda span: {**span, "end_ns": span["start_ns"]}
        ),
        lambda records: _mutate_first_host_span(
            records, lambda span: {**span, "phase_id": "unknown.host.phase"}
        ),
        lambda records: _mutate_first_host_span(
            records,
            lambda span: {
                **span,
                "attributes": {**span["attributes"], "evaluation_id": "unknown"},
            },
        ),
    ),
)
def test_host_span_reversal_phase_or_identity_tamper_is_integrity_error(
    tmp_path: Path, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    _mutate_event_records(root, mutate)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda records: _mutate_first_optimizer_span(
            records, lambda span: {**span, "phase_id": "host.h2d_submit"}
        ),
        lambda records: _mutate_first_optimizer_span(
            records, lambda span: {**span, "attributes": {}}
        ),
        lambda records: _mutate_first_optimizer_span(
            records,
            lambda span: {
                **span,
                "attributes": {"accepted_iteration_id": 2},
            },
        ),
        lambda records: _mutate_first_host_span(
            records, lambda span: {**span, "phase_id": "optimizer.lifecycle"}
        ),
    ),
)
def test_optimizer_lifecycle_span_taxonomy_or_identity_tamper_is_integrity_error(
    tmp_path: Path, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    _mutate_event_records(root, mutate)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_optimizer_lifecycle_spans_are_nonexclusive_and_nonattributing(
    tmp_path: Path,
) -> None:
    baseline_root = _build_artifact(tmp_path / "baseline")
    changed_root = _build_artifact(tmp_path / "changed")

    def widen_optimizer_spans(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            (
                {**record, "start_ns": 0, "end_ns": 1_000_000}
                if record.get("record_type") == "optimizer_span"
                else record
            )
            for record in records
        ]

    _mutate_event_records(changed_root, widen_optimizer_spans)
    _, baseline_result = validate_and_publish(baseline_root)
    _, changed_result = validate_and_publish(changed_root)

    assert baseline_result["verdict"] == "HOST_BOUNDARY_DOMINANT"
    assert changed_result["verdict"] == baseline_result["verdict"]
    assert changed_result["metrics"] == baseline_result["metrics"]


def test_manifested_claims_are_consumed_from_one_immutable_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_artifact(tmp_path)
    artifact_path = root / "artifact.json"
    manifest_path = root / "manifest.json"
    original_read_bytes = Path.read_bytes
    reads = {artifact_path: 0, manifest_path: 0}

    def count_claim_reads(path: Path) -> bytes:
        if path in reads:
            reads[path] += 1
            if reads[path] > 1:
                return b"mutated-after-validation\n"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_claim_reads)
    _, result = validate_and_publish(root)

    assert result["verdict"] == "HOST_BOUNDARY_DOMINANT"
    assert reads == {artifact_path: 1, manifest_path: 1}


@pytest.mark.parametrize(
    "mutate",
    (
        lambda records: [records[0], {**records[0]}, *records[1:]],
        lambda records: [
            {**records[0], "evaluation_id": "unknown-evaluation"},
            *records[1:],
        ],
        lambda records: [{**records[0], "parameter_sha256": "f" * 64}, *records[1:]],
        lambda records: [
            records[0],
            {
                **records[1],
                "previous_evaluation_id": "evaluation-2",
                "next_evaluation_id": "evaluation-1",
            },
            *records[2:],
        ],
    ),
)
def test_duplicate_reversed_overlapping_or_miscorrelated_event_is_integrity_error(
    tmp_path: Path, mutate: object
) -> None:
    root = _build_artifact(tmp_path)
    _mutate_event_records(root, mutate)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_missing_completion_or_phase_family_is_unattributable(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)

    def remove_lifecycle_completion(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        kept = [record for index, record in enumerate(records) if index != 1]
        lifecycle_sequence = 0
        for record in kept:
            if record.get("record_type") == "lifecycle":
                record["sequence"] = lifecycle_sequence
                lifecycle_sequence += 1
        return kept

    _mutate_event_records(
        root,
        remove_lifecycle_completion,
    )

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"
    assert result["metrics"] is None


def test_ambiguous_multi_owner_device_intervals_are_unattributed(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path, host_ns=20, newton_ns=60, other_ns=20)
    for iteration in range(1, 8):
        _mutate_trace(
            root,
            _add_ambiguous_device_intervals,
            recompute_summary=True,
            iteration=iteration,
        )

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"
    assert result["metrics"] is None


def test_reversed_or_overlapping_iteration_interval_is_unattributable(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)

    def overlap(document: dict[str, object]) -> dict[str, object]:
        envelopes = [
            event
            for event in document["traceEvents"]
            if event.get("name") == "optimizer.accepted_iteration"
        ]
        document["traceEvents"].append({**envelopes[0]})
        return document

    _mutate_trace(root, overlap, recompute_summary=False)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"


def test_legitimate_overlapping_same_owner_device_events_are_unioned(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)

    def duplicate_kernel(document: dict[str, object]) -> dict[str, object]:
        event = next(
            event
            for event in document["traceEvents"]
            if event.get("pid") == 2
            and event.get("args", {}).get("name") == "newton.residual_jvp"
        )
        document["traceEvents"].append({**event, "name": "overlapping same owner"})
        return document

    _mutate_trace(root, duplicate_kernel, recompute_summary=True)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "HOST_BOUNDARY_DOMINANT"


@pytest.mark.parametrize(
    ("mutate", "recompute_summary"),
    (
        (
            lambda document: {
                **document,
                "traceEvents": [
                    event
                    for event in document["traceEvents"]
                    if not (event.get("pid") == 2 and event.get("ph") == "X")
                ],
            },
            True,
        ),
        (
            lambda document: {
                **document,
                "traceEvents": [
                    {
                        **event,
                        "args": {**event.get("args", {}), "name": "unknown.phase"},
                    }
                    if event.get("pid") == 2
                    and event.get("args", {}).get("name") == "adjoint.lu_factor"
                    else event
                    for event in document["traceEvents"]
                ],
            },
            False,
        ),
    ),
)
def test_absent_cuda_records_or_unknown_trace_phase_is_unattributable(
    tmp_path: Path, mutate: object, recompute_summary: bool
) -> None:
    root = _build_artifact(tmp_path)
    _mutate_trace(root, mutate, recompute_summary=recompute_summary)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"
    assert result["metrics"] is None


def test_reversed_device_interval_is_unattributable(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)

    def reverse_device(document: dict[str, object]) -> dict[str, object]:
        event = next(
            event
            for event in document["traceEvents"]
            if event.get("pid") == 2 and event.get("ph") == "X"
        )
        event["dur"] = 0
        return document

    _mutate_trace(root, reverse_device, recompute_summary=False)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "UNATTRIBUTABLE"


def test_missing_segment_file_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    (root / "children/profiled-0/segments/iteration-04/trace.json.gz").unlink()

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"
    assert result["valid"] is False
    assert result["metrics"] is None


def test_duplicate_segment_sample_id_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target = next(
        entry
        for entry in manifest["entries"]
        if entry["process_id"] == "profiled-0"
        and entry["role"] == "raw_trace"
        and entry["sample_id"] == "iteration-02"
    )
    target["sample_id"] = "iteration-01"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_swapped_segment_evaluation_bindings_are_integrity_error(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = [
        entry
        for entry in manifest["entries"]
        if entry["process_id"] == "profiled-0"
        and entry["role"] in {"raw_trace", "trace_summary"}
        and entry["sample_id"] in {"iteration-01", "iteration-02"}
    ]
    for entry in entries:
        entry["evaluation_id"] = (
            "evaluation-2" if entry["sample_id"] == "iteration-01" else "evaluation-1"
        )
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_control_segment_evidence_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_entry = next(
        entry
        for entry in manifest["entries"]
        if entry["process_id"] == "profiled-0"
        and entry["role"] == "raw_trace"
        and entry["sample_id"] == "iteration-01"
    )
    source = root / source_entry["relative_path"]
    relative_path = "children/control-0/segments/iteration-01/trace.json.gz"
    destination = root / relative_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    manifest["entries"].append(
        {
            **source_entry,
            "relative_path": relative_path,
            "process_id": "control-0",
        }
    )
    manifest["entries"].sort(key=lambda entry: entry["relative_path"])
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_boundary_pause_arithmetic_tamper_is_integrity_error(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    relative_path = "process/profiled-0/child.json"
    path = root / relative_path
    child = json.loads(path.read_text())
    child["boundary_pause_records"][0]["duration_ns"] += 1
    path.write_bytes(canonical_json_bytes(child))
    _rewrite_manifest_hash(root, relative_path)

    _, result = validate_and_publish(root)

    assert result["verdict"] == "INTEGRITY_ERROR"


def test_missing_root_publishes_external_integrity_result(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-artifact"

    result_path, result = validate_and_publish(missing_root)

    assert result["verdict"] == "INTEGRITY_ERROR"
    assert result_path == Path(f"{missing_root}.validation") / "validation_result.json"
    assert not missing_root.exists()


def test_external_validation_is_exclusive_and_non_overwriting(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    validate_and_publish(root)

    with pytest.raises(FileExistsError, match="already exists"):
        validate_and_publish(root)


def test_writer_rejects_existing_or_authoritative_temporary_root(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    root = _build_artifact(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        write_timeline_receipt(root, metadata, ())

    with pytest.raises(ValueError, match="must not be written under /tmp"):
        write_timeline_receipt(
            tmp_path / "authoritative", replace(metadata, authoritative=True), ()
        )
