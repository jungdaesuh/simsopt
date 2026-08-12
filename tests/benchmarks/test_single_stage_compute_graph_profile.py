from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    _validate_profile,
)
from benchmarks.single_stage_compute_graph_profile import (
    ComputeGraphProfileError,
    Phase0ReceiptProfileMismatch,
    _disjoint_phase_intervals,
    build_attribution_control_profile_evidence,
    build_profile_evidence,
    canonical_json_bytes,
    parse_profile_evidence,
    summarize_compute_graph_profile,
    write_profile_evidence,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    Interval,
    ScopeAttribution,
    load_trace_document,
)
from simsopt_jax.runtime.trace_annotations import PhaseId

_HOST_PID = 11
_DEVICE_PID = 12
_PARAMETER_SHA256 = "a" * 64
_SPECIMEN_SHA256 = "b" * 64
_INPUT_BUNDLE_SHA256 = "1" * 64
_SOURCE_SHA256 = "c" * 64
_RUNTIME_IDENTITY_SHA256 = "f" * 64
_GATE_CHECKPOINT_SHA256 = "d" * 64
_WARM_CHECKPOINT_SHA256 = "e" * 64
_WARM_P50_NS = 1234.5


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


def _lifecycle(event: str, timestamp_us: float) -> dict[str, object]:
    return _span(
        f"optimizer.lifecycle.{event}",
        timestamp_us,
        1.0,
        args={
            "evaluation_id": _PARAMETER_SHA256,
            "parameter_sha256": _PARAMETER_SHA256,
            "evaluation_kind": "trial",
            "outer_iteration_id": None,
        },
    )


def _execute(module: str, timestamp_us: float, mode: str) -> dict[str, object]:
    return _span(
        f"CommonPjRtLoadedExecutable::Execute ({module})",
        timestamp_us,
        20.0,
        args={"name": module, "execution_mode": mode},
    )


def _kernel(
    module: str,
    phase: str | None,
    timestamp_us: float,
    duration_us: float,
    *,
    graph_id: str | None = None,
) -> dict[str, object]:
    name = f"jit({module})/{phase}" if phase is not None else f"jit({module})"
    graph_metadata = (
        {}
        if graph_id is None
        else {"cuda_graph_id": graph_id, "cuda_graph_node_id": "1"}
    )
    return _span(
        f"{module}_kernel",
        timestamp_us,
        duration_us,
        pid=_DEVICE_PID,
        args={
            "context_id": "$$1",
            "correlation_id": module,
            "hlo_module": module,
            "hlo_op": f"{module}/fusion",
            "kernel_details": "regs:16",
            "name": name,
            "scope_range_id": module,
            "tf_op": "XlaModule:",
            **graph_metadata,
        },
    )


def _graph_events(
    graph_id: str, correlation_id: str, timestamp_us: float
) -> tuple[dict[str, object], ...]:
    return (
        _span(
            "command_buffer::execute",
            timestamp_us,
            12.0,
            args={"device": "0", "num_commands": "1", "num_executions": "1"},
        ),
        _span(
            f"cuGraphLaunch (CudaGraph:{graph_id})",
            timestamp_us + 1.0,
            10.0,
            args={
                "context_id": "$$1",
                "correlation_id": correlation_id,
                "cuda_graph_id": graph_id,
                "device_id": "0",
                "scope_range_id": correlation_id,
            },
        ),
    )


def _memcpy(
    timestamp_us: float,
    duration_us: float,
    *,
    correlation_id: str = "memcpy",
    graph_id: str | None = None,
) -> dict[str, object]:
    graph_metadata = (
        {}
        if graph_id is None
        else {"cuda_graph_id": graph_id, "cuda_graph_node_id": "1"}
    )
    return _span(
        "MemcpyD2D",
        timestamp_us,
        duration_us,
        pid=_DEVICE_PID,
        args={
            "context_id": "$$1",
            "correlation_id": correlation_id,
            "memcpy_details": "kind_src:device kind_dst:device size:8 dest:0 async:1",
            **graph_metadata,
        },
    )


def _trace(
    *,
    first_mode: str = "command_buffer",
    second_mode: str = "uncaptured",
    second_phase: str | None = "adjoint.lu_solve",
    graph_modules: tuple[str, ...] = ("jit_forward",),
    extra_events: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    first_graph_id = "101" if "jit_forward" in graph_modules else None
    second_graph_id = "102" if "jit_gradient" in graph_modules else None
    return {
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": True},
        "traceEvents": [
            _metadata(_HOST_PID, "/host:CPU"),
            _metadata(_DEVICE_PID, "/device:GPU:0"),
            _lifecycle("evaluator_entry", 10.0),
            _execute("jit_forward", 20.0, first_mode),
            _execute("jit_gradient", 45.0, second_mode),
            *(
                _graph_events("101", "jit_forward", 22.0)
                if first_graph_id is not None
                else ()
            ),
            *(
                _graph_events("102", "jit_gradient", 42.0)
                if second_graph_id is not None
                else ()
            ),
            *extra_events,
            _kernel(
                "jit_forward",
                "newton.residual_jvp",
                25.0,
                10.0,
                graph_id=first_graph_id,
            ),
            _kernel(
                "jit_gradient",
                second_phase,
                45.0,
                10.0,
                graph_id=second_graph_id,
            ),
            _lifecycle("device_ready", 80.0),
            _lifecycle("evaluator_return", 90.0),
            {},
        ],
    }


def _profile_evidence(
    root: Path,
    *,
    first_mode: str = "command_buffer",
    second_mode: str = "uncaptured",
):
    trace_root = root / "traces"
    trace_root.mkdir()
    trace_path = trace_root / "profile.trace.json"
    trace_path.write_bytes(
        canonical_json_bytes(_trace(first_mode=first_mode, second_mode=second_mode))
    )
    evidence = build_profile_evidence(
        trace_path=trace_path,
        artifact_root=root,
        candidate_sha256=_PARAMETER_SHA256,
        specimen_sha256=_SPECIMEN_SHA256,
        input_bundle_sha256=_INPUT_BUNDLE_SHA256,
        source_sha256=_SOURCE_SHA256,
        runtime_identity_sha256=_RUNTIME_IDENTITY_SHA256,
        lane_id="rtx5090",
        gpu_uuid="GPU-profile-test",
        gate_checkpoint_sha256=_GATE_CHECKPOINT_SHA256,
        warm_checkpoint_sha256=_WARM_CHECKPOINT_SHA256,
        warm_p50_ns=_WARM_P50_NS,
    )
    return evidence, trace_path


def test_profile_emits_exact_phase0_profile_and_command_buffer_documents() -> None:
    summary = summarize_compute_graph_profile(_trace(), _PARAMETER_SHA256)

    assert summary.evaluation_envelope_ns == 81_000
    assert summary.device_active_ns == 20_000
    assert summary.attributed_union_ns == 20_000
    assert summary.unattributed_ns == 0
    assert summary.attribution_coverage == 1.0
    assert summary.pjrt_execute_count == 2
    assert summary.kernel_launch_count == 2
    assert summary.kernel_duration_ns == (10_000, 10_000)
    assert summary.inter_launch_gap_ns == 10_000
    assert summary.device_active_share == pytest.approx(20_000 / 81_000)
    assert summary.inter_launch_gap_share == pytest.approx(10_000 / 81_000)

    documents = summary.phase0_documents()
    assert set(documents) == {"profile", "command_buffer"}
    profile = documents["profile"]
    assert isinstance(profile, dict)
    assert profile["phase_interval_unions"] == [
        {"phase_id": "adjoint.lu_solve", "intervals": [[45_000, 55_000]]},
        {"phase_id": "newton.residual_jvp", "intervals": [[25_000, 35_000]]},
    ]
    command_buffer = documents["command_buffer"]
    assert isinstance(command_buffer, dict)
    assert command_buffer == {
        "observed_pjrt_execution_modes": ["command_buffer", "uncaptured"],
        "resolved_xla_configuration": None,
        "observed_capture_participation": True,
        "command_buffer_execute_count": 1,
        "graph_api_launch_count": 1,
        "graph_device_activity_count": 1,
        "graph_kernel_activity_count": 1,
        "graph_memcpy_activity_count": 0,
        "graph_other_device_activity_count": 0,
        "direct_device_activity_count": 1,
        "direct_kernel_activity_count": 1,
        "direct_memcpy_activity_count": 0,
        "direct_other_device_activity_count": 0,
        "graph_device_union_ns": 10_000,
        "direct_device_union_ns": 10_000,
        "graph_direct_overlap_ns": 0,
        "classified_device_union_ns": 20_000,
        "ab_control": None,
    }
    _validate_profile(profile, "measurement.profile")


def test_attribution_control_profile_allows_low_coverage_without_weakening_direct(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    trace_path = trace_root / "profile.trace.json"
    trace_path.write_bytes(canonical_json_bytes(_trace(second_phase=None)))
    kwargs = {
        "trace_path": trace_path,
        "artifact_root": tmp_path,
        "candidate_sha256": _PARAMETER_SHA256,
        "specimen_sha256": _SPECIMEN_SHA256,
        "input_bundle_sha256": _INPUT_BUNDLE_SHA256,
        "source_sha256": _SOURCE_SHA256,
        "runtime_identity_sha256": _RUNTIME_IDENTITY_SHA256,
        "lane_id": "rtx5090",
        "gpu_uuid": "GPU-profile-test",
        "gate_checkpoint_sha256": _GATE_CHECKPOINT_SHA256,
        "warm_checkpoint_sha256": _WARM_CHECKPOINT_SHA256,
        "warm_p50_ns": _WARM_P50_NS,
    }

    with pytest.raises(ComputeGraphProfileError, match="below"):
        build_profile_evidence(**kwargs)
    control = build_attribution_control_profile_evidence(**kwargs)
    assert control.profile.attribution_coverage == 0.5


def test_phase_interval_sweep_preserves_overlap_and_boundary_semantics() -> None:
    def item(
        start_ns: int,
        end_ns: int,
        phase: PhaseId | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            interval=Interval(start_ns, end_ns),
            attribution=ScopeAttribution(phase, phase, phase is None),
        )

    intervals = _disjoint_phase_intervals(
        (
            item(0, 10, PhaseId.NEWTON_RESIDUAL_JVP),
            item(2, 8, PhaseId.NEWTON_RESIDUAL_JVP),
            item(4, 6, PhaseId.ADJOINT_LU_SOLVE),
            item(8, 9, None),
        )
    )

    assert intervals == (
        (
            "newton.residual_jvp",
            (Interval(0, 4), Interval(6, 8), Interval(9, 10)),
        ),
    )


@pytest.mark.parametrize(
    "phase",
    (
        PhaseId.NEWTON_WARM_START,
        PhaseId.NEWTON_SOLVER_CONTROL,
        PhaseId.NEWTON_JACOBIAN_CONSTRUCTION,
        PhaseId.NEWTON_DENSE_MATERIALIZATION,
        PhaseId.NEWTON_LU_FACTOR,
        PhaseId.NEWTON_REFINEMENT,
    ),
)
def test_phase_interval_sweep_retains_newton_profile_phases(
    phase: PhaseId,
) -> None:
    item = SimpleNamespace(
        interval=Interval(10, 20),
        attribution=ScopeAttribution(phase, phase, False),
    )

    assert _disjoint_phase_intervals((item,)) == ((phase.value, (Interval(10, 20),)),)


def test_profile_counts_only_canonical_pjrt_family_and_all_cuda_kernels() -> None:
    document = _trace(
        extra_events=(
            _span("PjRtCApiLoadedExecutable::Execute", 21.0, 18.0),
            _span(
                "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice",
                22.0,
                17.0,
            ),
        )
    )

    summary = summarize_compute_graph_profile(document, _PARAMETER_SHA256)

    assert summary.pjrt_execute_count == 2
    assert summary.kernel_launch_count == 2


def test_profile_maps_exact_biotsavart_hlo_module_owner() -> None:
    document = _trace()
    events = document["traceEvents"]
    assert isinstance(events, list)
    first_kernel = next(
        event
        for event in events
        if isinstance(event, dict)
        and isinstance(event.get("args"), dict)
        and event["args"].get("hlo_module") == "jit_forward"
    )
    args = first_kernel["args"]
    assert isinstance(args, dict)
    args["name"] = "jit(biotsavart_forward)/fusion"
    args["hlo_module"] = "jit_biotsavart_forward"

    summary = summarize_compute_graph_profile(document, _PARAMETER_SHA256)

    phases = dict(summary.phase_interval_unions)
    assert phases["biotsavart.forward"] == (Interval(25_000, 35_000),)


def test_profile_rejects_compilation_inside_exact_envelope() -> None:
    with pytest.raises(ComputeGraphProfileError, match="compilation occurred"):
        summarize_compute_graph_profile(
            _trace(extra_events=(_span("PJRT_Client_Compile", 30.0, 1.0),)),
            _PARAMETER_SHA256,
        )


def test_profile_rejects_less_than_ninety_percent_attribution() -> None:
    with pytest.raises(ComputeGraphProfileError, match="below 0.900000"):
        summarize_compute_graph_profile(
            _trace(second_phase=None),
            _PARAMETER_SHA256,
        )


def test_default_execution_policy_with_graph_trace_is_observed_captured() -> None:
    summary = summarize_compute_graph_profile(
        _trace(
            first_mode="default", second_mode="default", graph_modules=("jit_forward",)
        ),
        _PARAMETER_SHA256,
    )

    assert summary.command_buffer is not None
    assert summary.command_buffer.observed_pjrt_execution_modes == ("default",)
    assert summary.command_buffer.resolved_xla_configuration is None
    assert summary.command_buffer.observed_capture_participation is True
    assert summary.command_buffer.graph_api_launch_count == 1
    assert summary.command_buffer.graph_device_activity_count == 1
    assert summary.command_buffer.direct_device_activity_count == 1
    assert summary.command_buffer.graph_device_union_ns == 10_000
    assert summary.command_buffer.direct_device_union_ns == 10_000
    assert summary.command_buffer.graph_direct_overlap_ns == 0
    assert summary.command_buffer.classified_device_union_ns == 20_000


def test_default_execution_policy_without_graph_trace_is_observed_uncaptured() -> None:
    summary = summarize_compute_graph_profile(
        _trace(
            first_mode="default",
            second_mode="default",
            graph_modules=(),
        ),
        _PARAMETER_SHA256,
    )

    assert summary.command_buffer is not None
    assert summary.command_buffer.observed_pjrt_execution_modes == ("default",)
    assert summary.command_buffer.resolved_xla_configuration is None
    assert summary.command_buffer.observed_capture_participation is False
    assert summary.command_buffer.graph_api_launch_count == 0
    assert summary.command_buffer.graph_device_activity_count == 0
    assert summary.command_buffer.direct_device_activity_count == 2
    assert summary.command_buffer.graph_device_union_ns == 0
    assert summary.command_buffer.direct_device_union_ns == 20_000
    assert summary.command_buffer.graph_direct_overlap_ns == 0
    assert summary.command_buffer.classified_device_union_ns == 20_000


def test_mixed_graph_and_direct_kernels_use_trace_participation() -> None:
    summary = summarize_compute_graph_profile(
        _trace(
            first_mode="default",
            second_mode="default",
            graph_modules=("jit_forward",),
        ),
        _PARAMETER_SHA256,
    )

    assert summary.command_buffer is not None
    assert summary.command_buffer.graph_device_union_ns == 10_000
    assert summary.command_buffer.direct_device_union_ns == 10_000
    assert summary.command_buffer.graph_api_launch_count == 1
    assert summary.command_buffer.graph_device_activity_count == 1
    assert summary.command_buffer.direct_device_activity_count == 1


def test_one_graph_api_launch_can_bind_multiple_kernel_activities() -> None:
    summary = summarize_compute_graph_profile(
        _trace(
            first_mode="default",
            second_mode="default",
            extra_events=(
                _kernel(
                    "jit_forward",
                    "newton.residual_jvp",
                    36.0,
                    2.0,
                    graph_id="101",
                ),
            ),
        ),
        _PARAMETER_SHA256,
    )

    assert summary.command_buffer is not None
    assert summary.command_buffer.graph_api_launch_count == 1
    assert summary.command_buffer.graph_device_activity_count == 2
    assert summary.command_buffer.graph_kernel_activity_count == 2
    assert summary.command_buffer.graph_memcpy_activity_count == 0


def test_reused_graph_id_is_disambiguated_by_launch_correlation() -> None:
    document = _trace(
        first_mode="default",
        second_mode="default",
        graph_modules=("jit_forward", "jit_gradient"),
    )
    events = document["traceEvents"]
    assert isinstance(events, list)
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("args"), dict):
            continue
        args = event["args"]
        if args.get("cuda_graph_id") != "102":
            continue
        args["cuda_graph_id"] = "101"
        if event.get("name") == "cuGraphLaunch (CudaGraph:102)":
            event["name"] = "cuGraphLaunch (CudaGraph:101)"

    summary = summarize_compute_graph_profile(document, _PARAMETER_SHA256)

    assert summary.command_buffer is not None
    assert summary.command_buffer.graph_api_launch_count == 2
    assert summary.command_buffer.graph_device_activity_count == 2
    assert summary.command_buffer.direct_device_activity_count == 0


def test_graph_node_memcpy_can_be_the_only_activity_for_one_launch() -> None:
    summary = summarize_compute_graph_profile(
        _trace(
            first_mode="default",
            second_mode="default",
            extra_events=(
                *_graph_events("103", "graph-memcpy", 60.0),
                _memcpy(
                    62.0,
                    2.0,
                    correlation_id="graph-memcpy",
                    graph_id="103",
                ),
            ),
        ),
        _PARAMETER_SHA256,
    )

    assert summary.command_buffer is not None
    assert summary.command_buffer.graph_api_launch_count == 2
    assert summary.command_buffer.graph_device_activity_count == 2
    assert summary.command_buffer.graph_kernel_activity_count == 1
    assert summary.command_buffer.graph_memcpy_activity_count == 1
    assert summary.command_buffer.graph_device_union_ns == 12_000
    assert summary.command_buffer.classified_device_union_ns == 22_000


def test_graph_launch_without_any_graph_activity_fails_closed() -> None:
    summary = summarize_compute_graph_profile(
        _trace(
            first_mode="default",
            second_mode="default",
            extra_events=_graph_events("103", "orphan-launch", 60.0),
        ),
        _PARAMETER_SHA256,
    )

    assert summary.command_buffer is None
    assert summary.command_buffer_unavailable_reason == (
        "every cuGraphLaunch must bind at least one graph device activity"
    )


def test_concurrent_graph_and_direct_activity_records_overlap_and_total_union() -> None:
    summary = summarize_compute_graph_profile(
        _trace(
            first_mode="default",
            second_mode="default",
            extra_events=(
                _kernel(
                    "jit_forward",
                    "newton.residual_jvp",
                    30.0,
                    10.0,
                ),
            ),
        ),
        _PARAMETER_SHA256,
    )

    assert summary.command_buffer is not None
    assert summary.command_buffer.graph_device_union_ns == 10_000
    assert summary.command_buffer.direct_device_union_ns == 20_000
    assert summary.command_buffer.graph_direct_overlap_ns == 5_000
    assert summary.command_buffer.classified_device_union_ns == 25_000
    assert (
        summary.command_buffer.graph_device_union_ns
        + summary.command_buffer.direct_device_union_ns
        - summary.command_buffer.graph_direct_overlap_ns
        == summary.command_buffer.classified_device_union_ns
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("missing_node", "device graph metadata requires both graph and node IDs"),
        ("malformed_node", "device graph and node IDs must be positive"),
        ("unknown_graph", "device graph metadata is not bound to a cuGraphLaunch"),
        ("missing_execute", "command_buffer::execute and cuGraphLaunch threads"),
    ),
)
def test_malformed_or_inconsistent_graph_metadata_fails_closed(
    mutation: str, reason: str
) -> None:
    document = _trace(first_mode="default", second_mode="default")
    events = document["traceEvents"]
    assert isinstance(events, list)
    graph_kernel = next(
        event
        for event in events
        if isinstance(event, dict)
        and isinstance(event.get("args"), dict)
        and event["args"].get("cuda_graph_id") == "101"
        and "kernel_details" in event["args"]
    )
    graph_args = graph_kernel["args"]
    assert isinstance(graph_args, dict)
    if mutation == "missing_node":
        del graph_args["cuda_graph_node_id"]
    elif mutation == "malformed_node":
        graph_args["cuda_graph_node_id"] = "node"
    elif mutation == "unknown_graph":
        graph_args["cuda_graph_id"] = "999"
    else:
        events[:] = [
            event
            for event in events
            if not (
                isinstance(event, dict)
                and event.get("name") == "command_buffer::execute"
            )
        ]

    summary = summarize_compute_graph_profile(document, _PARAMETER_SHA256)

    assert summary.command_buffer is None
    assert summary.command_buffer_unavailable_reason is not None
    assert reason in summary.command_buffer_unavailable_reason
    with pytest.raises(Phase0ReceiptProfileMismatch):
        summary.phase0_documents()


def test_inter_launch_gap_excludes_overlapping_memcpy_and_uses_envelope_share() -> None:
    summary = summarize_compute_graph_profile(
        _trace(extra_events=(_memcpy(38.0, 2.0),)),
        _PARAMETER_SHA256,
    )

    assert summary.inter_launch_gap_ns == 8_000
    assert summary.inter_launch_gap_share == pytest.approx(8_000 / 81_000)


def test_canonical_json_is_stable_and_finite() -> None:
    document = summarize_compute_graph_profile(
        _trace(), _PARAMETER_SHA256
    ).phase0_documents()

    encoded = canonical_json_bytes(document)

    assert encoded.endswith(b"\n")
    assert encoded == canonical_json_bytes(document)
    assert b"NaN" not in encoded


def test_bound_profile_artifact_round_trips_and_recomputes_raw_trace(
    tmp_path: Path,
) -> None:
    evidence, trace_path = _profile_evidence(tmp_path)
    artifact_path = tmp_path / "profile-evidence.json"

    write_profile_evidence(artifact_path, evidence)
    parsed = parse_profile_evidence(artifact_path, expected_identity=evidence.identity)

    assert parsed == evidence
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert document["identity"] == {
        "candidate_sha256": _PARAMETER_SHA256,
        "specimen_sha256": _SPECIMEN_SHA256,
        "input_bundle_sha256": _INPUT_BUNDLE_SHA256,
        "source_sha256": _SOURCE_SHA256,
        "runtime_identity_sha256": _RUNTIME_IDENTITY_SHA256,
        "lane_id": "rtx5090",
        "gpu_uuid": "GPU-profile-test",
        "gate_checkpoint_sha256": _GATE_CHECKPOINT_SHA256,
        "warm_checkpoint_sha256": _WARM_CHECKPOINT_SHA256,
        "warm_p50_ns": _WARM_P50_NS,
    }
    assert document["trace"] == {
        "path": "traces/profile.trace.json",
        "schema_id": "jax-profiler-chrome-trace-jax-0.10.0-v1",
        "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    assert document["profile"] == evidence.profile.profile_phase0_json()
    assert parsed.phase0_documents() == evidence.profile.phase0_documents()


def test_parser_rejects_runner_identity_mismatch(tmp_path: Path) -> None:
    evidence, _trace_path = _profile_evidence(tmp_path)
    artifact_path = tmp_path / "profile-evidence.json"
    write_profile_evidence(artifact_path, evidence)

    with pytest.raises(ComputeGraphProfileError, match="runner-owned expected"):
        parse_profile_evidence(
            artifact_path,
            expected_identity=replace(evidence.identity, warm_p50_ns=999.0),
        )


def test_parser_rejects_runtime_identity_mismatch(tmp_path: Path) -> None:
    evidence, _trace_path = _profile_evidence(tmp_path)
    artifact_path = tmp_path / "profile-evidence.json"
    write_profile_evidence(artifact_path, evidence)

    with pytest.raises(ComputeGraphProfileError, match="runner-owned expected"):
        parse_profile_evidence(
            artifact_path,
            expected_identity=replace(
                evidence.identity, runtime_identity_sha256="0" * 64
            ),
        )


def test_parser_rejects_input_bundle_identity_mismatch(tmp_path: Path) -> None:
    evidence, _trace_path = _profile_evidence(tmp_path)
    artifact_path = tmp_path / "profile-evidence.json"
    write_profile_evidence(artifact_path, evidence)

    with pytest.raises(ComputeGraphProfileError, match="runner-owned expected"):
        parse_profile_evidence(
            artifact_path,
            expected_identity=replace(evidence.identity, input_bundle_sha256="0" * 64),
        )


def test_parser_rejects_trace_byte_or_recomputed_profile_tampering(
    tmp_path: Path,
) -> None:
    evidence, trace_path = _profile_evidence(tmp_path)
    artifact_path = tmp_path / "profile-evidence.json"
    write_profile_evidence(artifact_path, evidence)
    trace_document = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_events = trace_document["traceEvents"]
    assert isinstance(trace_events, list)
    kernel = next(
        event
        for event in trace_events
        if isinstance(event, dict)
        and isinstance(event.get("args"), dict)
        and "kernel_details" in event["args"]
    )
    kernel["dur"] = 11.0
    trace_path.write_bytes(canonical_json_bytes(trace_document))

    with pytest.raises(ComputeGraphProfileError, match="trace-recomputed"):
        parse_profile_evidence(artifact_path, expected_identity=evidence.identity)


def test_default_policy_graph_participation_is_recomputed_in_bound_artifact(
    tmp_path: Path,
) -> None:
    evidence, _trace_path = _profile_evidence(
        tmp_path, first_mode="default", second_mode="default"
    )
    artifact_path = tmp_path / "profile-evidence.json"
    write_profile_evidence(artifact_path, evidence)

    parsed = parse_profile_evidence(artifact_path, expected_identity=evidence.identity)
    classification = parsed.to_json()["diagnostics"]
    assert isinstance(classification, dict)
    command_buffer_classification = classification["command_buffer_classification"]
    assert isinstance(command_buffer_classification, dict)
    assert command_buffer_classification["state"] == "available"
    command_buffer = command_buffer_classification["evidence"]
    assert isinstance(command_buffer, dict)
    assert command_buffer["observed_pjrt_execution_modes"] == ["default"]
    assert command_buffer["resolved_xla_configuration"] is None
    assert command_buffer["observed_capture_participation"] is True
    assert parsed.phase0_documents()["command_buffer"] == command_buffer


def test_parser_enforces_trace_cannot_resolve_xla_configuration(tmp_path: Path) -> None:
    evidence, _trace_path = _profile_evidence(tmp_path)
    artifact_path = tmp_path / "profile-evidence.json"
    write_profile_evidence(artifact_path, evidence)
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    diagnostics = document["diagnostics"]
    assert isinstance(diagnostics, dict)
    classification = diagnostics["command_buffer_classification"]
    assert isinstance(classification, dict)
    command_buffer = classification["evidence"]
    assert isinstance(command_buffer, dict)
    command_buffer["resolved_xla_configuration"] = "--xla_gpu_enable_command_buffer"
    artifact_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ComputeGraphProfileError, match="trace-recomputed"):
        parse_profile_evidence(artifact_path, expected_identity=evidence.identity)


def test_profile_artifact_writer_is_exclusive(tmp_path: Path) -> None:
    evidence, _trace_path = _profile_evidence(tmp_path)
    artifact_path = tmp_path / "profile-evidence.json"
    write_profile_evidence(artifact_path, evidence)

    with pytest.raises(FileExistsError):
        write_profile_evidence(artifact_path, evidence)


def test_existing_rtx_preflight_fails_closed_without_exact_lifecycle() -> None:
    trace_root = Path(".artifacts/compute-graph-phase0/rtx5090-trace-preflight/trace")
    traces = tuple(sorted(trace_root.rglob("*.trace.json.gz")))
    if not traces:
        pytest.skip("RTX preflight trace is not present in this checkout")

    document = load_trace_document(traces[0])
    with pytest.raises(ComputeGraphProfileError, match="exactly ENTRY, READY, RETURN"):
        summarize_compute_graph_profile(document, _PARAMETER_SHA256)
