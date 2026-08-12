from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryResult,
    ProcessGpuMemorySample,
)
from benchmarks.single_stage_compute_graph_c0_capture import (
    C0CaptureError,
    TraceCaptureFacts,
    _bind_identity_anchor,
    build_capture_document,
    summarize_c0_trace,
)
from benchmarks.single_stage_compute_graph_c0_evaluator import (
    CAPTURE_SCHEMA_ID,
    ChildRequest,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    HLO_MODULE_SET_IDENTITY_SOURCE,
    SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    canonical_hlo_module_set_identity,
)

_HOST_PID = 1
_DEVICE_PID = 2
_PARAMETER_SHA256 = "a" * 64


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


def _kernel(module: str, timestamp_us: float) -> dict[str, object]:
    return _span(
        "kernel",
        timestamp_us,
        5.0,
        pid=_DEVICE_PID,
        args={
            "context_id": "$$1",
            "correlation_id": "1",
            "hlo_module": module,
            "hlo_op": f"{module}/multiply",
            "kernel_details": "regs:16",
            "scope_range_id": "2",
            "tf_op": "XlaModule:",
        },
    )


def _trace(*extra_events: dict[str, object]) -> dict[str, object]:
    return {
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": True},
        "traceEvents": [
            _metadata(_HOST_PID, "/host:CPU"),
            _metadata(_DEVICE_PID, "/device:GPU:0"),
            _lifecycle("evaluator_entry", 10.0),
            _span(
                "CommonPjRtLoadedExecutable::Execute (arguments_are_tupled=false)",
                20.0,
                50.0,
            ),
            *extra_events,
            _kernel("jit_forward", 30.0),
            _kernel("jit_gradient", 40.0),
            _lifecycle("device_ready", 80.0),
            _lifecycle("evaluator_return", 90.0),
            {},
        ],
    }


def test_trace_summary_counts_one_canonical_pjrt_family_and_all_kernels() -> None:
    document = _trace(
        _span(
            "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice",
            21.0,
            48.0,
        ),
        _span("PjRtCApiLoadedExecutable::Execute", 22.0, 47.0),
    )

    facts = summarize_c0_trace(document, _PARAMETER_SHA256)

    assert facts.pjrt_execute_count == 1
    assert facts.kernel_launch_count == 2
    assert facts.hlo_modules == ("jit_forward", "jit_gradient")
    assert facts.hlo_module_set_identity == canonical_hlo_module_set_identity(
        facts.hlo_modules
    )
    assert facts == summarize_c0_trace(document, _PARAMETER_SHA256)


def test_trace_summary_supports_single_device_helper_fallback() -> None:
    document = _trace()
    document["traceEvents"][3]["name"] = (  # type: ignore[index]
        "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"
    )

    assert summarize_c0_trace(document, _PARAMETER_SHA256).pjrt_execute_count == 1


def test_trace_summary_rejects_compilation_inside_evaluation() -> None:
    with pytest.raises(C0CaptureError, match="compilation occurred"):
        summarize_c0_trace(
            _trace(_span("PJRT_Client_Compile", 25.0, 2.0)),
            _PARAMETER_SHA256,
        )


def test_trace_summary_rejects_candidate_identity_mismatch() -> None:
    with pytest.raises(C0CaptureError, match="frozen candidate"):
        summarize_c0_trace(_trace(), "b" * 64)


def test_capture_document_labels_sampled_memory_and_trace_identity() -> None:
    memory = ProcessGpuMemoryResult(
        gpu_uuid="GPU-1",
        provider_pid=123,
        samples=(
            ProcessGpuMemorySample(sampled_at_unix_ns=1, used_memory_mib=20),
            ProcessGpuMemorySample(sampled_at_unix_ns=2, used_memory_mib=25),
        ),
        peak_used_memory_mib=25,
    )
    facts = TraceCaptureFacts(
        hlo_module_set_identity=canonical_hlo_module_set_identity(("jit_forward",)),
        hlo_modules=("jit_forward",),
        pjrt_execute_count=2,
        kernel_launch_count=10,
    )

    document = build_capture_document(
        ChildRequest("profile", None), _PARAMETER_SHA256, facts, memory
    )

    assert document["schema_id"] == CAPTURE_SCHEMA_ID
    assert document["sampled_process_gpu_memory_peak_bytes"] == 25 * 1024 * 1024
    assert (
        document["sampled_process_gpu_memory_source"]
        == SAMPLED_PROCESS_GPU_MEMORY_SOURCE
    )
    assert document["hlo_module_set_identity_source"] == HLO_MODULE_SET_IDENTITY_SOURCE
    assert document["pjrt_execute_count"] == 2
    assert document["kernel_launch_count"] == 10


def test_first_capture_omits_warm_counts() -> None:
    memory = ProcessGpuMemoryResult(
        gpu_uuid="GPU-1",
        provider_pid=123,
        samples=(ProcessGpuMemorySample(sampled_at_unix_ns=1, used_memory_mib=1),),
        peak_used_memory_mib=1,
    )
    document = build_capture_document(
        ChildRequest("first", None),
        _PARAMETER_SHA256,
        TraceCaptureFacts(
            canonical_hlo_module_set_identity(("jit_f",)), ("jit_f",), 1, 1
        ),
        memory,
    )

    assert document["pjrt_execute_count"] is None
    assert document["kernel_launch_count"] is None


def test_capture_document_rejects_mislabeled_module_set_identity() -> None:
    memory = ProcessGpuMemoryResult(
        gpu_uuid="GPU-1",
        provider_pid=123,
        samples=(ProcessGpuMemorySample(sampled_at_unix_ns=1, used_memory_mib=1),),
        peak_used_memory_mib=1,
    )
    with pytest.raises(C0CaptureError, match="does not match trace modules"):
        build_capture_document(
            ChildRequest("warm", 1),
            _PARAMETER_SHA256,
            TraceCaptureFacts(
                canonical_hlo_module_set_identity(("other",)),
                ("jit_f",),
                1,
                1,
            ),
            memory,
        )


def test_warm_identity_must_match_first_capture(tmp_path: Path) -> None:
    anchor = tmp_path / "identity.json"
    identity = canonical_hlo_module_set_identity(("jit-f",))
    _bind_identity_anchor(anchor, ChildRequest("profile", None), identity)
    _bind_identity_anchor(anchor, ChildRequest("warm", 1), identity)

    assert (
        json.loads(anchor.read_text(encoding="utf-8"))["hlo_module_set_identity"]
        == identity
    )
    with pytest.raises(C0CaptureError, match="differs"):
        _bind_identity_anchor(
            anchor,
            ChildRequest("warm", 2),
            canonical_hlo_module_set_identity(("other",)),
        )
