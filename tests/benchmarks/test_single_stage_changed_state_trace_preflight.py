from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import benchmarks.run_single_stage_changed_state_gpu_timeline as timeline_runner
import benchmarks.single_stage_changed_state_trace_preflight as preflight_module
import pytest
from benchmarks.single_stage_changed_state_trace_preflight import (
    EVIDENCE_FILENAME,
    PREFLIGHT_SCHEMA_ID,
    DeviceIdentity,
    _canonical_json_bytes,
    _execute_canary,
    evaluate_preflight_sessions,
    evaluate_trace_scope_survival,
    run_trace_preflight,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    TRACE_SCHEMA_ID,
)
from simsopt_jax.runtime.trace_annotations import PhaseId


def _kernel_event(*, phase: PhaseId, timestamp_us: int) -> dict[str, object]:
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
            "kernel_details": "regs:16 static_shared:0 dynamic_shared:0 grid:1,1,1 block:32,1,1",
            "name": "canary_kernel",
            "scope_range_id": "1",
            "tf_op": f"jit_canary/{phase.value}/multiply",
        },
    }


def _trace_document(*events: dict[str, object]) -> dict[str, object]:
    return {
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
            *events,
            {},
        ],
    }


def _identity() -> DeviceIdentity:
    return DeviceIdentity(name="synthetic-gpu", uuid="GPU-synthetic")


def test_exact_parser_accepts_two_uniquely_surviving_required_scopes() -> None:
    evidence = evaluate_trace_scope_survival(
        _trace_document(
            _kernel_event(phase=PhaseId.NEWTON_RESIDUAL_JVP, timestamp_us=10),
            _kernel_event(phase=PhaseId.ADJOINT_LU_SOLVE, timestamp_us=20),
        ),
        device_identity=_identity(),
    )

    assert evidence == {
        "schema_id": PREFLIGHT_SCHEMA_ID,
        "state": "pass",
        "trace_schema_id": TRACE_SCHEMA_ID,
        "required_scopes": [
            PhaseId.NEWTON_RESIDUAL_JVP.value,
            PhaseId.ADJOINT_LU_SOLVE.value,
        ],
        "observed_evidence": [
            {
                "phase_id": PhaseId.NEWTON_RESIDUAL_JVP.value,
                "device_kernel_intervals_containing_scope": 1,
                "uniquely_attributed_device_kernel_intervals": 1,
                "ambiguous_device_kernel_intervals": 0,
            },
            {
                "phase_id": PhaseId.ADJOINT_LU_SOLVE.value,
                "device_kernel_intervals_containing_scope": 1,
                "uniquely_attributed_device_kernel_intervals": 1,
                "ambiguous_device_kernel_intervals": 0,
            },
        ],
        "device_identity": {
            "name": "synthetic-gpu",
            "uuid": "GPU-synthetic",
        },
        "profiler_policy": {
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
        "session_evidence": [],
        "failure_reason": None,
    }


def test_actual_producer_output_passes_runner_acceptance_without_schema_copy() -> None:
    identity = _identity()
    document = _trace_document(
        _kernel_event(phase=PhaseId.NEWTON_RESIDUAL_JVP, timestamp_us=10),
        _kernel_event(phase=PhaseId.ADJOINT_LU_SOLVE, timestamp_us=20),
    )
    evidence = evaluate_preflight_sessions(
        (document, document),
        device_identity=identity,
    )

    timeline_runner._validate_preflight_evidence(
        evidence,
        trace_schema_id=TRACE_SCHEMA_ID,
        device_name=identity.name,
        device_uuid=identity.uuid,
    )

    drifted = copy.deepcopy(evidence)
    first_observation = drifted["observed_evidence"][0]
    first_observation["uniquely_attributed_device_intervals"] = first_observation.pop(
        "uniquely_attributed_device_kernel_intervals"
    )
    with pytest.raises(
        timeline_runner.TimelineRunnerError,
        match="scope evidence",
    ):
        timeline_runner._validate_preflight_evidence(
            drifted,
            trace_schema_id=TRACE_SCHEMA_ID,
            device_name=identity.name,
            device_uuid=identity.uuid,
        )


def test_missing_required_scope_fails_closed() -> None:
    evidence = evaluate_trace_scope_survival(
        _trace_document(
            _kernel_event(phase=PhaseId.NEWTON_RESIDUAL_JVP, timestamp_us=10)
        ),
        device_identity=_identity(),
    )

    assert evidence["state"] == "failed"
    assert "adjoint.lu_solve" in str(evidence["failure_reason"])


def test_sequential_preflight_requires_two_gpu_zero_sessions() -> None:
    valid = _trace_document(
        _kernel_event(phase=PhaseId.NEWTON_RESIDUAL_JVP, timestamp_us=10),
        _kernel_event(phase=PhaseId.ADJOINT_LU_SOLVE, timestamp_us=20),
    )

    missing_session = evaluate_preflight_sessions((valid,), device_identity=_identity())
    wrong_device = copy.deepcopy(valid)
    wrong_device["traceEvents"][1]["args"]["name"] = "/device:GPU:1"
    wrong_device_evidence = evaluate_preflight_sessions(
        (valid, wrong_device), device_identity=_identity()
    )

    assert missing_session["state"] == "failed"
    assert "expected 2 sequential traces" in str(missing_session["failure_reason"])
    assert wrong_device_evidence["state"] == "failed"
    assert "/device:GPU:0" in str(wrong_device_evidence["failure_reason"])


def test_ambiguous_device_attribution_fails_closed() -> None:
    ambiguous = _kernel_event(
        phase=PhaseId.NEWTON_RESIDUAL_JVP,
        timestamp_us=10,
    )
    args = ambiguous["args"]
    assert isinstance(args, dict)
    args["tf_op"] = f"jit_canary/{PhaseId.ADJOINT_LU_SOLVE.value}/subtract"
    evidence = evaluate_trace_scope_survival(
        _trace_document(
            ambiguous,
            _kernel_event(phase=PhaseId.ADJOINT_LU_SOLVE, timestamp_us=20),
        ),
        device_identity=_identity(),
    )

    assert evidence["state"] == "failed"
    assert evidence["observed_evidence"][0]["ambiguous_device_kernel_intervals"] == 1


def test_unknown_trace_schema_is_serialized_as_failure() -> None:
    document = _trace_document(
        _kernel_event(phase=PhaseId.NEWTON_RESIDUAL_JVP, timestamp_us=10),
        _kernel_event(phase=PhaseId.ADJOINT_LU_SOLVE, timestamp_us=20),
    )
    document["unexpected"] = True

    evidence = evaluate_trace_scope_survival(
        document,
        device_identity=_identity(),
    )

    assert evidence["state"] == "failed"
    assert evidence["observed_evidence"] == []
    assert "unknown_trace_schema" in str(evidence["failure_reason"])


def test_evidence_encoding_is_canonical_json() -> None:
    encoded = _canonical_json_bytes({"z": 1, "a": [2, 3]})

    assert encoded == b'{"a":[2,3],"z":1}\n'


def test_run_requires_fresh_output_root_before_gpu_execution(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()

    with pytest.raises(FileExistsError):
        run_trace_preflight(occupied, device_identity=_identity())

    assert not (occupied / EVIDENCE_FILENAME).exists()


def test_standalone_cli_help_bootstraps_repo_imports_without_pythonpath(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "single_stage_changed_state_trace_preflight.py"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        (sys.executable, str(script), "--help"),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output-root" in completed.stdout


def test_runtime_failure_is_persisted_as_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_canary(_trace_root: Path) -> None:
        raise RuntimeError("synthetic profiler failure")

    monkeypatch.setattr(preflight_module, "_execute_canary", fail_canary)
    output_root = tmp_path / "preflight"

    evidence = run_trace_preflight(output_root, device_identity=_identity())

    assert evidence["state"] == "failed"
    assert evidence["failure_reason"] == "RuntimeError: synthetic profiler failure"
    assert json.loads((output_root / EVIDENCE_FILENAME).read_bytes()) == evidence


def test_canary_warms_both_exact_callables_before_profiler_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    captured_options: list[object] = []

    class FakeProfileOptions:
        def __init__(self) -> None:
            self.host_tracer_level = -1
            self.python_tracer_level = -1
            self.advanced_configuration: dict[str, int] = {}

    def residual(_values: object) -> str:
        calls.append("residual")
        return "residual-result"

    def adjoint(_values: object) -> str:
        calls.append("adjoint")
        return "adjoint-result"

    @contextmanager
    def fake_trace_session() -> Iterator[None]:
        calls.append("session-enter")
        try:
            yield
        finally:
            calls.append("session-exit")

    monkeypatch.setattr(
        preflight_module,
        "_build_jitted_canary",
        lambda: (residual, adjoint),
    )
    monkeypatch.setattr(preflight_module.jax, "devices", lambda _kind: ("gpu",))
    monkeypatch.setattr(preflight_module.jnp, "arange", lambda *_a, **_kw: "host")
    monkeypatch.setattr(preflight_module.jax, "device_put", lambda _v, _d: "device")
    monkeypatch.setattr(
        preflight_module.jax,
        "block_until_ready",
        lambda value: calls.append(f"block:{value}"),
    )
    monkeypatch.setattr(preflight_module, "trace_session", fake_trace_session)
    monkeypatch.setattr(
        preflight_module.jax.profiler, "ProfileOptions", FakeProfileOptions
    )
    monkeypatch.setattr(
        preflight_module.jax.profiler,
        "start_trace",
        lambda _path, *, profiler_options: (
            captured_options.append(profiler_options),
            calls.append("profiler-start"),
        ),
    )
    monkeypatch.setattr(
        preflight_module.jax.profiler,
        "stop_trace",
        lambda: calls.append("profiler-stop"),
    )
    monkeypatch.setattr(
        preflight_module,
        "_single_trace_path",
        lambda path: path / "trace.json.gz",
    )

    _execute_canary(tmp_path)

    assert calls == [
        "session-enter",
        "residual",
        "block:residual-result",
        "adjoint",
        "block:adjoint-result",
        "session-exit",
        "session-enter",
        "profiler-start",
        "residual",
        "block:residual-result",
        "adjoint",
        "block:adjoint-result",
        "profiler-stop",
        "session-exit",
        "session-enter",
        "profiler-start",
        "residual",
        "block:residual-result",
        "adjoint",
        "block:adjoint-result",
        "profiler-stop",
        "session-exit",
    ]
    assert len(captured_options) == 2
    options = captured_options[0]
    assert isinstance(options, FakeProfileOptions)
    assert options.host_tracer_level == 1
    assert options.python_tracer_level == 0
    assert not hasattr(options, "device_tracer_level")
    assert options.advanced_configuration == {
        "gpu_max_activity_api_events": 33_554_432,
        "gpu_max_callback_api_events": 33_554_432,
    }


def test_device_identity_rejects_missing_claim() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        DeviceIdentity(name="", uuid="GPU-synthetic")


def test_canonical_evidence_round_trips_without_non_json_values() -> None:
    evidence = evaluate_trace_scope_survival(
        _trace_document(
            _kernel_event(phase=PhaseId.NEWTON_RESIDUAL_JVP, timestamp_us=10),
            _kernel_event(phase=PhaseId.ADJOINT_LU_SOLVE, timestamp_us=20),
        ),
        device_identity=_identity(),
    )

    assert json.loads(_canonical_json_bytes(evidence)) == evidence
