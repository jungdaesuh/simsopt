"""Runner contract tests for custom quasi-Newton measurements."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks import custom_quasi_newton_runtime as runtime
from benchmarks.fixtures.custom_quasi_newton import (
    Fixture,
    _certified_traceable_endpoint,
    fixture,
)
from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryMonitorError,
    ProcessGpuMemoryResult,
    ProcessGpuMemorySample,
    ProcessGpuMemoryUnavailable,
)


class _Child:
    pid = 1234
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        del timeout
        return "", ""


def _write_minimal_child_payload(
    output: Path,
    *,
    device: str,
    device_identity: dict[str, object],
    method: str = "lbfgs",
    case: str | None = None,
    solver_route: str | None = None,
    maxiter: int | None = None,
    trial_trace: str | None = None,
    capture_boozer_trial_trace: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": runtime._RUNNER_SCHEMA_VERSION,
        "provider_child": True,
        "requested_device": device,
        "method": method,
        "runtime_environment": runtime._runtime_environment_payload(),
        "git_commit": "candidate-commit",
        "git_clean": False,
        "device_identity": device_identity,
        "measurements": [
            {
                "provider": "custom",
                "device_identity": device_identity,
                "peak_vram_mib": None,
                "diagnostic_artifacts": {
                    "memory_trace": None,
                    "trial_trace": trial_trace,
                },
            }
        ],
    }
    measurement = payload["measurements"][0]
    if case is not None:
        measurement["case"] = case
    if solver_route is not None:
        measurement["solver_route"] = solver_route
    if maxiter is not None:
        measurement["maxiter"] = maxiter
    if capture_boozer_trial_trace:
        payload["capture_boozer_trial_trace"] = True
    (output / "measurements.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_provider_child_timeout_is_fail_closed(monkeypatch) -> None:
    child = _Child()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(runtime, "_PROVIDER_CHILD_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(runtime, "_PROVIDER_CHILD_POLL_SECONDS", 0)

    with pytest.raises(RuntimeError, match="second watchdog"):
        runtime._run_provider_child_process(["provider"])


def test_boozer_provider_uses_the_explicit_extended_watchdog() -> None:
    assert runtime._provider_child_timeout_seconds("boozer") == 1800
    assert runtime._provider_child_timeout_seconds("coil47,boozer") == 1800
    assert runtime._provider_child_timeout_seconds("coil47") == 120


def test_provider_child_rss_limit_is_fail_closed(monkeypatch) -> None:
    child = _Child()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(runtime, "_PROVIDER_CHILD_TIMEOUT_SECONDS", 120)
    monkeypatch.setattr(
        runtime, "_child_rss_kib", lambda _pid: runtime._PROVIDER_CHILD_RSS_LIMIT_KIB
    )

    with pytest.raises(RuntimeError, match="8-GiB RSS watchdog"):
        runtime._run_provider_child_process(["provider"])


def test_provider_child_discards_unbounded_stdout(monkeypatch) -> None:
    child = _Child()
    child.returncode = 0
    calls: dict[str, object] = {}

    def fake_popen(*_args, **kwargs):
        calls.update(kwargs)
        return child

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)

    runtime._run_provider_child_process(["provider"])

    assert calls["stdout"] is runtime.subprocess.DEVNULL


def test_provider_process_monitor_receives_the_exact_direct_child_pid_and_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _Child()
    child.returncode = 0
    observed: dict[str, object] = {}

    class Monitor:
        def __init__(self, *, gpu_uuid: str, provider_pid: int) -> None:
            observed.update(gpu_uuid=gpu_uuid, provider_pid=provider_pid)

        def start(self) -> None:
            observed["started"] = True

        def finish(self) -> ProcessGpuMemoryResult:
            observed["finished"] = True
            return ProcessGpuMemoryResult(
                gpu_uuid="GPU-authenticated",
                provider_pid=child.pid,
                samples=(ProcessGpuMemorySample(123, 640),),
                peak_used_memory_mib=640,
            )

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(runtime, "ProcessGpuMemoryMonitor", Monitor)

    measurement = runtime._run_provider_child_process(
        ["provider"], gpu_uuid="GPU-authenticated"
    )

    assert observed == {
        "gpu_uuid": "GPU-authenticated",
        "provider_pid": child.pid,
        "started": True,
        "finished": True,
    }
    assert isinstance(measurement, ProcessGpuMemoryResult)
    assert measurement.peak_used_memory_mib == 640


def test_gpu_child_rows_retain_observed_peak_and_memory_artifact_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "custom"
    identity = {
        "requested_device": "gpu",
        "backend": "gpu",
        "jax_device": "cuda:0",
        "gpu_uuid": "GPU-authenticated",
    }
    selected = runtime._NvidiaSmiIdentity(
        index=0,
        uuid="GPU-authenticated",
        model="NVIDIA A100",
        total_memory_bytes=40960 * 1024 * 1024,
        driver_version="590.48",
        compute_capability="8.0",
    )
    measurement = ProcessGpuMemoryResult(
        gpu_uuid=selected.uuid,
        provider_pid=1234,
        samples=(
            ProcessGpuMemorySample(100, 256),
            ProcessGpuMemorySample(200, 768),
        ),
        peak_used_memory_mib=768,
    )

    monkeypatch.setattr(runtime, "_selected_nvidia_smi_identity", lambda _id: selected)

    def run_child(*_args, **kwargs):
        assert kwargs["gpu_uuid"] == selected.uuid
        _write_minimal_child_payload(output, device="gpu", device_identity=identity)
        return measurement

    monkeypatch.setattr(runtime, "_run_provider_child_process", run_child)
    monkeypatch.setattr(
        runtime.jax,
        "default_backend",
        lambda: pytest.fail("parent must not initialize JAX"),
    )
    monkeypatch.setattr(
        runtime.jax,
        "devices",
        lambda: pytest.fail("parent must not enumerate JAX devices"),
    )

    rows, provenance = runtime._run_provider_child(
        provider="custom",
        cases="coil47",
        device="gpu",
        intent="fast",
        method="lbfgs",
        maxiter=20,
        maxcor=10,
        output=output,
    )

    artifact_path = output / "gpu_memory.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    child_payload = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    assert artifact == {
        "availability": "available",
        "gpu_uuid": selected.uuid,
        "peak_used_memory_mib": 768,
        "provider_pid": 1234,
        "samples": [
            {"sampled_at_unix_ns": 100, "used_memory_mib": 256},
            {"sampled_at_unix_ns": 200, "used_memory_mib": 768},
        ],
        "schema_version": 1,
        "target_pid_observed": True,
        "unavailable_reason": None,
    }
    assert rows == child_payload["measurements"]
    assert rows[0]["peak_vram_mib"] == 768
    assert rows[0]["diagnostic_artifacts"]["memory_trace"] == "gpu_memory.json"
    assert provenance["gpu_memory_path"] == "custom/gpu_memory.json"
    assert provenance["gpu_memory_sha256"] == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    assert provenance["measurements_sha256"] == hashlib.sha256(
        (output / "measurements.json").read_bytes()
    ).hexdigest()


def test_parent_forwards_and_binds_opt_in_boozer_trial_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "custom"
    identity = {
        "requested_device": "cpu",
        "backend": "cpu",
        "jax_device": "TFRT_CPU_0",
        "gpu_uuid": None,
    }
    observed_validation: dict[str, object] = {}

    def run_child(command, **kwargs):
        assert command.count("--capture-boozer-trial-trace") == 1
        assert kwargs["gpu_uuid"] is None
        output.mkdir(parents=True, exist_ok=True)
        trial_trace = output / "boozer_trial_trace.json"
        trial_trace.write_text('{"schema_version": 1}\n', encoding="utf-8")
        _write_minimal_child_payload(
            output,
            device="cpu",
            device_identity=identity,
            method="bfgs",
            case="boozer",
            solver_route="custom_bfgs_stepwise",
            maxiter=2,
            trial_trace=trial_trace.name,
            capture_boozer_trial_trace=True,
        )
        return ProcessGpuMemoryUnavailable(
            reason="cpu-device",
            gpu_uuid=None,
            provider_pid=1234,
        )

    def validate_trace(manifest_path: Path, **kwargs):
        observed_validation.update(manifest_path=manifest_path, **kwargs)
        return SimpleNamespace(record_count=1)

    monkeypatch.setattr(runtime, "_run_provider_child_process", run_child)
    monkeypatch.setattr(runtime, "validate_boozer_trial_trace", validate_trace)

    rows, provenance = runtime._run_provider_child(
        provider="custom",
        cases="boozer",
        device="cpu",
        intent="parity",
        method="bfgs",
        maxiter=2,
        maxcor=10,
        output=output,
        capture_boozer_trial_trace=True,
    )

    trial_trace_path = output / "boozer_trial_trace.json"
    assert rows[0]["diagnostic_artifacts"]["trial_trace"] == trial_trace_path.name
    assert provenance["trial_trace_path"] == "custom/boozer_trial_trace.json"
    assert provenance["trial_trace_sha256"] == hashlib.sha256(
        trial_trace_path.read_bytes()
    ).hexdigest()
    assert observed_validation == {
        "manifest_path": trial_trace_path,
        "expected_provider": "custom",
        "expected_production_route": "custom_bfgs_stepwise",
        "expected_maxiter": 2,
    }


@pytest.mark.parametrize(
    ("child_route", "child_maxiter", "message"),
    (
        ("wrong-route", 2, "solver route"),
        ("custom_bfgs_stepwise", 3, "maxiter"),
    ),
)
def test_parent_rejects_child_boozer_trial_trace_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    child_route: str,
    child_maxiter: int,
    message: str,
) -> None:
    output = tmp_path / "custom"
    identity = {
        "requested_device": "cpu",
        "backend": "cpu",
        "jax_device": "TFRT_CPU_0",
        "gpu_uuid": None,
    }

    def run_child(*_args, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "boozer_trial_trace.json").write_text(
            '{"schema_version": 1}\n', encoding="utf-8"
        )
        _write_minimal_child_payload(
            output,
            device="cpu",
            device_identity=identity,
            method="bfgs",
            case="boozer",
            solver_route=child_route,
            maxiter=child_maxiter,
            trial_trace="boozer_trial_trace.json",
            capture_boozer_trial_trace=True,
        )
        return ProcessGpuMemoryUnavailable(
            reason="cpu-device",
            gpu_uuid=None,
            provider_pid=1234,
        )

    monkeypatch.setattr(runtime, "_run_provider_child_process", run_child)
    monkeypatch.setattr(
        runtime,
        "validate_boozer_trial_trace",
        lambda *_args, **_kwargs: SimpleNamespace(record_count=1),
    )

    with pytest.raises(ValueError, match=message):
        runtime._run_provider_child(
            provider="custom",
            cases="boozer",
            device="cpu",
            intent="parity",
            method="bfgs",
            maxiter=2,
            maxcor=10,
            output=output,
            capture_boozer_trial_trace=True,
        )


def test_capture_disabled_preserves_child_command_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "custom"
    identity = {
        "requested_device": "cpu",
        "backend": "cpu",
        "jax_device": "TFRT_CPU_0",
        "gpu_uuid": None,
    }

    def run_child(command, **kwargs):
        assert "--capture-boozer-trial-trace" not in command
        assert kwargs["gpu_uuid"] is None
        _write_minimal_child_payload(
            output,
            device="cpu",
            device_identity=identity,
        )
        return ProcessGpuMemoryUnavailable(
            reason="cpu-device",
            gpu_uuid=None,
            provider_pid=1234,
        )

    monkeypatch.setattr(runtime, "_run_provider_child_process", run_child)
    monkeypatch.setattr(
        runtime,
        "validate_boozer_trial_trace",
        lambda *_args, **_kwargs: pytest.fail("capture-off must not validate a trace"),
    )

    rows, provenance = runtime._run_provider_child(
        provider="custom",
        cases="coil47",
        device="cpu",
        intent="parity",
        method="lbfgs",
        maxiter=2,
        maxcor=10,
        output=output,
    )

    child_payload = json.loads((output / "measurements.json").read_text())
    assert "capture_boozer_trial_trace" not in child_payload
    assert rows[0]["diagnostic_artifacts"]["trial_trace"] is None
    assert "trial_trace_path" not in provenance
    assert "trial_trace_sha256" not in provenance


@pytest.mark.parametrize(
    ("device", "gpu_uuid", "reason"),
    (
        ("cpu", None, "cpu-device"),
        ("gpu", "GPU-authenticated", "provider-pid-not-observed"),
    ),
)
def test_unavailable_gpu_memory_is_retained_without_an_inferred_peak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device: str,
    gpu_uuid: str | None,
    reason: str,
) -> None:
    output = tmp_path / "custom"
    identity = {
        "requested_device": device,
        "backend": device,
        "jax_device": "TFRT_CPU_0" if device == "cpu" else "cuda:0",
        "gpu_uuid": gpu_uuid,
    }
    selected = runtime._NvidiaSmiIdentity(
        index=0,
        uuid="GPU-authenticated",
        model="NVIDIA A100",
        total_memory_bytes=40960 * 1024 * 1024,
        driver_version="590.48",
        compute_capability="8.0",
    )
    if device == "cpu":
        monkeypatch.setattr(
            runtime,
            "_selected_nvidia_smi_identity",
            lambda _id: pytest.fail("CPU must not query NVIDIA identity"),
        )
    else:
        monkeypatch.setattr(
            runtime, "_selected_nvidia_smi_identity", lambda _id: selected
        )

    def run_child(*_args, **kwargs):
        assert kwargs["gpu_uuid"] == gpu_uuid
        _write_minimal_child_payload(output, device=device, device_identity=identity)
        return ProcessGpuMemoryUnavailable(
            reason=reason,
            gpu_uuid=gpu_uuid,
            provider_pid=1234,
        )

    monkeypatch.setattr(runtime, "_run_provider_child_process", run_child)

    rows, provenance = runtime._run_provider_child(
        provider="custom",
        cases="coil47",
        device=device,
        intent="fast",
        method="lbfgs",
        maxiter=20,
        maxcor=10,
        output=output,
    )

    artifact = json.loads((output / "gpu_memory.json").read_text(encoding="utf-8"))
    assert artifact["availability"] == "unavailable"
    assert artifact["unavailable_reason"] == reason
    assert artifact["samples"] == []
    assert artifact["peak_used_memory_mib"] is None
    assert rows[0]["peak_vram_mib"] is None
    assert rows[0]["diagnostic_artifacts"]["memory_trace"] == "gpu_memory.json"
    assert provenance["gpu_memory_path"] == "custom/gpu_memory.json"


def test_parent_rejects_child_identity_that_differs_from_monitored_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "custom"
    selected = runtime._NvidiaSmiIdentity(
        index=0,
        uuid="GPU-authenticated",
        model="NVIDIA A100",
        total_memory_bytes=40960 * 1024 * 1024,
        driver_version="590.48",
        compute_capability="8.0",
    )
    monkeypatch.setattr(runtime, "_selected_nvidia_smi_identity", lambda _id: selected)

    def run_child(*_args, **_kwargs):
        _write_minimal_child_payload(
            output,
            device="gpu",
            device_identity={
                "requested_device": "gpu",
                "backend": "gpu",
                "jax_device": "cuda:0",
                "gpu_uuid": "GPU-different",
            },
        )
        return ProcessGpuMemoryResult(
            gpu_uuid=selected.uuid,
            provider_pid=1234,
            samples=(ProcessGpuMemorySample(100, 256),),
            peak_used_memory_mib=256,
        )

    monkeypatch.setattr(runtime, "_run_provider_child_process", run_child)

    with pytest.raises(ValueError, match="monitored GPU UUID"):
        runtime._run_provider_child(
            provider="custom",
            cases="coil47",
            device="gpu",
            intent="fast",
            method="lbfgs",
            maxiter=20,
            maxcor=10,
            output=output,
        )


def test_monitor_failure_prevents_child_payload_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = runtime._NvidiaSmiIdentity(
        index=0,
        uuid="GPU-authenticated",
        model="NVIDIA A100",
        total_memory_bytes=40960 * 1024 * 1024,
        driver_version="590.48",
        compute_capability="8.0",
    )
    monkeypatch.setattr(runtime, "_selected_nvidia_smi_identity", lambda _id: selected)
    monkeypatch.setattr(
        runtime,
        "_run_provider_child_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProcessGpuMemoryMonitorError("malformed GPU sample")
        ),
    )

    with pytest.raises(ProcessGpuMemoryMonitorError, match="malformed GPU sample"):
        runtime._run_provider_child(
            provider="custom",
            cases="coil47",
            device="gpu",
            intent="fast",
            method="lbfgs",
            maxiter=20,
            maxcor=10,
            output=tmp_path / "custom",
        )
    assert not (tmp_path / "custom" / "gpu_memory.json").exists()


@pytest.mark.parametrize(
    ("iterations", "maxiter", "status", "success", "finite", "expected"),
    [
        (4, 20, 0, True, True, "converged"),
        (20, 20, 1, False, True, "iteration-limit"),
        (3, 20, 2, False, True, "line-search-failed"),
        (3, 20, 6, False, True, "nonfinite"),
        (3, 20, 99, False, True, "callback-stopped"),
        (3, 20, None, False, True, "failed"),
        (3, 20, 1, False, False, "nonfinite"),
    ],
)
def test_stopping_reason_labels_terminal_state(
    iterations: int,
    maxiter: int,
    status: int | None,
    success: bool,
    finite: bool,
    expected: str,
) -> None:
    assert (
        runtime._stopping_reason(
            iterations=iterations,
            maxiter=maxiter,
            status=status,
            success=success,
            finite=finite,
        )
        == expected
    )


def test_runtime_environment_payload_records_device_allocator_contract(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    payload = runtime._runtime_environment_payload()

    assert payload["JAX_PLATFORMS"] == "cuda"
    assert payload["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert payload["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    assert "SIMSOPT_BACKEND_MODE" in payload


def test_runner_schema_v7_declares_the_promotion_contract() -> None:
    assert runtime._RUNNER_SCHEMA_VERSION == 7


def test_nvidia_smi_identity_parser_binds_uuid_model_memory_and_driver() -> None:
    records = runtime._parse_nvidia_smi_identity_rows(
        "0, GPU-abc, NVIDIA GeForce RTX 5090, 32607, 590.48, 12.0\n"
        "1, GPU-def, NVIDIA A100-PCIE-40GB, 40960, 590.48, 8.0\n"
    )

    assert records[0].index == 0
    assert records[0].uuid == "GPU-abc"
    assert records[0].model == "NVIDIA GeForce RTX 5090"
    assert records[0].total_memory_bytes == 32607 * 1024 * 1024
    assert records[0].driver_version == "590.48"
    assert records[0].compute_capability == "12.0"


@pytest.mark.parametrize(
    ("device", "intent", "mode"),
    (
        ("cpu", "fast", "jax_cpu_fast"),
        ("cpu", "parity", "jax_cpu_parity"),
        ("gpu", "fast", "jax_gpu_fast"),
        ("gpu", "parity", "jax_gpu_parity"),
    ),
)
def test_intent_environment_requires_canonical_profile(
    monkeypatch,
    device: str,
    intent: str,
    mode: str,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", mode)

    assert runtime._validate_intent_environment(device, intent) == mode


def test_intent_environment_rejects_missing_or_mismatched_profile(monkeypatch) -> None:
    monkeypatch.delenv("SIMSOPT_BACKEND_MODE", raising=False)
    with pytest.raises(RuntimeError, match="expected 'jax_cpu_parity'"):
        runtime._validate_intent_environment("cpu", "parity")


def test_main_rejects_intent_before_fixture_construction(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_fast")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_quasi_newton_runtime.py",
            "--device",
            "cpu",
            "--intent",
            "parity",
            "--output",
            str(tmp_path),
        ],
    )

    def fixture_must_not_run(_name: str) -> Fixture:
        pytest.fail("profile validation must run before fixture construction")

    monkeypatch.setattr(runtime, "fixture", fixture_must_not_run)
    with pytest.raises(RuntimeError, match="expected 'jax_cpu_parity'"):
        runtime.main()

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_fast")
    with pytest.raises(RuntimeError, match="expected 'jax_cpu_parity'"):
        runtime._validate_intent_environment("cpu", "parity")


def test_parent_orchestrator_does_not_construct_fixture_or_initialize_jax(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    output = tmp_path / "parent"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_quasi_newton_runtime.py",
            "--device",
            "gpu",
            "--intent",
            "fast",
            "--providers",
            "custom",
            "--cases",
            "coil47",
            "--method",
            "lbfgs",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        runtime,
        "fixture",
        lambda _name: pytest.fail("parent must not construct a fixture"),
    )
    monkeypatch.setattr(
        runtime.jax,
        "default_backend",
        lambda: pytest.fail("parent must not initialize the JAX backend"),
    )
    monkeypatch.setattr(
        runtime.jax,
        "devices",
        lambda: pytest.fail("parent must not enumerate JAX devices"),
    )
    identity = {
        "requested_device": "gpu",
        "backend": "gpu",
        "jax_device": "cuda:0",
    }
    commit, clean = runtime._checkout_provenance()
    monkeypatch.setattr(
        runtime,
        "_run_provider_child",
        lambda **_kwargs: (
            [{"device_identity": identity}],
            {
                "provider": "custom",
                "measurements_path": "custom/measurements.json",
                "measurements_sha256": "0" * 64,
                "measurement_count": 1,
                "git_commit": commit,
                "git_clean": clean,
                "runtime_environment": runtime._runtime_environment_payload(),
                "requested_device": "gpu",
                "method": "lbfgs",
                "device_identity": identity,
            },
        ),
    )

    assert runtime.main() == 0
    payload = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    assert payload["backend"] == "gpu"
    assert payload["devices"] == ["cuda:0"]
    assert payload["provider_children"][0]["git_commit"] == commit


def test_provider_child_opt_in_writes_and_attaches_boozer_trial_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "provider-child"
    identity = runtime.DeviceIdentity(
        requested_device="cpu",
        backend="cpu",
        platform="cpu",
        jax_device="TFRT_CPU_0",
        device_kind="cpu",
        device_id=0,
        process_index=0,
        gpu_uuid=None,
        gpu_model=None,
        compute_capability=None,
        total_memory_bytes=None,
        driver_version=None,
        cuda_version=None,
        visible_devices=None,
        hostname="test-host",
        scheduler_job_id=None,
    )
    fixture_case = Fixture(
        name="boozer",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_boozer_capture_contract",
        certificate="synthetic",
        method="bfgs",
    )
    row = {
        "case": "boozer",
        "provider": "custom",
        "solver_route": "custom_bfgs_stepwise",
        "maxiter": 2,
        "device_identity": {
            "requested_device": "cpu",
            "backend": "cpu",
            "jax_device": "TFRT_CPU_0",
            "gpu_uuid": None,
        },
        "diagnostic_artifacts": {"memory_trace": None, "trial_trace": None},
    }
    diagnostic_calls: list[dict[str, object]] = []

    def run_diagnostic(selected_fixture: Fixture, **kwargs):
        diagnostic_calls.append({"fixture": selected_fixture, **kwargs})
        manifest_path = kwargs["manifest_path"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
        return SimpleNamespace(trial_trace=manifest_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_quasi_newton_runtime.py",
            "--device",
            "cpu",
            "--intent",
            "parity",
            "--providers",
            "custom",
            "--cases",
            "boozer",
            "--method",
            "bfgs",
            "--maxiter",
            "2",
            "--output",
            str(output),
            "--provider-child",
            "--capture-boozer-trial-trace",
        ],
    )
    monkeypatch.setattr(runtime, "_validate_intent_environment", lambda *_args: "cpu")
    monkeypatch.setattr(runtime, "fixture_method", lambda _name: "bfgs")
    monkeypatch.setattr(runtime, "fixture", lambda _name: fixture_case)
    monkeypatch.setattr(runtime, "_measurement", lambda *_args, **_kwargs: row)
    monkeypatch.setattr(runtime, "run_boozer_host_diagnostic", run_diagnostic)
    monkeypatch.setattr(runtime.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        runtime.jax,
        "devices",
        lambda: (SimpleNamespace(platform="cpu"),),
    )
    monkeypatch.setattr(runtime, "_device_identity", lambda _device: identity)
    monkeypatch.setattr(runtime, "_checkout_provenance", lambda: ("commit", False))

    assert runtime.main() == 0

    payload = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    assert payload["capture_boozer_trial_trace"] is True
    assert payload["measurements"][0]["diagnostic_artifacts"]["trial_trace"] == (
        "boozer_trial_trace.json"
    )
    assert len(diagnostic_calls) == 1
    assert diagnostic_calls[0]["fixture"] is fixture_case
    assert diagnostic_calls[0]["provider"] == "custom"
    assert diagnostic_calls[0]["maxiter"] == 2


@pytest.mark.parametrize(
    "receipt_name",
    (
        "rosenbrock-pre-refactor-trajectory",
        "bfgs-pre-refactor-trajectory",
    ),
)
def test_tracked_pre_refactor_trajectory_receipt_is_self_consistent(
    receipt_name: str,
) -> None:
    receipt = (
        Path(__file__).resolve().parents[2]
        / "docs/receipts/custom-quasi-newton"
        / receipt_name
    )
    manifest = json.loads((receipt / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["verdict"] == "diagnostic-pass-not-promotion"
    for artifact in manifest["artifacts"]:
        path = receipt / artifact["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]

    pre_refactor = (receipt / "raw/pre_refactor.json").read_bytes()
    candidate = (receipt / "raw/candidate_worktree.json").read_bytes()
    assert pre_refactor == candidate


def test_provider_child_timeout_override_wins_and_validates() -> None:
    assert runtime._provider_child_timeout_seconds("rosenbrock", None) == 120
    assert runtime._provider_child_timeout_seconds("boozer", None) == 1800
    assert runtime._provider_child_timeout_seconds("boozer", 21600) == 21600
    with pytest.raises(ValueError, match="positive second count"):
        runtime._provider_child_timeout_seconds("boozer", 0)


def test_measurement_records_fixture_build_costs() -> None:
    measurement = runtime._measurement(
        fixture("rosenbrock"),
        "custom",
        "cpu",
        "parity",
        np.asarray([-1.2, 1.0], dtype=np.float64),
        maxiter=1,
        maxcor=10,
        method="lbfgs",
        fixture_build_seconds=1.25,
        fixture_build_peak_rss_kib=123,
    )

    assert measurement.fixture_build_seconds == 1.25
    assert measurement.fixture_build_peak_rss_kib == 123
    assert measurement.solver_route == "stepwise"
    assert measurement.preparation_seconds >= 0.0
    assert measurement.first_execution_seconds >= 0.0
    assert measurement.cold_seconds == pytest.approx(
        measurement.preparation_seconds + measurement.first_execution_seconds
    )
    assert measurement.device_identity.requested_device == "cpu"
    assert measurement.device_identity.platform == "cpu"
    assert measurement.work_counters.accepted_iterations == measurement.iterations
    assert measurement.work_counters.transfer_calls >= 1
    assert measurement.diagnostic_artifacts == {
        "memory_trace": None,
        "trial_trace": None,
    }
    assert measurement.endpoint_certificate.stopping_reason == (
        measurement.stopping_reason
    )
    assert measurement.scientific_observables == {}
    assert measurement.scientific_certification_seconds >= 0.0
    assert measurement.solver_start_rss_kib >= 0
    assert measurement.solver_peak_rss_kib >= measurement.solver_start_rss_kib
    assert measurement.solver_peak_rss_delta_kib >= 0
    assert measurement.initial_parameters == (-1.2, 1.0)
    assert isinstance(measurement.final_parameters, tuple)
    assert jnp.isfinite(measurement.final_objective)
    assert measurement.warm_transfer_audit
    assert {entry.phase for entry in measurement.warm_transfer_audit} >= {
        "advance",
        "final_result",
    }
    assert measurement.fixture_contract["final_certificate_fields"]
    assert measurement.fixture_contract["generator_sha256"]
    assert {entry.phase for entry in measurement.phase_rss} == {
        "preparation",
        "cold_solver",
        "warm_solver",
    }
    assert all(entry.sample_count >= 2 for entry in measurement.phase_rss)
    assert all(
        entry.peak_rss_kib >= entry.start_rss_kib
        and entry.peak_rss_kib >= entry.end_rss_kib
        for entry in measurement.phase_rss
    )


@pytest.mark.parametrize(
    ("intent", "expected_route", "expected_run_mode"),
    (("parity", "stepwise", "stepwise"), ("fast", "fused_stepwise", "fused_stepwise")),
)
def test_custom_lbfgs_route_and_prepared_program_follow_intent(
    intent: str,
    expected_route: str,
    expected_run_mode: str,
) -> None:
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_custom(
        fixture_case,
        initial,
        maxcor=3,
        run_mode=expected_run_mode,
    )

    assert runtime._solver_route("custom", "lbfgs", intent=intent) == expected_route
    assert prepared.program.run_mode == expected_run_mode


def test_fast_custom_lbfgs_has_zero_advance_observations() -> None:
    fixture_case = runtime._measurement(
        fixture("rosenbrock"),
        "custom",
        "cpu",
        "fast",
        np.asarray([-1.2, 1.0], dtype=np.float64),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert fixture_case.solver_route == "fused_stepwise"
    assert fixture_case.work_counters.advance_observations == 0
    assert fixture_case.work_counters.advance_observations <= fixture_case.iterations + 1
    assert sum(
        entry.calls
        for entry in fixture_case.warm_transfer_audit
        if entry.phase == "advance"
    ) == 0


def test_rss_phase_records_named_scope() -> None:
    with runtime._RSSPhase("test") as phase:
        np.zeros(1024, dtype=np.float64)

    measurement = phase.measurement()
    assert measurement.phase == "test"
    assert measurement.scope == "self_proc_status_poll_10ms"
    assert measurement.sample_count >= 2


def test_bfgs_memory_analysis_has_a_separate_rss_phase() -> None:
    fixture_case = fixture("bfgs_quadratic")
    measurement = runtime._measurement(
        fixture_case,
        "custom",
        "cpu",
        "parity",
        np.asarray(fixture_case.initial, dtype=np.float64),
        maxiter=1,
        maxcor=10,
        method="bfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert {entry.phase for entry in measurement.phase_rss} == {
        "algorithm_memory_analysis",
        "cold_solver",
        "warm_solver",
    }


def test_measurement_passes_prepared_custom_program_to_both_runs(monkeypatch) -> None:
    original_run_custom = runtime._run_custom
    prepared_runs: list[object | None] = []

    def recording_run_custom(*args, **kwargs):
        prepared_runs.append(kwargs.get("prepared"))
        return original_run_custom(*args, **kwargs)

    monkeypatch.setattr(runtime, "_run_custom", recording_run_custom)
    fixture_case = fixture("rosenbrock")
    runtime._measurement(
        fixture_case,
        "custom",
        "cpu",
        "parity",
        np.asarray(fixture_case.initial, dtype=np.float64),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert len(prepared_runs) == 2
    assert all(prepared is not None for prepared in prepared_runs)


def test_measurement_native_provider_has_no_prepared_argument() -> None:
    fixture_case = fixture("rosenbrock")
    measurement = runtime._measurement(
        fixture_case,
        "native",
        "cpu",
        "parity",
        np.asarray(fixture_case.initial, dtype=np.float64),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert measurement.provider == "native"


def test_native_measurement_resets_mutable_provider_between_runs() -> None:
    state = {
        "evaluations": 0,
        "resets": 0,
        "first_evaluation_after_reset": [],
        "awaiting_first_evaluation": False,
    }

    def native_value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        if state["awaiting_first_evaluation"]:
            state["first_evaluation_after_reset"].append(state["evaluations"])
            state["awaiting_first_evaluation"] = False
        state["evaluations"] += 1
        value = float(np.sum(np.square(x)) + state["evaluations"])
        return value, 2.0 * np.asarray(x, dtype=np.float64)

    def reset_native() -> None:
        state["evaluations"] = 0
        state["resets"] += 1
        state["awaiting_first_evaluation"] = True

    fixture_case = Fixture(
        name="stateful_native",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_stateful_native",
        certificate="synthetic native reset contract",
        method="lbfgs",
        native_value_and_grad=native_value_and_grad,
        native_reset=reset_native,
    )

    measurement = runtime._measurement(
        fixture_case,
        "native",
        "cpu",
        "parity",
        fixture_case.initial.copy(),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert state["resets"] == 2
    assert state["first_evaluation_after_reset"] == [0, 0]
    assert np.isfinite(measurement.final_objective)


def test_measurement_classifies_nonfinite_endpoint_before_iteration_limit(
    monkeypatch,
) -> None:
    fixture_case = Fixture(
        name="synthetic_nonfinite_endpoint",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_nonfinite_endpoint",
        certificate="synthetic endpoint classification contract",
        method="lbfgs",
    )
    result = SimpleNamespace(
        f_k=jnp.asarray(1.0, dtype=jnp.float64),
        g_k=jnp.asarray([np.nan], dtype=jnp.float64),
        k=jnp.asarray(1),
        x_k=jnp.asarray([np.inf], dtype=jnp.float64),
    )

    monkeypatch.setattr(
        runtime,
        "_initial_value_and_grad",
        lambda *_args, **_kwargs: (1.0, np.asarray([1.0], dtype=np.float64)),
    )
    monkeypatch.setattr(runtime, "_prepare_custom", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runtime,
        "_run_custom",
        lambda *_args, **_kwargs: (result, 1, 1, False, (), None),
    )

    measurement = runtime._measurement(
        fixture_case,
        "custom",
        "cpu",
        "parity",
        fixture_case.initial.copy(),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert measurement.stopping_reason == "nonfinite"
    certificate = measurement.fixture_contract["final_certificate_fields"]
    assert isinstance(certificate, dict)
    assert certificate["stopping_reason"] == "nonfinite"


def test_optax_comparator_uses_a_jitted_step(monkeypatch) -> None:
    jit_calls = 0
    original_jit = runtime.jax.jit

    def counted_jit(fun, *args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(fun, *args, **kwargs)

    monkeypatch.setattr(runtime.jax, "jit", counted_jit)
    result = runtime._run_optax(
        fixture("rosenbrock"),
        np.asarray([-1.2, 1.0], dtype=np.float64),
        maxiter=1,
        maxcor=3,
    )

    assert jit_calls == 2
    assert result[0] is not None


def test_optax_prepared_program_reuses_compiled_step(monkeypatch) -> None:
    jit_calls = 0
    original_jit = runtime.jax.jit

    def counted_jit(fun, *args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(fun, *args, **kwargs)

    monkeypatch.setattr(runtime.jax, "jit", counted_jit)
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_optax(fixture_case, initial, maxcor=3)

    first = runtime._run_optax(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        prepared=prepared,
    )
    second = runtime._run_optax(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        prepared=prepared,
    )

    assert jit_calls == 2
    np.testing.assert_array_equal(np.asarray(first[0][0]), np.asarray(second[0][0]))
    assert first[0][3] == second[0][3]


def test_optax_prepared_program_rejects_mismatched_inputs() -> None:
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_optax(fixture_case, initial, maxcor=3)

    with pytest.raises(ValueError, match="does not match"):
        runtime._run_optax(
            fixture_case,
            initial + np.asarray([0.1, 0.0], dtype=np.float64),
            maxiter=1,
            maxcor=3,
            prepared=prepared,
        )


def test_custom_prepared_program_rejects_mismatched_inputs() -> None:
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_custom(fixture_case, initial, maxcor=3)

    with pytest.raises(ValueError, match="does not match"):
        runtime._run_custom(
            fixture_case,
            initial + np.asarray([0.1, 0.0], dtype=np.float64),
            maxiter=1,
            maxcor=3,
            method="lbfgs",
            prepared=prepared,
        )


def test_custom_boozer_bfgs_mints_fresh_accepted_incumbent_controllers() -> None:
    controllers: list[SimpleNamespace] = []

    class Controller:
        def __init__(self) -> None:
            self.pending: np.ndarray | None = None
            self.accepted: list[np.ndarray] = []

        def value_and_grad(self, parameters: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = np.asarray(parameters, dtype=np.float64)
            self.pending = candidate.copy()
            delta = candidate - 0.25
            return float(np.dot(delta, delta)), 2.0 * delta

        def accept(self, parameters: np.ndarray) -> None:
            candidate = np.asarray(parameters, dtype=np.float64)
            np.testing.assert_array_equal(candidate, self.pending)
            self.accepted.append(candidate.copy())
            self.pending = None

    def controller_factory() -> Controller:
        controller = Controller()
        controllers.append(SimpleNamespace(controller=controller))
        return controller

    fixture_case = Fixture(
        name="boozer",
        objective=lambda x: jnp.sum((x - 0.25) ** 2),
        initial=np.asarray([2.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_boozer_incumbent_contract",
        certificate="synthetic",
        method="bfgs",
        accepted_incumbent_host_value_and_grad=controller_factory,
    )

    first = runtime._run_custom(
        fixture_case,
        fixture_case.initial,
        maxiter=2,
        maxcor=3,
        method="bfgs",
    )
    second = runtime._run_custom(
        fixture_case,
        fixture_case.initial,
        maxiter=2,
        maxcor=3,
        method="bfgs",
    )

    assert len(controllers) == 2
    assert all(entry.controller.accepted for entry in controllers)
    assert first[0].k == second[0].k
    np.testing.assert_array_equal(first[0].x_k, second[0].x_k)


def test_custom_prepared_program_reuses_compiled_transitions(monkeypatch) -> None:
    prepare_calls = 0
    original_prepare = runtime.prepare_lbfgs_private

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(runtime, "prepare_lbfgs_private", counted_prepare)
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_custom(fixture_case, initial, maxcor=3)
    first = runtime._run_custom(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        prepared=prepared,
    )
    second = runtime._run_custom(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        prepared=prepared,
    )

    assert prepare_calls == 1
    np.testing.assert_array_equal(np.asarray(first[0].x_k), np.asarray(second[0].x_k))
    assert first[0].k == second[0].k


def test_optax_prepared_nan_input_reaches_nonfinite_status() -> None:
    fixture_case = Fixture(
        name="nan_input",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([np.nan], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_nonfinite",
        certificate="solver-runtime-only",
        method="lbfgs",
    )
    prepared = runtime._prepare_optax(
        fixture_case,
        fixture_case.initial,
        maxcor=3,
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial.copy(),
        maxiter=1,
        maxcor=3,
        prepared=prepared,
    )

    assert result[2] == 6
    assert result[3] is False


def test_optax_prepared_signed_zero_input_is_bound_exactly() -> None:
    objective = lambda x: jnp.where(jnp.signbit(x[0]), 1.0, 0.0)
    fixture_case = Fixture(
        name="signed_zero_input",
        objective=objective,
        initial=np.asarray([0.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_signed_zero",
        certificate="solver-runtime-only",
        method="lbfgs",
    )
    prepared = runtime._prepare_optax(
        fixture_case,
        fixture_case.initial,
        maxcor=3,
    )

    with pytest.raises(ValueError, match="does not match"):
        runtime._run_optax(
            fixture_case,
            np.asarray([-0.0], dtype=np.float64),
            maxiter=1,
            maxcor=3,
            prepared=prepared,
        )


def test_optax_stop_check_uses_gradient_after_update() -> None:
    fixture_case = Fixture(
        name="one_dimensional_quadratic",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_quadratic",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=2,
        maxcor=3,
    )

    assert result[0][3] == 1
    assert result[3] is True
    np.testing.assert_array_equal(np.asarray(result[0][0]), np.asarray([0.0]))


def test_optax_initial_terminal_state_takes_no_step() -> None:
    fixture_case = Fixture(
        name="already_converged_quadratic",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([0.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_quadratic",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=5,
        maxcor=3,
    )

    assert result[0][3] == 0
    assert result[2] == 0
    assert result[3] is True


def test_optax_nonfinite_zero_gradient_is_not_success() -> None:
    fixture_case = Fixture(
        name="nonfinite_constant",
        objective=lambda _x: jnp.asarray(jnp.inf, dtype=jnp.float64),
        initial=np.asarray([0.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_nonfinite",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=5,
        maxcor=3,
    )

    assert result[0][3] == 0
    assert result[2] == 6
    assert result[3] is False


def test_optax_line_search_failure_is_labeled() -> None:
    fixture_case = Fixture(
        name="linear_objective",
        objective=lambda x: x[0],
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_linear",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=3,
        maxcor=3,
    )

    assert result[0][3] == 1
    assert result[2] == 2
    assert result[3] is False


def test_fixture_contract_records_provenance_observables_and_tolerances() -> None:
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    initial_objective, initial_gradient = runtime._initial_value_and_grad(
        fixture_case,
        initial,
    )

    contract = runtime._fixture_contract_payload(
        fixture_case,
        initial,
        initial_objective=initial_objective,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        method="lbfgs",
        maxiter=20,
        maxcor=10,
        device="cpu",
        intent="parity",
    )

    assert contract["generator_sha256"]
    assert contract["source_sha256"]
    assert contract["initial_parameters"] == [-1.2, 1.0]
    assert contract["expected_initial_observables"]["objective"] == initial_objective
    assert contract["solver_options"] == {
        "device": "cpu",
        "ftol": 0.0,
        "gtol": 1.0e-10,
        "intent": "parity",
        "maxcor": 10,
        "maxfun": None,
        "maxiter": 20,
        "maxls": 20,
        "method": "lbfgs",
    }
    assert contract["tolerances"]


def test_dense_bfgs_memory_contract_reports_no_donation_upper_bound() -> None:
    contract = runtime._bfgs_memory_contract(47, np.float64)

    assert contract["inverse_hessian_bytes"] == 47 * 47 * 8
    assert contract["simultaneous_old_new_hessian_bytes"] == 2 * 47 * 47 * 8
    assert (
        contract["derived_peak_live_upper_bound_bytes"]
        > contract["simultaneous_old_new_hessian_bytes"]
    )
    assert contract["buffer_donation"] is False


def test_dense_bfgs_memory_analysis_is_update_only_and_bounded() -> None:
    report = runtime._dense_bfgs_update_memory_analysis(5, np.dtype(np.float64))
    contract = runtime._bfgs_memory_contract(5, np.float64)

    assert report["dense_update_compiled_memory_is_update_only"] is True
    assert report["dense_update_peak_live_bytes"] >= report["dense_update_output_bytes"]
    assert report["dense_update_temp_bytes"] >= 0
    if runtime.jax.default_backend() == "cpu":
        assert (
            report["dense_update_peak_live_bytes"]
            <= contract["derived_peak_live_upper_bound_bytes"]
        )
    else:
        # Device compiler temporaries are physical backend accounting, not the
        # logical no-donation bound derived from the algorithm's live arrays.
        assert report["dense_update_peak_live_bytes"] > 0


@pytest.mark.slow
def test_coil47_fixture_exposes_native_objective_callback() -> None:
    fixture_case = fixture("coil47")

    assert fixture_case.native_value_and_grad is not None
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    native_value, native_gradient = fixture_case.native_value_and_grad(initial)
    jax_value, jax_gradient = runtime._initial_value_and_grad(fixture_case, initial)

    assert native_value == pytest.approx(jax_value, abs=1.0e-12, rel=1.0e-12)
    np.testing.assert_allclose(native_gradient, jax_gradient, atol=1.0e-8, rtol=1.0e-10)


@pytest.mark.slow
def test_boozer_fixture_exposes_matched_native_objective_callback() -> None:
    fixture_case = fixture("boozer")

    assert fixture_case.native_value_and_grad is not None
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    native_value, native_gradient = fixture_case.native_value_and_grad(initial)
    jax_value, jax_gradient = runtime._initial_value_and_grad(fixture_case, initial)

    assert native_value == pytest.approx(jax_value, abs=1.0e-15, rel=1.0e-12)
    np.testing.assert_allclose(native_gradient, jax_gradient, atol=2.0e-12, rtol=2.0e-9)


def test_native_provider_rejects_unmatched_source_fixture() -> None:
    fixture_case = Fixture(
        name="unmatched",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="source_owned_unmatched",
        certificate="unmatched",
        method="bfgs",
    )

    with pytest.raises(ValueError, match="unmatched source-owned fixture"):
        runtime._initial_value_and_grad(
            fixture_case,
            fixture_case.initial,
            provider="native",
        )


def test_boozer_scientific_endpoint_uses_certified_traceable_route() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/fixtures/custom_quasi_newton.py"
    )
    tree = ast.parse(fixture_path.read_text(encoding="utf-8"))
    endpoint = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "scientific_endpoint"
    )
    endpoint_calls = {
        call.func.id
        for call in ast.walk(endpoint)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_certified_traceable_endpoint" in endpoint_calls
    assert "forward_result" not in endpoint_calls

    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_certified_traceable_endpoint"
    )
    helper_calls = {
        call.func.id
        for call in ast.walk(helper)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "run_code_traceable" in helper_calls
    assert "reporting_metrics_from_solution" in helper_calls


def test_certified_traceable_endpoint_fails_closed_on_inner_failure() -> None:
    observed: dict[str, object] = {}

    def coil_set_spec_from_dofs(coil_dofs):
        observed["coil_dofs"] = np.asarray(coil_dofs)
        return "endpoint-coil-spec"

    def run_code_traceable(
        coil_source,
        sdofs,
        iota,
        G,
        *,
        materialize_dense_linearization,
    ):
        observed["solve_args"] = (
            coil_source,
            np.asarray(sdofs),
            float(iota),
            float(G),
            materialize_dense_linearization,
        )
        return {
            "x": jnp.asarray([9.0], dtype=jnp.float64),
            "success": jnp.asarray(False),
        }

    def reporting_metrics_from_solution(*_args, **_kwargs):
        pytest.fail("failed inner solves must not produce reporting metrics")

    evidence = _certified_traceable_endpoint(
        np.asarray([1.5, -2.0], dtype=np.float64),
        coil_set_spec_from_dofs=coil_set_spec_from_dofs,
        run_code_traceable=run_code_traceable,
        install_traceable_solved_runtime_state=lambda _solved: pytest.fail(
            "failed inner solves must not install state"
        ),
        surface_dofs=np.asarray([0.25], dtype=np.float64),
        iota_seed=-0.406,
        G_seed=3.25,
        reporting_metrics_from_solution=reporting_metrics_from_solution,
    )

    assert evidence.inner_success is False
    assert evidence.observables == ()
    np.testing.assert_array_equal(observed["coil_dofs"], [1.5, -2.0])
    solve_args = observed["solve_args"]
    assert isinstance(solve_args, tuple)
    assert solve_args[0] == "endpoint-coil-spec"
    np.testing.assert_array_equal(solve_args[1], [0.25])
    assert solve_args[2:] == (-0.406, 3.25, False)
