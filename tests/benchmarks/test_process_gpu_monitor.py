"""Contract tests for direct-process NVIDIA GPU-memory sampling."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import FrozenInstanceError

import pytest
from benchmarks import process_gpu_monitor as gpu_monitor


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=("nvidia-smi",), returncode=0, stdout=stdout
    )


def _binary_completed(
    argv: tuple[str, ...], stdout: bytes, *, returncode: int = 0
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=argv,
        returncode=returncode,
        stdout=stdout,
        stderr=b"",
    )


def test_supervisor_gpu_zero_dual_query_proves_exact_pid_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    outputs = {
        gpu_monitor.SUPERVISOR_GPU_INVENTORY_QUERY: b"GPU-target, 32768\n",
        gpu_monitor.SUPERVISOR_COMPUTE_APPS_QUERY: (
            b"99, GPU-target, 512\n123, GPU-other, 1024\n"
        ),
    }
    monkeypatch.setattr(gpu_monitor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(gpu_monitor, "_sha256_file", lambda _path: "a" * 64)

    def run(
        argv: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return _binary_completed(argv, outputs[argv])

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-target",
        visible_device="GPU-target",
        supervisor_pid=123,
        supervisor_start_ticks=456,
    )

    assert calls == [
        gpu_monitor.SUPERVISOR_GPU_INVENTORY_QUERY,
        gpu_monitor.SUPERVISOR_COMPUTE_APPS_QUERY,
    ]
    assert observation.gate_passes is True
    assert observation.matching_rows == ()
    assert observation.physical_memory_bytes == 32768 * 1024 * 1024


def test_supervisor_gpu_zero_rejects_exact_parent_pid_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter((b"GPU-target, 32768\n", b"123, GPU-target, 0\n"))
    monkeypatch.setattr(gpu_monitor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(gpu_monitor, "_sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda argv, **_kwargs: _binary_completed(argv, next(outputs)),
    )

    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-target",
        visible_device="GPU-target",
        supervisor_pid=123,
        supervisor_start_ticks=456,
    )

    assert observation.gate_passes is False
    assert observation.matching_rows == (
        gpu_monitor.SupervisorComputeAppRow(123, "GPU-target", 0),
    )


def test_supervisor_gpu_zero_rejects_duplicate_compute_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        (
            b"GPU-target, 32768\n",
            b"99, GPU-target, 1\n99, GPU-target, 2\n",
        )
    )
    monkeypatch.setattr(gpu_monitor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(gpu_monitor, "_sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda argv, **_kwargs: _binary_completed(argv, next(outputs)),
    )

    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-target",
        visible_device="GPU-target",
        supervisor_pid=123,
        supervisor_start_ticks=456,
    )

    assert observation.parse_valid is False
    assert observation.compute_rows == ()
    assert observation.gate_passes is False


@pytest.mark.parametrize("failed_query", ("inventory", "compute"))
def test_supervisor_gpu_zero_retains_nonzero_query_and_never_trusts_rows(
    monkeypatch: pytest.MonkeyPatch,
    failed_query: str,
) -> None:
    outputs = {
        gpu_monitor.SUPERVISOR_GPU_INVENTORY_QUERY: b"GPU-target, 32768\n",
        gpu_monitor.SUPERVISOR_COMPUTE_APPS_QUERY: b"123, GPU-target, 1\n",
    }
    monkeypatch.setattr(gpu_monitor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(gpu_monitor, "_sha256_file", lambda _path: "a" * 64)

    def run(
        argv: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        is_failed = (
            failed_query == "inventory"
            and argv == gpu_monitor.SUPERVISOR_GPU_INVENTORY_QUERY
        ) or (
            failed_query == "compute"
            and argv == gpu_monitor.SUPERVISOR_COMPUTE_APPS_QUERY
        )
        return _binary_completed(
            argv,
            outputs[argv],
            returncode=7 if is_failed else 0,
        )

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-target",
        visible_device="GPU-target",
        supervisor_pid=123,
        supervisor_start_ticks=456,
    )

    failed = (
        observation.gpu_inventory_query
        if failed_query == "inventory"
        else observation.compute_apps_query
    )
    assert failed.returncode == 7
    assert failed.stdout == outputs[failed.argv]
    assert observation.parse_valid is False
    assert observation.matching_rows == ()
    assert observation.gate_passes is False


@pytest.mark.parametrize(
    ("inventory", "compute"),
    (
        (b"", b""),
        (b"GPU-target, 32768\nGPU-target, 32768\n", b""),
        (b"GPU-other, 32768\n", b""),
        (b"GPU-target, malformed\n", b""),
        (b"GPU-target, 32768\n", b"not-a-pid, GPU-target, 1\n"),
    ),
)
def test_supervisor_gpu_zero_rejects_empty_duplicate_wrong_and_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
    inventory: bytes,
    compute: bytes,
) -> None:
    outputs = iter((inventory, compute))
    monkeypatch.setattr(gpu_monitor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(gpu_monitor, "_sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda argv, **_kwargs: _binary_completed(argv, next(outputs)),
    )

    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-target",
        visible_device="GPU-target",
        supervisor_pid=123,
        supervisor_start_ticks=456,
    )

    assert observation.gate_passes is False


def test_supervisor_gpu_zero_retains_timeout_and_runs_both_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    monkeypatch.setattr(gpu_monitor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(gpu_monitor, "_sha256_file", lambda _path: "a" * 64)

    def run(
        argv: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(
                argv, 5.0, output=b"partial", stderr=b"late"
            )
        return _binary_completed(argv, b"")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-target",
        visible_device="GPU-target",
        supervisor_pid=123,
        supervisor_start_ticks=456,
    )

    assert calls == 2
    assert observation.gpu_inventory_query.timed_out is True
    assert observation.gpu_inventory_query.stdout == b"partial"
    assert observation.gate_passes is False


def test_supervisor_gpu_zero_retains_query_launch_failure_and_runs_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    monkeypatch.setattr(gpu_monitor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(gpu_monitor, "_sha256_file", lambda _path: "a" * 64)

    def run(
        argv: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("launch failed")
        return _binary_completed(argv, b"")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-target",
        visible_device="GPU-target",
        supervisor_pid=123,
        supervisor_start_ticks=456,
    )

    assert calls == 2
    assert observation.gpu_inventory_query.launched is False
    assert observation.gpu_inventory_query.returncode is None
    assert observation.compute_apps_query.launched is True
    assert observation.parse_valid is False
    assert observation.gate_passes is False


def test_supervisor_gpu_zero_uses_prevalidated_executable_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_monitor,
        "supervisor_query_executable_sha256",
        lambda: pytest.fail("capture must not re-resolve the executable"),
    )
    monkeypatch.setattr(
        gpu_monitor,
        "_run_supervisor_query",
        lambda argv, *, executable_sha256, timeout_seconds: (
            gpu_monitor.SupervisorGpuQuery(
                argv,
                executable_sha256,
                False,
                False,
                None,
                b"",
                b"",
            )
        ),
    )

    observation = gpu_monitor.capture_supervisor_gpu_zero(
        gpu_uuid="GPU-test",
        visible_device="GPU-test",
        supervisor_pid=1,
        supervisor_start_ticks=2,
        query_executable_sha256="a" * 64,
    )

    assert observation.gpu_inventory_query.query_executable_sha256 == "a" * 64
    assert observation.compute_apps_query.query_executable_sha256 == "a" * 64


def test_monitor_binds_samples_to_the_exact_gpu_uuid_and_direct_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = threading.Event()
    calls: list[tuple[object, dict[str, object]]] = []

    def run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        queried.set()
        return _completed(
            "GPU-other, 41, 900\n"
            "GPU-authenticated, 31415, 768\n"
            "GPU-authenticated, 2718, 1200\n"
        )

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=31415,
        interval_seconds=60.0,
        max_samples=4,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    measurement = monitor.finish()

    assert isinstance(measurement, gpu_monitor.ProcessGpuMemoryResult)
    assert measurement.gpu_uuid == "GPU-authenticated"
    assert measurement.provider_pid == 31415
    assert measurement.target_pid_observed is True
    assert tuple(sample.used_memory_mib for sample in measurement.samples) == (768,)
    assert measurement.peak_used_memory_mib == 768
    command, kwargs = calls[0]
    assert command == (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    assert kwargs == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 5.0,
    }


def test_monitor_result_and_samples_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = threading.Event()

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        queried.set()
        return _completed("GPU-authenticated, 123, 64\n")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=60.0,
        max_samples=1,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    measurement = monitor.finish()
    assert isinstance(measurement, gpu_monitor.ProcessGpuMemoryResult)

    with pytest.raises(FrozenInstanceError):
        measurement.peak_used_memory_mib = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        measurement.samples[0].used_memory_mib = 0  # type: ignore[misc]


def test_monitor_reports_the_peak_across_timestamped_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    three_queries = threading.Event()
    outputs = iter(
        (
            "GPU-authenticated, 123, 128\n",
            "GPU-authenticated, 123, 512\n",
            "GPU-authenticated, 123, 256\n",
        )
    )
    call_count = 0

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            three_queries.set()
        return _completed(next(outputs, ""))

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=0.001,
        max_samples=8,
    )

    monitor.start()
    assert three_queries.wait(timeout=1.0)
    measurement = monitor.finish()
    assert isinstance(measurement, gpu_monitor.ProcessGpuMemoryResult)

    assert tuple(sample.used_memory_mib for sample in measurement.samples) == (
        128,
        512,
        256,
    )
    assert measurement.peak_used_memory_mib == 512
    assert all(sample.sampled_at_unix_ns > 0 for sample in measurement.samples)


def test_never_observed_pid_is_explicitly_unavailable_without_a_zero_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = threading.Event()

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        queried.set()
        return _completed("GPU-authenticated, 999, 256\n")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=60.0,
        max_samples=1,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    measurement = monitor.finish()

    assert measurement == gpu_monitor.ProcessGpuMemoryUnavailable(
        reason="provider-pid-not-observed",
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
    )
    assert measurement.target_pid_observed is False
    assert not hasattr(measurement, "peak_used_memory_mib")


def test_cpu_measurement_is_explicitly_unavailable_and_never_queries_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("CPU must not invoke nvidia-smi"),
    )

    measurement = gpu_monitor.cpu_gpu_memory_unavailable(provider_pid=123)

    assert measurement == gpu_monitor.ProcessGpuMemoryUnavailable(
        reason="cpu-device",
        gpu_uuid=None,
        provider_pid=123,
    )


def test_gpu_memory_artifact_parser_recomputes_peak_and_rejects_tampering(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "gpu_memory.json"
    artifact_path.write_text(
        '{"availability":"available","gpu_uuid":"GPU-authenticated",'
        '"peak_used_memory_mib":512,"provider_pid":123,"samples":['
        '{"sampled_at_unix_ns":1,"used_memory_mib":256},'
        '{"sampled_at_unix_ns":2,"used_memory_mib":512}],'
        '"schema_version":1,"target_pid_observed":true,'
        '"unavailable_reason":null}\n',
        encoding="utf-8",
    )

    parsed = gpu_monitor.parse_process_gpu_memory_artifact(artifact_path)
    assert parsed.peak_used_memory_mib == 512

    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8").replace(
            '"peak_used_memory_mib":512', '"peak_used_memory_mib":513'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="peak"):
        gpu_monitor.parse_process_gpu_memory_artifact(artifact_path)


@pytest.mark.parametrize(
    "output",
    (
        "GPU-authenticated, 123\n",
        "GPU-authenticated, not-a-pid, 64\n",
        "GPU-authenticated, 123, N/A\n",
        "GPU-authenticated, 123, -1\n",
        ", 123, 64\n",
    ),
)
def test_malformed_nvidia_smi_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
) -> None:
    queried = threading.Event()

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        queried.set()
        return _completed(output)

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=60.0,
        max_samples=1,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    with pytest.raises(gpu_monitor.ProcessGpuMemoryMonitorError, match="malformed"):
        monitor.finish()


def test_target_pid_on_a_different_gpu_uuid_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = threading.Event()

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        queried.set()
        return _completed("GPU-unexpected, 123, 64\n")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=60.0,
        max_samples=1,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    with pytest.raises(
        gpu_monitor.ProcessGpuMemoryMonitorError,
        match="GPU-unexpected.*GPU-authenticated",
    ):
        monitor.finish()


def test_duplicate_target_pid_rows_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = threading.Event()

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        queried.set()
        return _completed("GPU-authenticated, 123, 64\nGPU-authenticated, 123, 65\n")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=60.0,
        max_samples=1,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    with pytest.raises(gpu_monitor.ProcessGpuMemoryMonitorError, match="duplicate"):
        monitor.finish()


def test_sample_capacity_overflow_fails_closed_instead_of_dropping_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    two_queries = threading.Event()
    call_count = 0

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            two_queries.set()
        return _completed("GPU-authenticated, 123, 64\n")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=0.001,
        max_samples=1,
    )

    monitor.start()
    assert two_queries.wait(timeout=1.0)
    with pytest.raises(gpu_monitor.ProcessGpuMemoryMonitorError, match="capacity"):
        monitor.finish()


def test_nvidia_smi_command_failure_is_not_converted_to_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = threading.Event()

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        queried.set()
        raise subprocess.CalledProcessError(9, ("nvidia-smi",), stderr="driver error")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=60.0,
        max_samples=1,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    with pytest.raises(
        gpu_monitor.ProcessGpuMemoryMonitorError,
        match="nvidia-smi query failed",
    ):
        monitor.finish()


def test_unexpected_polling_failure_is_not_hidden_by_the_background_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = threading.Event()

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        queried.set()
        raise RuntimeError("unexpected test failure")

    monkeypatch.setattr(gpu_monitor.subprocess, "run", run)
    monitor = gpu_monitor.ProcessGpuMemoryMonitor(
        gpu_uuid="GPU-authenticated",
        provider_pid=123,
        interval_seconds=60.0,
        max_samples=1,
    )

    monitor.start()
    assert queried.wait(timeout=1.0)
    with pytest.raises(
        gpu_monitor.ProcessGpuMemoryMonitorError,
        match="unexpected GPU-memory polling failure",
    ) as exception:
        monitor.finish()
    assert isinstance(exception.value.__cause__, RuntimeError)
