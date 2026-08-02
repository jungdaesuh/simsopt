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
