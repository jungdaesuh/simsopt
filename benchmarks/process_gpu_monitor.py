"""Bounded NVIDIA GPU-memory sampling for one direct provider process."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

_NVIDIA_SMI_PROCESS_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
    "--format=csv,noheader,nounits",
)


class ProcessGpuMemoryMonitorError(RuntimeError):
    """The NVIDIA process-memory evidence could not be sampled faithfully."""


@dataclass(frozen=True)
class ProcessGpuMemorySample:
    """One wall-clock-stamped NVIDIA process-memory observation in MiB."""

    sampled_at_unix_ns: int
    used_memory_mib: int


@dataclass(frozen=True)
class ProcessGpuMemoryResult:
    """Immutable samples proving one PID's memory on one authenticated GPU."""

    gpu_uuid: str
    provider_pid: int
    samples: tuple[ProcessGpuMemorySample, ...]
    peak_used_memory_mib: int
    target_pid_observed: Literal[True] = field(default=True, init=False)


GpuMemoryUnavailableReason: TypeAlias = Literal[
    "cpu-device", "provider-pid-not-observed"
]


@dataclass(frozen=True)
class ProcessGpuMemoryUnavailable:
    """Explicit evidence that process GPU memory was not measurable."""

    reason: GpuMemoryUnavailableReason
    gpu_uuid: str | None
    provider_pid: int
    target_pid_observed: Literal[False] = field(default=False, init=False)


ProcessGpuMemoryMeasurement: TypeAlias = (
    ProcessGpuMemoryResult | ProcessGpuMemoryUnavailable
)


@dataclass(frozen=True)
class ProcessGpuMemoryArtifact:
    """Stable raw-artifact schema shared by available and unavailable results."""

    schema_version: Literal[1]
    availability: Literal["available", "unavailable"]
    unavailable_reason: GpuMemoryUnavailableReason | None
    gpu_uuid: str | None
    provider_pid: int
    target_pid_observed: bool
    samples: tuple[ProcessGpuMemorySample, ...]
    peak_used_memory_mib: int | None


@dataclass(frozen=True)
class _NvidiaProcessRow:
    gpu_uuid: str
    pid: int
    used_memory_mib: int


def cpu_gpu_memory_unavailable(*, provider_pid: int) -> ProcessGpuMemoryUnavailable:
    """Represent a CPU provider explicitly without invoking NVIDIA tooling."""

    if provider_pid <= 0:
        raise ValueError("provider_pid must be positive")
    return ProcessGpuMemoryUnavailable(
        reason="cpu-device",
        gpu_uuid=None,
        provider_pid=provider_pid,
    )


def process_gpu_memory_artifact(
    measurement: ProcessGpuMemoryMeasurement,
) -> ProcessGpuMemoryArtifact:
    """Normalize one monitor result into the versioned raw-artifact schema."""

    if isinstance(measurement, ProcessGpuMemoryResult):
        return ProcessGpuMemoryArtifact(
            schema_version=1,
            availability="available",
            unavailable_reason=None,
            gpu_uuid=measurement.gpu_uuid,
            provider_pid=measurement.provider_pid,
            target_pid_observed=measurement.target_pid_observed,
            samples=measurement.samples,
            peak_used_memory_mib=measurement.peak_used_memory_mib,
        )
    return ProcessGpuMemoryArtifact(
        schema_version=1,
        availability="unavailable",
        unavailable_reason=measurement.reason,
        gpu_uuid=measurement.gpu_uuid,
        provider_pid=measurement.provider_pid,
        target_pid_observed=measurement.target_pid_observed,
        samples=(),
        peak_used_memory_mib=None,
    )


def _parse_nvidia_process_rows(output: str) -> tuple[_NvidiaProcessRow, ...]:
    rows: list[_NvidiaProcessRow] = []
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3 or any(not field for field in fields):
            raise ProcessGpuMemoryMonitorError(
                f"malformed nvidia-smi process row: {line!r}"
            )
        gpu_uuid, pid_text, used_memory_mib_text = fields
        try:
            pid = int(pid_text)
            used_memory_mib = int(used_memory_mib_text)
        except ValueError as error:
            raise ProcessGpuMemoryMonitorError(
                f"malformed nvidia-smi process row: {line!r}"
            ) from error
        if pid <= 0 or used_memory_mib < 0:
            raise ProcessGpuMemoryMonitorError(
                f"malformed nvidia-smi process row: {line!r}"
            )
        rows.append(
            _NvidiaProcessRow(
                gpu_uuid=gpu_uuid,
                pid=pid,
                used_memory_mib=used_memory_mib,
            )
        )
    return tuple(rows)


class ProcessGpuMemoryMonitor:
    """Poll one direct PID and exact GPU UUID into a bounded immutable result.

    Call ``start`` after spawning the provider and ``finish`` after it exits.
    Monitoring failures are raised; an unseen PID returns explicit unavailable
    evidence and is never represented by an inferred zero-memory sample.
    """

    def __init__(
        self,
        *,
        gpu_uuid: str,
        provider_pid: int,
        interval_seconds: float = 0.1,
        max_samples: int = 32768,
        query_timeout_seconds: float = 5.0,
    ) -> None:
        if not gpu_uuid or gpu_uuid.strip() != gpu_uuid:
            raise ValueError("gpu_uuid must be a nonempty exact NVIDIA UUID")
        if provider_pid <= 0:
            raise ValueError("provider_pid must be positive")
        if interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if query_timeout_seconds <= 0.0:
            raise ValueError("query_timeout_seconds must be positive")

        self.gpu_uuid = gpu_uuid
        self.provider_pid = provider_pid
        self.interval_seconds = float(interval_seconds)
        self.max_samples = max_samples
        self.query_timeout_seconds = float(query_timeout_seconds)
        self._samples: list[ProcessGpuMemorySample] = []
        self._failure: Exception | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: Literal["new", "running", "finished"] = "new"
        self._measurement: ProcessGpuMemoryMeasurement | None = None

    def start(self) -> None:
        """Start immediate sampling; a monitor instance can be started once."""

        with self._lock:
            if self._state != "new":
                raise RuntimeError("GPU-memory monitor can be started only once")
            self._state = "running"
            self._thread = threading.Thread(
                target=self._poll_until_stopped,
                name=f"gpu-memory-pid-{self.provider_pid}",
                daemon=True,
            )
            self._thread.start()

    def finish(self) -> ProcessGpuMemoryMeasurement:
        """Stop sampling and return evidence, raising any polling failure."""

        with self._lock:
            if self._state == "new":
                raise RuntimeError("GPU-memory monitor was not started")
            if self._state == "finished":
                completed_measurement = self._measurement
                if completed_measurement is None:
                    raise RuntimeError("GPU-memory monitor has no completed result")
                return completed_measurement
            thread = self._thread
        if thread is None:
            raise RuntimeError("GPU-memory monitor thread was not created")

        self._stop.set()
        thread.join()

        with self._lock:
            failure = self._failure
            samples = tuple(self._samples)
        if failure is not None:
            if isinstance(failure, ProcessGpuMemoryMonitorError):
                raise failure
            raise ProcessGpuMemoryMonitorError(
                f"unexpected GPU-memory polling failure for provider PID "
                f"{self.provider_pid} on GPU {self.gpu_uuid}"
            ) from failure
        if samples:
            measurement: ProcessGpuMemoryMeasurement = ProcessGpuMemoryResult(
                gpu_uuid=self.gpu_uuid,
                provider_pid=self.provider_pid,
                samples=samples,
                peak_used_memory_mib=max(sample.used_memory_mib for sample in samples),
            )
        else:
            measurement = ProcessGpuMemoryUnavailable(
                reason="provider-pid-not-observed",
                gpu_uuid=self.gpu_uuid,
                provider_pid=self.provider_pid,
            )
        with self._lock:
            self._measurement = measurement
            self._state = "finished"
        return measurement

    def _poll_until_stopped(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._query_sample()
                if sample is not None:
                    self._record_sample(sample)
            # Preserve daemon-thread failures for synchronous re-raising in finish().
            except Exception as error:  # noqa: BLE001
                with self._lock:
                    self._failure = error
                self._stop.set()
                return
            if self._stop.wait(self.interval_seconds):
                return

    def _query_sample(self) -> ProcessGpuMemorySample | None:
        try:
            completed = subprocess.run(
                _NVIDIA_SMI_PROCESS_QUERY,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.query_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ProcessGpuMemoryMonitorError(
                f"nvidia-smi query failed for provider PID {self.provider_pid} "
                f"on GPU {self.gpu_uuid}"
            ) from error

        target_rows = tuple(
            row
            for row in _parse_nvidia_process_rows(completed.stdout)
            if row.pid == self.provider_pid
        )
        if len(target_rows) > 1:
            raise ProcessGpuMemoryMonitorError(
                f"duplicate nvidia-smi rows for provider PID {self.provider_pid}"
            )
        if not target_rows:
            return None
        target = target_rows[0]
        if target.gpu_uuid != self.gpu_uuid:
            raise ProcessGpuMemoryMonitorError(
                f"provider PID {self.provider_pid} appeared on GPU "
                f"{target.gpu_uuid}, expected authenticated GPU {self.gpu_uuid}"
            )
        return ProcessGpuMemorySample(
            sampled_at_unix_ns=time.time_ns(),
            used_memory_mib=target.used_memory_mib,
        )

    def _record_sample(self, sample: ProcessGpuMemorySample) -> None:
        with self._lock:
            if len(self._samples) >= self.max_samples:
                raise ProcessGpuMemoryMonitorError(
                    f"GPU-memory sample capacity {self.max_samples} exhausted "
                    f"for provider PID {self.provider_pid}"
                )
            self._samples.append(sample)


__all__ = [
    "GpuMemoryUnavailableReason",
    "ProcessGpuMemoryMeasurement",
    "ProcessGpuMemoryMonitor",
    "ProcessGpuMemoryMonitorError",
    "ProcessGpuMemoryArtifact",
    "ProcessGpuMemoryResult",
    "ProcessGpuMemorySample",
    "ProcessGpuMemoryUnavailable",
    "cpu_gpu_memory_unavailable",
    "process_gpu_memory_artifact",
]
