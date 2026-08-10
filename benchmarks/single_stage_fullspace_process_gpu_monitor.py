"""Exact-process GPU-memory evidence for CFS-SQP1 child executions."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryMeasurement,
    ProcessGpuMemoryMonitor,
    ProcessGpuMemoryResult,
)
from benchmarks.single_stage_fullspace_receipt import (
    SQP_MEMORY_SCHEMA_VERSION,
    canonical_json_bytes,
)
from benchmarks.single_stage_fullspace_snapshot import JsonValue


@dataclass(frozen=True, slots=True)
class LinuxProcessIdentity:
    """PID-reuse-resistant identity captured from one live Linux process."""

    pid: int
    start_ticks: int
    argv: tuple[str, ...]


def read_linux_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> LinuxProcessIdentity:
    """Read exact PID, kernel start ticks, and argv from procfs."""

    if type(pid) is not int or pid <= 0:
        raise ValueError("process PID must be a positive integer")
    process_root = proc_root / str(pid)
    stat_payload = (process_root / "stat").read_text(encoding="utf-8")
    _prefix, separator, stat_tail = stat_payload.rpartition(") ")
    stat_fields = stat_tail.split()
    if separator == "" or len(stat_fields) < 20:
        raise ValueError("process stat payload is malformed")
    start_ticks_text = stat_fields[19]
    if not start_ticks_text.isdecimal():
        raise ValueError("process start ticks are malformed")
    argv_bytes = (process_root / "cmdline").read_bytes()
    argv = tuple(
        field.decode("utf-8", "strict") for field in argv_bytes.split(b"\0") if field
    )
    if not argv:
        raise ValueError("process argv is empty")
    return LinuxProcessIdentity(pid=pid, start_ticks=int(start_ticks_text), argv=argv)


class BoundProcessGpuMemoryMonitor:
    """Bind the existing NVIDIA sampler to an exact live process identity."""

    def __init__(
        self,
        *,
        gpu_uuid: str,
        provider_pid: int,
        expected_argv: Sequence[str],
        proc_root: Path = Path("/proc"),
        interval_seconds: float = 0.1,
    ) -> None:
        identity = read_linux_process_identity(provider_pid, proc_root=proc_root)
        expected = tuple(expected_argv)
        if identity.argv != expected:
            raise ValueError("spawned child argv differs from the monitored process")
        self.identity = identity
        self.gpu_uuid = gpu_uuid
        self.interval_seconds = interval_seconds
        self._monitor = ProcessGpuMemoryMonitor(
            gpu_uuid=gpu_uuid,
            provider_pid=provider_pid,
            interval_seconds=interval_seconds,
        )

    def start(self) -> None:
        self._monitor.start()

    def finish(self) -> ProcessGpuMemoryMeasurement:
        return self._monitor.finish()


def bound_gpu_memory_payload(
    monitor: BoundProcessGpuMemoryMonitor,
    measurement: ProcessGpuMemoryMeasurement,
    *,
    parent_pid: int,
    physical_device_memory_bytes: int,
    runtime_argv: Sequence[str],
) -> dict[str, JsonValue]:
    """Serialize the validator's exact PID/device-bound SQP memory schema."""

    if not isinstance(measurement, ProcessGpuMemoryResult):
        raise TypeError("CFS-SQP1 requires an observed process GPU-memory sample")
    if measurement.gpu_uuid != monitor.gpu_uuid:
        raise ValueError("GPU-memory evidence differs from the bound device UUID")
    if type(parent_pid) is not int or parent_pid <= 0:
        raise ValueError("parent PID must be a positive integer")
    if (
        type(physical_device_memory_bytes) is not int
        or physical_device_memory_bytes <= 0
    ):
        raise ValueError("physical GPU memory must be a positive integer")
    peak_memory_bytes = measurement.peak_used_memory_mib * 1024 * 1024
    return {
        "schema_version": SQP_MEMORY_SCHEMA_VERSION,
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "parent_pid": parent_pid,
        "child_pid": monitor.identity.pid,
        "child_start_time_ticks": monitor.identity.start_ticks,
        "child_argv_sha256": hashlib.sha256(
            canonical_json_bytes(list(runtime_argv))
        ).hexdigest(),
        "device_uuid": measurement.gpu_uuid,
        "sample_count": len(measurement.samples),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_fraction": peak_memory_bytes / physical_device_memory_bytes,
    }


__all__ = (
    "BoundProcessGpuMemoryMonitor",
    "LinuxProcessIdentity",
    "bound_gpu_memory_payload",
    "read_linux_process_identity",
)
