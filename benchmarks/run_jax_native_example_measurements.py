"""Collect matched native/JAX timing and peak-memory evidence."""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchmarks.jax_native_example_measurement_contract import (
    MEASUREMENT_EVIDENCE_KIND,
    MEASUREMENT_PROFILE_IDS,
    MEASUREMENT_SCHEMA_VERSION,
    WARM_SAMPLE_COUNT,
    MeasurementProfileId,
    MeasurementScale,
)
from examples.jax._lane_environment import build_execution_environment

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NVIDIA_MEMORY_MULTIPLIER = 1024 * 1024
_PROFILE_ENVIRONMENT_NAMES = frozenset(
    (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "SIMSOPT_BACKEND_MODE",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
)


class MeasurementRunnerError(RuntimeError):
    """The runner cannot produce complete, trustworthy measurement evidence."""


@dataclass(frozen=True)
class MeasurementSchedule:
    """Cold, warmup, and seven balanced warm profile orders."""

    cold: tuple[MeasurementProfileId, ...]
    warmup: tuple[MeasurementProfileId, ...]
    warm: tuple[tuple[MeasurementProfileId, ...], ...]


@dataclass(frozen=True)
class CollectionRun:
    """One position in the isolated timing and allocation-memory protocol."""

    profile_id: MeasurementProfileId
    phase: Literal["cold", "warmup", "warm", "allocation_memory"]
    sample_index: int | None
    order_position: int
    measured: bool
    allocation_sensitive: bool


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, document: object) -> None:
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def publish_artifact_exclusive(
    document: object,
    *,
    artifact_root: Path,
    run_directory_name: str,
) -> Path:
    """Publish one immutable artifact without replacing retained evidence."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_directory = artifact_root / run_directory_name
    run_directory.mkdir()
    _fsync_directory(artifact_root)
    artifact_path = run_directory / "artifact.json"
    _write_json_exclusive(artifact_path, document)
    return artifact_path


def _rotate(
    values: tuple[MeasurementProfileId, ...], offset: int
) -> tuple[MeasurementProfileId, ...]:
    normalized = offset % len(values)
    return values[normalized:] + values[:normalized]


def build_measurement_schedule(mirror_index: int) -> MeasurementSchedule:
    """Return a deterministic rotating five-profile measurement schedule."""
    if mirror_index < 0:
        raise ValueError("mirror_index must be nonnegative")
    cold = _rotate(MEASUREMENT_PROFILE_IDS, mirror_index)
    return MeasurementSchedule(
        cold=cold,
        warmup=tuple(reversed(cold)),
        warm=tuple(_rotate(cold, index) for index in range(WARM_SAMPLE_COUNT)),
    )


def build_collection_plan(mirror_index: int) -> tuple[CollectionRun, ...]:
    """Expand one balanced schedule into the exact 47-process protocol."""
    schedule = build_measurement_schedule(mirror_index)
    runs: list[CollectionRun] = [
        CollectionRun(
            profile_id=profile_id,
            phase="cold",
            sample_index=None,
            order_position=order_position,
            measured=True,
            allocation_sensitive=False,
        )
        for order_position, profile_id in enumerate(schedule.cold)
    ]
    runs.extend(
        CollectionRun(
            profile_id=profile_id,
            phase="warmup",
            sample_index=None,
            order_position=order_position,
            measured=False,
            allocation_sensitive=False,
        )
        for order_position, profile_id in enumerate(schedule.warmup)
    )
    for sample_index, order in enumerate(schedule.warm):
        runs.extend(
            CollectionRun(
                profile_id=profile_id,
                phase="warm",
                sample_index=sample_index,
                order_position=order_position,
                measured=True,
                allocation_sensitive=False,
            )
            for order_position, profile_id in enumerate(order)
        )
    runs.extend(
        CollectionRun(
            profile_id=profile_id,
            phase="allocation_memory",
            sample_index=None,
            order_position=order_position,
            measured=True,
            allocation_sensitive=True,
        )
        for order_position, profile_id in enumerate(("jax_gpu_fast", "jax_gpu_parity"))
    )
    return tuple(runs)


def classify_termination(returncode: int) -> str:
    """Classify normal, explicit-exit, signal, and possible-OOM termination."""
    if returncode == 0:
        return "normal"
    if returncode == -signal.SIGKILL:
        return "resource_limit_or_oom"
    if returncode < 0:
        return f"signal_{-returncode}"
    return f"exit_{returncode}"


def parse_nvidia_smi_compute_apps(output: str) -> dict[int, int]:
    """Parse process-attributed NVIDIA memory rows into bytes by PID."""
    stripped = output.strip()
    if not stripped or stripped == "No running processes found":
        return {}
    result: dict[int, int] = {}
    for line in stripped.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2:
            raise MeasurementRunnerError(f"malformed nvidia-smi process row: {line!r}")
        try:
            process_id = int(fields[0])
            memory_mib = int(fields[1].removesuffix(" MiB").strip())
        except ValueError as error:
            raise MeasurementRunnerError(
                f"malformed nvidia-smi process row: {line!r}"
            ) from error
        if process_id <= 0 or memory_mib < 0:
            raise MeasurementRunnerError(f"malformed nvidia-smi process row: {line!r}")
        result[process_id] = memory_mib * _NVIDIA_MEMORY_MULTIPLIER
    return result


def build_profile_command(
    *,
    python_executable: str,
    case_id: str,
    profile_id: MeasurementProfileId,
    input_bundle_path: Path,
    result_directory: Path,
    scale: MeasurementScale,
) -> tuple[str, ...]:
    """Build a profile command that consumes the one canonical input bundle."""
    if not case_id:
        raise ValueError("case_id must not be empty")
    if profile_id == "native_cpu":
        lane = "native-cpu"
    elif profile_id in ("jax_cpu_fast", "jax_cpu_parity"):
        lane = "jax-cpu"
    else:
        lane = "jax-gpu"
    return (
        python_executable,
        "-m",
        "examples.jax.parity.child",
        "--case",
        case_id,
        "--lane",
        lane,
        "--input-bundle",
        str(input_bundle_path),
        "--result-directory",
        str(result_directory),
        "--scale",
        scale,
    )


def _jax_profile_parts(
    profile_id: MeasurementProfileId,
) -> tuple[Literal["cpu", "gpu"], Literal["fast", "parity"]]:
    if profile_id == "jax_cpu_fast":
        return "cpu", "fast"
    if profile_id == "jax_cpu_parity":
        return "cpu", "parity"
    if profile_id == "jax_gpu_fast":
        return "gpu", "fast"
    if profile_id == "jax_gpu_parity":
        return "gpu", "parity"
    raise ValueError("native_cpu is not a JAX profile")


def build_measurement_environment(
    profile_id: MeasurementProfileId,
    *,
    allocation_sensitive: bool,
    base_environment: Mapping[str, str],
    gpu_index: int = 0,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, str]:
    """Resolve one isolated profile without inheriting stale backend policy."""
    if gpu_index < 0:
        raise ValueError("gpu_index must be nonnegative")
    scrubbed = {
        name: value
        for name, value in base_environment.items()
        if name not in _PROFILE_ENVIRONMENT_NAMES
    }
    if profile_id == "native_cpu":
        if allocation_sensitive:
            raise ValueError("allocation-sensitive collection is GPU-only")
        return scrubbed
    device, intent = _jax_profile_parts(profile_id)
    if allocation_sensitive and device != "gpu":
        raise ValueError("allocation-sensitive collection is GPU-only")
    _, environment = build_execution_environment(
        device,
        intent,
        scrubbed,
        repo_root=repo_root,
    )
    if device == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        "false" if allocation_sensitive else "true"
    )
    return environment


__all__ = [
    "CollectionRun",
    "MEASUREMENT_EVIDENCE_KIND",
    "MEASUREMENT_SCHEMA_VERSION",
    "MeasurementSchedule",
    "MeasurementRunnerError",
    "build_collection_plan",
    "build_measurement_environment",
    "build_measurement_schedule",
    "build_profile_command",
    "classify_termination",
    "parse_nvidia_smi_compute_apps",
    "publish_artifact_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(
        "The measurement CLI is not available until source-owned commands are bound."
    )
