"""Collect matched native/JAX timing and peak-memory evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import secrets
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypeVar

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
for _import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from examples.jax._lane_environment import build_execution_environment
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.arbiter import LaneObservation, arbitrate
from examples.jax.parity.cases import get_case
from examples.jax.parity.receipts import load_lane_observation
from simsopt.optimization_trajectory import read_optimization_window_timing
from simsopt.single_stage_boozer_vacuum import (
    JAX_FAST_DRIVER_ID,
    JAX_OPTAX_DRIVER_ID,
    JAX_PARITY_DRIVER_ID,
)
from simsopt_jax.config import ExecutionIntent

from benchmarks.jax_native_example_measurement_contract import (
    MEASUREMENT_EVIDENCE_KIND,
    MEASUREMENT_PROFILE_IDS,
    MEASUREMENT_SCHEMA_VERSION,
    WARM_SAMPLE_COUNT,
    MeasurementProfileId,
    MeasurementScale,
    validate_measurement_artifact,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    build_snapshot_module_launch,
)
from benchmarks.single_stage_speed_campaign_receipt import (
    DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
    CampaignMetadata,
    CampaignReceipt,
    EndpointAudit,
    EndpointCertificateAudit,
    EndpointObservables,
    LaneEndpoint,
    LaneReceipt,
    ParityRow,
    ReceiptLaneId,
    SampleMeasurement,
    SamplePhase,
    TrajectoryPoint,
    single_stage_speed_parity_tolerance,
    write_campaign_receipt,
)

_NVIDIA_MEMORY_MULTIPLIER = 1024 * 1024
_DEFAULT_POLL_INTERVAL_SECONDS = 0.05
_SINGLE_STAGE_CASE_ID = "native-single-stage-boozer-vacuum-optimization"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_GPU_PREFLIGHT_SAMPLE_COUNT = 5
_GPU_PREFLIGHT_INTERVAL_SECONDS = 0.2
_GPU_CONCURRENT_MEMORY_FRACTION_MAX = 0.05
_GPU_CONCURRENT_UTILIZATION_PERCENT_MAX = 5
SingleStageMeasurementProfileId = Literal["jax_gpu_optax"]
RunnerProfileId = MeasurementProfileId | SingleStageMeasurementProfileId
_ProfileId = TypeVar("_ProfileId", MeasurementProfileId, RunnerProfileId)
SINGLE_STAGE_SPEED_PROFILE_IDS: tuple[RunnerProfileId, ...] = (
    "native_cpu",
    "jax_gpu_fast",
    "jax_gpu_optax",
    "jax_cpu_fast",
)
_SINGLE_STAGE_RECEIPT_LANE_IDS: dict[RunnerProfileId, ReceiptLaneId] = {
    "native_cpu": "native_cpu",
    "jax_gpu_fast": "jax_gpu_custom",
    "jax_gpu_optax": "jax_gpu_optax",
    "jax_cpu_fast": "jax_cpu_custom",
}
_SINGLE_STAGE_EXPECTED_DRIVERS: dict[RunnerProfileId, str] = {
    "native_cpu": "simsopt_scipy_bfgs_with_boozer_newton",
    "jax_gpu_fast": JAX_FAST_DRIVER_ID,
    "jax_gpu_optax": JAX_OPTAX_DRIVER_ID,
    "jax_cpu_fast": JAX_FAST_DRIVER_ID,
}
_SINGLE_STAGE_PARITY_OBSERVABLES: Final[tuple[tuple[str, str], ...]] = (
    ("final_objective", "objective"),
    ("final_iota", "iota"),
    ("final_volume", "volume"),
    ("final_non_qs_ratio", "non_qs_ratio"),
    ("final_boozer_residual", "boozer_residual"),
)
_EXACT_ADJOINT_ENVIRONMENT_NAME: Final = "SIMSOPT_EXACT_ADJOINT_DENSE_LU"
_PROFILE_ENVIRONMENT_NAMES = frozenset(
    (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "SIMSOPT_BACKEND_MODE",
        _EXACT_ADJOINT_ENVIRONMENT_NAME,
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
)
# Inherited threading configuration changes OpenMP reduction order, which
# forks native FP trajectories between launch contexts (proven 2026-08-04:
# bitwise-identical 1,000-iteration replays under two env constructions from
# one shell, endpoint inner-solve failure from another). Scrub every
# numerical-threading variable and pin a deterministic replacement.
_NUMERICAL_ENVIRONMENT_PREFIXES = (
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "NUMEXPR_",
    "VECLIB_",
    "BLIS_",
)


def _pinned_threading_environment() -> dict[str, str]:
    thread_count = str(os.cpu_count())
    return {
        "OMP_NUM_THREADS": thread_count,
        "OMP_DYNAMIC": "FALSE",
        "OMP_SCHEDULE": "STATIC",
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
    }


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

    profile_id: RunnerProfileId
    phase: Literal["cold", "warmup", "warm", "allocation_memory"]
    sample_index: int | None
    order_position: int
    measured: bool
    allocation_sensitive: bool


@dataclass(frozen=True)
class MonitoredCommandResult:
    """Parent-observed result for one isolated child process."""

    returncode: int
    termination: str
    wall_seconds: float
    peak_process_tree_rss_bytes: int
    peak_gpu_process_bytes: int | None
    gpu_counter_status: Literal["available", "unavailable", "not_applicable"]
    stdout_sha256: str
    stderr_sha256: str


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


def _rotate(values: tuple[_ProfileId, ...], offset: int) -> tuple[_ProfileId, ...]:
    normalized = offset % len(values)
    return tuple(
        values[(normalized + index) % len(values)] for index in range(len(values))
    )


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


def build_single_stage_speed_collection_plan(
    mirror_index: int,
) -> tuple[CollectionRun, ...]:
    """Return the campaign's four-lane rotating isolated-process plan."""
    cold = _rotate(SINGLE_STAGE_SPEED_PROFILE_IDS, mirror_index)
    warmup = tuple(reversed(cold))
    warm = tuple(_rotate(cold, index) for index in range(WARM_SAMPLE_COUNT))
    runs = [
        CollectionRun(profile_id, "cold", None, position, True, False)
        for position, profile_id in enumerate(cold)
    ]
    runs.extend(
        CollectionRun(profile_id, "warmup", None, position, False, False)
        for position, profile_id in enumerate(warmup)
    )
    for sample_index, order in enumerate(warm):
        runs.extend(
            CollectionRun(
                profile_id,
                "warm",
                sample_index,
                position,
                True,
                False,
            )
            for position, profile_id in enumerate(order)
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


def evaluate_gpu_concurrent_load_preflight(
    *,
    processes: Mapping[int, int],
    utilization_samples: tuple[tuple[int, int], ...],
    total_memory_bytes: int,
) -> str:
    """Validate and serialize the checked bounded GPU concurrency policy."""
    if len(utilization_samples) != _GPU_PREFLIGHT_SAMPLE_COUNT:
        raise MeasurementRunnerError(
            f"GPU preflight requires {_GPU_PREFLIGHT_SAMPLE_COUNT} samples"
        )
    if total_memory_bytes <= 0:
        raise MeasurementRunnerError("GPU total memory must be positive")
    if any(
        gpu_percent < 0
        or gpu_percent > 100
        or memory_percent < 0
        or memory_percent > 100
        for gpu_percent, memory_percent in utilization_samples
    ):
        raise MeasurementRunnerError("GPU utilization samples must be percentages")
    max_gpu_percent = max(sample[0] for sample in utilization_samples)
    if max_gpu_percent > _GPU_CONCURRENT_UTILIZATION_PERCENT_MAX:
        raise MeasurementRunnerError(
            "GPU concurrent utilization exceeds the checked limit: "
            f"{max_gpu_percent}% > {_GPU_CONCURRENT_UTILIZATION_PERCENT_MAX}%"
        )
    max_memory_percent = max(sample[1] for sample in utilization_samples)
    process_memory_bytes = sum(processes.values())
    memory_fraction = process_memory_bytes / total_memory_bytes
    if memory_fraction > _GPU_CONCURRENT_MEMORY_FRACTION_MAX:
        raise MeasurementRunnerError(
            "GPU concurrent process memory exceeds the checked limit: "
            f"{memory_fraction:.9g} > {_GPU_CONCURRENT_MEMORY_FRACTION_MAX:.9g}"
        )
    process_detail = (
        ",".join(
            f"{process_id}:{processes[process_id]}" for process_id in sorted(processes)
        )
        if processes
        else "none"
    )
    return (
        "pass:"
        f"background_processes={process_detail};"
        f"background_memory_fraction={memory_fraction:.9g};"
        f"max_gpu_utilization_percent={max_gpu_percent};"
        f"max_memory_utilization_percent={max_memory_percent};"
        f"samples={len(utilization_samples)}"
    )


def _process_tree_pids(root_process_id: int) -> frozenset[int]:
    pending = [root_process_id]
    observed: set[int] = set()
    while pending:
        process_id = pending.pop()
        if process_id in observed:
            continue
        observed.add(process_id)
        children_path = (
            Path("/proc") / str(process_id) / "task" / str(process_id) / "children"
        )
        try:
            children = children_path.read_text(encoding="ascii").split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        pending.extend(int(child) for child in children)
    return frozenset(observed)


def _process_rss_bytes(process_id: int) -> int:
    status_path = Path("/proc") / str(process_id) / "status"
    try:
        lines = status_path.read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    return 0


def _nvidia_compute_apps(gpu_index: int) -> dict[int, int]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise MeasurementRunnerError(
            "nvidia-smi process query failed: " + completed.stderr.strip()
        )
    return parse_nvidia_smi_compute_apps(completed.stdout)


def _sample_process_tree(
    root_process_id: int,
    *,
    device: Literal["cpu", "gpu"],
    gpu_index: int,
) -> tuple[int, int | None, bool]:
    process_ids = _process_tree_pids(root_process_id)
    rss_bytes = sum(_process_rss_bytes(process_id) for process_id in process_ids)
    if device == "cpu":
        return rss_bytes, None, True
    try:
        memory_by_process = _nvidia_compute_apps(gpu_index)
    except MeasurementRunnerError:
        return rss_bytes, None, False
    gpu_bytes = sum(memory_by_process.get(process_id, 0) for process_id in process_ids)
    return rss_bytes, gpu_bytes, True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute_monitored_command(
    *,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    device: Literal["cpu", "gpu"],
    gpu_index: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    max_process_tree_rss_bytes: int | None = None,
) -> MonitoredCommandResult:
    """Run one child with bounded wall time and optional process-tree RSS."""
    if poll_interval_seconds <= 0.0 or timeout_seconds <= 0.0:
        raise ValueError("poll interval and timeout must be positive")
    if max_process_tree_rss_bytes is not None and max_process_tree_rss_bytes <= 0:
        raise ValueError("process-tree RSS bound must be positive")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    peak_rss_bytes = 0
    peak_gpu_bytes = 0 if device == "gpu" else None
    gpu_counter_available = True
    timed_out = False
    memory_limited = False
    started_ns = time.monotonic_ns()
    with stdout_path.open("xb") as stdout_stream, stderr_path.open(
        "xb"
    ) as stderr_stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=max_process_tree_rss_bytes is not None,
        )
        while process.poll() is None:
            rss_bytes, gpu_bytes, sample_available = _sample_process_tree(
                process.pid,
                device=device,
                gpu_index=gpu_index,
            )
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            if device == "gpu":
                gpu_counter_available = gpu_counter_available and sample_available
                if gpu_bytes is not None and peak_gpu_bytes is not None:
                    peak_gpu_bytes = max(peak_gpu_bytes, gpu_bytes)
            elapsed_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
            if (
                max_process_tree_rss_bytes is not None
                and rss_bytes > max_process_tree_rss_bytes
            ):
                memory_limited = True
                os.killpg(process.pid, signal.SIGTERM)
                break
            if elapsed_seconds > timeout_seconds:
                timed_out = True
                if max_process_tree_rss_bytes is None:
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                break
            time.sleep(poll_interval_seconds)
        if timed_out or memory_limited:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                if max_process_tree_rss_bytes is None:
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
        returncode = process.wait()
        stdout_stream.flush()
        stderr_stream.flush()
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
    _fsync_directory(stdout_path.parent)
    if (timed_out or memory_limited) and returncode == 0:
        returncode = -signal.SIGTERM
    wall_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
    if peak_rss_bytes <= 0:
        raise MeasurementRunnerError("process-tree RSS monitor recorded no samples")
    if device == "cpu":
        gpu_status: Literal["available", "unavailable", "not_applicable"] = (
            "not_applicable"
        )
    elif not gpu_counter_available or not peak_gpu_bytes:
        gpu_status = "unavailable"
        peak_gpu_bytes = None
    else:
        gpu_status = "available"
    return MonitoredCommandResult(
        returncode=returncode,
        termination=(
            "process_tree_memory_limit"
            if memory_limited
            else "wall_time_limit"
            if timed_out
            else classify_termination(returncode)
        ),
        wall_seconds=wall_seconds,
        peak_process_tree_rss_bytes=peak_rss_bytes,
        peak_gpu_process_bytes=peak_gpu_bytes,
        gpu_counter_status=gpu_status,
        stdout_sha256=_sha256_file(stdout_path),
        stderr_sha256=_sha256_file(stderr_path),
    )


def build_profile_command(
    *,
    python_executable: str,
    case_id: str,
    profile_id: RunnerProfileId,
    input_bundle_path: Path,
    result_directory: Path,
    scale: MeasurementScale,
    trajectory_path: Path | None = None,
    optimization_timing_path: Path | None = None,
    immutable_snapshot_provenance_path: Path | None = None,
    isolated_site: bool = False,
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
    python_prefix = (python_executable, "-S") if isolated_site else (python_executable,)
    command = (
        *python_prefix,
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
    if trajectory_path is not None:
        command += ("--trajectory-path", str(trajectory_path))
    if optimization_timing_path is not None:
        command += ("--optimization-timing-path", str(optimization_timing_path))
    if immutable_snapshot_provenance_path is not None:
        command += (
            "--immutable-snapshot-provenance",
            str(immutable_snapshot_provenance_path),
        )
    if profile_id == "jax_gpu_optax":
        if trajectory_path is None:
            raise ValueError("jax_gpu_optax requires a trajectory path")
        command += ("--optimizer-backend", "optax-lbfgs")
    return command


def _jax_profile_parts(
    profile_id: RunnerProfileId,
) -> tuple[Literal["cpu", "gpu"], Literal["fast", "parity"]]:
    if profile_id == "jax_cpu_fast":
        return "cpu", "fast"
    if profile_id == "jax_cpu_parity":
        return "cpu", "parity"
    if profile_id in ("jax_gpu_fast", "jax_gpu_optax"):
        return "gpu", "fast"
    if profile_id == "jax_gpu_parity":
        return "gpu", "parity"
    raise ValueError("native_cpu is not a JAX profile")


def build_measurement_environment(
    profile_id: RunnerProfileId,
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
        and not name.startswith(_NUMERICAL_ENVIRONMENT_PREFIXES)
    }
    scrubbed.update(_pinned_threading_environment())
    if profile_id == "native_cpu":
        if allocation_sensitive:
            raise ValueError("allocation-sensitive collection is GPU-only")
        source_root = str(repo_root / "src")
        inherited_pythonpath = scrubbed.get("PYTHONPATH")
        scrubbed["PYTHONPATH"] = (
            source_root
            if not inherited_pythonpath
            else os.pathsep.join((source_root, inherited_pythonpath))
        )
        scrubbed["MPI4PY_RC_INITIALIZE"] = "false"
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


def _profile_device(profile_id: RunnerProfileId) -> Literal["cpu", "gpu"]:
    return (
        "gpu"
        if profile_id in ("jax_gpu_fast", "jax_gpu_optax", "jax_gpu_parity")
        else "cpu"
    )


def _profile_intent(
    profile_id: RunnerProfileId,
) -> Literal["native", "fast", "parity"]:
    if profile_id == "native_cpu":
        return "native"
    return "parity" if profile_id.endswith("_parity") else "fast"


def _expected_backend_mode(profile_id: RunnerProfileId) -> str:
    return "jax_gpu_fast" if profile_id == "jax_gpu_optax" else profile_id


def _environment_sha256(environment: Mapping[str, str]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(dict(sorted(environment.items())))
    ).hexdigest()


def _sample_stem(run: CollectionRun, sequence_index: int) -> str:
    sample_index = "none" if run.sample_index is None else f"{run.sample_index:02d}"
    return (
        f"{sequence_index:03d}-{run.phase}-{run.profile_id}-"
        f"sample-{sample_index}-position-{run.order_position}"
    )


def _validate_observation(
    observation: LaneObservation,
    *,
    profile_id: RunnerProfileId,
    scale: MeasurementScale,
    input_fingerprint: str,
) -> None:
    expected_lane = (
        "native-cpu"
        if profile_id == "native_cpu"
        else "jax-gpu"
        if _profile_device(profile_id) == "gpu"
        else "jax-cpu"
    )
    if observation.lane != expected_lane:
        raise MeasurementRunnerError(
            f"{profile_id} emitted lane {observation.lane}, expected {expected_lane}"
        )
    if observation.backend_mode != _expected_backend_mode(profile_id):
        raise MeasurementRunnerError(
            f"{profile_id} emitted backend mode {observation.backend_mode}"
        )
    if observation.platform != _profile_device(profile_id):
        raise MeasurementRunnerError(
            f"{profile_id} emitted platform {observation.platform}"
        )
    if observation.precision != "fp64":
        raise MeasurementRunnerError(
            f"{profile_id} emitted precision {observation.precision}"
        )
    if observation.scale != scale:
        raise MeasurementRunnerError(
            f"{profile_id} emitted scale {observation.scale}, expected {scale}"
        )
    if observation.input_fingerprint != input_fingerprint:
        raise MeasurementRunnerError(
            f"{profile_id} did not consume the canonical input bundle"
        )
    if profile_id == "jax_gpu_optax" and observation.driver != JAX_OPTAX_DRIVER_ID:
        raise MeasurementRunnerError(
            f"jax_gpu_optax driver must be {JAX_OPTAX_DRIVER_ID}, "
            f"got {observation.driver}"
        )
    if not observation.success:
        raise MeasurementRunnerError(
            f"{profile_id} failed scientifically: {observation.raw_status}"
        )
    provenance = observation.provenance
    if provenance is None:
        raise MeasurementRunnerError(f"{profile_id} omitted lane provenance")
    expected_sync = (
        "native synchronous execution"
        if profile_id == "native_cpu"
        else "jax.block_until_ready over published observation values"
    )
    if provenance.measurement_synchronization != expected_sync:
        raise MeasurementRunnerError(f"{profile_id} omitted required synchronization")


def _execute_collection_run(
    *,
    case_id: str,
    scale: MeasurementScale,
    input_bundle_path: Path,
    staging_directory: Path,
    run: CollectionRun,
    sequence_index: int,
    environment: Mapping[str, str],
    python_executable: str,
    isolated_site: bool,
    repo_root: Path,
    gpu_index: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    input_fingerprint: str,
) -> tuple[MonitoredCommandResult, LaneObservation]:
    stem = _sample_stem(run, sequence_index)
    result_directory = staging_directory / "receipts" / stem
    result_directory.mkdir(parents=True)
    trajectory_path = (
        result_directory / "trajectory.jsonl"
        if case_id == _SINGLE_STAGE_CASE_ID
        else None
    )
    command = build_profile_command(
        python_executable=python_executable,
        case_id=case_id,
        profile_id=run.profile_id,
        input_bundle_path=input_bundle_path,
        result_directory=result_directory,
        scale=scale,
        trajectory_path=trajectory_path,
        isolated_site=isolated_site,
    )
    monitored = execute_monitored_command(
        command=command,
        environment=environment,
        cwd=repo_root,
        stdout_path=staging_directory / "logs" / f"{stem}.stdout",
        stderr_path=staging_directory / "logs" / f"{stem}.stderr",
        device=_profile_device(run.profile_id),
        gpu_index=gpu_index,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    if monitored.returncode != 0:
        raise MeasurementRunnerError(
            f"{run.profile_id} {run.phase} failed with "
            f"{monitored.termination}; see logs/{stem}.stderr"
        )
    observation = load_lane_observation(result_directory)
    _validate_observation(
        observation,
        profile_id=run.profile_id,
        scale=scale,
        input_fingerprint=input_fingerprint,
    )
    return monitored, observation


def _timing_sample_document(
    run: CollectionRun,
    monitored: MonitoredCommandResult,
) -> dict[str, object]:
    measured = run.phase != "warmup"
    return {
        "phase": run.phase,
        "sample_index": run.sample_index,
        "order_position": run.order_position,
        "measured": measured,
        "isolated_process": True,
        "returncode": monitored.returncode,
        "termination": monitored.termination,
        "scientific_success": True,
        "timing_synchronized": True,
        "setup_compile_seconds": None,
        "solver_seconds": None,
        "total_seconds": monitored.wall_seconds if measured else None,
        "peak_process_tree_rss_bytes": monitored.peak_process_tree_rss_bytes,
        "gpu_peak_process_bytes": monitored.peak_gpu_process_bytes,
        "stdout_sha256": monitored.stdout_sha256,
        "stderr_sha256": monitored.stderr_sha256,
    }


def _allocation_sample_document(
    monitored: MonitoredCommandResult,
    *,
    poll_interval_seconds: float,
    concurrent_use_preflight: str,
) -> dict[str, object]:
    if (
        monitored.gpu_counter_status != "available"
        or monitored.peak_gpu_process_bytes is None
    ):
        raise MeasurementRunnerError(
            "GPU allocation-memory counter is unavailable; complete evidence "
            "cannot be published"
        )
    return {
        "isolated_process": True,
        "xla_python_client_preallocate": "false",
        "returncode": monitored.returncode,
        "termination": monitored.termination,
        "scientific_success": True,
        "peak_process_tree_rss_bytes": monitored.peak_process_tree_rss_bytes,
        "gpu_peak_process_bytes": monitored.peak_gpu_process_bytes,
        "gpu_counter_status": monitored.gpu_counter_status,
        "monitor_owner": "nvidia_smi_process_poll",
        "monitor_interval_seconds": poll_interval_seconds,
        "concurrent_use_preflight": concurrent_use_preflight,
    }


def _scientific_comparison_document(
    *,
    relationship,
    observations: Mapping[MeasurementProfileId, LaneObservation],
) -> dict[str, object]:
    native = observations["native_cpu"]

    def compare(
        cpu_profile: MeasurementProfileId,
        gpu_profile: MeasurementProfileId,
        execution_intent: ExecutionIntent,
    ) -> dict[str, object]:
        if relationship.case_id == _SINGLE_STAGE_CASE_ID:
            expected_driver = (
                JAX_FAST_DRIVER_ID
                if execution_intent == "fast"
                else JAX_PARITY_DRIVER_ID
            )
            for profile_id in (cpu_profile, gpu_profile):
                actual_driver = observations[profile_id].driver
                if actual_driver != expected_driver:
                    raise MeasurementRunnerError(
                        f"{profile_id} driver must be {expected_driver}, "
                        f"got {actual_driver}"
                    )
        result = arbitrate(
            relationship.comparison_routes,
            {
                "native-cpu": native,
                "jax-cpu": observations[cpu_profile],
                "jax-gpu": observations[gpu_profile],
            },
            required_lanes=frozenset(("native-cpu", "jax-cpu", "jax-gpu")),
            expected_workflow_stages=relationship.workflow_stages,
            execution_intent=execution_intent,
        )
        if result.verdict != "pass":
            raise MeasurementRunnerError(
                f"{cpu_profile}/{gpu_profile} scientific comparison failed"
            )
        return {
            "verdict": result.verdict,
            "comparisons": [
                {
                    "phase": comparison.phase,
                    "observable": comparison.observable,
                    "lane_pair": comparison.lane_pair,
                    "passed": comparison.passed,
                    "tolerance_bucket": comparison.tolerance_bucket,
                    "diagnostic": comparison.diagnostic,
                }
                for comparison in result.comparisons
            ],
        }

    return {
        "fast": compare("jax_cpu_fast", "jax_gpu_fast", "fast"),
        "parity": compare("jax_cpu_parity", "jax_gpu_parity", "parity"),
    }


def _git_output(repo_root: Path, arguments: tuple[str, ...]) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def _worktree_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for arguments in (
        ("status", "--porcelain=v1", "-z"),
        ("diff", "--binary", "HEAD"),
    ):
        digest.update(_git_output(repo_root, arguments))
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    return _git_output(repo_root, ("rev-parse", "HEAD")).decode("ascii").strip()


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    raise MeasurementRunnerError("CPU model is unavailable")


def _ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _gpu_identity(gpu_index: int) -> tuple[str, str, str, str]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise MeasurementRunnerError(
            "nvidia-smi device query failed: " + completed.stderr.strip()
        )
    rows = tuple(line for line in completed.stdout.splitlines() if line.strip())
    if len(rows) != 1:
        raise MeasurementRunnerError("nvidia-smi did not identify exactly one GPU")
    fields = tuple(field.strip() for field in rows[0].split(","))
    if len(fields) != 3 or not all(fields):
        raise MeasurementRunnerError("nvidia-smi returned malformed GPU identity")
    banner = subprocess.run(
        ("nvidia-smi",),
        check=False,
        capture_output=True,
        text=True,
    )
    if banner.returncode != 0 or "CUDA Version:" not in banner.stdout:
        raise MeasurementRunnerError("nvidia-smi CUDA version is unavailable")
    cuda_version = banner.stdout.split("CUDA Version:", maxsplit=1)[1].split()[0]
    return fields[0], fields[1], fields[2], cuda_version


def _gpu_load_sample(gpu_index: int) -> tuple[int, int, int]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=utilization.gpu,utilization.memory,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise MeasurementRunnerError(
            "nvidia-smi utilization query failed: " + completed.stderr.strip()
        )
    rows = tuple(line for line in completed.stdout.splitlines() if line.strip())
    if len(rows) != 1:
        raise MeasurementRunnerError(
            "nvidia-smi did not return one GPU utilization row"
        )
    fields = tuple(field.strip() for field in rows[0].split(","))
    if len(fields) != 3:
        raise MeasurementRunnerError("malformed GPU utilization row")
    try:
        gpu_percent, memory_percent, memory_mib = (
            int(float(field)) for field in fields
        )
    except ValueError as error:
        raise MeasurementRunnerError("malformed GPU utilization row") from error
    return (
        gpu_percent,
        memory_percent,
        memory_mib * _NVIDIA_MEMORY_MULTIPLIER,
    )


def _gpu_concurrent_use_preflight(gpu_index: int) -> str:
    processes = _nvidia_compute_apps(gpu_index)
    samples: list[tuple[int, int]] = []
    total_memory_values: list[int] = []
    for sample_index in range(_GPU_PREFLIGHT_SAMPLE_COUNT):
        gpu_percent, memory_percent, total_memory_bytes = _gpu_load_sample(gpu_index)
        samples.append((gpu_percent, memory_percent))
        total_memory_values.append(total_memory_bytes)
        if sample_index + 1 < _GPU_PREFLIGHT_SAMPLE_COUNT:
            time.sleep(_GPU_PREFLIGHT_INTERVAL_SECONDS)
    if len(set(total_memory_values)) != 1:
        raise MeasurementRunnerError("GPU total memory changed during preflight")
    return evaluate_gpu_concurrent_load_preflight(
        processes=processes,
        utilization_samples=tuple(samples),
        total_memory_bytes=total_memory_values[0],
    )


def _simsoptpp_identity() -> tuple[str, str]:
    specification = importlib.util.find_spec("simsoptpp")
    if specification is None or specification.origin is None:
        raise MeasurementRunnerError("simsoptpp extension is unavailable")
    path = Path(specification.origin).resolve()
    return str(path), _sha256_file(path)


def _provenance_document(
    *,
    repo_root: Path,
    gpu_index: int,
    poll_interval_seconds: float,
) -> dict[str, object]:
    gpu_model, gpu_uuid, driver_version, cuda_version = _gpu_identity(gpu_index)
    _, simsoptpp_sha256 = _simsoptpp_identity()
    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise MeasurementRunnerError("logical CPU count is unavailable")
    return {
        "repo_commit": _git_commit(repo_root),
        "worktree_sha256": _worktree_sha256(repo_root),
        "python_version": platform.python_version(),
        "simsopt_version": importlib.metadata.version("simsopt"),
        "simsoptpp_sha256": simsoptpp_sha256,
        "jax_version": importlib.metadata.version("jax"),
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "xla_version": f"jaxlib-{importlib.metadata.version('jaxlib')}",
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": cpu_count,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "ram_bytes": _ram_bytes(),
        "thread_environment": {
            name: os.environ.get(name, "unset")
            for name in (
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "gpu_model": gpu_model,
        "gpu_uuid": gpu_uuid,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "monitor_interval_seconds": poll_interval_seconds,
    }


def _profile_document(
    *,
    profile_id: MeasurementProfileId,
    scale: MeasurementScale,
    identity: Mapping[str, str],
    environment_sha256: str,
    timing_samples: Mapping[str, object],
    allocation_sample: object,
) -> dict[str, object]:
    cold = timing_samples["cold"]
    warmup = timing_samples["warmup"]
    warm = timing_samples["warm"]
    if (
        not isinstance(cold, dict)
        or not isinstance(warmup, dict)
        or not isinstance(warm, list)
    ):
        raise MeasurementRunnerError(f"{profile_id} timing samples are incomplete")
    warm_seconds = tuple(float(sample["total_seconds"]) for sample in warm)
    median = float(statistics.median(warm_seconds))
    mad = float(statistics.median(abs(value - median) for value in warm_seconds))
    timing_records = (cold, warmup, *warm)
    peak_rss = max(
        int(sample["peak_process_tree_rss_bytes"]) for sample in timing_records
    )
    allocation_gpu_bytes = (
        int(allocation_sample["gpu_peak_process_bytes"])
        if isinstance(allocation_sample, dict)
        else None
    )
    return {
        "profile_id": profile_id,
        "device": _profile_device(profile_id),
        "intent": _profile_intent(profile_id),
        "scale": scale,
        **dict(identity),
        "scientific_comparison_passed": True,
        "timing_environment": {
            "xla_python_client_preallocate": (
                "not_applicable" if profile_id == "native_cpu" else "true"
            ),
            "persistent_cache_policy": "fresh_isolated",
            "environment_sha256": environment_sha256,
        },
        "timing_samples": dict(timing_samples),
        "allocation_memory_sample": allocation_sample,
        "summary": {
            "cold_total_seconds": cold["total_seconds"],
            "warm_total_seconds_median": median,
            "warm_total_seconds_mad": mad,
            "peak_timing_process_tree_rss_bytes": peak_rss,
            "peak_allocation_gpu_process_bytes": allocation_gpu_bytes,
        },
    }


def _publish_staging_directory(
    staging_directory: Path,
    *,
    artifact_root: Path,
    run_directory_name: str,
) -> Path:
    final_directory = artifact_root / run_directory_name
    staging_directory.rename(final_directory)
    _fsync_directory(artifact_root)
    return final_directory / "artifact.json"


def collect_case_measurements(
    *,
    case_id: str,
    scale: MeasurementScale,
    artifact_root: Path,
    python_executable: str,
    isolated_site: bool = False,
    repo_root: Path = _REPO_ROOT,
    base_environment: Mapping[str, str] = os.environ,
    gpu_index: int = 0,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Collect, validate, and exclusively publish all five profiles for one case."""
    gpu_preflight = _gpu_concurrent_use_preflight(gpu_index)
    contract_pair = load_runtime_contract_pair(
        repo_root / "examples" / "jax" / "manifest.json",
        repo_root / "examples" / "jax" / "parity_manifest.json",
        repo_root=repo_root,
    )
    relationships = tuple(
        relationship
        for relationship in contract_pair.parity.relationships
        if relationship.case_id == case_id
    )
    if len(relationships) != 1:
        raise MeasurementRunnerError(
            f"{case_id} must own exactly one parity relationship"
        )
    relationship = relationships[0]
    example_records = tuple(
        record
        for record in contract_pair.examples
        if record.id == relationship.jax_example_id
    )
    if len(example_records) != 1:
        raise MeasurementRunnerError(
            f"{case_id} must own exactly one JAX example record"
        )
    example_record = example_records[0]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
    artifact_root.mkdir(parents=True, exist_ok=True)
    staging_directory = artifact_root / f".partial-{run_id}"
    staging_directory.mkdir()
    case = get_case(case_id)
    input_root = staging_directory / "inputs"
    bundle = case.create_input(input_root, scale)
    input_bundle_path = input_root / "input_bundle.json"
    plan = build_collection_plan(mirror_index=0)
    environments: dict[MeasurementProfileId, dict[str, str]] = {}
    environment_hashes: dict[MeasurementProfileId, str] = {}
    for profile_id in MEASUREMENT_PROFILE_IDS:
        cache_directory = staging_directory / "caches" / profile_id
        cache_directory.mkdir(parents=True)
        environment = build_measurement_environment(
            profile_id,
            allocation_sensitive=False,
            base_environment=base_environment,
            gpu_index=gpu_index,
            repo_root=repo_root,
        )
        if profile_id != "native_cpu":
            environment["JAX_COMPILATION_CACHE_DIR"] = str(cache_directory.resolve())
            environment["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
            environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        environments[profile_id] = environment
        environment_hashes[profile_id] = _environment_sha256(environment)
    samples_by_profile: dict[MeasurementProfileId, dict[str, object]] = {
        profile_id: {"cold": None, "warmup": None, "warm": []}
        for profile_id in MEASUREMENT_PROFILE_IDS
    }
    allocation_by_profile: dict[MeasurementProfileId, object] = {
        profile_id: None for profile_id in MEASUREMENT_PROFILE_IDS
    }
    cold_observations: dict[MeasurementProfileId, LaneObservation] = {}
    for sequence_index, run in enumerate(plan):
        profile_id = run.profile_id
        if profile_id == "jax_gpu_optax":
            raise MeasurementRunnerError(
                "legacy measurement plan cannot contain jax_gpu_optax"
            )
        environment = (
            build_measurement_environment(
                profile_id,
                allocation_sensitive=True,
                base_environment=base_environment,
                gpu_index=gpu_index,
                repo_root=repo_root,
            )
            if run.allocation_sensitive
            else environments[profile_id]
        )
        monitored, observation = _execute_collection_run(
            case_id=case_id,
            scale=scale,
            input_bundle_path=input_bundle_path,
            staging_directory=staging_directory,
            run=run,
            sequence_index=sequence_index,
            environment=environment,
            python_executable=python_executable,
            isolated_site=isolated_site,
            repo_root=repo_root,
            gpu_index=gpu_index,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
            input_fingerprint=bundle.input_fingerprint,
        )
        if run.phase == "allocation_memory":
            allocation_by_profile[profile_id] = _allocation_sample_document(
                monitored,
                poll_interval_seconds=poll_interval_seconds,
                concurrent_use_preflight=gpu_preflight,
            )
            continue
        sample = _timing_sample_document(run, monitored)
        profile_samples = samples_by_profile[profile_id]
        if run.phase == "warm":
            warm_samples = profile_samples["warm"]
            if not isinstance(warm_samples, list):
                raise MeasurementRunnerError("warm sample owner is not a list")
            warm_samples.append(sample)
        else:
            profile_samples[run.phase] = sample
        if run.phase == "cold":
            cold_observations[profile_id] = observation
    if tuple(cold_observations) != MEASUREMENT_PROFILE_IDS:
        raise MeasurementRunnerError("cold observations are incomplete or unordered")
    scientific_comparison = _scientific_comparison_document(
        relationship=relationship,
        observations=cold_observations,
    )
    comparison_sha256 = hashlib.sha256(
        _canonical_json_bytes(scientific_comparison)
    ).hexdigest()
    scientific_path = staging_directory / "scientific_comparison.json"
    _write_json_exclusive(scientific_path, scientific_comparison)
    native_source_path = repo_root / "examples" / relationship.native_source
    jax_source_path = repo_root / "examples" / "jax" / example_record.path
    identity = {
        "input_sha256": bundle.input_fingerprint,
        "native_source_sha256": _sha256_file(native_source_path),
        "jax_source_sha256": _sha256_file(jax_source_path),
        "scientific_comparison_sha256": comparison_sha256,
    }
    profiles = {
        profile_id: _profile_document(
            profile_id=profile_id,
            scale=scale,
            identity=identity,
            environment_sha256=environment_hashes[profile_id],
            timing_samples=samples_by_profile[profile_id],
            allocation_sample=allocation_by_profile[profile_id],
        )
        for profile_id in MEASUREMENT_PROFILE_IDS
    }
    schedule = build_measurement_schedule(0)
    artifact = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "evidence_kind": MEASUREMENT_EVIDENCE_KIND,
        "certification_eligible": False,
        "claim_policy": {
            "performance_threshold": None,
            "memory_threshold": None,
            "cross_device_speedup_claim": False,
            "rss_vram_ratio_claim": False,
        },
        "mirror_id": relationship.jax_example_id,
        "scale": scale,
        "profile_ids": list(MEASUREMENT_PROFILE_IDS),
        "identity": identity,
        "provenance": _provenance_document(
            repo_root=repo_root,
            gpu_index=gpu_index,
            poll_interval_seconds=poll_interval_seconds,
        ),
        "schedule": {
            "cold": list(schedule.cold),
            "warmup": list(schedule.warmup),
            "warm": [list(order) for order in schedule.warm],
        },
        "profiles": profiles,
    }
    validate_measurement_artifact(artifact)
    _write_json_exclusive(staging_directory / "artifact.json", artifact)
    return _publish_staging_directory(
        staging_directory,
        artifact_root=artifact_root,
        run_directory_name=run_id,
    )


def _single_stage_campaign_workspace(artifact_root: Path) -> Path:
    if artifact_root.exists():
        raise MeasurementRunnerError(
            "single-stage campaign artifact root must not already exist"
        )
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    workspace = artifact_root.parent / (
        f".{artifact_root.name}.partial-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"
    )
    workspace.mkdir()
    return workspace


def _validate_single_stage_campaign_artifact_root(artifact_root: Path) -> Path:
    resolved_root = artifact_root.resolve()
    temporary_root = Path("/tmp").resolve()
    if resolved_root.is_relative_to(temporary_root):
        raise MeasurementRunnerError(
            "single-stage campaign artifacts must not be written under /tmp"
        )
    return resolved_root


def _single_stage_trajectory(path: Path) -> tuple[TrajectoryPoint, ...]:
    if not path.is_file():
        raise MeasurementRunnerError(f"missing single-stage trajectory: {path}")
    points: list[TrajectoryPoint] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            raw_point = json.loads(line)
        except json.JSONDecodeError as error:
            raise MeasurementRunnerError(
                f"malformed trajectory record {path}:{line_number}"
            ) from error
        if not isinstance(raw_point, dict):
            raise MeasurementRunnerError(
                f"trajectory record {path}:{line_number} must be an object"
            )
        try:
            points.append(
                TrajectoryPoint(
                    iteration=int(raw_point["iteration"]),
                    objective=float(raw_point["objective"]),
                    wall_seconds_from_start=float(raw_point["wall_seconds_from_start"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MeasurementRunnerError(
                f"malformed trajectory record {path}:{line_number}"
            ) from error
    if not points:
        raise MeasurementRunnerError(f"empty single-stage trajectory: {path}")
    return tuple(points)


def _validate_single_stage_trajectory_count(
    *,
    profile_id: RunnerProfileId,
    phase: SamplePhase,
    trajectory: tuple[TrajectoryPoint, ...],
    observation: LaneObservation,
    iteration_budget: int,
) -> None:
    if observation.nit is None or len(trajectory) != observation.nit:
        raise MeasurementRunnerError(
            f"{profile_id} {phase} trajectory has {len(trajectory)} "
            f"records for reported nit={observation.nit}"
        )
    if (
        observation.nit != iteration_budget
        or trajectory[-1].iteration != iteration_budget
    ):
        raise MeasurementRunnerError(
            f"{profile_id} {phase} trajectory must end exactly at "
            f"iteration budget {iteration_budget}"
        )


def _single_stage_singleton(
    observation: LaneObservation,
    value_key: str,
    *,
    scalar_type: type[np.generic],
    description: str,
) -> np.ndarray:
    try:
        value = np.asarray(observation.values[value_key])
    except KeyError as error:
        raise MeasurementRunnerError(
            f"{observation.lane} omitted required {value_key}"
        ) from error
    if value.shape not in ((), (1,)) or value.dtype != np.dtype(scalar_type):
        raise MeasurementRunnerError(
            f"{observation.lane} required {value_key} must be exactly one {description}"
        )
    return value


def _single_stage_scalar(observation: LaneObservation, observable: str) -> float:
    value_key = f"final:{observable}"
    value = _single_stage_singleton(
        observation,
        value_key,
        scalar_type=np.float64,
        description="FP64 scalar",
    )
    scalar = float(value.item())
    if not math.isfinite(scalar):
        raise MeasurementRunnerError(
            f"{observation.lane} required {value_key} must be finite"
        )
    return scalar


def _validate_single_stage_finite_vector(
    observation: LaneObservation, observable: Literal["gradient", "parameters"]
) -> None:
    value_key = f"final:{observable}"
    try:
        value = np.asarray(observation.values[value_key])
    except KeyError as error:
        raise MeasurementRunnerError(
            f"{observation.lane} omitted required {value_key}"
        ) from error
    if value.dtype != np.dtype(np.float64) or value.size == 0:
        raise MeasurementRunnerError(
            f"{observation.lane} required {value_key} must be a nonempty FP64 array"
        )
    if not bool(np.all(np.isfinite(value))):
        raise MeasurementRunnerError(
            f"{observation.lane} required {value_key} must be finite"
        )


def _single_stage_array_sha256(observation: LaneObservation, value_key: str) -> str:
    try:
        value = np.asarray(observation.values[value_key])
    except KeyError as error:
        raise MeasurementRunnerError(
            f"{observation.lane} omitted required {value_key}"
        ) from error
    if value.dtype != np.dtype(np.float64) or value.size == 0:
        raise MeasurementRunnerError(
            f"{observation.lane} required {value_key} must be a nonempty FP64 array"
        )
    if not bool(np.all(np.isfinite(value))):
        raise MeasurementRunnerError(
            f"{observation.lane} required {value_key} must be finite"
        )
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _single_stage_bool(observation: LaneObservation, value_key: str) -> bool:
    value = _single_stage_singleton(
        observation,
        value_key,
        scalar_type=np.bool_,
        description="boolean scalar",
    )
    return bool(value.item())


def _single_stage_int(observation: LaneObservation, value_key: str) -> int:
    value = _single_stage_singleton(
        observation,
        value_key,
        scalar_type=np.int64,
        description="int64 scalar",
    )
    return int(value.item())


def _single_stage_endpoint_audit(
    profile_id: RunnerProfileId,
    observation: LaneObservation,
) -> EndpointAudit:
    counters = (observation.nit, observation.nfev, observation.njev)
    if any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in counters
    ):
        raise MeasurementRunnerError(
            f"{observation.lane} omitted valid optimizer counters"
        )
    nit, nfev, njev = counters
    assert nit is not None and nfev is not None and njev is not None
    provenance = observation.provenance
    if provenance is None:
        raise MeasurementRunnerError(f"{profile_id} omitted required provenance")
    observed_route_selector = provenance.lane_environment_policy.get(
        _EXACT_ADJOINT_ENVIRONMENT_NAME
    )
    if profile_id == "native_cpu":
        if observed_route_selector is not None:
            raise MeasurementRunnerError(
                "native_cpu provenance contains a JAX exact-adjoint selector"
            )
        adjoint_route = None
    else:
        if observed_route_selector != "1":
            raise MeasurementRunnerError(
                f"{profile_id} did not observe the direct exact-adjoint selector"
            )
        adjoint_route = DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE
    final_gradient = np.asarray(observation.values["final:gradient"])
    return EndpointAudit(
        backend_mode=observation.backend_mode,
        driver=observation.driver,
        input_fingerprint=observation.input_fingerprint,
        configuration_fingerprint=observation.configuration_fingerprint,
        effective_construction_fingerprint=(
            observation.effective_construction_fingerprint
        ),
        initial_parameters_sha256=_single_stage_array_sha256(
            observation, "initial:parameters"
        ),
        final_parameters_sha256=_single_stage_array_sha256(
            observation, "final:parameters"
        ),
        final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
        normalized_status=observation.normalized_status,
        raw_status=observation.raw_status,
        nit=nit,
        nfev=nfev,
        njev=njev,
        adjoint_route=adjoint_route,
        certificate=EndpointCertificateAudit(
            success=_single_stage_bool(
                observation, "final:endpoint_certificate_success"
            ),
            initial_stationary=_single_stage_bool(
                observation, "final:endpoint_initial_stationary"
            ),
            terminal_stationary=_single_stage_bool(
                observation, "final:endpoint_terminal_stationary"
            ),
            constraints_satisfied=_single_stage_bool(
                observation, "final:endpoint_constraints_satisfied"
            ),
            outer_status=_single_stage_int(observation, "final:outer_solver_status"),
        ),
    )


def _validate_single_stage_campaign_observation(
    observation: LaneObservation,
    *,
    profile_id: RunnerProfileId,
    input_fingerprint: str,
) -> None:
    """Validate campaign evidence without relaxing the ordinary parity arbiter."""
    expected_lane = (
        "native-cpu"
        if profile_id == "native_cpu"
        else "jax-gpu"
        if _profile_device(profile_id) == "gpu"
        else "jax-cpu"
    )
    if observation.lane != expected_lane:
        raise MeasurementRunnerError(
            f"{profile_id} emitted lane {observation.lane}, expected {expected_lane}"
        )
    if observation.backend_mode != _expected_backend_mode(profile_id):
        raise MeasurementRunnerError(
            f"{profile_id} emitted backend mode {observation.backend_mode}"
        )
    if observation.platform != _profile_device(profile_id):
        raise MeasurementRunnerError(
            f"{profile_id} emitted platform {observation.platform}"
        )
    if observation.precision != "fp64":
        raise MeasurementRunnerError(
            f"{profile_id} emitted precision {observation.precision}"
        )
    if observation.scale != "native_default":
        raise MeasurementRunnerError(f"{profile_id} did not use native_default")
    if observation.input_fingerprint != input_fingerprint:
        raise MeasurementRunnerError(
            f"{profile_id} did not consume the canonical input bundle"
        )
    expected_driver = _SINGLE_STAGE_EXPECTED_DRIVERS[profile_id]
    if observation.driver != expected_driver:
        raise MeasurementRunnerError(
            f"{profile_id} driver must be {expected_driver}, got {observation.driver}"
        )
    if observation.normalized_status not in ("converged", "budget_exhausted"):
        raise MeasurementRunnerError(
            f"{profile_id} has invalid campaign status {observation.normalized_status}"
        )
    if not _single_stage_bool(observation, "final:inner_solver_success"):
        raise MeasurementRunnerError(
            f"{profile_id} did not report inner solver success"
        )
    for _, observable in _SINGLE_STAGE_PARITY_OBSERVABLES:
        _single_stage_scalar(observation, observable)
    for observable in ("gradient", "parameters"):
        _validate_single_stage_finite_vector(observation, observable)
    provenance = observation.provenance
    if (
        provenance is None
        or not provenance.repository_commit
        or not provenance.executed_sources
        or not provenance.python_version
        or not provenance.jax_version
    ):
        raise MeasurementRunnerError(f"{profile_id} omitted required provenance")
    expected_sync = (
        "native synchronous execution"
        if profile_id == "native_cpu"
        else "jax.block_until_ready over published observation values"
    )
    if provenance.measurement_synchronization != expected_sync:
        raise MeasurementRunnerError(f"{profile_id} omitted required synchronization")
    observed_route_selector = provenance.lane_environment_policy.get(
        _EXACT_ADJOINT_ENVIRONMENT_NAME
    )
    if profile_id == "native_cpu":
        if observed_route_selector is not None:
            raise MeasurementRunnerError(
                "native_cpu provenance contains a JAX exact-adjoint selector"
            )
    else:
        if observed_route_selector != "1":
            raise MeasurementRunnerError(
                f"{profile_id} did not observe the direct exact-adjoint selector"
            )
        if provenance.lane_environment_policy.get(
            "SIMSOPT_BACKEND_MODE"
        ) != _expected_backend_mode(profile_id):
            raise MeasurementRunnerError(
                f"{profile_id} provenance backend policy mismatch"
            )
        expected_platforms = (
            {"cuda", "gpu"} if _profile_device(profile_id) == "gpu" else {"cpu"}
        )
        if not {device.platform for device in provenance.devices} & expected_platforms:
            raise MeasurementRunnerError(f"{profile_id} device provenance mismatch")


def _validate_single_stage_campaign_identity_pair(
    reference: LaneObservation,
    profile_id: RunnerProfileId,
    observation: LaneObservation,
) -> None:
    reference_provenance = reference.provenance
    provenance = observation.provenance
    assert reference_provenance is not None
    assert provenance is not None
    for field in (
        "input_fingerprint",
        "configuration_fingerprint",
        "effective_construction_fingerprint",
    ):
        if getattr(observation, field) != getattr(reference, field):
            raise MeasurementRunnerError(
                f"{profile_id} {field} does not match native_cpu"
            )
    try:
        native_initial_parameters = np.asarray(reference.values["initial:parameters"])
        lane_initial_parameters = np.asarray(observation.values["initial:parameters"])
    except KeyError as error:
        raise MeasurementRunnerError(
            f"{profile_id} omitted matched initial parameters"
        ) from error
    if (
        native_initial_parameters.dtype != np.dtype(np.float64)
        or lane_initial_parameters.dtype != np.dtype(np.float64)
        or native_initial_parameters.size == 0
        or not bool(np.all(np.isfinite(native_initial_parameters)))
        or not np.array_equal(lane_initial_parameters, native_initial_parameters)
    ):
        raise MeasurementRunnerError(
            f"{profile_id} initial parameters do not exactly match native_cpu"
        )
    for field in (
        "repository_commit",
        "repository_dirty",
        "tracked_diff_sha256",
        "untracked_files",
        "python_version",
        "jax_version",
    ):
        if getattr(provenance, field) != getattr(reference_provenance, field):
            raise MeasurementRunnerError(
                f"{profile_id} {field} does not match native_cpu"
            )
    reference_sources = {
        source.path: source.sha256 for source in reference_provenance.executed_sources
    }
    lane_sources = {
        source.path: source.sha256 for source in provenance.executed_sources
    }
    for source_path in sorted(set(reference_sources) & set(lane_sources)):
        if reference_sources[source_path] != lane_sources[source_path]:
            raise MeasurementRunnerError(
                f"{profile_id} executed source does not match native_cpu: {source_path}"
            )


def _validate_single_stage_campaign_identity(
    observations: Mapping[RunnerProfileId, LaneObservation],
) -> None:
    reference = observations["native_cpu"]
    for profile_id in SINGLE_STAGE_SPEED_PROFILE_IDS[1:]:
        _validate_single_stage_campaign_identity_pair(
            reference,
            profile_id,
            observations[profile_id],
        )


def _execute_single_stage_speed_run(
    *,
    bundle_path: Path,
    workspace: Path,
    run: CollectionRun,
    sequence_index: int,
    environment: Mapping[str, str],
    python_executable: str,
    isolated_site: bool,
    repo_root: Path,
    gpu_index: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    immutable_snapshot_provenance_path: Path | None = None,
) -> tuple[MonitoredCommandResult, LaneObservation, Path, float]:
    stem = _sample_stem(run, sequence_index)
    result_directory = workspace / "receipts" / stem
    result_directory.mkdir(parents=True)
    trajectory_path = result_directory / "trajectory.jsonl"
    optimization_timing_path = result_directory / "optimization_timing.json"
    command = build_profile_command(
        python_executable=python_executable,
        case_id=_SINGLE_STAGE_CASE_ID,
        profile_id=run.profile_id,
        input_bundle_path=bundle_path,
        result_directory=result_directory,
        scale="native_default",
        trajectory_path=trajectory_path,
        optimization_timing_path=optimization_timing_path,
        immutable_snapshot_provenance_path=immutable_snapshot_provenance_path,
        isolated_site=isolated_site,
    )
    child_environment = environment
    child_cwd = repo_root
    if immutable_snapshot_provenance_path is not None:
        if isolated_site:
            raise MeasurementRunnerError(
                "immutable snapshot execution is incompatible with isolated_site"
            )
        module = "examples.jax.parity.child"
        try:
            module_index = command.index(module)
        except ValueError as error:
            raise MeasurementRunnerError(
                "profile command omitted the canonical parity child"
            ) from error
        launch = build_snapshot_module_launch(
            Path(python_executable),
            repo_root,
            module,
            command[module_index + 1 :],
            environment,
        )
        command = launch.argv
        child_environment = launch.environment
        child_cwd = launch.cwd
    monitored = execute_monitored_command(
        command=command,
        environment=child_environment,
        cwd=child_cwd,
        stdout_path=workspace / "logs" / f"{stem}.stdout",
        stderr_path=workspace / "logs" / f"{stem}.stderr",
        device=_profile_device(run.profile_id),
        gpu_index=gpu_index,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    if monitored.returncode != 0:
        raise MeasurementRunnerError(
            f"{run.profile_id} {run.phase} failed with {monitored.termination}; "
            f"see logs/{stem}.stderr"
        )
    try:
        optimization_timing = read_optimization_window_timing(optimization_timing_path)
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        raise MeasurementRunnerError(
            f"{run.profile_id} {run.phase} has an invalid optimizer timing sidecar"
        ) from error
    if optimization_timing.wall_seconds > monitored.wall_seconds:
        raise MeasurementRunnerError(
            f"{run.profile_id} {run.phase} optimizer timing exceeds subprocess wall"
        )
    _write_json_exclusive(
        workspace / "logs" / f"{stem}.timing.json",
        {
            "optimization_wall_seconds": optimization_timing.wall_seconds,
            "subprocess_wall_seconds": monitored.wall_seconds,
        },
    )
    return (
        monitored,
        load_lane_observation(result_directory),
        trajectory_path,
        optimization_timing.wall_seconds,
    )


def _single_stage_campaign_parity_rows(
    *,
    profile_id: RunnerProfileId,
    observations: Mapping[RunnerProfileId, LaneObservation],
) -> tuple[ParityRow, ...]:
    """Compare the five campaign observables under the frozen tolerance SSOT."""
    if profile_id == "native_cpu":
        return ()
    rows: list[ParityRow] = []
    for receipt_observable, observable in _SINGLE_STAGE_PARITY_OBSERVABLES:
        native_value = _single_stage_scalar(observations["native_cpu"], observable)
        lane_value = _single_stage_scalar(observations[profile_id], observable)
        tolerance = single_stage_speed_parity_tolerance(native_value)
        if abs(lane_value - native_value) > tolerance:
            raise MeasurementRunnerError(
                f"{profile_id} direct parity failed for final:{observable}"
            )
        rows.append(
            ParityRow(
                observable=receipt_observable,
                native_value=native_value,
                lane_value=lane_value,
                tolerance=tolerance,
            )
        )
    return tuple(rows)


def collect_single_stage_speed_campaign(
    *,
    artifact_root: Path,
    python_executable: str,
    isolated_site: bool = False,
    repo_root: Path = _REPO_ROOT,
    base_environment: Mapping[str, str] = os.environ,
    gpu_index: int = 0,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Collect one four-lane native-default campaign at a caller-owned root.

    Fixed-budget endpoints are eligible when direct final-observable parity,
    FP64 finiteness, and the inner solver succeed.
    """
    artifact_root = _validate_single_stage_campaign_artifact_root(artifact_root)
    _gpu_concurrent_use_preflight(gpu_index)
    workspace = _single_stage_campaign_workspace(artifact_root)
    case = get_case(_SINGLE_STAGE_CASE_ID)
    bundle = case.create_input(workspace / "inputs", "native_default")
    iteration_budget = bundle.configuration["outer_maxiter"]
    if (
        isinstance(iteration_budget, bool)
        or not isinstance(iteration_budget, int)
        or iteration_budget <= 0
    ):
        raise MeasurementRunnerError(
            "single-stage iteration budget must be a positive integer"
        )
    plan = build_single_stage_speed_collection_plan(mirror_index=0)
    samples: dict[RunnerProfileId, list[SampleMeasurement]] = {
        profile_id: [] for profile_id in SINGLE_STAGE_SPEED_PROFILE_IDS
    }
    endpoints: dict[RunnerProfileId, LaneObservation] = {}
    campaign_reference: LaneObservation | None = None
    environments: dict[RunnerProfileId, dict[str, str]] = {}
    for profile_id in SINGLE_STAGE_SPEED_PROFILE_IDS:
        environment = build_measurement_environment(
            profile_id,
            allocation_sensitive=False,
            base_environment=base_environment,
            gpu_index=gpu_index,
            repo_root=repo_root,
        )
        if profile_id != "native_cpu":
            cache_directory = workspace / "caches" / profile_id
            cache_directory.mkdir(parents=True)
            environment[_EXACT_ADJOINT_ENVIRONMENT_NAME] = "1"
            environment["JAX_COMPILATION_CACHE_DIR"] = str(cache_directory.resolve())
            environment["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
            environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        environments[profile_id] = environment
    for sequence_index, run in enumerate(plan):
        (
            _monitored,
            observation,
            trajectory_path,
            optimization_wall_seconds,
        ) = _execute_single_stage_speed_run(
            bundle_path=workspace / "inputs" / "input_bundle.json",
            workspace=workspace,
            run=run,
            sequence_index=sequence_index,
            environment=environments[run.profile_id],
            python_executable=python_executable,
            isolated_site=isolated_site,
            repo_root=repo_root,
            gpu_index=gpu_index,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
        _validate_single_stage_campaign_observation(
            observation,
            profile_id=run.profile_id,
            input_fingerprint=bundle.input_fingerprint,
        )
        if campaign_reference is None:
            campaign_reference = observation
        else:
            _validate_single_stage_campaign_identity_pair(
                campaign_reference,
                run.profile_id,
                observation,
            )
        if run.phase == "allocation_memory":
            raise MeasurementRunnerError(
                "single-stage timing campaign must not include allocation runs"
            )
        phase: SamplePhase = run.phase
        sample_index = 0 if run.sample_index is None else run.sample_index
        trajectory = _single_stage_trajectory(trajectory_path)
        _validate_single_stage_trajectory_count(
            profile_id=run.profile_id,
            phase=phase,
            trajectory=trajectory,
            observation=observation,
            iteration_budget=iteration_budget,
        )
        if trajectory[-1].wall_seconds_from_start > optimization_wall_seconds:
            raise MeasurementRunnerError(
                f"{run.profile_id} {phase} trajectory exceeds its optimizer window"
            )
        samples[run.profile_id].append(
            SampleMeasurement(
                phase=phase,
                sample_index=sample_index,
                wall_seconds=optimization_wall_seconds,
                trajectory=trajectory,
            )
        )
        if run.phase == "warm" and run.sample_index == WARM_SAMPLE_COUNT - 1:
            endpoints[run.profile_id] = observation
    if set(endpoints) != set(SINGLE_STAGE_SPEED_PROFILE_IDS):
        raise MeasurementRunnerError(
            "single-stage campaign did not retain last warm endpoints"
        )
    _validate_single_stage_campaign_identity(endpoints)
    lane_receipts = tuple(
        LaneReceipt(
            lane_id=_SINGLE_STAGE_RECEIPT_LANE_IDS[profile_id],
            samples=tuple(samples[profile_id]),
            endpoint=LaneEndpoint(
                observables=EndpointObservables(
                    final_objective=_single_stage_scalar(
                        endpoints[profile_id], "objective"
                    ),
                    final_iota=_single_stage_scalar(endpoints[profile_id], "iota"),
                    final_volume=_single_stage_scalar(endpoints[profile_id], "volume"),
                    final_non_qs_ratio=_single_stage_scalar(
                        endpoints[profile_id], "non_qs_ratio"
                    ),
                    final_boozer_residual=_single_stage_scalar(
                        endpoints[profile_id], "boozer_residual"
                    ),
                    inner_solver_success=_single_stage_bool(
                        endpoints[profile_id], "final:inner_solver_success"
                    ),
                ),
                precision="fp64",
                audit=_single_stage_endpoint_audit(
                    profile_id,
                    endpoints[profile_id],
                ),
                parity_rows=_single_stage_campaign_parity_rows(
                    profile_id=profile_id,
                    observations=endpoints,
                ),
            ),
        )
        for profile_id in SINGLE_STAGE_SPEED_PROFILE_IDS
    )
    gpu_model, _, _, _ = _gpu_identity(gpu_index)
    hostname = platform.node()
    if not hostname:
        raise MeasurementRunnerError("hostname is unavailable")
    runtime_provenance = endpoints["native_cpu"].provenance
    assert runtime_provenance is not None
    runtime_jax_version = runtime_provenance.jax_version
    assert runtime_jax_version is not None
    receipt = CampaignReceipt(
        metadata=CampaignMetadata(
            campaign_id="single-stage-speed-20260804",
            git_describe=_git_output(repo_root, ("describe", "--always", "--dirty"))
            .decode("utf-8")
            .strip(),
            hostname=hostname,
            device_name=gpu_model,
            python_version=runtime_provenance.python_version,
            jax_version=runtime_jax_version,
            iteration_budget=iteration_budget,
            scale="native_default",
            created_utc=datetime.now(UTC).isoformat(),
        ),
        lanes=lane_receipts,
    )
    return write_campaign_receipt(artifact_root, receipt)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--single-stage-speed-campaign", action="store_true")
    parser.add_argument(
        "--scale",
        choices=("bounded", "native_default"),
        default="bounded",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--isolated-site", action="store_true")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=_DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Collect a legacy five-profile artifact or the four-lane speed campaign."""
    options = _argument_parser().parse_args(arguments)
    scale: MeasurementScale = options.scale
    if options.single_stage_speed_campaign:
        if options.case is not None and options.case != _SINGLE_STAGE_CASE_ID:
            _argument_parser().error(
                "--case must be omitted or name the single-stage campaign case"
            )
        if scale != "native_default":
            _argument_parser().error(
                "--single-stage-speed-campaign requires --scale native_default"
            )
        artifact_path = collect_single_stage_speed_campaign(
            artifact_root=options.artifact_root.resolve(),
            python_executable=sys.executable,
            isolated_site=options.isolated_site,
            gpu_index=options.gpu_index,
            poll_interval_seconds=options.poll_interval_seconds,
            timeout_seconds=options.timeout_seconds,
        )
    else:
        if options.case is None:
            _argument_parser().error(
                "--case is required without --single-stage-speed-campaign"
            )
        artifact_path = collect_case_measurements(
            case_id=options.case,
            scale=scale,
            artifact_root=options.artifact_root.resolve(),
            python_executable=sys.executable,
            isolated_site=options.isolated_site,
            gpu_index=options.gpu_index,
            poll_interval_seconds=options.poll_interval_seconds,
            timeout_seconds=options.timeout_seconds,
        )
    print(artifact_path)
    return 0


__all__ = [
    "MEASUREMENT_EVIDENCE_KIND",
    "MEASUREMENT_SCHEMA_VERSION",
    "SINGLE_STAGE_SPEED_PROFILE_IDS",
    "CollectionRun",
    "MeasurementRunnerError",
    "MeasurementSchedule",
    "MonitoredCommandResult",
    "build_collection_plan",
    "build_measurement_environment",
    "build_measurement_schedule",
    "build_profile_command",
    "build_single_stage_speed_collection_plan",
    "classify_termination",
    "collect_case_measurements",
    "collect_single_stage_speed_campaign",
    "evaluate_gpu_concurrent_load_preflight",
    "execute_monitored_command",
    "main",
    "parse_nvidia_smi_compute_apps",
    "publish_artifact_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
