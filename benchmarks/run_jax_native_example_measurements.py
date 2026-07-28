"""Collect matched native/JAX timing and peak-memory evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
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
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
for _import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from benchmarks.jax_native_example_measurement_contract import (
    MEASUREMENT_EVIDENCE_KIND,
    MEASUREMENT_PROFILE_IDS,
    MEASUREMENT_SCHEMA_VERSION,
    WARM_SAMPLE_COUNT,
    MeasurementProfileId,
    MeasurementScale,
    validate_measurement_artifact,
)
from examples.jax._lane_environment import build_execution_environment
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.arbiter import LaneObservation, arbitrate
from examples.jax.parity.cases import get_case
from examples.jax.parity.receipts import load_lane_observation

_NVIDIA_MEMORY_MULTIPLIER = 1024 * 1024
_DEFAULT_POLL_INTERVAL_SECONDS = 0.05
_DEFAULT_TIMEOUT_SECONDS = 900.0
_GPU_PREFLIGHT_SAMPLE_COUNT = 5
_GPU_PREFLIGHT_INTERVAL_SECONDS = 0.2
_GPU_CONCURRENT_MEMORY_FRACTION_MAX = 0.05
_GPU_CONCURRENT_UTILIZATION_PERCENT_MAX = 5
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
) -> MonitoredCommandResult:
    """Run one child and poll process-tree RSS plus process-attributed VRAM."""
    if poll_interval_seconds <= 0.0 or timeout_seconds <= 0.0:
        raise ValueError("poll interval and timeout must be positive")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    peak_rss_bytes = 0
    peak_gpu_bytes = 0 if device == "gpu" else None
    gpu_counter_available = True
    timed_out = False
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
            if elapsed_seconds > timeout_seconds:
                timed_out = True
                process.terminate()
                break
            time.sleep(poll_interval_seconds)
        if timed_out:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        returncode = process.wait()
        stdout_stream.flush()
        stderr_stream.flush()
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
    _fsync_directory(stdout_path.parent)
    if timed_out and returncode == 0:
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
        termination=classify_termination(returncode),
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
    profile_id: MeasurementProfileId,
    input_bundle_path: Path,
    result_directory: Path,
    scale: MeasurementScale,
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
    return (
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


def _profile_device(profile_id: MeasurementProfileId) -> Literal["cpu", "gpu"]:
    return "gpu" if profile_id in ("jax_gpu_fast", "jax_gpu_parity") else "cpu"


def _profile_intent(
    profile_id: MeasurementProfileId,
) -> Literal["native", "fast", "parity"]:
    if profile_id == "native_cpu":
        return "native"
    return "parity" if profile_id.endswith("_parity") else "fast"


def _expected_backend_mode(profile_id: MeasurementProfileId) -> str:
    return profile_id


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
    profile_id: MeasurementProfileId,
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
    command = build_profile_command(
        python_executable=python_executable,
        case_id=case_id,
        profile_id=run.profile_id,
        input_bundle_path=input_bundle_path,
        result_directory=result_directory,
        scale=scale,
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
    ) -> dict[str, object]:
        result = arbitrate(
            relationship.comparison_routes,
            {
                "native-cpu": native,
                "jax-cpu": observations[cpu_profile],
                "jax-gpu": observations[gpu_profile],
            },
            required_lanes=frozenset(("native-cpu", "jax-cpu", "jax-gpu")),
            expected_workflow_stages=relationship.workflow_stages,
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
        "fast": compare("jax_cpu_fast", "jax_gpu_fast"),
        "parity": compare("jax_cpu_parity", "jax_gpu_parity"),
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
        environment = (
            build_measurement_environment(
                run.profile_id,
                allocation_sensitive=True,
                base_environment=base_environment,
                gpu_index=gpu_index,
                repo_root=repo_root,
            )
            if run.allocation_sensitive
            else environments[run.profile_id]
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
            allocation_by_profile[run.profile_id] = _allocation_sample_document(
                monitored,
                poll_interval_seconds=poll_interval_seconds,
                concurrent_use_preflight=gpu_preflight,
            )
            continue
        sample = _timing_sample_document(run, monitored)
        profile_samples = samples_by_profile[run.profile_id]
        if run.phase == "warm":
            warm_samples = profile_samples["warm"]
            if not isinstance(warm_samples, list):
                raise MeasurementRunnerError("warm sample owner is not a list")
            warm_samples.append(sample)
        else:
            profile_samples[run.phase] = sample
        if run.phase == "cold":
            cold_observations[run.profile_id] = observation
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


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
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
    """Collect and publish one complete five-profile artifact."""
    options = _argument_parser().parse_args(arguments)
    scale: MeasurementScale = options.scale
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
    "CollectionRun",
    "MEASUREMENT_EVIDENCE_KIND",
    "MEASUREMENT_SCHEMA_VERSION",
    "MeasurementSchedule",
    "MeasurementRunnerError",
    "MonitoredCommandResult",
    "build_collection_plan",
    "build_measurement_environment",
    "build_measurement_schedule",
    "build_profile_command",
    "classify_termination",
    "collect_case_measurements",
    "execute_monitored_command",
    "evaluate_gpu_concurrent_load_preflight",
    "main",
    "parse_nvidia_smi_compute_apps",
    "publish_artifact_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
