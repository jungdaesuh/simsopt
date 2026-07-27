"""Collect matched, non-certifying JAX fast-versus-parity benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
for _import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from benchmarks.jax_example_execution_mode_contract import (
    BENCHMARK_EVIDENCE_KIND,
    BENCHMARK_RULE,
    BENCHMARK_RULE_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    METRIC_OWNERS,
    REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES,
    REPRESENTATIVE_WORKLOAD_IDS,
    WARM_PAIR_COUNT,
    BenchmarkContractError,
    evaluate_benchmark_artifact,
)
from examples.jax import run_examples as example_runner
from examples.jax._lane_environment import build_execution_environment
from examples.jax._manifest import (
    JaxExampleRecord,
    ManifestValidationError,
    load_manifest,
)
from simsopt_jax.config import (
    ExecutionIntent,
    JaxDevice,
    JaxExecutionProfile,
)

Intent = Literal["fast", "parity"]
Phase = Literal["cold", "warmup", "warm"]

_BENCHMARK_ENVIRONMENT_NAMES = frozenset(
    {
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    }
)
_ENVIRONMENT_FINGERPRINT_PREFIXES = ("JAX_", "SIMSOPT_", "XLA_", "TF_GPU_")
_ENVIRONMENT_FINGERPRINT_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONPATH",
    }
)
_NVIDIA_MEMORY_MULTIPLIER = 1024 * 1024
_DEFAULT_POLL_INTERVAL_SECONDS = 0.05
_DEFAULT_CHILD_TIMEOUT_SECONDS = 900.0
_GPU_PREFLIGHT_SAMPLE_INTERVAL_SECONDS = 0.2
_CPU_AOT_INCOMPATIBILITY_MARKER = (
    b"Machine type used for XLA:CPU compilation doesn't match the machine type "
    b"for execution"
)


class BenchmarkRunnerError(RuntimeError):
    """The benchmark runner cannot collect trustworthy matched evidence."""


@dataclass(frozen=True)
class ScheduledRun:
    """One fixed position in the cold/warmup/balanced-warm protocol."""

    intent: Intent
    phase: Phase
    pair_index: int | None
    order_position: int
    measured: bool


@dataclass(frozen=True)
class _MonitoredProcess:
    returncode: int
    wall_seconds: float
    peak_host_rss_bytes: int
    peak_gpu_memory_bytes: int | None
    gpu_memory_available: bool


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _claim_run_directory(artifact_root: Path, run_directory_name: str) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(artifact_root.parent)
    run_directory = artifact_root / run_directory_name
    run_directory.mkdir()
    _fsync_directory(artifact_root)
    return run_directory


def publish_artifact_exclusive(
    document: object,
    *,
    artifact_root: Path,
    run_directory_name: str,
) -> Path:
    """Publish a standalone artifact without replacing any prior evidence."""
    run_directory = _claim_run_directory(artifact_root, run_directory_name)
    artifact_path = run_directory / "artifact.json"
    _write_json_exclusive(artifact_path, document)
    return artifact_path


def build_measurement_schedule(workload_index: int) -> tuple[ScheduledRun, ...]:
    """Return the fixed cold, reverse-warmup, and balanced paired schedule."""
    cold_order: tuple[Intent, Intent] = (
        ("fast", "parity") if workload_index % 2 == 0 else ("parity", "fast")
    )
    runs: list[ScheduledRun] = [
        ScheduledRun(
            intent=intent,
            phase="cold",
            pair_index=None,
            order_position=order_position,
            measured=True,
        )
        for order_position, intent in enumerate(cold_order)
    ]
    runs.extend(
        ScheduledRun(
            intent=intent,
            phase="warmup",
            pair_index=None,
            order_position=order_position,
            measured=False,
        )
        for order_position, intent in enumerate(reversed(cold_order))
    )
    for pair_index in range(WARM_PAIR_COUNT):
        order: tuple[Intent, Intent] = (
            ("fast", "parity")
            if (workload_index + pair_index) % 2 == 0
            else ("parity", "fast")
        )
        runs.extend(
            ScheduledRun(
                intent=intent,
                phase="warm",
                pair_index=pair_index,
                order_position=order_position,
                measured=True,
            )
            for order_position, intent in enumerate(order)
        )
    return tuple(runs)


def _environment_fingerprint(environment: Mapping[str, str]) -> str:
    selected = {
        name: value
        for name, value in environment.items()
        if name in _ENVIRONMENT_FINGERPRINT_NAMES
        or name.startswith(_ENVIRONMENT_FINGERPRINT_PREFIXES)
    }
    return _sha256_bytes(_canonical_json_bytes(selected))


def build_profile_environment(
    device: JaxDevice,
    intent: ExecutionIntent,
    *,
    cache_directory: Path,
    base_environment: Mapping[str, str],
    repo_root: Path,
    gpu_index: int = 0,
) -> tuple[JaxExecutionProfile, dict[str, str], str]:
    """Build a pinned measured profile with an owned persistent-cache path."""
    scrubbed = {
        name: value
        for name, value in base_environment.items()
        if name not in _BENCHMARK_ENVIRONMENT_NAMES
    }
    profile, environment = build_execution_environment(
        device,
        intent,
        scrubbed,
        repo_root=repo_root,
    )
    if device == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(cache_directory.resolve()),
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        }
    )
    return profile, environment, _environment_fingerprint(environment)


def classify_termination(returncode: int) -> str:
    """Classify process termination without treating SIGKILL as ordinary failure."""
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
        parts = tuple(part.strip() for part in line.split(","))
        if len(parts) != 2:
            raise BenchmarkRunnerError(f"malformed nvidia-smi process row: {line!r}")
        memory_text = parts[1].removesuffix(" MiB").strip()
        try:
            pid = int(parts[0])
            memory_mib = int(memory_text)
        except ValueError as error:
            raise BenchmarkRunnerError(
                f"malformed nvidia-smi process row: {line!r}"
            ) from error
        if pid <= 0 or memory_mib < 0:
            raise BenchmarkRunnerError(f"malformed nvidia-smi process row: {line!r}")
        result[pid] = memory_mib * _NVIDIA_MEMORY_MULTIPLIER
    return result


def evaluate_gpu_concurrent_load_preflight(
    *,
    processes: Mapping[int, int],
    utilization_samples: Sequence[tuple[int, int]],
    total_memory_bytes: int,
) -> dict[str, str]:
    """Admit only bounded background GPU load and retain its exact snapshot."""
    sample_count = int(BENCHMARK_RULE["gpu_concurrent_sample_count"])
    if len(utilization_samples) != sample_count:
        raise BenchmarkRunnerError(
            f"GPU concurrent-load preflight requires {sample_count} samples"
        )
    if total_memory_bytes <= 0:
        raise BenchmarkRunnerError("GPU total memory must be positive")
    if any(
        gpu_percent < 0
        or gpu_percent > 100
        or memory_percent < 0
        or memory_percent > 100
        for gpu_percent, memory_percent in utilization_samples
    ):
        raise BenchmarkRunnerError("GPU utilization samples must be percentages")
    max_gpu_percent = max(sample[0] for sample in utilization_samples)
    max_memory_percent = max(sample[1] for sample in utilization_samples)
    utilization_limit = int(BENCHMARK_RULE["gpu_concurrent_utilization_percent_max"])
    if max_gpu_percent > utilization_limit:
        raise BenchmarkRunnerError(
            "GPU concurrent utilization exceeds the checked-in limit: "
            f"{max_gpu_percent}% > {utilization_limit}%"
        )
    process_memory_bytes = sum(processes.values())
    memory_fraction = process_memory_bytes / total_memory_bytes
    memory_fraction_limit = float(BENCHMARK_RULE["gpu_concurrent_memory_fraction_max"])
    if memory_fraction > memory_fraction_limit:
        raise BenchmarkRunnerError(
            "GPU concurrent process memory exceeds the checked-in limit: "
            f"{memory_fraction:.6g} > {memory_fraction_limit:.6g}"
        )
    process_detail = (
        ",".join(f"{pid}:{processes[pid]}" for pid in sorted(processes))
        if processes
        else "none"
    )
    return {
        "status": "pass",
        "detail": (
            f"background_processes={process_detail};"
            f"background_memory_fraction={memory_fraction:.9g};"
            f"max_gpu_utilization_percent={max_gpu_percent};"
            f"max_memory_utilization_percent={max_memory_percent};"
            f"samples={sample_count}"
        ),
    }


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
        raise BenchmarkRunnerError(
            "nvidia-smi process query failed: " + completed.stderr.strip()
        )
    return parse_nvidia_smi_compute_apps(completed.stdout)


def _process_tree_pids(root_pid: int) -> frozenset[int]:
    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        children_path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
        try:
            children = children_path.read_text(encoding="ascii").split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        pending.extend(int(child) for child in children)
    return frozenset(observed)


def _process_rss_bytes(pid: int) -> int:
    status_path = Path("/proc") / str(pid) / "status"
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


def _sample_process_tree(
    root_pid: int,
    *,
    device: JaxDevice,
    gpu_index: int,
) -> tuple[int, int | None, bool]:
    pids = _process_tree_pids(root_pid)
    host_rss_bytes = sum(_process_rss_bytes(pid) for pid in pids)
    if device == "cpu":
        return host_rss_bytes, None, True
    try:
        memory_by_pid = _nvidia_compute_apps(gpu_index)
    except BenchmarkRunnerError:
        return host_rss_bytes, None, False
    return host_rss_bytes, sum(memory_by_pid.get(pid, 0) for pid in pids), True


def _monitor_process(
    process: subprocess.Popen[bytes],
    *,
    started_ns: int,
    device: JaxDevice,
    gpu_index: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> _MonitoredProcess:
    peak_host_rss_bytes = 0
    peak_gpu_memory_bytes = 0 if device == "gpu" else None
    gpu_memory_available = True
    timed_out = False
    while process.poll() is None:
        host_rss, gpu_memory, sample_available = _sample_process_tree(
            process.pid,
            device=device,
            gpu_index=gpu_index,
        )
        peak_host_rss_bytes = max(peak_host_rss_bytes, host_rss)
        if device == "gpu":
            gpu_memory_available = gpu_memory_available and sample_available
            if gpu_memory is not None and peak_gpu_memory_bytes is not None:
                peak_gpu_memory_bytes = max(peak_gpu_memory_bytes, gpu_memory)
        if (time.monotonic_ns() - started_ns) / 1_000_000_000 > timeout_seconds:
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
    wall_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
    if timed_out and returncode == 0:
        returncode = -signal.SIGTERM
    if device == "gpu" and peak_gpu_memory_bytes == 0:
        gpu_memory_available = False
        peak_gpu_memory_bytes = None
    return _MonitoredProcess(
        returncode=returncode,
        wall_seconds=wall_seconds,
        peak_host_rss_bytes=peak_host_rss_bytes,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        gpu_memory_available=gpu_memory_available,
    )


def _safe_log_stem(run: ScheduledRun, sequence_index: int) -> str:
    pair = "none" if run.pair_index is None else f"{run.pair_index:02d}"
    return (
        f"{sequence_index:02d}-{run.phase}-{run.intent}-"
        f"pair-{pair}-position-{run.order_position}"
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _measure_child(
    *,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    repo_root: Path,
    run_directory: Path,
    workload_id: str,
    workload_index: int,
    run: ScheduledRun,
    sequence_index: int,
    device: JaxDevice,
    gpu_index: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> tuple[_MonitoredProcess, str, str, str, str]:
    logs_directory = run_directory / "logs" / f"{workload_index:02d}-{workload_id}"
    logs_directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_log_stem(run, sequence_index)
    stdout_path = logs_directory / f"{stem}.stdout"
    stderr_path = logs_directory / f"{stem}.stderr"
    with stdout_path.open("xb") as stdout_stream, stderr_path.open(
        "xb"
    ) as stderr_stream:
        started_ns = time.monotonic_ns()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=dict(environment),
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
        monitored = _monitor_process(
            process,
            started_ns=started_ns,
            device=device,
            gpu_index=gpu_index,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
    _fsync_file(stdout_path)
    _fsync_file(stderr_path)
    _fsync_directory(logs_directory)
    return (
        monitored,
        stdout_path.relative_to(run_directory).as_posix(),
        stderr_path.relative_to(run_directory).as_posix(),
        _sha256_file(stdout_path),
        _sha256_file(stderr_path),
    )


def _policy_max_dense_jacobian_bytes(environment: Mapping[str, str]) -> int:
    probe = (
        "from simsopt_jax.config import get_backend_policy; "
        "value=get_backend_policy().max_dense_jacobian_bytes; "
        "print('null' if value is None else int(value))"
    )
    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=_REPO_ROOT,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkRunnerError(
            "backend-policy probe failed: " + completed.stderr.strip()
        )
    final_line = completed.stdout.strip().splitlines()[-1]
    if final_line == "null":
        raise BenchmarkRunnerError("max_dense_jacobian_bytes must be resolved")
    try:
        value = int(final_line)
    except ValueError as error:
        raise BenchmarkRunnerError(
            "backend-policy probe returned invalid bytes"
        ) from error
    if value <= 0:
        raise BenchmarkRunnerError("max_dense_jacobian_bytes must be positive")
    return value


def _child_result_fields(
    *,
    stdout_path: Path,
    example: JaxExampleRecord,
    profile: JaxExecutionProfile,
    monitored: _MonitoredProcess,
) -> tuple[bool, str, str, str]:
    if monitored.returncode != 0:
        return False, "unavailable", "unavailable", "unavailable"
    stdout = stdout_path.read_text(encoding="utf-8")
    try:
        result = example_runner._parse_child_result(stdout, example.id)
        example_runner._validate_child_result(result, example, profile)
    except example_runner.ChildResultValidationError:
        return False, "unavailable", "unavailable", "unavailable"
    return True, result.backend_mode, result.platform, result.precision


def _outcome_document(
    *,
    run: ScheduledRun,
    profile: JaxExecutionProfile,
    environment_sha256: str,
    input_sha256: str,
    source_tree_sha256: str,
    cache_identity: str,
    dense_materialized_bytes: int,
    monitored: _MonitoredProcess,
    scientific_success: bool,
    cache_load_compatible: bool,
    backend_mode: str,
    platform_name: str,
    precision: str,
    stdout_path: str,
    stderr_path: str,
    stdout_sha256: str,
    stderr_sha256: str,
) -> dict[str, object]:
    if profile.device == "cpu":
        gpu_memory: dict[str, object] = {
            "status": "not_applicable",
            "peak_process_bytes": None,
            "owner": "not_applicable",
        }
    elif monitored.gpu_memory_available:
        gpu_memory = {
            "status": "available",
            "peak_process_bytes": monitored.peak_gpu_memory_bytes,
            "owner": "nvidia_smi_process_poll",
        }
    else:
        gpu_memory = {
            "status": "unavailable",
            "peak_process_bytes": None,
            "owner": "nvidia_smi_unavailable",
        }
    synchronized = scientific_success and monitored.returncode == 0
    return {
        "profile": profile.mode,
        "intent": run.intent,
        "phase": run.phase,
        "pair_index": run.pair_index,
        "order_position": run.order_position,
        "measured": run.measured,
        "input_sha256": input_sha256,
        "source_tree_sha256": source_tree_sha256,
        "cache_identity": cache_identity,
        "returncode": monitored.returncode,
        "termination": classify_termination(monitored.returncode),
        "scientific_success": scientific_success,
        "cache_load_compatible": cache_load_compatible,
        "backend_mode": backend_mode,
        "platform": platform_name,
        "precision": precision,
        "timing_synchronized": synchronized,
        "synchronization_owner": METRIC_OWNERS["synchronization"],
        "elapsed_seconds": monitored.wall_seconds if run.measured else None,
        "diagnostic_wall_seconds": monitored.wall_seconds,
        "peak_host_rss_bytes": monitored.peak_host_rss_bytes,
        "gpu_memory": gpu_memory,
        "dense_materialized_bytes": dense_materialized_bytes,
        "environment_sha256": environment_sha256,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def _source_tree_sha256(repo_root: Path) -> str:
    completed = subprocess.run(
        (
            "git",
            "ls-files",
            "-z",
            "--",
            "src/simsopt_jax",
            "examples/jax",
            "benchmarks/jax_example_execution_mode_contract.py",
            "benchmarks/run_jax_example_execution_mode_benchmark.py",
        ),
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    relative_paths = tuple(
        Path(raw.decode("utf-8")) for raw in completed.stdout.split(b"\0") if raw
    )
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths):
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(repo_root / relative_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    raise BenchmarkRunnerError("CPU model is unavailable from /proc/cpuinfo")


def _gpu_identity(gpu_index: int) -> dict[str, object]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name,uuid,driver_version,clocks.current.graphics,power.draw",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkRunnerError(
            "nvidia-smi device query failed: " + completed.stderr.strip()
        )
    rows = tuple(line for line in completed.stdout.splitlines() if line.strip())
    if len(rows) != 1:
        raise BenchmarkRunnerError("nvidia-smi did not identify one physical GPU")
    fields = tuple(field.strip() for field in rows[0].split(","))
    if len(fields) != 5 or not all(fields):
        raise BenchmarkRunnerError("nvidia-smi returned malformed device identity")
    return {
        "kind": "gpu",
        "model": fields[0],
        "uuid": fields[1],
        "driver": fields[2],
        "clock_policy": f"observed_graphics_clock_mhz={fields[3]}",
        "power_policy": f"observed_power_draw_watts={fields[4]}",
    }


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
        raise BenchmarkRunnerError(
            "nvidia-smi utilization query failed: " + completed.stderr.strip()
        )
    rows = tuple(line for line in completed.stdout.splitlines() if line.strip())
    if len(rows) != 1:
        raise BenchmarkRunnerError("nvidia-smi did not return one utilization row")
    fields = tuple(field.strip() for field in rows[0].split(","))
    if len(fields) != 3:
        raise BenchmarkRunnerError("nvidia-smi returned malformed utilization data")
    try:
        gpu_percent, memory_percent, total_memory_mib = (
            int(float(field)) for field in fields
        )
    except ValueError as error:
        raise BenchmarkRunnerError(
            "nvidia-smi returned malformed utilization data"
        ) from error
    return (
        gpu_percent,
        memory_percent,
        total_memory_mib * _NVIDIA_MEMORY_MULTIPLIER,
    )


def _preflight(device: JaxDevice, gpu_index: int) -> dict[str, str]:
    if device == "gpu":
        processes = _nvidia_compute_apps(gpu_index)
        samples: list[tuple[int, int]] = []
        total_memory_values: list[int] = []
        sample_count = int(BENCHMARK_RULE["gpu_concurrent_sample_count"])
        for sample_index in range(sample_count):
            gpu_percent, memory_percent, total_memory_bytes = _gpu_load_sample(
                gpu_index
            )
            samples.append((gpu_percent, memory_percent))
            total_memory_values.append(total_memory_bytes)
            if sample_index + 1 < sample_count:
                time.sleep(_GPU_PREFLIGHT_SAMPLE_INTERVAL_SECONDS)
        if len(set(total_memory_values)) != 1:
            raise BenchmarkRunnerError("GPU total memory changed during preflight")
        return evaluate_gpu_concurrent_load_preflight(
            processes=processes,
            utilization_samples=tuple(samples),
            total_memory_bytes=total_memory_values[0],
        )
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 0:
        raise BenchmarkRunnerError("logical CPU count is unavailable")
    load_one = os.getloadavg()[0]
    normalized_load = load_one / cpu_count
    if normalized_load > 1.0:
        raise BenchmarkRunnerError(
            f"CPU concurrent-load preflight failed: normalized_load={normalized_load:.6g}"
        )
    return {
        "status": "pass",
        "detail": (
            f"one_minute_load={load_one:.6g};logical_cpus={cpu_count};"
            f"normalized_load={normalized_load:.6g}"
        ),
    }


def _provenance(
    *,
    run_id: str,
    started_at_utc: str,
    repo_root: Path,
    manifest_path: Path,
    device: JaxDevice,
    gpu_index: int,
) -> dict[str, object]:
    jax_version = importlib.metadata.version("jax")
    jaxlib_version = importlib.metadata.version("jaxlib")
    device_record = (
        _gpu_identity(gpu_index)
        if device == "gpu"
        else {
            "kind": "cpu",
            "model": _cpu_model(),
            "uuid": None,
            "driver": "not_applicable",
            "clock_policy": "unavailable",
            "power_policy": "unavailable",
        }
    )
    return {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "repo_commit": _git_commit(repo_root),
        "source_tree_sha256": _source_tree_sha256(repo_root),
        "manifest_sha256": _sha256_file(manifest_path),
        "python_version": platform.python_version(),
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "xla_version": f"jaxlib-{jaxlib_version}",
        "host": {
            "hostname": socket.gethostname(),
            "cpu_model": _cpu_model(),
            "os": platform.platform(),
        },
        "device": device_record,
        "concurrent_load_preflight": _preflight(device, gpu_index),
    }


def _input_fingerprint(
    example: JaxExampleRecord,
    *,
    source_sha256: str,
    command: tuple[str, ...],
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "example_id": example.id,
                "path": example.path,
                "smoke_args": list(example.smoke_args),
                "source_sha256": source_sha256,
                "command": list(command[1:]),
            }
        )
    )


def _selected_examples(
    manifest_path: Path, repo_root: Path
) -> tuple[JaxExampleRecord, ...]:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    by_id = {record.id: record for record in manifest.jax_examples}
    selected: list[JaxExampleRecord] = []
    for workload_id in REPRESENTATIVE_WORKLOAD_IDS:
        try:
            record = by_id[workload_id]
        except KeyError as error:
            raise BenchmarkRunnerError(
                f"representative workload is absent from manifest: {workload_id}"
            ) from error
        if record.status != "ready" or set(record.lanes) != {
            "cpu-smoke",
            "gpu-strict",
        }:
            raise BenchmarkRunnerError(
                f"representative workload is not ready on CPU and GPU: {workload_id}"
            )
        selected.append(record)
    return tuple(selected)


def _profile_documents(
    *,
    device: JaxDevice,
    workload_id: str,
    workload_index: int,
    run_directory: Path,
    base_environment: Mapping[str, str],
    repo_root: Path,
    gpu_index: int,
) -> tuple[
    dict[Intent, dict[str, object]],
    dict[Intent, JaxExecutionProfile],
    dict[Intent, dict[str, str]],
    dict[Intent, str],
    dict[Intent, str],
    int,
]:
    documents: dict[Intent, dict[str, object]] = {}
    profiles: dict[Intent, JaxExecutionProfile] = {}
    environments: dict[Intent, dict[str, str]] = {}
    environment_hashes: dict[Intent, str] = {}
    cache_identities: dict[Intent, str] = {}
    policy_limits: list[int] = []
    for intent in ("fast", "parity"):
        cache_directory = (
            run_directory / "caches" / f"{workload_index:02d}-{workload_id}" / intent
        )
        cache_directory.mkdir(parents=True)
        if tuple(cache_directory.iterdir()):
            raise BenchmarkRunnerError(f"cold cache is not empty: {cache_directory}")
        profile, environment, environment_sha256 = build_profile_environment(
            device,
            intent,
            cache_directory=cache_directory,
            base_environment=base_environment,
            repo_root=repo_root,
            gpu_index=gpu_index,
        )
        policy_limits.append(_policy_max_dense_jacobian_bytes(environment))
        if tuple(cache_directory.iterdir()):
            raise BenchmarkRunnerError(
                f"policy probe populated the cold cache: {cache_directory}"
            )
        cache_identity = _sha256_bytes(str(cache_directory.resolve()).encode("utf-8"))
        documents[intent] = {
            "cache": {
                "identity": cache_identity,
                "cold_initial_state": "empty",
                "cold_entry_count_before": 0,
            },
            "environment_sha256": environment_sha256,
            "cold": None,
            "warmup": None,
            "warm": [],
        }
        profiles[intent] = profile
        environments[intent] = environment
        environment_hashes[intent] = environment_sha256
        cache_identities[intent] = cache_identity
    if len(set(policy_limits)) != 1:
        raise BenchmarkRunnerError(
            "fast and parity resolve different max_dense_jacobian_bytes"
        )
    return (
        documents,
        profiles,
        environments,
        environment_hashes,
        cache_identities,
        policy_limits[0],
    )


def _workload_document(
    *,
    example: JaxExampleRecord,
    workload_index: int,
    device: JaxDevice,
    run_directory: Path,
    base_environment: Mapping[str, str],
    repo_root: Path,
    source_tree_sha256: str,
    gpu_index: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    progress: TextIO,
) -> dict[str, object]:
    command = example_runner.build_child_command(example, repo_root=repo_root)
    source_path = repo_root / "examples" / "jax" / example.path
    source_sha256 = _sha256_file(source_path)
    command_sha256 = _sha256_bytes(_canonical_json_bytes(list(command)))
    input_sha256 = _input_fingerprint(
        example, source_sha256=source_sha256, command=command
    )
    (
        profile_documents,
        profiles,
        environments,
        environment_hashes,
        cache_identities,
        max_dense_jacobian_bytes,
    ) = _profile_documents(
        device=device,
        workload_id=example.id,
        workload_index=workload_index,
        run_directory=run_directory,
        base_environment=base_environment,
        repo_root=repo_root,
        gpu_index=gpu_index,
    )
    schedule = build_measurement_schedule(workload_index)
    for sequence_index, scheduled in enumerate(schedule):
        print(
            f"RUN {example.id} {scheduled.phase} {scheduled.intent} "
            f"{sequence_index + 1}/{len(schedule)}",
            file=progress,
            flush=True,
        )
        monitored, stdout_relative, stderr_relative, stdout_sha, stderr_sha = (
            _measure_child(
                command=command,
                environment=environments[scheduled.intent],
                repo_root=repo_root,
                run_directory=run_directory,
                workload_id=example.id,
                workload_index=workload_index,
                run=scheduled,
                sequence_index=sequence_index,
                device=device,
                gpu_index=gpu_index,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
            )
        )
        stdout_path = run_directory / stdout_relative
        stderr_path = run_directory / stderr_relative
        scientific_success, backend_mode, platform_name, precision = (
            _child_result_fields(
                stdout_path=stdout_path,
                example=example,
                profile=profiles[scheduled.intent],
                monitored=monitored,
            )
        )
        outcome = _outcome_document(
            run=scheduled,
            profile=profiles[scheduled.intent],
            environment_sha256=environment_hashes[scheduled.intent],
            input_sha256=input_sha256,
            source_tree_sha256=source_tree_sha256,
            cache_identity=cache_identities[scheduled.intent],
            dense_materialized_bytes=REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES[
                example.id
            ],
            monitored=monitored,
            scientific_success=scientific_success,
            cache_load_compatible=(
                _CPU_AOT_INCOMPATIBILITY_MARKER not in stderr_path.read_bytes()
            ),
            backend_mode=backend_mode,
            platform_name=platform_name,
            precision=precision,
            stdout_path=stdout_relative,
            stderr_path=stderr_relative,
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
        )
        profile_document = profile_documents[scheduled.intent]
        if scheduled.phase == "warm":
            warm = profile_document["warm"]
            if not isinstance(warm, list):
                raise BenchmarkRunnerError("internal warm outcome owner is invalid")
            warm.append(outcome)
        else:
            profile_document[scheduled.phase] = outcome
    warm_schedule = [
        {
            "pair_index": pair_index,
            "order": [
                "fast",
                "parity",
            ]
            if (workload_index + pair_index) % 2 == 0
            else ["parity", "fast"],
        }
        for pair_index in range(WARM_PAIR_COUNT)
    ]
    return {
        "id": example.id,
        "input_sha256": input_sha256,
        "source_sha256": source_sha256,
        "command_sha256": command_sha256,
        "max_dense_jacobian_bytes": max_dense_jacobian_bytes,
        "dense_materialization_owner": METRIC_OWNERS["dense_materialized_bytes"],
        "dense_materialized_bytes": REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES[
            example.id
        ],
        "schedule": warm_schedule,
        "profiles": profile_documents,
    }


def _run_id(device: JaxDevice) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{device}-{uuid.uuid4().hex[:12]}"


def collect_benchmark(
    *,
    device: JaxDevice,
    artifact_root: Path,
    manifest_path: Path,
    repo_root: Path,
    base_environment: Mapping[str, str],
    gpu_index: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    progress: TextIO,
) -> tuple[Path, bool]:
    """Collect and publish all matched repetitions for one physical device."""
    started_at = datetime.now(UTC)
    run_id = _run_id(device)
    provenance = _provenance(
        run_id=run_id,
        started_at_utc=started_at.isoformat().replace("+00:00", "Z"),
        repo_root=repo_root,
        manifest_path=manifest_path,
        device=device,
        gpu_index=gpu_index,
    )
    run_directory = _claim_run_directory(artifact_root, run_id)
    examples = _selected_examples(manifest_path, repo_root)
    source_tree_sha256 = str(provenance["source_tree_sha256"])
    workloads = [
        _workload_document(
            example=example,
            workload_index=workload_index,
            device=device,
            run_directory=run_directory,
            base_environment=base_environment,
            repo_root=repo_root,
            source_tree_sha256=source_tree_sha256,
            gpu_index=gpu_index,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
            progress=progress,
        )
        for workload_index, example in enumerate(examples)
    ]
    artifact: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "rule_version": BENCHMARK_RULE_VERSION,
        "rule": dict(BENCHMARK_RULE),
        "evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "certification_eligible": False,
        "metric_owners": dict(METRIC_OWNERS),
        "device": device,
        "profiles": {
            "fast": f"jax_{device}_fast",
            "parity": f"jax_{device}_parity",
        },
        "workload_ids": list(REPRESENTATIVE_WORKLOAD_IDS),
        "provenance": provenance,
        "workloads": workloads,
    }
    try:
        decision = evaluate_benchmark_artifact(artifact)
    except BenchmarkContractError as error:
        _write_json_exclusive(
            run_directory / "contract_failure.json",
            {"error": str(error), "promoted": False},
        )
        _write_json_exclusive(run_directory / "artifact.json", artifact)
        return run_directory / "artifact.json", False
    _write_json_exclusive(
        run_directory / "decision.json",
        {
            "device": decision.device,
            "promoted": decision.promoted,
            "reasons": list(decision.reasons),
            "workloads": [asdict(summary) for summary in decision.workloads],
        },
    )
    artifact_path = run_directory / "artifact.json"
    _write_json_exclusive(artifact_path, artifact)
    return artifact_path, decision.promoted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(".artifacts/jax-example-execution-modes"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "examples" / "jax" / "manifest.json",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=_DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--child-timeout-seconds",
        type=float,
        default=_DEFAULT_CHILD_TIMEOUT_SECONDS,
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    if options.gpu_index < 0:
        raise SystemExit("--gpu-index must be nonnegative")
    if options.poll_interval_seconds <= 0.0:
        raise SystemExit("--poll-interval-seconds must be positive")
    if options.child_timeout_seconds <= 0.0:
        raise SystemExit("--child-timeout-seconds must be positive")
    device: JaxDevice = options.device
    try:
        artifact_path, promoted = collect_benchmark(
            device=device,
            artifact_root=options.artifact_root.resolve(),
            manifest_path=options.manifest.resolve(),
            repo_root=_REPO_ROOT,
            base_environment=os.environ,
            gpu_index=options.gpu_index,
            poll_interval_seconds=options.poll_interval_seconds,
            timeout_seconds=options.child_timeout_seconds,
            progress=sys.stderr,
        )
    except (
        BenchmarkRunnerError,
        ManifestValidationError,
        FileExistsError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "device": device,
                "promoted": promoted,
            },
            sort_keys=True,
        )
    )
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
