"""Bounded subprocess execution for typed native/JAX parity cases."""

from __future__ import annotations

import dataclasses
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.receipts import load_lane_observation, write_lane_observation
from examples.jax.parity.runtime import ParityLane, build_parity_lane_environment


class RunnerError(RuntimeError):
    """One or more isolated lane children failed their execution contract."""


@dataclass(frozen=True)
class ChildProcessResult:
    stdout: str
    stderr: str
    returncode: int
    parent_peak_rss_bytes: int | None


ChildExecutor = Callable[
    [tuple[str, ...], Path, dict[str, str]],
    subprocess.CompletedProcess[str] | ChildProcessResult,
]


def _linux_process_peak_rss_bytes(process_id: int) -> int | None:
    try:
        status = Path(f"/proc/{process_id}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    return None


def execute_child_process(
    command: tuple[str, ...], cwd: Path, environment: dict[str, str]
) -> ChildProcessResult:
    """Execute one child while the parent samples Linux's process RSS high-water."""
    with tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8"
    ) as stdout_file, tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        parent_peak_rss_bytes: int | None = None
        while process.poll() is None:
            sampled = _linux_process_peak_rss_bytes(process.pid)
            if sampled is not None:
                parent_peak_rss_bytes = max(parent_peak_rss_bytes or 0, sampled)
            time.sleep(0.005)
        returncode = process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    return ChildProcessResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        parent_peak_rss_bytes=parent_peak_rss_bytes,
    )


@dataclass(frozen=True)
class ChildExecution:
    lane: ParityLane
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int
    elapsed_seconds: float
    result_directory: Path
    parent_peak_rss_bytes: int | None


def build_child_command(
    *,
    python_executable: str,
    case_id: str,
    lane: ParityLane,
    input_bundle_path: Path,
    result_directory: Path,
    smoke: bool,
) -> tuple[str, ...]:
    """Return the only supported child argv for one case and lane."""
    command = (
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
    )
    return (*command, "--smoke") if smoke else command


def execute_case_lanes(
    *,
    case_id: str,
    lanes: tuple[ParityLane, ...],
    input_bundle_path: Path,
    run_directory: Path,
    repo_root: Path,
    base_environment: Mapping[str, str],
    python_executable: str,
    smoke: bool,
    executor: ChildExecutor = execute_child_process,
) -> tuple[tuple[ChildExecution, ...], dict[str, LaneObservation]]:
    """Execute all requested lanes and validate their published observations."""
    if not lanes or len(lanes) != len(set(lanes)):
        raise RunnerError("lanes must be a non-empty unique tuple")
    executions: list[ChildExecution] = []
    observations: dict[str, LaneObservation] = {}
    failures: list[str] = []
    for lane in lanes:
        result_directory = run_directory / case_id / lane
        result_directory.mkdir(parents=True, exist_ok=True)
        command = build_child_command(
            python_executable=python_executable,
            case_id=case_id,
            lane=lane,
            input_bundle_path=input_bundle_path,
            result_directory=result_directory,
            smoke=smoke,
        )
        environment = build_parity_lane_environment(
            lane, base_environment, repo_root=repo_root
        )
        started = perf_counter()
        completed = executor(command, result_directory, environment)
        parent_peak_rss_bytes = (
            completed.parent_peak_rss_bytes
            if isinstance(completed, ChildProcessResult)
            else None
        )
        execution = ChildExecution(
            lane=lane,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            elapsed_seconds=perf_counter() - started,
            result_directory=result_directory,
            parent_peak_rss_bytes=parent_peak_rss_bytes,
        )
        executions.append(execution)
        if completed.returncode != 0:
            failures.append(
                f"{lane}: child returned {completed.returncode}; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
            continue
        try:
            observation = load_lane_observation(result_directory)
        except (OSError, TypeError, ValueError) as error:
            failures.append(f"{lane}: invalid lane receipt: {error}")
            continue
        if observation.lane != lane:
            failures.append(
                f"{lane}: lane receipt names unexpected lane {observation.lane}"
            )
            continue
        if parent_peak_rss_bytes is not None:
            provenance = observation.provenance
            if provenance is None:
                failures.append(f"{lane}: lane receipt omitted provenance")
                continue
            observation = dataclasses.replace(
                observation,
                provenance=dataclasses.replace(
                    provenance,
                    host_peak_rss_bytes=parent_peak_rss_bytes,
                    host_peak_rss_method="parent-sampled /proc child VmHWM",
                ),
            )
            write_lane_observation(result_directory, observation)
        observations[lane] = observation
    if failures:
        raise RunnerError("; ".join(failures))
    return tuple(executions), observations
