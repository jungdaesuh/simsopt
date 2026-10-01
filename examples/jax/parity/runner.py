"""Bounded subprocess execution for typed native/JAX parity cases."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.receipts import load_lane_observation
from examples.jax.parity.runtime import ParityLane, build_parity_lane_environment
from simsopt_jax.examples import ExecutionScale
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel


class RunnerError(RuntimeError):
    """One or more isolated lane children failed their execution contract."""


ChildExecutor = Callable[
    [tuple[str, ...], Path, dict[str, str]], subprocess.CompletedProcess[str]
]


def execute_child_process(
    command: tuple[str, ...], cwd: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Execute one child and capture its output."""
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        check=False,
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


def build_child_command(
    *,
    python_executable: str,
    case_id: str,
    lane: ParityLane,
    input_bundle_path: Path,
    result_directory: Path,
    scale: ExecutionScale,
) -> tuple[str, ...]:
    """Return the only supported child argv for one case and lane."""
    command = (
        python_executable,
        "-S",
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
    return command


def execute_case_lanes(
    *,
    case_id: str,
    lanes: tuple[ParityLane, ...],
    input_bundle_path: Path,
    run_directory: Path,
    repo_root: Path,
    base_environment: Mapping[str, str],
    python_executable: str,
    scale: ExecutionScale,
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
            scale=scale,
        )
        environment = build_parity_lane_environment(
            lane, base_environment, repo_root=repo_root
        )
        environment["PYTHONPATH"] = pythonpath_with_loaded_kernel(
            *environment["PYTHONPATH"].split(os.pathsep)
        )
        started = perf_counter()
        completed = executor(command, result_directory, environment)
        execution = ChildExecution(
            lane=lane,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            elapsed_seconds=perf_counter() - started,
            result_directory=result_directory,
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
        if observation.scale != scale:
            failures.append(
                f"{lane}: lane receipt scale {observation.scale} does not match {scale}"
            )
            continue
        observations[lane] = observation
    if failures:
        raise RunnerError("; ".join(failures))
    return tuple(executions), observations
