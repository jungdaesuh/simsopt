"""Sequence-neutral JSONL recording for accepted optimization iterations."""

from __future__ import annotations

import json
import math
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import IO, Final, Self

__all__ = (
    "OptimizationMeasurementWindow",
    "OptimizationTrajectoryRecorder",
    "OptimizationWindowTiming",
    "read_optimization_window_timing",
)


_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")
_TIMING_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class OptimizationWindowTiming:
    """Validated elapsed time for one optimizer-only measurement window."""

    wall_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0.0:
            raise ValueError(
                "optimization wall_seconds must be finite and nonnegative, "
                f"got {self.wall_seconds}"
            )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_optimization_window_timing(
    path: Path,
    timing: OptimizationWindowTiming,
) -> None:
    document = {
        "schema_version": _TIMING_SCHEMA_VERSION,
        "wall_seconds": timing.wall_seconds,
    }
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            separators=_JSON_SEPARATORS,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def read_optimization_window_timing(path: Path | str) -> OptimizationWindowTiming:
    """Read an exact v1 optimizer-window timing sidecar or fail closed."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "wall_seconds",
    }:
        raise ValueError("optimization timing sidecar must use the exact v1 schema")
    schema_version = document["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise TypeError("optimization timing schema_version must be an integer")
    if schema_version != _TIMING_SCHEMA_VERSION:
        raise ValueError("optimization timing sidecar has an unsupported schema")
    wall_seconds = document["wall_seconds"]
    if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)):
        raise TypeError("optimization timing wall_seconds must be a JSON number")
    return OptimizationWindowTiming(wall_seconds=float(wall_seconds))


class OptimizationMeasurementWindow:
    """Own optimizer timing and optional accepted-iteration instrumentation.

    Timing starts before the caller's required initial evaluation. On clean
    exit, elapsed time is captured before trajectory close and sidecar I/O, so
    endpoint evaluation and publication remain outside the measured window.
    """

    def __init__(
        self,
        *,
        trajectory_path: Path | None,
        timing_path: Path | None,
    ) -> None:
        self._started = perf_counter()
        self._timing_path = timing_path
        self._trajectory = (
            OptimizationTrajectoryRecorder(trajectory_path)
            if trajectory_path is not None
            else None
        )

    def __enter__(self) -> OptimizationTrajectoryRecorder | None:
        return self._trajectory

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        elapsed_seconds = perf_counter() - self._started
        if self._trajectory is not None:
            self._trajectory.close()
        if exc_type is None and self._timing_path is not None:
            _write_optimization_window_timing(
                self._timing_path,
                OptimizationWindowTiming(wall_seconds=elapsed_seconds),
            )


class OptimizationTrajectoryRecorder:
    """Write validated accepted-iteration records to a new JSONL file.

    The file is opened exclusively when the recorder is constructed. Each
    ``record`` call receives the already-known objective and iteration values;
    it never evaluates an objective or changes optimizer state.
    """

    def __init__(self, path: Path | str) -> None:
        self._started = perf_counter()
        self._stream: IO[str] = Path(path).open(  # noqa: SIM115
            mode="x",
            encoding="utf-8",
            newline="\n",
        )
        self._last_iteration = 0
        self._last_wall_seconds: float | None = None

    @classmethod
    def create(cls, path: Path | str) -> OptimizationTrajectoryRecorder:
        """Create a recorder whose target path must not already exist."""

        return cls(path)

    def record(
        self,
        iteration: int,
        objective: float,
        *,
        wall_seconds_from_start: float | None = None,
    ) -> None:
        """Append one accepted iteration after validating its scalar fields.

        ``iteration`` is the exact positive successor of the previous record.
        The first record is iteration 1. The objective must be finite. When no
        elapsed time is supplied, the value
        is measured from recorder construction with ``perf_counter``.
        """

        iteration_value = operator.index(iteration)
        expected_iteration = self._last_iteration + 1
        if iteration_value != expected_iteration:
            raise ValueError(
                "iteration must be the exact successor "
                f"{expected_iteration}, got {iteration_value}"
            )

        objective_value = float(objective)
        if not math.isfinite(objective_value):
            raise ValueError(f"objective must be finite, got {objective_value}")

        if wall_seconds_from_start is None:
            wall_value = perf_counter() - self._started
        else:
            wall_value = float(wall_seconds_from_start)
        if not math.isfinite(wall_value) or wall_value < 0.0:
            raise ValueError(
                "wall_seconds_from_start must be finite and nonnegative, "
                f"got {wall_value}"
            )
        if self._last_wall_seconds is not None and wall_value < self._last_wall_seconds:
            raise ValueError(
                "wall_seconds_from_start must be nondecreasing, got "
                f"{wall_value} after {self._last_wall_seconds}"
            )

        record = {
            "iteration": iteration_value,
            "objective": objective_value,
            "wall_seconds_from_start": wall_value,
        }
        line = json.dumps(
            record,
            allow_nan=False,
            separators=_JSON_SEPARATORS,
            sort_keys=True,
        )
        self._stream.write(f"{line}\n")
        self._stream.flush()
        self._last_iteration = iteration_value
        self._last_wall_seconds = wall_value

    def close(self) -> None:
        """Close the JSONL stream."""

        self._stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
