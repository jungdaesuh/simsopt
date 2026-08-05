"""Sequence-neutral JSONL recording for accepted optimization iterations."""

from __future__ import annotations

import json
import math
import operator
from pathlib import Path
from time import perf_counter
from typing import IO, Final, Self

__all__ = ("OptimizationTrajectoryRecorder",)


_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")


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
