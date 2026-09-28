"""Explicit options for measurement-only parity-case execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MeasurementExecution:
    """Per-process trajectory instrumentation.

    The request requires at least one instrumentation output path.
    """

    trajectory_path: Path | None = None
    optimization_timing_path: Path | None = None

    def __post_init__(self) -> None:
        if self.trajectory_path is None and self.optimization_timing_path is None:
            raise ValueError("measurement execution requires an instrumentation path")


__all__ = ("MeasurementExecution",)
