"""Explicit options for measurement-only parity-case execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SingleStageOuterOptimizerBackend = Literal["optax-lbfgs"]


@dataclass(frozen=True, slots=True)
class MeasurementExecution:
    """Per-process trajectory instrumentation and optimizer selection.

    The request requires a trajectory path or optimizer backend. A case that
    supports measurements owns the meaning of ``optimizer_backend``.
    """

    trajectory_path: Path | None = None
    optimizer_backend: SingleStageOuterOptimizerBackend | None = None

    def __post_init__(self) -> None:
        if self.trajectory_path is None and self.optimizer_backend is None:
            raise ValueError(
                "measurement execution requires a trajectory path or optimizer backend"
            )


__all__ = ("MeasurementExecution", "SingleStageOuterOptimizerBackend")
