"""Optimistix driver options for ``simsopt.solve.jax``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..contracts import OptionsBase


class LinearSolver(StrEnum):
    LSMR = "lsmr"
    QR = "qr"


@dataclass(frozen=True)
class OptimistixLMOptions(OptionsBase):
    maxiter: int = 1500
    tol: float = 1e-10
    linear_solver: LinearSolver = LinearSolver.LSMR
    materialize_dense_linearization: bool = False
    max_dense_linearization_bytes: int | None = None


__all__ = ["LinearSolver", "OptimistixLMOptions"]
