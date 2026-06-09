"""Optimistix driver options for ``simsopt_jax.solve``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..contracts import OptionsBase


class LinearSolver(StrEnum):
    LSMR = "lsmr"
    QR = "qr"


@dataclass(frozen=True)
class OptimistixLBFGSOptions(OptionsBase):
    maxiter: int = 15000
    tol: float = 1e-10
    history_length: int = 200


@dataclass(frozen=True)
class OptimistixLMOptions(OptionsBase):
    maxiter: int = 1500
    tol: float = 1e-10
    linear_solver: LinearSolver = LinearSolver.LSMR
    materialize_dense_linearization: bool = False
    max_dense_linearization_bytes: int | None = None


__all__ = ["LinearSolver", "OptimistixLBFGSOptions", "OptimistixLMOptions"]
