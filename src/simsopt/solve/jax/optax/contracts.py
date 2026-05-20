"""Optax driver options for ``simsopt.solve.jax``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..contracts import OptionsBase


class OptaxLineSearch(StrEnum):
    ZOOM = "zoom"


@dataclass(frozen=True)
class OptaxLBFGSOptions(OptionsBase):
    maxiter: int = 15000
    gtol: float = 1e-10
    memory_size: int = 200
    line_search: OptaxLineSearch = OptaxLineSearch.ZOOM
    scale_init_precond: bool = True
    max_linesearch_steps: int = 30


@dataclass(frozen=True)
class OptaxAdamOptions(OptionsBase):
    maxiter: int = 10000
    learning_rate: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    gtol: float | None = None
    weight_decay: float = 0.0


__all__ = ["OptaxAdamOptions", "OptaxLBFGSOptions", "OptaxLineSearch"]
