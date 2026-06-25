"""Public trace payloads for ``simsopt_jax.solve`` optimizer results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InvalidStepEvent:
    iteration: int
    step_scale: float
    reason: str


@dataclass(frozen=True, slots=True)
class OptimizerStateTraceEntry:
    iteration: int
    fun: float
    grad_norm_inf: float
