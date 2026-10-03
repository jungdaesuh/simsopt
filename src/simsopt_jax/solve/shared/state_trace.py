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
    # True objective evaluations of the whole solve up to this accepted iterate, x0's
    # included; recorded by the SciPy L-BFGS-B restart route only, None on every other
    # route.
    nfev: int | None = None
