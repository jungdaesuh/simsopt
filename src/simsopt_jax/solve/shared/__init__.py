"""Shared public helper types for ``simsopt_jax.solve``."""

from .state_trace import InvalidStepEvent, OptimizerStateTraceEntry
from .status import InvalidStepReason, LineSearchStatus

__all__ = [
    "InvalidStepEvent",
    "InvalidStepReason",
    "LineSearchStatus",
    "OptimizerStateTraceEntry",
]
