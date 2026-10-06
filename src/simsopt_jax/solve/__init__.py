"""Typed lightweight contracts for SIMSOPT JAX solve drivers."""

from .contracts import (
    Driver,
    HessianInverse,
    InverseHessianOperator,
    LbfgsbRestartEvent,
    LbfgsbRestartReason,
    NONFINITE_RESULT_STATUS,
    OptimizerCallbackEvent,
    OptimizerResult,
    OptionsBase,
    SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS,
    STATUS_CODES,
    ScipyBFGSCallbackEvent,
    ScipyLBFGSBCallbackEvent,
    SimsoptBFGSCallbackEvent,
    SimsoptLBFGSBCallbackEvent,
    SimsoptLMQRCallbackEvent,
)
from .scipy import ScipyBFGSOptions, ScipyBounds, ScipyLBFGSBOptions, ScipyLMOptions
from .simsopt import (
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
    SimsoptLMQROptions,
)

__all__ = [
    "Driver",
    "HessianInverse",
    "InverseHessianOperator",
    "LbfgsbRestartEvent",
    "LbfgsbRestartReason",
    "NONFINITE_RESULT_STATUS",
    "OptimizerCallbackEvent",
    "OptimizerResult",
    "OptionsBase",
    "SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS",
    "STATUS_CODES",
    "ScipyBFGSOptions",
    "ScipyBFGSCallbackEvent",
    "ScipyBounds",
    "ScipyLBFGSBOptions",
    "ScipyLBFGSBCallbackEvent",
    "ScipyLMOptions",
    "SimsoptBFGSOptions",
    "SimsoptBFGSCallbackEvent",
    "SimsoptLBFGSBOptions",
    "SimsoptLBFGSBCallbackEvent",
    "SimsoptLMQROptions",
    "SimsoptLMQRCallbackEvent",
]
