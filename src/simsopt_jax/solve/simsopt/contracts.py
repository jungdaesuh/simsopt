"""In-tree simsopt driver options for ``simsopt_jax.solve``."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import OptionsBase


@dataclass(frozen=True)
class SimsoptLBFGSBOptions(OptionsBase):
    maxiter: int = 15000
    maxfun: int = 15000
    gtol: float = 1e-10
    ftol: float = 1e-10
    maxcor: int = 200
    maxls: int = 20


@dataclass(frozen=True)
class SimsoptBFGSOptions(OptionsBase):
    maxiter: int = 1500
    gtol: float = 1e-10
    xrtol: float = 0.0


@dataclass(frozen=True)
class SimsoptAdamHostOptions(OptionsBase):
    maxiter: int = 10000
    learning_rate: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    gtol: float | None = None


@dataclass(frozen=True)
class SimsoptAdamOptions(SimsoptAdamHostOptions):
    pass


@dataclass(frozen=True)
class SimsoptTraceLBFGSOptions(OptionsBase):
    maxiter: int = 15000
    gtol: float = 1e-10
    maxcor: int = 200
    ftol: float = 1e-10
    maxls: int = 20
    initial_step_size: float | None = None
    record_optimizer_state_trace: bool = True
    max_optimizer_state_trace_bytes: int | None = None
    invalid_step_log_capacity: int = 256


@dataclass(frozen=True)
class SimsoptLMGMRESHostOptions(OptionsBase):
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float | None = None


@dataclass(frozen=True)
class SimsoptLMGMRESOptions(SimsoptLMGMRESHostOptions):
    materialize_dense_linearization: bool = True
    max_dense_linearization_bytes: int | None = None


@dataclass(frozen=True)
class SimsoptLMQROptions(OptionsBase):
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float | None = None
    max_dense_linearization_bytes: int | None = None


__all__ = [
    "SimsoptAdamHostOptions",
    "SimsoptAdamOptions",
    "SimsoptBFGSOptions",
    "SimsoptLBFGSBOptions",
    "SimsoptLMGMRESHostOptions",
    "SimsoptLMGMRESOptions",
    "SimsoptLMQROptions",
    "SimsoptTraceLBFGSOptions",
]
