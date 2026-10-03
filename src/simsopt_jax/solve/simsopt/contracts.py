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
    maxcor: int = 10
    maxls: int = 20


@dataclass(frozen=True)
class SimsoptBFGSOptions(OptionsBase):
    maxiter: int = 1500
    gtol: float = 1e-10
    xrtol: float = 0.0
    line_search_max_steps: int = 20


@dataclass(frozen=True)
class SimsoptLMQROptions(OptionsBase):
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float | None = None
    max_dense_linearization_bytes: int | None = None


__all__ = [
    "SimsoptBFGSOptions",
    "SimsoptLBFGSBOptions",
    "SimsoptLMQROptions",
]
