"""SciPy driver options for ``simsopt.solve.jax``."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import OptionsBase


@dataclass(frozen=True)
class ScipyLBFGSBOptions(OptionsBase):
    maxiter: int = 15000
    maxfun: int = 15000
    gtol: float = 1e-10
    ftol: float = 1e-10
    maxcor: int = 200
    maxls: int = 20


@dataclass(frozen=True)
class ScipyLMOptions(OptionsBase):
    max_nfev: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8


@dataclass(frozen=True)
class ScipyBFGSOptions(OptionsBase):
    maxiter: int = 1500
    gtol: float = 1e-10
    xrtol: float = 0.0
    norm: float = float("inf")


__all__ = ["ScipyBFGSOptions", "ScipyLBFGSBOptions", "ScipyLMOptions"]
