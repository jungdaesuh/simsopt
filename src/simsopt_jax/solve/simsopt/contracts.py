"""In-tree simsopt driver options for ``simsopt_jax.solve``."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import (
    OptionsBase,
    _validate_nonnegative_integers,
    _validate_positive_integers,
    _validate_tolerances,
)


@dataclass(frozen=True)
class SimsoptLBFGSBOptions(OptionsBase):
    maxiter: int = 15000
    maxfun: int = 15000
    gtol: float = 1e-10
    ftol: float = 1e-10
    maxcor: int = 10
    maxls: int = 20

    def validate(self) -> None:
        _validate_nonnegative_integers(maxiter=self.maxiter, maxfun=self.maxfun)
        _validate_positive_integers(maxcor=self.maxcor, maxls=self.maxls)
        _validate_tolerances(gtol=self.gtol, ftol=self.ftol)


@dataclass(frozen=True)
class SimsoptBFGSOptions(OptionsBase):
    maxiter: int = 1500
    gtol: float = 1e-10
    xrtol: float = 0.0
    line_search_max_steps: int = 20

    def validate(self) -> None:
        _validate_nonnegative_integers(maxiter=self.maxiter)
        _validate_positive_integers(line_search_max_steps=self.line_search_max_steps)
        _validate_tolerances(gtol=self.gtol, xrtol=self.xrtol)


@dataclass(frozen=True)
class SimsoptLMQROptions(OptionsBase):
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float | None = None
    max_dense_linearization_bytes: int | None = None

    def validate(self) -> None:
        # This driver's maxiter is MINPACK's evaluation limit, not an iteration count.
        _validate_positive_integers(maxiter=self.maxiter)
        _validate_tolerances(ftol=self.ftol, xtol=self.xtol)
        if self.gtol is not None:
            _validate_tolerances(gtol=self.gtol)
        if self.max_dense_linearization_bytes is not None:
            _validate_positive_integers(
                max_dense_linearization_bytes=self.max_dense_linearization_bytes
            )


__all__ = [
    "SimsoptBFGSOptions",
    "SimsoptLBFGSBOptions",
    "SimsoptLMQROptions",
]
