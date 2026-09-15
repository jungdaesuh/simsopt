"""SciPy driver options for ``simsopt_jax.solve``."""

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

    @classmethod
    def native_matched(
        cls, *, maxiter: int, maxcor: int, tol: float, maxls: int = 20
    ) -> "ScipyLBFGSBOptions":
        """Equal to ``minimize(..., options={maxiter, maxcor[, maxls]}, tol=tol)``.

        ``scipy.optimize.minimize`` expands ``tol`` into both ``ftol`` and
        ``gtol`` for L-BFGS-B and leaves ``maxfun`` at SciPy's default, which
        is this class's default. ``maxls`` is named when the native call names
        it; omitted, it stays at SciPy's default of 20, so a caller matching a
        native script that names nothing else still names nothing else here.
        """
        return cls(maxiter=maxiter, maxcor=maxcor, ftol=tol, gtol=tol, maxls=maxls)


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
