"""SciPy driver options for ``simsopt_jax.solve``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import OptionsBase


@dataclass(frozen=True)
class ScipyBounds:
    """Closed box ``lower[i] <= x[i] <= upper[i]`` for SciPy's L-BFGS-B.

    An infinite side (``-inf`` below, ``+inf`` above) leaves that side free,
    which is how SciPy reads it; ``+inf`` below or ``-inf`` above is rejected
    because no finite point satisfies it.  The coordinates are Python floats, so the
    numbers SciPy receives are exactly the ones the box was built from.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper):
            raise ValueError(
                f"bounds need one upper per lower, got {len(self.lower)} lower "
                f"and {len(self.upper)} upper"
            )
        # ``<=`` is also false when either side is NaN.
        if not all(low <= high for low, high in zip(self.lower, self.upper)):
            raise ValueError("every lower bound must be <= its upper bound (no NaN)")
        if any(
            low == float("inf") or high == float("-inf")
            for low, high in zip(self.lower, self.upper)
        ):
            raise ValueError(
                "a lower bound of +inf or an upper bound of -inf admits no finite point"
            )

    @classmethod
    def from_arrays(cls, lower: object, upper: object) -> "ScipyBounds":
        """Build the box from two 1-D arrays, e.g. an Optimizable's bounds."""
        return cls(
            lower=tuple(
                float(value) for value in np.asarray(lower, dtype=float).ravel()
            ),
            upper=tuple(
                float(value) for value in np.asarray(upper, dtype=float).ravel()
            ),
        )

    def scipy_bounds(self) -> list[tuple[float, float]]:
        """The ``[(lower, upper), ...]`` list ``scipy.optimize.minimize`` takes."""
        return list(zip(self.lower, self.upper, strict=True))


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


__all__ = ["ScipyBFGSOptions", "ScipyBounds", "ScipyLBFGSBOptions", "ScipyLMOptions"]
