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
    """SciPy L-BFGS-B's options, plus one policy of this route.

    ``restart_after_nonwolfe_stop``: if SciPy stops on its relative-reduction
    (ftol) test right after a line search whose accepted step fails the
    curvature condition ``|g_new.d| <= 0.9 |g_old.d|`` (a dcsrch WARNING
    accepted like convergence), start a new L-BFGS-B call from the accepted
    point with an empty memory and exactly the ``maxiter``/``maxfun`` left.
    Such a stop in a call's first iteration, or with no budget left, is not
    restarted and ends the solve unsuccessful
    (``SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS``); a final iteration that holds
    an internal L-BFGS-B retry is not judged.  Every judged stop is in
    ``OptimizerResult.restart_log``.  Off, the route is SciPy unchanged;
    :meth:`native_matched` leaves it off.
    """

    maxiter: int = 15000
    maxfun: int = 15000
    gtol: float = 1e-10
    ftol: float = 1e-10
    maxcor: int = 200
    maxls: int = 20
    restart_after_nonwolfe_stop: bool = False

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
        No restart policy: the native call is one SciPy call.
        """
        return cls(maxiter=maxiter, maxcor=maxcor, ftol=tol, gtol=tol, maxls=maxls)

    @classmethod
    def evaluation_budgeted(
        cls, *, maxfun: int, maxcor: int, tol: float, maxls: int = 20
    ) -> "ScipyLBFGSBOptions":
        """The fallback policy: a budget of ``maxfun`` true evaluations, with the
        non-Wolfe restart on.

        ``maxiter = maxfun``. x0 costs one evaluation and an iteration normally at
        least one, so the evaluation limit fires at or before the iteration limit;
        they coincide when every iteration used exactly one, and SciPy then reports
        its ITERATIONS message. An iteration whose trials SciPy's memo served entirely
        costs none, so the iteration limit can also fire first. Either way SciPy's
        status is 1. The wrapper tests the budgets only at iteration ends, so the solve
        can pass ``maxfun`` within its final iteration, by at most ``2 * maxls``
        evaluations (``solve.lbfgsb_accounting``). Restarted calls get the literal
        remainders.
        """
        return cls(
            maxiter=maxfun,
            maxfun=maxfun,
            maxcor=maxcor,
            ftol=tol,
            gtol=tol,
            maxls=maxls,
            restart_after_nonwolfe_stop=True,
        )


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
