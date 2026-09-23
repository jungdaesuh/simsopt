"""Official ``least_squares_serial_solve`` stopping policy and provider outcome.

``simsopt.solve.serial.least_squares_serial_solve`` throws SciPy's
``OptimizeResult`` away: it publishes ``prob.x`` and returns ``None``. A native
parity lane must report what the official provider reported, so
:func:`solve_official_least_squares` runs that wrapper unchanged and records the
result of the single ``scipy.optimize.least_squares`` call it makes, restoring
the module binding afterwards. Rebinding a module global is only safe while one
thread is solving, which is what the parity child and the mirror tests are.

The official call passes no solver keywords, so SciPy's defaults (``trf``,
``ftol = xtol = gtol = 1e-8``, ``max_nfev = 100 * n``) are the official stopping
rule at every scale. :func:`declared_work_budget` and
:func:`official_stopping_keywords` keep that one rule per scale for every lane:
the bounded scale adds its declared evaluation budget and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve
from simsopt.solve import serial as official_serial
from simsopt_jax.examples import ExecutionScale


def declared_work_budget(scale: ExecutionScale, max_steps: int) -> int | None:
    """Return the bounded scale's declared budget, or SciPy's default (``None``)."""
    return max_steps if scale == "bounded" else None


def official_stopping_keywords(budget: int | None) -> dict[str, int]:
    """Return the only scale-dependent keyword the official wrapper receives."""
    return {} if budget is None else {"max_nfev": budget}


def solve_official_least_squares(
    problem: LeastSquaresProblem,
    **keywords: float | int | str,
) -> OptimizeResult:
    """Solve through the official wrapper and return SciPy's own result."""
    recorded: list[OptimizeResult] = []
    official_least_squares = official_serial.least_squares

    def recording_least_squares(
        objective: Callable[[np.ndarray], np.ndarray],
        initial: np.ndarray,
        **inner_keywords: object,
    ) -> OptimizeResult:
        result = official_least_squares(objective, initial, **inner_keywords)
        recorded.append(result)
        return result

    official_serial.least_squares = recording_least_squares
    try:
        least_squares_serial_solve(problem, **keywords)
    finally:
        official_serial.least_squares = official_least_squares
    (result,) = recorded
    return result
