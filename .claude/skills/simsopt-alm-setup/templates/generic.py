#!/usr/bin/env python
r"""Generic ALM problem: minimize f(x) subject to g_i(x) <= 0 for any f and g
whose values and gradients you can compute.

Copy this file to ``alm_problem.py`` next to ``run_alm.py`` and edit the parts
marked ``SETUP``. As shipped it solves

    minimize    (x0 - 2)^2 + (x1 - 1)^2
    subject to  x0 + x1 <= 2       (row "sum_at_most_max")
                x0      >= 0.25    (row "x0_at_least_min")

whose solution is x = (1.5, 0.5) with the first row active.

Each row is a measured quantity q(x) with a bound: an upper bound gives
``g = (q - bound) / scale``, a lower bound ``g = (bound - q) / scale``, so
``g <= 0`` is feasible and ``scale`` (the size of the bound, in q's units)
makes every row O(1). The physics depends on x alone, so the solver gets
``cached_alm_evaluator(physics)``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

from simsopt.solve.alm import ALMPhysics, ALMResult, ALMSettings, cached_alm_evaluator


class SignProbe(NamedTuple):
    """A point where the listed rows are known to be violated (g > 0) or
    satisfied (g <= 0), independently of the row code."""

    label: str
    x: np.ndarray
    violated: Tuple[str, ...]
    satisfied: Tuple[str, ...]


# The sign that turns ``quantity - bound`` into a row value ``g``:
UPPER_BOUND = 1.0   # quantity <= bound:  g = (quantity - bound) / scale
LOWER_BOUND = -1.0  # quantity >= bound:  g = (bound - quantity) / scale


class Row(NamedTuple):
    """One constraint row: ``name`` bounds ``quantity`` at ``bound`` from the
    side ``sense`` (UPPER_BOUND or LOWER_BOUND); ``scale`` > 0 is in the
    quantity's units."""

    name: str
    quantity: str
    sense: float
    bound: float
    scale: float


# SETUP: the rows. Units are those of the quantity.
ROWS = (
    Row(name="sum_at_most_max", quantity="coordinate_sum", sense=UPPER_BOUND, bound=2.0, scale=2.0),
    Row(name="x0_at_least_min", quantity="x0", sense=LOWER_BOUND, bound=0.25, scale=1.0),
)


def objective(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """SETUP: f(x) and its gradient."""
    value = (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2
    grad = np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)])
    return float(value), grad


def quantities(x: np.ndarray) -> dict:
    """SETUP: each constrained quantity at x as ``name -> (value, gradient)``."""
    return {
        "coordinate_sum": (float(x[0] + x[1]), np.array([1.0, 1.0])),
        "x0": (float(x[0]), np.array([1.0, 0.0])),
    }


def signed_row(row: Row, value: float, grad: np.ndarray) -> Tuple[float, np.ndarray]:
    """The row's scaled signed value and gradient (``g <= 0`` is feasible)."""
    return row.sense * (value - row.bound) / row.scale, row.sense * grad / row.scale


class GenericProblem:
    """The problem-module contract of the simsopt-alm-setup skill (see its
    ``references/api.md``) for a physics that depends on x alone."""

    name = "generic"
    constraint_names = tuple(row.name for row in ROWS)
    # None keeps run_directional_taylor_test's default steps.
    taylor_epsilons: Optional[Tuple[float, ...]] = None

    def __init__(self, smoke: bool):
        # SETUP: the starting point.
        self.x0 = np.array([3.0, 2.0])
        self.evaluator = cached_alm_evaluator(self.physics)
        # f and every row are O(1), so the default tolerances (1e-6) act as
        # relative ones. Every other field keeps its default.
        self.settings = ALMSettings(
            # At a fixed penalty each outer iteration shrinks the violation by
            # a roughly constant factor; this problem needs about 10.
            max_outer_iterations=20,
        )
        # maxiter is the L-BFGS-B iteration budget of the whole minimize_alm
        # call (all subproblems), not a per-subproblem limit.
        self.inner_options = {"maxiter": 200 if smoke else 2000}

    def physics(self, x) -> ALMPhysics:
        x = np.asarray(x, dtype=float)
        f, grad_f = objective(x)
        measured = quantities(x)
        rows = [signed_row(row, *measured[row.quantity]) for row in ROWS]
        return ALMPhysics(
            base_value=f,
            base_grad=grad_f,
            constraint_values=np.array([value for value, _grad in rows]),
            constraint_grads=tuple(grad for _value, grad in rows),
        )

    def solver_callbacks(self) -> dict:
        """Extra ``minimize_alm`` arguments; a stateless physics needs none."""
        return {}

    def sign_probes(self) -> Tuple[SignProbe, ...]:
        # SETUP: points where you know, from the physics, which rows hold.
        return (
            SignProbe(
                label="sum 5 above its max 2, x0 = 3 above its min",
                x=np.array([3.0, 2.0]),
                violated=("sum_at_most_max",),
                satisfied=("x0_at_least_min",),
            ),
            SignProbe(
                label="origin: sum 0 below its max, x0 = 0 below its min 0.25",
                x=np.array([0.0, 0.0]),
                violated=("x0_at_least_min",),
                satisfied=("sum_at_most_max",),
            ),
        )

    def finish(self, result: ALMResult) -> dict:
        """Use the returned x (possibly a restored best-feasible iterate)."""
        return {"x": [float(value) for value in result.x]}


def build_problem(smoke: bool = False) -> GenericProblem:
    return GenericProblem(smoke)
