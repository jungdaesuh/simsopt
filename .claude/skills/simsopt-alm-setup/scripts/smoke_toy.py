#!/usr/bin/env python
"""Solve two toy problems with ``simsopt.solve.alm`` and check the answers:
the install check that follows ``check_env.py``.

    python smoke_toy.py

1. The package docstring's problem: min ||x||^2 s.t. 1 - x0 <= 0, whose
   solution is x = (1, 0).
2. min (x0 - 2)^2 + (x1 - 1)^2 s.t. x0 + x1 - 2 <= 0 and -x0 <= 0, whose
   solution is x = (1.5, 0.5) with multipliers (1, 0): one active and one
   inactive row.

Each must end ``converged`` within ``feasibility_tol`` of feasibility, within
1e-4 of the solution and, for problem 2, with multipliers within 1e-3 of
(1, 0). The last line printed is ``SMOKE_TOY {json}`` (which Python and which
simsopt ran, and each problem's outcome); the exit status is 0 when both pass.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import simsopt
import simsopt.solve.alm as alm
from simsopt.solve.alm import ALMPhysics, ALMSettings, cached_alm_evaluator, minimize_alm

RESULT_PREFIX = "SMOKE_TOY "
X_TOLERANCE = 1e-4
MULTIPLIER_TOLERANCE = 1e-3


def docstring_physics(x) -> ALMPhysics:
    return ALMPhysics(
        base_value=float(x @ x),
        base_grad=2.0 * x,
        constraint_values=np.array([1.0 - x[0]]),
        constraint_grads=(np.array([-1.0, 0.0]),),
    )


def two_row_physics(x) -> ALMPhysics:
    return ALMPhysics(
        base_value=(x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        base_grad=np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        constraint_values=np.array([x[0] + x[1] - 2.0, -x[0]]),
        constraint_grads=(np.array([1.0, 1.0]), np.array([-1.0, 0.0])),
    )


def solve_and_judge(name, physics, x0, constraint_names, solution, multipliers) -> dict:
    settings = ALMSettings(max_outer_iterations=20)
    result = minimize_alm(np.asarray(x0, dtype=float), constraint_names,
                          cached_alm_evaluator(physics), settings, {"maxiter": 2000})
    x_error = float(np.max(np.abs(result.x - np.asarray(solution))))
    multiplier_error = (None if multipliers is None
                        else float(np.max(np.abs(result.multipliers - np.asarray(multipliers)))))
    passed = (
        result.termination_reason == "converged"
        and result.max_violation <= settings.feasibility_tol
        and x_error <= X_TOLERANCE
        and (multiplier_error is None or multiplier_error <= MULTIPLIER_TOLERANCE)
    )
    return {
        "problem": name,
        "passed": bool(passed),
        "termination_reason": result.termination_reason,
        "x": result.x.tolist(),
        "x_error": x_error,
        "max_violation": float(result.max_violation),
        "multipliers": result.multipliers.tolist(),
        "multiplier_error": multiplier_error,
        "outer_iterations": int(result.outer_iterations),
    }


def main() -> int:
    outcomes = [
        solve_and_judge("docstring", docstring_physics, [3.0, 2.0], ["x0_at_least_one"],
                        solution=[1.0, 0.0], multipliers=None),
        solve_and_judge("two_rows", two_row_physics, [0.0, 0.0], ["sum_at_most_two", "x0_nonnegative"],
                        solution=[1.5, 0.5], multipliers=[1.0, 0.0]),
    ]
    passed = all(outcome["passed"] for outcome in outcomes)
    for outcome in outcomes:
        print(f"{outcome['problem']}: {'PASS' if outcome['passed'] else 'FAIL'} "
              f"({outcome['termination_reason']}, x={np.round(outcome['x'], 6).tolist()})")
    print(RESULT_PREFIX + json.dumps({
        "passed": passed,
        "python": sys.version.split()[0],
        "simsopt_file": os.path.realpath(simsopt.__file__),
        "alm_file": os.path.realpath(alm.__file__),
        "problems": outcomes,
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
