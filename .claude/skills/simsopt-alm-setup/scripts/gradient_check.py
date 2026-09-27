#!/usr/bin/env python
"""Taylor-test the objective and every constraint row of the problem in
``alm_problem.py`` separately.

    PYTHONPATH=<problem dir> python gradient_check.py [--smoke] [--directions N] [--epsilons E1,E2,...] [--seed S]

For f and for each row g_i, ``run_directional_taylor_test`` compares the
gradient ``problem.physics(x)`` returns with central differences of the value
along random unit directions at ``problem.x0``: the error must fall at least as
fast as the step. Testing each row on its own matters, because in the
augmented Lagrangian an inactive row (``max(0, multiplier + penalty * g) = 0``)
drops out, and its gradient would go unchecked. Every quantity's test visits
the same points, so each point's physics is evaluated once and shared.

The steps are ``--epsilons``, else the problem's ``taylor_epsilons``, else the
library default. Every value and difference must be finite (else
``nonfinite``); then a quantity passes in one of three ways, reported as its
``verdict`` and printed with the reason:

- ``ratio_test``: the error falls at least as fast as the step.
- ``no_ratio``: no ratio exists because the error is at the test's round-off
  floor at every step: the central difference is exact (a linear or
  quadratic quantity, such as a linear row) or the errors sit at round-off
  (or only one step was given). The gradient agrees with the differences.
- ``accuracy``: the ratio test fails but, along every direction, the smallest
  error is at most ``ACCURACY_TOLERANCE`` times max(1, |derivative|): at
  steps where round-off dominates the errors stop falling although the
  gradient is right to that accuracy.

A quantity whose derivative is zero along every direction passes as
``vacuous`` (nothing was tested); any other outcome is ``failed``. The last
line printed is ``GRADIENT_CHECK {json}``; the exit status is 0 when every
quantity passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from alm_problem import build_problem
from simsopt.solve.alm import ALMPhysics, run_directional_taylor_test

RESULT_PREFIX = "GRADIENT_CHECK "
OBJECTIVE_LABEL = "f"
ACCURACY_TOLERANCE = 1e-6


class PhysicsMemo:
    """``problem.physics(x)``, evaluated once per distinct x."""

    def __init__(self, physics):
        self._physics = physics
        self._by_x = {}

    def __call__(self, x) -> ALMPhysics:
        key = np.asarray(x, dtype=float).tobytes()
        if key not in self._by_x:
            self._by_x[key] = self._physics(x)
        return self._by_x[key]

    @property
    def evaluations(self) -> int:
        """How many distinct points were evaluated."""
        return len(self._by_x)


def quantity_evaluator(memo: PhysicsMemo, row_index: Optional[int]):
    """An ``evaluate_problem`` whose ``total`` is f (``row_index`` None) or
    row ``row_index``, for ``run_directional_taylor_test``."""

    def evaluate(x, multipliers, penalty) -> dict:
        physics = memo(x)
        if row_index is None:
            return {"total": physics.base_value, "grad": physics.base_grad}
        return {"total": float(physics.constraint_values[row_index]),
                "grad": physics.constraint_grads[row_index]}

    return evaluate


def judge(label: str, taylor: dict) -> dict:
    directions = taylor["direction_results"]
    finite = all(
        np.isfinite(result["directional_derivative"])
        and np.all(np.isfinite(result["central_estimates"]))
        for result in directions
    ) and np.isfinite(taylor["base_total"])
    scale = max(1.0, abs(taylor["base_total"]))
    vacuous = finite and all(
        abs(result["directional_derivative"]) <= 1e-14 * scale for result in directions
    )
    accurate = finite and all(
        min(result["errors"]) <= ACCURACY_TOLERANCE * max(1.0, abs(result["directional_derivative"]))
        for result in directions
    )
    if not finite:
        verdict, note = "nonfinite", "a value or difference is not finite"
    elif vacuous:
        verdict, note = "vacuous", "zero derivative along every direction: nothing tested"
    elif taylor["max_ratio"] is None:
        verdict, note = "no_ratio", (
            "no ratio: only one step" if len(taylor["epsilons"]) < 2 else
            "no ratio: error at the round-off floor at every step (exact for a linear "
            "or quadratic quantity, else round-off)")
    elif taylor["passed"]:
        verdict, note = "ratio_test", ""
    elif accurate:
        verdict, note = "accuracy", "ratio test at round-off; smallest error within tolerance"
    else:
        verdict, note = "failed", "error does not fall with the step: the gradient is wrong"
    return {
        "quantity": label,
        "passed": verdict in ("ratio_test", "no_ratio", "accuracy", "vacuous"),
        "verdict": verdict,
        "note": note,
        "finite": bool(finite),
        "value": taylor["base_total"],
        "max_ratio": taylor["max_ratio"],
        "ratio_threshold": taylor["ratio_threshold"],
        "directional_derivatives": [result["directional_derivative"] for result in directions],
        "errors_first_direction": taylor["errors"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="build the problem at its smoke size")
    parser.add_argument("--directions", type=int, help="random directions per quantity")
    parser.add_argument("--epsilons", help="comma-separated steps, largest first")
    parser.add_argument("--seed", type=int, default=1, help="seed of the random directions")
    args = parser.parse_args(argv)

    problem = build_problem(smoke=args.smoke)
    epsilons = (tuple(float(value) for value in args.epsilons.split(","))
                if args.epsilons else problem.taylor_epsilons)
    options = {"epsilons": epsilons, "seed": args.seed}
    if args.directions is not None:
        options["direction_count"] = args.directions
    memo = PhysicsMemo(problem.physics)
    labels = (OBJECTIVE_LABEL,) + tuple(problem.constraint_names)
    indices = (None,) + tuple(range(len(problem.constraint_names)))
    taylors = [
        run_directional_taylor_test(quantity_evaluator(memo, index), problem.x0, np.zeros(0), 1.0, **options)
        for index in indices
    ]
    outcomes = [judge(label, taylor) for label, taylor in zip(labels, taylors)]
    passed = all(outcome["passed"] for outcome in outcomes)
    for outcome in outcomes:
        status = "PASS" if outcome["passed"] else ("NONFINITE" if not outcome["finite"] else "FAIL")
        flag = f"  ({outcome['note']})" if outcome["note"] else ""
        max_ratio = "n/a" if outcome["max_ratio"] is None else f"{outcome['max_ratio']:.3f}"
        print(f"{outcome['quantity']:<32} {status:<9} value={outcome['value']:+.4e} "
              f"max_ratio={max_ratio}{flag}")
    print(RESULT_PREFIX + json.dumps({
        "passed": passed,
        "problem": problem.name,
        "epsilons": taylors[0]["epsilons"],
        "physics_evaluations": memo.evaluations,
        "quantities": outcomes,
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
