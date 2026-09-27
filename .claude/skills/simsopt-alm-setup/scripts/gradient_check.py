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
library default; at least two steps and one direction are required (else
exit 2). Each random direction is judged on its own, from the finite
differences ``c_k`` (the measured change per step) against the claimed
derivative ``d``, never from ``d`` alone:

- The round-off floor at step ``e_k`` is ``ROUNDOFF_FACTOR * eps *
  (|f(x0)| + |c_k| e_k) / e_k``, with ``eps`` the float64 machine epsilon: the
  noise of a central difference of values accurate to ``ROUNDOFF_FACTOR``
  machine epsilons of their size. It grows as the step shrinks.
- A step is informative when its floor is at most
  ``INFORMATIVE_NOISE_FRACTION`` of ``|c_k|``. Only informative steps judge,
  and the library's ratio test (whose floor is absolute, so it cannot see a
  wrong derivative below it) only labels the outcome. The direction passes
  when the error ``|c_k - d|`` at its smallest informative step is at most
  max(``RELATIVE_TOLERANCE`` |c_k|, floor), or, failing that, when the error
  falls by the library's ratio threshold between every two consecutive
  informative steps (at least two): truncation shrinking as it should, as
  for a derivative far smaller than the higher ones. It fails otherwise.
- A direction with no informative step is ``no_change`` (nothing tested),
  unless ``|d|`` exceeds every floor: a claimed derivative where the
  quantity does not change fails.

A quantity's ``verdict``: ``nonfinite`` if any value or difference is not
finite; ``not_tested`` if every direction is ``no_change`` (round-off swamps
the change at these steps: use larger steps, or the quantity does not depend
on x); ``failed`` if any direction fails; else ``accuracy`` if some
direction's ratio test failed (round-off), ``no_ratio`` if some direction had
no ratio (the error at the ratio test's floor at every step, as for a linear
or quadratic quantity), and ``ratio_test`` otherwise. Only ``ratio_test``,
``no_ratio`` and ``accuracy`` pass. The last line printed is
``GRADIENT_CHECK {json}``; the exit status is 0 when every quantity passes, 1
otherwise (``not_tested`` included), and 2 for invalid arguments.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Optional

import numpy as np

from alm_problem import build_problem
from simsopt.solve.alm import ALMPhysics, run_directional_taylor_test

RESULT_PREFIX = "GRADIENT_CHECK "
OBJECTIVE_LABEL = "f"
# Values are trusted to this many machine epsilons of their magnitude.
ROUNDOFF_FACTOR = 100.0
# Accepted error, relative to the measured change, at an informative step.
RELATIVE_TOLERANCE = 1e-6
# A step informs when its round-off floor is at most this fraction of the change.
INFORMATIVE_NOISE_FRACTION = 1e-2
MACHINE_EPSILON = float(np.finfo(float).eps)
DEFAULT_DIRECTIONS = inspect.signature(run_directional_taylor_test).parameters["direction_count"].default
PASSING_VERDICTS = ("ratio_test", "no_ratio", "accuracy")


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


def judge_direction(taylor: dict) -> dict:
    """The verdict of one direction's ``run_directional_taylor_test`` result
    (the rules are in the module docstring)."""
    result = taylor["direction_results"][0]
    steps = np.asarray(taylor["epsilons"], dtype=float)
    changes = np.abs(np.asarray(result["central_estimates"], dtype=float))
    errors = np.asarray(result["errors"], dtype=float)
    claimed = float(result["directional_derivative"])
    value = float(taylor["base_total"])
    if not (np.isfinite(value) and np.isfinite(claimed) and np.all(np.isfinite(changes))):
        return {"verdict": "nonfinite", "relative_error": None}
    floors = ROUNDOFF_FACTOR * MACHINE_EPSILON * (abs(value) + changes * steps) / steps
    informative = floors <= INFORMATIVE_NOISE_FRACTION * changes
    if not np.any(informative):
        verdict = "failed" if abs(claimed) > float(floors.max()) else "no_change"
        return {"verdict": verdict, "relative_error": None}
    informative_errors = errors[informative]
    last = int(np.flatnonzero(informative)[-1])
    relative_error = float(errors[last] / changes[last])
    converged = errors[last] <= max(RELATIVE_TOLERANCE * changes[last], floors[last])
    falling = informative_errors.size >= 2 and bool(np.all(
        informative_errors[1:] <= taylor["ratio_threshold"] * informative_errors[:-1]))
    if converged and taylor["max_ratio"] is None:
        verdict = "no_ratio"
    elif converged and not taylor["passed"]:
        verdict = "accuracy"
    elif converged or falling:
        verdict = "ratio_test"
    else:
        verdict = "failed"
    return {"verdict": verdict, "relative_error": relative_error}


def judge(label: str, taylors: list) -> dict:
    """The quantity's verdict from its per-direction results (see the module docstring)."""
    directions = [judge_direction(taylor) for taylor in taylors]
    verdicts = [direction["verdict"] for direction in directions]
    tested = [direction for direction in directions if direction["verdict"] != "no_change"]
    worst_error = max((direction["relative_error"] for direction in tested
                       if direction["relative_error"] is not None), default=None)
    untested = len(directions) - len(tested)
    if "nonfinite" in verdicts:
        verdict, note = "nonfinite", "a value or difference is not finite"
    elif not tested:
        verdict, note = "not_tested", ("round-off swamps the change at every step along every "
                                       "direction: use larger steps, or the quantity does not "
                                       "depend on x")
    elif "failed" in verdicts:
        verdict, note = "failed", (
            "the gradient does not match the finite differences" if worst_error is None else
            f"relative error {worst_error:.1e} at the smallest informative step, and the error does "
            "not fall with the step: the gradient does not match the finite differences")
    elif "accuracy" in verdicts:
        verdict, note = "accuracy", (f"ratio test stopped at round-off along {verdicts.count('accuracy')} "
                                     f"of {len(verdicts)} directions; relative error {worst_error:.1e}, "
                                     "within tolerance")
    elif "no_ratio" in verdicts:
        verdict, note = "no_ratio", (f"no ratio along {verdicts.count('no_ratio')} of {len(verdicts)} "
                                     "directions: error at the ratio test's floor at every step (an exact "
                                     "difference, as for a linear or quadratic quantity, or round-off); "
                                     f"relative error {worst_error:.1e}")
    else:
        verdict, note = "ratio_test", ""
    if untested and tested:
        note = (note + "; " if note else "") + (f"{untested} of {len(directions)} directions had no "
                                                "informative step")
    ratios = [taylor["max_ratio"] for taylor in taylors if taylor["max_ratio"] is not None]
    return {
        "quantity": label,
        "passed": verdict in PASSING_VERDICTS,
        "verdict": verdict,
        "note": note,
        "value": taylors[0]["base_total"],
        "max_ratio": max(ratios) if ratios else None,
        "relative_error": worst_error,
        "direction_verdicts": verdicts,
        "directional_derivatives": [taylor["directional_derivative"] for taylor in taylors],
        "finite_differences": [taylor["central_estimates"] for taylor in taylors],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="build the problem at its smoke size")
    parser.add_argument("--directions", type=int, default=DEFAULT_DIRECTIONS,
                        help="random directions per quantity")
    parser.add_argument("--epsilons", help="comma-separated steps, largest first")
    parser.add_argument("--seed", type=int, default=1, help="seed of the random directions")
    args = parser.parse_args(argv)

    if args.directions < 1:
        parser.error(f"--directions must be at least 1, got {args.directions}")
    if args.epsilons is not None and len(args.epsilons.split(",")) < 2:
        parser.error("--epsilons needs at least two steps: one step gives no ratio to test")
    problem = build_problem(smoke=args.smoke)
    epsilons = (tuple(float(value) for value in args.epsilons.split(","))
                if args.epsilons else problem.taylor_epsilons)
    if epsilons is not None and len(epsilons) < 2:
        parser.error("the problem's taylor_epsilons need at least two steps: one step gives no ratio")
    # The library's directions, drawn once and tested one at a time, so each
    # is judged on its own; every quantity visits the same points.
    random = np.random.RandomState(args.seed)
    directions = [random.standard_normal(size=np.shape(problem.x0)) for _ in range(args.directions)]
    memo = PhysicsMemo(problem.physics)
    labels = (OBJECTIVE_LABEL,) + tuple(problem.constraint_names)
    indices = (None,) + tuple(range(len(problem.constraint_names)))
    outcomes = [
        judge(label, [run_directional_taylor_test(quantity_evaluator(memo, index), problem.x0, np.zeros(0),
                                                  1.0, direction=direction, epsilons=epsilons)
                      for direction in directions])
        for label, index in zip(labels, indices)
    ]
    passed = all(outcome["passed"] for outcome in outcomes)
    status_names = {"nonfinite": "NONFINITE", "not_tested": "NOT TESTED", "failed": "FAIL"}
    for outcome in outcomes:
        status = "PASS" if outcome["passed"] else status_names[outcome["verdict"]]
        flag = f"  ({outcome['note']})" if outcome["note"] else ""
        max_ratio = "n/a" if outcome["max_ratio"] is None else f"{outcome['max_ratio']:.3f}"
        print(f"{outcome['quantity']:<32} {status:<10} value={outcome['value']:+.4e} "
              f"max_ratio={max_ratio}{flag}")
    print(RESULT_PREFIX + json.dumps({
        "passed": passed,
        "problem": problem.name,
        "epsilons": None if epsilons is None else list(epsilons),
        "directions": args.directions,
        "physics_evaluations": memo.evaluations,
        "quantities": outcomes,
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
