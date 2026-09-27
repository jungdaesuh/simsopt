#!/usr/bin/env python
"""Check the gradient of the objective and of every constraint row of the
problem in ``alm_problem.py`` against finite differences over a wide sweep of
step sizes.

    PYTHONPATH=<problem dir> python gradient_check.py [--smoke] [--directions N] [--seed S]
                                                     [--max-step E] [--min-step E] [--steps-per-decade N]

For f and for each row q separately (an inactive row drops out of the
augmented Lagrangian, so its gradient must be checked on its own), along each
of ``--directions`` random directions, ``run_directional_taylor_test``
evaluates the central differences

    c(e) = (q(x0 + e v) - q(x0 - e v)) / (2 e |v|),    v = s * g,  s_i = max(|x0_i|, 1)

with ``g`` standard normal, so each dof moves relative to its own size (a
current of 1e5 A and a coil coefficient of 1 m alike), at every relative step
``e`` of a geometric sweep from ``--max-step`` down to ``--min-step``
(``--steps-per-decade`` per decade), together with the claimed directional
derivative ``d = grad q . v / |v|``. Every quantity visits the same points,
so each point's physics is evaluated once. A step whose difference is not
finite (e.g. a failed inner solve) is skipped.

Resolution: a difference can only be trusted to its resolution ``r``, one
unit in the last place of its two values over the step: a change below it is
invisible to the evaluation. A step judges to a tolerance only when ``r`` is
at most that tolerance times the direction's derivative scale
``S = max(|d|, median |c|)``; with ``S = 0`` (a direction that never changes)
every step judges. Three rules decide each direction:

- PASS: two consecutive steps that resolve ``PASS_TOLERANCE`` both agree
  with the claim: ``|c - d| <= PASS_TOLERANCE * max(|d|, |c|, G)``, with ``G``
  the largest claimed directional derivative of the quantity (a direction
  nearly orthogonal to the gradient is judged at the gradient's scale), or
  both ``|c|`` and ``|d|`` are within the round-off floor of that step. The
  floor is the spread (max - min) of the differences at the ``FLOOR_STEPS``
  smallest judging steps, where round-off dominates, carried to each step as
  1 / step, and at least ``ABSOLUTE_FLOOR_ULPS`` machine epsilons times the
  objective's largest claimed directional derivative (so an exactly zero
  row passes a round-off-size claim; the objective, not the largest row,
  sets it, so one huge row cannot hide errors in the others). Two
  consecutive steps rule out a coincidental crossing.
- FAIL: among the steps that resolve ``FAIL_TOLERANCE``, runs of at least
  ``CONVERGED_RUN`` consecutive steps whose neighbours agree to
  ``CONVERGED_TOLERANCE`` (up to their resolution) have converged; the run
  nearest the smallest steps judges (the derivative is the small-step limit;
  a plateau at larger steps is the slope of a nearby kink or saturation). A
  run of exact zeros judges only along a direction that never changes;
  elsewhere identical values are the precision of the evaluation (e.g. a
  large intermediate that cancels). With ``v`` the run's median, the
  direction fails when ``|v - d|`` exceeds ``FAIL_TOLERANCE * max(|v|, |d|)``
  (or, when ``|v|`` is within the run's floor, that floor) plus the run's
  resolution, and no smaller judging step comes back within
  ``FAIL_TOLERANCE`` of the claim.
- NOT TESTED otherwise: nothing converged (a kink, noise, oscillation), the
  steps cannot resolve the derivative, or the result is in between.

PASS takes precedence over FAIL. A quantity
passes when every direction passes, fails when any direction fails, is
``not_tested`` otherwise, and ``nonfinite`` when its value or claimed
gradient at x0 is not finite. The note says what was seen; a NOT TESTED one
asks to check at a nearby point or to inspect the quantity. The last line
printed is ``GRADIENT_CHECK {json}`` (finite numbers only; ``null`` where
undefined); the exit status is 0 when every quantity passes, 1 otherwise,
and 2 for invalid arguments.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from typing import List, NamedTuple, Optional

import numpy as np

from alm_problem import build_problem
from simsopt.solve.alm import ALMPhysics, run_directional_taylor_test

RESULT_PREFIX = "GRADIENT_CHECK "
OBJECTIVE_LABEL = "f"
DEFAULT_DIRECTIONS = inspect.signature(run_directional_taylor_test).parameters["direction_count"].default
# The sweep of relative steps e (the step is e * max(max|x0|, 1)).
MAX_STEP = 1.0
MIN_STEP = 1e-10
STEPS_PER_DECADE = 2
# PASS: agreement of two consecutive steps with the claim, relative to max(|d|, |c|).
PASS_TOLERANCE = 1e-6
# FAIL: this many consecutive steps agreeing with each other to CONVERGED_TOLERANCE ...
CONVERGED_RUN = 3
CONVERGED_TOLERANCE = 1e-4
# ... at a value this far from the claim, relative to max(|v|, |d|): 5 x the
# convergence tolerance, so a converged value's own spread cannot fail a
# right claim, and below 1e-3, so a 0.1% wrong gradient fails.
FAIL_TOLERANCE = 5e-4
# The round-off floor is the spread of the differences at this many smallest steps.
FLOOR_STEPS = 3
# A derivative within this many machine epsilons of the objective's gradient
# scale is round-off (the absolute floor of every quantity).
ABSOLUTE_FLOOR_ULPS = 1e3


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


def quantity_value(physics: ALMPhysics, row_index: Optional[int]) -> float:
    return float(physics.base_value if row_index is None else physics.constraint_values[row_index])


def quantity_evaluator(memo: PhysicsMemo, row_index: Optional[int]):
    """An ``evaluate_problem`` whose ``total`` is f (``row_index`` None) or
    row ``row_index``, for ``run_directional_taylor_test``."""

    def evaluate(x, multipliers, penalty) -> dict:
        physics = memo(x)
        grad = physics.base_grad if row_index is None else physics.constraint_grads[row_index]
        return {"total": quantity_value(physics, row_index), "grad": grad}

    return evaluate


class Sweep(NamedTuple):
    """One direction, steps largest first: the claimed derivative, the
    differences, and their resolution (one unit in the last place of the two
    values, over the step: a difference this small is a value that did not
    change)."""

    claimed: float
    differences: np.ndarray
    resolution: np.ndarray


def sweep_direction(memo: PhysicsMemo, row_index: Optional[int], x0: np.ndarray,
                    direction: np.ndarray, steps: np.ndarray) -> Sweep:
    """The differences along ``direction`` (not normalized) at x0 +- e * direction
    for each relative step e; the library normalizes the direction, so its
    steps are e * |direction|, and its points are read back from the memo."""
    norm = float(np.linalg.norm(direction))
    taylor = run_directional_taylor_test(quantity_evaluator(memo, row_index), x0, np.zeros(0), 1.0,
                                         direction=direction, epsilons=tuple(steps * norm))
    unit = np.asarray(taylor["direction"], dtype=float)
    plus = np.array([quantity_value(memo(x0 + float(step) * unit), row_index) for step in steps * norm])
    minus = np.array([quantity_value(memo(x0 - float(step) * unit), row_index) for step in steps * norm])
    resolution = (np.spacing(np.abs(plus)) + np.spacing(np.abs(minus))) / (2.0 * steps * norm)
    return Sweep(float(taylor["directional_derivative"]),
                 np.asarray(taylor["central_estimates"], dtype=float), resolution)


def round_off_floor(differences: np.ndarray, usable: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """The round-off of each difference: the spread (max - min) of the
    differences at the FLOOR_STEPS smallest usable steps, where round-off
    dominates, carried to every step as 1 / step (round-off of a difference
    grows as the step shrinks); 0 where the differences are exact."""
    chosen = np.flatnonzero(usable)[-FLOOR_STEPS:]
    if chosen.size == 0:
        return np.zeros_like(steps)
    spread = float(differences[chosen].max() - differences[chosen].min())
    return spread * steps[chosen[-1]] / steps


def within(a: float, b: float, tolerance: float, scale: float, floor: float, allowance: float = 0.0) -> bool:
    """``|a - b| <= tolerance * max(|a|, |b|, scale) + allowance``, or ``a`` and
    ``b`` both within ``floor`` (round-off)."""
    return (abs(a - b) <= tolerance * max(abs(a), abs(b), scale) + allowance
            or (abs(a) <= floor and abs(b) <= floor))


def converged_runs(c: np.ndarray, usable: np.ndarray, floor: np.ndarray, resolution: np.ndarray) -> List[tuple]:
    """The maximal runs ``(start, stop)`` of at least CONVERGED_RUN consecutive
    usable steps whose neighbours agree to CONVERGED_TOLERANCE, up to their
    resolution (or are both within the round-off floor), largest steps first."""
    runs, start = [], 0
    while start < len(c):
        stop = start
        while stop + 1 < len(c) and usable[stop] and usable[stop + 1] and \
                within(c[stop], c[stop + 1], CONVERGED_TOLERANCE, 0.0, max(floor[stop], floor[stop + 1]),
                       resolution[stop] + resolution[stop + 1]):
            stop += 1
        if stop - start + 1 >= CONVERGED_RUN:
            runs.append((start, stop))
        start = stop + 1
    return runs


def judge_direction(sweep: Sweep, steps: np.ndarray, gradient_scale: float, absolute_floor: float) -> dict:
    """The three rules of the module docstring for one direction."""
    c, d, resolution = sweep.differences, sweep.claimed, sweep.resolution
    finite = np.isfinite(c)
    # A step can judge the derivative to a tolerance only when its resolution
    # (one unit in the last place of its values, over the step) is below the
    # tolerance times the direction's derivative scale: otherwise a zero or a
    # few-ulp difference is the precision of the value, not the derivative.
    # A direction that never changes (scale 0) is fully resolved: it is flat.
    scale = max(abs(d), float(np.median(np.abs(c[finite]))) if finite.any() else 0.0)
    passing = finite & ((scale == 0.0) | (resolution <= PASS_TOLERANCE * scale))
    judging = finite & ((scale == 0.0) | (resolution <= FAIL_TOLERANCE * scale))
    floor = np.maximum(round_off_floor(c, judging, steps), absolute_floor)
    for k in range(len(c) - 1):
        if passing[k] and passing[k + 1] and within(c[k], d, PASS_TOLERANCE, gradient_scale, floor[k]) \
                and within(c[k + 1], d, PASS_TOLERANCE, gradient_scale, floor[k + 1]):
            error = max(abs(c[k] - d), abs(c[k + 1] - d)) / max(abs(d), abs(c[k]), abs(c[k + 1]),
                                                                  gradient_scale, 1e-300)
            return {"verdict": "passed", "claimed": d, "steps": [steps[k], steps[k + 1]],
                    "note": f"agrees to {error:.1e} at steps {steps[k]:.1e} and {steps[k + 1]:.1e}"}
    runs = converged_runs(c, judging, floor, resolution)
    # Exactly zero differences (identical values at x0 + e v and x0 - e v) are
    # a derivative only along a direction that never changes; where other steps
    # do change, they are the precision of the evaluation (e.g. a large
    # intermediate that cancels), so they cannot fail a claim.
    if np.any(c[finite] != 0.0):
        runs = [(start, stop) for start, stop in runs if np.any(c[start:stop + 1] != 0.0)]
    if runs:
        # The derivative is the small-step limit: the run nearest it judges,
        # and not when a smaller judging step comes back to the claim (then the
        # run is a plateau at large steps: the slope of a nearby kink or of a
        # saturation, not the derivative at x0).
        start, stop = runs[-1]
        value = float(np.median(c[start:stop + 1]))
        run_floor = float(floor[start:stop + 1].max())
        allowance = float(resolution[start:stop + 1].max())
        near_zero = abs(value) <= run_floor
        miss = abs(value - d)
        margin = run_floor if near_zero else FAIL_TOLERANCE * max(abs(value), abs(d))
        returns = any(judging[k] and within(c[k], d, FAIL_TOLERANCE, 0.0, 0.0, resolution[k])
                      for k in range(stop + 1, len(c)))
        if miss > margin + allowance and not returns:
            relative = f"relative {miss / max(abs(value), abs(d)):.1e}" if not near_zero else \
                f"absolute {miss:.2e}, round-off {run_floor:.1e}"
            return {"verdict": "failed", "claimed": d, "converged": value,
                    "steps": [steps[start], steps[stop]],
                    "note": (f"the differences converge to {value:.6e} over steps {steps[start]:.1e} to "
                             f"{steps[stop]:.1e}, but the claim is {d:.6e} ({relative})")}
    seen = c[judging]
    seen_range = (f"[{seen.min():.3e}, {seen.max():.3e}]" if seen.size else "no resolved difference")
    return {"verdict": "not_tested", "claimed": d,
            "note": (f"the differences ranged over {seen_range} against a claim of {d:.3e} without settling "
                     "on it; check at a nearby point or inspect the quantity (a kink, noise or oscillation)")}


def judge(label: str, value: float, steps: np.ndarray, sweeps: List[Sweep], absolute_floor: float) -> dict:
    """The quantity's verdict from its directions; ``absolute_floor`` is the
    problem-wide round-off floor of a derivative (see ``main``)."""
    report = {"quantity": label, "value": value,
              "directional_derivatives": [sweep.claimed for sweep in sweeps],
              "finite_differences": [sweep.differences.tolist() for sweep in sweeps]}
    if not (np.isfinite(value) and all(np.isfinite(sweep.claimed) for sweep in sweeps)):
        return {**report, "passed": False, "verdict": "nonfinite",
                "note": "the value or the claimed gradient at x0 is not finite", "directions": []}
    # The quantity's own gradient scale: a direction nearly orthogonal to the
    # gradient is judged against it, not against its own tiny derivative.
    gradient_scale = max(abs(sweep.claimed) for sweep in sweeps)
    directions = [judge_direction(sweep, steps, gradient_scale, absolute_floor) for sweep in sweeps]
    verdicts = [direction["verdict"] for direction in directions]
    count = len(verdicts)
    if "failed" in verdicts:
        verdict = "failed"
        example = directions[verdicts.index("failed")]
        note = f"{verdicts.count('failed')} of {count} directions fail; e.g. {example['note']}"
    elif verdicts.count("passed") == count:
        verdict = "passed"
        note = f"all {count} directions; e.g. {directions[0]['note']}"
    else:
        verdict = "not_tested"
        example = directions[verdicts.index("not_tested")]
        note = (f"{verdicts.count('passed')} of {count} directions pass, {verdicts.count('not_tested')} "
                f"not tested; e.g. {example['note']}")
    return {**report, "passed": verdict == "passed", "verdict": verdict, "note": note, "directions": directions}


def finite_json(value):
    """``value`` with every non-finite float replaced by None (JSON has no NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0.0):
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {text}")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def step_sweep(max_step: float, min_step: float, per_decade: int) -> np.ndarray:
    """``e`` from ``max_step`` down to ``min_step``, ``per_decade`` per decade."""
    count = int(round(math.log10(max_step / min_step) * per_decade)) + 1
    return np.geomspace(max_step, min_step, count)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="build the problem at its smoke size")
    parser.add_argument("--directions", type=positive_int, default=DEFAULT_DIRECTIONS,
                        help="random directions per quantity")
    parser.add_argument("--seed", type=int, default=1, help="seed of the random directions")
    parser.add_argument("--max-step", type=positive_float, default=MAX_STEP, help="largest relative step")
    parser.add_argument("--min-step", type=positive_float, default=MIN_STEP, help="smallest relative step")
    parser.add_argument("--steps-per-decade", type=positive_int, default=STEPS_PER_DECADE,
                        help="steps per decade of the sweep")
    args = parser.parse_args(argv)
    if args.min_step >= args.max_step or \
            len(step_sweep(args.max_step, args.min_step, args.steps_per_decade)) < CONVERGED_RUN + 1:
        parser.error(f"the sweep from --max-step {args.max_step:g} down to --min-step {args.min_step:g} "
                     f"must hold at least {CONVERGED_RUN + 1} steps")
    relative = step_sweep(args.max_step, args.min_step, args.steps_per_decade)
    problem = build_problem(smoke=args.smoke)
    x0 = np.asarray(problem.x0, dtype=float)
    steps = relative

    # Directions drawn once, each dof scaled to its own size; every quantity
    # visits the same points.
    random = np.random.RandomState(args.seed)
    scale = np.maximum(np.abs(x0), 1.0)
    directions = [scale * random.standard_normal(size=x0.shape) for _ in range(args.directions)]
    memo = PhysicsMemo(problem.physics)
    labels = (OBJECTIVE_LABEL,) + tuple(problem.constraint_names)
    indices = (None,) + tuple(range(len(problem.constraint_names)))
    sweeps = [[sweep_direction(memo, index, x0, direction, steps) for direction in directions]
              for index in indices]
    # The objective's gradient scale sets the absolute floor of a derivative: a
    # claim within ABSOLUTE_FLOOR_ULPS machine epsilons of it is round-off, so an
    # exactly zero row passes a round-off-size claim and fails a real one. (The
    # objective, not the largest row, anchors it: one huge row must not hide
    # errors in the others.)
    objective_claims = [abs(sweep.claimed) for sweep in sweeps[0] if np.isfinite(sweep.claimed)]
    absolute_floor = ABSOLUTE_FLOOR_ULPS * np.finfo(float).eps * max(objective_claims, default=0.0)
    outcomes = [judge(label, quantity_value(memo(x0), index), steps, row, absolute_floor)
                for label, index, row in zip(labels, indices, sweeps)]
    passed = all(outcome["passed"] for outcome in outcomes)
    status_names = {"passed": "PASS", "failed": "FAIL", "not_tested": "NOT TESTED", "nonfinite": "NONFINITE"}
    for outcome in outcomes:
        print(f"{outcome['quantity']:<32} {status_names[outcome['verdict']]:<10} "
              f"value={outcome['value']:+.4e}  ({outcome['note']})")
    print(RESULT_PREFIX + json.dumps(finite_json({
        "passed": passed,
        "problem": problem.name,
        "steps": steps.tolist(),
        "directions": args.directions,
        "physics_evaluations": memo.evaluations,
        "quantities": outcomes,
    }), allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
