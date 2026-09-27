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

Round-off band: a difference is known only to the round-off of its two
evaluated values over the step, ``b = eps * max|q| over the stencil / (e |v|)``;
a difference within ``b`` of zero is zero at that step.

Converged ranges: a window is at least ``CONVERGED_RUN`` consecutive steps
whose differences are all zero (value 0, known to the smallest band in it),
or all nonzero and agreeing with their neighbours within
``CONVERGED_TOLERANCE`` relative up to their bands (value ``v``, their
median, known to their median band). Two
windows agree when their values do within ``FAIL_TOLERANCE`` relative plus
their bands, or a nonzero value lies within a zero window's band. Two consecutive steps that agree
with the claim within ``PASS_TOLERANCE`` relative to ``max(|d|, |c|, G)``,
each band within that tolerance, are a range converged to the claim. ``G``
is the quantity's largest claimed directional derivative: a direction nearly
orthogonal to the gradient is judged at the gradient's scale.

An FD check cannot know which step range holds the derivative when ranges
converge to different values, so then it does not decide. Per direction:

1. No window: NOT TESTED (no step range converges: noise, a kink, or a
   step range that misses the derivative).
2. Ranges that disagree (windows with different values, or windows that
   miss the claim while a range converges to it): NOT TESTED, listing the
   ranges (the differences depend on the step size: a float32 or quantized
   term, a warm-started inner solve with a loose tolerance, or a kink).
3. One consensus value across the windows:
   - FAIL if the claim misses a nonzero ``v`` by more than
     ``FAIL_TOLERANCE * |v|`` plus its band plus the scatter of the smaller
     steps (their distance from ``v`` beyond their bands, grown as 1 / step
     to the windows' smallest step: evaluation noise could have biased the
     windows that much); or, for a zero consensus, lies outside its band by
     more than the round-off of a derivative at the problem's scale
     (``ABSOLUTE_FLOOR_ULPS`` machine epsilons of the objective's largest
     claimed directional derivative);
   - PASS if a range converged to the claim lies inside a window; for a
     zero consensus, if the claim lies inside its band and that band is
     within ``PASS_TOLERANCE`` of the larger of ``G`` and the objective's
     largest claimed directional derivative (a problem with no derivative
     scale cannot tell zero from round-off);
   - NOT TESTED otherwise.

A nonzero claim never passes through a floor: the round-off floor and the
scatter only withhold a FAIL.

"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from alm_problem import build_problem
from simsopt.solve.alm import ALMPhysics, run_directional_taylor_test

RESULT_PREFIX = "GRADIENT_CHECK "
OBJECTIVE_LABEL = "f"
DEFAULT_DIRECTIONS = inspect.signature(run_directional_taylor_test).parameters["direction_count"].default
# The sweep of relative steps e: dof i moves by e * max(|x0_i|, 1) * g_i, g standard normal.
MAX_STEP = 1.0
MIN_STEP = 1e-10
STEPS_PER_DECADE = 2
# A range converged to the claim: two consecutive steps within this of it, relative to max(|d|, |c|, G).
PASS_TOLERANCE = 1e-6
# A window: this many consecutive steps agreeing with each other to CONVERGED_TOLERANCE ...
CONVERGED_RUN = 3
CONVERGED_TOLERANCE = 1e-4
# ... is known to this, relative to its value (a claim or window farther off disagrees):
# 5 x the convergence tolerance, so a window's own spread cannot fail a right
# claim, and below 1e-3, so a 0.1% wrong gradient fails.
FAIL_TOLERANCE = 5e-4
# A derivative within this many machine epsilons of the objective's gradient
# scale is round-off: a zero consensus fails a claim only beyond it.
ABSOLUTE_FLOOR_ULPS = 1e3
# numpy.random.RandomState seeds.
SEED_LIMIT = 2 ** 32


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
    differences, and their round-off bands."""

    claimed: float
    differences: np.ndarray
    band: np.ndarray


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
    band = np.finfo(float).eps * np.maximum(np.abs(plus), np.abs(minus)) / (steps * norm)
    return Sweep(float(taylor["directional_derivative"]), np.asarray(taylor["central_estimates"], dtype=float),
                 band)


class Window(NamedTuple):
    """Steps ``start`` to ``stop`` (inclusive; larger steps first) whose
    differences converged to ``value``, known to ``band``; ``zero`` when every
    difference in it is zero to its band (then ``band`` is the smallest band
    in it; otherwise the median one)."""

    start: int
    stop: int
    value: float
    band: float
    zero: bool


def runs(member: np.ndarray, linked: np.ndarray) -> List[Tuple[int, int]]:
    """The maximal runs ``(start, stop)`` of at least CONVERGED_RUN consecutive
    member steps, each linked to the next (``linked[k]``: steps k and k + 1)."""
    found, start = [], 0
    while start < len(member):
        stop = start
        while member[stop] and stop + 1 < len(member) and member[stop + 1] and linked[stop]:
            stop += 1
        if member[start] and stop - start + 1 >= CONVERGED_RUN:
            found.append((start, stop))
        start = stop + 1
    return found


def windows(sweep: Sweep) -> List[Window]:
    """The windows of one direction (module docstring), larger steps first."""
    finite = np.isfinite(sweep.differences) & np.isfinite(sweep.band)
    c = np.where(finite, sweep.differences, 0.0)
    band = np.where(finite, sweep.band, 0.0)
    size = np.abs(c)
    zero = finite & (size <= band)
    # Neighbours agree to CONVERGED_TOLERANCE up to their round-off.
    agreeing = (np.abs(np.diff(c)) <= CONVERGED_TOLERANCE * np.maximum(size[:-1], size[1:])
                + band[:-1] + band[1:])
    found = [Window(start, stop, float(np.median(c[start:stop + 1])), float(np.median(band[start:stop + 1])),
                    False)
             for start, stop in runs(finite & ~zero, agreeing)]
    found += [Window(start, stop, 0.0, float(band[start:stop + 1].min()), True)
              for start, stop in runs(zero, np.ones(len(c) - 1, dtype=bool))]
    return sorted(found)


def consistent(first: Window, second: Window) -> bool:
    """Whether two windows converged to the same value: nonzero values within
    FAIL_TOLERANCE of each other, a nonzero value within a zero window's band."""
    if first.zero and second.zero:
        return True
    value = max(abs(first.value), abs(second.value))
    return abs(first.value - second.value) <= FAIL_TOLERANCE * value + first.band + second.band


def describe(window: Window, steps: np.ndarray) -> str:
    span = f"steps {steps[window.start]:.1e} to {steps[window.stop]:.1e}"
    if window.zero:
        return f"zero to {window.band:.1e} over {span}"
    return f"{window.value:.6e} over {span}"


def judge_direction(sweep: Sweep, steps: np.ndarray, gradient_scale: float, objective_scale: float) -> dict:
    """The rules of the module docstring for one direction; the scales are the
    quantity's and the objective's largest claimed directional derivatives."""
    c, d, band = sweep.differences, sweep.claimed, sweep.band
    found = windows(sweep)
    seen = c[np.isfinite(c)]
    seen_range = f"[{seen.min():.3e}, {seen.max():.3e}]" if seen.size else "no finite value"
    if not found:
        return {"verdict": "not_tested", "claimed": d,
                "note": (f"no step range converges (the differences ranged over {seen_range} against a claim "
                         f"of {d:.3e}): noise, a kink, or a step range that misses the derivative; check at a "
                         "nearby point or inspect the quantity")}
    scale = np.maximum(np.maximum(abs(d), np.abs(c)), gradient_scale)
    at_claim = np.isfinite(c) & (np.abs(c - d) <= PASS_TOLERANCE * scale) & (band <= PASS_TOLERANCE * scale)
    pairs = np.flatnonzero(at_claim[:-1] & at_claim[1:])
    nonzero = [window for window in found if not window.zero]
    if nonzero:
        spans = [slice(window.start, window.stop + 1) for window in nonzero]
        value = float(np.median(np.concatenate([c[span] for span in spans])))
        consensus = Window(nonzero[0].start, nonzero[-1].stop, value,
                           float(np.median(np.concatenate([band[span] for span in spans]))), False)
        # The scatter of the smaller steps beyond their bands (noise of the
        # evaluation, which grows as 1 / step) could have biased the windows
        # by up to its size at their smallest step; a miss within it is no FAIL.
        below = slice(consensus.stop + 1, None)
        excess = np.where(np.isfinite(c[below]), np.abs(c[below] - value) - band[below], 0.0)
        scatter = float(np.max(excess * steps[below], initial=0.0)) / steps[consensus.stop]
        misses = abs(d - value) > FAIL_TOLERANCE * abs(value) + consensus.band + scatter
    else:
        consensus = min(found, key=lambda window: window.band)
        misses = abs(d) > consensus.band + ABSOLUTE_FLOOR_ULPS * np.finfo(float).eps * objective_scale
    if not all(consistent(first, second) for first in found for second in found) or (misses and pairs.size):
        ranges = [describe(window, steps) for window in found]
        ranges += [f"the claim over steps {steps[k]:.1e} to {steps[k + 1]:.1e}" for k in pairs[:1]]
        return {"verdict": "not_tested", "claimed": d,
                "note": (f"the finite differences depend on the step size ({'; '.join(ranges)}; claim {d:.6e}): "
                         "a float32 or quantized term, a warm-started inner solve with a loose tolerance, or a "
                         "kink; check the evaluator or test at a nearby point")}
    if misses:
        miss = abs(d - consensus.value)
        relative = (f"absolute {miss:.2e}" if consensus.value == 0.0
                    else f"relative {miss / abs(consensus.value):.1e}")
        return {"verdict": "failed", "claimed": d, "converged": consensus.value,
                "steps": [steps[consensus.start], steps[consensus.stop]],
                "note": f"the differences converge to {describe(consensus, steps)}, but the claim is {d:.6e} "
                        f"({relative})"}
    if consensus.zero and abs(d) <= consensus.band <= PASS_TOLERANCE * max(gradient_scale, objective_scale):
        return {"verdict": "passed", "claimed": d, "steps": [steps[consensus.start], steps[consensus.stop]],
                "note": f"the differences are {describe(consensus, steps)}, and so is the claim ({d:.1e})"}
    inside = [int(k) for k in pairs if any(window.start <= k and k + 1 <= window.stop for window in nonzero)]
    if inside:
        k = inside[0]
        error = float(np.max(np.abs(c[k:k + 2] - d) / scale[k:k + 2]))
        return {"verdict": "passed", "claimed": d, "steps": [steps[k], steps[k + 1]],
                "note": f"agrees to {error:.1e} at steps {steps[k]:.1e} and {steps[k + 1]:.1e}"}
    return {"verdict": "not_tested", "claimed": d,
            "note": (f"the differences converge to {describe(consensus, steps)}, within the tolerance of the "
                     f"claim {d:.6e}, but do not resolve it to {PASS_TOLERANCE:g}; check at a nearby point or "
                     "inspect the quantity")}


def judge(label: str, value: float, steps: np.ndarray, sweeps: List[Sweep], objective_scale: float) -> dict:
    """The quantity's verdict from its directions; ``objective_scale`` is the
    objective's largest claimed directional derivative (see ``main``)."""
    report = {"quantity": label, "value": value,
              "directional_derivatives": [sweep.claimed for sweep in sweeps],
              "finite_differences": [sweep.differences.tolist() for sweep in sweeps]}
    if not (np.isfinite(value) and all(np.isfinite(sweep.claimed) for sweep in sweeps)):
        return {**report, "passed": False, "verdict": "nonfinite",
                "note": "the value or the claimed gradient at x0 is not finite", "directions": []}
    # The quantity's own gradient scale: a direction nearly orthogonal to the
    # gradient is judged against it, not against its own tiny derivative.
    gradient_scale = max(abs(sweep.claimed) for sweep in sweeps)
    directions = [judge_direction(sweep, steps, gradient_scale, objective_scale) for sweep in sweeps]
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


def seed(text: str) -> int:
    value = int(text)
    if not 0 <= value < SEED_LIMIT:
        raise argparse.ArgumentTypeError(f"must be in [0, {SEED_LIMIT}), got {text}")
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
    parser.add_argument("--seed", type=seed, default=1, help="seed of the random directions")
    parser.add_argument("--max-step", type=positive_float, default=MAX_STEP, help="largest relative step")
    parser.add_argument("--min-step", type=positive_float, default=MIN_STEP, help="smallest relative step")
    parser.add_argument("--steps-per-decade", type=positive_int, default=STEPS_PER_DECADE,
                        help="steps per decade of the sweep")
    args = parser.parse_args(argv)
    if args.min_step >= args.max_step or \
            len(step_sweep(args.max_step, args.min_step, args.steps_per_decade)) < CONVERGED_RUN + 1:
        parser.error(f"the sweep from --max-step {args.max_step:g} down to --min-step {args.min_step:g} "
                     f"must hold at least {CONVERGED_RUN + 1} steps")
    steps = step_sweep(args.max_step, args.min_step, args.steps_per_decade)
    problem = build_problem(smoke=args.smoke)
    x0 = np.asarray(problem.x0, dtype=float)

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
    # The problem's derivative scale is the objective's (not the largest row's:
    # one huge row must not hide errors in the others).
    objective_scale = max((abs(sweep.claimed) for sweep in sweeps[0] if np.isfinite(sweep.claimed)), default=0.0)
    outcomes = [judge(label, quantity_value(memo(x0), index), steps, row, objective_scale)
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
