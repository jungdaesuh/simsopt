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

Resolution: a difference is known only to its resolution ``r``: the larger
of one unit in the last place of its two values over the step, and the noise
of the evaluation (round-off inside it, an inner solve's tolerance), which
is the spread of the differences at the ``NOISE_STEPS`` smallest steps,
where it dominates, carried to every step as 1 / step. A change below ``r``
is invisible, so a noisy step can neither pass nor fail a claim. The absolute floor ``B`` of a derivative is
``ABSOLUTE_FLOOR_ULPS`` times the largest of: machine epsilon times the
objective's largest claimed directional derivative (one huge row cannot
raise it for the others), machine epsilon times the quantity's own value
over a unit relative step (a constant of 1e4 is flat to its own round-off),
and the smallest normal number.

Windows: a window is at least ``CONVERGED_RUN`` consecutive steps. In a
nonzero window every difference is above ``B``, resolved to
``FAIL_TOLERANCE`` of itself, and agrees with its neighbours to
``CONVERGED_TOLERANCE`` (up to their resolution); a claim within
``FAIL_TOLERANCE`` of its median ``v``, relative to ``v`` (plus the
resolution), agrees with it. In a near-zero window every difference and its
resolution are within ``B`` of zero; a claim within ``B`` of its median
agrees. No scale is taken from the differences at large, so noise cannot
make a step look resolved.

The derivative is the small-step limit, so the window nearest the smallest
steps judges: a nonzero window at larger steps that disagrees with it is a
plateau (the slope of a nearby kink or saturation), and so is one followed
by two consecutive smaller steps within ``FAIL_TOLERANCE`` of the claim.
Identical values (a window of exact zeros) along a direction that changes
elsewhere are either a flat stretch (a clipped row near its bound) or a
change hidden below the precision of an intermediate (a cancellation), and
the two look alike: such a window judges only below every other window, and
only in the claim's favour. Rules, FAIL first:

- FAIL: the judging window disagrees with the claim, and neither identical
  values below it nor two smaller steps come back to the claim.
- PASS: no FAIL, and either the judging window is a near-zero window that
  agrees (or there is none, and identical values agree); or two
  consecutive steps agree with the claim,
  ``|c - d| <= PASS_TOLERANCE * max(|d|, |c|, G)``, each resolved to that
  tolerance (``G`` is the quantity's largest claimed directional
  derivative, so a direction nearly orthogonal to the gradient is judged at
  the gradient's scale), and the judging window, if any, agrees. A zero
  claim passes only through a zero window.
- NOT TESTED otherwise: nothing converged (a kink, noise, oscillation), the
  steps cannot resolve the derivative, or the windows contradict each other.

A quantity passes when every direction passes, fails when any direction
fails, is ``not_tested`` otherwise, and ``nonfinite`` when its value or
claimed gradient at x0 is not finite. The note says what was seen; a NOT
TESTED one asks to check at a nearby point or to inspect the quantity. The
last line printed is ``GRADIENT_CHECK {json}`` (finite numbers only;
``null`` where undefined); the exit status is 0 when every quantity passes,
1 otherwise, and 2 for invalid arguments.
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
# PASS: agreement of two consecutive steps with the claim, relative to max(|d|, |c|, G).
PASS_TOLERANCE = 1e-6
# A converged window: this many consecutive steps agreeing with each other to CONVERGED_TOLERANCE ...
CONVERGED_RUN = 3
CONVERGED_TOLERANCE = 1e-4
# ... disagrees with a claim this far from its value, relative to the value:
# 5 x the convergence tolerance, so a window's own spread cannot fail a right
# claim, and below 1e-3, so a 0.1% wrong gradient fails.
FAIL_TOLERANCE = 5e-4
# A derivative within this many machine epsilons of the objective's gradient
# scale, or of the quantity's value over a unit relative step, is round-off.
ABSOLUTE_FLOOR_ULPS = 1e3
# The noise of an evaluation is the spread of the differences at this many
# smallest finite steps, where it dominates.
NOISE_STEPS = 3
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
    differences, their resolution (a change of the difference this small is
    invisible), and the length of the direction (a unit relative step)."""

    claimed: float
    differences: np.ndarray
    resolution: np.ndarray
    norm: float


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
    differences = np.asarray(taylor["central_estimates"], dtype=float)
    # One unit in the last place of the two values, over the step ...
    last_place = (np.spacing(np.abs(plus)) + np.spacing(np.abs(minus))) / (2.0 * steps * norm)
    # ... or the noise of the evaluation (round-off inside it, an inner solve's
    # tolerance): the spread of the differences at the smallest steps, where
    # it dominates, carried to every step as 1 / step.
    smallest = np.flatnonzero(np.isfinite(differences))[-NOISE_STEPS:]
    noise = (float(np.ptp(differences[smallest])) * steps[smallest[-1]] / steps if smallest.size
             else np.zeros_like(steps))
    return Sweep(float(taylor["directional_derivative"]), differences, np.maximum(last_place, noise), norm)


class Window(NamedTuple):
    """Steps ``start`` to ``stop`` (inclusive; larger steps first) whose
    differences converged to ``value``; a claim within ``band`` of it agrees.
    ``near_zero``: the differences are zero to the absolute floor;
    ``identical``: they are exactly 0 (identical values at x0 +- e v)."""

    start: int
    stop: int
    value: float
    band: float
    near_zero: bool
    identical: bool

    def agrees(self, claimed: float) -> bool:
        return abs(claimed - self.value) <= self.band


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


def windows(sweep: Sweep, floor: float) -> List[Window]:
    """The converged windows of one direction (module docstring), larger
    steps first."""
    finite = np.isfinite(sweep.differences) & np.isfinite(sweep.resolution)
    c = np.where(finite, sweep.differences, 0.0)
    resolution = np.where(finite, sweep.resolution, 0.0)
    size = np.abs(c)
    resolved = finite & (size > floor) & (resolution <= FAIL_TOLERANCE * size)
    agreeing = np.abs(np.diff(c)) <= (CONVERGED_TOLERANCE * np.maximum(size[:-1], size[1:])
                                      + resolution[:-1] + resolution[1:])
    near_zero = finite & (size + resolution <= floor)
    found = []
    for start, stop in runs(resolved, agreeing):
        value = float(np.median(c[start:stop + 1]))
        band = FAIL_TOLERANCE * abs(value) + float(resolution[start:stop + 1].max())
        found.append(Window(start, stop, value, band, False, False))
    for start, stop in runs(near_zero, np.ones(len(c) - 1, dtype=bool)):
        found.append(Window(start, stop, float(np.median(c[start:stop + 1])), floor, True,
                            not np.any(c[start:stop + 1])))
    return sorted(found)


def describe(window: Window, steps: np.ndarray) -> str:
    span = f"over steps {steps[window.start]:.1e} to {steps[window.stop]:.1e}"
    if window.near_zero:
        return f"the differences are zero to {window.band:.1e} {span}"
    return f"the differences converge to {window.value:.6e} {span}"


def judge_direction(sweep: Sweep, steps: np.ndarray, gradient_scale: float, floor: float) -> dict:
    """The rules of the module docstring for one direction, FAIL first."""
    c, d, resolution = sweep.differences, sweep.claimed, sweep.resolution
    found = windows(sweep, floor)
    changes = bool(np.any(c[np.isfinite(c)] != 0.0))
    # Identical values along a direction that changes elsewhere may hide a
    # change below the precision of an intermediate: they judge only after
    # (below) the last other window, and only in the claim's favour.
    judging = [window for window in found if not (changes and window.identical)]
    last = judging[-1] if judging else None
    identical = [window for window in found if changes and window.identical
                 and (last is None or window.start > last.stop)]
    supported = bool(identical) and identical[-1].agrees(d)
    # Two consecutive steps below the last window that come back to the claim
    # show that window is a plateau (the slope of a nearby kink or saturation).
    near = np.maximum(abs(d), np.abs(np.where(np.isfinite(c), c, 0.0)))
    back = np.isfinite(c) & (np.abs(c - d) <= FAIL_TOLERANCE * near) & (resolution <= FAIL_TOLERANCE * near)
    returns = last is not None and bool(np.any(back[last.stop + 1:-1] & back[last.stop + 2:]))
    if last is not None and not last.agrees(d) and not supported and not returns:
        miss = abs(d - last.value)
        relative = f"absolute {miss:.2e}" if last.near_zero else f"relative {miss / abs(last.value):.1e}"
        return {"verdict": "failed", "claimed": d, "converged": last.value,
                "steps": [steps[last.start], steps[last.stop]],
                "note": f"{describe(last, steps)}, but the claim is {d:.6e} ({relative})"}
    zero = last if last is not None and last.near_zero and last.agrees(d) else \
        identical[-1] if last is None and supported else None
    if zero is not None:
        return {"verdict": "passed", "claimed": d, "steps": [steps[zero.start], steps[zero.stop]],
                "note": f"{describe(zero, steps)}, as claimed ({d:.1e})"}
    scale = np.maximum(np.maximum(abs(d), np.abs(c)), gradient_scale)
    agreeing = (np.isfinite(c) & (np.abs(c - d) <= PASS_TOLERANCE * scale)
                & (resolution <= PASS_TOLERANCE * scale))
    pairs = np.flatnonzero(agreeing[:-1] & agreeing[1:])
    if pairs.size and (last is None or last.agrees(d)):
        k = int(pairs[0])
        error = float(np.max(np.abs(c[k:k + 2] - d) / scale[k:k + 2]))
        return {"verdict": "passed", "claimed": d, "steps": [steps[k], steps[k + 1]],
                "note": f"agrees to {error:.1e} at steps {steps[k]:.1e} and {steps[k + 1]:.1e}"}
    seen = c[np.isfinite(c)]
    seen_range = f"[{seen.min():.3e}, {seen.max():.3e}]" if seen.size else "no finite value"
    return {"verdict": "not_tested", "claimed": d,
            "note": (f"the differences ranged over {seen_range} against a claim of {d:.3e}, neither agreeing "
                     "with it at two steps nor converging away from it; check at a nearby point or inspect "
                     "the quantity (a kink, noise or oscillation)")}


def judge(label: str, value: float, steps: np.ndarray, sweeps: List[Sweep], objective_floor: float) -> dict:
    """The quantity's verdict from its directions; ``objective_floor`` is the
    objective's part of the absolute floor (see ``main``)."""
    report = {"quantity": label, "value": value,
              "directional_derivatives": [sweep.claimed for sweep in sweeps],
              "finite_differences": [sweep.differences.tolist() for sweep in sweeps]}
    if not (np.isfinite(value) and all(np.isfinite(sweep.claimed) for sweep in sweeps)):
        return {**report, "passed": False, "verdict": "nonfinite",
                "note": "the value or the claimed gradient at x0 is not finite", "directions": []}
    # The quantity's own gradient scale: a direction nearly orthogonal to the
    # gradient is judged against it, not against its own tiny derivative.
    gradient_scale = max(abs(sweep.claimed) for sweep in sweeps)
    epsilon, tiny = float(np.finfo(float).eps), float(np.finfo(float).tiny)
    directions = [judge_direction(sweep, steps, gradient_scale,
                                  max(objective_floor, ABSOLUTE_FLOOR_ULPS * epsilon * abs(value) / sweep.norm,
                                      ABSOLUTE_FLOOR_ULPS * tiny))
                  for sweep in sweeps]
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
    # The objective's gradient scale anchors the absolute floor of a derivative:
    # a claim within ABSOLUTE_FLOOR_ULPS machine epsilons of it is round-off, so
    # an exactly zero row passes a round-off-size claim and fails a real one.
    # (The objective, not the largest row, anchors it: one huge row must not
    # hide errors in the others.)
    objective_claims = [abs(sweep.claimed) for sweep in sweeps[0] if np.isfinite(sweep.claimed)]
    objective_floor = ABSOLUTE_FLOOR_ULPS * np.finfo(float).eps * max(objective_claims, default=0.0)
    outcomes = [judge(label, quantity_value(memo(x0), index), steps, row, objective_floor)
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
