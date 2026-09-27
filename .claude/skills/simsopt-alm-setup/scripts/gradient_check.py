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
library default; at least three steps and one direction are required (else
exit 2). The verdict comes from an explicit error model of the central
differences the library computes, ``c(e) = (q(x0 + e u) - q(x0 - e u)) / (2 e)``
along a unit direction ``u``:

    c(e) = d_true + a e^2 + noise(e),  noise(e) ~ ROUNDOFF_FACTOR * eps * |q| / e

with ``eps`` the float64 machine epsilon and ``|q| = |q(x0)| + |c(e)| e``. For
each direction, a least-squares fit of ``c = d + a e^2`` over a window of at
least three consecutive steps (weights from the noise model) extrapolates
``d_hat``, the derivative the differences imply, with its standard error.
The noise model is a floor: the fit residuals of all directions of the
quantity, pooled, scale it up when the data scatter more (noise beyond the
model, or truncation beyond ``e^2``). The window with the smallest pooled
uncertainty is used, and ``z`` is the two-sided ``CONFIDENCE`` quantile of
Student's t at the pooled residual degrees of freedom. With the claimed
derivative ``d`` and ``delta = |d - d_hat|``,
``tolerance = max(RELATIVE_TOLERANCE |d_hat|, z sigma)``:

- ``failed``: ``delta > FAIL_FACTOR * tolerance`` (a clear margin).
- ``not_tested``: otherwise, when ``|d_hat| <= z sigma`` (no measurable
  change), when ``z sigma > DECISION_LIMIT |d_hat|`` (too uncertain to
  confirm the gradient to that accuracy), or when ``delta`` is between the
  tolerance and the fail margin. The note says which steps to try: larger
  when round-off dominates, smaller when the data do not follow ``e^2``.
- ``passed``: ``delta <= tolerance``.

A quantity is ``nonfinite`` if any value or difference is not finite,
``failed`` if any direction fails, ``not_tested`` if no direction passes,
and ``passed`` otherwise (untested directions are counted in the note). The
library's ratio test is reported (``max_ratio``) but does not decide. The
last line printed is ``GRADIENT_CHECK {json}``; the exit status is 0 when
every quantity passes, 1 otherwise, and 2 for invalid arguments.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import t as student_t

from alm_problem import build_problem
from simsopt.solve.alm import ALMPhysics, run_directional_taylor_test

RESULT_PREFIX = "GRADIENT_CHECK "
OBJECTIVE_LABEL = "f"
# The noise floor: values are trusted to this many machine epsilons of their magnitude.
ROUNDOFF_FACTOR = 1.0
# A claimed derivative within this fraction of d_hat always passes (a relative floor).
RELATIVE_TOLERANCE = 1e-6
# Two-sided confidence of the z sigma bound (Student's t at the pooled residual dof).
CONFIDENCE = 0.99
# How far beyond the tolerance a difference must be to fail, not merely be undecided.
FAIL_FACTOR = 2.0
# A pass must confirm the gradient to this relative accuracy; a larger z sigma is not tested.
DECISION_LIMIT = 1e-3
MINIMUM_STEPS = 3
MACHINE_EPSILON = float(np.finfo(float).eps)
DEFAULT_DIRECTIONS = inspect.signature(run_directional_taylor_test).parameters["direction_count"].default


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


class WindowFit(NamedTuple):
    """The ``c = d + a e^2`` fit of one direction over one window of steps."""

    d_hat: float
    unit_sigma: float  # standard error of d_hat for the noise model alone
    chi2: float        # weighted residual sum of squares, in noise-model units
    weighted: bool     # False when the noise model is zero (all values and changes 0)


def fit_window(steps: np.ndarray, estimates: np.ndarray, value: float) -> WindowFit:
    noise = ROUNDOFF_FACTOR * MACHINE_EPSILON * (abs(value) + np.abs(estimates) * steps) / steps
    weighted = bool(np.all(noise > 0.0))
    weights = 1.0 / noise if weighted else np.ones_like(noise)
    # e^2 relative to the window's largest step keeps the system well scaled.
    design = np.column_stack([np.ones_like(steps), (steps / steps[0]) ** 2]) * weights[:, None]
    target = estimates * weights
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    covariance = np.linalg.inv(design.T @ design)
    return WindowFit(float(coefficients[0]), float(np.sqrt(covariance[0, 0])),
                     float(residual @ residual), weighted)


def pooled_fits(steps: np.ndarray, estimates: list, value: float) -> dict:
    """The window (at least MINIMUM_STEPS consecutive steps, the same for every
    direction) whose pooled uncertainty is smallest, with its fits, the
    residual scale of the noise model, the pooled dof and each sigma."""
    best = None
    for start in range(len(steps)):
        for stop in range(start + MINIMUM_STEPS, len(steps) + 1):
            fits = [fit_window(steps[start:stop], row[start:stop], value) for row in estimates]
            dof = len(fits) * (stop - start - 2)
            ratio = sum(fit.chi2 for fit in fits) / dof
            weighted = all(fit.weighted for fit in fits)
            # The noise model is a floor; larger residuals scale it up.
            scale = float(np.sqrt(max(1.0, ratio) if weighted else ratio))
            sigmas = [fit.unit_sigma * scale for fit in fits]
            score = sum(sigma * sigma for sigma in sigmas)
            if best is None or score < best["score"]:
                best = {"score": score, "fits": fits, "sigmas": sigmas, "dof": dof, "scale": scale,
                        "window": (start, stop)}
    return best


def judge_direction(claimed: float, fit: WindowFit, sigma: float, z: float) -> dict:
    """The verdict of one direction (the rules are in the module docstring)."""
    delta = abs(claimed - fit.d_hat)
    bound = z * sigma
    tolerance = max(RELATIVE_TOLERANCE * abs(fit.d_hat), bound)
    if delta > FAIL_FACTOR * tolerance:
        verdict = "failed"
    elif abs(fit.d_hat) <= bound:
        verdict = "no_change"
    elif bound > DECISION_LIMIT * abs(fit.d_hat):
        verdict = "uncertain"
    elif delta <= tolerance:
        verdict = "passed"
    else:
        verdict = "borderline"
    magnitude = abs(fit.d_hat)
    return {"verdict": verdict, "d_hat": fit.d_hat, "claimed": claimed, "delta": delta,
            "bound": bound, "relative_error": delta / magnitude if magnitude > 0.0 else None,
            "relative_bound": bound / magnitude if magnitude > 0.0 else None}


def steps_to_try(steps: np.ndarray, scale: float) -> str:
    """Advice for an undecided quantity: residuals near the noise model mean
    round-off limits the steps (try larger ones); far above it, the data do not
    follow e^2 (try smaller ones, or the quantity is not smooth there)."""
    if scale <= 10.0:
        larger = ",".join(f"{step:.3g}" for step in 10.0 * steps[:MINIMUM_STEPS])
        return f"round-off limits these steps: try larger ones, e.g. --epsilons {larger}"
    smaller = ",".join(f"{step:.3g}" for step in steps[-MINIMUM_STEPS:] / 10.0)
    return (f"the differences do not follow d + a e^2 (residuals {scale:.1e} x round-off): try "
            f"smaller steps, e.g. --epsilons {smaller}, or check the quantity is smooth at x0")


def judge(label: str, taylors: list) -> dict:
    """The quantity's verdict from its directions (see the module docstring)."""
    steps = np.asarray(taylors[0]["epsilons"], dtype=float)
    value = float(taylors[0]["base_total"])
    claims = [float(taylor["directional_derivative"]) for taylor in taylors]
    estimates = [np.asarray(taylor["central_estimates"], dtype=float) for taylor in taylors]
    ratios = [taylor["max_ratio"] for taylor in taylors if taylor["max_ratio"] is not None]
    report = {"quantity": label, "value": value, "max_ratio": max(ratios) if ratios else None,
              "directional_derivatives": claims,
              "finite_differences": [row.tolist() for row in estimates]}
    if not (np.isfinite(value) and np.all(np.isfinite(claims))
            and all(np.all(np.isfinite(row)) for row in estimates)):
        return {**report, "passed": False, "verdict": "nonfinite",
                "note": "a value or difference is not finite", "directions": []}
    pooled = pooled_fits(steps, estimates, value)
    z = float(student_t.ppf(0.5 + CONFIDENCE / 2.0, pooled["dof"]))
    directions = [judge_direction(claimed, fit, sigma, z)
                  for claimed, fit, sigma in zip(claims, pooled["fits"], pooled["sigmas"])]
    verdicts = [direction["verdict"] for direction in directions]
    relative = [direction["relative_error"] for direction in directions
                if direction["relative_error"] is not None]
    worst = f"{max(relative):.1e}" if relative else "n/a"
    window = steps[pooled["window"][0]:pooled["window"][1]]
    model = (f"fit over steps {window[0]:.3g}..{window[-1]:.3g}, residual scale {pooled['scale']:.1e}, "
             f"z={z:.2f}")
    if "failed" in verdicts:
        verdict = "failed"
        note = (f"{verdicts.count('failed')} of {len(verdicts)} directions: the claimed derivative misses "
                f"the extrapolated one by relative {worst}, beyond {FAIL_FACTOR:g} x the bound ({model})")
    elif "passed" not in verdicts:
        verdict = "not_tested"
        note = f"no direction decides ({', '.join(sorted(set(verdicts)))}); {steps_to_try(steps, pooled['scale'])}"
    else:
        verdict = "passed"
        undecided = len(verdicts) - verdicts.count("passed")
        note = f"relative error {worst} ({model})" + (
            f"; {undecided} of {len(verdicts)} directions undecided" if undecided else "")
    return {**report, "passed": verdict == "passed", "verdict": verdict, "note": note,
            "relative_error": max(relative) if relative else None, "residual_scale": pooled["scale"],
            "dof": pooled["dof"], "z": z, "window": [float(window[0]), float(window[-1])],
            "directions": directions}


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
    if args.epsilons is not None and len(args.epsilons.split(",")) < MINIMUM_STEPS:
        parser.error(f"--epsilons needs at least {MINIMUM_STEPS} steps: the error model fits "
                     "d + a e^2 and needs residuals to estimate its uncertainty")
    problem = build_problem(smoke=args.smoke)
    epsilons = (tuple(float(value) for value in args.epsilons.split(","))
                if args.epsilons else problem.taylor_epsilons)
    if epsilons is not None and len(epsilons) < MINIMUM_STEPS:
        parser.error(f"the problem's taylor_epsilons need at least {MINIMUM_STEPS} steps")
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
        flag = f"  ({outcome['note']})"
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
