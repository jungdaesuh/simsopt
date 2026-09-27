#!/usr/bin/env python
"""Taylor-test the objective and every constraint row of the problem in
``alm_problem.py`` separately.

    PYTHONPATH=<problem dir> python gradient_check.py [--smoke] [--directions N] [--epsilons E1,E2,...] [--seed S]

For f and for each row q, ``run_directional_taylor_test`` evaluates q at
``x0 +- e u`` along random unit directions ``u`` and forms the central
differences ``c(e) = (q(x0 + e u) - q(x0 - e u)) / (2 e)``. Testing each row on
its own matters, because in the augmented Lagrangian an inactive row
(``max(0, multiplier + penalty * g) = 0``) drops out, and its gradient would go
unchecked. Every quantity visits the same points, so each point's physics is
evaluated once and shared. The steps are ``--epsilons``, else the problem's
``taylor_epsilons``, else the library default: at least three distinct,
positive, finite steps, and at least one direction (else exit 2).

The verdict comes from an explicit model of the differences:

    c(e) = d_true + a e^2 + noise(e),    noise(e) ~ s * eps * Q / e

``eps`` is the float64 machine epsilon, ``Q`` the largest ``|q|`` sampled at
that step (``q(x0)``, ``q(x0 +- e u)``), and ``s >= 1`` the noise scale: 1 is
round-off of the values alone; the largest per-direction residual variance
of the fit raises it when the values are noisier (a cancellation, an inner
solve). In order, for each quantity:

1. Flat directions: when every ``|c(e)|`` is within ``FLAT_FACTOR`` times the
   round-off of the values (``eps * Q / e``, s = 1; exactly 0 when every
   sampled value is 0), the quantity does not change measurably along that
   direction. With the bound at the largest step, the claimed derivative
   ``d`` agrees when ``|d| <= DECISION_LIMIT * bound`` (zero at this
   resolution), fails when ``|d| > FAIL_FACTOR * bound``, and is undecided
   in between (it predicts a change below the resolution). A quantity flat
   along every direction with every claim agreeing passes ("flat along the
   sampled directions"); flat directions never pass a quantity that changes
   along others, since they carry no scale for its gradient.
2. The fit: for every other direction, a weighted least-squares fit of
   ``c = d_hat + a e^2`` over all the steps (weights from the noise model)
   gives the residuals that set ``s`` (pooled over directions) and ``z``,
   Student's t at ``CONFIDENCE`` (two-sided) for the pooled residual degrees
   of freedom. When ``|a| > z`` times its standard error the extrapolated
   ``d_hat`` is used; otherwise the ``e^2`` term is not measurable and
   extrapolating it would only amplify the small steps' noise, so ``d_hat``
   is the weighted mean of ``c``, with ``sigma`` the noise of its best single
   step (rounding can repeat across steps, so averaging is not credited).
3. Smoothness, two goodness-of-fit tests; either rejecting makes the
   quantity ``not_smooth``, never ``failed``:
   - kink: the second differences ``(q(x0 + e u) - 2 q(x0) + q(x0 - e u)) / e``
     are fitted to ``J + b e`` (a smooth quantity has ``J = 0``: the one-sided
     slopes agree as ``e -> 0``); the test rejects when ``|J| > z sigma_J``
     and ``|J| > KINK_FRACTION`` of the largest ``|c|`` (a slope jump, not a
     higher-order term);
   - resolution: ``s * eps``, the value noise the residuals imply relative to
     the values, exceeds ``NOISE_LIMIT``: no plausible noise explains the
     misfit, so ``c = d + a e^2`` does not describe the data at these steps.
4. Otherwise the model fits, and with ``delta = |d - d_hat|`` and
   ``tolerance = max(RELATIVE_TOLERANCE |d_hat|, z sigma)``: ``failed`` when
   ``delta > FAIL_FACTOR * tolerance``; ``passed`` when ``delta <= tolerance``
   and ``z sigma <= DECISION_LIMIT |d_hat|``; undecided otherwise (including
   a change within the noise, ``|d_hat| <= z sigma``).

A quantity is ``nonfinite`` if any value or difference is not finite;
``failed`` if any direction fails (a flat one included); else ``not_smooth``
if a smoothness test rejects; else ``passed`` if it is flat and agreeing
along every direction, or some fitted direction passed; else ``not_tested``. A ``not_tested`` note gives at most one
suggestion, derived from the model and never towards smaller steps: steps
larger by the power of ten the round-off needs, or, when a fitted
truncation ``a e^2`` that differs from 0 beyond the noise would exceed
``TRUNCATION_LIMIT |d_hat|`` at those, that no step decides. A ``not_smooth`` note gives no
step advice. The library's ratio test is reported (``max_ratio``) but does
not decide. The last line printed is ``GRADIENT_CHECK {json}``; the exit
status is 0 when every quantity passes, 1 otherwise, and 2 for invalid
arguments.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from typing import List, NamedTuple, Optional

import numpy as np
from scipy.stats import t as student_t

from alm_problem import build_problem
from simsopt.solve.alm import ALMPhysics, run_directional_taylor_test

RESULT_PREFIX = "GRADIENT_CHECK "
OBJECTIVE_LABEL = "f"
MACHINE_EPSILON = float(np.finfo(float).eps)
MINIMUM_STEPS = 3
DEFAULT_DIRECTIONS = inspect.signature(run_directional_taylor_test).parameters["direction_count"].default
# A difference within this many times the values' round-off is no measurable change.
FLAT_FACTOR = 3.0
# A claimed derivative within this fraction of d_hat always passes (a relative floor).
RELATIVE_TOLERANCE = 1e-6
# Two-sided confidence of the z sigma bounds (Student's t at the pooled residual dof).
CONFIDENCE = 0.99
# How far beyond the tolerance a difference must be to fail, not merely be undecided.
FAIL_FACTOR = 2.0
# A pass must confirm the gradient to this relative accuracy.
DECISION_LIMIT = 1e-3
# Kink test: a slope jump this small relative to the largest |c| is a higher-order term.
KINK_FRACTION = 1e-3
# Resolution test: value noise above this fraction of the values is not noise.
NOISE_LIMIT = 1e-2
# Residual scales above this are reported as noise beyond round-off of the value.
NOISY_SCALE = 100.0
# Larger steps are suggested only while the fitted truncation a e^2 stays below
# this fraction of |d_hat|.
TRUNCATION_LIMIT = 0.1


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


class DirectionSamples(NamedTuple):
    """One direction's data, steps largest first."""

    claimed: float        # the claimed directional derivative, grad . u
    differences: np.ndarray  # c(e)
    plus: np.ndarray      # q(x0 + e u)
    minus: np.ndarray     # q(x0 - e u)
    max_ratio: Optional[float]


def sample_direction(memo: PhysicsMemo, row_index: Optional[int], x0: np.ndarray,
                     direction: np.ndarray, steps: np.ndarray) -> DirectionSamples:
    """Run the library's Taylor test along ``direction`` and read back the
    values it evaluated (the memo holds them: the same points, bit for bit)."""
    taylor = run_directional_taylor_test(quantity_evaluator(memo, row_index), x0, np.zeros(0), 1.0,
                                         direction=direction, epsilons=tuple(steps))
    unit = np.asarray(taylor["direction"], dtype=float)
    x = np.asarray(x0, dtype=float)
    plus = [quantity_value(memo(x + float(step) * unit), row_index) for step in steps]
    minus = [quantity_value(memo(x - float(step) * unit), row_index) for step in steps]
    return DirectionSamples(float(taylor["directional_derivative"]),
                            np.asarray(taylor["central_estimates"], dtype=float),
                            np.asarray(plus), np.asarray(minus), taylor["max_ratio"])


def roundoff(steps: np.ndarray, value: float, sample: DirectionSamples) -> np.ndarray:
    """The round-off of each central difference: ``eps * Q / e``, with ``Q`` the
    largest |q| sampled at that step. ``Q`` is at least ``sqrt(eps)`` times the
    direction's largest |q|: an exact zero (a clipped or inactive branch)
    carries no special precision, and the weights stay comparable."""
    magnitude = np.maximum(np.maximum(np.abs(sample.plus), np.abs(sample.minus)), abs(value))
    floor = np.sqrt(MACHINE_EPSILON) * float(magnitude.max())
    return MACHINE_EPSILON * np.maximum(magnitude, floor) / steps


class LinearFit(NamedTuple):
    """Weighted least squares of ``y = p0 + p1 * t`` with unit-noise weights."""

    intercept: float
    slope: float
    intercept_sigma: float  # for the given noise, before any scale
    slope_sigma: float
    chi2: float             # residual sum of squares in noise units


def weighted_line(t: np.ndarray, y: np.ndarray, noise: np.ndarray) -> LinearFit:
    """By QR of the weighted design, so the covariance is not squared-conditioned."""
    design = np.column_stack([np.ones_like(t), t]) / noise[:, None]
    target = y / noise
    orthogonal, triangular = np.linalg.qr(design)
    coefficients = np.linalg.solve(triangular, orthogonal.T @ target)
    residual = target - design @ coefficients
    inverse = np.linalg.inv(triangular)
    covariance = inverse @ inverse.T
    return LinearFit(float(coefficients[0]), float(coefficients[1]), float(np.sqrt(covariance[0, 0])),
                     float(np.sqrt(covariance[1, 1])), float(residual @ residual))


def flat_verdict(sample: DirectionSamples, noise: np.ndarray) -> Optional[dict]:
    """The verdict of a direction whose differences are all within the round-off
    of the values (step 1 of the module docstring); None when it changes."""
    if not np.all(np.abs(sample.differences) <= FLAT_FACTOR * noise):
        return None
    bound = FLAT_FACTOR * float(noise.min())
    claimed = abs(sample.claimed)
    if claimed <= DECISION_LIMIT * bound:
        verdict = "flat"            # the claim is zero at this resolution, as the data are
    elif claimed > FAIL_FACTOR * bound:
        verdict = "flat_failed"     # the claim predicts a change the data would show
    else:
        verdict = "flat_undecided"  # the claim predicts a change below the resolution
    return {"verdict": verdict, "claimed": sample.claimed, "d_hat": 0.0, "delta": claimed, "bound": bound,
            "truncation": 0.0}


def advice(steps: np.ndarray, needed_factor: float, truncation_cap: Optional[float]) -> str:
    """One suggestion: steps larger by a power of ten covering ``needed_factor``,
    unless that passes ``truncation_cap`` (the largest step before the fitted
    truncation dominates); then no step decides."""
    factor = 10.0 ** max(1, math.ceil(math.log10(max(needed_factor, 1.0))))
    if truncation_cap is not None and steps[0] * factor > truncation_cap:
        return (f"no steps decide it under the model: round-off needs steps about {factor:g} x larger, "
                f"truncation needs steps below {truncation_cap:.2g}")
    larger = ",".join(f"{step:.3g}" for step in steps[:MINIMUM_STEPS] * factor)
    return f"round-off limits these steps: try --epsilons {larger}"


def largest_miss(directions: List[dict]) -> str:
    """The largest ``|claimed - d_hat|`` among ``directions``: relative to
    ``|d_hat|``, or absolute when ``d_hat`` is 0 (a flat direction)."""
    worst = max(directions, key=lambda direction: direction["delta"])
    if worst["d_hat"] != 0.0:
        return f"relative {worst['delta'] / abs(worst['d_hat']):.1e}"
    return f"absolute {worst['delta']:.2e}"


def judge(label: str, value: float, steps: np.ndarray, samples: List[DirectionSamples]) -> dict:
    """The quantity's verdict (the rules are in the module docstring)."""
    ratios = [sample.max_ratio for sample in samples if sample.max_ratio is not None]
    report = {"quantity": label, "value": value, "max_ratio": max(ratios) if ratios else None,
              "directional_derivatives": [sample.claimed for sample in samples],
              "finite_differences": [sample.differences.tolist() for sample in samples]}
    finite = np.isfinite(value) and all(
        np.isfinite(sample.claimed) and np.all(np.isfinite(sample.differences))
        and np.all(np.isfinite(sample.plus)) and np.all(np.isfinite(sample.minus)) for sample in samples)
    if not finite:
        return {**report, "passed": False, "verdict": "nonfinite",
                "note": "a value or difference is not finite", "directions": []}

    # Every changing direction has a positive round-off, so all fits are weighted
    # in the same (noise) units; a direction with q = 0 at every sample is flat.
    noises = [roundoff(steps, value, sample) for sample in samples]
    directions = [flat_verdict(sample, noise) for sample, noise in zip(samples, noises)]
    changing = [index for index, direction in enumerate(directions) if direction is None]

    scale, z, smooth_rejection, fits = 1.0, None, None, {}
    if changing:
        # 2. The fit c = d_hat + a e^2, in e^2 relative to the largest step.
        t = (steps / steps[0]) ** 2
        fits = {index: weighted_line(t, samples[index].differences, noises[index]) for index in changing}
        dof = len(changing) * (len(steps) - 2)
        # The noise is common to the quantity; the largest per-direction residual
        # variance estimates it, since rounding can repeat across one direction's
        # steps and leave its residuals near zero.
        scale = float(np.sqrt(max(1.0, max(fit.chi2 for fit in fits.values()) / (len(steps) - 2))))
        z = float(student_t.ppf(0.5 + CONFIDENCE / 2.0, dof))
        # 3. Smoothness: a slope jump in the second differences, then the noise the misfit implies.
        for index in changing:
            sample = samples[index]
            second = (sample.plus - 2.0 * value + sample.minus) / steps
            kink = weighted_line(steps / steps[0], second, scale * np.sqrt(12.0) * noises[index])
            largest_change = float(np.max(np.abs(sample.differences)))
            if abs(kink.intercept) > z * kink.intercept_sigma and \
                    abs(kink.intercept) > KINK_FRACTION * largest_change:
                smooth_rejection = (f"the one-sided slopes differ by {abs(kink.intercept):.2e} as the step "
                                    f"shrinks (kink test, z={z:.2f})")
                break
        if smooth_rejection is None and scale * MACHINE_EPSILON > NOISE_LIMIT:
            smooth_rejection = (f"the misfit to d + a e^2 would be value noise of {scale * MACHINE_EPSILON:.1e} "
                                f"of the values, above {NOISE_LIMIT:g} (resolution test)")
        # 4. Decisions under the model. The e^2 term is used only when it differs
        # from 0 beyond the noise; otherwise extrapolating it would only amplify
        # the noise of the small steps, and d_hat is the weighted mean of c.
        for index in changing:
            fit, sample = fits[index], samples[index]
            significant = abs(fit.slope) > z * scale * fit.slope_sigma
            if significant:
                d_hat, sigma = fit.intercept, scale * fit.intercept_sigma
            else:
                weights = 1.0 / noises[index] ** 2
                d_hat = float(np.sum(weights * sample.differences) / np.sum(weights))
                # No better than the best single step: rounding errors of one
                # direction's steps can be correlated, so averaging gains nothing.
                sigma = scale * float(noises[index].min())
            delta = abs(sample.claimed - d_hat)
            bound = z * sigma
            tolerance = max(RELATIVE_TOLERANCE * abs(d_hat), bound)
            if smooth_rejection is not None:
                verdict = "not_smooth"
            elif delta > FAIL_FACTOR * tolerance:
                verdict = "failed"
            elif delta <= tolerance and bound <= DECISION_LIMIT * abs(d_hat):
                verdict = "passed"
            else:
                verdict = "undecided"
            directions[index] = {"verdict": verdict, "claimed": sample.claimed, "d_hat": d_hat,
                                 "delta": delta, "bound": bound,
                                 # The fitted truncation a e_max^2 when significant.
                                 "truncation": fit.slope if significant else 0.0}

    verdicts = [direction["verdict"] for direction in directions]
    count = len(verdicts)
    noisy = (f"; the differences scatter {scale:.1e} x more than round-off of the values: noisier than "
             "the value suggests (a cancellation or an inner solve), or truncation beyond e^2"
             ) if scale > NOISY_SCALE else ""
    if "flat_failed" in verdicts or "failed" in verdicts:
        failed = [direction for direction in directions if direction["verdict"] in ("failed", "flat_failed")]
        verdict = "failed"
        note = (f"{len(failed)} of {count} directions: the claimed derivative misses the measured one by "
                f"{largest_miss(failed)}, beyond {FAIL_FACTOR:g} x the bound, under a model that fits")
    elif smooth_rejection is not None:
        verdict = "not_smooth"
        note = (f"the quantity is not smooth at this point along these steps: {smooth_rejection}. Move x "
                "slightly or check a kink/branch (structure finer than the smallest step also shows this way)")
    elif verdicts.count("flat") == count:
        verdict = "passed"
        bound = max(direction["bound"] for direction in directions)
        note = (f"flat along the sampled directions: every difference and claimed derivative is within "
                f"the round-off bound {bound:.1e} (errors below it are not detected)")
    elif "passed" in verdicts:
        # Flat directions carry no scale for the gradient, so only fitted ones pass it.
        verdict = "passed"
        passing = [direction for direction in directions if direction["verdict"] == "passed"]
        undecided = count - len(passing)
        note = f"within {largest_miss(passing)} of the measured derivative" + (
            f"; {undecided} of {count} directions not deciding" if undecided else "") + noisy
    else:
        verdict = "not_tested"
        # Round-off shrinks as 1/e: the factor that brings every bound below
        # DECISION_LIMIT of the derivative, capped where fitted truncation grows.
        needed, cap = 1.0, None
        for direction in directions:
            magnitude_hat = max(abs(direction["d_hat"]), abs(direction["claimed"]))
            needed = max(needed, direction["bound"] / (DECISION_LIMIT * magnitude_hat)
                         if magnitude_hat > 0.0 else 10.0)
            if direction["truncation"] != 0.0 and magnitude_hat > 0.0:
                limit = steps[0] * math.sqrt(TRUNCATION_LIMIT * magnitude_hat / abs(direction["truncation"]))
                cap = limit if cap is None else min(cap, limit)
        note = (f"no direction decides ({', '.join(sorted(set(verdicts)))}); {advice(steps, needed, cap)}"
                + noisy)
    fitted = [direction for direction in directions if direction["d_hat"] != 0.0]
    return {**report, "passed": verdict == "passed", "verdict": verdict, "note": note,
            "relative_error": max((direction["delta"] / abs(direction["d_hat"]) for direction in fitted),
                                  default=None),
            "residual_scale": scale, "z": z, "directions": directions}


def steps_problem(steps) -> Optional[str]:
    """Why ``steps`` cannot be used (None when they can): the model needs at
    least MINIMUM_STEPS distinct, positive, finite steps."""
    array = np.asarray(steps, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        return f"steps must be positive and finite, got {list(steps)}"
    if len(np.unique(array)) != len(array) or len(array) < MINIMUM_STEPS:
        return (f"at least {MINIMUM_STEPS} distinct steps are needed (the model fits d + a e^2 and "
                f"needs residuals), got {list(steps)}")
    return None


def epsilons_argument(text: str) -> tuple:
    """``--epsilons``: comma-separated steps (argparse reports a bad value, exit 2)."""
    steps = tuple(float(value) for value in text.split(","))
    problem = steps_problem(steps)
    if problem is not None:
        raise argparse.ArgumentTypeError(problem)
    return steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="build the problem at its smoke size")
    parser.add_argument("--directions", type=int, default=DEFAULT_DIRECTIONS,
                        help="random directions per quantity")
    parser.add_argument("--epsilons", type=epsilons_argument, help="comma-separated steps (at least 3)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the random directions")
    args = parser.parse_args(argv)
    if args.directions < 1:
        parser.error(f"--directions must be at least 1, got {args.directions}")
    problem = build_problem(smoke=args.smoke)
    chosen = args.epsilons if args.epsilons is not None else problem.taylor_epsilons
    if chosen is not None and steps_problem(chosen) is not None:
        parser.error(f"the problem's taylor_epsilons: {steps_problem(chosen)}")

    # The library's directions, drawn once; every quantity visits the same points.
    random = np.random.RandomState(args.seed)
    directions = [random.standard_normal(size=np.shape(problem.x0)) for _ in range(args.directions)]
    memo = PhysicsMemo(problem.physics)
    x0 = np.asarray(problem.x0, dtype=float)
    if chosen is None:
        # The library's default steps, read back from one test (its points stay in the memo).
        chosen = run_directional_taylor_test(quantity_evaluator(memo, None), x0, np.zeros(0), 1.0,
                                             direction=directions[0])["epsilons"]
    steps = np.sort(np.asarray(chosen, dtype=float))[::-1]
    labels = (OBJECTIVE_LABEL,) + tuple(problem.constraint_names)
    indices = (None,) + tuple(range(len(problem.constraint_names)))
    outcomes = [
        judge(label, quantity_value(memo(x0), index), steps,
              [sample_direction(memo, index, x0, direction, steps) for direction in directions])
        for label, index in zip(labels, indices)
    ]
    passed = all(outcome["passed"] for outcome in outcomes)
    status_names = {"passed": "PASS", "nonfinite": "NONFINITE", "not_tested": "NOT TESTED",
                    "not_smooth": "NOT SMOOTH", "failed": "FAIL"}
    for outcome in outcomes:
        max_ratio = "n/a" if outcome["max_ratio"] is None else f"{outcome['max_ratio']:.3f}"
        print(f"{outcome['quantity']:<32} {status_names[outcome['verdict']]:<10} "
              f"value={outcome['value']:+.4e} max_ratio={max_ratio}  ({outcome['note']})")
    print(RESULT_PREFIX + json.dumps({
        "passed": passed,
        "problem": problem.name,
        "epsilons": steps.tolist(),
        "directions": args.directions,
        "physics_evaluations": memo.evaluations,
        "quantities": outcomes,
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
