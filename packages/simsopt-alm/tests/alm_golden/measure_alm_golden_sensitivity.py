#!/usr/bin/env python3
"""Measure how far each ALM golden scenario moves under last-bit noise.

Another CPU, SIMD path or BLAS build changes the last bits of the physics a
scenario evaluates, and ALM trajectories amplify such differences by different
amounts. This script emulates them with three perturbations of ``ulps`` units
of eps (u uniform in [-1, 1], one level of ``ULPS`` and one seed of ``SEEDS``
per sample):

* absolute: the evaluator sees ``x + ulps * eps * u * max(||x||_inf, 1)``.
  An absolute error scaled to the operands, not to the result, is what makes a
  near-zero value such as ``g = x0 - 1`` or a total that cancels its terms
  flip sign; the operands of every synthetic physics are the coordinates of x
  and O(1) constants (targets 0.97 to 10), hence ``max(||x||_inf, 1)``.
  Evaluating at a moved point keeps the evaluation self-consistent (a clipped
  violation stays exactly 0 where its signed value is negative);
* relative: every float the evaluation returns (scalars, arrays, gradient
  lists) is multiplied by ``1 + ulps * eps * u``;
* start: each coordinate of the run's x0 moves by one ulp, up or down (not
  a resumed run's, which must start at its checkpoint).

The evaluation noise is a deterministic function of the seed and the
evaluation's inputs (x, multipliers, penalty), as on a real machine: the same
inputs evaluated twice give the same bits, so the noise creates no ties or
order flips between repeated evaluations of one point that another CPU could
not produce. For every scenario it records:

* ``label_only``: the reason the scenario replays its outcomes only, when a
  perturbed run reaches other outcomes, takes another path (other boundaries),
  or moves a boundary value by more than ``NUMERIC_REPLAY_CEILING / 10``, so
  that its tolerance could not catch a regression of the ceiling's size;
* ``observed_outcomes``: the golden's outcomes and every outcome a perturbed
  run reached; off the recording environment a label-only scenario may reach
  any of them (the perturbed runs must also keep
  ``alm_golden_scenarios.result_invariant_violations`` empty);
* ``always_observed``: the outcomes every perturbed run reached; off the
  recording environment a scenario must reach each of its intended outcomes
  (``INTENDED_OUTCOMES``) that is among them;
* ``unstable_intended_outcomes``: the intended outcomes some perturbed run
  missed, which only the recording environment's exact replay checks;
* ``spread``: per boundary quantity, the largest deviation from the golden
  (``alm_golden_scenarios.boundary_deviations``) over the calibration runs.

Calibration uses ``SEEDS`` (144 samples over ``ULPS``), validation
``HOLDOUT_SEEDS`` (96 more); ``always_observed`` spans both. The measurement
fails, asking for more calibration seeds, when a held-out run of a numeric
scenario leaves its tolerances or path, or a held-out run of a label-only
scenario reaches an outcome outside ``observed_outcomes``: each recorded
outcome set is closed under the held-out samples.

The replay test sets each numeric tolerance to ``10 x max(spread, eps)``.
Rerun from the package directory (``packages/simsopt-alm``) whenever the
goldens or the ALM solver sources change (the test pins the source blob ids),
in the environment the goldens were recorded in (the generator's
``RECORDING_ENVIRONMENT_VARIABLES``; the script exits otherwise)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell \
        python tests/alm_golden/measure_alm_golden_sensitivity.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import alm_golden_scenarios as golden
from generate_alm_golden import (
    INTENDED_OUTCOMES,
    RECORDING_COMMAND_PREFIX,
    alm_source_blob_ids,
    recording_environment,
    require_recording_environment,
)

ULPS = (1, 2, 3, 4)
SEEDS = tuple(range(1, 37))
HOLDOUT_SEEDS = tuple(range(100, 124))
TOLERANCE_FACTOR = 10.0
# A regression of this relative size must fail the numeric replay.
NUMERIC_REPLAY_CEILING = 1e-10
EPS = golden.EPS
OUTPUT = golden.FIXTURE_DIR / "sensitivity.json"


def _perturbed(value, scale, rng):
    if isinstance(value, float):
        return value * (1.0 + scale * rng.uniform(-1.0, 1.0))
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return value * (1.0 + scale * rng.uniform(-1.0, 1.0, size=value.shape))
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(item, np.ndarray) and item.dtype.kind == "f" for item in value
    ):
        return type(value)(_perturbed(item, scale, rng) for item in value)
    return value


def _input_rng(seed, x, multipliers, penalty):
    digest = hashlib.sha256()
    for value in (np.int64(seed), x, multipliers, np.float64(penalty)):
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return np.random.RandomState(int.from_bytes(digest.digest()[:4], "little"))


def _noisy(evaluate_problem, scale, seed):
    def evaluate(x, multipliers, penalty):
        x = np.asarray(x, dtype=float)
        rng = _input_rng(seed, x, multipliers, penalty)
        operand = max(float(np.max(np.abs(x), initial=0.0)), 1.0)
        moved = x + scale * operand * rng.uniform(-1.0, 1.0, size=x.shape)
        evaluation = evaluate_problem(moved, multipliers, penalty)
        return {key: _perturbed(value, scale, rng) for key, value in evaluation.items()}

    return evaluate


def _perturbed_run(scenario, ulps, seed):
    """One run of ``scenario`` with its start and every evaluation perturbed."""
    wrap_evaluate_original = golden.TrajectoryRecorder.wrap_evaluate
    execute_original = golden.execute
    start_rng = np.random.RandomState(seed)

    def wrap_evaluate(recorder, evaluate_problem):
        return wrap_evaluate_original(recorder, _noisy(evaluate_problem, ulps * EPS, seed))

    def execute(run):
        if run.resume_state is not None:
            # A resumed run must start at its checkpoint's x.
            return execute_original(run)
        x0 = np.asarray(run.x0, dtype=float)
        direction = np.where(start_rng.uniform(size=x0.shape) < 0.5, -np.inf, np.inf)
        return execute_original(dataclasses.replace(run, x0=np.nextafter(x0, direction)))

    golden.TrajectoryRecorder.wrap_evaluate = wrap_evaluate
    golden.execute = execute
    try:
        return scenario.run()
    finally:
        golden.TrajectoryRecorder.wrap_evaluate = wrap_evaluate_original
        golden.execute = execute_original


def _samples(seeds):
    return [(ulps, seed) for ulps in ULPS for seed in seeds]


def measure_scenario(scenario) -> dict:
    """Calibrate on ``SEEDS``, then check the calibration on ``HOLDOUT_SEEDS``."""
    recorded = golden.load_golden(scenario.name)
    expected = golden.scenario_boundary_values(recorded["trajectory"])
    spread = {quantity: 0.0 for quantity in golden.BOUNDARY_QUANTITIES}
    observed = set(recorded["outcomes"])
    always = set(recorded["outcomes"])
    label_only = None
    runs = {}
    for sample in _samples(SEEDS) + _samples(HOLDOUT_SEEDS):
        trajectory = _perturbed_run(scenario, *sample)
        violations = golden.result_invariant_violations(trajectory)
        if violations:
            raise SystemExit(f"{scenario.name} under {sample}: {violations}")
        runs[sample] = trajectory
        always &= golden.scenario_outcomes(trajectory)
    for ulps, seed in _samples(SEEDS):
        trajectory = runs[ulps, seed]
        where = f"{ulps} ulp, seed {seed}"
        outcomes = sorted(golden.scenario_outcomes(trajectory))
        observed.update(outcomes)
        if outcomes != recorded["outcomes"]:
            changed = sorted(set(outcomes) ^ set(recorded["outcomes"]))
            label_only = label_only or f"outcomes change under {where}: {changed}"
            continue
        structural, deviations = golden.boundary_deviations(
            expected, golden.scenario_boundary_values(trajectory)
        )
        if structural is not None:
            label_only = label_only or f"path changes under {where}: {structural}"
            continue
        for quantity, deviation in deviations.items():
            spread[quantity] = max(spread[quantity], deviation)
    if label_only is None:
        ceiling = NUMERIC_REPLAY_CEILING / TOLERANCE_FACTOR
        sensitive = {q: d for q, d in spread.items() if d > ceiling}
        if sensitive:
            label_only = (
                "boundary values move beyond "
                f"{ceiling:.0e} under {ULPS[0]}-{ULPS[-1]} ulp noise: "
                + ", ".join(f"{q} {d:.1e}" for q, d in sensitive.items())
            )
    # Held-out samples: a numeric scenario must stay within its tolerances,
    # a label-only scenario within its observed outcomes.
    for ulps, seed in _samples(HOLDOUT_SEEDS):
        trajectory = runs[ulps, seed]
        outcomes = golden.scenario_outcomes(trajectory)
        if label_only is not None:
            unobserved = sorted(outcomes - observed)
            if unobserved:
                raise SystemExit(
                    f"{scenario.name}: held-out {ulps} ulp, seed {seed} reached "
                    f"{unobserved}, never observed in calibration; widen SEEDS"
                )
            continue
        structural, deviations = golden.boundary_deviations(
            expected, golden.scenario_boundary_values(trajectory)
        )
        exceeded = {
            q: d for q, d in deviations.items()
            if d > TOLERANCE_FACTOR * max(spread[q], EPS)
        }
        if sorted(outcomes) != recorded["outcomes"] or structural or exceeded:
            raise SystemExit(
                f"{scenario.name}: held-out {ulps} ulp, seed {seed} breaks the "
                f"calibration ({structural or exceeded or sorted(outcomes)}); widen it"
            )
    return {
        "label_only": label_only,
        "observed_outcomes": sorted(observed),
        "always_observed": sorted(always),
        "unstable_intended_outcomes": sorted(set(INTENDED_OUTCOMES[scenario.name]) - always),
        "spread": spread,
    }


def main() -> None:
    require_recording_environment(Path(__file__).name)
    if not golden.bitwise_environment():
        raise SystemExit(
            f"the goldens were recorded in {golden.recorded_environment()}, this "
            f"process runs {golden.current_environment()}; regenerate them first"
        )
    scenarios = {}
    for scenario in golden.SCENARIOS:
        scenarios[scenario.name] = measure_scenario(scenario)
        print(scenario.name, json.dumps(scenarios[scenario.name]), file=sys.stderr)
    payload = {
        "measure": f"{RECORDING_COMMAND_PREFIX} python "
                   "tests/alm_golden/measure_alm_golden_sensitivity.py",
        "noise": {
            "ulps": list(ULPS),
            "seeds": list(SEEDS),
            "holdout_seeds": list(HOLDOUT_SEEDS),
            "tolerance_factor": TOLERANCE_FACTOR,
            "numeric_replay_ceiling": NUMERIC_REPLAY_CEILING,
        },
        "environment": recording_environment(),
        "source_blob_ids": alm_source_blob_ids(),
        "scenarios": scenarios,
    }
    Path(OUTPUT).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
