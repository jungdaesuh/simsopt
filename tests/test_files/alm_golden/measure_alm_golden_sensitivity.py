#!/usr/bin/env python3
"""Measure how far each ALM golden scenario moves under last-bit noise.

Another CPU, SIMD path or BLAS build changes the last bits of the physics a
scenario evaluates, and ALM trajectories amplify such differences by different
amounts. This script emulates them: every evaluation the solver receives has
each float (scalars, arrays, gradient lists) multiplied by
``1 + ulps * eps * u`` with ``u`` uniform in [-1, 1], for each noise level in
``ULPS`` and each seed in ``SEEDS``. The noise is a deterministic function of
the seed and the evaluation's inputs (x, multipliers, penalty), as on a real
machine: the same inputs evaluated twice give the same bits, so the noise
creates no ties or order flips between repeated evaluations of one point that
another CPU could not produce. For every scenario it records:

* ``label_only``: the reason the scenario replays its outcomes only, when a
  perturbed run reaches other outcomes, takes another path (other boundaries),
  or moves a boundary value by more than ``NUMERIC_REPLAY_CEILING / 10``, so
  that its tolerance could not catch a regression of the ceiling's size;
* ``observed_outcomes``: the golden's outcomes and every outcome a perturbed
  run reached; off the recording environment a label-only scenario may reach
  any of them (the perturbed runs must also keep
  ``alm_golden_scenarios.result_invariant_violations`` empty);
* ``spread``: per boundary quantity, the largest deviation from the golden
  (``alm_golden_scenarios.boundary_deviations``) over all perturbed runs.

The replay test sets each numeric tolerance to ``10 x max(spread, eps)``.
Rerun from the repository root, single-threaded, whenever the goldens or the
ALM sources change (the test pins the source blob ids)::

    OMP_NUM_THREADS=1 python tests/test_files/alm_golden/measure_alm_golden_sensitivity.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

import alm_golden_scenarios as golden
from generate_alm_golden import alm_source_blob_ids

ULPS = (2, 4)
SEEDS = (1, 2, 3, 4, 5, 6)
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
        evaluation = evaluate_problem(x, multipliers, penalty)
        rng = _input_rng(seed, x, multipliers, penalty)
        return {key: _perturbed(value, scale, rng) for key, value in evaluation.items()}

    return evaluate


def _perturbed_run(scenario, ulps, seed):
    """One run of ``scenario`` with every evaluation perturbed."""
    original = golden.TrajectoryRecorder.wrap_evaluate

    def wrap_evaluate(recorder, evaluate_problem):
        return original(recorder, _noisy(evaluate_problem, ulps * EPS, seed))

    golden.TrajectoryRecorder.wrap_evaluate = wrap_evaluate
    try:
        return scenario.run()
    finally:
        golden.TrajectoryRecorder.wrap_evaluate = original


def measure_scenario(scenario) -> dict:
    recorded = golden.load_golden(scenario.name)
    expected = golden.scenario_boundary_values(recorded["trajectory"])
    spread = {quantity: 0.0 for quantity in golden.BOUNDARY_QUANTITIES}
    observed = set(recorded["outcomes"])
    label_only = None
    for ulps in ULPS:
        for seed in SEEDS:
            trajectory = _perturbed_run(scenario, ulps, seed)
            where = f"{ulps} ulp, seed {seed}"
            violations = golden.result_invariant_violations(trajectory)
            if violations:
                raise SystemExit(f"{scenario.name} under {where}: {violations}")
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
    return {"label_only": label_only, "observed_outcomes": sorted(observed), "spread": spread}


def main() -> None:
    scenarios = {}
    for scenario in golden.SCENARIOS:
        scenarios[scenario.name] = measure_scenario(scenario)
        print(scenario.name, json.dumps(scenarios[scenario.name]), file=sys.stderr)
    payload = {
        "measure": "OMP_NUM_THREADS=1 python "
                   "tests/test_files/alm_golden/measure_alm_golden_sensitivity.py",
        "noise": {
            "ulps": list(ULPS),
            "seeds": list(SEEDS),
            "tolerance_factor": TOLERANCE_FACTOR,
            "numeric_replay_ceiling": NUMERIC_REPLAY_CEILING,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "machine": platform.machine(),
        },
        "source_blob_ids": alm_source_blob_ids(),
        "scenarios": scenarios,
    }
    Path(OUTPUT).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
