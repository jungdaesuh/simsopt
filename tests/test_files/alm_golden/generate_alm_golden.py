#!/usr/bin/env python3
"""Record the ALM golden trajectories of ``alm_golden_scenarios.py``.

Regenerate from the repository root, single-threaded, and only for a reviewed
change that is meant to alter ALM behavior (a refactor that needs new goldens
has changed behavior)::

    OMP_NUM_THREADS=1 python tests/test_files/alm_golden/generate_alm_golden.py

``--output-dir DIR`` writes the fixtures and manifest to ``DIR`` instead, e.g.
to compare the trajectories two environments or two trees produce
(``diff -r``; only ``manifest.json``'s environment and source ids may differ).

For each scenario in ``alm_golden_scenarios.SCENARIOS`` this script:

1. runs it and derives its observable outcomes: the outer-step actions, the
   termination reason, the best-feasible restore reason, the history flags
   the run raised, the callbacks it received, a dual update that needed a
   penalty raise, a nonfinite evaluation the solver survived, a spent inner
   budget, and whether it resumed;
2. refuses to write when a scenario misses an outcome it exists to cover
   (``INTENDED_OUTCOMES``);
3. writes ``<scenario>.json`` (the encoded trajectory plus its outcomes) and
   ``manifest.json`` (fixture digests, the Python, numpy and SciPy versions
   the replay needs for bitwise equality, the git blob ids of the
   ``simsopt.solve.alm`` sources that produced them, and the outcome union).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

import alm_golden_scenarios as golden
import simsopt.solve.alm as alm

# What each scenario exists to exercise; generation fails if one is missed.
INTENDED_OUTCOMES = {
    "toy_convex": ("termination:converged",),
    "penalty_ramp_to_cap": (
        "action:penalty_increase",
        "action:sufficient_decrease_hold",
        "action:infeasible_stall_penalty_increase",
        "action:dual_update",
        "action:penalty_cap_reached",
        "termination:penalty_cap_reached",
        "callback:snapshot_accepted_state",
    ),
    "resume_penalty_ramp_mid_run": (
        "resumed",
        "callback:restore_incumbent_state",
        "termination:penalty_cap_reached",
    ),
    "hybrid_mismatch_repair": (
        "flag:signal_mismatch_active",
        "action:signal_mismatch_subproblem_limit_penalty_increase",
        "restored:final_iterate_infeasible",
        "termination:max_outer_restored_best_feasible",
    ),
    "hybrid_mismatch_penalty_increase": (
        "action:signal_mismatch_penalty_increase",
        "termination:max_outer_after_signal_mismatch_penalty_increase",
    ),
    "hybrid_mismatch_stall": (
        "action:signal_mismatch_stall",
        "termination:signal_mismatch_stall",
    ),
    "constraints_inactive_stall": (
        "action:constraints_inactive_stall",
        "termination:constraints_inactive_stall",
    ),
    "plateau_restore_best_feasible": (
        "action:subproblem_limit_penalty_increase",
        "action:subproblem_limit",
        "restored:final_iterate_worse_than_best_feasible",
        "termination:plateau_stall",
    ),
    "resume_plateau_best_feasible": (
        "resumed",
        "restored:final_iterate_worse_than_best_feasible",
        "termination:plateau_stall",
    ),
    "trust_radius_retries": (
        "nonfinite_evaluation",
        "inner_attempts_retried",
        "termination:max_outer_after_dual_update",
    ),
    "frozen_warm_start_restore": (
        "flag:dual_update_penalty_increase",
        "action:infeasible_stall_penalty_increase",
        "action:penalty_cap_reached",
        "restored:final_iterate_infeasible",
        "termination:penalty_cap_reached_restored_best_feasible",
    ),
    "cached_physics_smoothing": (
        "smoothing_changed",
        "action:constraints_inactive_converged",
        "termination:constraints_inactive_converged",
    ),
    "multiplier_cap_process_budget": (
        "flag:multiplier_cap_binding",
        "action:process_budget_exhausted",
        "termination:process_budget_exhausted",
    ),
    "inner_iteration_budget": ("inner_maxiter_budget_spent",),
    "dual_update_penalty_cap": (
        "dual_update_penalty_raise",
        "action:penalty_cap_reached",
        "termination:penalty_cap_reached",
    ),
    "preinner_converged": (
        "action:converged",
        "callback:snapshot_accepted_state",
        "termination:converged",
    ),
    "zero_inner_budget": (
        "action:subproblem_limit",
        "termination:plateau_stall",
    ),
}


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def alm_source_blob_ids() -> dict[str, str]:
    package_dir = Path(alm.__file__).resolve().parent
    paths = sorted(package_dir.glob("*.py")) + [Path(golden.__file__).resolve()]
    return {
        (f"simsopt/solve/alm/{path.name}" if path.parent == package_dir else path.name):
        _git_blob_id(path.read_bytes())
        for path in paths
    }


def _write_json(path: Path, payload: dict) -> bytes:
    text = json.dumps(payload, indent=1) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=golden.FIXTURE_DIR)
    output_dir = parser.parse_args().output_dir
    if set(INTENDED_OUTCOMES) != set(golden.SCENARIOS_BY_NAME):
        raise SystemExit("INTENDED_OUTCOMES and the scenario catalog disagree")

    recorded = []
    for scenario in golden.SCENARIOS:
        trajectory = scenario.run()
        outcomes = golden.scenario_outcomes(trajectory)
        missed = sorted(set(INTENDED_OUTCOMES[scenario.name]) - outcomes)
        if missed:
            raise SystemExit(f"{scenario.name} missed its intended outcomes {missed}")
        recorded.append((scenario, trajectory, outcomes))

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    union = set()
    for scenario, trajectory, outcomes in recorded:
        union |= outcomes
        fixture = golden.fixture_path(scenario.name).name
        data = _write_json(
            output_dir / fixture,
            {
                "format": golden.FIXTURE_FORMAT,
                "scenario": scenario.name,
                "description": scenario.description,
                "outcomes": sorted(outcomes),
                "trajectory": trajectory,
            },
        )
        entries.append(
            {
                "scenario": scenario.name,
                "fixture": fixture,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    _write_json(
        output_dir / "manifest.json",
        {
            "format": golden.FIXTURE_FORMAT,
            "regenerate": (
                "OMP_NUM_THREADS=1 python "
                "tests/test_files/alm_golden/generate_alm_golden.py"
            ),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "source_blob_ids": alm_source_blob_ids(),
            "outcome_union": sorted(union),
            "scenarios": entries,
        },
    )
    print(f"wrote {len(entries)} goldens and manifest.json to {output_dir}")


if __name__ == "__main__":
    main()
