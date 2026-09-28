#!/usr/bin/env python3
"""Record the ALM golden trajectories of ``alm_golden_scenarios.py``.

Regenerate from the repository root, and only for a reviewed change that is
meant to alter ALM behavior (a refactor that needs new goldens has changed
behavior), in the recording environment (``RECORDING_ENVIRONMENT_VARIABLES``:
one thread per library and OpenBLAS's Haswell kernels)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell \
        python tests/test_files/alm_golden/generate_alm_golden.py

``--output-dir DIR`` writes the fixtures and manifest to ``DIR`` instead, in
any environment, e.g. to compare the trajectories two environments or two
trees produce (``diff -r``; only ``manifest.json``'s environment and source
ids may differ). Writing to this directory requires the recording
environment.

``--provenance-only`` rewrites only ``manifest.json`` (environment and source
blob ids) after a change that must not alter ALM behavior, e.g. comments or
annotations: it re-runs every scenario and refuses to write unless each fixture
it would write is byte-identical to the one on disk.

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
   and the machine the replay needs for bitwise equality, the git blob ids of
   the ``simsopt.solve.alm`` sources that produced them and of the scripts
   whose rules judge a replay, and the outcome union).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy

import alm_golden_scenarios as golden
import simsopt.solve.alm as alm

# The environment the goldens and their sensitivity are recorded in: one
# thread per library and OpenBLAS's Haswell (AVX2) kernels, which every x86_64
# CI runner has; OpenBLAS would otherwise pick the host's own (SkylakeX on an
# AVX-512 host, absent on GitHub's AMD runners), and the kernels decide the
# last bits of every replay.
RECORDING_ENVIRONMENT_VARIABLES = MappingProxyType({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OPENBLAS_CORETYPE": "Haswell",
})
RECORDING_COMMAND_PREFIX = " ".join(
    f"{name}={value}" for name, value in RECORDING_ENVIRONMENT_VARIABLES.items()
)


def recording_environment() -> dict:
    """This process's environment fingerprint, as ``manifest.json`` and
    ``sensitivity.json`` record it."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "machine": platform.machine(),
        "openblas_coretype": golden.openblas_coretype(),
    }


def require_recording_environment(script: str) -> None:
    """Exit unless this process has ``RECORDING_ENVIRONMENT_VARIABLES`` and
    runs the OpenBLAS kernels they pin."""
    differing = {
        name: os.environ.get(name)
        for name, value in RECORDING_ENVIRONMENT_VARIABLES.items()
        if os.environ.get(name) != value
    }
    kernels = golden.openblas_coretype()
    if differing or kernels != RECORDING_ENVIRONMENT_VARIABLES["OPENBLAS_CORETYPE"]:
        raise SystemExit(
            f"run in the recording environment: {RECORDING_COMMAND_PREFIX} python "
            f"tests/test_files/alm_golden/{script} (this process: {differing or 'variables set'}, "
            f"OpenBLAS kernels {kernels!r})"
        )


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


# The golden-directory scripts whose rules decide a replay: the scenarios,
# INTENDED_OUTCOMES here, and the noise and tolerance rules of the
# sensitivity measurement.
PROVENANCE_SCRIPTS = (
    "alm_golden_scenarios.py",
    "generate_alm_golden.py",
    "measure_alm_golden_sensitivity.py",
)


def alm_source_blob_ids() -> dict[str, str]:
    """Git blob ids of the imported ``simsopt.solve.alm`` sources and of
    ``PROVENANCE_SCRIPTS``, which ``manifest.json`` and ``sensitivity.json``
    record."""
    package_dir = Path(alm.__file__).resolve().parent
    paths = sorted(package_dir.glob("*.py")) + [
        golden.FIXTURE_DIR / script for script in PROVENANCE_SCRIPTS
    ]
    return {
        (f"simsopt/solve/alm/{path.name}" if path.parent == package_dir else path.name):
        _git_blob_id(path.read_bytes())
        for path in paths
    }


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=1) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=golden.FIXTURE_DIR)
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="rewrite only manifest.json; refuse unless every fixture is unchanged",
    )
    arguments = parser.parse_args()
    output_dir = arguments.output_dir
    if output_dir.resolve() == golden.FIXTURE_DIR:
        require_recording_environment(Path(__file__).name)
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

    fixtures = {}
    entries = []
    union = set()
    for scenario, trajectory, outcomes in recorded:
        union |= outcomes
        fixture = golden.fixture_path(scenario.name).name
        data = _json_bytes(
            {
                "format": golden.FIXTURE_FORMAT,
                "scenario": scenario.name,
                "description": scenario.description,
                "outcomes": sorted(outcomes),
                "trajectory": trajectory,
            },
        )
        fixtures[fixture] = data
        entries.append(
            {
                "scenario": scenario.name,
                "fixture": fixture,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    if arguments.provenance_only:
        changed = sorted(
            fixture for fixture, data in fixtures.items()
            if not (output_dir / fixture).is_file()
            or (output_dir / fixture).read_bytes() != data
        )
        if changed:
            raise SystemExit(
                f"--provenance-only: {changed} would change; regenerate the "
                "goldens instead (a behavior change needs review)"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for fixture, data in fixtures.items():
            (output_dir / fixture).write_bytes(data)
    manifest = _json_bytes(
        {
            "format": golden.FIXTURE_FORMAT,
            "regenerate": (
                f"{RECORDING_COMMAND_PREFIX} python "
                "tests/test_files/alm_golden/generate_alm_golden.py"
            ),
            "environment": recording_environment(),
            "source_blob_ids": alm_source_blob_ids(),
            "outcome_union": sorted(union),
            "scenarios": entries,
        },
    )
    (output_dir / "manifest.json").write_bytes(manifest)
    if arguments.provenance_only:
        print(f"{len(entries)} goldens unchanged; wrote manifest.json to {output_dir}")
    else:
        print(f"wrote {len(entries)} goldens and manifest.json to {output_dir}")


if __name__ == "__main__":
    main()
