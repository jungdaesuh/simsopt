"""The shipped multistep GSCO example reports its completion from per-stage data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_public_multistep_example_discloses_completion_and_unknown_iterations() -> None:
    repository = Path(__file__).resolve().parents[2]
    example = repository / "examples/jax/3_Advanced/wireframe_gsco_multistep.py"
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, str(example), "--smoke", "--json"],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    published = json.loads(completed.stdout.splitlines()[-1])
    observables = published["observables"]
    assert published["status"] == "ok"
    assert observables["final_adjustment_run"] is True
    assert observables["constraints_satisfied"] is True
    # The example derives its completion from the per-stage arrays, never from
    # "final normal error below initial", which it publishes as a diagnostic
    # only.
    assert observables["terminal_status"] == "converged"
    assert observables["terminal_reason"] == "stable_current_final_adjustment_complete"
    assert observables["solver_success"] is True
    assert isinstance(observables["normal_error_decreased"], bool)
    assert all(
        iterations < budget
        for iterations, budget in zip(
            observables["stage_iterations"],
            observables["stage_allocated_iteration_budget"],
            strict=True,
        )
    )
    assert observables["iterations"] is None
    assert observables["allocated_iteration_budget"] == 120
    assert len(observables["stage_iterations"]) == 3
    assert observables["stage_allocated_iteration_budget"] == [40, 40, 40]
    assert all(0 <= value <= 40 for value in observables["stage_iterations"])
    assert len(observables["stage_objectives"]) == 3
