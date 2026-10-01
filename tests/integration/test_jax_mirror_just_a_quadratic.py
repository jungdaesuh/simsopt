from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax._lane_environment import build_execution_environment
from simsopt_jax.config import ExecutionIntent
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "jax" / "1_Simple" / "just_a_quadratic.py"
# A/reference-simple/runs/native-just-a-quadratic/captured-natural-omp1/capture.json
# measured on an independent official build of upstream 9e027eac3.
OFFICIAL_SOLUTION = (1.0, 1.9999999992096138, 2.9999999998773283)
OFFICIAL_SOLVER_STATUS = 1
OFFICIAL_FUNCTION_EVALUATIONS = 4
OFFICIAL_JACOBIAN_EVALUATIONS = 4


@pytest.mark.parametrize("intent", ("fast", "parity"))
def test_just_a_quadratic_matches_native_scientific_contract(
    intent: ExecutionIntent,
) -> None:
    assert EXAMPLE.is_file(), "the exact-name JAX mirror must exist"
    _, environment = build_execution_environment(
        "cpu",
        intent,
        os.environ,
        repo_root=REPO_ROOT,
    )
    environment["PYTHONPATH"] = pythonpath_with_loaded_kernel(
        *environment["PYTHONPATH"].split(os.pathsep)
    )
    completed = subprocess.run(
        (sys.executable, "-S", str(EXAMPLE), "--smoke", "--json"),
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    observables = payload["observables"]
    assert payload["example_id"] == "native-just-a-quadratic"
    assert payload["backend_mode"] == f"jax_cpu_{intent}"
    assert payload["platform"] == "cpu"
    assert payload["precision"] == "fp64"
    assert payload["status"] == "ok"
    np.testing.assert_array_equal(observables["initial_parameters"], np.zeros(3))
    np.testing.assert_array_equal(observables["targets"], (1.0, 2.0, 3.0))
    np.testing.assert_array_equal(observables["weights"], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        observables["initial_residuals"],
        (-1.0, -np.sqrt(2.0) * 2.0, -np.sqrt(3.0) * 3.0),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert observables["initial_objective"] == pytest.approx(36.0)
    # The official run stops on `gtol` 7.90e-10 away from the exact minimizer
    # (1, 2, 3), so an endpoint accuracy gate of our own would reject upstream
    # itself. The mirror is asserted against the official end point instead; the
    # distance to the exact minimizer stays a published diagnostic.
    np.testing.assert_allclose(
        observables["solution"],
        OFFICIAL_SOLUTION,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert observables["solver_status"] == OFFICIAL_SOLVER_STATUS
    assert observables["function_evaluations"] == OFFICIAL_FUNCTION_EVALUATIONS
    assert observables["jacobian_evaluations"] == OFFICIAL_JACOBIAN_EVALUATIONS
    assert observables["objective"] <= 1.0e-16
    assert observables["residual_norm"] <= 1.0e-8
    assert observables["gradient_inf_norm"] <= 1.0e-8
    assert observables["solver_success"] is True
