from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.jax._lane_environment import build_execution_environment
from simsopt_jax.config import ExecutionIntent


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "jax" / "1_Simple" / "just_a_quadratic.py"


@pytest.mark.parametrize("intent", ("fast", "parity"))
def test_just_a_quadratic_matches_native_scientific_contract(
    intent: ExecutionIntent,
) -> None:
    assert EXAMPLE.is_file(), "the exact-name JAX mirror must exist"
    _, environment = build_execution_environment("cpu", intent, os.environ)
    completed = subprocess.run(
        (sys.executable, str(EXAMPLE), "--smoke", "--json"),
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
    np.testing.assert_allclose(
        observables["solution"],
        (1.0, 2.0, 3.0),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert observables["objective"] <= 1.0e-16
    assert observables["residual_norm"] <= 1.0e-8
    assert observables["gradient_inf_norm"] <= 1.0e-8
    assert observables["solver_success"] is True
