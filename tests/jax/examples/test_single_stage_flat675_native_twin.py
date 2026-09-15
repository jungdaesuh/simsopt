"""Native C++ twin of the flat-675 example: smoke, CLI, and objective parity.

The example is a script, not a package, so the in-process checks load it the
same way the JAX sibling contract does — by file location, under a name that
cannot collide with ``examples/jax/3_Advanced/single_stage_flat675.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    flat675_objective,
)

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / "examples" / "3_Advanced" / "single_stage_flat675.py"
JAX = ROOT / "examples" / "jax" / "3_Advanced" / "single_stage_flat675.py"
BUILD = ROOT / "build" / "cp311-cp311-linux_x86_64"

ARCHIVED_NATIVE_B3_OBJECTIVE = 1.8133486877704454
OBJECTIVE_RTOL = 1.0e-10

JSON_TOP_LEVEL_KEYS = frozenset(
    {
        "example_id",
        "backend_mode",
        "platform",
        "precision",
        "scale",
        "status",
        "observables",
    }
)
JSON_OBSERVABLE_KEYS = frozenset(
    {
        "scale",
        "configuration",
        "formulation",
        "outer_dof_count",
        "lbfgs_history",
        "lbfgs_max_line_search_steps",
        "max_steps",
        "iterations_run",
        "objective_evaluations",
        "final_objective",
        "endpoint_finite",
        "host_step_transfers",
        "host_callback_transfers",
        "host_unclassified_transfers",
        "host_endpoint_transfers",
        "initial_weighted_terms",
    }
)


def _load_script(path: Path, module_name: str) -> ModuleType:
    specification = spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native_example() -> ModuleType:
    return _load_script(NATIVE, "native_single_stage_flat675")


@pytest.fixture(scope="module")
def jax_example() -> ModuleType:
    return _load_script(JAX, "jax_single_stage_flat675")


def _cpu_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "MPI4PY_RC_INITIALIZE": "false",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "8",
            "PYTHONPATH": f"{ROOT / 'src'}:{BUILD}",
        }
    )
    return environment


def _run_native(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(NATIVE), *args],
        cwd=str(ROOT),
        env=_cpu_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _json_payload(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert completed.returncode == 0, (
        f"native example failed rc={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout)


def test_smoke_json_is_ok_on_cpu() -> None:
    payload = _json_payload(_run_native(["--smoke", "--json"]))
    observables = payload["observables"]
    assert payload["status"] == "ok"
    assert payload["example_id"] == "flat675-single-stage-coupled-optimization"
    assert payload["backend_mode"] == "native_cpu"
    assert payload["platform"] == "cpu"
    assert payload["precision"] == "fp64"
    assert payload["scale"] == "bounded"
    assert frozenset(payload) >= JSON_TOP_LEVEL_KEYS
    assert frozenset(observables) >= JSON_OBSERVABLE_KEYS
    assert observables["scale"] == "bounded"
    assert observables["configuration"] == "repository-geometry"
    assert observables["formulation"] == "flat-coupled-single-stage"
    assert observables["outer_dof_count"] == FLAT675_OUTER_DOF_COUNT
    assert observables["max_steps"] == 2
    assert observables["endpoint_finite"] is True
    assert np.isfinite(observables["final_objective"])
    terms = observables["initial_weighted_terms"]
    assert frozenset(terms) == frozenset(FLAT675_OBJECTIVE_TERM_KEYS)


def test_cli_matches_the_jax_example_flags(
    native_example: ModuleType, jax_example: ModuleType
) -> None:
    assert native_example.BOUNDED_STEPS == jax_example.BOUNDED_STEPS
    assert native_example.NATIVE_DEFAULT_STEPS == jax_example.NATIVE_DEFAULT_STEPS
    assert native_example.EXAMPLE_ID == jax_example.EXAMPLE_ID
    native_help = native_example._parser().format_help()
    for flag in ("--smoke", "--json", "--max-steps", "--output-dir", "--bundle"):
        assert flag in native_help


def test_same_point_objective_agrees_with_jax_at_budget_three(
    native_example: ModuleType, jax_example: ModuleType
) -> None:
    """Both lanes evaluate the same 675-vector at the start and after budget 3."""
    native_problem = native_example._repository_problem(native_scale=False)
    jax_problem = jax_example._repository_problem("bounded")
    start = np.array(native_problem.pack(), dtype=np.float64, copy=True)
    jax_start = np.array(
        jax_problem.start_candidate.outer_vector(), dtype=np.float64, copy=True
    )
    assert start.shape == (FLAT675_OUTER_DOF_COUNT,)
    assert jax_start.shape == start.shape
    # Independent least-squares fits can differ at the ulp; the same-point
    # check below evaluates both objectives at ``start``, not at each
    # constructor's own default.

    def jax_value_at(vector: np.ndarray) -> float:
        return float(
            np.asarray(
                flat675_objective(
                    vector,
                    material=jax_problem.material,
                    objective_policy=jax_problem.objective_policy,
                    boozer_policy=jax_problem.boozer_policy,
                ),
                dtype=np.float64,
            )
        )

    native_start, native_gradient = native_problem.value_and_gradient(start)
    start_relative = abs(native_start - jax_value_at(start)) / max(
        abs(native_start), 1.0
    )
    assert start_relative <= OBJECTIVE_RTOL, (
        f"start native {native_start!r} vs jax relative {start_relative:.3e}"
    )
    assert native_gradient.shape == (FLAT675_OUTER_DOF_COUNT,)
    assert np.all(np.isfinite(native_gradient))

    payload = native_example._solve(
        native_problem,
        max_steps=3,
        scale="bounded",
        configuration="repository-geometry",
    )
    assert payload["observables"]["max_steps"] == 3
    endpoint = native_problem.pack()
    native_end = float(payload["observables"]["final_objective"])
    end_relative = abs(native_end - jax_value_at(endpoint)) / max(abs(native_end), 1.0)
    assert end_relative <= OBJECTIVE_RTOL, (
        f"budget-3 native {native_end!r} vs jax relative {end_relative:.3e}"
    )


def test_bundle_budget_three_matches_archived_native_b3(
    native_example: ModuleType,
) -> None:
    if not native_example.BUNDLE_ROOT.is_dir():
        pytest.skip("the frozen genuine-675 input bundle is host-local")
    payload = _json_payload(_run_native(["--bundle", "--json", "--max-steps", "3"]))
    observables = payload["observables"]
    assert payload["status"] == "ok"
    assert observables["configuration"] == "certified-frozen-bundle"
    assert observables["max_steps"] == 3
    assert observables["outer_dof_count"] == FLAT675_OUTER_DOF_COUNT
    objective = float(observables["final_objective"])
    relative = (
        abs(objective - ARCHIVED_NATIVE_B3_OBJECTIVE) / ARCHIVED_NATIVE_B3_OBJECTIVE
    )
    assert relative <= OBJECTIVE_RTOL, (
        f"native B3 {objective!r} vs archived {ARCHIVED_NATIVE_B3_OBJECTIVE!r} "
        f"relative {relative:.3e}"
    )
