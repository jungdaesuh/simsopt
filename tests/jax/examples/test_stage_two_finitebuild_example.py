"""Behavioral contract for the shipped finite-build Stage-II JAX example.

The example is judged on what it computes, not on how its source is spelled.
One bounded run of the shipped script, executed as the runner executes it (a
fresh ``--smoke --json`` child), is shared by every test below, and each test states one thing that run must be true of: the published
observable set is exactly the agreed schema, every published number is finite,
the solve lowered the objective it decomposes, and the filament packs still
clear one another.  A source-shape test cannot see any of that -- an example
that imported the right names and published the right dictionary keys while
returning an unusable coil set would pass it.

That shared run uses the host SciPy provider upstream calls, and the last
test checks that the published ``solver_driver`` names it.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import jax
import numpy as np
import pytest
import simsoptpp
from examples.jax._lane_environment import build_execution_environment
from simsopt_jax.solve.driver import Driver

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = (
    REPO_ROOT
    / "examples"
    / "jax"
    / "3_Advanced"
    / "stage_two_optimization_finitebuild.py"
)
# ``main()`` runs the shipped bounded lane (``--smoke``) on this budget.
BOUNDED_STEPS = 3
# Every key the example publishes.  Compared as a set, so a dropped observable
# and a silently added one both fail: downstream parity and receipt consumers
# read this dictionary by name.
PUBLISHED_OBSERVABLES = frozenset(
    {
        "initial_objective",
        "solution",
        "final_objective",
        "squared_flux",
        "length_penalty",
        "distance_penalty",
        "minimum_clearance",
        "coil_lengths",
        "gradient",
        "solver_driver",
        "solver_success",
        "solver_status",
        "solver_iterations",
    }
)
NUMERIC_OBSERVABLES = (
    "initial_objective",
    "solution",
    "final_objective",
    "squared_flux",
    "length_penalty",
    "distance_penalty",
    "minimum_clearance",
    "coil_lengths",
    "gradient",
)


def _module_constant(name: str) -> object:
    """A module-level literal of the shipped script, read without running it."""
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"), filename=str(EXAMPLE))
    value = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )
    return ast.literal_eval(value)


def _child_environment() -> dict[str, str]:
    _, environment = build_execution_environment(
        "cpu", "fast", os.environ, repo_root=REPO_ROOT
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(simsoptpp.__file__).resolve().parent),
            str(REPO_ROOT / "src"),
            str(sysconfig.get_paths()["purelib"]),
            str(Path(jax.__file__).resolve().parents[1]),
        )
    )
    return environment


@pytest.fixture(scope="module")
def bounded_result(tmp_path_factory) -> dict[str, object]:
    """One bounded run of the shipped script's default mode, shared below."""
    output_directory = tmp_path_factory.mktemp("finitebuild-bounded")
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            str(EXAMPLE),
            "--smoke",
            "--json",
            "--output-dir",
            str(output_directory),
        ),
        cwd=REPO_ROOT,
        env=_child_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert isinstance(payload, dict)
    return payload


def test_bounded_solve_publishes_exactly_the_agreed_observable_schema(
    bounded_result: dict[str, object],
) -> None:
    assert set(bounded_result["observables"]) == PUBLISHED_OBSERVABLES


def test_bounded_solve_reports_a_sound_result_that_spent_its_whole_budget(
    bounded_result: dict[str, object],
) -> None:
    """``ok`` is earned, and it is earned without the budget converging.

    A three-step budget must not reach the solver's own stopping criteria: if
    it did, this problem would be trivial and the improvement contract below
    would be measuring nothing.
    """
    observables = bounded_result["observables"]

    assert bounded_result["status"] == "ok"
    assert observables["solver_iterations"] == BOUNDED_STEPS
    assert observables["solver_success"] is False
    assert observables["solver_driver"] == Driver.SCIPY_LBFGSB.value


def test_bounded_solve_publishes_finite_numbers_everywhere(
    bounded_result: dict[str, object],
) -> None:
    for name in NUMERIC_OBSERVABLES:
        values = np.asarray(bounded_result["observables"][name], dtype=np.float64)
        assert np.all(np.isfinite(values)), (
            f"{name} published a nonfinite value: {bounded_result['observables'][name]!r}"
        )


def test_bounded_solve_lowers_the_objective_it_decomposes(
    bounded_result: dict[str, object],
) -> None:
    """The run improved, and the three published terms are that objective.

    Improvement alone would pass on a published objective unrelated to the
    diagnostics beside it, so the decomposition is checked in the same test:
    the flux, length and distance terms must sum to what was published as the
    final objective.
    """
    observables = bounded_result["observables"]
    squared_flux = observables["squared_flux"]
    length_penalty = observables["length_penalty"]
    distance_penalty = observables["distance_penalty"]

    assert observables["final_objective"] < observables["initial_objective"]
    assert squared_flux > 0.0
    assert length_penalty >= 0.0
    assert distance_penalty >= 0.0
    np.testing.assert_allclose(
        observables["final_objective"],
        squared_flux + length_penalty + distance_penalty,
        rtol=1.0e-14,
        atol=0.0,
    )


def test_bounded_solve_keeps_the_filament_packs_clear_of_one_another(
    bounded_result: dict[str, object],
) -> None:
    """A crossed or coincident pack publishes a nonpositive clearance."""
    observables = bounded_result["observables"]
    coil_lengths = observables["coil_lengths"]

    assert observables["minimum_clearance"] > 0.0
    assert len(coil_lengths) == _module_constant("NUM_BASE_CURVES")
    assert all(length > 0.0 for length in coil_lengths)


def test_bounded_solve_publishes_one_gradient_entry_per_solved_coordinate(
    bounded_result: dict[str, object],
) -> None:
    observables = bounded_result["observables"]

    assert len(observables["solution"]) > 0
    assert len(observables["gradient"]) == len(observables["solution"])


def test_bounded_solve_names_the_upstream_provider(
    bounded_result: dict[str, object],
) -> None:
    """The run says which optimizer produced it: the provider upstream calls."""
    assert bounded_result["observables"]["solver_driver"] == Driver.SCIPY_LBFGSB.value
