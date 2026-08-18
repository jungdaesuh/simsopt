"""Behavioral contract for the shipped finite-build Stage-II JAX example.

The example is judged on what it computes, not on how its source is spelled.
One bounded run of the public ``solve()`` entry point is shared by every test
below, and each test states one thing that run must be true of: the published
observable set is exactly the agreed schema, every published number is finite,
the solve lowered the objective it decomposes, and the filament packs still
clear one another.  A source-shape test cannot see any of that -- an example
that imported the right names and published the right dictionary keys while
returning an unusable coil set would pass it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from simsopt_jax.examples import ExampleResult

EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "jax"
    / "3_Advanced"
    / "stage_two_optimization_finitebuild.py"
)
# ``main()`` runs the shipped bounded lane on this budget; ``solve()`` is
# driven with the same one so the tested run is the shipped one.
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


def _example() -> ModuleType:
    """Load the shipped script the way a reader would run it.

    Example scripts live outside any importable package, so this mirrors the
    loader the neighbouring example tests already use.
    """
    specification = importlib.util.spec_from_file_location(
        "jax_example_stage_two_optimization_finitebuild",
        EXAMPLE,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _example()


@pytest.fixture(scope="module")
def bounded_result(example: ModuleType, tmp_path_factory) -> ExampleResult:
    """One bounded solve, shared by every contract in this file."""
    output_directory = tmp_path_factory.mktemp("finitebuild-bounded")
    return example.solve(output_directory, BOUNDED_STEPS, "bounded")


def test_bounded_solve_publishes_exactly_the_agreed_observable_schema(
    bounded_result: ExampleResult,
) -> None:
    assert set(bounded_result.observables) == PUBLISHED_OBSERVABLES


def test_bounded_solve_reports_a_sound_result_that_spent_its_whole_budget(
    bounded_result: ExampleResult,
) -> None:
    """``ok`` is earned, and it is earned without the budget converging.

    A three-step budget must not reach the solver's own stopping criteria: if
    it did, this problem would be trivial and the improvement contract below
    would be measuring nothing.
    """
    observables = bounded_result.observables

    assert bounded_result.status == "ok"
    assert observables["solver_iterations"] == BOUNDED_STEPS
    assert observables["solver_success"] is False


def test_bounded_solve_publishes_finite_numbers_everywhere(
    bounded_result: ExampleResult,
) -> None:
    for name in NUMERIC_OBSERVABLES:
        values = np.asarray(bounded_result.observables[name], dtype=np.float64)
        assert np.all(np.isfinite(values)), (
            f"{name} published a nonfinite value: {bounded_result.observables[name]!r}"
        )


def test_bounded_solve_lowers_the_objective_it_decomposes(
    bounded_result: ExampleResult,
) -> None:
    """The run improved, and the three published terms are that objective.

    Improvement alone would pass on a published objective unrelated to the
    diagnostics beside it, so the decomposition is checked in the same test:
    the flux, length and distance terms must sum to what was published as the
    final objective.
    """
    observables = bounded_result.observables
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
    example: ModuleType,
    bounded_result: ExampleResult,
) -> None:
    """A crossed or coincident pack publishes a nonpositive clearance."""
    observables = bounded_result.observables
    coil_lengths = observables["coil_lengths"]

    assert observables["minimum_clearance"] > 0.0
    assert len(coil_lengths) == example.NUM_BASE_CURVES
    assert all(length > 0.0 for length in coil_lengths)


def test_bounded_solve_publishes_one_gradient_entry_per_solved_coordinate(
    bounded_result: ExampleResult,
) -> None:
    observables = bounded_result.observables

    assert len(observables["solution"]) > 0
    assert len(observables["gradient"]) == len(observables["solution"])
