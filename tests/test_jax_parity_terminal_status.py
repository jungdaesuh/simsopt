"""The parity lanes' terminal label states the optimizer's own verdict.

The finding these tests pin: seven bounded cases stopped on their iteration cap
(SciPy status 1 on every lane) while their label said ``converged``, because the
label was derived from "finite and decreased" alone. A budget stop is neither
convergence nor failure, and a finite, decreasing objective promotes nothing.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import replace

import numpy as np
import pytest

from examples.jax.parity.terminal_status import (
    StageTermination,
    lane_terminal_status,
    stage_stopping_reason,
    stage_termination_from_values,
    status_convention_for_driver,
)
from simsopt_contracts.optimization_endpoint import (
    BUDGET_STOPPING_REASONS,
    normalized_terminal_status,
)

_CAPPED_SCIPY_STAGE = StageTermination(
    status_convention="scipy-lbfgsb",
    provider_success=False,
    provider_status=1,
    iterations=50,
    max_iterations=50,
    initial_gradient=np.asarray([1.0, -2.0]),
    final_gradient=np.asarray([1.0e-3, -2.0e-3]),
    final_parameters=np.asarray([0.5, 0.25]),
    final_objective=1.0e-3,
)
_CONVERGED_SCIPY_STAGE = replace(
    _CAPPED_SCIPY_STAGE,
    provider_success=True,
    provider_status=0,
    iterations=37,
)


@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        (_CONVERGED_SCIPY_STAGE, "converged"),
        (_CAPPED_SCIPY_STAGE, "iteration-limit"),
        # SciPy L-BFGS-B merges both budgets into status 1; stopping short of
        # the iteration cap can only be the evaluation budget.
        (replace(_CAPPED_SCIPY_STAGE, iterations=12), "evaluation-limit"),
        (
            replace(_CAPPED_SCIPY_STAGE, provider_status=2, iterations=0),
            "line-search-failed",
        ),
        (replace(_CAPPED_SCIPY_STAGE, final_objective=float("nan")), "nonfinite"),
        (
            replace(_CAPPED_SCIPY_STAGE, final_gradient=np.asarray([np.nan, 0.0])),
            "nonfinite",
        ),
        (
            replace(_CAPPED_SCIPY_STAGE, final_parameters=np.asarray([np.inf, 0.0])),
            "nonfinite",
        ),
        # A provider that claims success with a non-success status is not believed.
        (replace(_CAPPED_SCIPY_STAGE, provider_success=True), "failed"),
        (replace(_CAPPED_SCIPY_STAGE, iterations=51), "failed"),
        (
            replace(_CAPPED_SCIPY_STAGE, status_convention="private-lbfgsb"),
            "iteration-limit",
        ),
        (
            replace(
                _CAPPED_SCIPY_STAGE,
                status_convention="private-lbfgsb",
                provider_status=2,
            ),
            "line-search-failed",
        ),
        (
            replace(
                _CAPPED_SCIPY_STAGE,
                status_convention="private-lbfgsb",
                provider_status=6,
            ),
            "nonfinite",
        ),
        (
            replace(
                _CAPPED_SCIPY_STAGE,
                status_convention="private-lbfgsb",
                provider_status=99,
            ),
            "callback-stopped",
        ),
        (
            replace(_CAPPED_SCIPY_STAGE, status_convention="scipy-bfgs"),
            "iteration-limit",
        ),
        (
            replace(
                _CAPPED_SCIPY_STAGE, status_convention="scipy-bfgs", provider_status=2
            ),
            "line-search-failed",
        ),
    ),
)
def test_stage_stopping_reason_reads_the_emitter_tables(
    stage: StageTermination,
    expected: str,
) -> None:
    assert stage_stopping_reason(stage) == expected


def test_budget_stop_with_a_decreasing_objective_is_neither_converged_nor_failed() -> (
    None
):
    status = lane_terminal_status(
        scientific_predicate=True,
        stages=(_CAPPED_SCIPY_STAGE, _CAPPED_SCIPY_STAGE),
    )

    assert status.normalized_status == "budget_exhausted"
    assert status.success is False
    assert status.stage_stopping_reasons == ("iteration-limit", "iteration-limit")


@pytest.mark.parametrize(
    ("predicate", "reasons", "expected_status"),
    (
        (True, ("converged",), "converged"),
        (True, ("converged", "converged"), "converged"),
        (True, ("iteration-limit",), "budget_exhausted"),
        (True, ("evaluation-limit",), "budget_exhausted"),
        (True, ("converged", "iteration-limit"), "budget_exhausted"),
        (True, ("iteration-limit", "converged"), "budget_exhausted"),
        (True, ("line-search-failed",), "failed"),
        (True, ("converged", "line-search-failed"), "failed"),
        (True, ("iteration-limit", "nonfinite"), "failed"),
        (True, ("callback-stopped",), "failed"),
        (True, ("failed",), "failed"),
        (False, ("converged",), "failed"),
        (False, ("iteration-limit", "iteration-limit"), "failed"),
    ),
)
def test_normalized_terminal_status_truth_table(
    predicate: bool,
    reasons: tuple[str, ...],
    expected_status: str,
) -> None:
    status = normalized_terminal_status(
        scientific_predicate=predicate,
        stage_stopping_reasons=reasons,
    )

    assert status.normalized_status == expected_status
    assert status.success is (expected_status == "converged")
    assert status.stage_stopping_reasons == reasons


def test_a_lane_without_stages_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        normalized_terminal_status(scientific_predicate=True, stage_stopping_reasons=())


def test_only_the_two_declared_budgets_count_as_budget_stops() -> None:
    assert BUDGET_STOPPING_REASONS == frozenset({"iteration-limit", "evaluation-limit"})


def test_driver_ids_name_their_emitter_and_unknown_drivers_are_refused() -> None:
    assert status_convention_for_driver("scipy_lbfgsb") == "scipy-lbfgsb"
    assert status_convention_for_driver("simsopt_lbfgsb") == "private-lbfgsb"
    with pytest.raises(KeyError):
        status_convention_for_driver("simsopt_bfgs")


def test_stage_termination_reads_the_published_phase_values() -> None:
    initial = {
        "initial:objective_gradient": np.asarray([3.0, -4.0]),
        "initial:parameters": np.asarray([1.0, 1.0]),
        "initial:objective": np.asarray(2.0),
    }
    first = {
        "first:objective_gradient": np.asarray([0.1, -0.2]),
        "first:parameters": np.asarray([0.5, 0.25]),
        "first:objective": np.asarray(0.5),
    }

    stage = stage_termination_from_values(
        status_convention="scipy-lbfgsb",
        provider_success=False,
        provider_status=1,
        iterations=50,
        max_iterations=50,
        start=("initial", initial),
        end=("first", first),
        gradient_observable="objective_gradient",
    )

    np.testing.assert_array_equal(
        stage.initial_gradient, initial["initial:objective_gradient"]
    )
    np.testing.assert_array_equal(
        stage.final_gradient, first["first:objective_gradient"]
    )
    np.testing.assert_array_equal(stage.final_parameters, first["first:parameters"])
    assert stage.final_objective == 0.5
    assert stage_stopping_reason(stage) == "iteration-limit"
