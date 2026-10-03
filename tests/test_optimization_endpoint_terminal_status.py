"""The shared terminal-status fold of ``simsopt_contracts.optimization_endpoint``.

A budget stop is neither convergence nor failure, and a finite, decreasing
objective promotes nothing: the scientific predicate can only demote.
"""

from __future__ import annotations

import pytest

from simsopt_contracts.optimization_endpoint import (
    BUDGET_STOPPING_REASONS,
    normalized_terminal_status,
)


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
