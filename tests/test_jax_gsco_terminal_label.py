"""The GSCO terminal label is derived from the returned arrays, never asserted.

``simsoptpp::GSCO`` returns its currents, loop counts and history arrays and
only prints its stop line, so ``simsopt_jax.examples.solver_terminal_status``
reads the stop reason back out of the returned data. These tests pin that
derivation against the C++ stop rules (``wireframe_optimization.cpp``) and
against counts the official upstream ``9e027eac3`` example runs printed,
written here as literals.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
from simsopt_jax.examples.solver_terminal_status import (
    GSCO_MAXIMUM_ITERATION_REACHED,
    GSCO_MINIMUM_OBJECTIVE_REACHED,
    GSCO_MULTISTEP_NO_ACCEPTED_UPDATE,
    GSCO_MULTISTEP_STABLE_CURRENTS,
    GSCO_MULTISTEP_STAGE_BUDGET_REACHED,
    GSCO_NO_ACCEPTED_UPDATE,
    GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP,
    gsco_multistep_terminal_label,
    gsco_terminal_label,
)


def test_gsco_official_history_tail_reads_as_minimum_objective_reached() -> None:
    """Official gsco-modular ends on an accepted undo pair, below its cap.

    The official upstream run of ``wireframe_gsco_modular.py``:
    ``history:loops[-3:] == [2073, 2073, 2073]`` and
    ``history:currents[-3:] == [+208333.33, -208333.33, +208333.33]``,
    with ``history:iterations[-1] == 1848`` of ``max_iter`` 2000 and the
    printed stdout line "Stopping iterations: minimum objective reached".
    """

    loops = np.asarray([2071, 2073, 2073, 2073], dtype=np.int64)
    currents = np.asarray(
        [1.0, 208333.33333333334, -208333.33333333334, 208333.33333333334],
        dtype=np.float64,
    )
    label = gsco_terminal_label(
        accepted_updates=1848,
        max_iterations=2000,
        loop_history=loops,
        current_history=currents,
        endpoint_usable=True,
    )
    assert label.raw_status == GSCO_MINIMUM_OBJECTIVE_REACHED
    assert label.normalized_status == "converged"
    assert label.success is True


def test_gsco_reaching_its_cap_is_a_budget_stop_not_convergence() -> None:
    loops = np.asarray([0, 64, 54], dtype=np.int64)
    currents = np.asarray(
        [0.0, -208333.33333333334, -208333.33333333334],
        dtype=np.float64,
    )
    label = gsco_terminal_label(
        accepted_updates=40,
        max_iterations=40,
        loop_history=loops,
        current_history=currents,
        endpoint_usable=True,
    )
    assert label.raw_status == GSCO_MAXIMUM_ITERATION_REACHED
    assert label.normalized_status == "budget_exhausted"
    assert label.success is False


def test_gsco_stopping_below_its_cap_without_an_undo_pair_is_convergence() -> None:
    """Upstream prints "no eligible loops" or "minimum objective reached" here.

    ``simsoptpp::GSCO`` returns ``x``, ``loop_count`` and six history arrays and
    no stop flag (``wireframe_optimization.cpp``), so a rejected last
    candidate cannot be separated from an empty eligibility set. Both are the
    solver stopping itself, so both are ``converged`` under one raw status.
    """

    loops = np.asarray([0, 12, 34], dtype=np.int64)
    currents = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
    label = gsco_terminal_label(
        accepted_updates=2,
        max_iterations=40,
        loop_history=loops,
        current_history=currents,
        endpoint_usable=True,
    )
    assert label.raw_status == GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP
    assert label.normalized_status == "converged"


def test_gsco_unusable_endpoint_is_failed() -> None:
    label = gsco_terminal_label(
        accepted_updates=40,
        max_iterations=40,
        loop_history=np.asarray([0, 1, 2], dtype=np.int64),
        current_history=np.asarray([0.0, 1.0, 1.0], dtype=np.float64),
        endpoint_usable=False,
    )
    assert label.normalized_status == "failed"
    assert label.success is False


def test_gsco_without_a_single_accepted_update_is_failed() -> None:
    """Zero accepted loops returns the caller's currents untouched."""

    label = gsco_terminal_label(
        accepted_updates=0,
        max_iterations=2000,
        loop_history=np.asarray([0], dtype=np.int64),
        current_history=np.asarray([0.0], dtype=np.float64),
        endpoint_usable=True,
    )
    assert label.raw_status == GSCO_NO_ACCEPTED_UPDATE
    assert label.normalized_status == "failed"
    assert label.success is False


def test_multistep_label_reads_the_per_stage_budget() -> None:
    """Official multistep: 2167/763/539/379/155/2/136 accepted of 2500 each.

    The per-call ``iterations`` and allocated iteration budget of the official
    upstream run of ``wireframe_gsco_multistep.py``.
    """

    official = np.asarray([2167, 763, 539, 379, 155, 2, 136], dtype=np.int64)
    budget = np.full(official.shape, 2500, dtype=np.int64)
    converged = gsco_multistep_terminal_label(
        stage_iterations=official,
        stage_budget=budget,
        final_adjustment_run=True,
        endpoint_usable=True,
        limit_raw_status="stage_history_capacity_exhausted_without_final_adjustment",
    )
    assert converged.normalized_status == "converged"
    assert converged.raw_status == GSCO_MULTISTEP_STABLE_CURRENTS
    assert converged.success is True

    at_budget = official.copy()
    at_budget[3] = 2500
    exhausted = gsco_multistep_terminal_label(
        stage_iterations=at_budget,
        stage_budget=budget,
        final_adjustment_run=True,
        endpoint_usable=True,
        limit_raw_status="stage_history_capacity_exhausted_without_final_adjustment",
    )
    assert exhausted.normalized_status == "budget_exhausted"
    assert exhausted.raw_status == GSCO_MULTISTEP_STAGE_BUDGET_REACHED
    assert exhausted.success is False

    empty_first = official.copy()
    empty_first[0] = 0
    degenerate = gsco_multistep_terminal_label(
        stage_iterations=empty_first,
        stage_budget=budget,
        final_adjustment_run=True,
        endpoint_usable=True,
        limit_raw_status="stage_history_capacity_exhausted_without_final_adjustment",
    )
    assert degenerate.normalized_status == "failed"
    assert degenerate.raw_status == GSCO_MULTISTEP_NO_ACCEPTED_UPDATE
    assert degenerate.success is False
