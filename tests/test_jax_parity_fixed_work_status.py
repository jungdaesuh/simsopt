"""The greedy parity producers derive their stop reason, never assert one.

Upstream GPMO and GSCO return arrays and print their stop line; the parity
lanes must read the reason back out of the returned data, the same way in every
lane. These tests pin the derivation against the official C++ rules at upstream
``9e027eac38028d57aa23777be52a781aa860e347`` and against the numbers the
official captures recorded, and they ratchet the producers at the source level
so no case can go back to ``"converged" if <scientific predicate> else
"failed"`` with the configured budget published as a measured count.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases._fixed_work_status import (
    FIXED_WORK_NORMALIZED_STATUS,
    GPMO_GRID_FULL,
    GPMO_ITERATION_COUNT_COMPLETED,
    GPMO_MAGNET_CAP_REACHED,
    GPMO_NONZERO_COUNT_STALLED,
    GSCO_MAXIMUM_ITERATION_REACHED,
    GSCO_MINIMUM_OBJECTIVE_REACHED,
    GSCO_MULTISTEP_NO_ACCEPTED_UPDATE,
    GSCO_MULTISTEP_STABLE_CURRENTS,
    GSCO_MULTISTEP_STAGE_BUDGET_REACHED,
    GSCO_NO_ACCEPTED_UPDATE,
    GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP,
    fixed_work_label,
    gpmo_stop_reason,
    gsco_multistep_terminal_label,
    gsco_terminal_label,
)
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_backtracking_outputs_usable,
    gpmo_baseline_outputs_usable,
)

CASES = Path(__file__).resolve().parents[1] / "examples" / "jax" / "parity" / "cases"

#: Greedy/fixed-work producer module -> ``LaneObservation`` constructions in it.
GREEDY_PRODUCERS = {
    "_permanent_magnet_arbvec.py": 1,
    "native_permanent_magnet_simple.py": 2,
    "native_permanent_magnet_qa.py": 1,
    "_wireframe_gsco.py": 1,
    "native_wireframe_gsco_multistep.py": 1,
}
HELPERS = frozenset(
    {
        "fixed_work_label",
        "gsco_terminal_label",
        "gsco_multistep_terminal_label",
    }
)


def test_gpmo_grid_full_outranks_the_magnet_cap() -> None:
    """``permanent_magnet_optimization.cpp:968-978`` tests ``N`` first."""

    assert (
        gpmo_stop_reason(nonzero_count=10, grid_size=10, magnet_cap=10)
        == GPMO_GRID_FULL
    )
    assert (
        gpmo_stop_reason(nonzero_count=10, grid_size=64, magnet_cap=10)
        == GPMO_MAGNET_CAP_REACHED
    )
    assert (
        gpmo_stop_reason(nonzero_count=9, grid_size=64, magnet_cap=10)
        == GPMO_ITERATION_COUNT_COMPLETED
    )


def test_gpmo_baseline_has_no_early_exit() -> None:
    """``GPMO_baseline`` (cpp:1237-1326) never breaks, so K always completes."""

    assert (
        gpmo_stop_reason(nonzero_count=40, grid_size=None, magnet_cap=None)
        == GPMO_ITERATION_COUNT_COMPLETED
    )


def test_official_muse_endpoint_is_the_magnet_cap() -> None:
    """Official MUSE: 5000 magnets of a 7530-dipole grid, cap 5000.

    The official reference record ``native-permanent-magnet-muse``:
    ``final:nonzero_count`` 5000, ``configuration.max_nMagnets`` 5000,
    ``configuration.ndipoles`` 7530, ``configuration.K`` 10000.
    """

    assert (
        gpmo_stop_reason(nonzero_count=5000, grid_size=7530, magnet_cap=5000)
        == GPMO_MAGNET_CAP_REACHED
    )


def test_a_greedy_lane_is_never_converged_and_never_counts_its_budget() -> None:
    label = fixed_work_label(
        raw_status=GPMO_MAGNET_CAP_REACHED,
        outputs_usable=True,
    )
    assert label.normalized_status == FIXED_WORK_NORMALIZED_STATUS
    assert label.normalized_status != "converged"
    assert label.success is True

    failed = fixed_work_label(raw_status=GPMO_GRID_FULL, outputs_usable=False)
    assert failed.normalized_status == "failed"
    assert failed.success is False


def test_official_pm4stell_objective_rise_is_not_a_failure() -> None:
    """Official PM4Stell rises 0.16177 -> 0.83456 and still finished (PM4-6).

    The branch's old predicate demanded ``final < initial``; that gate would
    have failed the official run, so it is a reported diagnostic and only the
    returned outputs decide ``success``.
    """

    initial_objective = 0.1617679350296839
    final_objective = 0.8345639630526893
    assert final_objective > initial_objective
    label = fixed_work_label(
        raw_status=gpmo_stop_reason(
            nonzero_count=1000,
            grid_size=8712,
            magnet_cap=1000,
        ),
        outputs_usable=True,
    )
    assert label.raw_status == GPMO_MAGNET_CAP_REACHED
    assert label.success is True


def test_gsco_official_history_tail_reads_as_minimum_objective_reached() -> None:
    """Official gsco-modular ends on an accepted undo pair, below its cap.

    The official capture of ``native-wireframe-gsco-modular``:
    ``final:history:loops[-3:] == [2073, 2073, 2073]`` and
    ``final:history:currents[-3:] == [+208333.33, -208333.33, +208333.33]``,
    with ``final:history:iterations[-1] == 1848`` of ``max_iter`` 2000 and the
    captured stdout line "Stopping iterations: minimum objective reached".
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
    no stop flag (``wireframe_optimization.cpp:340-349``), so a rejected last
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


def _is_constant(node: ast.expr, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


@pytest.mark.parametrize(("module", "observations"), sorted(GREEDY_PRODUCERS.items()))
def test_greedy_producer_takes_its_label_from_the_shared_derivation(
    module: str,
    observations: int,
) -> None:
    tree = ast.parse((CASES / module).read_text(encoding="utf-8"))
    nodes = tuple(ast.walk(tree))

    predicate_ternaries = [
        node
        for node in nodes
        if isinstance(node, ast.IfExp)
        and _is_constant(node.body, "converged")
        and _is_constant(node.orelse, "failed")
    ]
    assert predicate_ternaries == []

    helper_calls = [
        node
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in HELPERS
    ]
    lane_observations = [
        node
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LaneObservation"
    ]
    assert len(lane_observations) == observations
    assert len(helper_calls) >= 1


def test_a_run_that_placed_no_magnet_is_never_usable() -> None:
    """The non-degeneracy floor the wave-1 rewrite dropped.

    An all-zero moment array is what ``GPMO_ArbVec_backtracking`` was given as
    ``x_init``; it is finite and it is under every cap, so the finiteness and
    upper-bound terms alone report a degenerate run as a success.
    """

    moments = np.zeros((7530, 3), dtype=np.float64)
    assert (
        gpmo_backtracking_outputs_usable(
            moments=moments,
            nonzero_count=0,
            grid_size=7530,
            magnet_cap=5000,
        )
        is False
    )
    assert (
        gpmo_backtracking_outputs_usable(
            moments=moments,
            nonzero_count=5000,
            grid_size=7530,
            magnet_cap=5000,
        )
        is True
    )
    assert (
        gpmo_backtracking_outputs_usable(
            moments=moments,
            nonzero_count=5001,
            grid_size=7530,
            magnet_cap=5000,
        )
        is False
    )


def test_baseline_places_exactly_one_new_site_per_iteration() -> None:
    """``permanent_magnet_optimization.cpp:1280-1325``: equality, not ``<=``.

    Official pm-simple: ``K`` 500 and ``final:nonzero_count`` 500
    (the official reference record ``native-permanent-magnet-simple``).
    """

    moments = np.zeros((14336, 3), dtype=np.float64)
    assert (
        gpmo_baseline_outputs_usable(moments=moments, nonzero_count=500, iterations=500)
        is True
    )
    assert (
        gpmo_baseline_outputs_usable(moments=moments, nonzero_count=499, iterations=500)
        is False
    )
    assert (
        gpmo_baseline_outputs_usable(moments=moments, nonzero_count=0, iterations=500)
        is False
    )


def test_a_non_finite_moment_is_not_usable_even_when_every_count_fits() -> None:
    moments = np.zeros((4, 3), dtype=np.float64)
    moments[0, 0] = np.nan
    assert (
        gpmo_backtracking_outputs_usable(
            moments=moments, nonzero_count=2, grid_size=4, magnet_cap=4
        )
        is False
    )
    assert (
        gpmo_baseline_outputs_usable(moments=moments, nonzero_count=2, iterations=2)
        is False
    )


def test_the_fourth_gpmo_exit_is_reachable_only_with_no_magnet_placed() -> None:
    """``permanent_magnet_optimization.cpp:948-959`` reads an unwritten slot.

    ``num_nonzeros(print_iter)`` is tested after ``print_GPMO`` incremented
    ``print_iter`` and after ``num_nonzeros(print_iter - 1)`` was written, so
    the compared slot is still the zero the array was initialised with. The
    exit therefore fires exactly when the last two recorded counts are ``0``,
    and ``print_iter > 10`` requires eleven records.
    """

    stalled = np.zeros(11, dtype=np.int64)
    assert (
        gpmo_stop_reason(
            nonzero_count=0,
            grid_size=7530,
            magnet_cap=5000,
            recorded_nonzero_counts=stalled,
        )
        == GPMO_NONZERO_COUNT_STALLED
    )
    # Ten records: upstream's ``print_iter > 10`` is not satisfied yet.
    assert (
        gpmo_stop_reason(
            nonzero_count=0,
            grid_size=7530,
            magnet_cap=5000,
            recorded_nonzero_counts=stalled[:10],
        )
        == GPMO_ITERATION_COUNT_COMPLETED
    )
    # The official MUSE history: twelve records, counts rising to the cap.
    official = np.asarray(
        [0, 1, 501, 1001, 1501, 2001, 2501, 3001, 3501, 4001, 4501, 5000],
        dtype=np.int64,
    )
    assert (
        gpmo_stop_reason(
            nonzero_count=5000,
            grid_size=7530,
            magnet_cap=5000,
            recorded_nonzero_counts=official,
        )
        == GPMO_MAGNET_CAP_REACHED
    )


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

    The official reference record ``native-wireframe-gsco-multistep``:
    ``step<k>:iterations`` and ``step<k>:allocated_iteration_budget``.
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


#: Producer module -> the shared usability predicate it must call. The rule is
#: one derivation, not a copy per case: before this ratchet the two baseline
#: cases each inlined their own ``selected <= iterations`` expression and the
#: backtracking helper inlined its own bound, which is how the equality
#: invariant and the non-degeneracy floor were lost one file at a time.
USABILITY_PREDICATES = {
    "_permanent_magnet_arbvec.py": "gpmo_backtracking_outputs_usable",
    "native_permanent_magnet_simple.py": "gpmo_baseline_outputs_usable",
}


@pytest.mark.parametrize(
    ("module", "predicate"),
    sorted(USABILITY_PREDICATES.items()),
)
def test_gpmo_producer_takes_its_usability_rule_from_the_shared_derivation(
    module: str,
    predicate: str,
) -> None:
    tree = ast.parse((CASES / module).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == predicate
    ]
    assert len(calls) == 1
    # Keyword-only, so a caller cannot silently swap the count for the cap.
    assert calls[0].args == []
