"""The official GPMO history rules, pinned against the official records.

Authority: ``simsopt/solve/permanent_magnet_optimization.py`` and
``examples/2_Intermediate/permanent_magnet_MUSE.py`` at upstream
``9e027eac38028d57aa23777be52a781aa860e347``. Every official number is read
from the tracked fixture ``examples/jax/parity/official_reference`` -- the
canonical record for a shipped-scale run and the ``ci`` variant for upstream's
own reduced configuration -- so this file runs on a clean checkout.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
import pytest
from examples.jax.parity.official_reference import load_official_reference
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_backtracking_outputs_usable,
    gpmo_history_period,
    recorded_history_length,
    select_minimum_objective_snapshot,
)

OFFICIAL_MUSE = load_official_reference("native-permanent-magnet-muse")
OFFICIAL_PM4STELL_CI = load_official_reference(
    "native-permanent-magnet-pm4stell",
    variant="ci",
)
OFFICIAL_MUSE_CONFIGURATION = OFFICIAL_MUSE.structure("configuration")
OFFICIAL_PM4STELL_CI_CONFIGURATION = OFFICIAL_PM4STELL_CI.structure("configuration")
assert isinstance(OFFICIAL_MUSE_CONFIGURATION, dict)
assert isinstance(OFFICIAL_PM4STELL_CI_CONFIGURATION, dict)
#: ``history:R2`` of the official MUSE run (K=10000, nhistory=20, cap 5000).
OFFICIAL_MUSE_R2 = OFFICIAL_MUSE.array("history:R2")
#: ``history:R2`` of the official PM4Stell CI run (K=100, nhistory=10, cap 20).
#: Non-monotone: the minimum is row 1, the endpoint is row 3.
OFFICIAL_PM4STELL_CI_R2 = OFFICIAL_PM4STELL_CI.array("history:R2")


def _moment_history(objective: np.ndarray) -> np.ndarray:
    """One distinguishable ``(ndipoles, 3, rows)`` snapshot per recorded row."""

    rows = objective.size
    history = np.zeros((4, 3, rows), dtype=np.float64)
    for row in range(rows):
        history[:, :, row] = float(row + 1)
    return history


def test_the_official_period_is_upstream_int_division() -> None:
    """``k % int(K / nhistory)`` (``permanent_magnet_optimization.cpp:942``)."""

    assert (
        gpmo_history_period(
            iterations=OFFICIAL_MUSE_CONFIGURATION["K"],
            history_count=OFFICIAL_MUSE_CONFIGURATION["nhistory"],
        )
        == 500
    )
    assert (
        gpmo_history_period(
            iterations=OFFICIAL_PM4STELL_CI_CONFIGURATION["K"],
            history_count=OFFICIAL_PM4STELL_CI_CONFIGURATION["nhistory"],
        )
        == 10
    )
    # ``nhistory == K`` is the boundary upstream allows: one row per iteration.
    assert gpmo_history_period(iterations=20, history_count=20) == 1


def test_nhistory_above_k_is_refused_as_upstream_refuses_it() -> None:
    """``GPMO`` raises for ``nhistory > K``; the C++ would compute ``k % 0``."""

    with pytest.raises(ValueError, match="nhistory"):
        gpmo_history_period(iterations=20, history_count=21)


def test_the_recorded_length_is_the_filled_prefix() -> None:
    """``errors = algorithm_history[algorithm_history != 0]``.

    The C++ allocates ``nhistory + 2`` slots and fills a prefix; the official
    MUSE run filled 12 of 22 (``history:length`` 12).
    """

    slots = int(OFFICIAL_MUSE_CONFIGURATION["nhistory"]) + 2
    padded = np.zeros(slots, dtype=np.float64)
    padded[: OFFICIAL_MUSE_R2.size] = OFFICIAL_MUSE_R2
    assert recorded_history_length(padded) == OFFICIAL_MUSE.scalar("history:length")


def test_a_hole_in_the_recorded_history_is_refused() -> None:
    holed = np.asarray([1.0, 0.0, 3.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="filled prefix"):
        recorded_history_length(holed)


def test_official_muse_history_selects_its_endpoint() -> None:
    """Monotone ``R2`` -> the last recorded row, which is the run endpoint.

    The official capture's ``final:objective_half_sum_squares``
    (5.80033065866346e-05) is that row, so upstream's ``pm_opt.m`` after the
    selection is the endpoint at shipped scale.
    """

    selection = select_minimum_objective_snapshot(
        np.asarray(OFFICIAL_MUSE_R2, dtype=np.float64),
        _moment_history(OFFICIAL_MUSE_R2),
    )
    official_length = int(OFFICIAL_MUSE.scalar("history:length"))
    assert selection.length == official_length
    assert selection.index == official_length - 1
    assert np.array_equal(
        selection.moments,
        np.full((4, 3), float(official_length)),
    )


def test_a_non_monotone_history_does_not_select_the_endpoint() -> None:
    """Official PM4Stell CI rises after row 1, so argmin is not the endpoint.

    This is the shape that separates "post metrics from the selected snapshot"
    from "post metrics from the GPMO endpoint": the two rows differ.
    """

    selection = select_minimum_objective_snapshot(
        np.asarray(OFFICIAL_PM4STELL_CI_R2, dtype=np.float64),
        _moment_history(OFFICIAL_PM4STELL_CI_R2),
    )
    official_length = int(OFFICIAL_PM4STELL_CI.scalar("history:length"))
    assert selection.length == official_length
    assert selection.index == 1
    assert selection.index != official_length - 1
    assert np.array_equal(selection.moments, np.full((4, 3), 2.0))


def test_the_selection_is_invariant_under_the_half_factor() -> None:
    """The JAX kernel records ``sum(r * r)``; upstream records half of it."""

    objective = np.asarray(OFFICIAL_PM4STELL_CI_R2, dtype=np.float64)
    history = _moment_history(OFFICIAL_PM4STELL_CI_R2)
    assert (
        select_minimum_objective_snapshot(2.0 * objective, history).index
        == select_minimum_objective_snapshot(objective, history).index
    )


def test_a_short_moment_history_is_refused() -> None:
    objective = np.asarray(OFFICIAL_PM4STELL_CI_R2, dtype=np.float64)
    with pytest.raises(ValueError, match="ndipoles"):
        select_minimum_objective_snapshot(objective, np.zeros((4, 3, 2)))


def test_a_nan_moment_row_is_invisible_to_the_nonzero_selection_filter() -> None:
    """The mechanism the MUSE example's success gate has to survive.

    ``selected = np.flatnonzero(np.linalg.norm(moments, axis=1) > 0.0)`` drops a
    NaN row, because ``norm([nan, 0, 0])`` is ``nan`` and ``nan > 0.0`` is
    False. A finiteness check applied to ``moments[selected]`` therefore never
    sees the NaN, and the example published ``status='ok'`` with a NaN in the
    solution. The rule checks the FULL array instead.
    """

    moments = np.zeros((3, 3), dtype=np.float64)
    moments[0] = (np.nan, 0.0, 0.0)
    moments[1] = (1.0, 0.0, 0.0)
    selected = np.flatnonzero(np.linalg.norm(moments, axis=1) > 0.0)
    assert selected.tolist() == [1]
    assert bool(np.all(np.isfinite(moments[selected]))) is True
    assert (
        gpmo_backtracking_outputs_usable(
            moments=moments,
            nonzero_count=int(selected.size),
            grid_size=3,
            magnet_cap=20,
        )
        is False
    )
