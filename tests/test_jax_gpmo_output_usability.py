"""The GPMO usability predicates of ``simsopt_jax_adapters.examples.gpmo_rules``.

Upstream GPMO returns arrays and prints its stop line, so a run's outputs are
judged usable from the returned moments and magnet count alone: finite moments,
and a magnet count the C++ algorithm can actually produce
(``permanent_magnet_optimization.cpp`` at upstream ``9e027eac3``).
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_backtracking_outputs_usable,
    gpmo_baseline_outputs_usable,
)


def test_a_run_that_placed_no_magnet_is_never_usable() -> None:
    """The non-degeneracy floor.

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
    """``permanent_magnet_optimization.cpp`` ``GPMO_baseline``: equality, not ``<=``.

    Official upstream ``permanent_magnet_simple.py``: ``K`` 500 and a final
    nonzero count of 500.
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
