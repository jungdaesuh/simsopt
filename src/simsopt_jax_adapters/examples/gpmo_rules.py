"""Upstream GPMO rules shared by the permanent-magnet examples.

Official source: ``simsopt/solve/permanent_magnet_optimization.py`` and
``src/simsoptpp/permanent_magnet_optimization.cpp`` at upstream
``9e027eac38028d57aa23777be52a781aa860e347``. Only the MUSE script consumes the
recorded history (``examples/2_Intermediate/permanent_magnet_MUSE.py:184-185``);
the period, the filled-prefix rule and the two "did this solver do its work"
predicates apply to every GPMO caller alike, so they live here once instead
of once per caller.

Pure: numpy only, no I/O, no JAX, no globals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GPMOHistorySelection:
    """The recorded snapshot the official MUSE selection rule picks."""

    #: Row index inside the provider's recorded history.
    index: int
    #: Number of rows the provider actually recorded.
    length: int
    #: ``(ndipoles, 3)`` physical moments of the selected row.
    moments: NDArray[np.float64]


def gpmo_history_period(*, iterations: int, history_count: int) -> int:
    """Upstream's ``int(K / nhistory)`` record period.

    ``GPMO`` refuses ``nhistory > K`` (``permanent_magnet_optimization.py:395``)
    because the C++ would then compute ``k % 0``; the same refusal is the reason
    this helper exists rather than an inline division.
    """

    if history_count > iterations:
        raise ValueError(
            "nhistory must be less than or equal to K for the GPMO algorithm; "
            f"got nhistory={history_count}, K={iterations}"
        )
    return iterations // history_count


def recorded_history_length(objective_history: NDArray[np.float64]) -> int:
    """Number of rows the provider filled, upstream's ``errors != 0`` prefix.

    ``GPMO`` publishes ``errors = algorithm_history[algorithm_history != 0]``
    (``solve/permanent_magnet_optimization.py:473``) and then indexes the
    *unfiltered* ``m_history`` with ``argmin(errors)``; that is only consistent
    because the filled rows are a prefix. The prefix length is reproduced here,
    not the boolean filter, so the two stay consistent by construction.
    """

    filled = np.flatnonzero(
        np.asarray(objective_history, dtype=np.float64).reshape(-1) != 0.0
    )
    if filled.size == 0:
        raise ValueError("the provider recorded no history row")
    length = int(filled[-1]) + 1
    if filled.size != length:
        raise ValueError("the recorded objective history is not a filled prefix")
    return length


def select_minimum_objective_snapshot(
    objective_history: NDArray[np.float64],
    moment_history: NDArray[np.float64],
) -> GPMOHistorySelection:
    """Official MUSE rule: the recorded snapshot with the smallest objective.

    ``permanent_magnet_MUSE.py:184-185`` at upstream ``9e027eac3``:
    ``min_ind = np.argmin(R2_history); pm_opt.m = np.ravel(m_history[:, :,
    min_ind])``, evaluated *before* the printed volume, the dipole squared flux
    on the plotting surface and the total magnet volume. ``moment_history`` is
    indexed ``(ndipoles, 3, rows)``, the layout both providers hand over after
    the ``mmax`` rescale.
    """

    length = recorded_history_length(objective_history)
    objective = np.asarray(objective_history, dtype=np.float64).reshape(-1)[:length]
    moments = np.asarray(moment_history, dtype=np.float64)
    if moments.ndim != 3 or moments.shape[2] < length:
        raise ValueError("moment history must be (ndipoles, 3, rows) with every row")
    index = int(np.argmin(objective))
    return GPMOHistorySelection(
        index=index,
        length=length,
        moments=np.ascontiguousarray(moments[:, :, index]),
    )


def gpmo_backtracking_outputs_usable(
    *,
    moments: NDArray[np.float64],
    nonzero_count: int,
    grid_size: int,
    magnet_cap: int,
) -> bool:
    """Finite moments and a non-degenerate magnet count under both limits.

    The upper bounds are the two limits the C++ stops on
    (``permanent_magnet_optimization.cpp:965``). The lower bound is the
    completion policy's non-degeneracy requirement: a run that placed no magnet
    returned the ``x_init`` it was given, so "the solver stopped itself" says
    nothing about it. Upstream's own runs place magnets at every scale
    (official MUSE 5000, official PM4Stell 1000, official CI both 20), so the
    floor never demotes an official run.
    """

    return bool(
        np.all(np.isfinite(moments))
        and 0 < nonzero_count <= magnet_cap
        and nonzero_count <= grid_size
    )


def gpmo_baseline_outputs_usable(
    *,
    moments: NDArray[np.float64],
    nonzero_count: int,
    iterations: int,
) -> bool:
    """Finite moments and exactly one distinct magnet per iteration.

    ``GPMO_baseline`` (``permanent_magnet_optimization.cpp:1280-1325``) selects
    a site, writes one component of it and immediately blanks all three of its
    components in ``R2s`` and in ``Gamma_complement``, so the site can never be
    selected again. After ``K`` iterations exactly ``K`` rows are nonzero:
    equality is the invariant the solver guarantees, and a short count means
    the lane did not run this algorithm.
    """

    return bool(np.all(np.isfinite(moments)) and nonzero_count == iterations)
