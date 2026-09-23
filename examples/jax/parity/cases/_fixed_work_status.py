"""Terminal status of the parity lanes whose provider never reports convergence.

``examples.jax.parity.terminal_status`` folds the *optimizer* emitter tables
(SciPy L-BFGS-B, the private on-device L-BFGS-B) into the arbiter vocabulary.
The greedy upstream solvers mirrored here have no such emitter: they return
arrays and print their stop line, so a lane must derive the stop reason from
the returned data, identically in every lane. This module is that single
derivation.

Two families, because upstream stops them for different reasons:

* **GPMO** (``simsoptpp::GPMO_baseline`` / ``GPMO_ArbVec_backtracking``) places
  one magnet per iteration until the nonzero count reaches the grid size or
  ``max_nMagnets``, otherwise it completes its ``K`` iterations
  (``permanent_magnet_optimization.cpp:963-980`` at upstream
  ``9e027eac38028d57aa23777be52a781aa860e347``). None of the three is
  convergence, so the normalized category is ``not_applicable`` and the
  arbiter then requires null ``nit``/``nfev``/``njev``: the configured ``K``
  is not a measured count. The relax-and-split continuation of
  ``permanent_magnet_QA.py`` is the same kind of fact -- a fixed amount of work
  that ran to completion -- and shares the category.
  Upstream's fourth GPMO exit ("number of nonzero dipoles unchanged over three
  backtracking cycles", ``permanent_magnet_optimization.cpp:948-959``) sits
  inside ``if (verbose && ...)``. The official scripts enable ``verbose``
  (``initialize_default_kwargs`` sets it, ``permanent_magnet_helper_functions``
  ``:250``) and so do the lanes, so the exit is in the vocabulary and is
  derived from the recorded nonzero counts by ``gpmo_stop_reason`` -- for the
  providers that have it. A lane is never labelled with a stop its own solver
  cannot take, so the counts are supplied only by the native
  ``GPMO_ArbVec_backtracking``; the JAX kernel has no such branch and runs to
  ``K`` in the one shape the exit can take (no magnet placed), which is a
  ``failed`` lane in both.

* **GSCO** (``simsoptpp::GSCO``) has three stop conditions
  (``src/simsoptpp/wireframe_optimization.cpp:264-279`` **of this branch**, the
  binary the lanes link): no eligible loop, the best loop would undo the
  previous one, or the last allowed iteration. The first two are the solver
  deciding it is done -- ``converged``; the third is the declared budget --
  ``budget_exhausted``. A run that adopted no loop at all moved nothing and is
  ``failed``, whichever of the first two conditions ended it. That family now
  lives in the INSTALLED package as
  ``simsopt_jax.examples.solver_terminal_status``, because the three GSCO
  examples run as standalone scripts with no repository root on ``sys.path``
  and cannot import this package; it is re-exported below so that one
  derivation serves the examples and the parity lanes alike. The undo-pair
  rule there also states what a sampled history costs: it cannot show the
  pair, so the label falls back to the coarser of the two ``converged``
  reasons.

Pure in its own computation: no I/O, no globals. The re-export pulls
``simsopt_jax.examples``, which imports JAX -- nothing new for a parity lane,
which already imports it (``_wireframe_gsco.py:18``).
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from simsopt_jax.examples.solver_terminal_status import (  # re-exported: one derivation
    GSCO_MAXIMUM_ITERATION_REACHED,
    GSCO_MINIMUM_OBJECTIVE_REACHED,
    GSCO_MULTISTEP_NO_ACCEPTED_UPDATE,
    GSCO_MULTISTEP_STABLE_CURRENTS,
    GSCO_MULTISTEP_STAGE_BUDGET_REACHED,
    GSCO_NO_ACCEPTED_UPDATE,
    GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP,
    LaneTerminalLabel,
    gsco_multistep_terminal_label,
    gsco_terminal_label,
)

__all__ = [
    "FIXED_WORK_NORMALIZED_STATUS",
    "GPMO_GRID_FULL",
    "GPMO_ITERATION_COUNT_COMPLETED",
    "GPMO_MAGNET_CAP_REACHED",
    "GPMO_NONZERO_COUNT_STALLED",
    "GSCO_MAXIMUM_ITERATION_REACHED",
    "GSCO_MINIMUM_OBJECTIVE_REACHED",
    "GSCO_MULTISTEP_NO_ACCEPTED_UPDATE",
    "GSCO_MULTISTEP_STABLE_CURRENTS",
    "GSCO_MULTISTEP_STAGE_BUDGET_REACHED",
    "GSCO_NO_ACCEPTED_UPDATE",
    "GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP",
    "RELAX_AND_SPLIT_CONTINUATION_COMPLETED",
    "LaneTerminalLabel",
    "fixed_work_label",
    "gpmo_backtracking_stall",
    "gpmo_stop_reason",
    "gsco_multistep_terminal_label",
    "gsco_terminal_label",
]

#: A provider that reports no convergence status at all. The arbiter leaves
#: ``optimizer_outcome`` out of the lane applicability match for this category
#: and demands null optimizer counters.
FIXED_WORK_NORMALIZED_STATUS: Final = "not_applicable"

#: ``num_nonzero >= N``: every dipole in the grid carries a magnet.
GPMO_GRID_FULL: Final = "grid_full"
#: ``num_nonzero >= max_nMagnets``: the magnet cap was reached.
GPMO_MAGNET_CAP_REACHED: Final = "magnet_cap_reached"
#: The ``K``-iteration loop ran out without either limit being reached.
GPMO_ITERATION_COUNT_COMPLETED: Final = "iteration_count_completed"
#: Upstream's fourth exit: "number of nonzero dipoles unchanged over three
#: backtracking cycles" (``permanent_magnet_optimization.cpp:948-959``).
GPMO_NONZERO_COUNT_STALLED: Final = "nonzero_count_unchanged_over_backtracking_cycles"

#: The fixed relax-and-split continuation ran every configured stage.
RELAX_AND_SPLIT_CONTINUATION_COMPLETED: Final = "relax_and_split_continuation_completed"


def gpmo_backtracking_stall(recorded_nonzero_counts: NDArray[np.int64]) -> bool:
    """Mirror upstream's "nonzero count unchanged" exit on the recorded counts.

    ``permanent_magnet_optimization.cpp:948-959``: inside the ``verbose``
    record, upstream writes ``num_nonzeros(print_iter - 1) = num_nonzero``
    *after* ``print_GPMO`` has already incremented ``print_iter``, and then
    tests ``num_nonzeros(print_iter) == num_nonzeros(print_iter - 1) ==
    num_nonzeros(print_iter - 2)`` with ``print_iter > 10``. Slot
    ``print_iter`` has never been written at that point and the array is
    zero-initialised, so the exit fires exactly when the current and the
    previous nonzero counts are both ``0`` -- a run that is placing no magnets
    at all.

    ``recorded_nonzero_counts`` must be the *written* prefix in record order,
    so ``print_iter`` is its length and the two tested slots are its last two
    entries. Written is not the same as recorded: ``num_nonzeros(0)`` is
    written before the loop (``:783``) and every verbose record writes one more
    slot, but the magnet-limit / grid-full exit calls ``print_GPMO`` and writes
    **no** count (``:964-985``), so after that exit the counts are one entry
    shorter than the objective history. ``gpmo_stop_reason`` is what guarantees
    the precondition: it reports the two limits first, and a run that reached a
    limit cannot also have left through this exit, which breaks immediately and
    requires a zero count -- below every cap. Measured on the bounded MUSE
    native lane: six recorded objective rows against the five written counts
    ``[0, 1, 6, 11, 16]``.
    """

    counts = np.asarray(recorded_nonzero_counts, dtype=np.int64).reshape(-1)
    if counts.size < 11:
        return False
    return bool(counts[-1] == 0 and counts[-2] == 0)


def gpmo_stop_reason(
    *,
    nonzero_count: int,
    grid_size: int | None,
    magnet_cap: int | None,
    recorded_nonzero_counts: NDArray[np.int64] | None = None,
) -> str:
    """Name the GPMO stop condition the returned magnet count implies.

    Mirrors ``permanent_magnet_optimization.cpp:942-985``. The three exits are
    mutually exclusive over a finished run, so they are reported in the order
    that the returned data determines them:

    * the two magnet limits first (a run that fills the grid reports
      ``grid_full`` even when it also reaches the cap, as the C++ block does);
    * the ``verbose`` stall exit only then, because it cannot have ended a run
      whose final count reached a limit: it ``break``s at the record that fires
      it, and it fires only on a zero count. Reporting the limits first is also
      what makes ``recorded_nonzero_counts`` the written prefix the predicate
      needs -- the limit exit appends an objective row and no count.

    Both limits are ``None`` for ``GPMO_baseline``
    (``permanent_magnet_optimization.cpp:1237-1326``), whose loop has no early
    exit of any kind, so that solver always completes its iteration count.
    ``recorded_nonzero_counts`` is ``None`` for any provider that does not
    implement the stall exit: ``GPMO_baseline`` for the same reason, and the
    JAX arbitrary-vector kernel, which has no such branch at all.
    """

    if grid_size is not None and nonzero_count >= grid_size:
        return GPMO_GRID_FULL
    if magnet_cap is not None and nonzero_count >= magnet_cap:
        return GPMO_MAGNET_CAP_REACHED
    if recorded_nonzero_counts is not None and gpmo_backtracking_stall(
        recorded_nonzero_counts
    ):
        return GPMO_NONZERO_COUNT_STALLED
    return GPMO_ITERATION_COUNT_COMPLETED


def fixed_work_label(*, raw_status: str, outputs_usable: bool) -> LaneTerminalLabel:
    """Label a lane whose provider reports finished work rather than a status.

    ``success`` means "returned the finite required outputs under the upstream
    stop rule"; it is never a claim of convergence, and the normalized category
    stays ``not_applicable`` so no aggregate can count the lane as converged.
    A lane that did not return usable outputs is ``failed``.
    """

    if not outputs_usable:
        return LaneTerminalLabel(
            normalized_status="failed",
            raw_status=raw_status,
            success=False,
        )
    return LaneTerminalLabel(
        normalized_status=FIXED_WORK_NORMALIZED_STATUS,
        raw_status=raw_status,
        success=True,
    )
