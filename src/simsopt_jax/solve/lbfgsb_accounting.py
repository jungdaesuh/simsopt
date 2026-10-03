"""Evaluation accounting of a SciPy L-BFGS-B solve, checked against its trace.

What the pinned implementation allows (SciPy 1.17.1, git 527eb7fd; ``__lbfgsb.c`` and
``_lbfgsb_py.py``):

* One line search evaluates at most ``maxls`` trials: ``lnsrlb`` sets
  ``iback = ifun - 1`` when it requests a trial (``__lbfgsb.c:2659-2661``) and
  ``mainlb`` tests ``iback >= maxls`` before returning that request to the driver
  (``:917-918``), so the ``(maxls + 1)``-th request is never evaluated.
* A failed search with a non-empty memory clears the memory and restarts the SAME
  iteration (``:938-950``), with ``lnsrlb``'s counters reset (``:2635-2636``); a failure
  with an empty memory ends ABNORMAL (``:924-936``) with no NEW_X. One iteration
  therefore evaluates at most ``2 * maxls`` trials, and the first iteration of a call,
  whose memory is empty, at most ``maxls``.
* The wrapper tests ``nfev > maxfun`` only at NEW_X (``_lbfgsb_py.py:466-489``), so
  every boundary before the final iteration is ``<= maxfun`` and a call ends at most
  ``maxfun + 2 * maxls``. x0's evaluation precedes iteration 1.
* The restart route counts true evaluations only: a restarted call's served start is not
  one (``dispatch._minimize_lbfgsb_with_restarts``); its chain stops no later than one
  call would.

The trace boundaries are the cumulative true-evaluation counts at each NEW_X, after x0's
evaluation (count 1). A final segment of evaluations after the last NEW_X (an ABNORMAL
tail) is ``nfev`` minus the last boundary. These are necessary bounds from the source,
not a replay: a history that passes them is consistent with the implementation, not
proven to be the one it produced.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import LbfgsbRestartReason, OptimizerResult
from .scipy.contracts import ScipyLBFGSBOptions


__all__ = ["lbfgsb_evaluation_accounting_defects", "lbfgsb_result_accounting_defects"]


def lbfgsb_evaluation_accounting_defects(
    *,
    trace_nfev: Sequence[int | None],
    call_start_iterations: Sequence[int],
    nit: int,
    nfev: int,
    true_evaluations: int,
    maxiter: int,
    maxfun: int,
    maxls: int,
) -> tuple[str, ...]:
    """Every way the counts break the pinned accounting; empty means valid.

    ``trace_nfev`` holds each accepted iteration's cumulative true-evaluation count;
    every entry must be populated. ``call_start_iterations`` are the (1-based)
    iterations that begin a SciPy call with an empty memory: always 1, plus the
    iteration after each restart. ``true_evaluations`` is the lane-level count of
    objective calls (below SciPy's memo).
    """
    missing = [index for index, count in enumerate(trace_nfev) if count is None]
    if missing:
        return (
            f"trace entries {missing} have no nfev: the accounting cannot be checked",
        )
    counts = [int(count) for count in trace_nfev if count is not None]
    defects: list[str] = []
    if nfev != true_evaluations:
        defects.append(
            f"nfev {nfev} differs from the {true_evaluations} recorded true evaluations"
        )
    if nit != len(counts):
        defects.append(
            f"nit {nit} differs from the {len(counts)} accepted-iterate entries"
        )
    if nit > maxiter:
        defects.append(f"nit {nit} exceeds maxiter {maxiter}")
    if nfev < 1:
        return (*defects, "nfev is below 1, but x0 is always evaluated")
    boundaries = [1, *counts]
    segments = [later - earlier for earlier, later in zip(boundaries, boundaries[1:])]
    if any(segment < 0 for segment in segments) or boundaries[-1] > nfev:
        return (
            *defects,
            f"trace counts {counts} are not nondecreasing from 1 up to nfev {nfev}",
        )
    tail = nfev - boundaries[-1]
    # The final segment is the tail after the last NEW_X if it holds evaluations, else
    # the last iteration.
    pre_final = boundaries[-1] if tail > 0 else boundaries[-2] if counts else 0
    if pre_final > maxfun:
        defects.append(
            f"{pre_final} evaluations before the final iteration exceed maxfun {maxfun}"
        )
    fresh_calls = set(call_start_iterations)
    # The tail belongs to the unfinished iteration after the last NEW_X.
    for iteration, segment in enumerate([*segments, tail], start=1):
        ceiling, rule = (
            (maxls, "maxls (a call's first iteration)")
            if iteration in fresh_calls
            else (2 * maxls, "2 * maxls")
        )
        if segment > ceiling:
            where = (
                "after the last NEW_X"
                if iteration > len(segments)
                else f"in iteration {iteration}"
            )
            defects.append(f"{segment} evaluations {where} exceed {rule} = {ceiling}")
    if nfev > maxfun + 2 * maxls:
        defects.append(f"nfev {nfev} exceeds maxfun + 2 * maxls = {maxfun + 2 * maxls}")
    return tuple(defects)


def lbfgsb_result_accounting_defects(
    result: OptimizerResult, *, true_evaluations: int
) -> tuple[str, ...]:
    """The accounting of a restart-route result against its ``options_used`` budgets."""
    options = result.options_used
    if (
        not isinstance(options, ScipyLBFGSBOptions)
        or not options.restart_after_nonwolfe_stop
    ):
        return (
            "only the SciPy L-BFGS-B restart route records the accepted-iterate trace",
        )
    if result.optimizer_state_trace is None:
        return ("the result has no accepted-iterate trace",)
    restarted = [
        event.iteration + 1
        for event in result.restart_log
        if event.reason is LbfgsbRestartReason.RESTARTED
    ]
    return lbfgsb_evaluation_accounting_defects(
        trace_nfev=[entry.nfev for entry in result.optimizer_state_trace],
        call_start_iterations=(1, *restarted),
        nit=result.nit,
        nfev=result.nfev,
        true_evaluations=true_evaluations,
        maxiter=options.maxiter,
        maxfun=options.maxfun,
        maxls=options.maxls,
    )
