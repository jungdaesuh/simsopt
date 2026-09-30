"""Hybrid-signal continuation rules: the step of a hard-feasible iterate while
the surrogate and hard constraint signals disagree.

A signal mismatch needs an evaluator that returns the hybrid quartet
(``hard_signed_constraint_values``, ``hard_violation_values``,
``surrogate_signed_constraint_values``, ``hard_dual_update_values``), so no
other evaluator reaches these rules. :class:`~.policy.DefaultContinuationPolicy`
calls them; the paired no-converge guards stay in that policy.
"""

from __future__ import annotations

from typing import Union

from .continuation import ALMContinue, ALMPostInnerView, ALMRaisePenalty


def signal_mismatch_step(
    view: ALMPostInnerView,
) -> Union[ALMRaisePenalty, ALMContinue]:
    """The step after an inner solve that ends hard-feasible under mismatch
    (which always has a live surrogate shift).

    A stalled mismatch (no progress, or any continuation) raises the penalty,
    unless ``continue_on_signal_mismatch`` repairs it. A repaired or
    progressing mismatch resets the stall count and continues the subproblem
    up to its limit, where it raises the penalty.
    """
    settings = view.settings
    stalled = not view.inner.meaningful_progress or view.continuation_iteration > 0
    # Opt-in repair keeps a stalled mismatch on the bounded continuation path
    # instead of raising the penalty.
    repair = stalled and settings.continue_on_signal_mismatch
    if stalled and not repair:
        # This exit keeps the stall count the step came with.
        return ALMRaisePenalty(
            action="signal_mismatch_penalty_increase",
            max_outer_termination="max_outer_after_signal_mismatch_penalty_increase",
            feasible_stall_count=view.feasible_stall_count,
        )
    if view.continuation_iteration == settings.max_subproblem_continuations:
        return ALMRaisePenalty(
            action="signal_mismatch_subproblem_limit_penalty_increase",
            max_outer_termination="max_outer",
            feasible_stall_count=0,
            subproblem_limit_reason="max_subproblem_continuations",
            signal_mismatch_repair=repair,
        )
    return ALMContinue(
        trust_radius=view.trust_radius,
        update_stationarity_tol=view.inner.measured.update_stationarity_tol,
        feasible_stall_count=0,
        signal_mismatch_repair=repair,
    )
