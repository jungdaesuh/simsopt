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

from .continuation import ALMContinue, ALMPostInnerView, ALMRaisePenalty, ALMStop


def signal_mismatch_step(
    view: ALMPostInnerView,
) -> Union[ALMStop, ALMRaisePenalty, ALMContinue]:
    """The step after an inner solve that ends hard-feasible under mismatch.

    A stalled mismatch (no progress, or any continuation) stops when the
    surrogate shift is zero and raises the penalty otherwise, unless
    ``continue_on_signal_mismatch`` repairs a live shift. A repaired or
    progressing mismatch resets the stall count and continues the subproblem
    up to its limit, where it raises the penalty.
    """
    settings = view.settings
    routing_state = view.inner.measured.routing_state
    stalled = not view.inner.meaningful_progress or view.continuation_iteration > 0
    # Opt-in repair keeps a stalled mismatch with a live surrogate shift on
    # the bounded continuation path instead of raising the penalty.
    repair = (
        stalled
        and settings.continue_on_signal_mismatch
        and not routing_state.surrogate_positive_shift_zero
    )
    if stalled and not repair:
        # These two exits keep the stall count the step came with.
        if routing_state.surrogate_positive_shift_zero:
            return ALMStop(
                action="signal_mismatch_stall",
                termination_reason="signal_mismatch_stall",
                message_prefix=(
                    "ALM stopped after hard-feasible and surrogate-active "
                    "signals repeated without corrective progress"
                ),
                feasible_stall_count=view.feasible_stall_count,
            )
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
