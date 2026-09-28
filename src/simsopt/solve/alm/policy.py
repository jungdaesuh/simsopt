"""Continuation policy: the decisions of one ALM continuation step.

:func:`~.control.minimize_alm` measures, solves and executes; an
:class:`ALMContinuationPolicy` decides, from a read-only view
(:mod:`.continuation`), how to run each inner attempt, whether to retry a
stalled trial, and how the step ends. :class:`DefaultContinuationPolicy` is
the library default. Policies are stateless: what a rule carries between
steps comes in through the view and goes back in the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Protocol, Union

import numpy as np

from . import hybrid
from .continuation import (
    ALMContinue,
    ALMConverge,
    ALMDualUpdateStep,
    ALMHold,
    ALMInnerPlan,
    ALMInnerPlanView,
    ALMIterateMeasurement,
    ALMPostInnerView,
    ALMPreInnerView,
    ALMRaisePenalty,
    ALMStalledTrialView,
    ALMStepDecision,
    ALMStop,
)
from .core import (
    ALMSettings,
    _complementarity_residual,
    _finite_alm_integer,
    _finite_alm_value,
)


class ALMContinuationPolicy(Protocol):
    """The decisions of one continuation step that differ between users.

    A policy has the last word on convergence: the loop executes its
    ``ALMConverge`` as given and adds no veto of its own. The success
    guarantees (a KKT point at the shifted multipliers ``max(0, λ + ρg)``:
    hard feasibility and complementarity at ``feasibility_tol``, stationarity
    at ``stationarity_tol``; no hybrid signal mismatch, no binding multiplier
    cap) hold for :class:`DefaultContinuationPolicy` and for policies that
    keep its vetoes, e.g. by delegating their convergence decisions to it.
    """

    def inner_plan(self, view: ALMInnerPlanView) -> ALMInnerPlan:
        """L-BFGS-B options and profile name for one inner attempt."""

    def retry_stalled_trial(self, view: ALMStalledTrialView) -> Optional[float]:
        """A trust radius to retry a stalled trial with; None keeps the start
        iterate and forces a penalty cycle."""

    def before_inner(self, view: ALMPreInnerView) -> Optional[ALMConverge]:
        """Convergence at the start iterate; None runs the inner solve."""

    def after_inner(self, view: ALMPostInnerView) -> ALMStepDecision:
        """How the step ends, given the judged inner solve."""


@dataclass(frozen=True)
class DefaultContinuationPolicy:
    """The library's continuation rules: the ALM dual update and penalty raise
    with the ALGENCAN safeguard, plus the stall, plateau and subproblem-limit
    safeguards of an inexact inner solve. It neither caps a boxed inner
    solve's work nor grows the trust radius between subproblems."""

    def inner_plan(self, view: ALMInnerPlanView) -> ALMInnerPlan:
        """The caller's options with the staged ``gtol``; no inner-work caps."""
        return ALMInnerPlan(
            profile="unbounded" if view.attempt_radius is None else "boxed",
            options=MappingProxyType(
                _staged_inner_options(view.inner_options, view.update_stationarity_tol)
            ),
        )

    def retry_stalled_trial(self, view: ALMStalledTrialView) -> Optional[float]:
        return None

    def before_inner(self, view: ALMPreInnerView) -> Optional[ALMConverge]:
        """Converge at a start iterate that already meets the KKT tolerances,
        unless hybrid signals disagree, the hard constraints are all inactive
        (that arm needs an inner solve) or the multiplier cap binds."""
        start = view.start
        settings = view.settings
        if (
            _kkt_point(start, settings)
            and not _constraints_inactive_candidate(start, settings.feasibility_tol)
            and not start.signal_mismatch_active
            # Cap-binding multipliers mean the prior dual update was
            # clamped; the KKT residual is held small by the cap, not by
            # convergence. Same guard as both post-inner converged arms; the
            # outer loop continues (a later dual update may unclamp).
            and not view.last_cap_binding_active
        ):
            return ALMConverge(
                action="converged",
                termination_reason="converged",
                message=(
                    "ALM converged: "
                    f"max_violation={start.max_feasibility_violation:.3e}, "
                    f"stationarity={start.stationarity_norm:.3e}"
                ),
                feasible_stall_count=view.feasible_stall_count,
            )
        return None

    def after_inner(self, view: ALMPostInnerView) -> ALMStepDecision:
        """How the step ends, in this order: converged, converged or stalled
        with inactive hard constraints, dual update, the infeasible-stall
        penalty cycle, a hard-feasible signal mismatch (:mod:`.hybrid`), a
        hard-feasible step whose dual gate is unmet, and otherwise a penalty
        hold or raise by the ALGENCAN sufficient-decrease test."""
        measured = view.inner.measured
        settings = view.settings
        constraints_inactive = _constraints_inactive_candidate(
            measured, settings.feasibility_tol
        )
        if (
            _kkt_point(measured, settings)
            and not constraints_inactive
            and not measured.signal_mismatch_active
            # A clamped dual update holds the KKT residual small.
            and not view.last_cap_binding_active
        ):
            return ALMConverge(
                action="converged",
                termination_reason="converged",
                message=(
                    "ALM converged: "
                    f"max_violation={measured.max_feasibility_violation:.3e}, "
                    f"stationarity={measured.stationarity_norm:.3e}"
                ),
                feasible_stall_count=view.feasible_stall_count,
            )
        if constraints_inactive:
            # The same cap guard applies to the constraints-inactive arm.
            if _kkt_point(measured, settings) and not view.last_cap_binding_active:
                return ALMConverge(
                    action="constraints_inactive_converged",
                    termination_reason="constraints_inactive_converged",
                    message=(
                        "ALM converged with inactive hard constraints: "
                        "max_violation="
                        f"{measured.routing_state.hard_max_violation:.3e}, "
                        f"stationarity={measured.stationarity_norm:.3e}"
                    ),
                    feasible_stall_count=view.feasible_stall_count,
                )
            if not view.inner.meaningful_progress and view.continuation_iteration > 0:
                return ALMStop(
                    action="constraints_inactive_stall",
                    termination_reason="constraints_inactive_stall",
                    message_prefix=(
                        "ALM stopped after hard constraints became inactive "
                        "without further stationarity progress"
                    ),
                    feasible_stall_count=view.feasible_stall_count,
                )
        if _dual_update_gate_satisfied(
            max_feasibility_violation=measured.max_feasibility_violation,
            hard_max_violation=measured.routing_state.hard_max_violation,
            stationarity_norm=measured.stationarity_norm,
            kkt_stationarity_norm=measured.kkt_stationarity_norm,
            update_feasibility_tol=measured.update_feasibility_tol,
            update_stationarity_tol=measured.update_stationarity_tol,
        ):
            # Raise the penalty with the update after a forced cycle or when
            # the hard violation stays above the gate without shrinking.
            penalty_required = view.inner.infeasible_stall or (
                measured.routing_state.hard_max_violation
                > measured.effective_feasibility_tol
                and view.inner.feasibility_delta
                <= view.inner.feasibility_delta_tolerance
            )
            return ALMDualUpdateStep(
                penalty_reason=(
                    view.inner.infeasible_stall_reason
                    or "hard_feasibility_not_improved_after_dual_update"
                )
                if penalty_required
                else None,
                feasible_stall_count=view.feasible_stall_count,
            )
        if view.inner.infeasible_stall:
            return ALMRaisePenalty(
                action="infeasible_stall_penalty_increase",
                max_outer_termination="max_outer_after_infeasible_stall",
                feasible_stall_count=view.feasible_stall_count,
            )
        hard_feasible_for_update = (
            measured.routing_state.hard_max_violation
            <= measured.effective_feasibility_tol
        )
        # Only an evaluator with the hybrid quartet can report a mismatch.
        if measured.signal_mismatch_active and hard_feasible_for_update:
            return hybrid.signal_mismatch_step(view)
        if hard_feasible_for_update:
            return _feasible_step(view)
        # ALGENCAN safeguard (Birgin & Martinez 2014, Algorithm 1.1): on the
        # infeasible arm, raise the penalty only when the shifted
        # infeasibility measure ||max(g, -lambda/rho)||_inf failed the
        # sufficient-decrease test against the PREVIOUS ACCEPTED iterate's
        # measure (the loop carries it across outer boundaries, so
        # constraints the caller refreshes only per outer are compared
        # refresh-to-refresh instead of against themselves). When the test
        # passes the current penalty is already driving feasibility: hold
        # penalty and multipliers and let the next outer keep working. No
        # dual update on hold: the dual-update gate is unmet here (the
        # subproblem is under-solved), where no surveyed framework justifies a
        # multiplier step. If the caller's outer_state_callback refreshes
        # nothing, a hold re-solves an identical subproblem once; an unmoved
        # iterate then fails the test and raises on the next outer.
        if (
            view.inner.sufficient_decrease_measure
            <= settings.penalty_sufficient_decrease_tau
            * view.inner.sufficient_decrease_reference
        ):
            return ALMHold(feasible_stall_count=view.feasible_stall_count)
        return ALMRaisePenalty(
            action="penalty_increase",
            max_outer_termination="max_outer_after_penalty_increase",
            feasible_stall_count=view.feasible_stall_count,
        )


DEFAULT_CONTINUATION_POLICY = DefaultContinuationPolicy()

# After two consecutive feasible-but-no-progress outer updates, treat the run as
# plateaued and stop burning boxed continuation cycles.
_PLATEAU_STALL_LIMIT = 2


def _feasible_step(view: ALMPostInnerView) -> Union[ALMStop, ALMRaisePenalty, ALMContinue]:
    """A hard-feasible step whose subproblem missed the dual-update gate.

    A second step in a row without meaningful progress stops on the plateau;
    at the continuation limit the penalty rises; otherwise the subproblem
    continues, with a tighter update stationarity tolerance when it made no
    progress (the tolerance persists and is checkpointed).
    """
    settings = view.settings
    measured = view.inner.measured
    feasible_stall_count = (
        0 if view.inner.meaningful_progress else view.feasible_stall_count + 1
    )
    if feasible_stall_count >= _PLATEAU_STALL_LIMIT:
        # Multipliers and penalty are not advanced.
        return ALMStop(
            action="subproblem_limit",
            termination_reason="plateau_stall",
            message_prefix=(
                "ALM stopped after repeated feasible stationarity "
                "plateau without meaningful progress"
            ),
            feasible_stall_count=feasible_stall_count,
            subproblem_limit_reason="plateau_stall",
            marks_max_outer=True,
        )
    if view.continuation_iteration == settings.max_subproblem_continuations:
        # An exhausted continuation range raises the penalty so the
        # next outer does not re-solve an identical subproblem.
        return ALMRaisePenalty(
            action="subproblem_limit_penalty_increase",
            max_outer_termination="max_outer_after_subproblem_limit_penalty_increase",
            feasible_stall_count=feasible_stall_count,
            subproblem_limit_reason="max_subproblem_continuations",
        )
    update_stationarity_tol = measured.update_stationarity_tol
    if not view.inner.meaningful_progress:
        update_stationarity_tol = min(
            update_stationarity_tol,
            max(settings.stationarity_tol, 0.5 * measured.stationarity_norm),
        )
    return ALMContinue(
        trust_radius=view.trust_radius,
        update_stationarity_tol=update_stationarity_tol,
        feasible_stall_count=feasible_stall_count,
    )


def _kkt_point(measured: ALMIterateMeasurement, settings: ALMSettings) -> bool:
    """Whether ``measured`` passes the KKT stopping test at the shifted
    multipliers ``max(0, λ + ρg)`` its augmented gradient carries: generic and
    hard violations and the complementarity residual within
    ``feasibility_tol``, the augmented-gradient norm within
    ``stationarity_tol``. Without the complementarity test, a multiplier on
    an inactive row could cancel the objective gradient and pass."""
    return (
        _strict_feasibility_satisfied(
            measured.max_feasibility_violation,
            measured.routing_state.hard_max_violation,
            settings.feasibility_tol,
        )
        and measured.stationarity_norm <= settings.stationarity_tol
        and _complementarity_residual(measured.evaluation, measured.routing_state)
        <= settings.feasibility_tol
    )


def _strict_feasibility_satisfied(
    max_feasibility_violation: float,
    hard_max_violation: float,
    feasibility_tol: float,
) -> bool:
    """Return whether both canonical generic and hard maxima meet the strict gate."""
    return float(max_feasibility_violation) <= float(feasibility_tol) and float(
        hard_max_violation
    ) <= float(feasibility_tol)


def _dual_update_gate_satisfied(
    *,
    max_feasibility_violation: float,
    hard_max_violation: float,
    stationarity_norm: float,
    kkt_stationarity_norm: Optional[float],
    update_feasibility_tol: float,
    update_stationarity_tol: float,
) -> bool:
    dual_update_max_violation = max(
        float(max_feasibility_violation),
        float(hard_max_violation),
    )
    dual_update_stationarity_norm = (
        float(kkt_stationarity_norm)
        if kkt_stationarity_norm is not None
        else float(stationarity_norm)
    )
    return dual_update_max_violation <= float(
        update_feasibility_tol
    ) and dual_update_stationarity_norm <= float(update_stationarity_tol)


def _constraints_inactive_candidate(
    measured: ALMIterateMeasurement,
    feasibility_tol: float,
) -> bool:
    """Hybrid signals agree that every hard constraint is strictly satisfied
    and inactive: no surrogate activity and a zero hard shift."""
    routing_state = measured.routing_state
    return (
        routing_state.signal_state.explicit_hybrid_signals
        and routing_state.hard_max_violation <= feasibility_tol
        and not np.any(routing_state.surrogate_activity_mask)
        and routing_state.hard_positive_shift_zero
        and not measured.signal_mismatch_active
    )


def _positive_alm_integer(name: str, value) -> int:
    value_i = _finite_alm_integer(name, value)
    if value_i <= 0:
        raise ValueError(f"{name} must be positive")
    return value_i


def _positive_alm_value(name: str, value) -> float:
    value_f = _finite_alm_value(name, value)
    if value_f <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value_f


def _staged_inner_options(
    inner_options: Mapping[str, object],
    update_stationarity_tol: float,
    *,
    default_maxls: int = 20,
) -> dict:
    """Validated L-BFGS-B options. ``gtol`` is at least
    ``min(1e-4, 0.1 * update_stationarity_tol)``, so an inner solve stops near
    the outer stationarity target; ``maxls`` defaults to ``default_maxls``."""
    options = dict(inner_options)
    if "maxiter" in options:
        options["maxiter"] = _positive_alm_integer(
            "inner_options.maxiter",
            options["maxiter"],
        )
    if "maxfun" in options and options["maxfun"] is not None:
        options["maxfun"] = _positive_alm_integer(
            "inner_options.maxfun",
            options["maxfun"],
        )
    if "ftol" in options:
        options["ftol"] = _positive_alm_value("inner_options.ftol", options["ftol"])
    if "gtol" in options:
        options["gtol"] = _positive_alm_value("inner_options.gtol", options["gtol"])

    base_gtol = float(options.get("gtol", 1e-12))
    staged_gtol = max(
        np.finfo(float).eps,
        min(1e-4, 0.1 * float(update_stationarity_tol)),
    )
    options["gtol"] = max(base_gtol, staged_gtol)
    options["maxls"] = _positive_alm_integer(
        "inner_options.maxls",
        options.get("maxls", default_maxls),
    )
    return options
