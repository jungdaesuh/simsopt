"""What one ALM continuation step measured, what a policy reads, what it decides.

The loop (:mod:`.control`) measures the outer iterate and the inner solve
(:class:`ALMIterateMeasurement`, :class:`ALMInnerSolveOutcome`), shows them to
an ``ALMContinuationPolicy`` (:mod:`.policy`) through a view, and executes the
frozen decision it gets back. A view is borrowed for one synchronous call and
read-only all the way down: its arrays are non-writable views of the loop's,
its mappings read-only proxies, its lists tuples. It is not a snapshot: the
loop's later writes show through, so a policy copies what it keeps. Every
decision carries the ``feasible_stall_count`` the loop keeps after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .core import ALMConstraintRoutingState, ALMSettings


@dataclass(frozen=True)
class ALMIterateMeasurement:
    """One evaluation of the problem and the solver quantities derived from it.

    ``evaluation`` is the evaluation at (x, ``multipliers``, ``penalty``). Inside
    the loop the arrays alias the evaluator's dict; in an outer-step event all
    of it is an owned read-only copy, in a policy view a borrowed read-only
    view. The tolerances are the ones this iterate was judged with.
    """

    evaluation: Mapping[str, object]
    multipliers: np.ndarray
    penalty: float
    solver_constraint_values: np.ndarray
    feasibility_values: np.ndarray
    max_feasibility_violation: float
    routing_state: ALMConstraintRoutingState
    stationarity_norm: float
    kkt_stationarity_norm: Optional[float]
    signal_mismatch_active: bool
    update_feasibility_tol: float
    update_stationarity_tol: float
    effective_feasibility_tol: float


@dataclass(frozen=True)
class ALMInnerSolveOutcome:
    """The inner solve of one continuation step and how the loop judged it.

    ``x``/``measured`` are the accepted iterate (the start iterate when every
    trial was rejected); ``bounds`` and ``inner_options`` are the last
    attempt's box (None: unbounded) and L-BFGS-B options. The
    ``sufficient_decrease_*`` pair is the ALGENCAN measure and its reference.
    """

    x: np.ndarray
    measured: ALMIterateMeasurement
    iterations: int
    attempts: int
    optimizer_success: bool
    optimizer_message: str
    bounds: Optional[Sequence[Tuple[float, float]]]
    inner_options: Optional[Mapping[str, object]]
    inner_profile: Optional[str]
    infeasible_stall: bool
    infeasible_stall_reason: Optional[str]
    inner_false_success: bool
    nonfinite_candidate_evaluation: bool
    nonfinite_candidate_fields: Optional[Sequence[str]]
    meaningful_progress: bool
    feasibility_delta: float
    feasibility_delta_tolerance: float
    sufficient_decrease_measure: float
    sufficient_decrease_reference: float


@dataclass(frozen=True)
class ALMInnerPlanView:
    """One inner L-BFGS-B attempt about to run.

    ``inner_options`` are this process's options (``maxiter`` is what remains
    of its inner budget); ``attempt_radius`` is the trust radius boxing the
    attempt (None: unboxed); ``start_feasible`` is measured once per inner
    solve, at the start iterate.
    """

    settings: ALMSettings
    inner_options: Mapping[str, object]
    update_stationarity_tol: float
    attempt_radius: Optional[float]
    continuation_iteration: int
    start_feasible: bool


@dataclass(frozen=True)
class ALMInnerPlan:
    """L-BFGS-B ``options`` for one attempt (read-only) and the ``profile``
    name the step publishes for them."""

    profile: str
    options: Mapping[str, object]


@dataclass(frozen=True)
class ALMStalledTrialView:
    """A trial the loop classified as an infeasible stall (no move, no gain).

    ``attempt_radius`` is the attempt's trust radius as given (None: no trust
    region); ``inner_false_success`` says L-BFGS-B reported success anyway.
    """

    settings: ALMSettings
    attempt_index: int
    attempt_radius: Optional[float]
    inner_false_success: bool


@dataclass(frozen=True)
class ALMPreInnerView:
    """What the convergence check before the inner solve may read.

    ``start`` is the outer iterate measured before the inner solve;
    ``trust_radius`` is the loop's radius and ``feasible_stall_count`` the
    count the step started with.
    """

    settings: ALMSettings
    outer_iteration: int
    continuation_iteration: int
    start: ALMIterateMeasurement
    last_cap_binding_active: bool
    feasible_stall_count: int
    trust_radius: Optional[float]


@dataclass(frozen=True)
class ALMPostInnerView:
    """What the decision after the inner solve may read: the
    :class:`ALMPreInnerView` fields plus the judged ``inner`` solve, with
    ``trust_radius`` after the inner solve's own trust rule.
    """

    settings: ALMSettings
    outer_iteration: int
    continuation_iteration: int
    start: ALMIterateMeasurement
    inner: ALMInnerSolveOutcome
    last_cap_binding_active: bool
    feasible_stall_count: int
    trust_radius: Optional[float]


@dataclass(frozen=True)
class ALMConverge:
    """Return success: publish ``action`` and report ``message``."""

    action: str
    termination_reason: str
    message: str
    feasible_stall_count: int


@dataclass(frozen=True)
class ALMStop:
    """Return failure after publishing ``action``; the best-feasible restore
    applies. ``marks_max_outer`` publishes ``max_outer`` on the final outer."""

    action: str
    termination_reason: str
    message_prefix: str
    feasible_stall_count: int
    subproblem_limit_reason: Optional[str] = None
    marks_max_outer: bool = False


# The decisions below do not return. ``max_outer_termination`` is the run's
# termination reason when such a step ends the final outer iteration.


@dataclass(frozen=True)
class ALMDualUpdateStep:
    """Update the multipliers and end the outer iteration; with a
    ``penalty_reason`` raise the penalty in the same step."""

    penalty_reason: Optional[str]
    feasible_stall_count: int
    max_outer_termination: ClassVar[str] = "max_outer_after_dual_update"


@dataclass(frozen=True)
class ALMRaisePenalty:
    """Raise the penalty and end the outer iteration (or return at the cap)."""

    action: str
    max_outer_termination: str
    feasible_stall_count: int
    subproblem_limit_reason: Optional[str] = None
    signal_mismatch_repair: bool = False


@dataclass(frozen=True)
class ALMHold:
    """Keep the penalty and multipliers and end the outer iteration."""

    feasible_stall_count: int
    max_outer_termination: ClassVar[str] = "max_outer_after_sufficient_decrease_hold"


@dataclass(frozen=True)
class ALMContinue:
    """Solve the subproblem again with this trust radius and update
    stationarity tolerance."""

    trust_radius: Optional[float]
    update_stationarity_tol: float
    feasible_stall_count: int
    signal_mismatch_repair: bool = False
    max_outer_termination: ClassVar[str] = "max_outer"


ALMStepDecision = Union[
    ALMConverge, ALMStop, ALMDualUpdateStep, ALMRaisePenalty, ALMHold, ALMContinue
]
