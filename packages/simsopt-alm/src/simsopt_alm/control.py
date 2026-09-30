"""ALM outer loop: continuation steps, dual and penalty updates, published
steps and boundaries, the lean result and the best-feasible restore.

The public entry point is :func:`minimize_alm`; the evaluator contract and
the outer-step event fields are in the package docstring. The loop reads the
math from :mod:`.core`, its decisions from the continuation policy
(:mod:`.continuation`, :mod:`.policy`), one inner solve from :mod:`.inner`,
the evaluation contract from :mod:`.evaluation`, and publishes the carriers
of :mod:`.events` and :mod:`.boundary`. The opt-in plugins are the history
recorder (:mod:`simsopt_alm.history`) and checkpoint/resume
(:mod:`simsopt_alm.checkpoint`); this module imports neither.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Dict, Generic, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .core import (
    ALMSettings,
    _bound_reduced_stationarity_norm,
    _constraint_routing_state,
    _extract_constraint_state,
    _next_penalty,
    _project_nonnegative_multipliers_with_diagnostics,
    _stationarity_metrics,
    alm_penalty_schedule_tolerances,
    validate_initial_multipliers,
)
from .boundary import ALMOuterBoundary, _validate_resume_boundary
from .continuation import (
    ALMContinue,
    ALMConverge,
    ALMDualUpdateStep,
    ALMHold,
    ALMInnerSolveOutcome,
    ALMIterateMeasurement,
    ALMPostInnerView,
    ALMPreInnerView,
    ALMStepDecision,
    ALMStop,
)
from .evaluation import (
    ALMEvaluator,
    _checked_evaluation,
    _incumbent_objective_value,
    _measure_iterate,
)
from .events import (
    AcceptedStateT,
    ALMDualUpdate,
    ALMFeasibleIncumbent,
    ALMLoopState,
    ALMOuterStepEvent,
    _accepted_state_geometry_identity,
    _borrowed_read_only_value,
    _frozen_event_value,
    _frozen_loop_state,
    _validate_geometry_identity_pair,
)
from .inner import (
    ALMInnerAttemptRequest,
    ALMInnerAttemptResult,
    _feasibility_improvement_and_floor,
    _inner_options_with_remaining_maxiter,
    _made_meaningful_inner_progress,
    _normalize_base_bounds,
    _checked_trust_radius,
    _project_onto_bounds,
    _run_alm_inner_attempts,
)
from .policy import DEFAULT_CONTINUATION_POLICY, ALMContinuationPolicy

def _require_alm_penalty_within_cap(
    penalty,
    settings: ALMSettings,
    *,
    label: str,
) -> float:
    penalty_f = float(penalty)
    if not np.isfinite(penalty_f) or penalty_f <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    if settings.penalty_max is not None and penalty_f > float(settings.penalty_max):
        raise ValueError(
            f"{label} ({penalty_f}) must be <= "
            f"settings.penalty_max ({settings.penalty_max})"
        )
    return penalty_f

@dataclass
class ALMRunState:
    x: np.ndarray
    total_inner_iterations: int
    trust_radius: Optional[float]
    cap_binding_detected: bool
    cap_binding_indices: Set[int]
    penalty_cap_reached: bool
    penalty_cap_requested: Optional[float]
    # Non-sticky predicate carrying the most recent dual update's
    # ``multiplier_cap_binding``. Resets to False on each non-binding dual
    # update; sticks True until the next dual update that isn't capped.
    # Used to gate both ``converged`` and ``constraints_inactive_converged``
    # so a result that satisfies KKT-on-paper but holds at the multiplier
    # cap (broken Lagrangian interpretation) does not get labeled converged.
    # The sticky ``cap_binding_detected`` remains as a diagnostic.
    last_cap_binding_active: bool = False
    # Previous accepted iterate's sufficient-decrease measure, carried across
    # continuation/outer boundaries for the ALGENCAN penalty safeguard. A
    # same-outer pre-inner re-evaluation CANNOT serve as the reference:
    # constraints the caller refreshes only at outer boundaries are
    # bitwise identical pre- and post-inner within one outer, which would freeze the decrease ratio at
    # exactly 1 whenever such a constraint is the binding violation. None
    # until the first post-inner evaluation exists (the first step then
    # falls back to its own pre-inner measure — the seed iterate).
    previous_sufficient_decrease_measure: Optional[float] = None
    # Carriers used to compute the stored sufficient-decrease measure.  They
    # deliberately remain the pre-transition multipliers/penalty even when a
    # dual or penalty transition changes the live state afterward.
    previous_sufficient_decrease_multipliers: Optional[np.ndarray] = None
    previous_sufficient_decrease_penalty: Optional[float] = None
    # The latest published step's action (a checkpoint's
    # ``completed_action``) and the termination reason if the outer range
    # runs out after the latest step that did not return: its decision's
    # ``max_outer_termination`` on the final outer, else its action.
    last_action: Optional[str] = None
    exhausted_termination: str = "terminated"
    # The caller's base bounds as (lower, upper) pairs (±inf: none), fixed for
    # the run; stationarity is measured against them.
    base_bounds: Optional[List[Tuple[float, float]]] = None

@dataclass(frozen=True)
class ALMPenaltyIncreaseResult:
    penalty: float
    penalty_cap_reached: bool
    penalty_cap_requested: Optional[float]
    update_feasibility_tol: float
    update_stationarity_tol: float
    penalty_update_state: ALMIterateMeasurement
    requested_penalty: Optional[float]

def _run_loop_state(
    run_state: ALMRunState,
    *,
    x: np.ndarray,
    multipliers: np.ndarray,
    penalty: float,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
) -> ALMLoopState[AcceptedStateT]:
    """``run_state``'s carriers with the given iterate (arrays alias the loop)."""
    return ALMLoopState(
        x=x,
        multipliers=multipliers,
        penalty=penalty,
        update_feasibility_tol=update_feasibility_tol,
        update_stationarity_tol=update_stationarity_tol,
        trust_radius=run_state.trust_radius,
        best_feasible=best_feasible,
        total_inner_iterations=int(run_state.total_inner_iterations),
        last_cap_binding_active=bool(run_state.last_cap_binding_active),
        cap_binding_detected=bool(run_state.cap_binding_detected),
        cap_binding_indices=tuple(sorted(run_state.cap_binding_indices)),
        penalty_cap_reached=bool(run_state.penalty_cap_reached),
        penalty_cap_requested=run_state.penalty_cap_requested,
        sufficient_decrease_measure=run_state.previous_sufficient_decrease_measure,
        sufficient_decrease_multipliers=(
            run_state.previous_sufficient_decrease_multipliers
        ),
        sufficient_decrease_penalty=run_state.previous_sufficient_decrease_penalty,
    )

class ALMProcessBudgetExhausted(Exception):
    """This-process accepted-iteration budget is spent.

    Raised by the driver accepted-callback after the last legal accept so
    ALM does not start another inner solve that would write
    ``accepted_iterations > runtime_maxiter``.
    """

def _call_inner_options(
    inner_options: Optional[dict],
    run_state: ALMRunState,
    call_start_inner_iterations: int,
) -> Optional[dict]:
    """``inner_options`` with ``maxiter`` what this ``minimize_alm`` call's
    L-BFGS-B budget has left, computed before each step so that the whole
    call never runs more than ``maxiter`` iterations."""
    return _inner_options_with_remaining_maxiter(
        inner_options,
        int(run_state.total_inner_iterations) - int(call_start_inner_iterations),
    )

def _inner_budget_spent(step_inner_options: Optional[dict]) -> bool:
    """Whether the call's budget is spent: a spent budget ends the run before
    any step but the first of a fresh call (which measures the start)."""
    return step_inner_options is not None and int(step_inner_options.get("maxiter", -1)) == 0

def _effective_feasibility_gate(
    settings: ALMSettings,
    update_feasibility_tol: float,
) -> float:
    return max(
        float(settings.feasibility_tol),
        min(
            float(update_feasibility_tol),
            float(settings.relaxed_feasibility_gate_cap),
        ),
    )

def _build_constraint_metadata_tuples(
    constraint_names: Sequence[str],
    constraint_blocks: Optional[Sequence[str]],
) -> Tuple[Tuple[str, ...], Optional[Tuple[str, ...]]]:
    names_tuple = tuple(str(name) for name in constraint_names)
    if constraint_blocks is None:
        return names_tuple, None
    if len(constraint_blocks) != len(constraint_names):
        raise ValueError("constraint_blocks length must match constraint_names")
    blocks_tuple = tuple(str(block) for block in constraint_blocks)
    return names_tuple, blocks_tuple

@dataclass(frozen=True)
class _ALMNormalizedRunInputs:
    x: np.ndarray
    multipliers: np.ndarray
    penalty: float
    constraint_names_tuple: Tuple[str, ...]
    constraint_blocks_tuple: Optional[Tuple[str, ...]]
    trust_radius: Optional[float]
    update_feasibility_tol: float
    update_stationarity_tol: float
    base_bounds: Optional[List[Tuple[float, float]]]

def _normalize_alm_run_inputs(
    x0,
    constraint_names: Sequence[str],
    settings: ALMSettings,
    initial_multipliers: Optional[np.ndarray],
    initial_penalty: Optional[float],
    constraint_blocks: Optional[Sequence[str]],
    snapshot_accepted_state_fn,
    restore_incumbent_state_fn,
    base_bounds=None,
) -> _ALMNormalizedRunInputs:
    if (snapshot_accepted_state_fn is None) != (restore_incumbent_state_fn is None):
        raise ValueError(
            "snapshot_accepted_state_fn and restore_incumbent_state_fn must be provided together"
        )
    # ALMSettings.__post_init__ owns penalty_max / history_max_entries
    # validation; only runtime-supplied values are validated here.
    constraint_names_tuple, constraint_blocks_tuple = _build_constraint_metadata_tuples(
        constraint_names, constraint_blocks
    )
    base_bounds = _normalize_base_bounds(base_bounds, np.size(x0))
    # Like L-BFGS-B, start from x0 projected onto the box.
    x = _project_onto_bounds(np.asarray(x0, dtype=float), base_bounds)
    multipliers = (
        validate_initial_multipliers(initial_multipliers, len(constraint_names))
        if initial_multipliers is not None
        else np.zeros(len(constraint_names), dtype=float)
    )
    penalty = _require_alm_penalty_within_cap(
        float(initial_penalty)
        if initial_penalty is not None
        else float(settings.penalty_init),
        settings,
        label="initial ALM penalty",
    )
    update_feasibility_tol, update_stationarity_tol = alm_penalty_schedule_tolerances(
        settings,
        penalty,
    )
    return _ALMNormalizedRunInputs(
        x=x,
        multipliers=multipliers,
        penalty=penalty,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        trust_radius=_checked_trust_radius(settings.trust_radius_init),
        update_feasibility_tol=update_feasibility_tol,
        update_stationarity_tol=update_stationarity_tol,
        base_bounds=base_bounds,
    )

def _apply_alm_penalty_increase(
    *,
    settings: ALMSettings,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    x: np.ndarray,
    multipliers: np.ndarray,
    penalty: float,
    constraint_names_tuple: Tuple[str, ...],
    constraint_blocks_tuple: Optional[Tuple[str, ...]],
    base_bounds: Optional[List[Tuple[float, float]]] = None,
) -> ALMPenaltyIncreaseResult:
    next_penalty, cap_hit, requested_penalty = _next_penalty(
        penalty,
        penalty_scale=settings.penalty_scale,
        penalty_max=settings.penalty_max,
    )
    next_feasibility_tol, next_stationarity_tol = alm_penalty_schedule_tolerances(
        settings,
        next_penalty,
    )
    # The re-evaluation is judged with the clamped gate, like every other
    # measurement; routing state and KKT diagnostics must share one
    # active-set definition.
    next_effective_feasibility_tol = _effective_feasibility_gate(
        settings,
        next_feasibility_tol,
    )
    penalty_argument = float(next_penalty)
    penalty_update_state = _measure_iterate(
        _checked_evaluation(
            evaluate_problem,
            x,
            multipliers,
            penalty_argument,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
            context="ALM penalty update evaluation",
        ),
        x=x,
        base_bounds=base_bounds,
        multipliers=multipliers,
        penalty=penalty_argument,
        update_feasibility_tol=next_feasibility_tol,
        update_stationarity_tol=next_stationarity_tol,
        effective_feasibility_tol=next_effective_feasibility_tol,
    )
    return ALMPenaltyIncreaseResult(
        penalty=next_penalty,
        penalty_cap_reached=bool(cap_hit),
        penalty_cap_requested=requested_penalty if cap_hit else None,
        update_feasibility_tol=next_feasibility_tol,
        update_stationarity_tol=next_stationarity_tol,
        penalty_update_state=penalty_update_state,
        requested_penalty=requested_penalty,
    )

@dataclass(frozen=True)
class ALMResult:
    """What :func:`minimize_alm` returns, on success and on failure alike.

    It holds what a caller needs to use and judge the answer, plus what cannot
    be recovered afterwards (why an incumbent was restored, the stop-test
    norms, the last inner result); per-step detail and the loop's safeguard
    flags (multiplier and penalty caps, trust radius) are in the outer-step
    events. ``x`` is the best hard-feasible incumbent when
    ``restored_best_feasible`` (``restored_best_feasible_reason`` says why),
    else the last iterate. At ``x``: ``objective`` is f without penalty terms
    (the evaluator's ``physics_total``, ``base_value`` or ``base_total``, else
    the augmented ``total``); ``constraint_values`` the signed g the solver
    used, in ``constraint_names`` order (a hybrid evaluator's hard channel is
    in ``evaluation``, and the per-row feasibility values are
    ``evaluation["feasibility_values"]``); ``max_violation`` the largest
    of the evaluator's ``feasibility_values``; ``stationarity_norm`` the
    augmented-gradient norm (the evaluator's ``stationarity_norm`` when given;
    at a base bound, without the components pointing out of the box, whose
    full value stays in ``evaluation``) and ``kkt_stationarity_norm`` the active-set KKT residual (None without the
    gradients it needs); ``multipliers`` and ``penalty`` the ones that
    evaluation used; ``evaluation`` an owned copy of the evaluator's dict,
    read-only all the way down as in an event. ``nit`` counts L-BFGS-B
    iterations over all subproblems (a resumed run's earlier ones included)
    and ``outer_iterations`` the outer iterations that ran a step;
    evaluations are not counted.
    ``inner_result`` is L-BFGS-B's
    result for the latest subproblem behind ``x`` (None when there is none,
    e.g. an incumbent restored from a checkpoint). ``x``, ``constraint_values``
    and ``multipliers`` are read-only copies.
    """

    x: np.ndarray
    success: bool
    termination_reason: str
    message: str
    objective: float
    constraint_names: Tuple[str, ...]
    constraint_values: np.ndarray
    max_violation: float
    multipliers: np.ndarray
    penalty: float
    stationarity_norm: float
    kkt_stationarity_norm: Optional[float]
    nit: int
    outer_iterations: int
    restored_best_feasible: bool
    restored_best_feasible_reason: Optional[str]
    evaluation: Mapping[str, object]
    inner_result: Optional[object]

def _read_only_float_array(values) -> np.ndarray:
    array = np.array(values, dtype=float)
    array.setflags(write=False)
    return array

def _build_alm_result(
    *,
    run_state: ALMRunState,
    constraint_names_tuple: Tuple[str, ...],
    success: bool,
    message: str,
    termination_reason: str,
    outer_iterations: int,
    evaluation: Mapping[str, object],
    multipliers: np.ndarray,
    penalty: float,
    inner_result: Optional[object],
    stationarity_norm: float,
    kkt_stationarity_norm: Optional[float],
    restored_best_feasible_reason: Optional[str] = None,
) -> ALMResult:
    """The returned result at ``run_state.x`` (restored when a reason is given)."""
    (
        solver_constraint_values,
        _feasibility_values,
        _dual_update_values,
        max_violation,
    ) = _extract_constraint_state(evaluation)
    return ALMResult(
        x=_read_only_float_array(run_state.x),
        success=bool(success),
        termination_reason=str(termination_reason),
        message=message,
        objective=_incumbent_objective_value(evaluation),
        constraint_names=constraint_names_tuple,
        constraint_values=_read_only_float_array(solver_constraint_values),
        max_violation=float(max_violation),
        multipliers=_read_only_float_array(multipliers),
        penalty=float(penalty),
        stationarity_norm=float(stationarity_norm),
        kkt_stationarity_norm=(
            None if kkt_stationarity_norm is None else float(kkt_stationarity_norm)
        ),
        nit=int(run_state.total_inner_iterations),
        outer_iterations=int(outer_iterations),
        restored_best_feasible=restored_best_feasible_reason is not None,
        restored_best_feasible_reason=restored_best_feasible_reason,
        evaluation=_frozen_event_value(evaluation, {}),
        inner_result=inner_result,
    )

@dataclass(frozen=True)
class _ALMRestoredIterate:
    x: np.ndarray
    evaluation: dict
    multipliers_state: np.ndarray
    penalty_state: float
    inner_result: Optional[object]
    restored_best_feasible: bool
    restored_best_feasible_reason: Optional[str]

def _restore_alm_best_feasible_on_failure(
    *,
    current_x: np.ndarray,
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    settings: ALMSettings,
    restore_incumbent_state_fn: Optional[Callable[[AcceptedStateT], None]],
    evaluation: dict,
    multipliers_state: np.ndarray,
    penalty_state: float,
    inner_result,
) -> _ALMRestoredIterate:
    """Restore the best hard-feasible incumbent when the final iterate is
    hard-infeasible or has a worse objective. Final-iterate feasibility is
    judged on the HARD (certified) channel computed from ``evaluation``, never
    the surrogate channel, which can read feasible while the design is
    hard-infeasible. ``restore_incumbent_state_fn`` must fail-closed on live
    geometry; this helper only validates the snapshot carrier/identity pair
    before restore.
    """
    restored_best_feasible = False
    restored_best_feasible_reason = None
    restored_evaluation = evaluation
    restored_x = current_x.copy()
    restored_multipliers_state = np.asarray(multipliers_state, dtype=float).copy()
    restored_penalty_state = float(penalty_state)
    restored_inner_result = inner_result

    restore_reasons: List[str] = []
    if best_feasible is not None:
        final_hard_max_violation = _constraint_routing_state(
            evaluation,
            multipliers_state,
            penalty_state,
            settings.feasibility_tol,
        ).hard_max_violation
        if final_hard_max_violation > settings.feasibility_tol:
            restore_reasons.append("final_iterate_infeasible")
        if _incumbent_objective_value(restored_evaluation) > _incumbent_objective_value(
            best_feasible.evaluation
        ):
            restore_reasons.append("final_iterate_worse_than_best_feasible")

    if restore_reasons:
        restored_best_feasible = True
        restored_best_feasible_reason = ",".join(restore_reasons)
        restored_x = best_feasible.x.copy()
        restored_evaluation = best_feasible.evaluation
        restored_multipliers_state = best_feasible.multipliers.copy()
        restored_penalty_state = best_feasible.penalty
        restored_inner_result = best_feasible.inner_result
        if (
            restore_incumbent_state_fn is not None
            and best_feasible.incumbent_state is not None
        ):
            _validate_geometry_identity_pair(
                accepted_state=best_feasible.incumbent_state,
                geometry_identity=best_feasible.geometry_identity,
                context="ALM best-feasible restore",
            )
            restore_incumbent_state_fn(best_feasible.incumbent_state)

    return _ALMRestoredIterate(
        x=restored_x,
        evaluation=restored_evaluation,
        multipliers_state=restored_multipliers_state,
        penalty_state=restored_penalty_state,
        inner_result=restored_inner_result,
        restored_best_feasible=bool(restored_best_feasible),
        restored_best_feasible_reason=restored_best_feasible_reason,
    )

def _build_alm_failure_result_with_optional_restore(
    *,
    settings: ALMSettings,
    constraint_names_tuple: Tuple[str, ...],
    run_state: ALMRunState,
    last_outer_iteration: int,
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    restore_incumbent_state_fn: Optional[Callable[[AcceptedStateT], None]],
    termination_reason: str,
    message_prefix: str,
    evaluation: dict,
    multipliers_state: np.ndarray,
    penalty_state: float,
    inner_result,
    restored_message_prefix: Optional[str] = None,
    restored_termination_reason: Optional[str] = None,
) -> ALMResult:
    restored_state = _restore_alm_best_feasible_on_failure(
        current_x=run_state.x,
        best_feasible=best_feasible,
        settings=settings,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        evaluation=evaluation,
        multipliers_state=multipliers_state,
        penalty_state=penalty_state,
        inner_result=inner_result,
    )
    # Keep the mutable boundary carrier authoritative when terminal failure
    # restores a best-feasible incumbent.  The completed-outer callback is
    # sampled from ``run_state``; leaving its old endpoint x in place would
    # pair the restored result with a different physical geometry.
    run_state.x = restored_state.x.copy()
    restored_routing_state = _constraint_routing_state(
        restored_state.evaluation,
        restored_state.multipliers_state,
        restored_state.penalty_state,
        settings.feasibility_tol,
    )
    (
        restored_stationarity_norm,
        restored_kkt_stationarity_norm,
        _restored_signal_mismatch_active,
    ) = _stationarity_metrics(
        restored_state.evaluation,
        restored_routing_state,
        settings.feasibility_tol,
    )
    restored_stationarity_norm = _bound_reduced_stationarity_norm(
        restored_state.evaluation, run_state.x, run_state.base_bounds
    )
    restored_max_feasibility_violation = _extract_constraint_state(
        restored_state.evaluation
    )[3]
    effective_message_prefix = message_prefix
    effective_termination_reason = termination_reason
    if restored_state.restored_best_feasible:
        if restored_message_prefix is not None:
            effective_message_prefix = restored_message_prefix
        if restored_termination_reason is not None:
            effective_termination_reason = restored_termination_reason
    message = (
        f"{effective_message_prefix}: "
        f"max_violation={restored_max_feasibility_violation:.3e}, "
        f"stationarity={restored_stationarity_norm:.3e}"
    )
    return _build_alm_result(
        run_state=run_state,
        constraint_names_tuple=constraint_names_tuple,
        success=False,
        message=message,
        termination_reason=effective_termination_reason,
        outer_iterations=last_outer_iteration,
        evaluation=restored_state.evaluation,
        multipliers=restored_state.multipliers_state,
        penalty=restored_state.penalty_state,
        inner_result=restored_state.inner_result,
        stationarity_norm=restored_stationarity_norm,
        kkt_stationarity_norm=restored_kkt_stationarity_norm,
        restored_best_feasible_reason=restored_state.restored_best_feasible_reason,
    )

def _handle_alm_dual_update_transition(
    *,
    multipliers: np.ndarray,
    routing_state,
    penalty_argument,
    settings: ALMSettings,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
) -> ALMDualUpdate:
    new_multipliers, cap_binding, cap_binding_indices = (
        _project_nonnegative_multipliers_with_diagnostics(
            multipliers,
            routing_state.signal_state.preferred_dual_update_values,
            penalty_argument,
            settings.multiplier_max,
        )
    )
    penalty_tolerance_scale = float(settings.penalty_scale)
    return ALMDualUpdate(
        multipliers=new_multipliers,
        update_feasibility_tol=max(
            update_feasibility_tol / penalty_tolerance_scale,
            settings.feasibility_tol,
        ),
        update_stationarity_tol=max(
            update_stationarity_tol / penalty_tolerance_scale,
            settings.stationarity_tol,
        ),
        multiplier_cap_binding=bool(cap_binding),
        multiplier_cap_binding_indices=list(cap_binding_indices),
    )

def _handle_alm_penalty_cap_termination(
    *,
    settings: ALMSettings,
    constraint_names_tuple: Tuple[str, ...],
    run_state: ALMRunState,
    last_outer_iteration: int,
    best_feasible,
    restore_incumbent_state_fn,
    penalty_transition,
    multipliers: np.ndarray,
    penalty: float,
    result,
):
    return _build_alm_failure_result_with_optional_restore(
        settings=settings,
        constraint_names_tuple=constraint_names_tuple,
        run_state=run_state,
        last_outer_iteration=last_outer_iteration,
        best_feasible=best_feasible,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        termination_reason="penalty_cap_reached",
        message_prefix=(
            "ALM stopped after the requested penalty update "
            f"{penalty_transition.requested_penalty:.3e} exceeded "
            f"the configured penalty cap {penalty:.3e}."
        ),
        restored_message_prefix=(
            "ALM stopped at the penalty cap after restoring best feasible iterate"
        ),
        restored_termination_reason=("penalty_cap_reached_restored_best_feasible"),
        evaluation=penalty_transition.penalty_update_state.evaluation,
        multipliers_state=multipliers,
        penalty_state=penalty,
        inner_result=result,
    )

class _ALMContinuationDecision(str, Enum):
    RETURN = "return"
    BREAK_OUTER = "break_outer"
    CONTINUE_CONTINUATION = "continue_continuation"

class _ALMOuterDecision(str, Enum):
    RETURN = "return"
    NEXT_OUTER = "next_outer"
    EXHAUST = "exhaust"
    BUDGET_SPENT = "budget_spent"

@dataclass(frozen=True)
class _ALMContinuationStepResult(Generic[AcceptedStateT]):
    decision: _ALMContinuationDecision
    result: Optional[object]
    multipliers: np.ndarray
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    feasible_stall_count: int
    last_result: Optional[object]
    final_eval: Optional[dict]
    final_multipliers: np.ndarray
    final_penalty: float
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]]

@dataclass(frozen=True)
class _ALMOuterIterationResult(Generic[AcceptedStateT]):
    decision: _ALMOuterDecision
    result: Optional[object]
    multipliers: np.ndarray
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    last_result: Optional[object]
    final_eval: Optional[dict]
    final_multipliers: np.ndarray
    final_penalty: float
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]]

@dataclass(frozen=True)
class _ContinuationContext(Generic[AcceptedStateT]):
    """Inputs every arm of one continuation step shares and never rebinds.

    The step builds it once it has measured the start iterate and adds
    ``inner`` after the inner solve. The mutable carriers
    (`_ContinuationStepState`, `ALMRunState`) and the inner result travel
    separately.
    """

    settings: ALMSettings
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict]
    on_outer_step: Optional[Callable[[ALMOuterStepEvent[AcceptedStateT]], None]]
    restore_incumbent_state_fn: Optional[Callable[[AcceptedStateT], None]]
    constraint_names_tuple: Tuple[str, ...]
    constraint_blocks_tuple: Optional[Tuple[str, ...]]
    outer_iteration: int
    continuation_iteration: int
    is_final_outer: bool
    start_x: np.ndarray
    start: ALMIterateMeasurement
    inner: Optional[ALMInnerSolveOutcome] = None

@dataclass
class _ContinuationStepState(Generic[AcceptedStateT]):
    """Mutable carrier for the 10 sticky locals threaded through every
    `_run_alm_continuation_step` return.

    Each field mirrors a kwarg on `_run_alm_continuation_step` that may be
    rewritten as the body progresses (penalty update, dual update, inner
    attempt acceptance, etc.). Reads/writes happen through this carrier so
    that the 15 return sites collapse to a single `_finalize_continuation_step`
    call. The frozen public `_ALMContinuationStepResult` is rebuilt from this
    carrier at each return.
    """

    multipliers: np.ndarray
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    feasible_stall_count: int
    last_result: Optional[object]
    final_eval: Optional[dict]
    final_multipliers: np.ndarray
    final_penalty: float
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]]

def _finalize_continuation_step(
    state: _ContinuationStepState[AcceptedStateT],
    decision: _ALMContinuationDecision,
    result: Optional[object],
) -> _ALMContinuationStepResult[AcceptedStateT]:
    """Package the mutable continuation-step carrier into the frozen public
    result dataclass. Single source of truth for the 12-field shape."""
    return _ALMContinuationStepResult(
        decision=decision,
        result=result,
        multipliers=state.multipliers,
        penalty=state.penalty,
        update_feasibility_tol=state.update_feasibility_tol,
        update_stationarity_tol=state.update_stationarity_tol,
        feasible_stall_count=state.feasible_stall_count,
        last_result=state.last_result,
        final_eval=state.final_eval,
        final_multipliers=state.final_multipliers,
        final_penalty=state.final_penalty,
        best_feasible=state.best_feasible,
    )

def _publish_outer_step(
    context: _ContinuationContext[AcceptedStateT],
    *,
    state: _ContinuationStepState[AcceptedStateT],
    run_state: ALMRunState,
    action: str,
    outer_termination: Optional[str] = None,
    subproblem_limit_reason: Optional[str] = None,
    signal_mismatch_repair: bool = False,
    dual_update: Optional[ALMDualUpdate] = None,
    penalty_update: Optional[ALMIterateMeasurement] = None,
    dual_update_penalty_reason: Optional[str] = None,
) -> None:
    """Record this step's final decision and pass it to ``on_outer_step`` as
    an event of owned, read-only copies (built only when someone listens).
    Every continuation step calls this exactly once, before any terminal
    result (and best-feasible restore) is built."""
    run_state.last_action = action
    if context.on_outer_step is None:
        return
    memo: Dict[int, object] = {}
    event = ALMOuterStepEvent(
        outer_iteration=context.outer_iteration,
        continuation_iteration=context.continuation_iteration,
        constraint_names=context.constraint_names_tuple,
        action=action,
        outer_termination=outer_termination,
        subproblem_limit_reason=subproblem_limit_reason,
        signal_mismatch_repair=signal_mismatch_repair,
        feasible_stall_count=int(state.feasible_stall_count),
        start_x=_frozen_event_value(context.start_x, memo),
        start=_frozen_event_value(context.start, memo),
        inner=_frozen_event_value(context.inner, memo),
        dual_update=_frozen_event_value(dual_update, memo),
        penalty_update=_frozen_event_value(penalty_update, memo),
        dual_update_penalty_reason=dual_update_penalty_reason,
        after=_frozen_loop_state(
            _run_loop_state(
                run_state,
                x=run_state.x,
                multipliers=state.multipliers,
                penalty=state.penalty,
                update_feasibility_tol=state.update_feasibility_tol,
                update_stationarity_tol=state.update_stationarity_tol,
                best_feasible=state.best_feasible,
            ),
            memo,
        ),
    )
    context.on_outer_step(event)

def _max_outer_termination(context: _ContinuationContext[AcceptedStateT]) -> Optional[str]:
    """``outer_termination`` of a step that ends the outer without returning."""
    return "max_outer" if context.is_final_outer else None

def _exhausted_termination(
    context: _ContinuationContext[AcceptedStateT],
    *,
    max_outer_termination: str,
    action: str,
) -> str:
    """The run's termination reason if the outer range runs out after a step
    that did not return: the decision's label on the final outer, else the
    action it published."""
    return max_outer_termination if context.is_final_outer else action

def _emit_alm_stall_failure_step(
    context: _ContinuationContext[AcceptedStateT],
    *,
    state: _ContinuationStepState[AcceptedStateT],
    run_state: ALMRunState,
    action: str,
    termination_reason: str,
    message_prefix: str,
    inner_result: Optional[object],
    outer_termination: Optional[str] = None,
    subproblem_limit_reason: Optional[str] = None,
) -> _ALMContinuationStepResult[AcceptedStateT]:
    """Publish a terminal failure arm (constraints-inactive / signal-mismatch
    / plateau stall, process budget) and finalize the RETURN step result
    built from `_build_alm_failure_result_with_optional_restore`.
    """
    _publish_outer_step(
        context,
        state=state,
        run_state=run_state,
        action=action,
        outer_termination=outer_termination,
        subproblem_limit_reason=subproblem_limit_reason,
    )
    return _finalize_continuation_step(
        state,
        _ALMContinuationDecision.RETURN,
        _build_alm_failure_result_with_optional_restore(
            settings=context.settings,
            constraint_names_tuple=context.constraint_names_tuple,
            run_state=run_state,
            last_outer_iteration=context.outer_iteration,
            best_feasible=state.best_feasible,
            restore_incumbent_state_fn=context.restore_incumbent_state_fn,
            termination_reason=termination_reason,
            message_prefix=message_prefix,
            evaluation=state.final_eval,
            multipliers_state=state.multipliers,
            penalty_state=state.penalty,
            inner_result=inner_result,
        ),
    )

def _execute_step_decision(
    context: _ContinuationContext[AcceptedStateT],
    decision: ALMStepDecision,
    *,
    state: _ContinuationStepState[AcceptedStateT],
    run_state: ALMRunState,
) -> _ALMContinuationStepResult[AcceptedStateT]:
    """Carry out the policy's post-inner ``decision``: update the carriers,
    publish the step exactly once, and finalize it. A converge, a stop or a
    penalty raise that hits the cap RETURNs; a continue continues the
    subproblem; a dual update, raise or hold ends the outer iteration."""
    state.feasible_stall_count = decision.feasible_stall_count
    measured = context.inner.measured
    if isinstance(decision, ALMConverge):
        _publish_outer_step(
            context, state=state, run_state=run_state, action=decision.action
        )
        return _finalize_continuation_step(
            state,
            _ALMContinuationDecision.RETURN,
            _build_alm_result(
                run_state=run_state,
                constraint_names_tuple=context.constraint_names_tuple,
                success=True,
                message=decision.message,
                termination_reason=decision.termination_reason,
                outer_iterations=context.outer_iteration,
                evaluation=state.final_eval,
                multipliers=state.multipliers,
                penalty=state.penalty,
                inner_result=state.last_result,
                stationarity_norm=measured.stationarity_norm,
                kkt_stationarity_norm=measured.kkt_stationarity_norm,
            ),
        )
    if isinstance(decision, ALMStop):
        return _emit_alm_stall_failure_step(
            context,
            state=state,
            run_state=run_state,
            action=decision.action,
            termination_reason=decision.termination_reason,
            message_prefix=decision.message_prefix,
            inner_result=state.last_result,
            outer_termination=(
                _max_outer_termination(context)
                if decision.marks_max_outer
                else None
            ),
            subproblem_limit_reason=decision.subproblem_limit_reason,
        )
    # A step that ends the final outer without returning publishes
    # ``max_outer``, so the exhausted-outer reason is a ``max_outer*`` label.
    if isinstance(decision, ALMContinue):
        # The radius is set before the event publishes it.
        run_state.trust_radius = decision.trust_radius
        state.update_stationarity_tol = decision.update_stationarity_tol
        run_state.exhausted_termination = _exhausted_termination(
            context,
            max_outer_termination=decision.max_outer_termination,
            action="subproblem_continue",
        )
        _publish_outer_step(
            context,
            state=state,
            run_state=run_state,
            action="subproblem_continue",
            outer_termination=_max_outer_termination(context),
            signal_mismatch_repair=decision.signal_mismatch_repair,
        )
        return _finalize_continuation_step(
            state, _ALMContinuationDecision.CONTINUE_CONTINUATION, None
        )
    if isinstance(decision, ALMHold):
        run_state.exhausted_termination = _exhausted_termination(
            context,
            max_outer_termination=decision.max_outer_termination,
            action="sufficient_decrease_hold",
        )
        _publish_outer_step(
            context,
            state=state,
            run_state=run_state,
            action="sufficient_decrease_hold",
            outer_termination=_max_outer_termination(context),
        )
        return _finalize_continuation_step(
            state, _ALMContinuationDecision.BREAK_OUTER, None
        )
    dual_update = None
    if isinstance(decision, ALMDualUpdateStep):
        # The dual update ends this outer: ``_run_alm_outer_iteration``
        # restarts the feasible-stall counter.
        dual_update = _handle_alm_dual_update_transition(
            multipliers=state.multipliers,
            routing_state=measured.routing_state,
            penalty_argument=measured.penalty,
            settings=context.settings,
            update_feasibility_tol=state.update_feasibility_tol,
            update_stationarity_tol=state.update_stationarity_tol,
        )
        state.multipliers = dual_update.multipliers
        state.update_feasibility_tol = dual_update.update_feasibility_tol
        state.update_stationarity_tol = dual_update.update_stationarity_tol
        state.final_multipliers = state.multipliers.copy()
        state.final_penalty = state.penalty
        # The non-sticky current predicate updates on every dual update
        # (True or False). Sticky `cap_binding_detected` remains
        # diagnostic-only.
        run_state.last_cap_binding_active = bool(
            dual_update.multiplier_cap_binding
        )
        if dual_update.multiplier_cap_binding:
            run_state.cap_binding_detected = True
            run_state.cap_binding_indices.update(
                dual_update.multiplier_cap_binding_indices
            )
        if decision.penalty_reason is None:
            state.final_eval = _checked_evaluation(
                context.evaluate_problem,
                run_state.x,
                state.multipliers,
                measured.penalty,
                constraint_names_tuple=context.constraint_names_tuple,
                constraint_blocks_tuple=context.constraint_blocks_tuple,
                context="ALM final state evaluation",
            )
            state.final_multipliers = state.multipliers.copy()
            state.final_penalty = state.penalty
            run_state.exhausted_termination = _exhausted_termination(
                context,
                max_outer_termination=decision.max_outer_termination,
                action="dual_update",
            )
            _publish_outer_step(
                context,
                state=state,
                run_state=run_state,
                action="dual_update",
                outer_termination=_max_outer_termination(context),
                dual_update=dual_update,
            )
            return _finalize_continuation_step(
                state, _ALMContinuationDecision.BREAK_OUTER, None
            )
        action = "dual_update"
        subproblem_limit_reason = None
        signal_mismatch_repair = False
        dual_update_penalty_reason = decision.penalty_reason
    else:
        action = decision.action
        subproblem_limit_reason = decision.subproblem_limit_reason
        signal_mismatch_repair = decision.signal_mismatch_repair
        dual_update_penalty_reason = None

    # Raise the penalty (with the dual update, or for a raise decision).
    penalty_transition = _apply_alm_penalty_increase(
        settings=context.settings,
        evaluate_problem=context.evaluate_problem,
        x=run_state.x,
        multipliers=state.multipliers,
        penalty=state.penalty,
        constraint_names_tuple=context.constraint_names_tuple,
        constraint_blocks_tuple=context.constraint_blocks_tuple,
        base_bounds=run_state.base_bounds,
    )
    cap_reached = penalty_transition.penalty_cap_reached
    state.penalty = penalty_transition.penalty
    run_state.penalty_cap_reached = cap_reached
    run_state.penalty_cap_requested = penalty_transition.penalty_cap_requested
    state.update_feasibility_tol = penalty_transition.update_feasibility_tol
    state.update_stationarity_tol = penalty_transition.update_stationarity_tol
    state.final_eval = penalty_transition.penalty_update_state.evaluation
    state.final_multipliers = state.multipliers.copy()
    state.final_penalty = state.penalty
    run_state.exhausted_termination = _exhausted_termination(
        context, max_outer_termination=decision.max_outer_termination, action=action
    )
    # A capped raise keeps the dual-update reason on its event.
    _publish_outer_step(
        context,
        state=state,
        run_state=run_state,
        action="penalty_cap_reached" if cap_reached else action,
        outer_termination=None if cap_reached else _max_outer_termination(context),
        subproblem_limit_reason=subproblem_limit_reason,
        signal_mismatch_repair=signal_mismatch_repair,
        dual_update=dual_update,
        penalty_update=penalty_transition.penalty_update_state,
        dual_update_penalty_reason=dual_update_penalty_reason,
    )
    if not cap_reached:
        return _finalize_continuation_step(
            state, _ALMContinuationDecision.BREAK_OUTER, None
        )
    return _finalize_continuation_step(
        state,
        _ALMContinuationDecision.RETURN,
        _handle_alm_penalty_cap_termination(
            settings=context.settings,
            constraint_names_tuple=context.constraint_names_tuple,
            run_state=run_state,
            last_outer_iteration=context.outer_iteration,
            best_feasible=state.best_feasible,
            restore_incumbent_state_fn=context.restore_incumbent_state_fn,
            penalty_transition=penalty_transition,
            multipliers=state.multipliers,
            penalty=state.penalty,
            result=state.last_result,
        ),
    )

def _sufficient_decrease_measure(
    hard_signed_values: np.ndarray,
    multipliers: np.ndarray,
    penalty: float,
) -> float:
    """ALGENCAN's infeasibility-and-complementarity measure ``||V||_inf``
    with ``V_i = max(g_i, -lambda_i/rho)`` (Birgin & Martinez 2014,
    Algorithm 1.1). Coincides with the plain max violation when all
    multipliers are zero; unlike the plain violation it stays nonzero while
    a feasible constraint still carries an unconverged multiplier."""
    g = np.asarray(hard_signed_values, dtype=float).reshape(-1)
    if g.size == 0:
        return 0.0
    lam = np.asarray(multipliers, dtype=float).reshape(-1)
    return float(np.max(np.abs(np.maximum(g, -lam / float(penalty)))))

def _improved_incumbent(
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    *,
    x: np.ndarray,
    measured: ALMIterateMeasurement,
    inner_result: Optional[object],
    settings: ALMSettings,
    snapshot_accepted_state_fn: Optional[Callable[[], AcceptedStateT]],
) -> Optional[ALMFeasibleIncumbent[AcceptedStateT]]:
    """``best_feasible``, or the measured iterate ``x`` as the new incumbent
    when it is hard-feasible at ``feasibility_tol`` and ranks strictly better
    (``_incumbent_objective_value``); the caller's accepted state is
    snapshotted only for a new incumbent. Only the HARD (certified) channel
    counts: a surrogate can read feasible where the design is not."""
    if measured.routing_state.hard_max_violation > settings.feasibility_tol or (
        best_feasible is not None
        and _incumbent_objective_value(measured.evaluation)
        >= _incumbent_objective_value(best_feasible.evaluation)
    ):
        return best_feasible
    return ALMFeasibleIncumbent(
        x=x.copy(),
        evaluation=measured.evaluation,
        multipliers=np.array(measured.multipliers, dtype=float),
        penalty=float(measured.penalty),
        inner_result=inner_result,
        incumbent_state=(
            None if snapshot_accepted_state_fn is None else snapshot_accepted_state_fn()
        ),
    )

def _incumbent_at_step_start(
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    *,
    start: ALMIterateMeasurement,
    live_x: np.ndarray,
    problem_refreshed: bool,
    last_result: Optional[object],
    settings: ALMSettings,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    constraint_names_tuple: Tuple[str, ...],
    constraint_blocks_tuple: Optional[Tuple[str, ...]],
    snapshot_accepted_state_fn: Optional[Callable[[], AcceptedStateT]],
    restore_incumbent_state_fn: Optional[Callable[[AcceptedStateT], None]],
) -> Optional[ALMFeasibleIncumbent[AcceptedStateT]]:
    """The incumbent a step's inner solve starts from. After an
    ``outer_state_callback`` (``problem_refreshed``), which may have changed
    the problem, ``best_feasible`` is evaluated again at its x, multipliers
    and penalty (the start's evaluation when those are the start's; away from
    the live x, in its own restored state, the live state put back after) and
    kept only while hard-feasible. Then the measured start itself competes (a
    feasible x0 above all), before an inner solve can leave it."""
    if problem_refreshed and best_feasible is not None:
        at_live_x = np.array_equal(best_feasible.x, live_x)
        if (
            at_live_x
            and np.array_equal(best_feasible.multipliers, start.multipliers)
            and best_feasible.penalty == start.penalty
        ):
            evaluation = start.evaluation
        else:
            swap_state = (
                not at_live_x
                and restore_incumbent_state_fn is not None
                and best_feasible.incumbent_state is not None
            )
            if swap_state:
                live_state = snapshot_accepted_state_fn()
                _validate_geometry_identity_pair(
                    accepted_state=best_feasible.incumbent_state,
                    geometry_identity=best_feasible.geometry_identity,
                    context="ALM best-feasible re-evaluation",
                )
                restore_incumbent_state_fn(best_feasible.incumbent_state)
            evaluation = _checked_evaluation(
                evaluate_problem,
                best_feasible.x,
                best_feasible.multipliers,
                best_feasible.penalty,
                constraint_names_tuple=constraint_names_tuple,
                constraint_blocks_tuple=constraint_blocks_tuple,
                context="ALM best-feasible re-evaluation",
            )
            if swap_state:
                restore_incumbent_state_fn(live_state)
        still_feasible = _constraint_routing_state(
            evaluation, best_feasible.multipliers, best_feasible.penalty, settings.feasibility_tol
        ).hard_max_violation <= settings.feasibility_tol
        best_feasible = replace(best_feasible, evaluation=evaluation) if still_feasible else None
    return _improved_incumbent(
        best_feasible,
        x=live_x,
        measured=start,
        inner_result=last_result,
        settings=settings,
        snapshot_accepted_state_fn=snapshot_accepted_state_fn,
    )

def _inner_solve_outcome(
    attempt: ALMInnerAttemptResult,
    *,
    measured: ALMIterateMeasurement,
    meaningful_progress: bool,
    feasibility_delta: float,
    feasibility_delta_tolerance: float,
    sufficient_decrease_measure: float,
    sufficient_decrease_reference: float,
) -> ALMInnerSolveOutcome:
    """The published outcome of ``attempt`` and the loop's judgment of it."""
    optimizer_result = attempt.optimizer_result
    return ALMInnerSolveOutcome(
        x=attempt.x,
        measured=measured,
        iterations=int(attempt.iterations),
        attempts=int(attempt.attempts),
        optimizer_success=bool(getattr(optimizer_result, "success", False)),
        optimizer_message=str(getattr(optimizer_result, "message", "")),
        bounds=attempt.bounds,
        inner_options=attempt.last_inner_options,
        inner_profile=attempt.last_inner_profile,
        infeasible_stall=bool(attempt.forced_infeasible_penalty_cycle),
        infeasible_stall_reason=attempt.forced_infeasible_penalty_reason,
        inner_false_success=bool(attempt.forced_inner_false_success),
        nonfinite_candidate_evaluation=bool(attempt.nonfinite_candidate_evaluation),
        nonfinite_candidate_fields=attempt.nonfinite_candidate_fields,
        meaningful_progress=meaningful_progress,
        feasibility_delta=feasibility_delta,
        feasibility_delta_tolerance=feasibility_delta_tolerance,
        sufficient_decrease_measure=sufficient_decrease_measure,
        sufficient_decrease_reference=sufficient_decrease_reference,
    )

def _run_alm_continuation_step(
    *,
    settings: ALMSettings,
    run_state: ALMRunState,
    multipliers: np.ndarray,
    penalty: float,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    feasible_stall_count: int,
    last_result: Optional[object],
    final_eval: Optional[dict],
    final_multipliers: np.ndarray,
    final_penalty: float,
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    inner_options: Optional[dict],
    outer_iteration: int,
    continuation_iteration: int,
    is_final_outer: bool,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    inner_callback: Optional[Callable[[np.ndarray], None]],
    accepted_callback: Optional[Callable[[np.ndarray], None]],
    on_outer_step: Optional[Callable[[ALMOuterStepEvent[AcceptedStateT]], None]],
    snapshot_accepted_state_fn: Optional[Callable[[], AcceptedStateT]],
    restore_incumbent_state_fn: Optional[Callable[[AcceptedStateT], None]],
    constraint_names_tuple: Tuple[str, ...],
    constraint_blocks_tuple: Optional[Tuple[str, ...]],
    problem_refreshed: bool,
    base_bounds=None,
    continuation_policy: ALMContinuationPolicy = DEFAULT_CONTINUATION_POLICY,
) -> _ALMContinuationStepResult[AcceptedStateT]:
    """One continuation step. ``problem_refreshed``: an
    ``outer_state_callback`` ran since the step before, so the best-feasible
    incumbent is judged again under the current problem."""
    state = _ContinuationStepState[AcceptedStateT](
        multipliers=multipliers,
        penalty=penalty,
        update_feasibility_tol=update_feasibility_tol,
        update_stationarity_tol=update_stationarity_tol,
        feasible_stall_count=feasible_stall_count,
        last_result=last_result,
        final_eval=final_eval,
        final_multipliers=final_multipliers,
        final_penalty=final_penalty,
        best_feasible=best_feasible,
    )
    start_x = run_state.x.copy()
    penalty_argument = float(state.penalty)
    current_eval = _checked_evaluation(
        evaluate_problem,
        run_state.x,
        state.multipliers,
        penalty_argument,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        context="ALM outer iterate evaluation",
    )
    effective_feasibility_tol = _effective_feasibility_gate(
        settings, state.update_feasibility_tol
    )
    context = _ContinuationContext[AcceptedStateT](
        settings=settings,
        evaluate_problem=evaluate_problem,
        on_outer_step=on_outer_step,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        outer_iteration=outer_iteration,
        continuation_iteration=continuation_iteration,
        is_final_outer=is_final_outer,
        start_x=start_x,
        start=_measure_iterate(
            current_eval,
            x=run_state.x,
            base_bounds=run_state.base_bounds,
            multipliers=state.multipliers,
            penalty=penalty_argument,
            update_feasibility_tol=state.update_feasibility_tol,
            update_stationarity_tol=state.update_stationarity_tol,
            effective_feasibility_tol=effective_feasibility_tol,
        ),
    )
    start = context.start

    # The policy's read-only views of this step, built once per object; the
    # memo's keys (the start and inner measurements) live through the step.
    policy_memo: Dict[int, object] = {}
    pre_inner_view = ALMPreInnerView(
        settings=settings,
        outer_iteration=outer_iteration,
        continuation_iteration=continuation_iteration,
        start=_borrowed_read_only_value(start, policy_memo),
        last_cap_binding_active=run_state.last_cap_binding_active,
        feasible_stall_count=state.feasible_stall_count,
        trust_radius=run_state.trust_radius,
    )
    converge = continuation_policy.before_inner(pre_inner_view)
    if converge is not None:
        state.feasible_stall_count = converge.feasible_stall_count
        state.best_feasible = ALMFeasibleIncumbent(
            x=run_state.x.copy(),
            evaluation=current_eval,
            multipliers=state.multipliers.copy(),
            penalty=state.penalty,
            inner_result=state.last_result,
        )
        _publish_outer_step(
            context, state=state, run_state=run_state, action=converge.action
        )
        # The caller's accepted state is snapshotted after on_outer_step
        # (pinned callback order), so it joins the incumbent only now.
        if snapshot_accepted_state_fn is not None:
            state.best_feasible = replace(
                state.best_feasible, incumbent_state=snapshot_accepted_state_fn()
            )
        state.final_eval = current_eval
        state.final_multipliers = state.multipliers.copy()
        state.final_penalty = state.penalty
        return _finalize_continuation_step(
            state,
            _ALMContinuationDecision.RETURN,
            _build_alm_result(
                run_state=run_state,
                constraint_names_tuple=constraint_names_tuple,
                success=True,
                message=converge.message,
                termination_reason=converge.termination_reason,
                outer_iterations=outer_iteration,
                evaluation=current_eval,
                multipliers=state.multipliers,
                penalty=state.penalty,
                inner_result=state.last_result,
                stationarity_norm=start.stationarity_norm,
                kkt_stationarity_norm=start.kkt_stationarity_norm,
            ),
        )

    state.best_feasible = _incumbent_at_step_start(
        state.best_feasible,
        start=start,
        live_x=run_state.x,
        problem_refreshed=problem_refreshed,
        last_result=state.last_result,
        settings=settings,
        evaluate_problem=evaluate_problem,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        snapshot_accepted_state_fn=snapshot_accepted_state_fn,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
    )
    inner_attempt = _run_alm_inner_attempts(
        ALMInnerAttemptRequest(
            x=run_state.x,
            current_eval=current_eval,
            multipliers=state.multipliers,
            penalty_argument=penalty_argument,
            evaluate_problem=evaluate_problem,
            inner_options=inner_options,
            settings=settings,
            continuation_iteration=continuation_iteration,
            trust_radius=run_state.trust_radius,
            base_bounds=base_bounds,
            current_max_feasibility_violation=start.max_feasibility_violation,
            update_feasibility_tol=state.update_feasibility_tol,
            update_stationarity_tol=state.update_stationarity_tol,
            effective_feasibility_tol=effective_feasibility_tol,
            inner_callback=inner_callback,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
            continuation_policy=continuation_policy,
        )
    )

    result = inner_attempt.optimizer_result
    state.last_result = result
    run_state.total_inner_iterations += inner_attempt.iterations
    run_state.x = inner_attempt.x
    run_state.trust_radius = inner_attempt.trust_radius
    state.final_eval = inner_attempt.evaluation
    process_budget_exhausted = False
    if inner_attempt.evaluation is not current_eval and accepted_callback is not None:
        try:
            accepted_callback(run_state.x.copy())
        except ALMProcessBudgetExhausted:
            process_budget_exhausted = True
    # Post-inner routing must use the same clamped feasibility gate as the
    # pre-inner routing above; an unclamped pass would let the active masks
    # diverge within one outer iteration on early ALM steps.
    measured = _measure_iterate(
        state.final_eval,
        x=run_state.x,
        base_bounds=run_state.base_bounds,
        multipliers=state.multipliers,
        penalty=penalty_argument,
        update_feasibility_tol=state.update_feasibility_tol,
        update_stationarity_tol=state.update_stationarity_tol,
        effective_feasibility_tol=effective_feasibility_tol,
    )
    # ALGENCAN safeguard bookkeeping: measure this accepted iterate now (at
    # the multipliers/penalty its subproblem was solved with — before any
    # dual-update arm mutates them), remember the previous accepted
    # iterate's measure as the sufficient-decrease reference, and roll the
    # carrier forward. The first step ever has no predecessor and references
    # its own pre-inner (seed) measure. The measure is published with the
    # step so a checkpoint resume can rebuild the carried reference; without
    # it every resume restarts from the pre-inner fallback, which is
    # bitwise-degenerate for per-outer-refreshed constraints.
    post_sufficient_decrease_measure = _sufficient_decrease_measure(
        measured.routing_state.signal_state.hard_signed_constraint_values,
        state.multipliers,
        penalty_argument,
    )
    sufficient_decrease_reference = run_state.previous_sufficient_decrease_measure
    if sufficient_decrease_reference is None:
        sufficient_decrease_reference = _sufficient_decrease_measure(
            start.routing_state.signal_state.hard_signed_constraint_values,
            state.multipliers,
            penalty_argument,
        )
    run_state.previous_sufficient_decrease_measure = post_sufficient_decrease_measure
    run_state.previous_sufficient_decrease_multipliers = state.multipliers.copy()
    run_state.previous_sufficient_decrease_penalty = float(penalty_argument)
    feasibility_delta, feasibility_delta_tol = _feasibility_improvement_and_floor(
        start.max_feasibility_violation,
        measured.max_feasibility_violation,
    )
    state.final_multipliers = state.multipliers.copy()
    state.final_penalty = state.penalty
    made_inner_progress = _made_meaningful_inner_progress(
        start_x,
        run_state.x,
        float(current_eval["total"]),
        float(state.final_eval["total"]),
        start.max_feasibility_violation,
        measured.max_feasibility_violation,
        start.stationarity_norm,
        measured.stationarity_norm,
    )

    state.best_feasible = _improved_incumbent(
        state.best_feasible,
        x=run_state.x,
        measured=measured,
        inner_result=result,
        settings=settings,
        snapshot_accepted_state_fn=snapshot_accepted_state_fn,
    )

    # Every inner plan re-derives the staged values (gtol, profile caps)
    # from this step's ``inner_options`` (the caller's, with the call's
    # remaining maxiter); the published ``inner.inner_options`` are the
    # options the inner solve actually used.
    context = replace(
        context,
        inner=_inner_solve_outcome(
            inner_attempt,
            measured=measured,
            meaningful_progress=bool(made_inner_progress),
            feasibility_delta=float(feasibility_delta),
            feasibility_delta_tolerance=float(feasibility_delta_tol),
            sufficient_decrease_measure=float(post_sufficient_decrease_measure),
            sufficient_decrease_reference=float(sufficient_decrease_reference),
        ),
    )
    if process_budget_exhausted:
        return _emit_alm_stall_failure_step(
            context,
            state=state,
            run_state=run_state,
            action="process_budget_exhausted",
            termination_reason="process_budget_exhausted",
            message_prefix=(
                "ALM stopped after exhausting the this-process "
                "accepted-iteration budget"
            ),
            inner_result=result,
            outer_termination="process_budget_exhausted",
        )
    return _execute_step_decision(
        context,
        continuation_policy.after_inner(
            ALMPostInnerView(
                settings=settings,
                outer_iteration=outer_iteration,
                continuation_iteration=continuation_iteration,
                start=pre_inner_view.start,
                inner=_borrowed_read_only_value(context.inner, policy_memo),
                last_cap_binding_active=pre_inner_view.last_cap_binding_active,
                feasible_stall_count=pre_inner_view.feasible_stall_count,
                trust_radius=run_state.trust_radius,
            )
        ),
        state=state,
        run_state=run_state,
    )

def _finalize_outer_iteration(
    step: _ALMContinuationStepResult[AcceptedStateT],
    decision: _ALMOuterDecision,
    result: Optional[object],
) -> _ALMOuterIterationResult[AcceptedStateT]:
    """Package the latest continuation-step carrier as the frozen outer
    iteration result. The 11 sticky fields are sourced directly from the
    continuation step rather than re-listed at every outer return.
    """
    return _ALMOuterIterationResult(
        decision=decision,
        result=result,
        multipliers=step.multipliers,
        penalty=step.penalty,
        update_feasibility_tol=step.update_feasibility_tol,
        update_stationarity_tol=step.update_stationarity_tol,
        last_result=step.last_result,
        final_eval=step.final_eval,
        final_multipliers=step.final_multipliers,
        final_penalty=step.final_penalty,
        best_feasible=step.best_feasible,
    )

def _run_alm_outer_iteration(
    *,
    settings: ALMSettings,
    run_state: ALMRunState,
    multipliers: np.ndarray,
    penalty: float,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    last_result: Optional[object],
    final_eval: Optional[dict],
    final_multipliers: np.ndarray,
    final_penalty: float,
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    inner_options: Optional[dict],
    call_start_inner_iterations: int,
    outer_iteration: int,
    is_final_outer: bool,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    inner_callback: Optional[Callable[[np.ndarray], None]],
    accepted_callback: Optional[Callable[[np.ndarray], None]],
    outer_state_callback: Optional[Callable[[int, np.ndarray, float], None]],
    on_outer_step: Optional[Callable[[ALMOuterStepEvent[AcceptedStateT]], None]],
    snapshot_accepted_state_fn: Optional[Callable[[], AcceptedStateT]],
    restore_incumbent_state_fn: Optional[Callable[[AcceptedStateT], None]],
    constraint_names_tuple: Tuple[str, ...],
    constraint_blocks_tuple: Optional[Tuple[str, ...]],
    base_bounds=None,
    continuation_policy: ALMContinuationPolicy = DEFAULT_CONTINUATION_POLICY,
) -> _ALMOuterIterationResult[AcceptedStateT]:
    if outer_state_callback is not None:
        outer_state_callback(outer_iteration, multipliers.copy(), penalty)
    feasible_stall_count = 0
    for continuation_iteration in range(settings.max_subproblem_continuations + 1):
        step = _run_alm_continuation_step(
            settings=settings,
            run_state=run_state,
            multipliers=multipliers,
            penalty=penalty,
            update_feasibility_tol=update_feasibility_tol,
            update_stationarity_tol=update_stationarity_tol,
            feasible_stall_count=feasible_stall_count,
            last_result=last_result,
            final_eval=final_eval,
            final_multipliers=final_multipliers,
            final_penalty=final_penalty,
            best_feasible=best_feasible,
            inner_options=_call_inner_options(
                inner_options, run_state, call_start_inner_iterations
            ),
            outer_iteration=outer_iteration,
            continuation_iteration=continuation_iteration,
            is_final_outer=is_final_outer,
            evaluate_problem=evaluate_problem,
            inner_callback=inner_callback,
            accepted_callback=accepted_callback,
            on_outer_step=on_outer_step,
            snapshot_accepted_state_fn=snapshot_accepted_state_fn,
            restore_incumbent_state_fn=restore_incumbent_state_fn,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
            problem_refreshed=outer_state_callback is not None and continuation_iteration == 0,
            base_bounds=base_bounds,
            continuation_policy=continuation_policy,
        )
        multipliers = step.multipliers
        penalty = step.penalty
        update_feasibility_tol = step.update_feasibility_tol
        update_stationarity_tol = step.update_stationarity_tol
        feasible_stall_count = step.feasible_stall_count
        last_result = step.last_result
        final_eval = step.final_eval
        final_multipliers = step.final_multipliers
        final_penalty = step.final_penalty
        best_feasible = step.best_feasible
        if step.decision == _ALMContinuationDecision.RETURN:
            return _finalize_outer_iteration(
                step, _ALMOuterDecision.RETURN, step.result
            )
        if step.decision == _ALMContinuationDecision.BREAK_OUTER:
            break
        if step.decision != _ALMContinuationDecision.CONTINUE_CONTINUATION:
            raise AssertionError(f"unhandled continuation decision {step.decision!r}")
        # A spent budget ends the run before the outer's next step.
        if continuation_iteration < settings.max_subproblem_continuations and (
            _inner_budget_spent(
                _call_inner_options(inner_options, run_state, call_start_inner_iterations)
            )
        ):
            return _finalize_outer_iteration(step, _ALMOuterDecision.BUDGET_SPENT, None)
    # The continuation range is never empty (max_subproblem_continuations >= 0),
    # so ``step`` holds the latest carrier.
    exhaust_decision = (
        _ALMOuterDecision.EXHAUST if is_final_outer else _ALMOuterDecision.NEXT_OUTER
    )
    return _finalize_outer_iteration(step, exhaust_decision, None)

def _publish_outer_boundary(
    on_outer_boundary: Optional[Callable[[ALMOuterBoundary[AcceptedStateT]], None]],
    *,
    run_state: ALMRunState,
    completed_outer_iterations: int,
    multipliers: np.ndarray,
    penalty: float,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    constraint_names_tuple: Tuple[str, ...],
    constraint_blocks_tuple: Optional[Tuple[str, ...]],
    snapshot_accepted_state_fn: Optional[Callable[[], AcceptedStateT]],
    result: Optional[ALMResult] = None,
) -> None:
    """Pass the settled loop to ``on_outer_boundary`` (built only when set).

    With the returned ``result`` the boundary is terminal: its x, multipliers,
    penalty and termination reason replace the loop's.
    """
    if on_outer_boundary is None:
        return
    x = run_state.x
    termination_reason = None
    if result is not None:
        x = result.x
        multipliers = result.multipliers
        penalty = result.penalty
        termination_reason = str(result.termination_reason)
    accepted_state = (
        None if snapshot_accepted_state_fn is None else snapshot_accepted_state_fn()
    )
    on_outer_boundary(
        ALMOuterBoundary(
            completed_outer_iterations=int(completed_outer_iterations),
            completed_action=(
                "outer_completed"
                if run_state.last_action is None
                else run_state.last_action
            ),
            termination_reason=termination_reason,
            constraint_names=constraint_names_tuple,
            constraint_blocks=constraint_blocks_tuple,
            accepted_state=accepted_state,
            geometry_identity=_accepted_state_geometry_identity(accepted_state),
            state=_frozen_loop_state(
                _run_loop_state(
                    run_state,
                    x=x,
                    multipliers=multipliers,
                    penalty=penalty,
                    update_feasibility_tol=update_feasibility_tol,
                    update_stationarity_tol=update_stationarity_tol,
                    best_feasible=best_feasible,
                ),
                {},
            ),
        )
    )

def minimize_alm(
    x0,
    constraint_names: Sequence[str],
    evaluate_problem: ALMEvaluator,
    settings: ALMSettings,
    inner_options: dict,
    inner_callback: Optional[Callable[[np.ndarray], None]] = None,
    accepted_callback: Optional[Callable[[np.ndarray], None]] = None,
    outer_state_callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    snapshot_accepted_state_fn: Optional[Callable[[], AcceptedStateT]] = None,
    restore_incumbent_state_fn: Optional[Callable[[AcceptedStateT], None]] = None,
    initial_multipliers: Optional[np.ndarray] = None,
    initial_penalty: Optional[float] = None,
    constraint_blocks: Optional[Sequence[str]] = None,
    base_bounds=None,
    resume_from: Optional[ALMOuterBoundary[AcceptedStateT]] = None,
    on_outer_step: Optional[Callable[[ALMOuterStepEvent[AcceptedStateT]], None]] = None,
    on_outer_boundary: Optional[Callable[[ALMOuterBoundary[AcceptedStateT]], None]] = None,
    continuation_policy: ALMContinuationPolicy = DEFAULT_CONTINUATION_POLICY,
) -> ALMResult:
    if resume_from is not None and (
        initial_multipliers is not None or initial_penalty is not None
    ):
        raise ValueError(
            "resume_from cannot be combined with initial_multipliers or initial_penalty"
        )
    normalized = _normalize_alm_run_inputs(
        x0,
        constraint_names,
        settings,
        initial_multipliers,
        initial_penalty,
        constraint_blocks,
        snapshot_accepted_state_fn,
        restore_incumbent_state_fn,
        base_bounds,
    )
    multipliers = normalized.multipliers
    penalty = normalized.penalty
    constraint_names_tuple = normalized.constraint_names_tuple
    constraint_blocks_tuple = normalized.constraint_blocks_tuple
    update_feasibility_tol = normalized.update_feasibility_tol
    update_stationarity_tol = normalized.update_stationarity_tol
    final_eval: Optional[dict] = None
    last_result: Optional[object] = None
    final_multipliers = multipliers.copy()
    final_penalty = penalty
    last_outer_iteration = 0
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]] = None
    first_outer_iteration = 1

    if resume_from is not None:
        _validate_resume_boundary(resume_from)
        resumed = resume_from.state
        resume_x = np.asarray(resumed.x, dtype=float)
        # A checkpoint of a run with these bounds lies inside them; one
        # outside came from other bounds, and projecting it would not
        # continue the run that wrote it.
        if not np.array_equal(_project_onto_bounds(resume_x, normalized.base_bounds), resume_x):
            raise ValueError("resume_from.state.x lies outside base_bounds")
        if not np.array_equal(normalized.x, resume_x):
            raise ValueError("x0 does not match resume_from.state.x")
        if resume_from.constraint_names != constraint_names_tuple:
            raise ValueError(
                "resume_from constraint_names do not match current constraint_names"
            )
        if (resume_from.constraint_blocks is None) != (
            constraint_blocks_tuple is None
        ) or resume_from.constraint_blocks != constraint_blocks_tuple:
            raise ValueError(
                "resume_from constraint_blocks do not match current constraint_blocks"
            )
        completed_outer = int(resume_from.completed_outer_iterations)
        if completed_outer < 0 or completed_outer >= settings.max_outer_iterations:
            raise ValueError(
                "resume_from completed_outer_iterations must be less than "
                "settings.max_outer_iterations"
            )
        if resume_from.accepted_state is not None:
            if restore_incumbent_state_fn is None:
                raise ValueError(
                    "resume_from carries accepted_state but no restore callback was provided"
                )
            # Live identity is restore_fn's contract; the boundary is unchanged.
            restore_incumbent_state_fn(resume_from.accepted_state)
        multipliers = np.asarray(resumed.multipliers, dtype=float).copy()
        penalty = _require_alm_penalty_within_cap(
            resumed.penalty,
            settings,
            label="resume ALM penalty",
        )
        final_multipliers = multipliers.copy()
        final_penalty = penalty
        update_feasibility_tol = float(resumed.update_feasibility_tol)
        update_stationarity_tol = float(resumed.update_stationarity_tol)
        run_state = ALMRunState(
            x=resume_x.copy(),
            total_inner_iterations=int(resumed.total_inner_iterations),
            trust_radius=resumed.trust_radius,
            cap_binding_detected=bool(resumed.cap_binding_detected),
            cap_binding_indices=set(resumed.cap_binding_indices),
            penalty_cap_reached=bool(resumed.penalty_cap_reached),
            penalty_cap_requested=resumed.penalty_cap_requested,
            last_cap_binding_active=bool(resumed.last_cap_binding_active),
            previous_sufficient_decrease_measure=resumed.sufficient_decrease_measure,
            previous_sufficient_decrease_multipliers=(
                None
                if resumed.sufficient_decrease_multipliers is None
                else np.asarray(
                    resumed.sufficient_decrease_multipliers, dtype=float
                ).copy()
            ),
            previous_sufficient_decrease_penalty=resumed.sufficient_decrease_penalty,
            # A resumable boundary follows a non-final outer, whose exhausted
            # termination is the action it published (``completed_action``).
            last_action=resume_from.completed_action,
            exhausted_termination=resume_from.completed_action,
            base_bounds=normalized.base_bounds,
        )
        best_feasible = resumed.best_feasible
        first_outer_iteration = completed_outer + 1
        last_outer_iteration = completed_outer
    else:
        run_state = ALMRunState(
            x=normalized.x,
            total_inner_iterations=0,
            trust_radius=normalized.trust_radius,
            cap_binding_detected=False,
            cap_binding_indices=set(),
            penalty_cap_reached=False,
            penalty_cap_requested=None,
            base_bounds=normalized.base_bounds,
        )

    inner_iterations_at_start = int(run_state.total_inner_iterations)
    # Whether the run ends because the call's maxiter budget is spent; then
    # its termination reason is the latest step's action.
    budget_spent = False

    for outer_iteration in range(
        first_outer_iteration, settings.max_outer_iterations + 1
    ):
        is_final_outer = outer_iteration == settings.max_outer_iterations
        # Only a fresh call's first outer starts on a spent budget (maxiter=0).
        if outer_iteration > 1 and _inner_budget_spent(
            _call_inner_options(inner_options, run_state, inner_iterations_at_start)
        ):
            budget_spent = True
            break
        # An outer counts once it runs a step (the result's outer_iterations
        # and the boundaries' completed count).
        last_outer_iteration = outer_iteration
        outcome = _run_alm_outer_iteration(
            settings=settings,
            run_state=run_state,
            multipliers=multipliers,
            penalty=penalty,
            update_feasibility_tol=update_feasibility_tol,
            update_stationarity_tol=update_stationarity_tol,
            last_result=last_result,
            final_eval=final_eval,
            final_multipliers=final_multipliers,
            final_penalty=final_penalty,
            best_feasible=best_feasible,
            inner_options=inner_options,
            call_start_inner_iterations=inner_iterations_at_start,
            outer_iteration=outer_iteration,
            is_final_outer=is_final_outer,
            evaluate_problem=evaluate_problem,
            inner_callback=inner_callback,
            accepted_callback=accepted_callback,
            outer_state_callback=outer_state_callback,
            on_outer_step=on_outer_step,
            snapshot_accepted_state_fn=snapshot_accepted_state_fn,
            restore_incumbent_state_fn=restore_incumbent_state_fn,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
            base_bounds=normalized.base_bounds,
            continuation_policy=continuation_policy,
        )
        multipliers = outcome.multipliers
        penalty = outcome.penalty
        update_feasibility_tol = outcome.update_feasibility_tol
        update_stationarity_tol = outcome.update_stationarity_tol
        last_result = outcome.last_result
        final_eval = outcome.final_eval
        final_multipliers = outcome.final_multipliers
        final_penalty = outcome.final_penalty
        best_feasible = outcome.best_feasible
        if outcome.decision == _ALMOuterDecision.RETURN:
            _publish_outer_boundary(
                on_outer_boundary,
                run_state=run_state,
                completed_outer_iterations=outer_iteration,
                multipliers=multipliers,
                penalty=penalty,
                update_feasibility_tol=update_feasibility_tol,
                update_stationarity_tol=update_stationarity_tol,
                best_feasible=best_feasible,
                constraint_names_tuple=constraint_names_tuple,
                constraint_blocks_tuple=constraint_blocks_tuple,
                snapshot_accepted_state_fn=snapshot_accepted_state_fn,
                result=outcome.result,
            )
            return outcome.result
        if outcome.decision == _ALMOuterDecision.EXHAUST:
            break
        if outcome.decision == _ALMOuterDecision.BUDGET_SPENT:
            budget_spent = True
            break
        if outcome.decision != _ALMOuterDecision.NEXT_OUTER:
            raise AssertionError(f"unhandled outer decision {outcome.decision!r}")
        _publish_outer_boundary(
            on_outer_boundary,
            run_state=run_state,
            completed_outer_iterations=outer_iteration,
            multipliers=multipliers,
            penalty=penalty,
            update_feasibility_tol=update_feasibility_tol,
            update_stationarity_tol=update_stationarity_tol,
            best_feasible=best_feasible,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
            snapshot_accepted_state_fn=snapshot_accepted_state_fn,
        )

    if final_eval is None:
        # Only a resume whose budget was spent at its boundary ends without an
        # outer. The uninterrupted run's final evaluation was at this x with
        # these multipliers and penalty; there is no inner result.
        final_eval = _checked_evaluation(
            evaluate_problem,
            run_state.x,
            final_multipliers,
            final_penalty,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
            context="ALM resumed final state evaluation",
        )

    final_result = _build_alm_failure_result_with_optional_restore(
        settings=settings,
        constraint_names_tuple=constraint_names_tuple,
        run_state=run_state,
        last_outer_iteration=last_outer_iteration,
        best_feasible=best_feasible,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        termination_reason=(
            run_state.last_action if budget_spent else run_state.exhausted_termination
        ),
        evaluation=final_eval,
        multipliers_state=final_multipliers,
        penalty_state=final_penalty,
        inner_result=last_result,
        message_prefix=(
            "ALM spent the inner maxiter budget"
            if budget_spent
            else "ALM exhausted outer iterations (max outer iterations reached)"
        ),
        restored_message_prefix=(
            "ALM spent the inner maxiter budget after restoring best feasible iterate"
            if budget_spent
            else "ALM exhausted outer iterations after restoring best feasible "
            "iterate (max outer iterations reached)"
        ),
        restored_termination_reason="max_outer_restored_best_feasible",
    )
    _publish_outer_boundary(
        on_outer_boundary,
        run_state=run_state,
        completed_outer_iterations=last_outer_iteration,
        multipliers=final_multipliers,
        penalty=final_penalty,
        update_feasibility_tol=update_feasibility_tol,
        update_stationarity_tol=update_stationarity_tol,
        best_feasible=best_feasible,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        snapshot_accepted_state_fn=snapshot_accepted_state_fn,
        result=final_result,
    )
    return final_result
