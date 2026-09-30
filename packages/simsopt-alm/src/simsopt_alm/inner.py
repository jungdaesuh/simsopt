"""One inner subproblem solve of the ALM loop: L-BFGS-B attempts in a trust box.

:func:`_run_alm_inner_attempts` minimizes the augmented Lagrangian at fixed
multipliers and penalty from the loop's current iterate
(:class:`ALMInnerAttemptRequest`) and returns the accepted iterate, or the
start iterate when no attempt produced a usable step
(:class:`ALMInnerAttemptResult`). It hides how: the evaluator
cache and its early stop once the dual-update gate holds, the elevated
fallback for a non-finite trial, the trust box (intersected with the
caller's bounds), the acceptance and infeasible-stall tests with their
tolerances, the one shrink/grow trust rule, the policy's inner plan and
stalled-trial retry, and this process's ``maxiter``. The loop also reads
two judgments that share those tolerances:
:func:`_feasibility_improvement_and_floor` and
:func:`_made_meaningful_inner_progress`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .continuation import ALMInnerPlanView, ALMStalledTrialView
from .core import (
    ALMSettings,
    _bound_reduced_stationarity_norm,
    _constraint_routing_state,
    _extract_constraint_state,
    _nonnegative_alm_integer,
)
from .evaluation import (
    _attach_alm_constraint_metadata,
    _contract_checked_evaluation,
    _unusable_evaluation_fields,
    _search_step_rejected,
)
from .events import _borrowed_read_only_value, _require_acyclic_containers
from .policy import (
    DEFAULT_CONTINUATION_POLICY,
    ALMContinuationPolicy,
    _dual_update_gate_satisfied,
    _positive_alm_integer,
)

def _inner_options_with_remaining_maxiter(
    inner_options: Optional[dict],
    consumed: int,
) -> Optional[dict]:
    """Return a copy whose maxiter is this-process remaining inner budget.

    Remaining 0 is a legal exhausted budget. It must not reach the inner
    plan, whose options require a positive scipy maxiter.
    """
    if inner_options is None or "maxiter" not in inner_options:
        return inner_options
    process_maxiter = _nonnegative_alm_integer(
        "inner_options.maxiter",
        inner_options["maxiter"],
    )
    remaining_options = dict(inner_options)
    remaining_options["maxiter"] = max(process_maxiter - int(consumed), 0)
    return remaining_options

def _exhausted_inner_optimizer_result(x: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray(x, dtype=float).copy(),
        nit=0,
        success=True,
        message="inner maxiter exhausted",
    )

@dataclass(frozen=True)
class ALMInnerAttemptRequest:
    x: np.ndarray
    current_eval: dict
    multipliers: np.ndarray
    penalty_argument: float
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict]
    inner_options: dict
    settings: ALMSettings
    continuation_iteration: int
    trust_radius: Optional[float]
    current_max_feasibility_violation: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    effective_feasibility_tol: float
    inner_callback: Optional[Callable[[np.ndarray], None]]
    constraint_names_tuple: Tuple[str, ...]
    constraint_blocks_tuple: Optional[Tuple[str, ...]]
    base_bounds: Optional[object] = None
    continuation_policy: ALMContinuationPolicy = DEFAULT_CONTINUATION_POLICY

@dataclass(frozen=True)
class ALMInnerAttemptResult:
    optimizer_result: object
    evaluation: dict
    x: np.ndarray
    bounds: object
    attempts: int
    iterations: int
    trust_radius: Optional[float]
    last_inner_options: Optional[dict]
    last_inner_profile: Optional[str]
    forced_infeasible_penalty_cycle: bool
    forced_infeasible_penalty_reason: Optional[str]
    forced_inner_false_success: bool
    nonfinite_candidate_evaluation: bool
    nonfinite_candidate_fields: Optional[List[str]]

@dataclass
class _ALMInnerAttemptEvaluator:
    request: ALMInnerAttemptRequest
    cached_x: Optional[np.ndarray] = None
    cached_evaluation: Optional[dict] = None
    # The request's base bounds as (lower, upper) pairs, for stationarity.
    base_bounds: Optional[List[Tuple[float, float]]] = None
    # L-BFGS-B iterations completed in the current attempt: SciPy calls
    # ``callback`` once per completed iteration.
    attempt_iterations: int = 0

    def _fresh_evaluation(self, x) -> dict:
        return _sanitize_nonfinite_inner_evaluation(
            _contract_checked_evaluation(
                self.request.evaluate_problem(
                    x,
                    self.request.multipliers,
                    self.request.penalty_argument,
                ),
                x=x,
                constraint_count=len(self.request.constraint_names_tuple),
                context="ALM inner trial evaluation",
            ),
            fallback_evaluation=self.request.current_eval,
        )

    def evaluation_at(self, x: np.ndarray) -> dict:
        """Sanitized evaluation at ``x``; reuses the last ``fun`` evaluation."""
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return self.cached_evaluation
        return self._fresh_evaluation(x)

    def placed_on_bounds(self, x: np.ndarray, evaluation: dict):
        """``(x, evaluation)`` of an attempt's result: ``x`` clipped into the
        base box, and snapped onto the bounds it lies within rounding of
        (:func:`_snap_onto_bounds`) unless the snap raises the total
        (:func:`_snap_keeps_total`); a moved x is evaluated there."""
        clipped = _project_onto_bounds(x, self.base_bounds)
        snapped = _snap_onto_bounds(x, evaluation["grad"], self.base_bounds)
        if not np.array_equal(snapped, clipped):
            snapped_evaluation = self.evaluation_at(snapped)
            if _snap_keeps_total(evaluation, snapped_evaluation):
                return snapped, snapped_evaluation
        if np.array_equal(clipped, x):
            return x, evaluation
        return clipped, self.evaluation_at(clipped)

    def fun(self, inner_x):
        evaluation = self._fresh_evaluation(inner_x)
        self.cached_x = np.asarray(inner_x, dtype=float).copy()
        self.cached_evaluation = evaluation
        grad = np.asarray(evaluation["grad"], dtype=float)
        return float(evaluation["total"]), grad.copy()

    def callback(self, inner_x):
        self.attempt_iterations += 1
        inner_x_arr = np.asarray(inner_x, dtype=float).copy()
        if self.request.inner_callback is not None:
            # The callback receives an owned snapshot.  Its mutation must not
            # alter the candidate subsequently evaluated by ALM.
            self.request.inner_callback(inner_x_arr.copy())
        evaluation = self.evaluation_at(inner_x_arr)
        if _search_step_rejected(evaluation):
            return
        (
            _solver_constraint_values,
            _callback_feasibility_values,
            _callback_dual_update_values,
            callback_max_feasibility_violation,
        ) = _extract_constraint_state(evaluation)
        callback_routing_state = _constraint_routing_state(
            evaluation,
            self.request.multipliers,
            self.request.penalty_argument,
            self.request.effective_feasibility_tol,
        )
        callback_stationarity_norm = _bound_reduced_stationarity_norm(
            evaluation, inner_x_arr, self.base_bounds
        )
        if callback_routing_state.signal_state.explicit_hybrid_signals:
            if _dual_update_gate_satisfied(
                max_feasibility_violation=callback_max_feasibility_violation,
                hard_max_violation=callback_routing_state.hard_max_violation,
                stationarity_norm=callback_stationarity_norm,
                update_feasibility_tol=self.request.update_feasibility_tol,
                update_stationarity_tol=self.request.update_stationarity_tol,
            ):
                raise _EarlyStopInnerSolve(
                    inner_x, evaluation, self.attempt_iterations
                )
        elif (
            callback_max_feasibility_violation <= self.request.effective_feasibility_tol
            and callback_stationarity_norm <= self.request.update_stationarity_tol
        ):
            raise _EarlyStopInnerSolve(inner_x, evaluation, self.attempt_iterations)

# Feasibility slack of an accepted candidate, in the rows' units.
_ACCEPTANCE_TOTAL_ATOL = 1e-10

# A candidate's total may exceed its start's by this fraction of the step's
# first-order scale (see _total_tolerance).
_ACCEPTANCE_TOTAL_RTOL = 1e-3

_ACCEPTANCE_MOVE_TOL = 1e-12

_INFEASIBLE_STALL_MOVE_TOL = 1e-8

_INFEASIBLE_STALL_FEASIBILITY_ATOL = 1e-12

_INFEASIBLE_STALL_FEASIBILITY_RTOL = 1e-6

# Floors of a stationarity-norm drop that counts as progress.
_INFEASIBLE_STALL_OBJECTIVE_ATOL = 1e-10

# An objective drop counts as progress (not a stall) when it beats this
# fraction of the step's first-order scale (_total_tolerance), and a
# stationarity drop when it beats this fraction of the norm.
_INFEASIBLE_STALL_OBJECTIVE_RTOL = 1e-6

class _EarlyStopInnerSolve(RuntimeError):
    """Raised by the inner callback to stop L-BFGS-B at the KKT gate, with the
    iterate, its evaluation and the iterations the attempt completed."""

    def __init__(self, x, evaluation: dict, completed_iterations: int):
        super().__init__("ALM inner solve satisfied the KKT stationarity gate.")
        self.x = np.asarray(x, dtype=float).copy()
        self.evaluation = evaluation
        self.completed_iterations = int(completed_iterations)

def _total_tolerance(gradient_norm: float, step_norm: float, rtol: float) -> float:
    """``rtol`` of a step's first-order scale ``||grad L|| ||dx||`` (the
    largest change its linear model can make, from the start's gradient): a
    tolerance on a change of the total that an offset added to f leaves
    unchanged and that scales by c when f does. There is no round-off
    allowance: the reported totals decide, and evaluation noise that hides a
    change stalls the run rather than passing it."""
    return float(rtol) * float(gradient_norm) * float(step_norm)

def _elevated_rejection_total(reference_total: float) -> float:
    """The total a nonfinite trial shows L-BFGS-B, above the reference so its
    line search backs off. Its size scales with |f|, so an offset added to f
    changes that backtracking path; it decides nothing, as the candidate is
    rejected by its ``nonfinite_evaluation`` flag."""
    return (
        float(reference_total)
        + max(abs(float(reference_total)), 1.0)
        + _ACCEPTANCE_TOTAL_ATOL
    )

def _sanitize_nonfinite_inner_evaluation(
    evaluation: dict,
    *,
    fallback_evaluation: dict,
) -> dict:
    """``evaluation`` (the solver's snapshot) when every field the loop
    reads is finite and the evaluator did not flag it
    ``nonfinite_evaluation``; otherwise the solver-owned
    ``fallback_evaluation`` with an elevated total
    (:func:`_elevated_rejection_total`), flagged ``nonfinite_evaluation``
    with the unusable fields (:func:`_unusable_evaluation_fields`)."""
    invalid_fields = _unusable_evaluation_fields(evaluation)
    if not invalid_fields:
        return evaluation

    sanitized = dict(fallback_evaluation)
    sanitized["total"] = _elevated_rejection_total(float(fallback_evaluation["total"]))
    sanitized["nonfinite_evaluation"] = True
    sanitized["nonfinite_fields"] = list(invalid_fields)
    return sanitized

def _move_tolerance(reference_x) -> float:
    reference_norm = float(np.linalg.norm(np.asarray(reference_x, dtype=float)))
    return _INFEASIBLE_STALL_MOVE_TOL * max(1.0, reference_norm)

def _made_meaningful_inner_progress(
    start_x: np.ndarray,
    end_x: np.ndarray,
    current_total: float,
    final_total: float,
    current_max_feasibility_violation: float,
    final_max_feasibility_violation: float,
    current_stationarity_norm: float,
    final_stationarity_norm: float,
) -> bool:
    move_norm = float(
        np.linalg.norm(
            np.asarray(end_x, dtype=float) - np.asarray(start_x, dtype=float)
        )
    )
    move_scale = max(1.0, float(np.linalg.norm(np.asarray(start_x, dtype=float))))
    moved = move_norm > 1e-8 * move_scale

    improved_objective = float(current_total) - float(final_total) > _total_tolerance(
        current_stationarity_norm, move_norm, _INFEASIBLE_STALL_OBJECTIVE_RTOL
    )
    improved_stationarity = _improved_beyond_floor(
        current_stationarity_norm,
        final_stationarity_norm,
        _INFEASIBLE_STALL_OBJECTIVE_ATOL,
        _INFEASIBLE_STALL_OBJECTIVE_RTOL,
    )
    improved_feasibility = _improved_beyond_floor(
        current_max_feasibility_violation,
        final_max_feasibility_violation,
        _INFEASIBLE_STALL_FEASIBILITY_ATOL,
        _INFEASIBLE_STALL_FEASIBILITY_RTOL,
    )

    return moved or improved_objective or improved_stationarity or improved_feasibility

def _acceptable_total_upper_bound(
    current_total: float, gradient_norm: float, step_norm: float
) -> float:
    """The largest total a candidate ``step_norm`` from a start with this
    gradient norm may have: the start's plus ``_ACCEPTANCE_TOTAL_RTOL`` of the
    step's first-order scale (:func:`_total_tolerance`)."""
    return float(current_total) + _total_tolerance(
        gradient_norm, step_norm, _ACCEPTANCE_TOTAL_RTOL
    )

def _candidate_is_acceptable(
    current_eval: dict,
    candidate_eval: dict,
    result,
    moved_norm: float,
) -> bool:
    """Whether a trial is a subproblem step at all: usable (not rejected, not
    flagged), a step L-BFGS-B took, and no worse on the total than the start
    (:func:`_acceptable_total_upper_bound`)."""
    if _search_step_rejected(candidate_eval) or candidate_eval.get("nonfinite_evaluation"):
        return False
    if not (
        bool(getattr(result, "success", False))
        or int(getattr(result, "nit", 0)) > 0
        or float(moved_norm) > _ACCEPTANCE_MOVE_TOL
    ):
        return False
    candidate_total = float(candidate_eval["total"])
    return candidate_total <= _acceptable_total_upper_bound(
        float(current_eval["total"]),
        float(np.linalg.norm(current_eval["grad"])),
        moved_norm,
    )

def _within_feasibility_slack(
    current_eval: dict, candidate_eval: dict, update_feasibility_tol: float
) -> bool:
    """Whether the candidate's max violation stays within the scheduled
    slack, max(``update_feasibility_tol``, the start's) plus
    ``_ACCEPTANCE_TOTAL_ATOL``: a trust-region preference, which shrinks a
    box while one is left, never a veto on the subproblem's solution."""
    current_max_feasibility_violation = _extract_constraint_state(current_eval)[3]
    candidate_max_feasibility_violation = _extract_constraint_state(candidate_eval)[3]
    return float(candidate_max_feasibility_violation) <= (
        max(float(update_feasibility_tol), float(current_max_feasibility_violation))
        + _ACCEPTANCE_TOTAL_ATOL
    )

def _improvement_and_floor(
    before, after, atol: float, rtol: float
) -> Tuple[float, float]:
    """Return ``(before - after, max(atol, rtol * max(|before|, |after|, 1)))``.

    The drop counts as progress only when it exceeds the floor.
    """
    before_f, after_f = float(before), float(after)
    return before_f - after_f, max(
        atol, rtol * max(abs(before_f), abs(after_f), 1.0)
    )

def _improved_beyond_floor(before, after, atol: float, rtol: float) -> bool:
    drop, floor = _improvement_and_floor(before, after, atol, rtol)
    return drop > floor

def _feasibility_improvement_and_floor(
    before_violation, after_violation
) -> Tuple[float, float]:
    """The max-violation drop over an inner solve and the floor it must beat
    to count as a feasibility gain (the infeasible-stall tolerances)."""
    return _improvement_and_floor(
        before_violation,
        after_violation,
        _INFEASIBLE_STALL_FEASIBILITY_ATOL,
        _INFEASIBLE_STALL_FEASIBILITY_RTOL,
    )

def _classify_infeasible_inner_stall(
    current_eval: dict,
    candidate_eval: dict,
    result,
    moved_norm: float,
    move_tolerance: float,
    feasibility_gate: float,
) -> Tuple[bool, bool, Optional[str]]:
    # A failed search step is a rejected candidate, not an infeasible stall.
    if _search_step_rejected(candidate_eval):
        return False, False, None
    if float(moved_norm) > float(move_tolerance):
        return False, False, None

    current_max_feasibility_violation = _extract_constraint_state(current_eval)[3]
    candidate_max_feasibility_violation = _extract_constraint_state(candidate_eval)[3]
    if candidate_max_feasibility_violation <= float(feasibility_gate):
        return False, False, None

    if _improved_beyond_floor(
        current_max_feasibility_violation,
        candidate_max_feasibility_violation,
        _INFEASIBLE_STALL_FEASIBILITY_ATOL,
        _INFEASIBLE_STALL_FEASIBILITY_RTOL,
    ):
        return False, False, None
    current_total, candidate_total = float(current_eval["total"]), float(candidate_eval["total"])
    if current_total - candidate_total > _total_tolerance(
        float(np.linalg.norm(current_eval["grad"])),
        moved_norm,
        _INFEASIBLE_STALL_OBJECTIVE_RTOL,
    ):
        return False, False, None

    result_success = bool(getattr(result, "success", False))
    result_message = str(getattr(result, "message", "")).upper()
    if result_success and "RELATIVE REDUCTION OF F" in result_message:
        return True, True, "relative_objective_termination_without_feasibility_gain"
    if result_success:
        return True, True, "successful_inner_solve_without_feasibility_gain"
    return True, False, "failed_inner_solve_without_feasibility_gain"

def _checked_trust_radius(trust_radius: Optional[float]) -> Optional[float]:
    """``trust_radius`` as a float: None means no box; any other radius (the
    settings', a checkpoint's or a policy's) must be finite and positive."""
    if trust_radius is None:
        return None
    radius = float(trust_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"ALM trust radius must be finite and positive (None: no box); got {trust_radius!r}"
        )
    return radius

def _normalize_base_bounds(base_bounds, size: int):
    if base_bounds is None:
        return None
    if hasattr(base_bounds, "lb") and hasattr(base_bounds, "ub"):
        lower_values = np.asarray(base_bounds.lb, dtype=float).reshape(-1)
        upper_values = np.asarray(base_bounds.ub, dtype=float).reshape(-1)
        if lower_values.shape != (size,) or upper_values.shape != (size,):
            raise ValueError("ALM base bounds must match the iterate shape.")
        raw_pairs = list(zip(lower_values, upper_values))
    else:
        raw_pairs = list(base_bounds)
        if len(raw_pairs) != size:
            raise ValueError("ALM base bounds must match the iterate shape.")
    normalized_pairs = []
    for lower_bound, upper_bound in raw_pairs:
        lower_value = -np.inf if lower_bound is None else float(lower_bound)
        upper_value = np.inf if upper_bound is None else float(upper_bound)
        if lower_value > upper_value:
            raise ValueError("ALM base lower bound exceeds upper bound.")
        normalized_pairs.append((lower_value, upper_value))
    return normalized_pairs

def _project_onto_bounds(
    x: np.ndarray, base_bounds: Optional[List[Tuple[float, float]]]
) -> np.ndarray:
    """An owned copy of ``x`` clipped to the ``(lower, upper)`` pairs."""
    if base_bounds is None:
        return x.copy()
    lower, upper = np.asarray(base_bounds, dtype=float).T
    return np.clip(x, lower.reshape(x.shape), upper.reshape(x.shape))

# L-BFGS-B ends a step at x + stp * d without re-projecting, so a coordinate
# it drives onto a bound can stop a few ulps off it, inside or out (up to 47
# ulps in a 4000-problem probe).
_BOUND_SNAP_ULPS = 64

def _snap_onto_bounds(
    x: np.ndarray, grad, base_bounds: Optional[List[Tuple[float, float]]]
) -> np.ndarray:
    """``x`` in the box, and on a finite bound where it lies within
    ``_BOUND_SNAP_ULPS`` of it and ``grad`` pushes out of the box there: the
    rounding of a step that reached the bound. Other coordinates keep their
    value (the stationarity test treats only x == bound as on it)."""
    projected = _project_onto_bounds(x, base_bounds)
    if base_bounds is None:
        return projected
    lower, upper = np.asarray(base_bounds, dtype=float).T
    grad_array = np.asarray(grad, dtype=float).reshape(projected.shape)
    near_upper = (grad_array < 0.0) & (
        upper - projected <= _BOUND_SNAP_ULPS * np.spacing(np.abs(upper))
    )
    near_lower = (grad_array > 0.0) & (
        projected - lower <= _BOUND_SNAP_ULPS * np.spacing(np.abs(lower))
    )
    return np.where(near_upper, upper, np.where(near_lower, lower, projected))

def _snap_keeps_total(unsnapped: dict, snapped: dict) -> bool:
    """Whether a snap onto the bounds keeps its candidate: only if its
    reported total is below the unsnapped one. The snap moves x along the
    outward gradient, so to first order it lowers the total; a reported rise
    means the objective is not smooth on the snap's scale, and a tie means
    the totals cannot tell (e.g. an offset of f past the change's
    resolution), so the unsnapped candidate stays and the run stalls rather
    than certifying the bound. There is no round-off allowance."""
    return float(snapped["total"]) < float(unsnapped["total"])

def _intersect_bounds(trust_bounds, base_bounds):
    if trust_bounds is None:
        return base_bounds
    if base_bounds is None:
        return trust_bounds
    intersected_bounds = []
    for (trust_lower, trust_upper), (base_lower, base_upper) in zip(
        trust_bounds,
        base_bounds,
    ):
        lower_bound = max(float(trust_lower), float(base_lower))
        upper_bound = min(float(trust_upper), float(base_upper))
        if lower_bound > upper_bound:
            raise ValueError("ALM trust-region bounds do not intersect base bounds.")
        intersected_bounds.append((lower_bound, upper_bound))
    return intersected_bounds

def _build_box_bounds(
    center: np.ndarray,
    trust_radius: Optional[float],
    base_bounds=None,
):
    center_array = np.asarray(center, dtype=float)
    normalized_base_bounds = _normalize_base_bounds(
        base_bounds,
        center_array.size,
    )
    checked_trust_radius = _checked_trust_radius(trust_radius)
    if checked_trust_radius is None:
        return normalized_base_bounds
    # This is a lightweight trust-region proxy implemented with L-BFGS-B bounds:
    # each continuation centers a symmetric box around the current iterate.
    widths = checked_trust_radius * np.maximum(1.0, np.abs(center_array))
    trust_bounds = [
        (float(value - width), float(value + width))
        for value, width in zip(center_array, widths)
    ]
    return _intersect_bounds(trust_bounds, normalized_base_bounds)

def _run_alm_inner_attempts(request: ALMInnerAttemptRequest) -> ALMInnerAttemptResult:
    evaluator = _ALMInnerAttemptEvaluator(
        request,
        base_bounds=_normalize_base_bounds(request.base_bounds, request.x.size),
    )
    accepted_result = None
    accepted_eval = None
    accepted_x = None
    accepted_bounds = None
    attempts = 0
    attempt_iterations = 0
    attempt_radius = request.trust_radius
    last_attempt_result = None
    current_feasible_enough = (
        request.current_max_feasibility_violation <= request.effective_feasibility_tol
    )
    last_inner_options = None
    last_inner_profile = None
    forced_infeasible_penalty_cycle = False
    forced_infeasible_penalty_reason = None
    forced_inner_false_success = False
    nonfinite_candidate_evaluation = False
    nonfinite_candidate_fields: Optional[List[str]] = None
    trust_radius = request.trust_radius

    process_inner_maxiter = None
    if "maxiter" in request.inner_options:
        process_inner_maxiter = _nonnegative_alm_integer(
            "inner_options.maxiter",
            request.inner_options["maxiter"],
        )

    for attempt_index in range(1, request.settings.max_inner_attempts + 1):
        attempts = attempt_index
        if process_inner_maxiter is not None:
            remaining_maxiter = process_inner_maxiter - attempt_iterations
            if remaining_maxiter <= 0:
                if last_attempt_result is None:
                    last_attempt_result = _exhausted_inner_optimizer_result(request.x)
                    last_inner_options = dict(request.inner_options)
                    last_inner_options["maxiter"] = 0
                break
            attempt_inner_options = dict(request.inner_options)
            attempt_inner_options["maxiter"] = remaining_maxiter
        else:
            attempt_inner_options = request.inner_options
        plan = request.continuation_policy.inner_plan(
            ALMInnerPlanView(
                settings=request.settings,
                inner_options=_borrowed_read_only_value(attempt_inner_options, {}),
                update_stationarity_tol=request.update_stationarity_tol,
                attempt_radius=_checked_trust_radius(attempt_radius),
                continuation_iteration=request.continuation_iteration,
                start_feasible=current_feasible_enough,
            )
        )
        inner_attempt_options = dict(plan.options)
        if process_inner_maxiter is not None:
            # A plan may lower the call's remaining budget, never exceed it.
            plan_maxiter = inner_attempt_options.get("maxiter", remaining_maxiter)
            inner_attempt_options["maxiter"] = min(
                _positive_alm_integer("inner plan maxiter", plan_maxiter), remaining_maxiter
            )
        attempt_bounds = _build_box_bounds(
            request.x,
            attempt_radius,
            base_bounds=request.base_bounds,
        )
        last_inner_options = dict(inner_attempt_options)
        last_inner_profile = plan.profile
        evaluator.attempt_iterations = 0
        try:
            result = minimize(
                evaluator.fun,
                request.x,
                jac=True,
                method="L-BFGS-B",
                bounds=attempt_bounds,
                callback=evaluator.callback,
                options=inner_attempt_options,
            )
            candidate_x = np.asarray(result.x, dtype=float).copy()
            candidate_eval = evaluator.evaluation_at(candidate_x)
        except _EarlyStopInnerSolve as early_stop:
            result = SimpleNamespace(
                x=early_stop.x,
                nit=early_stop.completed_iterations,
                success=True,
                message=str(early_stop),
            )
            candidate_x = early_stop.x
            candidate_eval = early_stop.evaluation
        candidate_x, candidate_eval = evaluator.placed_on_bounds(candidate_x, candidate_eval)
        last_attempt_result = result
        attempt_iterations += int(getattr(result, "nit", 0))
        moved_norm = float(np.linalg.norm(candidate_x - request.x))
        move_tolerance = _move_tolerance(request.x)
        if candidate_eval.get("nonfinite_evaluation"):
            nonfinite_candidate_evaluation = True
            nonfinite_candidate_fields = list(
                candidate_eval.get("nonfinite_fields", [])
            )
        acceptable = _candidate_is_acceptable(
            request.current_eval, candidate_eval, result, moved_norm
        )
        within_slack = acceptable and _within_feasibility_slack(
            request.current_eval, candidate_eval, request.update_feasibility_tol
        )
        (
            infeasible_inner_stall,
            inner_false_success,
            inner_stall_reason,
        ) = _classify_infeasible_inner_stall(
            request.current_eval,
            candidate_eval,
            result,
            moved_norm,
            move_tolerance,
            request.effective_feasibility_tol,
        )
        # A retry in a smaller box needs a box to shrink, an attempt and
        # inner iterations left to run it.
        smaller_box_left = (
            attempt_radius is not None
            and attempt_radius > request.settings.trust_radius_min
            and attempt_index < request.settings.max_inner_attempts
            and (process_inner_maxiter is None or attempt_iterations < process_inner_maxiter)
        )
        # A step beyond the feasibility slack retries in a smaller box while
        # one is left. Without one (no box, or no retry left to run) the
        # subproblem's solution stands: the outer policy sees its violation
        # and raises the penalty, where a rollback to the start would stall.
        if acceptable and not infeasible_inner_stall and (
            within_slack or not smaller_box_left
        ):
            accepted_result = result
            accepted_eval = _attach_alm_constraint_metadata(
                candidate_eval,
                request.constraint_names_tuple,
                request.constraint_blocks_tuple,
            )
            _require_acyclic_containers(
                accepted_eval,
                context="ALM accepted inner evaluation",
                path="evaluation",
            )
            accepted_x = candidate_x
            accepted_bounds = attempt_bounds
            if attempt_radius is not None:
                # Only a step within the slack that used half the box grows it.
                trust_radius = float(attempt_radius)
                if within_slack and moved_norm >= 0.5 * trust_radius:
                    trust_radius *= float(request.settings.trust_radius_grow)
            break
        if infeasible_inner_stall:
            retry_radius = request.continuation_policy.retry_stalled_trial(
                ALMStalledTrialView(
                    settings=request.settings,
                    attempt_index=attempt_index,
                    attempt_radius=attempt_radius,
                    inner_false_success=bool(inner_false_success),
                )
            )
            if retry_radius is not None:
                attempt_radius = retry_radius
                trust_radius = float(attempt_radius)
                continue
            accepted_result = result
            accepted_eval = request.current_eval
            accepted_x = request.x.copy()
            accepted_bounds = attempt_bounds
            forced_infeasible_penalty_cycle = True
            forced_infeasible_penalty_reason = inner_stall_reason
            forced_inner_false_success = bool(inner_false_success)
            if attempt_radius is not None:
                trust_radius = float(attempt_radius)
            break
        if not smaller_box_left:
            # No usable step (rejected, flagged, or a higher total): keep the
            # start iterate.
            accepted_result = result
            accepted_eval = request.current_eval
            accepted_x = request.x.copy()
            accepted_bounds = attempt_bounds
            if attempt_radius is not None:
                trust_radius = float(attempt_radius)
            break
        attempt_radius = max(
            request.settings.trust_radius_min,
            float(attempt_radius) * float(request.settings.trust_radius_shrink),
        )
        trust_radius = float(attempt_radius)
        continue

    if accepted_result is None or accepted_eval is None or accepted_x is None:
        if last_attempt_result is None:
            raise RuntimeError(
                "ALM failed before any inner optimization result was produced."
            )
        accepted_result = last_attempt_result
        accepted_eval = request.current_eval
        accepted_x = request.x.copy()
        accepted_bounds = None

    return ALMInnerAttemptResult(
        optimizer_result=accepted_result,
        evaluation=accepted_eval,
        x=accepted_x,
        bounds=accepted_bounds,
        attempts=attempts,
        iterations=attempt_iterations,
        trust_radius=trust_radius,
        last_inner_options=last_inner_options,
        last_inner_profile=last_inner_profile,
        forced_infeasible_penalty_cycle=forced_infeasible_penalty_cycle,
        forced_infeasible_penalty_reason=forced_infeasible_penalty_reason,
        forced_inner_false_success=forced_inner_false_success,
        nonfinite_candidate_evaluation=nonfinite_candidate_evaluation,
        nonfinite_candidate_fields=nonfinite_candidate_fields,
    )
