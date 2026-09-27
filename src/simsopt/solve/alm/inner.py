"""One inner subproblem solve of the ALM loop: L-BFGS-B attempts in a trust box.

:func:`_run_alm_inner_attempts` minimizes the augmented Lagrangian at fixed
multipliers and penalty from the loop's current iterate
(:class:`ALMInnerAttemptRequest`) and returns the accepted iterate or the
start iterate (:class:`ALMInnerAttemptResult`). It hides how: the evaluator
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
    _constraint_routing_state,
    _extract_constraint_state,
    _nonnegative_alm_integer,
    _stationarity_metrics,
)
from .evaluation import (
    _OWNED_EVALUATION_ARRAY_FIELDS,
    _attach_alm_constraint_metadata,
    _clone_evaluation_dict,
    _nonfinite_evaluation_fields,
)
from .events import _borrowed_read_only_value, _require_acyclic_containers
from .policy import (
    DEFAULT_CONTINUATION_POLICY,
    ALMContinuationPolicy,
    _dual_update_gate_satisfied,
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

    def _fresh_evaluation(self, x) -> dict:
        return _sanitize_nonfinite_inner_evaluation(
            self.request.evaluate_problem(
                x,
                self.request.multipliers,
                self.request.penalty_argument,
            ),
            fallback_evaluation=self.request.current_eval,
        )

    def evaluation_at(self, x: np.ndarray) -> dict:
        """Sanitized evaluation at ``x``; reuses the last ``fun`` evaluation."""
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return self.cached_evaluation
        return self._fresh_evaluation(x)

    def fun(self, inner_x):
        evaluation = self._fresh_evaluation(inner_x)
        self.cached_x = np.asarray(inner_x, dtype=float).copy()
        # Finite sanitize already returned one owned snapshot. The nonfinite
        # branch copies only the owned fields, so that dict is cloned once.
        if evaluation.get("nonfinite_evaluation"):
            evaluation = _clone_evaluation_dict(
                evaluation,
                owned_keys=_INNER_CACHE_OWNED_KEYS,
            )
        self.cached_evaluation = evaluation
        grad = np.asarray(evaluation["grad"], dtype=float)
        return float(evaluation["total"]), grad.copy()

    def callback(self, inner_x):
        inner_x_arr = np.asarray(inner_x, dtype=float).copy()
        if self.request.inner_callback is not None:
            # The callback receives an owned snapshot.  Its mutation must not
            # alter the candidate subsequently evaluated by ALM.
            self.request.inner_callback(inner_x_arr.copy())
        evaluation = self.evaluation_at(inner_x_arr)
        if evaluation.get("search_step_success") is False:
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
        (
            callback_stationarity_norm,
            callback_kkt_stationarity_norm,
            _callback_signal_mismatch_active,
        ) = _stationarity_metrics(
            evaluation,
            callback_routing_state,
            self.request.effective_feasibility_tol,
        )
        if callback_routing_state.signal_state.explicit_hybrid_signals:
            if _dual_update_gate_satisfied(
                max_feasibility_violation=callback_max_feasibility_violation,
                hard_max_violation=callback_routing_state.hard_max_violation,
                stationarity_norm=callback_stationarity_norm,
                kkt_stationarity_norm=callback_kkt_stationarity_norm,
                update_feasibility_tol=self.request.update_feasibility_tol,
                update_stationarity_tol=self.request.update_stationarity_tol,
            ):
                raise _EarlyStopInnerSolve(inner_x, evaluation)
        elif (
            callback_max_feasibility_violation <= self.request.effective_feasibility_tol
            and callback_stationarity_norm <= self.request.update_stationarity_tol
        ):
            raise _EarlyStopInnerSolve(inner_x, evaluation)

_ACCEPTANCE_TOTAL_ATOL = 1e-10

_ACCEPTANCE_TOTAL_RTOL = 1e-3

_ACCEPTANCE_MOVE_TOL = 1e-12

_INFEASIBLE_STALL_MOVE_TOL = 1e-8

_INFEASIBLE_STALL_FEASIBILITY_ATOL = 1e-12

_INFEASIBLE_STALL_FEASIBILITY_RTOL = 1e-6

_INFEASIBLE_STALL_OBJECTIVE_ATOL = 1e-10

# This rejects roundoff-scale objective movement only; material objective drops
# with fixed feasibility remain progress and must not fall back to the old 5% gate.
_INFEASIBLE_STALL_OBJECTIVE_RTOL = 1e-6

class _EarlyStopInnerSolve(RuntimeError):
    def __init__(self, x, evaluation: dict):
        super().__init__("ALM inner solve satisfied the KKT stationarity gate.")
        self.x = np.asarray(x, dtype=float).copy()
        self.evaluation = evaluation

def _elevated_rejection_total(reference_total: float) -> float:
    return (
        float(reference_total)
        + max(abs(float(reference_total)), 1.0)
        + _ACCEPTANCE_TOTAL_ATOL
    )

_INNER_CACHE_OWNED_KEYS = frozenset(
    (*_OWNED_EVALUATION_ARRAY_FIELDS, "constraint_grads")
)

def _sanitize_nonfinite_inner_evaluation(
    evaluation: dict,
    *,
    fallback_evaluation: dict,
) -> dict:
    invalid_fields = _nonfinite_evaluation_fields(evaluation)
    if not invalid_fields:
        # L4: shallow-copy + per-array clone so the no-invalid-field fast
        # path matches the contract of the sanitized branch (callers always
        # get an owned dict).
        return _clone_evaluation_dict(
            evaluation,
            owned_keys=_INNER_CACHE_OWNED_KEYS,
        )

    sanitized = dict(fallback_evaluation)
    sanitized["total"] = _elevated_rejection_total(float(fallback_evaluation["total"]))
    for field in _OWNED_EVALUATION_ARRAY_FIELDS:
        if field in fallback_evaluation:
            sanitized[field] = np.asarray(
                fallback_evaluation[field], dtype=float
            ).copy()
    if "constraint_grads" in fallback_evaluation:
        sanitized["constraint_grads"] = [
            np.asarray(constraint_grad, dtype=float).copy()
            for constraint_grad in fallback_evaluation["constraint_grads"]
        ]
    for field in ("constraint_names", "constraint_blocks", "constraint_scale_sources"):
        if field in fallback_evaluation:
            sanitized[field] = list(fallback_evaluation[field])
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

    improved_objective = _improved_beyond_floor(
        current_total,
        final_total,
        _INFEASIBLE_STALL_OBJECTIVE_ATOL,
        _INFEASIBLE_STALL_OBJECTIVE_RTOL,
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

def _acceptable_total_upper_bound(current_total: float) -> float:
    total_scale = max(np.finfo(float).eps, abs(float(current_total)))
    return (
        float(current_total)
        + _ACCEPTANCE_TOTAL_ATOL
        + (_ACCEPTANCE_TOTAL_RTOL * total_scale)
    )

def _candidate_is_acceptable(
    current_eval: dict,
    candidate_eval: dict,
    result,
    moved_norm: float,
    update_feasibility_tol: float,
) -> bool:
    if candidate_eval.get("search_step_success") is False:
        return False
    if not (
        bool(getattr(result, "success", False))
        or int(getattr(result, "nit", 0)) > 0
        or float(moved_norm) > _ACCEPTANCE_MOVE_TOL
    ):
        return False

    current_max_feasibility_violation = _extract_constraint_state(current_eval)[3]
    candidate_max_feasibility_violation = _extract_constraint_state(candidate_eval)[3]
    allowed_max_feasibility = (
        max(
            float(update_feasibility_tol),
            float(current_max_feasibility_violation),
        )
        + _ACCEPTANCE_TOTAL_ATOL
    )
    if float(candidate_max_feasibility_violation) > allowed_max_feasibility:
        return False

    candidate_total = float(candidate_eval["total"])
    return candidate_total <= _acceptable_total_upper_bound(
        float(current_eval["total"])
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
    if candidate_eval.get("search_step_success") is False:
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
    if _improved_beyond_floor(
        current_eval["total"],
        candidate_eval["total"],
        _INFEASIBLE_STALL_OBJECTIVE_ATOL,
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

def _normalize_trust_radius(trust_radius: Optional[float]) -> Optional[float]:
    if trust_radius is None:
        return None
    normalized = float(trust_radius)
    if normalized <= 0.0:
        return None
    return normalized

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
    normalized_trust_radius = _normalize_trust_radius(trust_radius)
    if normalized_trust_radius is None:
        return normalized_base_bounds
    # This is a lightweight trust-region proxy implemented with L-BFGS-B bounds:
    # each continuation centers a symmetric box around the current iterate.
    widths = normalized_trust_radius * np.maximum(1.0, np.abs(center_array))
    trust_bounds = [
        (float(value - width), float(value + width))
        for value, width in zip(center_array, widths)
    ]
    return _intersect_bounds(trust_bounds, normalized_base_bounds)

def _run_alm_inner_attempts(request: ALMInnerAttemptRequest) -> ALMInnerAttemptResult:
    evaluator = _ALMInnerAttemptEvaluator(request)
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
                attempt_radius=_normalize_trust_radius(attempt_radius),
                continuation_iteration=request.continuation_iteration,
                start_feasible=current_feasible_enough,
            )
        )
        inner_attempt_options = dict(plan.options)
        attempt_bounds = _build_box_bounds(
            request.x,
            attempt_radius,
            base_bounds=request.base_bounds,
        )
        last_inner_options = dict(inner_attempt_options)
        last_inner_profile = plan.profile
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
                nit=1,
                success=True,
                message=str(early_stop),
            )
            candidate_x = early_stop.x
            candidate_eval = early_stop.evaluation
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
            request.current_eval,
            candidate_eval,
            result,
            moved_norm,
            request.update_feasibility_tol,
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
        if acceptable and not infeasible_inner_stall:
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
                if moved_norm >= 0.5 * float(attempt_radius):
                    trust_radius = float(attempt_radius) * float(
                        request.settings.trust_radius_grow
                    )
                else:
                    trust_radius = float(attempt_radius)
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
        if attempt_radius is None:
            accepted_result = result
            accepted_eval = request.current_eval
            accepted_x = request.x.copy()
            accepted_bounds = attempt_bounds
            break
        next_radius = max(
            request.settings.trust_radius_min,
            float(attempt_radius) * float(request.settings.trust_radius_shrink),
        )
        exhausted_attempts = attempt_index == request.settings.max_inner_attempts
        if attempt_radius <= request.settings.trust_radius_min or exhausted_attempts:
            accepted_result = result
            accepted_eval = request.current_eval
            accepted_x = request.x.copy()
            accepted_bounds = attempt_bounds
            trust_radius = float(attempt_radius)
            break
        attempt_radius = float(next_radius)
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
