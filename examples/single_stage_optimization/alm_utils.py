from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize, nnls


@dataclass(frozen=True)
class ALMSettings:
    max_outer_iterations: int = 10
    max_subproblem_continuations: int = 20
    penalty_init: float = 1.0
    penalty_scale: float = 10.0
    # Production ALM implementations commonly expose a maximum penalty
    # safeguard; keeping this bounded avoids runaway objective scaling in the
    # L-BFGS-B inner solve.
    penalty_max: float | None = 1.0e8
    feasibility_tol: float = 1e-6
    stationarity_tol: float = 1e-6
    trust_radius_init: float | None = None
    trust_radius_min: float = 1e-4
    trust_radius_shrink: float = 0.5
    trust_radius_grow: float = 1.5
    max_inner_attempts: int = 4
    relaxed_feasibility_gate_cap: float = 1e-2
    multiplier_max: float | None = 1.0e6


@dataclass(frozen=True)
class ALMInnerSolveProfile:
    name: str
    maxiter_cap: int | None
    maxls_cap: int | None
    ftol_floor: float | None
    default_maxls: int = 20


@dataclass(frozen=True)
class ALMFeasibleIncumbent:
    x: np.ndarray
    evaluation: dict
    multipliers: np.ndarray
    penalty: float
    inner_result: object
    incumbent_state: object | None = None


@dataclass(frozen=True)
class ALMConstraintSignalState:
    explicit_stage2_signals: bool
    hard_signed_constraint_values: np.ndarray
    hard_violation_values: np.ndarray
    surrogate_signed_constraint_values: np.ndarray
    preferred_dual_update_values: np.ndarray


@dataclass(frozen=True)
class ALMConstraintRoutingState:
    signal_state: ALMConstraintSignalState
    hard_activity_mask: np.ndarray
    surrogate_activity_mask: np.ndarray
    signal_mismatch_active: bool
    hard_positive_shift: np.ndarray
    surrogate_positive_shift: np.ndarray
    hard_positive_shift_zero: bool
    surrogate_positive_shift_zero: bool
    hard_max_violation: float
    surrogate_max_value: float


_UNBOUNDED_INNER_PROFILE = ALMInnerSolveProfile(
    name="unbounded",
    maxiter_cap=None,
    maxls_cap=None,
    ftol_floor=None,
)
_BOXED_INFEASIBLE_INITIAL_PROFILE = ALMInnerSolveProfile(
    name="boxed_infeasible_initial",
    maxiter_cap=150,
    maxls_cap=50,
    ftol_floor=1e-13,
    default_maxls=50,
)
_BOXED_INFEASIBLE_CONTINUATION_PROFILE = ALMInnerSolveProfile(
    name="boxed_infeasible_continuation",
    maxiter_cap=100,
    maxls_cap=50,
    ftol_floor=1e-13,
    default_maxls=50,
)
_BOXED_FEASIBLE_INITIAL_PROFILE = ALMInnerSolveProfile(
    name="boxed_feasible_initial",
    maxiter_cap=100,
    maxls_cap=40,
    ftol_floor=1e-11,
    default_maxls=40,
)
_BOXED_FEASIBLE_CONTINUATION_PROFILE = ALMInnerSolveProfile(
    name="boxed_feasible_continuation",
    maxiter_cap=80,
    maxls_cap=40,
    ftol_floor=1e-11,
    default_maxls=40,
)
_DEFAULT_TAYLOR_EPSILONS = tuple(float(0.5**power) for power in range(7, 13))
_BOXED_INNER_PROFILES = {
    (False, False): _BOXED_INFEASIBLE_INITIAL_PROFILE,
    (False, True): _BOXED_INFEASIBLE_CONTINUATION_PROFILE,
    (True, False): _BOXED_FEASIBLE_INITIAL_PROFILE,
    (True, True): _BOXED_FEASIBLE_CONTINUATION_PROFILE,
}
_TARGET_INNER_OPTION_KEYS = (
    "maxcor",
    "ftol",
    "maxls",
    "maxfun",
    "maxgrad",
    "initial_step_size",
)
_ACCEPTANCE_TOTAL_ATOL = 1e-10
_ACCEPTANCE_TOTAL_RTOL = 1e-3
_ACCEPTANCE_MOVE_TOL = 1e-12
_INFEASIBLE_STALL_MOVE_TOL = 1e-8
_INFEASIBLE_STALL_FEASIBILITY_ATOL = 1e-12
_INFEASIBLE_STALL_FEASIBILITY_RTOL = 1e-6
_INFEASIBLE_STALL_OBJECTIVE_ATOL = 1e-10
_INFEASIBLE_STALL_OBJECTIVE_RTOL = 1e-6
# After two consecutive feasible-but-no-progress outer updates, treat the lane as
# plateaued and stop burning boxed continuation cycles.
_PLATEAU_STALL_LIMIT = 2


class _EarlyStopInnerSolve(RuntimeError):
    def __init__(self, x, evaluation: dict):
        super().__init__("ALM inner solve satisfied the KKT stationarity gate.")
        self.x = np.asarray(x, dtype=float).copy()
        self.evaluation = evaluation


def validate_alm_cli_args(args) -> None:
    if args.alm_max_outer_iters <= 0:
        raise ValueError("--alm-max-outer-iters must be positive")
    max_subproblem_continuations = getattr(args, "alm_max_subproblem_continuations", None)
    if max_subproblem_continuations is not None and max_subproblem_continuations <= 0:
        raise ValueError("--alm-max-subproblem-continuations must be positive")
    if args.alm_penalty_init <= 0.0:
        raise ValueError("--alm-penalty-init must be positive")
    if args.alm_penalty_scale <= 1.0:
        raise ValueError("--alm-penalty-scale must be greater than 1")
    penalty_max = getattr(args, "alm_penalty_max", None)
    if penalty_max is not None and penalty_max <= 0.0:
        raise ValueError("--alm-penalty-max must be positive")
    if penalty_max is not None and penalty_max < args.alm_penalty_init:
        raise ValueError(
            f"--alm-penalty-max ({penalty_max}) must be >= --alm-penalty-init ({args.alm_penalty_init})"
        )
    if args.alm_feas_tol <= 0.0:
        raise ValueError("--alm-feas-tol must be positive")
    if args.alm_stationarity_tol <= 0.0:
        raise ValueError("--alm-stationarity-tol must be positive")
    trust_radius_init = getattr(args, "alm_trust_radius_init", None)
    trust_radius_min = getattr(args, "alm_trust_radius_min", None)
    trust_radius_shrink = getattr(args, "alm_trust_radius_shrink", None)
    trust_radius_grow = getattr(args, "alm_trust_radius_grow", None)
    max_inner_attempts = getattr(args, "alm_max_inner_attempts", None)
    curvature_smoothing = getattr(args, "alm_curvature_smoothing", None)
    distance_smoothing = getattr(args, "alm_distance_smoothing", None)
    if trust_radius_init is not None and trust_radius_init < 0.0:
        raise ValueError("--alm-trust-radius-init must be nonnegative")
    if trust_radius_min is not None and trust_radius_min <= 0.0:
        raise ValueError("--alm-trust-radius-min must be positive")
    if trust_radius_shrink is not None and not (0.0 < trust_radius_shrink < 1.0):
        raise ValueError("--alm-trust-radius-shrink must be between 0 and 1")
    if trust_radius_grow is not None and trust_radius_grow <= 1.0:
        raise ValueError("--alm-trust-radius-grow must be greater than 1")
    if max_inner_attempts is not None and max_inner_attempts <= 0:
        raise ValueError("--alm-max-inner-attempts must be positive")
    if curvature_smoothing is not None and curvature_smoothing <= 0.0:
        raise ValueError("--alm-curvature-smoothing must be positive")
    if distance_smoothing is not None and distance_smoothing <= 0.0:
        raise ValueError("--alm-distance-smoothing must be positive")


def positive_part(value: float) -> float:
    return float(max(value, 0.0))


def upper_bound_residual(metric_value: float, upper_bound: float) -> float:
    return positive_part(metric_value - upper_bound)


def lower_bound_residual(metric_value: float, lower_bound: float) -> float:
    return positive_part(lower_bound - metric_value)


def normalized_quadratic_penalty_residual(
    penalty_value: float,
    penalty_grad,
    normalization: float = 1.0,
    reference_grad=None,
):
    normalization = max(float(normalization), np.finfo(float).eps)
    normalized_value = positive_part(penalty_value) / normalization
    if normalized_value <= 0.0:
        reference = penalty_grad if reference_grad is None else reference_grad
        return 0.0, zero_gradient_like(reference)

    residual = float(np.sqrt(normalized_value))
    scale = 0.5 / (residual * normalization)
    return residual, scale * np.asarray(penalty_grad, dtype=float)


def normalized_lp_penalty_residual(
    penalty_value: float,
    penalty_grad,
    p: int,
    normalization: float = 1.0,
    reference_grad=None,
):
    if p <= 0:
        raise ValueError("p must be positive")
    normalization = max(float(normalization), np.finfo(float).eps)
    scaled_value = (float(p) * positive_part(penalty_value)) / normalization
    if scaled_value <= 0.0:
        reference = penalty_grad if reference_grad is None else reference_grad
        return 0.0, zero_gradient_like(reference)

    residual = float(scaled_value ** (1.0 / float(p)))
    scale = (scaled_value ** (1.0 / float(p) - 1.0)) / normalization
    return residual, scale * np.asarray(penalty_grad, dtype=float)


def augmented_objective(
    base_value: float,
    base_grad,
    constraint_values,
    constraint_grads,
    multipliers,
    penalty: float,
):
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    constraint_values = np.asarray(constraint_values, dtype=float)
    constraint_grad_list = [
        np.asarray(constraint_grad, dtype=float) for constraint_grad in constraint_grads
    ]
    multipliers = np.asarray(multipliers, dtype=float)
    total_value = float(base_value) + float(np.dot(multipliers, constraint_values))
    total_value += 0.5 * float(penalty) * float(np.dot(constraint_values, constraint_values))

    total_grad = np.array(base_grad, copy=True)
    for multiplier, constraint_value, constraint_grad in zip(
        multipliers, constraint_values, constraint_grad_list
    ):
        weight = float(multiplier) + float(penalty) * float(constraint_value)
        total_grad = total_grad + weight * constraint_grad

    return _build_augmented_evaluation(
        base_value=float(base_value),
        base_grad=np.asarray(base_grad, dtype=float),
        total_value=total_value,
        total_grad=total_grad,
        constraint_values=constraint_values,
        constraint_grads=constraint_grad_list,
        dual_update_values=constraint_values,
        feasibility_values=constraint_values,
    )


def augmented_inequality_objective(
    base_value: float,
    base_grad,
    constraint_values,
    constraint_grads,
    multipliers,
    penalty: float,
):
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    constraint_values = np.asarray(constraint_values, dtype=float)
    constraint_grad_list = [
        np.asarray(constraint_grad, dtype=float) for constraint_grad in constraint_grads
    ]
    multipliers = np.asarray(multipliers, dtype=float)
    positive_shift = np.maximum(0.0, multipliers + float(penalty) * constraint_values)

    total_value = float(base_value)
    if constraint_values.size > 0:
        total_value += 0.5 / float(penalty) * float(
            np.dot(positive_shift, positive_shift) - np.dot(multipliers, multipliers)
        )

    total_grad = np.array(base_grad, copy=True)
    for active_multiplier, constraint_grad in zip(positive_shift, constraint_grad_list):
        total_grad = total_grad + float(active_multiplier) * constraint_grad

    feasibility_values = np.maximum(constraint_values, 0.0)
    return _build_augmented_evaluation(
        base_value=float(base_value),
        base_grad=np.asarray(base_grad, dtype=float),
        total_value=total_value,
        total_grad=total_grad,
        constraint_values=constraint_values,
        constraint_grads=constraint_grad_list,
        dual_update_values=constraint_values,
        feasibility_values=feasibility_values,
    )


def zero_gradient_like(reference_grad):
    return np.zeros_like(np.asarray(reference_grad))


def _build_augmented_evaluation(
    *,
    base_value: float,
    base_grad,
    total_value: float,
    total_grad,
    constraint_values: np.ndarray,
    constraint_grads: Sequence[np.ndarray],
    dual_update_values,
    feasibility_values,
):
    dual_update_array = np.asarray(dual_update_values, dtype=float)
    feasibility_array = np.asarray(feasibility_values, dtype=float)
    stationarity_norm = float(np.linalg.norm(np.asarray(total_grad, dtype=float)))
    max_feasibility_violation = _max_value(feasibility_array)
    return {
        "total": float(total_value),
        "base_value": float(base_value),
        "base_grad": np.asarray(base_grad, dtype=float),
        "grad": np.asarray(total_grad, dtype=float),
        "constraint_values": np.asarray(constraint_values, dtype=float),
        "constraint_grads": [np.asarray(constraint_grad, dtype=float) for constraint_grad in constraint_grads],
        "dual_update_values": dual_update_array,
        "feasibility_values": feasibility_array,
        "max_violation": max_feasibility_violation,
        "max_feasibility_violation": max_feasibility_violation,
        "stationarity_norm": stationarity_norm,
    }


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _max_value(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else 0.0


def alm_result_diagnostics_fields(alm_result) -> dict:
    return {
        "ALM_MULTIPLIER_CAP_BINDING": getattr(alm_result, "multiplier_cap_binding", None),
        "ALM_MULTIPLIER_CAP_BINDING_INDICES": getattr(
            alm_result,
            "multiplier_cap_binding_indices",
            None,
        ),
        "ALM_FINAL_BASE_OBJECTIVE": getattr(alm_result, "final_base_objective", None),
        "ALM_FINAL_PENALTY_OBJECTIVE": getattr(alm_result, "final_penalty_objective", None),
        "ALM_FINAL_PENALTY_OBJECTIVE_RATIO": getattr(
            alm_result,
            "final_penalty_objective_ratio",
            None,
        ),
        "ALM_FINAL_TOTAL_GRAD_NORM": getattr(alm_result, "final_total_grad_norm", None),
        "ALM_FINAL_BASE_GRAD_NORM": getattr(alm_result, "final_base_grad_norm", None),
        "ALM_FINAL_PENALTY_GRAD_NORM": getattr(
            alm_result,
            "final_penalty_grad_norm",
            None,
        ),
        "ALM_FINAL_PENALTY_GRAD_RATIO": getattr(
            alm_result,
            "final_penalty_grad_ratio",
            None,
        ),
        "ALM_PENALTY_CAP_REACHED": getattr(alm_result, "penalty_cap_reached", None),
        "ALM_PENALTY_CAP_REQUESTED": getattr(
            alm_result,
            "penalty_cap_requested",
            None,
        ),
    }


def _nonfinite_evaluation_fields(evaluation: dict) -> tuple[str, ...]:
    invalid_fields: list[str] = []

    if not np.isfinite(float(evaluation["total"])):
        invalid_fields.append("total")

    grad = np.asarray(evaluation["grad"], dtype=float)
    if not np.all(np.isfinite(grad)):
        invalid_fields.append("grad")

    optional_scalar_fields = (
        "stationarity_norm",
        "metric_stationarity_norm",
        "max_violation",
        "max_feasibility_violation",
        "base_value",
        "base_total",
    )
    for field_name in optional_scalar_fields:
        if field_name not in evaluation:
            continue
        if not np.isfinite(float(evaluation[field_name])):
            invalid_fields.append(field_name)

    optional_array_fields = (
        "constraint_values",
        "feasibility_values",
        "dual_update_values",
        "metric_grad",
        "base_grad",
        "constraint_activity_tolerances",
    )
    for field_name in optional_array_fields:
        if field_name not in evaluation:
            continue
        field_values = np.asarray(evaluation[field_name], dtype=float)
        if not np.all(np.isfinite(field_values)):
            invalid_fields.append(field_name)

    if "constraint_grads" in evaluation and evaluation["constraint_grads"] is not None:
        for grad_index, constraint_grad in enumerate(evaluation["constraint_grads"]):
            constraint_grad_array = np.asarray(constraint_grad, dtype=float)
            if not np.all(np.isfinite(constraint_grad_array)):
                invalid_fields.append(f"constraint_grads[{grad_index}]")

    return tuple(invalid_fields)


def _require_finite_evaluation(evaluation: dict, *, context: str) -> None:
    invalid_fields = _nonfinite_evaluation_fields(evaluation)
    if invalid_fields:
        invalid_summary = ", ".join(invalid_fields)
        raise ValueError(f"{context} produced non-finite ALM data: {invalid_summary}")


def _elevated_rejection_total(reference_total: float) -> float:
    return float(reference_total) + max(abs(float(reference_total)), 1.0) + _ACCEPTANCE_TOTAL_ATOL


def _sanitize_nonfinite_inner_evaluation(
    evaluation: dict,
    *,
    fallback_evaluation: dict,
) -> dict:
    invalid_fields = _nonfinite_evaluation_fields(evaluation)
    if not invalid_fields:
        return evaluation

    sanitized = deepcopy(fallback_evaluation)
    sanitized["total"] = _elevated_rejection_total(float(fallback_evaluation["total"]))
    sanitized["grad"] = np.asarray(fallback_evaluation["grad"], dtype=float).copy()
    if "metric_grad" in fallback_evaluation:
        sanitized["metric_grad"] = np.asarray(
            fallback_evaluation["metric_grad"],
            dtype=float,
        ).copy()
    if "base_grad" in fallback_evaluation:
        sanitized["base_grad"] = np.asarray(
            fallback_evaluation["base_grad"],
            dtype=float,
        ).copy()
    sanitized["nonfinite_evaluation"] = True
    sanitized["nonfinite_fields"] = list(invalid_fields)
    return sanitized


def _move_tolerance(reference_x) -> float:
    reference_norm = float(np.linalg.norm(np.asarray(reference_x, dtype=float)))
    return _INFEASIBLE_STALL_MOVE_TOL * max(1.0, reference_norm)


def _conditioning_metrics(evaluation: dict) -> dict[str, float | None]:
    total_value = float(evaluation["total"])
    base_objective = float(
        evaluation.get(
            "base_total",
            evaluation.get("base_value", total_value),
        )
    )
    if not np.isfinite(base_objective):
        base_objective = total_value
    penalty_objective = total_value - base_objective
    penalty_objective_ratio = abs(penalty_objective) / max(abs(base_objective), 1.0)

    total_grad = np.asarray(evaluation["grad"], dtype=float).reshape(-1)
    total_grad_norm = float(np.linalg.norm(total_grad))

    base_grad_raw = evaluation.get("base_grad")
    if base_grad_raw is None:
        base_grad_norm = None
        penalty_grad_norm = None
        penalty_grad_ratio = None
    else:
        base_grad = np.asarray(base_grad_raw, dtype=float).reshape(-1)
        base_grad_norm = float(np.linalg.norm(base_grad))
        penalty_grad_norm = float(np.linalg.norm(total_grad - base_grad))
        penalty_grad_ratio = penalty_grad_norm / max(base_grad_norm, 1.0)

    return {
        "conditioning_base_objective": float(base_objective),
        "conditioning_penalty_objective": float(penalty_objective),
        "conditioning_penalty_objective_ratio": float(penalty_objective_ratio),
        "conditioning_total_grad_norm": float(total_grad_norm),
        "conditioning_base_grad_norm": (
            None if base_grad_norm is None else float(base_grad_norm)
        ),
        "conditioning_penalty_grad_norm": (
            None if penalty_grad_norm is None else float(penalty_grad_norm)
        ),
        "conditioning_penalty_grad_ratio": (
            None if penalty_grad_ratio is None else float(penalty_grad_ratio)
        ),
        "penalty_gradient_norm": (
            None if penalty_grad_norm is None else float(penalty_grad_norm)
        ),
    }


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
        np.linalg.norm(np.asarray(end_x, dtype=float) - np.asarray(start_x, dtype=float))
    )
    move_scale = max(1.0, float(np.linalg.norm(np.asarray(start_x, dtype=float))))
    moved = move_norm > 1e-8 * move_scale

    objective_drop = float(current_total) - float(final_total)
    objective_tol = max(1e-8, 0.05 * abs(float(current_total)))
    improved_objective = objective_drop > objective_tol

    stationarity_drop = float(current_stationarity_norm) - float(final_stationarity_norm)
    stationarity_tol = max(1e-6, 0.05 * float(current_stationarity_norm))
    improved_stationarity = stationarity_drop > stationarity_tol

    feasibility_drop, feasibility_tol = _feasibility_improvement_metrics(
        current_max_feasibility_violation,
        final_max_feasibility_violation,
    )
    improved_feasibility = feasibility_drop > feasibility_tol

    return moved or improved_objective or improved_stationarity or improved_feasibility


def _acceptable_total_upper_bound(current_total: float) -> float:
    total_scale = max(np.finfo(float).eps, abs(float(current_total)))
    return float(current_total) + _ACCEPTANCE_TOTAL_ATOL + (
        _ACCEPTANCE_TOTAL_RTOL * total_scale
    )


def _candidate_is_acceptable(
    current_eval: dict,
    candidate_eval: dict,
    result,
    moved_norm: float,
    update_feasibility_tol: float,
) -> bool:
    if not (
        bool(getattr(result, "success", False))
        or int(getattr(result, "nit", 0)) > 0
        or float(moved_norm) > _ACCEPTANCE_MOVE_TOL
    ):
        return False

    (
        _current_solver_values,
        _current_feasibility_values,
        _current_dual_update_values,
        current_max_feasibility_violation,
    ) = _extract_constraint_state(current_eval)
    (
        _candidate_solver_values,
        _candidate_feasibility_values,
        _candidate_dual_update_values,
        candidate_max_feasibility_violation,
    ) = _extract_constraint_state(candidate_eval)
    allowed_max_feasibility = max(
        float(update_feasibility_tol),
        float(current_max_feasibility_violation),
    ) + _ACCEPTANCE_TOTAL_ATOL
    if float(candidate_max_feasibility_violation) > allowed_max_feasibility:
        return False

    candidate_total = float(candidate_eval["total"])
    return candidate_total <= _acceptable_total_upper_bound(float(current_eval["total"]))


def _feasibility_improvement_metrics(
    current_max_feasibility_violation: float,
    candidate_max_feasibility_violation: float,
) -> tuple[float, float]:
    feasibility_improvement = (
        float(current_max_feasibility_violation) - float(candidate_max_feasibility_violation)
    )
    feasibility_improvement_floor = max(
        _INFEASIBLE_STALL_FEASIBILITY_ATOL,
        _INFEASIBLE_STALL_FEASIBILITY_RTOL
        * max(
            abs(float(current_max_feasibility_violation)),
            abs(float(candidate_max_feasibility_violation)),
            1.0,
        ),
    )
    return feasibility_improvement, feasibility_improvement_floor


def _classify_infeasible_inner_stall(
    current_eval: dict,
    candidate_eval: dict,
    result,
    moved_norm: float,
    move_tolerance: float,
    feasibility_gate: float,
) -> tuple[bool, bool, str | None]:
    if float(moved_norm) > float(move_tolerance):
        return False, False, None

    current_max_feasibility_violation = _extract_constraint_state(current_eval)[3]
    candidate_max_feasibility_violation = _extract_constraint_state(candidate_eval)[3]
    if candidate_max_feasibility_violation <= float(feasibility_gate):
        return False, False, None

    feasibility_improvement, feasibility_improvement_floor = _feasibility_improvement_metrics(
        current_max_feasibility_violation,
        candidate_max_feasibility_violation,
    )
    if feasibility_improvement > feasibility_improvement_floor:
        return False, False, None

    current_total = float(current_eval["total"])
    candidate_total = float(candidate_eval["total"])
    objective_improvement = current_total - candidate_total
    improvement_floor = max(
        _INFEASIBLE_STALL_OBJECTIVE_ATOL,
        _INFEASIBLE_STALL_OBJECTIVE_RTOL
        * max(abs(current_total), abs(candidate_total), 1.0),
    )
    if objective_improvement > improvement_floor:
        return False, False, None

    result_success = bool(getattr(result, "success", False))
    result_message = str(getattr(result, "message", "")).upper()
    if result_success and "RELATIVE REDUCTION OF F" in result_message:
        return True, True, "relative_objective_termination_without_feasibility_gain"
    if result_success:
        return True, True, "successful_inner_solve_without_feasibility_gain"
    return True, False, "failed_inner_solve_without_feasibility_gain"


def _normalize_trust_radius(trust_radius: float | None) -> float | None:
    if trust_radius is None:
        return None
    normalized = float(trust_radius)
    if normalized <= 0.0:
        return None
    return normalized


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


def _build_box_bounds(center: np.ndarray, trust_radius: float | None):
    widths = _trust_region_widths(center, trust_radius)
    if widths is None:
        return None
    return [
        (float(value - width), float(value + width))
        for value, width in zip(np.asarray(center, dtype=float), widths)
    ]


def _trust_region_widths(center: np.ndarray, trust_radius: float | None):
    normalized_trust_radius = _normalize_trust_radius(trust_radius)
    if normalized_trust_radius is None:
        return None
    # This is a lightweight trust-region proxy implemented with L-BFGS-B bounds:
    # each continuation centers a symmetric box around the current iterate.
    return normalized_trust_radius * np.maximum(
        1.0, np.abs(np.asarray(center, dtype=float))
    )


def _target_inner_physical_x(opt_x, center: np.ndarray, widths: np.ndarray | None):
    opt_x_arr = np.asarray(opt_x, dtype=float)
    if widths is None:
        return opt_x_arr
    return np.asarray(center, dtype=float) + np.asarray(widths, dtype=float) * np.tanh(
        opt_x_arr
    )


def _build_target_inner_value_and_grad(
    *,
    evaluate_value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    center: np.ndarray,
    widths: np.ndarray | None,
):
    import jax
    import jax.numpy as jnp

    center_arr = np.asarray(center, dtype=float).copy()
    widths_arr = None if widths is None else np.asarray(widths, dtype=float).copy()
    result_spec = (
        jax.ShapeDtypeStruct((), np.float64),
        jax.ShapeDtypeStruct(center_arr.shape, np.float64),
    )

    if widths_arr is None:

        def _physical_x_jax(opt_x):
            return jnp.asarray(opt_x, dtype=jnp.float64)

        optimizer_x0 = np.asarray(center_arr, dtype=float)
    else:
        center_jax = jax.device_put(np.asarray(center_arr, dtype=np.float64))
        widths_jax = jax.device_put(np.asarray(widths_arr, dtype=np.float64))

        def _physical_x_jax(opt_x):
            opt_x = jnp.asarray(opt_x, dtype=jnp.float64)
            return center_jax + widths_jax * jnp.tanh(opt_x)

        optimizer_x0 = np.zeros_like(center_arr, dtype=float)

    def _host_eval(host_x):
        value, grad = evaluate_value_and_grad(np.asarray(host_x, dtype=float))
        return np.asarray(value, dtype=np.float64), np.asarray(grad, dtype=np.float64)

    def _value_and_grad(opt_x):
        opt_x = jnp.asarray(opt_x, dtype=jnp.float64)
        physical_x = _physical_x_jax(opt_x)
        value, grad_x = jax.pure_callback(_host_eval, result_spec, physical_x)
        grad_x = jnp.asarray(grad_x, dtype=jnp.float64)
        if widths_arr is None:
            return jnp.asarray(value, dtype=jnp.float64), grad_x
        grad_scale = widths_jax * (1.0 - jnp.square(jnp.tanh(opt_x)))
        return jnp.asarray(value, dtype=jnp.float64), grad_x * grad_scale

    return _value_and_grad, optimizer_x0


def _resolve_target_inner_optimizer(inner_optimizer_contract):
    if inner_optimizer_contract is None:
        return None, None

    from simsopt.geo.optimizer_jax import TargetOptimizerContract, target_minimize

    if not isinstance(inner_optimizer_contract, TargetOptimizerContract):
        raise ValueError(
            "minimize_alm() only supports TargetOptimizerContract for "
            "inner_optimizer_contract."
        )
    if inner_optimizer_contract.use_least_squares_objective:
        raise ValueError(
            "minimize_alm() only supports scalar target ALM inner solves."
        )
    if inner_optimizer_contract.method != "lbfgs-ondevice":
        raise ValueError(
            "minimize_alm() only supports method='lbfgs-ondevice' for "
            "target ALM inner solves."
        )
    return inner_optimizer_contract.method, target_minimize


def _select_inner_solve_profile(
    *,
    trust_radius: float | None,
    continuation_iteration: int,
    feasible_enough: bool,
) -> ALMInnerSolveProfile:
    if _normalize_trust_radius(trust_radius) is None:
        return _UNBOUNDED_INNER_PROFILE
    return _BOXED_INNER_PROFILES[
        (bool(feasible_enough), bool(continuation_iteration > 0))
    ]


def _incumbent_objective_value(evaluation: dict) -> float:
    if "physics_total" in evaluation:
        return float(evaluation["physics_total"])
    if "base_value" in evaluation:
        return float(evaluation["base_value"])
    if "base_total" in evaluation:
        return float(evaluation["base_total"])
    return float(evaluation["total"])


def _project_nonnegative_multipliers(
    multipliers: np.ndarray,
    dual_update_values: np.ndarray,
    penalty: float,
    multiplier_max: float | None,
) -> np.ndarray:
    projected, _cap_binding, _cap_binding_indices = _project_nonnegative_multipliers_with_diagnostics(
        multipliers,
        dual_update_values,
        penalty,
        multiplier_max,
    )
    return projected


def _updated_nonnegative_multipliers(
    multipliers: np.ndarray,
    dual_update_values: np.ndarray,
    penalty: float,
) -> np.ndarray:
    return np.maximum(
        0.0,
        np.asarray(multipliers, dtype=float) + float(penalty) * np.asarray(dual_update_values, dtype=float),
    )


def _project_nonnegative_multipliers_with_diagnostics(
    multipliers: np.ndarray,
    dual_update_values: np.ndarray,
    penalty: float,
    multiplier_max: float | None,
) -> tuple[np.ndarray, bool, list[int]]:
    updated = _updated_nonnegative_multipliers(
        multipliers,
        dual_update_values,
        penalty,
    )
    if multiplier_max is None:
        return updated, False, []
    cap = float(multiplier_max)
    cap_binding_mask = updated > cap
    return (
        np.minimum(updated, cap),
        bool(np.any(cap_binding_mask)),
        np.flatnonzero(cap_binding_mask).astype(int).tolist(),
    )


def _next_penalty(
    penalty: float,
    *,
    penalty_scale: float,
    penalty_max: float | None,
) -> tuple[float, bool, float]:
    requested_penalty = penalty * penalty_scale
    if penalty_max is None:
        if not np.isfinite(requested_penalty):
            return penalty, True, requested_penalty
        return requested_penalty, False, requested_penalty

    if penalty_max <= 0.0:
        raise ValueError("ALM penalty_max must be positive when provided")
    if not np.isfinite(requested_penalty) or requested_penalty > penalty_max:
        return penalty_max, True, requested_penalty
    return requested_penalty, False, requested_penalty


def _build_inner_options(
    inner_options: dict,
    update_stationarity_tol: float,
    *,
    profile: ALMInnerSolveProfile,
) -> dict:
    options = dict(inner_options)
    base_gtol = float(options.get("gtol", 1e-12))
    staged_gtol = max(
        np.finfo(float).eps,
        min(1e-4, 0.1 * float(update_stationarity_tol)),
    )
    options["gtol"] = max(base_gtol, staged_gtol)
    options["maxls"] = max(1, int(options.get("maxls", profile.default_maxls)))

    if profile.maxiter_cap is None or profile.maxls_cap is None or profile.ftol_floor is None:
        return options

    requested_maxiter = max(1, int(options.get("maxiter", 150)))
    requested_maxls = max(1, int(options.get("maxls", profile.default_maxls)))
    requested_maxfun = options.get("maxfun")
    base_ftol = float(options.get("ftol", 1e-15))

    options["maxiter"] = min(requested_maxiter, profile.maxiter_cap)
    options["maxls"] = min(requested_maxls, profile.maxls_cap)
    options["ftol"] = max(base_ftol, profile.ftol_floor)

    maxfun_cap = max(
        20,
        2 * int(options["maxiter"]) * max(2, int(options["maxls"])),
    )
    if requested_maxfun is None:
        options["maxfun"] = maxfun_cap
    else:
        options["maxfun"] = min(max(1, int(requested_maxfun)), maxfun_cap)
    return options


def _build_target_inner_callback(
    inner_callback: Callable[[np.ndarray], None] | None,
    *,
    center: np.ndarray,
    widths: np.ndarray | None,
):
    if inner_callback is None:
        return None

    callback_center = np.asarray(center, dtype=float).copy()
    callback_widths = None if widths is None else np.asarray(widths, dtype=float).copy()

    def _callback(opt_x, *, _center=callback_center, _widths=callback_widths):
        inner_callback(_target_inner_physical_x(opt_x, _center, _widths))

    return _callback


def _build_target_inner_options(inner_attempt_options: dict) -> dict:
    return {
        key: inner_attempt_options[key]
        for key in _TARGET_INNER_OPTION_KEYS
        if key in inner_attempt_options
    }


def _run_target_inner_solve(
    *,
    evaluate_value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    center: np.ndarray,
    attempt_radius: float | None,
    inner_attempt_options: dict,
    method: str,
    optimizer: Callable[..., object],
    inner_callback: Callable[[np.ndarray], None] | None,
):
    trust_widths = _trust_region_widths(center, attempt_radius)
    target_fun, target_x0 = _build_target_inner_value_and_grad(
        evaluate_value_and_grad=evaluate_value_and_grad,
        center=center,
        widths=trust_widths,
    )
    result = optimizer(
        target_fun,
        target_x0,
        method=method,
        tol=float(inner_attempt_options["gtol"]),
        maxiter=int(inner_attempt_options["maxiter"]),
        options=_build_target_inner_options(inner_attempt_options),
        value_and_grad=True,
        callback=_build_target_inner_callback(
            inner_callback,
            center=center,
            widths=trust_widths,
        ),
    )
    candidate_x = _target_inner_physical_x(result.x, center, trust_widths)
    result.x = np.asarray(candidate_x, dtype=float).copy()
    return result, candidate_x


def _termination_reason_from_history(
    history: list[dict],
    *,
    success: bool,
    restored_best_feasible: bool,
) -> str:
    if success:
        return "converged"
    if not history:
        return "terminated"

    latest_entry = history[-1]
    latest_action = latest_entry.get("action")
    outer_termination = latest_entry.get("outer_termination")
    if outer_termination == "max_outer":
        if restored_best_feasible:
            return "max_outer_restored_best_feasible"
        if latest_action == "dual_update":
            return "max_outer_after_dual_update"
        if latest_action == "subproblem_limit":
            return "max_outer_after_subproblem_limit"
        if latest_action == "infeasible_stall_penalty_increase":
            return "max_outer_after_infeasible_stall"
        if latest_action == "penalty_increase":
            return "max_outer_after_penalty_increase"
        return "max_outer"
    if restored_best_feasible:
        return "restored_best_feasible"
    if isinstance(latest_action, str) and latest_action:
        return latest_action
    return "terminated"


def run_directional_taylor_test(
    evaluate_problem: Callable[[np.ndarray, np.ndarray, float], dict],
    x0,
    multipliers,
    penalty: float,
    *,
    direction=None,
    epsilons: Sequence[float] | None = None,
    seed: int = 1,
    ratio_threshold: float = 0.6,
) -> dict:
    x = np.asarray(x0, dtype=float).copy()
    multiplier_array = np.asarray(multipliers, dtype=float).copy()
    if direction is None:
        rng = np.random.RandomState(seed)
        direction_array = rng.standard_normal(size=x.shape)
    else:
        direction_array = np.asarray(direction, dtype=float).copy()
    if direction_array.shape != x.shape:
        raise ValueError("direction must have the same shape as x0")
    direction_norm = float(np.linalg.norm(direction_array))
    if direction_norm <= np.finfo(float).eps:
        raise ValueError("direction must be nonzero")
    unit_direction = direction_array / direction_norm

    taylor_epsilons = (
        _DEFAULT_TAYLOR_EPSILONS
        if epsilons is None
        else tuple(float(epsilon) for epsilon in epsilons)
    )
    if len(taylor_epsilons) == 0:
        raise ValueError("epsilons must be non-empty")

    base_eval = evaluate_problem(x, multiplier_array, float(penalty))
    base_total = float(base_eval["total"])
    base_grad = np.asarray(base_eval["grad"], dtype=float)
    directional_derivative = float(np.dot(base_grad.reshape(-1), unit_direction.reshape(-1)))
    error_floor = 1e-10 * max(1.0, abs(directional_derivative))

    errors = []
    central_estimates = []
    ratios = []
    passed = True
    previous_error = None
    for epsilon in taylor_epsilons:
        plus_eval = evaluate_problem(x + float(epsilon) * unit_direction, multiplier_array, float(penalty))
        minus_eval = evaluate_problem(x - float(epsilon) * unit_direction, multiplier_array, float(penalty))
        central_estimate = (
            float(plus_eval["total"]) - float(minus_eval["total"])
        ) / (2.0 * float(epsilon))
        error = abs(central_estimate - directional_derivative)
        central_estimates.append(float(central_estimate))
        errors.append(float(error))

        ratio = None
        if previous_error is not None and previous_error > error_floor:
            ratio = float(error / previous_error)
            if error > error_floor and ratio > float(ratio_threshold):
                passed = False
        ratios.append(ratio)
        previous_error = error

    finite_ratios = [ratio for ratio in ratios if ratio is not None]
    return {
        "passed": bool(passed),
        "seed": int(seed),
        "penalty": float(penalty),
        "direction": unit_direction.tolist(),
        "directional_derivative": directional_derivative,
        "base_total": base_total,
        "epsilons": [float(epsilon) for epsilon in taylor_epsilons],
        "central_estimates": central_estimates,
        "errors": errors,
        "ratios": finite_ratios,
        "max_ratio": max(finite_ratios) if finite_ratios else None,
        "ratio_threshold": float(ratio_threshold),
    }


def _extract_constraint_state(evaluation: dict):
    solver_constraint_values = _as_float_array(evaluation["constraint_values"])
    feasibility_values = _as_float_array(
        evaluation.get("feasibility_values", solver_constraint_values)
    )
    dual_update_values = _as_float_array(
        evaluation.get("dual_update_values", solver_constraint_values)
    )
    max_feasibility_violation = float(
        evaluation.get(
            "max_feasibility_violation",
            _max_value(feasibility_values),
        )
    )
    return (
        solver_constraint_values,
        feasibility_values,
        dual_update_values,
        max_feasibility_violation,
    )


def _extract_stage2_constraint_signal_state(
    evaluation: dict,
) -> ALMConstraintSignalState:
    explicit_stage2_signals = any(
        key in evaluation
        for key in (
            "hard_signed_constraint_values",
            "hard_violation_values",
            "surrogate_signed_constraint_values",
            "hard_dual_update_values",
        )
    )
    (
        solver_constraint_values,
        feasibility_values,
        dual_update_values,
        _max_feasibility_violation,
    ) = _extract_constraint_state(evaluation)
    hard_signed_constraint_values = _as_float_array(
        evaluation.get("hard_signed_constraint_values", dual_update_values)
    )
    hard_violation_values = _as_float_array(
        evaluation.get("hard_violation_values", feasibility_values)
    )
    surrogate_signed_constraint_values = _as_float_array(
        evaluation.get("surrogate_signed_constraint_values", solver_constraint_values)
    )
    preferred_dual_update_values = _as_float_array(
        evaluation.get("hard_dual_update_values", hard_signed_constraint_values)
    )
    return ALMConstraintSignalState(
        explicit_stage2_signals=bool(explicit_stage2_signals),
        hard_signed_constraint_values=hard_signed_constraint_values,
        hard_violation_values=hard_violation_values,
        surrogate_signed_constraint_values=surrogate_signed_constraint_values,
        preferred_dual_update_values=preferred_dual_update_values,
    )


def _constraint_activity_tolerances(evaluation: dict, constraint_values: np.ndarray) -> np.ndarray:
    raw_tolerances = evaluation.get("constraint_activity_tolerances")
    if raw_tolerances is None:
        return np.zeros_like(constraint_values)

    tolerances = np.asarray(raw_tolerances, dtype=float)
    if tolerances.shape == constraint_values.shape:
        return tolerances
    if tolerances.size == 1:
        return np.full_like(constraint_values, float(tolerances.reshape(())))
    raise ValueError("constraint_activity_tolerances shape must match constraint_values")


def _constraint_activity_mask(
    constraint_values: np.ndarray,
    feasibility_values: np.ndarray,
    activity_tolerances: np.ndarray,
    feasibility_gate: float,
) -> np.ndarray:
    return np.logical_and(
        np.asarray(feasibility_values, dtype=float) <= float(feasibility_gate),
        np.asarray(constraint_values, dtype=float)
        >= -np.asarray(activity_tolerances, dtype=float),
    )


def _positive_shift(
    multipliers: np.ndarray,
    penalty: float,
    constraint_values: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        0.0,
        np.asarray(multipliers, dtype=float)
        + float(penalty) * np.asarray(constraint_values, dtype=float),
    )


def _constraint_routing_state(
    evaluation: dict,
    multipliers: np.ndarray,
    penalty: float,
    feasibility_gate: float,
) -> ALMConstraintRoutingState:
    signal_state = _extract_stage2_constraint_signal_state(evaluation)
    activity_tolerances = _constraint_activity_tolerances(
        evaluation,
        signal_state.surrogate_signed_constraint_values,
    )
    hard_activity_mask = _constraint_activity_mask(
        signal_state.hard_signed_constraint_values,
        signal_state.hard_violation_values,
        activity_tolerances,
        feasibility_gate,
    )
    surrogate_activity_mask = _constraint_activity_mask(
        signal_state.surrogate_signed_constraint_values,
        signal_state.hard_violation_values,
        activity_tolerances,
        feasibility_gate,
    )
    masks_disagree = not np.array_equal(hard_activity_mask, surrogate_activity_mask)
    signal_mismatch_active = signal_state.explicit_stage2_signals and masks_disagree
    hard_positive_shift = _positive_shift(
        multipliers,
        penalty,
        signal_state.preferred_dual_update_values,
    )
    surrogate_positive_shift = _positive_shift(
        multipliers,
        penalty,
        signal_state.surrogate_signed_constraint_values,
    )
    hard_feasible_under_gate = (
        _max_value(signal_state.hard_violation_values) <= float(feasibility_gate)
    )
    direct_boundary_mismatch = (
        signal_state.explicit_stage2_signals
        and hard_feasible_under_gate
        and np.any(surrogate_positive_shift > 0.0)
    )
    if direct_boundary_mismatch:
        signal_mismatch_active = True
    return ALMConstraintRoutingState(
        signal_state=signal_state,
        hard_activity_mask=hard_activity_mask,
        surrogate_activity_mask=surrogate_activity_mask,
        signal_mismatch_active=bool(signal_mismatch_active),
        hard_positive_shift=hard_positive_shift,
        surrogate_positive_shift=surrogate_positive_shift,
        hard_positive_shift_zero=bool(not np.any(hard_positive_shift > 0.0)),
        surrogate_positive_shift_zero=bool(not np.any(surrogate_positive_shift > 0.0)),
        hard_max_violation=_max_value(signal_state.hard_violation_values),
        surrogate_max_value=_max_value(signal_state.surrogate_signed_constraint_values),
    )


def _kkt_stationarity_norm(
    total_grad,
    constraint_grads,
    constraint_values: np.ndarray,
    feasibility_values: np.ndarray,
    activity_tolerances: np.ndarray,
    feasibility_gate: float,
) -> float | None:
    if constraint_grads is None:
        return None

    total_grad_array = np.asarray(total_grad, dtype=float).reshape(-1)
    if total_grad_array.size == 0:
        return 0.0

    active_constraint_grads = []
    for constraint_grad, constraint_value, feasibility_value, activity_tolerance in zip(
        constraint_grads,
        constraint_values,
        feasibility_values,
        activity_tolerances,
    ):
        if float(feasibility_value) > float(feasibility_gate):
            continue
        if float(constraint_value) < -float(activity_tolerance):
            continue
        active_constraint_grads.append(np.asarray(constraint_grad, dtype=float).reshape(-1))

    if not active_constraint_grads:
        return None

    active_matrix = np.column_stack(active_constraint_grads)
    multipliers, _residual_norm = nnls(active_matrix, -total_grad_array)
    residual = total_grad_array + active_matrix @ multipliers
    return float(np.linalg.norm(residual))


def _stationarity_metrics(
    evaluation: dict,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> tuple[float, float | None, float, bool]:
    metric_grad = np.asarray(
        evaluation.get("metric_grad", evaluation["grad"]),
        dtype=float,
    )
    raw_stationarity_norm = float(
        evaluation.get(
            "metric_stationarity_norm",
            evaluation.get("stationarity_norm", np.linalg.norm(metric_grad)),
        )
    )
    if routing_state.signal_mismatch_active:
        kkt_stationarity_norm = None
        effective_stationarity_norm = raw_stationarity_norm
    else:
        preferred_dual_update_values = routing_state.signal_state.preferred_dual_update_values
        kkt_stationarity_norm = _kkt_stationarity_norm(
            metric_grad,
            evaluation.get("constraint_grads"),
            preferred_dual_update_values,
            routing_state.signal_state.hard_violation_values,
            _constraint_activity_tolerances(evaluation, preferred_dual_update_values),
            feasibility_gate,
        )
        effective_stationarity_norm = (
            raw_stationarity_norm
            if kkt_stationarity_norm is None
            else min(raw_stationarity_norm, kkt_stationarity_norm)
        )
    return (
        raw_stationarity_norm,
        kkt_stationarity_norm,
        effective_stationarity_norm,
        bool(routing_state.signal_mismatch_active),
    )


def minimize_alm(
    x0,
    constraint_names: Sequence[str],
    evaluate_problem: Callable[[np.ndarray, np.ndarray, float], dict],
    settings: ALMSettings,
    inner_options: dict,
    inner_optimizer_contract=None,
    inner_callback: Callable[[np.ndarray], None] | None = None,
    accepted_callback: Callable[[np.ndarray], None] | None = None,
    outer_state_callback: Callable[[int, np.ndarray, float], None] | None = None,
    snapshot_accepted_state_fn: Callable[[], object] | None = None,
    restore_incumbent_state_fn: Callable[[object], None] | None = None,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None = None,
    initial_multipliers: np.ndarray | None = None,
    initial_penalty: float | None = None,
):
    if (snapshot_accepted_state_fn is None) != (restore_incumbent_state_fn is None):
        raise ValueError(
            "snapshot_accepted_state_fn and restore_incumbent_state_fn must be provided together"
        )
    target_inner_method, target_inner_optimizer = _resolve_target_inner_optimizer(
        inner_optimizer_contract
    )
    if settings.penalty_max is not None and settings.penalty_max <= 0.0:
        raise ValueError("settings.penalty_max must be positive when provided")
    if settings.penalty_max is not None and settings.penalty_max < settings.penalty_init:
        raise ValueError(
            f"settings.penalty_max ({settings.penalty_max}) must be >= "
            f"settings.penalty_init ({settings.penalty_init})"
        )
    x = np.asarray(x0, dtype=float).copy()
    multipliers = (
        np.asarray(initial_multipliers, dtype=float).copy()
        if initial_multipliers is not None
        else np.zeros(len(constraint_names), dtype=float)
    )
    penalty = (
        float(initial_penalty)
        if initial_penalty is not None
        else float(settings.penalty_init)
    )
    total_inner_iterations = 0
    history = []
    final_eval = None
    last_result = None
    final_multipliers = multipliers.copy()
    final_penalty = penalty
    last_outer_iteration = 0
    cap_binding_detected = False
    cap_binding_indices: set[int] = set()
    penalty_cap_reached = False
    penalty_cap_requested = None
    trust_radius = _normalize_trust_radius(settings.trust_radius_init)
    update_feasibility_tol = max(settings.feasibility_tol, 1.0 / penalty)
    update_stationarity_tol = max(settings.stationarity_tol, 1.0 / penalty)
    best_feasible: ALMFeasibleIncumbent | None = None

    def _emit_history_snapshot(latest_entry: dict) -> None:
        if history_callback is None:
            return
        history_callback(
            [dict(entry) for entry in history],
            dict(latest_entry),
            multipliers.copy(),
            float(penalty),
        )

    def _build_result(
        *,
        success: bool,
        message: str,
        termination_reason: str,
        outer_iterations: int,
        evaluation: dict,
        multipliers_state: np.ndarray,
        penalty_state: float,
        inner_result,
        restored_best_feasible: bool,
        restored_best_feasible_reason: str | None,
        final_max_feasibility_violation: float,
        final_stationarity_norm: float,
        final_raw_stationarity_norm: float,
        final_kkt_stationarity_norm: float | None,
        final_feasibility_tolerance: float,
        final_stationarity_tolerance: float,
    ):
        (
            solver_constraint_values,
            feasibility_values,
            _dual_update_values,
            _max_feasibility_violation,
        ) = _extract_constraint_state(evaluation)
        routing_state = _constraint_routing_state(
            evaluation,
            multipliers_state,
            penalty_state,
            final_feasibility_tolerance,
        )
        inner_optimizer_success = None
        inner_optimizer_message = None
        if inner_result is not None:
            inner_optimizer_success = bool(getattr(inner_result, "success", False))
            inner_optimizer_message = str(getattr(inner_result, "message", ""))
        conditioning = _conditioning_metrics(evaluation)
        return SimpleNamespace(
            x=x.copy(),
            success=bool(success),
            message=message,
            termination_reason=str(termination_reason),
            nit=total_inner_iterations,
            outer_iterations=int(outer_iterations),
            constraint_names=list(constraint_names),
            constraint_values=[float(value) for value in feasibility_values],
            solver_constraint_values=[float(value) for value in solver_constraint_values],
            hard_signed_constraint_values=[
                float(value)
                for value in routing_state.signal_state.hard_signed_constraint_values
            ],
            hard_violation_values=[
                float(value)
                for value in routing_state.signal_state.hard_violation_values
            ],
            surrogate_signed_constraint_values=[
                float(value)
                for value in routing_state.signal_state.surrogate_signed_constraint_values
            ],
            multipliers=[float(value) for value in multipliers_state],
            penalty=float(penalty_state),
            trust_radius=trust_radius,
            history=history,
            inner_result=inner_result,
            optimizer_success=inner_optimizer_success,
            optimizer_message=inner_optimizer_message,
            converged_to_tolerances=bool(success),
            restored_best_feasible=bool(restored_best_feasible),
            restored_best_feasible_reason=restored_best_feasible_reason,
            final_max_feasibility_violation=float(final_max_feasibility_violation),
            final_stationarity_norm=float(final_stationarity_norm),
            final_raw_stationarity_norm=float(final_raw_stationarity_norm),
            final_kkt_stationarity_norm=(
                None
                if final_kkt_stationarity_norm is None
                else float(final_kkt_stationarity_norm)
            ),
            final_feasibility_tolerance=float(final_feasibility_tolerance),
            final_stationarity_tolerance=float(final_stationarity_tolerance),
            final_objective=float(evaluation["total"]),
            final_base_objective=conditioning["conditioning_base_objective"],
            final_penalty_objective=conditioning["conditioning_penalty_objective"],
            final_penalty_objective_ratio=conditioning["conditioning_penalty_objective_ratio"],
            final_total_grad_norm=conditioning["conditioning_total_grad_norm"],
            final_base_grad_norm=conditioning["conditioning_base_grad_norm"],
            final_penalty_grad_norm=conditioning["conditioning_penalty_grad_norm"],
            final_penalty_gradient_norm=conditioning["penalty_gradient_norm"],
            final_penalty_grad_ratio=conditioning["conditioning_penalty_grad_ratio"],
            final_hard_max_violation=float(routing_state.hard_max_violation),
            final_surrogate_max_value=float(routing_state.surrogate_max_value),
            hard_positive_shift_zero=bool(routing_state.hard_positive_shift_zero),
            signal_mismatch_active=bool(routing_state.signal_mismatch_active),
            multiplier_cap_binding=bool(cap_binding_detected),
            multiplier_cap_binding_indices=sorted(cap_binding_indices),
            penalty_cap_reached=bool(penalty_cap_reached),
            penalty_max=(
                None if settings.penalty_max is None else float(settings.penalty_max)
            ),
            penalty_cap_requested=(
                None if penalty_cap_requested is None else float(penalty_cap_requested)
            ),
        )

    def _try_penalty_increase() -> SimpleNamespace | None:
        """Apply penalty scaling and return a cap-reached result if the cap fires."""
        nonlocal penalty, penalty_cap_reached, penalty_cap_requested
        nonlocal feasible_stall_count, update_feasibility_tol, update_stationarity_tol
        next_p, cap_hit, requested_p = _next_penalty(
            penalty,
            penalty_scale=settings.penalty_scale,
            penalty_max=settings.penalty_max,
        )
        if cap_hit:
            penalty_cap_reached = True
            penalty_cap_requested = requested_p
            history_entry["action"] = "penalty_cap_reached"
            history_entry["trust_radius"] = trust_radius
            _emit_history_snapshot(history_entry)
            return _build_result(
                success=False,
                message=(
                    "ALM stopped after the requested penalty update "
                    f"{requested_p:.3e} exceeded the configured "
                    f"penalty cap {next_p:.3e}."
                ),
                termination_reason="penalty_cap_reached",
                outer_iterations=outer_iteration,
                evaluation=accepted_eval,
                multipliers_state=multipliers,
                penalty_state=penalty,
                inner_result=result,
                restored_best_feasible=False,
                restored_best_feasible_reason=None,
                final_max_feasibility_violation=max_feasibility_violation,
                final_stationarity_norm=stationarity_norm,
                final_raw_stationarity_norm=raw_stationarity_norm,
                final_kkt_stationarity_norm=kkt_stationarity_norm,
                final_feasibility_tolerance=settings.feasibility_tol,
                final_stationarity_tolerance=settings.stationarity_tol,
            )
        penalty = next_p
        feasible_stall_count = 0
        update_feasibility_tol = max(settings.feasibility_tol, 1.0 / penalty)
        update_stationarity_tol = max(settings.stationarity_tol, 1.0 / penalty)
        return None

    def _restore_best_feasible_on_failure(
        *,
        evaluation: dict,
        multipliers_state: np.ndarray,
        penalty_state: float,
        inner_result,
        final_max_feasibility_violation: float,
    ) -> SimpleNamespace:
        restored_best_feasible = False
        restored_best_feasible_reason = None
        restored_evaluation = evaluation
        restored_x = x.copy()
        restored_multipliers_state = np.asarray(multipliers_state, dtype=float).copy()
        restored_penalty_state = float(penalty_state)
        restored_inner_result = inner_result

        restore_reasons: list[str] = []
        if best_feasible is not None:
            if final_max_feasibility_violation > settings.feasibility_tol:
                restore_reasons.append("final_iterate_infeasible")
            if (
                _incumbent_objective_value(restored_evaluation)
                > _incumbent_objective_value(best_feasible.evaluation)
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
                restore_incumbent_state_fn(best_feasible.incumbent_state)

        return SimpleNamespace(
            x=restored_x,
            evaluation=restored_evaluation,
            multipliers_state=restored_multipliers_state,
            penalty_state=restored_penalty_state,
            inner_result=restored_inner_result,
            restored_best_feasible=bool(restored_best_feasible),
            restored_best_feasible_reason=restored_best_feasible_reason,
        )

    def _build_failure_result_with_optional_restore(
        *,
        termination_reason: str,
        message_prefix: str,
        evaluation: dict,
        multipliers_state: np.ndarray,
        penalty_state: float,
        inner_result,
        restored_message_prefix: str | None = None,
        restored_termination_reason: str | None = None,
    ) -> SimpleNamespace:
        nonlocal x
        restored_state = _restore_best_feasible_on_failure(
            evaluation=evaluation,
            multipliers_state=multipliers_state,
            penalty_state=penalty_state,
            inner_result=inner_result,
            final_max_feasibility_violation=_extract_constraint_state(evaluation)[3],
        )
        x = restored_state.x.copy()
        restored_routing_state = _constraint_routing_state(
            restored_state.evaluation,
            restored_state.multipliers_state,
            restored_state.penalty_state,
            settings.feasibility_tol,
        )
        (
            restored_raw_stationarity_norm,
            restored_kkt_stationarity_norm,
            restored_stationarity_norm,
            _restored_signal_mismatch_active,
        ) = _stationarity_metrics(
            restored_state.evaluation,
            restored_routing_state,
            settings.feasibility_tol,
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
        return _build_result(
            success=False,
            message=message,
            termination_reason=effective_termination_reason,
            outer_iterations=last_outer_iteration,
            evaluation=restored_state.evaluation,
            multipliers_state=restored_state.multipliers_state,
            penalty_state=restored_state.penalty_state,
            inner_result=restored_state.inner_result,
            restored_best_feasible=restored_state.restored_best_feasible,
            restored_best_feasible_reason=restored_state.restored_best_feasible_reason,
            final_max_feasibility_violation=restored_max_feasibility_violation,
            final_stationarity_norm=restored_stationarity_norm,
            final_raw_stationarity_norm=restored_raw_stationarity_norm,
            final_kkt_stationarity_norm=restored_kkt_stationarity_norm,
            final_feasibility_tolerance=settings.feasibility_tol,
            final_stationarity_tolerance=settings.stationarity_tol,
        )

    for outer_iteration in range(1, settings.max_outer_iterations + 1):
        last_outer_iteration = outer_iteration
        if outer_state_callback is not None:
            outer_state_callback(outer_iteration, multipliers.copy(), penalty)

        feasible_stall_count = 0
        is_final_outer = outer_iteration == settings.max_outer_iterations
        for continuation_iteration in range(settings.max_subproblem_continuations + 1):
            start_x = x.copy()
            current_eval = evaluate_problem(x, multipliers, penalty)
            _require_finite_evaluation(
                current_eval,
                context="ALM outer iterate evaluation",
            )
            effective_feasibility_tol = _effective_feasibility_gate(
                settings,
                update_feasibility_tol,
            )
            (
                current_solver_constraint_values,
                current_feasibility_values,
                current_dual_update_values,
                current_max_feasibility_violation,
            ) = _extract_constraint_state(current_eval)
            current_routing_state = _constraint_routing_state(
                current_eval,
                multipliers,
                penalty,
                effective_feasibility_tol,
            )
            (
                current_raw_stationarity_norm,
                current_kkt_stationarity_norm,
                current_stationarity_norm,
                current_signal_mismatch_active,
            ) = _stationarity_metrics(
                current_eval,
                current_routing_state,
                effective_feasibility_tol,
            )
            current_conditioning = _conditioning_metrics(current_eval)
            current_constraints_inactive_candidate = (
                current_routing_state.signal_state.explicit_stage2_signals
                and current_routing_state.hard_max_violation <= settings.feasibility_tol
                and not np.any(current_routing_state.surrogate_activity_mask)
                and current_routing_state.hard_positive_shift_zero
                and not current_signal_mismatch_active
            )

            if (
                current_max_feasibility_violation <= settings.feasibility_tol
                and current_stationarity_norm <= settings.stationarity_tol
                and not current_constraints_inactive_candidate
                and not current_signal_mismatch_active
            ):
                history.append(
                    {
                        "outer_iteration": int(outer_iteration),
                        "continuation_iteration": int(continuation_iteration),
                        "inner_iterations": 0,
                        "inner_success": True,
                        "inner_message": "ALM skipped inner solve; current iterate already satisfies the KKT stationarity gate.",
                        "penalty": float(penalty),
                        "max_violation": current_max_feasibility_violation,
                        "stationarity_norm": current_stationarity_norm,
                        "raw_stationarity_norm": current_raw_stationarity_norm,
                        "kkt_stationarity_norm": current_kkt_stationarity_norm,
                        "constraint_values": [float(value) for value in current_feasibility_values],
                        "solver_constraint_values": [
                            float(value) for value in current_solver_constraint_values
                        ],
                        "hard_signed_constraint_values": [
                            float(value)
                            for value in current_routing_state.signal_state.hard_signed_constraint_values
                        ],
                        "hard_violation_values": [
                            float(value)
                            for value in current_routing_state.signal_state.hard_violation_values
                        ],
                        "surrogate_signed_constraint_values": [
                            float(value)
                            for value in current_routing_state.signal_state.surrogate_signed_constraint_values
                        ],
                        "hard_max_violation": float(current_routing_state.hard_max_violation),
                        "surrogate_max_value": float(current_routing_state.surrogate_max_value),
                        "hard_positive_shift_zero": bool(
                            current_routing_state.hard_positive_shift_zero
                        ),
                        "signal_mismatch_active": bool(current_signal_mismatch_active),
                        "multipliers": [float(value) for value in multipliers],
                        "post_update_multipliers": [float(value) for value in multipliers],
                        "feasibility_tolerance": float(update_feasibility_tol),
                        "effective_feasibility_tolerance": float(effective_feasibility_tol),
                        "stationarity_tolerance": float(update_stationarity_tol),
                        "trust_radius": trust_radius,
                        "inner_maxiter": None,
                        "inner_maxls": None,
                        "inner_maxfun": None,
                        "inner_profile": None,
                        "inner_attempts": 0,
                        "accepted_move_norm": 0.0,
                        "accepted_move_norm_scaled": 0.0,
                        "infeasible_stall_move_tolerance": float(_move_tolerance(start_x)),
                        "objective_delta": 0.0,
                        "feasibility_delta": 0.0,
                        "feasibility_delta_tolerance": 0.0,
                        "stationarity_delta": 0.0,
                        "meaningful_progress": False,
                        "feasible_stall_count": 0,
                        "infeasible_stall": False,
                        "inner_false_success": False,
                        "inner_stall_reason": None,
                        "active_violation_index": None,
                        "active_constraint_name": None,
                        "nonfinite_candidate_evaluation": False,
                        "nonfinite_candidate_fields": None,
                        "multiplier_cap_binding": False,
                        "multiplier_cap_binding_indices": [],
                        "action": "converged",
                    }
                )
                history[-1].update(current_conditioning)
                _emit_history_snapshot(history[-1])
                final_eval = current_eval
                final_multipliers = multipliers.copy()
                final_penalty = penalty
                message = (
                    "ALM converged: "
                    f"max_violation={current_max_feasibility_violation:.3e}, "
                    f"stationarity={current_stationarity_norm:.3e}"
                )
                return _build_result(
                    success=True,
                    message=message,
                    termination_reason="converged",
                    outer_iterations=outer_iteration,
                    evaluation=current_eval,
                    multipliers_state=multipliers,
                    penalty_state=penalty,
                    inner_result=last_result,
                    restored_best_feasible=False,
                    restored_best_feasible_reason=None,
                    final_max_feasibility_violation=current_max_feasibility_violation,
                    final_stationarity_norm=current_stationarity_norm,
                    final_raw_stationarity_norm=current_raw_stationarity_norm,
                    final_kkt_stationarity_norm=current_kkt_stationarity_norm,
                    final_feasibility_tolerance=settings.feasibility_tol,
                    final_stationarity_tolerance=settings.stationarity_tol,
                )

            _cached_eval = SimpleNamespace(x=None, evaluation=None)

            def inner_fun(inner_x):
                evaluation = _sanitize_nonfinite_inner_evaluation(
                    evaluate_problem(inner_x, multipliers, penalty),
                    fallback_evaluation=current_eval,
                )
                _cached_eval.x = np.asarray(inner_x, dtype=float).copy()
                _cached_eval.evaluation = evaluation
                return float(evaluation["total"]), np.asarray(evaluation["grad"], dtype=float)

            def alm_inner_callback(inner_x):
                if inner_callback is not None:
                    inner_callback(inner_x)
                inner_x_arr = np.asarray(inner_x, dtype=float)
                if _cached_eval.x is not None and np.array_equal(inner_x_arr, _cached_eval.x):
                    evaluation = _cached_eval.evaluation
                else:
                    evaluation = _sanitize_nonfinite_inner_evaluation(
                        evaluate_problem(inner_x_arr, multipliers, penalty),
                        fallback_evaluation=current_eval,
                    )
                (
                    _solver_constraint_values,
                    callback_feasibility_values,
                    callback_dual_update_values,
                    callback_max_feasibility_violation,
                ) = _extract_constraint_state(evaluation)
                callback_routing_state = _constraint_routing_state(
                    evaluation,
                    multipliers,
                    penalty,
                    effective_feasibility_tol,
                )
                (
                    _callback_raw_stationarity_norm,
                    _callback_kkt_stationarity_norm,
                    callback_stationarity_norm,
                    _callback_signal_mismatch_active,
                ) = _stationarity_metrics(
                    evaluation,
                    callback_routing_state,
                    effective_feasibility_tol,
                )
                if (
                    callback_max_feasibility_violation <= effective_feasibility_tol
                    and callback_stationarity_norm <= update_stationarity_tol
                ):
                    raise _EarlyStopInnerSolve(inner_x, evaluation)

            accepted_result = None
            accepted_eval = None
            attempts = 0
            attempt_iterations = 0
            attempt_radius = trust_radius
            last_attempt_result = None
            current_feasible_enough = (
                current_max_feasibility_violation <= effective_feasibility_tol
            )
            last_inner_options = None
            last_inner_profile = None
            forced_infeasible_penalty_cycle = False
            forced_infeasible_penalty_reason = None
            forced_inner_false_success = False
            nonfinite_candidate_evaluation = False
            nonfinite_candidate_fields: list[str] | None = None

            for attempt_index in range(1, settings.max_inner_attempts + 1):
                attempts = attempt_index
                current_inner_profile = _select_inner_solve_profile(
                    trust_radius=attempt_radius,
                    continuation_iteration=continuation_iteration,
                    feasible_enough=current_feasible_enough,
                )
                inner_attempt_options = _build_inner_options(
                    inner_options,
                    update_stationarity_tol,
                    profile=current_inner_profile,
                )
                last_inner_options = dict(inner_attempt_options)
                last_inner_profile = current_inner_profile.name
                try:
                    if target_inner_optimizer is None:
                        result = minimize(
                            inner_fun,
                            x,
                            jac=True,
                            method="L-BFGS-B",
                            bounds=_build_box_bounds(x, attempt_radius),
                            callback=alm_inner_callback,
                            options=inner_attempt_options,
                        )
                        candidate_x = np.asarray(result.x, dtype=float).copy()
                    else:
                        result, candidate_x = _run_target_inner_solve(
                            evaluate_value_and_grad=inner_fun,
                            center=x,
                            attempt_radius=attempt_radius,
                            inner_attempt_options=inner_attempt_options,
                            method=target_inner_method,
                            optimizer=target_inner_optimizer,
                            inner_callback=inner_callback,
                        )
                    if _cached_eval.x is not None and np.array_equal(candidate_x, _cached_eval.x):
                        candidate_eval = _cached_eval.evaluation
                    else:
                        candidate_eval = _sanitize_nonfinite_inner_evaluation(
                            evaluate_problem(candidate_x, multipliers, penalty),
                            fallback_evaluation=current_eval,
                        )
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
                moved_norm = float(np.linalg.norm(candidate_x - x))
                move_tolerance = _move_tolerance(x)
                if candidate_eval.get("nonfinite_evaluation"):
                    nonfinite_candidate_evaluation = True
                    nonfinite_candidate_fields = list(candidate_eval.get("nonfinite_fields", []))
                acceptable = _candidate_is_acceptable(
                    current_eval,
                    candidate_eval,
                    result,
                    moved_norm,
                    update_feasibility_tol,
                )
                (
                    infeasible_inner_stall,
                    inner_false_success,
                    inner_stall_reason,
                ) = _classify_infeasible_inner_stall(
                    current_eval,
                    candidate_eval,
                    result,
                    moved_norm,
                    move_tolerance,
                    effective_feasibility_tol,
                )
                if acceptable and not infeasible_inner_stall:
                    accepted_result = result
                    accepted_eval = candidate_eval
                    if attempt_radius is not None:
                        if moved_norm >= 0.5 * float(attempt_radius):
                            trust_radius = float(attempt_radius) * float(settings.trust_radius_grow)
                        else:
                            trust_radius = float(attempt_radius)
                    break
                if infeasible_inner_stall:
                    accepted_result = result
                    accepted_eval = current_eval
                    forced_infeasible_penalty_cycle = True
                    forced_infeasible_penalty_reason = inner_stall_reason
                    forced_inner_false_success = bool(inner_false_success)
                    if attempt_radius is not None:
                        trust_radius = float(attempt_radius)
                    break
                if attempt_radius is None:
                    accepted_result = result
                    accepted_eval = current_eval
                    break
                next_radius = max(
                    settings.trust_radius_min,
                    float(attempt_radius) * float(settings.trust_radius_shrink),
                )
                exhausted_attempts = attempt_index == settings.max_inner_attempts
                if attempt_radius <= settings.trust_radius_min or exhausted_attempts:
                    accepted_result = result
                    accepted_eval = current_eval
                    trust_radius = float(attempt_radius)
                    break
                attempt_radius = float(next_radius)
                trust_radius = float(attempt_radius)
                continue

            if accepted_result is None or accepted_eval is None:
                if last_attempt_result is None:
                    raise RuntimeError("ALM failed before any inner optimization result was produced.")
                accepted_result = last_attempt_result
                accepted_eval = current_eval

            result = accepted_result
            last_result = result
            total_inner_iterations += attempt_iterations
            candidate_x = np.asarray(result.x, dtype=float).copy()
            if accepted_eval is current_eval:
                x = start_x.copy()
                final_eval = current_eval
            else:
                x = candidate_x
                final_eval = accepted_eval
                if accepted_callback is not None:
                    accepted_callback(x.copy())
            accepted_move_norm = float(np.linalg.norm(x - start_x))
            (
                solver_constraint_values,
                feasibility_values,
                dual_update_values,
                max_feasibility_violation,
            ) = _extract_constraint_state(final_eval)
            routing_state = _constraint_routing_state(
                final_eval,
                multipliers,
                penalty,
                update_feasibility_tol,
            )
            (
                raw_stationarity_norm,
                kkt_stationarity_norm,
                stationarity_norm,
                signal_mismatch_active,
            ) = _stationarity_metrics(
                final_eval,
                routing_state,
                update_feasibility_tol,
            )
            feasibility_delta, feasibility_delta_tol = _feasibility_improvement_metrics(
                current_max_feasibility_violation,
                max_feasibility_violation,
            )
            hard_signed_constraint_values = (
                routing_state.signal_state.hard_signed_constraint_values
            )
            hard_violation_values = routing_state.signal_state.hard_violation_values
            surrogate_signed_constraint_values = (
                routing_state.signal_state.surrogate_signed_constraint_values
            )
            active_violation_index = (
                None
                if hard_violation_values.size == 0
                or routing_state.hard_max_violation <= 0.0
                else int(np.argmax(hard_violation_values))
            )
            active_constraint_name = (
                None
                if active_violation_index is None
                else str(constraint_names[active_violation_index])
            )
            final_multipliers = multipliers.copy()
            final_penalty = penalty
            made_inner_progress = _made_meaningful_inner_progress(
                start_x,
                x,
                float(current_eval["total"]),
                float(final_eval["total"]),
                current_max_feasibility_violation,
                max_feasibility_violation,
                current_stationarity_norm,
                stationarity_norm,
            )

            if max_feasibility_violation <= settings.feasibility_tol:
                improves_best_feasible = (
                    best_feasible is None
                    or _incumbent_objective_value(final_eval)
                    < _incumbent_objective_value(best_feasible.evaluation)
                )
                if improves_best_feasible:
                    incumbent_state = (
                        None
                        if snapshot_accepted_state_fn is None
                        else snapshot_accepted_state_fn()
                    )
                    best_feasible = ALMFeasibleIncumbent(
                        x=x.copy(),
                        evaluation=final_eval,
                        multipliers=multipliers.copy(),
                        penalty=penalty,
                        inner_result=result,
                        incumbent_state=incumbent_state,
                    )

            history_entry = {
                "outer_iteration": int(outer_iteration),
                "continuation_iteration": int(continuation_iteration),
                "inner_iterations": int(attempt_iterations),
                "inner_success": bool(getattr(result, "success", False)),
                "inner_message": str(getattr(result, "message", "")),
                "penalty": float(penalty),
                "max_violation": max_feasibility_violation,
                "stationarity_norm": stationarity_norm,
                "raw_stationarity_norm": raw_stationarity_norm,
                "kkt_stationarity_norm": kkt_stationarity_norm,
                "constraint_values": [
                    float(value) for value in feasibility_values
                ],
                "solver_constraint_values": [float(value) for value in solver_constraint_values],
                "hard_signed_constraint_values": [
                    float(value) for value in hard_signed_constraint_values
                ],
                "hard_violation_values": [
                    float(value) for value in hard_violation_values
                ],
                "surrogate_signed_constraint_values": [
                    float(value) for value in surrogate_signed_constraint_values
                ],
                "hard_max_violation": float(routing_state.hard_max_violation),
                "surrogate_max_value": float(routing_state.surrogate_max_value),
                "hard_positive_shift_zero": bool(routing_state.hard_positive_shift_zero),
                "signal_mismatch_active": bool(signal_mismatch_active),
                "multipliers": [float(value) for value in multipliers],
                "post_update_multipliers": [float(value) for value in multipliers],
                "feasibility_tolerance": float(update_feasibility_tol),
                "effective_feasibility_tolerance": float(effective_feasibility_tol),
                "stationarity_tolerance": float(update_stationarity_tol),
                "trust_radius": trust_radius,
                "inner_maxiter": None
                if last_inner_options is None or "maxiter" not in last_inner_options
                else int(last_inner_options["maxiter"]),
                "inner_maxls": None
                if last_inner_options is None or "maxls" not in last_inner_options
                else int(last_inner_options["maxls"]),
                "inner_maxfun": None
                if last_inner_options is None or "maxfun" not in last_inner_options
                else int(last_inner_options["maxfun"]),
                "inner_profile": last_inner_profile,
                "inner_attempts": int(attempts),
                "accepted_move_norm": accepted_move_norm,
                "accepted_move_norm_scaled": accepted_move_norm / max(
                    1.0,
                    float(np.linalg.norm(start_x)),
                ),
                "infeasible_stall_move_tolerance": float(_move_tolerance(start_x)),
                "objective_delta": float(current_eval["total"]) - float(final_eval["total"]),
                "feasibility_delta": float(feasibility_delta),
                "feasibility_delta_tolerance": float(feasibility_delta_tol),
                "stationarity_delta": float(current_stationarity_norm) - float(stationarity_norm),
                "meaningful_progress": bool(made_inner_progress),
                "feasible_stall_count": int(feasible_stall_count),
                "infeasible_stall": bool(forced_infeasible_penalty_cycle),
                "inner_false_success": bool(forced_inner_false_success),
                "inner_stall_reason": forced_infeasible_penalty_reason,
                "active_violation_index": active_violation_index,
                "active_constraint_name": active_constraint_name,
                "nonfinite_candidate_evaluation": bool(nonfinite_candidate_evaluation),
                "nonfinite_candidate_fields": nonfinite_candidate_fields,
                "multiplier_cap_binding": False,
                "multiplier_cap_binding_indices": [],
            }
            history_entry.update(_conditioning_metrics(final_eval))
            history.append(history_entry)
            hard_feasible_strict = routing_state.hard_max_violation <= settings.feasibility_tol
            hard_feasible_for_update = (
                routing_state.hard_max_violation <= effective_feasibility_tol
            )
            constraints_inactive_candidate = (
                routing_state.signal_state.explicit_stage2_signals
                and hard_feasible_strict
                and not np.any(routing_state.surrogate_activity_mask)
                and routing_state.hard_positive_shift_zero
                and not signal_mismatch_active
            )

            if (
                max_feasibility_violation <= settings.feasibility_tol
                and stationarity_norm <= settings.stationarity_tol
                and not constraints_inactive_candidate
                and not signal_mismatch_active
            ):
                history_entry["action"] = "converged"
                _emit_history_snapshot(history_entry)
                message = (
                    "ALM converged: "
                    f"max_violation={max_feasibility_violation:.3e}, "
                    f"stationarity={stationarity_norm:.3e}"
                )
                return _build_result(
                    success=True,
                    message=message,
                    termination_reason="converged",
                    outer_iterations=outer_iteration,
                    evaluation=final_eval,
                    multipliers_state=multipliers,
                    penalty_state=penalty,
                    inner_result=result,
                    restored_best_feasible=False,
                    restored_best_feasible_reason=None,
                    final_max_feasibility_violation=max_feasibility_violation,
                    final_stationarity_norm=stationarity_norm,
                    final_raw_stationarity_norm=raw_stationarity_norm,
                    final_kkt_stationarity_norm=kkt_stationarity_norm,
                    final_feasibility_tolerance=settings.feasibility_tol,
                    final_stationarity_tolerance=settings.stationarity_tol,
                )

            if forced_infeasible_penalty_cycle:
                cap_result = _try_penalty_increase()
                if cap_result is not None:
                    return cap_result
                history_entry["action"] = "infeasible_stall_penalty_increase"
                history_entry["trust_radius"] = trust_radius
                if is_final_outer:
                    history_entry["outer_termination"] = "max_outer"
                _emit_history_snapshot(history_entry)
                break

            if constraints_inactive_candidate:
                if stationarity_norm <= settings.stationarity_tol:
                    history_entry["action"] = "constraints_inactive_converged"
                    history_entry["trust_radius"] = trust_radius
                    _emit_history_snapshot(history_entry)
                    message = (
                        "ALM converged with inactive hard constraints: "
                        f"max_violation={routing_state.hard_max_violation:.3e}, "
                        f"stationarity={stationarity_norm:.3e}"
                    )
                    return _build_result(
                        success=True,
                        message=message,
                        termination_reason="constraints_inactive_converged",
                        outer_iterations=outer_iteration,
                        evaluation=final_eval,
                        multipliers_state=multipliers,
                        penalty_state=penalty,
                        inner_result=result,
                        restored_best_feasible=False,
                        restored_best_feasible_reason=None,
                        final_max_feasibility_violation=max_feasibility_violation,
                        final_stationarity_norm=stationarity_norm,
                        final_raw_stationarity_norm=raw_stationarity_norm,
                        final_kkt_stationarity_norm=kkt_stationarity_norm,
                        final_feasibility_tolerance=settings.feasibility_tol,
                        final_stationarity_tolerance=settings.stationarity_tol,
                    )
                if not made_inner_progress and continuation_iteration > 0:
                    history_entry["action"] = "constraints_inactive_stall"
                    history_entry["trust_radius"] = trust_radius
                    _emit_history_snapshot(history_entry)
                    return _build_failure_result_with_optional_restore(
                        termination_reason="constraints_inactive_stall",
                        message_prefix=(
                            "ALM stopped after hard constraints became inactive without "
                            "further stationarity progress"
                        ),
                        evaluation=final_eval,
                        multipliers_state=multipliers,
                        penalty_state=penalty,
                        inner_result=result,
                    )

            if signal_mismatch_active and hard_feasible_strict:
                if not made_inner_progress or continuation_iteration > 0:
                    if routing_state.surrogate_positive_shift_zero:
                        history_entry["action"] = "signal_mismatch_stall"
                        history_entry["trust_radius"] = trust_radius
                        _emit_history_snapshot(history_entry)
                        return _build_failure_result_with_optional_restore(
                            termination_reason="signal_mismatch_stall",
                            message_prefix=(
                                "ALM stopped after hard-feasible and surrogate-active "
                                "signals repeated without corrective progress"
                            ),
                            evaluation=final_eval,
                            multipliers_state=multipliers,
                            penalty_state=penalty,
                            inner_result=result,
                        )
                    cap_result = _try_penalty_increase()
                    if cap_result is not None:
                        return cap_result
                    history_entry["action"] = "signal_mismatch_penalty_increase"
                    history_entry["trust_radius"] = trust_radius
                    if is_final_outer:
                        history_entry["outer_termination"] = "max_outer"
                    _emit_history_snapshot(history_entry)
                    break
                feasible_stall_count = 0
                if trust_radius is not None:
                    trust_radius = max(
                        trust_radius,
                        max(
                            settings.trust_radius_min,
                            float(trust_radius) * float(settings.trust_radius_grow),
                        ),
                    )
                history_entry["subproblem_limit_reason"] = None
                history_entry["action"] = "subproblem_continue"
                history_entry["trust_radius"] = trust_radius
                history_entry["feasible_stall_count"] = int(feasible_stall_count)
                _emit_history_snapshot(history_entry)
                continue

            if hard_feasible_for_update and stationarity_norm <= update_stationarity_tol:
                feasible_stall_count = 0
                (
                    multipliers,
                    multiplier_cap_binding,
                    multiplier_cap_binding_indices,
                ) = _project_nonnegative_multipliers_with_diagnostics(
                    multipliers,
                    routing_state.signal_state.preferred_dual_update_values,
                    penalty,
                    settings.multiplier_max,
                )
                history_entry["post_update_multipliers"] = [float(value) for value in multipliers]
                history_entry["multiplier_cap_binding"] = bool(multiplier_cap_binding)
                history_entry["multiplier_cap_binding_indices"] = list(multiplier_cap_binding_indices)
                if multiplier_cap_binding:
                    cap_binding_detected = True
                    cap_binding_indices.update(multiplier_cap_binding_indices)
                update_feasibility_tol = max(
                    update_feasibility_tol / float(settings.penalty_scale),
                    settings.feasibility_tol,
                )
                update_stationarity_tol = max(
                    update_stationarity_tol / float(settings.penalty_scale),
                    settings.stationarity_tol,
                )
                history_entry["action"] = "dual_update"
                history_entry["trust_radius"] = trust_radius
                if is_final_outer:
                    history_entry["outer_termination"] = "max_outer"
                _emit_history_snapshot(history_entry)
                break

            if hard_feasible_for_update:
                feasible_stall_count = 0 if made_inner_progress else feasible_stall_count + 1
                hit_stall_limit = (
                    continuation_iteration == settings.max_subproblem_continuations
                    or feasible_stall_count >= _PLATEAU_STALL_LIMIT
                )
                if hit_stall_limit:
                    history_entry["subproblem_limit_reason"] = (
                        "plateau_stall"
                        if feasible_stall_count >= _PLATEAU_STALL_LIMIT
                        else "max_subproblem_continuations"
                    )
                    history_entry["trust_radius"] = trust_radius
                    history_entry["feasible_stall_count"] = int(feasible_stall_count)
                    history_entry["action"] = "subproblem_limit"
                    if is_final_outer:
                        history_entry["outer_termination"] = "max_outer"
                    _emit_history_snapshot(history_entry)
                    break
                if trust_radius is not None:
                    trust_radius = max(
                        trust_radius,
                        max(
                            settings.trust_radius_min,
                            float(trust_radius) * float(settings.trust_radius_grow),
                        ),
                    )
                if not made_inner_progress:
                    update_stationarity_tol = max(
                        update_stationarity_tol,
                        max(settings.stationarity_tol, 0.5 * stationarity_norm),
                    )
                history_entry["subproblem_limit_reason"] = None
                history_entry["action"] = "subproblem_continue"
                history_entry["trust_radius"] = trust_radius
                history_entry["feasible_stall_count"] = int(feasible_stall_count)
                _emit_history_snapshot(history_entry)
                continue

            cap_result = _try_penalty_increase()
            if cap_result is not None:
                return cap_result
            history_entry["action"] = "penalty_increase"
            history_entry["trust_radius"] = trust_radius
            if is_final_outer:
                history_entry["outer_termination"] = "max_outer"
            _emit_history_snapshot(history_entry)
            break

        if is_final_outer:
            break

    if final_eval is None or last_result is None:
        raise RuntimeError("ALM failed before any inner optimization result was produced.")

    termination_reason = _termination_reason_from_history(
        history,
        success=False,
        restored_best_feasible=False,
    )
    return _build_failure_result_with_optional_restore(
        termination_reason=termination_reason,
        evaluation=final_eval,
        multipliers_state=final_multipliers,
        penalty_state=final_penalty,
        inner_result=last_result,
        message_prefix=(
            "ALM exhausted outer iterations (max outer iterations reached)"
        ),
        restored_message_prefix=(
            "ALM exhausted outer iterations after restoring best feasible iterate "
            "(max outer iterations reached)"
        ),
        restored_termination_reason="max_outer_restored_best_feasible",
    )
