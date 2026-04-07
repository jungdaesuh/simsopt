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
    feasibility_tol: float = 1e-6
    stationarity_tol: float = 1e-6
    trust_radius_init: float | None = None
    trust_radius_min: float = 1e-4
    trust_radius_shrink: float = 0.5
    trust_radius_grow: float = 1.5
    max_inner_attempts: int = 4


@dataclass(frozen=True)
class ALMInnerSolveProfile:
    name: str
    maxiter_cap: int | None
    maxls_cap: int | None
    ftol_floor: float | None
    default_maxls: int = 20


_UNBOUNDED_INNER_PROFILE = ALMInnerSolveProfile(
    name="unbounded",
    maxiter_cap=None,
    maxls_cap=None,
    ftol_floor=None,
)
_BOXED_INFEASIBLE_INITIAL_PROFILE = ALMInnerSolveProfile(
    name="boxed_infeasible_initial",
    maxiter_cap=16,
    maxls_cap=12,
    ftol_floor=1e-13,
)
_BOXED_INFEASIBLE_CONTINUATION_PROFILE = ALMInnerSolveProfile(
    name="boxed_infeasible_continuation",
    maxiter_cap=10,
    maxls_cap=10,
    ftol_floor=1e-13,
)
_BOXED_FEASIBLE_INITIAL_PROFILE = ALMInnerSolveProfile(
    name="boxed_feasible_initial",
    maxiter_cap=12,
    maxls_cap=8,
    ftol_floor=1e-11,
)
_BOXED_FEASIBLE_CONTINUATION_PROFILE = ALMInnerSolveProfile(
    name="boxed_feasible_continuation",
    maxiter_cap=8,
    maxls_cap=8,
    ftol_floor=1e-11,
)
_DEFAULT_TAYLOR_EPSILONS = tuple(float(0.5**power) for power in range(7, 13))
_BOXED_INNER_PROFILES = {
    (False, False): _BOXED_INFEASIBLE_INITIAL_PROFILE,
    (False, True): _BOXED_INFEASIBLE_CONTINUATION_PROFILE,
    (True, False): _BOXED_FEASIBLE_INITIAL_PROFILE,
    (True, True): _BOXED_FEASIBLE_CONTINUATION_PROFILE,
}
_ACCEPTANCE_TOTAL_ATOL = 1e-10
_ACCEPTANCE_TOTAL_RTOL = 1e-3
_ACCEPTANCE_MOVE_TOL = 1e-12
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
    constraint_values = np.asarray(constraint_values, dtype=float)
    multipliers = np.asarray(multipliers, dtype=float)
    total_value = float(base_value) + float(np.dot(multipliers, constraint_values))
    total_value += 0.5 * float(penalty) * float(np.dot(constraint_values, constraint_values))

    total_grad = np.array(base_grad, copy=True)
    for multiplier, constraint_value, constraint_grad in zip(
        multipliers, constraint_values, constraint_grads
    ):
        weight = float(multiplier) + float(penalty) * float(constraint_value)
        total_grad = total_grad + weight * np.asarray(constraint_grad)

    return {
        "total": total_value,
        "grad": total_grad,
        "constraint_values": constraint_values,
        "max_violation": float(np.max(constraint_values)) if constraint_values.size > 0 else 0.0,
        "stationarity_norm": float(np.linalg.norm(total_grad)),
    }


def augmented_inequality_objective(
    base_value: float,
    base_grad,
    constraint_values,
    constraint_grads,
    multipliers,
    penalty: float,
):
    constraint_values = np.asarray(constraint_values, dtype=float)
    multipliers = np.asarray(multipliers, dtype=float)
    positive_shift = np.maximum(0.0, multipliers + float(penalty) * constraint_values)

    total_value = float(base_value)
    if constraint_values.size > 0:
        total_value += 0.5 / float(penalty) * float(
            np.dot(positive_shift, positive_shift) - np.dot(multipliers, multipliers)
        )

    total_grad = np.array(base_grad, copy=True)
    for active_multiplier, constraint_grad in zip(positive_shift, constraint_grads):
        total_grad = total_grad + float(active_multiplier) * np.asarray(
            constraint_grad,
            dtype=float,
        )

    return {
        "total": total_value,
        "grad": total_grad,
        "constraint_values": constraint_values,
        "max_violation": float(np.max(constraint_values)) if constraint_values.size > 0 else 0.0,
        "stationarity_norm": float(np.linalg.norm(total_grad)),
    }


def zero_gradient_like(reference_grad):
    return np.zeros_like(np.asarray(reference_grad))


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _max_value(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else 0.0


def _made_meaningful_inner_progress(
    start_x: np.ndarray,
    end_x: np.ndarray,
    current_total: float,
    final_total: float,
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

    return moved or improved_objective or improved_stationarity


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


def _should_converge_on_feasible_plateau(
    *,
    max_feasibility_violation: float,
    stationarity_norm: float,
    settings: ALMSettings,
    update_stationarity_tol: float,
    initial_update_stationarity_tol: float,
) -> bool:
    return (
        float(max_feasibility_violation) <= float(settings.feasibility_tol)
        and float(stationarity_norm) <= float(update_stationarity_tol)
        and float(update_stationarity_tol) < float(initial_update_stationarity_tol)
    )


def _normalize_trust_radius(trust_radius: float | None) -> float | None:
    if trust_radius is None:
        return None
    normalized = float(trust_radius)
    if normalized <= 0.0:
        return None
    return normalized


def _build_box_bounds(center: np.ndarray, trust_radius: float | None):
    normalized_trust_radius = _normalize_trust_radius(trust_radius)
    if normalized_trust_radius is None:
        return None
    widths = normalized_trust_radius * np.maximum(
        1.0, np.abs(np.asarray(center, dtype=float))
    )
    return [
        (float(value - width), float(value + width))
        for value, width in zip(np.asarray(center, dtype=float), widths)
    ]


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
    dual_update_values: np.ndarray,
    feasibility_values: np.ndarray,
    feasibility_gate: float,
) -> tuple[float, float | None, float]:
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
    kkt_stationarity_norm = _kkt_stationarity_norm(
        metric_grad,
        evaluation.get("constraint_grads"),
        dual_update_values,
        feasibility_values,
        _constraint_activity_tolerances(evaluation, dual_update_values),
        feasibility_gate,
    )
    effective_stationarity_norm = (
        raw_stationarity_norm
        if kkt_stationarity_norm is None
        else min(raw_stationarity_norm, kkt_stationarity_norm)
    )
    return raw_stationarity_norm, kkt_stationarity_norm, effective_stationarity_norm


def minimize_alm(
    x0,
    constraint_names: Sequence[str],
    evaluate_problem: Callable[[np.ndarray, np.ndarray, float], dict],
    settings: ALMSettings,
    inner_options: dict,
    inner_callback: Callable[[np.ndarray], None] | None = None,
    accepted_callback: Callable[[np.ndarray], None] | None = None,
    outer_state_callback: Callable[[int, np.ndarray, float], None] | None = None,
):
    x = np.asarray(x0, dtype=float).copy()
    multipliers = np.zeros(len(constraint_names), dtype=float)
    penalty = float(settings.penalty_init)
    total_inner_iterations = 0
    history = []
    final_eval = None
    last_result = None
    final_multipliers = multipliers.copy()
    final_penalty = penalty
    last_outer_iteration = 0
    trust_radius = _normalize_trust_radius(settings.trust_radius_init)
    update_feasibility_tol = max(settings.feasibility_tol, 1.0 / penalty)
    update_stationarity_tol = max(settings.stationarity_tol, 1.0 / penalty)
    initial_update_stationarity_tol = float(update_stationarity_tol)
    termination_reason = "max_outer"

    def _build_result(
        *,
        success: bool,
        message: str,
        outer_iterations: int,
        evaluation: dict,
        multipliers_state: np.ndarray,
        penalty_state: float,
        inner_result,
    ):
        (
            solver_constraint_values,
            feasibility_values,
            _dual_update_values,
            _max_feasibility_violation,
        ) = _extract_constraint_state(evaluation)
        return SimpleNamespace(
            x=x.copy(),
            success=bool(success),
            message=message,
            nit=total_inner_iterations,
            outer_iterations=int(outer_iterations),
            constraint_names=list(constraint_names),
            constraint_values=[float(value) for value in feasibility_values],
            solver_constraint_values=[float(value) for value in solver_constraint_values],
            multipliers=[float(value) for value in multipliers_state],
            penalty=float(penalty_state),
            trust_radius=trust_radius,
            history=history,
            inner_result=inner_result,
            final_objective=float(evaluation["total"]),
        )

    for outer_iteration in range(1, settings.max_outer_iterations + 1):
        last_outer_iteration = outer_iteration
        if outer_state_callback is not None:
            outer_state_callback(outer_iteration, multipliers.copy(), penalty)

        feasible_stall_count = 0
        for continuation_iteration in range(settings.max_subproblem_continuations + 1):
            start_x = x.copy()
            current_eval = evaluate_problem(x, multipliers, penalty)
            (
                current_solver_constraint_values,
                current_feasibility_values,
                current_dual_update_values,
                current_max_feasibility_violation,
            ) = _extract_constraint_state(current_eval)
            (
                current_raw_stationarity_norm,
                current_kkt_stationarity_norm,
                current_stationarity_norm,
            ) = _stationarity_metrics(
                current_eval,
                current_dual_update_values,
                current_feasibility_values,
                update_feasibility_tol,
            )

            if (
                current_max_feasibility_violation <= settings.feasibility_tol
                and current_stationarity_norm <= settings.stationarity_tol
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
                        "multipliers": [float(value) for value in multipliers],
                        "feasibility_tolerance": float(update_feasibility_tol),
                        "stationarity_tolerance": float(update_stationarity_tol),
                        "trust_radius": trust_radius,
                        "inner_attempts": 0,
                        "accepted_move_norm": 0.0,
                        "action": "converged",
                    }
                )
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
                    outer_iterations=outer_iteration,
                    evaluation=current_eval,
                    multipliers_state=multipliers,
                    penalty_state=penalty,
                    inner_result=last_result,
                )

            def inner_fun(inner_x):
                evaluation = evaluate_problem(inner_x, multipliers, penalty)
                return float(evaluation["total"]), np.asarray(evaluation["grad"], dtype=float)

            def alm_inner_callback(inner_x):
                if inner_callback is not None:
                    inner_callback(inner_x)
                evaluation = evaluate_problem(np.asarray(inner_x, dtype=float), multipliers, penalty)
                (
                    _solver_constraint_values,
                    callback_feasibility_values,
                    callback_dual_update_values,
                    callback_max_feasibility_violation,
                ) = _extract_constraint_state(evaluation)
                (
                    _callback_raw_stationarity_norm,
                    _callback_kkt_stationarity_norm,
                    callback_stationarity_norm,
                ) = _stationarity_metrics(
                    evaluation,
                    callback_dual_update_values,
                    callback_feasibility_values,
                    update_feasibility_tol,
                )
                if (
                    callback_max_feasibility_violation <= update_feasibility_tol
                    and callback_stationarity_norm <= update_stationarity_tol
                ):
                    raise _EarlyStopInnerSolve(inner_x, evaluation)

            accepted_result = None
            accepted_eval = None
            attempts = 0
            attempt_iterations = 0
            attempt_radius = trust_radius
            last_attempt_result = None
            current_feasible_enough = current_max_feasibility_violation <= update_feasibility_tol
            last_inner_options = None
            last_inner_profile = None

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
                    candidate_eval = evaluate_problem(candidate_x, multipliers, penalty)
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
                acceptable = _candidate_is_acceptable(
                    current_eval,
                    candidate_eval,
                    result,
                    moved_norm,
                    update_feasibility_tol,
                )
                if acceptable:
                    accepted_result = result
                    accepted_eval = candidate_eval
                    if attempt_radius is not None:
                        if moved_norm >= 0.5 * float(attempt_radius):
                            trust_radius = float(attempt_radius) * float(settings.trust_radius_grow)
                        else:
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
            (
                raw_stationarity_norm,
                kkt_stationarity_norm,
                stationarity_norm,
            ) = _stationarity_metrics(
                final_eval,
                dual_update_values,
                feasibility_values,
                update_feasibility_tol,
            )
            final_multipliers = multipliers.copy()
            final_penalty = penalty
            made_inner_progress = _made_meaningful_inner_progress(
                start_x,
                x,
                float(current_eval["total"]),
                float(final_eval["total"]),
                current_stationarity_norm,
                stationarity_norm,
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
                "multipliers": [float(value) for value in multipliers],
                "feasibility_tolerance": float(update_feasibility_tol),
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
                "objective_delta": float(current_eval["total"]) - float(final_eval["total"]),
                "stationarity_delta": float(current_stationarity_norm) - float(stationarity_norm),
                "meaningful_progress": bool(made_inner_progress),
                "feasible_stall_count": int(feasible_stall_count),
            }
            history.append(history_entry)

            if (
                max_feasibility_violation <= settings.feasibility_tol
                and stationarity_norm <= settings.stationarity_tol
            ):
                history_entry["action"] = "converged"
                message = (
                    "ALM converged: "
                    f"max_violation={max_feasibility_violation:.3e}, "
                    f"stationarity={stationarity_norm:.3e}"
                )
                return _build_result(
                    success=True,
                    message=message,
                    outer_iterations=outer_iteration,
                    evaluation=final_eval,
                    multipliers_state=multipliers,
                    penalty_state=penalty,
                    inner_result=result,
                )

            if (
                max_feasibility_violation <= update_feasibility_tol
                and stationarity_norm <= update_stationarity_tol
                and made_inner_progress
            ):
                feasible_stall_count = 0
                multipliers = np.maximum(0.0, multipliers + penalty * dual_update_values)
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
                break

            if max_feasibility_violation <= update_feasibility_tol:
                feasible_stall_count = 0 if made_inner_progress else feasible_stall_count + 1
                plateau_converged = _should_converge_on_feasible_plateau(
                    max_feasibility_violation=max_feasibility_violation,
                    stationarity_norm=stationarity_norm,
                    settings=settings,
                    update_stationarity_tol=update_stationarity_tol,
                    initial_update_stationarity_tol=initial_update_stationarity_tol,
                )
                hit_stall_limit = (
                    continuation_iteration == settings.max_subproblem_continuations
                    or feasible_stall_count >= _PLATEAU_STALL_LIMIT
                )
                if hit_stall_limit:
                    history_entry["trust_radius"] = trust_radius
                    history_entry["feasible_stall_count"] = int(feasible_stall_count)
                    if plateau_converged:
                        history_entry["action"] = "feasible_plateau"
                        termination_reason = "feasible_plateau"
                        break
                    history_entry["action"] = "subproblem_limit"
                    termination_reason = "subproblem_limit"
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
                history_entry["action"] = "subproblem_continue"
                history_entry["trust_radius"] = trust_radius
                history_entry["feasible_stall_count"] = int(feasible_stall_count)
                continue

            penalty *= float(settings.penalty_scale)
            feasible_stall_count = 0
            update_feasibility_tol = max(settings.feasibility_tol, 1.0 / penalty)
            update_stationarity_tol = max(settings.stationarity_tol, 1.0 / penalty)
            history_entry["action"] = "penalty_increase"
            history_entry["trust_radius"] = trust_radius
            break

        if termination_reason in {"subproblem_limit", "feasible_plateau"}:
            break

        if (
            outer_iteration == settings.max_outer_iterations
            and termination_reason == "max_outer"
        ):
            history[-1]["action"] = "max_outer"
            break

    if final_eval is None or last_result is None:
        raise RuntimeError("ALM failed before any inner optimization result was produced.")

    (
        solver_constraint_values,
        feasibility_values,
        _dual_update_values,
        max_feasibility_violation,
    ) = _extract_constraint_state(final_eval)
    stationarity_norm = float(final_eval["stationarity_norm"])
    if termination_reason == "feasible_plateau":
        message = (
            "ALM reached a feasible stationary plateau before final tolerance: "
            f"max_violation={max_feasibility_violation:.3e}, "
            f"stationarity={stationarity_norm:.3e}, "
            f"final_feas_tol={settings.feasibility_tol:.3e}, "
            f"final_stationarity_tol={settings.stationarity_tol:.3e}"
        )
    elif termination_reason == "subproblem_limit":
        message = (
            "ALM reached subproblem continuation limit: "
            f"max_violation={max_feasibility_violation:.3e}, "
            f"stationarity={stationarity_norm:.3e}"
        )
    else:
        message = (
            "ALM reached max outer iterations: "
            f"max_violation={max_feasibility_violation:.3e}, "
            f"stationarity={stationarity_norm:.3e}"
        )
    return _build_result(
        success=False,
        message=message,
        outer_iterations=last_outer_iteration,
        evaluation=final_eval,
        multipliers_state=final_multipliers,
        penalty_state=final_penalty,
        inner_result=last_result,
    )
