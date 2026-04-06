from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ALMSettings:
    max_outer_iterations: int = 10
    penalty_init: float = 1.0
    penalty_scale: float = 10.0
    feasibility_tol: float = 1e-6
    stationarity_tol: float = 1e-6
    trust_radius_init: float | None = None
    trust_radius_min: float = 1e-4
    trust_radius_shrink: float = 0.5
    trust_radius_grow: float = 1.5
    max_inner_attempts: int = 4


def validate_alm_cli_args(args) -> None:
    if args.alm_max_outer_iters <= 0:
        raise ValueError("--alm-max-outer-iters must be positive")
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


def zero_gradient_like(reference_grad):
    return np.zeros_like(np.asarray(reference_grad))


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _max_value(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else 0.0


def _build_box_bounds(center: np.ndarray, trust_radius: float | None):
    if trust_radius is None:
        return None
    widths = float(trust_radius) * np.maximum(1.0, np.abs(np.asarray(center, dtype=float)))
    return [
        (float(value - width), float(value + width))
        for value, width in zip(np.asarray(center, dtype=float), widths)
    ]


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


def minimize_alm(
    x0,
    constraint_names: Sequence[str],
    evaluate_problem: Callable[[np.ndarray, np.ndarray, float], dict],
    settings: ALMSettings,
    inner_options: dict,
    inner_callback: Callable[[np.ndarray], None] | None = None,
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
    trust_radius = (
        None
        if settings.trust_radius_init is None
        else float(settings.trust_radius_init)
    )
    update_feasibility_tol = max(settings.feasibility_tol, 1.0 / penalty)
    update_stationarity_tol = max(settings.stationarity_tol, 1.0 / penalty)

    for outer_iteration in range(1, settings.max_outer_iterations + 1):
        if outer_state_callback is not None:
            outer_state_callback(outer_iteration, multipliers.copy(), penalty)

        start_x = x.copy()
        current_eval = evaluate_problem(x, multipliers, penalty)

        def inner_fun(inner_x):
            evaluation = evaluate_problem(inner_x, multipliers, penalty)
            return float(evaluation["total"]), np.asarray(evaluation["grad"], dtype=float)

        accepted_result = None
        accepted_eval = None
        attempts = 0
        attempt_iterations = 0
        attempt_radius = trust_radius
        last_attempt_result = None

        for attempt_index in range(1, settings.max_inner_attempts + 1):
            attempts = attempt_index
            result = minimize(
                inner_fun,
                x,
                jac=True,
                method="L-BFGS-B",
                bounds=_build_box_bounds(x, attempt_radius),
                callback=inner_callback,
                options=inner_options,
            )
            last_attempt_result = result
            attempt_iterations += int(getattr(result, "nit", 0))
            candidate_x = np.asarray(result.x, dtype=float).copy()
            candidate_eval = evaluate_problem(candidate_x, multipliers, penalty)
            candidate_total = float(candidate_eval["total"])
            moved_norm = float(np.linalg.norm(candidate_x - x))
            acceptable = candidate_total <= float(current_eval["total"]) + 1e-12 and (
                bool(getattr(result, "success", False))
                or int(getattr(result, "nit", 0)) > 0
                or moved_norm > 1e-12
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
                trust_radius = float(next_radius)
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
        accepted_move_norm = float(np.linalg.norm(x - start_x))
        (
            solver_constraint_values,
            feasibility_values,
            dual_update_values,
            max_feasibility_violation,
        ) = _extract_constraint_state(final_eval)
        stationarity_norm = float(final_eval["stationarity_norm"])
        final_multipliers = multipliers.copy()
        final_penalty = penalty

        history_entry = {
            "outer_iteration": int(outer_iteration),
            "inner_iterations": int(attempt_iterations),
            "inner_success": bool(getattr(result, "success", False)),
            "inner_message": str(getattr(result, "message", "")),
            "penalty": float(penalty),
            "max_violation": max_feasibility_violation,
            "stationarity_norm": stationarity_norm,
            "constraint_values": [
                float(value) for value in feasibility_values
            ],
            "solver_constraint_values": [float(value) for value in solver_constraint_values],
            "multipliers": [float(value) for value in multipliers],
            "feasibility_tolerance": float(update_feasibility_tol),
            "stationarity_tolerance": float(update_stationarity_tol),
            "trust_radius": trust_radius,
            "inner_attempts": int(attempts),
            "accepted_move_norm": accepted_move_norm,
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
            return SimpleNamespace(
                x=x,
                success=True,
                message=message,
                nit=total_inner_iterations,
                outer_iterations=outer_iteration,
                constraint_names=list(constraint_names),
                constraint_values=[
                    float(value) for value in feasibility_values
                ],
                solver_constraint_values=[float(value) for value in solver_constraint_values],
                multipliers=[float(value) for value in multipliers],
                penalty=float(penalty),
                trust_radius=trust_radius,
                history=history,
                inner_result=result,
                final_objective=float(final_eval["total"]),
            )

        if outer_iteration == settings.max_outer_iterations:
            history_entry["action"] = "max_outer"
            break

        if (
            max_feasibility_violation <= update_feasibility_tol
            and stationarity_norm <= update_stationarity_tol
        ):
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
        elif max_feasibility_violation <= update_feasibility_tol:
            history_entry["action"] = "subproblem_continue"
        else:
            penalty *= float(settings.penalty_scale)
            update_feasibility_tol = max(settings.feasibility_tol, 1.0 / penalty)
            update_stationarity_tol = max(settings.stationarity_tol, 1.0 / penalty)
            history_entry["action"] = "penalty_increase"

    if final_eval is None or last_result is None:
        raise RuntimeError("ALM failed before any inner optimization result was produced.")

    (
        solver_constraint_values,
        feasibility_values,
        _dual_update_values,
        max_feasibility_violation,
    ) = _extract_constraint_state(final_eval)
    stationarity_norm = float(final_eval["stationarity_norm"])
    message = (
        "ALM reached max outer iterations: "
        f"max_violation={max_feasibility_violation:.3e}, "
        f"stationarity={stationarity_norm:.3e}"
    )
    return SimpleNamespace(
        x=x,
        success=False,
        message=message,
        nit=total_inner_iterations,
        outer_iterations=settings.max_outer_iterations,
        constraint_names=list(constraint_names),
        constraint_values=[
            float(value) for value in feasibility_values
        ],
        solver_constraint_values=[float(value) for value in solver_constraint_values],
        multipliers=[float(value) for value in final_multipliers],
        penalty=float(final_penalty),
        trust_radius=trust_radius,
        history=history,
        inner_result=last_result,
        final_objective=float(final_eval["total"]),
    )
