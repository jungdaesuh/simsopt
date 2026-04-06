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


def positive_part(value: float) -> float:
    return float(max(value, 0.0))


def upper_bound_residual(metric_value: float, upper_bound: float) -> float:
    return positive_part(metric_value - upper_bound)


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

    for outer_iteration in range(1, settings.max_outer_iterations + 1):
        if outer_state_callback is not None:
            outer_state_callback(outer_iteration, multipliers.copy(), penalty)

        def inner_fun(inner_x):
            evaluation = evaluate_problem(inner_x, multipliers, penalty)
            return float(evaluation["total"]), np.asarray(evaluation["grad"], dtype=float)

        result = minimize(
            inner_fun,
            x,
            jac=True,
            method="L-BFGS-B",
            callback=inner_callback,
            options=inner_options,
        )
        last_result = result
        total_inner_iterations += int(getattr(result, "nit", 0))
        x = np.asarray(result.x, dtype=float).copy()
        final_eval = evaluate_problem(x, multipliers, penalty)
        final_multipliers = multipliers.copy()
        final_penalty = penalty

        history_entry = {
            "outer_iteration": int(outer_iteration),
            "inner_iterations": int(getattr(result, "nit", 0)),
            "inner_success": bool(getattr(result, "success", False)),
            "inner_message": str(getattr(result, "message", "")),
            "penalty": float(penalty),
            "max_violation": float(final_eval["max_violation"]),
            "stationarity_norm": float(final_eval["stationarity_norm"]),
            "constraint_values": [
                float(value) for value in np.asarray(final_eval["constraint_values"], dtype=float)
            ],
            "multipliers": [float(value) for value in multipliers],
        }
        history.append(history_entry)

        if (
            final_eval["max_violation"] <= settings.feasibility_tol
            and final_eval["stationarity_norm"] <= settings.stationarity_tol
        ):
            message = (
                "ALM converged: "
                f"max_violation={final_eval['max_violation']:.3e}, "
                f"stationarity={final_eval['stationarity_norm']:.3e}"
            )
            return SimpleNamespace(
                x=x,
                success=True,
                message=message,
                nit=total_inner_iterations,
                outer_iterations=outer_iteration,
                constraint_names=list(constraint_names),
                constraint_values=[
                    float(value) for value in np.asarray(final_eval["constraint_values"], dtype=float)
                ],
                multipliers=[float(value) for value in multipliers],
                penalty=float(penalty),
                history=history,
                inner_result=result,
                final_objective=float(final_eval["total"]),
            )

        if outer_iteration == settings.max_outer_iterations:
            break

        multipliers = np.maximum(
            0.0,
            multipliers + penalty * np.asarray(final_eval["constraint_values"], dtype=float),
        )
        penalty *= float(settings.penalty_scale)

    if final_eval is None or last_result is None:
        raise RuntimeError("ALM failed before any inner optimization result was produced.")

    message = (
        "ALM reached max outer iterations: "
        f"max_violation={final_eval['max_violation']:.3e}, "
        f"stationarity={final_eval['stationarity_norm']:.3e}"
    )
    return SimpleNamespace(
        x=x,
        success=False,
        message=message,
        nit=total_inner_iterations,
        outer_iterations=settings.max_outer_iterations,
        constraint_names=list(constraint_names),
        constraint_values=[
            float(value) for value in np.asarray(final_eval["constraint_values"], dtype=float)
        ],
        multipliers=[float(value) for value in final_multipliers],
        penalty=float(final_penalty),
        history=history,
        inner_result=last_result,
        final_objective=float(final_eval["total"]),
    )
