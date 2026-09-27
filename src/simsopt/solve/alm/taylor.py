"""Directional Taylor test of an ALM evaluator's gradient.

:func:`run_directional_taylor_test` compares ``grad`` from
``evaluate_problem(x, multipliers, penalty)`` with central differences of
``total`` along unit directions and checks that the error falls at least as
fast as the step (ratio test). It is a check to run before a solve, not part
of one.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .core import _as_float_list

_DEFAULT_TAYLOR_EPSILONS = tuple(float(0.5**power) for power in range(7, 13))

def _normalized_taylor_directions(
    x: np.ndarray,
    *,
    direction,
    seed: int,
    direction_count: int,
) -> Tuple[np.ndarray, ...]:
    if direction is None:
        if int(direction_count) < 1:
            raise ValueError("direction_count must be positive")
        rng = np.random.RandomState(seed)
        direction_arrays = [
            rng.standard_normal(size=x.shape) for _ in range(int(direction_count))
        ]
    else:
        direction_arrays = [np.asarray(direction, dtype=float).copy()]

    unit_directions = []
    for direction_array in direction_arrays:
        if direction_array.shape != x.shape:
            raise ValueError("direction must have the same shape as x0")
        direction_norm = float(np.linalg.norm(direction_array))
        if direction_norm <= np.finfo(float).eps:
            raise ValueError("direction must be nonzero")
        unit_directions.append(direction_array / direction_norm)
    return tuple(unit_directions)

def _directional_taylor_result(
    evaluate_problem: Callable[[np.ndarray, np.ndarray, float], dict],
    x: np.ndarray,
    multiplier_array: np.ndarray,
    penalty: float,
    base_grad: np.ndarray,
    unit_direction: np.ndarray,
    taylor_epsilons: Sequence[float],
    ratio_threshold: float,
) -> Tuple[dict, bool]:
    directional_derivative = float(
        np.dot(base_grad.reshape(-1), unit_direction.reshape(-1))
    )
    error_floor = 1e-10 * max(1.0, abs(directional_derivative))
    errors = []
    central_estimates = []
    ratios = []
    passed = True
    previous_error = None
    for epsilon in taylor_epsilons:
        step = float(epsilon) * unit_direction
        plus_eval = evaluate_problem(x + step, multiplier_array, penalty)
        minus_eval = evaluate_problem(x - step, multiplier_array, penalty)
        central_estimate = (float(plus_eval["total"]) - float(minus_eval["total"])) / (
            2.0 * float(epsilon)
        )
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
    return (
        {
            "direction": unit_direction.tolist(),
            "directional_derivative": directional_derivative,
            "central_estimates": central_estimates,
            "errors": errors,
            "ratios": finite_ratios,
            "max_ratio": max(finite_ratios) if finite_ratios else None,
        },
        passed,
    )

def run_directional_taylor_test(
    evaluate_problem: Callable[[np.ndarray, np.ndarray, float], dict],
    x0,
    multipliers,
    penalty: float,
    *,
    direction=None,
    epsilons: Optional[Sequence[float]] = None,
    seed: int = 1,
    ratio_threshold: float = 0.35,
    direction_count: int = 4,
) -> dict:
    x = np.asarray(x0, dtype=float).copy()
    multiplier_array = np.asarray(multipliers, dtype=float).copy()
    unit_directions = _normalized_taylor_directions(
        x,
        direction=direction,
        seed=seed,
        direction_count=direction_count,
    )

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
    passed = True
    direction_results = []
    finite_ratios = []
    for unit_direction in unit_directions:
        direction_result, direction_passed = _directional_taylor_result(
            evaluate_problem,
            x,
            multiplier_array,
            float(penalty),
            base_grad,
            unit_direction,
            taylor_epsilons,
            float(ratio_threshold),
        )
        passed = passed and direction_passed
        finite_ratios.extend(direction_result["ratios"])
        direction_results.append(direction_result)
    first_result = direction_results[0]
    return {
        "passed": bool(passed),
        "seed": int(seed),
        "penalty": float(penalty),
        "direction": first_result["direction"],
        "directions": [result["direction"] for result in direction_results],
        "direction_count": len(direction_results),
        "directional_derivative": first_result["directional_derivative"],
        "base_total": base_total,
        "epsilons": _as_float_list(taylor_epsilons),
        "central_estimates": first_result["central_estimates"],
        "errors": first_result["errors"],
        "direction_results": direction_results,
        "ratios": finite_ratios,
        "max_ratio": max(finite_ratios) if finite_ratios else None,
        "ratio_threshold": float(ratio_threshold),
    }
