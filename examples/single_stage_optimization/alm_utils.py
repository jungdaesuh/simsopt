import copy
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize, nnls


ALM_SCHEMA_VERSION = "alm_normalized_constraints_v2"
_HISTORY_DIAGNOSTICS_SOURCE_KEY = "_constraint_history_diagnostics_source"


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
    history_max_entries: int | None = 512


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


@dataclass
class ALMRunState:
    x: np.ndarray
    total_inner_iterations: int
    trust_radius: float | None
    history: list[dict]
    history_truncated_count: int
    cap_binding_detected: bool
    cap_binding_indices: set[int]
    penalty_cap_reached: bool
    penalty_cap_requested: float | None


@dataclass(frozen=True)
class ALMFinalState:
    success: bool
    message: str
    termination_reason: str
    outer_iterations: int
    evaluation: dict
    multipliers: np.ndarray
    penalty: float
    inner_result: object
    restored_best_feasible: bool
    restored_best_feasible_reason: str | None
    max_feasibility_violation: float
    stationarity_norm: float
    raw_stationarity_norm: float
    kkt_stationarity_norm: float | None
    feasibility_tolerance: float
    stationarity_tolerance: float


@dataclass(frozen=True)
class ALMHistoryEntry:
    payload: dict


@dataclass(frozen=True)
class ALMPenaltyIncreaseResult:
    penalty: float
    penalty_cap_reached: bool
    penalty_cap_requested: float | None
    feasible_stall_count: int
    update_feasibility_tol: float
    update_stationarity_tol: float
    penalty_update_state: SimpleNamespace
    cap_hit: bool
    requested_penalty: float | None


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
    trust_radius: float | None
    current_max_feasibility_violation: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    effective_feasibility_tol: float
    inner_callback: Callable[[np.ndarray], None] | None
    constraint_names_tuple: tuple[str, ...]
    constraint_blocks_tuple: tuple[str, ...] | None


@dataclass(frozen=True)
class ALMInnerAttemptResult:
    optimizer_result: object
    evaluation: dict
    x: np.ndarray
    bounds: object
    attempts: int
    iterations: int
    trust_radius: float | None
    last_inner_options: dict | None
    last_inner_profile: str | None
    forced_infeasible_penalty_cycle: bool
    forced_infeasible_penalty_reason: str | None
    forced_inner_false_success: bool
    nonfinite_candidate_evaluation: bool
    nonfinite_candidate_fields: list[str] | None


@dataclass
class _ALMInnerAttemptEvaluator:
    request: ALMInnerAttemptRequest
    cached_x: np.ndarray | None = None
    cached_evaluation: dict | None = None

    def fun(self, inner_x):
        evaluation = _sanitize_nonfinite_inner_evaluation(
            self.request.evaluate_problem(
                inner_x,
                self.request.multipliers,
                self.request.penalty_argument,
            ),
            fallback_evaluation=self.request.current_eval,
        )
        self.cached_x = np.asarray(inner_x, dtype=float).copy()
        self.cached_evaluation = evaluation
        return float(evaluation["total"]), np.asarray(evaluation["grad"], dtype=float)

    def callback(self, inner_x):
        if self.request.inner_callback is not None:
            self.request.inner_callback(inner_x)
        inner_x_arr = np.asarray(inner_x, dtype=float)
        if self.cached_x is not None and np.array_equal(inner_x_arr, self.cached_x):
            evaluation = self.cached_evaluation
        else:
            evaluation = _sanitize_nonfinite_inner_evaluation(
                self.request.evaluate_problem(
                    inner_x_arr,
                    self.request.multipliers,
                    self.request.penalty_argument,
                ),
                fallback_evaluation=self.request.current_eval,
            )
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
            _callback_raw_stationarity_norm,
            _callback_kkt_stationarity_norm,
            callback_stationarity_norm,
            _callback_signal_mismatch_active,
        ) = _stationarity_metrics(
            evaluation,
            callback_routing_state,
            self.request.effective_feasibility_tol,
        )
        if (
            callback_max_feasibility_violation <= self.request.effective_feasibility_tol
            and callback_stationarity_norm <= self.request.update_stationarity_tol
        ):
            raise _EarlyStopInnerSolve(inner_x, evaluation)


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
_OWNED_EVALUATION_ARRAY_FIELDS = (
    "grad",
    "metric_grad",
    "base_grad",
    "constraint_values",
    "feasibility_values",
    "dual_update_values",
    "hard_signed_constraint_values",
    "hard_violation_values",
    "surrogate_signed_constraint_values",
    "hard_dual_update_values",
    "raw_constraint_values",
    "raw_solver_constraint_values",
    "raw_dual_update_values",
    "raw_hard_signed_constraint_values",
    "raw_hard_violation_values",
    "raw_surrogate_signed_constraint_values",
    "raw_hard_dual_update_values",
    "normalized_signed_constraint_values",
    "normalized_feasibility_values",
    "constraint_scales",
    "constraint_activity_tolerances",
    "positive_shift_values",
    "augmented_term_by_constraint",
)
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


def augmented_inequality_objective(
    base_value: float,
    base_grad,
    constraint_values,
    constraint_grads,
    multipliers,
    penalty,
):
    constraint_values = np.asarray(constraint_values, dtype=float)
    constraint_grad_list = _constraint_grad_list(constraint_grads)
    multipliers = np.asarray(multipliers, dtype=float)
    penalty_values = _penalty_values(penalty, constraint_values.size)
    positive_shift = np.maximum(0.0, multipliers + penalty_values * constraint_values)

    total_value = float(base_value)
    augmented_terms = _augmented_terms(positive_shift, multipliers, penalty_values)
    if constraint_values.size > 0:
        total_value += float(np.sum(augmented_terms))

    base_grad_array = np.asarray(base_grad, dtype=float)
    total_grad = base_grad_array.copy()
    if constraint_values.size > 0:
        total_grad += np.tensordot(
            positive_shift,
            np.stack(constraint_grad_list, axis=0),
            axes=(0, 0),
        )

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
        positive_shift_values=positive_shift,
        augmented_term_by_constraint=augmented_terms,
    )


def normalize_alm_constraints(
    signed_values,
    constraint_grads,
    feasibility_values,
    activity_tolerances,
    scales,
):
    payload = normalize_alm_constraint_signals(
        signed_values,
        feasibility_values,
        activity_tolerances,
        scales,
    )
    payload["normalized_constraint_grads"] = normalize_alm_constraint_grads(
        constraint_grads,
        scales,
    )
    return payload


def normalize_alm_constraint_signals(
    signed_values,
    feasibility_values,
    activity_tolerances,
    scales,
):
    scale_array = np.asarray(scales, dtype=float)
    if np.any(~np.isfinite(scale_array)) or np.any(scale_array <= 0.0):
        raise ValueError("ALM constraint scales must be finite and positive")
    signed_array = np.asarray(signed_values, dtype=float)
    feasibility_array = np.asarray(feasibility_values, dtype=float)
    activity_tolerance_array = np.asarray(activity_tolerances, dtype=float)
    if signed_array.shape != scale_array.shape:
        raise ValueError("signed_values shape must match scales")
    if feasibility_array.shape != scale_array.shape:
        raise ValueError("feasibility_values shape must match scales")
    if activity_tolerance_array.shape != scale_array.shape:
        raise ValueError("activity_tolerances shape must match scales")
    return {
        "normalized_signed_values": signed_array / scale_array,
        "normalized_feasibility_values": feasibility_array / scale_array,
        "normalized_activity_tolerances": activity_tolerance_array / scale_array,
    }


def normalize_alm_constraint_grads(constraint_grads, scales):
    scale_array = np.asarray(scales, dtype=float)
    if np.any(~np.isfinite(scale_array)) or np.any(scale_array <= 0.0):
        raise ValueError("ALM constraint scales must be finite and positive")
    if len(constraint_grads) != scale_array.size:
        raise ValueError("constraint_grads length must match scales")
    return [
        np.asarray(grad, dtype=float) / float(scale)
        for grad, scale in zip(constraint_grads, scale_array)
    ]


def _constraint_grad_list(constraint_grads) -> list[np.ndarray]:
    return [np.asarray(constraint_grad, dtype=float) for constraint_grad in constraint_grads]


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
    positive_shift_values=None,
    augmented_term_by_constraint=None,
):
    dual_update_array = np.asarray(dual_update_values, dtype=float)
    feasibility_array = np.asarray(feasibility_values, dtype=float)
    stationarity_norm = float(np.linalg.norm(np.asarray(total_grad, dtype=float)))
    max_feasibility_violation = _max_value(feasibility_array)
    result = {
        "total": float(total_value),
        "base_value": float(base_value),
        "base_grad": np.asarray(base_grad, dtype=float),
        "grad": np.asarray(total_grad, dtype=float),
        "constraint_values": np.asarray(constraint_values, dtype=float),
        "constraint_grads": [
            np.asarray(constraint_grad, dtype=float)
            for constraint_grad in constraint_grads
        ],
        "dual_update_values": dual_update_array,
        "feasibility_values": feasibility_array,
        "max_violation": max_feasibility_violation,
        "max_feasibility_violation": max_feasibility_violation,
        "stationarity_norm": stationarity_norm,
    }
    if positive_shift_values is not None:
        result["positive_shift_values"] = np.asarray(
            positive_shift_values,
            dtype=float,
        )
    if augmented_term_by_constraint is not None:
        result["augmented_term_by_constraint"] = np.asarray(
            augmented_term_by_constraint,
            dtype=float,
        )
    return result


def _augmented_terms(
    positive_shift: np.ndarray,
    multipliers: np.ndarray,
    penalty_values: np.ndarray,
) -> np.ndarray:
    return (
        0.5
        * (positive_shift - multipliers)
        * (positive_shift + multipliers)
        / penalty_values
    )


def _positive_shift_and_augmented_terms(
    evaluation: dict,
    multiplier_array: np.ndarray,
    penalty_values: np.ndarray,
    solver_constraint_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive_shift = evaluation.get("positive_shift_values")
    augmented_terms = evaluation.get("augmented_term_by_constraint")
    if positive_shift is not None and augmented_terms is not None:
        return (
            np.asarray(positive_shift, dtype=float),
            np.asarray(augmented_terms, dtype=float),
        )
    positive_shift = np.maximum(
        0.0,
        multiplier_array
        + penalty_values * np.asarray(solver_constraint_values, dtype=float),
    )
    return positive_shift, _augmented_terms(
        positive_shift,
        multiplier_array,
        penalty_values,
    )


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _as_float_list(values) -> list[float]:
    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _optional_array_to_float_list(values) -> list[float] | None:
    return None if values is None else _as_float_list(values)


def _max_value(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else 0.0


def _optional_float_list(evaluation: dict, key: str, fallback) -> list[float] | None:
    values = evaluation.get(key)
    if values is None:
        values = fallback
    if values is None:
        return None
    return _as_float_list(values)


def _optional_float_array(evaluation: dict, key: str, fallback) -> np.ndarray | None:
    values = evaluation.get(key)
    if values is None:
        values = fallback
    if values is None:
        return None
    return np.asarray(values, dtype=float).reshape(-1).copy()


def _explicit_raw_signed_constraint_values(evaluation: dict) -> np.ndarray | None:
    raw_constraint_values = _optional_float_array(evaluation, "raw_constraint_values", None)
    if raw_constraint_values is not None:
        return raw_constraint_values
    return _optional_float_array(
        evaluation,
        "raw_surrogate_signed_constraint_values",
        None,
    )


def _optional_string_list(evaluation: dict, key: str) -> list[str] | None:
    values = evaluation.get(key)
    if values is None:
        return None
    return [str(value) for value in values]


def alm_raw_dual_estimates(multipliers, evaluation: dict) -> list[float] | None:
    constraint_scales = evaluation.get("constraint_scales")
    if constraint_scales is None:
        return None
    scales = np.asarray(constraint_scales, dtype=float)
    multiplier_array = np.asarray(multipliers, dtype=float)
    if scales.shape != multiplier_array.shape:
        raise ValueError("constraint_scales shape must match multipliers")
    return _as_float_list(multiplier_array / scales)


def _raw_dual_estimates(multipliers, evaluation: dict) -> list[float] | None:
    return alm_raw_dual_estimates(multipliers, evaluation)


def _constraint_label_history_diagnostics(
    constraint_names: Sequence[str],
    constraint_blocks: list[str] | None,
    feasibility_values: np.ndarray,
    raw_hard_violation_values: list[float] | None,
    positive_shift: np.ndarray,
    augmented_terms: np.ndarray,
) -> dict:
    if constraint_blocks is None:
        return {
            "block_max_raw_hard_violation": None,
            "block_max_normalized_violation": None,
            "block_augmented_term": None,
            "block_positive_shift_norm": None,
            "blocking_constraint_name": None,
            "blocking_constraint_block": None,
        }

    block_names = list(dict.fromkeys(constraint_blocks))
    block_index_by_name = {
        block_name: index for index, block_name in enumerate(block_names)
    }
    block_indices = np.fromiter(
        (block_index_by_name[block] for block in constraint_blocks),
        dtype=int,
        count=len(constraint_blocks),
    )

    block_max_normalized_violation_array = np.zeros(len(block_names), dtype=float)
    np.maximum.at(
        block_max_normalized_violation_array,
        block_indices,
        np.asarray(feasibility_values, dtype=float),
    )
    block_augmented_term_array = np.zeros(len(block_names), dtype=float)
    np.add.at(
        block_augmented_term_array,
        block_indices,
        np.asarray(augmented_terms, dtype=float),
    )
    block_shift_square_sum_array = np.zeros(len(block_names), dtype=float)
    np.add.at(
        block_shift_square_sum_array,
        block_indices,
        np.asarray(positive_shift, dtype=float) ** 2,
    )
    raw_hard_violation_array = (
        None
        if raw_hard_violation_values is None
        else np.asarray(raw_hard_violation_values, dtype=float)
    )
    if raw_hard_violation_array is None:
        block_max_raw_hard_violation = {}
    else:
        raw_hard_violation_by_block = np.zeros(len(block_names), dtype=float)
        np.maximum.at(
            raw_hard_violation_by_block,
            block_indices,
            raw_hard_violation_array,
        )
        block_max_raw_hard_violation = {
            block: float(value)
            for block, value in zip(block_names, raw_hard_violation_by_block)
        }

    blocking_constraint_name = None
    blocking_constraint_block = None
    if feasibility_values.size > 0 and _max_value(feasibility_values) > 0.0:
        blocking_index = int(np.argmax(feasibility_values))
        blocking_constraint_name = str(constraint_names[blocking_index])
        blocking_constraint_block = constraint_blocks[blocking_index]

    return {
        "block_max_raw_hard_violation": block_max_raw_hard_violation,
        "block_max_normalized_violation": {
            block: float(value)
            for block, value in zip(block_names, block_max_normalized_violation_array)
        },
        "block_augmented_term": {
            block: float(value)
            for block, value in zip(block_names, block_augmented_term_array)
        },
        "block_positive_shift_norm": {
            block: float(np.sqrt(value))
            for block, value in zip(block_names, block_shift_square_sum_array)
        },
        "blocking_constraint_name": blocking_constraint_name,
        "blocking_constraint_block": blocking_constraint_block,
    }


def _base_objective_value(evaluation: dict) -> float | None:
    base_objective = evaluation.get("base_total")
    if base_objective is None:
        base_objective = evaluation.get("base_value")
    if base_objective is None:
        return None
    return float(base_objective)


def _objective_to_augmented_term_ratio(
    evaluation: dict,
    augmented_terms: np.ndarray,
) -> float | None:
    if augmented_terms.size == 0:
        return None
    augmented_term = float(np.sum(augmented_terms))
    if augmented_term == 0.0:
        return None
    base_objective = _base_objective_value(evaluation)
    if base_objective is None:
        return None
    return abs(base_objective) / abs(augmented_term)


def _surrogate_hard_sign_mismatch(
    surrogate_signed_values: np.ndarray,
    hard_signed_values: np.ndarray,
) -> list[bool]:
    surrogate_signs = np.sign(np.asarray(surrogate_signed_values, dtype=float))
    hard_signs = np.sign(np.asarray(hard_signed_values, dtype=float))
    return (surrogate_signs != hard_signs).tolist()


def _surrogate_kkt_stationarity_norm(
    evaluation: dict,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> float | None:
    metric_grad = np.asarray(
        evaluation.get("metric_grad", evaluation["grad"]),
        dtype=float,
    )
    surrogate_values = routing_state.signal_state.surrogate_signed_constraint_values
    return _kkt_stationarity_norm(
        metric_grad,
        evaluation.get("constraint_grads"),
        surrogate_values,
        np.maximum(surrogate_values, 0.0),
        _constraint_activity_tolerances(evaluation, surrogate_values),
        feasibility_gate,
    )


def _lbfgsb_projected_gradient_max_norm(
    gradient,
    x,
    bounds,
) -> float:
    projected_gradient = np.asarray(gradient, dtype=float).reshape(-1).copy()
    if bounds is not None:
        x_array = np.asarray(x, dtype=float).reshape(-1)
        for index, bound in enumerate(bounds):
            lower_bound, upper_bound = bound
            coordinate = float(x_array[index])
            gradient_value = float(projected_gradient[index])
            at_lower_bound = lower_bound is not None and coordinate <= float(lower_bound)
            at_upper_bound = upper_bound is not None and coordinate >= float(upper_bound)
            if (at_lower_bound and gradient_value > 0.0) or (
                at_upper_bound and gradient_value < 0.0
            ):
                projected_gradient[index] = 0.0
    return float(np.linalg.norm(projected_gradient, ord=np.inf))


def _multiplier_interpretation(evaluation: dict) -> str:
    gradient_kinds = evaluation.get("gradient_value_kinds")
    dual_update_kinds = evaluation.get("dual_update_value_kinds")
    if gradient_kinds is None or dual_update_kinds is None:
        return "differentiable_alm_multipliers"
    for gradient_kind, dual_update_kind in zip(gradient_kinds, dual_update_kinds):
        if str(gradient_kind) != str(dual_update_kind):
            return "search_multipliers"
    return "differentiable_alm_multipliers"


def _constraint_history_diagnostics_source(
    evaluation: dict,
    multipliers: np.ndarray,
    penalty,
    constraint_names: Sequence[str],
    solver_constraint_values: np.ndarray,
    feasibility_values: np.ndarray,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> dict:
    multiplier_array = np.asarray(multipliers, dtype=float).reshape(-1).copy()
    penalty_values = _penalty_values(penalty, multiplier_array.size)
    solver_constraint_array = (
        np.asarray(solver_constraint_values, dtype=float).reshape(-1).copy()
    )
    feasibility_array = np.asarray(feasibility_values, dtype=float).reshape(-1).copy()
    positive_shift, augmented_terms = _positive_shift_and_augmented_terms(
        evaluation,
        multiplier_array,
        penalty_values,
        solver_constraint_array,
    )
    raw_hard_violation_values = _optional_float_array(
        evaluation,
        "raw_hard_violation_values",
        None,
    )
    return {
        "constraint_names": [str(name) for name in constraint_names],
        "feasibility_values": feasibility_array,
        "raw_signed_constraint_values": _explicit_raw_signed_constraint_values(evaluation),
        "normalized_signed_constraint_values": _optional_float_array(
            evaluation,
            "normalized_signed_constraint_values",
            solver_constraint_array,
        ),
        "raw_hard_violation_values": raw_hard_violation_values,
        "normalized_feasibility_values": _optional_float_array(
            evaluation,
            "normalized_feasibility_values",
            feasibility_array,
        ),
        "constraint_scales": _optional_float_array(
            evaluation,
            "constraint_scales",
            None,
        ),
        "constraint_blocks": _optional_string_list(evaluation, "constraint_blocks"),
        "normalized_multipliers": multiplier_array,
        "raw_dual_estimates": _raw_dual_estimates(multiplier_array, evaluation),
        "penalty_values": penalty_values,
        "positive_shift_values": (
            np.asarray(positive_shift, dtype=float).reshape(-1).copy()
        ),
        "augmented_term_by_constraint": (
            np.asarray(augmented_terms, dtype=float).reshape(-1).copy()
        ),
        "active_pressure_by_constraint": (
            np.asarray(positive_shift, dtype=float).reshape(-1)
            * solver_constraint_array
        ),
        "surrogate_minus_hard_normalized_gap": (
            np.asarray(
                routing_state.signal_state.surrogate_signed_constraint_values,
                dtype=float,
            ).reshape(-1)
            - np.asarray(
                routing_state.signal_state.hard_signed_constraint_values,
                dtype=float,
            ).reshape(-1)
        ),
        "surrogate_hard_sign_mismatch_by_constraint": _surrogate_hard_sign_mismatch(
            routing_state.signal_state.surrogate_signed_constraint_values,
            routing_state.signal_state.hard_signed_constraint_values,
        ),
        "objective_to_augmented_term_ratio": _objective_to_augmented_term_ratio(
            evaluation,
            augmented_terms,
        ),
        "augmented_gradient_norm": float(
            np.linalg.norm(np.asarray(evaluation["grad"], dtype=float))
        ),
        "surrogate_kkt_stationarity_norm": _surrogate_kkt_stationarity_norm(
            evaluation,
            routing_state,
            feasibility_gate,
        ),
        "multiplier_interpretation": _multiplier_interpretation(evaluation),
    }


def _constraint_history_diagnostics_from_source(source: dict) -> dict:
    raw_hard_violation_values = _optional_array_to_float_list(
        source["raw_hard_violation_values"]
    )
    raw_hard_max_violation = (
        None
        if raw_hard_violation_values is None
        else _max_value(np.asarray(raw_hard_violation_values, dtype=float))
    )
    block_diagnostics = _constraint_label_history_diagnostics(
        source["constraint_names"],
        source["constraint_blocks"],
        source["feasibility_values"],
        raw_hard_violation_values,
        source["positive_shift_values"],
        source["augmented_term_by_constraint"],
    )
    return {
        "raw_signed_constraint_values": _optional_array_to_float_list(
            source["raw_signed_constraint_values"]
        ),
        "normalized_signed_constraint_values": _as_float_list(
            source["normalized_signed_constraint_values"]
        ),
        "raw_hard_violation_values": raw_hard_violation_values,
        "normalized_feasibility_values": _as_float_list(
            source["normalized_feasibility_values"]
        ),
        "constraint_scales": _optional_array_to_float_list(
            source["constraint_scales"]
        ),
        "constraint_blocks": source["constraint_blocks"],
        "normalized_multipliers": _as_float_list(source["normalized_multipliers"]),
        "raw_dual_estimates": source["raw_dual_estimates"],
        "penalty_values": _as_float_list(source["penalty_values"]),
        "positive_shift_values": _as_float_list(source["positive_shift_values"]),
        "augmented_term_by_constraint": _as_float_list(
            source["augmented_term_by_constraint"]
        ),
        "active_pressure_by_constraint": _as_float_list(
            source["active_pressure_by_constraint"]
        ),
        "surrogate_minus_hard_normalized_gap": _as_float_list(
            source["surrogate_minus_hard_normalized_gap"]
        ),
        "surrogate_hard_sign_mismatch_by_constraint": source[
            "surrogate_hard_sign_mismatch_by_constraint"
        ],
        "objective_to_augmented_term_ratio": source[
            "objective_to_augmented_term_ratio"
        ],
        "augmented_gradient_norm": source["augmented_gradient_norm"],
        "surrogate_kkt_stationarity_norm": source["surrogate_kkt_stationarity_norm"],
        "multiplier_interpretation": source["multiplier_interpretation"],
        "max_raw_hard_violation": raw_hard_max_violation,
        **block_diagnostics,
    }


def _constraint_history_diagnostics(
    evaluation: dict,
    multipliers: np.ndarray,
    penalty,
    constraint_names: Sequence[str],
    solver_constraint_values: np.ndarray,
    feasibility_values: np.ndarray,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> dict:
    return _constraint_history_diagnostics_from_source(
        _constraint_history_diagnostics_source(
            evaluation,
            multipliers,
            penalty,
            constraint_names,
            solver_constraint_values,
            feasibility_values,
            routing_state,
            feasibility_gate,
        )
    )


def _materialize_history_entry_diagnostics(entry: dict) -> dict:
    if _HISTORY_DIAGNOSTICS_SOURCE_KEY not in entry:
        return dict(entry)
    materialized = dict(entry)
    source = materialized.pop(_HISTORY_DIAGNOSTICS_SOURCE_KEY)
    materialized.update(_constraint_history_diagnostics_from_source(source))
    return materialized


def _alm_summary_diagnostics(
    *,
    evaluation: dict,
    multipliers: np.ndarray,
    penalty,
    constraint_names: Sequence[str],
    solver_constraint_values: np.ndarray,
    feasibility_values: np.ndarray,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> dict:
    multiplier_array = np.asarray(multipliers, dtype=float)
    penalty_values = _penalty_values(penalty, multiplier_array.size)
    positive_shift, augmented_terms = _positive_shift_and_augmented_terms(
        evaluation,
        multiplier_array,
        penalty_values,
        solver_constraint_values,
    )
    raw_hard_violation_values = _optional_float_list(
        evaluation,
        "raw_hard_violation_values",
        None,
    )
    return {
        "raw_hard_violation_values": raw_hard_violation_values,
        "augmented_gradient_norm": float(
            np.linalg.norm(np.asarray(evaluation["grad"], dtype=float))
        ),
        "surrogate_kkt_stationarity_norm": _surrogate_kkt_stationarity_norm(
            evaluation,
            routing_state,
            feasibility_gate,
        ),
        "multiplier_interpretation": _multiplier_interpretation(evaluation),
        **_constraint_label_history_diagnostics(
            constraint_names,
            _optional_string_list(evaluation, "constraint_blocks"),
            feasibility_values,
            raw_hard_violation_values,
            positive_shift,
            augmented_terms,
        ),
    }


def _alm_summary(
    *,
    termination_reason: str,
    evaluation: dict,
    multipliers: np.ndarray,
    penalty,
    constraint_names: Sequence[str],
    routing_state: ALMConstraintRoutingState,
    feasibility_values: np.ndarray,
    solver_constraint_values: np.ndarray,
    final_stationarity_norm: float,
    final_feasibility_tolerance: float,
    multiplier_cap_binding: bool,
    penalty_cap_reached: bool,
    history: list[dict],
    history_truncated_count: int,
) -> dict:
    diagnostics = _alm_summary_diagnostics(
        evaluation=evaluation,
        multipliers=multipliers,
        penalty=penalty,
        constraint_names=constraint_names,
        solver_constraint_values=solver_constraint_values,
        feasibility_values=feasibility_values,
        routing_state=routing_state,
        feasibility_gate=final_feasibility_tolerance,
    )
    raw_hard_violations = diagnostics["raw_hard_violation_values"]
    penalty_values = _penalty_values(penalty, len(constraint_names))
    return {
        "termination_reason": str(termination_reason),
        "max_normalized_violation": _max_value(feasibility_values),
        "max_raw_hard_violation_by_constraint": (
            None
            if raw_hard_violations is None
            else {
                str(name): float(value)
                for name, value in zip(constraint_names, raw_hard_violations)
            }
        ),
        "stationarity_norm": float(final_stationarity_norm),
        "augmented_gradient_norm": diagnostics["augmented_gradient_norm"],
        "surrogate_kkt_stationarity_norm": diagnostics[
            "surrogate_kkt_stationarity_norm"
        ],
        "penalty": _representative_penalty(penalty_values),
        "penalty_values": _as_float_list(penalty_values),
        "penalty_cap_reached": bool(penalty_cap_reached),
        "multiplier_cap_binding": bool(multiplier_cap_binding),
        "signal_mismatch_active": bool(routing_state.signal_mismatch_active),
        "blocking_constraint_name": diagnostics["blocking_constraint_name"],
        "blocking_constraint_block": diagnostics["blocking_constraint_block"],
        "block_max_normalized_violation": diagnostics[
            "block_max_normalized_violation"
        ],
        "block_max_raw_hard_violation": diagnostics["block_max_raw_hard_violation"],
        "latest_history_action": None if not history else history[-1].get("action"),
        "history_truncated_count": int(history_truncated_count),
        "inner_lbfgsb_projected_gradient_norm": (
            None
            if not history
            else history[-1].get("inner_lbfgsb_projected_gradient_norm")
        ),
        "multiplier_interpretation": diagnostics["multiplier_interpretation"],
    }


def alm_result_diagnostics_fields(alm_result) -> dict:
    return {
        "ALM_SUMMARY": getattr(alm_result, "alm_summary", None),
        "ALM_MULTIPLIER_INTERPRETATION": getattr(
            alm_result,
            "multiplier_interpretation",
            None,
        ),
        "ALM_FINAL_AUGMENTED_GRADIENT_NORM": getattr(
            alm_result,
            "final_augmented_gradient_norm",
            None,
        ),
        "ALM_FINAL_SURROGATE_KKT_STATIONARITY_NORM": getattr(
            alm_result,
            "final_surrogate_kkt_stationarity_norm",
            None,
        ),
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
        "ALM_FINAL_PENALTY_VALUES": getattr(alm_result, "penalty_values", None),
        "ALM_BLOCK_PENALTIES": getattr(alm_result, "block_penalties", None),
        "ALM_BLOCK_PENALTY_CAP_REACHED": getattr(
            alm_result,
            "block_penalty_cap_reached",
            None,
        ),
        "ALM_BLOCK_PENALTY_CAP_REQUESTED": getattr(
            alm_result,
            "block_penalty_cap_requested",
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

    sanitized = dict(fallback_evaluation)
    sanitized["total"] = _elevated_rejection_total(float(fallback_evaluation["total"]))
    for field in _OWNED_EVALUATION_ARRAY_FIELDS:
        if field in fallback_evaluation:
            sanitized[field] = np.asarray(fallback_evaluation[field], dtype=float).copy()
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


def _snapshot_history_entry(entry: dict) -> dict:
    return copy.deepcopy(_materialize_history_entry_diagnostics(entry))


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
    penalty_objective_ratio = (
        None
        if base_objective == 0.0
        else abs(penalty_objective) / abs(base_objective)
    )

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
        "conditioning_penalty_objective_ratio": (
            None if penalty_objective_ratio is None else float(penalty_objective_ratio)
        ),
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
    objective_tol = max(
        _INFEASIBLE_STALL_OBJECTIVE_ATOL,
        _INFEASIBLE_STALL_OBJECTIVE_RTOL
        * max(abs(float(current_total)), abs(float(final_total)), 1.0),
    )
    improved_objective = objective_drop > objective_tol

    stationarity_drop = float(current_stationarity_norm) - float(final_stationarity_norm)
    stationarity_tol = max(
        _INFEASIBLE_STALL_OBJECTIVE_ATOL,
        _INFEASIBLE_STALL_OBJECTIVE_RTOL
        * max(
            abs(float(current_stationarity_norm)),
            abs(float(final_stationarity_norm)),
            1.0,
        ),
    )
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
    normalized_trust_radius = _normalize_trust_radius(trust_radius)
    if normalized_trust_radius is None:
        return None
    # This is a lightweight trust-region proxy implemented with L-BFGS-B bounds:
    # each continuation centers a symmetric box around the current iterate.
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
    penalty,
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
    penalty,
) -> np.ndarray:
    dual_update_array = np.asarray(dual_update_values, dtype=float)
    penalty_values = _penalty_values(penalty, dual_update_array.size)
    return np.maximum(
        0.0,
        np.asarray(multipliers, dtype=float) + penalty_values * dual_update_array,
    )


def _project_nonnegative_multipliers_with_diagnostics(
    multipliers: np.ndarray,
    dual_update_values: np.ndarray,
    penalty,
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
        np.flatnonzero(cap_binding_mask).tolist(),
    )


def _penalty_values(penalty, size: int) -> np.ndarray:
    penalty_array = np.asarray(penalty, dtype=float)
    if penalty_array.shape == ():
        values = np.full(int(size), float(penalty_array), dtype=float)
    elif penalty_array.shape == (int(size),):
        values = penalty_array.astype(float, copy=False)
    else:
        raise ValueError("penalty shape must be scalar or match constraint count")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("ALM penalty values must be finite and positive")
    return values


def _representative_penalty(penalty) -> float:
    values = np.asarray(penalty, dtype=float)
    if values.shape == ():
        return float(values)
    return _max_value(values)


def _tolerance_schedule_penalty(penalty) -> float:
    values = np.asarray(penalty, dtype=float)
    if values.shape == ():
        return float(values)
    return float(np.min(values))


def _penalty_schedule_tolerance(tolerance: float, penalty) -> float:
    return max(float(tolerance), 1.0 / _tolerance_schedule_penalty(penalty))


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


def _normalized_taylor_directions(
    x: np.ndarray,
    *,
    direction,
    seed: int,
    direction_count: int,
) -> tuple[np.ndarray, ...]:
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
) -> tuple[dict, bool]:
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
    epsilons: Sequence[float] | None = None,
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
        "epsilons": [float(epsilon) for epsilon in taylor_epsilons],
        "central_estimates": first_result["central_estimates"],
        "errors": first_result["errors"],
        "direction_results": direction_results,
        "ratios": finite_ratios,
        "max_ratio": max(finite_ratios) if finite_ratios else None,
        "ratio_threshold": float(ratio_threshold),
    }


def _extract_constraint_state(evaluation: dict):
    solver_constraint_values = _as_float_array(evaluation["constraint_values"])
    missing_fields = [
        field
        for field in ("feasibility_values", "dual_update_values")
        if field not in evaluation
    ]
    if missing_fields:
        raise KeyError(
            "ALM evaluation missing required normalized signal fields: "
            + ", ".join(missing_fields)
        )
    feasibility_values = _as_float_array(evaluation["feasibility_values"])
    dual_update_values = _as_float_array(evaluation["dual_update_values"])
    if feasibility_values.shape != solver_constraint_values.shape:
        raise ValueError("feasibility_values shape must match constraint_values")
    if dual_update_values.shape != solver_constraint_values.shape:
        raise ValueError("dual_update_values shape must match constraint_values")
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
    stage2_signal_fields = (
        "hard_signed_constraint_values",
        "hard_violation_values",
        "surrogate_signed_constraint_values",
        "hard_dual_update_values",
    )
    explicit_stage2_signals = any(
        key in evaluation
        for key in stage2_signal_fields
    )
    (
        solver_constraint_values,
        feasibility_values,
        dual_update_values,
        _max_feasibility_violation,
    ) = _extract_constraint_state(evaluation)
    if explicit_stage2_signals:
        missing_fields = [
            field for field in stage2_signal_fields if field not in evaluation
        ]
        if missing_fields:
            raise KeyError(
                "ALM stage-2 signal evaluation missing required fields: "
                + ", ".join(missing_fields)
            )
        hard_signed_constraint_values = _as_float_array(
            evaluation["hard_signed_constraint_values"]
        )
        hard_violation_values = _as_float_array(evaluation["hard_violation_values"])
        surrogate_signed_constraint_values = _as_float_array(
            evaluation["surrogate_signed_constraint_values"]
        )
        preferred_dual_update_values = _as_float_array(
            evaluation["hard_dual_update_values"]
        )
    else:
        hard_signed_constraint_values = dual_update_values
        hard_violation_values = feasibility_values
        surrogate_signed_constraint_values = solver_constraint_values
        preferred_dual_update_values = dual_update_values
    expected_shape = solver_constraint_values.shape
    for field_name, values in (
        ("hard_signed_constraint_values", hard_signed_constraint_values),
        ("hard_violation_values", hard_violation_values),
        ("surrogate_signed_constraint_values", surrogate_signed_constraint_values),
        ("hard_dual_update_values", preferred_dual_update_values),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{field_name} shape must match constraint_values")
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
    penalty,
    constraint_values: np.ndarray,
) -> np.ndarray:
    constraint_array = np.asarray(constraint_values, dtype=float)
    return np.maximum(
        0.0,
        np.asarray(multipliers, dtype=float)
        + _penalty_values(penalty, constraint_array.size) * constraint_array,
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
    # Gate surrogate activity on hard-certified feasibility while using the
    # surrogate signed value for the activity band.
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
    for constraint_grad, constraint_value, _feasibility_value, activity_tolerance in zip(
        constraint_grads,
        constraint_values,
        feasibility_values,
        activity_tolerances,
    ):
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
            "stationarity_norm",
            np.linalg.norm(evaluation["grad"]),
        )
    )
    if routing_state.signal_mismatch_active:
        kkt_stationarity_norm = None
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
    return (
        raw_stationarity_norm,
        kkt_stationarity_norm,
        raw_stationarity_norm,
        bool(routing_state.signal_mismatch_active),
    )


def _build_constraint_metadata_tuples(
    constraint_names: Sequence[str],
    constraint_blocks: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
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
    active_penalty: float
    constraint_names_tuple: tuple[str, ...]
    constraint_blocks_tuple: tuple[str, ...] | None
    trust_radius: float | None
    update_feasibility_tol: float
    update_stationarity_tol: float


def _normalize_alm_run_inputs(
    x0,
    constraint_names: Sequence[str],
    settings: ALMSettings,
    initial_multipliers: np.ndarray | None,
    initial_penalty: float | None,
    constraint_blocks: Sequence[str] | None,
    snapshot_accepted_state_fn,
    restore_incumbent_state_fn,
) -> _ALMNormalizedRunInputs:
    if (snapshot_accepted_state_fn is None) != (restore_incumbent_state_fn is None):
        raise ValueError(
            "snapshot_accepted_state_fn and restore_incumbent_state_fn must be provided together"
        )
    if settings.penalty_max is not None and settings.penalty_max <= 0.0:
        raise ValueError("settings.penalty_max must be positive when provided")
    if settings.penalty_max is not None and settings.penalty_max < settings.penalty_init:
        raise ValueError(
            f"settings.penalty_max ({settings.penalty_max}) must be >= "
            f"settings.penalty_init ({settings.penalty_init})"
        )
    if settings.history_max_entries is not None and settings.history_max_entries <= 0:
        raise ValueError("settings.history_max_entries must be positive or None")
    constraint_names_tuple, constraint_blocks_tuple = _build_constraint_metadata_tuples(
        constraint_names, constraint_blocks
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
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("initial ALM penalty must be finite and positive")
    if settings.penalty_max is not None and penalty > float(settings.penalty_max):
        raise ValueError(
            f"initial ALM penalty ({penalty}) must be <= "
            f"settings.penalty_max ({settings.penalty_max})"
        )
    active_penalty = float(penalty)
    return _ALMNormalizedRunInputs(
        x=x,
        multipliers=multipliers,
        penalty=penalty,
        active_penalty=active_penalty,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        trust_radius=_normalize_trust_radius(settings.trust_radius_init),
        update_feasibility_tol=_penalty_schedule_tolerance(
            settings.feasibility_tol, active_penalty
        ),
        update_stationarity_tol=_penalty_schedule_tolerance(
            settings.stationarity_tol, active_penalty
        ),
    )


def _attach_alm_constraint_metadata(
    evaluation: dict,
    constraint_names_tuple: tuple[str, ...],
    constraint_blocks_tuple: tuple[str, ...] | None,
) -> dict:
    if constraint_blocks_tuple is None:
        return evaluation
    annotated = dict(evaluation)
    annotated["constraint_names"] = constraint_names_tuple
    annotated["constraint_blocks"] = constraint_blocks_tuple
    return annotated


def _build_alm_history_entry(
    *,
    outer_iteration: int,
    continuation_iteration: int,
    constraint_names: Sequence[str],
    penalty: float,
    penalty_argument,
    multipliers: np.ndarray,
    post_update_multipliers: np.ndarray,
    routing_state,
    feasibility_values: np.ndarray,
    solver_constraint_values: np.ndarray,
    max_feasibility_violation: float,
    stationarity_norm: float,
    raw_stationarity_norm: float,
    kkt_stationarity_norm: float | None,
    signal_mismatch_active: bool,
    update_feasibility_tol: float,
    effective_feasibility_tol: float,
    update_stationarity_tol: float,
    trust_radius: float | None,
    start_x: np.ndarray,
    inner_iterations: int,
    inner_success: bool,
    inner_message: str,
    inner_maxiter: int | None,
    inner_maxls: int | None,
    inner_maxfun: int | None,
    inner_profile: str | None,
    inner_lbfgsb_projected_gradient_norm: float | None,
    inner_attempts: int,
    accepted_move_norm: float,
    objective_delta: float,
    feasibility_delta: float,
    feasibility_delta_tolerance: float,
    stationarity_delta: float,
    meaningful_progress: bool,
    feasible_stall_count: int,
    infeasible_stall: bool,
    inner_false_success: bool,
    inner_stall_reason: str | None,
    active_violation_index: int | None,
    active_constraint_name: str | None,
    nonfinite_candidate_evaluation: bool,
    nonfinite_candidate_fields: list[str] | None,
) -> dict:
    signal_state = routing_state.signal_state
    return {
        "outer_iteration": int(outer_iteration),
        "continuation_iteration": int(continuation_iteration),
        "constraint_names": [str(name) for name in constraint_names],
        "inner_iterations": int(inner_iterations),
        "inner_success": bool(inner_success),
        "inner_message": str(inner_message),
        "penalty": float(penalty),
        "penalty_values": _as_float_list(
            _penalty_values(penalty_argument, len(constraint_names))
        ),
        "block_penalties": None,
        "max_violation": float(max_feasibility_violation),
        "stationarity_norm": float(stationarity_norm),
        "raw_stationarity_norm": float(raw_stationarity_norm),
        "kkt_stationarity_norm": kkt_stationarity_norm,
        "constraint_values": [float(value) for value in feasibility_values],
        "solver_constraint_values": [
            float(value) for value in solver_constraint_values
        ],
        "hard_signed_constraint_values": [
            float(value) for value in signal_state.hard_signed_constraint_values
        ],
        "hard_violation_values": [
            float(value) for value in signal_state.hard_violation_values
        ],
        "surrogate_signed_constraint_values": [
            float(value) for value in signal_state.surrogate_signed_constraint_values
        ],
        "hard_max_violation": float(routing_state.hard_max_violation),
        "surrogate_max_value": float(routing_state.surrogate_max_value),
        "hard_positive_shift_zero": bool(routing_state.hard_positive_shift_zero),
        "signal_mismatch_active": bool(signal_mismatch_active),
        "multipliers": [float(value) for value in multipliers],
        "post_update_multipliers": [float(value) for value in post_update_multipliers],
        "feasibility_tolerance": float(update_feasibility_tol),
        "effective_feasibility_tolerance": float(effective_feasibility_tol),
        "stationarity_tolerance": float(update_stationarity_tol),
        "trust_radius": trust_radius,
        "inner_maxiter": inner_maxiter,
        "inner_maxls": inner_maxls,
        "inner_maxfun": inner_maxfun,
        "inner_profile": inner_profile,
        "inner_lbfgsb_projected_gradient_norm": inner_lbfgsb_projected_gradient_norm,
        "inner_attempts": int(inner_attempts),
        "accepted_move_norm": float(accepted_move_norm),
        "accepted_move_norm_scaled": float(accepted_move_norm) / max(
            1.0, float(np.linalg.norm(start_x))
        ),
        "infeasible_stall_move_tolerance": float(_move_tolerance(start_x)),
        "objective_delta": float(objective_delta),
        "feasibility_delta": float(feasibility_delta),
        "feasibility_delta_tolerance": float(feasibility_delta_tolerance),
        "stationarity_delta": float(stationarity_delta),
        "meaningful_progress": bool(meaningful_progress),
        "feasible_stall_count": int(feasible_stall_count),
        "infeasible_stall": bool(infeasible_stall),
        "inner_false_success": bool(inner_false_success),
        "inner_stall_reason": inner_stall_reason,
        "active_violation_index": active_violation_index,
        "active_constraint_name": active_constraint_name,
        "nonfinite_candidate_evaluation": bool(nonfinite_candidate_evaluation),
        "nonfinite_candidate_fields": nonfinite_candidate_fields,
        "multiplier_cap_binding": False,
        "multiplier_cap_binding_indices": [],
    }


def _append_alm_history_entry(
    history: list[dict],
    entry: ALMHistoryEntry,
    history_max_entries: int | None,
    history_truncated_count: int,
) -> tuple[dict, int]:
    history.append(entry.payload)
    if history_max_entries is not None:
        excess_entries = len(history) - int(history_max_entries)
        if excess_entries > 0:
            history_truncated_count += excess_entries
            del history[:excess_entries]
    return history[-1], history_truncated_count


def _attach_alm_history_diagnostics(
    entry: dict,
    evaluation: dict,
    multipliers_state: np.ndarray,
    penalty_state,
    constraint_names: Sequence[str],
    solver_values: np.ndarray,
    feasibility_state: np.ndarray,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> None:
    entry[_HISTORY_DIAGNOSTICS_SOURCE_KEY] = _constraint_history_diagnostics_source(
        evaluation,
        multipliers_state,
        penalty_state,
        constraint_names,
        solver_values,
        feasibility_state,
        routing_state,
        feasibility_gate,
    )


def _emit_alm_history_snapshot(
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    history: list[dict],
    latest_entry: dict,
    multipliers: np.ndarray,
    penalty: float,
) -> None:
    if history_callback is None:
        return
    # Callback contract: history is borrowed/read-only ALM state. The latest
    # entry and multipliers are owned snapshots for checkpoint writers that
    # persist the current iteration.
    history_callback(
        history,
        _snapshot_history_entry(latest_entry),
        multipliers.copy(),
        float(penalty),
    )


def _evaluate_alm_penalty_state(
    *,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    x: np.ndarray,
    multipliers: np.ndarray,
    penalty: float,
    feasibility_gate: float,
    constraint_names_tuple: tuple[str, ...],
    constraint_blocks_tuple: tuple[str, ...] | None,
) -> SimpleNamespace:
    penalty_argument = float(penalty)
    evaluation = _attach_alm_constraint_metadata(
        evaluate_problem(x, multipliers, penalty_argument),
        constraint_names_tuple,
        constraint_blocks_tuple,
    )
    _require_finite_evaluation(
        evaluation,
        context="ALM penalty update evaluation",
    )
    (
        solver_values,
        feasibility_state,
        _dual_update_state,
        max_violation,
    ) = _extract_constraint_state(evaluation)
    routing_state = _constraint_routing_state(
        evaluation,
        multipliers,
        penalty_argument,
        feasibility_gate,
    )
    (
        raw_stationarity,
        kkt_stationarity,
        stationarity,
        signal_mismatch,
    ) = _stationarity_metrics(evaluation, routing_state, feasibility_gate)
    return SimpleNamespace(
        penalty_argument=penalty_argument,
        evaluation=evaluation,
        solver_values=solver_values,
        feasibility_state=feasibility_state,
        max_violation=max_violation,
        routing_state=routing_state,
        raw_stationarity=raw_stationarity,
        kkt_stationarity=kkt_stationarity,
        stationarity=stationarity,
        signal_mismatch=signal_mismatch,
    )


def _refresh_alm_history_for_penalty_update(
    entry: dict,
    updated_state,
    *,
    feasibility_gate: float,
    penalty: float,
    constraint_names: Sequence[str],
    settings: ALMSettings,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    multipliers: np.ndarray,
) -> None:
    routing_state = updated_state.routing_state
    signal_state = routing_state.signal_state
    entry["penalty"] = float(penalty)
    entry["penalty_values"] = _as_float_list(
        _penalty_values(updated_state.penalty_argument, len(constraint_names))
    )
    entry["block_penalties"] = None
    entry["max_violation"] = float(updated_state.max_violation)
    entry["stationarity_norm"] = float(updated_state.stationarity)
    entry["raw_stationarity_norm"] = float(updated_state.raw_stationarity)
    entry["kkt_stationarity_norm"] = (
        None
        if updated_state.kkt_stationarity is None
        else float(updated_state.kkt_stationarity)
    )
    entry["constraint_values"] = [
        float(value) for value in updated_state.feasibility_state
    ]
    entry["solver_constraint_values"] = [
        float(value) for value in updated_state.solver_values
    ]
    entry["hard_signed_constraint_values"] = [
        float(value) for value in signal_state.hard_signed_constraint_values
    ]
    entry["hard_violation_values"] = [
        float(value) for value in signal_state.hard_violation_values
    ]
    entry["surrogate_signed_constraint_values"] = [
        float(value) for value in signal_state.surrogate_signed_constraint_values
    ]
    entry["hard_max_violation"] = float(routing_state.hard_max_violation)
    entry["surrogate_max_value"] = float(routing_state.surrogate_max_value)
    entry["hard_positive_shift_zero"] = bool(routing_state.hard_positive_shift_zero)
    entry["signal_mismatch_active"] = bool(updated_state.signal_mismatch)
    entry["feasibility_tolerance"] = float(update_feasibility_tol)
    entry["effective_feasibility_tolerance"] = float(
        _effective_feasibility_gate(settings, update_feasibility_tol)
    )
    entry["stationarity_tolerance"] = float(update_stationarity_tol)
    entry.update(_conditioning_metrics(updated_state.evaluation))
    _attach_alm_history_diagnostics(
        entry,
        updated_state.evaluation,
        multipliers,
        updated_state.penalty_argument,
        constraint_names,
        updated_state.solver_values,
        updated_state.feasibility_state,
        routing_state,
        feasibility_gate,
    )


def _apply_alm_penalty_increase(
    *,
    settings: ALMSettings,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    x: np.ndarray,
    multipliers: np.ndarray,
    penalty: float,
    history_entry: dict,
    trust_radius: float | None,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    constraint_names: Sequence[str],
    constraint_names_tuple: tuple[str, ...],
    constraint_blocks_tuple: tuple[str, ...] | None,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    history: list[dict],
) -> ALMPenaltyIncreaseResult:
    next_penalty, cap_hit, requested_penalty = _next_penalty(
        penalty,
        penalty_scale=settings.penalty_scale,
        penalty_max=settings.penalty_max,
    )
    if cap_hit:
        history_entry["action"] = "penalty_cap_reached"
        history_entry["trust_radius"] = trust_radius
    next_feasibility_tol = min(
        update_feasibility_tol,
        _penalty_schedule_tolerance(settings.feasibility_tol, next_penalty),
    )
    next_stationarity_tol = min(
        update_stationarity_tol,
        _penalty_schedule_tolerance(settings.stationarity_tol, next_penalty),
    )
    penalty_update_state = _evaluate_alm_penalty_state(
        evaluate_problem=evaluate_problem,
        x=x,
        multipliers=multipliers,
        penalty=next_penalty,
        feasibility_gate=next_feasibility_tol,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
    )
    _refresh_alm_history_for_penalty_update(
        history_entry,
        penalty_update_state,
        feasibility_gate=next_feasibility_tol,
        penalty=next_penalty,
        constraint_names=constraint_names,
        settings=settings,
        update_feasibility_tol=next_feasibility_tol,
        update_stationarity_tol=next_stationarity_tol,
        multipliers=multipliers,
    )
    if cap_hit:
        _emit_alm_history_snapshot(
            history_callback,
            history,
            history_entry,
            multipliers,
            next_penalty,
        )
    return ALMPenaltyIncreaseResult(
        penalty=next_penalty,
        penalty_cap_reached=bool(cap_hit),
        penalty_cap_requested=requested_penalty if cap_hit else None,
        feasible_stall_count=0,
        update_feasibility_tol=next_feasibility_tol,
        update_stationarity_tol=next_stationarity_tol,
        penalty_update_state=penalty_update_state,
        cap_hit=bool(cap_hit),
        requested_penalty=requested_penalty,
    )


def _build_alm_result(
    *,
    settings: ALMSettings,
    constraint_names: Sequence[str],
    run_state: ALMRunState,
    final_state: ALMFinalState,
):
    penalty_argument = float(final_state.penalty)
    (
        solver_constraint_values,
        feasibility_values,
        _dual_update_values,
        _max_feasibility_violation,
    ) = _extract_constraint_state(final_state.evaluation)
    routing_state = _constraint_routing_state(
        final_state.evaluation,
        final_state.multipliers,
        penalty_argument,
        final_state.feasibility_tolerance,
    )
    raw_constraint_values = _explicit_raw_signed_constraint_values(
        final_state.evaluation
    )
    raw_solver_constraint_values = _optional_float_array(
        final_state.evaluation,
        "raw_solver_constraint_values",
        raw_constraint_values,
    )
    normalized_signed_constraint_values = _optional_float_list(
        final_state.evaluation,
        "normalized_signed_constraint_values",
        solver_constraint_values,
    )
    normalized_feasibility_values = _optional_float_list(
        final_state.evaluation,
        "normalized_feasibility_values",
        feasibility_values,
    )
    raw_constraint_values_list = _optional_array_to_float_list(raw_constraint_values)
    raw_solver_constraint_values_list = _optional_array_to_float_list(
        raw_solver_constraint_values
    )
    inner_optimizer_success = None
    inner_optimizer_message = None
    if final_state.inner_result is not None:
        inner_optimizer_success = bool(
            getattr(final_state.inner_result, "success", False)
        )
        inner_optimizer_message = str(getattr(final_state.inner_result, "message", ""))
    for history_index, entry in enumerate(run_state.history):
        run_state.history[history_index] = _materialize_history_entry_diagnostics(entry)
    conditioning = _conditioning_metrics(final_state.evaluation)
    alm_summary = _alm_summary(
        termination_reason=final_state.termination_reason,
        evaluation=final_state.evaluation,
        multipliers=final_state.multipliers,
        penalty=penalty_argument,
        constraint_names=constraint_names,
        routing_state=routing_state,
        feasibility_values=feasibility_values,
        solver_constraint_values=solver_constraint_values,
        final_stationarity_norm=final_state.stationarity_norm,
        final_feasibility_tolerance=final_state.feasibility_tolerance,
        multiplier_cap_binding=run_state.cap_binding_detected,
        penalty_cap_reached=run_state.penalty_cap_reached,
        history=run_state.history,
        history_truncated_count=run_state.history_truncated_count,
    )
    return SimpleNamespace(
        alm_schema_version=ALM_SCHEMA_VERSION,
        alm_summary=alm_summary,
        x=run_state.x.copy(),
        success=bool(final_state.success),
        message=final_state.message,
        termination_reason=str(final_state.termination_reason),
        nit=run_state.total_inner_iterations,
        outer_iterations=int(final_state.outer_iterations),
        constraint_names=list(constraint_names),
        constraint_values=[float(value) for value in feasibility_values],
        solver_constraint_values=[float(value) for value in solver_constraint_values],
        normalized_constraint_values=normalized_signed_constraint_values,
        normalized_solver_constraint_values=_as_float_list(solver_constraint_values),
        normalized_signed_constraint_values=normalized_signed_constraint_values,
        normalized_feasibility_values=normalized_feasibility_values,
        raw_constraint_values=raw_constraint_values_list,
        raw_solver_constraint_values=raw_solver_constraint_values_list,
        hard_signed_constraint_values=[
            float(value)
            for value in routing_state.signal_state.hard_signed_constraint_values
        ],
        hard_violation_values=[
            float(value) for value in routing_state.signal_state.hard_violation_values
        ],
        surrogate_signed_constraint_values=[
            float(value)
            for value in routing_state.signal_state.surrogate_signed_constraint_values
        ],
        raw_hard_signed_constraint_values=_optional_float_list(
            final_state.evaluation,
            "raw_hard_signed_constraint_values",
            None,
        ),
        raw_hard_violation_values=_optional_float_list(
            final_state.evaluation,
            "raw_hard_violation_values",
            None,
        ),
        raw_surrogate_signed_constraint_values=_optional_float_list(
            final_state.evaluation,
            "raw_surrogate_signed_constraint_values",
            None,
        ),
        raw_dual_update_values=_optional_float_list(
            final_state.evaluation,
            "raw_dual_update_values",
            None,
        ),
        raw_hard_dual_update_values=_optional_float_list(
            final_state.evaluation,
            "raw_hard_dual_update_values",
            None,
        ),
        constraint_scales=_optional_float_list(
            final_state.evaluation,
            "constraint_scales",
            None,
        ),
        constraint_blocks=_optional_string_list(
            final_state.evaluation,
            "constraint_blocks",
        ),
        constraint_scale_sources=_optional_string_list(
            final_state.evaluation,
            "constraint_scale_sources",
        ),
        raw_dual_estimates=_raw_dual_estimates(
            final_state.multipliers,
            final_state.evaluation,
        ),
        multiplier_interpretation=_multiplier_interpretation(final_state.evaluation),
        multipliers=[float(value) for value in final_state.multipliers],
        penalty=_representative_penalty(penalty_argument),
        penalty_values=_as_float_list(
            _penalty_values(penalty_argument, len(constraint_names))
        ),
        block_penalties=None,
        block_penalty_cap_reached=None,
        block_penalty_cap_requested=None,
        trust_radius=run_state.trust_radius,
        history=run_state.history,
        inner_result=final_state.inner_result,
        optimizer_success=inner_optimizer_success,
        optimizer_message=inner_optimizer_message,
        converged_to_tolerances=bool(final_state.success),
        restored_best_feasible=bool(final_state.restored_best_feasible),
        restored_best_feasible_reason=final_state.restored_best_feasible_reason,
        final_max_feasibility_violation=float(final_state.max_feasibility_violation),
        final_stationarity_norm=float(final_state.stationarity_norm),
        final_raw_stationarity_norm=float(final_state.raw_stationarity_norm),
        final_kkt_stationarity_norm=(
            None
            if final_state.kkt_stationarity_norm is None
            else float(final_state.kkt_stationarity_norm)
        ),
        final_augmented_gradient_norm=alm_summary["augmented_gradient_norm"],
        final_surrogate_kkt_stationarity_norm=alm_summary[
            "surrogate_kkt_stationarity_norm"
        ],
        final_feasibility_tolerance=float(final_state.feasibility_tolerance),
        final_stationarity_tolerance=float(final_state.stationarity_tolerance),
        final_objective=float(final_state.evaluation["total"]),
        final_base_objective=conditioning["conditioning_base_objective"],
        final_penalty_objective=conditioning["conditioning_penalty_objective"],
        final_penalty_objective_ratio=conditioning[
            "conditioning_penalty_objective_ratio"
        ],
        final_total_grad_norm=conditioning["conditioning_total_grad_norm"],
        final_base_grad_norm=conditioning["conditioning_base_grad_norm"],
        final_penalty_grad_norm=conditioning["conditioning_penalty_grad_norm"],
        final_penalty_gradient_norm=conditioning["penalty_gradient_norm"],
        final_penalty_grad_ratio=conditioning["conditioning_penalty_grad_ratio"],
        final_hard_max_violation=float(routing_state.hard_max_violation),
        final_surrogate_max_value=float(routing_state.surrogate_max_value),
        hard_positive_shift_zero=bool(routing_state.hard_positive_shift_zero),
        signal_mismatch_active=bool(routing_state.signal_mismatch_active),
        multiplier_cap_binding=bool(run_state.cap_binding_detected),
        multiplier_cap_binding_indices=sorted(run_state.cap_binding_indices),
        penalty_cap_reached=bool(run_state.penalty_cap_reached),
        penalty_max=None if settings.penalty_max is None else float(settings.penalty_max),
        penalty_cap_requested=(
            None
            if run_state.penalty_cap_requested is None
            else float(run_state.penalty_cap_requested)
        ),
    )


def _restore_alm_best_feasible_on_failure(
    *,
    current_x: np.ndarray,
    best_feasible: ALMFeasibleIncumbent | None,
    settings: ALMSettings,
    restore_incumbent_state_fn: Callable[[object], None] | None,
    evaluation: dict,
    multipliers_state: np.ndarray,
    penalty_state: float,
    inner_result,
    final_max_feasibility_violation: float,
) -> SimpleNamespace:
    restored_best_feasible = False
    restored_best_feasible_reason = None
    restored_evaluation = evaluation
    restored_x = current_x.copy()
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


def _build_alm_failure_result_with_optional_restore(
    *,
    settings: ALMSettings,
    constraint_names: Sequence[str],
    run_state: ALMRunState,
    last_outer_iteration: int,
    best_feasible: ALMFeasibleIncumbent | None,
    restore_incumbent_state_fn: Callable[[object], None] | None,
    termination_reason: str,
    message_prefix: str,
    evaluation: dict,
    multipliers_state: np.ndarray,
    penalty_state: float,
    inner_result,
    final_max_feasibility_violation: float,
    restored_message_prefix: str | None = None,
    restored_termination_reason: str | None = None,
) -> SimpleNamespace:
    restored_state = _restore_alm_best_feasible_on_failure(
        current_x=run_state.x,
        best_feasible=best_feasible,
        settings=settings,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        evaluation=evaluation,
        multipliers_state=multipliers_state,
        penalty_state=penalty_state,
        inner_result=inner_result,
        final_max_feasibility_violation=final_max_feasibility_violation,
    )
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
    restored_run_state = ALMRunState(
        x=restored_state.x,
        total_inner_iterations=run_state.total_inner_iterations,
        trust_radius=run_state.trust_radius,
        history=run_state.history,
        history_truncated_count=run_state.history_truncated_count,
        cap_binding_detected=run_state.cap_binding_detected,
        cap_binding_indices=run_state.cap_binding_indices,
        penalty_cap_reached=run_state.penalty_cap_reached,
        penalty_cap_requested=run_state.penalty_cap_requested,
    )
    return _build_alm_result(
        settings=settings,
        constraint_names=constraint_names,
        run_state=restored_run_state,
        final_state=ALMFinalState(
            success=False,
            message=message,
            termination_reason=effective_termination_reason,
            outer_iterations=last_outer_iteration,
            evaluation=restored_state.evaluation,
            multipliers=restored_state.multipliers_state,
            penalty=restored_state.penalty_state,
            inner_result=restored_state.inner_result,
            restored_best_feasible=restored_state.restored_best_feasible,
            restored_best_feasible_reason=restored_state.restored_best_feasible_reason,
            max_feasibility_violation=restored_max_feasibility_violation,
            stationarity_norm=restored_stationarity_norm,
            raw_stationarity_norm=restored_raw_stationarity_norm,
            kkt_stationarity_norm=restored_kkt_stationarity_norm,
            feasibility_tolerance=settings.feasibility_tol,
            stationarity_tolerance=settings.stationarity_tol,
        ),
    )


@dataclass(frozen=True)
class _ALMDualUpdateResult:
    multipliers: np.ndarray
    update_feasibility_tol: float
    update_stationarity_tol: float
    multiplier_cap_binding: bool
    multiplier_cap_binding_indices: list[int]


def _handle_alm_dual_update_transition(
    *,
    multipliers: np.ndarray,
    routing_state,
    penalty_argument,
    settings: ALMSettings,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
) -> _ALMDualUpdateResult:
    new_multipliers, cap_binding, cap_binding_indices = (
        _project_nonnegative_multipliers_with_diagnostics(
            multipliers,
            routing_state.signal_state.preferred_dual_update_values,
            penalty_argument,
            settings.multiplier_max,
        )
    )
    penalty_tolerance_scale = float(settings.penalty_scale)
    return _ALMDualUpdateResult(
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
    constraint_names: Sequence[str],
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
        constraint_names=constraint_names,
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
            "ALM stopped at the penalty cap after restoring best "
            "feasible iterate"
        ),
        restored_termination_reason=(
            "penalty_cap_reached_restored_best_feasible"
        ),
        evaluation=penalty_transition.penalty_update_state.evaluation,
        multipliers_state=multipliers,
        penalty_state=penalty,
        inner_result=result,
        final_max_feasibility_violation=(
            penalty_transition.penalty_update_state.max_violation
        ),
    )


class _ALMContinuationDecision(str, Enum):
    RETURN = "return"
    BREAK_OUTER = "break_outer"
    CONTINUE_CONTINUATION = "continue_continuation"


class _ALMOuterDecision(str, Enum):
    RETURN = "return"
    NEXT_OUTER = "next_outer"
    EXHAUST = "exhaust"


@dataclass(frozen=True)
class _ALMContinuationStepResult:
    decision: _ALMContinuationDecision
    result: object | None
    multipliers: np.ndarray
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    feasible_stall_count: int
    last_result: object | None
    final_eval: dict | None
    final_multipliers: np.ndarray
    final_penalty: float
    best_feasible: ALMFeasibleIncumbent | None
    inner_options: dict | None


@dataclass(frozen=True)
class _ALMOuterIterationResult:
    decision: _ALMOuterDecision
    result: object | None
    multipliers: np.ndarray
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    last_result: object | None
    final_eval: dict | None
    final_multipliers: np.ndarray
    final_penalty: float
    best_feasible: ALMFeasibleIncumbent | None
    inner_options: dict | None


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
    nonfinite_candidate_fields: list[str] | None = None
    trust_radius = request.trust_radius

    for attempt_index in range(1, request.settings.max_inner_attempts + 1):
        attempts = attempt_index
        current_inner_profile = _select_inner_solve_profile(
            trust_radius=attempt_radius,
            continuation_iteration=request.continuation_iteration,
            feasible_enough=current_feasible_enough,
        )
        inner_attempt_options = _build_inner_options(
            request.inner_options,
            request.update_stationarity_tol,
            profile=current_inner_profile,
        )
        attempt_bounds = _build_box_bounds(request.x, attempt_radius)
        last_inner_options = dict(inner_attempt_options)
        last_inner_profile = current_inner_profile.name
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
            if evaluator.cached_x is not None and np.array_equal(
                candidate_x,
                evaluator.cached_x,
            ):
                candidate_eval = evaluator.cached_evaluation
            else:
                candidate_eval = _sanitize_nonfinite_inner_evaluation(
                    request.evaluate_problem(
                        candidate_x,
                        request.multipliers,
                        request.penalty_argument,
                    ),
                    fallback_evaluation=request.current_eval,
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
        moved_norm = float(np.linalg.norm(candidate_x - request.x))
        move_tolerance = _move_tolerance(request.x)
        if candidate_eval.get("nonfinite_evaluation"):
            nonfinite_candidate_evaluation = True
            nonfinite_candidate_fields = list(candidate_eval.get("nonfinite_fields", []))
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


@dataclass
class _ContinuationStepState:
    """Mutable carrier for the 11 sticky locals threaded through every
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
    last_result: object | None
    final_eval: dict | None
    final_multipliers: np.ndarray
    final_penalty: float
    best_feasible: ALMFeasibleIncumbent | None
    inner_options: dict | None


def _finalize_continuation_step(
    state: _ContinuationStepState,
    decision: _ALMContinuationDecision,
    result: object | None,
) -> _ALMContinuationStepResult:
    """Package the mutable continuation-step carrier into the frozen public
    result dataclass. Single source of truth for the 13-field shape."""
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
        inner_options=state.inner_options,
    )


def _build_alm_converged_step_result(
    *,
    settings: ALMSettings,
    constraint_names: Sequence[str],
    run_state: ALMRunState,
    outer_iteration: int,
    evaluation: dict,
    multipliers: np.ndarray,
    penalty: float,
    inner_result: object | None,
    max_feasibility_violation: float,
    stationarity_norm: float,
    raw_stationarity_norm: float,
    kkt_stationarity_norm: float,
    termination_reason: str,
    message: str,
) -> object:
    """Shared `_build_alm_result` constructor for the converged termination
    arms (pre-inner ``converged`` and ``constraints_inactive_converged`` and
    post-inner ``converged``)."""
    return _build_alm_result(
        settings=settings,
        constraint_names=constraint_names,
        run_state=run_state,
        final_state=ALMFinalState(
            success=True,
            message=message,
            termination_reason=termination_reason,
            outer_iterations=outer_iteration,
            evaluation=evaluation,
            multipliers=multipliers,
            penalty=penalty,
            inner_result=inner_result,
            restored_best_feasible=False,
            restored_best_feasible_reason=None,
            max_feasibility_violation=max_feasibility_violation,
            stationarity_norm=stationarity_norm,
            raw_stationarity_norm=raw_stationarity_norm,
            kkt_stationarity_norm=kkt_stationarity_norm,
            feasibility_tolerance=settings.feasibility_tol,
            stationarity_tolerance=settings.stationarity_tol,
        ),
    )


@dataclass(frozen=True)
class _PenaltyIncreaseOutcome:
    """Result of `_apply_continuation_penalty_increase`.

    ``cap_result`` is non-None when the penalty cap was hit, in which case the
    caller must return RETURN with that payload. Otherwise the caller appends
    its history-action string and returns BREAK_OUTER.
    """

    cap_result: object | None


def _apply_continuation_penalty_increase(
    *,
    state: _ContinuationStepState,
    settings: ALMSettings,
    run_state: ALMRunState,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    history_entry: dict,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    constraint_names: Sequence[str],
    constraint_names_tuple: tuple[str, ...],
    constraint_blocks_tuple: tuple[str, ...] | None,
    last_outer_iteration: int,
    restore_incumbent_state_fn: Callable[[object], None] | None,
    inner_result_for_cap: object | None,
) -> _PenaltyIncreaseOutcome:
    """Execute the shared penalty-increase + cap-handling sequence used at
    three sites inside `_run_alm_continuation_step` (infeasible-stall,
    signal-mismatch, feasible-update fallback).

    Mutates ``state`` (penalty, feasible_stall_count, tolerances, final_eval,
    final_multipliers, final_penalty) and ``run_state`` (penalty_cap_*). Does
    NOT touch ``history_entry["action"]`` — the caller annotates that string
    plus the optional ``outer_termination`` marker after a non-cap-hit.
    """
    penalty_transition = _apply_alm_penalty_increase(
        settings=settings,
        evaluate_problem=evaluate_problem,
        x=run_state.x,
        multipliers=state.multipliers,
        penalty=state.penalty,
        history_entry=history_entry,
        trust_radius=run_state.trust_radius,
        update_feasibility_tol=state.update_feasibility_tol,
        update_stationarity_tol=state.update_stationarity_tol,
        constraint_names=constraint_names,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        history_callback=history_callback,
        history=run_state.history,
    )
    state.penalty = penalty_transition.penalty
    run_state.penalty_cap_reached = penalty_transition.penalty_cap_reached
    run_state.penalty_cap_requested = penalty_transition.penalty_cap_requested
    state.feasible_stall_count = penalty_transition.feasible_stall_count
    state.update_feasibility_tol = penalty_transition.update_feasibility_tol
    state.update_stationarity_tol = penalty_transition.update_stationarity_tol
    state.final_eval = penalty_transition.penalty_update_state.evaluation
    state.final_multipliers = state.multipliers.copy()
    state.final_penalty = state.penalty
    if not penalty_transition.cap_hit:
        return _PenaltyIncreaseOutcome(cap_result=None)
    return _PenaltyIncreaseOutcome(
        cap_result=_handle_alm_penalty_cap_termination(
            settings=settings,
            constraint_names=constraint_names,
            run_state=run_state,
            last_outer_iteration=last_outer_iteration,
            best_feasible=state.best_feasible,
            restore_incumbent_state_fn=restore_incumbent_state_fn,
            penalty_transition=penalty_transition,
            multipliers=state.multipliers,
            penalty=state.penalty,
            result=inner_result_for_cap,
        ),
    )


def _annotate_break_outer_history(
    *,
    history_entry: dict,
    action: str,
    trust_radius: float | None,
    is_final_outer: bool,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    history: list[dict],
    multipliers: np.ndarray,
    penalty: float,
) -> None:
    """Common history-finalisation for the BREAK_OUTER arms that emit a
    snapshot after a single ``action`` annotation. The optional
    ``outer_termination`` marker is set when ``is_final_outer`` per guardrail
    #4 (max_outer annotation at every break-arm site).
    """
    history_entry["action"] = action
    history_entry["trust_radius"] = trust_radius
    if is_final_outer:
        history_entry["outer_termination"] = "max_outer"
    _emit_alm_history_snapshot(
        history_callback,
        history,
        history_entry,
        multipliers,
        penalty,
    )


def _grow_continuation_trust_radius(
    run_state: ALMRunState,
    settings: ALMSettings,
) -> None:
    """Grow the live trust radius in-place using the standard ALM growth rule.

    Identical at the signal-mismatch ``CONTINUE_CONTINUATION`` arm and the
    feasible-update ``CONTINUE_CONTINUATION`` arm; centralised so the trust
    radius schedule lives in one place.
    """
    if run_state.trust_radius is None:
        return
    run_state.trust_radius = max(
        run_state.trust_radius,
        max(
            settings.trust_radius_min,
            float(run_state.trust_radius) * float(settings.trust_radius_grow),
        ),
    )


def _emit_alm_subproblem_continue(
    *,
    state: _ContinuationStepState,
    run_state: ALMRunState,
    history_entry: dict,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
) -> _ALMContinuationStepResult:
    """Annotate ``subproblem_continue`` history fields, emit the snapshot,
    and finalize a CONTINUE_CONTINUATION step result.

    Shared by the signal-mismatch retry arm and the feasible-update retry
    arm; the two sites are bytewise-identical apart from prior trust-radius
    bookkeeping done by their callers.
    """
    history_entry["subproblem_limit_reason"] = None
    history_entry["action"] = "subproblem_continue"
    history_entry["trust_radius"] = run_state.trust_radius
    history_entry["feasible_stall_count"] = int(state.feasible_stall_count)
    _emit_alm_history_snapshot(
        history_callback,
        run_state.history,
        history_entry,
        state.multipliers,
        state.penalty,
    )
    return _finalize_continuation_step(
        state, _ALMContinuationDecision.CONTINUE_CONTINUATION, None
    )


def _emit_alm_stall_failure_step(
    *,
    state: _ContinuationStepState,
    settings: ALMSettings,
    constraint_names: Sequence[str],
    run_state: ALMRunState,
    last_outer_iteration: int,
    restore_incumbent_state_fn: Callable[[object], None] | None,
    history_entry: dict,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    action: str,
    termination_reason: str,
    message_prefix: str,
    inner_result: object | None,
    max_feasibility_violation: float,
) -> _ALMContinuationStepResult:
    """Emit the snapshot for ``constraints_inactive_stall`` /
    ``signal_mismatch_stall`` arms and finalize the RETURN step result built
    from `_build_alm_failure_result_with_optional_restore`.
    """
    history_entry["action"] = action
    history_entry["trust_radius"] = run_state.trust_radius
    _emit_alm_history_snapshot(
        history_callback,
        run_state.history,
        history_entry,
        state.multipliers,
        state.penalty,
    )
    return _finalize_continuation_step(
        state,
        _ALMContinuationDecision.RETURN,
        _build_alm_failure_result_with_optional_restore(
            settings=settings,
            constraint_names=constraint_names,
            run_state=run_state,
            last_outer_iteration=last_outer_iteration,
            best_feasible=state.best_feasible,
            restore_incumbent_state_fn=restore_incumbent_state_fn,
            termination_reason=termination_reason,
            message_prefix=message_prefix,
            evaluation=state.final_eval,
            multipliers_state=state.multipliers,
            penalty_state=state.penalty,
            inner_result=inner_result,
            final_max_feasibility_violation=max_feasibility_violation,
        ),
    )


def _emit_alm_converged_step(
    *,
    state: _ContinuationStepState,
    settings: ALMSettings,
    constraint_names: Sequence[str],
    run_state: ALMRunState,
    history_entry: dict,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    outer_iteration: int,
    inner_result: object | None,
    max_feasibility_violation: float,
    stationarity_norm: float,
    raw_stationarity_norm: float,
    kkt_stationarity_norm: float,
    termination_reason: str,
    message: str,
    action: str,
) -> _ALMContinuationStepResult:
    """Annotate the converged history action, emit the snapshot, and finalize
    the RETURN step result built from ``_build_alm_converged_step_result``.
    Shared by the post-inner converged arm and the constraints-inactive
    converged arm; both pass ``state.final_eval`` as the evaluation.
    """
    history_entry["action"] = action
    history_entry["trust_radius"] = run_state.trust_radius
    _emit_alm_history_snapshot(
        history_callback,
        run_state.history,
        history_entry,
        state.multipliers,
        state.penalty,
    )
    return _finalize_continuation_step(
        state,
        _ALMContinuationDecision.RETURN,
        _build_alm_converged_step_result(
            settings=settings,
            constraint_names=constraint_names,
            run_state=run_state,
            outer_iteration=outer_iteration,
            evaluation=state.final_eval,
            multipliers=state.multipliers,
            penalty=state.penalty,
            inner_result=inner_result,
            max_feasibility_violation=max_feasibility_violation,
            stationarity_norm=stationarity_norm,
            raw_stationarity_norm=raw_stationarity_norm,
            kkt_stationarity_norm=kkt_stationarity_norm,
            termination_reason=termination_reason,
            message=message,
        ),
    )


def _emit_alm_penalty_increase_arm(
    *,
    state: _ContinuationStepState,
    settings: ALMSettings,
    run_state: ALMRunState,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    history_entry: dict,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    constraint_names: Sequence[str],
    constraint_names_tuple: tuple[str, ...],
    constraint_blocks_tuple: tuple[str, ...] | None,
    last_outer_iteration: int,
    restore_incumbent_state_fn: Callable[[object], None] | None,
    inner_result: object | None,
    is_final_outer: bool,
    action: str,
) -> _ALMContinuationStepResult:
    """Run the shared "penalty increase + cap-handling + BREAK_OUTER" arm.

    On cap-hit, returns a RETURN step with the cap-termination payload. On
    no-cap-hit, annotates the BREAK_OUTER history (action + trust_radius +
    optional ``outer_termination`` for ``is_final_outer``) and emits the
    snapshot. Used by the infeasible-stall, signal-mismatch, and feasible
    fallback penalty-increase arms — three sites that previously each
    consumed ~78 lines of bytewise-identical scaffolding.
    """
    outcome = _apply_continuation_penalty_increase(
        state=state,
        settings=settings,
        run_state=run_state,
        evaluate_problem=evaluate_problem,
        history_entry=history_entry,
        history_callback=history_callback,
        constraint_names=constraint_names,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        last_outer_iteration=last_outer_iteration,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        inner_result_for_cap=inner_result,
    )
    if outcome.cap_result is not None:
        return _finalize_continuation_step(
            state, _ALMContinuationDecision.RETURN, outcome.cap_result
        )
    _annotate_break_outer_history(
        history_entry=history_entry,
        action=action,
        trust_radius=run_state.trust_radius,
        is_final_outer=is_final_outer,
        history_callback=history_callback,
        history=run_state.history,
        multipliers=state.multipliers,
        penalty=state.penalty,
    )
    return _finalize_continuation_step(
        state, _ALMContinuationDecision.BREAK_OUTER, None
    )


def _build_skipped_inner_history_entry(
    *,
    outer_iteration: int,
    continuation_iteration: int,
    constraint_names: Sequence[str],
    state: _ContinuationStepState,
    penalty_argument: float,
    routing_state: object,
    feasibility_values: np.ndarray,
    solver_constraint_values: np.ndarray,
    max_feasibility_violation: float,
    stationarity_norm: float,
    raw_stationarity_norm: float,
    kkt_stationarity_norm: float,
    signal_mismatch_active: bool,
    effective_feasibility_tol: float,
    trust_radius: float | None,
    start_x: np.ndarray,
) -> dict:
    """Build the ``converged`` history entry for the pre-inner-solve fast-path
    where the current iterate already satisfies the KKT stationarity gate.

    Centralises the long ``_build_alm_history_entry`` keyword listing —
    every "skipped inner solve" field defaults to its zero/None counterpart.
    """
    payload = _build_alm_history_entry(
        outer_iteration=outer_iteration,
        continuation_iteration=continuation_iteration,
        constraint_names=constraint_names,
        penalty=state.penalty,
        penalty_argument=penalty_argument,
        multipliers=state.multipliers,
        post_update_multipliers=state.multipliers,
        routing_state=routing_state,
        feasibility_values=feasibility_values,
        solver_constraint_values=solver_constraint_values,
        max_feasibility_violation=max_feasibility_violation,
        stationarity_norm=stationarity_norm,
        raw_stationarity_norm=raw_stationarity_norm,
        kkt_stationarity_norm=kkt_stationarity_norm,
        signal_mismatch_active=signal_mismatch_active,
        update_feasibility_tol=state.update_feasibility_tol,
        effective_feasibility_tol=effective_feasibility_tol,
        update_stationarity_tol=state.update_stationarity_tol,
        trust_radius=trust_radius,
        start_x=start_x,
        inner_iterations=0,
        inner_success=True,
        inner_message=(
            "ALM skipped inner solve; current iterate already satisfies the "
            "KKT stationarity gate."
        ),
        inner_maxiter=None,
        inner_maxls=None,
        inner_maxfun=None,
        inner_profile=None,
        inner_lbfgsb_projected_gradient_norm=None,
        inner_attempts=0,
        accepted_move_norm=0.0,
        objective_delta=0.0,
        feasibility_delta=0.0,
        feasibility_delta_tolerance=0.0,
        stationarity_delta=0.0,
        meaningful_progress=False,
        feasible_stall_count=0,
        infeasible_stall=False,
        inner_false_success=False,
        inner_stall_reason=None,
        active_violation_index=None,
        active_constraint_name=None,
        nonfinite_candidate_evaluation=False,
        nonfinite_candidate_fields=None,
    )
    payload["action"] = "converged"
    return payload


def _run_alm_continuation_step(
    *,
    settings: ALMSettings,
    run_state: ALMRunState,
    multipliers: np.ndarray,
    penalty: float,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    feasible_stall_count: int,
    last_result: object | None,
    final_eval: dict | None,
    final_multipliers: np.ndarray,
    final_penalty: float,
    best_feasible: ALMFeasibleIncumbent | None,
    inner_options: dict | None,
    outer_iteration: int,
    continuation_iteration: int,
    is_final_outer: bool,
    last_outer_iteration: int,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    inner_callback: Callable[[np.ndarray], None] | None,
    accepted_callback: Callable[[np.ndarray], None] | None,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    snapshot_accepted_state_fn: Callable[[], object] | None,
    restore_incumbent_state_fn: Callable[[object], None] | None,
    constraint_names: Sequence[str],
    constraint_names_tuple: tuple[str, ...],
    constraint_blocks_tuple: tuple[str, ...] | None,
) -> _ALMContinuationStepResult:
    state = _ContinuationStepState(
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
        inner_options=inner_options,
    )
    start_x = run_state.x.copy()
    penalty_argument = float(state.penalty)
    current_eval = _attach_alm_constraint_metadata(
        evaluate_problem(run_state.x, state.multipliers, penalty_argument),
        constraint_names_tuple,
        constraint_blocks_tuple,
    )
    _require_finite_evaluation(current_eval, context="ALM outer iterate evaluation")
    effective_feasibility_tol = _effective_feasibility_gate(
        settings, state.update_feasibility_tol
    )
    (
        current_solver_constraint_values,
        current_feasibility_values,
        _current_dual_update_values,
        current_max_feasibility_violation,
    ) = _extract_constraint_state(current_eval)
    current_routing_state = _constraint_routing_state(
        current_eval,
        state.multipliers,
        penalty_argument,
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
        history_entry, run_state.history_truncated_count = _append_alm_history_entry(
            run_state.history,
            ALMHistoryEntry(
                _build_skipped_inner_history_entry(
                    outer_iteration=outer_iteration,
                    continuation_iteration=continuation_iteration,
                    constraint_names=constraint_names,
                    state=state,
                    penalty_argument=penalty_argument,
                    routing_state=current_routing_state,
                    feasibility_values=current_feasibility_values,
                    solver_constraint_values=current_solver_constraint_values,
                    max_feasibility_violation=current_max_feasibility_violation,
                    stationarity_norm=current_stationarity_norm,
                    raw_stationarity_norm=current_raw_stationarity_norm,
                    kkt_stationarity_norm=current_kkt_stationarity_norm,
                    signal_mismatch_active=current_signal_mismatch_active,
                    effective_feasibility_tol=effective_feasibility_tol,
                    trust_radius=run_state.trust_radius,
                    start_x=start_x,
                )
            ),
            settings.history_max_entries,
            run_state.history_truncated_count,
        )
        history_entry.update(_conditioning_metrics(current_eval))
        _attach_alm_history_diagnostics(
            history_entry,
            current_eval,
            state.multipliers,
            penalty_argument,
            constraint_names,
            current_solver_constraint_values,
            current_feasibility_values,
            current_routing_state,
            effective_feasibility_tol,
        )
        _emit_alm_history_snapshot(
            history_callback,
            run_state.history,
            history_entry,
            state.multipliers,
            state.penalty,
        )
        state.final_eval = current_eval
        state.final_multipliers = state.multipliers.copy()
        state.final_penalty = state.penalty
        state.best_feasible = ALMFeasibleIncumbent(
            x=run_state.x.copy(),
            evaluation=current_eval,
            multipliers=state.multipliers.copy(),
            penalty=state.penalty,
            inner_result=state.last_result,
            incumbent_state=(
                None
                if snapshot_accepted_state_fn is None
                else snapshot_accepted_state_fn()
            ),
        )
        message = (
            "ALM converged: "
            f"max_violation={current_max_feasibility_violation:.3e}, "
            f"stationarity={current_stationarity_norm:.3e}"
        )
        return _finalize_continuation_step(
            state,
            _ALMContinuationDecision.RETURN,
            _build_alm_converged_step_result(
                settings=settings,
                constraint_names=constraint_names,
                run_state=run_state,
                outer_iteration=outer_iteration,
                evaluation=current_eval,
                multipliers=state.multipliers,
                penalty=state.penalty,
                inner_result=state.last_result,
                max_feasibility_violation=current_max_feasibility_violation,
                stationarity_norm=current_stationarity_norm,
                raw_stationarity_norm=current_raw_stationarity_norm,
                kkt_stationarity_norm=current_kkt_stationarity_norm,
                termination_reason="converged",
                message=message,
            ),
        )

    inner_attempt = _run_alm_inner_attempts(
        ALMInnerAttemptRequest(
            x=run_state.x,
            current_eval=current_eval,
            multipliers=state.multipliers,
            penalty_argument=penalty_argument,
            evaluate_problem=evaluate_problem,
            inner_options=state.inner_options,
            settings=settings,
            continuation_iteration=continuation_iteration,
            trust_radius=run_state.trust_radius,
            current_max_feasibility_violation=current_max_feasibility_violation,
            update_feasibility_tol=state.update_feasibility_tol,
            update_stationarity_tol=state.update_stationarity_tol,
            effective_feasibility_tol=effective_feasibility_tol,
            inner_callback=inner_callback,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
        )
    )

    result = inner_attempt.optimizer_result
    state.last_result = result
    run_state.total_inner_iterations += inner_attempt.iterations
    run_state.x = inner_attempt.x
    run_state.trust_radius = inner_attempt.trust_radius
    state.final_eval = inner_attempt.evaluation
    if inner_attempt.evaluation is not current_eval and accepted_callback is not None:
        accepted_callback(run_state.x.copy())
    accepted_move_norm = float(np.linalg.norm(run_state.x - start_x))
    (
        solver_constraint_values,
        feasibility_values,
        _dual_update_values,
        max_feasibility_violation,
    ) = _extract_constraint_state(state.final_eval)
    routing_state = _constraint_routing_state(
        state.final_eval,
        state.multipliers,
        penalty_argument,
        state.update_feasibility_tol,
    )
    (
        raw_stationarity_norm,
        kkt_stationarity_norm,
        stationarity_norm,
        signal_mismatch_active,
    ) = _stationarity_metrics(
        state.final_eval,
        routing_state,
        state.update_feasibility_tol,
    )
    feasibility_delta, feasibility_delta_tol = _feasibility_improvement_metrics(
        current_max_feasibility_violation,
        max_feasibility_violation,
    )
    hard_violation_values = routing_state.signal_state.hard_violation_values
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
    state.final_multipliers = state.multipliers.copy()
    state.final_penalty = state.penalty
    made_inner_progress = _made_meaningful_inner_progress(
        start_x,
        run_state.x,
        float(current_eval["total"]),
        float(state.final_eval["total"]),
        current_max_feasibility_violation,
        max_feasibility_violation,
        current_stationarity_norm,
        stationarity_norm,
    )

    if max_feasibility_violation <= settings.feasibility_tol:
        improves_best_feasible = (
            state.best_feasible is None
            or _incumbent_objective_value(state.final_eval)
            < _incumbent_objective_value(state.best_feasible.evaluation)
        )
        if improves_best_feasible:
            state.best_feasible = ALMFeasibleIncumbent(
                x=run_state.x.copy(),
                evaluation=state.final_eval,
                multipliers=state.multipliers.copy(),
                penalty=state.penalty,
                inner_result=result,
                incumbent_state=(
                    None
                    if snapshot_accepted_state_fn is None
                    else snapshot_accepted_state_fn()
                ),
            )

    state.inner_options = inner_attempt.last_inner_options
    history_entry = _build_alm_history_entry(
        outer_iteration=outer_iteration,
        continuation_iteration=continuation_iteration,
        constraint_names=constraint_names,
        penalty=state.penalty,
        penalty_argument=penalty_argument,
        multipliers=state.multipliers,
        post_update_multipliers=state.multipliers,
        routing_state=routing_state,
        feasibility_values=feasibility_values,
        solver_constraint_values=solver_constraint_values,
        max_feasibility_violation=max_feasibility_violation,
        stationarity_norm=stationarity_norm,
        raw_stationarity_norm=raw_stationarity_norm,
        kkt_stationarity_norm=kkt_stationarity_norm,
        signal_mismatch_active=signal_mismatch_active,
        update_feasibility_tol=state.update_feasibility_tol,
        effective_feasibility_tol=effective_feasibility_tol,
        update_stationarity_tol=state.update_stationarity_tol,
        trust_radius=run_state.trust_radius,
        start_x=start_x,
        inner_iterations=int(inner_attempt.iterations),
        inner_success=bool(getattr(result, "success", False)),
        inner_message=str(getattr(result, "message", "")),
        inner_maxiter=None
        if state.inner_options is None or "maxiter" not in state.inner_options
        else int(state.inner_options["maxiter"]),
        inner_maxls=None
        if state.inner_options is None or "maxls" not in state.inner_options
        else int(state.inner_options["maxls"]),
        inner_maxfun=None
        if state.inner_options is None or "maxfun" not in state.inner_options
        else int(state.inner_options["maxfun"]),
        inner_profile=inner_attempt.last_inner_profile,
        inner_lbfgsb_projected_gradient_norm=_lbfgsb_projected_gradient_max_norm(
            state.final_eval["grad"],
            run_state.x,
            inner_attempt.bounds,
        ),
        inner_attempts=int(inner_attempt.attempts),
        accepted_move_norm=float(accepted_move_norm),
        objective_delta=(
            float(current_eval["total"]) - float(state.final_eval["total"])
        ),
        feasibility_delta=float(feasibility_delta),
        feasibility_delta_tolerance=float(feasibility_delta_tol),
        stationarity_delta=float(current_stationarity_norm) - float(stationarity_norm),
        meaningful_progress=bool(made_inner_progress),
        feasible_stall_count=int(state.feasible_stall_count),
        infeasible_stall=bool(inner_attempt.forced_infeasible_penalty_cycle),
        inner_false_success=bool(inner_attempt.forced_inner_false_success),
        inner_stall_reason=inner_attempt.forced_infeasible_penalty_reason,
        active_violation_index=active_violation_index,
        active_constraint_name=active_constraint_name,
        nonfinite_candidate_evaluation=bool(
            inner_attempt.nonfinite_candidate_evaluation
        ),
        nonfinite_candidate_fields=inner_attempt.nonfinite_candidate_fields,
    )
    history_entry, run_state.history_truncated_count = _append_alm_history_entry(
        run_state.history,
        ALMHistoryEntry(history_entry),
        settings.history_max_entries,
        run_state.history_truncated_count,
    )
    history_entry.update(_conditioning_metrics(state.final_eval))
    _attach_alm_history_diagnostics(
        history_entry,
        state.final_eval,
        state.multipliers,
        penalty_argument,
        constraint_names,
        solver_constraint_values,
        feasibility_values,
        routing_state,
        state.update_feasibility_tol,
    )
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
        message = (
            "ALM converged: "
            f"max_violation={max_feasibility_violation:.3e}, "
            f"stationarity={stationarity_norm:.3e}"
        )
        return _emit_alm_converged_step(
            state=state,
            settings=settings,
            constraint_names=constraint_names,
            run_state=run_state,
            history_entry=history_entry,
            history_callback=history_callback,
            outer_iteration=outer_iteration,
            inner_result=result,
            max_feasibility_violation=max_feasibility_violation,
            stationarity_norm=stationarity_norm,
            raw_stationarity_norm=raw_stationarity_norm,
            kkt_stationarity_norm=kkt_stationarity_norm,
            termination_reason="converged",
            message=message,
            action="converged",
        )

    if inner_attempt.forced_infeasible_penalty_cycle:
        return _emit_alm_penalty_increase_arm(
            state=state,
            settings=settings,
            run_state=run_state,
            evaluate_problem=evaluate_problem,
            history_entry=history_entry,
            history_callback=history_callback,
            constraint_names=constraint_names,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
            last_outer_iteration=last_outer_iteration,
            restore_incumbent_state_fn=restore_incumbent_state_fn,
            inner_result=result,
            is_final_outer=is_final_outer,
            action="infeasible_stall_penalty_increase",
        )

    if constraints_inactive_candidate:
        if stationarity_norm <= settings.stationarity_tol:
            message = (
                "ALM converged with inactive hard constraints: "
                f"max_violation={routing_state.hard_max_violation:.3e}, "
                f"stationarity={stationarity_norm:.3e}"
            )
            return _emit_alm_converged_step(
                state=state,
                settings=settings,
                constraint_names=constraint_names,
                run_state=run_state,
                history_entry=history_entry,
                history_callback=history_callback,
                outer_iteration=outer_iteration,
                inner_result=result,
                max_feasibility_violation=max_feasibility_violation,
                stationarity_norm=stationarity_norm,
                raw_stationarity_norm=raw_stationarity_norm,
                kkt_stationarity_norm=kkt_stationarity_norm,
                termination_reason="constraints_inactive_converged",
                message=message,
                action="constraints_inactive_converged",
            )
        if not made_inner_progress and continuation_iteration > 0:
            return _emit_alm_stall_failure_step(
                state=state,
                settings=settings,
                constraint_names=constraint_names,
                run_state=run_state,
                last_outer_iteration=last_outer_iteration,
                restore_incumbent_state_fn=restore_incumbent_state_fn,
                history_entry=history_entry,
                history_callback=history_callback,
                action="constraints_inactive_stall",
                termination_reason="constraints_inactive_stall",
                message_prefix=(
                    "ALM stopped after hard constraints became inactive without "
                    "further stationarity progress"
                ),
                inner_result=result,
                max_feasibility_violation=max_feasibility_violation,
            )

    if signal_mismatch_active and hard_feasible_strict:
        if not made_inner_progress or continuation_iteration > 0:
            if routing_state.surrogate_positive_shift_zero:
                return _emit_alm_stall_failure_step(
                    state=state,
                    settings=settings,
                    constraint_names=constraint_names,
                    run_state=run_state,
                    last_outer_iteration=last_outer_iteration,
                    restore_incumbent_state_fn=restore_incumbent_state_fn,
                    history_entry=history_entry,
                    history_callback=history_callback,
                    action="signal_mismatch_stall",
                    termination_reason="signal_mismatch_stall",
                    message_prefix=(
                        "ALM stopped after hard-feasible and surrogate-active "
                        "signals repeated without corrective progress"
                    ),
                    inner_result=result,
                    max_feasibility_violation=max_feasibility_violation,
                )
            return _emit_alm_penalty_increase_arm(
                state=state,
                settings=settings,
                run_state=run_state,
                evaluate_problem=evaluate_problem,
                history_entry=history_entry,
                history_callback=history_callback,
                constraint_names=constraint_names,
                constraint_names_tuple=constraint_names_tuple,
                constraint_blocks_tuple=constraint_blocks_tuple,
                last_outer_iteration=last_outer_iteration,
                restore_incumbent_state_fn=restore_incumbent_state_fn,
                inner_result=result,
                is_final_outer=is_final_outer,
                action="signal_mismatch_penalty_increase",
            )
        state.feasible_stall_count = 0
        _grow_continuation_trust_radius(run_state, settings)
        return _emit_alm_subproblem_continue(
            state=state,
            run_state=run_state,
            history_entry=history_entry,
            history_callback=history_callback,
        )

    if hard_feasible_for_update and stationarity_norm <= state.update_stationarity_tol:
        state.feasible_stall_count = 0
        dual_update = _handle_alm_dual_update_transition(
            multipliers=state.multipliers,
            routing_state=routing_state,
            penalty_argument=penalty_argument,
            settings=settings,
            update_feasibility_tol=state.update_feasibility_tol,
            update_stationarity_tol=state.update_stationarity_tol,
        )
        state.multipliers = dual_update.multipliers
        state.update_feasibility_tol = dual_update.update_feasibility_tol
        state.update_stationarity_tol = dual_update.update_stationarity_tol
        history_entry["post_update_multipliers"] = [
            float(value) for value in state.multipliers
        ]
        history_entry["multiplier_cap_binding"] = dual_update.multiplier_cap_binding
        history_entry["multiplier_cap_binding_indices"] = list(
            dual_update.multiplier_cap_binding_indices
        )
        if dual_update.multiplier_cap_binding:
            run_state.cap_binding_detected = True
            run_state.cap_binding_indices.update(
                dual_update.multiplier_cap_binding_indices
            )
        _annotate_break_outer_history(
            history_entry=history_entry,
            action="dual_update",
            trust_radius=run_state.trust_radius,
            is_final_outer=is_final_outer,
            history_callback=history_callback,
            history=run_state.history,
            multipliers=state.multipliers,
            penalty=state.penalty,
        )
        return _finalize_continuation_step(
            state, _ALMContinuationDecision.BREAK_OUTER, None
        )

    if hard_feasible_for_update:
        state.feasible_stall_count = (
            0 if made_inner_progress else state.feasible_stall_count + 1
        )
        hit_stall_limit = (
            continuation_iteration == settings.max_subproblem_continuations
            or state.feasible_stall_count >= _PLATEAU_STALL_LIMIT
        )
        if hit_stall_limit:
            history_entry["subproblem_limit_reason"] = (
                "plateau_stall"
                if state.feasible_stall_count >= _PLATEAU_STALL_LIMIT
                else "max_subproblem_continuations"
            )
            history_entry["trust_radius"] = run_state.trust_radius
            history_entry["feasible_stall_count"] = int(state.feasible_stall_count)
            history_entry["action"] = "subproblem_limit"
            if is_final_outer:
                history_entry["outer_termination"] = "max_outer"
            _emit_alm_history_snapshot(
                history_callback,
                run_state.history,
                history_entry,
                state.multipliers,
                state.penalty,
            )
            if state.feasible_stall_count >= _PLATEAU_STALL_LIMIT:
                return _finalize_continuation_step(
                    state,
                    _ALMContinuationDecision.RETURN,
                    _build_alm_failure_result_with_optional_restore(
                        settings=settings,
                        constraint_names=constraint_names,
                        run_state=run_state,
                        last_outer_iteration=last_outer_iteration,
                        best_feasible=state.best_feasible,
                        restore_incumbent_state_fn=restore_incumbent_state_fn,
                        termination_reason="plateau_stall",
                        message_prefix=(
                            "ALM stopped after repeated feasible stationarity "
                            "plateau without meaningful progress"
                        ),
                        evaluation=state.final_eval,
                        multipliers_state=state.multipliers,
                        penalty_state=state.penalty,
                        inner_result=result,
                        final_max_feasibility_violation=max_feasibility_violation,
                    ),
                )
            return _finalize_continuation_step(
                state, _ALMContinuationDecision.BREAK_OUTER, None
            )
        _grow_continuation_trust_radius(run_state, settings)
        if not made_inner_progress:
            state.update_stationarity_tol = min(
                state.update_stationarity_tol,
                max(settings.stationarity_tol, 0.5 * stationarity_norm),
            )
        return _emit_alm_subproblem_continue(
            state=state,
            run_state=run_state,
            history_entry=history_entry,
            history_callback=history_callback,
        )

    return _emit_alm_penalty_increase_arm(
        state=state,
        settings=settings,
        run_state=run_state,
        evaluate_problem=evaluate_problem,
        history_entry=history_entry,
        history_callback=history_callback,
        constraint_names=constraint_names,
        constraint_names_tuple=constraint_names_tuple,
        constraint_blocks_tuple=constraint_blocks_tuple,
        last_outer_iteration=last_outer_iteration,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        inner_result=result,
        is_final_outer=is_final_outer,
        action="penalty_increase",
    )


def _finalize_outer_iteration(
    step: _ALMContinuationStepResult,
    decision: _ALMOuterDecision,
    result: object | None,
) -> _ALMOuterIterationResult:
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
        inner_options=step.inner_options,
    )


def _run_alm_outer_iteration(
    *,
    settings: ALMSettings,
    run_state: ALMRunState,
    multipliers: np.ndarray,
    penalty: float,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    last_result: object | None,
    final_eval: dict | None,
    final_multipliers: np.ndarray,
    final_penalty: float,
    best_feasible: ALMFeasibleIncumbent | None,
    inner_options: dict | None,
    outer_iteration: int,
    is_final_outer: bool,
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    inner_callback: Callable[[np.ndarray], None] | None,
    accepted_callback: Callable[[np.ndarray], None] | None,
    outer_state_callback: Callable[[int, np.ndarray, float], None] | None,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None,
    snapshot_accepted_state_fn: Callable[[], object] | None,
    restore_incumbent_state_fn: Callable[[object], None] | None,
    constraint_names: Sequence[str],
    constraint_names_tuple: tuple[str, ...],
    constraint_blocks_tuple: tuple[str, ...] | None,
) -> _ALMOuterIterationResult:
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
            inner_options=inner_options,
            outer_iteration=outer_iteration,
            continuation_iteration=continuation_iteration,
            is_final_outer=is_final_outer,
            last_outer_iteration=outer_iteration,
            evaluate_problem=evaluate_problem,
            inner_callback=inner_callback,
            accepted_callback=accepted_callback,
            history_callback=history_callback,
            snapshot_accepted_state_fn=snapshot_accepted_state_fn,
            restore_incumbent_state_fn=restore_incumbent_state_fn,
            constraint_names=constraint_names,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
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
        inner_options = step.inner_options
        if step.decision == _ALMContinuationDecision.RETURN:
            return _finalize_outer_iteration(
                step, _ALMOuterDecision.RETURN, step.result
            )
        if step.decision == _ALMContinuationDecision.BREAK_OUTER:
            break
    exhaust_decision = (
        _ALMOuterDecision.EXHAUST if is_final_outer else _ALMOuterDecision.NEXT_OUTER
    )
    return _ALMOuterIterationResult(
        decision=exhaust_decision,
        result=None,
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
    )


def _minimize_alm_impl(
    x0,
    constraint_names: Sequence[str],
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    settings: ALMSettings,
    inner_options: dict,
    inner_callback: Callable[[np.ndarray], None] | None = None,
    accepted_callback: Callable[[np.ndarray], None] | None = None,
    outer_state_callback: Callable[[int, np.ndarray, float], None] | None = None,
    snapshot_accepted_state_fn: Callable[[], object] | None = None,
    restore_incumbent_state_fn: Callable[[object], None] | None = None,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None = None,
    initial_multipliers: np.ndarray | None = None,
    initial_penalty: float | None = None,
    constraint_blocks: Sequence[str] | None = None,
):
    normalized = _normalize_alm_run_inputs(
        x0,
        constraint_names,
        settings,
        initial_multipliers,
        initial_penalty,
        constraint_blocks,
        snapshot_accepted_state_fn,
        restore_incumbent_state_fn,
    )
    multipliers = normalized.multipliers
    penalty = normalized.penalty
    constraint_names_tuple = normalized.constraint_names_tuple
    constraint_blocks_tuple = normalized.constraint_blocks_tuple
    update_feasibility_tol = normalized.update_feasibility_tol
    update_stationarity_tol = normalized.update_stationarity_tol
    run_state = ALMRunState(
        x=normalized.x,
        total_inner_iterations=0,
        trust_radius=normalized.trust_radius,
        history=[],
        history_truncated_count=0,
        cap_binding_detected=False,
        cap_binding_indices=set(),
        penalty_cap_reached=False,
        penalty_cap_requested=None,
    )
    final_eval: dict | None = None
    last_result: object | None = None
    final_multipliers = multipliers.copy()
    final_penalty = penalty
    last_outer_iteration = 0
    best_feasible: ALMFeasibleIncumbent | None = None
    inner_options_state: dict | None = inner_options

    for outer_iteration in range(1, settings.max_outer_iterations + 1):
        last_outer_iteration = outer_iteration
        is_final_outer = outer_iteration == settings.max_outer_iterations
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
            inner_options=inner_options_state,
            outer_iteration=outer_iteration,
            is_final_outer=is_final_outer,
            evaluate_problem=evaluate_problem,
            inner_callback=inner_callback,
            accepted_callback=accepted_callback,
            outer_state_callback=outer_state_callback,
            history_callback=history_callback,
            snapshot_accepted_state_fn=snapshot_accepted_state_fn,
            restore_incumbent_state_fn=restore_incumbent_state_fn,
            constraint_names=constraint_names,
            constraint_names_tuple=constraint_names_tuple,
            constraint_blocks_tuple=constraint_blocks_tuple,
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
        inner_options_state = outcome.inner_options
        if outcome.decision == _ALMOuterDecision.RETURN:
            return outcome.result
        if outcome.decision == _ALMOuterDecision.EXHAUST:
            break

    if final_eval is None or last_result is None:
        raise RuntimeError(
            "ALM failed before any inner optimization result was produced."
        )

    termination_reason = _termination_reason_from_history(
        run_state.history,
        success=False,
        restored_best_feasible=False,
    )
    return _build_alm_failure_result_with_optional_restore(
        settings=settings,
        constraint_names=constraint_names,
        run_state=run_state,
        last_outer_iteration=last_outer_iteration,
        best_feasible=best_feasible,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
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
        final_max_feasibility_violation=_extract_constraint_state(final_eval)[3],
    )


def minimize_alm(
    x0,
    constraint_names: Sequence[str],
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    settings: ALMSettings,
    inner_options: dict,
    inner_callback: Callable[[np.ndarray], None] | None = None,
    accepted_callback: Callable[[np.ndarray], None] | None = None,
    outer_state_callback: Callable[[int, np.ndarray, float], None] | None = None,
    snapshot_accepted_state_fn: Callable[[], object] | None = None,
    restore_incumbent_state_fn: Callable[[object], None] | None = None,
    history_callback: Callable[[list[dict], dict, np.ndarray, float], None] | None = None,
    initial_multipliers: np.ndarray | None = None,
    initial_penalty: float | None = None,
    constraint_blocks: Sequence[str] | None = None,
):
    return _minimize_alm_impl(
        x0=x0,
        constraint_names=constraint_names,
        evaluate_problem=evaluate_problem,
        settings=settings,
        inner_options=inner_options,
        inner_callback=inner_callback,
        accepted_callback=accepted_callback,
        outer_state_callback=outer_state_callback,
        snapshot_accepted_state_fn=snapshot_accepted_state_fn,
        restore_incumbent_state_fn=restore_incumbent_state_fn,
        history_callback=history_callback,
        initial_multipliers=initial_multipliers,
        initial_penalty=initial_penalty,
        constraint_blocks=constraint_blocks,
    )
