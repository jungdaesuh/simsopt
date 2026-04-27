import copy
from dataclasses import dataclass, field as dataclass_field
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np
from scipy.optimize import minimize, nnls


ALM_SCHEMA_VERSION = "alm_normalized_constraints_v2"
_MappingValue = TypeVar("_MappingValue")
_HISTORY_DIAGNOSTICS_SOURCE_KEY = "_constraint_history_diagnostics_source"


def _read_only_mapping(mapping: Mapping[str, _MappingValue]) -> Mapping[str, _MappingValue]:
    return MappingProxyType(dict(mapping))


def _cow_mapping_set(
    current: Mapping[str, _MappingValue],
    key: str,
    value: _MappingValue,
) -> Mapping[str, _MappingValue]:
    if current.get(key) == value:
        return current
    if isinstance(current, dict):
        updated = current
    else:
        updated = dict(current)
    updated[key] = value
    return updated


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
    block_penalties_enabled: bool = False
    block_penalty_init: Mapping[str, float] | None = None
    block_penalty_scale: Mapping[str, float] | None = None
    block_penalty_max: Mapping[str, float | None] | None = None
    block_penalty_improvement_fraction: float = 0.9
    block_penalty_patience: int = 1
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
    block_penalty_state: object | None = None


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


@dataclass(frozen=True)
class ALMBlockPenaltyState:
    constraint_blocks: tuple[str, ...]
    penalties_by_block: Mapping[str, float]
    scales_by_block: Mapping[str, float]
    max_by_block: Mapping[str, float | None]
    cap_reached_by_block: Mapping[str, bool]
    requested_by_block: Mapping[str, float | None]
    stall_counts_by_block: Mapping[str, int]
    previous_violations_by_block: Mapping[str, float]
    penalty_vector: np.ndarray = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self):
        penalties_by_block = _read_only_mapping(self.penalties_by_block)
        object.__setattr__(self, "penalties_by_block", penalties_by_block)
        object.__setattr__(
            self,
            "scales_by_block",
            _read_only_mapping(self.scales_by_block),
        )
        object.__setattr__(self, "max_by_block", _read_only_mapping(self.max_by_block))
        object.__setattr__(
            self,
            "cap_reached_by_block",
            _read_only_mapping(self.cap_reached_by_block),
        )
        object.__setattr__(
            self,
            "requested_by_block",
            _read_only_mapping(self.requested_by_block),
        )
        object.__setattr__(
            self,
            "stall_counts_by_block",
            _read_only_mapping(self.stall_counts_by_block),
        )
        object.__setattr__(
            self,
            "previous_violations_by_block",
            _read_only_mapping(self.previous_violations_by_block),
        )
        vector = np.asarray(
            [penalties_by_block[block] for block in self.constraint_blocks],
            dtype=float,
        )
        vector.setflags(write=False)
        object.__setattr__(self, "penalty_vector", vector)


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
_OWNED_EVALUATION_ARRAY_FIELDS = ("grad", "metric_grad", "base_grad")
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


def _raw_signed_constraint_values(evaluation: dict, fallback) -> np.ndarray | None:
    raw_constraint_values = _optional_float_array(evaluation, "raw_constraint_values", None)
    if raw_constraint_values is not None:
        return raw_constraint_values
    return _optional_float_array(
        evaluation,
        "raw_surrogate_signed_constraint_values",
        fallback,
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


def _constraint_block_history_diagnostics(
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
        routing_state.signal_state.hard_violation_values,
    )
    return {
        "constraint_names": [str(name) for name in constraint_names],
        "feasibility_values": feasibility_array,
        "raw_signed_constraint_values": _raw_signed_constraint_values(
            evaluation,
            solver_constraint_array,
        ),
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
    block_diagnostics = _constraint_block_history_diagnostics(
        source["constraint_names"],
        source["constraint_blocks"],
        source["feasibility_values"],
        raw_hard_violation_values,
        source["positive_shift_values"],
        source["augmented_term_by_constraint"],
    )
    return {
        "raw_signed_constraint_values": _as_float_list(
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
        routing_state.signal_state.hard_violation_values,
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
        **_constraint_block_history_diagnostics(
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


def _ordered_blocks(constraint_blocks: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for block in constraint_blocks:
        block_name = str(block)
        if not block_name:
            raise ValueError("ALM constraint block names must be non-empty")
        if block_name not in seen:
            ordered.append(block_name)
            seen.add(block_name)
    return tuple(ordered)


def _block_float_map(
    mapping: Mapping[str, float] | None,
    blocks: Sequence[str],
    default: float,
    *,
    name: str,
) -> dict[str, float]:
    block_names = _ordered_blocks(blocks)
    values = {block: float(default) for block in block_names}
    if mapping is not None:
        unknown_blocks = set(mapping) - set(block_names)
        if unknown_blocks:
            raise ValueError(
                f"{name} contains unknown ALM blocks: {sorted(unknown_blocks)}"
            )
        values.update({str(block): float(value) for block, value in mapping.items()})
    for block, value in values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name}[{block!r}] must be finite and positive")
    return values


def _block_optional_float_map(
    mapping: Mapping[str, float | None] | None,
    blocks: Sequence[str],
    default: float | None,
    *,
    name: str,
) -> dict[str, float | None]:
    block_names = _ordered_blocks(blocks)
    values = {
        block: (None if default is None else float(default))
        for block in block_names
    }
    if mapping is not None:
        unknown_blocks = set(mapping) - set(block_names)
        if unknown_blocks:
            raise ValueError(
                f"{name} contains unknown ALM blocks: {sorted(unknown_blocks)}"
            )
        values.update(
            {
                str(block): (None if value is None else float(value))
                for block, value in mapping.items()
            }
        )
    for block, value in values.items():
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name}[{block!r}] must be finite and positive or None")
    return values


def _validate_block_penalty_control_settings(settings: ALMSettings) -> None:
    improvement_fraction = float(settings.block_penalty_improvement_fraction)
    if not (0.0 < improvement_fraction < 1.0):
        raise ValueError("block_penalty_improvement_fraction must be between 0 and 1")
    if settings.block_penalty_patience <= 0:
        raise ValueError("block_penalty_patience must be positive")


def _block_penalty_growth_hits_cap(
    requested_penalty: float,
    penalty_max: float | None,
) -> bool:
    if penalty_max is None:
        return not np.isfinite(requested_penalty)
    return not np.isfinite(requested_penalty) or requested_penalty > float(penalty_max)


def _initial_block_penalty_state(
    settings: ALMSettings,
    constraint_blocks: Sequence[str],
    initial_penalty: float,
) -> ALMBlockPenaltyState:
    if len(constraint_blocks) == 0:
        raise ValueError("block penalties require at least one constraint block")
    _validate_block_penalty_control_settings(settings)
    block_names = _ordered_blocks(constraint_blocks)
    penalties_by_block = _block_float_map(
        settings.block_penalty_init,
        block_names,
        float(initial_penalty),
        name="block_penalty_init",
    )
    scales_by_block = _block_float_map(
        settings.block_penalty_scale,
        block_names,
        float(settings.penalty_scale),
        name="block_penalty_scale",
    )
    for block, scale in scales_by_block.items():
        if scale <= 1.0:
            raise ValueError(
                f"block_penalty_scale[{block!r}] must be greater than 1"
            )
    max_by_block = _block_optional_float_map(
        settings.block_penalty_max,
        block_names,
        settings.penalty_max,
        name="block_penalty_max",
    )
    for block, penalty_max in max_by_block.items():
        if penalty_max is not None and penalties_by_block[block] > penalty_max:
            raise ValueError(
                f"block_penalty_max[{block!r}] ({penalty_max}) must be >= "
                f"block_penalty_init[{block!r}] ({penalties_by_block[block]})"
            )
    return ALMBlockPenaltyState(
        constraint_blocks=tuple(str(block) for block in constraint_blocks),
        penalties_by_block=penalties_by_block,
        scales_by_block=scales_by_block,
        max_by_block=max_by_block,
        cap_reached_by_block={block: False for block in block_names},
        requested_by_block={block: None for block in block_names},
        stall_counts_by_block={block: 0 for block in block_names},
        previous_violations_by_block={},
    )


def _block_penalty_vector(state: ALMBlockPenaltyState) -> np.ndarray:
    return state.penalty_vector


def _block_penalty_summary(state: ALMBlockPenaltyState | None) -> dict[str, float] | None:
    if state is None:
        return None
    return {block: float(value) for block, value in state.penalties_by_block.items()}


def _block_penalty_cap_summary(
    state: ALMBlockPenaltyState | None,
) -> dict[str, bool] | None:
    if state is None:
        return None
    return {block: bool(value) for block, value in state.cap_reached_by_block.items()}


def _block_penalty_requested_summary(
    state: ALMBlockPenaltyState | None,
) -> dict[str, float | None] | None:
    if state is None:
        return None
    return {
        block: (None if value is None else float(value))
        for block, value in state.requested_by_block.items()
    }


def _block_max_violations(
    constraint_blocks: Sequence[str],
    feasibility_values: np.ndarray,
) -> dict[str, float]:
    block_values: dict[str, float] = {}
    for block, value in zip(constraint_blocks, np.asarray(feasibility_values, dtype=float)):
        block_name = str(block)
        block_values[block_name] = max(block_values.get(block_name, 0.0), float(value))
    return block_values


def _next_block_penalty_state(
    state: ALMBlockPenaltyState,
    block_violations: Mapping[str, float],
    settings: ALMSettings,
) -> tuple[ALMBlockPenaltyState, list[str], list[str], dict[str, float]]:
    _validate_block_penalty_control_settings(settings)
    improvement_fraction = float(settings.block_penalty_improvement_fraction)
    penalties_by_block = state.penalties_by_block
    cap_reached_by_block = state.cap_reached_by_block
    requested_by_block = state.requested_by_block
    stall_counts_by_block = state.stall_counts_by_block
    previous_violations_by_block = state.previous_violations_by_block
    grown_blocks: list[str] = []
    cap_hit_blocks: list[str] = []
    requested_growth_by_block: dict[str, float] = {}

    def _set_penalty(block: str, value: float) -> None:
        nonlocal penalties_by_block
        penalties_by_block = _cow_mapping_set(penalties_by_block, block, value)

    def _set_cap_reached(block: str, value: bool) -> None:
        nonlocal cap_reached_by_block
        cap_reached_by_block = _cow_mapping_set(cap_reached_by_block, block, value)

    def _set_requested(block: str, value: float | None) -> None:
        nonlocal requested_by_block
        requested_by_block = _cow_mapping_set(requested_by_block, block, value)

    def _set_stall_count(block: str, value: int) -> None:
        nonlocal stall_counts_by_block
        stall_counts_by_block = _cow_mapping_set(stall_counts_by_block, block, value)

    def _set_previous_violation(block: str, value: float) -> None:
        nonlocal previous_violations_by_block
        previous_violations_by_block = _cow_mapping_set(
            previous_violations_by_block,
            block,
            value,
        )

    for block in state.penalties_by_block:
        violation = float(block_violations.get(block, 0.0))
        previous_violation = previous_violations_by_block.get(block)
        if violation <= float(settings.feasibility_tol):
            _set_stall_count(block, 0)
            _set_previous_violation(block, violation)
            continue

        failed_to_improve = (
            previous_violation is None
            or violation
            > max(
                float(settings.feasibility_tol),
                float(previous_violation) * improvement_fraction,
            )
        )
        if failed_to_improve:
            _set_stall_count(block, stall_counts_by_block.get(block, 0) + 1)
        else:
            _set_stall_count(block, 0)

        _set_previous_violation(block, violation)
        if stall_counts_by_block[block] < int(settings.block_penalty_patience):
            continue

        requested_penalty = (
            float(penalties_by_block[block]) * float(state.scales_by_block[block])
        )
        penalty_max = state.max_by_block[block]
        requested_growth_by_block[block] = requested_penalty
        _set_requested(block, requested_penalty)
        if _block_penalty_growth_hits_cap(requested_penalty, penalty_max):
            _set_cap_reached(block, True)
            cap_hit_blocks.append(block)
        else:
            _set_penalty(block, requested_penalty)
            grown_blocks.append(block)
        _set_stall_count(block, 0)

    if (
        penalties_by_block is state.penalties_by_block
        and cap_reached_by_block is state.cap_reached_by_block
        and requested_by_block is state.requested_by_block
        and stall_counts_by_block is state.stall_counts_by_block
        and previous_violations_by_block is state.previous_violations_by_block
    ):
        return state, grown_blocks, cap_hit_blocks, requested_growth_by_block

    return (
        ALMBlockPenaltyState(
            constraint_blocks=state.constraint_blocks,
            penalties_by_block=penalties_by_block,
            scales_by_block=state.scales_by_block,
            max_by_block=state.max_by_block,
            cap_reached_by_block=cap_reached_by_block,
            requested_by_block=requested_by_block,
            stall_counts_by_block=stall_counts_by_block,
            previous_violations_by_block=previous_violations_by_block,
        ),
        grown_blocks,
        cap_hit_blocks,
        requested_growth_by_block,
    )


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
    if settings.block_penalties_enabled:
        if constraint_blocks is None:
            raise ValueError("constraint_blocks must be provided when block penalties are enabled")
        if len(constraint_blocks) != len(constraint_names):
            raise ValueError("constraint_blocks length must match constraint_names")
    if settings.history_max_entries is not None and settings.history_max_entries <= 0:
        raise ValueError("settings.history_max_entries must be positive or None")
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
    block_penalty_state = (
        _initial_block_penalty_state(settings, constraint_blocks, penalty)
        if settings.block_penalties_enabled
        else None
    )
    active_penalty = (
        _block_penalty_vector(block_penalty_state)
        if block_penalty_state is not None
        else float(penalty)
    )
    penalty = _representative_penalty(active_penalty)
    total_inner_iterations = 0
    history = []
    final_eval = None
    last_result = None
    final_multipliers = multipliers.copy()
    final_penalty = penalty
    final_block_penalty_state = block_penalty_state
    last_outer_iteration = 0
    history_truncated_count = 0
    cap_binding_detected = False
    cap_binding_indices: set[int] = set()
    penalty_cap_reached = False
    penalty_cap_requested = None
    trust_radius = _normalize_trust_radius(settings.trust_radius_init)
    update_feasibility_tol = _penalty_schedule_tolerance(
        settings.feasibility_tol,
        active_penalty,
    )
    update_stationarity_tol = _penalty_schedule_tolerance(
        settings.stationarity_tol,
        active_penalty,
    )
    best_feasible: ALMFeasibleIncumbent | None = None

    def _current_penalty_argument():
        if block_penalty_state is None:
            return float(penalty)
        return _block_penalty_vector(block_penalty_state)

    def _penalty_argument_for_state(
        penalty_state: float,
        state: ALMBlockPenaltyState | None,
    ):
        if state is None:
            return float(penalty_state)
        return _block_penalty_vector(state)

    def _sync_scalar_penalty_from_blocks() -> None:
        nonlocal penalty
        penalty = _representative_penalty(_current_penalty_argument())

    def _evaluate_current_penalty_state(feasibility_gate: float) -> SimpleNamespace:
        penalty_argument = _current_penalty_argument()
        evaluation = evaluate_problem(x, multipliers, penalty_argument)
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

    def _refresh_history_for_penalty_update(
        entry: dict,
        updated_state,
        feasibility_gate: float,
    ) -> None:
        routing_state = updated_state.routing_state
        signal_state = routing_state.signal_state
        entry["penalty"] = float(penalty)
        entry["penalty_values"] = _as_float_list(
            _penalty_values(updated_state.penalty_argument, len(constraint_names))
        )
        entry["block_penalties"] = _block_penalty_summary(block_penalty_state)
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
        _attach_history_diagnostics(
            entry,
            updated_state.evaluation,
            multipliers,
            updated_state.penalty_argument,
            updated_state.solver_values,
            updated_state.feasibility_state,
            routing_state,
            feasibility_gate,
        )

    def _publish_current_penalty_state(
        entry: dict,
        feasibility_gate: float,
    ) -> SimpleNamespace:
        nonlocal final_eval, final_multipliers, final_penalty, final_block_penalty_state
        updated_state = _evaluate_current_penalty_state(feasibility_gate)
        _refresh_history_for_penalty_update(entry, updated_state, feasibility_gate)
        final_eval = updated_state.evaluation
        final_multipliers = multipliers.copy()
        final_penalty = penalty
        final_block_penalty_state = block_penalty_state
        return updated_state

    def _current_penalty_scale() -> float:
        if block_penalty_state is None:
            return float(settings.penalty_scale)
        return min(float(value) for value in block_penalty_state.scales_by_block.values())

    def _append_history_entry(entry: dict) -> dict:
        nonlocal history_truncated_count
        history.append(entry)
        if settings.history_max_entries is not None:
            excess_entries = len(history) - int(settings.history_max_entries)
            if excess_entries > 0:
                history_truncated_count += excess_entries
                del history[:excess_entries]
        return history[-1]

    def _attach_history_diagnostics(
        entry: dict,
        evaluation: dict,
        multipliers_state: np.ndarray,
        penalty_state,
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

    def _emit_history_snapshot(latest_entry: dict) -> None:
        if history_callback is None:
            return
        # Callback contract: history is borrowed/read-only ALM state. The
        # latest entry and multipliers are owned snapshots for checkpoint
        # writers that persist the current iteration.
        history_callback(
            history,
            _snapshot_history_entry(latest_entry),
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
        block_penalty_state_for_result: ALMBlockPenaltyState | None = None,
    ):
        penalty_argument = _penalty_argument_for_state(
            penalty_state,
            block_penalty_state_for_result,
        )
        (
            solver_constraint_values,
            feasibility_values,
            _dual_update_values,
            _max_feasibility_violation,
        ) = _extract_constraint_state(evaluation)
        routing_state = _constraint_routing_state(
            evaluation,
            multipliers_state,
            penalty_argument,
            final_feasibility_tolerance,
        )
        raw_constraint_values = _raw_signed_constraint_values(
            evaluation,
            solver_constraint_values,
        )
        raw_solver_constraint_values = _optional_float_array(
            evaluation,
            "raw_solver_constraint_values",
            raw_constraint_values,
        )
        normalized_signed_constraint_values = _optional_float_list(
            evaluation,
            "normalized_signed_constraint_values",
            solver_constraint_values,
        )
        normalized_feasibility_values = _optional_float_list(
            evaluation,
            "normalized_feasibility_values",
            feasibility_values,
        )
        raw_constraint_values_list = _optional_array_to_float_list(raw_constraint_values)
        raw_solver_constraint_values_list = _optional_array_to_float_list(
            raw_solver_constraint_values
        )
        inner_optimizer_success = None
        inner_optimizer_message = None
        if inner_result is not None:
            inner_optimizer_success = bool(getattr(inner_result, "success", False))
            inner_optimizer_message = str(getattr(inner_result, "message", ""))
        for history_index, entry in enumerate(history):
            history[history_index] = _materialize_history_entry_diagnostics(entry)
        conditioning = _conditioning_metrics(evaluation)
        alm_summary = _alm_summary(
            termination_reason=termination_reason,
            evaluation=evaluation,
            multipliers=multipliers_state,
            penalty=penalty_argument,
            constraint_names=constraint_names,
            routing_state=routing_state,
            feasibility_values=feasibility_values,
            solver_constraint_values=solver_constraint_values,
            final_stationarity_norm=final_stationarity_norm,
            final_feasibility_tolerance=final_feasibility_tolerance,
            multiplier_cap_binding=cap_binding_detected,
            penalty_cap_reached=penalty_cap_reached,
            history=history,
            history_truncated_count=history_truncated_count,
        )
        return SimpleNamespace(
            alm_schema_version=ALM_SCHEMA_VERSION,
            alm_summary=alm_summary,
            x=x.copy(),
            success=bool(success),
            message=message,
            termination_reason=str(termination_reason),
            nit=total_inner_iterations,
            outer_iterations=int(outer_iterations),
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
                float(value)
                for value in routing_state.signal_state.hard_violation_values
            ],
            surrogate_signed_constraint_values=[
                float(value)
                for value in routing_state.signal_state.surrogate_signed_constraint_values
            ],
            raw_hard_signed_constraint_values=_optional_float_list(
                evaluation,
                "raw_hard_signed_constraint_values",
                routing_state.signal_state.hard_signed_constraint_values,
            ),
            raw_hard_violation_values=_optional_float_list(
                evaluation,
                "raw_hard_violation_values",
                routing_state.signal_state.hard_violation_values,
            ),
            raw_surrogate_signed_constraint_values=_optional_float_list(
                evaluation,
                "raw_surrogate_signed_constraint_values",
                routing_state.signal_state.surrogate_signed_constraint_values,
            ),
            raw_dual_update_values=_optional_float_list(
                evaluation,
                "raw_dual_update_values",
                solver_constraint_values,
            ),
            raw_hard_dual_update_values=_optional_float_list(
                evaluation,
                "raw_hard_dual_update_values",
                routing_state.signal_state.hard_signed_constraint_values,
            ),
            constraint_scales=_optional_float_list(evaluation, "constraint_scales", None),
            constraint_blocks=_optional_string_list(evaluation, "constraint_blocks"),
            constraint_scale_sources=_optional_string_list(
                evaluation,
                "constraint_scale_sources",
            ),
            raw_dual_estimates=_raw_dual_estimates(multipliers_state, evaluation),
            multiplier_interpretation=_multiplier_interpretation(evaluation),
            multipliers=[float(value) for value in multipliers_state],
            penalty=_representative_penalty(penalty_argument),
            penalty_values=_as_float_list(
                _penalty_values(penalty_argument, len(constraint_names))
            ),
            block_penalties=_block_penalty_summary(block_penalty_state_for_result),
            block_penalty_cap_reached=_block_penalty_cap_summary(
                block_penalty_state_for_result,
            ),
            block_penalty_cap_requested=_block_penalty_requested_summary(
                block_penalty_state_for_result,
            ),
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
            final_augmented_gradient_norm=alm_summary["augmented_gradient_norm"],
            final_surrogate_kkt_stationarity_norm=alm_summary[
                "surrogate_kkt_stationarity_norm"
            ],
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
        nonlocal penalty, penalty_cap_reached, penalty_cap_requested, block_penalty_state
        nonlocal feasible_stall_count, update_feasibility_tol, update_stationarity_tol
        if block_penalty_state is not None:
            block_violations = _block_max_violations(
                block_penalty_state.constraint_blocks,
                _extract_constraint_state(accepted_eval)[1],
            )
            (
                block_penalty_state,
                grown_blocks,
                cap_hit_blocks,
                requested_growth_by_block,
            ) = _next_block_penalty_state(
                block_penalty_state,
                block_violations,
                settings,
            )
            _sync_scalar_penalty_from_blocks()
            history_entry["block_penalty_growth_blocks"] = list(grown_blocks)
            history_entry["block_penalty_cap_reached"] = _block_penalty_cap_summary(
                block_penalty_state,
            )
            history_entry["block_penalty_cap_requested"] = _block_penalty_requested_summary(
                block_penalty_state,
            )
            history_entry["block_penalty_requested_growth"] = {
                block: float(value)
                for block, value in requested_growth_by_block.items()
            }
            penalty_argument = _current_penalty_argument()
            update_feasibility_tol = _penalty_schedule_tolerance(
                settings.feasibility_tol,
                penalty_argument,
            )
            update_stationarity_tol = _penalty_schedule_tolerance(
                settings.stationarity_tol,
                penalty_argument,
            )
            if cap_hit_blocks:
                penalty_update_state = _publish_current_penalty_state(
                    history_entry,
                    update_feasibility_tol,
                )
                penalty_cap_reached = True
                penalty_cap_requested = max(
                    requested_growth_by_block[block] for block in cap_hit_blocks
                )
                history_entry["action"] = "penalty_cap_reached"
                history_entry["trust_radius"] = trust_radius
                _emit_history_snapshot(history_entry)
                return _build_result(
                    success=False,
                    message=(
                        "ALM stopped after block penalty growth exceeded configured "
                        f"caps for blocks {cap_hit_blocks}."
                    ),
                    termination_reason="penalty_cap_reached",
                    outer_iterations=outer_iteration,
                    evaluation=penalty_update_state.evaluation,
                    multipliers_state=multipliers,
                    penalty_state=penalty,
                    inner_result=result,
                    restored_best_feasible=False,
                    restored_best_feasible_reason=None,
                    final_max_feasibility_violation=penalty_update_state.max_violation,
                    final_stationarity_norm=penalty_update_state.stationarity,
                    final_raw_stationarity_norm=penalty_update_state.raw_stationarity,
                    final_kkt_stationarity_norm=penalty_update_state.kkt_stationarity,
                    final_feasibility_tolerance=settings.feasibility_tol,
                    final_stationarity_tolerance=settings.stationarity_tol,
                    block_penalty_state_for_result=block_penalty_state,
                )
            feasible_stall_count = 0
            _publish_current_penalty_state(
                history_entry,
                update_feasibility_tol,
            )
            return None
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
        update_feasibility_tol = _penalty_schedule_tolerance(
            settings.feasibility_tol,
            penalty,
        )
        update_stationarity_tol = _penalty_schedule_tolerance(
            settings.stationarity_tol,
            penalty,
        )
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
        restored_block_penalty_state = final_block_penalty_state

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
            restored_block_penalty_state = best_feasible.block_penalty_state
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
            block_penalty_state=restored_block_penalty_state,
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
        x = restored_state.x
        restored_routing_state = _constraint_routing_state(
            restored_state.evaluation,
            restored_state.multipliers_state,
            _penalty_argument_for_state(
                restored_state.penalty_state,
                restored_state.block_penalty_state,
            ),
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
            block_penalty_state_for_result=restored_state.block_penalty_state,
        )

    for outer_iteration in range(1, settings.max_outer_iterations + 1):
        last_outer_iteration = outer_iteration
        if outer_state_callback is not None:
            outer_state_callback(outer_iteration, multipliers.copy(), penalty)

        feasible_stall_count = 0
        is_final_outer = outer_iteration == settings.max_outer_iterations
        for continuation_iteration in range(settings.max_subproblem_continuations + 1):
            start_x = x.copy()
            penalty_argument = _current_penalty_argument()
            current_eval = evaluate_problem(x, multipliers, penalty_argument)
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
                _append_history_entry(
                    {
                        "outer_iteration": int(outer_iteration),
                        "continuation_iteration": int(continuation_iteration),
                        "constraint_names": [str(name) for name in constraint_names],
                        "inner_iterations": 0,
                        "inner_success": True,
                        "inner_message": "ALM skipped inner solve; current iterate already satisfies the KKT stationarity gate.",
                        "penalty": float(penalty),
                        "penalty_values": _as_float_list(
                            _penalty_values(penalty_argument, len(constraint_names))
                        ),
                        "block_penalties": _block_penalty_summary(block_penalty_state),
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
                        "inner_lbfgsb_projected_gradient_norm": None,
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
                _attach_history_diagnostics(
                    history[-1],
                    current_eval,
                    multipliers,
                    penalty_argument,
                    current_solver_constraint_values,
                    current_feasibility_values,
                    current_routing_state,
                    effective_feasibility_tol,
                )
                _emit_history_snapshot(history[-1])
                final_eval = current_eval
                final_multipliers = multipliers.copy()
                final_penalty = penalty
                final_block_penalty_state = block_penalty_state
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
                    block_penalty_state_for_result=block_penalty_state,
                )

            _cached_eval = SimpleNamespace(x=None, evaluation=None)

            def inner_fun(inner_x):
                evaluation = _sanitize_nonfinite_inner_evaluation(
                    evaluate_problem(inner_x, multipliers, penalty_argument),
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
                        evaluate_problem(inner_x_arr, multipliers, penalty_argument),
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
                    penalty_argument,
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
            accepted_x = None
            accepted_bounds = None
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
                attempt_bounds = _build_box_bounds(x, attempt_radius)
                last_inner_options = dict(inner_attempt_options)
                last_inner_profile = current_inner_profile.name
                try:
                    result = minimize(
                        inner_fun,
                        x,
                        jac=True,
                        method="L-BFGS-B",
                        bounds=attempt_bounds,
                        callback=alm_inner_callback,
                        options=inner_attempt_options,
                    )
                    candidate_x = np.asarray(result.x, dtype=float).copy()
                    if _cached_eval.x is not None and np.array_equal(candidate_x, _cached_eval.x):
                        candidate_eval = _cached_eval.evaluation
                    else:
                        candidate_eval = _sanitize_nonfinite_inner_evaluation(
                            evaluate_problem(candidate_x, multipliers, penalty_argument),
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
                    accepted_x = candidate_x
                    accepted_bounds = attempt_bounds
                    if attempt_radius is not None:
                        if moved_norm >= 0.5 * float(attempt_radius):
                            trust_radius = float(attempt_radius) * float(settings.trust_radius_grow)
                        else:
                            trust_radius = float(attempt_radius)
                    break
                if infeasible_inner_stall:
                    accepted_result = result
                    accepted_eval = current_eval
                    accepted_x = start_x
                    accepted_bounds = attempt_bounds
                    forced_infeasible_penalty_cycle = True
                    forced_infeasible_penalty_reason = inner_stall_reason
                    forced_inner_false_success = bool(inner_false_success)
                    if attempt_radius is not None:
                        trust_radius = float(attempt_radius)
                    break
                if attempt_radius is None:
                    accepted_result = result
                    accepted_eval = current_eval
                    accepted_x = start_x
                    accepted_bounds = attempt_bounds
                    break
                next_radius = max(
                    settings.trust_radius_min,
                    float(attempt_radius) * float(settings.trust_radius_shrink),
                )
                exhausted_attempts = attempt_index == settings.max_inner_attempts
                if attempt_radius <= settings.trust_radius_min or exhausted_attempts:
                    accepted_result = result
                    accepted_eval = current_eval
                    accepted_x = start_x
                    accepted_bounds = attempt_bounds
                    trust_radius = float(attempt_radius)
                    break
                attempt_radius = float(next_radius)
                trust_radius = float(attempt_radius)
                continue

            if accepted_result is None or accepted_eval is None or accepted_x is None:
                if last_attempt_result is None:
                    raise RuntimeError("ALM failed before any inner optimization result was produced.")
                accepted_result = last_attempt_result
                accepted_eval = current_eval
                accepted_x = start_x
                accepted_bounds = None

            result = accepted_result
            last_result = result
            total_inner_iterations += attempt_iterations
            x = accepted_x
            final_eval = accepted_eval
            if accepted_eval is not current_eval and accepted_callback is not None:
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
                penalty_argument,
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
            final_block_penalty_state = block_penalty_state
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
                        block_penalty_state=block_penalty_state,
                    )

            history_entry = {
                "outer_iteration": int(outer_iteration),
                "continuation_iteration": int(continuation_iteration),
                "constraint_names": [str(name) for name in constraint_names],
                "inner_iterations": int(attempt_iterations),
                "inner_success": bool(getattr(result, "success", False)),
                "inner_message": str(getattr(result, "message", "")),
                "penalty": float(penalty),
                "penalty_values": _as_float_list(
                    _penalty_values(penalty_argument, len(constraint_names))
                ),
                "block_penalties": _block_penalty_summary(block_penalty_state),
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
                "inner_lbfgsb_projected_gradient_norm": _lbfgsb_projected_gradient_max_norm(
                    final_eval["grad"],
                    x,
                    accepted_bounds,
                ),
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
            history_entry = _append_history_entry(history_entry)
            history_entry.update(_conditioning_metrics(final_eval))
            _attach_history_diagnostics(
                history_entry,
                final_eval,
                multipliers,
                penalty_argument,
                solver_constraint_values,
                feasibility_values,
                routing_state,
                update_feasibility_tol,
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
                    block_penalty_state_for_result=block_penalty_state,
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
                        block_penalty_state_for_result=block_penalty_state,
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
                    penalty_argument,
                    settings.multiplier_max,
                )
                history_entry["post_update_multipliers"] = [float(value) for value in multipliers]
                history_entry["multiplier_cap_binding"] = bool(multiplier_cap_binding)
                history_entry["multiplier_cap_binding_indices"] = list(multiplier_cap_binding_indices)
                if multiplier_cap_binding:
                    cap_binding_detected = True
                    cap_binding_indices.update(multiplier_cap_binding_indices)
                penalty_tolerance_scale = _current_penalty_scale()
                update_feasibility_tol = max(
                    update_feasibility_tol / penalty_tolerance_scale,
                    settings.feasibility_tol,
                )
                update_stationarity_tol = max(
                    update_stationarity_tol / penalty_tolerance_scale,
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
