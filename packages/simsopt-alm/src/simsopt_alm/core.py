"""Bertsekas / Birgin–Martinez ALM math: L, dual project, routing, settings.

Formula bodies only. Continuation and the inner L-BFGS-B loop live in
:mod:`simsopt_alm.control`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import nnls

ALM_SCHEMA_VERSION = "alm_normalized_constraints_v2"

def _finite_alm_value(name: str, value) -> float:
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{name} must be finite")
    return value_f

def _finite_alm_value_or_none(name: str, value) -> Optional[float]:
    if value is None:
        return None
    return _finite_alm_value(name, value)

def _finite_alm_integer(name: str, value) -> int:
    _finite_alm_value(name, value)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)

def _finite_alm_integer_or_none(name: str, value) -> Optional[int]:
    if value is None:
        return None
    return _finite_alm_integer(name, value)

def _nonnegative_alm_integer(name: str, value) -> int:
    value_i = _finite_alm_integer(name, value)
    if value_i < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value_i

def _strict_alm_bool(name: str, value) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value

_MUST_BE_POSITIVE = (lambda value: value <= 0.0, "must be positive")
_POSITIVE_WHEN_PROVIDED = (lambda value: value <= 0.0, "must be positive when provided")
_MUST_BE_NONNEGATIVE = (lambda value: value < 0.0, "must be nonnegative")
_MUST_EXCEED_ONE = (lambda value: value <= 1.0, "must be greater than 1")
_IN_OPEN_UNIT_INTERVAL = (lambda value: not (0.0 < value < 1.0), "must be in (0, 1)")

# ``ALMSettings`` numeric fields: (name, finite parser, (out_of_range, message)).
# All fields are parsed in this order before any range check runs; a None
# (disabled optional) value skips its range check. Zero subproblem
# continuations is valid: the inner loop is ``range(n + 1)``.
_ALM_SETTINGS_BOUNDS = (
    ("max_outer_iterations", _finite_alm_integer, _MUST_BE_POSITIVE),
    ("max_subproblem_continuations", _finite_alm_integer, _MUST_BE_NONNEGATIVE),
    ("penalty_init", _finite_alm_value, _MUST_BE_POSITIVE),
    ("penalty_scale", _finite_alm_value, _MUST_EXCEED_ONE),
    ("penalty_max", _finite_alm_value_or_none, _POSITIVE_WHEN_PROVIDED),
    ("penalty_sufficient_decrease_tau", _finite_alm_value, _IN_OPEN_UNIT_INTERVAL),
    ("feasibility_tol", _finite_alm_value, _MUST_BE_POSITIVE),
    ("stationarity_tol", _finite_alm_value, _MUST_BE_POSITIVE),
    ("trust_radius_init", _finite_alm_value_or_none, _POSITIVE_WHEN_PROVIDED),
    ("trust_radius_min", _finite_alm_value, _MUST_BE_POSITIVE),
    ("trust_radius_shrink", _finite_alm_value, _IN_OPEN_UNIT_INTERVAL),
    ("trust_radius_grow", _finite_alm_value, _MUST_EXCEED_ONE),
    ("max_inner_attempts", _finite_alm_integer, _MUST_BE_POSITIVE),
    ("relaxed_feasibility_gate_cap", _finite_alm_value, _MUST_BE_POSITIVE),
    ("multiplier_max", _finite_alm_value_or_none, _POSITIVE_WHEN_PROVIDED),
    ("history_max_entries", _finite_alm_integer_or_none, _POSITIVE_WHEN_PROVIDED),
)

@dataclass(frozen=True)
class ALMSettings:
    max_outer_iterations: int = 10
    max_subproblem_continuations: int = 20
    penalty_init: float = 1.0
    penalty_scale: float = 10.0
    # Production ALM implementations commonly expose a maximum penalty
    # safeguard; keeping this bounded avoids runaway objective scaling in the
    # L-BFGS-B inner solve.
    penalty_max: Optional[float] = 1.0e8
    feasibility_tol: float = 1e-6
    stationarity_tol: float = 1e-6
    trust_radius_init: Optional[float] = None
    trust_radius_min: float = 1e-4
    trust_radius_shrink: float = 0.5
    trust_radius_grow: float = 1.5
    max_inner_attempts: int = 4
    relaxed_feasibility_gate_cap: float = 1e-2
    multiplier_max: Optional[float] = 1.0e6
    # Read only by ALMHistoryRecorder.from_settings; minimize_alm keeps no history.
    history_max_entries: Optional[int] = 512
    # Opt-in: when True, a hard-feasible/surrogate-active signal mismatch with
    # a live surrogate positive shift stays on the bounded inner-continuation
    # path instead of taking the penalty-increase arm. This does not
    # bypass the dual-update stationarity gate.
    continue_on_signal_mismatch: bool = False
    # ALGENCAN sufficient-decrease safeguard (Birgin & Martinez 2014,
    # Algorithm 1.1, default tau=0.5): the generic infeasible penalty-increase
    # arm holds the penalty fixed when the shifted infeasibility measure
    # ||max(g, -lambda/rho)||_inf shrank to <= tau x the previous accepted
    # iterate's measure. Without this the penalty ramps unconditionally every
    # outer iteration and a late-activating constraint meets an
    # already-inflated penalty (the resulting jump in step size can break a
    # stateful inner solve).
    penalty_sufficient_decrease_tau: float = 0.5

    def __post_init__(self) -> None:
        # Every construction path is validated here, so a value such as
        # `ALMSettings(trust_radius_grow=0.5)` is rejected rather than
        # silently shrinking the trust radius.
        parsed = {
            name: parse(f"ALMSettings.{name}", getattr(self, name))
            for name, parse, _bound in _ALM_SETTINGS_BOUNDS
        }
        _strict_alm_bool(
            "ALMSettings.continue_on_signal_mismatch",
            self.continue_on_signal_mismatch,
        )
        for name, _parse, (out_of_range, requirement) in _ALM_SETTINGS_BOUNDS:
            value = parsed[name]
            if value is not None and out_of_range(value):
                raise ValueError(f"ALMSettings.{name} {requirement}")
            if name == "penalty_max" and value is not None and (
                value < parsed["penalty_init"]
            ):
                raise ValueError(
                    f"ALMSettings.penalty_max ({self.penalty_max}) must be >= "
                    f"penalty_init ({self.penalty_init})"
                )

@dataclass(frozen=True)
class ALMConstraintSignalState:
    explicit_hybrid_signals: bool
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

_HYBRID_SIGNAL_FIELDS = (
    "hard_signed_constraint_values",
    "hard_violation_values",
    "surrogate_signed_constraint_values",
    "hard_dual_update_values",
)

def require_positive_alm_threshold(name: str, value) -> float:
    """Validate that an ALM threshold is finite and strictly positive.

    Caller pre-handles the disabled-constraint case (``value is None``).
    Zero, negative, NaN, and infinity are all rejected. Used at the shared
    threshold-input boundary so the downstream ``max(raw, FLOOR)`` floor in
    metadata constructors becomes defense-in-depth, not silent recovery.
    """
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(
            f"ALM threshold {name!r} must be a finite positive value; got {value!r}"
        )
    return value_f

def positive_part(value: float) -> float:
    return float(max(value, 0.0))

def upper_bound_residual(metric_value: float, upper_bound: float) -> float:
    return positive_part(metric_value - upper_bound)

def lower_bound_residual(metric_value: float, lower_bound: float) -> float:
    return positive_part(lower_bound - metric_value)

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

def _constraint_grad_list(constraint_grads) -> List[np.ndarray]:
    return [
        np.asarray(constraint_grad, dtype=float) for constraint_grad in constraint_grads
    ]

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
    # Stored ndarrays are copied, except constraint_grads. A float ndarray
    # there aliases the caller; the solver snapshots it on entry.
    # `np.asarray` keeps the alias when the dtype already matches.
    result = {
        "total": float(total_value),
        "base_value": float(base_value),
        "base_grad": np.asarray(base_grad, dtype=float).copy(),
        "grad": np.asarray(total_grad, dtype=float).copy(),
        "constraint_values": np.asarray(constraint_values, dtype=float).copy(),
        "constraint_grads": [
            np.asarray(constraint_grad, dtype=float)
            for constraint_grad in constraint_grads
        ],
        "dual_update_values": dual_update_array.copy(),
        "feasibility_values": feasibility_array.copy(),
        "max_violation": max_feasibility_violation,
        "max_feasibility_violation": max_feasibility_violation,
        "stationarity_norm": stationarity_norm,
    }
    if positive_shift_values is not None:
        # Copy to avoid aliasing caller-owned mutable buffers, matching
        # the ownership contract of the principal evaluation arrays above.
        result["positive_shift_values"] = np.asarray(
            positive_shift_values,
            dtype=float,
        ).copy()
    if augmented_term_by_constraint is not None:
        result["augmented_term_by_constraint"] = np.asarray(
            augmented_term_by_constraint,
            dtype=float,
        ).copy()
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
) -> Tuple[np.ndarray, np.ndarray]:
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

def _as_float_list(values) -> List[float]:
    return np.asarray(values, dtype=float).reshape(-1).tolist()

def _max_value(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else 0.0

def alm_raw_dual_estimates(multipliers, evaluation: dict) -> Optional[List[float]]:
    constraint_scales = evaluation.get("constraint_scales")
    if constraint_scales is None:
        return None
    scales = np.asarray(constraint_scales, dtype=float)
    multiplier_array = np.asarray(multipliers, dtype=float)
    if scales.shape != multiplier_array.shape:
        raise ValueError("constraint_scales shape must match multipliers")
    return _as_float_list(multiplier_array / scales)

def _conditioning_metrics(evaluation: dict) -> Dict[str, Optional[float]]:
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
        None if base_objective == 0.0 else abs(penalty_objective) / abs(base_objective)
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

def _kkt_base_grad(evaluation: dict) -> np.ndarray:
    """Gradient of the bare objective for the KKT residual.

    KKT stationarity is `‖∇f + Σλ_i∇c_i‖`, NOT `‖∇L_A + Σλ_i∇c_i‖`. The
    augmented gradient `∇L_A` already contains the active-constraint term, so
    nnls would collapse the residual to ~0 once the inner solve converges and
    hide multiplier-quality defects. Order: `base_grad`, `metric_grad`, `grad`.
    """
    return np.asarray(
        evaluation.get(
            "base_grad",
            evaluation.get("metric_grad", evaluation["grad"]),
        ),
        dtype=float,
    )

def _surrogate_kkt_stationarity_norm(
    evaluation: dict,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> Optional[float]:
    surrogate_values = routing_state.signal_state.surrogate_signed_constraint_values
    return _kkt_stationarity_norm(
        _kkt_base_grad(evaluation),
        evaluation.get("constraint_grads"),
        surrogate_values,
        np.maximum(surrogate_values, 0.0),
        _constraint_activity_tolerances(evaluation, surrogate_values),
        feasibility_gate,
    )

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
    multiplier_max: Optional[float],
) -> Tuple[np.ndarray, bool, List[int]]:
    updated = _updated_nonnegative_multipliers(
        multipliers,
        dual_update_values,
        penalty,
    )
    # Defense in depth after _require_finite_evaluation rejects non-finite
    # evaluation fields. `updated > cap` is False for NaN and `np.minimum(NaN, cap)` is
    # NaN, so a non-finite multiplier would otherwise propagate silently
    # with `cap_binding=False`. Surface the contract violation loudly.
    if not np.all(np.isfinite(updated)):
        nonfinite_indices = np.flatnonzero(~np.isfinite(updated)).tolist()
        raise ValueError(
            "ALM multiplier projection produced non-finite values at indices "
            f"{nonfinite_indices}; upstream evaluation should have been rejected "
            "by _require_finite_evaluation"
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

def _tolerance_schedule_penalty(penalty) -> float:
    values = np.asarray(penalty, dtype=float)
    if values.shape == ():
        return float(values)
    return float(np.min(values))

def _penalty_schedule_tolerance(tolerance: float, penalty) -> float:
    return max(float(tolerance), 1.0 / _tolerance_schedule_penalty(penalty))

def _penalty_feasibility_schedule_tolerance(tolerance: float, penalty) -> float:
    return max(float(tolerance), 1.0 / _tolerance_schedule_penalty(penalty) ** 0.1)

def alm_penalty_schedule_tolerances(
    settings: ALMSettings,
    penalty: Union[float, Sequence[float], np.ndarray],
) -> Tuple[float, float]:
    """Return the feasibility and stationarity tolerances for one ALM penalty."""
    return (
        _penalty_feasibility_schedule_tolerance(settings.feasibility_tol, penalty),
        _penalty_schedule_tolerance(settings.stationarity_tol, penalty),
    )

def _next_penalty(
    penalty: float,
    *,
    penalty_scale: float,
    penalty_max: Optional[float],
) -> Tuple[float, bool, float]:
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
    # ``feasibility_values`` is the canonical per-constraint state.  Keep the
    # producer summary in the evaluation for compatibility, but never let it
    # override the scalar used by ALM decisions or diagnostics.
    max_feasibility_violation = _max_value(feasibility_values)
    return (
        solver_constraint_values,
        feasibility_values,
        dual_update_values,
        max_feasibility_violation,
    )

def _extract_hybrid_signal_state(
    evaluation: dict,
) -> ALMConstraintSignalState:
    explicit_hybrid_signals = any(
        key in evaluation for key in _HYBRID_SIGNAL_FIELDS
    )
    (
        solver_constraint_values,
        feasibility_values,
        dual_update_values,
        _max_feasibility_violation,
    ) = _extract_constraint_state(evaluation)
    if explicit_hybrid_signals:
        missing_fields = [
            field for field in _HYBRID_SIGNAL_FIELDS if field not in evaluation
        ]
        if missing_fields:
            raise KeyError(
                "ALM hybrid signal evaluation missing required fields: "
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
        explicit_hybrid_signals=bool(explicit_hybrid_signals),
        hard_signed_constraint_values=hard_signed_constraint_values,
        hard_violation_values=hard_violation_values,
        surrogate_signed_constraint_values=surrogate_signed_constraint_values,
        preferred_dual_update_values=preferred_dual_update_values,
    )

def _constraint_activity_tolerances(
    evaluation: dict, constraint_values: np.ndarray
) -> np.ndarray:
    raw_tolerances = evaluation.get("constraint_activity_tolerances")
    if raw_tolerances is None:
        return np.zeros_like(constraint_values)

    tolerances = np.asarray(raw_tolerances, dtype=float)
    if tolerances.shape == constraint_values.shape:
        return tolerances
    if tolerances.size == 1:
        return np.full_like(constraint_values, float(tolerances.reshape(())))
    raise ValueError(
        "constraint_activity_tolerances shape must match constraint_values"
    )

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
    signal_state = _extract_hybrid_signal_state(evaluation)
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
    signal_mismatch_active = signal_state.explicit_hybrid_signals and masks_disagree
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
    hard_feasible_under_gate = _max_value(signal_state.hard_violation_values) <= float(
        feasibility_gate
    )
    # Hard-feasible, yet a row whose surrogate still pushes (positive shift)
    # has a surrogate signal more than the gate away from its hard signal.
    # Identical channels never mismatch, even at an active boundary.
    signals_differ = np.abs(
        signal_state.surrogate_signed_constraint_values
        - signal_state.hard_signed_constraint_values
    ) > float(feasibility_gate)
    direct_boundary_mismatch = (
        signal_state.explicit_hybrid_signals
        and hard_feasible_under_gate
        and np.any((surrogate_positive_shift > 0.0) & signals_differ)
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

def _complementarity_gap(routing_state: ALMConstraintRoutingState) -> float:
    """``sum_i λ⁺_i s_i``, ``s_i = max(0, -g_i)``, on the signal the augmented
    Lagrangian uses (``λ⁺ = max(0, λ + ρg)`` its shifted multipliers). Every
    term is nonnegative, and invariant under (g_i, λ_i, ρ_i) -> (M g_i, λ_i/M,
    ρ_i/M²). At a feasible x it is f(x) - ℓ(x, λ⁺), ℓ = f + sum_i λ⁺_i g_i the
    Lagrangian; for convex f and g it bounds f(x) - f* by weak duality, plus
    ``||∇ℓ(x, λ⁺)|| ||x - x*||`` (∇ℓ at λ⁺ is the augmented gradient)."""
    slack = np.maximum(0.0, -routing_state.signal_state.surrogate_signed_constraint_values)
    return float(np.sum(routing_state.surrogate_positive_shift * slack))

def _kkt_stationarity_norm(
    total_grad,
    constraint_grads,
    constraint_values: np.ndarray,
    feasibility_values: np.ndarray,
    activity_tolerances: np.ndarray,
    feasibility_gate: float,
) -> Optional[float]:
    if constraint_grads is None:
        return None

    total_grad_array = np.asarray(total_grad, dtype=float).reshape(-1)
    if total_grad_array.size == 0:
        return 0.0
    # In `thresholded_physics` mode the base objective is identically zero, so
    # the bare physics gradient is structurally zero and `nnls(A, -grad_f)`
    # collapses to `lambda=0, residual=0` regardless of multiplier quality.
    # The diagnostic is meaningless in that regime; report unavailable instead
    # of emitting a misleading "0" that operators read as "converged KKT".
    if not np.any(total_grad_array):
        return None

    active_constraint_grads = []
    for (
        constraint_grad,
        constraint_value,
        _feasibility_value,
        activity_tolerance,
    ) in zip(
        constraint_grads,
        constraint_values,
        feasibility_values,
        activity_tolerances,
    ):
        if float(constraint_value) < -float(activity_tolerance):
            continue
        active_constraint_grads.append(
            np.asarray(constraint_grad, dtype=float).reshape(-1)
        )

    if not active_constraint_grads:
        return None

    active_matrix = np.column_stack(active_constraint_grads)
    # nnls can raise RuntimeError("too many iterations") on pathological
    # active-Jacobians; bound iterations and surface the failure as None
    # (the caller already treats None as "diagnostic unavailable") rather
    # than aborting the ALM run through a diagnostic helper. Shape errors
    # still propagate as ValueError — only the iteration-cap failure is caught.
    try:
        multipliers, _residual_norm = nnls(
            active_matrix,
            -total_grad_array,
            maxiter=10 * active_matrix.shape[1],
        )
    except RuntimeError as exc:
        warnings.warn(
            f"_kkt_stationarity_norm: nnls failed for active-matrix shape "
            f"{active_matrix.shape}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    residual = total_grad_array + active_matrix @ multipliers
    return float(np.linalg.norm(residual))

def _augmented_stationarity_norm(evaluation: dict) -> float:
    """The evaluator's ``stationarity_norm``, else ``||grad||``: the
    augmented-gradient norm, before any bound reduction."""
    return float(evaluation.get("stationarity_norm", np.linalg.norm(evaluation["grad"])))

def _bound_reduced_stationarity_norm(
    evaluation: dict,
    x,
    base_bounds: Optional[Sequence[Tuple[float, float]]],
) -> float:
    """The loop's stationarity at ``x``: an evaluator's own
    ``stationarity_norm`` (one that is not ``||grad||``) as given; otherwise
    ``||grad||`` without the components pointing out of the box where ``x``
    equals a base bound (``(lower, upper)`` pairs) exactly: the bound's
    multiplier holds them. The x0 clip and L-BFGS-B's projection put a
    coordinate exactly on its bound; any other x is interior or outside."""
    stationarity_norm = _augmented_stationarity_norm(evaluation)
    if base_bounds is None:
        return stationarity_norm
    lower, upper = np.asarray(base_bounds, dtype=float).T
    grad = np.asarray(evaluation["grad"], dtype=float).reshape(-1)
    x_array = np.asarray(x, dtype=float).reshape(-1)
    blocked = ((x_array == upper) & (grad < 0.0)) | ((x_array == lower) & (grad > 0.0))
    if not np.any(blocked) or stationarity_norm != float(np.linalg.norm(grad)):
        return stationarity_norm
    return float(np.linalg.norm(np.where(blocked, 0.0, grad)))

def _stationarity_metrics(
    evaluation: dict,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> Tuple[float, Optional[float], bool]:
    """Return ``(stationarity_norm, kkt_stationarity_norm, signal_mismatch_active)``.

    ``stationarity_norm`` is the raw augmented-Lagrangian gradient norm used as
    the inner-solve convergence trigger; callers that need the same value under
    the ``raw_stationarity_norm`` history-schema key alias it locally.
    ``kkt_stationarity_norm`` is the active-set KKT residual used for the inner
    stationarity gate. Hybrid evaluations compute it on the surrogate channel, the
    differentiable subproblem; mismatch remains a separate success guard.
    """
    stationarity_norm = _augmented_stationarity_norm(evaluation)
    if routing_state.signal_state.explicit_hybrid_signals:
        kkt_stationarity_norm = _surrogate_kkt_stationarity_norm(
            evaluation,
            routing_state,
            feasibility_gate,
        )
    else:
        preferred_dual_update_values = (
            routing_state.signal_state.preferred_dual_update_values
        )
        kkt_stationarity_norm = _kkt_stationarity_norm(
            _kkt_base_grad(evaluation),
            evaluation.get("constraint_grads"),
            preferred_dual_update_values,
            routing_state.signal_state.hard_violation_values,
            _constraint_activity_tolerances(evaluation, preferred_dual_update_values),
            feasibility_gate,
        )
    return (
        stationarity_norm,
        kkt_stationarity_norm,
        bool(routing_state.signal_mismatch_active),
    )

def validate_initial_multipliers(multipliers, n_constraints: int) -> np.ndarray:
    """Validate ALM initial multipliers at the driver boundary.

    Inequality-ALM multipliers are non-negative by construction (Lagrange
    multipliers for ``c_i <= 0``); NaN/Inf are always invalid; shape mismatch
    against ``n_constraints`` would silently broadcast or raise far from the
    actual fault. Returns an owned, finite, non-negative copy.
    """
    arr = np.asarray(multipliers, dtype=float)
    if arr.shape != (n_constraints,):
        raise ValueError(
            f"ALM initial_multipliers shape {tuple(arr.shape)} != ({n_constraints},)"
        )
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise ValueError(f"ALM initial_multipliers non-finite at indices {bad}")
    if (arr < 0.0).any():
        bad = np.where(arr < 0.0)[0].tolist()
        raise ValueError(f"ALM initial_multipliers negative at indices {bad}")
    return arr.copy()
