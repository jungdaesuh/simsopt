from __future__ import annotations

from typing import Mapping

import numpy as np

from .single_stage_geometry import (
    topology_gate_deficit,
    topology_gate_rejection_increment,
)
from .single_stage_search_policy import (
    HardwareSearchPolicy,
    SearchContext,
    decide_hardware_search_action,
)

_FINITE_SCALAR_FIELDS = (
    "total",
    "physics_total",
    "base_total",
    "J_QS",
    "J_QS_objective",
    "J_Boozer",
    "J_Boozer_objective",
    "J_iota",
    "J_volume",
    "J_len",
    "J_cc",
    "J_cs",
    "J_surf",
    "J_curvature",
    "max_violation",
    "max_feasibility_violation",
    "stationarity_norm",
    "metric_stationarity_norm",
)
_FINITE_VECTOR_FIELDS = (
    "grad",
    "dJ_QS",
    "dJ_QS_objective",
    "dJ_Boozer",
    "dJ_Boozer_objective",
    "dJ_iota",
    "dJ_volume",
    "dJ_len",
    "dJ_cc",
    "dJ_cs",
    "dJ_surf",
    "dJ_curvature",
    "surface_weights",
    "constraint_values",
    "dual_update_values",
    "feasibility_values",
    "constraint_activity_tolerances",
)
_FINITE_VECTOR_LIST_FIELDS = ("constraint_grads",)
_FINITE_EPS = float(np.finfo(float).eps)
_DEFAULT_FRONTIER_SEARCH_CONTRACT_PENALTY_SCALE = 4.0


def annotate_search_evaluation_finiteness(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    annotated = dict(evaluation)
    invalid_fields: list[str] = []

    for field_name in _FINITE_SCALAR_FIELDS:
        if field_name not in annotated or annotated[field_name] is None:
            continue
        if not np.isfinite(float(annotated[field_name])):
            invalid_fields.append(field_name)

    for field_name in _FINITE_VECTOR_FIELDS:
        if field_name not in annotated or annotated[field_name] is None:
            continue
        if not _finite_array(annotated[field_name]):
            invalid_fields.append(field_name)

    for field_name in _FINITE_VECTOR_LIST_FIELDS:
        values = annotated.get(field_name)
        if values is None:
            continue
        for index, value in enumerate(values):
            if not _finite_array(value):
                invalid_fields.append(f"{field_name}[{index}]")

    annotated["finite_eval_ok"] = not invalid_fields
    annotated["nonfinite_fields"] = list(invalid_fields)
    if invalid_fields:
        annotated["nonfinite_evaluation"] = True
    return annotated


def evaluate_frontier_trust_status(
    search_eval: Mapping[str, object],
    *,
    enabled: bool,
    threshold: float | None = None,
) -> dict[str, object]:
    if not enabled:
        return {
            "enabled": False,
            "ok": None,
            "residual": None,
            "threshold": None,
            "excess": None,
        }
    if threshold is None:
        raise ValueError("frontier trust evaluation requires a threshold")
    residual = float(search_eval["J_Boozer"])
    trust_threshold = float(threshold)
    excess = max(residual - trust_threshold, 0.0)
    return {
        "enabled": True,
        "ok": bool(np.isfinite(residual) and residual <= trust_threshold),
        "residual": residual,
        "threshold": trust_threshold,
        "excess": excess,
    }


def evaluate_frontier_trust_penalty(
    search_eval: Mapping[str, object],
    *,
    enabled: bool,
    threshold: float | None = None,
    penalty_scale: float | None = None,
) -> dict[str, object]:
    grad = np.asarray(search_eval["grad"], dtype=float)
    if not enabled:
        return {
            "enabled": False,
            "penalty": 0.0,
            "grad": np.zeros_like(grad),
            "scale": None,
            "excess_ratio": None,
        }
    if threshold is None or penalty_scale is None:
        raise ValueError("frontier trust penalty requires threshold and penalty_scale")
    scale = max(float(penalty_scale), _FINITE_EPS)
    residual = float(search_eval["J_Boozer"])
    excess_ratio = max((residual - float(threshold)) / scale, 0.0)
    if excess_ratio == 0.0:
        penalty_grad = np.zeros_like(grad)
    else:
        penalty_grad = (
            (2.0 * excess_ratio / scale)
            * np.asarray(search_eval["dJ_Boozer"], dtype=float)
        )
    return {
        "enabled": True,
        "penalty": float(excess_ratio ** 2),
        "grad": penalty_grad,
        "scale": scale,
        "excess_ratio": float(excess_ratio),
    }


def annotate_frontier_search_eval(
    search_eval: Mapping[str, object],
    *,
    enabled: bool,
    threshold: float | None = None,
    penalty_scale: float | None = None,
) -> dict[str, object]:
    annotated = annotate_search_evaluation_finiteness(search_eval)
    trust_status = evaluate_frontier_trust_status(
        annotated,
        enabled=enabled,
        threshold=threshold,
    )
    trust_penalty = evaluate_frontier_trust_penalty(
        annotated,
        enabled=enabled,
        threshold=threshold,
        penalty_scale=penalty_scale,
    )
    if not enabled:
        return annotated
    annotated["frontier_base_total"] = float(annotated["total"])
    annotated["frontier_rank_total"] = (
        float(annotated["total"]) + trust_penalty["penalty"]
    )
    annotated["total"] = annotated["frontier_rank_total"]
    annotated["grad"] = np.asarray(annotated["grad"], dtype=float) + trust_penalty["grad"]
    annotated["frontier_trust_ok"] = trust_status["ok"]
    annotated["frontier_boozer_trust_threshold"] = trust_status["threshold"]
    annotated["frontier_boozer_trust_excess"] = trust_status["excess"]
    annotated["frontier_boozer_trust_penalty_scale"] = trust_penalty["scale"]
    annotated["frontier_boozer_trust_excess_ratio"] = trust_penalty["excess_ratio"]
    annotated["frontier_trust_penalty"] = trust_penalty["penalty"]
    return annotated


def evaluate_frontier_hardware_search_contract(
    hardware_status: Mapping[str, object],
    *,
    policy: HardwareSearchPolicy,
    context: SearchContext,
) -> dict[str, object]:
    decision = decide_hardware_search_action(policy, hardware_status, context)
    violation_ratios = hardware_violation_ratios(hardware_status)
    return {
        "success": bool(hardware_status["success"]),
        "violations": list(hardware_status.get("violations", [])),
        "violation_ratios": violation_ratios,
        "max_violation_ratio": max(violation_ratios.values(), default=0.0),
        "reject": decision.reject,
        "warning_only": decision.warning_only,
        "rejection_increment": decision.rejection_increment,
        "reason": decision.reason,
    }


def evaluate_frontier_hardware_search_penalty(
    hardware_status: Mapping[str, object],
    *,
    previous_objective: float,
    penalty_scale: float = _DEFAULT_FRONTIER_SEARCH_CONTRACT_PENALTY_SCALE,
) -> dict[str, object]:
    violation_ratios = hardware_violation_ratios(hardware_status)
    max_violation_ratio = max(violation_ratios.values(), default=0.0)
    penalty = (
        _frontier_search_contract_penalty_base(previous_objective)
        * max(float(penalty_scale), 0.0)
        * max_violation_ratio
    )
    return {
        "success": bool(hardware_status["success"]),
        "violations": list(hardware_status.get("violations", [])),
        "violation_ratios": violation_ratios,
        "max_violation_ratio": float(max_violation_ratio),
        "penalty": float(penalty),
        "penalty_scale": float(max(float(penalty_scale), 0.0)),
    }


def evaluate_frontier_topology_search_contract(
    topology_status: Mapping[str, object],
    *,
    previous_objective: float,
    penalty_scale: float,
) -> dict[str, object]:
    enabled = bool(topology_status["enabled"])
    success = bool(topology_status["success"])
    deficit = topology_gate_deficit(topology_status)
    rejection_increment = None
    if enabled and not success:
        rejection_increment = topology_gate_rejection_increment(
            previous_objective,
            topology_status,
            penalty_scale,
        )
    return {
        "enabled": enabled,
        "success": success,
        "deficit": float(deficit),
        "reject": bool(enabled and not success),
        "rejection_increment": rejection_increment,
    }


def evaluate_frontier_topology_search_penalty(
    topology_status: Mapping[str, object],
    *,
    previous_objective: float,
    penalty_scale: float,
) -> dict[str, object]:
    deficit = topology_gate_deficit(topology_status)
    penalty = (
        _frontier_search_contract_penalty_base(previous_objective)
        * max(float(penalty_scale), 0.0)
        * float(deficit)
    )
    return {
        "enabled": bool(topology_status["enabled"]),
        "success": bool(topology_status["success"]),
        "deficit": float(deficit),
        "penalty": float(penalty),
        "penalty_scale": float(max(float(penalty_scale), 0.0)),
    }


def apply_frontier_search_contract_penalties(
    search_eval: Mapping[str, object],
    *,
    hardware_penalty: Mapping[str, object] | None = None,
    topology_penalty: Mapping[str, object] | None = None,
) -> dict[str, object]:
    annotated = dict(search_eval)
    existing_rank_total = float(annotated.get("frontier_rank_total", annotated["total"]))
    contract_penalty = 0.0

    if hardware_penalty is not None:
        contract_penalty += float(hardware_penalty["penalty"])
        annotated["frontier_hardware_penalty"] = float(hardware_penalty["penalty"])
        annotated["frontier_hardware_penalty_scale"] = float(
            hardware_penalty["penalty_scale"]
        )
        annotated["frontier_hardware_violation_ratios"] = dict(
            hardware_penalty["violation_ratios"]
        )
        annotated["frontier_hardware_max_violation_ratio"] = float(
            hardware_penalty["max_violation_ratio"]
        )

    if topology_penalty is not None:
        contract_penalty += float(topology_penalty["penalty"])
        annotated["frontier_topology_penalty"] = float(topology_penalty["penalty"])
        annotated["frontier_topology_penalty_scale"] = float(
            topology_penalty["penalty_scale"]
        )
        annotated["frontier_topology_deficit"] = float(topology_penalty["deficit"])

    annotated["frontier_contract_penalty"] = float(contract_penalty)
    annotated["frontier_rank_total"] = existing_rank_total + float(contract_penalty)
    annotated["total"] = annotated["frontier_rank_total"]
    return annotated


def evaluate_frontier_hard_invalidation(
    *,
    search_eval: Mapping[str, object] | None,
    surface_success: bool,
    surface_status: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not surface_success:
        if surface_status is not None:
            if not all(bool(value) for value in surface_status.get("solve_success", [])):
                return {
                    "invalid": True,
                    "reason": "surface_solve_failed",
                    "fields": ["solve_success"],
                }
            if any(bool(value) for value in surface_status.get("self_intersections", [])):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["self_intersections"],
                }
            if not bool(surface_status.get("volumes_ordered", True)):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["volumes"],
                }
            if not bool(surface_status.get("gap_ok", True)):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["adjacent_gaps"],
                }
            if not bool(surface_status.get("vessel_gap_ok", True)):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["outer_vessel_gap"],
                }
            if not bool(surface_status.get("nesting_ok", True)):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["bad_nesting_phis"],
                }
        return {
            "invalid": True,
            "reason": "surface_solve_failed",
            "fields": ["surface_status"],
        }
    if search_eval is None:
        return {
            "invalid": True,
            "reason": "missing_search_eval",
            "fields": ["search_eval"],
        }
    invalid_fields = list(search_eval.get("nonfinite_fields", []))
    if search_eval.get("finite_eval_ok") is False or invalid_fields:
        return {
            "invalid": True,
            "reason": "nonfinite_evaluation",
            "fields": invalid_fields,
        }
    return {
        "invalid": False,
        "reason": None,
        "fields": [],
    }


def hardware_violation_ratios(
    hardware_status: Mapping[str, object],
) -> dict[str, float]:
    ratios = {
        "curve_curve_min_dist": _lower_bound_violation_ratio(
            hardware_status.get("curve_curve_min_dist"),
            hardware_status.get("cc_dist"),
        ),
        "curve_surface_min_dist": _lower_bound_violation_ratio(
            hardware_status.get("curve_surface_min_dist"),
            hardware_status.get("cs_dist"),
        ),
        "surface_vessel_min_dist": _lower_bound_violation_ratio(
            hardware_status.get("surface_vessel_min_dist"),
            hardware_status.get("ss_dist"),
        ),
        "max_curvature": _upper_bound_violation_ratio(
            hardware_status.get("max_curvature"),
            hardware_status.get("curvature_threshold"),
        ),
        "banana_current": _box_bound_violation_ratio(
            hardware_status.get("banana_current_A"),
            hardware_status.get("banana_current_max_A"),
        ),
        "tf_current": _box_bound_violation_ratio(
            hardware_status.get("tf_current_A"),
            hardware_status.get("tf_current_limit_A"),
        ),
    }
    for name, entry in dict(hardware_status.get("constraints", {})).items():
        threshold = abs(float(entry["threshold"]))
        if threshold > 0.0:
            ratios[str(name)] = float(entry["violation"]) / max(
                threshold,
                _FINITE_EPS,
            )
    explicit_ratios = hardware_status.get("violation_ratios")
    if explicit_ratios is not None:
        for name, value in dict(explicit_ratios).items():
            ratios[str(name)] = float(value)
    return ratios


def _lower_bound_violation_ratio(value, minimum) -> float:
    if value is None or minimum is None:
        return 0.0
    minimum_value = max(abs(float(minimum)), _FINITE_EPS)
    return max(float(minimum) - float(value), 0.0) / minimum_value


def _upper_bound_violation_ratio(value, maximum) -> float:
    if value is None or maximum is None:
        return 0.0
    maximum_value = max(abs(float(maximum)), _FINITE_EPS)
    return max(float(value) - float(maximum), 0.0) / maximum_value


def _box_bound_violation_ratio(value, maximum) -> float:
    if value is None or maximum is None:
        return 0.0
    maximum_value = max(abs(float(maximum)), _FINITE_EPS)
    return max(abs(float(value)) - float(maximum), 0.0) / maximum_value


def _frontier_search_contract_penalty_base(previous_objective: float) -> float:
    return max(abs(float(previous_objective)), 1.0)


def _finite_array(value) -> bool:
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
