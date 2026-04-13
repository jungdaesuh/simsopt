from __future__ import annotations

import math
from typing import Mapping

PARETO_OBJECTIVE_SPECS = (
    ("iota", "max", "FINAL_IOTA"),
    ("volume", "max", "FINAL_VOLUME"),
    ("qa_error", "min", "NONQS_RATIO"),
    ("boozer_residual", "min", "BOOZER_RESIDUAL"),
)

REFERENCE_METRIC_FIELDS = {
    "iota": "FRONTIER_REFERENCE_IOTA",
    "volume": "FRONTIER_REFERENCE_VOLUME",
    "qa_error": "FRONTIER_REFERENCE_QA",
    "boozer_residual": "FRONTIER_REFERENCE_BOOZER",
}

DEFAULT_DOMINANCE_TOLERANCE = {
    "iota": 1.0e-6,
    "volume": 1.0e-6,
    "qa_error": 1.0e-6,
    "boozer_residual": 1.0e-9,
}

CONSTRAINT_METRIC_FIELDS = {
    "coil_length": "COIL_LENGTH",
    "curve_curve_min_dist": "CURVE_CURVE_MIN_DIST",
    "curve_surface_min_dist": "CURVE_SURFACE_MIN_DIST",
    "surface_vessel_min_dist": "SURFACE_VESSEL_MIN_DIST",
    "max_curvature": "MAX_CURVATURE",
    "hardware_constraints_ok": "HARDWARE_CONSTRAINTS_OK",
    "final_feasibility_ok": "FINAL_FEASIBILITY_OK",
    "final_topology_gate_success": "FINAL_TOPOLOGY_GATE_SUCCESS",
    "frontier_trust_ok": "FRONTIER_TRUST_OK",
}


def _as_finite_float(value) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def extract_objective_metrics(results: Mapping[str, object]) -> dict[str, float | None]:
    return {
        metric_name: _as_finite_float(results.get(result_key))
        for metric_name, _, result_key in PARETO_OBJECTIVE_SPECS
    }


def extract_reference_metrics(results: Mapping[str, object]) -> dict[str, float | None]:
    return {
        metric_name: _as_finite_float(results.get(result_key))
        for metric_name, result_key in REFERENCE_METRIC_FIELDS.items()
    }


def extract_constraint_metrics(results: Mapping[str, object]) -> dict[str, object]:
    return {
        metric_name: results.get(result_key)
        for metric_name, result_key in CONSTRAINT_METRIC_FIELDS.items()
    }


def objective_metrics_complete(metrics: Mapping[str, float | None]) -> bool:
    return all(metrics.get(metric_name) is not None for metric_name, _, _ in PARETO_OBJECTIVE_SPECS)


def is_certified_results(results: Mapping[str, object]) -> bool:
    if not bool(results.get("FINAL_FEASIBILITY_OK")):
        return False
    if not bool(results.get("HARDWARE_CONSTRAINTS_OK")):
        return False
    topology_ok = results.get("FINAL_TOPOLOGY_GATE_SUCCESS")
    if topology_ok is False:
        return False
    trust_ok = results.get("FRONTIER_TRUST_OK")
    if trust_ok is False:
        return False
    return objective_metrics_complete(extract_objective_metrics(results))


def _better_or_equal(
    direction: str,
    candidate_value: float,
    incumbent_value: float,
    tolerance: float,
) -> bool:
    if direction == "max":
        return candidate_value >= incumbent_value - tolerance
    return candidate_value <= incumbent_value + tolerance


def _strictly_better(
    direction: str,
    candidate_value: float,
    incumbent_value: float,
    tolerance: float,
) -> bool:
    if direction == "max":
        return candidate_value > incumbent_value + tolerance
    return candidate_value < incumbent_value - tolerance


def dominates(
    candidate_metrics: Mapping[str, float],
    incumbent_metrics: Mapping[str, float],
    *,
    tolerance: Mapping[str, float] | None = None,
) -> bool:
    tolerances = DEFAULT_DOMINANCE_TOLERANCE if tolerance is None else tolerance
    strictly_better_on_any = False
    for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS:
        candidate_value = candidate_metrics.get(metric_name)
        incumbent_value = incumbent_metrics.get(metric_name)
        if candidate_value is None or incumbent_value is None:
            return False
        metric_tolerance = float(tolerances.get(metric_name, 0.0))
        if not _better_or_equal(
            direction,
            float(candidate_value),
            float(incumbent_value),
            metric_tolerance,
        ):
            return False
        if _strictly_better(
            direction,
            float(candidate_value),
            float(incumbent_value),
            metric_tolerance,
        ):
            strictly_better_on_any = True
    return strictly_better_on_any


def normalized_objective_distance(
    left_metrics: Mapping[str, float],
    right_metrics: Mapping[str, float],
    *,
    reference_metrics: Mapping[str, float | None] | None = None,
) -> float | None:
    squared_distance = 0.0
    for metric_name, _, _ in PARETO_OBJECTIVE_SPECS:
        left_value = left_metrics.get(metric_name)
        right_value = right_metrics.get(metric_name)
        if left_value is None or right_value is None:
            return None
        reference_value = None if reference_metrics is None else reference_metrics.get(metric_name)
        scale = objective_metric_scale(metric_name, reference_value)
        squared_distance += ((float(left_value) - float(right_value)) / scale) ** 2
    return math.sqrt(squared_distance)


def objective_metric_scale(metric_name: str, reference_value: float | None) -> float:
    base = 0.0 if reference_value is None else abs(float(reference_value))
    if metric_name == "iota":
        return max(base * 0.25, 0.05)
    if metric_name == "volume":
        return max(base * 0.10, 0.01)
    return max(base, 1.0e-6)


def objective_metric_direction_map() -> dict[str, str]:
    return {
        metric_name: direction for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS
    }
