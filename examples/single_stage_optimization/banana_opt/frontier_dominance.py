from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

PARETO_OBJECTIVE_SPECS = (
    ("iota", "max", "FINAL_IOTA"),
    ("volume", "max", "FINAL_VOLUME"),
    ("qa_error", "min", "NONQS_RATIO"),
    ("boozer_residual", "min", "BOOZER_RESIDUAL"),
)

PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION = "frontier_pareto_normalization_v1"
PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE = (
    "seed_relative_reference_fraction_with_floor"
)
PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR = (
    "fixed_ideal_nadir_span_with_floor"
)
PARETO_OBJECTIVE_NORMALIZATION_SPEC_SCHEMA_VERSION = (
    "frontier_pareto_normalization_spec_v1"
)
PARETO_OBJECTIVE_NORMALIZATION_RULES = {
    "iota": {
        "direction": "max",
        "scale_kind": "reference_fraction_with_floor",
        "reference_fraction": 0.25,
        "floor": 0.05,
    },
    "volume": {
        "direction": "max",
        "scale_kind": "reference_fraction_with_floor",
        "reference_fraction": 0.10,
        "floor": 0.01,
    },
    "qa_error": {
        "direction": "min",
        "scale_kind": "reference_fraction_with_floor",
        "reference_fraction": 1.0,
        "floor": 1.0e-6,
    },
    "boozer_residual": {
        "direction": "min",
        "scale_kind": "reference_fraction_with_floor",
        "reference_fraction": 1.0,
        "floor": 1.0e-6,
    },
}
PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES = {
    "iota": {
        "direction": "max",
        "scale_kind": "ideal_nadir_span_with_floor",
        "floor": 0.05,
    },
    "volume": {
        "direction": "max",
        "scale_kind": "ideal_nadir_span_with_floor",
        "floor": 0.01,
    },
    "qa_error": {
        "direction": "min",
        "scale_kind": "ideal_nadir_span_with_floor",
        "floor": 1.0e-6,
    },
    "boozer_residual": {
        "direction": "min",
        "scale_kind": "ideal_nadir_span_with_floor",
        "floor": 1.0e-6,
    },
}
SUPPORTED_PARETO_NORMALIZATION_KINDS = (
    PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
    PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
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
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> float | None:
    squared_distance = 0.0
    for metric_name, _, _ in PARETO_OBJECTIVE_SPECS:
        left_value = left_metrics.get(metric_name)
        right_value = right_metrics.get(metric_name)
        if left_value is None or right_value is None:
            return None
        reference_value = None if reference_metrics is None else reference_metrics.get(metric_name)
        scale = objective_metric_scale(
            metric_name,
            reference_value,
            pareto_objective_normalization=pareto_objective_normalization,
        )
        squared_distance += ((float(left_value) - float(right_value)) / scale) ** 2
    return math.sqrt(squared_distance)


def objective_metric_scale(
    metric_name: str,
    reference_value: float | None,
    *,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> float:
    if (
        pareto_objective_normalization is not None
        and str(pareto_objective_normalization.get("kind"))
        == PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR
    ):
        ideal_metrics = pareto_objective_normalization.get("ideal_metrics")
        nadir_metrics = pareto_objective_normalization.get("nadir_metrics")
        if not isinstance(ideal_metrics, Mapping) or not isinstance(nadir_metrics, Mapping):
            raise ValueError(
                "fixed ideal/nadir Pareto normalization requires ideal_metrics and nadir_metrics"
            )
        if metric_name not in ideal_metrics or metric_name not in nadir_metrics:
            raise ValueError(
                f"fixed ideal/nadir Pareto normalization is missing {metric_name!r}"
            )
        rules = PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES[metric_name]
        return max(
            abs(float(ideal_metrics[metric_name]) - float(nadir_metrics[metric_name])),
            float(rules["floor"]),
        )
    rules = PARETO_OBJECTIVE_NORMALIZATION_RULES[metric_name]
    base = 0.0 if reference_value is None else abs(float(reference_value))
    return max(
        base * float(rules["reference_fraction"]),
        float(rules["floor"]),
    )


def objective_metric_direction_map() -> dict[str, str]:
    return {
        metric_name: direction for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS
    }


def load_pareto_normalization_spec(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    schema_version = payload.get("schema_version", payload.get("SCHEMA_VERSION"))
    if schema_version != PARETO_OBJECTIVE_NORMALIZATION_SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"{path} must declare schema_version="
            f"{PARETO_OBJECTIVE_NORMALIZATION_SPEC_SCHEMA_VERSION!r}; got {schema_version!r}"
        )
    return payload


def build_pareto_objective_normalization(
    reference_metrics: Mapping[str, float] | None,
    *,
    kind: str = PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
    normalization_spec_path: str | Path | None = None,
) -> dict[str, object]:
    resolved_reference_metrics = None
    if reference_metrics is not None:
        resolved_reference_metrics = {
            metric_name: float(reference_metrics[metric_name])
            for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
            if reference_metrics.get(metric_name) is not None
        }
    if kind == PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE:
        return {
            "schema_version": PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
            "kind": PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
            "distance_metric": "euclidean",
            "reference_metrics": resolved_reference_metrics,
            "metric_rules": {
                metric_name: dict(rule_payload)
                for metric_name, rule_payload in PARETO_OBJECTIVE_NORMALIZATION_RULES.items()
            },
        }
    if kind != PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR:
        raise ValueError(f"Unsupported Pareto normalization kind: {kind}")
    if normalization_spec_path is None:
        raise ValueError(
            "fixed ideal/nadir normalization requires --frontier-normalization-spec-file"
        )
    normalization_spec = load_pareto_normalization_spec(normalization_spec_path)
    ideal_metrics = _coerce_defined_normalization_metrics(
        normalization_spec,
        field_name="ideal_metrics",
    )
    nadir_metrics = _coerce_defined_normalization_metrics(
        normalization_spec,
        field_name="nadir_metrics",
    )
    return {
        "schema_version": PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
        "kind": PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
        "distance_metric": "euclidean",
        "reference_metrics": resolved_reference_metrics,
        "ideal_metrics": ideal_metrics,
        "nadir_metrics": nadir_metrics,
        "metric_rules": {
            metric_name: dict(rule_payload)
            for metric_name, rule_payload in PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES.items()
        },
    }


def resolve_pareto_normalization_reference_metrics(
    fallback_reference_metrics: Mapping[str, float | None] | None,
    *,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> Mapping[str, float | None] | None:
    if pareto_objective_normalization is None:
        return fallback_reference_metrics
    reference_metrics = pareto_objective_normalization.get("reference_metrics")
    if not isinstance(reference_metrics, Mapping):
        return fallback_reference_metrics
    return {
        metric_name: (
            None if reference_metrics.get(metric_name) is None else float(reference_metrics[metric_name])
        )
        for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
    }


def _coerce_defined_normalization_metrics(
    payload: Mapping[str, object],
    *,
    field_name: str,
) -> dict[str, float]:
    metrics_payload = payload.get(field_name)
    if not isinstance(metrics_payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    metrics = {
        metric_name: float(metrics_payload[metric_name])
        for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
        if metric_name in metrics_payload
    }
    if len(metrics) != len(PARETO_OBJECTIVE_SPECS):
        missing = [
            metric_name
            for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
            if metric_name not in metrics
        ]
        raise ValueError(f"{field_name} is missing metrics: {missing}")
    return metrics
