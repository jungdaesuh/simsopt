"""Pareto objective extraction, normalization, and dominance predicates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

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
    if value is None or isinstance(value, (bool, np.bool_)):
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
    """Pareto dominance with Laumanns-2002 ε-tight semantics.

    Returns True iff `candidate_metrics` weakly dominates `incumbent_metrics`
    on every metric in `PARETO_OBJECTIVE_SPECS` and strictly dominates on at
    least one. Weak dominance uses the (>=, -tau) test; strict dominance uses
    the (>, +tau) test (see Laumanns, Thiele, Deb, Zitzler 2002, "Combining
    Convergence and Diversity in Evolutionary Multi-objective Optimization").
    The 2*tau gap between these tests creates an intentional ε-archive
    buffer band where ε-close neighbors are mutually non-dominated; this is
    a convergence/diversity guarantee, not a bug. Per-metric tau defaults to
    `DEFAULT_DOMINANCE_TOLERANCE`.

    Both inputs MUST contain all four `PARETO_OBJECTIVE_SPECS` keys with
    finite numeric values. Missing keys raise KeyError; None values raise
    ValueError. Completeness is the upstream invariant: callers are required
    to filter incomplete members (see `objective_metrics_complete`) before
    invoking this predicate. Non-finite values raise ValueError.
    """
    tolerances = DEFAULT_DOMINANCE_TOLERANCE if tolerance is None else tolerance
    strictly_better_on_any = False
    for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS:
        candidate_value = candidate_metrics[metric_name]
        incumbent_value = incumbent_metrics[metric_name]
        if candidate_value is None or incumbent_value is None:
            raise ValueError(
                f"Pareto metric {metric_name!r} is None; dominance requires "
                f"complete metrics (filter upstream via objective_metrics_complete)"
            )
        if not math.isfinite(float(candidate_value)) or not math.isfinite(
            float(incumbent_value)
        ):
            raise ValueError(f"Non-finite Pareto metric {metric_name!r}")
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


def pareto_scaled_difference(
    metric_name: str,
    left_value: float,
    right_value: float,
    *,
    scale_reference: float | None,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> float:
    """Single-source-of-truth ``(left - right) / scale`` for Pareto metrics.

    The scale is derived from ``objective_metric_scale`` against the supplied
    ``scale_reference`` (the seed reference for the metric). Used by
    ``_normalized_delta`` (recommendation policy) and
    ``normalized_objective_distance`` (archive duplicate detection); both
    funnel through this helper so the seed-relative-vs-ideal-nadir scaling
    rules are enforced exactly once. Lane-internal Chebyshev scalarization
    uses a different per-lane scale (see ``FrontierGoalConfig.{iota,volume,qs,boozer}_scale``)
    and does not call this helper.
    """
    scale = objective_metric_scale(
        metric_name,
        scale_reference,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    return (float(left_value) - float(right_value)) / scale


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
        squared_distance += (
            pareto_scaled_difference(
                metric_name,
                left_value,
                right_value,
                scale_reference=reference_value,
                pareto_objective_normalization=pareto_objective_normalization,
            )
            ** 2
        )
    return math.sqrt(squared_distance)


def objective_metric_scale(
    metric_name: str,
    reference_value: float | None,
    *,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> float:
    """Per-metric normalization scale for `(value - reference) / scale` deltas.

    Two branches:
    - Fixed ideal/nadir branch (`PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR`):
      scale is the ideal-to-nadir span, floored. Direction-aware ordering of
      ideal vs. nadir is enforced at spec-load time by
      `_validate_ideal_nadir_directions`; the `abs(...)` here only guards
      against floating-point sign noise on the difference.
    - Seed-relative branch (default): scale is `abs(reference) * fraction`,
      floored. The `abs(reference_value)` is a no-op for the four metrics
      defined in `PARETO_OBJECTIVE_SPECS` because they are non-negative by
      domain (`iota`, `volume` >= 0; `qa_error`, `boozer_residual` >= 0). It
      is preserved as a defensive sign guard, but it would mask a sign flip
      if a signed Pareto metric were added in the future. Any future signed
      metric MUST switch to the ideal/nadir branch (where direction is
      explicit) rather than rely on this branch.
    """
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
    schema_version = payload.get("schema_version")
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
        normalization_rules = PARETO_OBJECTIVE_NORMALIZATION_RULES
        extra_payload: dict[str, object] = {}
    elif kind == PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR:
        if normalization_spec_path is None:
            raise ValueError(
                "fixed ideal/nadir normalization requires "
                "--frontier-normalization-spec-file"
            )
        normalization_spec = load_pareto_normalization_spec(normalization_spec_path)
        normalization_rules = PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES
        ideal_metrics_payload = _coerce_defined_normalization_metrics(
            normalization_spec,
            field_name="ideal_metrics",
        )
        nadir_metrics_payload = _coerce_defined_normalization_metrics(
            normalization_spec,
            field_name="nadir_metrics",
        )
        _validate_ideal_nadir_directions(
            ideal_metrics_payload,
            nadir_metrics_payload,
        )
        extra_payload = {
            "ideal_metrics": ideal_metrics_payload,
            "nadir_metrics": nadir_metrics_payload,
        }
    else:
        raise ValueError(f"Unsupported Pareto normalization kind: {kind}")

    return {
        "schema_version": PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
        "kind": kind,
        "distance_metric": "euclidean",
        "reference_metrics": resolved_reference_metrics,
        **extra_payload,
        "metric_rules": {
            metric_name: dict(rule_payload)
            for metric_name, rule_payload in normalization_rules.items()
        },
    }


def build_pareto_objective_normalization_from_persisted_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild a Pareto normalization dict from a persisted manifest payload.

    The manifest's ``PARETO_OBJECTIVE_NORMALIZATION`` block is the single
    source of truth for the resolved normalization once the campaign is
    underway: it captures ``kind``, ``ideal_metrics`` / ``nadir_metrics``
    (for ``ideal_nadir``), ``reference_metrics``, ``metric_rules``, and
    ``distance_metric``. The original CLI ``--frontier-normalization-spec-file``
    is intentionally NOT persisted (the resolved values supersede it), so
    a resume that defaults ``args.frontier_normalization_spec_file=None``
    cannot route through :func:`build_pareto_objective_normalization`'s
    spec-file branch for ``ideal_nadir`` campaigns.

    This helper rebuilds an equivalent normalization dict directly from the
    persisted payload, enforcing the same frozen contracts that
    :func:`_validate_pareto_normalization_payload` checks at manifest-load
    time plus the direction-aware ideal/nadir invariant from
    :func:`_validate_ideal_nadir_directions`. Drifted payloads raise
    ``ValueError``.
    """
    schema_version = payload.get("schema_version")
    if schema_version != PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError(
            f"Persisted Pareto normalization payload must declare "
            f"schema_version={PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION!r}; "
            f"got {schema_version!r}"
        )
    if payload.get("distance_metric") != "euclidean":
        raise ValueError(
            "Persisted Pareto normalization payload must use distance_metric='euclidean'"
        )
    kind = payload.get("kind")
    metric_rules_payload = payload.get("metric_rules")
    if not isinstance(metric_rules_payload, Mapping):
        raise ValueError(
            "Persisted Pareto normalization payload missing 'metric_rules' mapping"
        )
    persisted_metric_rules = {
        metric_name: dict(rule_payload)
        for metric_name, rule_payload in metric_rules_payload.items()
    }
    reference_metrics_payload = payload.get("reference_metrics")
    resolved_reference_metrics: dict[str, float] | None
    if reference_metrics_payload is None:
        resolved_reference_metrics = None
    elif isinstance(reference_metrics_payload, Mapping):
        resolved_reference_metrics = {
            metric_name: float(reference_metrics_payload[metric_name])
            for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
            if reference_metrics_payload.get(metric_name) is not None
        }
    else:
        raise ValueError(
            "Persisted Pareto normalization payload 'reference_metrics' must be a mapping or null"
        )
    if kind == PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE:
        if persisted_metric_rules != PARETO_OBJECTIVE_NORMALIZATION_RULES:
            raise ValueError(
                "Persisted seed-relative Pareto normalization metric_rules drifted "
                "from the frozen contract"
            )
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
    if kind == PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR:
        if persisted_metric_rules != PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES:
            raise ValueError(
                "Persisted ideal/nadir Pareto normalization metric_rules drifted "
                "from the frozen contract"
            )
        ideal_metrics_payload = payload.get("ideal_metrics")
        nadir_metrics_payload = payload.get("nadir_metrics")
        if not isinstance(ideal_metrics_payload, Mapping):
            raise ValueError(
                "Persisted ideal/nadir Pareto normalization payload missing "
                "'ideal_metrics' mapping"
            )
        if not isinstance(nadir_metrics_payload, Mapping):
            raise ValueError(
                "Persisted ideal/nadir Pareto normalization payload missing "
                "'nadir_metrics' mapping"
            )
        ideal_metrics = _coerce_defined_normalization_metrics(
            payload,
            field_name="ideal_metrics",
        )
        nadir_metrics = _coerce_defined_normalization_metrics(
            payload,
            field_name="nadir_metrics",
        )
        _validate_ideal_nadir_directions(ideal_metrics, nadir_metrics)
        return {
            "schema_version": PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
            "kind": PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
            "distance_metric": "euclidean",
            "reference_metrics": resolved_reference_metrics,
            "ideal_metrics": ideal_metrics,
            "nadir_metrics": nadir_metrics,
            "metric_rules": {
                metric_name: dict(rule_payload)
                for metric_name, rule_payload
                in PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES.items()
            },
        }
    raise ValueError(
        f"Unsupported Pareto normalization kind in persisted payload: {kind!r}"
    )


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


def _validate_ideal_nadir_directions(
    ideal_metrics: Mapping[str, float],
    nadir_metrics: Mapping[str, float],
) -> None:
    """Reject direction-inverted ideal/nadir specs.

    For `direction == "max"` the ideal value MUST be strictly greater than
    the nadir; for `direction == "min"` the ideal MUST be strictly less than
    the nadir. Without this check, `objective_metric_scale` would absorb the
    inversion via `abs(...)` while downstream `(value - reference) / scale`
    deltas silently flip sign.
    """
    inversions: list[str] = []
    for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS:
        ideal_value = float(ideal_metrics[metric_name])
        nadir_value = float(nadir_metrics[metric_name])
        if direction == "max":
            if not (ideal_value > nadir_value):
                inversions.append(
                    f"{metric_name} (max-direction): ideal={ideal_value!r} "
                    f"must be > nadir={nadir_value!r}"
                )
        else:
            if not (ideal_value < nadir_value):
                inversions.append(
                    f"{metric_name} (min-direction): ideal={ideal_value!r} "
                    f"must be < nadir={nadir_value!r}"
                )
    if inversions:
        raise ValueError(
            "Direction-inverted Pareto ideal/nadir spec: " + "; ".join(inversions)
        )
