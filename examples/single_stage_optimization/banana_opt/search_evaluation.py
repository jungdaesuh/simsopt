from __future__ import annotations

from typing import Mapping

import numpy as np

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
    "J_curvature",
    "max_violation",
    "max_feasibility_violation",
    "stationarity_norm",
    "metric_stationarity_norm",
    "frontier_rank_total",
    "frontier_base_total",
    "frontier_trust_penalty",
    "frontier_epsilon_penalty",
    "frontier_goal_total",
    "frontier_scalarization_total",
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
    "dJ_curvature",
    "surface_weights",
    "constraint_values",
    "dual_update_values",
    "feasibility_values",
    "constraint_activity_tolerances",
    "frontier_goal_grad",
    "frontier_scalarization_grad",
)
_FINITE_VECTOR_LIST_FIELDS = ("constraint_grads",)
FINITE_EPS = float(np.finfo(float).eps)


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


def _finite_array(value) -> bool:
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
