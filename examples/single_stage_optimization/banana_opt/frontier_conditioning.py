from __future__ import annotations

import math
from typing import Mapping

import numpy as np

FRONTIER_CONDITIONING_SCHEMA_VERSION = "frontier_conditioning_v1"
FRONTIER_CONDITIONING_MAX_RATIO = 1.0e3


def build_frontier_conditioning_report(
    objective_eval: Mapping[str, object],
    *,
    sample_label: str,
) -> dict[str, object]:
    term_values = {
        "qa_objective": _finite_abs(objective_eval.get("J_QS_objective")),
        "boozer_objective": _finite_abs(objective_eval.get("J_Boozer_objective")),
        "iota_objective": _finite_abs(objective_eval.get("J_iota")),
        "volume_objective": _finite_abs(objective_eval.get("J_volume")),
        "trust_penalty": _finite_abs(objective_eval.get("frontier_trust_penalty")),
        "epsilon_penalty": _finite_abs(objective_eval.get("frontier_epsilon_penalty")),
    }
    grad_norms = {
        "qa_objective": _finite_norm(objective_eval.get("dJ_QS_objective")),
        "boozer_objective": _finite_norm(objective_eval.get("dJ_Boozer_objective")),
        "iota_objective": _finite_norm(objective_eval.get("dJ_iota")),
        "volume_objective": _finite_norm(objective_eval.get("dJ_volume")),
    }
    value_ratio = _usable_ratio(term_values.values())
    grad_ratio = _usable_ratio(grad_norms.values())
    return {
        "schema_version": FRONTIER_CONDITIONING_SCHEMA_VERSION,
        "sample_label": sample_label,
        "term_values": term_values,
        "grad_norms": grad_norms,
        "max_value_ratio": value_ratio,
        "max_grad_ratio": grad_ratio,
        "usable_scale_ok": (
            value_ratio is not None
            and grad_ratio is not None
            and value_ratio <= FRONTIER_CONDITIONING_MAX_RATIO
            and grad_ratio <= FRONTIER_CONDITIONING_MAX_RATIO
        ),
    }


def build_frontier_conditioning_gate(
    *,
    seed_report: Mapping[str, object] | None,
    first_accepted_report: Mapping[str, object] | None,
) -> dict[str, object]:
    samples = {
        "seed": None if seed_report is None else dict(seed_report),
        "first_accepted": (
            None if first_accepted_report is None else dict(first_accepted_report)
        ),
    }
    sample_ok = {
        sample_name: (
            None
            if report is None
            else bool(report.get("usable_scale_ok"))
        )
        for sample_name, report in samples.items()
    }
    usable_scale_ok = (
        seed_report is not None
        and first_accepted_report is not None
        and bool(seed_report.get("usable_scale_ok"))
        and bool(first_accepted_report.get("usable_scale_ok"))
    )
    return {
        "schema_version": FRONTIER_CONDITIONING_SCHEMA_VERSION,
        "max_ratio_limit": FRONTIER_CONDITIONING_MAX_RATIO,
        "sample_reports": samples,
        "sample_ok": sample_ok,
        "usable_scale_ok": usable_scale_ok,
    }


def _finite_abs(value) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return abs(numeric)


def _finite_norm(value) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        return None
    return float(np.linalg.norm(array))


def _usable_ratio(values) -> float | None:
    finite_positive = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not finite_positive:
        return None
    return float(max(finite_positive) / min(finite_positive))
