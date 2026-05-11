from __future__ import annotations

import math
from typing import Mapping

import numpy as np

# Diagnostic max/min-ratio gate on raw objective magnitudes. This is not a
# numerical preconditioner and does not rescale optimizer variables or gradients.
FRONTIER_CONDITIONING_SCHEMA_VERSION = "frontier_conditioning_v1"
FRONTIER_CONDITIONING_MAX_RATIO = 1.0e3

# Names of the four core conditioning terms whose magnitudes participate in the
# `usable_scale_ok` ratio gate. Auxiliary penalties (`trust_penalty`,
# `epsilon_penalty`) are reported alongside but are not part of the invariant.
_CORE_CONDITIONING_TERM_NAMES = (
    "qa_objective",
    "boozer_objective",
    "iota_objective",
    "volume_objective",
)


def build_frontier_conditioning_report(
    objective_eval: Mapping[str, object],
    *,
    sample_label: str,
    effective_iotas_weight: float,
    effective_volume_weight: float,
    effective_res_weight: float,
) -> dict[str, object]:
    """Build a conditioning diagnostic at the optimizer-total contribution scale.

    The four core terms (``qa_objective``, ``boozer_objective``, ``iota_objective``,
    ``volume_objective``) are reported as ``effective_weight * |J_term|`` so that
    the bounded ``J_iota = -tanh(...) ∈ (-1, 1)`` and the unbounded
    ``J_QS_objective = J_QS / qs_reference`` are compared at the same scale —
    namely each term's actual contribution to the optimizer's scalar total. The
    ``qa_objective`` weight is the goal config's ``effective_qs_weight`` (always
    ``1.0`` in the frontier goal); we therefore do not pass it as an argument and
    leave ``J_QS_objective`` unscaled.

    Auxiliary penalties (``trust_penalty``, ``epsilon_penalty``) are included in
    ``term_values`` for visibility but are excluded from the ratio gate.

    Any core term that is missing, non-finite, or non-positive is recorded in
    ``incomplete_value_terms`` / ``incomplete_grad_terms``; in that case the
    corresponding ratio is ``None`` and ``usable_scale_ok`` is ``False``.
    """
    iotas_weight = float(effective_iotas_weight)
    volume_weight = float(effective_volume_weight)
    res_weight = float(effective_res_weight)
    term_values = {
        "qa_objective": _finite_abs(objective_eval.get("J_QS_objective")),
        "boozer_objective": _scaled_finite_abs(
            objective_eval.get("J_Boozer_objective"), res_weight
        ),
        "iota_objective": _scaled_finite_abs(
            objective_eval.get("J_iota"), iotas_weight
        ),
        "volume_objective": _scaled_finite_abs(
            objective_eval.get("J_volume"), volume_weight
        ),
        "trust_penalty": _finite_abs(objective_eval.get("frontier_trust_penalty")),
        "epsilon_penalty": _finite_abs(objective_eval.get("frontier_epsilon_penalty")),
    }
    grad_norms = {
        "qa_objective": _finite_norm(objective_eval.get("dJ_QS_objective")),
        "boozer_objective": _scaled_finite_norm(
            objective_eval.get("dJ_Boozer_objective"), res_weight
        ),
        "iota_objective": _scaled_finite_norm(
            objective_eval.get("dJ_iota"), iotas_weight
        ),
        "volume_objective": _scaled_finite_norm(
            objective_eval.get("dJ_volume"), volume_weight
        ),
    }
    core_value_terms = {name: term_values[name] for name in _CORE_CONDITIONING_TERM_NAMES}
    core_grad_terms = {name: grad_norms[name] for name in _CORE_CONDITIONING_TERM_NAMES}
    value_ratio, incomplete_value_terms = _usable_ratio_or_none(core_value_terms)
    grad_ratio, incomplete_grad_terms = _usable_ratio_or_none(core_grad_terms)
    usable_scale_ok = (
        not incomplete_value_terms
        and not incomplete_grad_terms
        and value_ratio is not None
        and grad_ratio is not None
        and value_ratio <= FRONTIER_CONDITIONING_MAX_RATIO
        and grad_ratio <= FRONTIER_CONDITIONING_MAX_RATIO
    )
    return {
        "schema_version": FRONTIER_CONDITIONING_SCHEMA_VERSION,
        "sample_label": sample_label,
        "term_values": term_values,
        "grad_norms": grad_norms,
        "max_value_ratio": value_ratio,
        "max_grad_ratio": grad_ratio,
        "incomplete_value_terms": incomplete_value_terms,
        "incomplete_grad_terms": incomplete_grad_terms,
        "usable_scale_ok": usable_scale_ok,
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


def _scaled_finite_abs(value, weight: float) -> float | None:
    magnitude = _finite_abs(value)
    if magnitude is None:
        return None
    if not math.isfinite(weight):
        return None
    return abs(weight) * magnitude


def _finite_norm(value) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        return None
    return float(np.linalg.norm(array))


def _scaled_finite_norm(value, weight: float) -> float | None:
    magnitude = _finite_norm(value)
    if magnitude is None:
        return None
    if not math.isfinite(weight):
        return None
    return abs(weight) * magnitude


def _usable_ratio_or_none(
    named_values: Mapping[str, float | None],
) -> tuple[float | None, list[str]]:
    incomplete = [
        name
        for name, value in named_values.items()
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0
    ]
    if incomplete:
        return None, incomplete
    finite_positive = [float(value) for value in named_values.values()]
    return float(max(finite_positive) / min(finite_positive)), []
