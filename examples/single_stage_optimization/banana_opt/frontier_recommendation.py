from __future__ import annotations

from dataclasses import replace

from .frontier_archive import FrontierArchiveMember
from .frontier_dominance import objective_metric_scale

_BALANCED_POLICY_METRICS = (
    ("iota", True),
    ("volume", True),
    ("qa_error", False),
    ("boozer_residual", False),
)


def recommend_frontier_member(
    members: list[FrontierArchiveMember],
    *,
    policy_name: str = "balanced",
) -> dict[str, object] | None:
    if not members:
        return None
    if policy_name != "balanced":
        raise ValueError(f"Unsupported frontier recommendation policy: {policy_name}")

    ranked_members = sorted(
        members,
        key=lambda member: _balanced_policy_sort_key(member),
    )
    recommended_member = ranked_members[0]
    policy_score = _balanced_policy_score(recommended_member)
    return {
        "recommended_member": replace(
            recommended_member,
            recommendation_flags={
                **recommended_member.recommendation_flags,
                "balanced": True,
                "balanced_score": policy_score,
            },
        ),
        "policy_name": policy_name,
        "policy_inputs": {
            "objective_metrics": dict(recommended_member.objective_metrics),
            "reference_metrics": dict(recommended_member.reference_metrics),
        },
        "policy_rationale": (
            "Select the certified archive member with the best normalized balanced "
            "improvement score relative to the shared seed reference metrics."
        ),
        "policy_score": policy_score,
    }


def _balanced_policy_sort_key(member: FrontierArchiveMember) -> tuple[object, ...]:
    metric_tiebreak_keys = []
    for metric_name, maximize in _BALANCED_POLICY_METRICS:
        value = member.objective_metrics.get(metric_name)
        metric_tiebreak_keys.append(
            _maximize_tiebreak_key(value) if maximize else _minimize_tiebreak_key(value)
        )
    return (
        -_balanced_policy_score(member),
        *metric_tiebreak_keys,
        member.member_id,
    )


def _balanced_policy_score(member: FrontierArchiveMember) -> float:
    return sum(
        _normalized_delta(member, metric_name, maximize=maximize)
        for metric_name, maximize in _BALANCED_POLICY_METRICS
    )


def _normalized_delta(
    member: FrontierArchiveMember,
    metric_name: str,
    *,
    maximize: bool,
) -> float:
    value = member.objective_metrics.get(metric_name)
    reference = member.reference_metrics.get(metric_name)
    if value is None or reference is None:
        return 0.0
    scale = objective_metric_scale(metric_name, reference)
    delta = (float(value) - float(reference)) / scale
    return delta if maximize else -delta


def _maximize_tiebreak_key(value: float | None) -> tuple[bool, float]:
    if value is None:
        return (True, 0.0)
    return (False, -float(value))


def _minimize_tiebreak_key(value: float | None) -> tuple[bool, float]:
    if value is None:
        return (True, 0.0)
    return (False, float(value))
