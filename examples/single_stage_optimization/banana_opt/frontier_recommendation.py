from __future__ import annotations

from dataclasses import replace

from .frontier_archive import FrontierArchiveMember
from .frontier_contracts import SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES
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
    if policy_name not in SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES:
        raise ValueError(f"Unsupported frontier recommendation policy: {policy_name}")
    if policy_name == "balanced":
        return _recommend_balanced(members)
    if policy_name == "max_iota_under_safe_boozer":
        return _recommend_max_iota_under_safe_boozer(members)
    if policy_name == "max_volume_under_safe_hardware":
        return _recommend_max_volume_under_safe_hardware(members)
    return _recommend_closest_to_seed(members)


def _recommend_balanced(
    members: list[FrontierArchiveMember],
) -> dict[str, object]:
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
        "policy_name": "balanced",
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


def _recommend_max_iota_under_safe_boozer(
    members: list[FrontierArchiveMember],
) -> dict[str, object]:
    eligible_members = [
        member
        for member in members
        if member.constraint_metrics.get("frontier_trust_ok") is not False
    ]
    if not eligible_members:
        eligible_members = list(members)
    ranked_members = sorted(eligible_members, key=_max_iota_under_safe_boozer_sort_key)
    recommended_member = ranked_members[0]
    return {
        "recommended_member": replace(
            recommended_member,
            recommendation_flags={
                **recommended_member.recommendation_flags,
                "max_iota_under_safe_boozer": True,
            },
        ),
        "policy_name": "max_iota_under_safe_boozer",
        "policy_inputs": {
            "eligible_member_ids": [member.member_id for member in eligible_members],
        },
        "policy_rationale": (
            "Select the certified archive member with the highest iota among "
            "members that still satisfy the frontier Boozer trust contract."
        ),
        "policy_score": recommended_member.objective_metrics.get("iota"),
    }


def _recommend_max_volume_under_safe_hardware(
    members: list[FrontierArchiveMember],
) -> dict[str, object]:
    eligible_members = [
        member
        for member in members
        if bool(member.constraint_metrics.get("hardware_constraints_ok", False))
    ]
    if not eligible_members:
        eligible_members = list(members)
    ranked_members = sorted(
        eligible_members,
        key=_max_volume_under_safe_hardware_sort_key,
    )
    recommended_member = ranked_members[0]
    return {
        "recommended_member": replace(
            recommended_member,
            recommendation_flags={
                **recommended_member.recommendation_flags,
                "max_volume_under_safe_hardware": True,
            },
        ),
        "policy_name": "max_volume_under_safe_hardware",
        "policy_inputs": {
            "eligible_member_ids": [member.member_id for member in eligible_members],
        },
        "policy_rationale": (
            "Select the certified archive member with the largest volume among "
            "members that still satisfy the hard hardware certification contract."
        ),
        "policy_score": recommended_member.objective_metrics.get("volume"),
    }


def _recommend_closest_to_seed(
    members: list[FrontierArchiveMember],
) -> dict[str, object]:
    ranked_members = sorted(members, key=_closest_to_seed_sort_key)
    recommended_member = ranked_members[0]
    return {
        "recommended_member": replace(
            recommended_member,
            recommendation_flags={
                **recommended_member.recommendation_flags,
                "closest_to_seed": True,
            },
        ),
        "policy_name": "closest_to_seed",
        "policy_inputs": {
            "distance_metric": "normalized_objective_distance",
        },
        "policy_rationale": (
            "Select the certified archive member that stays closest to the shared "
            "seed reference in normalized Pareto-objective space."
        ),
        "policy_score": (
            None
            if recommended_member.distance_from_seed is None
            else -float(recommended_member.distance_from_seed)
        ),
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


def _max_iota_under_safe_boozer_sort_key(
    member: FrontierArchiveMember,
) -> tuple[object, ...]:
    return (
        _maximize_tiebreak_key(member.objective_metrics.get("iota")),
        _minimize_tiebreak_key(member.objective_metrics.get("boozer_residual")),
        _maximize_tiebreak_key(member.objective_metrics.get("volume")),
        _minimize_tiebreak_key(member.objective_metrics.get("qa_error")),
        member.member_id,
    )


def _max_volume_under_safe_hardware_sort_key(
    member: FrontierArchiveMember,
) -> tuple[object, ...]:
    return (
        _maximize_tiebreak_key(member.objective_metrics.get("volume")),
        _maximize_tiebreak_key(member.objective_metrics.get("iota")),
        _minimize_tiebreak_key(member.objective_metrics.get("qa_error")),
        _minimize_tiebreak_key(member.objective_metrics.get("boozer_residual")),
        member.member_id,
    )


def _closest_to_seed_sort_key(
    member: FrontierArchiveMember,
) -> tuple[object, ...]:
    return (
        member.distance_from_seed is None,
        float("inf") if member.distance_from_seed is None else float(member.distance_from_seed),
        -_balanced_policy_score(member),
        member.member_id,
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
