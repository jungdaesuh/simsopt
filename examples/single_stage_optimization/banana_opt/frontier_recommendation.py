from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .frontier_archive import FrontierArchiveMember
from .frontier_contracts import SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES
from .frontier_dominance import (
    objective_metric_scale,
    resolve_pareto_normalization_reference_metrics,
)

_BALANCED_POLICY_METRICS = (
    ("iota", True),
    ("volume", True),
    ("qa_error", False),
    ("boozer_residual", False),
)


@dataclass(frozen=True)
class FrontierRecommendationGateRule:
    constraint_metric: str
    required_value: bool
    missing_is_eligible: bool
    rationale: str


_POLICY_GATE_RULES: dict[str, FrontierRecommendationGateRule | None] = {
    "balanced": None,
    "closest_to_seed": None,
    "max_iota_under_safe_boozer": FrontierRecommendationGateRule(
        constraint_metric="frontier_trust_ok",
        required_value=True,
        missing_is_eligible=True,
        rationale=(
            "Boozer-trust gating is permissive for missing values so legacy archive "
            "members without trust metadata are not discarded automatically."
        ),
    ),
    "max_volume_under_safe_hardware": FrontierRecommendationGateRule(
        constraint_metric="hardware_constraints_ok",
        required_value=True,
        missing_is_eligible=False,
        rationale=(
            "Hardware-safe recommendation requires an explicit hard-hardware pass; "
            "missing hardware metadata is treated as unsafe."
        ),
    ),
}


def recommend_frontier_member(
    members: list[FrontierArchiveMember],
    *,
    policy_name: str = "balanced",
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    if not members:
        return None
    if policy_name not in SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES:
        raise ValueError(f"Unsupported frontier recommendation policy: {policy_name}")
    if policy_name == "balanced":
        return _recommend_balanced(
            members,
            pareto_objective_normalization=pareto_objective_normalization,
        )
    if policy_name == "max_iota_under_safe_boozer":
        return _recommend_max_iota_under_safe_boozer(members)
    if policy_name == "max_volume_under_safe_hardware":
        return _recommend_max_volume_under_safe_hardware(members)
    return _recommend_closest_to_seed(members)


def _recommend_balanced(
    members: list[FrontierArchiveMember],
    *,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ranked_members = sorted(
        members,
        key=lambda member: _balanced_policy_sort_key(
            member,
            pareto_objective_normalization=pareto_objective_normalization,
        ),
    )
    recommended_member = ranked_members[0]
    policy_score = _balanced_policy_score(
        recommended_member,
        pareto_objective_normalization=pareto_objective_normalization,
    )
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
            "pareto_objective_normalization_kind": (
                None
                if pareto_objective_normalization is None
                else pareto_objective_normalization.get("kind")
            ),
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
    eligible_members, gate_inputs = _eligible_members_for_policy(
        members,
        policy_name="max_iota_under_safe_boozer",
    )
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
            **gate_inputs,
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
    eligible_members, gate_inputs = _eligible_members_for_policy(
        members,
        policy_name="max_volume_under_safe_hardware",
    )
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
            **gate_inputs,
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

def _balanced_policy_sort_key(
    member: FrontierArchiveMember,
    *,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> tuple[object, ...]:
    metric_tiebreak_keys = []
    for metric_name, maximize in _BALANCED_POLICY_METRICS:
        value = member.objective_metrics.get(metric_name)
        metric_tiebreak_keys.append(
            _maximize_tiebreak_key(value) if maximize else _minimize_tiebreak_key(value)
        )
    return (
        -_balanced_policy_score(
            member,
            pareto_objective_normalization=pareto_objective_normalization,
        ),
        *metric_tiebreak_keys,
        member.member_id,
    )


def _balanced_policy_score(
    member: FrontierArchiveMember,
    *,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> float:
    return sum(
        _normalized_delta(
            member,
            metric_name,
            maximize=maximize,
            pareto_objective_normalization=pareto_objective_normalization,
        )
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
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> float:
    value = member.objective_metrics.get(metric_name)
    reference_metrics = resolve_pareto_normalization_reference_metrics(
        member.reference_metrics,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    reference = None if reference_metrics is None else reference_metrics.get(metric_name)
    if value is None or reference is None:
        return 0.0
    scale = objective_metric_scale(
        metric_name,
        reference,
        pareto_objective_normalization=pareto_objective_normalization,
    )
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


def _eligible_members_for_policy(
    members: list[FrontierArchiveMember],
    *,
    policy_name: str,
) -> tuple[list[FrontierArchiveMember], dict[str, object]]:
    gate_rule = _POLICY_GATE_RULES[policy_name]
    if gate_rule is None:
        return list(members), {
            "gate_constraint_metric": None,
            "gate_missing_is_eligible": None,
            "gate_fallback_to_all_members": False,
        }
    eligible_members = [
        member for member in members if _member_satisfies_gate_rule(member, gate_rule)
    ]
    used_fallback = False
    if not eligible_members:
        eligible_members = list(members)
        used_fallback = True
    return eligible_members, {
        "gate_constraint_metric": gate_rule.constraint_metric,
        "gate_missing_is_eligible": gate_rule.missing_is_eligible,
        "gate_required_value": gate_rule.required_value,
        "gate_rationale": gate_rule.rationale,
        "gate_fallback_to_all_members": used_fallback,
    }


def _member_satisfies_gate_rule(
    member: FrontierArchiveMember,
    gate_rule: FrontierRecommendationGateRule,
) -> bool:
    value = member.constraint_metrics.get(gate_rule.constraint_metric)
    if value is None:
        return gate_rule.missing_is_eligible
    return bool(value) is gate_rule.required_value
