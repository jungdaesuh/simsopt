from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

from .frontier_archive import FrontierArchiveMember
from .frontier_contracts import SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES
from .frontier_dominance import (
    PARETO_OBJECTIVE_SPECS,
    normalized_objective_distance,
    pareto_scaled_difference,
    resolve_pareto_normalization_reference_metrics,
)

_BALANCED_POLICY_METRICS = tuple(
    (metric_name, direction == "max")
    for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS
)

Direction = Literal["asc", "desc"]
KeyFn = Callable[[FrontierArchiveMember], tuple[object, ...]]
KeyFactory = Callable[[Mapping[str, object] | None], KeyFn]


@dataclass(frozen=True, slots=True)
class FrontierRecommendationGateRule:
    constraint_metric: str
    required_value: bool
    missing_is_eligible: bool
    rationale: str


_BOOZER_TRUST_GATE_RULE = FrontierRecommendationGateRule(
    constraint_metric="frontier_trust_ok",
    required_value=True,
    missing_is_eligible=True,
    rationale=(
        "Boozer-trust gating is permissive for missing values so legacy archive "
        "members without trust metadata are not discarded automatically."
    ),
)
_HARDWARE_SAFE_GATE_RULE = FrontierRecommendationGateRule(
    constraint_metric="hardware_constraints_ok",
    required_value=True,
    missing_is_eligible=False,
    rationale=(
        "Hardware-safe recommendation requires an explicit hard-hardware pass; "
        "missing hardware metadata is treated as unsafe."
    ),
)


def recommend_frontier_member(
    members: list[FrontierArchiveMember],
    *,
    policy_name: str = "balanced",
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    if not members:
        return None
    policy = RECOMMENDATION_POLICIES.get(policy_name)
    if policy is None:
        raise ValueError(f"Unsupported frontier recommendation policy: {policy_name}")

    key_factory, gate_rule, rationale = policy
    selection = select_best(
        members,
        key=key_factory(pareto_objective_normalization),
        gate_rule=gate_rule,
    )
    if selection is None:
        return None
    recommended_member, eligible_members, gate_inputs = selection
    policy_score = _policy_score(
        policy_name,
        recommended_member,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    recommendation_flags = {
        **recommended_member.recommendation_flags,
        policy_name: True,
    }
    if policy_name == "balanced":
        recommendation_flags["balanced_score"] = policy_score
    return {
        "recommended_member": replace(
            recommended_member,
            recommendation_flags=recommendation_flags,
        ),
        "policy_name": policy_name,
        "policy_inputs": _policy_inputs(
            policy_name,
            recommended_member,
            eligible_members,
            gate_inputs,
            pareto_objective_normalization=pareto_objective_normalization,
        ),
        "policy_rationale": rationale,
        "policy_score": policy_score,
        "gate_fallback": False,
    }


def select_best(
    members: Sequence[FrontierArchiveMember],
    *,
    key: KeyFn,
    gate_rule: FrontierRecommendationGateRule | None,
) -> tuple[
    FrontierArchiveMember,
    tuple[FrontierArchiveMember, ...],
    dict[str, object],
] | None:
    eligible_members, gate_inputs = _eligible_members_for_gate_rule(
        members,
        gate_rule=gate_rule,
    )
    if not eligible_members:
        return None
    return min(eligible_members, key=key), eligible_members, gate_inputs


def lex_priority(metrics: Sequence[tuple[str, Direction]]) -> KeyFn:
    def key(member: FrontierArchiveMember) -> tuple[object, ...]:
        return (
            *(
                _maximize_tiebreak_key(member.objective_metrics.get(metric_name))
                if direction == "desc"
                else _minimize_tiebreak_key(member.objective_metrics.get(metric_name))
                for metric_name, direction in metrics
            ),
            member.member_id,
        )

    return key


def scalar_score(
    score_fn: Callable[[FrontierArchiveMember], float],
    *,
    tiebreak_metrics: Sequence[tuple[str, bool]] = (),
) -> KeyFn:
    def key(member: FrontierArchiveMember) -> tuple[object, ...]:
        return (
            -score_fn(member),
            *(
                _maximize_tiebreak_key(member.objective_metrics.get(metric_name))
                if maximize
                else _minimize_tiebreak_key(member.objective_metrics.get(metric_name))
                for metric_name, maximize in tiebreak_metrics
            ),
            member.member_id,
        )

    return key


def _balanced_key(
    pareto_objective_normalization: Mapping[str, object] | None,
) -> KeyFn:
    return scalar_score(
        lambda member: _balanced_policy_score(
            member,
            pareto_objective_normalization=pareto_objective_normalization,
        ),
        tiebreak_metrics=_BALANCED_POLICY_METRICS,
    )


def _closest_to_seed_key(
    pareto_objective_normalization: Mapping[str, object] | None,
) -> KeyFn:
    """Recompute seed-relative distance per call against the active normalization.

    The distance was previously cached on the archive member at insert time;
    that cache went stale whenever the campaign's `pareto_objective_normalization`
    differed at recommendation time. The recommendation key now lives entirely
    in recommendation logic.
    """
    balanced_fallback = scalar_score(
        lambda member: _balanced_policy_score(
            member,
            pareto_objective_normalization=pareto_objective_normalization,
        ),
        tiebreak_metrics=_BALANCED_POLICY_METRICS,
    )

    def key(member: FrontierArchiveMember) -> tuple[object, ...]:
        distance = _closest_to_seed_distance(
            member,
            pareto_objective_normalization=pareto_objective_normalization,
        )
        return (
            distance is None,
            float("inf") if distance is None else float(distance),
            *balanced_fallback(member),
        )

    return key


def _closest_to_seed_distance(
    member: FrontierArchiveMember,
    *,
    pareto_objective_normalization: Mapping[str, object] | None,
) -> float | None:
    """Distance from `member.objective_metrics` to the seed reference under the
    active Pareto normalization. Returns `None` when no reference is available
    or when any objective metric / matched reference value is missing.
    """
    reference_metrics = resolve_pareto_normalization_reference_metrics(
        member.reference_metrics,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    if reference_metrics is None:
        return None
    return normalized_objective_distance(
        member.objective_metrics,
        reference_metrics,
        reference_metrics=reference_metrics,
        pareto_objective_normalization=pareto_objective_normalization,
    )


def _fixed_key(key: KeyFn) -> KeyFactory:
    return lambda _: key


RECOMMENDATION_POLICIES: Mapping[
    str,
    tuple[KeyFactory, FrontierRecommendationGateRule | None, str],
] = MappingProxyType(
    {
        "balanced": (
            _balanced_key,
            None,
            "Select the certified archive member with the best normalized balanced "
            "improvement score relative to the shared seed reference metrics.",
        ),
        "max_iota_under_safe_boozer": (
            _fixed_key(
                lex_priority(
                    (
                        ("iota", "desc"),
                        ("boozer_residual", "asc"),
                        ("volume", "desc"),
                        ("qa_error", "asc"),
                    )
                )
            ),
            _BOOZER_TRUST_GATE_RULE,
            "Select the certified archive member with the highest iota among "
            "members that still satisfy the frontier Boozer trust contract.",
        ),
        "max_volume_under_safe_hardware": (
            _fixed_key(
                lex_priority(
                    (
                        ("volume", "desc"),
                        ("iota", "desc"),
                        ("qa_error", "asc"),
                        ("boozer_residual", "asc"),
                    )
                )
            ),
            _HARDWARE_SAFE_GATE_RULE,
            "Select the certified archive member with the largest volume among "
            "members that still satisfy the hard hardware certification contract.",
        ),
        "closest_to_seed": (
            _closest_to_seed_key,
            None,
            "Select the certified archive member that stays closest to the shared "
            "seed reference in normalized Pareto-objective space.",
        ),
    }
)

if tuple(RECOMMENDATION_POLICIES) != SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES:
    raise ValueError("Frontier recommendation policy registry drifted from contract")


def _policy_score(
    policy_name: str,
    member: FrontierArchiveMember,
    *,
    pareto_objective_normalization: Mapping[str, object] | None,
) -> object:
    if policy_name == "balanced":
        return _balanced_policy_score(
            member,
            pareto_objective_normalization=pareto_objective_normalization,
        )
    if policy_name == "max_iota_under_safe_boozer":
        return member.objective_metrics.get("iota")
    if policy_name == "max_volume_under_safe_hardware":
        return member.objective_metrics.get("volume")
    distance = _closest_to_seed_distance(
        member,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    return None if distance is None else -float(distance)


def _policy_inputs(
    policy_name: str,
    recommended_member: FrontierArchiveMember,
    eligible_members: tuple[FrontierArchiveMember, ...],
    gate_inputs: dict[str, object],
    *,
    pareto_objective_normalization: Mapping[str, object] | None,
) -> dict[str, object]:
    if policy_name == "balanced":
        return {
            "objective_metrics": dict(recommended_member.objective_metrics),
            "reference_metrics": dict(recommended_member.reference_metrics),
            "pareto_objective_normalization_kind": (
                None
                if pareto_objective_normalization is None
                else pareto_objective_normalization.get("kind")
            ),
        }
    if policy_name == "closest_to_seed":
        return {"distance_metric": "normalized_objective_distance"}
    return {
        "eligible_member_ids": [member.member_id for member in eligible_members],
        **gate_inputs,
    }


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


def _normalized_delta(
    member: FrontierArchiveMember,
    metric_name: str,
    *,
    maximize: bool,
    pareto_objective_normalization: Mapping[str, object] | None = None,
) -> float:
    """Normalized improvement of `member.objective_metrics[metric_name]` against
    the active reference. Treats both inputs as upstream-certified invariants:
    certified archive members must have all four `objective_metrics` populated
    (`is_certified_results` enforces `objective_metrics_complete`), and the
    active normalization must publish a reference value for each metric. Any
    missing input is a contract break, not a "no data, treat as at-reference"
    fallback (F4.2).
    """
    value = member.objective_metrics.get(metric_name)
    reference_metrics = resolve_pareto_normalization_reference_metrics(
        member.reference_metrics,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    reference = None if reference_metrics is None else reference_metrics.get(metric_name)
    if value is None or reference is None:
        raise ValueError(
            "_normalized_delta requires complete metric and reference data; "
            f"got value={value!r}, reference={reference!r} for {metric_name!r}"
        )
    delta = pareto_scaled_difference(
        metric_name,
        value,
        reference,
        scale_reference=reference,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    return delta if maximize else -delta


def _maximize_tiebreak_key(value: float | None) -> tuple[bool, float]:
    if value is None:
        return (True, 0.0)
    return (False, -float(value))


def _minimize_tiebreak_key(value: float | None) -> tuple[bool, float]:
    if value is None:
        return (True, 0.0)
    return (False, float(value))


def _eligible_members_for_gate_rule(
    members: Sequence[FrontierArchiveMember],
    *,
    gate_rule: FrontierRecommendationGateRule | None,
) -> tuple[tuple[FrontierArchiveMember, ...], dict[str, object]]:
    if gate_rule is None:
        return tuple(members), {
            "gate_constraint_metric": None,
            "gate_missing_is_eligible": None,
        }
    eligible_members = tuple(
        member for member in members if _member_satisfies_gate_rule(member, gate_rule)
    )
    return eligible_members, {
        "gate_constraint_metric": gate_rule.constraint_metric,
        "gate_missing_is_eligible": gate_rule.missing_is_eligible,
        "gate_required_value": gate_rule.required_value,
        "gate_rationale": gate_rule.rationale,
    }


def _member_satisfies_gate_rule(
    member: FrontierArchiveMember,
    gate_rule: FrontierRecommendationGateRule,
) -> bool:
    value = member.constraint_metrics.get(gate_rule.constraint_metric)
    if value is None:
        return gate_rule.missing_is_eligible
    return bool(value) == gate_rule.required_value
