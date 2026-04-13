from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

from .frontier_dominance import (
    DEFAULT_DOMINANCE_TOLERANCE,
    dominates,
    extract_constraint_metrics,
    extract_objective_metrics,
    extract_reference_metrics,
    is_certified_results,
    normalized_objective_distance,
    objective_metric_direction_map,
)

DEFAULT_DUPLICATE_DISTANCE_THRESHOLD = 0.10


@dataclass(frozen=True)
class FrontierArchiveMember:
    member_id: str
    lane_id: str
    campaign_id: str
    archive_state: str
    dominance_signature: dict[str, object]
    objective_metrics: dict[str, float | None]
    reference_metrics: dict[str, float | None]
    constraint_metrics: dict[str, object]
    hard_certification_ok: bool
    soft_search_score: float | None
    distance_from_seed: float | None
    hypervolume_contribution: float | None
    recommendation_flags: dict[str, object]
    rerun_contract: dict[str, object]
    result_source: str
    results_path: str
    termination_reason: str | None
    success: bool

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def build_archive_member_from_results(
    *,
    campaign_id: str,
    lane_id: str,
    payload: Mapping[str, object],
    rerun_contract: Mapping[str, object],
) -> FrontierArchiveMember:
    results = payload["results"]
    objective_metrics = extract_objective_metrics(results)
    reference_metrics = extract_reference_metrics(results)
    distance_from_seed = normalized_objective_distance(
        _defined_metrics(objective_metrics),
        _defined_metrics(reference_metrics),
        reference_metrics=reference_metrics,
    )
    hard_certification_ok = is_certified_results(results)
    soft_search_score = _coerce_optional_float(
        results.get("FRONTIER_RANK_OBJECTIVE_J", results.get("SEARCH_OBJECTIVE_J"))
    )
    member_id = f"{campaign_id}:{lane_id}"
    return FrontierArchiveMember(
        member_id=member_id,
        lane_id=lane_id,
        campaign_id=campaign_id,
        archive_state="certified" if hard_certification_ok else "rejected",
        dominance_signature={},
        objective_metrics=objective_metrics,
        reference_metrics=reference_metrics,
        constraint_metrics=extract_constraint_metrics(results),
        hard_certification_ok=hard_certification_ok,
        soft_search_score=soft_search_score,
        distance_from_seed=distance_from_seed,
        hypervolume_contribution=None,
        recommendation_flags={},
        rerun_contract=dict(rerun_contract),
        result_source=str(payload["result_source"]),
        results_path=str(Path(payload["results_path"])),
        termination_reason=_coerce_optional_str(results.get("TERMINATION_MESSAGE")),
        success=bool(results.get("OPTIMIZER_SUCCESS")),
    )


def update_frontier_archive(
    members: list[FrontierArchiveMember],
    candidate: FrontierArchiveMember,
    *,
    dominance_tolerance: Mapping[str, float] | None = None,
    duplicate_distance_threshold: float = DEFAULT_DUPLICATE_DISTANCE_THRESHOLD,
) -> tuple[list[FrontierArchiveMember], dict[str, object]]:
    tolerances = DEFAULT_DOMINANCE_TOLERANCE if dominance_tolerance is None else dominance_tolerance
    if candidate.archive_state != "certified":
        return members, {
            "action": "rejected",
            "member_id": candidate.member_id,
        }

    dominating_members = [
        member.member_id
        for member in members
        if dominates(member.objective_metrics, candidate.objective_metrics, tolerance=tolerances)
    ]
    if dominating_members:
        return members, {
            "action": "dominated",
            "member_id": candidate.member_id,
            "dominated_by": dominating_members,
        }

    duplicate_index = _find_duplicate_member_index(
        members,
        candidate,
        duplicate_distance_threshold=duplicate_distance_threshold,
    )
    if duplicate_index is not None:
        incumbent = members[duplicate_index]
        if dominates(candidate.objective_metrics, incumbent.objective_metrics, tolerance=tolerances):
            return _insert_candidate_and_drop_dominated(
                members,
                candidate,
                tolerances=tolerances,
                action="duplicate_replaced",
                replaced_member_id=incumbent.member_id,
            )
        winner = _prefer_member(candidate, incumbent)
        if winner.member_id == incumbent.member_id:
            return members, {
                "action": "duplicate_skipped",
                "member_id": candidate.member_id,
                "duplicate_of": incumbent.member_id,
            }
        updated_members = [
            member for member in members if member.member_id != incumbent.member_id
        ]
        updated_members.append(candidate)
        updated_members, archive_update = _insert_candidate_and_drop_dominated(
            updated_members,
            candidate,
            tolerances=tolerances,
            action="duplicate_replaced",
            replaced_member_id=incumbent.member_id,
            candidate_already_present=True,
        )
        return updated_members, archive_update

    return _insert_candidate_and_drop_dominated(
        members,
        candidate,
        tolerances=tolerances,
        action="inserted",
    )


def annotate_archive_members(
    members: list[FrontierArchiveMember],
    *,
    tolerances: Mapping[str, float] | None = None,
) -> list[FrontierArchiveMember]:
    effective_tolerances = DEFAULT_DOMINANCE_TOLERANCE if tolerances is None else tolerances
    annotated_members: list[FrontierArchiveMember] = []
    for member in members:
        dominates_ids = [
            other.member_id
            for other in members
            if other.member_id != member.member_id
            and dominates(member.objective_metrics, other.objective_metrics, tolerance=effective_tolerances)
        ]
        dominated_by_ids = [
            other.member_id
            for other in members
            if other.member_id != member.member_id
            and dominates(other.objective_metrics, member.objective_metrics, tolerance=effective_tolerances)
        ]
        annotated_members.append(
            replace(
                member,
                dominance_signature={
                    "dominates": dominates_ids,
                    "dominated_by": dominated_by_ids,
                },
            )
        )
    return sorted(annotated_members, key=lambda member: member.member_id)


def archive_best_by_metric(members: list[FrontierArchiveMember]) -> dict[str, dict[str, object]]:
    if not members:
        return {}
    best_by_metric: dict[str, dict[str, object]] = {}
    for metric_name, direction in objective_metric_direction_map().items():
        ranked = sorted(
            members,
            key=lambda member: _metric_sort_key(member, metric_name, direction),
        )
        best_member = ranked[0]
        best_by_metric[metric_name] = {
            "member_id": best_member.member_id,
            "value": best_member.objective_metrics.get(metric_name),
        }
    return best_by_metric


def serialize_frontier_archive(
    members: list[FrontierArchiveMember],
    *,
    dominance_tolerance: Mapping[str, float] | None = None,
    duplicate_distance_threshold: float = DEFAULT_DUPLICATE_DISTANCE_THRESHOLD,
) -> dict[str, object]:
    return {
        "dominance_tolerance": dict(
            DEFAULT_DOMINANCE_TOLERANCE if dominance_tolerance is None else dominance_tolerance
        ),
        "duplicate_distance_threshold": float(duplicate_distance_threshold),
        "members": [member.to_json_dict() for member in members],
        "best_by_metric": archive_best_by_metric(members),
    }


def _insert_candidate_and_drop_dominated(
    members: list[FrontierArchiveMember],
    candidate: FrontierArchiveMember,
    *,
    tolerances: Mapping[str, float],
    action: str,
    replaced_member_id: str | None = None,
    candidate_already_present: bool = False,
) -> tuple[list[FrontierArchiveMember], dict[str, object]]:
    dominated_member_ids = [
        member.member_id
        for member in members
        if member.member_id != candidate.member_id
        and dominates(candidate.objective_metrics, member.objective_metrics, tolerance=tolerances)
    ]
    updated_members = [
        member for member in members if member.member_id not in dominated_member_ids
    ]
    if not candidate_already_present:
        updated_members.append(candidate)
    update: dict[str, object] = {
        "action": action,
        "member_id": candidate.member_id,
        "dominated_members": dominated_member_ids,
    }
    if replaced_member_id is not None:
        update["replaced_member_id"] = replaced_member_id
    return annotate_archive_members(updated_members, tolerances=tolerances), update


def _coerce_optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_optional_str(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _defined_metrics(
    metrics: Mapping[str, float | None],
) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if value is not None
    }


def _find_duplicate_member_index(
    members: list[FrontierArchiveMember],
    candidate: FrontierArchiveMember,
    *,
    duplicate_distance_threshold: float,
) -> int | None:
    for index, member in enumerate(members):
        distance = normalized_objective_distance(
            member.objective_metrics,
            candidate.objective_metrics,
            reference_metrics=_shared_reference_metrics(member, candidate),
        )
        if distance is not None and distance <= duplicate_distance_threshold:
            return index
    return None


def _shared_reference_metrics(
    left: FrontierArchiveMember,
    right: FrontierArchiveMember,
) -> dict[str, float | None]:
    reference_metrics: dict[str, float | None] = {}
    for key in left.reference_metrics:
        left_value = left.reference_metrics.get(key)
        reference_metrics[key] = left_value if left_value is not None else right.reference_metrics.get(key)
    return reference_metrics


def _prefer_member(
    candidate: FrontierArchiveMember,
    incumbent: FrontierArchiveMember,
) -> FrontierArchiveMember:
    candidate_key = _member_preference_key(candidate)
    incumbent_key = _member_preference_key(incumbent)
    return candidate if candidate_key < incumbent_key else incumbent


def _member_preference_key(member: FrontierArchiveMember) -> tuple[object, ...]:
    return (
        member.soft_search_score is None,
        float("inf") if member.soft_search_score is None else member.soft_search_score,
        member.result_source != "final",
        member.member_id,
    )


def _metric_sort_key(
    member: FrontierArchiveMember,
    metric_name: str,
    direction: str,
) -> tuple[object, ...]:
    value = member.objective_metrics.get(metric_name)
    if value is None:
        return (True, 0.0, member.member_id)
    signed_value = -float(value) if direction == "max" else float(value)
    return (False, signed_value, member.member_id)
