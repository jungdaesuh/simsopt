from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

from .frontier_contracts import (
    FRONTIER_ARCHIVE_SCHEMA_VERSION,
    FRONTIER_ARCHIVE_STATE_CERTIFIED,
    FRONTIER_ARCHIVE_STATE_PROVISIONAL,
    FRONTIER_ARCHIVE_STATE_REJECTED,
    frontier_archive_membership_rules_contract,
    frontier_archive_state_semantics_contract,
    pareto_objective_vector_contract,
    validate_frontier_archive_payload,
)
from .frontier_dominance import (
    DEFAULT_DOMINANCE_TOLERANCE,
    PARETO_OBJECTIVE_SPECS,
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

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> FrontierArchiveMember:
        return frontier_archive_member_from_json_dict(payload)


def build_archive_member_from_results(
    *,
    campaign_id: str,
    lane_id: str,
    payload: Mapping[str, object],
    rerun_contract: Mapping[str, object],
    archive_state: str | None = None,
) -> FrontierArchiveMember:
    results = payload["results"]
    objective_metrics = extract_objective_metrics(results)
    reference_metrics = extract_reference_metrics(results)
    distance_from_seed = normalized_objective_distance(
        _defined_metrics(objective_metrics),
        _defined_metrics(reference_metrics),
        reference_metrics=reference_metrics,
    )
    epsilon_constraint_status = _evaluate_epsilon_constraint_status(
        objective_metrics,
        rerun_contract,
    )
    hard_certification_ok = (
        is_certified_results(results)
        and epsilon_constraint_status["ok"]
    )
    soft_search_score = _coerce_optional_float(
        results.get("FRONTIER_RANK_OBJECTIVE_J", results.get("SEARCH_OBJECTIVE_J"))
    )
    resolved_archive_state = _resolve_archive_state(
        archive_state,
        hard_certification_ok=hard_certification_ok,
    )
    member_id = _build_member_id(
        campaign_id,
        lane_id,
        archive_state=resolved_archive_state,
    )
    return FrontierArchiveMember(
        member_id=member_id,
        lane_id=lane_id,
        campaign_id=campaign_id,
        archive_state=resolved_archive_state,
        dominance_signature={},
        objective_metrics=objective_metrics,
        reference_metrics=reference_metrics,
        constraint_metrics={
            **extract_constraint_metrics(results),
            "epsilon_constraints_ok": epsilon_constraint_status["ok"],
            "epsilon_constraint_violations": epsilon_constraint_status["violations"],
        },
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


def finalize_archive_member(
    member: FrontierArchiveMember,
) -> FrontierArchiveMember:
    final_archive_state = (
        FRONTIER_ARCHIVE_STATE_CERTIFIED
        if member.hard_certification_ok
        else FRONTIER_ARCHIVE_STATE_REJECTED
    )
    return replace(
        member,
        member_id=_build_member_id(
            member.campaign_id,
            member.lane_id,
            archive_state=final_archive_state,
        ),
        archive_state=final_archive_state,
    )


def archive_members_in_state(
    members: list[FrontierArchiveMember],
    archive_state: str,
) -> list[FrontierArchiveMember]:
    return [
        member for member in members if member.archive_state == archive_state
    ]


def certified_archive_members(
    members: list[FrontierArchiveMember],
) -> list[FrontierArchiveMember]:
    return archive_members_in_state(
        members,
        FRONTIER_ARCHIVE_STATE_CERTIFIED,
    )


def frontier_archive_member_from_json_dict(
    payload: Mapping[str, object],
) -> FrontierArchiveMember:
    objective_metrics_payload = payload.get("objective_metrics", {})
    reference_metrics_payload = payload.get("reference_metrics", {})
    return FrontierArchiveMember(
        member_id=str(payload["member_id"]),
        lane_id=str(payload["lane_id"]),
        campaign_id=str(payload["campaign_id"]),
        archive_state=str(payload["archive_state"]),
        dominance_signature=dict(payload.get("dominance_signature", {})),
        objective_metrics={
            str(key): None if value is None else float(value)
            for key, value in objective_metrics_payload.items()
        },
        reference_metrics={
            str(key): None if value is None else float(value)
            for key, value in reference_metrics_payload.items()
        },
        constraint_metrics=dict(payload.get("constraint_metrics", {})),
        hard_certification_ok=bool(payload.get("hard_certification_ok", False)),
        soft_search_score=_coerce_optional_float(payload.get("soft_search_score")),
        distance_from_seed=_coerce_optional_float(payload.get("distance_from_seed")),
        hypervolume_contribution=_coerce_optional_float(
            payload.get("hypervolume_contribution")
        ),
        recommendation_flags=dict(payload.get("recommendation_flags", {})),
        rerun_contract=dict(payload.get("rerun_contract", {})),
        result_source=str(payload.get("result_source", "final")),
        results_path=str(payload.get("results_path", "")),
        termination_reason=_coerce_optional_str(payload.get("termination_reason")),
        success=bool(payload.get("success", False)),
    )


def update_frontier_archive(
    members: list[FrontierArchiveMember],
    candidate: FrontierArchiveMember,
    *,
    dominance_tolerance: Mapping[str, float] | None = None,
    duplicate_distance_threshold: float = DEFAULT_DUPLICATE_DISTANCE_THRESHOLD,
) -> tuple[list[FrontierArchiveMember], dict[str, object]]:
    tolerances = DEFAULT_DOMINANCE_TOLERANCE if dominance_tolerance is None else dominance_tolerance
    if candidate.archive_state != FRONTIER_ARCHIVE_STATE_CERTIFIED:
        return members, {
            "action": FRONTIER_ARCHIVE_STATE_REJECTED,
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
    hypervolume_reference: Mapping[str, float] | None = None,
) -> dict[str, object]:
    reference_metrics = (
        None
        if hypervolume_reference is None
        else {str(key): float(value) for key, value in hypervolume_reference.items()}
    )
    annotated_members = annotate_hypervolume_contributions(
        members,
        hypervolume_reference=reference_metrics,
    )
    payload = {
        "schema_version": FRONTIER_ARCHIVE_SCHEMA_VERSION,
        "pareto_objective_vector": pareto_objective_vector_contract(),
        "archive_membership_rules": frontier_archive_membership_rules_contract(),
        "archive_state_semantics": frontier_archive_state_semantics_contract(),
        "dominance_tolerance": dict(
            DEFAULT_DOMINANCE_TOLERANCE if dominance_tolerance is None else dominance_tolerance
        ),
        "duplicate_distance_threshold": float(duplicate_distance_threshold),
        "hypervolume_reference": reference_metrics,
        "hypervolume_total": frontier_archive_hypervolume(
            annotated_members,
            hypervolume_reference=reference_metrics,
        ),
        "members": [member.to_json_dict() for member in annotated_members],
        "best_by_metric": archive_best_by_metric(annotated_members),
    }
    validate_frontier_archive_payload(payload)
    return payload


def parse_hypervolume_reference(
    reference_spec: str | None,
) -> dict[str, float] | None:
    if reference_spec is None:
        return None
    stripped = str(reference_spec).strip()
    if not stripped:
        return None
    metric_names = [metric_name for metric_name, _, _ in PARETO_OBJECTIVE_SPECS]
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise ValueError("hypervolume reference JSON must be an object")
        return {
            metric_name: float(payload[metric_name])
            for metric_name in metric_names
        }
    if "=" in stripped:
        values: dict[str, float] = {}
        for entry in stripped.split(","):
            key, _, raw_value = entry.partition("=")
            values[str(key).strip()] = float(raw_value)
        return {
            metric_name: float(values[metric_name])
            for metric_name in metric_names
        }
    numeric_values = [float(item.strip()) for item in stripped.split(",") if item.strip()]
    if len(numeric_values) != len(metric_names):
        raise ValueError(
            "hypervolume reference must provide iota,volume,qa_error,boozer_residual"
        )
    return {
        metric_name: numeric_values[index]
        for index, metric_name in enumerate(metric_names)
    }


def resolve_hypervolume_reference(
    *,
    reference_spec: str | None,
    seed_results: Mapping[str, object] | None = None,
    members: list[FrontierArchiveMember] | None = None,
) -> dict[str, float] | None:
    parsed_reference = parse_hypervolume_reference(reference_spec)
    if parsed_reference is not None:
        return parsed_reference
    if seed_results is not None:
        seed_metrics = extract_objective_metrics(seed_results)
        if all(seed_metrics.get(metric_name) is not None for metric_name, _, _ in PARETO_OBJECTIVE_SPECS):
            return {
                metric_name: float(seed_metrics[metric_name])
                for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
            }
    if members:
        for member in members:
            if all(
                member.reference_metrics.get(metric_name) is not None
                for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
            ):
                return {
                    metric_name: float(member.reference_metrics[metric_name])
                    for metric_name, _, _ in PARETO_OBJECTIVE_SPECS
                }
    return None


def frontier_archive_hypervolume(
    members: list[FrontierArchiveMember],
    *,
    hypervolume_reference: Mapping[str, float] | None,
) -> float | None:
    if hypervolume_reference is None:
        return None
    boxes = _hypervolume_boxes(
        members,
        hypervolume_reference=hypervolume_reference,
    )
    return _union_hypervolume(boxes)


def annotate_hypervolume_contributions(
    members: list[FrontierArchiveMember],
    *,
    hypervolume_reference: Mapping[str, float] | None,
) -> list[FrontierArchiveMember]:
    if hypervolume_reference is None:
        return [replace(member, hypervolume_contribution=None) for member in members]
    total_hypervolume = frontier_archive_hypervolume(
        members,
        hypervolume_reference=hypervolume_reference,
    )
    if total_hypervolume is None:
        return [replace(member, hypervolume_contribution=None) for member in members]
    annotated_members: list[FrontierArchiveMember] = []
    for member in members:
        reduced_members = [
            other for other in members if other.member_id != member.member_id
        ]
        reduced_hypervolume = frontier_archive_hypervolume(
            reduced_members,
            hypervolume_reference=hypervolume_reference,
        )
        contribution = float(total_hypervolume) - float(
            0.0 if reduced_hypervolume is None else reduced_hypervolume
        )
        annotated_members.append(
            replace(
                member,
                hypervolume_contribution=max(0.0, contribution),
            )
        )
    return annotated_members


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


def _evaluate_epsilon_constraint_status(
    objective_metrics: Mapping[str, float | None],
    rerun_contract: Mapping[str, object],
) -> dict[str, object]:
    scalarization_type = str(rerun_contract.get("scalarization_type", ""))
    if scalarization_type != "epsilon_constraint_sweep_v1":
        return {"ok": True, "violations": {}}
    scalarization_params = rerun_contract.get("scalarization_params", {})
    if not isinstance(scalarization_params, Mapping):
        return {"ok": True, "violations": {}}
    violation_map: dict[str, float] = {}
    for metric_name, param_key in (
        ("qa_error", "epsilon_constraint_qa_max"),
        ("boozer_residual", "epsilon_constraint_boozer_max"),
    ):
        limit = scalarization_params.get(param_key)
        metric_value = objective_metrics.get(metric_name)
        if limit is None or metric_value is None:
            continue
        excess = float(metric_value) - float(limit)
        if excess > 0.0:
            violation_map[metric_name] = excess
    return {
        "ok": not violation_map,
        "violations": violation_map,
    }


def _hypervolume_boxes(
    members: list[FrontierArchiveMember],
    *,
    hypervolume_reference: Mapping[str, float],
) -> list[tuple[float, ...]]:
    boxes: list[tuple[float, ...]] = []
    for member in members:
        box = []
        for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS:
            metric_value = member.objective_metrics.get(metric_name)
            reference_value = hypervolume_reference.get(metric_name)
            if metric_value is None or reference_value is None:
                box = []
                break
            if direction == "max":
                extent = float(metric_value) - float(reference_value)
            else:
                extent = float(reference_value) - float(metric_value)
            if not math.isfinite(extent):
                box = []
                break
            box.append(max(0.0, extent))
        if box and any(extent > 0.0 for extent in box):
            boxes.append(tuple(box))
    return boxes


def _union_hypervolume(boxes: list[tuple[float, ...]]) -> float:
    if not boxes:
        return 0.0
    dimension = len(boxes[0])
    if dimension == 1:
        return max(box[0] for box in boxes)
    boundaries = sorted({0.0, *[box[0] for box in boxes]})
    hypervolume = 0.0
    lower = 0.0
    for upper in boundaries[1:]:
        width = upper - lower
        if width <= 0.0:
            lower = upper
            continue
        active_boxes = [
            box[1:]
            for box in boxes
            if box[0] >= upper
        ]
        hypervolume += width * _union_hypervolume(active_boxes)
        lower = upper
    return hypervolume


def _build_member_id(
    campaign_id: str,
    lane_id: str,
    *,
    archive_state: str,
) -> str:
    base_member_id = f"{campaign_id}:{lane_id}"
    if archive_state == FRONTIER_ARCHIVE_STATE_PROVISIONAL:
        return f"{base_member_id}:provisional"
    return base_member_id


def _resolve_archive_state(
    archive_state: str | None,
    *,
    hard_certification_ok: bool,
) -> str:
    if archive_state is not None:
        return archive_state
    return (
        FRONTIER_ARCHIVE_STATE_CERTIFIED
        if hard_certification_ok
        else FRONTIER_ARCHIVE_STATE_REJECTED
    )
