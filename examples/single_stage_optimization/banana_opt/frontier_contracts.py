from __future__ import annotations

from typing import Mapping

from .frontier_dominance import (
    PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES,
    PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
    PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
    DEFAULT_DOMINANCE_TOLERANCE,
    PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
    PARETO_OBJECTIVE_NORMALIZATION_RULES,
    PARETO_OBJECTIVE_SPECS,
)

FRONTIER_ARCHIVE_STATE_PROVISIONAL = "provisional"
FRONTIER_ARCHIVE_STATE_CERTIFIED = "certified"
FRONTIER_ARCHIVE_STATE_REJECTED = "rejected"
FRONTIER_ARCHIVE_STATES = (
    FRONTIER_ARCHIVE_STATE_PROVISIONAL,
    FRONTIER_ARCHIVE_STATE_CERTIFIED,
    FRONTIER_ARCHIVE_STATE_REJECTED,
)
FRONTIER_FINAL_OUTPUT_ARCHIVE_STATES = (FRONTIER_ARCHIVE_STATE_CERTIFIED,)

FRONTIER_ARCHIVE_SCHEMA_VERSION = "frontier_archive_v1"
FRONTIER_CAMPAIGN_PROGRESS_SCHEMA_VERSION = "frontier_campaign_progress_v1"
FRONTIER_LANE_CONTRACT_SCHEMA_VERSION = "frontier_lane_contract_v1"
FRONTIER_LANE_RECORD_SCHEMA_VERSION = "frontier_lane_record_v1"
FRONTIER_CAMPAIGN_MANIFEST_SCHEMA_VERSION = "frontier_campaign_manifest_v1"
FRONTIER_CAMPAIGN_SUMMARY_SCHEMA_VERSION = "frontier_campaign_summary_v1"
FRONTIER_CAMPAIGN_RECOMMENDED_SCHEMA_VERSION = "frontier_campaign_recommended_v1"
FRONTIER_SOLVER_CHECKPOINT_SCHEMA_VERSION = "single_stage_solver_checkpoint_v1"

FRONTIER_CERTIFICATION_ONLY_METRICS = (
    "coil_length",
    "curve_curve_min_dist",
    "curve_surface_min_dist",
    "surface_vessel_min_dist",
    "max_curvature",
    "hardware_constraints_ok",
    "final_feasibility_ok",
    "final_topology_gate_success",
    "frontier_trust_ok",
    "finite_eval_ok",
)

SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES = (
    "balanced",
    "max_iota_under_safe_boozer",
    "max_volume_under_safe_hardware",
    "closest_to_seed",
)


def pareto_objective_vector_contract() -> list[dict[str, str]]:
    return [
        {"metric": metric_name, "direction": direction}
        for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS
    ]


def frontier_archive_membership_rules_contract() -> dict[str, object]:
    return {
        "objective_metrics": [metric_name for metric_name, _, _ in PARETO_OBJECTIVE_SPECS],
        "certification_only_metrics": list(FRONTIER_CERTIFICATION_ONLY_METRICS),
        "final_output_archive_states": list(FRONTIER_FINAL_OUTPUT_ARCHIVE_STATES),
        "requires_hard_certification": True,
        "duplicate_distance_threshold_kind": "normalized_objective_distance",
        "dominance_tolerance": dict(DEFAULT_DOMINANCE_TOLERANCE),
    }


def frontier_archive_state_semantics_contract() -> dict[str, object]:
    return {
        "supported_states": list(FRONTIER_ARCHIVE_STATES),
        "provisional_visible_in_progress": True,
        "provisional_visible_in_final_outputs": False,
        "certified_only_final_archive": True,
    }


def validate_frontier_lane_contract_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=FRONTIER_LANE_CONTRACT_SCHEMA_VERSION,
        field="schema_version",
    )
    _require_keys(
        payload,
        (
            "lane_id",
            "campaign_id",
            "engine",
            "scalarization_type",
            "scalarization_params",
            "constraint_mode",
            "optimizer_budget",
            "rng_seed",
            "rerun_contract",
        ),
    )
    _require_mapping(payload, "scalarization_params")
    _require_mapping(payload, "rerun_contract")


def validate_frontier_lane_record_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=FRONTIER_LANE_RECORD_SCHEMA_VERSION,
        field="schema_version",
    )
    _require_keys(
        payload,
        (
            "lane_id",
            "campaign_id",
            "engine",
            "scalarization_type",
            "scalarization_params",
            "constraint_mode",
            "optimizer_budget",
            "rng_seed",
            "rerun_contract",
            "status",
            "command",
            "weights",
            "lane_budget",
            "provisional_member_ids",
            "certified_member_ids",
            "final_certified",
        ),
    )
    _require_mapping(payload, "scalarization_params")
    _require_mapping(payload, "rerun_contract")
    _require_mapping(payload, "weights")
    _require_list(payload, "command")
    _require_list(payload, "provisional_member_ids")
    _require_list(payload, "certified_member_ids")


def validate_frontier_campaign_progress_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=FRONTIER_CAMPAIGN_PROGRESS_SCHEMA_VERSION,
        field="schema_version",
    )
    _require_keys(
        payload,
        (
            "campaign_id",
            "frontier_version",
            "frontier_engine",
            "target_payload",
            "lane_records",
            "provisional_archive_members",
            "archive_members",
        ),
    )
    lane_records = _require_list(payload, "lane_records")
    provisional_members = _require_list(payload, "provisional_archive_members")
    archive_members = _require_list(payload, "archive_members")
    for lane_record in lane_records:
        if not isinstance(lane_record, Mapping):
            raise ValueError("lane_records entries must be mappings")
        validate_frontier_lane_record_payload(lane_record)
    for member_payload in provisional_members:
        if not isinstance(member_payload, Mapping):
            raise ValueError("provisional_archive_members entries must be mappings")
        validate_frontier_archive_member_payload(
            member_payload,
            expected_state=FRONTIER_ARCHIVE_STATE_PROVISIONAL,
        )
    for member_payload in archive_members:
        if not isinstance(member_payload, Mapping):
            raise ValueError("archive_members entries must be mappings")
        validate_frontier_archive_member_payload(
            member_payload,
            expected_state=FRONTIER_ARCHIVE_STATE_CERTIFIED,
        )


def validate_frontier_archive_member_payload(
    payload: Mapping[str, object],
    *,
    expected_state: str | None = None,
) -> None:
    _require_keys(
        payload,
        (
            "member_id",
            "lane_id",
            "campaign_id",
            "archive_state",
            "dominance_signature",
            "objective_metrics",
            "reference_metrics",
            "constraint_metrics",
            "hard_certification_ok",
            "soft_search_score",
            "distance_from_seed",
            "hypervolume_contribution",
            "recommendation_flags",
            "rerun_contract",
            "result_source",
            "results_path",
            "termination_reason",
            "success",
        ),
    )
    archive_state = str(payload["archive_state"])
    if archive_state not in FRONTIER_ARCHIVE_STATES:
        raise ValueError(f"Unsupported frontier archive state: {archive_state}")
    if expected_state is not None and archive_state != expected_state:
        raise ValueError(
            f"Expected frontier archive state {expected_state!r}, got {archive_state!r}"
        )
    for field_name in (
        "dominance_signature",
        "objective_metrics",
        "reference_metrics",
        "constraint_metrics",
        "recommendation_flags",
        "rerun_contract",
    ):
        _require_mapping(payload, field_name)


def validate_frontier_archive_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=FRONTIER_ARCHIVE_SCHEMA_VERSION,
        field="schema_version",
    )
    _require_keys(
        payload,
        (
            "pareto_objective_vector",
            "archive_membership_rules",
            "archive_state_semantics",
            "dominance_tolerance",
            "duplicate_distance_threshold",
            "hypervolume_reference",
            "hypervolume_total",
            "members",
            "best_by_metric",
        ),
    )
    if payload["pareto_objective_vector"] != pareto_objective_vector_contract():
        raise ValueError("pareto_objective_vector does not match the frozen v4 contract")
    if payload["archive_membership_rules"] != frontier_archive_membership_rules_contract():
        raise ValueError("archive_membership_rules does not match the frozen v4 contract")
    if payload["archive_state_semantics"] != frontier_archive_state_semantics_contract():
        raise ValueError("archive_state_semantics does not match the frozen v4 contract")
    _require_mapping(payload, "dominance_tolerance")
    _require_mapping(payload, "best_by_metric")
    members = _require_list(payload, "members")
    for member_payload in members:
        if not isinstance(member_payload, Mapping):
            raise ValueError("frontier archive members must be mappings")
        validate_frontier_archive_member_payload(
            member_payload,
            expected_state=FRONTIER_ARCHIVE_STATE_CERTIFIED,
        )


def validate_frontier_campaign_manifest_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=FRONTIER_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        field="SCHEMA_VERSION",
    )
    _require_keys(
        payload,
        (
            "FRONTIER_VERSION",
            "FRONTIER_ENGINE",
            "FRONTIER_CAMPAIGN_ID",
            "SEED_ARTIFACT_PATH",
            "FRONTIER_REFERENCE_MODE",
            "FRONTIER_SCALARIZATION_FAMILY",
            "FRONTIER_CONSTRAINT_MODE",
            "PARETO_OBJECTIVE_VECTOR",
            "PARETO_OBJECTIVE_NORMALIZATION",
            "DOMINANCE_TOLERANCE",
            "DUPLICATE_DISTANCE_THRESHOLD",
            "FRONTIER_RECOMMENDATION_POLICY",
            "ARCHIVE_MEMBERSHIP_RULES",
            "ARCHIVE_STATE_SEMANTICS",
            "CERTIFICATION_ONLY_METRICS",
            "FRONTIER_RUNTIME_CALIBRATION",
            "FRONTIER_EARLY_STOP_POLICY",
            "LANE_SPECS",
        ),
    )
    if payload["PARETO_OBJECTIVE_VECTOR"] != pareto_objective_vector_contract():
        raise ValueError("Manifest Pareto objective vector drifted from the frozen contract")
    if payload["ARCHIVE_MEMBERSHIP_RULES"] != frontier_archive_membership_rules_contract():
        raise ValueError("Manifest archive membership rules drifted from the frozen contract")
    if payload["ARCHIVE_STATE_SEMANTICS"] != frontier_archive_state_semantics_contract():
        raise ValueError("Manifest archive state semantics drifted from the frozen contract")
    if list(payload["CERTIFICATION_ONLY_METRICS"]) != list(FRONTIER_CERTIFICATION_ONLY_METRICS):
        raise ValueError("Manifest certification-only metrics drifted from the frozen contract")
    if str(payload["FRONTIER_RECOMMENDATION_POLICY"]) not in SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES:
        raise ValueError("Manifest recommendation policy is not part of the frozen contract")
    normalization = _require_mapping(payload, "PARETO_OBJECTIVE_NORMALIZATION")
    _validate_pareto_normalization_payload(normalization)
    _require_mapping(payload, "DOMINANCE_TOLERANCE")
    frontier_runtime_calibration = _require_mapping(
        payload,
        "FRONTIER_RUNTIME_CALIBRATION",
    )
    _require_mapping(frontier_runtime_calibration, "profile")
    _require_mapping(frontier_runtime_calibration, "resolved_defaults")
    frontier_early_stop_policy = _require_mapping(
        payload,
        "FRONTIER_EARLY_STOP_POLICY",
    )
    _require_keys(
        frontier_early_stop_policy,
        ("patience_lanes", "min_certified", "min_hypervolume_gain"),
    )
    _require_list(payload, "LANE_SPECS")


def validate_frontier_recommended_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=FRONTIER_CAMPAIGN_RECOMMENDED_SCHEMA_VERSION,
        field="schema_version",
    )
    _require_keys(
        payload,
        (
            "recommended_member_id",
            "policy_name",
            "policy_inputs",
            "policy_rationale",
            "policy_score",
            "recommended_metrics",
            "frontier_archive_size",
        ),
    )
    policy_name = payload["policy_name"]
    if policy_name is not None and str(policy_name) not in SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES:
        raise ValueError(f"Unsupported frontier recommendation policy: {policy_name}")


def validate_frontier_campaign_summary_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=FRONTIER_CAMPAIGN_SUMMARY_SCHEMA_VERSION,
        field="schema_version",
    )
    _require_keys(
        payload,
        (
            "frontier_version",
            "frontier_engine",
            "frontier_campaign_id",
            "output_root",
            "manifest_path",
            "progress_path",
            "archive_path",
            "recommended_path",
            "frontier_num_lanes",
            "frontier_lane_specs",
            "frontier_lanes",
            "frontier_archive",
            "frontier_archive_size",
            "frontier_archive_best_by_metric",
            "frontier_feasible_lane_count",
            "frontier_non_dominated_count",
            "frontier_dominance_updates",
            "frontier_hypervolume_reference",
            "frontier_hypervolume",
            "frontier_hypervolume_history",
            "frontier_runtime_calibration",
            "frontier_early_stop",
            "recommended_member",
            "target_run",
            "target_comparison",
        ),
    )
    _require_list(payload, "frontier_lane_specs")
    frontier_lanes = _require_list(payload, "frontier_lanes")
    frontier_hypervolume_history = _require_list(
        payload,
        "frontier_hypervolume_history",
    )
    frontier_archive = _require_mapping(payload, "frontier_archive")
    frontier_runtime_calibration = _require_mapping(
        payload,
        "frontier_runtime_calibration",
    )
    frontier_early_stop = _require_mapping(payload, "frontier_early_stop")
    recommended_member = _require_mapping(payload, "recommended_member")
    validate_frontier_archive_payload(frontier_archive)
    validate_frontier_recommended_payload(recommended_member)
    _require_mapping(frontier_runtime_calibration, "resolved_defaults")
    _require_mapping(frontier_early_stop, "policy")
    for history_entry in frontier_hypervolume_history:
        if not isinstance(history_entry, Mapping):
            raise ValueError("frontier_hypervolume_history entries must be mappings")
        _require_keys(
            history_entry,
            ("lane_id", "status", "archive_size", "hypervolume"),
        )
    for lane_payload in frontier_lanes:
        if not isinstance(lane_payload, Mapping):
            raise ValueError("frontier_lanes entries must be mappings")
        validate_frontier_lane_record_payload(lane_payload)
        if lane_payload.get("provisional_member_ids"):
            raise ValueError(
                "frontier_lanes in final summary must not expose provisional_member_ids"
            )
    _validate_optional_nsga3_summary_payload(payload)


def _validate_pareto_normalization_payload(payload: Mapping[str, object]) -> None:
    _require_schema_version(
        payload,
        expected=PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
        field="schema_version",
    )
    normalization_kind = str(payload.get("kind"))
    if str(payload.get("distance_metric")) != "euclidean":
        raise ValueError("Unsupported Pareto normalization distance metric")
    metric_rules = _require_mapping(payload, "metric_rules")
    if normalization_kind == PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE:
        if metric_rules != PARETO_OBJECTIVE_NORMALIZATION_RULES:
            raise ValueError(
                "Pareto normalization metric rules drifted from the frozen contract"
            )
        return
    if normalization_kind == PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR:
        if metric_rules != PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES:
            raise ValueError(
                "Ideal/nadir Pareto normalization metric rules drifted from the frozen contract"
            )
        _require_mapping(payload, "ideal_metrics")
        _require_mapping(payload, "nadir_metrics")
        return
    raise ValueError("Unsupported Pareto normalization kind")


def _validate_optional_nsga3_summary_payload(payload: Mapping[str, object]) -> None:
    generation_history = payload.get("frontier_generation_history")
    if generation_history is not None:
        generation_history_list = _require_list(
            payload,
            "frontier_generation_history",
        )
        for entry in generation_history_list:
            if not isinstance(entry, Mapping):
                raise ValueError("frontier_generation_history entries must be mappings")
            _require_keys(
                entry,
                (
                    "generation",
                    "population_size",
                    "feasible_count",
                    "archive_size",
                    "archive_growth",
                    "cv_min",
                    "cv_mean",
                    "cv_max",
                    "failure_histogram",
                    "cache_hits",
                    "cache_misses",
                    "hypervolume",
                ),
            )
            _require_mapping(entry, "failure_histogram")
    engine_stats = payload.get("frontier_engine_stats")
    if engine_stats is not None:
        _require_mapping(payload, "frontier_engine_stats")
    evaluator_spec = payload.get("frontier_evaluator_spec")
    if evaluator_spec is not None:
        spec_payload = _require_mapping(payload, "frontier_evaluator_spec")
        _require_keys(spec_payload, ("schema_version", "run_identity"))
    for path_field in (
        "frontier_evaluator_spec_path",
        "frontier_population_checkpoint_path",
        "frontier_generation_history_path",
    ):
        path_value = payload.get(path_field)
        if path_value is not None and not isinstance(path_value, str):
            raise ValueError(f"{path_field} must be a string when present")


def _require_schema_version(
    payload: Mapping[str, object],
    *,
    expected: str,
    field: str,
) -> None:
    observed = payload.get(field)
    if observed != expected:
        raise ValueError(f"Expected {field}={expected!r}, got {observed!r}")


def _require_keys(payload: Mapping[str, object], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Missing required frontier contract keys: {missing}")


def _require_mapping(
    payload: Mapping[str, object],
    field_name: str,
) -> Mapping[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_list(payload: Mapping[str, object], field_name: str) -> list[object]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value
