from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .frontier_archive import (
    DEFAULT_DUPLICATE_DISTANCE_THRESHOLD,
    FrontierArchiveMember,
    annotate_hypervolume_contributions,
    archive_best_by_metric,
    certified_archive_members,
    frontier_archive_hypervolume,
    frontier_archive_member_from_json_dict,
    resolve_hypervolume_reference,
    serialize_frontier_archive,
    update_frontier_archive,
)
from .frontier_contracts import (
    FRONTIER_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
    FRONTIER_CAMPAIGN_RECOMMENDED_SCHEMA_VERSION,
    FRONTIER_CAMPAIGN_SUMMARY_SCHEMA_VERSION,
    FRONTIER_CERTIFICATION_ONLY_METRICS,
    frontier_archive_membership_rules_contract,
    frontier_archive_state_semantics_contract,
    pareto_objective_vector_contract,
    validate_frontier_campaign_manifest_payload,
    validate_frontier_campaign_summary_payload,
    validate_frontier_recommended_payload,
)
from .frontier_dominance import (
    DEFAULT_DOMINANCE_TOLERANCE,
    build_pareto_objective_normalization,
)
from .frontier_engine_multilane_local import FrontierLaneSpec
from .frontier_runtime_calibration import (
    FrontierResolvedRuntimeDefaults,
    build_frontier_early_stop_policy,
    effective_lane_budget,
    effective_total_budget,
    resolve_frontier_runtime_defaults,
)
from .frontier_scalarization import (
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
    FRONTIER_REFERENCE_MODE_EPSILON,
    frontier_scalarization_family,
)

DEFAULT_SUMMARY_JSON = "single_stage_frontier_campaign_summary.json"
DEFAULT_MANIFEST_JSON = "campaign_manifest.json"
DEFAULT_PROGRESS_JSON = "campaign_progress.json"
DEFAULT_ARCHIVE_JSON = "frontier_archive.json"
DEFAULT_RECOMMENDED_JSON = "frontier_recommended.json"


@dataclass(frozen=True)
class FrontierCampaignPaths:
    manifest_path: Path
    progress_path: Path
    archive_path: Path
    recommended_path: Path
    summary_path: Path


@dataclass(frozen=True)
class FrontierCampaignManifest:
    schema_version: str
    frontier_version: str
    frontier_engine: str
    frontier_campaign_id: str
    seed_artifact_path: str
    seed_results_path: str | None
    seed_surface_identity: str
    frontier_reference_mode: str
    frontier_reference_points: list[dict[str, float]]
    frontier_scalarization_family: str
    frontier_constraint_mode: str
    frontier_hypervolume_reference: str | None
    frontier_hypervolume_reference_metrics: dict[str, float] | None
    frontier_reference_points_file: str | None
    frontier_epsilon_spec_file: str | None
    pareto_objective_vector: list[dict[str, str]]
    pareto_objective_normalization: dict[str, object]
    dominance_tolerance: dict[str, float]
    duplicate_distance_threshold: float
    frontier_recommendation_policy: str
    archive_membership_rules: dict[str, object]
    archive_state_semantics: dict[str, object]
    certification_only_metrics: list[str]
    frontier_runtime_calibration: dict[str, object]
    frontier_early_stop_policy: dict[str, object]
    lane_budget: int | None
    total_budget: int
    rng_seed: int
    created_at: str
    stage2_artifact_init_only: bool | None
    lane_specs: list[dict[str, object]]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "SCHEMA_VERSION": self.schema_version,
            "FRONTIER_VERSION": self.frontier_version,
            "FRONTIER_ENGINE": self.frontier_engine,
            "FRONTIER_CAMPAIGN_ID": self.frontier_campaign_id,
            "SEED_ARTIFACT_PATH": self.seed_artifact_path,
            "SEED_RESULTS_PATH": self.seed_results_path,
            "SEED_SURFACE_IDENTITY": self.seed_surface_identity,
            "FRONTIER_REFERENCE_MODE": self.frontier_reference_mode,
            "FRONTIER_REFERENCE_POINTS": self.frontier_reference_points,
            "FRONTIER_SCALARIZATION_FAMILY": self.frontier_scalarization_family,
            "FRONTIER_CONSTRAINT_MODE": self.frontier_constraint_mode,
            "FRONTIER_HYPERVOLUME_REFERENCE": self.frontier_hypervolume_reference,
            "FRONTIER_HYPERVOLUME_REFERENCE_METRICS": self.frontier_hypervolume_reference_metrics,
            "FRONTIER_REFERENCE_POINTS_FILE": self.frontier_reference_points_file,
            "FRONTIER_EPSILON_SPEC_FILE": self.frontier_epsilon_spec_file,
            "PARETO_OBJECTIVE_VECTOR": self.pareto_objective_vector,
            "PARETO_OBJECTIVE_NORMALIZATION": self.pareto_objective_normalization,
            "DOMINANCE_TOLERANCE": self.dominance_tolerance,
            "DUPLICATE_DISTANCE_THRESHOLD": self.duplicate_distance_threshold,
            "FRONTIER_RECOMMENDATION_POLICY": self.frontier_recommendation_policy,
            "ARCHIVE_MEMBERSHIP_RULES": self.archive_membership_rules,
            "ARCHIVE_STATE_SEMANTICS": self.archive_state_semantics,
            "CERTIFICATION_ONLY_METRICS": self.certification_only_metrics,
            "FRONTIER_RUNTIME_CALIBRATION": self.frontier_runtime_calibration,
            "FRONTIER_EARLY_STOP_POLICY": self.frontier_early_stop_policy,
            "LANE_BUDGET": self.lane_budget,
            "TOTAL_BUDGET": self.total_budget,
            "RNG_SEED": self.rng_seed,
            "CREATED_AT": self.created_at,
            "STAGE2_ARTIFACT_INIT_ONLY": self.stage2_artifact_init_only,
            "LANE_SPECS": self.lane_specs,
        }


def resolve_frontier_campaign_paths(
    output_root: Path,
    *,
    summary_path: Path,
) -> FrontierCampaignPaths:
    return FrontierCampaignPaths(
        manifest_path=output_root / DEFAULT_MANIFEST_JSON,
        progress_path=output_root / DEFAULT_PROGRESS_JSON,
        archive_path=output_root / DEFAULT_ARCHIVE_JSON,
        recommended_path=output_root / DEFAULT_RECOMMENDED_JSON,
        summary_path=summary_path,
    )


def frontier_constraint_mode(lane_specs: list[FrontierLaneSpec]) -> str:
    constraint_modes: list[str] = []
    for lane_spec in lane_specs:
        if lane_spec.scalarization_type == FRONTIER_REFERENCE_MODE_EPSILON:
            constraint_modes.append("frontier_epsilon_constraint_v1")
        elif lane_spec.scalarization_type == FRONTIER_REFERENCE_MODE_ACHIEVEMENT:
            constraint_modes.append("frontier_achievement_chebyshev_v1")
        else:
            constraint_modes.append("frontier_v2_single_lane_contract")
    unique_modes = sorted(set(constraint_modes))
    if not unique_modes:
        return "empty"
    if len(unique_modes) == 1:
        return unique_modes[0]
    return "mixed:" + ",".join(unique_modes)


def build_frontier_campaign_manifest(
    args,
    *,
    campaign_id: str,
    stage2_bs_path: Path,
    stage2_results_path: Path | None,
    stage2_results: dict | None,
    lane_specs: list[FrontierLaneSpec],
    runtime_defaults: FrontierResolvedRuntimeDefaults | None = None,
) -> dict[str, object]:
    if runtime_defaults is None:
        runtime_defaults = resolve_frontier_runtime_defaults(
            profile_name=getattr(
                args,
                "frontier_runtime_calibration_profile",
                "reduced_fixture_v1",
            ),
            requested_num_lanes=getattr(args, "frontier_num_lanes", None),
            requested_lane_budget=getattr(args, "frontier_lane_budget", None),
            requested_total_budget=getattr(args, "frontier_total_budget", None),
            requested_checkpoint_every=getattr(args, "checkpoint_every", None),
            requested_early_stop_patience_lanes=getattr(
                args,
                "frontier_early_stop_patience_lanes",
                None,
            ),
            requested_early_stop_min_certified=getattr(
                args,
                "frontier_early_stop_min_certified",
                None,
            ),
            requested_early_stop_min_hypervolume_gain=getattr(
                args,
                "frontier_early_stop_min_hypervolume_gain",
                None,
            ),
        )
    hypervolume_reference = resolve_hypervolume_reference(
        reference_spec=args.frontier_hypervolume_reference,
        seed_results=stage2_results,
    )
    effective_lane_budgets = [
        effective_lane_budget(lane.lane_budget, runtime_defaults)
        for lane in lane_specs
    ]
    manifest_lane_budget = None
    if effective_lane_budgets:
        first_lane_budget = effective_lane_budgets[0]
        if all(budget == first_lane_budget for budget in effective_lane_budgets):
            manifest_lane_budget = int(first_lane_budget)
    total_budget = effective_total_budget(
        effective_lane_budgets,
        runtime_defaults,
    )
    manifest = FrontierCampaignManifest(
        schema_version=FRONTIER_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        frontier_version=args.frontier_version,
        frontier_engine=args.frontier_engine,
        frontier_campaign_id=campaign_id,
        seed_artifact_path=str(stage2_bs_path),
        seed_results_path=None
        if stage2_results_path is None
        else str(stage2_results_path),
        seed_surface_identity=Path(args.plasma_surf_filename).name,
        frontier_reference_mode=args.frontier_reference_mode,
        frontier_reference_points=[
            dict(lane.scalarization_params) for lane in lane_specs
        ],
        frontier_scalarization_family=frontier_scalarization_family(lane_specs),
        frontier_constraint_mode=frontier_constraint_mode(lane_specs),
        frontier_hypervolume_reference=args.frontier_hypervolume_reference,
        frontier_hypervolume_reference_metrics=hypervolume_reference,
        frontier_reference_points_file=args.frontier_reference_points_file,
        frontier_epsilon_spec_file=args.frontier_epsilon_spec_file,
        pareto_objective_vector=pareto_objective_vector_contract(),
        pareto_objective_normalization=build_pareto_objective_normalization(
            hypervolume_reference,
            kind=getattr(
                args,
                "frontier_normalization_kind",
                "seed_relative_reference_fraction_with_floor",
            ),
            normalization_spec_path=getattr(
                args,
                "frontier_normalization_spec_file",
                None,
            ),
        ),
        dominance_tolerance=dict(DEFAULT_DOMINANCE_TOLERANCE),
        duplicate_distance_threshold=DEFAULT_DUPLICATE_DISTANCE_THRESHOLD,
        frontier_recommendation_policy=args.frontier_recommendation_policy,
        archive_membership_rules=frontier_archive_membership_rules_contract(),
        archive_state_semantics=frontier_archive_state_semantics_contract(),
        certification_only_metrics=list(FRONTIER_CERTIFICATION_ONLY_METRICS),
        frontier_runtime_calibration=runtime_defaults.to_json_dict(),
        frontier_early_stop_policy=build_frontier_early_stop_policy(
            runtime_defaults
        ),
        lane_budget=manifest_lane_budget,
        total_budget=int(total_budget),
        rng_seed=int(args.frontier_rng_seed),
        created_at=datetime.now(timezone.utc).isoformat(),
        stage2_artifact_init_only=None
        if stage2_results is None
        else stage2_results.get("init_only"),
        lane_specs=[lane.to_json_dict() for lane in lane_specs],
    )
    payload = manifest.to_json_dict()
    validate_frontier_campaign_manifest_payload(payload)
    return payload


def build_frontier_hypervolume_history(
    lane_records: list[dict[str, object]],
    *,
    hypervolume_reference: dict[str, float] | None,
    pareto_objective_normalization: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if hypervolume_reference is None:
        return []
    running_archive: list[FrontierArchiveMember] = []
    history: list[dict[str, object]] = []
    for lane_record in lane_records:
        archive_member_payload = lane_record.get("archive_member")
        if isinstance(archive_member_payload, dict):
            archive_member = frontier_archive_member_from_json_dict(
                archive_member_payload
            )
            running_archive, _ = update_frontier_archive(
                running_archive,
                archive_member,
                pareto_objective_normalization=pareto_objective_normalization,
            )
        certified_members = certified_archive_members(running_archive)
        annotated_members = annotate_hypervolume_contributions(
            certified_members,
            hypervolume_reference=hypervolume_reference,
        )
        history.append(
            {
                "lane_id": lane_record.get("lane_id"),
                "status": lane_record.get("status"),
                "archive_size": len(annotated_members),
                "hypervolume": frontier_archive_hypervolume(
                    annotated_members,
                    hypervolume_reference=hypervolume_reference,
                ),
            }
        )
    return history


def build_recommended_summary(
    recommendation_payload: dict[str, object] | None,
    *,
    archive_size: int,
    policy_name: str,
) -> dict[str, object]:
    if recommendation_payload is None:
        payload = {
            "schema_version": FRONTIER_CAMPAIGN_RECOMMENDED_SCHEMA_VERSION,
            "recommended_member_id": None,
            "policy_name": policy_name,
            "policy_inputs": None,
            "policy_rationale": None,
            "policy_score": None,
            "recommended_metrics": None,
            "frontier_archive_size": archive_size,
        }
        validate_frontier_recommended_payload(payload)
        return payload

    recommended_member = recommendation_payload["recommended_member"]
    payload = {
        "schema_version": FRONTIER_CAMPAIGN_RECOMMENDED_SCHEMA_VERSION,
        "recommended_member_id": recommended_member.member_id,
        "policy_name": recommendation_payload["policy_name"],
        "policy_inputs": recommendation_payload["policy_inputs"],
        "policy_rationale": recommendation_payload["policy_rationale"],
        "policy_score": recommendation_payload["policy_score"],
        "recommended_metrics": dict(recommended_member.objective_metrics),
        "frontier_archive_size": archive_size,
    }
    validate_frontier_recommended_payload(payload)
    return payload


def build_target_comparison(
    *,
    delta_fn,
    target_payload: dict[str, object] | None,
    recommended_payload: dict[str, object] | None,
) -> dict[str, object] | None:
    if target_payload is None or target_payload.get("status") != "completed":
        return None
    if recommended_payload is None:
        return None
    target_results = target_payload["results"]
    recommended_member = recommended_payload["recommended_member"]
    return {
        "recommended_minus_target_final_iota": delta_fn(
            recommended_member.objective_metrics.get("iota"),
            target_results.get("FINAL_IOTA"),
        ),
        "recommended_minus_target_final_volume": delta_fn(
            recommended_member.objective_metrics.get("volume"),
            target_results.get("FINAL_VOLUME"),
        ),
        "recommended_minus_target_nonqs_ratio": delta_fn(
            recommended_member.objective_metrics.get("qa_error"),
            target_results.get("NONQS_RATIO"),
        ),
        "recommended_minus_target_boozer_residual": delta_fn(
            recommended_member.objective_metrics.get("boozer_residual"),
            target_results.get("BOOZER_RESIDUAL"),
        ),
    }


def build_frontier_campaign_summary(
    args,
    *,
    campaign_id: str,
    stage2_bs_path: Path,
    stage2_results_path: Path | None,
    stage2_results: dict | None,
    paths: FrontierCampaignPaths,
    lane_specs: list[FrontierLaneSpec],
    target_payload: dict[str, object] | None,
    lane_records: list[dict[str, object]],
    archive_members: list[FrontierArchiveMember],
    recommendation_payload: dict[str, object] | None,
    delta_fn,
    runtime_defaults: FrontierResolvedRuntimeDefaults,
    early_stop_status: dict[str, object],
) -> dict[str, object]:
    hypervolume_reference = resolve_hypervolume_reference(
        reference_spec=args.frontier_hypervolume_reference,
        seed_results=stage2_results,
        members=archive_members,
    )
    pareto_objective_normalization = build_pareto_objective_normalization(
        hypervolume_reference,
        kind=getattr(
            args,
            "frontier_normalization_kind",
            "seed_relative_reference_fraction_with_floor",
        ),
        normalization_spec_path=getattr(
            args,
            "frontier_normalization_spec_file",
            None,
        ),
    )
    certified_members = certified_archive_members(archive_members)
    annotated_certified_members = annotate_hypervolume_contributions(
        certified_members,
        hypervolume_reference=hypervolume_reference,
    )
    hypervolume_history = build_frontier_hypervolume_history(
        lane_records,
        hypervolume_reference=hypervolume_reference,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    recommended_member = build_recommended_summary(
        recommendation_payload,
        archive_size=len(annotated_certified_members),
        policy_name=args.frontier_recommendation_policy,
    )
    feasible_lane_count = sum(
        1
        for lane_record in lane_records
        if lane_record.get("final_certified")
    )
    dominance_updates = [
        lane_record.get("archive_update")
        for lane_record in lane_records
        if lane_record.get("archive_update") is not None
    ]
    summary: dict[str, object] = {
        "schema_version": FRONTIER_CAMPAIGN_SUMMARY_SCHEMA_VERSION,
        "frontier_version": args.frontier_version,
        "frontier_engine": args.frontier_engine,
        "frontier_campaign_id": campaign_id,
        "dry_run": bool(args.dry_run),
        "output_root": str(Path(args.output_root).resolve()),
        "manifest_path": str(paths.manifest_path),
        "progress_path": str(paths.progress_path),
        "archive_path": str(paths.archive_path),
        "recommended_path": str(paths.recommended_path),
        "stage2_bs_path": str(stage2_bs_path),
        "stage2_results_path": None
        if stage2_results_path is None
        else str(stage2_results_path),
        "stage2_artifact_init_only": None
        if stage2_results is None
        else stage2_results.get("init_only"),
        "frontier_num_lanes": len(lane_specs),
        "frontier_lane_specs": [lane.to_json_dict() for lane in lane_specs],
        "frontier_lanes": [
            _sanitize_lane_record_for_final_output(lane_record)
            for lane_record in lane_records
        ],
        "frontier_archive": serialize_frontier_archive(
            annotated_certified_members,
            hypervolume_reference=hypervolume_reference,
        ),
        "frontier_archive_size": len(annotated_certified_members),
        "frontier_archive_best_by_metric": archive_best_by_metric(
            annotated_certified_members
        ),
        "frontier_feasible_lane_count": feasible_lane_count,
        "frontier_non_dominated_count": len(annotated_certified_members),
        "frontier_dominance_updates": dominance_updates,
        "frontier_hypervolume_reference": hypervolume_reference,
        "frontier_hypervolume": frontier_archive_hypervolume(
            annotated_certified_members,
            hypervolume_reference=hypervolume_reference,
        ),
        "frontier_hypervolume_history": hypervolume_history,
        "frontier_runtime_calibration": runtime_defaults.to_json_dict(),
        "frontier_early_stop": dict(early_stop_status),
        "target_run": None,
        "recommended_member": recommended_member,
        "target_comparison": None,
    }
    if target_payload is not None:
        target_summary = {
            "status": target_payload["status"],
            "command": target_payload["command"],
        }
        if target_payload["status"] == "completed":
            target_results_summary = target_payload.get(
                "results_summary",
                target_payload.get("results"),
            )
            target_summary.update(
                {
                    "results_path": str(target_payload["results_path"]),
                    "result_source": target_payload["result_source"],
                    "results": target_results_summary,
                }
            )
        elif target_payload["status"] == "failed":
            target_summary["error_type"] = target_payload.get("error_type")
            target_summary["error_message"] = target_payload.get("error_message")
        summary["target_run"] = target_summary
    if recommendation_payload is not None:
        summary["target_comparison"] = build_target_comparison(
            delta_fn=delta_fn,
            target_payload=target_payload,
            recommended_payload=recommendation_payload,
        )
    validate_frontier_campaign_summary_payload(summary)
    return summary


def _sanitize_lane_record_for_final_output(
    lane_record: dict[str, object],
) -> dict[str, object]:
    sanitized = dict(lane_record)
    sanitized["provisional_member_ids"] = []
    return sanitized


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, str(path))
    except BaseException:
        os.unlink(tmp_path)
        raise
