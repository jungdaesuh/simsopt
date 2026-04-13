from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_single_stage_goal_mode_comparison as goal_mode_comparison  # noqa: E402
from banana_opt.frontier_archive import (  # noqa: E402
    build_archive_member_from_results,
    serialize_frontier_archive,
    update_frontier_archive,
)
from banana_opt.frontier_campaign_reporting import (  # noqa: E402
    DEFAULT_SUMMARY_JSON,
    build_frontier_campaign_manifest,
    build_frontier_campaign_summary,
    build_recommended_summary,
    resolve_frontier_campaign_paths,
    write_json,
)
from banana_opt.frontier_engine_base import (  # noqa: E402
    FrontierCampaignProgress,
    FrontierLaneContract,
    FrontierLaneRecord,
    build_frontier_lane_contract,
    build_frontier_lane_record,
    load_frontier_campaign_progress,
    serialize_goal_mode_payload,
    write_frontier_campaign_progress,
)
from banana_opt.frontier_engine_multilane_local import (  # noqa: E402
    FrontierLaneSpec,
    generate_multilane_local_specs,  # re-exported for test compatibility
)
from banana_opt.frontier_scalarization import (  # noqa: E402
    FRONTIER_REFERENCE_MODE_EPSILON,
    FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
    FRONTIER_REFERENCE_MODE_SHARED,
    SUPPORTED_FRONTIER_REFERENCE_MODES,
    generate_frontier_lane_specs,
)
from banana_opt.frontier_recommendation import recommend_frontier_member  # noqa: E402
from workflow_runner_common import resolved_optional_path, resolved_path  # noqa: E402

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_frontier_campaign"


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-lane frontier campaign by scheduling multiple existing "
            "single-stage frontier_v2 lanes from one validated Stage 2 seed."
        ),
        parents=[goal_mode_comparison.build_parser(add_help=False)],
        add_help=add_help,
        conflict_handler="resolve",
    )
    parser.set_defaults(output_root=str(DEFAULT_OUTPUT_ROOT), summary_json=None)
    parser.add_argument(
        "--frontier-version",
        default="frontier_v3_multilane_local_v1",
    )
    parser.add_argument(
        "--frontier-engine",
        choices=["multilane_local"],
        default="multilane_local",
    )
    parser.add_argument(
        "--frontier-reference-mode",
        choices=SUPPORTED_FRONTIER_REFERENCE_MODES,
        default=FRONTIER_REFERENCE_MODE_SHARED,
    )
    parser.add_argument(
        "--frontier-hypervolume-reference",
        default=None,
    )
    parser.add_argument(
        "--frontier-reference-points-file",
        default=None,
    )
    parser.add_argument(
        "--frontier-epsilon-spec-file",
        default=None,
    )
    parser.add_argument("--frontier-num-lanes", type=int, default=3)
    parser.add_argument(
        "--frontier-lane-budget",
        type=int,
        default=None,
        help="Optional per-lane maxiter override for frontier lanes.",
    )
    parser.add_argument(
        "--frontier-total-budget",
        type=int,
        default=None,
        help="Optional campaign budget metadata. Defaults to num_lanes * lane_budget.",
    )
    parser.add_argument(
        "--frontier-recommendation-policy",
        choices=["balanced"],
        default="balanced",
    )
    parser.add_argument("--frontier-rng-seed", type=int, default=0)
    parser.add_argument(
        "--skip-target",
        action="store_true",
        help="Skip the target baseline lane and run frontier lanes only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously started frontier campaign from campaign_progress.json.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_goal_mode_case_safe(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    output_root: Path,
) -> dict[str, object]:
    command = goal_mode_comparison.build_single_stage_goal_mode_command(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        case_output_root=output_root / goal_mode,
    )
    try:
        payload = goal_mode_comparison.run_goal_mode_case(
            args,
            goal_mode=goal_mode,
            stage2_bs_path=stage2_bs_path,
            output_root=output_root,
        )
    except Exception as error:
        return {
            "status": "failed",
            "command": command,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    if args.dry_run:
        return {
            "status": "dry_run",
            "command": payload["command"],
        }
    completed_payload = {
        "status": "completed",
        **payload,
    }
    completed_payload["results_summary"] = goal_mode_comparison.result_metric_subset(
        payload["results"]
    )
    return completed_payload


def build_frontier_lane_args(
    args: argparse.Namespace,
    lane_spec: FrontierLaneSpec,
) -> argparse.Namespace:
    lane_args = argparse.Namespace(**vars(args))
    lane_args.iotas_weight = lane_spec.iotas_weight
    lane_args.frontier_volume_weight = lane_spec.frontier_volume_weight
    lane_args.res_weight = lane_spec.res_weight
    if lane_spec.lane_budget is not None:
        lane_args.maxiter = int(lane_spec.lane_budget)
    for attribute_name in (
        "frontier_reference_iota",
        "frontier_reference_iota_scale",
        "frontier_reference_volume",
        "frontier_reference_volume_scale",
        "frontier_reference_qa",
        "frontier_reference_boozer",
        "frontier_boozer_trust_threshold",
        "frontier_boozer_trust_penalty_scale",
    ):
        setattr(
            lane_args,
            attribute_name,
            lane_spec.scalarization_params.get(attribute_name),
        )
    return lane_args


def lane_rng_seed(base_seed: int, *, lane_index: int) -> int:
    return int(base_seed) + lane_index


def build_lane_rerun_contract(
    args: argparse.Namespace,
    lane_spec: FrontierLaneSpec,
    *,
    stage2_bs_path: Path,
) -> dict[str, object]:
    return {
        "single_stage_goal_mode": "frontier",
        "stage2_bs_path": str(stage2_bs_path),
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "iotas_weight": lane_spec.iotas_weight,
        "frontier_volume_weight": lane_spec.frontier_volume_weight,
        "res_weight": lane_spec.res_weight,
        "maxiter": args.maxiter
        if lane_spec.lane_budget is None
        else lane_spec.lane_budget,
        "constraint_method": args.constraint_method,
        "hardware_search_mode": args.hardware_search_mode,
        "scalarization_type": lane_spec.scalarization_type,
        "scalarization_params": dict(lane_spec.scalarization_params),
    }


def _lane_reference_point(lane_spec: FrontierLaneSpec) -> dict[str, float] | None:
    metric_key_map = {
        "frontier_reference_iota": "iota",
        "frontier_reference_volume": "volume",
        "frontier_reference_qa": "qa_error",
        "frontier_reference_boozer": "boozer_residual",
    }
    reference_point = {
        metric_name: float(lane_spec.scalarization_params[scalarization_key])
        for scalarization_key, metric_name in metric_key_map.items()
        if scalarization_key in lane_spec.scalarization_params
    }
    return reference_point or None


def build_frontier_lane_contract_for_spec(
    args: argparse.Namespace,
    lane_spec: FrontierLaneSpec,
    *,
    campaign_id: str,
    stage2_bs_path: Path,
    lane_budget: int,
    lane_index: int,
) -> FrontierLaneContract:
    return build_frontier_lane_contract(
        campaign_id=campaign_id,
        lane_id=lane_spec.lane_id,
        engine=args.frontier_engine,
        reference_point=_lane_reference_point(lane_spec),
        scalarization_type=lane_spec.scalarization_type,
        scalarization_params=lane_spec.scalarization_params,
        constraint_mode="frontier_v2_single_lane_contract",
        warm_start_source=str(stage2_bs_path),
        optimizer_budget=lane_budget,
        rng_seed=lane_rng_seed(args.frontier_rng_seed, lane_index=lane_index),
        rerun_contract=build_lane_rerun_contract(
            args,
            lane_spec,
            stage2_bs_path=stage2_bs_path,
        ),
    )


def load_resume_lane_specs(
    manifest_path: Path,
) -> list[FrontierLaneSpec] | None:
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lane_specs_payload = manifest.get("LANE_SPECS")
    if not isinstance(lane_specs_payload, list):
        return None
    return [
        FrontierLaneSpec.from_json_dict(item)
        for item in lane_specs_payload
    ]


def load_resume_manifest(manifest_path: Path) -> dict[str, object] | None:
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def persist_campaign_progress(
    path: Path,
    *,
    campaign_id: str,
    frontier_version: str,
    frontier_engine: str,
    target_payload: dict[str, object] | None,
    lane_records: list[FrontierLaneRecord],
    archive_members: list,
) -> None:
    write_frontier_campaign_progress(
        path,
        FrontierCampaignProgress(
            schema_version="frontier_campaign_progress_v1",
            campaign_id=campaign_id,
            frontier_version=frontier_version,
            frontier_engine=frontier_engine,
            target_payload=serialize_goal_mode_payload(target_payload),
            lane_records=lane_records,
            archive_members=archive_members,
        ),
    )


def build_lane_record_from_payload(
    lane_contract: FrontierLaneContract,
    lane_spec: FrontierLaneSpec,
    lane_budget: int,
    lane_payload: dict[str, object],
    *,
    archive_member=None,
    archive_update: dict[str, object] | None = None,
) -> FrontierLaneRecord:
    return build_frontier_lane_record(
        lane_contract,
        command=lane_payload["command"],
        weights={
            "iotas_weight": lane_spec.iotas_weight,
            "frontier_volume_weight": lane_spec.frontier_volume_weight,
            "res_weight": lane_spec.res_weight,
        },
        lane_budget=lane_budget,
        status=lane_payload["status"],
        result_source=lane_payload.get("result_source"),
        termination_reason=(
            None
            if lane_payload["status"] != "completed"
            or lane_payload["results"].get("TERMINATION_MESSAGE") is None
            else str(lane_payload["results"]["TERMINATION_MESSAGE"])
        ),
        success=(
            None
            if lane_payload["status"] != "completed"
            or lane_payload["results"].get("OPTIMIZER_SUCCESS") is None
            else bool(lane_payload["results"]["OPTIMIZER_SUCCESS"])
        ),
        archive_state=None if archive_member is None else archive_member.archive_state,
        archive_member=archive_member,
        archive_update=archive_update,
        results_path=None
        if lane_payload.get("results_path") is None
        else str(lane_payload["results_path"]),
        results=lane_payload.get("results"),
        error_type=lane_payload.get("error_type"),
        error_message=lane_payload.get("error_message"),
    )


def main() -> int:
    args = parse_args()
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    paths = resolve_frontier_campaign_paths(
        output_root,
        summary_path=summary_path,
    )
    paths.summary_path.parent.mkdir(parents=True, exist_ok=True)

    resumed_progress = None
    resume_manifest = None
    if args.resume:
        resumed_progress = load_frontier_campaign_progress(paths.progress_path)
        resume_manifest = load_resume_manifest(paths.manifest_path)
        args = argparse.Namespace(**vars(args))
        args.frontier_version = resumed_progress.frontier_version
        args.frontier_engine = resumed_progress.frontier_engine
        if resume_manifest is not None:
            seed_artifact_path = resume_manifest.get("SEED_ARTIFACT_PATH")
            if seed_artifact_path is not None:
                args.stage2_bs_path = str(resolved_path(str(seed_artifact_path)))

    if args.dry_run:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_comparison.maybe_load_validated_stage2_seed_metadata(args)
        )
    else:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_comparison.load_validated_stage2_seed_metadata(args)
        )
    resumed_lane_specs = (
        load_resume_lane_specs(paths.manifest_path)
        if args.resume
        else None
    )
    lane_specs = (
        resumed_lane_specs
        if resumed_lane_specs is not None
        else generate_frontier_lane_specs(
            reference_mode=args.frontier_reference_mode,
            num_lanes=args.frontier_num_lanes,
            iotas_weight=args.iotas_weight,
            frontier_volume_weight=args.frontier_volume_weight,
            res_weight=args.res_weight,
            lane_budget=args.frontier_lane_budget,
            stage2_results=stage2_results,
            reference_points_file=args.frontier_reference_points_file,
            epsilon_spec_file=args.frontier_epsilon_spec_file,
        )
    )
    campaign_id = (
        resumed_progress.campaign_id
        if resumed_progress is not None
        else uuid.uuid4().hex[:12]
    )
    manifest = build_frontier_campaign_manifest(
        args,
        campaign_id=campaign_id,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        lane_specs=lane_specs,
    )
    if not args.resume or not paths.manifest_path.exists():
        write_json(paths.manifest_path, manifest)

    target_payload = (
        None
        if resumed_progress is None
        else resumed_progress.target_payload
    )
    lane_records_by_id = (
        {}
        if resumed_progress is None
        else {
            record.lane_contract.lane_id: record
            for record in resumed_progress.lane_records
        }
    )
    archive_members = [] if resumed_progress is None else list(
        resumed_progress.archive_members
    )

    persist_campaign_progress(
        paths.progress_path,
        campaign_id=campaign_id,
        frontier_version=args.frontier_version,
        frontier_engine=args.frontier_engine,
        target_payload=target_payload,
        lane_records=list(lane_records_by_id.values()),
        archive_members=archive_members,
    )

    if target_payload is None and not args.skip_target:
        target_payload = run_goal_mode_case_safe(
            args,
            goal_mode="target",
            stage2_bs_path=stage2_bs_path,
            output_root=output_root / "target_baseline",
        )
        persist_campaign_progress(
            paths.progress_path,
            campaign_id=campaign_id,
            frontier_version=args.frontier_version,
            frontier_engine=args.frontier_engine,
            target_payload=target_payload,
            lane_records=list(lane_records_by_id.values()),
            archive_members=archive_members,
        )

    for lane_index, lane_spec in enumerate(lane_specs):
        if lane_spec.lane_id in lane_records_by_id:
            continue
        lane_args = build_frontier_lane_args(args, lane_spec)
        lane_budget = int(lane_args.maxiter)
        lane_contract = build_frontier_lane_contract_for_spec(
            lane_args,
            lane_spec,
            campaign_id=campaign_id,
            stage2_bs_path=stage2_bs_path,
            lane_budget=lane_budget,
            lane_index=lane_index,
        )
        lane_payload = run_goal_mode_case_safe(
            lane_args,
            goal_mode="frontier",
            stage2_bs_path=stage2_bs_path,
            output_root=output_root / "lanes" / lane_spec.lane_id,
        )
        archive_member = None
        archive_update = None
        if lane_payload["status"] == "completed":
            archive_member = build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id=lane_spec.lane_id,
                payload=lane_payload,
                rerun_contract=lane_contract.rerun_contract,
            )
            archive_members, archive_update = update_frontier_archive(
                archive_members,
                archive_member,
            )
        lane_records_by_id[lane_spec.lane_id] = build_lane_record_from_payload(
            lane_contract,
            lane_spec,
            lane_budget,
            lane_payload,
            archive_member=archive_member,
            archive_update=archive_update,
        )
        persist_campaign_progress(
            paths.progress_path,
            campaign_id=campaign_id,
            frontier_version=args.frontier_version,
            frontier_engine=args.frontier_engine,
            target_payload=target_payload,
            lane_records=list(lane_records_by_id.values()),
            archive_members=archive_members,
        )

    ordered_lane_records = [
        lane_records_by_id[lane_spec.lane_id]
        for lane_spec in lane_specs
        if lane_spec.lane_id in lane_records_by_id
    ]
    lane_record_payloads = [
        lane_record.to_json_dict()
        for lane_record in ordered_lane_records
    ]

    recommendation_payload = None
    if not args.dry_run:
        recommendation_payload = recommend_frontier_member(
            archive_members,
            policy_name=args.frontier_recommendation_policy,
        )

    write_json(
        paths.archive_path,
        serialize_frontier_archive(archive_members),
    )
    write_json(
        paths.recommended_path,
        build_recommended_summary(
            recommendation_payload,
            archive_size=len(archive_members),
            policy_name=args.frontier_recommendation_policy,
        ),
    )

    summary = build_frontier_campaign_summary(
        args,
        campaign_id=campaign_id,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        paths=paths,
        lane_specs=lane_specs,
        target_payload=target_payload,
        lane_records=lane_record_payloads,
        archive_members=archive_members,
        recommendation_payload=recommendation_payload,
        delta_fn=goal_mode_comparison.delta,
    )
    write_json(paths.summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
