from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_single_stage_goal_mode_comparison as goal_mode_comparison  # noqa: E402
from banana_opt.frontier_archive import (  # noqa: E402
    DEFAULT_DUPLICATE_DISTANCE_THRESHOLD,
    archive_best_by_metric,
    build_archive_member_from_results,
    serialize_frontier_archive,
    update_frontier_archive,
)
from banana_opt.frontier_dominance import (  # noqa: E402
    DEFAULT_DOMINANCE_TOLERANCE,
    PARETO_OBJECTIVE_SPECS,
)
from banana_opt.frontier_engine_multilane_local import (  # noqa: E402
    FrontierLaneSpec,
    generate_multilane_local_specs,
)
from banana_opt.frontier_recommendation import recommend_frontier_member  # noqa: E402
from workflow_runner_common import resolved_optional_path, resolved_path  # noqa: E402

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_frontier_campaign"
DEFAULT_SUMMARY_JSON = "single_stage_frontier_campaign_summary.json"
DEFAULT_MANIFEST_JSON = "campaign_manifest.json"
DEFAULT_ARCHIVE_JSON = "frontier_archive.json"
DEFAULT_RECOMMENDED_JSON = "frontier_recommended.json"


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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_campaign_manifest(
    args: argparse.Namespace,
    *,
    campaign_id: str,
    stage2_bs_path: Path,
    stage2_results_path: Path | None,
    stage2_results: dict | None,
    lane_specs: list[FrontierLaneSpec],
) -> dict[str, object]:
    lane_budget = args.frontier_lane_budget if args.frontier_lane_budget is not None else args.maxiter
    total_budget = (
        args.frontier_total_budget
        if args.frontier_total_budget is not None
        else int(args.frontier_num_lanes) * int(lane_budget)
    )
    return {
        "FRONTIER_VERSION": args.frontier_version,
        "FRONTIER_ENGINE": args.frontier_engine,
        "FRONTIER_CAMPAIGN_ID": campaign_id,
        "SEED_ARTIFACT_PATH": str(stage2_bs_path),
        "SEED_RESULTS_PATH": None if stage2_results_path is None else str(stage2_results_path),
        "SEED_SURFACE_IDENTITY": Path(args.plasma_surf_filename).name,
        "FRONTIER_REFERENCE_MODE": "shared_seed_relative_frontier_v2",
        "FRONTIER_REFERENCE_POINTS": [
            lane.scalarization_params for lane in lane_specs
        ],
        "FRONTIER_SCALARIZATION_FAMILY": "weight_schedule_v1",
        "FRONTIER_CONSTRAINT_MODE": "frontier_v2_single_lane_contract",
        "PARETO_OBJECTIVE_VECTOR": [
            {"metric": metric_name, "direction": direction}
            for metric_name, direction, _ in PARETO_OBJECTIVE_SPECS
        ],
        "PARETO_OBJECTIVE_NORMALIZATION": {
            "kind": "frontier_v2_seed_relative",
        },
        "DOMINANCE_TOLERANCE": dict(DEFAULT_DOMINANCE_TOLERANCE),
        "DUPLICATE_DISTANCE_THRESHOLD": DEFAULT_DUPLICATE_DISTANCE_THRESHOLD,
        "FRONTIER_RECOMMENDATION_POLICY": args.frontier_recommendation_policy,
        "LANE_BUDGET": lane_budget,
        "TOTAL_BUDGET": total_budget,
        "RNG_SEED": args.frontier_rng_seed,
        "CREATED_AT": datetime.now(timezone.utc).isoformat(),
        "STAGE2_ARTIFACT_INIT_ONLY": None if stage2_results is None else stage2_results.get("init_only"),
        "LANE_SPECS": [lane.to_json_dict() for lane in lane_specs],
    }


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
    return {
        "status": "completed",
        **payload,
    }


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
    return lane_args


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
        "maxiter": args.maxiter if lane_spec.lane_budget is None else lane_spec.lane_budget,
        "constraint_method": args.constraint_method,
        "hardware_search_mode": args.hardware_search_mode,
        "scalarization_type": lane_spec.scalarization_type,
        "scalarization_params": dict(lane_spec.scalarization_params),
    }


def build_target_comparison(
    target_payload: dict[str, object] | None,
    recommended_payload: dict[str, object] | None,
) -> dict[str, object] | None:
    if target_payload is None or target_payload.get("status") != "completed":
        return None
    if recommended_payload is None:
        return None
    target_results = target_payload["results"]
    recommended_member = recommended_payload["recommended_member"]
    _delta = goal_mode_comparison.delta
    return {
        "recommended_minus_target_final_iota": _delta(
            recommended_member.objective_metrics.get("iota"),
            target_results.get("FINAL_IOTA"),
        ),
        "recommended_minus_target_final_volume": _delta(
            recommended_member.objective_metrics.get("volume"),
            target_results.get("FINAL_VOLUME"),
        ),
        "recommended_minus_target_nonqs_ratio": _delta(
            recommended_member.objective_metrics.get("qa_error"),
            target_results.get("NONQS_RATIO"),
        ),
        "recommended_minus_target_boozer_residual": _delta(
            recommended_member.objective_metrics.get("boozer_residual"),
            target_results.get("BOOZER_RESIDUAL"),
        ),
    }


def _build_completed_case_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "results_path": str(payload["results_path"]),
        "result_source": payload["result_source"],
        "results": goal_mode_comparison.result_metric_subset(payload["results"]),
    }


def _build_recommended_summary(
    recommendation_payload: dict[str, object] | None,
    *,
    archive_size: int,
    policy_name: str,
) -> dict[str, object]:
    if recommendation_payload is None:
        return {
            "recommended_member_id": None,
            "policy_name": policy_name,
            "policy_inputs": None,
            "policy_rationale": None,
            "policy_score": None,
            "recommended_metrics": None,
            "frontier_archive_size": archive_size,
        }

    recommended_member = recommendation_payload["recommended_member"]
    return {
        "recommended_member_id": recommended_member.member_id,
        "policy_name": recommendation_payload["policy_name"],
        "policy_inputs": recommendation_payload["policy_inputs"],
        "policy_rationale": recommendation_payload["policy_rationale"],
        "policy_score": recommendation_payload["policy_score"],
        "recommended_metrics": dict(recommended_member.objective_metrics),
        "frontier_archive_size": archive_size,
    }


def build_frontier_campaign_summary(
    args: argparse.Namespace,
    *,
    campaign_id: str,
    stage2_bs_path: Path,
    stage2_results_path: Path | None,
    stage2_results: dict | None,
    manifest_path: Path,
    archive_path: Path,
    recommended_path: Path,
    lane_specs: list[FrontierLaneSpec],
    target_payload: dict[str, object] | None,
    lane_records: list[dict[str, object]],
    archive_members,
    recommendation_payload: dict[str, object] | None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "frontier_version": args.frontier_version,
        "frontier_engine": args.frontier_engine,
        "frontier_campaign_id": campaign_id,
        "dry_run": bool(args.dry_run),
        "output_root": str(resolved_path(args.output_root)),
        "manifest_path": str(manifest_path),
        "archive_path": str(archive_path),
        "recommended_path": str(recommended_path),
        "stage2_bs_path": str(stage2_bs_path),
        "stage2_results_path": None if stage2_results_path is None else str(stage2_results_path),
        "stage2_artifact_init_only": None if stage2_results is None else stage2_results.get("init_only"),
        "frontier_num_lanes": len(lane_specs),
        "frontier_lane_specs": [lane.to_json_dict() for lane in lane_specs],
        "frontier_lanes": lane_records,
        "frontier_archive": serialize_frontier_archive(archive_members),
        "frontier_archive_size": len(archive_members),
        "frontier_archive_best_by_metric": archive_best_by_metric(archive_members),
        "target_run": None,
        "recommended_member": _build_recommended_summary(
            recommendation_payload,
            archive_size=len(archive_members),
            policy_name=args.frontier_recommendation_policy,
        ),
        "target_comparison": None,
    }
    if target_payload is not None:
        target_summary = {
            "status": target_payload["status"],
            "command": target_payload["command"],
        }
        if target_payload["status"] == "completed":
            target_summary.update(_build_completed_case_summary(target_payload))
        elif target_payload["status"] == "failed":
            target_summary["error_type"] = target_payload.get("error_type")
            target_summary["error_message"] = target_payload.get("error_message")
        summary["target_run"] = target_summary
    if recommendation_payload is not None:
        summary["target_comparison"] = build_target_comparison(
            target_payload,
            recommendation_payload,
        )
    return summary


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def main() -> int:
    args = parse_args()
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    manifest_path = output_root / DEFAULT_MANIFEST_JSON
    archive_path = output_root / DEFAULT_ARCHIVE_JSON
    recommended_path = output_root / DEFAULT_RECOMMENDED_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_comparison.maybe_load_validated_stage2_seed_metadata(args)
        )
    else:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_comparison.load_validated_stage2_seed_metadata(args)
        )

    lane_specs = generate_multilane_local_specs(
        num_lanes=args.frontier_num_lanes,
        iotas_weight=args.iotas_weight,
        frontier_volume_weight=args.frontier_volume_weight,
        res_weight=args.res_weight,
        lane_budget=args.frontier_lane_budget,
    )
    campaign_id = uuid.uuid4().hex[:12]
    manifest = build_campaign_manifest(
        args,
        campaign_id=campaign_id,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        lane_specs=lane_specs,
    )
    _write_json(manifest_path, manifest)

    target_payload = None
    if not args.skip_target:
        target_output_root = output_root / "target_baseline"
        target_payload = run_goal_mode_case_safe(
            args,
            goal_mode="target",
            stage2_bs_path=stage2_bs_path,
            output_root=target_output_root,
        )

    archive_members = []
    lane_records: list[dict[str, object]] = []
    for lane_spec in lane_specs:
        lane_args = build_frontier_lane_args(args, lane_spec)
        lane_output_root = output_root / "lanes" / lane_spec.lane_id
        lane_payload = run_goal_mode_case_safe(
            lane_args,
            goal_mode="frontier",
            stage2_bs_path=stage2_bs_path,
            output_root=lane_output_root,
        )
        lane_record: dict[str, object] = {
            "lane_id": lane_spec.lane_id,
            "status": lane_payload["status"],
            "command": lane_payload["command"],
            "scalarization_type": lane_spec.scalarization_type,
            "scalarization_params": dict(lane_spec.scalarization_params),
            "weights": {
                "iotas_weight": lane_spec.iotas_weight,
                "frontier_volume_weight": lane_spec.frontier_volume_weight,
                "res_weight": lane_spec.res_weight,
            },
            "lane_budget": int(lane_args.maxiter),
        }
        if lane_payload["status"] == "completed":
            lane_record.update(_build_completed_case_summary(lane_payload))
            member = build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id=lane_spec.lane_id,
                payload=lane_payload,
                rerun_contract=build_lane_rerun_contract(
                    lane_args,
                    lane_spec,
                    stage2_bs_path=stage2_bs_path,
                ),
            )
            lane_record["archive_state"] = member.archive_state
            lane_record["archive_member_id"] = member.member_id
            archive_members, archive_update = update_frontier_archive(
                archive_members,
                member,
            )
            lane_record["archive_update"] = archive_update
        elif lane_payload["status"] == "failed":
            lane_record["error_type"] = lane_payload["error_type"]
            lane_record["error_message"] = lane_payload["error_message"]
        lane_records.append(lane_record)

    recommendation_payload = None
    if not args.dry_run:
        recommendation_payload = recommend_frontier_member(
            archive_members,
            policy_name=args.frontier_recommendation_policy,
        )

    archive_payload = serialize_frontier_archive(archive_members)
    _write_json(archive_path, archive_payload)

    recommended_payload = _build_recommended_summary(
        recommendation_payload,
        archive_size=len(archive_members),
        policy_name=args.frontier_recommendation_policy,
    )
    _write_json(recommended_path, recommended_payload)

    summary = build_frontier_campaign_summary(
        args,
        campaign_id=campaign_id,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        manifest_path=manifest_path,
        archive_path=archive_path,
        recommended_path=recommended_path,
        lane_specs=lane_specs,
        target_payload=target_payload,
        lane_records=lane_records,
        archive_members=archive_members,
        recommendation_payload=recommendation_payload,
    )
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
