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
    FRONTIER_ARCHIVE_STATE_PROVISIONAL,
    build_archive_member_from_results,
    certified_archive_members,
    finalize_archive_member,
    resolve_hypervolume_reference,
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
from banana_opt.frontier_contracts import (  # noqa: E402
    SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES,
    validate_frontier_campaign_summary_payload,
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
from banana_opt.frontier_engine_nsga3 import (  # noqa: E402
    build_nsga3_hypervolume_history,
    load_nsga3_frontier_campaign_artifacts,
    run_nsga3_frontier_campaign,
)
from banana_opt.frontier_scalarization import (  # noqa: E402
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
    FRONTIER_REFERENCE_MODE_EPSILON,
    FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
    FRONTIER_REFERENCE_MODE_SHARED,
    SUPPORTED_FRONTIER_REFERENCE_MODES,
    generate_frontier_lane_specs,
)
from banana_opt.frontier_recommendation import recommend_frontier_member  # noqa: E402
from banana_opt.frontier_dominance import (  # noqa: E402
    PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
    PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
    build_pareto_objective_normalization,
)
from banana_opt.frontier_runtime_calibration import (  # noqa: E402
    FRONTIER_RUNTIME_CALIBRATION_PROFILES,
    build_initial_frontier_early_stop_status,
    effective_lane_budget,
    resolve_frontier_runtime_defaults,
    update_frontier_early_stop_status,
)
from workflow_runner_common import (  # noqa: E402
    discover_single_solver_checkpoint_path,
    resolved_optional_path,
    resolved_path,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_frontier_campaign"
FRONTIER_LANE_WARM_START_MODE_SEED = "seed"
FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED = "reuse_latest_certified"


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
        choices=["multilane_local", "nsga3"],
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
    parser.add_argument(
        "--frontier-full-simplex-partitions",
        type=int,
        default=None,
        help=(
            "Optional Das-Dennis partition count for auto-generated full-simplex "
            "achievement mode. When provided, the full reference-direction family "
            "for that partition count is emitted and --frontier-num-lanes is used "
            "only when partitions are omitted."
        ),
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
        choices=SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES,
        default="balanced",
    )
    parser.add_argument(
        "--frontier-normalization-kind",
        choices=(
            PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
            PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
        ),
        default=PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
    )
    parser.add_argument(
        "--frontier-normalization-spec-file",
        default=None,
        help=(
            "Optional explicit normalization spec JSON. Required for fixed "
            "ideal/nadir normalization and ignored for the default seed-relative kind."
        ),
    )
    parser.add_argument(
        "--frontier-lane-warm-start-mode",
        choices=[
            FRONTIER_LANE_WARM_START_MODE_SEED,
            FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED,
        ],
        default=FRONTIER_LANE_WARM_START_MODE_SEED,
    )
    parser.add_argument("--frontier-rng-seed", type=int, default=0)
    parser.add_argument(
        "--frontier-runtime-calibration-profile",
        choices=sorted(FRONTIER_RUNTIME_CALIBRATION_PROFILES),
        default="reduced_fixture_v1",
    )
    parser.add_argument(
        "--frontier-early-stop-patience-lanes",
        type=int,
        default=None,
        help="Optional no-improvement lane streak before the campaign stops early.",
    )
    parser.add_argument(
        "--frontier-early-stop-min-certified",
        type=int,
        default=None,
        help="Minimum certified archive size before early-stop logic activates.",
    )
    parser.add_argument(
        "--frontier-early-stop-min-hypervolume-gain",
        type=float,
        default=None,
        help="Minimum hypervolume gain required to reset the early-stop patience counter.",
    )
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
    lane_args.frontier_scalarization_type = lane_spec.scalarization_type
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
        "frontier_chebyshev_rho",
        "frontier_chebyshev_sharpness",
        "frontier_chebyshev_weight_iota",
        "frontier_chebyshev_weight_volume",
        "frontier_chebyshev_weight_qa",
        "frontier_chebyshev_weight_boozer",
        "epsilon_constraint_qa_max",
        "epsilon_constraint_boozer_max",
        "frontier_epsilon_penalty_weight",
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
        "constraint_mode": _lane_constraint_mode(lane_spec),
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
    warm_start_source: str,
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
        constraint_mode=_lane_constraint_mode(lane_spec),
        warm_start_source=warm_start_source,
        optimizer_budget=lane_budget,
        rng_seed=lane_rng_seed(args.frontier_rng_seed, lane_index=lane_index),
        rerun_contract=build_lane_rerun_contract(
            args,
            lane_spec,
            stage2_bs_path=stage2_bs_path,
        ),
    )


def resolve_frontier_lane_warm_start(
    *,
    base_stage2_bs_path: Path,
    lane_records_by_id: dict[str, FrontierLaneRecord],
    lane_specs: list[FrontierLaneSpec],
    lane_index: int,
    warm_start_mode: str,
) -> tuple[Path, str]:
    resolved_base_path = Path(base_stage2_bs_path).resolve()
    if warm_start_mode != FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED:
        return resolved_base_path, str(resolved_base_path)
    for candidate_lane_spec in reversed(lane_specs[:lane_index]):
        lane_record = lane_records_by_id.get(candidate_lane_spec.lane_id)
        if lane_record is None or not lane_record.final_certified:
            continue
        warm_start_path = _warm_start_path_from_lane_record(lane_record)
        if warm_start_path is None or not warm_start_path.exists():
            continue
        return warm_start_path.resolve(), str(warm_start_path.resolve())
    return resolved_base_path, str(resolved_base_path)


def _warm_start_path_from_lane_record(
    lane_record: FrontierLaneRecord,
) -> Path | None:
    if lane_record.result_source != "final" or lane_record.results_path is None:
        return None
    results_path = Path(lane_record.results_path)
    return results_path.with_name("biot_savart_opt.json")


def _lane_constraint_mode(lane_spec: FrontierLaneSpec) -> str:
    if lane_spec.scalarization_type == FRONTIER_REFERENCE_MODE_EPSILON:
        return "frontier_epsilon_constraint_v1"
    if lane_spec.scalarization_type == FRONTIER_REFERENCE_MODE_ACHIEVEMENT:
        return "frontier_achievement_chebyshev_v1"
    return "frontier_v2_single_lane_contract"


def maybe_resume_goal_mode_payload_from_artifacts(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    output_root: Path,
) -> dict[str, object] | None:
    case_output_root = output_root / goal_mode
    if not case_output_root.exists():
        return None
    command = goal_mode_comparison.build_single_stage_goal_mode_command(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        case_output_root=case_output_root,
    )
    resumed_results = _load_resumed_results(case_output_root)
    if resumed_results is None:
        return None
    result_source, results_path, results = resumed_results
    return {
        "status": "completed",
        "command": command,
        "results_path": results_path,
        "result_source": result_source,
        "results": results,
        "results_summary": goal_mode_comparison.result_metric_subset(results),
    }


def _load_resumed_results(
    case_output_root: Path,
) -> tuple[str, Path, dict[str, object]] | None:
    try:
        results_path = goal_mode_comparison.discover_single_results_path(
            case_output_root,
        )
        return (
            "final",
            results_path,
            goal_mode_comparison.load_json(results_path),
        )
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        result_source, results_path = (
            goal_mode_comparison.discover_single_stage_salvage_results_path(
                case_output_root,
            )
        )
        return (
            result_source,
            results_path,
            goal_mode_comparison.load_json(results_path),
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def maybe_resume_solver_checkpoint_path(
    output_root: Path,
) -> Path | None:
    try:
        return discover_single_solver_checkpoint_path(output_root)
    except FileNotFoundError:
        return None


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
    provisional_archive_members: list,
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
            provisional_archive_members=provisional_archive_members,
            archive_members=archive_members,
        ),
    )


def build_lane_record_from_payload(
    lane_contract: FrontierLaneContract,
    lane_spec: FrontierLaneSpec,
    lane_budget: int,
    lane_payload: dict[str, object],
    *,
    provisional_archive_member=None,
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
        provisional_archive_member=provisional_archive_member,
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


def resume_or_run_goal_mode_case(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    output_root: Path,
    resume: bool,
) -> dict[str, object]:
    payload = (
        maybe_resume_goal_mode_payload_from_artifacts(
            args,
            goal_mode=goal_mode,
            stage2_bs_path=stage2_bs_path,
            output_root=output_root,
        )
        if resume
        else None
    )
    if payload is not None:
        return payload
    if resume:
        resume_checkpoint = maybe_resume_solver_checkpoint_path(output_root / goal_mode)
        if resume_checkpoint is not None:
            args.resume_solver_checkpoint = str(resume_checkpoint)
    return run_goal_mode_case_safe(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        output_root=output_root,
    )


def main() -> int:
    args = parse_args()
    runtime_defaults = resolve_frontier_runtime_defaults(
        profile_name=args.frontier_runtime_calibration_profile,
        requested_num_lanes=args.frontier_num_lanes,
        requested_lane_budget=args.frontier_lane_budget,
        requested_total_budget=args.frontier_total_budget,
        requested_checkpoint_every=args.checkpoint_every,
        requested_early_stop_patience_lanes=args.frontier_early_stop_patience_lanes,
        requested_early_stop_min_certified=args.frontier_early_stop_min_certified,
        requested_early_stop_min_hypervolume_gain=args.frontier_early_stop_min_hypervolume_gain,
    )
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
        if paths.progress_path.exists():
            resumed_progress = load_frontier_campaign_progress(paths.progress_path)
        resume_manifest = load_resume_manifest(paths.manifest_path)
        args = argparse.Namespace(**vars(args))
        if resumed_progress is not None:
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
            num_lanes=runtime_defaults.num_lanes,
            iotas_weight=args.iotas_weight,
            frontier_volume_weight=args.frontier_volume_weight,
            res_weight=args.res_weight,
            lane_budget=runtime_defaults.lane_budget,
            stage2_results=stage2_results,
            reference_points_file=args.frontier_reference_points_file,
            epsilon_spec_file=args.frontier_epsilon_spec_file,
            full_simplex_partitions=args.frontier_full_simplex_partitions,
        )
    )
    if len(lane_specs) != runtime_defaults.num_lanes:
        runtime_defaults = resolve_frontier_runtime_defaults(
            profile_name=args.frontier_runtime_calibration_profile,
            requested_num_lanes=len(lane_specs),
            requested_lane_budget=args.frontier_lane_budget,
            requested_total_budget=args.frontier_total_budget,
            requested_checkpoint_every=args.checkpoint_every,
            requested_early_stop_patience_lanes=args.frontier_early_stop_patience_lanes,
            requested_early_stop_min_certified=args.frontier_early_stop_min_certified,
            requested_early_stop_min_hypervolume_gain=args.frontier_early_stop_min_hypervolume_gain,
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
        runtime_defaults=runtime_defaults,
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
    provisional_archive_members = [] if resumed_progress is None else list(
        resumed_progress.provisional_archive_members
    )
    hypervolume_reference = resolve_hypervolume_reference(
        reference_spec=args.frontier_hypervolume_reference,
        seed_results=stage2_results,
        members=archive_members,
    )
    pareto_objective_normalization = build_pareto_objective_normalization(
        hypervolume_reference,
        kind=args.frontier_normalization_kind,
        normalization_spec_path=args.frontier_normalization_spec_file,
    )
    early_stop_status = build_initial_frontier_early_stop_status(
        runtime_defaults=runtime_defaults,
        archive_members=archive_members,
    )

    def persist_progress() -> None:
        persist_campaign_progress(
            paths.progress_path,
            campaign_id=campaign_id,
            frontier_version=args.frontier_version,
            frontier_engine=args.frontier_engine,
            target_payload=target_payload,
            lane_records=list(lane_records_by_id.values()),
            provisional_archive_members=provisional_archive_members,
            archive_members=archive_members,
        )

    persist_progress()

    if (target_payload is None or target_payload.get("status") != "completed") and not args.skip_target:
        target_args = argparse.Namespace(**vars(args))
        target_args.checkpoint_every = int(runtime_defaults.checkpoint_every)
        target_payload = resume_or_run_goal_mode_case(
            target_args,
            goal_mode="target",
            stage2_bs_path=stage2_bs_path,
            output_root=output_root / "target_baseline",
            resume=args.resume,
        )
        persist_progress()

    engine_artifacts = None
    if args.frontier_engine == "nsga3":
        loaded_nsga3_artifacts = False
        if args.resume and resumed_progress is not None:
            engine_artifacts = load_nsga3_frontier_campaign_artifacts(
                output_root=output_root,
                archive_members=archive_members,
                provisional_archive_members=provisional_archive_members,
            )
            loaded_nsga3_artifacts = engine_artifacts is not None
        if engine_artifacts is None:
            engine_artifacts = run_nsga3_frontier_campaign(
                args,
                campaign_id=campaign_id,
                output_root=output_root,
                stage2_bs_path=stage2_bs_path,
                stage2_results_path=stage2_results_path,
                stage2_results=stage2_results,
                hypervolume_reference=hypervolume_reference,
                pareto_objective_normalization=pareto_objective_normalization,
                total_budget=runtime_defaults.total_budget,
            )
        provisional_archive_members = list(
            engine_artifacts.provisional_archive_members
        )
        archive_members = list(engine_artifacts.archive_members)
        if not loaded_nsga3_artifacts:
            persist_progress()
    else:
        for lane_index, lane_spec in enumerate(lane_specs):
            existing_lane_record = lane_records_by_id.get(lane_spec.lane_id)
            if (
                existing_lane_record is not None
                and existing_lane_record.status == "completed"
            ):
                continue
            lane_args = build_frontier_lane_args(args, lane_spec)
            lane_budget = effective_lane_budget(
                lane_spec.lane_budget,
                runtime_defaults,
            )
            lane_args.maxiter = int(lane_budget)
            lane_args.checkpoint_every = int(runtime_defaults.checkpoint_every)
            lane_stage2_bs_path, warm_start_source = resolve_frontier_lane_warm_start(
                base_stage2_bs_path=stage2_bs_path,
                lane_records_by_id=lane_records_by_id,
                lane_specs=lane_specs,
                lane_index=lane_index,
                warm_start_mode=args.frontier_lane_warm_start_mode,
            )
            lane_contract = build_frontier_lane_contract_for_spec(
                lane_args,
                lane_spec,
                campaign_id=campaign_id,
                stage2_bs_path=lane_stage2_bs_path,
                warm_start_source=warm_start_source,
                lane_budget=lane_budget,
                lane_index=lane_index,
            )
            lane_output_root = output_root / "lanes" / lane_spec.lane_id
            lane_payload = resume_or_run_goal_mode_case(
                lane_args,
                goal_mode="frontier",
                stage2_bs_path=lane_stage2_bs_path,
                output_root=lane_output_root,
                resume=args.resume,
            )
            provisional_archive_member = None
            archive_member = None
            archive_update = None
            if lane_payload["status"] == "completed":
                provisional_archive_member = build_archive_member_from_results(
                    campaign_id=campaign_id,
                    lane_id=lane_spec.lane_id,
                    payload=lane_payload,
                    rerun_contract=lane_contract.rerun_contract,
                    archive_state=FRONTIER_ARCHIVE_STATE_PROVISIONAL,
                    pareto_objective_normalization=pareto_objective_normalization,
                )
                provisional_archive_members.append(provisional_archive_member)
                archive_member = finalize_archive_member(provisional_archive_member)
                archive_members, archive_update = update_frontier_archive(
                    archive_members,
                    archive_member,
                    pareto_objective_normalization=pareto_objective_normalization,
                )
            lane_records_by_id[lane_spec.lane_id] = build_lane_record_from_payload(
                lane_contract,
                lane_spec,
                lane_budget,
                lane_payload,
                provisional_archive_member=provisional_archive_member,
                archive_member=archive_member,
                archive_update=archive_update,
            )
            persist_progress()
            early_stop_status = update_frontier_early_stop_status(
                status=early_stop_status,
                certified_archive_members_list=archive_members,
                hypervolume_reference=hypervolume_reference,
                runtime_defaults=runtime_defaults,
            )
            if early_stop_status["triggered"]:
                early_stop_status["stopped_after_lane_id"] = lane_spec.lane_id
                break

    ordered_lane_records = [
        lane_records_by_id[lane_spec.lane_id]
        for lane_spec in lane_specs
        if lane_spec.lane_id in lane_records_by_id
    ]
    lane_record_payloads = [
        lane_record.to_json_dict()
        for lane_record in ordered_lane_records
    ]
    certified_members = certified_archive_members(archive_members)

    recommendation_payload = None
    if not args.dry_run:
        recommendation_payload = recommend_frontier_member(
            certified_members,
            policy_name=args.frontier_recommendation_policy,
            pareto_objective_normalization=pareto_objective_normalization,
        )

    hypervolume_reference = resolve_hypervolume_reference(
        reference_spec=args.frontier_hypervolume_reference,
        seed_results=stage2_results,
        members=certified_members,
    )
    write_json(
        paths.archive_path,
        serialize_frontier_archive(
            certified_members,
            hypervolume_reference=hypervolume_reference,
        ),
    )
    write_json(
        paths.recommended_path,
        build_recommended_summary(
            recommendation_payload,
            archive_size=len(certified_members),
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
        archive_members=certified_members,
        recommendation_payload=recommendation_payload,
        delta_fn=goal_mode_comparison.delta,
        runtime_defaults=runtime_defaults,
        early_stop_status=early_stop_status,
    )
    if engine_artifacts is not None:
        summary["frontier_generation_history"] = list(
            engine_artifacts.generation_history
        )
        summary["frontier_hypervolume_history"] = build_nsga3_hypervolume_history(
            engine_artifacts.generation_history
        )
        if engine_artifacts.generation_history:
            summary["frontier_feasible_lane_count"] = int(
                engine_artifacts.generation_history[-1]["feasible_count"]
            )
        summary["frontier_engine_stats"] = dict(engine_artifacts.engine_stats)
        summary["frontier_evaluator_spec"] = dict(engine_artifacts.evaluator_spec)
        summary["frontier_evaluator_spec_path"] = (
            engine_artifacts.evaluator_spec_path
        )
        summary["frontier_population_checkpoint_path"] = (
            engine_artifacts.population_checkpoint_path
        )
        summary["frontier_generation_history_path"] = (
            engine_artifacts.generation_history_path
        )
        validate_frontier_campaign_summary_payload(summary)
    write_json(paths.summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
