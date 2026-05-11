from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
    HYPERVOLUME_REFERENCE_AUTO_SEED_SENTINEL,
    build_archive_member_from_results,
    certified_archive_members,
    finalize_archive_member,
    parse_hypervolume_reference,
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
from banana_opt.frontier_campaign_execution import (  # noqa: E402
    FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED,
    FRONTIER_LANE_WARM_START_MODE_SEED,
    FrontierLaneExecution,
    FrontierLaneExecutionResult,
    build_frontier_lane_execution,
    build_frontier_lane_execution_groups,
)
from banana_opt.frontier_contracts import (  # noqa: E402
    FRONTIER_CAMPAIGN_PROGRESS_SCHEMA_VERSION,
    SUPPORTED_FRONTIER_ENGINES,
    SUPPORTED_FRONTIER_RECOMMENDATION_POLICIES,
    validate_frontier_campaign_manifest_payload,
)
from banana_opt.frontier_progress_state import (  # noqa: E402
    FrontierCampaignProgress,
    FrontierLaneContract,
    FrontierLaneRecord,
    build_frontier_lane_record,
    load_frontier_campaign_progress,
    serialize_goal_mode_payload,
    write_frontier_campaign_progress,
)
from banana_opt.frontier_scalarization import (  # noqa: E402
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX,
    FRONTIER_REFERENCE_MODE_EPSILON,
    FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
    FRONTIER_REFERENCE_MODE_SHARED,
    FrontierLaneSpec,
    SUPPORTED_FRONTIER_REFERENCE_MODES,
    generate_frontier_lane_specs,
)
from banana_opt.frontier_recommendation import recommend_frontier_member  # noqa: E402
from banana_opt.frontier_dominance import (  # noqa: E402
    PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
    PARETO_OBJECTIVE_NORMALIZATION_KIND_SEED_RELATIVE,
    build_pareto_objective_normalization,
    build_pareto_objective_normalization_from_persisted_payload,
)
from banana_opt.frontier_runtime_calibration import (  # noqa: E402
    SUPPORTED_FRONTIER_RUNTIME_CALIBRATION_PROFILES,
    build_initial_frontier_early_stop_status,
    resolve_frontier_runtime_defaults_from_args,
    update_frontier_early_stop_status,
)
from workflow_runner_common import (  # noqa: E402
    discover_single_solver_checkpoint_path,
    resolved_optional_path,
    resolved_path,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_frontier_campaign"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


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
        choices=SUPPORTED_FRONTIER_ENGINES,
        default="multilane_local",
        help="Local frontier lane runner; NSGA3/global engines are not supported.",
    )
    parser.add_argument(
        "--frontier-reference-mode",
        choices=SUPPORTED_FRONTIER_REFERENCE_MODES,
        default=FRONTIER_REFERENCE_MODE_SHARED,
        help=(
            "Lane reference family. The default shared mode is the legacy "
            "iota/volume share sweep; choose achievement_chebyshev_full_simplex_v1 "
            "for generated 4-objective reference directions."
        ),
    )
    parser.add_argument(
        "--frontier-hypervolume-reference",
        default=None,
        help=(
            "REQUIRED. The campaign-wide hypervolume reference is a "
            "campaign-level IMMUTABLE INVARIANT — operators must commit "
            "to it explicitly. Accepted forms:\n"
            "  * 'auto:seed' (explicit opt-in to the stage-2 seed metrics; "
            "    commits the campaign to the semantics 'no certified lane "
            "    regresses vs the seed on any Pareto axis' — if any axis "
            "    regresses, archive serialization raises and the operator "
            "    must rerun with explicit values)\n"
            "  * Comma list of 4 floats in iota,volume,qa_error,boozer_residual "
            "    order, e.g. '0.10,0.08,0.020,0.015'\n"
            "  * key=val list, e.g. "
            "    'iota=0.10,volume=0.08,qa_error=0.020,boozer_residual=0.015'\n"
            "  * JSON object form with the same keys"
        ),
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
        type=_positive_int,
        default=None,
        help=(
            "Optional Das-Dennis partition count for auto-generated full-simplex "
            "achievement mode. When provided, the full reference-direction family "
            "for that partition count is emitted and --frontier-num-lanes is used "
            "only when partitions are omitted."
        ),
    )
    parser.add_argument("--frontier-num-lanes", type=_positive_int, default=3)
    parser.add_argument(
        "--frontier-lane-budget",
        type=_positive_int,
        default=None,
        help="Optional per-lane maxiter override for frontier lanes.",
    )
    parser.add_argument(
        "--frontier-total-budget",
        type=_positive_int,
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
    parser.add_argument(
        "--frontier-lane-workers",
        type=_positive_int,
        default=1,
        help=(
            "Maximum local worker threads for independent frontier lane groups. "
            "Values above 1 only apply when lane warm starts and early-stop state "
            "do not create inter-lane dependencies."
        ),
    )
    parser.add_argument("--frontier-rng-seed", type=int, default=0)
    parser.add_argument(
        "--frontier-runtime-calibration-profile",
        choices=sorted(SUPPORTED_FRONTIER_RUNTIME_CALIBRATION_PROFILES),
        default="reduced_fixture_v1",
    )
    parser.add_argument(
        "--frontier-early-stop-patience-lanes",
        type=_non_negative_int,
        default=None,
        help="Optional no-improvement lane streak before the campaign stops early.",
    )
    parser.add_argument(
        "--frontier-early-stop-min-certified",
        type=_non_negative_int,
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
    parser.add_argument(
        "--allow-resume-arg-drift",
        action="store_true",
        help=(
            "Permit CLI flags to override the persisted manifest on resume. "
            "Default behavior restores manifest-authoritative fields silently so "
            "the resumed campaign honors the original contract; with this flag "
            "set, the user-supplied CLI values win and the manifest is rewritten "
            "to match. --frontier-lane-warm-start-mode and --frontier-lane-workers "
            "are runtime-only and not part of the manifest contract; supply them "
            "on resume as needed."
        ),
    )
    return parser


_REFERENCE_MODE_REQUIRED_FLAGS: dict[str, frozenset[str]] = {
    FRONTIER_REFERENCE_MODE_REFERENCE_POINTS: frozenset(
        {"frontier_reference_points_file"}
    ),
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT: frozenset(
        {"frontier_reference_points_file"}
    ),
    FRONTIER_REFERENCE_MODE_EPSILON: frozenset({"frontier_epsilon_spec_file"}),
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX: frozenset(
        {"frontier_full_simplex_partitions"}
    ),
}
_REFERENCE_MODE_FLAG_BLAME: dict[str, str] = {
    "frontier_reference_points_file": "--frontier-reference-points-file",
    "frontier_epsilon_spec_file": "--frontier-epsilon-spec-file",
    "frontier_full_simplex_partitions": "--frontier-full-simplex-partitions",
}


def _modes_for_attribute(attribute: str) -> set[str]:
    return {
        mode
        for mode, required in _REFERENCE_MODE_REQUIRED_FLAGS.items()
        if attribute in required
    }


def _validate_reference_mode_flags(args: argparse.Namespace) -> None:
    reference_mode = args.frontier_reference_mode
    expected_flags = _REFERENCE_MODE_REQUIRED_FLAGS.get(reference_mode, frozenset())
    for attribute, cli_flag in _REFERENCE_MODE_FLAG_BLAME.items():
        if getattr(args, attribute, None) is None:
            continue
        if attribute in expected_flags:
            continue
        raise argparse.ArgumentTypeError(
            f"{cli_flag} is only valid with "
            f"--frontier-reference-mode in "
            f"{sorted(_modes_for_attribute(attribute))!r}; got {reference_mode!r}"
        )


def _require_hypervolume_reference(args: argparse.Namespace) -> None:
    """Enforce that --frontier-hypervolume-reference is supplied explicitly.

    The campaign-wide hypervolume reference is a campaign-level IMMUTABLE
    INVARIANT: there is no implicit default. Operators must either
    commit to explicit metric values or opt in to the seed-derived
    reference via the ``auto:seed`` sentinel. Failing here keeps the
    ambiguous "seed silently wins" behavior out of every downstream
    artifact (manifest, archive, summary, early-stop snapshot).

    Also normalizes the spec by parsing it once so that malformed
    explicit specs (e.g. wrong cardinality, bad JSON) surface at
    argument parse time rather than after stage-2 seed loading.
    """
    reference_spec = args.frontier_hypervolume_reference
    if reference_spec is None:
        raise argparse.ArgumentTypeError(
            "--frontier-hypervolume-reference is required. Pass explicit "
            "metric values (e.g. '0.10,0.08,0.020,0.015' or "
            "'iota=0.10,volume=0.08,qa_error=0.020,boozer_residual=0.015') or "
            f"the {HYPERVOLUME_REFERENCE_AUTO_SEED_SENTINEL!r} sentinel to "
            "opt in to the seed-derived reference."
        )
    if str(reference_spec).strip() == HYPERVOLUME_REFERENCE_AUTO_SEED_SENTINEL:
        return
    parse_hypervolume_reference(reference_spec)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    _validate_reference_mode_flags(args)
    _require_hypervolume_reference(args)
    return args


_LOGGER = logging.getLogger("frontier_campaign")


def _log_lane_workers_overridden_to_sequential(
    *,
    args: argparse.Namespace,
    lane_execution_groups: list[list[tuple[int, FrontierLaneSpec]]],
    runtime_defaults,
) -> None:
    if int(args.frontier_lane_workers) <= 1:
        return
    if not lane_execution_groups:
        return
    if any(len(group) > 1 for group in lane_execution_groups):
        return
    reasons: list[str] = []
    if args.frontier_lane_warm_start_mode == FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED:
        reasons.append("warm_start_mode=reuse_latest_certified")
    if int(runtime_defaults.early_stop_patience_lanes) > 0:
        reasons.append(
            f"early_stop_patience_lanes={runtime_defaults.early_stop_patience_lanes}"
        )
    if not reasons:
        # frontier_lanes_require_ordered_execution returned True only when
        # lane_workers itself is 1 — that path is excluded above.
        reasons.append("unknown_dependency_predicate")
    _LOGGER.warning(
        "frontier_lane_workers=%d but lane execution groups serialized to size 1; "
        "ordered execution forced by: %s",
        int(args.frontier_lane_workers),
        ", ".join(reasons),
    )


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


def run_frontier_lane_execution(
    execution: FrontierLaneExecution,
    *,
    resume: bool,
) -> FrontierLaneExecutionResult:
    lane_payload = resume_or_run_goal_mode_case(
        execution.lane_args,
        goal_mode="frontier",
        stage2_bs_path=execution.stage2_bs_path,
        output_root=execution.output_root,
        resume=resume,
    )
    return FrontierLaneExecutionResult(
        execution=execution,
        lane_payload=lane_payload,
    )


def run_frontier_lane_execution_group(
    executions: list[FrontierLaneExecution],
    *,
    resume: bool,
    lane_workers: int,
) -> list[FrontierLaneExecutionResult]:
    if len(executions) == 1 or int(lane_workers) == 1:
        return [
            run_frontier_lane_execution(execution, resume=resume)
            for execution in executions
        ]
    with ThreadPoolExecutor(
        max_workers=min(int(lane_workers), len(executions))
    ) as executor:
        run_lane = partial(run_frontier_lane_execution, resume=resume)
        return list(executor.map(run_lane, executions))


def _resume_payload(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    case_output_root: Path,
    result_source: str,
    results_path: Path,
    results: dict[str, object],
) -> dict[str, object]:
    command = goal_mode_comparison.build_single_stage_goal_mode_command(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        case_output_root=case_output_root,
    )
    return {
        "status": "completed",
        "command": command,
        "results_path": results_path,
        "result_source": result_source,
        "results": results,
        "results_summary": goal_mode_comparison.result_metric_subset(results),
    }


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
    final_results = _load_final_results(case_output_root)
    if final_results is not None:
        result_source, results_path, results = final_results
        return _resume_payload(
            args,
            goal_mode=goal_mode,
            stage2_bs_path=stage2_bs_path,
            case_output_root=case_output_root,
            result_source=result_source,
            results_path=results_path,
            results=results,
        )
    salvage_results = _load_salvage_results(case_output_root)
    if salvage_results is None:
        return None
    result_source, results_path, results = salvage_results
    return _resume_payload(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        case_output_root=case_output_root,
        result_source=result_source,
        results_path=results_path,
        results=results,
    )


def _load_final_results(
    case_output_root: Path,
) -> tuple[str, Path, dict[str, object]] | None:
    try:
        results_path = goal_mode_comparison.discover_single_results_path(
            case_output_root,
        )
    except FileNotFoundError:
        return None
    try:
        return (
            "final",
            results_path,
            goal_mode_comparison.load_json(results_path),
        )
    except json.JSONDecodeError:
        return None


def _load_salvage_results(
    case_output_root: Path,
) -> tuple[str, Path, dict[str, object]] | None:
    try:
        result_source, results_path = (
            goal_mode_comparison.discover_single_stage_salvage_results_path(
                case_output_root,
            )
        )
    except FileNotFoundError:
        return None
    try:
        return (
            result_source,
            results_path,
            goal_mode_comparison.load_json(results_path),
        )
    except json.JSONDecodeError:
        return None


def maybe_resume_solver_checkpoint_path(
    output_root: Path,
) -> Path | None:
    try:
        return discover_single_solver_checkpoint_path(output_root)
    except FileNotFoundError:
        return None


def resume_lane_specs_from_manifest(
    manifest: dict[str, object] | None,
) -> list[FrontierLaneSpec] | None:
    if manifest is None:
        return None
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frontier_campaign_manifest_payload(manifest)
    return manifest


def _apply_manifest_overrides_to_args(
    args: argparse.Namespace,
    manifest: dict[str, object],
) -> argparse.Namespace:
    """Overlay manifest-authoritative fields onto args.

    Returns a new ``argparse.Namespace`` so the original args object is not
    mutated. Restores every field that is part of the persisted manifest
    contract:

    - ``FRONTIER_RUNTIME_CALIBRATION.profile.profile_name`` →
      ``frontier_runtime_calibration_profile``.
    - ``FRONTIER_RUNTIME_CALIBRATION.resolved_defaults.lane_budget`` /
      ``total_budget`` → ``frontier_lane_budget`` / ``frontier_total_budget``.
    - ``FRONTIER_EARLY_STOP_POLICY.{patience_lanes, min_certified,
      min_hypervolume_gain}`` → matching ``frontier_early_stop_*`` fields.
    - ``RNG_SEED`` → ``frontier_rng_seed``.
    - ``FRONTIER_HYPERVOLUME_REFERENCE`` → ``frontier_hypervolume_reference``.
    - ``PARETO_OBJECTIVE_NORMALIZATION.kind`` →
      ``frontier_normalization_kind``.
    - ``FRONTIER_RECOMMENDATION_POLICY`` →
      ``frontier_recommendation_policy``.
    - ``FRONTIER_REFERENCE_MODE`` → ``frontier_reference_mode``.
    - ``FRONTIER_REFERENCE_POINTS_FILE`` /
      ``FRONTIER_EPSILON_SPEC_FILE`` → matching ``frontier_*_file`` fields.
    - ``FRONTIER_LANE_WARM_START_MODE`` → ``frontier_lane_warm_start_mode``.

    Skips ``--frontier-lane-workers``: not part of the manifest contract
    (runtime-only setting; the user can adjust thread count without
    triggering arg-drift). ``--frontier-full-simplex-partitions`` is
    captured indirectly via ``LANE_SPECS`` (handled by
    ``resume_lane_specs_from_manifest``).
    """

    runtime_calibration = manifest["FRONTIER_RUNTIME_CALIBRATION"]
    profile = runtime_calibration["profile"]
    resolved = runtime_calibration["resolved_defaults"]
    early_stop = manifest["FRONTIER_EARLY_STOP_POLICY"]
    normalization = manifest["PARETO_OBJECTIVE_NORMALIZATION"]

    overrides: dict[str, object] = {
        "frontier_runtime_calibration_profile": profile["profile_name"],
        "frontier_lane_budget": int(resolved["lane_budget"]),
        "frontier_total_budget": int(resolved["total_budget"]),
        "frontier_early_stop_patience_lanes": int(early_stop["patience_lanes"]),
        "frontier_early_stop_min_certified": int(early_stop["min_certified"]),
        "frontier_early_stop_min_hypervolume_gain": float(
            early_stop["min_hypervolume_gain"]
        ),
        "frontier_rng_seed": int(manifest["RNG_SEED"]),
        "frontier_hypervolume_reference": manifest["FRONTIER_HYPERVOLUME_REFERENCE"],
        "frontier_normalization_kind": str(normalization["kind"]),
        "frontier_recommendation_policy": str(
            manifest["FRONTIER_RECOMMENDATION_POLICY"]
        ),
        "frontier_reference_mode": str(manifest["FRONTIER_REFERENCE_MODE"]),
        "frontier_lane_warm_start_mode": str(
            manifest["FRONTIER_LANE_WARM_START_MODE"]
        ),
    }

    reference_points_file = manifest["FRONTIER_REFERENCE_POINTS_FILE"]
    overrides["frontier_reference_points_file"] = (
        None if reference_points_file is None else str(reference_points_file)
    )
    epsilon_spec_file = manifest["FRONTIER_EPSILON_SPEC_FILE"]
    overrides["frontier_epsilon_spec_file"] = (
        None if epsilon_spec_file is None else str(epsilon_spec_file)
    )

    return argparse.Namespace(**{**vars(args), **overrides})


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
    early_stop_status: dict[str, object] | None = None,
) -> None:
    write_frontier_campaign_progress(
        path,
        FrontierCampaignProgress(
            schema_version=FRONTIER_CAMPAIGN_PROGRESS_SCHEMA_VERSION,
            campaign_id=campaign_id,
            frontier_version=frontier_version,
            frontier_engine=frontier_engine,
            target_payload=serialize_goal_mode_payload(target_payload),
            lane_records=lane_records,
            provisional_archive_members=provisional_archive_members,
            archive_members=archive_members,
            early_stop_status=early_stop_status,
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
        # Persist only the trimmed metric subset; the full results dict is
        # used solely to build the archive member upstream of this call.
        results=lane_payload.get("results_summary"),
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
    if not resume:
        return run_goal_mode_case_safe(
            args,
            goal_mode=goal_mode,
            stage2_bs_path=stage2_bs_path,
            output_root=output_root,
        )

    case_output_root = output_root / goal_mode
    if case_output_root.exists():
        final_results = _load_final_results(case_output_root)
        if final_results is not None:
            result_source, results_path, results = final_results
            return _resume_payload(
                args,
                goal_mode=goal_mode,
                stage2_bs_path=stage2_bs_path,
                case_output_root=case_output_root,
                result_source=result_source,
                results_path=results_path,
                results=results,
            )

    # Prefer continuing the optimizer from a solver checkpoint over treating a
    # partial salvage snapshot as final. A killed lane that wrote both must
    # resume the run from the checkpoint, not freeze on the partial best.
    resume_checkpoint = maybe_resume_solver_checkpoint_path(case_output_root)
    if resume_checkpoint is not None:
        args = argparse.Namespace(
            **{**vars(args), "resume_solver_checkpoint": str(resume_checkpoint)},
        )
        return run_goal_mode_case_safe(
            args,
            goal_mode=goal_mode,
            stage2_bs_path=stage2_bs_path,
            output_root=output_root,
        )

    if case_output_root.exists():
        salvage_results = _load_salvage_results(case_output_root)
        if salvage_results is not None:
            result_source, results_path, results = salvage_results
            return _resume_payload(
                args,
                goal_mode=goal_mode,
                stage2_bs_path=stage2_bs_path,
                case_output_root=case_output_root,
                result_source=result_source,
                results_path=results_path,
                results=results,
            )

    return run_goal_mode_case_safe(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        output_root=output_root,
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

    resume_manifest = None
    if args.resume:
        resume_manifest = load_resume_manifest(paths.manifest_path)
        if resume_manifest is not None:
            if args.allow_resume_arg_drift:
                _LOGGER.warning(
                    "--allow-resume-arg-drift set; CLI runtime calibration / "
                    "early-stop / RNG / normalization values may diverge from "
                    "the persisted manifest and the manifest will be rewritten."
                )
                args = argparse.Namespace(**vars(args))
            else:
                args = _apply_manifest_overrides_to_args(args, resume_manifest)
            seed_artifact_path = resume_manifest.get("SEED_ARTIFACT_PATH")
            if seed_artifact_path is not None:
                args.stage2_bs_path = str(resolved_path(str(seed_artifact_path)))
        else:
            args = argparse.Namespace(**vars(args))

    runtime_defaults = resolve_frontier_runtime_defaults_from_args(args)

    if args.dry_run:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_comparison.maybe_load_validated_stage2_seed_metadata(args)
        )
    else:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_comparison.load_validated_stage2_seed_metadata(args)
        )

    # Resolve normalization before replaying progress so duplicate detection
    # and distance metrics match the original campaign's invariants.
    initial_hypervolume_reference = resolve_hypervolume_reference(
        reference_spec=args.frontier_hypervolume_reference,
        seed_results=stage2_results,
    )
    # When resuming a manifest that locked in a normalization (especially the
    # ``ideal_nadir`` kind whose ``--frontier-normalization-spec-file`` is not
    # persisted), the manifest's PARETO_OBJECTIVE_NORMALIZATION payload is the
    # authoritative SSOT for the resolved normalization. Rehydrating from the
    # spec-file path would require the operator to re-pass the original CLI
    # arg on every resume; instead, rebuild from the persisted payload so the
    # manifest alone is sufficient to resume.
    if (
        resume_manifest is not None
        and not args.allow_resume_arg_drift
    ):
        pareto_objective_normalization = (
            build_pareto_objective_normalization_from_persisted_payload(
                resume_manifest["PARETO_OBJECTIVE_NORMALIZATION"],
            )
        )
    else:
        pareto_objective_normalization = build_pareto_objective_normalization(
            initial_hypervolume_reference,
            kind=args.frontier_normalization_kind,
            normalization_spec_path=args.frontier_normalization_spec_file,
        )

    resumed_progress = None
    if args.resume and paths.progress_path.exists():
        resumed_progress = load_frontier_campaign_progress(
            paths.progress_path,
            pareto_objective_normalization=pareto_objective_normalization,
        )
        args.frontier_version = resumed_progress.frontier_version
        args.frontier_engine = resumed_progress.frontier_engine
    resumed_lane_specs = resume_lane_specs_from_manifest(resume_manifest)
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
        # Resumed manifests are authoritative; their lane count overrides CLI intent.
        runtime_defaults = resolve_frontier_runtime_defaults_from_args(
            args,
            requested_num_lanes=len(lane_specs),
        )
    if resumed_progress is not None:
        campaign_id = resumed_progress.campaign_id
    else:
        campaign_id = uuid.uuid4().hex[:12]
    should_refresh_manifest = (
        not paths.manifest_path.exists()
        or (args.resume and args.allow_resume_arg_drift)
    )
    # Only (re)build the campaign manifest payload when it will actually be
    # written to disk. When resuming without drift, the on-disk manifest is
    # the authoritative SSOT and the in-memory rebuild would (a) be discarded
    # and (b) routes back through ``build_pareto_objective_normalization`` --
    # which fails for ``ideal_nadir`` campaigns because the manifest never
    # persists the original ``--frontier-normalization-spec-file`` path.
    if should_refresh_manifest:
        manifest = build_frontier_campaign_manifest(
            args,
            campaign_id=campaign_id,
            stage2_bs_path=stage2_bs_path,
            stage2_results_path=stage2_results_path,
            stage2_results=stage2_results,
            lane_specs=lane_specs,
            runtime_defaults=runtime_defaults,
        )
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
    # The campaign-wide hypervolume reference is IMMUTABLE: the value
    # resolved above (CLI spec + seed) is the SSOT used by early-stop,
    # history, the persisted archive, and the summary. No member-derived
    # fallback, no post-loop re-resolve, no reporting-time nadir
    # extension — if the certified archive regresses on any axis vs the
    # resolved reference, ``serialize_frontier_archive`` fails fast and
    # the operator must rerun with an explicit
    # ``--frontier-hypervolume-reference``.
    hypervolume_reference = initial_hypervolume_reference
    early_stop_status = (
        dict(resumed_progress.early_stop_status)
        if resumed_progress is not None
        and resumed_progress.early_stop_status is not None
        else build_initial_frontier_early_stop_status(
            runtime_defaults=runtime_defaults,
            archive_members=archive_members,
        )
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
            early_stop_status=early_stop_status,
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

    lane_execution_groups = (
        []
        if early_stop_status["triggered"]
        else build_frontier_lane_execution_groups(
            lane_specs,
            lane_records_by_id=lane_records_by_id,
            warm_start_mode=args.frontier_lane_warm_start_mode,
            early_stop_patience_lanes=runtime_defaults.early_stop_patience_lanes,
            lane_workers=args.frontier_lane_workers,
        )
    )
    _log_lane_workers_overridden_to_sequential(
        args=args,
        lane_execution_groups=lane_execution_groups,
        runtime_defaults=runtime_defaults,
    )
    for lane_execution_group in lane_execution_groups:
        executions = [
            build_frontier_lane_execution(
                args,
                lane_spec,
                campaign_id=campaign_id,
                lane_index=lane_index,
                lane_records_by_id=lane_records_by_id,
                lane_specs=lane_specs,
                runtime_defaults=runtime_defaults,
                stage2_bs_path=stage2_bs_path,
                output_root=output_root,
            )
            for lane_index, lane_spec in lane_execution_group
        ]
        lane_results = run_frontier_lane_execution_group(
            executions,
            resume=args.resume,
            lane_workers=args.frontier_lane_workers,
        )
        for lane_result in lane_results:
            execution = lane_result.execution
            lane_spec = execution.lane_spec
            lane_payload = lane_result.lane_payload
            provisional_archive_member = None
            archive_member = None
            archive_update = None
            if lane_payload["status"] == "completed":
                provisional_archive_member = build_archive_member_from_results(
                    campaign_id=campaign_id,
                    lane_id=lane_spec.lane_id,
                    payload=lane_payload,
                    rerun_contract=execution.lane_contract.rerun_contract,
                    archive_state=FRONTIER_ARCHIVE_STATE_PROVISIONAL,
                )
                provisional_archive_members.append(provisional_archive_member)
                archive_member = finalize_archive_member(provisional_archive_member)
                archive_members, archive_update = update_frontier_archive(
                    archive_members,
                    archive_member,
                    pareto_objective_normalization=pareto_objective_normalization,
                )
            lane_records_by_id[lane_spec.lane_id] = build_lane_record_from_payload(
                execution.lane_contract,
                lane_spec,
                execution.lane_budget,
                lane_payload,
                provisional_archive_member=provisional_archive_member,
                archive_member=archive_member,
                archive_update=archive_update,
            )
            early_stop_status = update_frontier_early_stop_status(
                status=early_stop_status,
                certified_archive_members_list=archive_members,
                hypervolume_reference=hypervolume_reference,
                runtime_defaults=runtime_defaults,
            )
            if early_stop_status["triggered"]:
                early_stop_status["stopped_after_lane_id"] = lane_spec.lane_id
                break
        persist_progress()
        if early_stop_status["triggered"]:
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

    # Final reporting uses the SAME immutable ``hypervolume_reference``
    # resolved before the lane loop. ``serialize_frontier_archive`` asserts
    # the reference is a per-axis nadir for the certified archive; if a lane
    # regressed vs the resolved reference it raises here so the user can
    # rerun with an explicit ``--frontier-hypervolume-reference`` instead of
    # silently extending the reference at reporting time.
    write_json(
        paths.archive_path,
        serialize_frontier_archive(
            certified_members,
            campaign_id=campaign_id,
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
        output_root=output_root,
        paths=paths,
        lane_specs=lane_specs,
        target_payload=target_payload,
        lane_records=lane_record_payloads,
        archive_members=certified_members,
        recommendation_payload=recommendation_payload,
        delta_fn=goal_mode_comparison.delta,
        runtime_defaults=runtime_defaults,
        early_stop_status=early_stop_status,
        hypervolume_reference=hypervolume_reference,
        pareto_objective_normalization=pareto_objective_normalization,
    )
    write_json(paths.summary_path, summary)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0
    target_failed = (
        not args.skip_target
        and target_payload is not None
        and target_payload.get("status") == "failed"
    )
    if target_failed:
        return 1
    return 0 if certified_members else 1


if __name__ == "__main__":
    raise SystemExit(main())
