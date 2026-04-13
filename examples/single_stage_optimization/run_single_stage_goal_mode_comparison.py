from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_runner_common import (  # noqa: E402
    SINGLE_STAGE_SCRIPT_PATH,
    discover_single_results_path,
    load_json,
    load_stage2_artifact_results,
    resolved_optional_path,
    resolved_path,
    run_command,
    snapshot_single_results_paths,
    timeout_or_none,
    validate_stage2_seed_not_init_only,
)
from banana_opt.artifact_contracts import upgrade_legacy_stage2_artifact_results  # noqa: E402

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_goal_mode_comparison"
DEFAULT_SUMMARY_JSON = "single_stage_goal_mode_comparison_summary.json"
GOAL_MODES = ("target", "frontier")


def _append_optional_flag(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_bool_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched single-stage target-vs-frontier comparisons from one explicit "
            "Stage 2 seed artifact."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plasma-surf-filename",
        required=True,
        help="VMEC wout filename used as the single-stage target surface.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        required=True,
        help="Path to the Stage 2 biot_savart_opt.json seed artifact shared by both goal modes.",
    )
    parser.add_argument(
        "--allow-init-only-stage2-seed",
        action="store_true",
        help=(
            "Allow reusing a Stage 2 artifact whose sibling results.json reports "
            "init_only=true. Disabled by default because init-only smoke seeds can "
            "land single-stage in the wrong transform basin."
        ),
    )
    parser.add_argument("--equilibria-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--summary-json",
        default=None,
        help=f"Optional summary path. Defaults to <output-root>/{DEFAULT_SUMMARY_JSON}.",
    )
    parser.add_argument("--single-stage-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--nphi", type=int, default=91)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--maxcor", type=int, default=300)
    parser.add_argument("--ftol", type=float, default=1e-15)
    parser.add_argument("--gtol", type=float, default=1e-15)
    parser.add_argument("--constraint-method", choices=["penalty", "alm"], default="penalty")
    parser.add_argument("--alm-max-outer-iters", type=int, default=10)
    parser.add_argument("--alm-penalty-init", type=float, default=1.0)
    parser.add_argument("--alm-penalty-scale", type=float, default=10.0)
    parser.add_argument("--alm-penalty-max", type=float, default=1.0e8)
    parser.add_argument("--alm-feas-tol", type=float, default=1e-6)
    parser.add_argument("--alm-stationarity-tol", type=float, default=1e-6)
    parser.add_argument("--alm-trust-radius-init", type=float, default=0.05)
    parser.add_argument("--alm-trust-radius-min", type=float, default=1e-4)
    parser.add_argument("--alm-trust-radius-shrink", type=float, default=0.5)
    parser.add_argument("--alm-trust-radius-grow", type=float, default=1.5)
    parser.add_argument("--alm-max-inner-attempts", type=int, default=4)
    parser.add_argument("--alm-max-subproblem-continuations", type=int, default=20)
    parser.add_argument("--alm-distance-smoothing", type=float, default=0.005)
    parser.add_argument("--alm-curvature-smoothing", type=float, default=0.05)
    parser.add_argument("--alm-formulation", choices=["weighted_sum", "thresholded_physics"], default=os.environ.get("ALM_FORMULATION", "weighted_sum"))
    parser.add_argument("--alm-qs-threshold", type=float, default=float(os.environ["ALM_QS_THRESHOLD"]) if "ALM_QS_THRESHOLD" in os.environ else None)
    parser.add_argument("--alm-boozer-threshold", type=float, default=float(os.environ["ALM_BOOZER_THRESHOLD"]) if "ALM_BOOZER_THRESHOLD" in os.environ else None)
    parser.add_argument(
        "--alm-iota-penalty-threshold",
        type=float,
        default=float(os.environ["ALM_IOTA_PENALTY_THRESHOLD"]) if "ALM_IOTA_PENALTY_THRESHOLD" in os.environ else None,
    )
    parser.add_argument(
        "--alm-length-penalty-threshold",
        type=float,
        default=float(os.environ["ALM_LENGTH_PENALTY_THRESHOLD"]) if "ALM_LENGTH_PENALTY_THRESHOLD" in os.environ else None,
    )
    parser.add_argument("--iota-target", type=float, default=0.15)
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument("--boozer-I", type=float, default=float(os.environ["BOOZER_I"]) if "BOOZER_I" in os.environ else None)
    parser.add_argument(
        "--plasma-current-A",
        type=float,
        default=float(os.environ["PLASMA_CURRENT_A"]) if "PLASMA_CURRENT_A" in os.environ else None,
    )
    parser.add_argument("--banana-surf-radius", type=float, default=float(os.environ["BANANA_SURF_RADIUS"]) if "BANANA_SURF_RADIUS" in os.environ else None)
    parser.add_argument("--num-surfaces", type=int, choices=[1, 2], default=int(os.environ.get("NUM_SURFACES", "1")))
    parser.add_argument("--inner-surface-ratio", type=float, default=float(os.environ.get("INNER_SURFACE_RATIO", "0.8")))
    parser.add_argument("--surface-gap-threshold", type=float, default=float(os.environ.get("SURFACE_GAP_THRESHOLD", "0.0")))
    parser.add_argument("--multisurface-ramp-iterations", type=int, default=int(os.environ.get("MULTISURFACE_RAMP_ITERATIONS", "5")))
    parser.add_argument("--inner-surface-initial-weight", type=float, default=float(os.environ.get("INNER_SURFACE_INITIAL_WEIGHT", "0.0")))
    parser.add_argument("--multisurface-initial-step-scale", type=float, default=float(os.environ.get("MULTISURFACE_INITIAL_STEP_SCALE", "1.0")))
    parser.add_argument("--multisurface-initial-step-maxiter", type=int, default=int(os.environ.get("MULTISURFACE_INITIAL_STEP_MAXITER", "0")))
    parser.add_argument("--boozer-stage", choices=["initial", "final"], default=os.environ.get("BOOZER_STAGE", "initial"))
    parser.add_argument("--boozer-stage-refinement", action="store_true")
    parser.add_argument("--refinement-boozer-stage", choices=["initial", "final"], default=os.environ.get("REFINEMENT_BOOZER_STAGE", "final"))
    parser.add_argument("--refinement-maxiter", type=int, default=int(os.environ.get("REFINEMENT_MAXITER", "100")))
    parser.add_argument("--refinement-chunk-maxiter", type=int, default=int(os.environ.get("REFINEMENT_CHUNK_MAXITER", "20")))
    parser.add_argument("--refinement-max-stalled-chunks", type=int, default=int(os.environ.get("REFINEMENT_MAX_STALLED_CHUNKS", "2")))
    parser.add_argument("--res-weight", type=float, default=1000.0)
    parser.add_argument("--iotas-weight", type=float, default=100.0)
    parser.add_argument(
        "--frontier-volume-weight",
        type=float,
        default=float(os.environ["FRONTIER_VOLUME_WEIGHT"]) if "FRONTIER_VOLUME_WEIGHT" in os.environ else None,
        help=(
            "Independent volume-reward weight for frontier mode. When omitted, forwarded "
            "as-is (None) so the single-stage script falls back to --iotas-weight."
        ),
    )
    parser.add_argument("--cc-weight", type=float, default=100.0)
    parser.add_argument("--curvature-weight", type=float, default=0.1)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--length-target", type=float, default=float(os.environ["SS_LENGTH_TARGET"]) if "SS_LENGTH_TARGET" in os.environ else None)
    parser.add_argument("--cs-weight", type=float, default=1.0)
    parser.add_argument("--surf-dist-weight", type=float, default=1000.0)
    parser.add_argument("--cc-dist", type=float, default=0.05)
    parser.add_argument("--cs-dist", type=float, default=0.02)
    parser.add_argument("--ss-dist", type=float, default=0.04)
    parser.add_argument("--curvature-threshold", type=float, default=40.0)
    parser.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("CHECKPOINT_EVERY", "0")))
    parser.add_argument("--topology-gate-fieldlines", type=int, default=int(os.environ.get("TOPOLOGY_GATE_FIELDLINES", "4")))
    parser.add_argument("--topology-gate-tmax", type=float, default=float(os.environ.get("TOPOLOGY_GATE_TMAX", "2.0")))
    parser.add_argument("--topology-gate-tol", type=float, default=float(os.environ.get("TOPOLOGY_GATE_TOL", "1e-7")))
    parser.add_argument("--topology-gate-survival-threshold", type=float, default=float(os.environ.get("TOPOLOGY_GATE_SURVIVAL_THRESHOLD", "0.25")))
    parser.add_argument("--topology-gate-penalty-scale", type=float, default=float(os.environ.get("TOPOLOGY_GATE_PENALTY_SCALE", "4.0")))
    parser.add_argument("--topology-scorer-every", type=int, default=int(os.environ.get("TOPOLOGY_SCORER_EVERY", "0")))
    parser.add_argument("--topology-scorer-nfieldlines", type=int, default=int(os.environ.get("TOPOLOGY_SCORER_NFIELDLINES", "12")))
    parser.add_argument("--topology-scorer-tmax", type=float, default=float(os.environ.get("TOPOLOGY_SCORER_TMAX", "50.0")))
    parser.add_argument("--confinement-objective-weight", type=float, default=float(os.environ.get("CONFINEMENT_OBJECTIVE_WEIGHT", "0.0")))
    parser.add_argument("--confinement-surrogate-worst-k", type=int, default=int(os.environ.get("CONFINEMENT_SURROGATE_WORST_K", "3")))
    parser.add_argument("--confinement-surrogate-early-threshold", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_THRESHOLD", "0.2")))
    parser.add_argument("--confinement-surrogate-mean-weight", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_MEAN_WEIGHT", "0.2")))
    parser.add_argument("--confinement-surrogate-worst-weight", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_WORST_WEIGHT", "0.6")))
    parser.add_argument("--confinement-surrogate-early-weight", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_WEIGHT", "0.2")))
    parser.add_argument(
        "--hardware-search-mode",
        choices=["hard", "warn", "adaptive"],
        default="hard",
    )
    parser.add_argument("--hardware-search-soft-iterations", type=int, default=0)
    parser.add_argument("--basin-hops", type=int, default=0)
    parser.add_argument("--basin-stepsize", type=float, default=0.01)
    parser.add_argument("--basin-temperature", type=float, default=1.0)
    parser.add_argument("--basin-niter-success", type=int, default=0)
    parser.add_argument("--basin-seed", type=int, default=-1)
    parser.add_argument("--init-only", action="store_true")
    return parser.parse_args()


def load_validated_stage2_seed_metadata(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path | None = None,
) -> tuple[Path, Path, dict]:
    if stage2_bs_path is None:
        stage2_bs_path = resolved_path(args.stage2_bs_path)
    stage2_results_path, stage2_results = load_stage2_artifact_results(stage2_bs_path)
    stage2_results = upgrade_legacy_stage2_artifact_results(stage2_results)
    actual_surface = stage2_results.get("PLASMA_SURF_FILENAME")
    expected_surface = Path(args.plasma_surf_filename).name
    if actual_surface is None:
        raise ValueError(
            f"Stage 2 artifact results.json is missing PLASMA_SURF_FILENAME: {stage2_results_path}"
        )
    if Path(str(actual_surface)).name != expected_surface:
        raise ValueError(
            "Stage 2 artifact surface mismatch: "
            f"--plasma-surf-filename requests {expected_surface!r}, but "
            f"{stage2_results_path} reports {actual_surface!r}."
        )
    validate_stage2_seed_not_init_only(
        stage2_results_path,
        stage2_results,
        owner_label="run_single_stage_goal_mode_comparison.py",
        allow_init_only=getattr(args, "allow_init_only_stage2_seed", False),
    )
    return stage2_bs_path, stage2_results_path, stage2_results


def maybe_load_validated_stage2_seed_metadata(
    args: argparse.Namespace,
) -> tuple[Path, Path | None, dict | None]:
    stage2_bs_path = resolved_path(args.stage2_bs_path)
    stage2_results_path = stage2_bs_path.with_name("results.json")
    if not stage2_bs_path.exists() or not stage2_results_path.exists():
        return stage2_bs_path, None, None
    _, loaded_results_path, stage2_results = load_validated_stage2_seed_metadata(
        args,
        stage2_bs_path=stage2_bs_path,
    )
    return stage2_bs_path, loaded_results_path, stage2_results


def build_single_stage_goal_mode_command(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    case_output_root: Path,
) -> list[str]:
    equilibria_dir = resolved_optional_path(args.equilibria_dir)
    command = [
        args.python_executable,
        str(SINGLE_STAGE_SCRIPT_PATH),
        "--plasma-surf-filename",
        Path(args.plasma_surf_filename).name,
        "--stage2-bs-path",
        str(stage2_bs_path),
        "--output-root",
        str(case_output_root),
        "--single-stage-goal-mode",
        goal_mode,
        "--nphi",
        str(args.nphi),
        "--ntheta",
        str(args.ntheta),
        "--mpol",
        str(args.mpol),
        "--ntor",
        str(args.ntor),
        "--maxiter",
        str(args.maxiter),
        "--maxcor",
        str(args.maxcor),
        "--ftol",
        str(args.ftol),
        "--gtol",
        str(args.gtol),
        "--constraint-method",
        args.constraint_method,
        "--alm-formulation",
        args.alm_formulation,
        "--iota-target",
        str(args.iota_target),
        "--vol-target",
        str(args.vol_target),
        "--res-weight",
        str(args.res_weight),
        "--iotas-weight",
        str(args.iotas_weight),
        "--cc-weight",
        str(args.cc_weight),
        "--curvature-weight",
        str(args.curvature_weight),
        "--length-weight",
        str(args.length_weight),
        "--cs-weight",
        str(args.cs_weight),
        "--surf-dist-weight",
        str(args.surf_dist_weight),
        "--cc-dist",
        str(args.cc_dist),
        "--cs-dist",
        str(args.cs_dist),
        "--ss-dist",
        str(args.ss_dist),
        "--curvature-threshold",
        str(args.curvature_threshold),
        "--num-surfaces",
        str(args.num_surfaces),
        "--inner-surface-ratio",
        str(args.inner_surface_ratio),
        "--surface-gap-threshold",
        str(args.surface_gap_threshold),
        "--multisurface-ramp-iterations",
        str(args.multisurface_ramp_iterations),
        "--inner-surface-initial-weight",
        str(args.inner_surface_initial_weight),
        "--multisurface-initial-step-scale",
        str(args.multisurface_initial_step_scale),
        "--multisurface-initial-step-maxiter",
        str(args.multisurface_initial_step_maxiter),
        "--boozer-stage",
        args.boozer_stage,
        "--refinement-boozer-stage",
        args.refinement_boozer_stage,
        "--refinement-maxiter",
        str(args.refinement_maxiter),
        "--refinement-chunk-maxiter",
        str(args.refinement_chunk_maxiter),
        "--refinement-max-stalled-chunks",
        str(args.refinement_max_stalled_chunks),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--topology-gate-fieldlines",
        str(args.topology_gate_fieldlines),
        "--topology-gate-tmax",
        str(args.topology_gate_tmax),
        "--topology-gate-tol",
        str(args.topology_gate_tol),
        "--topology-gate-survival-threshold",
        str(args.topology_gate_survival_threshold),
        "--topology-gate-penalty-scale",
        str(args.topology_gate_penalty_scale),
        "--topology-scorer-every",
        str(args.topology_scorer_every),
        "--topology-scorer-nfieldlines",
        str(args.topology_scorer_nfieldlines),
        "--topology-scorer-tmax",
        str(args.topology_scorer_tmax),
        "--confinement-objective-weight",
        str(args.confinement_objective_weight),
        "--confinement-surrogate-worst-k",
        str(args.confinement_surrogate_worst_k),
        "--confinement-surrogate-early-threshold",
        str(args.confinement_surrogate_early_threshold),
        "--confinement-surrogate-mean-weight",
        str(args.confinement_surrogate_mean_weight),
        "--confinement-surrogate-worst-weight",
        str(args.confinement_surrogate_worst_weight),
        "--confinement-surrogate-early-weight",
        str(args.confinement_surrogate_early_weight),
        "--hardware-search-mode",
        args.hardware_search_mode,
        "--hardware-search-soft-iterations",
        str(args.hardware_search_soft_iterations),
    ]
    if equilibria_dir is not None:
        command.extend(["--equilibria-dir", str(equilibria_dir)])
    _append_optional_flag(command, "--boozer-I", args.boozer_I)
    _append_optional_flag(command, "--plasma-current-A", args.plasma_current_A)
    _append_optional_flag(command, "--banana-surf-radius", args.banana_surf_radius)
    _append_optional_flag(command, "--length-target", args.length_target)
    _append_optional_flag(command, "--frontier-volume-weight", args.frontier_volume_weight)
    _append_optional_flag(command, "--alm-qs-threshold", args.alm_qs_threshold)
    _append_optional_flag(command, "--alm-boozer-threshold", args.alm_boozer_threshold)
    _append_optional_flag(command, "--alm-iota-penalty-threshold", args.alm_iota_penalty_threshold)
    _append_optional_flag(command, "--alm-length-penalty-threshold", args.alm_length_penalty_threshold)
    _append_bool_flag(command, "--boozer-stage-refinement", args.boozer_stage_refinement)
    if args.constraint_method == "alm":
        command.extend(
            [
                "--alm-max-outer-iters",
                str(args.alm_max_outer_iters),
                "--alm-penalty-init",
                str(args.alm_penalty_init),
                "--alm-penalty-scale",
                str(args.alm_penalty_scale),
                "--alm-penalty-max",
                str(args.alm_penalty_max),
                "--alm-feas-tol",
                str(args.alm_feas_tol),
                "--alm-stationarity-tol",
                str(args.alm_stationarity_tol),
                "--alm-trust-radius-init",
                str(args.alm_trust_radius_init),
                "--alm-trust-radius-min",
                str(args.alm_trust_radius_min),
                "--alm-trust-radius-shrink",
                str(args.alm_trust_radius_shrink),
                "--alm-trust-radius-grow",
                str(args.alm_trust_radius_grow),
                "--alm-max-inner-attempts",
                str(args.alm_max_inner_attempts),
                "--alm-max-subproblem-continuations",
                str(args.alm_max_subproblem_continuations),
                "--alm-distance-smoothing",
                str(args.alm_distance_smoothing),
                "--alm-curvature-smoothing",
                str(args.alm_curvature_smoothing),
            ]
        )
    if args.basin_hops > 0:
        command.extend(
            [
                "--basin-hops",
                str(args.basin_hops),
                "--basin-stepsize",
                str(args.basin_stepsize),
                "--basin-temperature",
                str(args.basin_temperature),
            ]
        )
        if args.basin_niter_success > 0:
            command.extend(["--basin-niter-success", str(args.basin_niter_success)])
        if args.basin_seed >= 0:
            command.extend(["--basin-seed", str(args.basin_seed)])
    if args.init_only:
        command.append("--init-only")
    return command


def _result_metric_subset(results: dict) -> dict:
    return {
        "goal_mode": results.get("SINGLE_STAGE_GOAL_MODE"),
        "goal_mode_impl": results.get("SINGLE_STAGE_GOAL_MODE_IMPL"),
        "target_iota": results.get("TARGET_IOTA"),
        "target_volume": results.get("TARGET_VOLUME"),
        "boozer_surface_target_volumes": results.get("BOOZER_SURFACE_TARGET_VOLUMES"),
        "termination_message": results.get("TERMINATION_MESSAGE"),
        "optimizer_success": results.get("OPTIMIZER_SUCCESS"),
        "final_feasibility_ok": results.get("FINAL_FEASIBILITY_OK"),
        "hardware_constraints_ok": results.get("HARDWARE_CONSTRAINTS_OK"),
        "final_topology_gate_success": results.get("FINAL_TOPOLOGY_GATE_SUCCESS"),
        "final_iota": results.get("FINAL_IOTA"),
        "final_volume": results.get("FINAL_VOLUME"),
        "nonqs_ratio": results.get("NONQS_RATIO"),
        "boozer_residual": results.get("BOOZER_RESIDUAL"),
        "coil_length": results.get("COIL_LENGTH"),
        "max_curvature": results.get("MAX_CURVATURE"),
        "curve_curve_min_dist": results.get("CURVE_CURVE_MIN_DIST"),
        "curve_surface_min_dist": results.get("CURVE_SURFACE_MIN_DIST"),
        "surface_vessel_min_dist": results.get("SURFACE_VESSEL_MIN_DIST"),
        "invalid_state_rejects_total": results.get("INVALID_STATE_REJECTS_TOTAL"),
        "topology_gate_rejects": results.get("TOPOLOGY_GATE_REJECTS"),
        "hardware_rejects": results.get("HARDWARE_REJECTS"),
        "surface_solve_rejects": results.get("SURFACE_SOLVE_REJECTS"),
        "best_feasible_available": results.get("BEST_FEASIBLE_AVAILABLE"),
        "best_feasible_stage": results.get("BEST_FEASIBLE_STAGE"),
        "best_feasible_frontier_rank_objective_j": results.get("BEST_FEASIBLE_FRONTIER_RANK_OBJECTIVE_J"),
        "best_feasible_frontier_trust_ok": results.get("BEST_FEASIBLE_FRONTIER_TRUST_OK"),
        "best_feasible_final_iota": results.get("BEST_FEASIBLE_FINAL_IOTA"),
        "best_feasible_final_volume": results.get("BEST_FEASIBLE_FINAL_VOLUME"),
        "best_feasible_qa_objective": results.get("BEST_FEASIBLE_QA_OBJECTIVE"),
        "best_feasible_boozer_objective": results.get("BEST_FEASIBLE_BOOZER_OBJECTIVE"),
        "best_feasible_search_objective_j": results.get("BEST_FEASIBLE_SEARCH_OBJECTIVE_J"),
        "best_feasible_base_objective_j": results.get("BEST_FEASIBLE_BASE_OBJECTIVE_J"),
        "best_feasible_curve_curve_min_dist": results.get("BEST_FEASIBLE_CURVE_CURVE_MIN_DIST"),
        "best_feasible_curve_surface_min_dist": results.get("BEST_FEASIBLE_CURVE_SURFACE_MIN_DIST"),
        "best_feasible_surface_vessel_min_dist": results.get("BEST_FEASIBLE_SURFACE_VESSEL_MIN_DIST"),
        "best_feasible_max_curvature": results.get("BEST_FEASIBLE_MAX_CURVATURE"),
        "best_feasible_hardware_constraints_ok": results.get("BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK"),
        "best_feasible_final_topology_gate_success": results.get("BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_SUCCESS"),
        "search_objective_j": results.get("SEARCH_OBJECTIVE_J"),
        "objective_j": results.get("OBJECTIVE_J"),
        "base_objective_j": results.get("BASE_OBJECTIVE_J"),
        "frontier_rank_objective_j": results.get("FRONTIER_RANK_OBJECTIVE_J"),
        "frontier_trust_ok": results.get("FRONTIER_TRUST_OK"),
        "frontier_boozer_trust_threshold": results.get("FRONTIER_BOOZER_TRUST_THRESHOLD"),
        "frontier_boozer_trust_excess": results.get("FRONTIER_BOOZER_TRUST_EXCESS"),
        "frontier_trust_rejects": results.get("FRONTIER_TRUST_REJECTS"),
        "frontier_reference_iota": results.get("FRONTIER_REFERENCE_IOTA"),
        "frontier_reference_volume": results.get("FRONTIER_REFERENCE_VOLUME"),
        "frontier_reference_qa": results.get("FRONTIER_REFERENCE_QA"),
        "frontier_reference_boozer": results.get("FRONTIER_REFERENCE_BOOZER"),
        "frontier_effective_iota_weight": results.get("FRONTIER_EFFECTIVE_IOTA_WEIGHT"),
        "frontier_effective_volume_weight": results.get("FRONTIER_EFFECTIVE_VOLUME_WEIGHT"),
        "frontier_effective_boozer_weight": results.get("FRONTIER_EFFECTIVE_BOOZER_WEIGHT"),
        "frontier_volume_objective": results.get("FRONTIER_VOLUME_OBJECTIVE"),
    }


def _delta(frontier_value, target_value):
    if frontier_value is None or target_value is None:
        return None
    return float(frontier_value) - float(target_value)


def build_summary(
    args: argparse.Namespace,
    commands_by_mode: dict[str, list[str]],
    *,
    stage2_bs_path: Path,
    stage2_results_path: Path | None = None,
    stage2_results: dict | None = None,
    mode_payloads: dict[str, dict] | None = None,
) -> dict:
    output_root = resolved_path(args.output_root)
    summary = {
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "stage2_bs_path": str(stage2_bs_path),
        "output_root": str(output_root),
        "goal_modes": list(GOAL_MODES),
        "dry_run": bool(args.dry_run),
        "search_objective_values_comparable": False,
        "mode_runs": {
            goal_mode: {
                "output_root": str(output_root / goal_mode),
                "command": commands_by_mode[goal_mode],
            }
            for goal_mode in GOAL_MODES
        },
    }
    if stage2_results_path is not None:
        summary["stage2_results_path"] = str(stage2_results_path)
    if stage2_results is not None:
        summary["stage2_artifact_plasma_surf_filename"] = stage2_results.get(
            "PLASMA_SURF_FILENAME"
        )
        summary["stage2_artifact_init_only"] = stage2_results.get("init_only")
        summary["stage2_banana_current_a"] = stage2_results.get("BANANA_CURRENT_A")
        summary["stage2_banana_current_max_a"] = stage2_results.get("BANANA_CURRENT_MAX_A")
    if mode_payloads is None:
        return summary

    for goal_mode, payload in mode_payloads.items():
        mode_entry = summary["mode_runs"][goal_mode]
        mode_entry["results_path"] = str(payload["results_path"])
        mode_entry["results"] = _result_metric_subset(payload["results"])

    target_results = mode_payloads["target"]["results"]
    frontier_results = mode_payloads["frontier"]["results"]
    summary["comparison"] = {
        "frontier_minus_target_final_iota": _delta(
            frontier_results.get("FINAL_IOTA"),
            target_results.get("FINAL_IOTA"),
        ),
        "frontier_minus_target_final_volume": _delta(
            frontier_results.get("FINAL_VOLUME"),
            target_results.get("FINAL_VOLUME"),
        ),
        "frontier_minus_target_nonqs_ratio": _delta(
            frontier_results.get("NONQS_RATIO"),
            target_results.get("NONQS_RATIO"),
        ),
        "frontier_minus_target_boozer_residual": _delta(
            frontier_results.get("BOOZER_RESIDUAL"),
            target_results.get("BOOZER_RESIDUAL"),
        ),
        "both_final_feasibility_ok": bool(
            target_results.get("FINAL_FEASIBILITY_OK")
            and frontier_results.get("FINAL_FEASIBILITY_OK")
        ),
        "both_hardware_feasible": bool(
            target_results.get("HARDWARE_CONSTRAINTS_OK")
            and frontier_results.get("HARDWARE_CONSTRAINTS_OK")
        ),
        "both_optimizer_success": bool(
            target_results.get("OPTIMIZER_SUCCESS")
            and frontier_results.get("OPTIMIZER_SUCCESS")
        ),
    }
    return summary


def run_goal_mode_case(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    output_root: Path,
) -> dict:
    case_output_root = output_root / goal_mode
    case_output_root.mkdir(parents=True, exist_ok=True)
    previous_snapshot = snapshot_single_results_paths(case_output_root)
    command = build_single_stage_goal_mode_command(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        case_output_root=case_output_root,
    )
    if args.dry_run:
        return {"command": command}
    run_command(
        command,
        timeout_seconds=timeout_or_none(args.single_stage_timeout_seconds),
    )
    results_path = discover_single_results_path(
        case_output_root,
        previous_snapshot=previous_snapshot,
    )
    return {
        "command": command,
        "results_path": results_path,
        "results": load_json(results_path),
    }


def main() -> int:
    args = parse_args()
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stage2_bs_path, stage2_results_path, stage2_results = (
            maybe_load_validated_stage2_seed_metadata(args)
        )
    else:
        stage2_bs_path, stage2_results_path, stage2_results = (
            load_validated_stage2_seed_metadata(args)
        )

    mode_runs = {
        goal_mode: run_goal_mode_case(
            args,
            goal_mode=goal_mode,
            stage2_bs_path=stage2_bs_path,
            output_root=output_root,
        )
        for goal_mode in GOAL_MODES
    }
    commands_by_mode = {
        goal_mode: mode_runs[goal_mode]["command"] for goal_mode in GOAL_MODES
    }
    summary = build_summary(
        args,
        commands_by_mode,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        mode_payloads=(
            None
            if args.dry_run
            else {
                goal_mode: {
                    "results_path": mode_runs[goal_mode]["results_path"],
                    "results": mode_runs[goal_mode]["results"],
                }
                for goal_mode in GOAL_MODES
            }
        ),
    )

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
