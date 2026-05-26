from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_single_stage_goal_mode_comparison as goal_mode_runner  # noqa: E402
import run_single_stage_thresholded_physics_alm as recovery_runner  # noqa: E402
import run_stage2_alm as stage2_alm_runner  # noqa: E402
from banana_opt.current_contracts import (  # noqa: E402
    resolve_plasma_current_settings_for_num_surfaces,
)
from banana_opt.hardware_constraint_schema import (  # noqa: E402
    build_bootability_recovery_payload_fields,
)
from banana_opt.stage2_single_stage_handoff import (  # noqa: E402
    BOOTABILITY_STAGE_PROBE,
    BOOTABILITY_STAGE_RECOVERY,
    bootability_passes,
    build_stage2_seed_production_handoff_fields,
    probe_stage2_seed_bootability,
    resolve_stage2_finite_current_mode,
    validate_stage2_seed_bootability_contract,
)
from workflow_runner_common import (  # noqa: E402
    Stage2ArtifactConfig,
    build_stage2_command,
    ensure_stage2_artifact_result,
    load_json,
    load_validated_stage2_seed_results,
    resolve_stage2_artifact_path,
    resolved_optional_path,
    resolved_handoff_equilibrium_path,
    resolved_path,
    resolve_single_stage_iota_target_arg,
    run_command,
    timeout_or_none,
    write_json,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_stage2_to_single_stage"
DEFAULT_SUMMARY_JSON = "stage2_to_single_stage_summary.json"
DATABASE_EQUILIBRIA_DIR = SCRIPT_DIR.parents[1] / "DATABASE" / "EQUILIBRIA"
RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM = "pre_boozer_stage2_alm"
RECOVERY_STAGE_THRESHOLDED_PHYSICS_ALM = "thresholded_physics_alm"
RECOVERY_STAGES = (
    RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM,
    RECOVERY_STAGE_THRESHOLDED_PHYSICS_ALM,
)
LANE_STAGE2_HARDWARE = "stage2_hardware"
LANE_PRE_BOOZER_REPAIR = "pre_boozer_repair"
LANE_BOOZER_ACTIVATION = "boozer_activation"
LANE_SINGLE_STAGE_TARGET = "single_stage_target"
LANE_PROMOTION = "promotion"
LANE_SEQUENCE = (
    LANE_STAGE2_HARDWARE,
    LANE_PRE_BOOZER_REPAIR,
    LANE_BOOZER_ACTIVATION,
    LANE_SINGLE_STAGE_TARGET,
    LANE_PROMOTION,
)
BLOCKING_REASON_PRE_BOOZER_REPAIR_REQUIRED = "pre_boozer_repair_required"
SEED_SOURCE_DIRECT_STAGE2_DONOR = "direct_stage2_donor"
SEED_SOURCE_RECOVERED_STAGE2_DONOR = "recovered_stage2_donor"


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified Stage 2 -> bootability probe -> recovery -> single-stage "
            "workflow with fail-closed production handoff checks."
        ),
        parents=[goal_mode_runner.build_parser(add_help=False)],
        add_help=add_help,
        conflict_handler="resolve",
    )
    parser.set_defaults(output_root=str(DEFAULT_OUTPUT_ROOT), summary_json=None)
    parser.add_argument(
        "--stage2-bs-path",
        default=None,
        help=(
            "Existing Stage 2 biot_savart_opt.json donor. When omitted, the wrapper "
            "generates one via the Stage 2 ALM helper."
        ),
    )
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Optional explicit equilibrium path forwarded into the bootability probe.",
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=int(os.environ.get("NUM_TF_COILS", "20")),
        help="Expected number of TF coils in the loaded donor artifact.",
    )
    parser.add_argument(
        "--goal-mode",
        choices=goal_mode_runner.GOAL_MODES,
        default="target",
        help="Goal mode for the final single-stage run.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--probe-only",
        dest="probe_only",
        action="store_true",
        help="Run the Stage 2.5 bootability probe and stop.",
    )
    mode_group.add_argument(
        "--recovery-only",
        "--repair-only",
        action="store_true",
        dest="recovery_only",
        help="Run probe plus the bounded recovery stage and stop before the full single-stage run.",
    )
    parser.add_argument(
        "--bootability-iota-tolerance",
        type=float,
        default=5.0e-3,
        help="Absolute |iota_solved - iota_target| tolerance recorded as donor iota telemetry.",
    )
    parser.add_argument(
        "--stage2-profile",
        choices=sorted(stage2_alm_runner.DEFAULT_STAGE2_PROFILES),
        default=None,
        help="Built-in Stage 2 ALM profile used when --stage2-bs-path is omitted.",
    )
    parser.add_argument(
        "--stage2-spec-json",
        default=None,
        help="Explicit Stage 2 ALM spec JSON used when --stage2-bs-path is omitted.",
    )
    parser.add_argument(
        "--stage2-output-root",
        default=None,
        help="Optional output root for generated Stage 2 artifacts. Defaults to <output-root>/stage2.",
    )
    parser.add_argument(
        "--stage2-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional timeout for generated Stage 2 runs.",
    )
    parser.add_argument(
        "--stage2-cc-threshold",
        type=float,
        default=None,
        help="Optional Stage 2 generation override for the coil-coil spacing threshold.",
    )
    parser.add_argument(
        "--stage2-curvature-threshold",
        type=float,
        default=None,
        help="Optional Stage 2 generation override for the curvature threshold.",
    )
    parser.add_argument(
        "--stage2-order",
        type=int,
        default=None,
        help="Optional Stage 2 generation override for the banana-coil Fourier order.",
    )
    parser.add_argument(
        "--stage2-tf-current-A",
        type=float,
        default=None,
        help="Optional Stage 2 generation override for the per-TF-coil current.",
    )
    parser.add_argument(
        "--stage2-toroidal-flux",
        type=float,
        default=None,
        help="Optional Stage 2 generation override for the normalized toroidal flux label.",
    )
    parser.add_argument(
        "--stage2-basin-seed",
        type=int,
        default=None,
        help="Optional Stage 2 generation basin-hopping seed override.",
    )
    parser.add_argument(
        "--recovery-output-root",
        "--repair-output-root",
        dest="recovery_output_root",
        default=None,
        help="Optional output root for recovery-only artifacts. Defaults to <output-root>/recovery.",
    )
    parser.add_argument(
        "--recovery-maxiter",
        "--repair-maxiter",
        dest="recovery_maxiter",
        type=int,
        default=80,
        help="Maximum optimizer iterations allowed for the bounded recovery stage.",
    )
    parser.add_argument(
        "--recovery-ftol",
        "--repair-ftol",
        dest="recovery_ftol",
        type=float,
        default=1.0e-15,
        help="L-BFGS-B ftol forwarded into the recovery stage.",
    )
    parser.add_argument(
        "--recovery-gtol",
        "--repair-gtol",
        dest="recovery_gtol",
        type=float,
        default=1.0e-15,
        help="L-BFGS-B gtol forwarded into the recovery stage.",
    )
    parser.add_argument(
        "--recovery-stage",
        "--repair-stage",
        dest="recovery_stage",
        choices=RECOVERY_STAGES,
        default=RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM,
        help="Bounded recovery strategy used between the probe and the full single-stage run.",
    )
    parser.add_argument(
        "--skip-recovery",
        "--skip-repair",
        dest="skip_recovery",
        action="store_true",
        help="Skip the recovery stage even if the donor fails the probe.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def update_results_json(
    results_path: Path,
    payload: dict[str, object],
) -> dict:
    results = load_json(results_path)
    results.update(payload)
    write_json(results_path, results)
    return results


def validate_handoff_cli_args(args: argparse.Namespace) -> None:
    if args.bootability_iota_tolerance <= 0.0:
        raise ValueError("--bootability-iota-tolerance must be positive")
    if args.recovery_only and args.skip_recovery:
        raise ValueError("--recovery-only cannot be combined with --skip-recovery")
    if args.stage2_bs_path is not None and (
        args.stage2_profile is not None or args.stage2_spec_json is not None
    ):
        raise ValueError(
            "Choose either --stage2-bs-path or one Stage 2 generation source, not both."
        )
    if (
        args.stage2_bs_path is None
        and args.stage2_profile is None
        and args.stage2_spec_json is None
    ):
        raise ValueError(
            "Provide --stage2-bs-path or one of --stage2-profile/--stage2-spec-json."
        )


def resolve_output_root(path_or_none, *, default_root: Path) -> Path:
    if path_or_none is None:
        return default_root
    return resolved_path(path_or_none)


def resolve_summary_path(args: argparse.Namespace, output_root: Path) -> Path:
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is not None:
        return summary_path
    return output_root / DEFAULT_SUMMARY_JSON


def load_stage2_seed_metadata_for_handoff(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
) -> tuple[Path, dict]:
    _, stage2_results_path, stage2_results = load_validated_stage2_seed_results(
        args,
        owner_label="run_stage2_to_single_stage.py",
        stage2_bs_path=stage2_bs_path,
    )
    return stage2_results_path, stage2_results


def build_stage2_generation_args(
    args: argparse.Namespace,
    *,
    output_root: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        python_executable=args.python_executable,
        dry_run=args.dry_run,
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        profile=args.stage2_profile,
        stage2_spec_json=args.stage2_spec_json,
        equilibria_dir=args.equilibria_dir,
        output_root=str(output_root),
        summary_json=None,
        stage2_timeout_seconds=args.stage2_timeout_seconds,
        cc_threshold=args.stage2_cc_threshold,
        curvature_threshold=args.stage2_curvature_threshold,
        order=args.stage2_order,
        tf_current_A=args.stage2_tf_current_A,
        toroidal_flux=args.stage2_toroidal_flux,
        basin_seed=args.stage2_basin_seed,
        constraint_method="alm",
        stage2_iota_mode="report",
        stage2_iota_target=resolve_single_stage_iota_target_arg(args),
        stage2_iota_tolerance=5.0e-3,
        stage2_iota_weight=1.0,
        stage2_iota_vol_target=args.vol_target,
        stage2_iota_constraint_weight=1.0,
        stage2_iota_num_tf_coils=20,
        stage2_iota_nphi=91,
        stage2_iota_ntheta=32,
        stage2_iota_mpol=8,
        stage2_iota_ntor=6,
    )


def resolve_stage2_input(
    args: argparse.Namespace,
    *,
    output_root: Path,
) -> dict[str, object]:
    if args.stage2_bs_path is not None:
        stage2_bs_path = resolved_path(args.stage2_bs_path)
        stage2_results_path, stage2_results = load_stage2_seed_metadata_for_handoff(
            args,
            stage2_bs_path=stage2_bs_path,
        )
        return {
            "source": "existing_artifact",
            "stage2_bs_path": stage2_bs_path,
            "stage2_results_path": stage2_results_path,
            "stage2_results": stage2_results,
            "artifact_reused": True,
            "command": None,
            "config_source": None,
        }

    stage2_output_root = resolve_output_root(
        args.stage2_output_root,
        default_root=output_root / "stage2",
    )
    stage2_args = build_stage2_generation_args(args, output_root=stage2_output_root)
    resolved_spec, config_source = stage2_alm_runner.resolve_stage2_spec_payload(
        stage2_args
    )
    config = stage2_alm_runner.build_stage2_alm_config(
        stage2_args,
        resolved_spec=resolved_spec,
    )
    constraint_metadata = stage2_alm_runner.build_stage2_constraint_artifacts(
        args=stage2_args,
        config=config,
        source_label=config_source,
    )
    artifact_path = stage2_alm_runner.resolve_stage2_artifact_path(config)
    stage2_command = stage2_alm_runner.build_stage2_command(
        config,
        constraint_override_reason=constraint_metadata["OVERRIDE_REASON"],
        constraint_profile_label=constraint_metadata["CONSTRAINT_PROFILE"],
        python_executable=args.python_executable,
    )
    artifact_reused = artifact_path.exists()
    if not args.dry_run:
        ensured_artifact = ensure_stage2_artifact_result(
            config,
            constraint_override_reason=constraint_metadata["OVERRIDE_REASON"],
            constraint_profile_label=constraint_metadata["CONSTRAINT_PROFILE"],
            python_executable=args.python_executable,
            timeout_seconds=timeout_or_none(args.stage2_timeout_seconds),
            dry_run=False,
        )
        artifact_path = ensured_artifact.artifact_path
        artifact_reused = ensured_artifact.artifact_reused
    if args.dry_run and not artifact_path.exists():
        return {
            "source": "generated_artifact",
            "stage2_bs_path": artifact_path,
            "stage2_results_path": artifact_path.with_name("results.json"),
            "stage2_results": None,
            "artifact_reused": artifact_reused,
            "command": stage2_command,
            "config_source": config_source,
        }
    stage2_results_path, stage2_results = stage2_alm_runner.load_validated_stage2_artifact(
        config,
        constraint_metadata=constraint_metadata,
    )
    return {
        "source": "generated_artifact",
        "stage2_bs_path": artifact_path,
        "stage2_results_path": stage2_results_path,
        "stage2_results": stage2_results,
        "artifact_reused": artifact_reused,
        "command": stage2_command,
        "config_source": config_source,
    }


def run_single_stage_command_with_salvage(
    command: list[str],
    *,
    output_root: Path,
    timeout_seconds: float | None,
) -> tuple[str, Path, dict]:
    previous_results_snapshot = goal_mode_runner.snapshot_single_results_paths(
        output_root
    )
    previous_preserved_snapshot = (
        goal_mode_runner.snapshot_single_stage_preserved_results_paths(output_root)
    )
    run_command(
        command,
        timeout_seconds=timeout_seconds,
        dry_run=False,
    )
    return goal_mode_runner.load_single_stage_results_with_salvage(
        output_root,
        previous_results_snapshot=previous_results_snapshot,
        previous_preserved_snapshot=previous_preserved_snapshot,
    )


def build_probe_status(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    stage2_results: dict,
    stage: str,
    warm_start_boozer_surface_path: Path | None = None,
) -> dict[str, object]:
    effective_seed_surf_path = warm_start_boozer_surface_path
    if effective_seed_surf_path is None:
        explicit_seed_surf_path = getattr(args, "stage2_seed_surf_path", None)
        if explicit_seed_surf_path is not None:
            effective_seed_surf_path = resolved_path(explicit_seed_surf_path)
    constraint_weight = (
        None
        if float(args.constraint_weight) < 0.0
        else float(args.constraint_weight)
    )
    requested_finite_current_mode = getattr(args, "finite_current_mode", None)
    finite_current_mode = resolve_stage2_finite_current_mode(
        stage2_results,
        requested_finite_current_mode,
    )
    current_settings = resolve_plasma_current_settings_for_num_surfaces(
        raw_boozer_I=args.boozer_I,
        plasma_current_A=args.plasma_current_A,
        finite_current_mode=finite_current_mode,
        default_plasma_current_A=float(stage2_results.get("PROXY_PLASMA_CURRENT_A", 0.0)),
        num_surfaces=args.num_surfaces,
        requested_finite_current_mode=requested_finite_current_mode,
    )
    return probe_stage2_seed_bootability(
        stage2_bs_path=stage2_bs_path,
        stage2_artifact_results=stage2_results,
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        equilibria_dir=args.equilibria_dir,
        equilibrium_path=resolved_handoff_equilibrium_path(args),
        database_equilibria_dir=DATABASE_EQUILIBRIA_DIR,
        num_tf_coils=args.num_tf_coils,
        nphi=args.nphi,
        ntheta=args.ntheta,
        mpol=args.mpol,
        ntor=args.ntor,
        vol_target=args.vol_target,
        iota_target=resolve_single_stage_iota_target_arg(args),
        iota_tolerance=args.bootability_iota_tolerance,
        constraint_weight=constraint_weight,
        boozer_I=current_settings.boozer_I,
        stage=stage,
        stage2_seed_surf_path=effective_seed_surf_path,
    )


def _required_stage2_float(stage2_results: dict, key: str) -> float:
    value = stage2_results.get(key)
    if value is None:
        raise ValueError(f"Stage 2 repair requires {key} in results.json.")
    return float(value)


def _required_stage2_int(stage2_results: dict, key: str) -> int:
    value = stage2_results.get(key)
    if value is None:
        raise ValueError(f"Stage 2 repair requires {key} in results.json.")
    return int(value)


def _required_stage2_str(stage2_results: dict, key: str) -> str:
    value = stage2_results.get(key)
    if value is None or value == "":
        raise ValueError(f"Stage 2 repair requires {key} in results.json.")
    return str(value)


def _required_pre_boozer_stage2_finite_current_mode(stage2_results: dict) -> str:
    finite_current_mode = _required_stage2_str(stage2_results, "FINITE_CURRENT_MODE")
    if finite_current_mode != "wataru_proxy_field":
        raise ValueError(
            "Pre-Boozer Stage 2 repair requires "
            "FINITE_CURRENT_MODE='wataru_proxy_field'."
        )
    return finite_current_mode


def _stage2_float_override(
    explicit_value: float | None,
    stage2_results: dict,
    key: str,
) -> float:
    if explicit_value is not None:
        return float(explicit_value)
    return _required_stage2_float(stage2_results, key)


def _stage2_int_override(
    explicit_value: int | None,
    stage2_results: dict,
    key: str,
) -> int:
    if explicit_value is not None:
        return int(explicit_value)
    return _required_stage2_int(stage2_results, key)


def build_pre_boozer_stage2_repair_config(
    args: argparse.Namespace,
    *,
    original_stage2_results: dict,
    recovery_output_root: Path,
) -> Stage2ArtifactConfig:
    return Stage2ArtifactConfig(
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        output_root=recovery_output_root,
        equilibria_dir=(
            None
            if args.equilibria_dir is None
            else str(resolved_path(args.equilibria_dir))
        ),
        tf_current_A=_stage2_float_override(
            args.stage2_tf_current_A,
            original_stage2_results,
            "TF_CURRENT_A",
        ),
        major_radius=_required_stage2_float(original_stage2_results, "MAJOR_RADIUS"),
        toroidal_flux=_stage2_float_override(
            args.stage2_toroidal_flux,
            original_stage2_results,
            "TOROIDAL_FLUX",
        ),
        length_weight=_required_stage2_float(original_stage2_results, "LENGTH_WEIGHT"),
        cc_weight=_required_stage2_float(original_stage2_results, "CC_WEIGHT"),
        cc_threshold=_stage2_float_override(
            args.stage2_cc_threshold,
            original_stage2_results,
            "CC_THRESHOLD",
        ),
        curvature_weight=_required_stage2_float(
            original_stage2_results,
            "CURVATURE_WEIGHT",
        ),
        curvature_threshold=_stage2_float_override(
            args.stage2_curvature_threshold,
            original_stage2_results,
            "CURVATURE_THRESHOLD",
        ),
        banana_surf_radius=_stage2_float_override(
            args.banana_surf_radius,
            original_stage2_results,
            "banana_surf_radius",
        ),
        order=_stage2_int_override(args.stage2_order, original_stage2_results, "order"),
        constraint_method="alm",
        alm_max_outer_iters=args.alm_max_outer_iters,
        alm_penalty_init=args.alm_penalty_init,
        alm_penalty_scale=args.alm_penalty_scale,
        alm_penalty_max=args.alm_penalty_max,
        alm_feas_tol=args.alm_feas_tol,
        alm_stationarity_tol=args.alm_stationarity_tol,
        alm_trust_radius_init=args.alm_trust_radius_init,
        alm_trust_radius_min=args.alm_trust_radius_min,
        alm_trust_radius_shrink=args.alm_trust_radius_shrink,
        alm_trust_radius_grow=args.alm_trust_radius_grow,
        alm_max_inner_attempts=args.alm_max_inner_attempts,
        alm_max_subproblem_continuations=args.alm_max_subproblem_continuations,
        alm_distance_smoothing=args.alm_distance_smoothing,
        alm_curvature_smoothing=args.alm_curvature_smoothing,
        banana_init_current_A=_required_stage2_float(
            original_stage2_results,
            "BANANA_INIT_CURRENT_A",
        ),
        banana_current_max_A=_required_stage2_float(
            original_stage2_results,
            "BANANA_CURRENT_MAX_A",
        ),
        length_target=(
            float(args.length_target)
            if args.length_target is not None
            else _required_stage2_float(original_stage2_results, "LENGTH_TARGET")
        ),
        finite_current_mode=_required_pre_boozer_stage2_finite_current_mode(
            original_stage2_results,
        ),
        proxy_plasma_current_A=_required_stage2_float(
            original_stage2_results,
            "PROXY_PLASMA_CURRENT_A",
        ),
        vf_current_A=_required_stage2_float(original_stage2_results, "VF_CURRENT_A"),
        vf_template_path=original_stage2_results.get("VF_TEMPLATE_PATH"),
        stage2_iota_mode="report",
        stage2_iota_target=resolve_single_stage_iota_target_arg(args),
        stage2_iota_tolerance=5.0e-3,
        stage2_iota_weight=1.0,
        stage2_iota_vol_target=args.vol_target,
        stage2_iota_constraint_weight=1.0,
        stage2_iota_num_tf_coils=20,
        stage2_iota_nphi=91,
        stage2_iota_ntheta=32,
        stage2_iota_mpol=8,
        stage2_iota_ntor=6,
    )


def build_pre_boozer_stage2_repair_command(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    original_stage2_results: dict,
    recovery_output_root: Path,
) -> list[str]:
    config = build_pre_boozer_stage2_repair_config(
        args,
        original_stage2_results=original_stage2_results,
        recovery_output_root=recovery_output_root,
    )
    command = build_stage2_command(
        config,
        constraint_override_reason="pre_boozer_repair",
        constraint_profile_label="pre_boozer_repair",
        python_executable=args.python_executable,
    )
    command.extend(
        [
            "--stage2-bs-path",
            str(stage2_bs_path),
            "--maxiter",
            str(args.recovery_maxiter),
            "--maxcor",
            str(args.maxcor),
            "--ftol",
            str(args.recovery_ftol),
            "--gtol",
            str(args.recovery_gtol),
            "--nphi",
            str(args.nphi),
            "--ntheta",
            str(args.ntheta),
        ]
    )
    equilibrium_path = resolved_handoff_equilibrium_path(args)
    if equilibrium_path is not None:
        command.extend(["--equilibrium-path", str(equilibrium_path)])
    if getattr(args, "seed_order_upgrade", None) is not None:
        command.extend(["--seed-order-upgrade", str(args.seed_order_upgrade)])
    return command


def build_recovery_command(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    recovery_output_root: Path,
    original_stage2_results: dict | None = None,
) -> list[str]:
    if args.recovery_stage == RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM:
        if original_stage2_results is None:
            raise ValueError(
                "Pre-Boozer Stage 2 repair requires the original Stage 2 results.json."
            )
        return build_pre_boozer_stage2_repair_command(
            args,
            stage2_bs_path=stage2_bs_path,
            original_stage2_results=original_stage2_results,
            recovery_output_root=recovery_output_root,
        )
    recovery_args = SimpleNamespace(
        python_executable=args.python_executable,
        stage2_bs_path=str(stage2_bs_path),
        stage2_seed_role="recovery",
        output_root=str(recovery_output_root),
        allow_init_only_stage2_seed=args.allow_init_only_stage2_seed,
        equilibria_dir=args.equilibria_dir,
        equilibrium_path=resolved_handoff_equilibrium_path(args),
        stage2_seed_surf_path=getattr(args, "stage2_seed_surf_path", None),
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        nphi=args.nphi,
        ntheta=args.ntheta,
        mpol=args.mpol,
        ntor=args.ntor,
        constraint_weight=args.constraint_weight,
        boozer_I=args.boozer_I,
        plasma_current_A=args.plasma_current_A,
        single_stage_banana_current_mode=args.single_stage_banana_current_mode,
        num_tf_coils=args.num_tf_coils,
        banana_surf_radius=args.banana_surf_radius,
        stage2_seed_tf_current_A=args.stage2_seed_tf_current_A,
        maxiter=args.recovery_maxiter,
        iota_target=args.iota_target,
        flip_banana=args.flip_banana,
        vol_target=args.vol_target,
        cc_dist=args.cc_dist,
        cs_dist=args.cs_dist,
        curvature_threshold=args.curvature_threshold,
        hardware_search_mode=recovery_runner.DEFAULT_HARDWARE_SEARCH_MODE,
        alm_max_outer_iters=args.alm_max_outer_iters,
        alm_max_subproblem_continuations=args.alm_max_subproblem_continuations,
        alm_penalty_init=args.alm_penalty_init,
        alm_penalty_scale=args.alm_penalty_scale,
        alm_penalty_max=args.alm_penalty_max,
        alm_feas_tol=args.alm_feas_tol,
        alm_stationarity_tol=args.alm_stationarity_tol,
        alm_trust_radius_init=args.alm_trust_radius_init,
        alm_trust_radius_min=args.alm_trust_radius_min,
        alm_trust_radius_shrink=args.alm_trust_radius_shrink,
        alm_trust_radius_grow=args.alm_trust_radius_grow,
        alm_max_inner_attempts=args.alm_max_inner_attempts,
        alm_distance_smoothing=args.alm_distance_smoothing,
        alm_curvature_smoothing=args.alm_curvature_smoothing,
        alm_qs_threshold=(
            recovery_runner.DEFAULT_ALM_QS_THRESHOLD
            if args.alm_qs_threshold is None
            else args.alm_qs_threshold
        ),
        alm_boozer_threshold=(
            recovery_runner.DEFAULT_ALM_BOOZER_THRESHOLD
            if args.alm_boozer_threshold is None
            else args.alm_boozer_threshold
        ),
        alm_iota_penalty_threshold=(
            recovery_runner.DEFAULT_ALM_IOTA_PENALTY_THRESHOLD
            if args.alm_iota_penalty_threshold is None
            else args.alm_iota_penalty_threshold
        ),
        alm_length_penalty_threshold=(
            recovery_runner.DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD
            if args.alm_length_penalty_threshold is None
            else args.alm_length_penalty_threshold
        ),
    )
    command = recovery_runner.build_single_stage_thresholded_physics_command(
        recovery_args
    )
    command.extend(
        ["--ftol", str(args.recovery_ftol), "--gtol", str(args.recovery_gtol)]
    )
    return command


def run_recovery_stage(
    args: argparse.Namespace,
    *,
    original_stage2_bs_path: Path,
    original_stage2_results_path: Path,
    original_stage2_results: dict,
    recovery_output_root: Path,
) -> dict[str, object]:
    command = build_recovery_command(
        args,
        stage2_bs_path=original_stage2_bs_path,
        original_stage2_results=original_stage2_results,
        recovery_output_root=recovery_output_root,
    )
    if args.dry_run:
        return {
            "status": "dry_run",
            "command": command,
            "output_root": str(recovery_output_root),
            "recovery_termination_reason": "dry_run",
        }
    if args.recovery_stage == RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM:
        return run_pre_boozer_stage2_repair(
            args,
            command=command,
            original_stage2_bs_path=original_stage2_bs_path,
            original_stage2_results_path=original_stage2_results_path,
            original_stage2_results=original_stage2_results,
            recovery_output_root=recovery_output_root,
        )
    try:
        result_source, results_path, results = run_single_stage_command_with_salvage(
            command,
            output_root=recovery_output_root,
            timeout_seconds=timeout_or_none(args.single_stage_timeout_seconds),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {
            "status": "failed",
            "command": command,
            "output_root": str(recovery_output_root),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "recovery_termination_reason": (
                "timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "subprocess_failed"
            ),
        }
    artifact_bundle = goal_mode_runner.single_stage_artifact_bundle_from_results(
        result_source,
        results_path,
    )
    recovered_bs_path = artifact_bundle["bs_path"]
    if not recovered_bs_path.exists():
        return {
            "status": "failed",
            "command": command,
            "output_root": str(recovery_output_root),
            "results_path": str(results_path),
            "result_source": result_source,
            "error_type": "FileNotFoundError",
            "error_message": (
                "Recovery stage did not materialize "
                f"{recovered_bs_path.name}: {recovered_bs_path}"
            ),
            "recovery_termination_reason": "missing_recovery_artifact",
        }
    # The recovered BiotSavart artifact is a single-stage output, but the probe
    # expects Stage 2 artifact metadata (TF_CURRENT_A, NUM_TF_COILS, FINITE_CURRENT_MODE,
    # banana_surf_radius, MAJOR_RADIUS, TOROIDAL_FLUX). Those invariants did not change
    # during recovery — only the banana-coil DOFs did — so we pass the original Stage 2
    # results through while pointing the probe at the recovered coils file.
    recovery_probe = build_probe_status(
        args,
        stage2_bs_path=recovered_bs_path,
        stage2_results=original_stage2_results,
        stage=BOOTABILITY_STAGE_RECOVERY,
        warm_start_boozer_surface_path=artifact_bundle["outer_boozer_surface_path"],
    )
    recovery_iters = results.get("iterations")
    recovery_iters_value = None if recovery_iters is None else int(recovery_iters)
    recovery_succeeded = bootability_passes(recovery_probe)
    recovery_termination_reason = (
        "bootable" if recovery_succeeded else "not_bootable_after_budget"
    )
    if recovery_succeeded:
        results = stamp_and_validate_stage2_production_handoff(
            results_path,
            recovery_probe,
            source_stage2_results=original_stage2_results,
            original_stage2_bs_path=original_stage2_bs_path,
            original_stage2_results_path=original_stage2_results_path,
            recovery_attempted=True,
            recovery_succeeded=True,
            recovery_iters=recovery_iters_value,
            recovery_termination_reason=recovery_termination_reason,
            seed_source=SEED_SOURCE_RECOVERED_STAGE2_DONOR,
        )
    else:
        handoff_payload = build_bootability_recovery_payload_fields(
            recovery_probe,
            stage2_bs_path=str(original_stage2_bs_path),
            stage2_results_path=str(original_stage2_results_path),
            recovery_attempted=True,
            recovery_succeeded=False,
            recovery_iters=recovery_iters_value,
            recovery_termination_reason=recovery_termination_reason,
        )
        handoff_payload["UNIFIED_SEED_SOURCE"] = SEED_SOURCE_RECOVERED_STAGE2_DONOR
        results = update_results_json(results_path, handoff_payload)
    return {
        "status": "completed",
        "command": command,
        "output_root": str(recovery_output_root),
        "results_path": str(results_path),
        "result_source": result_source,
        "results": results,
        "recovered_bs_path": str(recovered_bs_path),
        "warm_start_surface_stem": str(artifact_bundle["surface_stem"]),
        "recovery_probe": recovery_probe,
        "recovery_succeeded": recovery_succeeded,
        "recovery_iters": recovery_iters_value,
        "recovery_termination_reason": recovery_termination_reason,
    }


def run_pre_boozer_stage2_repair(
    args: argparse.Namespace,
    *,
    command: list[str],
    original_stage2_bs_path: Path,
    original_stage2_results_path: Path,
    original_stage2_results: dict,
    recovery_output_root: Path,
) -> dict[str, object]:
    try:
        run_command(
            command,
            timeout_seconds=timeout_or_none(args.single_stage_timeout_seconds),
            dry_run=False,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {
            "status": "failed",
            "command": command,
            "output_root": str(recovery_output_root),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "recovery_termination_reason": (
                "timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "subprocess_failed"
            ),
        }
    recovered_bs_path = resolve_stage2_artifact_path(
        build_pre_boozer_stage2_repair_config(
            args,
            original_stage2_results=original_stage2_results,
            recovery_output_root=recovery_output_root,
        )
    )
    results_path = recovered_bs_path.with_name("results.json")
    if not recovered_bs_path.exists():
        return {
            "status": "failed",
            "command": command,
            "output_root": str(recovery_output_root),
            "results_path": str(results_path),
            "result_source": RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM,
            "error_type": "FileNotFoundError",
            "error_message": (
                "Pre-Boozer repair did not materialize "
                f"{recovered_bs_path.name}: {recovered_bs_path}"
            ),
            "recovery_termination_reason": "missing_recovery_artifact",
        }
    results = load_json(results_path)
    recovery_probe = build_probe_status(
        args,
        stage2_bs_path=recovered_bs_path,
        stage2_results=results,
        stage=BOOTABILITY_STAGE_RECOVERY,
    )
    recovery_iters = results.get("iterations")
    recovery_iters_value = None if recovery_iters is None else int(recovery_iters)
    recovery_succeeded = bootability_passes(recovery_probe)
    recovery_termination_reason = (
        "bootable" if recovery_succeeded else "not_bootable_after_budget"
    )
    if recovery_succeeded:
        results = stamp_and_validate_stage2_production_handoff(
            results_path,
            recovery_probe,
            source_stage2_results=original_stage2_results,
            iota_mode_stage2_results=results,
            original_stage2_bs_path=original_stage2_bs_path,
            original_stage2_results_path=original_stage2_results_path,
            recovery_attempted=True,
            recovery_succeeded=True,
            recovery_iters=recovery_iters_value,
            recovery_termination_reason=recovery_termination_reason,
            seed_source=SEED_SOURCE_RECOVERED_STAGE2_DONOR,
        )
    else:
        handoff_payload = build_bootability_recovery_payload_fields(
            recovery_probe,
            stage2_bs_path=str(original_stage2_bs_path),
            stage2_results_path=str(original_stage2_results_path),
            recovery_attempted=True,
            recovery_succeeded=False,
            recovery_iters=recovery_iters_value,
            recovery_termination_reason=recovery_termination_reason,
        )
        handoff_payload["UNIFIED_SEED_SOURCE"] = SEED_SOURCE_RECOVERED_STAGE2_DONOR
        results = update_results_json(results_path, handoff_payload)
    return {
        "status": "completed",
        "command": command,
        "output_root": str(recovery_output_root),
        "results_path": str(results_path),
        "result_source": RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM,
        "results": results,
        "recovered_bs_path": str(recovered_bs_path),
        "warm_start_surface_stem": None,
        "recovery_probe": recovery_probe,
        "recovery_succeeded": recovery_succeeded,
        "recovery_iters": recovery_iters_value,
        "recovery_termination_reason": recovery_termination_reason,
    }


def _with_warm_start_surface_stem(
    args: argparse.Namespace,
    warm_start_surface_stem: Path | None,
) -> argparse.Namespace:
    if warm_start_surface_stem is None:
        return args
    return SimpleNamespace(
        **vars(args),
        warm_start_surface_stem=str(warm_start_surface_stem),
    )


def _with_stage2_seed_role(
    args: argparse.Namespace,
    stage2_seed_role: str,
) -> argparse.Namespace:
    return SimpleNamespace(**vars(args), stage2_seed_role=stage2_seed_role)


def _wout_off_spec_for_handoff(stage2_results: dict) -> bool:
    if "WOUT_OFF_SPEC" not in stage2_results:
        raise ValueError("Source Stage 2 handoff results missing WOUT_OFF_SPEC.")
    wout_off_spec = stage2_results["WOUT_OFF_SPEC"]
    if wout_off_spec is not True and wout_off_spec is not False:
        raise ValueError("Source Stage 2 handoff WOUT_OFF_SPEC must be boolean.")
    return wout_off_spec


def stage2_production_handoff_payload(
    bootability_status: dict[str, object],
    *,
    source_stage2_results: dict,
    iota_mode_stage2_results: dict | None = None,
    original_stage2_bs_path: Path,
    original_stage2_results_path: Path,
    recovery_attempted: bool,
    recovery_succeeded: bool,
    recovery_iters: int | None,
    recovery_termination_reason: str | None,
    seed_source: str,
) -> dict[str, object]:
    payload = handoff_results_payload(
        bootability_status,
        original_stage2_bs_path=original_stage2_bs_path,
        original_stage2_results_path=original_stage2_results_path,
        recovery_attempted=recovery_attempted,
        recovery_succeeded=recovery_succeeded,
        recovery_iters=recovery_iters,
        recovery_termination_reason=recovery_termination_reason,
        seed_source=seed_source,
    )
    payload.update(
        build_stage2_seed_production_handoff_fields(
            wout_off_spec=_wout_off_spec_for_handoff(source_stage2_results),
        )
    )
    stage2_iota_source = iota_mode_stage2_results or source_stage2_results
    stage2_iota_mode = stage2_iota_source.get("STAGE2_IOTA_MODE")
    if stage2_iota_mode is not None:
        payload["STAGE2_IOTA_MODE"] = stage2_iota_mode
    return payload


def stamp_and_validate_stage2_production_handoff(
    results_path: Path,
    bootability_status: dict[str, object],
    *,
    source_stage2_results: dict,
    iota_mode_stage2_results: dict | None = None,
    original_stage2_bs_path: Path,
    original_stage2_results_path: Path,
    recovery_attempted: bool,
    recovery_succeeded: bool,
    recovery_iters: int | None,
    recovery_termination_reason: str | None,
    seed_source: str,
) -> dict:
    payload = stage2_production_handoff_payload(
        bootability_status,
        source_stage2_results=source_stage2_results,
        iota_mode_stage2_results=iota_mode_stage2_results,
        original_stage2_bs_path=original_stage2_bs_path,
        original_stage2_results_path=original_stage2_results_path,
        recovery_attempted=recovery_attempted,
        recovery_succeeded=recovery_succeeded,
        recovery_iters=recovery_iters,
        recovery_termination_reason=recovery_termination_reason,
        seed_source=seed_source,
    )
    results = {**load_json(results_path), **payload}
    validate_stage2_seed_bootability_contract(results)
    write_json(results_path, results)
    return results


def build_full_single_stage_command(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    full_output_root: Path,
    warm_start_surface_stem: Path | None = None,
) -> list[str]:
    goal_mode_args = _with_stage2_seed_role(args, "handoff")
    goal_mode_args = _with_warm_start_surface_stem(
        goal_mode_args,
        warm_start_surface_stem,
    )
    command = goal_mode_runner.build_single_stage_goal_mode_command(
        goal_mode_args,
        goal_mode=args.goal_mode,
        stage2_bs_path=stage2_bs_path,
        case_output_root=full_output_root / args.goal_mode,
    )
    return command


def run_full_single_stage(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    full_output_root: Path,
    warm_start_surface_stem: Path | None = None,
) -> dict[str, object]:
    full_payload = {
        "status": "dry_run" if args.dry_run else "completed",
        "command": build_full_single_stage_command(
            args,
            stage2_bs_path=stage2_bs_path,
            full_output_root=full_output_root,
            warm_start_surface_stem=warm_start_surface_stem,
        ),
        "output_root": str(full_output_root),
    }
    if args.dry_run:
        return full_payload
    goal_mode_output_root = full_output_root
    goal_mode_args = _with_stage2_seed_role(args, "handoff")
    goal_mode_args = _with_warm_start_surface_stem(
        goal_mode_args,
        warm_start_surface_stem,
    )
    result_payload = goal_mode_runner.run_goal_mode_case(
        goal_mode_args,
        goal_mode=args.goal_mode,
        stage2_bs_path=stage2_bs_path,
        output_root=goal_mode_output_root,
    )
    full_payload.update(
        {
            "results_path": str(result_payload["results_path"]),
            "result_source": result_payload["result_source"],
            "results": result_payload["results"],
        }
    )
    return full_payload


def handoff_results_payload(
    bootability_status: dict[str, object],
    *,
    original_stage2_bs_path: Path,
    original_stage2_results_path: Path,
    recovery_attempted: bool,
    recovery_succeeded: bool,
    recovery_iters: int | None,
    recovery_termination_reason: str | None,
    seed_source: str,
) -> dict[str, object]:
    payload = build_bootability_recovery_payload_fields(
        bootability_status,
        stage2_bs_path=str(original_stage2_bs_path),
        stage2_results_path=str(original_stage2_results_path),
        recovery_attempted=recovery_attempted,
        recovery_succeeded=recovery_succeeded,
        recovery_iters=recovery_iters,
        recovery_termination_reason=recovery_termination_reason,
    )
    payload["UNIFIED_SEED_SOURCE"] = seed_source
    return payload


def next_required_lane(
    *,
    initial_probe: dict[str, object] | None,
    full_payload: dict[str, object] | None,
    blocking_reason: str | None,
) -> str:
    if initial_probe is None:
        return LANE_BOOZER_ACTIVATION
    if blocking_reason == BLOCKING_REASON_PRE_BOOZER_REPAIR_REQUIRED:
        return LANE_PRE_BOOZER_REPAIR
    if full_payload is None:
        return LANE_SINGLE_STAGE_TARGET
    if full_payload.get("status") == "completed":
        return LANE_PROMOTION
    return LANE_SINGLE_STAGE_TARGET


def build_summary(
    args: argparse.Namespace,
    *,
    stage2_input: dict[str, object],
    initial_probe: dict[str, object] | None,
    recovery_payload: dict[str, object] | None,
    full_payload: dict[str, object] | None,
    blocking_reason: str | None = None,
) -> dict[str, object]:
    summary = {
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "mode": (
            "probe_only"
            if args.probe_only
            else "recovery_only"
            if args.recovery_only
            else "full"
        ),
        "goal_mode": args.goal_mode,
        "dry_run": bool(args.dry_run),
        "lane_sequence": list(LANE_SEQUENCE),
        "next_required_lane": next_required_lane(
            initial_probe=initial_probe,
            full_payload=full_payload,
            blocking_reason=blocking_reason,
        ),
        "stage2_input": {
            "source": stage2_input["source"],
            "stage2_bs_path": str(stage2_input["stage2_bs_path"]),
            "stage2_results_path": str(stage2_input["stage2_results_path"]),
            "artifact_reused": stage2_input["artifact_reused"],
            "config_source": stage2_input.get("config_source"),
            "command": stage2_input.get("command"),
        },
        "bootability_probe": initial_probe,
        "recovery": recovery_payload,
        "full_single_stage": full_payload,
        "blocking_reason": blocking_reason,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_handoff_cli_args(args)
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = resolve_summary_path(args, output_root)

    stage2_input = resolve_stage2_input(args, output_root=output_root)
    if stage2_input["stage2_results"] is None:
        summary = build_summary(
            args,
            stage2_input=stage2_input,
            initial_probe=None,
            recovery_payload=None,
            full_payload=None,
        )
        write_json(summary_path, summary)
        return 0

    original_stage2_bs_path = stage2_input["stage2_bs_path"]
    original_stage2_results_path = stage2_input["stage2_results_path"]
    stage2_results = stage2_input["stage2_results"]
    initial_probe = build_probe_status(
        args,
        stage2_bs_path=original_stage2_bs_path,
        stage2_results=stage2_results,
        stage=BOOTABILITY_STAGE_PROBE,
    )
    if args.probe_only:
        summary = build_summary(
            args,
            stage2_input=stage2_input,
            initial_probe=initial_probe,
            recovery_payload=None,
            full_payload=None,
        )
        write_json(summary_path, summary)
        return 0

    recovery_payload = None
    handoff_bs_path = original_stage2_bs_path
    handoff_warm_start_surface_stem = None
    handoff_bootability = initial_probe
    recovery_attempted = False
    recovery_succeeded = False
    recovery_iters = None
    recovery_termination_reason = None
    seed_source = SEED_SOURCE_DIRECT_STAGE2_DONOR

    if not bootability_passes(initial_probe):
        if args.skip_recovery:
            summary = build_summary(
                args,
                stage2_input=stage2_input,
                initial_probe=initial_probe,
                recovery_payload=None,
                full_payload=None,
                blocking_reason=BLOCKING_REASON_PRE_BOOZER_REPAIR_REQUIRED,
            )
            write_json(summary_path, summary)
            return 0
        recovery_payload = run_recovery_stage(
            args,
            original_stage2_bs_path=original_stage2_bs_path,
            original_stage2_results_path=original_stage2_results_path,
            original_stage2_results=stage2_results,
            recovery_output_root=resolve_output_root(
                args.recovery_output_root,
                default_root=output_root / "recovery",
            ),
        )
        recovery_attempted = True
        recovery_succeeded = bool(recovery_payload.get("recovery_succeeded"))
        recovery_iters = recovery_payload.get("recovery_iters")
        recovery_termination_reason = recovery_payload.get("recovery_termination_reason")
        if recovery_succeeded:
            handoff_bs_path = resolved_path(recovery_payload["recovered_bs_path"])
            warm_start_surface_stem = recovery_payload.get("warm_start_surface_stem")
            handoff_warm_start_surface_stem = (
                None
                if warm_start_surface_stem is None
                else resolved_path(warm_start_surface_stem)
            )
            handoff_bootability = recovery_payload["recovery_probe"]
            seed_source = SEED_SOURCE_RECOVERED_STAGE2_DONOR
        else:
            summary = build_summary(
                args,
                stage2_input=stage2_input,
                initial_probe=initial_probe,
                recovery_payload=recovery_payload,
                full_payload=None,
                blocking_reason=BLOCKING_REASON_PRE_BOOZER_REPAIR_REQUIRED,
            )
            write_json(summary_path, summary)
            return 0

    if args.recovery_only:
        summary = build_summary(
            args,
            stage2_input=stage2_input,
            initial_probe=initial_probe,
            recovery_payload=recovery_payload,
            full_payload=None,
            blocking_reason=None,
        )
        write_json(summary_path, summary)
        return 0

    if not args.dry_run:
        handoff_results_path = (
            Path(recovery_payload["results_path"])
            if recovery_succeeded and recovery_payload is not None
            else original_stage2_results_path
        )
        handoff_source_results = (
            recovery_payload["results"]
            if recovery_succeeded and recovery_payload is not None
            else stage2_results
        )
        stamp_and_validate_stage2_production_handoff(
            handoff_results_path,
            handoff_bootability,
            source_stage2_results=handoff_source_results,
            original_stage2_bs_path=original_stage2_bs_path,
            original_stage2_results_path=original_stage2_results_path,
            recovery_attempted=recovery_attempted,
            recovery_succeeded=recovery_succeeded,
            recovery_iters=recovery_iters,
            recovery_termination_reason=recovery_termination_reason,
            seed_source=seed_source,
        )

    full_output_root = output_root / "full"
    full_output_root.mkdir(parents=True, exist_ok=True)
    full_payload = run_full_single_stage(
        args,
        stage2_bs_path=handoff_bs_path,
        full_output_root=full_output_root,
        warm_start_surface_stem=handoff_warm_start_surface_stem,
    )
    if full_payload["status"] == "completed":
        full_results_path = Path(full_payload["results_path"])
        full_payload["results"] = update_results_json(
            full_results_path,
            handoff_results_payload(
                handoff_bootability,
                original_stage2_bs_path=original_stage2_bs_path,
                original_stage2_results_path=original_stage2_results_path,
                recovery_attempted=recovery_attempted,
                recovery_succeeded=recovery_succeeded,
                recovery_iters=recovery_iters,
                recovery_termination_reason=recovery_termination_reason,
                seed_source=seed_source,
            ),
        )

    summary = build_summary(
        args,
        stage2_input=stage2_input,
        initial_probe=initial_probe,
        recovery_payload=recovery_payload,
        full_payload=full_payload,
    )
    write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
