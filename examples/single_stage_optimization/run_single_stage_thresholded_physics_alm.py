from __future__ import annotations

import argparse
import json
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
    resolved_path,
    resolved_optional_path,
    timeout_or_none,
    run_command,
    snapshot_single_results_paths,
)
from banana_opt.artifact_contracts import upgrade_legacy_stage2_artifact_results  # noqa: E402

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_thresholded_physics_alm"
DEFAULT_SUMMARY_JSON = "single_stage_thresholded_physics_alm_summary.json"
DEFAULT_HARDWARE_SEARCH_MODE = "warn"
DEFAULT_ALM_QS_THRESHOLD = 3.0e-3
DEFAULT_ALM_BOOZER_THRESHOLD = 1.0e-2
DEFAULT_ALM_IOTA_PENALTY_THRESHOLD = 1.0e-4
DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-stage ALM rerun using the thresholded_physics formulation "
            "on top of an explicit Stage 2 artifact."
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
        help="Path to the Stage 2 biot_savart_opt.json seed artifact.",
    )
    parser.add_argument("--equilibria-dir", default=None)
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
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
    parser.add_argument("--iota-target", type=float, default=0.20)
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument("--cc-dist", type=float, default=0.05)
    parser.add_argument("--cs-dist", type=float, default=0.02)
    parser.add_argument("--curvature-threshold", type=float, default=40.0)
    parser.add_argument(
        "--hardware-search-mode",
        choices=["warn"],
        default=DEFAULT_HARDWARE_SEARCH_MODE,
        help=(
            "Single-surface ALM reruns require warning-only handling so the "
            "constraint model can see hardware violations instead of reverting "
            "to hard rejection."
        ),
    )
    parser.add_argument("--alm-max-outer-iters", type=int, default=20)
    parser.add_argument("--alm-max-subproblem-continuations", type=int, default=4)
    parser.add_argument("--alm-penalty-init", type=float, default=1.0)
    parser.add_argument("--alm-penalty-scale", type=float, default=10.0)
    parser.add_argument("--alm-feas-tol", type=float, default=1e-4)
    parser.add_argument("--alm-stationarity-tol", type=float, default=1e-4)
    parser.add_argument("--alm-trust-radius-init", type=float, default=0.05)
    parser.add_argument("--alm-trust-radius-min", type=float, default=1e-4)
    parser.add_argument("--alm-trust-radius-shrink", type=float, default=0.5)
    parser.add_argument("--alm-trust-radius-grow", type=float, default=1.5)
    parser.add_argument("--alm-max-inner-attempts", type=int, default=4)
    parser.add_argument("--alm-distance-smoothing", type=float, default=0.005)
    parser.add_argument("--alm-curvature-smoothing", type=float, default=0.05)
    parser.add_argument("--alm-qs-threshold", type=float, default=DEFAULT_ALM_QS_THRESHOLD)
    parser.add_argument(
        "--alm-boozer-threshold",
        type=float,
        default=DEFAULT_ALM_BOOZER_THRESHOLD,
    )
    parser.add_argument(
        "--alm-iota-penalty-threshold",
        type=float,
        default=DEFAULT_ALM_IOTA_PENALTY_THRESHOLD,
    )
    parser.add_argument(
        "--alm-length-penalty-threshold",
        type=float,
        default=DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD,
    )
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


def build_single_stage_thresholded_physics_command(
    args: argparse.Namespace,
) -> list[str]:
    stage2_bs_path = resolved_path(args.stage2_bs_path)
    output_root = resolved_path(args.output_root)
    equilibria_dir = resolved_optional_path(args.equilibria_dir)
    plasma_surf_filename = Path(args.plasma_surf_filename).name
    command = [
        args.python_executable,
        str(SINGLE_STAGE_SCRIPT_PATH),
        "--plasma-surf-filename",
        plasma_surf_filename,
        "--stage2-bs-path",
        str(stage2_bs_path),
        "--output-root",
        str(output_root),
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
        "--iota-target",
        str(args.iota_target),
        "--vol-target",
        str(args.vol_target),
        "--cc-dist",
        str(args.cc_dist),
        "--cs-dist",
        str(args.cs_dist),
        "--curvature-threshold",
        str(args.curvature_threshold),
        "--hardware-search-mode",
        args.hardware_search_mode,
        "--constraint-method",
        "alm",
        "--alm-formulation",
        "thresholded_physics",
        "--alm-max-outer-iters",
        str(args.alm_max_outer_iters),
        "--alm-max-subproblem-continuations",
        str(args.alm_max_subproblem_continuations),
        "--alm-penalty-init",
        str(args.alm_penalty_init),
        "--alm-penalty-scale",
        str(args.alm_penalty_scale),
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
        "--alm-distance-smoothing",
        str(args.alm_distance_smoothing),
        "--alm-curvature-smoothing",
        str(args.alm_curvature_smoothing),
        "--alm-qs-threshold",
        str(args.alm_qs_threshold),
        "--alm-boozer-threshold",
        str(args.alm_boozer_threshold),
        "--alm-iota-penalty-threshold",
        str(args.alm_iota_penalty_threshold),
        "--alm-length-penalty-threshold",
        str(args.alm_length_penalty_threshold),
    ]
    if equilibria_dir is not None:
        command.extend(["--equilibria-dir", str(equilibria_dir)])
    return command


def build_summary(
    args: argparse.Namespace,
    command: list[str],
    *,
    stage2_bs_path: Path | None = None,
    stage2_results_path: Path | None = None,
    stage2_results: dict | None = None,
    results_path: Path | None = None,
    results: dict | None = None,
) -> dict:
    if stage2_bs_path is None:
        stage2_bs_path = resolved_path(args.stage2_bs_path)
    output_root = resolved_path(args.output_root)
    summary = {
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "stage2_bs_path": str(stage2_bs_path),
        "output_root": str(output_root),
        "command": command,
        "dry_run": bool(args.dry_run),
    }
    if stage2_results_path is not None:
        summary["stage2_results_path"] = str(stage2_results_path)
    if stage2_results is not None:
        summary["stage2_artifact_plasma_surf_filename"] = stage2_results.get(
            "PLASMA_SURF_FILENAME"
        )
    if results_path is None or results is None:
        return summary
    summary.update(
        {
            "results_path": str(results_path),
            "termination_message": results.get("TERMINATION_MESSAGE"),
            "optimizer_success": results.get("OPTIMIZER_SUCCESS"),
            "alm_outer_iterations": results.get("ALM_OUTER_ITERATIONS"),
            "alm_final_penalty": results.get("ALM_FINAL_PENALTY"),
            "curve_curve_min_dist": results.get("CURVE_CURVE_MIN_DIST"),
            "curve_surface_min_dist": results.get("CURVE_SURFACE_MIN_DIST"),
            "surface_vessel_min_dist": results.get("SURFACE_VESSEL_MIN_DIST"),
            "max_curvature": results.get("MAX_CURVATURE"),
            "nonqs_ratio": results.get("NONQS_RATIO"),
            "boozer_residual": results.get("BOOZER_RESIDUAL"),
            "final_iota": results.get("FINAL_IOTA"),
            "hardware_constraints_ok": results.get("HARDWARE_CONSTRAINTS_OK"),
        }
    )
    return summary


def main() -> int:
    args = parse_args()
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    command = build_single_stage_thresholded_physics_command(args)
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stage2_bs_path, stage2_results_path, stage2_results = (
            maybe_load_validated_stage2_seed_metadata(args)
        )
        summary = build_summary(
            args,
            command,
            stage2_bs_path=stage2_bs_path,
            stage2_results_path=stage2_results_path,
            stage2_results=stage2_results,
        )
    else:
        stage2_bs_path, stage2_results_path, stage2_results = (
            load_validated_stage2_seed_metadata(args)
        )
        previous_snapshot = snapshot_single_results_paths(output_root)
        run_command(
            command,
            timeout_seconds=timeout_or_none(args.single_stage_timeout_seconds),
        )
        results_path = discover_single_results_path(
            output_root,
            previous_snapshot=previous_snapshot,
        )
        results = load_json(results_path)
        summary = build_summary(
            args,
            command,
            stage2_bs_path=stage2_bs_path,
            stage2_results_path=stage2_results_path,
            stage2_results=stage2_results,
            results_path=results_path,
            results=results,
        )

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
