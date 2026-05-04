from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
)
from banana_opt.artifact_contracts import upgrade_legacy_stage2_artifact_results

SINGLE_STAGE_SCRIPT_PATH = SCRIPT_DIR / "SINGLE_STAGE" / "single_stage_banana_example.py"
DRY_RUN_MARKER_FILENAME = "DRY_RUN_ONLY.txt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_thresholded_physics_alm"
DEFAULT_SUMMARY_JSON = "single_stage_thresholded_physics_alm_summary.json"
DEFAULT_ALM_QS_THRESHOLD = 3.0e-3
DEFAULT_ALM_BOOZER_THRESHOLD = 1.0e-2
DEFAULT_ALM_IOTA_PENALTY_THRESHOLD = 1.0e-4
DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0


def resolved_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def resolved_optional_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    return resolved_path(raw_path)


def timeout_or_none(timeout_seconds: float) -> float | None:
    return None if timeout_seconds <= 0.0 else float(timeout_seconds)


def first_present_value(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as infile:
        loaded = json.load(infile)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return loaded


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = SCRIPT_DIR,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return
    subprocess.run(
        list(command),
        cwd=str(cwd),
        check=True,
        timeout=timeout_seconds,
    )


def load_stage2_artifact_results(stage2_bs_path: str | Path) -> tuple[Path, dict[str, object]]:
    resolved_stage2_bs_path = resolved_path(stage2_bs_path)
    stage2_results_path = resolved_stage2_bs_path.with_name("results.json")
    return stage2_results_path, load_json(stage2_results_path)


def validate_stage2_seed_not_init_only(
    stage2_results_path: Path,
    stage2_results: Mapping[str, object],
    *,
    owner_label: str,
    allow_init_only: bool = False,
) -> None:
    if allow_init_only or stage2_results.get("init_only") is not True:
        return
    raise ValueError(
        f"{owner_label} requires a non-init-only Stage 2 artifact, but "
        f"{stage2_results_path} reports init_only=true. Pass "
        "--allow-init-only-stage2-seed to override this guard."
    )


def _single_results_matches(output_root: str | Path) -> list[Path]:
    return sorted(Path(output_root).glob("mpol=*-ntor=*/results.json"))


def snapshot_single_results_paths(output_root: str | Path) -> dict[Path, int]:
    return {
        path: path.stat().st_mtime_ns
        for path in _single_results_matches(output_root)
    }


def discover_single_results_path(
    output_root: str | Path,
    *,
    previous_snapshot: Mapping[Path, int] | None = None,
) -> Path:
    matches = _single_results_matches(output_root)
    if not matches:
        raise FileNotFoundError(
            f"Expected at least one single-stage results.json under {output_root}, found 0"
        )
    if previous_snapshot is not None:
        new_matches = [path for path in matches if path not in previous_snapshot]
        if len(new_matches) == 1:
            return new_matches[0]
        if len(new_matches) > 1:
            raise FileNotFoundError(
                "Expected exactly one new single-stage results.json after the run, "
                f"found {len(new_matches)} under {output_root}"
            )
        updated_matches = [
            path
            for path in matches
            if previous_snapshot.get(path) != path.stat().st_mtime_ns
        ]
        if len(updated_matches) == 1:
            return updated_matches[0]
        if len(updated_matches) > 1:
            raise FileNotFoundError(
                "Expected exactly one updated single-stage results.json after the run, "
                f"found {len(updated_matches)} under {output_root}"
            )
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        "Expected exactly one single-stage results.json when no prior snapshot was "
        f"available, found {len(matches)} under {output_root}"
    )


def dry_run_marker_path(output_root: str | Path) -> Path:
    return resolved_path(output_root) / DRY_RUN_MARKER_FILENAME


def write_dry_run_marker(
    output_root: str | Path,
    *,
    summary_path: Path,
    runner_label: str,
) -> Path:
    marker_path = dry_run_marker_path(output_root)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        "\n".join(
            (
                f"{runner_label} dry run only.",
                "No solver outputs were materialized.",
                f"Summary JSON: {summary_path}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return marker_path


def clear_dry_run_marker(output_root: str | Path) -> None:
    try:
        dry_run_marker_path(output_root).unlink()
    except FileNotFoundError:
        return


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
    parser.add_argument(
        "--allow-init-only-stage2-seed",
        action="store_true",
        help=(
            "Allow reusing a Stage 2 artifact whose sibling results.json reports "
            "init_only=true."
        ),
    )
    parser.add_argument("--equilibria-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--single-stage-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--backend", choices=["cpu", "jax"], default="jax")
    parser.add_argument(
        "--optimizer-backend",
        choices=["scipy", "ondevice", "scipy-jax"],
        default=None,
    )
    parser.add_argument(
        "--boozer-optimizer-backend",
        choices=["scipy", "ondevice"],
        default=None,
    )
    parser.add_argument(
        "--boozer-least-squares-algorithm",
        choices=["quasi-newton", "lm"],
        default=None,
    )
    parser.add_argument("--minimal-artifacts", action="store_true")
    parser.add_argument("--benchmark-mode", action="store_true")
    parser.add_argument("--nphi", type=int, default=91)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--iota-target", type=float, default=0.20)
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument("--cc-dist", type=float, default=COIL_COIL_MIN_DIST_M)
    parser.add_argument("--cs-dist", type=float, default=COIL_PLASMA_MIN_DIST_M)
    parser.add_argument("--ss-dist", type=float, default=PLASMA_VESSEL_MIN_DIST_M)
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=MAX_CURVATURE_INV_M,
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=COIL_LENGTH_TARGET_M,
    )
    parser.add_argument(
        "--banana-current-max-A",
        type=float,
        default=BANANA_CURRENT_HARD_LIMIT_A,
    )
    parser.add_argument("--alm-max-outer-iters", type=int, default=20)
    parser.add_argument("--alm-max-subproblem-continuations", type=int, default=20)
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
) -> tuple[Path, Path, dict[str, object]]:
    if stage2_bs_path is None:
        stage2_bs_path = resolved_path(args.stage2_bs_path)
    if not stage2_bs_path.exists():
        raise FileNotFoundError(f"Stage 2 biot_savart_opt.json does not exist: {stage2_bs_path}")
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
        owner_label="run_single_stage_thresholded_physics_alm.py",
        allow_init_only=bool(args.allow_init_only_stage2_seed),
    )
    return stage2_bs_path, stage2_results_path, stage2_results


def maybe_load_validated_stage2_seed_metadata(
    args: argparse.Namespace,
) -> tuple[Path, Path | None, dict[str, object] | None]:
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
        "--backend",
        args.backend,
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
        "--ss-dist",
        str(args.ss_dist),
        "--curvature-threshold",
        str(args.curvature_threshold),
        "--length-target",
        str(args.length_target),
        "--banana-current-max-A",
        str(args.banana_current_max_A),
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
    if args.optimizer_backend is not None:
        command.extend(["--optimizer-backend", args.optimizer_backend])
    if args.boozer_optimizer_backend is not None:
        command.extend(["--boozer-optimizer-backend", args.boozer_optimizer_backend])
    if args.boozer_least_squares_algorithm is not None:
        command.extend(
            [
                "--boozer-least-squares-algorithm",
                args.boozer_least_squares_algorithm,
            ]
        )
    if args.minimal_artifacts:
        command.append("--minimal-artifacts")
    if args.benchmark_mode:
        command.append("--benchmark-mode")
    return command


def build_summary(
    args: argparse.Namespace,
    command: list[str],
    *,
    stage2_bs_path: Path | None = None,
    stage2_results_path: Path | None = None,
    stage2_results: dict[str, object] | None = None,
    results_path: Path | None = None,
    results: dict[str, object] | None = None,
) -> dict[str, object]:
    if stage2_bs_path is None:
        stage2_bs_path = resolved_path(args.stage2_bs_path)
    output_root = resolved_path(args.output_root)
    summary: dict[str, object] = {
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "stage2_bs_path": str(stage2_bs_path),
        "output_root": str(output_root),
        "backend": args.backend,
        "optimizer_backend": args.optimizer_backend,
        "command": command,
        "dry_run": bool(args.dry_run),
        "output_contract": (
            "dry_run_summary_only"
            if args.dry_run
            else "materialized_single_stage_results"
        ),
        "contains_solver_outputs": bool(results_path is not None and results is not None),
        "dry_run_marker_path": str(dry_run_marker_path(output_root)),
    }
    if stage2_results_path is not None:
        summary["stage2_results_path"] = str(stage2_results_path)
    if stage2_results is not None:
        summary["stage2_artifact_plasma_surf_filename"] = stage2_results.get(
            "PLASMA_SURF_FILENAME"
        )
        summary["stage2_artifact_init_only"] = stage2_results.get("init_only")
    if results_path is None or results is None:
        return summary

    summary["backend"] = results.get("backend", args.backend)
    summary["optimizer_backend"] = results.get(
        "optimizer_backend",
        args.optimizer_backend,
    )
    alm_converged = first_present_value(
        results,
        "ALM_CONVERGED",
        "ALM_CONVERGED_TO_TOLERANCES",
    )
    alm_final_multipliers = first_present_value(
        results,
        "ALM_FINAL_MULTIPLIERS",
        "ALM_MULTIPLIERS",
    )
    alm_final_constraint_values = first_present_value(
        results,
        "ALM_FINAL_CONSTRAINT_VALUES",
        "ALM_CONSTRAINT_VALUES",
    )

    summary.update(
        {
            "results_path": str(results_path),
            "termination_message": results.get("TERMINATION_MESSAGE"),
            "optimizer_success": results.get("OPTIMIZER_SUCCESS"),
            "constraint_method": results.get("CONSTRAINT_METHOD"),
            "alm_formulation": results.get("ALM_FORMULATION"),
            "alm_outer_iterations": results.get("ALM_OUTER_ITERATIONS"),
            "alm_final_penalty": results.get("ALM_FINAL_PENALTY"),
            "alm_converged": alm_converged,
            "alm_termination_reason": results.get("ALM_TERMINATION_REASON"),
            "alm_final_multipliers": alm_final_multipliers,
            "alm_final_constraint_values": alm_final_constraint_values,
            "alm_final_max_feasibility_violation": results.get(
                "ALM_FINAL_MAX_FEASIBILITY_VIOLATION"
            ),
            "alm_final_stationarity_norm": results.get("ALM_FINAL_STATIONARITY_NORM"),
            "curve_curve_min_dist": results.get("CURVE_CURVE_MIN_DIST"),
            "curve_surface_min_dist": results.get("CURVE_SURFACE_MIN_DIST"),
            "surface_vessel_min_dist": results.get("SURFACE_VESSEL_MIN_DIST"),
            "max_curvature": results.get("MAX_CURVATURE"),
            "final_non_qs": results.get("FINAL_NON_QS"),
            "final_boozer_residual": results.get("FINAL_BOOZER_RESIDUAL"),
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
        write_dry_run_marker(
            output_root,
            summary_path=summary_path,
            runner_label="run_single_stage_thresholded_physics_alm.py",
        )
    else:
        clear_dry_run_marker(output_root)
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
