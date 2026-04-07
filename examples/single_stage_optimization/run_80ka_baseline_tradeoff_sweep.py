from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_helpers import (  # noqa: E402
    build_weight_cases,
    select_non_dominated_records,
)
from workflow_runner_common import (  # noqa: E402
    SINGLE_STAGE_SCRIPT_PATH,
    Stage2ArtifactConfig,
    build_stage2_command,
    discover_single_results_path,
    ensure_stage2_artifact,
    load_json,
    parse_csv,
    run_command,
    snapshot_single_results_paths,
)

DEFAULT_PLASMA_SURF_FILENAME = "wout_nfp22ginsburg_000_014417_iota15.nc"
DEFAULT_SWEEP_OUTPUT_ROOT = SCRIPT_DIR / "outputs_80ka_baseline_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a weighted tradeoff sweep for the coil-only, zero-plasma-current "
            "80 kA per TF coil baseline."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
    )
    parser.add_argument("--equilibria-dir", default=None)
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_SWEEP_OUTPUT_ROOT),
        help="Root directory for per-case single-stage outputs and sweep summaries.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary path. Defaults to <output-root>/sweep_summary.json.",
    )
    parser.add_argument(
        "--stage2-output-root",
        default=str(SCRIPT_DIR / "STAGE_2"),
        help="Parent directory where Stage 2 writes outputs-[plasma]/...",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=None,
        help="Explicit Stage 2 biot_savart_opt.json path. Overrides derived seed settings.",
    )
    parser.add_argument(
        "--stage2-init-only",
        action="store_true",
        help="Build only the Stage 2 initialized artifact when a seed must be generated.",
    )
    parser.add_argument("--stage2-constraint-method", choices=["penalty", "alm"], default="penalty")
    parser.add_argument("--stage2-basin-hops", type=int, default=0)
    parser.add_argument("--stage2-basin-stepsize", type=float, default=0.01)
    parser.add_argument("--stage2-basin-seed", type=int, default=-1)
    parser.add_argument("--stage2-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--tf-current-A", type=float, default=8.0e4)
    parser.add_argument("--major-radius", type=float, default=0.915)
    parser.add_argument("--toroidal-flux", type=float, default=0.24)
    parser.add_argument("--stage2-length-weight", type=float, default=0.0005)
    parser.add_argument("--stage2-cc-weight", type=float, default=100.0)
    parser.add_argument("--stage2-cc-threshold", type=float, default=0.05)
    parser.add_argument("--stage2-curvature-weight", type=float, default=0.0001)
    parser.add_argument("--stage2-curvature-threshold", type=float, default=40.0)
    parser.add_argument("--banana-surf-radius", type=float, default=0.22)
    parser.add_argument("--stage2-order", type=int, default=2)
    parser.add_argument("--single-stage-constraint-method", choices=["penalty", "alm"], default="penalty")
    parser.add_argument("--single-stage-maxiter", type=int, default=300)
    parser.add_argument("--single-stage-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--single-stage-init-only", action="store_true")
    parser.add_argument("--plasma-current-A", type=float, default=0.0)
    parser.add_argument("--res-weight", type=float, default=1000.0)
    parser.add_argument("--iotas-weight", type=float, default=100.0)
    parser.add_argument("--cc-weight", type=float, default=100.0)
    parser.add_argument("--curvature-weight", type=float, default=0.1)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--cs-weight", type=float, default=1.0)
    parser.add_argument("--surf-dist-weight", type=float, default=1000.0)
    parser.add_argument(
        "--scan-weights",
        default="res_weight,iotas_weight,cc_weight,curvature_weight,length_weight",
        help="Comma-separated SingleStageWeightCase field names to scale.",
    )
    parser.add_argument(
        "--weight-multipliers",
        default="0.5,1.0,2.0",
        help="Comma-separated multipliers applied one-at-a-time to the selected weights.",
    )
    return parser.parse_args()


def _timeout_or_none(timeout_seconds: float) -> float | None:
    return None if timeout_seconds <= 0.0 else float(timeout_seconds)


def make_stage2_config(args: argparse.Namespace) -> Stage2ArtifactConfig:
    basin_seed = None if args.stage2_basin_seed < 0 else args.stage2_basin_seed
    return Stage2ArtifactConfig(
        plasma_surf_filename=args.plasma_surf_filename,
        output_root=Path(args.stage2_output_root),
        equilibria_dir=args.equilibria_dir,
        tf_current_A=args.tf_current_A,
        major_radius=args.major_radius,
        toroidal_flux=args.toroidal_flux,
        length_weight=args.stage2_length_weight,
        cc_weight=args.stage2_cc_weight,
        cc_threshold=args.stage2_cc_threshold,
        curvature_weight=args.stage2_curvature_weight,
        curvature_threshold=args.stage2_curvature_threshold,
        banana_surf_radius=args.banana_surf_radius,
        order=args.stage2_order,
        constraint_method=args.stage2_constraint_method,
        alm_max_outer_iters=10,
        alm_penalty_init=1.0,
        alm_penalty_scale=10.0,
        basin_hops=args.stage2_basin_hops,
        basin_stepsize=args.stage2_basin_stepsize,
        basin_seed=basin_seed,
        init_only=args.stage2_init_only,
    )


def build_single_stage_command(
    args: argparse.Namespace,
    *,
    case,
    stage2_bs_path: Path,
    case_output_root: Path,
) -> list[str]:
    command = [
        args.python_executable,
        str(SINGLE_STAGE_SCRIPT_PATH),
        "--plasma-surf-filename",
        args.plasma_surf_filename,
        "--stage2-bs-path",
        str(stage2_bs_path),
        "--output-root",
        str(case_output_root),
        "--constraint-method",
        args.single_stage_constraint_method,
        "--maxiter",
        str(args.single_stage_maxiter),
        "--plasma-current-A",
        str(args.plasma_current_A),
        "--res-weight",
        str(case.res_weight),
        "--iotas-weight",
        str(case.iotas_weight),
        "--cc-weight",
        str(case.cc_weight),
        "--curvature-weight",
        str(case.curvature_weight),
        "--length-weight",
        str(case.length_weight),
        "--cs-weight",
        str(case.cs_weight),
        "--surf-dist-weight",
        str(case.surf_dist_weight),
    ]
    if args.equilibria_dir is not None:
        command.extend(["--equilibria-dir", args.equilibria_dir])
    if args.single_stage_init_only:
        command.append("--init-only")
    return command


def build_case_record(case_name: str, case_output_root: Path, results: dict) -> dict:
    record = {
        "CASE_NAME": case_name,
        "CASE_OUTPUT_ROOT": str(case_output_root),
        **results,
    }
    if "CURVE_CURVE_MIN_DIST" in results:
        record["NEG_CURVE_CURVE_MIN_DIST"] = -float(results["CURVE_CURVE_MIN_DIST"])
    return record


def build_summary(stage2_bs_path: Path, stage2_config: Stage2ArtifactConfig, records: list[dict]) -> dict:
    nondominated_metrics = [
        "FIELD_ERROR",
        "IOTA_ERROR_ABS",
        "COIL_LENGTH",
        "MAX_CURVATURE",
        "NEG_CURVE_CURVE_MIN_DIST",
    ]
    non_dominated = select_non_dominated_records(records, nondominated_metrics)
    stage2_config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(stage2_config).items()
    }
    return {
        "stage2_bs_path": str(stage2_bs_path),
        "stage2_config": stage2_config_payload,
        "records": records,
        "nondominated_metrics": nondominated_metrics,
        "non_dominated_case_names": [record["CASE_NAME"] for record in non_dominated],
        "non_dominated_records": non_dominated,
    }


def run_case(
    args: argparse.Namespace,
    *,
    case,
    stage2_bs_path: Path,
    sweep_output_root: Path,
) -> dict:
    case_output_root = sweep_output_root / case.name
    case_output_root.mkdir(parents=True, exist_ok=True)
    previous_snapshot = snapshot_single_results_paths(case_output_root)
    command = build_single_stage_command(
        args,
        case=case,
        stage2_bs_path=stage2_bs_path,
        case_output_root=case_output_root,
    )
    if args.dry_run:
        return {
            "CASE_NAME": case.name,
            "CASE_OUTPUT_ROOT": str(case_output_root),
            "COMMAND": command,
        }
    run_command(
        command,
        timeout_seconds=_timeout_or_none(args.single_stage_timeout_seconds),
    )
    results_path = discover_single_results_path(
        case_output_root,
        previous_snapshot=previous_snapshot,
    )
    return build_case_record(case.name, case_output_root, load_json(results_path))


def build_sweep_cases(args: argparse.Namespace):
    return build_weight_cases(
        {
            "res_weight": args.res_weight,
            "iotas_weight": args.iotas_weight,
            "cc_weight": args.cc_weight,
            "curvature_weight": args.curvature_weight,
            "length_weight": args.length_weight,
            "cs_weight": args.cs_weight,
            "surf_dist_weight": args.surf_dist_weight,
        },
        parse_csv(args.scan_weights, str),
        parse_csv(args.weight_multipliers, float),
    )


def main() -> int:
    args = parse_args()
    stage2_config = make_stage2_config(args)
    stage2_bs_path = (
        Path(args.stage2_bs_path)
        if args.stage2_bs_path is not None
        else ensure_stage2_artifact(
            stage2_config,
            python_executable=args.python_executable,
            timeout_seconds=_timeout_or_none(args.stage2_timeout_seconds),
            dry_run=args.dry_run,
        )
    )
    sweep_output_root = Path(args.output_root)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json is not None
        else sweep_output_root / "sweep_summary.json"
    )
    cases = build_sweep_cases(args)
    records = [
        run_case(
            args,
            case=case,
            stage2_bs_path=stage2_bs_path,
            sweep_output_root=sweep_output_root,
        )
        for case in cases
    ]
    summary = build_summary(stage2_bs_path, stage2_config, records) if not args.dry_run else {
        "stage2_bs_path": str(stage2_bs_path),
        "stage2_command": build_stage2_command(
            stage2_config,
            python_executable=args.python_executable,
        ) if args.stage2_bs_path is None else None,
        "records": records,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(f"Wrote sweep summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
