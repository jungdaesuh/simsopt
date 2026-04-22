from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_single_stage_goal_mode_comparison as goal_mode_runner  # noqa: E402
from workflow_runner_common import (  # noqa: E402
    clear_dry_run_marker,
    parse_csv,
    resolved_optional_path,
    resolved_path,
    timeout_or_none,
    write_csv_rows,
    write_dry_run_marker,
    write_json,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_iota_target_sweep"
DEFAULT_SUMMARY_JSON = "single_stage_iota_target_sweep_summary.json"
DEFAULT_SUMMARY_CSV = "single_stage_iota_target_sweep_summary.csv"
CSV_FIELDNAMES = (
    "case_id",
    "iota_target",
    "status",
    "result_source",
    "results_path",
    "termination_message",
    "optimizer_success",
    "final_feasibility_ok",
    "hardware_constraints_ok",
    "final_iota",
    "final_volume",
    "field_error",
    "nonqs_ratio",
    "boozer_residual",
    "coil_length",
    "max_curvature",
    "banana_current_a",
    "banana_current_mode",
    "banana_current_max_abs_a",
    "error_type",
    "error_message",
)
NUMERIC_RESULT_SUMMARY_KEYS = (
    "target_iota",
    "target_volume",
    "final_iota",
    "final_volume",
    "field_error",
    "nonqs_ratio",
    "boozer_residual",
    "coil_length",
    "max_curvature",
    "banana_current_a",
    "banana_current_max_abs_a",
    "plasma_current_a",
    "initial_iota",
)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible single-stage iota-target sweep from one explicit "
            "Stage 2 donor."
        ),
        parents=[goal_mode_runner.build_parser(add_help=False)],
        add_help=add_help,
        conflict_handler="resolve",
    )
    parser.set_defaults(output_root=str(DEFAULT_OUTPUT_ROOT), summary_json=None)
    parser.add_argument(
        "--summary-json",
        default=None,
        help=f"Optional summary path. Defaults to <output-root>/{DEFAULT_SUMMARY_JSON}.",
    )
    parser.add_argument(
        "--iota-targets",
        default=None,
        help=(
            "Comma-separated target-iota list. When omitted, the inherited "
            "--iota-target value is used as a single-case sweep."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=f"Optional CSV path. Defaults to <output-root>/{DEFAULT_SUMMARY_CSV}.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolved_iota_targets(args: argparse.Namespace) -> list[float]:
    if args.iota_targets is None:
        return [float(args.iota_target)]
    return parse_csv(args.iota_targets, float)


def _case_label(iota_target: float) -> str:
    return f"iota_{str(iota_target).replace('-', 'm').replace('.', 'p')}"


def _result_summary(results: dict) -> dict:
    summary = goal_mode_runner.result_metric_subset(results)
    summary.update(
        {
            "field_error": results.get("FIELD_ERROR"),
            "plasma_current_a": results.get("PLASMA_CURRENT_A"),
            "initial_iota": results.get("INITIAL_IOTA"),
        }
    )
    for key in NUMERIC_RESULT_SUMMARY_KEYS:
        summary[key] = _optional_float(summary.get(key))
    return summary


def _csv_row(case_payload: dict) -> dict[str, object]:
    row = {
        "case_id": case_payload["case_id"],
        "iota_target": case_payload["iota_target"],
        "status": case_payload["status"],
        "result_source": case_payload.get("result_source"),
        "results_path": case_payload.get("results_path"),
        "error_type": case_payload.get("error_type"),
        "error_message": case_payload.get("error_message"),
    }
    results_summary = case_payload.get("results_summary") or {}
    for key in CSV_FIELDNAMES:
        if key in row:
            continue
        row[key] = results_summary.get(key)
    return row


def _case_args(args: argparse.Namespace, *, iota_target: float) -> argparse.Namespace:
    payload = dict(vars(args))
    payload["iota_target"] = float(iota_target)
    return SimpleNamespace(**payload)


def run_sweep_case(
    args: argparse.Namespace,
    *,
    iota_target: float,
    stage2_bs_path: Path,
    output_root: Path,
) -> dict[str, object]:
    case_id = _case_label(iota_target)
    case_output_root = output_root / case_id
    case_args = _case_args(args, iota_target=iota_target)
    payload = {
        "case_id": case_id,
        "case_output_root": str(case_output_root),
        "iota_target": float(iota_target),
    }
    try:
        run_payload = goal_mode_runner.run_goal_mode_case(
            case_args,
            goal_mode="target",
            stage2_bs_path=stage2_bs_path,
            output_root=case_output_root,
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        payload.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        return payload

    payload["command"] = run_payload["command"]
    if args.dry_run:
        payload["status"] = "dry_run"
        return payload

    payload.update(
        {
            "status": "completed",
            "result_source": run_payload["result_source"],
            "results_path": str(run_payload["results_path"]),
            "results_summary": _result_summary(run_payload["results"]),
        }
    )
    return payload


def build_summary(
    args: argparse.Namespace,
    *,
    iota_targets: list[float],
    stage2_bs_path: Path,
    stage2_results_path: Path | None,
    stage2_results: dict | None,
    case_payloads: list[dict[str, object]],
    summary_csv_path: Path,
) -> dict[str, object]:
    summary = {
        "experiment_family": "single_stage_iota_target_sweep",
        "dry_run": bool(args.dry_run),
        "output_materialization": (
            "dry_run_summary_only" if args.dry_run else "materialized_single_stage_results"
        ),
        "goal_mode": "target",
        "init_only": bool(args.init_only),
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "stage2_bs_path": str(stage2_bs_path),
        "iota_targets": [float(value) for value in iota_targets],
        "summary_csv": str(summary_csv_path),
        "cases": case_payloads,
    }
    if stage2_results_path is not None:
        summary["stage2_results_path"] = str(stage2_results_path)
    if stage2_results is not None:
        summary["stage2_artifact_plasma_surf_filename"] = stage2_results.get(
            "PLASMA_SURF_FILENAME"
        )
        summary["stage2_artifact_init_only"] = stage2_results.get("init_only")
        summary["stage2_artifact_banana_current_a"] = stage2_results.get(
            "BANANA_CURRENT_A"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    summary_csv_path = resolved_optional_path(args.summary_csv)
    if summary_csv_path is None:
        summary_csv_path = output_root / DEFAULT_SUMMARY_CSV
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_runner.maybe_load_validated_stage2_seed_metadata(args)
        )
    else:
        stage2_bs_path, stage2_results_path, stage2_results = (
            goal_mode_runner.load_validated_stage2_seed_metadata(args)
        )
    iota_targets = _resolved_iota_targets(args)
    case_payloads = [
        run_sweep_case(
            args,
            iota_target=iota_target,
            stage2_bs_path=stage2_bs_path,
            output_root=output_root,
        )
        for iota_target in iota_targets
    ]
    summary = build_summary(
        args,
        iota_targets=iota_targets,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        case_payloads=case_payloads,
        summary_csv_path=summary_csv_path,
    )
    write_json(summary_path, summary)
    csv_rows = [_csv_row(payload) for payload in case_payloads]
    write_csv_rows(summary_csv_path, csv_rows, fieldnames=CSV_FIELDNAMES)

    if args.dry_run:
        write_dry_run_marker(
            output_root,
            summary_path=summary_path,
            runner_label="run_single_stage_iota_target_sweep.py",
        )
    else:
        clear_dry_run_marker(output_root)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
