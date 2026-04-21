from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stage2_to_single_stage as unified_runner  # noqa: E402
from banana_opt.stage2_single_stage_handoff import bootability_passes  # noqa: E402
from workflow_runner_common import (  # noqa: E402
    parse_csv,
    resolved_optional_path,
    resolved_path,
    write_csv_rows,
    write_json,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_donor_repair"
DEFAULT_SUMMARY_JSON = "single_stage_donor_repair_summary.json"
DEFAULT_SUMMARY_CSV = "single_stage_donor_repair_summary.csv"
DEFAULT_BEST_DONOR_JSON = "best_repaired_donor.json"
CSV_FIELDNAMES = (
    "case_id",
    "iota_target",
    "status",
    "handoff_bootable",
    "bootability_reason",
    "bootability_stage",
    "bootability_solved_iota",
    "bootability_abs_iota_error",
    "recovery_attempted",
    "recovery_succeeded",
    "recovery_iters",
    "recovery_termination_reason",
    "selected_seed_source",
    "selected_stage2_bs_path",
    "selected_results_path",
    "blocking_reason",
)


def _resolved_output_path(
    raw_path: str | None,
    *,
    output_root: Path,
    default_filename: str,
) -> Path:
    resolved = resolved_optional_path(raw_path)
    if resolved is not None:
        return resolved
    return output_root / default_filename


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch the Stage 2.5 bootability / bootstrap-recovery workflow over one "
            "or more target iota values and rank the resulting donors."
        ),
        parents=[unified_runner.build_parser(add_help=False)],
        add_help=add_help,
        conflict_handler="resolve",
    )
    parser.set_defaults(output_root=str(DEFAULT_OUTPUT_ROOT), summary_json=None)
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=f"Optional CSV path. Defaults to <output-root>/{DEFAULT_SUMMARY_CSV}.",
    )
    parser.add_argument(
        "--best-donor-json",
        default=None,
        help=(
            "Optional path for the best-donor manifest. Defaults to "
            f"<output-root>/{DEFAULT_BEST_DONOR_JSON}."
        ),
    )
    parser.add_argument(
        "--iota-targets",
        default=None,
        help=(
            "Comma-separated Stage 2.5 bootability targets. When omitted, the "
            "inherited --iota-target value is used as a single-case repair pass."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_donor_repair_args(args: argparse.Namespace) -> None:
    unified_runner.validate_handoff_cli_args(args)
    if args.probe_only:
        raise ValueError(
            "--probe-only is not supported by run_single_stage_donor_repair.py; "
            "use --skip-recovery to rank probe results without recovery."
        )
    if args.force_full_single_stage_after_recovery_fail:
        raise ValueError(
            "--force-full-single-stage-after-recovery-fail is not supported by "
            "run_single_stage_donor_repair.py because this entrypoint never runs "
            "the full single-stage workflow."
        )


def _resolved_iota_targets(args: argparse.Namespace) -> list[float]:
    if args.iota_targets is None:
        return [float(args.iota_target)]
    return parse_csv(args.iota_targets, float)


def _case_label(iota_target: float) -> str:
    return f"repair_iota_{str(iota_target).replace('-', 'm').replace('.', 'p')}"


def _case_args(
    args: argparse.Namespace,
    *,
    iota_target: float,
) -> argparse.Namespace:
    payload = dict(vars(args))
    payload["iota_target"] = float(iota_target)
    return SimpleNamespace(**payload)


def _case_csv_row(case_payload: dict[str, object]) -> dict[str, object]:
    bootability_status = case_payload.get("bootability_status")
    if not isinstance(bootability_status, dict):
        bootability_status = {}
    return {
        "case_id": case_payload["case_id"],
        "iota_target": case_payload["iota_target"],
        "status": case_payload["status"],
        "handoff_bootable": case_payload.get("handoff_bootable"),
        "bootability_reason": bootability_status.get("BOOTABILITY_REASON"),
        "bootability_stage": bootability_status.get("BOOTABILITY_STAGE"),
        "bootability_solved_iota": bootability_status.get("BOOTABILITY_SOLVED_IOTA"),
        "bootability_abs_iota_error": bootability_status.get("BOOTABILITY_ABS_IOTA_ERROR"),
        "recovery_attempted": case_payload.get("recovery_attempted"),
        "recovery_succeeded": case_payload.get("recovery_succeeded"),
        "recovery_iters": case_payload.get("recovery_iters"),
        "recovery_termination_reason": case_payload.get(
            "recovery_termination_reason"
        ),
        "selected_seed_source": case_payload.get("selected_seed_source"),
        "selected_stage2_bs_path": case_payload.get("selected_stage2_bs_path"),
        "selected_results_path": case_payload.get("selected_results_path"),
        "blocking_reason": case_payload.get("blocking_reason"),
    }


def _case_rank_key(case_payload: dict[str, object]) -> tuple[int, float, int, str]:
    handoff_bootable = bool(case_payload.get("handoff_bootable"))
    bootability_status = case_payload.get("bootability_status")
    if isinstance(bootability_status, dict):
        raw_abs_iota_error = bootability_status.get("BOOTABILITY_ABS_IOTA_ERROR")
    else:
        raw_abs_iota_error = None
    abs_iota_error = (
        float(raw_abs_iota_error)
        if raw_abs_iota_error is not None
        else float("inf")
    )
    raw_recovery_iters = case_payload.get("recovery_iters")
    recovery_iters = (
        int(raw_recovery_iters)
        if raw_recovery_iters is not None
        else 0 if handoff_bootable else 10**9
    )
    return (
        0 if handoff_bootable else 1,
        abs_iota_error,
        recovery_iters,
        str(case_payload["case_id"]),
    )


def build_best_donor_payload(
    case_payload: dict[str, object],
    *,
    summary_path: Path,
) -> dict[str, object]:
    return {
        "case_id": case_payload["case_id"],
        "iota_target": case_payload["iota_target"],
        "handoff_bootable": case_payload["handoff_bootable"],
        "selected_seed_source": case_payload["selected_seed_source"],
        "selected_stage2_bs_path": case_payload["selected_stage2_bs_path"],
        "selected_results_path": case_payload["selected_results_path"],
        "summary_path": str(summary_path),
        "bootability_status": case_payload.get("bootability_status"),
    }


def run_repair_case(
    args: argparse.Namespace,
    *,
    iota_target: float,
    stage2_input: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    case_id = _case_label(iota_target)
    case_output_root = output_root / case_id
    case_args = _case_args(args, iota_target=iota_target)
    case_payload: dict[str, object] = {
        "case_id": case_id,
        "case_output_root": str(case_output_root),
        "iota_target": float(iota_target),
    }
    original_stage2_bs_path = Path(stage2_input["stage2_bs_path"])
    original_stage2_results_path = Path(stage2_input["stage2_results_path"])
    if stage2_input["stage2_results"] is None:
        recovery_command = None
        if not case_args.skip_recovery:
            recovery_command = unified_runner.build_recovery_command(
                case_args,
                stage2_bs_path=original_stage2_bs_path,
                recovery_output_root=case_output_root / "recovery",
            )
        case_payload.update(
            {
                "status": "dry_run",
                "stage2_input": {
                    "stage2_bs_path": str(original_stage2_bs_path),
                    "stage2_results_path": str(original_stage2_results_path),
                },
                "recovery_command": recovery_command,
            }
        )
        return case_payload

    stage2_results = dict(stage2_input["stage2_results"])
    initial_probe = unified_runner.build_probe_status(
        case_args,
        stage2_bs_path=original_stage2_bs_path,
        stage2_results=stage2_results,
        stage=unified_runner.BOOTABILITY_STAGE_PROBE,
    )
    handoff_status = initial_probe
    recovery_payload = None
    recovery_attempted = False
    recovery_succeeded = False
    recovery_iters = None
    recovery_termination_reason = None
    selected_seed_source = unified_runner.SEED_SOURCE_DIRECT_STAGE2_DONOR
    selected_stage2_bs_path = original_stage2_bs_path
    selected_results_path = original_stage2_results_path
    blocking_reason = None

    if not bootability_passes(initial_probe):
        if case_args.skip_recovery:
            blocking_reason = "initial_probe_failed_skip_recovery"
        else:
            recovery_attempted = True
            recovery_output_root = case_output_root / "recovery"
            recovery_output_root.mkdir(parents=True, exist_ok=True)
            recovery_payload = unified_runner.run_recovery_stage(
                case_args,
                original_stage2_bs_path=original_stage2_bs_path,
                original_stage2_results_path=original_stage2_results_path,
                original_stage2_results=stage2_results,
                recovery_output_root=recovery_output_root,
            )
            recovery_termination_reason = recovery_payload.get(
                "recovery_termination_reason"
            )
            if recovery_payload["status"] == "completed":
                recovery_succeeded = bool(recovery_payload["recovery_succeeded"])
                recovery_iters = recovery_payload["recovery_iters"]
                handoff_status = recovery_payload["recovery_probe"]
                if recovery_succeeded:
                    selected_seed_source = (
                        unified_runner.SEED_SOURCE_RECOVERED_STAGE2_DONOR
                    )
                    selected_stage2_bs_path = resolved_path(
                        recovery_payload["recovered_bs_path"]
                    )
                    selected_results_path = resolved_path(
                        recovery_payload["results_path"]
                    )
                else:
                    blocking_reason = "recovery_failed"
            else:
                blocking_reason = "recovery_failed"

    handoff_bootable = bootability_passes(handoff_status)
    case_payload.update(
        {
            "status": "completed" if handoff_bootable else "failed",
            "initial_probe": initial_probe,
            "recovery": recovery_payload,
            "bootability_status": handoff_status,
            "handoff_bootable": handoff_bootable,
            "recovery_attempted": recovery_attempted,
            "recovery_succeeded": recovery_succeeded,
            "recovery_iters": recovery_iters,
            "recovery_termination_reason": recovery_termination_reason,
            "selected_seed_source": selected_seed_source,
            "selected_stage2_bs_path": str(selected_stage2_bs_path),
            "selected_results_path": str(selected_results_path),
            "blocking_reason": blocking_reason,
        }
    )
    return case_payload


def build_summary(
    args: argparse.Namespace,
    *,
    stage2_input: dict[str, object],
    case_payloads: list[dict[str, object]],
    summary_csv_path: Path,
    best_case: dict[str, object] | None,
    best_donor_path: Path,
) -> dict[str, object]:
    return {
        "experiment_family": "single_stage_donor_repair",
        "dry_run": bool(args.dry_run),
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "iota_targets": [case["iota_target"] for case in case_payloads],
        "stage2_input": {
            "source": stage2_input["source"],
            "stage2_bs_path": str(stage2_input["stage2_bs_path"]),
            "stage2_results_path": str(stage2_input["stage2_results_path"]),
            "artifact_reused": stage2_input["artifact_reused"],
            "config_source": stage2_input.get("config_source"),
            "command": stage2_input.get("command"),
        },
        "summary_csv": str(summary_csv_path),
        "best_donor_json": (
            None if best_case is None else str(best_donor_path)
        ),
        "best_case_id": None if best_case is None else best_case["case_id"],
        "best_case": best_case,
        "cases": case_payloads,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_donor_repair_args(args)
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = _resolved_output_path(
        args.summary_json,
        output_root=output_root,
        default_filename=DEFAULT_SUMMARY_JSON,
    )
    summary_csv_path = _resolved_output_path(
        args.summary_csv,
        output_root=output_root,
        default_filename=DEFAULT_SUMMARY_CSV,
    )
    best_donor_path = _resolved_output_path(
        args.best_donor_json,
        output_root=output_root,
        default_filename=DEFAULT_BEST_DONOR_JSON,
    )

    stage2_input = unified_runner.resolve_stage2_input(
        args,
        output_root=output_root / "stage2_source",
    )
    iota_targets = _resolved_iota_targets(args)
    case_payloads = [
        run_repair_case(
            args,
            iota_target=iota_target,
            stage2_input=stage2_input,
            output_root=output_root,
        )
        for iota_target in iota_targets
    ]
    ranked_cases = sorted(case_payloads, key=_case_rank_key)
    best_case = None
    if ranked_cases and bool(ranked_cases[0].get("handoff_bootable")):
        best_case = ranked_cases[0]
    if best_case is not None:
        write_json(
            best_donor_path,
            build_best_donor_payload(best_case, summary_path=summary_path),
        )
    summary = build_summary(
        args,
        stage2_input=stage2_input,
        case_payloads=case_payloads,
        summary_csv_path=summary_csv_path,
        best_case=best_case,
        best_donor_path=best_donor_path,
    )
    write_json(summary_path, summary)
    write_csv_rows(
        summary_csv_path,
        [_case_csv_row(case_payload) for case_payload in case_payloads],
        fieldnames=CSV_FIELDNAMES,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
