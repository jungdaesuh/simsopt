from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stage2_alm as stage2_alm_runner  # noqa: E402
from workflow_runner_common import (  # noqa: E402
    parse_csv,
    resolved_optional_path,
    resolved_path,
    timeout_or_none,
    write_csv_rows,
    write_json,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_stage2_iota_decision_gate"
DEFAULT_SUMMARY_JSON = "stage2_iota_decision_gate_summary.json"
DEFAULT_SUMMARY_CSV = "stage2_iota_decision_gate_summary.csv"
DEFAULT_BENCHMARK_MODES = "report,soft,alm"
ALLOWED_MODES = ("off", "report", "soft", "alm")
CSV_FIELDNAMES = (
    "mode",
    "status",
    "run_wallclock_seconds",
    "runtime_multiplier_vs_baseline",
    "optimizer_success",
    "hardware_constraints_ok",
    "boozer_bootable",
    "iota_feasible",
    "bootability_reason",
    "stage2_iota_initial",
    "stage2_iota_final",
    "stage2_iota_abs_error",
    "field_error",
    "coil_length",
    "max_curvature",
    "stage2_iota_probe_seconds",
    "stage2_iota_bootstrap_seconds",
    "stage2_iota_runtime_seconds",
    "stage2_iota_runtime_calls",
    "stage2_results_path",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one canonical Stage 2 configuration across multiple iota modes and "
            "emit a decision-gate summary for whether Stage 2-native iota work is "
            "worth carrying further."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plasma-surf-filename",
        required=True,
        help="VMEC wout filename used as the canonical Stage 2 benchmark case.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--profile",
        choices=sorted(stage2_alm_runner.DEFAULT_STAGE2_PROFILES),
        help="Built-in Stage 2 ALM profile.",
    )
    source_group.add_argument(
        "--stage2-spec-json",
        help="Path to a full Stage 2 ALM spec JSON file.",
    )
    parser.add_argument("--equilibria-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--summary-json",
        default=None,
        help=f"Optional summary path. Defaults to <output-root>/{DEFAULT_SUMMARY_JSON}.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=f"Optional CSV path. Defaults to <output-root>/{DEFAULT_SUMMARY_CSV}.",
    )
    parser.add_argument("--stage2-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--cc-threshold", type=float, default=None)
    parser.add_argument("--curvature-threshold", type=float, default=None)
    parser.add_argument("--order", type=int, default=None)
    parser.add_argument("--tf-current-A", type=float, default=None)
    parser.add_argument("--toroidal-flux", type=float, default=None)
    parser.add_argument(
        "--benchmark-modes",
        default=DEFAULT_BENCHMARK_MODES,
        help=(
            "Comma-separated Stage 2 iota modes to compare. Allowed values: "
            "off, report, soft, alm."
        ),
    )
    parser.add_argument(
        "--baseline-mode",
        choices=ALLOWED_MODES,
        default="report",
        help="Mode used as the runtime and iota-error baseline for comparisons.",
    )
    parser.add_argument(
        "--stage2-iota-target",
        type=float,
        default=None,
        help="Shared iota target used for report/soft/alm benchmark modes.",
    )
    parser.add_argument("--stage2-iota-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--stage2-iota-weight", type=float, default=1.0)
    parser.add_argument("--stage2-iota-vol-target", type=float, default=0.10)
    parser.add_argument("--stage2-iota-constraint-weight", type=float, default=1.0)
    parser.add_argument("--stage2-iota-num-tf-coils", type=int, default=20)
    parser.add_argument("--stage2-iota-nphi", type=int, default=91)
    parser.add_argument("--stage2-iota-ntheta", type=int, default=32)
    parser.add_argument("--stage2-iota-mpol", type=int, default=8)
    parser.add_argument("--stage2-iota-ntor", type=int, default=6)
    parser.add_argument(
        "--minimum-iota-error-improvement",
        type=float,
        default=1.0e-3,
        help=(
            "Minimum absolute iota-error improvement required before soft/hard "
            "modes are treated as materially better than the baseline."
        ),
    )
    parser.add_argument(
        "--max-acceptable-runtime-multiplier",
        type=float,
        default=2.0,
        help=(
            "Largest acceptable wallclock multiplier versus the baseline before the "
            "decision gate recommends stopping at the unified-runner seam."
        ),
    )
    parser.add_argument(
        "--donor-repair-summary",
        default=None,
        help=(
            "Optional summary JSON from run_single_stage_donor_repair.py. When "
            "provided, the recommendation can account for whether donor repair "
            "already solves the practical workflow problem."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> list[str]:
    benchmark_modes = parse_csv(args.benchmark_modes, str)
    modes: list[str] = []
    for raw_mode in benchmark_modes:
        mode = raw_mode.strip()
        if mode not in ALLOWED_MODES:
            raise ValueError(
                f"Unsupported Stage 2 iota benchmark mode {mode!r}; expected one of "
                f"{', '.join(ALLOWED_MODES)}."
            )
        if mode not in modes:
            modes.append(mode)
    if args.baseline_mode not in modes:
        raise ValueError("--baseline-mode must be included in --benchmark-modes.")
    if any(mode != "off" for mode in modes) and args.stage2_iota_target is None:
        raise ValueError(
            "--stage2-iota-target is required when benchmarking report/soft/alm modes."
        )
    if args.minimum_iota_error_improvement < 0.0:
        raise ValueError("--minimum-iota-error-improvement must be non-negative.")
    if args.max_acceptable_runtime_multiplier <= 0.0:
        raise ValueError("--max-acceptable-runtime-multiplier must be positive.")
    return modes


def build_stage2_mode_args(
    args: argparse.Namespace,
    *,
    mode: str,
    output_root: Path,
) -> argparse.Namespace:
    return SimpleNamespace(
        python_executable=args.python_executable,
        dry_run=args.dry_run,
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        profile=args.profile,
        stage2_spec_json=args.stage2_spec_json,
        equilibria_dir=args.equilibria_dir,
        output_root=str(output_root),
        summary_json=None,
        stage2_timeout_seconds=args.stage2_timeout_seconds,
        cc_threshold=args.cc_threshold,
        curvature_threshold=args.curvature_threshold,
        order=args.order,
        tf_current_A=args.tf_current_A,
        toroidal_flux=args.toroidal_flux,
        stage2_iota_mode=mode,
        stage2_iota_target=None if mode == "off" else args.stage2_iota_target,
        stage2_iota_tolerance=args.stage2_iota_tolerance,
        stage2_iota_weight=args.stage2_iota_weight,
        stage2_iota_vol_target=args.stage2_iota_vol_target,
        stage2_iota_constraint_weight=args.stage2_iota_constraint_weight,
        stage2_iota_num_tf_coils=args.stage2_iota_num_tf_coils,
        stage2_iota_nphi=args.stage2_iota_nphi,
        stage2_iota_ntheta=args.stage2_iota_ntheta,
        stage2_iota_mpol=args.stage2_iota_mpol,
        stage2_iota_ntor=args.stage2_iota_ntor,
    )


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _recommendation_payload(
    *,
    recommendation: str,
    reason: str,
    max_acceptable_runtime_multiplier: float,
    minimum_iota_error_improvement: float,
    soft_improvement: float | None,
    soft_multiplier: float | None,
    alm_multiplier: float | None,
    donor_repair_signal: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "recommendation": recommendation,
        "reason": reason,
        "runtime_multiplier_threshold": max_acceptable_runtime_multiplier,
        "minimum_iota_error_improvement": minimum_iota_error_improvement,
        "soft_iota_error_improvement": soft_improvement,
        "soft_runtime_multiplier": soft_multiplier,
        "alm_runtime_multiplier": alm_multiplier,
        "donor_repair_signal": donor_repair_signal,
    }


def _mode_metrics(stage2_results: dict[str, object]) -> dict[str, object]:
    target = _float_or_none(stage2_results.get("STAGE2_IOTA_TARGET"))
    final_iota = _float_or_none(stage2_results.get("STAGE2_IOTA_FINAL"))
    abs_iota_error = _float_or_none(stage2_results.get("BOOTABILITY_ABS_IOTA_ERROR"))
    if abs_iota_error is None and target is not None and final_iota is not None:
        abs_iota_error = abs(final_iota - target)
    return {
        "termination_message": stage2_results.get("TERMINATION_MESSAGE"),
        "optimizer_success": stage2_results.get("OPTIMIZER_SUCCESS"),
        "hardware_constraints_ok": stage2_results.get("HARDWARE_CONSTRAINTS_OK"),
        "boozer_bootable": stage2_results.get("BOOZER_BOOTABLE"),
        "iota_feasible": stage2_results.get("IOTA_FEASIBLE"),
        "bootability_reason": stage2_results.get("BOOTABILITY_REASON"),
        "stage2_iota_target": target,
        "stage2_iota_initial": _float_or_none(stage2_results.get("STAGE2_IOTA_INITIAL")),
        "stage2_iota_final": final_iota,
        "stage2_iota_abs_error": abs_iota_error,
        "field_error": _float_or_none(stage2_results.get("FIELD_ERROR")),
        "coil_length": _float_or_none(stage2_results.get("COIL_LENGTH")),
        "max_curvature": _float_or_none(stage2_results.get("MAX_CURVATURE")),
        "stage2_iota_probe_seconds": _float_or_none(
            stage2_results.get("STAGE2_IOTA_PROBE_SECONDS")
        ),
        "stage2_iota_bootstrap_seconds": _float_or_none(
            stage2_results.get("STAGE2_IOTA_BOOTSTRAP_SECONDS")
        ),
        "stage2_iota_runtime_seconds": _float_or_none(
            stage2_results.get("STAGE2_IOTA_RUNTIME_SECONDS")
        ),
        "stage2_iota_runtime_calls": stage2_results.get("STAGE2_IOTA_RUNTIME_CALLS"),
    }


def run_mode_case(
    args: argparse.Namespace,
    *,
    mode: str,
    output_root: Path,
) -> dict[str, object]:
    mode_args = build_stage2_mode_args(args, mode=mode, output_root=output_root)
    resolved_spec, resolved_spec_source = stage2_alm_runner.resolve_stage2_spec_payload(
        mode_args
    )
    config = stage2_alm_runner.build_stage2_alm_config(
        mode_args,
        resolved_spec=resolved_spec,
    )
    artifact_path = stage2_alm_runner.resolve_stage2_artifact_path(config)
    artifact_reused = artifact_path.exists()
    command = stage2_alm_runner.build_stage2_command(
        config,
        python_executable=args.python_executable,
    )
    payload: dict[str, object] = {
        "mode": mode,
        "status": "dry_run" if args.dry_run else "completed",
        "artifact_path": str(artifact_path),
        "artifact_reused": artifact_reused,
        "command": command,
        "resolved_spec_source": resolved_spec_source,
        "resolved_stage2_config": stage2_alm_runner._jsonable_stage2_config(config),
    }
    if args.dry_run:
        return payload

    run_start = time.perf_counter()
    stage2_alm_runner.ensure_stage2_artifact(
        config,
        python_executable=args.python_executable,
        timeout_seconds=timeout_or_none(args.stage2_timeout_seconds),
        dry_run=False,
    )
    payload["run_wallclock_seconds"] = time.perf_counter() - run_start
    stage2_results_path, stage2_results = stage2_alm_runner.load_validated_stage2_artifact(
        config
    )
    payload["stage2_results_path"] = str(stage2_results_path)
    payload.update(_mode_metrics(stage2_results))
    return payload


def _load_donor_repair_signal(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as infile:
        summary = json.load(infile)
    if not isinstance(summary, dict):
        raise ValueError(
            f"Donor-repair summary must be a JSON object: {path}"
        )
    best_case = summary.get("best_case")
    if not isinstance(best_case, dict):
        return {
            "summary_path": str(path),
            "best_case_present": False,
            "best_case_bootable": False,
        }
    return {
        "summary_path": str(path),
        "best_case_present": True,
        "best_case_bootable": bool(best_case.get("handoff_bootable")),
        "best_case_id": best_case.get("case_id"),
        "selected_seed_source": best_case.get("selected_seed_source"),
    }


def _runtime_multiplier(
    payload: dict[str, object],
    *,
    baseline_payload: dict[str, object] | None,
) -> float | None:
    if baseline_payload is None:
        return None
    baseline_seconds = baseline_payload.get("run_wallclock_seconds")
    current_seconds = payload.get("run_wallclock_seconds")
    if baseline_seconds is None or current_seconds is None:
        return None
    baseline_value = float(baseline_seconds)
    if baseline_value <= 0.0:
        return None
    return float(current_seconds) / baseline_value


def _decision_summary(
    payloads_by_mode: dict[str, dict[str, object]],
    *,
    baseline_mode: str,
    minimum_iota_error_improvement: float,
    max_acceptable_runtime_multiplier: float,
    donor_repair_signal: dict[str, object] | None,
) -> dict[str, object]:
    baseline_payload = payloads_by_mode.get(baseline_mode)
    soft_payload = payloads_by_mode.get("soft")
    alm_payload = payloads_by_mode.get("alm")
    baseline_error = (
        None
        if baseline_payload is None
        else _float_or_none(baseline_payload.get("stage2_iota_abs_error"))
    )
    soft_error = (
        None
        if soft_payload is None
        else _float_or_none(soft_payload.get("stage2_iota_abs_error"))
    )
    soft_improvement = None
    if baseline_error is not None and soft_error is not None:
        soft_improvement = baseline_error - soft_error
    soft_multiplier = None
    if soft_payload is not None:
        soft_multiplier = _runtime_multiplier(
            soft_payload,
            baseline_payload=baseline_payload,
        )
    alm_multiplier = None
    if alm_payload is not None:
        alm_multiplier = _runtime_multiplier(
            alm_payload,
            baseline_payload=baseline_payload,
        )

    if baseline_payload is None or baseline_payload.get("status") == "dry_run":
        return _recommendation_payload(
            recommendation="insufficient_runtime_data",
            reason="dry_run_or_missing_baseline",
            max_acceptable_runtime_multiplier=max_acceptable_runtime_multiplier,
            minimum_iota_error_improvement=minimum_iota_error_improvement,
            soft_improvement=soft_improvement,
            soft_multiplier=soft_multiplier,
            alm_multiplier=alm_multiplier,
            donor_repair_signal=donor_repair_signal,
        )

    donor_repair_bootable = bool(
        donor_repair_signal is not None
        and donor_repair_signal.get("best_case_bootable")
    )
    if (
        donor_repair_bootable
        and (
            soft_improvement is None
            or soft_improvement < minimum_iota_error_improvement
            or soft_multiplier is None
            or soft_multiplier > max_acceptable_runtime_multiplier
        )
    ):
        recommendation = "prefer_unified_runner_donor_repair"
        reason = "donor_repair_already_solves_bootability_with_less_risk"
    elif (
        alm_payload is not None
        and bool(alm_payload.get("hardware_constraints_ok"))
        and bool(alm_payload.get("boozer_bootable"))
        and bool(alm_payload.get("iota_feasible"))
        and alm_multiplier is not None
        and alm_multiplier <= max_acceptable_runtime_multiplier
    ):
        recommendation = "proceed_to_hard_stage2_alm_iota"
        reason = "alm_mode_hits_hardware_and_iota_with_acceptable_runtime"
    elif (
        soft_payload is not None
        and bool(soft_payload.get("hardware_constraints_ok"))
        and soft_improvement is not None
        and soft_improvement >= minimum_iota_error_improvement
        and soft_multiplier is not None
        and soft_multiplier <= max_acceptable_runtime_multiplier
    ):
        recommendation = "keep_soft_stage2_iota_for_more_measurement"
        reason = "soft_mode_improves_iota_without_losing_hardware_or_runtime_budget"
    else:
        recommendation = "stop_at_unified_runner_or_reporting_probe"
        reason = "stage2_native_iota_cost_or_signal_is_not_yet_compelling"
    return _recommendation_payload(
        recommendation=recommendation,
        reason=reason,
        max_acceptable_runtime_multiplier=max_acceptable_runtime_multiplier,
        minimum_iota_error_improvement=minimum_iota_error_improvement,
        soft_improvement=soft_improvement,
        soft_multiplier=soft_multiplier,
        alm_multiplier=alm_multiplier,
        donor_repair_signal=donor_repair_signal,
    )


def _csv_row(
    payload: dict[str, object],
    *,
    runtime_multiplier_vs_baseline: float | None,
) -> dict[str, object]:
    row = {fieldname: None for fieldname in CSV_FIELDNAMES}
    row.update(payload)
    row["runtime_multiplier_vs_baseline"] = runtime_multiplier_vs_baseline
    return row


def build_summary(
    args: argparse.Namespace,
    *,
    modes: list[str],
    mode_payloads: list[dict[str, object]],
    summary_csv_path: Path,
    recommendation: dict[str, object],
    donor_repair_signal: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "experiment_family": "stage2_iota_decision_gate",
        "dry_run": bool(args.dry_run),
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "benchmark_modes": modes,
        "baseline_mode": args.baseline_mode,
        "summary_csv": str(summary_csv_path),
        "mode_results": mode_payloads,
        "recommendation": recommendation,
        "donor_repair_signal": donor_repair_signal,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = validate_args(args)
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
    donor_repair_signal = _load_donor_repair_signal(
        resolved_optional_path(args.donor_repair_summary)
    )

    mode_payloads = [
        run_mode_case(
            args,
            mode=mode,
            output_root=output_root / mode,
        )
        for mode in modes
    ]
    payloads_by_mode = {payload["mode"]: payload for payload in mode_payloads}
    baseline_payload = payloads_by_mode.get(args.baseline_mode)
    recommendation = _decision_summary(
        payloads_by_mode,
        baseline_mode=args.baseline_mode,
        minimum_iota_error_improvement=args.minimum_iota_error_improvement,
        max_acceptable_runtime_multiplier=args.max_acceptable_runtime_multiplier,
        donor_repair_signal=donor_repair_signal,
    )
    write_csv_rows(
        summary_csv_path,
        [
            _csv_row(
                payload,
                runtime_multiplier_vs_baseline=_runtime_multiplier(
                    payload,
                    baseline_payload=baseline_payload,
                ),
            )
            for payload in mode_payloads
        ],
        fieldnames=CSV_FIELDNAMES,
    )
    summary = build_summary(
        args,
        modes=modes,
        mode_payloads=mode_payloads,
        summary_csv_path=summary_csv_path,
        recommendation=recommendation,
        donor_repair_signal=donor_repair_signal,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
