from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from simsopt._core.optimizable import load

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_single_stage_goal_mode_comparison as goal_mode_runner  # noqa: E402
from banana_opt.stage2_single_stage_handoff import (  # noqa: E402
    build_equilibrium_path,
    partition_loaded_stage2_coils,
)
from banana_opt.artifact_contracts import (  # noqa: E402
    STAGE2_BS_SHA256_KEY,
    compute_stage2_bs_sha256,
)
from banana_opt.single_stage_geometry import build_surface_configs  # noqa: E402
from workflow_runner_common import (  # noqa: E402
    clear_dry_run_marker,
    load_json,
    parse_csv,
    resolved_optional_path,
    resolved_path,
    run_poincare_artifact,
    timeout_or_none,
    write_csv_rows,
    write_dry_run_marker,
    write_json,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_banana_current_scan"
DEFAULT_SUMMARY_JSON = "banana_current_scan_summary.json"
DEFAULT_SUMMARY_CSV = "banana_current_scan_summary.csv"
DEFAULT_BANANA_CURRENT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
CSV_FIELDNAMES = (
    "case_id",
    "banana_current_a",
    "classification",
    "single_stage_status",
    "poincare_status",
    "results_path",
    "poincare_metrics_path",
    "termination_message",
    "optimizer_success",
    "final_feasibility_ok",
    "hardware_constraints_ok",
    "final_iota",
    "field_error",
    "nonqs_ratio",
    "boozer_residual",
    "coil_length",
    "max_curvature",
    "error_type",
    "error_message",
)


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan banana-coil current from zero to the optimized donor value, "
            "attempting single-stage init-only Boozer startup plus Poincare fallback "
            "artifacts at each point."
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
        "--banana-currents-A",
        dest="banana_currents_a",
        default=None,
        help=(
            "Comma-separated banana-current setpoints in physical amperes. "
            "When omitted, the scan defaults to five evenly spaced amp setpoints "
            "from zero through donor current."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=f"Optional CSV path. Defaults to <output-root>/{DEFAULT_SUMMARY_CSV}.",
    )
    parser.add_argument(
        "--poincare-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional timeout for each non-interactive Poincare subprocess.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _case_label(banana_current_a: float) -> str:
    token = f"{float(banana_current_a):g}".replace("-", "m").replace(".", "p")
    return f"banana_current_{token}A"


def _result_summary(results: dict) -> dict:
    summary = goal_mode_runner.result_metric_subset(results)
    summary.update(
        {
            "field_error": results.get("FIELD_ERROR"),
            "banana_current_a": results.get("BANANA_CURRENT_A"),
            "plasma_current_a": results.get("PLASMA_CURRENT_A"),
        }
    )
    return summary


def _csv_row(case_payload: dict) -> dict[str, object]:
    row = {
        "case_id": case_payload["case_id"],
        "banana_current_a": case_payload["banana_current_a"],
        "classification": case_payload["classification"],
        "single_stage_status": case_payload.get("single_stage_status"),
        "poincare_status": case_payload.get("poincare_status"),
        "results_path": case_payload.get("results_path"),
        "poincare_metrics_path": case_payload.get("poincare_metrics_path"),
        "error_type": case_payload.get("error_type"),
        "error_message": case_payload.get("error_message"),
    }
    results_summary = case_payload.get("results_summary") or {}
    for key in CSV_FIELDNAMES:
        if key in row:
            continue
        row[key] = results_summary.get(key)
    return row


def _load_poincare_metrics(output_dir: Path) -> tuple[Path, dict]:
    metric_matches = sorted(output_dir.glob("PoincareMetrics_*.json"))
    if not metric_matches:
        raise FileNotFoundError(
            f"Expected at least one PoincareMetrics_*.json under {output_dir}, found 0"
        )
    metrics_path = metric_matches[0]
    return metrics_path, load_json(metrics_path)


def _set_case_error(payload: dict[str, object], error: Exception) -> None:
    payload["error_type"] = type(error).__name__
    payload["error_message"] = str(error)


def _append_case_error_context(
    payload: dict[str, object],
    *,
    context: str,
    error: Exception,
) -> None:
    error_message = payload.get("error_message")
    if error_message is not None:
        payload["error_message"] = f"{error_message}; {context}: {error}"


def _scale_banana_current_chain(
    banana_coils,
    *,
    target_banana_current_a: float,
) -> None:
    """Scale the shared banana-coil current DOF so that the outermost coil reports
    ``target_banana_current_a``, while preserving stellsym sign flips on mirrored
    coils. Stage 2 builds banana coils with
    ``ScaledCurrent(Current(1), banana_init_current_A)`` plus ``ScaledCurrent(..., -1)``
    wrappers for the stellsym pair, so every coil ultimately points at the same
    underlying scalar ``Current`` DOF. ``ScaledCurrent`` does not expose
    ``set_value``; we have to mutate the innermost ``Current`` DOF instead.
    """
    if not banana_coils:
        raise ValueError(
            "Banana-current scan requires at least one banana coil in the partition."
        )
    innermost_current = banana_coils[0].current
    chain_scale = 1.0
    while hasattr(innermost_current, "current_to_scale"):
        chain_scale *= float(innermost_current.scale)
        innermost_current = innermost_current.current_to_scale
    if chain_scale == 0.0:
        raise ValueError(
            "Banana-current scan cannot rescale a donor whose outermost banana-coil "
            "ScaledCurrent wrapper has zero scale."
        )
    new_dof_value = float(target_banana_current_a) / chain_scale
    # The banana scan deliberately overrides the donor current even when the
    # underlying Current DOF is fixed in the saved artifact.
    innermost_current.local_full_x = [new_dof_value]


def _materialize_stage2_seed_variant(
    *,
    stage2_bs_path: Path,
    stage2_results: dict,
    variant_root: Path,
    banana_current_a: float,
    requested_num_tf_coils: int,
) -> tuple[Path, Path]:
    bs = load(str(stage2_bs_path))
    coil_partitions = partition_loaded_stage2_coils(
        bs.coils,
        stage2_results=stage2_results,
        requested_num_tf_coils=requested_num_tf_coils,
    )
    _scale_banana_current_chain(
        coil_partitions.banana_coils,
        target_banana_current_a=banana_current_a,
    )

    variant_root.mkdir(parents=True, exist_ok=True)
    variant_bs_path = variant_root / "biot_savart_opt.json"
    variant_results_path = variant_root / "results.json"
    bs.save(str(variant_bs_path))

    variant_results = dict(stage2_results)
    variant_results["BANANA_CURRENT_A"] = float(banana_current_a)
    variant_results["STAGE2_BS_PATH"] = str(stage2_bs_path)
    variant_results[STAGE2_BS_SHA256_KEY] = compute_stage2_bs_sha256(variant_bs_path)
    write_json(variant_results_path, variant_results)
    return variant_bs_path, variant_results_path


def _materialize_poincare_fallback_inputs(
    *,
    args: argparse.Namespace,
    variant_bs_path: Path,
    stage2_results: dict,
    fallback_root: Path,
) -> Path:
    recorded_equilibrium_path = stage2_results.get("PLASMA_SURF_PATH")
    recorded_candidate = (
        Path(str(recorded_equilibrium_path))
        if recorded_equilibrium_path is not None
        else None
    )
    if recorded_candidate is not None and recorded_candidate.exists():
        equilibrium_file = str(recorded_candidate.resolve())
    else:
        if args.equilibrium_path is None and args.equilibria_dir is None:
            raise ValueError(
                "Banana-current scan fallback Poincare generation requires either "
                "PLASMA_SURF_PATH in the Stage 2 artifact, --equilibrium-path, or "
                "--equilibria-dir."
            )
        equilibrium_file = build_equilibrium_path(
            Path(args.plasma_surf_filename).name,
            args.equilibria_dir,
            equilibrium_path=args.equilibrium_path,
        )
    surface_configs = build_surface_configs(
        equilibrium_file,
        args.nphi,
        args.ntheta,
        float(stage2_results["TOROIDAL_FLUX"]),
        float(stage2_results["MAJOR_RADIUS"]),
        float(args.vol_target),
        1,
        0.8,
    )
    fallback_root.mkdir(parents=True, exist_ok=True)
    bs = load(str(variant_bs_path))
    bs.save(str(fallback_root / "biot_savart_init.json"))
    surface_configs[-1]["initial_surface"].save(str(fallback_root / "surf_init.json"))
    return fallback_root


def _single_stage_case_args(
    args: argparse.Namespace,
    *,
    variant_bs_path: Path,
) -> argparse.Namespace:
    payload = dict(vars(args))
    payload["stage2_bs_path"] = str(variant_bs_path)
    payload["init_only"] = True
    return SimpleNamespace(**payload)


def _default_banana_currents_a(
    *,
    donor_banana_current_a: float,
) -> list[float]:
    return [
        fraction * donor_banana_current_a
        for fraction in DEFAULT_BANANA_CURRENT_FRACTIONS
    ]


def _resolved_banana_currents_a(
    args: argparse.Namespace,
    *,
    donor_banana_current_a: float,
) -> list[float]:
    if args.banana_currents_a not in {None, ""}:
        return parse_csv(str(args.banana_currents_a), float)
    return _default_banana_currents_a(donor_banana_current_a=donor_banana_current_a)


def run_current_case(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    stage2_results: dict,
    banana_current_a: float,
    output_root: Path,
    poincare_timeout_seconds: float | None,
) -> dict[str, object]:
    case_id = _case_label(banana_current_a)
    case_root = output_root / case_id
    stage2_variant_root = case_root / "stage2_variant"
    single_stage_root = case_root / "single_stage"
    fallback_root = case_root / "poincare_fallback"
    payload: dict[str, object] = {
        "case_id": case_id,
        "banana_current_a": float(banana_current_a),
        "case_output_root": str(case_root),
        "classification": "dry_run" if args.dry_run else "boozer_failed",
    }

    variant_bs_path = stage2_variant_root / "biot_savart_opt.json"
    variant_results_path = stage2_variant_root / "results.json"
    if not args.dry_run:
        variant_bs_path, variant_results_path = _materialize_stage2_seed_variant(
            stage2_bs_path=stage2_bs_path,
            stage2_results=stage2_results,
            variant_root=stage2_variant_root,
            banana_current_a=banana_current_a,
            requested_num_tf_coils=args.num_tf_coils,
        )
    payload["stage2_variant_bs_path"] = str(variant_bs_path)
    payload["stage2_variant_results_path"] = str(variant_results_path)

    single_stage_args = _single_stage_case_args(args, variant_bs_path=variant_bs_path)
    try:
        run_payload = goal_mode_runner.run_goal_mode_case(
            single_stage_args,
            goal_mode="target",
            stage2_bs_path=variant_bs_path,
            output_root=single_stage_root,
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
                "single_stage_status": "failed",
            }
        )
        _set_case_error(payload, error)
    else:
        payload["single_stage_status"] = "dry_run" if args.dry_run else "completed"
        payload["single_stage_command"] = run_payload["command"]
        if not args.dry_run:
            payload["results_path"] = str(run_payload["results_path"])
            payload["result_source"] = run_payload["result_source"]
            payload["results_summary"] = _result_summary(run_payload["results"])

    poincare_output_root = None
    if args.dry_run:
        poincare_output_root = single_stage_root / "target"
        payload["poincare_status"] = "dry_run"
        payload["poincare_command"] = run_poincare_artifact(
            output_dir=poincare_output_root,
            python_executable=args.python_executable,
            timeout_seconds=poincare_timeout_seconds,
            dry_run=True,
        )
        return payload

    if payload.get("single_stage_status") == "completed":
        poincare_output_root = Path(str(payload["results_path"])).parent
    else:
        try:
            poincare_output_root = _materialize_poincare_fallback_inputs(
                args=args,
                variant_bs_path=variant_bs_path,
                stage2_results=stage2_results,
                fallback_root=fallback_root,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            payload["poincare_status"] = "failed"
            if payload.get("error_type") is None:
                _set_case_error(payload, error)
            else:
                _append_case_error_context(
                    payload,
                    context="poincare_fallback_setup_failed",
                    error=error,
                )
            return payload

    try:
        payload["poincare_command"] = run_poincare_artifact(
            output_dir=poincare_output_root,
            python_executable=args.python_executable,
            timeout_seconds=poincare_timeout_seconds,
        )
        poincare_metrics_path, poincare_metrics = _load_poincare_metrics(
            poincare_output_root
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        payload["poincare_status"] = "failed"
        if payload.get("error_type") is None:
            _set_case_error(payload, error)
    else:
        payload["poincare_status"] = "completed"
        payload["poincare_output_root"] = str(poincare_output_root)
        payload["poincare_metrics_path"] = str(poincare_metrics_path)
        payload["poincare_metrics"] = poincare_metrics

    if payload.get("single_stage_status") == "completed":
        payload["classification"] = "success"
    elif payload.get("poincare_status") == "completed":
        payload["classification"] = "poincare_only_fallback"
    else:
        payload["classification"] = "boozer_failed"
    return payload


def build_summary(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path,
    stage2_results_path: Path,
    stage2_results: dict,
    banana_currents_a: list[float],
    case_payloads: list[dict[str, object]],
    summary_csv_path: Path,
) -> dict[str, object]:
    return {
        "experiment_family": "banana_current_scan",
        "dry_run": bool(args.dry_run),
        "output_materialization": (
            "dry_run_summary_only" if args.dry_run else "materialized_scan_outputs"
        ),
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "stage2_bs_path": str(stage2_bs_path),
        "stage2_results_path": str(stage2_results_path),
        "stage2_artifact_init_only": stage2_results.get("init_only"),
        "optimized_banana_current_a": stage2_results.get("BANANA_CURRENT_A"),
        "requested_banana_currents_a": [float(value) for value in banana_currents_a],
        "summary_csv": str(summary_csv_path),
        "cases": case_payloads,
    }


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

    stage2_bs_path, stage2_results_path, stage2_results = (
        goal_mode_runner.load_validated_stage2_seed_metadata(args)
    )
    banana_currents_a = _resolved_banana_currents_a(
        args,
        donor_banana_current_a=float(stage2_results["BANANA_CURRENT_A"]),
    )
    poincare_timeout_seconds = timeout_or_none(args.poincare_timeout_seconds)
    case_payloads = [
        run_current_case(
            args,
            stage2_bs_path=stage2_bs_path,
            stage2_results=stage2_results,
            banana_current_a=banana_current_a,
            output_root=output_root,
            poincare_timeout_seconds=poincare_timeout_seconds,
        )
        for banana_current_a in banana_currents_a
    ]
    summary = build_summary(
        args,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        banana_currents_a=banana_currents_a,
        case_payloads=case_payloads,
        summary_csv_path=summary_csv_path,
    )
    write_json(summary_path, summary)
    write_csv_rows(
        summary_csv_path,
        [_csv_row(payload) for payload in case_payloads],
        fieldnames=CSV_FIELDNAMES,
    )

    if args.dry_run:
        write_dry_run_marker(
            output_root,
            summary_path=summary_path,
            runner_label="run_banana_current_scan.py",
        )
    else:
        clear_dry_run_marker(output_root)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
