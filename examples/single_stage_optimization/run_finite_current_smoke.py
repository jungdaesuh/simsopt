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
    Stage2ArtifactConfig,
    add_seed_order_upgrade_argument,
    build_stage2_command,
    discover_single_results_path,
    ensure_stage2_artifact,
    load_json,
    load_stage2_artifact_results,
    parse_csv,
    run_command,
    snapshot_single_results_paths,
    timeout_or_none,
)
from banana_opt.artifact_contracts import (  # noqa: E402
    resolve_expected_stage2_tf_current_A,
    resolve_expected_stage2_tf_current_sum_abs_A,
    upgrade_legacy_stage2_artifact_results,
    validate_smoke_results as _validate_smoke_results_impl,
)
from banana_opt.current_contracts import physical_current_to_boozer_I  # noqa: E402
from banana_opt.current_contracts import (  # noqa: E402
    DEFAULT_FINITE_CURRENT_MODE,
    resolve_boozer_current_convention,
)
from banana_opt.hardware_contracts import (  # noqa: E402
    BANANA_WINDING_MINOR_RADIUS_M,
    COIL_COIL_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    TF_CURRENT_CW_DEFAULT_A,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
)
from banana_opt.constraint_contract import (  # noqa: E402
    resolve_constraint_contract_from_wire_names,
)
from banana_opt.single_stage_banana_current_mode import (  # noqa: E402
    BANANA_CURRENT_MODE_INDEPENDENT,
    BANANA_CURRENT_MODE_SHARED,
)

DEFAULT_PLASMA_SURF_FILENAME = "wout_nfp22ginsburg_000_014417_iota15.nc"
DEFAULT_SMOKE_OUTPUT_ROOT = SCRIPT_DIR / "outputs_finite_current_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight finite-current smoke harness against a frozen "
            "coil-only Stage 2 artifact."
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
        default=str(DEFAULT_SMOKE_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary path. Defaults to <output-root>/smoke_summary.json.",
    )
    parser.add_argument(
        "--stage2-output-root",
        default=str(SCRIPT_DIR / "STAGE_2"),
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=None,
        help="Explicit Stage 2 biot_savart_opt.json path. Overrides derived seed settings.",
    )
    add_seed_order_upgrade_argument(parser)
    parser.add_argument("--stage2-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--single-stage-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--currents-A", default="0,8000,-35200")
    parser.add_argument(
        "--single-stage-banana-current-mode",
        choices=[BANANA_CURRENT_MODE_SHARED, BANANA_CURRENT_MODE_INDEPENDENT],
        default=BANANA_CURRENT_MODE_SHARED,
        help=(
            "Banana-current control mode forwarded into the single-stage run. "
            "'shared' preserves the legacy one-current contract, while "
            "'independent' gives each loaded banana coil its own current DOF."
        ),
    )
    parser.add_argument("--tf-current-A", type=float, default=TF_CURRENT_CW_DEFAULT_A)
    parser.add_argument("--major-radius", type=float, default=VACUUM_VESSEL_MAJOR_RADIUS_M)
    parser.add_argument("--toroidal-flux", type=float, default=0.24)
    parser.add_argument("--stage2-length-weight", type=float, default=0.0005)
    parser.add_argument("--stage2-cc-weight", type=float, default=100.0)
    parser.add_argument("--stage2-cc-threshold", type=float, default=COIL_COIL_MIN_DIST_M)
    parser.add_argument("--stage2-curvature-weight", type=float, default=0.0001)
    parser.add_argument("--stage2-curvature-threshold", type=float, default=MAX_CURVATURE_INV_M)
    parser.add_argument("--banana-surf-radius", type=float, default=BANANA_WINDING_MINOR_RADIUS_M)
    parser.add_argument("--stage2-order", type=int, default=2)
    parser.add_argument("--nphi", type=int, default=41)
    parser.add_argument("--ntheta", type=int, default=16)
    parser.add_argument("--mpol", type=int, default=4)
    parser.add_argument("--ntor", type=int, default=4)
    return parser.parse_args()


def _resolve_smoke_constraint_contract(args: argparse.Namespace) -> dict[str, float]:
    contract, _trace = resolve_constraint_contract_from_wire_names(
        cli_overrides={
            "tf_current_A": args.tf_current_A,
            "cc_threshold": args.stage2_cc_threshold,
            "curvature_threshold": args.stage2_curvature_threshold,
            "banana_surf_radius": args.banana_surf_radius,
        },
        offspec_major_radius_m=args.major_radius,
        accept_offspec_major_radius=False,
    )
    return dict(contract)


def make_stage2_config(args: argparse.Namespace) -> Stage2ArtifactConfig:
    constraint_contract = _resolve_smoke_constraint_contract(args)
    return Stage2ArtifactConfig(
        plasma_surf_filename=args.plasma_surf_filename,
        output_root=Path(args.stage2_output_root),
        equilibria_dir=args.equilibria_dir,
        tf_current_A=float(constraint_contract["TF_CURRENT_A"]),
        major_radius=float(constraint_contract["VACUUM_VESSEL_MAJOR_RADIUS_M"]),
        toroidal_flux=args.toroidal_flux,
        length_weight=args.stage2_length_weight,
        cc_weight=args.stage2_cc_weight,
        cc_threshold=float(constraint_contract["CC_THRESHOLD"]),
        curvature_weight=args.stage2_curvature_weight,
        curvature_threshold=float(constraint_contract["CURVATURE_THRESHOLD"]),
        banana_surf_radius=float(constraint_contract["banana_surf_radius"]),
        order=args.stage2_order,
        constraint_method="penalty",
        alm_max_outer_iters=10,
        alm_penalty_init=1.0,
        alm_penalty_scale=10.0,
        alm_penalty_max=1.0e8,
        basin_hops=0,
        basin_stepsize=0.01,
        basin_seed=None,
        init_only=True,
        banana_current_max_A=float(constraint_contract["BANANA_CURRENT_MAX_A"]),
        length_target=float(constraint_contract["COIL_LENGTH_TARGET_M"]),
    )


def build_smoke_command(
    args: argparse.Namespace,
    *,
    current_A: float,
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
        "--plasma-current-A",
        str(current_A),
        "--single-stage-banana-current-mode",
        args.single_stage_banana_current_mode,
        "--nphi",
        str(args.nphi),
        "--ntheta",
        str(args.ntheta),
        "--mpol",
        str(args.mpol),
        "--ntor",
        str(args.ntor),
        "--constraint-method",
        "penalty",
        "--init-only",
    ]
    if getattr(args, "seed_order_upgrade", None) is not None:
        command.extend(["--seed-order-upgrade", str(args.seed_order_upgrade)])
    if args.equilibria_dir is not None:
        command.extend(["--equilibria-dir", args.equilibria_dir])
    return command


def run_smoke_case(
    args: argparse.Namespace,
    *,
    current_A: float,
    stage2_bs_path: Path,
    expected_stage2_tf_current_A: float,
    expected_stage2_tf_current_sum_abs_A: float,
    smoke_output_root: Path,
) -> dict:
    current_label = str(current_A).replace("-", "m").replace(".", "p")
    case_output_root = smoke_output_root / f"current_{current_label}"
    case_output_root.mkdir(parents=True, exist_ok=True)
    previous_snapshot = snapshot_single_results_paths(case_output_root)
    command = build_smoke_command(
        args,
        current_A=current_A,
        stage2_bs_path=stage2_bs_path,
        case_output_root=case_output_root,
    )
    if args.dry_run:
        return {
            "requested_current_A": current_A,
            "case_output_root": str(case_output_root),
            "command": command,
        }
    run_command(
        command,
        timeout_seconds=timeout_or_none(args.single_stage_timeout_seconds),
    )
    results_path = discover_single_results_path(
        case_output_root,
        previous_snapshot=previous_snapshot,
    )
    results = load_json(results_path)
    validation = validate_smoke_results(
        results,
        requested_current_A=current_A,
        expected_stage2_tf_current_A=expected_stage2_tf_current_A,
        expected_stage2_tf_current_sum_abs_A=expected_stage2_tf_current_sum_abs_A,
    )
    return {
        "requested_current_A": current_A,
        "case_output_root": str(case_output_root),
        "results_path": str(results_path),
        "validation": validation,
        "results": results,
    }


def validate_smoke_results(
    results: dict,
    *,
    requested_current_A: float,
    expected_stage2_tf_current_A: float,
    expected_stage2_tf_current_sum_abs_A: float,
) -> dict:
    return _validate_smoke_results_impl(
        results,
        requested_current_A=requested_current_A,
        expected_boozer_I=physical_current_to_boozer_I(
            requested_current_A,
            convention=resolve_boozer_current_convention(DEFAULT_FINITE_CURRENT_MODE),
        ),
        expected_stage2_tf_current_A=expected_stage2_tf_current_A,
        expected_stage2_tf_current_sum_abs_A=expected_stage2_tf_current_sum_abs_A,
        expected_finite_current_mode=DEFAULT_FINITE_CURRENT_MODE,
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
            timeout_seconds=timeout_or_none(args.stage2_timeout_seconds),
            dry_run=args.dry_run,
        )
    )
    stage2_results_path = None
    stage2_artifact_results = None
    expected_stage2_tf_current_A = None
    expected_stage2_tf_current_sum_abs_A = None
    if not args.dry_run:
        stage2_results_path, stage2_artifact_results = load_stage2_artifact_results(
            stage2_bs_path
        )
        stage2_artifact_results = upgrade_legacy_stage2_artifact_results(
            stage2_artifact_results
        )
        expected_stage2_tf_current_A = resolve_expected_stage2_tf_current_A(
            stage2_artifact_results
        )
        expected_stage2_tf_current_sum_abs_A = (
            resolve_expected_stage2_tf_current_sum_abs_A(stage2_artifact_results)
        )
    smoke_output_root = Path(args.output_root)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json is not None
        else smoke_output_root / "smoke_summary.json"
    )
    currents = parse_csv(args.currents_A, float)
    records = [
        run_smoke_case(
            args,
            current_A=current_A,
            stage2_bs_path=stage2_bs_path,
            expected_stage2_tf_current_A=expected_stage2_tf_current_A,
            expected_stage2_tf_current_sum_abs_A=expected_stage2_tf_current_sum_abs_A,
            smoke_output_root=smoke_output_root,
        )
        for current_A in currents
    ]
    summary = {
        "stage2_bs_path": str(stage2_bs_path),
        "stage2_results_path": str(stage2_results_path) if stage2_results_path is not None else None,
        "stage2_artifact_results": stage2_artifact_results,
        "stage2_command": build_stage2_command(
            stage2_config,
            python_executable=args.python_executable,
        ) if args.stage2_bs_path is None else None,
        "records": records,
        "all_cases_passed": all(
            record.get("validation", {}).get("passed", False)
            for record in records
        ) if not args.dry_run else None,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(f"Wrote smoke summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
