from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_runner_common import (  # noqa: E402
    Stage2ArtifactConfig,
    build_stage2_command,
    clear_dry_run_marker,
    dry_run_marker_path,
    ensure_stage2_artifact,
    load_stage2_artifact_results,
    resolve_stage2_artifact_path,
    resolved_path,
    resolved_optional_path,
    timeout_or_none,
    write_dry_run_marker,
)
from banana_opt.artifact_contracts import (  # noqa: E402
    basin_metadata_from_config,
    upgrade_legacy_stage2_artifact_results,
    validate_stage2_artifact_metadata,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_stage2_alm"
STAGE2_CC_THRESHOLD_FLOOR = 0.05
STAGE2_CURVATURE_THRESHOLD_FLOOR = 40.0
DEFAULT_SUMMARY_JSON = "stage2_alm_summary.json"
_BASE_STAGE2_PROFILE = {
    "major_radius": 0.915,
    "toroidal_flux": 0.24,
    "length_weight": 0.0005,
    "cc_weight": 100.0,
    "cc_threshold": 0.05,
    "curvature_weight": 0.0001,
    "curvature_threshold": 40.0,
    "banana_surf_radius": 0.22,
    "order": 2,
    "banana_init_current_A": 1.0e4,
    "banana_current_max_A": 1.6e4,
    "alm_max_outer_iters": 10,
    "alm_penalty_init": 1.0,
    "alm_penalty_scale": 10.0,
    "basin_hops": 0,
    "basin_stepsize": 0.01,
    "basin_temperature": 1.0,
    "basin_niter_success": 0,
    "basin_seed": None,
    "init_only": False,
}
DEFAULT_STAGE2_PROFILES = {
    "standard_80ka": {**_BASE_STAGE2_PROFILE, "tf_current_A": 8.0e4},
    "standard_100ka": {**_BASE_STAGE2_PROFILE, "tf_current_A": 1.0e5},
}
STAGE2_SPEC_KEYS = (
    "major_radius",
    "toroidal_flux",
    "length_weight",
    "cc_weight",
    "cc_threshold",
    "curvature_weight",
    "curvature_threshold",
    "banana_surf_radius",
    "tf_current_A",
    "order",
    "banana_init_current_A",
    "banana_current_max_A",
    "alm_max_outer_iters",
    "alm_penalty_init",
    "alm_penalty_scale",
    "basin_hops",
    "basin_stepsize",
    "basin_temperature",
    "basin_niter_success",
    "basin_seed",
    "init_only",
)


def _jsonable_stage2_config(config: Stage2ArtifactConfig) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure a Stage 2 ALM artifact for any plasma surface using either a "
            "named built-in profile or a fully explicit Stage 2 spec JSON."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plasma-surf-filename",
        required=True,
        help="VMEC wout filename used as the Stage 2 target surface.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--profile",
        choices=sorted(DEFAULT_STAGE2_PROFILES),
        help="Built-in Stage 2 ALM parameter profile.",
    )
    source_group.add_argument(
        "--stage2-spec-json",
        help="Path to a full Stage 2 ALM spec JSON file.",
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
    parser.add_argument("--stage2-timeout-seconds", type=float, default=0.0)
    parser.add_argument(
        "--cc-threshold",
        type=float,
        default=None,
        help="Optional override for the Stage 2 coil-coil spacing threshold.",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=None,
        help="Optional override for the Stage 2 curvature threshold.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="Optional override for the banana coil Fourier order.",
    )
    parser.add_argument(
        "--tf-current-A",
        type=float,
        default=None,
        help="Optional override for the per-TF-coil current in SI amperes.",
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=None,
        help="Optional override for the VMEC flux-surface label s in [0, 1].",
    )
    return parser.parse_args()


def _normalize_basin_seed(*, basin_hops: int, basin_seed: int | None) -> int | None:
    if basin_hops <= 0:
        return None
    if basin_seed is not None and int(basin_seed) >= 0:
        return int(basin_seed)
    return int.from_bytes(os.urandom(4), "big")


def _load_stage2_spec_json(spec_json_path: str | Path) -> tuple[Path, dict]:
    spec_path = resolved_path(spec_json_path)
    with spec_path.open("r", encoding="utf-8") as infile:
        loaded = json.load(infile)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Stage 2 spec JSON must contain an object at the top level: {spec_path}"
        )
    unknown_keys = sorted(set(loaded) - set(STAGE2_SPEC_KEYS))
    if unknown_keys:
        raise ValueError(
            f"Unknown Stage 2 spec keys in {spec_path}: {', '.join(unknown_keys)}"
        )
    missing_keys = [key for key in STAGE2_SPEC_KEYS if key not in loaded]
    if missing_keys:
        raise ValueError(
            f"Stage 2 spec JSON must define all required keys: {', '.join(missing_keys)}"
        )
    return spec_path, {key: loaded[key] for key in STAGE2_SPEC_KEYS}


def resolve_stage2_spec_payload(args: argparse.Namespace) -> tuple[dict, str]:
    if args.profile is not None:
        resolved_spec = dict(DEFAULT_STAGE2_PROFILES[args.profile])
        source_label = f"profile:{args.profile}"
    else:
        spec_json_path, resolved_spec = _load_stage2_spec_json(args.stage2_spec_json)
        source_label = f"json:{spec_json_path}"

    overrides = {
        "toroidal_flux": args.toroidal_flux,
        "cc_threshold": args.cc_threshold,
        "curvature_threshold": args.curvature_threshold,
        "order": args.order,
        "tf_current_A": args.tf_current_A,
    }
    for key, value in overrides.items():
        if value is not None:
            resolved_spec[key] = value
    return resolved_spec, source_label


def build_stage2_alm_config(
    args: argparse.Namespace,
    *,
    resolved_spec: dict,
) -> Stage2ArtifactConfig:
    output_root = resolved_path(args.output_root)
    equilibria_dir = resolved_optional_path(args.equilibria_dir)
    basin_hops = int(resolved_spec["basin_hops"])
    basin_seed = _normalize_basin_seed(
        basin_hops=basin_hops,
        basin_seed=(
            None if resolved_spec["basin_seed"] is None else int(resolved_spec["basin_seed"])
        ),
    )
    raw_cc = float(resolved_spec["cc_threshold"])
    cc_threshold = max(raw_cc, STAGE2_CC_THRESHOLD_FLOOR)
    if raw_cc < STAGE2_CC_THRESHOLD_FLOOR:
        print(
            f"WARNING: cc_threshold {raw_cc} below Stage 2 "
            f"solver floor, clamped to {STAGE2_CC_THRESHOLD_FLOOR}"
        )
    raw_curvature = float(resolved_spec["curvature_threshold"])
    curvature_threshold = max(raw_curvature, STAGE2_CURVATURE_THRESHOLD_FLOOR)
    if raw_curvature < STAGE2_CURVATURE_THRESHOLD_FLOOR:
        print(
            f"WARNING: curvature_threshold {raw_curvature} "
            f"below Stage 2 solver floor, clamped to "
            f"{STAGE2_CURVATURE_THRESHOLD_FLOOR}"
        )
    return Stage2ArtifactConfig(
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        output_root=output_root,
        equilibria_dir=None if equilibria_dir is None else str(equilibria_dir),
        tf_current_A=float(resolved_spec["tf_current_A"]),
        major_radius=float(resolved_spec["major_radius"]),
        toroidal_flux=float(resolved_spec["toroidal_flux"]),
        length_weight=float(resolved_spec["length_weight"]),
        cc_weight=float(resolved_spec["cc_weight"]),
        cc_threshold=cc_threshold,
        curvature_weight=float(resolved_spec["curvature_weight"]),
        curvature_threshold=curvature_threshold,
        banana_surf_radius=float(resolved_spec["banana_surf_radius"]),
        order=int(resolved_spec["order"]),
        constraint_method="alm",
        alm_max_outer_iters=int(resolved_spec["alm_max_outer_iters"]),
        alm_penalty_init=float(resolved_spec["alm_penalty_init"]),
        alm_penalty_scale=float(resolved_spec["alm_penalty_scale"]),
        basin_hops=basin_hops,
        basin_stepsize=float(resolved_spec["basin_stepsize"]),
        basin_temperature=float(resolved_spec["basin_temperature"]),
        basin_niter_success=int(resolved_spec["basin_niter_success"]),
        basin_seed=basin_seed,
        init_only=bool(resolved_spec["init_only"]),
        banana_init_current_A=float(resolved_spec["banana_init_current_A"]),
        banana_current_max_A=float(resolved_spec["banana_current_max_A"]),
    )


def _expected_stage2_artifact_metadata(config: Stage2ArtifactConfig) -> dict:
    return {
        "PLASMA_SURF_FILENAME": Path(config.plasma_surf_filename).name,
        "TF_CURRENT_A": config.tf_current_A,
        "BANANA_INIT_CURRENT_A": config.banana_init_current_A,
        "BANANA_CURRENT_MAX_A": config.banana_current_max_A,
        "MAJOR_RADIUS": config.major_radius,
        "TOROIDAL_FLUX": config.toroidal_flux,
        "LENGTH_WEIGHT": config.length_weight,
        "CC_WEIGHT": config.cc_weight,
        "CC_THRESHOLD": config.cc_threshold,
        "CURVATURE_WEIGHT": config.curvature_weight,
        "CURVATURE_THRESHOLD": config.curvature_threshold,
        "banana_surf_radius": config.banana_surf_radius,
        "order": config.order,
        "CONSTRAINT_METHOD": config.constraint_method,
        **basin_metadata_from_config(config),
        "init_only": config.init_only,
    }


def load_validated_stage2_artifact(
    config: Stage2ArtifactConfig,
) -> tuple[Path, dict]:
    artifact_path = resolve_stage2_artifact_path(config)
    stage2_results_path, stage2_results = load_stage2_artifact_results(artifact_path)
    stage2_results = upgrade_legacy_stage2_artifact_results(stage2_results)
    validate_stage2_artifact_metadata(
        stage2_results_path,
        stage2_results,
        expected_metadata=_expected_stage2_artifact_metadata(config),
        owner_label="run_stage2_alm.py",
        experiment_family="generic Stage 2 ALM",
    )
    return stage2_results_path, stage2_results


def build_summary(
    args: argparse.Namespace,
    *,
    config: Stage2ArtifactConfig,
    resolved_spec_source: str,
    command: list[str],
    artifact_path: Path,
    artifact_reused: bool,
    stage2_results_path: Path | None = None,
    stage2_results: dict | None = None,
) -> dict:
    summary = {
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "resolved_spec_source": resolved_spec_source,
        "artifact_path": str(artifact_path),
        "artifact_reused": artifact_reused,
        "command": command,
        "dry_run": bool(args.dry_run),
        "output_contract": (
            "dry_run_summary_only" if args.dry_run else "materialized_stage2_artifact"
        ),
        "contains_solver_outputs": bool(stage2_results_path is not None and stage2_results is not None),
        "dry_run_marker_path": str(dry_run_marker_path(config.output_root)),
        "resolved_stage2_config": _jsonable_stage2_config(config),
    }
    if stage2_results_path is None or stage2_results is None:
        return summary
    summary.update(
        {
            "stage2_results_path": str(stage2_results_path),
            "termination_message": stage2_results.get("TERMINATION_MESSAGE"),
            "optimizer_success": stage2_results.get("OPTIMIZER_SUCCESS"),
            "alm_outer_iterations": stage2_results.get("ALM_OUTER_ITERATIONS"),
            "alm_final_penalty": stage2_results.get("ALM_FINAL_PENALTY"),
            "curve_curve_min_dist": stage2_results.get("CURVE_CURVE_MIN_DIST"),
            "max_curvature": stage2_results.get("MAX_CURVATURE"),
            "coil_length": stage2_results.get("COIL_LENGTH"),
            "field_error": stage2_results.get("FIELD_ERROR"),
            "hardware_constraints_ok": stage2_results.get("HARDWARE_CONSTRAINTS_OK"),
        }
    )
    return summary


def main() -> int:
    args = parse_args()
    resolved_spec, resolved_spec_source = resolve_stage2_spec_payload(args)
    config = build_stage2_alm_config(args, resolved_spec=resolved_spec)
    artifact_path = resolve_stage2_artifact_path(config)
    artifact_reused = artifact_path.exists()
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    command = build_stage2_command(config, python_executable=args.python_executable)
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary = build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=artifact_path,
            artifact_reused=artifact_reused,
        )
        write_dry_run_marker(
            output_root,
            summary_path=summary_path,
            runner_label="run_stage2_alm.py",
        )
    else:
        clear_dry_run_marker(output_root)
        artifact_path = ensure_stage2_artifact(
            config,
            python_executable=args.python_executable,
            timeout_seconds=timeout_or_none(args.stage2_timeout_seconds),
            dry_run=False,
        )
        stage2_results_path, stage2_results = load_validated_stage2_artifact(config)
        summary = build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=artifact_path,
            artifact_reused=artifact_reused,
            stage2_results_path=stage2_results_path,
            stage2_results=stage2_results,
        )

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
