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
from workflow_helpers import (  # noqa: E402
    canonical_stage2_iota_constraint_weight,
    validate_stage2_iota_args,
)
from banana_opt.artifact_contracts import (  # noqa: E402
    basin_metadata_from_config,
    upgrade_legacy_stage2_artifact_results,
    validate_stage2_artifact_metadata,
)
from banana_opt.current_contracts import DEFAULT_FINITE_CURRENT_MODE  # noqa: E402
from banana_opt.hardware_contracts import (  # noqa: E402
    BANANA_CURRENT_HARD_LIMIT_A,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_HARD_LIMIT_M,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    TARGET_LCFS_MAX_MINOR_RADIUS_M,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    fixed_stage2_artifact_hardware_contract,
    fixed_stage2_clearance_contract,
    TF_CURRENT_CW_DEFAULT_A,
    validate_major_radius,
)
from banana_opt.constraint_contract import (  # noqa: E402
    apply_offspec_engineering_override_reason,
    build_constraint_metadata,
    resolve_constraint_contract_from_wire_names,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_stage2_alm"
STAGE2_CC_THRESHOLD_FLOOR = COIL_COIL_MIN_DIST_M
DEFAULT_SUMMARY_JSON = "stage2_alm_summary.json"
_BASE_STAGE2_PROFILE = {
    "major_radius": VACUUM_VESSEL_MAJOR_RADIUS_M,
    "toroidal_flux": 0.24,
    "length_weight": 0.0005,
    "cc_weight": 100.0,
    "cc_threshold": COIL_COIL_MIN_DIST_M,
    "curvature_weight": 0.0001,
    "curvature_threshold": MAX_CURVATURE_INV_M,
    "banana_surf_radius": 0.21,
    "order": 2,
    "banana_init_current_A": 1.0e4,
    "banana_current_max_A": BANANA_CURRENT_HARD_LIMIT_A,
    "length_target": COIL_LENGTH_TARGET_M,
    "target_lcfs_max_major_radius_m": TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    "target_lcfs_max_minor_radius_m": TARGET_LCFS_MAX_MINOR_RADIUS_M,
    "finite_current_mode": DEFAULT_FINITE_CURRENT_MODE,
    "proxy_plasma_current_A": 0.0,
    "vf_current_A": 0.0,
    "vf_template_path": None,
    "alm_max_outer_iters": 10,
    "alm_penalty_init": 1.0,
    "alm_penalty_scale": 10.0,
    "alm_penalty_max": 1.0e8,
    "alm_feas_tol": 1.0e-6,
    "alm_stationarity_tol": 1.0e-6,
    "alm_trust_radius_init": 0.05,
    "alm_trust_radius_min": 1.0e-4,
    "alm_trust_radius_shrink": 0.5,
    "alm_trust_radius_grow": 1.5,
    "alm_max_inner_attempts": 4,
    "alm_max_subproblem_continuations": 20,
    "alm_distance_smoothing": 0.005,
    "alm_curvature_smoothing": 0.25,
    "basin_hops": 0,
    "basin_stepsize": 0.01,
    "basin_temperature": 1.0,
    "basin_niter_success": 0,
    "basin_seed": None,
    "init_only": False,
}
DEFAULT_STAGE2_PROFILES = {
    "standard_80ka": {**_BASE_STAGE2_PROFILE, "tf_current_A": TF_CURRENT_CW_DEFAULT_A},
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
    "length_target",
    "target_lcfs_max_major_radius_m",
    "target_lcfs_max_minor_radius_m",
    "finite_current_mode",
    "proxy_plasma_current_A",
    "vf_current_A",
    "vf_template_path",
    "alm_max_outer_iters",
    "alm_penalty_init",
    "alm_penalty_scale",
    "alm_penalty_max",
    "alm_feas_tol",
    "alm_stationarity_tol",
    "alm_trust_radius_init",
    "alm_trust_radius_min",
    "alm_trust_radius_shrink",
    "alm_trust_radius_grow",
    "alm_max_inner_attempts",
    "alm_max_subproblem_continuations",
    "alm_distance_smoothing",
    "alm_curvature_smoothing",
    "basin_hops",
    "basin_stepsize",
    "basin_temperature",
    "basin_niter_success",
    "basin_seed",
    "init_only",
)
_STAGE2_CONTRACT_OVERRIDE_KEYS = (
    "tf_current_A",
    "banana_current_max_A",
    "length_target",
    "cc_threshold",
    "curvature_threshold",
    "banana_surf_radius",
    "target_lcfs_max_major_radius_m",
    "target_lcfs_max_minor_radius_m",
)
_STAGE2_SPEC_OVERRIDE_KEYS = (
    "toroidal_flux",
    "order",
    *_STAGE2_CONTRACT_OVERRIDE_KEYS,
)
OPTIONAL_STAGE2_SPEC_KEYS = (
    "finite_current_mode",
    "proxy_plasma_current_A",
    "vf_current_A",
    "vf_template_path",
    "length_target",
    "target_lcfs_max_major_radius_m",
    "target_lcfs_max_minor_radius_m",
    "alm_feas_tol",
    "alm_stationarity_tol",
    "alm_trust_radius_init",
    "alm_trust_radius_min",
    "alm_trust_radius_shrink",
    "alm_trust_radius_grow",
    "alm_max_inner_attempts",
    "alm_max_subproblem_continuations",
    "alm_distance_smoothing",
    "alm_curvature_smoothing",
)


def _jsonable_stage2_config(config: Stage2ArtifactConfig) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--allow-offspec-engineering-constraints",
        action="store_true",
        help=argparse.SUPPRESS,
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
        "--banana-current-max-A",
        type=float,
        default=None,
        help="Optional override for the banana-current hardware ceiling in SI amperes.",
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=None,
        help=(
            "Optional override for the Stage 2 coil-length target in meters "
            f"(default {COIL_LENGTH_TARGET_M:.1f}, hardware ceiling {COIL_LENGTH_HARD_LIMIT_M:.1f})."
        ),
    )
    parser.add_argument(
        "--target-lcfs-max-major-radius-m",
        type=float,
        default=None,
        help="Optional override for the requested-plasma LCFS major-radius ceiling.",
    )
    parser.add_argument(
        "--target-lcfs-max-minor-radius-m",
        type=float,
        default=None,
        help="Optional override for the requested-plasma LCFS minor-radius ceiling.",
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=None,
        help="Optional override for the Stage 2 banana winding-surface minor radius.",
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=None,
        help="Optional override for the VMEC flux-surface label s in [0, 1].",
    )
    parser.add_argument(
        "--stage2-iota-mode",
        choices=["off", "report", "alm"],
        default="off",
        help=(
            "Optional Stage 2 iota mode. 'report' runs only the final verification "
            "probe, and 'alm' adds a hard Stage 2 ALM iota_penalty constraint. "
            "This wrapper pins --constraint-method=alm, so soft mode is not exposed here."
        ),
    )
    parser.add_argument(
        "--stage2-iota-target",
        type=float,
        default=None,
        help="Target iota used when --stage2-iota-mode is enabled.",
    )
    parser.add_argument(
        "--stage2-iota-tolerance",
        type=float,
        default=5.0e-3,
        help="Absolute iota tolerance used when --stage2-iota-mode is enabled.",
    )
    parser.add_argument(
        "--stage2-iota-vol-target",
        type=float,
        default=0.10,
        help="Outer-surface target volume passed to the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-constraint-weight",
        type=float,
        default=1.0,
        help=(
            "Boozer constraint weight used by the Stage 2 iota solve. Use a "
            "non-positive value to select the exact Boozer Newton solve."
        ),
    )
    parser.add_argument(
        "--stage2-iota-num-tf-coils",
        type=int,
        default=20,
        help="Expected TF-coil count used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-nphi",
        type=int,
        default=91,
        help="Surface quadrature nphi used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-ntheta",
        type=int,
        default=32,
        help="Surface quadrature ntheta used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-mpol",
        type=int,
        default=8,
        help="Boozer-surface mpol used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-ntor",
        type=int,
        default=6,
        help="Boozer-surface ntor used by the Stage 2 Boozer/iota solve.",
    )
    return parser.parse_args(argv)


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
    missing_keys = [
        key
        for key in STAGE2_SPEC_KEYS
        if key not in loaded and key not in OPTIONAL_STAGE2_SPEC_KEYS
    ]
    if missing_keys:
        raise ValueError(
            f"Stage 2 spec JSON must define all required keys: {', '.join(missing_keys)}"
        )
    return spec_path, {
        key: loaded[key] if key in loaded else _BASE_STAGE2_PROFILE[key]
        for key in STAGE2_SPEC_KEYS
    }


def _stage2_arg_overrides(
    args: argparse.Namespace,
    keys: tuple[str, ...],
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _stage2_override_reason(
    args: argparse.Namespace,
    keys: tuple[str, ...],
) -> str | None:
    override_names = sorted(_stage2_arg_overrides(args, keys))
    if not override_names:
        return None
    return f"cli:{','.join(override_names)}"


def resolve_stage2_spec_payload(args: argparse.Namespace) -> tuple[dict, str]:
    if args.profile is not None:
        resolved_spec = dict(DEFAULT_STAGE2_PROFILES[args.profile])
        source_label = f"profile:{args.profile}"
    else:
        spec_json_path, resolved_spec = _load_stage2_spec_json(args.stage2_spec_json)
        source_label = f"json:{spec_json_path}"
    resolved_spec.update(_stage2_arg_overrides(args, _STAGE2_SPEC_OVERRIDE_KEYS))
    return resolved_spec, source_label


def _stage2_spec_constraint_layer(resolved_spec: dict) -> dict[str, object]:
    return {
        key: resolved_spec[key]
        for key in _STAGE2_CONTRACT_OVERRIDE_KEYS
        if key in resolved_spec
    }


def _resolve_stage2_constraint_contract(
    args: argparse.Namespace,
    *,
    resolved_spec: dict,
) -> dict[str, float]:
    validate_major_radius(resolved_spec["major_radius"])
    allow_offspec_engineering_constraints = bool(
        getattr(args, "allow_offspec_engineering_constraints", False)
    )
    contract, _ = resolve_constraint_contract_from_wire_names(
        profile=_stage2_spec_constraint_layer(resolved_spec),
        cli_overrides=_stage2_arg_overrides(args, _STAGE2_CONTRACT_OVERRIDE_KEYS),
        allow_offspec_engineering=allow_offspec_engineering_constraints,
    )
    return dict(contract)


def _stage2_config_constraint_layer(config: Stage2ArtifactConfig) -> dict[str, float]:
    return {
        "tf_current_A": float(config.tf_current_A),
        "banana_current_max_A": float(config.banana_current_max_A),
        "length_target": float(config.length_target),
        "cc_threshold": float(config.cc_threshold),
        "curvature_threshold": float(config.curvature_threshold),
        "banana_surf_radius": float(config.banana_surf_radius),
        "target_lcfs_max_major_radius_m": float(config.target_lcfs_max_major_radius_m),
        "target_lcfs_max_minor_radius_m": float(config.target_lcfs_max_minor_radius_m),
    }


def build_stage2_constraint_artifacts(
    *,
    args: argparse.Namespace,
    config: Stage2ArtifactConfig,
    source_label: str,
) -> dict:
    """Route the resolved Stage 2 spec through the shared constraint contract.

    Returns the artifact-ready metadata block (CONSTRAINT_PROFILE,
    EFFECTIVE_VALUES, OVERRIDE_REASON, CONTRACT_HASH, CONTRACT_SCHEMA_VERSION)
    built against the effective values that actually drive
    :class:`Stage2ArtifactConfig`.
    """
    allow_offspec_engineering_constraints = bool(
        getattr(args, "allow_offspec_engineering_constraints", False)
    )
    constraint_layer = _stage2_config_constraint_layer(config)
    contract, _ = resolve_constraint_contract_from_wire_names(
        cli_overrides=constraint_layer,
        allow_offspec_engineering=allow_offspec_engineering_constraints,
    )
    override_reason = _stage2_override_reason(
        args,
        _STAGE2_CONTRACT_OVERRIDE_KEYS,
    )
    override_reason = apply_offspec_engineering_override_reason(
        override_reason,
        layer=constraint_layer,
        allow_offspec_engineering=allow_offspec_engineering_constraints,
    )
    return build_constraint_metadata(
        contract,
        profile_name=source_label,
        override_reason=override_reason,
    )


_CONSTRAINT_METADATA_KEYS = (
    "CONSTRAINT_PROFILE",
    "EFFECTIVE_VALUES",
    "OVERRIDE_REASON",
    "CONTRACT_HASH",
    "CONTRACT_SCHEMA_VERSION",
)
_CONSTRAINT_IDENTITY_KEYS = (
    "EFFECTIVE_VALUES",
    "CONTRACT_HASH",
    "CONTRACT_SCHEMA_VERSION",
)


def _artifact_uses_legacy_constraint_metadata(stage2_results: dict) -> bool:
    return (
        int(stage2_results.get("CONTRACT_SCHEMA_VERSION", 0)) == 0
        and stage2_results.get("CONTRACT_HASH") in {None, ""}
    )


def _constraint_metadata_expectation(constraint_metadata: dict) -> dict:
    return {key: constraint_metadata[key] for key in _CONSTRAINT_IDENTITY_KEYS}


def build_stage2_alm_config(
    args: argparse.Namespace,
    *,
    resolved_spec: dict,
) -> Stage2ArtifactConfig:
    constraint_contract = _resolve_stage2_constraint_contract(
        args,
        resolved_spec=resolved_spec,
    )
    output_root = resolved_path(args.output_root)
    equilibria_dir = resolved_optional_path(args.equilibria_dir)
    basin_hops = int(resolved_spec["basin_hops"])
    basin_seed = _normalize_basin_seed(
        basin_hops=basin_hops,
        basin_seed=(
            None if resolved_spec["basin_seed"] is None else int(resolved_spec["basin_seed"])
        ),
    )
    raw_cc = float(constraint_contract["CC_THRESHOLD"])
    cc_threshold = max(raw_cc, STAGE2_CC_THRESHOLD_FLOOR)
    if raw_cc < STAGE2_CC_THRESHOLD_FLOOR:
        print(
            f"WARNING: cc_threshold {raw_cc} below Stage 2 "
            f"solver floor, clamped to {STAGE2_CC_THRESHOLD_FLOOR}"
        )
    curvature_threshold = float(constraint_contract["CURVATURE_THRESHOLD"])
    tf_current_A = float(constraint_contract["TF_CURRENT_A"])
    stage2_iota_mode = getattr(args, "stage2_iota_mode", "off")
    stage2_iota_target = getattr(args, "stage2_iota_target", None)
    stage2_iota_tolerance = getattr(args, "stage2_iota_tolerance", 5.0e-3)
    stage2_iota_vol_target = getattr(args, "stage2_iota_vol_target", 0.10)
    stage2_iota_constraint_weight = getattr(
        args,
        "stage2_iota_constraint_weight",
        1.0,
    )
    stage2_iota_num_tf_coils = getattr(args, "stage2_iota_num_tf_coils", 20)
    stage2_iota_nphi = getattr(args, "stage2_iota_nphi", 91)
    stage2_iota_ntheta = getattr(args, "stage2_iota_ntheta", 32)
    stage2_iota_mpol = getattr(args, "stage2_iota_mpol", 8)
    stage2_iota_ntor = getattr(args, "stage2_iota_ntor", 6)
    return Stage2ArtifactConfig(
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        output_root=output_root,
        equilibria_dir=None if equilibria_dir is None else str(equilibria_dir),
        tf_current_A=tf_current_A,
        major_radius=float(constraint_contract["VACUUM_VESSEL_MAJOR_RADIUS_M"]),
        toroidal_flux=float(resolved_spec["toroidal_flux"]),
        length_weight=float(resolved_spec["length_weight"]),
        cc_weight=float(resolved_spec["cc_weight"]),
        cc_threshold=cc_threshold,
        curvature_weight=float(resolved_spec["curvature_weight"]),
        curvature_threshold=curvature_threshold,
        banana_surf_radius=float(constraint_contract["banana_surf_radius"]),
        order=int(resolved_spec["order"]),
        constraint_method="alm",
        alm_max_outer_iters=int(resolved_spec["alm_max_outer_iters"]),
        alm_penalty_init=float(resolved_spec["alm_penalty_init"]),
        alm_penalty_scale=float(resolved_spec["alm_penalty_scale"]),
        alm_penalty_max=float(resolved_spec["alm_penalty_max"]),
        alm_feas_tol=float(resolved_spec["alm_feas_tol"]),
        alm_stationarity_tol=float(resolved_spec["alm_stationarity_tol"]),
        alm_trust_radius_init=float(resolved_spec["alm_trust_radius_init"]),
        alm_trust_radius_min=float(resolved_spec["alm_trust_radius_min"]),
        alm_trust_radius_shrink=float(resolved_spec["alm_trust_radius_shrink"]),
        alm_trust_radius_grow=float(resolved_spec["alm_trust_radius_grow"]),
        alm_max_inner_attempts=int(resolved_spec["alm_max_inner_attempts"]),
        alm_max_subproblem_continuations=int(
            resolved_spec["alm_max_subproblem_continuations"]
        ),
        alm_distance_smoothing=float(resolved_spec["alm_distance_smoothing"]),
        alm_curvature_smoothing=float(resolved_spec["alm_curvature_smoothing"]),
        basin_hops=basin_hops,
        basin_stepsize=float(resolved_spec["basin_stepsize"]),
        basin_temperature=float(resolved_spec["basin_temperature"]),
        basin_niter_success=int(resolved_spec["basin_niter_success"]),
        basin_seed=basin_seed,
        init_only=bool(resolved_spec["init_only"]),
        banana_init_current_A=float(resolved_spec["banana_init_current_A"]),
        banana_current_max_A=float(constraint_contract["BANANA_CURRENT_MAX_A"]),
        length_target=float(constraint_contract["COIL_LENGTH_TARGET_M"]),
        finite_current_mode=str(
            resolved_spec.get("finite_current_mode", DEFAULT_FINITE_CURRENT_MODE)
        ),
        proxy_plasma_current_A=float(
            resolved_spec.get("proxy_plasma_current_A", 0.0)
        ),
        vf_current_A=float(resolved_spec.get("vf_current_A", 0.0)),
        vf_template_path=resolved_spec.get("vf_template_path"),
        target_lcfs_max_major_radius_m=float(
            constraint_contract["TARGET_LCFS_MAX_MAJOR_RADIUS_M"]
        ),
        target_lcfs_max_minor_radius_m=float(
            constraint_contract["TARGET_LCFS_MAX_MINOR_RADIUS_M"]
        ),
        stage2_iota_mode=stage2_iota_mode,
        stage2_iota_target=stage2_iota_target,
        stage2_iota_tolerance=stage2_iota_tolerance,
        stage2_iota_weight=1.0,
        stage2_iota_vol_target=stage2_iota_vol_target,
        stage2_iota_constraint_weight=stage2_iota_constraint_weight,
        stage2_iota_num_tf_coils=stage2_iota_num_tf_coils,
        stage2_iota_nphi=stage2_iota_nphi,
        stage2_iota_ntheta=stage2_iota_ntheta,
        stage2_iota_mpol=stage2_iota_mpol,
        stage2_iota_ntor=stage2_iota_ntor,
    )


def _expected_stage2_alm_solver_metadata(config: Stage2ArtifactConfig) -> dict:
    return {
        "ALM_PENALTY_INIT": config.alm_penalty_init,
        "ALM_PENALTY_SCALE": config.alm_penalty_scale,
        "ALM_PENALTY_MAX": config.alm_penalty_max,
        "ALM_MAX_OUTER_ITERS": config.alm_max_outer_iters,
        "ALM_FEAS_TOL": config.alm_feas_tol,
        "ALM_STATIONARITY_TOL": config.alm_stationarity_tol,
        "ALM_TRUST_RADIUS_INIT": config.alm_trust_radius_init,
        "ALM_TRUST_RADIUS_MIN": config.alm_trust_radius_min,
        "ALM_TRUST_RADIUS_SHRINK": config.alm_trust_radius_shrink,
        "ALM_TRUST_RADIUS_GROW": config.alm_trust_radius_grow,
        "ALM_MAX_INNER_ATTEMPTS": config.alm_max_inner_attempts,
        "ALM_MAX_SUBPROBLEM_CONTINUATIONS": config.alm_max_subproblem_continuations,
        "ALM_DISTANCE_SMOOTHING": config.alm_distance_smoothing,
        "ALM_CURVATURE_SMOOTHING": config.alm_curvature_smoothing,
    }


def _expected_stage2_artifact_metadata(config: Stage2ArtifactConfig) -> dict:
    expected_iota_metadata = {
        "STAGE2_ROOT_FIX_ENABLED": config.stage2_iota_mode != "off",
        "STAGE2_IOTA_MODE": config.stage2_iota_mode,
    }
    if config.stage2_iota_mode != "off":
        expected_iota_metadata.update(
            {
                "STAGE2_IOTA_TARGET": config.stage2_iota_target,
                "STAGE2_IOTA_TOLERANCE": config.stage2_iota_tolerance,
                "STAGE2_IOTA_VOL_TARGET": config.stage2_iota_vol_target,
                "STAGE2_IOTA_CONSTRAINT_WEIGHT": canonical_stage2_iota_constraint_weight(
                    config.stage2_iota_constraint_weight
                ),
                "STAGE2_IOTA_NUM_TF_COILS": config.stage2_iota_num_tf_coils,
                "STAGE2_IOTA_NPHI": config.stage2_iota_nphi,
                "STAGE2_IOTA_NTHETA": config.stage2_iota_ntheta,
                "STAGE2_IOTA_MPOL": config.stage2_iota_mpol,
                "STAGE2_IOTA_NTOR": config.stage2_iota_ntor,
            }
        )
    if config.stage2_iota_mode == "soft":
        expected_iota_metadata["STAGE2_IOTA_WEIGHT"] = config.stage2_iota_weight
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
        **fixed_stage2_artifact_hardware_contract(),
        "LENGTH_TARGET": config.length_target,
        "banana_surf_radius": config.banana_surf_radius,
        "order": config.order,
        "CONSTRAINT_METHOD": config.constraint_method,
        **_expected_stage2_alm_solver_metadata(config),
        **basin_metadata_from_config(config),
        **expected_iota_metadata,
        "init_only": config.init_only,
    }


def _backfill_missing_stage2_alm_solver_metadata(
    stage2_results: dict,
    config: Stage2ArtifactConfig,
) -> dict:
    upgraded_results = dict(stage2_results)
    for key, value in _expected_stage2_alm_solver_metadata(config).items():
        if upgraded_results.get(key) is None:
            upgraded_results[key] = value
    return upgraded_results


def load_validated_stage2_artifact(
    config: Stage2ArtifactConfig,
    *,
    constraint_metadata: dict,
) -> tuple[Path, dict]:
    artifact_path = resolve_stage2_artifact_path(config)
    stage2_results_path, stage2_results = load_stage2_artifact_results(artifact_path)
    stage2_results = upgrade_legacy_stage2_artifact_results(stage2_results)
    stage2_results = _backfill_missing_stage2_alm_solver_metadata(stage2_results, config)
    validate_stage2_artifact_metadata(
        stage2_results_path,
        stage2_results,
        expected_metadata=_expected_stage2_artifact_metadata(config),
        owner_label="run_stage2_alm.py",
        experiment_family="generic Stage 2 ALM",
    )
    if not _artifact_uses_legacy_constraint_metadata(stage2_results):
        validate_stage2_artifact_metadata(
            stage2_results_path,
            stage2_results,
            expected_metadata=_constraint_metadata_expectation(constraint_metadata),
            owner_label="run_stage2_alm.py",
            experiment_family="generic Stage 2 ALM constraint contract",
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
    constraint_metadata: dict | None = None,
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
        "fixed_stage2_hardware_contract": fixed_stage2_clearance_contract(),
    }
    if constraint_metadata is not None:
        summary.update(constraint_metadata)
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
            "secondary_artifact_preserved": stage2_results.get(
                "STAGE2_SECONDARY_ARTIFACT_PRESERVED"
            ),
            "secondary_artifact_path": stage2_results.get("STAGE2_SECONDARY_BS_PATH"),
            "secondary_results_path": stage2_results.get(
                "STAGE2_SECONDARY_RESULTS_PATH"
            ),
            "coil_plasma_min_dist": stage2_results.get("CURVE_SURFACE_MIN_DIST"),
            "coil_plasma_threshold": stage2_results.get(
                "COIL_PLASMA_MIN_DIST_M",
                COIL_PLASMA_MIN_DIST_M,
            ),
            "plasma_vessel_min_dist": stage2_results.get(
                "SURFACE_VESSEL_MIN_DIST",
                stage2_results.get("PLASMA_VESSEL_MIN_DIST"),
            ),
            "plasma_vessel_threshold": stage2_results.get(
                "PLASMA_VESSEL_MIN_DIST_M",
                PLASMA_VESSEL_MIN_DIST_M,
            ),
        }
    )
    summary.update(
        {
            key: stage2_results.get(key)
            for key in _CONSTRAINT_METADATA_KEYS
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_stage2_iota_args(
        stage2_iota_mode=args.stage2_iota_mode,
        stage2_iota_target=args.stage2_iota_target,
        stage2_iota_tolerance=args.stage2_iota_tolerance,
        stage2_iota_vol_target=args.stage2_iota_vol_target,
        stage2_iota_num_tf_coils=args.stage2_iota_num_tf_coils,
        stage2_iota_nphi=args.stage2_iota_nphi,
        stage2_iota_ntheta=args.stage2_iota_ntheta,
        stage2_iota_mpol=args.stage2_iota_mpol,
        stage2_iota_ntor=args.stage2_iota_ntor,
        stage2_iota_weight=1.0,
        constraint_method="alm",
    )
    resolved_spec, resolved_spec_source = resolve_stage2_spec_payload(args)
    config = build_stage2_alm_config(args, resolved_spec=resolved_spec)
    constraint_metadata = build_stage2_constraint_artifacts(
        args=args,
        config=config,
        source_label=resolved_spec_source,
    )
    artifact_path = resolve_stage2_artifact_path(config)
    artifact_reused = artifact_path.exists()
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    command = build_stage2_command(
        config,
        constraint_override_reason=constraint_metadata["OVERRIDE_REASON"],
        constraint_profile_label=constraint_metadata["CONSTRAINT_PROFILE"],
        python_executable=args.python_executable,
    )
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
            constraint_metadata=constraint_metadata,
        )
        write_dry_run_marker(
            output_root,
            summary_path=summary_path,
            runner_label="run_stage2_alm.py",
        )
    else:
        clear_dry_run_marker(output_root)
        # NOTE: artifact_reused was computed before ensure_stage2_artifact.
        # Under concurrent workflows, a concurrent process could create the
        # artifact between the pre-check and the ensure call, causing
        # artifact_reused=False while ensure_stage2_artifact actually reused it.
        # The proper fix requires ensure_stage2_artifact to return a created/reused
        # flag; for now the pre-check is authoritative for the common case.
        artifact_path = ensure_stage2_artifact(
            config,
            constraint_override_reason=constraint_metadata["OVERRIDE_REASON"],
            constraint_profile_label=constraint_metadata["CONSTRAINT_PROFILE"],
            python_executable=args.python_executable,
            timeout_seconds=timeout_or_none(args.stage2_timeout_seconds),
            dry_run=False,
        )
        stage2_results_path, stage2_results = load_validated_stage2_artifact(
            config,
            constraint_metadata=constraint_metadata,
        )
        summary = build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=artifact_path,
            artifact_reused=artifact_reused,
            stage2_results_path=stage2_results_path,
            stage2_results=stage2_results,
            constraint_metadata=constraint_metadata,
        )

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
