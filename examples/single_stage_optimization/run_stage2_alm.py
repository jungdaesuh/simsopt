import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    BANANA_WINDING_MINOR_RADIUS_M,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    TF_CURRENT_HARD_LIMIT_A,
    fixed_stage2_clearance_contract,
    validate_banana_winding_surface_radius,
    validate_tf_current_limit,
)

SOLVER_SCRIPT = SCRIPT_DIR / "STAGE_2" / "banana_coil_solver.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_stage2_alm"
DEFAULT_SUMMARY_JSON = "stage2_alm_summary.json"
_BASE_STAGE2_PROFILE = {
    "major_radius": 0.915,
    "toroidal_flux": 0.24,
    "length_weight": 0.0005,
    "length_target": COIL_LENGTH_TARGET_M,
    "cc_weight": 100.0,
    "cc_threshold": COIL_COIL_MIN_DIST_M,
    "curvature_weight": 0.0001,
    "curvature_threshold": MAX_CURVATURE_INV_M,
    "banana_surf_radius": BANANA_WINDING_MINOR_RADIUS_M,
    "order": 2,
    "banana_init_current_A": 1.0e4,
    "banana_current_max_A": BANANA_CURRENT_HARD_LIMIT_A,
    "backend": "cpu",
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
    "alm_taylor_test": False,
    "alm_taylor_test_seed": 1,
    "init_only": False,
}
DEFAULT_STAGE2_PROFILES = {
    "standard_80ka": {
        **_BASE_STAGE2_PROFILE,
        "tf_current_A": TF_CURRENT_HARD_LIMIT_A,
    },
}
STAGE2_SPEC_KEYS = tuple(_BASE_STAGE2_PROFILE.keys()) + ("tf_current_A",)
OPTIONAL_STAGE2_SPEC_KEYS = (
    "length_target",
    "backend",
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
    "alm_taylor_test",
    "alm_taylor_test_seed",
    "init_only",
)
REQUIRED_STAGE2_SPEC_KEYS = tuple(
    key for key in STAGE2_SPEC_KEYS if key not in OPTIONAL_STAGE2_SPEC_KEYS
)
_STAGE2_SPEC_KEY_SET = frozenset(STAGE2_SPEC_KEYS)
_OPTIONAL_STAGE2_ARTIFACT_METADATA_KEYS = (
    "LENGTH_TARGET",
    "COIL_PLASMA_MIN_DIST_M",
    "PLASMA_VESSEL_MIN_DIST_M",
    "ALM_FEAS_TOL",
    "ALM_STATIONARITY_TOL",
    "ALM_TRUST_RADIUS_INIT",
    "ALM_TRUST_RADIUS_MIN",
    "ALM_TRUST_RADIUS_SHRINK",
    "ALM_TRUST_RADIUS_GROW",
    "ALM_MAX_INNER_ATTEMPTS",
    "ALM_MAX_SUBPROBLEM_CONTINUATIONS",
    "ALM_DISTANCE_SMOOTHING",
    "ALM_CURVATURE_SMOOTHING",
    "ALM_TAYLOR_TEST_ENABLED",
    "ALM_TAYLOR_TEST_SEED",
)


@dataclass(frozen=True)
class Stage2AlmConfig:
    plasma_surf_filename: str
    output_root: Path
    equilibria_dir: str | None
    major_radius: float
    toroidal_flux: float
    length_weight: float
    length_target: float
    cc_weight: float
    cc_threshold: float
    curvature_weight: float
    curvature_threshold: float
    banana_surf_radius: float
    order: int
    tf_current_A: float
    banana_init_current_A: float
    banana_current_max_A: float
    backend: str
    constraint_method: str
    alm_max_outer_iters: int
    alm_penalty_init: float
    alm_penalty_scale: float
    alm_penalty_max: float
    alm_feas_tol: float
    alm_stationarity_tol: float
    alm_trust_radius_init: float
    alm_trust_radius_min: float
    alm_trust_radius_shrink: float
    alm_trust_radius_grow: float
    alm_max_inner_attempts: int
    alm_max_subproblem_continuations: int
    alm_distance_smoothing: float
    alm_curvature_smoothing: float
    alm_taylor_test: bool
    alm_taylor_test_seed: int
    init_only: bool


def _resolved_path(path_value: str | Path) -> Path:
    return Path(path_value).expanduser().resolve()


def _resolved_optional_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    return _resolved_path(path_value)


def _jsonable_stage2_config(config: Stage2AlmConfig) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 2 ALM wrapper against the live simsopt-jax Stage 2 solver."
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
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--stage2-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--backend", choices=["cpu", "jax"], default=None)
    parser.add_argument("--toroidal-flux", type=float, default=None)
    parser.add_argument("--cc-threshold", type=float, default=None)
    parser.add_argument("--curvature-threshold", type=float, default=None)
    parser.add_argument("--order", type=int, default=None)
    parser.add_argument("--tf-current-A", type=float, default=None)
    return parser.parse_args()


def _load_stage2_spec_json(spec_json_path: str | Path) -> tuple[Path, dict[str, object]]:
    spec_path = _resolved_path(spec_json_path)
    with spec_path.open("r", encoding="utf-8") as infile:
        loaded = json.load(infile)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Stage 2 spec JSON must contain an object at the top level: {spec_path}"
        )
    unknown_keys = sorted(set(loaded) - _STAGE2_SPEC_KEY_SET)
    if unknown_keys:
        raise ValueError(
            f"Unknown Stage 2 spec keys in {spec_path}: {', '.join(unknown_keys)}"
        )
    missing_keys = [key for key in REQUIRED_STAGE2_SPEC_KEYS if key not in loaded]
    if missing_keys:
        raise ValueError(
            f"Stage 2 spec JSON must define all required keys: {', '.join(missing_keys)}"
        )
    return spec_path, {
        key: loaded[key] if key in loaded else _BASE_STAGE2_PROFILE[key]
        for key in STAGE2_SPEC_KEYS
    }


def resolve_stage2_spec_payload(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    if args.profile is not None:
        resolved_spec = dict(DEFAULT_STAGE2_PROFILES[args.profile])
        source_label = f"profile:{args.profile}"
    else:
        spec_path, resolved_spec = _load_stage2_spec_json(args.stage2_spec_json)
        source_label = f"json:{spec_path}"

    overrides = {
        "backend": args.backend,
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
    resolved_spec: dict[str, object],
) -> Stage2AlmConfig:
    cc_threshold = max(float(resolved_spec["cc_threshold"]), COIL_COIL_MIN_DIST_M)
    curvature_threshold = min(
        float(resolved_spec["curvature_threshold"]),
        MAX_CURVATURE_INV_M,
    )
    return Stage2AlmConfig(
        plasma_surf_filename=Path(args.plasma_surf_filename).name,
        output_root=_resolved_path(args.output_root),
        equilibria_dir=(
            None
            if args.equilibria_dir is None
            else str(_resolved_path(args.equilibria_dir))
        ),
        major_radius=float(resolved_spec["major_radius"]),
        toroidal_flux=float(resolved_spec["toroidal_flux"]),
        length_weight=float(resolved_spec["length_weight"]),
        length_target=min(float(resolved_spec["length_target"]), COIL_LENGTH_TARGET_M),
        cc_weight=float(resolved_spec["cc_weight"]),
        cc_threshold=cc_threshold,
        curvature_weight=float(resolved_spec["curvature_weight"]),
        curvature_threshold=curvature_threshold,
        banana_surf_radius=validate_banana_winding_surface_radius(
            float(resolved_spec["banana_surf_radius"])
        ),
        order=int(resolved_spec["order"]),
        tf_current_A=validate_tf_current_limit(float(resolved_spec["tf_current_A"])),
        banana_init_current_A=float(resolved_spec["banana_init_current_A"]),
        banana_current_max_A=float(resolved_spec["banana_current_max_A"]),
        backend=str(resolved_spec["backend"]),
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
        alm_taylor_test=bool(resolved_spec["alm_taylor_test"]),
        alm_taylor_test_seed=int(resolved_spec["alm_taylor_test_seed"]),
        init_only=bool(resolved_spec["init_only"]),
    )


def build_stage2_command(
    config: Stage2AlmConfig,
    *,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        str(SOLVER_SCRIPT),
        "--plasma-surf-filename",
        config.plasma_surf_filename,
        "--output-root",
        str(config.output_root),
        "--backend",
        config.backend,
        "--constraint-method",
        config.constraint_method,
        "--major-radius",
        f"{config.major_radius}",
        "--toroidal-flux",
        f"{config.toroidal_flux}",
        "--length-weight",
        f"{config.length_weight}",
        "--length-target",
        f"{config.length_target}",
        "--cc-weight",
        f"{config.cc_weight}",
        "--cc-threshold",
        f"{config.cc_threshold}",
        "--curvature-weight",
        f"{config.curvature_weight}",
        "--curvature-threshold",
        f"{config.curvature_threshold}",
        "--banana-surf-radius",
        f"{config.banana_surf_radius}",
        "--order",
        str(config.order),
        "--tf-current-A",
        f"{config.tf_current_A}",
        "--banana-init-current-A",
        f"{config.banana_init_current_A}",
        "--banana-current-max-A",
        f"{config.banana_current_max_A}",
        "--alm-max-outer-iters",
        str(config.alm_max_outer_iters),
        "--alm-penalty-init",
        f"{config.alm_penalty_init}",
        "--alm-penalty-scale",
        f"{config.alm_penalty_scale}",
        "--alm-penalty-max",
        f"{config.alm_penalty_max}",
        "--alm-feas-tol",
        f"{config.alm_feas_tol}",
        "--alm-stationarity-tol",
        f"{config.alm_stationarity_tol}",
        "--alm-trust-radius-init",
        f"{config.alm_trust_radius_init}",
        "--alm-trust-radius-min",
        f"{config.alm_trust_radius_min}",
        "--alm-trust-radius-shrink",
        f"{config.alm_trust_radius_shrink}",
        "--alm-trust-radius-grow",
        f"{config.alm_trust_radius_grow}",
        "--alm-max-inner-attempts",
        str(config.alm_max_inner_attempts),
        "--alm-max-subproblem-continuations",
        str(config.alm_max_subproblem_continuations),
        "--alm-distance-smoothing",
        f"{config.alm_distance_smoothing}",
        "--alm-curvature-smoothing",
        f"{config.alm_curvature_smoothing}",
    ]
    if config.equilibria_dir is not None:
        command.extend(["--equilibria-dir", config.equilibria_dir])
    if config.alm_taylor_test:
        command.append("--alm-taylor-test")
        command.extend(["--alm-taylor-test-seed", str(config.alm_taylor_test_seed)])
    if config.init_only:
        command.append("--init-only")
    return command


def _candidate_results_paths(config: Stage2AlmConfig) -> list[Path]:
    search_root = config.output_root / f"outputs-{config.plasma_surf_filename}"
    if not search_root.exists():
        return []
    return sorted(search_root.glob("**/results.json"))


def _expected_stage2_artifact_metadata(config: Stage2AlmConfig) -> dict[str, object]:
    return {
        "PLASMA_SURF_FILENAME": config.plasma_surf_filename,
        "TF_CURRENT_A": config.tf_current_A,
        "BANANA_INIT_CURRENT_A": config.banana_init_current_A,
        "BANANA_CURRENT_MAX_A": config.banana_current_max_A,
        "MAJOR_RADIUS": config.major_radius,
        "TOROIDAL_FLUX": config.toroidal_flux,
        "LENGTH_WEIGHT": config.length_weight,
        "LENGTH_TARGET": config.length_target,
        "CC_WEIGHT": config.cc_weight,
        "CC_THRESHOLD": config.cc_threshold,
        "CURVATURE_WEIGHT": config.curvature_weight,
        "CURVATURE_THRESHOLD": config.curvature_threshold,
        **fixed_stage2_clearance_contract(),
        "banana_surf_radius": config.banana_surf_radius,
        "order": config.order,
        "CONSTRAINT_METHOD": config.constraint_method,
        "backend": config.backend,
        "ALM_MAX_OUTER_ITERS": config.alm_max_outer_iters,
        "ALM_PENALTY_INIT": config.alm_penalty_init,
        "ALM_PENALTY_SCALE": config.alm_penalty_scale,
        "ALM_PENALTY_MAX": config.alm_penalty_max,
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
        "ALM_TAYLOR_TEST_ENABLED": config.alm_taylor_test,
        "ALM_TAYLOR_TEST_SEED": config.alm_taylor_test_seed,
        "init_only": config.init_only,
    }


def _backfill_optional_stage2_artifact_metadata(
    result: dict[str, object],
    config: Stage2AlmConfig,
) -> dict[str, object]:
    upgraded = dict(result)
    defaults = _expected_stage2_artifact_metadata(config)
    for key in _OPTIONAL_STAGE2_ARTIFACT_METADATA_KEYS:
        if upgraded.get(key) is None:
            upgraded[key] = defaults[key]
    return upgraded


def _metadata_value_matches(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _result_matches_config(result: dict[str, object], config: Stage2AlmConfig) -> bool:
    expected = _expected_stage2_artifact_metadata(config)
    return all(
        _metadata_value_matches(result.get(key), expected_value)
        for key, expected_value in expected.items()
    )


def load_validated_stage2_artifact(config: Stage2AlmConfig) -> tuple[Path, dict[str, object]]:
    for results_path in reversed(_candidate_results_paths(config)):
        with results_path.open("r", encoding="utf-8") as infile:
            loaded = json.load(infile)
        loaded = _backfill_optional_stage2_artifact_metadata(loaded, config)
        if _result_matches_config(loaded, config):
            return results_path, loaded
    raise FileNotFoundError(
        "Could not find a Stage 2 ALM results.json matching the requested config under "
        f"{config.output_root / f'outputs-{config.plasma_surf_filename}'}"
    )


def build_summary(
    args: argparse.Namespace,
    *,
    config: Stage2AlmConfig,
    resolved_spec_source: str,
    command: list[str],
    artifact_path: Path,
    stage2_results_path: Path | None = None,
    stage2_results: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = {
        "plasma_surf_filename": config.plasma_surf_filename,
        "resolved_spec_source": resolved_spec_source,
        "artifact_path": str(artifact_path),
        "command": command,
        "dry_run": bool(args.dry_run),
        "contains_solver_outputs": bool(
            stage2_results_path is not None and stage2_results is not None
        ),
        "resolved_stage2_config": _jsonable_stage2_config(config),
        "fixed_stage2_hardware_contract": {
            **fixed_stage2_clearance_contract(),
            "LENGTH_TARGET": COIL_LENGTH_TARGET_M,
        },
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
            "coil_plasma_min_dist": stage2_results.get("CURVE_SURFACE_MIN_DIST"),
            "coil_plasma_threshold": stage2_results.get(
                "COIL_PLASMA_MIN_DIST_M",
                COIL_PLASMA_MIN_DIST_M,
            ),
            "plasma_vessel_min_dist": stage2_results.get("SURFACE_VESSEL_MIN_DIST"),
            "plasma_vessel_threshold": stage2_results.get(
                "PLASMA_VESSEL_MIN_DIST_M",
                PLASMA_VESSEL_MIN_DIST_M,
            ),
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
    config.output_root.mkdir(parents=True, exist_ok=True)
    command = build_stage2_command(config, python_executable=args.python_executable)

    summary_path = _resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = config.output_root / DEFAULT_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_root = config.output_root / f"outputs-{config.plasma_surf_filename}"
    if args.dry_run:
        summary = build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=artifact_root,
        )
    else:
        subprocess.run(
            command,
            check=True,
            timeout=None if args.stage2_timeout_seconds <= 0 else args.stage2_timeout_seconds,
        )
        stage2_results_path, stage2_results = load_validated_stage2_artifact(config)
        summary = build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=stage2_results_path.parent,
            stage2_results_path=stage2_results_path,
            stage2_results=stage2_results,
        )

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
