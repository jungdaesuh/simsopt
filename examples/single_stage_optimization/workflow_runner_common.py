from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_helpers import (
    Stage2SeedSpec,
    local_stage2_bs_path,
    resolve_wataru_vf_template_path,
    validate_normalized_toroidal_flux,
)
from banana_opt.artifact_contracts import (
    STAGE2_BS_SHA256_KEY,
    compute_stage2_bs_sha256,
    upgrade_legacy_stage2_artifact_results,
)

STAGE2_SCRIPT_PATH = SCRIPT_DIR / "STAGE_2" / "banana_coil_solver.py"
SINGLE_STAGE_SCRIPT_PATH = SCRIPT_DIR / "SINGLE_STAGE" / "single_stage_banana_example.py"
POINCARE_SCRIPT_PATH = SCRIPT_DIR / "POINCARE_PLOTTING" / "poincare_surfaces.py"
DRY_RUN_MARKER_FILENAME = "DRY_RUN_ONLY.txt"
STAGE2_SIDECAR_REQUIRED_ERROR = (
    "Stage 2 restarts require the sibling results.json sidecar so the "
    "loaded coils can be partitioned via the coil_groups manifest."
)

T = TypeVar("T")


@dataclass(frozen=True)
class Stage2ArtifactConfig:
    plasma_surf_filename: str
    output_root: Path
    equilibria_dir: str | None
    tf_current_A: float
    major_radius: float
    toroidal_flux: float
    length_weight: float
    cc_weight: float
    cc_threshold: float
    curvature_weight: float
    curvature_threshold: float
    banana_surf_radius: float
    order: int
    constraint_method: str
    alm_max_outer_iters: int
    alm_penalty_init: float
    alm_penalty_scale: float
    basin_hops: int
    basin_stepsize: float
    alm_penalty_max: float = 1.0e8
    alm_feas_tol: float = 1e-6
    alm_stationarity_tol: float = 1e-6
    alm_trust_radius_init: float = 0.05
    alm_trust_radius_min: float = 1e-4
    alm_trust_radius_shrink: float = 0.5
    alm_trust_radius_grow: float = 1.5
    alm_max_inner_attempts: int = 4
    alm_max_subproblem_continuations: int = 20
    alm_distance_smoothing: float = 0.005
    alm_curvature_smoothing: float = 0.25
    basin_temperature: float = 1.0
    basin_niter_success: int = 0
    basin_seed: int | None = None
    init_only: bool = False
    banana_init_current_A: float = 1.0e4
    banana_current_max_A: float = 1.6e4
    finite_current_mode: str = "wataru_proxy_field"
    proxy_plasma_current_A: float = 0.0
    vf_current_A: float = 0.0
    vf_template_path: str | None = None
    stage2_iota_mode: str = "off"
    stage2_iota_target: float | None = None
    stage2_iota_tolerance: float = 5.0e-3
    stage2_iota_weight: float = 1.0
    stage2_iota_vol_target: float = 0.10
    stage2_iota_constraint_weight: float = 1.0
    stage2_iota_num_tf_coils: int = 20
    stage2_iota_nphi: int = 91
    stage2_iota_ntheta: int = 32
    stage2_iota_mpol: int = 8
    stage2_iota_ntor: int = 6

    def __post_init__(self) -> None:
        validate_normalized_toroidal_flux(
            self.toroidal_flux,
            field_name="Stage2ArtifactConfig.toroidal_flux",
        )
        object.__setattr__(
            self,
            "vf_template_path",
            resolve_wataru_vf_template_path(self.vf_template_path),
        )
        if self.stage2_iota_mode != "off" and self.stage2_iota_target is None:
            raise ValueError(
                "Stage2ArtifactConfig.stage2_iota_target is required when "
                "stage2_iota_mode is enabled."
            )
        if self.stage2_iota_mode == "soft" and self.stage2_iota_weight <= 0.0:
            raise ValueError(
                "Stage2ArtifactConfig.stage2_iota_weight must be positive in soft mode."
            )
        if self.stage2_iota_mode == "soft" and self.constraint_method == "alm":
            raise ValueError(
                "Stage2ArtifactConfig.stage2_iota_mode='soft' is incompatible with "
                "constraint_method='alm'."
            )
        if self.stage2_iota_mode == "alm" and self.constraint_method != "alm":
            raise ValueError(
                "Stage2ArtifactConfig.stage2_iota_mode='alm' requires "
                "constraint_method='alm'."
            )


def parse_csv(raw: str, cast: Callable[[str], T]) -> list[T]:
    values = [segment.strip() for segment in raw.split(",") if segment.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return [cast(value) for value in values]


def build_stage2_seed_spec(config: Stage2ArtifactConfig) -> Stage2SeedSpec:
    return Stage2SeedSpec(
        plasma_surf_filename=config.plasma_surf_filename,
        major_radius=config.major_radius,
        toroidal_flux=config.toroidal_flux,
        length_weight=config.length_weight,
        cc_weight=config.cc_weight,
        cc_threshold=config.cc_threshold,
        curvature_weight=config.curvature_weight,
        curvature_threshold=config.curvature_threshold,
        banana_surf_radius=config.banana_surf_radius,
        tf_current_A=config.tf_current_A,
        order=config.order,
        banana_init_current_A=config.banana_init_current_A,
        banana_current_max_A=config.banana_current_max_A,
        finite_current_mode=config.finite_current_mode,
        proxy_plasma_current_A=config.proxy_plasma_current_A,
        vf_current_A=config.vf_current_A,
        vf_template_path=config.vf_template_path,
    )


def resolve_stage2_artifact_path(config: Stage2ArtifactConfig) -> Path:
    return local_stage2_bs_path(
        config.output_root,
        build_stage2_seed_spec(config),
        constraint_method=config.constraint_method,
        alm_max_outer_iters=config.alm_max_outer_iters,
        alm_penalty_init=config.alm_penalty_init,
        alm_penalty_scale=config.alm_penalty_scale,
        alm_penalty_max=config.alm_penalty_max,
        alm_max_subproblem_continuations=config.alm_max_subproblem_continuations,
        alm_feas_tol=config.alm_feas_tol,
        alm_stationarity_tol=config.alm_stationarity_tol,
        alm_trust_radius_init=config.alm_trust_radius_init,
        alm_trust_radius_min=config.alm_trust_radius_min,
        alm_trust_radius_shrink=config.alm_trust_radius_shrink,
        alm_trust_radius_grow=config.alm_trust_radius_grow,
        alm_max_inner_attempts=config.alm_max_inner_attempts,
        alm_distance_smoothing=config.alm_distance_smoothing,
        alm_curvature_smoothing=config.alm_curvature_smoothing,
        basin_hops=config.basin_hops,
        basin_stepsize=config.basin_stepsize,
        basin_temperature=config.basin_temperature,
        basin_niter_success=config.basin_niter_success,
        basin_seed=config.basin_seed,
        stage2_iota_mode=config.stage2_iota_mode,
        stage2_iota_target=config.stage2_iota_target,
        stage2_iota_tolerance=config.stage2_iota_tolerance,
        stage2_iota_weight=config.stage2_iota_weight,
        stage2_iota_vol_target=config.stage2_iota_vol_target,
        stage2_iota_constraint_weight=config.stage2_iota_constraint_weight,
        stage2_iota_num_tf_coils=config.stage2_iota_num_tf_coils,
        stage2_iota_nphi=config.stage2_iota_nphi,
        stage2_iota_ntheta=config.stage2_iota_ntheta,
        stage2_iota_mpol=config.stage2_iota_mpol,
        stage2_iota_ntor=config.stage2_iota_ntor,
    )


def build_stage2_command(
    config: Stage2ArtifactConfig,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    command = [
        python_executable,
        str(STAGE2_SCRIPT_PATH),
        "--plasma-surf-filename",
        config.plasma_surf_filename,
        "--output-root",
        str(config.output_root),
        "--tf-current-A",
        str(config.tf_current_A),
        "--major-radius",
        str(config.major_radius),
        "--toroidal-flux",
        str(config.toroidal_flux),
        "--length-weight",
        str(config.length_weight),
        "--cc-weight",
        str(config.cc_weight),
        "--cc-threshold",
        str(config.cc_threshold),
        "--curvature-weight",
        str(config.curvature_weight),
        "--curvature-threshold",
        str(config.curvature_threshold),
        "--banana-surf-radius",
        str(config.banana_surf_radius),
        "--banana-init-current-A",
        str(config.banana_init_current_A),
        "--banana-current-max-A",
        str(config.banana_current_max_A),
        "--order",
        str(config.order),
        "--constraint-method",
        config.constraint_method,
    ]
    if config.finite_current_mode not in {None, ""}:
        command.extend(["--finite-current-mode", config.finite_current_mode])
    if abs(float(config.proxy_plasma_current_A)) > 1.0e-12:
        command.extend(
            [
                "--proxy-plasma-current-A",
                str(config.proxy_plasma_current_A),
            ]
        )
    if abs(float(config.vf_current_A)) > 1.0e-12:
        command.extend(["--vf-current-A", str(config.vf_current_A)])
    if config.vf_template_path not in {None, ""}:
        command.extend(["--vf-template-path", str(config.vf_template_path)])
    if config.equilibria_dir is not None:
        command.extend(["--equilibria-dir", config.equilibria_dir])
    if config.constraint_method == "alm":
        command.extend(
            [
                "--alm-max-outer-iters",
                str(config.alm_max_outer_iters),
                "--alm-penalty-init",
                str(config.alm_penalty_init),
                "--alm-penalty-scale",
                str(config.alm_penalty_scale),
                "--alm-penalty-max",
                str(config.alm_penalty_max),
                "--alm-feas-tol",
                str(config.alm_feas_tol),
                "--alm-stationarity-tol",
                str(config.alm_stationarity_tol),
                "--alm-trust-radius-init",
                str(config.alm_trust_radius_init),
                "--alm-trust-radius-min",
                str(config.alm_trust_radius_min),
                "--alm-trust-radius-shrink",
                str(config.alm_trust_radius_shrink),
                "--alm-trust-radius-grow",
                str(config.alm_trust_radius_grow),
                "--alm-max-inner-attempts",
                str(config.alm_max_inner_attempts),
                "--alm-max-subproblem-continuations",
                str(config.alm_max_subproblem_continuations),
                "--alm-distance-smoothing",
                str(config.alm_distance_smoothing),
                "--alm-curvature-smoothing",
                str(config.alm_curvature_smoothing),
            ]
        )
    if config.basin_hops > 0:
        command.extend(
            [
                "--basin-hops",
                str(config.basin_hops),
                "--basin-stepsize",
                str(config.basin_stepsize),
                "--basin-temperature",
                str(config.basin_temperature),
            ]
        )
        if config.basin_niter_success > 0:
            command.extend(["--basin-niter-success", str(config.basin_niter_success)])
        if config.basin_seed is not None:
            command.extend(["--basin-seed", str(config.basin_seed)])
    if config.init_only:
        command.append("--init-only")
    if config.stage2_iota_mode != "off":
        command.extend(
            [
                "--stage2-iota-mode",
                config.stage2_iota_mode,
                "--stage2-iota-target",
                str(config.stage2_iota_target),
                "--stage2-iota-tolerance",
                str(config.stage2_iota_tolerance),
                "--stage2-iota-vol-target",
                str(config.stage2_iota_vol_target),
                "--stage2-iota-constraint-weight",
                str(config.stage2_iota_constraint_weight),
                "--stage2-iota-num-tf-coils",
                str(config.stage2_iota_num_tf_coils),
                "--stage2-iota-nphi",
                str(config.stage2_iota_nphi),
                "--stage2-iota-ntheta",
                str(config.stage2_iota_ntheta),
                "--stage2-iota-mpol",
                str(config.stage2_iota_mpol),
                "--stage2-iota-ntor",
                str(config.stage2_iota_ntor),
            ]
        )
        if config.stage2_iota_mode == "soft":
            command.extend(
                [
                    "--stage2-iota-weight",
                    str(config.stage2_iota_weight),
                ]
            )
    return command


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


def ensure_stage2_artifact(
    config: Stage2ArtifactConfig,
    *,
    python_executable: str = sys.executable,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
) -> Path:
    artifact_path = resolve_stage2_artifact_path(config)
    if artifact_path.exists() or dry_run:
        return artifact_path
    run_command(
        build_stage2_command(config, python_executable=python_executable),
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
    )
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Expected Stage 2 artifact was not created: {artifact_path}"
        )
    return artifact_path


def load_stage2_artifact_results(stage2_bs_path: str | Path) -> tuple[Path, dict]:
    stage2_bs_path = Path(stage2_bs_path)
    stage2_results_path = stage2_bs_path.with_name("results.json")
    if not stage2_results_path.is_file():
        raise ValueError(STAGE2_SIDECAR_REQUIRED_ERROR)
    stage2_results = load_json(stage2_results_path)
    recorded_digest = stage2_results.get(STAGE2_BS_SHA256_KEY)
    if recorded_digest in {None, ""}:
        warnings.warn(
            "Stage 2 artifact results.json is missing STAGE2_BS_SHA256; "
            "allowing legacy artifact without checksum binding.",
            RuntimeWarning,
            stacklevel=2,
        )
        return stage2_results_path, stage2_results
    actual_digest = compute_stage2_bs_sha256(stage2_bs_path)
    if str(recorded_digest) != actual_digest:
        raise ValueError(
            "Stage 2 artifact checksum mismatch: "
            f"{stage2_results_path} reports {STAGE2_BS_SHA256_KEY}={recorded_digest!r}, "
            f"but {stage2_bs_path} hashes to {actual_digest!r}."
        )
    return stage2_results_path, stage2_results


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


def _single_solver_checkpoint_matches(output_root: str | Path) -> list[Path]:
    return sorted(
        Path(output_root).glob("mpol=*-ntor=*/solver_state_checkpoint.json")
    )


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
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one single-stage results.json under {output_root}, found {len(matches)}"
        )
    return matches[0]


def discover_single_solver_checkpoint_path(output_root: str | Path) -> Path:
    matches = _single_solver_checkpoint_matches(output_root)
    if not matches:
        raise FileNotFoundError(
            f"Expected at least one single-stage solver_state_checkpoint.json under {output_root}, found 0"
        )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one single-stage solver_state_checkpoint.json under "
            f"{output_root}, found {len(matches)}"
        )
    return matches[0]


def dry_run_marker_path(output_root: str | Path) -> Path:
    return Path(output_root) / DRY_RUN_MARKER_FILENAME


def write_dry_run_marker(
    output_root: str | Path,
    *,
    summary_path: str | Path,
    runner_label: str,
) -> Path:
    marker_path = dry_run_marker_path(output_root)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        (
            f"{runner_label} dry run only.\n"
            "No solver outputs were materialized in this directory.\n"
            f"See the summary JSON for the planned command and resolved inputs: {Path(summary_path)}\n"
        ),
        encoding="utf-8",
    )
    return marker_path


def clear_dry_run_marker(output_root: str | Path) -> None:
    marker_path = dry_run_marker_path(output_root)
    if marker_path.exists():
        marker_path.unlink()


def _json_portable_value(payload: object) -> object:
    if isinstance(payload, float):
        if not math.isfinite(payload):
            return None
        return payload
    if isinstance(payload, Mapping):
        return {
            key: _json_portable_value(value)
            for key, value in payload.items()
        }
    if isinstance(payload, Sequence) and not isinstance(
        payload,
        (str, bytes, bytearray),
    ):
        return [_json_portable_value(value) for value in payload]
    return payload


def write_json(path: str | Path, payload: object) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as outfile:
        json.dump(_json_portable_value(payload), outfile, indent=2, allow_nan=False)


def write_csv_rows(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str],
) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    fieldname: row.get(fieldname)
                    for fieldname in fieldnames
                }
            )


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


def resolved_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def resolved_optional_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    return resolved_path(raw_path)


def load_validated_stage2_seed_results(
    args: object,
    *,
    owner_label: str,
    stage2_bs_path: str | Path | None = None,
) -> tuple[Path, Path, dict]:
    resolved_stage2_bs_path = (
        resolved_path(getattr(args, "stage2_bs_path"))
        if stage2_bs_path is None
        else resolved_path(stage2_bs_path)
    )
    stage2_results_path, stage2_results = load_stage2_artifact_results(
        resolved_stage2_bs_path
    )
    stage2_results = upgrade_legacy_stage2_artifact_results(
        stage2_results,
        known_num_tf_coils=getattr(args, "num_tf_coils", None),
        known_tf_current_A=getattr(args, "stage2_seed_tf_current_A", None),
    )
    actual_surface = stage2_results.get("PLASMA_SURF_FILENAME")
    expected_surface = Path(getattr(args, "plasma_surf_filename")).name
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
        owner_label=owner_label,
        allow_init_only=bool(
            getattr(args, "allow_init_only_stage2_seed", False)
        ),
    )
    return resolved_stage2_bs_path, stage2_results_path, stage2_results


def maybe_load_validated_stage2_seed_results(
    args: object,
    *,
    owner_label: str,
) -> tuple[Path, Path | None, dict | None]:
    stage2_bs_path = resolved_path(getattr(args, "stage2_bs_path"))
    stage2_results_path = stage2_bs_path.with_name("results.json")
    if not stage2_bs_path.exists() or not stage2_results_path.exists():
        return stage2_bs_path, None, None
    return load_validated_stage2_seed_results(
        args,
        owner_label=owner_label,
        stage2_bs_path=stage2_bs_path,
    )


def append_optional_flag(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def append_bool_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def append_single_stage_handoff_flags(command: list[str], args: object) -> None:
    equilibrium_path = resolved_optional_path(getattr(args, "equilibrium_path", None))
    if equilibrium_path is not None:
        command.extend(["--equilibrium-path", str(equilibrium_path)])
    append_optional_flag(
        command,
        "--constraint-weight",
        getattr(args, "constraint_weight", None),
    )
    append_optional_flag(command, "--num-tf-coils", getattr(args, "num_tf_coils", None))
    append_optional_flag(
        command,
        "--stage2-seed-tf-current-A",
        getattr(args, "stage2_seed_tf_current_A", None),
    )
    append_optional_flag(command, "--boozer-I", getattr(args, "boozer_I", None))
    append_optional_flag(
        command,
        "--plasma-current-A",
        getattr(args, "plasma_current_A", None),
    )
    append_optional_flag(
        command,
        "--banana-surf-radius",
        getattr(args, "banana_surf_radius", None),
    )
    append_bool_flag(
        command,
        "--allow-init-only-stage2-seed",
        bool(getattr(args, "allow_init_only_stage2_seed", False)),
    )


def run_poincare_artifact(
    *,
    output_dir: str | Path,
    python_executable: str = sys.executable,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
) -> list[str]:
    command = [python_executable, str(POINCARE_SCRIPT_PATH)]
    if dry_run:
        return command
    env = os.environ.copy()
    env["POINCARE_OUT_DIR"] = str(resolved_path(output_dir))
    subprocess.run(
        command,
        cwd=str(SCRIPT_DIR),
        check=True,
        timeout=timeout_seconds,
        env=env,
    )
    return command


def timeout_or_none(timeout_seconds: float) -> float | None:
    return None if timeout_seconds <= 0.0 else float(timeout_seconds)
