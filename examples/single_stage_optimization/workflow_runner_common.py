from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_helpers import Stage2SeedSpec, local_stage2_bs_path

STAGE2_SCRIPT_PATH = SCRIPT_DIR / "STAGE_2" / "banana_coil_solver.py"
SINGLE_STAGE_SCRIPT_PATH = SCRIPT_DIR / "SINGLE_STAGE" / "single_stage_banana_example.py"

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
    basin_seed: int | None
    init_only: bool


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
    )


def resolve_stage2_artifact_path(config: Stage2ArtifactConfig) -> Path:
    return local_stage2_bs_path(
        config.output_root,
        build_stage2_seed_spec(config),
        constraint_method=config.constraint_method,
        alm_max_outer_iters=config.alm_max_outer_iters,
        alm_penalty_init=config.alm_penalty_init,
        alm_penalty_scale=config.alm_penalty_scale,
        basin_hops=config.basin_hops,
        basin_stepsize=config.basin_stepsize,
        basin_seed=config.basin_seed,
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
        "--order",
        str(config.order),
        "--constraint-method",
        config.constraint_method,
    ]
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
            ]
        )
    if config.basin_hops > 0:
        command.extend(
            [
                "--basin-hops",
                str(config.basin_hops),
                "--basin-stepsize",
                str(config.basin_stepsize),
            ]
        )
        if config.basin_seed is not None:
            command.extend(["--basin-seed", str(config.basin_seed)])
    if config.init_only:
        command.append("--init-only")
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
    return stage2_results_path, load_json(stage2_results_path)


def _single_results_matches(output_root: str | Path) -> list[Path]:
    return sorted(Path(output_root).glob("mpol=*-ntor=*/results.json"))


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


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)
