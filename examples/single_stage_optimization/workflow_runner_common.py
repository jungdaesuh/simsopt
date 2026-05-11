from __future__ import annotations

import csv
import fcntl
import inspect
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_helpers import (
    DEFAULT_STAGE2_LENGTH_TARGET,
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
from banana_opt.constraint_contract import resolve_constraint_contract_from_wire_names
from banana_opt.hardware_contracts import validate_major_radius

STAGE2_SCRIPT_PATH = SCRIPT_DIR / "STAGE_2" / "banana_coil_solver.py"
SINGLE_STAGE_SCRIPT_PATH = SCRIPT_DIR / "SINGLE_STAGE" / "single_stage_banana_example.py"
POINCARE_SCRIPT_PATH = SCRIPT_DIR / "POINCARE_PLOTTING" / "poincare_surfaces.py"
DRY_RUN_MARKER_FILENAME = "DRY_RUN_ONLY.txt"
STAGE2_SIDECAR_REQUIRED_ERROR = (
    "Stage 2 restarts require the sibling results.json sidecar so the "
    "loaded coils can be partitioned via the coil_groups manifest."
)

T = TypeVar("T")
# Single-stage and baseline-sweep ALM defaults. Stage 2 intentionally keeps a
# wider curvature smoothing default on Stage2ArtifactConfig.
SINGLE_STAGE_ALM_CLI_FIELDS = (
    ("max_outer_iters", int, 10),
    ("penalty_init", float, 1.0),
    ("penalty_scale", float, 10.0),
    ("penalty_max", float, 1.0e8),
    ("feas_tol", float, 1.0e-6),
    ("stationarity_tol", float, 1.0e-6),
    ("trust_radius_init", float, 0.05),
    ("trust_radius_min", float, 1.0e-4),
    ("trust_radius_shrink", float, 0.5),
    ("trust_radius_grow", float, 1.5),
    ("max_inner_attempts", int, 4),
    ("max_subproblem_continuations", int, 20),
    ("distance_smoothing", float, 0.005),
    ("curvature_smoothing", float, 0.05),
)
STAGE2_ALM_DEFAULT_OVERRIDES = {
    "curvature_smoothing": 0.25,
}
STAGE2_ALM_CLI_FIELDS = tuple(
    (
        suffix,
        value_type,
        STAGE2_ALM_DEFAULT_OVERRIDES.get(suffix, default),
    )
    for suffix, value_type, default in SINGLE_STAGE_ALM_CLI_FIELDS
)
STAGE2_ALM_DEFAULTS = {
    suffix: default for suffix, _value_type, default in STAGE2_ALM_CLI_FIELDS
}
STAGE2_ARTIFACT_PATH_FIELD_NAMES = tuple(
    name
    for name, parameter in inspect.signature(local_stage2_bs_path).parameters.items()
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY
)


def stage2_alm_default(suffix: str) -> int | float:
    return STAGE2_ALM_DEFAULTS[suffix]


def alm_flag(suffix: str) -> str:
    return f"--alm-{suffix.replace('_', '-')}"


def single_stage_alm_flag(suffix: str) -> str:
    return f"--single-stage-alm-{suffix.replace('_', '-')}"


def validate_constraint_cli_overrides(overrides: Mapping[str, float | None]) -> None:
    resolve_constraint_contract_from_wire_names(cli_overrides=overrides)


def append_alm_cli_flags(
    command: list[str],
    source: object,
    *,
    attr_prefix: str = "alm_",
    cli_fields: Sequence[tuple[str, type, int | float]] = SINGLE_STAGE_ALM_CLI_FIELDS,
) -> None:
    for suffix, _value_type, _default in cli_fields:
        command.extend(
            [alm_flag(suffix), str(getattr(source, f"{attr_prefix}{suffix}"))]
        )


@dataclass(frozen=True)
class Stage2ArtifactEnsureResult:
    artifact_path: Path
    artifact_reused: bool


_STAGE2_ALM_FIELD_NAMES = tuple(
    f"alm_{suffix}" for suffix, _value_type, _default in STAGE2_ALM_CLI_FIELDS
)
STAGE2_ARTIFACT_CONFIG_FLAT_FIELD_NAMES = (
    "plasma_surf_filename",
    "output_root",
    "equilibria_dir",
    "tf_current_A",
    "major_radius",
    "toroidal_flux",
    "length_weight",
    "cc_weight",
    "cc_threshold",
    "curvature_weight",
    "curvature_threshold",
    "banana_surf_radius",
    "order",
    "constraint_method",
    "alm_max_outer_iters",
    "alm_penalty_init",
    "alm_penalty_scale",
    "basin_hops",
    "basin_stepsize",
    *_STAGE2_ALM_FIELD_NAMES[3:],
    "basin_temperature",
    "basin_niter_success",
    "basin_seed",
    "init_only",
    "banana_init_current_A",
    "banana_current_max_A",
    "length_target",
    "finite_current_mode",
    "proxy_plasma_current_A",
    "vf_current_A",
    "vf_template_path",
    "target_lcfs_max_major_radius_m",
    "target_lcfs_max_minor_radius_m",
    "stage2_iota_mode",
    "stage2_iota_target",
    "stage2_iota_tolerance",
    "stage2_iota_weight",
    "stage2_iota_vol_target",
    "stage2_iota_constraint_weight",
    "stage2_iota_num_tf_coils",
    "stage2_iota_nphi",
    "stage2_iota_ntheta",
    "stage2_iota_mpol",
    "stage2_iota_ntor",
)


@dataclass(frozen=True, slots=True)
class Stage2ArtifactIOConfig:
    plasma_surf_filename: str
    output_root: Path
    equilibria_dir: str | None


@dataclass(frozen=True, slots=True)
class Stage2GeometryConfig:
    major_radius: float
    toroidal_flux: float
    banana_surf_radius: float
    order: int

    def __post_init__(self) -> None:
        validate_major_radius(self.major_radius)
        validate_normalized_toroidal_flux(
            self.toroidal_flux,
            field_name="Stage2ArtifactConfig.toroidal_flux",
        )


@dataclass(frozen=True, slots=True)
class Stage2HardwareConfig:
    tf_current_A: float
    banana_init_current_A: float = 1.0e4
    banana_current_max_A: float = 1.6e4


@dataclass(frozen=True, slots=True)
class Stage2ObjectiveWeights:
    length_weight: float
    cc_weight: float
    curvature_weight: float


@dataclass(frozen=True, slots=True)
class Stage2ConstraintPolicy:
    constraint_method: str
    cc_threshold: float
    curvature_threshold: float
    length_target: float = DEFAULT_STAGE2_LENGTH_TARGET
    target_lcfs_max_major_radius_m: float = 0.92
    target_lcfs_max_minor_radius_m: float = 0.15


@dataclass(frozen=True, slots=True)
class Stage2AlmControls:
    alm_max_outer_iters: int
    alm_penalty_init: float
    alm_penalty_scale: float
    alm_penalty_max: float = stage2_alm_default("penalty_max")
    alm_feas_tol: float = stage2_alm_default("feas_tol")
    alm_stationarity_tol: float = stage2_alm_default("stationarity_tol")
    alm_trust_radius_init: float = stage2_alm_default("trust_radius_init")
    alm_trust_radius_min: float = stage2_alm_default("trust_radius_min")
    alm_trust_radius_shrink: float = stage2_alm_default("trust_radius_shrink")
    alm_trust_radius_grow: float = stage2_alm_default("trust_radius_grow")
    alm_max_inner_attempts: int = stage2_alm_default("max_inner_attempts")
    alm_max_subproblem_continuations: int = stage2_alm_default(
        "max_subproblem_continuations"
    )
    alm_distance_smoothing: float = stage2_alm_default("distance_smoothing")
    alm_curvature_smoothing: float = stage2_alm_default("curvature_smoothing")


@dataclass(frozen=True, slots=True)
class Stage2BasinControls:
    basin_hops: int
    basin_stepsize: float
    basin_temperature: float = 1.0
    basin_niter_success: int = 0
    basin_seed: int | None = None
    init_only: bool = False


@dataclass(frozen=True, slots=True)
class Stage2FiniteCurrentConfig:
    finite_current_mode: str = "wataru_proxy_field"
    proxy_plasma_current_A: float = 0.0
    vf_current_A: float = 0.0
    vf_template_path: str | None = None

    @property
    def effective_vf_template_path(self) -> str | None:
        return resolve_wataru_vf_template_path(self.vf_template_path)


@dataclass(frozen=True, slots=True)
class Stage2IotaConfig:
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


@dataclass(frozen=True, slots=True, init=False)
class Stage2ArtifactConfig:
    _io: Stage2ArtifactIOConfig
    _geometry: Stage2GeometryConfig
    _hardware: Stage2HardwareConfig
    _objective_weights: Stage2ObjectiveWeights
    _constraint_policy: Stage2ConstraintPolicy
    _alm: Stage2AlmControls
    _basin: Stage2BasinControls
    _finite_current: Stage2FiniteCurrentConfig
    _iota: Stage2IotaConfig

    def __init__(
        self,
        plasma_surf_filename: str,
        output_root: Path,
        equilibria_dir: str | None,
        tf_current_A: float,
        major_radius: float,
        toroidal_flux: float,
        length_weight: float,
        cc_weight: float,
        cc_threshold: float,
        curvature_weight: float,
        curvature_threshold: float,
        banana_surf_radius: float,
        order: int,
        constraint_method: str,
        alm_max_outer_iters: int,
        alm_penalty_init: float,
        alm_penalty_scale: float,
        basin_hops: int,
        basin_stepsize: float,
        alm_penalty_max: float = stage2_alm_default("penalty_max"),
        alm_feas_tol: float = stage2_alm_default("feas_tol"),
        alm_stationarity_tol: float = stage2_alm_default("stationarity_tol"),
        alm_trust_radius_init: float = stage2_alm_default("trust_radius_init"),
        alm_trust_radius_min: float = stage2_alm_default("trust_radius_min"),
        alm_trust_radius_shrink: float = stage2_alm_default("trust_radius_shrink"),
        alm_trust_radius_grow: float = stage2_alm_default("trust_radius_grow"),
        alm_max_inner_attempts: int = stage2_alm_default("max_inner_attempts"),
        alm_max_subproblem_continuations: int = stage2_alm_default(
            "max_subproblem_continuations"
        ),
        alm_distance_smoothing: float = stage2_alm_default("distance_smoothing"),
        alm_curvature_smoothing: float = stage2_alm_default("curvature_smoothing"),
        basin_temperature: float = 1.0,
        basin_niter_success: int = 0,
        basin_seed: int | None = None,
        init_only: bool = False,
        banana_init_current_A: float = 1.0e4,
        banana_current_max_A: float = 1.6e4,
        length_target: float = DEFAULT_STAGE2_LENGTH_TARGET,
        finite_current_mode: str = "wataru_proxy_field",
        proxy_plasma_current_A: float = 0.0,
        vf_current_A: float = 0.0,
        vf_template_path: str | None = None,
        target_lcfs_max_major_radius_m: float = 0.92,
        target_lcfs_max_minor_radius_m: float = 0.15,
        stage2_iota_mode: str = "off",
        stage2_iota_target: float | None = None,
        stage2_iota_tolerance: float = 5.0e-3,
        stage2_iota_weight: float = 1.0,
        stage2_iota_vol_target: float = 0.10,
        stage2_iota_constraint_weight: float = 1.0,
        stage2_iota_num_tf_coils: int = 20,
        stage2_iota_nphi: int = 91,
        stage2_iota_ntheta: int = 32,
        stage2_iota_mpol: int = 8,
        stage2_iota_ntor: int = 6,
    ) -> None:
        object.__setattr__(
            self,
            "_io",
            Stage2ArtifactIOConfig(
                plasma_surf_filename=plasma_surf_filename,
                output_root=output_root,
                equilibria_dir=equilibria_dir,
            ),
        )
        object.__setattr__(
            self,
            "_geometry",
            Stage2GeometryConfig(
                major_radius=major_radius,
                toroidal_flux=toroidal_flux,
                banana_surf_radius=banana_surf_radius,
                order=order,
            ),
        )
        object.__setattr__(
            self,
            "_hardware",
            Stage2HardwareConfig(
                tf_current_A=tf_current_A,
                banana_init_current_A=banana_init_current_A,
                banana_current_max_A=banana_current_max_A,
            ),
        )
        object.__setattr__(
            self,
            "_objective_weights",
            Stage2ObjectiveWeights(
                length_weight=length_weight,
                cc_weight=cc_weight,
                curvature_weight=curvature_weight,
            ),
        )
        object.__setattr__(
            self,
            "_constraint_policy",
            Stage2ConstraintPolicy(
                constraint_method=constraint_method,
                length_target=length_target,
                cc_threshold=cc_threshold,
                curvature_threshold=curvature_threshold,
                target_lcfs_max_major_radius_m=target_lcfs_max_major_radius_m,
                target_lcfs_max_minor_radius_m=target_lcfs_max_minor_radius_m,
            ),
        )
        object.__setattr__(
            self,
            "_alm",
            Stage2AlmControls(
                alm_max_outer_iters=alm_max_outer_iters,
                alm_penalty_init=alm_penalty_init,
                alm_penalty_scale=alm_penalty_scale,
                alm_penalty_max=alm_penalty_max,
                alm_feas_tol=alm_feas_tol,
                alm_stationarity_tol=alm_stationarity_tol,
                alm_trust_radius_init=alm_trust_radius_init,
                alm_trust_radius_min=alm_trust_radius_min,
                alm_trust_radius_shrink=alm_trust_radius_shrink,
                alm_trust_radius_grow=alm_trust_radius_grow,
                alm_max_inner_attempts=alm_max_inner_attempts,
                alm_max_subproblem_continuations=alm_max_subproblem_continuations,
                alm_distance_smoothing=alm_distance_smoothing,
                alm_curvature_smoothing=alm_curvature_smoothing,
            ),
        )
        object.__setattr__(
            self,
            "_basin",
            Stage2BasinControls(
                basin_hops=basin_hops,
                basin_stepsize=basin_stepsize,
                basin_temperature=basin_temperature,
                basin_niter_success=basin_niter_success,
                basin_seed=basin_seed,
                init_only=init_only,
            ),
        )
        object.__setattr__(
            self,
            "_finite_current",
            Stage2FiniteCurrentConfig(
                finite_current_mode=finite_current_mode,
                proxy_plasma_current_A=proxy_plasma_current_A,
                vf_current_A=vf_current_A,
                vf_template_path=vf_template_path,
            ),
        )
        object.__setattr__(
            self,
            "_iota",
            Stage2IotaConfig(
                stage2_iota_mode=stage2_iota_mode,
                stage2_iota_target=stage2_iota_target,
                stage2_iota_tolerance=stage2_iota_tolerance,
                stage2_iota_weight=stage2_iota_weight,
                stage2_iota_vol_target=stage2_iota_vol_target,
                stage2_iota_constraint_weight=stage2_iota_constraint_weight,
                stage2_iota_num_tf_coils=stage2_iota_num_tf_coils,
                stage2_iota_nphi=stage2_iota_nphi,
                stage2_iota_ntheta=stage2_iota_ntheta,
                stage2_iota_mpol=stage2_iota_mpol,
                stage2_iota_ntor=stage2_iota_ntor,
            ),
        )
        validate_constraint_cli_overrides(
            {
                "tf_current_A": self.tf_current_A,
                "banana_current_max_A": self.banana_current_max_A,
                "length_target": self.length_target,
                "cc_threshold": self.cc_threshold,
                "curvature_threshold": self.curvature_threshold,
                "banana_surf_radius": self.banana_surf_radius,
                "target_lcfs_max_major_radius_m": self.target_lcfs_max_major_radius_m,
                "target_lcfs_max_minor_radius_m": self.target_lcfs_max_minor_radius_m,
            }
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

    @property
    def plasma_surf_filename(self) -> str:
        return self._io.plasma_surf_filename

    @property
    def output_root(self) -> Path:
        return self._io.output_root

    @property
    def equilibria_dir(self) -> str | None:
        return self._io.equilibria_dir

    @property
    def tf_current_A(self) -> float:
        return self._hardware.tf_current_A

    @property
    def major_radius(self) -> float:
        return self._geometry.major_radius

    @property
    def toroidal_flux(self) -> float:
        return self._geometry.toroidal_flux

    @property
    def length_weight(self) -> float:
        return self._objective_weights.length_weight

    @property
    def cc_weight(self) -> float:
        return self._objective_weights.cc_weight

    @property
    def cc_threshold(self) -> float:
        return self._constraint_policy.cc_threshold

    @property
    def curvature_weight(self) -> float:
        return self._objective_weights.curvature_weight

    @property
    def curvature_threshold(self) -> float:
        return self._constraint_policy.curvature_threshold

    @property
    def banana_surf_radius(self) -> float:
        return self._geometry.banana_surf_radius

    @property
    def order(self) -> int:
        return self._geometry.order

    @property
    def constraint_method(self) -> str:
        return self._constraint_policy.constraint_method

    @property
    def alm_max_outer_iters(self) -> int:
        return self._alm.alm_max_outer_iters

    @property
    def alm_penalty_init(self) -> float:
        return self._alm.alm_penalty_init

    @property
    def alm_penalty_scale(self) -> float:
        return self._alm.alm_penalty_scale

    @property
    def basin_hops(self) -> int:
        return self._basin.basin_hops

    @property
    def basin_stepsize(self) -> float:
        return self._basin.basin_stepsize

    @property
    def alm_penalty_max(self) -> float:
        return self._alm.alm_penalty_max

    @property
    def alm_feas_tol(self) -> float:
        return self._alm.alm_feas_tol

    @property
    def alm_stationarity_tol(self) -> float:
        return self._alm.alm_stationarity_tol

    @property
    def alm_trust_radius_init(self) -> float:
        return self._alm.alm_trust_radius_init

    @property
    def alm_trust_radius_min(self) -> float:
        return self._alm.alm_trust_radius_min

    @property
    def alm_trust_radius_shrink(self) -> float:
        return self._alm.alm_trust_radius_shrink

    @property
    def alm_trust_radius_grow(self) -> float:
        return self._alm.alm_trust_radius_grow

    @property
    def alm_max_inner_attempts(self) -> int:
        return self._alm.alm_max_inner_attempts

    @property
    def alm_max_subproblem_continuations(self) -> int:
        return self._alm.alm_max_subproblem_continuations

    @property
    def alm_distance_smoothing(self) -> float:
        return self._alm.alm_distance_smoothing

    @property
    def alm_curvature_smoothing(self) -> float:
        return self._alm.alm_curvature_smoothing

    @property
    def basin_temperature(self) -> float:
        return self._basin.basin_temperature

    @property
    def basin_niter_success(self) -> int:
        return self._basin.basin_niter_success

    @property
    def basin_seed(self) -> int | None:
        return self._basin.basin_seed

    @property
    def init_only(self) -> bool:
        return self._basin.init_only

    @property
    def banana_init_current_A(self) -> float:
        return self._hardware.banana_init_current_A

    @property
    def banana_current_max_A(self) -> float:
        return self._hardware.banana_current_max_A

    @property
    def length_target(self) -> float:
        return self._constraint_policy.length_target

    @property
    def finite_current_mode(self) -> str:
        return self._finite_current.finite_current_mode

    @property
    def proxy_plasma_current_A(self) -> float:
        return self._finite_current.proxy_plasma_current_A

    @property
    def vf_current_A(self) -> float:
        return self._finite_current.vf_current_A

    @property
    def vf_template_path(self) -> str | None:
        return self._finite_current.vf_template_path

    @property
    def target_lcfs_max_major_radius_m(self) -> float:
        return self._constraint_policy.target_lcfs_max_major_radius_m

    @property
    def target_lcfs_max_minor_radius_m(self) -> float:
        return self._constraint_policy.target_lcfs_max_minor_radius_m

    @property
    def stage2_iota_mode(self) -> str:
        return self._iota.stage2_iota_mode

    @property
    def stage2_iota_target(self) -> float | None:
        return self._iota.stage2_iota_target

    @property
    def stage2_iota_tolerance(self) -> float:
        return self._iota.stage2_iota_tolerance

    @property
    def stage2_iota_weight(self) -> float:
        return self._iota.stage2_iota_weight

    @property
    def stage2_iota_vol_target(self) -> float:
        return self._iota.stage2_iota_vol_target

    @property
    def stage2_iota_constraint_weight(self) -> float:
        return self._iota.stage2_iota_constraint_weight

    @property
    def stage2_iota_num_tf_coils(self) -> int:
        return self._iota.stage2_iota_num_tf_coils

    @property
    def stage2_iota_nphi(self) -> int:
        return self._iota.stage2_iota_nphi

    @property
    def stage2_iota_ntheta(self) -> int:
        return self._iota.stage2_iota_ntheta

    @property
    def stage2_iota_mpol(self) -> int:
        return self._iota.stage2_iota_mpol

    @property
    def stage2_iota_ntor(self) -> int:
        return self._iota.stage2_iota_ntor

    @property
    def effective_vf_template_path(self) -> str | None:
        # Fresh Stage 2 wrappers materialize the repo-default VF template only
        # when building command/identity artifacts; keep the stored field raw.
        return self._finite_current.effective_vf_template_path


def stage2_artifact_config_flat_field_names() -> tuple[str, ...]:
    return STAGE2_ARTIFACT_CONFIG_FLAT_FIELD_NAMES


def stage2_artifact_config_flat_dict(
    config: Stage2ArtifactConfig,
) -> dict[str, object]:
    return {
        field_name: getattr(config, field_name)
        for field_name in STAGE2_ARTIFACT_CONFIG_FLAT_FIELD_NAMES
    }


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
        length_target=config.length_target,
        finite_current_mode=config.finite_current_mode,
        proxy_plasma_current_A=config.proxy_plasma_current_A,
        vf_current_A=config.vf_current_A,
        vf_template_path=config.effective_vf_template_path,
        target_lcfs_max_major_radius_m=config.target_lcfs_max_major_radius_m,
        target_lcfs_max_minor_radius_m=config.target_lcfs_max_minor_radius_m,
    )


def _stage2_artifact_path_kwargs(config: Stage2ArtifactConfig) -> dict[str, object]:
    flat_config = stage2_artifact_config_flat_dict(config)
    return {
        field_name: flat_config[field_name]
        for field_name in STAGE2_ARTIFACT_PATH_FIELD_NAMES
    }


def resolve_stage2_artifact_path(config: Stage2ArtifactConfig) -> Path:
    return local_stage2_bs_path(
        config.output_root,
        build_stage2_seed_spec(config),
        **_stage2_artifact_path_kwargs(config),
    )


def build_stage2_command(
    config: Stage2ArtifactConfig,
    *,
    constraint_override_reason: str | None = None,
    constraint_profile_label: str | None = None,
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
        "--length-target",
        str(config.length_target),
        "--target-lcfs-max-major-radius-m",
        str(config.target_lcfs_max_major_radius_m),
        "--target-lcfs-max-minor-radius-m",
        str(config.target_lcfs_max_minor_radius_m),
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
    effective_vf_template_path = config.effective_vf_template_path
    if effective_vf_template_path not in {None, ""}:
        command.extend(["--vf-template-path", str(effective_vf_template_path)])
    if config.equilibria_dir is not None:
        command.extend(["--equilibria-dir", config.equilibria_dir])
    if constraint_profile_label not in {None, ""}:
        command.extend(["--constraint-profile-label", constraint_profile_label])
    if constraint_override_reason not in {None, ""}:
        command.extend(["--constraint-override-reason", constraint_override_reason])
    if config.constraint_method == "alm":
        append_alm_cli_flags(command, config, cli_fields=STAGE2_ALM_CLI_FIELDS)
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


def _build_subprocess_env(
    *,
    env_overrides: Mapping[str, str] | None = None,
    inherit_alm: bool = False,
) -> dict[str, str]:
    subprocess_env = {
        key: value
        for key, value in os.environ.items()
        if inherit_alm or not key.startswith("ALM_")
    }
    if env_overrides is not None:
        subprocess_env.update(env_overrides)
    return subprocess_env


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = SCRIPT_DIR,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
    inherit_alm_env: bool = False,
) -> None:
    """Run a subprocess with repo-standard ALM environment isolation.

    The child environment starts from ``os.environ`` with inherited ``ALM_*``
    keys stripped unless ``inherit_alm_env`` is true. Explicit ``env`` entries
    are applied last, so caller-supplied ``ALM_*`` overrides are preserved.
    """
    if dry_run:
        return
    subprocess.run(
        list(command),
        cwd=str(cwd),
        check=True,
        timeout=timeout_seconds,
        env=_build_subprocess_env(
            env_overrides=env,
            inherit_alm=inherit_alm_env,
        ),
    )


def ensure_stage2_artifact(
    config: Stage2ArtifactConfig,
    *,
    constraint_override_reason: str | None = None,
    constraint_profile_label: str | None = None,
    python_executable: str = sys.executable,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
) -> Path:
    return ensure_stage2_artifact_result(
        config,
        constraint_override_reason=constraint_override_reason,
        constraint_profile_label=constraint_profile_label,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
    ).artifact_path


def ensure_stage2_artifact_result(
    config: Stage2ArtifactConfig,
    *,
    constraint_override_reason: str | None = None,
    constraint_profile_label: str | None = None,
    python_executable: str = sys.executable,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
) -> Stage2ArtifactEnsureResult:
    artifact_path = resolve_stage2_artifact_path(config)
    if dry_run:
        return Stage2ArtifactEnsureResult(
            artifact_path=artifact_path,
            artifact_reused=artifact_path.exists(),
        )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_path.with_name(f".{artifact_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        artifact_reused = artifact_path.exists()
        if not artifact_reused:
            run_command(
                build_stage2_command(
                    config,
                    constraint_override_reason=constraint_override_reason,
                    constraint_profile_label=constraint_profile_label,
                    python_executable=python_executable,
                ),
                timeout_seconds=timeout_seconds,
                dry_run=False,
            )
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Expected Stage 2 artifact was not created: {artifact_path}"
                )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return Stage2ArtifactEnsureResult(
        artifact_path=artifact_path,
        artifact_reused=artifact_reused,
    )


def load_stage2_artifact_results(stage2_bs_path: str | Path) -> tuple[Path, dict]:
    stage2_bs_path = Path(stage2_bs_path)
    stage2_results_path = stage2_bs_path.with_name("results.json")
    if not stage2_results_path.is_file():
        raise ValueError(STAGE2_SIDECAR_REQUIRED_ERROR)
    stage2_results = load_json(stage2_results_path)
    recorded_digest = stage2_results.get(STAGE2_BS_SHA256_KEY)
    if recorded_digest in {None, ""}:
        raise ValueError(
            "Stage 2 artifact results.json is missing STAGE2_BS_SHA256; "
            "cannot verify checksum binding."
        )
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
    if isinstance(payload, PurePath):
        return str(payload)
    if isinstance(payload, np.ndarray):
        return _json_portable_value(payload.tolist())
    if isinstance(payload, np.generic):
        return _json_portable_value(payload.item())
    if isinstance(payload, float):
        if not math.isfinite(payload):
            return None
        return payload
    if isinstance(payload, Mapping):
        return {
            str(key): _json_portable_value(value)
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


def json_dumps(payload: object, *, indent: int | None = None) -> str:
    return json.dumps(_json_portable_value(payload), indent=indent, allow_nan=False)


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


def resolve_single_stage_iota_target_arg(args: object) -> float:
    iota_target = float(getattr(args, "iota_target"))
    return -iota_target if bool(getattr(args, "flip_banana", False)) else iota_target


def format_flip_banana_banner(
    requested_iota_target: float,
    effective_iota_target: float,
) -> str:
    requested = float(requested_iota_target)
    effective = float(effective_iota_target)
    return (
        "[FLIP_BANANA] --flip-banana active: iota_target will be negated "
        f"(requested {requested:.6g} -> effective {effective:.6g}; "
        "mirror banana convention)."
    )


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value is None else int(value)


def add_seed_order_upgrade_argument(parser) -> None:
    parser.add_argument(
        "--seed-order-upgrade",
        type=int,
        default=env_optional_int("SEED_ORDER_UPGRADE"),
        help=(
            "Optional Fourier order upgrade applied by the single-stage entrypoint "
            "when loading the Stage 2 seed."
        ),
    )


def add_stage2_warm_start_seed_arguments(parser) -> None:
    parser.add_argument(
        "--stage2-seed-surf-path",
        default=os.environ.get("STAGE2_SEED_SURF_PATH"),
        help=(
            "Optional saved surface or Boozer-surface artifact forwarded into the "
            "single-stage entrypoint as a Stage 2 warm-start seed."
        ),
    )
    parser.add_argument(
        "--warm-start-surface-stem",
        default=None,
        help=(
            "Optional stem for saved single-stage surface artifacts "
            "(for example /path/to/surf_best_feasible). When set, the single-stage "
            "entrypoint reuses the saved Boozer surface geometry/iota/G as its "
            "initialization seed."
        ),
    )


def append_single_stage_handoff_flags(command: list[str], args: object) -> None:
    equilibrium_path = resolved_optional_path(getattr(args, "equilibrium_path", None))
    if equilibrium_path is not None:
        command.extend(["--equilibrium-path", str(equilibrium_path)])
    stage2_seed_surf_path = resolved_optional_path(
        getattr(args, "stage2_seed_surf_path", None)
    )
    if stage2_seed_surf_path is not None:
        command.extend(["--stage2-seed-surf-path", str(stage2_seed_surf_path)])
    warm_start_surface_stem = resolved_optional_path(
        getattr(args, "warm_start_surface_stem", None)
    )
    if warm_start_surface_stem is not None:
        command.extend(
            ["--warm-start-surface-stem", str(warm_start_surface_stem)]
        )
    append_optional_flag(
        command,
        "--seed-order-upgrade",
        getattr(args, "seed_order_upgrade", None),
    )
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
        "--flip-banana",
        bool(getattr(args, "flip_banana", False)),
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
    subprocess.run(
        command,
        cwd=str(SCRIPT_DIR),
        check=True,
        timeout=timeout_seconds,
        env=_build_subprocess_env(
            env_overrides={"POINCARE_OUT_DIR": str(resolved_path(output_dir))},
        ),
    )
    return command


def timeout_or_none(timeout_seconds: float) -> float | None:
    return None if timeout_seconds <= 0.0 else float(timeout_seconds)
