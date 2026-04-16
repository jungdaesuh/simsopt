from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def format_compact_float(value: float) -> str:
    return f"{value:g}"


def validate_normalized_toroidal_flux(
    value: float,
    *,
    field_name: str = "toroidal_flux",
) -> float:
    flux = float(value)
    if not 0.0 <= flux <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1 inclusive, got {value!r}."
        )
    return flux


@dataclass(frozen=True)
class Stage2SeedSpec:
    plasma_surf_filename: str
    major_radius: float
    toroidal_flux: float
    length_weight: float
    cc_weight: float
    cc_threshold: float
    curvature_weight: float
    curvature_threshold: float
    banana_surf_radius: float
    tf_current_A: float
    order: int
    banana_init_current_A: float = 1.0e4
    banana_current_max_A: float = 1.6e4
    finite_current_mode: str = "boozer_surrogate"
    proxy_plasma_current_A: float = 0.0
    vf_current_A: float = 0.0
    vf_template_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "toroidal_flux",
            validate_normalized_toroidal_flux(
                self.toroidal_flux,
                field_name="Stage2SeedSpec.toroidal_flux",
            ),
        )


@dataclass(frozen=True)
class SingleStageWeightCase:
    name: str
    res_weight: float
    iotas_weight: float
    cc_weight: float
    curvature_weight: float
    length_weight: float
    cs_weight: float
    surf_dist_weight: float


_DEFAULT_STAGE2_ALM_MAX_SUBPROBLEM_CONTINUATIONS = 20
_DEFAULT_STAGE2_ALM_FEAS_TOL = 1.0e-6
_DEFAULT_STAGE2_ALM_STATIONARITY_TOL = 1.0e-6
_DEFAULT_STAGE2_ALM_TRUST_RADIUS_INIT = 0.05
_DEFAULT_STAGE2_ALM_TRUST_RADIUS_MIN = 1.0e-4
_DEFAULT_STAGE2_ALM_TRUST_RADIUS_SHRINK = 0.5
_DEFAULT_STAGE2_ALM_TRUST_RADIUS_GROW = 1.5
_DEFAULT_STAGE2_ALM_MAX_INNER_ATTEMPTS = 4
_DEFAULT_STAGE2_ALM_DISTANCE_SMOOTHING = 0.005
_DEFAULT_STAGE2_ALM_CURVATURE_SMOOTHING = 0.25
_DEFAULT_STAGE2_IOTA_MODE = "off"
_DEFAULT_STAGE2_IOTA_TOLERANCE = 5.0e-3
_DEFAULT_STAGE2_IOTA_WEIGHT = 1.0
_DEFAULT_STAGE2_IOTA_VOL_TARGET = 0.10
_DEFAULT_STAGE2_IOTA_CONSTRAINT_WEIGHT = 1.0
_DEFAULT_STAGE2_IOTA_NUM_TF_COILS = 20
_DEFAULT_STAGE2_IOTA_NPHI = 91
_DEFAULT_STAGE2_IOTA_NTHETA = 32
_DEFAULT_STAGE2_IOTA_MPOL = 8
_DEFAULT_STAGE2_IOTA_NTOR = 6


def canonical_stage2_iota_constraint_weight(
    constraint_weight: float | None,
) -> float | None:
    if constraint_weight is None:
        return None
    normalized_constraint_weight = float(constraint_weight)
    return None if normalized_constraint_weight <= 0.0 else normalized_constraint_weight


def validate_stage2_iota_args(
    *,
    stage2_iota_mode: str,
    stage2_iota_target: float | None,
    stage2_iota_tolerance: float,
    stage2_iota_vol_target: float,
    stage2_iota_num_tf_coils: int,
    stage2_iota_nphi: int,
    stage2_iota_ntheta: int,
    stage2_iota_mpol: int,
    stage2_iota_ntor: int,
    stage2_iota_weight: float,
    constraint_method: str,
) -> None:
    if stage2_iota_mode == _DEFAULT_STAGE2_IOTA_MODE:
        return
    if stage2_iota_target is None:
        raise ValueError(
            "--stage2-iota-target is required when --stage2-iota-mode is enabled."
        )
    if stage2_iota_tolerance <= 0.0:
        raise ValueError("--stage2-iota-tolerance must be positive.")
    if stage2_iota_vol_target <= 0.0:
        raise ValueError("--stage2-iota-vol-target must be positive.")
    if stage2_iota_num_tf_coils <= 0:
        raise ValueError("--stage2-iota-num-tf-coils must be positive.")
    if stage2_iota_nphi <= 0 or stage2_iota_ntheta <= 0:
        raise ValueError(
            "--stage2-iota-nphi and --stage2-iota-ntheta must both be positive."
        )
    if stage2_iota_mpol <= 0 or stage2_iota_ntor <= 0:
        raise ValueError(
            "--stage2-iota-mpol and --stage2-iota-ntor must both be positive."
        )
    if stage2_iota_mode == "soft" and stage2_iota_weight <= 0.0:
        raise ValueError("--stage2-iota-weight must be positive in soft mode.")
    if stage2_iota_mode == "alm" and constraint_method != "alm":
        raise ValueError(
            "--stage2-iota-mode=alm requires --constraint-method=alm."
        )


def format_local_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CCT={format_compact_float(spec.cc_threshold)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-CT={format_compact_float(spec.curvature_threshold)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-INITC={format_compact_float(spec.banana_init_current_A)}"
        f"-MAXC={format_compact_float(spec.banana_current_max_A)}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
        f"{format_stage2_finite_current_suffix(spec)}"
    )


def format_database_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"MR={format_compact_float(spec.major_radius)}"
        f"-TF={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={format_compact_float(spec.banana_surf_radius)}"
        f"-INITC={format_compact_float(spec.banana_init_current_A)}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
        f"{format_stage2_finite_current_suffix(spec)}"
    )


def format_database_stage2_seed_dir_without_init_current(spec: Stage2SeedSpec) -> str:
    return (
        f"MR={format_compact_float(spec.major_radius)}"
        f"-TF={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={format_compact_float(spec.banana_surf_radius)}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
        f"{format_stage2_finite_current_suffix(spec)}"
    )


def format_legacy_database_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"MR={format_compact_float(spec.major_radius)}"
        f"-TF={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={format_compact_float(spec.banana_surf_radius)}"
        f"-Order={spec.order}"
        f"{format_stage2_finite_current_suffix(spec)}"
    )


def format_local_stage2_seed_dir_without_tf(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CCT={format_compact_float(spec.cc_threshold)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-CT={format_compact_float(spec.curvature_threshold)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-Order={spec.order}"
        f"{format_stage2_finite_current_suffix(spec)}"
    )


def format_local_stage2_seed_dir_without_init_current(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CCT={format_compact_float(spec.cc_threshold)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-CT={format_compact_float(spec.curvature_threshold)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
        f"{format_stage2_finite_current_suffix(spec)}"
    )


def format_legacy_local_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-Order={spec.order}"
        f"{format_stage2_finite_current_suffix(spec)}"
    )


def format_stage2_finite_current_suffix(spec: Stage2SeedSpec) -> str:
    if (
        spec.finite_current_mode == "boozer_surrogate"
        and abs(float(spec.proxy_plasma_current_A)) <= 1.0e-12
        and abs(float(spec.vf_current_A)) <= 1.0e-12
        and spec.vf_template_path in {None, ""}
    ):
        return ""
    suffix = f"-FCM={spec.finite_current_mode}"
    suffix += f"-PPC={format_compact_float(spec.proxy_plasma_current_A)}"
    suffix += f"-VFC={format_compact_float(spec.vf_current_A)}"
    if spec.vf_template_path not in {None, ""}:
        suffix += f"-VFT={Path(spec.vf_template_path).stem}"
    return suffix


def format_stage2_constraint_suffix(
    constraint_method: str,
    alm_max_outer_iters: int,
    alm_penalty_init: float,
    alm_penalty_scale: float,
    alm_penalty_max: float | None = None,
    alm_max_subproblem_continuations: int = _DEFAULT_STAGE2_ALM_MAX_SUBPROBLEM_CONTINUATIONS,
    alm_feas_tol: float = _DEFAULT_STAGE2_ALM_FEAS_TOL,
    alm_stationarity_tol: float = _DEFAULT_STAGE2_ALM_STATIONARITY_TOL,
    alm_trust_radius_init: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_INIT,
    alm_trust_radius_min: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_MIN,
    alm_trust_radius_shrink: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_SHRINK,
    alm_trust_radius_grow: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_GROW,
    alm_max_inner_attempts: int = _DEFAULT_STAGE2_ALM_MAX_INNER_ATTEMPTS,
    alm_distance_smoothing: float = _DEFAULT_STAGE2_ALM_DISTANCE_SMOOTHING,
    alm_curvature_smoothing: float = _DEFAULT_STAGE2_ALM_CURVATURE_SMOOTHING,
) -> str:
    if constraint_method == "alm":
        suffix = (
            f"-CM=alm-ALMOuter={alm_max_outer_iters}"
            f"-ALMMu={format_compact_float(alm_penalty_init)}"
            f"-ALMScale={format_compact_float(alm_penalty_scale)}"
        )
        if alm_penalty_max is not None:
            suffix += f"-ALMMax={format_compact_float(alm_penalty_max)}"
        optional_alm_segments = (
            (
                "ALMSub",
                int(alm_max_subproblem_continuations),
                _DEFAULT_STAGE2_ALM_MAX_SUBPROBLEM_CONTINUATIONS,
            ),
            ("ALMFeas", float(alm_feas_tol), _DEFAULT_STAGE2_ALM_FEAS_TOL),
            (
                "ALMStat",
                float(alm_stationarity_tol),
                _DEFAULT_STAGE2_ALM_STATIONARITY_TOL,
            ),
            (
                "ALMTR",
                float(alm_trust_radius_init),
                _DEFAULT_STAGE2_ALM_TRUST_RADIUS_INIT,
            ),
            (
                "ALMTRMin",
                float(alm_trust_radius_min),
                _DEFAULT_STAGE2_ALM_TRUST_RADIUS_MIN,
            ),
            (
                "ALMTRShrink",
                float(alm_trust_radius_shrink),
                _DEFAULT_STAGE2_ALM_TRUST_RADIUS_SHRINK,
            ),
            (
                "ALMTRGrow",
                float(alm_trust_radius_grow),
                _DEFAULT_STAGE2_ALM_TRUST_RADIUS_GROW,
            ),
            (
                "ALMInner",
                int(alm_max_inner_attempts),
                _DEFAULT_STAGE2_ALM_MAX_INNER_ATTEMPTS,
            ),
            (
                "ALMDist",
                float(alm_distance_smoothing),
                _DEFAULT_STAGE2_ALM_DISTANCE_SMOOTHING,
            ),
            (
                "ALMCurv",
                float(alm_curvature_smoothing),
                _DEFAULT_STAGE2_ALM_CURVATURE_SMOOTHING,
            ),
        )
        for label, value, default in optional_alm_segments:
            if value != default:
                suffix += f"-{label}={format_compact_float(float(value))}"
        return suffix
    return "-CM=penalty"


def format_stage2_basin_suffix(
    basin_hops: int,
    basin_stepsize: float,
    basin_temperature: float = 1.0,
    basin_niter_success: int = 0,
    basin_seed: int | None = None,
) -> str:
    if basin_hops <= 0:
        return ""
    seed_value = "none" if basin_seed is None else str(basin_seed)
    suffix = (
        f"-BH={basin_hops}"
        f"-BS={format_compact_float(basin_stepsize)}"
        f"-BSeed={seed_value}"
    )
    if basin_temperature != 1.0:
        suffix += f"-BT={format_compact_float(basin_temperature)}"
    if basin_niter_success > 0:
        suffix += f"-BNS={basin_niter_success}"
    return suffix


def format_stage2_iota_suffix(
    stage2_iota_mode: str = _DEFAULT_STAGE2_IOTA_MODE,
    stage2_iota_target: float | None = None,
    stage2_iota_tolerance: float = _DEFAULT_STAGE2_IOTA_TOLERANCE,
    stage2_iota_weight: float = _DEFAULT_STAGE2_IOTA_WEIGHT,
    stage2_iota_vol_target: float = _DEFAULT_STAGE2_IOTA_VOL_TARGET,
    stage2_iota_constraint_weight: float = _DEFAULT_STAGE2_IOTA_CONSTRAINT_WEIGHT,
    stage2_iota_num_tf_coils: int = _DEFAULT_STAGE2_IOTA_NUM_TF_COILS,
    stage2_iota_nphi: int = _DEFAULT_STAGE2_IOTA_NPHI,
    stage2_iota_ntheta: int = _DEFAULT_STAGE2_IOTA_NTHETA,
    stage2_iota_mpol: int = _DEFAULT_STAGE2_IOTA_MPOL,
    stage2_iota_ntor: int = _DEFAULT_STAGE2_IOTA_NTOR,
) -> str:
    if stage2_iota_mode == _DEFAULT_STAGE2_IOTA_MODE:
        return ""
    if stage2_iota_target is None:
        raise ValueError(
            "stage2_iota_target is required when stage2_iota_mode is enabled."
        )
    canonical_constraint_weight = canonical_stage2_iota_constraint_weight(
        stage2_iota_constraint_weight
    )
    suffix = (
        f"-IM={stage2_iota_mode}"
        f"-ITarget={format_compact_float(stage2_iota_target)}"
        f"-ITol={format_compact_float(stage2_iota_tolerance)}"
        f"-IVol={format_compact_float(stage2_iota_vol_target)}"
        f"-ICW={'exact' if canonical_constraint_weight is None else format_compact_float(canonical_constraint_weight)}"
        f"-INTF={int(stage2_iota_num_tf_coils)}"
        f"-INPhi={int(stage2_iota_nphi)}"
        f"-INTheta={int(stage2_iota_ntheta)}"
        f"-IMPol={int(stage2_iota_mpol)}"
        f"-INTor={int(stage2_iota_ntor)}"
    )
    if stage2_iota_mode == "soft":
        suffix += f"-IW={format_compact_float(stage2_iota_weight)}"
    return suffix


def format_local_stage2_run_dir(
    spec: Stage2SeedSpec,
    *,
    constraint_method: str,
    alm_max_outer_iters: int,
    alm_penalty_init: float,
    alm_penalty_scale: float,
    alm_penalty_max: float | None = None,
    alm_max_subproblem_continuations: int = _DEFAULT_STAGE2_ALM_MAX_SUBPROBLEM_CONTINUATIONS,
    alm_feas_tol: float = _DEFAULT_STAGE2_ALM_FEAS_TOL,
    alm_stationarity_tol: float = _DEFAULT_STAGE2_ALM_STATIONARITY_TOL,
    alm_trust_radius_init: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_INIT,
    alm_trust_radius_min: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_MIN,
    alm_trust_radius_shrink: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_SHRINK,
    alm_trust_radius_grow: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_GROW,
    alm_max_inner_attempts: int = _DEFAULT_STAGE2_ALM_MAX_INNER_ATTEMPTS,
    alm_distance_smoothing: float = _DEFAULT_STAGE2_ALM_DISTANCE_SMOOTHING,
    alm_curvature_smoothing: float = _DEFAULT_STAGE2_ALM_CURVATURE_SMOOTHING,
    basin_hops: int,
    basin_stepsize: float,
    basin_temperature: float = 1.0,
    basin_niter_success: int = 0,
    basin_seed: int | None = None,
    stage2_iota_mode: str = _DEFAULT_STAGE2_IOTA_MODE,
    stage2_iota_target: float | None = None,
    stage2_iota_tolerance: float = _DEFAULT_STAGE2_IOTA_TOLERANCE,
    stage2_iota_weight: float = _DEFAULT_STAGE2_IOTA_WEIGHT,
    stage2_iota_vol_target: float = _DEFAULT_STAGE2_IOTA_VOL_TARGET,
    stage2_iota_constraint_weight: float = _DEFAULT_STAGE2_IOTA_CONSTRAINT_WEIGHT,
    stage2_iota_num_tf_coils: int = _DEFAULT_STAGE2_IOTA_NUM_TF_COILS,
    stage2_iota_nphi: int = _DEFAULT_STAGE2_IOTA_NPHI,
    stage2_iota_ntheta: int = _DEFAULT_STAGE2_IOTA_NTHETA,
    stage2_iota_mpol: int = _DEFAULT_STAGE2_IOTA_MPOL,
    stage2_iota_ntor: int = _DEFAULT_STAGE2_IOTA_NTOR,
) -> str:
    return (
        format_local_stage2_seed_dir(spec)
        + format_stage2_constraint_suffix(
            constraint_method,
            alm_max_outer_iters,
            alm_penalty_init,
            alm_penalty_scale,
            alm_penalty_max,
            alm_max_subproblem_continuations,
            alm_feas_tol,
            alm_stationarity_tol,
            alm_trust_radius_init,
            alm_trust_radius_min,
            alm_trust_radius_shrink,
            alm_trust_radius_grow,
            alm_max_inner_attempts,
            alm_distance_smoothing,
            alm_curvature_smoothing,
        )
        + format_stage2_basin_suffix(
            basin_hops,
            basin_stepsize,
            basin_temperature,
            basin_niter_success,
            basin_seed,
        )
        + format_stage2_iota_suffix(
            stage2_iota_mode,
            stage2_iota_target,
            stage2_iota_tolerance,
            stage2_iota_weight,
            stage2_iota_vol_target,
            stage2_iota_constraint_weight,
            stage2_iota_num_tf_coils,
            stage2_iota_nphi,
            stage2_iota_ntheta,
            stage2_iota_mpol,
            stage2_iota_ntor,
        )
    )


def local_stage2_bs_path(
    output_root: str | Path,
    spec: Stage2SeedSpec,
    *,
    constraint_method: str,
    alm_max_outer_iters: int,
    alm_penalty_init: float,
    alm_penalty_scale: float,
    alm_penalty_max: float | None = None,
    alm_max_subproblem_continuations: int = _DEFAULT_STAGE2_ALM_MAX_SUBPROBLEM_CONTINUATIONS,
    alm_feas_tol: float = _DEFAULT_STAGE2_ALM_FEAS_TOL,
    alm_stationarity_tol: float = _DEFAULT_STAGE2_ALM_STATIONARITY_TOL,
    alm_trust_radius_init: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_INIT,
    alm_trust_radius_min: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_MIN,
    alm_trust_radius_shrink: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_SHRINK,
    alm_trust_radius_grow: float = _DEFAULT_STAGE2_ALM_TRUST_RADIUS_GROW,
    alm_max_inner_attempts: int = _DEFAULT_STAGE2_ALM_MAX_INNER_ATTEMPTS,
    alm_distance_smoothing: float = _DEFAULT_STAGE2_ALM_DISTANCE_SMOOTHING,
    alm_curvature_smoothing: float = _DEFAULT_STAGE2_ALM_CURVATURE_SMOOTHING,
    basin_hops: int,
    basin_stepsize: float,
    basin_temperature: float = 1.0,
    basin_niter_success: int = 0,
    basin_seed: int | None = None,
    stage2_iota_mode: str = _DEFAULT_STAGE2_IOTA_MODE,
    stage2_iota_target: float | None = None,
    stage2_iota_tolerance: float = _DEFAULT_STAGE2_IOTA_TOLERANCE,
    stage2_iota_weight: float = _DEFAULT_STAGE2_IOTA_WEIGHT,
    stage2_iota_vol_target: float = _DEFAULT_STAGE2_IOTA_VOL_TARGET,
    stage2_iota_constraint_weight: float = _DEFAULT_STAGE2_IOTA_CONSTRAINT_WEIGHT,
    stage2_iota_num_tf_coils: int = _DEFAULT_STAGE2_IOTA_NUM_TF_COILS,
    stage2_iota_nphi: int = _DEFAULT_STAGE2_IOTA_NPHI,
    stage2_iota_ntheta: int = _DEFAULT_STAGE2_IOTA_NTHETA,
    stage2_iota_mpol: int = _DEFAULT_STAGE2_IOTA_MPOL,
    stage2_iota_ntor: int = _DEFAULT_STAGE2_IOTA_NTOR,
) -> Path:
    return (
        Path(output_root)
        / f"outputs-{spec.plasma_surf_filename}"
        / format_local_stage2_run_dir(
            spec,
            constraint_method=constraint_method,
            alm_max_outer_iters=alm_max_outer_iters,
            alm_penalty_init=alm_penalty_init,
            alm_penalty_scale=alm_penalty_scale,
            alm_penalty_max=alm_penalty_max,
            alm_max_subproblem_continuations=alm_max_subproblem_continuations,
            alm_feas_tol=alm_feas_tol,
            alm_stationarity_tol=alm_stationarity_tol,
            alm_trust_radius_init=alm_trust_radius_init,
            alm_trust_radius_min=alm_trust_radius_min,
            alm_trust_radius_shrink=alm_trust_radius_shrink,
            alm_trust_radius_grow=alm_trust_radius_grow,
            alm_max_inner_attempts=alm_max_inner_attempts,
            alm_distance_smoothing=alm_distance_smoothing,
            alm_curvature_smoothing=alm_curvature_smoothing,
            basin_hops=basin_hops,
            basin_stepsize=basin_stepsize,
            basin_temperature=basin_temperature,
            basin_niter_success=basin_niter_success,
            basin_seed=basin_seed,
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
        )
        / "biot_savart_opt.json"
    )


def build_weight_cases(
    base_weights: Mapping[str, float],
    scan_weights: Sequence[str],
    multipliers: Sequence[float],
) -> list[SingleStageWeightCase]:
    baseline = SingleStageWeightCase(
        name="baseline",
        res_weight=float(base_weights["res_weight"]),
        iotas_weight=float(base_weights["iotas_weight"]),
        cc_weight=float(base_weights["cc_weight"]),
        curvature_weight=float(base_weights["curvature_weight"]),
        length_weight=float(base_weights["length_weight"]),
        cs_weight=float(base_weights["cs_weight"]),
        surf_dist_weight=float(base_weights["surf_dist_weight"]),
    )
    cases = [baseline]
    seen = {
        (
            baseline.res_weight,
            baseline.iotas_weight,
            baseline.cc_weight,
            baseline.curvature_weight,
            baseline.length_weight,
            baseline.cs_weight,
            baseline.surf_dist_weight,
        )
    }
    for weight_name in scan_weights:
        for multiplier in multipliers:
            scaled_multiplier = float(multiplier)
            if scaled_multiplier == 1.0:
                continue
            fields = {
                "res_weight": baseline.res_weight,
                "iotas_weight": baseline.iotas_weight,
                "cc_weight": baseline.cc_weight,
                "curvature_weight": baseline.curvature_weight,
                "length_weight": baseline.length_weight,
                "cs_weight": baseline.cs_weight,
                "surf_dist_weight": baseline.surf_dist_weight,
            }
            fields[weight_name] = fields[weight_name] * scaled_multiplier
            signature = (
                fields["res_weight"],
                fields["iotas_weight"],
                fields["cc_weight"],
                fields["curvature_weight"],
                fields["length_weight"],
                fields["cs_weight"],
                fields["surf_dist_weight"],
            )
            if signature in seen:
                continue
            seen.add(signature)
            multiplier_label = str(scaled_multiplier).replace("-", "m").replace(".", "p")
            cases.append(
                SingleStageWeightCase(
                    name=f"{weight_name}-x{multiplier_label}",
                    **fields,
                )
            )
    return cases


def augment_metrics(record: Mapping[str, object]) -> dict[str, float]:
    augmented = {
        key: float(value)
        for key, value in record.items()
        if isinstance(value, (int, float))
    }
    final_iota = record.get("FINAL_IOTA")
    target_iota = record.get("TARGET_IOTA")
    if isinstance(final_iota, (int, float)) and isinstance(target_iota, (int, float)):
        augmented["IOTA_ERROR_ABS"] = abs(float(final_iota) - float(target_iota))
    return augmented


def dominates(
    lhs: Mapping[str, object],
    rhs: Mapping[str, object],
    metrics: Sequence[str],
) -> bool:
    lhs_metrics = augment_metrics(lhs)
    rhs_metrics = augment_metrics(rhs)
    if any(metric not in lhs_metrics or metric not in rhs_metrics for metric in metrics):
        return False
    return all(lhs_metrics[metric] <= rhs_metrics[metric] for metric in metrics) and any(
        lhs_metrics[metric] < rhs_metrics[metric] for metric in metrics
    )


def select_non_dominated_records(
    records: Iterable[Mapping[str, object]],
    metrics: Sequence[str],
) -> list[Mapping[str, object]]:
    materialized = list(records)
    return [
        record
        for index, record in enumerate(materialized)
        if not any(
            other_index != index and dominates(other, record, metrics)
            for other_index, other in enumerate(materialized)
        )
    ]
