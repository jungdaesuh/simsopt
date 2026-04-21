from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from simsopt._core.optimizable import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier
from simsopt.geo.surfaceobjectives import Volume

from .coil_groups import (
    COIL_GROUP_ROLE_BANANA,
    COIL_GROUP_ROLE_PROXY,
    COIL_GROUP_ROLE_TF,
    COIL_GROUP_ROLE_VF,
    ManifestResolution,
    partition_coils_by_manifest,
    resolve_manifest,
)
from .current_contracts import resolve_finite_current_mode, resolve_loaded_tf_current_A
from .hardware_contracts import (
    ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE_ENV,
    BANANA_WINDING_MINOR_RADIUS_M,
    MAX_CURVATURE_INV_M,
    env_flag,
    validate_banana_winding_surface_radius,
    validate_plasma_vessel_clearance,
    validate_tf_current_limit,
)
from .single_stage_geometry import build_surface_configs

BOOTABILITY_REASON_OK = "ok"
BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA = "missing_artifact_metadata"
BOOTABILITY_REASON_BOOZER_SOLVE_FAILED = "boozer_solve_failed"
BOOTABILITY_REASON_SELF_INTERSECTION = "self_intersection"
BOOTABILITY_REASON_IOTA_MISMATCH = "iota_mismatch"

BOOTABILITY_STAGE_PROBE = "probe"
BOOTABILITY_STAGE_RECOVERY = "recovery"

BOOZER_FAILURE_POLICY_REPORT_FAILURE = "report_failure"
BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS = "restore_last_success"

__all__ = [
    "BOOTABILITY_REASON_BOOZER_SOLVE_FAILED",
    "BOOTABILITY_REASON_IOTA_MISMATCH",
    "BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA",
    "BOOTABILITY_REASON_OK",
    "BOOTABILITY_REASON_SELF_INTERSECTION",
    "BOOTABILITY_STAGE_PROBE",
    "BOOTABILITY_STAGE_RECOVERY",
    "BOOZER_FAILURE_POLICY_REPORT_FAILURE",
    "BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS",
    "BoozerInitializationResult",
    "BoozerSolveAttempt",
    "BoozerSolveSnapshot",
    "Stage2CoilPartitions",
    "WarmStartBoozerSeed",
    "bootability_passes",
    "build_equilibrium_path",
    "classify_bootability_result",
    "compute_tf_G0",
    "initialize_boozer_surface",
    "load_warm_start_boozer_seed",
    "partition_loaded_stage2_coils",
    "probe_stage2_seed_bootability",
    "resolve_warm_start_boozer_surface_path",
    "restore_boozer_solve_state",
    "resolve_stage2_finite_current_mode",
    "resolve_single_stage_banana_surf_radius",
    "resolve_stage2_num_tf_coils",
    "resolve_stage2_tf_current_A",
    "run_boozer_with_failure_policy",
    "snapshot_boozer_solve_state",
    "validate_loaded_stage2_coils_partition",
    "validate_stage2_seed_contract",
]


@dataclass(frozen=True)
class BoozerInitializationResult:
    boozer_surface: object | None
    solve_success: bool
    self_intersecting: bool | None
    success: bool
    solved_iota: float | None
    solved_G: float | None
    volume: float | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BoozerSolveSnapshot:
    surface_dofs: np.ndarray
    iota: float
    G: float | None
    success: bool


@dataclass(frozen=True)
class BoozerSolveAttempt:
    solve_success: bool
    solved_iota: float | None
    solved_G: float | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class WarmStartBoozerSeed:
    surface: object
    iota: float | None
    G: float | None
    source_path: Path


@dataclass(frozen=True)
class Stage2CoilPartitions:
    tf_coils: tuple[object, ...]
    banana_coils: tuple[object, ...]
    proxy_coils: tuple[object, ...]
    vf_coils: tuple[object, ...]
    num_tf_coils: int
    num_banana_coils: int
    num_proxy_coils: int
    num_vf_coils: int
    finite_current_mode: str
    coil_groups_manifest_is_legacy_inferred: bool = False


def resolve_stage2_tf_current_A(stage2_results, tf_coils):
    return resolve_loaded_tf_current_A(stage2_results.get("TF_CURRENT_A"), tf_coils)


def resolve_stage2_num_tf_coils(stage2_results, requested_num_tf_coils):
    requested_num_tf_coils = int(requested_num_tf_coils)
    recorded_num_tf_coils = stage2_results.get("NUM_TF_COILS")
    if recorded_num_tf_coils is None:
        return requested_num_tf_coils
    resolved_num_tf_coils = int(recorded_num_tf_coils)
    if resolved_num_tf_coils <= 0:
        raise ValueError(
            f"Stage 2 artifact reports invalid NUM_TF_COILS={recorded_num_tf_coils!r}; "
            "cannot partition loaded coils."
        )
    if resolved_num_tf_coils != requested_num_tf_coils:
        raise ValueError(
            "Loaded Stage 2 artifact reports "
            f"NUM_TF_COILS={resolved_num_tf_coils}, but --num-tf-coils={requested_num_tf_coils}. "
            "Single-stage reload now refuses to re-slice coils with inconsistent TF-count provenance."
        )
    return resolved_num_tf_coils


def resolve_single_stage_banana_surf_radius(
    stage2_results,
    requested_banana_surf_radius,
):
    artifact_banana_surf_radius = validate_banana_winding_surface_radius(
        stage2_results["banana_surf_radius"]
    )
    if requested_banana_surf_radius is None:
        return artifact_banana_surf_radius
    resolved_banana_surf_radius = validate_banana_winding_surface_radius(
        requested_banana_surf_radius
    )
    if abs(resolved_banana_surf_radius - artifact_banana_surf_radius) > 1.0e-12:
        raise ValueError(
            "Single-stage banana winding surface must match the loaded Stage 2 artifact "
            f"radius {artifact_banana_surf_radius:.6f} m; got "
            f"{resolved_banana_surf_radius:.6f} m."
        )
    return resolved_banana_surf_radius


def resolve_stage2_finite_current_mode(
    stage2_results: Mapping[str, object],
    requested_finite_current_mode: str | None,
) -> str:
    return resolve_finite_current_mode(
        requested_finite_current_mode,
        artifact_mode=stage2_results.get("FINITE_CURRENT_MODE"),
        artifact_mode_source=stage2_results.get("FINITE_CURRENT_MODE_SOURCE"),
    )


def _resolve_stage2_coil_manifest(
    stage2_results: Mapping[str, object],
    *,
    requested_num_tf_coils: int,
    total_loaded_coils: int,
) -> ManifestResolution:
    # TF count consistency is enforced up-front to preserve the pre-manifest
    # provenance contract, then the manifest resolver takes over.
    resolved_num_tf_coils = resolve_stage2_num_tf_coils(
        stage2_results,
        requested_num_tf_coils=requested_num_tf_coils,
    )
    if resolved_num_tf_coils > total_loaded_coils:
        raise ValueError(
            f"Loaded Stage 2 BiotSavart artifact has only {total_loaded_coils} coils, but "
            f"NUM_TF_COILS={resolved_num_tf_coils}. Cannot partition TF and banana coils."
        )
    resolution = resolve_manifest(
        stage2_results,
        total_loaded_coils=total_loaded_coils,
        requested_num_tf_coils=resolved_num_tf_coils,
    )
    num_banana_coils = resolution.manifest.count_for_role(COIL_GROUP_ROLE_BANANA)
    if num_banana_coils <= 0:
        raise ValueError(
            f"Loaded Stage 2 BiotSavart artifact has {total_loaded_coils} coils and "
            f"NUM_TF_COILS={resolved_num_tf_coils}, leaving no banana coils to optimize."
        )
    return resolution


def partition_loaded_stage2_coils(
    coils,
    *,
    stage2_results: Mapping[str, object],
    requested_num_tf_coils: int,
) -> Stage2CoilPartitions:
    resolution = _resolve_stage2_coil_manifest(
        stage2_results,
        requested_num_tf_coils=requested_num_tf_coils,
        total_loaded_coils=len(coils),
    )
    role_partitions = partition_coils_by_manifest(coils, resolution.manifest)
    tf_coils = role_partitions.get(COIL_GROUP_ROLE_TF, ())
    banana_coils = role_partitions.get(COIL_GROUP_ROLE_BANANA, ())
    proxy_coils = role_partitions.get(COIL_GROUP_ROLE_PROXY, ())
    vf_coils = role_partitions.get(COIL_GROUP_ROLE_VF, ())
    return Stage2CoilPartitions(
        tf_coils=tf_coils,
        banana_coils=banana_coils,
        proxy_coils=proxy_coils,
        vf_coils=vf_coils,
        num_tf_coils=len(tf_coils),
        num_banana_coils=len(banana_coils),
        num_proxy_coils=len(proxy_coils),
        num_vf_coils=len(vf_coils),
        finite_current_mode=resolve_stage2_finite_current_mode(stage2_results, None),
        coil_groups_manifest_is_legacy_inferred=resolution.is_legacy_inferred,
    )


def validate_loaded_stage2_coils_partition(
    coils,
    *,
    stage2_results: Mapping[str, object],
    requested_num_tf_coils: int,
):
    partition_loaded_stage2_coils(
        coils,
        stage2_results=stage2_results,
        requested_num_tf_coils=int(requested_num_tf_coils),
    )


def build_equilibrium_path(
    plasma_surf_filename: str,
    equilibria_dir: str | Path,
    *,
    equilibrium_path: str | Path | None = None,
    database_equilibria_dir: str | Path | None = None,
) -> str:
    if equilibrium_path is not None:
        return str(equilibrium_path)

    candidate_paths = [Path(equilibria_dir) / plasma_surf_filename]
    if database_equilibria_dir is not None:
        candidate_paths.append(Path(database_equilibria_dir) / plasma_surf_filename)
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return str(candidate_path)
    return str(candidate_paths[0])


def validate_stage2_seed_contract(stage2_results):
    tf_current_A = stage2_results.get("TF_CURRENT_A")
    if tf_current_A is None:
        raise ValueError(
            "Stage 2 seed artifact is missing TF_CURRENT_A even after legacy-contract "
            "upgrade. Pass --stage2-seed-tf-current-A explicitly or use a newer "
            "artifact with TF-current metadata."
        )
    validate_tf_current_limit(tf_current_A)
    validate_banana_winding_surface_radius(
        stage2_results.get("banana_surf_radius", BANANA_WINDING_MINOR_RADIUS_M)
    )
    if (
        float(stage2_results.get("CURVATURE_THRESHOLD", MAX_CURVATURE_INV_M))
        > MAX_CURVATURE_INV_M
    ):
        raise ValueError(
            "Stage 2 seed curvature threshold exceeds the hardware ceiling of "
            f"{MAX_CURVATURE_INV_M:.1f} m^-1."
        )
    surface_vessel_min_dist = stage2_results.get(
        "SURFACE_VESSEL_MIN_DIST",
        stage2_results.get("PLASMA_VESSEL_MIN_DIST"),
    )
    if surface_vessel_min_dist is None:
        warnings.warn(
            "Stage 2 seed artifact predates the plasma-vessel clearance gate "
            "(missing SURFACE_VESSEL_MIN_DIST). Re-run stage-2 to verify the "
            "LCFS clearance against the HBT-EP vacuum vessel.",
            stacklevel=2,
        )
        return
    validate_plasma_vessel_clearance(
        surface_vessel_min_dist,
        accept_offspec=env_flag(ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE_ENV),
    )


def compute_tf_G0(tf_coils) -> float:
    # Keep G0 tied to the TF bundle only. Proxy/VF coils already enter the
    # loaded Biot-Savart field, so folding them into the toroidal-current seed
    # here would double count their effect during the Boozer initialization.
    current_sum = float(sum(abs(coil.current.get_value()) for coil in tf_coils))
    return 2.0 * np.pi * current_sum * (4.0 * np.pi * 10.0 ** (-7) / (2.0 * np.pi))


def _coerce_boozer_scalar(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _boozer_result_state(boozer_surface) -> Mapping[str, object]:
    res = getattr(boozer_surface, "res", None)
    if res is None:
        return {}
    return res


def _read_boozer_solve_scalars(boozer_surface) -> tuple[float | None, float | None]:
    res = _boozer_result_state(boozer_surface)
    return (
        _coerce_boozer_scalar(res.get("iota")),
        _coerce_boozer_scalar(res.get("G")),
    )


def _attempt_from_current_boozer_state(
    boozer_surface,
    *,
    solve_success: bool,
    error: Exception | None = None,
) -> BoozerSolveAttempt:
    solved_iota, solved_G = _read_boozer_solve_scalars(boozer_surface)
    return BoozerSolveAttempt(
        solve_success=bool(solve_success),
        solved_iota=solved_iota,
        solved_G=solved_G,
        error_type=None if error is None else type(error).__name__,
        error_message=None if error is None else str(error),
    )


def snapshot_boozer_solve_state(boozer_surface) -> BoozerSolveSnapshot:
    res = _boozer_result_state(boozer_surface)
    _, solved_G = _read_boozer_solve_scalars(boozer_surface)
    return BoozerSolveSnapshot(
        surface_dofs=np.asarray(boozer_surface.surface.x, dtype=float).copy(),
        iota=float(res["iota"]),
        G=solved_G,
        success=bool(res.get("success", False)),
    )


def restore_boozer_solve_state(
    boozer_surface,
    snapshot: BoozerSolveSnapshot,
) -> None:
    boozer_surface.surface.x = snapshot.surface_dofs.copy()
    res = getattr(boozer_surface, "res", None)
    if res is None:
        res = {}
        boozer_surface.res = res
    res["iota"] = snapshot.iota
    res["G"] = snapshot.G
    res["success"] = snapshot.success
    if hasattr(boozer_surface, "need_to_run_code"):
        boozer_surface.need_to_run_code = False


def run_boozer_with_failure_policy(
    boozer_surface,
    iota,
    G,
    *,
    failure_policy: str,
    last_successful_state: BoozerSolveSnapshot | None = None,
) -> BoozerSolveAttempt:
    if failure_policy not in (
        BOOZER_FAILURE_POLICY_REPORT_FAILURE,
        BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS,
    ):
        raise ValueError(f"Unsupported Boozer failure policy {failure_policy!r}.")
    if (
        failure_policy == BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS
        and last_successful_state is None
    ):
        raise ValueError(
            "restore_last_success Boozer failure policy requires a successful snapshot."
        )

    def _restore_if_needed() -> None:
        if failure_policy == BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS:
            restore_boozer_solve_state(boozer_surface, last_successful_state)

    try:
        result = boozer_surface.run_code(iota, G)
    except Exception as error:
        attempt = _attempt_from_current_boozer_state(
            boozer_surface,
            solve_success=False,
            error=error,
        )
        _restore_if_needed()
        return attempt

    res = _boozer_result_state(boozer_surface)
    if result is None:
        attempt = _attempt_from_current_boozer_state(
            boozer_surface,
            solve_success=bool(res.get("success", False)),
        )
        if not attempt.solve_success:
            _restore_if_needed()
        return attempt

    solve_success = bool(result.get("success", False))
    solved_iota = _coerce_boozer_scalar(result.get("iota", res.get("iota")))
    solved_G = _coerce_boozer_scalar(result.get("G", res.get("G")))
    if not solve_success:
        _restore_if_needed()
    return BoozerSolveAttempt(
        solve_success=solve_success,
        solved_iota=solved_iota,
        solved_G=solved_G,
    )


def _surface_volume_or_none(surface) -> float | None:
    try:
        return float(surface.volume())
    except Exception:
        return None


def _surface_dofs(surface) -> np.ndarray:
    get_dofs = getattr(surface, "get_dofs", None)
    if callable(get_dofs):
        return np.asarray(get_dofs(), dtype=float)
    return np.asarray(surface.dofs, dtype=float)


def resolve_warm_start_boozer_surface_path(
    warm_start_surface_stem: str | Path,
    *,
    surface_name: str,
) -> Path:
    surface_stem = Path(warm_start_surface_stem)
    candidate_paths = [
        surface_stem.with_name(
            f"{surface_stem.name}_{surface_name}_boozer_surface.json"
        )
    ]
    if surface_name == "outer":
        candidate_paths.append(
            surface_stem.with_name(f"{surface_stem.name}_boozer_surface.json")
        )
    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return candidate_path
    raise FileNotFoundError(
        "Warm-start Boozer surface artifact not found. Checked: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def load_warm_start_boozer_seed(
    warm_start_boozer_surface_path: str | Path,
    *,
    artifact_loader=load,
) -> WarmStartBoozerSeed:
    source_path = Path(warm_start_boozer_surface_path)
    warm_start_artifact = artifact_loader(str(source_path))
    if hasattr(warm_start_artifact, "surface"):
        res = _boozer_result_state(warm_start_artifact)
        iota = _coerce_boozer_scalar(res.get("iota"))
        G = _coerce_boozer_scalar(res.get("G"))
        warm_start_surface = warm_start_artifact.surface
    else:
        iota = None
        G = None
        warm_start_surface = warm_start_artifact
    return WarmStartBoozerSeed(
        surface=warm_start_surface,
        iota=iota,
        G=G,
        source_path=source_path,
    )


def attempt_initialize_boozer_surface(
    surf_prev,
    mpol,
    ntor,
    bs,
    vol_target,
    constraint_weight,
    iota,
    G0,
    boozer_I=0.0,
    *,
    initial_surface_guess=None,
    nfp=5,
    surface_cls=SurfaceXYZTensorFourier,
    volume_cls=Volume,
    boozer_surface_cls=BoozerSurface,
) -> BoozerInitializationResult:
    surf = surface_cls(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_theta=surf_prev.quadpoints_theta,
        quadpoints_phi=surf_prev.quadpoints_phi,
        dofs=(
            None
            if initial_surface_guess is None
            else _surface_dofs(initial_surface_guess)
        ),
    )
    if initial_surface_guess is None:
        surf.least_squares_fit(surf_prev.gamma())

    if constraint_weight is not None:
        vol = volume_cls(surf)
        boozer_surface = boozer_surface_cls(
            bs,
            surf,
            vol,
            vol_target,
            constraint_weight,
            options={"verbose": True},
            I=boozer_I,
        )
    else:
        surf_exact = surface_cls(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=True,
            quadpoints_theta=np.linspace(0, 1, 2 * mpol + 1, endpoint=False),
            quadpoints_phi=np.linspace(0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
            dofs=_surface_dofs(surf),
        )
        vol = volume_cls(surf_exact)
        boozer_surface = boozer_surface_cls(
            bs,
            surf_exact,
            vol,
            vol_target,
            None,
            options={"verbose": True},
            I=boozer_I,
        )

    solve_attempt = run_boozer_with_failure_policy(
        boozer_surface,
        iota,
        G0,
        failure_policy=BOOZER_FAILURE_POLICY_REPORT_FAILURE,
    )
    if solve_attempt.error_type is not None:
        return BoozerInitializationResult(
            boozer_surface=boozer_surface,
            solve_success=False,
            self_intersecting=None,
            success=False,
            solved_iota=solve_attempt.solved_iota,
            solved_G=solve_attempt.solved_G,
            volume=_surface_volume_or_none(boozer_surface.surface),
            error_type=solve_attempt.error_type,
            error_message=solve_attempt.error_message,
        )

    solve_success = solve_attempt.solve_success
    try:
        self_intersecting = bool(boozer_surface.surface.is_self_intersecting())
    except Exception:
        self_intersecting = True
    return BoozerInitializationResult(
        boozer_surface=boozer_surface,
        solve_success=solve_success,
        self_intersecting=self_intersecting,
        success=solve_success and not self_intersecting,
        solved_iota=solve_attempt.solved_iota,
        solved_G=solve_attempt.solved_G,
        volume=_surface_volume_or_none(boozer_surface.surface),
    )


def initialize_boozer_surface(
    surf_prev,
    mpol,
    ntor,
    bs,
    vol_target,
    constraint_weight,
    iota,
    G0,
    boozer_I=0.0,
    *,
    initial_surface_guess=None,
    nfp=5,
    surface_cls=SurfaceXYZTensorFourier,
    volume_cls=Volume,
    boozer_surface_cls=BoozerSurface,
):
    """Initialize a Boozer surface via either the "least squares" or "exact" path.

    constraint_weight: set to a finite weight to use Boozer least-squares; pass
        ``None`` to use the Boozer "exact" algorithm.
    iota: initial guess for the rotational transform on the surface.
    G0: net toroidal current linking the torus hole.
    nfp: number of field periods (default 5 for banana coils).

    Raises ``RuntimeError`` on solver failure; use :func:`attempt_initialize_boozer_surface`
    for a non-raising variant that returns a :class:`BoozerInitializationResult`.
    """
    result = attempt_initialize_boozer_surface(
        surf_prev,
        mpol,
        ntor,
        bs,
        vol_target,
        constraint_weight,
        iota,
        G0,
        boozer_I,
        initial_surface_guess=initial_surface_guess,
        nfp=nfp,
        surface_cls=surface_cls,
        volume_cls=volume_cls,
        boozer_surface_cls=boozer_surface_cls,
    )
    if result.success:
        return result.boozer_surface

    print(
        "Boozer initialization failed: "
        f"solve_success={result.solve_success}, "
        f"self_intersecting={result.self_intersecting}, "
        f"volume={result.volume}, "
        f"iota_guess={iota}, "
        f"iota_solved={result.solved_iota}"
    )
    if result.error_type is not None:
        print(
            "Boozer initialization raised "
            f"{result.error_type}: {result.error_message}"
        )
    raise RuntimeError("Something went wrong with the Boozer solve...")


def _bootability_failure(
    *,
    stage: str,
    target_iota: float,
    reason: str,
    solved_iota: float | None = None,
    self_intersecting: bool | None = None,
    solve_success: bool | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, object]:
    abs_iota_error = None
    if solved_iota is not None:
        abs_iota_error = abs(float(solved_iota) - float(target_iota))
    return {
        "BOOZER_BOOTABLE": False,
        "IOTA_FEASIBLE": False,
        "BOOTABILITY_REASON": reason,
        "BOOTABILITY_STAGE": stage,
        "BOOTABILITY_TARGET_IOTA": float(target_iota),
        "BOOTABILITY_SOLVED_IOTA": (
            None if solved_iota is None else float(solved_iota)
        ),
        "BOOTABILITY_SELF_INTERSECTING": self_intersecting,
        "BOOTABILITY_SOLVE_SUCCESS": solve_success,
        "BOOTABILITY_ABS_IOTA_ERROR": abs_iota_error,
        "BOOTABILITY_ERROR_TYPE": error_type,
        "BOOTABILITY_ERROR_MESSAGE": error_message,
    }


def classify_bootability_result(
    initialization: BoozerInitializationResult,
    *,
    stage: str,
    target_iota: float,
    iota_tolerance: float,
) -> dict[str, object]:
    if initialization.error_type is not None or not initialization.solve_success:
        return _bootability_failure(
            stage=stage,
            target_iota=target_iota,
            reason=BOOTABILITY_REASON_BOOZER_SOLVE_FAILED,
            solved_iota=initialization.solved_iota,
            self_intersecting=initialization.self_intersecting,
            solve_success=initialization.solve_success,
            error_type=initialization.error_type,
            error_message=initialization.error_message,
        )
    if initialization.self_intersecting:
        return _bootability_failure(
            stage=stage,
            target_iota=target_iota,
            reason=BOOTABILITY_REASON_SELF_INTERSECTION,
            solved_iota=initialization.solved_iota,
            self_intersecting=True,
            solve_success=True,
        )
    solved_iota = initialization.solved_iota
    if solved_iota is None:
        return _bootability_failure(
            stage=stage,
            target_iota=target_iota,
            reason=BOOTABILITY_REASON_BOOZER_SOLVE_FAILED,
            solve_success=True,
        )
    abs_iota_error = abs(float(solved_iota) - float(target_iota))
    if abs_iota_error > float(iota_tolerance):
        return _bootability_failure(
            stage=stage,
            target_iota=target_iota,
            reason=BOOTABILITY_REASON_IOTA_MISMATCH,
            solved_iota=solved_iota,
            self_intersecting=False,
            solve_success=True,
        )
    return {
        "BOOZER_BOOTABLE": True,
        "IOTA_FEASIBLE": True,
        "BOOTABILITY_REASON": BOOTABILITY_REASON_OK,
        "BOOTABILITY_STAGE": stage,
        "BOOTABILITY_TARGET_IOTA": float(target_iota),
        "BOOTABILITY_SOLVED_IOTA": float(solved_iota),
        "BOOTABILITY_SELF_INTERSECTING": False,
        "BOOTABILITY_SOLVE_SUCCESS": True,
        "BOOTABILITY_ABS_IOTA_ERROR": abs_iota_error,
        "BOOTABILITY_ERROR_TYPE": None,
        "BOOTABILITY_ERROR_MESSAGE": None,
    }


def bootability_passes(bootability_status: Mapping[str, object]) -> bool:
    return bool(
        bootability_status.get("BOOZER_BOOTABLE")
        and bootability_status.get("IOTA_FEASIBLE")
    )


def _required_handoff_metadata_keys(
    stage2_artifact_results: Mapping[str, object],
) -> list[str]:
    required_keys = (
        "MAJOR_RADIUS",
        "TOROIDAL_FLUX",
        "banana_surf_radius",
    )
    return [key for key in required_keys if stage2_artifact_results.get(key) is None]


def _probe_initialization_inputs(
    *,
    stage2_artifact_results: Mapping[str, object],
    plasma_surf_filename: str,
    equilibria_dir: str | Path,
    equilibrium_path: str | Path | None,
    database_equilibria_dir: str | Path | None,
    nphi: int,
    ntheta: int,
    vol_target: float,
    iota_target: float,
    tf_coils,
    stage2_seed_surf_path: str | Path | None,
    artifact_loader,
) -> tuple[object, float, object | None, float, float]:
    default_G = compute_tf_G0(tf_coils)
    if stage2_seed_surf_path is not None:
        warm_start_seed = load_warm_start_boozer_seed(
            stage2_seed_surf_path,
            artifact_loader=artifact_loader,
        )
        return (
            warm_start_seed.surface,
            float(vol_target),
            warm_start_seed.surface,
            iota_target if warm_start_seed.iota is None else warm_start_seed.iota,
            default_G if warm_start_seed.G is None else warm_start_seed.G,
        )

    equilibrium_file = build_equilibrium_path(
        plasma_surf_filename,
        equilibria_dir,
        equilibrium_path=equilibrium_path,
        database_equilibria_dir=database_equilibria_dir,
    )
    surface_configs = build_surface_configs(
        equilibrium_file,
        nphi,
        ntheta,
        float(stage2_artifact_results["TOROIDAL_FLUX"]),
        float(stage2_artifact_results["MAJOR_RADIUS"]),
        vol_target,
        1,
        0.8,
    )
    outer_surface_config = surface_configs[-1]
    return (
        outer_surface_config["initial_surface"],
        outer_surface_config["target_volume"],
        None,
        iota_target,
        default_G,
    )


def probe_stage2_seed_bootability(
    *,
    stage2_bs_path: str | Path,
    stage2_artifact_results: Mapping[str, object],
    plasma_surf_filename: str,
    equilibria_dir: str | Path,
    num_tf_coils: int,
    nphi: int,
    ntheta: int,
    mpol: int,
    ntor: int,
    vol_target: float,
    iota_target: float,
    iota_tolerance: float,
    constraint_weight: float | None,
    boozer_I: float = 0.0,
    stage: str = BOOTABILITY_STAGE_PROBE,
    equilibrium_path: str | Path | None = None,
    database_equilibria_dir: str | Path | None = None,
    stage2_seed_surf_path: str | Path | None = None,
    bs_loader=load,
) -> dict[str, object]:
    missing_metadata = _required_handoff_metadata_keys(stage2_artifact_results)
    if missing_metadata:
        return _bootability_failure(
            stage=stage,
            target_iota=iota_target,
            reason=BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA,
            error_message=(
                "Stage 2 artifact results.json is missing "
                + ", ".join(missing_metadata)
            ),
        )
    try:
        validate_stage2_seed_contract(stage2_artifact_results)
        bs = bs_loader(stage2_bs_path)
        coil_partitions = partition_loaded_stage2_coils(
            bs.coils,
            stage2_results=stage2_artifact_results,
            requested_num_tf_coils=num_tf_coils,
        )
        tf_coils = coil_partitions.tf_coils
        resolve_stage2_tf_current_A(stage2_artifact_results, tf_coils)
        (
            initial_surface,
            target_volume,
            initial_surface_guess,
            initial_iota,
            initial_G,
        ) = _probe_initialization_inputs(
            stage2_artifact_results=stage2_artifact_results,
            plasma_surf_filename=plasma_surf_filename,
            equilibria_dir=equilibria_dir,
            equilibrium_path=equilibrium_path,
            database_equilibria_dir=database_equilibria_dir,
            nphi=nphi,
            ntheta=ntheta,
            vol_target=vol_target,
            iota_target=iota_target,
            tf_coils=tf_coils,
            stage2_seed_surf_path=stage2_seed_surf_path,
            artifact_loader=bs_loader,
        )
        initialization = attempt_initialize_boozer_surface(
            initial_surface,
            mpol,
            ntor,
            bs,
            target_volume,
            constraint_weight,
            initial_iota,
            initial_G,
            boozer_I,
            initial_surface_guess=initial_surface_guess,
            nfp=initial_surface.nfp,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        return _bootability_failure(
            stage=stage,
            target_iota=iota_target,
            reason=BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    return classify_bootability_result(
        initialization,
        stage=stage,
        target_iota=iota_target,
        iota_tolerance=iota_tolerance,
    )
