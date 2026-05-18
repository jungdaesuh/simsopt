from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.geo.surfaceobjectives import Volume

from .boozer_finite_current import BoozerSurfaceFiniteI
from .json_compat import load_boozer_finite_i

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
    COIL_LENGTH_MIN_FRACTION,
    MAX_CURVATURE_INV_M,
    POLOIDAL_EXTENT_HALF_WIDTH_RAD,
    validate_banana_winding_surface_radius,
    validate_major_radius,
    validate_tf_current_limit,
)
from .hardware_constraint_schema import (
    build_hardware_constraint_status,
    build_threshold_overrides,
    hardware_constraint_artifact_field_names,
    hardware_constraint_artifact_value_field_names,
)
from .single_stage_geometry import (
    build_surface_configs,
    surface_self_intersection_status,
)
from .wout_convention import validate_wout_convention_artifact_fields

BOOTABILITY_REASON_OK = "ok"
BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA = "missing_artifact_metadata"
BOOTABILITY_REASON_BOOZER_SOLVE_FAILED = "boozer_solve_failed"
BOOTABILITY_REASON_SELF_INTERSECTION = "self_intersection"
BOOTABILITY_REASON_IOTA_MISMATCH = "iota_mismatch"

BOOTABILITY_STAGE_PROBE = "probe"
BOOTABILITY_STAGE_RECOVERY = "recovery"
_PROMOTABLE_STAGE2_IOTA_MODES = frozenset({"alm"})
_PRODUCTION_HANDOFF_SEED_ROLE = "bootable_handoff"

BOOZER_FAILURE_POLICY_REPORT_FAILURE = "report_failure"
BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS = "restore_last_success"

# SSOT for the Boozer residual trust gate (Phase 1 of
# docs/stage2_single_stage_boozer_handoff_impl_plan_2026-05-15.md).
#
# Newton convergence is enforced by the upstream BoozerSurface solver at
# ``options['newton_tol']`` (1e-13 for the exact path, 1e-11 for the LS path).
# A "trusted" solve is one whose reconstructed constrained-residual norm sits
# inside a small multiple of that tolerance — the multiplier keeps the gate
# above floating-point chatter while still rejecting damped/marginal solves
# that report success but lack the residual quality needed for a gradient
# signal.
BOOZER_TRUST_TOL_MULTIPLIER = 10.0
BOOZER_TRUST_DEFAULT_NEWTON_TOL_EXACT = 1.0e-13
BOOZER_TRUST_DEFAULT_NEWTON_TOL_LS = 1.0e-11
# Loose non-physical-iota filter (NOT a campaign envelope). The HBT-EP
# campaign rotational transforms live in the 0.15 — 0.30 band but the
# trust gate must accept anything that could conceivably be a healthy
# solve from a divergent equilibrium. ``|iota| < 1`` rejects only the
# clearly broken regime where iota approaches integer resonance; tighter
# filtering would mask early-iteration physics. Anything at or above this
# bound is treated as nonphysical and the iota objective is disabled
# until the next trustworthy solve.
BOOZER_TRUST_IOTA_ABS_BOUND = 1.0

# Reasons the Boozer trust gate disables the iota term. Surfaced into the
# artifact via ``BOOZER_TRUST_REASON`` so downstream reports can explain
# exactly why iota was inactive without re-reading the residual norm.
BOOZER_TRUST_REASON_OK = "trusted"
BOOZER_TRUST_REASON_SOLVE_FAILED = "solve_failed"
BOOZER_TRUST_REASON_SELF_INTERSECTING = "self_intersecting"
BOOZER_TRUST_REASON_RESIDUAL_TOO_LARGE = "residual_too_large"
BOOZER_TRUST_REASON_RESIDUAL_UNAVAILABLE = "residual_unavailable"
BOOZER_TRUST_REASON_IOTA_NONPHYSICAL = "iota_nonphysical"

load = load_boozer_finite_i

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
    "BOOZER_TRUST_DEFAULT_NEWTON_TOL_EXACT",
    "BOOZER_TRUST_DEFAULT_NEWTON_TOL_LS",
    "BOOZER_TRUST_IOTA_ABS_BOUND",
    "BOOZER_TRUST_REASON_IOTA_NONPHYSICAL",
    "BOOZER_TRUST_REASON_OK",
    "BOOZER_TRUST_REASON_RESIDUAL_TOO_LARGE",
    "BOOZER_TRUST_REASON_RESIDUAL_UNAVAILABLE",
    "BOOZER_TRUST_REASON_SELF_INTERSECTING",
    "BOOZER_TRUST_REASON_SOLVE_FAILED",
    "BOOZER_TRUST_TOL_MULTIPLIER",
    "BoozerInitializationResult",
    "BoozerSolveAttempt",
    "BoozerSolveSnapshot",
    "BoozerTrustState",
    "Stage2CoilPartitions",
    "WarmStartBoozerSeed",
    "boozer_trust_artifact_fields",
    "boozer_trust_state_from_initialization",
    "boozer_trust_tolerance",
    "bootability_passes",
    "build_stage2_seed_production_handoff_fields",
    "build_equilibrium_path",
    "classify_bootability_result",
    "compute_boozer_constrained_residual_norm",
    "compute_boozer_trust_state",
    "compute_tf_G0",
    "evaluate_stage2_seed_hardware_contract",
    "validate_stage2_seed_bootability_contract",
    "validate_stage2_seed_handoff_contract",
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


def _first_stage2_result_value(
    stage2_results: Mapping[str, object],
    field_names: tuple[str, ...],
) -> object | None:
    for field_name in field_names:
        value = stage2_results.get(field_name)
        if value is not None:
            return value
    return None


def _stage2_seed_measured_values(
    stage2_results: Mapping[str, object],
) -> dict[str, float | None]:
    measured_values: dict[str, float | None] = {}
    for constraint_name in hardware_constraint_artifact_field_names():
        value = _first_stage2_result_value(
            stage2_results,
            hardware_constraint_artifact_value_field_names(constraint_name),
        )
        measured_values[constraint_name] = None if value is None else float(value)
    return measured_values


def _required_stage2_result_value(
    stage2_results: Mapping[str, object],
    field_name: str,
    contract_name: str,
) -> object:
    value = stage2_results.get(field_name)
    if value is None:
        raise ValueError(
            f"Stage 2 seed artifact is missing {field_name}; cannot verify "
            f"{contract_name}."
        )
    return value


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
class BoozerTrustState:
    """Trust state for a single Boozer solve result.

    A "trusted" Boozer surface is one whose iota and G scalars can drive a
    gradient-based iota objective: the Newton residual converged inside the
    trust tolerance, the surface is not self-intersecting, and the iota value
    is physically plausible. Untrusted-but-evaluable solves disable the iota
    term without disturbing the base flux/hardware objective.
    """

    solve_success: bool
    self_intersecting: bool | None
    constrained_residual_norm: float | None
    trust_tol: float
    boozer_trusted: bool
    iota_objective_active: bool
    trust_reason: str


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
class _BoozerInitializationState:
    surface_dofs: np.ndarray
    res: dict[str, object] | None
    need_to_run_code: bool


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


def evaluate_stage2_seed_hardware_contract(
    stage2_results: Mapping[str, object],
) -> dict[str, object]:
    measured_values = _stage2_seed_measured_values(stage2_results)
    length_target = _first_stage2_result_value(
        stage2_results,
        ("length_target", "LENGTH_TARGET"),
    )
    length_min_target = (
        None if length_target is None else COIL_LENGTH_MIN_FRACTION * float(length_target)
    )
    return build_hardware_constraint_status(
        measured_values,
        applies_to="artifact",
        threshold_overrides=build_threshold_overrides(
            (
                ("coil_length_min", length_min_target),
            )
        ),
        require_values=True,
    )


def validate_stage2_seed_bootability_contract(
    stage2_results: Mapping[str, object],
) -> None:
    if (
        bootability_passes(stage2_results)
        and stage2_results.get("STAGE2_IOTA_MODE") in _PROMOTABLE_STAGE2_IOTA_MODES
        and stage2_results.get("BOOZER_TRUSTED") is True
        and stage2_results.get("IOTA_NEAR_TARGET") is True
        and stage2_results.get("WOUT_OFF_SPEC") is False
        and stage2_results.get("SEED_ROLE") == _PRODUCTION_HANDOFF_SEED_ROLE
        and stage2_results.get("DIAGNOSTIC_ONLY") is False
        and stage2_results.get("PRODUCTION_HANDOFF_READY") is True
        and "HANDOFF_BLOCKING_GATE" in stage2_results
        and stage2_results.get("HANDOFF_BLOCKING_GATE") is None
        and stage2_results.get("PROMOTION_READY") is True
    ):
        return
    raise ValueError(
        "Stage 2 seed artifact is not single-stage bootable: "
        f"SEED_ROLE={stage2_results.get('SEED_ROLE')!r}, "
        f"DIAGNOSTIC_ONLY={stage2_results.get('DIAGNOSTIC_ONLY')!r}, "
        f"PRODUCTION_HANDOFF_READY={stage2_results.get('PRODUCTION_HANDOFF_READY')!r}, "
        f"HANDOFF_BLOCKING_GATE={stage2_results.get('HANDOFF_BLOCKING_GATE')!r}, "
        f"PROMOTION_READY={stage2_results.get('PROMOTION_READY')!r}, "
        f"STAGE2_IOTA_MODE={stage2_results.get('STAGE2_IOTA_MODE')!r}, "
        f"WOUT_OFF_SPEC={stage2_results.get('WOUT_OFF_SPEC')!r}, "
        f"BOOZER_BOOTABLE={stage2_results.get('BOOZER_BOOTABLE')!r}, "
        f"BOOZER_TRUSTED={stage2_results.get('BOOZER_TRUSTED')!r}, "
        f"IOTA_NEAR_TARGET={stage2_results.get('IOTA_NEAR_TARGET')!r}, "
        f"IOTA_FEASIBLE={stage2_results.get('IOTA_FEASIBLE')!r}, "
        f"BOOTABILITY_REASON={stage2_results.get('BOOTABILITY_REASON')!r}, "
        f"BOOTABILITY_SOLVED_IOTA={stage2_results.get('BOOTABILITY_SOLVED_IOTA')!r}, "
        f"BOOTABILITY_TARGET_IOTA={stage2_results.get('BOOTABILITY_TARGET_IOTA')!r}."
    )


def build_stage2_seed_production_handoff_fields(
    *,
    wout_off_spec: bool,
) -> dict[str, object]:
    if wout_off_spec is not True and wout_off_spec is not False:
        raise ValueError("Production handoff requires explicit boolean WOUT_OFF_SPEC.")
    return {
        "WOUT_OFF_SPEC": wout_off_spec,
        "SEED_ROLE": _PRODUCTION_HANDOFF_SEED_ROLE,
        "DIAGNOSTIC_ONLY": False,
        "PRODUCTION_HANDOFF_READY": True,
        "HANDOFF_BLOCKING_GATE": None,
        "PROMOTION_READY": True,
    }


def validate_stage2_seed_contract(stage2_results):
    tf_current_A = stage2_results.get("TF_CURRENT_A")
    if tf_current_A is None:
        raise ValueError(
            "Stage 2 seed artifact is missing TF_CURRENT_A even after legacy-contract "
            "upgrade. Pass --stage2-seed-tf-current-A explicitly or use a newer "
            "artifact with TF-current metadata."
        )
    validate_tf_current_limit(tf_current_A)
    validate_wout_convention_artifact_fields(
        stage2_results_path="<stage2_seed_contract>",
        stage2_artifact_results=dict(stage2_results),
    )
    if stage2_results.get("WOUT_OFF_SPEC") is not False:
        raise ValueError(
            "Stage 2 seed artifact violates the WOUT convention contract: "
            "WOUT_OFF_SPEC=True."
        )
    major_radius = _required_stage2_result_value(
        stage2_results,
        "MAJOR_RADIUS",
        "the vacuum-vessel major-radius contract",
    )
    validate_major_radius(major_radius)
    banana_surf_radius = _required_stage2_result_value(
        stage2_results,
        "banana_surf_radius",
        "the banana winding-surface hardware contract",
    )
    validate_banana_winding_surface_radius(banana_surf_radius)
    curvature_threshold = _required_stage2_result_value(
        stage2_results,
        "CURVATURE_THRESHOLD",
        "the curvature hardware contract",
    )
    if float(curvature_threshold) > MAX_CURVATURE_INV_M:
        raise ValueError(
            "Stage 2 seed curvature threshold exceeds the hardware ceiling of "
            f"{MAX_CURVATURE_INV_M:.1f} m^-1."
        )
    poloidal_extent_threshold_rad = _required_stage2_result_value(
        stage2_results,
        "POLOIDAL_EXTENT_THRESHOLD_RAD",
        "the poloidal-extent hardware contract",
    )
    if float(poloidal_extent_threshold_rad) > POLOIDAL_EXTENT_HALF_WIDTH_RAD:
        raise ValueError(
            "Stage 2 seed poloidal-extent threshold exceeds the HBT-EP half-width "
            f"ceiling of {POLOIDAL_EXTENT_HALF_WIDTH_RAD:.6f} rad."
        )
    hardware_status = evaluate_stage2_seed_hardware_contract(stage2_results)
    if not hardware_status["success"]:
        raise ValueError(
            "Stage 2 seed artifact violates the full HBT-EP hardware contract: "
            + "; ".join(hardware_status["violations"])
        )


def validate_stage2_seed_handoff_contract(stage2_results):
    validate_stage2_seed_contract(stage2_results)
    validate_stage2_seed_bootability_contract(stage2_results)


def compute_tf_G0(tf_coils) -> float:
    # Keep G0 tied to the TF bundle only. Proxy/VF coils already enter the
    # loaded Biot-Savart field, so folding them into the toroidal-current seed
    # here would double count their effect during the Boozer initialization.
    current_sum = float(sum(coil.current.get_value() for coil in tf_coils))
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


def _boozer_solve_kind(boozer_surface) -> str:
    """Return ``"exact"`` or ``"ls"`` for the most recent Boozer solve.

    Reads ``boozer_surface.res['type']`` when available, falling back to the
    BoozerSurface ``boozer_type`` attribute. The kind drives both the default
    Newton tolerance and the residual reconstruction shape.
    """

    res = _boozer_result_state(boozer_surface)
    res_type = res.get("type")
    if res_type in ("ls", "exact"):
        return res_type
    boozer_type = getattr(boozer_surface, "boozer_type", None)
    if boozer_type in ("ls", "exact"):
        return boozer_type
    # Default to the more permissive LS tolerance — a missing type marker is
    # itself a trust signal handled by the residual-norm gate.
    return "ls"


def _boozer_newton_tol(boozer_surface) -> float:
    options = getattr(boozer_surface, "options", None) or {}
    explicit = options.get("newton_tol")
    if explicit is not None:
        return float(explicit)
    kind = _boozer_solve_kind(boozer_surface)
    if kind == "exact":
        return BOOZER_TRUST_DEFAULT_NEWTON_TOL_EXACT
    return BOOZER_TRUST_DEFAULT_NEWTON_TOL_LS


def boozer_trust_tolerance(boozer_surface) -> float:
    """Return the constrained-residual-norm trust threshold for ``boozer_surface``.

    Equal to ``BOOZER_TRUST_TOL_MULTIPLIER * options['newton_tol']`` where
    ``newton_tol`` is the upstream BoozerSurface solver tolerance. Solves whose
    reconstructed constrained-residual norm exceed this threshold are flagged
    as untrusted and the iota term is disabled.
    """

    return BOOZER_TRUST_TOL_MULTIPLIER * _boozer_newton_tol(boozer_surface)


def compute_boozer_constrained_residual_norm(boozer_surface) -> float | None:
    """Reconstruct the Boozer Newton success-check norm.

    The plan (Phase 1, "include the same components used by the Newton
    success check") asks for the vector the upstream solver compares against
    ``newton_tol``. The two upstream paths differ:

    * Exact Newton (``solve_residual_equation_exactly_newton``):
      ``norm = np.linalg.norm(b)`` where
      ``b = concat(r[mask], [label residual], [z0 if not stellsym])``.
      We reconstruct ``b`` from ``res['residual']`` (raw ``r``), ``res['mask']``,
      ``label.J() - targetlabel`` and (optionally) ``surface.gamma()[0,0,2]``.

    * LS Newton (``minimize_boozer_penalty_constraints_newton``):
      ``norm = np.linalg.norm(dval)`` where ``dval`` is the gradient of the
      scalarized least-squares objective. The 2-norm of ``res['residual']``
      itself is *not* what the solver checks — at LS convergence the residual
      sits at a non-zero minimum even while the gradient vanishes. The
      Newton LS path persists the gradient under ``res['jacobian']``
      (boozersurface.py:635-653); the LBFGS path persists it under
      ``res['gradient']`` (boozersurface.py:565-570). Both are gradient
      vectors of the scalarized LS objective — semantically equivalent for
      the trust-gate norm. We resolve which key is populated for the
      specific solve via explicit key presence (no silent fallback).

    Returns ``None`` when the relevant vector cannot be read so the trust
    gate can fail closed.
    """

    res = _boozer_result_state(boozer_surface)
    if not res:
        return None
    kind = _boozer_solve_kind(boozer_surface)
    if kind == "ls":
        # LS-family success is ``||grad|| <= newton_tol``. Two upstream
        # writers persist the gradient under different keys:
        #   - LS Newton (``minimize_boozer_penalty_constraints_newton``):
        #     ``res['jacobian'] = dval`` (boozersurface.py:653).
        #   - LBFGS (``minimize_boozer_penalty_constraints_LBFGS``):
        #     ``res['gradient'] = res.jac`` (boozersurface.py:570).
        # No ``res.get(..., res.get(...))`` fallback: pick by key presence
        # so a future writer that populates neither (or both) surfaces
        # explicitly as ``None`` instead of silently degrading.
        has_jacobian = "jacobian" in res and res.get("jacobian") is not None
        has_gradient = "gradient" in res and res.get("gradient") is not None
        if has_jacobian:
            gradient_vector = res["jacobian"]
        elif has_gradient:
            gradient_vector = res["gradient"]
        else:
            return None
        return float(
            np.linalg.norm(np.asarray(gradient_vector, dtype=float).ravel())
        )
    # Exact-Newton path: ``res['residual']`` holds the unmasked raw Boozer
    # residual. Reconstruct the constrained ``b`` vector used by the success
    # check (boozersurface.py:980-984).
    raw_residual = res.get("residual")
    if raw_residual is None:
        return None
    residual_array = np.asarray(raw_residual, dtype=float).ravel()
    surface = getattr(boozer_surface, "surface", None)
    if surface is None:
        return None
    mask = res.get("mask")
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool).ravel()
    if mask_array.size != residual_array.size:
        return None
    masked = residual_array[mask_array]
    label = getattr(boozer_surface, "label", None)
    target_label = getattr(boozer_surface, "targetlabel", None)
    if label is None or target_label is None:
        return None
    label_residual = float(label.J()) - float(target_label)
    components = [masked, np.array([label_residual], dtype=float)]
    if not getattr(surface, "stellsym", True):
        components.append(
            np.array([float(surface.gamma()[0, 0, 2])], dtype=float)
        )
    constrained = np.concatenate(components)
    return float(np.linalg.norm(constrained))


def _iota_is_physically_plausible(iota: float | None) -> bool:
    if iota is None:
        return False
    if not np.isfinite(iota):
        return False
    return abs(float(iota)) < BOOZER_TRUST_IOTA_ABS_BOUND


def compute_boozer_trust_state(
    boozer_surface,
    *,
    solve_success: bool,
    self_intersecting: bool | None,
) -> BoozerTrustState:
    """Compute the SSOT Boozer trust state for a single solve.

    ``solve_success`` and ``self_intersecting`` come from the existing failure-
    policy plumbing; the residual norm is reconstructed from ``boozer_surface``
    so callers do not need to plumb it through the call stack.
    """

    trust_tol = boozer_trust_tolerance(boozer_surface)
    if not bool(solve_success):
        return BoozerTrustState(
            solve_success=False,
            self_intersecting=self_intersecting,
            constrained_residual_norm=None,
            trust_tol=trust_tol,
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason=BOOZER_TRUST_REASON_SOLVE_FAILED,
        )
    if self_intersecting is True:
        return BoozerTrustState(
            solve_success=True,
            self_intersecting=True,
            constrained_residual_norm=None,
            trust_tol=trust_tol,
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason=BOOZER_TRUST_REASON_SELF_INTERSECTING,
        )
    residual_norm = compute_boozer_constrained_residual_norm(boozer_surface)
    if residual_norm is None or not np.isfinite(residual_norm):
        return BoozerTrustState(
            solve_success=True,
            self_intersecting=self_intersecting,
            constrained_residual_norm=residual_norm,
            trust_tol=trust_tol,
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason=BOOZER_TRUST_REASON_RESIDUAL_UNAVAILABLE,
        )
    if residual_norm > trust_tol:
        return BoozerTrustState(
            solve_success=True,
            self_intersecting=self_intersecting,
            constrained_residual_norm=residual_norm,
            trust_tol=trust_tol,
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason=BOOZER_TRUST_REASON_RESIDUAL_TOO_LARGE,
        )
    solved_iota = _coerce_boozer_scalar(_boozer_result_state(boozer_surface).get("iota"))
    if not _iota_is_physically_plausible(solved_iota):
        return BoozerTrustState(
            solve_success=True,
            self_intersecting=self_intersecting,
            constrained_residual_norm=residual_norm,
            trust_tol=trust_tol,
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason=BOOZER_TRUST_REASON_IOTA_NONPHYSICAL,
        )
    return BoozerTrustState(
        solve_success=True,
        self_intersecting=self_intersecting,
        constrained_residual_norm=residual_norm,
        trust_tol=trust_tol,
        boozer_trusted=True,
        iota_objective_active=True,
        trust_reason=BOOZER_TRUST_REASON_OK,
    )


def boozer_trust_state_from_initialization(
    initialization: BoozerInitializationResult,
) -> BoozerTrustState:
    """Compute the trust state for a one-shot Boozer initialization probe.

    Used by ``classify_bootability_result()`` so the bootability sidecar can
    report the same trust signal that the runtime evaluator uses.
    """

    boozer_surface = initialization.boozer_surface
    if boozer_surface is None:
        return BoozerTrustState(
            solve_success=bool(initialization.solve_success),
            self_intersecting=initialization.self_intersecting,
            constrained_residual_norm=None,
            trust_tol=float("nan"),
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason=BOOZER_TRUST_REASON_SOLVE_FAILED,
        )
    return compute_boozer_trust_state(
        boozer_surface,
        solve_success=bool(initialization.solve_success),
        self_intersecting=initialization.self_intersecting,
    )


def _trust_state_bool_field(value: bool | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def boozer_trust_artifact_fields(
    trust_state: BoozerTrustState | None,
) -> dict[str, object]:
    """Build the artifact-key payload for a Boozer trust state.

    Always returns the five top-level trust keys (with ``None`` placeholders
    when no solve has been attempted), so the artifact contract is identical
    across the off / soft / probe lanes and registry ingest never sees a
    missing key.
    """

    if trust_state is None:
        return {
            "BOOZER_SOLVE_SUCCESS": None,
            "BOOZER_SELF_INTERSECTING": None,
            "BOOZER_CONSTRAINED_RESIDUAL_NORM": None,
            "BOOZER_TRUSTED": None,
            "IOTA_OBJECTIVE_ACTIVE": None,
            "BOOZER_TRUST_REASON": None,
            "BOOZER_TRUST_TOL": None,
        }
    residual_norm = trust_state.constrained_residual_norm
    return {
        "BOOZER_SOLVE_SUCCESS": _trust_state_bool_field(trust_state.solve_success),
        "BOOZER_SELF_INTERSECTING": _trust_state_bool_field(
            trust_state.self_intersecting
        ),
        "BOOZER_CONSTRAINED_RESIDUAL_NORM": (
            None
            if residual_norm is None or not np.isfinite(residual_norm)
            else float(residual_norm)
        ),
        "BOOZER_TRUSTED": _trust_state_bool_field(trust_state.boozer_trusted),
        "IOTA_OBJECTIVE_ACTIVE": _trust_state_bool_field(
            trust_state.iota_objective_active
        ),
        "BOOZER_TRUST_REASON": trust_state.trust_reason,
        "BOOZER_TRUST_TOL": (
            None
            if not np.isfinite(trust_state.trust_tol)
            else float(trust_state.trust_tol)
        ),
    }


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


def _volume_label_value(volume_label) -> float:
    return float(volume_label.J())


def _surface_dofs(surface) -> np.ndarray:
    get_dofs = getattr(surface, "get_dofs", None)
    if callable(get_dofs):
        return np.asarray(get_dofs(), dtype=float)
    if hasattr(surface, "local_full_x"):
        return np.asarray(surface.local_full_x, dtype=float)
    return np.asarray(surface.dofs, dtype=float)


def _assign_surface_dofs(surface, dofs) -> None:
    resolved_dofs = np.asarray(dofs, dtype=float)
    if hasattr(surface, "local_full_x"):
        surface.local_full_x = resolved_dofs
        return
    set_dofs = getattr(surface, "set_dofs", None)
    if callable(set_dofs):
        set_dofs(resolved_dofs)
        return
    surface.dofs = resolved_dofs


def _seed_surface_from_guess(surface, initial_surface_guess) -> None:
    guess_dofs = _surface_dofs(initial_surface_guess)
    if len(guess_dofs) == len(_surface_dofs(surface)):
        _assign_surface_dofs(surface, guess_dofs)
        return
    gamma_fn = getattr(initial_surface_guess, "gamma", None)
    if callable(gamma_fn):
        surface.least_squares_fit(gamma_fn())
        return
    # Dof-only guess with different DOF count: fall back to direct assignment.
    # Real simsopt Surface targets reject mismatched lengths via local_full_x;
    # dof-only fakes accept whatever the assignment helper writes.
    _assign_surface_dofs(surface, guess_dofs)


def _snapshot_boozer_initialization_state(
    boozer_surface,
) -> _BoozerInitializationState:
    res = boozer_surface.res
    return _BoozerInitializationState(
        surface_dofs=_surface_dofs(boozer_surface.surface).copy(),
        res=None if res is None else dict(res),
        need_to_run_code=bool(boozer_surface.need_to_run_code),
    )


def _restore_boozer_initialization_state(
    boozer_surface,
    snapshot: _BoozerInitializationState,
) -> None:
    _assign_surface_dofs(boozer_surface.surface, snapshot.surface_dofs)
    boozer_surface.res = None if snapshot.res is None else dict(snapshot.res)
    boozer_surface.need_to_run_code = snapshot.need_to_run_code


def _check_self_intersection_after_initialization(
    boozer_surface,
    pre_solve_state: _BoozerInitializationState,
) -> bool:
    self_intersecting, check_completed = surface_self_intersection_status(
        boozer_surface.surface
    )
    if not check_completed:
        _restore_boozer_initialization_state(boozer_surface, pre_solve_state)
    return self_intersecting


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
    artifact_loader=load_boozer_finite_i,
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
    boozer_surface_cls=BoozerSurfaceFiniteI,
) -> BoozerInitializationResult:
    surf = surface_cls(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_theta=surf_prev.quadpoints_theta,
        quadpoints_phi=surf_prev.quadpoints_phi,
    )
    if initial_surface_guess is None:
        surf.least_squares_fit(surf_prev.gamma())
    else:
        _seed_surface_from_guess(surf, initial_surface_guess)

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
        )
        _assign_surface_dofs(surf_exact, _surface_dofs(surf))
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

    pre_solve_state = _snapshot_boozer_initialization_state(boozer_surface)
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
            volume=_volume_label_value(vol),
            error_type=solve_attempt.error_type,
            error_message=solve_attempt.error_message,
        )

    solve_success = solve_attempt.solve_success
    self_intersecting = _check_self_intersection_after_initialization(
        boozer_surface,
        pre_solve_state,
    )
    return BoozerInitializationResult(
        boozer_surface=boozer_surface,
        solve_success=solve_success,
        self_intersecting=self_intersecting,
        success=solve_success and not self_intersecting,
        solved_iota=solve_attempt.solved_iota,
        solved_G=solve_attempt.solved_G,
        volume=_volume_label_value(vol),
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
    boozer_surface_cls=BoozerSurfaceFiniteI,
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
    trust_state: BoozerTrustState | None = None,
) -> dict[str, object]:
    abs_iota_error = None
    if solved_iota is not None:
        abs_iota_error = abs(float(solved_iota) - float(target_iota))
    payload = {
        "BOOZER_BOOTABLE": False,
        "IOTA_NEAR_TARGET": False,
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
    payload.update(boozer_trust_artifact_fields(trust_state))
    return payload


def classify_bootability_result(
    initialization: BoozerInitializationResult,
    *,
    stage: str,
    target_iota: float,
    iota_tolerance: float,
) -> dict[str, object]:
    trust_state = boozer_trust_state_from_initialization(initialization)
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
            trust_state=trust_state,
        )
    if initialization.self_intersecting:
        return _bootability_failure(
            stage=stage,
            target_iota=target_iota,
            reason=BOOTABILITY_REASON_SELF_INTERSECTION,
            solved_iota=initialization.solved_iota,
            self_intersecting=True,
            solve_success=True,
            trust_state=trust_state,
        )
    solved_iota = initialization.solved_iota
    if solved_iota is None:
        return _bootability_failure(
            stage=stage,
            target_iota=target_iota,
            reason=BOOTABILITY_REASON_BOOZER_SOLVE_FAILED,
            solve_success=True,
            trust_state=trust_state,
        )
    abs_iota_error = abs(float(solved_iota) - float(target_iota))
    iota_near_target = abs_iota_error <= float(iota_tolerance)
    bootability_reason = (
        BOOTABILITY_REASON_OK
        if iota_near_target
        else BOOTABILITY_REASON_IOTA_MISMATCH
    )
    payload = {
        "BOOZER_BOOTABLE": True,
        "IOTA_NEAR_TARGET": iota_near_target,
        "IOTA_FEASIBLE": iota_near_target,
        "BOOTABILITY_REASON": bootability_reason,
        "BOOTABILITY_STAGE": stage,
        "BOOTABILITY_TARGET_IOTA": float(target_iota),
        "BOOTABILITY_SOLVED_IOTA": float(solved_iota),
        "BOOTABILITY_SELF_INTERSECTING": False,
        "BOOTABILITY_SOLVE_SUCCESS": True,
        "BOOTABILITY_ABS_IOTA_ERROR": abs_iota_error,
        "BOOTABILITY_ERROR_TYPE": None,
        "BOOTABILITY_ERROR_MESSAGE": None,
    }
    payload.update(boozer_trust_artifact_fields(trust_state))
    return payload


def bootability_passes(bootability_status: Mapping[str, object]) -> bool:
    return bootability_status.get("BOOZER_BOOTABLE") is True


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
    bs_loader=load_boozer_finite_i,
    warm_start_loader=load_boozer_finite_i,
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
            artifact_loader=warm_start_loader,
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
