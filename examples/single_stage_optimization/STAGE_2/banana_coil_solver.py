import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

# SIMSOPT imports
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, apply_symmetries_to_curves
from simsopt.geo import (
    curves_to_vtk,
    create_equally_spaced_curves,
    CurveLength,
    CurveCurveDistance,
    CurveCWSFourierCPP,
    LpCurveCurvature,
)
from simsopt.geo.curveobjectives import CurveSurfaceDistance
from simsopt.geo.framedcurve import (
    FramedCurveSurfaceTangent,
    surface_tangent_normal_direction,
)
from simsopt.geo.strain_optimization import LPTorsionalStrainPenalty
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt import load as simsopt_load

from alm_utils import (
    minimize_alm,
    run_directional_taylor_test,
    validate_alm_cli_args,
)
from plotting_utils import cross_section_plot
from workflow_helpers import (
    Stage2SeedSpec,
    canonical_stage2_iota_constraint_weight,
    format_local_stage2_run_dir,
    resolve_finite_current_vf_template_path,
    validate_stage2_iota_args,
    validate_normalized_toroidal_flux,
)
from workflow_runner_common import (
    load_stage2_artifact_results,
    write_json,
)
from banana_opt.json_compat import load_boozer_finite_i
from banana_opt.artifact_contracts import (
    STAGE2_BS_SHA256_KEY,
    compute_stage2_bs_sha256,
    upgrade_legacy_stage2_artifact_results,
)
from banana_opt.alm_adaptive_smoothing import (
    ALM_SMOOTHING_FLOOR_FRACTION,
    adapt_alm_smoothing_from_history,
)
from banana_opt.alm_defaults import stage2_alm_default
from banana_opt.constraint_contract import (
    build_constraint_metadata,
    resolve_constraint_contract_from_wire_names,
)
from banana_opt.coil_order_upgrade import (
    realized_cws_winding_radii,
    upgrade_loaded_seed_biot_savart_order,
)
from banana_opt.coil_groups import (
    COIL_GROUP_ROLE_BANANA,
    CoilGroupsManifest,
)
from banana_opt.edge_delivered_iota import (
    DEFAULT_EDGE_TRACE_TURNS,
    EDGE_HELICITY_STATUS_UNKNOWN,
    EDGE_IOTA_MODE_OFF,
    EDGE_IOTA_MODE_REPORT,
    EDGE_IOTA_MODE_SOFT,
    EdgeIotaConfig,
    edge_iota_config_hash,
    edge_iota_report_payload,
    evaluate_edge_iota_profile,
    load_lcfs_boundary,
    read_eqdsk,
    validate_edge_iota_config,
    validate_tokamak_iota_against_q,
    write_profile_json,
)
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.basin_hopping import (
    run_basin_hopping,
    telemetry_values as basin_telemetry_values,
)
from banana_opt.stage2_geometry import (
    FiniteBuildSettings,
    VFCoilBuildResult,
    build_finite_build_banana_coils,
    coerce_vf_coil_build_result,
    configure_winding_surface_shape_dofs,
    curve_curve_min_distance_segments_m,
    curve_surface_min_distance_segments_m,
    finite_build_frame_aware_curvature_limit_inv_m,
    initialize_coils as _initialize_coils,
    is_self_intersecting,
    load_plasma_geometry_for_working_major_radius,
    load_vmec_surface as _load_stage2_vmec_surface,
    load_plasma_geometry as _load_plasma_geometry,
    magnetic_field_plots as _magnetic_field_plots,
    rotation_aware_curvature_report,
    rotation_aware_projected_half_extent_m,
    select_plasma_geometry_preflight_candidate,
    shared_vf_current_control_for_coils,
    surface_surface_min_distance as _surface_surface_min_distance,
    WINDING_DOF_CORRIDOR_SCALE_MAP,
)
from banana_opt.single_stage_geometry import (
    build_winding_dof_scale_vector,
    run_scaled_winding_minimize,
)
from banana_opt.hardware_contracts import (
    BANANA_CC_OBJECTIVE_MARGIN_M,
    BANANA_CURRENT_HARD_LIMIT_A,
    BANANA_FOLD_GEODESIC_CURVATURE_LIMIT_INV_M,
    BANANA_FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION,
    BANANA_SELF_DISTANCE_WINDOW_M,
    BANANA_SELF_ENVELOPE_GROC_RADIUS_FLOOR_M,
    BANANA_SELF_ENVELOPE_MIN_DISTANCE_M,
    BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
    BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR,
    BANANA_WIDTH_MAX_M,
    BANANA_WIDTH_MIN_M,
    BANANA_WINDING_MINOR_RADIUS_M,
    BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_HARD_LIMIT_M,
    COIL_LENGTH_MIN_FRACTION,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    HARDWARE_KEEPOUT_MIN_DISTANCE_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    POLOIDAL_EXTENT_HALF_WIDTH_RAD,
    STAGE2_HARDWARE_KEEPOUT_WEIGHT_DEFAULT,
    STAGE2_POLOIDAL_WEIGHT_DEFAULT,
    STAGE2_SELF_INTERSECT_WEIGHT_DEFAULT,
    STAGE2_VESSEL_KEEPOUT_WEIGHT_DEFAULT,
    STAGE2_WIDTH_WEIGHT_DEFAULT,
    TF_CURRENT_CW_DEFAULT_A,
    TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    TARGET_LCFS_MAX_MINOR_RADIUS_M,
    TF_CURRENT_HARD_LIMIT_A,
    TYPE_KK_FINITE_BUILD_GAPSIZE_B_M,
    TYPE_KK_FINITE_BUILD_GAPSIZE_N_M,
    TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B,
    TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N,
    TYPE_KK_INNER_RADIUS_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    required_banana_cc_centerline_m,
    validate_banana_winding_surface_radius,
    validate_coil_length_target,
    validate_major_radius,
    validate_target_lcfs_major_radius,
    validate_target_lcfs_minor_radius,
    validate_tf_current_limit,
)
from banana_opt.hardware_constraint_schema import (
    build_bootability_recovery_payload_fields,
    hardware_constraint_alm_names,
)
from banana_opt.hardware_keepout import (
    CurveHardwareKeepout,
    CurveHardwareSdfFreeSpaceReward,
    CurveHardwareSdfKeepout,
    CurveVesselAvailableEnvelopeReward,
    CurveVesselEnvelopeKeepout,
    hardware_keepout_metadata,
    hardware_keepout_results_fields,
    hardware_sdf_metadata_from_data,
    live_winding_r0,
    load_hardware_keepout,
    load_hardware_sdf,
)
from banana_opt.stage2_resonant_flux import (
    MAX_RESONANT_DENOMINATOR,
    build_stage2_resonant_flux_penalty,
    enumerate_resonant_rationals,
)
from banana_opt.lbfgsb_defaults import DEFAULT_LBFGSB_MAXCOR
from banana_opt.current_contracts import (
    BoozerCurrentConvention,
    apply_vf_current_upper_bound,
    apply_penalty_traversal_forbidden_box_bounds,
    DEFAULT_FINITE_CURRENT_MODE,
    FiniteCurrentMode,
    physical_current_to_boozer_I,
    resolve_boozer_current_convention,
    resolve_finite_current_mode,
    resolve_jhalpern30_fresh_vf_current_A,
    unwrap_current_optimizable,
    validate_proxy_vf_current_convention_for_mode,
)
from banana_opt.design_only_fields import build_design_only_results_fields
from banana_opt.finite_current_profiles import (
    FINITE_CURRENT_PROFILES,
    JHALPERN30_FINITE_CURRENT_MODE,
    get_finite_current_profile,
)
from banana_opt.jhalpern30_compat import (
    jhalpern30_iota_target_sign,
    resolve_jhalpern30_banana_current_replay,
    sha256_file,
)
from banana_opt.stage2_single_stage_handoff import (
    boozer_trust_artifact_fields,
    partition_loaded_stage2_coils,
    probe_stage2_seed_bootability,
)
from banana_opt.boozer_topology_bridge import (
    HelicalFieldContentObjective,
    boozer_topology_bridge_artifact_fields,
    safe_compute_helical_field_content_S_HEL,
)
from banana_opt.boozer_warm_start import save_boozer_surface_with_state
from banana_opt.topology_bridge import (
    DEFAULT_NFIELDLINES as TOPOLOGY_BRIDGE_DEFAULT_NFIELDLINES,
    DEFAULT_TMAX as TOPOLOGY_BRIDGE_DEFAULT_TMAX,
    DEFAULT_TOL as TOPOLOGY_BRIDGE_DEFAULT_TOL,
    fieldline_iota_proxy_artifact_fields,
    safe_compute_fieldline_iota_proxy as safe_compute_phase3a_fieldline_iota_proxy,
)
from banana_opt.wout_convention import wout_convention_artifact_fields
from topology_scorer import padded_bounds
from banana_opt.stage2_objectives import (
    build_stage2_alm_settings,
    build_stage2_iota_runtime,
    build_stage2_results as _build_stage2_results_impl,
    evaluate_stage2_alm_problem as _evaluate_stage2_alm_problem,
    evaluate_stage2_hardware_constraints as _evaluate_stage2_hardware_constraints,
    evaluate_stage2_iota_state,
    make_stage2_fun,
    smooth_min_curve_surface_signed_constraint,
    smooth_max_curvature_signed_constraint,
    smooth_min_distance_signed_constraint,
    stage2_constraint_activity_tolerances,
    validate_stage2_coil_partition_counts,
)
from banana_opt.ellipse_width import ProjectedEllipseWidth
from banana_opt.fold_buildability import CurveSurfaceGeodesicCurvature
from banana_opt.poloidal_extent import (
    PoloidalExtent,
    max_poloidal_extent_rad,
    smooth_max_poloidal_extent_signed_constraint,
)
from banana_opt.self_intersect import (
    CurveGlobalRadiusOfCurvature,
    CurveSelfDistance,
    CurveSelfIntersect,
)

REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = (
    DATABASE_EQUILIBRIA_DIR
    if os.path.isdir(DATABASE_EQUILIBRIA_DIR)
    else os.path.join(EXAMPLE_ROOT, "equilibria")
)
# Same exported shells/sensors/solenoid/REMC/limiter/quartz point cloud the
# single-stage path consumes (SINGLE_STAGE DEFAULT_HARDWARE_KEEPOUT_JSON_PATH).
DEFAULT_HARDWARE_KEEPOUT_JSON_PATH = os.path.join(
    REPO_ROOT,
    "CAD",
    "banana_coils",
    "hbt_clearance_viewer",
    "tools",
    "hardware_keepout.json",
)
DEFAULT_HARDWARE_KEEPOUT_GLB_PATH = os.path.join(
    REPO_ROOT,
    "CAD",
    "banana_coils",
    "hbt_assembly.glb",
)
DEFAULT_STAGE2_IOTA_TOLERANCE = 5.0e-3
DEFAULT_STAGE2_IOTA_VOL_TARGET = 0.10
DEFAULT_STAGE2_IOTA_CONSTRAINT_WEIGHT = 1.0
DEFAULT_STAGE2_IOTA_NUM_TF_COILS = 20
DEFAULT_STAGE2_IOTA_NPHI = 91
DEFAULT_STAGE2_IOTA_NTHETA = 32
DEFAULT_STAGE2_IOTA_MPOL = 8
DEFAULT_STAGE2_IOTA_NTOR = 6
SECONDARY_STAGE2_ARTIFACT_REASON = "exact_hardware_pass_iota_fail"
SECONDARY_STAGE2_ARTIFACT_DIRNAME = "secondary_exact_hardware_pass_iota_fail"
SECONDARY_STAGE2_TERMINATION_SUFFIX = (
    "preserved_secondary_exact_hardware_pass_iota_fail"
)


@dataclass(frozen=True)
class Stage2FiniteCurrentConfig:
    finite_current_mode: FiniteCurrentMode
    proxy_plasma_current_A: float
    vf_current_A: float
    vf_current_max_A: float
    vf_template_path: str | None
    boozer_current_convention: BoozerCurrentConvention


def stage2_alm_constraint_names(
    *,
    include_coil_surface: bool,
    include_poloidal_extent: bool = False,
    include_hardware_keepout: bool = False,
    include_iota_penalty: bool = False,
) -> tuple[str, ...]:
    available_names = {
        "coil_length",
        "coil_length_min",
        "coil_coil_spacing",
        "max_curvature",
        "banana_current",
        "width_min",
        "width_max",
        "self_intersect",
    }
    if include_coil_surface:
        available_names.add("coil_surface_spacing")
    if include_poloidal_extent:
        available_names.add("poloidal_extent")
    if include_hardware_keepout:
        available_names.add("hardware_keepout")
    constraint_names = list(hardware_constraint_alm_names(names=available_names))
    if include_iota_penalty:
        constraint_names.append("iota_penalty")
    return tuple(constraint_names)


def _print_taylor_test_summary(name: str, result: dict) -> None:
    max_ratio = result["max_ratio"]
    max_ratio_str = "n/a" if max_ratio is None else f"{max_ratio:.3e}"
    print(
        f"[{name}] passed={result['passed']}, "
        f"directional_derivative={result['directional_derivative']:.6e}, "
        f"max_ratio={max_ratio_str}"
    )
    for epsilon, error in zip(result["epsilons"], result["errors"]):
        print(f"[{name}] eps={epsilon:.3e}, err={error:.3e}")


def validate_banana_current_cli_args(args) -> None:
    banana_init_current_A = float(args.banana_init_current_A)
    banana_current_max_A = float(args.banana_current_max_A)
    accepts_offspec_sign = bool(args.accept_offspec_banana_current_sign)
    accepts_offspec_current_max = bool(args.accept_offspec_banana_current_max)
    if accepts_offspec_sign:
        if (
            banana_init_current_A == 0.0
            or abs(banana_init_current_A) > BANANA_CURRENT_HARD_LIMIT_A
        ):
            raise ValueError(
                f"--banana-init-current-A must be non-zero with magnitude <= "
                f"{BANANA_CURRENT_HARD_LIMIT_A:.0f}."
            )
    elif not (-BANANA_CURRENT_HARD_LIMIT_A <= banana_init_current_A < 0.0):
        raise ValueError(
            f"--banana-init-current-A must be in the interval "
            f"[-{BANANA_CURRENT_HARD_LIMIT_A:.0f}, 0)."
        )
    if banana_current_max_A <= 0.0:
        raise ValueError("--banana-current-max-A must be positive.")
    if (
        banana_current_max_A > BANANA_CURRENT_HARD_LIMIT_A
        and not accepts_offspec_current_max
    ):
        raise ValueError(
            f"--banana-current-max-A must be in the interval "
            f"(0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}]."
        )
    if abs(banana_init_current_A) > banana_current_max_A:
        raise ValueError(
            "abs(--banana-init-current-A) cannot exceed --banana-current-max-A."
        )


def validate_finite_build_cli_args(args) -> None:
    if not getattr(args, "finite_build", False):
        if getattr(args, "finitebuild_frame_aware_curvature_threshold", None) is True:
            raise ValueError(
                "--finitebuild-frame-aware-curvature-threshold is incompatible "
                "with --filament-only (the frame-aware limit is derived from "
                "finite-build winding-pack geometry)."
            )
        for flag_attr, flag_name in (
            (
                "stage2_couple_pack_rotation_to_fold",
                "--stage2-couple-pack-rotation-to-fold",
            ),
            (
                "stage2_rotation_aware_curvature_cap",
                "--stage2-rotation-aware-curvature-cap",
            ),
        ):
            if bool(getattr(args, flag_attr, False)):
                raise ValueError(
                    f"{flag_name} requires --finite-build (it acts on the "
                    "winding-pack rotation frame, which only exists for a "
                    "finite-build pack)."
                )
        if float(getattr(args, "stage2_pack_twist_strain_weight", 0.0)) > 0.0:
            raise ValueError(
                "--stage2-pack-twist-strain-weight requires --finite-build "
                "(the torsional-strain regularizer acts on the pack frame)."
            )
        return
    # --finite-build + --stage2-bs-path is a warm start: the multi-filament pack
    # is built from the seed's master banana curve and seed banana current
    # (load_stage2_seed_configuration), not a fresh circle. The jhalpern30
    # current path has no finite-build pack, so reject that combination only.
    if (
        getattr(args, "stage2_bs_path", None)
        and getattr(args, "finite_current_mode", DEFAULT_FINITE_CURRENT_MODE)
        == JHALPERN30_FINITE_CURRENT_MODE
    ):
        raise ValueError(
            "--finite-build is not supported with a jhalpern30 --stage2-bs-path "
            "seed; the jhalpern30 banana current path has no finite-build pack."
        )
    if int(args.finitebuild_numfilaments_n) <= 0:
        raise ValueError("--finitebuild-numfilaments-n must be positive.")
    if int(args.finitebuild_numfilaments_b) <= 0:
        raise ValueError("--finitebuild-numfilaments-b must be positive.")
    gapsize_n = float(args.finitebuild_gapsize_n)
    gapsize_b = float(args.finitebuild_gapsize_b)
    if not np.isfinite(gapsize_n) or gapsize_n <= 0.0:
        raise ValueError("--finitebuild-gapsize-n must be positive and finite.")
    if not np.isfinite(gapsize_b) or gapsize_b <= 0.0:
        raise ValueError("--finitebuild-gapsize-b must be positive and finite.")
    if (
        getattr(args, "finitebuild_pin_current", False)
        and getattr(args, "constraint_method", "penalty") == "alm"
    ):
        raise ValueError(
            "--finitebuild-pin-current is supported only with "
            "--constraint-method penalty (the pinned current DOF has no box bound)."
        )
    if bool(
        getattr(args, "stage2_rotation_aware_curvature_cap", False)
    ) and not bool(getattr(args, "stage2_couple_pack_rotation_to_fold", False)):
        raise ValueError(
            "--stage2-rotation-aware-curvature-cap requires "
            "--stage2-couple-pack-rotation-to-fold: relaxing the in-run cap to "
            "the realized rotation-aware value is only coherent when the pack "
            "twist alpha(theta) is actually driven by buildability."
        )


def validate_winding_surface_shape_cli_args(args) -> None:
    free_mpol = int(getattr(args, "winding_surface_free_mpol", 0))
    free_ntor = int(getattr(args, "winding_surface_free_ntor", 0))
    if free_mpol < 0:
        raise ValueError("--winding-surface-free-mpol must be non-negative.")
    if free_ntor < 0:
        raise ValueError("--winding-surface-free-ntor must be non-negative.")
    # The size DOFs (--winding-surface-free-r0 / --winding-surface-free-minor)
    # re-center / re-size the winding surface and ARE valid on a loaded seed:
    # re-centering an already-converged seed is their intended use (T1.5). Only
    # the shape-mode frees stay fresh-only -- a loaded seed's recorded (m, n)
    # mode content is what its coils converged against, so reopening those modes
    # mid-resume has no consistent meaning.
    if getattr(args, "stage2_bs_path", None) and (free_mpol > 0 or free_ntor > 0):
        raise ValueError(
            "--winding-surface-free-mpol / --winding-surface-free-ntor require "
            "fresh Stage 2 initialization; a loaded seed preserves its recorded "
            "shape modes. Use --winding-surface-free-r0 / "
            "--winding-surface-free-minor to re-center or re-size a loaded seed."
        )


def free_loaded_winding_surface_size_dofs(banana_curve, args):
    """Free the bounded winding-surface SIZE DOFs on a LOADED master surface (T1.5).

    The loaded-seed path skips ``_initialize_coils``, so the size-DOF freeing that
    re-centers (``rc(0,0)``) / re-sizes (``rc(1,0)`` / ``zs(1,0)``) a resumed
    winding surface happens here instead, directly on ``banana_curve.surf`` -- the
    same master torus the value-live steering terms read (``live_winding_r0``) and
    the ALM seed-clip bounds. Only ``free_r0`` / ``free_minor`` reach this point;
    ``validate_winding_surface_shape_cli_args`` rejects shape-mode frees on loaded
    seeds. Returns the freed DOF names, or ``()`` when neither lever is requested
    so the default-off loaded path stays byte-identical (no surface mutation).
    """
    free_r0 = bool(getattr(args, "winding_surface_free_r0", False))
    free_minor = bool(getattr(args, "winding_surface_free_minor", False))
    if not (free_r0 or free_minor):
        return ()
    return configure_winding_surface_shape_dofs(
        banana_curve.surf,
        free_r0=free_r0,
        free_minor=free_minor,
    )


def validate_stage2_vessel_keepout_cli_args(args) -> None:
    if float(args.stage2_vessel_keepout_weight) < 0.0:
        raise ValueError("--stage2-vessel-keepout-weight must be >= 0.")
    if float(args.stage2_available_envelope_reward_weight) < 0.0:
        raise ValueError(
            "--stage2-available-envelope-reward-weight must be >= 0."
        )
    if float(args.stage2_hardware_sdf_free_space_reward_weight) < 0.0:
        raise ValueError(
            "--stage2-hardware-sdf-free-space-reward-weight must be >= 0."
        )
    if float(args.stage2_hardware_sdf_free_space_reward_weight) > 0.0:
        if args.stage2_hardware_keepout_backend != "sdf":
            raise ValueError(
                "--stage2-hardware-sdf-free-space-reward-weight requires "
                "--stage2-hardware-keepout-backend=sdf."
            )
        if not args.stage2_hardware_keepout_sdf_manifest:
            raise ValueError(
                "--stage2-hardware-sdf-free-space-reward-weight requires "
                "--stage2-hardware-keepout-sdf-manifest."
            )


def resolve_stage2_resonant_iota_target(args):
    """Explicit config resolution for the audit-8 resonant flux reweighting:
    the dedicated --stage2-resonant-iota-target wins; otherwise the existing
    Stage-2 iota-target plumbing (--stage2-iota-target / STAGE2_IOTA_TARGET)
    is reused so iota-targeted lanes opt in with a single weight flag.
    Returns None when neither is configured (the validator rejects that
    combination whenever the weight is nonzero)."""
    if args.stage2_resonant_iota_target is not None:
        return float(args.stage2_resonant_iota_target)
    if args.stage2_iota_target is not None:
        return float(args.stage2_iota_target)
    return None


def resolve_stage2_resonant_qmax(args) -> int:
    q_max = int(
        getattr(
            args,
            "stage2_resonant_qmax",
            MAX_RESONANT_DENOMINATOR,
        )
    )
    if q_max < 1:
        raise ValueError(f"--stage2-resonant-qmax must be >= 1; got {q_max}.")
    if q_max > MAX_RESONANT_DENOMINATOR:
        raise ValueError(
            f"--stage2-resonant-qmax must be <= {MAX_RESONANT_DENOMINATOR}; "
            f"got {q_max}."
        )
    return q_max


def validate_stage2_resonant_flux_cli_args(args) -> None:
    """Audit-8 CLI contract: weight >= 0; a nonzero weight requires an iota
    target and a non-empty q<=MAX_RESONANT_DENOMINATOR rational window (checked here so a
    misconfigured lane dies at argparse time, not mid-setup)."""
    weight = float(args.stage2_resonant_flux_weight)
    if weight < 0.0:
        raise ValueError("--stage2-resonant-flux-weight must be >= 0.")
    q_max = resolve_stage2_resonant_qmax(args)
    if weight == 0.0:
        return
    iota_target = resolve_stage2_resonant_iota_target(args)
    if iota_target is None:
        raise ValueError(
            "--stage2-resonant-flux-weight > 0 requires "
            "--stage2-resonant-iota-target (or --stage2-iota-target)."
        )
    rationals = enumerate_resonant_rationals(
        iota_target,
        float(args.stage2_resonant_delta),
        q_max,
    )
    if not rationals:
        raise ValueError(
            f"no rationals p/q with q <= {int(args.stage2_resonant_qmax)} lie "
            f"within +/-{float(args.stage2_resonant_delta)} of iota target "
            f"{iota_target}; a nonzero --stage2-resonant-flux-weight would be "
            "a silent no-op."
        )


def stage2_self_envelope_default_floor(mode: str) -> float:
    if mode == "groc":
        return float(BANANA_SELF_ENVELOPE_GROC_RADIUS_FLOOR_M)
    if mode in {"hinge", "off"}:
        return float(BANANA_SELF_ENVELOPE_MIN_DISTANCE_M)
    raise ValueError("--self-envelope-mode must be one of: hinge, groc, off.")


def resolve_stage2_self_envelope_floor(args) -> float:
    mode = str(getattr(args, "self_envelope_mode", "hinge"))
    explicit_floor = getattr(args, "self_envelope_floor", None)
    if explicit_floor is None:
        return stage2_self_envelope_default_floor(mode)
    return float(explicit_floor)


def validate_stage2_buildability_objective_cli_args(args) -> None:
    self_envelope_mode = str(getattr(args, "self_envelope_mode", "hinge"))
    self_envelope_weight = float(getattr(args, "self_envelope_weight", 1.0))
    self_envelope_floor = resolve_stage2_self_envelope_floor(args)
    self_distance_window = float(
        getattr(args, "self_distance_window", BANANA_SELF_DISTANCE_WINDOW_M)
    )
    self_envelope_sampling_margin = float(
        getattr(args, "self_envelope_sampling_margin", 0.0)
    )
    cc_objective_margin = float(
        getattr(args, "cc_objective_margin", BANANA_CC_OBJECTIVE_MARGIN_M)
    )
    fold_weight = float(getattr(args, "fold_weight", 1.0))
    fold_limit = float(
        getattr(
            args,
            "fold_geodesic_curvature_limit",
            BANANA_FOLD_GEODESIC_CURVATURE_LIMIT_INV_M,
        )
    )
    fold_material_limit = getattr(args, "fold_material_binormal_curvature_limit", None)
    fold_margin = float(
        getattr(
            args,
            "fold_geodesic_curvature_margin_fraction",
            BANANA_FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION,
        )
    )
    if self_envelope_mode not in {"hinge", "groc", "off"}:
        raise ValueError("--self-envelope-mode must be one of: hinge, groc, off.")
    if not np.isfinite(self_envelope_weight) or self_envelope_weight < 0.0:
        raise ValueError("--self-envelope-weight must be finite and >= 0.")
    if not np.isfinite(self_envelope_floor) or self_envelope_floor <= 0.0:
        raise ValueError("--self-envelope-floor must be finite and > 0.")
    if not np.isfinite(self_distance_window) or self_distance_window < 0.0:
        raise ValueError("--self-distance-window must be finite and >= 0.")
    if (
        not np.isfinite(self_envelope_sampling_margin)
        or self_envelope_sampling_margin < 0.0
    ):
        raise ValueError("--self-envelope-sampling-margin must be finite and >= 0.")
    if self_envelope_mode != "hinge" and self_envelope_sampling_margin > 0.0:
        raise ValueError(
            "--self-envelope-sampling-margin is only supported with "
            "--self-envelope-mode hinge."
        )
    if not np.isfinite(cc_objective_margin) or cc_objective_margin < 0.0:
        raise ValueError("--cc-objective-margin must be finite and >= 0.")
    if not np.isfinite(fold_weight) or fold_weight < 0.0:
        raise ValueError("--fold-weight must be finite and >= 0.")
    if not np.isfinite(fold_limit) or fold_limit <= 0.0:
        raise ValueError("--fold-geodesic-curvature-limit must be finite and > 0.")
    if fold_material_limit is not None and (
        not np.isfinite(float(fold_material_limit)) or float(fold_material_limit) <= 0.0
    ):
        raise ValueError(
            "--fold-material-binormal-curvature-limit must be finite and > 0."
        )
    if not np.isfinite(fold_margin) or not (0.0 <= fold_margin < 1.0):
        raise ValueError(
            "--fold-geodesic-curvature-margin-fraction must be finite and in [0, 1)."
        )


def build_stage2_resonant_flux_term_if_requested(args, surface, field):
    """Weight-gated audit-8 construction. Returns ``(term, weight, rationals)``;
    the default weight 0.0 returns ``(None, 0.0, ())`` WITHOUT constructing
    any new objects, keeping the legacy objective graph byte-identical."""
    weight = float(args.stage2_resonant_flux_weight)
    if weight < 0.0:
        raise ValueError("--stage2-resonant-flux-weight must be >= 0.")
    q_max = resolve_stage2_resonant_qmax(args)
    if weight == 0.0:
        return None, 0.0, ()
    term, rationals = build_stage2_resonant_flux_penalty(
        surface,
        field,
        iota_target=resolve_stage2_resonant_iota_target(args),
        delta=float(args.stage2_resonant_delta),
        q_max=q_max,
    )
    return term, weight, rationals


def stage2_frame_aware_curvature_tightening(
    curvature_threshold_inv_m,
    finite_build_settings,
    opt_in,
):
    """Audit 5a: tighten the in-run curvature threshold when enabled.

    Returns ``(threshold_inv_m, pack_limit_inv_m, applied)``;
    ``pack_limit_inv_m`` is ``None`` when tightening is off, and ``applied`` is
    False whenever the caller threshold was already at least as strict as the
    pack limit.
    """
    if not opt_in:
        return float(curvature_threshold_inv_m), None, False
    if finite_build_settings is None:
        # Broken contract: the limit derives from the winding-pack
        # geometry, so it cannot be honored without finite-build settings. The
        # CLI gate (validate_finite_build_cli_args) already rejects this combo;
        # raise loudly rather than silently skipping the requested tightening.
        raise ValueError(
            "frame-aware curvature tightening was requested but no finite-build "
            "settings are present; the limit is derived from the winding-pack "
            "geometry and cannot be applied without --finite-build."
        )
    pack_limit_inv_m = finite_build_frame_aware_curvature_limit_inv_m(
        finite_build_settings,
        TYPE_KK_INNER_RADIUS_MARGIN_M,
    )
    if pack_limit_inv_m < float(curvature_threshold_inv_m):
        return pack_limit_inv_m, pack_limit_inv_m, True
    return float(curvature_threshold_inv_m), pack_limit_inv_m, False


def stage2_frame_aware_curvature_threshold_enabled(args) -> bool:
    """Default the Type-KK bend-radius objective on for finite-build runs."""
    raw_value = getattr(args, "finitebuild_frame_aware_curvature_threshold", None)
    if raw_value is None:
        return bool(getattr(args, "finite_build", False))
    return bool(raw_value)


def resolve_finite_build_settings(args):
    """Return a FiniteBuildSettings from CLI args, or None when finite-build is off.

    A negative ``--finitebuild-rotation-order`` maps to ``None`` (pack orientation
    fixed, no rotation DOFs).
    """
    if not getattr(args, "finite_build", False):
        return None
    rotation_order = int(args.finitebuild_rotation_order)
    return FiniteBuildSettings(
        numfilaments_n=int(args.finitebuild_numfilaments_n),
        numfilaments_b=int(args.finitebuild_numfilaments_b),
        gapsize_n=float(args.finitebuild_gapsize_n),
        gapsize_b=float(args.finitebuild_gapsize_b),
        rotation_order=None if rotation_order < 0 else rotation_order,
        frame=str(args.finitebuild_frame),
    )


def validate_vf_current_bound_config(
    *,
    vf_current_A: float,
    vf_current_max_A: float,
    finite_current_mode: FiniteCurrentMode,
) -> float:
    vf_current_max_A = float(vf_current_max_A)
    if not np.isfinite(vf_current_max_A) or vf_current_max_A <= 0.0:
        raise ValueError("--vf-current-max-A must be finite and positive.")
    profile = get_finite_current_profile(finite_current_mode)
    if (
        profile.vf_current_mutability == "shared_unfixed_scaled_current"
        and abs(float(vf_current_A)) > vf_current_max_A
    ):
        raise ValueError(
            "abs(VF_CURRENT_A) cannot exceed --vf-current-max-A for shared "
            "optimizable VF current."
        )
    return vf_current_max_A


def validate_vf_current_optimizer_config(
    *,
    finite_current_mode: FiniteCurrentMode,
    constraint_method: str,
    init_only: bool = False,
) -> None:
    profile = get_finite_current_profile(finite_current_mode)
    if (
        profile.vf_current_mutability == "shared_unfixed_scaled_current"
        and str(constraint_method) == "alm"
        and not bool(init_only)
    ):
        raise ValueError(
            "shared optimizable VF current requires penalty/L-BFGS-B bounds; "
            "Stage 2 ALM does not support VF current bounds."
        )


def validate_stage2_tf_current_value(
    tf_current_A,
    *,
    accepts_offspec_sign: bool,
    accepts_offspec_magnitude: bool,
    field_name: str,
) -> None:
    tf_current_A = float(tf_current_A)
    if accepts_offspec_sign or accepts_offspec_magnitude:
        if not np.isfinite(tf_current_A) or tf_current_A == 0.0:
            raise ValueError(f"{field_name} must be finite and non-zero.")
        if tf_current_A > 0.0 and not accepts_offspec_sign:
            raise ValueError(
                f"Positive {field_name} requires --accept-offspec-tf-current-sign."
            )
        if (
            abs(tf_current_A) > TF_CURRENT_HARD_LIMIT_A
            and not accepts_offspec_magnitude
        ):
            raise ValueError(
                f"|{field_name}| above the hardware limit requires "
                "--accept-offspec-tf-current-magnitude."
            )
        return
    validate_tf_current_limit(tf_current_A)


def validate_stage2_tf_current_cli_args(args) -> None:
    validate_stage2_tf_current_value(
        args.tf_current_A,
        accepts_offspec_sign=bool(args.accept_offspec_tf_current_sign),
        accepts_offspec_magnitude=bool(args.accept_offspec_tf_current_magnitude),
        field_name="--tf-current-A",
    )


def stage2_current_contract_allows_offspec(args) -> bool:
    return (
        bool(args.accept_offspec_tf_current_sign)
        or bool(args.accept_offspec_tf_current_magnitude)
        or bool(args.accept_offspec_banana_current_max)
    )


def stage2_length_contract_allows_offspec(args) -> bool:
    return bool(getattr(args, "accept_offspec_coil_length", False))


def stage2_curvature_contract_allows_offspec(args) -> bool:
    return bool(getattr(args, "accept_offspec_curvature", False))


def validate_stage2_iota_cli_args(args) -> None:
    validate_stage2_iota_args(
        stage2_iota_target=args.stage2_iota_target,
        stage2_iota_tolerance=args.stage2_iota_tolerance,
        stage2_iota_vol_target=args.stage2_iota_vol_target,
        stage2_iota_num_tf_coils=args.stage2_iota_num_tf_coils,
        stage2_iota_nphi=args.stage2_iota_nphi,
        stage2_iota_ntheta=args.stage2_iota_ntheta,
        stage2_iota_mpol=args.stage2_iota_mpol,
        stage2_iota_ntor=args.stage2_iota_ntor,
    )
    stage2_iota_objective_mode = getattr(
        args,
        "stage2_iota_objective_mode",
        "report",
    )
    if stage2_iota_objective_mode == "report":
        return
    if args.stage2_iota_target is None:
        raise ValueError(
            "--stage2-iota-objective-mode=soft requires --stage2-iota-target."
        )
    if args.constraint_method == "alm":
        raise ValueError(
            "--stage2-iota-objective-mode=soft is only supported by the "
            "penalty/L-BFGS and basin-hopping Stage 2 paths; the ALM hard-iota "
            "constraint remains disabled."
        )


def parse_stage2_edge_iota_radial_band(value: str) -> tuple[float, float]:
    parts = str(value).split(",")
    if len(parts) != 2:
        raise ValueError("--stage2-edge-iota-radial-band must be 'lower,upper'.")
    return (float(parts[0]), float(parts[1]))


def parse_stage2_edge_iota_helicity(value: str) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"+1", "1", "co", "co-helicity", "co_helicity"}:
        return 1
    if normalized in {"-1", "counter", "counter-helicity", "counter_helicity"}:
        return -1
    raise ValueError("--stage2-edge-iota-helicity must be +1/co or -1/counter.")


def build_stage2_edge_iota_config(args) -> EdgeIotaConfig:
    helicity_value = getattr(args, "stage2_edge_iota_helicity", None)
    helicity_sign = (
        1
        if helicity_value in {None, ""}
        else parse_stage2_edge_iota_helicity(helicity_value)
    )
    return EdgeIotaConfig(
        eqdsk_path=getattr(args, "stage2_edge_iota_eqdsk", None),
        lcfs_path=getattr(args, "stage2_edge_iota_lcfs", None),
        edge_band=parse_stage2_edge_iota_radial_band(
            getattr(args, "stage2_edge_iota_radial_band", "0.75,1.0")
        ),
        sample_count=getattr(args, "stage2_edge_iota_sample_count", 6),
        helicity_sign=helicity_sign,
        trace_turns=getattr(
            args,
            "stage2_edge_iota_trace_turns",
            DEFAULT_EDGE_TRACE_TURNS,
        ),
        steps_per_turn=getattr(args, "stage2_edge_iota_steps_per_turn", 240),
        q_validation_rel_tol=getattr(
            args,
            "stage2_edge_iota_q_validation_rel_tol",
            0.002,
        ),
        edge_delta_abs_iota_target_min=getattr(
            args,
            "stage2_edge_iota_target_min",
            0.10,
        ),
        edge_survival_fraction_min=getattr(
            args,
            "stage2_edge_iota_survival_fraction_min",
            1.0,
        ),
        edge_width_max=getattr(args, "stage2_edge_iota_width_max", None),
        coil_partition={
            "banana_source": "stage2_artifact_coil_groups",
            "tf_source": "eqdsk",
        },
    )


def validate_stage2_edge_iota_cli_args(args) -> None:
    mode = getattr(args, "stage2_edge_iota_mode", EDGE_IOTA_MODE_OFF)
    if mode == EDGE_IOTA_MODE_OFF:
        return
    helicity_value = getattr(args, "stage2_edge_iota_helicity", None)
    if helicity_value in {None, ""}:
        raise ValueError(
            f"--stage2-edge-iota-mode={mode} requires --stage2-edge-iota-helicity."
        )
    config = build_stage2_edge_iota_config(args)
    validate_edge_iota_config(config, mode)
    if mode == EDGE_IOTA_MODE_SOFT:
        raise ValueError(
            "--stage2-edge-iota-mode=soft is not implemented yet; use report "
            "mode until the non-gradient routing story is explicit."
        )


def validate_s_hel_objective_cli_args(args) -> None:
    if not getattr(args, "enable_s_hel_objective", False):
        return
    s_hel_objective_weight = float(args.s_hel_objective_weight)
    if not np.isfinite(s_hel_objective_weight) or s_hel_objective_weight <= 0.0:
        raise ValueError("--s-hel-objective-weight must be finite and positive.")


def resolve_stage2_iota_constraint_weight(constraint_weight: float) -> float | None:
    return canonical_stage2_iota_constraint_weight(constraint_weight)


def stage2_iota_runtime_is_active(stage2_iota_runtime) -> bool:
    return str(getattr(stage2_iota_runtime, "mode", "")) in {
        "soft",
        "alm",
        "alm-floor",
    }


def build_stage2_iota_runtime_if_requested(
    *,
    args,
    equilibrium_file,
    bs,
    tf_coils,
    major_radius,
    toroidal_flux,
    proxy_plasma_current_A,
    boozer_current_convention,
):
    stage2_iota_objective_mode = getattr(
        args,
        "stage2_iota_objective_mode",
        "report",
    )
    if stage2_iota_objective_mode == "report":
        return None
    return build_stage2_iota_runtime(
        equilibrium_file=equilibrium_file,
        bs=bs,
        tf_coils=tf_coils,
        major_radius=major_radius,
        toroidal_flux=toroidal_flux,
        nphi=args.stage2_iota_nphi,
        ntheta=args.stage2_iota_ntheta,
        mpol=args.stage2_iota_mpol,
        ntor=args.stage2_iota_ntor,
        vol_target=args.stage2_iota_vol_target,
        iota_target=float(args.stage2_iota_target),
        iota_tolerance=args.stage2_iota_tolerance,
        constraint_weight=resolve_stage2_iota_constraint_weight(
            args.stage2_iota_constraint_weight,
        ),
        num_tf_coils=args.stage2_iota_num_tf_coils,
        mode=stage2_iota_objective_mode,
        weight=1.0,
        boozer_I=physical_current_to_boozer_I(
            proxy_plasma_current_A,
            convention=boozer_current_convention,
        ),
    )


def resolve_s_hel_objective_weight(args) -> float:
    if not bool(getattr(args, "enable_s_hel_objective", False)):
        return 0.0
    return float(getattr(args, "s_hel_objective_weight"))


def build_s_hel_objective(args, field, surface):
    weight = resolve_s_hel_objective_weight(args)
    if weight <= 0.0:
        return None, weight
    return HelicalFieldContentObjective(field, surface), weight


def build_lbfgsb_bounds(optimizable):
    return list(
        zip(
            np.asarray(optimizable.lower_bounds, dtype=float),
            np.asarray(optimizable.upper_bounds, dtype=float),
        )
    )


def _stage2_alm_env_float(env_name: str, suffix: str) -> float:
    return float(os.environ.get(env_name, str(stage2_alm_default(suffix))))


def _stage2_alm_env_int(env_name: str, suffix: str) -> int:
    return int(os.environ.get(env_name, str(stage2_alm_default(suffix))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage 2 banana coil optimization against a fixed plasma surface.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=os.environ.get(
            "PLASMA_SURF_FILENAME", "wout_nfp22ginsburg_000_014417_iota15.nc"
        ),
        help="VMEC wout filename under the equilibria directory.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=os.environ.get("EQUILIBRIA_DIR", DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains the equilibrium wout files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=os.environ.get("EQUILIBRIUM_PATH"),
        help="Explicit path to the equilibrium file. Overrides --equilibria-dir.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("STAGE2_OUTPUT_ROOT", SCRIPT_DIR),
        help="Directory where outputs-[plasma] will be written.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=os.environ.get("STAGE2_BS_PATH"),
        help="Optional path to a saved Stage 2 biot_savart_opt.json seed to restart from.",
    )
    parser.add_argument(
        "--stage2-seed-surf-path",
        default=os.environ.get("STAGE2_SEED_SURF_PATH"),
        help=(
            "Optional saved Boozer-surface artifact used to warm-start the "
            "Stage 2 iota runtime."
        ),
    )
    parser.add_argument(
        "--stage2-plasma-surface-path",
        default=os.environ.get("STAGE2_PLASMA_SURFACE_PATH"),
        help=(
            "Optional saved Surface artifact used DIRECTLY as the Stage-2 plasma "
            "target (the SquaredFlux field-fit surface), overriding the surface "
            "derived from --equilibrium-path. For coherent warm-starts: point at a "
            "converged seed's own boozer surface so loaded coils start at their "
            "true field error instead of being re-fit to a re-derived surface. "
            "ONLY the field-fit target changes; the LCFS shell checks, the iota-"
            "runtime diagnostic, and any cold-path proxy geometry stay derived from "
            "--equilibrium-path. Default: derive from --equilibrium-path."
        ),
    )
    parser.add_argument(
        "--stage2-seed-current-traversal",
        action="store_true",
        default=os.environ.get("STAGE2_SEED_CURRENT_TRAVERSAL", "").lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Allow a loaded Stage 2 seed to keep its geometry while retargeting "
            "proxy/VF traversal currents for current-homotopy rungs."
        ),
    )
    parser.add_argument(
        "--seed-order-upgrade",
        type=int,
        default=(
            int(os.environ["SEED_ORDER_UPGRADE"])
            if "SEED_ORDER_UPGRADE" in os.environ
            else None
        ),
        help=(
            "Optional Fourier order upgrade applied to a loaded Stage 2 seed "
            "before rebuilding the banana symmetry family."
        ),
    )
    parser.add_argument(
        "--constraint-profile-label",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--constraint-override-reason",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-lcfs-max-major-radius-m",
        type=float,
        default=TARGET_LCFS_MAX_MAJOR_RADIUS_M,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-lcfs-max-minor-radius-m",
        type=float,
        default=TARGET_LCFS_MAX_MINOR_RADIUS_M,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stage2-plasma-scaling-mode",
        choices=("lcfs", "working"),
        default=os.environ.get("STAGE2_PLASMA_SCALING_MODE", "lcfs"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--nphi", type=int, default=int(os.environ.get("NPHI", "255")))
    parser.add_argument(
        "--ntheta", type=int, default=int(os.environ.get("NTHETA", "64"))
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build and save the initialized configuration without running the optimizer.",
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=float(
            os.environ.get("BANANA_SURF_RADIUS", str(BANANA_WINDING_MINOR_RADIUS_M))
        ),
        help=(
            "Coil surface minor radius. Defaults to the hardware contract "
            "banana winding minor radius."
        ),
    )
    parser.add_argument(
        "--winding-surface-free-mpol",
        type=int,
        default=int(os.environ.get("WINDING_SURFACE_FREE_MPOL", "0")),
        help=(
            "Fresh Stage 2 CWS only: unfix coil-winding-surface R/Z Fourier "
            "shape modes up to this poloidal index. Default 0 keeps the "
            "historical fixed circular winding torus."
        ),
    )
    parser.add_argument(
        "--winding-surface-free-ntor",
        type=int,
        default=int(os.environ.get("WINDING_SURFACE_FREE_NTOR", "0")),
        help=(
            "Fresh Stage 2 CWS only: unfix coil-winding-surface R/Z Fourier "
            "shape modes up to +/- this toroidal index. Default 0 keeps the "
            "historical fixed circular winding torus."
        ),
    )
    parser.add_argument(
        "--winding-surface-free-r0",
        action="store_true",
        default=os.environ.get("WINDING_SURFACE_FREE_R0", "").lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Unfix the coil-winding-surface major radius rc(0,0) as a bounded "
            "translation DOF (vessel-clearance corridor). Valid on a fresh CWS "
            "or a loaded seed (re-centers the loaded winding surface; T1.5). "
            "Default OFF keeps the historical fixed winding R0."
        ),
    )
    parser.add_argument(
        "--winding-surface-free-minor",
        action="store_true",
        default=os.environ.get("WINDING_SURFACE_FREE_MINOR", "").lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Unfix the coil-winding-surface minor radius rc(1,0)/zs(1,0) as "
            "bounded DOFs (last-resort lever floored at the on-spec a). Valid on "
            "a fresh CWS or a loaded seed (re-sizes the loaded winding surface; "
            "T1.5). Default OFF keeps the historical fixed minor radius."
        ),
    )
    parser.add_argument(
        "--winding-dof-scale",
        action="store_true",
        default=os.environ.get("WINDING_DOF_SCALE", "").lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Apply a per-DOF variable transform (u = x/scale) around the L-BFGS-B "
            "penalty solve so the small-corridor winding SIZE DOFs "
            "(rc(0,0)=R0, rc(1,0)/zs(1,0)=minor) are scaled by their corridor "
            "widths and not swamped by the curve harmonics. Default OFF == "
            "byte-identical (scale == 1 everywhere); no effect under "
            "--constraint-method alm or --basin-hops (penalty path only)."
        ),
    )
    parser.add_argument(
        "--tf-current-A",
        type=float,
        default=float(os.environ.get("TF_CURRENT_A", str(TF_CURRENT_CW_DEFAULT_A))),
        help=(
            "Per-TF-coil current in physical SI amperes. Negative current is the "
            "HBT clockwise toroidal-field convention (default -8e4 = -80 kA)."
        ),
    )
    parser.add_argument(
        "--banana-init-current-A",
        type=float,
        default=float(os.environ.get("BANANA_INIT_CURRENT_A", "-1e4")),
        help=(
            "Fresh-initialization banana-coil current in SI amperes. Negative "
            "current matches the CW TF convention for positive rotational transform."
        ),
    )
    parser.add_argument(
        "--banana-current-max-A",
        type=float,
        default=float(os.environ.get("BANANA_CURRENT_MAX_A", "16000")),
        help="Hard upper bound on the realized banana-coil current in SI amperes.",
    )
    parser.add_argument(
        "--finite-current-mode",
        choices=tuple(FINITE_CURRENT_PROFILES),
        default=os.environ.get("FINITE_CURRENT_MODE"),
        help=(
            "Finite-current construction mode. Default preserves the Wataru "
            "proxy-field model; jhalpern30_proxy_field enables historical "
            "replay construction."
        ),
    )
    parser.add_argument(
        "--flip-banana",
        action="store_true",
        default=os.environ.get("FLIP_BANANA", "").lower() in {"1", "true", "yes", "on"},
        help=(
            "Use the historical jhalpern30 flipped banana-current sign and "
            "record the matching iota-target sign metadata."
        ),
    )
    parser.add_argument(
        "--proxy-plasma-current-A",
        type=float,
        default=(
            float(os.environ["PROXY_PLASMA_CURRENT_A"])
            if "PROXY_PLASMA_CURRENT_A" in os.environ
            else None
        ),
        help=(
            "Physical SI amperes for the proxy plasma-current coil. Wataru mode "
            "requires a nonnegative magnitude; jhalpern30 mode treats the value "
            "as a signed physical scalar."
        ),
    )
    parser.add_argument(
        "--vf-current-A",
        type=float,
        default=(
            float(os.environ["VF_CURRENT_A"]) if "VF_CURRENT_A" in os.environ else None
        ),
        help=(
            "Physical SI amperes for the VF scalar. Wataru mode applies it to "
            "fixed independent sign-preserving template currents. Fresh "
            "jhalpern30 mode derives it from --proxy-plasma-current-A / 6.5; "
            "seeded jhalpern30 current traversal may retarget the shared mutable "
            "VF current scalar."
        ),
    )
    parser.add_argument(
        "--vf-current-max-A",
        type=float,
        default=float(
            os.environ.get("VF_CURRENT_MAX_A", str(BANANA_CURRENT_HARD_LIMIT_A))
        ),
        help=(
            "Hard L-BFGS-B bound on the realized shared VF current in SI amperes "
            "when the selected finite-current profile makes VF current optimizable."
        ),
    )
    parser.add_argument(
        "--vf-template-path",
        default=os.environ.get("VF_TEMPLATE_PATH"),
        help="Optional BiotSavart JSON template that defines the VF coil geometry/signs.",
    )
    parser.add_argument(
        "--major-radius",
        type=float,
        default=float(
            os.environ.get("MAJOR_RADIUS", str(VACUUM_VESSEL_MAJOR_RADIUS_M))
        ),
        help=(
            "Vacuum-vessel major radius (fixed contract, "
            f"= {VACUUM_VESSEL_MAJOR_RADIUS_M:.3f} m)."
        ),
    )
    parser.add_argument(
        "--accept-offspec-major-radius",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accept-offspec-banana-current-sign",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accept-offspec-banana-current-max",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accept-offspec-tf-current-sign",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accept-offspec-tf-current-magnitude",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accept-offspec-coil-length",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accept-offspec-curvature",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accept-offspec-winding-radius",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=float(os.environ.get("TOROIDAL_FLUX", "0.24")),
        help="Flux-surface label s in [0, 1] used when loading the VMEC surface.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=int(os.environ.get("COIL_ORDER", "2")),
        help="Fourier order for the banana coil.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=int(os.environ.get("MAXITER", "300")),
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--maxcor",
        type=int,
        default=int(os.environ.get("MAXCOR", str(DEFAULT_LBFGSB_MAXCOR))),
        help=(
            "L-BFGS-B memory (number of correction pairs, "
            f"default {DEFAULT_LBFGSB_MAXCOR})."
        ),
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=float(os.environ.get("FTOL", "1e-15")),
        help="L-BFGS-B function change tolerance. Default 1e-15 (factr~4.5) effectively lets maxiter control termination.",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=float(os.environ.get("GTOL", "1e-15")),
        help="L-BFGS-B projected gradient tolerance. Default 1e-15 effectively lets maxiter control termination.",
    )
    parser.add_argument(
        "--constraint-method",
        choices=["penalty", "alm"],
        default=os.environ.get("CONSTRAINT_METHOD", "penalty"),
        help="Use the weighted-penalty objective or the augmented Lagrangian outer loop.",
    )
    parser.add_argument(
        "--alm-max-outer-iters",
        type=int,
        default=_stage2_alm_env_int("ALM_MAX_OUTER_ITERS", "max_outer_iters"),
        help=(
            "Maximum number of ALM outer iterations "
            f"(default {stage2_alm_default('max_outer_iters')})."
        ),
    )
    parser.add_argument(
        "--alm-penalty-init",
        type=float,
        default=_stage2_alm_env_float("ALM_PENALTY_INIT", "penalty_init"),
        help=(
            "Initial ALM penalty parameter "
            f"(default {stage2_alm_default('penalty_init')})."
        ),
    )
    parser.add_argument(
        "--alm-penalty-scale",
        type=float,
        default=_stage2_alm_env_float("ALM_PENALTY_SCALE", "penalty_scale"),
        help=(
            "Multiplicative ALM penalty growth factor "
            f"(default {stage2_alm_default('penalty_scale')})."
        ),
    )
    parser.add_argument(
        "--alm-penalty-max",
        type=float,
        default=_stage2_alm_env_float("ALM_PENALTY_MAX", "penalty_max"),
        help=(
            "Maximum ALM penalty parameter before capped termination "
            f"(default {stage2_alm_default('penalty_max')})."
        ),
    )
    parser.add_argument(
        "--alm-feas-tol",
        type=float,
        default=_stage2_alm_env_float("ALM_FEAS_TOL", "feas_tol"),
        help=(
            "Dimensionless normalized ALM max-violation stopping tolerance "
            f"(default {stage2_alm_default('feas_tol')})."
        ),
    )
    parser.add_argument(
        "--alm-stationarity-tol",
        type=float,
        default=_stage2_alm_env_float("ALM_STATIONARITY_TOL", "stationarity_tol"),
        help=(
            "ALM augmented-gradient stopping tolerance "
            f"(default {stage2_alm_default('stationarity_tol')})."
        ),
    )
    parser.add_argument(
        "--alm-trust-radius-init",
        type=float,
        default=_stage2_alm_env_float("ALM_TRUST_RADIUS_INIT", "trust_radius_init"),
        help="Initial relative trust radius for bounded ALM inner solves (0 disables bounds).",
    )
    parser.add_argument(
        "--alm-trust-radius-min",
        type=float,
        default=_stage2_alm_env_float("ALM_TRUST_RADIUS_MIN", "trust_radius_min"),
        help="Minimum relative trust radius for bounded ALM inner solves.",
    )
    parser.add_argument(
        "--alm-trust-radius-shrink",
        type=float,
        default=_stage2_alm_env_float("ALM_TRUST_RADIUS_SHRINK", "trust_radius_shrink"),
        help="Multiplicative shrink factor for the ALM inner trust radius.",
    )
    parser.add_argument(
        "--alm-trust-radius-grow",
        type=float,
        default=_stage2_alm_env_float("ALM_TRUST_RADIUS_GROW", "trust_radius_grow"),
        help="Multiplicative growth factor for the ALM inner trust radius after good steps.",
    )
    parser.add_argument(
        "--alm-max-inner-attempts",
        type=int,
        default=_stage2_alm_env_int("ALM_MAX_INNER_ATTEMPTS", "max_inner_attempts"),
        help="Maximum number of trust-radius retries per ALM outer iteration.",
    )
    parser.add_argument(
        "--alm-max-subproblem-continuations",
        type=int,
        default=_stage2_alm_env_int(
            "ALM_MAX_SUBPROBLEM_CONTINUATIONS", "max_subproblem_continuations"
        ),
        help="Maximum accepted-feasible continuation solves before forcing an ALM return.",
    )
    parser.add_argument(
        "--alm-distance-smoothing",
        type=float,
        default=_stage2_alm_env_float("ALM_DISTANCE_SMOOTHING", "distance_smoothing"),
        help="Distance soft-min temperature for Stage 2 ALM spacing constraints.",
    )
    parser.add_argument(
        "--alm-curvature-smoothing",
        type=float,
        # Stage 2 uses 0.25 (broader softmax window over the banana coil's
        # kappa array) vs single-stage's 0.05. The wider window is appropriate
        # here because Stage 2 operates on a single banana coil with fewer
        # quadrature points and less sensitivity to curvature perturbations.
        default=_stage2_alm_env_float("ALM_CURVATURE_SMOOTHING", "curvature_smoothing"),
        help="Curvature soft-max temperature for Stage 2 ALM curvature constraints.",
    )
    parser.add_argument(
        "--alm-taylor-test",
        action="store_true",
        help="Run a directional Taylor test on the initialized Stage 2 ALM subproblem before optimization.",
    )
    parser.add_argument(
        "--alm-taylor-test-seed",
        type=int,
        default=int(os.environ.get("ALM_TAYLOR_TEST_SEED", "1")),
        help="Random seed used to build the Stage 2 ALM Taylor-test direction.",
    )
    parser.add_argument(
        "--alm-fix-signal-mismatch-guard",
        action="store_true",
        default=os.environ.get("ALM_FIX_SIGNAL_MISMATCH_GUARD", "0")
        not in (
            "",
            "0",
            "false",
            "False",
        ),
        help=(
            "Opt-in rollout flag: when hard constraints are feasible but the "
            "surrogate signal remains active, keep the bounded Stage 2 ALM "
            "inner-continuation path alive instead of taking the legacy "
            "penalty-bump arm. The dual-update stationarity gate remains "
            "unchanged. Mirrors the single-stage CLI flag of the same name."
        ),
    )
    parser.add_argument(
        "--stage2-iota-target",
        type=float,
        default=(
            None
            if os.environ.get("STAGE2_IOTA_TARGET") is None
            else float(os.environ["STAGE2_IOTA_TARGET"])
        ),
        help="Target iota used by the optional Stage 2 reporting-only probe.",
    )
    parser.add_argument(
        "--stage2-iota-objective-mode",
        choices=("report", "soft"),
        default=os.environ.get("STAGE2_IOTA_OBJECTIVE_MODE", "report"),
        help=(
            "How --stage2-iota-target is used. 'report' keeps the current "
            "post-run bootability probe only. 'soft' enables the existing "
            "Boozer/iota soft-penalty objective in penalty and basin-hopping "
            "Stage 2 runs."
        ),
    )
    parser.add_argument(
        "--stage2-iota-tolerance",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_IOTA_TOLERANCE",
                str(DEFAULT_STAGE2_IOTA_TOLERANCE),
            )
        ),
        help="Absolute |iota_solved - iota_target| tolerance for the Stage 2 iota path.",
    )
    parser.add_argument(
        "--stage2-iota-vol-target",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_IOTA_VOL_TARGET",
                str(DEFAULT_STAGE2_IOTA_VOL_TARGET),
            )
        ),
        help="Outer-surface target volume used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-constraint-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_IOTA_CONSTRAINT_WEIGHT",
                str(DEFAULT_STAGE2_IOTA_CONSTRAINT_WEIGHT),
            )
        ),
        help=(
            "Boozer constraint weight used by the Stage 2 Boozer/iota solve. "
            "Use a non-positive value to select the exact Boozer Newton solver."
        ),
    )
    parser.add_argument(
        "--stage2-iota-num-tf-coils",
        type=int,
        default=int(
            os.environ.get(
                "STAGE2_IOTA_NUM_TF_COILS",
                str(DEFAULT_STAGE2_IOTA_NUM_TF_COILS),
            )
        ),
        help="Expected TF-coil count used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-nphi",
        type=int,
        default=int(os.environ.get("STAGE2_IOTA_NPHI", str(DEFAULT_STAGE2_IOTA_NPHI))),
        help="Surface quadrature nphi used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-ntheta",
        type=int,
        default=int(
            os.environ.get("STAGE2_IOTA_NTHETA", str(DEFAULT_STAGE2_IOTA_NTHETA))
        ),
        help="Surface quadrature ntheta used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-mpol",
        type=int,
        default=int(os.environ.get("STAGE2_IOTA_MPOL", str(DEFAULT_STAGE2_IOTA_MPOL))),
        help="Boozer-surface mpol used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-ntor",
        type=int,
        default=int(os.environ.get("STAGE2_IOTA_NTOR", str(DEFAULT_STAGE2_IOTA_NTOR))),
        help="Boozer-surface ntor used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-edge-iota-mode",
        choices=(EDGE_IOTA_MODE_OFF, EDGE_IOTA_MODE_REPORT, EDGE_IOTA_MODE_SOFT),
        default=os.environ.get("STAGE2_EDGE_IOTA_MODE", EDGE_IOTA_MODE_OFF),
        help=(
            "Post-run fixed-boundary edge-delivered-iota oracle. 'off' is "
            "behavior-neutral. 'report' writes EDGE_* summary fields and a "
            "profile JSON. 'soft' is reserved for future non-gradient routing."
        ),
    )
    parser.add_argument(
        "--stage2-edge-iota-eqdsk",
        default=os.environ.get("STAGE2_EDGE_IOTA_EQDSK"),
        help="Explicit G-EQDSK path for the edge-delivered-iota report oracle.",
    )
    parser.add_argument(
        "--stage2-edge-iota-lcfs",
        default=os.environ.get("STAGE2_EDGE_IOTA_LCFS"),
        help="Explicit LCFS JSON path for edge-band normalization.",
    )
    parser.add_argument(
        "--stage2-edge-iota-radial-band",
        default=os.environ.get("STAGE2_EDGE_IOTA_RADIAL_BAND", "0.75,1.0"),
        help="Comma-separated r/a edge band sampled by the edge-iota report.",
    )
    parser.add_argument(
        "--stage2-edge-iota-sample-count",
        type=int,
        default=int(os.environ.get("STAGE2_EDGE_IOTA_SAMPLE_COUNT", "6")),
        help="Number of radial samples in --stage2-edge-iota-radial-band.",
    )
    parser.add_argument(
        "--stage2-edge-iota-target-min",
        type=float,
        default=float(os.environ.get("STAGE2_EDGE_IOTA_TARGET_MIN", "0.10")),
        help="Promotion-facing minimum p10 edge |iota| magnitude lift.",
    )
    parser.add_argument(
        "--stage2-edge-iota-helicity",
        default=os.environ.get("STAGE2_EDGE_IOTA_HELICITY"),
        help="Explicit co-helicity sign: +1/co or -1/counter.",
    )
    parser.add_argument(
        "--stage2-edge-iota-trace-turns",
        type=int,
        default=int(
            os.environ.get(
                "STAGE2_EDGE_IOTA_TRACE_TURNS",
                str(DEFAULT_EDGE_TRACE_TURNS),
            )
        ),
        help="Toroidal turns for tokamak and hybrid edge-iota traces.",
    )
    parser.add_argument(
        "--stage2-edge-iota-steps-per-turn",
        type=int,
        default=int(os.environ.get("STAGE2_EDGE_IOTA_STEPS_PER_TURN", "240")),
        help="Field-line integration samples per toroidal turn.",
    )
    parser.add_argument(
        "--stage2-edge-iota-q-validation-rel-tol",
        type=float,
        default=float(os.environ.get("STAGE2_EDGE_IOTA_Q_VALIDATION_REL_TOL", "0.002")),
        help="Relative tolerance for tokamak-only trace validation against 1/q.",
    )
    parser.add_argument(
        "--stage2-edge-iota-survival-fraction-min",
        type=float,
        default=float(os.environ.get("STAGE2_EDGE_IOTA_SURVIVAL_FRACTION_MIN", "1.0")),
        help="Required surviving fraction across the configured edge band.",
    )
    parser.add_argument(
        "--stage2-edge-iota-width-max",
        type=float,
        default=(
            None
            if os.environ.get("STAGE2_EDGE_IOTA_WIDTH_MAX") is None
            else float(os.environ["STAGE2_EDGE_IOTA_WIDTH_MAX"])
        ),
        help="Optional maximum edge width/chaos indicator for a passing report.",
    )
    parser.add_argument(
        "--enable-topology-bridge-diagnostics",
        action="store_true",
        default=bool(int(os.environ.get("ENABLE_TOPOLOGY_BRIDGE_DIAGNOSTICS", "0"))),
        help=(
            "Phase 3b gate: persist HELICAL_FIELD_CONTENT, "
            "PRE_BOOZER_TOPOLOGY_SCORE, FIELDLINE_IOTA_PROXY, and the "
            "FIELDLINE_IOTA_PROXY_{VALID,N_TRANSITS,REASON} diagnostics "
            "post-solve. OFF by default - adds ~5-30s per Stage 2 run for "
            "field-line tracing + helical content FFT. Enable for donor "
            "ranking, convergence studies, or topology audits "
            "(--enable-topology-bridge-diagnostics or set "
            "ENABLE_TOPOLOGY_BRIDGE_DIAGNOSTICS=1). When the gate is off "
            "the topology diagnostic artifact keys remain None so downstream consumers can "
            "distinguish 'diagnostic skipped' from 'diagnostic ran and "
            "failed'. S_HEL_OBJECTIVE_WEIGHT is still persisted when the "
            "live --enable-s-hel-objective schedule is active."
        ),
    )
    parser.add_argument(
        "--no-enable-topology-bridge-diagnostics",
        dest="enable_topology_bridge_diagnostics",
        action="store_false",
        help=(
            "Force-disable Phase 3 diagnostic persistence even when "
            "ENABLE_TOPOLOGY_BRIDGE_DIAGNOSTICS=1 is set in the "
            "environment; HELICAL_FIELD_CONTENT, PRE_BOOZER_TOPOLOGY_SCORE, "
            "FIELDLINE_IOTA_PROXY, FIELDLINE_IOTA_PROXY_VALID, "
            "FIELDLINE_IOTA_PROXY_N_TRANSITS, and FIELDLINE_IOTA_PROXY_REASON "
            "remain None in the persisted artifact. S_HEL_OBJECTIVE_WEIGHT "
            "still records the live schedule when --enable-s-hel-objective is set."
        ),
    )
    parser.add_argument(
        "--enable-s-hel-objective",
        action="store_true",
        default=bool(int(os.environ.get("ENABLE_S_HEL_OBJECTIVE", "0"))),
        help=(
            "Enable the Phase 3b helical-content objective in the "
            "untrusted-but-evaluable Stage 2 objective lane. This is the explicit "
            "post-gradient-validation schedule gate; use "
            "--s-hel-objective-weight to set the current ramp value."
        ),
    )
    parser.add_argument(
        "--s-hel-objective-weight",
        type=float,
        default=float(os.environ.get("S_HEL_OBJECTIVE_WEIGHT", "1e-3")),
        help=(
            "Phase 3b helical-content objective weight used only when "
            "--enable-s-hel-objective is set. The plan's first schedule rung "
            "is 1e-3."
        ),
    )
    parser.add_argument(
        "--topology-bridge-nfieldlines",
        type=int,
        default=int(
            os.environ.get(
                "TOPOLOGY_BRIDGE_NFIELDLINES",
                str(TOPOLOGY_BRIDGE_DEFAULT_NFIELDLINES),
            )
        ),
        help=(
            "Phase 3a diagnostic: number of midplane seed radii for the "
            "post-solve field-line iota proxy in banana_opt.topology_bridge."
        ),
    )
    parser.add_argument(
        "--topology-bridge-tmax",
        type=float,
        default=float(
            os.environ.get(
                "TOPOLOGY_BRIDGE_TMAX",
                str(TOPOLOGY_BRIDGE_DEFAULT_TMAX),
            )
        ),
        help=(
            "Phase 3a diagnostic: hard tmax cap (plan line 271) for the "
            "post-solve field-line tracer."
        ),
    )
    parser.add_argument(
        "--topology-bridge-tol",
        type=float,
        default=float(
            os.environ.get(
                "TOPOLOGY_BRIDGE_TOL",
                str(TOPOLOGY_BRIDGE_DEFAULT_TOL),
            )
        ),
        help=(
            "Phase 3a diagnostic: ODE tolerance (plan line 273 default 1e-8) "
            "for the post-solve field-line tracer."
        ),
    )
    parser.add_argument(
        "--topology-bridge-n-transits-target",
        type=int,
        default=int(os.environ.get("TOPOLOGY_BRIDGE_N_TRANSITS_TARGET", "100")),
        help=(
            "Phase 3a diagnostic: minimum toroidal transits a line must "
            "complete to contribute to the iota proxy mean (plan acceptance "
            "rung default 100)."
        ),
    )
    parser.add_argument(
        "--topology-bridge-escape-radius",
        type=float,
        default=None,
        help=(
            "Phase 3a diagnostic: cylindrical R escape bound (plan line 272). "
            "When omitted, defaults to 1.25 * max(R) of the Boozer surface."
        ),
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=float(os.environ.get("LENGTH_WEIGHT", "0.0005")),
        help="Curve-length penalty weight (soft target at --length-target).",
    )
    parser.add_argument(
        "--length-min-weight",
        type=float,
        default=float(os.environ.get("LENGTH_MIN_WEIGHT", "1.0")),
        help="Weight on the one-sided below-floor curve-length penalty, separate "
        "from --length-weight. The floor is 0.5 * --length-target; the penalty is "
        "0.5 * w * min(L - floor, 0)^2 and is exactly zero for L >= floor (no "
        "effect on runs that stay above it). The soft --length-weight (default "
        "0.0005) cannot hold this floor on its own, so a degenerate small-coil "
        "basin is reachable; default 1.0 keeps the optimizer above the floor. "
        "Legacy-replay note: pre-split runs combined this term with --length-weight "
        "(both at LENGTH_WEIGHT~=5e-4); to reproduce that exact below-floor "
        "strength set LENGTH_MIN_WEIGHT=0.0005.",
    )
    parser.add_argument(
        "--stage2-poloidal-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_POLOIDAL_WEIGHT",
                str(STAGE2_POLOIDAL_WEIGHT_DEFAULT),
            )
        ),
        help="Stage 2 weighted poloidal-extent hinge weight (penalty path).",
    )
    parser.add_argument(
        "--stage2-width-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_WIDTH_WEIGHT",
                str(STAGE2_WIDTH_WEIGHT_DEFAULT),
            )
        ),
        help="Stage 2 weighted width hinge weight (penalty path).",
    )
    parser.add_argument(
        "--stage2-selfint-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_SELF_INTERSECT_WEIGHT",
                str(STAGE2_SELF_INTERSECT_WEIGHT_DEFAULT),
            )
        ),
        help="Stage 2 curve self-intersect penalty weight.",
    )
    parser.add_argument(
        "--self-envelope-mode",
        choices=("hinge", "groc", "off"),
        default=os.environ.get("STAGE2_SELF_ENVELOPE_MODE", "hinge"),
        help=(
            "Stage 2 self-envelope objective mode: true-arc distance hinge, "
            "global-radius-of-curvature hinge, or diagnostic-only off."
        ),
    )
    parser.add_argument(
        "--self-envelope-weight",
        type=float,
        default=float(os.environ.get("STAGE2_SELF_ENVELOPE_WEIGHT", "1.0")),
        help="Stage 2 self-envelope penalty weight for hinge/groc modes.",
    )
    parser.add_argument(
        "--self-envelope-floor",
        type=float,
        default=(
            None
            if os.environ.get("STAGE2_SELF_ENVELOPE_FLOOR") is None
            else float(os.environ["STAGE2_SELF_ENVELOPE_FLOOR"])
        ),
        help=(
            "Self-envelope activation floor. Defaults by mode: hinge/off use "
            f"{BANANA_SELF_ENVELOPE_MIN_DISTANCE_M:.6f} m distance; groc uses "
            f"{BANANA_SELF_ENVELOPE_GROC_RADIUS_FLOOR_M:.6f} m radius."
        ),
    )
    parser.add_argument(
        "--self-distance-window",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_SELF_DISTANCE_WINDOW",
                str(BANANA_SELF_DISTANCE_WINDOW_M),
            )
        ),
        help="Physical arc-length exclusion window for self-envelope checks.",
    )
    parser.add_argument(
        "--self-envelope-sampling-margin",
        type=float,
        default=float(os.environ.get("STAGE2_SELF_ENVELOPE_SAMPLING_MARGIN", "0.0")),
        help=(
            "Conservative additive margin on the hinge-mode point-pair "
            "self-envelope threshold, used to make point-sampled optimization "
            "imply the segment-segment CAD screen at a chosen sampling density."
        ),
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=float(os.environ.get("LENGTH_TARGET", str(COIL_LENGTH_TARGET_M))),
        help=(
            "Curve-length target in meters. Values above the "
            f"{COIL_LENGTH_HARD_LIMIT_M:.1f} m hardware ceiling require "
            "--accept-offspec-coil-length."
        ),
    )
    parser.add_argument(
        "--cc-threshold",
        type=float,
        default=float(os.environ.get("CC_THRESHOLD", str(COIL_COIL_MIN_DIST_M))),
        help="Coil-coil distance threshold in meters.",
    )
    parser.add_argument(
        "--cc-objective-margin",
        type=float,
        default=float(
            os.environ.get("CC_OBJECTIVE_MARGIN", str(BANANA_CC_OBJECTIVE_MARGIN_M))
        ),
        help=(
            "Extra coil-coil objective buffer in meters. The hard gate remains "
            "--cc-threshold; the optimizer is steered toward "
            "--cc-threshold + this margin."
        ),
    )
    parser.add_argument(
        "--cc-weight",
        type=float,
        default=float(os.environ.get("CC_WEIGHT", "100")),
        help="Coil-coil distance penalty weight.",
    )
    parser.add_argument(
        "--curvature-weight",
        type=float,
        default=float(os.environ.get("CURVATURE_WEIGHT", "0.0001")),
        help="Curvature penalty weight.",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=float(os.environ.get("CURVATURE_THRESHOLD", str(MAX_CURVATURE_INV_M))),
        help="Curvature penalty threshold in m^-1 (default 100, matching the hardware ceiling).",
    )
    parser.add_argument(
        "--theta-center",
        type=float,
        default=float(os.environ.get("THETA_CENTER", "0.5")),
        help="Initial banana-coil poloidal center in normalized angle coordinates.",
    )
    parser.add_argument(
        "--phi-center",
        type=float,
        default=float(os.environ.get("PHI_CENTER", "0.06")),
        help="Initial banana-coil toroidal center in normalized angle coordinates.",
    )
    parser.add_argument(
        "--theta-width",
        type=float,
        default=float(os.environ.get("THETA_WIDTH", "0.1")),
        help="Initial banana-coil poloidal width in normalized angle coordinates.",
    )
    parser.add_argument(
        "--phi-width",
        type=float,
        default=float(os.environ.get("PHI_WIDTH", "0.03")),
        help="Initial banana-coil toroidal width in normalized angle coordinates.",
    )
    parser.add_argument(
        "--curvature-p-norm",
        type=int,
        default=int(os.environ.get("CURVATURE_P_NORM", "4")),
        help="Lp norm exponent for curvature penalty (default 4).",
    )
    parser.add_argument(
        "--fold-weight",
        type=float,
        default=float(os.environ.get("STAGE2_FOLD_WEIGHT", "1.0")),
        help=(
            "Weight on the fold-curvature hinge. The legacy fold frame measures "
            "surface geodesic curvature; --stage2-couple-pack-rotation-to-fold "
            "measures material-frame binormal curvature. The hinge activates at "
            "limit * (1 - margin-fraction); FOLD_OK stays at the hard limit."
        ),
    )
    parser.add_argument(
        "--fold-geodesic-curvature-limit",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_FOLD_GEODESIC_CURVATURE_LIMIT",
                str(BANANA_FOLD_GEODESIC_CURVATURE_LIMIT_INV_M),
            )
        ),
        help="Hard fold limit on abs(surface geodesic curvature), in m^-1.",
    )
    parser.add_argument(
        "--fold-material-binormal-curvature-limit",
        type=float,
        default=(
            None
            if os.environ.get("STAGE2_FOLD_MATERIAL_BINORMAL_CURVATURE_LIMIT") is None
            else float(os.environ["STAGE2_FOLD_MATERIAL_BINORMAL_CURVATURE_LIMIT"])
        ),
        help=(
            "Hard fold limit on abs(material-frame binormal curvature), in m^-1, "
            "used only with --stage2-couple-pack-rotation-to-fold. Defaults to "
            "the Type-KK fold limit when unset."
        ),
    )
    parser.add_argument(
        "--fold-geodesic-curvature-margin-fraction",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION",
                str(BANANA_FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION),
            )
        ),
        help="Fractional safety margin used by the fold objective threshold.",
    )
    parser.add_argument(
        "--num-quadpoints",
        type=int,
        default=int(os.environ.get("NUM_QUADPOINTS", "128")),
        help="Number of quadrature points for coil discretization (default 128).",
    )
    parser.add_argument(
        "--squared-flux-weight",
        type=float,
        default=float(os.environ.get("SQUARED_FLUX_WEIGHT", "1.0")),
        help="Weight on the SquaredFlux term (default 1.0).",
    )
    parser.add_argument(
        "--basin-hops",
        type=int,
        default=int(os.environ.get("BASIN_HOPS", "0")),
        help="Number of basin-hopping restarts (0 = single L-BFGS-B run, default). "
        "Each hop perturbs the coil DOFs and re-runs L-BFGS-B. "
        "Total runs = basin_hops + 1. Keeps the best result.",
    )
    parser.add_argument(
        "--basin-stepsize",
        type=float,
        default=float(os.environ.get("BASIN_STEPSIZE", "0.01")),
        help="Initial perturbation scale passed to SciPy basin-hopping (default 0.01).",
    )
    parser.add_argument(
        "--basin-temperature",
        type=float,
        default=float(os.environ.get("BASIN_TEMPERATURE", "1.0")),
        help="Metropolis temperature for basin-hopping uphill acceptance (default 1.0).",
    )
    parser.add_argument(
        "--basin-niter-success",
        type=int,
        default=int(os.environ.get("BASIN_NITER_SUCCESS", "0")),
        help="Stop basin-hopping early after this many hops without improvement (0 = disabled, default).",
    )
    parser.add_argument(
        "--basin-seed",
        type=int,
        default=int(os.environ.get("BASIN_SEED", "-1")),
        help="RNG seed for basin-hopping (-1 = random, default). Set for reproducibility.",
    )
    finite_build_group = parser.add_mutually_exclusive_group()
    finite_build_group.add_argument(
        "--finite-build",
        dest="finite_build",
        action="store_true",
        default=True,
        help=(
            "Optimize each banana coil as a multi-filament winding pack (finite "
            "build) instead of a zero-thickness filament. The field (SquaredFlux) "
            "sees the real Type KK 2x7 pack and the pack-rotation profile is "
            "optimized. This is the Stage 2 default representation."
        ),
    )
    finite_build_group.add_argument(
        "--filament-only",
        dest="finite_build",
        action="store_false",
        help=(
            "Diagnostic-only opt-out that runs the legacy zero-thickness banana "
            "centerline model. CAD-bound artifacts must use finite-build."
        ),
    )
    parser.add_argument(
        "--finitebuild-numfilaments-n",
        type=int,
        default=TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N,
        help="Filaments in the normal direction of the banana pack (default 2). "
        "Used only with --finite-build. Type KK regular-grid approximation.",
    )
    parser.add_argument(
        "--finitebuild-numfilaments-b",
        type=int,
        default=TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B,
        help="Filaments in the binormal direction of the banana pack (default 7). "
        "Used only with --finite-build. Type KK regular-grid approximation.",
    )
    parser.add_argument(
        "--finitebuild-gapsize-n",
        type=float,
        default=TYPE_KK_FINITE_BUILD_GAPSIZE_N_M,
        help="Normal-direction filament center spacing, meters (default Type KK "
        "conductor-pack depth for a 2-row spanning approximation). Used only with "
        "--finite-build.",
    )
    parser.add_argument(
        "--finitebuild-gapsize-b",
        type=float,
        default=TYPE_KK_FINITE_BUILD_GAPSIZE_B_M,
        help="Binormal-direction filament center spacing, meters (default Type KK "
        "conductor-pack width divided by 6 for a 7-column spanning approximation). "
        "Used only with --finite-build.",
    )
    parser.add_argument(
        "--finitebuild-rotation-order",
        type=int,
        default=1,
        help="Fourier order of the optimizable pack-rotation profile (default 1). "
        "A negative value fixes the pack orientation (no rotation DOFs). "
        "Used only with --finite-build.",
    )
    parser.add_argument(
        "--finitebuild-frame",
        choices=("centroid", "frenet", "surface_tangent"),
        default="surface_tangent",
        help="Pre-rotation orthonormal frame for the filament pack (default "
        "surface_tangent, a local fork extension that lays the Type KK pack flat "
        "against the winding surface). Used only with --finite-build.",
    )
    parser.add_argument(
        "--finitebuild-pin-current",
        action="store_true",
        help="Fix the banana coil current at --banana-init-current-A instead of "
        "optimizing it. Useful for a current scan, where vacuum field-fit otherwise "
        "leaves the current under-constrained and it drifts toward zero. Used only "
        "with --finite-build.",
    )
    parser.add_argument(
        "--finitebuild-frame-aware-curvature-threshold",
        dest="finitebuild_frame_aware_curvature_threshold",
        action="store_true",
        default=None,
        help="Tighten the in-run curvature threshold (penalty objective, ALM "
        "max_curvature constraint, and the in-run hardware gate) to the "
        "frame-aware winding-pack limit 1/(single-filament bend floor + pack "
        "corner reach) when that is stricter than --curvature-threshold, so "
        "finite-build runs stop converging into the post-hoc "
        "FINITEBUILD_CURVATURE_OK reject region. Never loosens the threshold. "
        "Enabled by default for finite-build runs. Incompatible with "
        "--filament-only when explicitly requested.",
    )
    parser.add_argument(
        "--no-finitebuild-frame-aware-curvature-threshold",
        dest="finitebuild_frame_aware_curvature_threshold",
        action="store_false",
        help="Diagnostic opt-out: keep the centerline curvature threshold even "
        "when --finite-build is active.",
    )
    parser.add_argument(
        "--stage2-couple-pack-rotation-to-fold",
        dest="stage2_couple_pack_rotation_to_fold",
        action="store_true",
        default=os.environ.get("STAGE2_COUPLE_PACK_ROTATION_TO_FOLD", "")
        not in ("", "0", "false", "False"),
        help="T3.2/G2 (default off): build the fold-curvature objective from "
        "the SHARED finite-build pack frame so its live pack-rotation "
        "alpha(theta) drives material-frame binormal buildability, not only "
        "the magnetic field. Off = byte-identical (fresh ZeroRotation fold "
        "frame). Requires --finite-build.",
    )
    parser.add_argument(
        "--stage2-pack-twist-strain-weight",
        dest="stage2_pack_twist_strain_weight",
        type=float,
        default=float(os.environ.get("STAGE2_PACK_TWIST_STRAIN_WEIGHT", "0.0")),
        help="T3.2/G2 (default 0.0): weight of an LPTorsionalStrainPenalty "
        "windability regularizer on the shared pack frame (epsilon_tor = "
        "tau^2 w^2 / 12, Paz-Soldan 2020; width = pack corner span). Weight 0 "
        "never constructs the term. Bounds the twist rate of a freed alpha(theta). "
        "Requires --finite-build.",
    )
    parser.add_argument(
        "--stage2-rotation-aware-curvature-cap",
        dest="stage2_rotation_aware_curvature_cap",
        action="store_true",
        help="T3.2/G3 (default off, NON-PROMOTION-READY): relax the in-run "
        "curvature threshold from the measured edgewise pack cap up "
        "to the realized rotation-aware cap (scalarized to the tightest realized "
        "point) using the live pack frame. The honest FINITEBUILD_CURVATURE_OK "
        "gate (alpha=0 corner frame) is unchanged, so a design that only builds "
        "with a favorable twist stays non-promotable until the D-5 ruling. "
        "Requires --stage2-couple-pack-rotation-to-fold.",
    )
    parser.add_argument(
        "--stage2-vessel-keepout-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_VESSEL_KEEPOUT_WEIGHT",
                str(STAGE2_VESSEL_KEEPOUT_WEIGHT_DEFAULT),
            )
        ),
        help="Weight on the analytic vessel-envelope keep-out term "
        "(CurveVesselEnvelopeKeepout: Type KK swept-envelope corners vs the "
        "fixed vessel torus from hardware_contracts), applied to the "
        "symmetry-expanded banana centerlines. Defaults ON at single-stage "
        "parity (STAGE2_VESSEL_KEEPOUT_WEIGHT_DEFAULT); pass 0 (or export "
        "STAGE2_VESSEL_KEEPOUT_WEIGHT=0) to disable the term entirely for "
        "legacy/byte-identical reproduction. In ALM mode the weighted term "
        "rides the smooth objective rather than adding a constraint row.",
    )
    parser.add_argument(
        "--stage2-available-envelope-reward-weight",
        type=float,
        default=float(os.environ.get("STAGE2_AVAILABLE_ENVELOPE_REWARD_WEIGHT", "0.0")),
        help=(
            "Default-off reward for filling usable positive-clearance vessel "
            "envelope volume with the banana coil channel. This is a smooth "
            "objective term only; direct CAD, finite-build, and confinement "
            "promotion gates remain unchanged."
        ),
    )
    parser.add_argument(
        "--stage2-hardware-sdf-free-space-reward-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT",
                "0.0",
            )
        ),
        help=(
            "Default-off reward for positive CAD/SDF hardware clearance. "
            "Requires --stage2-hardware-keepout-backend=sdf and "
            "--stage2-hardware-keepout-sdf-manifest; this is a smooth "
            "steering term and does not replace the posthoc CAD oracle."
        ),
    )
    parser.add_argument(
        "--stage2-hardware-keepout-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_HARDWARE_KEEPOUT_WEIGHT",
                str(STAGE2_HARDWARE_KEEPOUT_WEIGHT_DEFAULT),
            )
        ),
        help="Weight on the static in-vessel hardware keep-out term "
        "(CurveHardwareKeepout: the swept Type KK U-channel envelope of every "
        "symmetry-expanded banana centerline vs the exported "
        "shells/sensors/solenoid/REMC/limiter/quartz point cloud), wired exactly as "
        "the single-stage path does (same hardware_keepout.json, same "
        "HARDWARE_KEEPOUT_MIN_DISTANCE_M contract threshold). Defaults ON at "
        "single-stage parity (STAGE2_HARDWARE_KEEPOUT_WEIGHT_DEFAULT) so an "
        "unconfigured Stage-2 run feels the fixed hardware cloud; pass 0 (or "
        "export STAGE2_HARDWARE_KEEPOUT_WEIGHT=0) to disable the term entirely "
        "for legacy/byte-identical reproduction. In ALM mode this Jhardware term "
        "is promoted to the zero-slack hardware_keepout constraint row.",
    )
    parser.add_argument(
        "--stage2-hardware-keepout-backend",
        choices=("point_cloud", "sdf"),
        default=os.environ.get("STAGE2_HARDWARE_KEEPOUT_BACKEND", "point_cloud"),
        help=(
            "Backend for the Stage-2 in-loop hardware keep-out term. "
            "point_cloud preserves the historical hardware_keepout.json cloud; "
            "sdf samples the Type KK swept U-channel surface against a "
            "CAD-derived SDF proxy. The posthoc hardware_contact_report CAD "
            "oracle remains required."
        ),
    )
    parser.add_argument(
        "--stage2-hardware-keepout-json",
        type=str,
        default=os.environ.get(
            "STAGE2_HARDWARE_KEEPOUT_JSON",
            DEFAULT_HARDWARE_KEEPOUT_JSON_PATH,
        ),
        help="Path to the hardware_keepout.json point cloud "
        "(hbt_clearance_viewer/tools/export_hardware_keepout.py), the same "
        "cloud the single-stage --hardware-keepout-json consumes. Required when "
        "--stage2-hardware-keepout-weight is > 0.",
    )
    parser.add_argument(
        "--stage2-hardware-keepout-sdf-manifest",
        type=str,
        default=os.environ.get("STAGE2_HARDWARE_KEEPOUT_SDF_MANIFEST"),
        help=(
            "Path to hardware_sdf.json when "
            "--stage2-hardware-keepout-backend=sdf. The manifest references the "
            "SDF data payload and sha-binds it to the CAD GLB; it is optimizer "
            "proxy evidence only."
        ),
    )
    parser.add_argument(
        "--stage2-hardware-keepout-glb",
        type=str,
        default=os.environ.get(
            "STAGE2_HARDWARE_KEEPOUT_GLB",
            DEFAULT_HARDWARE_KEEPOUT_GLB_PATH,
        ),
        help="Path to the live hbt_assembly.glb used to fail-closed validate "
        "the hardware_keepout.json cloud when --stage2-hardware-keepout-weight "
        "is > 0.",
    )
    parser.add_argument(
        "--stage2-hardware-keepout-alm-scale",
        type=float,
        default=(
            float(os.environ["STAGE2_HARDWARE_KEEPOUT_ALM_SCALE"])
            if "STAGE2_HARDWARE_KEEPOUT_ALM_SCALE" in os.environ
            else None
        ),
        help="Per-constraint ALM normalization scale for the hardware keep-out "
        "constraint row (default: the schema BANANA_HARDWARE_KEEPOUT_ALM_SCALE). "
        "Smaller values make the augmented Lagrangian weight the keep-out row "
        "more heavily relative to the other constraints; larger values soften it. "
        "Effective ONLY under --constraint-method alm with "
        "--stage2-hardware-keepout-weight > 0 (in penalty mode the keep-out is a "
        "weighted objective term and the weight is the dial instead). Tunes ALM "
        "convergence emphasis only — it does NOT change the zero-slack threshold "
        "(0) or the contract-pinned min-distance, so the keep-out safety floor is "
        "unchanged. Must be > 0.",
    )
    parser.add_argument(
        "--stage2-hardware-keepout-tolerance",
        type=float,
        default=(
            float(os.environ["STAGE2_HARDWARE_KEEPOUT_TOLERANCE"])
            if "STAGE2_HARDWARE_KEEPOUT_TOLERANCE" in os.environ
            else None
        ),
        help="ALM activity tolerance for the hardware keep-out constraint row "
        "(default: 1e-6, matching self_intersect). This is the band within which "
        "the row is treated as active for the augmented Lagrangian's adaptive "
        "smoothing / activity diagnostics; smaller values classify the row as "
        "active at smaller violations. Effective ONLY under --constraint-method "
        "alm with --stage2-hardware-keepout-weight > 0. Like the ALM scale, it "
        "tunes ALM convergence/diagnostics only — not the zero-slack threshold or "
        "the contract-pinned min-distance.",
    )
    parser.add_argument(
        "--stage2-resonant-flux-weight",
        type=float,
        default=float(os.environ.get("STAGE2_RESONANT_FLUX_WEIGHT", "0.0")),
        help="Audit-8 static resonant reweighting weight w_res: adds "
        "w_res * J_res to the objective, where J_res is the FFT spectral "
        "power of B.n restricted to the harmonics of the selected low-order "
        "rationals within --stage2-resonant-delta of the resonant iota target "
        "(island suppression at the source). The mode mask is computed ONCE "
        "at setup (static; no in-loop Boozer dependence). 0 disables the "
        "term entirely (default; legacy-identical objective). In ALM mode "
        "the weighted term rides the smooth objective rather than adding a "
        "constraint row.",
    )
    parser.add_argument(
        "--stage2-resonant-iota-target",
        type=float,
        default=(
            None
            if os.environ.get("STAGE2_RESONANT_IOTA_TARGET") is None
            else float(os.environ["STAGE2_RESONANT_IOTA_TARGET"])
        ),
        help="Iota target for the audit-8 resonant reweighting mode "
        "selection. When omitted, --stage2-iota-target is reused; one of "
        "the two is required whenever --stage2-resonant-flux-weight > 0.",
    )
    parser.add_argument(
        "--stage2-resonant-delta",
        type=float,
        default=float(os.environ.get("STAGE2_RESONANT_DELTA", "0.02")),
        help="Half-width of the rational window |p/q - iota_target| <= delta "
        "for the audit-8 resonant reweighting (default 0.02, matching the "
        "campaign's resonance-window evidence).",
    )
    parser.add_argument(
        "--stage2-resonant-qmax",
        type=int,
        default=int(
            os.environ.get("STAGE2_RESONANT_QMAX", str(MAX_RESONANT_DENOMINATOR))
        ),
        help="Maximum rational denominator q for the audit-8 resonant "
        f"reweighting (hard cap {MAX_RESONANT_DENOMINATOR}; larger requests "
        "raise). The default cap is calibrated for the current low-iota "
        "campaign band, not the older iota~0.30 regime.",
    )
    args = parser.parse_args()
    try:
        validate_banana_current_cli_args(args)
        validate_stage2_tf_current_cli_args(args)
        validate_stage2_iota_cli_args(args)
        validate_stage2_edge_iota_cli_args(args)
        validate_s_hel_objective_cli_args(args)
        validate_finite_build_cli_args(args)
        validate_stage2_vessel_keepout_cli_args(args)
        validate_stage2_resonant_flux_cli_args(args)
        validate_stage2_buildability_objective_cli_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def build_equilibrium_path(args):
    if args.equilibrium_path is not None:
        return args.equilibrium_path

    candidate_paths = [
        os.path.join(args.equilibria_dir, args.plasma_surf_filename),
        os.path.join(DATABASE_EQUILIBRIA_DIR, args.plasma_surf_filename),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return candidate_paths[0]


def build_hbt_reference_surfaces(
    nfp,
    banana_surf_radius,
    *,
    winding_surface_free_mpol=0,
    winding_surface_free_ntor=0,
):
    surfaces = build_banana_reference_surfaces(
        nfp,
        banana_surf_radius,
        coil_winding_surface_mpol=max(1, int(winding_surface_free_mpol)),
        coil_winding_surface_ntor=max(0, int(winding_surface_free_ntor)),
    )
    return (
        surfaces.lcfs_clearance_reference,
        surfaces.coil_winding_surface,
        surfaces.vessel,
    )


def build_stage2_results_sidecar_path(stage2_bs_artifact_path):
    return os.path.join(os.path.dirname(stage2_bs_artifact_path), "results.json")


def build_stage2_secondary_artifact_paths(stage2_bs_artifact_path):
    secondary_root = (
        Path(stage2_bs_artifact_path).parent / SECONDARY_STAGE2_ARTIFACT_DIRNAME
    )
    return (
        str(secondary_root / "biot_savart_opt.json"),
        str(secondary_root / "results.json"),
    )


def build_stage2_secondary_artifact_metadata(
    *,
    secondary_stage2_bs_path=None,
    secondary_stage2_results_path=None,
    secondary_source=None,
):
    preserved = (
        secondary_stage2_bs_path is not None
        and secondary_stage2_results_path is not None
    )
    return {
        "STAGE2_SECONDARY_ARTIFACT_PRESERVED": preserved,
        "STAGE2_SECONDARY_ARTIFACT_REASON": (
            SECONDARY_STAGE2_ARTIFACT_REASON if preserved else None
        ),
        "STAGE2_SECONDARY_ARTIFACT_SOURCE": secondary_source if preserved else None,
        "STAGE2_SECONDARY_BS_PATH": secondary_stage2_bs_path,
        "STAGE2_SECONDARY_RESULTS_PATH": secondary_stage2_results_path,
    }


def append_termination_suffix(termination_message, suffix):
    if termination_message:
        return f"{termination_message}; {suffix}"
    return suffix


def build_secondary_stage2_results_kwargs(
    *,
    stage2_results_kwargs,
    secondary_state,
    tf_current_A,
    new_banana_curve,
    new_surf,
    termination_message,
):
    return stage2_results_kwargs | {
        "alm_result": None,
        "banana_current_A": secondary_state["banana_current_A"],
        "banana_to_tf_current_ratio": secondary_state["banana_current_A"]
        / tf_current_A,
        "termination_message": append_termination_suffix(
            termination_message,
            SECONDARY_STAGE2_TERMINATION_SUFFIX,
        ),
        "optimizer_success": False,
        "final_volume": new_surf.volume(),
        "intersecting": is_self_intersecting(new_banana_curve),
        "final_max_curvature": secondary_state["max_curvature"],
        "final_coil_length": secondary_state["coil_length"],
        "final_curve_curve_min_dist": secondary_state["curve_curve_min_dist"],
        "final_curve_surface_min_dist": secondary_state["curve_surface_min_dist"],
        "final_coil_width": secondary_state["coil_width"],
        "final_self_intersect_penalty": secondary_state["self_intersect_penalty"],
        "final_shortest_self_distance": secondary_state["shortest_self_distance"],
        "hardware_status": secondary_state["hardware_status"],
    }


def _interpolated_field_for_topology_bridge(biotsavart, surface, *, nfp: int):
    """Wrap ``biotsavart`` in an ``InterpolatedField`` sized to the surface.

    H1 fix: the legacy code path called the field-line tracer directly
    with the raw ``BiotSavart`` field. Every integrator step re-evaluated
    the full ``O(N_coils * quadpoints)`` Biot-Savart sum, which for a
    typical 25-coil / 80-quadpoint configuration over ~5 seed lines x
    tmax=3000 (~5000 timesteps each) cost millions of evaluations. The
    InterpolatedField wrapper builds a cubic-spline lookup table once
    (40 x 40 x 20 over the surface envelope, matching the recipe used
    by ``scripts/fieldline_iota_proxy_convergence.py:83-103``); the
    tracer then samples a constant-cost spline per step.

    The grid is sized via :func:`topology_scorer.padded_bounds` so the
    interpolant envelope strictly contains the working surface and the
    escape cage; integrator excursions that leave the box are caught by
    the escape-cage stopping criteria (set up by the Phase 3a path).
    The interpolant is ``degree=3`` (cubic) and stellarator-symmetric
    when the upstream surface is.
    """
    from simsopt.field import InterpolatedField

    gamma = np.asarray(surface.gamma(), dtype=float)
    R = np.sqrt(gamma[..., 0] ** 2 + gamma[..., 1] ** 2)
    rmin = float(np.min(R))
    rmax = float(np.max(R))
    zmax = float(np.max(np.abs(gamma[..., 2])))
    irmin, irmax, izmax = padded_bounds(rmin, rmax, zmax)
    stellsym = bool(getattr(surface, "stellsym", True))
    zrange = (0.0, izmax, 20) if stellsym else (-izmax, izmax, 20)
    return InterpolatedField(
        biotsavart,
        3,
        (irmin, irmax, 40),
        (0.0, 2.0 * np.pi / int(max(nfp, 1)), 40),
        zrange,
        True,
        nfp=int(max(nfp, 1)),
        stellsym=stellsym,
    )


def build_stage2_iota_hot_loop_payload(
    *,
    args,
    stage2_iota_runtime,
):
    constraint_weight = canonical_stage2_iota_constraint_weight(
        getattr(
            args,
            "stage2_iota_constraint_weight",
            DEFAULT_STAGE2_IOTA_CONSTRAINT_WEIGHT,
        )
    )
    payload = {
        "STAGE2_IOTA_EFFECTIVE_WEIGHT": None,
        "STAGE2_IOTA_OBJECTIVE_MODE": str(
            getattr(args, "stage2_iota_objective_mode", "report")
        ),
        "STAGE2_IOTA_VOL_TARGET": float(
            getattr(args, "stage2_iota_vol_target", DEFAULT_STAGE2_IOTA_VOL_TARGET)
        ),
        "STAGE2_IOTA_CONSTRAINT_WEIGHT": constraint_weight,
        "STAGE2_IOTA_NUM_TF_COILS": int(
            getattr(args, "stage2_iota_num_tf_coils", DEFAULT_STAGE2_IOTA_NUM_TF_COILS)
        ),
        "STAGE2_IOTA_NPHI": int(
            getattr(args, "stage2_iota_nphi", DEFAULT_STAGE2_IOTA_NPHI)
        ),
        "STAGE2_IOTA_NTHETA": int(
            getattr(args, "stage2_iota_ntheta", DEFAULT_STAGE2_IOTA_NTHETA)
        ),
        "STAGE2_IOTA_MPOL": int(
            getattr(args, "stage2_iota_mpol", DEFAULT_STAGE2_IOTA_MPOL)
        ),
        "STAGE2_IOTA_NTOR": int(
            getattr(args, "stage2_iota_ntor", DEFAULT_STAGE2_IOTA_NTOR)
        ),
        "STAGE2_IOTA_OBJECTIVE_COUPLED": False,
        "STAGE2_IOTA_HOT_LOOP_ENABLED": False,
        "STAGE2_IOTA_BOOTSTRAP_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_CALLS": None,
        "STAGE2_IOTA_INITIAL": None,
        "STAGE2_IOTA_INITIAL_PENALTY": None,
        "STAGE2_IOTA_FINAL": None,
        "STAGE2_IOTA_FINAL_PENALTY": None,
        "STAGE2_IOTA_FINAL_ABS_ERROR": None,
        "STAGE2_IOTA_FINAL_FEASIBLE": None,
        "STAGE2_IOTA_FINAL_SOLVE_FAILED": None,
        "STAGE2_IOTA_VALUE": None,
        "STAGE2_IOTA_PENALTY": None,
        "STAGE2_IOTA_ABS_ERROR": None,
        "STAGE2_IOTA_FEASIBLE": None,
        "STAGE2_IOTA_PENALTY_THRESHOLD": None,
        # Phase 1 Boozer residual trust gate (Stage 2 / single-stage handoff
        # plan). Always emitted so the artifact contract is identical whether
        # the optional post-run iota probe is requested or skipped.
        "BOOZER_SOLVE_SUCCESS": None,
        "BOOZER_SELF_INTERSECTING": None,
        "BOOZER_CONSTRAINED_RESIDUAL_NORM": None,
        "BOOZER_TRUSTED": None,
        "IOTA_OBJECTIVE_ACTIVE": None,
        "BOOZER_TRUST_REASON": None,
        "BOOZER_TRUST_TOL": None,
        # Phase 3 non-Boozer topology bridge diagnostics. Populated post-solve
        # when the bridge is invoked; left ``None`` otherwise so the artifact
        # schema is identical across lanes.
        # FIELDLINE_IOTA_PROXY{,_VALID,_N_TRANSITS,_REASON} are emitted by
        # Phase 3a (banana_opt.topology_bridge); HELICAL_FIELD_CONTENT,
        # and PRE_BOOZER_TOPOLOGY_SCORE are emitted by Phase 3b
        # (banana_opt.boozer_topology_bridge). S_HEL_OBJECTIVE_WEIGHT is also
        # set when the live S_HEL objective schedule is enabled.
        "FIELDLINE_IOTA_PROXY": None,
        "FIELDLINE_IOTA_PROXY_VALID": None,
        "FIELDLINE_IOTA_PROXY_N_TRANSITS": None,
        "FIELDLINE_IOTA_PROXY_REASON": None,
        "HELICAL_FIELD_CONTENT": None,
        "S_HEL_OBJECTIVE_WEIGHT": None,
        "PRE_BOOZER_TOPOLOGY_SCORE": None,
    }
    if stage2_iota_runtime is None:
        return payload

    runtime_active = stage2_iota_runtime_is_active(stage2_iota_runtime)
    final_state = stage2_iota_runtime.last_state
    payload.update(
        {
            "STAGE2_IOTA_OBJECTIVE_COUPLED": runtime_active,
            "STAGE2_IOTA_HOT_LOOP_ENABLED": runtime_active,
            "STAGE2_IOTA_BOOTSTRAP_SECONDS": stage2_iota_runtime.stats.bootstrap_seconds,
            "STAGE2_IOTA_RUNTIME_SECONDS": stage2_iota_runtime.stats.runtime_seconds,
            "STAGE2_IOTA_RUNTIME_CALLS": stage2_iota_runtime.stats.runtime_calls,
            "STAGE2_IOTA_EFFECTIVE_WEIGHT": stage2_iota_runtime.effective_weight,
            "STAGE2_IOTA_INITIAL": stage2_iota_runtime.initial_state.iota,
            "STAGE2_IOTA_INITIAL_PENALTY": stage2_iota_runtime.initial_state.penalty,
            "STAGE2_IOTA_FINAL": final_state.iota,
            "STAGE2_IOTA_FINAL_PENALTY": final_state.penalty,
            "STAGE2_IOTA_FINAL_ABS_ERROR": final_state.abs_error,
            "STAGE2_IOTA_FINAL_FEASIBLE": final_state.feasible,
            "STAGE2_IOTA_FINAL_SOLVE_FAILED": final_state.solve_failed,
            "STAGE2_IOTA_VALUE": final_state.iota,
            "STAGE2_IOTA_PENALTY": final_state.penalty,
            "STAGE2_IOTA_ABS_ERROR": final_state.abs_error,
            "STAGE2_IOTA_FEASIBLE": final_state.feasible,
            "STAGE2_IOTA_PENALTY_THRESHOLD": stage2_iota_runtime.penalty_threshold,
        }
    )
    evaluator = getattr(stage2_iota_runtime, "guarded_boozer_evaluator", None)
    trust_state = getattr(evaluator, "last_trust_state", None) if evaluator else None
    payload.update(boozer_trust_artifact_fields(trust_state))
    # Phase 3 topology bridge: compute S_HEL on the working surface using
    # the same Boozer-surface + BiotSavart pair the iota runtime is sitting
    # on. The field-line proxy is left None here because tracing is expensive
    # and lives outside the hot-loop artifact; donor-side tools can populate
    # FIELDLINE_IOTA_PROXY from the persisted Boozer artifact post-solve.
    #
    # The artifact path uses the safe wrapper because a degenerate ``|B|``
    # grid (non-finite samples, zero non-DC spectral power) on a marginal
    # Boozer surface is a documented diagnostic-unavailable case — the
    # plan ("Failed diagnostics fail closed as unavailable metrics, not as
    # fake zero-quality success", line 305) requires reporting ``None``
    # rather than crashing the solver post-optimization.
    boozer_surface = getattr(stage2_iota_runtime, "boozer_surface", None)
    surface = getattr(boozer_surface, "surface", None) if boozer_surface else None
    biotsavart = getattr(boozer_surface, "biotsavart", None) if boozer_surface else None
    # Plan line 286-291 + Phase 3b gating: persist HELICAL_FIELD_CONTENT /
    # PRE_BOOZER_TOPOLOGY_SCORE only when the topology-bridge diagnostics
    # are explicitly enabled. The artifact-schema defaults above keep the
    # keys present as None when the gate is off so downstream consumers
    # don't see a missing key. The opt-in default (False) matches the
    # ``--enable-topology-bridge-diagnostics`` CLI flag default; the
    # ``getattr`` fallback applies only when an external caller invokes
    # this helper without going through argparse.
    bridge_enabled = bool(getattr(args, "enable_topology_bridge_diagnostics", False))
    s_hel_objective_weight = resolve_s_hel_objective_weight(args)
    if s_hel_objective_weight > 0.0:
        payload["S_HEL_OBJECTIVE_WEIGHT"] = s_hel_objective_weight
    if bridge_enabled and surface is not None and biotsavart is not None:
        # Phase 3b S_HEL: evaluated directly on the raw BiotSavart field
        # (the S_HEL helper takes a single ``field.B()`` sample on a
        # static surface grid — no integrator loop, so InterpolatedField
        # caching would not amortize). The integration-site ordering
        # contract (H3) is: S_HEL runs FIRST so its ``set_points`` side
        # effect is overwritten by the tracer's per-step ``set_points``
        # below.
        s_hel_value = safe_compute_helical_field_content_S_HEL(biotsavart, surface)
        # Phase 3a: canonical post-solve field-line iota proxy. Uses the
        # banana_opt.topology_bridge module (escape-radius stopping
        # criterion + n_transits_target gating + structured failure
        # reasons). The safe wrapper is mandatory here per plan line 305:
        # tracer-exception failure must be reported as a failure tag in
        # the artifact, never as a solver crash. Plan lines 270/283: this
        # is the only field-line-proxy invocation per run and runs
        # post-solve, never per-iteration.
        gamma = np.asarray(surface.gamma(), dtype=float)
        rmax = float(np.max(np.sqrt(gamma[..., 0] ** 2 + gamma[..., 1] ** 2)))
        topology_bridge_escape_radius = getattr(
            args, "topology_bridge_escape_radius", None
        )
        if topology_bridge_escape_radius is None:
            topology_bridge_escape_radius = 1.25 * rmax
        # H1 fix: wrap the BiotSavart field in an InterpolatedField
        # before tracing so the tracer's per-step field evaluation is a
        # constant-cost cubic-spline sample rather than the full
        # ``O(N_coils * quadpoints)`` Biot-Savart sum. Cost is amortized
        # by the single grid build; the tracer call sees ~5 x tmax /
        # dt_average evaluations, easily into the millions for tmax=3000.
        nfp = int(getattr(surface, "nfp", 1) or 1)
        tracer_field = _interpolated_field_for_topology_bridge(
            biotsavart, surface, nfp=nfp
        )
        phase3a_result = safe_compute_phase3a_fieldline_iota_proxy(
            tracer_field,
            surface,
            nfieldlines=int(
                getattr(
                    args,
                    "topology_bridge_nfieldlines",
                    TOPOLOGY_BRIDGE_DEFAULT_NFIELDLINES,
                )
            ),
            tmax=float(
                getattr(
                    args,
                    "topology_bridge_tmax",
                    TOPOLOGY_BRIDGE_DEFAULT_TMAX,
                )
            ),
            tol=float(
                getattr(
                    args,
                    "topology_bridge_tol",
                    TOPOLOGY_BRIDGE_DEFAULT_TOL,
                )
            ),
            escape_radius=float(topology_bridge_escape_radius),
            n_transits_target=int(
                getattr(
                    args,
                    "topology_bridge_n_transits_target",
                    100,
                )
            ),
        )
        # Phase 3a owns the FIELDLINE_IOTA_PROXY / FIELDLINE_IOTA_PROXY_VALID
        # keys; Phase 3b emits only the helical-content + composite keys.
        # The composite consumes the Phase 3a result so the
        # PRE_BOOZER_TOPOLOGY_SCORE proximity term reflects the same iota
        # proxy that downstream consumers see in the artifact.
        payload.update(fieldline_iota_proxy_artifact_fields(phase3a_result))
        # Phase 3b: S_HEL_OBJECTIVE_WEIGHT reports the explicit schedule
        # value when the live objective is enabled, otherwise 0.0 because
        # the metric is being tracked but contributes no optimizer force.
        # ``safe_compute_phase3a_fieldline_iota_proxy`` is typed to always
        # return a ``Phase3aFieldlineIotaProxyResult`` (failure cases are
        # encoded inside the dataclass via ``valid=False`` /
        # ``iota_proxy=None`` / ``reason=...``), so we trust the type
        # signature and consume the fields directly.
        fieldline_iota_mean = phase3a_result.iota_proxy
        fieldline_iota_valid = bool(phase3a_result.valid)
        payload.update(
            boozer_topology_bridge_artifact_fields(
                s_hel=s_hel_value,
                fieldline_iota_proxy_mean=fieldline_iota_mean,
                fieldline_iota_proxy_valid=fieldline_iota_valid,
                s_hel_objective_weight=s_hel_objective_weight,
                iota_target=stage2_iota_runtime.target,
            )
        )
    return payload


def build_stage2_iota_report_payload(
    *,
    args,
    stage2_bs_artifact_path,
    stage2_results_payload,
    stage2_iota_runtime=None,
    stage2_seed_surf_path=None,
):
    probe_enabled = args.stage2_iota_target is not None
    stage2_results_path = build_stage2_results_sidecar_path(stage2_bs_artifact_path)
    recorded_stage2_seed_path = stage2_results_payload.get(
        "STAGE2_BS_PATH",
        stage2_bs_artifact_path,
    )
    recorded_stage2_seed_results_path = stage2_results_payload.get(
        "STAGE2_RESULTS_PATH"
    )
    payload = {
        "STAGE2_ROOT_FIX_ENABLED": probe_enabled,
        "STAGE2_IOTA_TARGET": (
            None if args.stage2_iota_target is None else float(args.stage2_iota_target)
        ),
        "STAGE2_IOTA_TOLERANCE": (
            None if not probe_enabled else float(args.stage2_iota_tolerance)
        ),
        "STAGE2_IOTA_PROBE_SECONDS": None,
        "BOOTABILITY_STAGE2_BS_PATH": stage2_bs_artifact_path,
        "BOOTABILITY_STAGE2_RESULTS_PATH": stage2_results_path,
        "BOOTABILITY_STAGE2_SURF_PATH": (
            None if stage2_seed_surf_path is None else str(stage2_seed_surf_path)
        ),
    }
    payload.update(
        build_stage2_iota_hot_loop_payload(
            args=args,
            stage2_iota_runtime=stage2_iota_runtime,
        )
    )
    payload.update(
        build_bootability_recovery_payload_fields(
            None,
            stage2_bs_path=recorded_stage2_seed_path,
            stage2_results_path=recorded_stage2_seed_results_path,
            include_recovery=False,
        )
    )
    if not probe_enabled:
        return payload

    probe_start = time.perf_counter()
    constraint_weight = resolve_stage2_iota_constraint_weight(
        args.stage2_iota_constraint_weight
    )
    bootability_status = probe_stage2_seed_bootability(
        stage2_bs_path=stage2_bs_artifact_path,
        stage2_artifact_results=stage2_results_payload,
        plasma_surf_filename=os.path.basename(args.plasma_surf_filename),
        equilibria_dir=args.equilibria_dir,
        equilibrium_path=build_equilibrium_path(args),
        num_tf_coils=args.stage2_iota_num_tf_coils,
        nphi=args.stage2_iota_nphi,
        ntheta=args.stage2_iota_ntheta,
        mpol=args.stage2_iota_mpol,
        ntor=args.stage2_iota_ntor,
        vol_target=args.stage2_iota_vol_target,
        iota_target=float(args.stage2_iota_target),
        iota_tolerance=args.stage2_iota_tolerance,
        constraint_weight=constraint_weight,
        stage2_seed_surf_path=stage2_seed_surf_path,
        allow_offspec_current_contract=stage2_current_contract_allows_offspec(args),
    )
    payload.update(
        build_bootability_recovery_payload_fields(
            bootability_status,
            stage2_bs_path=recorded_stage2_seed_path,
            stage2_results_path=recorded_stage2_seed_results_path,
            include_recovery=False,
        )
    )
    payload["STAGE2_IOTA_PROBE_SECONDS"] = time.perf_counter() - probe_start
    return payload


def _stage2_edge_iota_empty_payload() -> dict[str, object]:
    return {
        "EDGE_IOTA_STATUS": None,
        "EDGE_IOTA_PROFILE_JSON": None,
        "EDGE_DELTA_ABS_IOTA_MIN": None,
        "EDGE_DELTA_ABS_IOTA_P10": None,
        "EDGE_DELTA_ABS_IOTA_MEAN": None,
        "EDGE_DELTA_SIGNED_IOTA_MIN": None,
        "EDGE_DELTA_SIGNED_IOTA_MEAN": None,
        "EDGE_SURFACE_SURVIVAL_FRACTION": None,
        "EDGE_WIDTH_MAX": None,
        "EDGE_HELICITY_STATUS": EDGE_HELICITY_STATUS_UNKNOWN,
    }


def _stage2_edge_iota_config_payload(
    *,
    args,
    mode: str,
    config: EdgeIotaConfig,
) -> dict[str, object]:
    explicit_helicity = getattr(args, "stage2_edge_iota_helicity", None)
    return {
        "EDGE_IOTA_MODE": mode,
        "EDGE_IOTA_EQDSK": config.eqdsk_path,
        "EDGE_IOTA_LCFS": config.lcfs_path,
        "EDGE_IOTA_RADIAL_BAND": [
            float(config.edge_band[0]),
            float(config.edge_band[1]),
        ],
        "EDGE_IOTA_TARGET_MIN": float(config.edge_delta_abs_iota_target_min),
        "EDGE_IOTA_HELICITY": (
            None
            if explicit_helicity in {None, ""}
            else int(config.helicity_sign)
        ),
        "EDGE_IOTA_CONFIG_HASH": (
            None if mode == EDGE_IOTA_MODE_OFF else edge_iota_config_hash(config)
        ),
    }


def _biot_savart_cylindrical_field(biot_savart):
    def field(R_m: float, phi_rad: float, Z_m: float) -> tuple[float, float, float]:
        cos_phi = np.cos(float(phi_rad))
        sin_phi = np.sin(float(phi_rad))
        point = np.ascontiguousarray(
            [[float(R_m) * cos_phi, float(R_m) * sin_phi, float(Z_m)]],
            dtype=float,
        )
        biot_savart.set_points(point)
        Bx, By, Bz = biot_savart.B()[0]
        return (
            float(Bx * cos_phi + By * sin_phi),
            float(-Bx * sin_phi + By * cos_phi),
            float(Bz),
        )

    return field


def _sum_cylindrical_fields(base_field, added_field):
    def field(R_m: float, phi_rad: float, Z_m: float) -> tuple[float, float, float]:
        base_r, base_phi, base_z = base_field(R_m, phi_rad, Z_m)
        added_r, added_phi, added_z = added_field(R_m, phi_rad, Z_m)
        return (base_r + added_r, base_phi + added_phi, base_z + added_z)

    return field


def _stage2_edge_iota_banana_biot_savart(stage2_biot_savart, stage2_results_payload):
    manifest = CoilGroupsManifest.from_json_payload(
        stage2_results_payload["COIL_GROUPS"]
    )
    banana_group = manifest.by_role(COIL_GROUP_ROLE_BANANA)
    if banana_group is None or banana_group.count == 0:
        raise ValueError("Stage 2 edge-iota report requires banana coil-group metadata.")
    coils = list(stage2_biot_savart.coils)
    return BiotSavart(coils[banana_group.start : banana_group.stop])


def build_stage2_edge_iota_profile_artifact(
    *,
    config: EdgeIotaConfig,
    stage2_bs_artifact_path,
    stage2_results_payload,
    stage2_biot_savart,
):
    eqdsk = read_eqdsk(config.eqdsk_path)
    lcfs = load_lcfs_boundary(config.lcfs_path)
    minor_radius_m = lcfs.minor_radius_from_axis(float(eqdsk.rmaxis))
    tokamak_field = eqdsk.build_axisymmetric_field()
    validation = validate_tokamak_iota_against_q(
        tokamak_field,
        edge_band=config.edge_band,
        sample_count=config.sample_count,
        minor_radius_m=minor_radius_m,
        relative_tolerance=config.q_validation_rel_tol,
        turns=config.trace_turns,
        steps_per_turn=config.steps_per_turn,
    )
    if not validation.passed:
        raise ValueError(
            "Stage 2 edge-iota tokamak-only trace failed EQDSK q-profile validation."
        )
    banana_biot_savart = _stage2_edge_iota_banana_biot_savart(
        stage2_biot_savart,
        stage2_results_payload,
    )
    banana_field = _biot_savart_cylindrical_field(banana_biot_savart)
    hybrid_field = _sum_cylindrical_fields(tokamak_field, banana_field)
    profile = evaluate_edge_iota_profile(
        tokamak_field=tokamak_field,
        hybrid_field=hybrid_field,
        axis_r_m=float(eqdsk.rmaxis),
        axis_z_m=float(eqdsk.zmaxis),
        minor_radius_m=minor_radius_m,
        config=config,
        lcfs_boundary=lcfs,
    )
    profile_json_path = Path(stage2_bs_artifact_path).with_name("edge_iota_profile.json")
    write_profile_json(
        profile,
        profile_json_path,
        config=config,
        eqdsk=eqdsk,
        validation=validation,
    )
    return profile, str(profile_json_path)


def build_stage2_edge_iota_report_payload(
    *,
    args,
    stage2_bs_artifact_path,
    stage2_results_payload,
    stage2_biot_savart,
) -> dict[str, object]:
    mode = getattr(args, "stage2_edge_iota_mode", EDGE_IOTA_MODE_OFF)
    if mode == EDGE_IOTA_MODE_OFF:
        return {
            "EDGE_IOTA_MODE": EDGE_IOTA_MODE_OFF,
            "EDGE_IOTA_EQDSK": getattr(args, "stage2_edge_iota_eqdsk", None),
            "EDGE_IOTA_LCFS": getattr(args, "stage2_edge_iota_lcfs", None),
            "EDGE_IOTA_RADIAL_BAND": None,
            "EDGE_IOTA_TARGET_MIN": None,
            "EDGE_IOTA_HELICITY": None,
            "EDGE_IOTA_CONFIG_HASH": None,
            **_stage2_edge_iota_empty_payload(),
        }
    config = build_stage2_edge_iota_config(args)
    payload = _stage2_edge_iota_config_payload(args=args, mode=mode, config=config)
    profile, profile_json_path = build_stage2_edge_iota_profile_artifact(
        config=config,
        stage2_bs_artifact_path=stage2_bs_artifact_path,
        stage2_results_payload=stage2_results_payload,
        stage2_biot_savart=stage2_biot_savart,
    )
    payload.update(edge_iota_report_payload(profile, profile_json_path=profile_json_path))
    return payload


def stage2_warm_start_boozer_surface_path(stage2_bs_artifact_path):
    artifact_path = Path(stage2_bs_artifact_path)
    variant = artifact_path.stem.removeprefix("biot_savart_opt")
    return artifact_path.with_name(f"surf_opt{variant}_boozer_surface.json")


def save_stage2_warm_start_boozer_surface(stage2_iota_runtime, stage2_bs_artifact_path):
    if stage2_iota_runtime is None:
        return None
    warm_start_path = stage2_warm_start_boozer_surface_path(stage2_bs_artifact_path)
    state_path = save_boozer_surface_with_state(
        stage2_iota_runtime.boozer_surface,
        warm_start_path,
    )
    return warm_start_path if state_path is not None else None


def materialize_stage2_artifact_results(
    *,
    args,
    stage2_bs_artifact_path,
    results_kwargs,
    stage2_iota_runtime,
    new_bs,
    new_surf,
    constraint_metadata,
):
    if constraint_metadata is None:
        raise ValueError(
            "materialize_stage2_artifact_results requires constraint_metadata; "
            "call build_stage2_constraint_artifact_metadata before writing the "
            "Stage 2 artifact so CONTRACT_HASH, CONSTRAINT_PROFILE, EFFECTIVE_VALUES, "
            "OVERRIDE_REASON and CONTRACT_SCHEMA_VERSION are always persisted."
        )
    artifact_output_root = os.path.dirname(stage2_bs_artifact_path)
    os.makedirs(artifact_output_root, exist_ok=True)
    validate_stage2_coil_partition_counts(
        total_coils=len(new_bs.coils),
        num_tf_coils=results_kwargs["num_tf_coils"],
        num_banana_coils=results_kwargs["num_banana_coils"],
        num_proxy_coils=results_kwargs["num_proxy_coils"],
        num_vf_coils=results_kwargs["num_vf_coils"],
        context="Stage 2 artifact writer partition metadata",
    )
    new_bs.save(stage2_bs_artifact_path)
    field_error = _magnetic_field_plots(new_surf, new_bs, artifact_output_root + "/")
    results = _build_stage2_results_impl(
        **results_kwargs,
        field_error=field_error,
    )
    results[STAGE2_BS_SHA256_KEY] = compute_stage2_bs_sha256(stage2_bs_artifact_path)
    results.update(constraint_metadata)
    warm_start_surface_path = save_stage2_warm_start_boozer_surface(
        stage2_iota_runtime,
        stage2_bs_artifact_path,
    )
    results.update(
        build_stage2_iota_report_payload(
            args=args,
            stage2_bs_artifact_path=stage2_bs_artifact_path,
            stage2_results_payload=results,
            stage2_iota_runtime=stage2_iota_runtime,
            stage2_seed_surf_path=warm_start_surface_path,
        )
    )
    results.update(
        build_stage2_edge_iota_report_payload(
            args=args,
            stage2_bs_artifact_path=stage2_bs_artifact_path,
            stage2_results_payload=results,
            stage2_biot_savart=new_bs,
        )
    )
    return results


def build_stage2_constraint_artifact_metadata(
    *,
    args,
    tf_current_A,
    banana_current_max_A,
    length_target,
    target_lcfs_max_major_radius_m,
    target_lcfs_max_minor_radius_m,
    cc_threshold,
    curvature_threshold,
    banana_surf_radius,
    profile_name=None,
    override_reason=None,
):
    """Route validated Stage 2 solver values through the shared contract."""
    cli_overrides = {
        "tf_current_A": float(tf_current_A),
        "banana_current_max_A": float(banana_current_max_A),
        "length_target": float(length_target),
        "target_lcfs_max_major_radius_m": float(target_lcfs_max_major_radius_m),
        "target_lcfs_max_minor_radius_m": float(target_lcfs_max_minor_radius_m),
        "cc_threshold": float(cc_threshold),
        "curvature_threshold": float(curvature_threshold),
        "banana_surf_radius": float(banana_surf_radius),
    }
    contract, _trace = resolve_constraint_contract_from_wire_names(
        cli_overrides=cli_overrides,
        allow_offspec_current_contract=stage2_current_contract_allows_offspec(args),
        allow_offspec_length_contract=stage2_length_contract_allows_offspec(args),
        allow_offspec_curvature_contract=stage2_curvature_contract_allows_offspec(args),
        allow_offspec_winding_radius_contract=args.accept_offspec_winding_radius,
    )
    resolved_override_reason = override_reason
    if (
        stage2_length_contract_allows_offspec(args)
        and float(length_target) > COIL_LENGTH_HARD_LIMIT_M
    ):
        length_override_reason = "offspec_coil_length_target"
        resolved_override_reason = (
            length_override_reason
            if resolved_override_reason in {None, ""}
            else f"{resolved_override_reason};{length_override_reason}"
        )
    if (
        stage2_curvature_contract_allows_offspec(args)
        and float(curvature_threshold) > MAX_CURVATURE_INV_M
    ):
        curvature_override_reason = "offspec_curvature_threshold"
        resolved_override_reason = (
            curvature_override_reason
            if resolved_override_reason in {None, ""}
            else f"{resolved_override_reason};{curvature_override_reason}"
        )
    return build_constraint_metadata(
        contract,
        profile_name=(
            "stage2_solver" if profile_name in {None, ""} else str(profile_name)
        ),
        override_reason=resolved_override_reason,
    )


def evaluate_stage2_hardware_constraints(
    coil_length,
    length_target,
    curve_curve_min_dist,
    cc_threshold,
    max_curvature,
    curvature_threshold,
    poloidal_extent_rad=None,
    poloidal_extent_threshold_rad=None,
    self_envelope_min_dist=None,
    self_envelope_min_distance=None,
    fold_geodesic_curvature_max=None,
    fold_geodesic_curvature_limit=None,
    fold_curvature_mode="surface_geodesic",
    final_plasma_major_radius_m=None,
    final_plasma_minor_radius_m=None,
):
    return _evaluate_stage2_hardware_constraints(
        coil_length,
        length_target,
        curve_curve_min_dist,
        cc_threshold,
        max_curvature,
        curvature_threshold,
        poloidal_extent_rad=poloidal_extent_rad,
        poloidal_extent_threshold_rad=poloidal_extent_threshold_rad,
        self_envelope_min_dist=self_envelope_min_dist,
        self_envelope_min_distance=self_envelope_min_distance,
        fold_geodesic_curvature_max=fold_geodesic_curvature_max,
        fold_geodesic_curvature_limit=fold_geodesic_curvature_limit,
        fold_curvature_mode=fold_curvature_mode,
        final_plasma_major_radius_m=final_plasma_major_radius_m,
        final_plasma_minor_radius_m=final_plasma_minor_radius_m,
    )


def _stage2_poloidal_extent_rad(banana_curve):
    # B1.3: read the LIVE winding radius (curve's CWS surf) so the recorded
    # poloidal extent tracks a moved rc(0,0) under --winding-surface-free-r0;
    # falls back to the spec constant for non-CWS curves.
    return max_poloidal_extent_rad(
        banana_curve,
        live_winding_r0([banana_curve], BANANA_WINDING_SURFACE_MAJOR_RADIUS_M),
    )


def _evaluate_stage2_flux_objective_on_own_grid(Jf):
    Jf.recompute_bell()
    Jf.field.clear_cached_properties()
    return float(Jf.J())


def _normalize_rows(values):
    norms = np.linalg.norm(values, axis=1)
    return np.divide(
        values,
        norms[:, None],
        out=np.zeros_like(values, dtype=float),
        where=norms[:, None] > 0.0,
    )


def _finite_build_projected_bend_half_extent_m(finite_build, banana_curve):
    # Adopted self-intersection model: the binding bend constraint is the OUTER
    # build channel's inner-surface, so the projected half-extent uses the fixed
    # Type-KK outer-channel half-extents (depth=normal, width=binormal), not the
    # conductor-pack grid. ``finite_build.frame`` still selects whether a
    # winding-surface normal exists to project the channel onto.
    kappa = np.asarray(banana_curve.kappa(), dtype=float)
    half_n_m = float(TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M)
    half_b_m = float(TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M)
    if finite_build.frame != "surface_tangent":
        # centroid / frenet frames carry no winding-surface normal to project
        # the channel onto, so use the worst corner reach as the conservative
        # buildability bound for those frames.
        return np.full(kappa.shape, np.hypot(half_n_m, half_b_m), dtype=float)

    gamma = np.asarray(banana_curve.gamma(), dtype=float)
    gammadash = np.asarray(banana_curve.gammadash(), dtype=float)
    gammadashdash = np.asarray(banana_curve.gammadashdash(), dtype=float)
    tangent = _normalize_rows(gammadash)
    curvature_vector = gammadashdash - np.sum(
        gammadashdash * tangent, axis=1
    )[:, None] * tangent
    bend_direction = _normalize_rows(curvature_vector)
    surface_normal = np.asarray(
        surface_tangent_normal_direction(
            gamma,
            # B1.3/T1.4 value-live: read the moving winding R0 so the recorded
            # finite-build buildability diagnostics (FINITEBUILD_CURVATURE_OK /
            # _INNER_EDGE_RADIUS_M / _FRAME_AWARE_MIN_RADIUS_MARGIN_M) measure the
            # pack-projection frame about the TRUE torus under
            # --winding-surface-free-r0, not the frozen 0.903 spec constant.
            live_winding_r0([banana_curve], BANANA_WINDING_SURFACE_MAJOR_RADIUS_M),
            0.0,
        ),
        dtype=float,
    )
    normal_axis = _normalize_rows(
        surface_normal - np.sum(surface_normal * tangent, axis=1)[:, None] * tangent
    )
    binormal_axis = np.cross(tangent, normal_axis, axis=1)
    return (
        np.abs(np.sum(bend_direction * normal_axis, axis=1)) * half_n_m
        + np.abs(np.sum(bend_direction * binormal_axis, axis=1)) * half_b_m
    )


def _finite_build_artifact_metadata(
    finite_build,
    banana_curve,
    net_banana_current_A,
    *,
    cc_min_dist_m=None,
    cs_min_dist_m=None,
    cc_nominal_m=None,
    cs_nominal_m=None,
    self_envelope_min_dist_m=None,
    self_envelope_nominal_m=None,
    self_envelope_nominal_contract_m=None,
    self_envelope_sampling_margin_m=None,
    self_distance_window_m=None,
    self_envelope_mode=None,
    self_envelope_groc_radius_m=None,
    self_envelope_groc_radius_floor_m=None,
    fold_geodesic_curvature_max_inv_m=None,
    fold_geodesic_curvature_limit_inv_m=None,
    fold_geodesic_curvature_threshold_inv_m=None,
    fold_curvature_mode="surface_geodesic",
    fold_penalty=None,
    curvature_margin_m=0.0,
    pack_framedcurve=None,
):
    """results.json fields describing the finite-build banana winding pack.

    Buildability diagnostics:
    - Bend feasibility: the conductor pack is projected into each centerline bend
      plane. The inner-wire bend radius is the local centerline radius minus that
      projected half-extent and must stay above the Type KK single-filament floor.
    - Envelope clearance: coil-to-coil clearance is already a Type KK frame
      face-touch centerline floor. Coil-surface clearance still subtracts the
      plasma-facing normal half-build.
    """
    kappa = np.asarray(banana_curve.kappa(), dtype=float)
    max_curvature = float(np.max(kappa))
    curvature_radius_m = np.divide(
        1.0,
        kappa,
        out=np.full(kappa.shape, float("inf"), dtype=float),
        where=kappa > 0.0,
    )
    min_curv_radius_m = float(np.min(curvature_radius_m))
    half_n_m = float(finite_build.pack_half_extent_n_m)
    half_b_m = float(finite_build.pack_half_extent_b_m)
    projected_half_extent_m = _finite_build_projected_bend_half_extent_m(
        finite_build,
        banana_curve,
    )
    binding_half_build_m = float(np.max(projected_half_extent_m))
    inner_edge_radius_m = float(np.min(curvature_radius_m - projected_half_extent_m))
    # Adopted self-intersection model: the curvature-cap floor is the outer
    # build channel's inner-radius margin (jacket-protected), not the conductor
    # wire bend radius. ``curvature_margin_m`` is the optional steering safety
    # margin layered on top.
    inner_radius_margin_m = TYPE_KK_INNER_RADIUS_MARGIN_M + float(curvature_margin_m)
    required_centerline_radius_m = projected_half_extent_m + inner_radius_margin_m
    min_radius_margin_m = float(
        np.min(curvature_radius_m - required_centerline_radius_m)
    )
    strictest_required_radius_m = float(np.max(required_centerline_radius_m))
    frame_aware_limit_inv_m = (
        float("inf")
        if strictest_required_radius_m <= 0.0
        else 1.0 / strictest_required_radius_m
    )
    # Pack-to-plasma uses the NORMAL (radial) half-build only: the
    # channel rides tangent to the toroidal surface (BANANA_WINDING_CHANNEL_
    # ORIENTATION), so the plasma-facing extent is the channel depth/2, not the
    # corner. (Assumes the pack normal axis ~ the surface radial; documented.)
    cc_reach_m = finite_build.pack_reach_m
    cs_reach_m = half_n_m
    nfilaments = int(finite_build.nfilaments)
    metadata = {
        "FINITE_BUILD_ENABLED": True,
        "FINITEBUILD_NUMFILAMENTS_N": int(finite_build.numfilaments_n),
        "FINITEBUILD_NUMFILAMENTS_B": int(finite_build.numfilaments_b),
        "FINITEBUILD_GAPSIZE_N_M": float(finite_build.gapsize_n),
        "FINITEBUILD_GAPSIZE_B_M": float(finite_build.gapsize_b),
        "FINITEBUILD_ROTATION_ORDER": finite_build.rotation_order,
        "FINITEBUILD_FRAME": finite_build.frame,
        "FINITEBUILD_FILAMENTS_PER_BANANA": nfilaments,
        "BANANA_FILAMENT_CURRENT_A": float(net_banana_current_A) / nfilaments,
        "FINITEBUILD_PACK_HALF_EXTENT_N_M": half_n_m,
        "FINITEBUILD_PACK_HALF_EXTENT_B_M": half_b_m,
        "FINITEBUILD_BINDING_HALF_BUILD_M": binding_half_build_m,
        "FINITEBUILD_PACK_REACH_M": cc_reach_m,
        "FINITEBUILD_CS_REACH_M": cs_reach_m,
        "FINITEBUILD_MIN_CURVATURE_RADIUS_M": min_curv_radius_m,
        "FINITEBUILD_INNER_EDGE_RADIUS_M": inner_edge_radius_m,
        # Legacy results.json key; value is now the outer-channel inner-radius
        # margin (+ steering margin) used as the curvature-cap floor, not the
        # retired single-filament wire bend radius.
        "FINITEBUILD_SINGLE_FILAMENT_MIN_BEND_RADIUS_M": float(
            inner_radius_margin_m
        ),
        "FINITEBUILD_FRAME_AWARE_MAX_PROJECTED_HALF_EXTENT_M": binding_half_build_m,
        "FINITEBUILD_FRAME_AWARE_MIN_REQUIRED_CENTERLINE_RADIUS_M": (
            strictest_required_radius_m
        ),
        "FINITEBUILD_FRAME_AWARE_CURVATURE_LIMIT_INV_M": frame_aware_limit_inv_m,
        "FINITEBUILD_FRAME_AWARE_MIN_RADIUS_MARGIN_M": min_radius_margin_m,
        "FINITEBUILD_CURVATURE_OK": bool(min_radius_margin_m >= 0.0),
    }
    if cc_min_dist_m is not None and cc_nominal_m is not None:
        cc_edge_gap_m = float(cc_min_dist_m) - float(cc_nominal_m)
        metadata["FINITEBUILD_CC_ENVELOPE_MIN_DIST_M"] = float(cc_min_dist_m)
        metadata["FINITEBUILD_CC_EDGE_GAP_M"] = cc_edge_gap_m
        metadata["FINITEBUILD_CC_ENVELOPE_OK"] = bool(
            float(cc_min_dist_m) >= float(cc_nominal_m)
        )
    if cs_min_dist_m is not None and cs_nominal_m is not None:
        cs_envelope_m = float(cs_min_dist_m) - cs_reach_m
        metadata["FINITEBUILD_CS_ENVELOPE_MIN_DIST_M"] = cs_envelope_m
        metadata["FINITEBUILD_CS_ENVELOPE_OK"] = bool(
            cs_envelope_m >= float(cs_nominal_m)
        )
    if self_envelope_min_dist_m is not None and self_envelope_nominal_m is not None:
        metadata["FINITEBUILD_SELF_ENVELOPE_MIN_DIST_M"] = float(
            self_envelope_min_dist_m
        )
        metadata["FINITEBUILD_SELF_ENVELOPE_MIN_DISTANCE_M"] = float(
            self_envelope_nominal_m
        )
        if self_envelope_nominal_contract_m is not None:
            metadata["FINITEBUILD_SELF_ENVELOPE_NOMINAL_MIN_DISTANCE_M"] = float(
                self_envelope_nominal_contract_m
            )
        if self_envelope_sampling_margin_m is not None:
            metadata["FINITEBUILD_SELF_ENVELOPE_SAMPLING_MARGIN_M"] = float(
                self_envelope_sampling_margin_m
            )
        if self_distance_window_m is not None:
            metadata["FINITEBUILD_SELF_DISTANCE_WINDOW_M"] = float(
                self_distance_window_m
            )
        if self_envelope_mode is not None:
            metadata["FINITEBUILD_SELF_ENVELOPE_MODE"] = str(self_envelope_mode)
        if self_envelope_groc_radius_m is not None:
            metadata["FINITEBUILD_SELF_ENVELOPE_GROC_RADIUS_M"] = float(
                self_envelope_groc_radius_m
            )
        if self_envelope_groc_radius_floor_m is not None:
            metadata["FINITEBUILD_SELF_ENVELOPE_GROC_RADIUS_FLOOR_M"] = float(
                self_envelope_groc_radius_floor_m
            )
        metadata["FINITEBUILD_SELF_ENVELOPE_OK"] = bool(
            float(self_envelope_min_dist_m) >= float(self_envelope_nominal_m)
        )
    if (
        fold_geodesic_curvature_max_inv_m is not None
        and fold_geodesic_curvature_limit_inv_m is not None
    ):
        fold_max_inv_m = float(fold_geodesic_curvature_max_inv_m)
        fold_limit_inv_m = float(fold_geodesic_curvature_limit_inv_m)
        fold_mode = str(fold_curvature_mode)
        metadata["FOLD_CURVATURE_MODE"] = fold_mode
        metadata["FOLD_CURVATURE_MAX_INV_M"] = fold_max_inv_m
        metadata["FOLD_CURVATURE_LIMIT_INV_M"] = fold_limit_inv_m
        if fold_mode == "material_frame_binormal":
            metadata["FOLD_MATERIAL_FRAME_BINORMAL_CURVATURE_MAX_INV_M"] = (
                fold_max_inv_m
            )
            metadata["FOLD_MATERIAL_FRAME_BINORMAL_CURVATURE_LIMIT_INV_M"] = (
                fold_limit_inv_m
            )
        else:
            metadata["FOLD_GEODESIC_CURVATURE_MAX_INV_M"] = fold_max_inv_m
            metadata["FOLD_GEODESIC_CURVATURE_LIMIT_INV_M"] = fold_limit_inv_m
        if fold_geodesic_curvature_threshold_inv_m is not None:
            fold_threshold_inv_m = float(fold_geodesic_curvature_threshold_inv_m)
            metadata["FOLD_CURVATURE_OBJECTIVE_THRESHOLD_INV_M"] = (
                fold_threshold_inv_m
            )
            if fold_mode == "material_frame_binormal":
                metadata[
                    "FOLD_MATERIAL_FRAME_BINORMAL_CURVATURE_OBJECTIVE_THRESHOLD_INV_M"
                ] = fold_threshold_inv_m
            else:
                metadata["FOLD_GEODESIC_CURVATURE_OBJECTIVE_THRESHOLD_INV_M"] = (
                    fold_threshold_inv_m
                )
        if fold_penalty is not None:
            metadata["FOLD_PENALTY"] = float(fold_penalty)
        metadata["FOLD_OK"] = bool(fold_max_inv_m <= fold_limit_inv_m)
    if pack_framedcurve is not None and finite_build.frame == "surface_tangent":
        # T3.2/G1: realized rotation-aware curvature headroom. Diagnostic only —
        # the in-loop steering cap and FINITEBUILD_CURVATURE_OK gate are
        # unchanged; this measures how much of the over-conservative-cap arclength
        # the live pack twist alpha(theta) actually makes buildable.
        metadata.update(
            rotation_aware_curvature_report(
                finite_build,
                banana_curve,
                pack_framedcurve,
                inner_radius_margin_m,
            )
        )
    return metadata


def _segment_exact_clearance_artifact_fields(objective_curves, lcfs_surf):
    """results.json fields with exact segment-based cc/cs minima (audit 5b).

    The recorded CURVE_CURVE_MIN_DIST / CURVE_SURFACE_MIN_DIST are point-cloud
    minima over the quadrature samples and can overestimate clearance reached
    BETWEEN samples. These additive capture-time keys record exact
    piecewise-linear (closed chord polyline) minima of the same curves:

    - CURVE_CURVE_MIN_DIST_SEGMENT_EXACT: exact segment-segment minimum;
      always <= CURVE_CURVE_MIN_DIST (conservative for clearance gates).
    - CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT: segment-to-surface-point-cloud
      minimum; always <= CURVE_SURFACE_MIN_DIST on the same samples. The
      surface side stays the sampled point cloud, so its residual
      discretization error direction (possible overestimate of the true
      continuous-surface distance) is unchanged from the point-cloud metric.
    """
    return {
        "CURVE_CURVE_MIN_DIST_SEGMENT_EXACT": float(
            curve_curve_min_distance_segments_m(objective_curves)
        ),
        "CURVE_CURVE_MIN_DIST_SEGMENT_EXACT_METHOD": (
            "closed_chord_polyline_segment_segment"
        ),
        "CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT": float(
            curve_surface_min_distance_segments_m(
                objective_curves,
                lcfs_surf.gamma().reshape((-1, 3)),
            )
        ),
        "CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT_METHOD": (
            "closed_chord_polyline_segment_to_surface_point_cloud"
        ),
    }


def _capture_stage2_artifact_state(
    *,
    dofs,
    JF,
    BASE_OBJECTIVE,
    Jf,
    new_bs,
    new_surf,
    Jls,
    Jccdist,
    Jcsdist,
    new_banana_curve,
    banana_current_optimizable,
    new_tf_coils,
    length_target,
    cc_threshold,
    curvature_threshold,
    coil_surface_threshold,
    plasma_vessel_min_dist,
    plasma_vessel_threshold,
    poloidal_extent_threshold_rad,
    banana_current_max_A,
    final_plasma_major_radius_m,
    final_plasma_minor_radius_m,
    stage2_iota_runtime=None,
    Jw=None,
    Jself=None,
    Jself_envelope=None,
    Jfold=None,
    length_min_target=None,
    width_min_threshold=None,
    width_max_threshold=None,
    self_intersect_min_distance=None,
    self_envelope_mode=None,
    self_envelope_min_distance=None,
    self_envelope_nominal_min_distance=None,
    self_envelope_sampling_margin=None,
    self_distance_window=None,
    self_envelope_groc_radius_floor=None,
    fold_geodesic_curvature_limit=None,
    fold_geodesic_curvature_threshold=None,
    fold_geodesic_curvature_margin_fraction=None,
    fold_curvature_mode="surface_geodesic",
):
    candidate_x = np.asarray(dofs, dtype=float).copy()
    JF.x = candidate_x
    BASE_OBJECTIVE.x = candidate_x
    coil_length = float(Jls.J())
    curve_curve_min_dist = float(Jccdist.shortest_distance())
    curve_surface_min_dist = float(Jcsdist.shortest_distance())
    max_curvature = float(np.max(new_banana_curve.kappa()))
    poloidal_extent_rad = _stage2_poloidal_extent_rad(new_banana_curve)
    banana_current_A = float(banana_current_optimizable.get_value())
    tf_current_A = float(new_tf_coils[0].current.get_value())
    coil_width = float(Jw.J())
    self_intersect_penalty = float(Jself.J())
    shortest_self_distance = float(Jself.shortest_self_distance())
    self_envelope_penalty = float(Jself_envelope.J())
    self_envelope_min_dist = float(Jself_envelope.shortest_self_distance())
    shortest_groc = getattr(Jself_envelope, "shortest_groc", None)
    self_envelope_groc_radius = (
        None if shortest_groc is None else float(shortest_groc())
    )
    fold_penalty = None if Jfold is None else float(Jfold.J())
    fold_geodesic_curvature_max = (
        None if Jfold is None else float(Jfold.max_abs_frame_binormal_curvature())
    )
    fold_ok = (
        None
        if fold_geodesic_curvature_max is None
        or fold_geodesic_curvature_limit is None
        else bool(
            fold_geodesic_curvature_max
            <= float(fold_geodesic_curvature_limit)
        )
    )
    hardware_status = _evaluate_stage2_hardware_constraints(
        coil_length,
        length_target,
        curve_curve_min_dist,
        cc_threshold,
        max_curvature,
        curvature_threshold,
        curve_surface_min_dist=curve_surface_min_dist,
        coil_surface_threshold=coil_surface_threshold,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        plasma_vessel_threshold=plasma_vessel_threshold,
        poloidal_extent_rad=poloidal_extent_rad,
        poloidal_extent_threshold_rad=poloidal_extent_threshold_rad,
        coil_width=coil_width,
        width_min_threshold=width_min_threshold,
        width_max_threshold=width_max_threshold,
        self_intersect_penalty=self_intersect_penalty,
        self_intersect_threshold=0.0,
        shortest_self_distance=shortest_self_distance,
        self_intersect_min_distance=self_intersect_min_distance,
        self_envelope_min_dist=self_envelope_min_dist,
        self_envelope_min_distance=self_envelope_min_distance,
        fold_geodesic_curvature_max=fold_geodesic_curvature_max,
        fold_geodesic_curvature_limit=fold_geodesic_curvature_limit,
        fold_curvature_mode=fold_curvature_mode,
        banana_current_A=banana_current_A,
        banana_current_threshold=banana_current_max_A,
        tf_current_A=tf_current_A,
        tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
        final_plasma_major_radius_m=final_plasma_major_radius_m,
        final_plasma_minor_radius_m=final_plasma_minor_radius_m,
    )
    iota_state = (
        None
        if stage2_iota_runtime is None
        else evaluate_stage2_iota_state(stage2_iota_runtime)
    )
    return {
        "x": candidate_x,
        "field_objective": _evaluate_stage2_flux_objective_on_own_grid(Jf),
        "coil_length": coil_length,
        "curve_curve_min_dist": curve_curve_min_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "max_curvature": max_curvature,
        "poloidal_extent_rad": poloidal_extent_rad,
        "banana_current_A": banana_current_A,
        "tf_current_A": tf_current_A,
        "coil_width": coil_width,
        "width_min_threshold": float(width_min_threshold),
        "width_max_threshold": float(width_max_threshold),
        "self_intersect_penalty": self_intersect_penalty,
        "self_intersect_threshold": 0.0,
        "shortest_self_distance": shortest_self_distance,
        "self_intersect_min_distance": float(self_intersect_min_distance),
        "self_envelope_mode": None if self_envelope_mode is None else str(self_envelope_mode),
        "self_envelope_penalty": self_envelope_penalty,
        "self_envelope_min_dist": self_envelope_min_dist,
        "self_envelope_min_distance": float(self_envelope_min_distance),
        "self_envelope_nominal_min_distance": (
            None
            if self_envelope_nominal_min_distance is None
            else float(self_envelope_nominal_min_distance)
        ),
        "self_envelope_sampling_margin": (
            None
            if self_envelope_sampling_margin is None
            else float(self_envelope_sampling_margin)
        ),
        "self_distance_window": (
            None if self_distance_window is None else float(self_distance_window)
        ),
        "self_envelope_groc_radius": self_envelope_groc_radius,
        "self_envelope_groc_radius_floor": (
            None
            if self_envelope_groc_radius_floor is None
            else float(self_envelope_groc_radius_floor)
        ),
        "fold_penalty": fold_penalty,
        "fold_curvature_mode": str(fold_curvature_mode),
        "fold_curvature_max": fold_geodesic_curvature_max,
        "fold_geodesic_curvature_max": fold_geodesic_curvature_max,
        "fold_curvature_limit": (
            None
            if fold_geodesic_curvature_limit is None
            else float(fold_geodesic_curvature_limit)
        ),
        "fold_geodesic_curvature_limit": (
            None
            if fold_geodesic_curvature_limit is None
            else float(fold_geodesic_curvature_limit)
        ),
        "fold_curvature_threshold": (
            None
            if fold_geodesic_curvature_threshold is None
            else float(fold_geodesic_curvature_threshold)
        ),
        "fold_geodesic_curvature_threshold": (
            None
            if fold_geodesic_curvature_threshold is None
            else float(fold_geodesic_curvature_threshold)
        ),
        "fold_geodesic_curvature_margin_fraction": (
            None
            if fold_geodesic_curvature_margin_fraction is None
            else float(fold_geodesic_curvature_margin_fraction)
        ),
        "fold_ok": fold_ok,
        "length_min_target": float(length_min_target),
        "hardware_status": hardware_status,
        "stage2_iota_value": None if iota_state is None else iota_state.iota,
        "stage2_iota_penalty": None if iota_state is None else iota_state.penalty,
        "stage2_iota_abs_error": None if iota_state is None else iota_state.abs_error,
        "stage2_iota_feasible": None if iota_state is None else iota_state.feasible,
        "stage2_iota_solve_failed": (
            None if iota_state is None else iota_state.solve_failed
        ),
    }


def load_stage2_plasma_surface(plasma_surface_path, *, expected_nfp):
    """Load a saved plasma target surface for the Stage-2 SquaredFlux field-fit.

    Warm-start coherence lever: a converged Stage-2 seed fits its OWN boozer
    surface, which is not generally reproduced by re-deriving a surface from the
    wout at a clean flux label (different volume/shape). Loading that surface
    directly as the field-fit target lets the seed's coils start at their true
    field error instead of being re-fit to a re-derived surface. The artifact must
    be a Surface whose field-period count matches the device; the major-radius /
    LCFS shell checks keep using the --equilibrium-path geometry.
    """
    path = Path(plasma_surface_path)
    if not path.is_file():
        raise ValueError(f"--stage2-plasma-surface-path file not found: {path}")
    loaded = simsopt_load(str(path))
    # Saved boozer-surface artifacts deserialize as a BoozerSurface wrapping the
    # underlying flux Surface; the SquaredFlux target is that inner Surface. A bare
    # Surface artifact has no ``.surface`` attribute and passes through unchanged.
    surface = getattr(loaded, "surface", loaded)
    if not (hasattr(surface, "nfp") and hasattr(surface, "gamma")):
        raise ValueError(
            "--stage2-plasma-surface-path did not load a Surface "
            f"(got {type(loaded).__name__})."
        )
    if int(surface.nfp) != int(expected_nfp):
        raise ValueError(
            f"--stage2-plasma-surface-path nfp {int(surface.nfp)} does not match "
            f"the device nfp {int(expected_nfp)} (from --equilibrium-path)."
        )
    return surface


def load_stage2_seed_configuration(
    seed_bs_path,
    surf,
    num_tf_coils,
    out_dir,
    *,
    stage2_results,
    seed_order_upgrade=None,
    finite_build=None,
):
    bs = load_boozer_finite_i(seed_bs_path)
    if seed_order_upgrade is not None:
        loaded_coil_partitions = partition_loaded_stage2_coils(
            bs.coils,
            stage2_results=stage2_results,
            requested_num_tf_coils=num_tf_coils,
        )
        loaded_master_banana_curve = next(
            coil.curve
            for coil in loaded_coil_partitions.banana_coils
            if isinstance(coil.curve, CurveCWSFourierCPP)
        )
        if int(seed_order_upgrade) != int(loaded_master_banana_curve.order):
            bs, _, _ = upgrade_loaded_seed_biot_savart_order(
                bs,
                banana_coils=loaded_coil_partitions.banana_coils,
                tf_coils=loaded_coil_partitions.tf_coils,
                proxy_coils=loaded_coil_partitions.proxy_coils,
                vf_coils=loaded_coil_partitions.vf_coils,
                new_order=int(seed_order_upgrade),
            )

    coil_partitions = partition_loaded_stage2_coils(
        bs.coils,
        stage2_results=stage2_results,
        requested_num_tf_coils=num_tf_coils,
    )
    tf_coils = list(coil_partitions.tf_coils)
    banana_coils = list(coil_partitions.banana_coils)
    proxy_coils = list(coil_partitions.proxy_coils)
    vf_coils = list(coil_partitions.vf_coils)
    banana_curve = banana_coils[0].curve

    if finite_build is not None:
        # Warm-start finite build: rebuild the multi-filament pack from the seed's
        # MASTER banana curve (the identity-symmetry CurveCWSFourierCPP) and the
        # seed's NET banana current, then re-expand by symmetry. The seed bs holds
        # the already-symmetry-expanded thin banana pack, so only the master curve
        # and its net current carry over; the seed TF/proxy/VF coils are reused
        # verbatim. This mirrors the cold finite-build path in initialize_coils.
        if coil_partitions.finite_current_mode == JHALPERN30_FINITE_CURRENT_MODE:
            # Mirror the cold-path guard: the jhalpern30 banana current path has
            # no finite-build pack. Caught here too because the seed mode can be
            # resolved from the artifact rather than the CLI flag.
            raise ValueError(
                "Finite-build optimization is not supported with a jhalpern30 "
                "Stage 2 seed."
            )
        if not isinstance(banana_curve, CurveCWSFourierCPP):
            raise ValueError(
                "Finite-build warm start requires a CurveCWSFourierCPP master "
                "banana curve in the Stage 2 seed."
            )
        seed_net_banana_current_A = float(banana_coils[0].current.get_value())
        banana_coils = build_finite_build_banana_coils(
            banana_curve,
            seed_net_banana_current_A,
            finite_build,
            banana_curve.surf,
        )
        bs = BiotSavart(tf_coils + banana_coils + proxy_coils + vf_coils)

    bs.set_points(surf.gamma().reshape((-1, 3)))
    coils = bs.coils
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir + "curves_init", close=True)
    unitn = surf.unitnormal()
    pointData = {"B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]}
    surf.to_vtk(out_dir + "surf_init", extra_data=pointData)

    return bs, curves, banana_curve, banana_coils, tf_coils, proxy_coils, vf_coils


def load_stage2_seed_results(seed_bs_path, *, known_tf_current_A):
    stage2_results_path, loaded_results = load_stage2_artifact_results(seed_bs_path)
    return stage2_results_path, upgrade_legacy_stage2_artifact_results(
        loaded_results,
        known_num_tf_coils=20,
        known_tf_current_A=known_tf_current_A,
    )


def _resolve_seeded_numeric_field(
    cli_value,
    artifact_value,
    *,
    field_name,
    allow_override=False,
):
    if cli_value is None:
        return 0.0 if artifact_value is None else float(artifact_value)
    if allow_override:
        return float(cli_value)
    if artifact_value is None:
        return float(cli_value)
    if not np.isclose(float(cli_value), float(artifact_value), rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"{field_name}={float(cli_value):.6f} does not match the loaded Stage 2 "
            f"artifact metadata value {float(artifact_value):.6f}."
        )
    return float(cli_value)


def _resolve_seeded_path_field(cli_value, artifact_value, *, field_name):
    if cli_value in {None, ""}:
        return artifact_value
    if artifact_value in {None, ""}:
        return cli_value
    if str(cli_value) != str(artifact_value):
        raise ValueError(
            f"{field_name}={cli_value!r} does not match the loaded Stage 2 artifact "
            f"metadata value {artifact_value!r}."
        )
    return cli_value


def _is_legacy_zero_vf_donor(stage2_results):
    vf_current_A = stage2_results.get("VF_CURRENT_A")
    num_vf_coils = stage2_results.get("NUM_VF_COILS")
    return (
        stage2_results.get("VF_TEMPLATE_PATH") in {None, ""}
        and float(0.0 if vf_current_A is None else vf_current_A) == 0.0
        and int(0 if num_vf_coils is None else num_vf_coils) == 0
    )


def _resolve_stage2_finite_current_config(
    args,
    *,
    stage2_results,
) -> Stage2FiniteCurrentConfig:
    requested_finite_current_mode = getattr(
        args,
        "finite_current_mode",
        DEFAULT_FINITE_CURRENT_MODE,
    )
    allow_seed_current_traversal = bool(
        getattr(args, "stage2_seed_current_traversal", False)
    )
    requested_proxy_plasma_current_A = getattr(args, "proxy_plasma_current_A", None)
    requested_vf_current_A = getattr(args, "vf_current_A", None)
    requested_vf_template_path = getattr(args, "vf_template_path", None)
    finite_current_mode = resolve_finite_current_mode(
        requested_finite_current_mode,
        artifact_mode=(
            None
            if stage2_results is None
            else stage2_results.get("FINITE_CURRENT_MODE")
        ),
        artifact_mode_source=(
            None
            if stage2_results is None
            else stage2_results.get("FINITE_CURRENT_MODE_SOURCE")
        ),
    )
    finite_current_profile = get_finite_current_profile(finite_current_mode)
    validate_vf_current_optimizer_config(
        finite_current_mode=finite_current_mode,
        constraint_method=getattr(args, "constraint_method", "penalty"),
        init_only=getattr(args, "init_only", False),
    )
    if stage2_results is None:
        # Fresh Stage 2: auto-resolve the bundled VF template so the zero-current
        # VF bundle is always serialized. This is the Wataru-faithful shape;
        # jhalpern30 resolves its historical 20-coil template instead.
        proxy_plasma_current_A = (
            0.0
            if requested_proxy_plasma_current_A is None
            else float(requested_proxy_plasma_current_A)
        )
        if finite_current_profile.mode == JHALPERN30_FINITE_CURRENT_MODE:
            vf_current_A = resolve_jhalpern30_fresh_vf_current_A(
                proxy_plasma_current_A=proxy_plasma_current_A,
                requested_vf_current_A=requested_vf_current_A,
            )
        elif requested_vf_current_A is None:
            vf_current_A = 0.0
        else:
            vf_current_A = float(requested_vf_current_A)
        vf_template_path = resolve_finite_current_vf_template_path(
            finite_current_mode,
            requested_vf_template_path,
        )
    else:
        # Seeded restart: trust the donor artifact verbatim. Legacy zero-VF
        # donors must stay zero-VF — silently upgrading their VF_TEMPLATE_PATH
        # to the bundled default would desync artifact metadata from the
        # actual bs.coils layout (which partition_loaded_stage2_coils slices
        # from the saved BiotSavart). A dedicated migration tool can opt-in to
        # promoting legacy donors to the full-VF shape.
        proxy_plasma_current_A = _resolve_seeded_numeric_field(
            requested_proxy_plasma_current_A,
            stage2_results.get("PROXY_PLASMA_CURRENT_A"),
            field_name="--proxy-plasma-current-A",
            allow_override=allow_seed_current_traversal,
        )
        vf_current_A = _resolve_seeded_numeric_field(
            requested_vf_current_A,
            stage2_results.get("VF_CURRENT_A"),
            field_name="--vf-current-A",
            allow_override=allow_seed_current_traversal,
        )
        if (
            allow_seed_current_traversal
            and finite_current_profile.mode == JHALPERN30_FINITE_CURRENT_MODE
            and requested_proxy_plasma_current_A is not None
            and requested_vf_current_A is None
        ):
            vf_current_A = (
                proxy_plasma_current_A * finite_current_profile.vf_current_ratio
            )
        if requested_vf_template_path not in {None, ""} and _is_legacy_zero_vf_donor(
            stage2_results
        ):
            raise ValueError(
                "Legacy zero-VF Stage 2 donors cannot override --vf-template-path "
                "on restart; migrate the artifact to a full-VF layout first."
            )
        vf_template_path = _resolve_seeded_path_field(
            requested_vf_template_path,
            stage2_results.get("VF_TEMPLATE_PATH"),
            field_name="--vf-template-path",
        )

    proxy_plasma_current_A, vf_current_A = (
        validate_proxy_vf_current_convention_for_mode(
            finite_current_mode,
            proxy_plasma_current_A=proxy_plasma_current_A,
            vf_current_A=vf_current_A,
        )
    )
    vf_current_max_A = validate_vf_current_bound_config(
        vf_current_A=vf_current_A,
        vf_current_max_A=getattr(
            args,
            "vf_current_max_A",
            BANANA_CURRENT_HARD_LIMIT_A,
        ),
        finite_current_mode=finite_current_mode,
    )
    if vf_template_path in {None, ""} and (
        vf_current_A != 0.0
        or finite_current_profile.mode == JHALPERN30_FINITE_CURRENT_MODE
    ):
        raise ValueError(
            "--vf-template-path is required when --vf-current-A is non-zero."
        )
    return Stage2FiniteCurrentConfig(
        finite_current_mode=finite_current_mode,
        proxy_plasma_current_A=proxy_plasma_current_A,
        vf_current_A=vf_current_A,
        vf_current_max_A=vf_current_max_A,
        vf_template_path=vf_template_path,
        boozer_current_convention=resolve_boozer_current_convention(
            finite_current_mode,
        ),
    )


def _assign_scalar_current_value(current, value):
    if hasattr(current, "set_dofs"):
        current.set_dofs([float(value)])
        return current
    current_optimizable, scale = unwrap_current_optimizable(current)
    if scale == 0.0:
        raise ValueError("Current scale must be non-zero for current retargeting.")
    current_optimizable.set_dofs([float(value) / scale])
    return current_optimizable


def _realized_vf_current_A(vf_current_control, fallback_vf_current_A):
    if vf_current_control is None:
        return float(fallback_vf_current_A)
    return float(vf_current_control.get_value())


def _assign_fixed_scalar_current(current, value):
    current_optimizable = _assign_scalar_current_value(current, value)
    current_optimizable.fix_all()


def _retarget_stage2_seed_auxiliary_currents(
    proxy_coils,
    vf_coils,
    *,
    proxy_plasma_current_A,
    vf_current_A,
    vf_current_mutability="independent_fixed_current",
):
    loaded_proxy_current_A = (
        float(proxy_coils[0].current.get_value()) if len(proxy_coils) == 1 else None
    )
    if len(proxy_coils) == 1:
        _assign_fixed_scalar_current(proxy_coils[0].current, proxy_plasma_current_A)
    elif abs(float(proxy_plasma_current_A)) > 1.0e-12:
        raise ValueError(
            "Stage 2 seed current traversal requires one loaded proxy coil when "
            "--proxy-plasma-current-A is non-zero."
        )

    if vf_current_mutability == "shared_unfixed_scaled_current":
        vf_current_control = shared_vf_current_control_for_coils(vf_coils)
        if vf_current_control is None:
            if abs(float(vf_current_A)) > 1.0e-12:
                raise ValueError(
                    "Stage 2 seed current traversal requires loaded VF coils when "
                    "--vf-current-A is non-zero."
                )
            return
        if loaded_proxy_current_A is None or loaded_proxy_current_A == 0.0:
            raise ValueError(
                "Stage 2 seed current traversal requires a non-zero loaded proxy "
                "coil to retarget shared VF current signs."
            )
        old_proxy_sign = float(np.sign(loaded_proxy_current_A))
        new_proxy_sign = float(np.sign(proxy_plasma_current_A))
        if new_proxy_sign == 0.0:
            raise ValueError(
                "Stage 2 seed current traversal requires non-zero proxy current "
                "for shared VF current signs."
            )
        control_value_A = float(vf_current_A) * new_proxy_sign / old_proxy_sign
        _assign_scalar_current_value(vf_current_control, control_value_A)
        vf_current_control.unfix_all()
        return

    for vf_coil in vf_coils:
        sign = float(np.sign(vf_coil.current.get_value()))
        if sign == 0.0 and abs(float(vf_current_A)) > 1.0e-12:
            raise ValueError(
                "Stage 2 seed current traversal cannot infer a VF sign from a "
                "zero-current loaded VF coil."
            )
        _assign_fixed_scalar_current(vf_coil.current, float(vf_current_A) * sign)
    if not vf_coils and abs(float(vf_current_A)) > 1.0e-12:
        raise ValueError(
            "Stage 2 seed current traversal requires loaded VF coils when "
            "--vf-current-A is non-zero."
        )


def _build_initialize_coils_kwargs(
    *,
    args,
    finite_current_config: Stage2FiniteCurrentConfig,
    equilibrium_file,
    surface_scale_factor,
    toroidal_flux,
    nphi,
    ntheta,
):
    return {
        "equilibrium_file": equilibrium_file,
        "surface_scale_factor": surface_scale_factor,
        "toroidal_flux": toroidal_flux,
        "nphi": nphi,
        "ntheta": ntheta,
        "proxy_plasma_current_A": finite_current_config.proxy_plasma_current_A,
        "vf_current_A": finite_current_config.vf_current_A,
        "vf_template_path": finite_current_config.vf_template_path,
        "finite_current_mode": finite_current_config.finite_current_mode,
        "flip_banana": bool(getattr(args, "flip_banana", False)),
        "banana_i_fixed_s2": os.environ.get("BANANA_I_FIXED_S2"),
        "finite_build": resolve_finite_build_settings(args),
        "winding_surface_free_mpol": int(
            getattr(args, "winding_surface_free_mpol", 0)
        ),
        "winding_surface_free_ntor": int(
            getattr(args, "winding_surface_free_ntor", 0)
        ),
        "winding_surface_free_r0": bool(
            getattr(args, "winding_surface_free_r0", False)
        ),
        "winding_surface_free_minor": bool(
            getattr(args, "winding_surface_free_minor", False)
        ),
    }


@dataclass
class Stage2AlmAdaptiveSmoothingState:
    distance_smoothing: float
    curvature_smoothing: float
    events: list[dict] = field(default_factory=list)


def _stage2_alm_adaptive_smoothing_update(
    distance_smoothing,
    curvature_smoothing,
    latest_history_entry,
    *,
    distance_smoothing_min,
    curvature_smoothing_min,
):
    previous_distance_smoothing = float(distance_smoothing)
    previous_curvature_smoothing = float(curvature_smoothing)
    adapted = adapt_alm_smoothing_from_history(
        previous_distance_smoothing,
        previous_curvature_smoothing,
        latest_history_entry,
        distance_smoothing_min=distance_smoothing_min,
        curvature_smoothing_min=curvature_smoothing_min,
    )
    next_distance_smoothing = float(adapted["distance_smoothing"])
    next_curvature_smoothing = float(adapted["curvature_smoothing"])
    distance_gap_count = int(adapted["gap_counts"]["distance"])
    curvature_gap_count = int(adapted["gap_counts"]["curvature"])
    smoothing_changed = bool(
        next_distance_smoothing != previous_distance_smoothing
        or next_curvature_smoothing != previous_curvature_smoothing
    )
    event = None
    if smoothing_changed:
        event = {
            "outer_iteration": latest_history_entry["outer_iteration"],
            "distance_gap_count": distance_gap_count,
            "curvature_gap_count": curvature_gap_count,
            "previous_distance_smoothing": previous_distance_smoothing,
            "distance_smoothing": next_distance_smoothing,
            "previous_curvature_smoothing": previous_curvature_smoothing,
            "curvature_smoothing": next_curvature_smoothing,
        }
    return {
        "distance_smoothing": next_distance_smoothing,
        "curvature_smoothing": next_curvature_smoothing,
        "distance_gap_count": distance_gap_count,
        "curvature_gap_count": curvature_gap_count,
        "smoothing_changed": smoothing_changed,
        "event": event,
    }


def _make_stage2_alm_adaptive_smoothing_callback(
    distance_smoothing,
    curvature_smoothing,
    *,
    distance_smoothing_min,
    curvature_smoothing_min,
):
    state = Stage2AlmAdaptiveSmoothingState(
        distance_smoothing=float(distance_smoothing),
        curvature_smoothing=float(curvature_smoothing),
    )

    def history_callback(history, latest_history_entry, multipliers, penalty):
        del history, multipliers, penalty
        update = _stage2_alm_adaptive_smoothing_update(
            state.distance_smoothing,
            state.curvature_smoothing,
            latest_history_entry,
            distance_smoothing_min=distance_smoothing_min,
            curvature_smoothing_min=curvature_smoothing_min,
        )
        state.distance_smoothing = float(update["distance_smoothing"])
        state.curvature_smoothing = float(update["curvature_smoothing"])
        if update["event"] is not None:
            state.events.append(dict(update["event"]))

    return state, history_callback


def _stage2_alm_adaptive_smoothing_results(state):
    return {
        "ALM_EFFECTIVE_DISTANCE_SMOOTHING": float(state.distance_smoothing),
        "ALM_EFFECTIVE_CURVATURE_SMOOTHING": float(state.curvature_smoothing),
        "ALM_ADAPTIVE_SMOOTHING_EVENTS": list(state.events),
    }


def main(parsed_args=None):
    # PRE-INITIALIZATION
    # ---------------------------------------------------------------------------------------
    args = parse_args() if parsed_args is None else parsed_args
    validate_alm_cli_args(args)
    validate_stage2_iota_cli_args(args)
    validate_stage2_edge_iota_cli_args(args)
    validate_winding_surface_shape_cli_args(args)
    validate_stage2_buildability_objective_cli_args(args)
    if parsed_args is not None:
        validate_banana_current_cli_args(args)
        validate_stage2_tf_current_cli_args(args)
        validate_finite_build_cli_args(args)

    # File for the desired boundary magnetic surface:
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)

    # Make Directory for output
    OUT_DIR = os.path.join(args.output_root, f"outputs-{plasma_surf_filename}") + "/"
    os.makedirs(OUT_DIR, exist_ok=True)

    seed_stage2_results = None
    if args.stage2_bs_path:
        _, seed_stage2_results = load_stage2_seed_results(
            args.stage2_bs_path,
            known_tf_current_A=args.tf_current_A,
        )
    finite_current_config = _resolve_stage2_finite_current_config(
        args,
        stage2_results=seed_stage2_results,
    )
    finite_current_mode = finite_current_config.finite_current_mode
    finite_current_profile = get_finite_current_profile(finite_current_mode)
    proxy_plasma_current_A = finite_current_config.proxy_plasma_current_A
    vf_current_A = finite_current_config.vf_current_A
    vf_template_path = finite_current_config.vf_template_path
    boozer_current_convention = finite_current_config.boozer_current_convention
    if bool(args.flip_banana) and finite_current_mode != JHALPERN30_FINITE_CURRENT_MODE:
        raise ValueError("--flip-banana is only supported by jhalpern30_proxy_field.")

    nphi = args.nphi
    ntheta = args.ntheta
    # Create the TF coils in HBT - these will be fixed but create background toroidal field:
    tf_curves = create_equally_spaced_curves(
        20, 1, stellsym=False, R0=0.976, R1=0.4, order=1
    )
    tf_current_A = args.tf_current_A
    tf_currents = [Current(1.0) * tf_current_A for i in range(20)]

    # All the TF degrees of freedom are fixed
    for tf_curve in tf_curves:
        tf_curve.fix_all()
    for tf_current in tf_currents:
        tf_current.fix_all()

    tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

    # INITIALIZATION FOR BANANA COILS
    # ---------------------------------------------------------------------------------------
    # Initialize at inboard midplane (theta_center = 0.5) and mirrored over plane of symmetry
    theta_center = args.theta_center
    phi_center = args.phi_center
    theta_width = args.theta_width
    phi_width = args.phi_width

    num_quadpoints = args.num_quadpoints  # number of quadature points for coils
    order = args.order  # number of Fourier modes for coils

    if args.accept_offspec_major_radius:
        R0 = float(args.major_radius)
        if R0 <= 0.0:
            raise ValueError("--major-radius must be positive.")
    else:
        R0 = validate_major_radius(
            args.major_radius
        )  # major radius (vacuum-vessel contract)
    s = validate_normalized_toroidal_flux(
        args.toroidal_flux,
        field_name="--toroidal-flux",
    )  # VMEC flux-surface label
    target_lcfs_major_radius_m = validate_target_lcfs_major_radius(
        args.target_lcfs_max_major_radius_m
    )
    target_lcfs_minor_radius_m = validate_target_lcfs_minor_radius(
        args.target_lcfs_max_minor_radius_m
    )
    banana_surf_radius = validate_banana_winding_surface_radius(
        args.banana_surf_radius,
        accept_offspec=args.accept_offspec_winding_radius,
    )

    # Scale the plasma family from the LCFS target.  The vessel/winding R0 stays
    # at the hardware contract value and is not reused as a plasma radius.
    lcfs_probe = _load_stage2_vmec_surface(file_loc, 1.0, nphi, ntheta)
    (
        lcfs_clearance_reference,
        surf_coils,
        VV,
    ) = build_hbt_reference_surfaces(
        lcfs_probe.nfp,
        banana_surf_radius,
        winding_surface_free_mpol=args.winding_surface_free_mpol,
        winding_surface_free_ntor=args.winding_surface_free_ntor,
    )
    if args.stage2_plasma_scaling_mode == "working":
        plasma_geometry = load_plasma_geometry_for_working_major_radius(
            R0,
            s,
            file_loc,
            nphi,
            ntheta,
        )
        target_lcfs_major_radius_m = plasma_geometry.lcfs_major_radius_m
    else:
        geometry_preflight = select_plasma_geometry_preflight_candidate(
            lcfs_surface=lcfs_probe,
            requested_s=s,
            target_lcfs_major_radius_m=target_lcfs_major_radius_m,
            target_lcfs_minor_radius_m=target_lcfs_minor_radius_m,
            vessel_surface=VV,
        )
        selected_geometry = geometry_preflight.selected
        if (
            abs(selected_geometry.s_working - s) > 1.0e-12
            or abs(
                selected_geometry.target_lcfs_major_radius_m
                - target_lcfs_major_radius_m
            )
            > 1.0e-12
        ):
            print(
                "Stage 2 geometry preflight selected "
                f"s={selected_geometry.s_working:.6f}, "
                "target_lcfs_major_radius_m="
                f"{selected_geometry.target_lcfs_major_radius_m:.6f} "
                f"from {len(geometry_preflight.candidates)} candidates."
            )
        s = selected_geometry.s_working
        target_lcfs_major_radius_m = selected_geometry.target_lcfs_major_radius_m
        plasma_geometry = _load_plasma_geometry(
            target_lcfs_major_radius_m,
            s,
            file_loc,
            nphi,
            ntheta,
        )
    new_surf = plasma_geometry.working_surface
    if args.stage2_plasma_surface_path:
        # Warm-start coherence: use a converged seed's own saved plasma surface as
        # the SquaredFlux field-fit target instead of the wout-re-derived working
        # surface, so loaded coils start at their true field error. NFP-validated
        # against the device. SCOPE: only the field-fit target moves; the LCFS shell
        # checks below, the iota-runtime diagnostic, and any cold-path proxy geometry
        # keep using the --equilibrium-path-derived plasma_geometry.
        new_surf = load_stage2_plasma_surface(
            args.stage2_plasma_surface_path, expected_nfp=lcfs_probe.nfp
        )
    lcfs_surf = plasma_geometry.lcfs_surface
    banana_surf_nfp = new_surf.nfp
    if (
        args.stage2_plasma_scaling_mode != "working"
        and plasma_geometry.lcfs_minor_radius_m > target_lcfs_minor_radius_m
    ):
        raise ValueError(
            "Scaled LCFS minor radius violates the HBT-EP plasma target "
            f"({plasma_geometry.lcfs_minor_radius_m:.6f} m > "
            f"{target_lcfs_minor_radius_m:.6f} m). Choose a smaller plasma "
            "target or a VMEC surface family whose LCFS fits the shell."
        )
    if banana_surf_nfp != lcfs_probe.nfp:
        raise ValueError("Stage 2 geometry preflight selected inconsistent NFP.")
    plasma_vessel_min_dist = _surface_surface_min_distance(lcfs_surf, VV)

    finite_build_settings = resolve_finite_build_settings(args)
    FINITE_BUILD = finite_build_settings is not None
    new_vf_build_result = VFCoilBuildResult(coils=[], current_control=None)
    winding_surface_free_dof_names = ()
    if args.stage2_bs_path:
        print(f"Loading Stage 2 seed from {args.stage2_bs_path}")
        (
            new_bs,
            new_curves,
            new_banana_curve,
            new_banana_coils,
            new_tf_coils,
            new_proxy_coils,
            new_vf_coils,
        ) = load_stage2_seed_configuration(
            args.stage2_bs_path,
            new_surf,
            len(tf_coils),
            OUT_DIR,
            stage2_results=seed_stage2_results,
            seed_order_upgrade=getattr(args, "seed_order_upgrade", None),
            finite_build=finite_build_settings,
        )
        if args.stage2_seed_current_traversal:
            _retarget_stage2_seed_auxiliary_currents(
                new_proxy_coils,
                new_vf_coils,
                proxy_plasma_current_A=proxy_plasma_current_A,
                vf_current_A=vf_current_A,
                vf_current_mutability=finite_current_profile.vf_current_mutability,
            )
        if (
            finite_current_profile.vf_current_mutability
            == "shared_unfixed_scaled_current"
        ):
            new_vf_build_result = VFCoilBuildResult(
                coils=new_vf_coils,
                current_control=shared_vf_current_control_for_coils(new_vf_coils),
            )
        else:
            new_vf_build_result = VFCoilBuildResult(
                coils=new_vf_coils,
                current_control=None,
            )
        tf_current_A = float(new_tf_coils[0].current.get_value())
        validate_stage2_tf_current_value(
            tf_current_A,
            accepts_offspec_sign=bool(args.accept_offspec_tf_current_sign),
            accepts_offspec_magnitude=bool(args.accept_offspec_tf_current_magnitude),
            field_name="loaded Stage 2 seed TF current",
        )
        # T1.5 warm-seed re-centering: the loaded path skips _initialize_coils,
        # so free the bounded winding-surface size DOFs on the LOADED master
        # surface here. Returns () (no surface mutation) when no size lever is
        # requested, so the default-off loaded path stays byte-identical.
        winding_surface_free_dof_names = free_loaded_winding_surface_size_dofs(
            new_banana_curve, args
        )
    else:
        (
            new_bs,
            new_curves,
            new_banana_curve,
            new_banana_coils,
            new_proxy_coils,
            raw_vf_build_result,
            winding_surface_free_dof_names,
        ) = _initialize_coils(
            new_surf,
            surf_coils,
            tf_coils,
            num_quadpoints,
            order,
            args.banana_init_current_A,
            phi_center,
            theta_center,
            phi_width,
            theta_width,
            OUT_DIR,
            **_build_initialize_coils_kwargs(
                args=args,
                finite_current_config=finite_current_config,
                equilibrium_file=file_loc,
                surface_scale_factor=plasma_geometry.scale_factor,
                toroidal_flux=s,
                nphi=nphi,
                ntheta=ntheta,
            ),
            return_vf_build_result=True,
        )
        new_vf_build_result = coerce_vf_coil_build_result(
            raw_vf_build_result,
            requires_current_control=(
                finite_current_profile.vf_current_mutability
                == "shared_unfixed_scaled_current"
            ),
        )
        new_vf_coils = new_vf_build_result.coils
        new_tf_coils = tf_coils
    order = int(new_banana_curve.order)
    new_surf_coils = surf_coils
    winding_surface_free_mpol = int(getattr(args, "winding_surface_free_mpol", 0))
    winding_surface_free_ntor = int(getattr(args, "winding_surface_free_ntor", 0))
    winding_surface_mpol = int(
        getattr(new_surf_coils, "mpol", max(1, winding_surface_free_mpol))
    )
    winding_surface_ntor = int(
        getattr(new_surf_coils, "ntor", max(0, winding_surface_free_ntor))
    )
    winding_surface_free_dof_names = tuple(winding_surface_free_dof_names)
    # SquaredFlux geometry penalties act on the optimizable banana curves only;
    # TF / proxy / VF curves are fixed field sources and must not enter the
    # clearance or length objectives.
    if FINITE_BUILD:
        # Finite-build banana coils carry ScaledCurrent(banana_net_current, 1/nfil);
        # the first (identity-symmetry) coil exposes the shared NET-current
        # optimizable, so current bounds and net-current metadata are unchanged.
        banana_current_optimizable = new_banana_coils[0].current.current_to_scale
        # Clearance acts on the symmetry-expanded pack CENTERLINES (pack centers),
        # not the individual filaments, so intra-pack filament gaps do not pollute
        # the coil-to-coil / coil-to-surface penalties.
        objective_curves = apply_symmetries_to_curves(
            [new_banana_curve], new_surf_coils.nfp, new_surf_coils.stellsym
        )
    else:
        banana_current_optimizable = new_banana_coils[0].current
        objective_curves = [coil.curve for coil in new_banana_coils]
    initial_banana_current_A = float(banana_current_optimizable.get_value())
    pin_banana_current = FINITE_BUILD and getattr(args, "finitebuild_pin_current", False)
    if pin_banana_current:
        # Fix the shared net-current DOF so the optimizer cannot drift it (vacuum
        # field-fit otherwise leaves banana current under-constrained -> collapses
        # toward zero). Realized current stays at --banana-init-current-A. The
        # banana-current box bound is skipped below since the DOF is no longer free.
        unwrap_current_optimizable(banana_current_optimizable)[0].fix_all()
    vf_current_control = new_vf_build_result.current_control

    # MAIN OPTIMIZATION
    # ---------------------------------------------------------------------------------------
    # Number of iterations to perform:
    MAXITER = args.maxiter
    # boolean for determining whether coil self-intersects
    intersecting = False

    # Weight on the curve lengths in the objective function
    # We'll penalize the coil if it becomes longer than the hardware contract target.
    LENGTH_WEIGHT = args.length_weight
    # Separate, stronger weight for the one-sided BELOW-floor length penalty. The
    # soft LENGTH_WEIGHT (~5e-4) is tuned for the max-length target and is far too
    # weak to hold the 0.95 m floor against the field-fit gradient, so a cold start
    # can ride into the degenerate small-coil basin (observed L ~ 0.58-0.69 m).
    LENGTH_MIN_WEIGHT = args.length_min_weight
    LENGTH_TARGET = validate_coil_length_target(
        args.length_target,
        accept_offspec_coil_length=stage2_length_contract_allows_offspec(args),
        field_name="--length-target",
    )

    # Threshold and weight for the coil-to-coil distance penalty
    required_cc_centerline_m = required_banana_cc_centerline_m()
    if args.cc_threshold < required_cc_centerline_m:
        raise ValueError(
            f"--cc-threshold must be >= {required_cc_centerline_m:.3f} m."
        )
    CC_THRESHOLD = float(args.cc_threshold)
    CC_WEIGHT = args.cc_weight
    SELF_ENVELOPE_MODE = str(getattr(args, "self_envelope_mode", "hinge"))
    SELF_ENVELOPE_WEIGHT = (
        0.0
        if SELF_ENVELOPE_MODE == "off"
        else float(getattr(args, "self_envelope_weight", 1.0))
    )
    SELF_ENVELOPE_FLOOR = resolve_stage2_self_envelope_floor(args)
    SELF_ENVELOPE_SAMPLING_MARGIN = float(
        getattr(args, "self_envelope_sampling_margin", 0.0)
    )
    SELF_ENVELOPE_OBJECTIVE_FLOOR = (
        SELF_ENVELOPE_FLOOR + SELF_ENVELOPE_SAMPLING_MARGIN
        if SELF_ENVELOPE_MODE == "hinge"
        else SELF_ENVELOPE_FLOOR
    )
    SELF_ENVELOPE_REPORT_FLOOR = (
        2.0 * SELF_ENVELOPE_FLOOR
        if SELF_ENVELOPE_MODE == "groc"
        else SELF_ENVELOPE_OBJECTIVE_FLOOR
    )
    CC_OBJECTIVE_MARGIN = float(
        getattr(args, "cc_objective_margin", BANANA_CC_OBJECTIVE_MARGIN_M)
    )
    CC_OBJECTIVE_THRESHOLD = CC_THRESHOLD + CC_OBJECTIVE_MARGIN
    CS_THRESHOLD = COIL_PLASMA_MIN_DIST_M
    # The Type KK coil-to-coil floor is already the ruled frame face-touch
    # centerline clearance. Only coil-plasma needs finite-build normal inflation.
    finite_build_cs_reach_m = (
        finite_build_settings.pack_half_extent_n_m if FINITE_BUILD else 0.0
    )
    cc_clearance_threshold = CC_THRESHOLD
    cs_clearance_threshold = CS_THRESHOLD + finite_build_cs_reach_m
    if CC_OBJECTIVE_MARGIN > 0.0:
        print(
            "Coil-coil objective buffer: "
            f"hard={CC_THRESHOLD:.4f} m, objective={CC_OBJECTIVE_THRESHOLD:.4f} m"
        )

    # Threshold and weight for the coil curvature penalty
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = float(args.curvature_threshold)
    if (
        args.curvature_threshold > MAX_CURVATURE_INV_M
        and not stage2_curvature_contract_allows_offspec(args)
    ):
        raise ValueError(
            f"--curvature-threshold must be <= {MAX_CURVATURE_INV_M:.1f} m^-1."
        )
    # Audit 5a: drive finite-build solves with the frame-aware winding-pack
    # curvature limit instead of the centerline cap, so finite-build runs cannot
    # converge into the post-hoc FINITEBUILD_CURVATURE_OK reject region. Only
    # ever tightens.
    pre_tightening_curvature_threshold = CURVATURE_THRESHOLD
    (
        CURVATURE_THRESHOLD,
        frame_aware_curvature_limit_inv_m,
        frame_aware_curvature_limit_applied,
    ) = stage2_frame_aware_curvature_tightening(
        CURVATURE_THRESHOLD,
        finite_build_settings,
        stage2_frame_aware_curvature_threshold_enabled(args),
    )
    if frame_aware_curvature_limit_applied:
        print(
            "Frame-aware curvature threshold: "
            f"{pre_tightening_curvature_threshold:.4f} -> "
            f"{CURVATURE_THRESHOLD:.4f} m^-1 "
            "(outer-channel edgewise reach)"
        )
    rotation_aware_curvature_cap_applied = False
    rotation_aware_curvature_cap_inv_m = None
    if (
        bool(getattr(args, "stage2_rotation_aware_curvature_cap", False))
        and FINITE_BUILD
    ):
        # T3.2/G3 (NON-PROMOTION-READY): relax the in-run curvature threshold from
        # the conservative measured edgewise cap up to the realized rotation-aware
        # cap, scalarized to the TIGHTEST realized point (1/(margin + max reach)).
        # Derived from the seed pack twist alpha(theta); as the coupled optimizer
        # improves alpha the true cap only rises, so this is a conservative
        # construction-time relaxation. The honest FINITEBUILD_CURVATURE_OK gate
        # (alpha=0 corner frame, recorded below) is UNCHANGED, so a design that
        # only builds under a favorable twist stays non-promotable until D-5.
        realized_reach_m = rotation_aware_projected_half_extent_m(
            finite_build_settings,
            new_banana_curve,
            new_banana_coils[0].curve.framedcurve,
        )
        rotation_aware_curvature_cap_inv_m = 1.0 / (
            TYPE_KK_INNER_RADIUS_MARGIN_M
            + float(np.max(realized_reach_m))
        )
        if rotation_aware_curvature_cap_inv_m > CURVATURE_THRESHOLD:
            print(
                "Rotation-aware curvature cap (non-promotion-ready): "
                f"{CURVATURE_THRESHOLD:.4f} -> "
                f"{rotation_aware_curvature_cap_inv_m:.4f} m^-1 "
                "(realized seed twist; FINITEBUILD_CURVATURE_OK unchanged)"
            )
            CURVATURE_THRESHOLD = rotation_aware_curvature_cap_inv_m
            rotation_aware_curvature_cap_applied = True

    # Define the individual terms objective function:
    Jf = SquaredFlux(new_surf, new_bs)  # penalty on B dot n
    Jls = CurveLength(new_banana_curve)  # penalty on curve length
    Jccdist = CurveCurveDistance(
        objective_curves, cc_clearance_threshold
    )  # penalty on ruled Type KK coil-to-coil centerline distance
    Jccdist_objective = CurveCurveDistance(objective_curves, CC_OBJECTIVE_THRESHOLD)
    Jcsdist = CurveSurfaceDistance(objective_curves, lcfs_surf, cs_clearance_threshold)

    # Lp-norm curvature penalty (configurable via --curvature-p-norm)
    Jc = LpCurveCurvature(new_banana_curve, args.curvature_p_norm, CURVATURE_THRESHOLD)
    # The poloidal-extent and ellipse-width buildability frames are oriented
    # about the REALIZED CWS winding torus (not the 0.903 spec constant), exactly
    # as the fold/vessel/hardware-keepout terms below are, so a re-centered or
    # freed winding surface is measured in its true frame. The terms then read
    # this radius LIVE from the curve each evaluation (B1.3); the value passed
    # here is the construction-time fallback and the non-CWS fallback.
    buildability_winding_radii = realized_cws_winding_radii(new_banana_coils)
    buildability_winding_r0 = (
        BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
        if buildability_winding_radii is None
        else buildability_winding_radii[0]
    )
    Jpe = PoloidalExtent(
        new_banana_curve,
        buildability_winding_r0,
        POLOIDAL_EXTENT_HALF_WIDTH_RAD,
    )
    Jw = ProjectedEllipseWidth(
        new_banana_curve,
        buildability_winding_r0,
        banana_surf_radius,
    )
    Jself = CurveSelfIntersect(
        new_banana_curve,
        BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
        neighbor_skip=int(
            BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR * new_banana_curve.order
        ),
    )
    if SELF_ENVELOPE_MODE == "groc":
        Jself_envelope = CurveGlobalRadiusOfCurvature(
            new_banana_curve,
            SELF_ENVELOPE_FLOOR,
        )
    else:
        Jself_envelope = CurveSelfDistance(
            new_banana_curve,
            SELF_ENVELOPE_FLOOR,
            self_window_m=args.self_distance_window,
            sampling_margin_m=SELF_ENVELOPE_SAMPLING_MARGIN,
        )
    FOLD_WEIGHT = float(getattr(args, "fold_weight", 1.0))
    FOLD_GEODESIC_CURVATURE_LIMIT = float(
        getattr(
            args,
            "fold_geodesic_curvature_limit",
            BANANA_FOLD_GEODESIC_CURVATURE_LIMIT_INV_M,
        )
    )
    raw_fold_material_binormal_limit = getattr(
        args,
        "fold_material_binormal_curvature_limit",
        None,
    )
    # Default to the geodesic Type-KK fold limit as a conservative placeholder:
    # geodesic and material-frame binormal curvature coincide only at alpha=0, so a
    # distinct material-frame fold limit needs the pack-rotation buildability ruling
    # (D-4/D-5). Override explicitly via --fold-material-binormal-curvature-limit.
    # Sound as a default because the coupled / relaxed-cap path is default-off and
    # non-promotable: the honest alpha=0-corner FINITEBUILD_CURVATURE_OK still
    # governs promotion.
    FOLD_MATERIAL_BINORMAL_CURVATURE_LIMIT = (
        BANANA_FOLD_GEODESIC_CURVATURE_LIMIT_INV_M
        if raw_fold_material_binormal_limit is None
        else float(raw_fold_material_binormal_limit)
    )
    FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION = float(
        getattr(
            args,
            "fold_geodesic_curvature_margin_fraction",
            BANANA_FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION,
        )
    )
    fold_winding_radii = realized_cws_winding_radii(new_banana_coils)
    fold_winding_r0 = (
        BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
        if fold_winding_radii is None
        else fold_winding_radii[0]
    )
    couple_pack_rotation_to_fold = (
        bool(getattr(args, "stage2_couple_pack_rotation_to_fold", False))
        and FINITE_BUILD
    )
    if couple_pack_rotation_to_fold:
        # T3.2/G2: reuse the SHARED finite-build pack frame (SSOT — one frame for
        # field and fold) so its live pack twist alpha(theta) drives the material
        # frame-binormal fold penalty, not only the magnetic field. Off = a fresh
        # ZeroRotation fold frame (byte-identical to legacy).
        fold_framedcurve = new_banana_coils[0].curve.framedcurve
        FOLD_CURVATURE_MODE = "material_frame_binormal"
        FOLD_CURVATURE_LABEL = "material-frame binormal curvature"
        FOLD_CURVATURE_LIMIT = FOLD_MATERIAL_BINORMAL_CURVATURE_LIMIT
    else:
        fold_framedcurve = FramedCurveSurfaceTangent(
            new_banana_curve, fold_winding_r0, 0.0
        )
        FOLD_CURVATURE_MODE = "surface_geodesic"
        FOLD_CURVATURE_LABEL = "surface geodesic curvature"
        FOLD_CURVATURE_LIMIT = FOLD_GEODESIC_CURVATURE_LIMIT
    FOLD_CURVATURE_OBJECTIVE_THRESHOLD = (
        FOLD_CURVATURE_LIMIT
        * (1.0 - FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION)
    )
    Jfold = CurveSurfaceGeodesicCurvature(
        fold_framedcurve,
        p=2,
        threshold=FOLD_CURVATURE_OBJECTIVE_THRESHOLD,
    )
    PACK_TWIST_STRAIN_WEIGHT = float(
        getattr(args, "stage2_pack_twist_strain_weight", 0.0)
    )
    Jtwist = None
    if PACK_TWIST_STRAIN_WEIGHT > 0.0 and FINITE_BUILD:
        # T3.2/G2 windability regularizer (default off): bound the twist rate of a
        # freed alpha(theta) via the torsional strain epsilon = tau^2 w^2 / 12
        # (Paz-Soldan 2020). width = full pack corner span (2 * corner reach), the
        # worst-case transverse extent. Weight 0 never constructs the term.
        Jtwist = LPTorsionalStrainPenalty(
            new_banana_coils[0].curve.framedcurve,
            width=2.0 * finite_build_settings.pack_reach_m,
            p=2,
        )
    # Default-on at single-stage parity (2026-06-15 keep-out parity decision;
    # formulation audit 5c origin): analytic vessel-envelope keep-out, proven in
    # single-stage. Steers the Type KK swept-envelope corners of every
    # symmetry-expanded banana centerline inside the fixed vessel torus (R0/a
    # from hardware_contracts). An explicit weight 0 never constructs the term,
    # restoring the legacy objective graph for byte-identical reproduction.
    # The U-channel frame is oriented about the REALIZED CWS winding torus
    # (not the 0.903 spec constant) so re-centered lineages are measured in
    # their true frame, mirroring the single-stage resolve_keepout_winding_r0
    # wiring (2026-06-10 winding-frame fix). No silent 0.903 fallback.
    VESSEL_KEEPOUT_WEIGHT = float(args.stage2_vessel_keepout_weight)
    AVAILABLE_ENVELOPE_REWARD_WEIGHT = float(
        args.stage2_available_envelope_reward_weight
    )
    HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT = float(
        args.stage2_hardware_sdf_free_space_reward_weight
    )
    if VESSEL_KEEPOUT_WEIGHT > 0.0 or AVAILABLE_ENVELOPE_REWARD_WEIGHT > 0.0:
        realized_winding_radii = realized_cws_winding_radii(new_banana_coils)
        envelope_winding_r0 = (
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
            if realized_winding_radii is None
            else realized_winding_radii[0]
        )
    if VESSEL_KEEPOUT_WEIGHT > 0.0:
        Jvessel = CurveVesselEnvelopeKeepout(
            objective_curves, winding_r0=envelope_winding_r0
        )
    else:
        Jvessel = None
    if AVAILABLE_ENVELOPE_REWARD_WEIGHT > 0.0:
        Javailable_envelope_reward = CurveVesselAvailableEnvelopeReward(
            objective_curves,
            winding_r0=envelope_winding_r0,
        )
    else:
        Javailable_envelope_reward = None
    # Default-on at single-stage parity (2026-06-15 keep-out parity decision;
    # Stage-2 path-parity, hardware keep-out coverage plan 2026-06-13): the
    # static in-vessel hardware keep-out (shells / sensors / solenoid / REMC /
    # limiter / quartz point cloud), wired exactly as the single-stage path does
    # (load_hardware_keepout + CurveHardwareKeepout over the swept Type KK
    # envelope of the symmetry-expanded banana centerlines). The cloud and the
    # HARDWARE_KEEPOUT_MIN_DISTANCE_M contract threshold are shared with
    # single-stage; the JSON's recommended min distance is reconciled
    # fail-closed against the contract so a run never steers against a mismatched
    # cloud. The U-channel frame is oriented about the REALIZED CWS winding torus
    # (same r0 the vessel term uses, not the 0.903 spec constant). An explicit
    # weight 0 never constructs the term, restoring the legacy objective graph
    # for byte-identical reproduction.
    HARDWARE_KEEPOUT_WEIGHT = float(args.stage2_hardware_keepout_weight)
    hardware_keepout_backend = args.stage2_hardware_keepout_backend
    hardware_keepout_group_labels = []
    hardware_keepout_metadata_fields = {}
    Jhardware_sdf_free_space_reward = None
    hardware_sdf_data = None
    if (
        HARDWARE_KEEPOUT_WEIGHT > 0.0
        or HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT > 0.0
    ):
        hardware_keepout_winding_radii = realized_cws_winding_radii(
            new_banana_coils
        )
        hardware_keepout_winding_r0 = (
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
            if hardware_keepout_winding_radii is None
            else hardware_keepout_winding_radii[0]
        )
        if hardware_keepout_backend == "sdf":
            if not args.stage2_hardware_keepout_sdf_manifest:
                raise ValueError(
                    "--stage2-hardware-keepout-backend=sdf with active "
                    "hardware SDF terms requires "
                    "--stage2-hardware-keepout-sdf-manifest"
                )
            hardware_sdf_data = load_hardware_sdf(
                args.stage2_hardware_keepout_sdf_manifest,
                glb_path=args.stage2_hardware_keepout_glb,
            )
            hardware_keepout_metadata_fields = hardware_sdf_metadata_from_data(
                hardware_sdf_data
            )
            hardware_keepout_group_labels = list(hardware_sdf_data.group_labels)
    if HARDWARE_KEEPOUT_WEIGHT > 0.0:
        if hardware_keepout_backend == "point_cloud":
            if not args.stage2_hardware_keepout_json:
                raise ValueError(
                    "--stage2-hardware-keepout-weight > 0 with "
                    "--stage2-hardware-keepout-backend=point_cloud requires "
                    "--stage2-hardware-keepout-json (the exported "
                    "sensor/mount point cloud)"
                )
            (
                keepout_points,
                keepout_point_weight,
                keepout_min_distance,
                _keepout_provenance,
            ) = load_hardware_keepout(
                args.stage2_hardware_keepout_json,
                glb_path=args.stage2_hardware_keepout_glb,
            )
            hardware_keepout_metadata_fields = hardware_keepout_metadata(
                args.stage2_hardware_keepout_json,
                glb_path=args.stage2_hardware_keepout_glb,
            )
            hardware_keepout_group_labels = hardware_keepout_metadata_fields[
                "HARDWARE_KEEPOUT_GROUPS"
            ]
            if abs(keepout_min_distance - HARDWARE_KEEPOUT_MIN_DISTANCE_M) > 1e-15:
                raise ValueError(
                    f"keep-out JSON recommends min distance {keepout_min_distance} m "
                    f"but the hardware contract pins {HARDWARE_KEEPOUT_MIN_DISTANCE_M} m; "
                    "update banana_opt/hardware_contracts.py and the cloud together"
                )
            Jhardware = CurveHardwareKeepout(
                objective_curves,
                keepout_points,
                HARDWARE_KEEPOUT_MIN_DISTANCE_M,
                keepout_point_weight,
                winding_r0=hardware_keepout_winding_r0,
            )
        elif hardware_keepout_backend == "sdf":
            Jhardware = CurveHardwareSdfKeepout(
                objective_curves,
                hardware_sdf_data,
                winding_r0=hardware_keepout_winding_r0,
            )
        else:
            raise ValueError(
                f"unsupported Stage-2 hardware keep-out backend "
                f"{hardware_keepout_backend!r}"
            )
    else:
        Jhardware = None
    if HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT > 0.0:
        Jhardware_sdf_free_space_reward = CurveHardwareSdfFreeSpaceReward(
            objective_curves,
            hardware_sdf_data,
            winding_r0=hardware_keepout_winding_r0,
        )
    # Opt-in (2026-06-11 formulation audit 8, active half): static resonant
    # reweighting of the flux spectrum at the target-iota rationals (island
    # suppression at the source). The FFT mode mask is computed ONCE here
    # from the resonant iota target; weight 0 (the default) never constructs
    # the term, keeping the legacy objective graph identical. Misconfiguration
    # (no iota target, empty rational window, grid too coarse for the NFP
    # harmonics) raises loudly at setup instead of silently penalising
    # nothing. NOTE: the mask lives in VMEC quadrature angles, not Boozer
    # angles (approximation documented in banana_opt/stage2_resonant_flux.py).
    Jres, RESONANT_FLUX_WEIGHT, resonant_flux_rationals = (
        build_stage2_resonant_flux_term_if_requested(args, new_surf, new_bs)
    )
    if Jres is not None:
        print(
            "Resonant flux reweighting (audit 8): w_res="
            f"{RESONANT_FLUX_WEIGHT:g}, rationals="
            f"{[str(r) for r in resonant_flux_rationals]}, "
            f"resonant FFT bins={int(np.count_nonzero(Jres.mode_mask))}"
        )
    LENGTH_MIN_TARGET = COIL_LENGTH_MIN_FRACTION * LENGTH_TARGET
    Jlsmin = QuadraticPenalty(Jls, LENGTH_MIN_TARGET, "min")
    Jwmin = QuadraticPenalty(Jw, BANANA_WIDTH_MIN_M, "min")
    Jwmax = QuadraticPenalty(Jw, BANANA_WIDTH_MAX_M, "max")
    print(f"Initial coil length: {Jls.J():.2f} [m]")
    print(
        f"Initial coil width: {Jw.J():.4f} [m] "
        f"(min={BANANA_WIDTH_MIN_M:.4f}, max={BANANA_WIDTH_MAX_M:.4f})"
    )
    print(
        f"Initial coil self-distance: {Jself.shortest_self_distance():.4f} [m] "
        f"(min={BANANA_SELF_INTERSECT_MIN_DISTANCE_M:.4f})"
    )
    if SELF_ENVELOPE_MODE == "groc":
        print(
            "Initial coil self-envelope GROC: "
            f"radius={Jself_envelope.shortest_groc():.4f} [m], "
            f"equivalent_distance={Jself_envelope.shortest_self_distance():.4f} [m] "
            f"(radius_min={SELF_ENVELOPE_FLOOR:.4f}, "
            f"distance_min={SELF_ENVELOPE_REPORT_FLOOR:.4f})"
        )
    else:
        print(
            "Initial coil self-envelope distance: "
            f"{Jself_envelope.shortest_self_distance():.4f} [m] "
            f"(mode={SELF_ENVELOPE_MODE}, min={SELF_ENVELOPE_REPORT_FLOOR:.4f}, "
            f"nominal={SELF_ENVELOPE_FLOOR:.4f}, "
            f"sampling_margin={SELF_ENVELOPE_SAMPLING_MARGIN:.4f}, "
            f"arc_window={args.self_distance_window:.4f})"
        )
    print(
        f"Initial fold {FOLD_CURVATURE_LABEL}: "
        f"{Jfold.max_abs_frame_binormal_curvature():.4f} [m^-1] "
        f"(objective={FOLD_CURVATURE_OBJECTIVE_THRESHOLD:.4f}, "
        f"hard={FOLD_CURVATURE_LIMIT:.4f})"
    )
    stage2_iota_runtime = build_stage2_iota_runtime_if_requested(
        args=args,
        equilibrium_file=file_loc,
        bs=new_bs,
        tf_coils=new_tf_coils,
        major_radius=R0,
        toroidal_flux=s,
        proxy_plasma_current_A=proxy_plasma_current_A,
        boozer_current_convention=boozer_current_convention,
    )
    deprecated_iota_alm_hot_loop = False

    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    SQUARED_FLUX_WEIGHT = args.squared_flux_weight
    CONSTRAINT_METHOD = args.constraint_method
    JF = (
        SQUARED_FLUX_WEIGHT * Jf
        + LENGTH_WEIGHT * QuadraticPenalty(Jls, LENGTH_TARGET, "max")
        + LENGTH_MIN_WEIGHT * Jlsmin
        + CC_WEIGHT * Jccdist_objective
        + CC_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * Jc
        + args.stage2_poloidal_weight * Jpe
        + args.stage2_width_weight * (Jwmin + Jwmax)
        + args.stage2_selfint_weight * Jself
        + SELF_ENVELOPE_WEIGHT * Jself_envelope
        + FOLD_WEIGHT * Jfold
    )
    BASE_OBJECTIVE = SQUARED_FLUX_WEIGHT * Jf
    if CC_OBJECTIVE_MARGIN > 0.0:
        BASE_OBJECTIVE = BASE_OBJECTIVE + CC_WEIGHT * Jccdist_objective
    if SELF_ENVELOPE_WEIGHT > 0.0:
        BASE_OBJECTIVE = (
            BASE_OBJECTIVE + SELF_ENVELOPE_WEIGHT * Jself_envelope
        )
    if FOLD_WEIGHT > 0.0:
        BASE_OBJECTIVE = BASE_OBJECTIVE + FOLD_WEIGHT * Jfold
    if Jtwist is not None:
        # T3.2/G2 windability regularizer rides both the penalty total and the
        # ALM base objective (smooth term, no constraint row), mirroring the fold
        # / vessel keep-out wiring. Default off (weight 0 -> Jtwist is None).
        JF = JF + PACK_TWIST_STRAIN_WEIGHT * Jtwist
        BASE_OBJECTIVE = BASE_OBJECTIVE + PACK_TWIST_STRAIN_WEIGHT * Jtwist
    if Jvessel is not None:
        JF = JF + VESSEL_KEEPOUT_WEIGHT * Jvessel
        # ALM path: the keep-out rides the smooth base objective (no new
        # constraint row), mirroring the single-stage weighted-penalty wiring.
        BASE_OBJECTIVE = BASE_OBJECTIVE + VESSEL_KEEPOUT_WEIGHT * Jvessel
    if Javailable_envelope_reward is not None:
        JF = (
            JF
            + AVAILABLE_ENVELOPE_REWARD_WEIGHT * Javailable_envelope_reward
        )
        # Smooth steering reward only: promotion still requires the direct CAD
        # contact oracle plus finite-build and confinement gates.
        BASE_OBJECTIVE = (
            BASE_OBJECTIVE
            + AVAILABLE_ENVELOPE_REWARD_WEIGHT * Javailable_envelope_reward
        )
    if Jhardware_sdf_free_space_reward is not None:
        JF = (
            JF
            + HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
            * Jhardware_sdf_free_space_reward
        )
        if CONSTRAINT_METHOD != "alm":
            BASE_OBJECTIVE = (
                BASE_OBJECTIVE
                + HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
                * Jhardware_sdf_free_space_reward
            )
    if Jhardware is not None:
        JF = JF + HARDWARE_KEEPOUT_WEIGHT * Jhardware
        # Penalty path: the static-hardware keep-out rides the weighted objective.
        # ALM path (2026-06-15 parity): the keep-out is promoted to a zero-slack
        # ALM constraint row instead (mirroring self_intersect and the single-
        # stage `hardware_keepout` row), so it must NOT also ride the smooth base
        # objective here — that would double-count the penalty. Vessel keep-out
        # (above) has no constraint-row analog and stays in the base objective.
        if CONSTRAINT_METHOD != "alm":
            BASE_OBJECTIVE = BASE_OBJECTIVE + HARDWARE_KEEPOUT_WEIGHT * Jhardware
    if Jres is not None:
        JF = JF + RESONANT_FLUX_WEIGHT * Jres
        # ALM path: the resonant reweighting is part of the (smooth, quadratic)
        # flux objective itself, so it rides the base objective; no constraint
        # row, mirroring the vessel keep-out wiring above.
        BASE_OBJECTIVE = BASE_OBJECTIVE + RESONANT_FLUX_WEIGHT * Jres
    if args.alm_taylor_test and CONSTRAINT_METHOD != "alm":
        raise ValueError("--alm-taylor-test requires --constraint-method=alm")

    rng_seed = None
    basin_hop_count = None
    basin_minimization_failures = None
    basin_accepted_hops = None
    basin_rejected_hops = None
    basin_best_objective = None
    basin_accept_test_rejections = None
    basin_accept_test_triggered = None
    basin_nonfinite_rejections = None
    basin_normalized_step_rejections = None
    basin_completed_hops = None
    basin_initial_objective = None
    basin_best_hop_objective = None
    basin_best_hop_index = None
    basin_best_result_source = None
    basin_objective_improvement = None
    alm_settings = None
    alm_taylor_result = None
    if args.basin_hops > 0:
        rng_seed = (
            args.basin_seed
            if args.basin_seed >= 0
            else int.from_bytes(os.urandom(4), "big")
        )
    stage2_seed_spec = Stage2SeedSpec(
        plasma_surf_filename=plasma_surf_filename,
        major_radius=R0,
        toroidal_flux=s,
        length_weight=LENGTH_WEIGHT,
        cc_weight=CC_WEIGHT,
        cc_threshold=CC_THRESHOLD,
        curvature_weight=CURVATURE_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        banana_surf_radius=banana_surf_radius,
        tf_current_A=tf_current_A,
        order=order,
        banana_init_current_A=initial_banana_current_A,
        banana_current_max_A=float(args.banana_current_max_A),
        length_target=LENGTH_TARGET,
        finite_current_mode=finite_current_mode,
        proxy_plasma_current_A=proxy_plasma_current_A,
        vf_current_A=vf_current_A,
        vf_template_path=vf_template_path,
        flip_banana=bool(args.flip_banana),
        target_lcfs_max_major_radius_m=float(args.target_lcfs_max_major_radius_m),
        target_lcfs_max_minor_radius_m=float(args.target_lcfs_max_minor_radius_m),
    )
    OUT_DIR_ITER = (
        OUT_DIR
        + format_local_stage2_run_dir(
            stage2_seed_spec,
            constraint_method=CONSTRAINT_METHOD,
            alm_max_outer_iters=args.alm_max_outer_iters,
            alm_penalty_init=args.alm_penalty_init,
            alm_penalty_scale=args.alm_penalty_scale,
            alm_penalty_max=args.alm_penalty_max,
            alm_max_subproblem_continuations=args.alm_max_subproblem_continuations,
            alm_feas_tol=args.alm_feas_tol,
            alm_stationarity_tol=args.alm_stationarity_tol,
            alm_trust_radius_init=args.alm_trust_radius_init,
            alm_trust_radius_min=args.alm_trust_radius_min,
            alm_trust_radius_shrink=args.alm_trust_radius_shrink,
            alm_trust_radius_grow=args.alm_trust_radius_grow,
            alm_max_inner_attempts=args.alm_max_inner_attempts,
            alm_distance_smoothing=args.alm_distance_smoothing,
            alm_curvature_smoothing=args.alm_curvature_smoothing,
            alm_fix_signal_mismatch_guard=args.alm_fix_signal_mismatch_guard,
            basin_hops=args.basin_hops,
            basin_stepsize=args.basin_stepsize,
            basin_temperature=args.basin_temperature,
            basin_niter_success=args.basin_niter_success,
            basin_seed=rng_seed,
            stage2_iota_target=args.stage2_iota_target,
            stage2_iota_tolerance=args.stage2_iota_tolerance,
            stage2_iota_vol_target=args.stage2_iota_vol_target,
            stage2_iota_constraint_weight=args.stage2_iota_constraint_weight,
            stage2_iota_num_tf_coils=args.stage2_iota_num_tf_coils,
            stage2_iota_nphi=args.stage2_iota_nphi,
            stage2_iota_ntheta=args.stage2_iota_ntheta,
            stage2_iota_mpol=args.stage2_iota_mpol,
            stage2_iota_ntor=args.stage2_iota_ntor,
        )
        + "/"
    )
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # minimize gets called, optimizes based on degrees of freedom from objective function
    dofs = BASE_OBJECTIVE.x if CONSTRAINT_METHOD == "alm" else JF.x
    # Opt-in winding-size DOF scaling (default OFF -> empty map -> identity).
    winding_dof_scale_map = (
        WINDING_DOF_CORRIDOR_SCALE_MAP
        if getattr(args, "winding_dof_scale", False)
        else {}
    )
    alm_base_bounds = None
    if CONSTRAINT_METHOD == "alm":
        # ALM (unlike L-BFGS-B) does not clip the seed into the base bounds, so a
        # freed winding-surface DOF whose seed sits outside its corridor (e.g.
        # an rc(0,0) seed below the 0.903 lower bound) would invert the first
        # trust-box intersection and raise. Clip the seed into the corridor up
        # front, mirroring the penalty path's L-BFGS-B x0 clipping. No-op when the
        # bounds are unbounded (the default, no freed size DOFs).
        alm_base_bounds = build_lbfgsb_bounds(BASE_OBJECTIVE)
        dofs = np.clip(
            dofs,
            np.array([lo for lo, _ in alm_base_bounds], dtype=float),
            np.array([hi for _, hi in alm_base_bounds], dtype=float),
        )

    def capture_artifact_state(candidate_x):
        return _capture_stage2_artifact_state(
            dofs=candidate_x,
            JF=JF,
            BASE_OBJECTIVE=BASE_OBJECTIVE,
            Jf=Jf,
            new_bs=new_bs,
            new_surf=new_surf,
            Jls=Jls,
            Jccdist=Jccdist,
            Jcsdist=Jcsdist,
            new_banana_curve=new_banana_curve,
            banana_current_optimizable=banana_current_optimizable,
            new_tf_coils=new_tf_coils,
            length_target=LENGTH_TARGET,
            cc_threshold=CC_THRESHOLD,
            curvature_threshold=CURVATURE_THRESHOLD,
            coil_surface_threshold=cs_clearance_threshold,
            plasma_vessel_min_dist=plasma_vessel_min_dist,
            plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
            poloidal_extent_threshold_rad=POLOIDAL_EXTENT_HALF_WIDTH_RAD,
            banana_current_max_A=float(args.banana_current_max_A),
            final_plasma_major_radius_m=plasma_geometry.lcfs_major_radius_m,
            final_plasma_minor_radius_m=plasma_geometry.lcfs_minor_radius_m,
            stage2_iota_runtime=stage2_iota_runtime,
            Jw=Jw,
            Jself=Jself,
            Jself_envelope=Jself_envelope,
            Jfold=Jfold,
            length_min_target=LENGTH_MIN_TARGET,
            width_min_threshold=BANANA_WIDTH_MIN_M,
            width_max_threshold=BANANA_WIDTH_MAX_M,
            self_intersect_min_distance=BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
            self_envelope_mode=SELF_ENVELOPE_MODE,
            self_envelope_min_distance=SELF_ENVELOPE_REPORT_FLOOR,
            self_envelope_nominal_min_distance=SELF_ENVELOPE_FLOOR,
            self_envelope_sampling_margin=SELF_ENVELOPE_SAMPLING_MARGIN,
            self_distance_window=(
                args.self_distance_window
                if SELF_ENVELOPE_MODE in {"hinge", "off"}
                else None
            ),
            self_envelope_groc_radius_floor=(
                SELF_ENVELOPE_FLOOR if SELF_ENVELOPE_MODE == "groc" else None
            ),
            fold_geodesic_curvature_limit=FOLD_CURVATURE_LIMIT,
            fold_curvature_mode=FOLD_CURVATURE_MODE,
            fold_geodesic_curvature_threshold=(
                FOLD_CURVATURE_OBJECTIVE_THRESHOLD
            ),
            fold_geodesic_curvature_margin_fraction=(
                FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION
            ),
        )

    selected_result_x = None
    best_exact_stage2_pass = None
    best_secondary_stage2_artifact = None
    lbfgsb_bounds = None
    if CONSTRAINT_METHOD != "alm":
        if not pin_banana_current:
            apply_penalty_traversal_forbidden_box_bounds(
                bound_targets={"banana_current": banana_current_optimizable},
                requested_thresholds={"banana_current": args.banana_current_max_A},
                seed_values={"banana_current": initial_banana_current_A},
                validate_seed=bool(args.stage2_bs_path),
                seed_context="Loaded Stage 2 seed",
                allow_offspec_threshold_names=(
                    frozenset({"banana_current"})
                    if args.accept_offspec_banana_current_max
                    else frozenset()
                ),
                preserve_seed_sign_names=frozenset({"banana_current"}),
            )
        if vf_current_control is not None:
            apply_vf_current_upper_bound(
                vf_current_control,
                finite_current_config.vf_current_max_A,
            )
        lbfgsb_bounds = build_lbfgsb_bounds(JF)
    s_hel_objective, s_hel_weight = build_s_hel_objective(args, new_bs, new_surf)
    fun = make_stage2_fun(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        stage2_iota_runtime=stage2_iota_runtime,
        s_hel_objective=s_hel_objective,
        s_hel_weight=s_hel_weight,
    )
    alm_result = None
    if CONSTRAINT_METHOD == "alm":
        alm_settings = build_stage2_alm_settings(args)
        alm_constraint_names = stage2_alm_constraint_names(
            include_coil_surface=Jcsdist is not None,
            include_poloidal_extent=True,
            include_hardware_keepout=Jhardware is not None,
            include_iota_penalty=deprecated_iota_alm_hot_loop,
        )
        alm_smoothing_state, history_callback = (
            _make_stage2_alm_adaptive_smoothing_callback(
                args.alm_distance_smoothing,
                args.alm_curvature_smoothing,
                distance_smoothing_min=(
                    float(args.alm_distance_smoothing) * ALM_SMOOTHING_FLOOR_FRACTION
                ),
                curvature_smoothing_min=(
                    float(args.alm_curvature_smoothing) * ALM_SMOOTHING_FLOOR_FRACTION
                ),
            )
        )

        def evaluate_problem(inner_dofs, multipliers, penalty):
            return _evaluate_stage2_alm_problem(
                inner_dofs,
                BASE_OBJECTIVE,
                new_bs,
                new_surf,
                Jf,
                Jls,
                LENGTH_TARGET,
                Jccdist,
                Jc,
                banana_current_optimizable,
                args.banana_current_max_A,
                alm_smoothing_state.distance_smoothing,
                alm_smoothing_state.curvature_smoothing,
                multipliers,
                penalty,
                stage2_constraint_activity_tolerances,
                smooth_min_distance_signed_constraint,
                smooth_max_curvature_signed_constraint,
                Jcsdist=Jcsdist,
                smooth_min_curve_surface_signed_constraint=smooth_min_curve_surface_signed_constraint,
                Jpoloidal=Jpe,
                poloidal_extent_threshold_rad=POLOIDAL_EXTENT_HALF_WIDTH_RAD,
                poloidal_extent_smoothing=alm_smoothing_state.curvature_smoothing,
                smooth_poloidal_extent_signed_constraint=smooth_max_poloidal_extent_signed_constraint,
                stage2_iota_runtime=(
                    stage2_iota_runtime if deprecated_iota_alm_hot_loop else None
                ),
                s_hel_objective=s_hel_objective,
                s_hel_weight=s_hel_weight,
                Jw=Jw,
                width_min_threshold=BANANA_WIDTH_MIN_M,
                width_max_threshold=BANANA_WIDTH_MAX_M,
                Jself=Jself,
                self_intersect_threshold=0.0,
                Jhardware=Jhardware,
                hardware_keepout_alm_scale=args.stage2_hardware_keepout_alm_scale,
                hardware_keepout_tolerance=args.stage2_hardware_keepout_tolerance,
                Jhardware_sdf_free_space_reward=Jhardware_sdf_free_space_reward,
                hardware_sdf_free_space_reward_weight=(
                    HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
                ),
                length_min_target=LENGTH_MIN_TARGET,
            )

        def stage2_contract_passes(candidate_state):
            if not candidate_state["hardware_status"]["success"]:
                return False
            if deprecated_iota_alm_hot_loop:
                return bool(candidate_state["stage2_iota_feasible"])
            return True

        def should_preserve_secondary_stage2_artifact(candidate_state):
            return (
                deprecated_iota_alm_hot_loop
                and candidate_state["hardware_status"]["success"]
                and candidate_state["stage2_iota_feasible"] is False
            )

        def maybe_record_secondary_stage2_artifact(candidate_state, *, source):
            nonlocal best_secondary_stage2_artifact
            if not should_preserve_secondary_stage2_artifact(candidate_state):
                return
            if (
                best_secondary_stage2_artifact is None
                or candidate_state["field_objective"]
                < best_secondary_stage2_artifact["field_objective"]
            ):
                best_secondary_stage2_artifact = {
                    "x": candidate_state["x"].copy(),
                    "field_objective": candidate_state["field_objective"],
                    "source": source,
                }
                print(
                    "[ALM] preserved secondary hardware-pass/iota-fail candidate "
                    f"source={source}, field_objective={candidate_state['field_objective']:.6e}, "
                    f"coil_length={candidate_state['coil_length']:.6f}"
                )

        def maybe_record_exact_stage2_pass(candidate_x, *, source):
            nonlocal best_exact_stage2_pass
            candidate_state = capture_artifact_state(candidate_x)
            maybe_record_secondary_stage2_artifact(candidate_state, source=source)
            if not stage2_contract_passes(candidate_state):
                return
            if (
                best_exact_stage2_pass is None
                or candidate_state["field_objective"]
                < best_exact_stage2_pass["field_objective"]
            ):
                best_exact_stage2_pass = {
                    "x": candidate_state["x"].copy(),
                    "field_objective": candidate_state["field_objective"],
                    "source": source,
                }
                pass_label = (
                    "hardware+iota-pass"
                    if deprecated_iota_alm_hot_loop
                    else "hardware-pass"
                )
                print(
                    f"[ALM] exact {pass_label} incumbent "
                    f"source={source}, field_objective={candidate_state['field_objective']:.6e}, "
                    f"coil_length={candidate_state['coil_length']:.6f}"
                )

        def outer_state_callback(outer_iteration, multipliers, penalty):
            print(
                f"[ALM] outer_iteration={outer_iteration}, "
                f"multipliers={multipliers.tolist()}, penalty={penalty:.3e}"
            )

        if args.alm_taylor_test:
            alm_taylor_result = run_directional_taylor_test(
                evaluate_problem,
                dofs,
                np.zeros(len(alm_constraint_names), dtype=float),
                alm_settings.penalty_init,
                seed=args.alm_taylor_test_seed,
            )
            _print_taylor_test_summary("ALM Taylor", alm_taylor_result)

    lbfgsb_options = {
        "maxiter": MAXITER,
        "maxcor": args.maxcor,
        "ftol": args.ftol,
        "gtol": args.gtol,
    }

    if args.init_only:
        res_nit = 0
        optimizer_success = True
        termination_message = "init_only"
        print("Skipping Stage 2 optimizer because --init-only was provided.")
    elif CONSTRAINT_METHOD == "alm":
        if args.basin_hops > 0:
            raise ValueError(
                "--basin-hops is not supported with --constraint-method=alm"
            )
        maybe_record_exact_stage2_pass(
            dofs,
            source="loaded_seed" if args.stage2_bs_path else "initial_state",
        )
        res = minimize_alm(
            dofs,
            alm_constraint_names,
            evaluate_problem,
            alm_settings,
            lbfgsb_options,
            accepted_callback=lambda candidate_x: maybe_record_exact_stage2_pass(
                candidate_x,
                source="accepted_iterate",
            ),
            outer_state_callback=outer_state_callback,
            history_callback=history_callback,
            base_bounds=alm_base_bounds,
        )
        alm_result = res
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        selected_result_x = np.asarray(res.x, dtype=float).copy()
        final_candidate_state = capture_artifact_state(selected_result_x)
        if best_exact_stage2_pass is not None and not stage2_contract_passes(
            final_candidate_state
        ):
            selected_result_x = best_exact_stage2_pass["x"].copy()
            optimizer_success = False
            restore_reason = (
                "restored_best_exact_hardware_pass_and_iota"
                if deprecated_iota_alm_hot_loop
                else "restored_best_exact_hardware_pass"
            )
            if termination_message:
                termination_message = f"{termination_message}; {restore_reason}"
            else:
                termination_message = restore_reason
            print(
                "[ALM] restoring best exact Stage 2-pass incumbent "
                f"from {best_exact_stage2_pass['source']}"
            )
        print(res.message)
    elif args.basin_hops > 0:
        # Basin-hopping: perturb DOFs and re-run L-BFGS-B multiple times, keep best
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "jac": True,
            "bounds": lbfgsb_bounds,
            "options": lbfgsb_options,
        }
        basin_niter_success = (
            args.basin_niter_success if args.basin_niter_success > 0 else None
        )
        print(
            f"Basin-hopping with {args.basin_hops} hops, "
            f"stepsize={args.basin_stepsize}, "
            f"T={args.basin_temperature}, "
            f"niter_success={basin_niter_success}, "
            f"seed={rng_seed}"
        )
        res, basin_telemetry = run_basin_hopping(
            fun,
            dofs,
            basin_hops=args.basin_hops,
            basin_stepsize=args.basin_stepsize,
            basin_temperature=args.basin_temperature,
            basin_niter_success=basin_niter_success,
            rng_seed=rng_seed,
            minimizer_kwargs=minimizer_kwargs,
        )
        basin_hop_count = res.nit if hasattr(res, "nit") else None
        basin_minimization_failures = (
            res.minimization_failures if hasattr(res, "minimization_failures") else None
        )
        (
            basin_accepted_hops,
            basin_rejected_hops,
            basin_best_objective,
            basin_accept_test_rejections,
            basin_accept_test_triggered,
        ) = basin_telemetry_values(basin_telemetry)
        basin_nonfinite_rejections = basin_telemetry.get("basin_nonfinite_rejections")
        basin_normalized_step_rejections = basin_telemetry.get(
            "basin_normalized_step_rejections"
        )
        basin_completed_hops = basin_telemetry.get("basin_completed_hops")
        basin_initial_objective = basin_telemetry.get("basin_initial_objective")
        basin_best_hop_objective = basin_telemetry.get("basin_best_hop_objective")
        basin_best_hop_index = basin_telemetry.get("basin_best_hop_index")
        basin_best_result_source = basin_telemetry.get("basin_best_result_source")
        basin_objective_improvement = basin_telemetry.get("basin_objective_improvement")
        if hasattr(res, "lowest_optimization_result") and hasattr(
            res.lowest_optimization_result, "nit"
        ):
            res_nit = res.lowest_optimization_result.nit
        else:
            res_nit = basin_hop_count
        if hasattr(res, "lowest_optimization_result"):
            termination_message = str(
                getattr(
                    res.lowest_optimization_result, "message", "basinhopping_complete"
                )
            )
            optimizer_success = bool(
                getattr(res.lowest_optimization_result, "success", True)
            )
        else:
            termination_message = str(getattr(res, "message", "basinhopping_complete"))
            optimizer_success = True
        print(
            f"Basin-hopping complete. Best fun={res.fun:.6e}, hops={args.basin_hops}, seed={rng_seed}"
        )
    else:
        res = run_scaled_winding_minimize(
            minimize,
            fun,
            dofs,
            scale=build_winding_dof_scale_vector(JF.dof_names, winding_dof_scale_map),
            bounds=lbfgsb_bounds,
            options=lbfgsb_options,
        )
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        print(res.message)

    # Ensure SIMSOPT state matches the measured result (needed after basin-hopping
    # and for init-only baselines that still serialize final metrics). Penalty and
    # basin-hopping paths reach here without an assignment, so the fallback must
    # take the optimizer result, not the pre-optimization seed dofs.
    if selected_result_x is None:
        selected_result_x = np.asarray(
            dofs if args.init_only else res.x, dtype=float
        ).copy()
    final_artifact_state = capture_artifact_state(selected_result_x)

    # POST-OPTIMIZATION PROCESSING AND OUTPUTS
    # ---------------------------------------------------------------------------------------
    if is_self_intersecting(new_banana_curve):
        print("BANANA COIL IS SELF-INTERSECTING!")
        intersecting = True

    if final_artifact_state is None:
        final_coil_length = float(Jls.J())
        final_curve_curve_min_dist = float(Jccdist.shortest_distance())
        final_curve_surface_min_dist = float(Jcsdist.shortest_distance())
        final_max_curvature = float(np.max(new_banana_curve.kappa()))
        final_poloidal_extent_rad = _stage2_poloidal_extent_rad(new_banana_curve)
        final_banana_current_A = float(banana_current_optimizable.get_value())
        final_coil_width = float(Jw.J())
        final_self_intersect_penalty = float(Jself.J())
        final_shortest_self_distance = float(Jself.shortest_self_distance())
        final_self_envelope_penalty = float(Jself_envelope.J())
        final_self_envelope_min_dist = float(Jself_envelope.shortest_self_distance())
        final_fold_penalty = float(Jfold.J())
        final_fold_geodesic_curvature_max = float(
            Jfold.max_abs_frame_binormal_curvature()
        )
        final_fold_ok = bool(
            final_fold_geodesic_curvature_max <= FOLD_CURVATURE_LIMIT
        )
        hardware_status = _evaluate_stage2_hardware_constraints(
            final_coil_length,
            LENGTH_TARGET,
            final_curve_curve_min_dist,
            CC_THRESHOLD,
            final_max_curvature,
            CURVATURE_THRESHOLD,
            curve_surface_min_dist=final_curve_surface_min_dist,
            coil_surface_threshold=cs_clearance_threshold,
            plasma_vessel_min_dist=plasma_vessel_min_dist,
            plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
            poloidal_extent_rad=final_poloidal_extent_rad,
            poloidal_extent_threshold_rad=POLOIDAL_EXTENT_HALF_WIDTH_RAD,
            coil_width=final_coil_width,
            width_min_threshold=BANANA_WIDTH_MIN_M,
            width_max_threshold=BANANA_WIDTH_MAX_M,
            self_intersect_penalty=final_self_intersect_penalty,
            self_intersect_threshold=0.0,
            shortest_self_distance=final_shortest_self_distance,
            self_intersect_min_distance=BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
            fold_geodesic_curvature_max=final_fold_geodesic_curvature_max,
            fold_geodesic_curvature_limit=FOLD_CURVATURE_LIMIT,
            fold_curvature_mode=FOLD_CURVATURE_MODE,
            banana_current_A=final_banana_current_A,
            banana_current_threshold=args.banana_current_max_A,
            tf_current_A=float(new_tf_coils[0].current.get_value()),
            tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
            final_plasma_major_radius_m=plasma_geometry.lcfs_major_radius_m,
            final_plasma_minor_radius_m=plasma_geometry.lcfs_minor_radius_m,
        )
    else:
        final_coil_length = final_artifact_state["coil_length"]
        final_curve_curve_min_dist = final_artifact_state["curve_curve_min_dist"]
        final_curve_surface_min_dist = final_artifact_state["curve_surface_min_dist"]
        final_max_curvature = final_artifact_state["max_curvature"]
        final_poloidal_extent_rad = final_artifact_state["poloidal_extent_rad"]
        final_banana_current_A = final_artifact_state["banana_current_A"]
        final_coil_width = final_artifact_state["coil_width"]
        final_self_intersect_penalty = final_artifact_state["self_intersect_penalty"]
        final_shortest_self_distance = final_artifact_state["shortest_self_distance"]
        final_self_envelope_penalty = final_artifact_state["self_envelope_penalty"]
        final_self_envelope_min_dist = final_artifact_state["self_envelope_min_dist"]
        final_fold_penalty = final_artifact_state["fold_penalty"]
        final_fold_geodesic_curvature_max = final_artifact_state[
            "fold_geodesic_curvature_max"
        ]
        final_fold_ok = final_artifact_state["fold_ok"]
        hardware_status = final_artifact_state["hardware_status"]
    print(
        f"Final coil width: {final_coil_width:.4f} [m] "
        f"(min={BANANA_WIDTH_MIN_M:.4f}, max={BANANA_WIDTH_MAX_M:.4f})"
    )
    print(
        f"Final coil self-distance: {final_shortest_self_distance:.4f} [m] "
        f"(min={BANANA_SELF_INTERSECT_MIN_DISTANCE_M:.4f})"
    )
    if SELF_ENVELOPE_MODE == "groc":
        print(
            "Final coil self-envelope GROC: "
            f"radius={final_artifact_state['self_envelope_groc_radius']:.4f} [m], "
            f"equivalent_distance={final_self_envelope_min_dist:.4f} [m] "
            f"(radius_min={SELF_ENVELOPE_FLOOR:.4f}, "
            f"distance_min={SELF_ENVELOPE_REPORT_FLOOR:.4f})"
        )
    else:
        print(
            "Final coil self-envelope distance: "
            f"{final_self_envelope_min_dist:.4f} [m] "
            f"(mode={SELF_ENVELOPE_MODE}, min={SELF_ENVELOPE_REPORT_FLOOR:.4f}, "
            f"nominal={SELF_ENVELOPE_FLOOR:.4f}, "
            f"sampling_margin={SELF_ENVELOPE_SAMPLING_MARGIN:.4f}, "
            f"arc_window={args.self_distance_window:.4f})"
        )
    print(
        f"Final fold {FOLD_CURVATURE_LABEL}: "
        f"{final_fold_geodesic_curvature_max:.4f} [m^-1] "
        f"(objective={FOLD_CURVATURE_OBJECTIVE_THRESHOLD:.4f}, "
        f"hard={FOLD_CURVATURE_LIMIT:.4f}, ok={final_fold_ok})"
    )
    final_iota_feasible = (
        None
        if final_artifact_state is None
        else final_artifact_state.get("stage2_iota_feasible")
    )
    if final_iota_feasible is None and stage2_iota_runtime is not None:
        final_iota_feasible = evaluate_stage2_iota_state(stage2_iota_runtime).feasible
    if not hardware_status["success"]:
        optimizer_success = False
        constraint_summary = "; ".join(hardware_status["violations"])
        if termination_message:
            termination_message = f"{termination_message}; hardware_constraints_failed"
        else:
            termination_message = "hardware_constraints_failed"
        print("/!\\ /!\\ Stage 2 hardware constraint violation /!\\ /!\\")
        print(constraint_summary)
    if deprecated_iota_alm_hot_loop and not bool(final_iota_feasible):
        optimizer_success = False
        if termination_message:
            termination_message = (
                f"{termination_message}; stage2_iota_constraint_failed"
            )
        else:
            termination_message = "stage2_iota_constraint_failed"
        print("/!\\ /!\\ Stage 2 iota constraint violation /!\\ /!\\")

    curves_to_vtk(new_curves, OUT_DIR_ITER + "curves_opt", close=True)
    new_bs.set_points(new_surf.gamma().reshape((-1, 3)))
    unitn = new_surf.unitnormal()
    pointData = {
        "B_N/B": np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]
        / np.sqrt(np.sum(new_bs.B().reshape(unitn.shape) ** 2, axis=2))[:, :, None]
    }
    new_surf.to_vtk(OUT_DIR_ITER + "surf_opt", extra_data=pointData)
    save_surf = getattr(new_surf, "save", None)
    if callable(save_surf):
        save_surf(OUT_DIR_ITER + "surf_opt.json")
    VV.to_vtk(OUT_DIR_ITER + "VV")

    # Create toroidal cross section plot
    cross_section_plot(
        new_surf_coils,
        new_surf,
        new_banana_curve,
        OUT_DIR_ITER + "CrossSectionPlot",
        lcfs_clearance_reference,
        VV,
    )
    stage2_bs_artifact_path = OUT_DIR_ITER + "biot_savart_opt.json"
    print(
        f"Banana Coil Current / TF Current = {banana_current_optimizable.get_value() / new_tf_coils[0].current.get_value():.3f}\n"
    )
    wout_convention_fields = wout_convention_artifact_fields(
        wout_path=file_loc,
        tf_current_A=tf_current_A,
    )
    vf_template_sha256 = (
        None if vf_template_path in {None, ""} else sha256_file(vf_template_path)
    )
    finite_current_profile = get_finite_current_profile(finite_current_mode)
    is_jhalpern30_mode = finite_current_profile.mode == JHALPERN30_FINITE_CURRENT_MODE
    if is_jhalpern30_mode:
        banana_replay = resolve_jhalpern30_banana_current_replay(
            flip_banana=bool(args.flip_banana),
            banana_i_fixed_s2=os.environ.get("BANANA_I_FIXED_S2"),
        )
        banana_current_sign = banana_replay.banana_current_sign
        banana_current_pinned = banana_replay.banana_current_pinned
        banana_i_fixed_s2_kA = banana_replay.banana_i_fixed_s2_kA
    else:
        banana_current_sign = 1
        banana_current_pinned = False
        banana_i_fixed_s2_kA = None
    final_vf_current_A = _realized_vf_current_A(vf_current_control, vf_current_A)

    stage2_realized_winding_radii = realized_cws_winding_radii(new_banana_coils)
    stage2_winding_surface_reembedded_on_live_surface = (
        bool(getattr(args, "winding_surface_free_r0", False))
        or bool(getattr(args, "winding_surface_free_minor", False))
        or int(getattr(args, "winding_surface_free_mpol", 0)) > 0
        or int(getattr(args, "winding_surface_free_ntor", 0)) > 0
    )

    stage2_results_kwargs = dict(
        args=args,
        plasma_surf_filename=plasma_surf_filename,
        file_loc=file_loc,
        stage2_bs_path=args.stage2_bs_path,
        tf_current_A=tf_current_A,
        tf_current_sum_abs_A=sum(
            abs(coil.current.get_value()) for coil in new_tf_coils
        ),
        wout_convention=wout_convention_fields["WOUT_CONVENTION"],
        wout_off_spec=wout_convention_fields["WOUT_OFF_SPEC"],
        num_tf_coils=len(new_tf_coils),
        num_banana_coils=len(new_banana_coils),
        num_proxy_coils=len(new_proxy_coils),
        num_vf_coils=len(new_vf_coils),
        initial_banana_current_A=initial_banana_current_A,
        banana_current_A=final_banana_current_A,
        banana_to_tf_current_ratio=(
            final_banana_current_A / new_tf_coils[0].current.get_value()
        ),
        finite_current_mode=finite_current_mode,
        boozer_current_convention=boozer_current_convention,
        proxy_plasma_current_A=proxy_plasma_current_A,
        vf_current_A=final_vf_current_A,
        vf_template_path=vf_template_path,
        proxy_placement_mode=finite_current_profile.proxy_placement_policy,
        proxy_vf_current_scalar_policy=(
            finite_current_profile.proxy_vf_current_scalar_policy
        ),
        vf_template_sha256=vf_template_sha256,
        vf_current_sign_policy=finite_current_profile.vf_current_sign_policy,
        vf_current_mutability=finite_current_profile.vf_current_mutability,
        flip_banana=bool(args.flip_banana),
        banana_current_sign=banana_current_sign,
        banana_current_pinned=banana_current_pinned,
        banana_i_fixed_s2_kA=banana_i_fixed_s2_kA,
        iota_target_sign=jhalpern30_iota_target_sign(
            flip_banana=bool(args.flip_banana)
        ),
        g0_policy=finite_current_profile.g0_policy,
        boozer_I=physical_current_to_boozer_I(
            proxy_plasma_current_A,
            convention=boozer_current_convention,
        ),
        total_coils=(
            len(new_tf_coils)
            + len(new_banana_coils)
            + len(new_proxy_coils)
            + len(new_vf_coils)
        ),
        cc_threshold=CC_THRESHOLD,
        cc_weight=CC_WEIGHT,
        cc_objective_threshold=CC_OBJECTIVE_THRESHOLD,
        cc_objective_margin=CC_OBJECTIVE_MARGIN,
        curvature_weight=CURVATURE_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        length_weight=LENGTH_WEIGHT,
        constraint_method=CONSTRAINT_METHOD,
        theta_center=theta_center,
        phi_center=phi_center,
        theta_width=theta_width,
        phi_width=phi_width,
        length_target=LENGTH_TARGET,
        major_radius=R0,
        toroidal_flux=s,
        nfp=banana_surf_nfp,
        banana_surf_radius=banana_surf_radius,
        winding_surface_mpol=winding_surface_mpol,
        winding_surface_ntor=winding_surface_ntor,
        winding_surface_free_mpol=winding_surface_free_mpol,
        winding_surface_free_ntor=winding_surface_free_ntor,
        winding_surface_free_dof_names=winding_surface_free_dof_names,
        realized_winding_radii=stage2_realized_winding_radii,
        winding_surface_reembedded_on_live_surface=(
            stage2_winding_surface_reembedded_on_live_surface
        ),
        order=order,
        max_iterations=MAXITER,
        iterations=res_nit,
        termination_message=termination_message,
        optimizer_success=optimizer_success,
        basin_seed=rng_seed if args.basin_hops > 0 else None,
        basin_iterations=basin_hop_count,
        basin_minimization_failures=basin_minimization_failures,
        basin_accepted_hops=basin_accepted_hops,
        basin_rejected_hops=basin_rejected_hops,
        basin_best_objective=basin_best_objective,
        basin_accept_test_rejections=basin_accept_test_rejections,
        basin_accept_test_triggered=basin_accept_test_triggered,
        basin_nonfinite_rejections=basin_nonfinite_rejections,
        basin_normalized_step_rejections=basin_normalized_step_rejections,
        basin_completed_hops=basin_completed_hops,
        basin_initial_objective=basin_initial_objective,
        basin_best_hop_objective=basin_best_hop_objective,
        basin_best_hop_index=basin_best_hop_index,
        basin_best_result_source=basin_best_result_source,
        basin_objective_improvement=basin_objective_improvement,
        alm_result=alm_result,
        alm_taylor_result=alm_taylor_result,
        final_volume=new_surf.volume(),
        final_plasma_major_radius_m=plasma_geometry.lcfs_major_radius_m,
        final_plasma_minor_radius_m=plasma_geometry.lcfs_minor_radius_m,
        intersecting=intersecting,
        final_max_curvature=final_max_curvature,
        final_coil_length=final_coil_length,
        final_curve_curve_min_dist=final_curve_curve_min_dist,
        final_curve_surface_min_dist=final_curve_surface_min_dist,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        final_poloidal_extent_rad=final_poloidal_extent_rad,
        poloidal_extent_threshold_rad=POLOIDAL_EXTENT_HALF_WIDTH_RAD,
        final_coil_width=final_coil_width,
        width_min_threshold=BANANA_WIDTH_MIN_M,
        width_max_threshold=BANANA_WIDTH_MAX_M,
        final_self_intersect_penalty=final_self_intersect_penalty,
        self_intersect_threshold=0.0,
        final_shortest_self_distance=final_shortest_self_distance,
        self_intersect_min_distance=BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
        final_self_envelope_penalty=final_self_envelope_penalty,
        final_self_envelope_min_dist=final_self_envelope_min_dist,
        self_envelope_mode=SELF_ENVELOPE_MODE,
        self_envelope_min_distance=SELF_ENVELOPE_REPORT_FLOOR,
        self_envelope_nominal_min_distance=SELF_ENVELOPE_FLOOR,
        self_envelope_sampling_margin=SELF_ENVELOPE_SAMPLING_MARGIN,
        self_distance_window=(
            args.self_distance_window
            if SELF_ENVELOPE_MODE in {"hinge", "off"}
            else None
        ),
        self_envelope_groc_radius=final_artifact_state[
            "self_envelope_groc_radius"
        ],
        self_envelope_groc_radius_floor=(
            SELF_ENVELOPE_FLOOR if SELF_ENVELOPE_MODE == "groc" else None
        ),
        final_fold_penalty=final_fold_penalty,
        final_fold_geodesic_curvature_max=final_fold_geodesic_curvature_max,
        fold_geodesic_curvature_limit=FOLD_CURVATURE_LIMIT,
        fold_curvature_mode=FOLD_CURVATURE_MODE,
        fold_geodesic_curvature_threshold=(
            FOLD_CURVATURE_OBJECTIVE_THRESHOLD
        ),
        fold_geodesic_curvature_margin_fraction=(
            FOLD_GEODESIC_CURVATURE_MARGIN_FRACTION
        ),
        fold_ok=final_fold_ok,
        length_min_target=LENGTH_MIN_TARGET,
        hardware_status=hardware_status,
    )
    constraint_metadata = build_stage2_constraint_artifact_metadata(
        args=args,
        tf_current_A=tf_current_A,
        banana_current_max_A=float(args.banana_current_max_A),
        length_target=LENGTH_TARGET,
        target_lcfs_max_major_radius_m=float(args.target_lcfs_max_major_radius_m),
        target_lcfs_max_minor_radius_m=float(args.target_lcfs_max_minor_radius_m),
        cc_threshold=CC_THRESHOLD,
        curvature_threshold=CURVATURE_THRESHOLD,
        banana_surf_radius=banana_surf_radius,
        profile_name=getattr(args, "constraint_profile_label", None),
        override_reason=getattr(args, "constraint_override_reason", None),
    )
    secondary_artifact_metadata = build_stage2_secondary_artifact_metadata()
    if (
        best_secondary_stage2_artifact is not None
        and selected_result_x is not None
        and not np.array_equal(best_secondary_stage2_artifact["x"], selected_result_x)
    ):
        secondary_stage2_bs_path, secondary_stage2_results_path = (
            build_stage2_secondary_artifact_paths(stage2_bs_artifact_path)
        )
        secondary_state = capture_artifact_state(best_secondary_stage2_artifact["x"])
        secondary_results = materialize_stage2_artifact_results(
            args=args,
            stage2_bs_artifact_path=secondary_stage2_bs_path,
            results_kwargs=build_secondary_stage2_results_kwargs(
                stage2_results_kwargs=stage2_results_kwargs,
                secondary_state=secondary_state,
                tf_current_A=new_tf_coils[0].current.get_value(),
                new_banana_curve=new_banana_curve,
                new_surf=new_surf,
                termination_message=termination_message,
            ),
            stage2_iota_runtime=stage2_iota_runtime,
            new_bs=new_bs,
            new_surf=new_surf,
            constraint_metadata=constraint_metadata,
        )
        # Audit 5b: the live geometry still holds the secondary state captured
        # above, so the exact segment minima match the recorded point-cloud
        # CURVE_*_MIN_DIST values of this secondary artifact.
        secondary_results.update(
            _segment_exact_clearance_artifact_fields(objective_curves, lcfs_surf)
        )
        secondary_results.update(
            {
                "STAGE2_VESSEL_KEEPOUT_WEIGHT": VESSEL_KEEPOUT_WEIGHT,
                "STAGE2_AVAILABLE_ENVELOPE_REWARD_WEIGHT": (
                    AVAILABLE_ENVELOPE_REWARD_WEIGHT
                ),
                "STAGE2_AVAILABLE_ENVELOPE_REWARD_ACTIVE": (
                    Javailable_envelope_reward is not None
                ),
                "STAGE2_AVAILABLE_ENVELOPE_REWARD": (
                    None
                    if Javailable_envelope_reward is None
                    else float(Javailable_envelope_reward.J())
                ),
                "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT": (
                    HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
                ),
                "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD_ACTIVE": (
                    Jhardware_sdf_free_space_reward is not None
                ),
                "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD": (
                    None
                    if Jhardware_sdf_free_space_reward is None
                    else float(Jhardware_sdf_free_space_reward.J())
                ),
            }
        )
        write_json(secondary_stage2_results_path, secondary_results)
        secondary_artifact_metadata = build_stage2_secondary_artifact_metadata(
            secondary_stage2_bs_path=secondary_stage2_bs_path,
            secondary_stage2_results_path=secondary_stage2_results_path,
            secondary_source=best_secondary_stage2_artifact["source"],
        )
        # Re-materialize the selected primary state before writing the main artifact,
        # since capture_artifact_state mutates the shared optimizer/objective dofs.
        capture_artifact_state(selected_result_x)

    # Save the results of optimization to a separate file
    results = materialize_stage2_artifact_results(
        args=args,
        stage2_bs_artifact_path=stage2_bs_artifact_path,
        results_kwargs=stage2_results_kwargs,
        stage2_iota_runtime=stage2_iota_runtime,
        new_bs=new_bs,
        new_surf=new_surf,
        constraint_metadata=constraint_metadata,
    )
    if CONSTRAINT_METHOD == "alm":
        results.update(_stage2_alm_adaptive_smoothing_results(alm_smoothing_state))
    results.update(secondary_artifact_metadata)
    results.update(
        {
            "STAGE2_VESSEL_KEEPOUT_WEIGHT": VESSEL_KEEPOUT_WEIGHT,
            "STAGE2_AVAILABLE_ENVELOPE_REWARD_WEIGHT": (
                AVAILABLE_ENVELOPE_REWARD_WEIGHT
            ),
            "STAGE2_AVAILABLE_ENVELOPE_REWARD_ACTIVE": (
                Javailable_envelope_reward is not None
            ),
            "STAGE2_AVAILABLE_ENVELOPE_REWARD": (
                None
                if Javailable_envelope_reward is None
                else float(Javailable_envelope_reward.J())
            ),
            "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT": (
                HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT
            ),
            "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD_ACTIVE": (
                Jhardware_sdf_free_space_reward is not None
            ),
            "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD": (
                None
                if Jhardware_sdf_free_space_reward is None
                else float(Jhardware_sdf_free_space_reward.J())
            ),
        }
    )
    if FINITE_BUILD:
        results.update(
            _finite_build_artifact_metadata(
                finite_build_settings,
                new_banana_curve,
                banana_current_optimizable.get_value(),
                cc_min_dist_m=results.get("CURVE_CURVE_MIN_DIST"),
                cs_min_dist_m=results.get("CURVE_SURFACE_MIN_DIST"),
                cc_nominal_m=required_banana_cc_centerline_m(),
                cs_nominal_m=COIL_PLASMA_MIN_DIST_M,
                self_envelope_min_dist_m=results.get("SELF_ENVELOPE_MIN_DIST_M"),
                self_envelope_nominal_m=results.get("SELF_ENVELOPE_THRESHOLD_M"),
                self_envelope_nominal_contract_m=results.get(
                    "SELF_ENVELOPE_NOMINAL_MIN_DISTANCE_M"
                ),
                self_envelope_sampling_margin_m=results.get(
                    "SELF_ENVELOPE_SAMPLING_MARGIN_M"
                ),
                self_distance_window_m=results.get("SELF_DISTANCE_WINDOW_M"),
                self_envelope_mode=results.get("SELF_ENVELOPE_MODE"),
                self_envelope_groc_radius_m=results.get(
                    "SELF_ENVELOPE_GROC_RADIUS_M"
                ),
                self_envelope_groc_radius_floor_m=results.get(
                    "SELF_ENVELOPE_GROC_RADIUS_FLOOR_M"
                ),
                fold_geodesic_curvature_max_inv_m=results.get(
                    "FOLD_CURVATURE_MAX_INV_M",
                    results.get("FOLD_GEODESIC_CURVATURE_MAX_INV_M"),
                ),
                fold_geodesic_curvature_limit_inv_m=results.get(
                    "FOLD_CURVATURE_LIMIT_INV_M",
                    results.get("FOLD_GEODESIC_CURVATURE_LIMIT_INV_M"),
                ),
                fold_geodesic_curvature_threshold_inv_m=results.get(
                    "FOLD_CURVATURE_OBJECTIVE_THRESHOLD_INV_M",
                    results.get("FOLD_GEODESIC_CURVATURE_OBJECTIVE_THRESHOLD_INV_M"),
                ),
                fold_curvature_mode=results.get(
                    "FOLD_CURVATURE_MODE",
                    "surface_geodesic",
                ),
                fold_penalty=results.get("FOLD_PENALTY"),
                # Shared finite-build pack frame carries the realized twist
                # alpha(theta); the metadata helper reads its rotated_frame() for
                # the rotation-aware curvature measurement (T3.2). None-safe.
                pack_framedcurve=new_banana_coils[0].curve.framedcurve,
            )
        )
    if FINITE_BUILD and stage2_frame_aware_curvature_threshold_enabled(args):
        # Audit 5a provenance: distinguishes the constant pack limit driven into
        # the optimizer from the realized per-point
        # FINITEBUILD_FRAME_AWARE_CURVATURE_LIMIT_INV_M recorded above.
        raw_frame_aware_threshold = getattr(
            args, "finitebuild_frame_aware_curvature_threshold", None
        )
        results.update(
            {
                "FINITEBUILD_FRAME_AWARE_CURVATURE_THRESHOLD_ENABLED": True,
                "FINITEBUILD_FRAME_AWARE_CURVATURE_THRESHOLD_MODE": (
                    "default_on"
                    if raw_frame_aware_threshold is None
                    else "explicit_on"
                ),
                "FINITEBUILD_FRAME_AWARE_CURVATURE_THRESHOLD_OPT_IN": (
                    raw_frame_aware_threshold is True
                ),
                "FINITEBUILD_FRAME_AWARE_CURVATURE_THRESHOLD_PACK_LIMIT_INV_M": (
                    float(frame_aware_curvature_limit_inv_m)
                ),
                "FINITEBUILD_FRAME_AWARE_CURVATURE_THRESHOLD_APPLIED": bool(
                    frame_aware_curvature_limit_applied
                ),
            }
        )
    if FINITE_BUILD:
        # T3.2/G2-G3 provenance (levers default off): records which pack-rotation
        # coupling levers were active, for the A/B that quantifies the twist
        # geometry gain. STAGE2_ROTATION_AWARE_CURVATURE_CAP_APPLIED=True marks a
        # NON-PROMOTION-READY relaxed run (FINITEBUILD_CURVATURE_OK stays honest).
        results.update(
            {
                "STAGE2_COUPLE_PACK_ROTATION_TO_FOLD": couple_pack_rotation_to_fold,
                "STAGE2_PACK_TWIST_STRAIN_WEIGHT": PACK_TWIST_STRAIN_WEIGHT,
                "STAGE2_PACK_TWIST_STRAIN_PENALTY": (
                    None if Jtwist is None else float(Jtwist.J())
                ),
                "STAGE2_ROTATION_AWARE_CURVATURE_CAP_APPLIED": (
                    rotation_aware_curvature_cap_applied
                ),
                "STAGE2_ROTATION_AWARE_CURVATURE_CAP_INV_M": (
                    None
                    if rotation_aware_curvature_cap_inv_m is None
                    else float(rotation_aware_curvature_cap_inv_m)
                ),
            }
        )
    if Jvessel is not None:
        # Audit 5c opt-in provenance, evaluated on the final selected geometry.
        results.update(
            {
                "STAGE2_VESSEL_KEEPOUT_WEIGHT": VESSEL_KEEPOUT_WEIGHT,
                "STAGE2_VESSEL_KEEPOUT_PENALTY": float(Jvessel.J()),
                "STAGE2_VESSEL_KEEPOUT_MIN_CLEARANCE_M": float(
                    Jvessel.shortest_clearance()
                ),
            }
        )
    if Jhardware is not None:
        # Phase 2 path-parity provenance, evaluated on the final selected
        # geometry (mirrors the single-stage hardware keep-out provenance).
        results.update(
            {
                "STAGE2_HARDWARE_KEEPOUT_WEIGHT": HARDWARE_KEEPOUT_WEIGHT,
                "STAGE2_HARDWARE_KEEPOUT_BACKEND": hardware_keepout_backend,
                "STAGE2_HARDWARE_KEEPOUT_PENALTY": float(Jhardware.J()),
                "STAGE2_HARDWARE_KEEPOUT_MIN_DISTANCE_M": float(
                    Jhardware.shortest_distance()
                ),
                "STAGE2_HARDWARE_KEEPOUT_JSON": str(
                    args.stage2_hardware_keepout_json
                ),
                "STAGE2_HARDWARE_KEEPOUT_GLB": str(
                    args.stage2_hardware_keepout_glb
                ),
                "STAGE2_HARDWARE_SDF_MANIFEST": (
                    None
                    if args.stage2_hardware_keepout_sdf_manifest is None
                    else str(args.stage2_hardware_keepout_sdf_manifest)
                ),
            }
        )
        if hardware_keepout_backend == "sdf":
            results.update(
                {
                    "STAGE2_HARDWARE_SDF_MIN_CLEARANCE_M": float(
                        Jhardware.shortest_clearance()
                    )
                }
            )
    results.update(
        hardware_keepout_results_fields(
            hardware_group_labels=(
                hardware_keepout_group_labels
                if (
                    Jhardware is not None
                    or Jhardware_sdf_free_space_reward is not None
                )
                else ()
            ),
            vessel_active=Jvessel is not None,
            metadata=(
                hardware_keepout_metadata_fields
                if (
                    Jhardware is not None
                    or Jhardware_sdf_free_space_reward is not None
                )
                else None
            ),
        )
    )
    if new_proxy_coils:
        results.update(
            build_design_only_results_fields(
                reason=f"finite_current_proxy_line_current: {finite_current_mode}",
            )
        )
    # Audit 5b (always-on, additive keys only): exact segment-based minima of
    # the same curves/surface whose point-cloud minima are recorded above. The
    # live geometry here matches the recorded state in every path: the final
    # capture (or the init/live state when no optimizer ran) set it before the
    # final_* metrics were read.
    results.update(
        _segment_exact_clearance_artifact_fields(objective_curves, lcfs_surf)
    )
    write_json(os.path.join(OUT_DIR_ITER, "results.json"), results)


if __name__ == "__main__":
    main()
