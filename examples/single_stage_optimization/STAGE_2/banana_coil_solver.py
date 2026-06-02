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
from simsopt.field import Current, Coil, apply_symmetries_to_curves
from simsopt.geo import (
    curves_to_vtk,
    create_equally_spaced_curves,
    CurveLength,
    CurveCurveDistance,
    CurveCWSFourierCPP,
    LpCurveCurvature,
)
from simsopt.geo.curveobjectives import CurveSurfaceDistance
from simsopt.objectives import SquaredFlux, QuadraticPenalty

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
    upgrade_loaded_seed_biot_savart_order,
)
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.basin_hopping import (
    run_basin_hopping,
    telemetry_values as basin_telemetry_values,
)
from banana_opt.stage2_geometry import (
    FiniteBuildSettings,
    VFCoilBuildResult,
    coerce_vf_coil_build_result,
    initialize_coils as _initialize_coils,
    is_self_intersecting,
    load_plasma_geometry_for_working_major_radius,
    load_vmec_surface as _load_stage2_vmec_surface,
    load_plasma_geometry as _load_plasma_geometry,
    magnetic_field_plots as _magnetic_field_plots,
    select_plasma_geometry_preflight_candidate,
    shared_vf_current_control_for_coils,
    surface_surface_min_distance as _surface_surface_min_distance,
)
from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
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
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    POLOIDAL_EXTENT_HALF_WIDTH_RAD,
    STAGE2_POLOIDAL_WEIGHT_DEFAULT,
    STAGE2_SELF_INTERSECT_WEIGHT_DEFAULT,
    STAGE2_WIDTH_WEIGHT_DEFAULT,
    TF_CURRENT_CW_DEFAULT_A,
    TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    TARGET_LCFS_MAX_MINOR_RADIUS_M,
    TF_CURRENT_HARD_LIMIT_A,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
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
from banana_opt.poloidal_extent import (
    PoloidalExtent,
    max_poloidal_extent_rad,
    smooth_max_poloidal_extent_signed_constraint,
)
from banana_opt.self_intersect import CurveSelfIntersect

REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = (
    DATABASE_EQUILIBRIA_DIR
    if os.path.isdir(DATABASE_EQUILIBRIA_DIR)
    else os.path.join(EXAMPLE_ROOT, "equilibria")
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
        return
    if getattr(args, "stage2_bs_path", None):
        raise ValueError(
            "--finite-build is not supported with --stage2-bs-path in this "
            "version; finite-build optimization starts from a fresh banana coil."
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


def validate_s_hel_objective_cli_args(args) -> None:
    if not getattr(args, "enable_s_hel_objective", False):
        return
    s_hel_objective_weight = float(args.s_hel_objective_weight)
    if not np.isfinite(s_hel_objective_weight) or s_hel_objective_weight <= 0.0:
        raise ValueError("--s-hel-objective-weight must be finite and positive.")


def resolve_stage2_iota_constraint_weight(constraint_weight: float) -> float | None:
    return canonical_stage2_iota_constraint_weight(constraint_weight)


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
        help="Curve-length penalty weight.",
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
    parser.add_argument(
        "--finite-build",
        action="store_true",
        help=(
            "Optimize each banana coil as a multi-filament winding pack (finite "
            "build) instead of a zero-thickness filament. The field (SquaredFlux) "
            "sees the real pack and the pack-rotation profile is optimized. Default "
            "off (thin filament). Not supported with --stage2-bs-path or the "
            "jhalpern30 banana path in this version."
        ),
    )
    parser.add_argument(
        "--finitebuild-numfilaments-n",
        type=int,
        default=2,
        help="Filaments in the normal direction of the banana pack (default 2). "
        "Used only with --finite-build. Defaults are not HBT-calibrated.",
    )
    parser.add_argument(
        "--finitebuild-numfilaments-b",
        type=int,
        default=3,
        help="Filaments in the binormal direction of the banana pack (default 3). "
        "Used only with --finite-build.",
    )
    parser.add_argument(
        "--finitebuild-gapsize-n",
        type=float,
        default=0.02,
        help="Gap between filaments in the normal direction, meters (default 0.02). "
        "Used only with --finite-build.",
    )
    parser.add_argument(
        "--finitebuild-gapsize-b",
        type=float,
        default=0.04,
        help="Gap between filaments in the binormal direction, meters (default 0.04). "
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
        choices=("centroid", "frenet"),
        default="centroid",
        help="Pre-rotation orthonormal frame for the filament pack (default "
        "centroid). Used only with --finite-build.",
    )
    args = parser.parse_args()
    try:
        validate_banana_current_cli_args(args)
        validate_stage2_tf_current_cli_args(args)
        validate_stage2_iota_cli_args(args)
        validate_s_hel_objective_cli_args(args)
        validate_finite_build_cli_args(args)
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


def build_hbt_reference_surfaces(nfp, banana_surf_radius):
    surfaces = build_banana_reference_surfaces(nfp, banana_surf_radius)
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

    final_state = stage2_iota_runtime.last_state
    payload.update(
        {
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
        final_plasma_major_radius_m=final_plasma_major_radius_m,
        final_plasma_minor_radius_m=final_plasma_minor_radius_m,
    )


def _stage2_poloidal_extent_rad(banana_curve):
    return max_poloidal_extent_rad(
        banana_curve,
        BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    )


def _evaluate_stage2_flux_objective_on_own_grid(Jf):
    Jf.recompute_bell()
    Jf.field.clear_cached_properties()
    return float(Jf.J())


def _finite_build_artifact_metadata(
    finite_build,
    banana_curve,
    net_banana_current_A,
    *,
    cc_min_dist_m=None,
    cs_min_dist_m=None,
    cc_nominal_m=None,
    cs_nominal_m=None,
    curvature_margin_m=0.0,
):
    """results.json fields describing the finite-build banana winding pack.

    Buildability diagnostics:
    - Bend feasibility: the conductor bends in the centerline's principal-normal
      plane, which is not aligned with either pack axis, so the conservative binding
      extent is the LARGER half-build. ``FINITEBUILD_INNER_EDGE_RADIUS_M =
      min_curv_radius - max(half_n, half_b)`` is the inner-fiber radius; the pack is
      bend-feasible when it is >= ``curvature_margin_m`` (an optional cable-bend
      floor; 0 = pure geometric limit). ``FINITEBUILD_CURVATURE_OK`` reports this.
    - Envelope clearance: clearance is enforced on centerlines, so the real pack
      envelope clears only if the centerline gap exceeds the nominal floor plus the
      pack corner reach (2x between two packs, 1x to the plasma). Reported as
      ``FINITEBUILD_CC_ENVELOPE_OK`` / ``FINITEBUILD_CS_ENVELOPE_OK`` when the
      centerline min distances and nominal floors are supplied.
    """
    max_curvature = float(np.max(banana_curve.kappa()))
    min_curv_radius_m = float("inf") if max_curvature == 0.0 else 1.0 / max_curvature
    half_n_m = float(finite_build.pack_half_extent_n_m)
    half_b_m = float(finite_build.pack_half_extent_b_m)
    binding_half_build_m = max(half_n_m, half_b_m)
    inner_edge_radius_m = min_curv_radius_m - binding_half_build_m
    pack_reach_m = finite_build.pack_reach_m
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
        "FINITEBUILD_PACK_REACH_M": pack_reach_m,
        "FINITEBUILD_MIN_CURVATURE_RADIUS_M": min_curv_radius_m,
        "FINITEBUILD_INNER_EDGE_RADIUS_M": inner_edge_radius_m,
        "FINITEBUILD_CURVATURE_OK": bool(
            inner_edge_radius_m >= float(curvature_margin_m)
        ),
    }
    if cc_min_dist_m is not None and cc_nominal_m is not None:
        cc_envelope_m = float(cc_min_dist_m) - 2.0 * pack_reach_m
        metadata["FINITEBUILD_CC_ENVELOPE_MIN_DIST_M"] = cc_envelope_m
        metadata["FINITEBUILD_CC_ENVELOPE_OK"] = bool(cc_envelope_m >= float(cc_nominal_m))
    if cs_min_dist_m is not None and cs_nominal_m is not None:
        cs_envelope_m = float(cs_min_dist_m) - pack_reach_m
        metadata["FINITEBUILD_CS_ENVELOPE_MIN_DIST_M"] = cs_envelope_m
        metadata["FINITEBUILD_CS_ENVELOPE_OK"] = bool(cs_envelope_m >= float(cs_nominal_m))
    return metadata


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
    length_min_target=None,
    width_min_threshold=None,
    width_max_threshold=None,
    self_intersect_min_distance=None,
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


def load_stage2_seed_configuration(
    seed_bs_path,
    surf,
    num_tf_coils,
    out_dir,
    *,
    stage2_results,
    seed_order_upgrade=None,
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
    bs.set_points(surf.gamma().reshape((-1, 3)))

    coils = bs.coils
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir + "curves_init", close=True)
    unitn = surf.unitnormal()
    pointData = {"B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]}
    surf.to_vtk(out_dir + "surf_init", extra_data=pointData)

    coil_partitions = partition_loaded_stage2_coils(
        coils,
        stage2_results=stage2_results,
        requested_num_tf_coils=num_tf_coils,
    )
    tf_coils = list(coil_partitions.tf_coils)
    banana_coils = list(coil_partitions.banana_coils)
    proxy_coils = list(coil_partitions.proxy_coils)
    vf_coils = list(coil_partitions.vf_coils)
    banana_curve = banana_coils[0].curve
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
    banana_surf_radius = validate_banana_winding_surface_radius(args.banana_surf_radius)

    # Scale the plasma family from the LCFS target.  The vessel/winding R0 stays
    # at the hardware contract value and is not reused as a plasma radius.
    lcfs_probe = _load_stage2_vmec_surface(file_loc, 1.0, nphi, ntheta)
    (
        lcfs_clearance_reference,
        surf_coils,
        VV,
    ) = build_hbt_reference_surfaces(lcfs_probe.nfp, banana_surf_radius)
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

    new_vf_build_result = VFCoilBuildResult(coils=[], current_control=None)
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
    else:
        (
            new_bs,
            new_curves,
            new_banana_curve,
            new_banana_coils,
            new_proxy_coils,
            raw_vf_build_result,
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
    finite_build_settings = resolve_finite_build_settings(args)
    FINITE_BUILD = finite_build_settings is not None
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
    LENGTH_TARGET = validate_coil_length_target(
        args.length_target,
        accept_offspec_coil_length=stage2_length_contract_allows_offspec(args),
        field_name="--length-target",
    )

    # Threshold and weight for the coil-to-coil distance penalty
    if args.cc_threshold < COIL_COIL_MIN_DIST_M:
        raise ValueError(f"--cc-threshold must be >= {COIL_COIL_MIN_DIST_M:.3f} m.")
    CC_THRESHOLD = float(args.cc_threshold)
    CC_WEIGHT = args.cc_weight
    CS_THRESHOLD = COIL_PLASMA_MIN_DIST_M
    # Finite-build clearance is enforced on the symmetry-expanded CENTERLINES, so
    # inflate the distance-penalty thresholds by the pack corner reach: centerline
    # clearance then implies the real pack ENVELOPE clears the nominal floor
    # (2x reach between two packs, 1x reach to the plasma). Thin mode: reach 0
    # (thresholds unchanged). The recorded CC_THRESHOLD/CS_THRESHOLD (metadata,
    # contract, HW-eval) stay nominal; only the objective penalties are inflated.
    finite_build_pack_reach_m = (
        finite_build_settings.pack_reach_m if FINITE_BUILD else 0.0
    )
    cc_clearance_threshold = CC_THRESHOLD + 2.0 * finite_build_pack_reach_m
    cs_clearance_threshold = CS_THRESHOLD + finite_build_pack_reach_m

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

    # Define the individual terms objective function:
    Jf = SquaredFlux(new_surf, new_bs)  # penalty on B dot n
    Jls = CurveLength(new_banana_curve)  # penalty on curve length
    Jccdist = CurveCurveDistance(
        objective_curves, cc_clearance_threshold
    )  # penalty on coil-to-coil distance (pack-envelope-inflated in finite build)
    Jcsdist = CurveSurfaceDistance(objective_curves, lcfs_surf, cs_clearance_threshold)

    # Lp-norm curvature penalty (configurable via --curvature-p-norm)
    Jc = LpCurveCurvature(new_banana_curve, args.curvature_p_norm, CURVATURE_THRESHOLD)
    Jpe = PoloidalExtent(
        new_banana_curve,
        BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        POLOIDAL_EXTENT_HALF_WIDTH_RAD,
    )
    Jw = ProjectedEllipseWidth(
        new_banana_curve,
        BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        BANANA_WINDING_MINOR_RADIUS_M,
    )
    Jself = CurveSelfIntersect(
        new_banana_curve,
        BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
        neighbor_skip=int(
            BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR * new_banana_curve.order
        ),
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
    stage2_iota_runtime = None
    deprecated_iota_alm_hot_loop = False

    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    SQUARED_FLUX_WEIGHT = args.squared_flux_weight
    CONSTRAINT_METHOD = args.constraint_method
    JF = (
        SQUARED_FLUX_WEIGHT * Jf
        + LENGTH_WEIGHT * (QuadraticPenalty(Jls, LENGTH_TARGET, "max") + Jlsmin)
        + CC_WEIGHT * Jccdist
        + CC_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * Jc
        + args.stage2_poloidal_weight * Jpe
        + args.stage2_width_weight * (Jwmin + Jwmax)
        + args.stage2_selfint_weight * Jself
    )
    BASE_OBJECTIVE = SQUARED_FLUX_WEIGHT * Jf
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
            coil_surface_threshold=CS_THRESHOLD,
            plasma_vessel_min_dist=plasma_vessel_min_dist,
            plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
            poloidal_extent_threshold_rad=POLOIDAL_EXTENT_HALF_WIDTH_RAD,
            banana_current_max_A=float(args.banana_current_max_A),
            final_plasma_major_radius_m=plasma_geometry.lcfs_major_radius_m,
            final_plasma_minor_radius_m=plasma_geometry.lcfs_minor_radius_m,
            stage2_iota_runtime=stage2_iota_runtime,
            Jw=Jw,
            Jself=Jself,
            length_min_target=LENGTH_MIN_TARGET,
            width_min_threshold=BANANA_WIDTH_MIN_M,
            width_max_threshold=BANANA_WIDTH_MAX_M,
            self_intersect_min_distance=BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
        )

    selected_result_x = None
    best_exact_stage2_pass = None
    best_secondary_stage2_artifact = None
    lbfgsb_bounds = None
    if CONSTRAINT_METHOD != "alm":
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
        res = minimize(
            fun,
            dofs,
            jac=True,
            method="L-BFGS-B",
            bounds=lbfgsb_bounds,
            options=lbfgsb_options,
        )
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        print(res.message)

    # Ensure SIMSOPT state matches the best result (needed after basin-hopping)
    final_artifact_state = None
    if not args.init_only:
        if selected_result_x is None:
            selected_result_x = np.asarray(res.x, dtype=float).copy()
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
        hardware_status = _evaluate_stage2_hardware_constraints(
            final_coil_length,
            LENGTH_TARGET,
            final_curve_curve_min_dist,
            CC_THRESHOLD,
            final_max_curvature,
            CURVATURE_THRESHOLD,
            curve_surface_min_dist=final_curve_surface_min_dist,
            coil_surface_threshold=CS_THRESHOLD,
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
        hardware_status = final_artifact_state["hardware_status"]
    print(
        f"Final coil width: {final_coil_width:.4f} [m] "
        f"(min={BANANA_WIDTH_MIN_M:.4f}, max={BANANA_WIDTH_MAX_M:.4f})"
    )
    print(
        f"Final coil self-distance: {final_shortest_self_distance:.4f} [m] "
        f"(min={BANANA_SELF_INTERSECT_MIN_DISTANCE_M:.4f})"
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
    if FINITE_BUILD:
        results.update(
            _finite_build_artifact_metadata(
                finite_build_settings,
                new_banana_curve,
                banana_current_optimizable.get_value(),
                cc_min_dist_m=results.get("CURVE_CURVE_MIN_DIST"),
                cs_min_dist_m=results.get("CURVE_SURFACE_MIN_DIST"),
                cc_nominal_m=COIL_COIL_MIN_DIST_M,
                cs_nominal_m=COIL_PLASMA_MIN_DIST_M,
            )
        )
    write_json(os.path.join(OUT_DIR_ITER, "results.json"), results)


if __name__ == "__main__":
    main()
