import argparse
import os
import sys
import time
from dataclasses import dataclass
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
from simsopt.field import Current, Coil
from simsopt.geo import (
    curves_to_vtk,
    create_equally_spaced_curves,
    CurveLength,
    CurveCurveDistance,
    CurveCWSFourierCPP,
    LpCurveCurvature,
)
from simsopt.geo.curveobjectives import CurveSurfaceDistance
from simsopt._core.optimizable import load
from simsopt.objectives import SquaredFlux, QuadraticPenalty
import json

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
    resolve_wataru_vf_template_path,
    validate_stage2_iota_args,
    validate_normalized_toroidal_flux,
)
from workflow_runner_common import (
    load_stage2_artifact_results,
)
from banana_opt.artifact_contracts import (
    STAGE2_BS_SHA256_KEY,
    compute_stage2_bs_sha256,
    upgrade_legacy_stage2_artifact_results,
)
from banana_opt.constraint_contract import (
    apply_offspec_engineering_override_reason,
    build_constraint_metadata,
    resolve_constraint_contract_from_wire_names,
)
from banana_opt.coil_order_upgrade import (
    upgrade_loaded_seed_biot_savart_order,
)
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.basin_hopping import run_basin_hopping, telemetry_values as basin_telemetry_values
from banana_opt.stage2_geometry import (
    initialize_coils as _initialize_coils,
    is_self_intersecting,
    load_plasma_geometry as _load_plasma_geometry,
    magnetic_field_plots as _magnetic_field_plots,
    surface_surface_min_distance as _surface_surface_min_distance,
)
from banana_opt.hardware_contracts import (
    ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE_ENV,
    ACCEPT_OFFSPEC_R0_SEED_ENV,
    ACCEPT_OFFSPEC_R0_SEED_HELP,
    BANANA_CURRENT_HARD_LIMIT_A,
    BANANA_WINDING_MINOR_RADIUS_M,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_HARD_LIMIT_M,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    TARGET_LCFS_MAX_MAJOR_RADIUS_M,
    TARGET_LCFS_MAX_MINOR_RADIUS_M,
    TF_CURRENT_HARD_LIMIT_A,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    env_flag,
    validate_major_radius,
    validate_plasma_vessel_clearance,
    validate_tf_current_limit,
)
from banana_opt.hardware_constraint_schema import (
    build_bootability_recovery_payload_fields,
    hardware_constraint_alm_names,
)
from banana_opt.lbfgsb_defaults import DEFAULT_LBFGSB_MAXCOR
from banana_opt.current_contracts import (
    BoozerCurrentConvention,
    apply_penalty_traversal_forbidden_box_bounds,
    DEFAULT_FINITE_CURRENT_MODE,
    FiniteCurrentMode,
    resolve_boozer_current_convention,
    resolve_finite_current_mode,
)
from banana_opt.stage2_single_stage_handoff import (
    partition_loaded_stage2_coils,
    probe_stage2_seed_bootability,
)
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

REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = DATABASE_EQUILIBRIA_DIR if os.path.isdir(DATABASE_EQUILIBRIA_DIR) else os.path.join(EXAMPLE_ROOT, "equilibria")
DEFAULT_STAGE2_IOTA_MODE = "off"
DEFAULT_STAGE2_IOTA_TOLERANCE = 5.0e-3
DEFAULT_STAGE2_IOTA_WEIGHT = 1.0
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
    vf_template_path: str | None
    boozer_current_convention: BoozerCurrentConvention


def stage2_alm_constraint_names(
    *,
    include_coil_surface: bool,
    include_iota_penalty: bool = False,
) -> tuple[str, ...]:
    available_names = {
        "coil_length",
        "coil_coil_spacing",
        "max_curvature",
        "banana_current",
    }
    if include_coil_surface:
        available_names.add("coil_surface_spacing")
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
    allow_offspec_engineering_constraints = bool(
        getattr(args, "allow_offspec_engineering_constraints", False)
    )
    if banana_init_current_A <= 0.0:
        raise ValueError("--banana-init-current-A must be positive.")
    if (
        banana_init_current_A > BANANA_CURRENT_HARD_LIMIT_A
        and not allow_offspec_engineering_constraints
    ):
        raise ValueError(
            f"--banana-init-current-A must be in the interval "
            f"(0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}] unless "
            "--allow-offspec-engineering-constraints is set."
        )
    if banana_current_max_A <= 0.0:
        raise ValueError("--banana-current-max-A must be positive.")
    if (
        banana_current_max_A > BANANA_CURRENT_HARD_LIMIT_A
        and not allow_offspec_engineering_constraints
    ):
        raise ValueError(
            f"--banana-current-max-A must be in the interval "
            f"(0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}] unless "
            "--allow-offspec-engineering-constraints is set."
        )
    if banana_init_current_A > banana_current_max_A:
        raise ValueError(
            "--banana-init-current-A cannot exceed --banana-current-max-A."
        )
    validate_tf_current_limit(args.tf_current_A)


def validate_stage2_iota_cli_args(args) -> None:
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
        stage2_iota_weight=args.stage2_iota_weight,
        constraint_method=args.constraint_method,
    )


def resolve_stage2_iota_constraint_weight(constraint_weight: float) -> float | None:
    return canonical_stage2_iota_constraint_weight(constraint_weight)


def build_lbfgsb_bounds(optimizable):
    return list(
        zip(
            np.asarray(optimizable.lower_bounds, dtype=float),
            np.asarray(optimizable.upper_bounds, dtype=float),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage 2 banana coil optimization against a fixed plasma surface.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=os.environ.get("PLASMA_SURF_FILENAME", "wout_nfp22ginsburg_000_014417_iota15.nc"),
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
        "--allow-offspec-engineering-constraints",
        action="store_true",
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
    parser.add_argument("--nphi", type=int, default=int(os.environ.get("NPHI", "255")))
    parser.add_argument("--ntheta", type=int, default=int(os.environ.get("NTHETA", "64")))
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build and save the initialized configuration without running the optimizer.",
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=float(os.environ.get("BANANA_SURF_RADIUS", str(BANANA_WINDING_MINOR_RADIUS_M))),
        help="Coil surface minor radius (default 0.21 m, concentric with HBT vacuum vessel).",
    )
    parser.add_argument(
        "--tf-current-A",
        type=float,
        default=float(os.environ.get("TF_CURRENT_A", str(TF_CURRENT_HARD_LIMIT_A))),
        help="Per-TF-coil current in physical SI amperes (default 8e4 = 80 kA).",
    )
    parser.add_argument(
        "--banana-init-current-A",
        type=float,
        default=float(os.environ.get("BANANA_INIT_CURRENT_A", "1e4")),
        help="Fresh-initialization banana-coil current in SI amperes.",
    )
    parser.add_argument(
        "--banana-current-max-A",
        type=float,
        default=float(os.environ.get("BANANA_CURRENT_MAX_A", "16000")),
        help="Hard upper bound on the realized banana-coil current in SI amperes.",
    )
    parser.add_argument(
        "--finite-current-mode",
        choices=["wataru_proxy_field"],
        default=os.environ.get("FINITE_CURRENT_MODE"),
        help=(
            "Passive artifact-provenance label. Retained for backward "
            "compatibility; Stage 2 always uses the Wataru proxy-field model "
            "after the root-fix refactor."
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
        help="Physical SI amperes for the Wataru-style proxy plasma-current coil.",
    )
    parser.add_argument(
        "--vf-current-A",
        type=float,
        default=(
            float(os.environ["VF_CURRENT_A"]) if "VF_CURRENT_A" in os.environ else None
        ),
        help="Physical SI amperes for each sign-preserving VF template current.",
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
            f"= {VACUUM_VESSEL_MAJOR_RADIUS_M:.3f} m). "
            "Off-spec values require --accept-offspec-r0-seed."
        ),
    )
    parser.add_argument(
        "--accept-offspec-r0-seed",
        action="store_true",
        default=env_flag(ACCEPT_OFFSPEC_R0_SEED_ENV),
        help=ACCEPT_OFFSPEC_R0_SEED_HELP,
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
        default=int(os.environ.get("ALM_MAX_OUTER_ITERS", "10")),
        help="Maximum number of ALM outer iterations (default 10).",
    )
    parser.add_argument(
        "--alm-penalty-init",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_INIT", "1.0")),
        help="Initial ALM penalty parameter (default 1.0).",
    )
    parser.add_argument(
        "--alm-penalty-scale",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_SCALE", "10.0")),
        help="Multiplicative ALM penalty growth factor (default 10.0).",
    )
    parser.add_argument(
        "--alm-penalty-max",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_MAX", "1e8")),
        help="Maximum ALM penalty parameter before capped termination (default 1e8).",
    )
    parser.add_argument(
        "--alm-feas-tol",
        type=float,
        default=float(os.environ.get("ALM_FEAS_TOL", "1e-6")),
        help="ALM max-violation stopping tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--alm-stationarity-tol",
        type=float,
        default=float(os.environ.get("ALM_STATIONARITY_TOL", "1e-6")),
        help="ALM augmented-gradient stopping tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--alm-trust-radius-init",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_INIT", "0.05")),
        help="Initial relative trust radius for bounded ALM inner solves (0 disables bounds).",
    )
    parser.add_argument(
        "--alm-trust-radius-min",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_MIN", "1e-4")),
        help="Minimum relative trust radius for bounded ALM inner solves.",
    )
    parser.add_argument(
        "--alm-trust-radius-shrink",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_SHRINK", "0.5")),
        help="Multiplicative shrink factor for the ALM inner trust radius.",
    )
    parser.add_argument(
        "--alm-trust-radius-grow",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_GROW", "1.5")),
        help="Multiplicative growth factor for the ALM inner trust radius after good steps.",
    )
    parser.add_argument(
        "--alm-max-inner-attempts",
        type=int,
        default=int(os.environ.get("ALM_MAX_INNER_ATTEMPTS", "4")),
        help="Maximum number of trust-radius retries per ALM outer iteration.",
    )
    parser.add_argument(
        "--alm-max-subproblem-continuations",
        type=int,
        default=int(os.environ.get("ALM_MAX_SUBPROBLEM_CONTINUATIONS", "20")),
        help="Maximum accepted-feasible continuation solves before forcing an ALM return.",
    )
    parser.add_argument(
        "--alm-distance-smoothing",
        type=float,
        default=float(os.environ.get("ALM_DISTANCE_SMOOTHING", "0.005")),
        help="Distance soft-min temperature for Stage 2 ALM spacing constraints.",
    )
    parser.add_argument(
        "--alm-curvature-smoothing",
        type=float,
        # Stage 2 uses 0.25 (broader softmax window over the banana coil's
        # kappa array) vs single-stage's 0.05. The wider window is appropriate
        # here because Stage 2 operates on a single banana coil with fewer
        # quadrature points and less sensitivity to curvature perturbations.
        default=float(os.environ.get("ALM_CURVATURE_SMOOTHING", "0.25")),
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
        "--stage2-iota-mode",
        choices=["off", "report", "soft", "alm"],
        default=os.environ.get("STAGE2_IOTA_MODE", DEFAULT_STAGE2_IOTA_MODE),
        help=(
            "Optional Stage 2 iota mode. 'report' records only a final verification "
            "probe, 'soft' adds a weighted Jiota hot-loop term, and 'alm' adds a hard "
            "Stage 2 ALM iota_penalty constraint."
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
        "--stage2-iota-weight",
        type=float,
        default=float(
            os.environ.get(
                "STAGE2_IOTA_WEIGHT",
                str(DEFAULT_STAGE2_IOTA_WEIGHT),
            )
        ),
        help="Jiota weight used when --stage2-iota-mode=soft.",
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
        default=int(
            os.environ.get("STAGE2_IOTA_NPHI", str(DEFAULT_STAGE2_IOTA_NPHI))
        ),
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
        default=int(
            os.environ.get("STAGE2_IOTA_MPOL", str(DEFAULT_STAGE2_IOTA_MPOL))
        ),
        help="Boozer-surface mpol used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--stage2-iota-ntor",
        type=int,
        default=int(
            os.environ.get("STAGE2_IOTA_NTOR", str(DEFAULT_STAGE2_IOTA_NTOR))
        ),
        help="Boozer-surface ntor used by the Stage 2 Boozer/iota solve.",
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=float(os.environ.get("LENGTH_WEIGHT", "0.0005")),
        help="Curve-length penalty weight.",
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=float(os.environ.get("LENGTH_TARGET", str(COIL_LENGTH_TARGET_M))),
        help="Curve-length target in meters.",
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
    args = parser.parse_args()
    try:
        validate_banana_current_cli_args(args)
        validate_stage2_iota_cli_args(args)
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
        secondary_stage2_bs_path is not None and secondary_stage2_results_path is not None
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
        "hardware_status": secondary_state["hardware_status"],
    }


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
        "STAGE2_IOTA_WEIGHT": float(
            getattr(args, "stage2_iota_weight", DEFAULT_STAGE2_IOTA_WEIGHT)
        ),
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
        "STAGE2_IOTA_HOT_LOOP_ENABLED": stage2_iota_runtime is not None,
        "STAGE2_IOTA_BOOTSTRAP_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_SECONDS": None,
        "STAGE2_IOTA_RUNTIME_CALLS": None,
        "STAGE2_IOTA_INITIAL": None,
        "STAGE2_IOTA_INITIAL_PENALTY": None,
        "STAGE2_IOTA_FINAL": None,
        "STAGE2_IOTA_FINAL_PENALTY": None,
        "STAGE2_IOTA_PENALTY_THRESHOLD": None,
    }
    if stage2_iota_runtime is None:
        return payload

    final_state = evaluate_stage2_iota_state(stage2_iota_runtime)
    final_solve_failed = bool(getattr(final_state, "solve_failed", False))
    final_iota = None if final_solve_failed else final_state.iota
    final_penalty = None if final_solve_failed else final_state.penalty
    payload.update(
        {
            "STAGE2_IOTA_BOOTSTRAP_SECONDS": stage2_iota_runtime.stats.bootstrap_seconds,
            "STAGE2_IOTA_RUNTIME_SECONDS": stage2_iota_runtime.stats.runtime_seconds,
            "STAGE2_IOTA_RUNTIME_CALLS": stage2_iota_runtime.stats.runtime_calls,
            "STAGE2_IOTA_EFFECTIVE_WEIGHT": stage2_iota_runtime.effective_weight,
            "STAGE2_IOTA_INITIAL": stage2_iota_runtime.initial_state.iota,
            "STAGE2_IOTA_INITIAL_PENALTY": stage2_iota_runtime.initial_state.penalty,
            "STAGE2_IOTA_FINAL": final_iota,
            "STAGE2_IOTA_FINAL_PENALTY": final_penalty,
            "STAGE2_IOTA_PENALTY_THRESHOLD": stage2_iota_runtime.penalty_threshold,
        }
    )
    return payload


def build_stage2_iota_report_payload(
    *,
    args,
    stage2_bs_artifact_path,
    stage2_results_payload,
    stage2_iota_runtime=None,
):
    probe_enabled = args.stage2_iota_mode != DEFAULT_STAGE2_IOTA_MODE
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
        "STAGE2_IOTA_MODE": args.stage2_iota_mode,
        "STAGE2_IOTA_TARGET": (
            None
            if args.stage2_iota_target is None
            else float(args.stage2_iota_target)
        ),
        "STAGE2_IOTA_TOLERANCE": (
            None
            if not probe_enabled
            else float(args.stage2_iota_tolerance)
        ),
        "STAGE2_IOTA_PROBE_SECONDS": None,
        "BOOTABILITY_STAGE2_BS_PATH": stage2_bs_artifact_path,
        "BOOTABILITY_STAGE2_RESULTS_PATH": stage2_results_path,
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
        equilibrium_path=args.equilibrium_path,
        num_tf_coils=args.stage2_iota_num_tf_coils,
        nphi=args.stage2_iota_nphi,
        ntheta=args.stage2_iota_ntheta,
        mpol=args.stage2_iota_mpol,
        ntor=args.stage2_iota_ntor,
        vol_target=args.stage2_iota_vol_target,
        iota_target=float(args.stage2_iota_target),
        iota_tolerance=args.stage2_iota_tolerance,
        constraint_weight=constraint_weight,
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
    results[STAGE2_BS_SHA256_KEY] = compute_stage2_bs_sha256(
        stage2_bs_artifact_path
    )
    results.update(constraint_metadata)
    results.update(
        build_stage2_iota_report_payload(
            args=args,
            stage2_bs_artifact_path=stage2_bs_artifact_path,
            stage2_results_payload=results,
            stage2_iota_runtime=stage2_iota_runtime,
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
    major_radius,
    accept_offspec_r0_seed,
    profile_name=None,
    override_reason=None,
):
    """Route clamped/validated Stage 2 solver values through the shared contract."""
    allow_offspec_engineering_constraints = bool(
        getattr(args, "allow_offspec_engineering_constraints", False)
    )
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
    offspec_major_radius_m = None
    if accept_offspec_r0_seed:
        offspec_major_radius_m = float(major_radius)
    contract, _trace = resolve_constraint_contract_from_wire_names(
        cli_overrides=cli_overrides,
        accept_offspec_major_radius=accept_offspec_r0_seed,
        offspec_major_radius_m=offspec_major_radius_m,
        allow_offspec_engineering=allow_offspec_engineering_constraints,
    )
    resolved_override_reason = override_reason
    if resolved_override_reason is None and accept_offspec_r0_seed:
        resolved_override_reason = "accept_offspec_r0_seed"
    resolved_override_reason = apply_offspec_engineering_override_reason(
        resolved_override_reason,
        layer=cli_overrides,
        allow_offspec_engineering=allow_offspec_engineering_constraints,
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
):
    return _evaluate_stage2_hardware_constraints(
        coil_length,
        length_target,
        curve_curve_min_dist,
        cc_threshold,
        max_curvature,
        curvature_threshold,
    )


def _capture_stage2_artifact_state(
    *,
    dofs,
    JF,
    BASE_OBJECTIVE,
    Jf,
    Jls,
    Jccdist,
    Jcsdist,
    new_banana_curve,
    new_banana_coils,
    new_tf_coils,
    length_target,
    cc_threshold,
    curvature_threshold,
    coil_surface_threshold,
    plasma_vessel_min_dist,
    plasma_vessel_threshold,
    banana_current_max_A,
    stage2_iota_runtime=None,
):
    candidate_x = np.asarray(dofs, dtype=float).copy()
    JF.x = candidate_x
    BASE_OBJECTIVE.x = candidate_x
    coil_length = float(Jls.J())
    curve_curve_min_dist = float(Jccdist.shortest_distance())
    curve_surface_min_dist = float(Jcsdist.shortest_distance())
    max_curvature = float(np.max(new_banana_curve.kappa()))
    banana_current_A = float(new_banana_coils[0].current.get_value())
    tf_current_A = float(new_tf_coils[0].current.get_value())
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
        banana_current_A=banana_current_A,
        banana_current_threshold=banana_current_max_A,
        tf_current_A=tf_current_A,
        tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
    )
    iota_state = (
        None
        if stage2_iota_runtime is None
        else evaluate_stage2_iota_state(stage2_iota_runtime)
    )
    return {
        "x": candidate_x,
        "field_objective": float(Jf.J()),
        "coil_length": coil_length,
        "curve_curve_min_dist": curve_curve_min_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "max_curvature": max_curvature,
        "banana_current_A": banana_current_A,
        "tf_current_A": tf_current_A,
        "hardware_status": hardware_status,
        "stage2_iota_value": None if iota_state is None else iota_state.iota,
        "stage2_iota_penalty": None if iota_state is None else iota_state.penalty,
        "stage2_iota_abs_error": None if iota_state is None else iota_state.abs_error,
        "stage2_iota_feasible": None if iota_state is None else iota_state.feasible,
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
    bs = load(seed_bs_path)
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


def _resolve_seeded_numeric_field(cli_value, artifact_value, *, field_name):
    if cli_value is None:
        return 0.0 if artifact_value is None else float(artifact_value)
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
    requested_proxy_plasma_current_A = getattr(args, "proxy_plasma_current_A", None)
    requested_vf_current_A = getattr(args, "vf_current_A", None)
    requested_vf_template_path = getattr(args, "vf_template_path", None)
    finite_current_mode = resolve_finite_current_mode(
        requested_finite_current_mode,
        artifact_mode=(
            None if stage2_results is None else stage2_results.get("FINITE_CURRENT_MODE")
        ),
        artifact_mode_source=(
            None
            if stage2_results is None
            else stage2_results.get("FINITE_CURRENT_MODE_SOURCE")
        ),
    )
    if stage2_results is None:
        # Fresh Stage 2: auto-resolve the bundled VF template so the zero-current
        # VF bundle is always serialized. This is the Wataru-faithful shape.
        proxy_plasma_current_A = (
            0.0
            if requested_proxy_plasma_current_A is None
            else float(requested_proxy_plasma_current_A)
        )
        vf_current_A = (
            0.0 if requested_vf_current_A is None else float(requested_vf_current_A)
        )
        vf_template_path = resolve_wataru_vf_template_path(requested_vf_template_path)
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
        )
        vf_current_A = _resolve_seeded_numeric_field(
            requested_vf_current_A,
            stage2_results.get("VF_CURRENT_A"),
            field_name="--vf-current-A",
        )
        if (
            requested_vf_template_path not in {None, ""}
            and _is_legacy_zero_vf_donor(stage2_results)
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

    if vf_current_A != 0.0 and vf_template_path in {None, ""}:
        raise ValueError(
            "--vf-template-path is required when --vf-current-A is non-zero."
        )
    return Stage2FiniteCurrentConfig(
        finite_current_mode=finite_current_mode,
        proxy_plasma_current_A=proxy_plasma_current_A,
        vf_current_A=vf_current_A,
        vf_template_path=vf_template_path,
        boozer_current_convention=resolve_boozer_current_convention(
            finite_current_mode,
        ),
    )


def _build_initialize_coils_kwargs(
    *,
    finite_current_config: Stage2FiniteCurrentConfig,
    equilibrium_file,
    target_major_radius,
    toroidal_flux,
    nphi,
    ntheta,
):
    return {
        "equilibrium_file": equilibrium_file,
        "target_major_radius": target_major_radius,
        "toroidal_flux": toroidal_flux,
        "nphi": nphi,
        "ntheta": ntheta,
        "proxy_plasma_current_A": finite_current_config.proxy_plasma_current_A,
        "vf_current_A": finite_current_config.vf_current_A,
        "vf_template_path": finite_current_config.vf_template_path,
    }


def main(parsed_args=None):
    # PRE-INITIALIZATION
    # ---------------------------------------------------------------------------------------
    args = parse_args() if parsed_args is None else parsed_args
    validate_alm_cli_args(args)
    validate_stage2_iota_cli_args(args)
    if parsed_args is not None:
        validate_banana_current_cli_args(args)

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
    proxy_plasma_current_A = finite_current_config.proxy_plasma_current_A
    vf_current_A = finite_current_config.vf_current_A
    vf_template_path = finite_current_config.vf_template_path
    boozer_current_convention = finite_current_config.boozer_current_convention

    nphi = args.nphi
    ntheta = args.ntheta
    # Create the TF coils in HBT - these will be fixed but create background toroidal field:
    tf_curves = create_equally_spaced_curves(20, 1, stellsym=False, R0=0.976, R1=0.4, order=1)
    tf_current_A = args.tf_current_A
    tf_currents = [Current(1.0) * tf_current_A for i in range(20)]

    # All the TF degrees of freedom are fixed
    for tf_curve in tf_curves:
        tf_curve.fix_all()
    for tf_current in tf_currents:
        tf_current.fix_all()

    tf_coils = [Coil(curve,current) for curve, current in zip(tf_curves,tf_currents)]


    # INITIALIZATION FOR BANANA COILS
    # ---------------------------------------------------------------------------------------
    # Initialize at inboard midplane (theta_center = 0.5) and mirrored over plane of symmetry
    theta_center = args.theta_center
    phi_center = args.phi_center
    theta_width = args.theta_width
    phi_width = args.phi_width

    num_quadpoints = args.num_quadpoints # number of quadature points for coils
    order = args.order # number of Fourier modes for coils

    accept_offspec_r0_seed = getattr(
        args,
        "accept_offspec_r0_seed",
        env_flag(ACCEPT_OFFSPEC_R0_SEED_ENV),
    )
    R0 = validate_major_radius(
        args.major_radius,
        accept_offspec=accept_offspec_r0_seed,
    ) # major radius (vacuum-vessel contract)
    s = validate_normalized_toroidal_flux(
        args.toroidal_flux,
        field_name="--toroidal-flux",
    ) # VMEC flux-surface label

    # Keep the optimization target on the requested working surface while
    # routing hardware reporting and clearance checks through the true LCFS.
    plasma_geometry = _load_plasma_geometry(R0, s, file_loc, nphi, ntheta)
    new_surf = plasma_geometry.working_surface
    lcfs_surf = plasma_geometry.lcfs_surface
    banana_surf_nfp = new_surf.nfp

    banana_surf_radius = args.banana_surf_radius
    if abs(banana_surf_radius - BANANA_WINDING_MINOR_RADIUS_M) > 1.0e-12:
        raise ValueError(
            "Stage 2 banana winding surface must remain concentric with the vessel at "
            f"minor radius {BANANA_WINDING_MINOR_RADIUS_M:.6f} m."
        )
    (
        lcfs_clearance_reference,
        surf_coils,
        VV,
    ) = build_hbt_reference_surfaces(banana_surf_nfp, banana_surf_radius)
    plasma_vessel_min_dist = _surface_surface_min_distance(lcfs_surf, VV)
    validate_plasma_vessel_clearance(
        plasma_vessel_min_dist,
        accept_offspec=env_flag(ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE_ENV),
    )

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
        tf_current_A = float(new_tf_coils[0].current.get_value())
        validate_tf_current_limit(tf_current_A)
    else:
        (
            new_bs,
            new_curves,
            new_banana_curve,
            new_banana_coils,
            new_proxy_coils,
            new_vf_coils,
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
                finite_current_config=finite_current_config,
                equilibrium_file=file_loc,
                target_major_radius=R0,
                toroidal_flux=s,
                nphi=nphi,
                ntheta=ntheta,
            ),
        )
        new_tf_coils = tf_coils
    order = int(new_banana_curve.order)
    new_surf_coils = surf_coils
    # SquaredFlux geometry penalties act on the optimizable banana curves only;
    # TF / proxy / VF curves are fixed field sources and must not enter the
    # clearance or length objectives.
    objective_curves = [coil.curve for coil in new_banana_coils]
    initial_banana_current_A = float(new_banana_coils[0].current.get_value())

    # MAIN OPTIMIZATION
    # ---------------------------------------------------------------------------------------
    # Number of iterations to perform:
    MAXITER = args.maxiter
    # boolean for determining whether coil self-intersects
    intersecting = False

    # Weight on the curve lengths in the objective function
    # We'll penalize the coil if it becomes longer than the hardware contract target.
    LENGTH_WEIGHT = args.length_weight
    allow_offspec_engineering_constraints = bool(
        args.allow_offspec_engineering_constraints
    )
    requested_length_target = float(args.length_target)
    LENGTH_TARGET = requested_length_target
    if (
        not allow_offspec_engineering_constraints
        and requested_length_target > COIL_LENGTH_HARD_LIMIT_M
    ):
        LENGTH_TARGET = COIL_LENGTH_HARD_LIMIT_M
        print(
            f"WARNING: --length-target {requested_length_target} above hardware ceiling, "
            f"clamped to {COIL_LENGTH_HARD_LIMIT_M}"
        )

    # Threshold and weight for the coil-to-coil distance penalty
    CC_THRESHOLD = max(args.cc_threshold, COIL_COIL_MIN_DIST_M)
    if args.cc_threshold < COIL_COIL_MIN_DIST_M:
        print(
            f"WARNING: --cc-threshold {args.cc_threshold} below hardware floor, "
            f"clamped to {COIL_COIL_MIN_DIST_M}"
        )
    CC_WEIGHT = args.cc_weight
    CS_THRESHOLD = COIL_PLASMA_MIN_DIST_M

    # Threshold and weight for the coil curvature penalty
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = float(args.curvature_threshold)
    if (
        not allow_offspec_engineering_constraints
        and args.curvature_threshold > MAX_CURVATURE_INV_M
    ):
        CURVATURE_THRESHOLD = MAX_CURVATURE_INV_M
        print(
            f"WARNING: --curvature-threshold {args.curvature_threshold} above hardware ceiling, "
            f"clamped to {MAX_CURVATURE_INV_M}"
        )

    # Define the individual terms objective function:
    Jf = SquaredFlux(new_surf, new_bs) # penalty on B dot n
    Jls = CurveLength(new_banana_curve) # penalty on curve length
    Jccdist = CurveCurveDistance(objective_curves, CC_THRESHOLD) #penalty on coil-to-coil distance
    Jcsdist = CurveSurfaceDistance(objective_curves, lcfs_surf, CS_THRESHOLD)

    # Lp-norm curvature penalty (configurable via --curvature-p-norm)
    Jc = LpCurveCurvature(new_banana_curve, args.curvature_p_norm, CURVATURE_THRESHOLD)
    print(f"Initial coil length: {Jls.J():.2f} [m]")
    stage2_iota_runtime = None
    if args.stage2_iota_mode in {"soft", "alm"}:
        stage2_iota_runtime = build_stage2_iota_runtime(
            equilibrium_file=file_loc,
            bs=new_bs,
            tf_coils=new_tf_coils,
            major_radius=R0,
            toroidal_flux=s,
            nphi=args.stage2_iota_nphi,
            ntheta=args.stage2_iota_ntheta,
            mpol=args.stage2_iota_mpol,
            ntor=args.stage2_iota_ntor,
            vol_target=args.stage2_iota_vol_target,
            iota_target=float(args.stage2_iota_target),
            iota_tolerance=args.stage2_iota_tolerance,
            constraint_weight=resolve_stage2_iota_constraint_weight(
                args.stage2_iota_constraint_weight
            ),
            num_tf_coils=args.stage2_iota_num_tf_coils,
            mode=args.stage2_iota_mode,
            weight=args.stage2_iota_weight,
        )
        print(
            "Initialized Stage 2 iota hot loop "
            f"mode={args.stage2_iota_mode}, "
            f"iota={stage2_iota_runtime.initial_state.iota:.4f}, "
            f"Jiota={stage2_iota_runtime.initial_state.penalty:.2e}"
        )

    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    SQUARED_FLUX_WEIGHT = args.squared_flux_weight
    CONSTRAINT_METHOD = args.constraint_method
    JF = SQUARED_FLUX_WEIGHT * Jf \
        + LENGTH_WEIGHT * QuadraticPenalty(Jls, LENGTH_TARGET, "max") \
        + CC_WEIGHT * Jccdist \
        + CC_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * Jc
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
        rng_seed = args.basin_seed if args.basin_seed >= 0 else int.from_bytes(os.urandom(4), 'big')
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
            basin_hops=args.basin_hops,
            basin_stepsize=args.basin_stepsize,
            basin_temperature=args.basin_temperature,
            basin_niter_success=args.basin_niter_success,
            basin_seed=rng_seed,
            stage2_iota_mode=args.stage2_iota_mode,
            stage2_iota_target=args.stage2_iota_target,
            stage2_iota_tolerance=args.stage2_iota_tolerance,
            stage2_iota_weight=args.stage2_iota_weight,
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
            Jls=Jls,
            Jccdist=Jccdist,
            Jcsdist=Jcsdist,
            new_banana_curve=new_banana_curve,
            new_banana_coils=new_banana_coils,
            new_tf_coils=new_tf_coils,
            length_target=LENGTH_TARGET,
            cc_threshold=CC_THRESHOLD,
            curvature_threshold=CURVATURE_THRESHOLD,
            coil_surface_threshold=CS_THRESHOLD,
            plasma_vessel_min_dist=plasma_vessel_min_dist,
            plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
            banana_current_max_A=float(args.banana_current_max_A),
            stage2_iota_runtime=stage2_iota_runtime,
        )

    selected_result_x = None
    best_exact_stage2_pass = None
    best_secondary_stage2_artifact = None
    lbfgsb_bounds = None
    if CONSTRAINT_METHOD != "alm":
        apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": new_banana_coils[0].current},
            requested_thresholds={"banana_current": args.banana_current_max_A},
            seed_values={"banana_current": initial_banana_current_A},
            validate_seed=bool(args.stage2_bs_path),
            seed_context="Loaded Stage 2 seed",
        )
        lbfgsb_bounds = build_lbfgsb_bounds(JF)
    fun = make_stage2_fun(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        stage2_iota_runtime=stage2_iota_runtime,
    )
    alm_result = None
    if CONSTRAINT_METHOD == "alm":
        alm_settings = build_stage2_alm_settings(args)
        alm_constraint_names = stage2_alm_constraint_names(
            include_coil_surface=Jcsdist is not None,
            include_iota_penalty=args.stage2_iota_mode == "alm",
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
                new_banana_coils[0].current,
                args.banana_current_max_A,
                args.alm_distance_smoothing,
                args.alm_curvature_smoothing,
                multipliers,
                penalty,
                stage2_constraint_activity_tolerances,
                smooth_min_distance_signed_constraint,
                smooth_max_curvature_signed_constraint,
                Jcsdist=Jcsdist,
                smooth_min_curve_surface_signed_constraint=smooth_min_curve_surface_signed_constraint,
                stage2_iota_runtime=(
                    stage2_iota_runtime if args.stage2_iota_mode == "alm" else None
                ),
            )

        def stage2_contract_passes(candidate_state):
            if not candidate_state["hardware_status"]["success"]:
                return False
            if args.stage2_iota_mode == "alm":
                return bool(candidate_state["stage2_iota_feasible"])
            return True

        def should_preserve_secondary_stage2_artifact(candidate_state):
            return (
                args.stage2_iota_mode == "alm"
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
                    if args.stage2_iota_mode == "alm"
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
            raise ValueError("--basin-hops is not supported with --constraint-method=alm")
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
        )
        alm_result = res
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        selected_result_x = np.asarray(res.x, dtype=float).copy()
        final_candidate_state = capture_artifact_state(selected_result_x)
        if (
            best_exact_stage2_pass is not None
            and not stage2_contract_passes(final_candidate_state)
        ):
            selected_result_x = best_exact_stage2_pass["x"].copy()
            optimizer_success = False
            restore_reason = (
                "restored_best_exact_hardware_pass_and_iota"
                if args.stage2_iota_mode == "alm"
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
            'method': 'L-BFGS-B',
            'jac': True,
            'bounds': lbfgsb_bounds,
            'options': lbfgsb_options,
        }
        basin_niter_success = args.basin_niter_success if args.basin_niter_success > 0 else None
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
        basin_hop_count = res.nit if hasattr(res, 'nit') else None
        basin_minimization_failures = res.minimization_failures if hasattr(res, 'minimization_failures') else None
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
        if hasattr(res, 'lowest_optimization_result') and hasattr(res.lowest_optimization_result, 'nit'):
            res_nit = res.lowest_optimization_result.nit
        else:
            res_nit = basin_hop_count
        if hasattr(res, 'lowest_optimization_result'):
            termination_message = str(getattr(res.lowest_optimization_result, 'message', 'basinhopping_complete'))
            optimizer_success = bool(getattr(res.lowest_optimization_result, 'success', True))
        else:
            termination_message = str(getattr(res, 'message', 'basinhopping_complete'))
            optimizer_success = True
        print(f"Basin-hopping complete. Best fun={res.fun:.6e}, hops={args.basin_hops}, seed={rng_seed}")
    else:
        res = minimize(
            fun,
            dofs,
            jac=True,
            method='L-BFGS-B',
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
        final_banana_current_A = float(new_banana_coils[0].current.get_value())
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
            banana_current_A=final_banana_current_A,
            banana_current_threshold=args.banana_current_max_A,
            tf_current_A=float(new_tf_coils[0].current.get_value()),
            tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
        )
    else:
        final_coil_length = final_artifact_state["coil_length"]
        final_curve_curve_min_dist = final_artifact_state["curve_curve_min_dist"]
        final_curve_surface_min_dist = final_artifact_state["curve_surface_min_dist"]
        final_max_curvature = final_artifact_state["max_curvature"]
        final_banana_current_A = final_artifact_state["banana_current_A"]
        hardware_status = final_artifact_state["hardware_status"]
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
    if args.stage2_iota_mode == "alm" and not bool(final_iota_feasible):
        optimizer_success = False
        if termination_message:
            termination_message = f"{termination_message}; stage2_iota_constraint_failed"
        else:
            termination_message = "stage2_iota_constraint_failed"
        print("/!\\ /!\\ Stage 2 iota constraint violation /!\\ /!\\")

    curves_to_vtk(new_curves, OUT_DIR_ITER + "curves_opt", close=True)
    new_bs.set_points(new_surf.gamma().reshape((-1, 3)))
    unitn = new_surf.unitnormal()
    pointData = {"B_N/B": np.sum(new_bs.B().reshape(unitn.shape) *
        unitn, axis=2)[:, :, None] / np.sqrt(np.sum(new_bs.B().reshape(unitn.shape)**2, axis=2))[:, :, None]}
    new_surf.to_vtk(OUT_DIR_ITER + "surf_opt", extra_data=pointData)
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
    print(f'Banana Coil Current / TF Current = {new_banana_coils[0].current.get_value() / new_tf_coils[0].current.get_value():.3f}\n')

    stage2_results_kwargs = dict(
        args=args,
        plasma_surf_filename=plasma_surf_filename,
        file_loc=file_loc,
        stage2_bs_path=args.stage2_bs_path,
        tf_current_A=tf_current_A,
        tf_current_sum_abs_A=sum(abs(coil.current.get_value()) for coil in new_tf_coils),
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
        vf_current_A=vf_current_A,
        vf_template_path=vf_template_path,
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
        major_radius=R0,
        accept_offspec_r0_seed=accept_offspec_r0_seed,
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
        with open(secondary_stage2_results_path, "w") as outfile:
            json.dump(secondary_results, outfile, indent=2)
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
    results.update(secondary_artifact_metadata)
    with open(os.path.join(OUT_DIR_ITER, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)


if __name__ == "__main__":
    main()
