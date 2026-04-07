import argparse
import hashlib
import os
import io
import json
from dataclasses import astuple, dataclass
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize, basinhopping

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import (
    SurfaceClassifier,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    BoozerSurface,
    curves_to_vtk,
    CurveLength,
    LpCurveCurvature,
)
from simsopt.geo.surfaceobjectives import (
    Volume,
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    SurfaceSurfaceDistance,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import (
    BiotSavart,
    LevelsetStoppingCriterion,
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinZStoppingCriterion,
    compute_fieldlines,
)
from simsopt.objectives import QuadraticPenalty
from simsopt.objectives.utilities import forward_backward
from simsopt._core.optimizable import load
from simsopt._core.derivative import derivative_dec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import sys
sys.path.insert(0, EXAMPLE_ROOT)
from alm_utils import (
    ALMSettings,
    minimize_alm,
    validate_alm_cli_args,
)
from plotting_utils import norm_field_plot, cross_section_plot
from topology_scorer import (
    midplane_seed_radii as _midplane_seed_radii,
    score_topology,
    stop_reason_label as _topology_stop_reason,
    toroidal_angle as _topology_toroidal_angle,
)
from workflow_helpers import (
    Stage2SeedSpec,
    format_database_stage2_seed_dir,
    format_legacy_database_stage2_seed_dir,
    format_legacy_local_stage2_seed_dir,
    format_local_stage2_seed_dir,
    format_local_stage2_seed_dir_without_tf,
)
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.single_stage_geometry import (
    build_scaled_outer_problem,
    build_surface_configs as _build_surface_configs_impl,
    build_surface_search_gate,
    build_surface_search_weights,
    collect_surface_run_metadata,
    disabled_topology_gate_status,
    evaluate_single_stage_hardware_constraints as _evaluate_single_stage_hardware_constraints,
    evaluate_single_stage_hardware_snapshot,
    evaluate_surface_stack,
    evaluate_topology_gate as _evaluate_topology_gate_impl,
    restore_surface_states,
    save_surface_artifacts,
    snapshot_surface_states,
    solve_surface_stack_at_dofs,
    compute_single_stage_surface_vessel_min_dist as _compute_single_stage_surface_vessel_min_dist,
    topology_gate_deficit as _topology_gate_deficit,
    topology_gate_rejection_increment,
)
from banana_opt.single_stage_constraints import (
    smooth_max_curvature_signed_constraint as _smooth_max_curvature_signed_constraint,
    smooth_min_curve_curve_signed_constraint as _smooth_min_curve_curve_signed_constraint,
    smooth_min_curve_surface_signed_constraint as _smooth_min_curve_surface_signed_constraint,
    smooth_min_surface_surface_signed_constraint as _smooth_min_surface_surface_signed_constraint,
)
from banana_opt.single_stage_objectives import (
    average_surface_objectives as _average_surface_objectives_impl,
    build_total_objective as _build_total_objective_impl,
    evaluate_base_objective as _evaluate_base_objective_impl,
    evaluate_total_objective as _evaluate_total_objective_impl,
    evaluate_alm_objective as _evaluate_alm_objective_impl,
)
SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = DATABASE_EQUILIBRIA_DIR if os.path.isdir(DATABASE_EQUILIBRIA_DIR) else os.path.join(EXAMPLE_ROOT, "equilibria")
DEFAULT_LOCAL_STAGE2_ROOT = os.path.join(EXAMPLE_ROOT, "STAGE_2")
DEFAULT_DATABASE_STAGE2_ROOT = os.path.join(REPO_ROOT, "DATABASE", "COIL_OPTIMIZATION", "outputs")
DEFAULT_SINGLE_STAGE_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
MU0_OVER_2PI = 2.0e-7
SINGLE_STAGE_ALM_CONSTRAINT_NAMES = (
    "coil_coil_spacing",
    "coil_surface_spacing",
    "max_curvature",
    "surface_vessel_spacing",
)
DEFAULT_STAGE2_SEEDS_BY_PLASMA = {
    "wout_nfp22ginsburg_000_014417_iota15.nc": {
        "major_radius": 0.915,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "cc_threshold": 0.05,
        "curvature_weight": 0.0001,
        "curvature_threshold": 40.0,
        "banana_surf_radius": 0.22,
        "tf_current_A": 1.0e5,
        "order": 2,
    },
    "wout_nfp22ginsburg_000_002084_iota20.nc": {
        "major_radius": 0.975,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "cc_threshold": 0.05,
        "curvature_weight": 0.0001,
        "curvature_threshold": 40.0,
        "banana_surf_radius": 0.22,
        "tf_current_A": 1.0e5,
        "order": 2,
    },
}
def physical_current_to_boozer_I(plasma_current_A):
    return MU0_OVER_2PI * plasma_current_A


def boozer_I_to_physical_current_A(boozer_I):
    return boozer_I / MU0_OVER_2PI


def add_confinement_surrogate_args(parser):
    parser.add_argument(
        "--confinement-objective-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_OBJECTIVE_WEIGHT", "0.0")),
        help="Checkpoint-ranking weight for the tail-sensitive confinement surrogate (0 = disabled).",
    )
    parser.add_argument(
        "--confinement-surrogate-worst-k",
        type=int,
        default=int(os.environ.get("CONFINEMENT_SURROGATE_WORST_K", "3")),
        help="Number of worst field lines emphasized by the confinement surrogate (default 3).",
    )
    parser.add_argument(
        "--confinement-surrogate-early-threshold",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_THRESHOLD", "0.2")),
        help="Normalized exit-time threshold below which lines count as early exits (default 0.2).",
    )
    parser.add_argument(
        "--confinement-surrogate-mean-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_MEAN_WEIGHT", "0.2")),
        help="Weight on mean line loss in the checkpoint confinement surrogate (default 0.2).",
    )
    parser.add_argument(
        "--confinement-surrogate-worst-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_WORST_WEIGHT", "0.6")),
        help="Weight on worst-k line loss in the checkpoint confinement surrogate (default 0.6).",
    )
    parser.add_argument(
        "--confinement-surrogate-early-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_WEIGHT", "0.2")),
        help="Weight on early-exit fraction in the checkpoint confinement surrogate (default 0.2).",
    )


def _resolve_unique_stage2_match(patterns, note):
    from glob import glob as _glob

    matches = []
    for pattern in patterns:
        matches.extend(_glob(pattern))
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        print(f"Note: found {note} at {os.path.dirname(unique_matches[0])}")
        return unique_matches[0]
    if len(unique_matches) > 1:
        match_dirs = "\n".join(f"  - {os.path.dirname(match)}" for match in unique_matches)
        raise FileNotFoundError(
            "Multiple Stage 2 outputs match the requested seed specification. "
            "Pass --stage2-bs-path explicitly to choose one.\n"
            f"Matches:\n{match_dirs}"
        )
    return None


def build_stage2_bs_path(args):
    if args.stage2_bs_path:
        return args.stage2_bs_path

    seed_spec = Stage2SeedSpec(
        plasma_surf_filename=args.plasma_surf_filename,
        major_radius=args.stage2_seed_major_radius,
        toroidal_flux=args.stage2_seed_toroidal_flux,
        length_weight=args.stage2_seed_length_weight,
        cc_weight=args.stage2_seed_cc_weight,
        cc_threshold=args.stage2_seed_cc_threshold,
        curvature_weight=args.stage2_seed_curvature_weight,
        curvature_threshold=args.stage2_seed_curvature_threshold,
        banana_surf_radius=args.stage2_seed_banana_surf_radius,
        tf_current_A=args.stage2_seed_tf_current_A,
        order=args.stage2_seed_order,
    )

    if args.stage2_source == "database":
        seed_dir = format_database_stage2_seed_dir(seed_spec)
        candidate = os.path.join(
            args.database_stage2_root,
            f"outputs-{args.plasma_surf_filename}",
            seed_dir,
            "biot_savart_opt.json",
        )
        if os.path.exists(candidate):
            return candidate

        legacy_dir = format_legacy_database_stage2_seed_dir(seed_spec)
        legacy = os.path.join(
            args.database_stage2_root,
            f"outputs-{args.plasma_surf_filename}",
            legacy_dir,
            "biot_savart_opt.json",
        )
        if os.path.exists(legacy):
            print(f"Note: found legacy Stage 2 database output at {legacy_dir}/ (missing TFC segment)")
            return legacy
        return candidate

    seed_dir = format_local_stage2_seed_dir(seed_spec)
    current_penalty_candidate = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        seed_dir + "-CM=penalty",
        "biot_savart_opt.json",
    )
    if os.path.exists(current_penalty_candidate):
        return current_penalty_candidate

    candidate = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        seed_dir,
        "biot_savart_opt.json",
    )
    if os.path.exists(candidate):
        print(f"Note: found legacy Stage 2 output at {seed_dir}/ (missing constraint-method segment)")
        return candidate

    no_tfc_dir = format_local_stage2_seed_dir_without_tf(seed_spec)
    no_tfc_candidate = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        no_tfc_dir,
        "biot_savart_opt.json",
    )
    if os.path.exists(no_tfc_candidate):
        print(f"Note: found legacy Stage 2 output at {no_tfc_dir}/ (missing TFC segment)")
        return no_tfc_candidate

    # Fallback: legacy directory format without CCT/CT segments
    legacy_dir = format_legacy_local_stage2_seed_dir(seed_spec)
    legacy = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        legacy_dir,
        "biot_savart_opt.json",
    )
    if os.path.exists(legacy):
        print(f"Note: found legacy Stage 2 output at {legacy_dir}/ (missing CCT/CT segments)")
        return legacy

    parent = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
    )
    current_matches = _resolve_unique_stage2_match(
        [
            os.path.join(parent, seed_dir + "-CM=penalty-BH=*", "biot_savart_opt.json"),
            os.path.join(parent, seed_dir + "-CM=alm-*", "biot_savart_opt.json"),
            os.path.join(parent, seed_dir + "-CM=alm-*-BH=*", "biot_savart_opt.json"),
        ],
        "current Stage 2 output",
    )
    if current_matches is not None:
        return current_matches

    no_tfc_matches = _resolve_unique_stage2_match(
        [
            os.path.join(parent, no_tfc_dir + "-CM=penalty", "biot_savart_opt.json"),
            os.path.join(parent, no_tfc_dir + "-CM=penalty-BH=*", "biot_savart_opt.json"),
            os.path.join(parent, no_tfc_dir + "-BH=*", "biot_savart_opt.json"),
        ],
        "legacy Stage 2 output (missing TFC segment)",
    )
    if no_tfc_matches is not None:
        return no_tfc_matches

    legacy_matches = _resolve_unique_stage2_match(
        [
            os.path.join(parent, legacy_dir + "-CM=penalty", "biot_savart_opt.json"),
            os.path.join(parent, legacy_dir + "-CM=penalty-BH=*", "biot_savart_opt.json"),
            os.path.join(parent, legacy_dir + "-BH=*", "biot_savart_opt.json"),
        ],
        "legacy Stage 2 output (missing CCT/CT segments)",
    )
    if legacy_matches is not None:
        return legacy_matches

    return current_penalty_candidate


def load_stage2_results(stage2_bs_path):
    stage2_results_path = os.path.join(os.path.dirname(stage2_bs_path), "results.json")
    with open(stage2_results_path, "r", encoding="utf-8") as infile:
        stage2_results = json.load(infile)
    return stage2_results_path, stage2_results


def infer_uniform_tf_current_A(tf_coils):
    if not tf_coils:
        return None
    tf_currents = np.asarray([coil.current.get_value() for coil in tf_coils], dtype=float)
    if np.allclose(tf_currents, tf_currents[0]):
        return float(tf_currents[0])
    return None


def resolve_stage2_tf_current_A(stage2_results, tf_coils):
    recorded_tf_current = stage2_results.get("TF_CURRENT_A")
    if recorded_tf_current is not None:
        return float(recorded_tf_current)
    return infer_uniform_tf_current_A(tf_coils)


def resolve_plasma_current_settings(args):
    raw_boozer_I = args.boozer_I
    plasma_current_A = args.plasma_current_A

    if plasma_current_A is not None:
        if raw_boozer_I is not None:
            raise ValueError("Cannot use --plasma-current-A together with --boozer-I")
        return {
            "boozer_I": physical_current_to_boozer_I(plasma_current_A),
            "plasma_current_A": float(plasma_current_A),
            "input_source": "physical_A",
            "mode": "boozer_surrogate",
        }

    if raw_boozer_I is not None:
        return {
            "boozer_I": float(raw_boozer_I),
            "plasma_current_A": boozer_I_to_physical_current_A(raw_boozer_I),
            "input_source": "raw_boozer_I",
            "mode": "boozer_surrogate",
        }

    return {
        "boozer_I": 0.0,
        "plasma_current_A": 0.0,
        "input_source": "default_zero",
        "mode": "disabled",
    }


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


def apply_default_stage2_seed_args(args):
    default_seed = DEFAULT_STAGE2_SEEDS_BY_PLASMA.get(args.plasma_surf_filename, {})
    if args.stage2_seed_major_radius is None:
        args.stage2_seed_major_radius = default_seed.get("major_radius", 0.915)
    if args.stage2_seed_toroidal_flux is None:
        args.stage2_seed_toroidal_flux = default_seed.get("toroidal_flux", 0.24)
    if args.stage2_seed_length_weight is None:
        args.stage2_seed_length_weight = default_seed.get("length_weight", 0.0005)
    if args.stage2_seed_cc_weight is None:
        args.stage2_seed_cc_weight = default_seed.get("cc_weight", 100.0)
    if args.stage2_seed_curvature_weight is None:
        args.stage2_seed_curvature_weight = default_seed.get("curvature_weight", 0.0001)
    if args.stage2_seed_cc_threshold is None:
        args.stage2_seed_cc_threshold = default_seed.get("cc_threshold", 0.05)
    if args.stage2_seed_curvature_threshold is None:
        args.stage2_seed_curvature_threshold = default_seed.get("curvature_threshold", 40.0)
    if args.stage2_seed_banana_surf_radius is None:
        args.stage2_seed_banana_surf_radius = default_seed.get("banana_surf_radius", 0.22)
    if args.stage2_seed_tf_current_A is None:
        args.stage2_seed_tf_current_A = default_seed.get("tf_current_A", 1.0e5)
    if args.stage2_seed_order is None:
        args.stage2_seed_order = default_seed.get("order", 2)
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-stage Boozer/quasi-symmetry optimization from a Stage 2 seed.",
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
        default=os.environ.get("SINGLE_STAGE_OUTPUT_ROOT", DEFAULT_SINGLE_STAGE_OUTPUT_ROOT),
        help="Directory where the single-stage output family will be written.",
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=float(os.environ["BANANA_SURF_RADIUS"]) if "BANANA_SURF_RADIUS" in os.environ else None,
        help="Coil surface minor radius. Defaults to the Stage 2 seed radius when omitted.",
    )
    parser.add_argument("--nphi", type=int, default=int(os.environ.get("NPHI", "255")))
    parser.add_argument("--ntheta", type=int, default=int(os.environ.get("NTHETA", "64")))
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build the initial Boozer surface, write init artifacts, and skip the optimizer.",
    )
    parser.add_argument("--mpol", type=int, default=int(os.environ.get("MPOL", "8")))
    parser.add_argument("--ntor", type=int, default=int(os.environ.get("NTOR", "6")))
    parser.add_argument("--vol-target", type=float, default=float(os.environ.get("VOL_TARGET", "0.10")))
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=float(os.environ.get("CONSTRAINT_WEIGHT", "1.0")),
        help=(
            "Boozer constraint weight. Use a non-negative value for least-squares mode "
            "(default 1.0). Use a negative value to select the exact Boozer Newton solver."
        ),
    )
    parser.add_argument(
        "--boozer-I",
        type=float,
        default=float(os.environ["BOOZER_I"]) if "BOOZER_I" in os.environ else None,
        help="Expert/internal Boozer-current input. Prefer --plasma-current-A.",
    )
    parser.add_argument(
        "--plasma-current-A",
        type=float,
        default=float(os.environ["PLASMA_CURRENT_A"]) if "PLASMA_CURRENT_A" in os.environ else None,
        help="User-facing enclosed toroidal plasma current in physical SI amperes.",
    )
    parser.add_argument("--maxiter", type=int, default=int(os.environ.get("MAXITER", "300")))
    parser.add_argument(
        "--num-surfaces",
        type=int,
        choices=[1, 2],
        default=int(os.environ.get("NUM_SURFACES", "1")),
        help="Number of nested Boozer surfaces to optimize together (v1 supports 1 or 2).",
    )
    parser.add_argument(
        "--inner-surface-ratio",
        type=float,
        default=float(os.environ.get("INNER_SURFACE_RATIO", "0.8")),
        help=(
            "When --num-surfaces=2, use this factor times the Stage 2 toroidal-flux label "
            "to build the inner equilibrium reference surface and derive its target volume."
        ),
    )
    parser.add_argument(
        "--surface-gap-threshold",
        type=float,
        default=float(os.environ.get("SURFACE_GAP_THRESHOLD", "0.0")),
        help="Minimum allowed point-cloud gap between adjacent optimized Boozer surfaces in multi-surface mode.",
    )
    parser.add_argument(
        "--multisurface-ramp-iterations",
        type=int,
        default=int(os.environ.get("MULTISURFACE_RAMP_ITERATIONS", "5")),
        help=(
            "Number of accepted outer iterations over which inner-surface search weight ramps "
            "from --inner-surface-initial-weight to 1.0 in two-surface mode. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--inner-surface-initial-weight",
        type=float,
        default=float(os.environ.get("INNER_SURFACE_INITIAL_WEIGHT", "0.0")),
        help=(
            "Initial search weight applied to inner-surface QS/Boozer terms in two-surface mode. "
            "Must be between 0 and 1."
        ),
    )
    parser.add_argument(
        "--multisurface-initial-step-scale",
        type=float,
        default=float(os.environ.get("MULTISURFACE_INITIAL_STEP_SCALE", "1.0")),
        help=(
            "Physical step scale for an optional first outer-optimization phase in two-surface mode. "
            "Values below 1.0 shrink early L-BFGS-B moves in a mathematically consistent scaled coordinate system."
        ),
    )
    parser.add_argument(
        "--multisurface-initial-step-maxiter",
        type=int,
        default=int(os.environ.get("MULTISURFACE_INITIAL_STEP_MAXITER", "0")),
        help=(
            "Maximum outer iterations to run in the scaled first phase of two-surface mode. "
            "Set to 0 to disable the early-step continuation phase."
        ),
    )
    parser.add_argument(
        "--topology-gate-fieldlines",
        type=int,
        default=int(os.environ.get("TOPOLOGY_GATE_FIELDLINES", "4")),
        help=(
            "Number of cheap outer-surface field lines used for the search-time topology gate in "
            "multi-surface mode. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--topology-gate-tmax",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_TMAX", "2.0")),
        help="Integration horizon for the search-time topology gate field-line traces.",
    )
    parser.add_argument(
        "--topology-gate-tol",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_TOL", "1e-7")),
        help="Integrator tolerance for the search-time topology gate field-line traces.",
    )
    parser.add_argument(
        "--topology-gate-survival-threshold",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_SURVIVAL_THRESHOLD", "0.25")),
        help=(
            "Minimum survival fraction required by the cheap search-time topology gate in "
            "multi-surface mode."
        ),
    )
    parser.add_argument(
        "--topology-gate-penalty-scale",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_PENALTY_SCALE", "4.0")),
        help=(
            "Scale factor for topology-deficit rejection severity. Applied only when a "
            "candidate fails the search-time topology gate and the solver falls back to "
            "the last accepted gradient."
        ),
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=float(os.environ.get("FTOL", "1e-15")),
        help="L-BFGS-B function tolerance (default: 1e-15).",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=float(os.environ.get("GTOL", "1e-15")),
        help="L-BFGS-B gradient tolerance (default: 1e-15).",
    )
    parser.add_argument(
        "--constraint-method",
        choices=["penalty", "alm"],
        default=os.environ.get("CONSTRAINT_METHOD", "penalty"),
        help="Use the legacy weighted-penalty objective or the augmented Lagrangian outer loop.",
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
        help="Distance soft-min temperature for single-stage ALM spacing constraints.",
    )
    parser.add_argument(
        "--alm-curvature-smoothing",
        type=float,
        default=float(os.environ.get("ALM_CURVATURE_SMOOTHING", "0.05")),
        help="Curvature smooth-max temperature for single-stage ALM curvature constraints.",
    )
    parser.add_argument("--iota-target", type=float, default=float(os.environ.get("IOTA_TARGET", "0.15")))
    parser.add_argument("--num-tf-coils", type=int, default=int(os.environ.get("NUM_TF_COILS", "20")))
    parser.add_argument(
        "--boozer-stage",
        choices=["initial", "final"],
        default=os.environ.get("BOOZER_STAGE", "initial"),
        help="Use least-squares Boozer residual during initial stage or exact residual during final stage.",
    )
    parser.add_argument("--cc-dist", type=float, default=float(os.environ.get("CC_DIST", "0.05")))
    parser.add_argument("--curvature-threshold", type=float, default=float(os.environ.get("CURVATURE_THRESHOLD", "40")))
    parser.add_argument("--cc-weight", type=float, default=float(os.environ.get("CC_WEIGHT", "100")))
    parser.add_argument("--curvature-weight", type=float, default=float(os.environ.get("CURVATURE_WEIGHT", "0.1")))
    parser.add_argument("--length-weight", type=float, default=float(os.environ.get("SS_LENGTH_WEIGHT", "1")),
                        help="Curve length penalty weight (default 1).")
    parser.add_argument("--res-weight", type=float, default=float(os.environ.get("RES_WEIGHT", "1000")),
                        help="Boozer residual penalty weight (default 1000).")
    parser.add_argument("--iotas-weight", type=float, default=float(os.environ.get("IOTAS_WEIGHT", "100")),
                        help="Iota target tracking weight (default 100).")
    parser.add_argument("--cs-weight", type=float, default=float(os.environ.get("CS_WEIGHT", "1")),
                        help="Coil-surface distance penalty weight (default 1).")
    parser.add_argument("--cs-dist", type=float, default=float(os.environ.get("CS_DIST", "0.02")),
                        help="Minimum coil-surface distance in meters (default 0.02).")
    parser.add_argument("--surf-dist-weight", type=float, default=float(os.environ.get("SURF_DIST_WEIGHT", "1000")),
                        help="Surface-vessel distance penalty weight (default 1000).")
    parser.add_argument("--ss-dist", type=float, default=float(os.environ.get("SS_DIST", "0.04")),
                        help="Minimum surface-vessel distance in meters (default 0.04).")
    parser.add_argument("--maxcor", type=int, default=int(os.environ.get("MAXCOR", "300")),
                        help="L-BFGS-B memory (number of corrections, default 300).")
    parser.add_argument(
        "--stage2-source",
        choices=["database", "local"],
        default=os.environ.get("STAGE2_SOURCE", "database"),
        help="Resolve the Stage 2 seed from the archive database or from local STAGE_2 outputs.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=os.environ.get("STAGE2_BS_PATH"),
        help="Explicit path to the Stage 2 biot_savart_opt.json seed. Overrides all derived seed settings.",
    )
    parser.add_argument(
        "--local-stage2-root",
        default=os.environ.get("LOCAL_STAGE2_ROOT", DEFAULT_LOCAL_STAGE2_ROOT),
        help="Directory that contains local STAGE_2 outputs-[plasma]/... runs.",
    )
    parser.add_argument(
        "--database-stage2-root",
        default=os.environ.get("DATABASE_STAGE2_ROOT", DEFAULT_DATABASE_STAGE2_ROOT),
        help="Directory that contains DATABASE/COIL_OPTIMIZATION/outputs.",
    )
    parser.add_argument(
        "--stage2-seed-major-radius",
        type=float,
        default=float(os.environ["STAGE2_SEED_MAJOR_RADIUS"]) if "STAGE2_SEED_MAJOR_RADIUS" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-toroidal-flux",
        type=float,
        default=float(os.environ["STAGE2_SEED_TOROIDAL_FLUX"]) if "STAGE2_SEED_TOROIDAL_FLUX" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-length-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_LENGTH_WEIGHT"]) if "STAGE2_SEED_LENGTH_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-cc-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CC_WEIGHT"]) if "STAGE2_SEED_CC_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-curvature-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CURVATURE_WEIGHT"]) if "STAGE2_SEED_CURVATURE_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-cc-threshold",
        type=float,
        default=float(os.environ["STAGE2_SEED_CC_THRESHOLD"]) if "STAGE2_SEED_CC_THRESHOLD" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-curvature-threshold",
        type=float,
        default=float(os.environ["STAGE2_SEED_CURVATURE_THRESHOLD"]) if "STAGE2_SEED_CURVATURE_THRESHOLD" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-banana-surf-radius",
        type=float,
        default=float(os.environ["STAGE2_SEED_BANANA_SURF_RADIUS"]) if "STAGE2_SEED_BANANA_SURF_RADIUS" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-tf-current-A",
        type=float,
        default=float(os.environ["STAGE2_SEED_TF_CURRENT_A"]) if "STAGE2_SEED_TF_CURRENT_A" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-order",
        type=int,
        default=int(os.environ["STAGE2_SEED_ORDER"]) if "STAGE2_SEED_ORDER" in os.environ else None,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.environ.get("CHECKPOINT_EVERY", "0")),
        help="Save checkpoint artifacts every N accepted iterations (0 = disabled, default).",
    )
    parser.add_argument(
        "--topology-scorer-every",
        type=int,
        default=int(os.environ.get("TOPOLOGY_SCORER_EVERY", "0")),
        help="Run medium-fidelity topology scorer every N accepted iterations (0 = disabled, default).",
    )
    parser.add_argument(
        "--topology-scorer-nfieldlines",
        type=int,
        default=int(os.environ.get("TOPOLOGY_SCORER_NFIELDLINES", "12")),
        help="Number of field lines for callback topology scorer (default 12).",
    )
    parser.add_argument(
        "--topology-scorer-tmax",
        type=float,
        default=float(os.environ.get("TOPOLOGY_SCORER_TMAX", "50.0")),
        help="Integration horizon for callback topology scorer (default 50.0).",
    )
    add_confinement_surrogate_args(parser)
    parser.add_argument(
        "--basin-hops",
        type=int,
        default=int(os.environ.get("BASIN_HOPS", "0")),
        help="Number of basin-hopping restarts (0 = single L-BFGS-B, default).",
    )
    parser.add_argument(
        "--basin-stepsize",
        type=float,
        default=float(os.environ.get("BASIN_STEPSIZE", "0.01")),
        help="Perturbation scale for basin-hopping (default 0.01).",
    )
    parser.add_argument(
        "--basin-seed",
        type=int,
        default=int(os.environ.get("BASIN_SEED", "-1")),
        help="RNG seed for basin-hopping (-1 = random). Set for reproducibility.",
    )
    return parser.parse_args()


class BoozerResidualExact(Optimizable):
    r"""
    This term returns the Boozer residual penalty term
    
    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.
    
    where
    
    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[(G + \iota I)\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]
    
    """

    def __init__(self, boozer_surface, bs, constraint_weight=0.0):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        # same number of points as on the solved surface
        nphis = in_surface.quadpoints_phi.size
        phis = np.linspace(0,1./in_surface.nfp,nphis*4,endpoint=False)
        nthetas = in_surface.quadpoints_theta.size
        thetas = np.linspace(0,1,nthetas*4,endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        import warnings
        warnings.warn("BoozerResidualExact: constraint_weight forced to 0.0", stacklevel=2)
        self.constraint_weight = 0.0
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """
        
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def _boozer_current_I(self):
        return self.boozer_surface.res.get("I", getattr(self.boozer_surface, "I", 0.0))

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        I = self._boozer_current_I()
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=True, I=I)
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)
        
        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil
        
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil
        
    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        
        surface = self.surface
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        I = self._boozer_current_I()
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=True, I=I)

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)
        
        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB

def initialize_boozer_surface(surf_prev, mpol, ntor, bs, vol_target, constraint_weight, iota, G0, boozer_I=0.0, nfp=5):
    """
    This initializes the boozer surface, using either the boozer "exact" algorithm, or the boozer "least squares" algorithm

    surf_prev: Any instance of simsopt.geo.Surface. This is the initial guess for the boozer surface solver
    mpol: SurfaceXYZTensorFourier resolution (both toroidal and poloidal)
    bs: simsopt.field.BiotSavart instance
    vol_target: target volume to be enclosed by the boozer surface
    constraint_weight: Set to 1.0 to use Boozer least square, None to use Boozer exact
    iota: initial guess for iota value on the surface
    G0: Value of net current going through the torus hole
    nfp: number of field periods (default 5 for banana coils)
    """
    surf = SurfaceXYZTensorFourier(
          mpol=mpol,ntor=ntor,nfp=nfp,stellsym=True,
          quadpoints_theta=surf_prev.quadpoints_theta,
          quadpoints_phi=surf_prev.quadpoints_phi
          )
    surf.least_squares_fit(surf_prev.gamma())

    if constraint_weight is not None:
        # Boozer least square approach
        print("Generating Boozer least squares surface...")
        vol = Volume(surf)
        boozer_surface = BoozerSurface(bs, surf, vol, vol_target, constraint_weight, options={'verbose':True}, I=boozer_I)
    else:
        # Boozer exact approach
        print("Generating Boozer exact surface...")
        surf_exact = SurfaceXYZTensorFourier(
              mpol=mpol,ntor=ntor,nfp=nfp,stellsym=True,
              quadpoints_theta=np.linspace(0,1,2*mpol+1,endpoint=False),
              quadpoints_phi=np.linspace(0,1./nfp,2*ntor+1,endpoint=False),
              dofs=surf.dofs
              )
    
        vol = Volume(surf_exact)
        boozer_surface = BoozerSurface(bs, surf_exact, vol, vol_target, None, options={'verbose':True}, I=boozer_I)

    # Run boozer surface algorithm
    res = boozer_surface.run_code(iota, G0)
    print(f"G0 from solve: {res['G']}")
    print(f"iota from solve: {res['iota']}")

    # Check if boozer algo is successful
    success1 = res['success'] # True if the boozer surface algo converged
    try:
        success2 = not boozer_surface.surface.is_self_intersecting() # True if surface is not self intersecting
    except Exception:
        success2 = False  # surface that folds is self-intersecting
    success = success1 and success2
    if not success:
        print(
            "Boozer initialization failed: "
            f"solve_success={success1}, "
            f"self_intersecting={not success2}, "
            f"volume={boozer_surface.surface.volume()}, "
            f"iota_guess={iota}, "
            f"iota_solved={res['iota']}"
        )
        raise RuntimeError("Something went wrong with the Boozer solve...")

    return boozer_surface


def build_surface_configs(
    file_loc,
    nphi,
    ntheta,
    seed_label,
    major_radius,
    outer_target_volume,
    num_surfaces,
    inner_surface_ratio,
):
    return _build_surface_configs_impl(
        file_loc,
        nphi,
        ntheta,
        seed_label,
        major_radius,
        outer_target_volume,
        num_surfaces,
        inner_surface_ratio,
        surface_factory=SurfaceRZFourier,
    )


def build_hbt_reference_surfaces(nfp, banana_surf_radius):
    surfaces = build_banana_reference_surfaces(nfp, banana_surf_radius)
    return surfaces.vessel, surfaces.hbt, surfaces.coil_winding_surface


def evaluate_single_stage_hardware_constraints(
    curve_curve_min_dist,
    cc_dist,
    curve_surface_min_dist,
    cs_dist,
    surface_vessel_min_dist,
    ss_dist,
    max_curvature,
    curvature_threshold,
):
    return _evaluate_single_stage_hardware_constraints(
        curve_curve_min_dist,
        cc_dist,
        curve_surface_min_dist,
        cs_dist,
        surface_vessel_min_dist,
        ss_dist,
        max_curvature,
        curvature_threshold,
    )


def compute_single_stage_surface_vessel_min_dist(
    surface_vessel_distance_obj,
    surface_status,
    outer_surface=None,
    vessel_surface=None,
):
    return _compute_single_stage_surface_vessel_min_dist(
        surface_vessel_distance_obj,
        surface_status,
        outer_surface,
        vessel_surface,
    )


def evaluate_topology_gate(surface, bfield, nfieldlines, tmax, tol, survival_threshold):
    return _evaluate_topology_gate_impl(
        surface,
        bfield,
        nfieldlines,
        tmax,
        tol,
        survival_threshold,
        surface_classifier_factory=SurfaceClassifier,
        levelset_stopping_criterion_cls=LevelsetStoppingCriterion,
        max_z_stopping_criterion_cls=MaxZStoppingCriterion,
        min_z_stopping_criterion_cls=MinZStoppingCriterion,
        min_r_stopping_criterion_cls=MinRStoppingCriterion,
        max_r_stopping_criterion_cls=MaxRStoppingCriterion,
        compute_fieldlines_fn=compute_fieldlines,
        midplane_seed_radii_fn=_midplane_seed_radii,
        topology_stop_reason_fn=_topology_stop_reason,
        topology_toroidal_angle_fn=_topology_toroidal_angle,
    )


def topology_gate_deficit(status):
    return _topology_gate_deficit(status)


def average_surface_objectives(objectives, weights=None):
    return _average_surface_objectives_impl(objectives, weights=weights)


def confinement_surrogate_kwargs():
    return {
        "surrogate_worst_k": CONFINEMENT_SURROGATE_WORST_K,
        "surrogate_early_exit_threshold": CONFINEMENT_SURROGATE_EARLY_THRESHOLD,
        "surrogate_mean_weight": CONFINEMENT_SURROGATE_MEAN_WEIGHT,
        "surrogate_worst_weight": CONFINEMENT_SURROGATE_WORST_WEIGHT,
        "surrogate_early_weight": CONFINEMENT_SURROGATE_EARLY_WEIGHT,
    }


def checkpoint_confinement_objective(proxy_objective, topology_result, confinement_weight):
    return float(proxy_objective) + float(confinement_weight) * float(topology_result["confinement_loss"])


def validate_confinement_surrogate_args(args):
    if args.confinement_objective_weight < 0.0:
        raise ValueError("--confinement-objective-weight must be non-negative")
    if args.confinement_surrogate_worst_k <= 0:
        raise ValueError("--confinement-surrogate-worst-k must be positive")
    if not (0.0 < args.confinement_surrogate_early_threshold <= 1.0):
        raise ValueError("--confinement-surrogate-early-threshold must be in (0, 1]")
    if min(
        args.confinement_surrogate_mean_weight,
        args.confinement_surrogate_worst_weight,
        args.confinement_surrogate_early_weight,
    ) < 0.0:
        raise ValueError("--confinement-surrogate-* weights must be non-negative")


@dataclass(frozen=True)
class RunIdentityConfig:
    stage2_bs_path: str
    stage: str
    constraint_weight: float
    constraint_method: str
    vol_target: float
    iota_target: float
    boozer_I: float
    plasma_current_A: float
    cc_dist: float
    cc_weight: float
    curvature_weight: float
    curvature_threshold: float
    banana_surf_radius: float
    nphi: int
    ntheta: int
    init_only: bool
    basin_hops: int
    basin_stepsize: float
    rng_seed: int | None
    ftol: float | None
    gtol: float | None
    alm_max_outer_iters: int
    alm_penalty_init: float
    alm_penalty_scale: float
    alm_feas_tol: float
    alm_stationarity_tol: float
    num_surfaces: int
    inner_surface_ratio: float
    surface_gap_threshold: float
    multisurface_ramp_iterations: int
    inner_surface_initial_weight: float
    multisurface_initial_step_scale: float
    multisurface_initial_step_maxiter: int
    topology_gate_fieldlines: int
    topology_gate_tmax: float
    topology_gate_tol: float
    topology_gate_survival_threshold: float
    topology_gate_penalty_scale: float
    topology_scorer_every: int
    topology_scorer_nfieldlines: int
    topology_scorer_tmax: float
    confinement_objective_weight: float
    confinement_surrogate_worst_k: int
    confinement_surrogate_early_threshold: float
    confinement_surrogate_mean_weight: float
    confinement_surrogate_worst_weight: float
    confinement_surrogate_early_weight: float
    alm_trust_radius_init: float
    alm_trust_radius_min: float
    alm_trust_radius_shrink: float
    alm_trust_radius_grow: float
    alm_max_inner_attempts: int
    alm_max_subproblem_continuations: int
    alm_distance_smoothing: float
    alm_curvature_smoothing: float


def make_run_identity_config(
    args,
    stage2_bs_path,
    stage,
    constraint_weight,
    constraint_method,
    vol_target,
    iota_target,
    boozer_I,
    plasma_current_A,
    banana_surf_radius,
    nphi,
    ntheta,
    rng_seed,
):
    return RunIdentityConfig(
        stage2_bs_path=stage2_bs_path,
        stage=stage,
        constraint_weight=constraint_weight,
        constraint_method=constraint_method,
        vol_target=vol_target,
        iota_target=iota_target,
        boozer_I=boozer_I,
        plasma_current_A=plasma_current_A,
        cc_dist=args.cc_dist,
        cc_weight=args.cc_weight,
        curvature_weight=args.curvature_weight,
        curvature_threshold=args.curvature_threshold,
        banana_surf_radius=banana_surf_radius,
        nphi=nphi,
        ntheta=ntheta,
        init_only=args.init_only,
        basin_hops=args.basin_hops,
        basin_stepsize=args.basin_stepsize,
        rng_seed=rng_seed,
        ftol=args.ftol,
        gtol=args.gtol,
        alm_max_outer_iters=args.alm_max_outer_iters,
        alm_penalty_init=args.alm_penalty_init,
        alm_penalty_scale=args.alm_penalty_scale,
        alm_feas_tol=args.alm_feas_tol,
        alm_stationarity_tol=args.alm_stationarity_tol,
        num_surfaces=args.num_surfaces,
        inner_surface_ratio=args.inner_surface_ratio,
        surface_gap_threshold=args.surface_gap_threshold,
        multisurface_ramp_iterations=args.multisurface_ramp_iterations,
        inner_surface_initial_weight=args.inner_surface_initial_weight,
        multisurface_initial_step_scale=args.multisurface_initial_step_scale,
        multisurface_initial_step_maxiter=args.multisurface_initial_step_maxiter,
        topology_gate_fieldlines=args.topology_gate_fieldlines,
        topology_gate_tmax=args.topology_gate_tmax,
        topology_gate_tol=args.topology_gate_tol,
        topology_gate_survival_threshold=args.topology_gate_survival_threshold,
        topology_gate_penalty_scale=args.topology_gate_penalty_scale,
        topology_scorer_every=args.topology_scorer_every,
        topology_scorer_nfieldlines=args.topology_scorer_nfieldlines,
        topology_scorer_tmax=args.topology_scorer_tmax,
        confinement_objective_weight=args.confinement_objective_weight,
        confinement_surrogate_worst_k=args.confinement_surrogate_worst_k,
        confinement_surrogate_early_threshold=args.confinement_surrogate_early_threshold,
        confinement_surrogate_mean_weight=args.confinement_surrogate_mean_weight,
        confinement_surrogate_worst_weight=args.confinement_surrogate_worst_weight,
        confinement_surrogate_early_weight=args.confinement_surrogate_early_weight,
        alm_trust_radius_init=args.alm_trust_radius_init,
        alm_trust_radius_min=args.alm_trust_radius_min,
        alm_trust_radius_shrink=args.alm_trust_radius_shrink,
        alm_trust_radius_grow=args.alm_trust_radius_grow,
        alm_max_inner_attempts=args.alm_max_inner_attempts,
        alm_max_subproblem_continuations=args.alm_max_subproblem_continuations,
        alm_distance_smoothing=args.alm_distance_smoothing,
        alm_curvature_smoothing=args.alm_curvature_smoothing,
    )


def build_run_identity_config(config):
    return "|".join(str(value) for value in astuple(config))


def evaluate_search_topology_gate(num_surfaces, outer_surface, bfield):
    if num_surfaces <= 1 or TOPOLOGY_GATE_FIELDLINES <= 0:
        return disabled_topology_gate_status(
            TOPOLOGY_GATE_TMAX,
            TOPOLOGY_GATE_TOL,
            TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
        )
    return evaluate_topology_gate(
        outer_surface,
        bfield,
        TOPOLOGY_GATE_FIELDLINES,
        TOPOLOGY_GATE_TMAX,
        TOPOLOGY_GATE_TOL,
        TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
    )


def skipped_topology_gate_status():
    return {
        "evaluated": False,
        "success": None,
        "survived_lines": None,
        "survival_fraction": None,
        "first_exit_time": None,
        "first_exit_angle": None,
        "first_exit_reason": None,
        "stop_reason_counts": None,
    }


def final_topology_gate_for_results(init_only, num_surfaces, outer_surface, bfield):
    if init_only:
        return skipped_topology_gate_status()
    status = evaluate_search_topology_gate(num_surfaces, outer_surface, bfield)
    status["evaluated"] = True
    return status


def evaluate_total_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    CC_WEIGHT,
    JCurveSurface,
    CS_WEIGHT,
    JCurvature,
    CURVATURE_WEIGHT,
    JSurfSurf=None,
    SURF_DIST_WEIGHT=0.0,
):
    return _evaluate_total_objective_impl(
        surface_weights,
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
        JCurveCurve,
        CC_WEIGHT,
        JCurveSurface,
        CS_WEIGHT,
        JCurvature,
        CURVATURE_WEIGHT,
        JSurfSurf=JSurfSurf,
        SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
    )


def evaluate_base_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
):
    return _evaluate_base_objective_impl(
        surface_weights,
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
    )


def evaluate_alm_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    JCurveSurface,
    JCurvature,
    multipliers,
    penalty,
    JSurfSurf=None,
):
    return _evaluate_alm_objective_impl(
        surface_weights,
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
        JCurveCurve,
        JCurveSurface,
        JCurvature,
        multipliers,
        penalty,
        objective_optimizable=JF,
        curves=curves,
        curve_curve_min_distance=CC_DIST,
        outer_surface=outer_surface_data["boozer_surface"].surface,
        curve_surface_min_distance=CS_DIST,
        banana_curve=banana_curve,
        curvature_threshold=CURVATURE_THRESHOLD,
        distance_smoothing=args.alm_distance_smoothing,
        curvature_smoothing=args.alm_curvature_smoothing,
        constraint_names=SINGLE_STAGE_ALM_CONSTRAINT_NAMES,
        curve_curve_constraint_fn=_smooth_min_curve_curve_signed_constraint,
        curve_surface_constraint_fn=_smooth_min_curve_surface_signed_constraint,
        curvature_constraint_fn=_smooth_max_curvature_signed_constraint,
        JSurfSurf=JSurfSurf,
        vessel_surface=VV,
        surface_surface_min_distance=SS_DIST,
        surface_surface_constraint_fn=_smooth_min_surface_surface_signed_constraint,
    )


def evaluate_search_objective(surface_weights):
    if CONSTRAINT_METHOD == "alm":
        return evaluate_alm_objective(
            surface_weights,
            nonQSs,
            brs,
            RES_WEIGHT,
            Jiota,
            IOTAS_WEIGHT,
            JCurveLength,
            LENGTH_WEIGHT,
            JCurveCurve,
            JCurveSurface,
            JCurvature,
            ALM_MULTIPLIERS,
            ALM_PENALTY,
            JSurfSurf=JSurfSurf,
        )
    return evaluate_total_objective(
        surface_weights,
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
        JCurveCurve,
        CC_WEIGHT,
        JCurveSurface,
        CS_WEIGHT,
        JCurvature,
        CURVATURE_WEIGHT,
        JSurfSurf=JSurfSurf,
        SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
    )


def set_alm_runtime_state(multipliers, penalty):
    global ALM_MULTIPLIERS, ALM_PENALTY
    ALM_MULTIPLIERS = np.asarray(multipliers, dtype=float).copy()
    ALM_PENALTY = float(penalty)


def build_single_stage_alm_settings(args):
    return ALMSettings(
        max_outer_iterations=args.alm_max_outer_iters,
        max_subproblem_continuations=args.alm_max_subproblem_continuations,
        penalty_init=args.alm_penalty_init,
        penalty_scale=args.alm_penalty_scale,
        feasibility_tol=args.alm_feas_tol,
        stationarity_tol=args.alm_stationarity_tol,
        trust_radius_init=(
            None if float(args.alm_trust_radius_init) == 0.0 else args.alm_trust_radius_init
        ),
        trust_radius_min=args.alm_trust_radius_min,
        trust_radius_shrink=args.alm_trust_radius_shrink,
        trust_radius_grow=args.alm_trust_radius_grow,
        max_inner_attempts=args.alm_max_inner_attempts,
    )


def build_total_objective(JnonQSRatio, RES_WEIGHT, JBoozerResidual, IOTAS_WEIGHT, Jiota, LENGTH_WEIGHT, JCurveLength, CC_WEIGHT, JCurveCurve, CS_WEIGHT, JCurveSurface, CURVATURE_WEIGHT, JCurvature, SURF_DIST_WEIGHT=0.0, JSurfSurf=None):
    return _build_total_objective_impl(
        JnonQSRatio,
        RES_WEIGHT,
        JBoozerResidual,
        IOTAS_WEIGHT,
        Jiota,
        LENGTH_WEIGHT,
        JCurveLength,
        CC_WEIGHT,
        JCurveCurve,
        CS_WEIGHT,
        JCurveSurface,
        CURVATURE_WEIGHT,
        JCurvature,
        SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
        JSurfSurf=JSurfSurf,
    )


def finalize_surface_stack(x, objective, surface_data, run_state, vessel_surface=None, surface_gap_threshold=0.0, vessel_gap_threshold=0.0):
    status = solve_surface_stack_at_dofs(
        x,
        objective,
        surface_data,
        run_state["surface_state"],
        vessel_surface=vessel_surface,
        surface_gap_threshold=surface_gap_threshold,
        vessel_gap_threshold=vessel_gap_threshold,
        enforce_nesting=True,
    )
    if status["success"]:
        run_state["accepted_x"] = np.asarray(x).copy()
        run_state["surface_state"] = snapshot_surface_states(surface_data)
        run_state["surface_status"] = status
        run_state["J"] = objective.J()
        run_state["dJ"] = objective.dJ().copy()
    else:
        print("Final optimized state rejected; falling back to last accepted state.")
        objective.x = run_state["accepted_x"].copy()
        restore_surface_states(surface_data, run_state["surface_state"])
        run_state["surface_status"] = evaluate_surface_stack(
            surface_data,
            vessel_surface=vessel_surface,
            surface_gap_threshold=surface_gap_threshold,
            vessel_gap_threshold=vessel_gap_threshold,
        )
    run_state["intersecting"] = any(run_state["surface_status"]["self_intersections"])
    return run_state["surface_status"]

def normPlot(surf, bs, filename):
    """Plot normal magnetic field — delegates to shared norm_field_plot."""
    mean_abs_relBfinal_norm, _, _, _, _ = norm_field_plot(surf, bs, filename)
    return mean_abs_relBfinal_norm


def evaluate_search_step(x):
    """
    Objective function for L-BFGS-B optimization.

    Evaluates the total objective function and its gradient for a given set of
    degrees of freedom (coil parameters). Attempts to solve for a valid Boozer
    surface; if unsuccessful (solver failure or self-intersection), returns an
    elevated objective value with the last accepted gradient to trigger line
    search backtracking without corrupting the L-BFGS-B Hessian approximation.

    Args:
        x: Current degrees of freedom (coil parameters)

    Returns:
        Dictionary with objective value and gradient for the current search step.
    """
    dx = np.linalg.norm(x - run_dict['x_prev'])
    outer_entry = surface_data[-1]
    run_dict['x_prev'] = x.copy()
    print(f"Step size: {dx:.2e}")

    run_dict['lscount']+=1
    search_gate = build_surface_search_gate(
        len(surface_data),
        run_dict['accepted_iterations'],
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
        SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        SS_DIST if len(surface_data) > 1 else 0.0,
    )

    stack_status = solve_surface_stack_at_dofs(
        x,
        JF,
        surface_data,
        run_dict['surface_state'],
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=search_gate['surface_gap_threshold'],
        vessel_gap_threshold=search_gate['vessel_gap_threshold'],
        enforce_nesting=search_gate['enforce_nesting'],
    )
    success = stack_status['success']

    rejection_increment = None
    objective_eval = None

    if success:
        search_surface_weights = build_surface_search_weights(
            len(surface_data),
            run_dict['accepted_iterations'],
            MULTISURFACE_RAMP_ITERATIONS,
            INNER_SURFACE_INITIAL_WEIGHT,
        )
        objective_eval = evaluate_search_objective(search_surface_weights)
        J = objective_eval['total']
        dJ = objective_eval['grad']
        topology_status = evaluate_search_topology_gate(
            len(surface_data),
            outer_entry['boozer_surface'].surface,
            bs,
        )
        run_dict['topology_gate_status'] = topology_status
        if topology_status['enabled'] and not topology_status['success']:
            success = False
            rejection_increment = topology_gate_rejection_increment(
                run_dict['J'],
                topology_status,
                TOPOLOGY_GATE_PENALTY_SCALE,
            )
            print("/!\\ /!\\ Topology gate rejected candidate /!\\ /!\\")
            print(
                "Cheap field-line survival "
                f"{topology_status['survived_lines']}/{topology_status['nfieldlines']} "
                f"(fraction={topology_status['survival_fraction']:.3f}, "
                f"threshold={topology_status['survival_threshold']:.3f})"
            )
            print(f"Topology rejection increment = {rejection_increment:.6e}")
            if topology_status['first_exit_time'] is not None:
                print(
                    "First topology exit at "
                    f"t={topology_status['first_exit_time']:.6e}, "
                    f"phi={topology_status['first_exit_angle']:.6e}, "
                    f"reason={topology_status['first_exit_reason']}"
                )

        if success:
            hardware_snapshot = evaluate_single_stage_hardware_snapshot(
                JCurveCurve,
                CC_DIST,
                JCurveSurface,
                CS_DIST,
                JSurfSurf,
                stack_status,
                SS_DIST,
                banana_curve,
                CURVATURE_THRESHOLD,
            )
            hardware_status = hardware_snapshot["status"]
            run_dict["hardware_constraint_status"] = hardware_status
            if not hardware_status["success"]:
                success = False
                rejection_increment = max(abs(run_dict['J']), 1.0)
                print("/!\\ /!\\ Hardware constraints violated /!\\ /!\\")
                for violation in hardware_status["violations"]:
                    print(violation)

        if success:
            run_dict['last_successful_eval'] = objective_eval
            run_dict['last_successful_eval_weights'] = np.asarray(search_surface_weights).copy()
            print(f"Volume: {outer_entry['boozer_surface'].surface.volume()}")
            print(f"Iota: {surface_iota_terms[-1].J()}")
            if len(surface_data) > 1:
                print(f"Surface search weights: {objective_eval['surface_weights'].tolist()}")
                print(f"Surface gate scale: {search_gate['gate_scale']:.6f}")
                print(f"Adjacent surface gaps: {stack_status['adjacent_gaps']}")
                print(f"Outer vessel gap: {stack_status['outer_vessel_gap']}")

    if not success:
        if stack_status['success']:
            print("/!\\ /!\\ Candidate rejected after surface solve /!\\ /!\\")
        else:
            run_dict['topology_gate_status'] = disabled_topology_gate_status(
                TOPOLOGY_GATE_TMAX,
                TOPOLOGY_GATE_TOL,
                TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
            )
            print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not all(stack_status['solve_success']):
            print("Boozer solver failed")
        if any(stack_status['self_intersections']):
            print("Surface is self-intersecting")
        if len(surface_data) > 1:
            if not stack_status['volumes_ordered']:
                print("Surface volumes are not strictly ordered")
            if not stack_status['gap_ok']:
                print(f"Adjacent surfaces too close: {stack_status['adjacent_gaps']}")
            if not stack_status['vessel_gap_ok']:
                print(f"Outer surface too close to vessel: {stack_status['outer_vessel_gap']}")
            if search_gate['enforce_nesting'] and not stack_status['nesting_ok']:
                print(f"Surfaces are not nested on phi slices: {stack_status['bad_nesting_phis']}")
        hardware_status = run_dict.get("hardware_constraint_status")
        if hardware_status is not None and not hardware_status["success"]:
            print("Hardware constraints violated")

        # Elevated J violates Armijo, so the line search backtracks.
        # Returning dJ_old (not negated) avoids the old -dJ corruption path
        # and produces y_k=0 if the step is ever accepted, safely skipping
        # the BFGS Hessian update.
        if rejection_increment is None:
            rejection_increment = max(abs(run_dict['J']), 1.0)
        J = run_dict['J'] + rejection_increment
        dJ = run_dict['dJ'].copy()
        JF.x = run_dict['accepted_x']
        restore_surface_states(surface_data, run_dict['surface_state'])

    evaluation = {"total": J, "grad": dJ}
    if CONSTRAINT_METHOD == "alm":
        metric_eval = objective_eval
        if metric_eval is None or "constraint_values" not in metric_eval:
            metric_eval = run_dict.get("last_successful_eval", run_dict.get("search_eval"))
        if metric_eval is not None and "constraint_values" in metric_eval:
            evaluation.update(
                {
                    "constraint_values": np.asarray(metric_eval["constraint_values"], dtype=float),
                    "max_violation": float(metric_eval["max_violation"]),
                    "stationarity_norm": float(metric_eval["stationarity_norm"]),
                    "metric_grad": np.asarray(
                        metric_eval.get("grad", dJ),
                        dtype=float,
                    ),
                    "metric_stationarity_norm": float(metric_eval["stationarity_norm"]),
                    "constraint_names": list(metric_eval.get("constraint_names", [])),
                    "constraint_grads": [
                        np.asarray(grad, dtype=float)
                        for grad in metric_eval.get("constraint_grads", [])
                    ],
                    "constraint_activity_tolerances": np.asarray(
                        metric_eval.get("constraint_activity_tolerances", []),
                        dtype=float,
                    ),
                    "feasibility_values": np.asarray(
                        metric_eval.get("feasibility_values", metric_eval["constraint_values"]),
                        dtype=float,
                    ),
                    "dual_update_values": np.asarray(
                        metric_eval.get("dual_update_values", metric_eval["constraint_values"]),
                        dtype=float,
                    ),
                    "max_feasibility_violation": float(
                        metric_eval.get(
                            "max_feasibility_violation",
                            metric_eval["max_violation"],
                        )
                    ),
                    "base_total": float(
                        metric_eval.get(
                            "base_total",
                            metric_eval.get("total", run_dict["J"]),
                        )
                    ),
                }
            )
    return evaluation


def fun(x):
    evaluation = evaluate_search_step(x)
    return evaluation["total"], evaluation["grad"]

def callback(x):
    """
    Callback function executed after each successful optimization iteration.

    Stores the accepted state (surface DOFs, iota, G), evaluates and prints
    detailed diagnostics for all objective function components, and logs the
    iteration summary to file. Used for monitoring optimization progress and
    recording convergence history.

    Args:
        x: Current degrees of freedom (coil parameters) from accepted step
    """
    # Update count for tracking
    run_dict['lscount'] = 0
    outer_entry = surface_data[-1]

    # Store last accepted state
    search_surface_weights = build_surface_search_weights(
        len(surface_data),
        run_dict['accepted_iterations'],
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
    )
    search_gate = build_surface_search_gate(
        len(surface_data),
        run_dict['accepted_iterations'],
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
        SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        SS_DIST if len(surface_data) > 1 else 0.0,
    )
    if 'last_successful_eval' in run_dict and np.array_equal(run_dict.get('last_successful_eval_weights', None), search_surface_weights):
        objective_eval = run_dict['last_successful_eval']
    else:
        objective_eval = evaluate_search_objective(search_surface_weights)
    run_dict['surface_state'] = snapshot_surface_states(surface_data)
    run_dict['accepted_x'] = x.copy()
    run_dict['J'] = objective_eval['total']
    run_dict['dJ'] = objective_eval['grad'].copy()
    run_dict['search_eval'] = objective_eval
    topology_status = run_dict['topology_gate_status']
    search_stack_status = evaluate_surface_stack(
        surface_data,
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=search_gate['surface_gap_threshold'],
        vessel_gap_threshold=search_gate['vessel_gap_threshold'],
        enforce_nesting=search_gate['enforce_nesting'],
    )
    full_stack_status = evaluate_surface_stack(
        surface_data,
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        vessel_gap_threshold=SS_DIST if len(surface_data) > 1 else 0.0,
        enforce_nesting=True,
    )
    run_dict['search_surface_status'] = search_stack_status
    run_dict['surface_status'] = full_stack_status
    run_dict['topology_gate_status'] = topology_status

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']
    
    J_QS = objective_eval['J_QS']
    dJ_QS = np.linalg.norm(objective_eval['dJ_QS'])
    J_Boozer = objective_eval['J_Boozer']
    dJ_Boozer = np.linalg.norm(objective_eval['dJ_Boozer'])
    J_iota = objective_eval['J_iota']
    dJ_iota = np.linalg.norm(objective_eval['dJ_iota'])
    J_len = JCurveLength.J()
    dJ_len = np.linalg.norm(JCurveLength.dJ())
    J_cc = JCurveCurve.J()
    dJ_cc = np.linalg.norm(JCurveCurve.dJ())
    J_cs = JCurveSurface.J()
    dJ_cs = np.linalg.norm(JCurveSurface.dJ())
    J_surf = objective_eval['J_surf']
    dJ_surf = np.linalg.norm(objective_eval['dJ_surf'])
    J_curvature = objective_eval['J_curvature']
    dJ_curvature = np.linalg.norm(objective_eval['dJ_curvature'])

    iota_values = [term.J() for term in surface_iota_terms]
    volume_values = [entry['boozer_surface'].surface.volume() for entry in surface_data]
    iota_str = ", ".join(f"{value:.4f}" for value in iota_values)
    volume_str = ", ".join(f"{value:.4f}" for value in volume_values)

    gamma = banana_curve.gamma()
    max_r = np.max(np.sqrt(gamma[:,0]**2 + gamma[:,1]**2))
    max_z = np.max(np.abs(gamma[:,2]))
    length = curvelength.J()
    hardware_snapshot = evaluate_single_stage_hardware_snapshot(
        JCurveCurve,
        CC_DIST,
        JCurveSurface,
        CS_DIST,
        JSurfSurf,
        full_stack_status,
        SS_DIST,
        banana_curve,
        CURVATURE_THRESHOLD,
        outer_entry["boozer_surface"].surface,
        VV,
    )
    curvecurve_min = hardware_snapshot["curve_curve_min_dist"]
    curvesurf_min = hardware_snapshot["curve_surface_min_dist"]
    surface_vessel_min = hardware_snapshot["surface_vessel_min_dist"]
    max_curvature = hardware_snapshot["max_curvature"]
    hardware_status = hardware_snapshot["status"]
    run_dict['hardware_constraint_status'] = hardware_status

    bs.set_points(outer_entry['boozer_surface'].surface.gamma().reshape((-1, 3)))
    unitn = outer_entry['boozer_surface'].surface.unitnormal()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)))
    run_dict['intersecting'] = any(full_stack_status['self_intersections'])

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    if CONSTRAINT_METHOD == "alm":
        print(f"{'Base Objective J':{width}} = {objective_eval['base_total']:.6e}", file=buffer)
        print(f"{'ALM outer iter':{width}} = {run_dict.get('alm_outer_iteration')}", file=buffer)
        print(f"{'ALM penalty μ':{width}} = {ALM_PENALTY:.6e}", file=buffer)
        print(f"{'ALM multipliers':{width}} = {ALM_MULTIPLIERS.tolist()}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'Curve Length Penalty':{width}} = {J_len:.6e} (dJ = {dJ_len:.6e})", file=buffer)
    print(f"{'Curve-Curve Penalty':{width}} = {J_cc:.6e} (min={curvecurve_min:.3e}) (dJ = {dJ_cc:.6e})", file=buffer)
    print(f"{'Curve-Surface Penalty':{width}} = {J_cs:.6e} (min={curvesurf_min:.3e}) (dJ = {dJ_cs:.6e})", file=buffer)
    print(f"{'Surf-Vessel Penalty':{width}} = {J_surf:.6e} (dJ = {dJ_surf:.6e})", file=buffer) 
    print(f"{'Curvature Penalty':{width}} = {J_curvature:.6e} (dJ = {dJ_curvature:.6e})", file=buffer) 
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    if len(surface_data) > 1:
        print(f"{'Surface search weights':{width}} = {objective_eval['surface_weights'].tolist()}", file=buffer)
        print(f"{'Surface gate scale':{width}} = {search_gate['gate_scale']:.6f}", file=buffer)
        print(f"{'Search gap threshold':{width}} = {search_gate['surface_gap_threshold']:.6e}", file=buffer)
        print(f"{'Search vessel gap':{width}} = {search_gate['vessel_gap_threshold']:.6e}", file=buffer)
        print(f"{'Search nesting enforced':{width}} = {search_gate['enforce_nesting']}", file=buffer)
        if topology_status['enabled']:
            print(
                f"{'Topology survival':{width}} = "
                f"{topology_status['survived_lines']}/{topology_status['nfieldlines']} "
                f"(fraction={topology_status['survival_fraction']:.6f}, "
                f"threshold={topology_status['survival_threshold']:.6f})",
                file=buffer,
            )
            print(f"{'Topology stop counts':{width}} = {topology_status['stop_reason_counts']}", file=buffer)
            print(f"{'Topology first exit time':{width}} = {topology_status['first_exit_time']}", file=buffer)
            print(f"{'Topology first exit angle':{width}} = {topology_status['first_exit_angle']}", file=buffer)
            print(f"{'Topology first exit reason':{width}} = {topology_status['first_exit_reason']}", file=buffer)
        print(f"{'Adjacent surface gaps':{width}} = {full_stack_status['adjacent_gaps']}", file=buffer)
        print(f"{'Outer vessel gap':{width}} = {full_stack_status['outer_vessel_gap']:.6e}", file=buffer)
        print(f"{'Surfaces nested':{width}} = {full_stack_status['nesting_ok']}", file=buffer)
        print(f"{'Bad nesting phis':{width}} = {full_stack_status['bad_nesting_phis']}", file=buffer)

    print(f"{'Intersecting':{width}} = {run_dict['intersecting']}", file=buffer)
    print(f"{'Max Curve R':{width}} = {max_r:.6e}", file=buffer)
    print(f"{'Max Curve Z':{width}} = {max_z:.6e}", file=buffer)
    print(f"{'Max Curvature':{width}} = {max_curvature:.6e}", file=buffer)
    print(f"{'Curve Length':{width}} = {length:.6e}", file=buffer)
    print(f"{'Surface-Vessel Min Dist':{width}} = {surface_vessel_min:.6e}", file=buffer)
    print(f"{'Hardware Constraints OK':{width}} = {hardware_status['success']}", file=buffer)
    if hardware_status["violations"]:
        print(f"{'Hardware Violations':{width}} = {hardware_status['violations']}", file=buffer)
    print("="*70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    filename = OUT_DIR_ITER + "/log.txt"
    with open(filename, "a") as f:
        f.write(output_str + "\n")

    # Advance iteration counter
    run_dict['accepted_iterations'] += 1
    run_dict['it'] += 1

    # Periodic checkpoint saving
    if CHECKPOINT_EVERY > 0 and run_dict['accepted_iterations'] % CHECKPOINT_EVERY == 0:
        ckpt_dir = os.path.join(OUT_DIR_ITER, f"checkpoint_iter{run_dict['accepted_iterations']:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        bs.save(os.path.join(ckpt_dir, "biot_savart.json"))
        save_surface_artifacts(surface_data, bs, ckpt_dir, "surf", also_write_outer_legacy=False)
        print(f"  [checkpoint] Saved iteration {run_dict['accepted_iterations']} to {ckpt_dir}")

    # Periodic topology scoring (medium-fidelity confinement evaluation)
    if TOPOLOGY_SCORER_EVERY > 0 and run_dict['accepted_iterations'] % TOPOLOGY_SCORER_EVERY == 0:
        outer_surf = outer_surface_data['boozer_surface'].surface
        topo_result = score_topology(
            outer_surf, bs,
            nfieldlines=TOPOLOGY_SCORER_NFIELDLINES,
            tmax=TOPOLOGY_SCORER_TMAX,
            **confinement_surrogate_kwargs(),
        )
        checkpoint_objective_total = checkpoint_confinement_objective(
            J,
            topo_result,
            CONFINEMENT_OBJECTIVE_WEIGHT,
        )
        topo_entry = {
            "accepted_iteration": run_dict['accepted_iterations'],
            "J": float(J),
            "checkpoint_objective_total": checkpoint_objective_total,
            "survival_fraction": topo_result["survival_fraction"],
            "survived_lines": topo_result["survived_lines"],
            "nfieldlines": topo_result["nfieldlines"],
            "tmax": topo_result["tmax"],
            "mean_exit_time": topo_result["mean_exit_time"],
            "confinement_score": topo_result["confinement_score"],
            "mean_line_loss": topo_result["mean_line_loss"],
            "worst_k_line_loss": topo_result["worst_k_line_loss"],
            "early_exit_fraction": topo_result["early_exit_fraction"],
            "confinement_loss": topo_result["confinement_loss"],
            "confinement_surrogate_k": topo_result["confinement_surrogate_k"],
            "confinement_early_exit_threshold": topo_result["confinement_early_exit_threshold"],
            "stop_reason_counts": topo_result["stop_reason_counts"],
        }
        # Append to archive JSONL
        archive_path = os.path.join(OUT_DIR_ITER, "topology_archive.jsonl")
        with open(archive_path, "a") as af:
            af.write(json.dumps(topo_entry) + "\n")

        # Track best states
        if 'best_topology' not in run_dict or topo_entry['confinement_score'] > run_dict['best_topology']['confinement_score']:
            run_dict['best_topology'] = topo_entry
            # Save checkpoint for best-topology state
            best_dir = os.path.join(OUT_DIR_ITER, "best_topology")
            os.makedirs(best_dir, exist_ok=True)
            bs.save(os.path.join(best_dir, "biot_savart.json"))
            save_surface_artifacts(surface_data, bs, best_dir, "surf", also_write_outer_legacy=False)

        if CONFINEMENT_OBJECTIVE_WEIGHT > 0.0 and (
            'best_confinement_objective' not in run_dict
            or topo_entry['checkpoint_objective_total'] < run_dict['best_confinement_objective']['checkpoint_objective_total']
        ):
            run_dict['best_confinement_objective'] = topo_entry
            best_dir = os.path.join(OUT_DIR_ITER, "best_confinement_objective")
            os.makedirs(best_dir, exist_ok=True)
            bs.save(os.path.join(best_dir, "biot_savart.json"))
            save_surface_artifacts(surface_data, bs, best_dir, "surf", also_write_outer_legacy=False)

        print(
            f"  [topology] iter={run_dict['accepted_iterations']}: "
            f"survival={topo_result['survived_lines']}/{topo_result['nfieldlines']}, "
            f"confinement={topo_result['confinement_score']:.4f}, "
            f"loss={topo_result['confinement_loss']:.4f}, "
            f"mean_exit={topo_result['mean_exit_time']}"
        )


# Convergence tolerances for different mpol values (module-level for testability)
MULTISURFACE_RAMP_ITERATIONS = 0
INNER_SURFACE_INITIAL_WEIGHT = 1.0
TOPOLOGY_GATE_FIELDLINES = 0
TOPOLOGY_GATE_TMAX = 2.0
TOPOLOGY_GATE_TOL = 1e-7
TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.0
CONSTRAINT_METHOD = "penalty"
ALM_MULTIPLIERS = np.zeros(0, dtype=float)
ALM_PENALTY = 1.0
CHECKPOINT_EVERY = 0
TOPOLOGY_SCORER_EVERY = 0
TOPOLOGY_SCORER_NFIELDLINES = 12
TOPOLOGY_SCORER_TMAX = 50.0
CONFINEMENT_OBJECTIVE_WEIGHT = 0.0
CONFINEMENT_SURROGATE_WORST_K = 3
CONFINEMENT_SURROGATE_EARLY_THRESHOLD = 0.2
CONFINEMENT_SURROGATE_MEAN_WEIGHT = 0.2
CONFINEMENT_SURROGATE_WORST_WEIGHT = 0.6
CONFINEMENT_SURROGATE_EARLY_WEIGHT = 0.2


if __name__ == "__main__":
    # ==============================================================================
    # CONFIGURATION PARAMETERS
    # ==============================================================================
    args = apply_default_stage2_seed_args(parse_args())
    stage2_bs_path = build_stage2_bs_path(args)
    stage2_results_path, stage2_results = load_stage2_results(stage2_bs_path)
    R0 = float(stage2_results["MAJOR_RADIUS"])
    s = float(stage2_results["TOROIDAL_FLUX"])
    order = int(stage2_results.get("order", args.stage2_seed_order))

    banana_surf_radius = args.banana_surf_radius if args.banana_surf_radius is not None else float(stage2_results["banana_surf_radius"])
    nphi = args.nphi
    ntheta = args.ntheta
    mpol = args.mpol
    ntor = args.ntor

    # Optimization targets and weights
    vol_target = args.vol_target
    CONSTRAINT_WEIGHT = None if args.constraint_weight < 0 else args.constraint_weight
    CONSTRAINT_METHOD = args.constraint_method
    ALM_MULTIPLIERS = np.zeros(0, dtype=float)
    ALM_PENALTY = args.alm_penalty_init
    plasma_current_settings = resolve_plasma_current_settings(args)
    boozer_I = plasma_current_settings["boozer_I"]
    plasma_current_A = plasma_current_settings["plasma_current_A"]
    plasma_current_input_source = plasma_current_settings["input_source"]
    finite_current_mode = plasma_current_settings["mode"]
    MAXITER = args.maxiter
    CHECKPOINT_EVERY = args.checkpoint_every
    TOPOLOGY_SCORER_EVERY = args.topology_scorer_every
    TOPOLOGY_SCORER_NFIELDLINES = args.topology_scorer_nfieldlines
    TOPOLOGY_SCORER_TMAX = args.topology_scorer_tmax
    CONFINEMENT_OBJECTIVE_WEIGHT = args.confinement_objective_weight
    CONFINEMENT_SURROGATE_WORST_K = args.confinement_surrogate_worst_k
    CONFINEMENT_SURROGATE_EARLY_THRESHOLD = args.confinement_surrogate_early_threshold
    CONFINEMENT_SURROGATE_MEAN_WEIGHT = args.confinement_surrogate_mean_weight
    CONFINEMENT_SURROGATE_WORST_WEIGHT = args.confinement_surrogate_worst_weight
    CONFINEMENT_SURROGATE_EARLY_WEIGHT = args.confinement_surrogate_early_weight
    iota_target = args.iota_target
    num_tf_coils = args.num_tf_coils
    if not (0.0 <= args.inner_surface_initial_weight <= 1.0):
        raise ValueError("--inner-surface-initial-weight must be between 0 and 1")
    if args.multisurface_ramp_iterations < 0:
        raise ValueError("--multisurface-ramp-iterations must be non-negative")
    if not (0.0 < args.multisurface_initial_step_scale <= 1.0):
        raise ValueError("--multisurface-initial-step-scale must be in (0, 1]")
    if args.multisurface_initial_step_maxiter < 0:
        raise ValueError("--multisurface-initial-step-maxiter must be non-negative")
    if args.topology_gate_fieldlines < 0:
        raise ValueError("--topology-gate-fieldlines must be non-negative")
    if args.topology_gate_tmax <= 0:
        raise ValueError("--topology-gate-tmax must be positive")
    if args.topology_gate_tol <= 0:
        raise ValueError("--topology-gate-tol must be positive")
    if not (0.0 <= args.topology_gate_survival_threshold <= 1.0):
        raise ValueError("--topology-gate-survival-threshold must be between 0 and 1")
    if args.topology_gate_penalty_scale < 0.0:
        raise ValueError("--topology-gate-penalty-scale must be non-negative")
    validate_alm_cli_args(args)
    validate_confinement_surrogate_args(args)
    MULTISURFACE_RAMP_ITERATIONS = args.multisurface_ramp_iterations
    INNER_SURFACE_INITIAL_WEIGHT = args.inner_surface_initial_weight
    TOPOLOGY_GATE_FIELDLINES = args.topology_gate_fieldlines
    TOPOLOGY_GATE_TMAX = args.topology_gate_tmax
    TOPOLOGY_GATE_TOL = args.topology_gate_tol
    TOPOLOGY_GATE_SURVIVAL_THRESHOLD = args.topology_gate_survival_threshold
    TOPOLOGY_GATE_PENALTY_SCALE = args.topology_gate_penalty_scale

    # Output directory setup
    OUT_DIR = args.output_root
    os.makedirs(OUT_DIR, exist_ok=True)
    boozer_type = {'initial': 'least_squares', 'final': 'exact'}  # example
    stage = args.boozer_stage

    # ==============================================================================
    # LOAD EQUILIBRIUM AND COILS
    # ==============================================================================
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)
    bs = load(stage2_bs_path)

    # Initialize the boundary magnetic surface and scale it to the target major radius
    surface_configs = build_surface_configs(
        file_loc,
        nphi,
        ntheta,
        s,
        R0,
        vol_target,
        args.num_surfaces,
        args.inner_surface_ratio,
    )
    surf = surface_configs[-1]["initial_surface"]
    banana_surf_nfp = surf.nfp

    VV, hbt, surf_coils = build_hbt_reference_surfaces(banana_surf_nfp, banana_surf_radius)

    # Extract coil information
    coils = bs.coils
    curves = [c.curve for c in coils]
    tf_coils = coils[:num_tf_coils]
    tf_curves = [c.curve for c in tf_coils]
    banana_coils = coils[num_tf_coils:]
    banana_curves = [c.curve for c in banana_coils]
    banana_curve = banana_curves[0]
    stage2_tf_current_A = resolve_stage2_tf_current_A(stage2_results, tf_coils)
    tf_current_sum_abs_A = float(sum(abs(c.current.get_value()) for c in tf_coils))
    current_sum = tf_current_sum_abs_A

    # Calculate G0 parameter from TF coil currents
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    # ==============================================================================
    # OPTIMIZATION SETUP
    # ==============================================================================
    print(f"\n===== Starting single stage optimization for mpol = {mpol} =====")

    # Resolve basin-hopping RNG seed early so it's available for config_hash
    if args.basin_hops > 0:
        rng_seed = args.basin_seed if args.basin_seed >= 0 else int.from_bytes(os.urandom(4), 'big')
    else:
        rng_seed = None

    run_identity_config = make_run_identity_config(
        args,
        stage2_bs_path,
        stage,
        CONSTRAINT_WEIGHT,
        args.constraint_method,
        vol_target,
        iota_target,
        boozer_I,
        plasma_current_A,
        banana_surf_radius,
        nphi,
        ntheta,
        rng_seed,
    )
    config_str = build_run_identity_config(run_identity_config)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}-{config_hash}"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # Initialize Boozer surfaces with target parameters
    surface_data = []
    for config in surface_configs:
        boozer_surface = initialize_boozer_surface(
            config["initial_surface"],
            mpol,
            ntor,
            bs,
            config["target_volume"],
            CONSTRAINT_WEIGHT,
            iota_target,
            G0,
            boozer_I,
            nfp=banana_surf_nfp,
        )
        surface_data.append({
            "name": config["name"],
            "seed_label": config["seed_label"],
            "target_volume": config["target_volume"],
            "boozer_surface": boozer_surface,
        })
    outer_surface_data = surface_data[-1]

    # ==============================================================================
    # SAVE INITIAL STATE
    # ==============================================================================
    # Save initial coil configurations
    curves_to_vtk(curves, OUT_DIR_ITER + "/curves_init", close=True)
    bs.save(OUT_DIR_ITER + "/biot_savart_init.json")

    save_surface_artifacts(surface_data, bs, OUT_DIR_ITER, "surf_init", also_write_outer_legacy=True)
    print(f"Volume: {outer_surface_data['boozer_surface'].surface.volume()}")

    # Generate initial diagnostic plots
    initial_field_error = normPlot(outer_surface_data['boozer_surface'].surface, bs, OUT_DIR_ITER + "/NormPlotInitial")
    try:
        cross_section_plot(surf_coils, outer_surface_data['boozer_surface'].surface, banana_curve, OUT_DIR_ITER + "/CrossSectionInitial", hbt, VV)
    except Exception as e:
        print(f"WARNING: CrossSectionInitial plot failed (surface may fold at high mpol): {e}")
    initial_volume = outer_surface_data['boozer_surface'].surface.volume()
    initial_iota = Iotas(outer_surface_data['boozer_surface']).J()
    initial_max_curvature = np.max(banana_curve.kappa())
    initial_surface_volumes = [entry["boozer_surface"].surface.volume() for entry in surface_data]
    initial_surface_iotas = [Iotas(entry["boozer_surface"]).J() for entry in surface_data]

    # ==============================================================================
    # DEFINE OBJECTIVE FUNCTION COMPONENTS
    # ==============================================================================
    # Quasi-symmetry and Boozer coordinate residuals
    surface_iota_terms = [Iotas(entry["boozer_surface"]) for entry in surface_data]
    nonQSs = [NonQuasiSymmetricRatio(entry["boozer_surface"], BiotSavart(coils)) for entry in surface_data]
    if boozer_type[stage] == 'exact':
        brs = [BoozerResidualExact(entry["boozer_surface"], BiotSavart(coils)) for entry in surface_data]
    else:
        brs = [BoozerResidual(entry["boozer_surface"], BiotSavart(coils)) for entry in surface_data]

    # Objective function weights and parameters (all configurable via CLI)
    # Baseline default floors enforced via max() — weights are free, thresholds are clamped.
    LENGTH_WEIGHT = args.length_weight
    RES_WEIGHT = args.res_weight
    IOTAS_WEIGHT = args.iotas_weight
    CC_WEIGHT = args.cc_weight
    CC_DIST = max(args.cc_dist, 0.05)            # Baseline default floor
    if args.cc_dist < 0.05:
        print(f"WARNING: --cc-dist {args.cc_dist} below baseline default, clamped to 0.05")
    CS_WEIGHT = args.cs_weight
    CS_DIST = max(args.cs_dist, 0.02)            # Baseline default floor
    if args.cs_dist < 0.02:
        print(f"WARNING: --cs-dist {args.cs_dist} below baseline default, clamped to 0.02")
    SURF_DIST_WEIGHT = args.surf_dist_weight
    SS_DIST = max(args.ss_dist, 0.04)            # Baseline default floor
    if args.ss_dist < 0.04:
        print(f"WARNING: --ss-dist {args.ss_dist} below baseline default, clamped to 0.04")
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = max(args.curvature_threshold, 40)
    if args.curvature_threshold < 40:
        print(f"WARNING: --curvature-threshold {args.curvature_threshold} below hardware floor, clamped to 40")
    SURFACE_GAP_THRESHOLD = max(args.surface_gap_threshold, 0.0)
    if len(surface_data) > 1 and SURF_DIST_WEIGHT != 0:
        print("WARNING: SURF_DIST_WEIGHT is diagnostic-only in multi-surface mode; outer-vessel spacing is enforced as a rejection gate.")

    # Individual objective terms
    curvelength = CurveLength(banana_curves[0])
    length_target = curvelength.J()
    Jiota = QuadraticPenalty(surface_iota_terms[-1], iota_target)
    JnonQSRatio = average_surface_objectives(nonQSs)
    JBoozerResidual = average_surface_objectives(brs)
    JCurveLength = QuadraticPenalty(curvelength, length_target, 'max')
    JCurveCurve = CurveCurveDistance(curves, CC_DIST)
    JCurveSurface = CurveSurfaceDistance(curves, outer_surface_data['boozer_surface'].surface, CS_DIST)
    JSurfSurf = SurfaceSurfaceDistance(outer_surface_data['boozer_surface'].surface, VV, SS_DIST) if len(surface_data) == 1 else None
    JCurvature = LpCurveCurvature(banana_curves[0], 2, CURVATURE_THRESHOLD)

    # Combined objective function
    JF = build_total_objective(
        JnonQSRatio,
        RES_WEIGHT,
        JBoozerResidual,
        IOTAS_WEIGHT,
        Jiota,
        LENGTH_WEIGHT,
        JCurveLength,
        CC_WEIGHT,
        JCurveCurve,
        CS_WEIGHT,
        JCurveSurface,
        CURVATURE_WEIGHT,
        JCurvature,
        SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
        JSurfSurf=JSurfSurf,
    )

    # Extract degrees of freedom
    dofs = JF.x
    if CONSTRAINT_METHOD == "alm":
        ALM_MULTIPLIERS = np.zeros(4 if JSurfSurf is not None else 3, dtype=float)
        ALM_PENALTY = args.alm_penalty_init

    # ==============================================================================
    # INITIALIZE OPTIMIZATION STATE
    # ==============================================================================
    # Initialize run_dict after JF and boozer_surface are ready
    initial_search_surface_weights = build_surface_search_weights(
        len(surface_data),
        0,
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
    )
    initial_search_eval = evaluate_search_objective(initial_search_surface_weights)
    initial_search_gate = build_surface_search_gate(
        len(surface_data),
        0,
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
        SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        SS_DIST if len(surface_data) > 1 else 0.0,
    )
    run_dict = {
        'surface_state': snapshot_surface_states(surface_data),
        'J': initial_search_eval['total'],
        'dJ': initial_search_eval['grad'].copy(),
        'search_eval': initial_search_eval,
        'it': 1,
        'accepted_iterations': 0,
        'lscount': 0,
        'x_prev': dofs.copy(),
        'accepted_x': dofs.copy(),
        'intersecting': any(entry["boozer_surface"].surface.is_self_intersecting() for entry in surface_data),
        'surface_status': evaluate_surface_stack(
            surface_data,
            vessel_surface=VV if len(surface_data) > 1 else None,
            surface_gap_threshold=SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
            vessel_gap_threshold=SS_DIST if len(surface_data) > 1 else 0.0,
            enforce_nesting=True,
        ),
        'search_surface_status': evaluate_surface_stack(
            surface_data,
            vessel_surface=VV if len(surface_data) > 1 else None,
            surface_gap_threshold=initial_search_gate['surface_gap_threshold'],
            vessel_gap_threshold=initial_search_gate['vessel_gap_threshold'],
            enforce_nesting=initial_search_gate['enforce_nesting'],
        ),
        'topology_gate_status': final_topology_gate_for_results(
            args.init_only,
            len(surface_data),
            outer_surface_data['boozer_surface'].surface,
            bs,
        ),
    }

    # ==============================================================================
    # RUN OPTIMIZATION
    # ==============================================================================
    # Get convergence tolerances for current mpol
    ftol = args.ftol
    gtol = args.gtol

    basin_hop_count = None
    basin_minimization_failures = None
    termination_message = None
    optimizer_success = None
    phase1_iterations = None
    phase1_termination_message = None
    phase1_success = None
    alm_result = None
    if CONSTRAINT_METHOD == "alm" and args.num_surfaces != 1:
        raise ValueError("--constraint-method=alm currently requires --num-surfaces=1")
    if args.init_only:
        res_nit = 0
        final_volume = initial_volume
        final_iota = initial_iota
        final_max_curvature = initial_max_curvature
        fieldError = initial_field_error
        termination_message = "init_only"
        optimizer_success = True
        print("Skipping single-stage optimizer because --init-only was provided.")
    elif CONSTRAINT_METHOD == "alm":
        if args.basin_hops > 0:
            raise ValueError("--basin-hops is not supported with --constraint-method=alm")
        alm_settings = build_single_stage_alm_settings(args)

        def evaluate_problem(inner_x, multipliers, penalty):
            set_alm_runtime_state(multipliers, penalty)
            return evaluate_search_step(inner_x)

        def outer_state_callback(outer_iteration, multipliers, penalty):
            set_alm_runtime_state(multipliers, penalty)
            run_dict["alm_outer_iteration"] = int(outer_iteration)
            print(
                f"[ALM] outer_iteration={outer_iteration}, "
                f"multipliers={np.asarray(multipliers, dtype=float).tolist()}, "
                f"penalty={float(penalty):.3e}"
            )

        set_alm_runtime_state(
            np.zeros(4 if JSurfSurf is not None else 3, dtype=float),
            args.alm_penalty_init,
        )
        res = minimize_alm(
            dofs,
            list(SINGLE_STAGE_ALM_CONSTRAINT_NAMES[:3])
            + ([SINGLE_STAGE_ALM_CONSTRAINT_NAMES[3]] if JSurfSurf is not None else []),
            evaluate_problem,
            alm_settings,
            {
                "maxiter": MAXITER,
                "maxcor": args.maxcor,
                "ftol": ftol,
                "gtol": gtol,
            },
            accepted_callback=callback,
            outer_state_callback=outer_state_callback,
        )
        alm_result = res
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        print(res.message)
    elif args.basin_hops > 0:
        # Basin-hopping: perturb DOFs and re-run L-BFGS-B multiple times, keep best
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'jac': True,
            'callback': callback,
            'options': {'maxiter': MAXITER, 'maxcor': args.maxcor, 'ftol': ftol, 'gtol': gtol},
        }
        rng = np.random.RandomState(rng_seed)
        print(f"Basin-hopping with {args.basin_hops} hops, stepsize={args.basin_stepsize}, seed={rng_seed}")
        res = basinhopping(
            fun, dofs,
            minimizer_kwargs=minimizer_kwargs,
            niter=args.basin_hops,
            stepsize=args.basin_stepsize,
            seed=rng,
            disp=True,
        )
        basin_hop_count = res.nit if hasattr(res, 'nit') else None
        basin_minimization_failures = res.minimization_failures if hasattr(res, 'minimization_failures') else None
        if hasattr(res, 'lowest_optimization_result') and hasattr(res.lowest_optimization_result, 'nit'):
            res_nit = res.lowest_optimization_result.nit
        else:
            res_nit = basin_hop_count if basin_hop_count is not None else 0
        if hasattr(res, 'lowest_optimization_result'):
            termination_message = str(getattr(res.lowest_optimization_result, 'message', 'basinhopping_complete'))
            optimizer_success = bool(getattr(res.lowest_optimization_result, 'success', True))
        else:
            termination_message = str(getattr(res, 'message', 'basinhopping_complete'))
            optimizer_success = True
        print(f"Basin-hopping complete. Best fun={res.fun:.6e}, hops={args.basin_hops}, seed={rng_seed}")
    else:
        phase1_maxiter = 0
        if (
            args.num_surfaces > 1
            and args.multisurface_initial_step_maxiter > 0
            and args.multisurface_initial_step_scale < 1.0
        ):
            phase1_maxiter = min(MAXITER, args.multisurface_initial_step_maxiter)
            print(
                "Running scaled multisurface continuation phase with "
                f"step_scale={args.multisurface_initial_step_scale} and maxiter={phase1_maxiter}"
            )
            scaled_fun, scaled_callback = build_scaled_outer_problem(
                fun,
                callback,
                dofs.copy(),
                args.multisurface_initial_step_scale,
            )
            phase1_result = minimize(
                scaled_fun,
                np.zeros_like(dofs),
                jac=True,
                method='L-BFGS-B',
                callback=scaled_callback,
                options={'maxiter': phase1_maxiter, 'maxcor': args.maxcor, 'ftol': ftol, 'gtol': gtol},
            )
            phase1_iterations = phase1_result.nit
            phase1_termination_message = str(phase1_result.message)
            phase1_success = bool(phase1_result.success)
            print(phase1_termination_message)
            dofs = run_dict['accepted_x'].copy()
            run_dict['x_prev'] = dofs.copy()

        remaining_maxiter = max(MAXITER - (phase1_iterations or 0), 0)
        if remaining_maxiter > 0:
            res = minimize(
                fun,
                dofs,
                jac=True,
                method='L-BFGS-B',
                callback=callback,
                options={'maxiter': remaining_maxiter, 'maxcor': args.maxcor, 'ftol': ftol, 'gtol': gtol},
            )
            res_nit = (phase1_iterations or 0) + res.nit
            termination_message = str(res.message)
            optimizer_success = bool(res.success)
            if phase1_termination_message is not None:
                termination_message = f"phase1={phase1_termination_message}; phase2={termination_message}"
            print(res.message)
        else:
            res = SimpleNamespace(
                x=dofs.copy(),
                nit=phase1_iterations or 0,
                message=phase1_termination_message or "phase1_only",
                success=bool(phase1_success),
            )
            res_nit = res.nit
            termination_message = str(res.message)
            optimizer_success = bool(res.success)
            print(termination_message)

    if alm_result is not None:
        set_alm_runtime_state(alm_result.multipliers, alm_result.penalty)

    # ==============================================================================
    # SAVE OPTIMIZED STATE
    # ==============================================================================
    final_hardware_snapshot = None
    if not args.init_only:
        # Re-solve the surface stack at the reported optimizer endpoint so saved
        # artifacts reflect the actual final coil DOFs rather than the last
        # callback-accepted surface state.
        finalize_surface_stack(
            res.x,
            JF,
            surface_data,
            run_dict,
            vessel_surface=VV if len(surface_data) > 1 else None,
            surface_gap_threshold=SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
            vessel_gap_threshold=SS_DIST if len(surface_data) > 1 else 0.0,
        )

        full_objective_eval = evaluate_search_objective(np.ones(len(surface_data)))
        run_dict['J'] = full_objective_eval['total']
        run_dict['dJ'] = full_objective_eval['grad'].copy()
        run_dict['full_eval'] = full_objective_eval
        run_dict['base_eval'] = evaluate_base_objective(
            np.ones(len(surface_data)),
            nonQSs,
            brs,
            RES_WEIGHT,
            Jiota,
            IOTAS_WEIGHT,
            JCurveLength,
            LENGTH_WEIGHT,
        )

        # Save optimized coil configurations
        curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True)
        bs.save(OUT_DIR_ITER + "/biot_savart_opt.json")

        save_surface_artifacts(surface_data, bs, OUT_DIR_ITER, "surf_opt", also_write_outer_legacy=True)

        final_volume = outer_surface_data['boozer_surface'].surface.volume()
        final_iota = Iotas(outer_surface_data['boozer_surface']).J()
        final_hardware_snapshot = evaluate_single_stage_hardware_snapshot(
            JCurveCurve,
            CC_DIST,
            JCurveSurface,
            CS_DIST,
            JSurfSurf,
            run_dict["surface_status"],
            SS_DIST,
            banana_curve,
            CURVATURE_THRESHOLD,
            outer_surface_data["boozer_surface"].surface,
            VV,
        )
        final_max_curvature = final_hardware_snapshot["max_curvature"]
        final_surface_volumes = [entry["boozer_surface"].surface.volume() for entry in surface_data]
        final_surface_iotas = [Iotas(entry["boozer_surface"]).J() for entry in surface_data]
        print(f"Volume: {final_volume}")
        print(f"Iota: {final_iota}")
        print(f"Max Curvature: {final_max_curvature}")

        # Generate final diagnostic plots
        fieldError = normPlot(outer_surface_data['boozer_surface'].surface, bs, OUT_DIR_ITER + "/NormPlotOptimized")
        try:
            cross_section_plot(surf_coils, outer_surface_data['boozer_surface'].surface, banana_curve, OUT_DIR_ITER + "/CrossSectionOptimized", hbt, VV)
        except Exception as e:
            print(f"WARNING: CrossSectionOptimized plot failed (surface may fold at high mpol): {e}")
    else:
        final_surface_volumes = initial_surface_volumes
        final_surface_iotas = initial_surface_iotas
        run_dict['full_eval'] = None
        run_dict['base_eval'] = None
        run_dict['J'] = None
        run_dict['dJ'] = None

    final_topology_status = final_topology_gate_for_results(
        args.init_only,
        len(surface_data),
        outer_surface_data['boozer_surface'].surface,
        bs,
    )
    objective_j = float(run_dict['J']) if run_dict['J'] is not None else None
    base_objective_j = None if run_dict['base_eval'] is None else float(run_dict['base_eval']['total'])
    search_objective_j = float(run_dict['search_eval']['total'])
    final_search_surface_weights = run_dict['search_eval']['surface_weights'].tolist()
    if final_hardware_snapshot is None:
        final_hardware_snapshot = evaluate_single_stage_hardware_snapshot(
            JCurveCurve,
            CC_DIST,
            JCurveSurface,
            CS_DIST,
            JSurfSurf,
            run_dict["surface_status"],
            SS_DIST,
            banana_curve,
            CURVATURE_THRESHOLD,
            outer_surface_data["boozer_surface"].surface,
            VV,
        )
    coil_length = float(curvelength.J())
    curve_curve_min_dist = final_hardware_snapshot["curve_curve_min_dist"]
    curve_surface_min_dist = final_hardware_snapshot["curve_surface_min_dist"]
    surface_vessel_min_dist = final_hardware_snapshot["surface_vessel_min_dist"]
    nonqs_ratio = None if args.init_only else float(JnonQSRatio.J())
    boozer_residual = None if args.init_only else float(JBoozerResidual.J())
    final_hardware_status = final_hardware_snapshot["status"]
    if not final_hardware_status["success"]:
        optimizer_success = False
        if termination_message:
            termination_message = f"{termination_message}; hardware_constraints_failed"
        else:
            termination_message = "hardware_constraints_failed"

    # Save the results of optimization to a separate file
    results = {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "STAGE2_SOURCE": args.stage2_source,
        "STAGE2_BS_PATH": stage2_bs_path,
        "STAGE2_RESULTS_PATH": stage2_results_path,
        "STAGE2_SEED_MAJOR_RADIUS": R0,
        "STAGE2_SEED_TOROIDAL_FLUX": s,
        "STAGE2_SEED_BANANA_SURF_RADIUS": float(stage2_results["banana_surf_radius"]),
        "STAGE2_SEED_TF_CURRENT_A": stage2_tf_current_A,
        "STAGE2_SEED_ORDER": order,
        "STAGE2_TF_CURRENT_A": stage2_tf_current_A,
        "STAGE2_TF_CURRENT_SUM_ABS_A": tf_current_sum_abs_A,
        "mpol": mpol,
        "ntor": ntor,
        "nphi": nphi,
        "ntheta": ntheta,
        "NUM_SURFACES": args.num_surfaces,
        "INNER_SURFACE_RATIO": args.inner_surface_ratio,
        "SURFACE_GAP_THRESHOLD": SURFACE_GAP_THRESHOLD,
        "MULTISURFACE_RAMP_ITERATIONS": MULTISURFACE_RAMP_ITERATIONS,
        "INNER_SURFACE_INITIAL_WEIGHT": INNER_SURFACE_INITIAL_WEIGHT,
        "MULTISURFACE_INITIAL_STEP_SCALE": args.multisurface_initial_step_scale,
        "MULTISURFACE_INITIAL_STEP_MAXITER": args.multisurface_initial_step_maxiter,
        "TOPOLOGY_GATE_FIELDLINES": TOPOLOGY_GATE_FIELDLINES,
        "TOPOLOGY_GATE_TMAX": TOPOLOGY_GATE_TMAX,
        "TOPOLOGY_GATE_TOL": TOPOLOGY_GATE_TOL,
        "TOPOLOGY_GATE_SURVIVAL_THRESHOLD": TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
        "TOPOLOGY_GATE_PENALTY_SCALE": TOPOLOGY_GATE_PENALTY_SCALE,
        "boozer_stage": stage,
        "CONSTRAINT_WEIGHT": CONSTRAINT_WEIGHT,
        "CONSTRAINT_METHOD": CONSTRAINT_METHOD,
        "CC_DIST": CC_DIST,
        "CC_WEIGHT": CC_WEIGHT,
        "CS_DIST": CS_DIST,
        "CS_WEIGHT": CS_WEIGHT,
        "SS_DIST": SS_DIST,
        "SURF_DIST_WEIGHT": SURF_DIST_WEIGHT,
        "CURVATURE_WEIGHT": CURVATURE_WEIGHT,
        "CURVATURE_THRESHOLD": CURVATURE_THRESHOLD,
        "LENGTH_WEIGHT": LENGTH_WEIGHT,
        "RES_WEIGHT": RES_WEIGHT,
        "IOTAS_WEIGHT": IOTAS_WEIGHT,
        "MAJOR_RADIUS": R0,
        "TOROIDAL_FLUX": s,
        "banana_surf_radius": banana_surf_radius,
        "order": order,
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "iterations": res_nit,
        "FTOL": ftol,
        "GTOL": gtol,
        "ALM_MAX_OUTER_ITERS": args.alm_max_outer_iters if CONSTRAINT_METHOD == "alm" else None,
        "ALM_OUTER_ITERATIONS": getattr(alm_result, "outer_iterations", None),
        "ALM_PENALTY_INIT": args.alm_penalty_init if CONSTRAINT_METHOD == "alm" else None,
        "ALM_PENALTY_SCALE": args.alm_penalty_scale if CONSTRAINT_METHOD == "alm" else None,
        "ALM_FEAS_TOL": args.alm_feas_tol if CONSTRAINT_METHOD == "alm" else None,
        "ALM_STATIONARITY_TOL": args.alm_stationarity_tol if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_INIT": args.alm_trust_radius_init if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_MIN": args.alm_trust_radius_min if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_SHRINK": args.alm_trust_radius_shrink if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_GROW": args.alm_trust_radius_grow if CONSTRAINT_METHOD == "alm" else None,
        "ALM_MAX_INNER_ATTEMPTS": args.alm_max_inner_attempts if CONSTRAINT_METHOD == "alm" else None,
        "ALM_MAX_SUBPROBLEM_CONTINUATIONS": args.alm_max_subproblem_continuations if CONSTRAINT_METHOD == "alm" else None,
        "ALM_DISTANCE_SMOOTHING": args.alm_distance_smoothing if CONSTRAINT_METHOD == "alm" else None,
        "ALM_CURVATURE_SMOOTHING": args.alm_curvature_smoothing if CONSTRAINT_METHOD == "alm" else None,
        "ALM_FINAL_PENALTY": getattr(alm_result, "penalty", None),
        "ALM_FINAL_MULTIPLIERS": getattr(alm_result, "multipliers", None),
        "ALM_FINAL_CONSTRAINT_VALUES": getattr(alm_result, "constraint_values", None),
        "ALM_HISTORY": getattr(alm_result, "history", None),
        "TERMINATION_MESSAGE": termination_message,
        "OPTIMIZER_SUCCESS": optimizer_success,
        "CHECKPOINT_EVERY": CHECKPOINT_EVERY,
        "TOPOLOGY_SCORER_EVERY": TOPOLOGY_SCORER_EVERY,
        "TOPOLOGY_SCORER_NFIELDLINES": TOPOLOGY_SCORER_NFIELDLINES,
        "TOPOLOGY_SCORER_TMAX": TOPOLOGY_SCORER_TMAX,
        "CONFINEMENT_OBJECTIVE_WEIGHT": CONFINEMENT_OBJECTIVE_WEIGHT,
        "CONFINEMENT_SURROGATE_WORST_K": CONFINEMENT_SURROGATE_WORST_K,
        "CONFINEMENT_SURROGATE_EARLY_THRESHOLD": CONFINEMENT_SURROGATE_EARLY_THRESHOLD,
        "CONFINEMENT_SURROGATE_MEAN_WEIGHT": CONFINEMENT_SURROGATE_MEAN_WEIGHT,
        "CONFINEMENT_SURROGATE_WORST_WEIGHT": CONFINEMENT_SURROGATE_WORST_WEIGHT,
        "CONFINEMENT_SURROGATE_EARLY_WEIGHT": CONFINEMENT_SURROGATE_EARLY_WEIGHT,
        "basin_hops": args.basin_hops,
        "basin_stepsize": args.basin_stepsize if args.basin_hops > 0 else None,
        "basin_seed": rng_seed if args.basin_hops > 0 else None,
        "basin_iterations": basin_hop_count,
        "basin_minimization_failures": basin_minimization_failures,
        "PHASE1_ITERATIONS": phase1_iterations,
        "PHASE1_TERMINATION_MESSAGE": phase1_termination_message,
        "PHASE1_SUCCESS": phase1_success,
        "NFP": int(banana_surf_nfp),
        "FINAL_TOPOLOGY_GATE_EVALUATED": final_topology_status["evaluated"],
        "FINAL_TOPOLOGY_GATE_SUCCESS": final_topology_status["success"],
        "FINAL_TOPOLOGY_SURVIVED_LINES": final_topology_status["survived_lines"],
        "FINAL_TOPOLOGY_SURVIVAL_FRACTION": final_topology_status["survival_fraction"],
        "FINAL_TOPOLOGY_FIRST_EXIT_TIME": final_topology_status["first_exit_time"],
        "FINAL_TOPOLOGY_FIRST_EXIT_ANGLE": final_topology_status["first_exit_angle"],
        "FINAL_TOPOLOGY_FIRST_EXIT_REASON": final_topology_status["first_exit_reason"],
        "FINAL_TOPOLOGY_STOP_REASON_COUNTS": final_topology_status["stop_reason_counts"],
        "TARGET_VOLUME": float(vol_target),
        "TARGET_IOTA": float(iota_target),
        "PLASMA_CURRENT_A": float(plasma_current_A),
        "PLASMA_CURRENT_INPUT_SOURCE": plasma_current_input_source,
        "PLASMA_CURRENT_SURROGATE_SCOPE": "shared_all_surfaces" if args.num_surfaces > 1 else "single_surface",
        "FINITE_CURRENT_MODE": finite_current_mode,
        "BOOZER_I": float(boozer_I),
        "FINAL_VOLUME": float(final_volume),
        "FINAL_IOTA": float(final_iota),
        "FIELD_ERROR": float(fieldError),
        "OBJECTIVE_J": objective_j,
        "BASE_OBJECTIVE_J": base_objective_j,
        "SEARCH_OBJECTIVE_J": search_objective_j,
        "FINAL_SEARCH_SURFACE_WEIGHTS": final_search_surface_weights,
        "SELF_INTERSECTING": run_dict['intersecting'],
        "MAX_CURVATURE": float(final_max_curvature),
        "COIL_LENGTH": coil_length,
        "CURVE_CURVE_MIN_DIST": curve_curve_min_dist,
        "CURVE_SURFACE_MIN_DIST": curve_surface_min_dist,
        "SURFACE_VESSEL_MIN_DIST": surface_vessel_min_dist,
        "HARDWARE_CONSTRAINTS_OK": final_hardware_status["success"],
        "HARDWARE_CONSTRAINT_VIOLATIONS": final_hardware_status["violations"],
        "NONQS_RATIO": nonqs_ratio,
        "BOOZER_RESIDUAL": boozer_residual,
        "INITIAL_VOLUME": float(initial_volume),
        "INITIAL_IOTA": float(initial_iota),
        "INITIAL_FIELD_ERROR": float(initial_field_error),
        "INITIAL_MAX_CURVATURE": float(initial_max_curvature),
        "BEST_TOPOLOGY_ACCEPTED_ITERATION": run_dict.get("best_topology", {}).get("accepted_iteration"),
        "BEST_TOPOLOGY_CONFINEMENT_SCORE": run_dict.get("best_topology", {}).get("confinement_score"),
        "BEST_TOPOLOGY_CONFINEMENT_LOSS": run_dict.get("best_topology", {}).get("confinement_loss"),
        "BEST_CONFINEMENT_OBJECTIVE_ACCEPTED_ITERATION": run_dict.get("best_confinement_objective", {}).get("accepted_iteration"),
        "BEST_CONFINEMENT_OBJECTIVE_TOTAL": run_dict.get("best_confinement_objective", {}).get("checkpoint_objective_total"),
        "BEST_CONFINEMENT_OBJECTIVE_PROXY_J": run_dict.get("best_confinement_objective", {}).get("J"),
        "BEST_CONFINEMENT_OBJECTIVE_LOSS": run_dict.get("best_confinement_objective", {}).get("confinement_loss"),
    }
    results.update(
        collect_surface_run_metadata(
            surface_data,
            run_dict['surface_status'],
            initial_surface_volumes,
            initial_surface_iotas,
            final_surface_volumes,
            final_surface_iotas,
        )
    )
    with open(os.path.join(OUT_DIR_ITER, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)
