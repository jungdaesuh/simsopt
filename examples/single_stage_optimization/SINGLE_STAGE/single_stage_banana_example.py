import argparse
import hashlib
import os
import io
import json
from types import SimpleNamespace
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import minimize, basinhopping
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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
    augmented_objective,
    minimize_alm,
    validate_alm_cli_args,
)
from plotting_utils import norm_field_plot, cross_section_plot
from topology_scorer import midplane_seed_radii as _midplane_seed_radii, score_topology, stop_reason_label as _topology_stop_reason, toroidal_angle as _topology_toroidal_angle
SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = DATABASE_EQUILIBRIA_DIR if os.path.isdir(DATABASE_EQUILIBRIA_DIR) else os.path.join(EXAMPLE_ROOT, "equilibria")
DEFAULT_LOCAL_STAGE2_ROOT = os.path.join(EXAMPLE_ROOT, "STAGE_2")
DEFAULT_DATABASE_STAGE2_ROOT = os.path.join(REPO_ROOT, "DATABASE", "COIL_OPTIMIZATION", "outputs")
DEFAULT_SINGLE_STAGE_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
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
        "order": 2,
    },
}


def format_compact_float(value):
    return f"{value:g}"


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


def format_local_stage2_seed_dir(major_radius, toroidal_flux, length_weight, cc_weight, cc_threshold, curvature_weight, curvature_threshold, banana_surf_radius, order):
    return (
        f"R0={format_compact_float(major_radius)}"
        f"-s={format_compact_float(toroidal_flux)}"
        f"-LW={format_compact_float(length_weight)}"
        f"-CCW={format_compact_float(cc_weight)}"
        f"-CCT={format_compact_float(cc_threshold)}"
        f"-CW={format_compact_float(curvature_weight)}"
        f"-CT={format_compact_float(curvature_threshold)}"
        f"-SR={banana_surf_radius:0.3f}"
        f"-Order={order}"
    )


def format_database_stage2_seed_dir(major_radius, toroidal_flux, length_weight, cc_weight, curvature_weight, banana_surf_radius, order):
    return (
        f"MR={format_compact_float(major_radius)}"
        f"-TF={format_compact_float(toroidal_flux)}"
        f"-LW={format_compact_float(length_weight)}"
        f"-CCW={format_compact_float(cc_weight)}"
        f"-CW={format_compact_float(curvature_weight)}"
        f"-SR={format_compact_float(banana_surf_radius)}"
        f"-Order={order}"
    )


def build_stage2_bs_path(args):
    if args.stage2_bs_path:
        return args.stage2_bs_path

    if args.stage2_source == "database":
        seed_dir = format_database_stage2_seed_dir(
            args.stage2_seed_major_radius,
            args.stage2_seed_toroidal_flux,
            args.stage2_seed_length_weight,
            args.stage2_seed_cc_weight,
            args.stage2_seed_curvature_weight,
            args.stage2_seed_banana_surf_radius,
            args.stage2_seed_order,
        )
        return os.path.join(
            args.database_stage2_root,
            f"outputs-{args.plasma_surf_filename}",
            seed_dir,
            "biot_savart_opt.json",
        )

    seed_dir = format_local_stage2_seed_dir(
        args.stage2_seed_major_radius,
        args.stage2_seed_toroidal_flux,
        args.stage2_seed_length_weight,
        args.stage2_seed_cc_weight,
        args.stage2_seed_cc_threshold,
        args.stage2_seed_curvature_weight,
        args.stage2_seed_curvature_threshold,
        args.stage2_seed_banana_surf_radius,
        args.stage2_seed_order,
    )
    candidate = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        seed_dir,
        "biot_savart_opt.json",
    )
    if os.path.exists(candidate):
        return candidate

    # Fallback: legacy directory format without CCT/CT segments
    legacy_dir = (
        f"R0={format_compact_float(args.stage2_seed_major_radius)}"
        f"-s={format_compact_float(args.stage2_seed_toroidal_flux)}"
        f"-LW={format_compact_float(args.stage2_seed_length_weight)}"
        f"-CCW={format_compact_float(args.stage2_seed_cc_weight)}"
        f"-CW={format_compact_float(args.stage2_seed_curvature_weight)}"
        f"-SR={args.stage2_seed_banana_surf_radius:0.3f}"
        f"-Order={args.stage2_seed_order}"
    )
    legacy = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        legacy_dir,
        "biot_savart_opt.json",
    )
    if os.path.exists(legacy):
        print(f"Note: found legacy Stage 2 output at {legacy_dir}/ (missing CCT/CT segments)")
        return legacy

    # Fallback: basin-hopped directories have extra -BH=...-BS=...-BSeed=... suffix
    # Glob for any directory starting with the base seed_dir name
    from glob import glob as _glob
    parent = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
    )
    pattern = os.path.join(parent, seed_dir + "-BH=*", "biot_savart_opt.json")
    matches = sorted(_glob(pattern))
    if len(matches) == 1:
        print(f"Note: found unique basin-hopped Stage 2 output at {os.path.dirname(matches[0])}")
        return matches[0]
    if len(matches) > 1:
        match_dirs = "\n".join(f"  - {os.path.dirname(match)}" for match in matches)
        raise FileNotFoundError(
            "Multiple basin-hopped Stage 2 outputs match the requested seed specification. "
            "Pass --stage2-bs-path explicitly to choose one.\n"
            f"Matches:\n{match_dirs}"
        )

    return candidate


def load_stage2_results(stage2_bs_path):
    stage2_results_path = os.path.join(os.path.dirname(stage2_bs_path), "results.json")
    with open(stage2_results_path, "r", encoding="utf-8") as infile:
        stage2_results = json.load(infile)
    return stage2_results_path, stage2_results


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
    parser.add_argument("--boozer-I", type=float, default=float(os.environ.get("BOOZER_I", "0.0")))
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


def scale_surface_to_major_radius(surface, major_radius):
    scale = major_radius / surface.major_radius()
    surface.set_dofs(surface.get_dofs() * scale)
    return surface


def build_surface_configs(file_loc, nphi, ntheta, seed_label, major_radius, outer_target_volume, num_surfaces, inner_surface_ratio):
    outer_reference = SurfaceRZFourier.from_wout(file_loc, range="half period", nphi=nphi, ntheta=ntheta, s=seed_label)
    outer_reference = scale_surface_to_major_radius(outer_reference, major_radius)
    configs = [{
        "name": "outer",
        "seed_label": seed_label,
        "target_volume": outer_target_volume,
        "initial_surface": outer_reference,
    }]
    if num_surfaces == 1:
        return configs

    if not (0.0 < inner_surface_ratio < 1.0):
        raise ValueError("--inner-surface-ratio must be between 0 and 1 when --num-surfaces=2")

    inner_label = seed_label * inner_surface_ratio
    inner_reference = SurfaceRZFourier.from_wout(file_loc, range="half period", nphi=nphi, ntheta=ntheta, s=inner_label)
    inner_reference = scale_surface_to_major_radius(inner_reference, major_radius)
    inner_volume_ratio = inner_reference.volume() / outer_reference.volume()
    inner_target_volume = outer_target_volume * inner_volume_ratio
    if not (0.0 < inner_target_volume < outer_target_volume):
        raise RuntimeError("Derived inner target volume is not strictly inside the outer target volume")

    return [
        {
            "name": "inner",
            "seed_label": inner_label,
            "target_volume": inner_target_volume,
            "initial_surface": inner_reference,
        },
        configs[0],
    ]


def build_hbt_reference_surfaces(nfp, banana_surf_radius):
    vv = SurfaceRZFourier(nfp=nfp, stellsym=True)
    vv.set_rc(0, 0, 0.976)
    vv.set_rc(1, 0, 0.222)
    vv.set_zs(1, 0, 0.222)

    hbt = SurfaceRZFourier(nfp=nfp, stellsym=True)
    hbt.set_rc(0, 0, 0.9115)    # R0 of LCFS semi-circle center
    hbt.set_rc(1, 0, 0.1605)    # Minor radius (thick metal walls)
    hbt.set_zs(1, 0, 0.152)    # Z extent = ±0.152 m (flat top/bottom)

    surf_coils = SurfaceRZFourier(nfp=nfp, stellsym=True)
    surf_coils.set_rc(0, 0, 0.976)
    surf_coils.set_rc(1, 0, banana_surf_radius)
    surf_coils.set_zs(1, 0, banana_surf_radius)
    return vv, hbt, surf_coils


def surface_pointcloud_gap(surface_a, surface_b):
    points_a = surface_a.gamma().reshape((-1, 3))
    points_b = surface_b.gamma().reshape((-1, 3))
    nearest, _ = cKDTree(points_b).query(points_a, k=1)
    return float(np.min(nearest))


def surface_cross_section_rz(surface, phi, theta_samples=128):
    cross_section = surface.cross_section(phi, thetas=theta_samples)
    rz = np.zeros((cross_section.shape[0], 2))
    rz[:, 0] = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    rz[:, 1] = cross_section[:, 2]
    return rz


def planar_segments_intersect(p1, p2, q1, q2, tol=1e-12):
    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - tol <= c[0] <= max(a[0], b[0]) + tol
            and min(a[1], b[1]) - tol <= c[1] <= max(a[1], b[1]) + tol
        )

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (o3 > tol and o4 < -tol or o3 < -tol and o4 > tol):
        return True

    if abs(o1) <= tol and on_segment(p1, p2, q1):
        return True
    if abs(o2) <= tol and on_segment(p1, p2, q2):
        return True
    if abs(o3) <= tol and on_segment(q1, q2, p1):
        return True
    if abs(o4) <= tol and on_segment(q1, q2, p2):
        return True
    return False


def planar_polygons_intersect(poly_a, poly_b):
    edges_a = list(zip(poly_a, np.roll(poly_a, -1, axis=0)))
    edges_b = list(zip(poly_b, np.roll(poly_b, -1, axis=0)))
    return any(planar_segments_intersect(a0, a1, b0, b1) for a0, a1 in edges_a for b0, b1 in edges_b)


def cross_sections_are_nested(inner_surface, outer_surface, nphi_slices=9, theta_samples=128):
    bad_phis = []
    for phi in np.linspace(0.0, 1.0 / outer_surface.nfp, nphi_slices, endpoint=False):
        inner_rz = surface_cross_section_rz(inner_surface, phi, theta_samples=theta_samples)
        outer_rz = surface_cross_section_rz(outer_surface, phi, theta_samples=theta_samples)
        outer_path = MplPath(np.vstack([outer_rz, outer_rz[0]]))
        inner_inside_outer = bool(np.all(outer_path.contains_points(inner_rz, radius=1e-10)))
        disjoint = not planar_polygons_intersect(inner_rz, outer_rz)
        if not (inner_inside_outer and disjoint):
            bad_phis.append(float(phi))
    return len(bad_phis) == 0, bad_phis


def evaluate_surface_stack(
    surface_data,
    vessel_surface=None,
    surface_gap_threshold=0.0,
    vessel_gap_threshold=0.0,
    enforce_nesting=True,
):
    volumes = [entry["boozer_surface"].surface.volume() for entry in surface_data]
    iotas = [entry["boozer_surface"].res["iota"] for entry in surface_data]
    solve_success = [bool(entry["boozer_surface"].res["success"]) for entry in surface_data]
    def _safe_is_self_intersecting(surface):
        try:
            return bool(surface.is_self_intersecting())
        except Exception:
            return True  # surface that folds is self-intersecting
    self_intersections = [_safe_is_self_intersecting(entry["boozer_surface"].surface) for entry in surface_data]
    adjacent_gaps = []
    for left, right in zip(surface_data[:-1], surface_data[1:]):
        adjacent_gaps.append(surface_pointcloud_gap(left["boozer_surface"].surface, right["boozer_surface"].surface))
    outer_vessel_gap = None
    if vessel_surface is not None:
        outer_vessel_gap = surface_pointcloud_gap(surface_data[-1]["boozer_surface"].surface, vessel_surface)

    volumes_ordered = np.all(np.diff(volumes) > 0.0) if len(volumes) > 1 else True
    gap_ok = all(gap > surface_gap_threshold for gap in adjacent_gaps)
    vessel_gap_ok = outer_vessel_gap is None or outer_vessel_gap > vessel_gap_threshold
    nesting_ok = True
    bad_nesting_phis = []
    if (
        enforce_nesting
        and len(surface_data) > 1
        and all(hasattr(entry["boozer_surface"].surface, "cross_section") for entry in surface_data)
    ):
        try:
            nesting_ok, bad_nesting_phis = cross_sections_are_nested(
                surface_data[0]["boozer_surface"].surface,
                surface_data[-1]["boozer_surface"].surface,
            )
        except Exception:
            nesting_ok = False  # surface that folds is not properly nested
    success = all(solve_success) and not any(self_intersections) and volumes_ordered and gap_ok and vessel_gap_ok and nesting_ok
    return {
        "success": success,
        "solve_success": solve_success,
        "self_intersections": self_intersections,
        "volumes": volumes,
        "iotas": iotas,
        "adjacent_gaps": adjacent_gaps,
        "outer_vessel_gap": outer_vessel_gap,
        "volumes_ordered": volumes_ordered,
        "gap_ok": gap_ok,
        "vessel_gap_ok": vessel_gap_ok,
        "nesting_ok": nesting_ok,
        "bad_nesting_phis": bad_nesting_phis,
    }


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
    """Evaluate hard single-stage hardware constraints against realized geometry."""
    violations = []
    if curve_curve_min_dist < cc_dist:
        violations.append(
            f"coil_coil_min_dist {curve_curve_min_dist:.6f} below threshold {cc_dist:.6f}"
        )
    if curve_surface_min_dist < cs_dist:
        violations.append(
            f"coil_surface_min_dist {curve_surface_min_dist:.6f} below threshold {cs_dist:.6f}"
        )
    if surface_vessel_min_dist < ss_dist:
        violations.append(
            f"surface_vessel_min_dist {surface_vessel_min_dist:.6f} below threshold {ss_dist:.6f}"
        )
    if max_curvature > curvature_threshold:
        violations.append(
            f"max_curvature {max_curvature:.6f} exceeds threshold {curvature_threshold:.6f}"
        )
    return {
        "success": len(violations) == 0,
        "violations": violations,
        "curve_curve_min_dist": float(curve_curve_min_dist),
        "cc_dist": float(cc_dist),
        "curve_surface_min_dist": float(curve_surface_min_dist),
        "cs_dist": float(cs_dist),
        "surface_vessel_min_dist": float(surface_vessel_min_dist),
        "ss_dist": float(ss_dist),
        "max_curvature": float(max_curvature),
        "curvature_threshold": float(curvature_threshold),
    }


def compute_single_stage_surface_vessel_min_dist(
    surface_vessel_distance_obj,
    surface_status,
    outer_surface=None,
    vessel_surface=None,
):
    if surface_vessel_distance_obj is not None and hasattr(surface_vessel_distance_obj, "shortest_distance"):
        return float(surface_vessel_distance_obj.shortest_distance())
    outer_vessel_gap = surface_status.get("outer_vessel_gap")
    if outer_vessel_gap is not None:
        return float(outer_vessel_gap)
    if outer_surface is None or vessel_surface is None:
        raise ValueError(
            "Need outer_surface and vessel_surface when no surface-vessel distance object or cached gap is available."
        )
    return float(
        np.min(
            cdist(
                outer_surface.gamma().reshape((-1, 3)),
                vessel_surface.gamma().reshape((-1, 3)),
            )
        )
    )


def snapshot_surface_states(surface_data):
    return {
        "sdofs": [entry["boozer_surface"].surface.x.copy() for entry in surface_data],
        "iota": [entry["boozer_surface"].res["iota"] for entry in surface_data],
        "G": [entry["boozer_surface"].res["G"] for entry in surface_data],
    }


def restore_surface_states(surface_data, state):
    for entry, sdofs, iota, G in zip(surface_data, state["sdofs"], state["iota"], state["G"]):
        entry["boozer_surface"].surface.x = sdofs.copy()
        entry["boozer_surface"].res["iota"] = iota
        entry["boozer_surface"].res["G"] = G


def solve_surface_stack_at_dofs(
    x,
    objective,
    surface_data,
    state,
    vessel_surface=None,
    surface_gap_threshold=0.0,
    vessel_gap_threshold=0.0,
    enforce_nesting=True,
):
    restore_surface_states(surface_data, state)
    objective.x = x
    for entry, iota, G in zip(surface_data, state["iota"], state["G"]):
        entry["boozer_surface"].run_code(iota, G)
    return evaluate_surface_stack(
        surface_data,
        vessel_surface=vessel_surface,
        surface_gap_threshold=surface_gap_threshold,
        vessel_gap_threshold=vessel_gap_threshold,
        enforce_nesting=enforce_nesting,
    )


def average_surface_objectives(objectives, weights=None):
    if len(objectives) == 0:
        raise ValueError("Need at least one surface objective to average")
    if weights is None:
        weights = np.ones(len(objectives))
    if len(objectives) != len(weights):
        raise ValueError("Number of objectives and weights must match")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be positive")

    weighted_sum = None
    for weight, objective in zip(weights, objectives):
        weighted_objective = weight * objective
        weighted_sum = weighted_objective if weighted_sum is None else weighted_sum + weighted_objective
    return (1.0 / total_weight) * weighted_sum


def continuation_inner_surface_weight(num_surfaces, accepted_iterations, ramp_iterations, initial_weight):
    if num_surfaces <= 1:
        return 1.0
    if not (0.0 <= initial_weight <= 1.0):
        raise ValueError("inner-surface initial weight must be between 0 and 1")
    if ramp_iterations <= 0:
        return 1.0
    progress = min(max(accepted_iterations, 0), ramp_iterations) / ramp_iterations
    return initial_weight + (1.0 - initial_weight) * progress


def build_surface_search_weights(num_surfaces, accepted_iterations, ramp_iterations, initial_inner_weight):
    if num_surfaces <= 0:
        raise ValueError("num_surfaces must be positive")
    weights = np.ones(num_surfaces)
    if num_surfaces > 1:
        weights[:-1] = continuation_inner_surface_weight(
            num_surfaces,
            accepted_iterations,
            ramp_iterations,
            initial_inner_weight,
        )
    return weights


def build_surface_search_gate(
    num_surfaces,
    accepted_iterations,
    ramp_iterations,
    initial_inner_weight,
    surface_gap_threshold,
    vessel_gap_threshold,
):
    if num_surfaces <= 1:
        return {
            "surface_gap_threshold": float(surface_gap_threshold),
            "vessel_gap_threshold": float(vessel_gap_threshold),
            "enforce_nesting": True,
            "gate_scale": 1.0,
        }

    gate_scale = continuation_inner_surface_weight(
        num_surfaces,
        accepted_iterations,
        ramp_iterations,
        initial_inner_weight,
    )
    return {
        "surface_gap_threshold": float(surface_gap_threshold) * gate_scale,
        "vessel_gap_threshold": float(vessel_gap_threshold) * gate_scale,
        "enforce_nesting": bool(gate_scale >= 1.0),
        "gate_scale": float(gate_scale),
    }


def build_scaled_outer_problem(base_fun, base_callback, anchor_x, step_scale):
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1]")

    def scaled_fun(z):
        x = anchor_x + step_scale * z
        J, dJ = base_fun(x)
        return J, step_scale * dJ

    def scaled_callback(z):
        base_callback(anchor_x + step_scale * z)

    return scaled_fun, scaled_callback


def evaluate_topology_gate(surface, bfield, nfieldlines, tmax, tol, survival_threshold):
    if nfieldlines <= 0:
        return disabled_topology_gate_status(tmax, tol, survival_threshold)

    cross_section = surface.cross_section(phi=0.0, thetas=512)
    r = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    z = cross_section[:, 2]
    rmin = float(np.min(r))
    rmax = float(np.max(r))
    zmax = float(np.max(np.abs(z)))
    classifier = SurfaceClassifier(surface, h=0.03, p=2)
    stopping_criteria = [
        LevelsetStoppingCriterion(classifier.dist),
        MaxZStoppingCriterion(zmax * 1.05),
        MinZStoppingCriterion(-zmax * 1.05),
        MinRStoppingCriterion(rmin * 0.95),
        MaxRStoppingCriterion(rmax * 1.05),
    ]
    stop_labels = [
        "surface_exit",
        "max_z_guardrail",
        "min_z_guardrail",
        "min_r_guardrail",
        "max_r_guardrail",
    ]
    R0 = _midplane_seed_radii(surface, nfieldlines)
    Z0 = np.zeros((nfieldlines,))
    _, fieldlines_phi_hits = compute_fieldlines(
        bfield,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=[0.0],
        stopping_criteria=stopping_criteria,
    )

    survived = 0
    earliest_exit = None
    stop_reason_counts = {label: 0 for label in stop_labels}
    for hits in fieldlines_phi_hits:
        hits = np.asarray(hits)
        if hits.size == 0:
            survived += 1
            continue
        if hits.ndim == 1:
            hits = hits[None, :]
        negative_hits = hits[hits[:, 1] < 0]
        if negative_hits.size == 0:
            survived += 1
            continue
        first_stop = negative_hits[0]
        stop_index = int(-first_stop[1]) - 1
        stop_reason = _topology_stop_reason(stop_index, stop_labels)
        stop_reason_counts.setdefault(stop_reason, 0)
        stop_reason_counts[stop_reason] += 1
        exit_time = float(first_stop[0])
        exit_angle = _topology_toroidal_angle(first_stop[2], first_stop[3])
        if earliest_exit is None or exit_time < earliest_exit["first_exit_time"]:
            earliest_exit = {
                "first_exit_time": exit_time,
                "first_exit_angle": exit_angle,
                "stop_reason": stop_reason,
            }

    survival_fraction = survived / nfieldlines
    return {
        "enabled": True,
        "success": bool(survival_fraction >= survival_threshold),
        "nfieldlines": int(nfieldlines),
        "survived_lines": int(survived),
        "survival_fraction": float(survival_fraction),
        "survival_threshold": float(survival_threshold),
        "tmax": float(tmax),
        "tol": float(tol),
        "stop_reason_counts": stop_reason_counts,
        "first_exit_time": None if earliest_exit is None else earliest_exit["first_exit_time"],
        "first_exit_angle": None if earliest_exit is None else earliest_exit["first_exit_angle"],
        "first_exit_reason": None if earliest_exit is None else earliest_exit["stop_reason"],
    }


def disabled_topology_gate_status(tmax, tol, survival_threshold):
    return {
        "enabled": False,
        "success": True,
        "nfieldlines": 0,
        "survived_lines": 0,
        "survival_fraction": 1.0,
        "survival_threshold": float(survival_threshold),
        "tmax": float(tmax),
        "tol": float(tol),
        "stop_reason_counts": {},
        "first_exit_time": None,
        "first_exit_angle": None,
        "first_exit_reason": None,
    }


def topology_gate_deficit(status):
    if not status["enabled"]:
        return 0.0
    return max(0.0, float(status["survival_threshold"]) - float(status["survival_fraction"]))


def topology_gate_rejection_increment(last_objective, status, penalty_scale):
    base_increment = max(abs(last_objective), 1.0)
    deficit = topology_gate_deficit(status)
    return base_increment * (1.0 + penalty_scale * deficit)


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


def build_run_identity_config(
    args,
    stage2_bs_path,
    stage,
    constraint_weight,
    constraint_method,
    vol_target,
    iota_target,
    boozer_I,
    banana_surf_radius,
    nphi,
    ntheta,
    rng_seed,
):
    return (
        f"{stage2_bs_path}|{stage}|{constraint_weight}|{constraint_method}|{vol_target}|{iota_target}|{boozer_I}"
        f"|{args.cc_dist}|{args.cc_weight}|{args.curvature_weight}|{args.curvature_threshold}"
        f"|{banana_surf_radius}|{nphi}|{ntheta}|{args.init_only}"
        f"|{args.basin_hops}|{args.basin_stepsize}|{rng_seed}"
        f"|{args.ftol}|{args.gtol}"
        f"|{args.alm_max_outer_iters}|{args.alm_penalty_init}|{args.alm_penalty_scale}"
        f"|{args.alm_feas_tol}|{args.alm_stationarity_tol}"
        f"|{args.num_surfaces}|{args.inner_surface_ratio}|{args.surface_gap_threshold}"
        f"|{MULTISURFACE_RAMP_ITERATIONS}|{INNER_SURFACE_INITIAL_WEIGHT}"
        f"|{args.multisurface_initial_step_scale}|{args.multisurface_initial_step_maxiter}"
        f"|{TOPOLOGY_GATE_FIELDLINES}|{TOPOLOGY_GATE_TMAX}|{TOPOLOGY_GATE_TOL}|{TOPOLOGY_GATE_SURVIVAL_THRESHOLD}"
        f"|{TOPOLOGY_SCORER_EVERY}|{TOPOLOGY_SCORER_NFIELDLINES}|{TOPOLOGY_SCORER_TMAX}"
        f"|{CONFINEMENT_OBJECTIVE_WEIGHT}|{CONFINEMENT_SURROGATE_WORST_K}"
        f"|{CONFINEMENT_SURROGATE_EARLY_THRESHOLD}|{CONFINEMENT_SURROGATE_MEAN_WEIGHT}"
        f"|{CONFINEMENT_SURROGATE_WORST_WEIGHT}|{CONFINEMENT_SURROGATE_EARLY_WEIGHT}"
    )


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
    J_QS_obj = average_surface_objectives(nonQSs, weights=surface_weights)
    J_Boozer_obj = average_surface_objectives(brs, weights=surface_weights)
    total_objective = build_total_objective(
        J_QS_obj,
        RES_WEIGHT,
        J_Boozer_obj,
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
    return {
        "total": total_objective.J(),
        "grad": total_objective.dJ(),
        "J_QS": J_QS_obj.J(),
        "dJ_QS": J_QS_obj.dJ(),
        "J_Boozer": J_Boozer_obj.J(),
        "dJ_Boozer": J_Boozer_obj.dJ(),
        "J_iota": Jiota.J(),
        "dJ_iota": Jiota.dJ(),
        "J_len": JCurveLength.J(),
        "dJ_len": JCurveLength.dJ(),
        "J_cc": JCurveCurve.J(),
        "dJ_cc": JCurveCurve.dJ(),
        "J_cs": JCurveSurface.J(),
        "dJ_cs": JCurveSurface.dJ(),
        "J_surf": JSurfSurf.J() if JSurfSurf is not None else 0.0,
        "dJ_surf": JSurfSurf.dJ() if JSurfSurf is not None else np.zeros_like(total_objective.dJ()),
        "J_curvature": JCurvature.J(),
        "dJ_curvature": JCurvature.dJ(),
        "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
    }


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
    J_QS_obj = average_surface_objectives(nonQSs, weights=surface_weights)
    J_Boozer_obj = average_surface_objectives(brs, weights=surface_weights)
    base_objective = (
        J_QS_obj
        + RES_WEIGHT * J_Boozer_obj
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
    )
    return {
        "total": base_objective.J(),
        "grad": base_objective.dJ(),
        "J_QS": J_QS_obj.J(),
        "dJ_QS": J_QS_obj.dJ(),
        "J_Boozer": J_Boozer_obj.J(),
        "dJ_Boozer": J_Boozer_obj.dJ(),
        "J_iota": Jiota.J(),
        "dJ_iota": Jiota.dJ(),
        "J_len": JCurveLength.J(),
        "dJ_len": JCurveLength.dJ(),
        "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
    }


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
    base_eval = evaluate_base_objective(
        surface_weights,
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
    )

    constraint_names = [
        "coil_coil_spacing",
        "coil_surface_spacing",
        "max_curvature",
    ]
    constraint_values = [
        float(JCurveCurve.J()),
        float(JCurveSurface.J()),
        float(JCurvature.J()),
    ]
    constraint_grads = [
        np.asarray(JCurveCurve.dJ(), dtype=float),
        np.asarray(JCurveSurface.dJ(), dtype=float),
        np.asarray(JCurvature.dJ(), dtype=float),
    ]
    if JSurfSurf is not None:
        constraint_names.append("surface_vessel_spacing")
        constraint_values.append(float(JSurfSurf.J()))
        constraint_grads.append(np.asarray(JSurfSurf.dJ(), dtype=float))

    alm_eval = augmented_objective(
        base_eval["total"],
        base_eval["grad"],
        constraint_values,
        constraint_grads,
        multipliers,
        penalty,
    )
    base_total = float(base_eval["total"])
    base_eval.update(alm_eval)
    base_eval["base_total"] = base_total
    base_eval["constraint_names"] = constraint_names
    base_eval["J_cc"] = float(JCurveCurve.J())
    base_eval["dJ_cc"] = np.asarray(JCurveCurve.dJ(), dtype=float)
    base_eval["J_cs"] = float(JCurveSurface.J())
    base_eval["dJ_cs"] = np.asarray(JCurveSurface.dJ(), dtype=float)
    base_eval["J_surf"] = 0.0 if JSurfSurf is None else float(JSurfSurf.J())
    base_eval["dJ_surf"] = (
        np.zeros_like(np.asarray(base_eval["grad"], dtype=float))
        if JSurfSurf is None
        else np.asarray(JSurfSurf.dJ(), dtype=float)
    )
    base_eval["J_curvature"] = float(JCurvature.J())
    base_eval["dJ_curvature"] = np.asarray(JCurvature.dJ(), dtype=float)
    return base_eval


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


def build_total_objective(JnonQSRatio, RES_WEIGHT, JBoozerResidual, IOTAS_WEIGHT, Jiota, LENGTH_WEIGHT, JCurveLength, CC_WEIGHT, JCurveCurve, CS_WEIGHT, JCurveSurface, CURVATURE_WEIGHT, JCurvature, SURF_DIST_WEIGHT=0.0, JSurfSurf=None):
    objective = (
        JnonQSRatio
        + RES_WEIGHT * JBoozerResidual
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
        + CC_WEIGHT * JCurveCurve
        + CS_WEIGHT * JCurveSurface
        + CURVATURE_WEIGHT * JCurvature
    )
    if JSurfSurf is not None:
        objective = objective + SURF_DIST_WEIGHT * JSurfSurf
    return objective


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


def save_surface_artifacts(surface_data, biotsavart, out_dir, stem, also_write_outer_legacy):
    outer_entry = surface_data[-1]
    for entry in surface_data:
        name = entry["name"]
        path = os.path.join(out_dir, f"{stem}_{name}")
        biotsavart.set_points(entry["boozer_surface"].surface.gamma().reshape((-1, 3)))
        unitn = entry["boozer_surface"].surface.unitnormal()
        point_data = {
            "B_N/B": np.sum(biotsavart.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]
            / np.sqrt(np.sum(biotsavart.B().reshape(unitn.shape) ** 2, axis=2))[:, :, None]
        }
        entry["boozer_surface"].surface.to_vtk(path, extra_data=point_data)
        entry["boozer_surface"].surface.save(path + ".json")

    if also_write_outer_legacy:
        legacy_path = os.path.join(out_dir, stem)
        biotsavart.set_points(outer_entry["boozer_surface"].surface.gamma().reshape((-1, 3)))
        unitn = outer_entry["boozer_surface"].surface.unitnormal()
        point_data = {
            "B_N/B": np.sum(biotsavart.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]
            / np.sqrt(np.sum(biotsavart.B().reshape(unitn.shape) ** 2, axis=2))[:, :, None]
        }
        outer_entry["boozer_surface"].surface.to_vtk(legacy_path, extra_data=point_data)
        outer_entry["boozer_surface"].surface.save(legacy_path + ".json")


def collect_surface_run_metadata(surface_data, run_status, initial_surface_volumes, initial_surface_iotas, final_surface_volumes, final_surface_iotas):
    return {
        "SURFACE_NAMES": [entry["name"] for entry in surface_data],
        "SURFACE_SEED_LABELS": [float(entry["seed_label"]) for entry in surface_data],
        "SURFACE_TARGET_VOLUMES": [float(entry["target_volume"]) for entry in surface_data],
        "FINAL_SURFACE_VOLUMES": [float(value) for value in final_surface_volumes],
        "FINAL_SURFACE_IOTAS": [float(value) for value in final_surface_iotas],
        "SURFACE_SELF_INTERSECTING": [bool(value) for value in run_status["self_intersections"]],
        "ADJACENT_SURFACE_GAPS": [float(value) for value in run_status["adjacent_gaps"]],
        "OUTER_VESSEL_GAP": None if run_status["outer_vessel_gap"] is None else float(run_status["outer_vessel_gap"]),
        "SURFACES_NESTED": bool(run_status["nesting_ok"]),
        "BAD_NESTING_PHIS": [float(value) for value in run_status["bad_nesting_phis"]],
        "INITIAL_SURFACE_VOLUMES": [float(value) for value in initial_surface_volumes],
        "INITIAL_SURFACE_IOTAS": [float(value) for value in initial_surface_iotas],
    }

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
        run_dict['last_successful_eval'] = objective_eval
        run_dict['last_successful_eval_weights'] = np.asarray(search_surface_weights).copy()
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

        # Hardware metrics computed in callback(), not here — computing
        # shortest_distance() inside fun() can corrupt optimizer state.

        if success:
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
        accepted_eval = run_dict.get("last_successful_eval", run_dict.get("search_eval"))
        if accepted_eval is not None and "constraint_values" in accepted_eval:
            evaluation.update(
                {
                    "constraint_values": np.asarray(accepted_eval["constraint_values"], dtype=float),
                    "max_violation": float(accepted_eval["max_violation"]),
                    "stationarity_norm": float(accepted_eval["stationarity_norm"]),
                    "constraint_names": list(accepted_eval.get("constraint_names", [])),
                    "base_total": float(
                        accepted_eval.get(
                            "base_total",
                            accepted_eval.get("total", run_dict["J"]),
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
    max_curvature = np.max(banana_curve.kappa())
    length = curvelength.J()
    curvecurve_min = JCurveCurve.shortest_distance()
    curvesurf_min = JCurveSurface.shortest_distance()
    surface_vessel_min = compute_single_stage_surface_vessel_min_dist(
        JSurfSurf,
        full_stack_status,
        outer_entry["boozer_surface"].surface,
        VV,
    )
    hardware_status = evaluate_single_stage_hardware_constraints(
        curvecurve_min,
        CC_DIST,
        curvesurf_min,
        CS_DIST,
        surface_vessel_min,
        SS_DIST,
        max_curvature,
        CURVATURE_THRESHOLD,
    )
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
    boozer_I = args.boozer_I
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
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)

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

    config_str = build_run_identity_config(
        args,
        stage2_bs_path,
        stage,
        CONSTRAINT_WEIGHT,
        args.constraint_method,
        vol_target,
        iota_target,
        boozer_I,
        banana_surf_radius,
        nphi,
        ntheta,
        rng_seed,
    )
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
        alm_settings = ALMSettings(
            max_outer_iterations=args.alm_max_outer_iters,
            penalty_init=args.alm_penalty_init,
            penalty_scale=args.alm_penalty_scale,
            feasibility_tol=args.alm_feas_tol,
            stationarity_tol=args.alm_stationarity_tol,
        )

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
            ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"]
            + (["surface_vessel_spacing"] if JSurfSurf is not None else []),
            evaluate_problem,
            alm_settings,
            {
                "maxiter": MAXITER,
                "maxcor": args.maxcor,
                "ftol": ftol,
                "gtol": gtol,
            },
            inner_callback=callback,
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
        final_max_curvature = np.max(banana_curve.kappa())
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
    coil_length = float(curvelength.J())
    curve_curve_min_dist = float(JCurveCurve.shortest_distance())
    curve_surface_min_dist = float(JCurveSurface.shortest_distance())
    surface_vessel_min_dist = compute_single_stage_surface_vessel_min_dist(
        JSurfSurf,
        run_dict["surface_status"],
        outer_surface_data["boozer_surface"].surface,
        VV,
    )
    nonqs_ratio = None if args.init_only else float(JnonQSRatio.J())
    boozer_residual = None if args.init_only else float(JBoozerResidual.J())
    final_hardware_status = evaluate_single_stage_hardware_constraints(
        curve_curve_min_dist,
        CC_DIST,
        curve_surface_min_dist,
        CS_DIST,
        surface_vessel_min_dist,
        SS_DIST,
        float(final_max_curvature),
        CURVATURE_THRESHOLD,
    )
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
        "STAGE2_SEED_ORDER": order,
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
