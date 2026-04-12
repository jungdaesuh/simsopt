import argparse
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

# SIMSOPT imports
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil
from simsopt.geo import (
    curves_to_vtk,
    create_equally_spaced_curves,
    CurveLength,
    CurveCurveDistance,
    LpCurveCurvature,
)
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
    format_local_stage2_run_dir,
    validate_normalized_toroidal_flux,
)
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.basin_hopping import run_basin_hopping, telemetry_values as basin_telemetry_values
from banana_opt.stage2_geometry import (
    init_surface as _init_surface,
    initialize_coils as _initialize_coils,
    is_self_intersecting,
    magnetic_field_plots as _magnetic_field_plots,
)
from banana_opt.current_contracts import BANANA_CURRENT_HARD_LIMIT_A
from banana_opt.stage2_objectives import (
    build_stage2_alm_settings,
    build_stage2_results as _build_stage2_results_impl,
    evaluate_stage2_alm_problem as _evaluate_stage2_alm_problem,
    evaluate_stage2_hardware_constraints as _evaluate_stage2_hardware_constraints,
    make_stage2_fun,
    smooth_max_curvature_signed_constraint,
    smooth_min_distance_signed_constraint,
    stage2_constraint_activity_tolerances,
)

REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = DATABASE_EQUILIBRIA_DIR if os.path.isdir(DATABASE_EQUILIBRIA_DIR) else os.path.join(EXAMPLE_ROOT, "equilibria")
STAGE2_ALM_CONSTRAINT_NAMES = (
    "coil_length_upper_bound",
    "coil_coil_spacing",
    "max_curvature",
    "banana_current_upper_bound",
)


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
    if not (0.0 < banana_init_current_A <= BANANA_CURRENT_HARD_LIMIT_A):
        raise ValueError(
            f"--banana-init-current-A must be in the interval (0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}]."
        )
    if not (0.0 < banana_current_max_A <= BANANA_CURRENT_HARD_LIMIT_A):
        raise ValueError(
            f"--banana-current-max-A must be in the interval (0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}]."
        )
    if banana_init_current_A > banana_current_max_A:
        raise ValueError(
            "--banana-init-current-A cannot exceed --banana-current-max-A."
        )


def unwrap_current_optimizable(current):
    scale = 1.0
    current_optimizable = current
    while hasattr(current_optimizable, "current_to_scale") and hasattr(
        current_optimizable,
        "scale",
    ):
        scale *= float(current_optimizable.scale)
        current_optimizable = current_optimizable.current_to_scale
    if not hasattr(current_optimizable, "local_lower_bounds") or not hasattr(
        current_optimizable,
        "local_upper_bounds",
    ):
        raise TypeError("Banana current does not expose local bounds.")
    return current_optimizable, scale


def apply_banana_current_upper_bound(current, banana_current_max_A):
    current_optimizable, scale = unwrap_current_optimizable(current)
    if scale == 0.0:
        raise ValueError("Banana current scale must be non-zero to apply a bound.")
    lower_bounds = np.asarray(current_optimizable.local_lower_bounds, dtype=float).copy()
    upper_bounds = np.asarray(current_optimizable.local_upper_bounds, dtype=float).copy()
    scaled_magnitude_bound = float(banana_current_max_A) / abs(scale)
    lower_bounds[0] = max(lower_bounds[0], -scaled_magnitude_bound)
    upper_bounds[0] = min(upper_bounds[0], scaled_magnitude_bound)
    current_optimizable.local_lower_bounds = lower_bounds
    current_optimizable.local_upper_bounds = upper_bounds


def build_lbfgsb_bounds(optimizable):
    return list(
        zip(
            np.asarray(optimizable.lower_bounds, dtype=float),
            np.asarray(optimizable.upper_bounds, dtype=float),
        )
    )


def banana_current_exceeds_limit(current_A: float, banana_current_max_A: float) -> bool:
    return abs(float(current_A)) > float(banana_current_max_A)


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
        default=float(os.environ.get("BANANA_SURF_RADIUS", "0.22")),
        help="Coil surface minor radius.",
    )
    parser.add_argument(
        "--tf-current-A",
        type=float,
        default=float(os.environ.get("TF_CURRENT_A", "1e5")),
        help="Per-TF-coil current in physical SI amperes (default 1e5).",
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
        "--major-radius",
        type=float,
        default=float(os.environ.get("MAJOR_RADIUS", "0.915")),
        help="Target major radius used to rescale the plasma surface.",
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
        "--length-weight",
        type=float,
        default=float(os.environ.get("LENGTH_WEIGHT", "0.0005")),
        help="Curve-length penalty weight.",
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=float(os.environ.get("LENGTH_TARGET", "1.75")),
        help="Curve-length target in meters.",
    )
    parser.add_argument(
        "--cc-threshold",
        type=float,
        default=float(os.environ.get("CC_THRESHOLD", "0.05")),
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
        default=float(os.environ.get("CURVATURE_THRESHOLD", "40")),
        help="Curvature threshold.",
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
    return surfaces.hbt, surfaces.coil_winding_surface, surfaces.vessel


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


def load_stage2_seed_configuration(seed_bs_path, surf, num_tf_coils, out_dir):
    bs = load(seed_bs_path)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    coils = bs.coils
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir + "curves_init", close=True)
    unitn = surf.unitnormal()
    pointData = {"B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]}
    surf.to_vtk(out_dir + "surf_init", extra_data=pointData)

    banana_coils = coils[num_tf_coils:]
    banana_curve = banana_coils[0].curve
    tf_coils = coils[:num_tf_coils]
    return bs, curves, banana_curve, banana_coils, tf_coils


def main(parsed_args=None):
    # PRE-INITIALIZATION
    # ---------------------------------------------------------------------------------------
    args = parse_args() if parsed_args is None else parsed_args
    validate_alm_cli_args(args)
    if parsed_args is not None:
        validate_banana_current_cli_args(args)

    # File for the desired boundary magnetic surface:
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)

    # Make Directory for output
    OUT_DIR = os.path.join(args.output_root, f"outputs-{plasma_surf_filename}") + "/"
    os.makedirs(OUT_DIR, exist_ok=True)

    nphi = args.nphi
    ntheta = args.ntheta
    surf = None

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

    R0 = args.major_radius # major radius
    s = validate_normalized_toroidal_flux(
        args.toroidal_flux,
        field_name="--toroidal-flux",
    ) # VMEC flux-surface label

    new_surf = _init_surface(R0, s, file_loc, nphi, ntheta)
    banana_surf_nfp = new_surf.nfp

    banana_surf_radius = args.banana_surf_radius
    hbt, surf_coils, VV = build_hbt_reference_surfaces(banana_surf_nfp, banana_surf_radius)

    if args.stage2_bs_path:
        print(f"Loading Stage 2 seed from {args.stage2_bs_path}")
        init_coil_array = load_stage2_seed_configuration(
            args.stage2_bs_path,
            new_surf,
            len(tf_coils),
            OUT_DIR,
        )
        new_tf_coils = init_coil_array[4]
        tf_current_A = float(new_tf_coils[0].current.get_value())
    else:
        init_coil_array = _initialize_coils(
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
        )
        new_tf_coils = tf_coils
    new_bs = init_coil_array[0]
    new_curves = init_coil_array[1]
    new_banana_curve = init_coil_array[2]
    new_banana_coils = init_coil_array[3]
    new_surf_coils = surf_coils
    initial_banana_current_A = float(new_banana_coils[0].current.get_value())

    # MAIN OPTIMIZATION
    # ---------------------------------------------------------------------------------------
    # Number of iterations to perform:
    MAXITER = args.maxiter
    # boolean for determining whether coil self-intersects
    intersecting = False

    # Weight on the curve lengths in the objective function
    # We'll penalize the coil if it becomes longer than an target length of 1.75 m
    LENGTH_WEIGHT = args.length_weight
    LENGTH_TARGET = max(args.length_target, 1.75)  # Baseline default floor
    if args.length_target < 1.75:
        print(f"WARNING: --length-target {args.length_target} below baseline default, clamped to 1.75")

    # Threshold and weight for the coil-to-coil distance penalty
    CC_THRESHOLD = max(args.cc_threshold, 0.05)  # Baseline default floor
    if args.cc_threshold < 0.05:
        print(f"WARNING: --cc-threshold {args.cc_threshold} below baseline default, clamped to 0.05")
    CC_WEIGHT = args.cc_weight

    # Threshold and weight for the coil curvature penalty
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = max(args.curvature_threshold, 40)
    if args.curvature_threshold < 40:
        print(f"WARNING: --curvature-threshold {args.curvature_threshold} below hardware floor, clamped to 40")

    # Define the individual terms objective function:
    Jf = SquaredFlux(new_surf, new_bs) # penalty on B dot n
    Jls = CurveLength(new_banana_curve) # penalty on curve length
    Jccdist = CurveCurveDistance(new_curves, CC_THRESHOLD) #penalty on coil-to-coil distance

    # Lp-norm curvature penalty (configurable via --curvature-p-norm)
    Jc = LpCurveCurvature(new_banana_curve, args.curvature_p_norm, CURVATURE_THRESHOLD)
    print(f"Initial coil length: {Jls.J():.2f} [m]")

    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    SQUARED_FLUX_WEIGHT = args.squared_flux_weight
    CONSTRAINT_METHOD = args.constraint_method
    JF = SQUARED_FLUX_WEIGHT * Jf \
        + LENGTH_WEIGHT * QuadraticPenalty(Jls, LENGTH_TARGET, "max") \
        + CC_WEIGHT * Jccdist \
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
    )
    OUT_DIR_ITER = (
        OUT_DIR
        + format_local_stage2_run_dir(
            stage2_seed_spec,
            constraint_method=CONSTRAINT_METHOD,
            alm_max_outer_iters=args.alm_max_outer_iters,
            alm_penalty_init=args.alm_penalty_init,
            alm_penalty_scale=args.alm_penalty_scale,
            basin_hops=args.basin_hops,
            basin_stepsize=args.basin_stepsize,
            basin_temperature=args.basin_temperature,
            basin_niter_success=args.basin_niter_success,
            basin_seed=rng_seed,
        )
        + "/"
    )
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # minimize gets called, optimizes based on degrees of freedom from objective function
    dofs = BASE_OBJECTIVE.x if CONSTRAINT_METHOD == "alm" else JF.x
    lbfgsb_bounds = None
    if CONSTRAINT_METHOD != "alm":
        if args.stage2_bs_path and banana_current_exceeds_limit(
            initial_banana_current_A,
            args.banana_current_max_A,
        ):
            raise ValueError(
                "Loaded Stage 2 seed starts above --banana-current-max-A; "
                "penalty mode cannot accept an infeasible banana-current seed."
            )
        apply_banana_current_upper_bound(
            new_banana_coils[0].current,
            args.banana_current_max_A,
        )
        lbfgsb_bounds = build_lbfgsb_bounds(JF)
    fun = make_stage2_fun(JF, new_bs, new_surf, Jf, Jls, Jccdist, Jc)
    alm_result = None
    if CONSTRAINT_METHOD == "alm":
        alm_settings = build_stage2_alm_settings(args)

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
                np.zeros(len(STAGE2_ALM_CONSTRAINT_NAMES), dtype=float),
                alm_settings.penalty_init,
                seed=args.alm_taylor_test_seed,
            )
            _print_taylor_test_summary("ALM Taylor", alm_taylor_result)

    if args.init_only:
        res_nit = 0
        optimizer_success = True
        termination_message = "init_only"
        print("Skipping Stage 2 optimizer because --init-only was provided.")
    elif CONSTRAINT_METHOD == "alm":
        if args.basin_hops > 0:
            raise ValueError("--basin-hops is not supported with --constraint-method=alm")
        res = minimize_alm(
            dofs,
            STAGE2_ALM_CONSTRAINT_NAMES,
            evaluate_problem,
            alm_settings,
            {
                "maxiter": MAXITER,
                "maxcor": 300,
                "ftol": args.ftol,
                "gtol": args.gtol,
            },
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
            'bounds': lbfgsb_bounds,
            'options': {'maxiter': MAXITER, 'maxcor': 300, 'ftol': args.ftol, 'gtol': args.gtol},
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
            options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': args.ftol, 'gtol': args.gtol},
        )
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        print(res.message)


    # Ensure SIMSOPT state matches the best result (needed after basin-hopping)
    if not args.init_only:
        JF.x = res.x
        BASE_OBJECTIVE.x = res.x

    # POST-OPTIMIZATION PROCESSING AND OUTPUTS
    # ---------------------------------------------------------------------------------------
    if is_self_intersecting(new_banana_curve):
        print("BANANA COIL IS SELF-INTERSECTING!")
        intersecting = True

    final_coil_length = float(Jls.J())
    final_curve_curve_min_dist = float(Jccdist.shortest_distance())
    final_max_curvature = float(np.max(new_banana_curve.kappa()))
    final_banana_current_A = float(new_banana_coils[0].current.get_value())
    hardware_status = _evaluate_stage2_hardware_constraints(
        final_coil_length,
        LENGTH_TARGET,
        final_curve_curve_min_dist,
        CC_THRESHOLD,
        final_max_curvature,
        CURVATURE_THRESHOLD,
    )
    if banana_current_exceeds_limit(final_banana_current_A, args.banana_current_max_A):
        hardware_status["success"] = False
        hardware_status["violations"] = list(hardware_status["violations"]) + [
            (
                f"|banana_current| {abs(final_banana_current_A):.6f} exceeds maximum "
                f"{float(args.banana_current_max_A):.6f}"
            )
        ]
    if not hardware_status["success"]:
        optimizer_success = False
        constraint_summary = "; ".join(hardware_status["violations"])
        if termination_message:
            termination_message = f"{termination_message}; hardware_constraints_failed"
        else:
            termination_message = "hardware_constraints_failed"
        print("/!\\ /!\\ Stage 2 hardware constraint violation /!\\ /!\\")
        print(constraint_summary)

    curves_to_vtk(new_curves, OUT_DIR_ITER + "curves_opt", close=True)
    new_bs.set_points(new_surf.gamma().reshape((-1, 3)))
    unitn = new_surf.unitnormal()
    pointData = {"B_N/B": np.sum(new_bs.B().reshape(unitn.shape) *
        unitn, axis=2)[:, :, None] / np.sqrt(np.sum(new_bs.B().reshape(unitn.shape)**2, axis=2))[:, :, None]}
    new_surf.to_vtk(OUT_DIR_ITER + "surf_opt", extra_data=pointData)
    VV.to_vtk(OUT_DIR_ITER + "VV")

    # Create toroidal cross section plot
    cross_section_plot(new_surf_coils, new_surf, new_banana_curve, OUT_DIR_ITER + "CrossSectionPlot", hbt, VV)
    # Create field error plot
    fieldError = _magnetic_field_plots(new_surf, new_bs, OUT_DIR_ITER)

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    new_bs.save(OUT_DIR_ITER + "biot_savart_opt.json")
    #new_surf.save(OUT_DIR_ITER + "surf_opt.json");
    print(f'Banana Coil Current / TF Current = {new_banana_coils[0].current.get_value() / new_tf_coils[0].current.get_value():.3f}\n')

    # Save the results of optimization to a separate file
    results = _build_stage2_results_impl(
        args=args,
        plasma_surf_filename=plasma_surf_filename,
        file_loc=file_loc,
        stage2_bs_path=args.stage2_bs_path,
        tf_current_A=tf_current_A,
        tf_current_sum_abs_A=sum(abs(coil.current.get_value()) for coil in new_tf_coils),
        num_tf_coils=len(new_tf_coils),
        initial_banana_current_A=initial_banana_current_A,
        banana_current_A=final_banana_current_A,
        banana_to_tf_current_ratio=(
            final_banana_current_A / new_tf_coils[0].current.get_value()
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
        field_error=fieldError,
        intersecting=intersecting,
        final_max_curvature=final_max_curvature,
        final_coil_length=final_coil_length,
        final_curve_curve_min_dist=final_curve_curve_min_dist,
        hardware_status=hardware_status,
    )
    with open(os.path.join(OUT_DIR_ITER, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)


if __name__ == "__main__":
    main()
