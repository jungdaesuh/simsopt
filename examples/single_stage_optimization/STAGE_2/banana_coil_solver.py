import argparse
import os
import numpy as np

# SIMSOPT imports
from scipy.optimize import minimize, basinhopping
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves, \
                         CurveLength, CurveCurveDistance, LpCurveCurvature)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt._core.derivative import Derivative
from simsopt.geo import CurveCWSFourierCPP
import json
try:
    from numba import njit
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import sys
sys.path.insert(0, EXAMPLE_ROOT)
from alm_utils import (
    ALMSettings,
    augmented_inequality_objective,
    lower_bound_residual,
    minimize_alm,
    upper_bound_residual,
    validate_alm_cli_args,
    zero_gradient_like,
)
from plotting_utils import norm_field_plot, magnitude_field_plot, cross_section_plot

SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = DATABASE_EQUILIBRIA_DIR if os.path.isdir(DATABASE_EQUILIBRIA_DIR) else os.path.join(EXAMPLE_ROOT, "equilibria")
_SMOOTHING_EPS = float(np.finfo(float).eps)


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _smoothmax_selected(values: np.ndarray, temperature: float) -> tuple[float, np.ndarray]:
    temperature = max(float(temperature), _SMOOTHING_EPS)
    shifted = (np.asarray(values, dtype=float) - float(np.max(values))) / temperature
    weights = _stable_softmax(shifted)
    smooth_value = float(np.max(values)) + temperature * float(np.log(np.sum(np.exp(shifted))))
    return smooth_value, weights


def _smoothmin_selected(values: np.ndarray, temperature: float) -> tuple[float, np.ndarray]:
    temperature = max(float(temperature), _SMOOTHING_EPS)
    minimum_value = float(np.min(values))
    shifted = -(np.asarray(values, dtype=float) - minimum_value) / temperature
    weights = _stable_softmax(shifted)
    smooth_value = minimum_value - temperature * float(np.log(np.sum(np.exp(shifted))))
    return smooth_value, weights


def smooth_max_curvature_signed_constraint(
    curve,
    threshold: float,
    temperature: float,
    base_objective_optimizable,
):
    kappa = np.asarray(curve.kappa(), dtype=float)
    hard_max = float(np.max(kappa))
    active_mask = kappa >= (hard_max - 4.0 * float(temperature))
    if not np.any(active_mask):
        active_mask[np.argmax(kappa)] = True
    smooth_max, active_weights = _smoothmax_selected(kappa[active_mask], temperature)
    full_weights = np.zeros_like(kappa)
    full_weights[active_mask] = active_weights
    grad = np.asarray(
        curve.dkappa_by_dcoeff_vjp(full_weights)(base_objective_optimizable),
        dtype=float,
    )
    return smooth_max - float(threshold), grad


def smooth_min_distance_signed_constraint(
    curves,
    minimum_distance: float,
    temperature: float,
    base_objective_optimizable,
):
    pair_blocks = []
    hard_min = np.inf
    for i, curve_i in enumerate(curves):
        gamma_i = np.asarray(curve_i.gamma(), dtype=float)
        for j in range(i):
            curve_j = curves[j]
            gamma_j = np.asarray(curve_j.gamma(), dtype=float)
            diffs = gamma_i[:, None, :] - gamma_j[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            hard_min = min(hard_min, float(np.min(dists)))
            pair_blocks.append((i, j, diffs, dists))

    if not pair_blocks:
        return float(minimum_distance), zero_gradient_like(base_objective_optimizable.x)

    selection_window = 4.0 * float(temperature)
    selected_distances = []
    selected_entries = []
    for i, j, diffs, dists in pair_blocks:
        mask = dists <= (hard_min + selection_window)
        if not np.any(mask):
            mask[np.unravel_index(np.argmin(dists), dists.shape)] = True
        rows, cols = np.nonzero(mask)
        selected_distances.append(dists[rows, cols])
        selected_entries.append((i, j, rows, cols, diffs[rows, cols], dists[rows, cols]))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = _smoothmin_selected(flat_distances, temperature)

    point_gradients = [np.zeros_like(np.asarray(curve.gamma(), dtype=float)) for curve in curves]
    offset = 0
    for i, j, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset:offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(point_gradients[i], rows, local_weights[:, None] * directions)
        np.add.at(point_gradients[j], cols, -local_weights[:, None] * directions)

    derivative = Derivative({})
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    grad = np.asarray(derivative(base_objective_optimizable), dtype=float)
    return float(minimum_distance) - smooth_min, grad


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
        "--major-radius",
        type=float,
        default=float(os.environ.get("MAJOR_RADIUS", "0.915")),
        help="Target major radius used to rescale the plasma surface.",
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=float(os.environ.get("TOROIDAL_FLUX", "0.24")),
        help="Flux-surface label s used when loading the VMEC surface.",
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
        help="Distance soft-min temperature for Stage 2 ALM spacing constraints.",
    )
    parser.add_argument(
        "--alm-curvature-smoothing",
        type=float,
        default=float(os.environ.get("ALM_CURVATURE_SMOOTHING", "0.25")),
        help="Curvature soft-max temperature for Stage 2 ALM curvature constraints.",
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
        help="Perturbation scale for basin-hopping (fraction of DOF range, default 0.01).",
    )
    parser.add_argument(
        "--basin-seed",
        type=int,
        default=int(os.environ.get("BASIN_SEED", "-1")),
        help="RNG seed for basin-hopping (-1 = random, default). Set for reproducibility.",
    )
    return parser.parse_args()


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

def initSurface(R0, s, file_loc, nphi, ntheta):
    # Initialize the boundary magnetic surface and scale it to the target major radius
    surf = SurfaceRZFourier.from_wout(file_loc, range="full torus", nphi=nphi, ntheta=ntheta, s=s)
    # scale the surface down to the target appropriate major radius
    surf.set_dofs(surf.get_dofs()*R0/surf.major_radius())
    print('Major radius target: ', R0)
    print('Major radius actual: ', surf.major_radius())
    print('Minor radius: ', surf.minor_radius())
    return surf


def build_hbt_reference_surfaces(nfp, banana_surf_radius):
    hbt = SurfaceRZFourier(nfp=nfp, stellsym=True)
    hbt.set_rc(0, 0, 0.9115)    # R0 of LCFS semi-circle center
    hbt.set_rc(1, 0, 0.1605)    # Minor radius (thick metal walls)
    hbt.set_zs(1, 0, 0.152)    # Z extent = ±0.152 m (flat top/bottom)

    surf_coils = SurfaceRZFourier(nfp=nfp, stellsym=True)
    surf_coils.set_rc(0, 0, 0.976)
    surf_coils.set_rc(1, 0, banana_surf_radius)
    surf_coils.set_zs(1, 0, banana_surf_radius)

    vv = SurfaceRZFourier(nfp=nfp, stellsym=True)
    vv.set_rc(0, 0, 0.976)
    vv.set_rc(1, 0, 0.222)
    vv.set_zs(1, 0, 0.222)
    return hbt, surf_coils, vv


def initializeCoils(surf, surf_coils, tf_coils, num_quadpoints, order,
                    phi_center, theta_center, phi_width, theta_width, OUT_DIR):
    # Initialize banana coils on the coil winding surface
    banana_curve = CurveCWSFourierCPP(np.linspace(0, 1, num_quadpoints, endpoint=False), order=order, surf=surf_coils)
    banana_curve.set('phic(0)', phi_center)
    banana_curve.set('thetac(0)', theta_center)
    banana_curve.set('phic(1)', phi_width)
    banana_curve.set('thetas(1)', theta_width)

    # Apply symmetries - if stellsym = False, only one per half field period (and two if true)
    banana_coils = coils_via_symmetries([banana_curve], [ScaledCurrent(Current(1), 1e4)], surf_coils.nfp, surf_coils.stellsym)

    # Combined coil set to evaluate magnetic field
    coils = tf_coils + banana_coils
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    # Save initialization state
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_init", close=True)
    unitn = surf.unitnormal()
    pointData = {"B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]}
    surf.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
    return bs, curves, banana_curve, banana_coils

# Helper: evaluate gamma for CurveCWSFourier
def gamma_at_t(curve, t):
    g2 = np.zeros((len(t), 2))
    curve.gamma_2d_impl(g2, t)
    out = np.zeros((len(t), 3))
    curve.surf.gamma_lin(out, g2[:, 0], g2[:, 1])
    return out

# Compute total curve length
def compute_curve_length(pts):
    diffs = pts[1:] - pts[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(seg_lengths)
    return total_length

@njit
def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

@njit
def segment_segment_distance(P1, P2, Q1, Q2):
    """
    Minimum distance between segments P1P2 and Q1Q2.
    Sunday/Lumelsky algorithm with correct re-projection after clamping
    and relative parallelism threshold.
    """
    u = P2 - P1
    v = Q2 - Q1
    w0 = P1 - Q1

    a = np.dot(u, u)  # |u|^2
    b = np.dot(u, v)
    c = np.dot(v, v)  # |v|^2
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    ZERO_LEN = 1e-30  # degenerate segment threshold
    PAR_EPS = 1e-10   # relative parallelism: sin^2(theta) < PAR_EPS

    # Degenerate: P is a point
    if a < ZERO_LEN:
        if c < ZERO_LEN:
            return np.linalg.norm(w0)
        return np.linalg.norm(w0 - _clamp01(e / c) * v)

    # Degenerate: Q is a point
    if c < ZERO_LEN:
        return np.linalg.norm(w0 + _clamp01(-d / a) * u)

    denom = a * c - b * b  # >= 0 by Cauchy-Schwarz

    if denom < PAR_EPS * a * c:
        # Near-parallel: check all four endpoint-to-segment projections
        best_sq = np.inf

        # P1 -> segment Q
        dp = w0 - _clamp01(e / c) * v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        # P2 -> segment Q
        dp = w0 + u - _clamp01((e + b) / c) * v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        # Q1 -> segment P
        dp = w0 + _clamp01(-d / a) * u
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        # Q2 -> segment P
        dp = w0 + _clamp01((b - d) / a) * u - v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        # Interior check: when the true minimum is interior, numerators
        # scale with denom so the division is well-conditioned.
        if denom > 0.0:
            sc_int = (b * e - c * d) / denom
            tc_int = (a * e - b * d) / denom
            if 0.0 <= sc_int <= 1.0 and 0.0 <= tc_int <= 1.0:
                dp = w0 + sc_int * u - tc_int * v
                dsq = np.dot(dp, dp)
                if dsq < best_sq:
                    best_sq = dsq

        return np.sqrt(best_sq)

    # General case: unclamped line-line closest-point parameters
    sc = (b * e - c * d) / denom
    tc = (a * e - b * d) / denom

    # Clamp s and re-project t
    if sc < 0.0:
        sc = 0.0
        tc = e / c
    elif sc > 1.0:
        sc = 1.0
        tc = (e + b) / c

    # Clamp t and re-project s
    if tc < 0.0:
        tc = 0.0
        sc = _clamp01(-d / a)
    elif tc > 1.0:
        tc = 1.0
        sc = _clamp01((b - d) / a)

    dp = w0 + sc * u - tc * v
    return np.sqrt(np.dot(dp, dp))

@njit
def check_all_pairs(segments, tol, neighbor_skip):
    n_segments = segments.shape[0]
    for i in range(n_segments):
        for j in range(n_segments):
            if i == j:
                continue
            # compute minimal periodic distance between segments
            delta = abs(i - j)
            wrapped_delta = min(delta, n_segments - delta)
            if wrapped_delta <= neighbor_skip:
                continue
            P1, P2 = segments[i, 0], segments[i, 1]
            Q1, Q2 = segments[j, 0], segments[j, 1]
            dist = segment_segment_distance(P1, P2, Q1, Q2)
            if dist < tol:
                return True
    return False

def is_self_intersecting(curve, npts=2000, tol_factor=0.1, neighbor_skip=3): # maybe different skip works better
    """
    3D self-intersection checker for CurveCWSFourier objects.

    Parameters:
        curve: CurveCWSFourier object
        npts: number of discretization points (higher is better)
        tol_factor: tolerance as fraction of segment length (default 5%)
        neighbor_skip: number of neighboring segments to skip (default 3)

    Returns:
        True if self-intersecting, False otherwise
    """
    t = np.linspace(0, 1, npts+1)  # closed curve, include endpoint
    pts = gamma_at_t(curve, t)

    # Build segments
    segments = np.zeros((npts, 2, 3))
    for i in range(npts):
        segments[i, 0] = pts[i]
        segments[i, 1] = pts[i+1]

    # Compute segment length and tolerance
    total_length = compute_curve_length(pts)
    seg_length = total_length / npts
    tol = tol_factor * seg_length

    # Run pairwise checking
    return check_all_pairs(segments, tol, neighbor_skip)


def magneticFieldPlots(surf, bs, OUT_DIR_ITER):
    """Generate normal-field and magnitude-field diagnostic plots."""
    mean_abs_relBfinal_norm, modBfinal, surf_area, phi, theta = norm_field_plot(
        surf, bs, OUT_DIR_ITER + "NormFieldPlot")
    magnitude_field_plot(modBfinal, surf_area, phi, theta, OUT_DIR_ITER + "MagFieldPlot")
    return mean_abs_relBfinal_norm

def make_fun(JF, new_bs, new_surf, Jf, Jls, Jccdist, Jc):
    """Factory for the Stage 2 objective function.

    Returns a closure compatible with scipy.optimize.minimize(jac=True)
    that captures all required state explicitly rather than from module scope.
    """
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        unitn = new_surf.unitnormal()
        BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)))
        outstr = f"J={J:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
        outstr += f", Len={Jls.J():.1f}m"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}m"
        outstr += f", Curvature={Jc.J():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad
    return fun


def evaluate_stage2_hardware_constraints(
    coil_length,
    length_target,
    curve_curve_min_dist,
    cc_threshold,
    max_curvature,
    curvature_threshold,
):
    """Evaluate hard Stage 2 hardware constraints against realized geometry."""
    violations = []
    if coil_length > length_target:
        violations.append(
            f"coil_length {coil_length:.6f} exceeds target {length_target:.6f}"
        )
    if curve_curve_min_dist < cc_threshold:
        violations.append(
            f"coil_coil_min_dist {curve_curve_min_dist:.6f} below threshold {cc_threshold:.6f}"
        )
    if max_curvature > curvature_threshold:
        violations.append(
            f"max_curvature {max_curvature:.6f} exceeds threshold {curvature_threshold:.6f}"
        )
    return {
        "success": len(violations) == 0,
        "violations": violations,
        "coil_length": float(coil_length),
        "length_target": float(length_target),
        "curve_curve_min_dist": float(curve_curve_min_dist),
        "cc_threshold": float(cc_threshold),
        "max_curvature": float(max_curvature),
        "curvature_threshold": float(curvature_threshold),
    }

def evaluate_stage2_alm_problem(
    dofs,
    base_objective,
    new_bs,
    new_surf,
    Jf,
    Jls,
    length_target,
    Jccdist,
    Jc,
    distance_smoothing,
    curvature_smoothing,
    multipliers,
    penalty,
):
    base_objective.x = dofs
    base_value = float(base_objective.J())
    base_grad = np.asarray(base_objective.dJ(), dtype=float)
    base_objective_optimizable = base_objective

    coil_length = float(Jls.J())
    length_violation = upper_bound_residual(coil_length, length_target)
    length_grad = np.asarray(Jls.dJ(partials=True)(base_objective_optimizable), dtype=float)

    curve_curve_min_dist = float(Jccdist.shortest_distance())
    curve_curve_violation = lower_bound_residual(curve_curve_min_dist, Jccdist.minimum_distance)
    curve_curve_signed_value, curve_curve_grad = smooth_min_distance_signed_constraint(
        Jccdist.curves,
        Jccdist.minimum_distance,
        distance_smoothing,
        base_objective_optimizable,
    )

    max_curvature = float(np.max(Jc.curve.kappa()))
    curvature_violation = upper_bound_residual(max_curvature, Jc.threshold)
    curvature_signed_value, curvature_grad = smooth_max_curvature_signed_constraint(
        Jc.curve,
        Jc.threshold,
        curvature_smoothing,
        base_objective_optimizable,
    )

    evaluation = augmented_inequality_objective(
        base_value,
        base_grad,
        [
            coil_length - length_target,
            curve_curve_signed_value,
            curvature_signed_value,
        ],
        [length_grad, curve_curve_grad, curvature_grad],
        multipliers,
        penalty,
    )
    evaluation.update(
        {
            "base_value": base_value,
            "constraint_names": [
                "coil_length_upper_bound",
                "coil_coil_spacing",
                "max_curvature",
            ],
            "dual_update_values": [
                coil_length - length_target,
                curve_curve_signed_value,
                curvature_signed_value,
            ],
            "constraint_grads": [length_grad, curve_curve_grad, curvature_grad],
            "constraint_activity_tolerances": [
                1e-3,
                1.0 / distance_smoothing,
                1.0 / curvature_smoothing,
            ],
            "feasibility_values": [
                length_violation,
                curve_curve_violation,
                curvature_violation,
            ],
            "max_feasibility_violation": max(
                length_violation,
                curve_curve_violation,
                curvature_violation,
            ),
        }
    )

    unitn = new_surf.unitnormal()
    BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)))
    outstr = f"ALM J={evaluation['total']:.1e}, Jflux={base_value:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
    outstr += f", Len={coil_length:.1f}m, Len+={length_violation:.2e}, Leng={coil_length - length_target:.2e}"
    outstr += f", C-C-Sep={curve_curve_min_dist:.2f}m, CC+={curve_curve_violation:.2e}, CCg={curve_curve_signed_value:.2e}"
    outstr += f", Curvature={max_curvature:.2f}, Curv+={curvature_violation:.2e}, Curvg={curvature_signed_value:.2e}"
    outstr += f", ║∇L_A║={evaluation['stationarity_norm']:.1e}, μ={penalty:.1e}"
    print(outstr)
    return evaluation



if __name__ == "__main__":
    # PRE-INITIALIZATION
    # ---------------------------------------------------------------------------------------
    args = parse_args()
    validate_alm_cli_args(args)

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
    tf_currents = [Current(1.0) * 1e5 for i in range(20)]   # At some point, update with actual HBT TF current

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
    s = args.toroidal_flux # VMEC flux-surface label

    new_surf = initSurface(R0, s, file_loc, nphi, ntheta)
    banana_surf_nfp = new_surf.nfp

    banana_surf_radius = args.banana_surf_radius
    hbt, surf_coils, VV = build_hbt_reference_surfaces(banana_surf_nfp, banana_surf_radius)

    init_coil_array = initializeCoils(new_surf, surf_coils, tf_coils, num_quadpoints, order,
                                      phi_center, theta_center, phi_width, theta_width, OUT_DIR)
    new_bs = init_coil_array[0]
    new_curves = init_coil_array[1]
    new_banana_curve = init_coil_array[2]
    new_banana_coils = init_coil_array[3]
    new_tf_coils = tf_coils
    new_surf_coils = surf_coils

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

    rng_seed = None
    basin_hop_count = None
    basin_minimization_failures = None
    if args.basin_hops > 0:
        rng_seed = args.basin_seed if args.basin_seed >= 0 else int.from_bytes(os.urandom(4), 'big')
        bh_suffix = f"-BH={args.basin_hops}-BS={args.basin_stepsize:g}-BSeed={rng_seed}"
    else:
        bh_suffix = ""
    alm_suffix = ""
    if CONSTRAINT_METHOD == "alm":
        alm_suffix = (
            f"-CM=alm-ALMOuter={args.alm_max_outer_iters}"
            f"-ALMMu={args.alm_penalty_init:g}-ALMScale={args.alm_penalty_scale:g}"
        )
    else:
        alm_suffix = "-CM=penalty"
    OUT_DIR_ITER = f"{OUT_DIR}R0={R0:g}-s={s:g}-LW={LENGTH_WEIGHT:g}-CCW={CC_WEIGHT:g}-CCT={CC_THRESHOLD:g}-CW={CURVATURE_WEIGHT:g}-CT={CURVATURE_THRESHOLD:g}-SR={banana_surf_radius:0.3f}-Order={order}{alm_suffix}{bh_suffix}/"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # minimize gets called, optimizes based on degrees of freedom from objective function
    dofs = BASE_OBJECTIVE.x if CONSTRAINT_METHOD == "alm" else JF.x
    fun = make_fun(JF, new_bs, new_surf, Jf, Jls, Jccdist, Jc)
    alm_result = None
    if args.init_only:
        res_nit = 0
        optimizer_success = True
        termination_message = "init_only"
        print("Skipping Stage 2 optimizer because --init-only was provided.")
    elif CONSTRAINT_METHOD == "alm":
        if args.basin_hops > 0:
            raise ValueError("--basin-hops is not supported with --constraint-method=alm")
        alm_settings = ALMSettings(
            max_outer_iterations=args.alm_max_outer_iters,
            max_subproblem_continuations=args.alm_max_subproblem_continuations,
            penalty_init=args.alm_penalty_init,
            penalty_scale=args.alm_penalty_scale,
            feasibility_tol=args.alm_feas_tol,
            stationarity_tol=args.alm_stationarity_tol,
            trust_radius_init=(
                None if args.alm_trust_radius_init == 0.0 else args.alm_trust_radius_init
            ),
            trust_radius_min=args.alm_trust_radius_min,
            trust_radius_shrink=args.alm_trust_radius_shrink,
            trust_radius_grow=args.alm_trust_radius_grow,
            max_inner_attempts=args.alm_max_inner_attempts,
        )

        def evaluate_problem(inner_dofs, multipliers, penalty):
            return evaluate_stage2_alm_problem(
                inner_dofs,
                BASE_OBJECTIVE,
                new_bs,
                new_surf,
                Jf,
                Jls,
                LENGTH_TARGET,
                Jccdist,
                Jc,
                args.alm_distance_smoothing,
                args.alm_curvature_smoothing,
                multipliers,
                penalty,
            )

        def outer_state_callback(outer_iteration, multipliers, penalty):
            print(
                f"[ALM] outer_iteration={outer_iteration}, "
                f"multipliers={multipliers.tolist()}, penalty={penalty:.3e}"
            )

        res = minimize_alm(
            dofs,
            ["coil_length_upper_bound", "coil_coil_spacing", "max_curvature"],
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
            'options': {'maxiter': MAXITER, 'maxcor': 300, 'ftol': args.ftol, 'gtol': args.gtol},
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
            res_nit = basin_hop_count
        if hasattr(res, 'lowest_optimization_result'):
            termination_message = str(getattr(res.lowest_optimization_result, 'message', 'basinhopping_complete'))
            optimizer_success = bool(getattr(res.lowest_optimization_result, 'success', True))
        else:
            termination_message = str(getattr(res, 'message', 'basinhopping_complete'))
            optimizer_success = True
        print(f"Basin-hopping complete. Best fun={res.fun:.6e}, hops={args.basin_hops}, seed={rng_seed}")
    else:
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                       options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': args.ftol, 'gtol': args.gtol})
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
    hardware_status = evaluate_stage2_hardware_constraints(
        final_coil_length,
        LENGTH_TARGET,
        final_curve_curve_min_dist,
        CC_THRESHOLD,
        final_max_curvature,
        CURVATURE_THRESHOLD,
    )
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
    fieldError = magneticFieldPlots(new_surf, new_bs, OUT_DIR_ITER)

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    new_bs.save(OUT_DIR_ITER + "biot_savart_opt.json")
    #new_surf.save(OUT_DIR_ITER + "surf_opt.json");
    print(f'Banana Coil Current / TF Current = {new_banana_coils[0].current.get_value() / new_tf_coils[0].current.get_value():.3f}\n')

    # Save the results of optimization to a separate file
    results = {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "CC_THRESHOLD": CC_THRESHOLD,
        "CC_WEIGHT": CC_WEIGHT,
        "CURVATURE_WEIGHT": CURVATURE_WEIGHT,
        "CURVATURE_THRESHOLD": CURVATURE_THRESHOLD,
        "LENGTH_WEIGHT": LENGTH_WEIGHT,
        "CONSTRAINT_METHOD": CONSTRAINT_METHOD,
        "theta_center": theta_center,
        "phi_center": phi_center,
        "theta_width": theta_width,
        "phi_width": phi_width,
        "LENGTH_TARGET": LENGTH_TARGET,
        "MAJOR_RADIUS": R0,
        "TOROIDAL_FLUX": s,
        "NFP": int(banana_surf_nfp),
        "banana_surf_radius": banana_surf_radius,
        "order": order,
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "iterations": res_nit,
        "TERMINATION_MESSAGE": termination_message,
        "OPTIMIZER_SUCCESS": optimizer_success,
        "basin_hops": args.basin_hops,
        "basin_stepsize": args.basin_stepsize if args.basin_hops > 0 else None,
        "basin_seed": rng_seed if args.basin_hops > 0 else None,
        "basin_iterations": basin_hop_count,
        "basin_minimization_failures": basin_minimization_failures,
        "ALM_MAX_OUTER_ITERS": args.alm_max_outer_iters if CONSTRAINT_METHOD == "alm" else None,
        "ALM_MAX_SUBPROBLEM_CONTINUATIONS": args.alm_max_subproblem_continuations if CONSTRAINT_METHOD == "alm" else None,
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
        "ALM_DISTANCE_SMOOTHING": args.alm_distance_smoothing if CONSTRAINT_METHOD == "alm" else None,
        "ALM_CURVATURE_SMOOTHING": args.alm_curvature_smoothing if CONSTRAINT_METHOD == "alm" else None,
        "ALM_FINAL_PENALTY": getattr(alm_result, "penalty", None),
        "ALM_FINAL_MULTIPLIERS": getattr(alm_result, "multipliers", None),
        "ALM_FINAL_CONSTRAINT_VALUES": getattr(alm_result, "constraint_values", None),
        "ALM_FINAL_SOLVER_CONSTRAINT_VALUES": getattr(alm_result, "solver_constraint_values", None),
        "ALM_FINAL_TRUST_RADIUS": getattr(alm_result, "trust_radius", None),
        "ALM_HISTORY": getattr(alm_result, "history", None),
        "FINAL_VOLUME": float(new_surf.volume()),
        "FIELD_ERROR": float(fieldError),
        "SELF_INTERSECTING": intersecting,
        "MAX_CURVATURE": final_max_curvature,
        "COIL_LENGTH": final_coil_length,
        "CURVE_CURVE_MIN_DIST": final_curve_curve_min_dist,
        "HARDWARE_CONSTRAINTS_OK": hardware_status["success"],
        "HARDWARE_CONSTRAINT_VIOLATIONS": hardware_status["violations"],
    }
    with open(os.path.join(OUT_DIR_ITER, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)
