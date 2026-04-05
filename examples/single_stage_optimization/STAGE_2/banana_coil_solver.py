import argparse
from dataclasses import dataclass
import os
import sys
import json
import time

import jax
import numpy as np
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, EXAMPLE_ROOT)

SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
SRC_ROOT = os.path.join(SIMSOPT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, SIMSOPT_ROOT)
sys.path.insert(0, REPO_ROOT)

from jax_host_boundary import host_array, host_float
from repo_bootstrap import bootstrap_local_simsopt


bootstrap_local_simsopt(SRC_ROOT)
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = (
    DATABASE_EQUILIBRIA_DIR
    if os.path.isdir(DATABASE_EQUILIBRIA_DIR)
    else os.path.join(EXAMPLE_ROOT, "equilibria")
)
STAGE2_TARGET_OBJECTIVE_DOF_LAYOUT_ERROR = (
    "Stage 2 target objective DOF layout does not match the composite objective."
)
CURVATURE_THRESHOLD_FLOOR = 20.0
CURVATURE_THRESHOLD_CEILING = 40.0


def resolve_curvature_threshold(value: float) -> float:
    """Apply the shared HBT curvature-threshold policy for Stage 2."""
    return min(
        max(float(value), CURVATURE_THRESHOLD_FLOOR),
        CURVATURE_THRESHOLD_CEILING,
    )


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
        "--export-objective-json",
        default=os.environ.get("STAGE2_EXPORT_OBJECTIVE_JSON"),
        help="Optional path to write the initialized squared-flux/composite objective snapshot as JSON.",
    )
    parser.add_argument(
        "--override-dofs-json",
        default=os.environ.get("STAGE2_OVERRIDE_DOFS_JSON"),
        help=(
            "Optional JSON file containing an explicit DOF vector to load into the "
            "Stage 2 composite objective before probe/init/optimization."
        ),
    )
    parser.add_argument(
        "--trajectory-json",
        default=os.environ.get("STAGE2_TRAJECTORY_JSON"),
        help="Optional path to write per-evaluation objective diagnostics as JSON.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip heavy VTK/plot/save artifact generation while still writing results.json.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Build the initialized Stage 2 objective snapshot, export JSON if requested, and exit before optimization/postprocess.",
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
        "--num-quadpoints",
        type=int,
        default=int(os.environ.get("NUM_QUADPOINTS", "128")),
        help="Number of quadrature points per coil (default 128).",
    )
    parser.add_argument(
        "--curvature-p-norm",
        type=int,
        default=int(os.environ.get("CURVATURE_P_NORM", "4")),
        help="Lp norm exponent for curvature penalty (default 4).",
    )
    parser.add_argument(
        "--squared-flux-weight",
        type=float,
        default=float(os.environ.get("SQUARED_FLUX_WEIGHT", "1.0")),
        help="Squared flux objective weight (default 1.0).",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "jax"],
        default=os.environ.get("SIMSOPT_BACKEND")
        or os.environ.get("STAGE2_BACKEND", "cpu"),
        help="Field backend: 'cpu' (simsoptpp) or 'jax' (JAX Biot-Savart). "
        "Env: SIMSOPT_BACKEND (or legacy STAGE2_BACKEND).",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=["scipy", "hybrid", "ondevice"],
        default=os.environ.get("STAGE2_OPTIMIZER_BACKEND")
        or os.environ.get("OPTIMIZER_BACKEND", "scipy"),
        help=(
            "Stage 2 optimizer backend. "
            "'scipy' is the trusted reference lane; "
            "'ondevice' is the private JAX target optimizer lane. "
            "'hybrid' is accepted for contract consistency but not currently "
            "supported for the Stage 2 outer loop."
        ),
    )
    parser.add_argument(
        "--record-warm-timings",
        action="store_true",
        help=(
            "After the primary JAX ondevice Stage 2 optimization run, reset to "
            "the same initial DOFs and rerun once in-process to capture a warm "
            "timing."
        ),
    )
    parser.add_argument(
        "--field-diagnostic-stride",
        type=int,
        default=int(os.environ.get("FIELD_DIAGNOSTIC_STRIDE", "0")),
        help=(
            "Refresh the expensive squared-flux-only and mean-|B.n|/|B| "
            "diagnostics every N optimizer evaluations on explicit value/grad "
            "lanes. Use 0 to select the lane default."
        ),
    )
    parser.add_argument(
        "--profile-step-json",
        default=None,
        help=(
            "Write a one-step explicit Stage 2 objective breakdown JSON. "
            "Supported on reference/value-and-gradient lanes only."
        ),
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
    surf = SurfaceRZFourier.from_wout(
        file_loc, range="full torus", nphi=nphi, ntheta=ntheta, s=s
    )
    # scale the surface down to the target appropriate major radius
    surf.set_dofs(surf.get_dofs() * R0 / surf.major_radius())
    print("Major radius target: ", R0)
    print("Major radius actual: ", surf.major_radius())
    print("Minor radius: ", surf.minor_radius())
    return surf


def initializeCoils(
    surf,
    surf_coils,
    tf_coils,
    num_quadpoints,
    order,
    phi_center,
    theta_center,
    phi_width,
    theta_width,
    OUT_DIR,
):
    # Initialize banana coils on the coil winding surface
    # Keep the production banana-coil class shared across CPU and JAX lanes.
    # CurveCWSFourierCPP now exposes the JAX geometry/VJP surface needed by the
    # target lane, and reusing it here avoids backend-specific curve-derivative
    # drift in near-threshold Stage 2 states.
    banana_curve = CurveCWSFourierCPP(
        np.linspace(0, 1, num_quadpoints, endpoint=False), order=order, surf=surf_coils
    )
    banana_curve.set("phic(0)", phi_center)
    banana_curve.set("thetac(0)", theta_center)
    banana_curve.set("phic(1)", phi_width)
    banana_curve.set("thetas(1)", theta_width)

    # Apply symmetries - if stellsym = False, only one per half field period (and two if true)
    banana_coils = coils_via_symmetries(
        [banana_curve],
        [ScaledCurrent(Current(1), 1e4)],
        surf_coils.nfp,
        surf_coils.stellsym,
    )

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


# Helper: evaluate gamma for the Stage 2 banana-curve winding-surface class.
def gamma_at_t(curve, t):
    out = np.zeros((len(t), 3))
    curve.gamma_impl(out, np.asarray(t, dtype=np.float64))
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
    PAR_EPS = 1e-10  # relative parallelism: sin^2(theta) < PAR_EPS

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


def is_self_intersecting(
    curve, npts=2000, tol_factor=0.1, neighbor_skip=3
):  # maybe different skip works better
    """
    3D self-intersection checker for Stage 2 banana-curve objects.

    Parameters:
        curve: winding-surface banana curve
        npts: number of discretization points (higher is better)
        tol_factor: tolerance as fraction of segment length (default 5%)
        neighbor_skip: number of neighboring segments to skip (default 3)

    Returns:
        True if self-intersecting, False otherwise
    """
    t = np.linspace(0, 1, npts + 1)  # closed curve, include endpoint
    pts = gamma_at_t(curve, t)

    # Build segments
    segments = np.zeros((npts, 2, 3))
    for i in range(npts):
        segments[i, 0] = pts[i]
        segments[i, 1] = pts[i + 1]

    # Compute segment length and tolerance
    total_length = compute_curve_length(pts)
    seg_length = total_length / npts
    tol = tol_factor * seg_length

    # Run pairwise checking
    return check_all_pairs(segments, tol, neighbor_skip)


def magneticFieldPlots(surf, bs, OUT_DIR_ITER):
    """Generate normal-field and magnitude-field diagnostic plots."""
    mean_abs_relBfinal_norm, modBfinal, surf_area, phi, theta = norm_field_plot(
        surf, bs, OUT_DIR_ITER + "NormFieldPlot"
    )
    magnitude_field_plot(
        modBfinal, surf_area, phi, theta, OUT_DIR_ITER + "MagFieldPlot"
    )
    return mean_abs_relBfinal_norm


def compute_mean_abs_relbn(surf, bs):
    """Return the area-weighted mean |B·n|/|B| without plotting artifacts."""
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    del theta, phi  # keep logic aligned with plotting utility inputs
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1.0 / absn)[:, :, None]
    sqrt_area = np.sqrt(absn.reshape((-1, 1)) / float(absn.size))
    surf_area = sqrt_area**2
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bfinal = host_array(bs.B()).reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    return host_float(np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area))


def resolve_stage2_field_bs(new_bs, bs_jax):
    return bs_jax if bs_jax is not None else new_bs


def _block_until_ready_tree(value):
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_stage2_callback(callback):
    start = time.perf_counter()
    _block_until_ready_tree(callback())
    return float(time.perf_counter() - start)


def time_stage2_callback_result(callback):
    start = time.perf_counter()
    result = callback()
    _block_until_ready_tree(result)
    return float(time.perf_counter() - start), result


def profile_stage2_named_callbacks(callbacks):
    return {
        name: time_stage2_callback(callback) for name, callback in callbacks.items()
    }


def build_stage2_profile_breakdown(timings):
    total_s = float(sum(timings.values()))
    dominant = sorted(
        (
            {
                "name": name,
                "elapsed_s": elapsed_s,
                "share": (elapsed_s / total_s) if total_s > 0.0 else 0.0,
            }
            for name, elapsed_s in timings.items()
        ),
        key=lambda item: item["elapsed_s"],
        reverse=True,
    )
    return total_s, dominant


def build_stage2_objective_term_callbacks(
    context,
):
    return {
        "squared_flux": {
            "J": context.Jf.J,
            "dJ": lambda: compute_stage2_term_gradient(context.Jf, context.JF),
        },
        "length_penalty": {
            "J": lambda: compute_stage2_length_penalty_value(
                context.Jls.J(),
                context.length_target,
            ),
            "dJ": lambda: compute_stage2_length_penalty_gradient(
                host_float(context.Jls.J()),
                context.length_target,
                context.Jls,
                context.JF,
            ),
        },
        "coil_distance": {
            "J": context.Jccdist.J,
            "dJ": lambda: compute_stage2_term_gradient(context.Jccdist, context.JF),
        },
        "curvature": {
            "J": context.Jc.J,
            "dJ": lambda: compute_stage2_term_gradient(context.Jc, context.JF),
        },
    }


def profile_stage2_objective_terms(context):
    term_callbacks = build_stage2_objective_term_callbacks(context)
    value_timings = profile_stage2_named_callbacks(
        {name: callbacks["J"] for name, callbacks in term_callbacks.items()}
    )
    gradient_timings = profile_stage2_named_callbacks(
        {name: callbacks["dJ"] for name, callbacks in term_callbacks.items()}
    )
    value_total_s, dominant_value_terms = build_stage2_profile_breakdown(value_timings)
    gradient_total_s, dominant_gradient_terms = build_stage2_profile_breakdown(
        gradient_timings
    )
    return {
        "value_timings_s": value_timings,
        "value_total_s": value_total_s,
        "dominant_value_terms": dominant_value_terms,
        "gradient_timings_s": gradient_timings,
        "gradient_total_s": gradient_total_s,
        "dominant_gradient_terms": dominant_gradient_terms,
    }


def profile_stage2_squared_flux_internal_components(Jf):
    if getattr(Jf, "_use_jax_native", True):
        return {}, 0.0, [], {}, [], []

    field_B_for_J_s, field_B_for_J = time_stage2_callback_result(Jf.field.B)
    integral_only_s = time_stage2_callback(lambda: Jf._jit_integral(field_B_for_J))
    field_B_for_dJ_s, field_B_for_dJ = time_stage2_callback_result(Jf.field.B)
    integral_value_grad_s, (_, dJ_dB) = time_stage2_callback_result(
        lambda: Jf._jit_integral_value_grad(field_B_for_dJ)
    )
    field_B_vjp_component_timings = {}
    dominant_field_B_vjp_components = []
    dominant_field_B_vjp_coils = []
    if hasattr(Jf.field, "profile_B_vjp"):
        field_B_vjp_profile = Jf.field.profile_B_vjp(host_array(dJ_dB))
        field_B_vjp_s = float(field_B_vjp_profile["wall_time_s"])
        field_B_vjp_component_timings = {
            name: float(elapsed_s)
            for name, elapsed_s in field_B_vjp_profile["component_timings_s"].items()
        }
        dominant_field_B_vjp_components = list(
            field_B_vjp_profile["dominant_components"]
        )
        dominant_field_B_vjp_coils = list(field_B_vjp_profile["dominant_coils"])
    else:
        field_B_vjp_s, _ = time_stage2_callback_result(
            lambda: Jf.field.B_vjp(host_array(dJ_dB))
        )
    timings = {
        "field_B_for_J_s": field_B_for_J_s,
        "integral_only_s": integral_only_s,
        "field_B_for_dJ_s": field_B_for_dJ_s,
        "integral_value_grad_s": integral_value_grad_s,
        "field_B_vjp_s": field_B_vjp_s,
    }
    total_s, dominant = build_stage2_profile_breakdown(timings)
    return (
        timings,
        total_s,
        dominant,
        field_B_vjp_component_timings,
        dominant_field_B_vjp_components,
        dominant_field_B_vjp_coils,
    )


@dataclass(frozen=True)
class Stage2ObjectiveContext:
    """Stage 2 objective state plus the composite root used for DOF projection."""

    # `JF` is the composite/root Optimizable whose DOF layout defines the
    # gradient space for all explicit Stage 2 term projections.
    JF: object
    new_bs: object
    new_surf: object
    Jf: object
    Jls: object
    Jccdist: object
    Jc: object
    squared_flux_weight: float
    length_weight: float
    length_target: float
    cc_weight: float
    cc_threshold: float
    curvature_weight: float
    bs_jax: object | None = None


def make_stage2_objective_context(
    JF,
    new_bs,
    new_surf,
    Jf,
    Jls,
    Jccdist,
    Jc,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    *,
    bs_jax=None,
):
    return Stage2ObjectiveContext(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        float(squared_flux_weight),
        float(length_weight),
        float(length_target),
        float(cc_weight),
        float(cc_threshold),
        float(curvature_weight),
        bs_jax,
    )


def compute_stage2_field_diagnostics(
    new_bs,
    new_surf,
    *,
    bs_jax=None,
):
    field_bs = resolve_stage2_field_bs(new_bs, bs_jax)
    return {
        "mean_abs_relBfinal_norm": compute_mean_abs_relbn(new_surf, field_bs),
    }


def compute_stage2_length_penalty_value(curve_length, length_target):
    return 0.5 * max(host_float(curve_length) - host_float(length_target), 0.0) ** 2


def compute_stage2_max_curvature_value(Jc):
    banana_curve = getattr(Jc, "curve", None)
    if banana_curve is not None:
        return float(np.max(banana_curve.kappa()))
    return host_float(Jc.J())


def stage2_curvature_within_threshold(curvature: float, threshold: float) -> bool:
    return host_float(curvature) <= host_float(threshold)


def compute_stage2_term_gradient(term, root_objective):
    return host_array(term.dJ(partials=True)(root_objective))


def _stage2_term_payload_entry(weight, raw_value, raw_grad):
    raw_grad_array = host_array(raw_grad)
    weight_float = host_float(weight)
    raw_value_float = host_float(raw_value)
    weighted_grad = weight_float * raw_grad_array
    return {
        "weight": weight_float,
        "raw_J": raw_value_float,
        "J": weight_float * raw_value_float,
        "dJ": weighted_grad,
        "grad_norm": float(np.linalg.norm(weighted_grad)),
    }


def _serialize_stage2_term_payload(entries):
    return {
        name: {
            "weight": float(entry["weight"]),
            "raw_J": float(entry["raw_J"]),
            "J": float(entry["J"]),
            "dJ": np.asarray(entry["dJ"], dtype=float).tolist(),
            "grad_norm": float(entry["grad_norm"]),
        }
        for name, entry in entries.items()
    }


def _build_stage2_explicit_term_payload(context):
    squared_flux_grad = compute_stage2_term_gradient(context.Jf, context.JF)
    squared_flux = host_float(context.Jf.J())
    curve_length = host_float(context.Jls.J())
    length_penalty = compute_stage2_length_penalty_value(
        curve_length,
        context.length_target,
    )
    length_grad = compute_stage2_length_penalty_gradient(
        curve_length,
        context.length_target,
        context.Jls,
        context.JF,
    )
    coil_distance_penalty = host_float(context.Jccdist.J())
    coil_distance_grad = compute_stage2_term_gradient(context.Jccdist, context.JF)
    curvature_penalty = host_float(context.Jc.J())
    curvature = compute_stage2_max_curvature_value(context.Jc)
    curvature_grad = compute_stage2_term_gradient(context.Jc, context.JF)
    entries = {
        "squared_flux": _stage2_term_payload_entry(
            context.squared_flux_weight,
            squared_flux,
            squared_flux_grad,
        ),
        "length_penalty": _stage2_term_payload_entry(
            context.length_weight,
            length_penalty,
            length_grad,
        ),
        "coil_distance_penalty": _stage2_term_payload_entry(
            context.cc_weight,
            coil_distance_penalty,
            coil_distance_grad,
        ),
        "curvature_penalty": _stage2_term_payload_entry(
            context.curvature_weight,
            curvature_penalty,
            curvature_grad,
        ),
    }
    return entries, curve_length, curvature


def _build_stage2_target_term_payload(target_objective_bundle, dofs):
    raw_terms = getattr(target_objective_bundle, "raw_terms", None)
    terms = getattr(target_objective_bundle, "terms", ())
    if raw_terms is None or not terms:
        return None
    dofs64 = host_array(dofs)
    raw_values = host_array(raw_terms(dofs64))
    raw_gradients = host_array(jax.jacrev(raw_terms)(dofs64))
    entries = {}
    for index, term in enumerate(terms):
        entries[term.name] = _stage2_term_payload_entry(
            term.weight,
            raw_values[index],
            raw_gradients[index],
        )
    return _serialize_stage2_term_payload(entries)


def compute_stage2_length_penalty_gradient(
    curve_length,
    length_target,
    Jls,
    root_objective,
):
    active_diff = max(host_float(curve_length) - host_float(length_target), 0.0)
    return active_diff * compute_stage2_term_gradient(Jls, root_objective)


@dataclass(frozen=True)
class Stage2DistanceConstraintState:
    coil_coil_distance: float
    coil_distance_penalty: float
    cc_threshold: float
    violated: bool


def compute_stage2_distance_constraint_state(
    context,
    *,
    term_entries,
):
    """Return the Stage 2 coil-distance state for the current objective eval.

    This state is recomputed on every trajectory/objective snapshot, regardless
    of whether field diagnostics were reused or skipped by stride policy. The
    hard minimum-distance gate must never depend on diagnostic caching.
    """
    coil_coil_distance = host_float(context.Jccdist.shortest_distance())
    coil_distance_penalty = host_float(term_entries["coil_distance_penalty"]["raw_J"])
    cc_threshold = host_float(context.cc_threshold)
    violated = coil_coil_distance <= cc_threshold or not np.isfinite(
        coil_distance_penalty
    )
    return Stage2DistanceConstraintState(
        coil_coil_distance=coil_coil_distance,
        coil_distance_penalty=coil_distance_penalty,
        cc_threshold=cc_threshold,
        violated=bool(violated),
    )


def evaluate_stage2_objective(
    context,
    *,
    diagnostics=None,
    recompute_diagnostics=True,
):
    """Return composite objective diagnostics using the currently loaded DOFs."""
    term_entries, curve_length, curvature = _build_stage2_explicit_term_payload(context)
    grad = sum(np.asarray(entry["dJ"], dtype=float) for entry in term_entries.values())
    J = sum(host_float(entry["J"]) for entry in term_entries.values())
    squared_flux = host_float(term_entries["squared_flux"]["raw_J"])
    if recompute_diagnostics or diagnostics is None:
        diagnostics = compute_stage2_field_diagnostics(
            context.new_bs,
            context.new_surf,
            bs_jax=context.bs_jax,
        )
    distance_state = compute_stage2_distance_constraint_state(
        context,
        term_entries=term_entries,
    )
    diagnostics["coil_coil_distance"] = distance_state.coil_coil_distance
    snapshot = {
        "J": J,
        "Jf": squared_flux,
        "mean_abs_relBfinal_norm": host_float(diagnostics["mean_abs_relBfinal_norm"]),
        "curve_length": curve_length,
        "coil_coil_distance": distance_state.coil_coil_distance,
        "curvature": curvature,
        "grad_norm": float(np.linalg.norm(grad)),
        "distance_constraint_violated": distance_state.violated,
    }
    return snapshot, grad, diagnostics


def profile_stage2_explicit_step(
    context,
):
    """Return a one-step timing breakdown for the explicit Stage 2 objective."""
    objective_path_timings = {
        "JF_J_s": time_stage2_callback(context.JF.J),
        "JF_dJ_s": time_stage2_callback(context.JF.dJ),
    }

    start = time.perf_counter()
    snapshot, gradient, _ = evaluate_stage2_objective(
        context,
    )
    observed_step_total_s = float(time.perf_counter() - start)

    field_bs = resolve_stage2_field_bs(context.new_bs, context.bs_jax)
    extra_diagnostic_callbacks = {
        "Jf_J_s": lambda: host_float(context.Jf.J()),
        "mean_abs_relBfinal_norm_s": lambda: compute_mean_abs_relbn(
            context.new_surf, field_bs
        ),
        "curve_length_s": lambda: host_float(context.Jls.J()),
        "coil_coil_distance_s": lambda: host_float(context.Jccdist.shortest_distance()),
        "curvature_s": lambda: compute_stage2_max_curvature_value(context.Jc),
    }
    extra_diagnostic_timings = profile_stage2_named_callbacks(
        extra_diagnostic_callbacks
    )
    term_profile = profile_stage2_objective_terms(context)

    extra_diagnostic_total_s, dominant_extra_diagnostics = (
        build_stage2_profile_breakdown(extra_diagnostic_timings)
    )
    (
        squared_flux_internal_timings,
        squared_flux_internal_total_s,
        dominant_squared_flux_internal_components,
        squared_flux_field_b_vjp_component_timings,
        dominant_squared_flux_field_b_vjp_components,
        dominant_squared_flux_field_b_vjp_coils,
    ) = profile_stage2_squared_flux_internal_components(context.Jf)
    return {
        "objective_path_timings_s": objective_path_timings,
        "observed_step_total_s": observed_step_total_s,
        "extra_diagnostic_timings_s": extra_diagnostic_timings,
        "extra_diagnostic_total_s": extra_diagnostic_total_s,
        "dominant_extra_diagnostics": dominant_extra_diagnostics,
        "objective_term_value_timings_s": term_profile["value_timings_s"],
        "objective_term_value_total_s": term_profile["value_total_s"],
        "dominant_objective_value_terms": term_profile["dominant_value_terms"],
        "objective_term_gradient_timings_s": term_profile["gradient_timings_s"],
        "objective_term_gradient_total_s": term_profile["gradient_total_s"],
        "dominant_objective_gradient_terms": term_profile["dominant_gradient_terms"],
        "squared_flux_internal_timings_s": squared_flux_internal_timings,
        "squared_flux_internal_total_s": squared_flux_internal_total_s,
        "dominant_squared_flux_internal_components": dominant_squared_flux_internal_components,
        "squared_flux_field_b_vjp_component_timings_s": squared_flux_field_b_vjp_component_timings,
        "dominant_squared_flux_field_b_vjp_components": dominant_squared_flux_field_b_vjp_components,
        "dominant_squared_flux_field_b_vjp_coils": dominant_squared_flux_field_b_vjp_coils,
        "snapshot": snapshot,
    }


_STAGE2_COMPONENT_LABEL = "the Stage 2 outer loop"


def resolve_stage2_optimizer_contract(field_backend, optimizer_backend):
    """Resolve the optimizer contract for the Stage 2 outer loop."""
    from simsopt.geo.optimizer_jax import resolve_outer_loop_optimizer_contract

    return resolve_outer_loop_optimizer_contract(
        field_backend,
        optimizer_backend,
        component_label=_STAGE2_COMPONENT_LABEL,
    )


def resolve_stage2_optimizer_method(field_backend, optimizer_backend):
    """Resolve the shared optimizer substrate for the Stage 2 outer loop."""
    return resolve_stage2_optimizer_contract(field_backend, optimizer_backend).method


def should_build_stage2_target_objective(field_backend, optimizer_backend):
    """Return whether the scalar JAX Stage 2 objective should drive optimization."""
    return resolve_stage2_optimizer_contract(
        field_backend, optimizer_backend
    ).use_scalar_objective


def resolve_stage2_target_lane_requirements(
    field_backend,
    optimizer_backend,
    *,
    probe_only,
    export_objective_json,
):
    """Resolve target-lane requirements before building explicit objectives."""
    needs_target_probe_payload = (
        export_objective_json is not None and optimizer_backend == "ondevice"
    )
    probe_only_target_payload = (
        probe_only and needs_target_probe_payload and field_backend != "jax"
    )
    outer_contract = None
    use_scalar_objective = False
    if not probe_only_target_payload:
        outer_contract = resolve_stage2_optimizer_contract(
            field_backend,
            optimizer_backend,
        )
        use_scalar_objective = outer_contract.use_scalar_objective
    return (
        outer_contract,
        use_scalar_objective,
        needs_target_probe_payload,
        probe_only_target_payload,
    )


def validate_stage2_target_objective_dof_layout(
    target_objective_bundle,
    dofs,
):
    if target_objective_bundle.expected_dof_count != dofs.size:
        raise RuntimeError(STAGE2_TARGET_OBJECTIVE_DOF_LAYOUT_ERROR)


def resolve_stage2_field_diagnostic_stride(args):
    """Return the effective field-diagnostic refresh stride for the active lane."""
    requested_stride = int(args.field_diagnostic_stride)
    if requested_stride > 0:
        return requested_stride
    if args.backend == "jax" and args.optimizer_backend == "scipy":
        return 10
    return 1


def run_stage2_optimizer(
    value_and_grad_fun=None,
    dofs=None,
    *,
    contract,
    maxiter,
    ftol,
    gtol,
    maxcor=300,
    scalar_fun=None,
):
    """Run the Stage 2 outer optimization through the shared optimizer substrate."""
    from simsopt.geo.optimizer_jax import jax_minimize

    if contract.use_scalar_objective and scalar_fun is None:
        raise RuntimeError(
            "Stage 2 target-lane optimization requires a scalar JAX objective."
        )
    if (not contract.use_scalar_objective) and value_and_grad_fun is None:
        raise RuntimeError(
            "Stage 2 reference-lane optimization requires an explicit "
            "value-and-gradient objective."
        )
    return jax_minimize(
        scalar_fun if contract.use_scalar_objective else value_and_grad_fun,
        dofs,
        method=contract.method,
        tol=gtol,
        maxiter=maxiter,
        options={
            "maxcor": int(maxcor),
            "ftol": float(ftol),
        },
        value_and_grad=not contract.use_scalar_objective,
    )


def run_stage2_optimizer_timed(
    value_and_grad_fun=None,
    dofs=None,
    *,
    contract,
    maxiter,
    ftol,
    gtol,
    maxcor=300,
    scalar_fun=None,
):
    """Run the Stage 2 optimizer and return both result and elapsed time."""
    start = time.perf_counter()
    result = run_stage2_optimizer(
        value_and_grad_fun,
        dofs,
        contract=contract,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        maxcor=maxcor,
        scalar_fun=scalar_fun,
    )
    return result, float(time.perf_counter() - start)


def _build_stage2_probe_composite_payload(
    context,
    *,
    target_objective_bundle=None,
):
    """Return the probe composite snapshot/gradient using the active optimizer SSOT."""
    explicit_snapshot, explicit_grad, _ = evaluate_stage2_objective(
        context,
    )
    objective_source = "explicit-composite"
    composite_value = host_float(explicit_snapshot["J"])
    composite_grad = host_array(explicit_grad)
    composite_terms = None
    if target_objective_bundle is not None:
        composite_value, composite_grad = jax.value_and_grad(
            target_objective_bundle.objective
        )(host_array(context.JF.x))
        composite_value = host_float(composite_value)
        composite_grad = host_array(composite_grad)
        objective_source = "target-objective"
        composite_terms = _build_stage2_target_term_payload(
            target_objective_bundle,
            host_array(context.JF.x),
        )
    return (
        {
            **explicit_snapshot,
            "J": composite_value,
            "grad_norm": float(np.linalg.norm(composite_grad)),
            "objective_source": objective_source,
        },
        composite_grad,
        composite_terms,
    )


def build_stage2_probe_payload(
    JF,
    new_bs,
    new_surf,
    banana_curve,
    Jf,
    Jls,
    Jccdist,
    Jc,
    bs_jax=None,
    *,
    backend,
    optimizer_backend,
    equilibrium_path,
    nphi,
    ntheta,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    target_objective_bundle=None,
):
    """Serialize the initialized Stage 2 objective state for parity probes."""
    context = make_stage2_objective_context(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        squared_flux_weight,
        length_weight,
        length_target,
        cc_weight,
        cc_threshold,
        curvature_weight,
        bs_jax=bs_jax,
    )
    composite_snapshot, composite_grad, composite_terms = (
        _build_stage2_probe_composite_payload(
            context,
            target_objective_bundle=target_objective_bundle,
        )
    )
    flux_grad = host_array(Jf.dJ())
    curvature_threshold = host_float(context.Jc.threshold)
    curvature = host_float(composite_snapshot["curvature"])
    return {
        "backend": backend,
        "optimizer_backend": optimizer_backend,
        "banana_curve_class": type(banana_curve).__name__,
        "equilibrium_path": equilibrium_path,
        "nphi": int(nphi),
        "ntheta": int(ntheta),
        "dof_count": int(composite_grad.size),
        "curvature_threshold": curvature_threshold,
        "curvature_within_threshold": stage2_curvature_within_threshold(
            curvature,
            curvature_threshold,
        ),
        "curvature_margin": curvature_threshold - curvature,
        "squared_flux": {
            "J": host_float(Jf.J()),
            "dJ": flux_grad.tolist(),
            "grad_norm": float(np.linalg.norm(flux_grad)),
        },
        "composite": {
            **composite_snapshot,
            "dJ": composite_grad.tolist(),
            **({"terms": composite_terms} if composite_terms is not None else {}),
        },
    }


def write_json_file(path, payload):
    """Write JSON payloads for probe/export workflows."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def load_stage2_override_dofs(path, expected_shape):
    """Load a Stage 2 DOF override vector from JSON and validate its shape."""
    with open(path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)
    dofs = np.asarray(payload, dtype=float)
    if dofs.shape != expected_shape:
        raise ValueError(
            "Override DOF vector shape mismatch: "
            f"expected {expected_shape}, got {dofs.shape}."
        )
    return dofs


def append_stage2_trajectory_snapshot(trajectory_sink, snapshot, *, eval_index=None):
    """Append a Stage 2 diagnostic snapshot when trajectory capture is enabled."""
    if trajectory_sink is None:
        return
    trajectory_sink.append(
        {
            "eval_index": len(trajectory_sink) + 1
            if eval_index is None
            else int(eval_index),
            **snapshot,
        }
    )


def capture_stage2_trajectory_snapshot(
    trajectory_sink,
    JF,
    new_bs,
    new_surf,
    Jf,
    Jls,
    Jccdist,
    Jc,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    *,
    bs_jax=None,
):
    """Evaluate and append a Stage 2 diagnostic snapshot when requested."""
    context = make_stage2_objective_context(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        squared_flux_weight,
        length_weight,
        length_target,
        cc_weight,
        cc_threshold,
        curvature_weight,
        bs_jax=bs_jax,
    )
    snapshot, _, _ = evaluate_stage2_objective(
        context,
    )
    append_stage2_trajectory_snapshot(trajectory_sink, snapshot)
    return snapshot


def make_fun(
    JF,
    new_bs,
    new_surf,
    Jf,
    Jls,
    Jccdist,
    Jc,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    bs_jax=None,
    trajectory_sink=None,
    field_diagnostic_stride=1,
):
    """Factory for the Stage 2 objective function.

    Returns a closure compatible with ``simsopt.geo.optimizer_jax.jax_minimize``
    that captures all required state explicitly rather than from module scope.

    When *bs_jax* is provided the logging ⟨B·n⟩ diagnostic is computed
    from the JAX field (avoids stale CPU cache and redundant CPU field
    evaluation inside the optimizer loop).
    """
    eval_counter = {"count": 0}
    diagnostic_cache = {"snapshot": None}
    context = make_stage2_objective_context(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        squared_flux_weight,
        length_weight,
        length_target,
        cc_weight,
        cc_threshold,
        curvature_weight,
        bs_jax=bs_jax,
    )

    def fun(dofs):
        next_eval = eval_counter["count"] + 1
        recompute_diagnostics = (
            diagnostic_cache["snapshot"] is None
            or field_diagnostic_stride <= 1
            or next_eval == 1
            or next_eval % field_diagnostic_stride == 0
        )
        JF.x = dofs
        snapshot, grad, diagnostic_cache["snapshot"] = evaluate_stage2_objective(
            context,
            diagnostics=diagnostic_cache["snapshot"],
            recompute_diagnostics=recompute_diagnostics,
        )
        eval_counter["count"] = next_eval
        append_stage2_trajectory_snapshot(
            trajectory_sink,
            snapshot,
            eval_index=next_eval,
        )
        outstr = f"J={snapshot['J']:.1e}, Jf={snapshot['Jf']:.1e}, ⟨B·n⟩={snapshot['mean_abs_relBfinal_norm']:.1e}"
        outstr += f", Len={snapshot['curve_length']:.1f}m"
        outstr += f", C-C-Sep={snapshot['coil_coil_distance']:.2f}m"
        outstr += f", Curvature={snapshot['curvature']:.2f}"
        outstr += f", ║∇J║={snapshot['grad_norm']:.1e}"
        print(outstr)
        return snapshot["J"], grad

    return fun


if __name__ == "__main__":
    # PRE-INITIALIZATION
    # ---------------------------------------------------------------------------------------
    args = parse_args()
    args.curvature_threshold = resolve_curvature_threshold(args.curvature_threshold)

    # Deferred imports — simsopt modules require simsoptpp (C++ extension)
    # for coil/surface setup.  Placing them after arg parsing lets --help work
    # even when simsoptpp is not installed.
    from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
    from simsopt.field.coil import ScaledCurrent
    from simsopt.geo import (
        SurfaceRZFourier,
        curves_to_vtk,
        create_equally_spaced_curves,
        CurveLength,
        CurveCurveDistance,
        LpCurveCurvature,
        CurveCWSFourierCPP,
    )
    from simsopt.objectives import SquaredFlux, QuadraticPenalty
    from simsopt.objectives.stage2_target_objective_jax import (
        Stage2PenaltyConfig,
        build_stage2_target_objective,
    )
    from plotting_utils import (
        norm_field_plot,
        magnitude_field_plot,
        cross_section_plot,
    )

    # File for the desired boundary magnetic surface:
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)

    # Make Directory for output
    OUT_DIR = os.path.join(args.output_root, f"outputs-{plasma_surf_filename}") + "/"
    os.makedirs(OUT_DIR, exist_ok=True)

    # The proposed new HBT LCFS
    hbt = SurfaceRZFourier(nfp=5, stellsym=True)
    hbt.set_rc(0, 0, 0.9115)  # R0 of LCFS semi-circle center
    hbt.set_rc(1, 0, 0.1605)  # Minor radius (thick metal walls)
    hbt.set_zs(1, 0, 0.152)  # Z extent = ±0.152 m (flat top/bottom)

    nphi = args.nphi
    ntheta = args.ntheta

    # The surface the coils can lie on from Jeff - R0 = 0.976 and a~=0.22
    banana_surf_radius = args.banana_surf_radius
    banana_surf_nfp = 5
    surf_coils = SurfaceRZFourier(nfp=banana_surf_nfp, stellsym=True)
    surf_coils.set_rc(0, 0, 0.976)
    surf_coils.set_rc(1, 0, banana_surf_radius)
    surf_coils.set_zs(1, 0, banana_surf_radius)

    # The outer vacuum vessel of HBT, R0 = 0.976, a = 0.222
    # Solely for visualization purposes
    VV = SurfaceRZFourier(nfp=5, stellsym=True)
    VV.set_rc(0, 0, 0.976)
    VV.set_rc(1, 0, 0.222)
    VV.set_zs(1, 0, 0.222)

    # Create the TF coils in HBT - these will be fixed but create background toroidal field:
    tf_curves = create_equally_spaced_curves(
        20, 1, stellsym=False, R0=0.976, R1=0.4, order=1
    )
    tf_currents = [
        Current(1.0) * 1e5 for i in range(20)
    ]  # At some point, update with actual HBT TF current

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

    num_quadpoints = args.num_quadpoints  # number of quadrature points for coils
    order = args.order  # number of Fourier modes for coils

    R0 = args.major_radius  # major radius
    s = args.toroidal_flux  # VMEC flux-surface label

    new_surf = initSurface(R0, s, file_loc, nphi, ntheta)
    init_coil_array = initializeCoils(
        new_surf,
        surf_coils,
        tf_coils,
        num_quadpoints,
        order,
        phi_center,
        theta_center,
        phi_width,
        theta_width,
        OUT_DIR,
    )
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
    LENGTH_TARGET = max(args.length_target, 1.75)  # Hardware minimum: 1.75m

    # Threshold and weight for the coil-to-coil distance penalty
    CC_THRESHOLD = max(
        args.cc_threshold, 0.05
    )  # Hardware minimum: 5cm coil-coil spacing
    CC_WEIGHT = args.cc_weight

    # Threshold and weight for the coil curvature penalty
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = args.curvature_threshold
    SQUARED_FLUX_WEIGHT = args.squared_flux_weight

    (
        outer_contract,
        use_scalar_objective,
        needs_target_probe_payload,
        probe_only_target_payload,
    ) = resolve_stage2_target_lane_requirements(
        args.backend,
        args.optimizer_backend,
        probe_only=args.probe_only,
        export_objective_json=args.export_objective_json,
    )

    target_objective_bundle = None
    needs_target_objective_bundle = use_scalar_objective or needs_target_probe_payload
    if needs_target_objective_bundle:
        target_objective_bundle = build_stage2_target_objective(
            surface=new_surf,
            tf_coils=tf_coils,
            banana_coils=new_banana_coils,
            banana_curve=new_banana_curve,
            penalty_config=Stage2PenaltyConfig(
                squared_flux_weight=SQUARED_FLUX_WEIGHT,
                length_weight=LENGTH_WEIGHT,
                length_target=LENGTH_TARGET,
                cc_weight=CC_WEIGHT,
                cc_threshold=CC_THRESHOLD,
                curvature_weight=CURVATURE_WEIGHT,
                curvature_threshold=CURVATURE_THRESHOLD,
                curvature_p_norm=args.curvature_p_norm,
            ),
        )

    # Define the individual terms objective function:
    new_bs_jax = None
    if args.backend == "jax":
        from simsopt.field import BiotSavartJAX
        from simsopt.objectives import SquaredFluxJAX

        all_coils = tf_coils + list(new_banana_coils)
        new_bs_jax = BiotSavartJAX(all_coils)
        Jf = SquaredFluxJAX(new_surf, new_bs_jax)  # JAX forward + autodiff gradient
        print("Stage 2 backend: JAX")
    else:
        Jf = SquaredFlux(new_surf, new_bs)  # penalty on B dot n
        print("Stage 2 backend: CPU (simsoptpp)")
    Jls = CurveLength(new_banana_curve)  # penalty on curve length
    Jccdist = CurveCurveDistance(
        new_curves, CC_THRESHOLD
    )  # penalty on coil-to-coil distance

    # Lp-norm curvature penalty (configurable via --curvature-p-norm)
    Jc = LpCurveCurvature(
        new_banana_curve,
        args.curvature_p_norm,
        CURVATURE_THRESHOLD,
    )
    print(f"Initial coil length: {host_float(Jls.J()):.2f} [m]")
    Jls_penalty = QuadraticPenalty(Jls, LENGTH_TARGET, "max")

    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    JF = (
        SQUARED_FLUX_WEIGHT * Jf
        + LENGTH_WEIGHT * Jls_penalty
        + CC_WEIGHT * Jccdist
        + CURVATURE_WEIGHT * Jc
    )

    OUT_DIR_ITER = f"{OUT_DIR}R0={R0:g}-s={s:g}-LW={LENGTH_WEIGHT:g}-CCW={CC_WEIGHT:g}-CCT={CC_THRESHOLD:g}-CW={CURVATURE_WEIGHT:g}-CT={CURVATURE_THRESHOLD:g}-SR={banana_surf_radius:0.3f}-Order={order}-NQ={num_quadpoints}-CP={args.curvature_p_norm}-SFW={SQUARED_FLUX_WEIGHT:g}-backend={args.backend}/"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # minimize gets called, optimizes based on degrees of freedom from objective function
    dofs = JF.x
    if args.override_dofs_json is not None:
        dofs = load_stage2_override_dofs(
            args.override_dofs_json, np.asarray(dofs).shape
        )
        JF.x = np.asarray(dofs, dtype=float)
    if target_objective_bundle is not None:
        validate_stage2_target_objective_dof_layout(
            target_objective_bundle,
            dofs,
        )
    trajectory: list[dict[str, object]] | None = [] if args.trajectory_json else None
    final_snapshot = None
    optimizer_timings = None
    if args.record_warm_timings and not use_scalar_objective:
        raise ValueError(
            "--record-warm-timings is only supported on the JAX Stage 2 ondevice lane."
        )
    if args.profile_step_json is not None and use_scalar_objective:
        raise ValueError(
            "--profile-step-json is only supported on explicit Stage 2 reference lanes."
        )
    if args.record_warm_timings and (args.probe_only or args.init_only):
        raise ValueError(
            "--record-warm-timings requires an actual Stage 2 optimization run and "
            "cannot be combined with --probe-only or --init-only."
        )
    field_diagnostic_stride = resolve_stage2_field_diagnostic_stride(args)
    fun = None
    if not use_scalar_objective:
        fun = make_fun(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
            trajectory_sink=trajectory,
            field_diagnostic_stride=field_diagnostic_stride,
        )
    if args.export_objective_json:
        probe_payload = build_stage2_probe_payload(
            JF,
            new_bs,
            new_surf,
            new_banana_curve,
            Jf,
            Jls,
            Jccdist,
            Jc,
            bs_jax=new_bs_jax,
            backend=args.backend,
            optimizer_backend=args.optimizer_backend,
            equilibrium_path=file_loc,
            nphi=nphi,
            ntheta=ntheta,
            squared_flux_weight=SQUARED_FLUX_WEIGHT,
            length_weight=LENGTH_WEIGHT,
            length_target=LENGTH_TARGET,
            cc_weight=CC_WEIGHT,
            cc_threshold=CC_THRESHOLD,
            curvature_weight=CURVATURE_WEIGHT,
            target_objective_bundle=(
                target_objective_bundle
                if args.optimizer_backend == "ondevice"
                else None
            ),
        )
        write_json_file(args.export_objective_json, probe_payload)
        print(f"Wrote Stage 2 objective snapshot to {args.export_objective_json}")
    if args.profile_step_json is not None:
        profile_context = make_stage2_objective_context(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
        )
        profile_payload = profile_stage2_explicit_step(profile_context)
        write_json_file(args.profile_step_json, profile_payload)
        print(f"Wrote Stage 2 step profile to {args.profile_step_json}")
    if args.probe_only:
        if args.trajectory_json:
            write_json_file(
                args.trajectory_json,
                {"backend": args.backend, "evaluations": trajectory or []},
            )
        print(
            "Probe-only mode requested; exiting before optimization and post-processing."
        )
        sys.exit(0)
    if args.init_only:
        res_nit = 0
        print("Skipping Stage 2 optimizer because --init-only was provided.")
    else:
        initial_dofs = host_array(dofs).copy()
        assert outer_contract is not None
        if use_scalar_objective:
            capture_stage2_trajectory_snapshot(
                trajectory,
                JF,
                new_bs,
                new_surf,
                Jf,
                Jls,
                Jccdist,
                Jc,
                SQUARED_FLUX_WEIGHT,
                LENGTH_WEIGHT,
                LENGTH_TARGET,
                CC_WEIGHT,
                CC_THRESHOLD,
                CURVATURE_WEIGHT,
                bs_jax=new_bs_jax,
            )
        res, cold_elapsed_s = run_stage2_optimizer_timed(
            fun,
            dofs,
            contract=outer_contract,
            maxiter=MAXITER,
            maxcor=300,
            ftol=args.ftol,
            gtol=args.gtol,
            scalar_fun=(
                None
                if target_objective_bundle is None
                else target_objective_bundle.objective
            ),
        )
        JF.x = host_array(res.x)
        res_nit = res.nit
        if use_scalar_objective:
            assert target_objective_bundle is not None
            final_snapshot = capture_stage2_trajectory_snapshot(
                trajectory,
                JF,
                new_bs,
                new_surf,
                Jf,
                Jls,
                Jccdist,
                Jc,
                SQUARED_FLUX_WEIGHT,
                LENGTH_WEIGHT,
                LENGTH_TARGET,
                CC_WEIGHT,
                CC_THRESHOLD,
                CURVATURE_WEIGHT,
                bs_jax=new_bs_jax,
            )
            optimizer_timings = {
                "cold_run_s": float(cold_elapsed_s),
            }
            if args.record_warm_timings:
                JF.x = initial_dofs
                _, warm_elapsed_s = run_stage2_optimizer_timed(
                    None,
                    initial_dofs,
                    contract=outer_contract,
                    maxiter=MAXITER,
                    maxcor=300,
                    ftol=args.ftol,
                    gtol=args.gtol,
                    scalar_fun=target_objective_bundle.objective,
                )
                optimizer_timings["warm_run_s"] = float(warm_elapsed_s)
                optimizer_timings["compile_overhead_s"] = max(
                    float(cold_elapsed_s) - float(warm_elapsed_s),
                    0.0,
                )
                JF.x = host_array(res.x)
        print(res.message)
    if args.trajectory_json:
        write_json_file(
            args.trajectory_json,
            {"backend": args.backend, "evaluations": trajectory or []},
        )
        print(f"Wrote Stage 2 trajectory to {args.trajectory_json}")

    # POST-OPTIMIZATION PROCESSING AND OUTPUTS
    # Uses CPU new_bs intentionally: VTK, JSON save, and matplotlib all
    # require numpy arrays.  set_points() below forces fresh evaluation
    # with the optimized coil DOFs (shared coil objects).
    # ---------------------------------------------------------------------------------------
    if is_self_intersecting(new_banana_curve):
        print("BANANA COIL IS SELF-INTERSECTING!")
        intersecting = True

    new_bs.set_points(new_surf.gamma().reshape((-1, 3)))
    unitn = new_surf.unitnormal()
    B_field = new_bs.B().reshape(unitn.shape)
    pointData = {
        "B_N/B": np.sum(B_field * unitn, axis=2)[:, :, None]
        / np.sqrt(np.sum(B_field**2, axis=2))[:, :, None]
    }
    if args.skip_postprocess and final_snapshot is None:
        final_context = make_stage2_objective_context(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
        )
        final_snapshot, _, _ = evaluate_stage2_objective(
            final_context,
        )
    fieldError = compute_mean_abs_relbn(new_surf, new_bs)
    if args.skip_postprocess:
        print(
            "Skipping Stage 2 post-processing artifacts because --skip-postprocess was provided."
        )
    else:
        curves_to_vtk(new_curves, OUT_DIR_ITER + "curves_opt", close=True)
        new_surf.to_vtk(OUT_DIR_ITER + "surf_opt", extra_data=pointData)
        VV.to_vtk(OUT_DIR_ITER + "VV")

        # Create toroidal cross section plot
        cross_section_plot(
            new_surf_coils,
            new_surf,
            new_banana_curve,
            OUT_DIR_ITER + "CrossSectionPlot",
            hbt,
            VV,
        )
        # Create field error plot
        fieldError = magneticFieldPlots(new_surf, new_bs, OUT_DIR_ITER)

        # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
        new_bs.save(OUT_DIR_ITER + "biot_savart_opt.json")
    # new_surf.save(OUT_DIR_ITER + "surf_opt.json");
    print(
        f"Banana Coil Current / TF Current = {new_banana_coils[0].current.get_value() / new_tf_coils[0].current.get_value():.3f}\n"
    )

    # Save the results of optimization to a separate file
    if final_snapshot is None:
        final_context = make_stage2_objective_context(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
        )
        final_snapshot, _, _ = evaluate_stage2_objective(
            final_context,
        )
    results = {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "CC_THRESHOLD": CC_THRESHOLD,
        "CC_WEIGHT": CC_WEIGHT,
        "CURVATURE_WEIGHT": CURVATURE_WEIGHT,
        "CURVATURE_THRESHOLD": CURVATURE_THRESHOLD,
        "LENGTH_WEIGHT": LENGTH_WEIGHT,
        "theta_center": theta_center,
        "phi_center": phi_center,
        "theta_width": theta_width,
        "phi_width": phi_width,
        "LENGTH_TARGET": LENGTH_TARGET,
        "MAJOR_RADIUS": R0,
        "TOROIDAL_FLUX": s,
        "banana_surf_radius": banana_surf_radius,
        "order": order,
        "num_quadpoints": num_quadpoints,
        "curvature_p_norm": args.curvature_p_norm,
        "squared_flux_weight": SQUARED_FLUX_WEIGHT,
        "backend": args.backend,
        "optimizer_backend": args.optimizer_backend,
        "field_diagnostic_stride": int(field_diagnostic_stride),
        "banana_curve_class": type(new_banana_curve).__name__,
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "iterations": res_nit,
        "FINAL_DOFS": host_array(JF.x).tolist(),
        "FINAL_OBJECTIVE": final_snapshot["J"],
        "FINAL_SQUARED_FLUX": final_snapshot["Jf"],
        "FINAL_CURVE_LENGTH": final_snapshot["curve_length"],
        "FINAL_CC_DISTANCE": final_snapshot["coil_coil_distance"],
        "FINAL_MEAN_ABS_RELBN": final_snapshot["mean_abs_relBfinal_norm"],
        "FINAL_VOLUME": float(new_surf.volume()),
        "FIELD_ERROR": float(fieldError),
        "SELF_INTERSECTING": intersecting,
        "MAX_CURVATURE": float(np.max(new_banana_curve.kappa())),
        "FINAL_BANANA_GAMMA": np.asarray(
            new_banana_curve.gamma(), dtype=float
        ).tolist(),
    }
    if optimizer_timings is not None:
        results["OPTIMIZER_TIMINGS"] = optimizer_timings
    write_json_file(os.path.join(OUT_DIR_ITER, "results.json"), results)
