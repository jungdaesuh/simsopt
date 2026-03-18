import argparse
import os
import sys
import json

import numpy as np
from scipy.optimize import minimize
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, EXAMPLE_ROOT)

SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = (
    DATABASE_EQUILIBRIA_DIR
    if os.path.isdir(DATABASE_EQUILIBRIA_DIR)
    else os.path.join(EXAMPLE_ROOT, "equilibria")
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
        "--backend",
        choices=["cpu", "jax"],
        default=os.environ.get("SIMSOPT_BACKEND")
        or os.environ.get("STAGE2_BACKEND", "cpu"),
        help="Field backend: 'cpu' (simsoptpp) or 'jax' (JAX Biot-Savart). "
        "Env: SIMSOPT_BACKEND (or legacy STAGE2_BACKEND).",
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
    3D self-intersection checker for CurveCWSFourier objects.

    Parameters:
        curve: CurveCWSFourier object
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


def make_fun(JF, new_bs, new_surf, Jf, Jls, Jccdist, Jc, bs_jax=None):
    """Factory for the Stage 2 objective function.

    Returns a closure compatible with scipy.optimize.minimize(jac=True)
    that captures all required state explicitly rather than from module scope.

    When *bs_jax* is provided the logging ⟨B·n⟩ diagnostic is computed
    from the JAX field (avoids stale CPU cache and redundant CPU field
    evaluation inside the optimizer loop).
    """

    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        unitn = new_surf.unitnormal()
        if bs_jax is not None:
            B = np.asarray(bs_jax.B()).reshape(unitn.shape)
        else:
            B = new_bs.B().reshape(unitn.shape)
        BdotN = np.mean(np.abs(np.sum(B * unitn, axis=2)))
        outstr = f"J={J:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
        outstr += f", Len={Jls.J():.1f}m"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}m"
        outstr += f", Curvature={Jc.J():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad

    return fun


if __name__ == "__main__":
    # PRE-INITIALIZATION
    # ---------------------------------------------------------------------------------------
    args = parse_args()

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
    surf = None

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

    num_quadpoints = 128  # number of quadature points for coils
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
    LENGTH_TARGET = args.length_target

    # Threshold and weight for the coil-to-coil distance penalty
    CC_THRESHOLD = args.cc_threshold
    CC_WEIGHT = args.cc_weight

    # Threshold and weight for the coil curvature penalty
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = args.curvature_threshold

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

    # Changed p-norm of curvature penalty from 2 to 4 to prevent kinks/dents in the coils
    Jc = LpCurveCurvature(new_banana_curve, 4, CURVATURE_THRESHOLD)
    print(f"Initial coil length: {Jls.J():.2f} [m]")

    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    JF = (
        Jf
        + LENGTH_WEIGHT * QuadraticPenalty(Jls, LENGTH_TARGET, "max")
        + CC_WEIGHT * Jccdist
        + CURVATURE_WEIGHT * Jc
    )

    OUT_DIR_ITER = f"{OUT_DIR}R0={R0:g}-s={s:g}-LW={LENGTH_WEIGHT:g}-CCW={CC_WEIGHT:g}-CCT={CC_THRESHOLD:g}-CW={CURVATURE_WEIGHT:g}-CT={CURVATURE_THRESHOLD:g}-SR={banana_surf_radius:0.3f}-Order={order}-backend={args.backend}/"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # minimize gets called, optimizes based on degrees of freedom from objective function
    dofs = JF.x
    fun = make_fun(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        bs_jax=new_bs_jax,
    )
    if args.init_only:
        res_nit = 0
        print("Skipping Stage 2 optimizer because --init-only was provided.")
    else:
        res = minimize(
            fun,
            dofs,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": MAXITER,
                "maxcor": 300,
                "ftol": args.ftol,
                "gtol": args.gtol,
            },
        )
        res_nit = res.nit
        print(res.message)

    # POST-OPTIMIZATION PROCESSING AND OUTPUTS
    # Uses CPU new_bs intentionally: VTK, JSON save, and matplotlib all
    # require numpy arrays.  set_points() below forces fresh evaluation
    # with the optimized coil DOFs (shared coil objects).
    # ---------------------------------------------------------------------------------------
    if is_self_intersecting(new_banana_curve):
        print("BANANA COIL IS SELF-INTERSECTING!")
        intersecting = True

    curves_to_vtk(new_curves, OUT_DIR_ITER + "curves_opt", close=True)
    new_bs.set_points(new_surf.gamma().reshape((-1, 3)))
    unitn = new_surf.unitnormal()
    B_field = new_bs.B().reshape(unitn.shape)
    pointData = {
        "B_N/B": np.sum(B_field * unitn, axis=2)[:, :, None]
        / np.sqrt(np.sum(B_field**2, axis=2))[:, :, None]
    }
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
        "backend": args.backend,
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "iterations": res_nit,
        "FINAL_VOLUME": float(new_surf.volume()),
        "FIELD_ERROR": float(fieldError),
        "SELF_INTERSECTING": intersecting,
        "MAX_CURVATURE": float(np.max(new_banana_curve.kappa())),
    }
    with open(os.path.join(OUT_DIR_ITER, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)
