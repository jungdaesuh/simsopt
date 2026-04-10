import numpy as np

try:
    from numba import njit
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier, curves_to_vtk

from plotting_utils import magnitude_field_plot, norm_field_plot


def init_surface(R0, s, file_loc, nphi, ntheta):
    surf = SurfaceRZFourier.from_wout(
        file_loc, range="full torus", nphi=nphi, ntheta=ntheta, s=s
    )
    surf.set_dofs(surf.get_dofs() * R0 / surf.major_radius())
    print("Major radius target: ", R0)
    print("Major radius actual: ", surf.major_radius())
    print("Minor radius: ", surf.minor_radius())
    return surf


def initialize_coils(
    surf,
    surf_coils,
    tf_coils,
    num_quadpoints,
    order,
    banana_init_current_A,
    phi_center,
    theta_center,
    phi_width,
    theta_width,
    out_dir,
):
    banana_curve = CurveCWSFourierCPP(
        np.linspace(0, 1, num_quadpoints, endpoint=False),
        order=order,
        surf=surf_coils,
    )
    banana_curve.set("phic(0)", phi_center)
    banana_curve.set("thetac(0)", theta_center)
    banana_curve.set("phic(1)", phi_width)
    banana_curve.set("thetas(1)", theta_width)

    banana_coils = coils_via_symmetries(
        [banana_curve],
        [ScaledCurrent(Current(1), banana_init_current_A)],
        surf_coils.nfp,
        surf_coils.stellsym,
    )

    coils = tf_coils + banana_coils
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    curves = [coil.curve for coil in coils]
    curves_to_vtk(curves, out_dir + "curves_init", close=True)
    unitn = surf.unitnormal()
    point_data = {
        "B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]
    }
    surf.to_vtk(out_dir + "surf_init", extra_data=point_data)
    return bs, curves, banana_curve, banana_coils


def gamma_at_t(curve, t):
    g2 = np.zeros((len(t), 2))
    curve.gamma_2d_impl(g2, t)
    out = np.zeros((len(t), 3))
    curve.surf.gamma_lin(out, g2[:, 0], g2[:, 1])
    return out


def compute_curve_length(pts):
    diffs = pts[1:] - pts[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(seg_lengths)


@njit
def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@njit
def segment_segment_distance(P1, P2, Q1, Q2):
    u = P2 - P1
    v = Q2 - Q1
    w0 = P1 - Q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    zero_len = 1e-30
    par_eps = 1e-10

    if a < zero_len:
        if c < zero_len:
            return np.linalg.norm(w0)
        return np.linalg.norm(w0 - _clamp01(e / c) * v)

    if c < zero_len:
        return np.linalg.norm(w0 + _clamp01(-d / a) * u)

    denom = a * c - b * b

    if denom < par_eps * a * c:
        best_sq = np.inf

        dp = w0 - _clamp01(e / c) * v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        dp = w0 + u - _clamp01((e + b) / c) * v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        dp = w0 + _clamp01(-d / a) * u
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        dp = w0 + _clamp01((b - d) / a) * u - v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq

        if denom > 0.0:
            sc_int = (b * e - c * d) / denom
            tc_int = (a * e - b * d) / denom
            if 0.0 <= sc_int <= 1.0 and 0.0 <= tc_int <= 1.0:
                dp = w0 + sc_int * u - tc_int * v
                dsq = np.dot(dp, dp)
                if dsq < best_sq:
                    best_sq = dsq

        return np.sqrt(best_sq)

    sc = (b * e - c * d) / denom
    tc = (a * e - b * d) / denom

    if sc < 0.0:
        sc = 0.0
        tc = e / c
    elif sc > 1.0:
        sc = 1.0
        tc = (e + b) / c

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


def is_self_intersecting(curve, npts=2000, tol_factor=0.1, neighbor_skip=3):
    t = np.linspace(0, 1, npts + 1)
    pts = gamma_at_t(curve, t)

    segments = np.zeros((npts, 2, 3))
    for i in range(npts):
        segments[i, 0] = pts[i]
        segments[i, 1] = pts[i + 1]

    total_length = compute_curve_length(pts)
    seg_length = total_length / npts
    tol = tol_factor * seg_length
    return check_all_pairs(segments, tol, neighbor_skip)


def magnetic_field_plots(surf, bs, out_dir_iter):
    mean_abs_relBfinal_norm, modBfinal, surf_area, phi, theta = norm_field_plot(
        surf, bs, out_dir_iter + "NormFieldPlot"
    )
    magnitude_field_plot(modBfinal, surf_area, phi, theta, out_dir_iter + "MagFieldPlot")
    return mean_abs_relBfinal_norm
