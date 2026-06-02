import logging
from dataclasses import dataclass

import numpy as np
from scipy.io import netcdf_file

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
from simsopt.geo import (
    CurveCWSFourierCPP,
    CurveXYZFourier,
    SurfaceRZFourier,
    create_multifilament_grid,
    curves_to_vtk,
)

from plotting_utils import magnitude_field_plot, norm_field_plot
from workflow_helpers import validate_normalized_toroidal_flux
from banana_opt.hardware_contracts import (
    validate_target_lcfs_major_radius,
    validate_target_lcfs_minor_radius,
)
from banana_opt.current_contracts import unwrap_current_optimizable
from banana_opt.current_contracts import VFCoilBuildResult
from banana_opt.current_contracts import resolve_jhalpern30_fresh_vf_current_A
from banana_opt.finite_current_profiles import JHALPERN30_FINITE_CURRENT_MODE
from banana_opt.finite_current_profiles import get_finite_current_profile
from banana_opt.jhalpern30_compat import (
    build_jhalpern30_banana_coils,
    build_jhalpern30_proxy_plasma_current_coils,
    build_jhalpern30_vf_coil_build_result,
    resolve_jhalpern30_vf_template_path,
)
from banana_opt.json_compat import load_boozer_finite_i as load


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlasmaGeometry:
    working_surface: SurfaceRZFourier
    lcfs_surface: SurfaceRZFourier
    working_major_radius_m: float
    working_minor_radius_m: float
    lcfs_major_radius_m: float
    lcfs_minor_radius_m: float
    scale_factor: float


@dataclass(frozen=True)
class PlasmaGeometryPreflightCandidate:
    s_working: float
    target_lcfs_major_radius_m: float
    scale_factor: float
    lcfs_major_radius_m: float
    lcfs_minor_radius_m: float
    plasma_vessel_min_dist_m: float
    violations: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class PlasmaGeometryPreflightResult:
    selected: PlasmaGeometryPreflightCandidate
    candidates: tuple[PlasmaGeometryPreflightCandidate, ...]


def load_vmec_surface(file_loc, s, nphi, ntheta):
    surface_label = validate_normalized_toroidal_flux(
        s,
        field_name="Stage 2 VMEC surface label s",
    )
    return SurfaceRZFourier.from_wout(
        file_loc,
        range="full torus",
        nphi=nphi,
        ntheta=ntheta,
        s=surface_label,
    )


_load_vmec_surface = load_vmec_surface


def _scale_surface(surface, scale_factor):
    surface.set_dofs(surface.get_dofs() * float(scale_factor))
    return surface


def _unique_float_candidates(values):
    candidates = []
    for value in values:
        candidate = float(value)
        if not any(abs(candidate - existing) < 1.0e-12 for existing in candidates):
            candidates.append(candidate)
    return tuple(candidates)


def default_geometry_preflight_s_candidates(s_working):
    requested_s = validate_normalized_toroidal_flux(
        s_working,
        field_name="Stage 2 requested VMEC surface label s",
    )
    return _unique_float_candidates(
        (
            requested_s,
            0.50,
            0.45,
            0.40,
            0.35,
            0.30,
            0.25,
            0.24,
            0.20,
        )
    )


def default_geometry_preflight_target_lcfs_major_radius_candidates(
    target_lcfs_major_radius_m,
):
    target = validate_target_lcfs_major_radius(target_lcfs_major_radius_m)
    values = [target]
    radius = target - 0.01
    while radius >= 0.80 - 1.0e-12:
        values.append(round(radius, 12))
        radius -= 0.01
    return _unique_float_candidates(values)


def scaled_surface_surface_min_distance(surface_a, scale_factor, surface_b):
    return float(
        np.min(
            np.linalg.norm(
                surface_a.gamma().reshape((-1, 1, 3)) * float(scale_factor)
                - surface_b.gamma().reshape((1, -1, 3)),
                axis=2,
            )
        )
    )


def select_plasma_geometry_preflight_candidate(
    *,
    lcfs_surface,
    requested_s,
    target_lcfs_major_radius_m,
    target_lcfs_minor_radius_m,
    vessel_surface,
    s_candidates=None,
    target_lcfs_major_radius_candidates_m=None,
    distance_fn=scaled_surface_surface_min_distance,
):
    requested_surface_label = validate_normalized_toroidal_flux(
        requested_s,
        field_name="Stage 2 requested VMEC surface label s",
    )
    max_target_major = validate_target_lcfs_major_radius(target_lcfs_major_radius_m)
    max_target_minor = validate_target_lcfs_minor_radius(target_lcfs_minor_radius_m)
    resolved_s_candidates = (
        default_geometry_preflight_s_candidates(requested_surface_label)
        if s_candidates is None
        else _unique_float_candidates(
            validate_normalized_toroidal_flux(
                candidate,
                field_name="Stage 2 geometry preflight s candidate",
            )
            for candidate in s_candidates
        )
    )
    resolved_target_candidates = (
        default_geometry_preflight_target_lcfs_major_radius_candidates(max_target_major)
        if target_lcfs_major_radius_candidates_m is None
        else _unique_float_candidates(
            validate_target_lcfs_major_radius(candidate)
            for candidate in target_lcfs_major_radius_candidates_m
        )
    )

    base_lcfs_major_radius = float(lcfs_surface.major_radius())
    base_lcfs_minor_radius = float(lcfs_surface.minor_radius())
    candidates = []
    for s_candidate in resolved_s_candidates:
        for target_candidate in resolved_target_candidates:
            scale_factor = float(target_candidate) / base_lcfs_major_radius
            lcfs_minor_radius = base_lcfs_minor_radius * scale_factor
            vessel_gap = float(distance_fn(lcfs_surface, scale_factor, vessel_surface))
            violations = []
            if target_candidate > max_target_major + 1.0e-12:
                violations.append(f"lcfs_major_radius>{max_target_major:.6f}")
            if lcfs_minor_radius > max_target_minor + 1.0e-12:
                violations.append(f"lcfs_minor_radius>{max_target_minor:.6f}")
            candidates.append(
                PlasmaGeometryPreflightCandidate(
                    s_working=float(s_candidate),
                    target_lcfs_major_radius_m=float(target_candidate),
                    scale_factor=float(scale_factor),
                    lcfs_major_radius_m=float(target_candidate),
                    lcfs_minor_radius_m=float(lcfs_minor_radius),
                    plasma_vessel_min_dist_m=float(vessel_gap),
                    violations=tuple(violations),
                )
            )

    successful = [candidate for candidate in candidates if candidate.success]
    if not successful:
        best = max(
            candidates,
            key=lambda candidate: (
                candidate.plasma_vessel_min_dist_m,
                -candidate.lcfs_minor_radius_m,
                candidate.target_lcfs_major_radius_m,
            ),
        )
        raise ValueError(
            "No Stage 2 plasma geometry preflight candidate fits the HBT-EP "
            "shell. Best candidate was "
            f"s={best.s_working:.6f}, "
            f"target_lcfs_major_radius_m={best.target_lcfs_major_radius_m:.6f}, "
            f"lcfs_minor_radius_m={best.lcfs_minor_radius_m:.6f}, "
            f"plasma_vessel_min_dist_m={best.plasma_vessel_min_dist_m:.6f}, "
            f"violations={list(best.violations)}."
        )

    selected = min(
        successful,
        key=lambda candidate: (
            -candidate.target_lcfs_major_radius_m,
            abs(candidate.s_working - requested_surface_label),
            candidate.s_working,
        ),
    )
    return PlasmaGeometryPreflightResult(
        selected=selected,
        candidates=tuple(candidates),
    )


def load_plasma_geometry(target_lcfs_major_radius_m, s_working, file_loc, nphi, ntheta):
    working_surface = load_vmec_surface(file_loc, s_working, nphi, ntheta)
    lcfs_surface = load_vmec_surface(file_loc, 1.0, nphi, ntheta)
    target_lcfs_major_radius = validate_target_lcfs_major_radius(
        target_lcfs_major_radius_m
    )
    scale_factor = target_lcfs_major_radius / float(lcfs_surface.major_radius())
    lcfs_surface = _scale_surface(
        lcfs_surface,
        scale_factor,
    )
    working_surface = _scale_surface(working_surface, scale_factor)
    LOGGER.info("LCFS major radius target: %s", target_lcfs_major_radius)
    LOGGER.info(
        "Working surface major radius actual: %s", working_surface.major_radius()
    )
    LOGGER.info("Working surface minor radius: %s", working_surface.minor_radius())
    LOGGER.info("LCFS major radius: %s", lcfs_surface.major_radius())
    LOGGER.info("LCFS minor radius: %s", lcfs_surface.minor_radius())
    return PlasmaGeometry(
        working_surface=working_surface,
        lcfs_surface=lcfs_surface,
        working_major_radius_m=float(working_surface.major_radius()),
        working_minor_radius_m=float(working_surface.minor_radius()),
        lcfs_major_radius_m=float(lcfs_surface.major_radius()),
        lcfs_minor_radius_m=float(lcfs_surface.minor_radius()),
        scale_factor=float(scale_factor),
    )


def load_plasma_geometry_for_working_major_radius(
    target_working_major_radius_m,
    s_working,
    file_loc,
    nphi,
    ntheta,
):
    working_surface = load_vmec_surface(file_loc, s_working, nphi, ntheta)
    lcfs_surface = load_vmec_surface(file_loc, 1.0, nphi, ntheta)
    target_working_major_radius = float(target_working_major_radius_m)
    if target_working_major_radius <= 0.0:
        raise ValueError("Target working-surface major radius must be positive.")
    scale_factor = target_working_major_radius / float(working_surface.major_radius())
    working_surface = _scale_surface(working_surface, scale_factor)
    lcfs_surface = _scale_surface(lcfs_surface, scale_factor)
    LOGGER.info("Working surface major radius target: %s", target_working_major_radius)
    LOGGER.info(
        "Working surface major radius actual: %s", working_surface.major_radius()
    )
    LOGGER.info("Working surface minor radius: %s", working_surface.minor_radius())
    LOGGER.info("LCFS major radius: %s", lcfs_surface.major_radius())
    LOGGER.info("LCFS minor radius: %s", lcfs_surface.minor_radius())
    return PlasmaGeometry(
        working_surface=working_surface,
        lcfs_surface=lcfs_surface,
        working_major_radius_m=float(working_surface.major_radius()),
        working_minor_radius_m=float(working_surface.minor_radius()),
        lcfs_major_radius_m=float(lcfs_surface.major_radius()),
        lcfs_minor_radius_m=float(lcfs_surface.minor_radius()),
        scale_factor=float(scale_factor),
    )


def init_surface(target_lcfs_major_radius_m, s, file_loc, nphi, ntheta):
    return load_plasma_geometry(
        target_lcfs_major_radius_m,
        s,
        file_loc,
        nphi,
        ntheta,
    ).working_surface


def surface_surface_min_distance(surface_a, surface_b):
    return float(
        np.min(
            np.linalg.norm(
                surface_a.gamma().reshape((-1, 1, 3))
                - surface_b.gamma().reshape((1, -1, 3)),
                axis=2,
            )
        )
    )


def build_proxy_plasma_current_coils(
    *,
    equilibrium_file: str,
    surface_scale_factor: float,
    nphi: int,
    ntheta: int,
    toroidal_flux: float,
    plasma_current_A: float,
) -> list[Coil]:
    with netcdf_file(equilibrium_file, mmap=False) as equilibrium_netcdf:
        raxis_cc = equilibrium_netcdf.variables["raxis_cc"][:].copy()
        zaxis_cs = equilibrium_netcdf.variables["zaxis_cs"][:].copy()
    validate_normalized_toroidal_flux(
        toroidal_flux,
        field_name="proxy plasma-current toroidal_flux",
    )
    axis_scale = float(surface_scale_factor)
    proxy_curve = CurveXYZFourier(128, 1)
    # Wataru proxy placement is the zeroth VMEC magnetic-axis coefficient,
    # scaled with the surface family. It intentionally is not the full Fourier
    # magnetic axis and differs from jhalpern30's surface-major-radius ring.
    proxy_curve.set("xc(1)", float(raxis_cc[0]) * axis_scale)
    proxy_curve.set("ys(1)", float(raxis_cc[0]) * axis_scale)
    proxy_curve.set("zc(0)", float(zaxis_cs[0]) * axis_scale)
    proxy_curve.fix_all()
    proxy_current = Current(float(plasma_current_A))
    proxy_current.fix_all()
    return [Coil(proxy_curve, proxy_current)]


def build_vf_coil_build_result(
    *,
    vf_current_A: float,
    vf_template_path: str,
    load_fn=load,
) -> VFCoilBuildResult:
    loaded_template = load_fn(vf_template_path)
    template_coils = getattr(loaded_template, "coils", None)
    if not template_coils:
        raise ValueError(
            f"VF template {vf_template_path!r} does not contain any coils."
        )
    vf_coils: list[Coil] = []
    for template_coil in template_coils:
        sign = float(np.sign(template_coil.current.get_value()))
        if sign == 0.0:
            raise ValueError(
                "VF template coils must carry non-zero signed currents so the "
                "Wataru VF sign convention can be preserved."
            )
        vf_curve = template_coil.curve
        vf_curve.fix_all()
        vf_current = Current(float(vf_current_A) * sign)
        vf_current.fix_all()
        vf_coils.append(Coil(vf_curve, vf_current))
    return VFCoilBuildResult(coils=vf_coils, current_control=None)


def build_vf_coils(
    *,
    vf_current_A: float,
    vf_template_path: str,
    load_fn=load,
) -> list[Coil]:
    return build_vf_coil_build_result(
        vf_current_A=vf_current_A,
        vf_template_path=vf_template_path,
        load_fn=load_fn,
    ).coils


def build_vf_coils_for_profile(
    *,
    finite_current_mode: str,
    proxy_current_A: float,
    vf_current_A: float,
    vf_template_path: str,
    load_fn=load,
) -> VFCoilBuildResult:
    profile = get_finite_current_profile(finite_current_mode)
    if profile.vf_current_mutability == "shared_unfixed_scaled_current":
        resolve_jhalpern30_fresh_vf_current_A(
            proxy_plasma_current_A=float(proxy_current_A),
            requested_vf_current_A=float(vf_current_A),
        )
        return build_jhalpern30_vf_coil_build_result(
            float(proxy_current_A),
            resolve_jhalpern30_vf_template_path(vf_template_path),
            load_fn=load_fn,
        )
    if profile.vf_current_mutability == "independent_fixed_current":
        return build_vf_coil_build_result(
            vf_current_A=float(vf_current_A),
            vf_template_path=str(vf_template_path),
            load_fn=load_fn,
        )
    raise ValueError(
        f"Unsupported VF current mutability {profile.vf_current_mutability!r}."
    )


def coerce_vf_coil_build_result(
    value,
    *,
    requires_current_control: bool = False,
) -> VFCoilBuildResult:
    if isinstance(value, VFCoilBuildResult):
        if requires_current_control and value.coils and value.current_control is None:
            raise ValueError("shared VF current profile requires a VF current control.")
        return value
    if requires_current_control:
        raise ValueError(
            "shared VF current profile requires VFCoilBuildResult, not a raw coil list."
        )
    return VFCoilBuildResult(coils=list(value), current_control=None)


def shared_vf_current_control_for_coils(vf_coils: list[Coil]) -> object | None:
    if not vf_coils:
        return None
    first_current = vf_coils[0].current
    shared_current, first_scale = unwrap_current_optimizable(first_current)
    for coil in vf_coils[1:]:
        current, scale = unwrap_current_optimizable(coil.current)
        if current is not shared_current:
            raise ValueError(
                "shared VF current profile requires all VF coils to share one "
                "leaf Current."
            )
        if not np.isclose(abs(scale), abs(first_scale), rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                "shared VF current profile requires equal-magnitude VF scale factors."
            )
    shared_parent = getattr(first_current, "current_to_scale", None)
    if shared_parent is not None and all(
        getattr(coil.current, "current_to_scale", None) is shared_parent
        for coil in vf_coils
    ):
        return shared_parent
    return first_current


@dataclass(frozen=True)
class FiniteBuildSettings:
    """Banana winding-pack geometry for finite-build Stage-2 optimization.

    ``numfilaments_n`` x ``numfilaments_b`` filaments are laid on a regular grid in
    the pre-rotation frame's normal/binormal directions with the given gap sizes
    (meters). ``rotation_order`` is the Fourier order of the optimizable
    pack-rotation profile (``None`` fixes the pack orientation, adding no rotation
    DOFs). ``frame`` is the pre-rotation orthonormal frame: ``'centroid'`` (robust
    default) or ``'frenet'``. The pack approximates one finite-build coil per Singh
    et al., J. Plasma Phys. 86 (2020).
    """

    numfilaments_n: int
    numfilaments_b: int
    gapsize_n: float
    gapsize_b: float
    rotation_order: int | None
    frame: str

    @property
    def nfilaments(self) -> int:
        return int(self.numfilaments_n) * int(self.numfilaments_b)

    @property
    def pack_half_extent_n_m(self) -> float:
        """Half the pack's outer span along the normal direction (meters)."""
        return 0.5 * (int(self.numfilaments_n) - 1) * float(self.gapsize_n)

    @property
    def pack_half_extent_b_m(self) -> float:
        """Half the pack's outer span along the binormal direction (meters)."""
        return 0.5 * (int(self.numfilaments_b) - 1) * float(self.gapsize_b)

    @property
    def pack_reach_m(self) -> float:
        """Corner reach of the pack cross-section (meters): hypot(half_n, half_b).

        The worst-case distance from the centerline to a pack corner; used to
        inflate centerline-based clearance so the real pack envelope is cleared.
        """
        return float(np.hypot(self.pack_half_extent_n_m, self.pack_half_extent_b_m))


def build_finite_build_banana_coils(
    master_curve,
    total_banana_current_A,
    finite_build,
    surf_coils,
):
    """Symmetry-expanded multi-filament banana pack for finite-build optimization.

    The pack is built from the single ``master_curve`` centerline, so every filament
    shares the centerline degrees of freedom plus one ``FrameRotation`` (the
    pack-rotation DOFs). A single optimizable drives the NET banana current
    (``banana_net_current``); each filament scales it by ``1/nfilaments`` so the
    pack's net current equals the thin-filament banana current and one banana-current
    DOF drives the whole pack. The net-current optimizable is recoverable as the
    first (identity-symmetry) coil's ``current.current_to_scale``, so banana-current
    bounds and net-current metadata stay correct with unchanged thresholds. Symmetry
    is applied once, after the pack is built (finite-build-before-symmetry), matching
    the upstream finite-build example.
    """
    filaments = create_multifilament_grid(
        master_curve,
        finite_build.numfilaments_n,
        finite_build.numfilaments_b,
        finite_build.gapsize_n,
        finite_build.gapsize_b,
        rotation_order=finite_build.rotation_order,
        frame=finite_build.frame,
    )
    nfilaments = finite_build.nfilaments
    banana_net_current = ScaledCurrent(Current(1), float(total_banana_current_A))
    filament_current = ScaledCurrent(banana_net_current, 1.0 / nfilaments)
    return coils_via_symmetries(
        filaments,
        [filament_current] * nfilaments,
        surf_coils.nfp,
        surf_coils.stellsym,
    )


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
    *,
    equilibrium_file,
    surface_scale_factor,
    toroidal_flux,
    nphi,
    ntheta,
    proxy_plasma_current_A=0.0,
    vf_current_A=0.0,
    vf_template_path=None,
    finite_current_mode="wataru_proxy_field",
    flip_banana=False,
    banana_i_fixed_s2=None,
    finite_build=None,
    return_vf_build_result=False,
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

    if finite_current_mode == JHALPERN30_FINITE_CURRENT_MODE:
        if finite_build is not None:
            raise ValueError(
                "Finite-build optimization is not supported with the jhalpern30 "
                "banana current path."
            )
        banana_coils, _banana_replay = build_jhalpern30_banana_coils(
            banana_curve,
            surf_coils=surf_coils,
            flip_banana=bool(flip_banana),
            banana_i_fixed_s2=banana_i_fixed_s2,
        )
    elif finite_build is not None:
        banana_coils = build_finite_build_banana_coils(
            banana_curve,
            banana_init_current_A,
            finite_build,
            surf_coils,
        )
    else:
        banana_coils = coils_via_symmetries(
            [banana_curve],
            [ScaledCurrent(Current(1), banana_init_current_A)],
            surf_coils.nfp,
            surf_coils.stellsym,
        )

    finite_current_profile = get_finite_current_profile(finite_current_mode)
    if finite_current_profile.default_num_proxy_coils == 0:
        proxy_coils = []
    elif finite_current_mode == JHALPERN30_FINITE_CURRENT_MODE:
        proxy_coils = build_jhalpern30_proxy_plasma_current_coils(
            surf,
            float(proxy_plasma_current_A),
        )
    else:
        proxy_coils = build_proxy_plasma_current_coils(
            equilibrium_file=equilibrium_file,
            surface_scale_factor=float(surface_scale_factor),
            nphi=int(nphi),
            ntheta=int(ntheta),
            toroidal_flux=float(toroidal_flux),
            plasma_current_A=float(proxy_plasma_current_A),
        )
    vf_build_result = VFCoilBuildResult(coils=[], current_control=None)
    if vf_template_path not in {None, ""}:
        vf_build_result = build_vf_coils_for_profile(
            finite_current_mode=finite_current_mode,
            proxy_current_A=float(proxy_plasma_current_A),
            vf_current_A=float(vf_current_A),
            vf_template_path=str(vf_template_path),
        )
    vf_coils = vf_build_result.coils

    coils = tf_coils + banana_coils + proxy_coils + vf_coils
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    curves = [coil.curve for coil in coils]
    curves_to_vtk(curves, out_dir + "curves_init", close=True)
    unitn = surf.unitnormal()
    point_data = {
        "B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]
    }
    surf.to_vtk(out_dir + "surf_init", extra_data=point_data)
    if return_vf_build_result:
        return bs, curves, banana_curve, banana_coils, proxy_coils, vf_build_result
    return bs, curves, banana_curve, banana_coils, proxy_coils, vf_coils


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
    magnitude_field_plot(
        modBfinal, surf_area, phi, theta, out_dir_iter + "MagFieldPlot"
    )
    return mean_abs_relBfinal_norm
