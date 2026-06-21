import logging
from dataclasses import dataclass

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
from simsopt.geo import (
    CurveCWSFourierCPP,
    SurfaceRZFourier,
    create_multifilament_grid,
    curves_to_vtk,
)
from simsopt.geo.framedcurve import rotated_surface_tangent_frame

from plotting_utils import magnitude_field_plot, norm_field_plot
from workflow_helpers import validate_normalized_toroidal_flux
from banana_opt.hardware_contracts import (
    TYPE_KK_INNER_RADIUS_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
    validate_target_lcfs_major_radius,
    validate_target_lcfs_minor_radius,
)
from banana_opt.current_contracts import unwrap_current_optimizable
from banana_opt.current_contracts import VFCoilBuildResult
from banana_opt.current_contracts import resolve_jhalpern30_fresh_vf_current_A
from banana_opt.design_only_fields import mark_design_only_field
from banana_opt.finite_current_profiles import JHALPERN30_FINITE_CURRENT_MODE
from banana_opt.finite_current_profiles import get_finite_current_profile
from banana_opt.jhalpern30_compat import (
    build_jhalpern30_banana_coils,
    build_jhalpern30_proxy_plasma_current_coils,
    build_jhalpern30_vf_coil_build_result,
    resolve_jhalpern30_vf_template_path,
)
from banana_opt.json_compat import load_boozer_finite_i as load
from banana_opt.proxy_current_coils import build_planar_proxy_plasma_current_coils


LOGGER = logging.getLogger(__name__)


# Optional bounded winding-surface size DOFs for Stage 2 (default OFF).
#
# rc(0,0) is the coil-winding-surface MAJOR radius (R0 translation). The
# corridor (0.908, 0.993) m is the T0.3-verified ``both_clear`` vessel-clearance
# window: the on-spec base R0 of 0.903 m is infeasible (vessel clearance
# -3.299 mm), so the lower bound sits above it. NOTE: the on-spec seed
# rc(0,0)=0.903 is BELOW this lower bound by design; at optimize time L-BFGS-B
# clips the seed x0 into [lb, ub] before the first evaluation. (This module does
# not run the optimizer; the bound is recorded on the surf DOF for the solver.)
WINDING_SURFACE_FREE_R0_BOUNDS_M = (0.908, 0.993)
# rc(1,0)/zs(1,0) are the coil-winding-surface MINOR radius (a). Growing a
# worsens vessel clearance (T0.2), so this is a last-resort lever floored at the
# on-spec a=0.142 m with a 0.20 m ceiling.
WINDING_SURFACE_FREE_MINOR_BOUNDS_M = (0.142, 0.20)


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
    return build_planar_proxy_plasma_current_coils(
        placement="axis_fourier_zeroth",
        equilibrium_file=equilibrium_file,
        surface_scale_factor=float(surface_scale_factor),
        nphi=int(nphi),
        ntheta=int(ntheta),
        toroidal_flux=float(toroidal_flux),
        plasma_current_A=float(plasma_current_A),
    )


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
    the pre-rotation frame's normal/binormal directions with the given
    center-to-center spacings (meters). ``rotation_order`` is the Fourier order of the optimizable
    pack-rotation profile (``None`` fixes the pack orientation, adding no rotation
    DOFs). ``frame`` is the pre-rotation orthonormal frame: ``'centroid'`` or
    ``'frenet'`` from upstream SIMSOPT, or local-fork ``'surface_tangent'`` (lays
    the pack flat against the banana winding surface, normal axis tracking the
    surface normal). The pack approximates one finite-build coil per Singh et al.,
    J. Plasma Phys. 86 (2020).
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
    surface_major_radius = float(surf_coils.get_rc(0, 0))
    surface_midplane_z = 0.0
    if not bool(getattr(surf_coils, "stellsym", True)) and hasattr(surf_coils, "get_zc"):
        surface_midplane_z = float(surf_coils.get_zc(0, 0))
    filaments = create_multifilament_grid(
        master_curve,
        finite_build.numfilaments_n,
        finite_build.numfilaments_b,
        finite_build.gapsize_n,
        finite_build.gapsize_b,
        rotation_order=finite_build.rotation_order,
        frame=finite_build.frame,
        surface_major_radius=surface_major_radius,
        surface_midplane_z=surface_midplane_z,
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


def configure_winding_surface_shape_dofs(
    surf_coils,
    *,
    free_mpol=0,
    free_ntor=0,
    free_r0=False,
    free_minor=False,
):
    """Fix the CWS by default, then unfix bounded size / low-mode shape DOFs.

    ``free_mpol`` / ``free_ntor`` unfix a low-(m, n) shape range. ``free_r0``
    unfixes the major-radius translation ``rc(0,0)``; ``free_minor`` unfixes the
    minor-radius pair ``rc(1,0)`` / ``zs(1,0)``. The size DOFs are freeable
    independently of shape modes -- on the on-spec mpol=1/ntor=0 winding surface
    the shape range frees nothing net, so ``free_r0`` / ``free_minor`` are the
    real continuation levers. Each requested-free size DOF is bounded from the
    SSOT corridors above.
    """
    free_mpol = int(free_mpol)
    free_ntor = int(free_ntor)
    free_r0 = bool(free_r0)
    free_minor = bool(free_minor)
    if free_mpol < 0:
        raise ValueError("--winding-surface-free-mpol must be non-negative.")
    if free_ntor < 0:
        raise ValueError("--winding-surface-free-ntor must be non-negative.")

    # (DOF name, (lower, upper)) pairs for each requested-free size DOF.
    requested_size_bounds = []
    if free_r0:
        requested_size_bounds.append(("rc(0,0)", WINDING_SURFACE_FREE_R0_BOUNDS_M))
    if free_minor:
        requested_size_bounds.append(("rc(1,0)", WINDING_SURFACE_FREE_MINOR_BOUNDS_M))
        requested_size_bounds.append(("zs(1,0)", WINDING_SURFACE_FREE_MINOR_BOUNDS_M))
    requested_size_names = frozenset(name for name, _ in requested_size_bounds)

    surf_coils.fix_all()
    if free_mpol > 0 or free_ntor > 0:
        if free_mpol > int(surf_coils.mpol):
            raise ValueError(
                "--winding-surface-free-mpol exceeds the winding-surface mpol."
            )
        if free_ntor > int(surf_coils.ntor):
            raise ValueError(
                "--winding-surface-free-ntor exceeds the winding-surface ntor."
            )
        surf_coils.fixed_range(
            mmin=0,
            mmax=free_mpol,
            nmin=-free_ntor,
            nmax=free_ntor,
            fixed=False,
        )
        # The shape range can free the base size modes; re-fix them so they stay
        # pinned UNLESS the caller explicitly requested them free below.
        base_fixed_names = ("rc(0,0)", "rc(1,0)", "zs(1,0)")
        available_names = set(surf_coils.local_full_dof_names)
        for name in base_fixed_names:
            if name in available_names and name not in requested_size_names:
                surf_coils.fix(name)

    # Free + bound the requested size DOFs regardless of the shape-mode path.
    for name, (lower, upper) in requested_size_bounds:
        surf_coils.unfix(name)
        surf_coils.set_lower_bound(name, lower)
        surf_coils.set_upper_bound(name, upper)

    return tuple(
        name
        for name in surf_coils.local_full_dof_names
        if not surf_coils.is_fixed(name)
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
    winding_surface_free_mpol=0,
    winding_surface_free_ntor=0,
    winding_surface_free_r0=False,
    winding_surface_free_minor=False,
    return_vf_build_result=False,
):
    winding_surface_free_dof_names = configure_winding_surface_shape_dofs(
        surf_coils,
        free_mpol=winding_surface_free_mpol,
        free_ntor=winding_surface_free_ntor,
        free_r0=winding_surface_free_r0,
        free_minor=winding_surface_free_minor,
    )
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
    if proxy_coils:
        mark_design_only_field(
            bs,
            reason=f"finite_current_proxy_line_current: {finite_current_mode}",
        )
    bs.set_points(surf.gamma().reshape((-1, 3)))

    curves = [coil.curve for coil in coils]
    curves_to_vtk(curves, out_dir + "curves_init", close=True)
    unitn = surf.unitnormal()
    point_data = {
        "B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]
    }
    surf.to_vtk(out_dir + "surf_init", extra_data=point_data)
    if return_vf_build_result:
        return (
            bs,
            curves,
            banana_curve,
            banana_coils,
            proxy_coils,
            vf_build_result,
            winding_surface_free_dof_names,
        )
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


def finite_build_frame_aware_curvature_limit_inv_m(
    finite_build,
    inner_radius_margin_m=TYPE_KK_INNER_RADIUS_MARGIN_M,
):
    """Conservative frame-aware centerline curvature cap for the Type-KK outer
    build channel (1/m).

    Bench-measured self-intersection model: the 1/8" FEP jacket protects the
    conductor, so the binding bend constraint is geometric self-intersection of
    the OUTER build envelope at zero inner-surface radius -- not the conductor
    wire bend radius. Buildability requires centerline bend radius >= the
    relevant outer-channel half-dimension in the bend plane. Without relying on a
    realized pack twist, the strict axis-safe value is the edgewise half-width,
    so the conservative cap is ``1 / max(outer_half_depth, outer_half_width)``.
    For the Type-KK outer channel this is ~43.31/m. The ``finite_build`` pack-grid
    argument is accepted for interface parity with the rotation-aware helpers; the
    cap depends only on the fixed outer-channel contract.
    """
    outer_edge_reach_m = float(
        max(
            TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
            TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
        )
    )
    required_radius_m = float(inner_radius_margin_m) + outer_edge_reach_m
    if required_radius_m <= 0.0:
        return float("inf")
    return 1.0 / required_radius_m


def pack_projected_reach_m(half_extent_n_m, half_extent_b_m, bend_angle_rad):
    """Rectangle support-function reach (m) in a centerline bend direction.

    The cross-section is a rectangle of half-extents (``half_extent_n_m`` along the
    normal axis, ``half_extent_b_m`` along the binormal). ``bend_angle_rad`` is the
    angle between the centerline bend direction and the normal axis. The reach is
    the same rectangle support function used by
    ``rotation_aware_projected_half_extent_m``:
    ``half_n*abs(cos(angle)) + half_b*abs(sin(angle))``. A quadratic blend
    under-reports intermediate misalignment and is optimistic for the Type-KK
    outer-channel non-fold condition.
    """
    half_n = abs(float(half_extent_n_m))
    half_b = abs(float(half_extent_b_m))
    cos_abs = abs(float(np.cos(bend_angle_rad)))
    sin_abs = abs(float(np.sin(bend_angle_rad)))
    return half_n * cos_abs + half_b * sin_abs


def finite_build_rotation_aware_curvature_limit_inv_m(
    finite_build,
    inner_radius_margin_m=TYPE_KK_INNER_RADIUS_MARGIN_M,
    bend_angle_rad=0.0,
):
    """Rotation-aware centerline curvature cap for the Type-KK outer channel (1/m).

    Identical to ``finite_build_frame_aware_curvature_limit_inv_m`` except it uses
    the outer-channel reach interpolated into the actual centerline bend direction
    (``pack_projected_reach_m`` over the outer half-extents) instead of the
    edgewise fallback. ``bend_angle_rad`` is the angle from the outer-channel
    NORMAL axis (flatwise/thin). Axis-aligned flatwise bends shrink the required
    inner-edge radius and lift the cap (~123.03/m), while edgewise bends recover
    the legacy axis-safe value (~43.31/m). Intermediate angles use the rectangle
    support function directly; diagonal corner reach can be stricter than the
    edgewise endpoint, so callers must treat this as the realized-orientation cap,
    not a monotone cap-lift.
    """
    projected_reach_m = pack_projected_reach_m(
        TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
        TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
        bend_angle_rad,
    )
    required_radius_m = float(inner_radius_margin_m) + float(projected_reach_m)
    if required_radius_m <= 0.0:
        return float("inf")
    return 1.0 / required_radius_m


def _centerline_bend_direction(banana_curve):
    """Unit in-plane curvature (bend) direction per quadpoint.

    The component of ``gamma''`` perpendicular to the unit tangent, normalized.
    Zero at points where the curve is locally straight (no bend). Shared SSOT for
    the rotation-aware reach projection and the alpha warm-start, which both need
    the direction the centerline bends in.
    """
    gammadash = np.asarray(banana_curve.gammadash(), dtype=float)
    gammadashdash = np.asarray(banana_curve.gammadashdash(), dtype=float)
    tangent_norm = np.linalg.norm(gammadash, axis=1)
    tangent = np.divide(
        gammadash,
        tangent_norm[:, None],
        out=np.zeros_like(gammadash),
        where=tangent_norm[:, None] > 0.0,
    )
    curvature_vector = (
        gammadashdash - np.sum(gammadashdash * tangent, axis=1)[:, None] * tangent
    )
    bend_norm = np.linalg.norm(curvature_vector, axis=1)
    return np.divide(
        curvature_vector,
        bend_norm[:, None],
        out=np.zeros_like(curvature_vector),
        where=bend_norm[:, None] > 0.0,
    )


def rotation_aware_projected_half_extent_m(finite_build, banana_curve, framedcurve):
    """Per-quadpoint outer-channel half-extent projected into the centerline bend
    plane, using the REALIZED rotated pack frame (the live twist ``alpha(theta)``).

    The rotated ``(normal, binormal)`` axes are read straight from
    ``framedcurve.rotated_frame()`` -- the very frame ``CurveFilament`` lays the
    pack on. The reach is the rectangle SUPPORT FUNCTION in the bend direction,
    ``half_n*|bend.n| + half_b*|bend.b|`` (depth along the rotated normal, width
    along the rotated binormal) -- i.e. the worst (inner) corner half-extent,
    which is exactly the quantity the non-fold condition ``w|kappa_w| +
    h|kappa_h| < 1`` binds on. (A quadratic blend ``half_n*n^2 + half_b*b^2``
    under-reports the support function at intermediate misalignment -- it agrees
    only when the bend is axis-aligned -- and is therefore optimistic.) The
    ``finite_build`` pack-grid argument is accepted for interface parity. SSOT
    for the rotated-frame reach VALUE; the JAX twin ``_projected_reach_pure`` must
    match it (pinned by the numpy<->JAX consistency test).
    """
    half_n_m = TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M
    half_b_m = TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M
    bend_direction = _centerline_bend_direction(banana_curve)
    _, normal_axis, binormal_axis = (
        np.asarray(axis, dtype=float) for axis in framedcurve.rotated_frame()
    )
    normal_projection = np.sum(bend_direction * normal_axis, axis=1)
    binormal_projection = np.sum(bend_direction * binormal_axis, axis=1)
    return half_n_m * np.abs(normal_projection) + half_b_m * np.abs(binormal_projection)


def rotation_aware_curvature_report(
    finite_build,
    banana_curve,
    framedcurve,
    inner_radius_margin_m=TYPE_KK_INNER_RADIUS_MARGIN_M,
):
    """Realized rotation-aware curvature cap + cashable headroom (T3.2 / G1).

    Diagnostic only. Compares realized centerline ``|kappa|`` against:
    - the realized rotation-aware cap ``1 / (margin + reach(alpha(theta)))`` from
      the live rotated outer-channel frame, and
    - the conservative measured edgewise cap
      ``finite_build_frame_aware_curvature_limit_inv_m`` (the in-loop steering cap,
      ~43.31/m for the Type-KK outer channel).

    Returns the arclength fraction where the live twist turns an
    over-conservative-cap bend buildable (``...HEADROOM_ARCLEN_FRACTION``) and the
    fraction still infeasible (``...RESIDUAL_INFEASIBLE_FRACTION``). No objective
    or gate is touched.
    """
    margin_m = float(inner_radius_margin_m)
    kappa = np.asarray(banana_curve.kappa(), dtype=float)
    rotated_half_extent_m = rotation_aware_projected_half_extent_m(
        finite_build, banana_curve, framedcurve
    )
    required_radius_m = rotated_half_extent_m + margin_m
    rotation_aware_cap_inv_m = np.divide(
        1.0,
        required_radius_m,
        out=np.full(required_radius_m.shape, float("inf"), dtype=float),
        where=required_radius_m > 0.0,
    )
    conservative_cap_inv_m = float(
        finite_build_frame_aware_curvature_limit_inv_m(finite_build, margin_m)
    )
    arclen_weight = np.linalg.norm(
        np.asarray(banana_curve.gammadash(), dtype=float), axis=1
    )
    total_arclen = float(np.sum(arclen_weight))
    cashed = (kappa > conservative_cap_inv_m) & (kappa <= rotation_aware_cap_inv_m)
    residual = kappa > rotation_aware_cap_inv_m
    if total_arclen > 0.0:
        cashed_fraction = float(np.sum(arclen_weight[cashed]) / total_arclen)
        residual_fraction = float(np.sum(arclen_weight[residual]) / total_arclen)
    else:
        cashed_fraction = 0.0
        residual_fraction = 0.0
    return {
        "FINITEBUILD_ROTATION_AWARE_CAP_MIN_INV_M": float(
            np.min(rotation_aware_cap_inv_m)
        ),
        "FINITEBUILD_ROTATION_AWARE_CAP_MEAN_INV_M": float(
            np.mean(rotation_aware_cap_inv_m)
        ),
        "FINITEBUILD_CONSERVATIVE_CORNER_CAP_INV_M": conservative_cap_inv_m,
        "FINITEBUILD_ROTATION_AWARE_HEADROOM_ARCLEN_FRACTION": cashed_fraction,
        "FINITEBUILD_ROTATION_AWARE_RESIDUAL_INFEASIBLE_FRACTION": residual_fraction,
    }


def rotation_aware_warm_start_alpha_dofs(
    banana_curve,
    framedcurve,
    half_n_m=TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    half_b_m=TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
):
    """``FrameRotation`` dofs that lay the thinner pack axis into the centerline
    bend (rotation-aware curvature warm-start).

    Root cause this addresses: from ``alpha=0`` the optimizer satisfies
    buildability by flattening the centerline ``|kappa|`` and never folds the pack
    (twist-strain pressure pulls ``alpha`` back to 0), so the realized cap stays
    pinned at the conservative corner. Seeding ``alpha`` at the reach-minimizing
    profile starts the descent already folded, inside the basin the folded optimum
    lives in.

    The projected outer-channel reach is the rectangle support function
    ``half_n*|bend.n(alpha)| + half_b*|bend.b(alpha)|`` (see
    ``rotation_aware_projected_half_extent_m``). The rotated ``(n, b)`` preserve the
    in-plane bend norm, so with ``half_n < half_b`` the reach is minimized by the
    rotation that aligns the thin normal ``n`` with the bend direction (the optimum
    is the same for the support function and the old quadratic blend, so this
    warm-start target is unchanged by the support-function fix). From the
    ``rotated_surface_tangent_frame`` convention
    ``n(alpha) = cos(alpha) n0 - sin(alpha) b0`` this is
    ``alpha*(theta) = -atan2(bend.b0, bend.n0)`` (modulo pi; the reach is
    pi-periodic in alpha), where ``(t, n0, b0)`` is the UNROTATED (alpha=0)
    surface-tangent reference frame. The degenerate ``half_n > half_b`` case aligns
    the binormal instead (shift by pi/2). ``alpha*`` is director-unwrapped for
    continuity, then least-squares fit -- curvature-magnitude weighted so the
    high-``kappa`` apexes that actually bind the cap dominate -- onto the live
    ``FrameRotation`` Fourier basis. Pure: returns the dof vector; the caller
    assigns ``framedcurve.rotation.x``.
    """
    rotation = framedcurve.rotation
    quadpoints = np.asarray(banana_curve.quadpoints, dtype=float)
    bend_direction = _centerline_bend_direction(banana_curve)
    # Unrotated (alpha=0) surface-tangent reference frame, computed independently of
    # the rotation's current dofs so a re-warm-start reads the same reference.
    _, normal0, binormal0 = (
        np.asarray(axis, dtype=float)
        for axis in rotated_surface_tangent_frame(
            np.asarray(banana_curve.gamma(), dtype=float),
            np.asarray(banana_curve.gammadash(), dtype=float),
            np.zeros(quadpoints.shape[0]),
            framedcurve.major_radius,
            framedcurve.midplane_z,
        )
    )
    bend_dot_n0 = np.sum(bend_direction * normal0, axis=1)
    bend_dot_b0 = np.sum(bend_direction * binormal0, axis=1)
    alpha_principal = -np.arctan2(bend_dot_b0, bend_dot_n0)
    if half_n_m > half_b_m:
        alpha_principal = alpha_principal + 0.5 * np.pi
    # The reach is pi-periodic in alpha (n and -n give the same reach), so the raw
    # principal angle carries spurious pi jumps. Unwrap the doubled angle (single-
    # valued mod 2pi) then halve for a continuous director-consistent fit target.
    # Assumes the bend director has no net winding over the period (true for the
    # banana U-turn geometry); a residual seam at theta=1->0 is harmless because this
    # is a warm-start the optimizer refines from, not a hard target.
    alpha_target = 0.5 * np.unwrap(2.0 * alpha_principal)
    # Curvature-magnitude weights focus the low-order fit on the high-kappa apexes
    # (where the cap binds); flat segments (kappa~0, arbitrary alpha*) get ~0 weight
    # and do not distort the fit.
    curvature = np.asarray(banana_curve.kappa(), dtype=float)
    sqrt_weight = np.sqrt(np.maximum(curvature, 0.0))
    # Live FrameRotation Fourier basis (mirrors framedcurve.jaxrotation_pure):
    # alpha = scale * (dofs[0] + sum_j dofs[2j-1] sin(2 pi j theta)
    #                           + dofs[2j] cos(2 pi j theta)).
    order = int(rotation.order)
    scale = float(rotation.scale)
    columns = [np.ones_like(quadpoints)]
    for j in range(1, order + 1):
        columns.append(np.sin(2.0 * np.pi * j * quadpoints))
        columns.append(np.cos(2.0 * np.pi * j * quadpoints))
    design = scale * np.stack(columns, axis=1)
    dofs, _, _, _ = np.linalg.lstsq(
        sqrt_weight[:, None] * design, sqrt_weight * alpha_target, rcond=None
    )
    return dofs


def closed_polyline_segments(gamma):
    """(N, 2, 3) chord segments of a closed curve sampled at ``gamma`` points.

    Includes the wrap-around chord from the last sample back to the first, so
    the polyline is closed like the underlying periodic curve.
    """
    points = np.asarray(gamma, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("gamma must be an (N>=2, 3) point array.")
    return np.ascontiguousarray(
        np.stack((points, np.roll(points, -1, axis=0)), axis=1)
    )


def _pairwise_point_distances(points_a, points_b):
    return np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)


def _segment_pair_endpoint_min_distances(point_distances):
    """Per segment pair (i, j), min distance among the four chord endpoints.

    Segment ``i`` of a closed polyline spans points ``i`` and ``i+1`` (wrapped),
    so the four endpoint combinations are the distance matrix and its
    single-step rolls along each axis.
    """
    rolled_rows = np.roll(point_distances, -1, axis=0)
    return np.minimum.reduce(
        (
            point_distances,
            rolled_rows,
            np.roll(point_distances, -1, axis=1),
            np.roll(rolled_rows, -1, axis=1),
        )
    )


def _min_distance_between_closed_polylines(segments_a, segments_b, point_distances):
    """Exact min distance between two closed chord polylines.

    Seeds with the point-cloud minimum (a valid upper bound: every sample
    point is a segment endpoint), then refines only the segment pairs whose
    distance lower bound ``endpoint_min - (len_i + len_j)`` can beat it,
    using the exact ``segment_segment_distance`` kernel. Exact because the
    pruning bound is a true lower bound on the segment-pair distance.
    """
    best = float(np.min(point_distances))
    endpoint_min = _segment_pair_endpoint_min_distances(point_distances)
    lengths_a = np.linalg.norm(segments_a[:, 1] - segments_a[:, 0], axis=1)
    lengths_b = np.linalg.norm(segments_b[:, 1] - segments_b[:, 0], axis=1)
    lower_bounds = endpoint_min - (lengths_a[:, None] + lengths_b[None, :])
    for i, j in np.argwhere(lower_bounds < best):
        dist = segment_segment_distance(
            segments_a[i, 0],
            segments_a[i, 1],
            segments_b[j, 0],
            segments_b[j, 1],
        )
        if dist < best:
            best = float(dist)
    return best


def curve_curve_min_distance_segments_m(curves):
    """Exact min distance between the closed chord polylines of distinct curves.

    Piecewise-linear analogue of ``CurveCurveDistance.shortest_distance()`` on
    the same quadrature samples. Because every sample point is a segment
    endpoint, the result is always <= the point-cloud minimum: it additionally
    sees closest approaches BETWEEN samples. Relative to the true smooth
    curves it remains a chord approximation (sagitta error O(kappa * h^2)).
    """
    gammas = [np.asarray(curve.gamma(), dtype=np.float64) for curve in curves]
    segment_sets = [closed_polyline_segments(gamma) for gamma in gammas]
    best = float("inf")
    for i in range(len(segment_sets)):
        for j in range(i):
            best = min(
                best,
                _min_distance_between_closed_polylines(
                    segment_sets[i],
                    segment_sets[j],
                    _pairwise_point_distances(gammas[i], gammas[j]),
                ),
            )
    return best


def curve_surface_min_distance_segments_m(curves, surface_points):
    """Exact min distance from closed chord polylines to a surface point cloud.

    Segment-to-point-cloud method: the curve side becomes exact chord segments
    (always <= the point-cloud value computed on the same samples, i.e. it can
    only report a closer-or-equal approach), while the surface side stays the
    same sampled point cloud, whose discretization can still overestimate the
    distance to the continuous surface exactly as in
    ``CurveSurfaceDistance.shortest_distance()``.
    """
    points = np.ascontiguousarray(
        np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    )
    if points.shape[0] == 0:
        raise ValueError("surface_points must contain at least one point.")
    best = float("inf")
    for curve in curves:
        segments = closed_polyline_segments(curve.gamma())
        starts = segments[:, 0]
        chords = segments[:, 1] - segments[:, 0]
        chord_sq = np.einsum("ij,ij->i", chords, chords)
        for s in range(starts.shape[0]):
            offsets = points - starts[s]
            if chord_sq[s] < 1.0e-30:
                dist = float(np.min(np.linalg.norm(offsets, axis=1)))
            else:
                t = np.clip(offsets @ chords[s] / chord_sq[s], 0.0, 1.0)
                dist = float(
                    np.min(np.linalg.norm(offsets - t[:, None] * chords[s], axis=1))
                )
            if dist < best:
                best = dist
    return best
