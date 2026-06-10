"""Shared topology scoring for field-line confinement evaluation.

Single source of truth for field-line tracing metrics used by:
- the search-time topology gate in single_stage_banana_example.py
- the callback medium-fidelity scorer
- the strict Poincare validator in poincare_surfaces.py

All three paths use the same stopping criteria, seed logic (midplane radial
sweep matching upstream SIMSOPT), and metric computation so that metrics
cannot drift between callback and validation.
"""

import copy
import dataclasses
from collections.abc import Mapping
from pathlib import Path
import sys

import numpy as np
import simsoptpp as sopp
from simsopt.field.magnetic_axis_helpers import (
    MagneticAxisNotLocatedError,
    locate_magnetic_axis_point,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from banana_opt.design_only_fields import assert_topology_field_allowed
from banana_opt.topology.kam_birkhoff import (
    DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS,
    KAM_FRACTION_SEMANTICS,
    WBA_EVALUATION_NOT_EVALUATED_AXIS_NOT_LOCATED,
    WBA_EVALUATION_NOT_EVALUATED_NO_CLASSIFIED_SEEDS,
    WBA_EVALUATION_NOT_EVALUATED_SKIPPED_BY_CALLER,
    missing_magnetic_axis_classification,
    classify_fieldline_hits,
    classifier_settings_payload,
    summarize_seed_classifications,
)


SEED_MODE_MIDPLANE = "midplane_radial_sweep"
SEED_MODE_EXTENDED_SURFACE = "extended_surface_radial_sweep"


# ---------------------------------------------------------------------------
# Helpers (previously duplicated across poincare_surfaces.py and the solver)
# ---------------------------------------------------------------------------


def midplane_seed_radii(surf, nfieldlines, inset_fraction=0.05, min_inset=0.01):
    """Seed field lines slightly inside the phi=0 midplane cross-section."""
    cross_section = surf.cross_section(phi=0.0, thetas=512)
    r = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    z = cross_section[:, 2]
    midplane = np.argsort(np.abs(z))[:8]
    r_mid = np.sort(r[midplane])
    rin = r_mid[0]
    rout = r_mid[-1]
    span = rout - rin
    inset = min(max(inset_fraction * span, min_inset), 0.45 * span)
    return np.linspace(rin + inset, rout - inset, nfieldlines)


def _update_seed_radius_bounds(contract, radii, *, min_key="r_min", max_key="r_max"):
    if radii is None or len(radii) == 0:
        return contract
    contract[min_key] = float(np.min(radii))
    contract[max_key] = float(np.max(radii))
    return contract


def build_midplane_seed_contract(nfieldlines, inset_fraction, radii=None):
    contract = {
        "mode": SEED_MODE_MIDPLANE,
        "nplanes": 1,
        "nfieldlines": int(nfieldlines),
        "inset_fraction": float(inset_fraction),
        "phi": 0.0,
        "Z": 0.0,
    }
    return _update_seed_radius_bounds(contract, radii)


def _clone_surface_for_extension(surface):
    """Return a detached surface copy suitable for extend_via_normal()."""
    from simsopt.geo import SurfaceXYZTensorFourier

    if hasattr(surface, "copy"):
        return surface.copy()
    if isinstance(surface, SurfaceXYZTensorFourier):
        cloned = SurfaceXYZTensorFourier(
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
            clamped_dims=list(getattr(surface, "clamped_dims", [False, False, False])),
            quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=float),
            quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=float),
        )
        # SurfaceXYZTensorFourier inherits Optimizable, so `.x` is only the
        # free-DOF view. Use the full coefficient vector to preserve geometry
        # even when some surface coefficients are fixed.
        cloned.local_full_x = np.asarray(surface.get_dofs(), dtype=float).copy()
        return cloned
    return copy.deepcopy(surface)


def extended_surface_seed_radii(surface, nfieldlines, extend_distance=0.05):
    """Seed field lines across an outward-extended surface radial span."""
    extended_surface = _clone_surface_for_extension(surface)
    extended_surface.extend_via_normal(float(extend_distance))
    gamma = extended_surface.gamma()
    radii = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
    return np.linspace(float(np.min(radii)), float(np.max(radii)), int(nfieldlines))


def build_extended_surface_seed_contract(
    nfieldlines,
    extend_distance,
    radii=None,
):
    contract = {
        "mode": SEED_MODE_EXTENDED_SURFACE,
        "nplanes": 1,
        "nfieldlines": int(nfieldlines),
        "extend_distance": float(extend_distance),
        "phi": 0.0,
        "Z": 0.0,
        # Baseline-style default presentation renders launch at phi=0, Z=0 using the
        # global radial envelope of the outward-extended surface rather than
        # the phi=0 cross-section span itself.
        "radial_sampling_source": "global_extended_surface_bounds",
    }
    return _update_seed_radius_bounds(
        contract,
        radii,
        min_key="r_min_seed",
        max_key="r_max_seed",
    )


def padded_bounds(
    rmin, rmax, zmax, radial_padding_fraction=0.05, axial_padding_fraction=0.05
):
    """Add a modest interpolation buffer around the validation surface bounds."""
    rpad = max(radial_padding_fraction * (rmax - rmin), 0.02)
    zpad = max(axial_padding_fraction * zmax, 0.01)
    return max(0.0, rmin - rpad), rmax + rpad, zmax + zpad


@dataclasses.dataclass(frozen=True)
class TopologyTraceDomain:
    """Cylindrical bounds that tracing seeds and stop guards require."""

    surface_rmin: float
    surface_rmax: float
    surface_zmax: float
    stopping_rmin: float
    stopping_rmax: float
    stopping_zmax: float
    box_padding: float
    seed_rmin: float | None = None
    seed_rmax: float | None = None
    seed_zmax: float = 0.0

    @property
    def required_rmin(self):
        if self.seed_rmin is None:
            return float(self.stopping_rmin)
        return float(min(self.stopping_rmin, self.seed_rmin))

    @property
    def required_rmax(self):
        if self.seed_rmax is None:
            return float(self.stopping_rmax)
        return float(max(self.stopping_rmax, self.seed_rmax))

    @property
    def required_zmax(self):
        return float(max(self.stopping_zmax, abs(self.seed_zmax)))

    def as_metadata(self):
        payload = dataclasses.asdict(self)
        payload["required_rmin"] = self.required_rmin
        payload["required_rmax"] = self.required_rmax
        payload["required_zmax"] = self.required_zmax
        return payload


def _surface_cylindrical_bounds(surface):
    gamma = surface.gamma()
    rr = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
    zz = gamma[:, :, 2]
    return float(np.min(rr)), float(np.max(rr)), float(np.max(np.abs(zz)))


def _field_period_phirange(surface):
    return 0.0, (2.0 * np.pi) / float(max(int(surface.nfp), 1))


def surface_trace_domain(
    surface,
    *,
    box_padding=0.05,
    seed_radii=None,
    seed_zmax=0.0,
):
    """Return the metric domain that interpolation must cover before tracing."""
    surface_rmin, surface_rmax, surface_zmax = _surface_cylindrical_bounds(surface)
    if seed_radii is None:
        seed_rmin = None
        seed_rmax = None
    else:
        radii = np.asarray(seed_radii, dtype=float)
        seed_rmin = float(np.min(radii)) if radii.size else None
        seed_rmax = float(np.max(radii)) if radii.size else None
    return TopologyTraceDomain(
        surface_rmin=surface_rmin,
        surface_rmax=surface_rmax,
        surface_zmax=surface_zmax,
        stopping_rmin=float(surface_rmin * (1.0 - float(box_padding))),
        stopping_rmax=float(surface_rmax * (1.0 + float(box_padding))),
        stopping_zmax=float(surface_zmax * (1.0 + float(box_padding))),
        box_padding=float(box_padding),
        seed_rmin=seed_rmin,
        seed_rmax=seed_rmax,
        seed_zmax=float(abs(seed_zmax)),
    )


def extended_surface_trace_domain(
    surface,
    extend_distance,
    *,
    box_padding=0.05,
    seed_radii=None,
    seed_zmax=0.0,
):
    """Return the baseline-style trace domain from an outward-extended surface."""
    extended_surface = _clone_surface_for_extension(surface)
    extended_surface.extend_via_normal(float(extend_distance))
    return surface_trace_domain(
        extended_surface,
        box_padding=box_padding,
        seed_radii=seed_radii,
        seed_zmax=seed_zmax,
    )


def _interpolation_covers_trace_domain(
    rrange,
    phirange,
    zrange,
    trace_domain,
    required_phirange,
):
    covers_r = (
        float(rrange[0]) <= trace_domain.required_rmin
        and float(rrange[1]) >= trace_domain.required_rmax
    )
    phimin = float(phirange[0])
    phimax = float(phirange[1])
    required_phimin = float(required_phirange[0])
    required_phimax = float(required_phirange[1])
    covers_phi = phimin <= required_phimin and phimax >= required_phimax
    zmin = float(zrange[0])
    zmax = float(zrange[1])
    if zmin == 0.0:
        covers_z = zmax >= trace_domain.required_zmax
    else:
        covers_z = (
            zmin <= -trace_domain.required_zmax and zmax >= trace_domain.required_zmax
        )
    return bool(covers_r and covers_phi and covers_z)


def _field_model_payload(
    *,
    policy,
    selected_mode,
    reason,
    trace_domain,
    grid=None,
    rrange=None,
    phirange=None,
    zrange=None,
    required_phirange=None,
    extrapolate=None,
    interpolation_covers_trace_domain=None,
    max_abs_error=None,
    mean_abs_error=None,
    max_rel_error=None,
):
    return {
        "policy": str(policy),
        "selected_mode": str(selected_mode),
        "reason": str(reason),
        "tmax_threshold": float(TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD),
        "grid": grid,
        "rrange": None
        if rrange is None
        else [float(rrange[0]), float(rrange[1]), int(rrange[2])],
        "phirange": None
        if phirange is None
        else [float(phirange[0]), float(phirange[1]), int(phirange[2])],
        "zrange": None
        if zrange is None
        else [float(zrange[0]), float(zrange[1]), int(zrange[2])],
        "required_phirange": None
        if required_phirange is None
        else [float(required_phirange[0]), float(required_phirange[1])],
        "extrapolate": None if extrapolate is None else bool(extrapolate),
        "trace_domain": None if trace_domain is None else trace_domain.as_metadata(),
        "interpolation_covers_trace_domain": interpolation_covers_trace_domain,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
    }


def stop_reason_label(stop_index, stop_labels):
    """Map a stopping-criterion index to a human-readable label."""
    if 0 <= stop_index < len(stop_labels):
        return stop_labels[stop_index]
    return f"stop_{stop_index}"


def toroidal_angle(x, y):
    """Compute the toroidal angle from Cartesian (x, y)."""
    return float(np.mod(np.arctan2(y, x), 2 * np.pi))


def phi_hit_counts(fieldlines_phi_hits, phis):
    """Summarize how many Poincare hits were recorded on each phi plane."""
    return [
        int(
            sum(
                np.sum(_trace_hits_before_first_stop(fieldline)[0][:, 1] == i)
                for fieldline in fieldlines_phi_hits
            )
        )
        for i in range(len(phis))
    ]


TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD = 50.0
TOPOLOGY_INTERPOLATION_GRID = {
    "degree": 3,
    "nr": 40,
    "nphi": 40,
    "nz": 20,
}
TOPOLOGY_TRANSPORT_DIAGNOSTICS_SCHEMA_VERSION = (
    "single_stage_topology_transport_diagnostics_v1"
)

_GAMMA_C_UNAVAILABLE_REASON = (
    "Gamma_c requires bounce-integral drift metrics and flux-coordinate geometry "
    "that the single-stage vacuum topology scorer does not expose."
)
_EFFECTIVE_RIPPLE_UNAVAILABLE_REASON = (
    "EffectiveRipple (epsilon_eff) requires a Nemov-style bounce-integral "
    "transport backend and flux-coordinate geometry that the single-stage "
    "vacuum topology scorer does not expose."
)


def trace_fieldlines_xyz(
    field,
    xyz_inits,
    *,
    tmax,
    tol,
    phis,
    stopping_criteria,
):
    points = np.asarray(xyz_inits, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Topology seed points must have shape (n, 3), got {points.shape}"
        )
    res_tys = []
    res_phi_hits = []
    for point in points:
        res_ty, res_phi_hit = sopp.fieldline_tracing(
            field,
            (float(point[0]), float(point[1]), float(point[2])),
            float(tmax),
            float(tol),
            phis=[float(phi) for phi in phis],
            stopping_criteria=list(stopping_criteria),
        )
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(np.asarray(res_phi_hit))
    return res_tys, res_phi_hits


def prepare_topology_field(
    surface,
    bfield,
    tmax,
    *,
    field_policy="auto",
    interpolation_grid=None,
    trace_domain=None,
    seed_radii=None,
):
    from simsopt.field import InterpolatedField

    resolved_policy = str(field_policy)
    if resolved_policy not in {"auto", "always", "never"}:
        raise ValueError(
            f"Unsupported topology field policy {field_policy!r}; expected "
            "'auto', 'always', or 'never'"
        )
    resolved_trace_domain = (
        surface_trace_domain(surface, seed_radii=seed_radii)
        if trace_domain is None
        else trace_domain
    )
    required_phirange = _field_period_phirange(surface)
    if isinstance(bfield, InterpolatedField):
        return bfield, _field_model_payload(
            policy=resolved_policy,
            selected_mode="pre_interpolated",
            reason="already_interpolated",
            trace_domain=resolved_trace_domain,
            required_phirange=required_phirange,
        )
    should_interpolate = resolved_policy == "always" or (
        resolved_policy == "auto"
        and float(tmax) >= float(TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD)
    )
    if not should_interpolate:
        reason = "below_threshold" if resolved_policy == "auto" else "explicit_never"
        return bfield, _field_model_payload(
            policy=resolved_policy,
            selected_mode="native",
            reason=reason,
            trace_domain=resolved_trace_domain,
            required_phirange=required_phirange,
            interpolation_covers_trace_domain=False,
        )

    grid = dict(TOPOLOGY_INTERPOLATION_GRID)
    explicit_rrange = None
    explicit_phirange = None
    explicit_zrange = None
    extrapolate = True
    if interpolation_grid is not None:
        grid.update(
            {
                key: interpolation_grid[key]
                for key in ("degree", "nr", "nphi", "nz")
                if key in interpolation_grid
            }
        )
        explicit_rrange = interpolation_grid.get("rrange")
        explicit_phirange = interpolation_grid.get("phirange")
        explicit_zrange = interpolation_grid.get("zrange")
        if "extrapolate" in interpolation_grid:
            extrapolate = bool(interpolation_grid["extrapolate"])
    interp_rmin, interp_rmax, interp_zmax = padded_bounds(
        resolved_trace_domain.surface_rmin,
        resolved_trace_domain.surface_rmax,
        resolved_trace_domain.surface_zmax,
    )
    rrange = explicit_rrange or (
        float(interp_rmin),
        float(interp_rmax),
        int(grid["nr"]),
    )
    phirange = explicit_phirange or (
        required_phirange[0],
        required_phirange[1],
        int(grid["nphi"]),
    )
    zrange = explicit_zrange or (
        (0.0, interp_zmax, int(grid["nz"]))
        if getattr(surface, "stellsym", False)
        else (-interp_zmax, interp_zmax, int(grid["nz"]))
    )
    interpolation_covers_trace_domain = _interpolation_covers_trace_domain(
        rrange,
        phirange,
        zrange,
        resolved_trace_domain,
        required_phirange,
    )
    explicit_interpolation_domain = (
        explicit_rrange is not None
        or explicit_phirange is not None
        or explicit_zrange is not None
    )
    if explicit_interpolation_domain and not interpolation_covers_trace_domain:
        if resolved_policy == "always":
            raise ValueError(
                "Requested topology interpolation domain does not cover the "
                "trace seed and stopping domain"
            )
        return bfield, _field_model_payload(
            policy=resolved_policy,
            selected_mode="native",
            reason="trace_domain_not_covered",
            trace_domain=resolved_trace_domain,
            grid={
                "degree": int(grid["degree"]),
                "nr": int(grid["nr"]),
                "nphi": int(grid["nphi"]),
                "nz": int(grid["nz"]),
            },
            rrange=rrange,
            phirange=phirange,
            zrange=zrange,
            required_phirange=required_phirange,
            extrapolate=extrapolate,
            interpolation_covers_trace_domain=False,
        )
    interpolated_field = InterpolatedField(
        bfield,
        int(grid["degree"]),
        rrange,
        phirange,
        zrange,
        extrapolate,
        nfp=int(surface.nfp),
        stellsym=bool(getattr(surface, "stellsym", False)),
    )
    # Keep constructor range objects alive for the C++ interpolant used during
    # long field-line traces.
    interpolated_field._topology_interpolation_ranges = (rrange, phirange, zrange)
    gamma = surface.gamma()
    surface_points = gamma.reshape((-1, 3))
    interpolated_field.set_points(surface_points)
    bfield.set_points(surface_points)
    exact_B = np.asarray(bfield.B(), dtype=float)
    interp_B = np.asarray(interpolated_field.B(), dtype=float)
    abs_diff = np.abs(exact_B - interp_B)
    exact_norm = np.linalg.norm(exact_B, axis=1)
    diff_norm = np.linalg.norm(exact_B - interp_B, axis=1)
    denom = np.maximum(exact_norm, 1.0e-12)
    max_rel_error = float(np.max(diff_norm / denom)) if diff_norm.size else 0.0
    return interpolated_field, _field_model_payload(
        policy=resolved_policy,
        selected_mode="interpolated",
        reason="explicit_always" if resolved_policy == "always" else "tmax_threshold",
        trace_domain=resolved_trace_domain,
        grid={
            "degree": int(grid["degree"]),
            "nr": int(grid["nr"]),
            "nphi": int(grid["nphi"]),
            "nz": int(grid["nz"]),
        },
        rrange=rrange,
        phirange=phirange,
        zrange=zrange,
        required_phirange=required_phirange,
        extrapolate=extrapolate,
        interpolation_covers_trace_domain=interpolation_covers_trace_domain,
        max_abs_error=float(np.max(abs_diff)) if abs_diff.size else 0.0,
        mean_abs_error=float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        max_rel_error=max_rel_error,
    )


# ---------------------------------------------------------------------------
# Transport diagnostics
# ---------------------------------------------------------------------------


def _transport_metric_status(
    metric_name,
    display_name,
    *,
    status,
    reason,
    value=None,
    aliases=(),
):
    return {
        "metric_name": str(metric_name),
        "display_name": str(display_name),
        "aliases": [str(alias) for alias in aliases],
        "status": str(status),
        "value": None if value is None else float(value),
        "reason": str(reason),
    }


def _gamma_c_status(*, status, reason):
    return _transport_metric_status(
        "gamma_c",
        "Gamma_c",
        status=status,
        reason=reason,
    )


def _effective_ripple_status(*, status, reason):
    return _transport_metric_status(
        "effective_ripple",
        "EffectiveRipple",
        aliases=("epsilon_eff",),
        status=status,
        reason=reason,
    )


def _surface_field_structure_not_evaluated(reason, *, status):
    return {
        "status": str(status),
        "reason": str(reason),
        "error_type": None,
        "grid_shape": None,
        "modB_min": None,
        "modB_max": None,
        "modB_mean": None,
        "modB_std": None,
        "modB_coefficient_of_variation": None,
        "mirror_ratio": None,
        "effective_inverse_aspect_ratio_epsilon": None,
    }


def _transport_diagnostics_payload(
    *,
    status,
    summary,
    surface_field_structure,
    metric_status,
    gamma_c_reason,
    effective_ripple_reason,
):
    return {
        "schema_version": TOPOLOGY_TRANSPORT_DIAGNOSTICS_SCHEMA_VERSION,
        "status": str(status),
        "summary": str(summary),
        "surface_field_structure": surface_field_structure,
        "gamma_c": _gamma_c_status(
            status=metric_status,
            reason=gamma_c_reason,
        ),
        "effective_ripple": _effective_ripple_status(
            status=metric_status,
            reason=effective_ripple_reason,
        ),
    }


def topology_transport_diagnostics_not_evaluated(reason):
    return _transport_diagnostics_payload(
        status="not_evaluated",
        summary=reason,
        surface_field_structure=_surface_field_structure_not_evaluated(
            reason,
            status="not_evaluated",
        ),
        metric_status="not_evaluated",
        gamma_c_reason=reason,
        effective_ripple_reason=reason,
    )


def _surface_modB_samples(field, flat_points):
    field.set_points(flat_points)
    if hasattr(field, "AbsB"):
        return np.asarray(field.AbsB(), dtype=float).reshape((-1,))
    return np.linalg.norm(np.asarray(field.B(), dtype=float), axis=1).reshape((-1,))


def _surface_field_structure(surface, field):
    surface_points = np.asarray(surface.gamma(), dtype=float)
    if surface_points.ndim != 3 or surface_points.shape[-1] != 3:
        raise ValueError(
            "Topology transport diagnostics require a surface gamma grid with "
            f"shape (nphi, ntheta, 3), got {surface_points.shape}"
        )
    flat_points = surface_points.reshape((-1, 3))
    modB = _surface_modB_samples(field, flat_points)
    if modB.size == 0:
        raise ValueError("Topology transport diagnostics received no |B| samples")
    if not np.all(np.isfinite(modB)):
        raise ValueError("Topology transport diagnostics received NaN/Inf |B| samples")

    modB_min = float(np.min(modB))
    modB_max = float(np.max(modB))
    if modB_min <= 0.0:
        raise ValueError("Topology transport diagnostics require strictly positive |B|")
    modB_mean = float(np.mean(modB))
    modB_std = float(np.std(modB))
    mirror_ratio = float(modB_max / modB_min)
    return {
        "status": "evaluated",
        "reason": "surface_modB_grid",
        "error_type": None,
        "grid_shape": [
            int(surface_points.shape[0]),
            int(surface_points.shape[1]),
        ],
        "modB_min": modB_min,
        "modB_max": modB_max,
        "modB_mean": modB_mean,
        "modB_std": modB_std,
        "modB_coefficient_of_variation": (
            0.0 if modB_mean == 0.0 else float(modB_std / modB_mean)
        ),
        "mirror_ratio": mirror_ratio,
        "effective_inverse_aspect_ratio_epsilon": float(
            (mirror_ratio - 1.0) / (mirror_ratio + 1.0)
        ),
    }


def compute_topology_transport_diagnostics(surface, field):
    diagnostics = _transport_diagnostics_payload(
        status="partial",
        summary=(
            "Surface-field structure metrics evaluated from |B| on the Boozer "
            "surface grid. Exact Gamma_c and EffectiveRipple remain unavailable "
            "without a bounce-integral equilibrium transport backend."
        ),
        surface_field_structure=None,
        metric_status="unavailable",
        gamma_c_reason=_GAMMA_C_UNAVAILABLE_REASON,
        effective_ripple_reason=_EFFECTIVE_RIPPLE_UNAVAILABLE_REASON,
    )
    try:
        diagnostics["surface_field_structure"] = _surface_field_structure(
            surface, field
        )
    except Exception as error:
        diagnostics["status"] = "unavailable"
        diagnostics["summary"] = (
            "Surface-field structure metrics could not be evaluated from the "
            "current topology field model. Gamma_c and EffectiveRipple remain "
            "unavailable without a bounce-integral equilibrium transport backend."
        )
        diagnostics["surface_field_structure"] = {
            **_surface_field_structure_not_evaluated(
                str(error) or repr(error),
                status="error",
            ),
            "error_type": type(error).__name__,
        }
    return diagnostics


# ---------------------------------------------------------------------------
# Stopping criteria construction
# ---------------------------------------------------------------------------

STOP_LABELS_VALIDATION = [
    "surface_exit",
    "max_z_guardrail",
    "min_z_guardrail",
    "min_r_guardrail",
    "max_r_guardrail",
    "iteration_limit",
]

STOP_LABELS_DIAGNOSTIC = [
    "max_z_guardrail",
    "min_z_guardrail",
    "min_r_guardrail",
    "max_r_guardrail",
    "iteration_limit",
]

BROKEN_STOP_REASONS = frozenset({"iteration_limit"})

_TOPOLOGY_TRACE_MIN_ITERATIONS = 10_000
_TOPOLOGY_TRACE_ITERATIONS_PER_TMAX = 20_000
_TOPOLOGY_TRACE_MAX_ITERATIONS = 2_000_000


def topology_iteration_limit(tmax):
    """Return a generous but finite tracing iteration cap for a given horizon."""
    scaled_limit = int(
        np.ceil(float(max(tmax, 1.0)) * _TOPOLOGY_TRACE_ITERATIONS_PER_TMAX)
    )
    return int(
        min(
            _TOPOLOGY_TRACE_MAX_ITERATIONS,
            max(_TOPOLOGY_TRACE_MIN_ITERATIONS, scaled_limit),
        )
    )


def stop_reasons_indicate_broken(stop_reason_counts):
    return any(
        int(stop_reason_counts.get(reason, 0)) > 0 for reason in BROKEN_STOP_REASONS
    )


def _full_torus_surface(surface):
    """Create a full-torus copy of a surface for SurfaceClassifier.

    SurfaceClassifier builds a 3D grid over [0, 2pi] in phi, but
    surface.gamma() may only cover one half-period for stellarator-symmetric
    surfaces.  Evaluating signed_distance_from_surface at phi values outside
    the gamma coverage returns wrong distances, causing LevelsetStoppingCriterion
    to falsely trigger.  This helper creates a full-torus surface that covers
    all phi values.
    """
    from simsopt.geo import SurfaceXYZTensorFourier

    if not isinstance(surface, SurfaceXYZTensorFourier):
        return surface  # only XYZTensorFourier needs the fix
    nphi_input = len(surface.quadpoints_phi)
    ntheta_input = len(surface.quadpoints_theta)
    if nphi_input < 2:
        return surface
    phi_spacing = float(surface.quadpoints_phi[1] - surface.quadpoints_phi[0])
    phi_extent = phi_spacing * nphi_input
    if phi_extent <= 0.0:
        return surface
    phi_density = nphi_input / phi_extent
    full_torus_nphi = max(nphi_input, int(round(phi_density)))
    surf_full = SurfaceXYZTensorFourier(
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        mpol=surface.mpol,
        ntor=surface.ntor,
        quadpoints_phi=np.linspace(0, 1, full_torus_nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta_input, endpoint=False),
    )
    surf_full.x = surface.x
    return surf_full


def build_stopping_criteria(
    surface,
    include_surface_exit=True,
    box_padding=0.05,
    max_iterations=None,
    trace_domain=None,
):
    """Build stopping criteria from a Boozer surface.

    Returns (criteria_list, stop_labels) matching the convention used by
    both the topology gate and the strict Poincare validator.
    An explicit trace_domain overrides only the R/Z guardrail bounds; the
    surface-exit classifier, when enabled, is still built from surface.
    """
    from simsopt.field import (
        IterationStoppingCriterion,
        LevelsetStoppingCriterion,
        MaxRStoppingCriterion,
        MaxZStoppingCriterion,
        MinRStoppingCriterion,
        MinZStoppingCriterion,
    )
    from simsopt.geo import SurfaceClassifier

    resolved_trace_domain = (
        surface_trace_domain(surface, box_padding=box_padding)
        if trace_domain is None
        else trace_domain
    )

    box_criteria = [
        MaxZStoppingCriterion(resolved_trace_domain.stopping_zmax),
        MinZStoppingCriterion(-resolved_trace_domain.stopping_zmax),
        MinRStoppingCriterion(resolved_trace_domain.stopping_rmin),
        MaxRStoppingCriterion(resolved_trace_domain.stopping_rmax),
    ]
    iteration_limit = (
        _TOPOLOGY_TRACE_MAX_ITERATIONS
        if max_iterations is None
        else int(max_iterations)
    )
    if iteration_limit <= 0:
        raise ValueError("max_iterations must be positive")
    iteration_criterion = IterationStoppingCriterion(iteration_limit)

    if include_surface_exit:
        surf_for_classifier = _full_torus_surface(surface)
        classifier = SurfaceClassifier(surf_for_classifier, h=0.03, p=2)
        criteria = (
            [LevelsetStoppingCriterion(classifier.dist)]
            + box_criteria
            + [iteration_criterion]
        )
        return criteria, STOP_LABELS_VALIDATION
    else:
        return box_criteria + [iteration_criterion], STOP_LABELS_DIAGNOSTIC


# ---------------------------------------------------------------------------
# Metrics extraction (single source of truth)
# ---------------------------------------------------------------------------


def _normalize_trace_history(history):
    history = np.asarray(history, dtype=float)
    if history.size == 0:
        if history.ndim >= 2:
            return history.reshape((0, history.shape[-1]))
        return np.empty((0, 4), dtype=float)
    if history.ndim == 1:
        return history[None, :]
    return history


def _normalize_trace_hits(hits):
    hits = np.asarray(hits, dtype=float)
    if hits.size == 0:
        if hits.ndim >= 2 and hits.shape[-1] != 5:
            raise ValueError(
                f"Topology trace hit rows have invalid empty shape {hits.shape}"
            )
        return np.empty((0, 5), dtype=float)
    if hits.ndim == 1:
        return hits[None, :]
    return hits


def _trace_hits_before_first_stop(hits):
    normalized_hits = _normalize_trace_hits(hits)
    stop_indices = np.flatnonzero(normalized_hits[:, 1] < 0)
    if stop_indices.size == 0:
        return normalized_hits, None
    first_stop_index = int(stop_indices[0])
    return normalized_hits[:first_stop_index], normalized_hits[first_stop_index]


def validate_trace_arrays(fieldlines_tys, fieldlines_phi_hits):
    if len(fieldlines_tys) != len(fieldlines_phi_hits):
        raise ValueError("Topology tracing returned mismatched history and hit counts")
    for line_index, (history, hits) in enumerate(
        zip(fieldlines_tys, fieldlines_phi_hits)
    ):
        normalized_history = _normalize_trace_history(history)
        normalized_hits = _normalize_trace_hits(hits)
        if normalized_history.shape[1] < 4:
            raise ValueError(
                f"Topology trace history for line {line_index} has invalid shape {normalized_history.shape}"
            )
        if normalized_hits.shape[1] != 5:
            raise ValueError(
                f"Topology trace hit rows for line {line_index} have invalid shape {normalized_hits.shape}"
            )
        if normalized_history.size and not np.all(np.isfinite(normalized_history)):
            raise ValueError(
                f"Topology trace history for line {line_index} contains NaN/Inf"
            )
        if normalized_hits.size and not np.all(np.isfinite(normalized_hits)):
            raise ValueError(
                f"Topology trace hits for line {line_index} contain NaN/Inf"
            )


# A high-iota Poincare validation on an under-resolved Boozer surface can report a
# FALSE clean 50/50: the coarse surface makes the surface-exit guard too generous, so a
# marginal field line that truly exits is scored as surviving. Above this iota the fine
# grid is required before a "validated" verdict is trustworthy (the 127/32-vs-255/64
# false-positive footgun). Callers that don't supply iota + surface_resolution are
# unaffected (the guard is inert).
HIGH_IOTA_VALIDATION_THRESHOLD = 0.285
MIN_VALIDATION_SURFACE_NPHI = 255
MIN_VALIDATION_SURFACE_NTHETA = 64


def trace_metrics(
    fieldlines_tys,
    fieldlines_phi_hits,
    phis,
    stop_labels,
    mode="validation",
    *,
    iota=None,
    surface_resolution=None,
):
    """Extract structured metrics from field-line tracing results.

    This is the single implementation used by all scoring paths.

    ``iota`` and ``surface_resolution`` (an ``(nphi, ntheta)`` pair from the validation
    surface's quadpoint grid) are optional; when supplied they let the validation verdict
    refuse a clean 50/50 from an under-resolved surface at high iota (see the threshold
    constants above). Omitting them preserves the prior behaviour exactly.
    """
    validate_trace_arrays(fieldlines_tys, fieldlines_phi_hits)
    nfieldlines = len(fieldlines_tys)
    hit_counts = phi_hit_counts(fieldlines_phi_hits, phis)
    line_metrics = []
    stop_reason_counts = {label: 0 for label in stop_labels}
    survived = 0
    earliest_exit = None

    for seed_index, (history, hits) in enumerate(
        zip(fieldlines_tys, fieldlines_phi_hits)
    ):
        history = _normalize_trace_history(history)
        hits_before_stop, first_stop = _trace_hits_before_first_stop(hits)
        per_phi_counts = [
            int(np.sum(hits_before_stop[:, 1] == i)) for i in range(len(phis))
        ]
        final_time = float(history[-1, 0]) if len(history) else None

        if first_stop is None:
            survived += 1
            line_metrics.append(
                {
                    "seed_index": seed_index,
                    "survived": True,
                    "final_time": final_time,
                    "phi_hit_counts": per_phi_counts,
                    "stop_reason": None,
                    "first_exit_time": None,
                    "first_exit_angle": None,
                }
            )
            continue

        stop_index = int(-first_stop[1]) - 1
        reason = stop_reason_label(stop_index, stop_labels)
        stop_reason_counts.setdefault(reason, 0)
        stop_reason_counts[reason] += 1

        exit_time = float(first_stop[0])
        exit_angle = toroidal_angle(first_stop[2], first_stop[3])
        line_metric = {
            "seed_index": seed_index,
            "survived": False,
            "final_time": exit_time,
            "phi_hit_counts": per_phi_counts,
            "stop_reason": reason,
            "first_exit_time": exit_time,
            "first_exit_angle": exit_angle,
        }
        line_metrics.append(line_metric)

        if earliest_exit is None or exit_time < earliest_exit["first_exit_time"]:
            earliest_exit = line_metric

    survival_fraction = survived / nfieldlines if nfieldlines else 0.0
    if mode == "validation":
        if stop_reasons_indicate_broken(stop_reason_counts):
            validation_status = "broken"
        elif survived == nfieldlines:
            validation_status = "validated"
            if (
                iota is not None
                and abs(iota) >= HIGH_IOTA_VALIDATION_THRESHOLD
                and surface_resolution is not None
                and (
                    surface_resolution[0] < MIN_VALIDATION_SURFACE_NPHI
                    or surface_resolution[1] < MIN_VALIDATION_SURFACE_NTHETA
                )
            ):
                # Coarse high-iota surface: a clean count is not trustworthy here.
                validation_status = "under_resolved"
        else:
            validation_status = "fails_validation"
    else:
        validation_status = "diagnostic_only"

    exit_times = [
        m["first_exit_time"] for m in line_metrics if m["first_exit_time"] is not None
    ]
    mean_exit_time = float(np.mean(exit_times)) if exit_times else None

    return {
        "mode": mode,
        "nfieldlines": nfieldlines,
        "survived_lines": survived,
        "survival_fraction": survival_fraction,
        "per_phi_hit_counts": hit_counts,
        "stop_reason_counts": stop_reason_counts,
        "first_exit": earliest_exit,
        "mean_exit_time": mean_exit_time,
        "validation_status": validation_status,
        "line_metrics": line_metrics,
    }


def legacy_bounded_seed_fraction(
    fieldlines_phi_hits, cross_section_span, width_ratio=0.25
):
    """Scorer-setting-dependent bounded seed-line fraction.

    For each traced seed line, measures the radial+axial span of its Poincare
    hits. A line whose hits stay within a narrow band (< width_ratio times the
    cross-section span) and never triggers a stopping criterion is classified
    as bounded. The returned fraction is the count of bounded seed lines
    divided by the total traced seed count. Empty or inconclusive hit rows are
    counted as unbounded so they cannot inflate the fraction.

    This is not a seed-independent KAM invariant. It is tied to the scorer
    settings and seed contract: nfieldlines, nphis, tmax, width_ratio, and the
    midplane radial seed placement.
    """
    if cross_section_span <= 0.0 or not len(fieldlines_phi_hits):
        return 0.0, 0.0
    threshold = float(width_ratio) * float(cross_section_span)
    bounded = 0
    widths = []
    total = 0
    for line_hits in fieldlines_phi_hits:
        hits_before_stop, first_stop = _trace_hits_before_first_stop(line_hits)
        total += 1
        if hits_before_stop.size == 0:
            widths.append(float(cross_section_span))
            continue
        valid = hits_before_stop[hits_before_stop[:, 1] >= 0]
        if valid.shape[0] < 3:
            widths.append(float(cross_section_span))
            continue
        r = np.sqrt(valid[:, 2] ** 2 + valid[:, 3] ** 2)
        z = valid[:, 4]
        span_this = float(np.hypot(float(r.max() - r.min()), float(z.max() - z.min())))
        widths.append(span_this)
        lost = first_stop is not None
        if span_this <= threshold and not lost:
            bounded += 1
    if total == 0:
        return 0.0, 0.0
    return float(bounded) / float(total), float(np.median(widths))


def _surface_phi0_bounds(surface):
    cross_section = surface.cross_section(phi=0.0, thetas=512)
    radii = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    z = cross_section[:, 2]
    span = max(
        float(np.max(radii) - np.min(radii)),
        float(np.max(z) - np.min(z)),
        1.0e-3,
    )
    lower_radius = max(np.finfo(float).eps, float(np.min(radii) - span))
    return (
        (lower_radius, float(np.max(radii) + span)),
        (float(np.min(z) - span), float(np.max(z) + span)),
        (float(np.mean(radii)), float(np.mean(z))),
    )


def poloidal_axis_point(surface, bfield=None):
    """Return the magnetic-axis R/Z point used for WBA poloidal angles."""

    if bfield is None:
        raise ValueError("WBA poloidal-angle classification requires a magnetic field")
    r_bounds, z_bounds, initial_guess = _surface_phi0_bounds(surface)
    return locate_magnetic_axis_point(
        bfield,
        initial_guess,
        nfp=surface.nfp,
        phi0=0.0,
        r_bounds=r_bounds,
        z_bounds=z_bounds,
    )


def _wba_missing_magnetic_axis_result(fieldlines_phi_hits, settings):
    classifications = [
        missing_magnetic_axis_classification(
            seed_index=seed_index,
            return_count=0,
        )
        for seed_index, _line_hits in enumerate(fieldlines_phi_hits)
    ]
    summary = summarize_seed_classifications(classifications)
    return {
        **summary,
        "wba_axis": None,
        "wba_poincare_plane_index": int(settings.target_phi_index),
        "wba_settings": classifier_settings_payload(settings),
    }


def invariant_torus_classification(
    fieldlines_phi_hits,
    surface,
    *,
    plane_index=0,
    bfield=None,
    axis_point=None,
    min_returns=None,
):
    # min_returns=None keeps the strict-validation default (256); an explicit
    # value overrides only the returns bar (e.g. a cheaper in-loop pass).
    settings = DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS
    overrides = {}
    if int(plane_index) != settings.target_phi_index:
        overrides["target_phi_index"] = int(plane_index)
    if min_returns is not None and int(min_returns) != settings.min_returns:
        overrides["min_returns"] = int(min_returns)
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    if axis_point is None and bfield is None:
        return _wba_missing_magnetic_axis_result(fieldlines_phi_hits, settings)
    if axis_point is not None:
        axis = dict(axis_point)
    else:
        try:
            axis = poloidal_axis_point(surface, bfield=bfield)
        except MagneticAxisNotLocatedError:
            return {
                **wba_not_evaluated_payload(
                    WBA_EVALUATION_NOT_EVALUATED_AXIS_NOT_LOCATED
                ),
                "wba_seed_count": int(len(fieldlines_phi_hits)),
            }
    classifications = classify_fieldline_hits(
        fieldlines_phi_hits,
        stopped_before_hit_fn=_trace_hits_before_first_stop,
        axis_r=axis["r"],
        axis_z=axis["z"],
        settings=settings,
    )
    return {
        **summarize_seed_classifications(classifications),
        "wba_axis": axis,
        "wba_poincare_plane_index": int(settings.target_phi_index),
        "wba_settings": classifier_settings_payload(settings),
    }


def cross_section_span(surface):
    """Representative cross-section extent used by the legacy bounded-seed metric."""
    gamma = surface.gamma()
    r = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
    z = gamma[:, :, 2]
    return float(np.hypot(float(np.max(r) - np.min(r)), float(np.max(z) - np.min(z))))


def _normalized_line_lifetimes(line_metrics, tmax):
    """Return normalized line lifetimes in [0, 1], where 1 means full survival."""
    lifetimes = []
    for metric in line_metrics:
        if metric["survived"]:
            lifetimes.append(1.0)
            continue
        exit_time = float(metric["first_exit_time"])
        lifetimes.append(min(max(exit_time / tmax, 0.0), 1.0))
    return np.asarray(lifetimes, dtype=float)


def _empty_confinement_surrogate(effective_k, early_exit_threshold):
    return {
        "mean_line_loss": 0.0,
        "worst_k_line_loss": 0.0,
        "early_exit_fraction": 0.0,
        "confinement_loss": 0.0,
        "confinement_surrogate_k": effective_k,
        "confinement_early_exit_threshold": float(early_exit_threshold),
        "line_lifetimes": [],
        "line_losses": [],
    }


def summarize_confinement_surrogate(
    line_metrics,
    tmax,
    worst_k=3,
    early_exit_threshold=0.2,
    mean_weight=0.2,
    worst_weight=0.6,
    early_weight=0.2,
):
    """Build a tail-sensitive confinement surrogate from traced line metrics."""
    lifetimes = _normalized_line_lifetimes(line_metrics, tmax)
    effective_k = max(int(worst_k), 1)
    if lifetimes.size == 0:
        return _empty_confinement_surrogate(effective_k, early_exit_threshold)

    losses = 1.0 - lifetimes
    effective_k = min(effective_k, losses.size)
    worst_losses = np.partition(losses, -effective_k)[-effective_k:]
    early_exit_fraction = float(np.mean(lifetimes < early_exit_threshold))
    mean_line_loss = float(np.mean(losses))
    worst_k_line_loss = float(np.mean(worst_losses))
    confinement_loss = float(
        mean_weight * mean_line_loss
        + worst_weight * worst_k_line_loss
        + early_weight * early_exit_fraction
    )

    return {
        "mean_line_loss": mean_line_loss,
        "worst_k_line_loss": worst_k_line_loss,
        "early_exit_fraction": early_exit_fraction,
        "confinement_loss": confinement_loss,
        "confinement_surrogate_k": effective_k,
        "confinement_early_exit_threshold": float(early_exit_threshold),
        "line_lifetimes": lifetimes.tolist(),
        "line_losses": losses.tolist(),
    }


# ---------------------------------------------------------------------------
# Entry point: score_topology
# ---------------------------------------------------------------------------


def wba_not_evaluated_payload(reason):
    return {
        "invariant_torus_fraction": None,
        "invariant_torus_count": 0,
        "wba_fraction_denominator_policy": None,
        "wba_fraction_denominator_seed_count": 0,
        "wba_seed_count": 0,
        "wba_survived_seed_count": 0,
        "wba_classified_seed_count": 0,
        "wba_evaluation_state": reason,
        "wba_not_evaluated_reason": reason,
        "wba_classification_counts": {},
        "wba_rotation_number_median": None,
        "wba_matching_digits_min": None,
        "wba_matching_digits_median": None,
        "wba_seed_classifications": [],
        "wba_axis": None,
        "wba_poincare_plane_index": 0,
        "wba_settings": classifier_settings_payload(
            DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS
        ),
    }


def empty_topology_score_result(
    nfieldlines,
    tmax,
    *,
    surrogate_worst_k=1,
    surrogate_early_exit_threshold=0.0,
    kam_width_ratio=0.25,
    seed_contract=None,
    field_model=None,
    transport_diagnostics=None,
):
    effective_k = int(max(1, surrogate_worst_k))
    return {
        "survival_fraction": 0.0,
        "survived_lines": 0,
        "nfieldlines": int(nfieldlines),
        "tmax": float(tmax),
        "mean_exit_time": None,
        "confinement_score": 0.0,
        "mean_line_loss": 1.0,
        "worst_k_line_loss": 1.0,
        "early_exit_fraction": 1.0,
        "confinement_loss": np.inf,
        "confinement_surrogate_k": effective_k,
        "confinement_early_exit_threshold": float(surrogate_early_exit_threshold),
        "stop_reason_counts": {},
        "first_exit": None,
        "per_phi_hit_counts": [],
        "line_metrics": [],
        "line_lifetimes": [],
        "line_losses": [],
        "kam_fraction": None,
        "kam_fraction_semantics": None,
        **wba_not_evaluated_payload(WBA_EVALUATION_NOT_EVALUATED_NO_CLASSIFIED_SEEDS),
        "legacy_bounded_seed_fraction": 0.0,
        "legacy_bounded_seed_median_width": 0.0,
        "kam_median_width": 0.0,
        "kam_width_ratio": float(kam_width_ratio),
        "cross_section_span": 0.0,
        "seed_contract": seed_contract,
        "field_model": field_model,
        "transport_diagnostics": (
            topology_transport_diagnostics_not_evaluated("topology_score_not_evaluated")
            if transport_diagnostics is None
            else transport_diagnostics
        ),
    }


def finalize_topology_score_result(result, *, error_message=None, error_type=None):
    finalized = dict(result)
    broken = error_message is not None or stop_reasons_indicate_broken(
        finalized.get("stop_reason_counts", {})
    )
    if broken and error_message is None:
        error_message = "Topology tracing hit iteration limit"
        error_type = "IterationLimit"
    finalized["evaluation_state"] = "broken" if broken else "evaluated"
    finalized["broken"] = bool(broken)
    finalized["evaluation_error"] = error_message
    finalized["evaluation_error_type"] = error_type
    return finalized


def score_topology(
    surface,
    bfield,
    nfieldlines=12,
    tmax=50.0,
    tol=1e-7,
    nphis=4,
    surrogate_worst_k=3,
    surrogate_early_exit_threshold=0.2,
    surrogate_mean_weight=0.2,
    surrogate_worst_weight=0.6,
    surrogate_early_weight=0.2,
    kam_width_ratio=0.25,
    inset_fraction=0.05,
    field_policy=None,
    interpolation_grid=None,
    compute_transport_diagnostics=True,
    compute_invariant_torus_classification=True,
    magnetic_axis_point=None,
    wba_min_returns=None,
    design_only_results: Mapping[str, object] | None = None,
):
    """Score field-line confinement on a Boozer surface.

    Seeding is a midplane radial sweep (phi=0, Z=0). Returns a dict with
    survival_fraction, mean_exit_time, stop_reason_counts, per-line metrics,
    and invariant_torus_fraction (a WBA convergence-rate classifier, or None
    when the magnetic axis is not available/located or no survived seed has
    enough valid returns to classify). When compute_transport_diagnostics is
    False, skips the surface-field structure computation and returns a
    not-evaluated stub. The search-time survival gate also disables
    compute_invariant_torus_classification so WBA magnetic-axis solving cannot
    turn a confinement survival decision into a broken gate.

    wba_min_returns=None keeps the strict-validation Birkhoff-classifier returns
    bar (256). A caller may pass a lower value for a cheaper, coarser in-loop WBA
    signal (fewer Poincare returns required to classify); it overrides only the
    returns bar, not the other classifier settings.
    """
    assert_topology_field_allowed(
        bfield,
        design_only_results,
        allow_design_only_field=False,
        consumer="score_topology",
    )
    from simsopt.field import compute_fieldlines

    nfp = surface.nfp
    phis = [(i / nphis) * (2 * np.pi / nfp) for i in range(nphis)]
    resolved_field_policy = "auto" if field_policy is None else str(field_policy)

    stopping_criteria, stop_labels = build_stopping_criteria(
        surface,
        include_surface_exit=True,
        max_iterations=topology_iteration_limit(tmax),
    )
    radii = midplane_seed_radii(surface, nfieldlines, inset_fraction=inset_fraction)
    traced_field, field_model = prepare_topology_field(
        surface,
        bfield,
        tmax,
        field_policy=resolved_field_policy,
        interpolation_grid=interpolation_grid,
        seed_radii=radii,
    )
    if compute_transport_diagnostics:
        transport_diagnostics = compute_topology_transport_diagnostics(
            surface, traced_field
        )
    else:
        transport_diagnostics = topology_transport_diagnostics_not_evaluated(
            "skipped_by_caller"
        )

    seed_contract = build_midplane_seed_contract(nfieldlines, inset_fraction, radii)
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        traced_field,
        radii,
        np.zeros(int(nfieldlines)),
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
    )

    metrics = trace_metrics(
        fieldlines_tys, fieldlines_phi_hits, phis, stop_labels, mode="validation"
    )

    # Scalar confinement score: survival-weighted, with mean-exit-time tiebreak
    # Range: 0.0 (all exit immediately) to 1.0 (all survive full tmax)
    exit_times = [
        m["first_exit_time"]
        for m in metrics["line_metrics"]
        if m["first_exit_time"] is not None
    ]
    survived_times = [tmax for m in metrics["line_metrics"] if m["survived"]]
    all_times = exit_times + survived_times
    confinement_score = float(np.mean(all_times) / tmax) if all_times else 0.0
    surrogate = summarize_confinement_surrogate(
        metrics["line_metrics"],
        tmax,
        worst_k=surrogate_worst_k,
        early_exit_threshold=surrogate_early_exit_threshold,
        mean_weight=surrogate_mean_weight,
        worst_weight=surrogate_worst_weight,
        early_weight=surrogate_early_weight,
    )

    span = cross_section_span(surface)
    legacy_bounded_fraction, kam_median = legacy_bounded_seed_fraction(
        fieldlines_phi_hits,
        span,
        width_ratio=kam_width_ratio,
    )
    if compute_invariant_torus_classification:
        wba = invariant_torus_classification(
            fieldlines_phi_hits,
            surface,
            # Locate the axis on the exact field, not ``traced_field``: the axis
            # is a fixed-point solve whose residual acceptance is stricter than
            # an InterpolatedField's grid-error closure, whereas the long
            # confinement traces above tolerate interpolation.
            # Routing the interpolated tracer here made the WBA/topology score
            # report "broken" on valid configs whenever interpolation was active.
            bfield=bfield,
            axis_point=magnetic_axis_point,
            min_returns=wba_min_returns,
        )
    else:
        wba = wba_not_evaluated_payload(WBA_EVALUATION_NOT_EVALUATED_SKIPPED_BY_CALLER)
    invariant_torus_fraction = wba["invariant_torus_fraction"]
    promoted_kam_fraction = (
        None if invariant_torus_fraction is None else float(invariant_torus_fraction)
    )
    kam_fraction_semantics = (
        None if invariant_torus_fraction is None else KAM_FRACTION_SEMANTICS
    )

    return {
        "survival_fraction": metrics["survival_fraction"],
        "survived_lines": metrics["survived_lines"],
        "nfieldlines": nfieldlines,
        "tmax": tmax,
        "mean_exit_time": metrics["mean_exit_time"],
        "confinement_score": confinement_score,
        "mean_line_loss": surrogate["mean_line_loss"],
        "worst_k_line_loss": surrogate["worst_k_line_loss"],
        "early_exit_fraction": surrogate["early_exit_fraction"],
        "confinement_loss": surrogate["confinement_loss"],
        "confinement_surrogate_k": surrogate["confinement_surrogate_k"],
        "confinement_early_exit_threshold": surrogate[
            "confinement_early_exit_threshold"
        ],
        "stop_reason_counts": metrics["stop_reason_counts"],
        "first_exit": metrics["first_exit"],
        "per_phi_hit_counts": metrics["per_phi_hit_counts"],
        "line_metrics": metrics["line_metrics"],
        "line_lifetimes": surrogate["line_lifetimes"],
        "line_losses": surrogate["line_losses"],
        "kam_fraction": promoted_kam_fraction,
        "kam_fraction_semantics": kam_fraction_semantics,
        "invariant_torus_fraction": promoted_kam_fraction,
        "invariant_torus_count": int(wba["invariant_torus_count"]),
        "wba_fraction_denominator_policy": wba["wba_fraction_denominator_policy"],
        "wba_fraction_denominator_seed_count": int(
            wba["wba_fraction_denominator_seed_count"]
        ),
        "wba_seed_count": int(wba["wba_seed_count"]),
        "wba_survived_seed_count": int(wba["wba_survived_seed_count"]),
        "wba_classified_seed_count": int(wba["wba_classified_seed_count"]),
        "wba_evaluation_state": wba["wba_evaluation_state"],
        "wba_not_evaluated_reason": wba["wba_not_evaluated_reason"],
        "wba_classification_counts": wba["wba_classification_counts"],
        "wba_rotation_number_median": wba["wba_rotation_number_median"],
        "wba_matching_digits_min": wba["wba_matching_digits_min"],
        "wba_matching_digits_median": wba["wba_matching_digits_median"],
        "wba_seed_classifications": wba["wba_seed_classifications"],
        "wba_axis": wba["wba_axis"],
        "wba_poincare_plane_index": wba["wba_poincare_plane_index"],
        "wba_settings": wba["wba_settings"],
        "legacy_bounded_seed_fraction": float(legacy_bounded_fraction),
        "legacy_bounded_seed_median_width": float(kam_median),
        "kam_median_width": float(kam_median),
        "kam_width_ratio": float(kam_width_ratio),
        "cross_section_span": float(span),
        "seed_contract": seed_contract,
        "field_model": field_model,
        "transport_diagnostics": transport_diagnostics,
    }


def safe_score_topology(
    surface,
    bfield,
    *,
    nfieldlines,
    tmax,
    tol=1e-7,
    design_only_results: Mapping[str, object] | None = None,
    **kwargs,
):
    try:
        return finalize_topology_score_result(
            score_topology(
                surface,
                bfield,
                nfieldlines=nfieldlines,
                tmax=tmax,
                tol=tol,
                design_only_results=design_only_results,
                **kwargs,
            )
        )
    except Exception as error:
        resolved_field_policy = kwargs.get("field_policy") or "auto"
        field_model = {
            "policy": str(resolved_field_policy),
            "selected_mode": "native",
            "reason": "uninitialized",
            "tmax_threshold": float(TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD),
            "grid": None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_rel_error": None,
        }
        return finalize_topology_score_result(
            empty_topology_score_result(
                nfieldlines,
                tmax,
                surrogate_worst_k=kwargs.get("surrogate_worst_k", 1),
                surrogate_early_exit_threshold=kwargs.get(
                    "surrogate_early_exit_threshold",
                    0.0,
                ),
                kam_width_ratio=kwargs.get("kam_width_ratio", 0.25),
                seed_contract=build_midplane_seed_contract(
                    nfieldlines,
                    kwargs.get("inset_fraction", 0.05),
                ),
                field_model=field_model,
                transport_diagnostics=topology_transport_diagnostics_not_evaluated(
                    "topology_score_failed_before_transport_metrics"
                ),
            ),
            error_message=str(error) or repr(error),
            error_type=type(error).__name__,
        )
