"""Shared topology scoring for field-line confinement evaluation.

Single source of truth for field-line tracing metrics used by:
- the search-time topology gate in single_stage_banana_example.py
- the callback medium-fidelity scorer
- the strict Poincare validator in poincare_surfaces.py

All three paths use the same stopping criteria, seed logic, and metric
computation so that metrics cannot drift between callback and validation.
"""

from dataclasses import dataclass

import numpy as np
import simsoptpp as sopp


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


def padded_bounds(rmin, rmax, zmax, radial_padding_fraction=0.05, axial_padding_fraction=0.05):
    """Add a modest interpolation buffer around the validation surface bounds."""
    rpad = max(radial_padding_fraction * (rmax - rmin), 0.02)
    zpad = max(axial_padding_fraction * zmax, 0.01)
    return max(0.0, rmin - rpad), rmax + rpad, zmax + zpad


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
                np.sum(_normalize_trace_hits(fieldline)[:, 1] == i)
                for fieldline in fieldlines_phi_hits
            )
        )
        for i in range(len(phis))
    ]


@dataclass(frozen=True)
class TopologySeedTier:
    name: str
    default_seed_plane_count: int
    default_field_policy: str


TOPOLOGY_SEED_TIERS = {
    "cheap": TopologySeedTier(
        name="cheap",
        default_seed_plane_count=2,
        default_field_policy="never",
    ),
    "medium": TopologySeedTier(
        name="medium",
        default_seed_plane_count=4,
        default_field_policy="auto",
    ),
    "strict": TopologySeedTier(
        name="strict",
        default_seed_plane_count=8,
        default_field_policy="auto",
    ),
}

TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD = 50.0
TOPOLOGY_INTERPOLATION_GRID = {
    "degree": 3,
    "nr": 40,
    "nphi": 40,
    "nz": 20,
}
_TOPOLOGY_SEED_THETA_SAMPLES = 512
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


def _resolve_topology_seed_tier(seed_tier):
    try:
        return TOPOLOGY_SEED_TIERS[str(seed_tier)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported topology seed tier {seed_tier!r}; "
            f"expected one of {sorted(TOPOLOGY_SEED_TIERS)}"
        ) from error


def _resolve_seed_plane_count(seed_tier, nfieldlines, seed_plane_count):
    if nfieldlines <= 0:
        return 0
    default_plane_count = _resolve_topology_seed_tier(seed_tier).default_seed_plane_count
    resolved_plane_count = (
        default_plane_count
        if seed_plane_count is None
        else int(seed_plane_count)
    )
    if resolved_plane_count <= 0:
        raise ValueError("seed_plane_count must be positive")
    return min(int(nfieldlines), resolved_plane_count)


def _seed_plane_angles(nfp, plane_count):
    if plane_count <= 0:
        return np.empty((0,), dtype=float)
    field_period = (2.0 * np.pi) / float(max(int(nfp), 1))
    return np.linspace(0.0, field_period, plane_count, endpoint=False, dtype=float)


def _lines_per_plane(nfieldlines, plane_count):
    if plane_count <= 0:
        return []
    base, remainder = divmod(int(nfieldlines), int(plane_count))
    return [
        int(base + (1 if plane_index < remainder else 0))
        for plane_index in range(int(plane_count))
    ]


def _cross_section_rz(surface, phi_abs, theta_samples=_TOPOLOGY_SEED_THETA_SAMPLES):
    cross_section = np.asarray(
        surface.cross_section(phi=float(phi_abs) / (2.0 * np.pi), thetas=theta_samples),
        dtype=float,
    )
    rz = np.empty((cross_section.shape[0], 2), dtype=float)
    rz[:, 0] = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    rz[:, 1] = cross_section[:, 2]
    return cross_section, rz


def _angle_distance(values, target):
    wrapped = np.mod(values - target + np.pi, 2.0 * np.pi) - np.pi
    return np.abs(wrapped)


def _inset_rz_point(boundary_rz, centroid_rz, inset_fraction, min_inset):
    direction = np.asarray(boundary_rz, dtype=float) - np.asarray(centroid_rz, dtype=float)
    radius = float(np.linalg.norm(direction))
    if radius <= 0.0:
        return np.asarray(boundary_rz, dtype=float)
    inset = min(max(float(inset_fraction) * radius, float(min_inset)), 0.45 * radius)
    scale = max((radius - inset) / radius, 0.0)
    return np.asarray(centroid_rz, dtype=float) + scale * direction


def build_topology_seed_points(
    surface,
    nfieldlines,
    *,
    seed_tier="medium",
    seed_plane_count=None,
    inset_fraction=0.08,
    min_inset=0.01,
    theta_samples=_TOPOLOGY_SEED_THETA_SAMPLES,
):
    resolved_nfieldlines = int(nfieldlines)
    if resolved_nfieldlines < 0:
        raise ValueError("nfieldlines must be non-negative")

    resolved_plane_count = _resolve_seed_plane_count(
        seed_tier,
        resolved_nfieldlines,
        seed_plane_count,
    )
    plane_angles = _seed_plane_angles(surface.nfp, resolved_plane_count)
    points_per_plane = _lines_per_plane(resolved_nfieldlines, resolved_plane_count)

    xyz_inits = []
    seed_points = []
    for plane_index, (phi_abs, plane_line_count) in enumerate(
        zip(plane_angles, points_per_plane)
    ):
        if plane_line_count <= 0:
            continue
        _, rz = _cross_section_rz(surface, float(phi_abs), theta_samples=theta_samples)
        centroid_rz = np.mean(rz, axis=0)
        boundary_angles = np.mod(
            np.arctan2(rz[:, 1] - centroid_rz[1], rz[:, 0] - centroid_rz[0]),
            2.0 * np.pi,
        )
        poloidal_step = (2.0 * np.pi) / float(plane_line_count)
        poloidal_offset = (
            float(plane_index) / float(max(resolved_plane_count, 1))
        ) * poloidal_step
        target_angles = np.mod(
            poloidal_offset
            + np.arange(plane_line_count, dtype=float) * poloidal_step,
            2.0 * np.pi,
        )
        for target_angle in target_angles:
            boundary_index = int(np.argmin(_angle_distance(boundary_angles, target_angle)))
            seeded_rz = _inset_rz_point(
                rz[boundary_index],
                centroid_rz,
                inset_fraction,
                min_inset,
            )
            seeded_r = float(seeded_rz[0])
            seeded_z = float(seeded_rz[1])
            seeded_xyz = np.array(
                [
                    seeded_r * np.cos(phi_abs),
                    seeded_r * np.sin(phi_abs),
                    seeded_z,
                ],
                dtype=float,
            )
            xyz_inits.append(seeded_xyz)
            seed_points.append(
                {
                    "phi": float(phi_abs),
                    "R": seeded_r,
                    "Z": seeded_z,
                    "target_poloidal_angle": float(target_angle),
                }
            )

    return {
        "xyz_inits": np.asarray(xyz_inits, dtype=float).reshape((-1, 3)),
        "contract": {
            "tier": str(seed_tier),
            "seed_plane_count": int(resolved_plane_count),
            "seed_plane_angles": [float(angle) for angle in plane_angles],
            "lines_per_plane": [int(count) for count in points_per_plane],
            "poloidal_sampling": "evenly_spaced_cross_section",
            "theta_samples": int(theta_samples),
            "inset_fraction": float(inset_fraction),
            "min_inset": float(min_inset),
        },
        "seed_points": seed_points,
    }


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
):
    from simsopt.field import InterpolatedField

    resolved_policy = str(field_policy)
    if resolved_policy not in {"auto", "always", "never"}:
        raise ValueError(
            f"Unsupported topology field policy {field_policy!r}; expected "
            "'auto', 'always', or 'never'"
        )
    if isinstance(bfield, InterpolatedField):
        return bfield, {
            "policy": resolved_policy,
            "selected_mode": "pre_interpolated",
            "reason": "already_interpolated",
            "tmax_threshold": float(TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD),
            "grid": None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_rel_error": None,
        }
    should_interpolate = (
        resolved_policy == "always"
        or (
            resolved_policy == "auto"
            and float(tmax) >= float(TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD)
        )
    )
    if not should_interpolate:
        reason = "below_threshold" if resolved_policy == "auto" else "explicit_never"
        return bfield, {
            "policy": resolved_policy,
            "selected_mode": "native",
            "reason": reason,
            "tmax_threshold": float(TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD),
            "grid": None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_rel_error": None,
        }

    grid = dict(TOPOLOGY_INTERPOLATION_GRID)
    if interpolation_grid is not None:
        grid.update(interpolation_grid)
    gamma = surface.gamma()
    rr = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
    zz = gamma[:, :, 2]
    interp_rmin, interp_rmax, interp_zmax = padded_bounds(
        float(np.min(rr)),
        float(np.max(rr)),
        float(np.max(np.abs(zz))),
    )
    zrange = (
        (0.0, float(interp_zmax), int(grid["nz"]))
        if getattr(surface, "stellsym", False)
        else (-float(interp_zmax), float(interp_zmax), int(grid["nz"]))
    )
    interpolated_field = InterpolatedField(
        bfield,
        int(grid["degree"]),
        (float(interp_rmin), float(interp_rmax), int(grid["nr"])),
        (0.0, (2.0 * np.pi) / float(max(int(surface.nfp), 1)), int(grid["nphi"])),
        zrange,
        True,
        nfp=int(surface.nfp),
        stellsym=bool(getattr(surface, "stellsym", False)),
    )
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
    return interpolated_field, {
        "policy": resolved_policy,
        "selected_mode": "interpolated",
        "reason": "explicit_always" if resolved_policy == "always" else "tmax_threshold",
        "tmax_threshold": float(TOPOLOGY_INTERPOLATION_TMAX_THRESHOLD),
        "grid": {
            "degree": int(grid["degree"]),
            "nr": int(grid["nr"]),
            "nphi": int(grid["nphi"]),
            "nz": int(grid["nz"]),
        },
        "max_abs_error": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_error": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_error": max_rel_error,
    }


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
        diagnostics["surface_field_structure"] = _surface_field_structure(surface, field)
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
    scaled_limit = int(np.ceil(float(max(tmax, 1.0)) * _TOPOLOGY_TRACE_ITERATIONS_PER_TMAX))
    return int(
        min(
            _TOPOLOGY_TRACE_MAX_ITERATIONS,
            max(_TOPOLOGY_TRACE_MIN_ITERATIONS, scaled_limit),
        )
    )


def stop_reasons_indicate_broken(stop_reason_counts):
    return any(int(stop_reason_counts.get(reason, 0)) > 0 for reason in BROKEN_STOP_REASONS)


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
        nfp=surface.nfp, stellsym=surface.stellsym,
        mpol=surface.mpol, ntor=surface.ntor,
        quadpoints_phi=np.linspace(0, 1, full_torus_nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta_input, endpoint=False))
    surf_full.x = surface.x
    return surf_full


def build_stopping_criteria(
    surface,
    include_surface_exit=True,
    box_padding=0.05,
    max_iterations=None,
):
    """Build stopping criteria from a Boozer surface.

    Returns (criteria_list, stop_labels) matching the convention used by
    both the topology gate and the strict Poincare validator.
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

    gamma = surface.gamma()
    rr = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
    zz = gamma[:, :, 2]
    rmin = float(np.min(rr))
    rmax = float(np.max(rr))
    zmax = float(np.max(np.abs(zz)))

    box_criteria = [
        MaxZStoppingCriterion(zmax * (1 + box_padding)),
        MinZStoppingCriterion(-zmax * (1 + box_padding)),
        MinRStoppingCriterion(rmin * (1 - box_padding)),
        MaxRStoppingCriterion(rmax * (1 + box_padding)),
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
        criteria = [LevelsetStoppingCriterion(classifier.dist)] + box_criteria + [iteration_criterion]
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


def validate_trace_arrays(fieldlines_tys, fieldlines_phi_hits):
    if len(fieldlines_tys) != len(fieldlines_phi_hits):
        raise ValueError("Topology tracing returned mismatched history and hit counts")
    for line_index, (history, hits) in enumerate(zip(fieldlines_tys, fieldlines_phi_hits)):
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
            raise ValueError(f"Topology trace history for line {line_index} contains NaN/Inf")
        if normalized_hits.size and not np.all(np.isfinite(normalized_hits)):
            raise ValueError(f"Topology trace hits for line {line_index} contain NaN/Inf")


def trace_metrics(fieldlines_tys, fieldlines_phi_hits, phis, stop_labels, mode="validation"):
    """Extract structured metrics from field-line tracing results.

    This is the single implementation used by all scoring paths.
    """
    validate_trace_arrays(fieldlines_tys, fieldlines_phi_hits)
    nfieldlines = len(fieldlines_tys)
    hit_counts = phi_hit_counts(fieldlines_phi_hits, phis)
    line_metrics = []
    stop_reason_counts = {label: 0 for label in stop_labels}
    survived = 0
    earliest_exit = None

    for seed_index, (history, hits) in enumerate(zip(fieldlines_tys, fieldlines_phi_hits)):
        history = _normalize_trace_history(history)
        hits = _normalize_trace_hits(hits)
        negative_hits = hits[hits[:, 1] < 0]
        first_stop = negative_hits[0] if negative_hits.size else None
        per_phi_counts = [int(np.sum(hits[:, 1] == i)) for i in range(len(phis))]
        final_time = float(history[-1, 0]) if len(history) else None

        if first_stop is None:
            survived += 1
            line_metrics.append({
                "seed_index": seed_index,
                "survived": True,
                "final_time": final_time,
                "phi_hit_counts": per_phi_counts,
                "stop_reason": None,
                "first_exit_time": None,
                "first_exit_angle": None,
            })
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
        else:
            validation_status = "validated" if survived == nfieldlines else "fails_validation"
    else:
        validation_status = "diagnostic_only"

    exit_times = [m["first_exit_time"] for m in line_metrics if m["first_exit_time"] is not None]
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


def empty_topology_score_result(
    nfieldlines,
    tmax,
    *,
    surrogate_worst_k=1,
    surrogate_early_exit_threshold=0.0,
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
        "seed_contract": seed_contract,
        "field_model": field_model,
        "transport_diagnostics": (
            topology_transport_diagnostics_not_evaluated(
                "topology_score_not_evaluated"
            )
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
    seed_tier="medium",
    seed_plane_count=None,
    field_policy=None,
    interpolation_grid=None,
):
    """Score field-line confinement on a Boozer surface.

    This is the reusable entry point for all topology scoring:
    - search-time gate: nfieldlines=4, tmax=2
    - callback medium scorer: nfieldlines=12, tmax=50
    - strict validation: nfieldlines=50, tmax=7000

    Returns a dict with survival_fraction, mean_exit_time, stop_reason_counts,
    and per-line metrics.
    """
    nfp = surface.nfp
    phis = [(i / nphis) * (2 * np.pi / nfp) for i in range(nphis)]
    resolved_field_policy = (
        _resolve_topology_seed_tier(seed_tier).default_field_policy
        if field_policy is None
        else str(field_policy)
    )

    stopping_criteria, stop_labels = build_stopping_criteria(
        surface,
        include_surface_exit=True,
        max_iterations=topology_iteration_limit(tmax),
    )
    seed_bundle = build_topology_seed_points(
        surface,
        nfieldlines,
        seed_tier=seed_tier,
        seed_plane_count=seed_plane_count,
    )
    traced_field, field_model = prepare_topology_field(
        surface,
        bfield,
        tmax,
        field_policy=resolved_field_policy,
        interpolation_grid=interpolation_grid,
    )
    transport_diagnostics = compute_topology_transport_diagnostics(surface, traced_field)
    fieldlines_tys, fieldlines_phi_hits = trace_fieldlines_xyz(
        traced_field,
        seed_bundle["xyz_inits"],
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
    )

    metrics = trace_metrics(fieldlines_tys, fieldlines_phi_hits, phis, stop_labels, mode="validation")

    # Scalar confinement score: survival-weighted, with mean-exit-time tiebreak
    # Range: 0.0 (all exit immediately) to 1.0 (all survive full tmax)
    exit_times = [m["first_exit_time"] for m in metrics["line_metrics"] if m["first_exit_time"] is not None]
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
        "confinement_early_exit_threshold": surrogate["confinement_early_exit_threshold"],
        "stop_reason_counts": metrics["stop_reason_counts"],
        "first_exit": metrics["first_exit"],
        "per_phi_hit_counts": metrics["per_phi_hit_counts"],
        "line_metrics": metrics["line_metrics"],
        "line_lifetimes": surrogate["line_lifetimes"],
        "line_losses": surrogate["line_losses"],
        "seed_contract": seed_bundle["contract"],
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
    **kwargs,
):
    resolved_field_policy = kwargs.get("field_policy")
    if resolved_field_policy is None:
        resolved_field_policy = _resolve_topology_seed_tier(
            kwargs.get("seed_tier", "medium")
        ).default_field_policy
    seed_contract = {
        "tier": str(kwargs.get("seed_tier", "medium")),
        "seed_plane_count": None,
        "seed_plane_angles": [],
        "lines_per_plane": [],
        "poloidal_sampling": "evenly_spaced_cross_section",
        "theta_samples": int(_TOPOLOGY_SEED_THETA_SAMPLES),
        "inset_fraction": 0.08,
        "min_inset": 0.01,
    }
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
    transport_diagnostics = topology_transport_diagnostics_not_evaluated(
        "topology_score_failed_before_transport_metrics"
    )
    try:
        return finalize_topology_score_result(
            score_topology(
                surface,
                bfield,
                nfieldlines=nfieldlines,
                tmax=tmax,
                tol=tol,
                **kwargs,
            )
        )
    except Exception as error:
        return finalize_topology_score_result(
            empty_topology_score_result(
                nfieldlines,
                tmax,
                surrogate_worst_k=kwargs.get("surrogate_worst_k", 1),
                surrogate_early_exit_threshold=kwargs.get(
                    "surrogate_early_exit_threshold",
                    0.0,
                ),
                seed_contract=seed_contract,
                field_model=field_model,
                transport_diagnostics=transport_diagnostics,
            ),
            error_message=str(error) or repr(error),
            error_type=type(error).__name__,
        )
