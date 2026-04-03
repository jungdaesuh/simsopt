"""Shared topology scoring for field-line confinement evaluation.

Single source of truth for field-line tracing metrics used by:
- the search-time topology gate in single_stage_banana_example.py
- the callback medium-fidelity scorer
- the strict Poincare validator in poincare_surfaces.py

All three paths use the same stopping criteria, seed logic, and metric
computation so that metrics cannot drift between callback and validation.
"""

import numpy as np
from simsopt.field import compute_fieldlines
from simsopt.field import (
    LevelsetStoppingCriterion,
    MaxZStoppingCriterion,
    MinZStoppingCriterion,
    MaxRStoppingCriterion,
    MinRStoppingCriterion,
)
from simsopt.geo import SurfaceClassifier


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
        int(sum(np.sum(fieldline[:, 1] == i) for fieldline in fieldlines_phi_hits))
        for i in range(len(phis))
    ]


# ---------------------------------------------------------------------------
# Stopping criteria construction
# ---------------------------------------------------------------------------

STOP_LABELS_VALIDATION = [
    "surface_exit",
    "max_z_guardrail",
    "min_z_guardrail",
    "min_r_guardrail",
    "max_r_guardrail",
]

STOP_LABELS_DIAGNOSTIC = [
    "max_z_guardrail",
    "min_z_guardrail",
    "min_r_guardrail",
    "max_r_guardrail",
]


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


def build_stopping_criteria(surface, include_surface_exit=True, box_padding=0.05):
    """Build stopping criteria from a Boozer surface.

    Returns (criteria_list, stop_labels) matching the convention used by
    both the topology gate and the strict Poincare validator.
    """
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

    if include_surface_exit:
        surf_for_classifier = _full_torus_surface(surface)
        classifier = SurfaceClassifier(surf_for_classifier, h=0.03, p=2)
        criteria = [LevelsetStoppingCriterion(classifier.dist)] + box_criteria
        return criteria, STOP_LABELS_VALIDATION
    else:
        return box_criteria, STOP_LABELS_DIAGNOSTIC


# ---------------------------------------------------------------------------
# Metrics extraction (single source of truth)
# ---------------------------------------------------------------------------

def trace_metrics(fieldlines_tys, fieldlines_phi_hits, phis, stop_labels, mode="validation"):
    """Extract structured metrics from field-line tracing results.

    This is the single implementation used by all scoring paths.
    """
    nfieldlines = len(fieldlines_tys)
    hit_counts = phi_hit_counts(fieldlines_phi_hits, phis)
    line_metrics = []
    stop_reason_counts = {label: 0 for label in stop_labels}
    survived = 0
    earliest_exit = None

    for seed_index, (history, hits) in enumerate(zip(fieldlines_tys, fieldlines_phi_hits)):
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

    stopping_criteria, stop_labels = build_stopping_criteria(surface, include_surface_exit=True)

    R0 = midplane_seed_radii(surface, nfieldlines)
    Z0 = np.zeros((nfieldlines,))

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield,
        R0, Z0,
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
    }
