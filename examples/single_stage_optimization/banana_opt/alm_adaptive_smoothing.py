from __future__ import annotations

import numpy as np


ALM_DISTANCE_GAP_CONSTRAINT_NAMES = frozenset(
    (
        "coil_coil_spacing",
        "coil_surface_spacing",
        "surface_surface_spacing",
    )
)
ALM_CURVATURE_GAP_CONSTRAINT_NAMES = frozenset(("max_curvature", "poloidal_extent"))
ALM_GAP_SHRINK_RATE = 0.25
ALM_SMOOTHING_FLOOR_FRACTION = 1.0 / 8.0


def normalized_hard_surrogate_gap_counts(history_entry):
    constraint_names = [str(name) for name in history_entry["constraint_names"]]
    gaps = np.asarray(
        history_entry["surrogate_minus_hard_normalized_gap"],
        dtype=float,
    )
    mismatches = [
        bool(value)
        for value in history_entry["surrogate_hard_sign_mismatch_by_constraint"]
    ]
    feasibility_tolerance = float(history_entry["effective_feasibility_tolerance"])
    counts = {"distance": 0, "curvature": 0}
    for index, constraint_name in enumerate(constraint_names):
        mismatch = mismatches[index]
        gap_exceeds_gate = abs(float(gaps[index])) > feasibility_tolerance
        if not (mismatch or gap_exceeds_gate):
            continue
        if constraint_name in ALM_DISTANCE_GAP_CONSTRAINT_NAMES:
            counts["distance"] += 1
        if constraint_name in ALM_CURVATURE_GAP_CONSTRAINT_NAMES:
            counts["curvature"] += 1
    return counts


def shrink_alm_smoothing_for_gap_count(smoothing, smoothing_min, gap_count):
    factor = 1.0 / (1.0 + ALM_GAP_SHRINK_RATE * float(gap_count))
    return max(float(smoothing_min), float(smoothing) * factor)


def adapt_alm_smoothing_from_history(
    distance_smoothing,
    curvature_smoothing,
    history_entry,
    *,
    distance_smoothing_min,
    curvature_smoothing_min,
):
    counts = normalized_hard_surrogate_gap_counts(history_entry)
    return {
        "distance_smoothing": shrink_alm_smoothing_for_gap_count(
            distance_smoothing,
            distance_smoothing_min,
            counts["distance"],
        ),
        "curvature_smoothing": shrink_alm_smoothing_for_gap_count(
            curvature_smoothing,
            curvature_smoothing_min,
            counts["curvature"],
        ),
        "gap_counts": counts,
    }
