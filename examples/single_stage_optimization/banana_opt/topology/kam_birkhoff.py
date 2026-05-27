from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from collections.abc import Sequence

import numpy as np


DEFAULT_WBA_TARGET_PHI_INDEX = 0
DEFAULT_WBA_MIN_RETURNS = 256
DEFAULT_WBA_INVARIANT_MIN_DIGITS = 8.0
DEFAULT_WBA_ISLAND_MIN_DIGITS = 4.0
DEFAULT_WBA_RATIONAL_MAX_DENOMINATOR = 24
DEFAULT_WBA_RATIONAL_TOLERANCE = 1.0e-4
DEFAULT_WBA_DIFF_FLOOR = 1.0e-15
KAM_FRACTION_SEMANTICS = "weighted_birkhoff_invariant_torus_fraction"

KAM_CLASS_INVARIANT_TORUS = "invariant_torus"
KAM_CLASS_ISLAND_CHAIN = "island_chain"
KAM_CLASS_CHAOTIC = "chaotic"
KAM_CLASS_INSUFFICIENT_RETURNS = "insufficient_returns"
KAM_CLASS_LOST = "lost"


@dataclass(frozen=True, slots=True)
class BirkhoffClassifierSettings:
    """Controls the WBA convergence-rate invariant-torus classifier."""

    target_phi_index: int = DEFAULT_WBA_TARGET_PHI_INDEX
    min_returns: int = DEFAULT_WBA_MIN_RETURNS
    invariant_digits_min: float = DEFAULT_WBA_INVARIANT_MIN_DIGITS
    island_digits_min: float = DEFAULT_WBA_ISLAND_MIN_DIGITS
    island_rational_tolerance: float = DEFAULT_WBA_RATIONAL_TOLERANCE
    island_max_denominator: int = DEFAULT_WBA_RATIONAL_MAX_DENOMINATOR
    diff_floor: float = DEFAULT_WBA_DIFF_FLOOR


DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS = BirkhoffClassifierSettings()


@dataclass(frozen=True, slots=True)
class RationalApproximation:
    numerator: int
    denominator: int
    value: float
    error: float


@dataclass(frozen=True, slots=True)
class SeedClassification:
    seed_index: int
    classification: str
    return_count: int
    rotation_number: float | None
    matching_digits: float | None
    first_half_rotation_number: float | None
    second_half_rotation_number: float | None
    nearest_rational: dict[str, object] | None
    reason: str


def classifier_settings_payload(
    settings: BirkhoffClassifierSettings = DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS,
) -> dict[str, object]:
    return asdict(settings)


def weighted_birkhoff_weights(count: int) -> np.ndarray:
    """Return normalized C-infinity bump weights for a Birkhoff average."""

    n = int(count)
    if n <= 0:
        raise ValueError("weighted Birkhoff average requires at least one sample")
    t = (np.arange(n, dtype=float) + 0.5) / float(n)
    weights = np.exp(-1.0 / (t * (1.0 - t)))
    return weights / float(np.sum(weights))


def weighted_birkhoff_average(values: Sequence[float]) -> float:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1:
        raise ValueError("weighted Birkhoff average requires a one-dimensional series")
    if samples.size == 0:
        raise ValueError("weighted Birkhoff average requires at least one sample")
    if not np.all(np.isfinite(samples)):
        raise ValueError("weighted Birkhoff average received NaN/Inf samples")
    return float(np.sum(weighted_birkhoff_weights(samples.size) * samples))


def normalized_rotation_difference(left: float, right: float) -> float:
    diff = float(left) - float(right)
    return abs(diff - round(diff))


def matching_digits_from_difference(
    difference: float,
    *,
    floor: float = DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS.diff_floor,
) -> float:
    return float(-math.log10(max(float(difference), float(floor))))


def nearest_rational(
    value: float,
    *,
    max_denominator: int,
) -> RationalApproximation:
    normalized_value = float(value) % 1.0
    fraction = Fraction(normalized_value).limit_denominator(int(max_denominator))
    rational_value = float(fraction)
    return RationalApproximation(
        numerator=int(fraction.numerator),
        denominator=int(fraction.denominator),
        value=rational_value,
        error=normalized_rotation_difference(normalized_value, rational_value),
    )


def classify_angle_series(
    angles_rad: Sequence[float],
    *,
    seed_index: int = 0,
    settings: BirkhoffClassifierSettings = DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS,
) -> SeedClassification:
    angles = np.unwrap(np.asarray(angles_rad, dtype=float))
    if angles.ndim != 1:
        raise ValueError("WBA classification requires a one-dimensional angle series")
    if not np.all(np.isfinite(angles)):
        raise ValueError("WBA classification received NaN/Inf angles")

    return_count = int(angles.size)
    if return_count < int(settings.min_returns):
        return SeedClassification(
            seed_index=int(seed_index),
            classification=KAM_CLASS_INSUFFICIENT_RETURNS,
            return_count=return_count,
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="insufficient_poincare_returns",
        )

    increments = np.diff(angles) / (2.0 * np.pi)
    split = increments.size // 2
    first_rotation_raw = weighted_birkhoff_average(increments[:split])
    second_rotation_raw = weighted_birkhoff_average(increments[split:])
    rotation_number_raw = weighted_birkhoff_average(increments)
    first_rotation = first_rotation_raw % 1.0
    second_rotation = second_rotation_raw % 1.0
    rotation_number = rotation_number_raw % 1.0
    difference = normalized_rotation_difference(
        first_rotation_raw,
        second_rotation_raw,
    )
    matching_digits = matching_digits_from_difference(
        difference,
        floor=settings.diff_floor,
    )
    rational = nearest_rational(
        rotation_number,
        max_denominator=settings.island_max_denominator,
    )
    rational_payload = asdict(rational)

    if (
        matching_digits >= float(settings.island_digits_min)
        and rational.error <= float(settings.island_rational_tolerance)
    ):
        classification = KAM_CLASS_ISLAND_CHAIN
        reason = "weighted_birkhoff_average_near_low_order_rational"
    elif matching_digits >= float(settings.invariant_digits_min):
        classification = KAM_CLASS_INVARIANT_TORUS
        reason = "weighted_birkhoff_average_converged"
    else:
        classification = KAM_CLASS_CHAOTIC
        reason = "weighted_birkhoff_average_not_converged"

    return SeedClassification(
        seed_index=int(seed_index),
        classification=classification,
        return_count=return_count,
        rotation_number=float(rotation_number),
        matching_digits=float(matching_digits),
        first_half_rotation_number=float(first_rotation),
        second_half_rotation_number=float(second_rotation),
        nearest_rational=rational_payload,
        reason=reason,
    )


def classify_return_points(
    return_points_xyz: Sequence[Sequence[float]],
    *,
    axis_r: float,
    axis_z: float,
    seed_index: int,
    survived: bool,
    settings: BirkhoffClassifierSettings = DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS,
) -> SeedClassification:
    if not survived:
        points = np.asarray(return_points_xyz, dtype=float)
        return SeedClassification(
            seed_index=int(seed_index),
            classification=KAM_CLASS_LOST,
            return_count=0 if points.ndim == 0 else int(points.reshape((-1, 3)).shape[0]),
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="field_line_exited_before_trace_horizon",
        )

    points = np.asarray(return_points_xyz, dtype=float)
    if points.size == 0:
        points = np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"WBA return points must have shape (n, 3), got {points.shape}"
        )
    radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    angles = np.arctan2(points[:, 2] - float(axis_z), radii - float(axis_r))
    return classify_angle_series(
        angles,
        seed_index=seed_index,
        settings=settings,
    )


def poloidal_angles_from_hits(
    hits: Sequence[Sequence[float]],
    *,
    axis_r: float,
    axis_z: float,
    target_phi_index: int = DEFAULT_WBA_TARGET_PHI_INDEX,
) -> np.ndarray:
    hit_rows = np.asarray(hits, dtype=float)
    if hit_rows.size == 0:
        return np.empty((0,), dtype=float)
    if hit_rows.ndim != 2 or hit_rows.shape[1] != 5:
        raise ValueError(f"WBA hit rows must have shape (n, 5), got {hit_rows.shape}")
    plane_hits = hit_rows[hit_rows[:, 1] == int(target_phi_index)]
    radii = np.sqrt(plane_hits[:, 2] ** 2 + plane_hits[:, 3] ** 2)
    return np.arctan2(plane_hits[:, 4] - float(axis_z), radii - float(axis_r))


def classify_fieldline_hits(
    fieldlines_phi_hits: Sequence[Sequence[Sequence[float]]],
    *,
    stopped_before_hit_fn,
    axis_r: float,
    axis_z: float,
    settings: BirkhoffClassifierSettings = DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS,
) -> list[SeedClassification]:
    classifications = []
    for seed_index, line_hits in enumerate(fieldlines_phi_hits):
        hits_before_stop, first_stop = stopped_before_hit_fn(line_hits)
        angles = poloidal_angles_from_hits(
            hits_before_stop,
            axis_r=axis_r,
            axis_z=axis_z,
            target_phi_index=settings.target_phi_index,
        )
        if first_stop is not None:
            classifications.append(
                SeedClassification(
                    seed_index=int(seed_index),
                    classification=KAM_CLASS_LOST,
                    return_count=int(angles.size),
                    rotation_number=None,
                    matching_digits=None,
                    first_half_rotation_number=None,
                    second_half_rotation_number=None,
                    nearest_rational=None,
                    reason="field_line_exited_before_trace_horizon",
                )
            )
            continue
        classifications.append(
            classify_angle_series(
                angles,
                seed_index=seed_index,
                settings=settings,
            )
        )
    return classifications


def summarize_seed_classifications(
    classifications: Sequence[SeedClassification],
) -> dict[str, object]:
    total = len(classifications)
    survived = [
        item
        for item in classifications
        if item.classification != KAM_CLASS_LOST
    ]
    invariant = [
        item
        for item in survived
        if item.classification == KAM_CLASS_INVARIANT_TORUS
    ]
    classified = [
        item
        for item in survived
        if item.classification
        in {KAM_CLASS_INVARIANT_TORUS, KAM_CLASS_ISLAND_CHAIN, KAM_CLASS_CHAOTIC}
    ]
    counts = {
        KAM_CLASS_INVARIANT_TORUS: 0,
        KAM_CLASS_ISLAND_CHAIN: 0,
        KAM_CLASS_CHAOTIC: 0,
        KAM_CLASS_INSUFFICIENT_RETURNS: 0,
        KAM_CLASS_LOST: 0,
    }
    for item in classifications:
        counts[item.classification] = int(counts.get(item.classification, 0)) + 1

    eligible_count = len(survived)
    fraction = 0.0 if eligible_count == 0 else len(invariant) / float(eligible_count)
    rotation_numbers = [
        item.rotation_number
        for item in classified
        if item.rotation_number is not None and np.isfinite(item.rotation_number)
    ]
    matching_digits = [
        item.matching_digits
        for item in classified
        if item.matching_digits is not None and np.isfinite(item.matching_digits)
    ]
    return {
        "invariant_torus_fraction": float(fraction),
        "invariant_torus_count": int(len(invariant)),
        "wba_seed_count": int(total),
        "wba_survived_seed_count": int(eligible_count),
        "wba_classified_seed_count": int(len(classified)),
        "wba_classification_counts": counts,
        "wba_rotation_number_median": (
            None if not rotation_numbers else float(np.median(rotation_numbers))
        ),
        "wba_matching_digits_min": (
            None if not matching_digits else float(np.min(matching_digits))
        ),
        "wba_matching_digits_median": (
            None if not matching_digits else float(np.median(matching_digits))
        ),
        "wba_seed_classifications": [asdict(item) for item in classifications],
    }
