from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from collections.abc import Mapping, Sequence

import numpy as np


DEFAULT_WBA_TARGET_PHI_INDEX = 0
DEFAULT_WBA_MIN_RETURNS = 256
DEFAULT_WBA_INVARIANT_MIN_DIGITS = 8.0
DEFAULT_WBA_ISLAND_MIN_DIGITS = 4.0
DEFAULT_WBA_RATIONAL_MAX_DENOMINATOR = 24
DEFAULT_WBA_DIFF_FLOOR = 1.0e-15
# Known limitation: this single-surface rationality test does not include the
# radial rotation-number plateau discriminator needed to certify every island.
DEFAULT_WBA_EXACT_RATIONAL_TOLERANCE = 1.0e-8
DEFAULT_WBA_MIN_WINDING_SIGN_FRACTION = 0.95
DEFAULT_WBA_WINDING_INCREMENT_FLOOR = 1.0e-12
DEFAULT_WBA_ALIAS_WINDING_CONCENTRATION_MAX = 0.5
DEFAULT_WBA_ALIAS_ADVANCE_CONCENTRATION_MIN = 0.75
# Upper bound below which a section-angle series is treated as principal-valued
# (wrapped to [-pi, pi]) so the modulo-one aliasing recovery branch applies.
_WBA_PRINCIPAL_ANGLE_BOUND = np.pi + 1.0e-12
WBA_ISLAND_DISCRIMINATOR = "single_surface_exact_low_order_rational"
WBA_CLASSIFIER_KNOWN_LIMITATIONS = ("no_radial_rotation_number_plateau_discriminator",)
WBA_FRACTION_DENOMINATOR_POLICY = "survived_non_lost_seeds"
KAM_FRACTION_SEMANTICS = "weighted_birkhoff_invariant_torus_fraction"

KAM_CLASS_INVARIANT_TORUS = "invariant_torus"
KAM_CLASS_ISLAND_CHAIN = "island_chain"
KAM_CLASS_CHAOTIC = "chaotic"
KAM_CLASS_INSUFFICIENT_RETURNS = "insufficient_returns"
KAM_CLASS_INVALID_POLAR_REFERENCE = "invalid_poloidal_reference"
KAM_CLASS_MISSING_MAGNETIC_AXIS = "missing_magnetic_axis"
KAM_CLASS_LOST = "lost"

KAM_CLASSIFIABLE_STATES = frozenset(
    (
        KAM_CLASS_INVARIANT_TORUS,
        KAM_CLASS_ISLAND_CHAIN,
        KAM_CLASS_CHAOTIC,
    )
)

WBA_EVALUATION_EVALUATED = "evaluated"
WBA_EVALUATION_NOT_EVALUATED_NO_SEEDS = "not_evaluated_no_seeds"
WBA_EVALUATION_NOT_EVALUATED_NO_SURVIVED_SEEDS = "not_evaluated_no_survived_seeds"
WBA_EVALUATION_NOT_EVALUATED_INSUFFICIENT_RETURNS = "not_evaluated_insufficient_returns"
WBA_EVALUATION_NOT_EVALUATED_INVALID_POLAR_REFERENCE = (
    "not_evaluated_invalid_poloidal_reference"
)
WBA_EVALUATION_NOT_EVALUATED_MISSING_MAGNETIC_AXIS = (
    "not_evaluated_missing_magnetic_axis"
)
WBA_EVALUATION_NOT_EVALUATED_AXIS_NOT_LOCATED = "not_evaluated_axis_not_located"
WBA_EVALUATION_NOT_EVALUATED_NO_CLASSIFIED_SEEDS = "not_evaluated_no_classified_seeds"
WBA_EVALUATION_NOT_EVALUATED_SKIPPED_BY_CALLER = "not_evaluated_skipped_by_caller"


@dataclass(frozen=True, slots=True)
class BirkhoffClassifierSettings:
    """Controls the WBA convergence-rate invariant-torus classifier."""

    target_phi_index: int = DEFAULT_WBA_TARGET_PHI_INDEX
    min_returns: int = DEFAULT_WBA_MIN_RETURNS
    invariant_digits_min: float = DEFAULT_WBA_INVARIANT_MIN_DIGITS
    island_digits_min: float = DEFAULT_WBA_ISLAND_MIN_DIGITS
    exact_rational_tolerance: float = DEFAULT_WBA_EXACT_RATIONAL_TOLERANCE
    island_max_denominator: int = DEFAULT_WBA_RATIONAL_MAX_DENOMINATOR
    diff_floor: float = DEFAULT_WBA_DIFF_FLOOR
    min_winding_sign_fraction: float = DEFAULT_WBA_MIN_WINDING_SIGN_FRACTION
    winding_increment_floor: float = DEFAULT_WBA_WINDING_INCREMENT_FLOOR


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
    return {
        **asdict(settings),
        "island_discriminator": WBA_ISLAND_DISCRIMINATOR,
        "known_limitations": list(WBA_CLASSIFIER_KNOWN_LIMITATIONS),
    }


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


def normalized_wrapped_rotation_increments(angles_rad: Sequence[float]) -> np.ndarray:
    """Return section advances, using modulo-one increments for principal angles."""

    angles = np.asarray(angles_rad, dtype=float)
    if angles.ndim != 1:
        raise ValueError("WBA classification requires a one-dimensional angle series")
    if not np.all(np.isfinite(angles)):
        raise ValueError("WBA classification received NaN/Inf angles")
    if angles.size < 2:
        return np.empty((0,), dtype=float)
    raw_increments = np.diff(angles) / (2.0 * np.pi)
    if np.all(np.abs(angles) <= _WBA_PRINCIPAL_ANGLE_BOUND):
        modulo_advances = np.mod(raw_increments, 1.0)
        mean_phase = np.mean(np.exp(2j * np.pi * modulo_advances))
        reference_advance = float(np.angle(mean_phase) / (2.0 * np.pi)) % 1.0
        candidates = np.stack(
            (raw_increments - 1.0, raw_increments, raw_increments + 1.0),
            axis=0,
        )
        candidate_indices = np.argmin(np.abs(candidates - reference_advance), axis=0)
        return np.take_along_axis(
            candidates,
            candidate_indices.reshape(1, -1),
            axis=0,
        ).reshape(-1)
    return raw_increments


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
    angles = np.asarray(angles_rad, dtype=float)
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

    increments = normalized_wrapped_rotation_increments(angles)
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

    if matching_digits >= float(settings.island_digits_min) and rational.error <= float(
        settings.exact_rational_tolerance
    ):
        classification = KAM_CLASS_ISLAND_CHAIN
        reason = "weighted_birkhoff_average_exact_low_order_rational"
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


def poloidal_angle_series_has_consistent_winding(
    angles_rad: Sequence[float],
    *,
    settings: BirkhoffClassifierSettings = DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS,
) -> bool:
    raw_angles = np.asarray(angles_rad, dtype=float)
    angles = np.unwrap(raw_angles)
    if angles.ndim != 1:
        raise ValueError("WBA winding check requires a one-dimensional angle series")
    if not np.all(np.isfinite(angles)):
        raise ValueError("WBA winding check received NaN/Inf angles")
    if angles.size < int(settings.min_returns):
        return True
    increments = np.diff(angles)
    significant = increments[
        np.abs(increments) > float(settings.winding_increment_floor)
    ]
    if significant.size == 0:
        return False
    net_increment = float(np.sum(significant))
    if net_increment == 0.0:
        return False
    direction = 1.0 if net_increment > 0.0 else -1.0
    same_direction_fraction = float(np.mean(significant * direction > 0.0))
    if same_direction_fraction >= float(settings.min_winding_sign_fraction):
        return True

    if not np.all(np.abs(raw_angles) <= _WBA_PRINCIPAL_ANGLE_BOUND):
        return False
    modulo_advances = normalized_wrapped_rotation_increments(raw_angles)
    if modulo_advances.size == 0:
        return False
    advance_concentration = float(abs(np.mean(np.exp(2j * np.pi * modulo_advances))))
    if advance_concentration < DEFAULT_WBA_ALIAS_ADVANCE_CONCENTRATION_MIN:
        return False
    phasor_concentration = float(abs(np.mean(np.exp(1j * raw_angles))))
    return phasor_concentration <= DEFAULT_WBA_ALIAS_WINDING_CONCENTRATION_MAX


def invalid_poloidal_reference_classification(
    *,
    seed_index: int,
    return_count: int,
) -> SeedClassification:
    return SeedClassification(
        seed_index=int(seed_index),
        classification=KAM_CLASS_INVALID_POLAR_REFERENCE,
        return_count=int(return_count),
        rotation_number=None,
        matching_digits=None,
        first_half_rotation_number=None,
        second_half_rotation_number=None,
        nearest_rational=None,
        reason="poloidal_reference_does_not_wind_monotonically",
    )


def missing_magnetic_axis_classification(
    *,
    seed_index: int,
    return_count: int,
) -> SeedClassification:
    return SeedClassification(
        seed_index=int(seed_index),
        classification=KAM_CLASS_MISSING_MAGNETIC_AXIS,
        return_count=int(return_count),
        rotation_number=None,
        matching_digits=None,
        first_half_rotation_number=None,
        second_half_rotation_number=None,
        nearest_rational=None,
        reason="magnetic_axis_reference_not_available",
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
            return_count=0
            if points.ndim == 0
            else int(points.reshape((-1, 3)).shape[0]),
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
    if not poloidal_angle_series_has_consistent_winding(angles, settings=settings):
        return invalid_poloidal_reference_classification(
            seed_index=seed_index,
            return_count=int(angles.size),
        )
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
        if not poloidal_angle_series_has_consistent_winding(
            angles,
            settings=settings,
        ):
            classifications.append(
                invalid_poloidal_reference_classification(
                    seed_index=seed_index,
                    return_count=int(angles.size),
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
        item for item in classifications if item.classification != KAM_CLASS_LOST
    ]
    invariant = [
        item for item in survived if item.classification == KAM_CLASS_INVARIANT_TORUS
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
        KAM_CLASS_INVALID_POLAR_REFERENCE: 0,
        KAM_CLASS_MISSING_MAGNETIC_AXIS: 0,
        KAM_CLASS_LOST: 0,
    }
    for item in classifications:
        counts[item.classification] = int(counts.get(item.classification, 0)) + 1

    survived_count = len(survived)
    classified_count = len(classified)
    fraction = None if classified_count == 0 else len(invariant) / float(survived_count)
    if classified_count > 0:
        evaluation_state = WBA_EVALUATION_EVALUATED
        not_evaluated_reason = None
    elif total == 0:
        evaluation_state = WBA_EVALUATION_NOT_EVALUATED_NO_SEEDS
        not_evaluated_reason = evaluation_state
    elif survived_count == 0:
        evaluation_state = WBA_EVALUATION_NOT_EVALUATED_NO_SURVIVED_SEEDS
        not_evaluated_reason = evaluation_state
    elif counts[KAM_CLASS_INSUFFICIENT_RETURNS] == survived_count:
        evaluation_state = WBA_EVALUATION_NOT_EVALUATED_INSUFFICIENT_RETURNS
        not_evaluated_reason = evaluation_state
    elif counts[KAM_CLASS_INVALID_POLAR_REFERENCE] > 0:
        evaluation_state = WBA_EVALUATION_NOT_EVALUATED_INVALID_POLAR_REFERENCE
        not_evaluated_reason = evaluation_state
    elif counts[KAM_CLASS_MISSING_MAGNETIC_AXIS] == survived_count:
        evaluation_state = WBA_EVALUATION_NOT_EVALUATED_MISSING_MAGNETIC_AXIS
        not_evaluated_reason = evaluation_state
    else:
        evaluation_state = WBA_EVALUATION_NOT_EVALUATED_NO_CLASSIFIED_SEEDS
        not_evaluated_reason = evaluation_state
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
        "invariant_torus_fraction": (None if fraction is None else float(fraction)),
        "invariant_torus_count": int(len(invariant)),
        "wba_fraction_denominator_policy": WBA_FRACTION_DENOMINATOR_POLICY,
        "wba_fraction_denominator_seed_count": int(survived_count),
        "wba_seed_count": int(total),
        "wba_survived_seed_count": int(survived_count),
        "wba_classified_seed_count": int(classified_count),
        "wba_evaluation_state": evaluation_state,
        "wba_not_evaluated_reason": not_evaluated_reason,
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


def _seed_entry_classification(entry: object) -> str:
    """Return the classification label of one WBA seed payload entry.

    Accepts either a mapping (the ``asdict`` payload stored in topology
    results) or a :class:`SeedClassification`. Raises loudly when the field is
    absent so a misconfigured certification gate fails closed instead of
    silently treating an unlabelled line as benign.
    """

    if isinstance(entry, SeedClassification):
        return str(entry.classification)
    if isinstance(entry, Mapping):
        if "classification" not in entry:
            raise ValueError(
                "WBA seed classification entry is missing the 'classification' "
                "field; cannot evaluate the island/classifiability verdict"
            )
        return str(entry["classification"])
    raise TypeError(
        "WBA seed classification entry must be a SeedClassification or mapping, "
        f"got {type(entry).__name__}"
    )


def _seed_entry_return_count(entry: object) -> int:
    if isinstance(entry, SeedClassification):
        return int(entry.return_count)
    if isinstance(entry, Mapping):
        if "return_count" not in entry or entry["return_count"] is None:
            raise ValueError(
                "WBA seed classification entry is missing the 'return_count' "
                "field; cannot evaluate the classifiability floor"
            )
        return int(entry["return_count"])
    raise TypeError(
        "WBA seed classification entry must be a SeedClassification or mapping, "
        f"got {type(entry).__name__}"
    )


def island_verdict_counts(
    seed_classifications: Sequence[object],
    *,
    min_returns: int = DEFAULT_WBA_MIN_RETURNS,
) -> dict[str, object]:
    """Tally per-line island + classifiability statistics for a strict verdict.

    Single pass over the per-line WBA payload (``wba_seed_classifications``).
    A *surviving* line is any line not classified ``lost``; a surviving line is
    *classifiable* only when it accumulated at least ``min_returns`` Poincare
    returns (below that floor the WBA rotation number is unreliable, so the line
    is treated as not-classifiable). Island lines are counted by the absence
    test the certification verdict keys on -- the explicit ``island_chain``
    label -- never by the presence of ``invariant_torus``.
    """

    if int(min_returns) <= 0:
        raise ValueError("min_returns must be a positive integer")

    total = 0
    surviving = 0
    classifiable = 0
    island_chain = 0
    island_chain_seed_indices: list[int] = []
    for position, entry in enumerate(seed_classifications):
        total += 1
        classification = _seed_entry_classification(entry)
        if classification == KAM_CLASS_ISLAND_CHAIN:
            island_chain += 1
            if isinstance(entry, SeedClassification):
                island_chain_seed_indices.append(int(entry.seed_index))
            elif isinstance(entry, Mapping) and entry.get("seed_index") is not None:
                island_chain_seed_indices.append(int(entry["seed_index"]))
            else:
                island_chain_seed_indices.append(int(position))
        if classification != KAM_CLASS_LOST:
            surviving += 1
            if (
                classification in KAM_CLASSIFIABLE_STATES
                and _seed_entry_return_count(entry) >= int(min_returns)
            ):
                classifiable += 1

    classifiable_fraction = (
        None if surviving == 0 else float(classifiable) / float(surviving)
    )
    return {
        "seed_count": int(total),
        "surviving_seed_count": int(surviving),
        "classifiable_seed_count": int(classifiable),
        "classifiable_fraction": classifiable_fraction,
        "island_chain_count": int(island_chain),
        "island_chain_seed_indices": island_chain_seed_indices,
        "min_returns": int(min_returns),
    }
