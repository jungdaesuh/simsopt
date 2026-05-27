import math

from examples.single_stage_optimization.banana_opt.topology.kam_birkhoff import (
    BirkhoffClassifierSettings,
    KAM_CLASS_CHAOTIC,
    KAM_CLASS_INVALID_POLAR_REFERENCE,
    KAM_CLASS_INSUFFICIENT_RETURNS,
    KAM_CLASS_INVARIANT_TORUS,
    KAM_CLASS_ISLAND_CHAIN,
    KAM_CLASS_LOST,
    SeedClassification,
    classify_angle_series,
    classify_return_points,
    summarize_seed_classifications,
)


def standard_map_angles(
    *,
    stochasticity: float,
    theta0: float,
    momentum0: float,
    count: int,
) -> list[float]:
    theta = float(theta0)
    momentum = float(momentum0)
    angles = []
    for _ in range(count):
        angles.append(theta)
        momentum += float(stochasticity) * math.sin(theta)
        theta += momentum
    return angles


def test_weighted_birkhoff_standard_map_identifies_invariant_torus_below_breakup():
    settings = BirkhoffClassifierSettings(
        min_returns=512,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    angles = standard_map_angles(
        stochasticity=0.2,
        theta0=0.1,
        momentum0=2.0 * math.pi * ((math.sqrt(5.0) - 1.0) / 2.0),
        count=2048,
    )

    result = classify_angle_series(angles, settings=settings)

    assert result.classification == KAM_CLASS_INVARIANT_TORUS
    assert result.matching_digits is not None
    assert result.matching_digits >= settings.invariant_digits_min


def test_weighted_birkhoff_standard_map_rejects_chaotic_orbit_above_breakup():
    settings = BirkhoffClassifierSettings(
        min_returns=512,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    angles = standard_map_angles(
        stochasticity=1.2,
        theta0=0.3,
        momentum0=0.2,
        count=2048,
    )

    result = classify_angle_series(angles, settings=settings)

    assert result.classification == KAM_CLASS_CHAOTIC
    assert result.matching_digits is not None
    assert result.matching_digits < settings.island_digits_min


def test_weighted_birkhoff_reports_normalized_large_rotation_branch():
    settings = BirkhoffClassifierSettings(
        min_returns=64,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    angles = [2.0 * math.pi * 0.75 * step for step in range(256)]

    result = classify_angle_series(angles, settings=settings)

    assert result.rotation_number is not None
    assert result.first_half_rotation_number is not None
    assert result.second_half_rotation_number is not None
    assert math.isclose(result.rotation_number, 0.75)
    assert math.isclose(result.first_half_rotation_number, 0.75)
    assert math.isclose(result.second_half_rotation_number, 0.75)
    assert result.nearest_rational is not None
    assert result.nearest_rational["numerator"] == 3
    assert result.nearest_rational["denominator"] == 4
    assert math.isclose(float(result.nearest_rational["value"]), 0.75)
    assert math.isclose(float(result.nearest_rational["error"]), 0.0, abs_tol=1.0e-12)


def test_weighted_birkhoff_classifies_exact_low_order_rational_as_island_chain():
    settings = BirkhoffClassifierSettings(
        min_returns=64,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    angles = [2.0 * math.pi / 3.0 * step for step in range(256)]

    result = classify_angle_series(angles, settings=settings)

    assert result.classification == KAM_CLASS_ISLAND_CHAIN
    assert result.reason == "weighted_birkhoff_average_exact_low_order_rational"
    assert result.nearest_rational is not None
    assert result.nearest_rational["numerator"] == 1
    assert result.nearest_rational["denominator"] == 3


def test_weighted_birkhoff_keeps_near_rational_torus_in_invariant_numerator():
    settings = BirkhoffClassifierSettings(
        min_returns=64,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    rotation_number = 1.0 / 3.0 + 5.0e-5
    angles = [2.0 * math.pi * rotation_number * step for step in range(256)]

    result = classify_angle_series(angles, settings=settings)

    assert result.classification == KAM_CLASS_INVARIANT_TORUS
    assert result.nearest_rational is not None
    assert result.nearest_rational["numerator"] == 1
    assert result.nearest_rational["denominator"] == 3
    assert (
        0.0
        < float(result.nearest_rational["error"])
        < (settings.island_rational_tolerance)
    )


def test_return_point_classifier_rejects_nonwinding_poloidal_reference():
    settings = BirkhoffClassifierSettings(min_returns=64)
    points = [
        [1.4 + 0.01 * math.cos(step), 0.0, 0.01 * math.sin(step)] for step in range(128)
    ]

    result = classify_return_points(
        points,
        axis_r=1.0,
        axis_z=0.0,
        seed_index=5,
        survived=True,
        settings=settings,
    )

    assert result.classification == KAM_CLASS_INVALID_POLAR_REFERENCE
    assert result.return_count == 128


def test_invariant_torus_fraction_denominator_excludes_lost_and_insufficient():
    classifications = [
        SeedClassification(
            seed_index=0,
            classification=KAM_CLASS_INVARIANT_TORUS,
            return_count=512,
            rotation_number=0.31,
            matching_digits=9.0,
            first_half_rotation_number=0.31,
            second_half_rotation_number=0.31,
            nearest_rational=None,
            reason="weighted_birkhoff_average_converged",
        ),
        SeedClassification(
            seed_index=1,
            classification=KAM_CLASS_CHAOTIC,
            return_count=512,
            rotation_number=0.19,
            matching_digits=1.0,
            first_half_rotation_number=0.18,
            second_half_rotation_number=0.20,
            nearest_rational=None,
            reason="weighted_birkhoff_average_not_converged",
        ),
        SeedClassification(
            seed_index=2,
            classification=KAM_CLASS_INSUFFICIENT_RETURNS,
            return_count=12,
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="insufficient_poincare_returns",
        ),
        SeedClassification(
            seed_index=3,
            classification=KAM_CLASS_LOST,
            return_count=4,
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="field_line_exited_before_trace_horizon",
        ),
    ]

    summary = summarize_seed_classifications(classifications)

    assert summary["wba_seed_count"] == 4
    assert summary["wba_survived_seed_count"] == 3
    assert summary["wba_classified_seed_count"] == 2
    assert summary["wba_evaluation_state"] == "evaluated"
    assert summary["invariant_torus_count"] == 1
    assert summary["invariant_torus_fraction"] == 1.0 / 2.0


def test_invariant_torus_fraction_reports_not_evaluated_for_only_insufficient_returns():
    classifications = [
        SeedClassification(
            seed_index=0,
            classification=KAM_CLASS_INSUFFICIENT_RETURNS,
            return_count=60,
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="insufficient_poincare_returns",
        ),
        SeedClassification(
            seed_index=1,
            classification=KAM_CLASS_INSUFFICIENT_RETURNS,
            return_count=60,
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="insufficient_poincare_returns",
        ),
    ]

    summary = summarize_seed_classifications(classifications)

    assert summary["wba_survived_seed_count"] == 2
    assert summary["wba_classified_seed_count"] == 0
    assert summary["invariant_torus_fraction"] is None
    assert summary["wba_evaluation_state"] == "not_evaluated_insufficient_returns"
    assert summary["wba_not_evaluated_reason"] == "not_evaluated_insufficient_returns"
