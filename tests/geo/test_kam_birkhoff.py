import math

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology.kam_birkhoff import (
    BirkhoffClassifierSettings,
    KAM_CLASS_CHAOTIC,
    KAM_CLASS_INVALID_POLAR_REFERENCE,
    KAM_CLASS_INSUFFICIENT_RETURNS,
    KAM_CLASS_INVARIANT_TORUS,
    KAM_CLASS_ISLAND_CHAIN,
    KAM_CLASS_LOST,
    KAM_CLASS_MISSING_MAGNETIC_AXIS,
    WBA_FRACTION_DENOMINATOR_POLICY,
    SeedClassification,
    classify_angle_series,
    classifier_settings_payload,
    classify_return_points,
    normalized_wrapped_rotation_increments,
    poloidal_angle_series_has_consistent_winding,
    summarize_seed_classifications,
    weighted_birkhoff_average,
)


def _weighted_birkhoff_errors_over_doubling_ladder(series, *, levels, reference):
    # |WBA over the leading N samples - reference| at a geometric ladder of
    # window lengths N (ascending). The reference is the WBA over an independent,
    # much longer run so the finest ladder point is not its own reference.
    return np.array(
        [abs(weighted_birkhoff_average(series[:n]) - reference) for n in levels],
        dtype=float,
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


def test_weighted_birkhoff_settings_payload_discloses_island_limitations():
    payload = classifier_settings_payload()

    assert payload["exact_rational_tolerance"] == 1.0e-8
    assert payload["island_discriminator"] == "single_surface_exact_low_order_rational"
    assert payload["known_limitations"] == [
        "no_radial_rotation_number_plateau_discriminator"
    ]


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


def test_weighted_birkhoff_standard_map_sensitive_orbit_flips_across_breakup_region():
    settings = BirkhoffClassifierSettings(
        min_returns=2048,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    seeded_momentum = 2.0 * math.pi * ((math.sqrt(5.0) - 1.0) / 2.0) + 0.17
    below = classify_angle_series(
        standard_map_angles(
            stochasticity=0.95,
            theta0=0.1,
            momentum0=seeded_momentum,
            count=8192,
        ),
        settings=settings,
    )
    above = classify_angle_series(
        standard_map_angles(
            stochasticity=0.99,
            theta0=0.1,
            momentum0=seeded_momentum,
            count=8192,
        ),
        settings=settings,
    )

    assert below.classification == KAM_CLASS_INVARIANT_TORUS
    assert above.classification == KAM_CLASS_CHAOTIC


def test_weighted_birkhoff_super_algebraic_beats_algebraic_convergence_rate():
    # Characterizes the convergence-rate distinction that motivates the WBA
    # classifier: on a smooth observable sampled along an irrational (golden)
    # rotation the weighted Birkhoff average converges super-algebraically
    # (faster than any power of the window length), while an algebraically
    # convergent (~1/N) observable gains only a fixed number of digits per
    # window doubling. Measured as digits gained per doubling in the regime
    # before the super-algebraic series saturates at machine precision.
    #
    # NOTE: this is a property of the WBA primitive, not of classify_angle_series
    # -- the classifier keys on absolute two-half agreement, because the measured
    # rate is degenerate (undefined) on the cleanest tori, whose rotation number
    # is constant so the average is already exact at every window length.
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    reference_count = 8192
    # Small ascending window sizes: super-algebraic convergence reaches machine
    # precision quickly, so the rate is only observable before it saturates.
    levels = [16, 32, 64, 128, 256]

    theta = 2.0 * math.pi * golden_ratio * np.arange(reference_count)
    super_algebraic = np.cos(theta)
    algebraic = 1.0 + 1.0 / (np.arange(reference_count) + 1.0)

    super_errors = _weighted_birkhoff_errors_over_doubling_ladder(
        super_algebraic,
        levels=levels,
        reference=weighted_birkhoff_average(super_algebraic),
    )
    algebraic_errors = _weighted_birkhoff_errors_over_doubling_ladder(
        algebraic,
        levels=levels,
        reference=weighted_birkhoff_average(algebraic),
    )

    super_floored = np.maximum(super_errors, 1e-15)
    algebraic_floored = np.maximum(algebraic_errors, 1e-15)
    super_digits_per_doubling = -np.diff(np.log10(super_floored))
    algebraic_digits_per_doubling = -np.diff(np.log10(algebraic_floored))

    # Algebraic ~1/N gains about log10(2) ~ 0.3 digits per doubling and never
    # accelerates. The super-algebraic series gains several digits per doubling
    # and the rate grows with N (faster than any power) until it saturates.
    assert float(np.max(algebraic_digits_per_doubling)) < 0.6
    assert float(np.max(super_digits_per_doubling)) > 2.0
    assert float(np.max(super_digits_per_doubling)) > 4.0 * float(
        np.max(algebraic_digits_per_doubling)
    )


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


def test_weighted_birkhoff_unaliases_wrapped_large_rotation_branch():
    settings = BirkhoffClassifierSettings(
        min_returns=64,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    angles = [
        math.atan2(
            math.sin(2.0 * math.pi * 0.75 * step),
            math.cos(2.0 * math.pi * 0.75 * step),
        )
        for step in range(256)
    ]

    result = classify_angle_series(angles, settings=settings)

    assert result.rotation_number is not None
    assert math.isclose(result.rotation_number, 0.75)


def test_weighted_birkhoff_unaliases_wrapped_modulated_rotation_branch():
    settings = BirkhoffClassifierSettings(
        min_returns=64,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    increments = [
        0.42 * (1.0 + 0.3 * math.sin(2.0 * math.pi * step / 64.0))
        for step in range(512)
    ]
    cumulative = [0.0]
    for increment in increments:
        cumulative.append(cumulative[-1] + increment)
    angles = [
        math.atan2(math.sin(2.0 * math.pi * value), math.cos(2.0 * math.pi * value))
        for value in cumulative
    ]

    result = classify_angle_series(angles, settings=settings)

    assert result.rotation_number is not None
    assert math.isclose(result.rotation_number, 0.42, rel_tol=0.0, abs_tol=1.0e-6)


def test_weighted_birkhoff_keeps_small_reverse_principal_angle_jitter():
    increments = [0.30] * 256
    increments[64] = -0.01
    cumulative = [0.0]
    for increment in increments:
        cumulative.append(cumulative[-1] + increment)
    angles = [
        math.atan2(math.sin(2.0 * math.pi * value), math.cos(2.0 * math.pi * value))
        for value in cumulative
    ]

    recovered = normalized_wrapped_rotation_increments(angles)

    assert recovered[64] == pytest.approx(-0.01)
    assert np.max(recovered) < 0.31


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


def test_weighted_birkhoff_standard_map_librating_island_core_is_island_chain():
    settings = BirkhoffClassifierSettings(
        min_returns=512,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    angles = standard_map_angles(
        stochasticity=0.5,
        theta0=math.pi + 0.2,
        momentum0=0.01,
        count=8192,
    )

    result = classify_angle_series(angles, settings=settings)

    assert max(angles) - min(angles) > 0.4
    assert result.classification == KAM_CLASS_ISLAND_CHAIN
    assert result.nearest_rational is not None
    assert result.nearest_rational["numerator"] == 0
    assert result.nearest_rational["denominator"] == 1


def test_weighted_birkhoff_classifies_numerically_noisy_low_order_rational_as_island_chain():
    settings = BirkhoffClassifierSettings(
        min_returns=64,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
    )
    rotation_number = 1.0 / 3.0 + 5.0e-10
    angles = [2.0 * math.pi * rotation_number * step for step in range(256)]

    result = classify_angle_series(angles, settings=settings)

    assert result.classification == KAM_CLASS_ISLAND_CHAIN
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
    rational_error = float(result.nearest_rational["error"])
    assert 0.0 < rational_error < 1.0e-4
    assert rational_error > settings.exact_rational_tolerance


def test_weighted_birkhoff_wires_explicit_exact_rational_tolerance():
    settings = BirkhoffClassifierSettings(
        min_returns=64,
        invariant_digits_min=8.0,
        island_digits_min=4.0,
        exact_rational_tolerance=1.0e-3,
    )
    rotation_number = 1.0 / 3.0 + 5.0e-5
    angles = [2.0 * math.pi * rotation_number * step for step in range(256)]

    result = classify_angle_series(angles, settings=settings)

    assert result.classification == KAM_CLASS_ISLAND_CHAIN
    assert result.nearest_rational is not None
    assert result.nearest_rational["numerator"] == 1
    assert result.nearest_rational["denominator"] == 3


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


def test_winding_check_rejects_spread_nonmonotone_principal_angles():
    settings = BirkhoffClassifierSettings(min_returns=64)
    rng = np.random.default_rng(20260528)
    angles = rng.uniform(-math.pi, math.pi, size=512)

    assert not poloidal_angle_series_has_consistent_winding(
        angles,
        settings=settings,
    )


def test_winding_check_accepts_principal_winding_above_half_turn():
    # A genuine torus with rotation number 0.51 turn/return (just above the
    # 1/2-turn boundary) and a small physical libration of +-0.02 turn, given as
    # principal-valued (arctan2) section angles. Every per-return advance is
    # strictly forward, so this is a consistently-winding torus. np.unwrap
    # aliases the >1/2-turn returns into apparent backward steps, dropping the
    # sign-direction fraction to ~0.68; the principal-branch rewrap holds it at
    # 1.0 so the primary winding branch accepts directly.
    settings = BirkhoffClassifierSettings(min_returns=64)
    steps = np.arange(512)
    increments = 0.51 + 0.02 * np.sin(2.0 * math.pi * steps / 53.0)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    angles = np.arctan2(
        np.sin(2.0 * math.pi * cumulative),
        np.cos(2.0 * math.pi * cumulative),
    )

    assert bool(np.all(increments > 0.0))
    assert float(np.mean(increments)) > 0.5
    unwrap_increments = np.diff(np.unwrap(angles))
    net = float(np.sum(unwrap_increments))
    unwrap_direction = 1.0 if net > 0.0 else -1.0
    unwrap_sign_fraction = float(
        np.mean(unwrap_increments * unwrap_direction > 0.0)
    )
    assert unwrap_sign_fraction < settings.min_winding_sign_fraction
    assert poloidal_angle_series_has_consistent_winding(angles, settings=settings)


def test_winding_check_accepts_nonprincipal_winding_above_half_turn():
    # The same >1/2-turn winding handed in as raw cumulative angles (|angle|
    # grows past pi), the regime where the old code could not fall back to the
    # principal alias-recovery branch (its principal-value guard fails). The old
    # np.unwrap primary branch aliased every >1/2-turn return and rejected this
    # genuine winding outright; the principal-branch rewrap reconstructs the
    # forward advances and accepts.
    settings = BirkhoffClassifierSettings(min_returns=64)
    steps = np.arange(512)
    advance_radians = (0.51 + 0.02 * np.sin(2.0 * math.pi * steps / 53.0)) * (
        2.0 * math.pi
    )
    raw_angles = np.concatenate([[0.0], np.cumsum(advance_radians)])

    assert float(np.max(np.abs(raw_angles))) > math.pi
    assert poloidal_angle_series_has_consistent_winding(raw_angles, settings=settings)


def test_invariant_torus_fraction_denominator_excludes_lost_only():
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

    # Denominator is the classifiable seeds (invariant + chaotic = 2), not the
    # survivors (3, which would include the insufficient-returns line). Counting
    # the not-evaluated survivor in the denominator deflates the fraction from
    # 1/2 to 1/3, the deflation this test guards against.
    assert summary["wba_seed_count"] == 4
    assert summary["wba_fraction_denominator_policy"] == WBA_FRACTION_DENOMINATOR_POLICY
    assert summary["wba_fraction_denominator_seed_count"] == 2
    assert summary["wba_survived_seed_count"] == 3
    assert summary["wba_classified_seed_count"] == 2
    assert summary["wba_not_evaluated_seed_count"] == 1
    assert summary["wba_evaluation_state"] == "evaluated"
    assert summary["invariant_torus_count"] == 1
    assert summary["invariant_torus_fraction"] == 0.5


def test_invariant_torus_fraction_fails_closed_on_single_classifiable_seed():
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
            seed_index=2,
            classification=KAM_CLASS_INVALID_POLAR_REFERENCE,
            return_count=512,
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="invalid_poloidal_reference",
        ),
        SeedClassification(
            seed_index=3,
            classification=KAM_CLASS_MISSING_MAGNETIC_AXIS,
            return_count=512,
            rotation_number=None,
            matching_digits=None,
            first_half_rotation_number=None,
            second_half_rotation_number=None,
            nearest_rational=None,
            reason="magnetic_axis_not_available",
        ),
        SeedClassification(
            seed_index=4,
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

    # Only one seed is classifiable (the invariant torus); the other three
    # survivors were not evaluated. The old policy counted those survivors in
    # the denominator and reported a finite fraction (1/4 = 0.25). The corrected
    # policy excludes them, leaving a single classifiable seed -- below the
    # WBA_MIN_CLASSIFIABLE_SEEDS floor -- so the fraction fails closed (None)
    # rather than certifying a champion on one seed.
    assert summary["wba_fraction_denominator_policy"] == WBA_FRACTION_DENOMINATOR_POLICY
    assert summary["wba_fraction_denominator_seed_count"] == 1
    assert summary["wba_survived_seed_count"] == 4
    assert summary["wba_classified_seed_count"] == 1
    assert summary["wba_not_evaluated_seed_count"] == 3
    assert summary["invariant_torus_fraction"] is None
    assert (
        summary["wba_evaluation_state"]
        == "not_evaluated_too_few_classifiable_seeds"
    )
    assert (
        summary["wba_not_evaluated_reason"]
        == "not_evaluated_too_few_classifiable_seeds"
    )


def test_invariant_torus_fraction_evaluates_two_classifiable_seeds():
    # The boundary case: exactly WBA_MIN_CLASSIFIABLE_SEEDS classifiable seeds
    # (one invariant, one chaotic) evaluates rather than failing closed, and the
    # fraction is 1/2 over the classifiable denominator.
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
    ]

    summary = summarize_seed_classifications(classifications)

    assert summary["wba_classified_seed_count"] == 2
    assert summary["wba_not_evaluated_seed_count"] == 0
    assert summary["wba_evaluation_state"] == "evaluated"
    assert summary["invariant_torus_fraction"] == 0.5


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
