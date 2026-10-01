"""The official endpoint quality bands: the rule's properties, its three pre-registered ceilings and its post-hoc ones.

Rule v2 was pre-registered on 2026-09-20 before any sample existed; the CASE SELECTION is pre-registered only for the
three cases sampled under it that morning. BoozerQA (user decision C18) and coil forces (user decision C14) were added
post hoc, after their lane values had been seen, with the rule unchanged; they are named as such here so the two kinds
of evidence are never conflated.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import pytest
from examples.jax.parity.official_quality_bands import (
    OFFICIAL_BAND_CASE_IDS,
    upstream_scatter_ceiling,
    official_quality_band,
)
from examples.jax.parity.official_reference import (
    load_official_reference,
    load_official_sensitivity,
    official_sensitivity_case_ids,
)

# Computed independently by the sampling worker's own summary script from the raw runs
# (campaign record: investigations/band-sampling/REPORT.md) before this module existed.
PRE_REGISTERED_CEILINGS = {
    # finite build: the sampling worker's number (5.484766976546892e-07) included the branch-native value as
    # min(S); with upstream's nine values alone min(S) is upstream's k = 3 run. Independent arithmetic:
    # 5.461326240143537e-07 * (1 + (5.461326240143537e-07 - 5.439940489037902e-07) / 5.439940489037902e-07).
    "native-stage-two-optimization-finitebuild": 5.461326240143537e-07
    * (1.0 + (5.461326240143537e-07 - 5.439940489037902e-07) / 5.439940489037902e-07),
    "native-stage-two-optimization-minimal": 7.19721036206098e-07,
    "native-qfm": 3.0563869962160076e-07,
}

# Declared POST HOC with the rule unchanged; independent arithmetic over each tracked record's values.
POST_HOC_CEILINGS = {
    # BoozerQA: user decision C18 (2026-09-20 23:48 EDT); upstream's nine end objectives span k = 2 (min) to k = 5 (max).
    "native-boozerqa": 4.658128966161633e-08
    * (1.0 + (4.658128966161633e-08 - 4.1605197668107733e-08) / 4.1605197668107733e-08),
    # Coil forces: user decision C14 (2026-09-20 23:48 EDT, implemented 2026-09-21); upstream's nine end objectives
    # (capture key stage2:final:objective, lane key final:objective) span k = 3 (min) to k = 5 (max, the run whose
    # stage one stops with SciPy status 2). Independent arithmetic over the tracked record's values:
    "native-coil-forces": 2.9115050180755156e-05
    * (1.0 + (2.9115050180755156e-05 - 2.891377263322631e-05) / 2.891377263322631e-05),
}

OFFICIAL_CEILINGS = {**PRE_REGISTERED_CEILINGS, **POST_HOC_CEILINGS}


def test_every_band_case_has_a_tracked_sensitivity_record_and_nothing_else_does() -> (
    None
):
    assert (
        set(OFFICIAL_BAND_CASE_IDS)
        == set(official_sensitivity_case_ids())
        == set(OFFICIAL_CEILINGS)
    )


def test_pre_registered_and_post_hoc_case_selections_are_named_and_disjoint() -> None:
    assert set(PRE_REGISTERED_CEILINGS) == {
        "native-qfm",
        "native-stage-two-optimization-finitebuild",
        "native-stage-two-optimization-minimal",
    }
    assert not set(PRE_REGISTERED_CEILINGS) & set(POST_HOC_CEILINGS)


@pytest.mark.parametrize("case_id", sorted(OFFICIAL_CEILINGS))
def test_ceiling_matches_the_independently_computed_value(case_id: str) -> None:
    band = official_quality_band(case_id)
    assert band.max_value == pytest.approx(
        OFFICIAL_CEILINGS[case_id], rel=1.0e-15, abs=0.0
    )
    assert band.observable == load_official_sensitivity(case_id).lane_observable


@pytest.mark.parametrize("case_id", sorted(OFFICIAL_CEILINGS))
def test_unperturbed_sample_is_the_canonical_official_value(case_id: str) -> None:
    sensitivity = load_official_sensitivity(case_id)
    canonical = load_official_reference(case_id).scalar(sensitivity.observable)
    assert sensitivity.run(0).end_value == canonical


def test_ceiling_never_drops_below_the_largest_native_side_sample() -> None:
    samples = (3.0, 2.0, 2.5)
    assert upstream_scatter_ceiling(samples) == 3.0 * (1.0 + (3.0 - 2.0) / 2.0)
    assert upstream_scatter_ceiling(samples) >= max(samples)


def test_ceiling_ignores_order_and_interior_samples() -> None:
    assert upstream_scatter_ceiling((2.0, 3.0)) == upstream_scatter_ceiling(
        (3.0, 2.5, 2.0, 2.75)
    )


def test_identical_samples_give_a_band_without_headroom() -> None:
    assert upstream_scatter_ceiling((4.0, 4.0)) == 4.0


@pytest.mark.parametrize(
    "samples",
    [(1.0,), (1.0, float("nan")), (1.0, 0.0), (1.0, -2.0), (1.0, float("inf"))],
)
def test_rule_rejects_samples_it_cannot_bound(samples: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        upstream_scatter_ceiling(samples)
