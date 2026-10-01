"""The tracked TRACING scatter records are complete and self-consistent.

Like the rest of the official-reference fixture, these tests import neither simsopt nor jax: the record must be
readable on a clean checkout with nothing built.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json

import numpy as np
import pytest

from examples.jax.parity.official_reference import (
    TRACING_ROOT,
    TRACING_SCATTER_CASE_QUANTITIES,
    TRACING_SCATTER_QUANTITIES,
    UPSTREAM_COMMIT,
    MissingObservableError,
    load_official_reference,
    load_official_tracing_scatter,
    official_case_ids,
    official_tracing_scatter_case_ids,
)

TRACING_CASES = (
    "native-tracing-fieldlines-ncsx",
    "native-tracing-fieldlines-qa",
    "native-tracing-particle",
)
TRACING_KS = tuple(range(9))

#: The quantity a case records only when the object it traces has it, and the case that has it. Literal on purpose:
#: a field line has no parallel speed, so its record must NOT carry the key, and this file says which is which.
CASE_SPECIFIC_QUANTITIES = {
    "native-tracing-fieldlines-ncsx": (),
    "native-tracing-fieldlines-qa": (),
    "native-tracing-particle": ("final_parallel_speed_fraction_difference_max",),
}
#: Keys the coordinator ruled must NOT appear: the record carries upstream's numbers, never a derived bound.
FORBIDDEN_TRACING_KEYS = frozenset(
    {"ceiling", "band", "spread", "tolerance", "bound", "factor", "headroom"}
)
#: The protocol leaves run k=0 unperturbed: it is the canonical official run.
TRACING_UNPERTURBED_K = 0


def _json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _json_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _json_keys(item)}
    return set()


def test_tracing_records_cover_exactly_the_three_tracing_cases() -> None:
    assert official_tracing_scatter_case_ids() == TRACING_CASES
    assert set(TRACING_CASES) <= set(official_case_ids())
    assert TRACING_ROOT.is_dir()
    assert sorted(path.name for path in TRACING_ROOT.iterdir()) == [
        f"{case_id}.json" for case_id in TRACING_CASES
    ]


def test_every_tracing_record_round_trips_with_nine_runs_k0_to_k8() -> None:
    for case_id in official_tracing_scatter_case_ids():
        scatter = load_official_tracing_scatter(case_id)
        canonical = load_official_reference(case_id)
        assert scatter.case_id == case_id
        assert scatter.upstream_commit == UPSTREAM_COMMIT
        assert scatter.official_script == canonical.official_script
        assert scatter.official_script_sha256 == canonical.official_script_sha256
        assert scatter.lines == int(canonical.array("final:times").size)
        assert tuple(run.k for run in scatter.runs) == TRACING_KS
        assert len(scatter.perturbed_runs) == len(TRACING_KS) - 1
        assert scatter.protocol.unperturbed_k == TRACING_UNPERTURBED_K
        assert "nextafter" in scatter.protocol.perturbation_rule
        assert "20260920 + k" in scatter.protocol.seed_rule
        assert scatter.protocol.threads["OMP_NUM_THREADS"] == "1"
        assert scatter.protocol.perturbed_arguments
        assert scatter.protocol.pre_registered_in.startswith(
            "examples/jax/parity/official_reference/9e027eac3/README.md"
        )
        unperturbed = scatter.run(TRACING_UNPERTURBED_K)
        assert unperturbed.scatter is None
        assert unperturbed.perturbation.perturbed_arguments is None
        expected_quantities = (
            TRACING_SCATTER_QUANTITIES + CASE_SPECIFIC_QUANTITIES[case_id]
        )
        assert set(CASE_SPECIFIC_QUANTITIES[case_id]) <= set(
            TRACING_SCATTER_CASE_QUANTITIES
        )
        assert tuple(scatter.maxima.as_mapping()) == expected_quantities, case_id
        for run in scatter.perturbed_runs:
            assert run.scatter is not None
            assert tuple(run.scatter.as_mapping()) == expected_quantities
            assert run.perturbation.entries == run.perturbation.entries_changed
            assert run.perturbation.seed == 20260920 + run.k
            assert run.perturbation.perturbed_arguments == (
                scatter.protocol.perturbed_arguments
            )
            assert 0.0 < float(run.perturbation.max_rel_delta or 0.0) < 1.0e-15
        for run in scatter.runs:
            assert len(run.capture_sha256) == 64
            assert len(run.perturbation_sha256) == 64
        assert len({run.capture_sha256 for run in scatter.runs}) == len(scatter.runs)


def test_tracing_k0_reproduces_the_canonical_fixture_values_bitwise() -> None:
    for case_id in official_tracing_scatter_case_ids():
        scatter = load_official_tracing_scatter(case_id)
        canonical = load_official_reference(case_id)
        inline = {
            key
            for key in canonical.keys()
            if canonical.kind(key) == "array" and canonical.is_inline(key)
        }
        recorded = scatter.unperturbed.observable_sha256
        assert set(recorded) == inline, case_id
        assert inline >= {"final:times", "final:status", "poincare:counts"}
        for key, digest in recorded.items():
            assert canonical.digest(key).sha256 == digest, (case_id, key)
        assert sum(scatter.unperturbed.final_status_histogram.values()) == scatter.lines
        assert scatter.unperturbed.poincare_hits_total == int(
            np.sum(canonical.array("poincare:counts"))
        )


def test_tracked_maxima_are_the_maximum_of_the_tracked_per_run_scatter() -> None:
    for case_id in official_tracing_scatter_case_ids():
        scatter = load_official_tracing_scatter(case_id)
        per_run = [
            run.scatter.as_mapping()
            for run in scatter.perturbed_runs
            if run.scatter is not None
        ]
        for quantity in scatter.maxima.as_mapping():
            assert scatter.maxima.quantity(quantity) == max(
                values[quantity] for values in per_run
            ), (case_id, quantity)


def test_tracing_records_hold_no_derived_ceiling_or_band() -> None:
    for case_id in official_tracing_scatter_case_ids():
        payload = json.loads(
            (TRACING_ROOT / f"{case_id}.json").read_text(encoding="utf-8")
        )
        assert _json_keys(payload) & FORBIDDEN_TRACING_KEYS == set(), case_id


def test_asking_for_a_tracing_record_a_case_does_not_have_lists_the_available_ones() -> (
    None
):
    with pytest.raises(MissingObservableError) as raised:
        load_official_tracing_scatter("native-boozer")
    assert "native-boozer" in str(raised.value)
    assert TRACING_CASES[0] in str(raised.value)


def test_an_unknown_scatter_quantity_is_named_with_the_available_ones() -> None:
    maxima = load_official_tracing_scatter(TRACING_CASES[0]).maxima
    with pytest.raises(MissingObservableError) as raised:
        maxima.quantity("final_position_distance_p95")
    assert "final_position_distance_p95" in str(raised.value)
    assert "final_position_distance_max" in str(raised.value)
