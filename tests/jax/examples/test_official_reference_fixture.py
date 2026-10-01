"""The tracked official-reference fixture is complete and self-consistent.

These tests import neither simsopt nor jax: the fixture is the one place official numbers live, and reading it must
work on a clean checkout with nothing built.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
import math
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity.official_reference import (
    UPSTREAM_SCATTER_ROOT,
    CANONICAL_VARIANT,
    SENSITIVITY_ROOT,
    INLINE_ELEMENT_LIMIT,
    REFERENCE_ROOT,
    TRACING_ROOT,
    UPSTREAM_COMMIT,
    ArrayDigest,
    MissingObservableError,
    MissingVariantError,
    ObservableKindError,
    load_official_reference,
    load_official_sensitivity,
    official_case_ids,
    official_sensitivity_case_ids,
    official_variants,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "examples" / "jax" / "manifest.json"
README = REFERENCE_ROOT / "README.md"
PROTOCOL_DOCUMENT = (
    "examples/jax/parity/official_reference/9e027eac3/README.md, "
    "fixed before any sample existed"
)


def _official_case_ids_from_manifest() -> tuple[str, ...]:
    """Case ids an upstream source claims as its shipped mirror."""
    catalog = json.loads(MANIFEST.read_text(encoding="utf-8"))["source_catalog"]
    return tuple(
        sorted(
            str(entry["mirror_example_id"])
            for entry in catalog
            if entry["mirror_example_id"] is not None
            and entry["port_status"] == "ready"
        )
    )


def _readme_rows() -> dict[str, tuple[str, str, str, tuple[str, ...]]]:
    """``case_id -> (official_script, script_sha256, canonical_capture, variants)`` from the README table."""
    rows: dict[str, tuple[str, str, str, tuple[str, ...]]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        listed = () if cells[4] == "--" else tuple(cells[4].replace("`", "").split())
        rows[cells[0]] = (cells[1], cells[2], cells[3], listed)
    return rows


def test_fixture_covers_exactly_the_official_case_ids() -> None:
    expected = _official_case_ids_from_manifest()
    assert len(expected) == 25
    assert official_case_ids() == expected
    present = sorted(path.name for path in REFERENCE_ROOT.iterdir())
    assert present == sorted(
        [
            *(f"{case_id}.json" for case_id in expected),
            "README.md",
            SENSITIVITY_ROOT.name,
            TRACING_ROOT.name,
            UPSTREAM_SCATTER_ROOT.name,
        ]
    )
    assert SENSITIVITY_ROOT.is_dir()
    sampled = sorted(path.name for path in SENSITIVITY_ROOT.iterdir())
    assert sampled == [f"{case_id}.json" for case_id in official_sensitivity_case_ids()]
    assert set(official_sensitivity_case_ids()) <= set(expected)


def test_every_case_round_trips_and_inline_digests_are_self_consistent() -> None:
    variant_counts = {CANONICAL_VARIANT: 0, "ci": 0}
    for case_id in official_case_ids():
        variants = official_variants(case_id)
        assert variants[0] == CANONICAL_VARIANT
        assert len(set(variants)) == len(variants)
        for variant in variants:
            variant_counts[variant] += 1
            reference = load_official_reference(case_id, variant=variant)
            assert reference.case_id == case_id
            assert reference.variant == variant
            assert reference.upstream_commit == UPSTREAM_COMMIT
            assert len(reference.official_script_sha256) == 64
            assert reference.capture.path.count("/") >= 2
            assert len(reference.capture.capture_json_sha256) == 64
            assert reference.keys(), f"{case_id}:{variant}"
            for call in reference.provider_calls:
                assert call.function
                assert set(call.result) >= {"status", "success", "message", "nfev"}
            for key in reference.keys():
                kind = reference.kind(key)
                if kind == "scalar":
                    assert isinstance(
                        reference.scalar(key), (bool, int, float, str, type(None))
                    )
                elif kind == "structure":
                    assert isinstance(reference.structure(key), (dict, list))
                else:
                    digest = reference.digest(key)
                    assert digest.count == int(np.prod(digest.shape, dtype=np.int64))
                    if reference.is_inline(key):
                        assert digest.count <= INLINE_ELEMENT_LIMIT
                        values = reference.array(key)
                        assert values.shape == digest.shape
                        assert str(values.dtype) == digest.dtype
                        assert ArrayDigest.of(values) == digest, (
                            f"{case_id}:{variant}:{key}"
                        )
                    else:
                        assert digest.count > INLINE_ELEMENT_LIMIT
                        with pytest.raises(ObservableKindError):
                            reference.array(key)
    assert variant_counts == {CANONICAL_VARIANT: 25, "ci": 13}


def test_the_ci_variant_is_a_distinct_bounded_run_of_the_same_official_script() -> None:
    canonical = load_official_reference("native-permanent-magnet-muse")
    ci = load_official_reference("native-permanent-magnet-muse", variant="ci")
    assert ci.official_script == canonical.official_script
    assert ci.official_script_sha256 == canonical.official_script_sha256
    assert ci.capture.path.endswith("captured-omp1-ci")
    assert ci.capture.path != canonical.capture.path
    assert ci.capture.capture_json_sha256 != canonical.capture.capture_json_sha256
    assert ci.structure("configuration") != canonical.structure("configuration")


def test_asking_for_a_variant_a_case_does_not_have_names_the_available_ones() -> None:
    assert official_variants("native-minimize-curve-length") == (CANONICAL_VARIANT,)
    with pytest.raises(MissingVariantError) as raised:
        load_official_reference("native-minimize-curve-length", variant="ci")
    assert "ci" in str(raised.value)
    assert CANONICAL_VARIANT in str(raised.value)


def test_recorded_script_provenance_matches_the_tracked_readme_table() -> None:
    rows = _readme_rows()
    assert sorted(rows) == list(official_case_ids())
    for case_id, (script, script_sha256, capture_path, listed) in rows.items():
        reference = load_official_reference(case_id)
        assert reference.official_script == script
        assert reference.official_script_sha256 == script_sha256
        assert reference.capture.path == capture_path
        assert official_variants(case_id) == (CANONICAL_VARIANT, *listed)
        for variant in listed:
            assert (
                load_official_reference(case_id, variant=variant).capture.path
                != capture_path
            )


SENSITIVITY_CASES = (
    "native-boozerqa",
    "native-coil-forces",
    "native-qfm",
    "native-stage-two-optimization-finitebuild",
    "native-stage-two-optimization-minimal",
)
SENSITIVITY_KS = tuple(range(9))
#: Keys the coordinator ruled must NOT appear: the record carries upstream's numbers, never a derived bound.
FORBIDDEN_SENSITIVITY_KEYS = frozenset(
    {"ceiling", "band", "S", "max_S", "min_S", "spread", "tolerance", "bound"}
)


def _json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _json_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _json_keys(item)}
    return set()


#: Cases whose capture names the sampled quantity differently from the port lane (the band judges the lane key).
LANE_KEY_DIFFERS = {"native-coil-forces": ("stage2:final:objective", "final:objective")}


def test_sensitivity_records_cover_the_five_sampled_cases() -> None:
    assert official_sensitivity_case_ids() == SENSITIVITY_CASES


def test_sensitivity_lane_observable_names_the_port_key_of_the_sampled_quantity() -> (
    None
):
    for case_id in official_sensitivity_case_ids():
        sensitivity = load_official_sensitivity(case_id)
        phase, separator, name = sensitivity.lane_observable.partition(":")
        assert phase and separator and name, case_id
        if case_id in LANE_KEY_DIFFERS:
            assert (
                sensitivity.observable,
                sensitivity.lane_observable,
            ) == LANE_KEY_DIFFERS[case_id]
        else:
            assert sensitivity.lane_observable == sensitivity.observable, case_id


def test_sensitivity_k0_reproduces_the_canonical_fixture_value_bitwise() -> None:
    for case_id in official_sensitivity_case_ids():
        sensitivity = load_official_sensitivity(case_id)
        canonical = load_official_reference(case_id)
        assert canonical.kind(sensitivity.observable) == "scalar"
        unperturbed = sensitivity.run(sensitivity.protocol.unperturbed_k)
        assert unperturbed.k == 0
        assert unperturbed.end_value == canonical.scalar(sensitivity.observable), (
            case_id
        )
        assert sensitivity.official_script == canonical.official_script
        assert sensitivity.official_script_sha256 == canonical.official_script_sha256
        assert sensitivity.upstream_commit == UPSTREAM_COMMIT


def test_every_sensitivity_record_round_trips_with_nine_runs_k0_to_k8() -> None:
    for case_id in official_sensitivity_case_ids():
        sensitivity = load_official_sensitivity(case_id)
        assert sensitivity.case_id == case_id
        assert tuple(run.k for run in sensitivity.runs) == SENSITIVITY_KS
        assert len(sensitivity.runs) == len(SENSITIVITY_KS)
        assert len(sensitivity.end_values) == len(SENSITIVITY_KS)
        assert all(math.isfinite(value) for value in sensitivity.end_values)
        assert sensitivity.protocol.perturbed_call_index == 0
        assert "nextafter" in sensitivity.protocol.perturbation_rule
        assert "20260920 + k" in sensitivity.protocol.seed_rule
        assert sensitivity.protocol.threads["OMP_NUM_THREADS"] == "1"
        assert sensitivity.protocol.pre_registered_in == PROTOCOL_DOCUMENT
        call_counts = {len(run.provider_calls) for run in sensitivity.runs}
        assert len(call_counts) == 1, case_id
        for run in sensitivity.runs:
            assert len(run.capture_sha256) == 64
            assert len(run.perturbation_sha256) == 64
            assert run.capture_sha256 != run.perturbation_sha256
            assert tuple(call.index for call in run.provider_calls) == tuple(
                range(len(run.provider_calls))
            )
            for call in run.provider_calls:
                assert call.message
                assert call.nfev >= call.nit >= 0
                assert isinstance(call.success, bool)
                assert math.isfinite(call.fun)
        assert len({run.capture_sha256 for run in sensitivity.runs}) == len(
            sensitivity.runs
        )


def test_sensitivity_records_hold_no_derived_band_or_ceiling() -> None:
    for case_id in official_sensitivity_case_ids():
        payload = json.loads(
            (SENSITIVITY_ROOT / f"{case_id}.json").read_text(encoding="utf-8")
        )
        assert _json_keys(payload) & FORBIDDEN_SENSITIVITY_KEYS == set(), case_id


def test_asking_for_a_sensitivity_record_a_case_does_not_have_lists_the_available_ones() -> (
    None
):
    with pytest.raises(MissingObservableError) as raised:
        load_official_sensitivity("native-boozer")
    assert "native-boozer" in str(raised.value)
    assert "native-qfm" in str(raised.value)


def test_missing_key_error_names_the_key_and_lists_the_available_ones() -> None:
    reference = load_official_reference("native-minimize-curve-length")
    with pytest.raises(MissingObservableError) as raised:
        reference.scalar("final:no-such-observable")
    message = str(raised.value)
    assert "final:no-such-observable" in message
    assert "final:length" in message
