"""The tracked TRACING scatter records are complete, self-consistent and reproducible.

Like the rest of the official-reference fixture, these tests import neither simsopt nor jax: the record must be
readable on a clean checkout with nothing built.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity.official_reference import (
    REFERENCE_ROOT,
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
from examples.jax.parity.official_reference.build_official_reference import (
    array_entry,
    render_readme,
)
from examples.jax.parity.official_reference.build_official_tracing_scatter import (
    TRACING_DIRECTORY_NAME,
    TRACING_SUMMARY_NAME,
    TRACING_UNPERTURBED_K,
    write_official_tracing_scatter,
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
#: The optional quantity of the synthetic case below.
SYNTHETIC_CASE_QUANTITY = "final_parallel_speed_fraction_difference_max"

#: Keys the coordinator ruled must NOT appear: the record carries upstream's numbers, never a derived bound.
FORBIDDEN_TRACING_KEYS = frozenset(
    {"ceiling", "band", "spread", "tolerance", "bound", "factor", "headroom"}
)

SYNTHETIC_CASE = "native-synth-trace"
SYNTHETIC_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SYNTHETIC_SCRIPT = "examples/1_Simple/synth_trace.py"
OVER_INLINE_LIMIT = 1025


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
        assert "d1-diagnostic/NOTES.md" in scatter.protocol.pre_registered_in
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


def test_the_tracked_readme_is_in_step_with_the_generator_text() -> None:
    payloads = [
        json.loads((REFERENCE_ROOT / f"{case_id}.json").read_text(encoding="utf-8"))
        for case_id in official_case_ids()
    ]
    assert (REFERENCE_ROOT / "README.md").read_text(encoding="utf-8") == render_readme(
        payloads, UPSTREAM_COMMIT
    )
    assert f"## `{TRACING_DIRECTORY_NAME}/`" in (
        REFERENCE_ROOT / "README.md"
    ).read_text(encoding="utf-8")


def _scatter_values(k: int, case_quantity: bool = False) -> dict[str, object]:
    values = {
        "status_changes": 0,
        "hit_count_difference_per_line_max_abs": k,
        "hit_count_difference_per_plane_max_abs": k,
        "final_time_difference_max": 0.5 * k,
        "final_state_distance_max": 1.5 * k,
        "final_position_distance_max": 1.25 * k,
        "geometry_max_over_groups": 0.25 * k,
        "geometry_max_of_group_medians": 0.125 * k,
    }
    if case_quantity:
        values[SYNTHETIC_CASE_QUANTITY] = 0.75 * k
    return values


def _write_run(directory: Path, k: int, times: np.ndarray, states: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "capture-arrays.npz", **{"final:states": states})
    (directory / "capture.json").write_text(
        json.dumps(
            {
                "case_id": SYNTHETIC_CASE,
                "threads": {"OMP_NUM_THREADS": "1"},
                "array_file": {"arrays": {"final:states": {"dtype": "float64"}}},
                "observables": {
                    "final:times": times.tolist(),
                    "label": "traced",
                },
                "optimizer_calls": [],
            }
        ),
        encoding="utf-8",
    )
    (directory / "run.json").write_text(
        json.dumps({"case_id": SYNTHETIC_CASE, "k": k, "exit_code": 0}),
        encoding="utf-8",
    )
    (directory / "perturbation.json").write_text(
        json.dumps(
            {
                "k": k,
                "seed": 20260920 + k,
                "perturbed_arguments": None if k == 0 else ["R0", "Z0"],
                "entries": None if k == 0 else 4,
                "entries_changed": None if k == 0 else 4,
                "max_abs_delta": None if k == 0 else 2.220446049250313e-16,
                "max_rel_delta": None if k == 0 else 1.1e-16,
            }
        ),
        encoding="utf-8",
    )


def _build_synthetic_tree(
    tmp_path: Path,
    *,
    perturbed: tuple[int, ...] = (1,),
    case_quantity_ks: tuple[int, ...] = (),
) -> tuple[Path, Path]:
    """A tracing-sampling directory with an unperturbed run plus ``perturbed`` runs, and a reference root.

    ``case_quantity_ks`` names the perturbed runs whose scatter also records the per-case quantity, so a test can
    build a case that has it, a case that has not, and the inconsistent case where only some runs recorded it.
    """
    sampling = tmp_path / "tracing-sampling"
    reference = tmp_path / "reference"
    reference.mkdir(parents=True)

    times = np.asarray([1.0, 2.0], dtype=np.float64)
    states = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    for k in (0, *perturbed):
        _write_run(sampling / "runs" / SYNTHETIC_CASE / f"k{k}", k, times, states)

    (reference / f"{SYNTHETIC_CASE}.json").write_text(
        json.dumps(
            {
                "case_id": SYNTHETIC_CASE,
                "official_script": SYNTHETIC_SCRIPT,
                "official_script_sha256": hashlib.sha256(b"synth").hexdigest(),
                "observables": {
                    "final:times": array_entry(times),
                    "final:states": array_entry(states),
                    "poincare:hit_table": array_entry(
                        np.arange(OVER_INLINE_LIMIT, dtype=np.float64)
                    ),
                    "label": {"kind": "scalar", "value": "traced"},
                },
            }
        ),
        encoding="utf-8",
    )
    per_k = {str(k): _scatter_values(k, k in case_quantity_ks) for k in perturbed}
    maxima = {
        name: max(values[name] for values in per_k.values() if name in values)
        for name in {name for values in per_k.values() for name in values}
    }
    (sampling / TRACING_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": SYNTHETIC_CASE,
                        "lines": 2,
                        "k0_final_status_histogram": {"0": 2},
                        "k0_poincare_hits_total": 7,
                        "maxima_over_k": maxima,
                        "per_k": per_k,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return sampling, reference


def _generate(sampling: Path, reference: Path) -> tuple[Path, ...]:
    return write_official_tracing_scatter(
        tracing_sampling=sampling,
        reference_root=reference,
        upstream_commit=SYNTHETIC_COMMIT,
    )


def test_generator_is_deterministic_and_records_the_pre_registered_protocol(
    tmp_path: Path,
) -> None:
    sampling, reference = _build_synthetic_tree(tmp_path)
    first = _generate(sampling, reference)
    firsts = [path.read_bytes() for path in first]
    assert [path.name for path in first] == [f"{SYNTHETIC_CASE}.json"]
    assert _generate(sampling, reference) == first
    assert [path.read_bytes() for path in first] == firsts

    payload = json.loads(first[0].read_text(encoding="utf-8"))
    assert payload["case_id"] == SYNTHETIC_CASE
    assert payload["upstream_commit"] == SYNTHETIC_COMMIT
    assert payload["official_script"] == SYNTHETIC_SCRIPT
    assert [run["k"] for run in payload["runs"]] == [0, 1]
    assert payload["runs"][0]["scatter"] is None
    assert payload["runs"][1]["scatter"] == _scatter_values(1)
    assert payload["maxima_over_k"] == _scatter_values(1)
    assert payload["protocol"]["perturbed_arguments"] == ["R0", "Z0"]
    assert set(payload["unperturbed"]["observable_sha256"]) == {
        "final:times",
        "final:states",
    }
    assert _json_keys(payload) & FORBIDDEN_TRACING_KEYS == set()
    text = first[0].read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert (
        text
        == json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


def test_a_per_case_quantity_reaches_the_record_only_when_the_sampling_measured_it(
    tmp_path: Path,
) -> None:
    """A quantity only some cases have travels from the sampling into the record, and no further.

    The record is the ONE place that says which case measured which quantity: the contract reads the items it
    finds there. A case whose sampling did not measure the quantity must produce a record without the key, and
    the loader must present exactly the recorded list.
    """
    sampling, reference = _build_synthetic_tree(
        tmp_path, perturbed=(1, 2), case_quantity_ks=(1, 2)
    )
    payload = json.loads(_generate(sampling, reference)[0].read_text(encoding="utf-8"))
    assert payload["runs"][1]["scatter"] == _scatter_values(1, case_quantity=True)
    assert payload["runs"][2]["scatter"] == _scatter_values(2, case_quantity=True)
    assert payload["maxima_over_k"] == _scatter_values(2, case_quantity=True)
    assert payload["maxima_over_k"][SYNTHETIC_CASE_QUANTITY] == 1.5
    assert _json_keys(payload) & FORBIDDEN_TRACING_KEYS == set()

    without = tmp_path / "without"
    without.mkdir()
    other_sampling, other_reference = _build_synthetic_tree(without, perturbed=(1, 2))
    other = json.loads(
        _generate(other_sampling, other_reference)[0].read_text(encoding="utf-8")
    )
    assert SYNTHETIC_CASE_QUANTITY not in other["maxima_over_k"]
    assert SYNTHETIC_CASE_QUANTITY not in other["runs"][1]["scatter"]
    # The record file is written with sorted keys, so compare the SET the record holds.
    assert set(other["runs"][1]["scatter"]) == set(TRACING_SCATTER_QUANTITIES)


def test_generator_refuses_a_quantity_only_some_perturbed_runs_recorded(
    tmp_path: Path,
) -> None:
    """A maximum over a SUBSET of the samples is not the maximum the contract derives its ceiling from."""
    sampling, reference = _build_synthetic_tree(
        tmp_path, perturbed=(1, 2), case_quantity_ks=(2,)
    )
    with pytest.raises(SystemExit) as raised:
        _generate(sampling, reference)
    assert "different scatter quantities" in str(raised.value)


def test_generator_refuses_a_run_that_did_not_exit_cleanly(tmp_path: Path) -> None:
    sampling, reference = _build_synthetic_tree(tmp_path)
    (sampling / "runs" / SYNTHETIC_CASE / "k1" / "run.json").write_text(
        json.dumps({"case_id": SYNTHETIC_CASE, "k": 1, "exit_code": 1}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as raised:
        _generate(sampling, reference)
    assert "k=1" in str(raised.value)


def test_generator_refuses_maxima_that_contradict_the_sampling_summary(
    tmp_path: Path,
) -> None:
    sampling, reference = _build_synthetic_tree(tmp_path)
    summary = json.loads((sampling / TRACING_SUMMARY_NAME).read_text(encoding="utf-8"))
    summary["cases"][0]["maxima_over_k"]["final_position_distance_max"] = 99.0
    (sampling / TRACING_SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _generate(sampling, reference)
    assert "sampling summary" in str(raised.value)


def test_generator_refuses_an_unperturbed_run_that_is_not_the_canonical_run(
    tmp_path: Path,
) -> None:
    sampling, reference = _build_synthetic_tree(tmp_path)
    directory = sampling / "runs" / SYNTHETIC_CASE / "k0"
    capture = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    capture["observables"]["final:times"] = [1.0, 2.0000000000000004]
    (directory / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _generate(sampling, reference)
    assert "not the canonical official run" in str(raised.value)


def test_generator_refuses_when_the_unperturbed_run_lacks_an_inline_observable(
    tmp_path: Path,
) -> None:
    sampling, reference = _build_synthetic_tree(tmp_path)
    directory = sampling / "runs" / SYNTHETIC_CASE / "k0"
    capture = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    del capture["observables"]["final:times"]
    (directory / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _generate(sampling, reference)
    assert "did not record it" in str(raised.value)


def test_generator_refuses_a_sampled_case_without_a_canonical_record(
    tmp_path: Path,
) -> None:
    sampling, reference = _build_synthetic_tree(tmp_path)
    (reference / f"{SYNTHETIC_CASE}.json").unlink()
    with pytest.raises(SystemExit) as raised:
        _generate(sampling, reference)
    assert "no canonical record" in str(raised.value)


def test_an_absent_sampling_directory_simply_yields_no_records(tmp_path: Path) -> None:
    reference = tmp_path / "reference-empty"
    reference.mkdir()
    assert _generate(tmp_path / "nowhere", reference) == ()
    assert not (reference / TRACING_DIRECTORY_NAME).exists()
