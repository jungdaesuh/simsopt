"""The tracked official-reference fixture is complete, self-consistent and reproducible.

These tests import neither simsopt nor jax: the fixture is the one place official numbers live, and reading it must
work on a clean checkout with nothing built.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity import official_reference
from examples.jax.parity.official_reference import (
    CANONICAL_VARIANT,
    SENSITIVITY_ROOT,
    INLINE_ELEMENT_LIMIT,
    REFERENCE_ROOT,
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
from examples.jax.parity.official_reference.build_official_reference import (
    SENSITIVITY_DIRECTORY_NAME,
    VARIANT_DIRECTORY_NAMES,
    is_numeric_array,
    nested_shape,
    official_case_ids_from_manifest,
    parse_variant,
    resolve_capture_directory,
    sha256_file,
    write_official_reference,
)
from examples.jax.parity.official_reference.build_official_tracing_scatter import (
    TRACING_DIRECTORY_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "examples" / "jax" / "manifest.json"
README = REFERENCE_ROOT / "README.md"

CLAUDE_CASE = "native-synth-claude"
SIMPLE_CASE = "native-synth-simple"
PLANNED_CASE = "native-synth-planned"
SYNTHETIC_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# Literal sizes, deliberately NOT derived from INLINE_ELEMENT_LIMIT: they pin the shipped boundary, so widening
# the constant without regenerating and re-agreeing the rule makes this file fail.
INLINE_SIZE = 1024
OVER_SIZE = 1025


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
    expected = official_case_ids_from_manifest(MANIFEST)
    assert len(expected) == 25
    assert official_case_ids() == expected
    present = sorted(path.name for path in REFERENCE_ROOT.iterdir())
    assert present == sorted(
        [
            *(f"{case_id}.json" for case_id in expected),
            "README.md",
            SENSITIVITY_DIRECTORY_NAME,
            TRACING_DIRECTORY_NAME,
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
    assert ci.capture.path.endswith(VARIANT_DIRECTORY_NAMES["ci"])
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
        assert sensitivity.protocol.pre_registered_in.endswith(
            "v2, fixed 2026-09-20 02:52 EDT)"
        )
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


def test_nested_shape_and_numeric_array_classification() -> None:
    assert nested_shape([[1.0, 2.0], [3.0, 4.0]]) == (2, 2)
    assert nested_shape([[1.0], [2.0, 3.0]]) is None
    assert nested_shape([]) is None
    assert nested_shape(3.0) == ()
    assert is_numeric_array([1.0, 2.0])
    assert not is_numeric_array([{"a": 1}])
    assert not is_numeric_array([1.0, None])
    assert not is_numeric_array([])


def _write_capture(
    directory: Path, payload: dict[str, object], arrays: dict[str, np.ndarray] | None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        np.savez(directory / "capture-arrays.npz", **arrays)
        payload["array_file"] = {
            "arrays": {name: {"dtype": str(a.dtype)} for name, a in arrays.items()}
        }
    (directory / "capture.json").write_text(json.dumps(payload), encoding="utf-8")


def _build_synthetic_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A campaign root with both capture layouts, a manifest and an official source clone."""
    campaign = tmp_path / "campaign"
    official_src = tmp_path / "official-src"

    scripts = {
        CLAUDE_CASE: "examples/2_Intermediate/synth_claude.py",
        SIMPLE_CASE: "examples/1_Simple/synth_simple.py",
    }
    contracts = campaign / "claude" / "contracts"
    contracts.mkdir(parents=True)
    for case_id, script in scripts.items():
        script_path = official_src / script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(f"# {case_id}\n", encoding="utf-8")
        (contracts / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "official_script": script,
                    "script_sha256": sha256_file(script_path),
                }
            ),
            encoding="utf-8",
        )

    _write_capture(
        campaign / "claude" / "runs" / CLAUDE_CASE / "captured-omp1",
        {
            "case_id": CLAUDE_CASE,
            "python": "3.11.15",
            "threads": {"OMP_NUM_THREADS": "1"},
            "observables": {
                "final:objective": 1.5,
                "final:not_a_number": float("nan"),
                "final:with_nan": [1.0, float("nan"), -math.inf],
                "final:flag": True,
                "final:label": "converged",
                "final:nothing": None,
                "configuration": {"maxiter": 3, "bound": math.inf},
                "final:ledger": [{"iter": 0, "ok": True}],
            },
            "optimizer_calls": [
                {
                    "function": "scipy.optimize.minimize",
                    "method": "L-BFGS-B",
                    "options": {"maxiter": 7},
                    "tol": 1e-15,
                    "wall_seconds": 12.5,
                    "x0": [0.0, 1.0],
                    "result": {
                        "status": 0,
                        "success": True,
                        "message": "CONVERGED",
                        "nfev": 9,
                        "njev": 9,
                        "nit": 8,
                        "fun": 0.25,
                        "x": [0.5, 0.5],
                    },
                }
            ],
        },
        {
            "final:inline_limit": np.arange(INLINE_SIZE, dtype=np.float64),
            "final:over_limit": np.arange(OVER_SIZE, dtype=np.float64),
            "final:counts": np.arange(4, dtype=np.int32),
        },
    )

    _write_capture(
        campaign / "claude" / "runs" / CLAUDE_CASE / VARIANT_DIRECTORY_NAMES["ci"],
        {
            "case_id": CLAUDE_CASE,
            "python": "3.11.15",
            "threads": {"OMP_NUM_THREADS": "1"},
            "observables": {
                "final:objective": 2.5,
                "configuration": {"maxiter": 1, "bound": -math.inf},
            },
            "optimizer_calls": [
                {
                    "function": "scipy.optimize.minimize",
                    "method": "L-BFGS-B",
                    "options": {"maxiter": 1},
                    "tol": 1e-15,
                    "result": {
                        "status": 1,
                        "success": False,
                        "message": "MAXITER",
                        "nfev": 2,
                        "njev": 2,
                        "nit": 1,
                        "fun": 2.5,
                    },
                }
            ],
        },
        None,
    )

    _write_capture(
        campaign
        / "reference-simple"
        / "runs"
        / SIMPLE_CASE
        / "captured-controlled-omp1",
        {
            "case_id": SIMPLE_CASE,
            "python": "3.11.15",
            "threads": {"OMP_NUM_THREADS": "1"},
            "observables": {
                "final:length": 18.0,
                "final:residual": [[1.0, 2.0], [3.0, 4.0]],
            },
            "optimizer_calls": [
                {
                    "function": "scipy.optimize.least_squares",
                    "keywords": {"bounds": [[-math.inf], [math.inf]], "verbose": 2},
                    "result": {
                        "status": 1,
                        "success": True,
                        "message": "gtol",
                        "nfev": 4,
                        "njev": 4,
                        "cost": 0.5,
                    },
                }
            ],
        },
        None,
    )

    sampling = campaign / "investigations" / "band-sampling"
    synthetic_ends = {0: 5.850853627735323e-07, 1: 6.204797180223713e-07}
    for k, end_value in synthetic_ends.items():
        directory = sampling / "runs" / CLAUDE_CASE / f"k{k}"
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(
            json.dumps({"case_id": CLAUDE_CASE, "k": k, "exit_code": 0}),
            encoding="utf-8",
        )
        (directory / "perturbation.json").write_text(
            json.dumps({"k": k, "seed": 20260920 + k}), encoding="utf-8"
        )
        (directory / "capture.json").write_text(
            json.dumps(
                {
                    "case_id": CLAUDE_CASE,
                    "threads": {"OMP_NUM_THREADS": "1"},
                    "observables": {"final:objective": end_value},
                    "optimizer_calls": [
                        {
                            "function": "scipy.optimize.minimize",
                            "method": "L-BFGS-B",
                            "result": {
                                "status": k,
                                "success": k == 0,
                                "nit": 300,
                                "nfev": 389,
                                "njev": 389,
                                "message": "STOP",
                                "fun": end_value,
                                "x": [0.0],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    (sampling / "band-sampling.json").write_text(
        json.dumps(
            {
                CLAUDE_CASE: {
                    "observable": "final:objective",
                    "lane_observable": "final:objective",
                    "ceiling": 1.0,
                    "runs": [
                        {"k": k, "band_value": end_value}
                        for k, end_value in synthetic_ends.items()
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_catalog": [
                    {
                        "source": scripts[CLAUDE_CASE],
                        "mirror_example_id": CLAUDE_CASE,
                        "port_status": "ready",
                    },
                    {
                        "source": scripts[SIMPLE_CASE],
                        "mirror_example_id": SIMPLE_CASE,
                        "port_status": "ready",
                    },
                    {
                        "source": "x.py",
                        "mirror_example_id": PLANNED_CASE,
                        "port_status": "planned",
                    },
                    {
                        "source": "y.py",
                        "mirror_example_id": None,
                        "port_status": "not_applicable",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return campaign, official_src, manifest


def _generate(campaign: Path, official_src: Path, manifest: Path, output: Path) -> None:
    write_official_reference(
        campaign_root=campaign,
        manifest=manifest,
        contracts=campaign / "claude" / "contracts",
        official_src=official_src,
        upstream_commit=SYNTHETIC_COMMIT,
        output=output,
        band_sampling=campaign / "investigations" / "band-sampling",
    )


def test_generator_is_deterministic_and_applies_the_inline_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, official_src, manifest = _build_synthetic_tree(tmp_path)
    first, second = tmp_path / "out-1", tmp_path / "out-2"
    _generate(campaign, official_src, manifest, first)
    _generate(campaign, official_src, manifest, second)

    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(
        [
            f"{CLAUDE_CASE}.json",
            f"{SIMPLE_CASE}.json",
            "README.md",
            SENSITIVITY_DIRECTORY_NAME,
        ]
    )
    assert PLANNED_CASE not in names
    sensitivity_names = sorted(
        path.name for path in (first / SENSITIVITY_DIRECTORY_NAME).iterdir()
    )
    assert sensitivity_names == [f"{CLAUDE_CASE}.json"]
    for relative in [
        f"{CLAUDE_CASE}.json",
        f"{SIMPLE_CASE}.json",
        "README.md",
        f"{SENSITIVITY_DIRECTORY_NAME}/{CLAUDE_CASE}.json",
    ]:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), (
            relative
        )

    sensitivity = json.loads(
        (first / SENSITIVITY_DIRECTORY_NAME / f"{CLAUDE_CASE}.json").read_text(
            encoding="utf-8"
        )
    )
    assert sensitivity["observable"] == "final:objective"
    assert sensitivity["lane_observable"] == "final:objective"
    assert [run["k"] for run in sensitivity["runs"]] == [0, 1]
    assert sensitivity["runs"][0]["end_value"] == 5.850853627735323e-07
    assert sensitivity["runs"][1]["provider_calls"][0] == {
        "index": 0,
        "method": "L-BFGS-B",
        "status": 1,
        "success": False,
        "nit": 300,
        "nfev": 389,
        "njev": 389,
        "message": "STOP",
        "fun": 6.204797180223713e-07,
    }
    assert "x" not in sensitivity["runs"][1]["provider_calls"][0]
    assert _json_keys(sensitivity) & FORBIDDEN_SENSITIVITY_KEYS == set()
    assert (
        sensitivity["official_script_sha256"]
        == json.loads((first / f"{CLAUDE_CASE}.json").read_text(encoding="utf-8"))[
            "official_script_sha256"
        ]
    )

    payload = json.loads((first / f"{CLAUDE_CASE}.json").read_text(encoding="utf-8"))
    text = (first / f"{CLAUDE_CASE}.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert (
        text
        == json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    observables = payload["observables"]
    assert "values" in observables["final:inline_limit"]
    assert observables["final:inline_limit"]["count"] == INLINE_SIZE
    assert "values" not in observables["final:over_limit"]
    assert observables["final:over_limit"]["count"] == OVER_SIZE
    assert observables["final:ledger"]["kind"] == "structure"
    assert observables["final:not_a_number"] == {
        "kind": "scalar",
        "value": {"$nonfinite": "nan"},
    }
    call = payload["provider_calls"][0]
    assert "wall_seconds" not in call and "x0" not in call
    assert call["result"] == {
        "status": 0,
        "success": True,
        "message": "CONVERGED",
        "nfev": 9,
        "njev": 9,
        "nit": 8,
        "fun": 0.25,
    }

    monkeypatch.setattr(official_reference, "REFERENCE_ROOT", first)
    assert official_case_ids() == (CLAUDE_CASE, SIMPLE_CASE)
    reference = load_official_reference(CLAUDE_CASE)
    assert reference.upstream_commit == SYNTHETIC_COMMIT
    assert reference.capture.path == f"claude/runs/{CLAUDE_CASE}/captured-omp1"
    assert math.isnan(reference.scalar("final:not_a_number"))
    with_nan = reference.array("final:with_nan")
    assert math.isnan(with_nan[1]) and with_nan[2] == -math.inf
    assert reference.structure("configuration")["bound"] == math.inf
    assert reference.array("final:counts").dtype == np.int32
    assert np.array_equal(reference.array("final:inline_limit"), np.arange(INLINE_SIZE))
    with pytest.raises(ObservableKindError):
        reference.array("final:over_limit")
    assert reference.digest("final:over_limit").count == OVER_SIZE

    assert set(payload["variants"]) == {"ci"}
    ci_record = payload["variants"]["ci"]
    assert ci_record["capture"]["path"] == (
        f"claude/runs/{CLAUDE_CASE}/{VARIANT_DIRECTORY_NAMES['ci']}"
    )
    assert set(ci_record) == {"capture", "provider_calls", "observables"}
    assert official_variants(CLAUDE_CASE) == (CANONICAL_VARIANT, "ci")
    ci = load_official_reference(CLAUDE_CASE, variant="ci")
    assert ci.variant == "ci"
    assert ci.official_script_sha256 == reference.official_script_sha256
    assert ci.scalar("final:objective") == 2.5
    assert reference.scalar("final:objective") == 1.5
    assert ci.provider_calls[0].result["success"] is False
    assert ci.structure("configuration")["bound"] == -math.inf
    with pytest.raises(MissingObservableError):
        ci.array("final:over_limit")

    simple = load_official_reference(SIMPLE_CASE)
    assert (
        simple.capture.path
        == f"reference-simple/runs/{SIMPLE_CASE}/captured-controlled-omp1"
    )
    assert simple.capture.array_file_sha256 is None
    assert simple.provider_calls[0].method is None
    assert simple.provider_calls[0].options["bounds"] == [[-math.inf], [math.inf]]
    assert simple.array("final:residual").shape == (2, 2)
    assert official_variants(SIMPLE_CASE) == (CANONICAL_VARIANT,)
    assert "variants" not in json.loads(
        (first / f"{SIMPLE_CASE}.json").read_text(encoding="utf-8")
    )
    with pytest.raises(MissingVariantError):
        load_official_reference(SIMPLE_CASE, variant="ci")


def test_generator_refuses_a_clone_that_is_not_at_the_recorded_commit(
    tmp_path: Path,
) -> None:
    campaign, official_src, manifest = _build_synthetic_tree(tmp_path)
    (official_src / "examples" / "1_Simple" / "synth_simple.py").write_text(
        "# drifted\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as raised:
        _generate(campaign, official_src, manifest, tmp_path / "out-drift")
    assert SIMPLE_CASE in str(raised.value)


def test_canonical_capture_directory_is_unambiguous(tmp_path: Path) -> None:
    campaign, _, _ = _build_synthetic_tree(tmp_path)
    claude_runs = campaign / "claude" / "runs"
    simple_runs = campaign / "reference-simple" / "runs"
    (simple_runs / SIMPLE_CASE / "captured-natural-omp1").mkdir()
    with pytest.raises(SystemExit):
        resolve_capture_directory(SIMPLE_CASE, claude_runs, simple_runs)
    assert (
        resolve_capture_directory(CLAUDE_CASE, claude_runs, simple_runs).name
        == "captured-omp1"
    )


def test_variant_argument_parsing() -> None:
    assert parse_variant("ci=captured-omp1-ci") == ("ci", "captured-omp1-ci")
    for bad in ("ci", "=captured-omp1-ci", "ci=", f"{CANONICAL_VARIANT}=captured-omp1"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variant(bad)


def test_a_case_without_the_named_variant_directory_simply_has_no_variant(
    tmp_path: Path,
) -> None:
    campaign, official_src, manifest = _build_synthetic_tree(tmp_path)
    output = tmp_path / "out-no-variant"
    write_official_reference(
        campaign_root=campaign,
        manifest=manifest,
        contracts=campaign / "claude" / "contracts",
        official_src=official_src,
        upstream_commit=SYNTHETIC_COMMIT,
        output=output,
        band_sampling=campaign / "investigations" / "band-sampling",
        variants={"absent": "captured-nowhere-omp1"},
    )
    for case_id in (CLAUDE_CASE, SIMPLE_CASE):
        payload = json.loads((output / f"{case_id}.json").read_text(encoding="utf-8"))
        assert "variants" not in payload


def test_an_absent_sampling_directory_simply_yields_no_sensitivity_records(
    tmp_path: Path,
) -> None:
    campaign, official_src, manifest = _build_synthetic_tree(tmp_path)
    output = tmp_path / "out-no-sampling"
    write_official_reference(
        campaign_root=campaign,
        manifest=manifest,
        contracts=campaign / "claude" / "contracts",
        official_src=official_src,
        upstream_commit=SYNTHETIC_COMMIT,
        output=output,
        band_sampling=tmp_path / "nowhere",
    )
    assert not (output / SENSITIVITY_DIRECTORY_NAME).exists()


def test_generator_refuses_a_sensitivity_run_that_did_not_exit_cleanly(
    tmp_path: Path,
) -> None:
    campaign, official_src, manifest = _build_synthetic_tree(tmp_path)
    run_json = (
        campaign
        / "investigations"
        / "band-sampling"
        / "runs"
        / CLAUDE_CASE
        / "k1"
        / "run.json"
    )
    run_json.write_text(
        json.dumps({"case_id": CLAUDE_CASE, "k": 1, "exit_code": 1}), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as raised:
        _generate(campaign, official_src, manifest, tmp_path / "out-bad-exit")
    assert "k=1" in str(raised.value)


def test_generator_refuses_a_sensitivity_end_value_that_contradicts_the_summary(
    tmp_path: Path,
) -> None:
    campaign, official_src, manifest = _build_synthetic_tree(tmp_path)
    capture = (
        campaign
        / "investigations"
        / "band-sampling"
        / "runs"
        / CLAUDE_CASE
        / "k1"
        / "capture.json"
    )
    payload = json.loads(capture.read_text(encoding="utf-8"))
    payload["observables"]["final:objective"] = 1.0
    capture.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _generate(campaign, official_src, manifest, tmp_path / "out-bad-value")
    assert "sampling summary" in str(raised.value)
