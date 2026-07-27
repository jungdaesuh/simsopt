from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    read_array,
    write_array,
)
from examples.jax.parity.contracts import (
    InitialStateResult,
    ParityInputMetadata,
    validate_authoritative_source,
)
from examples.jax.parity.report import render_results_document


def test_parity_contracts_are_frozen() -> None:
    metadata = ParityInputMetadata(
        case_id="quadratic",
        random_seed=17,
        input_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.random_seed = 18  # type: ignore[misc]

    initial = InitialStateResult(
        objective_sum_squares=5.0,
        solver_cost=2.5,
        applicability={"residual": True, "constraints": False},
        arrays={},
    )
    assert initial.objective_gradient_factor == 2.0


def test_canonical_json_has_stable_key_order_and_rejects_nonfinite() -> None:
    left = canonical_json_bytes({"z": [1, 2], "a": {"b": True}})
    right = canonical_json_bytes({"a": {"b": True}, "z": [1, 2]})

    assert left == right == b'{"a":{"b":true},"z":[1,2]}\n'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"objective": float("nan")})


def test_npy_sidecar_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=">f8", order="F")

    first = write_array(first_root, "arrays/residual.npy", values)
    second = write_array(second_root, "arrays/residual.npy", values)

    assert first == second
    assert (first_root / first.path).read_bytes() == (
        second_root / second.path
    ).read_bytes()
    loaded = read_array(first_root, first)
    np.testing.assert_array_equal(loaded, values)
    assert loaded.dtype == np.dtype("<f8")
    assert loaded.flags.c_contiguous


@pytest.mark.parametrize("relative_path", ["../escape.npy", "/tmp/escape.npy"])
def test_sidecar_rejects_paths_outside_run_root(
    tmp_path: Path, relative_path: str
) -> None:
    with pytest.raises(ArtifactValidationError, match="canonical relative path"):
        write_array(tmp_path, relative_path, np.array([1.0]))


def test_sidecar_rejects_symlink_and_hash_mismatch(tmp_path: Path) -> None:
    reference = write_array(tmp_path, "arrays/value.npy", np.array([1.0]))
    target = tmp_path / reference.path
    target.write_bytes(target.read_bytes() + b"corruption")
    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        read_array(tmp_path, reference)

    outside = tmp_path / "outside.npy"
    np.save(outside, np.array([2.0]), allow_pickle=False)
    symlink = tmp_path / "arrays" / "link.npy"
    symlink.symlink_to(outside)
    linked = dataclasses.replace(reference, path="arrays/link.npy")
    with pytest.raises(ArtifactValidationError, match="symlink"):
        read_array(tmp_path, linked)


@pytest.mark.parametrize(
    "values",
    [np.array([np.nan]), np.array([object()], dtype=object)],
)
def test_sidecar_rejects_nonfinite_and_object_arrays(
    tmp_path: Path, values: np.ndarray
) -> None:
    with pytest.raises(ArtifactValidationError, match="forbidden"):
        write_array(tmp_path, "arrays/invalid.npy", values)


def test_authoritative_source_requires_clean_commit_and_binary_hash() -> None:
    validate_authoritative_source(
        authoritative=True,
        repository_dirty=False,
        repository_commit="c" * 40,
        executed_source_hashes={"examples/jax/run_parity.py": "d" * 64},
        simsoptpp_path="/opt/simsopt/simsoptpp.so",
        simsoptpp_sha256="e" * 64,
    )

    with pytest.raises(ValueError, match="clean repository"):
        validate_authoritative_source(
            authoritative=True,
            repository_dirty=True,
            repository_commit="c" * 40,
            executed_source_hashes={"examples/jax/run_parity.py": "d" * 64},
            simsoptpp_path=None,
            simsoptpp_sha256=None,
        )

    with pytest.raises(ValueError, match="hexadecimal"):
        validate_authoritative_source(
            authoritative=True,
            repository_dirty=False,
            repository_commit="z" * 40,
            executed_source_hashes={"examples/jax/run_parity.py": "d" * 64},
            simsoptpp_path=None,
            simsoptpp_sha256=None,
        )


def test_array_reference_round_trips_through_json(tmp_path: Path) -> None:
    reference = write_array(tmp_path, "arrays/x.npy", np.arange(3, dtype=np.float64))
    payload = json.loads(canonical_json_bytes(dataclasses.asdict(reference)))

    assert payload["path"] == "arrays/x.npy"
    assert payload["shape"] == [3]
    assert len(payload["sha256"]) == 64
    assert not os.path.isabs(payload["path"])


def test_results_report_is_generated_from_aggregate_fields() -> None:
    summary = {
        "schema_version": 1,
        "run_id": "20260726T000000Z-deadbeef",
        "verdict": "pass",
        "authoritative": False,
        "repository_dirty": True,
        "repository_commit": "a" * 40,
        "lanes": ["native-cpu", "jax-cpu", "jax-gpu"],
        "cases": [
            {
                "jax_example_id": "traceable-least-squares",
                "native_source": "1_Simple/just_a_quadratic.py",
                "classification": "full",
                "scale_tier": "bounded",
                "oracle_kind": "native_python_scipy",
                "verdict": "pass",
                "comparisons": [{"passed": True}, {"passed": True}],
            }
        ],
    }

    rendered = render_results_document(summary, artifact_reference="artifact/run")

    assert "Evidence class: **exploratory** (dirty checkout)" in rendered
    assert "cannot promote an authoritative parity claim" in rendered
    assert "| traceable-least-squares |" in rendered
    assert "| full | bounded | native_python_scipy | pass | 2/2 |" in rendered

    summary["authoritative"] = True
    summary["repository_dirty"] = False
    authoritative = render_results_document(summary, artifact_reference="artifact/run")
    assert "Evidence class: **authoritative** (clean checkout)" in authoritative
    assert "may promote only the classifications and bounded scale" in authoritative
