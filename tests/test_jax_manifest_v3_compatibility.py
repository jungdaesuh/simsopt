"""Compatibility-alias contract for non-covering legacy JAX examples."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import (
    ManifestV3ValidationError,
    build_v3_candidates,
    parse_examples_v3_document,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _document(relative_path: str) -> dict[str, object]:
    value = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _candidate() -> dict[str, object]:
    candidate = build_v3_candidates(
        examples_v2_document=_document(
            "tests/fixtures/jax_manifests/manifest_v2.json"
        ),
        parity_v1_document=_document(
            "tests/fixtures/jax_manifests/parity_manifest_v1.json"
        ),
        inventory_document=_document("examples/jax/one_to_one_inventory.json"),
        repo_root=REPO_ROOT,
    )
    value = json.loads(candidate.examples_bytes)
    assert isinstance(value, dict)
    return value


def _examples(document: dict[str, object]) -> list[dict[str, object]]:
    values = document["jax_examples"]
    assert isinstance(values, list)
    assert all(isinstance(value, dict) for value in values)
    return values


def test_compatibility_tutorials_bind_successor_warning_and_removal_interval() -> None:
    candidate = _candidate()
    examples = _examples(candidate)
    aliases = [
        example for example in examples if example["teaching_kind"] == "compatibility"
    ]
    assert len(aliases) == 5
    by_id = {str(example["id"]): example for example in examples}
    for alias in aliases:
        assert "compatibility" in alias, "compatibility alias metadata is missing"
        metadata = alias["compatibility"]
        assert isinstance(metadata, dict)
        assert frozenset(metadata) == {
            "successor_example_id",
            "warning",
            "removal_after",
        }
        successor_id = metadata["successor_example_id"]
        assert isinstance(successor_id, str)
        successor = by_id[successor_id]
        assert successor["teaching_kind"] == "one_to_one"
        warning = metadata["warning"]
        assert isinstance(warning, str)
        assert str(alias["id"]) in warning
        assert successor_id in warning
        assert metadata["removal_after"] == "one documented deprecation interval"

    assert all(
        example.get("compatibility") is None
        for example in examples
        if example["teaching_kind"] != "compatibility"
    )


def test_schema_rejects_missing_or_unresolvable_compatibility_metadata() -> None:
    candidate = _candidate()
    alias = next(
        example
        for example in _examples(candidate)
        if example["teaching_kind"] == "compatibility"
    )

    missing = copy.deepcopy(candidate)
    missing_alias = next(
        example for example in _examples(missing) if example["id"] == alias["id"]
    )
    missing_alias["compatibility"] = None
    with pytest.raises(ManifestV3ValidationError, match="compatibility metadata"):
        parse_examples_v3_document(missing, repo_root=REPO_ROOT)

    unknown = copy.deepcopy(candidate)
    unknown_alias = next(
        example for example in _examples(unknown) if example["id"] == alias["id"]
    )
    metadata = unknown_alias["compatibility"]
    assert isinstance(metadata, dict)
    metadata["successor_example_id"] = "missing-successor"
    with pytest.raises(ManifestV3ValidationError, match="unknown successor"):
        parse_examples_v3_document(unknown, repo_root=REPO_ROOT)
