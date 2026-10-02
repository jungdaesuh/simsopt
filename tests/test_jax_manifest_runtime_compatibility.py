"""Runtime adaptation contract for the canonical example manifest."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import ManifestV3ValidationError
from examples.jax.manifest_runtime import load_runtime_manifest
from examples.jax.official_source_catalog import OFFICIAL_NATIVE_EXAMPLE_SOURCES

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXAMPLES = REPO_ROOT / "examples" / "jax" / "manifest.json"


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _records(document: dict[str, object], key: str) -> list[dict[str, object]]:
    values = document[key]
    assert isinstance(values, list)
    assert all(isinstance(value, dict) for value in values)
    return values


def test_active_manifest_is_the_canonical_exact_mirror_contract() -> None:
    runtime = load_runtime_manifest(ACTIVE_EXAMPLES, repo_root=REPO_ROOT)
    catalog = _document(ACTIVE_EXAMPLES)
    jax_examples = _records(catalog, "jax_examples")
    one_to_one_doc = [
        example for example in jax_examples if example["teaching_kind"] == "one_to_one"
    ]
    assert runtime.schema_version == 3
    assert {str(row["source"]) for row in _records(catalog, "source_catalog")} == set(
        OFFICIAL_NATIVE_EXAMPLE_SOURCES
    )
    assert len(runtime.examples) == len(jax_examples)
    assert sum(example.status == "ready" for example in runtime.examples) == sum(
        example["status"] == "ready" for example in jax_examples
    )
    one_to_one = tuple(
        example for example in runtime.examples if example.teaching_kind == "one_to_one"
    )
    assert len(one_to_one) == len(one_to_one_doc)
    assert sum(example.status == "ready" for example in one_to_one) == sum(
        example["status"] == "ready" for example in one_to_one_doc
    )
    hybrid = next(
        example for example in one_to_one if example.classification == "hybrid"
    )
    assert hybrid.status == "planned"


def test_runtime_loader_rejects_a_retired_example_schema(tmp_path: Path) -> None:
    examples = _document(ACTIVE_EXAMPLES)
    examples["schema_version"] = 2
    retired_examples = tmp_path / "manifest.json"
    retired_examples.write_text(json.dumps(examples), encoding="utf-8")

    with pytest.raises(ManifestV3ValidationError, match="unsupported example schema"):
        load_runtime_manifest(retired_examples, repo_root=REPO_ROOT)
