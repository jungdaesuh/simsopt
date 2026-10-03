"""Pinned upstream scope stays independent of local example files."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import copy
import json
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import (
    ManifestV3ValidationError,
    parse_examples_v3_document,
)
from examples.jax.official_source_catalog import (
    OFFICIAL_NATIVE_EXAMPLE_SOURCES,
    OFFICIAL_UPSTREAM_COMMIT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
#: An official source with a mirror and no host outer policy.
OFFICIAL_SOURCE = "1_Simple/tracing_particle.py"


def _document(relative_path: str) -> dict[str, object]:
    value = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _mutated_active_manifest() -> dict[str, object]:
    return copy.deepcopy(_document("examples/jax/manifest.json"))


def _mapping_rows(document: dict[str, object], key: str) -> list[dict[str, object]]:
    values = document[key]
    assert isinstance(values, list)
    rows: list[dict[str, object]] = []
    for value in values:
        assert isinstance(value, dict)
        rows.append(value)
    return rows


def test_official_catalog_is_pinned() -> None:
    manifest = _document("examples/jax/manifest.json")
    official_records = manifest["source_catalog"]
    assert isinstance(official_records, list)
    assert tuple(record["source"] for record in official_records) == (
        OFFICIAL_NATIVE_EXAMPLE_SOURCES
    )
    assert OFFICIAL_UPSTREAM_COMMIT == "9e027eac38028d57aa23777be52a781aa860e347"


def test_loader_rejects_a_catalog_that_differs_from_the_pinned_inventory() -> None:
    manifest = _mutated_active_manifest()
    manifest["source_catalog"] = [
        row
        for row in _mapping_rows(manifest, "source_catalog")
        if row["source"] != OFFICIAL_SOURCE
    ]
    with pytest.raises(ManifestV3ValidationError, match="pinned official upstream"):
        parse_examples_v3_document(manifest, repo_root=REPO_ROOT)


def test_loader_rejects_a_local_source_registry() -> None:
    manifest = _mutated_active_manifest()
    manifest["experimental_sources"] = []
    with pytest.raises(ManifestV3ValidationError, match="invalid manifest root fields"):
        parse_examples_v3_document(manifest, repo_root=REPO_ROOT)
