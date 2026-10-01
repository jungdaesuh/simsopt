"""Pinned upstream scope stays independent of local example files."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import copy
import json
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import (
    ManifestV3ValidationError,
    load_manifest_contract_pair_documents,
)
from examples.jax.official_source_catalog import (
    OFFICIAL_NATIVE_EXAMPLE_SOURCES,
    OFFICIAL_UPSTREAM_COMMIT,
)
from examples.jax.parity._manifest import ParityManifestValidationError
from examples.jax.run_parity import _selected_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_EXECUTABLE_BATCH_SIZE = 25
#: An official source with a full parity case and no host outer policy.
OFFICIAL_SOURCE = "1_Simple/tracing_particle.py"


def _document(relative_path: str) -> dict[str, object]:
    value = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _mutated_active_documents() -> tuple[dict[str, object], dict[str, object]]:
    return (
        copy.deepcopy(_document("examples/jax/manifest.json")),
        copy.deepcopy(_document("examples/jax/parity_manifest.json")),
    )


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
    manifest, parity = _mutated_active_documents()
    manifest["source_catalog"] = [
        row
        for row in _mapping_rows(manifest, "source_catalog")
        if row["source"] != OFFICIAL_SOURCE
    ]
    with pytest.raises(ManifestV3ValidationError, match="pinned official upstream"):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)


def test_official_default_batch_is_every_executable_official_relationship() -> None:
    pair = load_manifest_contract_pair_documents(
        _document("examples/jax/manifest.json"),
        _document("examples/jax/parity_manifest.json"),
        repo_root=REPO_ROOT,
    )
    official_case_ids = tuple(
        relationship.case_id
        for relationship in pair.parity.relationships
        if relationship.case_id is not None
    )

    assert len(official_case_ids) == OFFICIAL_EXECUTABLE_BATCH_SIZE
    assert _selected_cases(["all-applicable"], official_case_ids) == official_case_ids


def test_loader_rejects_an_official_relationship_omission() -> None:
    manifest, parity = _mutated_active_documents()
    parity["relationships"] = [
        row
        for row in _mapping_rows(parity, "relationships")
        if row.get("native_source") != OFFICIAL_SOURCE
    ]
    with pytest.raises(
        ManifestV3ValidationError,
        match=(
            "official parity relationships must exactly follow one-to-one "
            "source ownership"
        ),
    ):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)


def test_loader_rejects_a_local_source_registry() -> None:
    manifest, parity = _mutated_active_documents()
    manifest["experimental_sources"] = []
    with pytest.raises(ManifestV3ValidationError, match="invalid manifest root fields"):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)


def test_loader_rejects_a_local_relationship_registry() -> None:
    manifest, parity = _mutated_active_documents()
    parity["experimental_relationships"] = []
    with pytest.raises(
        ParityManifestValidationError, match="invalid parity manifest root fields"
    ):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)
