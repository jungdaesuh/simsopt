"""Pinned upstream scope stays independent of local example files."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import examples.jax.outer_optimizer_policy as outer_optimizer_policy
import pytest
from examples.jax import manifest_contracts_v3
from examples.jax.manifest_contracts_v3 import (
    ManifestV3ValidationError,
    load_manifest_contract_pair_documents,
)
from examples.jax.official_source_catalog import (
    OFFICIAL_NATIVE_EXAMPLE_SOURCES,
    OFFICIAL_UPSTREAM_COMMIT,
)
from examples.jax.run_parity import _selected_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
# No branch-only source remains registered, so the experimental-registration
# rules are exercised by demoting one official source with a full parity case
# and no host outer policy: the pinned catalog loses it and the manifests carry
# its source row and relationship in the experimental groups.
EXPERIMENTAL_CASE = "native-tracing-particle"
EXPERIMENTAL_SOURCE = "1_Simple/tracing_particle.py"
MISSING_EXPERIMENTAL_SOURCE = "1_Simple/tracing_particle_missing.py"
OFFICIAL_EXECUTABLE_BATCH_SIZE = 25


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


@pytest.fixture
def demoted_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    """The active documents with ``EXPERIMENTAL_SOURCE`` registered as branch-only."""
    pinned = tuple(
        source
        for source in OFFICIAL_NATIVE_EXAMPLE_SOURCES
        if source != EXPERIMENTAL_SOURCE
    )
    monkeypatch.setattr(
        manifest_contracts_v3, "OFFICIAL_NATIVE_EXAMPLE_SOURCES", pinned
    )
    monkeypatch.setattr(
        manifest_contracts_v3, "OFFICIAL_NATIVE_EXAMPLE_SOURCE_SET", frozenset(pinned)
    )
    manifest, parity = _mutated_active_documents()
    sources = _mapping_rows(manifest, "source_catalog")
    manifest["source_catalog"] = [
        row for row in sources if row["source"] != EXPERIMENTAL_SOURCE
    ]
    manifest["experimental_sources"] = [
        row for row in sources if row["source"] == EXPERIMENTAL_SOURCE
    ]
    relationships = _mapping_rows(parity, "relationships")
    parity["relationships"] = [
        row for row in relationships if row["native_source"] != EXPERIMENTAL_SOURCE
    ]
    parity["experimental_relationships"] = [
        row for row in relationships if row["native_source"] == EXPERIMENTAL_SOURCE
    ]
    return manifest, parity


def test_official_catalog_is_pinned_and_excludes_local_extensions() -> None:
    manifest = _document("examples/jax/manifest.json")
    official_records = manifest["source_catalog"]
    experimental_records = manifest["experimental_sources"]
    assert isinstance(official_records, list)
    assert isinstance(experimental_records, list)
    assert tuple(record["source"] for record in official_records) == (
        OFFICIAL_NATIVE_EXAMPLE_SOURCES
    )
    assert experimental_records == []
    assert OFFICIAL_UPSTREAM_COMMIT == "9e027eac38028d57aa23777be52a781aa860e347"


def test_branch_only_local_sources_cannot_enter_official_catalog(
    demoted_documents: tuple[dict[str, object], dict[str, object]],
) -> None:
    manifest, parity = demoted_documents
    official_records = manifest["source_catalog"]
    experimental_records = manifest["experimental_sources"]
    official_relationships = parity["relationships"]
    experimental_relationships = parity["experimental_relationships"]
    assert isinstance(official_records, list)
    assert isinstance(experimental_records, list)
    assert isinstance(official_relationships, list)
    assert isinstance(experimental_relationships, list)
    manifest["source_catalog"] = sorted(
        [*official_records, *experimental_records], key=lambda row: row["source"]
    )
    manifest["experimental_sources"] = []
    parity["relationships"] = sorted(
        [*official_relationships, *experimental_relationships],
        key=lambda row: row["native_source"],
    )
    parity["experimental_relationships"] = []
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

    assert pair.parity.experimental_relationships == ()
    assert len(official_case_ids) == OFFICIAL_EXECUTABLE_BATCH_SIZE
    assert _selected_cases(["all-applicable"], official_case_ids) == official_case_ids


def test_experimental_cases_remain_explicit_but_leave_official_default_batch(
    demoted_documents: tuple[dict[str, object], dict[str, object]],
) -> None:
    manifest, parity = demoted_documents
    pair = load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)
    official_case_ids = tuple(
        relationship.case_id
        for relationship in pair.parity.relationships
        if relationship.case_id is not None
    )
    experimental_case_ids = tuple(
        relationship.case_id
        for relationship in pair.parity.experimental_relationships
        if relationship.case_id is not None
    )

    assert EXPERIMENTAL_CASE not in official_case_ids
    assert experimental_case_ids == (EXPERIMENTAL_CASE,)
    assert len(official_case_ids) == OFFICIAL_EXECUTABLE_BATCH_SIZE - 1
    assert _selected_cases(["all-applicable"], official_case_ids) == official_case_ids
    assert _selected_cases([EXPERIMENTAL_CASE], official_case_ids) == (
        EXPERIMENTAL_CASE,
    )


@pytest.mark.parametrize(
    ("origin", "destination", "case_id"),
    (
        ("relationships", "experimental_relationships", "native-qfm"),
        ("experimental_relationships", "relationships", EXPERIMENTAL_CASE),
    ),
    ids=("official-to-experimental", "experimental-to-official"),
)
def test_loader_rejects_relationship_moved_between_official_and_experimental_groups(
    demoted_documents: tuple[dict[str, object], dict[str, object]],
    origin: str,
    destination: str,
    case_id: str,
) -> None:
    manifest, parity = demoted_documents
    origin_rows = _mapping_rows(parity, origin)
    moved = next(row for row in origin_rows if row.get("case_id") == case_id)
    parity[origin] = [row for row in origin_rows if row is not moved]
    destination_rows = _mapping_rows(parity, destination)
    parity[destination] = [*destination_rows, moved]
    with pytest.raises(
        ManifestV3ValidationError,
        match=(
            "official parity relationships must exactly follow one-to-one "
            "source ownership"
        ),
    ):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)


def test_loader_rejects_experimental_relationship_omission(
    demoted_documents: tuple[dict[str, object], dict[str, object]],
) -> None:
    manifest, parity = demoted_documents
    official_relationships = copy.deepcopy(parity["relationships"])
    experimental_rows = _mapping_rows(parity, "experimental_relationships")
    parity["experimental_relationships"] = [
        row
        for row in experimental_rows
        if row.get("native_source") != EXPERIMENTAL_SOURCE
    ]
    assert parity["relationships"] == official_relationships
    assert len(parity["experimental_relationships"]) == len(experimental_rows) - 1
    with pytest.raises(
        ManifestV3ValidationError,
        match=(
            "experimental parity relationships must exactly follow one-to-one "
            "source ownership"
        ),
    ):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)


def test_loader_rejects_missing_experimental_source_file(
    demoted_documents: tuple[dict[str, object], dict[str, object]],
) -> None:
    manifest, parity = demoted_documents
    sources = _mapping_rows(manifest, "experimental_sources")
    source = next(row for row in sources if row["source"] == EXPERIMENTAL_SOURCE)
    source["source"] = MISSING_EXPERIMENTAL_SOURCE
    manifest["experimental_sources"] = sorted(
        sources, key=lambda row: str(row["source"])
    )
    with pytest.raises(
        ManifestV3ValidationError,
        match=re.escape(
            "experimental source registrations do not exist locally: "
            f"['{MISSING_EXPERIMENTAL_SOURCE}']"
        ),
    ):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)


def test_loader_rejects_policy_scope_mismatch_with_source_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # registry_scope is not a JSON field; flip the resolved official QFM policy
    # so the public loader's first error is the source-registration mismatch.
    flipped = tuple(
        replace(policy, registry_scope="experimental")
        if policy.example_id == "native-qfm"
        else policy
        for policy in outer_optimizer_policy._APPROVED_POLICIES
    )
    monkeypatch.setattr(outer_optimizer_policy, "_APPROVED_POLICIES", flipped)
    manifest, parity = _mutated_active_documents()
    with pytest.raises(
        ManifestV3ValidationError,
        match=(
            "outer optimizer policy scope does not match source registration: "
            "native-qfm"
        ),
    ):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)
