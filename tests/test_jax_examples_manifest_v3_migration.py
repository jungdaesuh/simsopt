"""Fail-closed migration contract for example schema v3 and parity schema v2."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import (
    ContractVersionError,
    ManifestV3ValidationError,
    build_v3_candidates,
    load_manifest_contract_pair_documents,
    parse_examples_v3_document,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_MANIFEST = REPO_ROOT / "examples" / "jax" / "manifest.json"
PARITY_MANIFEST = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"
INVENTORY = REPO_ROOT / "examples" / "jax" / "one_to_one_inventory.json"
EXAMPLES_V2_SHA256 = "2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05"
PARITY_V1_SHA256 = "060e55339194c203263da9d5690c2ff31bd6681f5713dc2ead0ce3313e313137"


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _candidates() -> tuple[dict[str, object], dict[str, object]]:
    candidate = build_v3_candidates(
        examples_v2_document=_document(EXAMPLES_MANIFEST),
        parity_v1_document=_document(PARITY_MANIFEST),
        inventory_document=_document(INVENTORY),
        repo_root=REPO_ROOT,
    )
    examples = json.loads(candidate.examples_bytes)
    parity = json.loads(candidate.parity_bytes)
    assert isinstance(examples, dict)
    assert isinstance(parity, dict)
    return examples, parity


def _records(document: dict[str, object], key: str) -> list[dict[str, object]]:
    values = document[key]
    assert isinstance(values, list)
    assert all(isinstance(value, dict) for value in values)
    return values


def _record_by(
    document: dict[str, object], collection: str, key: str, value: str
) -> dict[str, object]:
    matches = [
        record for record in _records(document, collection) if record.get(key) == value
    ]
    assert len(matches) == 1
    return matches[0]


def test_exact_v2_v1_bytes_map_deterministically_without_writing() -> None:
    before_examples = EXAMPLES_MANIFEST.read_bytes()
    before_parity = PARITY_MANIFEST.read_bytes()
    assert hashlib.sha256(before_examples).hexdigest() == EXAMPLES_V2_SHA256
    assert hashlib.sha256(before_parity).hexdigest() == PARITY_V1_SHA256

    arguments = {
        "examples_v2_document": _document(EXAMPLES_MANIFEST),
        "parity_v1_document": _document(PARITY_MANIFEST),
        "inventory_document": _document(INVENTORY),
        "repo_root": REPO_ROOT,
    }
    first = build_v3_candidates(**arguments)
    second = build_v3_candidates(**arguments)
    assert first == second
    assert EXAMPLES_MANIFEST.read_bytes() == before_examples
    assert PARITY_MANIFEST.read_bytes() == before_parity

    examples = json.loads(first.examples_bytes)
    parity = json.loads(first.parity_bytes)
    assert isinstance(examples, dict) and examples["schema_version"] == 3
    assert isinstance(parity, dict) and parity["schema_version"] == 2
    assert len(_records(examples, "source_catalog")) == 51
    assert len(_records(examples, "jax_examples")) == 37
    assert len(_records(parity, "relationships")) == 26
    assert first.examples_sha256 == hashlib.sha256(first.examples_bytes).hexdigest()
    assert first.parity_sha256 == hashlib.sha256(first.parity_bytes).hexdigest()
    assert first.semantic_diff["legacy_tutorial_count"] == 11
    assert first.semantic_diff["planned_one_to_one_count"] == 26


def test_candidate_has_exact_name_mirrors_and_noncovering_legacy_tutorials() -> None:
    examples, parity = _candidates()
    sources = _records(examples, "source_catalog")
    executable = _records(examples, "jax_examples")
    relationships = _records(parity, "relationships")
    by_id = {str(record["id"]): record for record in executable}

    owned_ids: set[str] = set()
    for source in sources:
        disposition = source["disposition"]
        mirror_id = source["mirror_example_id"]
        if disposition in {"eligible", "hybrid"}:
            assert isinstance(mirror_id, str)
            mirror = by_id[mirror_id]
            assert mirror["path"] == source["source"]
            assert mirror["teaching_kind"] == "one_to_one"
            owned_ids.add(mirror_id)
        else:
            assert mirror_id is None

    assert len(owned_ids) == 26
    assert {str(record["jax_example_id"]) for record in relationships} == owned_ids
    tutorial_ids = {
        str(record["id"])
        for record in executable
        if record["classification"] == "tutorial"
    }
    assert len(tutorial_ids) == 11
    assert tutorial_ids.isdisjoint(owned_ids)


def test_schema_v3_rejects_missing_duplicate_tutorial_and_alias_ownership() -> None:
    examples, _ = _candidates()
    source = _record_by(
        examples, "source_catalog", "source", "1_Simple/just_a_quadratic.py"
    )
    mirror_id = source["mirror_example_id"]
    assert isinstance(mirror_id, str)

    missing = copy.deepcopy(examples)
    missing_source = _record_by(
        missing, "source_catalog", "source", "1_Simple/just_a_quadratic.py"
    )
    missing_source["mirror_example_id"] = None
    with pytest.raises(ManifestV3ValidationError, match="eligible source requires"):
        parse_examples_v3_document(missing, repo_root=REPO_ROOT)

    duplicate = copy.deepcopy(examples)
    duplicate_source = _record_by(
        duplicate, "source_catalog", "source", "1_Simple/minimize_curve_length.py"
    )
    duplicate_source["mirror_example_id"] = mirror_id
    with pytest.raises(ManifestV3ValidationError, match="duplicate mirror ownership"):
        parse_examples_v3_document(duplicate, repo_root=REPO_ROOT)

    tutorial = copy.deepcopy(examples)
    tutorial_source = _record_by(
        tutorial, "source_catalog", "source", "1_Simple/just_a_quadratic.py"
    )
    tutorial_source["mirror_example_id"] = "traceable-least-squares"
    with pytest.raises(ManifestV3ValidationError, match="tutorial cannot own coverage"):
        parse_examples_v3_document(tutorial, repo_root=REPO_ROOT)

    alias = copy.deepcopy(examples)
    alias_mirror = _record_by(alias, "jax_examples", "id", mirror_id)
    alias_mirror["path"] = "1_Simple/not_the_native_filename.py"
    with pytest.raises(ManifestV3ValidationError, match="exact-name mirror path"):
        parse_examples_v3_document(alias, repo_root=REPO_ROOT)


def test_schema_v3_rejects_hybrid_without_explicit_gpu_slice_scope() -> None:
    examples, _ = _candidates()
    hybrid = _record_by(
        examples, "jax_examples", "id", "native-single-stage-optimization"
    )
    scopes = hybrid["supported_device_scopes"]
    assert isinstance(scopes, dict)
    del scopes["gpu"]
    with pytest.raises(ManifestV3ValidationError, match="hybrid GPU scope"):
        parse_examples_v3_document(examples, repo_root=REPO_ROOT)


def test_contract_pair_accepts_only_complete_legacy_or_canonical_versions() -> None:
    examples_v2 = _document(EXAMPLES_MANIFEST)
    parity_v1 = _document(PARITY_MANIFEST)
    examples_v3, parity_v2 = _candidates()

    legacy = load_manifest_contract_pair_documents(
        examples_v2, parity_v1, repo_root=REPO_ROOT
    )
    canonical = load_manifest_contract_pair_documents(
        examples_v3, parity_v2, repo_root=REPO_ROOT
    )
    assert legacy.version_pair == (2, 1)
    assert legacy.used_legacy_adapter is True
    assert canonical.version_pair == (3, 2)
    assert canonical.used_legacy_adapter is False

    with pytest.raises(ContractVersionError, match="mixed manifest versions"):
        load_manifest_contract_pair_documents(
            examples_v3, parity_v1, repo_root=REPO_ROOT
        )
    with pytest.raises(ContractVersionError, match="mixed manifest versions"):
        load_manifest_contract_pair_documents(
            examples_v2, parity_v2, repo_root=REPO_ROOT
        )

    unknown_examples = copy.deepcopy(examples_v3)
    unknown_examples["schema_version"] = 4
    with pytest.raises(ContractVersionError, match="unsupported example schema"):
        load_manifest_contract_pair_documents(
            unknown_examples, parity_v2, repo_root=REPO_ROOT
        )
    unknown_parity = copy.deepcopy(parity_v2)
    unknown_parity["schema_version"] = 3
    with pytest.raises(ContractVersionError, match="unsupported parity schema"):
        load_manifest_contract_pair_documents(
            examples_v3, unknown_parity, repo_root=REPO_ROOT
        )


def test_partial_conversion_is_rejected_instead_of_guessed() -> None:
    examples_v3, _ = _candidates()
    partial = copy.deepcopy(examples_v3)
    first = _records(partial, "jax_examples")[0]
    first["inspired_by"] = ["1_Simple/just_a_quadratic.py"]
    with pytest.raises(ManifestV3ValidationError, match="unexpected executable fields"):
        parse_examples_v3_document(partial, repo_root=REPO_ROOT)
