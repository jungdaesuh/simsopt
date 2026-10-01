"""Runtime adaptation contract for the canonical example/parity manifest pair."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import copy
import json
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import (
    ContractVersionError,
    load_manifest_contract_pair_documents,
)
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.official_source_catalog import OFFICIAL_NATIVE_EXAMPLE_SOURCES
from examples.jax.parity._manifest import ParityManifest, ParityRelationship
from examples.jax.parity.cases import implemented_case_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXAMPLES = REPO_ROOT / "examples" / "jax" / "manifest.json"
ACTIVE_PARITY = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"
OFFICIAL_EXECUTABLE_BATCH_SIZE = 25
QFM_CASE_ID = "native-qfm"


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _records(document: dict[str, object], key: str) -> list[dict[str, object]]:
    values = document[key]
    assert isinstance(values, list)
    assert all(isinstance(value, dict) for value in values)
    return values


def _executable_case_ids(
    relationships: tuple[ParityRelationship, ...],
) -> set[str]:
    return {
        relationship.case_id
        for relationship in relationships
        if relationship.case_id is not None
    }


def _assert_native_case_coverage(parity: ParityManifest) -> None:
    """Implemented native-* cases stay registered; combined cases do not."""
    implemented_native_cases = {
        case_id for case_id in implemented_case_ids() if case_id.startswith("native-")
    }
    official_case_ids = _executable_case_ids(parity.relationships)
    omitted = sorted(implemented_native_cases - official_case_ids)
    assert not omitted, f"Implemented native cases omitted: {omitted}"
    assert official_case_ids <= implemented_native_cases
    assert len(official_case_ids) == OFFICIAL_EXECUTABLE_BATCH_SIZE, (
        "official executable batch must remain "
        f"{OFFICIAL_EXECUTABLE_BATCH_SIZE}, got {len(official_case_ids)}"
    )
    combined_cases = set(implemented_case_ids()) - implemented_native_cases
    assert not combined_cases & official_case_ids


def _omit_qfm_executable_relationship(parity: dict[str, object]) -> None:
    relationship = next(
        record
        for record in _records(parity, "relationships")
        if record.get("case_id") == QFM_CASE_ID
    )
    relationship["classification"] = "unsupported"
    relationship["case_id"] = None
    relationship["blocker"] = "in-memory coverage probe omits native-qfm"
    relationship["comparison_routes"] = []
    relationship["workflow_stages"] = []
    relationship["omitted_scientific_stages"] = ["complete_native_workflow"]
    relationship["scale_tier"] = "not_applicable"
    relationship["cost_tier"] = "not_applicable"
    relationship.pop("scale_contracts", None)


def test_active_pair_is_the_canonical_exact_mirror_contract() -> None:
    runtime = load_runtime_contract_pair(
        ACTIVE_EXAMPLES,
        ACTIVE_PARITY,
        repo_root=REPO_ROOT,
    )
    catalog = _document(ACTIVE_EXAMPLES)
    jax_examples = _records(catalog, "jax_examples")
    one_to_one_doc = [
        example for example in jax_examples if example["teaching_kind"] == "one_to_one"
    ]
    assert runtime.version_pair == (3, 2)
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


def test_active_external_solver_free_mirrors_are_executable_parity_cases() -> None:
    runtime = load_runtime_contract_pair(
        ACTIVE_EXAMPLES,
        ACTIVE_PARITY,
        repo_root=REPO_ROOT,
    )
    _assert_native_case_coverage(runtime.parity)


def test_in_memory_qfm_omission_cannot_evade_native_case_coverage() -> None:
    examples = copy.deepcopy(_document(ACTIVE_EXAMPLES))
    parity = copy.deepcopy(_document(ACTIVE_PARITY))
    _omit_qfm_executable_relationship(parity)
    pair = load_manifest_contract_pair_documents(examples, parity, repo_root=REPO_ROOT)
    official_case_ids = _executable_case_ids(pair.parity.relationships)
    assert QFM_CASE_ID not in official_case_ids
    with pytest.raises(
        AssertionError, match=r"Implemented native cases omitted: \['native-qfm'\]"
    ):
        _assert_native_case_coverage(pair.parity)


def test_runtime_loader_rejects_a_retired_parity_schema(tmp_path: Path) -> None:
    parity = _document(ACTIVE_PARITY)
    parity["schema_version"] = 1
    retired_parity = tmp_path / "parity_manifest.json"
    retired_parity.write_text(json.dumps(parity), encoding="utf-8")

    with pytest.raises(ContractVersionError, match="unsupported parity schema"):
        load_runtime_contract_pair(
            ACTIVE_EXAMPLES,
            retired_parity,
            repo_root=REPO_ROOT,
        )
