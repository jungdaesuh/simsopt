"""Runtime adaptation contract across legacy and canonical manifest pairs."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from examples.jax._manifest import TIERS
from examples.jax.manifest_contracts_v3 import ContractVersionError
from examples.jax.manifest_runtime import (
    emit_compatibility_warning,
    load_runtime_contract_pair,
)
from examples.jax.parity.cases import implemented_case_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXAMPLES = REPO_ROOT / "examples" / "jax" / "manifest.json"
ACTIVE_PARITY = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"
LEGACY_PARITY = (
    REPO_ROOT / "tests" / "fixtures" / "jax_manifests" / "parity_manifest_v1.json"
)


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _records(document: dict[str, object], key: str) -> list[dict[str, object]]:
    values = document[key]
    assert isinstance(values, list)
    assert all(isinstance(value, dict) for value in values)
    return values


def _tracked_native_sources() -> set[str]:
    examples_root = REPO_ROOT / "examples"
    return {
        path.relative_to(examples_root).as_posix()
        for tier in TIERS
        for path in (examples_root / tier).glob("*.py")
    }


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
    tracked = _tracked_native_sources()
    assert runtime.version_pair == (3, 2)
    assert runtime.used_legacy_adapter is False
    assert {
        str(row["source"]) for row in _records(catalog, "source_catalog")
    } == tracked
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
    implemented_native_cases = {
        case_id for case_id in implemented_case_ids() if case_id.startswith("native-")
    }
    active_case_ids = {
        relationship.case_id
        for relationship in runtime.parity.relationships
        if relationship.case_id is not None
    }

    assert implemented_native_cases <= active_case_ids


def test_runtime_emits_bound_warning_only_for_compatibility_aliases() -> None:
    runtime = load_runtime_contract_pair(
        ACTIVE_EXAMPLES,
        ACTIVE_PARITY,
        repo_root=REPO_ROOT,
    )
    aliases = tuple(
        example for example in runtime.examples if example.compatibility is not None
    )
    catalog_aliases = [
        example
        for example in _records(_document(ACTIVE_EXAMPLES), "jax_examples")
        if example["teaching_kind"] == "compatibility"
    ]
    assert len(aliases) == len(catalog_aliases)
    for alias in aliases:
        metadata = alias.compatibility
        assert metadata is not None
        stream = StringIO()
        emitted = emit_compatibility_warning(alias, stream=stream)
        assert emitted is True
        assert stream.getvalue() == metadata.warning + "\n"
        assert metadata.removal_after == "one documented deprecation interval"

    combined = next(
        example for example in runtime.examples if example.teaching_kind == "combined"
    )
    stream = StringIO()
    assert emit_compatibility_warning(combined, stream=stream) is False
    assert stream.getvalue() == ""


def test_runtime_loader_rejects_mixed_contract_files() -> None:
    with pytest.raises(ContractVersionError, match="mixed manifest versions"):
        load_runtime_contract_pair(
            ACTIVE_EXAMPLES,
            LEGACY_PARITY,
            repo_root=REPO_ROOT,
        )
