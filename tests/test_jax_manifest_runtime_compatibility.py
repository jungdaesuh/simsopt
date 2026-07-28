"""Runtime adaptation contract across legacy and canonical manifest pairs."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import ContractVersionError
from examples.jax.manifest_runtime import (
    emit_compatibility_warning,
    load_runtime_contract_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXAMPLES = REPO_ROOT / "examples" / "jax" / "manifest.json"
ACTIVE_PARITY = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"
CANDIDATE_EXAMPLES = REPO_ROOT / "docs" / "jax_examples_manifest_v3_candidate.json"
CANDIDATE_PARITY = REPO_ROOT / "docs" / "jax_parity_manifest_v2_candidate.json"


def test_legacy_pair_remains_readable_without_inventing_v3_alias_metadata() -> None:
    runtime = load_runtime_contract_pair(
        ACTIVE_EXAMPLES,
        ACTIVE_PARITY,
        repo_root=REPO_ROOT,
    )
    assert runtime.version_pair == (2, 1)
    assert runtime.used_legacy_adapter is True
    assert len(runtime.examples) == 11
    assert sum(example.status == "ready" for example in runtime.examples) == 10
    assert all(example.compatibility is None for example in runtime.examples)


def test_canonical_pair_exposes_exact_mirrors_without_tutorial_coverage() -> None:
    runtime = load_runtime_contract_pair(
        CANDIDATE_EXAMPLES,
        CANDIDATE_PARITY,
        repo_root=REPO_ROOT,
    )
    assert runtime.version_pair == (3, 2)
    assert runtime.used_legacy_adapter is False
    assert len(runtime.examples) == 37
    assert sum(example.status == "ready" for example in runtime.examples) == 10

    one_to_one = tuple(
        example for example in runtime.examples if example.teaching_kind == "one_to_one"
    )
    assert len(one_to_one) == 27
    assert all(example.source == example.path for example in one_to_one)
    assert all(example.status == "planned" for example in one_to_one)

    tutorials = tuple(
        example for example in runtime.examples if example.classification == "tutorial"
    )
    assert len(tutorials) == 11
    assert all(example.source is None for example in tutorials)


def test_runtime_emits_bound_warning_only_for_compatibility_aliases() -> None:
    runtime = load_runtime_contract_pair(
        CANDIDATE_EXAMPLES,
        CANDIDATE_PARITY,
        repo_root=REPO_ROOT,
    )
    aliases = tuple(
        example for example in runtime.examples if example.compatibility is not None
    )
    assert len(aliases) == 5
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
            CANDIDATE_EXAMPLES,
            ACTIVE_PARITY,
            repo_root=REPO_ROOT,
        )
