"""Audit the static BoozerSurface legacy-to-JAX parity matrix."""

from __future__ import annotations

import ast
from pathlib import Path

from benchmarks.validation_ladder_contract import PARITY_LADDER_TOLERANCES
from .boozer_legacy_parity_contract import (
    BOOZER_LEGACY_PARITY_BY_TEST,
    BOOZER_LEGACY_PARITY_CONTRACT,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_BOOZER_TEST_FILE = _REPO_ROOT / "tests" / "geo" / "test_boozersurface.py"
_VALID_CATEGORIES = {
    "strict_parity",
    "jax_native_equivalent",
    "intentional_exclusion",
    "unsupported_jax_contract",
}


def _legacy_boozer_test_names() -> set[str]:
    module = ast.parse(_LEGACY_BOOZER_TEST_FILE.read_text())
    return {
        node.name
        for class_node in module.body
        if isinstance(class_node, ast.ClassDef)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


def test_boozer_legacy_parity_contract_covers_every_legacy_test():
    assert set(BOOZER_LEGACY_PARITY_BY_TEST) == _legacy_boozer_test_names()


def test_boozer_legacy_parity_contract_entries_are_complete():
    assert len(BOOZER_LEGACY_PARITY_BY_TEST) == len(BOOZER_LEGACY_PARITY_CONTRACT)
    for entry in BOOZER_LEGACY_PARITY_CONTRACT:
        assert entry.category in _VALID_CATEGORIES
        assert entry.owner_file
        assert entry.owner_test
        assert "planned::" not in entry.owner_test
        assert entry.notes
        if entry.tolerance_lane is not None:
            assert entry.tolerance_lane in PARITY_LADDER_TOLERANCES
