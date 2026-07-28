"""Parity contract for the VMEC-free Boozer single-stage mirror."""

from __future__ import annotations

from pathlib import Path

from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.cases import get_case


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "native-single-stage-boozer-vacuum-optimization"


def test_single_stage_boozer_vacuum_is_an_executable_parity_case() -> None:
    case = get_case(CASE_ID)
    runtime = load_runtime_contract_pair(
        REPO_ROOT / "examples" / "jax" / "manifest.json",
        REPO_ROOT / "examples" / "jax" / "parity_manifest.json",
        repo_root=REPO_ROOT,
    )
    relationship = next(
        item
        for item in runtime.parity.relationships
        if item.jax_example_id == CASE_ID
    )

    assert case.case_id == CASE_ID
    assert relationship.case_id == CASE_ID
    assert relationship.classification == "full"
    assert relationship.blocker is None
    assert not relationship.omitted_scientific_stages
    assert relationship.workflow_stages
    assert relationship.comparison_routes
