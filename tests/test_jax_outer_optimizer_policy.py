"""Approved host outer policies are explicit, owner-bound manifest contracts."""

from __future__ import annotations

import json
import dataclasses
from pathlib import Path
from unittest.mock import Mock

import pytest
from examples.jax import run_parity
from examples.jax.manifest_contracts_v3 import load_manifest_contract_pair_documents
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.run_examples import build_child_command

REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_ID = "native-single-stage-boozer-vacuum-optimization"
SERIAL_ID = "native-boozerqa-ls"


def test_approved_policies_are_available_through_the_real_runtime_registry() -> None:
    pair = load_runtime_contract_pair(
        REPO_ROOT / "examples/jax/manifest.json",
        REPO_ROOT / "examples/jax/parity_manifest.json",
        repo_root=REPO_ROOT,
    )
    exact, serial = (
        next(example for example in pair.examples if example.id == identity)
        for identity in (EXACT_ID, SERIAL_ID)
    )
    assert exact.status == serial.status == "ready"
    assert exact.outer_optimizer_policy is not None
    assert serial.outer_optimizer_policy is not None
    assert exact.outer_optimizer_policy.case_id == EXACT_ID
    assert serial.outer_optimizer_policy.case_id is None
    assert serial.teaching_kind == "combined"
    assert not any(
        relationship.jax_example_id == SERIAL_ID
        for relationship in pair.parity.relationships
    )
    assert (
        next(
            example
            for example in pair.examples
            if example.id == "native-just-a-quadratic"
        ).outer_optimizer_policy
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing_exact", "missing_serial", "copied", "swapped", "unknown", "wrong_case"),
)
def test_manifest_rejects_missing_or_borrowed_host_outer_declarations(
    mutation: str,
) -> None:
    manifest = json.loads((REPO_ROOT / "examples/jax/manifest.json").read_text())
    parity = json.loads((REPO_ROOT / "examples/jax/parity_manifest.json").read_text())
    examples = {example["id"]: example for example in manifest["jax_examples"]}
    if mutation == "missing_exact":
        del examples[EXACT_ID]["outer_optimizer_policy"]
    elif mutation == "missing_serial":
        del examples[SERIAL_ID]["outer_optimizer_policy"]
    elif mutation == "copied":
        examples["native-just-a-quadratic"]["outer_optimizer_policy"] = examples[
            EXACT_ID
        ]["outer_optimizer_policy"]
    elif mutation == "swapped":
        examples[SERIAL_ID]["outer_optimizer_policy"] = examples[EXACT_ID][
            "outer_optimizer_policy"
        ]
    elif mutation == "unknown":
        examples[EXACT_ID]["outer_optimizer_policy"] = "allow-all-scipy"
    else:
        relationship = next(
            relationship
            for relationship in parity["relationships"]
            if relationship["jax_example_id"] == EXACT_ID
        )
        relationship["case_id"] = "unregistered-borrower"
    with pytest.raises(ValueError, match="outer optimizer policy"):
        load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)


def test_planned_serial_record_without_declaration_keeps_legacy_default() -> None:
    manifest = json.loads((REPO_ROOT / "examples/jax/manifest.json").read_text())
    parity = json.loads((REPO_ROOT / "examples/jax/parity_manifest.json").read_text())
    serial = next(
        example for example in manifest["jax_examples"] if example["id"] == SERIAL_ID
    )
    serial["status"] = "planned"
    del serial["outer_optimizer_policy"]
    pair = load_manifest_contract_pair_documents(manifest, parity, repo_root=REPO_ROOT)
    assert (
        next(
            example for example in pair.examples.jax_examples if example.id == SERIAL_ID
        ).outer_optimizer_policy
        is None
    )


@pytest.mark.parametrize("example_id", (EXACT_ID, SERIAL_ID))
def test_example_child_command_rejects_a_missing_ready_host_declaration(
    example_id: str,
) -> None:
    pair = load_runtime_contract_pair(
        REPO_ROOT / "examples/jax/manifest.json",
        REPO_ROOT / "examples/jax/parity_manifest.json",
        repo_root=REPO_ROOT,
    )
    example = next(example for example in pair.examples if example.id == example_id)
    assert build_child_command(example, repo_root=REPO_ROOT)
    without_declaration = dataclasses.replace(example, outer_optimizer_policy=None)
    with pytest.raises(
        ValueError, match="requires its outer optimizer policy declaration"
    ):
        build_child_command(without_declaration, repo_root=REPO_ROOT)


def test_parity_runner_rejects_legacy_missing_policy_before_inputs_or_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = load_runtime_contract_pair(
        REPO_ROOT / "examples/jax/manifest.json",
        REPO_ROOT / "examples/jax/parity_manifest.json",
        repo_root=REPO_ROOT,
    )
    legacy_pair = dataclasses.replace(
        pair,
        version_pair=(2, 1),
        used_legacy_adapter=True,
        examples=tuple(
            dataclasses.replace(example, outer_optimizer_policy=None)
            for example in pair.examples
        ),
    )
    monkeypatch.setattr(
        run_parity, "load_runtime_contract_pair", lambda *args, **kwargs: legacy_pair
    )
    case = Mock()
    case.create_input.side_effect = AssertionError("input creation must not begin")
    launch = Mock(side_effect=AssertionError("child execution must not begin"))
    monkeypatch.setattr(run_parity, "get_case", lambda case_id: case)
    monkeypatch.setattr(run_parity, "execute_case_lanes", launch)
    assert (
        run_parity.main(
            [
                "--case",
                EXACT_ID,
                "--lanes",
                "jax-cpu",
                "--scale",
                "native_default",
                "--artifact-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    case.create_input.assert_not_called()
    launch.assert_not_called()
    failed_receipts = tuple(tmp_path.rglob("FAILURE.json"))
    assert failed_receipts, "Rejected runs must preserve a failure receipt"
    assert (
        "requires its outer optimizer policy declaration"
        in failed_receipts[0].read_text()
    )
