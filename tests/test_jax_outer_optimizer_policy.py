"""Approved host outer policies are explicit, owner-bound manifest contracts."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import dataclasses
import json
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import parse_examples_v3_document
from examples.jax.manifest_runtime import load_runtime_manifest
from examples.jax.outer_optimizer_policy import (
    OuterOptimizerPolicyError,
    parse_outer_optimizer_policy,
)
from examples.jax.run_examples import build_child_command

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_POLICY_ID = "native-qfm"


def test_approved_policies_are_available_through_the_real_runtime_registry() -> None:
    pair = load_runtime_manifest(
        REPO_ROOT / "examples/jax/manifest.json", repo_root=REPO_ROOT
    )
    host = next(example for example in pair.examples if example.id == HOST_POLICY_ID)
    assert host.status == "ready"
    assert host.outer_optimizer_policy is not None
    assert host.outer_optimizer_policy.example_id == HOST_POLICY_ID
    assert (
        next(
            example
            for example in pair.examples
            if example.id == "native-wireframe-rcls-basic"
        ).outer_optimizer_policy
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "copied", "unknown"),
)
def test_manifest_rejects_missing_or_borrowed_host_outer_declarations(
    mutation: str,
) -> None:
    manifest = json.loads((REPO_ROOT / "examples/jax/manifest.json").read_text())
    examples = {example["id"]: example for example in manifest["jax_examples"]}
    if mutation == "missing":
        del examples[HOST_POLICY_ID]["outer_optimizer_policy"]
    elif mutation == "copied":
        examples["native-just-a-quadratic"]["outer_optimizer_policy"] = examples[
            HOST_POLICY_ID
        ]["outer_optimizer_policy"]
    else:
        examples[HOST_POLICY_ID]["outer_optimizer_policy"] = "allow-all-scipy"
    with pytest.raises(ValueError, match="outer optimizer policy"):
        parse_examples_v3_document(manifest, repo_root=REPO_ROOT)


def test_planned_record_without_declaration_keeps_legacy_default() -> None:
    """Only a ready record must declare its approved host outer policy.

    Every approved policy now belongs to a source-owning record, whose readiness
    the manifest binds to its source, so the exemption is exercised at the
    policy parser rather than through a planned manifest record.
    """
    identity = {"example_id": HOST_POLICY_ID, "example_path": "1_Simple/qfm.py"}
    assert parse_outer_optimizer_policy(None, **identity, ready=False) is None
    with pytest.raises(
        OuterOptimizerPolicyError,
        match="requires its outer optimizer policy declaration",
    ):
        parse_outer_optimizer_policy(None, **identity, ready=True)


@pytest.mark.parametrize(
    "example_id",
    (
        "native-qfm",
        "native-just-a-quadratic",
        "native-minimize-curve-length",
        "native-surf-vol-area",
    ),
)
def test_example_child_command_rejects_a_missing_ready_host_declaration(
    example_id: str,
) -> None:
    pair = load_runtime_manifest(
        REPO_ROOT / "examples/jax/manifest.json", repo_root=REPO_ROOT
    )
    example = next(example for example in pair.examples if example.id == example_id)
    assert build_child_command(example, repo_root=REPO_ROOT)
    without_declaration = dataclasses.replace(example, outer_optimizer_policy=None)
    with pytest.raises(
        ValueError, match="requires its outer optimizer policy declaration"
    ):
        build_child_command(without_declaration, repo_root=REPO_ROOT)


def test_qfm_host_policy_is_bound_to_its_official_example() -> None:
    manifest = json.loads((REPO_ROOT / "examples/jax/manifest.json").read_text())
    parsed = parse_examples_v3_document(manifest, repo_root=REPO_ROOT)
    qfm = next(example for example in parsed.jax_examples if example.id == "native-qfm")
    assert qfm.outer_optimizer_policy is not None
    assert (
        qfm.outer_optimizer_policy.expected_driver == "scipy_lbfgsb_slsqp_qfm_sequence"
    )
    assert qfm.outer_optimizer_policy.example_path == "1_Simple/qfm.py"

    borrower = next(
        example
        for example in manifest["jax_examples"]
        if example["id"] == "native-just-a-quadratic"
    )
    borrower["outer_optimizer_policy"] = qfm.outer_optimizer_policy.policy_id
    with pytest.raises(ValueError, match="different example"):
        parse_examples_v3_document(manifest, repo_root=REPO_ROOT)
