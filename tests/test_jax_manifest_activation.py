"""Hash-locked activation and rollback contract for paired JAX manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from examples.jax.manifest_activation import (
    ActivationContract,
    ManifestActivationError,
    validate_activation_bundle,
    validate_rollback_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXAMPLES = REPO_ROOT / "examples" / "jax" / "manifest.json"
ACTIVE_PARITY = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"
CANDIDATE_EXAMPLES = REPO_ROOT / "docs" / "jax_examples_manifest_v3_candidate.json"
CANDIDATE_PARITY = REPO_ROOT / "docs" / "jax_parity_manifest_v2_candidate.json"
APPROVED_EXAMPLES_SHA = (
    "50292cdf3eda34a60d6709387f6f5042ad89a15fd70aa75fcd39f3064648209d"
)
APPROVED_PARITY_SHA = "b51c69d7d4e08d2d08ba121f930133198f340684911565d1d3871fdaa15d78fa"


def _valid_contract() -> ActivationContract:
    return validate_activation_bundle(
        active_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
        active_parity_bytes=ACTIVE_PARITY.read_bytes(),
        candidate_examples_bytes=CANDIDATE_EXAMPLES.read_bytes(),
        candidate_parity_bytes=CANDIDATE_PARITY.read_bytes(),
        approved_examples_sha256=APPROVED_EXAMPLES_SHA,
        approved_parity_sha256=APPROVED_PARITY_SHA,
        repo_root=REPO_ROOT,
    )


def test_activation_requires_exact_approved_candidate_pair() -> None:
    contract = _valid_contract()
    assert contract.before_version_pair == (2, 1)
    assert contract.after_version_pair == (3, 2)
    assert contract.before_sha256 == (
        "2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05",
        "060e55339194c203263da9d5690c2ff31bd6681f5713dc2ead0ce3313e313137",
    )
    assert contract.after_sha256 == (
        APPROVED_EXAMPLES_SHA,
        APPROVED_PARITY_SHA,
    )


@pytest.mark.parametrize(
    ("examples_approval", "parity_approval"),
    [
        ("0" * 64, APPROVED_PARITY_SHA),
        (APPROVED_EXAMPLES_SHA, "0" * 64),
    ],
)
def test_activation_rejects_unapproved_hashes(
    examples_approval: str, parity_approval: str
) -> None:
    with pytest.raises(ManifestActivationError, match="candidate approval mismatch"):
        validate_activation_bundle(
            active_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
            active_parity_bytes=ACTIVE_PARITY.read_bytes(),
            candidate_examples_bytes=CANDIDATE_EXAMPLES.read_bytes(),
            candidate_parity_bytes=CANDIDATE_PARITY.read_bytes(),
            approved_examples_sha256=examples_approval,
            approved_parity_sha256=parity_approval,
            repo_root=REPO_ROOT,
        )


def test_activation_rejects_byte_drift_even_when_json_semantics_survive() -> None:
    changed = CANDIDATE_EXAMPLES.read_bytes().replace(
        b'"schema_version":3', b'"schema_version": 3'
    )
    assert hashlib.sha256(changed).hexdigest() != APPROVED_EXAMPLES_SHA
    with pytest.raises(ManifestActivationError, match="candidate approval mismatch"):
        validate_activation_bundle(
            active_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
            active_parity_bytes=ACTIVE_PARITY.read_bytes(),
            candidate_examples_bytes=changed,
            candidate_parity_bytes=CANDIDATE_PARITY.read_bytes(),
            approved_examples_sha256=APPROVED_EXAMPLES_SHA,
            approved_parity_sha256=APPROVED_PARITY_SHA,
            repo_root=REPO_ROOT,
        )


def test_rollback_requires_exact_inverse_pair_and_revalidates_both_contracts() -> None:
    contract = _valid_contract()
    rollback = validate_rollback_bundle(
        activated_examples_bytes=CANDIDATE_EXAMPLES.read_bytes(),
        activated_parity_bytes=CANDIDATE_PARITY.read_bytes(),
        rollback_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
        rollback_parity_bytes=ACTIVE_PARITY.read_bytes(),
        activation=contract,
        repo_root=REPO_ROOT,
    )
    assert rollback.restored_version_pair == (2, 1)
    assert rollback.restored_sha256 == contract.before_sha256

    with pytest.raises(ManifestActivationError, match="rollback pair mismatch"):
        validate_rollback_bundle(
            activated_examples_bytes=CANDIDATE_EXAMPLES.read_bytes(),
            activated_parity_bytes=CANDIDATE_PARITY.read_bytes(),
            rollback_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
            rollback_parity_bytes=CANDIDATE_PARITY.read_bytes(),
            activation=contract,
            repo_root=REPO_ROOT,
        )
