"""Hash-locked activation and rollback contract for paired JAX manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from examples.jax.manifest_activation import (
    ActivationContract,
    ManifestActivationError,
    validate_activation_bundle,
    validate_rollback_bundle,
)
from examples.jax.manifest_contracts_v3 import build_v3_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXAMPLES = (
    REPO_ROOT / "tests" / "fixtures" / "jax_manifests" / "manifest_v2.json"
)
ACTIVE_PARITY = (
    REPO_ROOT / "tests" / "fixtures" / "jax_manifests" / "parity_manifest_v1.json"
)
INVENTORY = REPO_ROOT / "examples" / "jax" / "one_to_one_inventory.json"


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _generated_candidate_bytes() -> tuple[bytes, bytes]:
    candidate = build_v3_candidates(
        examples_v2_document=_document(ACTIVE_EXAMPLES),
        parity_v1_document=_document(ACTIVE_PARITY),
        inventory_document=_document(INVENTORY),
        repo_root=REPO_ROOT,
    )
    return candidate.examples_bytes, candidate.parity_bytes


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    ("examples_approval", "parity_approval"),
    [
        ("0" * 64, None),
        (None, "0" * 64),
    ],
)
def test_activation_rejects_unapproved_hashes(
    examples_approval: str | None, parity_approval: str | None
) -> None:
    examples_bytes, parity_bytes = _generated_candidate_bytes()
    approved_examples = (
        examples_approval if examples_approval is not None else _sha256(examples_bytes)
    )
    approved_parity = (
        parity_approval if parity_approval is not None else _sha256(parity_bytes)
    )
    with pytest.raises(ManifestActivationError, match="candidate approval mismatch"):
        validate_activation_bundle(
            active_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
            active_parity_bytes=ACTIVE_PARITY.read_bytes(),
            candidate_examples_bytes=examples_bytes,
            candidate_parity_bytes=parity_bytes,
            approved_examples_sha256=approved_examples,
            approved_parity_sha256=approved_parity,
            repo_root=REPO_ROOT,
        )


def test_activation_rejects_byte_drift_even_when_json_semantics_survive() -> None:
    examples_bytes, parity_bytes = _generated_candidate_bytes()
    changed = examples_bytes.replace(b'"schema_version":3', b'"schema_version": 3')
    assert _sha256(changed) != _sha256(examples_bytes)
    with pytest.raises(ManifestActivationError, match="candidate approval mismatch"):
        validate_activation_bundle(
            active_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
            active_parity_bytes=ACTIVE_PARITY.read_bytes(),
            candidate_examples_bytes=changed,
            candidate_parity_bytes=parity_bytes,
            approved_examples_sha256=_sha256(examples_bytes),
            approved_parity_sha256=_sha256(parity_bytes),
            repo_root=REPO_ROOT,
        )


def test_rollback_requires_exact_inverse_pair_and_revalidates_both_contracts() -> None:
    examples_bytes, parity_bytes = _generated_candidate_bytes()
    contract = ActivationContract(
        before_version_pair=(2, 1),
        after_version_pair=(3, 2),
        before_sha256=(
            _sha256(ACTIVE_EXAMPLES.read_bytes()),
            _sha256(ACTIVE_PARITY.read_bytes()),
        ),
        after_sha256=(_sha256(examples_bytes), _sha256(parity_bytes)),
    )
    rollback = validate_rollback_bundle(
        activated_examples_bytes=examples_bytes,
        activated_parity_bytes=parity_bytes,
        rollback_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
        rollback_parity_bytes=ACTIVE_PARITY.read_bytes(),
        activation=contract,
        repo_root=REPO_ROOT,
    )
    assert rollback.restored_version_pair == (2, 1)
    assert rollback.restored_sha256 == contract.before_sha256

    with pytest.raises(ManifestActivationError, match="rollback pair mismatch"):
        validate_rollback_bundle(
            activated_examples_bytes=examples_bytes,
            activated_parity_bytes=parity_bytes,
            rollback_examples_bytes=ACTIVE_EXAMPLES.read_bytes(),
            rollback_parity_bytes=parity_bytes,
            activation=contract,
            repo_root=REPO_ROOT,
        )
