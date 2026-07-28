"""Validate an exact paired manifest activation and its inverse rollback."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from examples.jax.manifest_contracts_v3 import (
    JaxExamplesManifestV3,
    ManifestContractPair,
    load_manifest_contract_pair_documents,
)


class ManifestActivationError(ValueError):
    """A manifest activation or rollback is not the exact approved pair."""


@dataclass(frozen=True)
class ActivationContract:
    """Hash-bound before/after identity for one atomic activation commit."""

    before_version_pair: tuple[int, int]
    after_version_pair: tuple[int, int]
    before_sha256: tuple[str, str]
    after_sha256: tuple[str, str]


@dataclass(frozen=True)
class RollbackContract:
    """Verified inverse transition back to the complete legacy pair."""

    restored_version_pair: tuple[int, int]
    restored_sha256: tuple[str, str]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _document(data: bytes, context: str) -> dict[str, object]:
    value = json.loads(data)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestActivationError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _pair(
    examples_bytes: bytes,
    parity_bytes: bytes,
    *,
    repo_root: Path,
    context: str,
) -> ManifestContractPair:
    examples_document = _document(examples_bytes, f"{context} examples manifest")
    parity_document = _document(parity_bytes, f"{context} parity manifest")
    return load_manifest_contract_pair_documents(
        examples_document,
        parity_document,
        repo_root=repo_root,
    )


def _hash_pair(examples_bytes: bytes, parity_bytes: bytes) -> tuple[str, str]:
    return _sha256(examples_bytes), _sha256(parity_bytes)


def _require_sha256(value: str, context: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ManifestActivationError(f"{context} must be a lowercase SHA-256")
    return value


def _validate_candidate_policy(pair: ManifestContractPair) -> None:
    if not isinstance(pair.examples, JaxExamplesManifestV3):
        raise ManifestActivationError("activation candidate must use example schema v3")
    if any(
        relationship.classification != "unsupported"
        for relationship in pair.parity.relationships
    ):
        raise ManifestActivationError(
            "schema activation must not promote scientific parity claims"
        )
    if len(pair.examples.source_catalog) != 52:
        raise ManifestActivationError("schema activation must retain all 52 sources")


def _require_version_pair(
    pair: ManifestContractPair,
    expected: tuple[int, int],
    context: str,
) -> None:
    if pair.version_pair != expected:
        raise ManifestActivationError(
            f"{context} requires version pair {expected}, got {pair.version_pair}"
        )


def validate_activation_bundle(
    *,
    active_examples_bytes: bytes,
    active_parity_bytes: bytes,
    candidate_examples_bytes: bytes,
    candidate_parity_bytes: bytes,
    approved_examples_sha256: str,
    approved_parity_sha256: str,
    repo_root: Path,
) -> ActivationContract:
    """Validate exact approval, complete version pairs, and no claim promotion."""
    approved = (
        _require_sha256(approved_examples_sha256, "examples approval"),
        _require_sha256(approved_parity_sha256, "parity approval"),
    )
    after_hashes = _hash_pair(candidate_examples_bytes, candidate_parity_bytes)
    if after_hashes != approved:
        raise ManifestActivationError(
            f"candidate approval mismatch: approved={approved}, actual={after_hashes}"
        )
    before_pair = _pair(
        active_examples_bytes,
        active_parity_bytes,
        repo_root=repo_root,
        context="active",
    )
    after_pair = _pair(
        candidate_examples_bytes,
        candidate_parity_bytes,
        repo_root=repo_root,
        context="candidate",
    )
    _require_version_pair(before_pair, (2, 1), "activation input")
    _require_version_pair(after_pair, (3, 2), "activation output")
    _validate_candidate_policy(after_pair)
    return ActivationContract(
        before_version_pair=before_pair.version_pair,
        after_version_pair=after_pair.version_pair,
        before_sha256=_hash_pair(active_examples_bytes, active_parity_bytes),
        after_sha256=after_hashes,
    )


def validate_rollback_bundle(
    *,
    activated_examples_bytes: bytes,
    activated_parity_bytes: bytes,
    rollback_examples_bytes: bytes,
    rollback_parity_bytes: bytes,
    activation: ActivationContract,
    repo_root: Path,
) -> RollbackContract:
    """Prove the rollback is the exact inverse of an approved paired transition."""
    activated_hashes = _hash_pair(
        activated_examples_bytes,
        activated_parity_bytes,
    )
    rollback_hashes = _hash_pair(rollback_examples_bytes, rollback_parity_bytes)
    if (
        activated_hashes != activation.after_sha256
        or rollback_hashes != activation.before_sha256
    ):
        raise ManifestActivationError(
            "rollback pair mismatch: the transition is not the exact inverse"
        )
    activated_pair = _pair(
        activated_examples_bytes,
        activated_parity_bytes,
        repo_root=repo_root,
        context="activated",
    )
    rollback_pair = _pair(
        rollback_examples_bytes,
        rollback_parity_bytes,
        repo_root=repo_root,
        context="rollback",
    )
    _require_version_pair(
        activated_pair,
        activation.after_version_pair,
        "rollback source",
    )
    _require_version_pair(
        rollback_pair,
        activation.before_version_pair,
        "rollback destination",
    )
    return RollbackContract(
        restored_version_pair=rollback_pair.version_pair,
        restored_sha256=rollback_hashes,
    )
