"""Emit an exact schema-v3/parity-v2 migration candidate without writing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from examples.jax.manifest_contracts_v3 import build_v3_candidates


class CandidateInputError(ValueError):
    """Active legacy inputs do not match the approved frozen inventory baseline."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--parity", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--dry-run", required=True, action="store_true")
    return parser


def _document(data: bytes, context: str) -> dict[str, object]:
    value = json.loads(data)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CandidateInputError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _baseline_hash(inventory: dict[str, object], key: str) -> str:
    baseline = inventory.get("baseline")
    if not isinstance(baseline, dict):
        raise CandidateInputError("inventory baseline must be a JSON object")
    value = baseline.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise CandidateInputError(f"inventory baseline {key} is invalid")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(arguments: list[str] | None = None) -> int:
    """Validate frozen inputs and print one canonical, no-write review envelope."""
    parsed = _parser().parse_args(arguments)
    examples_bytes = parsed.examples.read_bytes()
    parity_bytes = parsed.parity.read_bytes()
    inventory_bytes = parsed.inventory.read_bytes()
    examples_document = _document(examples_bytes, "examples manifest")
    parity_document = _document(parity_bytes, "parity manifest")
    inventory_document = _document(inventory_bytes, "inventory")
    input_hashes = {
        "examples_manifest_v2": _sha256(examples_bytes),
        "parity_manifest_v1": _sha256(parity_bytes),
    }
    expected_hashes = {
        "examples_manifest_v2": _baseline_hash(inventory_document, "manifest_sha256"),
        "parity_manifest_v1": _baseline_hash(
            inventory_document, "parity_manifest_sha256"
        ),
    }
    if input_hashes != expected_hashes:
        raise CandidateInputError(
            f"active legacy inputs differ from frozen baseline: "
            f"expected={expected_hashes}, actual={input_hashes}"
        )

    candidate = build_v3_candidates(
        examples_v2_document=examples_document,
        parity_v1_document=parity_document,
        inventory_document=inventory_document,
        repo_root=_REPO_ROOT,
    )
    envelope = {
        "mode": "no_write",
        "input_sha256": input_hashes,
        "candidate_sha256": {
            "examples_manifest_v3": candidate.examples_sha256,
            "parity_manifest_v2": candidate.parity_sha256,
        },
        "semantic_diff": dict(candidate.semantic_diff),
        "approval_gate": "explicit approval of both candidate SHA-256 values",
        "compatibility_duration": "one documented deprecation interval",
        "rollback_scope": [
            "examples manifest",
            "parity manifest",
            "activation readers and tests",
            "artifact observability",
            "compatibility behavior",
        ],
        "candidates": {
            "examples_manifest_v3": json.loads(candidate.examples_bytes),
            "parity_manifest_v2": json.loads(candidate.parity_bytes),
        },
    }
    sys.stdout.write(
        json.dumps(
            envelope,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
