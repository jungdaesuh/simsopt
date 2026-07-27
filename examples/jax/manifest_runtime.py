"""Runtime-only adapter over atomic legacy and canonical manifest pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TextIO

from examples.jax._manifest import JaxExamplesManifest
from examples.jax.manifest_contracts_v3 import (
    CompatibilityAlias,
    JaxExamplesManifestV3,
    ManifestContractPair,
    load_manifest_contract_pair_documents,
)
from examples.jax.parity._manifest import ParityManifest

RuntimeStatus = Literal["planned", "ready"]
RuntimeClassification = Literal["pure", "mirror", "adapter", "hybrid", "tutorial"]
RuntimeTeachingKind = Literal["legacy", "one_to_one", "combined", "compatibility"]
_LANE_BY_DEVICE: Final = {"cpu": "cpu-smoke", "gpu": "gpu-strict"}


class RuntimeManifestError(ValueError):
    """A validated manifest pair cannot be adapted to executable runtime state."""


@dataclass(frozen=True)
class RuntimeExample:
    """The minimal immutable record consumed by example runners."""

    id: str
    path: str
    status: RuntimeStatus
    lanes: tuple[str, ...]
    smoke_args: tuple[str, ...]
    classification: RuntimeClassification
    teaching_kind: RuntimeTeachingKind
    source: str | None
    compatibility: CompatibilityAlias | None


@dataclass(frozen=True)
class RuntimeContractPair:
    """Executable records and parity policy from one validated version pair."""

    version_pair: tuple[int, int]
    used_legacy_adapter: bool
    examples: tuple[RuntimeExample, ...]
    parity: ParityManifest


def _document(path: Path, context: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeManifestError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _legacy_examples(manifest: JaxExamplesManifest) -> tuple[RuntimeExample, ...]:
    return tuple(
        RuntimeExample(
            id=example.id,
            path=example.path,
            status=example.status,
            lanes=example.lanes,
            smoke_args=example.smoke_args,
            classification=(
                "pure"
                if example.execution_kind == "pure"
                else "adapter"
                if example.execution_kind == "adapter"
                else "hybrid"
            ),
            teaching_kind="legacy",
            source=None,
            compatibility=None,
        )
        for example in manifest.jax_examples
    )


def _canonical_lanes(device_scopes: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(_LANE_BY_DEVICE[device] for device, _scope in device_scopes)


def _canonical_examples(
    manifest: JaxExamplesManifestV3,
) -> tuple[RuntimeExample, ...]:
    source_by_example_id = {
        source.mirror_example_id: source.source
        for source in manifest.source_catalog
        if source.mirror_example_id is not None
    }
    return tuple(
        RuntimeExample(
            id=example.id,
            path=example.path,
            status=example.status,
            lanes=_canonical_lanes(example.supported_device_scopes),
            smoke_args=example.smoke_args,
            classification=example.classification,
            teaching_kind=example.teaching_kind,
            source=source_by_example_id.get(example.id),
            compatibility=example.compatibility,
        )
        for example in manifest.jax_examples
    )


def _runtime_pair(pair: ManifestContractPair) -> RuntimeContractPair:
    if isinstance(pair.examples, JaxExamplesManifest):
        examples = _legacy_examples(pair.examples)
    elif isinstance(pair.examples, JaxExamplesManifestV3):
        examples = _canonical_examples(pair.examples)
    else:
        raise RuntimeManifestError("unsupported validated examples manifest type")
    return RuntimeContractPair(
        version_pair=pair.version_pair,
        used_legacy_adapter=pair.used_legacy_adapter,
        examples=examples,
        parity=pair.parity,
    )


def load_runtime_contract_pair(
    examples_path: Path,
    parity_path: Path,
    *,
    repo_root: Path,
) -> RuntimeContractPair:
    """Read, validate, and adapt one complete legacy or canonical pair."""
    pair = load_manifest_contract_pair_documents(
        _document(examples_path, "examples manifest"),
        _document(parity_path, "parity manifest"),
        repo_root=repo_root,
    )
    return _runtime_pair(pair)


def emit_compatibility_warning(example: RuntimeExample, *, stream: TextIO) -> bool:
    """Emit the schema-owned warning for an actual compatibility alias."""
    if example.compatibility is None:
        return False
    print(example.compatibility.warning, file=stream)
    return True
