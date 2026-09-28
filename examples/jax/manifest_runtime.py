"""Runtime-only adapter over the atomic example and parity manifest pair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from examples.jax.manifest_contracts_v3 import (
    ExampleClassification,
    ExampleStatus,
    JaxExamplesManifestV3,
    ManifestContractPair,
    TeachingKind,
    load_manifest_contract_pair_documents,
)
from examples.jax.outer_optimizer_policy import OuterOptimizerPolicy
from examples.jax.parity._manifest import ParityManifest

_LANE_BY_DEVICE: Final = {"cpu": "cpu-smoke", "gpu": "gpu-strict"}


class RuntimeManifestError(ValueError):
    """A validated manifest pair cannot be adapted to executable runtime state."""


@dataclass(frozen=True)
class RuntimeExample:
    """The minimal immutable record consumed by example runners."""

    id: str
    path: str
    status: ExampleStatus
    lanes: tuple[str, ...]
    smoke_args: tuple[str, ...]
    classification: ExampleClassification
    teaching_kind: TeachingKind
    source: str | None
    outer_optimizer_policy: OuterOptimizerPolicy | None = None


@dataclass(frozen=True)
class RuntimeContractPair:
    """Executable records and parity policy from one validated version pair."""

    version_pair: tuple[int, int]
    examples: tuple[RuntimeExample, ...]
    parity: ParityManifest


def _document(path: Path, context: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeManifestError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _canonical_lanes(device_scopes: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(_LANE_BY_DEVICE[device] for device, _scope in device_scopes)


def _canonical_examples(
    manifest: JaxExamplesManifestV3,
) -> tuple[RuntimeExample, ...]:
    source_by_example_id = {
        source.mirror_example_id: source.source
        for source in manifest.all_sources
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
            outer_optimizer_policy=example.outer_optimizer_policy,
        )
        for example in manifest.jax_examples
    )


def _runtime_pair(pair: ManifestContractPair) -> RuntimeContractPair:
    return RuntimeContractPair(
        version_pair=pair.version_pair,
        examples=_canonical_examples(pair.examples),
        parity=pair.parity,
    )


def load_runtime_contract_pair(
    examples_path: Path,
    parity_path: Path,
    *,
    repo_root: Path,
) -> RuntimeContractPair:
    """Read, validate, and adapt one complete example/parity manifest pair."""
    pair = load_manifest_contract_pair_documents(
        _document(examples_path, "examples manifest"),
        _document(parity_path, "parity manifest"),
        repo_root=repo_root,
    )
    return _runtime_pair(pair)
