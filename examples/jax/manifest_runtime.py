"""Runtime-only adapter over the JAX example manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from examples.jax.manifest_contracts_v3 import (
    ExampleClassification,
    ExampleStatus,
    JaxExamplesManifestV3,
    TeachingKind,
    parse_examples_v3_document,
)
from examples.jax.outer_optimizer_policy import OuterOptimizerPolicy

_LANE_BY_DEVICE: Final = {"cpu": "cpu-smoke", "gpu": "gpu-strict"}


class RuntimeManifestError(ValueError):
    """A manifest document cannot be adapted to executable runtime state."""


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
class RuntimeManifest:
    """Executable records from one validated example manifest."""

    schema_version: int
    examples: tuple[RuntimeExample, ...]


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
            outer_optimizer_policy=example.outer_optimizer_policy,
        )
        for example in manifest.jax_examples
    )


def load_runtime_manifest(examples_path: Path, *, repo_root: Path) -> RuntimeManifest:
    """Read, validate, and adapt one example manifest."""
    manifest = parse_examples_v3_document(
        _document(examples_path, "examples manifest"), repo_root=repo_root
    )
    return RuntimeManifest(
        schema_version=manifest.schema_version,
        examples=_canonical_examples(manifest),
    )
