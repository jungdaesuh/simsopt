"""Typed, fail-closed ownership boundary for the JAX examples manifest."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

SourceDisposition = Literal["candidate", "deferred"]
ExampleStatus = Literal["planned", "ready"]
ExecutionKind = Literal["pure", "adapter", "hybrid"]
CoverageState = Literal["planned", "covered", "deferred"]

TIERS = frozenset(
    {"1_Simple", "2_Intermediate", "3_Advanced", "stellarator_benchmarks"}
)
DISPOSITIONS = frozenset({"candidate", "deferred"})
EXAMPLE_STATUSES = frozenset({"planned", "ready"})
EXECUTION_KINDS = frozenset({"pure", "adapter", "hybrid"})
LANES = frozenset({"cpu-smoke", "gpu-strict"})
SOURCE_FIELDS = frozenset({"source", "disposition", "deferred_reason"})
EXAMPLE_FIELDS = frozenset(
    {
        "id",
        "path",
        "status",
        "tier",
        "inspired_by",
        "execution_kind",
        "jax_surfaces",
        "host_boundaries",
        "extras",
        "smoke_args",
        "correctness_tests",
        "lanes",
    }
)


class ManifestValidationError(ValueError):
    """The JAX examples manifest violates its public integrity contract."""


@dataclass(frozen=True)
class SourceRecord:
    source: str
    disposition: SourceDisposition
    deferred_reason: str | None


@dataclass(frozen=True)
class JaxExampleRecord:
    id: str
    path: str
    status: ExampleStatus
    tier: str
    inspired_by: tuple[str, ...]
    execution_kind: ExecutionKind
    jax_surfaces: tuple[str, ...]
    host_boundaries: tuple[str, ...]
    extras: tuple[str, ...]
    smoke_args: tuple[str, ...]
    correctness_tests: tuple[str, ...]
    lanes: tuple[str, ...]


@dataclass(frozen=True)
class JaxExamplesManifest:
    source_catalog: tuple[SourceRecord, ...]
    jax_examples: tuple[JaxExampleRecord, ...]


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestValidationError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{context} must be a JSON array")
    return list(value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{context} must be a non-empty string")
    return value


def _strings(value: object, context: str) -> tuple[str, ...]:
    entries = _sequence(value, context)
    strings = tuple(_string(entry, f"{context} item") for entry in entries)
    if len(strings) != len(set(strings)):
        raise ManifestValidationError(f"{context} contains duplicates")
    return strings


def _source_record(value: object, index: int) -> SourceRecord:
    context = f"source_catalog[{index}]"
    record = _mapping(value, context)
    unexpected = set(record) - SOURCE_FIELDS
    if unexpected:
        raise ManifestValidationError(
            f"unexpected source fields in {context}: {sorted(unexpected)}"
        )
    source = _string(record.get("source"), f"{context}.source")
    disposition_value = _string(record.get("disposition"), f"{context}.disposition")
    if disposition_value not in DISPOSITIONS:
        raise ManifestValidationError(
            f"invalid disposition for {source}: {disposition_value}"
        )
    deferred_reason_value = record.get("deferred_reason")
    if disposition_value == "candidate":
        if "deferred_reason" in record:
            raise ManifestValidationError(
                f"candidate must not define deferred_reason: {source}"
            )
        return SourceRecord(source, "candidate", None)
    if deferred_reason_value is None:
        raise ManifestValidationError(
            f"deferred source requires deferred_reason: {source}"
        )
    return SourceRecord(
        source,
        "deferred",
        _string(deferred_reason_value, f"{context}.deferred_reason"),
    )


def _example_record(value: object, index: int) -> JaxExampleRecord:
    context = f"jax_examples[{index}]"
    record = _mapping(value, context)
    unexpected = set(record) - EXAMPLE_FIELDS
    missing = EXAMPLE_FIELDS - set(record)
    if unexpected or missing:
        raise ManifestValidationError(
            f"invalid example fields in {context}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    example_id = _string(record["id"], f"{context}.id")
    path = _string(record["path"], f"{context}.path")
    status_value = _string(record["status"], f"{context}.status")
    tier = _string(record["tier"], f"{context}.tier")
    execution_value = _string(record["execution_kind"], f"{context}.execution_kind")
    if status_value not in EXAMPLE_STATUSES:
        raise ManifestValidationError(
            f"invalid example status for {example_id}: {status_value}"
        )
    if tier not in TIERS:
        raise ManifestValidationError(f"invalid tier for {example_id}: {tier}")
    if execution_value not in EXECUTION_KINDS:
        raise ManifestValidationError(
            f"invalid execution kind for {example_id}: {execution_value}"
        )

    posix_path = PurePosixPath(path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ManifestValidationError(f"example path must be relative: {path}")
    if not posix_path.parts or posix_path.parts[0] != tier:
        raise ManifestValidationError(
            f"example path tier does not match {tier}: {path}"
        )

    lanes = _strings(record["lanes"], f"{context}.lanes")
    invalid_lanes = set(lanes) - LANES
    if invalid_lanes:
        raise ManifestValidationError(
            f"invalid lanes for {example_id}: {sorted(invalid_lanes)}"
        )

    return JaxExampleRecord(
        id=example_id,
        path=path,
        status="planned" if status_value == "planned" else "ready",
        tier=tier,
        inspired_by=_strings(record["inspired_by"], f"{context}.inspired_by"),
        execution_kind=(
            "pure"
            if execution_value == "pure"
            else "adapter"
            if execution_value == "adapter"
            else "hybrid"
        ),
        jax_surfaces=_strings(record["jax_surfaces"], f"{context}.jax_surfaces"),
        host_boundaries=_strings(
            record["host_boundaries"], f"{context}.host_boundaries"
        ),
        extras=_strings(record["extras"], f"{context}.extras"),
        smoke_args=_strings(record["smoke_args"], f"{context}.smoke_args"),
        correctness_tests=_strings(
            record["correctness_tests"], f"{context}.correctness_tests"
        ),
        lanes=lanes,
    )


def _native_source_paths(repo_root: Path) -> set[str]:
    examples_root = repo_root / "examples"
    return {
        source_path.relative_to(examples_root).as_posix()
        for tier in TIERS
        for source_path in (examples_root / tier).glob("*.py")
    }


def _validate_ready_source(example: JaxExampleRecord, repo_root: Path) -> None:
    if "cpu-smoke" not in example.lanes:
        raise ManifestValidationError(
            f"ready example requires cpu-smoke lane: {example.id}"
        )
    if not example.correctness_tests:
        raise ManifestValidationError(
            f"ready example requires correctness tests: {example.id}"
        )

    example_path = repo_root / "examples" / "jax" / example.path
    if not example_path.is_file():
        raise ManifestValidationError(
            f"ready example path does not exist: {example.path}"
        )
    missing_tests = [
        test_path
        for test_path in example.correctness_tests
        if not (repo_root / test_path).is_file()
    ]
    if missing_tests:
        raise ManifestValidationError(
            f"ready example correctness tests do not exist: {missing_tests}"
        )

    syntax_tree = ast.parse(example_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.ImportFrom)
    }
    if not any(
        module in {"simsopt_jax", "simsopt_jax_adapters"}
        or module.startswith(("simsopt_jax.", "simsopt_jax_adapters."))
        for module in imported_modules
    ):
        raise ManifestValidationError(
            f"ready example must import a public JAX surface: {example.id}"
        )
    forbidden = {
        module
        for module in imported_modules
        if module == "runpy"
        or module == "importlib"
        or module == "examples"
        or module.startswith("examples.")
    }
    if forbidden:
        raise ManifestValidationError(
            f"ready example is a forwarding wrapper: {example.id}: {sorted(forbidden)}"
        )


def _validate_manifest(manifest: JaxExamplesManifest, repo_root: Path) -> None:
    source_paths = [record.source for record in manifest.source_catalog]
    if len(source_paths) != len(set(source_paths)):
        raise ManifestValidationError("duplicate source path in source_catalog")
    native_sources = _native_source_paths(repo_root)
    if set(source_paths) != native_sources:
        raise ManifestValidationError(
            "source catalog does not match tracked native Python examples"
        )

    example_ids = [record.id for record in manifest.jax_examples]
    example_paths = [record.path for record in manifest.jax_examples]
    if len(example_ids) != len(set(example_ids)):
        raise ManifestValidationError("duplicate JAX example id")
    if len(example_paths) != len(set(example_paths)):
        raise ManifestValidationError("duplicate JAX example path")

    source_by_path = {record.source: record for record in manifest.source_catalog}
    linked_sources: set[str] = set()
    for example in manifest.jax_examples:
        if not example.inspired_by:
            raise ManifestValidationError(
                f"example requires at least one inspiration source: {example.id}"
            )
        unknown_sources = set(example.inspired_by) - set(source_by_path)
        if unknown_sources:
            raise ManifestValidationError(
                f"unknown inspiration source for {example.id}: {sorted(unknown_sources)}"
            )
        linked_sources.update(example.inspired_by)

        if example.execution_kind == "pure" and example.host_boundaries:
            raise ManifestValidationError(
                f"pure example must not declare host boundaries: {example.id}"
            )
        if example.execution_kind in {"adapter", "hybrid"} and not (
            example.host_boundaries
        ):
            raise ManifestValidationError(
                f"{example.execution_kind} example requires host boundaries: "
                f"{example.id}"
            )
        if example.status == "ready":
            _validate_ready_source(example, repo_root)

    for source in manifest.source_catalog:
        is_linked = source.source in linked_sources
        if source.disposition == "candidate" and not is_linked:
            raise ManifestValidationError(
                f"candidate source is not linked: {source.source}"
            )
        if source.disposition == "deferred" and is_linked:
            raise ManifestValidationError(
                f"deferred source must not be linked: {source.source}"
            )


def load_manifest(path: Path, *, repo_root: Path) -> JaxExamplesManifest:
    """Parse and validate one immutable JAX examples manifest."""

    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    expected_root_fields = {"source_catalog", "jax_examples"}
    if set(root) != expected_root_fields:
        raise ManifestValidationError(
            "manifest fields must be exactly source_catalog and jax_examples"
        )
    manifest = JaxExamplesManifest(
        source_catalog=tuple(
            _source_record(record, index)
            for index, record in enumerate(
                _sequence(root["source_catalog"], "source_catalog")
            )
        ),
        jax_examples=tuple(
            _example_record(record, index)
            for index, record in enumerate(
                _sequence(root["jax_examples"], "jax_examples")
            )
        ),
    )
    _validate_manifest(manifest, repo_root)
    return manifest


def derive_source_coverage(
    manifest: JaxExamplesManifest,
) -> dict[str, CoverageState]:
    """Derive source coverage exclusively from JAX records' inspiration links."""

    statuses_by_source: dict[str, list[ExampleStatus]] = {
        record.source: [] for record in manifest.source_catalog
    }
    for example in manifest.jax_examples:
        for source in example.inspired_by:
            statuses_by_source[source].append(example.status)

    return {
        source.source: (
            "deferred"
            if source.disposition == "deferred"
            else "covered"
            if "ready" in statuses_by_source[source.source]
            else "planned"
        )
        for source in manifest.source_catalog
    }
