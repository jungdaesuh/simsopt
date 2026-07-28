"""Typed, fail-closed ownership boundary for the JAX examples manifest."""

from __future__ import annotations

import ast
import json
import warnings
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
DEVICES = frozenset({"cpu", "gpu"})
_LANE_TO_DEVICE = {"cpu-smoke": "cpu", "gpu-strict": "gpu"}
_DEVICE_TO_LANE = {device: lane for lane, device in _LANE_TO_DEVICE.items()}
SOURCE_FIELDS = frozenset({"source", "disposition", "deferred_reason"})
EXAMPLE_COMMON_FIELDS = frozenset(
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
    }
)
EXAMPLE_V1_FIELDS = EXAMPLE_COMMON_FIELDS | {"lanes"}
EXAMPLE_V2_FIELDS = EXAMPLE_COMMON_FIELDS | {"devices"}


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

    @property
    def devices(self) -> tuple[str, ...]:
        """Return normalized CPU/GPU capability independent of source schema."""
        return tuple(_LANE_TO_DEVICE[lane] for lane in self.lanes)


@dataclass(frozen=True)
class JaxExamplesManifest:
    source_catalog: tuple[SourceRecord, ...]
    jax_examples: tuple[JaxExampleRecord, ...]
    schema_version: Literal[1, 2] = 2
    used_legacy_manifest_adapter: bool = False


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


def _example_record(
    value: object,
    index: int,
    *,
    schema_version: Literal[1, 2],
) -> JaxExampleRecord:
    context = f"jax_examples[{index}]"
    record = _mapping(value, context)
    if "intents" in record:
        raise ManifestValidationError(f"per-example intents are forbidden in {context}")
    if "lanes" in record and "devices" in record:
        raise ManifestValidationError(f"{context} must not mix lanes and devices")
    expected_fields = EXAMPLE_V1_FIELDS if schema_version == 1 else EXAMPLE_V2_FIELDS
    unexpected = set(record) - expected_fields
    missing = expected_fields - set(record)
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

    if schema_version == 1:
        lanes = _strings(record["lanes"], f"{context}.lanes")
        invalid_lanes = set(lanes) - LANES
        if invalid_lanes:
            raise ManifestValidationError(
                f"invalid lanes for {example_id}: {sorted(invalid_lanes)}"
            )
    else:
        devices = _strings(record["devices"], f"{context}.devices")
        invalid_devices = set(devices) - DEVICES
        if invalid_devices:
            raise ManifestValidationError(
                f"invalid devices for {example_id}: {sorted(invalid_devices)}"
            )
        lanes = tuple(_DEVICE_TO_LANE[device] for device in devices)

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
    if "gpu-strict" not in example.lanes:
        raise ManifestValidationError(
            f"ready example requires gpu-strict lane: {example.id}"
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


def _validate_manifest(
    manifest: JaxExamplesManifest,
    repo_root: Path,
    *,
    allow_historical_catalog: bool,
) -> None:
    source_paths = [record.source for record in manifest.source_catalog]
    if len(source_paths) != len(set(source_paths)):
        raise ManifestValidationError("duplicate source path in source_catalog")
    native_sources = _native_source_paths(repo_root)
    catalog_sources = set(source_paths)
    catalog_is_valid = (
        catalog_sources <= native_sources
        if allow_historical_catalog
        else catalog_sources == native_sources
    )
    if not catalog_is_valid:
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


def _schema_version(root: dict[str, object]) -> Literal[1, 2]:
    if "schema_version" not in root:
        expected_fields = {"source_catalog", "jax_examples"}
        if set(root) != expected_fields:
            raise ManifestValidationError(
                "legacy manifest fields must be exactly source_catalog and jax_examples"
            )
        return 1
    value = root["schema_version"]
    if value == 1:
        raise ManifestValidationError(
            "schema_version 1 must be absent for the legacy v1 contract"
        )
    if isinstance(value, bool) or value != 2:
        raise ManifestValidationError(f"unsupported manifest schema: {value!r}")
    expected_fields = {"schema_version", "source_catalog", "jax_examples"}
    if set(root) != expected_fields:
        raise ManifestValidationError(
            "v2 manifest fields must be exactly schema_version, source_catalog, "
            "and jax_examples"
        )
    return 2


def _parse_manifest_document(
    document: object,
    *,
    repo_root: Path,
    warn_legacy: bool,
    allow_historical_catalog: bool,
) -> JaxExamplesManifest:
    root = _mapping(document, "manifest")
    schema_version = _schema_version(root)
    if schema_version == 1 and warn_legacy:
        warnings.warn(
            "manifest schema v1 is deprecated; migrate to explicit v2 devices "
            "with examples/jax/migrate_manifest.py --dry-run",
            FutureWarning,
            stacklevel=3,
        )
    manifest = JaxExamplesManifest(
        source_catalog=tuple(
            _source_record(record, index)
            for index, record in enumerate(
                _sequence(root["source_catalog"], "source_catalog")
            )
        ),
        jax_examples=tuple(
            _example_record(record, index, schema_version=schema_version)
            for index, record in enumerate(
                _sequence(root["jax_examples"], "jax_examples")
            )
        ),
        schema_version=schema_version,
        used_legacy_manifest_adapter=schema_version == 1,
    )
    _validate_manifest(
        manifest,
        repo_root,
        allow_historical_catalog=allow_historical_catalog,
    )
    return manifest


def load_manifest(path: Path, *, repo_root: Path) -> JaxExamplesManifest:
    """Parse and validate one immutable JAX examples manifest."""

    return parse_manifest_document(
        json.loads(path.read_text(encoding="utf-8")),
        repo_root=repo_root,
        warn_legacy=True,
        allow_historical_catalog=False,
    )


def parse_manifest_document(
    document: object,
    *,
    repo_root: Path,
    warn_legacy: bool = False,
    allow_historical_catalog: bool = False,
) -> JaxExamplesManifest:
    """Parse manifest bytes already decoded by an atomic contract reader."""
    return _parse_manifest_document(
        document,
        repo_root=repo_root,
        warn_legacy=warn_legacy,
        allow_historical_catalog=allow_historical_catalog,
    )


def manifest_semantic_diff(
    before: JaxExamplesManifest,
    after: JaxExamplesManifest,
) -> dict[str, bool]:
    """Compare migration-relevant semantics without comparing storage fields."""
    source_catalog_equal = before.source_catalog == after.source_catalog
    before_ids = tuple(example.id for example in before.jax_examples)
    after_ids = tuple(example.id for example in after.jax_examples)
    example_ids_equal = set(before_ids) == set(after_ids)
    example_order_equal = before_ids == after_ids
    before_by_id = {example.id: example for example in before.jax_examples}
    after_by_id = {example.id: example for example in after.jax_examples}

    def field_equal(field: str) -> bool:
        return example_ids_equal and all(
            getattr(before_by_id[example_id], field)
            == getattr(after_by_id[example_id], field)
            for example_id in before_ids
        )

    readiness_equal = field_equal("status")
    lineage_equal = field_equal("inspired_by")
    paths_equal = field_equal("path")
    device_capabilities_equal = example_ids_equal and all(
        before_by_id[example_id].devices == after_by_id[example_id].devices
        for example_id in before_ids
    )
    semantic_equal = all(
        (
            source_catalog_equal,
            example_ids_equal,
            example_order_equal,
            readiness_equal,
            lineage_equal,
            paths_equal,
            device_capabilities_equal,
            before.jax_examples == after.jax_examples,
        )
    )
    return {
        "device_capabilities_equal": device_capabilities_equal,
        "example_ids_equal": example_ids_equal,
        "example_order_equal": example_order_equal,
        "lineage_equal": lineage_equal,
        "paths_equal": paths_equal,
        "readiness_equal": readiness_equal,
        "source_catalog_equal": source_catalog_equal,
        "semantic_equal": semantic_equal,
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def convert_v1_document_to_v2(
    document: object,
    *,
    repo_root: Path,
    allow_historical_catalog: bool = False,
) -> tuple[bytes, dict[str, bool]]:
    """Return deterministic v2 bytes after proving semantic equivalence."""
    before = _parse_manifest_document(
        document,
        repo_root=repo_root,
        warn_legacy=False,
        allow_historical_catalog=allow_historical_catalog,
    )
    if before.schema_version != 1:
        raise ManifestValidationError("migration input must use absent-schema v1")
    source_document = _mapping(document, "manifest")
    candidate_examples: list[dict[str, object]] = []
    for index, value in enumerate(
        _sequence(source_document["jax_examples"], "jax_examples")
    ):
        record = _mapping(value, f"jax_examples[{index}]")
        lanes = _strings(record.pop("lanes"), f"jax_examples[{index}].lanes")
        record["devices"] = [_LANE_TO_DEVICE[lane] for lane in lanes]
        candidate_examples.append(record)
    candidate: dict[str, object] = {
        "schema_version": 2,
        "source_catalog": source_document["source_catalog"],
        "jax_examples": candidate_examples,
    }
    after = _parse_manifest_document(
        candidate,
        repo_root=repo_root,
        warn_legacy=False,
        allow_historical_catalog=allow_historical_catalog,
    )
    semantic_diff = manifest_semantic_diff(before, after)
    if not semantic_diff["semantic_equal"]:
        raise ManifestValidationError(
            f"manifest migration changed semantics: {semantic_diff}"
        )
    return _canonical_json_bytes(candidate), semantic_diff


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
