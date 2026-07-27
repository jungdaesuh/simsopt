"""Typed no-write migration boundary for JAX example and parity manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Mapping

from examples.jax._manifest import (
    JaxExampleRecord,
    JaxExamplesManifest,
    parse_manifest_document,
)
from examples.jax.parity._manifest import (
    ParityManifest,
    ParityRelationship,
    parse_parity_manifest_document,
    parse_parity_relationships_document,
)

SourceDispositionV3 = Literal["eligible", "hybrid", "blocked", "not_applicable"]
PortStatus = Literal["planned", "ready", "blocked", "not_applicable"]
ExampleStatus = Literal["planned", "ready"]
ExampleClassification = Literal["mirror", "adapter", "hybrid", "tutorial"]
TeachingKind = Literal["one_to_one", "combined", "compatibility"]
DeviceScope = Literal[
    "full_workflow", "jax_region", "host_and_jax_slice", "jax_slice_only"
]

_TIERS = frozenset(
    {"1_Simple", "2_Intermediate", "3_Advanced", "stellarator_benchmarks"}
)
_SOURCE_DISPOSITIONS = frozenset({"eligible", "hybrid", "blocked", "not_applicable"})
_PORT_STATUSES = frozenset({"planned", "ready", "blocked", "not_applicable"})
_EXAMPLE_STATUSES = frozenset({"planned", "ready"})
_CLASSIFICATIONS = frozenset({"mirror", "adapter", "hybrid", "tutorial"})
_TEACHING_KINDS = frozenset({"one_to_one", "combined", "compatibility"})
_DEVICE_SCOPES = frozenset(
    {"full_workflow", "jax_region", "host_and_jax_slice", "jax_slice_only"}
)
_SOURCE_FIELDS = frozenset(
    {
        "source",
        "disposition",
        "port_status",
        "reason",
        "blocker",
        "reconsideration_condition",
        "dependencies",
        "mirror_example_id",
    }
)
_EXAMPLE_FIELDS = frozenset(
    {
        "id",
        "path",
        "status",
        "tier",
        "classification",
        "teaching_kind",
        "jax_surfaces",
        "host_boundaries",
        "extras",
        "smoke_args",
        "correctness_tests",
        "supported_device_scopes",
    }
)


class ManifestV3ValidationError(ValueError):
    """A schema-v3 example contract violates its ownership boundary."""


class ContractVersionError(ValueError):
    """Example and parity manifest versions cannot be read atomically."""


@dataclass(frozen=True)
class RuntimeDependencies:
    python_import_roots: tuple[str, ...]
    external_runtimes: tuple[str, ...]


@dataclass(frozen=True)
class SourceRecordV3:
    source: str
    disposition: SourceDispositionV3
    port_status: PortStatus
    reason: str
    blocker: str | None
    reconsideration_condition: str | None
    dependencies: RuntimeDependencies
    mirror_example_id: str | None


@dataclass(frozen=True)
class JaxExampleRecordV3:
    id: str
    path: str
    status: ExampleStatus
    tier: str
    classification: ExampleClassification
    teaching_kind: TeachingKind
    jax_surfaces: tuple[str, ...]
    host_boundaries: tuple[str, ...]
    extras: tuple[str, ...]
    smoke_args: tuple[str, ...]
    correctness_tests: tuple[str, ...]
    supported_device_scopes: tuple[tuple[str, DeviceScope], ...]

    @property
    def device_scopes(self) -> Mapping[str, DeviceScope]:
        """Expose immutable device-to-scientific-scope ownership."""
        return MappingProxyType(dict(self.supported_device_scopes))


@dataclass(frozen=True)
class JaxExamplesManifestV3:
    source_catalog: tuple[SourceRecordV3, ...]
    jax_examples: tuple[JaxExampleRecordV3, ...]
    schema_version: Literal[3] = 3


@dataclass(frozen=True)
class ManifestContractPair:
    version_pair: tuple[int, int]
    used_legacy_adapter: bool
    examples: JaxExamplesManifest | JaxExamplesManifestV3
    parity: ParityManifest


@dataclass(frozen=True)
class MigrationCandidate:
    examples_bytes: bytes
    parity_bytes: bytes
    examples_sha256: str
    parity_sha256: str
    semantic_diff: Mapping[str, int]


@dataclass(frozen=True)
class _CandidateDocuments:
    examples: dict[str, object]
    parity: dict[str, object]
    planned_one_to_one_count: int
    relationship_count: int


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestV3ValidationError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ManifestV3ValidationError(f"{context} must be a JSON array")
    return list(value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestV3ValidationError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _strings(
    value: object, context: str, *, require_sorted: bool = False
) -> tuple[str, ...]:
    entries = tuple(
        _string(entry, f"{context} item") for entry in _sequence(value, context)
    )
    if len(entries) != len(set(entries)):
        raise ManifestV3ValidationError(f"{context} contains duplicates")
    if require_sorted and entries != tuple(sorted(entries)):
        raise ManifestV3ValidationError(f"{context} must be sorted")
    return entries


def _exact_fields(
    record: dict[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(record)
    if actual != expected:
        raise ManifestV3ValidationError(
            f"unexpected {context} fields: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _enum(value: object, allowed: frozenset[str], context: str) -> str:
    result = _string(value, context)
    if result not in allowed:
        raise ManifestV3ValidationError(f"invalid {context}: {result}")
    return result


def _dependencies(value: object, context: str) -> RuntimeDependencies:
    record = _mapping(value, context)
    _exact_fields(
        record,
        frozenset({"python_import_roots", "external_runtimes"}),
        f"{context} dependency",
    )
    return RuntimeDependencies(
        python_import_roots=_strings(
            record["python_import_roots"],
            f"{context}.python_import_roots",
            require_sorted=True,
        ),
        external_runtimes=_strings(
            record["external_runtimes"],
            f"{context}.external_runtimes",
            require_sorted=True,
        ),
    )


def _source_record(value: object, index: int) -> SourceRecordV3:
    context = f"source_catalog[{index}]"
    record = _mapping(value, context)
    _exact_fields(record, _SOURCE_FIELDS, "source")
    disposition_value = _enum(
        record["disposition"], _SOURCE_DISPOSITIONS, f"{context}.disposition"
    )
    port_status_value = _enum(
        record["port_status"], _PORT_STATUSES, f"{context}.port_status"
    )
    blocker = _optional_string(record["blocker"], f"{context}.blocker")
    reconsideration = _optional_string(
        record["reconsideration_condition"],
        f"{context}.reconsideration_condition",
    )
    mirror_id = _optional_string(
        record["mirror_example_id"], f"{context}.mirror_example_id"
    )
    if disposition_value in {"eligible", "hybrid"}:
        if mirror_id is None:
            raise ManifestV3ValidationError(
                f"eligible source requires exactly one mirror: {record['source']}"
            )
        if blocker is not None or reconsideration is not None:
            raise ManifestV3ValidationError(
                f"eligible source cannot declare a blocker: {record['source']}"
            )
        if port_status_value not in {"planned", "ready"}:
            raise ManifestV3ValidationError(
                f"eligible source has invalid port status: {record['source']}"
            )
    elif disposition_value == "blocked":
        if mirror_id is not None or blocker is None or reconsideration is None:
            raise ManifestV3ValidationError(
                f"blocked source requires blocker and reconsideration: {record['source']}"
            )
        if port_status_value != "blocked":
            raise ManifestV3ValidationError(
                f"blocked source must have blocked port status: {record['source']}"
            )
    else:
        if mirror_id is not None or blocker is not None or reconsideration is None:
            raise ManifestV3ValidationError(
                f"not_applicable source contract is inconsistent: {record['source']}"
            )
        if port_status_value != "not_applicable":
            raise ManifestV3ValidationError(
                f"not_applicable source has invalid port status: {record['source']}"
            )
    return SourceRecordV3(
        source=_string(record["source"], f"{context}.source"),
        disposition=(
            "eligible"
            if disposition_value == "eligible"
            else "hybrid"
            if disposition_value == "hybrid"
            else "blocked"
            if disposition_value == "blocked"
            else "not_applicable"
        ),
        port_status=(
            "planned"
            if port_status_value == "planned"
            else "ready"
            if port_status_value == "ready"
            else "blocked"
            if port_status_value == "blocked"
            else "not_applicable"
        ),
        reason=_string(record["reason"], f"{context}.reason"),
        blocker=blocker,
        reconsideration_condition=reconsideration,
        dependencies=_dependencies(record["dependencies"], f"{context}.dependencies"),
        mirror_example_id=mirror_id,
    )


def _device_scopes(value: object, context: str) -> tuple[tuple[str, DeviceScope], ...]:
    record = _mapping(value, context)
    if not record or set(record) - {"cpu", "gpu"}:
        raise ManifestV3ValidationError(f"{context} has invalid devices")
    entries: list[tuple[str, DeviceScope]] = []
    for device in sorted(record):
        scope_value = _enum(record[device], _DEVICE_SCOPES, f"{context}.{device}")
        scope: DeviceScope = (
            "full_workflow"
            if scope_value == "full_workflow"
            else "jax_region"
            if scope_value == "jax_region"
            else "host_and_jax_slice"
            if scope_value == "host_and_jax_slice"
            else "jax_slice_only"
        )
        entries.append((device, scope))
    return tuple(entries)


def _example_record(value: object, index: int) -> JaxExampleRecordV3:
    context = f"jax_examples[{index}]"
    record = _mapping(value, context)
    _exact_fields(record, _EXAMPLE_FIELDS, "executable")
    path = _string(record["path"], f"{context}.path")
    tier = _enum(record["tier"], _TIERS, f"{context}.tier")
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != tier
        or relative.suffix != ".py"
    ):
        raise ManifestV3ValidationError(f"invalid executable path: {path}")
    status_value = _enum(record["status"], _EXAMPLE_STATUSES, f"{context}.status")
    classification_value = _enum(
        record["classification"], _CLASSIFICATIONS, f"{context}.classification"
    )
    teaching_value = _enum(
        record["teaching_kind"], _TEACHING_KINDS, f"{context}.teaching_kind"
    )
    host_boundaries = _strings(record["host_boundaries"], f"{context}.host_boundaries")
    scopes = _device_scopes(
        record["supported_device_scopes"], f"{context}.supported_device_scopes"
    )
    scope_by_device = dict(scopes)
    if classification_value == "mirror" and host_boundaries:
        raise ManifestV3ValidationError("pure mirror cannot declare host boundaries")
    if classification_value in {"adapter", "hybrid"} and not host_boundaries:
        raise ManifestV3ValidationError(
            f"{classification_value} requires host boundaries"
        )
    if (
        classification_value == "hybrid"
        and scope_by_device.get("gpu") != "jax_slice_only"
    ):
        raise ManifestV3ValidationError(
            "hybrid GPU scope must be declared as jax_slice_only"
        )
    return JaxExampleRecordV3(
        id=_string(record["id"], f"{context}.id"),
        path=path,
        status="planned" if status_value == "planned" else "ready",
        tier=tier,
        classification=(
            "mirror"
            if classification_value == "mirror"
            else "adapter"
            if classification_value == "adapter"
            else "hybrid"
            if classification_value == "hybrid"
            else "tutorial"
        ),
        teaching_kind=(
            "one_to_one"
            if teaching_value == "one_to_one"
            else "combined"
            if teaching_value == "combined"
            else "compatibility"
        ),
        jax_surfaces=_strings(record["jax_surfaces"], f"{context}.jax_surfaces"),
        host_boundaries=host_boundaries,
        extras=_strings(record["extras"], f"{context}.extras"),
        smoke_args=_strings(record["smoke_args"], f"{context}.smoke_args"),
        correctness_tests=_strings(
            record["correctness_tests"], f"{context}.correctness_tests"
        ),
        supported_device_scopes=scopes,
    )


def _tracked_native_sources(repo_root: Path) -> set[str]:
    examples_root = repo_root / "examples"
    return {
        path.relative_to(examples_root).as_posix()
        for tier in _TIERS
        for path in (examples_root / tier).glob("*.py")
    }


def _validate_v3_ownership(manifest: JaxExamplesManifestV3, repo_root: Path) -> None:
    sources = tuple(record.source for record in manifest.source_catalog)
    if sources != tuple(sorted(set(sources))):
        raise ManifestV3ValidationError("source catalog must be sorted and unique")
    if set(sources) != _tracked_native_sources(repo_root):
        raise ManifestV3ValidationError(
            "source catalog does not match tracked native examples"
        )
    example_ids = tuple(record.id for record in manifest.jax_examples)
    example_paths = tuple(record.path for record in manifest.jax_examples)
    if len(example_ids) != len(set(example_ids)):
        raise ManifestV3ValidationError("duplicate executable id")
    if len(example_paths) != len(set(example_paths)):
        raise ManifestV3ValidationError("duplicate executable path")
    by_id = {record.id: record for record in manifest.jax_examples}
    owners: dict[str, str] = {}
    for source in manifest.source_catalog:
        mirror_id = source.mirror_example_id
        if mirror_id is None:
            continue
        if mirror_id in owners:
            raise ManifestV3ValidationError(
                f"duplicate mirror ownership: {mirror_id}: "
                f"{owners[mirror_id]} and {source.source}"
            )
        owners[mirror_id] = source.source
        example = by_id.get(mirror_id)
        if example is None:
            raise ManifestV3ValidationError(
                f"eligible source requires existing mirror: {source.source}"
            )
        if example.classification == "tutorial":
            raise ManifestV3ValidationError(
                f"tutorial cannot own coverage: {example.id}"
            )
        if example.teaching_kind != "one_to_one":
            raise ManifestV3ValidationError(
                f"owned executable must be one_to_one: {example.id}"
            )
        if example.path != source.source:
            raise ManifestV3ValidationError(
                f"exact-name mirror path mismatch: {source.source} != {example.path}"
            )
        if source.disposition == "hybrid" and example.classification != "hybrid":
            raise ManifestV3ValidationError(
                f"hybrid source must own hybrid executable: {source.source}"
            )
        if source.disposition == "eligible" and example.classification not in {
            "mirror",
            "adapter",
        }:
            raise ManifestV3ValidationError(
                f"eligible source owns invalid classification: {source.source}"
            )
        if source.port_status != example.status:
            raise ManifestV3ValidationError(
                f"source and executable readiness disagree: {source.source}"
            )
    one_to_one_ids = {
        record.id
        for record in manifest.jax_examples
        if record.teaching_kind == "one_to_one"
    }
    if one_to_one_ids != set(owners):
        raise ManifestV3ValidationError(
            "one_to_one executable ownership is incomplete: "
            f"missing={sorted(one_to_one_ids - set(owners))}, "
            f"unexpected={sorted(set(owners) - one_to_one_ids)}"
        )
    for example in manifest.jax_examples:
        if example.status == "ready":
            path = repo_root / "examples" / "jax" / example.path
            if not path.is_file():
                raise ManifestV3ValidationError(
                    f"ready executable path does not exist: {example.path}"
                )
            for test_path in example.correctness_tests:
                if not (repo_root / test_path).is_file():
                    raise ManifestV3ValidationError(
                        f"ready correctness test does not exist: {test_path}"
                    )


def parse_examples_v3_document(
    document: object, *, repo_root: Path
) -> JaxExamplesManifestV3:
    """Parse schema v3 and enforce sole, exact-name source ownership."""
    root = _mapping(document, "manifest")
    _exact_fields(
        root,
        frozenset({"schema_version", "source_catalog", "jax_examples"}),
        "manifest root",
    )
    if root["schema_version"] != 3:
        raise ManifestV3ValidationError(
            f"unsupported example schema: {root['schema_version']!r}"
        )
    manifest = JaxExamplesManifestV3(
        source_catalog=tuple(
            _source_record(value, index)
            for index, value in enumerate(
                _sequence(root["source_catalog"], "source_catalog")
            )
        ),
        jax_examples=tuple(
            _example_record(value, index)
            for index, value in enumerate(
                _sequence(root["jax_examples"], "jax_examples")
            )
        ),
    )
    _validate_v3_ownership(manifest, repo_root)
    return manifest


def _parse_parity_v2_document(
    document: object,
    *,
    examples_manifest: JaxExamplesManifestV3,
    repo_root: Path,
) -> ParityManifest:
    relationships = parse_parity_relationships_document(
        document,
        repo_root=repo_root,
        schema_version=2,
    )
    examples_by_id = {record.id: record for record in examples_manifest.jax_examples}
    expected_order = tuple(
        (source.mirror_example_id, source.source)
        for source in examples_manifest.source_catalog
        if source.mirror_example_id is not None
    )
    actual_order = tuple(
        (relationship.jax_example_id, relationship.native_source)
        for relationship in relationships
    )
    if actual_order != expected_order:
        raise ManifestV3ValidationError(
            "parity v2 must exactly follow one-to-one source ownership"
        )
    for relationship in relationships:
        example = examples_by_id[relationship.jax_example_id]
        if example.classification == "tutorial":
            raise ManifestV3ValidationError("tutorial cannot be parity coverage")
        if relationship.classification != "unsupported" and example.status != "ready":
            raise ManifestV3ValidationError(
                f"planned executable cannot claim parity: {example.id}"
            )
    return ParityManifest(schema_version=2, relationships=relationships)


def _schema_version(document: object, contract: str) -> int:
    root = _mapping(document, contract)
    version = root.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ContractVersionError(f"unsupported {contract} schema: {version!r}")
    return version


def load_manifest_contract_pair_documents(
    examples_document: object,
    parity_document: object,
    *,
    repo_root: Path,
) -> ManifestContractPair:
    """Atomically accept legacy (v2/v1) or canonical (v3/v2), never a mix."""
    examples_version = _schema_version(examples_document, "example")
    parity_version = _schema_version(parity_document, "parity")
    if examples_version not in {2, 3}:
        raise ContractVersionError(f"unsupported example schema: {examples_version!r}")
    if parity_version not in {1, 2}:
        raise ContractVersionError(f"unsupported parity schema: {parity_version!r}")
    if (examples_version, parity_version) == (2, 1):
        examples = parse_manifest_document(
            examples_document,
            repo_root=repo_root,
            warn_legacy=False,
        )
        parity = parse_parity_manifest_document(
            parity_document,
            examples_manifest=examples,
            repo_root=repo_root,
        )
        return ManifestContractPair((2, 1), True, examples, parity)
    if (examples_version, parity_version) == (3, 2):
        examples_v3 = parse_examples_v3_document(
            examples_document,
            repo_root=repo_root,
        )
        parity_v2 = _parse_parity_v2_document(
            parity_document,
            examples_manifest=examples_v3,
            repo_root=repo_root,
        )
        return ManifestContractPair((3, 2), False, examples_v3, parity_v2)
    raise ContractVersionError(
        "mixed manifest versions are forbidden: "
        f"examples={examples_version}, parity={parity_version}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _stable_mirror_id(source: str) -> str:
    return "native-" + PurePosixPath(source).stem.replace("_", "-").lower()


def _ordered_union(groups: tuple[tuple[str, ...], ...]) -> list[str]:
    return sorted({entry for group in groups for entry in group})


def _tutorial_payload(example: JaxExampleRecord) -> dict[str, object]:
    scope = "full_workflow" if example.execution_kind == "pure" else "jax_region"
    return {
        "id": example.id,
        "path": example.path,
        "status": example.status,
        "tier": example.tier,
        "classification": "tutorial",
        "teaching_kind": (
            "combined" if len(example.inspired_by) > 1 else "compatibility"
        ),
        "jax_surfaces": list(example.jax_surfaces),
        "host_boundaries": list(example.host_boundaries),
        "extras": list(example.extras),
        "smoke_args": list(example.smoke_args),
        "correctness_tests": list(example.correctness_tests),
        "supported_device_scopes": {device: scope for device in example.devices},
    }


def _one_to_one_payload(
    source: str,
    target_classification: str,
    covering_examples: tuple[JaxExampleRecord, ...],
) -> dict[str, object]:
    if not covering_examples:
        raise ManifestV3ValidationError(
            f"target source has no current public JAX surface coverage: {source}"
        )
    surfaces = _ordered_union(
        tuple(example.jax_surfaces for example in covering_examples)
    )
    host_boundaries = _ordered_union(
        tuple(example.host_boundaries for example in covering_examples)
    )
    extras = _ordered_union(tuple(example.extras for example in covering_examples))
    devices = _ordered_union(tuple(example.devices for example in covering_examples))
    if target_classification == "hybrid":
        classification = "hybrid"
        scopes = {
            device: "jax_slice_only" if device == "gpu" else "host_and_jax_slice"
            for device in devices
        }
    else:
        classification = "mirror" if not host_boundaries else "adapter"
        scope = "full_workflow" if classification == "mirror" else "jax_region"
        scopes = {device: scope for device in devices}
    pure_host_boundaries = [] if classification == "mirror" else host_boundaries
    relative = PurePosixPath(source)
    return {
        "id": _stable_mirror_id(source),
        "path": source,
        "status": "planned",
        "tier": relative.parts[0],
        "classification": classification,
        "teaching_kind": "one_to_one",
        "jax_surfaces": surfaces,
        "host_boundaries": pure_host_boundaries,
        "extras": extras,
        "smoke_args": [],
        "correctness_tests": [],
        "supported_device_scopes": scopes,
    }


def _inventory_rows(document: object) -> tuple[dict[str, object], ...]:
    root = _mapping(document, "inventory")
    if root.get("schema_version") != 1:
        raise ManifestV3ValidationError("unsupported inventory schema")
    rows = tuple(
        _mapping(value, f"inventory.native_sources[{index}]")
        for index, value in enumerate(
            _sequence(root.get("native_sources"), "inventory.native_sources")
        )
    )
    sources = tuple(_string(row.get("source"), "inventory source") for row in rows)
    if sources != tuple(sorted(set(sources))):
        raise ManifestV3ValidationError("inventory sources must be sorted and unique")
    return rows


def _source_payload(
    inventory: dict[str, object], mirror_id: str | None
) -> dict[str, object]:
    source = _string(inventory.get("source"), "inventory source")
    target = _enum(
        inventory.get("recommended_target_classification"),
        frozenset({"mirror", "hybrid", "blocked", "not_applicable"}),
        f"inventory target for {source}",
    )
    reason = _string(inventory.get("reason"), f"inventory reason for {source}")
    reconsideration = _optional_string(
        inventory.get("reconsideration_condition"),
        f"inventory reconsideration for {source}",
    )
    disposition = "eligible" if target == "mirror" else target
    return {
        "source": source,
        "disposition": disposition,
        "port_status": (
            "planned"
            if target in {"mirror", "hybrid"}
            else "blocked"
            if target == "blocked"
            else "not_applicable"
        ),
        "reason": reason,
        "blocker": reason if target == "blocked" else None,
        "reconsideration_condition": (
            reconsideration if target in {"blocked", "not_applicable"} else None
        ),
        "dependencies": inventory.get("runtime_dependencies"),
        "mirror_example_id": mirror_id,
    }


def _pending_parity_payload(
    source: str,
    mirror_id: str,
    legacy_relationship: ParityRelationship | None,
) -> dict[str, object]:
    omitted = (
        sorted(
            set(legacy_relationship.workflow_stages)
            | set(legacy_relationship.omitted_scientific_stages)
        )
        if legacy_relationship is not None
        else ["complete_native_workflow"]
    )
    if not omitted:
        omitted = ["complete_native_workflow"]
    return {
        "case_id": None,
        "jax_example_id": mirror_id,
        "native_source": source,
        "classification": "unsupported",
        "classification_reason": (
            "The exact-name mirror is planned and has no source-owned "
            "RED-GREEN-REFACTOR parity receipt yet."
        ),
        "scale_tier": "not_applicable",
        "oracle_kind": (
            legacy_relationship.oracle_kind
            if legacy_relationship is not None
            else "pending_native_oracle"
        ),
        "cost_tier": (
            legacy_relationship.cost_tier
            if legacy_relationship is not None
            else "scheduled"
        ),
        "workflow_stages": [],
        "omitted_scientific_stages": omitted,
        "excluded_teaching_stages": (
            list(legacy_relationship.excluded_teaching_stages)
            if legacy_relationship is not None
            else []
        ),
        "comparison_routes": [],
        "correctness_tests": [],
        "blocker": "Awaiting the exact-name mirror and matched native/JAX evidence.",
    }


def _candidate_documents(
    legacy_examples: JaxExamplesManifest,
    legacy_parity: ParityManifest,
    inventory_rows: tuple[dict[str, object], ...],
) -> _CandidateDocuments:
    source_payloads: list[dict[str, object]] = []
    one_to_one_payloads: list[dict[str, object]] = []
    relationships: list[dict[str, object]] = []
    legacy_relationship_by_source = {
        relationship.native_source: relationship
        for relationship in legacy_parity.relationships
    }
    for inventory in inventory_rows:
        source = _string(inventory.get("source"), "inventory source")
        target = _string(
            inventory.get("recommended_target_classification"),
            f"inventory target for {source}",
        )
        if target in {"mirror", "hybrid"}:
            mirror_id = _stable_mirror_id(source)
            covering = tuple(
                example
                for example in legacy_examples.jax_examples
                if source in example.inspired_by
            )
            one_to_one_payloads.append(_one_to_one_payload(source, target, covering))
            relationships.append(
                _pending_parity_payload(
                    source,
                    mirror_id,
                    legacy_relationship_by_source.get(source),
                )
            )
        else:
            mirror_id = None
        source_payloads.append(_source_payload(inventory, mirror_id))
    return _CandidateDocuments(
        examples={
            "schema_version": 3,
            "source_catalog": source_payloads,
            "jax_examples": [
                *(
                    _tutorial_payload(example)
                    for example in legacy_examples.jax_examples
                ),
                *one_to_one_payloads,
            ],
        },
        parity={"schema_version": 2, "relationships": relationships},
        planned_one_to_one_count=len(one_to_one_payloads),
        relationship_count=len(relationships),
    )


def build_v3_candidates(
    *,
    examples_v2_document: object,
    parity_v1_document: object,
    inventory_document: object,
    repo_root: Path,
) -> MigrationCandidate:
    """Build validated canonical bytes without mutating either active manifest."""
    legacy_examples = parse_manifest_document(
        examples_v2_document,
        repo_root=repo_root,
        warn_legacy=False,
    )
    if legacy_examples.schema_version != 2:
        raise ManifestV3ValidationError("migration requires example schema v2")
    legacy_parity = parse_parity_manifest_document(
        parity_v1_document,
        examples_manifest=legacy_examples,
        repo_root=repo_root,
    )
    inventory_rows = _inventory_rows(inventory_document)
    if tuple(row.source for row in legacy_examples.source_catalog) != tuple(
        _string(row.get("source"), "inventory source") for row in inventory_rows
    ):
        raise ManifestV3ValidationError(
            "inventory does not exactly map the current v2 source catalog"
        )

    documents = _candidate_documents(
        legacy_examples,
        legacy_parity,
        inventory_rows,
    )
    load_manifest_contract_pair_documents(
        documents.examples,
        documents.parity,
        repo_root=repo_root,
    )
    examples_bytes = _canonical_json_bytes(documents.examples)
    parity_bytes = _canonical_json_bytes(documents.parity)
    metrics = MappingProxyType(
        {
            "legacy_tutorial_count": len(legacy_examples.jax_examples),
            "planned_one_to_one_count": documents.planned_one_to_one_count,
            "legacy_relationship_count": len(legacy_parity.relationships),
            "canonical_relationship_count": documents.relationship_count,
            "promoted_parity_claim_count": 0,
        }
    )
    return MigrationCandidate(
        examples_bytes=examples_bytes,
        parity_bytes=parity_bytes,
        examples_sha256=hashlib.sha256(examples_bytes).hexdigest(),
        parity_sha256=hashlib.sha256(parity_bytes).hexdigest(),
        semantic_diff=metrics,
    )
