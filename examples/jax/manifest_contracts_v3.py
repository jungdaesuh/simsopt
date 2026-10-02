"""Typed ownership boundary for the JAX example manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Mapping

from examples.jax.official_source_catalog import (
    OFFICIAL_NATIVE_EXAMPLE_SOURCES,
)
from examples.jax.outer_optimizer_policy import (
    OuterOptimizerPolicy,
    parse_outer_optimizer_policy,
)

SourceDispositionV3 = Literal["eligible", "hybrid", "blocked", "not_applicable"]
PortStatus = Literal["planned", "ready", "blocked", "not_applicable"]
ExampleStatus = Literal["planned", "ready"]
ExampleClassification = Literal["mirror", "adapter", "hybrid", "tutorial"]
TeachingKind = Literal["one_to_one", "combined"]
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
_TEACHING_KINDS = frozenset({"one_to_one", "combined"})
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
    outer_optimizer_policy: OuterOptimizerPolicy | None = None

    @property
    def device_scopes(self) -> Mapping[str, DeviceScope]:
        """Expose immutable device-to-scientific-scope ownership."""
        return MappingProxyType(dict(self.supported_device_scopes))


@dataclass(frozen=True)
class JaxExamplesManifestV3:
    """Official upstream coverage and the JAX examples that mirror it."""

    source_catalog: tuple[SourceRecordV3, ...]
    jax_examples: tuple[JaxExampleRecordV3, ...]
    schema_version: Literal[3] = 3


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
    _exact_fields(
        record,
        _EXAMPLE_FIELDS
        | ({"outer_optimizer_policy"} if "outer_optimizer_policy" in record else set()),
        "executable",
    )
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
        teaching_kind="one_to_one" if teaching_value == "one_to_one" else "combined",
        jax_surfaces=_strings(record["jax_surfaces"], f"{context}.jax_surfaces"),
        host_boundaries=host_boundaries,
        extras=_strings(record["extras"], f"{context}.extras"),
        smoke_args=_strings(record["smoke_args"], f"{context}.smoke_args"),
        correctness_tests=_strings(
            record["correctness_tests"], f"{context}.correctness_tests"
        ),
        supported_device_scopes=scopes,
        outer_optimizer_policy=parse_outer_optimizer_policy(
            record.get("outer_optimizer_policy"),
            example_id=_string(record["id"], f"{context}.id"),
            example_path=path,
            ready=status_value == "ready",
        ),
    )


def _validate_v3_ownership(manifest: JaxExamplesManifestV3, repo_root: Path) -> None:
    sources = tuple(record.source for record in manifest.source_catalog)
    if sources != OFFICIAL_NATIVE_EXAMPLE_SOURCES:
        raise ManifestV3ValidationError(
            "source catalog does not match the pinned official upstream inventory"
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
    required_fields = frozenset({"schema_version", "source_catalog", "jax_examples"})
    unexpected = set(root) - required_fields
    missing = required_fields - set(root)
    if unexpected or missing:
        raise ManifestV3ValidationError(
            "invalid manifest root fields: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
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
