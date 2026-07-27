"""Typed ownership boundary for native/JAX example parity policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from examples.jax._manifest import JaxExamplesManifest

Classification = Literal["full", "reduced", "unsupported"]
ScaleTier = Literal["bounded", "native_default", "not_applicable"]
Phase = Literal["initial", "final"]
LanePair = Literal[
    "native-cpu:jax-cpu",
    "native-cpu:jax-gpu",
    "jax-cpu:jax-gpu",
]

CLASSIFICATIONS = frozenset({"full", "reduced", "unsupported"})
SCALE_TIERS = frozenset({"bounded", "native_default", "not_applicable"})
PHASES = frozenset({"initial", "final"})
LANE_PAIRS = frozenset({"native-cpu:jax-cpu", "native-cpu:jax-gpu", "jax-cpu:jax-gpu"})
COMPARATORS = frozenset({"allclose", "exact", "equivalent"})
ROOT_FIELDS = frozenset({"schema_version", "relationships"})
RELATIONSHIP_FIELDS = frozenset(
    {
        "case_id",
        "jax_example_id",
        "native_source",
        "classification",
        "classification_reason",
        "scale_tier",
        "oracle_kind",
        "cost_tier",
        "workflow_stages",
        "omitted_scientific_stages",
        "excluded_teaching_stages",
        "comparison_routes",
        "correctness_tests",
        "blocker",
    }
)
ROUTE_FIELDS = frozenset(
    {"phase", "observable", "lane_pair", "applicable", "comparator", "tolerance_bucket"}
)


class ParityManifestValidationError(ValueError):
    """The parity manifest violates its fail-closed integrity contract."""


@dataclass(frozen=True)
class ComparisonRoute:
    phase: Phase
    observable: str
    lane_pair: LanePair
    applicable: bool
    comparator: str
    tolerance_bucket: str


@dataclass(frozen=True)
class ParityRelationship:
    case_id: str | None
    jax_example_id: str
    native_source: str
    classification: Classification
    classification_reason: str
    scale_tier: ScaleTier
    oracle_kind: str
    cost_tier: str
    workflow_stages: tuple[str, ...]
    omitted_scientific_stages: tuple[str, ...]
    excluded_teaching_stages: tuple[str, ...]
    comparison_routes: tuple[ComparisonRoute, ...]
    correctness_tests: tuple[str, ...]
    blocker: str | None


@dataclass(frozen=True)
class ParityManifest:
    schema_version: int
    relationships: tuple[ParityRelationship, ...]


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ParityManifestValidationError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ParityManifestValidationError(f"{context} must be a JSON array")
    return list(value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityManifestValidationError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _strings(value: object, context: str) -> tuple[str, ...]:
    values = tuple(
        _string(item, f"{context} item") for item in _sequence(value, context)
    )
    if len(values) != len(set(values)):
        raise ParityManifestValidationError(f"{context} contains duplicates")
    return values


def _route(value: object, context: str) -> ComparisonRoute:
    route = _mapping(value, context)
    unexpected = set(route) - ROUTE_FIELDS
    missing = ROUTE_FIELDS - set(route)
    if unexpected or missing:
        raise ParityManifestValidationError(
            f"unexpected comparison route fields in {context}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    phase_value = _string(route["phase"], f"{context}.phase")
    lane_pair_value = _string(route["lane_pair"], f"{context}.lane_pair")
    comparator = _string(route["comparator"], f"{context}.comparator")
    if phase_value not in PHASES:
        raise ParityManifestValidationError(f"invalid comparison phase: {phase_value}")
    if lane_pair_value not in LANE_PAIRS:
        raise ParityManifestValidationError(f"invalid lane pair: {lane_pair_value}")
    if comparator not in COMPARATORS:
        raise ParityManifestValidationError(f"invalid comparator: {comparator}")
    applicable = route["applicable"]
    if not isinstance(applicable, bool):
        raise ParityManifestValidationError(f"{context}.applicable must be boolean")
    return ComparisonRoute(
        phase="initial" if phase_value == "initial" else "final",
        observable=_string(route["observable"], f"{context}.observable"),
        lane_pair=(
            "native-cpu:jax-cpu"
            if lane_pair_value == "native-cpu:jax-cpu"
            else "native-cpu:jax-gpu"
            if lane_pair_value == "native-cpu:jax-gpu"
            else "jax-cpu:jax-gpu"
        ),
        applicable=applicable,
        comparator=comparator,
        tolerance_bucket=_string(
            route["tolerance_bucket"], f"{context}.tolerance_bucket"
        ),
    )


def _relationship(value: object, index: int, repo_root: Path) -> ParityRelationship:
    context = f"relationships[{index}]"
    record = _mapping(value, context)
    unexpected = set(record) - RELATIONSHIP_FIELDS
    missing = RELATIONSHIP_FIELDS - set(record)
    if unexpected or missing:
        raise ParityManifestValidationError(
            f"invalid parity relationship fields in {context}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    classification_value = _string(
        record["classification"], f"{context}.classification"
    )
    scale_value = _string(record["scale_tier"], f"{context}.scale_tier")
    if classification_value not in CLASSIFICATIONS:
        raise ParityManifestValidationError(
            f"invalid parity classification: {classification_value}"
        )
    if scale_value not in SCALE_TIERS:
        raise ParityManifestValidationError(f"invalid scale tier: {scale_value}")
    case_id = _optional_string(record["case_id"], f"{context}.case_id")
    blocker = _optional_string(record["blocker"], f"{context}.blocker")
    workflow_stages = _strings(record["workflow_stages"], f"{context}.workflow_stages")
    omitted_scientific_stages = _strings(
        record["omitted_scientific_stages"],
        f"{context}.omitted_scientific_stages",
    )
    excluded_teaching_stages = _strings(
        record["excluded_teaching_stages"],
        f"{context}.excluded_teaching_stages",
    )
    routes = tuple(
        _route(route, f"{context}.comparison_routes[{route_index}]")
        for route_index, route in enumerate(
            _sequence(record["comparison_routes"], f"{context}.comparison_routes")
        )
    )
    route_keys = tuple(
        (route.phase, route.observable, route.lane_pair) for route in routes
    )
    if len(route_keys) != len(set(route_keys)):
        raise ParityManifestValidationError("duplicate comparison route")
    if classification_value == "unsupported":
        if case_id is not None:
            raise ParityManifestValidationError(
                "unsupported relationship must not define case_id"
            )
        if blocker is None:
            raise ParityManifestValidationError(
                "unsupported relationship requires blocker"
            )
        if routes:
            raise ParityManifestValidationError(
                "unsupported relationship must not define comparison routes"
            )
        if workflow_stages:
            raise ParityManifestValidationError(
                "unsupported relationship must not define completed workflow stages"
            )
        if not omitted_scientific_stages:
            raise ParityManifestValidationError(
                "unsupported relationship requires omitted scientific stages"
            )
    else:
        if case_id is None:
            raise ParityManifestValidationError(
                f"{classification_value} relationship requires case_id"
            )
        if blocker is not None:
            raise ParityManifestValidationError(
                f"{classification_value} relationship must not define blocker"
            )
        if not routes:
            raise ParityManifestValidationError(
                f"{classification_value} relationship requires comparison routes"
            )
        if not workflow_stages:
            raise ParityManifestValidationError(
                f"{classification_value} relationship requires workflow stages"
            )
        if classification_value == "full" and omitted_scientific_stages:
            raise ParityManifestValidationError(
                "full relationship must not omit scientific stages"
            )
        if classification_value == "reduced" and not omitted_scientific_stages:
            raise ParityManifestValidationError(
                "reduced relationship requires omitted scientific stages"
            )
    correctness_tests = _strings(
        record["correctness_tests"], f"{context}.correctness_tests"
    )
    for test_path in correctness_tests:
        if not (repo_root / test_path).is_file():
            raise ParityManifestValidationError(
                f"correctness test does not exist: {test_path}"
            )
    return ParityRelationship(
        case_id=case_id,
        jax_example_id=_string(record["jax_example_id"], f"{context}.jax_example_id"),
        native_source=_string(record["native_source"], f"{context}.native_source"),
        classification=(
            "full"
            if classification_value == "full"
            else "reduced"
            if classification_value == "reduced"
            else "unsupported"
        ),
        classification_reason=_string(
            record["classification_reason"], f"{context}.classification_reason"
        ),
        scale_tier=(
            "bounded"
            if scale_value == "bounded"
            else "native_default"
            if scale_value == "native_default"
            else "not_applicable"
        ),
        oracle_kind=_string(record["oracle_kind"], f"{context}.oracle_kind"),
        cost_tier=_string(record["cost_tier"], f"{context}.cost_tier"),
        workflow_stages=workflow_stages,
        omitted_scientific_stages=omitted_scientific_stages,
        excluded_teaching_stages=excluded_teaching_stages,
        comparison_routes=routes,
        correctness_tests=correctness_tests,
        blocker=blocker,
    )


def load_parity_manifest(
    path: Path,
    *,
    examples_manifest: JaxExamplesManifest,
    repo_root: Path,
) -> ParityManifest:
    """Load parity policy and validate it against ready example lineage."""
    document = _mapping(json.loads(path.read_text(encoding="utf-8")), "root")
    unexpected = set(document) - ROOT_FIELDS
    missing = ROOT_FIELDS - set(document)
    if unexpected or missing:
        raise ParityManifestValidationError(
            f"invalid parity manifest root fields: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    schema_version = document["schema_version"]
    if schema_version != 1:
        raise ParityManifestValidationError(
            f"unsupported parity schema version: {schema_version!r}"
        )
    relationships = tuple(
        _relationship(value, index, repo_root)
        for index, value in enumerate(
            _sequence(document["relationships"], "relationships")
        )
    )
    relationship_keys = tuple(
        (relationship.jax_example_id, relationship.native_source)
        for relationship in relationships
    )
    if len(relationship_keys) != len(set(relationship_keys)):
        raise ParityManifestValidationError("duplicate parity relationship")
    case_ids = tuple(
        relationship.case_id
        for relationship in relationships
        if relationship.case_id is not None
    )
    if len(case_ids) != len(set(case_ids)):
        raise ParityManifestValidationError("duplicate parity case_id")
    ready_examples = {
        example.id: example
        for example in examples_manifest.jax_examples
        if example.status == "ready"
    }
    expected_order = tuple(
        (example_id, native_source)
        for example_id, example in ready_examples.items()
        for native_source in example.inspired_by
    )
    expected_keys = set(expected_order)
    for relationship in relationships:
        example = ready_examples.get(relationship.jax_example_id)
        if example is None:
            raise ParityManifestValidationError(
                f"unknown ready JAX example: {relationship.jax_example_id}"
            )
        if relationship.native_source not in example.inspired_by:
            raise ParityManifestValidationError(
                f"{relationship.native_source} is not inspired_by "
                f"{relationship.jax_example_id}"
            )
    actual_keys = set(relationship_keys)
    if actual_keys != expected_keys:
        raise ParityManifestValidationError(
            "parity relationships do not exactly cover ready inspired_by lineage: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    if relationship_keys != expected_order:
        raise ParityManifestValidationError(
            "parity relationships must follow deterministic ready-lineage order"
        )
    return ParityManifest(schema_version=1, relationships=relationships)
