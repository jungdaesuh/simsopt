"""Typed ownership boundary for native/JAX example parity policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

Classification = Literal["full", "reduced", "unsupported"]
ScaleTier = Literal["bounded", "native_default", "not_applicable"]
CostTier = Literal["smoke", "scheduled", "not_applicable"]
Phase = Literal[
    "area",
    "conservation",
    "construction",
    "final",
    "first",
    "flux",
    "history",
    "initial",
    "interpolation",
    "poincare",
    "second",
    "taylor",
    "toroidal_flux",
    "volume",
]
LanePair = Literal[
    "native-cpu:jax-cpu",
    "native-cpu:jax-gpu",
    "jax-cpu:jax-gpu",
]

CLASSIFICATIONS = frozenset({"full", "reduced", "unsupported"})
SCALE_TIERS = frozenset({"bounded", "native_default", "not_applicable"})
COST_TIERS = frozenset({"smoke", "scheduled"})
PHASES = frozenset(
    {
        "area",
        "conservation",
        "construction",
        "final",
        "first",
        "flux",
        "history",
        "initial",
        "interpolation",
        "poincare",
        "second",
        "taylor",
        "toroidal_flux",
        "volume",
    }
)
LANE_PAIRS = frozenset({"native-cpu:jax-cpu", "native-cpu:jax-gpu", "jax-cpu:jax-gpu"})
COMPARATORS = frozenset({"allclose", "exact", "equivalent", "not_worse"})
V2_ROOT_FIELDS = frozenset({"schema_version", "relationships"})
V2_OPTIONAL_ROOT_FIELDS = frozenset({"experimental_relationships"})
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
SCALE_CONTRACT_FIELDS = frozenset({"comparison_routes", "cost_tier"})
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
class ScaleContract:
    """Complete comparison policy and declared cost for one additional scale."""

    scale_tier: ScaleTier
    comparison_routes: tuple[ComparisonRoute, ...]
    cost_tier: CostTier


@dataclass(frozen=True)
class ParityRelationship:
    case_id: str | None
    jax_example_id: str
    native_source: str
    classification: Classification
    classification_reason: str
    scale_tier: ScaleTier
    oracle_kind: str
    cost_tier: CostTier
    workflow_stages: tuple[str, ...]
    omitted_scientific_stages: tuple[str, ...]
    excluded_teaching_stages: tuple[str, ...]
    comparison_routes: tuple[ComparisonRoute, ...]
    correctness_tests: tuple[str, ...]
    blocker: str | None
    scale_contracts: tuple[ScaleContract, ...] = ()

    @property
    def supported_scales(self) -> tuple[ScaleTier, ...]:
        """List only scales explicitly declared by this relationship."""
        return (self.scale_tier,) + tuple(
            contract.scale_tier for contract in self.scale_contracts
        )

    def resolve_scale(self, scale: ScaleTier) -> ParityRelationship:
        """Return the complete policy for a declared scale; reject all others."""
        if scale == self.scale_tier:
            return self
        for contract in self.scale_contracts:
            if contract.scale_tier == scale:
                return replace(
                    self,
                    scale_tier=scale,
                    comparison_routes=contract.comparison_routes,
                    cost_tier=contract.cost_tier,
                    scale_contracts=(),
                )
        raise ParityManifestValidationError(
            f"case {self.case_id} does not declare execution scale {scale!r}"
        )


@dataclass(frozen=True)
class ParityManifest:
    """Official relationships plus separately registered local experiments."""

    schema_version: int
    relationships: tuple[ParityRelationship, ...]
    experimental_relationships: tuple[ParityRelationship, ...] = ()

    @property
    def all_relationships(self) -> tuple[ParityRelationship, ...]:
        """Return registered relationships for explicit execution and evidence audit."""
        return self.relationships + self.experimental_relationships


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


def _cost_tier(value: object, context: str, *, unsupported: bool = False) -> CostTier:
    cost = _string(value, context)
    if cost not in ({"scheduled", "not_applicable"} if unsupported else COST_TIERS):
        raise ParityManifestValidationError(f"invalid cost tier: {cost}")
    return cast(CostTier, cost)


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
        phase=cast(Phase, phase_value),
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


def _validate_complete_route_matrix(routes: tuple[ComparisonRoute, ...]) -> None:
    grouped: dict[tuple[Phase, str], set[LanePair]] = {}
    for route in routes:
        if route.applicable:
            grouped.setdefault((route.phase, route.observable), set()).add(
                route.lane_pair
            )
    for key, lane_pairs in grouped.items():
        if lane_pairs != LANE_PAIRS:
            raise ParityManifestValidationError(
                f"{key} requires a complete direct lane-pair matrix"
            )


def _validate_source_owned_tolerance_matrix(
    routes: tuple[ComparisonRoute, ...],
) -> None:
    """Keep each source-owned scientific threshold identical across lane pairs."""
    grouped: dict[tuple[Phase, str], set[str]] = {}
    for route in routes:
        if route.applicable:
            grouped.setdefault((route.phase, route.observable), set()).add(
                route.tolerance_bucket
            )
    for key, tolerance_buckets in grouped.items():
        if (
            any(bucket.startswith("mirror_") for bucket in tolerance_buckets)
            and len(tolerance_buckets) != 1
        ):
            raise ParityManifestValidationError(
                f"{key} source-owned tolerance must apply to every lane pair"
            )


def _comparison_routes(value: object, context: str) -> tuple[ComparisonRoute, ...]:
    routes = tuple(
        _route(route, f"{context}[{route_index}]")
        for route_index, route in enumerate(_sequence(value, context))
    )
    route_keys = tuple(
        (route.phase, route.observable, route.lane_pair) for route in routes
    )
    if len(route_keys) != len(set(route_keys)):
        raise ParityManifestValidationError("duplicate comparison route")
    _validate_complete_route_matrix(routes)
    _validate_source_owned_tolerance_matrix(routes)
    return routes


def _relationship(value: object, index: int, repo_root: Path) -> ParityRelationship:
    context = f"relationships[{index}]"
    record = _mapping(value, context)
    unexpected = set(record) - RELATIONSHIP_FIELDS - {"scale_contracts"}
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
    routes = _comparison_routes(
        record["comparison_routes"], f"{context}.comparison_routes"
    )
    scale_contracts: list[ScaleContract] = []
    if "scale_contracts" in record:
        contracts = _mapping(record["scale_contracts"], f"{context}.scale_contracts")
        if scale_value == "not_applicable" or not contracts:
            raise ParityManifestValidationError(
                "scale contracts require an executable base and additional scale"
            )
        for additional_scale, contract_value in contracts.items():
            if (
                additional_scale not in {"bounded", "native_default"}
                or additional_scale == scale_value
            ):
                raise ParityManifestValidationError(
                    f"invalid additional scale contract: {additional_scale}"
                )
            contract = _mapping(
                contract_value, f"{context}.scale_contracts.{additional_scale}"
            )
            if set(contract) != SCALE_CONTRACT_FIELDS:
                raise ParityManifestValidationError(
                    f"invalid scale contract fields: {sorted(set(contract))}"
                )
            additional_routes = _comparison_routes(
                contract["comparison_routes"],
                f"{context}.scale_contracts.{additional_scale}.comparison_routes",
            )
            if not additional_routes:
                raise ParityManifestValidationError(
                    "additional scale requires comparison routes"
                )
            scale_contracts.append(
                ScaleContract(
                    cast(ScaleTier, additional_scale),
                    additional_routes,
                    _cost_tier(
                        contract["cost_tier"],
                        f"{context}.scale_contracts.{additional_scale}.cost_tier",
                    ),
                )
            )
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
        cost_tier=_cost_tier(
            record["cost_tier"],
            f"{context}.cost_tier",
            unsupported=classification_value == "unsupported",
        ),
        workflow_stages=workflow_stages,
        omitted_scientific_stages=omitted_scientific_stages,
        excluded_teaching_stages=excluded_teaching_stages,
        comparison_routes=routes,
        correctness_tests=correctness_tests,
        blocker=blocker,
        scale_contracts=tuple(scale_contracts),
    )


def parse_v2_parity_relationship_groups_document(
    value: object, *, repo_root: Path
) -> tuple[tuple[ParityRelationship, ...], tuple[ParityRelationship, ...]]:
    """Parse official and local-extension v2 groups without mixing their scope."""
    document = _mapping(value, "root")
    unexpected = set(document) - V2_ROOT_FIELDS - V2_OPTIONAL_ROOT_FIELDS
    missing = V2_ROOT_FIELDS - set(document)
    if unexpected or missing:
        raise ParityManifestValidationError(
            f"invalid parity manifest root fields: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if document["schema_version"] != 2:
        raise ParityManifestValidationError(
            f"unsupported parity schema version: {document['schema_version']!r}"
        )

    def parse_group(
        relationships: object, context: str
    ) -> tuple[ParityRelationship, ...]:
        return tuple(
            _relationship(record, index, repo_root)
            for index, record in enumerate(_sequence(relationships, context))
        )

    official = parse_group(document["relationships"], "relationships")
    experimental = parse_group(
        document.get("experimental_relationships", []), "experimental_relationships"
    )
    keys = tuple(
        (relationship.jax_example_id, relationship.native_source)
        for relationship in (*official, *experimental)
    )
    if len(keys) != len(set(keys)):
        raise ParityManifestValidationError("duplicate parity relationship")
    case_ids = tuple(
        relationship.case_id
        for relationship in (*official, *experimental)
        if relationship.case_id is not None
    )
    if len(case_ids) != len(set(case_ids)):
        raise ParityManifestValidationError("duplicate parity case_id")
    return official, experimental
