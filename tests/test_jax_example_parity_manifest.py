from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from examples.jax.manifest_contracts_v3 import (
    ManifestContractPair,
    ManifestV3ValidationError,
    load_manifest_contract_pair_documents,
)
from examples.jax.parity._manifest import (
    ParityManifestValidationError,
    ParityRelationship,
    parse_v2_parity_relationship_groups_document,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "manifest.json"
PARITY_MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"


def _json_document(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _document() -> dict[str, object]:
    return _json_document(PARITY_MANIFEST_PATH)


def _load_pair(parity_document: dict[str, object]) -> ManifestContractPair:
    return load_manifest_contract_pair_documents(
        _json_document(EXAMPLES_MANIFEST_PATH),
        parity_document,
        repo_root=REPO_ROOT,
    )


def _parse(document: dict[str, object]) -> tuple[ParityRelationship, ...]:
    official, experimental = parse_v2_parity_relationship_groups_document(
        document, repo_root=REPO_ROOT
    )
    return official + experimental


def _relationships(document: dict[str, object]) -> list[dict[str, object]]:
    relationships = document["relationships"]
    assert isinstance(relationships, list)
    assert all(isinstance(item, dict) for item in relationships)
    return relationships


def test_parity_manifest_covers_every_owned_mirror_exactly_once() -> None:
    pair = _load_pair(_document())

    expected = {
        (source.mirror_example_id, source.source)
        for source in pair.examples.all_sources
        if source.mirror_example_id is not None
    }
    actual = {
        (relationship.jax_example_id, relationship.native_source)
        for relationship in pair.parity.all_relationships
    }

    assert actual == expected
    assert len(actual) == len(pair.parity.all_relationships)


def test_parity_manifest_declares_scientific_workflow_stage_coverage() -> None:
    document = _document()
    relationships = _relationships(document)

    for relationship in relationships:
        assert "workflow_stages" in relationship
        assert "omitted_scientific_stages" in relationship
        assert "excluded_teaching_stages" in relationship


def test_additional_scale_resolves_full_routes_without_changing_base() -> None:
    document = _document()
    relationship = next(
        item
        for item in _relationships(document)
        if item["case_id"] == "native-just-a-quadratic"
    )
    original_routes = deepcopy(relationship["comparison_routes"])
    shipped_routes = deepcopy(original_routes)
    assert isinstance(shipped_routes, list)
    shipped_routes[0]["observable"] = "shipped_value"
    for route in shipped_routes:
        if route["observable"] == original_routes[0]["observable"]:
            route["observable"] = "shipped_value"
    relationship["scale_contracts"] = {
        "native_default": {
            "comparison_routes": shipped_routes,
            "cost_tier": "scheduled",
        }
    }

    parsed = _parse(document)
    selected = next(
        item for item in parsed if item.case_id == "native-just-a-quadratic"
    )
    assert selected.supported_scales == ("bounded", "native_default")
    assert selected.resolve_scale("bounded") is selected
    shipped = selected.resolve_scale("native_default")
    assert shipped.case_id == selected.case_id
    assert shipped.scale_tier == "native_default"
    assert selected.cost_tier == "smoke"
    assert shipped.cost_tier == "scheduled"
    assert shipped.comparison_routes != selected.comparison_routes
    assert selected.comparison_routes[0].observable != "shipped_value"
    assert shipped.comparison_routes[0].observable == "shipped_value"
    with pytest.raises(ParityManifestValidationError, match="does not declare"):
        selected.resolve_scale("not_applicable")


def test_native_default_only_relationship_keeps_single_scale_contract() -> None:
    document = _document()
    relationships = document["relationships"]
    assert isinstance(relationships, list)
    single_scale = next(
        item for item in relationships if item["case_id"] == "native-boozerqa"
    )
    single_scale["scale_tier"] = "native_default"
    single_scale["cost_tier"] = "scheduled"
    del single_scale["scale_contracts"]
    parsed = _parse(document)
    relationship = next(item for item in parsed if item.case_id == "native-boozerqa")
    assert relationship.supported_scales == ("native_default",)
    assert relationship.resolve_scale("native_default") is relationship
    with pytest.raises(ParityManifestValidationError, match="does not declare"):
        relationship.resolve_scale("bounded")


@pytest.mark.parametrize(
    ("contract", "message"),
    (
        (
            {
                "native_default": {
                    "comparison_routes": [],
                    "cost_tier": "scheduled",
                    "termination_policy": "pass",
                }
            },
            "invalid scale contract fields",
        ),
        (
            {"native_default": {"comparison_routes": [], "cost_tier": "scheduled"}},
            "additional scale requires comparison routes",
        ),
        (
            {"native_default": {"comparison_routes": []}},
            "invalid scale contract fields",
        ),
        ({"bounded": {"comparison_routes": []}}, "invalid additional scale"),
        ({"unknown": {"comparison_routes": []}}, "invalid additional scale"),
        ({}, "additional scale"),
    ),
)
def test_additional_scale_rejects_undeclared_policy_and_invalid_routes(
    contract: dict[str, object], message: str
) -> None:
    document = _document()
    relationship = next(
        item
        for item in _relationships(document)
        if item["case_id"] == "native-just-a-quadratic"
    )
    relationship["scale_contracts"] = contract
    with pytest.raises(ParityManifestValidationError, match=message):
        _parse(document)


@pytest.mark.parametrize(
    ("location", "cost_tier"),
    (
        ("base", "unreviewed"),
        ("base", "not_applicable"),
        ("additional", "unreviewed"),
        ("unsupported", "smoke"),
    ),
)
def test_parity_manifest_rejects_unknown_cost_tier_at_each_scale(
    location: str,
    cost_tier: str,
) -> None:
    document = _document()
    relationship = next(
        item
        for item in _relationships(document)
        if (
            item["classification"] == "unsupported"
            if location == "unsupported"
            else item["case_id"] == "native-wireframe-gsco-multistep"
        )
    )
    if location != "additional":
        relationship["cost_tier"] = cost_tier
    else:
        contracts = relationship["scale_contracts"]
        assert isinstance(contracts, dict)
        additional = contracts["native_default"]
        assert isinstance(additional, dict)
        additional["cost_tier"] = cost_tier

    with pytest.raises(ParityManifestValidationError, match="invalid cost tier"):
        _parse(document)


@pytest.mark.parametrize("mutation", ("missing_pair", "duplicate_pair"))
def test_additional_scale_validates_its_own_direct_route_matrix(
    mutation: str,
) -> None:
    document = _document()
    relationship = next(
        item
        for item in _relationships(document)
        if item["case_id"] == "native-just-a-quadratic"
    )
    routes = deepcopy(relationship["comparison_routes"])
    assert isinstance(routes, list)
    if mutation == "missing_pair":
        routes.pop(0)
    else:
        routes.append(deepcopy(routes[0]))
    relationship["scale_contracts"] = {
        "native_default": {"comparison_routes": routes, "cost_tier": "scheduled"}
    }
    with pytest.raises(
        ParityManifestValidationError,
        match="complete direct lane-pair matrix|duplicate comparison route",
    ):
        _parse(document)


def test_unsupported_relationship_cannot_gain_executable_scale() -> None:
    document = _document()
    relationship = next(
        item
        for item in _relationships(document)
        if item["classification"] == "unsupported"
    )
    relationship["scale_contracts"] = {"bounded": {"comparison_routes": []}}
    with pytest.raises(ParityManifestValidationError, match="executable base"):
        _parse(document)


@pytest.mark.parametrize(
    "phase",
    (
        "area",
        "conservation",
        "construction",
        "first",
        "flux",
        "history",
        "interpolation",
        "poincare",
        "second",
        "taylor",
        "toroidal_flux",
        "volume",
    ),
)
def test_parity_manifest_accepts_named_scientific_phases(phase: str) -> None:
    document = deepcopy(_document())
    relationship = next(
        item
        for item in _relationships(document)
        if item["classification"] != "unsupported"
    )
    routes = relationship["comparison_routes"]
    assert isinstance(routes, list)
    first_route = routes[0]
    assert isinstance(first_route, dict)
    renamed_key = (first_route["phase"], first_route["observable"])
    renamed = 0
    for route in routes:
        assert isinstance(route, dict)
        if (route["phase"], route["observable"]) == renamed_key:
            route["phase"] = phase
            renamed += 1
    assert renamed == 3, "the renamed key must keep its complete lane-pair matrix"

    _load_pair(document)


def test_parity_workflows_reach_cpu_and_strict_gpu_without_case_duplication() -> None:
    smoke_workflow = (REPO_ROOT / ".github" / "workflows" / "jax_smoke.yml").read_text(
        encoding="utf-8"
    )
    scheduled_workflow = (
        REPO_ROOT / ".github" / "workflows" / "jax_gpu_parity.yml"
    ).read_text(encoding="utf-8")
    smoke_jobs = smoke_workflow.split("jobs:", maxsplit=1)[1]
    public_integration = smoke_jobs.split("  jax-public-integration:", maxsplit=1)[
        1
    ].split("  jax-private-optimizer:", maxsplit=1)[0]
    # The self-hosted strict GPU job lives in the dispatch/schedule workflow, the
    # last job there, so the pull_request-triggered workflow never targets it.
    scheduled_workflow, gpu_strict = scheduled_workflow.split(
        "  jax-gpu-strict-purity:", maxsplit=1
    )
    assert "self-hosted" not in smoke_workflow
    assert "pull_request" not in scheduled_workflow.split("jobs:", maxsplit=1)[0]

    for job in (public_integration, gpu_strict):
        assert "examples/jax/run_parity.py" in job
        assert "--case all-applicable" in job
        assert job.count("--case ") == job.count("--case all-applicable")
        assert "actions/upload-artifact@v4" in job
        assert "retention-days:" in job
    assert "--lanes native-cpu,jax-cpu" in public_integration
    assert "--lanes native-cpu,jax-cpu,jax-gpu" in gpu_strict
    assert "run_examples.py --device cpu" in public_integration
    assert "run_examples.py --device cpu --intent parity" in public_integration
    assert "run_examples.py --device gpu" in gpu_strict
    assert "run_examples.py --device gpu --intent parity" in gpu_strict

    assert "workflow_dispatch:" in scheduled_workflow
    assert "schedule:" in scheduled_workflow
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in scheduled_workflow
    assert "JAX_TRANSFER_GUARD: disallow" in scheduled_workflow
    assert "--case all-applicable" in scheduled_workflow
    assert "--lanes native-cpu,jax-cpu,jax-gpu" in scheduled_workflow


_OWNERSHIP_ORDER = "must exactly follow one-to-one source ownership"


@pytest.mark.parametrize(
    ("mutation", "expected_error", "expected_message"),
    [
        (
            "duplicate_relationship",
            ParityManifestValidationError,
            "duplicate parity relationship",
        ),
        (
            "duplicate_case_id",
            ParityManifestValidationError,
            "duplicate parity case_id",
        ),
        ("nondeterministic_order", ManifestV3ValidationError, _OWNERSHIP_ORDER),
        ("unknown_example", ManifestV3ValidationError, _OWNERSHIP_ORDER),
        ("wrong_native_source", ManifestV3ValidationError, _OWNERSHIP_ORDER),
        (
            "unsupported_with_case",
            ParityManifestValidationError,
            "unsupported relationship must not define case_id",
        ),
        (
            "unsupported_without_blocker",
            ParityManifestValidationError,
            "unsupported relationship requires blocker",
        ),
        (
            "full_without_case",
            ParityManifestValidationError,
            "full relationship requires case_id",
        ),
        (
            "hard_coded_tolerance",
            ParityManifestValidationError,
            "unexpected comparison route fields",
        ),
        ("unknown_lane_pair", ParityManifestValidationError, "invalid lane pair"),
        (
            "duplicate_route",
            ParityManifestValidationError,
            "duplicate comparison route",
        ),
        (
            "incomplete_route_matrix",
            ParityManifestValidationError,
            "complete direct lane-pair matrix",
        ),
        (
            "inconsistent_source_tolerance",
            ParityManifestValidationError,
            "source-owned tolerance must apply to every lane pair",
        ),
        (
            "missing_test_owner",
            ParityManifestValidationError,
            "correctness test does not exist",
        ),
        (
            "full_with_omitted_stage",
            ParityManifestValidationError,
            "full relationship must not omit",
        ),
        (
            "reduced_without_omitted_stage",
            ParityManifestValidationError,
            "reduced relationship requires omitted",
        ),
        (
            "unsupported_with_completed_stage",
            ParityManifestValidationError,
            "unsupported relationship must not",
        ),
    ],
)
def test_parity_manifest_rejects_invalid_contracts(
    mutation: str,
    expected_error: type[ValueError],
    expected_message: str,
) -> None:
    document = deepcopy(_document())
    relationships = _relationships(document)
    supported = next(
        item for item in relationships if item["classification"] != "unsupported"
    )
    unsupported = next(
        item for item in relationships if item["classification"] == "unsupported"
    )

    if mutation == "duplicate_relationship":
        relationships.append(deepcopy(relationships[0]))
    elif mutation == "duplicate_case_id":
        second_supported = next(
            item
            for item in relationships
            if item["classification"] != "unsupported" and item is not supported
        )
        second_supported["case_id"] = supported["case_id"]
    elif mutation == "nondeterministic_order":
        relationships[0], relationships[1] = relationships[1], relationships[0]
    elif mutation == "unknown_example":
        supported["jax_example_id"] = "not-a-ready-example"
    elif mutation == "wrong_native_source":
        supported["native_source"] = "1_Simple/logger_example.py"
    elif mutation == "unsupported_with_case":
        unsupported["case_id"] = "forbidden-case"
    elif mutation == "unsupported_without_blocker":
        unsupported["blocker"] = None
    elif mutation == "full_without_case":
        supported["classification"] = "full"
        supported["case_id"] = None
    elif mutation == "hard_coded_tolerance":
        route = supported["comparison_routes"][0]
        assert isinstance(route, dict)
        route["rtol"] = 1.0e-8
    elif mutation == "unknown_lane_pair":
        route = supported["comparison_routes"][0]
        assert isinstance(route, dict)
        route["lane_pair"] = "native-cpu:unknown"
    elif mutation == "duplicate_route":
        supported["comparison_routes"].append(
            deepcopy(supported["comparison_routes"][0])
        )
    elif mutation == "incomplete_route_matrix":
        first_route = supported["comparison_routes"][0]
        assert isinstance(first_route, dict)
        supported["comparison_routes"] = [
            route
            for route in supported["comparison_routes"]
            if not (
                isinstance(route, dict)
                and route["phase"] == first_route["phase"]
                and route["observable"] == first_route["observable"]
                and route["lane_pair"] == "native-cpu:jax-gpu"
            )
        ]
    elif mutation == "inconsistent_source_tolerance":
        first_route = supported["comparison_routes"][0]
        assert isinstance(first_route, dict)
        for route in supported["comparison_routes"]:
            assert isinstance(route, dict)
            if (
                route["phase"] == first_route["phase"]
                and route["observable"] == first_route["observable"]
            ):
                route["tolerance_bucket"] = "mirror_boozer_value"
                if route["lane_pair"] == "jax-cpu:jax-gpu":
                    route["tolerance_bucket"] = "gpu_runtime"
    elif mutation == "missing_test_owner":
        supported["correctness_tests"] = ["tests/does_not_exist.py"]
    elif mutation == "full_with_omitted_stage":
        supported["classification"] = "full"
        supported["omitted_scientific_stages"] = ["forbidden omission"]
    elif mutation == "reduced_without_omitted_stage":
        supported["classification"] = "reduced"
        supported["omitted_scientific_stages"] = []
    elif mutation == "unsupported_with_completed_stage":
        unsupported["workflow_stages"] = ["forbidden stage"]
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(expected_error, match=expected_message):
        _load_pair(document)
