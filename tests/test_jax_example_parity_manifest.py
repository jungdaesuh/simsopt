from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from examples.jax._manifest import parse_manifest_document
from examples.jax.parity._manifest import (
    ParityManifestValidationError,
    load_parity_manifest,
    parse_parity_relationships_document,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "jax_manifests" / "manifest_v2.json"
)
PARITY_MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "jax_manifests" / "parity_manifest_v1.json"
)


def _document() -> dict[str, object]:
    document = json.loads(PARITY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _active_document() -> dict[str, object]:
    document = json.loads(
        (REPO_ROOT / "examples/jax/parity_manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(document, dict)
    return document


def _examples_manifest():
    return parse_manifest_document(
        json.loads(EXAMPLES_MANIFEST_PATH.read_text(encoding="utf-8")),
        repo_root=REPO_ROOT,
        allow_historical_catalog=True,
    )


def _write_document(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "parity_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _relationships(document: dict[str, object]) -> list[dict[str, object]]:
    relationships = document["relationships"]
    assert isinstance(relationships, list)
    assert all(isinstance(item, dict) for item in relationships)
    return relationships


def test_parity_manifest_covers_every_ready_inspiration_exactly_once() -> None:
    examples_manifest = _examples_manifest()
    parity_manifest = load_parity_manifest(
        PARITY_MANIFEST_PATH,
        examples_manifest=examples_manifest,
        repo_root=REPO_ROOT,
    )

    expected = {
        (example.id, native_source)
        for example in examples_manifest.jax_examples
        if example.status == "ready"
        for native_source in example.inspired_by
    }
    actual = {
        (relationship.jax_example_id, relationship.native_source)
        for relationship in parity_manifest.relationships
    }

    assert actual == expected
    assert len(actual) == len(parity_manifest.relationships)


def test_parity_manifest_declares_scientific_workflow_stage_coverage() -> None:
    document = _document()
    relationships = _relationships(document)

    for relationship in relationships:
        assert "workflow_stages" in relationship
        assert "omitted_scientific_stages" in relationship
        assert "excluded_teaching_stages" in relationship


def test_coil_flux_relationship_routes_each_scientific_observable() -> None:
    examples_manifest = _examples_manifest()
    parity_manifest = load_parity_manifest(
        PARITY_MANIFEST_PATH,
        examples_manifest=examples_manifest,
        repo_root=REPO_ROOT,
    )
    relationship = next(
        item
        for item in parity_manifest.relationships
        if item.case_id == "coil-flux-optimization"
    )

    assert {
        (route.phase, route.observable) for route in relationship.comparison_routes
    } == {
        (phase, observable)
        for phase in ("initial", "final")
        for observable in ("parameters", "flux", "flux_gradient", "coil_length")
    }


def test_additional_scale_resolves_full_routes_without_changing_base() -> None:
    document = _active_document()
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

    parsed = parse_parity_relationships_document(
        document, repo_root=REPO_ROOT, schema_version=2
    )
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
    document = _active_document()
    relationships = document["relationships"]
    assert isinstance(relationships, list)
    single_scale = next(
        item for item in relationships if item["case_id"] == "native-boozerqa"
    )
    single_scale["scale_tier"] = "native_default"
    single_scale["cost_tier"] = "scheduled"
    del single_scale["scale_contracts"]
    parsed = parse_parity_relationships_document(
        document, repo_root=REPO_ROOT, schema_version=2
    )
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
    document = _active_document()
    relationship = next(
        item
        for item in _relationships(document)
        if item["case_id"] == "native-just-a-quadratic"
    )
    relationship["scale_contracts"] = contract
    with pytest.raises(ParityManifestValidationError, match=message):
        parse_parity_relationships_document(
            document, repo_root=REPO_ROOT, schema_version=2
        )


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
    document = _active_document()
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
        parse_parity_relationships_document(
            document, repo_root=REPO_ROOT, schema_version=2
        )


def test_legacy_manifest_rejects_scale_contract_extension() -> None:
    document = _document()
    relationship = next(
        item
        for item in _relationships(document)
        if item["classification"] != "unsupported"
    )
    relationship["scale_contracts"] = {"native_default": {"comparison_routes": []}}
    with pytest.raises(ParityManifestValidationError, match="unexpected"):
        parse_parity_relationships_document(
            document, repo_root=REPO_ROOT, schema_version=1
        )


@pytest.mark.parametrize("mutation", ("missing_pair", "duplicate_pair"))
def test_additional_scale_validates_its_own_direct_route_matrix(
    mutation: str,
) -> None:
    document = _active_document()
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
        parse_parity_relationships_document(
            document, repo_root=REPO_ROOT, schema_version=2
        )


def test_unsupported_relationship_cannot_gain_executable_scale() -> None:
    document = _active_document()
    relationship = next(
        item
        for item in _relationships(document)
        if item["classification"] == "unsupported"
    )
    relationship["scale_contracts"] = {"bounded": {"comparison_routes": []}}
    with pytest.raises(ParityManifestValidationError, match="executable base"):
        parse_parity_relationships_document(
            document, repo_root=REPO_ROOT, schema_version=2
        )


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
def test_parity_manifest_accepts_named_scientific_phases(
    tmp_path: Path,
    phase: str,
) -> None:
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
    observable = first_route["observable"]
    for route in routes:
        assert isinstance(route, dict)
        if route["phase"] == "initial" and route["observable"] == observable:
            route["phase"] = phase

    load_parity_manifest(
        _write_document(tmp_path, document),
        examples_manifest=_examples_manifest(),
        repo_root=REPO_ROOT,
    )


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
    ].split("  jax-gpu-strict-purity:", maxsplit=1)[0]
    gpu_strict = smoke_jobs.split("  jax-gpu-strict-purity:", maxsplit=1)[1].split(
        "  jax-private-optimizer:", maxsplit=1
    )[0]

    for job in (public_integration, gpu_strict):
        assert "examples/jax/run_parity.py" in job
        assert "--case all-applicable" in job
        assert "traceable-least-squares" not in job
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


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("duplicate_relationship", "duplicate parity relationship"),
        ("duplicate_case_id", "duplicate parity case_id"),
        ("nondeterministic_order", "deterministic ready-lineage order"),
        ("unknown_example", "unknown ready JAX example"),
        ("wrong_native_source", "is not inspired_by"),
        ("unsupported_with_case", "unsupported relationship must not define case_id"),
        ("unsupported_without_blocker", "unsupported relationship requires blocker"),
        ("full_without_case", "full relationship requires case_id"),
        ("hard_coded_tolerance", "unexpected comparison route fields"),
        ("unknown_lane_pair", "invalid lane pair"),
        ("duplicate_route", "duplicate comparison route"),
        ("incomplete_route_matrix", "complete direct lane-pair matrix"),
        (
            "inconsistent_source_tolerance",
            "source-owned tolerance must apply to every lane pair",
        ),
        ("missing_test_owner", "correctness test does not exist"),
        ("full_with_omitted_stage", "full relationship must not omit"),
        ("reduced_without_omitted_stage", "reduced relationship requires omitted"),
        ("unsupported_with_completed_stage", "unsupported relationship must not"),
    ],
)
def test_parity_manifest_rejects_invalid_contracts(
    tmp_path: Path,
    mutation: str,
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
        reduced = next(
            item for item in relationships if item["classification"] == "reduced"
        )
        reduced["omitted_scientific_stages"] = []
    elif mutation == "unsupported_with_completed_stage":
        unsupported["workflow_stages"] = ["forbidden stage"]
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    examples_manifest = _examples_manifest()
    with pytest.raises(ParityManifestValidationError, match=expected_message):
        load_parity_manifest(
            _write_document(tmp_path, document),
            examples_manifest=examples_manifest,
            repo_root=REPO_ROOT,
        )


def test_traceable_final_jacobian_has_all_direct_routes() -> None:
    examples_manifest = _examples_manifest()
    parity_manifest = load_parity_manifest(
        PARITY_MANIFEST_PATH,
        examples_manifest=examples_manifest,
        repo_root=REPO_ROOT,
    )
    relationship = next(
        item
        for item in parity_manifest.relationships
        if item.case_id == "traceable-least-squares"
    )

    assert {
        route.lane_pair
        for route in relationship.comparison_routes
        if route.phase == "final" and route.observable == "residual_jacobian"
    } == {
        "native-cpu:jax-cpu",
        "native-cpu:jax-gpu",
        "jax-cpu:jax-gpu",
    }


def test_surface_owns_symmetric_jacobian_invariant_routes() -> None:
    examples_manifest = _examples_manifest()
    parity_manifest = load_parity_manifest(
        PARITY_MANIFEST_PATH,
        examples_manifest=examples_manifest,
        repo_root=REPO_ROOT,
    )
    relationships = {
        item.case_id: item for item in parity_manifest.relationships if item.case_id
    }
    invariant_routes = {
        (route.phase, route.lane_pair)
        for route in relationships["surface-geometry-optimization"].comparison_routes
        if route.observable == "residual_jacobian_invariants"
    }

    assert invariant_routes == {
        (phase, lane_pair)
        for phase in ("initial", "final")
        for lane_pair in (
            "native-cpu:jax-cpu",
            "native-cpu:jax-gpu",
            "jax-cpu:jax-gpu",
        )
    }
    assert all(
        route.observable != "residual_jacobian_invariants"
        for route in relationships["traceable-least-squares"].comparison_routes
    )
