from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from examples.jax._manifest import load_manifest
from examples.jax.parity._manifest import (
    ParityManifestValidationError,
    load_parity_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "manifest.json"
PARITY_MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"


def _document() -> dict[str, object]:
    document = json.loads(PARITY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


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
    examples_manifest = load_manifest(EXAMPLES_MANIFEST_PATH, repo_root=REPO_ROOT)
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
    examples_manifest = load_manifest(EXAMPLES_MANIFEST_PATH, repo_root=REPO_ROOT)
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


def test_parity_artifact_ignore_is_narrow_and_effective() -> None:
    ignore_lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ignore_lines.count(".artifacts/jax-example-parity/") == 1


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

    examples_manifest = load_manifest(EXAMPLES_MANIFEST_PATH, repo_root=REPO_ROOT)
    with pytest.raises(ParityManifestValidationError, match=expected_message):
        load_parity_manifest(
            _write_document(tmp_path, document),
            examples_manifest=examples_manifest,
            repo_root=REPO_ROOT,
        )


def test_traceable_final_jacobian_has_all_direct_routes() -> None:
    examples_manifest = load_manifest(EXAMPLES_MANIFEST_PATH, repo_root=REPO_ROOT)
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
