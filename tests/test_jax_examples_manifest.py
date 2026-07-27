from __future__ import annotations

import json
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import examples.jax._manifest as manifest_contract
from examples.jax._manifest import (
    ManifestValidationError,
    derive_source_coverage,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "manifest.json"
NATIVE_TIERS = {
    "1_Simple",
    "2_Intermediate",
    "3_Advanced",
    "stellarator_benchmarks",
}


def _tracked_native_examples() -> set[str]:
    examples_root = REPO_ROOT / "examples"
    return {
        path.relative_to(examples_root).as_posix()
        for tier in NATIVE_TIERS
        for path in (examples_root / tier).glob("*.py")
    }


def _manifest_document() -> dict[str, object]:
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _write_manifest(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _v2_document(document: dict[str, object]) -> dict[str, object]:
    candidate = deepcopy(document)
    candidate["schema_version"] = 2
    for record in _jax_records(candidate):
        lanes = record.pop("lanes")
        assert isinstance(lanes, list)
        record["devices"] = [
            {"cpu-smoke": "cpu", "gpu-strict": "gpu"}[str(lane)] for lane in lanes
        ]
    return candidate


def _source_records(document: dict[str, object]) -> list[dict[str, object]]:
    records = document["source_catalog"]
    assert isinstance(records, list)
    assert all(isinstance(record, dict) for record in records)
    return records


def _jax_records(document: dict[str, object]) -> list[dict[str, object]]:
    records = document["jax_examples"]
    assert isinstance(records, list)
    assert all(isinstance(record, dict) for record in records)
    return records


def test_source_catalog_exactly_matches_native_python_examples() -> None:
    manifest = load_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)

    assert {record.source for record in manifest.source_catalog} == (
        _tracked_native_examples()
    )
    assert len(manifest.source_catalog) == 51


def test_dual_reader_normalizes_absent_v1_and_explicit_v2(
    tmp_path: Path,
) -> None:
    v1_document = _manifest_document()
    with pytest.warns(FutureWarning, match="manifest schema v1"):
        v1 = load_manifest(_write_manifest(tmp_path, v1_document), repo_root=REPO_ROOT)
    v2 = load_manifest(
        _write_manifest(tmp_path, _v2_document(v1_document)),
        repo_root=REPO_ROOT,
    )

    assert v1.schema_version == 1
    assert v1.used_legacy_manifest_adapter is True
    assert v2.schema_version == 2
    assert v2.used_legacy_manifest_adapter is False
    assert v1.source_catalog == v2.source_catalog
    assert v1.jax_examples == v2.jax_examples
    assert all(
        example.devices == ("cpu", "gpu")
        for example in v2.jax_examples
        if example.status == "ready"
    )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    (
        ("explicit_v1", "schema_version"),
        ("unknown_version", "unsupported manifest schema"),
        ("mixed_fields", "lanes.*devices|devices.*lanes"),
        ("per_example_intents", "intents"),
    ),
)
def test_versioned_manifest_rejects_ambiguous_contracts(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    document = _v2_document(_manifest_document())
    first = _jax_records(document)[0]
    if mutation == "explicit_v1":
        document["schema_version"] = 1
    elif mutation == "unknown_version":
        document["schema_version"] = 99
    elif mutation == "mixed_fields":
        first["lanes"] = ["cpu-smoke", "gpu-strict"]
    elif mutation == "per_example_intents":
        first["intents"] = ["fast", "parity"]
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ManifestValidationError, match=expected_message):
        load_manifest(_write_manifest(tmp_path, document), repo_root=REPO_ROOT)


def test_v2_candidate_bytes_are_deterministic_and_semantically_identical() -> None:
    document = _manifest_document()

    first_bytes, first_diff = manifest_contract.convert_v1_document_to_v2(
        document,
        repo_root=REPO_ROOT,
    )
    second_bytes, second_diff = manifest_contract.convert_v1_document_to_v2(
        deepcopy(document),
        repo_root=REPO_ROOT,
    )

    assert first_bytes == second_bytes
    assert first_diff == second_diff
    assert first_diff["semantic_equal"] is True
    candidate = json.loads(first_bytes)
    assert candidate["schema_version"] == 2
    assert all("devices" in record for record in candidate["jax_examples"])
    assert all("lanes" not in record for record in candidate["jax_examples"])
    assert all("intents" not in record for record in candidate["jax_examples"])


def test_manifest_semantic_diff_detects_device_capability_drift(
    tmp_path: Path,
) -> None:
    v1_document = _manifest_document()
    v2_document = _v2_document(v1_document)
    planned = next(
        record for record in _jax_records(v2_document) if record["status"] == "planned"
    )
    planned["devices"] = ["cpu"]
    with pytest.warns(FutureWarning, match="manifest schema v1"):
        v1 = load_manifest(_write_manifest(tmp_path, v1_document), repo_root=REPO_ROOT)
    v2 = load_manifest(_write_manifest(tmp_path, v2_document), repo_root=REPO_ROOT)

    semantic_diff = manifest_contract.manifest_semantic_diff(v1, v2)

    assert semantic_diff["device_capabilities_equal"] is False
    assert semantic_diff["semantic_equal"] is False


def test_manifest_migration_dry_run_does_not_modify_input(
    tmp_path: Path,
) -> None:
    input_path = _write_manifest(tmp_path, _manifest_document())
    before = input_path.read_bytes()

    completed = subprocess.run(
        (
            sys.executable,
            str(REPO_ROOT / "examples" / "jax" / "migrate_manifest.py"),
            "--input",
            str(input_path),
            "--dry-run",
        ),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert input_path.read_bytes() == before
    candidate_text = completed.stdout.split("candidate_v2:\n", 1)[1]
    assert (
        f"candidate_sha256={hashlib.sha256(candidate_text.encode()).hexdigest()}"
        in completed.stdout
    )
    assert "manifest_schema_version=1" in completed.stdout
    assert "used_legacy_manifest_adapter=true" in completed.stdout
    assert '"semantic_equal":true' in completed.stdout
    assert "compatibility_duration=one release" in completed.stdout
    assert (
        "rollback_command=git checkout -- examples/jax/manifest.json"
        in completed.stdout
    )


def test_manifest_derives_coverage_without_storing_inverse_links() -> None:
    manifest = load_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)

    coverage = derive_source_coverage(manifest)

    assert set(coverage) == _tracked_native_examples()
    assert set(coverage.values()) <= {"planned", "covered", "deferred"}
    assert any(state == "planned" for state in coverage.values())
    assert any(state == "deferred" for state in coverage.values())


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("duplicate_source", "duplicate source path"),
        ("missing_source", "source catalog does not match"),
        ("invalid_disposition", "invalid disposition"),
        ("stored_inverse", "unexpected source fields"),
        ("candidate_reason", "candidate must not define deferred_reason"),
        ("deferred_without_reason", "deferred source requires deferred_reason"),
        ("unlinked_candidate", "candidate source is not linked"),
        ("linked_deferred", "deferred source must not be linked"),
        ("invalid_inspiration", "unknown inspiration source"),
        ("pure_with_boundary", "pure example must not declare host boundaries"),
        ("adapter_without_boundary", "adapter example requires host boundaries"),
        ("ready_without_cpu_lane", "ready example requires cpu-smoke lane"),
        ("ready_without_gpu_lane", "ready example requires gpu-strict lane"),
        ("ready_without_test", "ready example requires correctness tests"),
        ("ready_without_file", "ready example path does not exist"),
    ],
)
def test_manifest_rejects_invalid_contracts(
    tmp_path: Path, mutation: str, expected_message: str
) -> None:
    document = deepcopy(_manifest_document())
    sources = _source_records(document)
    examples = _jax_records(document)
    candidate = next(
        record for record in sources if record["disposition"] == "candidate"
    )
    deferred = next(record for record in sources if record["disposition"] == "deferred")
    planned = next(record for record in examples if record["status"] == "planned")

    if mutation == "duplicate_source":
        sources.append(deepcopy(sources[0]))
    elif mutation == "missing_source":
        sources.pop()
    elif mutation == "invalid_disposition":
        candidate["disposition"] = "ready"
    elif mutation == "stored_inverse":
        candidate["jax_example_ids"] = [planned["id"]]
    elif mutation == "candidate_reason":
        candidate["deferred_reason"] = "should not be present"
    elif mutation == "deferred_without_reason":
        deferred.pop("deferred_reason")
    elif mutation == "unlinked_candidate":
        source = next(
            record["source"]
            for record in sources
            if record["disposition"] == "candidate"
            and any(
                record["source"] in example["inspired_by"]
                and len(example["inspired_by"]) > 1
                for example in examples
            )
        )
        for example in examples:
            inspired_by = example["inspired_by"]
            assert isinstance(inspired_by, list)
            example["inspired_by"] = [item for item in inspired_by if item != source]
    elif mutation == "linked_deferred":
        inspired_by = planned["inspired_by"]
        assert isinstance(inspired_by, list)
        inspired_by.append(deferred["source"])
    elif mutation == "invalid_inspiration":
        planned["inspired_by"] = ["1_Simple/does_not_exist.py"]
    elif mutation == "pure_with_boundary":
        planned["execution_kind"] = "pure"
        planned["host_boundaries"] = ["native setup"]
    elif mutation == "adapter_without_boundary":
        planned["execution_kind"] = "adapter"
        planned["host_boundaries"] = []
    elif mutation == "ready_without_cpu_lane":
        planned["status"] = "ready"
        planned["lanes"] = ["gpu-strict"]
    elif mutation == "ready_without_gpu_lane":
        planned["status"] = "ready"
        planned["lanes"] = ["cpu-smoke"]
    elif mutation == "ready_without_test":
        planned["status"] = "ready"
        planned["correctness_tests"] = []
    elif mutation == "ready_without_file":
        planned["status"] = "ready"
        planned["path"] = f"{planned['tier']}/not_present.py"
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ManifestValidationError, match=expected_message):
        load_manifest(_write_manifest(tmp_path, document), repo_root=REPO_ROOT)


def test_ready_examples_are_public_jax_workflows_not_forwarders() -> None:
    manifest = load_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)

    for example in manifest.jax_examples:
        if example.status != "ready":
            continue
        source = (REPO_ROOT / "examples" / "jax" / example.path).read_text(
            encoding="utf-8"
        )
        assert "simsopt_jax" in source
        assert "runpy" not in source
        assert "examples.1_Simple" not in source
        assert "examples.2_Intermediate" not in source
        assert "examples.3_Advanced" not in source


def test_jax_workflow_reaches_examples_from_both_events_and_existing_jobs() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "jax_smoke.yml").read_text(
        encoding="utf-8"
    )
    push_section, pull_request_and_jobs = workflow.split("  pull_request:", maxsplit=1)
    pull_request_section, jobs = pull_request_and_jobs.split("jobs:", maxsplit=1)
    public_integration = jobs.split("  jax-public-integration:", maxsplit=1)[1].split(
        "  jax-gpu-strict-purity:", maxsplit=1
    )[0]
    gpu_strict = jobs.split("  jax-gpu-strict-purity:", maxsplit=1)[1].split(
        "  jax-private-optimizer:", maxsplit=1
    )[0]

    assert "'examples/jax/**'" in push_section
    assert "'examples/jax/**'" in pull_request_section
    assert "python examples/jax/run_examples.py --device cpu" in public_integration
    assert (
        "python examples/jax/run_examples.py --device cpu --intent parity"
        in public_integration
    )
    assert "python examples/jax/run_examples.py --device gpu" in gpu_strict
    assert (
        "python examples/jax/run_examples.py --device gpu --intent parity" in gpu_strict
    )
    assert "run_examples.py --lane" not in public_integration
    assert "run_examples.py --lane" not in gpu_strict
