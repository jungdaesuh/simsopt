"""Contract tests for the frozen one-to-one native example inventory."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from examples.jax._manifest import TIERS

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "examples" / "jax" / "one_to_one_inventory.json"
MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "manifest.json"
PROBE_PATH = REPO_ROOT / "examples" / "jax" / "probe_one_to_one_inventory.py"
VMEC_BLOCKED_CANDIDATES = frozenset(
    {
        "2_Intermediate/QH_fixed_resolution_boozer.py",
        "2_Intermediate/resolution_increase_boozer.py",
        "2_Intermediate/tracing_boozer.py",
    }
)

EXPECTED_TOP_LEVEL_FIELDS = frozenset({"schema_version", "baseline", "native_sources"})
EXPECTED_ROW_FIELDS = frozenset(
    {
        "source",
        "source_sha256",
        "current_disposition",
        "runtime_dependencies",
        "current_public_jax_surface_coverage",
        "capability_probe",
        "recommended_target_classification",
        "reason",
        "evidence",
        "reconsideration_condition",
    }
)
EXPECTED_BASELINE = {
    "head": "03d8da68b053815db537cd44e66887225c751c0e",
    "manifest_schema_version": 2,
    "manifest_sha256": "2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05",
    "parity_manifest_schema_version": 1,
    "parity_manifest_sha256": "060e55339194c203263da9d5690c2ff31bd6681f5713dc2ead0ce3313e313137",
    "native_source_count": 51,
    "candidate_count": 29,
    "deferred_count": 22,
    "ready_jax_example_count": 10,
    "planned_jax_example_count": 1,
    "full_parity_count": 2,
    "reduced_parity_count": 6,
    "unsupported_parity_count": 20,
}


def _load_inventory() -> dict[str, object]:
    assert INVENTORY_PATH.is_file(), "one-to-one inventory fixture is missing"
    value = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rows(value: dict[str, object]) -> list[dict[str, object]]:
    rows = value["native_sources"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    return rows


def _tracked_native_sources() -> set[str]:
    examples_root = REPO_ROOT / "examples"
    return {
        path.relative_to(examples_root).as_posix()
        for tier in TIERS
        for path in (examples_root / tier).glob("*.py")
    }


def _catalog_classification_sets() -> dict[str, frozenset[str]]:
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    catalog = document["source_catalog"]
    assert isinstance(catalog, list)
    buckets: dict[str, set[str]] = {
        "mirror": set(),
        "hybrid": set(),
        "blocked": set(),
        "not_applicable": set(),
    }
    for record in catalog:
        assert isinstance(record, dict)
        disposition = record["disposition"]
        classification = "mirror" if disposition == "eligible" else str(disposition)
        buckets[classification].add(str(record["source"]))
    return {key: frozenset(value) for key, value in buckets.items()}


def test_inventory_freezes_exact_baseline_and_all_native_sources() -> None:
    inventory = _load_inventory()
    assert frozenset(inventory) == EXPECTED_TOP_LEVEL_FIELDS
    assert inventory["schema_version"] == 1
    baseline = inventory["baseline"]
    assert isinstance(baseline, dict)
    for key, expected in EXPECTED_BASELINE.items():
        assert baseline[key] == expected

    rows = _rows(inventory)
    tracked_sources = _tracked_native_sources()
    assert len(rows) == len(tracked_sources)
    assert all(isinstance(row["source"], str) for row in rows)
    sources = [str(row["source"]) for row in rows]
    assert len(sources) == len(set(sources))
    assert sources == sorted(sources)
    assert set(sources) == tracked_sources


def test_inventory_classifies_every_source_without_silently_shrinking_scope() -> None:
    rows = _rows(_load_inventory())
    by_source = {str(row["source"]): row for row in rows}
    expected = _catalog_classification_sets()
    assert set(by_source) == set().union(*expected.values())
    for classification, sources in expected.items():
        assert {
            source
            for source, row in by_source.items()
            if row["recommended_target_classification"] == classification
        } == sources
    assert VMEC_BLOCKED_CANDIDATES <= expected["blocked"]

    expected_candidates = (
        expected["mirror"] | expected["hybrid"] | VMEC_BLOCKED_CANDIDATES
    )
    assert {
        source
        for source, row in by_source.items()
        if row["current_disposition"] == "candidate"
    } == expected_candidates


def test_inventory_rows_bind_dependencies_coverage_probe_and_evidence() -> None:
    rows = _rows(_load_inventory())
    for row in rows:
        assert frozenset(row) == EXPECTED_ROW_FIELDS
        source = str(row["source"])
        source_path = REPO_ROOT / "examples" / source
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest() == row["source_sha256"]
        )

        dependencies = row["runtime_dependencies"]
        assert isinstance(dependencies, dict)
        assert frozenset(dependencies) == {"python_import_roots", "external_runtimes"}
        for key in ("python_import_roots", "external_runtimes"):
            values = dependencies[key]
            assert isinstance(values, list)
            assert values == sorted(set(values))
            assert all(isinstance(item, str) and item for item in values)

        coverage = row["current_public_jax_surface_coverage"]
        assert isinstance(coverage, list)
        for entry in coverage:
            assert isinstance(entry, dict)
            assert frozenset(entry) == {"example_id", "jax_surfaces"}
            assert isinstance(entry["example_id"], str) and entry["example_id"]
            assert isinstance(entry["jax_surfaces"], list) and entry["jax_surfaces"]

        probe = row["capability_probe"]
        assert isinstance(probe, dict)
        assert frozenset(probe) == {"argv", "contract", "expected_exit"}
        assert probe["argv"] == [
            "python",
            "examples/jax/probe_one_to_one_inventory.py",
            "--inventory",
            "examples/jax/one_to_one_inventory.json",
            "--source",
            source,
        ]
        assert probe["contract"] == "syntax-source-hash-and-dependency-inventory"
        assert probe["expected_exit"] == 0

        assert isinstance(row["reason"], str) and row["reason"]
        evidence = row["evidence"]
        assert isinstance(evidence, list) and evidence
        assert all(isinstance(item, str) and item for item in evidence)
        if row["recommended_target_classification"] in {"blocked", "not_applicable"}:
            assert isinstance(row["reconsideration_condition"], str)
            assert row["reconsideration_condition"]
        else:
            assert row["reconsideration_condition"] is None


def test_inventory_capability_probe_revalidates_all_rows() -> None:
    assert PROBE_PATH.is_file(), "one-to-one inventory capability probe is missing"
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--inventory",
            str(INVENTORY_PATH),
            "--all",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["source_count"] == len(_tracked_native_sources())
    assert payload["validated"] is True
