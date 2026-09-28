from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import examples.jax._manifest as manifest_contract
import pytest
from examples.jax._manifest import (
    EXAMPLE_IMPLEMENTATION_PACKAGE,
    ExampleImplementationError,
    ManifestValidationError,
    derive_source_coverage,
    parse_manifest_document,
    resolve_example_implementation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "jax_manifests" / "manifest_v2.json"
APPROVED_MANIFEST_V2_SHA256 = (
    "2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05"
)
NATIVE_TIERS = {
    "1_Simple",
    "2_Intermediate",
    "3_Advanced",
    "stellarator_benchmarks",
}
# Frozen v2 catalog is 51 sources. This historical v2 contract still subtracts
# every post-v2 native source. The two periodic-field-line
# scripts are official upstream sources (they are in the pinned upstream
# inventory at 9e027eac3 and in the live v3 catalog, as blocked without a
# mirror) that reached this branch with an upstream merge, i.e. after the v2
# snapshot was frozen, so they are post-v2 natives here too.
POST_V2_NATIVE_SOURCES = frozenset(
    {
        "1_Simple/periodicfieldline_QA.py",
        "1_Simple/periodicfieldline_QH.py",
    }
)


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


def _load_manifest(path: Path):
    return parse_manifest_document(
        json.loads(path.read_text(encoding="utf-8")),
        repo_root=REPO_ROOT,
        warn_legacy=True,
        allow_historical_catalog=True,
    )


def _v1_document(document: dict[str, object]) -> dict[str, object]:
    candidate = deepcopy(document)
    assert candidate.pop("schema_version") == 2
    for record in _jax_records(candidate):
        devices = record.pop("devices")
        assert isinstance(devices, list)
        record["lanes"] = [
            {"cpu": "cpu-smoke", "gpu": "gpu-strict"}[str(device)] for device in devices
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
    manifest = _load_manifest(MANIFEST_PATH)

    tracked = _tracked_native_examples()
    # Every subtracted source must still be a file in the tree, so the
    # exclusion list cannot quietly keep hiding a source that is gone.
    assert POST_V2_NATIVE_SOURCES <= tracked
    assert {record.source for record in manifest.source_catalog} == (
        tracked - POST_V2_NATIVE_SOURCES
    )
    assert len(manifest.source_catalog) == 51


def test_canonical_manifest_is_exactly_the_approved_v2_candidate() -> None:
    manifest_bytes = MANIFEST_PATH.read_bytes()
    document = _manifest_document()

    assert hashlib.sha256(manifest_bytes).hexdigest() == APPROVED_MANIFEST_V2_SHA256
    assert document["schema_version"] == 2
    assert all("devices" in record for record in _jax_records(document))
    assert all("lanes" not in record for record in _jax_records(document))
    assert all("intents" not in record for record in _jax_records(document))


def test_dual_reader_normalizes_absent_v1_and_explicit_v2(
    tmp_path: Path,
) -> None:
    v2_document = _manifest_document()
    v1_document = _v1_document(v2_document)
    with pytest.warns(FutureWarning, match="manifest schema v1"):
        v1 = _load_manifest(_write_manifest(tmp_path, v1_document))
    v2 = _load_manifest(_write_manifest(tmp_path, v2_document))

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
    document = deepcopy(_manifest_document())
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
        _load_manifest(_write_manifest(tmp_path, document))


def test_v2_candidate_bytes_are_deterministic_and_semantically_identical() -> None:
    canonical_bytes = MANIFEST_PATH.read_bytes()
    document = _v1_document(_manifest_document())

    first_bytes, first_diff = manifest_contract.convert_v1_document_to_v2(
        document,
        repo_root=REPO_ROOT,
        allow_historical_catalog=True,
    )
    second_bytes, second_diff = manifest_contract.convert_v1_document_to_v2(
        deepcopy(document),
        repo_root=REPO_ROOT,
        allow_historical_catalog=True,
    )

    assert first_bytes == second_bytes
    assert first_bytes == canonical_bytes
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
    v2_document = _manifest_document()
    v1_document = _v1_document(v2_document)
    planned = next(
        record for record in _jax_records(v2_document) if record["status"] == "planned"
    )
    planned["devices"] = ["cpu"]
    with pytest.warns(FutureWarning, match="manifest schema v1"):
        v1 = _load_manifest(_write_manifest(tmp_path, v1_document))
    v2 = _load_manifest(_write_manifest(tmp_path, v2_document))

    semantic_diff = manifest_contract.manifest_semantic_diff(v1, v2)

    assert semantic_diff["device_capabilities_equal"] is False
    assert semantic_diff["semantic_equal"] is False


def test_manifest_migration_dry_run_does_not_modify_input(
    tmp_path: Path,
) -> None:
    input_path = _write_manifest(tmp_path, _v1_document(_manifest_document()))
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
    manifest = _load_manifest(MANIFEST_PATH)

    coverage = derive_source_coverage(manifest)

    assert set(coverage) == _tracked_native_examples() - POST_V2_NATIVE_SOURCES
    assert set(coverage.values()) <= {"planned", "covered", "deferred"}
    assert any(state == "planned" for state in coverage.values())
    assert any(state == "deferred" for state in coverage.values())


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("duplicate_source", "duplicate source path"),
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
        planned["devices"] = ["gpu"]
    elif mutation == "ready_without_gpu_lane":
        planned["status"] = "ready"
        planned["devices"] = ["cpu"]
    elif mutation == "ready_without_test":
        planned["status"] = "ready"
        planned["correctness_tests"] = []
    elif mutation == "ready_without_file":
        planned["status"] = "ready"
        planned["path"] = f"{planned['tier']}/not_present.py"
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ManifestValidationError, match=expected_message):
        _load_manifest(_write_manifest(tmp_path, document))


def test_ready_examples_are_public_jax_workflows_not_forwarders() -> None:
    manifest = _load_manifest(MANIFEST_PATH)

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


def _example_script(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _implementation_module(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / "src" / Path(*EXAMPLE_IMPLEMENTATION_PACKAGE.split(".")) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "__init__.py").touch()
    path.write_text(source, encoding="utf-8")
    return path


def _modules(implementation) -> list[str | None]:
    return [source.module for source in implementation.sources]


def test_a_script_that_imports_no_implementation_module_is_its_own_source(
    tmp_path: Path,
) -> None:
    script = _example_script(tmp_path, "self.py", "def main() -> int:\n    return 0\n")

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert _modules(implementation) == [None]
    assert implementation.sources[0].path == script


def test_a_stub_main_does_not_hide_the_module_it_delegates_to(tmp_path: Path) -> None:
    """A script that defines a stub ``main`` is not its own implementation."""

    _implementation_module(tmp_path, "_probe.py", "def main() -> int:\n    return 1\n")
    script = _example_script(
        tmp_path,
        "stub.py",
        f"from {EXAMPLE_IMPLEMENTATION_PACKAGE} import _probe\n\n\n"
        "def main() -> int:\n    return _probe.main()\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert _modules(implementation) == [
        None,
        EXAMPLE_IMPLEMENTATION_PACKAGE,
        f"{EXAMPLE_IMPLEMENTATION_PACKAGE}._probe",
    ]


@pytest.mark.parametrize(
    "statement",
    (
        "from {package}._probe import main",
        "from {package}._probe import main as entry",
        "from {package} import _probe",
        "from {package} import _probe as probe",
        "import {package}._probe",
        "import {package}._probe as probe",
    ),
)
def test_every_import_form_of_an_implementation_module_is_resolved(
    tmp_path: Path, statement: str
) -> None:
    """The bound name is irrelevant; the imported module is what gets scanned."""

    module_path = _implementation_module(
        tmp_path, "_probe.py", "def main() -> int:\n    return 1\n"
    )
    script = _example_script(
        tmp_path,
        "thin.py",
        statement.format(package=EXAMPLE_IMPLEMENTATION_PACKAGE) + "\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert module_path in {source.path for source in implementation.sources}


def test_resolution_follows_the_implementation_package_transitively(
    tmp_path: Path,
) -> None:
    _implementation_module(tmp_path, "_second.py", "VALUE = 2\n")
    _implementation_module(
        tmp_path,
        "_first.py",
        f"from {EXAMPLE_IMPLEMENTATION_PACKAGE}._second import VALUE\n\n\n"
        "def main() -> int:\n    return VALUE\n",
    )
    script = _example_script(
        tmp_path,
        "thin.py",
        f"from {EXAMPLE_IMPLEMENTATION_PACKAGE}._first import main\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert f"{EXAMPLE_IMPLEMENTATION_PACKAGE}._second" in _modules(implementation)


@pytest.mark.parametrize(
    "first_body",
    (
        "from ._second import VALUE\n\n\ndef main() -> int:\n    return VALUE\n",
        "from . import _second\n\n\ndef main() -> int:\n    return _second.VALUE\n",
    ),
)
def test_resolution_follows_relative_imports_inside_the_package(
    tmp_path: Path, first_body: str
) -> None:
    """A module reachable only through a relative import is still scanned."""

    second = _implementation_module(tmp_path, "_second.py", "VALUE = 2\n")
    _implementation_module(tmp_path, "_first.py", first_body)
    script = _example_script(
        tmp_path,
        "thin.py",
        f"from {EXAMPLE_IMPLEMENTATION_PACKAGE}._first import main\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert f"{EXAMPLE_IMPLEMENTATION_PACKAGE}._second" in _modules(implementation)
    assert second in {source.path for source in implementation.sources}


@pytest.mark.parametrize(
    "library_import",
    ("from ..geo import VALUE", "from simsopt_jax_adapters.geo import VALUE"),
)
def test_a_library_import_is_ignored_however_it_is_spelled(
    tmp_path: Path, library_import: str
) -> None:
    """One filter runs on absolute names: relative and absolute agree exactly."""

    library = tmp_path / "src" / "simsopt_jax_adapters" / "geo.py"
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_text("VALUE = 3\n", encoding="utf-8")
    _implementation_module(
        tmp_path,
        "_first.py",
        f"{library_import}\n\n\ndef main() -> int:\n    return VALUE\n",
    )
    script = _example_script(
        tmp_path,
        "thin.py",
        f"from {EXAMPLE_IMPLEMENTATION_PACKAGE}._first import main\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert _modules(implementation) == [
        None,
        EXAMPLE_IMPLEMENTATION_PACKAGE,
        f"{EXAMPLE_IMPLEMENTATION_PACKAGE}._first",
    ]
    # The library file EXISTS and is still not read: the rule ignores a sibling
    # package because of where it is, not because the target was missing.
    assert library.is_file()
    assert library not in {source.path for source in implementation.sources}


def test_every_ancestor_package_init_enters_the_closure(tmp_path: Path) -> None:
    """CPython runs each ``__init__.py`` on the way in, so the gate reads them."""

    package_root = tmp_path / "src" / Path(*EXAMPLE_IMPLEMENTATION_PACKAGE.split("."))
    subpackage = package_root / "_sub"
    subpackage.mkdir(parents=True)
    (package_root / "__init__.py").write_text("ROOT = 1\n", encoding="utf-8")
    (subpackage / "__init__.py").write_text("SUB = 2\n", encoding="utf-8")
    (subpackage / "_leaf.py").write_text(
        "def main() -> int:\n    return 0\n", encoding="utf-8"
    )
    script = _example_script(
        tmp_path,
        "thin.py",
        f"import {EXAMPLE_IMPLEMENTATION_PACKAGE}._sub._leaf\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert _modules(implementation) == [
        None,
        EXAMPLE_IMPLEMENTATION_PACKAGE,
        f"{EXAMPLE_IMPLEMENTATION_PACKAGE}._sub",
        f"{EXAMPLE_IMPLEMENTATION_PACKAGE}._sub._leaf",
    ]
    assert "ROOT = 1" in implementation.sources[1].text
    assert "SUB = 2" in implementation.sources[2].text


def test_a_package_shadows_a_module_of_the_same_name(tmp_path: Path) -> None:
    """When both exist, the ``__init__.py`` CPython executes is what gets read."""

    package_root = tmp_path / "src" / Path(*EXAMPLE_IMPLEMENTATION_PACKAGE.split("."))
    (package_root / "_a").mkdir(parents=True)
    (package_root / "__init__.py").touch()
    (package_root / "_a.py").write_text("SHADOWED = True\n", encoding="utf-8")
    (package_root / "_a" / "__init__.py").write_text(
        "EXECUTED = True\n\n\ndef main() -> int:\n    return 0\n", encoding="utf-8"
    )
    script = _example_script(
        tmp_path,
        "thin.py",
        f"from {EXAMPLE_IMPLEMENTATION_PACKAGE}._a import main\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    resolved = next(
        source
        for source in implementation.sources
        if source.module == f"{EXAMPLE_IMPLEMENTATION_PACKAGE}._a"
    )
    assert resolved.path == package_root / "_a" / "__init__.py"
    assert "EXECUTED = True" in resolved.text
    assert "SHADOWED" not in resolved.text


def test_resolution_refuses_a_namespace_subpackage(tmp_path: Path) -> None:
    """A directory with no ``__init__.py`` is refused, not silently dropped."""

    package_root = tmp_path / "src" / Path(*EXAMPLE_IMPLEMENTATION_PACKAGE.split("."))
    (package_root / "_namespace").mkdir(parents=True)
    (package_root / "__init__.py").touch()
    script = _example_script(
        tmp_path,
        "thin.py",
        f"import {EXAMPLE_IMPLEMENTATION_PACKAGE}._namespace\n\n\n"
        "def main() -> int:\n    return 0\n",
    )

    with pytest.raises(ExampleImplementationError, match="namespace package"):
        resolve_example_implementation(script, repo_root=tmp_path)


def test_resolution_does_not_follow_other_first_party_packages(
    tmp_path: Path,
) -> None:
    """A library import is a dependency, not a relocated example implementation."""

    library = tmp_path / "src" / "simsopt_jax_adapters" / "geo" / "library.py"
    library.parent.mkdir(parents=True)
    (library.parent / "__init__.py").touch()
    library.write_text("VALUE = 3\n", encoding="utf-8")
    script = _example_script(
        tmp_path,
        "self.py",
        "from simsopt_jax_adapters.geo.library import VALUE\n\n\n"
        "def main() -> int:\n    return VALUE\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert _modules(implementation) == [None]
    assert library.is_file()
    assert library not in {source.path for source in implementation.sources}


def test_resolution_refuses_a_script_with_no_main_anywhere(tmp_path: Path) -> None:
    """A non-empty closure does not waive the entry-point requirement."""

    _implementation_module(tmp_path, "_probe.py", "VALUE = 1\n")
    script = _example_script(
        tmp_path,
        "no_main.py",
        f"from {EXAMPLE_IMPLEMENTATION_PACKAGE}._probe import VALUE\n",
    )

    with pytest.raises(ExampleImplementationError, match="defines main"):
        resolve_example_implementation(script, repo_root=tmp_path)


def test_a_dynamically_loading_script_is_resolved_and_reported_to_the_ban(
    tmp_path: Path,
) -> None:
    """The other half: a script WITH ``main`` that also loads code dynamically.

    Resolution succeeds -- there is an entry point -- and ``runpy`` reaches the
    forwarding-wrapper ban through ``imports``, which is what actually rejects it.
    """

    script = _example_script(
        tmp_path,
        "opaque.py",
        "import runpy\n\n\ndef main() -> int:\n"
        "    return int(bool(runpy.run_path('other.py')))\n",
    )

    implementation = resolve_example_implementation(script, repo_root=tmp_path)

    assert _modules(implementation) == [None]
    assert "runpy" in implementation.sources[0].imports


def test_resolution_refuses_a_script_with_neither_main_nor_a_closure(
    tmp_path: Path,
) -> None:
    script = _example_script(
        tmp_path, "opaque.py", "import runpy\n\nrunpy.run_path('other.py')\n"
    )

    with pytest.raises(ExampleImplementationError, match="defines main"):
        resolve_example_implementation(script, repo_root=tmp_path)


def test_resolution_refuses_an_implementation_module_with_no_source(
    tmp_path: Path,
) -> None:
    script = _example_script(
        tmp_path,
        "thin.py",
        f"import {EXAMPLE_IMPLEMENTATION_PACKAGE}._absent\n\n\n"
        "def main() -> int:\n    return 0\n",
    )

    with pytest.raises(ExampleImplementationError, match="no source under"):
        resolve_example_implementation(script, repo_root=tmp_path)
