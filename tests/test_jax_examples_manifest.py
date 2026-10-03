from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import copy
import json
from pathlib import Path

import pytest
from examples.jax._manifest import (
    EXAMPLE_IMPLEMENTATION_PACKAGE,
    ExampleImplementationError,
    resolve_example_implementation,
)
from examples.jax.manifest_contracts_v3 import (
    ManifestV3ValidationError,
    parse_examples_v3_document,
)
from examples.jax.manifest_runtime import load_runtime_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "manifest.json"


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _records(document: dict[str, object], key: str) -> list[dict[str, object]]:
    values = document[key]
    assert isinstance(values, list)
    assert all(isinstance(value, dict) for value in values)
    return values


def _record_by(
    document: dict[str, object], collection: str, key: str, value: str
) -> dict[str, object]:
    matches = [
        record for record in _records(document, collection) if record.get(key) == value
    ]
    assert len(matches) == 1
    return matches[0]


def test_schema_v3_rejects_missing_duplicate_tutorial_and_alias_ownership() -> None:
    examples = _document(MANIFEST_PATH)
    source = _record_by(
        examples, "source_catalog", "source", "2_Intermediate/boozer.py"
    )
    mirror_id = source["mirror_example_id"]
    assert isinstance(mirror_id, str)

    missing = copy.deepcopy(examples)
    missing_source = _record_by(
        missing, "source_catalog", "source", "2_Intermediate/boozer.py"
    )
    missing_source["mirror_example_id"] = None
    with pytest.raises(ManifestV3ValidationError, match="eligible source requires"):
        parse_examples_v3_document(missing, repo_root=REPO_ROOT)

    duplicate = copy.deepcopy(examples)
    duplicate_source = _record_by(
        duplicate, "source_catalog", "source", "2_Intermediate/boozerQA.py"
    )
    duplicate_source["mirror_example_id"] = mirror_id
    with pytest.raises(ManifestV3ValidationError, match="duplicate mirror ownership"):
        parse_examples_v3_document(duplicate, repo_root=REPO_ROOT)

    tutorial = copy.deepcopy(examples)
    _record_by(tutorial, "jax_examples", "id", mirror_id)["classification"] = "tutorial"
    with pytest.raises(ManifestV3ValidationError, match="tutorial cannot own coverage"):
        parse_examples_v3_document(tutorial, repo_root=REPO_ROOT)

    alias = copy.deepcopy(examples)
    alias_mirror = _record_by(alias, "jax_examples", "id", mirror_id)
    alias_mirror["path"] = "2_Intermediate/not_the_native_filename.py"
    with pytest.raises(ManifestV3ValidationError, match="exact-name mirror path"):
        parse_examples_v3_document(alias, repo_root=REPO_ROOT)


def test_schema_v3_rejects_hybrid_without_explicit_gpu_slice_scope() -> None:
    examples = _document(MANIFEST_PATH)
    hybrid = _record_by(
        examples, "jax_examples", "id", "native-single-stage-optimization"
    )
    scopes = hybrid["supported_device_scopes"]
    assert isinstance(scopes, dict)
    del scopes["gpu"]
    with pytest.raises(ManifestV3ValidationError, match="hybrid GPU scope"):
        parse_examples_v3_document(examples, repo_root=REPO_ROOT)


def test_schema_v3_rejects_unregistered_executable_fields() -> None:
    examples = _document(MANIFEST_PATH)
    first = _records(examples, "jax_examples")[0]
    first["inspired_by"] = ["1_Simple/just_a_quadratic.py"]
    with pytest.raises(ManifestV3ValidationError, match="unexpected executable fields"):
        parse_examples_v3_document(examples, repo_root=REPO_ROOT)


def test_manifest_accepts_only_example_schema_v3() -> None:
    examples = _document(MANIFEST_PATH)

    assert parse_examples_v3_document(examples, repo_root=REPO_ROOT).schema_version == 3

    # The retired example-v2 schema is refused like any unknown one.
    for version in (2, 4):
        unknown_examples = copy.deepcopy(examples)
        unknown_examples["schema_version"] = version
        with pytest.raises(
            ManifestV3ValidationError, match="unsupported example schema"
        ):
            parse_examples_v3_document(unknown_examples, repo_root=REPO_ROOT)


def test_ready_examples_are_public_jax_workflows_not_forwarders() -> None:
    examples = load_runtime_manifest(MANIFEST_PATH, repo_root=REPO_ROOT).examples

    for example in examples:
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
        "  jax-private-optimizer:", maxsplit=1
    )[0]
    # The self-hosted strict GPU job is the last job of the dispatch/schedule
    # workflow; the pull_request-triggered workflow has no self-hosted job.
    gpu_strict = (
        (REPO_ROOT / ".github" / "workflows" / "jax_gpu_parity.yml")
        .read_text(encoding="utf-8")
        .split("  jax-gpu-strict-purity:", maxsplit=1)[1]
    )
    assert "self-hosted" not in workflow

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
