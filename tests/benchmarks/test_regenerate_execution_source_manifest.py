"""Gates on the execution-source manifest regenerator.

The manifest decides which bytes may execute in a certified run, so the only
interesting property of the tool that writes it is what it REFUSES.  Every test
here builds a synthetic repository whose membership the rule fully determines,
then perturbs one thing and asserts the refusal names it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from benchmarks.regenerate_execution_source_manifest import (
    ManifestRegenerationError,
    execution_source_membership,
    regenerate_execution_source_manifest,
)
from benchmarks.single_stage_fullspace_snapshot import (
    canonical_json_bytes,
    load_canonical_json_bytes,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
    DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION,
    DIAG5_FROZEN_NUMERICAL_PATHS,
    DIAG5_QUALIFIED_FILE_PATHS,
)

_EXTRA_BROAD_SOURCES = (
    "benchmarks/probe_benchmark.py",
    "examples/jax/probe_example.py",
    "src/simsopt_jax/probe_module.py",
)


def _write(root: Path, relative: str, payload: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _synthetic_repository(root: Path) -> Path:
    """A repository the membership rule fully determines, plus its manifest."""

    for root_name in ("benchmarks", "examples", "src"):
        (root / root_name).mkdir(parents=True, exist_ok=True)
    named = (
        set(DIAG5_QUALIFIED_FILE_PATHS)
        | set(DIAG5_FROZEN_NUMERICAL_PATHS)
        | set(_EXTRA_BROAD_SOURCES)
    ) - {DIAG4_EXECUTION_SOURCE_MANIFEST_PATH}
    for relative in sorted(named):
        _write(root, relative, f"# {relative}\n".encode())
    entries = {
        relative: {
            "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(execution_source_membership(root))
    }
    _write(
        root,
        DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
        canonical_json_bytes(
            {
                "entries": entries,
                "entries_sha256": hashlib.sha256(
                    canonical_json_bytes(entries)
                ).hexdigest(),
                "schema_version": DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION,
            }
        ),
    )
    return root


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    return _synthetic_repository(tmp_path / "repository")


def _member_count(repository: Path) -> int:
    return len(execution_source_membership(repository))


def test_regeneration_reproduces_a_manifest_that_already_describes_the_tree(
    repository: Path,
) -> None:
    """A no-op refreeze is byte-identical and reports nothing changed."""

    manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    body, changed = regenerate_execution_source_manifest(
        expected_count=_member_count(repository), repository=repository
    )

    assert changed == ()
    assert body == manifest_path.read_bytes()


def test_regeneration_rewrites_the_digest_of_an_edited_member(
    repository: Path,
) -> None:
    """The recorded digest follows the bytes, and the edit is named."""

    edited = "src/simsopt_jax/probe_module.py"
    (repository / edited).write_bytes(b"# edited\n")

    body, changed = regenerate_execution_source_manifest(
        expected_count=_member_count(repository), repository=repository
    )

    assert changed == (edited,)
    document = load_canonical_json_bytes(body)
    assert isinstance(document, dict)
    entries = document["entries"]
    assert entries[edited]["sha256"] == hashlib.sha256(b"# edited\n").hexdigest()
    assert entries[edited]["size_bytes"] == len(b"# edited\n")
    assert (
        hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
        == document["entries_sha256"]
    )


def test_regeneration_refuses_a_new_broad_root_file_nobody_admitted(
    repository: Path,
) -> None:
    """Silent membership growth is the failure this tool exists to prevent."""

    _write(repository, "benchmarks/newcomer.py", b"# newcomer\n")

    with pytest.raises(ManifestRegenerationError, match="without being declared"):
        regenerate_execution_source_manifest(
            expected_count=_member_count(repository), repository=repository
        )


def test_regeneration_admits_a_new_broad_root_file_that_was_named(
    repository: Path,
) -> None:
    """Deliberate growth passes, and the admitted path enters the entries."""

    before = _member_count(repository)
    _write(repository, "benchmarks/newcomer.py", b"# newcomer\n")

    body, changed = regenerate_execution_source_manifest(
        expected_count=before + 1,
        admitted=("benchmarks/newcomer.py",),
        repository=repository,
    )

    assert changed == ("benchmarks/newcomer.py",)
    document = load_canonical_json_bytes(body)
    assert isinstance(document, dict)
    assert len(document["entries"]) == before + 1
    assert "benchmarks/newcomer.py" in document["entries"]


def test_regeneration_refuses_to_admit_a_path_that_is_already_a_member(
    repository: Path,
) -> None:
    """Admitting a member is an operator naming the wrong file, not a no-op."""

    with pytest.raises(ManifestRegenerationError, match="already members"):
        regenerate_execution_source_manifest(
            expected_count=_member_count(repository),
            admitted=("src/simsopt_jax/probe_module.py",),
            repository=repository,
        )


def test_regeneration_refuses_to_admit_a_path_the_rule_does_not_select(
    repository: Path,
) -> None:
    """A misspelled or absent path must fail, not enter the certified surface."""

    _write(repository, "tests/benchmarks/probe_outside_the_rule.py", b"# outside\n")

    with pytest.raises(ManifestRegenerationError, match="does not select"):
        regenerate_execution_source_manifest(
            expected_count=_member_count(repository),
            admitted=("tests/benchmarks/probe_outside_the_rule.py",),
            repository=repository,
        )
    with pytest.raises(ManifestRegenerationError, match="does not select"):
        regenerate_execution_source_manifest(
            expected_count=_member_count(repository),
            admitted=("benchmarks/never_written.py",),
            repository=repository,
        )


def test_regeneration_refuses_a_member_that_vanished_from_the_tree(
    repository: Path,
) -> None:
    """A removed execution source is a gate removal and must be declared too."""

    (repository / "benchmarks/probe_benchmark.py").unlink()

    with pytest.raises(ManifestRegenerationError, match="leaving="):
        regenerate_execution_source_manifest(
            expected_count=_member_count(repository), repository=repository
        )


def test_regeneration_refuses_a_count_the_refreeze_did_not_bump(
    repository: Path,
) -> None:
    """The frozen twins and the manifest move together or not at all."""

    before = _member_count(repository)
    _write(repository, "benchmarks/newcomer.py", b"# newcomer\n")

    with pytest.raises(ManifestRegenerationError, match="--expect-count names"):
        regenerate_execution_source_manifest(
            expected_count=before,
            admitted=("benchmarks/newcomer.py",),
            repository=repository,
        )


def test_a_refused_regeneration_leaves_the_manifest_untouched(
    repository: Path,
) -> None:
    """Refusal is not a partial write: the reviewed manifest survives it."""

    manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    original = manifest_path.read_bytes()
    _write(repository, "benchmarks/newcomer.py", b"# newcomer\n")

    with pytest.raises(ManifestRegenerationError):
        regenerate_execution_source_manifest(
            expected_count=_member_count(repository), repository=repository
        )

    assert manifest_path.read_bytes() == original


def test_the_live_repository_membership_matches_its_published_manifest() -> None:
    """The committed manifest is the one this repository's rule produces."""

    live = Path(__file__).resolve().parents[2]
    document = load_canonical_json_bytes(
        (live / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH).read_bytes()
    )
    assert isinstance(document, dict)
    assert frozenset(document["entries"]) == execution_source_membership(live)
