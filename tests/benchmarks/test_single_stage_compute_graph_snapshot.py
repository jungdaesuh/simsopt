from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from benchmarks.single_stage_compute_graph_snapshot import (
    IMPORT_ATTESTATION_SCHEMA_ID,
    MANIFEST_FILENAME,
    SOURCE_MANIFEST_SCHEMA_ID,
    RoleRoot,
    SnapshotError,
    attest_snapshot_imports,
    canonical_json_bytes,
    expand_role_roots,
    load_snapshot_manifest,
    publish_immutable_snapshot,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _role_roots(source: Path) -> tuple[RoleRoot, ...]:
    packages = source / "packages"
    _write(packages / "simsopt" / "__init__.py", "PACKAGE = 'simsopt'\n")
    _write(packages / "simsopt_jax" / "__init__.py", "PACKAGE = 'simsopt_jax'\n")
    _write(
        packages / "simsopt_jax_adapters" / "__init__.py",
        "PACKAGE = 'simsopt_jax_adapters'\n",
    )
    configuration = _write(source / "config" / "native.json", "{}\n")
    benchmark = _write(source / "benchmark" / "run.py", "VALUE = 1\n")
    test = _write(source / "test" / "test_run.py", "def test_value(): pass\n")
    extension = _write(source / "extension" / "simsoptpp.py", "VALUE = 'native'\n")
    return (
        RoleRoot("execution_source", packages, "src"),
        RoleRoot("configuration", configuration, "inputs/native.json"),
        RoleRoot("benchmark", benchmark, "benchmarks/run.py"),
        RoleRoot("test", test, "tests/test_run.py"),
        RoleRoot("native_extension", extension, "src/simsoptpp.py"),
    )


def test_expand_role_roots_is_complete_sorted_and_role_bound(tmp_path: Path) -> None:
    inputs = expand_role_roots(_role_roots(tmp_path / "source"))

    assert tuple(item.relative_path for item in inputs) == tuple(
        sorted(item.relative_path for item in inputs)
    )
    assert {item.role for item in inputs} == {
        "execution_source",
        "configuration",
        "benchmark",
        "test",
        "native_extension",
    }
    assert {item.relative_path for item in inputs} == {
        "src/simsopt/__init__.py",
        "src/simsopt_jax/__init__.py",
        "src/simsopt_jax_adapters/__init__.py",
        "src/simsoptpp.py",
        "inputs/native.json",
        "benchmarks/run.py",
        "tests/test_run.py",
    }


def test_publish_creates_canonical_manifest_and_read_only_tree(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"

    publication = publish_immutable_snapshot(
        destination, _role_roots(tmp_path / "source")
    )
    entries, manifest_sha = load_snapshot_manifest(destination)

    document = json.loads(publication.manifest_path.read_bytes())
    assert document["schema_id"] == SOURCE_MANIFEST_SCHEMA_ID
    assert publication.manifest_path.read_bytes() == canonical_json_bytes(document)
    assert manifest_sha == publication.manifest_sha256
    assert entries == publication.entries
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        for path in destination.rglob("*")
        if path.is_file()
    )


def test_publication_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        publish_immutable_snapshot(destination, _role_roots(tmp_path / "source"))

    assert tuple(destination.iterdir()) == ()


def test_role_expansion_rejects_missing_role_collision_and_secret_path(
    tmp_path: Path,
) -> None:
    roots = _role_roots(tmp_path / "source")
    with pytest.raises(SnapshotError, match="must include"):
        expand_role_roots(roots[:-1])

    collision = roots + (RoleRoot("test", roots[1].source_path, "inputs/native.json"),)
    with pytest.raises(SnapshotError, match="assigned more than once"):
        expand_role_roots(collision)

    secret = _write(tmp_path / "source" / "secret", "SECRET=x\n")
    secret_roots = roots[:-1] + (RoleRoot("native_extension", secret, ".env"),)
    with pytest.raises(SnapshotError, match="environment-secret"):
        expand_role_roots(secret_roots)


def test_role_expansion_rejects_symlinked_source(tmp_path: Path) -> None:
    roots = list(_role_roots(tmp_path / "source"))
    target = _write(tmp_path / "target.py", "VALUE = 1\n")
    symlink = tmp_path / "source" / "symlink.py"
    symlink.symlink_to(target)
    roots[1] = RoleRoot("configuration", symlink, "inputs/native.py")

    with pytest.raises(SnapshotError, match="symlink"):
        expand_role_roots(tuple(roots))


def test_manifest_revalidation_detects_changed_and_unmanifested_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "snapshot"
    publish_immutable_snapshot(destination, _role_roots(tmp_path / "source"))
    changed = destination / "src" / "simsopt" / "__init__.py"
    changed.chmod(0o644)
    changed.write_text("CHANGED = True\n", encoding="utf-8")

    with pytest.raises(SnapshotError, match="differs from recorded bytes"):
        load_snapshot_manifest(destination)

    destination = tmp_path / "snapshot-extra"
    publish_immutable_snapshot(destination, _role_roots(tmp_path / "source-extra"))
    destination.chmod(0o755)
    extra = destination / "extra.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="unmanifested"):
        load_snapshot_manifest(destination)


def test_manifest_loader_rejects_noncanonical_and_duplicate_json(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "snapshot"
    publish_immutable_snapshot(destination, _role_roots(tmp_path / "source"))
    manifest = destination / MANIFEST_FILENAME
    manifest.chmod(0o644)
    document = json.loads(manifest.read_bytes())
    manifest.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(SnapshotError, match="not canonical"):
        load_snapshot_manifest(destination)

    manifest.write_text(
        '{"schema_id":"a","schema_id":"b","entries":[]}\n', encoding="utf-8"
    )
    with pytest.raises(SnapshotError, match="duplicate JSON key"):
        load_snapshot_manifest(destination)


def test_pinned_child_attests_all_imports_from_copied_tree(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    publication = publish_immutable_snapshot(
        destination, _role_roots(tmp_path / "source")
    )
    output = tmp_path / "attestation.json"

    attestation = attest_snapshot_imports(
        destination,
        interpreter=Path(sys.executable),
        output_path=output,
    )

    assert attestation.schema_id == IMPORT_ATTESTATION_SCHEMA_ID
    assert attestation.state == "pass"
    assert attestation.interpreter_path == str(Path(sys.executable).absolute())
    assert attestation.snapshot_manifest_sha256 == publication.manifest_sha256
    assert tuple(binding.module for binding in attestation.bindings) == (
        "simsopt",
        "simsopt_jax",
        "simsopt_jax_adapters",
        "simsoptpp",
    )
    assert all(
        (destination / binding.relative_path).is_file()
        for binding in attestation.bindings
    )
    assert output.read_bytes() == canonical_json_bytes(attestation.to_json())


def test_attestation_output_is_exclusive(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    publish_immutable_snapshot(destination, _role_roots(tmp_path / "source"))
    output = tmp_path / "attestation.json"
    output.write_text("user-owned\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        attest_snapshot_imports(
            destination,
            interpreter=Path(sys.executable),
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == "user-owned\n"


def test_attestation_requires_absolute_executable_pinned_interpreter(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "snapshot"
    publish_immutable_snapshot(destination, _role_roots(tmp_path / "source"))

    with pytest.raises(SnapshotError, match="must be absolute"):
        attest_snapshot_imports(destination, interpreter=Path("python"))
    with pytest.raises(SnapshotError, match="executable regular file"):
        attest_snapshot_imports(destination, interpreter=tmp_path / "missing-python")


def test_attestation_fails_if_required_import_cannot_resolve_from_snapshot(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "snapshot"
    publish_immutable_snapshot(destination, _role_roots(tmp_path / "source"))
    adapter = destination / "src" / "simsopt_jax_adapters" / "__init__.py"
    destination.chmod(0o755)
    adapter.parent.parent.chmod(0o755)
    adapter.parent.chmod(0o755)
    adapter.chmod(0o644)
    adapter.unlink()

    with pytest.raises(SnapshotError, match="absent or non-regular"):
        attest_snapshot_imports(
            destination,
            interpreter=Path(sys.executable),
        )


def test_attestation_environment_is_explicit_when_supplied(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    publish_immutable_snapshot(destination, _role_roots(tmp_path / "source"))
    environment = {"PATH": os.environ.get("PATH", "")}

    attestation = attest_snapshot_imports(
        destination,
        interpreter=Path(sys.executable),
        environment=environment,
    )

    assert attestation.state == "pass"
