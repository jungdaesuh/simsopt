from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from benchmarks.single_stage_compute_graph_snapshot import (
    SnapshotError,
    canonical_json_bytes,
    expand_role_roots,
    load_snapshot_manifest,
)
from benchmarks.single_stage_compute_graph_snapshot_publish import (
    DEFAULT_OVERLAY_LOCK_RELATIVE_PATH,
    PUBLICATION_SCHEMA_ID,
    SPECIMEN_DESTINATION_ROOT,
    main,
    publish_live_compute_graph_snapshot,
    select_live_role_roots,
)


def _write(path: Path, value: str = "VALUE = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _specimen(root: Path) -> Path:
    candidate = b"candidate-npy-bytes\n"
    arrays = {
        "axis_dofs": b"axis-dofs-npy-bytes\n",
        "coil_dofs": b"coil-dofs-npy-bytes\n",
        "surface_dofs": b"surface-dofs-npy-bytes\n",
    }
    bundle = {
        "schema_version": 2,
        "case_id": "fixture",
        "scale": "native_default",
        "random_seed": 1,
        "configuration": {},
        "configuration_fingerprint": "fixture",
        "arrays": {
            name: {
                "path": f"inputs/{name}.npy",
                "dtype": "<f8",
                "shape": [461],
                "order": "C",
                "sha256": _digest(array),
            }
            for name, array in arrays.items()
        },
        "input_fingerprint": "fixture",
    }
    bundle_bytes = canonical_json_bytes(bundle)
    specimen = {
        "schema_id": "fixture-specimen-v1",
        "specimen": {"input_bundle_sha256": _digest(bundle_bytes)},
        "input_bundle": {"relative_path": "input_bundle"},
        "candidate": {
            "relative_path": "changed_state_candidate.npy",
            "file_sha256": _digest(candidate),
        },
    }
    _write(root / "specimen.json", canonical_json_bytes(specimen).decode("utf-8"))
    (root / "changed_state_candidate.npy").write_bytes(candidate)
    _write(
        root / "input_bundle" / "input_bundle.json",
        bundle_bytes.decode("utf-8"),
    )
    (root / "input_bundle" / "inputs").mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        (root / "input_bundle" / "inputs" / f"{name}.npy").write_bytes(array)
    return root


def _repository(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Snapshot Test")
    _git(repository, "config", "user.email", "snapshot@example.invalid")
    _write(
        repository / ".gitignore",
        "__pycache__/\nartifacts/\n.artifacts/\n.env*\n!.env.example\n",
    )
    _write(repository / "src" / "package.py")
    _write(repository / "examples" / "case.py")
    _write(repository / "benchmarks" / "run.py")
    _write(repository / "tests" / "test_run.py")
    inputs = {
        "pyproject": _write(repository / "pyproject.toml", "[project]\nname='x'\n"),
        "plan": _write(repository / "docs" / "plan.md", "plan\n"),
        "overlay_lock": _write(repository / "benchmarks" / "overlay.lock", "a==1\n"),
    }
    _write(repository / "benchmarks" / "specimen.py")
    _write(repository / "src" / ".env.example", "EXAMPLE=1\n")
    _git(repository, "add", ".")
    _write(repository / "src" / "__pycache__" / "tracked.pyc", "tracked ignored\n")
    _git(repository, "add", "-f", "src/__pycache__/tracked.pyc")
    _git(repository, "commit", "-qm", "fixture")
    _write(repository / "src" / "untracked.py")
    _write(repository / "src" / "__pycache__" / "ignored.pyc", "ignored\n")
    _write(repository / "benchmarks" / "artifacts" / "ignored.json", "{}\n")
    _write(repository / "tests" / ".env.local", "SECRET=1\n")
    inputs["specimen_root"] = _specimen(repository / ".artifacts" / "fresh-specimen")
    return repository, inputs


def _publish(
    repository: Path,
    inputs: dict[str, Path],
    destination: Path,
    native_extension: Path,
):
    return publish_live_compute_graph_snapshot(
        repository,
        destination,
        pyproject=inputs["pyproject"],
        plan=inputs["plan"],
        specimen_root=inputs["specimen_root"],
        overlay_lock=inputs["overlay_lock"],
        native_extension=native_extension,
    )


def test_selection_assigns_every_live_nonignored_file_once(tmp_path: Path) -> None:
    repository, inputs = _repository(tmp_path)
    extension = _write(tmp_path / "host" / "simsoptpp.local.so", "local\n")

    roots = select_live_role_roots(
        repository,
        pyproject=inputs["pyproject"],
        plan=inputs["plan"],
        specimen_root=inputs["specimen_root"],
        overlay_lock=inputs["overlay_lock"],
        native_extension=extension,
    )
    expanded = expand_role_roots(roots)
    by_path = {
        snapshot_input.relative_path: snapshot_input.role for snapshot_input in expanded
    }

    assert by_path["src/package.py"] == "execution_source"
    assert by_path["examples/case.py"] == "execution_source"
    assert by_path["src/untracked.py"] == "execution_source"
    assert by_path["src/.env.example"] == "execution_source"
    assert by_path["benchmarks/run.py"] == "benchmark"
    assert by_path["benchmarks/specimen.py"] == "benchmark"
    assert by_path["tests/test_run.py"] == "test"
    assert by_path["pyproject.toml"] == "configuration"
    assert by_path[DEFAULT_OVERLAY_LOCK_RELATIVE_PATH] == "configuration"
    assert by_path[f"{SPECIMEN_DESTINATION_ROOT}/specimen.json"] == "configuration"
    assert (
        by_path[f"{SPECIMEN_DESTINATION_ROOT}/changed_state_candidate.npy"]
        == "configuration"
    )
    assert (
        by_path[f"{SPECIMEN_DESTINATION_ROOT}/input_bundle/inputs/coil_dofs.npy"]
        == "configuration"
    )
    assert by_path["src/simsoptpp.local.so"] == "native_extension"
    assert not any("__pycache__" in path for path in by_path)
    assert not any("artifacts" in path for path in by_path)
    assert not any(path.endswith(".env.local") for path in by_path)
    assert len(by_path) == len(expanded)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "requires specimen.json"),
        ("extra", "closed-world contract"),
        ("symlink", "contains symlink"),
        ("secret", "environment-secret"),
        ("changed", "candidate bytes do not match"),
    ),
)
def test_specimen_root_rejects_incomplete_or_unowned_bytes(
    tmp_path: Path, mutation: str, message: str
) -> None:
    repository, inputs = _repository(tmp_path)
    specimen_root = inputs["specimen_root"]
    if mutation == "missing":
        (specimen_root / "changed_state_candidate.npy").unlink()
    elif mutation == "extra":
        _write(specimen_root / "extra.json", "{}\n")
    elif mutation == "symlink":
        (specimen_root / "linked.npy").symlink_to(
            specimen_root / "changed_state_candidate.npy"
        )
    elif mutation == "secret":
        _write(specimen_root / ".env.production", "SECRET=1\n")
    else:
        (specimen_root / "changed_state_candidate.npy").write_bytes(b"changed\n")
    extension = _write(tmp_path / "host" / "simsoptpp.local.so", "local\n")

    with pytest.raises(SnapshotError, match=message):
        select_live_role_roots(
            repository,
            pyproject=inputs["pyproject"],
            plan=inputs["plan"],
            specimen_root=specimen_root,
            overlay_lock=inputs["overlay_lock"],
            native_extension=extension,
        )


def test_publication_is_deterministic_and_keeps_provenance_external(
    tmp_path: Path,
) -> None:
    repository, inputs = _repository(tmp_path)
    extension = _write(tmp_path / "host" / "simsoptpp.local.so", "local\n")

    first = _publish(repository, inputs, tmp_path / "snapshot-a", extension)
    second = _publish(repository, inputs, tmp_path / "snapshot-b", extension)

    assert first.publication.manifest_sha256 == second.publication.manifest_sha256
    assert first.cross_host_source_sha256 == second.cross_host_source_sha256
    assert not first.provenance_path.is_relative_to(first.publication.root)
    record = json.loads(first.provenance_path.read_text(encoding="utf-8"))
    assert record["schema_id"] == PUBLICATION_SCHEMA_ID
    assert record["snapshot_manifest_sha256"] == first.publication.manifest_sha256
    assert record["worktree"]["repository_commit"]
    assert len(record["worktree"]["tracked_diff_sha256"]) == 64
    assert len(record["worktree"]["untracked_manifest_sha256"]) == 64
    entries, _ = load_snapshot_manifest(first.publication.root)
    assert all(entry.relative_path != first.provenance_path.name for entry in entries)


def test_host_specific_extensions_share_cross_host_source_identity(
    tmp_path: Path,
) -> None:
    repository, inputs = _repository(tmp_path)
    local = _write(tmp_path / "local" / "simsoptpp.local.so", "local\n")
    landau = _write(tmp_path / "landau" / "simsoptpp.landau.so", "landau\n")

    local_result = _publish(repository, inputs, tmp_path / "local-snapshot", local)
    landau_result = _publish(repository, inputs, tmp_path / "landau-snapshot", landau)

    assert (
        local_result.cross_host_source_sha256 == landau_result.cross_host_source_sha256
    )
    assert (
        local_result.publication.manifest_sha256
        != landau_result.publication.manifest_sha256
    )
    assert local_result.native_extension_relative_path == "src/simsoptpp.local.so"
    assert landau_result.native_extension_relative_path == "src/simsoptpp.landau.so"


def test_selection_fails_on_missing_secret_symlink_and_collision(
    tmp_path: Path,
) -> None:
    repository, inputs = _repository(tmp_path)
    extension = _write(tmp_path / "host" / "simsoptpp.local.so", "local\n")

    with pytest.raises(SnapshotError, match="pyproject must be an existing"):
        select_live_role_roots(
            repository,
            pyproject=Path("missing.toml"),
            plan=inputs["plan"],
            specimen_root=inputs["specimen_root"],
            overlay_lock=inputs["overlay_lock"],
            native_extension=extension,
        )

    secret = _write(tmp_path / ".env.production", "SECRET=1\n")
    with pytest.raises(SnapshotError, match="environment-secret"):
        select_live_role_roots(
            repository,
            pyproject=secret,
            plan=inputs["plan"],
            specimen_root=inputs["specimen_root"],
            overlay_lock=inputs["overlay_lock"],
            native_extension=extension,
        )

    target = _write(tmp_path / "host" / "real-simsoptpp.so", "real\n")
    symlink = tmp_path / "host" / "simsoptpp.symlink.so"
    symlink.symlink_to(target)
    with pytest.raises(SnapshotError, match="symlink"):
        _publish(repository, inputs, tmp_path / "symlink-snapshot", symlink)

    _write(repository / "src" / "simsoptpp.local.so", "repo\n")
    _git(repository, "add", "src/simsoptpp.local.so")
    collision_extension = _write(
        tmp_path / "collision" / "simsoptpp.local.so", "external\n"
    )
    with pytest.raises(SnapshotError, match="assigned more than once"):
        _publish(
            repository,
            inputs,
            tmp_path / "collision-snapshot",
            collision_extension,
        )


def test_cli_publishes_snapshot_and_prints_exact_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository, inputs = _repository(tmp_path)
    extension = _write(tmp_path / "host" / "simsoptpp.local.so", "local\n")
    destination = tmp_path / "snapshot"
    provenance = tmp_path / "provenance.json"

    result = main(
        (
            "--repo-root",
            str(repository),
            "--destination",
            str(destination),
            "--pyproject",
            str(inputs["pyproject"]),
            "--plan",
            str(inputs["plan"]),
            "--specimen-root",
            str(inputs["specimen_root"]),
            "--overlay-lock",
            str(inputs["overlay_lock"]),
            "--native-extension",
            str(extension),
            "--provenance-output",
            str(provenance),
        )
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["snapshot_root"] == str(destination)
    assert output["provenance_path"] == str(provenance)
    assert destination.is_dir()
    assert provenance.is_file()
