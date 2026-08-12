"""Select and publish the live Phase 0 execution tree deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from benchmarks.single_stage_compute_graph_snapshot import (
    ManifestEntry,
    RoleRoot,
    SnapshotError,
    SnapshotPublication,
    canonical_json_bytes,
    publish_immutable_snapshot,
)

PUBLICATION_SCHEMA_ID: Final = "single-stage-compute-graph-snapshot-publication-v1"
DEFAULT_PLAN_RELATIVE_PATH: Final = (
    "docs/single_stage_jax_gpu_compute_graph_optimization_implementation_plan.md"
)
SPECIMEN_DESTINATION_ROOT: Final = "phase0-specimen"
DEFAULT_OVERLAY_LOCK_RELATIVE_PATH: Final = "benchmarks/landau_a100_overlay_lock.txt"
_AUTOMATIC_ROOT_ROLES: Final = {
    "src": "execution_source",
    "examples": "execution_source",
    "benchmarks": "benchmark",
    "tests": "test",
}
_CONFIGURATION_DESTINATIONS: Final = (
    "pyproject.toml",
    DEFAULT_PLAN_RELATIVE_PATH,
    DEFAULT_OVERLAY_LOCK_RELATIVE_PATH,
)


@dataclass(frozen=True, slots=True)
class GitWorktreeProvenance:
    """Externally stored identity of the live Git/worktree selection."""

    repository_commit: str
    git_status_short: tuple[str, ...]
    tracked_diff_sha256: str
    untracked_manifest_sha256: str
    source_state_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "repository_commit": self.repository_commit,
            "git_status_short": list(self.git_status_short),
            "tracked_diff_sha256": self.tracked_diff_sha256,
            "untracked_manifest_sha256": self.untracked_manifest_sha256,
            "source_state_sha256": self.source_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class SnapshotPublishResult:
    """Snapshot publication plus its separate worktree provenance record."""

    publication: SnapshotPublication
    provenance_path: Path
    cross_host_source_sha256: str
    native_extension_relative_path: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SnapshotError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode}: {stderr}"
        )
    return completed.stdout


def _repository_root(repository: Path) -> Path:
    requested = repository.resolve()
    observed = Path(
        _run_git(requested, ("rev-parse", "--show-toplevel")).decode("utf-8").strip()
    ).resolve()
    if observed != requested:
        raise SnapshotError("repository root must be the Git worktree top level")
    return observed


def _resolve_required_file(repository: Path, value: Path, label: str) -> Path:
    candidate = value if value.is_absolute() else repository / value
    candidate = candidate.absolute()
    if not candidate.is_file():
        raise SnapshotError(f"{label} must be an existing regular file")
    if _is_environment_variant(candidate.name):
        raise SnapshotError(f"{label} may not be an environment-secret file")
    return candidate


def _is_environment_variant(name: str) -> bool:
    return name.startswith(".env") and name != ".env.example"


def _safe_specimen_relative_path(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SnapshotError(f"{context} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise SnapshotError(f"{context} must be a safe relative path")
    if any(_is_environment_variant(part) for part in path.parts):
        raise SnapshotError(f"{context} may not name an environment-secret file")
    return path


def _json_object(path: Path, context: str) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise SnapshotError(f"{context} contains duplicate key {key!r}")
            document[key] = value
        return document

    document = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(document, dict):
        raise SnapshotError(f"{context} must be a JSON object")
    if payload != canonical_json_bytes(document):
        raise SnapshotError(f"{context} must use canonical JSON bytes")
    return document, payload


def _required_object(
    document: Mapping[str, object], key: str, context: str
) -> Mapping[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise SnapshotError(f"{context}.{key} must be a JSON object")
    return value


def _required_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SnapshotError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _validate_specimen_root(repository: Path, value: Path) -> Path:
    root = value if value.is_absolute() else repository / value
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise SnapshotError("specimen root must be an existing non-symlink directory")
    descendants = tuple(sorted(root.rglob("*")))
    for descendant in descendants:
        relative = descendant.relative_to(root)
        if descendant.is_symlink():
            raise SnapshotError(
                f"specimen root contains symlink {relative.as_posix()!r}"
            )
        if any(_is_environment_variant(part) for part in relative.parts):
            raise SnapshotError(
                f"specimen root contains environment-secret path {relative.as_posix()!r}"
            )
        if not descendant.is_dir() and not descendant.is_file():
            raise SnapshotError(
                f"specimen root contains non-regular path {relative.as_posix()!r}"
            )

    specimen_path = root / "specimen.json"
    candidate_path = root / "changed_state_candidate.npy"
    bundle_path = root / "input_bundle" / "input_bundle.json"
    if not all(path.is_file() for path in (specimen_path, candidate_path, bundle_path)):
        raise SnapshotError(
            "specimen root requires specimen.json, changed_state_candidate.npy, "
            "and input_bundle/input_bundle.json"
        )
    specimen, _ = _json_object(specimen_path, "specimen.json")
    candidate = _required_object(specimen, "candidate", "specimen.json")
    input_bundle = _required_object(specimen, "input_bundle", "specimen.json")
    specimen_contract = _required_object(specimen, "specimen", "specimen.json")
    if candidate.get("relative_path") != "changed_state_candidate.npy":
        raise SnapshotError("specimen candidate relative_path is not canonical")
    if input_bundle.get("relative_path") != "input_bundle":
        raise SnapshotError("specimen input_bundle relative_path is not canonical")
    expected_candidate_sha256 = _required_sha256(
        candidate.get("file_sha256"), "specimen.json.candidate.file_sha256"
    )
    if _sha256_file(candidate_path) != expected_candidate_sha256:
        raise SnapshotError("specimen candidate bytes do not match specimen.json")

    bundle, bundle_bytes = _json_object(bundle_path, "input_bundle/input_bundle.json")
    expected_bundle_sha256 = _required_sha256(
        specimen_contract.get("input_bundle_sha256"),
        "specimen.json.specimen.input_bundle_sha256",
    )
    if _sha256_bytes(bundle_bytes) != expected_bundle_sha256:
        raise SnapshotError("input bundle bytes do not match specimen.json")
    arrays = _required_object(bundle, "arrays", "input_bundle/input_bundle.json")
    expected_array_names = frozenset({"axis_dofs", "coil_dofs", "surface_dofs"})
    if frozenset(arrays) != expected_array_names:
        raise SnapshotError(
            "input bundle arrays must be exactly axis_dofs, coil_dofs, and surface_dofs"
        )
    expected_files = {
        "specimen.json",
        "changed_state_candidate.npy",
        "input_bundle/input_bundle.json",
    }
    for name, raw_reference in arrays.items():
        if not isinstance(raw_reference, dict):
            raise SnapshotError(f"input bundle array {name!r} must be a JSON object")
        relative_array = _safe_specimen_relative_path(
            raw_reference.get("path"), f"input bundle array {name!r}.path"
        )
        array_relative_path = PurePosixPath("input_bundle") / relative_array
        array_path = root.joinpath(*array_relative_path.parts)
        if not array_path.is_file():
            raise SnapshotError(f"input bundle array {name!r} is missing")
        expected_array_sha256 = _required_sha256(
            raw_reference.get("sha256"), f"input bundle array {name!r}.sha256"
        )
        if _sha256_file(array_path) != expected_array_sha256:
            raise SnapshotError(f"input bundle array {name!r} bytes do not match")
        expected_files.add(array_relative_path.as_posix())

    observed_files = {
        path.relative_to(root).as_posix() for path in descendants if path.is_file()
    }
    if observed_files != expected_files:
        raise SnapshotError(
            "specimen root file set does not match its closed-world contract; "
            f"missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )
    expected_directories = {
        parent.as_posix()
        for relative_path in expected_files
        for parent in PurePosixPath(relative_path).parents
        if parent != PurePosixPath(".")
    }
    observed_directories = {
        path.relative_to(root).as_posix() for path in descendants if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise SnapshotError("specimen root contains missing or extra directories")
    return root


def _listed_repository_paths(repository: Path) -> tuple[str, ...]:
    payload = _run_git(
        repository,
        (
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *tuple(_AUTOMATIC_ROOT_ROLES),
        ),
    )
    paths = tuple(
        part.decode("utf-8", errors="strict") for part in payload.split(b"\0") if part
    )
    if not paths:
        return ()
    check = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "check-ignore",
            "--no-index",
            "-z",
            "--stdin",
        ),
        input=b"\0".join(path.encode("utf-8") for path in paths) + b"\0",
        check=False,
        capture_output=True,
    )
    if check.returncode not in (0, 1):
        stderr = check.stderr.decode("utf-8", errors="replace").strip()
        raise SnapshotError(
            f"git check-ignore failed with exit {check.returncode}: {stderr}"
        )
    ignored = frozenset(
        part.decode("utf-8", errors="strict")
        for part in check.stdout.split(b"\0")
        if part
    )
    return tuple(sorted(path for path in paths if path not in ignored))


def select_live_role_roots(
    repository: Path,
    *,
    pyproject: Path,
    plan: Path,
    specimen_root: Path,
    overlay_lock: Path,
    native_extension: Path,
) -> tuple[RoleRoot, ...]:
    """Select all nonignored execution bytes and assign their manifest roles."""

    repository = _repository_root(repository)
    configuration_sources = tuple(
        _resolve_required_file(repository, path, label)
        for path, label in (
            (pyproject, "pyproject"),
            (plan, "plan"),
            (overlay_lock, "overlay lock"),
        )
    )
    if len(set(configuration_sources)) != len(configuration_sources):
        raise SnapshotError("configuration inputs must be distinct files")
    native_source = _resolve_required_file(
        repository, native_extension, "native extension"
    )
    specimen_source = _validate_specimen_root(repository, specimen_root)
    if not (
        native_source.name.startswith("simsoptpp")
        and native_source.name.endswith(".so")
    ):
        raise SnapshotError("native extension must be a simsoptpp .so file")

    explicit_sources = frozenset((*configuration_sources, native_source))
    roots: list[RoleRoot] = []
    for relative_path in _listed_repository_paths(repository):
        relative = PurePosixPath(relative_path)
        if any(_is_environment_variant(part) for part in relative.parts):
            continue
        source = (repository / relative_path).absolute()
        if source in explicit_sources or source.is_relative_to(specimen_source):
            continue
        role = _AUTOMATIC_ROOT_ROLES.get(relative.parts[0])
        if role is None:
            raise SnapshotError(f"unowned automatic input {relative_path!r}")
        roots.append(RoleRoot(role, source, relative_path))

    roots.extend(
        RoleRoot("configuration", source, destination)
        for source, destination in zip(
            configuration_sources, _CONFIGURATION_DESTINATIONS, strict=True
        )
    )
    roots.append(RoleRoot("configuration", specimen_source, SPECIMEN_DESTINATION_ROOT))
    roots.append(
        RoleRoot(
            "native_extension",
            native_source,
            f"src/{native_source.name}",
        )
    )
    return tuple(sorted(roots, key=lambda root: (root.relative_path, root.role)))


def collect_git_worktree_provenance(repository: Path) -> GitWorktreeProvenance:
    """Hash live Git metadata without mixing it into copied execution bytes."""

    repository = _repository_root(repository)
    repository_commit = (
        _run_git(repository, ("rev-parse", "HEAD")).decode("ascii").strip()
    )
    status = tuple(
        line
        for line in _run_git(repository, ("status", "--short", "--untracked-files=all"))
        .decode("utf-8", errors="strict")
        .splitlines()
        if line
    )
    tracked_diff_sha256 = _sha256_bytes(
        _run_git(repository, ("diff", "--binary", "HEAD", "--"))
    )
    untracked_paths = tuple(
        part.decode("utf-8", errors="strict")
        for part in _run_git(
            repository,
            ("ls-files", "--others", "--exclude-standard", "-z"),
        ).split(b"\0")
        if part
    )
    untracked_entries: list[dict[str, object]] = []
    for relative_path in sorted(untracked_paths):
        relative = PurePosixPath(relative_path)
        if any(_is_environment_variant(part) for part in relative.parts):
            continue
        path = repository / relative_path
        if not path.is_file() or path.is_symlink():
            raise SnapshotError(
                f"untracked provenance input {relative_path!r} is not a regular file"
            )
        untracked_entries.append(
            {
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    untracked_manifest_sha256 = _sha256_bytes(canonical_json_bytes(untracked_entries))
    state_document = {
        "repository_commit": repository_commit,
        "git_status_short": list(status),
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_manifest_sha256": untracked_manifest_sha256,
    }
    return GitWorktreeProvenance(
        repository_commit=repository_commit,
        git_status_short=status,
        tracked_diff_sha256=tracked_diff_sha256,
        untracked_manifest_sha256=untracked_manifest_sha256,
        source_state_sha256=_sha256_bytes(canonical_json_bytes(state_document)),
    )


def _entry_identity(entries: Sequence[ManifestEntry]) -> str:
    return _sha256_bytes(canonical_json_bytes([entry.to_json() for entry in entries]))


def publish_live_compute_graph_snapshot(
    repository: Path,
    destination: Path,
    *,
    pyproject: Path,
    plan: Path,
    specimen_root: Path,
    overlay_lock: Path,
    native_extension: Path,
    provenance_output: Path | None = None,
) -> SnapshotPublishResult:
    """Publish selected bytes and a separate, exclusive provenance record."""

    repository = _repository_root(repository)
    destination = destination.absolute()
    provenance_path = (
        destination.with_name(f"{destination.name}.source-provenance.json")
        if provenance_output is None
        else provenance_output.absolute()
    )
    if provenance_path.exists():
        raise FileExistsError(provenance_path)
    roots = select_live_role_roots(
        repository,
        pyproject=pyproject,
        plan=plan,
        specimen_root=specimen_root,
        overlay_lock=overlay_lock,
        native_extension=native_extension,
    )
    provenance = collect_git_worktree_provenance(repository)
    publication = publish_immutable_snapshot(destination, roots)
    cross_host_entries = tuple(
        entry
        for entry in publication.entries
        if entry.role != "native_extension"
        and entry.relative_path != DEFAULT_OVERLAY_LOCK_RELATIVE_PATH
    )
    native_entry = next(
        entry for entry in publication.entries if entry.role == "native_extension"
    )
    cross_host_source_sha256 = _entry_identity(cross_host_entries)
    record: Mapping[str, object] = {
        "schema_id": PUBLICATION_SCHEMA_ID,
        "repository_root": str(repository),
        "snapshot_root": str(publication.root),
        "snapshot_manifest_sha256": publication.manifest_sha256,
        "cross_host_source_sha256": cross_host_source_sha256,
        "native_extension": native_entry.to_json(),
        "worktree": provenance.to_json(),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("xb") as stream:
        stream.write(canonical_json_bytes(record))
    return SnapshotPublishResult(
        publication=publication,
        provenance_path=provenance_path,
        cross_host_source_sha256=cross_host_source_sha256,
        native_extension_relative_path=native_entry.relative_path,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--specimen-root", type=Path, required=True)
    parser.add_argument("--overlay-lock", type=Path, required=True)
    parser.add_argument("--native-extension", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    result = publish_live_compute_graph_snapshot(
        options.repo_root,
        options.destination,
        pyproject=options.pyproject,
        plan=options.plan,
        specimen_root=options.specimen_root,
        overlay_lock=options.overlay_lock,
        native_extension=options.native_extension,
        provenance_output=options.provenance_output,
    )
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "snapshot_root": str(result.publication.root),
                "snapshot_manifest_sha256": result.publication.manifest_sha256,
                "cross_host_source_sha256": result.cross_host_source_sha256,
                "native_extension_relative_path": (
                    result.native_extension_relative_path
                ),
                "provenance_path": str(result.provenance_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_OVERLAY_LOCK_RELATIVE_PATH",
    "DEFAULT_PLAN_RELATIVE_PATH",
    "PUBLICATION_SCHEMA_ID",
    "SPECIMEN_DESTINATION_ROOT",
    "GitWorktreeProvenance",
    "SnapshotPublishResult",
    "collect_git_worktree_provenance",
    "main",
    "publish_live_compute_graph_snapshot",
    "select_live_role_roots",
)
