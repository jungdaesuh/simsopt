"""Immutable source snapshot and child import attestation for Phase 0.

Callers assign every execution-bearing directory or file to a manifest role.
This module expands those roots without following symlinks, copies and hashes
each byte into a fresh immutable tree, then uses the requested interpreter in
an isolated child to prove all production packages resolve from that tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

SOURCE_MANIFEST_SCHEMA_ID: Final = "single-stage-compute-graph-source-manifest-v1"
IMPORT_ATTESTATION_SCHEMA_ID: Final = "single-stage-compute-graph-import-attestation-v1"
MANIFEST_FILENAME: Final = "phase0-source-manifest.json"

SnapshotRole = Literal[
    "execution_source", "configuration", "benchmark", "test", "native_extension"
]
_ROLES: Final = frozenset(
    {"execution_source", "configuration", "benchmark", "test", "native_extension"}
)
_REQUIRED_IMPORTS: Final = (
    "simsopt",
    "simsopt_jax",
    "simsopt_jax_adapters",
    "simsoptpp",
)
_SHA256_LENGTH: Final = 64
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_COPY_CHUNK_BYTES: Final = 1024 * 1024


class SnapshotError(ValueError):
    """Snapshot selection, publication, or attestation failed closed."""


@dataclass(frozen=True)
class RoleRoot:
    """One recursively complete source root assigned to a manifest role."""

    role: SnapshotRole
    source_path: Path
    relative_path: str


@dataclass(frozen=True)
class SnapshotInput:
    """One expanded regular file and its immutable-tree destination."""

    role: SnapshotRole
    source_path: Path
    relative_path: str


@dataclass(frozen=True)
class ManifestEntry:
    """Role and exact-byte identity for one copied execution-bearing file."""

    role: SnapshotRole
    relative_path: str
    size_bytes: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SnapshotPublication:
    """Published immutable root and canonical source-manifest identity."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    entries: tuple[ManifestEntry, ...]


@dataclass(frozen=True)
class ImportBinding:
    """One child-observed module origin tied to a manifest entry."""

    module: str
    relative_path: str
    size_bytes: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "module": self.module,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ImportAttestation:
    """Pinned-interpreter proof that production imports came from the snapshot."""

    snapshot_manifest_sha256: str
    interpreter_path: str
    python_version: str
    bindings: tuple[ImportBinding, ...]
    schema_id: str = IMPORT_ATTESTATION_SCHEMA_ID
    state: Literal["pass"] = "pass"

    def to_json(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "state": self.state,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "interpreter_path": self.interpreter_path,
            "python_version": self.python_version,
            "bindings": [binding.to_json() for binding in self.bindings],
        }


def canonical_json_bytes(document: object) -> bytes:
    """Return the sole canonical JSON encoding for snapshot evidence."""

    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _role(value: object, context: str) -> SnapshotRole:
    if not isinstance(value, str) or value not in _ROLES:
        raise SnapshotError(f"{context} must be one of {sorted(_ROLES)}")
    if value == "execution_source":
        return "execution_source"
    if value == "configuration":
        return "configuration"
    if value == "benchmark":
        return "benchmark"
    if value == "test":
        return "test"
    return "native_extension"


def _safe_relative_path(value: str, context: str) -> PurePosixPath:
    if not value:
        raise SnapshotError(f"{context} must be non-empty")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise SnapshotError(f"{context} must be a canonical safe relative path")
    if any(part.startswith(".env") and part != ".env.example" for part in path.parts):
        raise SnapshotError(f"{context} may not include an environment-secret file")
    return path


def _reject_symlink_components(path: Path, context: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise SnapshotError(f"{context} may not contain symlink components")
        if current.parent == current:
            return
        current = current.parent


def expand_role_roots(roots: Sequence[RoleRoot]) -> tuple[SnapshotInput, ...]:
    """Expand assigned roots into a sorted, collision-free complete file list."""

    if not roots:
        raise SnapshotError("at least one role root is required")
    inputs: list[SnapshotInput] = []
    for index, root in enumerate(roots):
        context = f"roots[{index}]"
        role = _role(root.role, f"{context}.role")
        source = root.source_path.absolute()
        relative_root = _safe_relative_path(
            root.relative_path, f"{context}.relative_path"
        )
        _reject_symlink_components(source, f"{context}.source_path")
        if source.is_file():
            inputs.append(SnapshotInput(role, source, str(relative_root)))
            continue
        if not source.is_dir():
            raise SnapshotError(
                f"{context}.source_path must be a regular file or directory"
            )
        descendants = tuple(sorted(source.rglob("*")))
        for descendant in descendants:
            relative_descendant = descendant.relative_to(source)
            _reject_symlink_components(descendant, f"{context}.source_path")
            if descendant.is_dir():
                continue
            if not descendant.is_file():
                raise SnapshotError(
                    f"{context}.source_path contains a non-regular file"
                )
            destination = relative_root.joinpath(
                *PurePosixPath(relative_descendant.as_posix()).parts
            )
            _safe_relative_path(str(destination), f"{context} expanded path")
            inputs.append(SnapshotInput(role, descendant, str(destination)))
    if not inputs:
        raise SnapshotError("role roots contain no regular files")
    by_destination: dict[str, SnapshotInput] = {}
    for item in inputs:
        if item.relative_path in by_destination:
            raise SnapshotError(
                f"snapshot destination {item.relative_path!r} is assigned more than once"
            )
        by_destination[item.relative_path] = item
    observed_roles = frozenset(item.role for item in inputs)
    if observed_roles != _ROLES:
        raise SnapshotError(
            "snapshot must include execution_source, configuration, benchmark, "
            "test, and native_extension roles"
        )
    return tuple(by_destination[path] for path in sorted(by_destination))


def _manifest_document(entries: Sequence[ManifestEntry]) -> dict[str, object]:
    return {
        "schema_id": SOURCE_MANIFEST_SCHEMA_ID,
        "entries": [entry.to_json() for entry in entries],
    }


def _make_tree_read_only(root: Path) -> None:
    files = tuple(path for path in root.rglob("*") if path.is_file())
    directories = tuple(path for path in root.rglob("*") if path.is_dir())
    for path in files:
        path.chmod(0o444)
    for path in reversed(directories):
        path.chmod(0o555)
    root.chmod(0o555)


def publish_immutable_snapshot(
    destination: Path, roots: Sequence[RoleRoot]
) -> SnapshotPublication:
    """Publish a fresh read-only tree after copying and hashing every input."""

    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = expand_role_roots(roots)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        entries: list[ManifestEntry] = []
        for item in inputs:
            target = staging.joinpath(*PurePosixPath(item.relative_path).parts)
            before = item.source_path.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise SnapshotError(f"source {item.source_path} is not a regular file")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size_bytes = 0
            with item.source_path.open("rb") as source_stream, target.open(
                "xb"
            ) as target_stream:
                while chunk := source_stream.read(_COPY_CHUNK_BYTES):
                    target_stream.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                target_stream.flush()
                os.fsync(target_stream.fileno())
            after = item.source_path.stat(follow_symlinks=False)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or size_bytes != before.st_size:
                raise SnapshotError(f"source {item.source_path} changed while copied")
            entries.append(
                ManifestEntry(
                    role=item.role,
                    relative_path=item.relative_path,
                    size_bytes=size_bytes,
                    sha256=digest.hexdigest(),
                )
            )
        manifest = _manifest_document(entries)
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_path = staging / MANIFEST_FILENAME
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _make_tree_read_only(staging)
        staging.rename(destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return SnapshotPublication(
            root=destination,
            manifest_path=destination / MANIFEST_FILENAME,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            entries=tuple(entries),
        )
    except Exception:
        if staging.exists():
            staging.chmod(0o700)
            for directory in (path for path in staging.rglob("*") if path.is_dir()):
                directory.chmod(0o700)
            shutil.rmtree(staging)
        raise


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise SnapshotError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SnapshotError(f"{context} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise SnapshotError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SnapshotError(f"{context} must be a nonnegative integer")
    return value


def _sha256(value: object, context: str) -> str:
    checked = _string(value, context)
    if len(checked) != _SHA256_LENGTH or any(c not in _LOWER_HEX for c in checked):
        raise SnapshotError(f"{context} must be a lowercase SHA-256 digest")
    return checked


def load_snapshot_manifest(
    snapshot_root: Path,
) -> tuple[tuple[ManifestEntry, ...], str]:
    """Revalidate manifest structure and every copied file byte."""

    snapshot_root = snapshot_root.resolve()
    manifest_path = snapshot_root / MANIFEST_FILENAME
    raw_bytes = manifest_path.read_bytes()

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(raw_bytes, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise SnapshotError(f"source manifest is not valid JSON: {error}") from error
    manifest = _mapping(document, "manifest")
    _exact_keys(manifest, frozenset({"schema_id", "entries"}), "manifest")
    if manifest["schema_id"] != SOURCE_MANIFEST_SCHEMA_ID:
        raise SnapshotError("source manifest schema_id is unsupported")
    if raw_bytes != canonical_json_bytes(manifest):
        raise SnapshotError("source manifest bytes are not canonical JSON")
    raw_entries = _sequence(manifest["entries"], "manifest.entries")
    entries: list[ManifestEntry] = []
    paths: list[str] = []
    roles: list[SnapshotRole] = []
    for index, raw_entry in enumerate(raw_entries):
        context = f"manifest.entries[{index}]"
        entry = _mapping(raw_entry, context)
        _exact_keys(
            entry,
            frozenset({"role", "relative_path", "size_bytes", "sha256"}),
            context,
        )
        role = _role(entry["role"], f"{context}.role")
        relative_path = str(
            _safe_relative_path(
                _string(entry["relative_path"], f"{context}.relative_path"),
                f"{context}.relative_path",
            )
        )
        size_bytes = _integer(entry["size_bytes"], f"{context}.size_bytes")
        digest = _sha256(entry["sha256"], f"{context}.sha256")
        path = snapshot_root.joinpath(*PurePosixPath(relative_path).parts)
        if path.is_symlink() or not path.is_file():
            raise SnapshotError(
                f"manifest file {relative_path!r} is absent or non-regular"
            )
        if path.stat().st_size != size_bytes or _file_sha256(path) != digest:
            raise SnapshotError(
                f"manifest file {relative_path!r} differs from recorded bytes"
            )
        paths.append(relative_path)
        roles.append(role)
        entries.append(ManifestEntry(role, relative_path, size_bytes, digest))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SnapshotError("manifest paths must be sorted and unique")
    if frozenset(roles) != _ROLES:
        raise SnapshotError("manifest does not contain every required execution role")
    copied_paths = tuple(
        sorted(
            path.relative_to(snapshot_root).as_posix()
            for path in snapshot_root.rglob("*")
            if path.is_file() and path != manifest_path
        )
    )
    if copied_paths != tuple(paths):
        raise SnapshotError("snapshot tree contains unmanifested or missing file bytes")
    return tuple(entries), _sha256_bytes(raw_bytes)


_ATTESTATION_SCRIPT: Final = """
import json
import importlib.machinery
import pathlib
import sys

snapshot = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(snapshot))
sys.path.insert(0, str(snapshot / "src"))
allowed_finders = (
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    importlib.machinery.PathFinder,
)
sys.meta_path[:] = [finder for finder in sys.meta_path if finder in allowed_finders]

import simsopt
import simsopt_jax
import simsopt_jax_adapters
import simsoptpp

print(json.dumps({
    "interpreter_path": str(pathlib.Path(sys.executable).absolute()),
    "python_version": sys.version,
    "origins": {
        "simsopt": simsopt.__file__,
        "simsopt_jax": simsopt_jax.__file__,
        "simsopt_jax_adapters": simsopt_jax_adapters.__file__,
        "simsoptpp": simsoptpp.__file__,
    },
}, sort_keys=True, separators=(",", ":")))
"""


def attest_snapshot_imports(
    snapshot_root: Path,
    *,
    interpreter: Path,
    output_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 120.0,
) -> ImportAttestation:
    """Run isolated direct imports and bind their origins to copied bytes."""

    snapshot_root = snapshot_root.resolve()
    entries, manifest_sha256 = load_snapshot_manifest(snapshot_root)
    entry_by_path = {entry.relative_path: entry for entry in entries}
    if not interpreter.is_absolute():
        raise SnapshotError("pinned interpreter path must be absolute")
    # Keep the venv entry-point path lexical. Resolving the symlink would run
    # the base interpreter without the selected venv's site-packages.
    interpreter = Path(os.path.abspath(interpreter))
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise SnapshotError("pinned interpreter must be an executable regular file")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise SnapshotError("attestation timeout must be finite and positive")
    completed = subprocess.run(
        [str(interpreter), "-I", "-c", _ATTESTATION_SCRIPT, str(snapshot_root)],
        cwd=snapshot_root,
        env=None if environment is None else dict(environment),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise SnapshotError(
            "pinned-interpreter import attestation failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr.strip()!r}"
        )
    try:
        child = _mapping(json.loads(completed.stdout), "child attestation")
    except json.JSONDecodeError as error:
        raise SnapshotError(
            "attestation child did not emit one JSON document"
        ) from error
    _exact_keys(
        child,
        frozenset({"interpreter_path", "python_version", "origins"}),
        "child attestation",
    )
    child_interpreter = Path(
        os.path.abspath(
            Path(
                _string(child["interpreter_path"], "child attestation.interpreter_path")
            )
        )
    )
    if child_interpreter != interpreter:
        raise SnapshotError("attestation child used a different interpreter")
    python_version = _string(
        child["python_version"], "child attestation.python_version"
    )
    origins = _mapping(child["origins"], "child attestation.origins")
    if frozenset(origins) != frozenset(_REQUIRED_IMPORTS):
        raise SnapshotError("attestation child omitted a required production import")
    bindings: list[ImportBinding] = []
    for module in _REQUIRED_IMPORTS:
        origin = Path(
            _string(origins[module], f"child attestation.origins.{module}")
        ).resolve()
        try:
            relative_path = origin.relative_to(snapshot_root).as_posix()
        except ValueError as error:
            raise SnapshotError(
                f"module {module!r} resolved outside the immutable snapshot"
            ) from error
        entry = entry_by_path.get(relative_path)
        if entry is None:
            raise SnapshotError(
                f"module {module!r} origin is absent from the source manifest"
            )
        bindings.append(
            ImportBinding(module, relative_path, entry.size_bytes, entry.sha256)
        )
    attestation = ImportAttestation(
        snapshot_manifest_sha256=manifest_sha256,
        interpreter_path=str(interpreter),
        python_version=python_version,
        bindings=tuple(bindings),
    )
    if output_path is not None:
        with output_path.open("xb") as stream:
            stream.write(canonical_json_bytes(attestation.to_json()))
    return attestation


__all__ = (
    "IMPORT_ATTESTATION_SCHEMA_ID",
    "MANIFEST_FILENAME",
    "SOURCE_MANIFEST_SCHEMA_ID",
    "ImportAttestation",
    "ImportBinding",
    "ManifestEntry",
    "RoleRoot",
    "SnapshotError",
    "SnapshotInput",
    "SnapshotPublication",
    "attest_snapshot_imports",
    "canonical_json_bytes",
    "expand_role_roots",
    "load_snapshot_manifest",
    "publish_immutable_snapshot",
)
