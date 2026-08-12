"""Produce the one-shot, CPU-only NEQ-GNTR3 trajectory qualification artifact."""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import platform
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

_WORKER_ENVIRONMENT: Final = "SIMSOPT_GNTR3_CPU_QUALIFICATION_WORKER_V1"
_PARENT_DESCRIPTOR_ENVIRONMENT: Final = (
    "SIMSOPT_GNTR3_CPU_QUALIFICATION_PARENT_DESCRIPTOR_V1"
)
_STAGING_DESCRIPTOR_ENVIRONMENT: Final = (
    "SIMSOPT_GNTR3_CPU_QUALIFICATION_STAGING_DESCRIPTOR_V1"
)
_WORKTREE_ROOT_ENVIRONMENT: Final = "SIMSOPT_GNTR3_CPU_QUALIFICATION_WORKTREE_ROOT_V1"
_EXECUTION_SOURCE_DIRECTORY: Final = "execution-source"
EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH: Final = (
    "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json"
)
EXECUTION_SOURCE_AUTHORITY_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-execution-source-authority-v1"
)
PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH: Final = "control/prequalification-plan.md"
_PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH: Final = (
    "docs/single_stage_jax_gpu_native_equivalent_quality_"
    "diag5_native_binding_recovery_plan.md"
)
_EXECUTION_SOURCE_DESCRIPTOR_ENVIRONMENT: Final = (
    "SIMSOPT_GNTR3_CPU_EXECUTION_SOURCE_DESCRIPTORS_V1"
)
_EXPECTED_OUTPUT_ROOT_TEXT: Final = (
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag5-cpu-qualification-20260812T090000Z"
)
_REQUIRED_ENVIRONMENT: Final = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}
_BROAD_EXECUTION_SOURCE_COUNTS: Final = (
    ("benchmarks", 113),
    ("src", 322),
    ("examples", 156),
)


class QualificationError(RuntimeError):
    """The one-shot qualification contract was not satisfied."""


def _lock_shared_descriptor(descriptor: int, context: str) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise QualificationError(f"{context} is exclusively locked") from error


def _unlock_close_descriptor(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class Publication:
    """Held parent and staging capabilities for one exact output publication."""

    staging_root: Path
    final_root: Path
    parent_descriptor: int
    staging_descriptor: int
    parent_device: int
    parent_inode: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class ExecutionSourceEntry:
    """One exact repository byte sequence authorized for worker execution."""

    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RetainedTreeFile:
    relative_path: str
    descriptor: int
    device: int
    inode: int
    size_bytes: int
    sha256: str
    mode: int
    link_count: int


@dataclass(frozen=True, slots=True)
class RetainedRegularTree:
    """One exact immutable input tree held by root and locked leaf descriptors."""

    source_root: Path
    root_descriptor: int
    root_device: int
    root_inode: int
    root_mode: int
    parent_descriptors: tuple[int, ...]
    directories: tuple[tuple[str, int, int, int], ...]
    files: tuple[RetainedTreeFile, ...]

    def validate(self) -> None:
        _validate_retained_regular_tree(self)

    def close(self) -> None:
        for entry in self.files:
            _unlock_close_descriptor(entry.descriptor)
        os.close(self.root_descriptor)
        for descriptor in self.parent_descriptors:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ExecutionSourceFileBinding:
    """Retained live and copied descriptors for one authorized source file."""

    entry: ExecutionSourceEntry
    live_path: Path
    live_descriptor: int
    live_device: int
    live_inode: int
    copied_path: Path | None = None
    copied_descriptor: int = -1
    copied_device: int = -1
    copied_inode: int = -1


@dataclass(frozen=True, slots=True)
class ExecutionSourceBindings:
    """Canonical authority plus every descriptor retained through publication."""

    worktree_root: Path
    worktree_descriptor: int
    worktree_device: int
    worktree_inode: int
    execution_root: Path | None
    execution_descriptor: int
    execution_device: int
    execution_inode: int
    authority_sha256: str
    entries_sha256: str
    manifest_live_path: Path
    manifest_live_descriptor: int
    manifest_live_device: int
    manifest_live_inode: int
    manifest_size_bytes: int
    manifest_copied_path: Path | None
    manifest_copied_descriptor: int
    manifest_copied_device: int
    manifest_copied_inode: int
    plan_source_path: Path
    plan_descriptor: int
    plan_device: int
    plan_inode: int
    plan_size_bytes: int
    plan_sha256: str
    plan_prefix_sha256: str
    entries: tuple[ExecutionSourceFileBinding, ...]

    def validate(
        self,
        *,
        copied_required: bool,
        copied_root: Path | None = None,
    ) -> None:
        _validate_execution_source_bindings(
            self,
            copied_required=copied_required,
            copied_root=copied_root,
        )

    def close(self) -> None:
        file_descriptors = [
            self.manifest_live_descriptor,
            self.plan_descriptor,
        ]
        if self.manifest_copied_descriptor >= 0:
            file_descriptors.append(self.manifest_copied_descriptor)
        for binding in self.entries:
            file_descriptors.append(binding.live_descriptor)
            if binding.copied_descriptor >= 0:
                file_descriptors.append(binding.copied_descriptor)
        for descriptor in file_descriptors:
            _unlock_close_descriptor(descriptor)
        if self.execution_descriptor >= 0:
            os.close(self.execution_descriptor)
        os.close(self.worktree_descriptor)


def _reject_symlink_components(path: Path) -> None:
    candidate = path.absolute()
    while True:
        if candidate.is_symlink():
            raise QualificationError(
                f"symlink path component is forbidden: {candidate}"
            )
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _prepare_publication(final_root: Path) -> Publication:
    """Atomically claim one deterministic staging name and retain both roots."""

    if not final_root.is_absolute() or not final_root.name:
        raise QualificationError("qualification output root must be absolute")
    _reject_symlink_components(final_root.parent)
    parent = final_root.parent.resolve(strict=True)
    requested = parent / final_root.name
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    staging_descriptor = -1
    try:
        parent_observed = os.fstat(parent_descriptor)
        names = frozenset(os.listdir(parent_descriptor))
        if requested.name in names:
            raise FileExistsError(
                f"qualification final root already exists: {requested}"
            )
        prefix = f"{requested.name}.partial-"
        if any(name.startswith(prefix) for name in names):
            raise FileExistsError("qualification staging sibling already exists")
        staging = parent / f"{prefix}claim"
        os.mkdir(staging.name, mode=0o755, dir_fd=parent_descriptor)
        staging_descriptor = os.open(
            staging.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        fcntl.flock(staging_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = os.fstat(staging_descriptor)
        publication = Publication(
            staging,
            requested,
            parent_descriptor,
            staging_descriptor,
            parent_observed.st_dev,
            parent_observed.st_ino,
            observed.st_dev,
            observed.st_ino,
        )
        _validate_publication_binding(publication, published=False)
        return publication
    except BaseException:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        os.close(parent_descriptor)
        raise


def _close_publication(publication: Publication) -> None:
    fcntl.flock(publication.staging_descriptor, fcntl.LOCK_UN)
    os.close(publication.staging_descriptor)
    os.close(publication.parent_descriptor)


def _validate_publication_binding(
    publication: Publication,
    *,
    published: bool,
) -> None:
    parent_observed = os.fstat(publication.parent_descriptor)
    try:
        parent_bound = publication.final_root.parent.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise QualificationError(
            "qualification output parent binding disappeared"
        ) from error
    if (
        not stat.S_ISDIR(parent_observed.st_mode)
        or (
            parent_observed.st_dev,
            parent_observed.st_ino,
        )
        != (publication.parent_device, publication.parent_inode)
        or (
            parent_bound.st_dev,
            parent_bound.st_ino,
        )
        != (publication.parent_device, publication.parent_inode)
    ):
        raise QualificationError("qualification output parent inode changed")
    locked = os.fstat(publication.staging_descriptor)
    name = publication.final_root.name if published else publication.staging_root.name
    try:
        bound = os.stat(
            name,
            dir_fd=publication.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise QualificationError("qualification staging binding disappeared") from error
    if (
        not stat.S_ISDIR(locked.st_mode)
        or (locked.st_dev, locked.st_ino) != (publication.device, publication.inode)
        or (bound.st_dev, bound.st_ino) != (publication.device, publication.inode)
    ):
        raise QualificationError("qualification staging inode changed")
    other = publication.staging_root.name if published else publication.final_root.name
    try:
        os.stat(
            other,
            dir_fd=publication.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise QualificationError("qualification publication names overlap")


def _bootstrap_constant(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != name or node.value is None:
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and len(value.args) == 1
            and not value.keywords
        ):
            return frozenset(ast.literal_eval(value.args[0]))
        return ast.literal_eval(value)
    raise QualificationError(f"bootstrap source constant is absent: {name}")


def _bootstrap_canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _bootstrap_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise QualificationError("execution-source authority has duplicate keys")
        payload[key] = value
    return payload


def _bootstrap_json(payload: bytes) -> object:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_bootstrap_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                QualificationError(f"execution-source authority contains {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(
            "execution-source authority is not UTF-8 JSON"
        ) from error
    if _bootstrap_canonical_json_bytes(decoded) != payload:
        raise QualificationError("execution-source authority is not canonical")
    return decoded


def _bootstrap_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
        or any(
            part.startswith(".env") and part != ".env.example" for part in path.parts
        )
    ):
        raise QualificationError(f"execution-source path is not canonical: {value!r}")
    return path


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _descriptor_bytes(descriptor: int, size_bytes: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size_bytes:
        chunk = os.pread(descriptor, min(1024 * 1024, size_bytes - offset), offset)
        if not chunk:
            raise QualificationError("retained execution-source descriptor truncated")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _open_relative_directory(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> int:
    current = os.dup(root_descriptor)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=current)
                except FileExistsError:
                    pass
            following = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _open_relative_regular(
    root_descriptor: int,
    relative_path: str,
    *,
    flags: int,
    mode: int = 0o444,
    create_parents: bool = False,
) -> int:
    relative = _bootstrap_relative_path(relative_path)
    parent = _open_relative_directory(
        root_descriptor,
        tuple(relative.parts[:-1]),
        create=create_parents,
    )
    try:
        return os.open(
            relative.name,
            flags | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode,
            dir_fd=parent,
        )
    finally:
        os.close(parent)


def _open_absolute_regular(path: Path) -> tuple[int, tuple[int, ...]]:
    """Open one absolute leaf beneath a retained no-symlink directory chain."""

    if not path.is_absolute() or path.name in ("", ".", ".."):
        raise QualificationError("native extension path must be absolute")
    descriptor = -1
    directory_descriptors = [
        os.open(
            "/",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    ]
    try:
        for part in path.parts[1:-1]:
            directory_descriptors.append(
                os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_descriptors[-1],
                )
            )
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptors[-1],
        )
        _lock_shared_descriptor(descriptor, "native extension")
        return descriptor, tuple(directory_descriptors)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        for directory_descriptor in directory_descriptors:
            os.close(directory_descriptor)
        raise


def _open_absolute_directory(path: Path) -> tuple[int, tuple[int, ...]]:
    """Open one absolute directory and retain its no-symlink parent chain."""

    if not path.is_absolute() or path == Path("/"):
        raise QualificationError("retained tree root must be a non-root absolute path")
    descriptors = [
        os.open(
            "/",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    ]
    try:
        for part in path.parts[1:]:
            descriptors.append(
                os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=descriptors[-1],
                )
            )
        return descriptors[-1], tuple(descriptors[:-1])
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def _validate_absolute_directory_chain(tree: RetainedRegularTree) -> None:
    descriptors = (*tree.parent_descriptors, tree.root_descriptor)
    if len(descriptors) != len(tree.source_root.parts):
        raise QualificationError("retained tree directory binding differs")
    for index, descriptor in enumerate(descriptors):
        locked = os.fstat(descriptor)
        if not stat.S_ISDIR(locked.st_mode):
            raise QualificationError("retained tree directory binding differs")
        if index:
            bound = os.stat(
                tree.source_root.parts[index],
                dir_fd=descriptors[index - 1],
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(bound.st_mode) or (locked.st_dev, locked.st_ino) != (
                bound.st_dev,
                bound.st_ino,
            ):
                raise QualificationError("retained tree directory binding differs")
    root = os.fstat(tree.root_descriptor)
    if (root.st_dev, root.st_ino) != (
        tree.root_device,
        tree.root_inode,
    ) or stat.S_IMODE(root.st_mode) != tree.root_mode:
        raise QualificationError("retained tree root identity differs")


def _validate_absolute_regular_chain(
    path: Path,
    descriptor: int,
    directory_descriptors: tuple[int, ...],
) -> os.stat_result:
    """Rebind an absolute leaf to its retained root-down directory capability."""

    if len(directory_descriptors) != len(path.parts) - 1:
        raise QualificationError("native extension directory binding differs")
    try:
        for index, directory_descriptor in enumerate(directory_descriptors):
            locked_directory = os.fstat(directory_descriptor)
            if not stat.S_ISDIR(locked_directory.st_mode):
                raise QualificationError("native extension directory binding differs")
            if index:
                bound_directory = os.stat(
                    path.parts[index],
                    dir_fd=directory_descriptors[index - 1],
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(bound_directory.st_mode) or (
                    locked_directory.st_dev,
                    locked_directory.st_ino,
                ) != (bound_directory.st_dev, bound_directory.st_ino):
                    raise QualificationError(
                        "native extension directory binding differs"
                    )
        locked = os.fstat(descriptor)
        bound = os.stat(
            path.name,
            dir_fd=directory_descriptors[-1],
            follow_symlinks=False,
        )
    except OSError as error:
        raise QualificationError(
            "native extension directory binding differs"
        ) from error
    if (
        not stat.S_ISREG(locked.st_mode)
        or not stat.S_ISREG(bound.st_mode)
        or (locked.st_dev, locked.st_ino, locked.st_size)
        != (bound.st_dev, bound.st_ino, bound.st_size)
    ):
        raise QualificationError("native extension path binding differs")
    return bound


def _observe_absolute_regular(path: Path) -> tuple[os.stat_result, str]:
    descriptor, directory_descriptors = _open_absolute_regular(path)
    try:
        observed = _validate_absolute_regular_chain(
            path,
            descriptor,
            directory_descriptors,
        )
        return observed, _descriptor_sha256(descriptor)
    finally:
        _unlock_close_descriptor(descriptor)
        for directory_descriptor in directory_descriptors:
            os.close(directory_descriptor)


def _relative_regular_stat(root_descriptor: int, relative_path: str) -> os.stat_result:
    descriptor = _open_relative_regular(
        root_descriptor,
        relative_path,
        flags=os.O_RDONLY,
    )
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _bound_regular_descriptor(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
) -> tuple[int, int, int]:
    _reject_symlink_components(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        _lock_shared_descriptor(descriptor, f"execution source {path}")
        locked = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        identity = (locked.st_dev, locked.st_ino, locked.st_size)
        if (
            not stat.S_ISREG(locked.st_mode)
            or locked.st_nlink != 1
            or identity != (bound.st_dev, bound.st_ino, bound.st_size)
            or locked.st_size != size_bytes
            or _descriptor_sha256(descriptor) != sha256
        ):
            raise QualificationError(f"execution-source binding differs: {path}")
        return descriptor, locked.st_dev, locked.st_ino
    except BaseException:
        _unlock_close_descriptor(descriptor)
        raise


def _bound_relative_regular_descriptor(
    root_descriptor: int,
    relative_path: str,
    *,
    size_bytes: int,
    sha256: str,
) -> tuple[int, int, int]:
    descriptor = _open_relative_regular(
        root_descriptor,
        relative_path,
        flags=os.O_RDONLY,
    )
    try:
        _lock_shared_descriptor(descriptor, f"execution source {relative_path}")
        locked = os.fstat(descriptor)
        if (
            not stat.S_ISREG(locked.st_mode)
            or locked.st_nlink != 1
            or locked.st_size != size_bytes
            or _descriptor_sha256(descriptor) != sha256
        ):
            raise QualificationError(
                f"execution-source binding differs: {relative_path}"
            )
        return descriptor, locked.st_dev, locked.st_ino
    except BaseException:
        _unlock_close_descriptor(descriptor)
        raise


def _observed_regular_descriptor(path: Path) -> tuple[int, int, int, int, str]:
    _reject_symlink_components(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _lock_shared_descriptor(descriptor, f"observed source {path}")
        locked = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(locked.st_mode)
            or locked.st_nlink != 1
            or (locked.st_dev, locked.st_ino, locked.st_size)
            != (bound.st_dev, bound.st_ino, bound.st_size)
        ):
            raise QualificationError(f"observed source binding differs: {path}")
        return (
            descriptor,
            locked.st_dev,
            locked.st_ino,
            locked.st_size,
            _descriptor_sha256(descriptor),
        )
    except BaseException:
        _unlock_close_descriptor(descriptor)
        raise


def _observed_relative_regular_descriptor(
    root_descriptor: int,
    relative_path: str,
) -> tuple[int, int, int, int, str]:
    descriptor = _open_relative_regular(
        root_descriptor,
        relative_path,
        flags=os.O_RDONLY,
    )
    try:
        _lock_shared_descriptor(descriptor, f"observed source {relative_path}")
        locked = os.fstat(descriptor)
        if not stat.S_ISREG(locked.st_mode) or locked.st_nlink != 1:
            raise QualificationError(
                f"observed source binding differs: {relative_path}"
            )
        return (
            descriptor,
            locked.st_dev,
            locked.st_ino,
            locked.st_size,
            _descriptor_sha256(descriptor),
        )
    except BaseException:
        _unlock_close_descriptor(descriptor)
        raise


def _parse_execution_source_authority(
    manifest_payload: bytes,
) -> tuple[tuple[ExecutionSourceEntry, ...], str]:
    decoded = _bootstrap_json(manifest_payload)
    if not isinstance(decoded, dict) or frozenset(decoded) != frozenset(
        {"schema_version", "entries", "entries_sha256"}
    ):
        raise QualificationError("execution-source authority fields differ")
    raw_entries = decoded["entries"]
    entries_sha256 = decoded["entries_sha256"]
    if (
        decoded["schema_version"] != EXECUTION_SOURCE_AUTHORITY_SCHEMA_VERSION
        or not isinstance(raw_entries, dict)
        or not isinstance(entries_sha256, str)
        or len(entries_sha256) != 64
        or any(character not in "0123456789abcdef" for character in entries_sha256)
        or hashlib.sha256(_bootstrap_canonical_json_bytes(raw_entries)).hexdigest()
        != entries_sha256
    ):
        raise QualificationError("execution-source authority identity differs")
    if EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH in raw_entries:
        raise QualificationError("execution-source authority contains itself")
    entries: list[ExecutionSourceEntry] = []
    for relative, raw_entry in raw_entries.items():
        if not isinstance(relative, str):
            raise QualificationError("execution-source path must be a string")
        _bootstrap_relative_path(relative)
        if not isinstance(raw_entry, dict) or frozenset(raw_entry) != frozenset(
            {"sha256", "size_bytes"}
        ):
            raise QualificationError(f"execution-source entry differs: {relative}")
        sha256 = raw_entry["sha256"]
        size_bytes = raw_entry["size_bytes"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise QualificationError(
                f"execution-source entry has invalid types: {relative}"
            )
        entries.append(ExecutionSourceEntry(relative, sha256, size_bytes))
    entries.sort(key=lambda entry: entry.relative_path)
    if any(
        entry.relative_path == _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH
        for entry in entries
    ):
        raise QualificationError("execution-source authority membership differs")
    return tuple(entries), entries_sha256


def _validate_execution_source_membership(
    worktree_descriptor: int,
    entries: tuple[ExecutionSourceEntry, ...],
    authority_source_payload: bytes,
) -> None:
    broad: set[str] = set()
    counts: dict[str, int] = {}
    for directory_name, expected_count in _BROAD_EXECUTION_SOURCE_COUNTS:
        directory_descriptor = _open_relative_directory(
            worktree_descriptor,
            (directory_name,),
            create=False,
        )
        try:
            directory_root = Path(f"/proc/self/fd/{directory_descriptor}")
            directory_paths: set[str] = set()
            for path in directory_root.rglob("*.py"):
                metadata = path.stat(follow_symlinks=False)
                if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                    raise QualificationError(
                        f"execution-source membership contains an invalid path: {path}"
                    )
                directory_paths.add(
                    PurePosixPath(
                        directory_name, path.relative_to(directory_root)
                    ).as_posix()
                )
            if len(directory_paths) != expected_count:
                raise QualificationError(
                    f"execution-source {directory_name} membership differs"
                )
            counts[directory_name] = len(directory_paths)
            broad.update(directory_paths)
        finally:
            os.close(directory_descriptor)
    authority_tree = ast.parse(
        authority_source_payload,
        filename=(
            "benchmarks/single_stage_native_equivalent_quality_successor_authority.py"
        ),
    )
    selected = set(broad)
    for name in ("DIAG5_QUALIFIED_FILE_PATHS", "DIAG5_FROZEN_NUMERICAL_PATHS"):
        value = _bootstrap_constant(authority_tree, name)
        if not isinstance(value, frozenset) or not all(
            isinstance(item, str) for item in value
        ):
            raise QualificationError(
                f"execution-source membership constant differs: {name}"
            )
        selected.update(value)
    selected.discard(EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH)
    selected.discard(_PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH)
    expected_entry_count = _bootstrap_constant(
        authority_tree,
        "DIAG5_EXECUTION_SOURCE_ENTRY_COUNT",
    )
    if type(expected_entry_count) is not int or expected_entry_count <= 0:
        raise QualificationError(
            "execution-source membership constant differs: "
            "DIAG5_EXECUTION_SOURCE_ENTRY_COUNT"
        )
    if (
        counts != dict(_BROAD_EXECUTION_SOURCE_COUNTS)
        or "src/simsopt/_version.py" not in selected
        or len(entries) != expected_entry_count
        or selected != {entry.relative_path for entry in entries}
    ):
        raise QualificationError("execution-source exact membership differs")


def _load_execution_source_authority(repository: Path) -> ExecutionSourceBindings:
    """Bind every manifest byte before the one-shot output claim."""

    worktree = repository.resolve(strict=True)
    worktree_descriptor = os.open(
        worktree,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    worktree_stat = os.fstat(worktree_descriptor)
    manifest_path = worktree / EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH
    try:
        (
            manifest_descriptor,
            manifest_device,
            manifest_inode,
            manifest_size_bytes,
            manifest_sha256,
        ) = _observed_relative_regular_descriptor(
            worktree_descriptor,
            EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
        )
    except BaseException:
        os.close(worktree_descriptor)
        raise
    manifest_payload = _descriptor_bytes(
        manifest_descriptor,
        manifest_size_bytes,
    )
    entries, entries_sha256 = _parse_execution_source_authority(manifest_payload)
    bindings: list[ExecutionSourceFileBinding] = []
    plan_descriptor = -1
    try:
        for entry in entries:
            path = worktree.joinpath(*PurePosixPath(entry.relative_path).parts)
            descriptor, device, inode = _bound_relative_regular_descriptor(
                worktree_descriptor,
                entry.relative_path,
                size_bytes=entry.size_bytes,
                sha256=entry.sha256,
            )
            bindings.append(
                ExecutionSourceFileBinding(
                    entry=entry,
                    live_path=path,
                    live_descriptor=descriptor,
                    live_device=device,
                    live_inode=inode,
                )
            )
        authority_binding = next(
            (
                binding
                for binding in bindings
                if binding.entry.relative_path == "benchmarks/"
                "single_stage_native_equivalent_quality_successor_authority.py"
            ),
            None,
        )
        if authority_binding is None:
            raise QualificationError(
                "required source is absent from execution authority"
            )
        authority_payload = _descriptor_bytes(
            authority_binding.live_descriptor,
            authority_binding.entry.size_bytes,
        )
        _validate_execution_source_membership(
            worktree_descriptor,
            entries,
            authority_payload,
        )
        authority_tree = ast.parse(
            authority_payload,
            filename=str(authority_binding.live_path),
        )
        expected_plan_prefix = _bootstrap_constant(
            authority_tree,
            "DIAG5_PLAN_SHA256",
        )
        if not isinstance(expected_plan_prefix, str):
            raise QualificationError("prequalification plan prefix is invalid")
        plan_source_path = worktree / _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH
        (
            plan_descriptor,
            plan_device,
            plan_inode,
            plan_size_bytes,
            plan_sha256,
        ) = _observed_relative_regular_descriptor(
            worktree_descriptor,
            _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH,
        )
        plan_payload = _descriptor_bytes(plan_descriptor, plan_size_bytes)
        plan_prefix, marker, record = plan_payload.partition(
            b"## Qualification Record\n"
        )
        if (
            not marker
            or record
            or hashlib.sha256(plan_prefix).hexdigest() != expected_plan_prefix
        ):
            raise QualificationError("prequalification plan control differs")
        authority = ExecutionSourceBindings(
            worktree_root=worktree,
            worktree_descriptor=worktree_descriptor,
            worktree_device=worktree_stat.st_dev,
            worktree_inode=worktree_stat.st_ino,
            execution_root=None,
            execution_descriptor=-1,
            execution_device=-1,
            execution_inode=-1,
            authority_sha256=manifest_sha256,
            entries_sha256=entries_sha256,
            manifest_live_path=manifest_path,
            manifest_live_descriptor=manifest_descriptor,
            manifest_live_device=manifest_device,
            manifest_live_inode=manifest_inode,
            manifest_size_bytes=manifest_size_bytes,
            manifest_copied_path=None,
            manifest_copied_descriptor=-1,
            manifest_copied_device=-1,
            manifest_copied_inode=-1,
            plan_source_path=plan_source_path,
            plan_descriptor=plan_descriptor,
            plan_device=plan_device,
            plan_inode=plan_inode,
            plan_size_bytes=plan_size_bytes,
            plan_sha256=plan_sha256,
            plan_prefix_sha256=expected_plan_prefix,
            entries=tuple(bindings),
        )
        authority.validate(copied_required=False)
        return authority
    except BaseException:
        if plan_descriptor >= 0:
            _unlock_close_descriptor(plan_descriptor)
        for binding in bindings:
            _unlock_close_descriptor(binding.live_descriptor)
        _unlock_close_descriptor(manifest_descriptor)
        os.close(worktree_descriptor)
        raise


def _validate_retained_source(
    path: Path,
    descriptor: int,
    device: int,
    inode: int,
    *,
    size_bytes: int,
    sha256: str,
    root_descriptor: int | None = None,
    relative_path: str | None = None,
) -> None:
    locked = os.fstat(descriptor)
    if (root_descriptor is None) is not (relative_path is None):
        raise QualificationError("retained source root binding is incomplete")
    bound = (
        path.stat(follow_symlinks=False)
        if root_descriptor is None
        else _relative_regular_stat(root_descriptor, relative_path)
    )
    expected_identity = (device, inode, size_bytes)
    if (
        not stat.S_ISREG(locked.st_mode)
        or locked.st_nlink != 1
        or (locked.st_dev, locked.st_ino, locked.st_size) != expected_identity
        or (bound.st_dev, bound.st_ino, bound.st_size) != expected_identity
        or _descriptor_sha256(descriptor) != sha256
    ):
        raise QualificationError(f"retained execution-source changed: {path}")


def _regular_tree_membership(root_descriptor: int) -> frozenset[str]:
    root = Path(f"/proc/self/fd/{root_descriptor}")
    observed: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            path = current / name
            metadata = path.stat(follow_symlinks=False)
            if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise QualificationError(
                    f"execution-source tree contains an invalid directory: {path}"
                )
        for name in file_names:
            path = current / name
            metadata = path.stat(follow_symlinks=False)
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise QualificationError(
                    f"execution-source tree contains an invalid file: {path}"
                )
            observed.add(path.relative_to(root).as_posix())
    return frozenset(observed)


def _retained_tree_inventory(
    root_descriptor: int,
) -> tuple[tuple[tuple[str, int, int, int], ...], tuple[str, ...]]:
    directories: list[tuple[str, int, int, int]] = []
    files: list[str] = []

    def visit(directory_descriptor: int, relative_parts: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(directory_descriptor)):
            observed = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            relative = PurePosixPath(*relative_parts, name).as_posix()
            if stat.S_ISDIR(observed.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_descriptor,
                )
                try:
                    locked = os.fstat(child)
                    if (locked.st_dev, locked.st_ino) != (
                        observed.st_dev,
                        observed.st_ino,
                    ):
                        raise QualificationError(
                            "retained input directory changed during admission"
                        )
                    directories.append(
                        (
                            relative,
                            locked.st_dev,
                            locked.st_ino,
                            stat.S_IMODE(locked.st_mode),
                        )
                    )
                    visit(child, (*relative_parts, name))
                finally:
                    os.close(child)
            elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                files.append(relative)
            else:
                raise QualificationError("retained input tree contains an invalid leaf")

    visit(root_descriptor, ())
    return tuple(sorted(directories)), tuple(sorted(files))


def _admit_regular_tree(source_root: Path) -> RetainedRegularTree:
    root = source_root.absolute()
    root_descriptor, parent_descriptors = _open_absolute_directory(root)
    opened: list[RetainedTreeFile] = []
    try:
        root_stat = os.fstat(root_descriptor)
        directories, relative_files = _retained_tree_inventory(root_descriptor)
        for relative in relative_files:
            descriptor = _open_relative_regular(
                root_descriptor,
                relative,
                flags=os.O_RDONLY,
            )
            try:
                _lock_shared_descriptor(descriptor, f"retained input {relative}")
                observed = os.fstat(descriptor)
                if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                    raise QualificationError("retained input leaf differs")
                opened.append(
                    RetainedTreeFile(
                        relative,
                        descriptor,
                        observed.st_dev,
                        observed.st_ino,
                        observed.st_size,
                        _descriptor_sha256(descriptor),
                        stat.S_IMODE(observed.st_mode),
                        observed.st_nlink,
                    )
                )
            except BaseException:
                if not opened or opened[-1].descriptor != descriptor:
                    _unlock_close_descriptor(descriptor)
                raise
        tree = RetainedRegularTree(
            source_root=root,
            root_descriptor=root_descriptor,
            root_device=root_stat.st_dev,
            root_inode=root_stat.st_ino,
            root_mode=stat.S_IMODE(root_stat.st_mode),
            parent_descriptors=parent_descriptors,
            directories=directories,
            files=tuple(opened),
        )
        tree.validate()
        return tree
    except BaseException:
        for entry in opened:
            _unlock_close_descriptor(entry.descriptor)
        os.close(root_descriptor)
        for descriptor in parent_descriptors:
            os.close(descriptor)
        raise


def _validate_retained_regular_tree(tree: RetainedRegularTree) -> None:
    _validate_absolute_directory_chain(tree)
    directories, relative_files = _retained_tree_inventory(tree.root_descriptor)
    if directories != tree.directories or relative_files != tuple(
        entry.relative_path for entry in tree.files
    ):
        raise QualificationError("retained input tree membership differs")
    for entry in tree.files:
        _validate_retained_source(
            tree.source_root / entry.relative_path,
            entry.descriptor,
            entry.device,
            entry.inode,
            size_bytes=entry.size_bytes,
            sha256=entry.sha256,
            root_descriptor=tree.root_descriptor,
            relative_path=entry.relative_path,
        )
        observed = os.fstat(entry.descriptor)
        if (
            stat.S_IMODE(observed.st_mode) != entry.mode
            or observed.st_nlink != entry.link_count
        ):
            raise QualificationError("retained input leaf topology differs")


def _validate_execution_source_bindings(
    bindings: ExecutionSourceBindings,
    *,
    copied_required: bool,
    copied_root: Path | None = None,
) -> None:
    worktree_stat = os.fstat(bindings.worktree_descriptor)
    worktree_path_stat = bindings.worktree_root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(worktree_stat.st_mode)
        or (worktree_stat.st_dev, worktree_stat.st_ino)
        != (bindings.worktree_device, bindings.worktree_inode)
        or (worktree_path_stat.st_dev, worktree_path_stat.st_ino)
        != (bindings.worktree_device, bindings.worktree_inode)
    ):
        raise QualificationError("execution-source worktree root changed")
    _validate_retained_source(
        bindings.manifest_live_path,
        bindings.manifest_live_descriptor,
        bindings.manifest_live_device,
        bindings.manifest_live_inode,
        size_bytes=bindings.manifest_size_bytes,
        sha256=bindings.authority_sha256,
        root_descriptor=bindings.worktree_descriptor,
        relative_path=EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
    )
    _validate_retained_source(
        bindings.plan_source_path,
        bindings.plan_descriptor,
        bindings.plan_device,
        bindings.plan_inode,
        size_bytes=bindings.plan_size_bytes,
        sha256=bindings.plan_sha256,
        root_descriptor=bindings.worktree_descriptor,
        relative_path=_PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH,
    )
    plan_payload = _descriptor_bytes(bindings.plan_descriptor, bindings.plan_size_bytes)
    plan_prefix, marker, record = plan_payload.partition(b"## Qualification Record\n")
    if (
        not marker
        or record
        or hashlib.sha256(plan_prefix).hexdigest() != bindings.plan_prefix_sha256
    ):
        raise QualificationError("retained prequalification plan changed")
    for binding in bindings.entries:
        _validate_retained_source(
            binding.live_path,
            binding.live_descriptor,
            binding.live_device,
            binding.live_inode,
            size_bytes=binding.entry.size_bytes,
            sha256=binding.entry.sha256,
            root_descriptor=bindings.worktree_descriptor,
            relative_path=binding.entry.relative_path,
        )
    authority_binding = next(
        (
            binding
            for binding in bindings.entries
            if binding.entry.relative_path == "benchmarks/"
            "single_stage_native_equivalent_quality_successor_authority.py"
        ),
        None,
    )
    if authority_binding is None:
        raise QualificationError("retained successor authority is absent")
    _validate_execution_source_membership(
        bindings.worktree_descriptor,
        tuple(binding.entry for binding in bindings.entries),
        _descriptor_bytes(
            authority_binding.live_descriptor,
            authority_binding.entry.size_bytes,
        ),
    )
    if not copied_required:
        return
    if (
        bindings.execution_root is None
        or bindings.manifest_copied_path is None
        or bindings.manifest_copied_descriptor < 0
    ):
        raise QualificationError("copied execution-source bindings are absent")
    execution_root = bindings.execution_root if copied_root is None else copied_root
    execution_stat = os.fstat(bindings.execution_descriptor)
    execution_path_stat = execution_root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(execution_stat.st_mode)
        or (execution_stat.st_dev, execution_stat.st_ino)
        != (bindings.execution_device, bindings.execution_inode)
        or (execution_path_stat.st_dev, execution_path_stat.st_ino)
        != (bindings.execution_device, bindings.execution_inode)
    ):
        raise QualificationError("copied execution-source root changed")
    manifest_copied_path = execution_root.joinpath(
        *PurePosixPath(EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH).parts
    )
    _validate_retained_source(
        manifest_copied_path,
        bindings.manifest_copied_descriptor,
        bindings.manifest_copied_device,
        bindings.manifest_copied_inode,
        size_bytes=bindings.manifest_size_bytes,
        sha256=bindings.authority_sha256,
        root_descriptor=bindings.execution_descriptor,
        relative_path=EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
    )
    for binding in bindings.entries:
        if binding.copied_path is None or binding.copied_descriptor < 0:
            raise QualificationError(
                f"copied execution-source binding is absent: {binding.entry.relative_path}"
            )
        copied_path = execution_root.joinpath(
            *PurePosixPath(binding.entry.relative_path).parts
        )
        _validate_retained_source(
            copied_path,
            binding.copied_descriptor,
            binding.copied_device,
            binding.copied_inode,
            size_bytes=binding.entry.size_bytes,
            sha256=binding.entry.sha256,
            root_descriptor=bindings.execution_descriptor,
            relative_path=binding.entry.relative_path,
        )
    expected_paths = frozenset(
        {
            EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
            *(binding.entry.relative_path for binding in bindings.entries),
        }
    )
    if _regular_tree_membership(bindings.execution_descriptor) != expected_paths:
        raise QualificationError("copied execution-source membership differs")


def _copy_descriptor(
    root_descriptor: int,
    relative_path: str,
    source_descriptor: int,
    size_bytes: int,
) -> None:
    destination = _open_relative_regular(
        root_descriptor,
        relative_path,
        flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode=0o444,
        create_parents=True,
    )
    try:
        offset = 0
        while offset < size_bytes:
            payload = os.pread(
                source_descriptor, min(1024 * 1024, size_bytes - offset), offset
            )
            if not payload:
                raise QualificationError(
                    f"execution-source truncated while copied: {relative_path}"
                )
            written = 0
            while written < len(payload):
                written += os.write(destination, payload[written:])
            offset += len(payload)
        os.fsync(destination)
    finally:
        os.close(destination)


def _bootstrap_copy_execution_source(
    authority: ExecutionSourceBindings,
    staging_root: Path,
) -> ExecutionSourceBindings:
    execution_root = staging_root / _EXECUTION_SOURCE_DIRECTORY
    execution_descriptor = -1
    try:
        execution_root.mkdir(mode=0o755, exist_ok=False)
        execution_descriptor = os.open(
            execution_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except BaseException:
        authority.close()
        raise
    execution_stat = os.fstat(execution_descriptor)
    directories = {execution_root}
    copied_bindings: list[ExecutionSourceFileBinding] = []
    try:
        for binding in authority.entries:
            destination = execution_root.joinpath(
                *PurePosixPath(binding.entry.relative_path).parts
            )
            _copy_descriptor(
                execution_descriptor,
                binding.entry.relative_path,
                binding.live_descriptor,
                binding.entry.size_bytes,
            )
            directories.update(destination.parents)
            descriptor, device, inode = _bound_relative_regular_descriptor(
                execution_descriptor,
                binding.entry.relative_path,
                size_bytes=binding.entry.size_bytes,
                sha256=binding.entry.sha256,
            )
            copied_bindings.append(
                ExecutionSourceFileBinding(
                    entry=binding.entry,
                    live_path=binding.live_path,
                    live_descriptor=binding.live_descriptor,
                    live_device=binding.live_device,
                    live_inode=binding.live_inode,
                    copied_path=destination,
                    copied_descriptor=descriptor,
                    copied_device=device,
                    copied_inode=inode,
                )
            )
        copied_manifest = execution_root.joinpath(
            *PurePosixPath(EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH).parts
        )
        _copy_descriptor(
            execution_descriptor,
            EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
            authority.manifest_live_descriptor,
            authority.manifest_size_bytes,
        )
        directories.update(copied_manifest.parents)
        (
            manifest_descriptor,
            manifest_device,
            manifest_inode,
        ) = _bound_relative_regular_descriptor(
            execution_descriptor,
            EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
            size_bytes=authority.manifest_size_bytes,
            sha256=authority.authority_sha256,
        )
    except BaseException:
        for binding in copied_bindings:
            _unlock_close_descriptor(binding.copied_descriptor)
        if execution_descriptor >= 0:
            os.close(execution_descriptor)
        authority.close()
        raise
    try:
        for directory in sorted(
            (path for path in directories if path.is_relative_to(execution_root)),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
            descriptor = os.open(
                directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        copied = ExecutionSourceBindings(
            worktree_root=authority.worktree_root,
            worktree_descriptor=authority.worktree_descriptor,
            worktree_device=authority.worktree_device,
            worktree_inode=authority.worktree_inode,
            execution_root=execution_root,
            execution_descriptor=execution_descriptor,
            execution_device=execution_stat.st_dev,
            execution_inode=execution_stat.st_ino,
            authority_sha256=authority.authority_sha256,
            entries_sha256=authority.entries_sha256,
            manifest_live_path=authority.manifest_live_path,
            manifest_live_descriptor=authority.manifest_live_descriptor,
            manifest_live_device=authority.manifest_live_device,
            manifest_live_inode=authority.manifest_live_inode,
            manifest_size_bytes=authority.manifest_size_bytes,
            manifest_copied_path=copied_manifest,
            manifest_copied_descriptor=manifest_descriptor,
            manifest_copied_device=manifest_device,
            manifest_copied_inode=manifest_inode,
            plan_source_path=authority.plan_source_path,
            plan_descriptor=authority.plan_descriptor,
            plan_device=authority.plan_device,
            plan_inode=authority.plan_inode,
            plan_size_bytes=authority.plan_size_bytes,
            plan_sha256=authority.plan_sha256,
            plan_prefix_sha256=authority.plan_prefix_sha256,
            entries=tuple(copied_bindings),
        )
        copied.validate(copied_required=True)
        return copied
    except BaseException:
        _unlock_close_descriptor(manifest_descriptor)
        for binding in copied_bindings:
            _unlock_close_descriptor(binding.copied_descriptor)
        os.close(execution_descriptor)
        authority.close()
        raise


def _execution_source_descriptor_payload(
    bindings: ExecutionSourceBindings,
) -> bytes:
    return _bootstrap_canonical_json_bytes(
        {
            "authority_sha256": bindings.authority_sha256,
            "entries": {
                binding.entry.relative_path: {
                    "copied_descriptor": binding.copied_descriptor,
                    "live_descriptor": binding.live_descriptor,
                }
                for binding in bindings.entries
            },
            "entries_sha256": bindings.entries_sha256,
            "execution_descriptor": bindings.execution_descriptor,
            "manifest_copied_descriptor": bindings.manifest_copied_descriptor,
            "manifest_live_descriptor": bindings.manifest_live_descriptor,
            "plan_descriptor": bindings.plan_descriptor,
            "worktree_descriptor": bindings.worktree_descriptor,
        }
    )


def _neutralize_editable_source_redirection() -> tuple[str, ...]:
    """Drop editable-install redirecting finders after native-extension import.

    scikit-build-core editable installs register a meta-path finder that
    resolves the repository packages to the live worktree ahead of every
    ``sys.path`` entry, escaping the sealed execution-source tree. The native
    extension must already be imported through its installed loader before
    this runs; every later production import then binds to ``PYTHONPATH``.
    """

    removed = tuple(
        type(finder).__name__
        for finder in sys.meta_path
        if "ScikitBuild" in type(finder).__name__
    )
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if "ScikitBuild" not in type(finder).__name__
    ]
    return removed


def _direct_bootstrap() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    expected = Path(_EXPECTED_OUTPUT_ROOT_TEXT)
    if arguments.output_root != expected:
        raise QualificationError(f"output root must be exactly {expected}")
    for name, value in _REQUIRED_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise QualificationError(f"{name} must equal {value!r}")
    repository = Path(__file__).resolve(strict=True).parents[1]
    publication = _prepare_publication(arguments.output_root)
    copied: ExecutionSourceBindings | None = None
    try:
        authority = _load_execution_source_authority(repository)
        copied = _bootstrap_copy_execution_source(
            authority,
            publication.staging_root,
        )
        if copied.execution_root is None:
            raise QualificationError("copied execution-source root is absent")
        _validate_publication_binding(publication, published=False)
        copied.validate(copied_required=True)
        descriptors = [
            publication.parent_descriptor,
            publication.staging_descriptor,
            copied.manifest_live_descriptor,
            copied.manifest_copied_descriptor,
            copied.plan_descriptor,
            copied.worktree_descriptor,
            copied.execution_descriptor,
        ]
        for binding in copied.entries:
            descriptors.extend((binding.live_descriptor, binding.copied_descriptor))
        for descriptor in descriptors:
            os.set_inheritable(descriptor, True)
        environment = dict(os.environ)
        environment[_WORKER_ENVIRONMENT] = "1"
        environment[_PARENT_DESCRIPTOR_ENVIRONMENT] = str(publication.parent_descriptor)
        environment[_STAGING_DESCRIPTOR_ENVIRONMENT] = str(
            publication.staging_descriptor
        )
        environment[_WORKTREE_ROOT_ENVIRONMENT] = str(repository)
        environment[_EXECUTION_SOURCE_DESCRIPTOR_ENVIRONMENT] = (
            _execution_source_descriptor_payload(copied).decode("utf-8")
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(copied.execution_root / "src"), str(copied.execution_root))
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        os.chdir(copied.execution_root)
        worker = copied.execution_root / Path(__file__).relative_to(repository)
        os.execve(
            sys.executable,
            (sys.executable, "-B", str(worker), *sys.argv[1:]),
            environment,
        )
    except BaseException:
        if copied is not None:
            copied.close()
        _close_publication(publication)
        raise


if __name__ == "__main__" and os.environ.get(_WORKER_ENVIRONMENT) != "1":
    _direct_bootstrap()

import simsoptpp

_REMOVED_EDITABLE_FINDERS: Final = _neutralize_editable_source_redirection()

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from examples.jax.parity.input_bundle import read_input_bundle
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NEQ_GNTR3_OPTIONS,
    NEQ_GNTR3_ROUTE,
    NEQ_GNTR3_SCHEMA_VERSION,
    NativeEquivalentQualityPolicy,
    PreparedNeqGntr3,
    build_native_equivalent_terminal_diagnostic,
    fullspace_scaling_from_bootstrap,
    prepare_neq_accepted_quality_diagnostics,
    prepare_neq_gntr3,
    prepare_neq_terminal_endpoint_diagnostics,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    build_native_single_stage_endpoint_runtime,
)

from benchmarks.single_stage_fullspace_snapshot import (
    DIAG5_CPU_SNAPSHOT_ROLES,
    ArtifactRef,
    JsonValue,
    SnapshotPublication,
    SourceRoot,
    canonical_json_bytes,
    capture_worktree_identity,
    load_canonical_json_bytes,
    load_snapshot,
    publish_immutable_snapshot,
    validate_sealed_native_extension,
)
from benchmarks.single_stage_native_equivalent_endpoint_audit import (
    endpoint_audit_bytes,
    endpoint_audit_payload,
    produce_native_equivalent_endpoint_audit,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    ARRAY_SPECS,
    DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS,
    DIAG4_ENDPOINT_OBSERVABLE_FIELDS,
    FINAL_CERTIFICATE_FIELDS,
    KktStatus,
    NativeEquivalentNumericalIdentity,
    ScientificOutcome,
    array_evidence_payload,
    diag4_terminal_numerical_payload,
    history_evidence_from_arrays,
    policy_evidence_payload,
    safeguard_telemetry_payload,
    terminal_numerical_payload,
    validate_native_equivalent_scientific_evidence,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG5_BLANK_PLAN_SHA256,
    DIAG5_BLANK_PLAN_SIZE_BYTES,
    DIAG5_FROZEN_NUMERICAL_PATHS,
    DIAG5_NATIVE_COPY_RELATIVE_PATH,
    DIAG5_PLAN_RELATIVE_PATH,
    DIAG5_PLAN_SHA256,
    DIAG5_QUALIFIED_FILE_PATHS,
    Diag5PredecessorFailureEvidence,
    validate_diag5_predecessor_failure,
    validate_diag5_predecessor_postmortem_artifact,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG5_PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH as PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH as PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION as PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG5_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION as PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION,
)
from benchmarks.single_stage_native_equivalent_reference import (
    REFERENCE_FILENAME,
    validate_native_equivalent_reference,
)

if _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH != DIAG5_PLAN_RELATIVE_PATH:
    raise QualificationError(
        "prequalification plan source path differs from authority"
    )
from benchmarks.single_stage_native_equivalent_reference import (
    load_canonical_json_bytes as load_reference_json_bytes,
)

QUALIFICATION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-v2"
)
MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-manifest-v2"
)
EXPECTED_OUTPUT_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag5-cpu-qualification-20260812T090000Z"
)
RETAINED_DIAG3_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr1-diag3-cb0-20260811T150010Z.partial-"
    "56a1ec6d730cc005db84f99e9965b868"
)
NATIVE_REFERENCE_ROOT: Final = RETAINED_DIAG3_ROOT / "native-reference"
INPUT_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    ".single-stage-speed-20260804.partial-20260805T052535Z-2add24ec/inputs"
)
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
WORKTREE_ROOT: Final = Path(os.environ.get(_WORKTREE_ROOT_ENVIRONMENT, REPOSITORY_ROOT))
SOURCE_SNAPSHOT_DIRECTORY: Final = "source-snapshot"
QUALIFICATION_FILENAME: Final = "scientific-evidence.json"
MANIFEST_FILENAME: Final = "artifact-manifest.json"
HISTORY_FILENAME: Final = "history.json"
POLICY_FILENAME: Final = "policy.json"
TERMINAL_FILENAME: Final = "terminal-numerical.json"
ENDPOINT_AUDIT_FILENAME: Final = "endpoint-audit.json"
SAFEGUARD_TELEMETRY_FILENAME: Final = "safeguard-telemetry.json"
SPEED_NOT_PRODUCED: Final = "NOT_PRODUCED"
_CALLBACK_TOKENS: Final = (
    "debug_callback",
    "host_callback",
    "io_callback",
    "xla_python_cpu_callback",
)
_IMPORTED_SOURCE_BINDINGS: Final = {
    "benchmarks.single_stage_fullspace_snapshot": (
        "benchmarks/single_stage_fullspace_snapshot.py"
    ),
    "benchmarks.single_stage_native_equivalent_endpoint_audit": (
        "benchmarks/single_stage_native_equivalent_endpoint_audit.py"
    ),
    "benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt": (
        "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py"
    ),
    "benchmarks.single_stage_native_equivalent_quality_successor_authority": (
        "benchmarks/single_stage_native_equivalent_quality_successor_authority.py"
    ),
    "benchmarks.single_stage_native_equivalent_reference": (
        "benchmarks/single_stage_native_equivalent_reference.py"
    ),
    "examples.jax.parity.input_bundle": "examples/jax/parity/input_bundle.py",
    "simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region": (
        "src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py"
    ),
    "simsopt_jax.objectives.single_stage_fullspace": (
        "src/simsopt_jax/objectives/single_stage_fullspace.py"
    ),
    "simsopt_jax.runtime.trace_annotations": (
        "src/simsopt_jax/runtime/trace_annotations.py"
    ),
    "simsopt_jax.solve.fullspace": "src/simsopt_jax/solve/fullspace.py",
    "simsopt_jax.solve.fullspace_gauss_newton_trust_region": (
        "src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py"
    ),
    "simsopt_jax.solve.fullspace_native_equivalent_quality": (
        "src/simsopt_jax/solve/fullspace_native_equivalent_quality.py"
    ),
    "simsopt_jax_adapters.geo.single_stage_fullspace": (
        "src/simsopt_jax_adapters/geo/single_stage_fullspace.py"
    ),
    "simsopt_jax_adapters.geo.single_stage_native_endpoint": (
        "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py"
    ),
}


@dataclass(frozen=True, slots=True)
class CpuRuntimeIdentity:
    """Observed CPU runtime identity established before material execution."""

    backend: str
    x64_enabled: bool
    devices: tuple[tuple[int, str, str], ...]
    environment: tuple[tuple[str, str | None], ...]
    argv: tuple[str, ...]
    cwd: str
    python_executable: str
    python_executable_sha256: str
    python_version: str
    platform: str
    jax_version: str
    jaxlib_version: str
    native_extension_path: str
    native_extension_sha256: str
    native_extension_size_bytes: int
    native_extension_link_count: int

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "argv": list(self.argv),
            "backend": self.backend,
            "cwd": self.cwd,
            "devices": [
                {"id": identifier, "kind": kind, "platform": device_platform}
                for identifier, device_platform, kind in self.devices
            ],
            "environment": dict(self.environment),
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "native_extension_path": self.native_extension_path,
            "native_extension_sha256": self.native_extension_sha256,
            "native_extension_size_bytes": self.native_extension_size_bytes,
            "native_extension_link_count": self.native_extension_link_count,
            "platform": self.platform,
            "python_executable": self.python_executable,
            "python_executable_sha256": self.python_executable_sha256,
            "python_version": self.python_version,
            "x64_enabled": self.x64_enabled,
        }


@dataclass(frozen=True, slots=True)
class ImportedSourceBinding:
    relative_path: str
    path: Path
    descriptor: int
    device: int
    inode: int
    size_bytes: int
    sha256: str
    link_count: int
    directory_descriptors: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ImportedSourceBindings:
    """Retained imported-source descriptors held through final deep validation."""

    entries: tuple[ImportedSourceBinding, ...]

    def validate(self) -> None:
        for entry in self.entries:
            locked = os.fstat(entry.descriptor)
            native = entry.relative_path.startswith("native/")
            bound = (
                _validate_absolute_regular_chain(
                    entry.path,
                    entry.descriptor,
                    entry.directory_descriptors,
                )
                if native
                else locked
            )
            expected_link_count = entry.link_count if native else 1
            if (
                not stat.S_ISREG(locked.st_mode)
                or entry.link_count != expected_link_count
                or locked.st_nlink != expected_link_count
                or bound.st_nlink != expected_link_count
                or (locked.st_dev, locked.st_ino, locked.st_size)
                != (entry.device, entry.inode, entry.size_bytes)
                or (bound.st_dev, bound.st_ino, bound.st_size)
                != (entry.device, entry.inode, entry.size_bytes)
                or _sha256_descriptor(entry.descriptor) != entry.sha256
            ):
                raise QualificationError(
                    f"imported source binding changed: {entry.relative_path}"
                )

    def close(self) -> None:
        for entry in self.entries:
            _unlock_close_descriptor(entry.descriptor)
            for directory_descriptor in entry.directory_descriptors:
                os.close(directory_descriptor)


@dataclass(frozen=True, slots=True)
class ProducedEvidence:
    """Complete scientific evidence and diagnostic timing before publication."""

    scientific_outcome: ScientificOutcome
    numerical_identity: NativeEquivalentNumericalIdentity
    timings_ns: tuple[tuple[str, int], ...]
    callback_count: int
    execution_source_manifest_sha256: str
    execution_source_entries_sha256: str
    prequalification_plan_control: tuple[tuple[str, JsonValue], ...]
    source_manifest_sha256: str
    source_manifest_entries: tuple[tuple[str, str, int, str], ...]
    native_extension_path: str
    native_extension_sha256: str
    native_extension_size_bytes: int
    native_extension_link_count: int
    native_extension_device: int
    native_extension_inode: int
    predecessor_postmortem: ArtifactRef
    native_reference_artifact_sha256: str
    input_fingerprint: str
    configuration_fingerprint: str
    policy_sha256: str
    imported_source_bindings: ImportedSourceBindings | None = None
    execution_source_bindings: ExecutionSourceBindings | None = None
    retained_input_trees: tuple[RetainedRegularTree, ...] = ()


class CpuQualificationProducer(Protocol):
    """Materialize and deep-validate one complete CPU scientific evidence set."""

    def produce(
        self,
        staging_root: Path,
        runtime_identity: CpuRuntimeIdentity,
    ) -> ProducedEvidence: ...

    def validate(
        self,
        artifact_root: Path,
        qualification: Mapping[str, JsonValue],
    ) -> ScientificOutcome: ...


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_descriptor(descriptor: int) -> str:
    return _descriptor_sha256(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise QualificationError(f"noncanonical artifact path: {value!r}")
    if path.as_posix() != value:
        raise QualificationError(f"noncanonical artifact path: {value!r}")
    return path


def _inherited_execution_source_bindings_unchecked(
    worktree_root: Path,
    execution_root: Path,
) -> ExecutionSourceBindings:
    raw_payload = os.environ.get(_EXECUTION_SOURCE_DESCRIPTOR_ENVIRONMENT)
    if raw_payload is None:
        raise QualificationError("execution-source descriptor authority is absent")
    payload = _bootstrap_json(raw_payload.encode("utf-8"))
    if not isinstance(payload, dict) or frozenset(payload) != frozenset(
        {
            "authority_sha256",
            "entries",
            "entries_sha256",
            "execution_descriptor",
            "manifest_copied_descriptor",
            "manifest_live_descriptor",
            "plan_descriptor",
            "worktree_descriptor",
        }
    ):
        raise QualificationError("execution-source descriptor fields differ")
    raw_entries = payload["entries"]
    manifest_live_descriptor = payload["manifest_live_descriptor"]
    manifest_copied_descriptor = payload["manifest_copied_descriptor"]
    plan_descriptor = payload["plan_descriptor"]
    worktree_descriptor = payload["worktree_descriptor"]
    execution_descriptor = payload["execution_descriptor"]
    if (
        not isinstance(raw_entries, dict)
        or type(manifest_live_descriptor) is not int
        or type(manifest_copied_descriptor) is not int
        or type(plan_descriptor) is not int
        or type(worktree_descriptor) is not int
        or type(execution_descriptor) is not int
    ):
        raise QualificationError("execution-source descriptor types differ")
    copied_manifest_stat = os.fstat(manifest_copied_descriptor)
    copied_manifest_payload = _descriptor_bytes(
        manifest_copied_descriptor,
        copied_manifest_stat.st_size,
    )
    entries, entries_sha256 = _parse_execution_source_authority(copied_manifest_payload)
    authority_sha256 = hashlib.sha256(copied_manifest_payload).hexdigest()
    if (
        payload["authority_sha256"] != authority_sha256
        or payload["entries_sha256"] != entries_sha256
        or frozenset(raw_entries) != frozenset(entry.relative_path for entry in entries)
    ):
        raise QualificationError("execution-source descriptor identity differs")
    worktree = worktree_root.resolve(strict=True)
    execution = execution_root.resolve(strict=True)
    worktree_stat = os.fstat(worktree_descriptor)
    execution_stat = os.fstat(execution_descriptor)
    bindings: list[ExecutionSourceFileBinding] = []
    for entry in entries:
        raw_descriptors = raw_entries[entry.relative_path]
        if not isinstance(raw_descriptors, dict) or frozenset(
            raw_descriptors
        ) != frozenset({"copied_descriptor", "live_descriptor"}):
            raise QualificationError(
                f"execution-source descriptor entry differs: {entry.relative_path}"
            )
        live_descriptor = raw_descriptors["live_descriptor"]
        copied_descriptor = raw_descriptors["copied_descriptor"]
        if type(live_descriptor) is not int or type(copied_descriptor) is not int:
            raise QualificationError(
                f"execution-source descriptor entry types differ: {entry.relative_path}"
            )
        live_stat = os.fstat(live_descriptor)
        copied_stat = os.fstat(copied_descriptor)
        os.set_inheritable(live_descriptor, False)
        os.set_inheritable(copied_descriptor, False)
        relative = PurePosixPath(entry.relative_path)
        bindings.append(
            ExecutionSourceFileBinding(
                entry=entry,
                live_path=worktree.joinpath(*relative.parts),
                live_descriptor=live_descriptor,
                live_device=live_stat.st_dev,
                live_inode=live_stat.st_ino,
                copied_path=execution.joinpath(*relative.parts),
                copied_descriptor=copied_descriptor,
                copied_device=copied_stat.st_dev,
                copied_inode=copied_stat.st_ino,
            )
        )
    manifest_live_stat = os.fstat(manifest_live_descriptor)
    plan_stat = os.fstat(plan_descriptor)
    plan_payload = _descriptor_bytes(plan_descriptor, plan_stat.st_size)
    plan_sha256 = hashlib.sha256(plan_payload).hexdigest()
    authority_binding = next(
        (
            binding
            for binding in bindings
            if binding.entry.relative_path
            == "benchmarks/single_stage_native_equivalent_quality_successor_authority.py"
        ),
        None,
    )
    if authority_binding is None:
        raise QualificationError("copied authority source is absent")
    authority_tree = ast.parse(
        _descriptor_bytes(
            authority_binding.copied_descriptor,
            authority_binding.entry.size_bytes,
        ),
        filename=str(authority_binding.copied_path),
    )
    expected_plan_prefix = _bootstrap_constant(authority_tree, "DIAG5_PLAN_SHA256")
    plan_prefix, marker, record = plan_payload.partition(b"## Qualification Record\n")
    if (
        not isinstance(expected_plan_prefix, str)
        or not marker
        or record
        or hashlib.sha256(plan_prefix).hexdigest() != expected_plan_prefix
    ):
        raise QualificationError("inherited prequalification plan differs")
    os.set_inheritable(manifest_live_descriptor, False)
    os.set_inheritable(manifest_copied_descriptor, False)
    os.set_inheritable(plan_descriptor, False)
    os.set_inheritable(worktree_descriptor, False)
    os.set_inheritable(execution_descriptor, False)
    manifest_relative = PurePosixPath(EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH)
    inherited = ExecutionSourceBindings(
        worktree_root=worktree,
        worktree_descriptor=worktree_descriptor,
        worktree_device=worktree_stat.st_dev,
        worktree_inode=worktree_stat.st_ino,
        execution_root=execution,
        execution_descriptor=execution_descriptor,
        execution_device=execution_stat.st_dev,
        execution_inode=execution_stat.st_ino,
        authority_sha256=authority_sha256,
        entries_sha256=entries_sha256,
        manifest_live_path=worktree.joinpath(*manifest_relative.parts),
        manifest_live_descriptor=manifest_live_descriptor,
        manifest_live_device=manifest_live_stat.st_dev,
        manifest_live_inode=manifest_live_stat.st_ino,
        manifest_size_bytes=len(copied_manifest_payload),
        manifest_copied_path=execution.joinpath(*manifest_relative.parts),
        manifest_copied_descriptor=manifest_copied_descriptor,
        manifest_copied_device=copied_manifest_stat.st_dev,
        manifest_copied_inode=copied_manifest_stat.st_ino,
        plan_source_path=(worktree / _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH),
        plan_descriptor=plan_descriptor,
        plan_device=plan_stat.st_dev,
        plan_inode=plan_stat.st_ino,
        plan_size_bytes=plan_stat.st_size,
        plan_sha256=plan_sha256,
        plan_prefix_sha256=expected_plan_prefix,
        entries=tuple(bindings),
    )
    inherited.validate(copied_required=True)
    return inherited


def _inherited_descriptor_numbers(raw_payload: str) -> frozenset[int]:
    payload = _bootstrap_json(raw_payload.encode("utf-8"))
    if not isinstance(payload, dict):
        return frozenset()
    values: set[int] = set()
    for name in (
        "execution_descriptor",
        "manifest_copied_descriptor",
        "manifest_live_descriptor",
        "plan_descriptor",
        "worktree_descriptor",
    ):
        value = payload.get(name)
        if type(value) is int:
            values.add(value)
    raw_entries = payload.get("entries")
    if isinstance(raw_entries, dict):
        for raw_entry in raw_entries.values():
            if not isinstance(raw_entry, dict):
                continue
            for name in ("copied_descriptor", "live_descriptor"):
                value = raw_entry.get(name)
                if type(value) is int:
                    values.add(value)
    return frozenset(values)


def _inherited_execution_source_bindings(
    worktree_root: Path,
    execution_root: Path,
) -> ExecutionSourceBindings:
    raw_payload = os.environ.get(_EXECUTION_SOURCE_DESCRIPTOR_ENVIRONMENT)
    if raw_payload is None:
        raise QualificationError("execution-source descriptor authority is absent")
    try:
        return _inherited_execution_source_bindings_unchecked(
            worktree_root,
            execution_root,
        )
    except BaseException:
        for descriptor in _inherited_descriptor_numbers(raw_payload):
            try:
                _unlock_close_descriptor(descriptor)
            except OSError:
                pass
        raise


def observe_cpu_runtime(
    environment: Mapping[str, str],
) -> CpuRuntimeIdentity:
    """Prove explicit FP64 CPU dispatch before any output or material execution."""

    for name, expected in _REQUIRED_ENVIRONMENT.items():
        if environment.get(name) != expected:
            raise QualificationError(f"{name} must equal {expected!r}")
    backend = jax.default_backend()
    devices = tuple(jax.devices())
    x64_enabled = bool(jax.config.jax_enable_x64)
    if (
        backend != "cpu"
        or not devices
        or any(device.platform != "cpu" for device in devices)
    ):
        raise QualificationError("qualification requires exclusively CPU JAX devices")
    if not x64_enabled:
        raise QualificationError("qualification requires JAX FP64")
    executable = Path(sys.executable).resolve(strict=True)
    native_extension = Path(simsoptpp.__file__).resolve(strict=True)
    native_metadata, native_sha256 = _observe_absolute_regular(native_extension)
    if not stat.S_ISREG(native_metadata.st_mode) or native_metadata.st_nlink < 1:
        raise QualificationError("qualification native extension is not regular")
    return CpuRuntimeIdentity(
        backend=backend,
        x64_enabled=x64_enabled,
        devices=tuple(
            (int(device.id), str(device.platform), str(device.device_kind))
            for device in devices
        ),
        environment=tuple(
            (name, environment.get(name)) for name in sorted(_REQUIRED_ENVIRONMENT)
        ),
        argv=tuple(sys.argv),
        cwd=str(Path.cwd().resolve(strict=True)),
        python_executable=str(executable),
        python_executable_sha256=_sha256_file(executable),
        python_version=platform.python_version(),
        platform=platform.platform(),
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        native_extension_path=str(native_extension),
        native_extension_sha256=native_sha256,
        native_extension_size_bytes=native_metadata.st_size,
        native_extension_link_count=native_metadata.st_nlink,
    )


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o644,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _publish_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    _publish_bytes(path, canonical_json_bytes(dict(payload)))


def _npy_bytes(values: np.ndarray) -> bytes:
    from io import BytesIO

    stream = BytesIO()
    np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return stream.getvalue()


def _publish_npy(path: Path, values: np.ndarray) -> None:
    _publish_bytes(path, _npy_bytes(values))


def _copy_retained_regular_tree(
    source: RetainedRegularTree,
    destination: Path,
) -> None:
    """Copy one admitted tree only from its locked source descriptors."""

    source.validate()
    destination.mkdir(mode=0o755, exist_ok=False)
    copied_directories = [destination]
    for relative, _, _, _ in source.directories:
        output = destination.joinpath(*PurePosixPath(relative).parts)
        output.mkdir(mode=0o755, exist_ok=False)
        copied_directories.append(output)
    destination_descriptor = os.open(
        destination,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        for entry in source.files:
            _copy_descriptor(
                destination_descriptor,
                entry.relative_path,
                entry.descriptor,
                entry.size_bytes,
            )
    finally:
        os.close(destination_descriptor)
    for directory in sorted(
        copied_directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    source.validate()
    copied_descriptor, copied_parent_descriptors = _open_absolute_directory(
        destination.absolute()
    )
    try:
        if stat.S_IMODE(os.fstat(copied_descriptor).st_mode) != 0o555:
            raise QualificationError("copied input root mode differs")
        copied_directories_inventory, copied_files = _retained_tree_inventory(
            copied_descriptor
        )
        if tuple(
            (relative, mode) for relative, _, _, mode in copied_directories_inventory
        ) != tuple((relative, 0o555) for relative, _, _, _ in source.directories):
            raise QualificationError("copied input directory membership differs")
        if copied_files != tuple(entry.relative_path for entry in source.files):
            raise QualificationError("copied input file membership differs")
        for entry in source.files:
            copied = _open_relative_regular(
                copied_descriptor,
                entry.relative_path,
                flags=os.O_RDONLY,
            )
            try:
                observed = os.fstat(copied)
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or observed.st_nlink != 1
                    or stat.S_IMODE(observed.st_mode) != 0o444
                    or observed.st_size != entry.size_bytes
                    or _descriptor_sha256(copied) != entry.sha256
                ):
                    raise QualificationError("copied input leaf differs")
            finally:
                os.close(copied)
    finally:
        os.close(copied_descriptor)
        for descriptor in copied_parent_descriptors:
            os.close(descriptor)


def _artifact_ref(
    path: Path,
    root: Path,
    schema_version: str,
) -> ArtifactRef:
    payload = path.read_bytes()
    return ArtifactRef(
        path.relative_to(root).as_posix(),
        _sha256(payload),
        len(payload),
        schema_version,
    )


def _artifact_ref_payload(reference: ArtifactRef) -> dict[str, JsonValue]:
    return {
        "relative_path": reference.relative_path,
        "schema_version": reference.schema_version,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
    }


def _artifact_ref_from_payload(value: JsonValue) -> ArtifactRef:
    if not isinstance(value, dict) or frozenset(value) != frozenset(
        {"relative_path", "schema_version", "sha256", "size_bytes"}
    ):
        raise QualificationError("artifact reference differs from schema")
    relative_path = value["relative_path"]
    schema_version = value["schema_version"]
    sha256 = value["sha256"]
    size_bytes = value["size_bytes"]
    if (
        not isinstance(relative_path, str)
        or not isinstance(schema_version, str)
        or not isinstance(sha256, str)
        or type(size_bytes) is not int
    ):
        raise QualificationError("artifact reference has invalid types")
    _safe_relative_path(relative_path)
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise QualificationError("artifact reference SHA-256 is invalid")
    return ArtifactRef(relative_path, sha256, size_bytes, schema_version)


def _load_json_artifact(root: Path, reference: ArtifactRef) -> JsonValue:
    path = root.joinpath(*_safe_relative_path(reference.relative_path).parts)
    _reject_symlink_components(path)
    payload = path.read_bytes()
    if len(payload) != reference.size_bytes or _sha256(payload) != reference.sha256:
        raise QualificationError("artifact reference bytes differ")
    return load_canonical_json_bytes(payload)


def _execution_source_role(relative: str) -> str:
    if relative == PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH:
        return "execution_source"
    if relative.startswith("docs/"):
        return "configuration"
    if relative.startswith("tests/"):
        return "test"
    if relative.startswith("benchmarks/"):
        return "benchmark"
    return "execution_source"


def _prequalification_plan_control(
    execution_sources: ExecutionSourceBindings,
) -> dict[str, JsonValue]:
    return {
        "plan_prefix_sha256": execution_sources.plan_prefix_sha256,
        "schema_version": PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION,
        "sha256": execution_sources.plan_sha256,
        "size_bytes": execution_sources.plan_size_bytes,
        "snapshot_relative_path": PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH,
        "source_relative_path": _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH,
    }


def _publish_predecessor_postmortem(
    execution_sources: ExecutionSourceBindings,
    staging_root: Path,
) -> ArtifactRef:
    binding = next(
        (
            candidate
            for candidate in execution_sources.entries
            if candidate.entry.relative_path
            == PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH
        ),
        None,
    )
    if binding is None:
        raise QualificationError(
            "predecessor postmortem is absent from source authority"
        )
    payload = _descriptor_bytes(binding.live_descriptor, binding.entry.size_bytes)
    _validate_predecessor_postmortem(execution_sources.worktree_root, payload)
    destination = staging_root.joinpath(
        *_safe_relative_path(PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH).parts
    )
    _publish_bytes(destination, payload)
    reference = _artifact_ref(
        destination, staging_root, PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
    )
    validate_diag5_predecessor_postmortem_artifact(staging_root, reference)
    return reference


def _validate_predecessor_postmortem(repository: Path, payload: bytes) -> None:
    if not payload:
        raise QualificationError("predecessor postmortem source bytes are absent")
    document = load_canonical_json_bytes(payload)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
    ):
        raise QualificationError("predecessor postmortem schema differs")
    reconstruction = document.get("reconstruction")
    if not isinstance(reconstruction, dict):
        raise QualificationError("predecessor postmortem reconstruction differs")
    postmortem_path = repository / PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH
    validate_diag5_predecessor_failure(
        Diag5PredecessorFailureEvidence(
            partial_root=Path(str(reconstruction.get("partial_root"))),
            failed_stage=str(reconstruction.get("failed_stage")),
            exception_class=str(reconstruction.get("exception_class")),
            exception_message=str(reconstruction.get("exception_message")),
            qualifier_sha256=str(reconstruction.get("qualifier_sha256")),
            execution_manifest_sha256=str(
                reconstruction.get("execution_manifest_sha256")
            ),
            execution_entries_sha256=str(
                reconstruction.get("execution_entries_sha256")
            ),
            execution_source_entry_count=reconstruction.get(
                "execution_source_entry_count"
            ),
            copied_tree_entry_count=reconstruction.get("copied_tree_entry_count"),
            predecessor_full_tree_sha256=str(
                reconstruction.get("predecessor_full_tree_sha256")
            ),
            postmortem_path=postmortem_path,
            postmortem_sha256=_sha256(payload),
        ),
        repository_root=repository,
    )


def _validate_public_source_membership(
    execution_sources: ExecutionSourceBindings,
) -> None:
    paths = {binding.entry.relative_path for binding in execution_sources.entries}
    required = (set(DIAG5_QUALIFIED_FILE_PATHS) | set(DIAG5_FROZEN_NUMERICAL_PATHS)) - {
        EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH
    }
    if (
        DIAG5_PLAN_RELATIVE_PATH != _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH
        or DIAG5_PLAN_RELATIVE_PATH in paths
        or not required.issubset(paths)
    ):
        raise QualificationError("public execution-source membership differs")


def _source_roots(
    repository: Path,
    execution_sources: ExecutionSourceBindings,
) -> tuple[SourceRoot, ...]:
    roots: list[SourceRoot] = []
    for binding in execution_sources.entries:
        relative = binding.entry.relative_path
        roots.append(
            SourceRoot(
                _execution_source_role(relative),
                repository.joinpath(*PurePosixPath(relative).parts),
                relative,
            )
        )
    roots.append(
        SourceRoot(
            "execution_source_manifest",
            repository.joinpath(
                *PurePosixPath(EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH).parts
            ),
            EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
        )
    )
    roots.append(
        SourceRoot(
            "prequalification_plan",
            execution_sources.plan_source_path,
            PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH,
        )
    )
    native_extension = Path(simsoptpp.__file__).resolve(strict=True)
    roots.append(
        SourceRoot(
            "native_extension",
            native_extension,
            DIAG5_NATIVE_COPY_RELATIVE_PATH,
        )
    )
    return tuple(roots)


def _validate_execution_source_snapshot(
    snapshot: SnapshotPublication,
    execution_sources: ExecutionSourceBindings,
    native_binding: ImportedSourceBinding,
) -> None:
    expected = {
        binding.entry.relative_path: (
            binding.entry.size_bytes,
            binding.entry.sha256,
        )
        for binding in execution_sources.entries
    }
    expected[EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH] = (
        execution_sources.manifest_size_bytes,
        execution_sources.authority_sha256,
    )
    expected[PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH] = (
        execution_sources.plan_size_bytes,
        execution_sources.plan_sha256,
    )
    observed = {
        entry.relative_path: (entry.size_bytes, entry.sha256)
        for entry in snapshot.entries
        if entry.role != "native_extension"
    }
    if observed != expected:
        raise QualificationError("execution-source snapshot differs from authority")
    expected_roles = {
        binding.entry.relative_path: _execution_source_role(binding.entry.relative_path)
        for binding in execution_sources.entries
    }
    expected_roles[EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH] = (
        "execution_source_manifest"
    )
    expected_roles[PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH] = (
        "prequalification_plan"
    )
    observed_roles = {
        entry.relative_path: entry.role
        for entry in snapshot.entries
        if entry.role != "native_extension"
    }
    if observed_roles != expected_roles:
        raise QualificationError("execution-source snapshot roles differ")
    native_entry = validate_sealed_native_extension(
        snapshot,
        expected_sha256=native_binding.sha256,
        expected_size_bytes=native_binding.size_bytes,
        expected_relative_path=native_binding.relative_path,
    )
    if native_entry.relative_path != native_binding.relative_path:
        raise QualificationError("execution-source snapshot native entry differs")
    manifest_path = snapshot.root.joinpath(
        *PurePosixPath(EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH).parts
    )
    manifest_payload = manifest_path.read_bytes()
    parsed_entries, entries_sha256 = _parse_execution_source_authority(manifest_payload)
    if (
        hashlib.sha256(manifest_payload).hexdigest()
        != execution_sources.authority_sha256
        or entries_sha256 != execution_sources.entries_sha256
        or tuple(entry.entry for entry in execution_sources.entries) != parsed_entries
    ):
        raise QualificationError("snapshotted execution-source authority differs")


def _validate_snapshot_qualification_identity(
    snapshot: SnapshotPublication,
    qualification: Mapping[str, JsonValue],
) -> None:
    manifest_path = snapshot.root.joinpath(
        *PurePosixPath(EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH).parts
    )
    manifest_payload = manifest_path.read_bytes()
    entries, entries_sha256 = _parse_execution_source_authority(manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    plan_control = qualification.get("prequalification_plan_control")
    if not isinstance(plan_control, dict):
        raise QualificationError("prequalification plan control is absent")
    expected = {
        entry.relative_path: (
            _execution_source_role(entry.relative_path),
            entry.size_bytes,
            entry.sha256,
        )
        for entry in entries
    }
    expected[EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH] = (
        "execution_source_manifest",
        len(manifest_payload),
        manifest_sha256,
    )
    expected[PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH] = (
        "prequalification_plan",
        plan_control.get("size_bytes"),
        plan_control.get("sha256"),
    )
    observed = {
        entry.relative_path: (entry.role, entry.size_bytes, entry.sha256)
        for entry in snapshot.entries
        if entry.role != "native_extension"
    }
    native_entries = tuple(
        entry for entry in snapshot.entries if entry.role == "native_extension"
    )
    native_binding = qualification.get("cpu_native_binding")
    if not isinstance(native_binding, dict):
        raise QualificationError("qualification CPU native binding differs")
    native_path = native_binding.get("cpu_native_extension_path")
    native_sha256 = native_binding.get("native_extension_sha256")
    native_size = native_binding.get("native_extension_size_bytes")
    if isinstance(native_path, str):
        observed_native, observed_native_sha256 = _observe_absolute_regular(
            Path(native_path)
        )
    else:
        observed_native = None
        observed_native_sha256 = None
    snapshotted_plan = snapshot.root.joinpath(
        *PurePosixPath(PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH).parts
    ).read_bytes()
    plan_prefix, marker, record = snapshotted_plan.partition(
        b"## Qualification Record\n"
    )
    if (
        qualification.get("execution_source_manifest_sha256") != manifest_sha256
        or qualification.get("execution_source_entries_sha256") != entries_sha256
        or observed != expected
        or not marker
        or record
        or hashlib.sha256(plan_prefix).hexdigest()
        != plan_control.get("plan_prefix_sha256")
        or len(native_entries) != 1
        or not isinstance(native_path, str)
        or native_entries[0].relative_path != DIAG5_NATIVE_COPY_RELATIVE_PATH
        or native_entries[0].sha256 != native_sha256
        or native_entries[0].size_bytes != native_size
        or observed_native is None
        or observed_native.st_size != native_size
        or observed_native_sha256 != native_sha256
        or Path(native_path).resolve(strict=True)
        != Path(simsoptpp.__file__).resolve(strict=True)
    ):
        raise QualificationError("source snapshot qualification identity differs")


def _validate_imported_source_bindings(
    repository: Path,
    snapshot: SnapshotPublication,
) -> None:
    """Join every production import to the immutable executed source snapshot."""

    entries = {entry.relative_path: entry for entry in snapshot.entries}
    qualifier_relative = (
        "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py"
    )
    expected_origins = {
        **_IMPORTED_SOURCE_BINDINGS,
        "__entrypoint__": qualifier_relative,
    }
    for module_name, relative in expected_origins.items():
        if module_name == "__entrypoint__":
            origin = Path(sys.argv[0])
            spec_origin = origin
        else:
            module = sys.modules.get(module_name)
            if module is None:
                raise QualificationError(
                    f"required production module is not imported: {module_name}"
                )
            raw_origin = getattr(module, "__file__", None)
            raw_spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if not isinstance(raw_origin, str) or not isinstance(raw_spec_origin, str):
                raise QualificationError(
                    f"production module origin is absent: {module_name}"
                )
            origin = Path(raw_origin)
            spec_origin = Path(raw_spec_origin)
        expected = (repository / relative).resolve(strict=True)
        if (
            origin.resolve(strict=True) != expected
            or spec_origin.resolve(strict=True) != expected
        ):
            raise QualificationError(
                f"production module resolved outside execution source: {module_name}"
            )
        observed = expected.stat(follow_symlinks=False)
        entry = entries.get(relative)
        if (
            expected.is_symlink()
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or entry is None
            or entry.size_bytes != observed.st_size
            or entry.sha256 != _sha256_file(expected)
        ):
            raise QualificationError(
                f"production module bytes differ from source snapshot: {module_name}"
            )
    native_entry = next(
        (entry for entry in snapshot.entries if entry.role == "native_extension"),
        None,
    )
    native_origin = Path(simsoptpp.__file__).resolve(strict=True)
    native_observed, native_sha256 = _observe_absolute_regular(native_origin)
    if (
        native_entry is None
        or native_entry.relative_path != DIAG5_NATIVE_COPY_RELATIVE_PATH
        or native_entry.size_bytes != native_observed.st_size
        or native_entry.sha256 != native_sha256
    ):
        raise QualificationError("native extension differs from source snapshot")


def _capture_imported_source_bindings(
    repository: Path,
    snapshot: SnapshotPublication,
    native_binding: ImportedSourceBinding,
) -> ImportedSourceBindings:
    _validate_imported_source_bindings(repository, snapshot)
    entries = {entry.relative_path: entry for entry in snapshot.entries}
    relative_paths = (
        "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
        *_IMPORTED_SOURCE_BINDINGS.values(),
    )
    opened: list[ImportedSourceBinding] = [native_binding]
    try:
        for relative in relative_paths:
            path = (repository / relative).resolve(strict=True)
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                _lock_shared_descriptor(descriptor, f"imported source {relative}")
            except BaseException:
                os.close(descriptor)
                raise
            observed = os.fstat(descriptor)
            entry = entries[relative]
            digest = _sha256_descriptor(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_size != entry.size_bytes
                or digest != entry.sha256
            ):
                _unlock_close_descriptor(descriptor)
                raise QualificationError(
                    f"imported source descriptor differs: {relative}"
                )
            opened.append(
                ImportedSourceBinding(
                    relative,
                    path,
                    descriptor,
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_size,
                    digest,
                    observed.st_nlink,
                )
            )
        native_entry = entries[native_binding.relative_path]
        native_binding_stat = os.fstat(native_binding.descriptor)
        if (
            native_binding.size_bytes != native_entry.size_bytes
            or native_binding.sha256 != native_entry.sha256
            or native_binding_stat.st_nlink != native_binding.link_count
        ):
            raise QualificationError("native extension descriptor differs")
        bindings = ImportedSourceBindings(tuple(opened))
        bindings.validate()
        return bindings
    except BaseException:
        ImportedSourceBindings(tuple(opened)).close()
        raise


def _capture_native_extension_binding(
    runtime_identity: CpuRuntimeIdentity,
) -> ImportedSourceBinding:
    native = Path(runtime_identity.native_extension_path)
    descriptor, directory_descriptors = _open_absolute_regular(native)
    try:
        observed = os.fstat(descriptor)
        digest = _sha256_descriptor(descriptor)
        bound = _validate_absolute_regular_chain(
            native,
            descriptor,
            directory_descriptors,
        )
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink < 1
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (bound.st_dev, bound.st_ino, bound.st_size)
            or observed.st_size != runtime_identity.native_extension_size_bytes
            or digest != runtime_identity.native_extension_sha256
            or observed.st_nlink != runtime_identity.native_extension_link_count
        ):
            raise QualificationError("native extension runtime binding differs")
        return ImportedSourceBinding(
            DIAG5_NATIVE_COPY_RELATIVE_PATH,
            native,
            descriptor,
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            digest,
            observed.st_nlink,
            directory_descriptors,
        )
    except BaseException:
        _unlock_close_descriptor(descriptor)
        for directory_descriptor in directory_descriptors:
            os.close(directory_descriptor)
        raise


def _reference_document(reference_root: Path) -> dict[str, JsonValue]:
    value = load_reference_json_bytes(
        (reference_root / REFERENCE_FILENAME).read_bytes()
    )
    if not isinstance(value, dict):
        raise QualificationError("native reference document must be an object")
    return value


def _reference_array(reference_root: Path, name: str) -> np.ndarray:
    document = _reference_document(reference_root)
    evidence = document.get("evidence")
    if not isinstance(evidence, dict):
        raise QualificationError("native reference evidence is absent")
    arrays = evidence.get("arrays")
    row = arrays.get(name) if isinstance(arrays, dict) else None
    if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
        raise QualificationError(f"native reference array {name!r} is absent")
    relative = _safe_relative_path(row["relative_path"])
    path = reference_root.joinpath(*relative.parts).resolve(strict=True)
    if not path.is_relative_to(reference_root.resolve(strict=True)):
        raise QualificationError("native reference array escapes its root")
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
    expected_shape = (255,) if name == "raw_equalities" else (716,)
    if values.dtype != np.dtype(np.float64) or values.shape != expected_shape:
        raise QualificationError(f"native reference array {name!r} differs")
    return np.ascontiguousarray(values, dtype=np.float64)


def _reference_fingerprints(reference_root: Path) -> tuple[str, str]:
    document = _reference_document(reference_root)
    authority_ref = document.get("authority_manifest")
    if not isinstance(authority_ref, dict) or not isinstance(
        authority_ref.get("relative_path"), str
    ):
        raise QualificationError("native reference authority is absent")
    relative = _safe_relative_path(authority_ref["relative_path"])
    authority = load_reference_json_bytes(
        reference_root.joinpath(*relative.parts).read_bytes()
    )
    if not isinstance(authority, dict):
        raise QualificationError("native reference authority is invalid")
    input_fingerprint = authority.get("input_fingerprint")
    configuration_fingerprint = authority.get("configuration_fingerprint")
    if not isinstance(input_fingerprint, str) or not isinstance(
        configuration_fingerprint, str
    ):
        raise QualificationError("native reference fingerprints are absent")
    return input_fingerprint, configuration_fingerprint


def _callback_count(
    prepared: PreparedNeqGntr3, accepted_quality: object, terminal: object
) -> int:
    executables = (
        prepared._run_loop,
        prepared._finalize,
        prepared._map_ledger,
        accepted_quality._run_quality,
        terminal._run_endpoint,
    )
    total = 0
    for executable in executables:
        as_text = getattr(executable, "as_text", None)
        if not callable(as_text):
            raise QualificationError("compiled executable has no inspectable text")
        executable_text = str(as_text()).lower()
        total += sum(executable_text.count(token) for token in _CALLBACK_TOKENS)
    return total


def _numerical_identity(
    prepared: PreparedNeqGntr3,
) -> NativeEquivalentNumericalIdentity:
    identity = prepared.identity
    if (
        identity.schema_version != NEQ_GNTR3_SCHEMA_VERSION
        or identity.route != NEQ_GNTR3_ROUTE
        or prepared.options != NEQ_GNTR3_OPTIONS
        or prepared.options.maximum_accepted_steps != 256
        or prepared.options.maximum_attempts != 300
        or prepared.options.maximum_nonlinear_corrections != 2
        or prepared.options.enable_step_bound_safeguard is not True
    ):
        raise QualificationError("prepared route differs from exact production GNTR3")
    return NativeEquivalentNumericalIdentity(
        identity.route,
        identity.schema_version,
        identity.problem_sha256,
        identity.optimizer_options_sha256,
        identity.base_neq_gntr1_policy_sha256,
        identity.scaling_sha256,
        identity.bootstrap_state_sha256,
        identity.initial_physical_state_sha256,
        identity.identity_sha256,
    )


def _host_scalar(value: object) -> int | float | bool:
    return np.asarray(value).item()


def _terminal_observables(endpoint: object) -> dict[str, float]:
    observables = endpoint.evaluation.observables
    return {
        "iota": float(np.asarray(observables.iota)),
        "G": float(np.asarray(observables.G)),
        "volume": float(np.asarray(observables.volume)),
        "major_radius": float(np.asarray(observables.major_radius)),
        "total_length": float(np.asarray(observables.total_length)),
        "non_qs_ratio": float(np.asarray(observables.non_qs_ratio)),
        "boozer_residual_value": float(np.asarray(observables.boozer_residual_scalar)),
        "boozer_residual_rms": float(np.asarray(observables.boozer_residual_rms)),
    }


def _publish_terminal(
    root: Path,
    diagnostic: object,
    quality_replay: object,
    terminal_evidence: object,
    prepared: PreparedNeqGntr3,
    numerical_identity: NativeEquivalentNumericalIdentity,
    endpoint_terms: Mapping[str, float],
    endpoint_observables: Mapping[str, float],
    terminal_seconds: float,
) -> tuple[dict[str, JsonValue], ArtifactRef]:
    base = diagnostic.base_result
    endpoint = base.endpoint
    raw = terminal_evidence.raw_endpoint
    transpose = endpoint.transpose_certificate
    arrays: dict[str, object] = {
        "optimizer_coordinates": base.optimizer_result.optimizer_coordinates,
        "physical_state": endpoint.physical_state,
        "raw_equalities": endpoint.raw_equalities,
        "scaled_equalities": endpoint.scaled_equalities,
        "objective_gradient": endpoint.objective_gradient,
        "multipliers": base.optimizer_result.multipliers,
        "raw_stationarity": raw.raw_stationarity_residual,
        "native_equalities": prepared.policy.native_raw_equalities,
        "constraint_inverse_scale": prepared.policy.constraint_inverse_scale,
        "accepted_optimizer_ledger": base.loop_result.accepted_optimizer_coordinates,
        "accepted_physical_ledger": base.accepted_physical_coordinates,
        "accepted_mask": base.accepted_state_mask,
        "accepted_quality_objectives": quality_replay.objectives,
        "accepted_quality_raw_equalities": quality_replay.raw_equalities,
        "accepted_quality_scaled_equalities": quality_replay.scaled_equalities,
        "accepted_quality_mask": quality_replay.accepted_state_mask,
        "accepted_quality_coordinates_finite": quality_replay.coordinates_finite,
        "accepted_quality_objective_finite": quality_replay.objective_finite,
        "accepted_quality_raw_equalities_finite": quality_replay.raw_equalities_finite,
        "accepted_quality_scaled_equalities_finite": quality_replay.scaled_equalities_finite,
        "accepted_quality_objective_satisfied": quality_replay.objective_satisfied,
        "accepted_quality_component_bounds_satisfied": quality_replay.component_bounds_satisfied,
        "accepted_quality_scaled_feasibility_satisfied": quality_replay.scaled_feasibility_satisfied,
        "accepted_quality_satisfied": quality_replay.quality_satisfied,
        "authoritative_objective_gradient": terminal_evidence.authoritative_objective_gradient,
        "bootstrap_anchor": prepared.scaling.bootstrap_anchor,
        "constraint_jacobian": base.optimizer_result.constraint_jacobian,
        "objective_residual_vector": terminal_evidence.objective_residual_vector,
        "reconstructed_objective_gradient": terminal_evidence.reconstructed_objective_gradient,
        "transpose_equality_probe": transpose.equality_probe,
        "transpose_jvp_action": transpose.jvp_action,
        "transpose_state_probe": transpose.state_probe,
        "transpose_vjp_action": transpose.vjp_action,
        "variable_scale": prepared.scaling.variable_scale,
    }
    array_payloads: dict[str, Mapping[str, JsonValue]] = {}
    for name in sorted(ARRAY_SPECS):
        dtype, shape = ARRAY_SPECS[name]
        values = np.ascontiguousarray(np.asarray(arrays[name]), dtype=np.dtype(dtype))
        if values.shape != shape:
            raise QualificationError(f"terminal array {name!r} has wrong shape")
        path = root / "arrays" / f"{name}.npy"
        _publish_npy(path, values)
        reference = _artifact_ref(
            path, root, f"{QUALIFICATION_SCHEMA_VERSION}-array-{name}"
        )
        array_payloads[name] = array_evidence_payload(
            reference=reference,
            name=name,
            values=values,
        )
    endpoint_terms_base = endpoint.evaluation.raw_terms
    config = prepared.problem.config
    weights = {
        "non_qs": float(config.non_qs_weight),
        "residual": float(config.residual_weight),
        "iota": float(config.iota_weight),
        "major_radius": float(config.major_radius_weight),
        "length": float(config.length_weight),
    }
    certificate = base.optimizer_result.final_certificate
    kkt_code = int(np.asarray(diagnostic.raw_kkt_status))
    kkt_status = KktStatus.AVAILABLE if kkt_code == 0 else KktStatus.NONFINITE
    legacy = terminal_numerical_payload(
        arrays=array_payloads,
        objective=float(np.asarray(endpoint.evaluation.weighted_total)),
        objective_terms={
            name: float(np.asarray(getattr(endpoint_terms_base, name)))
            for name in DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS
        },
        objective_weights=weights,
        reconstructed_objective=float(
            np.asarray(terminal_evidence.reconstructed_objective)
        ),
        authoritative_objective=float(
            np.asarray(terminal_evidence.authoritative_objective)
        ),
        final_certificate={
            name: float(np.asarray(getattr(certificate, name)))
            for name in FINAL_CERTIFICATE_FIELDS
        },
        kkt_status=kkt_status,
        raw_kkt_inf=(
            float(np.asarray(raw.raw_kkt_stationarity_infinity_norm))
            if kkt_status is KktStatus.AVAILABLE
            else None
        ),
        scaled_stationarity_inf=(
            float(np.asarray(base.optimizer_result.scaled_stationarity_inf))
            if kkt_status is KktStatus.AVAILABLE
            else None
        ),
        residual_value_defect=float(np.asarray(terminal_evidence.value_scaled_defect)),
        residual_gradient_defect=float(
            np.asarray(terminal_evidence.gradient_scaled_defect)
        ),
        transpose_primal_dot=float(np.asarray(transpose.primal_dot)),
        transpose_adjoint_dot=float(np.asarray(transpose.transpose_dot)),
        transpose_denominator=float(np.asarray(transpose.denominator)),
        transpose_defect=float(np.asarray(transpose.defect)),
        terminal_endpoint_diagnostics_seconds=terminal_seconds,
    )
    terminal = diag4_terminal_numerical_payload(
        terminal_numerical=legacy,
        numerical_identity=numerical_identity,
        endpoint_state_sha256=array_payloads["physical_state"]["content_sha256"],
        terminal_observables=_terminal_observables(endpoint),
        endpoint_objective_terms=endpoint_terms,
        endpoint_observables=endpoint_observables,
    )
    terminal_path = root / TERMINAL_FILENAME
    _publish_json(terminal_path, terminal)
    return terminal, _artifact_ref(
        terminal_path,
        root,
        f"{NEQ_GNTR3_SCHEMA_VERSION}-terminal",
    )


def _history_values(
    history: Mapping[str, JsonValue],
) -> tuple[int, int, int, str, bool, tuple[str, ...]]:
    rows = history.get("rows")
    if not isinstance(rows, list):
        raise QualificationError("history rows are absent")
    outcomes: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("outcome"), str):
            raise QualificationError("history row outcome is invalid")
        outcomes.append(row["outcome"])
    attempts = history.get("attempts")
    accepted = history.get("accepted_steps")
    retries = history.get("retryable_rejections")
    status = history.get("status")
    latch = history.get("quality_latch")
    if (
        type(attempts) is not int
        or type(accepted) is not int
        or type(retries) is not int
        or not isinstance(status, str)
        or type(latch) is not bool
    ):
        raise QualificationError("history counters are invalid")
    return attempts, accepted, retries, status, latch, tuple(outcomes)


def _safeguard_telemetry_from_loop_history(
    *,
    history: Mapping[str, JsonValue],
    history_reference: ArtifactRef,
    numerical_identity: NativeEquivalentNumericalIdentity,
    loop_history: object,
) -> dict[str, JsonValue]:
    """Build all four outer and twenty subtrial safeguard envelopes."""

    attempts, accepted_steps, retries, status, latch, outcomes = _history_values(
        history
    )
    return safeguard_telemetry_payload(
        history_evidence=history_reference,
        problem_sha256=numerical_identity.problem_sha256,
        optimizer_options_sha256=numerical_identity.optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=(numerical_identity.base_neq_gntr1_policy_sha256),
        scaling_sha256=numerical_identity.scaling_sha256,
        bootstrap_state_sha256=numerical_identity.bootstrap_state_sha256,
        initial_physical_state_sha256=(
            numerical_identity.initial_physical_state_sha256
        ),
        identity_sha256=numerical_identity.identity_sha256,
        loop_attempts=attempts,
        accepted_steps=accepted_steps,
        retryable_rejections=retries,
        terminal_status=status,
        quality_latch=latch,
        history_outcomes=outcomes,
        nonlinear_corrections=np.asarray(loop_history.nonlinear_corrections),
        maximum_individual_correction_step_ratio=np.asarray(
            loop_history.maximum_individual_correction_step_ratio
        ),
        correction_path_step_ratio=np.asarray(loop_history.correction_path_step_ratio),
        steihaug_solve_calls=np.asarray(loop_history.steihaug_solve_calls),
        **{
            name: np.asarray(getattr(loop_history, name))
            for name in (
                "subtrial_count",
                "selected_subtrial_index",
                "subtrial_trust_radius",
                "subtrial_outcome",
                "subtrial_actual_reduction",
                "subtrial_predicted_reduction",
                "subtrial_maximum_individual_correction_step_ratio",
                "subtrial_correction_path_step_ratio",
                "subtrial_corrected_radius_ratio",
                "subtrial_steihaug_iterations",
                "subtrial_steihaug_hvp_evaluations",
                "subtrial_steihaug_solve_calls",
                "subtrial_total_hvp_evaluations",
                "subtrial_nonlinear_corrections",
                "subtrial_joint_evaluations",
                "subtrial_joint_linearizations",
                "subtrial_joint_value_evaluations",
                "subtrial_objective_residual_linearizations",
                "subtrial_gram_factorizations",
                "subtrial_gram_solves",
            )
        },
    )


@dataclass(frozen=True, slots=True)
class ProductionCpuQualificationProducer:
    """Compile, execute, serialize, and independently validate exact GNTR3."""

    repository_root: Path = REPOSITORY_ROOT
    worktree_root: Path = WORKTREE_ROOT
    reference_root: Path = NATIVE_REFERENCE_ROOT
    input_root: Path = INPUT_ROOT

    def produce(
        self,
        staging_root: Path,
        runtime_identity: CpuRuntimeIdentity,
    ) -> ProducedEvidence:
        if (
            os.environ.get(_WORKER_ENVIRONMENT) != "1"
            or self.repository_root.name != _EXECUTION_SOURCE_DIRECTORY
        ):
            raise QualificationError(
                "production qualification must execute from bootstrapped source"
            )
        execution_source_bindings = _inherited_execution_source_bindings(
            self.worktree_root,
            self.repository_root,
        )
        native_binding: ImportedSourceBinding | None = None
        imported_source_bindings: ImportedSourceBindings | None = None
        retained_input_trees: tuple[RetainedRegularTree, ...] = ()
        try:
            execution_source_bindings.validate(copied_required=True)
            _validate_public_source_membership(execution_source_bindings)
            native_binding = _capture_native_extension_binding(runtime_identity)
            process_started_ns = time.perf_counter_ns()
            snapshot = publish_immutable_snapshot(
                staging_root / SOURCE_SNAPSHOT_DIRECTORY,
                _source_roots(self.repository_root, execution_source_bindings),
                worktree=capture_worktree_identity(self.worktree_root),
                required_roles=DIAG5_CPU_SNAPSHOT_ROLES,
            )
            execution_source_bindings.validate(copied_required=True)
            _validate_execution_source_snapshot(
                snapshot,
                execution_source_bindings,
                native_binding,
            )
            predecessor_postmortem = _publish_predecessor_postmortem(
                execution_source_bindings,
                staging_root,
            )
            ImportedSourceBindings((native_binding,)).validate()
            retained_native_binding = native_binding
            native_binding = None
            imported_source_bindings = _capture_imported_source_bindings(
                self.repository_root,
                snapshot,
                retained_native_binding,
            )
            reference_tree = _admit_regular_tree(self.reference_root)
            try:
                input_tree = _admit_regular_tree(self.input_root)
            except BaseException:
                reference_tree.close()
                raise
            retained_input_trees = (reference_tree, input_tree)
            return self._produce_with_retained_bindings(
                staging_root=staging_root,
                runtime_identity=runtime_identity,
                process_started_ns=process_started_ns,
                snapshot=snapshot,
                predecessor_postmortem=predecessor_postmortem,
                imported_source_bindings=imported_source_bindings,
                execution_source_bindings=execution_source_bindings,
                retained_input_trees=retained_input_trees,
            )
        except BaseException:
            for tree in retained_input_trees:
                tree.close()
            if imported_source_bindings is not None:
                imported_source_bindings.close()
            elif native_binding is not None:
                ImportedSourceBindings((native_binding,)).close()
            execution_source_bindings.close()
            raise

    def _produce_with_retained_bindings(
        self,
        *,
        staging_root: Path,
        runtime_identity: CpuRuntimeIdentity,
        process_started_ns: int,
        snapshot: SnapshotPublication,
        predecessor_postmortem: ArtifactRef,
        imported_source_bindings: ImportedSourceBindings,
        execution_source_bindings: ExecutionSourceBindings,
        retained_input_trees: tuple[RetainedRegularTree, RetainedRegularTree],
    ) -> ProducedEvidence:
        reference_tree, input_tree = retained_input_trees
        copied_reference = staging_root / "native-reference"
        copied_inputs = staging_root / "inputs"
        _copy_retained_regular_tree(reference_tree, copied_reference)
        _copy_retained_regular_tree(input_tree, copied_inputs)
        validation = validate_native_equivalent_reference(copied_reference)
        if not validation.usable:
            raise QualificationError("native-equivalent reference is not usable")
        native_raw_equalities = _reference_array(copied_reference, "raw_equalities")
        native_reference_state = _reference_array(copied_reference, "state")
        bootstrap = build_single_stage_fullspace_bootstrap()
        scaling = fullspace_scaling_from_bootstrap(bootstrap.z0, bootstrap.problem)
        constraint_inverse_scale = np.asarray(
            jax.device_get(scaling.constraint_inverse_scale), dtype=np.float64
        )
        policy = NativeEquivalentQualityPolicy(
            native_raw_equalities=native_raw_equalities,
            native_raw_equalities_sha256=exact_numeric_tree_sha256(
                native_raw_equalities
            ),
            constraint_inverse_scale=constraint_inverse_scale,
        )
        bundle, arrays = read_input_bundle(copied_inputs)
        expected_input, expected_configuration = _reference_fingerprints(
            copied_reference
        )
        if (
            bundle.input_fingerprint != expected_input
            or bundle.configuration_fingerprint != expected_configuration
        ):
            raise QualificationError("input bundle differs from native reference")
        native_runtime = build_native_single_stage_endpoint_runtime(
            bundle, arrays, bootstrap
        )
        imported_source_bindings.validate()
        execution_source_bindings.validate(copied_required=True)
        reference_tree.validate()
        input_tree.validate()

        compile_started_ns = time.perf_counter_ns()
        prepared = prepare_neq_gntr3(
            bootstrap.problem,
            bootstrap.z0,
            bootstrap.z0,
            policy,
        )
        numerical_identity = _numerical_identity(prepared)
        accepted_quality = prepare_neq_accepted_quality_diagnostics(
            prepared.problem,
            prepared.scaling,
            prepared.policy,
            jnp.zeros(
                (
                    prepared.options.maximum_accepted_steps + 1,
                    prepared.policy.state_size,
                ),
                dtype=jnp.float64,
            ),
            jnp.zeros((prepared.options.maximum_accepted_steps + 1,), dtype=jnp.bool_),
        )
        terminal = prepare_neq_terminal_endpoint_diagnostics(
            prepared.problem,
            prepared.scaling,
            prepared.initial_optimizer_coordinates,
            jnp.zeros((prepared.policy.equality_size,), dtype=jnp.float64),
        )
        callback_count = _callback_count(prepared, accepted_quality, terminal)
        if callback_count != 0:
            raise QualificationError("compiled qualification contains Python callbacks")
        compile_completed_ns = time.perf_counter_ns()
        jax.block_until_ready(prepared.initial_optimizer_coordinates)
        state_ready_ns = time.perf_counter_ns()

        solve_started_ns = time.perf_counter_ns()
        loop = prepared.run_solver_loop()
        jax.block_until_ready(loop)
        solve_stopped_ns = time.perf_counter_ns()
        finalized = prepared.finalize_result(loop)
        jax.block_until_ready(finalized)
        finalizer_completed_ns = time.perf_counter_ns()
        quality_replay = accepted_quality.run(
            loop.accepted_optimizer_coordinates,
            loop.accepted_state_mask,
        )
        jax.block_until_ready(quality_replay)
        quality_replay_completed_ns = time.perf_counter_ns()
        terminal_started_ns = time.perf_counter_ns()
        terminal_evidence = terminal.run_evidence(
            finalized.optimizer_result.optimizer_coordinates,
            finalized.optimizer_result.multipliers,
        )
        jax.block_until_ready(terminal_evidence)
        diagnostic = build_native_equivalent_terminal_diagnostic(
            finalized,
            terminal_evidence.raw_endpoint,
            prepared.policy,
        )
        jax.block_until_ready(diagnostic)
        terminal_completed_ns = time.perf_counter_ns()

        endpoint_audit = produce_native_equivalent_endpoint_audit(
            finalized,
            prepared.problem,
            prepared.scaling,
            prepared.policy,
            native_runtime,
            native_reference_state,
        )
        imported_source_bindings.validate()
        execution_source_bindings.validate(copied_required=True)
        reference_tree.validate()
        input_tree.validate()
        endpoint_document = endpoint_audit_payload(endpoint_audit)
        endpoint_audit_completed_ns = time.perf_counter_ns()
        host_loop, host_diagnostic, host_quality, host_terminal = jax.device_get(
            (loop, diagnostic, quality_replay, terminal_evidence)
        )
        serialization_started_ns = time.perf_counter_ns()

        history = history_evidence_from_arrays(
            host_loop.history,
            quality_latch=bool(
                _host_scalar(host_loop.device_quality_candidate_reached)
            ),
            first_quality_attempt=int(_host_scalar(host_loop.first_quality_attempt)),
            first_quality_accepted_step=int(
                _host_scalar(host_loop.first_quality_accepted_step)
            ),
        )
        history_path = staging_root / HISTORY_FILENAME
        _publish_json(history_path, history)
        history_reference = _artifact_ref(
            history_path,
            staging_root,
            f"{QUALIFICATION_SCHEMA_VERSION}-history",
        )
        telemetry = _safeguard_telemetry_from_loop_history(
            history=history,
            history_reference=history_reference,
            numerical_identity=numerical_identity,
            loop_history=host_loop.history,
        )
        telemetry_path = staging_root / SAFEGUARD_TELEMETRY_FILENAME
        _publish_json(telemetry_path, telemetry)
        telemetry_reference = _artifact_ref(
            telemetry_path,
            staging_root,
            f"{QUALIFICATION_SCHEMA_VERSION}-safeguard-telemetry",
        )
        endpoint_terms = dict(
            zip(
                DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS,
                endpoint_audit.gpu_endpoint_cross_evaluation.jax_raw_terms,
                strict=True,
            )
        )
        endpoint_observables = dict(
            zip(
                DIAG4_ENDPOINT_OBSERVABLE_FIELDS,
                endpoint_audit.gpu_endpoint_cross_evaluation.jax_observables,
                strict=True,
            )
        )
        terminal_document, terminal_reference = _publish_terminal(
            staging_root,
            host_diagnostic,
            host_quality,
            host_terminal,
            prepared,
            numerical_identity,
            endpoint_terms,
            endpoint_observables,
            (terminal_completed_ns - terminal_started_ns) / 1.0e9,
        )
        policy_document = policy_evidence_payload(
            policy_sha256=policy.policy_sha256,
            native_raw_equalities=native_raw_equalities,
            constraint_inverse_scale=constraint_inverse_scale,
        )
        policy_path = staging_root / POLICY_FILENAME
        _publish_json(policy_path, policy_document)
        policy_reference = _artifact_ref(
            policy_path,
            staging_root,
            f"{QUALIFICATION_SCHEMA_VERSION}-policy",
        )
        endpoint_path = staging_root / ENDPOINT_AUDIT_FILENAME
        _publish_bytes(endpoint_path, endpoint_audit_bytes(endpoint_audit))
        endpoint_reference = _artifact_ref(
            endpoint_path,
            staging_root,
            f"{QUALIFICATION_SCHEMA_VERSION}-endpoint-audit",
        )
        scientific = validate_native_equivalent_scientific_evidence(
            artifact_root=staging_root,
            history=history,
            safeguard_telemetry=telemetry,
            terminal_numerical=terminal_document,
            policy=policy_document,
            endpoint_audit=endpoint_document,
            expected_history_evidence=history_reference,
            expected_numerical_identity=numerical_identity,
            backend=runtime_identity.backend,
        )
        serialization_completed_ns = time.perf_counter_ns()
        timings = (
            ("process_started_monotonic_ns", process_started_ns),
            ("compile_started_monotonic_ns", compile_started_ns),
            ("compile_completed_monotonic_ns", compile_completed_ns),
            ("state_ready_monotonic_ns", state_ready_ns),
            ("solve_started_monotonic_ns", solve_started_ns),
            ("solve_stopped_monotonic_ns", solve_stopped_ns),
            ("finalizer_completed_monotonic_ns", finalizer_completed_ns),
            ("quality_replay_completed_monotonic_ns", quality_replay_completed_ns),
            ("terminal_completed_monotonic_ns", terminal_completed_ns),
            ("endpoint_audit_completed_monotonic_ns", endpoint_audit_completed_ns),
            ("serialization_started_monotonic_ns", serialization_started_ns),
            ("serialization_completed_monotonic_ns", serialization_completed_ns),
        )
        _publish_json(
            staging_root / "evidence-index.json",
            {
                "endpoint_audit": _artifact_ref_payload(endpoint_reference),
                "history": _artifact_ref_payload(history_reference),
                "policy": _artifact_ref_payload(policy_reference),
                "safeguard_telemetry": _artifact_ref_payload(telemetry_reference),
                "terminal_numerical": _artifact_ref_payload(terminal_reference),
            },
        )
        execution_source_bindings.validate(copied_required=True)
        reference_tree.validate()
        input_tree.validate()
        native_binding = next(
            binding
            for binding in imported_source_bindings.entries
            if binding.relative_path.startswith("native/")
        )
        return ProducedEvidence(
            scientific_outcome=scientific.outcome,
            numerical_identity=numerical_identity,
            timings_ns=timings,
            callback_count=callback_count,
            execution_source_manifest_sha256=(
                execution_source_bindings.authority_sha256
            ),
            execution_source_entries_sha256=(execution_source_bindings.entries_sha256),
            prequalification_plan_control=tuple(
                _prequalification_plan_control(execution_source_bindings).items()
            ),
            source_manifest_sha256=snapshot.manifest_sha256,
            source_manifest_entries=tuple(
                (entry.relative_path, entry.role, entry.size_bytes, entry.sha256)
                for entry in snapshot.entries
            ),
            native_extension_path=str(native_binding.path),
            native_extension_sha256=native_binding.sha256,
            native_extension_size_bytes=native_binding.size_bytes,
            native_extension_link_count=native_binding.link_count,
            native_extension_device=native_binding.device,
            native_extension_inode=native_binding.inode,
            predecessor_postmortem=predecessor_postmortem,
            native_reference_artifact_sha256=validation.artifact_sha256,
            input_fingerprint=bundle.input_fingerprint,
            configuration_fingerprint=bundle.configuration_fingerprint,
            policy_sha256=policy.policy_sha256,
            imported_source_bindings=imported_source_bindings,
            execution_source_bindings=execution_source_bindings,
            retained_input_trees=retained_input_trees,
        )

    def validate(
        self,
        artifact_root: Path,
        qualification: Mapping[str, JsonValue],
    ) -> ScientificOutcome:
        source = load_snapshot(
            artifact_root / SOURCE_SNAPSHOT_DIRECTORY,
            required_roles=DIAG5_CPU_SNAPSHOT_ROLES,
        )
        _validate_snapshot_qualification_identity(source, qualification)
        if source.manifest_sha256 != qualification.get("source_manifest_sha256"):
            raise QualificationError("source snapshot identity differs")
        source_entries = [
            {
                "relative_path": entry.relative_path,
                "role": entry.role,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in source.entries
        ]
        if qualification.get("source_manifest_entries") != source_entries:
            raise QualificationError("source snapshot entries differ")
        native_validation = validate_native_equivalent_reference(
            artifact_root / "native-reference"
        )
        if (
            not native_validation.usable
            or qualification.get("native_reference_artifact_sha256")
            != native_validation.artifact_sha256
        ):
            raise QualificationError("native reference authority differs")
        input_bundle, _input_arrays = read_input_bundle(artifact_root / "inputs")
        if (
            qualification.get("input_fingerprint") != input_bundle.input_fingerprint
            or qualification.get("configuration_fingerprint")
            != input_bundle.configuration_fingerprint
        ):
            raise QualificationError("input authority differs")
        evidence_index = load_canonical_json_bytes(
            (artifact_root / "evidence-index.json").read_bytes()
        )
        if not isinstance(evidence_index, dict):
            raise QualificationError("evidence index must be an object")
        refs = {
            name: _artifact_ref_from_payload(value)
            for name, value in evidence_index.items()
        }
        if frozenset(refs) != frozenset(
            {
                "endpoint_audit",
                "history",
                "policy",
                "safeguard_telemetry",
                "terminal_numerical",
            }
        ):
            raise QualificationError("evidence index differs")
        identity_value = qualification.get("numerical_identity")
        if not isinstance(identity_value, dict):
            raise QualificationError("qualification numerical identity is absent")
        expected_identity = NativeEquivalentNumericalIdentity(**identity_value)
        if (
            qualification.get("policy_sha256")
            != expected_identity.base_neq_gntr1_policy_sha256
        ):
            raise QualificationError("qualification policy identity differs")
        scientific = validate_native_equivalent_scientific_evidence(
            artifact_root=artifact_root,
            history=_load_json_artifact(artifact_root, refs["history"]),
            safeguard_telemetry=_load_json_artifact(
                artifact_root, refs["safeguard_telemetry"]
            ),
            terminal_numerical=_load_json_artifact(
                artifact_root, refs["terminal_numerical"]
            ),
            policy=_load_json_artifact(artifact_root, refs["policy"]),
            endpoint_audit=_load_json_artifact(artifact_root, refs["endpoint_audit"]),
            expected_history_evidence=refs["history"],
            expected_numerical_identity=expected_identity,
            backend="cpu",
        )
        if (
            qualification.get("scientific_outcome") != scientific.outcome.value
            or qualification.get("qualification_passed")
            is not (scientific.outcome is ScientificOutcome.QUALITY_HIT)
            or qualification.get("speed") != SPEED_NOT_PRODUCED
        ):
            raise QualificationError("qualification summary differs from evidence")
        return scientific.outcome


def _qualification_payload(
    produced: ProducedEvidence,
    runtime: CpuRuntimeIdentity,
    output_root: Path,
) -> dict[str, JsonValue]:
    scientific_outcome = produced.scientific_outcome
    timing_names = (
        "process_started_monotonic_ns",
        "compile_started_monotonic_ns",
        "compile_completed_monotonic_ns",
        "state_ready_monotonic_ns",
        "solve_started_monotonic_ns",
        "solve_stopped_monotonic_ns",
        "finalizer_completed_monotonic_ns",
        "quality_replay_completed_monotonic_ns",
        "terminal_completed_monotonic_ns",
        "endpoint_audit_completed_monotonic_ns",
        "serialization_started_monotonic_ns",
        "serialization_completed_monotonic_ns",
    )
    if tuple(name for name, _ in produced.timings_ns) != timing_names:
        raise QualificationError("qualification timing fields differ")
    timing_values = tuple(value for _, value in produced.timings_ns)
    if any(type(value) is not int for value in timing_values) or any(
        left >= right for left, right in zip(timing_values[:-1], timing_values[1:])
    ):
        raise QualificationError("qualification timing order differs")
    if produced.callback_count != 0:
        raise QualificationError("qualification callback count must be zero")
    if (
        produced.native_extension_path,
        produced.native_extension_sha256,
        produced.native_extension_size_bytes,
        produced.native_extension_link_count,
    ) != (
        runtime.native_extension_path,
        runtime.native_extension_sha256,
        runtime.native_extension_size_bytes,
        runtime.native_extension_link_count,
    ):
        raise QualificationError("qualification native runtime identity differs")
    timings = dict(produced.timings_ns)
    solve_seconds = (
        timings["solve_stopped_monotonic_ns"] - timings["solve_started_monotonic_ns"]
    ) / 1.0e9
    return {
        "backend": runtime.backend,
        "callback_count": produced.callback_count,
        "configuration_fingerprint": produced.configuration_fingerprint,
        "execution_source_entries_sha256": produced.execution_source_entries_sha256,
        "execution_source_manifest_sha256": (produced.execution_source_manifest_sha256),
        "input_fingerprint": produced.input_fingerprint,
        "cpu_native_binding": {
            "cpu_native_extension_device": produced.native_extension_device,
            "cpu_native_extension_inode": produced.native_extension_inode,
            "cpu_native_extension_link_count": produced.native_extension_link_count,
            "cpu_native_extension_path": produced.native_extension_path,
            "native_extension_sha256": produced.native_extension_sha256,
            "native_extension_size_bytes": produced.native_extension_size_bytes,
        },
        "native_reference_artifact_sha256": (produced.native_reference_artifact_sha256),
        "numerical_identity": asdict(produced.numerical_identity),
        "output_root": str(output_root),
        "policy_sha256": produced.policy_sha256,
        "prequalification_plan_control": dict(produced.prequalification_plan_control),
        "predecessor_postmortem": _artifact_ref_payload(
            produced.predecessor_postmortem
        ),
        "promotion_eligible": False,
        "qualification_passed": scientific_outcome is ScientificOutcome.QUALITY_HIT,
        "route": NEQ_GNTR3_ROUTE,
        "runtime": runtime.to_payload(),
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "scientific_outcome": scientific_outcome.value,
        "source_manifest_entries": [
            {
                "relative_path": relative_path,
                "role": role,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
            for relative_path, role, size_bytes, sha256 in produced.source_manifest_entries
        ],
        "source_manifest_sha256": produced.source_manifest_sha256,
        "speed": SPEED_NOT_PRODUCED,
        "synchronized_solve_seconds": solve_seconds,
        "timings_monotonic_ns": timings,
    }


def _manifest_payload(root: Path) -> dict[str, JsonValue]:
    files: list[dict[str, JsonValue]] = []
    directories: list[dict[str, JsonValue]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise QualificationError("qualification artifact contains a symlink")
        observed = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            directories.append({"mode": "0555", "relative_path": relative})
        elif stat.S_ISREG(observed.st_mode):
            if relative == MANIFEST_FILENAME:
                raise QualificationError("qualification manifest already exists")
            if observed.st_nlink != 1:
                raise QualificationError("qualification artifact contains a hardlink")
            payload = path.read_bytes()
            files.append(
                {
                    "mode": "0444",
                    "relative_path": relative,
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                }
            )
        else:
            raise QualificationError("qualification artifact contains a special file")
    qualification = load_canonical_json_bytes(
        (root / QUALIFICATION_FILENAME).read_bytes()
    )
    if not isinstance(qualification, dict):
        raise QualificationError("qualification document must be an object")
    return {
        "directories": directories,
        "execution_source_entries_sha256": qualification.get(
            "execution_source_entries_sha256"
        ),
        "execution_source_manifest_sha256": qualification.get(
            "execution_source_manifest_sha256"
        ),
        "files": files,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }


def _seal_and_sync(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise QualificationError("qualification artifact contains a symlink")
        observed = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            directories.append(path)
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise QualificationError("qualification artifact contains an invalid file")
        path.chmod(0o444)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        directory.chmod(0o555)
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _validate_manifest(root: Path, *, sealed: bool) -> None:
    manifest = load_canonical_json_bytes((root / MANIFEST_FILENAME).read_bytes())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        raise QualificationError("qualification manifest schema differs")
    expected = _manifest_payload_without_manifest(root)
    if manifest != expected:
        raise QualificationError("qualification manifest differs from artifact tree")
    if sealed:
        for path in (root, *root.rglob("*")):
            observed = path.stat(follow_symlinks=False)
            expected_mode = 0o555 if stat.S_ISDIR(observed.st_mode) else 0o444
            if stat.S_IMODE(observed.st_mode) != expected_mode:
                raise QualificationError("qualification artifact mode differs")


def _manifest_payload_without_manifest(root: Path) -> dict[str, JsonValue]:
    manifest_path = root / MANIFEST_FILENAME
    files: list[dict[str, JsonValue]] = []
    directories: list[dict[str, JsonValue]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path == manifest_path:
            continue
        observed = path.stat(follow_symlinks=False)
        if path.is_symlink():
            raise QualificationError("qualification artifact contains a symlink")
        if stat.S_ISDIR(observed.st_mode):
            directories.append({"mode": "0555", "relative_path": relative})
        elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
            payload = path.read_bytes()
            files.append(
                {
                    "mode": "0444",
                    "relative_path": relative,
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                }
            )
        else:
            raise QualificationError("qualification artifact contains an invalid file")
    qualification = load_canonical_json_bytes(
        (root / QUALIFICATION_FILENAME).read_bytes()
    )
    if not isinstance(qualification, dict):
        raise QualificationError("qualification document must be an object")
    return {
        "directories": directories,
        "execution_source_entries_sha256": qualification.get(
            "execution_source_entries_sha256"
        ),
        "execution_source_manifest_sha256": qualification.get(
            "execution_source_manifest_sha256"
        ),
        "files": files,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }


def _load_qualification(
    root: Path,
    *,
    expected_output_root: Path | None = None,
) -> dict[str, JsonValue]:
    value = load_canonical_json_bytes((root / QUALIFICATION_FILENAME).read_bytes())
    if not isinstance(value, dict):
        raise QualificationError("qualification document must be an object")
    expected_keys = frozenset(
        {
            "backend",
            "callback_count",
            "configuration_fingerprint",
            "execution_source_entries_sha256",
            "execution_source_manifest_sha256",
            "input_fingerprint",
            "cpu_native_binding",
            "native_reference_artifact_sha256",
            "numerical_identity",
            "output_root",
            "policy_sha256",
            "prequalification_plan_control",
            "predecessor_postmortem",
            "promotion_eligible",
            "qualification_passed",
            "route",
            "runtime",
            "schema_version",
            "scientific_outcome",
            "source_manifest_entries",
            "source_manifest_sha256",
            "speed",
            "synchronized_solve_seconds",
            "timings_monotonic_ns",
        }
    )
    if frozenset(value) != expected_keys:
        raise QualificationError("qualification document fields differ")
    if (
        value["schema_version"] != QUALIFICATION_SCHEMA_VERSION
        or value["route"] != NEQ_GNTR3_ROUTE
        or value["backend"] != "cpu"
        or value["speed"] != SPEED_NOT_PRODUCED
        or value["promotion_eligible"] is not False
        or value["output_root"] != str(expected_output_root or root)
        or value["callback_count"] != 0
    ):
        raise QualificationError("qualification identity or claim boundary differs")
    for name in (
        "execution_source_entries_sha256",
        "execution_source_manifest_sha256",
    ):
        digest = value[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise QualificationError(f"qualification {name} differs")
    cpu_native_binding = value["cpu_native_binding"]
    if not isinstance(cpu_native_binding, dict) or frozenset(
        cpu_native_binding
    ) != frozenset(
        {
            "cpu_native_extension_device",
            "cpu_native_extension_inode",
            "cpu_native_extension_link_count",
            "cpu_native_extension_path",
            "native_extension_sha256",
            "native_extension_size_bytes",
        }
    ):
        raise QualificationError("qualification CPU native binding fields differ")
    native_extension_path = cpu_native_binding["cpu_native_extension_path"]
    native_extension_size_bytes = cpu_native_binding["native_extension_size_bytes"]
    native_extension_link_count = cpu_native_binding["cpu_native_extension_link_count"]
    native_extension_device = cpu_native_binding["cpu_native_extension_device"]
    native_extension_inode = cpu_native_binding["cpu_native_extension_inode"]
    native_extension_sha256 = cpu_native_binding["native_extension_sha256"]
    if (
        not isinstance(native_extension_path, str)
        or not Path(native_extension_path).is_absolute()
        or type(native_extension_size_bytes) is not int
        or native_extension_size_bytes < 0
        or type(native_extension_link_count) is not int
        or native_extension_link_count < 1
        or type(native_extension_device) is not int
        or native_extension_device < 0
        or type(native_extension_inode) is not int
        or native_extension_inode < 1
        or not isinstance(native_extension_sha256, str)
        or len(native_extension_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in native_extension_sha256
        )
    ):
        raise QualificationError("qualification native extension identity differs")
    native_extension_metadata, observed_native_sha256 = _observe_absolute_regular(
        Path(native_extension_path)
    )
    if (
        not stat.S_ISREG(native_extension_metadata.st_mode)
        or native_extension_metadata.st_dev != native_extension_device
        or native_extension_metadata.st_ino != native_extension_inode
        or native_extension_metadata.st_size != native_extension_size_bytes
        or native_extension_metadata.st_nlink != native_extension_link_count
        or observed_native_sha256 != native_extension_sha256
    ):
        raise QualificationError("qualification CPU native binding differs")
    plan_control = value["prequalification_plan_control"]
    if not isinstance(plan_control, dict) or frozenset(plan_control) != frozenset(
        {
            "plan_prefix_sha256",
            "schema_version",
            "sha256",
            "size_bytes",
            "snapshot_relative_path",
            "source_relative_path",
        }
    ):
        raise QualificationError("prequalification plan control fields differ")
    plan_sha256 = plan_control["sha256"]
    if (
        plan_control["schema_version"] != PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION
        or plan_control["snapshot_relative_path"]
        != PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH
        or plan_control["source_relative_path"]
        != _PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH
        or plan_control["plan_prefix_sha256"] != DIAG5_PLAN_SHA256
        or plan_control["sha256"] != DIAG5_BLANK_PLAN_SHA256
        or plan_control["size_bytes"] != DIAG5_BLANK_PLAN_SIZE_BYTES
        or not isinstance(plan_sha256, str)
        or len(plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan_sha256)
        or type(plan_control["size_bytes"]) is not int
        or plan_control["size_bytes"] < 0
    ):
        raise QualificationError("prequalification plan control identity differs")
    predecessor_postmortem = _artifact_ref_from_payload(value["predecessor_postmortem"])
    if (
        predecessor_postmortem.relative_path
        != PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH
        or predecessor_postmortem.schema_version
        != PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
    ):
        raise QualificationError("predecessor postmortem reference differs")
    validate_diag5_predecessor_postmortem_artifact(root, predecessor_postmortem)
    outcome_value = value["scientific_outcome"]
    if not isinstance(outcome_value, str):
        raise QualificationError("qualification scientific outcome is invalid")
    outcome = ScientificOutcome(outcome_value)
    if value["qualification_passed"] is not (outcome is ScientificOutcome.QUALITY_HIT):
        raise QualificationError("qualification pass flag differs")
    identity = value["numerical_identity"]
    if not isinstance(identity, dict):
        raise QualificationError("qualification numerical identity is invalid")
    parsed_identity = NativeEquivalentNumericalIdentity(**identity)
    if (
        parsed_identity.numerical_route != NEQ_GNTR3_ROUTE
        or parsed_identity.numerical_result_schema_version != NEQ_GNTR3_SCHEMA_VERSION
    ):
        raise QualificationError("qualification numerical route differs")
    runtime = value["runtime"]
    if not isinstance(runtime, dict):
        raise QualificationError("qualification runtime is invalid")
    runtime_environment = runtime.get("environment")
    runtime_devices = runtime.get("devices")
    if (
        runtime.get("backend") != "cpu"
        or runtime.get("x64_enabled") is not True
        or runtime.get("native_extension_path") != native_extension_path
        or runtime.get("native_extension_sha256") != native_extension_sha256
        or runtime.get("native_extension_size_bytes") != native_extension_size_bytes
        or runtime.get("native_extension_link_count") != native_extension_link_count
        or runtime_environment != _REQUIRED_ENVIRONMENT
        or not isinstance(runtime_devices, list)
        or not runtime_devices
        or any(
            not isinstance(device, dict) or device.get("platform") != "cpu"
            for device in runtime_devices
        )
    ):
        raise QualificationError("qualification CPU runtime differs")
    timing_names = (
        "process_started_monotonic_ns",
        "compile_started_monotonic_ns",
        "compile_completed_monotonic_ns",
        "state_ready_monotonic_ns",
        "solve_started_monotonic_ns",
        "solve_stopped_monotonic_ns",
        "finalizer_completed_monotonic_ns",
        "quality_replay_completed_monotonic_ns",
        "terminal_completed_monotonic_ns",
        "endpoint_audit_completed_monotonic_ns",
        "serialization_started_monotonic_ns",
        "serialization_completed_monotonic_ns",
    )
    timings = value["timings_monotonic_ns"]
    if not isinstance(timings, dict) or frozenset(timings) != frozenset(timing_names):
        raise QualificationError("qualification timing fields differ")
    timing_values = tuple(timings[name] for name in timing_names)
    if any(type(item) is not int for item in timing_values) or any(
        left >= right for left, right in zip(timing_values[:-1], timing_values[1:])
    ):
        raise QualificationError("qualification timing order differs")
    synchronized_seconds = value["synchronized_solve_seconds"]
    expected_seconds = (timing_values[5] - timing_values[4]) / 1.0e9
    if (
        type(synchronized_seconds) is not float
        or synchronized_seconds <= 0.0
        or synchronized_seconds != expected_seconds
    ):
        raise QualificationError("qualification solve timing arithmetic differs")
    return value


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _renameat2_publication(publication: Publication) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            publication.parent_descriptor,
            os.fsencode(publication.staging_root.name),
            publication.parent_descriptor,
            os.fsencode(publication.final_root.name),
            1,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                publication.final_root,
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            publication.final_root,
        )


def _rename_publication(publication: Publication) -> None:
    """Publish the currently bound staging entry without replacing a final root."""

    _validate_publication_binding(publication, published=False)
    _renameat2_publication(publication)
    _validate_publication_binding(publication, published=True)


def _fsync_parent(publication: Publication) -> None:
    os.fsync(publication.parent_descriptor)


def validate_cpu_trajectory_qualification_artifact(
    artifact_root: Path,
    *,
    producer: CpuQualificationProducer,
) -> dict[str, JsonValue]:
    """Deep-load one sealed artifact and join its scientific evidence."""

    _validate_manifest(artifact_root, sealed=True)
    qualification = _load_qualification(artifact_root)
    observed_outcome = producer.validate(artifact_root, qualification)
    expected_outcome = ScientificOutcome(qualification["scientific_outcome"])
    if observed_outcome is not expected_outcome:
        raise QualificationError("artifact scientific outcome differs")
    return qualification


def _run_claimed_qualification(
    publication: Publication,
    *,
    producer: CpuQualificationProducer,
    runtime: CpuRuntimeIdentity,
) -> dict[str, JsonValue]:
    """Execute and publish while the caller retains both directory capabilities."""

    _validate_publication_binding(publication, published=False)
    produced = producer.produce(publication.staging_root, runtime)
    source_bindings = produced.imported_source_bindings
    execution_source_bindings = produced.execution_source_bindings
    retained_input_trees = produced.retained_input_trees
    try:
        if source_bindings is not None:
            source_bindings.validate()
        if execution_source_bindings is not None:
            execution_source_bindings.validate(copied_required=True)
        for tree in retained_input_trees:
            tree.validate()
        _validate_publication_binding(publication, published=False)
        qualification = _qualification_payload(
            produced,
            runtime,
            publication.final_root,
        )
        _publish_json(publication.staging_root / QUALIFICATION_FILENAME, qualification)
        _validate_publication_binding(publication, published=False)
        _publish_json(
            publication.staging_root / MANIFEST_FILENAME,
            _manifest_payload(publication.staging_root),
        )
        _validate_publication_binding(publication, published=False)
        _validate_manifest(publication.staging_root, sealed=False)
        staging_qualification = _load_qualification(
            publication.staging_root,
            expected_output_root=publication.final_root,
        )
        if (
            producer.validate(publication.staging_root, staging_qualification)
            is not produced.scientific_outcome
        ):
            raise QualificationError("staging scientific outcome differs")
        if source_bindings is not None:
            source_bindings.validate()
        if execution_source_bindings is not None:
            execution_source_bindings.validate(copied_required=True)
        for tree in retained_input_trees:
            tree.validate()
        _validate_publication_binding(publication, published=False)
        _seal_and_sync(publication.staging_root)
        _validate_publication_binding(publication, published=False)
        _validate_manifest(publication.staging_root, sealed=True)
        _load_qualification(
            publication.staging_root,
            expected_output_root=publication.final_root,
        )
        if source_bindings is not None:
            source_bindings.validate()
        if execution_source_bindings is not None:
            execution_source_bindings.validate(copied_required=True)
        for tree in retained_input_trees:
            tree.validate()
        _validate_publication_binding(publication, published=False)
        _rename_publication(publication)
        _fsync_parent(publication)
        _validate_publication_binding(publication, published=True)
        final_qualification = validate_cpu_trajectory_qualification_artifact(
            publication.final_root,
            producer=producer,
        )
        if source_bindings is not None:
            source_bindings.validate()
        if execution_source_bindings is not None:
            execution_source_bindings.validate(
                copied_required=True,
                copied_root=publication.final_root / _EXECUTION_SOURCE_DIRECTORY,
            )
        for tree in retained_input_trees:
            tree.validate()
        _validate_publication_binding(publication, published=True)
        if ScientificOutcome(final_qualification["scientific_outcome"]) is not (
            produced.scientific_outcome
        ):
            raise QualificationError("final scientific outcome differs")
        return final_qualification
    finally:
        if source_bindings is not None:
            source_bindings.close()
        if execution_source_bindings is not None:
            execution_source_bindings.close()
        for tree in retained_input_trees:
            tree.close()


def run_qualification(
    output_root: Path,
    *,
    producer: CpuQualificationProducer,
    environment: Mapping[str, str],
) -> dict[str, JsonValue]:
    """Claim once, retain any fault visibly, and hold the claim through deep-load."""

    runtime = observe_cpu_runtime(environment)
    publication = _prepare_publication(output_root)
    try:
        return _run_claimed_qualification(
            publication,
            producer=producer,
            runtime=runtime,
        )
    finally:
        _close_publication(publication)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _inherited_publication(output_root: Path) -> Publication:
    parent_descriptor = int(os.environ[_PARENT_DESCRIPTOR_ENVIRONMENT])
    staging_descriptor = int(os.environ[_STAGING_DESCRIPTOR_ENVIRONMENT])
    parent_observed = os.fstat(parent_descriptor)
    staging_observed = os.fstat(staging_descriptor)
    publication = Publication(
        output_root.parent / f"{output_root.name}.partial-claim",
        output_root,
        parent_descriptor,
        staging_descriptor,
        parent_observed.st_dev,
        parent_observed.st_ino,
        staging_observed.st_dev,
        staging_observed.st_ino,
    )
    _validate_publication_binding(publication, published=False)
    return publication


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.output_root != EXPECTED_OUTPUT_ROOT:
        raise QualificationError(f"output root must be exactly {EXPECTED_OUTPUT_ROOT}")
    producer = ProductionCpuQualificationProducer()
    if os.environ.get(_WORKER_ENVIRONMENT) == "1":
        runtime = observe_cpu_runtime(os.environ)
        publication = _inherited_publication(arguments.output_root)
        try:
            result = _run_claimed_qualification(
                publication,
                producer=producer,
                runtime=runtime,
            )
        finally:
            _close_publication(publication)
    else:
        result = run_qualification(
            arguments.output_root,
            producer=producer,
            environment=os.environ,
        )
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
