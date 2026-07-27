"""Exclusive, atomic publication for parity run directories."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import stat
import sys

from examples.jax.parity.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    read_bytes,
    write_bytes_exclusive,
)

_RUN_ID = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8,32}$")
_COMPLETION_MARKER = "COMPLETED.json"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


class PublicationError(RuntimeError):
    """A parity run cannot be exclusively or completely published."""


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    partial: Path
    final: Path


def _validate_run_id(run_id: str) -> None:
    if _RUN_ID.fullmatch(run_id) is None:
        raise PublicationError(f"invalid parity run ID: {run_id!r}")


def _fsync_directory(directory: Path) -> None:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if sys.platform != "linux" or any(not hasattr(os, name) for name in required):
        raise PublicationError("durable no-follow publication is unavailable")
    try:
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as error:
        raise PublicationError(
            f"publication directory is not trusted: {directory}"
        ) from error
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def begin_run(root: Path, run_id: str) -> RunPaths:
    """Claim one unique partial directory without overwriting prior evidence."""
    _validate_run_id(run_id)
    root.mkdir(parents=True, exist_ok=True)
    partial = root / f"{run_id}.partial"
    final = root / run_id
    if final.exists():
        raise PublicationError(f"parity run already exists: {run_id}")
    try:
        partial.mkdir()
    except FileExistsError as error:
        raise PublicationError(f"parity run already exists: {run_id}") from error
    _fsync_directory(root)
    return RunPaths(run_id=run_id, root=root, partial=partial, final=final)


def mark_run_failed(paths: RunPaths, reason: str) -> Path:
    """Leave a diagnostic marker in an unpublished partial run."""
    if not paths.partial.is_dir() or paths.partial.is_symlink():
        raise PublicationError(f"partial run does not exist: {paths.run_id}")
    marker = paths.partial / "FAILURE.json"
    try:
        write_bytes_exclusive(
            paths.partial,
            marker.name,
            canonical_json_bytes({"reason": reason, "status": "failed"}),
        )
    except ArtifactValidationError as error:
        raise PublicationError(
            f"failed-run marker cannot be published: {paths.run_id}"
        ) from error
    return marker


def _fsync_tree_descriptor(directory_fd: int, context: Path) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    for name in sorted(os.listdir(directory_fd)):
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except OSError as error:
            raise PublicationError(
                f"published run contains a substituted or unreadable entry: {context / name}"
            ) from error
        try:
            mode = os.fstat(descriptor).st_mode
            if stat.S_ISDIR(mode):
                _fsync_tree_descriptor(descriptor, context / name)
            elif not stat.S_ISREG(mode):
                raise PublicationError(
                    f"published run contains a non-regular entry: {context / name}"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if sys.platform != "linux" or any(not hasattr(os, name) for name in required):
        raise PublicationError("durable no-follow publication is unavailable")
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as error:
        raise PublicationError(
            f"partial run is not a trusted directory: {root}"
        ) from error
    try:
        _fsync_tree_descriptor(descriptor, root)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically publish one directory without replacing any target entry."""
    if sys.platform != "linux":
        raise PublicationError("atomic no-replace directory publication is unavailable")
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise PublicationError(
            "atomic no-replace directory publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {
        errno.EEXIST,
        errno.EISDIR,
        errno.ENOTDIR,
        errno.ENOTEMPTY,
    }:
        raise PublicationError(f"parity run already exists: {target.name}")
    raise PublicationError(
        f"atomic no-replace publication failed: {os.strerror(error_number)}"
    )


def _completion_payload(paths: RunPaths) -> bytes:
    try:
        summary = read_bytes(paths.final, "summary.json")
    except ArtifactValidationError as error:
        raise PublicationError(
            "published parity run requires a trusted summary.json"
        ) from error
    return canonical_json_bytes(
        {
            "run_id": paths.run_id,
            "schema_version": 1,
            "status": "complete",
            "summary_sha256": hashlib.sha256(summary).hexdigest(),
        }
    )


def publish_run(paths: RunPaths) -> Path:
    """Atomically rename a complete partial run into its immutable final name."""
    if (paths.partial / "FAILURE.json").exists():
        raise PublicationError(f"cannot publish failed partial run: {paths.run_id}")
    summary = paths.partial / "summary.json"
    if not summary.is_file() or summary.is_symlink():
        raise PublicationError("partial parity run requires summary.json")
    if paths.final.exists():
        raise PublicationError(f"parity run already exists: {paths.run_id}")
    _fsync_tree(paths.partial)
    _rename_no_replace(paths.partial, paths.final)
    try:
        write_bytes_exclusive(
            paths.final,
            _COMPLETION_MARKER,
            _completion_payload(paths),
        )
    except ArtifactValidationError as error:
        raise PublicationError(
            f"completion marker cannot be published: {paths.run_id}"
        ) from error
    _fsync_directory(paths.final)
    _fsync_directory(paths.root)
    return paths.final


def require_published_run(root: Path, run_id: str) -> Path:
    """Return only a complete final directory suitable for independent audit."""
    _validate_run_id(run_id)
    final = root / run_id
    try:
        marker_bytes = read_bytes(final, _COMPLETION_MARKER)
        summary = read_bytes(final, "summary.json")
        marker = json.loads(marker_bytes)
    except (ArtifactValidationError, json.JSONDecodeError):
        raise PublicationError(f"not a published run: {run_id}")
    expected = {
        "run_id": run_id,
        "schema_version": 1,
        "status": "complete",
        "summary_sha256": hashlib.sha256(summary).hexdigest(),
    }
    if marker != expected:
        raise PublicationError(f"not a published run: {run_id}")
    return final
