"""Exclusive, atomic publication for parity run directories."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from examples.jax.parity.artifacts import canonical_json_bytes

_RUN_ID = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8,32}$")


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
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
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
    marker.write_bytes(canonical_json_bytes({"reason": reason, "status": "failed"}))
    with marker.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_directory(paths.partial)
    return marker


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PublicationError(f"published run must not contain symlinks: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in reversed(directories):
        _fsync_directory(directory)


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
    try:
        paths.partial.rename(paths.final)
    except FileExistsError as error:
        raise PublicationError(f"parity run already exists: {paths.run_id}") from error
    _fsync_directory(paths.root)
    return paths.final


def require_published_run(root: Path, run_id: str) -> Path:
    """Return only a complete final directory suitable for independent audit."""
    _validate_run_id(run_id)
    final = root / run_id
    if (
        not final.is_dir()
        or final.is_symlink()
        or not (final / "summary.json").is_file()
    ):
        raise PublicationError(f"not a published run: {run_id}")
    return final
