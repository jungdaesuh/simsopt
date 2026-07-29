"""Canonical, hash-bound parity artifact serialization."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

import numpy as np
from examples.jax.parity.contracts import ArrayReference


class ArtifactValidationError(ValueError):
    """A parity artifact violates its containment or integrity contract."""


def _relative_artifact_path(relative_path: str) -> PurePosixPath:
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative_path
    ):
        raise ArtifactValidationError(
            f"artifact requires a canonical relative path: {relative_path!r}"
        )
    return path


def canonical_json_bytes(payload: object) -> bytes:
    """Return canonical UTF-8 JSON with stable ordering and no NaN values."""
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _relative_npy_path(relative_path: str) -> PurePosixPath:
    path = _relative_artifact_path(relative_path)
    if path.suffix != ".npy":
        raise ArtifactValidationError(
            f"array sidecar requires a canonical relative path: {relative_path!r}"
        )
    return path


def _contained_path(root: Path, relative_path: str) -> Path:
    path = _relative_npy_path(relative_path)
    if root.is_symlink():
        raise ArtifactValidationError("artifact root must not be a symlink")
    candidate = root.joinpath(*path.parts)
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactValidationError(
                f"array sidecar must not be a symlink: {path}"
            )
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    if not candidate_resolved.is_relative_to(root_resolved):
        raise ArtifactValidationError(f"array sidecar escapes artifact root: {path}")
    return candidate


def _canonical_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise ArtifactValidationError("non-numeric and object arrays are forbidden")
    if not bool(np.all(np.isfinite(array))):
        raise ArtifactValidationError("non-finite array values are forbidden")
    canonical_dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array.astype(canonical_dtype, copy=False))


def _require_descriptor_io() -> None:
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise ArtifactValidationError(
            "descriptor-relative no-follow artifact I/O is unavailable"
        )
    if os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd:
        raise ArtifactValidationError(
            "descriptor-relative no-follow artifact I/O is unavailable"
        )


def _directory_flags() -> int:
    _require_descriptor_io()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


@contextmanager
def _parent_descriptor(
    root: Path,
    path: PurePosixPath,
    *,
    create: bool,
) -> Iterator[tuple[int, str]]:
    if create:
        root.mkdir(parents=True, exist_ok=True)
    flags = _directory_flags()
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise ArtifactValidationError(
            "artifact root must be a real directory opened with no-follow semantics"
        ) from error
    try:
        for part in path.parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise ArtifactValidationError(
                    f"artifact path component is a symlink or non-directory: {path}"
                ) from error
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor, path.name
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_binary_writer(
    root: Path,
    path: PurePosixPath,
) -> Iterator[BinaryIO]:
    with _parent_descriptor(root, path, create=True) as (parent_fd, leaf):
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            descriptor = os.open(leaf, flags, 0o644, dir_fd=parent_fd)
        except FileExistsError as error:
            raise ArtifactValidationError(
                f"artifact leaf already exists: {path}"
            ) from error
        except OSError as error:
            raise ArtifactValidationError(
                f"artifact leaf cannot be opened with no-follow semantics: {path}"
            ) from error
        with os.fdopen(descriptor, "w+b") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent_fd)


@contextmanager
def _binary_reader(root: Path, path: PurePosixPath) -> Iterator[BinaryIO]:
    with _parent_descriptor(root, path, create=False) as (parent_fd, leaf):
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        except OSError as error:
            raise ArtifactValidationError(
                f"artifact leaf cannot be opened with no-follow semantics: {path}"
            ) from error
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ArtifactValidationError(
                f"artifact leaf is not a regular file: {path}"
            )
        with os.fdopen(descriptor, "rb") as stream:
            yield stream


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def write_bytes_exclusive(root: Path, relative_path: str, payload: bytes) -> None:
    """Write one immutable artifact leaf through a no-follow descriptor chain."""
    path = _relative_artifact_path(relative_path)
    with _exclusive_binary_writer(root, path) as stream:
        stream.write(payload)


def read_bytes(root: Path, relative_path: str) -> bytes:
    """Read one regular artifact leaf once through a no-follow descriptor chain."""
    path = _relative_artifact_path(relative_path)
    with _binary_reader(root, path) as stream:
        return stream.read()


def write_array(root: Path, relative_path: str, values: np.ndarray) -> ArrayReference:
    """Write a deterministic NPY v2 sidecar beneath ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    path = _relative_npy_path(relative_path)
    _contained_path(root, relative_path)
    array = _canonical_array(values)
    with _exclusive_binary_writer(root, path) as stream:
        np.lib.format.write_array(stream, array, version=(2, 0), allow_pickle=False)
        stream.flush()
        stream.seek(0)
        sha256 = _sha256_stream(stream)
    return ArrayReference(
        path=relative_path,
        dtype=array.dtype.str,
        shape=tuple(array.shape),
        order="C",
        sha256=sha256,
    )


def read_array(root: Path, reference: ArrayReference) -> np.ndarray:
    """Validate and load one canonical sidecar without pickle support."""
    path = _relative_npy_path(reference.path)
    _contained_path(root, reference.path)
    with _binary_reader(root, path) as stream:
        if _sha256_stream(stream) != reference.sha256:
            raise ArtifactValidationError(
                f"array sidecar SHA-256 mismatch: {reference.path}"
            )
        stream.seek(0)
        array = np.load(stream, allow_pickle=False)
    if (
        array.dtype.str != reference.dtype
        or tuple(array.shape) != reference.shape
        or not array.flags.c_contiguous
        or reference.order != "C"
    ):
        raise ArtifactValidationError(
            f"array sidecar metadata mismatch: {reference.path}"
        )
    return array
