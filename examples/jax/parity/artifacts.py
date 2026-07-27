"""Canonical, hash-bound parity artifact serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from examples.jax.parity.contracts import ArrayReference


class ArtifactValidationError(ValueError):
    """A parity artifact violates its containment or integrity contract."""


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
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative_path
        or path.suffix != ".npy"
    ):
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(root: Path, relative_path: str, values: np.ndarray) -> ArrayReference:
    """Write a deterministic NPY v2 sidecar beneath ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    target = _contained_path(root, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = _canonical_array(values)
    with target.open("wb") as stream:
        np.lib.format.write_array(stream, array, version=(2, 0), allow_pickle=False)
    return ArrayReference(
        path=relative_path,
        dtype=array.dtype.str,
        shape=tuple(array.shape),
        order="C",
        sha256=_sha256(target),
    )


def read_array(root: Path, reference: ArrayReference) -> np.ndarray:
    """Validate and load one canonical sidecar without pickle support."""
    target = _contained_path(root, reference.path)
    if not target.is_file():
        raise ArtifactValidationError(f"array sidecar does not exist: {reference.path}")
    if _sha256(target) != reference.sha256:
        raise ArtifactValidationError(
            f"array sidecar SHA-256 mismatch: {reference.path}"
        )
    with target.open("rb") as stream:
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
