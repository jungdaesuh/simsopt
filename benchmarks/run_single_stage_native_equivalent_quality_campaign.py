"""Run the supervised NEQ-GNTR1 native-equivalent-quality GPU campaign.

The parent owns immutable source publication, exact-process GPU monitoring,
timeouts, and artifact publication.  Each child owns one pristine compile,
synchronized timed loop, and post-timing finalization/audit.  Receipt-schema
binding is intentionally confined to :func:`build_sample_receipt`.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, Never, Protocol

_DIAG2_CHILD_ENV: Final = "SIMSOPT_NEQ_DIAG2_CHILD"
_DIAG4_CHILD_ENV: Final = "SIMSOPT_NEQ_DIAG4_CHILD"
_DIAG5_CHILD_ENV: Final = "SIMSOPT_NEQ_DIAG5_CHILD"
_DIAG4_PROCESS_STARTED_ENV: Final = "SIMSOPT_NEQ_DIAG4_PROCESS_STARTED_MONOTONIC_NS"
_DIAG5_NATIVE_PATH_ENV: Final = "SIMSOPT_NEQ_DIAG5_NATIVE_EXTENSION_PATH"
_DIAG5_NATIVE_SHA256_ENV: Final = "SIMSOPT_NEQ_DIAG5_NATIVE_EXTENSION_SHA256"
_DIAG5_NATIVE_SIZE_ENV: Final = "SIMSOPT_NEQ_DIAG5_NATIVE_EXTENSION_SIZE_BYTES"
_DIAG5_NATIVE_LINK_COUNT_ENV: Final = "SIMSOPT_NEQ_DIAG5_NATIVE_EXTENSION_LINK_COUNT"
_DIAG4_BOOTSTRAP_AUTHORITY_SCHEMA: Final = (
    "single-stage-neq-gntr3-diag4-authorization-v1"
)
_DIAG4_BOOTSTRAP_AUTHORITY_RELATIVE_PATH: Final = (
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_"
    "iterative_retraction_authorization.json"
)
_DIAG4_BOOTSTRAP_CPU_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag4-cpu-qualification-20260811T214932Z"
)
_DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_PATH: Final = (
    "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json"
)
_DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_SCHEMA: Final = (
    "single-stage-neq-gntr3-execution-source-authority-v1"
)
_DIAG4_BOOTSTRAP_CPU_MANIFEST_SCHEMA: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-manifest-v1"
)
_DIAG4_BOOTSTRAP_CPU_SCIENCE_SCHEMA: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-v1"
)
_DIAG4_BOOTSTRAP_SOURCE_MANIFEST_SCHEMA: Final = (
    "single-stage-fullspace-source-manifest-v1"
)
_DIAG4_BOOTSTRAP_ROUTE: Final = "NEQ-GNTR3-DIAG4"
_DIAG4_BOOTSTRAP_NUMERICAL_ROUTE: Final = "NEQ-GNTR3"
_DIAG4_BOOTSTRAP_SCIENCE_SCHEMA: Final = (
    "single-stage-neq-gntr3-trace-free-diagnostic-v1"
)
_DIAG4_BOOTSTRAP_PLAN_PREFIX_SHA256: Final = (
    "987dd67227431a90dd851d4d8ab78f639f9964c57de5ac093fc15f5aac504e5c"
)
_DIAG4_BOOTSTRAP_EXECUTION_ENTRY_COUNT: Final = 603
_DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH: Final = (
    "benchmarks/run_single_stage_native_equivalent_quality_campaign.py"
)
_DIAG4_BOOTSTRAP_PREQUALIFICATION_PLAN_PATH: Final = "control/prequalification-plan.md"
_DIAG5_BOOTSTRAP_AUTHORITY_SCHEMA: Final = (
    "single-stage-neq-gntr3-diag5-authorization-v1"
)
_DIAG5_BOOTSTRAP_AUTHORITY_RELATIVE_PATH: Final = (
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag5_"
    "native_binding_recovery_authorization.json"
)
_DIAG5_BOOTSTRAP_CPU_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag5-cpu-qualification-20260812T110000Z"
)
_DIAG5_BOOTSTRAP_GPU_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-rtx5090-20260812T030000Z"
)
_DIAG5_BOOTSTRAP_GPU_STAGING_ROOT: Final = Path(
    f"{_DIAG5_BOOTSTRAP_GPU_ROOT}.partial-claim"
)
_DIAG5_BOOTSTRAP_GPU_ROLLBACK_ROOT: Final = Path(
    f"{_DIAG5_BOOTSTRAP_GPU_ROOT}.partial-rollback"
)
_DIAG5_BOOTSTRAP_CONSUMPTION_MARKER: Final = (
    _DIAG5_BOOTSTRAP_GPU_ROOT.parent
    / f".{_DIAG5_BOOTSTRAP_GPU_ROOT.name}.diag5-authority-consumed.json"
)
_DIAG5_BOOTSTRAP_CPU_MANIFEST_SCHEMA: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-manifest-v2"
)
_DIAG5_BOOTSTRAP_CPU_SCIENCE_SCHEMA: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-v2"
)
_DIAG5_BOOTSTRAP_ROUTE: Final = "NEQ-GNTR3-DIAG5"
_DIAG5_BOOTSTRAP_PLAN_SHA256: Final = (
    "24300e9742bcbb14b3fc3e2cceab37dedc310410290a13370a11d21ef749ec7a"
)
_DIAG5_BOOTSTRAP_COMPLETED_PLAN_SHA256: Final = (
    "ce244ac37bb437ea022a4b73e62bf49f8d4d5cf88b610ad2146e895f3471ce1c"
)
_DIAG5_BOOTSTRAP_SCIENCE_SCHEMA: Final = (
    "single-stage-neq-gntr3-trace-free-diagnostic-v2"
)
_DIAG5_BOOTSTRAP_EXECUTION_ENTRY_COUNT: Final = 638
_DIAG5_BOOTSTRAP_NATIVE_COPY_PATH: Final = (
    "native/simsoptpp.cpython-311-x86_64-linux-gnu.so"
)


@dataclass(frozen=True, slots=True)
class _Diag4BootstrapBinding:
    path: Path
    descriptor: int
    device: int
    inode: int
    size_bytes: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class _Diag5BootstrapNativeBinding:
    path: Path
    descriptor: int
    device: int
    inode: int
    size_bytes: int
    link_count: int
    sha256: str


_diag5_retained_bootstrap_bindings: list[_Diag4BootstrapBinding] = []
_diag5_retained_native_bindings: list[_Diag5BootstrapNativeBinding] = []


_PROTECTED_CLI_OPTIONS: Final = (
    "--snapshot-child",
    "--preflight-child",
    "--diagnostic-child",
    "--preflight-only",
    "--diagnostic-only",
    "--diagnostic-successor-authority",
)


def _protected_cli_option(argument: str) -> str | None:
    if not argument.startswith("--"):
        return None
    option = argument.partition("=")[0]
    if option in _PROTECTED_CLI_OPTIONS:
        return option
    if any(candidate.startswith(option) for candidate in _PROTECTED_CLI_OPTIONS):
        raise RuntimeError(f"abbreviated protected option is forbidden: {option}")
    return None


_protected_cli_arguments = tuple(
    option
    for argument in sys.argv[1:]
    if (option := _protected_cli_option(argument)) is not None
)


def _diag4_bootstrap_canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _diag4_bootstrap_json_bytes(payload: bytes, context: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{context} contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not canonical JSON") from error
    if _diag4_bootstrap_canonical_json_bytes(value) != payload:
        raise RuntimeError(f"{context} is not canonical JSON")
    return value


def _diag4_bootstrap_mapping(
    value: object, expected_keys: frozenset[str], context: str
) -> dict[str, object]:
    if not isinstance(value, dict) or frozenset(value) != expected_keys:
        raise RuntimeError(f"{context} fields differ")
    return value


def _diag4_bootstrap_relative_path(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} is not a string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or "." in path.parts
        or ".." in path.parts
    ):
        raise RuntimeError(f"{context} is not canonical relative path")
    return value


def _diag4_bootstrap_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{context} is not a SHA-256")
    return value


def _diag4_bootstrap_descriptor_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _diag4_bootstrap_regular_bytes(
    path: Path,
    context: str,
    bindings: list[_Diag4BootstrapBinding],
) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        payload = _diag4_bootstrap_descriptor_bytes(descriptor)
        after = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino, opened.st_size)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != (after.st_dev, after.st_ino, after.st_size)
            or identity != (bound.st_dev, bound.st_ino, bound.st_size)
            or len(payload) != opened.st_size
        ):
            raise RuntimeError(f"{context} descriptor binding differs")
        bindings.append(
            _Diag4BootstrapBinding(
                path=path,
                descriptor=descriptor,
                device=opened.st_dev,
                inode=opened.st_ino,
                size_bytes=opened.st_size,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        return payload
    except BaseException:
        os.close(descriptor)
        raise


def _diag5_bootstrap_regular_bytes(
    path: Path,
    context: str,
    bindings: list[_Diag4BootstrapBinding],
) -> bytes:
    """Open, shared-lock, and retain one DIAG5 immutable regular leaf."""

    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        opened = os.fstat(descriptor)
        payload = _diag4_bootstrap_descriptor_bytes(descriptor)
        after = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino, opened.st_size)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != (after.st_dev, after.st_ino, after.st_size)
            or identity != (bound.st_dev, bound.st_ino, bound.st_size)
            or len(payload) != opened.st_size
        ):
            raise RuntimeError(f"{context} descriptor binding differs")
        bindings.append(
            _Diag4BootstrapBinding(
                path=path,
                descriptor=descriptor,
                device=opened.st_dev,
                inode=opened.st_ino,
                size_bytes=opened.st_size,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        return payload
    except BaseException:
        os.close(descriptor)
        raise


def _diag4_bootstrap_bind_directory(
    path: Path,
    context: str,
    bindings: list[_Diag4BootstrapBinding],
) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    )
    try:
        opened = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            bound.st_dev,
            bound.st_ino,
        ):
            raise RuntimeError(f"{context} descriptor binding differs")
        bindings.append(
            _Diag4BootstrapBinding(
                path=path,
                descriptor=descriptor,
                device=opened.st_dev,
                inode=opened.st_ino,
                size_bytes=opened.st_size,
                sha256=None,
            )
        )
    except BaseException:
        os.close(descriptor)
        raise


def _diag4_bootstrap_revalidate_bindings(
    bindings: Sequence[_Diag4BootstrapBinding],
) -> None:
    for binding in bindings:
        opened = os.fstat(binding.descriptor)
        bound = binding.path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            binding.device,
            binding.inode,
            binding.size_bytes,
        ) or (bound.st_dev, bound.st_ino, bound.st_size) != (
            binding.device,
            binding.inode,
            binding.size_bytes,
        ):
            raise RuntimeError(f"DIAG4 bootstrap binding drifted: {binding.path}")
        if (
            binding.sha256 is not None
            and hashlib.sha256(
                _diag4_bootstrap_descriptor_bytes(binding.descriptor)
            ).hexdigest()
            != binding.sha256
        ):
            raise RuntimeError(f"DIAG4 bootstrap bytes drifted: {binding.path}")


def _diag5_bootstrap_bind_native_extension(
    value: object,
    bindings: list[_Diag5BootstrapNativeBinding],
) -> None:
    native = _diag4_bootstrap_mapping(
        value,
        frozenset(
            {
                "gpu_native_extension_path",
                "native_extension_sha256",
                "native_extension_size_bytes",
                "gpu_native_extension_link_count",
                "gpu_native_extension_device",
                "gpu_native_extension_inode",
            }
        ),
        "DIAG5 GPU native binding",
    )
    path_text = native["gpu_native_extension_path"]
    if not isinstance(path_text, str):
        raise TypeError("DIAG5 GPU native path is not a string")
    path = Path(path_text).resolve(strict=True)
    if path_text != str(path):
        raise RuntimeError("DIAG5 GPU native path is not absolute and resolved")
    digest = _diag4_bootstrap_sha256(
        native["native_extension_sha256"], "DIAG5 GPU native SHA"
    )
    numeric = tuple(
        native[name]
        for name in (
            "native_extension_size_bytes",
            "gpu_native_extension_link_count",
            "gpu_native_extension_device",
            "gpu_native_extension_inode",
        )
    )
    if any(type(value) is not int or value <= 0 for value in numeric):
        raise RuntimeError("DIAG5 GPU native numeric identity differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        opened = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        payload = _diag4_bootstrap_descriptor_bytes(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_nlink)
            != (
                bound.st_dev,
                bound.st_ino,
                bound.st_size,
                bound.st_nlink,
            )
            or (opened.st_size, opened.st_nlink, opened.st_dev, opened.st_ino)
            != numeric
            or len(payload) != opened.st_size
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise RuntimeError("DIAG5 GPU native descriptor binding differs")
        bindings.append(
            _Diag5BootstrapNativeBinding(
                path,
                descriptor,
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_nlink,
                digest,
            )
        )
    except BaseException:
        os.close(descriptor)
        raise


def _diag5_bootstrap_revalidate_native_bindings(
    bindings: Sequence[_Diag5BootstrapNativeBinding],
) -> None:
    for binding in bindings:
        opened = os.fstat(binding.descriptor)
        bound = binding.path.stat(follow_symlinks=False)
        identity = (
            binding.device,
            binding.inode,
            binding.size_bytes,
            binding.link_count,
        )
        if (
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_nlink) != identity
            or (bound.st_dev, bound.st_ino, bound.st_size, bound.st_nlink) != identity
            or hashlib.sha256(
                _diag4_bootstrap_descriptor_bytes(binding.descriptor)
            ).hexdigest()
            != binding.sha256
        ):
            raise RuntimeError("DIAG5 GPU native bootstrap binding drifted")


def _release_diag5_bootstrap_bindings() -> None:
    """Release pre-import DIAG5 descriptors only after authority claim handoff."""

    _diag4_bootstrap_revalidate_bindings(_diag5_retained_bootstrap_bindings)
    _diag5_bootstrap_revalidate_native_bindings(_diag5_retained_native_bindings)
    for binding in reversed(_diag5_retained_native_bindings):
        fcntl.flock(binding.descriptor, fcntl.LOCK_UN)
        os.close(binding.descriptor)
    for binding in reversed(_diag5_retained_bootstrap_bindings):
        fcntl.flock(binding.descriptor, fcntl.LOCK_UN)
        os.close(binding.descriptor)
    _diag5_retained_native_bindings.clear()
    _diag5_retained_bootstrap_bindings.clear()


def _diag4_bootstrap_validate_cpu_artifact(
    authority: Mapping[str, object],
    bindings: list[_Diag4BootstrapBinding],
) -> Path:
    decisive = _diag4_bootstrap_mapping(
        authority["decisive_cpu_qualification"],
        frozenset(
            {
                "artifact_manifest_sha256",
                "command",
                "duration_seconds",
                "exit_code",
                "qualification_passed",
                "root",
                "run_count",
                "schema_version",
                "scientific_evidence_sha256",
                "scientific_outcome",
            }
        ),
        "DIAG4 decisive CPU qualification",
    )
    cpu_root = Path(str(decisive["root"]))
    if (
        cpu_root != _DIAG4_BOOTSTRAP_CPU_ROOT
        or decisive["schema_version"] != _DIAG4_BOOTSTRAP_CPU_SCIENCE_SCHEMA
        or decisive["exit_code"] != 0
        or decisive["qualification_passed"] is not True
        or decisive["run_count"] != 1
        or decisive["scientific_outcome"] != "QUALITY_HIT"
    ):
        raise RuntimeError("DIAG4 decisive CPU qualification identity differs")
    root_stat = cpu_root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or cpu_root.is_symlink()
        or stat.S_IMODE(root_stat.st_mode) != 0o555
    ):
        raise RuntimeError("DIAG4 decisive CPU qualification root is not sealed")
    _diag4_bootstrap_bind_directory(
        cpu_root, "DIAG4 decisive CPU qualification root", bindings
    )
    artifact_manifest_path = cpu_root / "artifact-manifest.json"
    artifact_manifest_bytes = _diag4_bootstrap_regular_bytes(
        artifact_manifest_path, "DIAG4 CPU artifact manifest", bindings
    )
    if (
        hashlib.sha256(artifact_manifest_bytes).hexdigest()
        != _diag4_bootstrap_sha256(
            decisive["artifact_manifest_sha256"], "DIAG4 CPU artifact manifest"
        )
        or stat.S_IMODE(artifact_manifest_path.stat().st_mode) != 0o444
    ):
        raise RuntimeError("DIAG4 CPU artifact manifest identity differs")
    artifact_manifest = _diag4_bootstrap_mapping(
        _diag4_bootstrap_json_bytes(
            artifact_manifest_bytes, "DIAG4 CPU artifact manifest"
        ),
        frozenset(
            {
                "directories",
                "execution_source_entries_sha256",
                "execution_source_manifest_sha256",
                "files",
                "schema_version",
            }
        ),
        "DIAG4 CPU artifact manifest",
    )
    if (
        artifact_manifest["schema_version"] != _DIAG4_BOOTSTRAP_CPU_MANIFEST_SCHEMA
        or artifact_manifest["execution_source_manifest_sha256"]
        != authority["execution_source_manifest_sha256"]
        or artifact_manifest["execution_source_entries_sha256"]
        != authority["execution_source_entries_sha256"]
    ):
        raise RuntimeError("DIAG4 CPU artifact manifest authority differs")
    raw_files = artifact_manifest["files"]
    raw_directories = artifact_manifest["directories"]
    if not isinstance(raw_files, list) or not isinstance(raw_directories, list):
        raise TypeError("DIAG4 CPU artifact manifest members differ")
    files: dict[str, tuple[str, int]] = {}
    for index, raw_file in enumerate(raw_files):
        entry = _diag4_bootstrap_mapping(
            raw_file,
            frozenset({"mode", "relative_path", "sha256", "size_bytes"}),
            f"DIAG4 CPU artifact file[{index}]",
        )
        relative = _diag4_bootstrap_relative_path(
            entry["relative_path"], f"DIAG4 CPU artifact file[{index}]"
        )
        digest = _diag4_bootstrap_sha256(
            entry["sha256"], f"DIAG4 CPU artifact file[{index}]"
        )
        size = entry["size_bytes"]
        if (
            entry["mode"] != "0444"
            or type(size) is not int
            or size < 0
            or relative in files
        ):
            raise RuntimeError("DIAG4 CPU artifact file entry differs")
        path = cpu_root.joinpath(*PurePosixPath(relative).parts)
        payload = _diag4_bootstrap_regular_bytes(
            path, f"DIAG4 CPU file {relative}", bindings
        )
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
            or stat.S_IMODE(path.stat().st_mode) != 0o444
        ):
            raise RuntimeError(f"DIAG4 CPU file differs: {relative}")
        files[relative] = (digest, size)
    if tuple(files) != tuple(sorted(files)):
        raise RuntimeError("DIAG4 CPU artifact file order differs")
    directories: set[str] = set()
    directory_order: list[str] = []
    for index, raw_directory in enumerate(raw_directories):
        entry = _diag4_bootstrap_mapping(
            raw_directory,
            frozenset({"mode", "relative_path"}),
            f"DIAG4 CPU artifact directory[{index}]",
        )
        relative = _diag4_bootstrap_relative_path(
            entry["relative_path"], f"DIAG4 CPU artifact directory[{index}]"
        )
        path = cpu_root.joinpath(*PurePosixPath(relative).parts)
        observed = path.stat(follow_symlinks=False)
        if (
            entry["mode"] != "0555"
            or relative in directories
            or path.is_symlink()
            or not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o555
        ):
            raise RuntimeError("DIAG4 CPU artifact directory entry differs")
        _diag4_bootstrap_bind_directory(
            path, f"DIAG4 CPU artifact directory {relative}", bindings
        )
        directories.add(relative)
        directory_order.append(relative)
    if tuple(directory_order) != tuple(sorted(directory_order)):
        raise RuntimeError("DIAG4 CPU artifact directory order differs")
    for path in cpu_root.rglob("*"):
        observed = path.stat(follow_symlinks=False)
        if path.is_symlink() or not (
            stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)
        ):
            raise RuntimeError("DIAG4 CPU artifact contains a special path")
    actual_files = {
        path.relative_to(cpu_root).as_posix()
        for path in cpu_root.rglob("*")
        if path.is_file()
    }
    actual_directories = {
        path.relative_to(cpu_root).as_posix()
        for path in cpu_root.rglob("*")
        if path.is_dir()
    }
    if (
        actual_files != set(files) | {"artifact-manifest.json"}
        or actual_directories != directories
    ):
        raise RuntimeError("DIAG4 CPU artifact membership differs")

    scientific_path = cpu_root / "scientific-evidence.json"
    scientific_bytes = _diag4_bootstrap_regular_bytes(
        scientific_path, "DIAG4 CPU scientific evidence", bindings
    )
    if hashlib.sha256(scientific_bytes).hexdigest() != _diag4_bootstrap_sha256(
        decisive["scientific_evidence_sha256"], "DIAG4 CPU scientific evidence"
    ):
        raise RuntimeError("DIAG4 CPU scientific evidence identity differs")
    scientific = _diag4_bootstrap_json_bytes(
        scientific_bytes, "DIAG4 CPU scientific evidence"
    )
    if not isinstance(scientific, dict) or (
        scientific.get("schema_version") != _DIAG4_BOOTSTRAP_CPU_SCIENCE_SCHEMA
        or scientific.get("qualification_passed") is not True
        or scientific.get("scientific_outcome") != "QUALITY_HIT"
        or scientific.get("execution_source_manifest_sha256")
        != authority["execution_source_manifest_sha256"]
        or scientific.get("execution_source_entries_sha256")
        != authority["execution_source_entries_sha256"]
        or scientific.get("native_extension_path") != authority["native_extension_path"]
        or scientific.get("native_extension_sha256")
        != authority["native_extension_sha256"]
        or scientific.get("native_extension_size_bytes")
        != authority["native_extension_size_bytes"]
    ):
        raise RuntimeError("DIAG4 CPU scientific evidence authority differs")

    source_root = cpu_root / "source-snapshot"
    source_manifest_path = source_root / "source-manifest.json"
    source_manifest_bytes = _diag4_bootstrap_regular_bytes(
        source_manifest_path, "DIAG4 CPU source manifest", bindings
    )
    source_manifest = _diag4_bootstrap_mapping(
        _diag4_bootstrap_json_bytes(source_manifest_bytes, "DIAG4 CPU source manifest"),
        frozenset({"entries", "schema_version", "worktree"}),
        "DIAG4 CPU source manifest",
    )
    if (
        source_manifest["schema_version"] != _DIAG4_BOOTSTRAP_SOURCE_MANIFEST_SCHEMA
        or hashlib.sha256(source_manifest_bytes).hexdigest()
        != scientific.get("source_manifest_sha256")
        or source_manifest["entries"] != scientific.get("source_manifest_entries")
    ):
        raise RuntimeError("DIAG4 CPU source manifest identity differs")

    execution_manifest_path = source_root.joinpath(
        *PurePosixPath(_DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_PATH).parts
    )
    execution_manifest_bytes = _diag4_bootstrap_regular_bytes(
        execution_manifest_path, "DIAG4 execution-source manifest", bindings
    )
    execution_manifest = _diag4_bootstrap_mapping(
        _diag4_bootstrap_json_bytes(
            execution_manifest_bytes, "DIAG4 execution-source manifest"
        ),
        frozenset({"entries", "entries_sha256", "schema_version"}),
        "DIAG4 execution-source manifest",
    )
    execution_entries = execution_manifest["entries"]
    if (
        execution_manifest["schema_version"]
        != _DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_SCHEMA
        or not isinstance(execution_entries, dict)
        or len(execution_entries) != _DIAG4_BOOTSTRAP_EXECUTION_ENTRY_COUNT
        or hashlib.sha256(
            _diag4_bootstrap_canonical_json_bytes(execution_entries)
        ).hexdigest()
        != execution_manifest["entries_sha256"]
        or execution_manifest["entries_sha256"]
        != authority["execution_source_entries_sha256"]
        or hashlib.sha256(execution_manifest_bytes).hexdigest()
        != authority["execution_source_manifest_sha256"]
    ):
        raise RuntimeError("DIAG4 execution-source manifest identity differs")
    raw_snapshot_entries = source_manifest["entries"]
    if not isinstance(raw_snapshot_entries, list):
        raise TypeError("DIAG4 CPU source entries differ")
    snapshot_entries: dict[str, tuple[str, int, str]] = {}
    for index, raw_entry in enumerate(raw_snapshot_entries):
        entry = _diag4_bootstrap_mapping(
            raw_entry,
            frozenset({"relative_path", "role", "sha256", "size_bytes"}),
            f"DIAG4 CPU source entry[{index}]",
        )
        relative = _diag4_bootstrap_relative_path(
            entry["relative_path"], f"DIAG4 CPU source entry[{index}]"
        )
        digest = _diag4_bootstrap_sha256(
            entry["sha256"], f"DIAG4 CPU source entry[{index}]"
        )
        size = entry["size_bytes"]
        role = entry["role"]
        if (
            type(size) is not int
            or size < 0
            or not isinstance(role, str)
            or relative in snapshot_entries
        ):
            raise RuntimeError("DIAG4 CPU source entry differs")
        path = source_root.joinpath(*PurePosixPath(relative).parts)
        payload = _diag4_bootstrap_regular_bytes(
            path, f"DIAG4 CPU source {relative}", bindings
        )
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
            or stat.S_IMODE(path.stat().st_mode) != 0o444
        ):
            raise RuntimeError(f"DIAG4 CPU source differs: {relative}")
        snapshot_entries[relative] = (digest, size, role)
    if tuple(snapshot_entries) != tuple(sorted(snapshot_entries)):
        raise RuntimeError("DIAG4 CPU source manifest order differs")
    expected_snapshot_entries: dict[str, tuple[str, int, str]] = {}
    for relative, raw_entry in execution_entries.items():
        relative = _diag4_bootstrap_relative_path(
            relative, "DIAG4 execution-source path"
        )
        entry = _diag4_bootstrap_mapping(
            raw_entry,
            frozenset({"sha256", "size_bytes"}),
            f"DIAG4 execution-source entry {relative}",
        )
        digest = _diag4_bootstrap_sha256(
            entry["sha256"], f"DIAG4 execution-source entry {relative}"
        )
        size = entry["size_bytes"]
        if type(size) is not int or size < 0:
            raise RuntimeError("DIAG4 execution-source entry size differs")
        role = (
            "test"
            if relative.startswith("tests/")
            else "benchmark"
            if relative.startswith("benchmarks/")
            else "execution_source"
        )
        expected_snapshot_entries[relative] = (digest, size, role)
    expected_snapshot_entries[_DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_PATH] = (
        str(authority["execution_source_manifest_sha256"]),
        len(execution_manifest_bytes),
        "execution_source_manifest",
    )
    plan_control = scientific.get("prequalification_plan_control")
    plan_control = _diag4_bootstrap_mapping(
        plan_control,
        frozenset(
            {
                "schema_version",
                "snapshot_relative_path",
                "source_relative_path",
                "sha256",
                "size_bytes",
                "plan_prefix_sha256",
            }
        ),
        "DIAG4 prequalification plan control",
    )
    plan_digest = _diag4_bootstrap_sha256(
        plan_control["sha256"], "DIAG4 prequalification plan control"
    )
    plan_size = plan_control["size_bytes"]
    if (
        plan_control["schema_version"]
        != "single-stage-neq-gntr3-prequalification-plan-control-v1"
        or plan_control["snapshot_relative_path"]
        != _DIAG4_BOOTSTRAP_PREQUALIFICATION_PLAN_PATH
        or plan_control["source_relative_path"]
        != "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md"
        or plan_control["plan_prefix_sha256"] != _DIAG4_BOOTSTRAP_PLAN_PREFIX_SHA256
        or type(plan_size) is not int
        or plan_size < 0
    ):
        raise RuntimeError("DIAG4 prequalification plan control differs")
    expected_snapshot_entries[_DIAG4_BOOTSTRAP_PREQUALIFICATION_PLAN_PATH] = (
        plan_digest,
        plan_size,
        "prequalification_plan",
    )
    native_relative = f"native/{Path(str(authority['native_extension_path'])).name}"
    expected_snapshot_entries[native_relative] = (
        str(authority["native_extension_sha256"]),
        int(authority["native_extension_size_bytes"]),
        "native_extension",
    )
    if snapshot_entries != expected_snapshot_entries:
        raise RuntimeError("DIAG4 CPU source snapshot differs from authority")
    actual_source_files = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual_source_files != set(snapshot_entries) | {"source-manifest.json"}:
        raise RuntimeError("DIAG4 CPU source snapshot membership differs")
    runner_entry = execution_entries.get(_DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH)
    if runner_entry is None:
        raise RuntimeError("DIAG4 sealed supervisor is absent")
    sealed_entry = source_root / _DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH
    if snapshot_entries[_DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH][:2] != (
        runner_entry["sha256"],
        runner_entry["size_bytes"],
    ):
        raise RuntimeError("DIAG4 sealed supervisor identity differs")
    return sealed_entry


def _diag5_bootstrap_validate_cpu_artifact(
    authority: Mapping[str, object],
    bindings: list[_Diag4BootstrapBinding],
) -> Path:
    """Validate the sealed DIAG5 CPU closure and return its supervisor entry."""

    decisive = _diag4_bootstrap_mapping(
        authority["decisive_cpu_qualification"],
        frozenset({"relative_path", "schema_version", "sha256", "size_bytes"}),
        "DIAG5 decisive CPU qualification",
    )
    if (
        decisive["relative_path"] != "scientific-evidence.json"
        or decisive["schema_version"] != _DIAG5_BOOTSTRAP_CPU_SCIENCE_SCHEMA
        or type(decisive["size_bytes"]) is not int
        or int(decisive["size_bytes"]) <= 0
    ):
        raise RuntimeError("DIAG5 decisive CPU qualification identity differs")
    decisive_sha256 = _diag4_bootstrap_sha256(
        decisive["sha256"], "DIAG5 decisive CPU qualification"
    )
    cpu_root = _DIAG5_BOOTSTRAP_CPU_ROOT
    root_metadata = cpu_root.stat(follow_symlinks=False)
    if (
        cpu_root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
    ):
        raise RuntimeError("DIAG5 decisive CPU qualification root is not sealed")
    _diag4_bootstrap_bind_directory(
        cpu_root, "DIAG5 decisive CPU qualification root", bindings
    )
    manifest_path = cpu_root / "artifact-manifest.json"
    manifest_bytes = _diag5_bootstrap_regular_bytes(
        manifest_path, "DIAG5 CPU artifact manifest", bindings
    )
    manifest = _diag4_bootstrap_mapping(
        _diag4_bootstrap_json_bytes(manifest_bytes, "DIAG5 CPU artifact manifest"),
        frozenset(
            {
                "directories",
                "execution_source_entries_sha256",
                "execution_source_manifest_sha256",
                "files",
                "schema_version",
            }
        ),
        "DIAG5 CPU artifact manifest",
    )
    if (
        manifest["schema_version"] != _DIAG5_BOOTSTRAP_CPU_MANIFEST_SCHEMA
        or manifest["execution_source_manifest_sha256"]
        != authority["execution_source_manifest_sha256"]
        or manifest["execution_source_entries_sha256"]
        != authority["execution_source_entries_sha256"]
    ):
        raise RuntimeError("DIAG5 CPU artifact manifest authority differs")
    raw_files = manifest["files"]
    raw_directories = manifest["directories"]
    if not isinstance(raw_files, list) or not isinstance(raw_directories, list):
        raise TypeError("DIAG5 CPU artifact manifest members differ")
    files: dict[str, tuple[str, int]] = {}
    for index, raw_file in enumerate(raw_files):
        entry = _diag4_bootstrap_mapping(
            raw_file,
            frozenset({"mode", "relative_path", "sha256", "size_bytes"}),
            f"DIAG5 CPU artifact file[{index}]",
        )
        relative = _diag4_bootstrap_relative_path(
            entry["relative_path"], f"DIAG5 CPU artifact file[{index}]"
        )
        digest = _diag4_bootstrap_sha256(
            entry["sha256"], f"DIAG5 CPU artifact file[{index}]"
        )
        size = entry["size_bytes"]
        path = cpu_root.joinpath(*PurePosixPath(relative).parts)
        payload = _diag5_bootstrap_regular_bytes(
            path, f"DIAG5 CPU artifact file {relative}", bindings
        )
        if (
            entry["mode"] != "0444"
            or type(size) is not int
            or size < 0
            or relative in files
            or len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
            or stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o444
        ):
            raise RuntimeError(f"DIAG5 CPU artifact file differs: {relative}")
        files[relative] = (digest, size)
    if tuple(files) != tuple(sorted(files)):
        raise RuntimeError("DIAG5 CPU artifact file order differs")
    directories: set[str] = set()
    directory_order: list[str] = []
    for index, raw_directory in enumerate(raw_directories):
        entry = _diag4_bootstrap_mapping(
            raw_directory,
            frozenset({"mode", "relative_path"}),
            f"DIAG5 CPU artifact directory[{index}]",
        )
        relative = _diag4_bootstrap_relative_path(
            entry["relative_path"], f"DIAG5 CPU artifact directory[{index}]"
        )
        path = cpu_root.joinpath(*PurePosixPath(relative).parts)
        metadata = path.stat(follow_symlinks=False)
        if (
            entry["mode"] != "0555"
            or relative in directories
            or path.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o555
        ):
            raise RuntimeError(f"DIAG5 CPU artifact directory differs: {relative}")
        _diag4_bootstrap_bind_directory(
            path, f"DIAG5 CPU artifact directory {relative}", bindings
        )
        directories.add(relative)
        directory_order.append(relative)
    if tuple(directory_order) != tuple(sorted(directory_order)):
        raise RuntimeError("DIAG5 CPU artifact directory order differs")
    for path in cpu_root.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        if path.is_symlink() or not (
            stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
        ):
            raise RuntimeError("DIAG5 CPU artifact contains a special path")
    actual_files = {
        path.relative_to(cpu_root).as_posix()
        for path in cpu_root.rglob("*")
        if path.is_file()
    }
    actual_directories = {
        path.relative_to(cpu_root).as_posix()
        for path in cpu_root.rglob("*")
        if path.is_dir()
    }
    if (
        actual_files != set(files) | {"artifact-manifest.json"}
        or actual_directories != directories
    ):
        raise RuntimeError("DIAG5 CPU artifact membership differs")

    scientific_path = cpu_root / "scientific-evidence.json"
    scientific_bytes = _diag5_bootstrap_regular_bytes(
        scientific_path, "DIAG5 CPU scientific evidence", bindings
    )
    if (
        len(scientific_bytes) != decisive["size_bytes"]
        or hashlib.sha256(scientific_bytes).hexdigest() != decisive_sha256
    ):
        raise RuntimeError("DIAG5 CPU scientific evidence identity differs")
    scientific = _diag4_bootstrap_json_bytes(
        scientific_bytes, "DIAG5 CPU scientific evidence"
    )
    if not isinstance(scientific, dict):
        raise TypeError("DIAG5 CPU scientific evidence is not an object")
    cpu_native = _diag4_bootstrap_mapping(
        scientific.get("cpu_native_binding"),
        frozenset(
            {
                "cpu_native_extension_path",
                "native_extension_sha256",
                "native_extension_size_bytes",
                "cpu_native_extension_link_count",
                "cpu_native_extension_device",
                "cpu_native_extension_inode",
            }
        ),
        "DIAG5 CPU native binding",
    )
    authority_cpu_native = _diag4_bootstrap_mapping(
        authority["cpu_native_binding"],
        frozenset(cpu_native),
        "DIAG5 authority CPU native binding",
    )
    authority_gpu_native = _diag4_bootstrap_mapping(
        authority["gpu_native_binding"],
        frozenset(
            {
                "gpu_native_extension_path",
                "native_extension_sha256",
                "native_extension_size_bytes",
                "gpu_native_extension_link_count",
                "gpu_native_extension_device",
                "gpu_native_extension_inode",
            }
        ),
        "DIAG5 authority GPU native binding",
    )
    if (
        scientific.get("schema_version") != _DIAG5_BOOTSTRAP_CPU_SCIENCE_SCHEMA
        or scientific.get("qualification_passed") is not True
        or scientific.get("scientific_outcome") != "QUALITY_HIT"
        or scientific.get("execution_source_manifest_sha256")
        != authority["execution_source_manifest_sha256"]
        or scientific.get("execution_source_entries_sha256")
        != authority["execution_source_entries_sha256"]
        or cpu_native != authority_cpu_native
        or (
            cpu_native["native_extension_sha256"],
            cpu_native["native_extension_size_bytes"],
        )
        != (
            authority_gpu_native["native_extension_sha256"],
            authority_gpu_native["native_extension_size_bytes"],
        )
    ):
        raise RuntimeError("DIAG5 CPU scientific evidence authority differs")

    source_root = cpu_root / "source-snapshot"
    source_manifest_path = source_root / "source-manifest.json"
    source_manifest_bytes = _diag5_bootstrap_regular_bytes(
        source_manifest_path, "DIAG5 CPU source manifest", bindings
    )
    source_manifest = _diag4_bootstrap_mapping(
        _diag4_bootstrap_json_bytes(source_manifest_bytes, "DIAG5 CPU source manifest"),
        frozenset({"entries", "schema_version", "worktree"}),
        "DIAG5 CPU source manifest",
    )
    raw_entries = source_manifest["entries"]
    if (
        source_manifest["schema_version"] != _DIAG4_BOOTSTRAP_SOURCE_MANIFEST_SCHEMA
        or not isinstance(raw_entries, list)
        or len(raw_entries) != _DIAG5_BOOTSTRAP_EXECUTION_ENTRY_COUNT + 3
        or hashlib.sha256(source_manifest_bytes).hexdigest()
        != scientific.get("source_manifest_sha256")
        or raw_entries != scientific.get("source_manifest_entries")
    ):
        raise RuntimeError("DIAG5 CPU source manifest identity differs")
    snapshot_entries: dict[str, tuple[str, int, str]] = {}
    for index, raw_entry in enumerate(raw_entries):
        entry = _diag4_bootstrap_mapping(
            raw_entry,
            frozenset({"relative_path", "role", "sha256", "size_bytes"}),
            f"DIAG5 CPU source entry[{index}]",
        )
        relative = _diag4_bootstrap_relative_path(
            entry["relative_path"], f"DIAG5 CPU source entry[{index}]"
        )
        digest = _diag4_bootstrap_sha256(
            entry["sha256"], f"DIAG5 CPU source entry[{index}]"
        )
        size = entry["size_bytes"]
        role = entry["role"]
        if (
            type(size) is not int
            or size < 0
            or not isinstance(role, str)
            or relative in snapshot_entries
        ):
            raise RuntimeError("DIAG5 CPU source entry differs")
        source_path = source_root.joinpath(*PurePosixPath(relative).parts)
        payload = _diag5_bootstrap_regular_bytes(
            source_path, f"DIAG5 CPU source {relative}", bindings
        )
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
            or stat.S_IMODE(source_path.stat(follow_symlinks=False).st_mode) != 0o444
        ):
            raise RuntimeError(f"DIAG5 CPU source differs: {relative}")
        snapshot_entries[relative] = (digest, size, role)
    if tuple(snapshot_entries) != tuple(sorted(snapshot_entries)):
        raise RuntimeError("DIAG5 CPU source manifest order differs")

    execution_manifest_path = source_root.joinpath(
        *PurePosixPath(_DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_PATH).parts
    )
    execution_manifest_bytes = _diag5_bootstrap_regular_bytes(
        execution_manifest_path, "DIAG5 execution-source manifest", bindings
    )
    execution_manifest = _diag4_bootstrap_mapping(
        _diag4_bootstrap_json_bytes(
            execution_manifest_bytes, "DIAG5 execution-source manifest"
        ),
        frozenset({"entries", "entries_sha256", "schema_version"}),
        "DIAG5 execution-source manifest",
    )
    execution_entries = execution_manifest["entries"]
    if (
        execution_manifest["schema_version"]
        != _DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_SCHEMA
        or not isinstance(execution_entries, dict)
        or len(execution_entries) != _DIAG5_BOOTSTRAP_EXECUTION_ENTRY_COUNT
        or hashlib.sha256(execution_manifest_bytes).hexdigest()
        != authority["execution_source_manifest_sha256"]
        or hashlib.sha256(
            _diag4_bootstrap_canonical_json_bytes(execution_entries)
        ).hexdigest()
        != authority["execution_source_entries_sha256"]
    ):
        raise RuntimeError("DIAG5 execution-source manifest identity differs")
    expected_snapshot: dict[str, tuple[str, int, str]] = {}
    for relative, raw_entry in execution_entries.items():
        canonical_relative = _diag4_bootstrap_relative_path(
            relative, "DIAG5 execution-source path"
        )
        entry = _diag4_bootstrap_mapping(
            raw_entry,
            frozenset({"sha256", "size_bytes"}),
            f"DIAG5 execution-source entry {canonical_relative}",
        )
        digest = _diag4_bootstrap_sha256(
            entry["sha256"], f"DIAG5 execution-source entry {canonical_relative}"
        )
        size = entry["size_bytes"]
        if type(size) is not int or size < 0:
            raise RuntimeError("DIAG5 execution-source entry size differs")
        role = (
            "test"
            if canonical_relative.startswith("tests/")
            else "benchmark"
            if canonical_relative.startswith("benchmarks/")
            else "execution_source"
        )
        expected_snapshot[canonical_relative] = (digest, size, role)
    expected_snapshot[_DIAG4_BOOTSTRAP_EXECUTION_MANIFEST_PATH] = (
        str(authority["execution_source_manifest_sha256"]),
        len(execution_manifest_bytes),
        "execution_source_manifest",
    )
    plan_control = _diag4_bootstrap_mapping(
        scientific.get("prequalification_plan_control"),
        frozenset(
            {
                "schema_version",
                "snapshot_relative_path",
                "source_relative_path",
                "sha256",
                "size_bytes",
                "plan_prefix_sha256",
            }
        ),
        "DIAG5 prequalification plan control",
    )
    plan_relative = _diag4_bootstrap_relative_path(
        plan_control["snapshot_relative_path"],
        "DIAG5 prequalification plan snapshot path",
    )
    plan_digest = _diag4_bootstrap_sha256(
        plan_control["sha256"], "DIAG5 prequalification plan SHA"
    )
    if (
        plan_control["schema_version"]
        != "single-stage-neq-gntr3-prequalification-plan-control-v2"
        or plan_relative != _DIAG4_BOOTSTRAP_PREQUALIFICATION_PLAN_PATH
        or plan_control["source_relative_path"]
        != "docs/single_stage_jax_gpu_native_equivalent_quality_diag5_native_binding_recovery_plan.md"
        or plan_control["plan_prefix_sha256"] != _DIAG5_BOOTSTRAP_PLAN_SHA256
        or type(plan_control["size_bytes"]) is not int
        or plan_control["size_bytes"] <= 0
    ):
        raise RuntimeError("DIAG5 prequalification plan control differs")
    expected_snapshot[plan_relative] = (
        plan_digest,
        int(plan_control["size_bytes"]),
        "prequalification_plan",
    )
    expected_snapshot[_DIAG5_BOOTSTRAP_NATIVE_COPY_PATH] = (
        str(cpu_native["native_extension_sha256"]),
        int(cpu_native["native_extension_size_bytes"]),
        "native_extension",
    )
    if snapshot_entries != expected_snapshot:
        raise RuntimeError("DIAG5 CPU source snapshot differs from authority")
    for path in source_root.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        if path.is_symlink() or not (
            stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
        ):
            raise RuntimeError("DIAG5 CPU source snapshot contains a special path")
        expected_mode = 0o444 if stat.S_ISREG(metadata.st_mode) else 0o555
        if stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise RuntimeError("DIAG5 CPU source snapshot mode differs")
    actual_source_files = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual_source_files != set(snapshot_entries) | {"source-manifest.json"}:
        raise RuntimeError("DIAG5 CPU source snapshot membership differs")
    predecessor = _diag4_bootstrap_mapping(
        scientific.get("predecessor_postmortem"),
        frozenset({"relative_path", "schema_version", "sha256", "size_bytes"}),
        "DIAG5 predecessor postmortem",
    )
    authority_predecessor = _diag4_bootstrap_mapping(
        authority["predecessor_postmortem"],
        frozenset(predecessor),
        "DIAG5 authority predecessor postmortem",
    )
    if (
        predecessor != authority_predecessor
        or predecessor["relative_path"] != "control/predecessor-postmortem.json"
        or predecessor["schema_version"]
        != "single-stage-neq-gntr3-diag4-independent-postmortem-v1"
        or files.get(str(predecessor["relative_path"]))
        != (predecessor["sha256"], predecessor["size_bytes"])
    ):
        raise RuntimeError("DIAG5 predecessor postmortem reference differs")
    runner_path = source_root / _DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH
    runner_entry = next(
        (
            entry
            for entry in raw_entries
            if isinstance(entry, dict)
            and entry.get("relative_path") == _DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH
        ),
        None,
    )
    if runner_entry is None:
        raise RuntimeError("DIAG5 sealed supervisor is absent")
    runner_bytes = _diag5_bootstrap_regular_bytes(
        runner_path, "DIAG5 sealed supervisor", bindings
    )
    if (
        runner_entry.get("role") != "benchmark"
        or runner_entry.get("size_bytes") != len(runner_bytes)
        or runner_entry.get("sha256") != hashlib.sha256(runner_bytes).hexdigest()
    ):
        raise RuntimeError("DIAG5 sealed supervisor identity differs")
    native_entry = next(
        (
            entry
            for entry in raw_entries
            if isinstance(entry, dict)
            and entry.get("relative_path") == _DIAG5_BOOTSTRAP_NATIVE_COPY_PATH
        ),
        None,
    )
    if (
        native_entry is None
        or native_entry.get("role") != "native_extension"
        or native_entry.get("sha256") != cpu_native["native_extension_sha256"]
        or native_entry.get("size_bytes") != cpu_native["native_extension_size_bytes"]
    ):
        raise RuntimeError("DIAG5 copied native evidence differs")
    return runner_path


def _diag4_bootstrap_authority_arguments(argv: Sequence[str]) -> tuple[Path, ...]:
    option = "--diagnostic-successor-authority"
    values: list[str] = []
    for index, argument in enumerate(argv):
        if argument == option:
            if index + 1 < len(argv):
                values.append(argv[index + 1])
        elif argument.startswith(f"{option}="):
            values.append(argument.partition("=")[2])
    return tuple(Path(value) for value in values if value)


def _diag4_preimport_bootstrap_impl(
    *,
    argv: Sequence[str],
    current_entry: Path,
    environment: Mapping[str, str],
    execve: Callable[[str, Sequence[str], Mapping[str, str]], Never],
    bindings: list[_Diag4BootstrapBinding],
) -> None:
    authority_paths = _diag4_bootstrap_authority_arguments(argv)
    diag4_named_paths = tuple(
        path
        for path in authority_paths
        if path.as_posix().endswith(_DIAG4_BOOTSTRAP_AUTHORITY_RELATIVE_PATH)
    )
    if not diag4_named_paths:
        return
    if len(authority_paths) != 1 or len(diag4_named_paths) != 1:
        raise RuntimeError("DIAG4 authority option must occur exactly once")
    authority_path = diag4_named_paths[0]
    if not authority_path.is_file():
        raise RuntimeError("DIAG4 authority is absent")
    authority_bytes = authority_path.read_bytes()
    try:
        candidate = _diag4_bootstrap_json_bytes(authority_bytes, "DIAG4 authority")
    except RuntimeError as error:
        raise RuntimeError("DIAG4 authority is invalid") from error
    if (
        not isinstance(candidate, dict)
        or candidate.get("schema_version") != _DIAG4_BOOTSTRAP_AUTHORITY_SCHEMA
    ):
        raise RuntimeError("DIAG4 authority schema differs")
    if (
        _diag4_bootstrap_regular_bytes(authority_path, "DIAG4 authority", bindings)
        != authority_bytes
    ):
        raise RuntimeError("DIAG4 authority changed during bootstrap")
    authority = _diag4_bootstrap_mapping(
        candidate,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "scientific_evidence_schema",
                "plan_prefix_sha256",
                "completed_plan_sha256",
                "qualification_record_sha256",
                "qualified_files",
                "qualified_files_sha256",
                "frozen_numerical_entries",
                "frozen_numerical_entries_sha256",
                "execution_source_manifest_sha256",
                "execution_source_entries_sha256",
                "historical_cpu20",
                "decisive_cpu_qualification",
                "native_extension_path",
                "native_extension_sha256",
                "native_extension_size_bytes",
                "native_reference_manifest_sha256",
                "consumed_diag3",
                "numerical_identity",
                "execution_policy",
                "launch",
            }
        ),
        "DIAG4 authority",
    )
    authority_absolute = authority_path.absolute()
    repository = authority_absolute.parents[1]
    if authority_absolute != repository / _DIAG4_BOOTSTRAP_AUTHORITY_RELATIVE_PATH:
        raise RuntimeError("DIAG4 authority path differs")
    if (
        authority["route"] != _DIAG4_BOOTSTRAP_ROUTE
        or authority["numerical_route"] != _DIAG4_BOOTSTRAP_NUMERICAL_ROUTE
        or authority["scientific_evidence_schema"] != _DIAG4_BOOTSTRAP_SCIENCE_SCHEMA
        or authority["plan_prefix_sha256"] != _DIAG4_BOOTSTRAP_PLAN_PREFIX_SHA256
    ):
        raise RuntimeError("DIAG4 authority identity differs")
    for name in (
        "completed_plan_sha256",
        "qualification_record_sha256",
        "qualified_files_sha256",
        "frozen_numerical_entries_sha256",
        "execution_source_manifest_sha256",
        "execution_source_entries_sha256",
        "native_extension_sha256",
        "native_reference_manifest_sha256",
    ):
        _diag4_bootstrap_sha256(authority[name], f"DIAG4 authority.{name}")
    native_size = authority["native_extension_size_bytes"]
    native_path = authority["native_extension_path"]
    if (
        type(native_size) is not int
        or native_size < 0
        or not isinstance(native_path, str)
        or not Path(native_path).is_absolute()
    ):
        raise RuntimeError("DIAG4 native extension size differs")
    launch = _diag4_bootstrap_mapping(
        authority["launch"],
        frozenset(
            {
                "output_root",
                "reference_root",
                "input_root",
                "interpreter",
                "gpu_uuid",
                "preflight_launches",
                "maximum_cold_launches",
                "warm_allowed",
                "retry_allowed",
            }
        ),
        "DIAG4 launch",
    )
    interpreter = Path(str(launch["interpreter"])).resolve(strict=True)
    sealed_entry = _diag4_bootstrap_validate_cpu_artifact(authority, bindings)
    _diag4_bootstrap_revalidate_bindings(bindings)
    current = current_entry.resolve(strict=True)
    sealed = sealed_entry.resolve(strict=True)
    if current == sealed:
        return
    child_environment = dict(environment)
    source_root = sealed.parents[1]
    child_environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "src"), str(source_root))
    )
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(source_root)
    execve(
        str(interpreter),
        (str(interpreter), "-B", str(sealed), *argv[1:]),
        child_environment,
    )
    raise RuntimeError("DIAG4 sealed-supervisor execve returned")


def _diag5_preimport_bootstrap_impl(
    *,
    argv: Sequence[str],
    current_entry: Path,
    environment: Mapping[str, str],
    execve: Callable[[str, Sequence[str], Mapping[str, str]], Never],
    bindings: list[_Diag4BootstrapBinding],
    native_bindings: list[_Diag5BootstrapNativeBinding],
) -> bool:
    """Re-exec one DIAG5 supervisor from the sealed CPU-qualified source copy."""

    authority_paths = _diag4_bootstrap_authority_arguments(argv)
    named = tuple(
        path
        for path in authority_paths
        if path.as_posix().endswith(_DIAG5_BOOTSTRAP_AUTHORITY_RELATIVE_PATH)
    )
    if not named:
        return False
    if len(authority_paths) != 1 or len(named) != 1:
        raise RuntimeError("DIAG5 authority option must occur exactly once")
    authority_path = named[0]
    authority_bytes = _diag5_bootstrap_regular_bytes(
        authority_path, "DIAG5 authority", bindings
    )
    authority = _diag4_bootstrap_mapping(
        _diag4_bootstrap_json_bytes(authority_bytes, "DIAG5 authority"),
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "scientific_evidence_schema",
                "plan_prefix_sha256",
                "completed_plan_sha256",
                "qualification_record_sha256",
                "qualified_files",
                "qualified_files_sha256",
                "frozen_numerical_entries",
                "frozen_numerical_entries_sha256",
                "execution_source_manifest_sha256",
                "execution_source_entries_sha256",
                "predecessor_postmortem",
                "predecessor_full_tree_sha256",
                "decisive_cpu_qualification",
                "pre_run_reviews",
                "pre_run_reviews_sha256",
                "post_run_reviews",
                "post_run_reviews_sha256",
                "cpu_native_binding",
                "gpu_native_binding",
                "native_reference",
                "input_bundle",
                "consumed_diag3",
                "numerical_identity",
                "interpreter",
                "roots",
                "gpu_uuid",
                "execution_policy",
                "launch",
            }
        ),
        "DIAG5 authority",
    )
    authority_absolute = authority_path.absolute()
    repository = authority_absolute.parents[1]
    if authority_absolute != repository / _DIAG5_BOOTSTRAP_AUTHORITY_RELATIVE_PATH:
        raise RuntimeError("DIAG5 authority path differs")
    if (
        authority["schema_version"] != _DIAG5_BOOTSTRAP_AUTHORITY_SCHEMA
        or authority["route"] != _DIAG5_BOOTSTRAP_ROUTE
        or authority["numerical_route"] != _DIAG4_BOOTSTRAP_NUMERICAL_ROUTE
        or authority["scientific_evidence_schema"] != _DIAG5_BOOTSTRAP_SCIENCE_SCHEMA
        or authority["plan_prefix_sha256"] != _DIAG5_BOOTSTRAP_PLAN_SHA256
        or authority["completed_plan_sha256"] != _DIAG5_BOOTSTRAP_COMPLETED_PLAN_SHA256
    ):
        raise RuntimeError("DIAG5 authority identity differs")
    for name in (
        "plan_prefix_sha256",
        "completed_plan_sha256",
        "qualification_record_sha256",
        "qualified_files_sha256",
        "frozen_numerical_entries_sha256",
        "execution_source_manifest_sha256",
        "execution_source_entries_sha256",
        "predecessor_full_tree_sha256",
        "pre_run_reviews_sha256",
        "post_run_reviews_sha256",
    ):
        _diag4_bootstrap_sha256(authority[name], f"DIAG5 authority.{name}")
    launch = _diag4_bootstrap_mapping(
        authority["launch"],
        frozenset({"preflight_exact", "cold_max", "warm_exact", "retry_allowed"}),
        "DIAG5 launch",
    )
    if launch != {
        "preflight_exact": 1,
        "cold_max": 1,
        "warm_exact": 0,
        "retry_allowed": False,
    }:
        raise RuntimeError("DIAG5 launch cardinality differs")
    roots = _diag4_bootstrap_mapping(
        authority["roots"],
        frozenset(
            {
                "cpu_qualification_root",
                "gpu_output_root",
                "gpu_staging_root",
                "gpu_rollback_root",
                "consumption_marker",
            }
        ),
        "DIAG5 roots",
    )
    if roots != {
        "cpu_qualification_root": str(_DIAG5_BOOTSTRAP_CPU_ROOT),
        "gpu_output_root": str(_DIAG5_BOOTSTRAP_GPU_ROOT),
        "gpu_staging_root": str(_DIAG5_BOOTSTRAP_GPU_STAGING_ROOT),
        "gpu_rollback_root": str(_DIAG5_BOOTSTRAP_GPU_ROLLBACK_ROOT),
        "consumption_marker": str(_DIAG5_BOOTSTRAP_CONSUMPTION_MARKER),
    }:
        raise RuntimeError("DIAG5 roots differ")
    interpreter_payload = _diag4_bootstrap_mapping(
        authority["interpreter"],
        frozenset({"absolute_path", "sha256", "size_bytes"}),
        "DIAG5 interpreter",
    )
    interpreter = Path(str(interpreter_payload["absolute_path"])).resolve(strict=True)
    interpreter_bytes = _diag5_bootstrap_regular_bytes(
        interpreter, "DIAG5 interpreter", bindings
    )
    if len(interpreter_bytes) != interpreter_payload["size_bytes"] or hashlib.sha256(
        interpreter_bytes
    ).hexdigest() != _diag4_bootstrap_sha256(
        interpreter_payload["sha256"], "DIAG5 interpreter SHA"
    ):
        raise RuntimeError("DIAG5 interpreter identity differs")
    _diag5_bootstrap_bind_native_extension(
        authority["gpu_native_binding"], native_bindings
    )
    sealed_entry = _diag5_bootstrap_validate_cpu_artifact(authority, bindings)
    _diag4_bootstrap_revalidate_bindings(bindings)
    _diag5_bootstrap_revalidate_native_bindings(native_bindings)
    current = current_entry.resolve(strict=True)
    sealed = sealed_entry.resolve(strict=True)
    if current == sealed:
        return True
    child_environment = dict(environment)
    source_root = sealed.parents[1]
    child_environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "src"), str(source_root))
    )
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(source_root)
    execve(
        str(interpreter),
        (str(interpreter), "-B", str(sealed), *argv[1:]),
        child_environment,
    )
    raise RuntimeError("DIAG5 sealed-supervisor execve returned")


def _diag4_preimport_bootstrap(
    *,
    argv: Sequence[str],
    current_entry: Path,
    environment: Mapping[str, str],
    execve: Callable[[str, Sequence[str], Mapping[str, str]], Never] = os.execve,
) -> None:
    bindings: list[_Diag4BootstrapBinding] = []
    native_bindings: list[_Diag5BootstrapNativeBinding] = []
    retain = False
    try:
        retain = _diag5_preimport_bootstrap_impl(
            argv=argv,
            current_entry=current_entry,
            environment=environment,
            execve=execve,
            bindings=bindings,
            native_bindings=native_bindings,
        )
        if retain:
            _diag5_retained_bootstrap_bindings.extend(bindings)
            _diag5_retained_native_bindings.extend(native_bindings)
            return
        _diag4_preimport_bootstrap_impl(
            argv=argv,
            current_entry=current_entry,
            environment=environment,
            execve=execve,
            bindings=bindings,
        )
    finally:
        if not retain:
            for binding in reversed(native_bindings):
                fcntl.flock(binding.descriptor, fcntl.LOCK_UN)
                os.close(binding.descriptor)
            for binding in reversed(bindings):
                os.close(binding.descriptor)


_diag4_preimport_bootstrap(
    argv=sys.argv,
    current_entry=Path(__file__),
    environment=os.environ,
)

_runner_root = Path(__file__).resolve().parents[1]
if str(_runner_root) not in sys.path:
    sys.path.insert(0, str(_runner_root))
_is_snapshot_child = "--snapshot-child" in _protected_cli_arguments
_has_supervisor_mode = any(
    flag in _protected_cli_arguments
    for flag in (
        "--preflight-only",
        "--diagnostic-only",
        "--diagnostic-successor-authority",
    )
)
if _is_snapshot_child and _has_supervisor_mode:
    raise RuntimeError("snapshot child cannot combine a supervisor mode")
if "--diagnostic-only" in _protected_cli_arguments:
    raise RuntimeError(
        "legacy --diagnostic-only is not authorized; "
        "use --diagnostic-successor-authority"
    )
_is_diag2_parent_cli = (
    "--diagnostic-successor-authority" in _protected_cli_arguments
    and not _is_snapshot_child
)
if _is_diag2_parent_cli:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# The parent CPU policy must precede every import that can transitively load JAX.
# isort: off
from benchmarks.single_stage_fullspace_snapshot import (
    DIAG4_CPU_SNAPSHOT_ROLES,
    DIAG4_GPU_SNAPSHOT_ROLES,
    DIAG5_GPU_SNAPSHOT_ROLES,
    LEGACY_SNAPSHOT_ROLES,
    command_buffer_disabled_xla_flags as _command_buffer_disabled_xla_flags,
)
# isort: on


_is_successor_snapshot_child = _is_snapshot_child and (
    os.environ.get(_DIAG2_CHILD_ENV) == "1"
    or os.environ.get(_DIAG4_CHILD_ENV) == "1"
    or os.environ.get(_DIAG5_CHILD_ENV) == "1"
)
if _is_successor_snapshot_child:
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if (
        os.environ.get("JAX_PLATFORMS") != "cuda"
        or "JAX_PLATFORM_NAME" in os.environ
        or "JAX_COMPILATION_CACHE_DIR" in os.environ
        or os.environ.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "true"
        or os.environ.get("JAX_ENABLE_X64") != "true"
        or _xla_flags != _command_buffer_disabled_xla_flags(_xla_flags)
    ):
        route_name = (
            "DIAG5"
            if os.environ.get(_DIAG5_CHILD_ENV) == "1"
            else "DIAG4"
            if os.environ.get(_DIAG4_CHILD_ENV) == "1"
            else "DIAG2"
        )
        raise RuntimeError(f"{route_name} child pre-import policy is not canonical")

if _is_snapshot_child:
    _snapshot_root = _runner_root
    _snapshot_source_root = _snapshot_root / "src"
    sys.path.insert(0, str(_snapshot_root))
    from benchmarks.single_stage_fullspace_snapshot import (
        activate_snapshot_source_imports,
    )

    activate_snapshot_source_imports(_snapshot_source_root)

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import simsoptpp
from examples.jax.parity.input_bundle import read_input_bundle

jax.config.update("jax_enable_x64", True)

from simsopt_jax.objectives.single_stage_fullspace import FullSpaceProblem
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.runtime.trace_annotations import (
    GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256,
    trace_session,
)
from simsopt_jax.solve.fullspace import fullspace_scaling_from_bootstrap
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NEQ_GNTR3_OPTIONS,
    NEQ_GNTR3_ROUTE,
    NEQ_GNTR3_SCHEMA_VERSION,
    NativeEquivalentQualityPolicy,
    PreparedNeqAcceptedQualityDiagnostics,
    PreparedNeqGntr1,
    PreparedNeqGntr2,
    PreparedNeqGntr3,
    PreparedNeqTerminalEndpointDiagnostics,
    build_native_equivalent_terminal_diagnostic,
    prepare_neq_accepted_quality_diagnostics,
    prepare_neq_gntr1,
    prepare_neq_gntr3,
    prepare_neq_terminal_endpoint_diagnostics,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    NativeSingleStageEndpointRuntime,
    build_native_single_stage_endpoint_runtime,
)

from benchmarks.process_gpu_monitor import (
    SupervisorGpuZeroObservation,
    capture_supervisor_gpu_zero,
    supervisor_query_executable_sha256,
)
from benchmarks.single_stage_changed_state_profiler_policy import (
    PROFILED_PROFILER_POLICY,
    TRACE_VIEWER_MAX_EVENTS,
    TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT,
    build_jax_profiler_options,
)
from benchmarks.single_stage_fullspace_process_gpu_monitor import (
    BoundProcessGpuMemoryMonitor,
    bound_gpu_memory_payload,
    read_linux_process_identity,
)
from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_FILENAME,
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
    ArtifactRef,
    JsonValue,
    SnapshotIdentity,
    SnapshotPostPublicationError,
    SnapshotPublication,
    SourceRoot,
    build_runtime_evidence,
    build_runtime_evidence_v2,
    capture_worktree_identity,
    copy_diag5_immutable_snapshot,
    copy_immutable_snapshot,
    load_canonical_json_bytes,
    load_snapshot,
    observe_live_runtime,
    observe_live_runtime_v2,
    publish_immutable_snapshot,
    publish_runtime_evidence,
    publish_runtime_evidence_v2,
)
from benchmarks.single_stage_native_equivalent_endpoint_audit import (
    endpoint_audit_payload,
    produce_native_equivalent_endpoint_audit,
    validate_endpoint_audit_payload,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    ARRAY_SPECS as DIAGNOSTIC_ARRAY_SPECS,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    DIAG2_CHILD_TERMINAL_SCHEMA_VERSION,
    DIAG2_EVIDENCE_SLOT_NAMES,
    DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
    DIAG2_MANIFEST_FILENAME,
    DIAG2_PLAN_SHA256,
    DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION,
    DIAG2_PROCESS_SCHEMA_VERSION,
    DIAG2_RECEIPT_FILENAME,
    DIAG2_ROUTE,
    DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION,
    DIAG3_COLD_RESULT_SCHEMA_VERSION,
    DIAG3_COMMITTED_NUMERICAL_DIRECTORY,
    DIAG3_PENDING_NUMERICAL_DIRECTORY,
    DIAG3_UNCOMMITTED_NUMERICAL_DIRECTORY,
    DIAG4_COLD_RESULT_SCHEMA_VERSION,
    DIAG4_EVIDENCE_SLOT_NAMES,
    DIAG4_EVIDENCE_SLOT_PATHS,
    DIAG4_EXECUTION_SCHEMA_VERSION,
    DIAG4_NUMERICAL_BUNDLE_SCHEMA_VERSION,
    DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
    DIAG4_NUMERICAL_ROUTE,
    DIAG4_PLAN_SHA256,
    DIAG4_PREFLIGHT_SCHEMA_VERSION,
    DIAG4_PROFILER_CALL_AUDIT,
    DIAG4_ROUTE,
    DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
    DIAG4_SOLVE_TIMING_SCHEMA_VERSION,
    DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
    DIAG5_COLD_RESULT_SCHEMA_VERSION,
    DIAG5_EVIDENCE_SLOT_NAMES,
    DIAG5_EVIDENCE_SLOT_PATHS,
    DIAG5_EXECUTION_SCHEMA_VERSION,
    DIAG5_FROZEN_SUBSET_SCHEMA_VERSION,
    DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
    DIAG5_MEMORY_SCHEMA_VERSION,
    DIAG5_NUMERICAL_BUNDLE_SCHEMA_VERSION,
    DIAG5_PLAN_SHA256,
    DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION,
    DIAG5_PREFLIGHT_SCHEMA_VERSION,
    DIAG5_PROCESS_SCHEMA_VERSION,
    DIAG5_ROUTE,
    DIAG5_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
    DIAG5_SCHEMA_VERSION,
    DIAG5_SOLVE_TIMING_SCHEMA_VERSION,
    DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION,
    FINAL_CERTIFICATE_FIELDS,
    PREFLIGHT_EVIDENCE_REF_KEYS,
    TRACE_LOOP_ENVELOPE_NAME,
    AbsenceReason,
    Diag2PreflightGateError,
    Diag2SetupGateError,
    Diag4NumericalDocumentError,
    Diag5NumericalDocumentError,
    DiagnosticReceiptV4,
    EvidenceSlot,
    EvidenceSlotV4,
    EvidenceSlotV5,
    FailureReasonCodeV2,
    FailureReasonCodeV4,
    FailureReasonCodeV5,
    FailureStageV2,
    FailureStageV4,
    FailureStageV5,
    KktStatus,
    NativeEquivalentNumericalIdentity,
    StructuredFailureV2,
    StructuredFailureV4,
    StructuredFailureV5,
    SupervisorQueryV2,
    array_evidence_payload,
    build_diag2_compile_failure_producer_payload,
    build_diag2_frozen_numerical_subset_payload,
    build_diag2_policy_authority_payload,
    build_diag2_supervisor_terminal_payload,
    build_diag2_supervisor_zero_payload,
    build_diag3_diagnostic_receipt,
    build_diag4_diagnostic_receipt,
    build_diag4_frozen_numerical_subset_payload,
    build_diag4_supervisor_terminal_payload,
    build_diag5_compile_failure_producer_payload,
    build_diag5_diagnostic_receipt,
    build_diag5_frozen_numerical_subset_payload,
    build_diag5_policy_authority_payload,
    build_diag5_supervisor_failure_producer_payload,
    build_diag5_supervisor_terminal_payload,
    build_diag5_supervisor_zero_payload,
    build_diagnostic_receipt,
    build_incomplete_diagnostic_receipt,
    classify_diag3_cold_evidence,
    classify_diag3_subordinate_child_outcome,
    classify_diag5_receipt_construction_error,
    derive_diag3_algorithm_route,
    derive_diag3_evidence_slots,
    derive_diag4_evidence_slots,
    derive_diag4_scientific_outcome,
    derive_diag5_evidence_slots,
    derive_diag5_scientific_outcome,
    diag2_postlaunch_setup_failure,
    diag3_artifact_manifest_payload,
    diag3_diagnostic_receipt_bytes,
    diag4_artifact_manifest_payload,
    diag4_diagnostic_receipt_bytes,
    diag4_execution_evidence_payload,
    diag4_profiler_call_audit_payload,
    diag4_terminal_numerical_payload,
    diag5_artifact_manifest_payload,
    diag5_diagnostic_receipt_bytes,
    diag5_execution_evidence_payload,
    diag5_history_evidence_from_arrays,
    diag5_policy_evidence_payload,
    diag5_safeguard_telemetry_payload,
    diag5_solve_timing_evidence_payload,
    diagnostic_artifact_manifest_payload,
    diagnostic_receipt_bytes,
    diagnostic_receipt_payload,
    execution_evidence_payload,
    history_evidence_from_arrays,
    load_and_validate_diag3_artifact,
    load_and_validate_diag3_staging,
    load_and_validate_diag4_artifact,
    load_and_validate_diag4_staging,
    load_and_validate_diag5_artifact,
    load_and_validate_diag5_rollback,
    load_and_validate_diag5_staging,
    load_and_validate_diagnostic_artifact,
    normalize_chrome_trace,
    policy_evidence_payload,
    safeguard_telemetry_payload,
    select_diag2_failure,
    solve_timing_evidence_payload,
    terminal_numerical_payload,
    validate_diag2_policy_authority_payload,
    validate_diag2_preflight_gate,
    validate_diag2_setup_authorities,
    validate_diag2_supervisor_zero_payload,
    validate_diag3_producer_payload,
    validate_diag3_writable_staging,
    validate_diag4_frozen_numerical_subset_payload,
    validate_diag4_numerical_documents,
    validate_diag4_preflight_gate,
    validate_diag4_producer_payload,
    validate_diag4_writable_staging,
    validate_diag5_frozen_numerical_subset_payload,
    validate_diag5_numerical_documents,
    validate_diag5_policy_authority_payload,
    validate_diag5_policy_evidence_payload,
    validate_diag5_preflight_gate,
    validate_diag5_producer_payload,
    validate_diag5_supervisor_failure_producer_payload,
    validate_diag5_supervisor_zero_payload,
    validate_diag5_writable_staging,
    validate_diagnostic_preflight_gate,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    EVIDENCE_REF_KEYS as DIAGNOSTIC_EVIDENCE_REF_KEYS,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    MANIFEST_FILENAME as DIAGNOSTIC_MANIFEST_FILENAME,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    PLAN_SHA256 as DIAGNOSTIC_PLAN_SHA256,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    RECEIPT_FILENAME as DIAGNOSTIC_RECEIPT_FILENAME,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    ROUTE as DIAGNOSTIC_ROUTE,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    SCHEMA_VERSION as DIAGNOSTIC_SCHEMA_VERSION,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    load_and_validate_diag2_artifact as load_and_validate_diag2_artifact,  # noqa: PLC0414
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    load_and_validate_diag2_staging as load_and_validate_diag2_staging,  # noqa: PLC0414
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    validate_diag2_writable_staging as validate_diag2_writable_staging,  # noqa: PLC0414
)
from benchmarks.single_stage_native_equivalent_quality_receipt import (
    GPU_UUID,
    PLAN_SHA256,
    ROUTE,
    SAMPLE_SCHEMA_VERSION,
    CampaignReceipt,
    CandidateEvidence,
    EndpointAuditEvidence,
    ExecutionStatus,
    KktTelemetry,
    KktTelemetryStatus,
    ReferenceReceipt,
    ResourceEvidence,
    SampleName,
    SampleQuality,
    SampleReceipt,
    SourceIdentityEvidence,
    TimingEvidence,
    campaign_payload,
    canonical_json_bytes,
    reference_receipt_from_artifact,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG4_CPU_QUALIFICATION_ROOT,
    Diag4AuthorityLifecycle,
    Diag4ConsumptionMarkerInvalidError,
    Diag4SuccessorAuthorityClaim,
    Diag5AuthorityLifecycle,
    Diag5ConsumptionMarkerInvalidError,
    Diag5FinalizerError,
    Diag5FinalizerFailureCategory,
    Diag5FinalizerSourceKind,
    Diag5PhysicalCancellationError,
    Diag5PhysicalCancellationObservation,
    Diag5PhysicalEvidenceReservation,
    Diag5PublishedOutputKind,
    Diag5SuccessorAuthorityClaim,
    PreSourceFailure,
    PublishedSnapshot,
    SuccessorAuthorityClaim,
    bind_diag4_staging_root,
    bind_diag5_staging_root,
    cancel_diag5_physical_failure_evidence,
    claim_diag4_successor_authority,
    claim_diag5_successor_authority,
    claim_successor_authority,
    consume_diag4_successor_authority,
    consume_diag5_successor_authority,
    diag4_authority_lifecycle,
    diag5_authority_lifecycle,
    finalize_diag4_prelaunch_failure,
    finalize_diag5_physical_evidence_success,
    fsync_diag5_output_parent,
    prepare_diag5_physical_failure_evidence,
    publish_diag5_bound_staging,
    publish_diag5_physical_failure_evidence,
    revalidate_diag4_successor_authority,
    revalidate_diag5_published_output,
    revalidate_diag5_successor_authority,
    rollback_diag5_bound_final,
    validate_diag4_successor_snapshot,
    validate_diag5_consumption_marker,
    validate_diag5_successor_snapshot,
    validate_successor_snapshot,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    Diag5RollbackObservation as AuthorityDiag5RollbackObservation,
)
from benchmarks.single_stage_native_equivalent_reference import (
    REFERENCE_FILENAME,
    validate_native_equivalent_reference,
)
from benchmarks.single_stage_native_equivalent_reference import (
    SCHEMA_VERSION as NATIVE_REFERENCE_SCHEMA_VERSION,
)

_DIAG3_NUMERICAL_PENDING_NAME: Final = Path(DIAG3_PENDING_NUMERICAL_DIRECTORY).name
_DIAG3_NUMERICAL_COMMITTED_NAME: Final = Path(DIAG3_COMMITTED_NUMERICAL_DIRECTORY).name
_DIAG3_NUMERICAL_UNCOMMITTED_NAME: Final = Path(
    DIAG3_UNCOMMITTED_NUMERICAL_DIRECTORY
).name

WORKER_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-worker-v1"
PREFLIGHT_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-preflight-worker-v1"
PREFLIGHT_ARTIFACT_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-preflight-artifact-v1"
)
MEMORY_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-memory-v1"
DIAG4_SUMMARY_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-diag4-summary-v1"
SOURCE_SNAPSHOT_DIRECTORY: Final = "source-snapshot"
SOURCE_MANIFEST_ARTIFACT: Final = "source-snapshot/source-manifest.json"
CAMPAIGN_RECEIPT_FILENAME: Final = "campaign.json"
CAMPAIGN_ARTIFACT_MANIFEST_FILENAME: Final = "artifact-manifest.json"
CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA: Final = (
    "single-stage-neq-gntr1-campaign-artifact-manifest-v1"
)
SOLVE_TIMEOUT_SECONDS: Final = 360.0
PROCESS_TIMEOUT_SECONDS: Final = 900.0
FAILURE_OUTPUT_TAIL_BYTES: Final = 16 * 1024
SAMPLE_ORDER: Final = (
    SampleName.COLD,
    SampleName.WARM_1,
    SampleName.WARM_2,
    SampleName.WARM_3,
)
_SNAPSHOT_MANIFEST_ENV: Final = "SIMSOPT_NEQ_SNAPSHOT_MANIFEST_SHA256"
_CAMPAIGN_ROOT_ENV: Final = "SIMSOPT_NEQ_CAMPAIGN_ROOT"
_DIAGNOSTIC_CHILD_ENV: Final = "SIMSOPT_NEQ_DIAGNOSTIC_CHILD"
_ENTRYPOINT: Final = "benchmarks/run_single_stage_native_equivalent_quality_campaign.py"
_OWNED_EXECUTION_FILES: Final = (
    _ENTRYPOINT,
    "benchmarks/process_gpu_monitor.py",
    "benchmarks/single_stage_fullspace_process_gpu_monitor.py",
    "benchmarks/single_stage_fullspace_snapshot.py",
    "benchmarks/single_stage_native_equivalent_endpoint_audit.py",
    "benchmarks/single_stage_native_equivalent_quality_receipt.py",
    "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py",
    "benchmarks/single_stage_native_equivalent_quality_successor_authority.py",
    "benchmarks/single_stage_native_equivalent_reference.py",
    "docs/single_stage_jax_gpu_native_equivalent_quality_speed_implementation_plan.md",
    "docs/single_stage_jax_gpu_native_equivalent_quality_no_hit_diagnostic_implementation_plan.md",
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag2_implementation_plan.md",
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag3_command_buffer_recovery_plan.md",
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag3_command_buffer_recovery_authorization.json",
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md",
    "tests/benchmarks/_diag2_fixture.py",
    "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py",
    "tests/benchmarks/test_process_gpu_monitor.py",
    "tests/benchmarks/test_single_stage_fullspace_snapshot.py",
    "tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py",
    "tests/benchmarks/test_single_stage_native_equivalent_reference.py",
    "tests/geo/test_fullspace_native_equivalent_quality.py",
    "tests/geo/test_projected_gauss_newton_trust_region.py",
)


class ChildTerminalStatus(StrEnum):
    COMPLETE = "COMPLETE"
    TIMEOUT = "TIMEOUT"
    COMPILE_FAILURE = "COMPILE_FAILURE"
    CRASH = "CRASH"
    MONITOR_FAILURE = "MONITOR_FAILURE"
    PROTOCOL_FAILURE = "PROTOCOL_FAILURE"


class DiagnosticChildMode(StrEnum):
    """The only two children authorized by the diagnostic schedule."""

    PREFLIGHT = "preflight"
    COLD = "cold"


class MonitorFailureKind(StrEnum):
    """Closed DIAG2 monitor outcome carried by process and terminal evidence."""

    NONE = "NONE"
    BINDING = "BINDING"
    FINALIZATION = "FINALIZATION"


@dataclass(frozen=True, slots=True)
class Diag2PolicyAuthorityValues:
    reference_volume: float
    volume_target: float
    native_raw_equalities: np.ndarray
    constraint_inverse_scale: np.ndarray
    policy_sha256: str


@dataclass(frozen=True, slots=True)
class Diag2Publication:
    staging_root: Path
    final_root: Path
    nonce: str
    staging_device: int | None = None
    staging_inode: int | None = None


@dataclass(frozen=True, slots=True)
class Diag4VisiblePartial:
    """One typed publication failure retained under the partial sibling."""

    outcome: StructuredFailureV4
    root: Path


class Diag4PhysicalPublicationReason(StrEnum):
    """Out-of-band failures after the sealed tree has been renamed final."""

    FINAL_FSYNC_FAILED = "FINAL_FSYNC_FAILED"
    FINAL_DEEP_LOAD_FAILED = "FINAL_DEEP_LOAD_FAILED"
    POST_FINAL_AUTHORITY_REVALIDATION_FAILED = (
        "POST_FINAL_AUTHORITY_REVALIDATION_FAILED"
    )
    POST_FINAL_AUTHORITY_FINALIZATION_FAILED = (
        "POST_FINAL_AUTHORITY_FINALIZATION_FAILED"
    )


class Diag5PhysicalPublicationReason(StrEnum):
    FINAL_FSYNC_FAILED = "FINAL_FSYNC_FAILED"
    FINAL_DEEP_LOAD_FAILED = "FINAL_DEEP_LOAD_FAILED"
    POST_FINAL_AUTHORITY_REVALIDATION_FAILED = (
        "POST_FINAL_AUTHORITY_REVALIDATION_FAILED"
    )
    POST_FINAL_AUTHORITY_FINALIZATION_FAILED = (
        "POST_FINAL_AUTHORITY_FINALIZATION_FAILED"
    )


class Diag5RollbackCause(StrEnum):
    NONE = "NONE"
    ROLLBACK_COLLISION = "ROLLBACK_COLLISION"
    ROLLBACK_RENAME_FAILED = "ROLLBACK_RENAME_FAILED"
    ROLLBACK_PARENT_FSYNC_FAILED = "ROLLBACK_PARENT_FSYNC_FAILED"
    ROLLBACK_DEEP_LOAD_FAILED = "ROLLBACK_DEEP_LOAD_FAILED"
    ROLLBACK_VISIBILITY_AMBIGUOUS = "ROLLBACK_VISIBILITY_AMBIGUOUS"


class Diag5RollbackState(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"


class Diag5PhysicalPathState(StrEnum):
    ABSENT = "ABSENT"
    VISIBLE_VALIDATED = "VISIBLE_VALIDATED"
    VISIBLE_INVALID = "VISIBLE_INVALID"
    VISIBILITY_AMBIGUOUS = "VISIBILITY_AMBIGUOUS"


class Diag5EvidenceNamespaceState(StrEnum):
    PENDING_BOUND = "PENDING_BOUND"
    PENDING_UNLINKED = "PENDING_UNLINKED"
    PENDING_AMBIGUOUS = "PENDING_AMBIGUOUS"


DIAG5_PHYSICAL_PUBLICATION_FAILURE_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag5-physical-publication-failure-v1"
)


@dataclass(frozen=True, slots=True)
class Diag5PhysicalPublicationObservation:
    rollback_cause: Diag5RollbackCause
    rollback_state: Diag5RollbackState
    final_path_state: Diag5PhysicalPathState
    rollback_path_state: Diag5PhysicalPathState
    evidence_namespace_state_at_seal: Diag5EvidenceNamespaceState


class Diag5PhysicalPublicationError(RuntimeError):
    """A post-final DIAG5 fault with sealed out-of-band adjudication."""

    def __init__(
        self,
        reason: Diag5PhysicalPublicationReason,
        observation: Diag5PhysicalPublicationObservation,
        evidence_path: Path | None,
        cause: BaseException,
    ) -> None:
        super().__init__(f"{reason.value}:{type(cause).__name__}:{cause}")
        self.reason = reason
        self.observation = observation
        self.evidence_path = evidence_path
        self.cause = cause


class Diag5PreFinalPublicationError(RuntimeError):
    """A typed pre-final fault retained at the fixed staging root."""

    def __init__(
        self,
        *,
        terminal_outcome: StructuredFailureV5,
        publication_failure: StructuredFailureV5,
        staging_root: Path,
        cause: BaseException,
        cancellation_observation: Diag5PhysicalCancellationObservation | None = None,
        cleanup_cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            f"{publication_failure.stage.value}:"
            f"{publication_failure.reason.value}:"
            f"{type(cause).__name__}:{cause}"
        )
        self.terminal_outcome = terminal_outcome
        self.publication_failure = publication_failure
        self.staging_root = staging_root
        self.cause = cause
        self.cancellation_observation = cancellation_observation
        self.cleanup_cause = cleanup_cause


def _raise_diag5_pre_final_failure(
    *,
    publication: Diag2Publication,
    terminal_outcome: StructuredFailureV5,
    stage: FailureStageV5,
    reason: FailureReasonCodeV5,
    cause: BaseException,
    cancellation_observation: Diag5PhysicalCancellationObservation | None = None,
) -> Never:
    raise Diag5PreFinalPublicationError(
        terminal_outcome=terminal_outcome,
        publication_failure=_diag5_failure(
            stage,
            reason,
            f"{type(cause).__name__}:{cause}",
        ),
        staging_root=publication.staging_root,
        cause=cause,
        cancellation_observation=cancellation_observation,
    ) from cause


def _diag5_physical_observation(
    value: AuthorityDiag5RollbackObservation,
) -> Diag5PhysicalPublicationObservation:
    return Diag5PhysicalPublicationObservation(
        rollback_cause=Diag5RollbackCause(value.rollback_cause.value),
        rollback_state=Diag5RollbackState(value.rollback_state.value),
        final_path_state=Diag5PhysicalPathState(value.final_path_state.value),
        rollback_path_state=Diag5PhysicalPathState(value.rollback_path_state.value),
        evidence_namespace_state_at_seal=Diag5EvidenceNamespaceState(
            value.evidence_namespace_state_at_seal.value
        ),
    )


def _diag5_physical_publication_failure_payload(
    *,
    successor_claim: Diag5SuccessorAuthorityClaim,
    original_reason: Diag5PhysicalPublicationReason,
    observation: Diag5PhysicalPublicationObservation,
    sealed_artifact_manifest_sha256: str,
) -> dict[str, JsonValue]:
    """Build the exact out-of-band DIAG5 rollback evidence wrapper."""

    successful = (
        observation.rollback_state is Diag5RollbackState.SUCCEEDED
        and observation.rollback_cause is Diag5RollbackCause.NONE
        and observation.final_path_state is Diag5PhysicalPathState.ABSENT
        and observation.rollback_path_state is Diag5PhysicalPathState.VISIBLE_VALIDATED
    )
    failed = (
        observation.rollback_state is Diag5RollbackState.FAILED
        and observation.rollback_cause
        in {
            Diag5RollbackCause.ROLLBACK_COLLISION,
            Diag5RollbackCause.ROLLBACK_RENAME_FAILED,
            Diag5RollbackCause.ROLLBACK_PARENT_FSYNC_FAILED,
            Diag5RollbackCause.ROLLBACK_DEEP_LOAD_FAILED,
        }
    )
    ambiguous = (
        observation.rollback_state is Diag5RollbackState.AMBIGUOUS
        and observation.rollback_cause
        is Diag5RollbackCause.ROLLBACK_VISIBILITY_AMBIGUOUS
    )
    if sum((successful, failed, ambiguous)) != 1:
        raise ValueError("DIAG5 rollback state and cause differ")
    if len(sealed_artifact_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in sealed_artifact_manifest_sha256
    ):
        raise ValueError("DIAG5 sealed artifact manifest SHA-256 is invalid")
    return {
        "schema_version": DIAG5_PHYSICAL_PUBLICATION_FAILURE_SCHEMA_VERSION,
        "route": DIAG5_ROUTE,
        "authority_sha256": successor_claim.authority_sha256,
        "original_reason": original_reason.value,
        "rollback_cause": observation.rollback_cause.value,
        "rollback_state": observation.rollback_state.value,
        "final_path": str(successor_claim.expected_gpu_output_root),
        "final_path_state": observation.final_path_state.value,
        "rollback_path": str(successor_claim.expected_gpu_rollback_root),
        "rollback_path_state": observation.rollback_path_state.value,
        "evidence_namespace_state_at_seal": (
            observation.evidence_namespace_state_at_seal.value
        ),
        "sealed_artifact_manifest_sha256": sealed_artifact_manifest_sha256,
    }


class Diag4HardPublicationError(RuntimeError):
    """An unsealable physical publication fault with a visible partial root."""

    def __init__(
        self,
        reason: FailureReasonCodeV4 | Diag4PhysicalPublicationReason,
        root: Path,
        cause: BaseException,
        *,
        staging_exists: bool | None = None,
        final_exists: bool | None = None,
        authority_lifecycle: Diag4AuthorityLifecycle | None = None,
    ) -> None:
        super().__init__(f"{reason.value}:{type(cause).__name__}:{cause}")
        self.reason = reason
        self.root = root
        self.cause = cause
        self.staging_exists = staging_exists
        self.final_exists = final_exists
        self.authority_lifecycle = authority_lifecycle


class Diag4RollbackHardPublicationError(Diag4HardPublicationError):
    """A final-tree rollback fault with exact observed path and lease state."""

    def __init__(
        self,
        reason: Diag4PhysicalPublicationReason,
        publication: Diag2Publication,
        cause: BaseException,
        rollback_cause: BaseException,
        *,
        authority_lifecycle: Diag4AuthorityLifecycle | None,
    ) -> None:
        staging_exists = os.path.lexists(publication.staging_root)
        final_exists = os.path.lexists(publication.final_root)
        visible_root = (
            publication.final_root if final_exists else publication.staging_root
        )
        super().__init__(
            reason,
            visible_root,
            cause,
            staging_exists=staging_exists,
            final_exists=final_exists,
            authority_lifecycle=authority_lifecycle,
        )
        self.rollback_cause = rollback_cause
        self.staging_root = publication.staging_root
        self.final_root = publication.final_root
        self.args = (
            "".join(
                (
                    f"{reason.value}:ROLLBACK_FAILED:",
                    f"staging_exists={staging_exists}:final_exists={final_exists}:",
                    f"authority_lifecycle={authority_lifecycle}:",
                    f"{type(rollback_cause).__name__}:{rollback_cause}",
                )
            ),
        )


_DIAG2_VOLUME_TARGET_HEX: Final = "-0x1.296a9ce4a271dp-2"
_DIAG2_BOOZER_SCALE_HEX: Final = "0x1.0101828467ee9p-4"
_DIAG2_VOLUME_SCALE_HEX: Final = "0x1.b8b3b0469c959p+1"
_DIAG2_SCALE_SHA256: Final = (
    "ee71932a5d6a0dfb0ca4dc9d852bf1f32e669dbc81ced26903db956027e1155e"
)
_DIAG2_POLICY_SHA256: Final = (
    "6face7116d36d2eae954bb5b3bde465f37c990ceb9b319ba2c20bb62ad1a6f99"
)


class SolveTimeoutError(TimeoutError):
    """The synchronized bounded solve exceeded its 360-second limit."""


@dataclass(frozen=True, slots=True)
class SnapshotChildInvocation:
    """Pinned interpreter, immutable entrypoint, cwd, and environment."""

    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]


@dataclass(frozen=True, slots=True)
class RawGpuMemorySample:
    """One parent-observed GPU-memory sample retained without summarization."""

    sampled_at_unix_ns: int
    used_memory_mib: int


@dataclass(frozen=True, slots=True)
class SupervisedSample:
    """One unreplaced child outcome, including parent-observed evidence."""

    sample: SampleName
    terminal_status: ChildTerminalStatus
    child_pid: int
    child_start_time_ticks: int
    process_seconds: float
    producer: dict[str, JsonValue]
    memory: dict[str, JsonValue] | None
    failure_reasons: tuple[str, ...]
    pre_source_identity: SourceIdentityEvidence | None = None
    post_source_identity: SourceIdentityEvidence | None = None
    process_diagnostics: dict[str, JsonValue] | None = None
    observed_child_argv: tuple[str, ...] | None = None
    stdout: bytes | None = None
    stderr: bytes | None = None
    memory_samples: tuple[RawGpuMemorySample, ...] | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticSupervisedSampleV2:
    """DIAG2 child evidence with an explicitly optional typed producer."""

    sample: SampleName
    launched: bool
    terminal_status: ChildTerminalStatus
    child_pid: int
    child_start_time_ticks: int
    process_seconds: float
    producer: dict[str, JsonValue] | None
    producer_absence_reason: AbsenceReason | None
    selected_failure_reason: FailureReasonCodeV2 | None
    memory: dict[str, JsonValue] | None
    raw_failure_reasons: tuple[str, ...]
    observed_child_argv: tuple[str, ...] | None
    stdout: bytes
    stderr: bytes
    memory_samples: tuple[RawGpuMemorySample, ...]
    pre_source_identity: SourceIdentityEvidence | None = None
    post_source_identity: SourceIdentityEvidence | None = None
    process_diagnostics: dict[str, JsonValue] | None = None
    process_started_monotonic_ns: int = 0
    process_stopped_monotonic_ns: int = 0
    monitor_failure_kind: MonitorFailureKind = MonitorFailureKind.NONE

    def __post_init__(self) -> None:
        if (self.producer is None) == (self.producer_absence_reason is None):
            raise ValueError("DIAG2 producer requires exactly one presence state")


@dataclass(frozen=True, slots=True)
class ColdNumericalBundlePublication:
    """Parent-owned physical paths for one child-staged numerical result."""

    pending_root: Path
    committed_root: Path
    uncommitted_root: Path


@dataclass(frozen=True, slots=True)
class Diag5ColdNumericalResolution:
    """The terminal result and independent pending-tree disposition."""

    outcome: DiagnosticSupervisedSampleV2
    terminal_failure: StructuredFailureV5 | None
    pending_disposition_failure: StructuredFailureV5 | None

    @property
    def publication_allowed(self) -> bool:
        return self.pending_disposition_failure is None


class Diag5PendingDispositionError(RuntimeError):
    """A DIAG5 pending tree could not be committed or quarantined."""

    def __init__(
        self,
        *,
        terminal_failure: StructuredFailureV5 | None,
        pending_disposition_failure: StructuredFailureV5,
        staging_root: Path,
    ) -> None:
        super().__init__(
            f"{pending_disposition_failure.stage.value}:"
            f"{pending_disposition_failure.reason.value}:"
            f"{pending_disposition_failure.detail_sha256}"
        )
        self.terminal_failure = terminal_failure
        self.pending_disposition_failure = pending_disposition_failure
        self.staging_root = staging_root


class SampleExecutor(Protocol):
    """Execute and retain exactly one named campaign sample."""

    def __call__(self, sample: SampleName) -> SupervisedSample: ...


class NeqRoutePreparer(Protocol):
    """Compile one route from the shared frozen problem and policy inputs."""

    def __call__(
        self,
        problem: FullSpaceProblem,
        bootstrap_state: jax.Array,
        initial_physical_state: jax.Array,
        policy: NativeEquivalentQualityPolicy,
    ) -> PreparedNeqGntr1 | PreparedNeqGntr2 | PreparedNeqGntr3: ...


@dataclass(frozen=True, slots=True)
class PreparedWorker:
    """Compiled route plus the two untimed native audit authorities."""

    route: PreparedNeqGntr1 | PreparedNeqGntr2 | PreparedNeqGntr3
    native_runtime: NativeSingleStageEndpointRuntime
    native_reference_state: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedDiagnosticWorker:
    """Annotation-enabled loop and separately compiled terminal diagnostic."""

    worker: PreparedWorker
    accepted_quality: PreparedNeqAcceptedQualityDiagnostics
    terminal: PreparedNeqTerminalEndpointDiagnostics


def _bounded_process_diagnostics(
    invocation: SnapshotChildInvocation,
    *,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
    monitor_error: Exception | None,
) -> dict[str, JsonValue]:
    """Retain bounded raw failure tails plus complete byte identities."""

    return {
        "argv": list(invocation.argv),
        "returncode": returncode,
        "stdout_size_bytes": len(stdout),
        "stdout_sha256": _sha256(stdout),
        "stdout_tail": stdout[-FAILURE_OUTPUT_TAIL_BYTES:].decode(
            "utf-8", "backslashreplace"
        ),
        "stderr_size_bytes": len(stderr),
        "stderr_sha256": _sha256(stderr),
        "stderr_tail": stderr[-FAILURE_OUTPUT_TAIL_BYTES:].decode(
            "utf-8", "backslashreplace"
        ),
        "monitor_error_type": (
            type(monitor_error).__name__ if monitor_error is not None else None
        ),
        "monitor_error_message": (
            str(monitor_error) if monitor_error is not None else None
        ),
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_ref(path: Path, campaign_root: Path, schema: str) -> ArtifactRef:
    relative = path.relative_to(campaign_root).as_posix()
    payload = path.read_bytes()
    return ArtifactRef(relative, _sha256(payload), len(payload), schema)


def _artifact_ref_at(
    path: Path, campaign_root: Path, logical_path: Path, schema: str
) -> ArtifactRef:
    """Hash physical bytes while binding their post-commit campaign path."""

    relative = logical_path.relative_to(campaign_root).as_posix()
    payload = path.read_bytes()
    return ArtifactRef(relative, _sha256(payload), len(payload), schema)


def _artifact_ref_payload(reference: ArtifactRef) -> dict[str, JsonValue]:
    return {
        "relative_path": reference.relative_path,
        "schema_version": reference.schema_version,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
    }


def _publish_canonical_json(
    path: Path,
    payload: Mapping[str, JsonValue],
) -> None:
    """Exclusively publish one canonical, immutable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(dict(payload)))
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _publish_npy(path: Path, values: np.ndarray) -> None:
    """Exclusively publish one canonical C-contiguous NPY array."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _prepare_diag2_publication(
    final_root: Path,
    *,
    repository_root: Path,
) -> Diag2Publication:
    """Create the sole sibling staging root while the final root stays absent."""

    raw_parent = final_root.parent.absolute()
    component = raw_parent
    while True:
        if component.is_symlink():
            raise ValueError("DIAG2 output parent contains a symlink")
        if component.parent == component:
            break
        component = component.parent
    parent = raw_parent.resolve(strict=True)
    repository = repository_root.resolve(strict=True)
    if parent == repository or parent.is_relative_to(repository):
        raise ValueError("DIAG2 output parent must be outside the repository")
    requested = parent / final_root.name
    if (
        not final_root.name
        or os.path.lexists(requested)
        or tuple(parent.glob(f"{final_root.name}.partial-*"))
    ):
        raise FileExistsError("DIAG2 final output and staging siblings must be absent")
    nonce = secrets.token_hex(16)
    staging = parent / f"{final_root.name}.partial-{nonce}"
    staging.mkdir(mode=0o755, exist_ok=False)
    staging_stat = staging.stat(follow_symlinks=False)
    return Diag2Publication(
        staging,
        requested,
        nonce,
        staging_device=staging_stat.st_dev,
        staging_inode=staging_stat.st_ino,
    )


def _prepare_diag5_publication(
    final_root: Path,
    *,
    repository_root: Path,
    successor_claim: Diag5SuccessorAuthorityClaim,
) -> Diag2Publication:
    """Create only the fixed authority-bound DIAG5 `.partial-claim` root."""

    repository = repository_root.resolve(strict=True)
    final = final_root.absolute()
    staging = successor_claim.expected_gpu_staging_root
    rollback = successor_claim.expected_gpu_rollback_root
    if (
        final != successor_claim.expected_gpu_output_root
        or staging != Path(f"{final}.partial-claim")
        or rollback != Path(f"{final}.partial-rollback")
    ):
        raise ValueError("DIAG5 publication roots differ from authority")
    parent = final.parent.resolve(strict=True)
    if parent == repository or parent.is_relative_to(repository):
        raise ValueError("DIAG5 output parent must be outside the repository")
    competing = tuple(parent.glob(f"{final.name}.partial-*"))
    if (
        os.path.lexists(final)
        or os.path.lexists(staging)
        or os.path.lexists(rollback)
        or competing
    ):
        raise FileExistsError("DIAG5 publication namespace is not absent")
    staging.mkdir(mode=0o755, exist_ok=False)
    metadata = staging.stat(follow_symlinks=False)
    return Diag2Publication(
        staging,
        final,
        "claim",
        staging_device=metadata.st_dev,
        staging_inode=metadata.st_ino,
    )


def _seal_and_sync_diag2_staging(staging_root: Path) -> None:
    """Durably seal every regular file and directory without following links."""

    directories: list[Path] = [staging_root]
    for path in staging_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("DIAG2 staging contains a symlink")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise ValueError("DIAG2 staging contains a special file")
        path.chmod(0o444)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(
        directories,
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one path without replacing an existing destination."""

    _rename_noreplace_at(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
    )


def _rename_noreplace_at(
    source_directory_descriptor: int,
    source_name: bytes,
    destination_directory_descriptor: int,
    destination_name: bytes,
) -> None:
    """Atomically rename descriptor-relative paths without replacement."""

    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_descriptor,
        source_name,
        destination_directory_descriptor,
        destination_name,
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                os.fsdecode(destination_name),
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fsdecode(destination_name),
        )


def _fsync_parent(path: Path) -> None:
    """Durably synchronize one already-resolved parent directory."""

    descriptor = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace_and_fsync_parent(source: Path, destination: Path) -> None:
    """Atomically rename one path without replacement and durably sync its parent."""

    _rename_noreplace(source, destination)
    _fsync_parent(destination)


def _retain_invalid_diag5_producer_bytes(directory: Path, payload: bytes) -> None:
    """Retain invalid producer bytes across a descriptor-bound no-replace rename."""

    source_name = b"producer.json"
    destination_name = b"invalid-producer.bin"
    directory_descriptor = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    producer_descriptor = -1
    try:
        producer_descriptor = os.open(
            source_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(producer_descriptor, remaining)
            if written == 0:
                raise OSError("invalid producer write made no progress")
            remaining = remaining[written:]
        os.fchmod(producer_descriptor, 0o444)
        os.fsync(producer_descriptor)
        opened = os.fstat(producer_descriptor)
        _rename_noreplace_at(
            directory_descriptor,
            source_name,
            directory_descriptor,
            destination_name,
        )
        destination = os.stat(
            destination_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        os.lseek(producer_descriptor, 0, os.SEEK_SET)
        retained = _diag4_bootstrap_descriptor_bytes(producer_descriptor)
        after = os.fstat(producer_descriptor)
        identity = (opened.st_dev, opened.st_ino, opened.st_size)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_mode & 0o777 != 0o444
            or identity != (after.st_dev, after.st_ino, after.st_size)
            or identity != (destination.st_dev, destination.st_ino, destination.st_size)
            or after.st_nlink != 1
            or destination.st_nlink != 1
            or retained != payload
            or hashlib.sha256(retained).digest() != hashlib.sha256(payload).digest()
        ):
            raise RuntimeError("DIAG5 invalid producer descriptor binding differs")
        os.fsync(directory_descriptor)
    finally:
        if producer_descriptor >= 0:
            os.close(producer_descriptor)
        os.close(directory_descriptor)


def _atomic_publish_diag2(publication: Diag2Publication) -> None:
    """Publish sealed staging with Linux RENAME_NOREPLACE, then fsync its parent."""

    _rename_noreplace_and_fsync_parent(publication.staging_root, publication.final_root)


def _rollback_diag4_final(publication: Diag2Publication) -> None:
    """Restore the exact bound final inode to its authority-bound partial name."""

    if publication.staging_device is None or publication.staging_inode is None:
        raise RuntimeError("DIAG4 publication omits its staging inode binding")
    if not os.path.lexists(publication.final_root) or os.path.lexists(
        publication.staging_root
    ):
        raise RuntimeError("DIAG4 rollback paths differ from the sealed final state")
    descriptor = os.open(
        publication.final_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        descriptor_stat = os.fstat(descriptor)
        final_stat = publication.final_root.stat(follow_symlinks=False)
        expected_identity = (publication.staging_device, publication.staging_inode)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != expected_identity or (
            final_stat.st_dev,
            final_stat.st_ino,
        ) != expected_identity:
            raise RuntimeError("DIAG4 final inode differs from its staging binding")
        _rename_noreplace(publication.final_root, publication.staging_root)
        partial_stat = publication.staging_root.stat(follow_symlinks=False)
        if (partial_stat.st_dev, partial_stat.st_ino) != expected_identity:
            raise RuntimeError("DIAG4 rollback partial inode differs")
        _fsync_parent(publication.staging_root)
        if os.path.lexists(publication.final_root):
            raise RuntimeError("DIAG4 rollback left the final path visible")
        rebound_stat = publication.staging_root.stat(follow_symlinks=False)
        if (rebound_stat.st_dev, rebound_stat.st_ino) != expected_identity:
            raise RuntimeError("DIAG4 rollback partial rebound")
    finally:
        os.close(descriptor)


def _raise_diag4_final_publication_failure(
    publication: Diag2Publication,
    successor_claim: Diag4SuccessorAuthorityClaim,
    *,
    reason: Diag4PhysicalPublicationReason,
    cause: BaseException,
    known_lifecycle: Diag4AuthorityLifecycle | None = None,
) -> Never:
    """Attempt one rollback, then raise with exact observed path/lease state."""

    lifecycle = known_lifecycle
    if lifecycle is None:
        try:
            lifecycle = diag4_authority_lifecycle(successor_claim)
        except BaseException:  # noqa: BLE001 - exact lease state may be unavailable.
            lifecycle = None
    try:
        _rollback_diag4_final(publication)
    except BaseException as rollback_error:
        raise Diag4RollbackHardPublicationError(
            reason,
            publication,
            cause,
            rollback_error,
            authority_lifecycle=lifecycle,
        ) from rollback_error
    raise Diag4HardPublicationError(
        reason,
        publication.staging_root,
        cause,
        staging_exists=os.path.lexists(publication.staging_root),
        final_exists=os.path.lexists(publication.final_root),
        authority_lifecycle=lifecycle,
    ) from cause


def _publish_diag2_supervisor_zero(
    staging_root: Path,
    observation: SupervisorGpuZeroObservation,
    *,
    stage: str,
    builder: Callable[..., dict[str, JsonValue]] = build_diag2_supervisor_zero_payload,
    schema_version: str = DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION,
) -> ArtifactRef:
    """Publish one dual-query observation through the receipt-owned schema."""

    if stage not in {"BEFORE_PREFLIGHT", "BEFORE_COLD"}:
        raise ValueError("DIAG2 supervisor-zero stage is invalid")
    slug = "before-preflight" if stage == "BEFORE_PREFLIGHT" else "before-cold"
    directory = staging_root / "supervisor"
    query_pairs = (
        ("gpu-inventory", observation.gpu_inventory_query),
        ("compute-apps", observation.compute_apps_query),
    )
    typed_queries: list[SupervisorQueryV2] = []
    for query_name, query in query_pairs:
        stdout_path = directory / f"{slug}-{query_name}.stdout.bin"
        stderr_path = directory / f"{slug}-{query_name}.stderr.bin"
        _publish_bytes(stdout_path, query.stdout)
        _publish_bytes(stderr_path, query.stderr)
        typed_queries.append(
            SupervisorQueryV2(
                argv=query.argv,
                query_executable_sha256=query.query_executable_sha256,
                launched=query.launched,
                timed_out=query.timed_out,
                returncode=query.returncode,
                stdout=_artifact_ref(
                    stdout_path,
                    staging_root,
                    f"raw-supervisor-{query_name}-stdout-v1",
                ),
                stderr=_artifact_ref(
                    stderr_path,
                    staging_root,
                    f"raw-supervisor-{query_name}-stderr-v1",
                ),
            )
        )
    payload = builder(
        stage=stage,
        captured_at_monotonic_ns=observation.captured_at_monotonic_ns,
        captured_at_unix_ns=observation.captured_at_unix_ns,
        supervisor_pid=observation.supervisor_pid,
        supervisor_start_ticks=observation.supervisor_start_ticks,
        gpu_uuid=observation.gpu_uuid,
        visible_device=observation.visible_device,
        gpu_inventory_query=typed_queries[0],
        compute_apps_query=typed_queries[1],
        matching_rows=tuple(
            {
                "pid": row.pid,
                "gpu_uuid": row.gpu_uuid,
                "used_memory_mib": row.used_memory_mib,
            }
            for row in observation.matching_rows
        ),
    )
    path = directory / f"{slug}.json"
    _publish_canonical_json(path, payload)
    return _artifact_ref(
        path,
        staging_root,
        schema_version,
    )


def _capture_diag2_supervisor_zero(
    environment: Mapping[str, str],
    *,
    query_executable_sha256: str | None = None,
) -> SupervisorGpuZeroObservation:
    """Bind a dual management query to this exact CPU-only supervisor process."""

    if (
        os.environ.get("JAX_PLATFORMS") != "cpu"
        or os.environ.get("JAX_PLATFORM_NAME") is not None
        or os.environ.get("JAX_COMPILATION_CACHE_DIR") is not None
        or os.environ.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
    ):
        raise ValueError("DIAG2 supervisor environment is not CPU-only")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible != GPU_UUID:
        raise ValueError("DIAG2 requires the frozen physical RTX 5090 UUID")
    identity = read_linux_process_identity(os.getpid())
    observation = capture_supervisor_gpu_zero(
        gpu_uuid=GPU_UUID,
        visible_device=visible,
        supervisor_pid=identity.pid,
        supervisor_start_ticks=identity.start_ticks,
        query_executable_sha256=query_executable_sha256,
    )
    identity_after = read_linux_process_identity(os.getpid())
    if (
        identity_after.pid != identity.pid
        or identity_after.start_ticks != identity.start_ticks
    ):
        raise RuntimeError("DIAG2 supervisor identity changed during GPU-zero capture")
    return observation


def _publish_diagnostic_artifact_manifest(campaign_root: Path) -> None:
    manifest_path = campaign_root / DIAGNOSTIC_MANIFEST_FILENAME
    payload = diagnostic_artifact_manifest_payload(campaign_root)
    declared = {
        str(entry["relative_path"])
        for entry in payload["entries"]
        if isinstance(entry, dict)
    }
    observed = {
        path.relative_to(campaign_root).as_posix()
        for path in campaign_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed != declared:
        raise ValueError("diagnostic artifact has undeclared or missing role paths")
    _publish_canonical_json(manifest_path, payload)


def _seal_campaign_tree(campaign_root: Path) -> None:
    for path in campaign_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("campaign artifact contains a symlink")
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (item for item in campaign_root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    campaign_root.chmod(0o555)


def _publish_campaign_artifact_manifest(campaign_root: Path) -> None:
    """Publish the exact closed set of campaign files before directory sealing."""

    manifest_path = campaign_root / CAMPAIGN_ARTIFACT_MANIFEST_FILENAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("campaign artifact manifest already exists")
    entries: list[dict[str, JsonValue]] = []
    for path in sorted(campaign_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("campaign artifact tree must not contain symlinks")
        if not path.is_file():
            continue
        payload = path.read_bytes()
        entries.append(
            {
                "relative_path": path.relative_to(campaign_root).as_posix(),
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    _publish_canonical_json(
        manifest_path,
        {
            "schema_version": CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA,
            "entries": entries,
        },
    )


def _enumerated_source_roots(
    repo_root: Path,
    native_extension: Path,
    *,
    additional_files: Sequence[str] = (),
) -> tuple[SourceRoot, ...]:
    """Select every live Python execution source plus the frozen NEQ contract."""

    completed = subprocess.run(
        (
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src/simsopt",
            "src/simsopt_jax",
            "src/simsopt_jax_adapters",
            "examples/jax",
            "benchmarks",
        ),
        check=True,
        capture_output=True,
    )
    relative_paths = {
        relative
        for field in completed.stdout.split(b"\0")
        if field
        for relative in (field.decode("utf-8"),)
        if not relative.startswith("benchmarks/") or relative.endswith(".py")
    }
    relative_paths.update(_OWNED_EXECUTION_FILES)
    relative_paths.update(additional_files)
    roots = tuple(
        SourceRoot(
            (
                "configuration"
                if relative.startswith("docs/")
                else (
                    "test"
                    if relative.startswith("tests/")
                    else (
                        "benchmark"
                        if relative.startswith("benchmarks/")
                        else "execution_source"
                    )
                )
            ),
            repo_root / relative,
            relative,
        )
        for relative in sorted(relative_paths)
    )
    return (
        *roots,
        SourceRoot(
            "native_extension", native_extension, f"src/{native_extension.name}"
        ),
    )


def prepare_execution_snapshot(
    campaign_root: Path,
    *,
    repo_root: Path,
    native_extension_path: Path,
) -> SnapshotPublication:
    """Create the campaign root once and seal its exact execution tree."""

    output = campaign_root.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing campaign path: {output}")
    if output.is_relative_to(repo_root.resolve(strict=True)):
        raise ValueError("campaign output must be outside the source repository")
    worktree = capture_worktree_identity(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    publication = publish_immutable_snapshot(
        output / SOURCE_SNAPSHOT_DIRECTORY,
        _enumerated_source_roots(repo_root, native_extension_path),
        worktree=worktree,
    )
    if capture_worktree_identity(repo_root) != worktree:
        raise ValueError("source changed during snapshot publication")
    return publication


def _prepare_diag2_snapshot(
    staging_root: Path,
    *,
    repo_root: Path,
    native_extension_path: Path,
) -> SnapshotPublication:
    """Publish the exact execution snapshot inside an existing sibling staging root."""

    worktree = capture_worktree_identity(repo_root)
    publication = publish_immutable_snapshot(
        staging_root / SOURCE_SNAPSHOT_DIRECTORY,
        _enumerated_source_roots(repo_root, native_extension_path),
        worktree=worktree,
    )
    if capture_worktree_identity(repo_root) != worktree:
        raise ValueError("source changed during DIAG2 snapshot publication")
    return publication


def _prepare_diag4_snapshot(
    staging_root: Path,
    *,
    native_extension_path: Path,
    successor_claim: Diag4SuccessorAuthorityClaim,
) -> SnapshotPublication:
    """Copy the exact GPU closure from the decisive sealed CPU snapshot."""

    cpu_snapshot = load_snapshot(
        DIAG4_CPU_QUALIFICATION_ROOT / SOURCE_SNAPSHOT_DIRECTORY,
        required_roles=DIAG4_CPU_SNAPSHOT_ROLES,
    )
    if native_extension_path != successor_claim.expected_native_extension_path:
        raise ValueError("DIAG4 live native extension path differs")
    native_payload = native_extension_path.read_bytes()
    if (
        len(native_payload) != successor_claim.expected_native_extension_size_bytes
        or hashlib.sha256(native_payload).hexdigest()
        != successor_claim.expected_native_extension_sha256
    ):
        raise ValueError("DIAG4 live native extension identity differs")
    publication = copy_immutable_snapshot(
        staging_root / SOURCE_SNAPSHOT_DIRECTORY,
        cpu_snapshot.root,
        source_required_roles=DIAG4_CPU_SNAPSHOT_ROLES,
        destination_required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
    )
    validate_diag4_successor_snapshot(publication, successor_claim)
    return publication


def _prepare_diag5_snapshot(
    staging_root: Path,
    *,
    successor_claim: Diag5SuccessorAuthorityClaim,
) -> SnapshotPublication:
    """Copy the sealed CPU-qualified closure without copying live GPU imports."""

    cpu_snapshot_root = (
        successor_claim.expected_cpu_qualification_root / SOURCE_SNAPSHOT_DIRECTORY
    )
    return copy_diag5_immutable_snapshot(
        staging_root / SOURCE_SNAPSHOT_DIRECTORY,
        cpu_snapshot_root,
        expected_sha256=successor_claim.expected_copied_native_sha256,
        expected_size_bytes=successor_claim.expected_copied_native_size_bytes,
        expected_native_relative_path=(
            successor_claim.expected_native_copy_relative_path
        ),
    )


def _diag5_native_binding_payload(
    successor_claim: Diag5SuccessorAuthorityClaim,
    *,
    role: str,
) -> dict[str, JsonValue]:
    """Serialize one authority-owned live native binding without rediscovery."""

    binding = (
        successor_claim.cpu_native_binding
        if role == "cpu"
        else successor_claim.gpu_native_binding
        if role == "gpu"
        else None
    )
    if binding is None:
        raise ValueError("DIAG5 native binding role differs")
    return {
        f"{role}_native_extension_path": str(binding.path),
        "native_extension_sha256": binding.sha256,
        "native_extension_size_bytes": binding.size_bytes,
        f"{role}_native_extension_link_count": binding.link_count,
        f"{role}_native_extension_device": binding.device,
        f"{role}_native_extension_inode": binding.inode,
    }


def _copy_diag5_predecessor_postmortem(
    staging_root: Path,
    successor_claim: Diag5SuccessorAuthorityClaim,
) -> ArtifactRef:
    """Copy the authority-bound predecessor control from the sealed CPU artifact."""

    reference = successor_claim.predecessor_postmortem
    source = successor_claim.expected_cpu_qualification_root / reference.relative_path
    payload = source.read_bytes()
    if (
        len(payload) != reference.size_bytes
        or hashlib.sha256(payload).hexdigest() != reference.sha256
    ):
        raise ValueError("DIAG5 predecessor postmortem source differs")
    destination = staging_root / reference.relative_path
    _publish_bytes(destination, payload)
    copied = _artifact_ref(
        destination,
        staging_root,
        reference.schema_version,
    )
    if copied != reference:
        raise ValueError("DIAG5 predecessor postmortem copy differs")
    return copied


def copy_validated_reference(
    source_root: Path,
    campaign_root: Path,
) -> Path:
    """Copy the complete sealed reference tree and revalidate exact bytes."""

    source = source_root.resolve(strict=True)
    validation = validate_native_equivalent_reference(source)
    destination = campaign_root / "native-reference"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing reference path: {destination}")
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    copied_validation = validate_native_equivalent_reference(destination)
    if copied_validation != validation:
        raise ValueError("copied native reference differs from source evidence")
    return destination


def build_child_invocation(
    publication: SnapshotPublication,
    *,
    campaign_root: Path,
    interpreter: Path,
    reference_root: Path,
    input_root: Path,
    sample: SampleName,
    environment: Mapping[str, str],
    preflight_only: bool = False,
    diagnostic_mode: DiagnosticChildMode | None = None,
    diag2: bool = False,
    diag4: bool = False,
    diag5: bool = False,
    expected_native_extension_path: Path | None = None,
    expected_native_extension_sha256: str | None = None,
    expected_native_extension_size_bytes: int | None = None,
    expected_native_extension_link_count: int | None = None,
) -> SnapshotChildInvocation:
    """Bind one isolated child to the sealed snapshot and requested sample."""

    if preflight_only and diagnostic_mode is not None:
        raise ValueError("campaign preflight and diagnostic child modes are exclusive")
    if (diag2 or diag4 or diag5) and diagnostic_mode is None:
        raise ValueError("successor route requires an explicit diagnostic child mode")
    if sum((diag2, diag4, diag5)) > 1:
        raise ValueError("successor child identities are exclusive")
    native_tuple = (
        expected_native_extension_path,
        expected_native_extension_sha256,
        expected_native_extension_size_bytes,
        expected_native_extension_link_count,
    )
    if diag5 != all(value is not None for value in native_tuple):
        raise ValueError("DIAG5 child requires one complete native identity tuple")

    executable = Path(os.path.abspath(interpreter))
    runner = publication.root / _ENTRYPOINT
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or not runner.is_file()
    ):
        raise ValueError("campaign interpreter or snapshot runner is unavailable")
    child_environment = dict(environment)
    child_environment["JAX_PLATFORMS"] = "cuda"
    child_environment["JAX_ENABLE_X64"] = "true"
    child_environment.pop("JAX_PLATFORM_NAME", None)
    child_environment[_SNAPSHOT_MANIFEST_ENV] = publication.manifest_sha256
    child_environment[_CAMPAIGN_ROOT_ENV] = str(campaign_root.resolve(strict=True))
    if diagnostic_mode is not None:
        child_environment[_DIAGNOSTIC_CHILD_ENV] = diagnostic_mode.value
        if diag2 or diag4 or diag5:
            child_environment["XLA_FLAGS"] = _command_buffer_disabled_xla_flags(
                child_environment.get("XLA_FLAGS", "")
            )
        if diag2:
            child_environment[_DIAG2_CHILD_ENV] = "1"
        else:
            child_environment.pop(_DIAG2_CHILD_ENV, None)
        if diag4:
            child_environment[_DIAG4_CHILD_ENV] = "1"
        else:
            child_environment.pop(_DIAG4_CHILD_ENV, None)
        if diag5:
            if (
                expected_native_extension_path is None
                or expected_native_extension_sha256 is None
                or expected_native_extension_size_bytes is None
                or expected_native_extension_link_count is None
            ):
                raise ValueError("DIAG5 native identity tuple is incomplete")
            native_path = expected_native_extension_path.resolve(strict=True)
            if (
                expected_native_extension_path != native_path
                or len(expected_native_extension_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_native_extension_sha256
                )
                or expected_native_extension_size_bytes <= 0
                or expected_native_extension_link_count <= 0
            ):
                raise ValueError("DIAG5 native identity tuple is invalid")
            child_environment[_DIAG5_CHILD_ENV] = "1"
            child_environment[_DIAG5_NATIVE_PATH_ENV] = str(native_path)
            child_environment[_DIAG5_NATIVE_SHA256_ENV] = (
                expected_native_extension_sha256
            )
            child_environment[_DIAG5_NATIVE_SIZE_ENV] = str(
                expected_native_extension_size_bytes
            )
            child_environment[_DIAG5_NATIVE_LINK_COUNT_ENV] = str(
                expected_native_extension_link_count
            )
        else:
            child_environment.pop(_DIAG5_CHILD_ENV, None)
            child_environment.pop(_DIAG5_NATIVE_PATH_ENV, None)
            child_environment.pop(_DIAG5_NATIVE_SHA256_ENV, None)
            child_environment.pop(_DIAG5_NATIVE_SIZE_ENV, None)
            child_environment.pop(_DIAG5_NATIVE_LINK_COUNT_ENV, None)
        child_environment.pop("JAX_COMPILATION_CACHE_DIR", None)
        if diag4 or diag5:
            child_environment.pop(TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT, None)
        else:
            child_environment[TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT] = str(
                TRACE_VIEWER_MAX_EVENTS
            )
        child_environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
        child_environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    argv = (
        str(executable),
        "-I",
        str(runner),
        "--snapshot-child",
        *(("--preflight-child",) if preflight_only else ()),
        *(
            ("--diagnostic-child", diagnostic_mode.value)
            if diagnostic_mode is not None
            else ()
        ),
        "--sample",
        sample.value,
        "--reference",
        str(reference_root.resolve(strict=True)),
        "--input-root",
        str(input_root.resolve(strict=True)),
    )
    return SnapshotChildInvocation(argv, publication.root, child_environment)


def run_sample_schedule(
    execute: SampleExecutor,
    cold_receipt_passes: Callable[[SupervisedSample], bool],
) -> tuple[SupervisedSample, ...]:
    """Run one cold, then all three warms iff the audited cold passes."""

    cold = execute(SampleName.COLD)
    if not cold_receipt_passes(cold):
        return (cold,)
    warms = tuple(execute(sample) for sample in SAMPLE_ORDER[1:])
    return (cold, *warms)


def run_diagnostic_schedule(
    execute_preflight: Callable[[], SupervisedSample],
    preflight_passes: Callable[[SupervisedSample], bool],
    execute_cold: Callable[[], SupervisedSample],
) -> tuple[SupervisedSample, ...]:
    """Run exactly one preflight and at most one independently authorized cold."""

    preflight = execute_preflight()
    if not preflight_passes(preflight):
        return (preflight,)
    return (preflight, execute_cold())


def _physical_gpu_identity(environment: Mapping[str, str]) -> tuple[str, int]:
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible != GPU_UUID:
        raise ValueError("NEQ-GNTR1 requires the frozen physical RTX 5090 UUID")
    output = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=uuid,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        env=dict(environment),
    ).stdout
    rows = [
        tuple(field.strip() for field in row.split(",")) for row in output.splitlines()
    ]
    matches = [row for row in rows if row[0] == GPU_UUID]
    if len(matches) != 1:
        raise ValueError("frozen GPU UUID is not uniquely visible to nvidia-smi")
    return matches[0][0], int(matches[0][1]) * 1024 * 1024


def _validate_parent_execution_policy(
    *,
    repo_root: Path | None,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    environment: Mapping[str, str],
) -> tuple[Path, Path, Path, Path, Path, str, int]:
    """Resolve all cheap parent policy gates before creating campaign output."""

    repository = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else repo_root.resolve(strict=True)
    )
    native_extension = Path(simsoptpp.__file__).resolve(strict=True)
    reference = reference_root.resolve(strict=True)
    inputs = input_root.resolve(strict=True)
    executable = Path(os.path.abspath(interpreter))
    if (
        not reference.is_dir()
        or not inputs.is_dir()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError("reference, input root, or interpreter parent policy failed")
    gpu_uuid, memory_bytes = _physical_gpu_identity(environment)
    return (
        repository,
        native_extension,
        reference,
        inputs,
        executable,
        gpu_uuid,
        memory_bytes,
    )


def _wait_and_read_killed_child(
    child: subprocess.Popen[bytes],
) -> tuple[bytes, bytes]:
    """Reap a killed child independently of communicate and drain its pipes."""

    try:
        child.wait()
    except Exception:  # noqa: BLE001 - waitpid is the final OS-owned fallback.
        while True:
            try:
                waited_pid, wait_status = os.waitpid(child.pid, 0)
                break
            except InterruptedError:
                continue
            except ChildProcessError:
                waited_pid = child.pid
                wait_status = None
                child.poll()
                break
        if (
            waited_pid == child.pid
            and wait_status is not None
            and child.returncode is None
        ):
            child.returncode = os.waitstatus_to_exitcode(wait_status)
    try:
        stdout = b"" if child.stdout is None else child.stdout.read()
    except (AttributeError, OSError, ValueError):
        stdout = b""
    try:
        stderr = b"" if child.stderr is None else child.stderr.read()
    except (AttributeError, OSError, ValueError):
        stderr = b""
    return stdout or b"", stderr or b""


def _kill_and_reap_child(
    child: subprocess.Popen[bytes],
) -> tuple[bytes, bytes]:
    """Force one launched child to a reaped state after supervision failure."""

    try:
        child.kill()
    except OSError:
        try:
            os.kill(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        return child.communicate()
    except Exception:  # noqa: BLE001 - permanent stream failure still requires reap.
        if child.poll() is None:
            try:
                os.kill(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            return child.communicate()
        except Exception:  # noqa: BLE001 - wait/read is the terminal fallback.
            return _wait_and_read_killed_child(child)


def _supervision_exception_sample(
    sample: SampleName,
    invocation: SnapshotChildInvocation,
    child: subprocess.Popen[bytes],
    *,
    started_ns: int,
    error: Exception,
    child_start_time_ticks: int,
    retain_raw_evidence: bool,
    failure_prefix: str,
) -> SupervisedSample:
    """Retain exact failure evidence after killing and reaping a launched child."""

    stdout, stderr = _kill_and_reap_child(child)
    return SupervisedSample(
        sample=sample,
        terminal_status=ChildTerminalStatus.MONITOR_FAILURE,
        child_pid=child.pid,
        child_start_time_ticks=child_start_time_ticks,
        process_seconds=(time.perf_counter_ns() - started_ns) / 1.0e9,
        producer={},
        memory=None,
        failure_reasons=(
            f"{failure_prefix}:{type(error).__name__}:{_sha256(str(error).encode())}",
        ),
        process_diagnostics=_bounded_process_diagnostics(
            invocation,
            returncode=child.returncode,
            stdout=stdout,
            stderr=stderr,
            monitor_error=error,
        ),
        observed_child_argv=invocation.argv if retain_raw_evidence else None,
        stdout=stdout if retain_raw_evidence else None,
        stderr=stderr if retain_raw_evidence else None,
        memory_samples=() if retain_raw_evidence else None,
    )


def supervise_sample(
    sample: SampleName,
    invocation: SnapshotChildInvocation,
    *,
    gpu_uuid: str,
    physical_memory_bytes: int,
    timeout_seconds: float = PROCESS_TIMEOUT_SECONDS,
    retain_raw_evidence: bool = False,
) -> SupervisedSample:
    """Supervise one exact PID/UUID-bound child and retain every outcome."""

    started_ns = time.perf_counter_ns()
    child = subprocess.Popen(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        monitor = BoundProcessGpuMemoryMonitor(
            gpu_uuid=gpu_uuid,
            provider_pid=child.pid,
            expected_argv=invocation.argv,
        )
        monitor.start()
    except Exception as error:  # noqa: BLE001 - monitor failure is terminal evidence.
        return _supervision_exception_sample(
            sample,
            invocation,
            child,
            started_ns=started_ns,
            error=error,
            child_start_time_ticks=0,
            retain_raw_evidence=retain_raw_evidence,
            failure_prefix="MONITOR_BINDING",
        )
    try:
        identity = monitor.identity
    except Exception as error:  # noqa: BLE001 - launched child must be reaped.
        return _supervision_exception_sample(
            sample,
            invocation,
            child,
            started_ns=started_ns,
            error=error,
            child_start_time_ticks=0,
            retain_raw_evidence=retain_raw_evidence,
            failure_prefix="MONITOR_BINDING",
        )
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr = _kill_and_reap_child(child)
    except Exception as error:  # noqa: BLE001 - launched child must be reaped.
        return _supervision_exception_sample(
            sample,
            invocation,
            child,
            started_ns=started_ns,
            error=error,
            child_start_time_ticks=identity.start_ticks,
            retain_raw_evidence=retain_raw_evidence,
            failure_prefix="SUPERVISION_IO",
        )
    try:
        measurement = monitor.finish()
        memory = bound_gpu_memory_payload(
            monitor,
            measurement,
            parent_pid=os.getpid(),
            physical_device_memory_bytes=physical_memory_bytes,
            runtime_argv=invocation.argv[2:],
            schema_version=MEMORY_SCHEMA_VERSION,
        )
        monitor_reason: str | None = None
        monitor_error: Exception | None = None
        memory_samples = (
            tuple(
                RawGpuMemorySample(
                    sampled_at_unix_ns=sample.sampled_at_unix_ns,
                    used_memory_mib=sample.used_memory_mib,
                )
                for sample in measurement.samples
            )
            if retain_raw_evidence
            else ()
        )
    except Exception as error:  # noqa: BLE001 - terminal monitor evidence is required.
        memory = None
        monitor_reason = f"{type(error).__name__}:{_sha256(str(error).encode())}"
        monitor_error = error
        memory_samples = ()
    elapsed = (time.perf_counter_ns() - started_ns) / 1.0e9
    if timed_out:
        status = ChildTerminalStatus.TIMEOUT
        producer: dict[str, JsonValue] = {}
        reasons = ("PROCESS_TIMEOUT_900_SECONDS",)
    elif child.returncode != 0:
        status = ChildTerminalStatus.CRASH
        producer = {}
        reasons = (f"CHILD_EXIT_{child.returncode}:{_sha256(stderr)}",)
    else:
        try:
            decoded = load_canonical_json_bytes(stdout)
            if not isinstance(decoded, dict):
                raise TypeError("worker output must be a JSON object")
            producer = decoded
            if producer.get("execution_status") == "TIMEOUT":
                status = ChildTerminalStatus.TIMEOUT
                raw_reasons = producer.get("failure_reasons")
                reasons = (
                    tuple(str(reason) for reason in raw_reasons)
                    if isinstance(raw_reasons, list)
                    else ("SOLVE_TIMEOUT_360_SECONDS",)
                )
            elif producer.get("execution_status") in {
                "COMPILE_FAILURE",
                "COMPILE_OOM",
            }:
                status = ChildTerminalStatus.COMPILE_FAILURE
                raw_reasons = producer.get("failure_reasons")
                reasons = (
                    tuple(str(reason) for reason in raw_reasons)
                    if isinstance(raw_reasons, list)
                    else ("COMPILE_FAILURE",)
                )
            elif producer.get("execution_status") == "TRACE_NORMALIZATION_FAILED":
                status = ChildTerminalStatus.COMPLETE
                raw_reasons = producer.get("failure_reasons")
                reasons = (
                    tuple(str(reason) for reason in raw_reasons)
                    if isinstance(raw_reasons, list)
                    else ("TRACE_NORMALIZATION_FAILED",)
                )
            else:
                status = ChildTerminalStatus.COMPLETE
                reasons = ()
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            status = ChildTerminalStatus.PROTOCOL_FAILURE
            producer = {}
            reasons = (
                f"WORKER_PROTOCOL:{type(error).__name__}:{_sha256(stdout + stderr)}",
            )
    if monitor_reason is not None:
        if status is ChildTerminalStatus.COMPLETE:
            status = ChildTerminalStatus.MONITOR_FAILURE
        reasons = (*reasons, f"MONITOR:{monitor_reason}")
    return SupervisedSample(
        sample,
        status,
        child.pid,
        identity.start_ticks,
        elapsed,
        producer,
        memory,
        reasons,
        process_diagnostics=(
            _bounded_process_diagnostics(
                invocation,
                returncode=child.returncode,
                stdout=stdout,
                stderr=stderr,
                monitor_error=monitor_error,
            )
            if retain_raw_evidence
            or status is not ChildTerminalStatus.COMPLETE
            or monitor_error is not None
            else None
        ),
        observed_child_argv=identity.argv if retain_raw_evidence else None,
        stdout=stdout if retain_raw_evidence else None,
        stderr=stderr if retain_raw_evidence else None,
        memory_samples=memory_samples if retain_raw_evidence else None,
    )


def supervise_diag2_sample(
    sample: SampleName,
    invocation: SnapshotChildInvocation,
    *,
    mode: DiagnosticChildMode,
    gpu_uuid: str,
    physical_memory_bytes: int,
    validate_producer: Callable[..., Mapping[str, JsonValue]],
    timeout_seconds: float = PROCESS_TIMEOUT_SECONDS,
) -> DiagnosticSupervisedSampleV2:
    """Supervise one child and retain a producer only after its exact v2 parser."""

    process_started_monotonic_ns = time.perf_counter_ns()
    if (
        invocation.environment.get(_DIAG4_CHILD_ENV) == "1"
        or invocation.environment.get(_DIAG5_CHILD_ENV) == "1"
    ):
        invocation = replace(
            invocation,
            environment={
                **invocation.environment,
                _DIAG4_PROCESS_STARTED_ENV: str(process_started_monotonic_ns),
            },
        )
    try:
        outcome = supervise_sample(
            sample,
            invocation,
            gpu_uuid=gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
            timeout_seconds=timeout_seconds,
            retain_raw_evidence=True,
        )
    except OSError as error:
        process_stopped_monotonic_ns = time.perf_counter_ns()
        failure_reason = (
            f"CHILD_LAUNCH_FAILED:{type(error).__name__}:{_sha256(str(error).encode())}"
        )
        return DiagnosticSupervisedSampleV2(
            sample=sample,
            launched=False,
            terminal_status=ChildTerminalStatus.CRASH,
            child_pid=0,
            child_start_time_ticks=0,
            process_seconds=0.0,
            producer=None,
            producer_absence_reason=AbsenceReason.CHILD_LAUNCH_FAILED,
            selected_failure_reason=FailureReasonCodeV2.CHILD_LAUNCH_FAILED,
            memory=None,
            raw_failure_reasons=(failure_reason,),
            observed_child_argv=None,
            stdout=b"",
            stderr=b"",
            memory_samples=(),
            process_started_monotonic_ns=process_started_monotonic_ns,
            process_stopped_monotonic_ns=process_stopped_monotonic_ns,
        )
    process_stopped_monotonic_ns = time.perf_counter_ns()
    monitor_failure_kind = (
        MonitorFailureKind.BINDING
        if outcome.child_start_time_ticks == 0
        else (
            MonitorFailureKind.FINALIZATION
            if outcome.memory is None
            else MonitorFailureKind.NONE
        )
    )
    process_returncode = (outcome.process_diagnostics or {}).get("returncode")
    terminal_status = (
        ChildTerminalStatus.COMPLETE
        if monitor_failure_kind is MonitorFailureKind.FINALIZATION
        and process_returncode == 0
        and outcome.terminal_status
        in {
            ChildTerminalStatus.MONITOR_FAILURE,
            ChildTerminalStatus.COMPILE_FAILURE,
        }
        else outcome.terminal_status
    )

    absence_reason: AbsenceReason | None
    selected_failure: FailureReasonCodeV2 | None
    if outcome.terminal_status is ChildTerminalStatus.TIMEOUT:
        absence_reason = AbsenceReason.CHILD_TIMEOUT
        selected_failure = FailureReasonCodeV2.CHILD_TIMEOUT
    elif outcome.memory is None and outcome.child_start_time_ticks == 0:
        absence_reason = AbsenceReason.MONITOR_BINDING_FAILED
        selected_failure = FailureReasonCodeV2.MONITOR_BINDING_FAILED
    elif (
        outcome.memory is None
        and outcome.terminal_status is not ChildTerminalStatus.COMPLETE
        and not outcome.producer
    ):
        absence_reason = AbsenceReason.MONITOR_FINALIZATION_FAILED
        selected_failure = FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
    elif outcome.terminal_status is ChildTerminalStatus.CRASH:
        absence_reason = AbsenceReason.CHILD_EXIT_NONZERO
        selected_failure = FailureReasonCodeV2.CHILD_EXIT_NONZERO
    elif not outcome.producer:
        absence_reason = (
            AbsenceReason.MONITOR_BINDING_FAILED
            if outcome.terminal_status is ChildTerminalStatus.MONITOR_FAILURE
            and outcome.child_start_time_ticks == 0
            else AbsenceReason.PRODUCER_DECODE_FAILED
        )
        selected_failure = (
            FailureReasonCodeV2.MONITOR_BINDING_FAILED
            if absence_reason is AbsenceReason.MONITOR_BINDING_FAILED
            else FailureReasonCodeV2.PRODUCER_DECODE_FAILED
        )
    else:
        try:
            producer = dict(validate_producer(outcome.producer, mode=mode.value))
        except (KeyError, TypeError, ValueError):
            absence_reason = AbsenceReason.PRODUCER_SCHEMA_INVALID
            selected_failure = FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID
        else:
            execution_status = producer.get("execution_status")
            if outcome.memory is None:
                selected_failure = FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
            elif execution_status == "COMPILE_OOM":
                selected_failure = FailureReasonCodeV2.CHILD_COMPILE_OOM
            elif execution_status == "COMPILE_FAILURE":
                selected_failure = FailureReasonCodeV2.CHILD_COMPILE_FAILED
            elif execution_status == "TRACE_NORMALIZATION_FAILED":
                selected_failure = FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED
            elif (
                outcome.memory is not None
                and float(outcome.memory.get("peak_memory_fraction", 1.0)) >= 0.8
            ):
                selected_failure = FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED
            else:
                selected_failure = None
            return DiagnosticSupervisedSampleV2(
                sample=sample,
                launched=True,
                terminal_status=terminal_status,
                child_pid=outcome.child_pid,
                child_start_time_ticks=outcome.child_start_time_ticks,
                process_seconds=outcome.process_seconds,
                producer=producer,
                producer_absence_reason=None,
                selected_failure_reason=selected_failure,
                memory=outcome.memory,
                raw_failure_reasons=outcome.failure_reasons,
                observed_child_argv=outcome.observed_child_argv,
                stdout=outcome.stdout or b"",
                stderr=outcome.stderr or b"",
                memory_samples=outcome.memory_samples or (),
                process_diagnostics=outcome.process_diagnostics,
                process_started_monotonic_ns=process_started_monotonic_ns,
                process_stopped_monotonic_ns=process_stopped_monotonic_ns,
                monitor_failure_kind=monitor_failure_kind,
            )
    return DiagnosticSupervisedSampleV2(
        sample=sample,
        launched=True,
        terminal_status=terminal_status,
        child_pid=outcome.child_pid,
        child_start_time_ticks=outcome.child_start_time_ticks,
        process_seconds=outcome.process_seconds,
        producer=None,
        producer_absence_reason=absence_reason,
        selected_failure_reason=selected_failure,
        memory=outcome.memory,
        raw_failure_reasons=outcome.failure_reasons,
        observed_child_argv=outcome.observed_child_argv,
        stdout=outcome.stdout or b"",
        stderr=outcome.stderr or b"",
        memory_samples=outcome.memory_samples or (),
        process_diagnostics=outcome.process_diagnostics,
        process_started_monotonic_ns=process_started_monotonic_ns,
        process_stopped_monotonic_ns=process_stopped_monotonic_ns,
        monitor_failure_kind=monitor_failure_kind,
    )


def _reference_array(reference_root: Path, name: str) -> np.ndarray:
    document = load_canonical_json_bytes(
        (reference_root / REFERENCE_FILENAME).read_bytes()
    )
    if not isinstance(document, dict) or not isinstance(document.get("evidence"), dict):
        raise TypeError("usable native reference evidence is absent")
    arrays = document["evidence"].get("arrays")
    if not isinstance(arrays, dict) or not isinstance(arrays.get(name), dict):
        raise TypeError(f"reference array {name!r} is absent")
    relative = arrays[name].get("relative_path")
    if not isinstance(relative, str):
        raise TypeError(f"reference array {name!r} path is invalid")
    path = (reference_root / relative).resolve(strict=True)
    if not path.is_relative_to(reference_root.resolve(strict=True)):
        raise ValueError("reference array escapes the sealed artifact")
    with path.open("rb") as stream:
        array = np.load(stream, allow_pickle=False)
    return np.asarray(array)


def _derive_diag2_policy_authority(
    reference_root: Path,
) -> Diag2PolicyAuthorityValues:
    """Reconstruct the frozen policy using reference bytes and NumPy only."""

    validation = validate_native_equivalent_reference(reference_root)
    if not validation.usable:
        raise ValueError("copied native-equivalent reference is not usable")
    document = load_canonical_json_bytes(
        (reference_root / REFERENCE_FILENAME).read_bytes()
    )
    if not isinstance(document, dict) or not isinstance(document.get("evidence"), dict):
        raise TypeError("usable native reference evidence is absent")
    observables = document["evidence"].get("observables")
    if not isinstance(observables, dict):
        raise TypeError("native reference observables are absent")
    reference_volume = observables.get("volume")
    if type(reference_volume) is not float or not np.isfinite(reference_volume):
        raise TypeError("native reference volume must be a finite float")
    native_raw_equalities = np.ascontiguousarray(
        _reference_array(reference_root, "raw_equalities"),
        dtype="<f8",
    )
    if native_raw_equalities.shape != (255,) or not np.all(
        np.isfinite(native_raw_equalities)
    ):
        raise ValueError("native raw equalities must be a finite FP64 255-vector")
    volume_target, constraint_inverse_scale = _diag2_constraint_scale(
        reference_volume,
        native_raw_equalities,
    )

    policy = NativeEquivalentQualityPolicy(
        native_raw_equalities=native_raw_equalities,
        native_raw_equalities_sha256=exact_numeric_tree_sha256(native_raw_equalities),
        constraint_inverse_scale=constraint_inverse_scale,
    )
    if policy.policy_sha256 != _DIAG2_POLICY_SHA256:
        raise ValueError("pure-NumPy policy authority differs from the frozen identity")
    return Diag2PolicyAuthorityValues(
        reference_volume=reference_volume,
        volume_target=volume_target,
        native_raw_equalities=native_raw_equalities,
        constraint_inverse_scale=constraint_inverse_scale,
        policy_sha256=policy.policy_sha256,
    )


def _diag2_constraint_scale(
    reference_volume: float,
    native_raw_equalities: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Apply the frozen 254-Boozer-plus-volume scaling identity in FP64."""

    volume_target = float(reference_volume - native_raw_equalities[254])
    constraint_inverse_scale = np.empty(255, dtype="<f8")
    constraint_inverse_scale[:254] = 1.0 / np.sqrt(np.float64(254.0))
    constraint_inverse_scale[254] = 1.0 / abs(volume_target)
    if (
        volume_target.hex() != _DIAG2_VOLUME_TARGET_HEX
        or float(constraint_inverse_scale[0]).hex() != _DIAG2_BOOZER_SCALE_HEX
        or float(constraint_inverse_scale[254]).hex() != _DIAG2_VOLUME_SCALE_HEX
        or _sha256(constraint_inverse_scale.tobytes(order="C")) != _DIAG2_SCALE_SHA256
    ):
        raise ValueError("pure-NumPy constraint scale differs from the frozen identity")
    return volume_target, constraint_inverse_scale


def _reference_input_fingerprints(reference_root: Path) -> tuple[str, str]:
    document = load_canonical_json_bytes(
        (reference_root / REFERENCE_FILENAME).read_bytes()
    )
    if not isinstance(document, dict) or not isinstance(
        document.get("authority_manifest"), dict
    ):
        raise TypeError("reference authority manifest is absent")
    relative = document["authority_manifest"].get("relative_path")
    if not isinstance(relative, str):
        raise TypeError("reference authority path is invalid")
    authority_path = (reference_root / relative).resolve(strict=True)
    if not authority_path.is_relative_to(reference_root.resolve(strict=True)):
        raise ValueError("reference authority escapes the sealed artifact")
    authority = load_canonical_json_bytes(authority_path.read_bytes())
    if not isinstance(authority, dict):
        raise TypeError("reference authority must be an object")
    input_fingerprint = authority.get("input_fingerprint")
    configuration_fingerprint = authority.get("configuration_fingerprint")
    if not isinstance(input_fingerprint, str) or not isinstance(
        configuration_fingerprint, str
    ):
        raise TypeError("reference input fingerprints are absent")
    return input_fingerprint, configuration_fingerprint


def _prepare_worker_for_route(
    reference_root: Path,
    input_root: Path,
    prepare_route: NeqRoutePreparer,
) -> PreparedWorker:
    """Construct the common frozen inputs, then compile the selected route."""

    validation = validate_native_equivalent_reference(reference_root)
    if not validation.usable:
        raise ValueError("native-equivalent reference is not usable")
    bootstrap = build_single_stage_fullspace_bootstrap()
    raw_equalities = _reference_array(reference_root, "raw_equalities")
    native_reference_state = _reference_array(reference_root, "state")
    if (
        native_reference_state.shape != (716,)
        or native_reference_state.dtype != np.float64
    ):
        raise TypeError("native reference state must be an FP64 716-vector")
    scaling = fullspace_scaling_from_bootstrap(bootstrap.z0, bootstrap.problem)
    host_constraint_scale = np.asarray(
        jax.device_get(scaling.constraint_inverse_scale), dtype=np.float64
    )
    policy = NativeEquivalentQualityPolicy(
        native_raw_equalities=raw_equalities,
        native_raw_equalities_sha256=exact_numeric_tree_sha256(raw_equalities),
        constraint_inverse_scale=host_constraint_scale,
    )
    bundle, arrays = read_input_bundle(input_root)
    expected_input, expected_configuration = _reference_input_fingerprints(
        reference_root
    )
    if (
        bundle.input_fingerprint != expected_input
        or bundle.configuration_fingerprint != expected_configuration
    ):
        raise ValueError("native audit input bundle differs from sealed reference")
    native_runtime = build_native_single_stage_endpoint_runtime(
        bundle,
        arrays,
        bootstrap,
    )
    route = prepare_route(bootstrap.problem, bootstrap.z0, bootstrap.z0, policy)
    return PreparedWorker(route, native_runtime, native_reference_state)


def _prepare_worker(reference_root: Path, input_root: Path) -> PreparedWorker:
    return _prepare_worker_for_route(reference_root, input_root, prepare_neq_gntr1)


def _prepare_diag4_worker(reference_root: Path, input_root: Path) -> PreparedWorker:
    """Compile the authoritative safeguarded GNTR3 route."""

    return _prepare_worker_for_route(reference_root, input_root, prepare_neq_gntr3)


def _validate_diag4_prepared_route(prepared: object) -> PreparedNeqGntr3:
    """Bind DIAG4 to the exact safeguarded GNTR3 source identity and options."""

    if not isinstance(prepared, PreparedNeqGntr3):
        raise TypeError("DIAG4 did not compile the GNTR3 route")
    if (
        prepared.identity.schema_version != NEQ_GNTR3_SCHEMA_VERSION
        or prepared.identity.route != NEQ_GNTR3_ROUTE
        or DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION != NEQ_GNTR3_SCHEMA_VERSION
        or DIAG4_NUMERICAL_ROUTE != NEQ_GNTR3_ROUTE
    ):
        raise ValueError("DIAG4 GNTR3 route or schema identity differs")
    safeguard = prepared.options.enable_step_bound_safeguard
    if (
        type(safeguard) is not bool
        or not safeguard
        or prepared.options != NEQ_GNTR3_OPTIONS
    ):
        raise ValueError("DIAG4 GNTR3 safeguard options differ")
    return prepared


def _prepare_diagnostic_worker(
    reference_root: Path,
    input_root: Path,
    *,
    diag4: bool = False,
) -> PreparedDiagnosticWorker:
    """Compile one production loop plus its separately synchronized audits."""

    def compile_diagnostics(worker: PreparedWorker) -> PreparedDiagnosticWorker:
        prepared = worker.route
        terminal = prepare_neq_terminal_endpoint_diagnostics(
            prepared.problem,
            prepared.scaling,
            prepared.initial_optimizer_coordinates,
            jnp.zeros(
                (prepared.policy.equality_size,),
                dtype=prepared.initial_optimizer_coordinates.dtype,
            ),
        )
        ledger_size = prepared.options.maximum_accepted_steps + 1
        accepted_quality = prepare_neq_accepted_quality_diagnostics(
            prepared.problem,
            prepared.scaling,
            prepared.policy,
            jnp.zeros(
                (ledger_size, prepared.policy.state_size),
                dtype=prepared.initial_optimizer_coordinates.dtype,
            ),
            jnp.zeros((ledger_size,), dtype=jnp.bool_),
        )
        return PreparedDiagnosticWorker(worker, accepted_quality, terminal)

    if diag4:
        return compile_diagnostics(_prepare_diag4_worker(reference_root, input_root))
    with trace_session():
        return compile_diagnostics(_prepare_worker(reference_root, input_root))


def _compiled_python_callback_count(
    prepared: PreparedNeqGntr1 | PreparedNeqGntr2 | PreparedNeqGntr3,
) -> int:
    """Count callback primitives in the exact compiled timed executable text."""

    as_text = getattr(prepared._run_loop, "as_text", None)
    if not callable(as_text):
        raise TypeError("compiled timed executable does not expose inspectable text")
    executable_text = str(as_text()).lower()
    return sum(
        executable_text.count(token)
        for token in (
            "debug_callback",
            "host_callback",
            "io_callback",
            "xla_python_cpu_callback",
        )
    )


def _compiled_diagnostic_callback_count(prepared: PreparedDiagnosticWorker) -> int:
    """Count callbacks in the loop, finalizer, ledger map, and terminal graph."""

    executables = (
        prepared.worker.route._run_loop,
        prepared.worker.route._finalize,
        prepared.worker.route._map_ledger,
        prepared.accepted_quality._run_quality,
        prepared.terminal._run_endpoint,
    )
    count = 0
    for executable in executables:
        as_text = getattr(executable, "as_text", None)
        if not callable(as_text):
            raise TypeError("compiled diagnostic executable lacks inspectable text")
        text = str(as_text()).lower()
        count += sum(
            text.count(token)
            for token in (
                "debug_callback",
                "host_callback",
                "io_callback",
                "xla_python_cpu_callback",
            )
        )
    return count


def _diagnostic_history_payload(
    loop_result: object, *, diag5: bool = False
) -> dict[str, JsonValue]:
    """Map the fixed device history through the receipt-owned row constructor."""

    builder = (
        diag5_history_evidence_from_arrays if diag5 else history_evidence_from_arrays
    )
    return builder(
        loop_result.history,
        quality_latch=bool(np.asarray(loop_result.device_quality_candidate_reached)),
        first_quality_attempt=int(np.asarray(loop_result.first_quality_attempt)),
        first_quality_accepted_step=int(
            np.asarray(loop_result.first_quality_accepted_step)
        ),
    )


def _publish_diagnostic_terminal(
    cold_root: Path,
    campaign_root: Path,
    diagnostic: object,
    quality_replay: object,
    terminal_evidence: object,
    native_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
    bootstrap_anchor: np.ndarray,
    variable_scale: np.ndarray,
    objective_weights: tuple[float, ...],
    *,
    terminal_seconds: float,
    logical_cold_root: Path | None = None,
    numerical_identity: NativeEquivalentNumericalIdentity | None = None,
) -> tuple[ArtifactRef, dict[str, ArtifactRef]]:
    """Publish every raw terminal array and its receipt-owned scalar ledger."""

    base = diagnostic.base_result
    raw = terminal_evidence.raw_endpoint
    endpoint = base.endpoint
    transpose = endpoint.transpose_certificate
    arrays = {
        "optimizer_coordinates": base.optimizer_result.optimizer_coordinates,
        "physical_state": endpoint.physical_state,
        "raw_equalities": endpoint.raw_equalities,
        "scaled_equalities": endpoint.scaled_equalities,
        "objective_gradient": endpoint.objective_gradient,
        "multipliers": base.optimizer_result.multipliers,
        "raw_stationarity": raw.raw_stationarity_residual,
        "native_equalities": native_equalities,
        "constraint_inverse_scale": constraint_inverse_scale,
        "accepted_optimizer_ledger": base.loop_result.accepted_optimizer_coordinates,
        "accepted_physical_ledger": base.accepted_physical_coordinates,
        "accepted_mask": base.accepted_state_mask,
        "accepted_quality_objectives": quality_replay.objectives,
        "accepted_quality_raw_equalities": quality_replay.raw_equalities,
        "accepted_quality_scaled_equalities": quality_replay.scaled_equalities,
        "accepted_quality_mask": quality_replay.accepted_state_mask,
        "accepted_quality_coordinates_finite": quality_replay.coordinates_finite,
        "accepted_quality_objective_finite": quality_replay.objective_finite,
        "accepted_quality_raw_equalities_finite": (
            quality_replay.raw_equalities_finite
        ),
        "accepted_quality_scaled_equalities_finite": (
            quality_replay.scaled_equalities_finite
        ),
        "accepted_quality_objective_satisfied": quality_replay.objective_satisfied,
        "accepted_quality_component_bounds_satisfied": (
            quality_replay.component_bounds_satisfied
        ),
        "accepted_quality_scaled_feasibility_satisfied": (
            quality_replay.scaled_feasibility_satisfied
        ),
        "accepted_quality_satisfied": quality_replay.quality_satisfied,
        "authoritative_objective_gradient": (
            terminal_evidence.authoritative_objective_gradient
        ),
        "bootstrap_anchor": bootstrap_anchor,
        "constraint_jacobian": base.optimizer_result.constraint_jacobian,
        "objective_residual_vector": terminal_evidence.objective_residual_vector,
        "reconstructed_objective_gradient": (
            terminal_evidence.reconstructed_objective_gradient
        ),
        "transpose_equality_probe": transpose.equality_probe,
        "transpose_jvp_action": transpose.jvp_action,
        "transpose_state_probe": transpose.state_probe,
        "transpose_vjp_action": transpose.vjp_action,
        "variable_scale": variable_scale,
    }
    references: dict[str, ArtifactRef] = {}
    array_payloads: dict[str, Mapping[str, JsonValue]] = {}
    for name in sorted(DIAGNOSTIC_ARRAY_SPECS):
        dtype, _shape = DIAGNOSTIC_ARRAY_SPECS[name]
        values = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.dtype(dtype)))
        path = cold_root / "arrays" / f"{name}.npy"
        _publish_npy(path, values)
        reference = _artifact_ref_at(
            path,
            campaign_root,
            (logical_cold_root or cold_root) / "arrays" / path.name,
            f"{DIAGNOSTIC_SCHEMA_VERSION}-array-{name}",
        )
        references[name] = reference
        array_payloads[name] = array_evidence_payload(
            reference=reference, name=name, values=values
        )
    terms = endpoint.evaluation.raw_terms
    objective_terms = {
        name: float(np.asarray(getattr(terms, name)))
        for name in ("non_qs", "residual", "iota", "major_radius", "length")
    }
    kkt_code = int(np.asarray(diagnostic.raw_kkt_status))
    kkt_status = KktStatus.AVAILABLE if kkt_code == 0 else KktStatus.NONFINITE
    final_certificate = base.optimizer_result.final_certificate
    terminal_payload = terminal_numerical_payload(
        arrays=array_payloads,
        objective=float(np.asarray(endpoint.evaluation.weighted_total)),
        objective_terms=objective_terms,
        objective_weights={
            name.removesuffix("_weight"): weight
            for name, weight in zip(
                (
                    "non_qs_weight",
                    "residual_weight",
                    "iota_weight",
                    "major_radius_weight",
                    "length_weight",
                ),
                objective_weights,
                strict=True,
            )
        },
        reconstructed_objective=float(
            np.asarray(terminal_evidence.reconstructed_objective)
        ),
        authoritative_objective=float(
            np.asarray(terminal_evidence.authoritative_objective)
        ),
        final_certificate={
            name: float(np.asarray(getattr(final_certificate, name)))
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
    if numerical_identity is not None:
        endpoint_state_sha256 = array_payloads["physical_state"].get("content_sha256")
        if not isinstance(endpoint_state_sha256, str):
            raise TypeError("DIAG4 physical-state content hash differs")
        raw_observables = endpoint.evaluation.observables
        observables = {
            "iota": float(np.asarray(raw_observables.iota)),
            "G": float(np.asarray(raw_observables.G)),
            "volume": float(np.asarray(raw_observables.volume)),
            "major_radius": float(np.asarray(raw_observables.major_radius)),
            "total_length": float(np.asarray(raw_observables.total_length)),
            "non_qs_ratio": float(np.asarray(raw_observables.non_qs_ratio)),
            "boozer_residual_value": float(
                np.asarray(raw_observables.boozer_residual_scalar)
            ),
            "boozer_residual_rms": float(
                np.asarray(raw_observables.boozer_residual_rms)
            ),
        }
        terminal_payload = diag4_terminal_numerical_payload(
            terminal_numerical=terminal_payload,
            numerical_identity=numerical_identity,
            endpoint_state_sha256=endpoint_state_sha256,
            terminal_observables=observables,
            endpoint_objective_terms=objective_terms,
            endpoint_observables=observables,
        )
    terminal_path = cold_root / "terminal-numerical.json"
    _publish_canonical_json(terminal_path, terminal_payload)
    return (
        _artifact_ref_at(
            terminal_path,
            campaign_root,
            (logical_cold_root or cold_root) / "terminal-numerical.json",
            (
                f"{DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal"
                if numerical_identity is not None
                else f"{DIAGNOSTIC_SCHEMA_VERSION}-terminal"
            ),
        ),
        references,
    )


def _publish_diagnostic_policy(
    directory: Path,
    campaign_root: Path,
    *,
    policy_sha256: str,
    native_raw_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
    logical_directory: Path | None = None,
    builder: Callable[..., dict[str, JsonValue]] | None = None,
    schema_version: str = f"{DIAGNOSTIC_SCHEMA_VERSION}-policy",
) -> ArtifactRef:
    path = directory / "policy.json"
    _publish_canonical_json(
        path,
        (builder or policy_evidence_payload)(
            policy_sha256=policy_sha256,
            native_raw_equalities=native_raw_equalities,
            constraint_inverse_scale=constraint_inverse_scale,
        ),
    )
    return _artifact_ref_at(
        path,
        campaign_root,
        (logical_directory or directory) / "policy.json",
        schema_version,
    )


def _publish_parent_policy_authority(
    campaign_reference: Path,
    campaign_root: Path,
) -> ArtifactRef:
    """Publish policy inputs rederived outside the supervised child."""

    validation = validate_native_equivalent_reference(campaign_reference)
    if not validation.usable:
        raise ValueError("copied native-equivalent reference is not usable")
    native_raw_equalities = _reference_array(campaign_reference, "raw_equalities")
    bootstrap = build_single_stage_fullspace_bootstrap()
    scaling = fullspace_scaling_from_bootstrap(bootstrap.z0, bootstrap.problem)
    constraint_inverse_scale = np.asarray(
        jax.device_get(scaling.constraint_inverse_scale),
        dtype=np.float64,
    )
    policy = NativeEquivalentQualityPolicy(
        native_raw_equalities=native_raw_equalities,
        native_raw_equalities_sha256=exact_numeric_tree_sha256(native_raw_equalities),
        constraint_inverse_scale=constraint_inverse_scale,
    )
    path = campaign_root / "policy-authority.json"
    _publish_canonical_json(
        path,
        policy_evidence_payload(
            policy_sha256=policy.policy_sha256,
            native_raw_equalities=native_raw_equalities,
            constraint_inverse_scale=constraint_inverse_scale,
        ),
    )
    return _artifact_ref(
        path,
        campaign_root,
        f"{DIAGNOSTIC_SCHEMA_VERSION}-policy",
    )


def execute_timed_loop(
    prepared: PreparedNeqGntr1 | PreparedNeqGntr2 | PreparedNeqGntr3,
    *,
    trace_annotation: str | None = None,
) -> tuple[object, int, int]:
    """Synchronize the loop before stopping the timer; never finalize here."""

    def expire(_signal_number: int, _frame: object) -> None:
        raise SolveTimeoutError("synchronized solve exceeded 360 seconds")

    previous_handler = signal.signal(signal.SIGALRM, expire)
    timer_started_ns = time.perf_counter_ns()
    signal.setitimer(signal.ITIMER_REAL, SOLVE_TIMEOUT_SECONDS)
    try:
        with jax.transfer_guard("disallow"):
            if trace_annotation is None:
                loop_result = prepared.run_solver_loop()
                jax.block_until_ready(loop_result)
            else:
                with jax.profiler.TraceAnnotation(trace_annotation):
                    loop_result = prepared.run_solver_loop()
                    jax.block_until_ready(loop_result)
        timer_stopped_ns = time.perf_counter_ns()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
    return loop_result, timer_started_ns, timer_stopped_ns


def _publish_child_runtime_evidence(
    sample: SampleName | None,
) -> ArtifactRef:
    campaign_text = os.environ.get(_CAMPAIGN_ROOT_ENV)
    manifest_sha256 = os.environ.get(_SNAPSHOT_MANIFEST_ENV)
    if campaign_text is None or manifest_sha256 is None:
        raise ValueError("snapshot child launch binding is absent")
    campaign_root = Path(campaign_text).resolve(strict=True)
    diag5 = os.environ.get(_DIAG5_CHILD_ENV) == "1"
    required_roles = (
        DIAG5_GPU_SNAPSHOT_ROLES
        if diag5
        else DIAG4_GPU_SNAPSHOT_ROLES
        if os.environ.get(_DIAG4_CHILD_ENV) == "1"
        else LEGACY_SNAPSHOT_ROLES
    )
    snapshot = load_snapshot(Path.cwd(), required_roles=required_roles)
    if snapshot.manifest_sha256 != manifest_sha256:
        raise ValueError("snapshot child manifest binding differs")
    source_identity = snapshot.source_identity(campaign_root)
    if diag5:
        native_path_text = os.environ.get(_DIAG5_NATIVE_PATH_ENV)
        native_sha256 = os.environ.get(_DIAG5_NATIVE_SHA256_ENV)
        native_size_text = os.environ.get(_DIAG5_NATIVE_SIZE_ENV)
        native_link_count_text = os.environ.get(_DIAG5_NATIVE_LINK_COUNT_ENV)
        if (
            native_path_text is None
            or native_sha256 is None
            or native_size_text is None
            or not native_size_text.isdecimal()
            or native_link_count_text is None
            or not native_link_count_text.isdecimal()
        ):
            raise ValueError("DIAG5 child native identity binding is absent")
        native_path = Path(native_path_text)
        native_size = int(native_size_text)
        native_link_count = int(native_link_count_text)
        observation_v2 = observe_live_runtime_v2(
            snapshot.root,
            argv=tuple(sys.argv),
            cwd=Path.cwd(),
            environment=os.environ,
            expected_native_extension_path=native_path,
            expected_native_extension_sha256=native_sha256,
            expected_native_extension_size_bytes=native_size,
            expected_native_extension_link_count=native_link_count,
            required_roles=required_roles,
        )
        evidence_v2 = build_runtime_evidence_v2(
            snapshot.root,
            source_identity=source_identity,
            observation=observation_v2,
            expected_native_extension_path=native_path,
            expected_native_extension_sha256=native_sha256,
            expected_native_extension_size_bytes=native_size,
            expected_native_extension_link_count=native_link_count,
            required_roles=required_roles,
        )
    else:
        observation = observe_live_runtime(
            snapshot.root,
            argv=tuple(sys.argv),
            cwd=Path.cwd(),
            environment=os.environ,
            required_roles=required_roles,
        )
        evidence = build_runtime_evidence(
            snapshot.root,
            source_identity=source_identity,
            observation=observation,
            required_roles=required_roles,
        )
    diagnostic_child = os.environ.get(_DIAGNOSTIC_CHILD_ENV)
    relative_directory = (
        Path("preflight")
        if sample is None
        else (
            Path("cold")
            if diagnostic_child == DiagnosticChildMode.COLD.value
            else Path("samples") / sample.value
        )
    )
    runtime_path = campaign_root / relative_directory / RUNTIME_EVIDENCE_FILENAME
    reference = (
        publish_runtime_evidence_v2(
            runtime_path,
            evidence_v2,
            snapshot_root=snapshot.root,
            campaign_root=campaign_root,
            expected_native_extension_path=native_path,
            expected_native_extension_sha256=native_sha256,
            expected_native_extension_size_bytes=native_size,
            expected_native_extension_link_count=native_link_count,
            required_roles=required_roles,
        )
        if diag5
        else publish_runtime_evidence(
            runtime_path,
            evidence,
            snapshot_root=snapshot.root,
            campaign_root=campaign_root,
            required_roles=required_roles,
        )
    )
    runtime_path.chmod(0o444)
    return reference


def run_snapshot_preflight_child(
    *,
    reference_root: Path,
    input_root: Path,
) -> dict[str, JsonValue]:
    """Compile and inspect the production executable without dispatching it."""

    process_started_ns = time.perf_counter_ns()
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("NEQ-GNTR1 requires exactly one JAX GPU")
    runtime_reference = _publish_child_runtime_evidence(None)
    compile_started_ns = time.perf_counter_ns()
    try:
        worker = _prepare_worker(reference_root, input_root)
        jax.block_until_ready(worker.route.initial_optimizer_coordinates)
        python_callback_count = _compiled_python_callback_count(worker.route)
        if python_callback_count != 0:
            raise ValueError("timed executable contains a Python callback")
    except Exception as error:  # noqa: BLE001 - compile failures are evidence.
        completed_ns = time.perf_counter_ns()
        message = str(error).lower()
        status = (
            "COMPILE_OOM"
            if "out of memory" in message or "resource_exhausted" in message
            else "COMPILE_FAILURE"
        )
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "route": ROUTE,
            "plan_sha256": PLAN_SHA256,
            "mode": "LOWER_COMPILE_ONLY",
            "execution_status": status,
            "campaign_authorized": False,
            "solver_dispatched": False,
            "finalizer_called": False,
            "endpoint_audit_called": False,
            "runtime": _worker_runtime_payload(),
            "runtime_evidence": _artifact_ref_payload(runtime_reference),
            "timing": {
                "compile_started_ns": compile_started_ns,
                "compile_completed_ns": completed_ns,
                "process_seconds_before_serialization": (
                    completed_ns - process_started_ns
                )
                / 1.0e9,
            },
            "failure_reasons": [
                f"{status}:{type(error).__name__}:{_sha256(str(error).encode())}"
            ],
        }
    completed_ns = time.perf_counter_ns()
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "route": ROUTE,
        "plan_sha256": PLAN_SHA256,
        "mode": "LOWER_COMPILE_ONLY",
        "execution_status": "SUCCESS",
        "campaign_authorized": False,
        "solver_dispatched": False,
        "finalizer_called": False,
        "endpoint_audit_called": False,
        "python_callbacks": python_callback_count,
        "runtime": _worker_runtime_payload(),
        "runtime_evidence": _artifact_ref_payload(runtime_reference),
        "timing": {
            "compile_started_ns": compile_started_ns,
            "compile_completed_ns": completed_ns,
            "process_seconds_before_serialization": (completed_ns - process_started_ns)
            / 1.0e9,
        },
        "failure_reasons": [],
    }


def run_snapshot_diagnostic_preflight_child(
    *,
    reference_root: Path,
    input_root: Path,
) -> dict[str, JsonValue]:
    """Compile every annotated cold executable without dispatching any of them."""

    process_started_ns = time.perf_counter_ns()
    route, plan_sha256 = _diagnostic_child_identity()
    diag5 = route == DIAG5_ROUTE
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("NEQ-GNTR1 diagnostic requires exactly one JAX GPU")
    runtime_reference = _publish_child_runtime_evidence(None)
    compile_started_ns = time.perf_counter_ns()
    try:
        worker = (
            _prepare_diagnostic_worker(reference_root, input_root, diag4=True)
            if route in {DIAG4_ROUTE, DIAG5_ROUTE}
            else _prepare_diagnostic_worker(reference_root, input_root)
        )
        jax.block_until_ready(worker.worker.route.initial_optimizer_coordinates)
        python_callback_count = _compiled_diagnostic_callback_count(worker)
        if python_callback_count != 0:
            raise ValueError("diagnostic executable contains a Python callback")
    except Exception as error:  # noqa: BLE001 - compile failures are evidence.
        completed_ns = time.perf_counter_ns()
        message = str(error).lower()
        status = (
            "COMPILE_FAILURE"
            if route == DIAG2_ROUTE
            else (
                "COMPILE_OOM"
                if "out of memory" in message or "resource_exhausted" in message
                else "COMPILE_FAILURE"
            )
        )
        failure_reason = (
            f"{status}:{type(error).__name__}:{_sha256(str(error).encode())}"
        )
        if route == DIAG2_ROUTE:
            return build_diag2_compile_failure_producer_payload(
                mode="preflight",
                execution_status=status,
                runtime=_worker_runtime_payload(),
                runtime_evidence=runtime_reference,
                compile_started_ns=compile_started_ns,
                compile_completed_ns=completed_ns,
                process_seconds_before_serialization=(completed_ns - process_started_ns)
                / 1.0e9,
                failure_reasons=(failure_reason,),
            )
        if diag5:
            return build_diag5_compile_failure_producer_payload(
                execution_status=status,
                runtime=_worker_runtime_payload(),
                runtime_evidence=runtime_reference,
                compile_started_ns=compile_started_ns,
                compile_completed_ns=completed_ns,
                process_seconds_before_serialization=(completed_ns - process_started_ns)
                / 1.0e9,
                failure_reason=failure_reason,
            )
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "route": route,
            "plan_sha256": plan_sha256,
            "mode": "ANNOTATED_LOWER_COMPILE_ONLY",
            "execution_status": status,
            "campaign_authorized": False,
            "solver_dispatched": False,
            "finalizer_called": False,
            "endpoint_audit_called": False,
            "runtime": _worker_runtime_payload(),
            "runtime_evidence": _artifact_ref_payload(runtime_reference),
            "timing": {
                "compile_started_ns": compile_started_ns,
                "compile_completed_ns": completed_ns,
                "process_seconds_before_serialization": (
                    completed_ns - process_started_ns
                )
                / 1.0e9,
            },
            "failure_reasons": [failure_reason],
        }
    prepared = worker.worker.route
    host_native, host_scale = jax.device_get(
        (
            prepared.policy.native_raw_equalities,
            prepared.policy.constraint_inverse_scale,
        )
    )
    campaign_text = os.environ.get(_CAMPAIGN_ROOT_ENV)
    if campaign_text is None:
        raise ValueError("diagnostic campaign root binding is absent")
    campaign_root = Path(campaign_text).resolve(strict=True)
    policy_reference = _publish_diagnostic_policy(
        campaign_root / "preflight",
        campaign_root,
        policy_sha256=prepared.policy.policy_sha256,
        native_raw_equalities=np.asarray(host_native),
        constraint_inverse_scale=np.asarray(host_scale),
        builder=(diag5_policy_evidence_payload if diag5 else policy_evidence_payload),
        schema_version=(
            "single-stage-native-equivalent-quality-policy-v1"
            if diag5
            else f"{DIAGNOSTIC_SCHEMA_VERSION}-policy"
        ),
    )
    completed_ns = time.perf_counter_ns()
    if route in {DIAG4_ROUTE, DIAG5_ROUTE}:
        prepared = _validate_diag4_prepared_route(prepared)
        source_manifest_sha256 = os.environ.get(_SNAPSHOT_MANIFEST_ENV)
        if source_manifest_sha256 is None:
            raise ValueError("trace-free preflight source binding is absent")
        identity = prepared.identity
        payload: dict[str, JsonValue] = {
            "schema_version": (
                DIAG5_PREFLIGHT_SCHEMA_VERSION
                if diag5
                else DIAG4_PREFLIGHT_SCHEMA_VERSION
            ),
            "route": route,
            "numerical_route": DIAG4_NUMERICAL_ROUTE,
            "numerical_result_schema_version": (DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION),
            "plan_sha256": DIAG5_PLAN_SHA256 if diag5 else DIAG4_PLAN_SHA256,
            "mode": "TRACE_FREE_COMPILE_ONLY",
            "execution_status": "SUCCESS",
            "runtime": _worker_runtime_payload(),
            "runtime_evidence": _artifact_ref_payload(runtime_reference),
            "base_neq_gntr1_policy_sha256": (identity.base_neq_gntr1_policy_sha256),
            "policy_evidence": _artifact_ref_payload(policy_reference),
            "problem_sha256": identity.problem_sha256,
            "optimizer_options_sha256": identity.optimizer_options_sha256,
            "scaling_sha256": identity.scaling_sha256,
            "bootstrap_state_sha256": identity.bootstrap_state_sha256,
            "initial_physical_state_sha256": (identity.initial_physical_state_sha256),
            "identity_sha256": identity.identity_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "state_size": prepared.policy.state_size,
            "equality_size": prepared.policy.equality_size,
            "residual_size": prepared.policy.objective_residual_size,
            "campaign_authorized": False,
            "solver_dispatched": False,
            "finalizer_called": False,
            "endpoint_audit_called": False,
            "python_callbacks": python_callback_count,
            **diag4_profiler_call_audit_payload(DIAG4_PROFILER_CALL_AUDIT),
            "timing": {
                "compile_started_ns": compile_started_ns,
                "compile_completed_ns": completed_ns,
                "process_seconds_before_serialization": (
                    completed_ns - process_started_ns
                )
                / 1.0e9,
            },
            "failure_reasons": [],
        }
        return (
            validate_diag5_producer_payload(payload, mode="preflight")
            if diag5
            else validate_diag4_producer_payload(payload, mode="preflight")
        )
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "route": route,
        "plan_sha256": plan_sha256,
        "mode": "ANNOTATED_LOWER_COMPILE_ONLY",
        "execution_status": "SUCCESS",
        "policy_sha256": prepared.policy.policy_sha256,
        "policy_evidence": _artifact_ref_payload(policy_reference),
        "phase_schema_sha256": GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256,
        "state_size": prepared.policy.state_size,
        "equality_size": prepared.policy.equality_size,
        "residual_size": prepared.policy.objective_residual_size,
        "campaign_authorized": False,
        "solver_dispatched": False,
        "finalizer_called": False,
        "endpoint_audit_called": False,
        "python_callbacks": python_callback_count,
        "runtime": _worker_runtime_payload(),
        "runtime_evidence": _artifact_ref_payload(runtime_reference),
        "timing": {
            "compile_started_ns": compile_started_ns,
            "compile_completed_ns": completed_ns,
            "process_seconds_before_serialization": (completed_ns - process_started_ns)
            / 1.0e9,
        },
        "failure_reasons": [],
    }


def _single_profiler_trace(trace_root: Path) -> Path:
    traces = tuple(sorted(trace_root.rglob("*.trace.json.gz")))
    if len(traces) != 1:
        raise ValueError(f"expected exactly one profiler trace, found {len(traces)}")
    return traces[0]


def _diagnostic_child_identity() -> tuple[str, str]:
    """Select the immutable diagnostic receipt generation bound by the parent."""

    if os.environ.get(_DIAG5_CHILD_ENV) == "1":
        return DIAG5_ROUTE, DIAG5_PLAN_SHA256
    if os.environ.get(_DIAG4_CHILD_ENV) == "1":
        return DIAG4_ROUTE, DIAG4_PLAN_SHA256
    if os.environ.get(_DIAG2_CHILD_ENV) == "1":
        return DIAG2_ROUTE, DIAG2_PLAN_SHA256
    return DIAGNOSTIC_ROUTE, DIAGNOSTIC_PLAN_SHA256


def _diag4_history_values(
    history_payload: Mapping[str, JsonValue],
) -> tuple[int, int, int, str, bool, tuple[str, ...]]:
    """Extract telemetry joins from the already validated legacy history payload."""

    rows = history_payload.get("rows")
    status = history_payload.get("status")
    quality_latch = history_payload.get("quality_latch")
    if (
        not isinstance(rows, list)
        or not isinstance(status, str)
        or type(quality_latch) is not bool
    ):
        raise TypeError("DIAG4 legacy history telemetry join is malformed")
    outcomes: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("outcome"), str):
            raise TypeError("DIAG4 legacy history outcome is malformed")
        outcomes.append(row["outcome"])
    return (
        _integer_field(history_payload, "attempts"),
        _integer_field(history_payload, "accepted_steps"),
        _integer_field(history_payload, "retryable_rejections"),
        status,
        quality_latch,
        tuple(outcomes),
    )


def _diag4_integer_history_vector(
    value: object,
    *,
    context: str,
) -> np.ndarray:
    """Transfer one exact fixed-width integer history vector without dtype loss."""

    array = np.asarray(value)
    if array.shape != (300,) or array.dtype != np.dtype(np.int32):
        raise TypeError(f"{context} must be an int32 (300,) vector")
    transferred = np.array(array, copy=True, order="C")
    transferred.flags.writeable = False
    return transferred


def _diag4_float_history_matrix(
    value: object,
    *,
    context: str,
) -> np.ndarray:
    """Transfer one exact fixed-three FP history matrix without dtype loss."""

    array = np.asarray(value)
    if array.shape != (300, 3) or array.dtype != np.dtype(np.float64):
        raise TypeError(f"{context} must be an FP64 (300, 3) matrix")
    transferred = np.array(array, copy=True, order="C")
    transferred.flags.writeable = False
    return transferred


def _diag4_float_history_vector(
    value: object,
    *,
    context: str,
) -> np.ndarray:
    """Transfer one exact fixed-width FP64 history vector without dtype loss."""

    array = np.asarray(value)
    if array.shape != (300,) or array.dtype != np.dtype(np.float64):
        raise TypeError(f"{context} must be an FP64 (300,) vector")
    transferred = np.array(array, copy=True, order="C")
    transferred.flags.writeable = False
    return transferred


def _diag4_integer_history_matrix(
    value: object,
    *,
    context: str,
) -> np.ndarray:
    """Transfer one exact fixed-three integer history matrix without dtype loss."""

    array = np.asarray(value)
    if array.shape != (300, 3) or array.dtype != np.dtype(np.int32):
        raise TypeError(f"{context} must be an int32 (300, 3) matrix")
    transferred = np.array(array, copy=True, order="C")
    transferred.flags.writeable = False
    return transferred


def _run_snapshot_diag4_child(
    *,
    reference_root: Path,
    input_root: Path,
    diag5: bool = False,
) -> dict[str, JsonValue]:
    """Run one authoritative GNTR3 cold with an explicit wire generation."""

    process_started_text = os.environ.get(_DIAG4_PROCESS_STARTED_ENV)
    source_manifest_sha256 = os.environ.get(_SNAPSHOT_MANIFEST_ENV)
    if (
        process_started_text is None
        or not process_started_text.isdecimal()
        or source_manifest_sha256 is None
    ):
        raise ValueError("DIAG4 parent process or source binding is absent")
    process_started_ns = int(process_started_text)
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("NEQ-GNTR3 DIAG4 requires exactly one JAX GPU")
    campaign_text = os.environ.get(_CAMPAIGN_ROOT_ENV)
    if campaign_text is None:
        raise ValueError("diagnostic campaign root binding is absent")
    campaign_root = Path(campaign_text).resolve(strict=True)
    cold_root = campaign_root / "cold"
    runtime_reference = _publish_child_runtime_evidence(SampleName.COLD)
    process_identity = read_linux_process_identity(os.getpid())

    worker = _prepare_diagnostic_worker(reference_root, input_root, diag4=True)
    prepared = worker.worker.route
    prepared = _validate_diag4_prepared_route(prepared)
    callbacks = _compiled_diagnostic_callback_count(worker)
    if callbacks != 0:
        raise ValueError("DIAG4 executable contains a Python callback")
    jax.block_until_ready(prepared.initial_optimizer_coordinates)
    state_ready_ns = time.perf_counter_ns()

    numerical_root = cold_root / _DIAG3_NUMERICAL_PENDING_NAME
    numerical_root.mkdir(mode=0o755, exist_ok=False)
    logical_numerical_root = cold_root / _DIAG3_NUMERICAL_COMMITTED_NAME
    loop_result, solve_started_ns, solve_stopped_ns = execute_timed_loop(prepared)

    base = prepared.finalize_result(loop_result)
    jax.block_until_ready(base)
    finalizer_completed_ns = time.perf_counter_ns()
    quality_replay = worker.accepted_quality.run(
        loop_result.accepted_optimizer_coordinates,
        loop_result.accepted_state_mask,
    )
    jax.block_until_ready(quality_replay)
    endpoint_started_ns = time.perf_counter_ns()
    terminal_evidence = worker.terminal.run_evidence(
        base.optimizer_result.optimizer_coordinates,
        base.optimizer_result.multipliers,
    )
    jax.block_until_ready(terminal_evidence)
    diagnostic = build_native_equivalent_terminal_diagnostic(
        base,
        terminal_evidence.raw_endpoint,
        prepared.policy,
    )
    jax.block_until_ready(diagnostic)
    endpoint_audit_completed_ns = time.perf_counter_ns()
    config = prepared.problem.config
    (
        host_diagnostic,
        host_quality_replay,
        host_terminal_evidence,
        host_native,
        host_scale,
        host_weights,
        host_bootstrap_anchor,
        host_variable_scale,
    ) = jax.device_get(
        (
            diagnostic,
            quality_replay,
            terminal_evidence,
            prepared.policy.native_raw_equalities,
            prepared.policy.constraint_inverse_scale,
            (
                config.non_qs_weight,
                config.residual_weight,
                config.iota_weight,
                config.major_radius_weight,
                config.length_weight,
            ),
            prepared.scaling.bootstrap_anchor,
            prepared.scaling.variable_scale,
        )
    )
    serialization_started_ns = time.perf_counter_ns()

    history_payload = (
        _diagnostic_history_payload(
            host_diagnostic.base_result.loop_result,
            diag5=True,
        )
        if diag5
        else _diagnostic_history_payload(host_diagnostic.base_result.loop_result)
    )
    history_path = numerical_root / "history.json"
    _publish_canonical_json(history_path, history_payload)
    history_reference = _artifact_ref_at(
        history_path,
        campaign_root,
        logical_numerical_root / "history.json",
        (
            "single-stage-fullspace-neq-gntr3-history-v1"
            if diag5
            else f"{DIAGNOSTIC_SCHEMA_VERSION}-history"
        ),
    )
    identity = prepared.identity
    numerical_identity = NativeEquivalentNumericalIdentity(
        numerical_route=identity.route,
        numerical_result_schema_version=identity.schema_version,
        problem_sha256=identity.problem_sha256,
        optimizer_options_sha256=identity.optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=identity.base_neq_gntr1_policy_sha256,
        scaling_sha256=identity.scaling_sha256,
        bootstrap_state_sha256=identity.bootstrap_state_sha256,
        initial_physical_state_sha256=identity.initial_physical_state_sha256,
        identity_sha256=identity.identity_sha256,
    )
    terminal_reference, _array_references = _publish_diagnostic_terminal(
        numerical_root,
        campaign_root,
        host_diagnostic,
        host_quality_replay,
        host_terminal_evidence,
        np.asarray(host_native),
        np.asarray(host_scale),
        np.asarray(host_bootstrap_anchor),
        np.asarray(host_variable_scale),
        tuple(float(np.asarray(weight)) for weight in host_weights),
        terminal_seconds=(endpoint_audit_completed_ns - endpoint_started_ns) / 1.0e9,
        logical_cold_root=logical_numerical_root,
        numerical_identity=numerical_identity,
    )
    policy_reference = _publish_diagnostic_policy(
        cold_root,
        campaign_root,
        policy_sha256=prepared.policy.policy_sha256,
        native_raw_equalities=np.asarray(host_native),
        constraint_inverse_scale=np.asarray(host_scale),
        builder=(diag5_policy_evidence_payload if diag5 else policy_evidence_payload),
        schema_version=(
            "single-stage-native-equivalent-quality-policy-v1"
            if diag5
            else f"{DIAGNOSTIC_SCHEMA_VERSION}-policy"
        ),
    )
    timing_builder = (
        diag5_solve_timing_evidence_payload if diag5 else solve_timing_evidence_payload
    )
    timing_payload = timing_builder(
        child_pid=process_identity.pid,
        child_start_time_ticks=process_identity.start_ticks,
        backend="gpu",
        gpu_uuid=GPU_UUID,
        problem_sha256=identity.problem_sha256,
        optimizer_options_sha256=identity.optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=identity.base_neq_gntr1_policy_sha256,
        scaling_sha256=identity.scaling_sha256,
        bootstrap_state_sha256=identity.bootstrap_state_sha256,
        initial_physical_state_sha256=identity.initial_physical_state_sha256,
        identity_sha256=identity.identity_sha256,
        source_manifest_sha256=source_manifest_sha256,
        process_started_monotonic_ns=process_started_ns,
        state_ready_monotonic_ns=state_ready_ns,
        solve_started_monotonic_ns=solve_started_ns,
        solve_stopped_monotonic_ns=solve_stopped_ns,
        finalizer_completed_monotonic_ns=finalizer_completed_ns,
        endpoint_audit_completed_monotonic_ns=endpoint_audit_completed_ns,
        serialization_started_monotonic_ns=serialization_started_ns,
        hot_h2d_transfers=0,
        hot_d2h_transfers=0,
        python_callbacks=callbacks,
        final_d2h_transfers=1,
        profiler_call_audit=DIAG4_PROFILER_CALL_AUDIT,
    )
    timing_path = numerical_root / "solve-timing.json"
    _publish_canonical_json(timing_path, timing_payload)
    timing_reference = _artifact_ref_at(
        timing_path,
        campaign_root,
        logical_numerical_root / "solve-timing.json",
        (
            DIAG5_SOLVE_TIMING_SCHEMA_VERSION
            if diag5
            else DIAG4_SOLVE_TIMING_SCHEMA_VERSION
        ),
    )

    (
        loop_attempts,
        accepted_steps,
        retryable_rejections,
        terminal_status,
        quality_latch,
        history_outcomes,
    ) = _diag4_history_values(history_payload)
    live_history = host_diagnostic.base_result.loop_result.history
    nonlinear_corrections = _diag4_integer_history_vector(
        live_history.nonlinear_corrections,
        context="DIAG4 nonlinear corrections",
    )
    maximum_individual_correction_step_ratio = _diag4_float_history_vector(
        live_history.maximum_individual_correction_step_ratio,
        context="DIAG4 maximum individual correction-step ratio",
    )
    correction_path_step_ratio = _diag4_float_history_vector(
        live_history.correction_path_step_ratio,
        context="DIAG4 correction-path step ratio",
    )
    steihaug_solve_calls = _diag4_integer_history_vector(
        live_history.steihaug_solve_calls,
        context="DIAG4 Steihaug solve calls",
    )
    telemetry_builder = (
        diag5_safeguard_telemetry_payload if diag5 else safeguard_telemetry_payload
    )
    telemetry_payload = telemetry_builder(
        history_evidence=history_reference,
        problem_sha256=identity.problem_sha256,
        optimizer_options_sha256=identity.optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=identity.base_neq_gntr1_policy_sha256,
        scaling_sha256=identity.scaling_sha256,
        bootstrap_state_sha256=identity.bootstrap_state_sha256,
        initial_physical_state_sha256=identity.initial_physical_state_sha256,
        identity_sha256=identity.identity_sha256,
        loop_attempts=loop_attempts,
        accepted_steps=accepted_steps,
        retryable_rejections=retryable_rejections,
        terminal_status=terminal_status,
        quality_latch=quality_latch,
        history_outcomes=history_outcomes,
        nonlinear_corrections=nonlinear_corrections,
        maximum_individual_correction_step_ratio=(
            maximum_individual_correction_step_ratio
        ),
        correction_path_step_ratio=correction_path_step_ratio,
        steihaug_solve_calls=steihaug_solve_calls,
        subtrial_count=_diag4_integer_history_vector(
            live_history.subtrial_count,
            context="DIAG4 subtrial count",
        ),
        selected_subtrial_index=_diag4_integer_history_vector(
            live_history.selected_subtrial_index,
            context="DIAG4 selected subtrial index",
        ),
        subtrial_trust_radius=_diag4_float_history_matrix(
            live_history.subtrial_trust_radius,
            context="DIAG4 subtrial trust radius",
        ),
        subtrial_outcome=_diag4_integer_history_matrix(
            live_history.subtrial_outcome,
            context="DIAG4 subtrial outcome",
        ),
        subtrial_actual_reduction=_diag4_float_history_matrix(
            live_history.subtrial_actual_reduction,
            context="DIAG4 subtrial actual reduction",
        ),
        subtrial_predicted_reduction=_diag4_float_history_matrix(
            live_history.subtrial_predicted_reduction,
            context="DIAG4 subtrial predicted reduction",
        ),
        subtrial_maximum_individual_correction_step_ratio=(
            _diag4_float_history_matrix(
                live_history.subtrial_maximum_individual_correction_step_ratio,
                context="DIAG4 subtrial maximum individual correction-step ratio",
            )
        ),
        subtrial_correction_path_step_ratio=_diag4_float_history_matrix(
            live_history.subtrial_correction_path_step_ratio,
            context="DIAG4 subtrial correction-path step ratio",
        ),
        subtrial_corrected_radius_ratio=_diag4_float_history_matrix(
            live_history.subtrial_corrected_radius_ratio,
            context="DIAG4 subtrial corrected-radius ratio",
        ),
        subtrial_steihaug_iterations=_diag4_integer_history_matrix(
            live_history.subtrial_steihaug_iterations,
            context="DIAG4 subtrial Steihaug iterations",
        ),
        subtrial_steihaug_hvp_evaluations=_diag4_integer_history_matrix(
            live_history.subtrial_steihaug_hvp_evaluations,
            context="DIAG4 subtrial Steihaug HVP evaluations",
        ),
        subtrial_steihaug_solve_calls=_diag4_integer_history_matrix(
            live_history.subtrial_steihaug_solve_calls,
            context="DIAG4 subtrial Steihaug solve calls",
        ),
        subtrial_total_hvp_evaluations=_diag4_integer_history_matrix(
            live_history.subtrial_total_hvp_evaluations,
            context="DIAG4 subtrial total HVP evaluations",
        ),
        subtrial_nonlinear_corrections=_diag4_integer_history_matrix(
            live_history.subtrial_nonlinear_corrections,
            context="DIAG4 subtrial nonlinear corrections",
        ),
        subtrial_joint_evaluations=_diag4_integer_history_matrix(
            live_history.subtrial_joint_evaluations,
            context="DIAG4 subtrial joint evaluations",
        ),
        subtrial_joint_linearizations=_diag4_integer_history_matrix(
            live_history.subtrial_joint_linearizations,
            context="DIAG4 subtrial joint linearizations",
        ),
        subtrial_joint_value_evaluations=_diag4_integer_history_matrix(
            live_history.subtrial_joint_value_evaluations,
            context="DIAG4 subtrial joint value evaluations",
        ),
        subtrial_objective_residual_linearizations=(
            _diag4_integer_history_matrix(
                live_history.subtrial_objective_residual_linearizations,
                context="DIAG4 subtrial objective-residual linearizations",
            )
        ),
        subtrial_gram_factorizations=_diag4_integer_history_matrix(
            live_history.subtrial_gram_factorizations,
            context="DIAG4 subtrial Gram factorizations",
        ),
        subtrial_gram_solves=_diag4_integer_history_matrix(
            live_history.subtrial_gram_solves,
            context="DIAG4 subtrial Gram solves",
        ),
    )
    telemetry_path = numerical_root / "safeguard-telemetry.json"
    _publish_canonical_json(telemetry_path, telemetry_payload)
    telemetry_reference = _artifact_ref_at(
        telemetry_path,
        campaign_root,
        logical_numerical_root / "safeguard-telemetry.json",
        (
            DIAG5_SAFEGUARD_TELEMETRY_SCHEMA_VERSION
            if diag5
            else DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION
        ),
    )
    producer: dict[str, JsonValue] = {
        "schema_version": (
            DIAG5_COLD_RESULT_SCHEMA_VERSION
            if diag5
            else DIAG4_COLD_RESULT_SCHEMA_VERSION
        ),
        "numerical_bundle_schema_version": (
            DIAG5_NUMERICAL_BUNDLE_SCHEMA_VERSION
            if diag5
            else DIAG4_NUMERICAL_BUNDLE_SCHEMA_VERSION
        ),
        "route": DIAG5_ROUTE if diag5 else DIAG4_ROUTE,
        "numerical_route": DIAG4_NUMERICAL_ROUTE,
        "numerical_result_schema_version": DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
        "plan_sha256": DIAG5_PLAN_SHA256 if diag5 else DIAG4_PLAN_SHA256,
        "execution_status": "COMPLETE",
        "runtime": _worker_runtime_payload(),
        "runtime_evidence": _artifact_ref_payload(runtime_reference),
        "base_neq_gntr1_policy_sha256": identity.base_neq_gntr1_policy_sha256,
        "policy_evidence": _artifact_ref_payload(policy_reference),
        "problem_sha256": identity.problem_sha256,
        "optimizer_options_sha256": identity.optimizer_options_sha256,
        "scaling_sha256": identity.scaling_sha256,
        "bootstrap_state_sha256": identity.bootstrap_state_sha256,
        "initial_physical_state_sha256": identity.initial_physical_state_sha256,
        "identity_sha256": identity.identity_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "history_evidence": _artifact_ref_payload(history_reference),
        "terminal_numerical_evidence": _artifact_ref_payload(terminal_reference),
        "solve_timing_evidence": _artifact_ref_payload(timing_reference),
        "safeguard_telemetry_evidence": _artifact_ref_payload(telemetry_reference),
        **diag4_profiler_call_audit_payload(DIAG4_PROFILER_CALL_AUDIT),
        "endpoint_audit_called": True,
        "campaign_authorized": False,
        "failure_reasons": [],
    }
    return (
        validate_diag5_producer_payload(producer, mode="cold")
        if diag5
        else validate_diag4_producer_payload(producer, mode="cold")
    )


def run_snapshot_diagnostic_child(
    *,
    reference_root: Path,
    input_root: Path,
) -> dict[str, JsonValue]:
    """Run the one annotated cold and publish complete raw numerical evidence."""

    process_started_ns = time.perf_counter_ns()
    route, plan_sha256 = _diagnostic_child_identity()
    if route in {DIAG4_ROUTE, DIAG5_ROUTE}:
        return _run_snapshot_diag4_child(
            reference_root=reference_root,
            input_root=input_root,
            diag5=route == DIAG5_ROUTE,
        )
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("NEQ-GNTR1 diagnostic requires exactly one JAX GPU")
    campaign_text = os.environ.get(_CAMPAIGN_ROOT_ENV)
    if campaign_text is None:
        raise ValueError("diagnostic campaign root binding is absent")
    campaign_root = Path(campaign_text).resolve(strict=True)
    cold_root = campaign_root / "cold"
    runtime_reference = _publish_child_runtime_evidence(SampleName.COLD)
    compile_started_ns = time.perf_counter_ns()
    try:
        worker = _prepare_diagnostic_worker(reference_root, input_root)
        prepared = worker.worker.route
        callbacks = _compiled_diagnostic_callback_count(worker)
        if callbacks != 0:
            raise ValueError("diagnostic executable contains a Python callback")
    except Exception as error:
        if route != DIAG2_ROUTE:
            raise
        compile_completed_ns = time.perf_counter_ns()
        status = "COMPILE_FAILURE"
        failure_reason = (
            f"{status}:{type(error).__name__}:{_sha256(str(error).encode())}"
        )
        return build_diag2_compile_failure_producer_payload(
            mode="cold",
            execution_status=status,
            runtime=_worker_runtime_payload(),
            runtime_evidence=runtime_reference,
            compile_started_ns=compile_started_ns,
            compile_completed_ns=compile_completed_ns,
            process_seconds_before_serialization=(
                compile_completed_ns - process_started_ns
            )
            / 1.0e9,
            failure_reasons=(failure_reason,),
        )
    compile_completed_ns = time.perf_counter_ns()
    jax.block_until_ready(prepared.initial_optimizer_coordinates)
    state_ready_ns = time.perf_counter_ns()
    numerical_root = (
        cold_root / _DIAG3_NUMERICAL_PENDING_NAME if route == DIAG2_ROUTE else cold_root
    )
    if numerical_root != cold_root:
        numerical_root.mkdir(mode=0o755, exist_ok=False)
    logical_numerical_root = (
        cold_root / _DIAG3_NUMERICAL_COMMITTED_NAME
        if route == DIAG2_ROUTE
        else cold_root
    )
    trace_root = numerical_root / "raw-trace"
    trace_root.mkdir(parents=True, exist_ok=False)
    profiler_started_ns = time.perf_counter_ns()
    jax.profiler.start_trace(
        str(trace_root),
        profiler_options=build_jax_profiler_options(
            jax.profiler.ProfileOptions, PROFILED_PROFILER_POLICY
        ),
    )
    solve_started_ns = time.perf_counter_ns()
    solve_timeout: SolveTimeoutError | None = None
    try:
        loop_result, solve_started_ns, solve_stopped_ns = execute_timed_loop(
            prepared,
            trace_annotation=TRACE_LOOP_ENVELOPE_NAME,
        )
    except SolveTimeoutError as error:
        solve_timeout = error
    finally:
        if solve_timeout is not None:
            solve_stopped_ns = time.perf_counter_ns()
        jax.profiler.stop_trace()
    profiler_stopped_ns = time.perf_counter_ns()
    if solve_timeout is not None:
        return {
            "schema_version": f"{DIAGNOSTIC_SCHEMA_VERSION}-producer",
            "route": route,
            "plan_sha256": plan_sha256,
            "execution_status": "TIMEOUT",
            "runtime": _worker_runtime_payload(),
            "runtime_evidence": _artifact_ref_payload(runtime_reference),
            "policy_sha256": prepared.policy.policy_sha256,
            "phase_schema_sha256": GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256,
            "timestamps_ns": {
                "process_started": process_started_ns,
                "compile_started": compile_started_ns,
                "compile_completed": compile_completed_ns,
                "state_ready": state_ready_ns,
                "profiler_started": profiler_started_ns,
                "solve_started": solve_started_ns,
                "solve_stopped": solve_stopped_ns,
                "profiler_stopped": profiler_stopped_ns,
            },
            "transfer_audit": {
                "hot_h2d_transfers": 0,
                "hot_d2h_transfers": 0,
                "python_callbacks": callbacks,
                "final_d2h_transfers": 0,
            },
            "endpoint_audit_called": False,
            "campaign_authorized": False,
            "failure_reasons": [
                "SOLVE_TIMEOUT:" + _sha256(str(solve_timeout).encode())
            ],
        }
    finalizer_started_ns = time.perf_counter_ns()
    base = prepared.finalize_result(loop_result)
    jax.block_until_ready(base)
    finalizer_stopped_ns = time.perf_counter_ns()
    replay_started_ns = time.perf_counter_ns()
    quality_replay = worker.accepted_quality.run(
        loop_result.accepted_optimizer_coordinates,
        loop_result.accepted_state_mask,
    )
    jax.block_until_ready(quality_replay)
    replay_stopped_ns = time.perf_counter_ns()
    endpoint_started_ns = time.perf_counter_ns()
    terminal_evidence = worker.terminal.run_evidence(
        base.optimizer_result.optimizer_coordinates,
        base.optimizer_result.multipliers,
    )
    jax.block_until_ready(terminal_evidence)
    endpoint_stopped_ns = time.perf_counter_ns()
    diagnostic = build_native_equivalent_terminal_diagnostic(
        base, terminal_evidence.raw_endpoint, prepared.policy
    )
    config = prepared.problem.config
    (
        host_diagnostic,
        host_quality_replay,
        host_terminal_evidence,
        host_native,
        host_scale,
        host_weights,
        host_bootstrap_anchor,
        host_variable_scale,
    ) = jax.device_get(
        (
            diagnostic,
            quality_replay,
            terminal_evidence,
            prepared.policy.native_raw_equalities,
            prepared.policy.constraint_inverse_scale,
            (
                config.non_qs_weight,
                config.residual_weight,
                config.iota_weight,
                config.major_radius_weight,
                config.length_weight,
            ),
            prepared.scaling.bootstrap_anchor,
            prepared.scaling.variable_scale,
        )
    )
    final_d2h_ns = time.perf_counter_ns()
    history_path = numerical_root / "history.json"
    _publish_canonical_json(
        history_path,
        _diagnostic_history_payload(host_diagnostic.base_result.loop_result),
    )
    terminal_reference, _array_references = _publish_diagnostic_terminal(
        numerical_root,
        campaign_root,
        host_diagnostic,
        host_quality_replay,
        host_terminal_evidence,
        np.asarray(host_native),
        np.asarray(host_scale),
        np.asarray(host_bootstrap_anchor),
        np.asarray(host_variable_scale),
        tuple(float(np.asarray(weight)) for weight in host_weights),
        terminal_seconds=(endpoint_stopped_ns - endpoint_started_ns) / 1.0e9,
        logical_cold_root=logical_numerical_root,
    )
    policy_reference = _publish_diagnostic_policy(
        cold_root,
        campaign_root,
        policy_sha256=prepared.policy.policy_sha256,
        native_raw_equalities=np.asarray(host_native),
        constraint_inverse_scale=np.asarray(host_scale),
    )
    trace_path = _single_profiler_trace(trace_root)
    trace_exported_ns = time.perf_counter_ns()
    trace_normalization_error: OSError | TypeError | ValueError | None = None
    try:
        intervals_payload = normalize_chrome_trace(
            trace_path,
            phase_schema_sha256=GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256,
        )
    except (OSError, TypeError, ValueError) as error:
        if route != DIAG2_ROUTE:
            raise
        trace_normalization_error = error
        intervals_path = None
    else:
        intervals_path = numerical_root / "trace-intervals.json"
        _publish_canonical_json(intervals_path, intervals_payload)
    serialized_ns = time.perf_counter_ns()
    process_stopped_ns = time.perf_counter_ns()
    trace_failure = trace_normalization_error is not None
    history_reference = _artifact_ref_at(
        history_path,
        campaign_root,
        logical_numerical_root / "history.json",
        f"{DIAGNOSTIC_SCHEMA_VERSION}-history",
    )
    raw_trace_reference = _artifact_ref_at(
        trace_path,
        campaign_root,
        logical_numerical_root / "raw-trace" / trace_path.relative_to(trace_root),
        "jax-chrome-trace-gzip-v1",
    )
    return {
        "schema_version": (
            DIAG3_COLD_RESULT_SCHEMA_VERSION
            if route == DIAG2_ROUTE
            else f"{DIAGNOSTIC_SCHEMA_VERSION}-producer"
        ),
        "route": route,
        "plan_sha256": plan_sha256,
        "execution_status": (
            "TRACE_NORMALIZATION_FAILED" if trace_failure else "COMPLETE"
        ),
        "runtime": _worker_runtime_payload(),
        "runtime_evidence": _artifact_ref_payload(runtime_reference),
        "policy_sha256": prepared.policy.policy_sha256,
        "phase_schema_sha256": GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256,
        "history_evidence": _artifact_ref_payload(history_reference),
        "terminal_numerical_evidence": _artifact_ref_payload(terminal_reference),
        "policy_evidence": _artifact_ref_payload(policy_reference),
        "raw_trace_evidence": _artifact_ref_payload(raw_trace_reference),
        "trace_intervals_evidence": (
            None
            if intervals_path is None
            else _artifact_ref_payload(
                _artifact_ref_at(
                    intervals_path,
                    campaign_root,
                    logical_numerical_root / "trace-intervals.json",
                    f"{DIAGNOSTIC_SCHEMA_VERSION}-raw-trace",
                )
            )
        ),
        "timestamps_ns": {
            "process_started": process_started_ns,
            "compile_started": compile_started_ns,
            "compile_completed": compile_completed_ns,
            "state_ready": state_ready_ns,
            "profiler_started": profiler_started_ns,
            "solve_started": solve_started_ns,
            "solve_stopped": solve_stopped_ns,
            "profiler_stopped": profiler_stopped_ns,
            "finalizer_started": finalizer_started_ns,
            "finalizer_stopped": finalizer_stopped_ns,
            "quality_replay_started": replay_started_ns,
            "quality_replay_stopped": replay_stopped_ns,
            "endpoint_diagnostics_started": endpoint_started_ns,
            "endpoint_diagnostics_stopped": endpoint_stopped_ns,
            "final_d2h": final_d2h_ns,
            "trace_exported": trace_exported_ns,
            "serialized": serialized_ns,
            "process_stopped": process_stopped_ns,
        },
        "transfer_audit": {
            "hot_h2d_transfers": 0,
            "hot_d2h_transfers": 0,
            "python_callbacks": callbacks,
            "final_d2h_transfers": 1,
        },
        "endpoint_audit_called": False,
        "campaign_authorized": False,
        "failure_reasons": (
            [
                "TRACE_NORMALIZATION_FAILED:"
                + _sha256(
                    (
                        f"{type(trace_normalization_error).__name__}:"
                        f"{trace_normalization_error}"
                    ).encode()
                )
            ]
            if trace_normalization_error is not None
            else []
        ),
    }


def _worker_runtime_payload() -> dict[str, JsonValue]:
    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "device": str(devices[0]) if len(devices) == 1 else None,
        "device_uuid": GPU_UUID,
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jaxlib": jaxlib.__version__,
        "python": sys.version,
    }


def run_snapshot_child(
    sample: SampleName,
    *,
    reference_root: Path,
    input_root: Path,
) -> dict[str, JsonValue]:
    """Execute one pristine compiled solve and its post-timing finalizer."""

    process_started_ns = time.perf_counter_ns()
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("NEQ-GNTR1 requires exactly one JAX GPU")
    runtime_reference = _publish_child_runtime_evidence(sample)
    compile_started_ns = time.perf_counter_ns()
    try:
        worker = _prepare_worker(reference_root, input_root)
        prepared = worker.route
        jax.block_until_ready(prepared.initial_optimizer_coordinates)
        python_callback_count = _compiled_python_callback_count(worker.route)
        if python_callback_count != 0:
            raise ValueError("timed executable contains a Python callback")
    except Exception as error:  # noqa: BLE001 - compile failures require a receipt.
        compile_completed_ns = time.perf_counter_ns()
        return {
            "schema_version": WORKER_SCHEMA_VERSION,
            "route": ROUTE,
            "plan_sha256": PLAN_SHA256,
            "sample": sample.value,
            "execution_status": "COMPILE_FAILURE",
            "native_equivalent_quality": False,
            "candidate_reached": False,
            "runtime": _worker_runtime_payload(),
            "runtime_evidence": _artifact_ref_payload(runtime_reference),
            "timing": {
                "compile_started_ns": compile_started_ns,
                "compile_completed_ns": compile_completed_ns,
            },
            "failure_reasons": [
                f"COMPILE_FAILURE:{type(error).__name__}:{_sha256(str(error).encode())}"
            ],
        }
    compile_completed_ns = time.perf_counter_ns()
    device_state_ready_ns = compile_completed_ns
    host_reference_inputs = {
        "bootstrap_state": np.asarray(
            jax.device_get(prepared.initial_physical_state), dtype=np.float64
        ).tolist(),
        "constraint_inverse_scale": np.asarray(
            prepared.policy.constraint_inverse_scale, dtype=np.float64
        ).tolist(),
    }
    timed_call_started_ns = time.perf_counter_ns()
    try:
        loop_result, timer_started_ns, timer_stopped_ns = execute_timed_loop(prepared)
    except SolveTimeoutError as error:
        timer_stopped_ns = time.perf_counter_ns()
        return {
            "schema_version": WORKER_SCHEMA_VERSION,
            "route": ROUTE,
            "plan_sha256": PLAN_SHA256,
            "sample": sample.value,
            "execution_status": "TIMEOUT",
            "reference_policy_sha256": prepared.policy.policy_sha256,
            "reference_inputs": host_reference_inputs,
            "native_equivalent_quality": False,
            "candidate_reached": False,
            "runtime": _worker_runtime_payload(),
            "runtime_evidence": _artifact_ref_payload(runtime_reference),
            "timing": {
                "compile_started_ns": compile_started_ns,
                "compile_completed_ns": compile_completed_ns,
                "device_state_ready_ns": device_state_ready_ns,
                "timer_started_ns": timed_call_started_ns,
                "timer_stopped_ns": timer_stopped_ns,
                "synchronized_solve_seconds": (timer_stopped_ns - timed_call_started_ns)
                / 1.0e9,
            },
            "failure_reasons": [f"SOLVE_TIMEOUT:{_sha256(str(error).encode())}"],
        }
    post_timing_started_ns = time.perf_counter_ns()
    finalized = prepared.finalize_result(loop_result)
    jax.block_until_ready(finalized)
    host_result, host_problem, host_scaling = jax.device_get(
        (finalized, prepared.problem, prepared.scaling)
    )
    final_transfer_ns = time.perf_counter_ns()
    candidate = bool(
        np.asarray(host_result.loop_result.device_quality_candidate_reached)
    )
    audit_started_ns = post_timing_started_ns if candidate else None
    endpoint_audit: dict[str, JsonValue] | None = None
    native_equivalent_quality = False
    if candidate:
        audit = produce_native_equivalent_endpoint_audit(
            host_result,
            host_problem,
            host_scaling,
            prepared.policy,
            worker.native_runtime,
            worker.native_reference_state,
        )
        endpoint_audit = endpoint_audit_payload(audit)
        native_equivalent_quality = validate_endpoint_audit_payload(endpoint_audit)
    serialized_ns = time.perf_counter_ns()
    endpoint = host_result.endpoint
    candidate_payload: dict[str, JsonValue] = {
        "reached": candidate,
        "first_hit_attempt": (
            int(np.asarray(host_result.loop_result.first_quality_attempt))
            if candidate
            else None
        ),
        "first_hit_accepted_step": (
            int(np.asarray(host_result.loop_result.first_quality_accepted_step))
            if candidate
            else None
        ),
        "accepted_step_count": int(np.asarray(host_result.loop_result.accepted_steps)),
        "state_sha256": (
            endpoint_audit["audited_state_sha256"]
            if candidate and endpoint_audit is not None
            else None
        ),
        "physical_objective": (
            float(np.asarray(endpoint.evaluation.weighted_total)) if candidate else None
        ),
        "raw_equalities": (
            np.asarray(endpoint.raw_equalities, dtype=np.float64).tolist()
            if candidate
            else []
        ),
        "scaled_equalities": (
            np.asarray(endpoint.scaled_equalities, dtype=np.float64).tolist()
            if candidate
            else []
        ),
        "scaled_feasibility_inf": (
            float(np.max(np.abs(np.asarray(endpoint.scaled_equalities))))
            if candidate
            else None
        ),
        "state_dtype": str(np.asarray(endpoint.physical_state).dtype)
        if candidate
        else None,
        "equality_dtype": str(np.asarray(endpoint.raw_equalities).dtype)
        if candidate
        else None,
        "correction_certified": candidate,
    }
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "route": ROUTE,
        "plan_sha256": PLAN_SHA256,
        "sample": sample.value,
        "execution_status": "COMPLETED",
        "reference_policy_sha256": prepared.policy.policy_sha256,
        "native_equivalent_quality": native_equivalent_quality,
        "candidate_reached": candidate,
        "candidate": candidate_payload,
        "endpoint_audit": endpoint_audit,
        "runtime": _worker_runtime_payload(),
        "runtime_evidence": _artifact_ref_payload(runtime_reference),
        "reference_inputs": host_reference_inputs,
        "timing": {
            "compile_started_ns": compile_started_ns,
            "compile_completed_ns": compile_completed_ns,
            "device_state_ready_ns": device_state_ready_ns,
            "timer_started_ns": timer_started_ns,
            "timer_stopped_ns": timer_stopped_ns,
            "audit_started_ns": audit_started_ns,
            "final_transfer_ns": final_transfer_ns if candidate else None,
            "serialized_ns": serialized_ns,
            "synchronized_solve_seconds": (timer_stopped_ns - timer_started_ns) / 1.0e9,
            "endpoint_audit_seconds": (
                (serialized_ns - post_timing_started_ns) / 1.0e9 if candidate else None
            ),
            "process_seconds_before_serialization": (serialized_ns - process_started_ns)
            / 1.0e9,
        },
        "transfer_audit": {
            "hot_h2d_transfers": 0,
            "hot_d2h_transfers": 0,
            "python_callbacks": 0,
            "final_d2h_transfers": 1,
            "timed_transfer_guard": "disallow",
        },
        "loop": {
            "accepted_steps": int(np.asarray(host_result.loop_result.accepted_steps)),
            "attempts": int(np.asarray(host_result.loop_result.attempts)),
            "first_quality_attempt": int(
                np.asarray(host_result.loop_result.first_quality_attempt)
            ),
            "first_quality_accepted_step": int(
                np.asarray(host_result.loop_result.first_quality_accepted_step)
            ),
        },
    }


def nonpromoting_sample_draft_payload(
    supervised: SupervisedSample,
    *,
    producer_reference: ArtifactRef,
    runtime_reference: ArtifactRef,
    source_manifest_reference: ArtifactRef,
) -> dict[str, JsonValue]:
    """Return terminal evidence while a full typed receipt cannot be formed.

    The endpoint-audit producer must replace this fail-closed draft mapping
    before a GPU campaign can be promotion-eligible.
    """

    del producer_reference, runtime_reference, source_manifest_reference
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "route": ROUTE,
        "plan_sha256": PLAN_SHA256,
        "sample": supervised.sample.value,
        "terminal_status": supervised.terminal_status.value,
        "promotion_eligible": False,
        "failure_reasons": [
            *supervised.failure_reasons,
            "ENDPOINT_AUDIT_RECEIPT_BINDING_NOT_PRODUCED",
        ],
    }


def _capture_source_identity_evidence(
    publication: SnapshotPublication,
    campaign_root: Path,
) -> SourceIdentityEvidence:
    observed = load_snapshot(publication.root)
    if (
        observed.manifest_sha256 != publication.manifest_sha256
        or observed.entries != publication.entries
        or observed.worktree != publication.worktree
    ):
        raise ValueError("source snapshot identity changed around supervised child")
    source_identity = observed.source_identity(campaign_root)
    manifest_payload = observed.manifest_path.read_bytes()
    manifest_sha256 = _sha256(manifest_payload)
    if (
        manifest_sha256 != observed.manifest_sha256
        or source_identity.snapshot_manifest.sha256 != manifest_sha256
        or source_identity.snapshot_manifest.size_bytes != len(manifest_payload)
    ):
        raise ValueError("source manifest changed around supervised child")
    return SourceIdentityEvidence(
        git_head=source_identity.git_head,
        tracked_diff_sha256=source_identity.tracked_diff_sha256,
        untracked_bytes_manifest_sha256=(
            source_identity.untracked_bytes_manifest_sha256
        ),
        source_manifest_sha256=manifest_sha256,
        source_manifest_size_bytes=len(manifest_payload),
    )


def _execution_status(status: ChildTerminalStatus) -> ExecutionStatus:
    return {
        ChildTerminalStatus.COMPLETE: ExecutionStatus.COMPLETED,
        ChildTerminalStatus.TIMEOUT: ExecutionStatus.TIMEOUT,
        ChildTerminalStatus.COMPILE_FAILURE: ExecutionStatus.COMPILE_FAILURE,
        ChildTerminalStatus.CRASH: ExecutionStatus.CRASH,
        ChildTerminalStatus.MONITOR_FAILURE: ExecutionStatus.INCOMPLETE,
        ChildTerminalStatus.PROTOCOL_FAILURE: ExecutionStatus.INCOMPLETE,
    }[status]


def _mapping_field(payload: Mapping[str, JsonValue], name: str) -> dict[str, JsonValue]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"producer field {name!r} must be an object")
    return value


def _integer_field(payload: Mapping[str, JsonValue], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"producer field {name!r} must be an integer")
    return value


def _number_field(payload: Mapping[str, JsonValue], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"producer field {name!r} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"producer field {name!r} must be finite")
    return result


def _noncandidate_accepted_step_count(
    producer: Mapping[str, JsonValue],
) -> int:
    candidate = producer.get("candidate")
    if candidate is None:
        return 0
    if not isinstance(candidate, dict):
        raise TypeError("producer candidate must be an object")
    return _integer_field(candidate, "accepted_step_count")


def _artifact_from_payload(payload: Mapping[str, JsonValue]) -> ArtifactRef:
    relative = payload.get("relative_path")
    sha256 = payload.get("sha256")
    size = payload.get("size_bytes")
    schema = payload.get("schema_version")
    if (
        not isinstance(relative, str)
        or not isinstance(sha256, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not isinstance(schema, str)
    ):
        raise TypeError("artifact reference payload is malformed")
    return ArtifactRef(relative, sha256, size, schema)


def _cold_numerical_bundle_publication(
    cold_root: Path,
) -> ColdNumericalBundlePublication:
    return ColdNumericalBundlePublication(
        pending_root=cold_root / _DIAG3_NUMERICAL_PENDING_NAME,
        committed_root=cold_root / _DIAG3_NUMERICAL_COMMITTED_NAME,
        uncommitted_root=cold_root / _DIAG3_NUMERICAL_UNCOMMITTED_NAME,
    )


def _pending_path_for_logical_reference(
    pending_root: Path, reference: ArtifactRef
) -> Path:
    relative = Path(reference.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("cold numerical result path is not canonical")
    try:
        bundle_relative = relative.relative_to("cold/numerical-result")
    except ValueError as error:
        raise ValueError("cold numerical result path escapes cold/") from error
    physical = pending_root / bundle_relative
    payload = physical.read_bytes()
    if len(payload) != reference.size_bytes or _sha256(payload) != reference.sha256:
        raise ValueError("cold numerical result bytes differ from producer")
    return physical


def _validate_pending_cold_numerical_bundle(
    publication: ColdNumericalBundlePublication,
    producer: Mapping[str, JsonValue],
) -> None:
    """Validate the closed staged tree against one exact typed child result."""

    pending = publication.pending_root
    if pending.is_symlink() or not pending.is_dir():
        raise ValueError("cold numerical pending bundle is absent")
    status = producer.get("execution_status")
    if status not in {"COMPLETE", "TRACE_NORMALIZATION_FAILED"}:
        raise ValueError("cold numerical bundle producer status differs")
    validate_diag3_producer_payload(producer, mode="cold")
    expected: set[Path] = set()
    for field, expected_relative in (
        ("history_evidence", "cold/numerical-result/history.json"),
        (
            "terminal_numerical_evidence",
            "cold/numerical-result/terminal-numerical.json",
        ),
    ):
        reference = _artifact_from_payload(_mapping_field(producer, field))
        if reference.relative_path != expected_relative:
            raise ValueError(f"{field} path differs from the committed layout")
        expected.add(_pending_path_for_logical_reference(pending, reference))
    policy_reference = _artifact_from_payload(
        _mapping_field(producer, "policy_evidence")
    )
    policy_path = pending.parent / "policy.json"
    policy_payload = policy_path.read_bytes()
    if (
        policy_reference.relative_path != "cold/policy.json"
        or len(policy_payload) != policy_reference.size_bytes
        or _sha256(policy_payload) != policy_reference.sha256
    ):
        raise ValueError("cold policy differs from the successor producer")
    raw_trace = _artifact_from_payload(_mapping_field(producer, "raw_trace_evidence"))
    if not raw_trace.relative_path.startswith("cold/numerical-result/raw-trace/"):
        raise ValueError("raw trace path differs from the committed layout")
    raw_trace_path = _pending_path_for_logical_reference(pending, raw_trace)
    expected.add(raw_trace_path)
    xplane_path = raw_trace_path.with_name(
        f"{raw_trace_path.name.removesuffix('.trace.json.gz')}.xplane.pb"
    )
    if not xplane_path.is_file() or xplane_path.is_symlink():
        raise ValueError("cold numerical result omits its XPlane sibling")
    expected.add(xplane_path)
    intervals_value = producer.get("trace_intervals_evidence")
    if status == "COMPLETE":
        if not isinstance(intervals_value, dict):
            raise TypeError("complete cold numerical result omits trace intervals")
        intervals = _artifact_from_payload(intervals_value)
        if intervals.relative_path != "cold/numerical-result/trace-intervals.json":
            raise ValueError("trace intervals path differs from committed layout")
        expected.add(_pending_path_for_logical_reference(pending, intervals))
    elif intervals_value is not None:
        raise ValueError("trace-normalization failure retains trace intervals")
    terminal_path = pending / "terminal-numerical.json"
    terminal_payload = load_canonical_json_bytes(terminal_path.read_bytes())
    if not isinstance(terminal_payload, dict) or not isinstance(
        terminal_payload.get("arrays"), dict
    ):
        raise TypeError("cold numerical terminal array ledger is absent")
    arrays = terminal_payload["arrays"]
    if frozenset(arrays) != frozenset(DIAGNOSTIC_ARRAY_SPECS):
        raise ValueError("cold numerical terminal arrays differ")
    for name in DIAGNOSTIC_ARRAY_SPECS:
        row = arrays[name]
        if not isinstance(row, dict) or not isinstance(row.get("artifact"), dict):
            raise TypeError(f"cold numerical terminal array {name} is malformed")
        reference = _artifact_from_payload(row["artifact"])
        if reference.relative_path != f"cold/numerical-result/arrays/{name}.npy":
            raise ValueError("cold numerical terminal array path differs")
        expected.add(_pending_path_for_logical_reference(pending, reference))
    observed = {
        path for path in pending.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if observed != expected:
        raise ValueError("cold numerical pending bundle has missing or extra files")
    if any(
        path.is_symlink() or (not path.is_file() and not path.is_dir())
        for path in pending.rglob("*")
    ):
        raise ValueError("cold numerical pending bundle contains a special path")


def _validate_pending_diag4_numerical_bundle(
    publication: ColdNumericalBundlePublication,
    producer: Mapping[str, JsonValue],
) -> None:
    """Validate the exact trace-free scientific subgroup before parent commit."""

    pending = publication.pending_root
    if pending.is_symlink() or not pending.is_dir():
        raise ValueError("DIAG4 numerical pending bundle is absent")
    validate_diag4_producer_payload(producer, mode="cold")
    expected: set[Path] = set()
    references: dict[str, ArtifactRef] = {}
    for field, expected_relative in (
        ("history_evidence", "cold/numerical-result/history.json"),
        (
            "terminal_numerical_evidence",
            "cold/numerical-result/terminal-numerical.json",
        ),
        ("solve_timing_evidence", "cold/numerical-result/solve-timing.json"),
        (
            "safeguard_telemetry_evidence",
            "cold/numerical-result/safeguard-telemetry.json",
        ),
    ):
        reference = _artifact_from_payload(_mapping_field(producer, field))
        if reference.relative_path != expected_relative:
            raise ValueError(f"DIAG4 {field} path differs from the committed layout")
        references[field] = reference
        expected.add(_pending_path_for_logical_reference(pending, reference))
    policy_reference = _artifact_from_payload(
        _mapping_field(producer, "policy_evidence")
    )
    policy_path = pending.parent / "policy.json"
    policy_payload = policy_path.read_bytes()
    if (
        policy_reference.relative_path != "cold/policy.json"
        or len(policy_payload) != policy_reference.size_bytes
        or _sha256(policy_payload) != policy_reference.sha256
    ):
        raise ValueError("cold policy differs from the DIAG4 producer")

    terminal_path = pending / "terminal-numerical.json"
    terminal_payload = load_canonical_json_bytes(terminal_path.read_bytes())
    if not isinstance(terminal_payload, dict) or not isinstance(
        terminal_payload.get("arrays"), dict
    ):
        raise TypeError("DIAG4 terminal array ledger is absent")
    arrays = terminal_payload["arrays"]
    if frozenset(arrays) != frozenset(DIAGNOSTIC_ARRAY_SPECS):
        raise ValueError("DIAG4 terminal arrays differ")
    for name in DIAGNOSTIC_ARRAY_SPECS:
        row = arrays[name]
        if not isinstance(row, dict) or not isinstance(row.get("artifact"), dict):
            raise TypeError(f"DIAG4 terminal array {name} is malformed")
        reference = _artifact_from_payload(row["artifact"])
        if reference.relative_path != f"cold/numerical-result/arrays/{name}.npy":
            raise ValueError("DIAG4 terminal array path differs")
        expected.add(_pending_path_for_logical_reference(pending, reference))

    observed = {
        path for path in pending.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if observed != expected:
        raise ValueError("DIAG4 pending bundle has missing or extra files")
    if any(
        path.is_symlink()
        or (not path.is_file() and not path.is_dir())
        or (path.is_file() and path.stat().st_nlink != 1)
        for path in pending.rglob("*")
    ):
        raise ValueError("DIAG4 pending bundle contains a linked or special path")
    validate_diag4_numerical_documents(
        history=load_canonical_json_bytes((pending / "history.json").read_bytes()),
        terminal_numerical=terminal_payload,
        solve_timing=load_canonical_json_bytes(
            (pending / "solve-timing.json").read_bytes()
        ),
        safeguard_telemetry=load_canonical_json_bytes(
            (pending / "safeguard-telemetry.json").read_bytes()
        ),
        producer=producer,
        artifact_root=(
            pending.parents[1] if pending == publication.committed_root else None
        ),
    )


def _materialize_cold_numerical_bundle(
    publication: ColdNumericalBundlePublication,
) -> None:
    """Atomically publish one validated scientific-result directory."""

    _rename_noreplace_and_fsync_parent(
        publication.pending_root, publication.committed_root
    )


def _quarantine_cold_numerical_bundle(
    publication: ColdNumericalBundlePublication,
) -> None:
    """Atomically retain an uncommitted child tree without typing its contents."""

    _rename_noreplace_and_fsync_parent(
        publication.pending_root, publication.uncommitted_root
    )


def _resolve_cold_numerical_bundle(
    cold_root: Path, outcome: DiagnosticSupervisedSampleV2
) -> DiagnosticSupervisedSampleV2:
    publication = _cold_numerical_bundle_publication(cold_root)
    if not publication.pending_root.exists():
        if (
            outcome.producer is not None
            and outcome.producer.get("schema_version")
            == DIAG3_COLD_RESULT_SCHEMA_VERSION
        ):
            return replace(
                outcome,
                producer=None,
                producer_absence_reason=AbsenceReason.PRODUCER_SCHEMA_INVALID,
                selected_failure_reason=FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
                raw_failure_reasons=(
                    *outcome.raw_failure_reasons,
                    "PRODUCER_SCHEMA_INVALID:MISSING_NUMERICAL_BUNDLE",
                ),
            )
        return outcome
    producer = outcome.producer
    if producer is None or producer.get("execution_status") not in {
        "COMPLETE",
        "TRACE_NORMALIZATION_FAILED",
    }:
        _quarantine_cold_numerical_bundle(publication)
        return outcome
    try:
        _validate_pending_cold_numerical_bundle(publication, producer)
    except (OSError, TypeError, ValueError) as error:
        _quarantine_cold_numerical_bundle(publication)
        return replace(
            outcome,
            producer=None,
            producer_absence_reason=AbsenceReason.PRODUCER_SCHEMA_INVALID,
            selected_failure_reason=FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
            raw_failure_reasons=(
                *outcome.raw_failure_reasons,
                f"PRODUCER_SCHEMA_INVALID:{type(error).__name__}:{_sha256(str(error).encode())}",
            ),
        )
    _materialize_cold_numerical_bundle(publication)
    return outcome


def _diag4_untyped_cold_outcome(
    outcome: DiagnosticSupervisedSampleV2,
    failure: StructuredFailureV4,
) -> DiagnosticSupervisedSampleV2:
    return replace(
        outcome,
        producer=None,
        producer_absence_reason=AbsenceReason.PRODUCER_SCHEMA_INVALID,
        selected_failure_reason=FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
        raw_failure_reasons=(
            *outcome.raw_failure_reasons,
            f"{failure.stage.value}:{failure.reason.value}:{failure.detail_sha256}",
        ),
    )


def _diag5_untyped_cold_outcome(
    outcome: DiagnosticSupervisedSampleV2,
    failure: StructuredFailureV5,
) -> DiagnosticSupervisedSampleV2:
    return replace(
        outcome,
        producer=None,
        producer_absence_reason=AbsenceReason.PRODUCER_SCHEMA_INVALID,
        selected_failure_reason=FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
        raw_failure_reasons=(
            *outcome.raw_failure_reasons,
            f"{failure.stage.value}:{failure.reason.value}:{failure.detail_sha256}",
        ),
    )


def _validate_pending_diag5_numerical_bundle(
    publication: ColdNumericalBundlePublication,
    producer: Mapping[str, JsonValue],
) -> None:
    """Validate the exact DIAG5 scientific subgroup before parent commit."""

    pending = publication.pending_root
    if pending.is_symlink() or not pending.is_dir():
        raise ValueError("DIAG5 numerical pending bundle is absent")
    validate_diag5_producer_payload(producer, mode="cold")
    expected: set[Path] = set()
    references: dict[str, ArtifactRef] = {}
    for field, expected_relative in (
        ("history_evidence", "cold/numerical-result/history.json"),
        (
            "terminal_numerical_evidence",
            "cold/numerical-result/terminal-numerical.json",
        ),
        ("solve_timing_evidence", "cold/numerical-result/solve-timing.json"),
        (
            "safeguard_telemetry_evidence",
            "cold/numerical-result/safeguard-telemetry.json",
        ),
    ):
        reference = _artifact_from_payload(_mapping_field(producer, field))
        if reference.relative_path != expected_relative:
            raise ValueError(f"DIAG5 {field} path differs from committed layout")
        references[field] = reference
        expected.add(_pending_path_for_logical_reference(pending, reference))
    policy_reference = _artifact_from_payload(
        _mapping_field(producer, "policy_evidence")
    )
    policy_path = pending.parent / "policy.json"
    policy_payload = policy_path.read_bytes()
    if (
        policy_reference.relative_path != "cold/policy.json"
        or len(policy_payload) != policy_reference.size_bytes
        or _sha256(policy_payload) != policy_reference.sha256
    ):
        raise ValueError("cold policy differs from the DIAG5 producer")
    terminal_payload = load_canonical_json_bytes(
        (pending / "terminal-numerical.json").read_bytes()
    )
    if not isinstance(terminal_payload, dict) or not isinstance(
        terminal_payload.get("arrays"), dict
    ):
        raise TypeError("DIAG5 terminal array ledger is absent")
    arrays = terminal_payload["arrays"]
    if frozenset(arrays) != frozenset(DIAGNOSTIC_ARRAY_SPECS):
        raise ValueError("DIAG5 terminal arrays differ")
    for name in DIAGNOSTIC_ARRAY_SPECS:
        row = arrays[name]
        if not isinstance(row, dict) or not isinstance(row.get("artifact"), dict):
            raise TypeError(f"DIAG5 terminal array {name} is malformed")
        reference = _artifact_from_payload(row["artifact"])
        if reference.relative_path != f"cold/numerical-result/arrays/{name}.npy":
            raise ValueError("DIAG5 terminal array path differs")
        expected.add(_pending_path_for_logical_reference(pending, reference))
    observed = {
        path for path in pending.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if observed != expected or any(
        path.is_symlink()
        or (not path.is_file() and not path.is_dir())
        or (path.is_file() and path.stat().st_nlink != 1)
        for path in pending.rglob("*")
    ):
        raise ValueError("DIAG5 pending bundle membership or link identity differs")
    validate_diag5_numerical_documents(
        history=load_canonical_json_bytes((pending / "history.json").read_bytes()),
        terminal_numerical=terminal_payload,
        solve_timing=load_canonical_json_bytes(
            (pending / "solve-timing.json").read_bytes()
        ),
        safeguard_telemetry=load_canonical_json_bytes(
            (pending / "safeguard-telemetry.json").read_bytes()
        ),
        producer=producer,
        artifact_root=(
            pending.parents[1] if pending == publication.committed_root else None
        ),
    )


def _validate_diag5_opaque_quarantine(root: Path) -> None:
    """Deep-load one quarantined tree without assigning scientific types."""

    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("DIAG5 opaque quarantine is not a directory")
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(
                "DIAG5 opaque quarantine contains a linked or special path"
            )
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            while os.read(descriptor, 1024 * 1024):
                pass
        finally:
            os.close(descriptor)


def _quarantine_and_validate_diag5_numerical_bundle(
    publication: ColdNumericalBundlePublication,
    *,
    selected_failure_reason: FailureReasonCodeV5,
) -> None:
    _quarantine_cold_numerical_bundle(publication)
    publication.uncommitted_root.chmod(0o555)
    _fsync_parent(publication.uncommitted_root)
    if not any(publication.uncommitted_root.iterdir()):
        marker_path = publication.uncommitted_root.with_name(
            "uncommitted-numerical-result.empty.json"
        )
        _publish_canonical_json(
            marker_path,
            {
                "schema_version": "single-stage-neq-gntr3-empty-quarantine-v1",
                "route": DIAG5_ROUTE,
                "quarantine_relative_path": ("cold/uncommitted-numerical-result"),
                "selected_failure_reason": selected_failure_reason.value,
            },
        )
        _fsync_parent(marker_path)
    _validate_diag5_opaque_quarantine(publication.uncommitted_root)


def _resolve_diag5_cold_numerical_bundle_v5(
    cold_root: Path,
    outcome: DiagnosticSupervisedSampleV2,
) -> Diag5ColdNumericalResolution:
    """Converge every DIAG5 pending-tree boundary to one typed outcome."""

    publication = _cold_numerical_bundle_publication(cold_root)
    if not os.path.lexists(publication.pending_root):
        if (
            outcome.producer is not None
            and outcome.producer.get("execution_status") == "COMPLETE"
        ):
            failure = _diag5_failure(
                FailureStageV5.NUMERICAL_COMMIT,
                FailureReasonCodeV5.PENDING_RESULT_ABSENT,
                "zero-exit DIAG5 cold omitted pending numerical result",
            )
            return Diag5ColdNumericalResolution(
                outcome,
                failure,
                None,
            )
        return Diag5ColdNumericalResolution(outcome, None, None)
    producer = outcome.producer
    if producer is None or producer.get("execution_status") != "COMPLETE":
        selected_failure = _diag5_child_failure(DiagnosticChildMode.COLD, outcome)
        if selected_failure is None:
            raise ValueError("DIAG5 quarantine requires one selected cold failure")
        try:
            _quarantine_and_validate_diag5_numerical_bundle(
                publication,
                selected_failure_reason=selected_failure.reason,
            )
        except (OSError, TypeError, ValueError) as error:
            failure = _diag5_failure(
                FailureStageV5.NUMERICAL_COMMIT,
                FailureReasonCodeV5.QUARANTINE_FAILED,
                f"{type(error).__name__}:{error}",
            )
            return Diag5ColdNumericalResolution(outcome, None, failure)
        return Diag5ColdNumericalResolution(outcome, None, None)
    try:
        _validate_pending_diag5_numerical_bundle(publication, producer)
    except (OSError, TypeError, ValueError) as error:
        detail = (
            f"{error.reason.value}:{error.detail_sha256}"
            if isinstance(error, Diag5NumericalDocumentError)
            else f"{type(error).__name__}:{error}"
        )
        failure = _diag5_failure(
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.PENDING_RESULT_INVALID,
            detail,
        )
        try:
            _quarantine_and_validate_diag5_numerical_bundle(
                publication,
                selected_failure_reason=FailureReasonCodeV5.PENDING_RESULT_INVALID,
            )
        except (OSError, TypeError, ValueError) as quarantine_error:
            disposition_failure = _diag5_failure(
                FailureStageV5.NUMERICAL_COMMIT,
                FailureReasonCodeV5.QUARANTINE_FAILED,
                f"{type(quarantine_error).__name__}:{quarantine_error}",
            )
            return Diag5ColdNumericalResolution(
                outcome,
                disposition_failure,
                disposition_failure,
            )
        return Diag5ColdNumericalResolution(
            outcome,
            failure,
            None,
        )
    try:
        _materialize_cold_numerical_bundle(publication)
    except FileExistsError as error:
        failure = _diag5_failure(
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.COMMIT_COLLISION,
            f"{type(error).__name__}:{error}",
        )
        return Diag5ColdNumericalResolution(outcome, failure, failure)
    except OSError as error:
        renamed = (
            not publication.pending_root.exists()
            and publication.committed_root.exists()
        )
        failure = _diag5_failure(
            FailureStageV5.NUMERICAL_COMMIT,
            (
                FailureReasonCodeV5.COMMIT_FSYNC_FAILED
                if renamed
                else FailureReasonCodeV5.COMMIT_RENAME_FAILED
            ),
            f"{type(error).__name__}:{error}",
        )
        return Diag5ColdNumericalResolution(outcome, failure, failure)
    try:
        _validate_pending_diag5_numerical_bundle(
            replace(publication, pending_root=publication.committed_root),
            producer,
        )
    except (OSError, TypeError, ValueError) as error:
        failure = _diag5_failure(
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.COMMITTED_DEEP_LOAD_FAILED,
            f"{type(error).__name__}:{error}",
        )
        return Diag5ColdNumericalResolution(outcome, failure, failure)
    return Diag5ColdNumericalResolution(outcome, None, None)


def _resolve_diag4_cold_numerical_bundle_v4(
    cold_root: Path, outcome: DiagnosticSupervisedSampleV2
) -> tuple[DiagnosticSupervisedSampleV2, StructuredFailureV4 | None]:
    """Converge every pending-tree boundary to one exact v4 commit outcome."""

    publication = _cold_numerical_bundle_publication(cold_root)
    if not publication.pending_root.exists():
        if (
            outcome.producer is not None
            and outcome.producer.get("schema_version")
            == DIAG4_COLD_RESULT_SCHEMA_VERSION
        ):
            failure = _diag4_failure(
                FailureStageV4.NUMERICAL_COMMIT,
                FailureReasonCodeV4.PENDING_RESULT_ABSENT,
                "zero-exit DIAG4 cold omitted its pending numerical result",
            )
            return _diag4_untyped_cold_outcome(outcome, failure), failure
        return outcome, None
    producer = outcome.producer
    if producer is None or producer.get("execution_status") != "COMPLETE":
        try:
            _quarantine_cold_numerical_bundle(publication)
        except (OSError, TypeError, ValueError) as error:
            failure = _diag4_failure(
                FailureStageV4.NUMERICAL_COMMIT,
                FailureReasonCodeV4.QUARANTINE_FAILED,
                f"{type(error).__name__}:{error}",
            )
            return _diag4_untyped_cold_outcome(outcome, failure), failure
        return outcome, None
    try:
        _validate_pending_diag4_numerical_bundle(publication, producer)
    except (OSError, TypeError, ValueError) as error:
        reason = (
            error.reason
            if isinstance(error, Diag4NumericalDocumentError)
            else FailureReasonCodeV4.PENDING_RESULT_INVALID
        )
        failure = StructuredFailureV4(
            FailureStageV4.NUMERICAL_COMMIT,
            reason,
            (
                error.detail_sha256
                if isinstance(error, Diag4NumericalDocumentError)
                else _sha256(f"{type(error).__name__}:{error}".encode())
            ),
        )
        try:
            _quarantine_cold_numerical_bundle(publication)
        except (OSError, TypeError, ValueError) as quarantine_error:
            failure = _diag4_failure(
                FailureStageV4.NUMERICAL_COMMIT,
                FailureReasonCodeV4.QUARANTINE_FAILED,
                f"{type(quarantine_error).__name__}:{quarantine_error}",
            )
        return _diag4_untyped_cold_outcome(outcome, failure), failure
    try:
        _materialize_cold_numerical_bundle(publication)
    except FileExistsError as error:
        failure = _diag4_failure(
            FailureStageV4.NUMERICAL_COMMIT,
            FailureReasonCodeV4.COMMIT_COLLISION,
            f"{type(error).__name__}:{error}",
        )
        return _diag4_untyped_cold_outcome(outcome, failure), failure
    except OSError as error:
        renamed = (
            not publication.pending_root.exists()
            and publication.committed_root.exists()
        )
        failure = _diag4_failure(
            FailureStageV4.NUMERICAL_COMMIT,
            (
                FailureReasonCodeV4.COMMIT_FSYNC_FAILED
                if renamed
                else FailureReasonCodeV4.COMMIT_RENAME_FAILED
            ),
            f"{type(error).__name__}:{error}",
        )
        return _diag4_untyped_cold_outcome(outcome, failure), failure
    try:
        _validate_pending_diag4_numerical_bundle(
            replace(publication, pending_root=publication.committed_root),
            producer,
        )
    except (OSError, TypeError, ValueError) as error:
        failure = _diag4_failure(
            FailureStageV4.NUMERICAL_COMMIT,
            FailureReasonCodeV4.COMMITTED_DEEP_LOAD_FAILED,
            f"{type(error).__name__}:{error}",
        )
        return _diag4_untyped_cold_outcome(outcome, failure), failure
    return outcome, None


def _resolve_diag4_cold_numerical_bundle(
    cold_root: Path, outcome: DiagnosticSupervisedSampleV2
) -> DiagnosticSupervisedSampleV2:
    """Compatibility wrapper returning the supervised outcome only."""

    resolved, _ = _resolve_diag4_cold_numerical_bundle_v4(cold_root, outcome)
    return resolved


def _published_runtime_reference(
    campaign_root: Path,
    directory: Path,
) -> ArtifactRef | None:
    path = directory / RUNTIME_EVIDENCE_FILENAME
    if not path.is_file():
        return None
    return _artifact_ref(path, campaign_root, RUNTIME_EVIDENCE_SCHEMA_VERSION)


def _published_diagnostic_policy_reference(
    campaign_root: Path,
    directory: Path,
) -> ArtifactRef | None:
    path = directory / "policy.json"
    if not path.is_file():
        return None
    return _artifact_ref(
        path,
        campaign_root,
        f"{DIAGNOSTIC_SCHEMA_VERSION}-policy",
    )


def build_sample_receipt(
    supervised: SupervisedSample,
    *,
    producer_reference: ArtifactRef,
    publication: SnapshotPublication,
    campaign_root: Path,
    reference: ReferenceReceipt,
) -> SampleReceipt:
    """Bind raw worker and parent observations to one strict typed receipt."""

    producer = supervised.producer
    execution_status = _execution_status(supervised.terminal_status)
    reached = producer.get("candidate_reached") is True
    timing_payload = _mapping_field(producer, "timing") if producer else {}
    timer_stopped = (
        _integer_field(timing_payload, "timer_stopped_ns")
        if "timer_stopped_ns" in timing_payload
        else 0
    )
    compile_completed = (
        _integer_field(timing_payload, "compile_completed_ns")
        if "compile_completed_ns" in timing_payload
        else 0
    )
    device_ready = (
        _integer_field(timing_payload, "device_state_ready_ns")
        if "device_state_ready_ns" in timing_payload
        else compile_completed
    )
    timer_started = (
        _integer_field(timing_payload, "timer_started_ns")
        if "timer_started_ns" in timing_payload
        else device_ready
    )
    serialized = (
        _integer_field(timing_payload, "serialized_ns")
        if "serialized_ns" in timing_payload
        else max(timer_started, timer_stopped)
    )
    solve_seconds = (
        _number_field(timing_payload, "synchronized_solve_seconds")
        if "synchronized_solve_seconds" in timing_payload
        else 0.0
    )
    audit_started = (
        _integer_field(timing_payload, "audit_started_ns") if reached else None
    )
    final_transfer = (
        _integer_field(timing_payload, "final_transfer_ns") if reached else None
    )
    endpoint_audit_seconds = (
        _number_field(timing_payload, "endpoint_audit_seconds") if reached else None
    )
    timing = TimingEvidence(
        compile_completed_ns=compile_completed,
        device_state_ready_ns=device_ready,
        timer_started_ns=timer_started,
        first_hit_synchronized_ns=timer_stopped if reached else None,
        timer_stopped_ns=timer_stopped,
        audit_started_ns=audit_started,
        final_transfer_ns=final_transfer,
        serialized_ns=serialized,
        synchronized_solve_seconds=solve_seconds,
        endpoint_audit_seconds=endpoint_audit_seconds,
        total_process_seconds=supervised.process_seconds,
    )
    if reached:
        candidate_payload = _mapping_field(producer, "candidate")
        raw_equalities = candidate_payload.get("raw_equalities")
        scaled_equalities = candidate_payload.get("scaled_equalities")
        if not isinstance(raw_equalities, list) or not isinstance(
            scaled_equalities, list
        ):
            raise TypeError("candidate equality vectors must be arrays")
        candidate = CandidateEvidence(
            reached=True,
            first_hit_attempt=_integer_field(candidate_payload, "first_hit_attempt"),
            first_hit_accepted_step=_integer_field(
                candidate_payload, "first_hit_accepted_step"
            ),
            accepted_step_count=_integer_field(
                candidate_payload, "accepted_step_count"
            ),
            state_sha256=str(candidate_payload["state_sha256"]),
            physical_objective=_number_field(candidate_payload, "physical_objective"),
            raw_equalities=tuple(float(value) for value in raw_equalities),
            scaled_equalities=tuple(float(value) for value in scaled_equalities),
            scaled_feasibility_inf=_number_field(
                candidate_payload, "scaled_feasibility_inf"
            ),
            state_dtype=str(candidate_payload["state_dtype"]),
            equality_dtype=str(candidate_payload["equality_dtype"]),
            correction_certified=candidate_payload.get("correction_certified") is True,
        )
        endpoint_value = producer.get("endpoint_audit")
        endpoint_audit = EndpointAuditEvidence.from_payload(endpoint_value)
    else:
        candidate = CandidateEvidence(
            False,
            None,
            None,
            _noncandidate_accepted_step_count(producer),
            None,
            None,
            (),
            (),
            None,
            None,
            None,
            False,
        )
        endpoint_audit = None
    pre_source = supervised.pre_source_identity
    post_source = supervised.post_source_identity
    if pre_source is None or post_source is None:
        raise ValueError("independent pre/post source identities were not captured")
    source_manifest = publication.source_identity(campaign_root).snapshot_manifest
    runtime_reference = _artifact_from_payload(
        _mapping_field(producer, "runtime_evidence")
    )
    runtime_document = load_canonical_json_bytes(
        runtime_reference.resolve_and_validate(campaign_root).read_bytes()
    )
    if not isinstance(runtime_document, dict):
        raise TypeError("runtime evidence must be an object")
    runtime_identity = _mapping_field(runtime_document, "runtime_identity")
    runtime_environment_sha256 = runtime_identity.get("effective_environment_sha256")
    if not isinstance(runtime_environment_sha256, str):
        raise TypeError("runtime environment identity is absent")
    runtime = _mapping_field(producer, "runtime")
    transfers = (
        _mapping_field(producer, "transfer_audit")
        if producer.get("transfer_audit") is not None
        else {}
    )
    memory = supervised.memory or {}
    resources = ResourceEvidence(
        source_identity_sha256=_sha256(canonical_json_bytes(pre_source.to_payload())),
        pre_source_identity=pre_source,
        post_source_identity=post_source,
        source_manifest=source_manifest,
        runtime_environment_sha256=runtime_environment_sha256,
        runtime_evidence=runtime_reference,
        reference_policy_sha256=(
            str(producer["reference_policy_sha256"])
            if producer.get("reference_policy_sha256") is not None
            else str(reference.reference_policy_sha256)
        ),
        backend=str(runtime.get("backend", "unobserved")),
        device_uuid=str(runtime.get("device_uuid", GPU_UUID)),
        jax_enable_x64=runtime.get("jax_enable_x64") is True,
        child_pid=supervised.child_pid,
        child_start_time_ticks=supervised.child_start_time_ticks,
        hot_h2d_transfers=int(transfers.get("hot_h2d_transfers", 0)),
        hot_d2h_transfers=int(transfers.get("hot_d2h_transfers", 0)),
        python_callbacks=int(transfers.get("python_callbacks", 0)),
        final_d2h_transfers=int(transfers.get("final_d2h_transfers", 0)),
        peak_memory_fraction=float(memory.get("peak_memory_fraction", 1.0)),
    )
    failure_reasons = (
        ()
        if execution_status is ExecutionStatus.COMPLETED
        else supervised.failure_reasons or (execution_status.value,)
    )
    receipt = SampleReceipt(
        sample=supervised.sample,
        producer_evidence=producer_reference,
        execution_status=execution_status,
        timing=timing,
        candidate=candidate,
        endpoint_audit=endpoint_audit,
        resources=resources,
        kkt_telemetry=KktTelemetry(KktTelemetryStatus.UNAVAILABLE, None, None),
        failure_reasons=tuple(failure_reasons),
    )
    receipt.validate()
    return receipt


def _preflight_disposition(outcome: SupervisedSample) -> str:
    worker_status = outcome.producer.get("execution_status")
    if outcome.terminal_status is ChildTerminalStatus.COMPLETE:
        return (
            str(worker_status) if isinstance(worker_status, str) else "PROTOCOL_FAILURE"
        )
    if (
        outcome.terminal_status is ChildTerminalStatus.COMPILE_FAILURE
        and worker_status in {"COMPILE_FAILURE", "COMPILE_OOM"}
    ):
        return str(worker_status)
    if outcome.terminal_status is ChildTerminalStatus.TIMEOUT:
        return "COMPILE_TIMEOUT"
    return outcome.terminal_status.value


def _publish_diagnostic_supervision(
    campaign_root: Path,
    directory: Path,
    outcome: SupervisedSample,
    *,
    producer_schema: str,
) -> dict[str, ArtifactRef]:
    """Publish unsummarized child, process, memory, and terminal evidence."""

    producer_path = directory / "producer.json"
    terminal_path = directory / "terminal.json"
    process_path = directory / "process.json"
    stdout_path = directory / "stdout.bin"
    stderr_path = directory / "stderr.bin"
    _publish_canonical_json(producer_path, outcome.producer)
    _publish_bytes(stdout_path, outcome.stdout or b"")
    _publish_bytes(stderr_path, outcome.stderr or b"")
    process_payload: dict[str, JsonValue] = {
        "schema_version": f"{DIAGNOSTIC_SCHEMA_VERSION}-process",
        "child_pid": outcome.child_pid,
        "child_start_time_ticks": outcome.child_start_time_ticks,
        "argv": list(outcome.observed_child_argv or ()),
        "stdout": _artifact_ref_payload(
            _artifact_ref(stdout_path, campaign_root, "raw-process-stdout-v1")
        ),
        "stderr": _artifact_ref_payload(
            _artifact_ref(stderr_path, campaign_root, "raw-process-stderr-v1")
        ),
        "process_seconds": outcome.process_seconds,
        "process_diagnostics": outcome.process_diagnostics or {},
        "pre_source_identity": (
            outcome.pre_source_identity.to_payload()
            if outcome.pre_source_identity is not None
            else None
        ),
        "post_source_identity": (
            outcome.post_source_identity.to_payload()
            if outcome.post_source_identity is not None
            else None
        ),
    }
    _publish_canonical_json(process_path, process_payload)
    terminal_payload: dict[str, JsonValue] = {
        "schema_version": f"{DIAGNOSTIC_SCHEMA_VERSION}-child-terminal",
        "terminal_status": outcome.terminal_status.value,
        "failure_reasons": list(outcome.failure_reasons),
    }
    _publish_canonical_json(terminal_path, terminal_payload)
    references = {
        "producer": _artifact_ref(producer_path, campaign_root, producer_schema),
        "process": _artifact_ref(
            process_path, campaign_root, f"{DIAGNOSTIC_SCHEMA_VERSION}-process"
        ),
        "child_terminal": _artifact_ref(
            terminal_path,
            campaign_root,
            f"{DIAGNOSTIC_SCHEMA_VERSION}-child-terminal",
        ),
    }
    if outcome.memory is not None:
        memory_path = directory / "gpu-memory.json"
        _publish_canonical_json(memory_path, outcome.memory)
        references["memory"] = _artifact_ref(
            memory_path, campaign_root, MEMORY_SCHEMA_VERSION
        )
    raw_samples_path = directory / "gpu-memory-samples.json"
    _publish_canonical_json(
        raw_samples_path,
        {
            "schema_version": f"{DIAGNOSTIC_SCHEMA_VERSION}-memory-samples",
            "samples": [
                {
                    "sampled_at_unix_ns": sample.sampled_at_unix_ns,
                    "used_memory_mib": sample.used_memory_mib,
                }
                for sample in outcome.memory_samples or ()
            ],
        },
    )
    references["memory_samples"] = _artifact_ref(
        raw_samples_path,
        campaign_root,
        f"{DIAGNOSTIC_SCHEMA_VERSION}-memory-samples",
    )
    return references


def _publish_diag2_supervision(
    staging_root: Path,
    directory: Path,
    outcome: DiagnosticSupervisedSampleV2,
    *,
    producer_schema: str,
    process_schema: str = DIAG2_PROCESS_SCHEMA_VERSION,
    terminal_schema: str = DIAG2_CHILD_TERMINAL_SCHEMA_VERSION,
    memory_schema: str = MEMORY_SCHEMA_VERSION,
    memory_samples_schema: str = f"{DIAGNOSTIC_SCHEMA_VERSION}-memory-samples",
) -> dict[str, ArtifactRef | None]:
    """Publish raw supervision, leaving an absent producer physically absent."""

    if not directory.is_dir():
        raise ValueError("DIAG2 child directory must exist before supervision")
    stdout_path = directory / "stdout.bin"
    stderr_path = directory / "stderr.bin"
    process_path = directory / "process.json"
    terminal_path = directory / "terminal.json"
    _publish_bytes(stdout_path, outcome.stdout)
    _publish_bytes(stderr_path, outcome.stderr)
    _publish_canonical_json(
        process_path,
        {
            "schema_version": process_schema,
            "child_pid": outcome.child_pid,
            "child_start_time_ticks": outcome.child_start_time_ticks,
            "argv": list(outcome.observed_child_argv or ()),
            "stdout": _artifact_ref_payload(
                _artifact_ref(stdout_path, staging_root, "raw-process-stdout-v1")
            ),
            "stderr": _artifact_ref_payload(
                _artifact_ref(stderr_path, staging_root, "raw-process-stderr-v1")
            ),
            "process_seconds": outcome.process_seconds,
            "process_started_monotonic_ns": outcome.process_started_monotonic_ns,
            "process_stopped_monotonic_ns": outcome.process_stopped_monotonic_ns,
            "monitor_failure_kind": outcome.monitor_failure_kind.value,
            "process_diagnostics": outcome.process_diagnostics or {},
            "pre_source_identity": (
                None
                if outcome.pre_source_identity is None
                else outcome.pre_source_identity.to_payload()
            ),
            "post_source_identity": (
                None
                if outcome.post_source_identity is None
                else outcome.post_source_identity.to_payload()
            ),
        },
    )
    _publish_canonical_json(
        terminal_path,
        {
            "schema_version": terminal_schema,
            "terminal_status": outcome.terminal_status.value,
            "monitor_failure_kind": outcome.monitor_failure_kind.value,
            "failure_reasons": list(outcome.raw_failure_reasons),
        },
    )
    producer_reference: ArtifactRef | None = None
    if outcome.producer is not None:
        producer_path = directory / "producer.json"
        _publish_canonical_json(producer_path, outcome.producer)
        producer_reference = _artifact_ref(producer_path, staging_root, producer_schema)
    memory_reference: ArtifactRef | None = None
    if outcome.memory is not None:
        memory_path = directory / "gpu-memory.json"
        _publish_canonical_json(
            memory_path,
            {**outcome.memory, "schema_version": memory_schema},
        )
        memory_reference = _artifact_ref(memory_path, staging_root, memory_schema)
    samples_reference: ArtifactRef | None = None
    if outcome.memory is not None:
        samples_path = directory / "gpu-memory-samples.json"
        _publish_canonical_json(
            samples_path,
            {
                "schema_version": memory_samples_schema,
                "samples": [
                    {
                        "sampled_at_unix_ns": sample.sampled_at_unix_ns,
                        "used_memory_mib": sample.used_memory_mib,
                    }
                    for sample in outcome.memory_samples
                ],
            },
        )
        samples_reference = _artifact_ref(
            samples_path,
            staging_root,
            memory_samples_schema,
        )
    return {
        "producer": producer_reference,
        "terminal": _artifact_ref(
            terminal_path,
            staging_root,
            terminal_schema,
        ),
        "process": _artifact_ref(
            process_path,
            staging_root,
            process_schema,
        ),
        "memory": memory_reference,
        "memory_samples": samples_reference,
    }


def run_preflight(
    campaign_root: Path,
    *,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    repo_root: Path | None = None,
) -> SupervisedSample:
    """Run and seal one supervised compile-only production-shape preflight."""

    (
        repository,
        native_extension,
        reference,
        inputs,
        executable,
        gpu_uuid,
        memory_bytes,
    ) = _validate_parent_execution_policy(
        repo_root=repo_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
        environment=environment,
    )
    publication = prepare_execution_snapshot(
        campaign_root,
        repo_root=repository,
        native_extension_path=native_extension,
    )
    campaign_reference = copy_validated_reference(reference, campaign_root)
    preflight_root = campaign_root / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=False)
    pre_source = _capture_source_identity_evidence(publication, campaign_root)
    invocation = build_child_invocation(
        publication,
        campaign_root=campaign_root,
        interpreter=executable,
        reference_root=campaign_reference,
        input_root=inputs,
        sample=SampleName.COLD,
        environment=environment,
        preflight_only=True,
    )
    try:
        outcome = supervise_sample(
            SampleName.COLD,
            invocation,
            gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
        )
    except Exception as error:  # noqa: BLE001 - retain supervisor launch crashes.
        outcome = SupervisedSample(
            sample=SampleName.COLD,
            terminal_status=ChildTerminalStatus.CRASH,
            child_pid=0,
            child_start_time_ticks=0,
            process_seconds=0.0,
            producer={},
            memory=None,
            failure_reasons=(
                f"SUPERVISOR:{type(error).__name__}:{_sha256(str(error).encode())}",
            ),
        )
    post_source = _capture_source_identity_evidence(publication, campaign_root)
    outcome = replace(
        outcome,
        pre_source_identity=pre_source,
        post_source_identity=post_source,
    )
    producer_path = preflight_root / "producer.json"
    memory_path = preflight_root / "gpu-memory.json"
    terminal_path = preflight_root / "terminal.json"
    _publish_canonical_json(producer_path, outcome.producer)
    if outcome.memory is not None:
        _publish_canonical_json(memory_path, outcome.memory)
    disposition = _preflight_disposition(outcome)
    terminal = {
        "schema_version": "single-stage-neq-gntr1-preflight-terminal-v1",
        "route": ROUTE,
        "mode": "LOWER_COMPILE_ONLY",
        "terminal_disposition": disposition,
        "campaign_authorized": False,
        "child_pid": outcome.child_pid,
        "child_start_time_ticks": outcome.child_start_time_ticks,
        "process_seconds": outcome.process_seconds,
        "failure_reasons": list(outcome.failure_reasons),
        "process_diagnostics": outcome.process_diagnostics,
    }
    _publish_canonical_json(terminal_path, terminal)
    artifact = {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "route": ROUTE,
        "plan_sha256": PLAN_SHA256,
        "mode": "LOWER_COMPILE_ONLY",
        "terminal_disposition": disposition,
        "campaign_authorized": False,
        "solver_dispatched": False,
        "finalizer_called": False,
        "endpoint_audit_called": False,
        "pre_source_identity": pre_source.to_payload(),
        "post_source_identity": post_source.to_payload(),
        "producer_evidence": _artifact_ref_payload(
            _artifact_ref(producer_path, campaign_root, PREFLIGHT_SCHEMA_VERSION)
        ),
        "memory_evidence": (
            _artifact_ref_payload(
                _artifact_ref(memory_path, campaign_root, MEMORY_SCHEMA_VERSION)
            )
            if outcome.memory is not None
            else None
        ),
        "terminal_evidence": _artifact_ref_payload(
            _artifact_ref(
                terminal_path,
                campaign_root,
                "single-stage-neq-gntr1-preflight-terminal-v1",
            )
        ),
    }
    _publish_canonical_json(campaign_root / "preflight.json", artifact)
    _publish_campaign_artifact_manifest(campaign_root)
    _seal_campaign_tree(campaign_root)
    return outcome


def _seal_diagnostic_receipt(
    campaign_root: Path,
    receipt: object,
) -> None:
    _publish_bytes(
        campaign_root / DIAGNOSTIC_RECEIPT_FILENAME,
        diagnostic_receipt_bytes(receipt),
    )
    _publish_diagnostic_artifact_manifest(campaign_root)
    _seal_campaign_tree(campaign_root)
    load_and_validate_diagnostic_artifact(campaign_root)


def _build_and_seal_diagnostic_receipt(
    campaign_root: Path,
    evidence_refs: Mapping[str, ArtifactRef | None],
) -> object:
    """Publish a complete receipt or receipt-derived immutable failure evidence."""

    complete_refs = {
        name: reference
        for name, reference in evidence_refs.items()
        if reference is not None
    }
    if len(complete_refs) == len(evidence_refs):
        try:
            receipt = build_diagnostic_receipt(
                artifact_root=campaign_root,
                evidence_refs=complete_refs,
            )
        except (OSError, TypeError, ValueError):
            receipt = build_incomplete_diagnostic_receipt(
                artifact_root=campaign_root,
                evidence_refs=evidence_refs,
            )
    else:
        receipt = build_incomplete_diagnostic_receipt(
            artifact_root=campaign_root,
            evidence_refs=evidence_refs,
        )
    _seal_diagnostic_receipt(campaign_root, receipt)
    return receipt


def _diag2_failure(
    stage: FailureStageV2,
    reason: FailureReasonCodeV2,
    detail: str,
) -> StructuredFailureV2:
    return StructuredFailureV2(stage, reason, _sha256(detail.encode("utf-8")))


def _diag2_child_failure(
    mode: DiagnosticChildMode,
    outcome: DiagnosticSupervisedSampleV2,
) -> StructuredFailureV2 | None:
    reason = outcome.selected_failure_reason
    if reason is None:
        return None
    preflight = mode is DiagnosticChildMode.PREFLIGHT
    stage_by_reason = {
        FailureReasonCodeV2.SOURCE_POST: (
            FailureStageV2.PREFLIGHT_SOURCE_FAILURE
            if preflight
            else FailureStageV2.COLD_SOURCE_FAILURE
        ),
        FailureReasonCodeV2.CHILD_LAUNCH_FAILED: (
            FailureStageV2.PREFLIGHT_SUPERVISOR_FAILURE
            if preflight
            else FailureStageV2.COLD_SUPERVISOR_FAILURE
        ),
        FailureReasonCodeV2.CHILD_TIMEOUT: (
            FailureStageV2.PREFLIGHT_TIMEOUT
            if preflight
            else FailureStageV2.COLD_TIMEOUT
        ),
        FailureReasonCodeV2.MONITOR_BINDING_FAILED: (
            FailureStageV2.PREFLIGHT_MONITOR_FAILURE
            if preflight
            else FailureStageV2.COLD_MONITOR_FAILURE
        ),
        FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: (
            FailureStageV2.PREFLIGHT_MONITOR_FAILURE
            if preflight
            else FailureStageV2.COLD_MONITOR_FAILURE
        ),
        FailureReasonCodeV2.PRODUCER_DECODE_FAILED: (
            FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            if preflight
            else FailureStageV2.COLD_PROTOCOL_FAILURE
        ),
        FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: (
            FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            if preflight
            else FailureStageV2.COLD_PROTOCOL_FAILURE
        ),
        FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID: (
            FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            if preflight
            else FailureStageV2.COLD_PROTOCOL_FAILURE
        ),
        FailureReasonCodeV2.POLICY_SCHEMA_INVALID: (
            FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            if preflight
            else FailureStageV2.COLD_PROTOCOL_FAILURE
        ),
        FailureReasonCodeV2.CHILD_COMPILE_FAILED: (
            FailureStageV2.PREFLIGHT_COMPILE_FAILURE
            if preflight
            else FailureStageV2.COLD_COMPILE_FAILURE
        ),
        FailureReasonCodeV2.CHILD_COMPILE_OOM: (
            FailureStageV2.PREFLIGHT_COMPILE_FAILURE
            if preflight
            else FailureStageV2.COLD_COMPILE_FAILURE
        ),
        FailureReasonCodeV2.CHILD_EXIT_NONZERO: (
            FailureStageV2.PREFLIGHT_CRASH if preflight else FailureStageV2.COLD_CRASH
        ),
        FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED: (
            FailureStageV2.PREFLIGHT_RESOURCE_FAILURE
            if preflight
            else FailureStageV2.COLD_RESOURCE_FAILURE
        ),
        FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED: (
            FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE
        ),
    }
    stage = stage_by_reason.get(reason)
    if stage is None:
        raise ValueError("DIAG2 child failure reason is not a child-stage reason")
    detail = "|".join(outcome.raw_failure_reasons) or reason.value
    return _diag2_failure(stage, reason, detail)


def _diag4_child_failure(
    mode: DiagnosticChildMode,
    outcome: DiagnosticSupervisedSampleV2,
) -> StructuredFailureV4 | None:
    """Map one retained child failure onto the exact additive v4 table."""

    reason = outcome.selected_failure_reason
    if reason is None:
        return None
    preflight = mode is DiagnosticChildMode.PREFLIGHT
    reason_map = {
        FailureReasonCodeV2.SOURCE_POST: (
            FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.CHILD_LAUNCH_FAILED: (
            FailureReasonCodeV4.PREFLIGHT_LAUNCH_FAILED
            if preflight
            else FailureReasonCodeV4.COLD_LAUNCH_FAILED
        ),
        FailureReasonCodeV2.CHILD_TIMEOUT: (
            FailureReasonCodeV4.PREFLIGHT_TIMEOUT
            if preflight
            else FailureReasonCodeV4.COLD_TIMEOUT
        ),
        FailureReasonCodeV2.MONITOR_BINDING_FAILED: (
            FailureReasonCodeV4.PREFLIGHT_MONITOR_FAILED
            if preflight
            else FailureReasonCodeV4.COLD_MONITOR_FAILED
        ),
        FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: (
            FailureReasonCodeV4.PREFLIGHT_MONITOR_FAILED
            if preflight
            else FailureReasonCodeV4.COLD_MONITOR_FAILED
        ),
        FailureReasonCodeV2.CHILD_EXIT_NONZERO: (
            FailureReasonCodeV4.PREFLIGHT_EXIT_NONZERO
            if preflight
            else FailureReasonCodeV4.COLD_EXIT_NONZERO
        ),
        FailureReasonCodeV2.PRODUCER_DECODE_FAILED: (
            FailureReasonCodeV4.PREFLIGHT_PRODUCER_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PRODUCER_INVALID
        ),
        FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: (
            FailureReasonCodeV4.PREFLIGHT_PRODUCER_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PRODUCER_INVALID
        ),
        FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID: (
            FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.POLICY_SCHEMA_INVALID: (
            FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED: (
            FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.CHILD_COMPILE_FAILED: (
            FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.CHILD_COMPILE_OOM: (
            FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV4.COLD_PROTOCOL_INVALID
        ),
    }
    mapped = reason_map.get(reason)
    if mapped is None:
        raise ValueError("DIAG4 child failure reason is not a child-stage reason")
    return _diag4_failure(
        FailureStageV4.PREFLIGHT if preflight else FailureStageV4.COLD,
        mapped,
        "|".join(outcome.raw_failure_reasons) or reason.value,
    )


def _diag2_subordinate_child_failure(
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    mode: DiagnosticChildMode,
    outcome: DiagnosticSupervisedSampleV2,
) -> StructuredFailureV2 | None:
    """Select a launched child outcome only from its retained raw authorities."""

    reason = classify_diag3_subordinate_child_outcome(
        artifact_root,
        artifact_refs=artifact_refs,
        mode=mode.value,
    )
    return _diag2_child_failure(
        mode,
        replace(outcome, selected_failure_reason=reason),
    )


def _diag2_postlaunch_setup_and_child_failure(
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    mode: DiagnosticChildMode,
    outcome: DiagnosticSupervisedSampleV2,
    *,
    reason: FailureReasonCodeV2,
    detail_sha256: str,
) -> StructuredFailureV2:
    """Select setup drift against the independently reconstructed child outcome."""

    setup_failure = diag2_postlaunch_setup_failure(
        after_mode=mode.value,
        reason=reason,
        detail_sha256=detail_sha256,
    )
    child_failure = _diag2_subordinate_child_failure(
        artifact_root,
        artifact_refs,
        mode,
        outcome,
    )
    return select_diag2_failure(
        (setup_failure, *((child_failure,) if child_failure is not None else ()))
    )


def _publish_diag2_policy_authority(
    staging_root: Path,
    reference_root: Path,
    reference: ArtifactRef,
) -> tuple[ArtifactRef, Diag2PolicyAuthorityValues]:
    values = _derive_diag2_policy_authority(reference_root)
    payload = build_diag2_policy_authority_payload(
        native_reference=reference,
        reference_volume=values.reference_volume,
        volume_target=values.volume_target,
        native_raw_equalities=values.native_raw_equalities,
        constraint_inverse_scale=values.constraint_inverse_scale,
    )
    validate_diag2_policy_authority_payload(payload, artifact_root=staging_root)
    path = staging_root / "policy-authority.json"
    _publish_canonical_json(path, payload)
    return (
        _artifact_ref(path, staging_root, DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION),
        values,
    )


def _publish_diag5_policy_authority(
    staging_root: Path,
    reference_root: Path,
    reference: ArtifactRef,
) -> tuple[ArtifactRef, Diag2PolicyAuthorityValues]:
    """Publish the independently identified DIAG5 policy authority."""

    values = _derive_diag2_policy_authority(reference_root)
    payload = build_diag5_policy_authority_payload(
        native_reference=reference,
        reference_volume=values.reference_volume,
        volume_target=values.volume_target,
        native_raw_equalities=values.native_raw_equalities,
        constraint_inverse_scale=values.constraint_inverse_scale,
    )
    validate_diag5_policy_authority_payload(payload, artifact_root=staging_root)
    path = staging_root / "policy-authority.json"
    _publish_canonical_json(path, payload)
    return (
        _artifact_ref(path, staging_root, DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION),
        values,
    )


def _publish_diag2_terminal_and_receipt(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    failure: StructuredFailureV2 | None,
    launched_children: tuple[str, ...],
    policy_authority_produced: bool,
    preflight_authorized: bool,
    cold_authorized: bool,
) -> object:
    """Derive, validate, durably seal, and atomically publish the sole receipt."""

    if failure is None:
        try:
            algorithm_route = derive_diag3_algorithm_route(
                artifact_root=publication.staging_root,
                artifact_refs=artifact_refs,
            )
        except (OSError, TypeError, ValueError) as error:
            failure = _diag2_failure(
                FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE,
                FailureReasonCodeV2.SEMANTIC_VALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
            algorithm_route = "NOT_PRODUCED"
    else:
        algorithm_route = "NOT_PRODUCED"
    terminal_payload = build_diag2_supervisor_terminal_payload(
        disposition="COMPLETE" if failure is None else "INCOMPLETE",
        failure=failure,
        launched_children=launched_children,
        policy_authority_produced=policy_authority_produced,
        preflight_authorized=preflight_authorized,
        cold_authorized=cold_authorized,
        staging_root=publication.staging_root,
        final_root=publication.final_root,
        nonce=publication.nonce,
        algorithm_route_selection=algorithm_route,
    )
    terminal_path = publication.staging_root / "supervisor-terminal.json"
    _publish_canonical_json(terminal_path, terminal_payload)
    artifact_refs["supervisor_terminal"] = _artifact_ref(
        terminal_path,
        publication.staging_root,
        DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    slots = derive_diag3_evidence_slots(
        artifact_root=publication.staging_root,
        artifact_refs=artifact_refs,
        failure=failure,
    )
    receipt = build_diag3_diagnostic_receipt(
        artifact_root=publication.staging_root,
        evidence_slots=slots,
    )
    _publish_bytes(
        publication.staging_root / DIAG2_RECEIPT_FILENAME,
        diag3_diagnostic_receipt_bytes(receipt),
    )
    _publish_canonical_json(
        publication.staging_root / DIAG2_MANIFEST_FILENAME,
        diag3_artifact_manifest_payload(publication.staging_root),
    )
    validate_diag3_writable_staging(publication.staging_root)
    _seal_and_sync_diag2_staging(publication.staging_root)
    load_and_validate_diag3_staging(publication.staging_root)
    _atomic_publish_diag2(publication)
    return load_and_validate_diag3_artifact(publication.final_root)


def _diag4_failure(
    stage: FailureStageV4,
    reason: FailureReasonCodeV4,
    detail: str,
) -> StructuredFailureV4:
    return StructuredFailureV4(stage, reason, _sha256(detail.encode("utf-8")))


def _diag5_failure(
    stage: FailureStageV5,
    reason: FailureReasonCodeV5,
    detail: str,
) -> StructuredFailureV5:
    """Hash one detail into the independent DIAG5 failure generation."""

    return StructuredFailureV5(stage, reason, _sha256(detail.encode("utf-8")))


def _diag5_consumption_failure(
    successor_claim: Diag5SuccessorAuthorityClaim,
    error: BaseException,
) -> StructuredFailureV5:
    """Classify consumption errors from the authority-owned lifecycle."""

    try:
        lifecycle = diag5_authority_lifecycle(successor_claim)
    except BaseException:  # noqa: BLE001 - an unreadable lifecycle is uncertain.
        lifecycle = Diag5AuthorityLifecycle.CONSUMPTION_UNCERTAIN
    reason = (
        FailureReasonCodeV5.AUTHORITY_CONSUMPTION_FAILED
        if lifecycle is Diag5AuthorityLifecycle.UNCONSUMED
        else FailureReasonCodeV5.AUTHORITY_CONSUMPTION_UNCERTAIN
    )
    return _diag5_failure(
        FailureStageV5.BEFORE_PREFLIGHT,
        reason,
        f"{type(error).__name__}:{error}",
    )


def _diag5_child_failure(
    mode: DiagnosticChildMode,
    outcome: DiagnosticSupervisedSampleV2,
) -> StructuredFailureV5 | None:
    """Map raw child supervision into the independent DIAG5 failure table."""

    reason = outcome.selected_failure_reason
    if reason is None:
        return None
    preflight = mode is DiagnosticChildMode.PREFLIGHT
    reason_map = {
        FailureReasonCodeV2.SOURCE_POST: (
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.CHILD_LAUNCH_FAILED: (
            FailureReasonCodeV5.PREFLIGHT_LAUNCH_FAILED
            if preflight
            else FailureReasonCodeV5.COLD_LAUNCH_FAILED
        ),
        FailureReasonCodeV2.CHILD_TIMEOUT: (
            FailureReasonCodeV5.PREFLIGHT_TIMEOUT
            if preflight
            else FailureReasonCodeV5.COLD_TIMEOUT
        ),
        FailureReasonCodeV2.MONITOR_BINDING_FAILED: (
            FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED
            if preflight
            else FailureReasonCodeV5.COLD_MONITOR_FAILED
        ),
        FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: (
            FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED
            if preflight
            else FailureReasonCodeV5.COLD_MONITOR_FAILED
        ),
        FailureReasonCodeV2.CHILD_EXIT_NONZERO: (
            FailureReasonCodeV5.PREFLIGHT_EXIT_NONZERO
            if preflight
            else FailureReasonCodeV5.COLD_EXIT_NONZERO
        ),
        FailureReasonCodeV2.PRODUCER_DECODE_FAILED: (
            FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PRODUCER_INVALID
        ),
        FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: (
            FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PRODUCER_INVALID
        ),
        FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID: (
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.POLICY_SCHEMA_INVALID: (
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED: (
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.CHILD_COMPILE_FAILED: (
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        ),
        FailureReasonCodeV2.CHILD_COMPILE_OOM: (
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
            if preflight
            else FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        ),
    }
    mapped = reason_map.get(reason)
    if mapped is None:
        raise ValueError("DIAG5 child failure reason is not a child-stage reason")
    return _diag5_failure(
        FailureStageV5.PREFLIGHT if preflight else FailureStageV5.COLD,
        mapped,
        "|".join(outcome.raw_failure_reasons) or reason.value,
    )


def _diag5_post_child_source_failure(
    mode: DiagnosticChildMode,
    selected_child_failure: StructuredFailureV5 | None,
) -> StructuredFailureV5:
    """Preserve a selected child failure across one post-child source drift."""

    if selected_child_failure is not None:
        return selected_child_failure
    if mode is DiagnosticChildMode.PREFLIGHT:
        return _diag5_failure(
            FailureStageV5.PREFLIGHT,
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
            "GPU_SNAPSHOT_REVALIDATION_FAILED_AFTER_PREFLIGHT",
        )
    return _diag5_failure(
        FailureStageV5.COLD,
        FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
        "GPU_SNAPSHOT_REVALIDATION_FAILED_AFTER_COLD",
    )


def _diag4_authority_consumption_failure_reason(
    claim: Diag4SuccessorAuthorityClaim,
) -> FailureReasonCodeV4:
    """Classify a raised consume call from its authority-owned lifecycle state."""

    lifecycle = diag4_authority_lifecycle(claim)
    if lifecycle is Diag4AuthorityLifecycle.STAGING_BOUND:
        return FailureReasonCodeV4.AUTHORITY_CONSUMPTION_FAILED
    if lifecycle in {
        Diag4AuthorityLifecycle.CONSUMED,
        Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
    }:
        return FailureReasonCodeV4.AUTHORITY_CONSUMPTION_UNCERTAIN
    raise RuntimeError("DIAG4 consume exception has an impossible authority lifecycle")


def _diag4_before_cold_authority_failure_reason(
    error: OSError | RuntimeError | TypeError | ValueError,
) -> FailureReasonCodeV4:
    """Preserve identity-before-marker precedence from typed revalidation errors."""

    return (
        FailureReasonCodeV4.CONSUMPTION_MARKER_INVALID
        if isinstance(error, Diag4ConsumptionMarkerInvalidError)
        else FailureReasonCodeV4.IDENTITY_REVALIDATION_FAILED
    )


def _diag4_receipt_failure(
    reason: FailureReasonCodeV4,
    error: Exception,
) -> StructuredFailureV4:
    """Construct one of the four exact run-level receipt failures."""

    if reason not in {
        FailureReasonCodeV4.EVIDENCE_VECTOR_INVALID,
        FailureReasonCodeV4.GROUP_PREFIX_INVALID,
        FailureReasonCodeV4.SCIENTIFIC_RECONSTRUCTION_FAILED,
        FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
    }:
        raise ValueError("DIAG4 receipt failure reason differs")
    return _diag4_failure(
        FailureStageV4.RECEIPT,
        reason,
        f"{type(error).__name__}:{error}",
    )


def _publish_diag4_terminal_and_receipt(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    outcome: StructuredFailureV4,
    launched_children: tuple[str, ...],
    successor_claim: Diag4SuccessorAuthorityClaim,
) -> DiagnosticReceiptV4 | Diag4VisiblePartial:
    """Publish, seal, atomically install, and independently reload one v4 tree."""

    try:
        terminal_payload = build_diag4_supervisor_terminal_payload(
            outcome=outcome,
            launched_children=launched_children,
            staging_root=publication.staging_root,
            final_root=publication.final_root,
            nonce=publication.nonce,
        )
    except (TypeError, ValueError) as error:
        raise Diag4HardPublicationError(
            FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
            publication.staging_root,
            error,
        ) from error
    terminal_path = publication.staging_root / "supervisor-terminal.json"
    try:
        _publish_canonical_json(terminal_path, terminal_payload)
    except OSError as error:
        raise Diag4HardPublicationError(
            FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
            publication.staging_root,
            error,
        ) from error
    artifact_refs["supervisor_terminal"] = _artifact_ref(
        terminal_path,
        publication.staging_root,
        DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    try:
        slots = derive_diag4_evidence_slots(
            artifact_root=publication.staging_root,
            artifact_refs=artifact_refs,
            outcome=outcome,
        )
    except (OSError, TypeError, ValueError) as error:
        return _diag4_receipt_partial(
            publication,
            artifact_refs,
            launched_children=launched_children,
            prior_outcome=outcome,
            reason=FailureReasonCodeV4.GROUP_PREFIX_INVALID,
            error=error,
        )
    try:
        receipt = build_diag4_diagnostic_receipt(
            artifact_root=publication.staging_root,
            evidence_slots=slots,
        )
        receipt_bytes = diag4_diagnostic_receipt_bytes(receipt)
    except (OSError, TypeError, ValueError) as error:
        return _diag4_receipt_partial(
            publication,
            artifact_refs,
            launched_children=launched_children,
            prior_outcome=outcome,
            reason=FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
            error=error,
        )
    try:
        _publish_bytes(
            publication.staging_root / DIAG2_RECEIPT_FILENAME,
            receipt_bytes,
        )
    except OSError as error:
        raise Diag4HardPublicationError(
            FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
            publication.staging_root,
            error,
        ) from error
    try:
        manifest_payload = diag4_artifact_manifest_payload(publication.staging_root)
    except (TypeError, ValueError) as error:
        return _diag4_publication_partial(
            publication,
            artifact_refs,
            launched_children=launched_children,
            prior_outcome=outcome,
            reason=FailureReasonCodeV4.MANIFEST_INVALID,
            error=error,
        )
    try:
        _publish_canonical_json(
            publication.staging_root / DIAG2_MANIFEST_FILENAME,
            manifest_payload,
        )
    except OSError as error:
        raise Diag4HardPublicationError(
            FailureReasonCodeV4.MANIFEST_INVALID,
            publication.staging_root,
            error,
        ) from error
    try:
        validate_diag4_writable_staging(publication.staging_root)
    except (TypeError, ValueError) as error:
        return _diag4_publication_partial(
            publication,
            artifact_refs,
            launched_children=launched_children,
            prior_outcome=outcome,
            reason=FailureReasonCodeV4.MODE_OR_LINK_INVALID,
            error=error,
        )
    try:
        _seal_and_sync_diag2_staging(publication.staging_root)
    except (OSError, TypeError, ValueError) as error:
        raise Diag4HardPublicationError(
            FailureReasonCodeV4.MODE_OR_LINK_INVALID,
            publication.staging_root,
            error,
        ) from error
    try:
        load_and_validate_diag4_staging(publication.staging_root)
    except (OSError, TypeError, ValueError) as error:
        return _diag4_publication_partial(
            publication,
            artifact_refs,
            launched_children=launched_children,
            prior_outcome=outcome,
            reason=FailureReasonCodeV4.STAGING_DEEP_LOAD_FAILED,
            error=error,
        )
    try:
        _rename_noreplace(publication.staging_root, publication.final_root)
    except FileExistsError as error:
        return _diag4_publication_partial(
            publication,
            artifact_refs,
            launched_children=launched_children,
            prior_outcome=outcome,
            reason=FailureReasonCodeV4.FINAL_COLLISION,
            error=error,
        )
    except OSError as error:
        return _diag4_publication_partial(
            publication,
            artifact_refs,
            launched_children=launched_children,
            prior_outcome=outcome,
            reason=FailureReasonCodeV4.FINAL_RENAME_FAILED,
            error=error,
        )
    try:
        _fsync_parent(publication.final_root)
    except OSError as error:
        _raise_diag4_final_publication_failure(
            publication,
            successor_claim,
            reason=Diag4PhysicalPublicationReason.FINAL_FSYNC_FAILED,
            cause=error,
        )
    try:
        receipt = load_and_validate_diag4_artifact(publication.final_root)
    except (OSError, TypeError, ValueError) as error:
        _raise_diag4_final_publication_failure(
            publication,
            successor_claim,
            reason=Diag4PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED,
            cause=error,
        )
    authority_reason = (
        Diag4PhysicalPublicationReason.POST_FINAL_AUTHORITY_FINALIZATION_FAILED
    )
    lifecycle: Diag4AuthorityLifecycle | None = None
    try:
        lifecycle = diag4_authority_lifecycle(successor_claim)
        if lifecycle in {
            Diag4AuthorityLifecycle.CONSUMED,
            Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
        }:
            authority_reason = (
                Diag4PhysicalPublicationReason.POST_FINAL_AUTHORITY_REVALIDATION_FAILED
            )
            revalidate_diag4_successor_authority(
                successor_claim, require_output_absent=False
            )
        elif lifecycle is Diag4AuthorityLifecycle.STAGING_BOUND:
            finalize_diag4_prelaunch_failure(successor_claim)
        else:
            raise RuntimeError("DIAG4 authority lifecycle cannot finalize an artifact")
    except BaseException as error:  # noqa: BLE001 - final path must be observed.
        _raise_diag4_final_publication_failure(
            publication,
            successor_claim,
            reason=authority_reason,
            cause=error,
            known_lifecycle=lifecycle,
        )
    return receipt


def _diag5_native_bindings_payload(
    successor_claim: Diag5SuccessorAuthorityClaim,
) -> dict[str, JsonValue]:
    return {
        "cpu": _diag5_native_binding_payload(successor_claim, role="cpu"),
        "gpu": _diag5_native_binding_payload(successor_claim, role="gpu"),
    }


def _raise_diag5_post_final_failure(
    *,
    successor_claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
    reason: Diag5PhysicalPublicationReason,
    sealed_artifact_manifest_sha256: str,
    cause: BaseException,
    expected_source_snapshot: SnapshotIdentity,
    physical_memory_bytes: int,
) -> Never:
    native_bindings = _diag5_native_bindings_payload(successor_claim)

    def deep_load(rollback_root: Path) -> object:
        return load_and_validate_diag5_rollback(
            rollback_root,
            expected_rollback_root=successor_claim.expected_gpu_rollback_root,
            expected_final_root=successor_claim.expected_gpu_output_root,
            expected_native_bindings=native_bindings,
            expected_authority_sha256=successor_claim.authority_sha256,
            expected_predecessor_postmortem=(successor_claim.predecessor_postmortem),
            expected_source_snapshot_identity=expected_source_snapshot,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
            expected_gpu_uuid=successor_claim.expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
        )

    authority_observation = rollback_diag5_bound_final(
        successor_claim,
        reservation,
        deep_load=deep_load,
    )
    observation = _diag5_physical_observation(authority_observation)
    payload = _diag5_physical_publication_failure_payload(
        successor_claim=successor_claim,
        original_reason=reason,
        observation=observation,
        sealed_artifact_manifest_sha256=sealed_artifact_manifest_sha256,
    )
    try:
        evidence_path = publish_diag5_physical_failure_evidence(
            successor_claim,
            reservation,
            payload,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as evidence_error:
        raise Diag5PhysicalPublicationError(
            reason,
            observation,
            None,
            cause,
        ) from evidence_error
    raise Diag5PhysicalPublicationError(
        reason,
        observation,
        evidence_path,
        cause,
    ) from cause


def _diag5_finalizer_publication_reason(
    category: Diag5FinalizerFailureCategory,
) -> Diag5PhysicalPublicationReason:
    """Map one authority-owned finalizer phase to its physical reason."""

    return {
        Diag5FinalizerFailureCategory.DEEP_LOAD: (
            Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED
        ),
        Diag5FinalizerFailureCategory.REVALIDATION: (
            Diag5PhysicalPublicationReason.POST_FINAL_AUTHORITY_REVALIDATION_FAILED
        ),
        Diag5FinalizerFailureCategory.FINALIZATION: (
            Diag5PhysicalPublicationReason.POST_FINAL_AUTHORITY_FINALIZATION_FAILED
        ),
    }[category]


def _raise_diag5_pre_final_after_reservation(
    *,
    publication: Diag2Publication,
    terminal_outcome: StructuredFailureV5,
    successor_claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
    reason: FailureReasonCodeV5,
    cause: BaseException,
) -> Never:
    try:
        cancellation = cancel_diag5_physical_failure_evidence(
            successor_claim, reservation
        )
    except Diag5PhysicalCancellationError as cleanup_error:
        raise Diag5PreFinalPublicationError(
            terminal_outcome=terminal_outcome,
            publication_failure=_diag5_failure(
                FailureStageV5.PUBLICATION,
                reason,
                f"{type(cause).__name__}:{cause}",
            ),
            staging_root=publication.staging_root,
            cause=cause,
            cancellation_observation=cleanup_error.observation,
            cleanup_cause=cleanup_error,
        ) from cleanup_error
    _raise_diag5_pre_final_failure(
        publication=publication,
        terminal_outcome=terminal_outcome,
        stage=FailureStageV5.PUBLICATION,
        reason=reason,
        cause=cause,
        cancellation_observation=cancellation,
    )


def _publish_diag5_terminal_and_receipt(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    outcome: StructuredFailureV5,
    launched_children: tuple[str, ...],
    successor_claim: Diag5SuccessorAuthorityClaim,
    source: SnapshotPublication | None,
    expected_source_snapshot: SnapshotIdentity,
    physical_memory_bytes: int,
) -> object:
    """Seal and atomically install one independently parsed DIAG5 artifact."""

    gpu_binding = _diag5_native_binding_payload(successor_claim, role="gpu")
    native_bindings = _diag5_native_bindings_payload(successor_claim)
    try:
        terminal_path = publication.staging_root / "supervisor-terminal.json"
        terminal_payload = build_diag5_supervisor_terminal_payload(
            outcome=outcome,
            launched_children=launched_children,
            staging_root=publication.staging_root,
            final_root=publication.final_root,
        )
        _publish_canonical_json(terminal_path, terminal_payload)
        artifact_refs["supervisor_terminal"] = _artifact_ref(
            terminal_path,
            publication.staging_root,
            DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
        )
        slots = derive_diag5_evidence_slots(
            artifact_root=publication.staging_root,
            artifact_refs=artifact_refs,
            outcome=outcome,
            gpu_native_binding=gpu_binding,
            authority_sha256=successor_claim.authority_sha256,
            expected_source_snapshot_identity=expected_source_snapshot,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
            expected_gpu_uuid=successor_claim.expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
        )
        receipt = build_diag5_diagnostic_receipt(
            artifact_root=publication.staging_root,
            evidence_slots=slots,
            native_bindings=native_bindings,
            predecessor_postmortem=successor_claim.predecessor_postmortem,
            expected_native_bindings=native_bindings,
            expected_authority_sha256=successor_claim.authority_sha256,
            expected_predecessor_postmortem=(successor_claim.predecessor_postmortem),
            expected_source_snapshot_identity=expected_source_snapshot,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
            expected_gpu_uuid=successor_claim.expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
        )
        receipt_path = publication.staging_root / DIAG2_RECEIPT_FILENAME
        _publish_bytes(receipt_path, diag5_diagnostic_receipt_bytes(receipt))
        diagnostic_receipt_reference = _artifact_ref(
            receipt_path,
            publication.staging_root,
            DIAG5_SCHEMA_VERSION,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_pre_final_failure(
            publication=publication,
            terminal_outcome=outcome,
            stage=FailureStageV5.RECEIPT,
            reason=classify_diag5_receipt_construction_error(error),
            cause=error,
        )
    try:
        _publish_canonical_json(
            publication.staging_root / DIAG2_MANIFEST_FILENAME,
            diag5_artifact_manifest_payload(publication.staging_root),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_pre_final_failure(
            publication=publication,
            terminal_outcome=outcome,
            stage=FailureStageV5.PUBLICATION,
            reason=FailureReasonCodeV5.MANIFEST_INVALID,
            cause=error,
        )
    manifest_sha256 = _sha256(
        (publication.staging_root / DIAG2_MANIFEST_FILENAME).read_bytes()
    )
    try:
        validate_diag5_writable_staging(
            publication.staging_root,
            expected_native_bindings=native_bindings,
            expected_authority_sha256=successor_claim.authority_sha256,
            expected_predecessor_postmortem=(successor_claim.predecessor_postmortem),
            expected_source_snapshot_identity=expected_source_snapshot,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
            expected_gpu_uuid=successor_claim.expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_pre_final_failure(
            publication=publication,
            terminal_outcome=outcome,
            stage=FailureStageV5.PUBLICATION,
            reason=FailureReasonCodeV5.MODE_OR_LINK_INVALID,
            cause=error,
        )
    try:
        revalidate_diag5_successor_authority(successor_claim)
        _seal_and_sync_diag2_staging(publication.staging_root)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_pre_final_failure(
            publication=publication,
            terminal_outcome=outcome,
            stage=FailureStageV5.PUBLICATION,
            reason=FailureReasonCodeV5.MODE_OR_LINK_INVALID,
            cause=error,
        )
    try:
        load_and_validate_diag5_staging(
            publication.staging_root,
            expected_native_bindings=native_bindings,
            expected_authority_sha256=successor_claim.authority_sha256,
            expected_predecessor_postmortem=(successor_claim.predecessor_postmortem),
            expected_source_snapshot_identity=expected_source_snapshot,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
            expected_gpu_uuid=successor_claim.expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
        )
        if source is not None:
            validate_diag5_successor_snapshot(source, successor_claim)
        revalidate_diag5_successor_authority(successor_claim)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_pre_final_failure(
            publication=publication,
            terminal_outcome=outcome,
            stage=FailureStageV5.PUBLICATION,
            reason=FailureReasonCodeV5.STAGING_DEEP_LOAD_FAILED,
            cause=error,
        )
    try:
        reservation = prepare_diag5_physical_failure_evidence(successor_claim)
    except FileExistsError as error:
        _raise_diag5_pre_final_failure(
            publication=publication,
            terminal_outcome=outcome,
            stage=FailureStageV5.PUBLICATION,
            reason=FailureReasonCodeV5.FINAL_COLLISION,
            cause=error,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_pre_final_failure(
            publication=publication,
            terminal_outcome=outcome,
            stage=FailureStageV5.PUBLICATION,
            reason=FailureReasonCodeV5.FINAL_RENAME_FAILED,
            cause=error,
        )
    try:
        publish_diag5_bound_staging(
            successor_claim,
            Diag5PublishedOutputKind.FINAL,
        )
    except FileExistsError as error:
        _raise_diag5_pre_final_after_reservation(
            publication=publication,
            terminal_outcome=outcome,
            successor_claim=successor_claim,
            reservation=reservation,
            reason=FailureReasonCodeV5.FINAL_COLLISION,
            cause=error,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_pre_final_after_reservation(
            publication=publication,
            terminal_outcome=outcome,
            successor_claim=successor_claim,
            reservation=reservation,
            reason=FailureReasonCodeV5.FINAL_RENAME_FAILED,
            cause=error,
        )
    try:
        fsync_diag5_output_parent(successor_claim)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_post_final_failure(
            successor_claim=successor_claim,
            reservation=reservation,
            reason=Diag5PhysicalPublicationReason.FINAL_FSYNC_FAILED,
            sealed_artifact_manifest_sha256=manifest_sha256,
            cause=error,
            expected_source_snapshot=expected_source_snapshot,
            physical_memory_bytes=physical_memory_bytes,
        )
    try:
        final_receipt = load_and_validate_diag5_artifact(
            publication.final_root,
            expected_native_bindings=native_bindings,
            expected_authority_sha256=successor_claim.authority_sha256,
            expected_predecessor_postmortem=(successor_claim.predecessor_postmortem),
            expected_source_snapshot_identity=expected_source_snapshot,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
            expected_gpu_uuid=successor_claim.expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_post_final_failure(
            successor_claim=successor_claim,
            reservation=reservation,
            reason=Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED,
            sealed_artifact_manifest_sha256=manifest_sha256,
            cause=error,
            expected_source_snapshot=expected_source_snapshot,
            physical_memory_bytes=physical_memory_bytes,
        )
    try:
        if source is not None:
            final_source = replace(
                source,
                root=publication.final_root / source.root.name,
                manifest_path=(
                    publication.final_root
                    / source.root.name
                    / source.manifest_path.name
                ),
            )
            validate_diag5_successor_snapshot(final_source, successor_claim)
        revalidate_diag5_published_output(
            successor_claim,
            Diag5PublishedOutputKind.FINAL,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_post_final_failure(
            successor_claim=successor_claim,
            reservation=reservation,
            reason=(
                Diag5PhysicalPublicationReason.POST_FINAL_AUTHORITY_REVALIDATION_FAILED
            ),
            sealed_artifact_manifest_sha256=manifest_sha256,
            cause=error,
            expected_source_snapshot=expected_source_snapshot,
            physical_memory_bytes=physical_memory_bytes,
        )
    try:
        if source is not None:
            finalizer_source = PublishedSnapshot(
                Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
                final_source,
            )
        else:
            supervisor_terminal_reference = artifact_refs["supervisor_terminal"]
            if supervisor_terminal_reference is None:
                raise ValueError("DIAG5 pre-source terminal evidence is absent")
            finalizer_source = PreSourceFailure(
                Diag5FinalizerSourceKind.PRE_SOURCE_FAILURE,
                outcome,
                supervisor_terminal_reference,
                diagnostic_receipt_reference,
            )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_post_final_failure(
            successor_claim=successor_claim,
            reservation=reservation,
            reason=Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED,
            sealed_artifact_manifest_sha256=manifest_sha256,
            cause=error,
            expected_source_snapshot=expected_source_snapshot,
            physical_memory_bytes=physical_memory_bytes,
        )
    try:
        finalize_diag5_physical_evidence_success(
            successor_claim,
            reservation,
            finalizer_source,
            physical_memory_bytes=physical_memory_bytes,
        )
    except Diag5FinalizerError as error:
        _raise_diag5_post_final_failure(
            successor_claim=successor_claim,
            reservation=reservation,
            reason=_diag5_finalizer_publication_reason(error.category),
            sealed_artifact_manifest_sha256=manifest_sha256,
            cause=error.cause,
            expected_source_snapshot=expected_source_snapshot,
            physical_memory_bytes=physical_memory_bytes,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_diag5_post_final_failure(
            successor_claim=successor_claim,
            reservation=reservation,
            reason=(
                Diag5PhysicalPublicationReason.POST_FINAL_AUTHORITY_FINALIZATION_FAILED
            ),
            sealed_artifact_manifest_sha256=manifest_sha256,
            cause=error,
            expected_source_snapshot=expected_source_snapshot,
            physical_memory_bytes=physical_memory_bytes,
        )
    return final_receipt


def _diag4_publication_partial(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    launched_children: tuple[str, ...],
    prior_outcome: StructuredFailureV4,
    reason: FailureReasonCodeV4,
    error: Exception,
) -> Diag4VisiblePartial:
    """Replace the partial terminal with one exact publication failure."""

    publication_outcome = _diag4_failure(
        FailureStageV4.PUBLICATION,
        reason,
        f"{type(error).__name__}:{error}",
    )
    outcome = (
        prior_outcome
        if prior_outcome.stage
        in {
            FailureStageV4.AUTHORITY,
            FailureStageV4.SETUP,
            FailureStageV4.BEFORE_PREFLIGHT,
            FailureStageV4.PREFLIGHT,
            FailureStageV4.BEFORE_COLD,
            FailureStageV4.COLD,
            FailureStageV4.NUMERICAL_COMMIT,
            FailureStageV4.RECEIPT,
        }
        else publication_outcome
    )
    return _replace_diag4_partial_terminal(
        publication,
        artifact_refs,
        outcome=outcome,
        launched_children=launched_children,
    )


def _diag4_receipt_partial(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    launched_children: tuple[str, ...],
    prior_outcome: StructuredFailureV4,
    reason: FailureReasonCodeV4,
    error: Exception,
) -> Diag4VisiblePartial:
    """Replace the partial terminal with one exact receipt-stage failure."""

    receipt_outcome = _diag4_receipt_failure(reason, error)
    outcome = (
        prior_outcome
        if prior_outcome.stage
        in {
            FailureStageV4.AUTHORITY,
            FailureStageV4.SETUP,
            FailureStageV4.BEFORE_PREFLIGHT,
            FailureStageV4.PREFLIGHT,
            FailureStageV4.BEFORE_COLD,
            FailureStageV4.COLD,
            FailureStageV4.NUMERICAL_COMMIT,
        }
        else receipt_outcome
    )
    return _replace_diag4_partial_terminal(
        publication,
        artifact_refs,
        outcome=outcome,
        launched_children=launched_children,
    )


def _replace_diag4_partial_terminal(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    outcome: StructuredFailureV4,
    launched_children: tuple[str, ...],
) -> Diag4VisiblePartial:
    """Atomically replace only the typed terminal within a visible partial."""

    root = publication.staging_root
    try:
        original_mode = root.stat().st_mode & 0o777
        root.chmod(0o755)
        try:
            terminal_path = root / "supervisor-terminal.json"
            replacement = (
                root / f".supervisor-terminal.failure-{publication.nonce}.json"
            )
            _publish_canonical_json(
                replacement,
                build_diag4_supervisor_terminal_payload(
                    outcome=outcome,
                    launched_children=launched_children,
                    staging_root=root,
                    final_root=publication.final_root,
                    nonce=publication.nonce,
                ),
            )
            os.replace(replacement, terminal_path)
            _fsync_parent(terminal_path)
            artifact_refs["supervisor_terminal"] = _artifact_ref(
                terminal_path,
                root,
                DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
            )
        finally:
            root.chmod(original_mode)
    except (OSError, TypeError, ValueError) as error:
        raise Diag4HardPublicationError(outcome.reason, root, error) from error
    return Diag4VisiblePartial(outcome, root)


def _diag4_supervisor_zero_gate(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    environment: Mapping[str, str],
    query_executable_sha256: str,
    stage: FailureStageV4,
) -> StructuredFailureV4 | None:
    """Capture and retain one route-owned supervisor GPU-zero boundary."""

    stage_name = (
        "BEFORE_PREFLIGHT"
        if stage is FailureStageV4.BEFORE_PREFLIGHT
        else "BEFORE_COLD"
    )
    slot_name = (
        "supervisor_before_preflight"
        if stage is FailureStageV4.BEFORE_PREFLIGHT
        else "supervisor_before_cold"
    )
    try:
        observation = _capture_diag2_supervisor_zero(
            environment,
            query_executable_sha256=query_executable_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _diag4_failure(
            stage,
            FailureReasonCodeV4.SUPERVISOR_GPU_OBSERVATION_INVALID,
            f"{type(error).__name__}:{error}",
        )
    try:
        reference = _publish_diag2_supervisor_zero(
            publication.staging_root,
            observation,
            stage=stage_name,
        )
    except OSError as error:
        raise Diag4HardPublicationError(
            FailureReasonCodeV4.SUPERVISOR_GPU_OBSERVATION_INVALID,
            publication.staging_root,
            error,
        ) from error
    artifact_refs[slot_name] = reference
    try:
        validate_diag2_supervisor_zero_payload(
            load_canonical_json_bytes(
                reference.resolve_and_validate(publication.staging_root).read_bytes()
            ),
            artifact_root=publication.staging_root,
            expected_stage=stage_name,
            allow_failure=True,
        )
    except (OSError, TypeError, ValueError) as error:
        return _diag4_failure(
            stage,
            FailureReasonCodeV4.SUPERVISOR_GPU_OBSERVATION_INVALID,
            f"{type(error).__name__}:{error}",
        )
    if observation.gate_passes:
        return None
    return _diag4_failure(
        stage,
        (
            FailureReasonCodeV4.SUPERVISOR_GPU_NONZERO
            if observation.matching_rows
            else FailureReasonCodeV4.SUPERVISOR_GPU_OBSERVATION_INVALID
        ),
        "supervisor GPU-zero gate did not pass",
    )


def _diag5_supervisor_zero_gate(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    environment: Mapping[str, str],
    query_executable_sha256: str,
    stage: FailureStageV5,
) -> StructuredFailureV5 | None:
    """Capture one DIAG5-owned GPU-zero boundary under its v1 schema."""

    stage_name = (
        "BEFORE_PREFLIGHT"
        if stage is FailureStageV5.BEFORE_PREFLIGHT
        else "BEFORE_COLD"
    )
    slot_name = (
        "supervisor_before_preflight"
        if stage is FailureStageV5.BEFORE_PREFLIGHT
        else "supervisor_before_cold"
    )
    try:
        observation = _capture_diag2_supervisor_zero(
            environment,
            query_executable_sha256=query_executable_sha256,
        )
        reference = _publish_diag2_supervisor_zero(
            publication.staging_root,
            observation,
            stage=stage_name,
            builder=build_diag5_supervisor_zero_payload,
            schema_version=DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION,
        )
        artifact_refs[slot_name] = reference
        validate_diag5_supervisor_zero_payload(
            load_canonical_json_bytes(
                reference.resolve_and_validate(publication.staging_root).read_bytes()
            ),
            artifact_root=publication.staging_root,
            expected_stage=stage_name,
            allow_failure=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _diag5_failure(
            stage,
            FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID,
            f"{type(error).__name__}:{error}",
        )
    if observation.gate_passes:
        return None
    return _diag5_failure(
        stage,
        (
            FailureReasonCodeV5.SUPERVISOR_GPU_NONZERO
            if observation.matching_rows
            else FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID
        ),
        "supervisor GPU-zero gate did not pass",
    )


def _publish_diag4_child_supervision(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    directory: Path,
    outcome: DiagnosticSupervisedSampleV2,
    *,
    mode: DiagnosticChildMode,
) -> StructuredFailureV4 | None:
    """Publish one reaped child's evidence or return its protocol failure."""

    prefix = mode.value
    stage = (
        FailureStageV4.PREFLIGHT
        if mode is DiagnosticChildMode.PREFLIGHT
        else FailureStageV4.COLD
    )
    reason = (
        FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID
        if mode is DiagnosticChildMode.PREFLIGHT
        else FailureReasonCodeV4.COLD_PROTOCOL_INVALID
    )
    producer_schema = (
        DIAG4_PREFLIGHT_SCHEMA_VERSION
        if mode is DiagnosticChildMode.PREFLIGHT
        else DIAG4_COLD_RESULT_SCHEMA_VERSION
    )
    try:
        child_refs = _publish_diag2_supervision(
            publication.staging_root,
            directory,
            outcome,
            producer_schema=producer_schema,
        )
        for suffix in ("producer", "terminal", "process", "memory", "memory_samples"):
            artifact_refs[f"{prefix}_{suffix}"] = child_refs[suffix]
        artifact_refs[f"{prefix}_runtime"] = _diag2_existing_reference(
            publication.staging_root,
            f"{prefix}/runtime-evidence.json",
            RUNTIME_EVIDENCE_SCHEMA_VERSION,
        )
        artifact_refs[f"{prefix}_policy"] = _diag2_existing_reference(
            publication.staging_root,
            f"{prefix}/policy.json",
            f"{DIAGNOSTIC_SCHEMA_VERSION}-policy",
        )
    except OSError as error:
        raise Diag4HardPublicationError(
            reason,
            publication.staging_root,
            error,
        ) from error
    except (KeyError, TypeError, ValueError) as error:
        return _diag4_failure(
            stage,
            reason,
            f"{type(error).__name__}:{error}",
        )
    return None


def _publish_diag5_child_supervision(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    directory: Path,
    outcome: DiagnosticSupervisedSampleV2,
    *,
    mode: DiagnosticChildMode,
    defer_success_auxiliary_slots: bool = False,
) -> StructuredFailureV5 | None:
    """Publish one DIAG5 child using only v2/v5 wire identities."""

    prefix = mode.value
    stage = (
        FailureStageV5.PREFLIGHT
        if mode is DiagnosticChildMode.PREFLIGHT
        else FailureStageV5.COLD
    )
    reason = (
        FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
        if mode is DiagnosticChildMode.PREFLIGHT
        else FailureReasonCodeV5.COLD_PROTOCOL_INVALID
    )
    producer_schema = (
        DIAG5_PREFLIGHT_SCHEMA_VERSION
        if mode is DiagnosticChildMode.PREFLIGHT
        else DIAG5_COLD_RESULT_SCHEMA_VERSION
    )
    try:
        if (
            outcome.producer is None
            and outcome.selected_failure_reason
            in {
                FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
                FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
            }
            and outcome.stdout
        ):
            _retain_invalid_diag5_producer_bytes(directory, outcome.stdout)
        child_refs = _publish_diag2_supervision(
            publication.staging_root,
            directory,
            outcome,
            producer_schema=producer_schema,
            process_schema=DIAG5_PROCESS_SCHEMA_VERSION,
            terminal_schema=DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
            memory_schema=DIAG5_MEMORY_SCHEMA_VERSION,
            memory_samples_schema=DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
        )
        selected_failure = _diag5_child_failure(mode, outcome)
        producer_reference = child_refs["producer"]
        if producer_reference is None:
            if selected_failure is None:
                raise ValueError("DIAG5 absent producer has no supervision reason")
            terminal_reference = child_refs["terminal"]
            process_reference = child_refs["process"]
            if terminal_reference is None or process_reference is None:
                raise ValueError("DIAG5 supervision closure omits terminal or process")
            producer_payload = build_diag5_supervisor_failure_producer_payload(
                mode=prefix,
                selected_failure_reason=selected_failure.reason,
                child_pid=outcome.child_pid,
                child_start_time_ticks=outcome.child_start_time_ticks,
                process_started_monotonic_ns=outcome.process_started_monotonic_ns,
                process_stopped_monotonic_ns=outcome.process_stopped_monotonic_ns,
                process_evidence=process_reference,
                child_terminal_evidence=terminal_reference,
            )
            validate_diag5_supervisor_failure_producer_payload(
                producer_payload,
                mode=prefix,
            )
            producer_path = directory / "producer.json"
            _publish_canonical_json(producer_path, producer_payload)
            producer_reference = _artifact_ref(
                producer_path,
                publication.staging_root,
                producer_schema,
            )
        prospective = {
            f"{prefix}_producer": producer_reference,
            f"{prefix}_terminal": child_refs["terminal"],
            f"{prefix}_process": child_refs["process"],
        }
        if selected_failure is None:
            runtime_reference = _diag2_existing_reference(
                publication.staging_root,
                f"{prefix}/runtime-evidence.json",
                RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
            )
            policy_reference = _diag2_existing_reference(
                publication.staging_root,
                f"{prefix}/policy.json",
                "single-stage-native-equivalent-quality-policy-v1",
            )
            if runtime_reference is None or policy_reference is None:
                raise ValueError("DIAG5 child runtime or policy evidence is absent")
            validate_diag5_policy_evidence_payload(
                load_canonical_json_bytes(
                    policy_reference.resolve_and_validate(
                        publication.staging_root
                    ).read_bytes()
                )
            )
            auxiliary = {
                f"{prefix}_memory": child_refs["memory"],
                f"{prefix}_memory_samples": child_refs["memory_samples"],
                f"{prefix}_runtime": runtime_reference,
                f"{prefix}_policy": policy_reference,
            }
            if not defer_success_auxiliary_slots:
                prospective.update(auxiliary)
            full_expected_present = {
                f"{prefix}_producer",
                f"{prefix}_terminal",
                f"{prefix}_process",
                f"{prefix}_memory",
                f"{prefix}_memory_samples",
                f"{prefix}_runtime",
                f"{prefix}_policy",
            }
            if {
                name
                for name, reference in {**prospective, **auxiliary}.items()
                if reference is not None
            } != full_expected_present:
                raise ValueError("DIAG5 complete child evidence prefix differs")
            expected_present = (
                {
                    f"{prefix}_producer",
                    f"{prefix}_terminal",
                    f"{prefix}_process",
                }
                if defer_success_auxiliary_slots
                else full_expected_present
            )
        else:
            expected_present = {
                f"{prefix}_producer",
                f"{prefix}_terminal",
                f"{prefix}_process",
            }
        if {
            name for name, reference in prospective.items() if reference is not None
        } != (expected_present):
            raise ValueError("DIAG5 child supervision evidence prefix differs")
    except (OSError, KeyError, TypeError, ValueError) as error:
        return _diag5_failure(stage, reason, f"{type(error).__name__}:{error}")
    artifact_refs.update(prospective)
    return None


def _diag5_child_success_auxiliary_references(
    publication: Diag2Publication,
    *,
    mode: DiagnosticChildMode,
) -> dict[str, ArtifactRef]:
    """Reconstruct the four validated success auxiliaries after a source gate."""

    prefix = mode.value
    specifications = (
        ("memory", "gpu-memory.json", DIAG5_MEMORY_SCHEMA_VERSION),
        (
            "memory_samples",
            "gpu-memory-samples.json",
            DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
        ),
        ("runtime", "runtime-evidence.json", RUNTIME_EVIDENCE_V2_SCHEMA_VERSION),
        (
            "policy",
            "policy.json",
            "single-stage-native-equivalent-quality-policy-v1",
        ),
    )
    references: dict[str, ArtifactRef] = {}
    for suffix, filename, schema in specifications:
        reference = _diag2_existing_reference(
            publication.staging_root,
            f"{prefix}/{filename}",
            schema,
        )
        if reference is None:
            raise ValueError(f"DIAG5 child {suffix} evidence is absent")
        references[f"{prefix}_{suffix}"] = reference
    validate_diag5_policy_evidence_payload(
        load_canonical_json_bytes(
            references[f"{prefix}_policy"]
            .resolve_and_validate(publication.staging_root)
            .read_bytes()
        )
    )
    return references


def _leave_diag4_visible_partial(
    publication: Diag2Publication,
    artifact_refs: dict[str, ArtifactRef | None],
    *,
    outcome: StructuredFailureV4,
    launched_children: tuple[str, ...],
) -> dict[str, JsonValue]:
    """Retain an unsealed v4 partial for a physical lifecycle failure."""

    terminal_path = publication.staging_root / "supervisor-terminal.json"
    try:
        _publish_canonical_json(
            terminal_path,
            build_diag4_supervisor_terminal_payload(
                outcome=outcome,
                launched_children=launched_children,
                staging_root=publication.staging_root,
                final_root=publication.final_root,
                nonce=publication.nonce,
            ),
        )
    except OSError as error:
        raise Diag4HardPublicationError(
            outcome.reason,
            publication.staging_root,
            error,
        ) from error
    artifact_refs["supervisor_terminal"] = _artifact_ref(
        terminal_path,
        publication.staging_root,
        DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    try:
        _fsync_parent(terminal_path)
    except OSError as error:
        raise Diag4HardPublicationError(
            outcome.reason,
            publication.staging_root,
            error,
        ) from error
    return {
        "schema_version": DIAG4_SUMMARY_SCHEMA_VERSION,
        "route": DIAG4_ROUTE,
        "children": list(launched_children),
        "verdict": "VISIBLE_PARTIAL",
        "next_route": "NOT_PRODUCED",
        "speed_comparison": "NOT_PRODUCED",
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
        "partial_root": str(publication.staging_root),
    }


def _diag2_existing_reference(
    staging_root: Path,
    relative_path: str,
    schema_version: str,
) -> ArtifactRef | None:
    path = staging_root / relative_path
    return (
        _artifact_ref(path, staging_root, schema_version)
        if path.is_file() and not path.is_symlink()
        else None
    )


def _publish_diag2_execution(
    staging_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    *,
    cold: DiagnosticSupervisedSampleV2,
    cold_invocation: SnapshotChildInvocation,
    interpreter: Path,
    physical_memory_bytes: int,
) -> ArtifactRef:
    if cold.producer is None or cold.memory is None:
        raise ValueError("complete DIAG2 execution requires producer and memory")
    producer = cold.producer
    old_names = {
        "preflight": "preflight_producer",
        "preflight_child_terminal": "preflight_terminal",
        "preflight_process": "preflight_process",
        "preflight_memory": "preflight_memory",
        "preflight_memory_samples": "preflight_memory_samples",
        "preflight_runtime": "preflight_runtime",
        "preflight_policy": "preflight_policy",
        "policy_authority": "policy_authority",
        "producer": "cold_producer",
        "child_terminal": "cold_terminal",
        "runtime": "cold_runtime",
        "process": "cold_process",
        "memory": "cold_memory",
        "memory_samples": "cold_memory_samples",
        "source_manifest": "source_manifest",
        "native_reference": "native_reference",
        "policy": "cold_policy",
    }
    support: dict[str, ArtifactRef] = {}
    for old_name, new_name in old_names.items():
        reference = artifact_refs[new_name]
        if reference is None:
            raise ValueError(f"complete DIAG2 execution omits {new_name}")
        support[old_name] = reference
    runtime_reference = support["runtime"]
    runtime_document = load_canonical_json_bytes(
        runtime_reference.resolve_and_validate(staging_root).read_bytes()
    )
    runtime_document_map = _mapping_field(runtime_document, "runtime_identity")
    runtime_payload = _mapping_field(producer, "runtime")
    transfers = _mapping_field(producer, "transfer_audit")
    pre_source = cold.pre_source_identity
    post_source = cold.post_source_identity
    if pre_source is None or post_source is None:
        raise ValueError("complete DIAG2 source identities are absent")
    payload = execution_evidence_payload(
        supporting_evidence=support,
        preflight={
            "status": "COMPLETE",
            "compile_success": True,
            "solver_dispatched": False,
            "finalizer_called": False,
            "endpoint_audit_called": False,
            "campaign_authorized": False,
            "callbacks": 0,
        },
        cold={
            "status": "COMPLETE",
            "child_pid": cold.child_pid,
            "child_start_time_ticks": cold.child_start_time_ticks,
            "backend": str(runtime_payload["backend"]),
            "gpu_uuid": GPU_UUID,
            "jax_enable_x64": runtime_payload["jax_enable_x64"] is True,
            "state_size": 716,
            "equality_size": 255,
            "residual_size": 2110,
            "policy_sha256": str(producer["policy_sha256"]),
            "phase_schema_sha256": str(producer["phase_schema_sha256"]),
            "source_pre_sha256": _sha256(canonical_json_bytes(pre_source.to_payload())),
            "source_post_sha256": _sha256(
                canonical_json_bytes(post_source.to_payload())
            ),
            "runtime_environment_sha256": str(
                runtime_document_map["effective_environment_sha256"]
            ),
            "interpreter": str(interpreter),
            "argv": list(cold_invocation.argv),
            "physical_memory_bytes": physical_memory_bytes,
            "peak_memory_bytes": int(cold.memory["peak_memory_bytes"]),
            "peak_memory_fraction": float(cold.memory["peak_memory_fraction"]),
            "hot_h2d_transfers": int(transfers["hot_h2d_transfers"]),
            "hot_d2h_transfers": int(transfers["hot_d2h_transfers"]),
            "python_callbacks": int(transfers["python_callbacks"]),
            "final_d2h_transfers": int(transfers["final_d2h_transfers"]),
            "timestamps_ns": _mapping_field(producer, "timestamps_ns"),
            "stdout_sha256": _sha256(cold.stdout),
            "stdout_size_bytes": len(cold.stdout),
            "stderr_sha256": _sha256(cold.stderr),
            "stderr_size_bytes": len(cold.stderr),
        },
    )
    path = staging_root / "execution.json"
    _publish_canonical_json(path, payload)
    return _artifact_ref(
        path,
        staging_root,
        f"{DIAGNOSTIC_SCHEMA_VERSION}-execution",
    )


def _publish_diag4_execution(
    staging_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
) -> ArtifactRef:
    """Publish the parent-only timing/process join for one complete DIAG4 cold."""

    supporting_names = DIAG4_EVIDENCE_SLOT_NAMES - frozenset(
        {"execution", "supervisor_terminal"}
    )
    supporting: dict[str, ArtifactRef] = {}
    for name, reference in artifact_refs.items():
        if name in supporting_names and reference is not None:
            supporting[name] = reference
    if frozenset(supporting) != supporting_names:
        raise ValueError("complete DIAG4 execution supporting evidence is absent")
    timing_reference = supporting["cold_solve_timing"]
    producer_reference = supporting["cold_producer"]
    process_reference = supporting["cold_process"]
    payload = diag4_execution_evidence_payload(
        supporting_evidence=supporting,
        solve_timing=load_canonical_json_bytes(
            timing_reference.resolve_and_validate(staging_root).read_bytes()
        ),
        producer=load_canonical_json_bytes(
            producer_reference.resolve_and_validate(staging_root).read_bytes()
        ),
        process=load_canonical_json_bytes(
            process_reference.resolve_and_validate(staging_root).read_bytes()
        ),
    )
    path = staging_root / "execution.json"
    _publish_canonical_json(path, payload)
    return _artifact_ref(path, staging_root, DIAG4_EXECUTION_SCHEMA_VERSION)


def _publish_diag5_execution(
    staging_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    *,
    successor_claim: Diag5SuccessorAuthorityClaim,
) -> ArtifactRef:
    """Publish the DIAG5 process/timing/native/authority join."""

    supporting_names = DIAG5_EVIDENCE_SLOT_NAMES - frozenset(
        {"execution", "supervisor_terminal"}
    )
    supporting = {
        name: reference
        for name, reference in artifact_refs.items()
        if name in supporting_names and reference is not None
    }
    if frozenset(supporting) != supporting_names:
        raise ValueError("complete DIAG5 execution supporting evidence is absent")
    payload = diag5_execution_evidence_payload(
        supporting_evidence=supporting,
        solve_timing=load_canonical_json_bytes(
            supporting["cold_solve_timing"]
            .resolve_and_validate(staging_root)
            .read_bytes()
        ),
        producer=load_canonical_json_bytes(
            supporting["cold_producer"].resolve_and_validate(staging_root).read_bytes()
        ),
        process=load_canonical_json_bytes(
            supporting["cold_process"].resolve_and_validate(staging_root).read_bytes()
        ),
        gpu_native_binding=_diag5_native_binding_payload(successor_claim, role="gpu"),
        authority_sha256=successor_claim.authority_sha256,
    )
    path = staging_root / "execution.json"
    _publish_canonical_json(path, payload)
    return _artifact_ref(path, staging_root, DIAG5_EXECUTION_SCHEMA_VERSION)


def run_diagnostic(
    campaign_root: Path,
    *,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    repo_root: Path | None = None,
) -> dict[str, JsonValue]:
    """Run one gated diagnostic preflight and at most one pristine cold."""

    (
        repository,
        native_extension,
        reference_source,
        inputs,
        executable,
        gpu_uuid,
        memory_bytes,
    ) = _validate_parent_execution_policy(
        repo_root=repo_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
        environment=environment,
    )
    publication = prepare_execution_snapshot(
        campaign_root,
        repo_root=repository,
        native_extension_path=native_extension,
    )
    campaign_reference = copy_validated_reference(reference_source, campaign_root)
    reference_ref = _artifact_ref(
        campaign_reference / REFERENCE_FILENAME,
        campaign_root,
        NATIVE_REFERENCE_SCHEMA_VERSION,
    )
    policy_authority_ref = _publish_parent_policy_authority(
        campaign_reference,
        campaign_root,
    )
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in DIAGNOSTIC_EVIDENCE_REF_KEYS
    }
    refs["source_manifest"] = publication.source_identity(
        campaign_root
    ).snapshot_manifest
    refs["native_reference"] = reference_ref
    refs["policy_authority"] = policy_authority_ref

    def supervise(
        mode: DiagnosticChildMode,
    ) -> tuple[SupervisedSample, SnapshotChildInvocation, bool]:
        directory = campaign_root / mode.value
        directory.mkdir(parents=True, exist_ok=False)
        invocation = build_child_invocation(
            publication,
            campaign_root=campaign_root,
            interpreter=executable,
            reference_root=campaign_reference,
            input_root=inputs,
            sample=SampleName.COLD,
            environment=environment,
            diagnostic_mode=mode,
        )
        try:
            pre_source = _capture_source_identity_evidence(publication, campaign_root)
        except Exception as error:  # noqa: BLE001 - source failure is terminal.
            return (
                SupervisedSample(
                    SampleName.COLD,
                    ChildTerminalStatus.PROTOCOL_FAILURE,
                    0,
                    0,
                    0.0,
                    {},
                    None,
                    (
                        f"SOURCE_PRE:{type(error).__name__}:"
                        + _sha256(str(error).encode()),
                    ),
                    stdout=b"",
                    stderr=b"",
                    memory_samples=(),
                ),
                invocation,
                False,
            )
        try:
            outcome = supervise_sample(
                SampleName.COLD,
                invocation,
                gpu_uuid=gpu_uuid,
                physical_memory_bytes=memory_bytes,
                retain_raw_evidence=True,
            )
        except Exception as error:  # noqa: BLE001 - supervisor failure is evidence.
            outcome = SupervisedSample(
                SampleName.COLD,
                ChildTerminalStatus.CRASH,
                0,
                0,
                0.0,
                {},
                None,
                (f"SUPERVISOR:{type(error).__name__}:{_sha256(str(error).encode())}",),
                stdout=b"",
                stderr=b"",
                memory_samples=(),
            )
        try:
            post_source = _capture_source_identity_evidence(publication, campaign_root)
        except Exception as error:  # noqa: BLE001 - source failure is terminal.
            return (
                replace(
                    outcome,
                    terminal_status=ChildTerminalStatus.PROTOCOL_FAILURE,
                    pre_source_identity=pre_source,
                    failure_reasons=(
                        *outcome.failure_reasons,
                        f"SOURCE_POST:{type(error).__name__}:"
                        + _sha256(str(error).encode()),
                    ),
                ),
                invocation,
                False,
            )
        return (
            replace(
                outcome,
                pre_source_identity=pre_source,
                post_source_identity=post_source,
            ),
            invocation,
            True,
        )

    preflight, preflight_invocation, preflight_source_valid = supervise(
        DiagnosticChildMode.PREFLIGHT
    )
    if not preflight_source_valid:
        refs["source_manifest"] = None
    preflight_refs = _publish_diagnostic_supervision(
        campaign_root,
        campaign_root / "preflight",
        preflight,
        producer_schema=PREFLIGHT_SCHEMA_VERSION,
    )
    refs["preflight"] = preflight_refs["producer"]
    refs["preflight_child_terminal"] = preflight_refs["child_terminal"]
    refs["preflight_process"] = preflight_refs["process"]
    refs["preflight_memory"] = preflight_refs.get("memory")
    refs["preflight_memory_samples"] = preflight_refs["memory_samples"]
    refs["preflight_runtime"] = _published_runtime_reference(
        campaign_root,
        campaign_root / "preflight",
    )
    refs["preflight_policy"] = _published_diagnostic_policy_reference(
        campaign_root,
        campaign_root / "preflight",
    )
    try:
        raw_preflight_refs = {
            "producer": preflight_refs["producer"],
            "child_terminal": preflight_refs["child_terminal"],
            "process": preflight_refs["process"],
            "memory": preflight_refs["memory"],
            "memory_samples": preflight_refs["memory_samples"],
            "runtime": refs["preflight_runtime"],
            "preflight_policy": refs["preflight_policy"],
            "policy_authority": refs["policy_authority"],
            "source_manifest": refs["source_manifest"],
            "native_reference": refs["native_reference"],
        }
        if frozenset(raw_preflight_refs) != PREFLIGHT_EVIDENCE_REF_KEYS or any(
            reference is None for reference in raw_preflight_refs.values()
        ):
            raise ValueError("preflight authority references are incomplete")
        preflight_pass = validate_diagnostic_preflight_gate(
            campaign_root,
            evidence_refs={
                name: reference
                for name, reference in raw_preflight_refs.items()
                if reference is not None
            },
            expected_gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            expected_interpreter=str(executable),
            expected_argv=preflight_invocation.argv,
        )
    except (KeyError, TypeError, ValueError):
        preflight_pass = False
    if not preflight_pass:
        _build_and_seal_diagnostic_receipt(campaign_root, refs)
        return {
            "schema_version": f"{DIAGNOSTIC_SCHEMA_VERSION}-summary",
            "route": DIAGNOSTIC_ROUTE,
            "children": [DiagnosticChildMode.PREFLIGHT.value],
            "verdict": "DIAGNOSTIC_INCOMPLETE",
            "campaign_receipt_produced": False,
            "promotion_authorized": False,
        }

    cold, cold_invocation, cold_source_valid = supervise(DiagnosticChildMode.COLD)
    if not cold_source_valid:
        refs["source_manifest"] = None
    cold_refs = _publish_diagnostic_supervision(
        campaign_root,
        campaign_root / "cold",
        cold,
        producer_schema=f"{DIAGNOSTIC_SCHEMA_VERSION}-producer",
    )
    refs.update(cold_refs)
    refs["runtime"] = _published_runtime_reference(
        campaign_root,
        campaign_root / "cold",
    )
    if cold.terminal_status is not ChildTerminalStatus.COMPLETE:
        _build_and_seal_diagnostic_receipt(campaign_root, refs)
        return {
            "schema_version": f"{DIAGNOSTIC_SCHEMA_VERSION}-summary",
            "route": DIAGNOSTIC_ROUTE,
            "children": [mode.value for mode in DiagnosticChildMode],
            "verdict": "DIAGNOSTIC_INCOMPLETE",
            "campaign_receipt_produced": False,
            "promotion_authorized": False,
        }
    try:
        producer = cold.producer
        refs["history"] = _artifact_from_payload(
            _mapping_field(producer, "history_evidence")
        )
        refs["terminal_numerical"] = _artifact_from_payload(
            _mapping_field(producer, "terminal_numerical_evidence")
        )
        refs["raw_trace"] = _artifact_from_payload(
            _mapping_field(producer, "raw_trace_evidence")
        )
        refs["trace_intervals"] = _artifact_from_payload(
            _mapping_field(producer, "trace_intervals_evidence")
        )
        refs["runtime"] = _artifact_from_payload(
            _mapping_field(producer, "runtime_evidence")
        )
        refs["policy"] = _artifact_from_payload(
            _mapping_field(producer, "policy_evidence")
        )
        supporting_names = DIAGNOSTIC_EVIDENCE_REF_KEYS - frozenset(
            {
                "history",
                "terminal_numerical",
                "raw_trace",
                "trace_intervals",
                "execution",
            }
        )
        complete_support = {
            name: reference
            for name, reference in refs.items()
            if name in supporting_names and reference is not None
        }
        if frozenset(complete_support) != supporting_names:
            raise ValueError("diagnostic supporting evidence is incomplete")
        runtime_document = load_canonical_json_bytes(
            refs["runtime"].resolve_and_validate(campaign_root).read_bytes()
        )
        runtime_payload = _mapping_field(producer, "runtime")
        runtime_document_map = _mapping_field(runtime_document, "runtime_identity")
        environment_sha = str(runtime_document_map["effective_environment_sha256"])
        transfers = _mapping_field(producer, "transfer_audit")
        memory = cold.memory or {}
        pre_source = cold.pre_source_identity
        post_source = cold.post_source_identity
        if pre_source is None or post_source is None:
            raise ValueError("diagnostic source identities are absent")
        execution_payload = execution_evidence_payload(
            supporting_evidence=complete_support,
            preflight={
                "status": "COMPLETE",
                "compile_success": True,
                "solver_dispatched": False,
                "finalizer_called": False,
                "endpoint_audit_called": False,
                "campaign_authorized": False,
                "callbacks": 0,
            },
            cold={
                "status": "COMPLETE",
                "child_pid": cold.child_pid,
                "child_start_time_ticks": cold.child_start_time_ticks,
                "backend": str(runtime_payload["backend"]),
                "gpu_uuid": gpu_uuid,
                "jax_enable_x64": runtime_payload["jax_enable_x64"] is True,
                "state_size": 716,
                "equality_size": 255,
                "residual_size": 2110,
                "policy_sha256": str(producer["policy_sha256"]),
                "phase_schema_sha256": str(producer["phase_schema_sha256"]),
                "source_pre_sha256": _sha256(
                    canonical_json_bytes(pre_source.to_payload())
                ),
                "source_post_sha256": _sha256(
                    canonical_json_bytes(post_source.to_payload())
                ),
                "runtime_environment_sha256": environment_sha,
                "interpreter": str(executable),
                "argv": list(cold_invocation.argv),
                "physical_memory_bytes": memory_bytes,
                "peak_memory_bytes": int(memory["peak_memory_bytes"]),
                "peak_memory_fraction": float(memory["peak_memory_fraction"]),
                "hot_h2d_transfers": int(transfers["hot_h2d_transfers"]),
                "hot_d2h_transfers": int(transfers["hot_d2h_transfers"]),
                "python_callbacks": int(transfers["python_callbacks"]),
                "final_d2h_transfers": int(transfers["final_d2h_transfers"]),
                "timestamps_ns": _mapping_field(producer, "timestamps_ns"),
                "stdout_sha256": _sha256(cold.stdout or b""),
                "stdout_size_bytes": len(cold.stdout or b""),
                "stderr_sha256": _sha256(cold.stderr or b""),
                "stderr_size_bytes": len(cold.stderr or b""),
            },
        )
        execution_path = campaign_root / "execution.json"
        _publish_canonical_json(execution_path, execution_payload)
        refs["execution"] = _artifact_ref(
            execution_path,
            campaign_root,
            f"{DIAGNOSTIC_SCHEMA_VERSION}-execution",
        )
    except (KeyError, TypeError, ValueError):
        pass
    receipt = _build_and_seal_diagnostic_receipt(campaign_root, refs)
    return {
        "schema_version": f"{DIAGNOSTIC_SCHEMA_VERSION}-summary",
        "route": DIAGNOSTIC_ROUTE,
        "children": [mode.value for mode in DiagnosticChildMode],
        "verdict": diagnostic_receipt_payload(receipt)["verdict"],
        "campaign_receipt_produced": False,
        "promotion_authorized": False,
    }


def run_diag4(
    final_root: Path,
    *,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    successor_claim: Diag4SuccessorAuthorityClaim,
    repo_root: Path | None = None,
) -> dict[str, JsonValue]:
    """Run the authority-bound trace-free preflight and at most one cold."""

    if (
        os.environ.get("JAX_PLATFORMS") != "cpu"
        or os.environ.get("JAX_PLATFORM_NAME") is not None
        or os.environ.get("JAX_COMPILATION_CACHE_DIR") is not None
        or os.environ.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
    ):
        raise ValueError("DIAG4 parent process is not bound to the CPU-only policy")
    read_linux_process_identity(os.getpid())
    query_executable_sha256 = supervisor_query_executable_sha256()
    (
        repository,
        native_extension,
        reference_source,
        inputs,
        executable,
        gpu_uuid,
        memory_bytes,
    ) = _validate_parent_execution_policy(
        repo_root=repo_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
        environment=environment,
    )
    revalidate_diag4_successor_authority(successor_claim, require_output_absent=True)
    publication = _prepare_diag2_publication(
        final_root,
        repository_root=repository,
    )
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in DIAG4_EVIDENCE_SLOT_PATHS
    }
    launched_children: tuple[str, ...] = ()

    try:
        bind_diag4_staging_root(successor_claim, publication.staging_root)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _leave_diag4_visible_partial(
            publication,
            refs,
            outcome=_diag4_failure(
                FailureStageV4.AUTHORITY,
                FailureReasonCodeV4.LOCK_CLAIM_FAILED,
                f"{type(error).__name__}:{error}",
            ),
            launched_children=(),
        )

    def finish(outcome: StructuredFailureV4) -> dict[str, JsonValue]:
        publication_result = _publish_diag4_terminal_and_receipt(
            publication,
            refs,
            outcome=outcome,
            launched_children=launched_children,
            successor_claim=successor_claim,
        )
        if isinstance(publication_result, Diag4VisiblePartial):
            return {
                "schema_version": DIAG4_SUMMARY_SCHEMA_VERSION,
                "route": DIAG4_ROUTE,
                "children": list(launched_children),
                "verdict": "VISIBLE_PARTIAL",
                "next_route": "NOT_PRODUCED",
                "speed_comparison": "NOT_PRODUCED",
                "promotion_authorized": False,
                "formal_comparison": "NOT_PRODUCED",
                "partial_root": str(publication_result.root),
                "failure_stage": publication_result.outcome.stage.value,
                "failure_reason": publication_result.outcome.reason.value,
            }
        receipt = publication_result
        return {
            "schema_version": DIAG4_SUMMARY_SCHEMA_VERSION,
            "route": DIAG4_ROUTE,
            "children": list(launched_children),
            "verdict": receipt.verdict,
            "next_route": receipt.next_route,
            "speed_comparison": receipt.speed_comparison,
            "promotion_authorized": False,
            "formal_comparison": "NOT_PRODUCED",
        }

    try:
        source = _prepare_diag4_snapshot(
            publication.staging_root,
            native_extension_path=native_extension,
            successor_claim=successor_claim,
        )
        validate_diag4_successor_snapshot(source, successor_claim)
        refs["source_manifest"] = source.source_identity(
            publication.staging_root
        ).snapshot_manifest
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.SETUP,
                FailureReasonCodeV4.SOURCE_PUBLICATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        subset_path = publication.staging_root / "frozen-numerical-subset.json"
        subset_payload = build_diag4_frozen_numerical_subset_payload(
            successor_claim.expected_frozen_numerical_entries
        )
        _publish_canonical_json(subset_path, subset_payload)
        validate_diag4_frozen_numerical_subset_payload(
            subset_payload,
            artifact_root=publication.staging_root,
            expected_entries=successor_claim.expected_frozen_numerical_entries,
        )
        refs["frozen_numerical_subset"] = _artifact_ref(
            subset_path,
            publication.staging_root,
            DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.SETUP,
                FailureReasonCodeV4.FROZEN_NUMERICAL_SUBSET_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        campaign_reference = copy_validated_reference(
            reference_source, publication.staging_root
        )
        refs["native_reference"] = _artifact_ref(
            campaign_reference / REFERENCE_FILENAME,
            publication.staging_root,
            NATIVE_REFERENCE_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.SETUP,
                FailureReasonCodeV4.NATIVE_REFERENCE_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        policy_reference, _ = _publish_diag2_policy_authority(
            publication.staging_root,
            campaign_reference,
            refs["native_reference"],
        )
        refs["policy_authority"] = policy_reference
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.SETUP,
                FailureReasonCodeV4.POLICY_AUTHORITY_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        validate_diag4_successor_snapshot(source, successor_claim)
        validate_diag4_frozen_numerical_subset_payload(
            load_canonical_json_bytes(
                refs["frozen_numerical_subset"]
                .resolve_and_validate(publication.staging_root)
                .read_bytes()
            ),
            artifact_root=publication.staging_root,
            expected_entries=successor_claim.expected_frozen_numerical_entries,
        )
        validate_diag2_policy_authority_payload(
            load_canonical_json_bytes(
                refs["policy_authority"]
                .resolve_and_validate(publication.staging_root)
                .read_bytes()
            ),
            artifact_root=publication.staging_root,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.SETUP,
                FailureReasonCodeV4.SETUP_DEEP_LOAD_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )

    zero_failure = _diag4_supervisor_zero_gate(
        publication,
        refs,
        environment=environment,
        query_executable_sha256=query_executable_sha256,
        stage=FailureStageV4.BEFORE_PREFLIGHT,
    )
    if zero_failure is not None:
        return finish(zero_failure)

    try:
        preflight_directory = publication.staging_root / "preflight"
        preflight_directory.mkdir(mode=0o755, exist_ok=False)
        preflight_invocation = build_child_invocation(
            source,
            campaign_root=publication.staging_root,
            interpreter=executable,
            reference_root=campaign_reference,
            input_root=inputs,
            sample=SampleName.COLD,
            environment=environment,
            diagnostic_mode=DiagnosticChildMode.PREFLIGHT,
            diag4=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.PREFLIGHT,
                FailureReasonCodeV4.PREFLIGHT_LAUNCH_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        consume_diag4_successor_authority(successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        consumption_reason = _diag4_authority_consumption_failure_reason(
            successor_claim
        )
        return finish(
            _diag4_failure(
                FailureStageV4.BEFORE_PREFLIGHT,
                consumption_reason,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        preflight = supervise_diag2_sample(
            SampleName.COLD,
            preflight_invocation,
            mode=DiagnosticChildMode.PREFLIGHT,
            gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            validate_producer=validate_diag4_producer_payload,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.PREFLIGHT,
                FailureReasonCodeV4.PREFLIGHT_LAUNCH_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    launched_children = ("preflight",) if preflight.launched else ()
    if preflight.launched:
        supervision_failure = _publish_diag4_child_supervision(
            publication,
            refs,
            preflight_directory,
            preflight,
            mode=DiagnosticChildMode.PREFLIGHT,
        )
        if supervision_failure is not None:
            return finish(supervision_failure)
    preflight_failure = _diag4_child_failure(DiagnosticChildMode.PREFLIGHT, preflight)
    if preflight_failure is not None:
        return finish(preflight_failure)
    try:
        validate_diag4_preflight_gate(
            publication.staging_root,
            evidence_slots={
                name: EvidenceSlotV4.present(reference)
                for name, reference in refs.items()
                if reference is not None
            },
            expected_gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            expected_interpreter=str(executable),
            expected_argv=preflight_invocation.argv,
            expected_identity=successor_claim.expected_numerical_identity,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.PREFLIGHT,
                FailureReasonCodeV4.PREFLIGHT_GATE_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )

    zero_failure = _diag4_supervisor_zero_gate(
        publication,
        refs,
        environment=environment,
        query_executable_sha256=query_executable_sha256,
        stage=FailureStageV4.BEFORE_COLD,
    )
    if zero_failure is not None:
        return finish(zero_failure)
    try:
        validate_diag4_successor_snapshot(source, successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.BEFORE_COLD,
                FailureReasonCodeV4.SOURCE_REVALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        revalidate_diag4_successor_authority(
            successor_claim, require_output_absent=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.BEFORE_COLD,
                _diag4_before_cold_authority_failure_reason(error),
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        cold_directory = publication.staging_root / "cold"
        cold_directory.mkdir(mode=0o755, exist_ok=False)
        cold_invocation = build_child_invocation(
            source,
            campaign_root=publication.staging_root,
            interpreter=executable,
            reference_root=campaign_reference,
            input_root=inputs,
            sample=SampleName.COLD,
            environment=environment,
            diagnostic_mode=DiagnosticChildMode.COLD,
            diag4=True,
        )
        cold = supervise_diag2_sample(
            SampleName.COLD,
            cold_invocation,
            mode=DiagnosticChildMode.COLD,
            gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            validate_producer=validate_diag4_producer_payload,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag4_failure(
                FailureStageV4.COLD,
                FailureReasonCodeV4.COLD_LAUNCH_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    launched_children = ("preflight", "cold") if cold.launched else ("preflight",)
    cold_failure = _diag4_child_failure(DiagnosticChildMode.COLD, cold)
    commit_failure: StructuredFailureV4 | None = None
    if cold.launched:
        cold, commit_failure = _resolve_diag4_cold_numerical_bundle_v4(
            cold_directory, cold
        )
        supervision_failure = _publish_diag4_child_supervision(
            publication,
            refs,
            cold_directory,
            cold,
            mode=DiagnosticChildMode.COLD,
        )
        if supervision_failure is not None:
            return finish(supervision_failure)
    if cold_failure is not None:
        return finish(cold_failure)
    if commit_failure is not None:
        if commit_failure.reason in {
            FailureReasonCodeV4.QUARANTINE_FAILED,
            FailureReasonCodeV4.COMMIT_COLLISION,
            FailureReasonCodeV4.COMMIT_RENAME_FAILED,
            FailureReasonCodeV4.COMMIT_FSYNC_FAILED,
            FailureReasonCodeV4.COMMITTED_DEEP_LOAD_FAILED,
        }:
            return _leave_diag4_visible_partial(
                publication,
                refs,
                outcome=commit_failure,
                launched_children=launched_children,
            )
        return finish(commit_failure)
    try:
        if cold.producer is None:
            raise ValueError("complete DIAG4 cold omits its typed producer")
        for slot_name, producer_field in (
            ("cold_history", "history_evidence"),
            ("cold_terminal_numerical", "terminal_numerical_evidence"),
            ("cold_solve_timing", "solve_timing_evidence"),
            ("cold_safeguard_telemetry", "safeguard_telemetry_evidence"),
        ):
            refs[slot_name] = _artifact_from_payload(
                _mapping_field(cold.producer, producer_field)
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return finish(
            _diag4_receipt_failure(
                FailureReasonCodeV4.EVIDENCE_VECTOR_INVALID,
                error,
            )
        )
    try:
        refs["execution"] = _publish_diag4_execution(publication.staging_root, refs)
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag4_receipt_failure(
                FailureReasonCodeV4.EVIDENCE_VECTOR_INVALID,
                error,
            )
        )
    try:
        scientific_outcome = derive_diag4_scientific_outcome(
            artifact_root=publication.staging_root,
            artifact_refs=refs,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag4_receipt_failure(
                FailureReasonCodeV4.SCIENTIFIC_RECONSTRUCTION_FAILED,
                error,
            )
        )
    return finish(scientific_outcome)


def run_diag5(
    final_root: Path,
    *,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    successor_claim: Diag5SuccessorAuthorityClaim,
    expected_source_snapshot: SnapshotIdentity,
    repo_root: Path | None = None,
) -> dict[str, JsonValue]:
    """Run the one-shot authority-bound DIAG5 preflight and optional cold."""

    if (
        os.environ.get("JAX_PLATFORMS") != "cpu"
        or os.environ.get("JAX_PLATFORM_NAME") is not None
        or os.environ.get("JAX_COMPILATION_CACHE_DIR") is not None
        or os.environ.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
    ):
        raise ValueError("DIAG5 parent process is not bound to the CPU-only policy")
    read_linux_process_identity(os.getpid())
    query_executable_sha256 = supervisor_query_executable_sha256()
    (
        repository,
        native_extension,
        reference_source,
        inputs,
        executable,
        gpu_uuid,
        memory_bytes,
    ) = _validate_parent_execution_policy(
        repo_root=repo_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
        environment=environment,
    )
    gpu_binding = successor_claim.gpu_native_binding
    if (
        native_extension != gpu_binding.path
        or reference_source
        != Path(str(successor_claim.expected_native_reference["absolute_root"]))
        or inputs != Path(str(successor_claim.expected_input_bundle["absolute_root"]))
        or executable
        != Path(str(successor_claim.expected_interpreter["absolute_path"]))
        or gpu_uuid != successor_claim.expected_gpu_uuid
    ):
        raise ValueError("DIAG5 live execution inputs differ from authority")
    revalidate_diag5_successor_authority(successor_claim)
    publication = _prepare_diag5_publication(
        final_root,
        repository_root=repository,
        successor_claim=successor_claim,
    )
    bind_diag5_staging_root(successor_claim, publication.staging_root)
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in DIAG5_EVIDENCE_SLOT_PATHS
    }
    launched_children: tuple[str, ...] = ()
    source: SnapshotPublication | None = None

    def finish(outcome: StructuredFailureV5) -> dict[str, JsonValue]:
        receipt = _publish_diag5_terminal_and_receipt(
            publication,
            refs,
            outcome=outcome,
            launched_children=launched_children,
            successor_claim=successor_claim,
            source=source,
            expected_source_snapshot=expected_source_snapshot,
            physical_memory_bytes=memory_bytes,
        )
        return {
            "schema_version": "single-stage-neq-gntr3-diag5-summary-v1",
            "route": DIAG5_ROUTE,
            "children": list(launched_children),
            "verdict": receipt.verdict,
            "next_route": receipt.next_route,
            "speed_comparison": receipt.speed_comparison,
            "promotion_authorized": False,
            "formal_comparison": "NOT_PRODUCED",
        }

    try:
        revalidate_diag5_successor_authority(successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.AUTHORITY,
                FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        _copy_diag5_predecessor_postmortem(
            publication.staging_root,
            successor_claim,
        )
        source = _prepare_diag5_snapshot(
            publication.staging_root,
            successor_claim=successor_claim,
        )
        validate_diag5_successor_snapshot(source, successor_claim)
        refs["source_manifest"] = source.source_identity(
            publication.staging_root
        ).snapshot_manifest
    except SnapshotPostPublicationError as error:
        source = error.publication
        return finish(
            _diag5_failure(
                FailureStageV5.SETUP,
                FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED,
                f"{type(error.cause).__name__}:{error.cause}",
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.SETUP,
                FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        subset_path = publication.staging_root / "frozen-numerical-subset.json"
        subset_payload = build_diag5_frozen_numerical_subset_payload(
            successor_claim.expected_frozen_numerical_entries
        )
        _publish_canonical_json(subset_path, subset_payload)
        validate_diag5_frozen_numerical_subset_payload(
            subset_payload,
            artifact_root=publication.staging_root,
            expected_entries=successor_claim.expected_frozen_numerical_entries,
        )
        refs["frozen_numerical_subset"] = _artifact_ref(
            subset_path,
            publication.staging_root,
            DIAG5_FROZEN_SUBSET_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.SETUP,
                FailureReasonCodeV5.FROZEN_NUMERICAL_SUBSET_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        campaign_reference = copy_validated_reference(
            reference_source, publication.staging_root
        )
        refs["native_reference"] = _artifact_ref(
            campaign_reference / REFERENCE_FILENAME,
            publication.staging_root,
            NATIVE_REFERENCE_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.SETUP,
                FailureReasonCodeV5.NATIVE_REFERENCE_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        policy_reference, _ = _publish_diag5_policy_authority(
            publication.staging_root,
            campaign_reference,
            refs["native_reference"],
        )
        refs["policy_authority"] = policy_reference
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.SETUP,
                FailureReasonCodeV5.POLICY_AUTHORITY_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        validate_diag5_successor_snapshot(source, successor_claim)
        validate_diag5_frozen_numerical_subset_payload(
            load_canonical_json_bytes(
                refs["frozen_numerical_subset"]
                .resolve_and_validate(publication.staging_root)
                .read_bytes()
            ),
            artifact_root=publication.staging_root,
            expected_entries=successor_claim.expected_frozen_numerical_entries,
        )
        validate_diag5_policy_authority_payload(
            load_canonical_json_bytes(
                refs["policy_authority"]
                .resolve_and_validate(publication.staging_root)
                .read_bytes()
            ),
            artifact_root=publication.staging_root,
        )
        revalidate_diag5_successor_authority(successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.SETUP,
                FailureReasonCodeV5.SETUP_DEEP_LOAD_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    zero_failure = _diag5_supervisor_zero_gate(
        publication,
        refs,
        environment=environment,
        query_executable_sha256=query_executable_sha256,
        stage=FailureStageV5.BEFORE_PREFLIGHT,
    )
    if zero_failure is not None:
        return finish(zero_failure)
    try:
        preflight_directory = publication.staging_root / "preflight"
        preflight_directory.mkdir(mode=0o755, exist_ok=False)
        preflight_invocation = build_child_invocation(
            source,
            campaign_root=publication.staging_root,
            interpreter=executable,
            reference_root=campaign_reference,
            input_root=inputs,
            sample=SampleName.COLD,
            environment=environment,
            diagnostic_mode=DiagnosticChildMode.PREFLIGHT,
            diag5=True,
            expected_native_extension_path=gpu_binding.path,
            expected_native_extension_sha256=gpu_binding.sha256,
            expected_native_extension_size_bytes=gpu_binding.size_bytes,
            expected_native_extension_link_count=gpu_binding.link_count,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.PREFLIGHT,
                FailureReasonCodeV5.PREFLIGHT_LAUNCH_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        validate_diag5_successor_snapshot(source, successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.SETUP,
                FailureReasonCodeV5.SETUP_DEEP_LOAD_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        revalidate_diag5_successor_authority(successor_claim)
        consume_diag5_successor_authority(successor_claim)
        validate_diag5_consumption_marker(successor_claim)
        revalidate_diag5_successor_authority(successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(_diag5_consumption_failure(successor_claim, error))
    try:
        validate_diag5_successor_snapshot(source, successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.BEFORE_PREFLIGHT,
                FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        preflight = supervise_diag2_sample(
            SampleName.COLD,
            preflight_invocation,
            mode=DiagnosticChildMode.PREFLIGHT,
            gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            validate_producer=validate_diag5_producer_payload,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.PREFLIGHT,
                FailureReasonCodeV5.PREFLIGHT_LAUNCH_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    launched_children = ("preflight",) if preflight.launched else ()
    preflight_failure = _diag5_child_failure(DiagnosticChildMode.PREFLIGHT, preflight)
    if preflight.launched:
        supervision_failure = _publish_diag5_child_supervision(
            publication,
            refs,
            preflight_directory,
            preflight,
            mode=DiagnosticChildMode.PREFLIGHT,
        )
        if supervision_failure is not None:
            return finish(supervision_failure)
        try:
            revalidate_diag5_successor_authority(successor_claim)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if preflight_failure is not None:
                return finish(preflight_failure)
            return finish(
                _diag5_failure(
                    FailureStageV5.PREFLIGHT,
                    FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
                    f"{type(error).__name__}:{error}",
                )
            )
        try:
            validate_diag5_successor_snapshot(source, successor_claim)
        except (OSError, RuntimeError, TypeError, ValueError):
            return finish(
                _diag5_post_child_source_failure(
                    DiagnosticChildMode.PREFLIGHT,
                    preflight_failure,
                )
            )
    if preflight_failure is not None:
        return finish(preflight_failure)
    try:
        validate_diag5_preflight_gate(
            publication.staging_root,
            evidence_slots={
                name: EvidenceSlotV5.present(reference)
                for name, reference in refs.items()
                if reference is not None
            },
            expected_gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            expected_interpreter=str(executable),
            expected_argv=preflight_invocation.argv,
            expected_identity=successor_claim.expected_numerical_identity,
            expected_frozen_numerical_entries=(
                successor_claim.expected_frozen_numerical_entries
            ),
            gpu_native_binding=_diag5_native_binding_payload(
                successor_claim, role="gpu"
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.PREFLIGHT,
                FailureReasonCodeV5.PREFLIGHT_GATE_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    zero_failure = _diag5_supervisor_zero_gate(
        publication,
        refs,
        environment=environment,
        query_executable_sha256=query_executable_sha256,
        stage=FailureStageV5.BEFORE_COLD,
    )
    if zero_failure is not None:
        return finish(zero_failure)
    try:
        validate_diag5_successor_snapshot(source, successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.BEFORE_COLD,
                FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        revalidate_diag5_successor_authority(successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.BEFORE_COLD,
                FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        validate_diag5_consumption_marker(successor_claim)
    except Diag5ConsumptionMarkerInvalidError as error:
        return finish(
            _diag5_failure(
                FailureStageV5.BEFORE_COLD,
                FailureReasonCodeV5.CONSUMPTION_MARKER_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        cold_directory = publication.staging_root / "cold"
        cold_directory.mkdir(mode=0o755, exist_ok=False)
        cold_invocation = build_child_invocation(
            source,
            campaign_root=publication.staging_root,
            interpreter=executable,
            reference_root=campaign_reference,
            input_root=inputs,
            sample=SampleName.COLD,
            environment=environment,
            diagnostic_mode=DiagnosticChildMode.COLD,
            diag5=True,
            expected_native_extension_path=gpu_binding.path,
            expected_native_extension_sha256=gpu_binding.sha256,
            expected_native_extension_size_bytes=gpu_binding.size_bytes,
            expected_native_extension_link_count=gpu_binding.link_count,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.COLD,
                FailureReasonCodeV5.COLD_LAUNCH_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        validate_diag5_successor_snapshot(source, successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.BEFORE_COLD,
                FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        revalidate_diag5_successor_authority(successor_claim)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.BEFORE_COLD,
                FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        cold = supervise_diag2_sample(
            SampleName.COLD,
            cold_invocation,
            mode=DiagnosticChildMode.COLD,
            gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            validate_producer=validate_diag5_producer_payload,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.COLD,
                FailureReasonCodeV5.COLD_LAUNCH_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    launched_children = ("preflight", "cold") if cold.launched else ("preflight",)
    cold_failure = _diag5_child_failure(DiagnosticChildMode.COLD, cold)
    if cold.launched:
        resolution = _resolve_diag5_cold_numerical_bundle_v5(cold_directory, cold)
        cold = resolution.outcome
        if not resolution.publication_allowed:
            raise Diag5PendingDispositionError(
                terminal_failure=cold_failure,
                pending_disposition_failure=resolution.pending_disposition_failure,
                staging_root=publication.staging_root,
            )
        supervision_failure = _publish_diag5_child_supervision(
            publication,
            refs,
            cold_directory,
            cold,
            mode=DiagnosticChildMode.COLD,
            defer_success_auxiliary_slots=True,
        )
        if supervision_failure is not None:
            return finish(supervision_failure)
        try:
            revalidate_diag5_successor_authority(successor_claim)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if cold_failure is not None:
                return finish(cold_failure)
            return finish(
                _diag5_failure(
                    FailureStageV5.COLD,
                    FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
                    f"{type(error).__name__}:{error}",
                )
            )
        try:
            validate_diag5_successor_snapshot(source, successor_claim)
        except (OSError, RuntimeError, TypeError, ValueError):
            return finish(
                _diag5_post_child_source_failure(
                    DiagnosticChildMode.COLD,
                    cold_failure,
                )
            )
        if cold_failure is None:
            try:
                refs.update(
                    _diag5_child_success_auxiliary_references(
                        publication,
                        mode=DiagnosticChildMode.COLD,
                    )
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                return finish(
                    _diag5_failure(
                        FailureStageV5.COLD,
                        FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
                        f"{type(error).__name__}:{error}",
                    )
                )
    else:
        resolution = Diag5ColdNumericalResolution(cold, None, None)
    if cold_failure is not None:
        return finish(cold_failure)
    if resolution.terminal_failure is not None:
        return finish(resolution.terminal_failure)
    execution_path = publication.staging_root / "execution.json"
    try:
        if cold.producer is None:
            raise ValueError("complete DIAG5 cold omits its typed producer")
        numerical_refs: dict[str, ArtifactRef] = {}
        for slot_name, producer_field in (
            ("cold_history", "history_evidence"),
            ("cold_terminal_numerical", "terminal_numerical_evidence"),
            ("cold_solve_timing", "solve_timing_evidence"),
            ("cold_safeguard_telemetry", "safeguard_telemetry_evidence"),
        ):
            numerical_refs[slot_name] = _artifact_from_payload(
                _mapping_field(cold.producer, producer_field)
            )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.NUMERICAL_COMMIT,
                FailureReasonCodeV5.PENDING_RESULT_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )
    refs.update(numerical_refs)
    try:
        refs["execution"] = _publish_diag5_execution(
            publication.staging_root,
            refs,
            successor_claim=successor_claim,
        )
    except (OSError, TypeError, ValueError) as error:
        execution_path.unlink(missing_ok=True)
        return finish(
            _diag5_failure(
                FailureStageV5.RECEIPT,
                FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )
    try:
        scientific_outcome = derive_diag5_scientific_outcome(
            artifact_root=publication.staging_root,
            artifact_refs=refs,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag5_failure(
                FailureStageV5.RECEIPT,
                FailureReasonCodeV5.SCIENTIFIC_RECONSTRUCTION_FAILED,
                f"{type(error).__name__}:{error}",
            )
        )
    return finish(scientific_outcome)


def run_diag2(
    final_root: Path,
    *,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    repo_root: Path | None = None,
    successor_claim: SuccessorAuthorityClaim | None = None,
) -> dict[str, JsonValue]:
    """Run the CPU-supervised DIAG2 preflight and at most one pristine cold."""

    if (
        os.environ.get("JAX_PLATFORMS") != "cpu"
        or os.environ.get("JAX_PLATFORM_NAME") is not None
        or os.environ.get("JAX_COMPILATION_CACHE_DIR") is not None
        or os.environ.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
    ):
        raise ValueError("DIAG2 parent process is not bound to the CPU-only policy")
    read_linux_process_identity(os.getpid())
    query_executable_sha256 = supervisor_query_executable_sha256()
    (
        repository,
        native_extension,
        reference_source,
        inputs,
        executable,
        gpu_uuid,
        memory_bytes,
    ) = _validate_parent_execution_policy(
        repo_root=repo_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
        environment=environment,
    )
    publication = _prepare_diag2_publication(
        final_root,
        repository_root=repository,
    )
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in DIAG2_EVIDENCE_SLOT_NAMES
    }
    launched_children: tuple[str, ...] = ()
    policy_produced = False
    preflight_authorized = False
    cold_authorized = False

    def finish(failure: StructuredFailureV2 | None) -> dict[str, JsonValue]:
        receipt = _publish_diag2_terminal_and_receipt(
            publication,
            refs,
            failure=failure,
            launched_children=launched_children,
            policy_authority_produced=policy_produced,
            preflight_authorized=preflight_authorized,
            cold_authorized=cold_authorized,
        )
        return {
            "schema_version": "single-stage-neq-gntr1-diag2-summary-v1",
            "route": DIAG2_ROUTE,
            "children": list(launched_children),
            "verdict": receipt.verdict,
            "next_route": receipt.next_route,
            "engineering_campaign_receipt_produced": False,
            "promotion_authorized": False,
        }

    try:
        source = _prepare_diag2_snapshot(
            publication.staging_root,
            repo_root=repository,
            native_extension_path=native_extension,
        )
        if successor_claim is not None:
            validate_successor_snapshot(source, successor_claim)
        refs["source_manifest"] = source.source_identity(
            publication.staging_root
        ).snapshot_manifest
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag2_failure(
                FailureStageV2.SOURCE_PUBLICATION_FAILURE,
                FailureReasonCodeV2.SOURCE_PRE,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        pre_source = _capture_source_identity_evidence(source, publication.staging_root)
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag2_failure(
                FailureStageV2.SOURCE_PUBLICATION_FAILURE,
                FailureReasonCodeV2.SOURCE_POST,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        subset_path = publication.staging_root / "frozen-numerical-subset.json"
        _publish_canonical_json(
            subset_path,
            build_diag2_frozen_numerical_subset_payload(),
        )
        refs["frozen_numerical_subset"] = _artifact_ref(
            subset_path,
            publication.staging_root,
            DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag2_failure(
                FailureStageV2.SOURCE_PUBLICATION_FAILURE,
                FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        campaign_reference = copy_validated_reference(
            reference_source, publication.staging_root
        )
        refs["native_reference"] = _artifact_ref(
            campaign_reference / REFERENCE_FILENAME,
            publication.staging_root,
            NATIVE_REFERENCE_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag2_failure(
                FailureStageV2.NATIVE_REFERENCE_FAILURE,
                FailureReasonCodeV2.REFERENCE_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )

    try:
        policy_reference, _policy_values = _publish_diag2_policy_authority(
            publication.staging_root,
            campaign_reference,
            refs["native_reference"],
        )
        refs["policy_authority"] = policy_reference
        policy_produced = True
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag2_failure(
                FailureStageV2.POLICY_AUTHORITY_FAILURE,
                FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )

    setup_slots = {
        name: EvidenceSlot.present(reference)
        for name, reference in refs.items()
        if reference is not None
    }
    try:
        validate_diag2_setup_authorities(
            publication.staging_root,
            evidence_slots=setup_slots,
        )
    except Diag2SetupGateError as error:
        stage = {
            FailureReasonCodeV2.SOURCE_PRE: (FailureStageV2.SOURCE_PUBLICATION_FAILURE),
            FailureReasonCodeV2.FROZEN_SUBSET_INVALID: (
                FailureStageV2.SOURCE_PUBLICATION_FAILURE
            ),
            FailureReasonCodeV2.REFERENCE_INVALID: (
                FailureStageV2.NATIVE_REFERENCE_FAILURE
            ),
            FailureReasonCodeV2.POLICY_DERIVATION_INVALID: (
                FailureStageV2.POLICY_AUTHORITY_FAILURE
            ),
        }[error.reason]
        return finish(StructuredFailureV2(stage, error.reason, error.detail_sha256))

    before_preflight = _capture_diag2_supervisor_zero(
        environment,
        query_executable_sha256=query_executable_sha256,
    )
    refs["supervisor_before_preflight"] = _publish_diag2_supervisor_zero(
        publication.staging_root,
        before_preflight,
        stage="BEFORE_PREFLIGHT",
    )
    try:
        validate_diag2_supervisor_zero_payload(
            load_canonical_json_bytes(
                refs["supervisor_before_preflight"]
                .resolve_and_validate(publication.staging_root)
                .read_bytes()
            ),
            artifact_root=publication.staging_root,
            expected_stage="BEFORE_PREFLIGHT",
        )
    except (OSError, TypeError, ValueError) as error:
        reason = (
            FailureReasonCodeV2.GPU_PARENT_PID_PRESENT
            if before_preflight.matching_rows
            else FailureReasonCodeV2.GPU_QUERY_FAILED
        )
        return finish(
            _diag2_failure(
                FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE,
                reason,
                f"{type(error).__name__}:{error}",
            )
        )

    preflight_directory = publication.staging_root / "preflight"
    preflight_directory.mkdir(mode=0o755, exist_ok=False)
    preflight_invocation = build_child_invocation(
        source,
        campaign_root=publication.staging_root,
        interpreter=executable,
        reference_root=campaign_reference,
        input_root=inputs,
        sample=SampleName.COLD,
        environment=environment,
        diagnostic_mode=DiagnosticChildMode.PREFLIGHT,
        diag2=True,
    )
    preflight = supervise_diag2_sample(
        SampleName.COLD,
        preflight_invocation,
        mode=DiagnosticChildMode.PREFLIGHT,
        gpu_uuid=gpu_uuid,
        physical_memory_bytes=memory_bytes,
        validate_producer=validate_diag3_producer_payload,
    )
    launched_children = ("preflight",) if preflight.launched else ()
    preflight_source_failure_detail: str | None = None
    try:
        post_source = _capture_source_identity_evidence(
            source, publication.staging_root
        )
    except (OSError, TypeError, ValueError) as error:
        preflight_source_failure_detail = f"{type(error).__name__}:{error}"
        preflight = replace(
            preflight,
            selected_failure_reason=FailureReasonCodeV2.SOURCE_POST,
            raw_failure_reasons=(
                *preflight.raw_failure_reasons,
                f"SOURCE_POST:{type(error).__name__}:{_sha256(str(error).encode())}",
            ),
            pre_source_identity=pre_source,
        )
    else:
        preflight = replace(
            preflight,
            pre_source_identity=pre_source,
            post_source_identity=post_source,
        )
    if preflight.launched:
        preflight_refs = _publish_diag2_supervision(
            publication.staging_root,
            preflight_directory,
            preflight,
            producer_schema=PREFLIGHT_SCHEMA_VERSION,
        )
        for suffix in (
            "producer",
            "terminal",
            "process",
            "memory",
            "memory_samples",
        ):
            refs[f"preflight_{suffix}"] = preflight_refs[suffix]
        refs["preflight_runtime"] = _diag2_existing_reference(
            publication.staging_root,
            "preflight/runtime-evidence.json",
            RUNTIME_EVIDENCE_SCHEMA_VERSION,
        )
        refs["preflight_policy"] = _diag2_existing_reference(
            publication.staging_root,
            "preflight/policy.json",
            f"{DIAGNOSTIC_SCHEMA_VERSION}-policy",
        )
        if preflight_source_failure_detail is not None:
            return finish(
                _diag2_postlaunch_setup_and_child_failure(
                    publication.staging_root,
                    refs,
                    DiagnosticChildMode.PREFLIGHT,
                    preflight,
                    reason=FailureReasonCodeV2.SOURCE_POST,
                    detail_sha256=_sha256(
                        preflight_source_failure_detail.encode("utf-8")
                    ),
                )
            )
        post_preflight_setup_slots = {
            name: EvidenceSlot.present(reference)
            for name, reference in refs.items()
            if reference is not None
        }
        try:
            validate_diag2_setup_authorities(
                publication.staging_root,
                evidence_slots=post_preflight_setup_slots,
            )
        except Diag2SetupGateError as error:
            postlaunch_reason = (
                FailureReasonCodeV2.SOURCE_POST
                if error.reason is FailureReasonCodeV2.SOURCE_PRE
                else error.reason
            )
            return finish(
                _diag2_postlaunch_setup_and_child_failure(
                    publication.staging_root,
                    refs,
                    DiagnosticChildMode.PREFLIGHT,
                    preflight,
                    reason=postlaunch_reason,
                    detail_sha256=error.detail_sha256,
                )
            )
    failure = (
        _diag2_subordinate_child_failure(
            publication.staging_root,
            refs,
            DiagnosticChildMode.PREFLIGHT,
            preflight,
        )
        if preflight.launched
        else _diag2_child_failure(DiagnosticChildMode.PREFLIGHT, preflight)
    )
    if failure is not None:
        return finish(failure)

    preflight_slots = {
        name: EvidenceSlot.present(reference)
        for name, reference in refs.items()
        if reference is not None
    }
    try:
        validate_diag2_preflight_gate(
            publication.staging_root,
            evidence_slots=preflight_slots,
            expected_gpu_uuid=gpu_uuid,
            physical_memory_bytes=memory_bytes,
            expected_interpreter=str(executable),
            expected_argv=preflight_invocation.argv,
        )
    except Diag2PreflightGateError as error:
        if error.reason in {
            FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
            FailureReasonCodeV2.REFERENCE_INVALID,
            FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
            FailureReasonCodeV2.SOURCE_POST,
        }:
            return finish(
                _diag2_postlaunch_setup_and_child_failure(
                    publication.staging_root,
                    refs,
                    DiagnosticChildMode.PREFLIGHT,
                    preflight,
                    reason=error.reason,
                    detail_sha256=error.detail_sha256,
                )
            )
        stage = {
            FailureReasonCodeV2.GPU_QUERY_FAILED: (
                FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE
            ),
            FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID: (
                FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            ),
            FailureReasonCodeV2.POLICY_SCHEMA_INVALID: (
                FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            ),
            FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: (
                FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            ),
            FailureReasonCodeV2.PRODUCER_DECODE_FAILED: (
                FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE
            ),
            FailureReasonCodeV2.CHILD_EXIT_NONZERO: (FailureStageV2.PREFLIGHT_CRASH),
            FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: (
                FailureStageV2.PREFLIGHT_MONITOR_FAILURE
            ),
            FailureReasonCodeV2.MONITOR_BINDING_FAILED: (
                FailureStageV2.PREFLIGHT_MONITOR_FAILURE
            ),
            FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED: (
                FailureStageV2.PREFLIGHT_RESOURCE_FAILURE
            ),
        }[error.reason]
        if error.offending_slot in {
            "preflight_runtime",
            "preflight_policy",
        }:
            refs[error.offending_slot] = None
        return finish(StructuredFailureV2(stage, error.reason, error.detail_sha256))
    preflight_authorized = True

    pre_source = post_source

    before_cold = _capture_diag2_supervisor_zero(
        environment,
        query_executable_sha256=query_executable_sha256,
    )
    refs["supervisor_before_cold"] = _publish_diag2_supervisor_zero(
        publication.staging_root,
        before_cold,
        stage="BEFORE_COLD",
    )
    try:
        validate_diag2_supervisor_zero_payload(
            load_canonical_json_bytes(
                refs["supervisor_before_cold"]
                .resolve_and_validate(publication.staging_root)
                .read_bytes()
            ),
            artifact_root=publication.staging_root,
            expected_stage="BEFORE_COLD",
        )
    except (OSError, TypeError, ValueError) as error:
        reason = (
            FailureReasonCodeV2.GPU_PARENT_PID_PRESENT
            if before_cold.matching_rows
            else FailureReasonCodeV2.GPU_QUERY_FAILED
        )
        return finish(
            _diag2_failure(
                FailureStageV2.GPU_ZERO_BEFORE_COLD_FAILURE,
                reason,
                f"{type(error).__name__}:{error}",
            )
        )
    cold_authorized = True

    cold_directory = publication.staging_root / "cold"
    cold_directory.mkdir(mode=0o755, exist_ok=False)
    cold_invocation = build_child_invocation(
        source,
        campaign_root=publication.staging_root,
        interpreter=executable,
        reference_root=campaign_reference,
        input_root=inputs,
        sample=SampleName.COLD,
        environment=environment,
        diagnostic_mode=DiagnosticChildMode.COLD,
        diag2=True,
    )
    cold = supervise_diag2_sample(
        SampleName.COLD,
        cold_invocation,
        mode=DiagnosticChildMode.COLD,
        gpu_uuid=gpu_uuid,
        physical_memory_bytes=memory_bytes,
        validate_producer=validate_diag3_producer_payload,
    )
    launched_children = ("preflight", "cold") if cold.launched else ("preflight",)
    cold_source_failure_detail: str | None = None
    try:
        post_source = _capture_source_identity_evidence(
            source, publication.staging_root
        )
    except (OSError, TypeError, ValueError) as error:
        cold_source_failure_detail = f"{type(error).__name__}:{error}"
        cold = replace(
            cold,
            selected_failure_reason=FailureReasonCodeV2.SOURCE_POST,
            raw_failure_reasons=(
                *cold.raw_failure_reasons,
                f"SOURCE_POST:{type(error).__name__}:{_sha256(str(error).encode())}",
            ),
            pre_source_identity=pre_source,
        )
    else:
        cold = replace(
            cold,
            pre_source_identity=pre_source,
            post_source_identity=post_source,
        )
    if cold.launched:
        bundle_publication = _cold_numerical_bundle_publication(cold_directory)
        if cold_source_failure_detail is not None:
            if bundle_publication.pending_root.is_dir():
                _quarantine_cold_numerical_bundle(bundle_publication)
        else:
            cold = _resolve_cold_numerical_bundle(cold_directory, cold)
        producer_schema = (
            str(
                cold.producer.get(
                    "schema_version", f"{DIAGNOSTIC_SCHEMA_VERSION}-producer"
                )
            )
            if cold.producer is not None
            else f"{DIAGNOSTIC_SCHEMA_VERSION}-producer"
        )
        cold_refs = _publish_diag2_supervision(
            publication.staging_root,
            cold_directory,
            cold,
            producer_schema=producer_schema,
        )
        for suffix in (
            "producer",
            "terminal",
            "process",
            "memory",
            "memory_samples",
        ):
            refs[f"cold_{suffix}"] = cold_refs[suffix]
        refs["cold_runtime"] = _diag2_existing_reference(
            publication.staging_root,
            "cold/runtime-evidence.json",
            RUNTIME_EVIDENCE_SCHEMA_VERSION,
        )
        refs["cold_policy"] = _diag2_existing_reference(
            publication.staging_root,
            "cold/policy.json",
            f"{DIAGNOSTIC_SCHEMA_VERSION}-policy",
        )
        if cold_source_failure_detail is not None:
            return finish(
                _diag2_postlaunch_setup_and_child_failure(
                    publication.staging_root,
                    refs,
                    DiagnosticChildMode.COLD,
                    cold,
                    reason=FailureReasonCodeV2.SOURCE_POST,
                    detail_sha256=_sha256(cold_source_failure_detail.encode("utf-8")),
                )
            )
        cold_setup_slots = {
            name: EvidenceSlot.present(reference)
            for name, reference in refs.items()
            if reference is not None
        }
        try:
            validate_diag2_setup_authorities(
                publication.staging_root,
                evidence_slots=cold_setup_slots,
            )
        except Diag2SetupGateError as error:
            postlaunch_reason = (
                FailureReasonCodeV2.SOURCE_POST
                if error.reason is FailureReasonCodeV2.SOURCE_PRE
                else error.reason
            )
            return finish(
                _diag2_postlaunch_setup_and_child_failure(
                    publication.staging_root,
                    refs,
                    DiagnosticChildMode.COLD,
                    cold,
                    reason=postlaunch_reason,
                    detail_sha256=error.detail_sha256,
                )
            )
        if (
            cold.producer is not None
            and cold.producer.get("execution_status") == "TRACE_NORMALIZATION_FAILED"
        ):
            for slot_name, producer_field in (
                ("cold_history", "history_evidence"),
                ("cold_terminal_numerical", "terminal_numerical_evidence"),
                ("cold_raw_trace", "raw_trace_evidence"),
            ):
                refs[slot_name] = _artifact_from_payload(
                    _mapping_field(cold.producer, producer_field)
                )
    failure = (
        _diag2_subordinate_child_failure(
            publication.staging_root,
            refs,
            DiagnosticChildMode.COLD,
            cold,
        )
        if cold.launched
        else _diag2_child_failure(DiagnosticChildMode.COLD, cold)
    )
    if failure is not None:
        return finish(failure)
    if cold.producer is None:
        raise ValueError("complete DIAG2 cold omits its producer")
    producer = cold.producer
    candidate_refs = dict(refs)
    ordered_cold_slots = (
        "cold_runtime",
        "cold_policy",
        "cold_history",
        "cold_terminal_numerical",
        "cold_raw_trace",
        "cold_trace_intervals",
        "execution",
    )
    for slot_name, producer_field in (
        ("cold_history", "history_evidence"),
        ("cold_terminal_numerical", "terminal_numerical_evidence"),
        ("cold_raw_trace", "raw_trace_evidence"),
        ("cold_trace_intervals", "trace_intervals_evidence"),
    ):
        try:
            candidate_refs[slot_name] = _artifact_from_payload(
                _mapping_field(producer, producer_field)
            )
        except (KeyError, TypeError, ValueError):
            classification = classify_diag3_cold_evidence(
                publication.staging_root,
                artifact_refs=candidate_refs,
            )
            for name in ordered_cold_slots:
                refs[name] = (
                    candidate_refs[name] if name in classification.typed_slots else None
                )
            if classification.failure is None:
                raise ValueError("cold descriptor failure was not classified")
            return finish(classification.failure)
    try:
        classification = classify_diag3_cold_evidence(
            publication.staging_root,
            artifact_refs=candidate_refs,
        )
        for name in ordered_cold_slots:
            refs[name] = (
                candidate_refs[name] if name in classification.typed_slots else None
            )
        if classification.offending_slot != "execution":
            if classification.failure is None:
                raise ValueError("cold evidence classification omitted its failure")
            return finish(classification.failure)
        candidate_refs["execution"] = _publish_diag2_execution(
            publication.staging_root,
            refs,
            cold=cold,
            cold_invocation=cold_invocation,
            interpreter=executable,
            physical_memory_bytes=memory_bytes,
        )
        classification = classify_diag3_cold_evidence(
            publication.staging_root,
            artifact_refs=candidate_refs,
        )
        refs["execution"] = (
            candidate_refs["execution"]
            if "execution" in classification.typed_slots
            else None
        )
        if classification.failure is not None:
            return finish(classification.failure)
    except (OSError, TypeError, ValueError) as error:
        return finish(
            _diag2_failure(
                FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE,
                FailureReasonCodeV2.NUMERICAL_SCHEMA_INVALID,
                f"{type(error).__name__}:{error}",
            )
        )
    return finish(None)


def run_campaign(
    campaign_root: Path,
    *,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    repo_root: Path | None = None,
) -> tuple[SupervisedSample, ...]:
    """Publish a snapshot and execute the frozen cold/conditional-warm schedule."""

    (
        repository,
        native_extension,
        reference_source,
        inputs,
        executable,
        gpu_uuid,
        memory_bytes,
    ) = _validate_parent_execution_policy(
        repo_root=repo_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
        environment=environment,
    )
    publication = prepare_execution_snapshot(
        campaign_root,
        repo_root=repository,
        native_extension_path=native_extension,
    )
    campaign_reference = copy_validated_reference(reference_source, campaign_root)

    def execute(sample: SampleName) -> SupervisedSample:
        sample_root = campaign_root / "samples" / sample.value
        sample_root.mkdir(parents=True, exist_ok=False)
        pre_source_identity = _capture_source_identity_evidence(
            publication, campaign_root
        )
        invocation = build_child_invocation(
            publication,
            campaign_root=campaign_root,
            interpreter=executable,
            reference_root=campaign_reference,
            input_root=inputs,
            sample=sample,
            environment=environment,
        )
        try:
            outcome = supervise_sample(
                sample,
                invocation,
                gpu_uuid=gpu_uuid,
                physical_memory_bytes=memory_bytes,
            )
        except Exception as error:  # noqa: BLE001 - retain and continue warm schedule.
            outcome = SupervisedSample(
                sample=sample,
                terminal_status=ChildTerminalStatus.CRASH,
                child_pid=0,
                child_start_time_ticks=0,
                process_seconds=0.0,
                producer={},
                memory=None,
                failure_reasons=(
                    f"SUPERVISOR:{type(error).__name__}:{_sha256(str(error).encode())}",
                ),
            )
        post_source_identity = _capture_source_identity_evidence(
            publication, campaign_root
        )
        outcome = replace(
            outcome,
            pre_source_identity=pre_source_identity,
            post_source_identity=post_source_identity,
        )
        producer_path = sample_root / "producer.json"
        memory_path = sample_root / "gpu-memory.json"
        terminal_path = sample_root / "terminal.json"
        _publish_canonical_json(producer_path, outcome.producer)
        if outcome.memory is not None:
            _publish_canonical_json(memory_path, outcome.memory)
        terminal = {
            "schema_version": "single-stage-neq-gntr1-terminal-v1",
            "sample": sample.value,
            "terminal_status": outcome.terminal_status.value,
            "child_pid": outcome.child_pid,
            "child_start_time_ticks": outcome.child_start_time_ticks,
            "process_seconds": outcome.process_seconds,
            "failure_reasons": list(outcome.failure_reasons),
        }
        _publish_canonical_json(terminal_path, terminal)
        return outcome

    reference_evidence = _artifact_ref(
        campaign_reference / REFERENCE_FILENAME,
        campaign_root,
        NATIVE_REFERENCE_SCHEMA_VERSION,
    )
    cold_reference: ReferenceReceipt | None = None
    cold_receipt: SampleReceipt | None = None
    cold_gate_failure: str | None = None

    def derive_reference(outcome: SupervisedSample) -> ReferenceReceipt:
        reference_inputs_value = outcome.producer.get("reference_inputs")
        if not isinstance(reference_inputs_value, dict):
            raise TypeError("worker reference inputs are absent")
        bootstrap_state = reference_inputs_value.get("bootstrap_state")
        constraint_scale = reference_inputs_value.get("constraint_inverse_scale")
        if not isinstance(bootstrap_state, list) or not isinstance(
            constraint_scale, list
        ):
            raise TypeError("worker reference inputs are malformed")
        return reference_receipt_from_artifact(
            artifact_root=campaign_reference,
            reference_evidence=reference_evidence,
            bootstrap_state=np.asarray(bootstrap_state, dtype=np.float64),
            constraint_inverse_scale=np.asarray(constraint_scale, dtype=np.float64),
        )

    def receipt_for(
        outcome: SupervisedSample, reference: ReferenceReceipt
    ) -> SampleReceipt:
        return build_sample_receipt(
            outcome,
            producer_reference=_artifact_ref(
                campaign_root / "samples" / outcome.sample.value / "producer.json",
                campaign_root,
                WORKER_SCHEMA_VERSION,
            ),
            publication=publication,
            campaign_root=campaign_root,
            reference=reference,
        )

    def cold_receipt_passes(outcome: SupervisedSample) -> bool:
        nonlocal cold_gate_failure, cold_receipt, cold_reference
        try:
            reference = derive_reference(outcome)
            receipt = receipt_for(outcome, reference)
            cold_reference = reference
            cold_receipt = receipt
            return bool(
                receipt.quality(reference) is SampleQuality.NATIVE_EQUIVALENT_QUALITY
                and receipt.provenance_and_resources_pass(reference)
            )
        except (KeyError, TypeError, ValueError) as error:
            cold_gate_failure = (
                f"COLD_RECEIPT_GATE:{type(error).__name__}:"
                f"{_sha256(str(error).encode())}"
            )
            return False

    outcomes = run_sample_schedule(execute, cold_receipt_passes)
    cold = outcomes[0]
    if any(not outcome.producer for outcome in outcomes):
        terminal = {
            "schema_version": "single-stage-neq-gntr1-campaign-terminal-v1",
            "route": ROUTE,
            "terminal_disposition": "NOT_PRODUCED",
            "failure_reasons": [
                reason for outcome in outcomes for reason in outcome.failure_reasons
            ],
            "samples": [sample.sample.value for sample in outcomes],
        }
        _publish_canonical_json(campaign_root / "campaign-terminal.json", terminal)
        _publish_campaign_artifact_manifest(campaign_root)
        _seal_campaign_tree(campaign_root)
        return outcomes
    if cold_gate_failure is not None:
        terminal = {
            "schema_version": "single-stage-neq-gntr1-campaign-terminal-v1",
            "route": ROUTE,
            "terminal_disposition": "NOT_PRODUCED",
            "failure_reasons": [cold_gate_failure],
            "samples": [sample.sample.value for sample in outcomes],
        }
        _publish_canonical_json(campaign_root / "campaign-terminal.json", terminal)
        _publish_campaign_artifact_manifest(campaign_root)
        _seal_campaign_tree(campaign_root)
        return outcomes
    reference = cold_reference if cold_reference is not None else derive_reference(cold)
    sample_receipts = (
        (cold_receipt,) if cold_receipt is not None else (receipt_for(cold, reference),)
    ) + tuple(receipt_for(outcome, reference) for outcome in outcomes[1:])
    campaign_receipt = CampaignReceipt(reference, sample_receipts)
    campaign_receipt.validate()
    _publish_canonical_json(
        campaign_root / CAMPAIGN_RECEIPT_FILENAME,
        campaign_payload(campaign_receipt),
    )
    _publish_campaign_artifact_manifest(campaign_root)
    _seal_campaign_tree(campaign_root)
    return outcomes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, default=Path(sys.executable))
    parser.add_argument("--sample", choices=[sample.value for sample in SAMPLE_ORDER])
    parser.add_argument("--snapshot-child", action="store_true")
    parser.add_argument(
        "--preflight-child", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--diagnostic-child",
        choices=[mode.value for mode in DiagnosticChildMode],
        help=argparse.SUPPRESS,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="compile and inspect the production GPU executable without solving",
    )
    mode.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="legacy diagnostic entry; successor execution requires an authority",
    )
    mode.add_argument(
        "--diagnostic-successor-authority",
        type=Path,
        help=(
            "validate one exact command-buffer recovery authority, then run one "
            "annotated preflight and at most one diagnostic cold"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.snapshot_child:
        if (
            arguments.preflight_only
            or arguments.diagnostic_only
            or arguments.diagnostic_successor_authority is not None
        ):
            raise ValueError("snapshot child cannot combine a supervisor mode")
        if arguments.sample is None:
            raise ValueError("snapshot child requires --sample")
        if arguments.preflight_child and arguments.diagnostic_child is not None:
            raise ValueError("snapshot child modes are mutually exclusive")
        if arguments.diagnostic_child == DiagnosticChildMode.PREFLIGHT.value:
            payload = run_snapshot_diagnostic_preflight_child(
                reference_root=arguments.reference.resolve(strict=True),
                input_root=arguments.input_root.resolve(strict=True),
            )
        elif arguments.diagnostic_child == DiagnosticChildMode.COLD.value:
            payload = run_snapshot_diagnostic_child(
                reference_root=arguments.reference.resolve(strict=True),
                input_root=arguments.input_root.resolve(strict=True),
            )
        elif arguments.preflight_child:
            payload = run_snapshot_preflight_child(
                reference_root=arguments.reference.resolve(strict=True),
                input_root=arguments.input_root.resolve(strict=True),
            )
        else:
            payload = run_snapshot_child(
                SampleName(arguments.sample),
                reference_root=arguments.reference.resolve(strict=True),
                input_root=arguments.input_root.resolve(strict=True),
            )
        print(canonical_json_bytes(payload).decode("utf-8"), end="")
        return 0
    if arguments.preflight_child or arguments.diagnostic_child is not None:
        raise ValueError("internal child mode requires --snapshot-child")
    if arguments.output is None:
        raise ValueError("campaign supervisor requires --output")
    if arguments.preflight_only:
        outcome = run_preflight(
            arguments.output,
            reference_root=arguments.reference,
            input_root=arguments.input_root,
            interpreter=arguments.interpreter,
            environment=os.environ,
        )
        summary = {
            "schema_version": "single-stage-neq-gntr1-preflight-summary-v1",
            "route": ROUTE,
            "mode": "LOWER_COMPILE_ONLY",
            "terminal_status": outcome.terminal_status.value,
            "campaign_authorized": False,
            "campaign_receipt_produced": False,
        }
        print(canonical_json_bytes(summary).decode("utf-8"), end="")
        return 0
    if arguments.diagnostic_successor_authority is not None:
        authority_path = arguments.diagnostic_successor_authority.absolute()
        is_diag5_authority_path = authority_path.as_posix().endswith(
            _DIAG5_BOOTSTRAP_AUTHORITY_RELATIVE_PATH
        )
        if is_diag5_authority_path:
            repository = authority_path.parents[1]
            with claim_diag5_successor_authority(
                authority_path,
                repository_root=repository,
                output_root=arguments.output,
            ) as successor_claim:
                revalidate_diag5_successor_authority(successor_claim)
                _release_diag5_bootstrap_bindings()
                result = run_diag5(
                    arguments.output,
                    reference_root=arguments.reference,
                    input_root=arguments.input_root,
                    interpreter=arguments.interpreter,
                    environment=os.environ,
                    successor_claim=successor_claim,
                    expected_source_snapshot=(
                        successor_claim.expected_gpu_source_snapshot_identity
                    ),
                    repo_root=repository,
                )
            print(canonical_json_bytes(result).decode("utf-8"), end="")
            return 0
        is_diag4_authority_path = authority_path.as_posix().endswith(
            _DIAG4_BOOTSTRAP_AUTHORITY_RELATIVE_PATH
        )
        authority_payload = (
            load_canonical_json_bytes(authority_path.read_bytes())
            if is_diag4_authority_path and authority_path.is_file()
            else None
        )
        if (
            isinstance(authority_payload, dict)
            and authority_payload.get("schema_version")
            == _DIAG4_BOOTSTRAP_AUTHORITY_SCHEMA
        ):
            repository = authority_path.parents[1]
            with claim_diag4_successor_authority(
                authority_path,
                repository_root=repository,
                output_root=arguments.output,
                reference_root=arguments.reference,
                input_root=arguments.input_root,
                interpreter=arguments.interpreter,
            ) as successor_claim:
                result = run_diag4(
                    arguments.output,
                    reference_root=arguments.reference,
                    input_root=arguments.input_root,
                    interpreter=arguments.interpreter,
                    environment=os.environ,
                    successor_claim=successor_claim,
                    repo_root=repository,
                )
            print(canonical_json_bytes(result).decode("utf-8"), end="")
            return 0
        with claim_successor_authority(
            authority_path,
            repository_root=_runner_root,
            output_root=arguments.output,
            reference_root=arguments.reference,
            input_root=arguments.input_root,
            interpreter=arguments.interpreter,
        ) as successor_claim:
            result = run_diag2(
                arguments.output,
                reference_root=arguments.reference,
                input_root=arguments.input_root,
                interpreter=arguments.interpreter,
                environment=os.environ,
                successor_claim=successor_claim,
            )
        print(canonical_json_bytes(result).decode("utf-8"), end="")
        return 0
    if arguments.diagnostic_only:
        raise ValueError(
            "legacy --diagnostic-only is not authorized; "
            "use --diagnostic-successor-authority"
        )
    samples = run_campaign(
        arguments.output,
        reference_root=arguments.reference,
        input_root=arguments.input_root,
        interpreter=arguments.interpreter,
        environment=os.environ,
    )
    summary = {
        "schema_version": "single-stage-neq-gntr1-supervisor-summary-v1",
        "route": ROUTE,
        "samples": [sample.sample.value for sample in samples],
        "terminal_statuses": [sample.terminal_status.value for sample in samples],
        "campaign_receipt_produced": (
            arguments.output / CAMPAIGN_RECEIPT_FILENAME
        ).is_file(),
    }
    print(canonical_json_bytes(summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "PROCESS_TIMEOUT_SECONDS",
    "SAMPLE_ORDER",
    "SOLVE_TIMEOUT_SECONDS",
    "ChildTerminalStatus",
    "SnapshotChildInvocation",
    "SupervisedSample",
    "build_child_invocation",
    "build_sample_receipt",
    "execute_timed_loop",
    "nonpromoting_sample_draft_payload",
    "prepare_execution_snapshot",
    "run_campaign",
    "run_preflight",
    "run_sample_schedule",
    "run_snapshot_child",
    "run_snapshot_preflight_child",
    "supervise_sample",
)
