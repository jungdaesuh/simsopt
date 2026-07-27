"""Repository, source, runtime, and memory provenance for parity receipts."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ExecutedSource:
    path: str
    sha256: str
    git_blob_id: str | None


@dataclass(frozen=True)
class DeviceMetadata:
    id: int
    platform: str
    device_kind: str
    process_index: int


@dataclass(frozen=True)
class RepositoryState:
    repository_commit: str
    repository_dirty: bool
    tracked_diff_sha256: str
    untracked_files: tuple[str, ...]


@dataclass(frozen=True)
class LaneProvenance:
    repository_commit: str
    repository_dirty: bool
    tracked_diff_sha256: str
    untracked_files: tuple[str, ...]
    executed_sources: tuple[ExecutedSource, ...]
    python_version: str
    jax_version: str | None
    simsopt_version: str
    simsopt_version_commit: str | None
    simsopt_version_checkout_compatible: bool | None
    lane_environment_policy: Mapping[str, str]
    jax_effective_transfer_guards: Mapping[str, str]
    devices: tuple[DeviceMetadata, ...]
    host_peak_rss_bytes: int
    host_peak_rss_method: str
    device_memory_peak_bytes: int | None
    device_memory_status: str
    memory_measurement_scope: str
    steady_state_memory_measured: bool
    measurement_synchronization: str
    simsoptpp_path: str | None
    simsoptpp_sha256: str | None
    simsoptpp_version: str | None
    simsoptpp_build_commit: str | None
    simsoptpp_checkout_compatible: bool | None
    authoritative: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lane_environment_policy",
            MappingProxyType(dict(self.lane_environment_policy)),
        )
        object.__setattr__(
            self,
            "jax_effective_transfer_guards",
            MappingProxyType(dict(self.jax_effective_transfer_guards)),
        )


_LANE_ENVIRONMENT_KEYS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_PRECISION",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "CUDA_VISIBLE_DEVICES",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)

REQUIRED_PROVENANCE_SOURCE_PATHS = (
    "examples/jax/manifest.json",
    "examples/jax/parity_manifest.json",
    "examples/jax/run_parity.py",
    "examples/jax/parity/child.py",
)


def _git(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generated_version_matches_checkout(
    repository_commit: str, generated_commit: str | None
) -> bool:
    """Return whether setuptools-scm's generated commit names this checkout."""
    if generated_commit is None:
        return False
    abbreviated = generated_commit.removeprefix("g")
    return len(abbreviated) >= 7 and repository_commit.startswith(abbreviated)


def collect_repository_state(repo_root: Path) -> RepositoryState:
    """Return a deterministic snapshot of the current Git worktree state."""
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    untracked = tuple(
        sorted(
            record[3:].decode("utf-8", errors="surrogateescape")
            for record in status.split(b"\0")
            if record.startswith(b"?? ")
        )
    )
    return RepositoryState(
        repository_commit=_git(repo_root, "rev-parse", "HEAD").decode("ascii").strip(),
        repository_dirty=bool(status),
        tracked_diff_sha256=_sha256_bytes(
            _git(repo_root, "diff", "--binary", "HEAD", "--")
        ),
        untracked_files=untracked,
    )


def _tracked_blob_ids(repo_root: Path) -> dict[str, str]:
    records = _git(repo_root, "ls-files", "-s", "-z").split(b"\0")
    blobs: dict[str, str] = {}
    for record in records:
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", maxsplit=1)
        _, blob_id, _ = metadata.decode("ascii").split()
        blobs[encoded_path.decode("utf-8", errors="surrogateescape")] = blob_id
    return blobs


def _module_source_path(module_file: str) -> Path:
    path = Path(module_file)
    if path.suffix == ".pyc":
        return Path(importlib.util.source_from_cache(str(path)))
    return path


def collect_executed_sources(repo_root: Path) -> tuple[ExecutedSource, ...]:
    """Hash every loaded in-checkout Python module after lane execution."""
    tracked_blobs = _tracked_blob_ids(repo_root)
    resolved_root = repo_root.resolve()
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        source_path = _module_source_path(module_file).resolve()
        if source_path.is_file() and source_path.is_relative_to(resolved_root):
            paths.add(source_path)
    records = []
    for source_path in sorted(paths):
        relative = source_path.relative_to(resolved_root).as_posix()
        records.append(
            ExecutedSource(
                path=relative,
                sha256=_sha256_file(source_path),
                git_blob_id=tracked_blobs.get(relative),
            )
        )
    return tuple(records)


def collect_explicit_sources(
    repo_root: Path, relative_paths: tuple[str, ...]
) -> tuple[ExecutedSource, ...]:
    """Hash declarative and runner inputs that need not be imported by a child."""
    tracked_blobs = _tracked_blob_ids(repo_root)
    records = []
    for relative in sorted(relative_paths):
        source_path = repo_root / relative
        records.append(
            ExecutedSource(
                path=relative,
                sha256=_sha256_file(source_path),
                git_blob_id=tracked_blobs.get(relative),
            )
        )
    return tuple(records)


def _merge_sources(
    *source_groups: tuple[ExecutedSource, ...],
) -> tuple[ExecutedSource, ...]:
    by_path: dict[str, ExecutedSource] = {}
    for source in (item for group in source_groups for item in group):
        previous = by_path.get(source.path)
        if previous is not None and previous != source:
            raise ValueError(f"conflicting source provenance for {source.path}")
        by_path[source.path] = source
    return tuple(by_path[path] for path in sorted(by_path))


def validate_sources_current(
    repo_root: Path, sources: tuple[ExecutedSource, ...]
) -> None:
    """Reject a receipt whose source bytes no longer match the checkout."""
    resolved_root = repo_root.resolve()
    seen: set[str] = set()
    for source in sources:
        if source.path in seen:
            raise ValueError(f"duplicate source provenance path: {source.path}")
        seen.add(source.path)
        resolved = (resolved_root / source.path).resolve()
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            raise ValueError(f"invalid source provenance path: {source.path}")
        if _sha256_file(resolved) != source.sha256:
            raise ValueError(f"executed source changed: {source.path}")


def _device_metadata() -> tuple[
    tuple[DeviceMetadata, ...], int | None, str, str | None, dict[str, str]
]:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return (), None, "unavailable: JAX not loaded", None, {}
    devices = tuple(
        DeviceMetadata(
            id=int(device.id),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
            process_index=int(device.process_index),
        )
        for device in jax_module.devices()
    )
    peak_values: list[int] = []
    for device in jax_module.devices():
        statistics = device.memory_stats()
        if isinstance(statistics, dict):
            value = statistics.get("peak_bytes_in_use")
            if isinstance(value, int):
                peak_values.append(value)
    peak = max(peak_values) if peak_values else None
    status = (
        "jax device.memory_stats peak_bytes_in_use"
        if peak is not None
        else ("unavailable: backend exposes no validated peak_bytes_in_use counter")
    )
    guards = {
        direction: str(getattr(jax_module.config, f"jax_transfer_guard_{direction}"))
        for direction in ("device_to_device", "device_to_host", "host_to_device")
    }
    return devices, peak, status, str(jax_module.__version__), guards


def collect_lane_provenance(repo_root: Path) -> LaneProvenance:
    """Collect one post-execution, non-secret lane provenance receipt."""
    repository = collect_repository_state(repo_root)
    devices, device_peak, device_status, jax_version, effective_guards = (
        _device_metadata()
    )
    simsopt_module = sys.modules.get("simsopt")
    simsopt_version = str(getattr(simsopt_module, "__version__", "unknown"))
    version_module = sys.modules.get("simsopt._version")
    version_commit_value = getattr(version_module, "commit_id", None)
    version_commit = (
        version_commit_value if isinstance(version_commit_value, str) else None
    )
    version_compatible = (
        generated_version_matches_checkout(repository.repository_commit, version_commit)
        if version_module is not None
        else None
    )
    simsoptpp_module = sys.modules.get("simsoptpp")
    simsoptpp_path = None
    simsoptpp_sha256 = None
    simsoptpp_version = None
    build_commit = None
    compatible = None
    if simsoptpp_module is not None:
        binary_path = Path(str(simsoptpp_module.__file__)).resolve()
        simsoptpp_path = str(binary_path)
        simsoptpp_sha256 = _sha256_file(binary_path)
        simsoptpp_version = str(getattr(simsoptpp_module, "__version__", "unknown"))
        build_commit = os.environ.get("SIMSOPT_PARITY_SIMSOPTPP_BUILD_COMMIT")
        compatible = build_commit == repository.repository_commit
    executed_sources = _merge_sources(
        collect_executed_sources(repo_root),
        collect_explicit_sources(repo_root, REQUIRED_PROVENANCE_SOURCE_PATHS),
    )
    sources_authoritative = all(
        source.git_blob_id is not None
        or (source.path == "src/simsopt/_version.py" and version_compatible is True)
        for source in executed_sources
    )
    authoritative = (
        not repository.repository_dirty
        and sources_authoritative
        and (simsoptpp_module is None or compatible is True)
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_peak_rss_bytes = int(peak_rss) * 1024
    return LaneProvenance(
        repository_commit=repository.repository_commit,
        repository_dirty=repository.repository_dirty,
        tracked_diff_sha256=repository.tracked_diff_sha256,
        untracked_files=repository.untracked_files,
        executed_sources=executed_sources,
        python_version=sys.version.split()[0],
        jax_version=jax_version,
        simsopt_version=simsopt_version,
        simsopt_version_commit=version_commit,
        simsopt_version_checkout_compatible=version_compatible,
        lane_environment_policy={
            key: os.environ[key] for key in _LANE_ENVIRONMENT_KEYS if key in os.environ
        },
        jax_effective_transfer_guards=effective_guards,
        devices=devices,
        host_peak_rss_bytes=host_peak_rss_bytes,
        host_peak_rss_method="child getrusage(RUSAGE_SELF).ru_maxrss fallback",
        device_memory_peak_bytes=device_peak,
        device_memory_status=device_status,
        memory_measurement_scope=(
            "combined import, compile/warmup, and one bounded execution"
        ),
        steady_state_memory_measured=False,
        measurement_synchronization=(
            "jax.block_until_ready over published observation values"
            if jax_version is not None
            else "native synchronous execution"
        ),
        simsoptpp_path=simsoptpp_path,
        simsoptpp_sha256=simsoptpp_sha256,
        simsoptpp_version=simsoptpp_version,
        simsoptpp_build_commit=build_commit,
        simsoptpp_checkout_compatible=compatible,
        authoritative=authoritative,
    )


def lane_provenance_payload(provenance: LaneProvenance) -> dict[str, object]:
    """Return the canonical JSON object for one lane provenance receipt."""
    return {
        "repository_commit": provenance.repository_commit,
        "repository_dirty": provenance.repository_dirty,
        "tracked_diff_sha256": provenance.tracked_diff_sha256,
        "untracked_files": list(provenance.untracked_files),
        "executed_sources": [
            dataclasses.asdict(source) for source in provenance.executed_sources
        ],
        "python_version": provenance.python_version,
        "jax_version": provenance.jax_version,
        "simsopt_version": provenance.simsopt_version,
        "simsopt_version_commit": provenance.simsopt_version_commit,
        "simsopt_version_checkout_compatible": (
            provenance.simsopt_version_checkout_compatible
        ),
        "lane_environment_policy": dict(provenance.lane_environment_policy),
        "jax_effective_transfer_guards": dict(provenance.jax_effective_transfer_guards),
        "devices": [dataclasses.asdict(device) for device in provenance.devices],
        "host_peak_rss_bytes": provenance.host_peak_rss_bytes,
        "host_peak_rss_method": provenance.host_peak_rss_method,
        "device_memory_peak_bytes": provenance.device_memory_peak_bytes,
        "device_memory_status": provenance.device_memory_status,
        "memory_measurement_scope": provenance.memory_measurement_scope,
        "steady_state_memory_measured": provenance.steady_state_memory_measured,
        "measurement_synchronization": provenance.measurement_synchronization,
        "simsoptpp_path": provenance.simsoptpp_path,
        "simsoptpp_sha256": provenance.simsoptpp_sha256,
        "simsoptpp_version": provenance.simsoptpp_version,
        "simsoptpp_build_commit": provenance.simsoptpp_build_commit,
        "simsoptpp_checkout_compatible": provenance.simsoptpp_checkout_compatible,
        "authoritative": provenance.authoritative,
    }


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"provenance field {field} must be a string or null")
    return value


def _required_string(value: object, field: str) -> str:
    result = _optional_string(value, field)
    if result is None:
        raise ValueError(f"provenance field {field} must be a non-empty string")
    return result


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"provenance field {field} must be nonnegative or null")
    return value


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"provenance field {field} must be boolean or null")


def lane_provenance_from_payload(value: object) -> LaneProvenance:
    """Validate and reconstruct one serialized lane provenance receipt."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("provenance must be a JSON object")
    required_fields = frozenset(
        {
            "repository_commit",
            "repository_dirty",
            "tracked_diff_sha256",
            "untracked_files",
            "executed_sources",
            "python_version",
            "jax_version",
            "simsopt_version",
            "simsopt_version_commit",
            "simsopt_version_checkout_compatible",
            "lane_environment_policy",
            "jax_effective_transfer_guards",
            "devices",
            "host_peak_rss_bytes",
            "host_peak_rss_method",
            "device_memory_peak_bytes",
            "device_memory_status",
            "memory_measurement_scope",
            "steady_state_memory_measured",
            "measurement_synchronization",
            "simsoptpp_path",
            "simsoptpp_sha256",
            "simsoptpp_version",
            "simsoptpp_build_commit",
            "simsoptpp_checkout_compatible",
            "authoritative",
        }
    )
    if set(value) != required_fields:
        raise ValueError("provenance has invalid fields")
    untracked = value["untracked_files"]
    if not isinstance(untracked, list) or not all(
        isinstance(item, str) and item for item in untracked
    ):
        raise ValueError("provenance untracked_files must be a string array")
    sources_value = value["executed_sources"]
    if not isinstance(sources_value, list) or not sources_value:
        raise ValueError("provenance executed_sources must be non-empty")
    sources: list[ExecutedSource] = []
    for source in sources_value:
        if not isinstance(source, dict) or set(source) != {
            "path",
            "sha256",
            "git_blob_id",
        }:
            raise ValueError("invalid executed source provenance")
        sources.append(
            ExecutedSource(
                path=_required_string(source["path"], "executed_sources.path"),
                sha256=_required_string(source["sha256"], "executed_sources.sha256"),
                git_blob_id=_optional_string(
                    source["git_blob_id"], "executed_sources.git_blob_id"
                ),
            )
        )
    policy = value["lane_environment_policy"]
    if not isinstance(policy, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in policy.items()
    ):
        raise ValueError("provenance lane environment policy must be string-valued")
    effective_guards = value["jax_effective_transfer_guards"]
    if not isinstance(effective_guards, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in effective_guards.items()
    ):
        raise ValueError("provenance effective transfer guards must be string-valued")
    devices_value = value["devices"]
    if not isinstance(devices_value, list):
        raise TypeError("provenance devices must be an array")
    devices: list[DeviceMetadata] = []
    for device in devices_value:
        if not isinstance(device, dict) or set(device) != {
            "id",
            "platform",
            "device_kind",
            "process_index",
        }:
            raise ValueError("invalid device provenance")
        device_id = _optional_int(device["id"], "devices.id")
        process_index = _optional_int(device["process_index"], "devices.process_index")
        if device_id is None or process_index is None:
            raise ValueError("device identifiers must not be null")
        devices.append(
            DeviceMetadata(
                id=device_id,
                platform=_required_string(device["platform"], "devices.platform"),
                device_kind=_required_string(
                    device["device_kind"], "devices.device_kind"
                ),
                process_index=process_index,
            )
        )
    repository_dirty = value["repository_dirty"]
    authoritative = value["authoritative"]
    steady_state_memory_measured = value["steady_state_memory_measured"]
    if (
        not isinstance(repository_dirty, bool)
        or not isinstance(authoritative, bool)
        or not isinstance(steady_state_memory_measured, bool)
    ):
        raise TypeError(
            "provenance dirty, authoritative, and memory-scope fields must be boolean"
        )
    host_peak = _optional_int(value["host_peak_rss_bytes"], "host_peak_rss_bytes")
    if host_peak is None:
        raise ValueError("host_peak_rss_bytes must not be null")
    return LaneProvenance(
        repository_commit=_required_string(
            value["repository_commit"], "repository_commit"
        ),
        repository_dirty=repository_dirty,
        tracked_diff_sha256=_required_string(
            value["tracked_diff_sha256"], "tracked_diff_sha256"
        ),
        untracked_files=tuple(untracked),
        executed_sources=tuple(sources),
        python_version=_required_string(value["python_version"], "python_version"),
        jax_version=_optional_string(value["jax_version"], "jax_version"),
        simsopt_version=_required_string(value["simsopt_version"], "simsopt_version"),
        simsopt_version_commit=_optional_string(
            value["simsopt_version_commit"], "simsopt_version_commit"
        ),
        simsopt_version_checkout_compatible=_optional_bool(
            value["simsopt_version_checkout_compatible"],
            "simsopt_version_checkout_compatible",
        ),
        lane_environment_policy=policy,
        jax_effective_transfer_guards=effective_guards,
        devices=tuple(devices),
        host_peak_rss_bytes=host_peak,
        host_peak_rss_method=_required_string(
            value["host_peak_rss_method"], "host_peak_rss_method"
        ),
        device_memory_peak_bytes=_optional_int(
            value["device_memory_peak_bytes"], "device_memory_peak_bytes"
        ),
        device_memory_status=_required_string(
            value["device_memory_status"], "device_memory_status"
        ),
        memory_measurement_scope=_required_string(
            value["memory_measurement_scope"], "memory_measurement_scope"
        ),
        steady_state_memory_measured=steady_state_memory_measured,
        measurement_synchronization=_required_string(
            value["measurement_synchronization"], "measurement_synchronization"
        ),
        simsoptpp_path=_optional_string(value["simsoptpp_path"], "simsoptpp_path"),
        simsoptpp_sha256=_optional_string(
            value["simsoptpp_sha256"], "simsoptpp_sha256"
        ),
        simsoptpp_version=_optional_string(
            value["simsoptpp_version"], "simsoptpp_version"
        ),
        simsoptpp_build_commit=_optional_string(
            value["simsoptpp_build_commit"], "simsoptpp_build_commit"
        ),
        simsoptpp_checkout_compatible=_optional_bool(
            value["simsoptpp_checkout_compatible"],
            "simsoptpp_checkout_compatible",
        ),
        authoritative=authoritative,
    )
