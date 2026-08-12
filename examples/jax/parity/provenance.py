"""Repository, source, runtime, and memory provenance for parity receipts."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, Mapping

SNAPSHOT_LANE_IDENTITY_SCHEMA_ID: Final = (
    "single-stage-compute-graph-lane-snapshot-identity-v2"
)
SnapshotProfileId = Literal["native_cpu", "jax_gpu_fast", "jax_gpu_optax"]


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


@dataclass(frozen=True, slots=True)
class SnapshotLaneIdentity:
    """Pre-execution identity whose referenced evidence is byte-validated."""

    profile_id: SnapshotProfileId
    lane: str
    backend_mode: str
    driver: str
    execution_platform: str
    runtime_identity_sha256: str
    source_sha256: str
    gpu_uuid: str
    snapshot_root: Path
    repository_commit: str
    repository_dirty: bool
    tracked_diff_sha256: str
    untracked_files: tuple[str, ...]
    manifest_entries: Mapping[str, str]
    native_extension_path: Path
    native_extension_sha256: str
    interpreter_path: Path
    python_version: str
    jax_version: str
    jaxlib_version: str
    bound_environment: Mapping[str, str]
    static_environment: Mapping[str, str]


_LANE_ENVIRONMENT_KEYS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_EXACT_ADJOINT_DENSE_LU",
    "SIMSOPT_PRECISION",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "CUDA_VISIBLE_DEVICES",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def normalize_snapshot_lane_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Return the exact allowlisted static policy observed by a snapshot child."""

    return {
        key: environment[key] for key in _LANE_ENVIRONMENT_KEYS if key in environment
    }


def validate_snapshot_lane_environment(
    identity: SnapshotLaneIdentity, environment: Mapping[str, str]
) -> None:
    """Reject any missing, changed, or additional allowlisted lane selector."""

    if normalize_snapshot_lane_environment(environment) != dict(
        identity.static_environment
    ):
        raise ValueError("snapshot static runtime environment changed")


REQUIRED_PROVENANCE_SOURCE_PATHS = (
    "examples/jax/manifest.json",
    "examples/jax/parity_manifest.json",
    "examples/jax/run_parity.py",
    "examples/jax/parity/child.py",
)

_SNAPSHOT_EVIDENCE_NAMES: Final = (
    "publication",
    "manifest",
    "import_attestation",
    "runner_spec",
    "runtime_provenance",
    "device_probe",
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


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _lower_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _json_object(path: Path, context: str) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        payload = path.read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite constant {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    if payload != _canonical_json_bytes(value):
        raise ValueError(f"{context} bytes are not canonical")
    return value


def _evidence_reference(
    evidence: Mapping[str, object], name: str
) -> tuple[Path, str, dict[str, object]]:
    reference = evidence.get(name)
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"snapshot identity evidence {name} reference is invalid")
    path_value = reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"snapshot identity evidence {name} path is invalid")
    path = Path(path_value)
    expected_sha256 = _lower_sha256(
        reference.get("sha256"), f"snapshot identity evidence {name} SHA"
    )
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"snapshot identity evidence {name} path is unavailable")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"snapshot identity evidence {name} bytes changed")
    return path, expected_sha256, _json_object(path, f"snapshot identity {name}")


def load_snapshot_lane_identity(path: Path) -> SnapshotLaneIdentity:
    """Load and cross-check one static snapshot identity before lane execution."""

    document = _json_object(path, "snapshot lane identity")
    expected_fields = {
        "schema_id",
        "profile_id",
        "lane",
        "backend_mode",
        "driver",
        "execution_platform",
        "runtime_identity_sha256",
        "source_sha256",
        "gpu_uuid",
        "snapshot_root",
        "static_environment",
        "evidence",
    }
    if (
        set(document) != expected_fields
        or document.get("schema_id") != SNAPSHOT_LANE_IDENTITY_SCHEMA_ID
    ):
        raise ValueError("snapshot lane identity schema is invalid")
    profile_value = document.get("profile_id")
    if profile_value not in ("native_cpu", "jax_gpu_fast", "jax_gpu_optax"):
        raise ValueError("snapshot lane identity profile is invalid")
    profile_id: SnapshotProfileId = profile_value
    expected_lane = "native-cpu" if profile_id == "native_cpu" else "jax-gpu"
    expected_platform = "cpu" if profile_id == "native_cpu" else "gpu"
    lane = document.get("lane")
    execution_platform = document.get("execution_platform")
    if lane != expected_lane or execution_platform != expected_platform:
        raise ValueError("snapshot lane identity profile route is invalid")
    for field in ("backend_mode", "driver", "gpu_uuid", "snapshot_root"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ValueError(f"snapshot lane identity {field} is invalid")
    runtime_sha256 = _lower_sha256(
        document.get("runtime_identity_sha256"), "snapshot runtime identity"
    )
    source_sha256 = _lower_sha256(
        document.get("source_sha256"), "snapshot source identity"
    )
    snapshot_root = Path(str(document["snapshot_root"]))
    if not snapshot_root.is_absolute() or not snapshot_root.is_dir():
        raise ValueError("snapshot lane identity root is unavailable")
    evidence = document.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(_SNAPSHOT_EVIDENCE_NAMES):
        raise ValueError("snapshot lane identity evidence schema is invalid")
    loaded = {
        name: _evidence_reference(evidence, name) for name in _SNAPSHOT_EVIDENCE_NAMES
    }
    manifest_path, manifest_sha256, manifest = loaded["manifest"]
    if manifest_path != snapshot_root / "phase0-source-manifest.json":
        raise ValueError("snapshot lane identity manifest path is invalid")
    entries_value = manifest.get("entries")
    if (
        manifest.get("schema_id") != "single-stage-compute-graph-source-manifest-v1"
        or not isinstance(entries_value, list)
        or not entries_value
    ):
        raise ValueError("snapshot lane identity manifest schema is invalid")
    manifest_entries: dict[str, str] = {}
    native_entries: list[tuple[Path, str]] = []
    for raw_entry in entries_value:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "role",
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("snapshot lane identity manifest entry is invalid")
        relative = raw_entry.get("relative_path")
        size = raw_entry.get("size_bytes")
        digest = _lower_sha256(raw_entry.get("sha256"), "manifest entry SHA")
        relative_path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or "." in relative_path.parts
            or str(relative_path) != relative
            or isinstance(size, bool)
            or not isinstance(size, int)
            or raw_entry.get("role")
            not in {
                "execution_source",
                "configuration",
                "benchmark",
                "test",
                "native_extension",
            }
        ):
            raise ValueError("snapshot lane identity manifest entry is invalid")
        source_path = (snapshot_root / relative).resolve()
        if (
            not source_path.is_relative_to(snapshot_root.resolve())
            or not source_path.is_file()
            or source_path.stat().st_size != size
            or _sha256_file(source_path) != digest
            or relative in manifest_entries
        ):
            raise ValueError("snapshot lane identity manifest bytes are stale")
        manifest_entries[relative] = digest
        if raw_entry.get("role") == "native_extension":
            native_entries.append((source_path, digest))
    if len(native_entries) != 1:
        raise ValueError("snapshot lane identity must bind one native extension")
    publication = loaded["publication"][2]
    publication_worktree = publication.get("worktree")
    if (
        set(publication)
        != {
            "schema_id",
            "repository_root",
            "snapshot_root",
            "snapshot_manifest_sha256",
            "cross_host_source_sha256",
            "native_extension",
            "worktree",
        }
        or publication.get("schema_id")
        != "single-stage-compute-graph-snapshot-publication-v1"
        or publication.get("snapshot_root") != str(snapshot_root)
        or publication.get("snapshot_manifest_sha256") != manifest_sha256
        or not isinstance(publication_worktree, dict)
        or publication_worktree.get("source_state_sha256") != source_sha256
    ):
        raise ValueError("snapshot lane identity publication binding is invalid")
    runtime = loaded["runtime_provenance"][2]
    runner_spec = loaded["runner_spec"][2]
    if (
        set(runner_spec)
        != {
            "schema_id",
            "lane_id",
            "warm_sample_count",
            "output_root",
            "input_root",
            "candidate_path",
            "native_reference_path",
            "provenance",
            "receipt_template",
        }
        or runner_spec.get("schema_id")
        != "single-stage-compute-graph-c0-runner-spec-v3"
        or runner_spec.get("provenance") != runtime
    ):
        raise ValueError("snapshot lane identity runner runtime binding is invalid")
    runtime_allocation = runtime.get("allocation")
    runtime_fields = runtime.get("runtime")
    if (
        runtime.get("source_state_sha256") != source_sha256
        or not isinstance(runtime_allocation, dict)
        or runtime_allocation.get("gpu_uuid") != document.get("gpu_uuid")
        or not isinstance(runtime_fields, dict)
    ):
        raise ValueError("snapshot lane identity runtime/device binding is invalid")
    device_probe = loaded["device_probe"][2]
    probe_gpu = device_probe.get("gpu")
    probe_native = device_probe.get("native_binary")
    if (
        set(device_probe)
        != {
            "schema_id",
            "lane_id",
            "source_state_sha256",
            "runtime_identity_sha256",
            "qualification_sha256",
            "gpu",
            "native_binary",
        }
        or device_probe.get("schema_id") != "single-stage-compute-graph-device-probe-v1"
        or device_probe.get("source_state_sha256") != source_sha256
        or device_probe.get("runtime_identity_sha256") != runtime_sha256
        or not isinstance(probe_gpu, dict)
        or set(probe_gpu) != {"uuid", "name", "memory_bytes"}
        or probe_gpu.get("uuid") != document.get("gpu_uuid")
        or not isinstance(probe_native, dict)
        or set(probe_native) != {"path", "sha256"}
    ):
        raise ValueError("snapshot lane identity device probe binding is invalid")
    attestation = loaded["import_attestation"][2]
    if (
        attestation.get("schema_id")
        != "single-stage-compute-graph-import-attestation-v1"
        or attestation.get("state") != "pass"
        or attestation.get("snapshot_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("snapshot lane identity import binding is invalid")
    repository_commit = publication_worktree.get("repository_commit")
    tracked_diff = publication_worktree.get("tracked_diff_sha256")
    status = publication_worktree.get("git_status_short")
    if (
        not isinstance(repository_commit, str)
        or not repository_commit
        or not isinstance(tracked_diff, str)
        or not tracked_diff
        or not isinstance(status, list)
        or not all(isinstance(item, str) for item in status)
    ):
        raise ValueError("snapshot lane identity repository binding is invalid")
    python_version = runtime_fields.get("python_version")
    jax_version = runtime_fields.get("jax_version")
    jaxlib_version = runtime_fields.get("jaxlib_version")
    interpreter_value = runtime.get("interpreter_path")
    runtime_environment = runtime.get("environment")
    static_environment = document.get("static_environment")
    if (
        not isinstance(python_version, str)
        or not isinstance(jax_version, str)
        or not isinstance(jaxlib_version, str)
        or not isinstance(interpreter_value, str)
        or not Path(interpreter_value).is_absolute()
        or not isinstance(runtime_environment, dict)
        or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in runtime_environment.items()
        )
        or not isinstance(static_environment, dict)
        or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in static_environment.items()
        )
        or normalize_snapshot_lane_environment(static_environment) != static_environment
    ):
        raise ValueError("snapshot lane identity runtime versions are invalid")
    return SnapshotLaneIdentity(
        profile_id=profile_id,
        lane=expected_lane,
        backend_mode=str(document["backend_mode"]),
        driver=str(document["driver"]),
        execution_platform=expected_platform,
        runtime_identity_sha256=runtime_sha256,
        source_sha256=source_sha256,
        gpu_uuid=str(document["gpu_uuid"]),
        snapshot_root=snapshot_root,
        repository_commit=repository_commit,
        repository_dirty=bool(status),
        tracked_diff_sha256=tracked_diff,
        untracked_files=tuple(item[3:] for item in status if item.startswith("?? ")),
        manifest_entries=MappingProxyType(manifest_entries),
        native_extension_path=native_entries[0][0],
        native_extension_sha256=native_entries[0][1],
        interpreter_path=Path(interpreter_value),
        python_version=python_version,
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
        bound_environment=MappingProxyType(dict(runtime_environment)),
        static_environment=MappingProxyType(dict(static_environment)),
    )


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


def collect_lane_provenance(
    repo_root: Path, *, measurement_synchronization: str
) -> LaneProvenance:
    """Collect a receipt using the synchronization recorded by its producer."""
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
        measurement_synchronization=measurement_synchronization,
        simsoptpp_path=simsoptpp_path,
        simsoptpp_sha256=simsoptpp_sha256,
        simsoptpp_version=simsoptpp_version,
        simsoptpp_build_commit=build_commit,
        simsoptpp_checkout_compatible=compatible,
        authoritative=authoritative,
    )


def _snapshot_executed_sources(
    identity: SnapshotLaneIdentity,
) -> tuple[ExecutedSource, ...]:
    resolved_root = identity.snapshot_root.resolve()
    relative_paths = set(REQUIRED_PROVENANCE_SOURCE_PATHS)
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        source_path = _module_source_path(module_file).resolve()
        project_module = (
            module_name == "simsoptpp"
            or module_name.startswith(
                ("simsopt.", "simsopt_jax.", "simsopt_jax_adapters.", "examples.jax.")
            )
            or module_name in {"simsopt", "simsopt_jax", "simsopt_jax_adapters"}
        )
        if project_module and not source_path.is_relative_to(resolved_root):
            raise ValueError(
                f"snapshot project module escaped immutable root: {module_name}"
            )
        if source_path.is_file() and source_path.is_relative_to(resolved_root):
            relative_paths.add(source_path.relative_to(resolved_root).as_posix())
    sources: list[ExecutedSource] = []
    for relative in sorted(relative_paths):
        source_path = (resolved_root / relative).resolve()
        expected_sha256 = identity.manifest_entries.get(relative)
        if (
            not source_path.is_relative_to(resolved_root)
            or not source_path.is_file()
            or expected_sha256 is None
            or _sha256_file(source_path) != expected_sha256
        ):
            raise ValueError(
                f"snapshot executed source is not manifest-bound: {relative}"
            )
        sources.append(ExecutedSource(relative, expected_sha256, None))
    return tuple(sources)


def collect_snapshot_lane_provenance(
    identity: SnapshotLaneIdentity, *, measurement_synchronization: str
) -> LaneProvenance:
    """Collect dynamic lane facts after work against validated immutable identity."""

    devices, device_peak, device_status, jax_version, effective_guards = (
        _device_metadata()
    )
    if Path(sys.executable).resolve() != identity.interpreter_path.resolve():
        raise ValueError("snapshot runtime interpreter changed")
    if sys.version.split()[0] != identity.python_version:
        raise ValueError("snapshot runtime Python version changed")
    if jax_version != identity.jax_version:
        raise ValueError("snapshot runtime JAX version changed")
    if importlib.metadata.version("jaxlib") != identity.jaxlib_version:
        raise ValueError("snapshot runtime jaxlib version changed")
    validate_snapshot_lane_environment(identity, os.environ)
    observed_platforms = {device.platform for device in devices}
    if identity.execution_platform == "gpu":
        if not observed_platforms.intersection({"cuda", "gpu"}):
            raise ValueError("snapshot GPU lane has no observed GPU device")
    elif observed_platforms and observed_platforms != {"cpu"}:
        raise ValueError("snapshot native lane observed a non-CPU JAX device")
    simsopt_module = sys.modules.get("simsopt")
    simsopt_version = str(getattr(simsopt_module, "__version__", "unknown"))
    version_module = sys.modules.get("simsopt._version")
    version_commit_value = getattr(version_module, "commit_id", None)
    version_commit = (
        version_commit_value if isinstance(version_commit_value, str) else None
    )
    version_compatible = (
        generated_version_matches_checkout(identity.repository_commit, version_commit)
        if version_module is not None
        else None
    )
    simsoptpp_module = sys.modules.get("simsoptpp")
    simsoptpp_path = None
    simsoptpp_sha256 = None
    simsoptpp_version = None
    build_commit = None
    compatible = None
    if simsoptpp_module is None:
        raise ValueError("snapshot lane did not load the bound native extension")
    if simsoptpp_module is not None:
        binary_path = Path(str(simsoptpp_module.__file__)).resolve()
        simsoptpp_path = str(binary_path)
        simsoptpp_sha256 = _sha256_file(binary_path)
        simsoptpp_version = str(getattr(simsoptpp_module, "__version__", "unknown"))
        build_commit = os.environ.get("SIMSOPT_PARITY_SIMSOPTPP_BUILD_COMMIT")
        compatible = build_commit == identity.repository_commit
        if (
            binary_path != identity.native_extension_path
            or simsoptpp_sha256 != identity.native_extension_sha256
        ):
            raise ValueError("snapshot native extension identity changed")
    executed_sources = _snapshot_executed_sources(identity)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return LaneProvenance(
        repository_commit=identity.repository_commit,
        repository_dirty=identity.repository_dirty,
        tracked_diff_sha256=identity.tracked_diff_sha256,
        untracked_files=identity.untracked_files,
        executed_sources=executed_sources,
        python_version=identity.python_version,
        jax_version=jax_version,
        simsopt_version=simsopt_version,
        simsopt_version_commit=version_commit,
        simsopt_version_checkout_compatible=version_compatible,
        lane_environment_policy={
            key: os.environ[key] for key in _LANE_ENVIRONMENT_KEYS if key in os.environ
        },
        jax_effective_transfer_guards=effective_guards,
        devices=devices,
        host_peak_rss_bytes=int(peak_rss) * 1024,
        host_peak_rss_method="child getrusage(RUSAGE_SELF).ru_maxrss fallback",
        device_memory_peak_bytes=device_peak,
        device_memory_status=device_status,
        memory_measurement_scope=(
            "combined import, compile/warmup, and one bounded execution"
        ),
        steady_state_memory_measured=False,
        measurement_synchronization=measurement_synchronization,
        simsoptpp_path=simsoptpp_path,
        simsoptpp_sha256=simsoptpp_sha256,
        simsoptpp_version=simsoptpp_version,
        simsoptpp_build_commit=build_commit,
        simsoptpp_checkout_compatible=compatible,
        authoritative=(
            not identity.repository_dirty
            and (simsoptpp_module is None or compatible is True)
        ),
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
