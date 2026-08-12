"""Publish immutable changed-state GPU timeline evidence.

The writer owns the on-disk schema, path policy, hashing, durability, and
atomic publication.  Callers provide already-captured evidence as typed file
records; the final artifact root is visible only after every byte is durable
and its manifest has been validated against the staged tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Final, Literal

ARTIFACT_SCHEMA_ID: Final = "single-stage-changed-state-gpu-timeline-segmented-v2"
CHILD_SCHEMA_ID: Final = "single-stage-changed-state-gpu-timeline-child-segmented-v2"
EVENT_SCHEMA_ID: Final = "single-stage-changed-state-gpu-timeline-event-segmented-v2"
OBSERVATION_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-observation-segmented-v2"
)
MANIFEST_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-manifest-segmented-v2"
)
VALIDATION_SCHEMA_ID: Final = (
    "single-stage-changed-state-gpu-timeline-validation-segmented-v2"
)

PRODUCTION_OPTIMIZER: Final = "SIMSOPT_LBFGSB"
PRODUCTION_DRIVER: Final = "minimize_lbfgs_host_core"
PRODUCTION_LINE_SEARCH: Final = "line_search_value_and_grad_host"
DIRECT_ADJOINT_ROUTE: Final = "exact_jacobian_dense_fp64_lu"
REQUIRED_ACCEPTED_ITERATIONS: Final = 7
REQUIRED_PROFILE_CHILDREN: Final = 3
REQUIRED_CONTROL_CHILDREN: Final = 3

ChildMode = Literal["profiled", "control"]
ManifestRole = Literal[
    "artifact_metadata",
    "child_metadata",
    "host_device_events",
    "numerical_observations",
    "raw_trace",
    "trace_summary",
    "optimization_timing",
    "trajectory",
    "provenance",
    "diagnostic",
    "identity_preimages",
    "input_evidence",
    "preflight_evidence",
    "source_evidence",
]

_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_MANIFEST_ROLES = frozenset(
    {
        "artifact_metadata",
        "child_metadata",
        "host_device_events",
        "numerical_observations",
        "raw_trace",
        "trace_summary",
        "optimization_timing",
        "trajectory",
        "provenance",
        "diagnostic",
        "identity_preimages",
        "input_evidence",
        "preflight_evidence",
        "source_evidence",
    }
)


@dataclass(frozen=True)
class RouteIdentity:
    """Exact production algorithm identities; no substitute route is valid."""

    optimizer: Literal["SIMSOPT_LBFGSB"] = PRODUCTION_OPTIMIZER
    driver: Literal["minimize_lbfgs_host_core"] = PRODUCTION_DRIVER
    line_search: Literal["line_search_value_and_grad_host"] = PRODUCTION_LINE_SEARCH
    adjoint_route: Literal["exact_jacobian_dense_fp64_lu"] = DIRECT_ADJOINT_ROUTE


@dataclass(frozen=True)
class ChildScheduleEntry:
    """One child position and its profile/control comparison pair."""

    child_id: str
    mode: ChildMode
    pair_index: int
    order_index: int


@dataclass(frozen=True)
class TimelineMetadata:
    """Artifact-wide identities and the complete alternating child schedule."""

    artifact_id: str
    created_utc: str
    source_state_sha256: str
    trace_schema_id: str
    phase_schema_version: str
    phase_ids: tuple[str, ...]
    hostname: str
    device_name: str
    device_uuid: str
    python_version: str
    jax_version: str
    jaxlib_version: str
    cuda_runtime: str
    cuda_driver: str
    cpu_identity: str
    affinity: str
    environment_sha256: str
    input_sha256: str
    configuration_sha256: str
    construction_sha256: str
    runtime_policy_sha256: str
    initial_parameters_sha256: str
    child_schedule: tuple[ChildScheduleEntry, ...]
    route: RouteIdentity = RouteIdentity()
    schema_id: Literal["single-stage-changed-state-gpu-timeline-segmented-v2"] = (
        ARTIFACT_SCHEMA_ID
    )
    scale: Literal["native_default"] = "native_default"
    precision: Literal["fp64"] = "fp64"
    accepted_iterations: int = REQUIRED_ACCEPTED_ITERATIONS
    authoritative: bool = True


@dataclass(frozen=True)
class ClaimFile:
    """One claim-bearing source file copied into the immutable receipt."""

    role: ManifestRole
    relative_path: str
    source_path: Path
    source_state_sha256: str
    process_id: str
    evaluation_ids_sha256: str
    sample_id: str | None = None
    evaluation_id: str | None = None
    segment_evaluation_ids_sha256: str | None = None


@dataclass(frozen=True)
class ManifestEntry:
    """Hash and correlation identity for one claim-bearing artifact file."""

    role: ManifestRole
    relative_path: str
    size_bytes: int
    sha256: str
    source_state_sha256: str
    process_id: str
    evaluation_ids_sha256: str
    sample_id: str | None
    evaluation_id: str | None
    segment_evaluation_ids_sha256: str | None


def canonical_json_bytes(document: object) -> bytes:
    """Return the sole canonical JSON representation used by this artifact."""

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


def evaluation_ids_sha256(evaluation_ids: tuple[str, ...]) -> str:
    """Hash one canonical sorted, unique aggregate evaluation identity."""

    if len(evaluation_ids) != len(set(evaluation_ids)) or any(
        not evaluation_id for evaluation_id in evaluation_ids
    ):
        raise ValueError("evaluation IDs must be nonempty and unique")
    canonical_ids = tuple(sorted(evaluation_ids))
    return hashlib.sha256(canonical_json_bytes(list(canonical_ids))).hexdigest()


def _require_nonempty(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _require_sha256(value: str, field: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in _HEX_DIGITS for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise ValueError(f"relative_path is not a canonical safe path: {value!r}")
    return path


def _validate_metadata(metadata: TimelineMetadata) -> None:
    if metadata.schema_id != ARTIFACT_SCHEMA_ID:
        raise ValueError(f"schema_id must be {ARTIFACT_SCHEMA_ID}")
    if metadata.artifact_id != ARTIFACT_SCHEMA_ID:
        raise ValueError(f"artifact_id must be {ARTIFACT_SCHEMA_ID}")
    if metadata.scale != "native_default" or metadata.precision != "fp64":
        raise ValueError("timeline must use native_default scale and fp64 precision")
    if metadata.accepted_iterations != REQUIRED_ACCEPTED_ITERATIONS:
        raise ValueError(f"accepted_iterations must be {REQUIRED_ACCEPTED_ITERATIONS}")
    if metadata.route != RouteIdentity():
        raise ValueError(
            "route must be the production host L-BFGS-B/direct-adjoint route"
        )
    for field in (
        "artifact_id",
        "created_utc",
        "trace_schema_id",
        "phase_schema_version",
        "hostname",
        "device_name",
        "device_uuid",
        "python_version",
        "jax_version",
        "jaxlib_version",
        "cuda_runtime",
        "cuda_driver",
        "cpu_identity",
        "affinity",
    ):
        _require_nonempty(str(getattr(metadata, field)), field)
    created_at = datetime.fromisoformat(metadata.created_utc)
    if created_at.tzinfo is None or created_at.utcoffset() != timedelta(0):
        raise ValueError("created_utc must be a timezone-aware UTC ISO timestamp")
    for field in (
        "source_state_sha256",
        "environment_sha256",
        "input_sha256",
        "configuration_sha256",
        "construction_sha256",
        "runtime_policy_sha256",
        "initial_parameters_sha256",
    ):
        _require_sha256(str(getattr(metadata, field)), field)
    if (
        len(metadata.phase_ids) != len(set(metadata.phase_ids))
        or not metadata.phase_ids
    ):
        raise ValueError("phase_ids must be non-empty and unique")
    if any(not phase_id.strip() for phase_id in metadata.phase_ids):
        raise ValueError("phase_ids must not contain empty values")

    expected_length = REQUIRED_PROFILE_CHILDREN + REQUIRED_CONTROL_CHILDREN
    schedule = metadata.child_schedule
    if len(schedule) != expected_length:
        raise ValueError(
            f"child_schedule must contain exactly {expected_length} entries"
        )
    if tuple(entry.order_index for entry in schedule) != tuple(range(expected_length)):
        raise ValueError(
            "child_schedule order_index values must be contiguous from zero"
        )
    if len({entry.child_id for entry in schedule}) != expected_length:
        raise ValueError("child_schedule child_id values must be unique")
    expected_modes = tuple(
        mode
        for _ in range(REQUIRED_PROFILE_CHILDREN)
        for mode in ("profiled", "control")
    )
    if tuple(entry.mode for entry in schedule) != expected_modes:
        raise ValueError("child_schedule must alternate profiled then control")
    for pair_index in range(REQUIRED_PROFILE_CHILDREN):
        pair = tuple(entry for entry in schedule if entry.pair_index == pair_index)
        if len(pair) != 2 or {entry.mode for entry in pair} != {"profiled", "control"}:
            raise ValueError(
                f"pair_index {pair_index} must contain one profiled/control pair"
            )


def _validate_claim_files(
    metadata: TimelineMetadata, files: tuple[ClaimFile, ...]
) -> None:
    paths: set[str] = set()
    role_bindings: set[tuple[str, str]] = set()
    child_ids = {entry.child_id for entry in metadata.child_schedule}
    required_roles: dict[str, set[str]] = {
        entry.child_id: {
            "child_metadata",
            "host_device_events",
            "numerical_observations",
            "optimization_timing",
            "trajectory",
            "provenance",
        }
        for entry in metadata.child_schedule
    }
    profiled_child_ids = {
        entry.child_id for entry in metadata.child_schedule if entry.mode == "profiled"
    }
    required_segment_ids = {
        f"iteration-{iteration:02d}"
        for iteration in range(1, REQUIRED_ACCEPTED_ITERATIONS + 1)
    }
    segment_bindings: dict[str, dict[str, dict[str, ClaimFile]]] = {
        child_id: {role: {} for role in ("raw_trace", "trace_summary")}
        for child_id in profiled_child_ids
    }
    allowed_roles = {
        child_id: roles | {"diagnostic"} for child_id, roles in required_roles.items()
    }
    for child_id in profiled_child_ids:
        allowed_roles[child_id].update({"raw_trace", "trace_summary"})
    allowed_roles["artifact"] = {
        "identity_preimages",
        "input_evidence",
        "preflight_evidence",
        "source_evidence",
    }
    for claim_file in files:
        _safe_relative_path(claim_file.relative_path)
        if claim_file.relative_path in {"artifact.json", "manifest.json"}:
            raise ValueError(f"reserved artifact path: {claim_file.relative_path}")
        if claim_file.relative_path in paths:
            raise ValueError(f"duplicate artifact path: {claim_file.relative_path}")
        paths.add(claim_file.relative_path)
        if claim_file.process_id not in child_ids | {"artifact"}:
            raise ValueError(f"unknown process_id: {claim_file.process_id}")
        if claim_file.role not in _MANIFEST_ROLES - {"artifact_metadata"}:
            raise ValueError(f"unknown manifest role: {claim_file.role}")
        if claim_file.role not in allowed_roles[claim_file.process_id]:
            raise ValueError(
                f"{claim_file.process_id}: role {claim_file.role!r} is not allowed "
                "for this child mode"
            )
        role_binding = (claim_file.process_id, claim_file.role)
        if (
            claim_file.role
            not in {
                "diagnostic",
                "input_evidence",
                "source_evidence",
                "raw_trace",
                "trace_summary",
            }
            and role_binding in role_bindings
        ):
            raise ValueError(
                "duplicate process/manifest-role binding: "
                f"{claim_file.process_id}/{claim_file.role}"
            )
        role_bindings.add(role_binding)
        if claim_file.process_id != "artifact" and (
            claim_file.sample_id is None or not claim_file.sample_id.strip()
        ):
            raise ValueError(
                f"{claim_file.relative_path} must bind a non-empty sample_id"
            )
        if claim_file.role in {"raw_trace", "trace_summary"}:
            if claim_file.process_id not in profiled_child_ids:
                raise ValueError("control children cannot publish segment evidence")
            sample_id = claim_file.sample_id
            if sample_id not in required_segment_ids:
                raise ValueError(
                    f"{claim_file.relative_path} has invalid segment sample_id"
                )
            if claim_file.evaluation_id is None or not claim_file.evaluation_id.strip():
                raise ValueError(
                    f"{claim_file.relative_path} must bind one evaluation_id"
                )
            if claim_file.segment_evaluation_ids_sha256 is None:
                raise ValueError(
                    f"{claim_file.relative_path} must bind segment evaluation IDs"
                )
            _require_sha256(
                claim_file.segment_evaluation_ids_sha256,
                f"{claim_file.relative_path}.segment_evaluation_ids_sha256",
            )
            role_segments = segment_bindings[claim_file.process_id][claim_file.role]
            if sample_id in role_segments:
                raise ValueError(
                    f"{claim_file.process_id}: duplicate {claim_file.role} segment "
                    f"{sample_id}"
                )
            role_segments[sample_id] = claim_file
            expected_filename = (
                "trace.json.gz"
                if claim_file.role == "raw_trace"
                else "trace_summary.json"
            )
            expected_path = (
                f"children/{claim_file.process_id}/segments/{sample_id}/"
                f"{expected_filename}"
            )
            if claim_file.relative_path != expected_path:
                raise ValueError(
                    f"{claim_file.process_id}: {claim_file.role} segment path "
                    "differs from frozen protocol"
                )
        elif claim_file.segment_evaluation_ids_sha256 is not None:
            raise ValueError(
                f"{claim_file.relative_path} cannot bind segment evaluation IDs"
            )
        _require_sha256(
            claim_file.evaluation_ids_sha256,
            f"{claim_file.relative_path}.evaluation_ids_sha256",
        )
        if claim_file.source_state_sha256 != metadata.source_state_sha256:
            raise ValueError(
                f"{claim_file.relative_path} source state differs from artifact metadata"
            )
        if not claim_file.source_path.is_file() or claim_file.source_path.is_symlink():
            raise ValueError(
                f"claim source must be a regular non-symlink file: {claim_file.source_path}"
            )
        if claim_file.process_id != "artifact":
            required_roles[claim_file.process_id].discard(claim_file.role)
    missing = {
        child_id: sorted(roles) for child_id, roles in required_roles.items() if roles
    }
    if missing:
        raise ValueError(f"children missing required claim-bearing roles: {missing}")
    child_state_by_id: dict[str, str] = {}
    for child_id in child_ids:
        child_files = tuple(
            claim_file
            for claim_file in files
            if claim_file.process_id == child_id and claim_file.role == "child_metadata"
        )
        if len(child_files) != 1:
            raise ValueError(f"{child_id}: expected exactly one child metadata file")
        try:
            child_document = json.loads(child_files[0].source_path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{child_id}: child metadata is not JSON") from error
        if (
            not isinstance(child_document, dict)
            or child_document.get("child_id") != child_id
        ):
            raise ValueError(f"{child_id}: child metadata identity differs")
        state = child_document.get("state")
        if state not in {"complete", "failed", "incomplete"}:
            raise ValueError(f"{child_id}: child metadata state is invalid")
        child_state_by_id[child_id] = state
    for child_id, roles in segment_bindings.items():
        raw_ids = set(roles["raw_trace"])
        summary_ids = set(roles["trace_summary"])
        if raw_ids != summary_ids:
            raise ValueError(f"{child_id}: trace/summary segment sets differ")
        expected_prefix = {
            f"iteration-{iteration:02d}" for iteration in range(1, len(raw_ids) + 1)
        }
        if raw_ids != expected_prefix:
            raise ValueError(
                f"{child_id}: retained segment IDs are not a contiguous prefix"
            )
        if (
            child_state_by_id[child_id] == "complete"
            and raw_ids != required_segment_ids
        ):
            raise ValueError(f"{child_id}: complete child requires all seven segments")
        for sample_id in raw_ids:
            raw_trace = roles["raw_trace"][sample_id]
            trace_summary = roles["trace_summary"][sample_id]
            if (
                raw_trace.evaluation_id != trace_summary.evaluation_id
                or raw_trace.evaluation_ids_sha256
                != trace_summary.evaluation_ids_sha256
                or raw_trace.segment_evaluation_ids_sha256
                != trace_summary.segment_evaluation_ids_sha256
            ):
                raise ValueError(
                    f"{child_id}: segment {sample_id} trace/summary binding differs"
                )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(1024 * 1024):
            digest.update(chunk)
            output_stream.write(chunk)
            size_bytes += len(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    _fsync_directory(destination.parent)
    return size_bytes, digest.hexdigest()


def _metadata_document(metadata: TimelineMetadata) -> dict[str, object]:
    document = asdict(metadata)
    document["child_schedule"] = [asdict(entry) for entry in metadata.child_schedule]
    document["phase_ids"] = list(metadata.phase_ids)
    document["route"] = asdict(metadata.route)
    return document


def _validate_staged_manifest(
    staging_root: Path, entries: tuple[ManifestEntry, ...]
) -> None:
    expected_paths = {entry.relative_path for entry in entries} | {"manifest.json"}
    actual_paths = {
        path.relative_to(staging_root).as_posix()
        for path in staging_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError(
            f"staged artifact file set differs from manifest: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    for entry in entries:
        path = staging_root / entry.relative_path
        content = path.read_bytes()
        if len(content) != entry.size_bytes:
            raise ValueError(f"staged size mismatch: {entry.relative_path}")
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ValueError(f"staged hash mismatch: {entry.relative_path}")


def write_timeline_receipt(
    artifact_root: Path,
    metadata: TimelineMetadata,
    files: tuple[ClaimFile, ...],
) -> Path:
    """Atomically create one fresh immutable artifact and return its root."""

    _validate_metadata(metadata)
    if artifact_root.exists():
        raise FileExistsError(f"artifact root already exists: {artifact_root}")
    if metadata.authoritative and artifact_root.resolve().is_relative_to(Path("/tmp")):
        raise ValueError(
            "authoritative timeline artifacts must not be written under /tmp"
        )
    _validate_claim_files(metadata, files)

    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(artifact_root.parent)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_root.name}.staging-", dir=artifact_root.parent
        )
    )
    try:
        artifact_bytes = canonical_json_bytes(_metadata_document(metadata))
        _write_bytes(staging_root / "artifact.json", artifact_bytes)
        entries: list[ManifestEntry] = [
            ManifestEntry(
                role="artifact_metadata",
                relative_path="artifact.json",
                size_bytes=len(artifact_bytes),
                sha256=hashlib.sha256(artifact_bytes).hexdigest(),
                source_state_sha256=metadata.source_state_sha256,
                process_id="artifact",
                evaluation_ids_sha256=evaluation_ids_sha256(()),
                sample_id=None,
                evaluation_id=None,
                segment_evaluation_ids_sha256=None,
            )
        ]
        for claim_file in files:
            size_bytes, digest = _copy_and_hash(
                claim_file.source_path, staging_root / claim_file.relative_path
            )
            entries.append(
                ManifestEntry(
                    role=claim_file.role,
                    relative_path=claim_file.relative_path,
                    size_bytes=size_bytes,
                    sha256=digest,
                    source_state_sha256=claim_file.source_state_sha256,
                    process_id=claim_file.process_id,
                    evaluation_ids_sha256=claim_file.evaluation_ids_sha256,
                    sample_id=claim_file.sample_id,
                    evaluation_id=claim_file.evaluation_id,
                    segment_evaluation_ids_sha256=(
                        claim_file.segment_evaluation_ids_sha256
                    ),
                )
            )
        entries_tuple = tuple(sorted(entries, key=lambda entry: entry.relative_path))
        manifest_document = {
            "schema_id": MANIFEST_SCHEMA_ID,
            "artifact_schema_id": ARTIFACT_SCHEMA_ID,
            "entries": [asdict(entry) for entry in entries_tuple],
        }
        _write_bytes(
            staging_root / "manifest.json", canonical_json_bytes(manifest_document)
        )
        _validate_staged_manifest(staging_root, entries_tuple)
        staging_root.rename(artifact_root)
        _fsync_directory(artifact_root.parent)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return artifact_root
