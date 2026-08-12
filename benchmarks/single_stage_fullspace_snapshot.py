"""Immutable source and live-runtime evidence for the full-space GPU lane.

The source publisher accepts only explicitly assigned execution roots, copies
their exact bytes without following symlinks, and atomically seals a canonical
manifest tree.  The runtime producer then binds live imports, interpreter,
JAX backend, physical GPU, process invocation, and effective performance
environment to that manifest before a claim-bearing run can start.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Final, Literal, TypeAlias

SOURCE_MANIFEST_SCHEMA_VERSION: Final = "single-stage-fullspace-source-manifest-v1"
RUNTIME_EVIDENCE_SCHEMA_VERSION: Final = "single-stage-fullspace-runtime-evidence-v1"
RUNTIME_EVIDENCE_V2_SCHEMA_VERSION: Final = "single-stage-fullspace-runtime-evidence-v2"
SOURCE_MANIFEST_FILENAME: Final = "source-manifest.json"
RUNTIME_EVIDENCE_FILENAME: Final = "runtime-evidence.json"

SnapshotRole = Literal[
    "execution_source",
    "configuration",
    "benchmark",
    "test",
    "native_extension",
    "execution_source_manifest",
    "prequalification_plan",
]
CommandRunner: TypeAlias = Callable[
    [Sequence[str], Mapping[str, str]], subprocess.CompletedProcess[str]
]
JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

LEGACY_SNAPSHOT_ROLES: Final[frozenset[SnapshotRole]] = frozenset(
    {"execution_source", "configuration", "benchmark", "test", "native_extension"}
)
DIAG4_GPU_SNAPSHOT_ROLES: Final[frozenset[SnapshotRole]] = frozenset(
    {
        "execution_source",
        "benchmark",
        "test",
        "execution_source_manifest",
        "native_extension",
    }
)
DIAG4_CPU_SNAPSHOT_ROLES: Final[frozenset[SnapshotRole]] = (
    DIAG4_GPU_SNAPSHOT_ROLES | frozenset({"prequalification_plan"})
)
# DIAG5 changes live-runtime/native-binding schemas, not snapshot membership.
# These aliases make the successor contract explicit without reinterpreting
# either DIAG4 manifest bytes or the generation-neutral source-manifest schema.
DIAG5_GPU_SNAPSHOT_ROLES: Final[frozenset[SnapshotRole]] = DIAG4_GPU_SNAPSHOT_ROLES
DIAG5_CPU_SNAPSHOT_ROLES: Final[frozenset[SnapshotRole]] = DIAG4_CPU_SNAPSHOT_ROLES
DIAG5_PREQUALIFICATION_PLAN_RELATIVE_PATH: Final = "control/prequalification-plan.md"
_SUPPORTED_SNAPSHOT_ROLE_SETS: Final = (
    LEGACY_SNAPSHOT_ROLES,
    DIAG4_GPU_SNAPSHOT_ROLES,
    DIAG4_CPU_SNAPSHOT_ROLES,
)
_IMPORT_MODULES: Final = (
    "simsopt",
    "simsopt_jax",
    "simsopt_jax_adapters",
    "simsoptpp",
)
_ENVIRONMENT_KEYS: Final = tuple(
    sorted(
        (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "JAX_COMPILATION_CACHE_DIR",
            "JAX_ENABLE_COMPILATION_CACHE",
            "JAX_CUDA_VISIBLE_DEVICES",
            "JAX_DEFAULT_MATMUL_PRECISION",
            "JAX_ENABLE_X64",
            "JAX_NUM_CPU_DEVICES",
            "JAX_PLATFORM_NAME",
            "JAX_PLATFORMS",
            "JAX_TRANSFER_GUARD",
            "LD_LIBRARY_PATH",
            "MKL_NUM_THREADS",
            "NVIDIA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "PATH",
            "PYTHONPATH",
            "TF_NUM_INTEROP_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS",
            "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
        )
    )
)
XLA_GPU_COMMAND_BUFFER_DISABLE_FLAG: Final = "--xla_gpu_enable_command_buffer="
_XLA_GPU_COMMAND_BUFFER_FLAG_PREFIX: Final = (
    XLA_GPU_COMMAND_BUFFER_DISABLE_FLAG.removesuffix("=")
)
_SHA256_LENGTH: Final = 64
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_COPY_CHUNK_BYTES: Final = 1024 * 1024


def command_buffer_disabled_xla_flags(value: str) -> str:
    """Canonicalize XLA flags with GPU command-buffer execution disabled."""

    retained = tuple(
        token
        for token in shlex.split(value)
        if token != _XLA_GPU_COMMAND_BUFFER_FLAG_PREFIX
        and not token.startswith(f"{_XLA_GPU_COMMAND_BUFFER_FLAG_PREFIX}=")
    )
    return shlex.join((*retained, XLA_GPU_COMMAND_BUFFER_DISABLE_FLAG))


def activate_snapshot_source_imports(source_root: Path) -> None:
    """Make copied SIMSOPT sources authoritative over editable-install hooks."""

    resolved_source = str(source_root.resolve(strict=True))
    sys.path[:] = [
        resolved_source,
        *(path for path in sys.path if path != resolved_source),
    ]
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            finder.__class__.__module__.startswith("_editable_")
            and "simsopt" in finder.__class__.__module__
        )
    ]


class SnapshotValidationError(ValueError):
    """The selected source or runtime evidence failed an integrity boundary."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Content-addressed canonical artifact inside one campaign root."""

    relative_path: str
    sha256: str
    size_bytes: int
    schema_version: str

    def resolve_and_validate(self, campaign_root: Path) -> Path:
        root = _strict_directory(campaign_root, "campaign root")
        try:
            relative = _safe_relative_path(self.relative_path, "artifact relative_path")
        except SnapshotValidationError as error:
            raise SnapshotValidationError(
                f"noncanonical artifact path: {self.relative_path!r}"
            ) from error
        path = root.joinpath(*relative.parts)
        try:
            _reject_symlink_components(path, "artifact path")
        except SnapshotValidationError as error:
            raise SnapshotValidationError(
                f"artifact path contains a symlink: {self.relative_path}"
            ) from error
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise SnapshotValidationError(
                f"artifact is not a regular file: {self.relative_path}"
            )
        payload = resolved.read_bytes()
        if len(payload) != self.size_bytes:
            raise SnapshotValidationError(
                f"artifact size mismatch: {self.relative_path}"
            )
        if _sha256_bytes(payload) != self.sha256:
            raise SnapshotValidationError(
                f"artifact digest mismatch: {self.relative_path}"
            )
        decoded = load_canonical_json_bytes(payload)
        if (
            not isinstance(decoded, dict)
            or decoded.get("schema_version") != self.schema_version
        ):
            raise SnapshotValidationError(
                f"artifact schema mismatch: {self.relative_path}"
            )
        return resolved


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Static source identity embedded unchanged in each run receipt."""

    snapshot_manifest: ArtifactRef
    git_head: str
    tracked_diff_sha256: str
    untracked_bytes_manifest_sha256: str
    repo_root: str


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Dynamic child-observed identity embedded unchanged in each run receipt."""

    argv: tuple[str, ...]
    cwd: str
    python_executable: str
    python_version: str
    jax_version: str
    jaxlib_version: str
    simsopt_module_path: str
    simsopt_jax_module_path: str
    native_extension_path: str
    backend: str
    device_uuid: str
    driver_version: str
    effective_environment_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeIdentityV2(RuntimeIdentity):
    """DIAG5 runtime identity with the independently loaded native binding."""

    native_extension_sha256: str
    native_extension_size_bytes: int
    native_extension_link_count: int


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize the evidence protocol's sole UTF-8 JSON representation."""

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


def load_canonical_json_bytes(payload: bytes) -> JsonValue:
    """Parse JSON while rejecting duplicate keys and noncanonical encodings."""

    def reject_duplicates(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotValidationError(f"duplicate key {key!r} in JSON")
            result[key] = value
        return result

    try:
        decoded: JsonValue = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotValidationError("payload is not UTF-8 JSON") from error
    if canonical_json_bytes(decoded) != payload:
        raise SnapshotValidationError("JSON artifact is not canonical")
    return decoded


@dataclass(frozen=True, slots=True)
class SourceRoot:
    """One explicit source file or recursively complete directory assignment."""

    role: SnapshotRole
    source_path: Path
    relative_path: str


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    """Exact-byte identity and role of one execution-bearing snapshot file."""

    role: SnapshotRole
    relative_path: str
    size_bytes: int
    sha256: str

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class WorktreeIdentity:
    """Git and exact untracked-byte identity of the live publication source."""

    git_head: str
    tracked_diff_sha256: str
    untracked_bytes_manifest_sha256: str
    repo_root: str

    def to_payload(self) -> dict[str, JsonValue]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    """Pathless canonical identity of one immutable source snapshot."""

    manifest_sha256: str
    entries: tuple[SnapshotEntry, ...]
    worktree: WorktreeIdentity


@dataclass(frozen=True, slots=True)
class SnapshotPublication:
    """A sealed snapshot plus its content-addressed source identity."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    entries: tuple[SnapshotEntry, ...]
    worktree: WorktreeIdentity

    def identity(self) -> SnapshotIdentity:
        """Return the path-independent identity of these sealed bytes."""

        return build_snapshot_identity(self.entries, self.worktree)

    def source_identity(self, campaign_root: Path) -> SourceIdentity:
        root = _strict_directory(campaign_root, "campaign root")
        manifest = self.manifest_path.resolve(strict=True)
        try:
            relative_path = manifest.relative_to(root).as_posix()
        except ValueError as error:
            raise SnapshotValidationError(
                "snapshot manifest must be inside the campaign root"
            ) from error
        payload = manifest.read_bytes()
        return SourceIdentity(
            snapshot_manifest=ArtifactRef(
                relative_path=relative_path,
                sha256=self.manifest_sha256,
                size_bytes=len(payload),
                schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
            ),
            git_head=self.worktree.git_head,
            tracked_diff_sha256=self.worktree.tracked_diff_sha256,
            untracked_bytes_manifest_sha256=(
                self.worktree.untracked_bytes_manifest_sha256
            ),
            repo_root=self.worktree.repo_root,
        )


class SnapshotPostPublicationError(OSError):
    """A published snapshot exists, but its post-rename durability step failed."""

    def __init__(self, publication: SnapshotPublication, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.publication = publication
        self.cause = cause


@dataclass(frozen=True, slots=True)
class ImportBinding:
    """A live module origin proven equal to one source-manifest entry."""

    module: str
    relative_path: str
    size_bytes: int
    sha256: str

    def to_payload(self) -> dict[str, JsonValue]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """Facts observed inside the process that will execute the GPU solve."""

    runtime_identity: RuntimeIdentity
    entrypoint_binding: ImportBinding
    import_bindings: tuple[ImportBinding, ...]
    effective_environment: tuple[tuple[str, str | None], ...]
    device_name: str
    platform_version: str


@dataclass(frozen=True, slots=True)
class RuntimeObservationV2(RuntimeObservation):
    """DIAG5 observation whose native import is separate from snapshot sources."""

    runtime_identity: RuntimeIdentityV2


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    """Canonical dynamic evidence bound to one immutable source identity."""

    source_identity: SourceIdentity
    observation: RuntimeObservation
    snapshot_manifest_sha256: str

    def to_payload(self) -> dict[str, JsonValue]:
        environment = {
            key: value for key, value in self.observation.effective_environment
        }
        return {
            "device": {
                "name": self.observation.device_name,
                "platform_version": self.observation.platform_version,
            },
            "effective_environment": environment,
            "entrypoint_binding": self.observation.entrypoint_binding.to_payload(),
            "import_bindings": [
                binding.to_payload() for binding in self.observation.import_bindings
            ],
            "runtime_identity": asdict(self.observation.runtime_identity),
            "schema_version": RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "source_identity": {
                **asdict(self.source_identity),
                "snapshot_manifest": asdict(self.source_identity.snapshot_manifest),
            },
        }


@dataclass(frozen=True, slots=True)
class RuntimeEvidenceV2:
    """Canonical DIAG5 runtime evidence; v1 evidence remains byte-stable."""

    source_identity: SourceIdentity
    observation: RuntimeObservationV2
    snapshot_manifest_sha256: str

    def to_payload(self) -> dict[str, JsonValue]:
        environment = {
            key: value for key, value in self.observation.effective_environment
        }
        return {
            "device": {
                "name": self.observation.device_name,
                "platform_version": self.observation.platform_version,
            },
            "effective_environment": environment,
            "entrypoint_binding": self.observation.entrypoint_binding.to_payload(),
            "import_bindings": [
                binding.to_payload() for binding in self.observation.import_bindings
            ],
            "runtime_identity": asdict(self.observation.runtime_identity),
            "schema_version": RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "source_identity": {
                **asdict(self.source_identity),
                "snapshot_manifest": asdict(self.source_identity.snapshot_manifest),
            },
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, _COPY_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in _LOWER_HEX for character in value)
    )


def _safe_relative_path(value: str, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise SnapshotValidationError(
            f"{context} must be a canonical non-empty relative path"
        )
    if any(part.startswith(".env") and part != ".env.example" for part in path.parts):
        raise SnapshotValidationError(f"{context} may not include an .env variant")
    return path


def _reject_symlink_components(path: Path, context: str) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise SnapshotValidationError(f"{context} contains a symlink component")
        if current.parent == current:
            return
        current = current.parent


def _strict_directory(path: Path, context: str) -> Path:
    _reject_symlink_components(path, context)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise SnapshotValidationError(f"{context} is not a directory")
    return resolved


def _run_command(
    argv: Sequence[str], environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv),
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )


def _git(repo_root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *arguments),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SnapshotValidationError(
            f"git {' '.join(arguments)} failed: {completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def capture_worktree_identity(repo_root: Path) -> WorktreeIdentity:
    """Hash HEAD, the binary tracked diff, and every non-secret untracked byte."""

    root = _strict_directory(repo_root, "repository root")
    git_head = _git(root, ("rev-parse", "HEAD")).decode("ascii").strip()
    if len(git_head) != 40 or any(
        character not in _LOWER_HEX for character in git_head
    ):
        raise SnapshotValidationError("git HEAD is not a lowercase SHA-1 object id")
    tracked_diff = _git(root, ("diff", "--binary", "--no-ext-diff", "HEAD", "--"))
    untracked_output = _git(root, ("ls-files", "--others", "--exclude-standard", "-z"))
    untracked_entries: list[dict[str, JsonValue]] = []
    raw_paths = untracked_output.split(b"\0")
    for raw_path in raw_paths[:-1]:
        relative_text = raw_path.decode("utf-8")
        preliminary = PurePosixPath(relative_text)
        if any(
            part.startswith(".env") and part != ".env.example"
            for part in preliminary.parts
        ):
            continue
        relative = _safe_relative_path(relative_text, "untracked path")
        source = root.joinpath(*relative.parts)
        _reject_symlink_components(source, "untracked path")
        if not source.is_file():
            raise SnapshotValidationError("untracked source is not a regular file")
        untracked_entries.append(
            {
                "relative_path": relative_text,
                "sha256": _sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    untracked_entries.sort(key=lambda item: str(item["relative_path"]))
    return WorktreeIdentity(
        git_head=git_head,
        tracked_diff_sha256=_sha256_bytes(tracked_diff),
        untracked_bytes_manifest_sha256=_sha256_bytes(
            canonical_json_bytes(untracked_entries)
        ),
        repo_root=str(root),
    )


def _validate_required_roles(
    required_roles: frozenset[SnapshotRole],
) -> frozenset[SnapshotRole]:
    for supported_roles in _SUPPORTED_SNAPSHOT_ROLE_SETS:
        if required_roles == supported_roles:
            return supported_roles
    raise SnapshotValidationError("required_roles is not a supported exact role set")


def _expand_source_roots(
    roots: Sequence[SourceRoot],
    *,
    required_roles: frozenset[SnapshotRole],
) -> tuple[tuple[SnapshotEntry, Path], ...]:
    if not roots:
        raise SnapshotValidationError("at least one explicit source root is required")
    selected: dict[str, tuple[SnapshotEntry, Path]] = {}
    observed_roles: set[str] = set()
    for index, root in enumerate(roots):
        context = f"source_roots[{index}]"
        if root.role not in required_roles:
            raise SnapshotValidationError(f"{context}.role is unsupported")
        source = root.source_path.absolute()
        destination_root = _safe_relative_path(
            root.relative_path, f"{context}.relative_path"
        )
        _reject_symlink_components(source, f"{context}.source_path")
        if source.is_file():
            descendants = (source,)
        elif source.is_dir():
            descendants = tuple(sorted(source.rglob("*")))
        else:
            raise SnapshotValidationError(
                f"{context}.source_path is not a regular file or directory"
            )
        for descendant in descendants:
            _reject_symlink_components(descendant, f"{context}.source_path")
            if descendant.is_dir():
                continue
            if not descendant.is_file():
                raise SnapshotValidationError(
                    f"{context}.source_path contains a non-regular file"
                )
            if source.is_file():
                relative = destination_root
            else:
                suffix = PurePosixPath(descendant.relative_to(source).as_posix())
                relative = destination_root.joinpath(*suffix.parts)
            relative_text = str(_safe_relative_path(str(relative), context))
            if relative_text == SOURCE_MANIFEST_FILENAME:
                raise SnapshotValidationError("source root collides with manifest path")
            if relative_text in selected:
                raise SnapshotValidationError(
                    f"snapshot destination {relative_text!r} is assigned more than once"
                )
            size_bytes = descendant.stat(follow_symlinks=False).st_size
            entry = SnapshotEntry(root.role, relative_text, size_bytes, "")
            selected[relative_text] = (entry, descendant)
            observed_roles.add(root.role)
    if observed_roles != required_roles:
        raise SnapshotValidationError(
            "source roots do not cover the exact required snapshot roles"
        )
    return tuple(selected[path] for path in sorted(selected))


def _manifest_payload(
    entries: Sequence[SnapshotEntry], worktree: WorktreeIdentity
) -> dict[str, JsonValue]:
    return {
        "entries": [entry.to_payload() for entry in entries],
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "worktree": worktree.to_payload(),
    }


def build_snapshot_identity(
    entries: Sequence[SnapshotEntry], worktree: WorktreeIdentity
) -> SnapshotIdentity:
    """Build an identity from the exact canonical source-manifest payload."""

    frozen_entries = tuple(entries)
    roles = frozenset(entry.role for entry in frozen_entries)
    _validate_required_roles(roles)
    paths = tuple(entry.relative_path for entry in frozen_entries)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise SnapshotValidationError(
            "snapshot identity paths are not sorted and unique"
        )
    for entry in frozen_entries:
        if (
            str(_safe_relative_path(entry.relative_path, "snapshot identity entry"))
            != entry.relative_path
        ):
            raise SnapshotValidationError("snapshot identity path is noncanonical")
        if (
            type(entry.size_bytes) is not int
            or entry.size_bytes < 0
            or not _is_sha256(entry.sha256)
        ):
            raise SnapshotValidationError("snapshot identity entry metadata is invalid")
    if (
        len(worktree.git_head) != 40
        or any(character not in "0123456789abcdef" for character in worktree.git_head)
        or not _is_sha256(worktree.tracked_diff_sha256)
        or not _is_sha256(worktree.untracked_bytes_manifest_sha256)
        or not Path(worktree.repo_root).is_absolute()
    ):
        raise SnapshotValidationError("snapshot identity worktree is invalid")
    manifest_bytes = canonical_json_bytes(_manifest_payload(frozen_entries, worktree))
    return SnapshotIdentity(
        manifest_sha256=_sha256_bytes(manifest_bytes),
        entries=frozen_entries,
        worktree=worktree,
    )


def project_diag5_gpu_snapshot_identity(
    cpu_identity: SnapshotIdentity,
) -> SnapshotIdentity:
    """Derive the GPU closure by removing only the sealed CPU plan control."""

    canonical_cpu = build_snapshot_identity(cpu_identity.entries, cpu_identity.worktree)
    if canonical_cpu != cpu_identity:
        raise SnapshotValidationError("DIAG5 CPU snapshot identity is noncanonical")
    if frozenset(entry.role for entry in cpu_identity.entries) != (
        DIAG5_CPU_SNAPSHOT_ROLES
    ):
        raise SnapshotValidationError("DIAG5 CPU snapshot roles differ")
    removed = tuple(
        entry for entry in cpu_identity.entries if entry.role == "prequalification_plan"
    )
    if len(removed) != 1 or removed[0].relative_path != (
        DIAG5_PREQUALIFICATION_PLAN_RELATIVE_PATH
    ):
        raise SnapshotValidationError("DIAG5 CPU prequalification-plan control differs")
    retained = tuple(
        entry
        for entry in cpu_identity.entries
        if entry.relative_path != DIAG5_PREQUALIFICATION_PLAN_RELATIVE_PATH
    )
    if (
        len(retained) + 1 != len(cpu_identity.entries)
        or frozenset(entry.role for entry in retained) != DIAG5_GPU_SNAPSHOT_ROLES
    ):
        raise SnapshotValidationError("DIAG5 GPU snapshot projection differs")
    return build_snapshot_identity(retained, cpu_identity.worktree)


def _seal_tree(root: Path) -> None:
    for path in (candidate for candidate in root.rglob("*") if candidate.is_file()):
        path.chmod(0o444)
    directories = tuple(
        candidate for candidate in root.rglob("*") if candidate.is_dir()
    )
    for path in reversed(directories):
        path.chmod(0o555)
    root.chmod(0o555)


def publish_immutable_snapshot(
    destination: Path,
    roots: Sequence[SourceRoot],
    *,
    worktree: WorktreeIdentity,
    required_roles: frozenset[SnapshotRole] = LEGACY_SNAPSHOT_ROLES,
) -> SnapshotPublication:
    """Atomically copy explicit roots into a fresh canonical read-only tree."""

    roles = _validate_required_roles(required_roles)
    target = destination.absolute()
    _reject_symlink_components(target.parent, "snapshot parent")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = _expand_source_roots(roots, required_roles=roles)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        entries: list[SnapshotEntry] = []
        for incomplete_entry, source in selected:
            before = source.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise SnapshotValidationError("source changed to a non-regular file")
            output = staging.joinpath(
                *PurePosixPath(incomplete_entry.relative_path).parts
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            copied_size = 0
            with source.open("rb") as source_stream, output.open("xb") as output_stream:
                while chunk := source_stream.read(_COPY_CHUNK_BYTES):
                    output_stream.write(chunk)
                    digest.update(chunk)
                    copied_size += len(chunk)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            after = source.stat(follow_symlinks=False)
            stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            observed = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if stable != observed or copied_size != before.st_size:
                raise SnapshotValidationError(f"source changed while copied: {source}")
            entries.append(
                SnapshotEntry(
                    role=incomplete_entry.role,
                    relative_path=incomplete_entry.relative_path,
                    size_bytes=copied_size,
                    sha256=digest.hexdigest(),
                )
            )
        manifest_payload = _manifest_payload(entries, worktree)
        manifest_bytes = canonical_json_bytes(manifest_payload)
        manifest_path = staging / SOURCE_MANIFEST_FILENAME
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _seal_tree(staging)
        publication = SnapshotPublication(
            root=target,
            manifest_path=target / SOURCE_MANIFEST_FILENAME,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            entries=tuple(entries),
            worktree=worktree,
        )
        staging.rename(target)
        try:
            parent_fd = os.open(
                target.parent,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise SnapshotPostPublicationError(publication, error) from error
        parent_error: OSError | None = None
        try:
            try:
                os.fsync(parent_fd)
            except OSError as error:
                parent_error = error
        finally:
            try:
                os.close(parent_fd)
            except OSError as error:
                if parent_error is None:
                    parent_error = error
        if parent_error is not None:
            raise SnapshotPostPublicationError(
                publication, parent_error
            ) from parent_error
        return publication
    except Exception:
        if staging.exists():
            staging.chmod(0o700)
            for path in (
                candidate for candidate in staging.rglob("*") if candidate.is_dir()
            ):
                path.chmod(0o700)
            shutil.rmtree(staging)
        raise


def copy_immutable_snapshot(
    destination: Path,
    source_root: Path,
    *,
    source_required_roles: frozenset[SnapshotRole],
    destination_required_roles: frozenset[SnapshotRole],
) -> SnapshotPublication:
    """Copy a sealed CPU snapshot into the exact narrower GPU role contract."""

    source_roles = _validate_required_roles(source_required_roles)
    destination_roles = _validate_required_roles(destination_required_roles)
    if (
        source_roles != DIAG4_CPU_SNAPSHOT_ROLES
        or destination_roles != DIAG4_GPU_SNAPSHOT_ROLES
    ):
        raise SnapshotValidationError("snapshot copy role transition is unsupported")
    source = load_snapshot(source_root, required_roles=source_roles)
    roots = tuple(
        SourceRoot(
            entry.role,
            source.root.joinpath(*PurePosixPath(entry.relative_path).parts),
            entry.relative_path,
        )
        for entry in source.entries
        if entry.role in destination_roles
    )
    publication = publish_immutable_snapshot(
        destination,
        roots,
        worktree=source.worktree,
        required_roles=destination_roles,
    )
    expected = tuple(
        entry for entry in source.entries if entry.role in destination_roles
    )
    if publication.entries != expected:
        raise SnapshotValidationError("copied snapshot entries differ from source")
    return publication


def validate_sealed_native_extension(
    snapshot: SnapshotPublication,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_relative_path: str,
) -> SnapshotEntry:
    """Reprove one copied native leaf independently of live-install topology."""

    if not _is_sha256(expected_sha256):
        raise SnapshotValidationError("expected native extension SHA-256 is invalid")
    if type(expected_size_bytes) is not int or expected_size_bytes < 0:
        raise SnapshotValidationError("expected native extension size is invalid")
    relative_path = _safe_relative_path(
        expected_relative_path, "expected native extension relative path"
    )
    if len(relative_path.parts) != 2 or relative_path.parts[0] != "native":
        raise SnapshotValidationError(
            "expected native extension relative path is invalid"
        )

    required_roles = _validate_required_roles(
        frozenset(entry.role for entry in snapshot.entries)
    )
    loaded = load_snapshot(snapshot.root, required_roles=required_roles)
    if loaded != snapshot:
        raise SnapshotValidationError("native extension snapshot identity differs")
    native_entries = tuple(
        entry for entry in loaded.entries if entry.role == "native_extension"
    )
    if len(native_entries) != 1:
        raise SnapshotValidationError(
            "snapshot must contain exactly one native extension"
        )
    entry = native_entries[0]
    if (
        entry.relative_path != relative_path.as_posix()
        or entry.sha256 != expected_sha256
        or entry.size_bytes != expected_size_bytes
    ):
        raise SnapshotValidationError("sealed native extension identity differs")

    path = loaded.root.joinpath(*PurePosixPath(entry.relative_path).parts)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        locked = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        locked_identity = (locked.st_dev, locked.st_ino, locked.st_size)
        bound_identity = (bound.st_dev, bound.st_ino, bound.st_size)
        if (
            not stat.S_ISREG(locked.st_mode)
            or not stat.S_ISREG(bound.st_mode)
            or locked_identity != bound_identity
            or locked.st_nlink != 1
            or bound.st_nlink != 1
            or stat.S_IMODE(locked.st_mode) != 0o444
            or stat.S_IMODE(bound.st_mode) != 0o444
            or locked.st_size != expected_size_bytes
            or _sha256_descriptor(descriptor) != expected_sha256
        ):
            raise SnapshotValidationError(
                "sealed native extension topology or bytes differ"
            )
    finally:
        os.close(descriptor)
    return entry


def copy_diag5_immutable_snapshot(
    destination: Path,
    source_root: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_native_relative_path: str,
) -> SnapshotPublication:
    """Atomically publish the DIAG5 GPU closure after validating its CPU source."""

    source = load_snapshot(source_root, required_roles=DIAG5_CPU_SNAPSHOT_ROLES)
    validate_sealed_native_extension(
        source,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        expected_relative_path=expected_native_relative_path,
    )
    publication = copy_immutable_snapshot(
        destination,
        source.root,
        source_required_roles=DIAG5_CPU_SNAPSHOT_ROLES,
        destination_required_roles=DIAG5_GPU_SNAPSHOT_ROLES,
    )
    validate_sealed_native_extension(
        publication,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        expected_relative_path=expected_native_relative_path,
    )
    return publication


def _load_json(payload: bytes, context: str) -> dict[str, object]:
    try:
        value = load_canonical_json_bytes(payload)
    except SnapshotValidationError as error:
        raise SnapshotValidationError(f"{context}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SnapshotValidationError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        raise SnapshotValidationError(f"{context} keys do not match schema")


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotValidationError(f"{context} must be a non-empty string")
    return value


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SnapshotValidationError(f"{context} must be an object")
    return value


def _array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise SnapshotValidationError(f"{context} must be an array")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SnapshotValidationError(f"{context} must be a nonnegative integer")
    return value


def _parse_worktree(value: object) -> WorktreeIdentity:
    if not isinstance(value, dict):
        raise SnapshotValidationError("manifest.worktree must be an object")
    _exact_keys(
        value,
        frozenset(
            {
                "git_head",
                "repo_root",
                "tracked_diff_sha256",
                "untracked_bytes_manifest_sha256",
            }
        ),
        "manifest.worktree",
    )
    git_head = _string(value["git_head"], "manifest.worktree.git_head")
    if len(git_head) != 40 or any(
        character not in _LOWER_HEX for character in git_head
    ):
        raise SnapshotValidationError("manifest.worktree.git_head is invalid")
    tracked = value["tracked_diff_sha256"]
    untracked = value["untracked_bytes_manifest_sha256"]
    if not _is_sha256(tracked) or not _is_sha256(untracked):
        raise SnapshotValidationError("manifest worktree digest is invalid")
    repo_root = _string(value["repo_root"], "manifest.worktree.repo_root")
    if not Path(repo_root).is_absolute():
        raise SnapshotValidationError("manifest.worktree.repo_root must be absolute")
    return WorktreeIdentity(
        git_head=git_head,
        tracked_diff_sha256=str(tracked),
        untracked_bytes_manifest_sha256=str(untracked),
        repo_root=repo_root,
    )


def load_snapshot(
    snapshot_root: Path,
    *,
    required_roles: frozenset[SnapshotRole] = LEGACY_SNAPSHOT_ROLES,
) -> SnapshotPublication:
    """Revalidate the manifest, exact file set, digests, roles, and read-only seal."""

    expected_roles = _validate_required_roles(required_roles)
    root = _strict_directory(snapshot_root, "snapshot root")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SnapshotValidationError("snapshot tree contains a symlink")
    manifest_path = root / SOURCE_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SnapshotValidationError("snapshot manifest is absent or non-regular")
    payload = manifest_path.read_bytes()
    document = _load_json(payload, "source manifest")
    _exact_keys(
        document,
        frozenset({"entries", "schema_version", "worktree"}),
        "source manifest",
    )
    if document["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise SnapshotValidationError("source manifest schema version is unsupported")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise SnapshotValidationError("source manifest entries must be an array")
    entries: list[SnapshotEntry] = []
    paths: list[str] = []
    roles: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        context = f"source manifest entries[{index}]"
        if not isinstance(raw_entry, dict):
            raise SnapshotValidationError(f"{context} must be an object")
        _exact_keys(
            raw_entry,
            frozenset({"relative_path", "role", "sha256", "size_bytes"}),
            context,
        )
        role = _string(raw_entry["role"], f"{context}.role")
        if role not in expected_roles:
            raise SnapshotValidationError(f"{context}.role is unsupported")
        relative = str(
            _safe_relative_path(
                _string(raw_entry["relative_path"], f"{context}.relative_path"), context
            )
        )
        size = _integer(raw_entry["size_bytes"], f"{context}.size_bytes")
        digest = raw_entry["sha256"]
        if not _is_sha256(digest):
            raise SnapshotValidationError(f"{context}.sha256 is invalid")
        path = root.joinpath(*PurePosixPath(relative).parts)
        _reject_symlink_components(path, context)
        if (
            not path.is_file()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise SnapshotValidationError(
                f"snapshot file {relative!r} differs from manifest"
            )
        if path.stat().st_mode & 0o222:
            raise SnapshotValidationError(f"snapshot file {relative!r} is writable")
        entries.append(SnapshotEntry(role, relative, size, str(digest)))
        paths.append(relative)
        roles.add(role)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SnapshotValidationError("source manifest paths are not sorted and unique")
    if roles != expected_roles:
        raise SnapshotValidationError("source manifest omits a required role")
    actual_files = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        )
    )
    if actual_files != tuple(paths):
        raise SnapshotValidationError("snapshot contains unmanifested or missing files")
    if root.stat().st_mode & 0o222 or manifest_path.stat().st_mode & 0o222:
        raise SnapshotValidationError("snapshot root or manifest is writable")
    if any(path.stat().st_mode & 0o222 for path in root.rglob("*") if path.is_dir()):
        raise SnapshotValidationError("snapshot contains a writable directory")
    return SnapshotPublication(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_bytes(payload),
        entries=tuple(entries),
        worktree=_parse_worktree(document["worktree"]),
    )


def effective_environment(
    environment: Mapping[str, str],
) -> tuple[tuple[str, str | None], ...]:
    """Return the exact non-secret environment subset that affects execution."""

    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise SnapshotValidationError("process environment must contain only strings")
    return tuple((key, environment.get(key)) for key in _ENVIRONMENT_KEYS)


def _module_origin(module: ModuleType) -> Path:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise SnapshotValidationError(f"module {module.__name__!r} has no file origin")
    path = Path(origin)
    _reject_symlink_components(path, f"module {module.__name__!r} origin")
    if not path.is_file():
        raise SnapshotValidationError(
            f"module {module.__name__!r} origin is non-regular"
        )
    return path.resolve(strict=True)


def _bind_imports(snapshot: SnapshotPublication) -> tuple[ImportBinding, ...]:
    entry_by_path = {entry.relative_path: entry for entry in snapshot.entries}
    bindings: list[ImportBinding] = []
    for module_name in _IMPORT_MODULES:
        module = importlib.import_module(module_name)
        origin = _module_origin(module)
        try:
            relative = origin.relative_to(snapshot.root).as_posix()
        except ValueError as error:
            raise SnapshotValidationError(
                f"module {module_name!r} resolved outside the immutable snapshot"
            ) from error
        entry = entry_by_path.get(relative)
        if entry is None or _sha256_file(origin) != entry.sha256:
            raise SnapshotValidationError(
                f"module {module_name!r} is not bound to manifested bytes"
            )
        bindings.append(
            ImportBinding(module_name, relative, entry.size_bytes, entry.sha256)
        )
    return tuple(bindings)


def _validate_installed_native_extension(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_link_count: int,
) -> Path:
    """Descriptor-bind one installed native extension to an expected identity."""

    if not path.is_absolute() or path != path.resolve(strict=True):
        raise SnapshotValidationError(
            "installed native extension path is not absolute and resolved"
        )
    if not _is_sha256(expected_sha256):
        raise SnapshotValidationError("installed native extension digest is invalid")
    if expected_size_bytes <= 0 or expected_link_count <= 0:
        raise SnapshotValidationError(
            "installed native extension size and link count must be positive"
        )
    _reject_symlink_components(path, "installed native extension path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        observed = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (observed.st_dev, observed.st_ino) != (named.st_dev, named.st_ino)
            or observed.st_size != expected_size_bytes
            or observed.st_nlink != expected_link_count
            or _sha256_descriptor(descriptor) != expected_sha256
        ):
            raise SnapshotValidationError(
                "installed native extension topology or bytes differ"
            )
    finally:
        os.close(descriptor)
    return path


def _bind_imports_v2(
    snapshot: SnapshotPublication,
    *,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
) -> tuple[ImportBinding, ...]:
    """Bind snapshot Python imports and the separate installed native import."""

    entry_by_path = {entry.relative_path: entry for entry in snapshot.entries}
    bindings: list[ImportBinding] = []
    for module_name in _IMPORT_MODULES[:-1]:
        origin = _module_origin(importlib.import_module(module_name))
        try:
            relative = origin.relative_to(snapshot.root).as_posix()
        except ValueError as error:
            raise SnapshotValidationError(
                f"module {module_name!r} resolved outside the immutable snapshot"
            ) from error
        entry = entry_by_path.get(relative)
        if (
            entry is None
            or entry.role != "execution_source"
            or _sha256_file(origin) != entry.sha256
        ):
            raise SnapshotValidationError(
                f"module {module_name!r} is not bound to manifested bytes"
            )
        bindings.append(
            ImportBinding(module_name, relative, entry.size_bytes, entry.sha256)
        )

    native_origin = _module_origin(importlib.import_module("simsoptpp"))
    expected_native = _validate_installed_native_extension(
        expected_native_extension_path,
        expected_sha256=expected_native_extension_sha256,
        expected_size_bytes=expected_native_extension_size_bytes,
        expected_link_count=expected_native_extension_link_count,
    )
    if native_origin != expected_native:
        raise SnapshotValidationError(
            "loaded native extension path differs from expected installed path"
        )
    native_entries = tuple(
        entry for entry in snapshot.entries if entry.role == "native_extension"
    )
    if (
        len(native_entries) != 1
        or native_entries[0].sha256 != expected_native_extension_sha256
        or native_entries[0].size_bytes != expected_native_extension_size_bytes
    ):
        raise SnapshotValidationError(
            "installed native extension differs from snapshot evidence copy"
        )
    native_entry = native_entries[0]
    snapshot_native = snapshot.root.joinpath(
        *PurePosixPath(native_entry.relative_path).parts
    )
    if native_origin == snapshot_native:
        raise SnapshotValidationError(
            "installed native extension path aliases the snapshot evidence copy"
        )
    bindings.append(
        ImportBinding(
            "simsoptpp",
            native_entry.relative_path,
            native_entry.size_bytes,
            native_entry.sha256,
        )
    )
    return tuple(bindings)


def _bind_entrypoint(
    snapshot: SnapshotPublication, argv: Sequence[str], runtime_cwd: Path
) -> ImportBinding:
    entrypoint = Path(argv[0])
    if not entrypoint.is_absolute():
        entrypoint = runtime_cwd / entrypoint
    _reject_symlink_components(entrypoint, "runtime entrypoint")
    origin = entrypoint.resolve(strict=True)
    if not origin.is_file():
        raise SnapshotValidationError("runtime entrypoint is not a regular file")
    try:
        relative = origin.relative_to(snapshot.root).as_posix()
    except ValueError as error:
        raise SnapshotValidationError(
            "runtime entrypoint resolved outside the immutable snapshot"
        ) from error
    entry = next(
        (
            candidate
            for candidate in snapshot.entries
            if candidate.relative_path == relative
        ),
        None,
    )
    if (
        entry is None
        or entry.role != "benchmark"
        or _sha256_file(origin) != entry.sha256
    ):
        raise SnapshotValidationError(
            "runtime entrypoint is not bound to manifested benchmark bytes"
        )
    return ImportBinding("__entrypoint__", relative, entry.size_bytes, entry.sha256)


def _physical_gpu(
    environment: Mapping[str, str], runner: CommandRunner
) -> tuple[str, str, str]:
    completed = runner(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ),
        environment,
    )
    if completed.returncode != 0:
        raise SnapshotValidationError(
            f"nvidia-smi identity query failed: {completed.stderr.strip()}"
        )
    return select_physical_gpu_identity(
        completed.stdout, environment.get("CUDA_VISIBLE_DEVICES", "")
    )


def select_physical_gpu_identity(
    nvidia_smi_stdout: str, visible_devices: str
) -> tuple[str, str, str]:
    """Resolve one stable physical UUID, model, and driver from live SMI rows."""

    rows: list[tuple[str, str, str, str]] = []
    for line in nvidia_smi_stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 4 or any(not field for field in fields):
            raise SnapshotValidationError("nvidia-smi emitted a malformed identity row")
        rows.append((fields[0], fields[1], fields[2], fields[3]))
    visible = visible_devices.strip()
    if "," in visible:
        raise SnapshotValidationError("runtime identity requires one visible GPU")
    if not visible:
        selected = rows
    elif visible.startswith("GPU-"):
        selected = [row for row in rows if row[1] == visible]
    elif visible.isdecimal():
        selected = [row for row in rows if row[0] == visible]
    else:
        raise SnapshotValidationError(
            "CUDA_VISIBLE_DEVICES is not an index or GPU UUID"
        )
    if len(selected) != 1:
        raise SnapshotValidationError(
            "runtime identity did not resolve exactly one physical GPU"
        )
    _index, uuid, name, driver = selected[0]
    if not uuid.startswith("GPU-"):
        raise SnapshotValidationError("physical GPU identity has no stable UUID")
    return uuid, name, driver


def observe_live_runtime(
    snapshot_root: Path,
    *,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    command_runner: CommandRunner = _run_command,
    required_roles: frozenset[SnapshotRole] = LEGACY_SNAPSHOT_ROLES,
) -> RuntimeObservation:
    """Observe the claim-bearing process after JAX selected exactly one GPU."""

    snapshot = load_snapshot(snapshot_root, required_roles=required_roles)
    if not argv or not all(isinstance(value, str) and value for value in argv):
        raise SnapshotValidationError("runtime argv must contain non-empty strings")
    runtime_cwd = _strict_directory(cwd, "runtime cwd")
    entrypoint_binding = _bind_entrypoint(snapshot, argv, runtime_cwd)
    bindings = _bind_imports(snapshot)
    jax_module = importlib.import_module("jax")
    jaxlib_module = importlib.import_module("jaxlib")
    backend = jax_module.default_backend()
    devices = tuple(jax_module.devices())
    if backend != "gpu" or len(devices) != 1 or devices[0].platform != "gpu":
        raise SnapshotValidationError("runtime is not bound to exactly one JAX GPU")
    device_uuid, device_name, driver_version = _physical_gpu(
        environment, command_runner
    )
    env = effective_environment(environment)
    env_payload = {key: value for key, value in env}
    binding_by_module = {binding.module: binding for binding in bindings}

    def path_for(module: str) -> str:
        return str(snapshot.root / binding_by_module[module].relative_path)

    runtime_identity = RuntimeIdentity(
        argv=tuple(argv),
        cwd=str(runtime_cwd),
        python_executable=os.path.abspath(sys.executable),
        python_version=sys.version,
        jax_version=_string(getattr(jax_module, "__version__", None), "JAX version"),
        jaxlib_version=_string(
            getattr(jaxlib_module, "__version__", None), "JAXLIB version"
        ),
        simsopt_module_path=path_for("simsopt"),
        simsopt_jax_module_path=path_for("simsopt_jax"),
        native_extension_path=path_for("simsoptpp"),
        backend=backend,
        device_uuid=device_uuid,
        driver_version=driver_version,
        effective_environment_sha256=_sha256_bytes(canonical_json_bytes(env_payload)),
    )
    platform_version = _string(
        getattr(devices[0].client, "platform_version", None),
        "JAX device platform version",
    )
    return RuntimeObservation(
        runtime_identity=runtime_identity,
        entrypoint_binding=entrypoint_binding,
        import_bindings=bindings,
        effective_environment=env,
        device_name=device_name,
        platform_version=platform_version,
    )


def observe_live_runtime_v2(
    snapshot_root: Path,
    *,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
    command_runner: CommandRunner = _run_command,
    required_roles: frozenset[SnapshotRole] = DIAG5_GPU_SNAPSHOT_ROLES,
) -> RuntimeObservationV2:
    """Observe DIAG5 with snapshot Python and an authority-bound native import."""

    snapshot = load_snapshot(snapshot_root, required_roles=required_roles)
    if not argv or not all(isinstance(value, str) and value for value in argv):
        raise SnapshotValidationError("runtime argv must contain non-empty strings")
    runtime_cwd = _strict_directory(cwd, "runtime cwd")
    entrypoint_binding = _bind_entrypoint(snapshot, argv, runtime_cwd)
    bindings = _bind_imports_v2(
        snapshot,
        expected_native_extension_path=expected_native_extension_path,
        expected_native_extension_sha256=expected_native_extension_sha256,
        expected_native_extension_size_bytes=expected_native_extension_size_bytes,
        expected_native_extension_link_count=expected_native_extension_link_count,
    )
    jax_module = importlib.import_module("jax")
    jaxlib_module = importlib.import_module("jaxlib")
    backend = jax_module.default_backend()
    devices = tuple(jax_module.devices())
    if backend != "gpu" or len(devices) != 1 or devices[0].platform != "gpu":
        raise SnapshotValidationError("runtime is not bound to exactly one JAX GPU")
    device_uuid, device_name, driver_version = _physical_gpu(
        environment, command_runner
    )
    env = effective_environment(environment)
    env_payload = {key: value for key, value in env}
    binding_by_module = {binding.module: binding for binding in bindings}

    def snapshot_path_for(module: str) -> str:
        return str(snapshot.root / binding_by_module[module].relative_path)

    runtime_identity = RuntimeIdentityV2(
        argv=tuple(argv),
        cwd=str(runtime_cwd),
        python_executable=os.path.abspath(sys.executable),
        python_version=sys.version,
        jax_version=_string(getattr(jax_module, "__version__", None), "JAX version"),
        jaxlib_version=_string(
            getattr(jaxlib_module, "__version__", None), "JAXLIB version"
        ),
        simsopt_module_path=snapshot_path_for("simsopt"),
        simsopt_jax_module_path=snapshot_path_for("simsopt_jax"),
        native_extension_path=str(expected_native_extension_path),
        backend=backend,
        device_uuid=device_uuid,
        driver_version=driver_version,
        effective_environment_sha256=_sha256_bytes(canonical_json_bytes(env_payload)),
        native_extension_sha256=expected_native_extension_sha256,
        native_extension_size_bytes=expected_native_extension_size_bytes,
        native_extension_link_count=expected_native_extension_link_count,
    )
    platform_version = _string(
        getattr(devices[0].client, "platform_version", None),
        "JAX device platform version",
    )
    observation = RuntimeObservationV2(
        runtime_identity=runtime_identity,
        entrypoint_binding=entrypoint_binding,
        import_bindings=bindings,
        effective_environment=env,
        device_name=device_name,
        platform_version=platform_version,
    )
    _validate_observation_v2(
        snapshot,
        observation,
        expected_native_extension_path=expected_native_extension_path,
        expected_native_extension_sha256=expected_native_extension_sha256,
        expected_native_extension_size_bytes=expected_native_extension_size_bytes,
        expected_native_extension_link_count=expected_native_extension_link_count,
    )
    return observation


def build_runtime_evidence(
    snapshot_root: Path,
    *,
    source_identity: SourceIdentity,
    observation: RuntimeObservation,
    required_roles: frozenset[SnapshotRole] = LEGACY_SNAPSHOT_ROLES,
) -> RuntimeEvidence:
    """Validate and bind one live observation to its source snapshot."""

    snapshot = load_snapshot(snapshot_root, required_roles=required_roles)
    if (
        source_identity.git_head != snapshot.worktree.git_head
        or (
            source_identity.tracked_diff_sha256 != snapshot.worktree.tracked_diff_sha256
        )
        or (
            source_identity.untracked_bytes_manifest_sha256
            != snapshot.worktree.untracked_bytes_manifest_sha256
        )
        or source_identity.repo_root != snapshot.worktree.repo_root
    ):
        raise SnapshotValidationError("source identity differs from snapshot worktree")
    manifest_ref = source_identity.snapshot_manifest
    manifest_payload = snapshot.manifest_path.read_bytes()
    if (
        manifest_ref.sha256 != snapshot.manifest_sha256
        or manifest_ref.size_bytes != len(manifest_payload)
        or manifest_ref.schema_version != SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise SnapshotValidationError("source identity manifest reference differs")
    _validate_observation(snapshot, observation)
    return RuntimeEvidence(source_identity, observation, snapshot.manifest_sha256)


def build_runtime_evidence_v2(
    snapshot_root: Path,
    *,
    source_identity: SourceIdentity,
    observation: RuntimeObservationV2,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
    required_roles: frozenset[SnapshotRole] = DIAG5_GPU_SNAPSHOT_ROLES,
) -> RuntimeEvidenceV2:
    """Validate DIAG5 source, live native identity, and runtime observation."""

    snapshot = load_snapshot(snapshot_root, required_roles=required_roles)
    if (
        source_identity.git_head != snapshot.worktree.git_head
        or source_identity.tracked_diff_sha256 != snapshot.worktree.tracked_diff_sha256
        or source_identity.untracked_bytes_manifest_sha256
        != snapshot.worktree.untracked_bytes_manifest_sha256
        or source_identity.repo_root != snapshot.worktree.repo_root
    ):
        raise SnapshotValidationError("source identity differs from snapshot worktree")
    manifest_ref = source_identity.snapshot_manifest
    manifest_payload = snapshot.manifest_path.read_bytes()
    if (
        manifest_ref.sha256 != snapshot.manifest_sha256
        or manifest_ref.size_bytes != len(manifest_payload)
        or manifest_ref.schema_version != SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise SnapshotValidationError("source identity manifest reference differs")
    _validate_observation_v2(
        snapshot,
        observation,
        expected_native_extension_path=expected_native_extension_path,
        expected_native_extension_sha256=expected_native_extension_sha256,
        expected_native_extension_size_bytes=expected_native_extension_size_bytes,
        expected_native_extension_link_count=expected_native_extension_link_count,
    )
    return RuntimeEvidenceV2(source_identity, observation, snapshot.manifest_sha256)


def _validate_observation(
    snapshot: SnapshotPublication, observation: RuntimeObservation
) -> None:
    runtime = observation.runtime_identity
    if runtime.backend != "gpu" or not runtime.device_uuid.startswith("GPU-"):
        raise SnapshotValidationError("runtime identity is not a physical GPU identity")
    if not runtime.argv or any(not value for value in runtime.argv):
        raise SnapshotValidationError("runtime identity argv is empty")
    if (
        not Path(runtime.cwd).is_absolute()
        or not Path(runtime.python_executable).is_absolute()
    ):
        raise SnapshotValidationError("runtime cwd and interpreter must be absolute")
    if any(
        not value
        for value in (
            runtime.cwd,
            runtime.python_executable,
            runtime.python_version,
            runtime.jax_version,
            runtime.jaxlib_version,
            runtime.driver_version,
            observation.device_name,
            observation.platform_version,
        )
    ):
        raise SnapshotValidationError("runtime identity contains an empty fact")
    if not _is_sha256(runtime.effective_environment_sha256):
        raise SnapshotValidationError("effective environment digest is invalid")
    environment = dict(observation.effective_environment)
    if tuple(environment) != _ENVIRONMENT_KEYS or len(environment) != len(
        observation.effective_environment
    ):
        raise SnapshotValidationError("effective environment keys are not canonical")
    if runtime.effective_environment_sha256 != _sha256_bytes(
        canonical_json_bytes(environment)
    ):
        raise SnapshotValidationError("effective environment digest differs")
    entry_by_path = {entry.relative_path: entry for entry in snapshot.entries}
    entrypoint = observation.entrypoint_binding
    entrypoint_entry = entry_by_path.get(entrypoint.relative_path)
    if (
        entrypoint.module != "__entrypoint__"
        or entrypoint_entry is None
        or entrypoint_entry.role != "benchmark"
        or entrypoint.size_bytes != entrypoint_entry.size_bytes
        or entrypoint.sha256 != entrypoint_entry.sha256
    ):
        raise SnapshotValidationError(
            "runtime entrypoint binding differs from manifested benchmark bytes"
        )
    if (
        tuple(binding.module for binding in observation.import_bindings)
        != _IMPORT_MODULES
    ):
        raise SnapshotValidationError(
            "runtime import bindings are incomplete or reordered"
        )
    for binding in observation.import_bindings:
        entry = entry_by_path.get(binding.relative_path)
        if (
            entry is None
            or binding.size_bytes != entry.size_bytes
            or binding.sha256 != entry.sha256
        ):
            raise SnapshotValidationError(
                f"runtime import binding for {binding.module!r} differs from manifest"
            )
        expected_role = (
            "native_extension" if binding.module == "simsoptpp" else "execution_source"
        )
        if entry.role != expected_role:
            raise SnapshotValidationError(
                f"runtime import binding for {binding.module!r} has the wrong role"
            )
    paths = {
        "simsopt": runtime.simsopt_module_path,
        "simsopt_jax": runtime.simsopt_jax_module_path,
        "simsoptpp": runtime.native_extension_path,
    }
    bindings = {binding.module: binding for binding in observation.import_bindings}
    recorded_roots: set[Path] = set()
    for module, absolute_path in paths.items():
        observed_path = Path(absolute_path)
        if not observed_path.is_absolute():
            raise SnapshotValidationError(f"runtime path for {module!r} is relative")
        relative = PurePosixPath(bindings[module].relative_path)
        if tuple(observed_path.parts[-len(relative.parts) :]) != relative.parts:
            raise SnapshotValidationError(f"runtime path for {module!r} differs")
        recorded_roots.add(observed_path.parents[len(relative.parts) - 1])
    if len(recorded_roots) != 1 or Path(runtime.cwd) != next(iter(recorded_roots)):
        raise SnapshotValidationError(
            "runtime imports and cwd do not share one recorded snapshot root"
        )
    recorded_root = next(iter(recorded_roots))
    argv_entrypoint = Path(runtime.argv[0])
    if not argv_entrypoint.is_absolute():
        argv_entrypoint = recorded_root / argv_entrypoint
    expected_entrypoint = recorded_root.joinpath(
        *PurePosixPath(entrypoint.relative_path).parts
    )
    if Path(os.path.normpath(argv_entrypoint)) != expected_entrypoint:
        raise SnapshotValidationError(
            "runtime argv entrypoint differs from the manifested entrypoint"
        )


def _validate_observation_v2(
    snapshot: SnapshotPublication,
    observation: RuntimeObservationV2,
    *,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
) -> None:
    """Reprove v1 runtime facts plus DIAG5's separate installed native binding."""

    expected_native = _validate_installed_native_extension(
        expected_native_extension_path,
        expected_sha256=expected_native_extension_sha256,
        expected_size_bytes=expected_native_extension_size_bytes,
        expected_link_count=expected_native_extension_link_count,
    )
    _validate_observation_v2_against_expected_binding(
        snapshot,
        observation,
        expected_native_extension_path=expected_native,
        expected_native_extension_sha256=expected_native_extension_sha256,
        expected_native_extension_size_bytes=expected_native_extension_size_bytes,
        expected_native_extension_link_count=expected_native_extension_link_count,
    )


def _validate_observation_v2_against_expected_binding(
    snapshot: SnapshotPublication,
    observation: RuntimeObservationV2,
    *,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
) -> None:
    """Join runtime facts to an independently descriptor-validated native binding."""

    runtime = observation.runtime_identity
    native_binding = next(
        (
            binding
            for binding in observation.import_bindings
            if binding.module == "simsoptpp"
        ),
        None,
    )
    if native_binding is None:
        raise SnapshotValidationError("runtime import bindings omit simsoptpp")
    snapshot_native = snapshot.root.joinpath(
        *PurePosixPath(native_binding.relative_path).parts
    )
    legacy_runtime = RuntimeIdentity(
        argv=runtime.argv,
        cwd=runtime.cwd,
        python_executable=runtime.python_executable,
        python_version=runtime.python_version,
        jax_version=runtime.jax_version,
        jaxlib_version=runtime.jaxlib_version,
        simsopt_module_path=runtime.simsopt_module_path,
        simsopt_jax_module_path=runtime.simsopt_jax_module_path,
        native_extension_path=str(snapshot_native),
        backend=runtime.backend,
        device_uuid=runtime.device_uuid,
        driver_version=runtime.driver_version,
        effective_environment_sha256=runtime.effective_environment_sha256,
    )
    _validate_observation(
        snapshot,
        RuntimeObservation(
            runtime_identity=legacy_runtime,
            entrypoint_binding=observation.entrypoint_binding,
            import_bindings=observation.import_bindings,
            effective_environment=observation.effective_environment,
            device_name=observation.device_name,
            platform_version=observation.platform_version,
        ),
    )

    if (
        runtime.native_extension_path != str(expected_native_extension_path)
        or runtime.native_extension_sha256 != expected_native_extension_sha256
        or runtime.native_extension_size_bytes != expected_native_extension_size_bytes
        or runtime.native_extension_link_count != expected_native_extension_link_count
    ):
        raise SnapshotValidationError(
            "runtime native extension identity differs from expected authority"
        )
    if expected_native_extension_path == snapshot_native:
        raise SnapshotValidationError(
            "installed native extension path aliases the snapshot evidence copy"
        )
    if (
        native_binding.sha256 != expected_native_extension_sha256
        or native_binding.size_bytes != expected_native_extension_size_bytes
    ):
        raise SnapshotValidationError(
            "runtime native extension differs from snapshot evidence copy"
        )


def publish_runtime_evidence(
    path: Path,
    evidence: RuntimeEvidence,
    *,
    snapshot_root: Path,
    campaign_root: Path,
    required_roles: frozenset[SnapshotRole] = LEGACY_SNAPSHOT_ROLES,
) -> ArtifactRef:
    """Exclusively publish canonical runtime evidence after full revalidation."""

    campaign = _strict_directory(campaign_root, "campaign root")
    parent = _strict_directory(path.parent, "runtime evidence parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    absolute_path = parent / path.name
    try:
        relative_path = absolute_path.relative_to(campaign).as_posix()
    except ValueError as error:
        raise SnapshotValidationError(
            "runtime evidence must be inside the campaign root"
        ) from error
    build_runtime_evidence(
        snapshot_root,
        source_identity=evidence.source_identity,
        observation=evidence.observation,
        required_roles=required_roles,
    )
    payload = canonical_json_bytes(evidence.to_payload())
    with absolute_path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return ArtifactRef(
        relative_path=relative_path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        schema_version=RUNTIME_EVIDENCE_SCHEMA_VERSION,
    )


def publish_runtime_evidence_v2(
    path: Path,
    evidence: RuntimeEvidenceV2,
    *,
    snapshot_root: Path,
    campaign_root: Path,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
    required_roles: frozenset[SnapshotRole] = DIAG5_GPU_SNAPSHOT_ROLES,
) -> ArtifactRef:
    """Exclusively publish DIAG5 evidence after live native revalidation."""

    campaign = _strict_directory(campaign_root, "campaign root")
    parent = _strict_directory(path.parent, "runtime evidence parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    absolute_path = parent / path.name
    try:
        relative_path = absolute_path.relative_to(campaign).as_posix()
    except ValueError as error:
        raise SnapshotValidationError(
            "runtime evidence must be inside the campaign root"
        ) from error
    build_runtime_evidence_v2(
        snapshot_root,
        source_identity=evidence.source_identity,
        observation=evidence.observation,
        expected_native_extension_path=expected_native_extension_path,
        expected_native_extension_sha256=expected_native_extension_sha256,
        expected_native_extension_size_bytes=expected_native_extension_size_bytes,
        expected_native_extension_link_count=expected_native_extension_link_count,
        required_roles=required_roles,
    )
    payload = canonical_json_bytes(evidence.to_payload())
    with absolute_path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return ArtifactRef(
        relative_path=relative_path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        schema_version=RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
    )


def _artifact_ref(value: object, context: str) -> ArtifactRef:
    mapping = _mapping(value, context)
    _exact_keys(
        mapping,
        frozenset({"relative_path", "schema_version", "sha256", "size_bytes"}),
        context,
    )
    relative_path = str(
        _safe_relative_path(
            _string(mapping["relative_path"], f"{context}.relative_path"),
            f"{context}.relative_path",
        )
    )
    digest = mapping["sha256"]
    if not _is_sha256(digest):
        raise SnapshotValidationError(f"{context}.sha256 is invalid")
    return ArtifactRef(
        relative_path=relative_path,
        schema_version=_string(mapping["schema_version"], f"{context}.schema_version"),
        sha256=str(digest),
        size_bytes=_integer(mapping["size_bytes"], f"{context}.size_bytes"),
    )


def source_identity_from_payload(value: object) -> SourceIdentity:
    """Parse the exact static identity schema used by run receipts."""

    mapping = _mapping(value, "runtime evidence source_identity")
    _exact_keys(
        mapping,
        frozenset(
            {
                "git_head",
                "repo_root",
                "snapshot_manifest",
                "tracked_diff_sha256",
                "untracked_bytes_manifest_sha256",
            }
        ),
        "runtime evidence source_identity",
    )
    git_head = _string(mapping["git_head"], "source_identity.git_head")
    if len(git_head) != 40 or any(
        character not in _LOWER_HEX for character in git_head
    ):
        raise SnapshotValidationError("source_identity.git_head is invalid")
    tracked = mapping["tracked_diff_sha256"]
    untracked = mapping["untracked_bytes_manifest_sha256"]
    if not _is_sha256(tracked) or not _is_sha256(untracked):
        raise SnapshotValidationError("source identity digest is invalid")
    return SourceIdentity(
        snapshot_manifest=_artifact_ref(
            mapping["snapshot_manifest"], "source_identity.snapshot_manifest"
        ),
        git_head=git_head,
        tracked_diff_sha256=str(tracked),
        untracked_bytes_manifest_sha256=str(untracked),
        repo_root=_string(mapping["repo_root"], "source_identity.repo_root"),
    )


def runtime_identity_from_payload(value: object) -> RuntimeIdentity:
    """Parse the exact dynamic identity schema used by run receipts."""

    mapping = _mapping(value, "runtime evidence runtime_identity")
    expected = frozenset(
        {
            "argv",
            "backend",
            "cwd",
            "device_uuid",
            "driver_version",
            "effective_environment_sha256",
            "jax_version",
            "jaxlib_version",
            "native_extension_path",
            "python_executable",
            "python_version",
            "simsopt_jax_module_path",
            "simsopt_module_path",
        }
    )
    _exact_keys(mapping, expected, "runtime evidence runtime_identity")
    argv = _array(mapping["argv"], "runtime_identity.argv")
    checked_argv = tuple(
        _string(value, f"runtime_identity.argv[{index}]")
        for index, value in enumerate(argv)
    )
    return RuntimeIdentity(
        argv=checked_argv,
        cwd=_string(mapping["cwd"], "runtime_identity.cwd"),
        python_executable=_string(
            mapping["python_executable"], "runtime_identity.python_executable"
        ),
        python_version=_string(
            mapping["python_version"], "runtime_identity.python_version"
        ),
        jax_version=_string(mapping["jax_version"], "runtime_identity.jax_version"),
        jaxlib_version=_string(
            mapping["jaxlib_version"], "runtime_identity.jaxlib_version"
        ),
        simsopt_module_path=_string(
            mapping["simsopt_module_path"], "runtime_identity.simsopt_module_path"
        ),
        simsopt_jax_module_path=_string(
            mapping["simsopt_jax_module_path"],
            "runtime_identity.simsopt_jax_module_path",
        ),
        native_extension_path=_string(
            mapping["native_extension_path"],
            "runtime_identity.native_extension_path",
        ),
        backend=_string(mapping["backend"], "runtime_identity.backend"),
        device_uuid=_string(mapping["device_uuid"], "runtime_identity.device_uuid"),
        driver_version=_string(
            mapping["driver_version"], "runtime_identity.driver_version"
        ),
        effective_environment_sha256=_string(
            mapping["effective_environment_sha256"],
            "runtime_identity.effective_environment_sha256",
        ),
    )


def runtime_identity_v2_from_payload(value: object) -> RuntimeIdentityV2:
    """Parse the exact DIAG5 dynamic identity without accepting v1 aliases."""

    mapping = _mapping(value, "runtime evidence runtime_identity")
    expected = frozenset(
        {
            "argv",
            "backend",
            "cwd",
            "device_uuid",
            "driver_version",
            "effective_environment_sha256",
            "jax_version",
            "jaxlib_version",
            "native_extension_link_count",
            "native_extension_path",
            "native_extension_sha256",
            "native_extension_size_bytes",
            "python_executable",
            "python_version",
            "simsopt_jax_module_path",
            "simsopt_module_path",
        }
    )
    _exact_keys(mapping, expected, "runtime evidence runtime_identity")
    argv = _array(mapping["argv"], "runtime_identity.argv")
    digest = mapping["native_extension_sha256"]
    if not _is_sha256(digest):
        raise SnapshotValidationError("runtime native extension digest is invalid")
    size_bytes = _integer(
        mapping["native_extension_size_bytes"],
        "runtime_identity.native_extension_size_bytes",
    )
    link_count = _integer(
        mapping["native_extension_link_count"],
        "runtime_identity.native_extension_link_count",
    )
    if size_bytes == 0 or link_count == 0:
        raise SnapshotValidationError(
            "runtime native extension size and link count must be positive"
        )
    return RuntimeIdentityV2(
        argv=tuple(
            _string(item, f"runtime_identity.argv[{index}]")
            for index, item in enumerate(argv)
        ),
        cwd=_string(mapping["cwd"], "runtime_identity.cwd"),
        python_executable=_string(
            mapping["python_executable"], "runtime_identity.python_executable"
        ),
        python_version=_string(
            mapping["python_version"], "runtime_identity.python_version"
        ),
        jax_version=_string(mapping["jax_version"], "runtime_identity.jax_version"),
        jaxlib_version=_string(
            mapping["jaxlib_version"], "runtime_identity.jaxlib_version"
        ),
        simsopt_module_path=_string(
            mapping["simsopt_module_path"], "runtime_identity.simsopt_module_path"
        ),
        simsopt_jax_module_path=_string(
            mapping["simsopt_jax_module_path"],
            "runtime_identity.simsopt_jax_module_path",
        ),
        native_extension_path=_string(
            mapping["native_extension_path"],
            "runtime_identity.native_extension_path",
        ),
        backend=_string(mapping["backend"], "runtime_identity.backend"),
        device_uuid=_string(mapping["device_uuid"], "runtime_identity.device_uuid"),
        driver_version=_string(
            mapping["driver_version"], "runtime_identity.driver_version"
        ),
        effective_environment_sha256=_string(
            mapping["effective_environment_sha256"],
            "runtime_identity.effective_environment_sha256",
        ),
        native_extension_sha256=str(digest),
        native_extension_size_bytes=size_bytes,
        native_extension_link_count=link_count,
    )


def _import_binding(value: object, index: int) -> ImportBinding:
    context = f"runtime evidence import_bindings[{index}]"
    mapping = _mapping(value, context)
    _exact_keys(
        mapping,
        frozenset({"module", "relative_path", "sha256", "size_bytes"}),
        context,
    )
    digest = mapping["sha256"]
    if not _is_sha256(digest):
        raise SnapshotValidationError(f"{context}.sha256 is invalid")
    return ImportBinding(
        module=_string(mapping["module"], f"{context}.module"),
        relative_path=str(
            _safe_relative_path(
                _string(mapping["relative_path"], f"{context}.relative_path"),
                f"{context}.relative_path",
            )
        ),
        size_bytes=_integer(mapping["size_bytes"], f"{context}.size_bytes"),
        sha256=str(digest),
    )


def validate_runtime_evidence(
    path: Path,
    *,
    snapshot_root: Path,
    campaign_root: Path,
    required_roles: frozenset[SnapshotRole] = LEGACY_SNAPSHOT_ROLES,
) -> RuntimeEvidence:
    """Parse canonical runtime evidence and reprove every source/runtime relation."""

    campaign = _strict_directory(campaign_root, "campaign root")
    _reject_symlink_components(path, "runtime evidence path")
    runtime_path = path.resolve(strict=True)
    if not runtime_path.is_file() or not runtime_path.is_relative_to(campaign):
        raise SnapshotValidationError(
            "runtime evidence is not a regular campaign-local file"
        )
    document = _load_json(runtime_path.read_bytes(), "runtime evidence")
    _exact_keys(
        document,
        frozenset(
            {
                "device",
                "effective_environment",
                "entrypoint_binding",
                "import_bindings",
                "runtime_identity",
                "schema_version",
                "snapshot_manifest_sha256",
                "source_identity",
            }
        ),
        "runtime evidence",
    )
    if document["schema_version"] != RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise SnapshotValidationError("runtime evidence schema version is unsupported")
    manifest_sha = document["snapshot_manifest_sha256"]
    if not _is_sha256(manifest_sha):
        raise SnapshotValidationError("runtime evidence manifest digest is invalid")
    source = source_identity_from_payload(document["source_identity"])
    source_manifest_path = source.snapshot_manifest.resolve_and_validate(campaign)
    snapshot = load_snapshot(snapshot_root, required_roles=required_roles)
    if source_manifest_path != snapshot.manifest_path:
        raise SnapshotValidationError(
            "runtime evidence names a different source manifest"
        )
    environment_mapping = _mapping(
        document["effective_environment"], "runtime evidence effective_environment"
    )
    if tuple(environment_mapping) != _ENVIRONMENT_KEYS:
        raise SnapshotValidationError(
            "runtime evidence environment keys are not canonical"
        )
    environment: list[tuple[str, str | None]] = []
    for key, value in environment_mapping.items():
        if value is not None and not isinstance(value, str):
            raise SnapshotValidationError(
                f"runtime evidence effective_environment.{key} is invalid"
            )
        environment.append((key, value))
    raw_bindings = _array(
        document["import_bindings"], "runtime evidence import_bindings"
    )
    bindings = tuple(
        _import_binding(value, index) for index, value in enumerate(raw_bindings)
    )
    entrypoint_binding = _import_binding(document["entrypoint_binding"], -1)
    device = _mapping(document["device"], "runtime evidence device")
    _exact_keys(
        device,
        frozenset({"name", "platform_version"}),
        "runtime evidence device",
    )
    evidence = RuntimeEvidence(
        source_identity=source,
        observation=RuntimeObservation(
            runtime_identity=runtime_identity_from_payload(
                document["runtime_identity"]
            ),
            entrypoint_binding=entrypoint_binding,
            import_bindings=bindings,
            effective_environment=tuple(environment),
            device_name=_string(device["name"], "runtime evidence device.name"),
            platform_version=_string(
                device["platform_version"],
                "runtime evidence device.platform_version",
            ),
        ),
        snapshot_manifest_sha256=str(manifest_sha),
    )
    rebuilt = build_runtime_evidence(
        snapshot.root,
        source_identity=evidence.source_identity,
        observation=evidence.observation,
        required_roles=required_roles,
    )
    if rebuilt.snapshot_manifest_sha256 != evidence.snapshot_manifest_sha256:
        raise SnapshotValidationError("runtime evidence snapshot digest differs")
    return evidence


def validate_runtime_evidence_v2(
    path: Path,
    *,
    snapshot_root: Path,
    campaign_root: Path,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
    required_roles: frozenset[SnapshotRole] = DIAG5_GPU_SNAPSHOT_ROLES,
) -> RuntimeEvidenceV2:
    """Deep-load DIAG5 evidence and reprove its live installed native binding."""

    campaign = _strict_directory(campaign_root, "campaign root")
    _reject_symlink_components(path, "runtime evidence path")
    runtime_path = path.resolve(strict=True)
    if not runtime_path.is_file() or not runtime_path.is_relative_to(campaign):
        raise SnapshotValidationError(
            "runtime evidence is not a regular campaign-local file"
        )
    evidence = _runtime_evidence_v2_from_bytes(runtime_path.read_bytes())
    source_manifest_path = (
        evidence.source_identity.snapshot_manifest.resolve_and_validate(campaign)
    )
    snapshot = load_snapshot(snapshot_root, required_roles=required_roles)
    if source_manifest_path != snapshot.manifest_path:
        raise SnapshotValidationError(
            "runtime evidence names a different source manifest"
        )
    rebuilt = build_runtime_evidence_v2(
        snapshot.root,
        source_identity=evidence.source_identity,
        observation=evidence.observation,
        expected_native_extension_path=expected_native_extension_path,
        expected_native_extension_sha256=expected_native_extension_sha256,
        expected_native_extension_size_bytes=expected_native_extension_size_bytes,
        expected_native_extension_link_count=expected_native_extension_link_count,
        required_roles=required_roles,
    )
    if rebuilt.snapshot_manifest_sha256 != evidence.snapshot_manifest_sha256:
        raise SnapshotValidationError("runtime evidence snapshot digest differs")
    return evidence


def _runtime_evidence_v2_from_bytes(payload: bytes) -> RuntimeEvidenceV2:
    """Parse canonical DIAG5 runtime bytes without resolving artifact paths."""

    document = _load_json(payload, "runtime evidence")
    _exact_keys(
        document,
        frozenset(
            {
                "device",
                "effective_environment",
                "entrypoint_binding",
                "import_bindings",
                "runtime_identity",
                "schema_version",
                "snapshot_manifest_sha256",
                "source_identity",
            }
        ),
        "runtime evidence",
    )
    if document["schema_version"] != RUNTIME_EVIDENCE_V2_SCHEMA_VERSION:
        raise SnapshotValidationError("runtime evidence schema version is unsupported")
    manifest_sha = document["snapshot_manifest_sha256"]
    if not _is_sha256(manifest_sha):
        raise SnapshotValidationError("runtime evidence manifest digest is invalid")
    source = source_identity_from_payload(document["source_identity"])
    environment_mapping = _mapping(
        document["effective_environment"], "runtime evidence effective_environment"
    )
    if tuple(environment_mapping) != _ENVIRONMENT_KEYS:
        raise SnapshotValidationError(
            "runtime evidence environment keys are not canonical"
        )
    environment: list[tuple[str, str | None]] = []
    for key, value in environment_mapping.items():
        if value is not None and not isinstance(value, str):
            raise SnapshotValidationError(
                f"runtime evidence effective_environment.{key} is invalid"
            )
        environment.append((key, value))
    bindings = tuple(
        _import_binding(value, index)
        for index, value in enumerate(
            _array(document["import_bindings"], "runtime evidence import_bindings")
        )
    )
    entrypoint_binding = _import_binding(document["entrypoint_binding"], -1)
    device = _mapping(document["device"], "runtime evidence device")
    _exact_keys(
        device,
        frozenset({"name", "platform_version"}),
        "runtime evidence device",
    )
    return RuntimeEvidenceV2(
        source_identity=source,
        observation=RuntimeObservationV2(
            runtime_identity=runtime_identity_v2_from_payload(
                document["runtime_identity"]
            ),
            entrypoint_binding=entrypoint_binding,
            import_bindings=bindings,
            effective_environment=tuple(environment),
            device_name=_string(device["name"], "runtime evidence device.name"),
            platform_version=_string(
                device["platform_version"],
                "runtime evidence device.platform_version",
            ),
        ),
        snapshot_manifest_sha256=str(manifest_sha),
    )


def validate_diag5_runtime_evidence_v2_bytes(
    payload: bytes,
    *,
    expected_snapshot_identity: SnapshotIdentity,
    expected_logical_campaign_root: Path,
    expected_logical_snapshot_root: Path,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
    expected_native_extension_link_count: int,
) -> RuntimeEvidenceV2:
    """Validate held runtime bytes against pathless source and logical-root identity."""

    campaign_root = Path(os.path.normpath(expected_logical_campaign_root))
    snapshot_root = Path(os.path.normpath(expected_logical_snapshot_root))
    native_extension_path = Path(os.path.normpath(expected_native_extension_path))
    if (
        not expected_logical_campaign_root.is_absolute()
        or not expected_logical_snapshot_root.is_absolute()
        or campaign_root != expected_logical_campaign_root
        or snapshot_root != expected_logical_snapshot_root
        or not snapshot_root.is_relative_to(campaign_root)
    ):
        raise SnapshotValidationError("logical runtime roots are invalid")
    if (
        not expected_native_extension_path.is_absolute()
        or native_extension_path != expected_native_extension_path
    ):
        raise SnapshotValidationError("expected native extension path is invalid")
    required_roles = _validate_required_roles(
        frozenset(entry.role for entry in expected_snapshot_identity.entries)
    )
    if required_roles != DIAG5_GPU_SNAPSHOT_ROLES:
        raise SnapshotValidationError("runtime snapshot identity roles differ")
    canonical_identity = build_snapshot_identity(
        expected_snapshot_identity.entries,
        expected_snapshot_identity.worktree,
    )
    if canonical_identity != expected_snapshot_identity:
        raise SnapshotValidationError("runtime snapshot identity is noncanonical")

    evidence = _runtime_evidence_v2_from_bytes(payload)
    manifest_bytes = canonical_json_bytes(
        _manifest_payload(
            expected_snapshot_identity.entries,
            expected_snapshot_identity.worktree,
        )
    )
    logical_manifest = snapshot_root / SOURCE_MANIFEST_FILENAME
    expected_manifest_ref = ArtifactRef(
        relative_path=logical_manifest.relative_to(campaign_root).as_posix(),
        sha256=expected_snapshot_identity.manifest_sha256,
        size_bytes=len(manifest_bytes),
        schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
    )
    if (
        evidence.source_identity.snapshot_manifest != expected_manifest_ref
        or evidence.source_identity.git_head
        != expected_snapshot_identity.worktree.git_head
        or evidence.source_identity.tracked_diff_sha256
        != expected_snapshot_identity.worktree.tracked_diff_sha256
        or evidence.source_identity.untracked_bytes_manifest_sha256
        != expected_snapshot_identity.worktree.untracked_bytes_manifest_sha256
        or evidence.source_identity.repo_root
        != expected_snapshot_identity.worktree.repo_root
        or evidence.snapshot_manifest_sha256
        != expected_snapshot_identity.manifest_sha256
    ):
        raise SnapshotValidationError(
            "runtime evidence source identity differs from expected snapshot"
        )
    logical_snapshot = SnapshotPublication(
        root=snapshot_root,
        manifest_path=logical_manifest,
        manifest_sha256=expected_snapshot_identity.manifest_sha256,
        entries=expected_snapshot_identity.entries,
        worktree=expected_snapshot_identity.worktree,
    )
    _validate_observation_v2_against_expected_binding(
        logical_snapshot,
        evidence.observation,
        expected_native_extension_path=native_extension_path,
        expected_native_extension_sha256=expected_native_extension_sha256,
        expected_native_extension_size_bytes=expected_native_extension_size_bytes,
        expected_native_extension_link_count=expected_native_extension_link_count,
    )
    return evidence


__all__ = (
    "DIAG4_CPU_SNAPSHOT_ROLES",
    "DIAG4_GPU_SNAPSHOT_ROLES",
    "DIAG5_CPU_SNAPSHOT_ROLES",
    "DIAG5_GPU_SNAPSHOT_ROLES",
    "DIAG5_PREQUALIFICATION_PLAN_RELATIVE_PATH",
    "LEGACY_SNAPSHOT_ROLES",
    "RUNTIME_EVIDENCE_FILENAME",
    "RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "RUNTIME_EVIDENCE_V2_SCHEMA_VERSION",
    "SOURCE_MANIFEST_FILENAME",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "ArtifactRef",
    "ImportBinding",
    "JsonValue",
    "RuntimeEvidence",
    "RuntimeEvidenceV2",
    "RuntimeIdentity",
    "RuntimeIdentityV2",
    "RuntimeObservation",
    "RuntimeObservationV2",
    "SnapshotEntry",
    "SnapshotIdentity",
    "SnapshotPostPublicationError",
    "SnapshotPublication",
    "SnapshotRole",
    "SnapshotValidationError",
    "SourceIdentity",
    "SourceRoot",
    "WorktreeIdentity",
    "build_runtime_evidence",
    "build_runtime_evidence_v2",
    "build_snapshot_identity",
    "capture_worktree_identity",
    "copy_diag5_immutable_snapshot",
    "copy_immutable_snapshot",
    "effective_environment",
    "load_canonical_json_bytes",
    "load_snapshot",
    "observe_live_runtime",
    "observe_live_runtime_v2",
    "project_diag5_gpu_snapshot_identity",
    "publish_immutable_snapshot",
    "publish_runtime_evidence",
    "publish_runtime_evidence_v2",
    "runtime_identity_from_payload",
    "runtime_identity_v2_from_payload",
    "select_physical_gpu_identity",
    "source_identity_from_payload",
    "validate_diag5_runtime_evidence_v2_bytes",
    "validate_runtime_evidence",
    "validate_runtime_evidence_v2",
    "validate_sealed_native_extension",
)
