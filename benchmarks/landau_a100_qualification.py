"""Fail-closed Landau A100 qualification receipt producer.

The collector is intended to run inside an existing Slurm allocation on Landau.
It never opens a network connection.  Command execution and the environment are
injectable so the complete qualification policy can be tested without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmarks.single_stage_compute_graph_snapshot import (
    MANIFEST_FILENAME,
    SOURCE_MANIFEST_SCHEMA_ID,
    SnapshotError,
    load_snapshot_manifest,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    canonical_json_bytes as snapshot_canonical_json_bytes,
)

QUALIFICATION_SCHEMA_ID: Final = "landau-a100-qualification-v5"
OBSERVATION_SCHEMA_ID: Final = "landau-a100-observations-v5"
REQUIRED_LINEAX_VERSION: Final = "0.1.1"
SPECIMEN_DOCUMENT_RELATIVE_PATH: Final = "phase0-specimen/specimen.json"
_NUMERICAL_POLICY_FIELDS: Final = frozenset(
    {
        "dense_batch_width",
        "point_chunk_size",
        "coil_chunk_size",
        "quadrature_block_sizes",
    }
)
REQUIRED_SLURM_ENVIRONMENT: Final = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "CUDA_VISIBLE_DEVICES",
)
SCHEDULER_GPU_ACCOUNTING_ENVIRONMENT: Final = "SLURM_JOB_GPUS"
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_LOCK_REQUIREMENT_PATTERN: Final = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")
_PACKAGE_SEPARATOR_PATTERN: Final = re.compile(r"[-_.]+")
_NATIVE_HASH_PATTERN: Final = re.compile(r"^# sha256\([^)]*simsoptpp[^)]*\) =\s*$")
_CUDA_LIBRARY_NAMES: Final = (
    "libcuda",
    "libcudart",
    "libcublas",
    "libcusolver",
    "libcusparse",
    "libcufft",
    "libcudnn",
    "libnvjitlink",
)


class QualificationInputError(ValueError):
    """Injected observations do not satisfy the observation schema."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result of one local command."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], CommandResult]


def _run_command(argv: Sequence[str], environment: Mapping[str, str]) -> CommandResult:
    completed = subprocess.run(
        tuple(argv),
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment),
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_package_name(name: str) -> str:
    return _PACKAGE_SEPARATOR_PATTERN.sub("-", name).lower()


def _command_document(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> dict[str, object]:
    result = runner(tuple(argv), environment)
    return {
        "argv": list(argv),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _command_stdout(command: Mapping[str, object]) -> str:
    stdout = command.get("stdout")
    return stdout if isinstance(stdout, str) else ""


def _parse_lock(path: Path) -> tuple[dict[str, str], str | None]:
    requirements: dict[str, str] = {}
    native_hash: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    expect_native_hash = False
    for line in lines:
        stripped = line.strip()
        if _NATIVE_HASH_PATTERN.fullmatch(stripped):
            expect_native_hash = True
            continue
        if expect_native_hash and stripped.startswith("#"):
            candidate = stripped.removeprefix("#").strip()
            if _SHA256_PATTERN.fullmatch(candidate):
                native_hash = candidate
                expect_native_hash = False
                continue
        match = _LOCK_REQUIREMENT_PATTERN.fullmatch(stripped)
        if match is not None:
            requirements[_canonical_package_name(match.group(1))] = match.group(2)
    requirements["lineax"] = REQUIRED_LINEAX_VERSION
    return dict(sorted(requirements.items())), native_hash


_PROBE_SOURCE: Final = (
    r"""import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

snapshot_root = Path(os.environ["SIMSOPT_PHASE0_SNAPSHOT_ROOT"]).resolve()
sys.path.insert(0, str(snapshot_root))
sys.path.insert(0, str(snapshot_root / "src"))
allowed_finders = (
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    importlib.machinery.PathFinder,
)
sys.meta_path[:] = [finder for finder in sys.meta_path if finder in allowed_finders]

import jax
import jax.numpy as jnp
import numpy as np
import simsopt
import simsopt_jax
import simsopt_jax_adapters
import simsoptpp
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_static_timing_environment,
    observe_effective_numerical_policies,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(module):
    path = Path(module.__file__).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


packages = {
    re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata["Name"]
}
seed = np.asarray([1.25, -2.0, 3.5], dtype=np.float64)
device_seed = jax.device_put(seed)
result = jax.jit(lambda value: jnp.sum(value * value))(device_seed)
result.block_until_ready()
host_result = np.asarray(jax.device_get(result))
resolved_libraries = []
for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
    fields = line.split()
    if not fields or not fields[-1].startswith("/"):
        continue
    path = Path(fields[-1]).resolve()
    if not path.is_file() or not any(name in path.name for name in """
    + repr(_CUDA_LIBRARY_NAMES)
    + r"""):
        continue
    resolved_libraries.append(
        {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
    )
unique_libraries = {row["path"]: row for row in resolved_libraries}
jax_devices = jax.devices()
devices = [
    {"id": device.id, "platform": device.platform, "kind": device.device_kind}
    for device in jax_devices
]
platform_version = str(jax_devices[0].client.platform_version) if jax_devices else ""
print(json.dumps({
    "interpreter": {
        "entrypoint_path": str(Path(sys.executable).absolute()),
        "target_path": str(Path(sys.executable).resolve()),
        "target_size": Path(sys.executable).resolve().stat().st_size,
        "target_sha256": sha256_file(Path(sys.executable).resolve()),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    },
    "jax": {
        "version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
        "platform_version": platform_version,
        "devices": devices,
    },
    "packages": dict(sorted(packages.items())),
    "imports": {
        "simsopt": file_identity(simsopt),
        "simsopt_jax": file_identity(simsopt_jax),
        "simsopt_jax_adapters": file_identity(simsopt_jax_adapters),
    },
    "native_binary": file_identity(simsoptpp),
    "resolved_cuda_libraries": [unique_libraries[key] for key in sorted(unique_libraries)],
    "static_timing_environment": normalize_static_timing_environment(os.environ),
    "effective_numerical_policies": observe_effective_numerical_policies(
        int(os.environ["SIMSOPT_PHASE0_QUADRATURE_NODES"])
    ),
    "smoke": {
        "transfer_guard": os.environ.get("JAX_TRANSFER_GUARD"),
        "input_dtype": str(device_seed.dtype),
        "output_dtype": str(result.dtype),
        "output_shape": list(result.shape),
        "finite": bool(np.all(np.isfinite(host_result))),
        "value": float(host_result),
    },
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
"""
)


def collect_landau_observations(
    *,
    snapshot_root: Path,
    python_executable: Path,
    overlay_lock: Path,
    environment: Mapping[str, str],
    source_provenance: Mapping[str, object] | None = None,
    runner: CommandRunner = _run_command,
    cpu_affinity: Sequence[int] | None = None,
) -> dict[str, object]:
    """Collect local Landau observations from one validated immutable snapshot."""

    resolved_snapshot = snapshot_root.resolve()
    if not python_executable.is_absolute():
        raise QualificationInputError("python executable path must be absolute")
    python_entrypoint = Path(os.path.abspath(python_executable))
    if not python_entrypoint.is_file() or not os.access(python_entrypoint, os.X_OK):
        raise QualificationInputError("python executable must be an executable file")
    python_target = python_entrypoint.resolve()
    resolved_lock = overlay_lock.resolve()
    try:
        manifest_entries, manifest_sha256 = load_snapshot_manifest(resolved_snapshot)
    except (OSError, SnapshotError) as error:
        raise QualificationInputError(
            f"immutable Phase 0 snapshot validation failed: {error}"
        ) from error
    if not resolved_lock.is_relative_to(resolved_snapshot):
        raise QualificationInputError(
            "overlay lock must be inside the immutable snapshot"
        )
    lock_relative_path = resolved_lock.relative_to(resolved_snapshot).as_posix()
    manifest_by_path = {entry.relative_path: entry for entry in manifest_entries}
    lock_entry = manifest_by_path.get(lock_relative_path)
    if lock_entry is None:
        raise QualificationInputError(
            "overlay lock is absent from phase0-source-manifest.json"
        )
    specimen_path = resolved_snapshot / SPECIMEN_DOCUMENT_RELATIVE_PATH
    specimen_entry = manifest_by_path.get(SPECIMEN_DOCUMENT_RELATIVE_PATH)
    if specimen_entry is None or specimen_entry.role != "configuration":
        raise QualificationInputError(
            "frozen specimen is absent from phase0-source-manifest.json"
        )
    specimen_document = json.loads(specimen_path.read_text(encoding="utf-8"))
    numerical_policies = _object(
        _object(specimen_document, "frozen specimen").get("effective_policies"),
        "frozen specimen effective_policies",
    )
    specimen_grids = _object(
        _object(specimen_document, "frozen specimen").get("specimen"),
        "frozen specimen payload",
    ).get("grids")
    quadrature_nodes = _object(specimen_grids, "frozen specimen grids").get(
        "quadrature_nodes"
    )
    if not isinstance(quadrature_nodes, int) or isinstance(quadrature_nodes, bool):
        raise QualificationInputError("frozen specimen quadrature_nodes is invalid")
    child_environment = dict(environment)
    child_environment["JAX_TRANSFER_GUARD"] = "disallow"
    child_environment["JAX_ENABLE_X64"] = "true"
    child_environment["JAX_PLATFORMS"] = "cuda"
    child_environment["SIMSOPT_PHASE0_SNAPSHOT_ROOT"] = str(resolved_snapshot)
    child_environment["SIMSOPT_PHASE0_QUADRATURE_NODES"] = str(quadrature_nodes)

    visible_device = child_environment.get("CUDA_VISIBLE_DEVICES", "")
    gpu_argv = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    if visible_device and "," not in visible_device:
        gpu_argv[1:1] = ["-i", visible_device]

    commands = {
        "hostname": _command_document(
            ("hostname", "--fqdn"), environment=child_environment, runner=runner
        ),
        "gpu": _command_document(
            gpu_argv, environment=child_environment, runner=runner
        ),
        "pip_freeze": _command_document(
            (str(python_entrypoint), "-m", "pip", "freeze"),
            environment=child_environment,
            runner=runner,
        ),
        "jax_probe": _command_document(
            (str(python_entrypoint), "-c", _PROBE_SOURCE),
            environment=child_environment,
            runner=runner,
        ),
    }
    if resolved_lock.is_file():
        expected_packages, expected_native_hash = _parse_lock(resolved_lock)
        lock_sha256: str | None = _sha256_file(resolved_lock)
    else:
        expected_packages = {"lineax": REQUIRED_LINEAX_VERSION}
        expected_native_hash = None
        lock_sha256 = None
    affinity = (
        tuple(sorted(os.sched_getaffinity(0)))
        if cpu_affinity is None
        else tuple(sorted(cpu_affinity))
    )
    return {
        "schema_id": OBSERVATION_SCHEMA_ID,
        "interpreter": {
            "entrypoint_path": str(python_entrypoint),
            "target_path": str(python_target),
            "target_size": python_target.stat().st_size,
            "target_sha256": _sha256_file(python_target),
        },
        "environment": {
            key: child_environment.get(key)
            for key in (
                *REQUIRED_SLURM_ENVIRONMENT,
                SCHEDULER_GPU_ACCOUNTING_ENVIRONMENT,
                "SLURM_STEP_ID",
                "LD_LIBRARY_PATH",
                "JAX_TRANSFER_GUARD",
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
            )
        },
        "commands": commands,
        "cpu_affinity": list(affinity),
        "source": {
            "snapshot_root": str(resolved_snapshot),
            "snapshot_manifest_path": str(resolved_snapshot / MANIFEST_FILENAME),
            "snapshot_manifest_schema_id": SOURCE_MANIFEST_SCHEMA_ID,
            "snapshot_manifest_sha256": manifest_sha256,
            "snapshot_manifest_entries": [
                entry.to_json() for entry in manifest_entries
            ],
            "external_provenance": (
                None if source_provenance is None else dict(source_provenance)
            ),
        },
        "overlay": {
            "lock_path": str(resolved_lock),
            "lock_sha256": lock_sha256,
            "lock_manifest_entry": lock_entry.to_json(),
            "expected_packages": expected_packages,
            "expected_native_binary_sha256": expected_native_hash,
            "actual_freeze": _command_stdout(commands["pip_freeze"]),
        },
        "numerical_policies": {
            "source_path": str(specimen_path),
            "source_sha256": specimen_entry.sha256,
            "source_manifest_entry": specimen_entry.to_json(),
            "declared": dict(numerical_policies),
        },
        "probe": _command_stdout(commands["jax_probe"]),
    }


def _object(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise QualificationInputError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise QualificationInputError(f"{field} must be a string")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise QualificationInputError(f"{field} must be an array")
    return value


def _parse_json_object(payload: str) -> Mapping[str, object]:
    if not payload.strip():
        return {}
    parsed = json.loads(payload)
    return _object(parsed, "probe")


def _freeze_packages(payload: str) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in payload.splitlines():
        match = _LOCK_REQUIREMENT_PATTERN.fullmatch(line.strip())
        if match is not None:
            packages[_canonical_package_name(match.group(1))] = match.group(2)
    return dict(sorted(packages.items()))


def _file_identity_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    path = value.get("path")
    size = value.get("size")
    sha256 = value.get("sha256")
    return (
        isinstance(path, str)
        and bool(path)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size > 0
        and isinstance(sha256, str)
        and _SHA256_PATTERN.fullmatch(sha256) is not None
    )


def _manifest_entries_by_path(
    source: Mapping[str, object], blockers: list[dict[str, str]]
) -> dict[str, Mapping[str, object]]:
    entries = _sequence(
        source.get("snapshot_manifest_entries", []),
        "source.snapshot_manifest_entries",
    )
    by_path: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(entries):
        entry = _object(value, f"source.snapshot_manifest_entries[{index}]")
        if set(entry) != {"role", "relative_path", "size_bytes", "sha256"}:
            _add_blocker(
                blockers,
                "SNAPSHOT_MANIFEST_ENTRY_INVALID",
                f"entry {index} has unexpected fields",
            )
            continue
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            _add_blocker(
                blockers,
                "SNAPSHOT_MANIFEST_ENTRY_INVALID",
                f"entry {index} has invalid relative_path",
            )
            continue
        if relative_path in by_path:
            _add_blocker(
                blockers,
                "SNAPSHOT_MANIFEST_ENTRY_INVALID",
                f"duplicate path {relative_path}",
            )
            continue
        size_bytes = entry.get("size_bytes")
        digest = entry.get("sha256")
        role = entry.get("role")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or role
            not in {
                "execution_source",
                "configuration",
                "benchmark",
                "test",
                "native_extension",
            }
        ):
            _add_blocker(
                blockers,
                "SNAPSHOT_MANIFEST_ENTRY_INVALID",
                relative_path,
            )
            continue
        by_path[relative_path] = entry
    return by_path


def _bind_file_to_snapshot_manifest(
    *,
    identity: Mapping[str, object],
    snapshot_root: Path,
    manifest_by_path: Mapping[str, Mapping[str, object]],
    context: str,
    blockers: list[dict[str, str]],
    expected_role: str,
) -> None:
    identity_path = Path(_string(identity.get("path"), f"{context}.path")).resolve()
    try:
        relative_path = identity_path.relative_to(snapshot_root).as_posix()
    except ValueError:
        _add_blocker(
            blockers,
            "IMPORT_OUTSIDE_IMMUTABLE_SNAPSHOT",
            f"{context}: {identity_path}",
        )
        return
    manifest_entry = manifest_by_path.get(relative_path)
    if manifest_entry is None:
        _add_blocker(
            blockers,
            "IMPORT_ABSENT_FROM_SNAPSHOT_MANIFEST",
            f"{context}: {relative_path}",
        )
        return
    if manifest_entry.get("role") != expected_role:
        _add_blocker(
            blockers,
            "IMPORT_SNAPSHOT_ROLE_MISMATCH",
            f"{context}: expected={expected_role}; actual={manifest_entry.get('role')}",
        )
    if identity.get("size") != manifest_entry.get("size_bytes") or identity.get(
        "sha256"
    ) != manifest_entry.get("sha256"):
        _add_blocker(
            blockers,
            "IMPORT_SNAPSHOT_IDENTITY_MISMATCH",
            f"{context}: {relative_path}",
        )


def _add_blocker(blockers: list[dict[str, str]], code: str, detail: str) -> None:
    blockers.append({"code": code, "detail": detail})


def _gpu_rows(payload: str) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        memory: int | None = None
        if fields[2].isdigit():
            memory = int(fields[2])
        rows.append(
            {
                "uuid": fields[0],
                "name": fields[1],
                "memory_mib": memory,
                "driver_version": fields[3],
            }
        )
    return tuple(rows)


def _compatibility_paths(environment: Mapping[str, object]) -> tuple[str, ...]:
    value = environment.get("LD_LIBRARY_PATH")
    if not isinstance(value, str):
        return ()
    return tuple(
        os.path.normpath(path)
        for path in value.split(":")
        if path and "compat" in path.lower()
    )


def _is_cuda_12_6_compat(path: str) -> bool:
    normalized = path.lower().replace("_", "-")
    return normalized.endswith("/cuda-12.6/compat")


def _is_unsupported_compat(path: str) -> bool:
    normalized = path.lower().replace("_", "-")
    return any(version in normalized for version in ("12.8", "12-8", "12.9", "12-9"))


def build_landau_qualification_receipt(
    observations: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate injected observations and return canonical PASS/BLOCKED evidence."""

    if observations.get("schema_id") != OBSERVATION_SCHEMA_ID:
        raise QualificationInputError("unsupported observation schema")
    environment = _object(observations.get("environment"), "environment")
    commands = _object(observations.get("commands"), "commands")
    collected_interpreter = _object(observations.get("interpreter"), "interpreter")
    source = _object(observations.get("source"), "source")
    overlay = _object(observations.get("overlay"), "overlay")
    numerical_policies = _object(
        observations.get("numerical_policies"), "numerical_policies"
    )
    blockers: list[dict[str, str]] = []

    for name in REQUIRED_SLURM_ENVIRONMENT:
        value = environment.get(name)
        if not isinstance(value, str) or not value.strip():
            _add_blocker(blockers, "SLURM_IDENTITY_MISSING", name)
    job_id = environment.get("SLURM_JOB_ID")
    if isinstance(job_id, str) and job_id and not job_id.isdigit():
        _add_blocker(blockers, "SLURM_JOB_ID_INVALID", job_id)
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if isinstance(visible, str) and "," in visible:
        _add_blocker(blockers, "MULTIPLE_VISIBLE_GPUS", visible)
    scheduler_gpu_value = environment.get(SCHEDULER_GPU_ACCOUNTING_ENVIRONMENT)
    scheduler_gpu_accounting = {
        "environment_key": SCHEDULER_GPU_ACCOUNTING_ENVIRONMENT,
        "state": (
            "reported"
            if isinstance(scheduler_gpu_value, str) and scheduler_gpu_value.strip()
            else "unavailable_not_configured"
        ),
        "value": scheduler_gpu_value,
    }

    failed_commands: list[str] = []
    for name, value in commands.items():
        command = _object(value, f"commands.{name}")
        returncode = command.get("returncode")
        if returncode != 0:
            failed_commands.append(name)
            _add_blocker(
                blockers,
                "COLLECTION_COMMAND_FAILED",
                f"{name}: returncode={returncode}",
            )

    hostname = _command_stdout(
        _object(commands.get("hostname"), "commands.hostname")
    ).strip()
    if not hostname:
        _add_blocker(
            blockers, "HOST_IDENTITY_MISSING", "hostname --fqdn returned empty"
        )

    gpu_rows = _gpu_rows(_command_stdout(_object(commands.get("gpu"), "commands.gpu")))
    if len(gpu_rows) != 1:
        _add_blocker(blockers, "PHYSICAL_GPU_COUNT_INVALID", str(len(gpu_rows)))
    elif "A100" not in _string(gpu_rows[0]["name"], "gpu.name").upper():
        _add_blocker(blockers, "PHYSICAL_GPU_NOT_A100", str(gpu_rows[0]["name"]))
    if len(gpu_rows) == 1:
        gpu = gpu_rows[0]
        if not str(gpu["uuid"]).startswith("GPU-"):
            _add_blocker(blockers, "PHYSICAL_GPU_UUID_INVALID", str(gpu["uuid"]))
        memory = gpu["memory_mib"]
        if not isinstance(memory, int) or memory < 38_000:
            _add_blocker(blockers, "PHYSICAL_GPU_MEMORY_INVALID", str(memory))
        if not str(gpu["driver_version"]).strip():
            _add_blocker(blockers, "DRIVER_IDENTITY_MISSING", "empty driver version")

    compat_paths = _compatibility_paths(environment)
    cuda_12_6_paths = tuple(path for path in compat_paths if _is_cuda_12_6_compat(path))
    unsupported_paths = tuple(
        path for path in compat_paths if _is_unsupported_compat(path)
    )
    if len(cuda_12_6_paths) != 1:
        _add_blocker(
            blockers,
            "CUDA_12_6_COMPAT_PATH_INVALID",
            f"expected one path, observed {list(cuda_12_6_paths)}",
        )
    if len(compat_paths) != 1 or compat_paths != cuda_12_6_paths:
        _add_blocker(
            blockers,
            "MIXED_CUDA_COMPAT_PATHS",
            f"observed {list(compat_paths)}",
        )
    if unsupported_paths:
        _add_blocker(
            blockers,
            "UNSUPPORTED_CUDA_COMPAT_PATH",
            ",".join(unsupported_paths),
        )

    probe_command = _object(commands.get("jax_probe"), "commands.jax_probe")
    try:
        probe = _parse_json_object(_command_stdout(probe_command))
    except (json.JSONDecodeError, QualificationInputError) as error:
        probe = {}
        _add_blocker(blockers, "JAX_PROBE_OUTPUT_INVALID", str(error))
    probe_interpreter = _object(probe.get("interpreter", {}), "probe.interpreter")
    expected_interpreter_keys = {
        "entrypoint_path",
        "target_path",
        "target_size",
        "target_sha256",
    }
    if set(collected_interpreter) != expected_interpreter_keys:
        _add_blocker(
            blockers,
            "INTERPRETER_IDENTITY_INCOMPLETE",
            f"observed fields={sorted(collected_interpreter)}",
        )
    entrypoint_path = collected_interpreter.get("entrypoint_path")
    target_path = collected_interpreter.get("target_path")
    target_size = collected_interpreter.get("target_size")
    target_sha256 = collected_interpreter.get("target_sha256")
    collected_identity_valid = (
        isinstance(entrypoint_path, str)
        and Path(entrypoint_path).is_absolute()
        and os.path.abspath(entrypoint_path) == entrypoint_path
        and isinstance(target_path, str)
        and Path(target_path).is_absolute()
        and isinstance(target_size, int)
        and not isinstance(target_size, bool)
        and target_size > 0
        and isinstance(target_sha256, str)
        and _SHA256_PATTERN.fullmatch(target_sha256) is not None
    )
    if not collected_identity_valid:
        _add_blocker(
            blockers,
            "INTERPRETER_IDENTITY_INVALID",
            json.dumps(dict(collected_interpreter), sort_keys=True),
        )
    probe_entrypoint = probe_interpreter.get("entrypoint_path")
    probe_target = {
        "target_path": probe_interpreter.get("target_path"),
        "target_size": probe_interpreter.get("target_size"),
        "target_sha256": probe_interpreter.get("target_sha256"),
    }
    collected_target = {
        "target_path": target_path,
        "target_size": target_size,
        "target_sha256": target_sha256,
    }
    if probe_entrypoint != entrypoint_path or probe_target != collected_target:
        _add_blocker(
            blockers,
            "INTERPRETER_PROBE_IDENTITY_MISMATCH",
            f"collected={dict(collected_interpreter)}; probe={dict(probe_interpreter)}",
        )
    prefix = probe_interpreter.get("prefix")
    base_prefix = probe_interpreter.get("base_prefix")
    if (
        not isinstance(prefix, str)
        or not prefix
        or not isinstance(base_prefix, str)
        or not base_prefix
        or prefix == base_prefix
    ):
        _add_blocker(
            blockers,
            "VIRTUALENV_IDENTITY_INVALID",
            f"prefix={prefix!r}; base_prefix={base_prefix!r}",
        )
    resolved_libraries = _sequence(
        probe.get("resolved_cuda_libraries", []), "resolved_cuda_libraries"
    )
    valid_libraries = tuple(
        row for row in resolved_libraries if _file_identity_valid(row)
    )
    if len(valid_libraries) != len(resolved_libraries) or not valid_libraries:
        _add_blocker(
            blockers,
            "CUDA_LIBRARY_IDENTITY_INCOMPLETE",
            "resolved CUDA libraries require path, size, and sha256",
        )
    libcuda_paths = tuple(
        _string(_object(row, "cuda library").get("path"), "cuda library path")
        for row in valid_libraries
        if Path(
            _string(_object(row, "cuda library").get("path"), "cuda library path")
        ).name.startswith("libcuda.so")
    )
    if len(cuda_12_6_paths) == 1 and (
        len(libcuda_paths) != 1
        or not libcuda_paths[0].startswith(cuda_12_6_paths[0].rstrip("/") + "/")
    ):
        _add_blocker(
            blockers,
            "CUDA_DRIVER_LIBRARY_NOT_FROM_12_6_COMPAT",
            f"resolved libcuda paths: {list(libcuda_paths)}",
        )

    jax_identity = _object(probe.get("jax", {}), "jax")
    if jax_identity.get("backend") != "gpu":
        _add_blocker(blockers, "JAX_BACKEND_NOT_GPU", str(jax_identity.get("backend")))
    if jax_identity.get("x64_enabled") is not True:
        _add_blocker(blockers, "JAX_X64_DISABLED", str(jax_identity.get("x64_enabled")))
    for field in ("version", "jaxlib_version"):
        if not isinstance(jax_identity.get(field), str) or not jax_identity.get(field):
            _add_blocker(blockers, "JAX_IDENTITY_INCOMPLETE", field)
    platform_version = jax_identity.get("platform_version")
    if not isinstance(platform_version, str) or not platform_version.strip():
        _add_blocker(
            blockers,
            "CUDA_RUNTIME_PLATFORM_IDENTITY_MISSING",
            str(platform_version),
        )
    devices = _sequence(jax_identity.get("devices", []), "jax.devices")
    if len(devices) != 1:
        _add_blocker(blockers, "JAX_DEVICE_COUNT_INVALID", str(len(devices)))
    elif "A100" not in str(_object(devices[0], "jax device").get("kind", "")).upper():
        _add_blocker(blockers, "JAX_DEVICE_NOT_A100", str(devices[0]))

    smoke = _object(probe.get("smoke", {}), "smoke")
    smoke_passed = (
        smoke.get("transfer_guard") == "disallow"
        and smoke.get("input_dtype") == "float64"
        and smoke.get("output_dtype") == "float64"
        and smoke.get("finite") is True
        and smoke.get("output_shape") == []
    )
    if not smoke_passed:
        _add_blocker(
            blockers,
            "STRICT_TRANSFER_FP64_SMOKE_FAILED",
            json.dumps(smoke, sort_keys=True),
        )

    expected_packages = _object(overlay.get("expected_packages"), "expected_packages")
    actual_packages = _freeze_packages(
        _string(overlay.get("actual_freeze"), "actual_freeze")
    )
    normalized_expected = {
        _string(name, "package name"): _string(version, "package version")
        for name, version in expected_packages.items()
    }
    if actual_packages != normalized_expected:
        missing = sorted(set(normalized_expected) - set(actual_packages))
        extra = sorted(set(actual_packages) - set(normalized_expected))
        mismatched = sorted(
            name
            for name in set(actual_packages) & set(normalized_expected)
            if actual_packages[name] != normalized_expected[name]
        )
        _add_blocker(
            blockers,
            "DEPENDENCY_OVERLAY_MISMATCH",
            f"missing={missing}; extra={extra}; mismatched={mismatched}",
        )
    if actual_packages.get("lineax") != REQUIRED_LINEAX_VERSION:
        _add_blocker(
            blockers,
            "LINEAX_VERSION_INVALID",
            str(actual_packages.get("lineax")),
        )
    probe_packages = _object(probe.get("packages", {}), "probe.packages")
    mismatched_probe_packages = sorted(
        name
        for name, version in normalized_expected.items()
        if probe_packages.get(name) != version
    )
    if mismatched_probe_packages:
        _add_blocker(
            blockers,
            "PROBED_DEPENDENCY_IDENTITY_MISMATCH",
            ",".join(mismatched_probe_packages),
        )
    lock_sha256 = overlay.get("lock_sha256")
    if (
        not isinstance(lock_sha256, str)
        or _SHA256_PATTERN.fullmatch(lock_sha256) is None
    ):
        _add_blocker(blockers, "OVERLAY_LOCK_IDENTITY_INVALID", str(lock_sha256))
    expected_native_hash = overlay.get("expected_native_binary_sha256")
    if (
        not isinstance(expected_native_hash, str)
        or _SHA256_PATTERN.fullmatch(expected_native_hash) is None
    ):
        _add_blocker(
            blockers,
            "EXPECTED_NATIVE_BINARY_IDENTITY_MISSING",
            str(expected_native_hash),
        )

    snapshot_root = Path(
        _string(source.get("snapshot_root"), "source.snapshot_root")
    ).resolve()
    expected_manifest_path = snapshot_root / MANIFEST_FILENAME
    if source.get("snapshot_manifest_path") != str(expected_manifest_path):
        _add_blocker(
            blockers,
            "SNAPSHOT_MANIFEST_PATH_INVALID",
            str(source.get("snapshot_manifest_path")),
        )
    if source.get("snapshot_manifest_schema_id") != SOURCE_MANIFEST_SCHEMA_ID:
        _add_blocker(
            blockers,
            "SNAPSHOT_MANIFEST_SCHEMA_INVALID",
            str(source.get("snapshot_manifest_schema_id")),
        )
    manifest_sha256 = source.get("snapshot_manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256_PATTERN.fullmatch(manifest_sha256) is None
    ):
        _add_blocker(
            blockers,
            "SNAPSHOT_MANIFEST_HASH_INVALID",
            str(manifest_sha256),
        )
    manifest_by_path = _manifest_entries_by_path(source, blockers)
    raw_manifest_entries = source.get("snapshot_manifest_entries", [])
    if (
        isinstance(manifest_sha256, str)
        and _SHA256_PATTERN.fullmatch(manifest_sha256) is not None
    ):
        recomputed_manifest_sha256 = _sha256_bytes(
            snapshot_canonical_json_bytes(
                {
                    "schema_id": SOURCE_MANIFEST_SCHEMA_ID,
                    "entries": raw_manifest_entries,
                }
            )
        )
        if recomputed_manifest_sha256 != manifest_sha256:
            _add_blocker(
                blockers,
                "SNAPSHOT_MANIFEST_HASH_MISMATCH",
                f"declared={manifest_sha256}; recomputed={recomputed_manifest_sha256}",
            )
    lock_manifest_entry = overlay.get("lock_manifest_entry")
    lock_path = Path(_string(overlay.get("lock_path"), "overlay.lock_path")).resolve()
    try:
        lock_relative_path = lock_path.relative_to(snapshot_root).as_posix()
    except ValueError:
        lock_relative_path = ""
    declared_lock_entry = (
        lock_manifest_entry if isinstance(lock_manifest_entry, dict) else None
    )
    actual_lock_entry = manifest_by_path.get(lock_relative_path)
    if (
        not lock_relative_path
        or declared_lock_entry is None
        or actual_lock_entry is None
        or dict(actual_lock_entry) != declared_lock_entry
        or declared_lock_entry.get("sha256") != lock_sha256
        or declared_lock_entry.get("role") != "configuration"
    ):
        _add_blocker(
            blockers,
            "OVERLAY_LOCK_MANIFEST_BINDING_INVALID",
            str(overlay.get("lock_path")),
        )
    imports = _object(probe.get("imports", {}), "imports")
    for name in ("simsopt", "simsopt_jax", "simsopt_jax_adapters"):
        identity = imports.get(name)
        if not _file_identity_valid(identity):
            _add_blocker(blockers, "IMPORT_IDENTITY_INVALID", name)
            continue
        _bind_file_to_snapshot_manifest(
            identity=_object(identity, f"imports.{name}"),
            snapshot_root=snapshot_root,
            manifest_by_path=manifest_by_path,
            context=name,
            blockers=blockers,
            expected_role="execution_source",
        )
    native_binary = probe.get("native_binary")
    if not _file_identity_valid(native_binary):
        _add_blocker(blockers, "NATIVE_BINARY_IDENTITY_INVALID", "simsoptpp")
    else:
        native_identity = _object(native_binary, "native_binary")
        _bind_file_to_snapshot_manifest(
            identity=native_identity,
            snapshot_root=snapshot_root,
            manifest_by_path=manifest_by_path,
            context="simsoptpp",
            blockers=blockers,
            expected_role="native_extension",
        )
        native_hash = native_identity.get("sha256")
        if (
            isinstance(expected_native_hash, str)
            and native_hash != expected_native_hash
        ):
            _add_blocker(
                blockers,
                "NATIVE_BINARY_HASH_MISMATCH",
                f"expected={expected_native_hash}; actual={native_hash}",
            )

    policy_source_path = numerical_policies.get("source_path")
    policy_source_sha256 = numerical_policies.get("source_sha256")
    policy_manifest_entry = numerical_policies.get("source_manifest_entry")
    declared_policies = _object(
        numerical_policies.get("declared"), "numerical_policies.declared"
    )
    raw_effective_policies = probe.get("effective_numerical_policies")
    effective_policies = (
        raw_effective_policies if isinstance(raw_effective_policies, dict) else {}
    )
    expected_policy_path = snapshot_root / SPECIMEN_DOCUMENT_RELATIVE_PATH
    expected_policy_entry = manifest_by_path.get(SPECIMEN_DOCUMENT_RELATIVE_PATH)
    if (
        policy_source_path != str(expected_policy_path)
        or not isinstance(policy_source_sha256, str)
        or _SHA256_PATTERN.fullmatch(policy_source_sha256) is None
        or not isinstance(policy_manifest_entry, dict)
        or expected_policy_entry is None
        or dict(expected_policy_entry) != policy_manifest_entry
        or expected_policy_entry.get("role") != "configuration"
        or expected_policy_entry.get("sha256") != policy_source_sha256
    ):
        _add_blocker(
            blockers,
            "NUMERICAL_POLICY_SOURCE_BINDING_INVALID",
            str(policy_source_path),
        )
    if frozenset(effective_policies) != _NUMERICAL_POLICY_FIELDS:
        _add_blocker(
            blockers,
            "NUMERICAL_POLICIES_INCOMPLETE",
            f"observed fields={sorted(effective_policies)}",
        )
    else:
        dense_batch_width = effective_policies["dense_batch_width"]
        chunk_sizes = (
            effective_policies["point_chunk_size"],
            effective_policies["coil_chunk_size"],
        )
        quadrature_blocks = effective_policies["quadrature_block_sizes"]
        if (
            not isinstance(dense_batch_width, int)
            or isinstance(dense_batch_width, bool)
            or dense_batch_width < 1
            or any(
                value is not None
                and (not isinstance(value, int) or isinstance(value, bool) or value < 1)
                for value in chunk_sizes
            )
            or not isinstance(quadrature_blocks, list)
            or not quadrature_blocks
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in quadrature_blocks
            )
        ):
            _add_blocker(
                blockers,
                "NUMERICAL_POLICIES_INVALID",
                json.dumps(dict(effective_policies), sort_keys=True),
            )
        if effective_policies != declared_policies:
            _add_blocker(
                blockers,
                "NUMERICAL_POLICY_DRIFT",
                json.dumps(
                    {
                        "declared": dict(declared_policies),
                        "observed": dict(effective_policies),
                    },
                    sort_keys=True,
                ),
            )

    static_timing_environment = probe.get("static_timing_environment")
    if not isinstance(static_timing_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in static_timing_environment.items()
    ):
        static_timing_environment = {}
        _add_blocker(
            blockers,
            "STATIC_TIMING_ENVIRONMENT_INVALID",
            "probe did not emit an exact string map",
        )

    affinity = observations.get("cpu_affinity")
    if (
        not isinstance(affinity, list)
        or not affinity
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0
            for cpu in affinity
        )
    ):
        _add_blocker(blockers, "CPU_AFFINITY_INVALID", str(affinity))

    ordered_blockers = sorted(blockers, key=lambda row: (row["code"], row["detail"]))
    receipt: dict[str, object] = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "state": "PASS" if not ordered_blockers else "BLOCKED",
        "blockers": ordered_blockers,
        "slurm": {
            **{key: environment.get(key) for key in REQUIRED_SLURM_ENVIRONMENT},
            "scheduler_gpu_accounting": scheduler_gpu_accounting,
            "exclusive_node_preferred": True,
        },
        "hostname": hostname,
        "interpreter": {
            **dict(collected_interpreter),
            "prefix": prefix,
            "base_prefix": base_prefix,
        },
        "physical_gpus": list(gpu_rows),
        "cuda": {
            "compatibility_policy": "cuda-12.6-forward-compat-only",
            "runtime_platform_version": platform_version,
            "compatibility_paths": list(compat_paths),
            "resolved_libraries": list(valid_libraries),
        },
        "jax": dict(jax_identity),
        "smoke": dict(smoke),
        "overlay": {
            "lock_path": overlay.get("lock_path"),
            "lock_sha256": overlay.get("lock_sha256"),
            "expected_packages": dict(normalized_expected),
            "actual_packages": actual_packages,
        },
        "source": dict(source),
        "imports": dict(imports),
        "native_binary": native_binary,
        "environment": dict(environment),
        "static_timing_environment": dict(static_timing_environment),
        "numerical_policies": {
            "source_path": policy_source_path,
            "source_sha256": policy_source_sha256,
            "source_manifest_entry": policy_manifest_entry,
            "effective": dict(effective_policies),
            "declared": dict(declared_policies),
        },
        "cpu_affinity": affinity,
        "failed_commands": sorted(failed_commands),
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_json_bytes(receipt))
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--overlay-lock", type=Path, required=True)
    parser.add_argument(
        "--source-provenance",
        type=Path,
        help="Optional externally produced immutable HEAD/worktree provenance JSON.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    source_provenance: Mapping[str, object] | None = None
    if options.source_provenance is not None:
        parsed = json.loads(options.source_provenance.read_text(encoding="utf-8"))
        source_provenance = _object(parsed, "source provenance")
    observations = collect_landau_observations(
        snapshot_root=options.snapshot_root,
        python_executable=options.python,
        overlay_lock=options.overlay_lock,
        environment=os.environ,
        source_provenance=source_provenance,
    )
    receipt = build_landau_qualification_receipt(observations)
    payload = _canonical_json_bytes(receipt)
    if options.output is None:
        sys.stdout.buffer.write(payload)
    else:
        with options.output.open("xb") as stream:
            stream.write(payload)
    return 0 if receipt["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
