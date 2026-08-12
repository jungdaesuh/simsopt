"""Construct provenance-bound Phase-0 C0 inputs from published artifacts.

This pre-gate workflow validates existing immutable artifacts, qualifies one
device lane, and emits the native-reference launch and C0 runner spec. It does
not execute either numerical benchmark child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from benchmarks.landau_a100_qualification import (
    QUALIFICATION_SCHEMA_ID,
)
from benchmarks.landau_a100_qualification import (
    _canonical_json_bytes as landau_canonical_json_bytes,
)
from benchmarks.single_stage_compute_graph_c0_runner import (
    C0_RUNNER_SPEC_SCHEMA_ID,
    _runtime_identity,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    SnapshotModuleLaunch,
    build_snapshot_module_launch,
    normalize_route_environment,
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    A100_LANE_ID,
    LANE_AGGREGATION_POLICY,
    PHASE0_SCHEMA_ID,
    REQUIRED_WARM_SAMPLES,
    RTX_LANE_ID,
    LaneId,
    Phase0ReceiptError,
    _validate_provenance,
    _validate_qualification,
    _validate_specimen,
    canonical_json_bytes,
    canonical_sha256,
    validate_phase0_receipt,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    IMPORT_ATTESTATION_SCHEMA_ID,
    ManifestEntry,
    load_snapshot_manifest,
)
from benchmarks.single_stage_compute_graph_snapshot_publish import (
    DEFAULT_OVERLAY_LOCK_RELATIVE_PATH,
    PUBLICATION_SCHEMA_ID,
    SPECIMEN_DESTINATION_ROOT,
)
from benchmarks.single_stage_compute_graph_specimen import (
    CANDIDATE_PATH,
    DOCUMENT_PATH,
    INPUT_BUNDLE_PATH,
)
from benchmarks.single_stage_compute_graph_specimen import (
    SCHEMA_ID as SPECIMEN_SCHEMA_ID,
)

WORKFLOW_SCHEMA_ID: Final = "single-stage-compute-graph-phase0-workflow-v1"
DEVICE_PROBE_SCHEMA_ID: Final = "single-stage-compute-graph-device-probe-v1"
NATIVE_REFERENCE_MODULE: Final = (
    "benchmarks.single_stage_compute_graph_native_reference"
)
_LOCAL_PROBE_SCHEMA_ID: Final = "single-stage-compute-graph-local-probe-v1"
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_LOCAL_REQUIRED_CHECKS: Final = (
    "source_snapshot",
    "import_bindings",
    "native_extension",
    "device_identity",
    "runtime_backend",
    "fp64_policy",
    "cpu_affinity",
    "strict_transfer_smoke",
)
_A100_REQUIRED_CHECKS: Final = _LOCAL_REQUIRED_CHECKS + (
    "slurm_allocation",
    "cuda_12_6_compatibility",
    "dependency_overlay",
    "resolved_cuda_libraries",
)


class Phase0WorkflowError(RuntimeError):
    """Published Phase-0 inputs are incomplete, inconsistent, or tampered."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], CommandResult]


@dataclass(frozen=True, slots=True)
class Phase0WorkflowInputs:
    snapshot_root: Path
    publication_path: Path
    import_attestation_path: Path
    interpreter: Path
    output_root: Path
    compilation_cache_directory: Path
    native_reference_path: Path
    lane_id: LaneId
    qualification_path: Path | None = None
    base_receipt_path: Path | None = None
    warm_sample_count: int = REQUIRED_WARM_SAMPLES


@dataclass(frozen=True, slots=True)
class Phase0WorkflowPlan:
    document: Mapping[str, object]
    native_reference_launch: SnapshotModuleLaunch
    receipt_template: Mapping[str, object]
    runner_spec: Mapping[str, object]
    qualification: Mapping[str, object]
    device_probe: Mapping[str, object]
    runtime_provenance: Mapping[str, object]


def _run_command(argv: Sequence[str], environment: Mapping[str, str]) -> CommandResult:
    completed = subprocess.run(
        tuple(argv), check=False, capture_output=True, text=True, env=dict(environment)
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise Phase0WorkflowError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _git_oid(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise Phase0WorkflowError(
            f"{context} must be a lowercase SHA-1 or SHA-256 Git object ID"
        )
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Phase0WorkflowError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise Phase0WorkflowError(f"{context} must be a JSON array")
    return value


def _canonical_document(path: Path, context: str) -> Mapping[str, object]:
    raw = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise Phase0WorkflowError(f"{context} contains duplicate key {key!r}")
            document[key] = value
        return document

    try:
        document = _mapping(
            json.loads(raw, object_pairs_hook=reject_duplicates), context
        )
    except json.JSONDecodeError as error:
        raise Phase0WorkflowError(f"{context} is not valid JSON: {error}") from error
    if raw != canonical_json_bytes(document):
        raise Phase0WorkflowError(f"{context} is not canonical JSON")
    return document


def _execution_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Keep only runtime-selection variables that are safe to receipt-bind."""

    return normalize_static_timing_environment(environment)


def _publication(
    snapshot_root: Path, publication_path: Path
) -> tuple[Mapping[str, object], tuple[ManifestEntry, ...], str]:
    publication = _canonical_document(publication_path, "snapshot publication")
    expected_keys = {
        "schema_id",
        "repository_root",
        "snapshot_root",
        "snapshot_manifest_sha256",
        "cross_host_source_sha256",
        "native_extension",
        "worktree",
    }
    if (
        set(publication) != expected_keys
        or publication.get("schema_id") != PUBLICATION_SCHEMA_ID
    ):
        raise Phase0WorkflowError("snapshot publication schema is invalid")
    snapshot_root = snapshot_root.absolute()
    if publication.get("snapshot_root") != str(snapshot_root):
        raise Phase0WorkflowError(
            "snapshot publication root does not match requested root"
        )
    entries, manifest_sha256 = load_snapshot_manifest(snapshot_root)
    if publication.get("snapshot_manifest_sha256") != manifest_sha256:
        raise Phase0WorkflowError("snapshot publication manifest identity mismatch")
    cross_host_entries = [
        entry.to_json()
        for entry in entries
        if entry.role != "native_extension"
        and entry.relative_path != DEFAULT_OVERLAY_LOCK_RELATIVE_PATH
    ]
    if publication.get("cross_host_source_sha256") != canonical_sha256(
        cross_host_entries
    ):
        raise Phase0WorkflowError("snapshot publication cross-host identity mismatch")
    native = _mapping(publication.get("native_extension"), "native extension")
    native_entries = tuple(
        entry for entry in entries if entry.role == "native_extension"
    )
    if len(native_entries) != 1 or native != native_entries[0].to_json():
        raise Phase0WorkflowError("snapshot publication native extension mismatch")
    worktree = _mapping(publication.get("worktree"), "worktree provenance")
    expected_worktree = {
        "repository_commit",
        "git_status_short",
        "tracked_diff_sha256",
        "untracked_manifest_sha256",
        "source_state_sha256",
    }
    if set(worktree) != expected_worktree:
        raise Phase0WorkflowError("worktree provenance schema is invalid")
    _git_oid(worktree.get("repository_commit"), "worktree.repository_commit")
    for field in (
        "tracked_diff_sha256",
        "untracked_manifest_sha256",
        "source_state_sha256",
    ):
        _sha256(worktree.get(field), f"worktree.{field}")
    state = {
        "repository_commit": worktree["repository_commit"],
        "git_status_short": worktree["git_status_short"],
        "tracked_diff_sha256": worktree["tracked_diff_sha256"],
        "untracked_manifest_sha256": worktree["untracked_manifest_sha256"],
    }
    if canonical_sha256(state) != worktree["source_state_sha256"]:
        raise Phase0WorkflowError("worktree source-state identity mismatch")
    return publication, entries, manifest_sha256


def _attestation(
    path: Path,
    *,
    snapshot_root: Path,
    interpreter: Path,
    entries: Sequence[ManifestEntry],
    manifest_sha256: str,
) -> None:
    document = _canonical_document(path, "import attestation")
    expected = {
        "schema_id",
        "state",
        "snapshot_manifest_sha256",
        "interpreter_path",
        "python_version",
        "bindings",
    }
    if (
        set(document) != expected
        or document.get("schema_id") != IMPORT_ATTESTATION_SCHEMA_ID
    ):
        raise Phase0WorkflowError("import attestation schema is invalid")
    if (
        document.get("state") != "pass"
        or document.get("snapshot_manifest_sha256") != manifest_sha256
    ):
        raise Phase0WorkflowError("import attestation does not bind the snapshot")
    lexical_interpreter = Path(os.path.abspath(interpreter))
    if not lexical_interpreter.is_absolute() or document.get("interpreter_path") != str(
        lexical_interpreter
    ):
        raise Phase0WorkflowError("import attestation interpreter mismatch")
    entry_by_path = {entry.relative_path: entry for entry in entries}
    raw_bindings = _sequence(document.get("bindings"), "import attestation bindings")
    bindings = []
    for raw_binding in raw_bindings:
        binding = _mapping(raw_binding, "import binding")
        module = binding.get("module")
        relative_path = binding.get("relative_path")
        if not isinstance(module, str) or not isinstance(relative_path, str):
            raise Phase0WorkflowError("import binding names must be strings")
        entry = entry_by_path.get(relative_path)
        if entry is None or binding != {
            "module": module,
            "relative_path": relative_path,
            "size_bytes": entry.size_bytes,
            "sha256": entry.sha256,
        }:
            raise Phase0WorkflowError(
                f"import binding for {module!r} differs from manifest"
            )
        bindings.append(binding)
    if {binding["module"] for binding in bindings} != {
        "simsopt",
        "simsopt_jax",
        "simsopt_jax_adapters",
        "simsoptpp",
    }:
        raise Phase0WorkflowError("import attestation omits a required package")


def _specimen(snapshot_root: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    specimen_root = snapshot_root / SPECIMEN_DESTINATION_ROOT
    document = _canonical_document(specimen_root / DOCUMENT_PATH, "specimen")
    if document.get("schema_id") != SPECIMEN_SCHEMA_ID:
        raise Phase0WorkflowError("specimen schema is invalid")
    specimen = _mapping(document.get("specimen"), "specimen contract")
    try:
        specimen_sha256 = _validate_specimen(specimen, "specimen contract")
    except Phase0ReceiptError as error:
        raise Phase0WorkflowError(f"specimen contract is invalid: {error}") from error
    if document.get("specimen_sha256") != specimen_sha256:
        raise Phase0WorkflowError("specimen identity mismatch")
    candidate_ref = _mapping(document.get("candidate"), "specimen candidate")
    candidate_path = specimen_root / CANDIDATE_PATH
    if candidate_ref.get("relative_path") != CANDIDATE_PATH or candidate_ref.get(
        "file_sha256"
    ) != _sha256_path(candidate_path):
        raise Phase0WorkflowError("specimen candidate file identity mismatch")
    candidate = np.load(candidate_path, allow_pickle=False)
    canonical = np.ascontiguousarray(candidate, dtype=np.dtype("<f8"))
    parameter_sha256 = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    if (
        candidate.dtype != np.dtype(np.float64)
        or candidate.shape != (461,)
        or parameter_sha256 != specimen.get("parameter_sha256")
    ):
        raise Phase0WorkflowError("specimen candidate parameter identity mismatch")
    bundle_path = specimen_root / INPUT_BUNDLE_PATH / "input_bundle.json"
    if _sha256_path(bundle_path) != specimen.get("input_bundle_sha256"):
        raise Phase0WorkflowError("specimen input-bundle identity mismatch")
    return document, specimen


_LOCAL_PROBE_SOURCE: Final = r"""import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import sys
from pathlib import Path

snapshot = Path(sys.argv[1]).absolute()
sys.path.insert(0, str(snapshot))
sys.path.insert(0, str(snapshot / "src"))
allowed_finders = (
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    importlib.machinery.PathFinder,
)
sys.meta_path[:] = [finder for finder in sys.meta_path if finder in allowed_finders]
import jax
import jax.numpy as jnp
import numpy as np
import simsoptpp
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_static_timing_environment,
    observe_effective_numerical_policies,
)

seed = jax.device_put(np.asarray([1.25, -2.0, 3.5], dtype=np.float64))
value = jax.jit(lambda item: jnp.sum(item * item))(seed)
value.block_until_ready()
libraries = []
for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
    fields = line.split()
    if fields and fields[-1].startswith("/") and "cuda" in Path(fields[-1]).name.lower():
        libraries.append(str(Path(fields[-1]).absolute()))
devices = jax.devices()
packages = {}
for name in ("jax", "jaxlib", "lineax"):
    packages[name] = importlib.metadata.version(name)
print(json.dumps({
    "schema_id": "single-stage-compute-graph-local-probe-v1",
    "interpreter": {
        "entrypoint_path": str(Path(sys.executable).absolute()),
        "target_path": str(Path(sys.executable).resolve()),
        "target_size": Path(sys.executable).resolve().stat().st_size,
        "target_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    },
    "jax_version": jax.__version__,
    "jaxlib_version": jax.lib.__version__,
    "jax_backend": jax.default_backend(),
    "x64_enabled": bool(jax.config.jax_enable_x64),
    "devices": [{"kind": device.device_kind, "platform": device.platform} for device in devices],
    "cuda_runtime": str(getattr(devices[0].client, "platform_version", "unknown")),
    "resolved_cuda_libraries": sorted(set(libraries)),
    "native_extension_path": str(Path(simsoptpp.__file__).absolute()),
    "strict_transfer_smoke": {
        "guard": os.environ.get("JAX_TRANSFER_GUARD"),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "finite": bool(np.isfinite(np.asarray(jax.device_get(value)))),
    },
    "packages": packages,
    "static_timing_environment": normalize_static_timing_environment(os.environ),
    "effective_numerical_policies": observe_effective_numerical_policies(250),
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
"""


def _local_qualification(
    *,
    snapshot_root: Path,
    interpreter: Path,
    attestation_document: Mapping[str, object],
    publication: Mapping[str, object],
    entries: Sequence[ManifestEntry],
    manifest_sha256: str,
    cache_directory: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
    declared_policies: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    probe_environment = _execution_environment(environment)
    probe_environment.update(
        {
            "JAX_TRANSFER_GUARD": "disallow",
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
        }
    )
    lexical_interpreter = Path(os.path.abspath(interpreter))
    probe_result = runner(
        (str(lexical_interpreter), "-I", "-c", _LOCAL_PROBE_SOURCE, str(snapshot_root)),
        probe_environment,
    )
    visible_device = probe_environment.get("CUDA_VISIBLE_DEVICES", "")
    if "," in visible_device:
        raise Phase0WorkflowError(
            "local qualification requires one CUDA_VISIBLE_DEVICES selection"
        )
    gpu_argv = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    if visible_device:
        gpu_argv[1:1] = ["-i", visible_device]
    gpu_result = runner(
        tuple(gpu_argv),
        probe_environment,
    )
    if probe_result.returncode != 0:
        raise Phase0WorkflowError(
            f"local runtime probe failed: {probe_result.stderr.strip()}"
        )
    if gpu_result.returncode != 0:
        raise Phase0WorkflowError(
            f"local device probe failed: {gpu_result.stderr.strip()}"
        )
    probe = _mapping(json.loads(probe_result.stdout), "local runtime probe")
    if probe.get("schema_id") != _LOCAL_PROBE_SCHEMA_ID:
        raise Phase0WorkflowError("local runtime probe schema is invalid")
    gpu_rows = [
        tuple(part.strip() for part in line.split(","))
        for line in gpu_result.stdout.splitlines()
        if line.strip()
    ]
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 4:
        raise Phase0WorkflowError(
            "local qualification requires exactly one physical GPU"
        )
    gpu_uuid, gpu_name, memory_mib_text, driver = gpu_rows[0]
    if "RTX 5090" not in gpu_name.upper() or not gpu_uuid.startswith("GPU-"):
        raise Phase0WorkflowError(
            "local physical device is not one identified RTX 5090"
        )
    devices = _sequence(probe.get("devices"), "local probe devices")
    smoke = _mapping(probe.get("strict_transfer_smoke"), "strict-transfer smoke")
    probe_interpreter = _mapping(probe.get("interpreter"), "local interpreter")
    interpreter_target = lexical_interpreter.resolve()
    if (
        probe_interpreter
        != {
            "entrypoint_path": str(lexical_interpreter),
            "target_path": str(interpreter_target),
            "target_size": interpreter_target.stat().st_size,
            "target_sha256": _sha256_path(interpreter_target),
            "prefix": probe_interpreter.get("prefix"),
            "base_prefix": probe_interpreter.get("base_prefix"),
        }
        or not isinstance(probe_interpreter.get("prefix"), str)
        or not probe_interpreter["prefix"]
        or not isinstance(probe_interpreter.get("base_prefix"), str)
        or not probe_interpreter["base_prefix"]
        or probe_interpreter["prefix"] == probe_interpreter["base_prefix"]
        or probe.get("jax_backend") != "gpu"
        or probe.get("x64_enabled") is not True
        or len(devices) != 1
        or "RTX 5090"
        not in str(_mapping(devices[0], "JAX device").get("kind", "")).upper()
        or smoke
        != {"guard": "disallow", "dtype": "float64", "shape": [], "finite": True}
    ):
        raise Phase0WorkflowError("local JAX FP64 device qualification failed")
    bindings = {
        str(binding["module"]): {
            "path": str(snapshot_root / str(binding["relative_path"])),
            "sha256": binding["sha256"],
        }
        for binding in _sequence(
            attestation_document["bindings"], "attestation bindings"
        )
    }
    if probe.get("native_extension_path") != bindings["simsoptpp"]["path"]:
        raise Phase0WorkflowError("runtime native extension differs from attestation")
    manifest_document = {
        "schema_id": "single-stage-compute-graph-source-manifest-v1",
        "entries": [entry.to_json() for entry in entries],
    }
    worktree = _mapping(publication["worktree"], "worktree")
    libraries = _sequence(
        probe.get("resolved_cuda_libraries"), "resolved CUDA libraries"
    )
    if not libraries:
        raise Phase0WorkflowError("local probe did not resolve a CUDA library")
    packages = _mapping(probe.get("packages"), "local packages")
    observed_policies = _mapping(
        probe.get("effective_numerical_policies"),
        "local effective numerical policies",
    )
    if observed_policies != declared_policies:
        raise Phase0WorkflowError(
            "local runtime numerical policies differ from specimen"
        )
    qualification = {
        "outcome": "qualified",
        "attempted_identity": {
            "hostname": os.uname().nodename,
            "requested_device": RTX_LANE_ID,
            "gpu_uuid": gpu_uuid,
        },
        "checks": [
            {
                "check_id": check_id,
                "passed": True,
                "evidence": f"validated by {_LOCAL_PROBE_SCHEMA_ID}",
            }
            for check_id in sorted(_LOCAL_REQUIRED_CHECKS)
        ],
        "blocker": None,
    }
    provenance = {
        "repository_commit": worktree["repository_commit"],
        "source_state_sha256": worktree["source_state_sha256"],
        "git_status_short": list(_sequence(worktree["git_status_short"], "git status")),
        "tracked_diff_sha256": worktree["tracked_diff_sha256"],
        "untracked_manifest_sha256": worktree["untracked_manifest_sha256"],
        "immutable_root": str(snapshot_root),
        "immutable_tree_sha256": canonical_sha256(
            [entry.to_json() for entry in entries]
        ),
        "source_manifest": manifest_document,
        "source_manifest_sha256": manifest_sha256,
        "interpreter_path": str(lexical_interpreter),
        "runtime": {
            "python_version": attestation_document["python_version"],
            "jax_version": probe["jax_version"],
            "jaxlib_version": probe["jaxlib_version"],
            "cuda_runtime": probe["cuda_runtime"],
            "cuda_driver": driver,
            "jax_backend": "gpu",
            "fp64_x64_enabled": True,
            "resolved_cuda_libraries": list(libraries),
        },
        "allocation": {
            "hostname": os.uname().nodename,
            "scheduler": "local",
            "allocation_id": "local",
            "job_id": "local",
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "gpu_memory_bytes": int(memory_mib_text) * 1024 * 1024,
            "cpu_affinity": ",".join(
                str(cpu) for cpu in sorted(os.sched_getaffinity(0))
            ),
            "cuda_compatibility_version": "native",
            "cuda_compatibility_path": "not-applicable",
        },
        "import_bindings": bindings,
        "package_overlay": {str(key): str(value) for key, value in packages.items()},
        "environment": dict(
            _mapping(
                probe["static_timing_environment"],
                "local static timing environment",
            )
        ),
        "policies": dict(observed_policies),
        "compilation_cache_directory": str(cache_directory.absolute()),
    }
    return qualification, provenance


def _landau_qualification(
    path: Path,
    *,
    snapshot_root: Path,
    interpreter: Path,
    attestation_document: Mapping[str, object],
    publication: Mapping[str, object],
    entries: Sequence[ManifestEntry],
    manifest_sha256: str,
    cache_directory: Path,
    execution_environment: Mapping[str, str],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    receipt = _canonical_document(path, "Landau qualification")
    if receipt.get("schema_id") != QUALIFICATION_SCHEMA_ID:
        raise Phase0WorkflowError("Landau qualification schema is invalid")
    without_hash = dict(receipt)
    declared_hash = _sha256(
        without_hash.pop("receipt_sha256", None), "Landau receipt_sha256"
    )
    if (
        hashlib.sha256(landau_canonical_json_bytes(without_hash)).hexdigest()
        != declared_hash
    ):
        raise Phase0WorkflowError("Landau qualification receipt hash mismatch")
    if receipt.get("state") != "PASS" or receipt.get("blockers") != []:
        raise Phase0WorkflowError("Landau qualification is blocked")
    interpreter_identity = _mapping(receipt.get("interpreter"), "Landau interpreter")
    lexical_interpreter = Path(os.path.abspath(interpreter))
    interpreter_target = lexical_interpreter.resolve()
    if interpreter_identity != {
        "entrypoint_path": str(lexical_interpreter),
        "target_path": str(interpreter_target),
        "target_size": interpreter_target.stat().st_size,
        "target_sha256": _sha256_path(interpreter_target),
        "prefix": interpreter_identity.get("prefix"),
        "base_prefix": interpreter_identity.get("base_prefix"),
    }:
        raise Phase0WorkflowError("Landau interpreter identity is invalid")
    if (
        not isinstance(interpreter_identity.get("prefix"), str)
        or not interpreter_identity["prefix"]
        or not isinstance(interpreter_identity.get("base_prefix"), str)
        or not interpreter_identity["base_prefix"]
        or interpreter_identity["prefix"] == interpreter_identity["base_prefix"]
    ):
        raise Phase0WorkflowError("Landau interpreter is not an identified venv")
    source = _mapping(receipt.get("source"), "Landau source")
    if (
        source.get("snapshot_root") != str(snapshot_root)
        or source.get("snapshot_manifest_sha256") != manifest_sha256
    ):
        raise Phase0WorkflowError("Landau qualification source mismatch")
    external = _mapping(source.get("external_provenance"), "Landau external provenance")
    if external != publication:
        raise Phase0WorkflowError(
            "Landau qualification publication provenance mismatch"
        )
    jax = _mapping(receipt.get("jax"), "Landau JAX")
    physical = _sequence(receipt.get("physical_gpus"), "Landau GPUs")
    if len(physical) != 1:
        raise Phase0WorkflowError("Landau qualification must identify one GPU")
    gpu = _mapping(physical[0], "Landau GPU")
    imports = _mapping(receipt.get("imports"), "Landau imports")
    native = _mapping(receipt.get("native_binary"), "Landau native binary")
    expected_bindings = {
        str(binding["module"]): {
            "path": str(snapshot_root / str(binding["relative_path"])),
            "sha256": binding["sha256"],
        }
        for binding in _sequence(
            attestation_document["bindings"], "attestation bindings"
        )
    }
    observed_bindings = {
        str(name): {"path": identity["path"], "sha256": identity["sha256"]}
        for name, raw_identity in imports.items()
        for identity in (_mapping(raw_identity, f"Landau import {name}"),)
    }
    observed_bindings["simsoptpp"] = {
        "path": native["path"],
        "sha256": native["sha256"],
    }
    if observed_bindings != expected_bindings:
        raise Phase0WorkflowError("Landau import identities differ from attestation")
    worktree = _mapping(publication["worktree"], "worktree")
    cuda = _mapping(receipt.get("cuda"), "Landau CUDA")
    overlay = _mapping(receipt.get("overlay"), "Landau overlay")
    slurm = _mapping(receipt.get("slurm"), "Landau Slurm")
    receipt_environment = _mapping(
        receipt.get("static_timing_environment"),
        "Landau static timing environment",
    )
    if dict(receipt_environment) != normalize_static_timing_environment(
        execution_environment
    ):
        raise Phase0WorkflowError("Landau static timing environment differs")
    numerical_policies = _mapping(
        receipt.get("numerical_policies"), "Landau numerical policies"
    )
    effective_policies = _mapping(
        numerical_policies.get("effective"), "Landau effective numerical policies"
    )
    platform_version = cuda.get("runtime_platform_version")
    if (
        not isinstance(platform_version, str)
        or not platform_version
        or jax.get("platform_version") != platform_version
    ):
        raise Phase0WorkflowError("Landau CUDA runtime platform identity is invalid")
    jax_backend = jax.get("backend")
    fp64_x64_enabled = jax.get("x64_enabled")
    if jax_backend != "gpu" or fp64_x64_enabled is not True:
        raise Phase0WorkflowError("Landau JAX runtime policy is invalid")
    compatibility_paths = _sequence(
        cuda.get("compatibility_paths"), "Landau compatibility paths"
    )
    if len(compatibility_paths) != 1:
        raise Phase0WorkflowError("Landau CUDA compatibility path is invalid")
    compatibility_path = compatibility_paths[0]
    if not isinstance(compatibility_path, str) or not compatibility_path:
        raise Phase0WorkflowError("Landau CUDA compatibility path is invalid")
    compatibility_directory = Path(compatibility_path).parent.name
    if not compatibility_directory.startswith("cuda-"):
        raise Phase0WorkflowError("Landau CUDA compatibility version is invalid")
    compatibility_version = compatibility_directory.removeprefix("cuda-")
    qualification = {
        "outcome": "qualified",
        "attempted_identity": {
            "hostname": receipt["hostname"],
            "requested_device": A100_LANE_ID,
            "gpu_uuid": gpu["uuid"],
        },
        "checks": [
            {
                "check_id": check_id,
                "passed": True,
                "evidence": f"Landau receipt {declared_hash}",
            }
            for check_id in sorted(_A100_REQUIRED_CHECKS)
        ],
        "blocker": None,
    }
    provenance = {
        "repository_commit": worktree["repository_commit"],
        "source_state_sha256": worktree["source_state_sha256"],
        "git_status_short": list(worktree["git_status_short"]),
        "tracked_diff_sha256": worktree["tracked_diff_sha256"],
        "untracked_manifest_sha256": worktree["untracked_manifest_sha256"],
        "immutable_root": str(snapshot_root),
        "immutable_tree_sha256": canonical_sha256(
            [entry.to_json() for entry in entries]
        ),
        "source_manifest": {
            "schema_id": "single-stage-compute-graph-source-manifest-v1",
            "entries": [entry.to_json() for entry in entries],
        },
        "source_manifest_sha256": manifest_sha256,
        "interpreter_path": str(Path(os.path.abspath(interpreter))),
        "runtime": {
            "python_version": attestation_document["python_version"],
            "jax_version": jax["version"],
            "jaxlib_version": jax["jaxlib_version"],
            "cuda_runtime": platform_version,
            "cuda_driver": gpu["driver_version"],
            "jax_backend": jax_backend,
            "fp64_x64_enabled": fp64_x64_enabled,
            "resolved_cuda_libraries": [
                row["path"]
                for row in _sequence(
                    cuda["resolved_libraries"], "Landau CUDA libraries"
                )
            ],
        },
        "allocation": {
            "hostname": receipt["hostname"],
            "scheduler": "slurm",
            "allocation_id": slurm["SLURM_JOB_ID"],
            "job_id": slurm["SLURM_JOB_ID"],
            "gpu_name": gpu["name"],
            "gpu_uuid": gpu["uuid"],
            "gpu_memory_bytes": int(gpu["memory_mib"]) * 1024 * 1024,
            "cpu_affinity": ",".join(str(cpu) for cpu in receipt["cpu_affinity"]),
            "cuda_compatibility_version": compatibility_version,
            "cuda_compatibility_path": compatibility_path,
        },
        "import_bindings": observed_bindings,
        "package_overlay": dict(
            _mapping(overlay["actual_packages"], "Landau packages")
        ),
        "environment": dict(receipt_environment),
        "policies": dict(effective_policies),
        "compilation_cache_directory": str(cache_directory.absolute()),
    }
    return qualification, provenance


def _blocked_a100() -> Mapping[str, object]:
    return {
        "outcome": "blocked",
        "attempted_identity": {"requested_device": A100_LANE_ID},
        "checks": [
            {
                "check_id": check_id,
                "passed": check_id != "slurm_allocation",
                "evidence": "not supplied to local pre-gate workflow",
            }
            for check_id in sorted(_A100_REQUIRED_CHECKS)
        ],
        "blocker": {
            "code": "LANDAU_QUALIFICATION_NOT_SUPPLIED",
            "check_id": "slurm_allocation",
            "reason": "no current Landau qualification artifact",
            "evidence_sha256": canonical_sha256({"state": "not_supplied"}),
        },
    }


def _device_probe_document(
    lane_id: LaneId,
    qualification: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Derive the standalone GPU/native probe from validated lane evidence."""

    allocation = _mapping(provenance.get("allocation"), "provenance allocation")
    bindings = _mapping(provenance.get("import_bindings"), "import bindings")
    native = _mapping(bindings.get("simsoptpp"), "native binary binding")
    return {
        "schema_id": DEVICE_PROBE_SCHEMA_ID,
        "lane_id": lane_id,
        "source_state_sha256": provenance.get("source_state_sha256"),
        "runtime_identity_sha256": _runtime_identity(provenance),
        "qualification_sha256": canonical_sha256(qualification),
        "gpu": {
            "uuid": allocation.get("gpu_uuid"),
            "name": allocation.get("gpu_name"),
            "memory_bytes": allocation.get("gpu_memory_bytes"),
        },
        "native_binary": {
            "path": native.get("path"),
            "sha256": native.get("sha256"),
        },
    }


def _validate_standalone_evidence(
    lane_id: LaneId,
    qualification: Mapping[str, object],
    device_probe: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
) -> None:
    """Validate exact standalone schemas and their source/runtime/device relation."""

    try:
        outcome = _validate_qualification(
            qualification,
            lane_id,
            "standalone qualification",
        )
        source_sha256, gpu_uuid, _cache = _validate_provenance(
            runtime_provenance,
            lane_id,
            "standalone runtime provenance",
        )
    except Phase0ReceiptError as error:
        raise Phase0WorkflowError(
            f"standalone Phase-0 evidence is invalid: {error}"
        ) from error
    if outcome != "qualified":
        raise Phase0WorkflowError("standalone qualification must be qualified")
    _sha256(source_sha256, "standalone source state")
    attempted_identity = _mapping(
        qualification.get("attempted_identity"),
        "standalone qualification attempted identity",
    )
    if (
        attempted_identity.get("requested_device") != lane_id
        or attempted_identity.get("gpu_uuid") != gpu_uuid
    ):
        raise Phase0WorkflowError(
            "standalone qualification GPU identity differs from runtime provenance"
        )
    if device_probe != _device_probe_document(
        lane_id,
        qualification,
        runtime_provenance,
    ):
        raise Phase0WorkflowError(
            "standalone device probe differs from qualification/runtime provenance"
        )
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
        or device_probe.get("schema_id") != DEVICE_PROBE_SCHEMA_ID
    ):
        raise Phase0WorkflowError("standalone device probe schema is invalid")
    probe_gpu = _mapping(device_probe.get("gpu"), "standalone device probe GPU")
    probe_native = _mapping(
        device_probe.get("native_binary"),
        "standalone device probe native binary",
    )
    if set(probe_gpu) != {"uuid", "name", "memory_bytes"} or set(probe_native) != {
        "path",
        "sha256",
    }:
        raise Phase0WorkflowError("standalone device probe schema is invalid")
    if probe_gpu.get("uuid") != gpu_uuid:
        raise Phase0WorkflowError("standalone device probe GPU identity mismatch")
    _sha256(probe_native.get("sha256"), "standalone native binary SHA")


def _standalone_artifact_references(
    qualification: Mapping[str, object],
    device_probe: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "qualification": {
            "relative_path": "qualification.json",
            "sha256": canonical_sha256(qualification),
        },
        "device_probe": {
            "relative_path": "device-probe.json",
            "sha256": canonical_sha256(device_probe),
        },
        "runtime_provenance": {
            "relative_path": "runtime-provenance.json",
            "sha256": canonical_sha256(runtime_provenance),
        },
    }


def build_phase0_workflow(
    inputs: Phase0WorkflowInputs,
    *,
    environment: Mapping[str, str] | None = None,
    runner: CommandRunner = _run_command,
) -> Phase0WorkflowPlan:
    """Validate the artifact chain and build native/C0 pre-gate inputs."""

    snapshot_root = inputs.snapshot_root.absolute()
    interpreter = Path(os.path.abspath(inputs.interpreter))
    if inputs.warm_sample_count < REQUIRED_WARM_SAMPLES:
        raise Phase0WorkflowError(
            f"warm_sample_count must be at least {REQUIRED_WARM_SAMPLES}"
        )
    output_root = inputs.output_root.absolute()
    cache_directory = inputs.compilation_cache_directory.absolute()
    if (
        output_root == cache_directory
        or output_root.is_relative_to(cache_directory)
        or cache_directory.is_relative_to(output_root)
    ):
        raise Phase0WorkflowError(
            "artifact root and compilation cache must be disjoint"
        )
    publication, entries, manifest_sha256 = _publication(
        snapshot_root, inputs.publication_path
    )
    attestation_document = _canonical_document(
        inputs.import_attestation_path, "import attestation"
    )
    _attestation(
        inputs.import_attestation_path,
        snapshot_root=snapshot_root,
        interpreter=interpreter,
        entries=entries,
        manifest_sha256=manifest_sha256,
    )
    specimen_document, specimen = _specimen(snapshot_root)
    base_environment = os.environ if environment is None else environment
    execution_environment = _execution_environment(base_environment)
    execution_environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(cache_directory),
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
        }
    )
    if inputs.lane_id == RTX_LANE_ID:
        qualification, provenance = _local_qualification(
            snapshot_root=snapshot_root,
            interpreter=interpreter,
            attestation_document=attestation_document,
            publication=publication,
            entries=entries,
            manifest_sha256=manifest_sha256,
            cache_directory=inputs.compilation_cache_directory,
            environment=execution_environment,
            runner=runner,
            declared_policies=_mapping(
                specimen_document.get("effective_policies"),
                "specimen effective policies",
            ),
        )
        lanes = [
            {
                "lane_id": RTX_LANE_ID,
                "device_class": "NVIDIA GeForce RTX 5090",
                "qualification": qualification,
                "measurement": None,
            },
            {
                "lane_id": A100_LANE_ID,
                "device_class": "NVIDIA A100",
                "qualification": _blocked_a100(),
                "measurement": None,
            },
        ]
    else:
        if inputs.qualification_path is None or inputs.base_receipt_path is None:
            raise Phase0WorkflowError(
                "A100 workflow requires Landau qualification and validated RTX base receipt"
            )
        base_receipt = _canonical_document(inputs.base_receipt_path, "RTX base receipt")
        validate_phase0_receipt(base_receipt)
        if base_receipt.get("specimen_sha256") != specimen_document["specimen_sha256"]:
            raise Phase0WorkflowError(
                "RTX base receipt is bound to a different specimen"
            )
        qualification, provenance = _landau_qualification(
            inputs.qualification_path,
            snapshot_root=snapshot_root,
            interpreter=interpreter,
            attestation_document=attestation_document,
            publication=publication,
            entries=entries,
            manifest_sha256=manifest_sha256,
            cache_directory=inputs.compilation_cache_directory,
            execution_environment=execution_environment,
        )
        lanes = [
            dict(_mapping(lane, "base receipt lane"))
            for lane in _sequence(base_receipt["lanes"], "base receipt lanes")
        ]
        a100_index = next(
            index for index, lane in enumerate(lanes) if lane["lane_id"] == A100_LANE_ID
        )
        lanes[a100_index] = {
            **lanes[a100_index],
            "qualification": qualification,
            "measurement": None,
        }
    try:
        _validate_qualification(qualification, inputs.lane_id, "qualification")
        _validate_provenance(provenance, inputs.lane_id, "provenance")
    except Phase0ReceiptError as error:
        raise Phase0WorkflowError(
            f"generated Phase-0 lane evidence is invalid: {error}"
        ) from error
    qualification = dict(qualification)
    runtime_provenance = dict(provenance)
    device_probe = _device_probe_document(
        inputs.lane_id,
        qualification,
        runtime_provenance,
    )
    _validate_standalone_evidence(
        inputs.lane_id,
        qualification,
        device_probe,
        runtime_provenance,
    )
    receipt_template = {
        "schema_id": PHASE0_SCHEMA_ID,
        "artifact_id": f"single-stage-compute-graph-phase0-{inputs.lane_id}",
        "evidence_kind": "compute_graph_engineering_phase0_noncampaign",
        "lane_aggregation_policy": LANE_AGGREGATION_POLICY,
        "specimen": dict(specimen),
        "specimen_sha256": specimen_document["specimen_sha256"],
        "lanes": lanes,
    }
    bundle_document = _canonical_document(
        snapshot_root
        / SPECIMEN_DESTINATION_ROOT
        / INPUT_BUNDLE_PATH
        / "input_bundle.json",
        "input bundle",
    )
    binding_fields = _mapping(
        runtime_provenance["import_bindings"],
        "import bindings",
    )
    runtime_identity_sha256 = _runtime_identity(runtime_provenance)
    runtime_contract = {
        "runtime": dict(_mapping(runtime_provenance["runtime"], "runtime provenance")),
        "static_environment": dict(
            _mapping(
                runtime_provenance["environment"],
                "environment provenance",
            )
        ),
        "route_environment": normalize_route_environment(execution_environment),
        "policies": dict(_mapping(runtime_provenance["policies"], "policy provenance")),
        "expected_runtime_identity_sha256": runtime_identity_sha256,
    }
    runtime_contract_json = json.dumps(
        runtime_contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    native_binding_args = (
        "--input-root",
        str(snapshot_root / SPECIMEN_DESTINATION_ROOT / INPUT_BUNDLE_PATH),
        "--candidate",
        str(snapshot_root / SPECIMEN_DESTINATION_ROOT / CANDIDATE_PATH),
        "--parameter-sha256",
        str(specimen["parameter_sha256"]),
        "--input-fingerprint",
        str(bundle_document["input_fingerprint"]),
        "--input-bundle-sha256",
        str(specimen["input_bundle_sha256"]),
        "--configuration-fingerprint",
        str(bundle_document["configuration_fingerprint"]),
        "--specimen-sha256",
        str(specimen_document["specimen_sha256"]),
        "--source-sha256",
        str(runtime_provenance["source_state_sha256"]),
        "--runtime-identity-sha256",
        runtime_identity_sha256,
        "--runtime-contract-json",
        runtime_contract_json,
        "--interpreter-path",
        str(interpreter),
        "--native-simsoptpp-path",
        str(_mapping(binding_fields["simsoptpp"], "simsoptpp binding")["path"]),
        "--native-simsoptpp-sha256",
        str(_mapping(binding_fields["simsoptpp"], "simsoptpp binding")["sha256"]),
    )
    native_launch = build_snapshot_module_launch(
        interpreter,
        snapshot_root,
        NATIVE_REFERENCE_MODULE,
        native_binding_args,
        execution_environment,
    )
    runner_spec = {
        "schema_id": C0_RUNNER_SPEC_SCHEMA_ID,
        "lane_id": inputs.lane_id,
        "warm_sample_count": inputs.warm_sample_count,
        "output_root": str(output_root),
        "input_root": str(
            snapshot_root / SPECIMEN_DESTINATION_ROOT / INPUT_BUNDLE_PATH
        ),
        "candidate_path": str(
            snapshot_root / SPECIMEN_DESTINATION_ROOT / CANDIDATE_PATH
        ),
        "native_reference_path": str(inputs.native_reference_path.absolute()),
        "provenance": runtime_provenance,
        "receipt_template": receipt_template,
    }
    document = {
        "schema_id": WORKFLOW_SCHEMA_ID,
        "lane_id": inputs.lane_id,
        "snapshot_manifest_sha256": manifest_sha256,
        "import_attestation_sha256": _sha256_path(inputs.import_attestation_path),
        "specimen_sha256": specimen_document["specimen_sha256"],
        "runtime_identity_sha256": runtime_identity_sha256,
        "standalone_evidence": _standalone_artifact_references(
            qualification,
            device_probe,
            runtime_provenance,
        ),
        "native_reference": {
            "argv": list(native_launch.argv),
            "cwd": str(native_launch.cwd),
            "environment": dict(native_launch.environment),
            "stdout_path": str(inputs.native_reference_path.absolute()),
        },
        "runner_spec_sha256": canonical_sha256(runner_spec),
    }
    return Phase0WorkflowPlan(
        document=document,
        native_reference_launch=native_launch,
        receipt_template=receipt_template,
        runner_spec=runner_spec,
        qualification=qualification,
        device_probe=device_probe,
        runtime_provenance=runtime_provenance,
    )


def _validate_workflow_plan(plan: Phase0WorkflowPlan) -> None:
    lane_id = plan.document.get("lane_id")
    if lane_id not in (RTX_LANE_ID, A100_LANE_ID):
        raise Phase0WorkflowError("workflow plan lane_id is invalid")
    _validate_standalone_evidence(
        lane_id,
        plan.qualification,
        plan.device_probe,
        plan.runtime_provenance,
    )
    if plan.document.get("schema_id") != WORKFLOW_SCHEMA_ID:
        raise Phase0WorkflowError("workflow plan schema is invalid")
    expected_references = _standalone_artifact_references(
        plan.qualification,
        plan.device_probe,
        plan.runtime_provenance,
    )
    if plan.document.get("standalone_evidence") != expected_references:
        raise Phase0WorkflowError("workflow standalone artifact references drifted")
    if plan.runner_spec.get("provenance") != plan.runtime_provenance:
        raise Phase0WorkflowError("runner spec runtime provenance drifted")
    if plan.runner_spec.get("receipt_template") != plan.receipt_template:
        raise Phase0WorkflowError("runner spec receipt template drifted")
    lanes = _sequence(plan.receipt_template.get("lanes"), "receipt template lanes")
    target_lanes = [
        _mapping(lane, "receipt template lane")
        for lane in lanes
        if isinstance(lane, dict) and lane.get("lane_id") == lane_id
    ]
    if (
        len(target_lanes) != 1
        or target_lanes[0].get("qualification") != plan.qualification
    ):
        raise Phase0WorkflowError("standalone qualification differs from target lane")
    if plan.document.get("runner_spec_sha256") != canonical_sha256(plan.runner_spec):
        raise Phase0WorkflowError("workflow runner spec identity drifted")
    if plan.document.get("runtime_identity_sha256") != _runtime_identity(
        plan.runtime_provenance
    ):
        raise Phase0WorkflowError("workflow runtime identity drifted")


def write_phase0_workflow_plan(root: Path, plan: Phase0WorkflowPlan) -> None:
    """Validate and exclusively write canonical pre-gate artifacts."""

    _validate_workflow_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    artifacts = (
        ("workflow.json", plan.document),
        ("phase0-receipt-template.json", plan.receipt_template),
        ("c0-runner-spec.json", plan.runner_spec),
        ("qualification.json", plan.qualification),
        ("device-probe.json", plan.device_probe),
        ("runtime-provenance.json", plan.runtime_provenance),
    )
    expected_bytes = {
        name: canonical_json_bytes(document) for name, document in artifacts
    }
    for name, _document in artifacts:
        with (root / name).open("xb") as stream:
            stream.write(expected_bytes[name])
    for name, payload in expected_bytes.items():
        if (root / name).read_bytes() != payload:
            raise Phase0WorkflowError(f"written workflow artifact {name} drifted")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--import-attestation", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--lane-id", choices=(RTX_LANE_ID, A100_LANE_ID), required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--base-receipt", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compilation-cache", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--warm-sample-count", type=int, default=REQUIRED_WARM_SAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan = build_phase0_workflow(
            Phase0WorkflowInputs(
                snapshot_root=args.snapshot_root,
                publication_path=args.publication,
                import_attestation_path=args.import_attestation,
                interpreter=args.interpreter,
                output_root=args.output_root,
                compilation_cache_directory=args.compilation_cache,
                native_reference_path=args.native_reference,
                lane_id=args.lane_id,
                qualification_path=args.qualification,
                base_receipt_path=args.base_receipt,
                warm_sample_count=args.warm_sample_count,
            )
        )
        write_phase0_workflow_plan(args.plan_output, plan)
    except (OSError, ValueError, RuntimeError) as error:
        sys.stderr.write(f"Phase-0 workflow failed: {error}\n")
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(plan.document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
