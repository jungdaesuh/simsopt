"""Runner-owned fresh-process launch contract for native trajectory evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmarks.single_stage_compute_graph_c0_runner import (
    CommandExecutor,
    CommandResult,
    _subprocess_executor,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    SnapshotModuleLaunch,
    build_snapshot_module_launch,
)
from benchmarks.single_stage_compute_graph_native_reference import (
    NativeReferenceBinding,
    _sha256_path,
)
from benchmarks.single_stage_compute_graph_native_trajectory import (
    _canonical_solver_graph_sha256,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_snapshot import load_snapshot_manifest
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    validate_raw_trajectory_document,
)

SCHEMA_ID: Final = "single-stage-native-trajectory-launch-receipt-v1"
PRODUCER_MODULE: Final = "benchmarks.single_stage_compute_graph_native_trajectory"
TIMEOUT_SECONDS: Final = 900.0


class NativeTrajectoryRunnerError(RuntimeError):
    """The fresh-process trajectory launch is not provenance-authoritative."""


@dataclass(frozen=True, slots=True)
class NativeTrajectoryLaunch:
    snapshot_root: Path
    snapshot_publication_path: Path
    import_attestation_path: Path
    snapshot_manifest_sha256: str
    snapshot_publication_sha256: str
    import_attestation_sha256: str
    input_root: Path
    candidate_path: Path
    solver_graph_path: Path
    output_path: Path
    receipt_path: Path
    parameter_sha256: str
    binding: NativeReferenceBinding


def _validate_snapshot_authority(launch: NativeTrajectoryLaunch) -> None:
    _entries, manifest_sha256 = load_snapshot_manifest(launch.snapshot_root)
    if (
        manifest_sha256 != launch.snapshot_manifest_sha256
        or _sha256_path(launch.snapshot_publication_path)
        != launch.snapshot_publication_sha256
        or _sha256_path(launch.import_attestation_path)
        != launch.import_attestation_sha256
    ):
        raise NativeTrajectoryRunnerError("snapshot authority bytes differ from launch")


def _runtime_environment(binding: NativeReferenceBinding) -> dict[str, str]:
    contract = binding.runtime_contract
    static = contract.get("static_environment")
    route = contract.get("route_environment")
    if (
        not isinstance(static, Mapping)
        or not isinstance(route, Mapping)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for environment in (static, route)
            for key, value in environment.items()
        )
    ):
        raise NativeTrajectoryRunnerError("native runtime environment is invalid")
    environment = {**static, **route}
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"] = canonical_json_bytes(
        contract
    ).decode("utf-8")
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY"] = (
        binding.runtime_identity_sha256
    )
    return environment


def _snapshot_launch(launch: NativeTrajectoryLaunch) -> SnapshotModuleLaunch:
    binding = launch.binding
    snapshot_root = launch.snapshot_root.resolve()
    if (
        not launch.input_root.resolve().is_relative_to(snapshot_root)
        or not launch.candidate_path.resolve().is_relative_to(snapshot_root)
        or not launch.solver_graph_path.resolve().is_relative_to(snapshot_root)
        or not Path(binding.native_simsoptpp_path)
        .resolve()
        .is_relative_to(snapshot_root)
    ):
        raise NativeTrajectoryRunnerError(
            "native launch inputs and extension must come from the snapshot"
        )
    _validate_snapshot_authority(launch)
    module_args = (
        "--input-root",
        str(launch.input_root.resolve()),
        "--candidate",
        str(launch.candidate_path.resolve()),
        "--output",
        str(launch.output_path.resolve()),
        "--parameter-sha256",
        launch.parameter_sha256,
        "--input-fingerprint",
        binding.input_fingerprint,
        "--input-bundle-sha256",
        binding.input_bundle_sha256,
        "--configuration-fingerprint",
        binding.configuration_fingerprint,
        "--specimen-sha256",
        binding.specimen_sha256,
        "--source-sha256",
        binding.source_sha256,
        "--solver-graph",
        str(launch.solver_graph_path.resolve()),
        "--runtime-identity-sha256",
        binding.runtime_identity_sha256,
        "--interpreter-path",
        binding.interpreter_path,
        "--native-simsoptpp-path",
        binding.native_simsoptpp_path,
        "--native-simsoptpp-sha256",
        binding.native_simsoptpp_sha256,
        "--runtime-contract-json",
        canonical_json_bytes(binding.runtime_contract).decode("utf-8"),
    )
    return build_snapshot_module_launch(
        Path(binding.interpreter_path),
        launch.snapshot_root,
        PRODUCER_MODULE,
        module_args,
        _runtime_environment(binding),
    )


def _relative_file(path: Path, root: Path, context: str) -> str:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise NativeTrajectoryRunnerError(f"{context} must be inside artifact root")
    return resolved.relative_to(resolved_root).as_posix()


def _validated_completion(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(
        {"returncode", "timed_out", "elapsed_ns", "stdout", "stderr"}
    ):
        raise NativeTrajectoryRunnerError("native child completion fields are invalid")
    returncode = value.get("returncode")
    timed_out = value.get("timed_out")
    elapsed_ns = value.get("elapsed_ns")
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    if (
        type(returncode) is not int
        or returncode != 0
        or type(timed_out) is not bool
        or timed_out
        or type(elapsed_ns) is not int
        or elapsed_ns < 1
        or stdout != ""
        or stderr != ""
    ):
        raise NativeTrajectoryRunnerError("native child completion is not successful")
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_ns": elapsed_ns,
        "stdout": stdout,
        "stderr": stderr,
    }


def _completion_from_result(result: CommandResult) -> dict[str, object]:
    return _validated_completion(
        {
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "elapsed_ns": result.elapsed_ns,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


def _receipt_document(
    launch: NativeTrajectoryLaunch,
    *,
    artifact_root: Path,
    child: SnapshotModuleLaunch,
    completion: Mapping[str, object],
) -> dict[str, object]:
    raw_bytes = launch.output_path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw_bytes != canonical_json_bytes(raw):
        raise NativeTrajectoryRunnerError("native trajectory output is not canonical")
    normalized = validate_raw_trajectory_document(raw)
    solver_graph_sha256 = _canonical_solver_graph_sha256(launch.solver_graph_path)
    native_extension = Path(launch.binding.native_simsoptpp_path)
    if _sha256_path(native_extension) != launch.binding.native_simsoptpp_sha256:
        raise NativeTrajectoryRunnerError("native extension bytes differ from launch")
    expected_identity = {
        "lane": "native",
        "parameter_sha256": launch.parameter_sha256,
        "specimen_sha256": launch.binding.specimen_sha256,
        "input_bundle_sha256": launch.binding.input_bundle_sha256,
        "solver_graph_sha256": solver_graph_sha256,
        "source_sha256": launch.binding.source_sha256,
    }
    if any(normalized[key] != value for key, value in expected_identity.items()):
        raise NativeTrajectoryRunnerError("native raw identity differs from launch")
    snapshot_root = launch.snapshot_root.resolve()
    _validate_snapshot_authority(launch)
    producer = snapshot_root / (PRODUCER_MODULE.replace(".", "/") + ".py")
    publication = launch.snapshot_publication_path.resolve()
    import_attestation = launch.import_attestation_path.resolve()
    return {
        "schema_id": SCHEMA_ID,
        "state": "PRODUCED",
        "producer": {
            "path": str(producer),
            "sha256": _sha256_path(producer),
            "working_directory": str(child.cwd),
        },
        "snapshot": {
            "root": str(snapshot_root),
            "manifest_sha256": launch.snapshot_manifest_sha256,
            "publication_path": str(publication),
            "publication_sha256": launch.snapshot_publication_sha256,
            "import_attestation_path": str(import_attestation),
            "import_attestation_sha256": launch.import_attestation_sha256,
        },
        "command_sha256": hashlib.sha256(
            canonical_json_bytes(list(child.argv))
        ).hexdigest(),
        "environment_sha256": hashlib.sha256(
            canonical_json_bytes(dict(child.environment))
        ).hexdigest(),
        "completion": dict(completion),
        "runtime": {
            "runtime_identity_sha256": launch.binding.runtime_identity_sha256,
            "interpreter_path": launch.binding.interpreter_path,
            "interpreter_sha256": _sha256_path(Path(launch.binding.interpreter_path)),
            "native_simsoptpp_path": launch.binding.native_simsoptpp_path,
            "native_simsoptpp_sha256": _sha256_path(native_extension),
        },
        "identity": expected_identity,
        "raw": {
            "relative_path": _relative_file(
                launch.output_path, artifact_root, "native raw output"
            ),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
    }


def launch_native_trajectory(
    launch: NativeTrajectoryLaunch,
    *,
    artifact_root: Path,
    executor: CommandExecutor = _subprocess_executor,
) -> Mapping[str, object]:
    """Launch the native producer once and exclusively write its launch receipt."""

    if launch.output_path.exists() or launch.receipt_path.exists():
        raise NativeTrajectoryRunnerError("native output paths must not already exist")
    launch.output_path.parent.mkdir(parents=True, exist_ok=True)
    launch.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    child = _snapshot_launch(launch)
    completed = executor(
        child.argv,
        child.environment,
        child.cwd,
        TIMEOUT_SECONDS,
    )
    if (
        completed.timed_out
        or completed.returncode != 0
        or completed.stdout
        or completed.stderr
    ):
        raise NativeTrajectoryRunnerError(
            "native trajectory child did not complete silently and successfully"
        )
    document = _receipt_document(
        launch,
        artifact_root=artifact_root,
        child=child,
        completion=_completion_from_result(completed),
    )
    with launch.receipt_path.open("xb") as stream:
        stream.write(canonical_json_bytes(document))
    return document


def validate_native_trajectory_launch(
    receipt_path: Path,
    launch: NativeTrajectoryLaunch,
    *,
    artifact_root: Path,
) -> Mapping[str, object]:
    """Rebuild all stable receipt fields and rehash the raw/runtime producer bytes."""

    raw = receipt_path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or raw != canonical_json_bytes(document):
        raise NativeTrajectoryRunnerError("native launch receipt is not canonical")
    completion = _validated_completion(document.get("completion"))
    expected = _receipt_document(
        launch,
        artifact_root=artifact_root,
        child=_snapshot_launch(launch),
        completion=completion,
    )
    if document != expected:
        raise NativeTrajectoryRunnerError(
            "native launch receipt differs from live bytes"
        )
    return document


__all__ = [
    "NativeTrajectoryLaunch",
    "NativeTrajectoryRunnerError",
    "launch_native_trajectory",
    "validate_native_trajectory_launch",
]
