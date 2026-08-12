"""Runner-owned launch receipts for diagnostic C0/C1/C2 trajectory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from benchmarks.single_stage_compute_graph_c0_runner import (
    CommandExecutor,
    CommandResult,
    _load_canonical_json_object,
    _sha256_path,
    _subprocess_executor,
)
from benchmarks.single_stage_compute_graph_canary_profile_runner import (
    validate_profile_count_evidence,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CanarySpec,
    _spec_identity,
    validate_spec,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    SnapshotModuleLaunch,
    build_snapshot_module_launch,
    normalize_route_environment,
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_snapshot import load_snapshot_manifest
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    validate_raw_trajectory_document,
)

PRODUCER_MODULE: Final = "benchmarks.single_stage_compute_graph_variant_trajectory"
PRODUCER_RELATIVE_PATH: Final = (
    "benchmarks/single_stage_compute_graph_variant_trajectory.py"
)
SCHEMA_ID: Final = "single-stage-compute-graph-variant-trajectory-launch-receipt-v1"
TIMEOUT_SECONDS: Final = 900.0
Lane = Literal["C0", "C1", "C2"]


class VariantTrajectoryRunnerError(RuntimeError):
    """Fresh-process diagnostic evidence is not provenance-authoritative."""


@dataclass(frozen=True, slots=True)
class VariantTrajectoryLaunch:
    """Exact snapshot child invocation and exclusive output destinations."""

    spec: CanarySpec
    spec_path: Path
    lane: Lane
    output_path: Path
    receipt_path: Path
    profile_count_output_path: Path | None = None
    canary_artifact_path: Path | None = None


def _relative_file(path: Path, root: Path, context: str) -> str:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise VariantTrajectoryRunnerError(f"{context} must be inside artifact root")
    return resolved.relative_to(resolved_root).as_posix()


def _validate_output_destinations(
    launch: VariantTrajectoryLaunch,
    artifact_root: Path,
) -> tuple[Path, ...]:
    paths = [launch.output_path, launch.receipt_path]
    if launch.profile_count_output_path is not None:
        paths.append(launch.profile_count_output_path)
    resolved_root = artifact_root.resolve()
    resolved_paths = tuple(path.resolve() for path in paths)
    if any(not path.is_relative_to(resolved_root) for path in resolved_paths):
        raise VariantTrajectoryRunnerError(
            "trajectory output paths must be inside artifact root"
        )
    if len(set(resolved_paths)) != len(resolved_paths):
        raise VariantTrajectoryRunnerError("trajectory output paths must be distinct")
    return tuple(paths)


def _producer_manifest_entry(spec: CanarySpec):
    entries, manifest_sha256 = load_snapshot_manifest(spec.snapshot_root)
    matches = tuple(
        entry for entry in entries if entry.relative_path == PRODUCER_RELATIVE_PATH
    )
    if (
        len(matches) != 1
        or matches[0].role != "benchmark"
        or manifest_sha256 != spec.snapshot_manifest_sha256
    ):
        raise VariantTrajectoryRunnerError(
            "trajectory producer is absent from the bound snapshot manifest"
        )
    return matches[0]


def _validate_launch_spec_binding(launch: VariantTrajectoryLaunch) -> None:
    persisted = validate_spec(
        _load_canonical_json_object(launch.spec_path, "canary spec")
    )
    if persisted != launch.spec:
        raise VariantTrajectoryRunnerError(
            "launch spec object differs from canonical spec bytes"
        )


def _diagnostic_environment(spec: CanarySpec) -> dict[str, str]:
    try:
        runtime_contract = json.loads(spec.runtime_contract_json)
    except json.JSONDecodeError as error:
        raise VariantTrajectoryRunnerError(
            "runtime contract is invalid JSON"
        ) from error
    if not isinstance(runtime_contract, dict):
        raise VariantTrajectoryRunnerError("runtime contract must be an object")
    static = runtime_contract.get("static_environment")
    if (
        not isinstance(static, dict)
        or not all(isinstance(key, str) for key in static)
        or not all(isinstance(value, str) for value in static.values())
        or normalize_static_timing_environment(static) != static
        or runtime_contract.get("expected_runtime_identity_sha256")
        != spec.runtime_identity_sha256
    ):
        raise VariantTrajectoryRunnerError("runtime contract binding is invalid")
    environment = dict(static)
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(spec.cache_directory),
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
        }
    )
    runtime_contract = dict(runtime_contract)
    runtime_contract["route_environment"] = normalize_route_environment(environment)
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"] = json.dumps(
        runtime_contract, sort_keys=True, separators=(",", ":")
    )
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY"] = (
        spec.runtime_identity_sha256
    )
    return environment


def _completion_document(result: CommandResult) -> dict[str, object]:
    return {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "elapsed_ns": result.elapsed_ns,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _validate_completion_document(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "returncode",
        "timed_out",
        "elapsed_ns",
        "stdout",
        "stderr",
    }:
        raise VariantTrajectoryRunnerError("child completion record is malformed")
    returncode = value["returncode"]
    timed_out = value["timed_out"]
    elapsed_ns = value["elapsed_ns"]
    stdout = value["stdout"]
    stderr = value["stderr"]
    if (
        isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or returncode != 0
        or timed_out is not False
        or isinstance(elapsed_ns, bool)
        or not isinstance(elapsed_ns, int)
        or elapsed_ns <= 0
        or stdout != ""
        or stderr != ""
    ):
        raise VariantTrajectoryRunnerError(
            "child completion record is not a silent successful execution"
        )
    return dict(value)


def build_variant_trajectory_launch(
    launch: VariantTrajectoryLaunch,
) -> SnapshotModuleLaunch:
    """Build the manifested diagnostic child from validated canary bindings."""

    _validate_launch_spec_binding(launch)
    if launch.lane in ("C1", "C2") and launch.lane != launch.spec.variant:
        raise VariantTrajectoryRunnerError("variant lane differs from canary spec")
    if (launch.profile_count_output_path is None) != (
        launch.canary_artifact_path is None
    ):
        raise VariantTrajectoryRunnerError(
            "profile-count output and canary artifact must be supplied together"
        )
    if launch.lane == "C0" and launch.profile_count_output_path is not None:
        raise VariantTrajectoryRunnerError("C0 cannot produce dense profile counts")
    _producer_manifest_entry(launch.spec)
    diagnostic_environment = _diagnostic_environment(launch.spec)
    module_args = [
        "--spec",
        str(launch.spec_path.resolve()),
        "--lane",
        launch.lane,
        "--output",
        str(launch.output_path.resolve()),
    ]
    if launch.profile_count_output_path is not None:
        module_args.extend(
            (
                "--profile-count-output",
                str(launch.profile_count_output_path.resolve()),
                "--canary-artifact",
                str(launch.canary_artifact_path.resolve()),
            )
        )
    return build_snapshot_module_launch(
        launch.spec.interpreter_path,
        launch.spec.snapshot_root,
        PRODUCER_MODULE,
        module_args,
        diagnostic_environment,
    )


def _expected_raw_identity(launch: VariantTrajectoryLaunch) -> dict[str, str]:
    if _sha256_path(launch.spec.native_reference_path) != (
        launch.spec.native_reference_sha256
    ):
        raise VariantTrajectoryRunnerError("native reference bytes differ from spec")
    native_reference = _load_canonical_json_object(
        launch.spec.native_reference_path, "native reference"
    )
    identity = native_reference.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(
        identity.get("input_bundle_sha256"), str
    ):
        raise VariantTrajectoryRunnerError(
            "validated input-bundle identity is unavailable"
        )
    return {
        "lane": launch.lane,
        "parameter_sha256": launch.spec.parameter_sha256,
        "specimen_sha256": launch.spec.specimen_sha256,
        "input_bundle_sha256": identity["input_bundle_sha256"],
        "solver_graph_sha256": launch.spec.solver_graph_sha256,
        "source_sha256": launch.spec.source_state_sha256,
    }


def _receipt_document(
    launch: VariantTrajectoryLaunch,
    child: SnapshotModuleLaunch,
    completion: Mapping[str, object],
    *,
    artifact_root: Path,
) -> dict[str, object]:
    _validate_launch_spec_binding(launch)
    producer = _producer_manifest_entry(launch.spec)
    raw_bytes = launch.output_path.read_bytes()
    raw = _load_canonical_json_object(launch.output_path, "variant raw trajectory")
    normalized = validate_raw_trajectory_document(raw)
    expected_raw_identity = _expected_raw_identity(launch)
    if any(normalized[key] != value for key, value in expected_raw_identity.items()):
        raise VariantTrajectoryRunnerError(
            "raw trajectory identity differs from launch"
        )
    profile_count_binding = None
    if launch.profile_count_output_path is not None:
        canary_artifact_sha256 = _sha256_path(launch.canary_artifact_path)
        profile_counts = _load_canonical_json_object(
            launch.profile_count_output_path, "profile count evidence"
        )
        validate_profile_count_evidence(
            profile_counts,
            spec=launch.spec,
            canary_artifact_sha256=canary_artifact_sha256,
        )
        profile_count_binding = {
            "relative_path": _relative_file(
                launch.profile_count_output_path,
                artifact_root,
                "profile count output",
            ),
            "sha256": _sha256_path(launch.profile_count_output_path),
            "canary_artifact_sha256": canary_artifact_sha256,
        }
    child_environment = dict(child.environment)
    return {
        "schema_id": SCHEMA_ID,
        "state": "PRODUCED",
        "evidence_kind": "fresh_process_oracle_outside_promotion_timing",
        "promotion_timing": False,
        "lane": launch.lane,
        "spec_sha256": _sha256_path(launch.spec_path),
        "producer": {
            "module": PRODUCER_MODULE,
            "relative_path": producer.relative_path,
            "manifest_role": producer.role,
            "sha256": producer.sha256,
        },
        "runtime": {
            "interpreter_path": str(launch.spec.interpreter_path),
            "interpreter_sha256": _sha256_path(launch.spec.interpreter_path),
            "runtime_identity_sha256": launch.spec.runtime_identity_sha256,
            "snapshot_root": str(launch.spec.snapshot_root),
            "snapshot_manifest_sha256": launch.spec.snapshot_manifest_sha256,
        },
        "launch": {
            "argv": list(child.argv),
            "argv_sha256": hashlib.sha256(canonical_json_bytes(child.argv)).hexdigest(),
            "working_directory": str(child.cwd),
            "environment_sha256": hashlib.sha256(
                canonical_json_bytes(child_environment)
            ).hexdigest(),
            "completion": dict(completion),
        },
        "identity": {
            **_spec_identity(launch.spec),
            "raw": expected_raw_identity,
        },
        "raw": {
            "relative_path": _relative_file(
                launch.output_path, artifact_root, "variant raw output"
            ),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
        "profile_counts": profile_count_binding,
    }


def launch_variant_trajectory(
    launch: VariantTrajectoryLaunch,
    *,
    artifact_root: Path,
    executor: CommandExecutor = _subprocess_executor,
) -> Mapping[str, object]:
    """Execute one bounded snapshot child and exclusively write its receipt."""

    output_paths = _validate_output_destinations(launch, artifact_root)
    if any(path.exists() for path in output_paths):
        raise VariantTrajectoryRunnerError("trajectory output paths must not exist")
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    child = build_variant_trajectory_launch(launch)
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
        raise VariantTrajectoryRunnerError(
            "variant trajectory child did not complete silently and successfully"
        )
    completion = _validate_completion_document(_completion_document(completed))
    document = _receipt_document(
        launch,
        child,
        completion,
        artifact_root=artifact_root,
    )
    with launch.receipt_path.open("xb") as stream:
        stream.write(canonical_json_bytes(document))
    return document


def validate_variant_trajectory_launch(
    receipt_path: Path,
    launch: VariantTrajectoryLaunch,
    *,
    artifact_root: Path,
) -> Mapping[str, object]:
    """Rebuild a receipt from current bound bytes and the exact child launch."""

    document = _load_canonical_json_object(receipt_path, "trajectory launch receipt")
    child = build_variant_trajectory_launch(launch)
    persisted_launch = document.get("launch")
    if not isinstance(persisted_launch, Mapping):
        raise VariantTrajectoryRunnerError("receipt launch record is malformed")
    completion = _validate_completion_document(persisted_launch.get("completion"))
    expected = _receipt_document(
        launch,
        child,
        completion,
        artifact_root=artifact_root,
    )
    if document != expected:
        raise VariantTrajectoryRunnerError(
            "trajectory launch receipt differs from live bound bytes"
        )
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lane", choices=("C0", "C1", "C2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--profile-count-output", type=Path)
    parser.add_argument("--canary-artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = validate_spec(_load_canonical_json_object(args.spec, "canary spec"))
    launch = VariantTrajectoryLaunch(
        spec=spec,
        spec_path=args.spec,
        lane=args.lane,
        output_path=args.output,
        receipt_path=args.receipt,
        profile_count_output_path=args.profile_count_output,
        canary_artifact_path=args.canary_artifact,
    )
    launch_variant_trajectory(launch, artifact_root=args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VariantTrajectoryLaunch",
    "VariantTrajectoryRunnerError",
    "build_variant_trajectory_launch",
    "launch_variant_trajectory",
    "validate_variant_trajectory_launch",
]
