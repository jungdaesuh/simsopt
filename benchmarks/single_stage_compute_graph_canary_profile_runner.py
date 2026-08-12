"""Collect separate, non-timing C1/C2 GPU profile evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmarks.single_stage_compute_graph_c0_runner import (
    CommandExecutor,
    CommandResult,
    _load_canonical_json_object,
    _sha256_path,
    _subprocess_executor,
    _write_exclusive_json,
)
from benchmarks.single_stage_compute_graph_canary_profile import (
    PROFILE_CHILD_SCHEMA_ID,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CanarySpec,
    _gate_parity,
    _sha,
    _spec_identity,
    _validated_variant_telemetry,
    child_launches,
    validate_spec,
)
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    CUDA_GRAPH_TRACE_GRANULARITY,
    SqliteLaneEvidence,
    parse_nsys_sqlite,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    SnapshotModuleLaunch,
    build_snapshot_module_launch,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    canonical_json_bytes,
)
from benchmarks.single_stage_compute_graph_profile import (
    ComputeGraphProfile,
    summarize_compute_graph_profile,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    load_trace_document,
    union_intervals,
)

PROFILE_MODULE: Final = "benchmarks.single_stage_compute_graph_canary_profile"
PROFILE_SCHEMA_ID: Final = "single-stage-compute-graph-canary-profile-evidence-v1"
PROFILE_COUNT_SCHEMA_ID: Final = "single-stage-compute-graph-canary-profile-counts-v1"
PROFILE_TIMEOUT_SECONDS: Final = 900.0


class CanaryProfileRunnerError(RuntimeError):
    """The profile lane cannot produce validated evidence."""


@dataclass(frozen=True, slots=True)
class ProfileLaunch:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path
    trace_root: Path
    report_path: Path
    sqlite_path: Path


def build_profile_launch(
    spec: CanarySpec,
    *,
    nsys_binary: Path,
    nvtx_library: Path,
    output_root: Path,
    base_environment: Mapping[str, str],
) -> ProfileLaunch:
    """Build one exact manifested snapshot child wrapped by Nsight Systems."""

    if not nsys_binary.is_file() or not os.access(nsys_binary, os.X_OK):
        raise CanaryProfileRunnerError("nsys binary must be executable")
    if not nvtx_library.is_file():
        raise CanaryProfileRunnerError("NVTX library must be a regular file")
    gate_launch = child_launches(spec, base_environment)[1]
    initial_reference = spec.native_initial_reference
    if not isinstance(initial_reference, Mapping) or not isinstance(
        initial_reference.get("parameter_sha256"), str
    ):
        raise CanaryProfileRunnerError("validated initial parameter SHA is unavailable")
    trace_root = output_root / "jax-trace"
    module_args = (
        "--variant",
        spec.variant,
        "--input-root",
        str(spec.input_root),
        "--candidate",
        str(spec.candidate_path),
        "--parameter-sha256",
        spec.parameter_sha256,
        "--initial-parameter-sha256",
        str(initial_reference["parameter_sha256"]),
        "--snapshot-root",
        str(spec.snapshot_root),
        "--trace-root",
        str(trace_root),
        "--nvtx-library",
        str(nvtx_library.resolve()),
    )
    child: SnapshotModuleLaunch = build_snapshot_module_launch(
        spec.interpreter_path,
        spec.snapshot_root,
        PROFILE_MODULE,
        module_args,
        gate_launch.environment,
    )
    prefix = output_root / "nsys-report"
    return ProfileLaunch(
        command=(
            str(nsys_binary.resolve()),
            "profile",
            "--trace=cuda,nvtx",
            f"--cuda-graph-trace={CUDA_GRAPH_TRACE_GRANULARITY}",
            "--export=sqlite",
            "--force-overwrite=false",
            f"--output={prefix}",
            *child.argv,
        ),
        environment=child.environment,
        cwd=child.cwd,
        trace_root=trace_root,
        report_path=prefix.with_suffix(".nsys-rep"),
        sqlite_path=prefix.with_suffix(".sqlite"),
    )


def _profile_child(result: CommandResult) -> Mapping[str, object]:
    if result.timed_out:
        raise CanaryProfileRunnerError("profile child timed out after 900 seconds")
    try:
        document = json.loads(
            result.stdout,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CanaryProfileRunnerError(f"profile child emitted nonfinite {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise CanaryProfileRunnerError("profile child emitted invalid JSON") from error
    if not isinstance(document, dict) or document.get("schema_id") != (
        PROFILE_CHILD_SCHEMA_ID
    ):
        raise CanaryProfileRunnerError("profile child schema is invalid")
    if result.returncode != 0 or document.get("status") != "PASS":
        raise CanaryProfileRunnerError("profile child did not pass")
    if result.stdout.encode("utf-8") != canonical_json_bytes(document):
        raise CanaryProfileRunnerError("profile child output is not canonical JSON")
    return document


def _phase_rows(profile: ComputeGraphProfile) -> list[dict[str, object]]:
    rows = []
    for phase_id, intervals in profile.phase_interval_unions:
        rows.append(
            {
                "phase_id": phase_id,
                "device_interval_count": len(intervals),
                "device_interval_union_ns": union_intervals(intervals)[1],
                "intervals": [
                    [interval.start_ns, interval.end_ns] for interval in intervals
                ],
            }
        )
    return rows


def _required_operation_evidence(
    telemetry: Mapping[str, object],
    profile: ComputeGraphProfile,
    counts: Mapping[str, int] | None,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    phase_times = {
        phase_id: union_intervals(intervals)[1]
        for phase_id, intervals in profile.phase_interval_unions
    }
    evidence: dict[str, object] = {
        "residual": {
            "count": None if counts is None else counts["residual_evaluation_count"],
            "device_interval_union_ns": phase_times.get("newton.residual_jvp"),
        },
        "jacobian_construction": {
            "count": telemetry["exact_newton_variant_dense_materialization_count"],
            "device_interval_union_ns": phase_times.get("newton.jacobian_construction"),
        },
        "dense_materialization": {
            "count": telemetry["exact_newton_variant_dense_materialization_count"],
            "device_interval_union_ns": phase_times.get("newton.dense_materialization"),
        },
        "lu_factorization": {
            "count": telemetry["exact_newton_variant_lu_factorization_count"],
            "device_interval_union_ns": phase_times.get("newton.lu_factor"),
        },
        "refinement": {
            "count": telemetry["exact_newton_variant_refinement_correction_count"],
            "device_interval_union_ns": (
                phase_times.get("newton.refinement")
                if telemetry["exact_newton_variant_refinement_correction_count"]
                else 0
            ),
        },
        "linearized_tangent_traversals": {
            "primal_traversal_count": (
                None if counts is None else counts["dense_primal_traversal_count"]
            ),
            "tangent_batch_count": (
                None if counts is None else counts["dense_tangent_batch_count"]
            ),
            "tangent_direction_count": (
                None if counts is None else counts["dense_tangent_direction_count"]
            ),
        },
    }
    missing = []
    if counts is None:
        missing.extend(
            (
                {
                    "field": "residual.count",
                    "required_source_hook": "separate_variant_oracle_profile_counts",
                },
                {
                    "field": "linearized_tangent_traversals",
                    "required_source_hook": "separate_variant_oracle_profile_counts",
                },
            )
        )
    required_phases = (
        ("jacobian_construction", "newton.jacobian_construction"),
        ("dense_materialization", "newton.dense_materialization"),
        ("lu_factorization", "newton.lu_factor"),
    )
    for field, phase_id in required_phases:
        if phase_id not in phase_times:
            missing.append(
                {
                    "field": f"{field}.device_interval_union_ns",
                    "required_source_hook": f"PhaseId.{phase_id.upper().replace('.', '_')}",
                }
            )
    if (
        telemetry["exact_newton_variant_refinement_correction_count"]
        and "newton.refinement" not in phase_times
    ):
        missing.append(
            {
                "field": "refinement.device_interval_union_ns",
                "required_source_hook": "PhaseId.NEWTON_REFINEMENT",
            }
        )
    return evidence, missing


def _revalidate_profile_numerics(
    child: Mapping[str, object], gate: Mapping[str, object], spec: CanarySpec
) -> None:
    required = (
        "objective_dtype",
        "objective",
        "gradient_dtype",
        "gradient",
        "inner_newton_success",
        "adjoint_success",
        "residual_certificates",
    )
    if (
        child.get("status") != "PASS"
        or child.get("mode") != "profile"
        or child.get("variant") != spec.variant
        or child.get("parameter_sha256") != spec.parameter_sha256
    ):
        raise CanaryProfileRunnerError("profile child identity/status is invalid")
    if any(child.get(field) != gate.get(field) for field in required):
        raise CanaryProfileRunnerError(
            "profile replay differs from the changed-state canary gate"
        )


def validate_profile_count_evidence(
    document: Mapping[str, object],
    *,
    spec: CanarySpec,
    canary_artifact_sha256: str,
) -> dict[str, int]:
    """Validate separately produced oracle counts against exact canary bytes."""

    if (
        set(document) != {"schema_id", "identity", "counts"}
        or document.get("schema_id") != PROFILE_COUNT_SCHEMA_ID
    ):
        raise CanaryProfileRunnerError("profile count evidence schema is invalid")
    expected_identity = {
        **_spec_identity(spec),
        "canary_artifact_sha256": canary_artifact_sha256,
    }
    if document.get("identity") != expected_identity:
        raise CanaryProfileRunnerError("profile count evidence identity differs")
    counts = document.get("counts")
    expected_keys = {
        "residual_evaluation_count",
        "dense_primal_traversal_count",
        "dense_tangent_batch_count",
        "dense_tangent_direction_count",
    }
    if not isinstance(counts, dict) or set(counts) != expected_keys:
        raise CanaryProfileRunnerError("profile counts are incomplete")
    normalized: dict[str, int] = {}
    for key in sorted(expected_keys):
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CanaryProfileRunnerError(f"profile count {key} is invalid")
        normalized[key] = value
    return normalized


def build_profile_artifact(
    *,
    spec: CanarySpec,
    canary_artifact: Mapping[str, object],
    canary_artifact_sha256: str,
    child: Mapping[str, object],
    profile: ComputeGraphProfile,
    nsys: SqliteLaneEvidence,
    trace_path: Path,
    report_path: Path,
    sqlite_path: Path,
    nsys_binary: Path,
    nsys_version: str,
    nvtx_library: Path,
    profile_counts: Mapping[str, int] | None = None,
    raw_child_path: Path | None = None,
    profile_count_evidence_path: Path | None = None,
    profile_launch: ProfileLaunch | None = None,
    version_probe_path: Path | None = None,
) -> dict[str, object]:
    """Bind raw profile facts without promoting unsupported phase claims."""

    if canary_artifact.get("identity") != _spec_identity(spec):
        raise CanaryProfileRunnerError("canary artifact identity differs from spec")
    if canary_artifact.get("status") != "MEASURED_NONPROMOTING":
        raise CanaryProfileRunnerError("canary gates/warm route are incomplete")
    gate = canary_artifact.get("gate")
    if not isinstance(gate, dict):
        raise CanaryProfileRunnerError("canary changed-state gate is missing")
    native = spec.native_reference
    if native is None:
        raise CanaryProfileRunnerError("validated native reference is unavailable")
    _gate_parity(gate, native, spec.variant)
    _revalidate_profile_numerics(child, gate, spec)
    telemetry = child.get("telemetry")
    capture = child.get("capture")
    if not isinstance(telemetry, dict) or not isinstance(capture, dict):
        raise CanaryProfileRunnerError("profile child evidence is incomplete")
    _validated_variant_telemetry(spec.variant, telemetry, "profile telemetry")
    if (
        capture.get("hlo_module_set_identity") != profile.hlo_module_set_identity
        or capture.get("kernel_launch_count") != profile.kernel_launch_count
        or capture.get("pjrt_execute_count") != profile.pjrt_execute_count
    ):
        raise CanaryProfileRunnerError("child and recomputed trace topology differ")
    hlo_ir_sha256 = _sha(capture.get("hlo_ir_sha256"), "profile HLO IR SHA")
    operations, missing = _required_operation_evidence(
        telemetry, profile, profile_counts
    )
    if raw_child_path is None:
        missing.append(
            {
                "field": "raw.raw_child",
                "required_source_hook": "profile_runner_raw_child_receipt",
            }
        )
    if profile_count_evidence_path is None:
        missing.append(
            {
                "field": "raw.profile_counts",
                "required_source_hook": "separate_variant_oracle_profile_counts",
            }
        )
    if profile_launch is None:
        missing.append(
            {
                "field": "profile_launch",
                "required_source_hook": "profile_runner_launch_receipt",
            }
        )
    if version_probe_path is None:
        missing.append(
            {
                "field": "raw.version_probe",
                "required_source_hook": "profile_runner_version_probe_receipt",
            }
        )
    newton_intervals = tuple(
        interval
        for phase_id, intervals in profile.phase_interval_unions
        if phase_id.startswith("newton.")
        for interval in intervals
    )
    return {
        "schema_id": PROFILE_SCHEMA_ID,
        "status": "BLOCKED" if missing else "PRODUCED",
        "promotion_timing": False,
        "profile_launch": (
            None
            if profile_launch is None
            else {
                "command": list(profile_launch.command),
                "command_sha256": hashlib.sha256(
                    canonical_json_bytes(list(profile_launch.command))
                ).hexdigest(),
                "environment_sha256": hashlib.sha256(
                    canonical_json_bytes(dict(profile_launch.environment))
                ).hexdigest(),
                "working_directory": str(profile_launch.cwd),
                "timeout_seconds": PROFILE_TIMEOUT_SECONDS,
            }
        ),
        "identity": {
            **_spec_identity(spec),
            "canary_artifact_sha256": canary_artifact_sha256,
        },
        "numerical_revalidation": dict(child),
        "hlo_topology": {
            "lowered_hlo_ir_sha256": hlo_ir_sha256,
            "module_name_set_identity": profile.hlo_module_set_identity,
            "module_names": capture.get("hlo_modules"),
            "identity_ceiling": (
                "lowered_canonical_HLO_IR_plus_executed_trace_module_name_set"
            ),
        },
        "launches": {
            "pjrt_execute_count": profile.pjrt_execute_count,
            "jax_kernel_launch_count": profile.kernel_launch_count,
            "nsys_kernel_activity_count": nsys.total_device_activity.count,
            "nsys_cuda_graph_launch_api_count": nsys.graph_launch.count,
            "nsys_uncaptured_kernel_activity_count": (
                nsys.uncaptured_device_activity.count
            ),
        },
        "device": {
            "interval_union_ns": profile.device_active_ns,
            "evaluation_envelope_ns": profile.evaluation_envelope_ns,
            "inter_launch_gap_ns": profile.inter_launch_gap_ns,
            "nsys_interval_union_ns": nsys.total_device_activity.duration_ns,
        },
        "newton_only": {
            "device_interval_count": len(newton_intervals),
            "device_interval_union_ns": union_intervals(newton_intervals)[1],
        },
        "phases": _phase_rows(profile),
        "required_operations": operations,
        "raw": {
            "version_probe": (
                None
                if version_probe_path is None
                else {
                    "path": str(version_probe_path),
                    "sha256": _sha256_path(version_probe_path),
                }
            ),
            "raw_child": (
                None
                if raw_child_path is None
                else {
                    "path": str(raw_child_path),
                    "sha256": _sha256_path(raw_child_path),
                }
            ),
            "profile_counts": (
                None
                if profile_count_evidence_path is None
                else {
                    "path": str(profile_count_evidence_path),
                    "sha256": _sha256_path(profile_count_evidence_path),
                }
            ),
            "jax_trace": {"path": str(trace_path), "sha256": _sha256_path(trace_path)},
            "nsys_report": {
                "path": str(report_path),
                "sha256": _sha256_path(report_path),
            },
            "nsys_sqlite": {
                "path": str(sqlite_path),
                "sha256": _sha256_path(sqlite_path),
            },
        },
        "tool": {
            "nsys_binary_path": str(nsys_binary),
            "nsys_binary_sha256": _sha256_path(nsys_binary),
            "nsys_version": nsys_version,
            "nvtx_library_path": str(nvtx_library),
            "nvtx_library_sha256": _sha256_path(nvtx_library),
        },
        "missing_required_source_hooks": missing,
    }


def _single_trace(trace_root: Path) -> Path:
    matches = tuple(sorted(trace_root.rglob("*.trace.json.gz")))
    if len(matches) != 1:
        raise CanaryProfileRunnerError("profile must produce exactly one JAX trace")
    return matches[0]


def execute_profile_launch(
    launch: ProfileLaunch, executor: CommandExecutor
) -> CommandResult:
    """Execute one profile child at the fixed 900-second boundary."""

    return executor(
        launch.command,
        launch.environment,
        launch.cwd,
        PROFILE_TIMEOUT_SECONDS,
    )


def run_profile(
    *,
    spec: CanarySpec,
    canary_artifact_path: Path,
    nsys_binary: Path,
    nvtx_library: Path,
    expected_nsys_version: str,
    output_root: Path,
    profile_count_evidence_path: Path | None = None,
    executor: CommandExecutor = _subprocess_executor,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run one bounded external profile and write canonical evidence."""

    canary_artifact = _load_canonical_json_object(
        canary_artifact_path, "canary artifact"
    )
    canary_artifact_sha256 = _sha256_path(canary_artifact_path)
    profile_counts = None
    if profile_count_evidence_path is not None:
        profile_counts = validate_profile_count_evidence(
            _load_canonical_json_object(
                profile_count_evidence_path, "profile count evidence"
            ),
            spec=spec,
            canary_artifact_sha256=canary_artifact_sha256,
        )
    output_root.mkdir(parents=True, exist_ok=False)
    launch = build_profile_launch(
        spec,
        nsys_binary=nsys_binary.resolve(),
        nvtx_library=nvtx_library.resolve(),
        output_root=output_root,
        base_environment=os.environ if environment is None else environment,
    )
    version_result = executor(
        (str(nsys_binary.resolve()), "--version"),
        launch.environment,
        launch.cwd,
        30.0,
    )
    version_probe_path = output_root / "raw-version-probe.json"
    _write_exclusive_json(
        version_probe_path,
        {
            "returncode": version_result.returncode,
            "timed_out": version_result.timed_out,
            "elapsed_ns": version_result.elapsed_ns,
            "stdout": version_result.stdout,
            "stderr": version_result.stderr,
        },
    )
    if version_result.timed_out or version_result.returncode != 0:
        raise CanaryProfileRunnerError("Nsight version probe failed")
    version = version_result.stdout.strip()
    if version != expected_nsys_version.strip():
        raise CanaryProfileRunnerError("Nsight version differs from requested version")
    result = execute_profile_launch(launch, executor)
    raw_child_path = output_root / "raw-child.json"
    _write_exclusive_json(
        raw_child_path,
        {
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "elapsed_ns": result.elapsed_ns,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    child = _profile_child(result)
    trace_path = _single_trace(launch.trace_root)
    if not launch.report_path.is_file() or not launch.sqlite_path.is_file():
        raise CanaryProfileRunnerError("Nsight report/SQLite output is incomplete")
    profile = summarize_compute_graph_profile(
        load_trace_document(trace_path), spec.parameter_sha256
    )
    nsys = parse_nsys_sqlite(launch.sqlite_path, spec.parameter_sha256)
    artifact = build_profile_artifact(
        spec=spec,
        canary_artifact=canary_artifact,
        canary_artifact_sha256=canary_artifact_sha256,
        child=child,
        profile=profile,
        nsys=nsys,
        trace_path=trace_path,
        report_path=launch.report_path,
        sqlite_path=launch.sqlite_path,
        nsys_binary=nsys_binary.resolve(),
        nsys_version=version,
        nvtx_library=nvtx_library.resolve(),
        profile_counts=profile_counts,
        raw_child_path=raw_child_path,
        profile_count_evidence_path=profile_count_evidence_path,
        profile_launch=launch,
        version_probe_path=version_probe_path,
    )
    _write_exclusive_json(output_root / "profile-evidence.json", artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--canary-artifact", type=Path, required=True)
    parser.add_argument("--nsys-binary", type=Path, required=True)
    parser.add_argument("--nvtx-library", type=Path, required=True)
    parser.add_argument("--nsys-version", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile-count-evidence", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec_document = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(spec_document, dict):
        raise CanaryProfileRunnerError("canary spec must be an object")
    artifact = run_profile(
        spec=validate_spec(spec_document),
        canary_artifact_path=args.canary_artifact,
        nsys_binary=args.nsys_binary,
        nvtx_library=args.nvtx_library,
        expected_nsys_version=args.nsys_version,
        output_root=args.output_root,
        profile_count_evidence_path=args.profile_count_evidence,
    )
    return 0 if artifact["status"] in {"PRODUCED", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
