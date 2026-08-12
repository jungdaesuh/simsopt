"""Fresh-process runner for the separate C1/C2 compute-graph canary-v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from benchmarks.single_stage_compute_graph_c0_runner import (
    PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
    PROCESS_TREE_RSS_SOURCE,
    CommandExecutor,
    CommandResult,
    _command_rss_document,
    _load_canonical_json_object,
    _load_json_object,
    _runtime_identity,
    _sha256_path,
    _subprocess_executor,
    _write_exclusive_json,
)
from benchmarks.single_stage_compute_graph_c0_runner import (
    _native_reference as _validated_native_reference,
)
from benchmarks.single_stage_compute_graph_canary_evaluator import (
    CanaryEvaluatorError,
    _validate_telemetry,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    SnapshotModuleLaunch,
    build_snapshot_module_launch,
    normalize_route_environment,
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_phase0_post_gate import (
    Phase0PostGateError,
    load_post_gate_context,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    PHASE0_GRADIENT_ATOL,
    PHASE0_GRADIENT_RTOL,
    PHASE0_OBJECTIVE_ATOL,
    PHASE0_OBJECTIVE_RTOL,
    REQUIRED_WARM_SAMPLES,
    Phase0ReceiptError,
    _validate_provenance,
    _validate_qualification,
    _validate_warm_measurement,
    canonical_sha256,
)
from benchmarks.single_stage_compute_graph_phase0_workflow import (
    Phase0WorkflowError,
    _attestation,
    _publication,
    _specimen,
)

CANARY_SPEC_SCHEMA_ID: Final = "single-stage-compute-graph-canary-runner-spec-v1"
CANARY_ARTIFACT_SCHEMA_ID: Final = "single-stage-compute-graph-canary-artifact-v1"
CANARY_STATE_SCHEMA_ID: Final = "single-stage-compute-graph-canary-state-v1"
CANARY_CHILD_SCHEMA_ID: Final = "single-stage-compute-graph-canary-child-v1"
EVALUATOR_MODULE: Final = "benchmarks.single_stage_compute_graph_canary_evaluator"
CHILD_TIMEOUT_SECONDS: Final = 900.0
STATE_PREFIX: Final = "state-"
FINAL_ARTIFACT_FILENAME: Final = "canary.json"
CanaryVariant = Literal["C1", "C2"]


class CanaryRunnerError(RuntimeError):
    """The canary runner input or child evidence is invalid."""


def _validated_variant_telemetry(
    variant: CanaryVariant, telemetry: Mapping[str, object], context: str
) -> None:
    try:
        _validate_telemetry(variant, telemetry)
    except CanaryEvaluatorError as error:
        raise CanaryRunnerError(f"{context} is invalid: {error}") from error


@dataclass(frozen=True, slots=True)
class CanarySpec:
    variant: CanaryVariant
    solver_graph_sha256: str
    source_state_sha256: str
    specimen_sha256: str
    candidate_file_sha256: str
    parameter_sha256: str
    device_identity_sha256: str
    gpu_uuid: str
    c0_gate_checkpoint_sha256: str
    c0_warm_checkpoint_sha256: str
    native_reference_sha256: str
    runtime_identity_sha256: str
    input_root: Path
    candidate_path: Path
    native_reference_path: Path
    snapshot_root: Path
    interpreter_path: Path
    cache_directory: Path
    output_root: Path
    snapshot_manifest_sha256: str = ""
    snapshot_publication_sha256: str = ""
    import_attestation_sha256: str = ""
    qualification_sha256: str = ""
    device_probe_sha256: str = ""
    runtime_provenance_sha256: str = ""
    c0_p50_ns: float = 0.0
    c0_p95_ns: float = 0.0
    c0_peak_rss_bytes: int = 0
    c0_peak_gpu_memory_bytes: int = 0
    runtime_contract_json: str = ""
    native_reference: Mapping[str, object] | None = None
    native_initial_reference: Mapping[str, object] | None = None


def _sha(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CanaryRunnerError(f"{context} must be a lowercase SHA-256")
    return value


def _path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CanaryRunnerError(f"{context} must be a non-empty path")
    return Path(value).resolve()


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CanaryRunnerError(f"{context} must be a non-empty string")
    return value


def _target_lane_document(
    receipt_template: Mapping[str, object], lane_id: str
) -> Mapping[str, object]:
    lanes = receipt_template.get("lanes")
    if not isinstance(lanes, list):
        raise CanaryRunnerError("C0 receipt template lanes are invalid")
    matches = [
        lane
        for lane in lanes
        if isinstance(lane, dict) and lane.get("lane_id") == lane_id
    ]
    if len(matches) != 1:
        raise CanaryRunnerError("C0 receipt template target lane is ambiguous")
    return matches[0]


def _document_contains_scalar(document: object, expected: str) -> bool:
    if document == expected:
        return True
    if isinstance(document, dict):
        return any(
            _document_contains_scalar(value, expected) for value in document.values()
        )
    if isinstance(document, list):
        return any(_document_contains_scalar(value, expected) for value in document)
    return False


def validate_spec(document: Mapping[str, object]) -> CanarySpec:
    """Derive every canary identity from validated, path-backed C0 evidence."""

    required = frozenset(
        {
            "schema_id",
            "variant",
            "c0_spec_path",
            "snapshot_publication_path",
            "import_attestation_path",
            "qualification_path",
            "device_probe_path",
            "runtime_provenance_path",
            "variant_solver_graph_path",
            "cache_directory",
            "output_root",
        }
    )
    if document.get("schema_id") != CANARY_SPEC_SCHEMA_ID:
        raise CanaryRunnerError("unsupported canary runner spec schema")
    if frozenset(document) != required:
        raise CanaryRunnerError("canary runner spec has unexpected or missing fields")
    variant_value = document.get("variant")
    if variant_value not in ("C1", "C2"):
        raise CanaryRunnerError("variant must be C1 or C2")
    variant: CanaryVariant = variant_value
    c0_spec_path = _path(document.get("c0_spec_path"), "c0_spec_path")
    publication_path = _path(
        document.get("snapshot_publication_path"), "snapshot_publication_path"
    )
    attestation_path = _path(
        document.get("import_attestation_path"), "import_attestation_path"
    )
    qualification_path = _path(document.get("qualification_path"), "qualification_path")
    probe_path = _path(document.get("device_probe_path"), "device_probe_path")
    provenance_path = _path(
        document.get("runtime_provenance_path"), "runtime_provenance_path"
    )
    graph_path = _path(
        document.get("variant_solver_graph_path"), "variant_solver_graph_path"
    )
    try:
        context = load_post_gate_context(c0_spec_path)
        _publication_document, entries, manifest_sha256 = _publication(
            context.snapshot_root, publication_path
        )
        del _publication_document
        _attestation(
            attestation_path,
            snapshot_root=context.snapshot_root,
            interpreter=context.interpreter,
            entries=entries,
            manifest_sha256=manifest_sha256,
        )
        specimen_document, specimen = _specimen(context.snapshot_root)
    except (Phase0PostGateError, Phase0WorkflowError) as error:
        raise CanaryRunnerError(f"invalid C0 publication evidence: {error}") from error
    c0_spec = context.c0_spec
    receipt_template = c0_spec.get("receipt_template")
    provenance = c0_spec.get("provenance")
    if not isinstance(receipt_template, dict) or not isinstance(provenance, dict):
        raise CanaryRunnerError("C0 spec lacks receipt/provenance documents")
    if receipt_template.get("specimen") != specimen or (
        receipt_template.get("specimen_sha256") != context.binding.specimen_sha256
    ):
        raise CanaryRunnerError("published specimen differs from C0 checkpoints")
    qualification = _load_canonical_json_object(
        qualification_path, "device qualification"
    )
    lane = _target_lane_document(receipt_template, context.binding.lane_id)
    if qualification != lane.get("qualification"):
        raise CanaryRunnerError("qualification path differs from C0 target lane")
    runtime_provenance = _load_canonical_json_object(
        provenance_path, "runtime provenance"
    )
    if runtime_provenance != provenance:
        raise CanaryRunnerError("runtime provenance path differs from C0 spec")
    try:
        if (
            _validate_qualification(
                qualification, context.binding.lane_id, "canary qualification"
            )
            != "qualified"
        ):
            raise CanaryRunnerError("C0 device qualification is blocked")
        source_sha, gpu_uuid, _cache = _validate_provenance(
            runtime_provenance,
            context.binding.lane_id,
            "canary runtime provenance",
        )
    except Phase0ReceiptError as error:
        raise CanaryRunnerError(f"invalid qualification/provenance: {error}") from error
    if (
        source_sha != context.binding.source_sha256
        or gpu_uuid != context.binding.gpu_uuid
        or _runtime_identity(runtime_provenance)
        != context.binding.runtime_identity_sha256
    ):
        raise CanaryRunnerError("C0 source/device/runtime checkpoint relation drifted")
    device_probe = _load_canonical_json_object(probe_path, "device probe")
    import_bindings = runtime_provenance.get("import_bindings")
    if not isinstance(import_bindings, dict) or not isinstance(
        import_bindings.get("simsoptpp"), dict
    ):
        raise CanaryRunnerError("runtime provenance lacks native binary binding")
    native_binding = import_bindings["simsoptpp"]
    native_sha = _sha(native_binding.get("sha256"), "native simsoptpp SHA")
    if not _document_contains_scalar(device_probe, gpu_uuid) or not (
        _document_contains_scalar(device_probe, native_sha)
    ):
        raise CanaryRunnerError("device probe does not bind GPU/native binary")
    graph_document = _load_canonical_json_object(graph_path, "variant solver graph")
    if graph_document.get("variant") != variant:
        raise CanaryRunnerError("variant solver graph selects a different variant")
    solver_graph_sha256 = canonical_sha256(graph_document)
    candidate_reference = specimen_document.get("candidate")
    if not isinstance(candidate_reference, dict):
        raise CanaryRunnerError("specimen candidate reference is missing")
    candidate_file_sha256 = _sha(
        candidate_reference.get("file_sha256"), "candidate file SHA"
    )
    if (
        _sha256_path(context.candidate_path) != candidate_file_sha256
        or specimen.get("parameter_sha256") != context.binding.candidate_sha256
    ):
        raise CanaryRunnerError("candidate bytes/parameter differ from specimen")
    from examples.jax.parity.input_bundle import read_input_bundle

    bundle, _arrays = read_input_bundle(context.input_root)
    native_reference_path = Path(
        _string(c0_spec.get("native_reference_path"), "native_reference_path")
    ).resolve()
    native_document, native_reference_sha256, reference = _validated_native_reference(
        native_reference_path,
        context.binding.candidate_sha256,
        expected_bindings={
            "input_bundle_sha256": context.input_bundle_sha256,
            "input_fingerprint": bundle.input_fingerprint,
            "configuration_fingerprint": bundle.configuration_fingerprint,
            "specimen_sha256": context.binding.specimen_sha256,
            "source_sha256": context.binding.source_sha256,
            "interpreter_path": str(context.interpreter),
            "native_simsoptpp_path": native_binding.get("path"),
            "native_simsoptpp_sha256": native_sha,
            "runtime_identity_sha256": context.binding.runtime_identity_sha256,
        },
    )
    initial_evaluation = native_document.get("initial_evaluation")
    if not isinstance(initial_evaluation, dict):
        raise CanaryRunnerError("native v3 reference lacks initial evaluation")
    native_initial_reference = {
        "native_objective": initial_evaluation.get("objective"),
        "native_gradient": initial_evaluation.get("gradient"),
        "parameter_sha256": initial_evaluation.get("parameter_sha256"),
    }
    gate_path = context.paths.c0_output_root / "gate-checkpoint.json"
    warm_path = context.paths.c0_output_root / "warm-checkpoint.json"
    warm = _load_canonical_json_object(warm_path, "C0 warm checkpoint")
    warm_measurement = warm.get("warm_measurement")
    try:
        c0_p50_ns = _validate_warm_measurement(warm_measurement, "C0 warm measurement")
    except Phase0ReceiptError as error:
        raise CanaryRunnerError(f"invalid C0 warm measurement: {error}") from error
    if not isinstance(warm_measurement, dict) or not isinstance(
        warm_measurement.get("samples"), list
    ):
        raise CanaryRunnerError("C0 warm samples are missing")
    c0_samples = warm_measurement["samples"]
    c0_rss_sampler = warm.get("process_tree_rss_sampler")
    if not isinstance(c0_rss_sampler, dict) or set(c0_rss_sampler) != {
        "source",
        "sample_interval_ns",
        "samples",
    }:
        raise CanaryRunnerError("C0 process-tree RSS sampler evidence is missing")
    c0_rss_rows = c0_rss_sampler.get("samples")
    if (
        c0_rss_sampler.get("source") != PROCESS_TREE_RSS_SOURCE
        or c0_rss_sampler.get("sample_interval_ns")
        != PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
        or not isinstance(c0_rss_rows, list)
        or len(c0_rss_rows) != len(c0_samples)
    ):
        raise CanaryRunnerError("C0 process-tree RSS sampler binding is invalid")
    for index, (row, sample) in enumerate(zip(c0_rss_rows, c0_samples, strict=True)):
        if (
            not isinstance(row, dict)
            or row.get("sample_index") != index
            or row.get("peak_bytes") != sample.get("peak_process_tree_rss_bytes")
            or isinstance(row.get("sample_count"), bool)
            or not isinstance(row.get("sample_count"), int)
            or int(row["sample_count"]) <= 0
            or isinstance(row.get("root_pid"), bool)
            or not isinstance(row.get("root_pid"), int)
            or int(row["root_pid"]) <= 0
            or isinstance(row.get("root_starttime_ticks"), bool)
            or not isinstance(row.get("root_starttime_ticks"), int)
            or int(row["root_starttime_ticks"]) <= 0
        ):
            raise CanaryRunnerError(f"C0 RSS sampler row {index} is invalid")
    c0_p95_ns = float(warm_measurement["p95_ns"])
    c0_peak_rss = max(
        int(sample["peak_process_tree_rss_bytes"]) for sample in c0_samples
    )
    c0_peak_gpu = max(
        int(sample["sampled_process_gpu_memory_peak_bytes"]) for sample in c0_samples
    )
    allocation = runtime_provenance.get("allocation")
    if not isinstance(allocation, dict):
        raise CanaryRunnerError("runtime provenance allocation is missing")
    runtime_contract_json = json.dumps(
        {
            "runtime": runtime_provenance["runtime"],
            "static_environment": runtime_provenance["environment"],
            "route_environment": {},
            "policies": runtime_provenance["policies"],
            "expected_runtime_identity_sha256": context.binding.runtime_identity_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    spec = CanarySpec(
        variant=variant,
        solver_graph_sha256=solver_graph_sha256,
        source_state_sha256=context.binding.source_sha256,
        specimen_sha256=context.binding.specimen_sha256,
        candidate_file_sha256=candidate_file_sha256,
        parameter_sha256=context.binding.candidate_sha256,
        device_identity_sha256=canonical_sha256(
            {
                "qualification": qualification,
                "probe": device_probe,
                "allocation": allocation,
            }
        ),
        gpu_uuid=gpu_uuid,
        c0_gate_checkpoint_sha256=_sha256_path(gate_path),
        c0_warm_checkpoint_sha256=_sha256_path(warm_path),
        native_reference_sha256=native_reference_sha256,
        runtime_identity_sha256=context.binding.runtime_identity_sha256,
        input_root=context.input_root,
        candidate_path=context.candidate_path,
        native_reference_path=native_reference_path,
        snapshot_root=context.snapshot_root,
        interpreter_path=context.interpreter,
        cache_directory=_path(document.get("cache_directory"), "cache_directory"),
        output_root=_path(document.get("output_root"), "output_root"),
        snapshot_manifest_sha256=manifest_sha256,
        snapshot_publication_sha256=_sha256_path(publication_path),
        import_attestation_sha256=_sha256_path(attestation_path),
        qualification_sha256=_sha256_path(qualification_path),
        device_probe_sha256=_sha256_path(probe_path),
        runtime_provenance_sha256=_sha256_path(provenance_path),
        c0_p50_ns=c0_p50_ns,
        c0_p95_ns=c0_p95_ns,
        c0_peak_rss_bytes=c0_peak_rss,
        c0_peak_gpu_memory_bytes=c0_peak_gpu,
        runtime_contract_json=runtime_contract_json,
        native_reference=reference,
        native_initial_reference=native_initial_reference,
    )
    for path, context in (
        (spec.input_root, "input_root"),
        (spec.snapshot_root, "snapshot_root"),
    ):
        if not path.is_dir():
            raise CanaryRunnerError(f"{context} must exist")
    for path, digest, context in (
        (spec.candidate_path, spec.candidate_file_sha256, "candidate"),
        (
            spec.native_reference_path,
            spec.native_reference_sha256,
            "native reference",
        ),
    ):
        if not path.is_file() or _sha256_path(path) != digest:
            raise CanaryRunnerError(f"{context} bytes differ from the bound SHA")
    if not spec.interpreter_path.is_file() or not os.access(
        spec.interpreter_path, os.X_OK
    ):
        raise CanaryRunnerError("interpreter_path must be executable")
    if spec.output_root == spec.cache_directory or (
        spec.output_root.is_relative_to(spec.cache_directory)
        or spec.cache_directory.is_relative_to(spec.output_root)
    ):
        raise CanaryRunnerError("variant artifact and cache roots must be disjoint")
    return spec


def _spec_identity(spec: CanarySpec) -> dict[str, object]:
    return {
        "variant": spec.variant,
        "solver_graph_sha256": spec.solver_graph_sha256,
        "source_state_sha256": spec.source_state_sha256,
        "specimen_sha256": spec.specimen_sha256,
        "candidate_file_sha256": spec.candidate_file_sha256,
        "parameter_sha256": spec.parameter_sha256,
        "device_identity_sha256": spec.device_identity_sha256,
        "gpu_uuid": spec.gpu_uuid,
        "c0_gate_checkpoint_sha256": spec.c0_gate_checkpoint_sha256,
        "c0_warm_checkpoint_sha256": spec.c0_warm_checkpoint_sha256,
        "native_reference_sha256": spec.native_reference_sha256,
        "runtime_identity_sha256": spec.runtime_identity_sha256,
        "snapshot_manifest_sha256": spec.snapshot_manifest_sha256,
        "snapshot_publication_sha256": spec.snapshot_publication_sha256,
        "import_attestation_sha256": spec.import_attestation_sha256,
        "qualification_sha256": spec.qualification_sha256,
        "device_probe_sha256": spec.device_probe_sha256,
        "runtime_provenance_sha256": spec.runtime_provenance_sha256,
        "cache_directory": str(spec.cache_directory),
    }


def child_launches(
    spec: CanarySpec, base_environment: Mapping[str, str]
) -> tuple[SnapshotModuleLaunch, ...]:
    """Build initial/changed-state gates plus ten isolated warm commands."""

    environment = normalize_static_timing_environment(base_environment)
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(spec.cache_directory),
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
        }
    )
    runtime_contract = json.loads(spec.runtime_contract_json)
    if not isinstance(runtime_contract, dict):
        raise CanaryRunnerError("runtime contract must be an object")
    runtime_contract["route_environment"] = normalize_route_environment(environment)
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"] = json.dumps(
        runtime_contract, sort_keys=True, separators=(",", ":")
    )
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY"] = (
        spec.runtime_identity_sha256
    )
    launches: list[SnapshotModuleLaunch] = []
    schedule = (("initial_gate", None), ("gate", None)) + tuple(
        ("warm", index) for index in range(REQUIRED_WARM_SAMPLES)
    )
    for mode, index in schedule:
        args = [
            "--variant",
            spec.variant,
            "--mode",
            mode,
            "--input-root",
            str(spec.input_root),
            "--candidate",
            str(spec.candidate_path),
            "--parameter-sha256",
            spec.parameter_sha256,
            "--gpu-uuid",
            spec.gpu_uuid,
            "--snapshot-root",
            str(spec.snapshot_root),
        ]
        if index is not None:
            args.extend(("--sample-index", str(index)))
        launch = build_snapshot_module_launch(
            spec.interpreter_path,
            spec.snapshot_root,
            EVALUATOR_MODULE,
            tuple(args),
            environment,
        )
        launches.append(launch)
    return tuple(launches)


def _native_reference(path: Path) -> Mapping[str, object]:
    reference = _load_json_object(path, "native reference")
    objective = reference.get("objective", reference.get("native_objective"))
    gradient = reference.get("gradient", reference.get("native_gradient"))
    if not isinstance(objective, (int, float)) or not isinstance(gradient, list):
        raise CanaryRunnerError("native reference lacks objective/gradient")
    return {"objective": float(objective), "gradient": gradient}


def _gate_parity(
    gate: Mapping[str, object],
    native_reference: Mapping[str, object],
    variant: CanaryVariant,
    *,
    expected_mode: str = "gate",
    expected_parameter_sha256: str | None = None,
) -> None:
    if (
        gate.get("status") != "PASS"
        or gate.get("mode") != expected_mode
        or gate.get("variant") != variant
    ):
        raise CanaryRunnerError("canary gate did not pass")
    if (
        expected_parameter_sha256 is not None
        and gate.get("parameter_sha256") != expected_parameter_sha256
    ):
        raise CanaryRunnerError("canary gate parameter identity is invalid")
    objective = gate.get("objective")
    gradient = gate.get("gradient")
    native_objective = native_reference.get(
        "objective", native_reference.get("native_objective")
    )
    native_gradient = native_reference.get(
        "gradient", native_reference.get("native_gradient")
    )
    if not isinstance(objective, (int, float)) or not isinstance(gradient, list):
        raise CanaryRunnerError("gate objective/gradient is malformed")
    if isinstance(native_objective, bool) or not isinstance(
        native_objective, (int, float)
    ):
        raise CanaryRunnerError("native reference objective is malformed")
    if (
        not math.isfinite(float(objective))
        or gate.get("objective_dtype") != "float64"
        or gate.get("gradient_dtype") != "float64"
        or gate.get("inner_newton_success") is not True
        or gate.get("adjoint_success") is not True
    ):
        raise CanaryRunnerError("gate finite/FP64/solver-success contract failed")
    certificates = gate.get("residual_certificates")
    if (
        not isinstance(certificates, dict)
        or not certificates
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in certificates.values()
        )
    ):
        raise CanaryRunnerError("gate residual certificates are invalid")
    if (
        not isinstance(native_gradient, list)
        or len(gradient) != 461
        or len(native_gradient) != 461
    ):
        raise CanaryRunnerError("gate and native gradients must have 461 entries")
    if not np_isclose(
        float(objective),
        float(native_objective),
        atol=PHASE0_OBJECTIVE_ATOL,
        rtol=PHASE0_OBJECTIVE_RTOL,
    ):
        raise CanaryRunnerError("canary objective fails native parity")
    if any(
        not np_isclose(
            float(value),
            float(reference),
            atol=PHASE0_GRADIENT_ATOL,
            rtol=PHASE0_GRADIENT_RTOL,
        )
        for value, reference in zip(gradient, native_gradient, strict=True)
    ):
        raise CanaryRunnerError("canary gradient fails native parity")
    telemetry = gate.get("telemetry")
    if not isinstance(telemetry, dict):
        raise CanaryRunnerError("gate telemetry is missing")
    _validated_variant_telemetry(variant, telemetry, "gate telemetry")
    _validate_memory_observation(gate, "gate")


def _validate_memory_observation(
    observation: Mapping[str, object], context: str
) -> None:
    rss = observation.get("peak_process_tree_rss_bytes")
    if isinstance(rss, bool) or not isinstance(rss, int) or rss <= 0:
        raise CanaryRunnerError(f"{context} peak_process_tree_rss_bytes is invalid")
    sample_count = observation.get("process_tree_rss_sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or observation.get("process_tree_rss_sample_interval_ns")
        != PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
        or observation.get("process_tree_rss_source") != PROCESS_TREE_RSS_SOURCE
        or not isinstance(observation.get("process_tree_rss_root_pid"), int)
        or int(observation["process_tree_rss_root_pid"]) <= 0
        or not isinstance(observation.get("process_tree_rss_root_starttime_ticks"), int)
        or int(observation["process_tree_rss_root_starttime_ticks"]) <= 0
    ):
        raise CanaryRunnerError(f"{context} process-tree RSS sampler is invalid")
    gpu = observation.get("gpu_memory")
    if not isinstance(gpu, dict):
        raise CanaryRunnerError(f"{context} GPU-memory evidence is missing")
    expected = {
        "provider_pid",
        "gpu_uuid",
        "sample_count",
        "sample_interval_ns",
        "peak_bytes",
        "source",
    }
    if set(gpu) != expected or gpu.get("source") != "nvidia-smi_direct_pid_gpu_uuid":
        raise CanaryRunnerError(f"{context} GPU-memory evidence is invalid")
    for key in ("provider_pid", "sample_count", "sample_interval_ns"):
        value = gpu.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CanaryRunnerError(f"{context} GPU-memory {key} is invalid")
    peak = gpu.get("peak_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise CanaryRunnerError(f"{context} GPU-memory peak is invalid")


def np_isclose(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return abs(left - right) <= atol + rtol * abs(right)


def build_artifact(
    spec: CanarySpec,
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build PASS or fail-closed BLOCKED canary-v1 evidence."""

    identity = _spec_identity(spec)
    blocked = next(
        (item for item in observations if item.get("status") == "BLOCKED"), None
    )
    if blocked is not None:
        return {
            "schema_id": CANARY_ARTIFACT_SCHEMA_ID,
            "status": "BLOCKED",
            "identity": identity,
            "blocker": blocked.get("blocker"),
            "warm_measurement": None,
        }
    if len(observations) != REQUIRED_WARM_SAMPLES + 2:
        return {
            "schema_id": CANARY_ARTIFACT_SCHEMA_ID,
            "status": "BLOCKED",
            "identity": identity,
            "blocker": {
                "code": "INCOMPLETE_WARM_ROUTE",
                "reason": (
                    "one initial gate, one changed-state gate, and ten warm "
                    "observations are required"
                ),
            },
            "warm_measurement": None,
        }
    initial_gate, gate, *warm = observations
    initial_reference = spec.native_initial_reference
    if initial_reference is None:
        raise CanaryRunnerError("validated native initial reference is unavailable")
    initial_parameter_sha256 = initial_reference.get("parameter_sha256")
    if not isinstance(initial_parameter_sha256, str):
        raise CanaryRunnerError("native initial parameter identity is invalid")
    _gate_parity(
        initial_gate,
        initial_reference,
        spec.variant,
        expected_mode="initial_gate",
        expected_parameter_sha256=initial_parameter_sha256,
    )
    native_reference = (
        spec.native_reference
        if spec.native_reference is not None
        else _native_reference(spec.native_reference_path)
    )
    _gate_parity(gate, native_reference, spec.variant)
    wall_ns = []
    rss_bytes = []
    gpu_bytes = []
    for index, observation in enumerate(warm):
        if (
            observation.get("schema_id") != CANARY_CHILD_SCHEMA_ID
            or observation.get("status") != "PASS"
            or observation.get("variant") != spec.variant
            or observation.get("mode") != "warm"
            or observation.get("sample_index") != index
        ):
            raise CanaryRunnerError(f"warm observation {index} is invalid")
        value = observation.get("wall_ns")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CanaryRunnerError(f"warm observation {index} wall_ns is invalid")
        wall_ns.append(value)
        telemetry = observation.get("telemetry")
        if not isinstance(telemetry, dict):
            raise CanaryRunnerError(f"warm observation {index} telemetry is missing")
        _validated_variant_telemetry(
            spec.variant, telemetry, f"warm observation {index} telemetry"
        )
        _validate_memory_observation(observation, f"warm observation {index}")
        rss_bytes.append(int(observation["peak_process_tree_rss_bytes"]))
        gpu_memory = cast(dict[str, object], observation["gpu_memory"])
        gpu_bytes.append(int(gpu_memory["peak_bytes"]))
    p50_ns = float(statistics.median(wall_ns))
    ordered_wall = sorted(wall_ns)
    p95_ns = float(ordered_wall[math.ceil(0.95 * len(ordered_wall)) - 1])
    candidate_peak_rss = max(rss_bytes)
    candidate_peak_gpu = max(gpu_bytes)
    performance_gates = {
        "p50_at_least_20_percent_faster": p50_ns <= 0.8 * spec.c0_p50_ns,
        "p95_at_most_10_percent_regression": p95_ns <= 1.1 * spec.c0_p95_ns,
        "process_tree_rss_evidence_available": True,
        "peak_process_tree_rss_at_most_10_percent_regression": (
            candidate_peak_rss <= 1.1 * spec.c0_peak_rss_bytes
        ),
        "peak_gpu_memory_at_most_10_percent_regression": (
            candidate_peak_gpu <= 1.1 * spec.c0_peak_gpu_memory_bytes
        ),
    }
    performance_passed = all(performance_gates.values())
    return {
        "schema_id": CANARY_ARTIFACT_SCHEMA_ID,
        "status": "MEASURED_NONPROMOTING",
        "identity": identity,
        "initial_point_gate": dict(initial_gate),
        "gate": dict(gate),
        "c0_reference": {
            "p50_ns": spec.c0_p50_ns,
            "p95_ns": spec.c0_p95_ns,
            "peak_process_tree_rss_bytes": spec.c0_peak_rss_bytes,
            "peak_gpu_memory_bytes": spec.c0_peak_gpu_memory_bytes,
        },
        "warm_measurement": {
            "sample_count": len(wall_ns),
            "wall_ns": wall_ns,
            "p50_wall_ns": p50_ns,
            "p95_wall_ns": p95_ns,
            "peak_process_tree_rss_bytes": candidate_peak_rss,
            "process_tree_rss_source": PROCESS_TREE_RSS_SOURCE,
            "process_tree_rss_sample_interval_ns": (
                PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
            ),
            "process_tree_rss_sample_counts": [
                int(observation["process_tree_rss_sample_count"])
                for observation in warm
            ],
            "process_tree_rss_roots": [
                {
                    "pid": int(observation["process_tree_rss_root_pid"]),
                    "starttime_ticks": int(
                        observation["process_tree_rss_root_starttime_ticks"]
                    ),
                }
                for observation in warm
            ],
            "peak_gpu_memory_bytes": candidate_peak_gpu,
        },
        "performance_gates": performance_gates,
        "performance_passed": performance_passed,
        "promotion_blocker": {
            "code": (
                "PROMOTION_FINALIZER_REQUIRED"
                if performance_passed
                else "PERFORMANCE_GATE_FAILED"
            ),
            "reason": (
                "only the evidence-recomputing promotion finalizer may promote"
                if performance_passed
                else "one or more candidate-vs-C0 performance gates failed"
            ),
        },
    }


def _raw_child_document(result: CommandResult) -> dict[str, object]:
    return {
        "returncode": result.returncode,
        "elapsed_ns": result.elapsed_ns,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "process_tree_rss": (
            None
            if result.process_tree_rss is None
            else {
                "peak_bytes": result.process_tree_rss.peak_bytes,
                "sample_count": result.process_tree_rss.sample_count,
                "sample_interval_ns": result.process_tree_rss.sample_interval_ns,
                "source": result.process_tree_rss.source,
                "root_pid": result.process_tree_rss.root_pid,
                "root_starttime_ticks": (result.process_tree_rss.root_starttime_ticks),
            }
        ),
    }


def _parsed_child_observation(
    result: CommandResult, context: str
) -> Mapping[str, object]:
    if result.timed_out:
        raise CanaryRunnerError(f"{context} timed out after 900 seconds")
    try:
        observation = json.loads(
            result.stdout,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CanaryRunnerError(f"{context} emitted nonfinite {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise CanaryRunnerError(f"{context} emitted invalid JSON") from error
    if not isinstance(observation, dict):
        raise CanaryRunnerError(f"{context} observation must be an object")
    if observation.get("schema_id") != CANARY_CHILD_SCHEMA_ID:
        raise CanaryRunnerError(f"{context} observation schema is invalid")
    if result.returncode != 0 and observation.get("status") != "BLOCKED":
        raise CanaryRunnerError(
            f"{context} failed with return code {result.returncode}"
        )
    forbidden = frozenset(
        {
            "peak_process_tree_rss_bytes",
            "process_tree_rss_sample_count",
            "process_tree_rss_sample_interval_ns",
            "process_tree_rss_source",
            "process_tree_rss_root_pid",
            "process_tree_rss_root_starttime_ticks",
        }
    )
    if forbidden.intersection(observation):
        raise CanaryRunnerError(
            f"{context} must not self-report parent-owned process-tree RSS"
        )
    try:
        rss_document = _command_rss_document(result, context)
    except RuntimeError as error:
        raise CanaryRunnerError(str(error)) from error
    return {**observation, **rss_document}


def _load_restart(
    spec: CanarySpec,
) -> tuple[list[Mapping[str, object]], list[dict[str, object]]]:
    states = sorted(spec.output_root.glob(f"{STATE_PREFIX}*.json"))
    if not states:
        raise CanaryRunnerError("existing canary root has no resumable state")
    state = _load_canonical_json_object(states[-1], "canary restart state")
    if (
        state.get("schema_id") != CANARY_STATE_SCHEMA_ID
        or state.get("state") != "RUNNING"
        or state.get("identity_sha256") != canonical_sha256(_spec_identity(spec))
    ):
        raise CanaryRunnerError("canary restart state identity is stale")
    completed = state.get("completed")
    if not isinstance(completed, list):
        raise CanaryRunnerError("canary restart completed rows are invalid")
    observations: list[Mapping[str, object]] = []
    rows: list[dict[str, object]] = []
    for expected_index, raw_row in enumerate(completed):
        if (
            not isinstance(raw_row, dict)
            or raw_row.get("sequence_index") != expected_index
        ):
            raise CanaryRunnerError("canary restart rows are not contiguous")
        raw_path = spec.output_root / _string(raw_row.get("raw_path"), "raw_path")
        observation_path = spec.output_root / _string(
            raw_row.get("observation_path"), "observation_path"
        )
        if (
            not raw_path.is_file()
            or not observation_path.is_file()
            or _sha256_path(raw_path) != raw_row.get("raw_sha256")
            or _sha256_path(observation_path) != raw_row.get("observation_sha256")
        ):
            raise CanaryRunnerError("canary restart child bytes drifted")
        observations.append(
            _load_canonical_json_object(observation_path, "restart observation")
        )
        rows.append(dict(raw_row))
    return observations, rows


def _write_terminal_artifact(spec: CanarySpec, artifact: Mapping[str, object]) -> None:
    digest = _write_exclusive_json(spec.output_root / FINAL_ARTIFACT_FILENAME, artifact)
    with (spec.output_root / "canary.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_canary(
    spec: CanarySpec,
    *,
    executor: CommandExecutor = _subprocess_executor,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Execute or resume one timeout-bounded, byte-checkpointed canary."""

    base_environment = os.environ if environment is None else environment
    if spec.output_root.exists():
        if (spec.output_root / FINAL_ARTIFACT_FILENAME).exists():
            raise CanaryRunnerError("canary root already has a terminal artifact")
        observations, completed_rows = _load_restart(spec)
    else:
        spec.output_root.mkdir(parents=True, exist_ok=False)
        spec.cache_directory.mkdir(parents=True, exist_ok=False)
        (spec.output_root / "children").mkdir()
        observations = []
        completed_rows = []
    schedule = (("initial_gate", None), ("gate", None)) + tuple(
        ("warm", index) for index in range(REQUIRED_WARM_SAMPLES)
    )
    try:
        for sequence_index in range(len(completed_rows), len(schedule)):
            mode, sample_index = schedule[sequence_index]
            # Rebuild immediately before execution: this revalidates every
            # manifested snapshot byte and the exact evaluator module path.
            launch = child_launches(spec, base_environment)[sequence_index]
            child_name = mode if sample_index is None else f"warm-{sample_index:02d}"
            child_root = spec.output_root / "children" / child_name
            child_root.mkdir(exist_ok=False)
            result = executor(
                launch.argv,
                launch.environment,
                launch.cwd,
                CHILD_TIMEOUT_SECONDS,
            )
            raw_path = child_root / "raw.json"
            raw_sha = _write_exclusive_json(raw_path, _raw_child_document(result))
            observation = _parsed_child_observation(result, f"{child_name} child")
            observation_path = child_root / "observation.json"
            observation_sha = _write_exclusive_json(observation_path, observation)
            observations.append(observation)
            completed_rows.append(
                {
                    "sequence_index": sequence_index,
                    "mode": mode,
                    "sample_index": sample_index,
                    "raw_path": raw_path.relative_to(spec.output_root).as_posix(),
                    "raw_sha256": raw_sha,
                    "observation_path": observation_path.relative_to(
                        spec.output_root
                    ).as_posix(),
                    "observation_sha256": observation_sha,
                }
            )
            _write_exclusive_json(
                spec.output_root / f"{STATE_PREFIX}{sequence_index:02d}.json",
                {
                    "schema_id": CANARY_STATE_SCHEMA_ID,
                    "state": "RUNNING",
                    "identity_sha256": canonical_sha256(_spec_identity(spec)),
                    "completed": completed_rows,
                },
            )
            if observation.get("status") == "BLOCKED":
                break
        artifact = build_artifact(spec, observations)
    except (OSError, ValueError, RuntimeError) as error:
        artifact = build_artifact(
            spec,
            (
                *observations,
                {
                    "status": "BLOCKED",
                    "blocker": {
                        "code": "CANARY_CHILD_OR_RESTART_BLOCKED",
                        "reason": str(error),
                    },
                },
            ),
        )
        _write_exclusive_json(
            spec.output_root / "state-terminal.json",
            {
                "schema_id": CANARY_STATE_SCHEMA_ID,
                "state": "BLOCKED",
                "identity_sha256": canonical_sha256(_spec_identity(spec)),
                "completed": completed_rows,
                "blocker": artifact["blocker"],
            },
        )
    _write_terminal_artifact(spec, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = _load_json_object(args.spec, "canary runner spec")
    artifact = run_canary(validate_spec(document))
    return 0 if artifact["status"] == "MEASURED_NONPROMOTING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
