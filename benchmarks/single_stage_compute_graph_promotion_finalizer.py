"""Finalize C1/C2 promotion from independently validated evidence artifacts.

The timing canary never promotes.  This module is the sole owner of the
promotion decision and accepts paths only so every identity is re-derived from
canonical on-disk evidence at the final integrity boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from benchmarks.single_stage_compute_graph_c0_runner import (
    CommandResult,
    _load_canonical_json_object,
    _sha256_path,
    _write_exclusive_json,
)
from benchmarks.single_stage_compute_graph_canary_profile_runner import (
    PROFILE_SCHEMA_ID,
    _profile_child,
    build_profile_artifact,
    build_profile_launch,
    validate_profile_count_evidence,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CANARY_ARTIFACT_SCHEMA_ID,
    CANARY_SPEC_SCHEMA_ID,
    FINAL_ARTIFACT_FILENAME,
    CanaryRunnerError,
    CanarySpec,
    _load_restart,
    _spec_identity,
    build_artifact,
    validate_spec,
)
from benchmarks.single_stage_compute_graph_canary_spec_builder import (
    CanarySpecBuilderError,
    _validate_receipt_binding,
)
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    parse_nsys_sqlite,
)
from benchmarks.single_stage_compute_graph_native_reference import (
    NativeReferenceBinding,
)
from benchmarks.single_stage_compute_graph_native_trajectory_runner import (
    NativeTrajectoryLaunch,
    NativeTrajectoryRunnerError,
    validate_native_trajectory_launch,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    REQUIRED_WARM_SAMPLES,
    Phase0ReceiptError,
    load_phase0_receipt,
)
from benchmarks.single_stage_compute_graph_profile import (
    summarize_compute_graph_profile,
)
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    TrajectoryOracleError,
    TrajectoryOracleIdentity,
    bind_raw_trajectory_inputs,
    require_passing_variant_trajectory_oracle,
)
from benchmarks.single_stage_compute_graph_variant_trajectory_runner import (
    VariantTrajectoryLaunch,
    VariantTrajectoryRunnerError,
    validate_variant_trajectory_launch,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    load_trace_document,
)

PROMOTION_FINALIZER_SPEC_SCHEMA_ID: Final = (
    "single-stage-compute-graph-promotion-finalizer-spec-v1"
)
PROMOTION_ARTIFACT_SCHEMA_ID: Final = "single-stage-compute-graph-promotion-artifact-v1"


class PromotionFinalizerError(RuntimeError):
    """Promotion evidence is incomplete, inconsistent, or non-promoting."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PromotionFinalizerError(f"{context} must be an object")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PromotionFinalizerError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionFinalizerError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PromotionFinalizerError(f"{context} must be finite")
    return result


def _path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PromotionFinalizerError(f"{context} must be a non-empty path")
    return Path(value).resolve()


def _load_canary_spec(path: Path) -> tuple[Mapping[str, object], CanarySpec]:
    document = _load_canonical_json_object(path, "promotion canary spec")
    if document.get("schema_id") != CANARY_SPEC_SCHEMA_ID:
        raise PromotionFinalizerError("unsupported canary spec schema")
    try:
        return document, validate_spec(document)
    except CanaryRunnerError as error:
        raise PromotionFinalizerError(f"invalid canary spec: {error}") from error


def _validate_canary_artifact(
    path: Path, spec: CanarySpec
) -> tuple[Mapping[str, object], str]:
    expected_path = spec.output_root / FINAL_ARTIFACT_FILENAME
    if path.resolve() != expected_path.resolve():
        raise PromotionFinalizerError(
            "base canary must be the terminal artifact in the validated output root"
        )
    document = _load_canonical_json_object(path, "base canary artifact")
    digest = _sha256_path(path)
    sidecar = spec.output_root / "canary.sha256"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != digest + "\n":
        raise PromotionFinalizerError("base canary digest sidecar differs from bytes")
    if (
        document.get("schema_id") != CANARY_ARTIFACT_SCHEMA_ID
        or document.get("status") != "MEASURED_NONPROMOTING"
        or document.get("identity") != _spec_identity(spec)
    ):
        raise PromotionFinalizerError("base canary schema, status, or identity differs")
    observations, completed_rows = _load_restart(spec)
    if len(completed_rows) != REQUIRED_WARM_SAMPLES + 2:
        raise PromotionFinalizerError("base canary raw child route is incomplete")
    rebuilt = build_artifact(spec, observations)
    if rebuilt != document:
        raise PromotionFinalizerError(
            "base canary differs from raw-recomputed child evidence"
        )
    gates = _mapping(document.get("performance_gates"), "performance_gates")
    if not gates or any(value is not True for value in gates.values()):
        raise PromotionFinalizerError("base canary performance gates did not all pass")
    if document.get("performance_passed") is not True:
        raise PromotionFinalizerError("base canary is not performance-passing")
    blocker = _mapping(document.get("promotion_blocker"), "promotion_blocker")
    if blocker.get("code") != "PROMOTION_FINALIZER_REQUIRED":
        raise PromotionFinalizerError("base canary was not reserved for finalization")

    c0_reference = _mapping(document.get("c0_reference"), "c0_reference")
    if c0_reference != {
        "p50_ns": spec.c0_p50_ns,
        "p95_ns": spec.c0_p95_ns,
        "peak_process_tree_rss_bytes": spec.c0_peak_rss_bytes,
        "peak_gpu_memory_bytes": spec.c0_peak_gpu_memory_bytes,
    }:
        raise PromotionFinalizerError("base canary C0 resource identity differs")
    warm = _mapping(document.get("warm_measurement"), "warm_measurement")
    wall_ns = warm.get("wall_ns")
    roots = warm.get("process_tree_rss_roots")
    sample_counts = warm.get("process_tree_rss_sample_counts")
    if (
        not isinstance(wall_ns, list)
        or len(wall_ns) != REQUIRED_WARM_SAMPLES
        or not isinstance(roots, list)
        or len(roots) != REQUIRED_WARM_SAMPLES
        or not isinstance(sample_counts, list)
        or len(sample_counts) != REQUIRED_WARM_SAMPLES
    ):
        raise PromotionFinalizerError(
            "base canary warm/resource samples are incomplete"
        )
    checked_wall = [
        _integer(value, f"warm_measurement.wall_ns[{index}]", minimum=1)
        for index, value in enumerate(wall_ns)
    ]
    checked_counts = [
        _integer(value, f"warm_measurement.sample_counts[{index}]", minimum=1)
        for index, value in enumerate(sample_counts)
    ]
    del checked_counts
    for index, root in enumerate(roots):
        root_document = _mapping(root, f"warm_measurement.roots[{index}]")
        if set(root_document) != {"pid", "starttime_ticks"}:
            raise PromotionFinalizerError("base canary RSS root field set is invalid")
        _integer(root_document.get("pid"), f"warm root {index} pid", minimum=1)
        _integer(
            root_document.get("starttime_ticks"),
            f"warm root {index} starttime",
            minimum=1,
        )
    ordered = sorted(checked_wall)
    expected_p50 = float(statistics.median(checked_wall))
    expected_p95 = float(ordered[math.ceil(0.95 * len(ordered)) - 1])
    if (
        warm.get("sample_count") != REQUIRED_WARM_SAMPLES
        or _finite_number(warm.get("p50_wall_ns"), "warm p50") != expected_p50
        or _finite_number(warm.get("p95_wall_ns"), "warm p95") != expected_p95
        or expected_p50 > 0.8 * spec.c0_p50_ns
        or expected_p95 > 1.1 * spec.c0_p95_ns
        or _integer(warm.get("peak_process_tree_rss_bytes"), "warm peak RSS", minimum=1)
        > 1.1 * spec.c0_peak_rss_bytes
        or _integer(warm.get("peak_gpu_memory_bytes"), "warm peak GPU memory")
        > 1.1 * spec.c0_peak_gpu_memory_bytes
    ):
        raise PromotionFinalizerError("base canary timing/resource arithmetic differs")
    return document, digest


def _validate_c0_receipt(
    *,
    path: Path,
    canary_spec_document: Mapping[str, object],
    spec: CanarySpec,
) -> str:
    c0_spec_path = _path(canary_spec_document.get("c0_spec_path"), "c0_spec_path")
    c0_spec = _load_canonical_json_object(c0_spec_path, "C0 runner spec")
    output_root = _path(c0_spec.get("output_root"), "C0 output_root")
    if path.resolve() != (output_root / "phase0-receipt.json").resolve():
        raise PromotionFinalizerError(
            "C0 receipt is not the canonical output-root receipt"
        )
    try:
        receipt, _audit = load_phase0_receipt(path)
        qualification = _load_canonical_json_object(
            _path(canary_spec_document.get("qualification_path"), "qualification_path"),
            "C0 qualification",
        )
        provenance = _load_canonical_json_object(
            _path(
                canary_spec_document.get("runtime_provenance_path"),
                "runtime_provenance_path",
            ),
            "C0 runtime provenance",
        )
        gate = _load_canonical_json_object(
            output_root / "gate-checkpoint.json", "C0 gate checkpoint"
        )
        warm = _load_canonical_json_object(
            output_root / "warm-checkpoint.json", "C0 warm checkpoint"
        )
        _validate_receipt_binding(
            receipt,
            spec,
            qualification=qualification,
            runtime_provenance=provenance,
            gate_checkpoint=gate,
            warm_checkpoint=warm,
        )
    except (CanarySpecBuilderError, Phase0ReceiptError) as error:
        raise PromotionFinalizerError(f"invalid C0 receipt binding: {error}") from error
    return _sha256_path(path)


def _validate_hashed_path(binding: object, context: str) -> None:
    document = _mapping(binding, context)
    if set(document) != {"path", "sha256"}:
        raise PromotionFinalizerError(f"{context} has an invalid field set")
    path = _path(document.get("path"), f"{context}.path")
    expected = document.get("sha256")
    if not isinstance(expected, str) or _sha256_path(path) != expected:
        raise PromotionFinalizerError(f"{context} bytes differ from bound SHA")


def _native_trajectory_launch(
    *,
    receipt_path: Path,
    raw_path: Path,
    canary_spec_document: Mapping[str, object],
    spec: CanarySpec,
) -> NativeTrajectoryLaunch:
    from examples.jax.parity.input_bundle import read_input_bundle

    bundle, _arrays = read_input_bundle(spec.input_root)
    publication_path = _path(
        canary_spec_document.get("snapshot_publication_path"),
        "snapshot_publication_path",
    )
    publication = _load_canonical_json_object(
        publication_path, "native trajectory snapshot publication"
    )
    published_native = _mapping(
        publication.get("native_extension"), "published native extension"
    )
    native_relative_path = published_native.get("relative_path")
    native_sha256 = published_native.get("sha256")
    if not isinstance(native_relative_path, str) or not isinstance(native_sha256, str):
        raise PromotionFinalizerError("published native extension binding is invalid")
    native_path = (spec.snapshot_root / native_relative_path).resolve()
    if (
        not native_path.is_relative_to(spec.snapshot_root.resolve())
        or _sha256_path(native_path) != native_sha256
    ):
        raise PromotionFinalizerError(
            "snapshot native extension differs from published bytes"
        )
    runtime_contract = json.loads(spec.runtime_contract_json)
    if not isinstance(runtime_contract, dict):
        raise PromotionFinalizerError("native trajectory runtime contract is invalid")
    input_bundle_sha256 = _sha256_path(spec.input_root / "input_bundle.json")
    binding = NativeReferenceBinding(
        input_bundle_sha256=input_bundle_sha256,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        specimen_sha256=spec.specimen_sha256,
        source_sha256=spec.source_state_sha256,
        runtime_identity_sha256=spec.runtime_identity_sha256,
        interpreter_path=str(spec.interpreter_path),
        native_simsoptpp_path=str(native_path),
        native_simsoptpp_sha256=native_sha256,
        runtime_contract=runtime_contract,
    )
    return NativeTrajectoryLaunch(
        snapshot_root=spec.snapshot_root,
        snapshot_publication_path=publication_path,
        import_attestation_path=_path(
            canary_spec_document.get("import_attestation_path"),
            "import_attestation_path",
        ),
        snapshot_manifest_sha256=spec.snapshot_manifest_sha256,
        snapshot_publication_sha256=spec.snapshot_publication_sha256,
        import_attestation_sha256=spec.import_attestation_sha256,
        input_root=spec.input_root,
        candidate_path=spec.candidate_path,
        solver_graph_path=_path(
            canary_spec_document.get("variant_solver_graph_path"),
            "variant_solver_graph_path",
        ),
        output_path=raw_path,
        receipt_path=receipt_path,
        parameter_sha256=spec.parameter_sha256,
        binding=binding,
    )


def _validate_native_trajectory_receipt(
    *,
    receipt_path: Path,
    raw_path: Path,
    artifact_root: Path,
    canary_spec_document: Mapping[str, object],
    spec: CanarySpec,
) -> str:
    launch = _native_trajectory_launch(
        receipt_path=receipt_path,
        raw_path=raw_path,
        canary_spec_document=canary_spec_document,
        spec=spec,
    )
    try:
        validate_native_trajectory_launch(
            receipt_path, launch, artifact_root=artifact_root
        )
    except NativeTrajectoryRunnerError as error:
        raise PromotionFinalizerError(
            f"invalid native trajectory launch receipt: {error}"
        ) from error
    return _sha256_path(receipt_path)


def _validate_variant_trajectory_receipts(
    *,
    c0_receipt_path: Path | None,
    variant_receipt_path: Path,
    trajectory_reference_raw_path: Path,
    variant_raw_path: Path,
    profile_count_path: Path,
    artifact_root: Path,
    canary_spec_path: Path,
    base_canary_artifact_path: Path,
    spec: CanarySpec,
) -> dict[str, dict[str, str]]:
    receipts: dict[str, dict[str, str]] = {}
    if spec.variant == "C1":
        if c0_receipt_path is None:
            raise PromotionFinalizerError("C1 promotion requires a C0 launch receipt")
        c0_launch = VariantTrajectoryLaunch(
            spec=spec,
            spec_path=canary_spec_path,
            lane="C0",
            output_path=trajectory_reference_raw_path,
            receipt_path=c0_receipt_path,
        )
        try:
            validate_variant_trajectory_launch(
                c0_receipt_path,
                c0_launch,
                artifact_root=artifact_root,
            )
        except VariantTrajectoryRunnerError as error:
            raise PromotionFinalizerError(
                f"invalid C0 trajectory launch receipt: {error}"
            ) from error
        receipts["C0"] = {
            "path": str(c0_receipt_path),
            "sha256": _sha256_path(c0_receipt_path),
        }
    elif c0_receipt_path is not None:
        raise PromotionFinalizerError("C2 promotion must not inject unused C0 evidence")

    variant_launch = VariantTrajectoryLaunch(
        spec=spec,
        spec_path=canary_spec_path,
        lane=spec.variant,
        output_path=variant_raw_path,
        receipt_path=variant_receipt_path,
        profile_count_output_path=profile_count_path,
        canary_artifact_path=base_canary_artifact_path,
    )
    try:
        validate_variant_trajectory_launch(
            variant_receipt_path,
            variant_launch,
            artifact_root=artifact_root,
        )
    except VariantTrajectoryRunnerError as error:
        raise PromotionFinalizerError(
            f"invalid {spec.variant} trajectory launch receipt: {error}"
        ) from error
    receipts[spec.variant] = {
        "path": str(variant_receipt_path),
        "sha256": _sha256_path(variant_receipt_path),
    }
    return receipts


def _validate_profile_artifact(
    path: Path,
    *,
    spec: CanarySpec,
    canary: Mapping[str, object],
    canary_sha256: str,
) -> str:
    profile = _load_canonical_json_object(path, "candidate profile evidence")
    if (
        profile.get("schema_id") != PROFILE_SCHEMA_ID
        or profile.get("status") != "PRODUCED"
        or profile.get("promotion_timing") is not False
        or profile.get("identity")
        != {**_spec_identity(spec), "canary_artifact_sha256": canary_sha256}
        or profile.get("missing_required_source_hooks") != []
    ):
        raise PromotionFinalizerError(
            "candidate profile is blocked or identity-drifted"
        )
    numerical = _mapping(
        profile.get("numerical_revalidation"), "profile numerical revalidation"
    )
    gate = _mapping(canary.get("gate"), "base canary gate")
    for field in (
        "objective_dtype",
        "objective",
        "gradient_dtype",
        "gradient",
        "inner_newton_success",
        "adjoint_success",
        "residual_certificates",
    ):
        if numerical.get(field) != gate.get(field):
            raise PromotionFinalizerError("profile numerics differ from base canary")
    hlo = _mapping(profile.get("hlo_topology"), "profile HLO topology")
    for field in ("lowered_hlo_ir_sha256", "module_name_set_identity"):
        value = hlo.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise PromotionFinalizerError(f"profile HLO {field} is invalid")
    modules = hlo.get("module_names")
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(module, str) and module for module in modules)
    ):
        raise PromotionFinalizerError("profile HLO module names are incomplete")
    launches = _mapping(profile.get("launches"), "profile launches")
    for field in (
        "pjrt_execute_count",
        "jax_kernel_launch_count",
        "nsys_kernel_activity_count",
        "nsys_cuda_graph_launch_api_count",
        "nsys_uncaptured_kernel_activity_count",
    ):
        _integer(
            launches.get(field),
            f"profile launches.{field}",
            minimum=(1 if field in launches and field[:4] != "nsys" else 0),
        )
    if _integer(launches.get("nsys_kernel_activity_count"), "Nsight activity") < 1:
        raise PromotionFinalizerError("profile has no Nsight kernel activity")
    operations = _mapping(profile.get("required_operations"), "required operations")
    if set(operations) != {
        "residual",
        "jacobian_construction",
        "dense_materialization",
        "lu_factorization",
        "refinement",
        "linearized_tangent_traversals",
    }:
        raise PromotionFinalizerError("profile required-operation set is incomplete")
    for name in (
        "residual",
        "jacobian_construction",
        "dense_materialization",
        "lu_factorization",
        "refinement",
    ):
        operation = _mapping(operations[name], f"required operation {name}")
        _integer(operation.get("count"), f"{name}.count")
        _integer(
            operation.get("device_interval_union_ns"),
            f"{name}.device_interval_union_ns",
        )
    tangents = _mapping(
        operations["linearized_tangent_traversals"], "linearized tangent traversals"
    )
    for field in (
        "primal_traversal_count",
        "tangent_batch_count",
        "tangent_direction_count",
    ):
        _integer(tangents.get(field), f"linearized tangents.{field}", minimum=1)
    raw = _mapping(profile.get("raw"), "profile raw evidence")
    if set(raw) != {
        "version_probe",
        "raw_child",
        "profile_counts",
        "jax_trace",
        "nsys_report",
        "nsys_sqlite",
    }:
        raise PromotionFinalizerError("profile raw evidence set is incomplete")
    for role, binding in raw.items():
        _validate_hashed_path(binding, f"profile raw {role}")
    tool = _mapping(profile.get("tool"), "profile tool evidence")
    for prefix in ("nsys_binary", "nvtx_library"):
        _validate_hashed_path(
            {
                "path": tool.get(f"{prefix}_path"),
                "sha256": tool.get(f"{prefix}_sha256"),
            },
            f"profile tool {prefix}",
        )
    raw_child_path = _path(
        _mapping(raw["raw_child"], "raw child binding").get("path"),
        "raw child path",
    )
    if (
        path.name != "profile-evidence.json"
        or raw_child_path != (path.parent / "raw-child.json").resolve()
    ):
        raise PromotionFinalizerError(
            "profile is not the terminal output-root artifact"
        )
    raw_child = _load_canonical_json_object(raw_child_path, "profile raw child")
    if set(raw_child) != {
        "returncode",
        "timed_out",
        "elapsed_ns",
        "stdout",
        "stderr",
    }:
        raise PromotionFinalizerError("profile raw child field set is invalid")
    if (
        not isinstance(raw_child.get("stdout"), str)
        or not isinstance(raw_child.get("stderr"), str)
        or not isinstance(raw_child.get("timed_out"), bool)
    ):
        raise PromotionFinalizerError("profile raw child scalar types are invalid")
    result = CommandResult(
        returncode=_integer(raw_child.get("returncode"), "profile returncode"),
        stdout=cast(str, raw_child["stdout"]),
        stderr=cast(str, raw_child["stderr"]),
        elapsed_ns=_integer(raw_child.get("elapsed_ns"), "profile elapsed_ns"),
        timed_out=raw_child.get("timed_out") is True,
    )
    child = _profile_child(result)
    count_path = _path(
        _mapping(raw["profile_counts"], "profile count binding").get("path"),
        "profile count path",
    )
    profile_counts = validate_profile_count_evidence(
        _load_canonical_json_object(count_path, "profile count evidence"),
        spec=spec,
        canary_artifact_sha256=canary_sha256,
    )
    trace_path = _path(
        _mapping(raw["jax_trace"], "JAX trace binding").get("path"),
        "JAX trace path",
    )
    report_path = _path(
        _mapping(raw["nsys_report"], "Nsight report binding").get("path"),
        "Nsight report path",
    )
    sqlite_path = _path(
        _mapping(raw["nsys_sqlite"], "Nsight SQLite binding").get("path"),
        "Nsight SQLite path",
    )
    nsys_binary = _path(tool.get("nsys_binary_path"), "Nsight binary path")
    nvtx_library = _path(tool.get("nvtx_library_path"), "NVTX library path")
    nsys_version = tool.get("nsys_version")
    if not isinstance(nsys_version, str) or not nsys_version:
        raise PromotionFinalizerError("Nsight version is invalid")
    version_probe_path = _path(
        _mapping(raw["version_probe"], "version probe binding").get("path"),
        "version probe path",
    )
    version_probe = _load_canonical_json_object(
        version_probe_path, "profile version probe"
    )
    if set(version_probe) != {
        "returncode",
        "timed_out",
        "elapsed_ns",
        "stdout",
        "stderr",
    } or not all(
        isinstance(version_probe.get(field), str) for field in ("stdout", "stderr")
    ):
        raise PromotionFinalizerError("profile version-probe receipt is invalid")
    if (
        _integer(version_probe.get("returncode"), "version-probe returncode") != 0
        or version_probe.get("timed_out") is not False
        or _integer(version_probe.get("elapsed_ns"), "version-probe elapsed_ns") < 1
        or str(version_probe["stdout"]).strip() != nsys_version
    ):
        raise PromotionFinalizerError("profile version-probe completion differs")
    runtime_contract = json.loads(spec.runtime_contract_json)
    contract = _mapping(runtime_contract, "profile runtime contract")
    static_environment = _mapping(
        contract.get("static_environment"), "profile static environment"
    )
    profile_launch = build_profile_launch(
        spec,
        nsys_binary=nsys_binary,
        nvtx_library=nvtx_library,
        output_root=path.parent,
        base_environment={
            str(key): str(value) for key, value in static_environment.items()
        },
    )
    rebuilt = build_profile_artifact(
        spec=spec,
        canary_artifact=canary,
        canary_artifact_sha256=canary_sha256,
        child=child,
        profile=summarize_compute_graph_profile(
            load_trace_document(trace_path), spec.parameter_sha256
        ),
        nsys=parse_nsys_sqlite(sqlite_path, spec.parameter_sha256),
        trace_path=trace_path,
        report_path=report_path,
        sqlite_path=sqlite_path,
        nsys_binary=nsys_binary,
        nsys_version=nsys_version,
        nvtx_library=nvtx_library,
        profile_counts=profile_counts,
        raw_child_path=raw_child_path,
        profile_count_evidence_path=count_path,
        profile_launch=profile_launch,
        version_probe_path=version_probe_path,
    )
    if rebuilt != profile:
        raise PromotionFinalizerError(
            "candidate profile differs from raw-recomputed trace and Nsight evidence"
        )
    return _sha256_path(path)


def finalize_promotion(
    *,
    canary_spec_path: Path,
    base_canary_artifact_path: Path,
    profile_evidence_path: Path,
    trajectory_oracle_path: Path,
    trajectory_artifact_root: Path,
    one_step_reference_raw_path: Path,
    trajectory_reference_raw_path: Path,
    variant_raw_path: Path,
    native_trajectory_receipt_path: Path,
    c0_trajectory_receipt_path: Path | None,
    variant_trajectory_receipt_path: Path,
    c0_receipt_path: Path,
    destination: Path,
) -> dict[str, object]:
    """Recompute all promotion gates and exclusively publish one decision."""

    try:
        canary_spec_document, spec = _load_canary_spec(canary_spec_path.resolve())
        if spec.variant == "C2" and (
            one_step_reference_raw_path.resolve()
            != trajectory_reference_raw_path.resolve()
        ):
            raise PromotionFinalizerError(
                "C2 one-step and trajectory references must be the same native raw file"
            )
        canary, canary_sha256 = _validate_canary_artifact(
            base_canary_artifact_path.resolve(), spec
        )
        c0_receipt_sha256 = _validate_c0_receipt(
            path=c0_receipt_path.resolve(),
            canary_spec_document=canary_spec_document,
            spec=spec,
        )
        profile_sha256 = _validate_profile_artifact(
            profile_evidence_path.resolve(),
            spec=spec,
            canary=canary,
            canary_sha256=canary_sha256,
        )
        profile_document = _load_canonical_json_object(
            profile_evidence_path.resolve(), "candidate profile evidence"
        )
        profile_raw = _mapping(profile_document.get("raw"), "profile raw evidence")
        profile_count_path = _path(
            _mapping(profile_raw.get("profile_counts"), "profile count binding").get(
                "path"
            ),
            "profile count path",
        )
        native_trajectory_receipt_sha256 = _validate_native_trajectory_receipt(
            receipt_path=native_trajectory_receipt_path.resolve(),
            raw_path=one_step_reference_raw_path.resolve(),
            artifact_root=trajectory_artifact_root.resolve(),
            canary_spec_document=canary_spec_document,
            spec=spec,
        )
        variant_trajectory_receipts = _validate_variant_trajectory_receipts(
            c0_receipt_path=(
                None
                if c0_trajectory_receipt_path is None
                else c0_trajectory_receipt_path.resolve()
            ),
            variant_receipt_path=variant_trajectory_receipt_path.resolve(),
            trajectory_reference_raw_path=trajectory_reference_raw_path.resolve(),
            variant_raw_path=variant_raw_path.resolve(),
            profile_count_path=profile_count_path,
            artifact_root=trajectory_artifact_root.resolve(),
            canary_spec_path=canary_spec_path.resolve(),
            base_canary_artifact_path=base_canary_artifact_path.resolve(),
            spec=spec,
        )
        specimen = _mapping(
            _mapping(
                _load_canonical_json_object(
                    _path(canary_spec_document.get("c0_spec_path"), "c0_spec_path"),
                    "C0 runner spec",
                ).get("receipt_template"),
                "C0 receipt template",
            ).get("specimen"),
            "C0 specimen",
        )
        input_bundle_sha256 = specimen.get("input_bundle_sha256")
        if not isinstance(input_bundle_sha256, str):
            raise PromotionFinalizerError(
                "C0 specimen input-bundle identity is missing"
            )
        trajectory_identity = TrajectoryOracleIdentity(
            variant=spec.variant,
            parameter_sha256=spec.parameter_sha256,
            specimen_sha256=spec.specimen_sha256,
            input_bundle_sha256=input_bundle_sha256,
            solver_graph_sha256=spec.solver_graph_sha256,
            one_step_reference_source_sha256=spec.source_state_sha256,
            trajectory_reference_source_sha256=spec.source_state_sha256,
            variant_source_sha256=spec.source_state_sha256,
        )
        raw_bindings = bind_raw_trajectory_inputs(
            artifact_root=trajectory_artifact_root.resolve(),
            one_step_reference_raw_path=one_step_reference_raw_path.resolve(),
            trajectory_reference_raw_path=trajectory_reference_raw_path.resolve(),
            variant_raw_path=variant_raw_path.resolve(),
        )
        audit = require_passing_variant_trajectory_oracle(
            trajectory_oracle_path.resolve(),
            artifact_root=trajectory_artifact_root.resolve(),
            expected_identity=trajectory_identity,
            expected_raw_bindings=raw_bindings,
        )
    except (OSError, ValueError, RuntimeError, TrajectoryOracleError) as error:
        if isinstance(error, PromotionFinalizerError):
            raise
        raise PromotionFinalizerError(
            f"promotion evidence validation failed: {error}"
        ) from error

    artifact = {
        "schema_id": PROMOTION_ARTIFACT_SCHEMA_ID,
        "status": "MEASURED_PROMOTABLE",
        "identity": _spec_identity(spec),
        "evidence": {
            "canary_spec": {
                "path": str(canary_spec_path.resolve()),
                "sha256": _sha256_path(canary_spec_path.resolve()),
            },
            "base_canary": {
                "path": str(base_canary_artifact_path.resolve()),
                "sha256": canary_sha256,
            },
            "candidate_profile": {
                "path": str(profile_evidence_path.resolve()),
                "sha256": profile_sha256,
            },
            "trajectory_oracle": {
                "path": str(trajectory_oracle_path.resolve()),
                "sha256": _sha256_path(trajectory_oracle_path.resolve()),
                "raw_inputs": raw_bindings.to_json(),
            },
            "native_trajectory_launch_receipt": {
                "path": str(native_trajectory_receipt_path.resolve()),
                "sha256": native_trajectory_receipt_sha256,
            },
            "variant_trajectory_launch_receipts": variant_trajectory_receipts,
            "c0_receipt": {
                "path": str(c0_receipt_path.resolve()),
                "sha256": c0_receipt_sha256,
            },
        },
        "gates": {
            "base_performance_and_resources": True,
            "candidate_profile_hlo_launch_and_counts": True,
            "trajectory_one_step": audit.one_step_passed,
            "trajectory_short_replay": audit.short_replay_passed,
            "trajectory_terminal": audit.terminal_passed,
            "c0_receipt_and_resources": True,
        },
    }
    _write_exclusive_json(destination.resolve(), artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = _load_canonical_json_object(args.spec.resolve(), "finalizer spec")
    required = {
        "schema_id",
        "canary_spec_path",
        "base_canary_artifact_path",
        "profile_evidence_path",
        "trajectory_oracle_path",
        "trajectory_artifact_root",
        "one_step_reference_raw_path",
        "trajectory_reference_raw_path",
        "variant_raw_path",
        "native_trajectory_receipt_path",
        "c0_trajectory_receipt_path",
        "variant_trajectory_receipt_path",
        "c0_receipt_path",
        "destination",
    }
    if (
        set(document) != required
        or document.get("schema_id") != PROMOTION_FINALIZER_SPEC_SCHEMA_ID
    ):
        raise PromotionFinalizerError("promotion finalizer spec field set is invalid")
    finalize_promotion(
        canary_spec_path=_path(document.get("canary_spec_path"), "canary_spec_path"),
        base_canary_artifact_path=_path(
            document.get("base_canary_artifact_path"), "base_canary_artifact_path"
        ),
        profile_evidence_path=_path(
            document.get("profile_evidence_path"), "profile_evidence_path"
        ),
        trajectory_oracle_path=_path(
            document.get("trajectory_oracle_path"), "trajectory_oracle_path"
        ),
        trajectory_artifact_root=_path(
            document.get("trajectory_artifact_root"), "trajectory_artifact_root"
        ),
        one_step_reference_raw_path=_path(
            document.get("one_step_reference_raw_path"),
            "one_step_reference_raw_path",
        ),
        trajectory_reference_raw_path=_path(
            document.get("trajectory_reference_raw_path"),
            "trajectory_reference_raw_path",
        ),
        variant_raw_path=_path(document.get("variant_raw_path"), "variant_raw_path"),
        native_trajectory_receipt_path=_path(
            document.get("native_trajectory_receipt_path"),
            "native_trajectory_receipt_path",
        ),
        c0_trajectory_receipt_path=(
            None
            if document.get("c0_trajectory_receipt_path") is None
            else _path(
                document.get("c0_trajectory_receipt_path"),
                "c0_trajectory_receipt_path",
            )
        ),
        variant_trajectory_receipt_path=_path(
            document.get("variant_trajectory_receipt_path"),
            "variant_trajectory_receipt_path",
        ),
        c0_receipt_path=_path(document.get("c0_receipt_path"), "c0_receipt_path"),
        destination=_path(document.get("destination"), "destination"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
