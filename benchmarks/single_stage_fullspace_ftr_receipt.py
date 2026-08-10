"""Immutable CFS-FTR1 ten-step Gate 2 receipt and semantic validator."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from simsopt_jax.solve.fullspace import FullSpaceRoute

from benchmarks import single_stage_fullspace_snapshot as _snapshot_contract
from benchmarks.single_stage_fullspace_receipt import (
    DeviceLane,
    RunPhase,
    RunRequest,
    artifact_ref_from_payload,
    contract_sha256_v3,
    expect_exact_keys,
    expect_integer,
    expect_mapping,
    expect_string,
    load_canonical_json_bytes,
    run_request_v3_from_payload,
)
from benchmarks.single_stage_fullspace_snapshot import ArtifactRef, JsonValue

FTR_RESULT_SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-ftr1-result-v1"
FTR_GATE_RECEIPT_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-cfs-ftr1-canary-10-gate-receipt-v1"
)
FTR_MEMORY_SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-ftr1-gpu-memory-v1"
FTR_PLAN_SHA256: Final = (
    "5086ecacd3147649edde5b65e63411cd6a384f8943d5d67b0906b09174dc1351"
)
FTR_BUDGET_SHA256: Final = (
    "84920e2c9c9a10c8a50f3927a192ef38516b0b6c1a0f545946ecb54b50d740ac"
)
FTR_MAXIMUM_MEMORY_FRACTION: Final = 0.8
FTR_MAXIMUM_JOINT_EVALUATIONS: Final = 1200
FTR_PROJECTED_SOLVE_MAX_SECONDS: Final = 287.30421751597896
FTR_LINEAR_RESIDUAL_MAXIMUM: Final = 1.0e-10
FTR_FORWARD_ERROR_MAXIMUM: Final = 1.0e-7
FTR_PRIMARY_DEVICE_UUID: Final = "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
_FTR_MEMORY_KEYS: Final = frozenset(
    (
        "schema_version",
        "monitor_scope",
        "parent_pid",
        "child_pid",
        "child_start_time_ticks",
        "child_argv_sha256",
        "device_uuid",
        "sample_count",
        "peak_memory_bytes",
        "peak_memory_fraction",
    )
)

_EXPECTED_REQUEST = RunRequest(
    RunPhase.CANARY,
    FullSpaceRoute.CFS_FTR1,
    DeviceLane.RTX5090,
    10,
    None,
)
_COMMON_RECEIPT_KEYS = frozenset(
    (
        "schema_version",
        "contract_sha256",
        "plan_sha256",
        "budget_sha256",
        "request",
        "source_identity",
        "runtime_evidence",
        "bootstrap_artifact",
        "raw_result",
        "gpu_memory",
        "gate_status",
        "failure_reasons",
        "canary_gate",
    )
)


@dataclass(frozen=True, slots=True)
class FtrGateResult:
    """Validated immutable Gate 2 receipt and its recomputed verdict."""

    artifact: ArtifactRef
    passed: bool
    failure_reasons: tuple[str, ...]


def _finite_number(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def derive_ftr_gate_detail(
    raw: dict[str, JsonValue], memory: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Recompute every frozen ten-step Gate 2 decision from raw evidence."""

    optimizer = expect_mapping(raw.get("optimizer_result"), context="FTR optimizer")
    endpoint = expect_mapping(raw.get("endpoint"), context="FTR endpoint")
    kkt = expect_mapping(raw.get("independent_kkt"), context="FTR endpoint KKT")
    transfers = expect_mapping(raw.get("transfer_audit"), context="FTR transfers")
    timing = expect_mapping(raw.get("timing"), context="FTR timing")
    reasons: list[str] = []

    if not (
        optimizer.get("all_finite") is True
        and optimizer.get("all_accepted_states_finite") is True
        and optimizer.get("solver_result_consistent") is True
        and optimizer.get("solve_certificates_valid") is True
        and endpoint.get("all_finite") is True
    ):
        reasons.append("NONFINITE_OR_INCONSISTENT_STATE")
    if optimizer.get("fatal") is not False or optimizer.get("status") not in (
        "CONVERGED",
        "ITERATION_LIMIT",
    ):
        reasons.append("FATAL_STATUS")
    if optimizer.get("iterations") != 10:
        reasons.append("ITERATION_COUNT")
    if (
        optimizer.get("coordinate_count") != 716
        or optimizer.get("equality_count") != 255
        or optimizer.get("dtype") != "float64"
    ):
        reasons.append("LEDGER_OR_DTYPE")
    history = optimizer.get("history")
    history_valid = isinstance(history, dict) and history.get("attempted_length") == 10
    if not history_valid:
        reasons.append("HISTORY_LENGTH")
    else:
        assert isinstance(history, dict)
        history_fields = tuple(
            key for key in history if key not in ("attempted_length", "accepted_length")
        )
        history_valid = all(
            isinstance(history[field], list) and len(history[field]) == 10
            for field in history_fields
        )
        accepted = history.get("accepted")
        filter_accepted = history.get("filter_accepted")
        selected_radius = history.get("selected_radius_index")
        history_valid = history_valid and (
            isinstance(accepted, list)
            and all(type(value) is bool for value in accepted)
            and isinstance(filter_accepted, list)
            and all(type(value) is bool for value in filter_accepted)
            and all(
                (not was_accepted) or passed_filter
                for was_accepted, passed_filter in zip(
                    accepted, filter_accepted, strict=True
                )
            )
            and history.get("accepted_length") == sum(accepted)
            and isinstance(selected_radius, list)
            and selected_radius == [0] * 10
        )
        if not history_valid:
            reasons.append("MODEL_FILTER_DECISIONS")
    evaluations = optimizer.get("joint_evaluations")
    if (
        isinstance(evaluations, bool)
        or not isinstance(evaluations, int)
        or evaluations < 1
        or evaluations > FTR_MAXIMUM_JOINT_EVALUATIONS
    ):
        reasons.append("EVALUATION_BUDGET")

    initial_objective = _finite_number(optimizer.get("initial_physical_objective"))
    final_objective = _finite_number(endpoint.get("physical_objective"))
    initial_feasibility = _finite_number(
        optimizer.get("initial_scaled_constraint_infinity_norm")
    )
    final_feasibility = _finite_number(endpoint.get("scaled_constraint_infinity_norm"))
    initial_stationarity = _finite_number(
        optimizer.get("initial_raw_kkt_stationarity_infinity_norm")
    )
    final_stationarity = _finite_number(
        endpoint.get("raw_kkt_stationarity_infinity_norm")
    )
    progress = (
        initial_objective,
        final_objective,
        initial_feasibility,
        final_feasibility,
        initial_stationarity,
        final_stationarity,
    )
    if any(value is None for value in progress):
        reasons.append("PROGRESS_EVIDENCE_NONFINITE")
    else:
        assert all(value is not None for value in progress)
        if not final_objective < initial_objective:
            reasons.append("OBJECTIVE_NOT_DECREASED")
        if not final_feasibility <= max(initial_feasibility, 1.0e-10):
            reasons.append("FEASIBILITY_NOT_MAINTAINED")
        if not final_stationarity < initial_stationarity:
            reasons.append("RAW_KKT_NOT_DECREASED")

    solve_values = {
        "NORMAL_RESIDUAL": _finite_number(
            optimizer.get("final_normal_relative_residual")
        ),
        "NORMAL_FORWARD_ERROR": _finite_number(
            optimizer.get("final_normal_forward_error_bound")
        ),
        "TANGENCY_RESIDUAL": _finite_number(
            optimizer.get("final_tangency_relative_residual")
        ),
        "MULTIPLIER_PROJECTION_RESIDUAL": _finite_number(
            optimizer.get("final_multiplier_projection_relative_residual")
        ),
        "MULTIPLIER_PROJECTION_FORWARD_ERROR": _finite_number(
            optimizer.get("final_multiplier_projection_forward_error_bound")
        ),
        "KKT_RESIDUAL": _finite_number(kkt.get("kkt_relative_residual")),
        "SCHUR_RESIDUAL": _finite_number(kkt.get("schur_relative_residual")),
        "KKT_SCALED_RESIDUAL": _finite_number(kkt.get("kkt_solution_scaled_residual")),
        "KKT_FORWARD_ERROR": _finite_number(kkt.get("kkt_forward_error_bound")),
    }
    if kkt.get("valid") is not True or kkt.get("all_finite") is not True:
        reasons.append("KKT_INVALID")
    for name, value in solve_values.items():
        maximum = (
            FTR_FORWARD_ERROR_MAXIMUM
            if name.endswith("FORWARD_ERROR")
            else FTR_LINEAR_RESIDUAL_MAXIMUM
        )
        if value is None or value > maximum:
            reasons.append(name)

    if (
        transfers.get("initial_h2d_calls") != 1
        or transfers.get("hot_h2d_calls") != 0
        or transfers.get("hot_d2h_calls") != 0
        or transfers.get("final_d2h_calls") != 1
    ):
        reasons.append("TRANSFER_BUDGET")
    peak_fraction = _finite_number(memory.get("peak_memory_fraction"))
    if peak_fraction is None or not 0.0 <= peak_fraction < FTR_MAXIMUM_MEMORY_FRACTION:
        reasons.append("MEMORY_BUDGET")
    synchronized_seconds = _finite_number(timing.get("synchronized_solve_seconds"))
    projected_seconds = (
        None if synchronized_seconds is None else 10.0 * synchronized_seconds
    )
    if (
        projected_seconds is None
        or projected_seconds >= FTR_PROJECTED_SOLVE_MAX_SECONDS
    ):
        reasons.append("PROJECTED_TIME_EXCEEDED")

    return {
        "schema_version": FTR_GATE_RECEIPT_SCHEMA_VERSION,
        "gate_status": "PASS" if not reasons else "FAIL",
        "failure_reasons": reasons,
        "expected_iterations": 10,
        "initial_state": "bootstrap",
        "initial_physical_objective": initial_objective,
        "final_physical_objective": final_objective,
        "initial_scaled_feasibility_inf": initial_feasibility,
        "final_scaled_feasibility_inf": final_feasibility,
        "initial_raw_kkt_stationarity_inf": initial_stationarity,
        "final_raw_kkt_stationarity_inf": final_stationarity,
        "synchronized_solve_seconds": synchronized_seconds,
        "projected_100_iteration_s": projected_seconds,
        "projection_formula": "10 * synchronized_solve_seconds",
    }


def ftr_gate_receipt_payload(
    raw: dict[str, JsonValue],
    raw_ref: ArtifactRef,
    memory: dict[str, JsonValue],
    memory_ref: ArtifactRef,
) -> dict[str, JsonValue]:
    """Bind a recomputed Gate 2 verdict to immutable raw and memory bytes."""

    detail = derive_ftr_gate_detail(raw, memory)
    return {
        "schema_version": FTR_GATE_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": contract_sha256_v3(),
        "plan_sha256": FTR_PLAN_SHA256,
        "budget_sha256": FTR_BUDGET_SHA256,
        "request": raw["request"],
        "source_identity": raw["source_identity"],
        "runtime_evidence": raw["runtime_evidence"],
        "bootstrap_artifact": raw["bootstrap_artifact"],
        "raw_result": asdict(raw_ref),
        "gpu_memory": asdict(memory_ref),
        "gate_status": detail["gate_status"],
        "failure_reasons": detail["failure_reasons"],
        "canary_gate": detail,
    }


def validate_ftr_device_binding(
    runtime: _snapshot_contract.RuntimeEvidence,
    memory: dict[str, JsonValue],
) -> None:
    """Bind Gate 2 to the frozen RTX 5090 and its process monitor."""

    expect_exact_keys(memory, _FTR_MEMORY_KEYS, context="FTR GPU memory")
    identity = runtime.observation.runtime_identity
    if (
        identity.device_uuid != FTR_PRIMARY_DEVICE_UUID
        or "5090" not in runtime.observation.device_name
    ):
        raise ValueError("FTR Gate 2 runtime is not the frozen RTX 5090")
    if expect_string(memory["device_uuid"], context="memory.device_uuid") != (
        identity.device_uuid
    ):
        raise ValueError("FTR GPU-memory device differs from runtime")
    if memory["monitor_scope"] != "whole-child-exact-pid-exact-device":
        raise ValueError("FTR GPU-memory monitor scope is invalid")
    for key in (
        "parent_pid",
        "child_pid",
        "child_start_time_ticks",
        "sample_count",
        "peak_memory_bytes",
    ):
        if expect_integer(memory[key], context=f"memory.{key}") <= 0:
            raise ValueError(f"FTR GPU-memory {key} must be positive")
    expected_argv_sha = hashlib.sha256(
        _snapshot_contract.canonical_json_bytes(list(identity.argv))
    ).hexdigest()
    if (
        expect_string(memory["child_argv_sha256"], context="memory.child_argv_sha256")
        != expected_argv_sha
    ):
        raise ValueError("FTR GPU-memory child argv differs from runtime")


def load_ftr_gate_result(campaign_root: Path, artifact: ArtifactRef) -> FtrGateResult:
    """Resolve, rehash, and semantically revalidate one FTR Gate 2 receipt."""

    root = campaign_root.resolve(strict=True)
    path = artifact.resolve_and_validate(root)
    receipt = expect_mapping(
        load_canonical_json_bytes(path.read_bytes()), context="FTR gate receipt"
    )
    expect_exact_keys(receipt, _COMMON_RECEIPT_KEYS, context="FTR gate receipt")
    if (
        artifact.schema_version != FTR_GATE_RECEIPT_SCHEMA_VERSION
        or receipt["schema_version"] != FTR_GATE_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("FTR gate receipt schema mismatch")
    if (
        receipt["contract_sha256"] != contract_sha256_v3()
        or receipt["plan_sha256"] != FTR_PLAN_SHA256
        or receipt["budget_sha256"] != FTR_BUDGET_SHA256
    ):
        raise ValueError("FTR gate receipt contract identity mismatch")
    if run_request_v3_from_payload(receipt["request"]) != _EXPECTED_REQUEST:
        raise ValueError("FTR gate receipt request differs from frozen Gate 2")

    raw_ref = artifact_ref_from_payload(receipt["raw_result"])
    memory_ref = artifact_ref_from_payload(receipt["gpu_memory"])
    raw = expect_mapping(
        load_canonical_json_bytes(raw_ref.resolve_and_validate(root).read_bytes()),
        context="FTR raw result",
    )
    memory = expect_mapping(
        load_canonical_json_bytes(memory_ref.resolve_and_validate(root).read_bytes()),
        context="FTR GPU memory",
    )
    if raw_ref.schema_version != FTR_RESULT_SCHEMA_VERSION or (
        raw.get("schema_version") != FTR_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("FTR raw result schema mismatch")
    if memory_ref.schema_version != FTR_MEMORY_SCHEMA_VERSION or (
        memory.get("schema_version") != FTR_MEMORY_SCHEMA_VERSION
    ):
        raise ValueError("FTR memory schema mismatch")
    for identity_key in ("contract_sha256", "plan_sha256", "budget_sha256"):
        if raw.get(identity_key) != receipt[identity_key]:
            raise ValueError(f"FTR raw and receipt {identity_key} differ")
    for evidence_key in (
        "request",
        "source_identity",
        "runtime_evidence",
        "bootstrap_artifact",
    ):
        if raw.get(evidence_key) != receipt[evidence_key]:
            raise ValueError(f"FTR raw and receipt {evidence_key} differ")

    runtime_ref = artifact_ref_from_payload(receipt["runtime_evidence"])
    bootstrap_ref = artifact_ref_from_payload(receipt["bootstrap_artifact"])
    source = _snapshot_contract.source_identity_from_payload(receipt["source_identity"])
    manifest_path = source.snapshot_manifest.resolve_and_validate(root)
    runtime = _snapshot_contract.validate_runtime_evidence(
        runtime_ref.resolve_and_validate(root),
        snapshot_root=manifest_path.parent,
        campaign_root=root,
    )
    if runtime.source_identity != source:
        raise ValueError("FTR runtime evidence source identity mismatch")
    validate_ftr_device_binding(runtime, dict(memory))
    from benchmarks.single_stage_fullspace_bootstrap import validate_bootstrap_artifact

    bootstrap = validate_bootstrap_artifact(
        bootstrap_ref.resolve_and_validate(root),
        campaign_root=root,
        snapshot_root=manifest_path.parent,
    )
    if bootstrap.get("runtime_evidence") != receipt["runtime_evidence"]:
        raise ValueError("FTR bootstrap runtime identity mismatch")

    detail = derive_ftr_gate_detail(dict(raw), dict(memory))
    if receipt["canary_gate"] != detail:
        raise ValueError("FTR gate detail differs from raw evidence")
    if receipt["failure_reasons"] != detail["failure_reasons"] or (
        receipt["gate_status"] != detail["gate_status"]
    ):
        raise ValueError("FTR outer verdict differs from recomputed evidence")
    reasons = tuple(str(reason) for reason in detail["failure_reasons"])
    return FtrGateResult(artifact, not reasons, reasons)


def gate_artifact_from_path(campaign_root: Path, path: Path) -> ArtifactRef:
    """Build a content-addressed reference for an existing immutable receipt."""

    root = campaign_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved.stat().st_mode & 0o222:
        raise ValueError("FTR gate receipt must be immutable")
    encoded = resolved.read_bytes()
    return ArtifactRef(
        relative_path=resolved.relative_to(root).as_posix(),
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        schema_version=FTR_GATE_RECEIPT_SCHEMA_VERSION,
    )


__all__ = (
    "FTR_BUDGET_SHA256",
    "FTR_GATE_RECEIPT_SCHEMA_VERSION",
    "FTR_MEMORY_SCHEMA_VERSION",
    "FTR_PLAN_SHA256",
    "FTR_PRIMARY_DEVICE_UUID",
    "FTR_RESULT_SCHEMA_VERSION",
    "FtrGateResult",
    "derive_ftr_gate_detail",
    "ftr_gate_receipt_payload",
    "gate_artifact_from_path",
    "load_ftr_gate_result",
    "validate_ftr_device_binding",
)
