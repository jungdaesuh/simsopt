from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from benchmarks import single_stage_fullspace_receipt as receipt
from benchmarks import validate_single_stage_fullspace_campaign as validator
from benchmarks.single_stage_fullspace_receipt import (
    SQP_BUDGET_SHA256,
    SQP_CERTIFICATE_ENVELOPE_SCHEMA_VERSION,
    SQP_KKT_RELATIVE_RESIDUAL_MAXIMUM,
    SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM,
    SQP_MEMORY_SCHEMA_VERSION,
    SQP_PLAN_SHA256,
    SQP_RESULT_SCHEMA_VERSION,
    SQP_SAMPLE_SCHEMA_VERSION,
    SQP_SCHUR_RELATIVE_RESIDUAL_MAXIMUM,
    SQP_WARM_SOLVE_MAX_SECONDS,
    V2_ROUTES,
    BaselineClassification,
    CampaignDisposition,
    CampaignReceiptV2,
    CompleteSample,
    DeviceLane,
    GateEvidence,
    JsonValue,
    RouteDisposition,
    RouteOutcome,
    RunPhase,
    RunRequest,
    SqpCampaignV2Collector,
    SqpGate,
    SqpGateResult,
    SqpSampleReceipt,
    campaign_receipt_v2_from_payload,
    campaign_receipt_v2_payload,
    canonical_json_bytes,
    contract_sha256_v1,
    contract_sha256_v2,
    load_canonical_json_bytes,
    run_request_payload_v2,
    run_request_v2_from_payload,
    sqp_sample_receipt_from_payload,
    sqp_sample_receipt_payload,
)
from benchmarks.single_stage_fullspace_snapshot import (
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ArtifactRef,
    RuntimeIdentity,
    SourceIdentity,
)
from simsopt_jax.solve.fullspace import LEGACY_V1_ROUTES, FullSpaceRoute
from simsopt_jax.solve.fullspace_certificate import (
    CFS_SQP1_CERTIFICATE_SCHEMA_VERSION,
)


def _artifact(relative_path: str, schema_version: str = "fixture-v1") -> ArtifactRef:
    payload = canonical_json_bytes(
        {"fixture_identity": relative_path, "schema_version": schema_version}
    )
    return ArtifactRef(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=schema_version,
    )


def _source(suffix: str = "a") -> SourceIdentity:
    return SourceIdentity(
        snapshot_manifest=ArtifactRef(
            f"snapshot-{suffix}/source-manifest.json",
            suffix * 64,
            1,
            SOURCE_MANIFEST_SCHEMA_VERSION,
        ),
        git_head=suffix * 40,
        tracked_diff_sha256="b" * 64,
        untracked_bytes_manifest_sha256="c" * 64,
        repo_root=f"/repo/{suffix}",
    )


def _runtime(device_uuid: str = "GPU-fixture") -> RuntimeIdentity:
    return RuntimeIdentity(
        argv=("python", "runner.py"),
        cwd="/snapshot",
        python_executable="/venv/bin/python",
        python_version="3.13",
        jax_version="0.6",
        jaxlib_version="0.6",
        simsopt_module_path="/snapshot/src/simsopt/__init__.py",
        simsopt_jax_module_path="/snapshot/src/simsopt_jax/__init__.py",
        native_extension_path="/snapshot/lib/simsoptpp.so",
        backend="gpu",
        device_uuid=device_uuid,
        driver_version="driver",
        effective_environment_sha256="d" * 64,
    )


def _sample(
    sample: CompleteSample,
    *,
    promotion_eligible: bool,
    source: SourceIdentity | None = None,
    runtime: RuntimeIdentity | None = None,
    bootstrap_sha: str = "e" * 64,
) -> SqpSampleReceipt:
    return SqpSampleReceipt(
        request=RunRequest(
            phase=RunPhase.COMPLETE,
            route=FullSpaceRoute.CFS_SQP1,
            device=DeviceLane.RTX5090,
            steps=None,
            sample=sample,
        ),
        source_identity=_source() if source is None else source,
        runtime_identity=_runtime() if runtime is None else runtime,
        runtime_evidence=_artifact(f"{sample}/runtime.json"),
        bootstrap_artifact=replace(
            _artifact(f"{sample}/bootstrap.json"), sha256=bootstrap_sha
        ),
        bootstrap_identity_sha256=bootstrap_sha,
        raw_result=_artifact(f"{sample}/raw.json"),
        gpu_memory=_artifact(f"{sample}/memory.json"),
        endpoint_certificate=(
            _artifact(f"{sample}/certificate.json") if promotion_eligible else None
        ),
        promotion_eligible=promotion_eligible,
        terminal_status="CONVERGED" if promotion_eligible else "ITERATION_LIMIT",
        synchronized_solve_seconds=100.0,
        total_child_wall_seconds=120.0,
        hot_h2d_transfers=0,
        hot_d2h_transfers=0,
        initial_h2d_transfers=1,
        final_d2h_transfers=1,
        peak_memory_bytes=1_000_000,
        peak_memory_fraction=0.5,
    )


def _optimizer_payload(sample: SqpSampleReceipt) -> dict[str, JsonValue]:
    feasibility: list[JsonValue] = [0.0]
    kkt_residual: list[JsonValue] = [0.0]
    objective: list[JsonValue] = [0.0]
    stationarity: list[JsonValue] = [0.0]
    status: list[JsonValue] = [0]
    step_length: list[JsonValue] = [0.0]
    history: dict[str, JsonValue] = {
        "accepted_length": 1,
        "feasibility_infinity_norm": feasibility,
        "kkt_relative_residual": kkt_residual,
        "objective": objective,
        "stationarity_infinity_norm": stationarity,
        "status": status,
        "step_length": step_length,
    }
    return {
        "all_accepted_states_finite": True,
        "all_finite": True,
        "bfgs_resets": 0,
        "converged": sample.terminal_status == "CONVERGED",
        "derivative_builds": 2,
        "failed": sample.terminal_status != "CONVERGED",
        "fatal": False,
        "final_kkt_relative_residual": 0.0,
        "final_kkt_reciprocal_condition": 0.01,
        "final_kkt_solution_scaled_residual": 0.0,
        "final_schur_relative_residual": 0.0,
        "final_bfgs_cholesky_relative_pivot": 0.01,
        "final_schur_cholesky_relative_pivot": 0.01,
        "history": history,
        "history_sha256": hashlib.sha256(canonical_json_bytes(history)).hexdigest(),
        "iterations": 1,
        "joint_evaluations": 2,
        "kkt_solves": 1,
        "line_search_evaluations": 1,
        "merit_penalty": 1.0,
        "initial_physical_objective": 2.0e-8,
        "initial_scaled_constraint_infinity_norm": 2.0e-12,
        "initial_raw_kkt_stationarity_infinity_norm": 2.0e-8,
        "optimizer_coordinates_sha256": _vector_sha256(_vector(716)),
        "physical_objective": 1.0e-8,
        "physical_state_sha256": _vector_sha256(_vector(716)),
        "raw_constraint_infinity_norm": 1.0e-12,
        "raw_kkt_stationarity_infinity_norm": 1.0e-8,
        "raw_multipliers_sha256": _vector_sha256(_vector(255)),
        "regularization_candidates_tested": 1,
        "regularization_uses": 0,
        "rejected_nonfinite_trials": 0,
        "scaled_constraint_infinity_norm": 1.0e-12,
        "scaled_multipliers_sha256": _vector_sha256(_vector(255)),
        "selected_regularization": 0.0,
        "status": sample.terminal_status,
    }


def _vector(size: int, value: float = 0.0) -> list[JsonValue]:
    result: list[JsonValue] = [value] * size
    return result


def _vector_sha256(value: list[JsonValue]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_payload(sample: SqpSampleReceipt) -> dict[str, JsonValue]:
    optimizer = _optimizer_payload(sample)
    physical_state = _vector(716)
    optimizer_coordinates = _vector(716)
    scaled_multipliers = _vector(255)
    raw_multipliers = _vector(255)
    endpoint: dict[str, JsonValue] = {
        "optimizer_coordinates": optimizer_coordinates,
        "optimizer_coordinates_sha256": _vector_sha256(optimizer_coordinates),
        "physical_objective": optimizer["physical_objective"],
        "physical_state": physical_state,
        "physical_state_sha256": _vector_sha256(physical_state),
        "raw_constraint_infinity_norm": optimizer["raw_constraint_infinity_norm"],
        "raw_kkt_stationarity_infinity_norm": optimizer[
            "raw_kkt_stationarity_infinity_norm"
        ],
        "raw_multipliers": raw_multipliers,
        "raw_multipliers_sha256": _vector_sha256(raw_multipliers),
        "scaled_constraint_infinity_norm": optimizer["scaled_constraint_infinity_norm"],
        "scaled_multipliers": scaled_multipliers,
        "scaled_multipliers_sha256": _vector_sha256(scaled_multipliers),
    }
    endpoint["all_finite"] = True
    return {
        "bootstrap_artifact": asdict(sample.bootstrap_artifact),
        "budget_sha256": SQP_BUDGET_SHA256,
        "contract_sha256": contract_sha256_v2(),
        "plan_sha256": SQP_PLAN_SHA256,
        "endpoint": endpoint,
        "endpoint_certificate": None,
        "optimizer_result": optimizer,
        "promotion_eligible": False,
        "request": asdict(sample.request),
        "runtime_evidence": asdict(sample.runtime_evidence),
        "schema_version": SQP_RESULT_SCHEMA_VERSION,
        "source_identity": asdict(sample.source_identity),
        "terminal_status": sample.terminal_status,
        "timing": {
            "synchronized_solve_seconds": sample.synchronized_solve_seconds,
            "total_child_wall_seconds": sample.total_child_wall_seconds,
        },
        "trajectory_equivalence_required": False,
        "transfer_audit": {
            "final_d2h_calls": sample.final_d2h_transfers,
            "hot_d2h_calls": sample.hot_d2h_transfers,
            "hot_h2d_calls": sample.hot_h2d_transfers,
            "initial_h2d_calls": sample.initial_h2d_transfers,
        },
    }


def _memory_payload(sample: SqpSampleReceipt) -> dict[str, JsonValue]:
    return {
        "child_argv_sha256": hashlib.sha256(
            canonical_json_bytes(list(sample.runtime_identity.argv))
        ).hexdigest(),
        "child_pid": 2,
        "child_start_time_ticks": 3,
        "device_uuid": sample.runtime_identity.device_uuid,
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "parent_pid": 1,
        "peak_memory_bytes": sample.peak_memory_bytes,
        "peak_memory_fraction": sample.peak_memory_fraction,
        "sample_count": 4,
        "schema_version": SQP_MEMORY_SCHEMA_VERSION,
    }


def _certificate_payload(
    sample: SqpSampleReceipt, raw: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    endpoint = raw["endpoint"]
    assert isinstance(endpoint, dict)
    state = endpoint["physical_state"]
    raw_multipliers = endpoint["raw_multipliers"]
    assert isinstance(state, list)
    assert isinstance(raw_multipliers, list)
    endpoint_numerics: dict[str, JsonValue] = {
        "all_finite_fp64": True,
        "boozer_residual_infinity_norm": 0.0,
        "constraints": _vector(255),
        "iota": 1.0,
        "major_radius": 1.5,
        "objective": endpoint["physical_objective"],
        "objective_ledger_consistent": True,
        "objective_gradient": _vector(716),
        "one_sided_length_penalty": 0.0,
        "raw_objective_terms": {
            "iota": 0.0,
            "length": 0.0,
            "major_radius": 0.0,
            "non_qs": endpoint["physical_objective"],
            "residual": 0.0,
        },
        "raw_kkt_stationarity_infinity_norm": endpoint[
            "raw_kkt_stationarity_infinity_norm"
        ],
        "scaled_constraints": _vector(255),
        "scaled_feasibility_infinity_norm": endpoint["scaled_constraint_infinity_norm"],
        "state": list(state),
        "stationarity_gradient": _vector(716),
        "volume_residual_absolute": 0.0,
    }
    checks: dict[str, JsonValue] = {
        "branch": True,
        "cross_evaluator": True,
        "field_line": True,
        "finite_fp64": True,
        "fixed_state_preserved": True,
        "inactive_hardware_terms_valid": True,
        "objective_reference_valid": True,
        "objective_ledger_consistent": True,
        "objective_threshold": True,
        "optimizer_termination": True,
        "post_projection_certifiable": True,
        "pre_projection_certifiable": True,
        "projection_bound_to_solver_endpoint": True,
        "projection_immaterial": True,
        "raw_kkt_stationarity": True,
        "scaled_feasibility": True,
        "solver_result_consistent": True,
    }
    scientific: dict[str, JsonValue] = {
        "branch": {
            "basin_classification": "same-root",
            "exact_solve_succeeded": True,
            "material_branch_switch": False,
            "performed": True,
            "reproduced_state_infinity_difference": 0.0,
        },
        "certified": True,
        "checks": checks,
        "cross_evaluator": {
            "jax_on_native_endpoint_objective": 1.0e-8,
            "native_on_jax_endpoint_objective": endpoint["physical_objective"],
            "performed": True,
        },
        "field_line": {
            "performed": True,
            "poincare_closed": True,
            "traced_iota": 1.0,
        },
        "inactive_hardware": {
            "metrics": _vector(4),
            "names": [
                "curvature",
                "curve_curve",
                "curve_surface",
                "surface_vessel",
            ],
            "weights": _vector(4),
        },
        "multipliers": list(raw_multipliers),
        "objective_reference": {"native_reference_objective": 4.4822247e-08},
        "post_projection": endpoint_numerics,
        "pre_projection": endpoint_numerics,
        "projection": {
            "evaluated": True,
            "post_state": list(state),
            "pre_state": list(state),
            "used": False,
        },
        "route": FullSpaceRoute.CFS_SQP1,
        "schema_version": CFS_SQP1_CERTIFICATE_SCHEMA_VERSION,
        "termination": "CONVERGED",
    }
    return {
        "bootstrap_identity_sha256": sample.bootstrap_identity_sha256,
        "endpoint_sha256": hashlib.sha256(
            canonical_json_bytes(raw["endpoint"])
        ).hexdigest(),
        "raw_result_sha256": sample.raw_result.sha256,
        "runtime_evidence_sha256": sample.runtime_evidence.sha256,
        "schema_version": SQP_CERTIFICATE_ENVELOPE_SCHEMA_VERSION,
        "scientific_certificate": scientific,
        "source_manifest_sha256": sample.source_identity.snapshot_manifest.sha256,
    }


def _outcomes(
    cold_ref: ArtifactRef, *, terminal_status: str
) -> tuple[RouteOutcome, ...]:
    gate = GateEvidence("PRIOR-ROUTE", _artifact("gates/prior.json"))
    legacy = tuple(
        RouteOutcome(
            route=route,
            disposition=RouteDisposition.NOT_SELECTED_BY_GATE,
            terminal_status=None,
            receipt=None,
            upstream_gate="PRIOR-ROUTE",
            gate_evidence=(gate,),
        )
        for route in LEGACY_V1_ROUTES
    )
    return (
        *legacy,
        RouteOutcome(
            route=FullSpaceRoute.CFS_SQP1,
            disposition=RouteDisposition.EXECUTED,
            terminal_status=terminal_status,
            receipt=cold_ref,
            upstream_gate=None,
            gate_evidence=(),
        ),
    )


def _campaign(
    sample_refs: tuple[ArtifactRef, ...], *, terminal_status: str = "CONVERGED"
) -> CampaignReceiptV2:
    return CampaignReceiptV2(
        disposition=CampaignDisposition.BOUNDED_NEGATIVE,
        baseline_classification=BaselineClassification.HISTORICAL_ENGINEERING_ONLY,
        contract_sha256=contract_sha256_v2(),
        route_outcomes=_outcomes(sample_refs[0], terminal_status=terminal_status),
        sqp_samples=sample_refs,
    )


def _collector(campaign_root: Path) -> SqpCampaignV2Collector:
    cold_ref = _artifact("collector-placeholder.json", SQP_SAMPLE_SCHEMA_VERSION)
    return SqpCampaignV2Collector.create(
        baseline_classification=BaselineClassification.HISTORICAL_ENGINEERING_ONLY,
        legacy_route_outcomes=_outcomes(cold_ref, terminal_status="CONVERGED")[:-1],
        campaign_root=campaign_root,
    )


def _passed_gates(collector: SqpCampaignV2Collector) -> SqpCampaignV2Collector:
    for gate, name in (
        (SqpGate.DERIVATIVE, "derivative"),
        (SqpGate.CANARY_1, "canary-1"),
        (SqpGate.CANARY_10, "canary-10"),
    ):
        collector = collector.record_gate(gate, _artifact(f"gates/{name}.json"))
    return collector


def _install_collector_gate_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_gate(_root: Path, gate: SqpGate, artifact: ArtifactRef) -> SqpGateResult:
        failed = "failed" in artifact.relative_path
        return SqpGateResult(
            gate,
            artifact,
            not failed,
            ("FIXTURE_GATE_FAILURE",) if failed else (),
        )

    monkeypatch.setattr(
        "benchmarks.single_stage_fullspace_receipt.load_sqp_gate_result", load_gate
    )


def _sample_artifact(relative_path: str, receipt: SqpSampleReceipt) -> ArtifactRef:
    payload = canonical_json_bytes(sqp_sample_receipt_payload(receipt))
    return ArtifactRef(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=SQP_SAMPLE_SCHEMA_VERSION,
    )


def _write_sample_placeholders(root: Path, references: tuple[ArtifactRef, ...]) -> None:
    for reference in references:
        path = root / reference.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_bytes(
                {
                    "fixture_identity": reference.relative_path,
                    "schema_version": reference.schema_version,
                }
            )
        )


def test_sqp_sample_receipt_round_trip_is_strict() -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)
    payload = sqp_sample_receipt_payload(sample)

    assert payload["schema_version"] == SQP_SAMPLE_SCHEMA_VERSION
    assert payload["contract_sha256"] == contract_sha256_v2()
    assert sqp_sample_receipt_from_payload(payload) == sample

    payload["unexpected"] = True
    with pytest.raises(ValueError, match="keys do not match"):
        sqp_sample_receipt_from_payload(payload)


def test_sqp_sample_cannot_claim_promotion_without_certificate() -> None:
    sample = replace(
        _sample(CompleteSample.COLD, promotion_eligible=False),
        promotion_eligible=True,
    )

    with pytest.raises(ValueError, match="requires an endpoint certificate"):
        sqp_sample_receipt_payload(sample)


def test_sqp_v2_request_rejects_a100_lane() -> None:
    request = replace(
        _sample(CompleteSample.COLD, promotion_eligible=False).request,
        device=DeviceLane.A100,
    )

    with pytest.raises(ValueError, match="require RTX 5090"):
        request.validate_v2()


def test_sqp_first_eval_derivative_gate_request_round_trips() -> None:
    request = RunRequest(
        phase=RunPhase.FIRST_EVAL,
        route=FullSpaceRoute.CFS_SQP1,
        device=DeviceLane.RTX5090,
        steps=None,
        sample=None,
    )

    payload = run_request_payload_v2(request)

    assert run_request_v2_from_payload(payload["request"]) == request


@pytest.mark.parametrize(
    ("phase", "steps", "sample", "message"),
    (
        (RunPhase.FIRST_EVAL, 1, None, "first-eval forbids"),
        (RunPhase.FIRST_EVAL, None, CompleteSample.COLD, "first-eval forbids"),
        (RunPhase.CANARY, 100, None, "steps=1 or 10"),
    ),
)
def test_sqp_request_rejects_invalid_derivative_and_canary_shapes(
    phase: RunPhase,
    steps: int | None,
    sample: CompleteSample | None,
    message: str,
) -> None:
    request = RunRequest(
        phase=phase,
        route=FullSpaceRoute.CFS_SQP1,
        device=DeviceLane.RTX5090,
        steps=steps,
        sample=sample,
    )

    with pytest.raises(ValueError, match=message):
        request.validate_v2()


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("nan-timing", "timing"),
        ("short-child", "timing"),
        ("hot-transfer", "hot transfers"),
        ("missing-boundary", "one initial H2D"),
        ("memory", "memory"),
    ),
)
def test_sqp_sample_rejects_invalid_timing_transfer_and_memory_metrics(
    defect: str, message: str
) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=False)
    if defect == "nan-timing":
        sample = replace(sample, synchronized_solve_seconds=float("nan"))
    elif defect == "short-child":
        sample = replace(sample, total_child_wall_seconds=99.0)
    elif defect == "hot-transfer":
        sample = replace(sample, hot_h2d_transfers=1)
    elif defect == "missing-boundary":
        sample = replace(sample, initial_h2d_transfers=0)
    else:
        sample = replace(sample, peak_memory_fraction=0.8)

    with pytest.raises(ValueError, match=message):
        sample.validate()


def test_sqp_raw_result_exact_schema_and_bindings_accept_canonical_payload() -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)

    validator._validate_sqp_raw_result(
        _raw_payload(sample), sample, sample.source_identity
    )


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("top-key", "keys do not match"),
        ("endpoint", "digest differs"),
        ("history", "history digest differs"),
        ("status", "converged and failed"),
    ),
)
def test_sqp_raw_result_rejects_top_endpoint_history_and_status_tamper(
    defect: str, message: str
) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)
    raw = _raw_payload(sample)
    if defect == "top-key":
        raw["unexpected"] = True
    elif defect == "endpoint":
        endpoint = raw["endpoint"]
        assert isinstance(endpoint, dict)
        endpoint["physical_state_sha256"] = "f" * 64
    else:
        optimizer = raw["optimizer_result"]
        assert isinstance(optimizer, dict)
        if defect == "history":
            history = optimizer["history"]
            assert isinstance(history, dict)
            objective = history["objective"]
            assert isinstance(objective, list)
            objective[0] = 1.0
        else:
            optimizer["fatal"] = True

    with pytest.raises(ValueError, match=message):
        validator._validate_sqp_raw_result(raw, sample, sample.source_identity)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("plan_sha256", "f" * 64, "plan digest"),
        ("budget_sha256", "f" * 64, "budget digest"),
    ),
)
def test_sqp_raw_result_rejects_frozen_plan_and_budget_tamper(
    field: str, replacement: JsonValue, message: str
) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)
    raw = _raw_payload(sample)
    raw[field] = replacement

    with pytest.raises(ValueError, match=message):
        validator._validate_sqp_raw_result(raw, sample, sample.source_identity)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "final_kkt_reciprocal_condition",
            0.0,
            "reciprocal condition",
        ),
        (
            "final_kkt_solution_scaled_residual",
            SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM * 2.0,
            "solution-scaled residual",
        ),
        (
            "final_kkt_relative_residual",
            SQP_KKT_RELATIVE_RESIDUAL_MAXIMUM * 2.0,
            "relative residual",
        ),
        (
            "final_schur_relative_residual",
            SQP_SCHUR_RELATIVE_RESIDUAL_MAXIMUM * 2.0,
            "Schur relative residual",
        ),
    ),
)
def test_sqp_raw_result_rejects_kkt_condition_and_solution_residual_gate_failure(
    field: str, replacement: JsonValue, message: str
) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)
    raw = _raw_payload(sample)
    optimizer = raw["optimizer_result"]
    assert isinstance(optimizer, dict)
    optimizer[field] = replacement

    with pytest.raises(ValueError, match=message):
        validator._validate_sqp_raw_result(raw, sample, sample.source_identity)


def test_sqp_failed_condition_candidate_retains_finite_diagnostics() -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=False)
    raw = _raw_payload(sample)
    optimizer = raw["optimizer_result"]
    assert isinstance(optimizer, dict)
    optimizer["final_kkt_reciprocal_condition"] = 0.0
    optimizer["final_kkt_solution_scaled_residual"] = (
        SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM * 2.0
    )

    validator._validate_sqp_raw_result(raw, sample, sample.source_identity)


def _nonfinite_derivative_gate_raw(*, zero_solution: bool) -> dict[str, JsonValue]:
    constraint_jacobian = np.zeros((255, 716), dtype=np.float64)
    constraint_jacobian[:, :255] = np.eye(255, dtype=np.float64)
    objective_gradient = np.zeros((716,), dtype=np.float64)
    scaled_constraints = (
        np.ones((255,), dtype=np.float64)
        if zero_solution
        else np.zeros((255,), dtype=np.float64)
    )
    joint_vjp_rows = np.vstack((objective_gradient, constraint_jacobian))
    direction = np.linspace(-0.75, 1.0, 716, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    cotangent = np.linspace(0.5, -1.25, 255, dtype=np.float64)
    cotangent /= np.linalg.norm(cotangent)
    av = constraint_jacobian @ direction
    atw = constraint_jacobian.T @ cotangent
    state = {
        "all_finite": True,
        "atw": atw.tolist(),
        "av": av.tolist(),
        "constraint_jacobian": constraint_jacobian.tolist(),
        "joint_vjp_rows": joint_vjp_rows.tolist(),
        "joint_vjp_rows_dtype": "float64",
        "joint_vjp_rows_sha256": hashlib.sha256(
            canonical_json_bytes(joint_vjp_rows.tolist())
        ).hexdigest(),
        "joint_vjp_rows_shape": [256, 716],
        "numerical_rank": 255,
        "objective_gradient": objective_gradient.tolist(),
        "physical_objective": 0.0,
        "rank_cutoff": 1.0e-12,
        "rank_relative_threshold": 1.0e-12,
        "scaled_constraints": scaled_constraints.tolist(),
        "scaled_constraints_sha256": hashlib.sha256(
            canonical_json_bytes(scaled_constraints.tolist())
        ).hexdigest(),
        "sigma_maximum": 1.0,
        "sigma_minimum": 1.0,
        "singular_values": np.ones((255,), dtype=np.float64).tolist(),
        "transpose_absolute_error": 0.0,
        "transpose_lhs": float(np.vdot(cotangent, av)),
        "transpose_relative_error": 0.0,
        "transpose_rhs": float(np.vdot(atw, direction)),
    }
    kkt_matrix = np.block(
        [
            [np.eye(716), constraint_jacobian.T],
            [constraint_jacobian, np.zeros((255, 255))],
        ]
    )
    multiplier_step = (
        np.zeros((255,), dtype=np.float64)
        if zero_solution
        else np.ones((255,), dtype=np.float64)
    )
    primal_step = np.zeros((716,), dtype=np.float64)
    solution = np.concatenate((primal_step, multiplier_step))
    right_hand_side = -np.concatenate((objective_gradient, scaled_constraints))
    residual = kkt_matrix @ solution - right_hand_side
    eigenvalue_magnitudes = np.abs(np.linalg.eigvalsh(kkt_matrix))
    sigma_maximum = float(np.max(eigenvalue_magnitudes))
    rho_k = float(np.min(eigenvalue_magnitudes)) / sigma_maximum
    residual_two = float(np.linalg.norm(residual, ord=2))
    solution_two = float(np.linalg.norm(solution, ord=2))
    zeta_2 = (
        residual_two / (sigma_maximum * solution_two)
        if solution_two > 0.0
        else math.inf
    )
    certified_error_bound = zeta_2 / (rho_k - zeta_2) if rho_k > zeta_2 else math.inf
    kkt_relative_residual = float(np.linalg.norm(residual, ord=np.inf)) / max(
        1.0,
        float(np.linalg.norm(kkt_matrix, ord=np.inf))
        * float(np.linalg.norm(solution, ord=np.inf))
        + float(np.linalg.norm(right_hand_side, ord=np.inf)),
    )
    schur_rhs = scaled_constraints
    schur_relative_residual = float(
        np.linalg.norm(
            constraint_jacobian @ constraint_jacobian.T @ multiplier_step - schur_rhs,
            ord=np.inf,
        )
    ) / max(
        1.0,
        float(np.linalg.norm(constraint_jacobian @ constraint_jacobian.T, ord=np.inf))
        * float(np.linalg.norm(multiplier_step, ord=np.inf))
        + float(np.linalg.norm(schur_rhs, ord=np.inf)),
    )
    kkt = {
        "all_finite": False,
        "bfgs_cholesky_relative_pivot": 1.0,
        "certified_relative_error_bound": (
            certified_error_bound if math.isfinite(certified_error_bound) else None
        ),
        "kkt_relative_residual": kkt_relative_residual,
        "multiplier_step": multiplier_step.tolist(),
        "primal_step": primal_step.tolist(),
        "reconstructed_residual_inf": float(np.linalg.norm(residual, ord=np.inf)),
        "reconstructed_residual_two": residual_two,
        "regularization_candidates_tested": 1,
        "rho_k": rho_k,
        "schur_cholesky_relative_pivot": 1.0,
        "schur_relative_residual": schur_relative_residual,
        "selected_regularization": 0.0,
        "valid": False,
        "zeta_2": zeta_2 if math.isfinite(zeta_2) else None,
    }
    reasons = ["KKT_INVALID", "KKT_NONFINITE", "KKT_RHO"]
    if not math.isfinite(zeta_2) or zeta_2 > 1.0e-10:
        reasons.append("KKT_ZETA")
    if kkt_relative_residual > 1.0e-10:
        reasons.append("KKT_RESIDUAL")
    if schur_relative_residual > 1.0e-10:
        reasons.append("SCHUR_RESIDUAL")
    if not math.isfinite(certified_error_bound) or certified_error_bound >= 1.0e-7:
        reasons.append("KKT_ERROR_BOUND")
    request = RunRequest(
        RunPhase.FIRST_EVAL,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        None,
        None,
    )
    return {
        "bootstrap_artifact": {},
        "budget_sha256": SQP_BUDGET_SHA256,
        "contract_sha256": contract_sha256_v2(),
        "derivative_kkt_gate": {
            "bootstrap": state,
            "changed": state,
            "changed_optimizer_coordinates": np.zeros((716,)).tolist(),
            "changed_physical_state": np.zeros((716,)).tolist(),
            "failure_reasons": reasons,
            "gate_status": "FAIL",
            "kkt": kkt,
            "optimizer_steps_executed": 0,
            "schema_version": receipt.SQP_DERIVATIVE_GATE_SCHEMA_VERSION,
        },
        "plan_sha256": SQP_PLAN_SHA256,
        "request": run_request_payload_v2(request)["request"],
        "runtime_evidence": {},
        "schema_version": receipt.SQP_DERIVATIVE_GATE_SCHEMA_VERSION,
        "source_identity": {},
        "terminal_status": "DERIVATIVE_KKT_GATE_COMPLETED",
        "timing": {},
        "transfer_audit": {
            "hot_h2d_calls": 0,
            "hot_d2h_calls": 0,
            "initial_h2d_calls": 1,
            "final_d2h_calls": 1,
        },
    }


@pytest.mark.parametrize("zero_solution", (False, True))
def test_derivative_gate_deep_validation_accepts_null_nonfinite_certificates(
    zero_solution: bool,
) -> None:
    raw = _nonfinite_derivative_gate_raw(zero_solution=zero_solution)
    encoded = canonical_json_bytes(raw)
    parsed = load_canonical_json_bytes(encoded)
    assert isinstance(parsed, dict)
    result = receipt._validate_derivative_gate(parsed)
    assert not result.passed
    assert "KKT_RHO" in result.failure_reasons
    assert "KKT_ERROR_BOUND" in result.failure_reasons


def test_sqp_zero_kkt_terminal_uses_null_diagnostics_and_empty_history() -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=False)
    raw = _raw_payload(sample)
    optimizer = raw["optimizer_result"]
    assert isinstance(optimizer, dict)
    for key in (
        "final_kkt_relative_residual",
        "final_kkt_reciprocal_condition",
        "final_kkt_solution_scaled_residual",
        "final_schur_relative_residual",
        "final_bfgs_cholesky_relative_pivot",
        "final_schur_cholesky_relative_pivot",
        "selected_regularization",
    ):
        optimizer[key] = None
    optimizer["iterations"] = 0
    optimizer["kkt_solves"] = 0
    optimizer["line_search_evaluations"] = 0
    optimizer["regularization_candidates_tested"] = 0
    history = optimizer["history"]
    assert isinstance(history, dict)
    history["accepted_length"] = 0
    for key in (
        "objective",
        "feasibility_infinity_norm",
        "stationarity_infinity_norm",
        "step_length",
        "kkt_relative_residual",
        "status",
    ):
        history[key] = []
    optimizer["history_sha256"] = hashlib.sha256(
        canonical_json_bytes(history)
    ).hexdigest()

    encoded = canonical_json_bytes(raw)
    assert b"NaN" not in encoded
    parsed = load_canonical_json_bytes(encoded)
    assert isinstance(parsed, dict)
    validator._validate_sqp_raw_result(parsed, sample, sample.source_identity)


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("zero-kkt-value", "requires null regularization"),
        ("outside-ladder", "not in the frozen ladder"),
        ("candidate-counter", "counters are inconsistent"),
    ),
)
def test_sqp_selected_regularization_is_bound_to_kkt_counters(
    defect: str, message: str
) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=False)
    raw = _raw_payload(sample)
    optimizer = raw["optimizer_result"]
    assert isinstance(optimizer, dict)
    if defect == "zero-kkt-value":
        optimizer["kkt_solves"] = 0
        optimizer["regularization_candidates_tested"] = 0
    elif defect == "outside-ladder":
        optimizer["selected_regularization"] = 1.0e-9
    else:
        optimizer["regularization_candidates_tested"] = 0

    with pytest.raises(ValueError, match=message):
        validator._validate_sqp_raw_result(raw, sample, sample.source_identity)


def test_sqp_history_prefix_length_is_bound_to_iterations() -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=False)
    raw = _raw_payload(sample)
    optimizer = raw["optimizer_result"]
    assert isinstance(optimizer, dict)
    history = optimizer["history"]
    assert isinstance(history, dict)
    history["accepted_length"] = 0

    with pytest.raises(ValueError, match="history length differs from iterations"):
        validator._validate_sqp_raw_result(raw, sample, sample.source_identity)


def test_sqp_retained_endpoint_vector_is_bound_to_its_digest() -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=False)
    raw = _raw_payload(sample)
    endpoint = raw["endpoint"]
    assert isinstance(endpoint, dict)
    state = endpoint["physical_state"]
    assert isinstance(state, list)
    state[0] = 1.0

    with pytest.raises(ValueError, match="physical_state digest differs"):
        validator._validate_sqp_raw_result(raw, sample, sample.source_identity)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("parent_pid", 0, "parent PID"),
        ("child_pid", 0, "child PID"),
        ("child_start_time_ticks", 0, "start identity"),
        ("child_argv_sha256", "f" * 64, "argv identity"),
        ("peak_memory_bytes", 0, "peak bytes"),
        ("peak_memory_fraction", 0.8, "frozen budget"),
    ),
)
def test_sqp_memory_rejects_pid_child_identity_and_peak_tamper(
    field: str, replacement: JsonValue, message: str
) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=False)
    memory = _memory_payload(sample)
    memory[field] = replacement
    if field == "peak_memory_fraction":
        sample = replace(sample, peak_memory_fraction=0.8)

    with pytest.raises(ValueError, match=message):
        validator._validate_sqp_memory(memory, sample, sample.runtime_identity)


@pytest.mark.parametrize(
    "field",
    (
        "endpoint_sha256",
        "raw_result_sha256",
        "source_manifest_sha256",
        "runtime_evidence_sha256",
        "bootstrap_identity_sha256",
    ),
)
def test_sqp_certificate_rejects_every_provenance_binding_tamper(field: str) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)
    raw = _raw_payload(sample)
    certificate = _certificate_payload(sample, raw)
    assert validator._validate_sqp_certificate(
        certificate, sample, sample.source_identity, raw
    )

    certificate[field] = "f" * 64
    with pytest.raises(ValueError, match="differs"):
        validator._validate_sqp_certificate(
            certificate, sample, sample.source_identity, raw
        )


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("failed-check", "failed checks"),
        ("multiplier", "multipliers differ"),
        ("extra-scientific-key", "keys do not match"),
    ),
)
def test_sqp_certificate_rejects_scientific_payload_tamper(
    defect: str, message: str
) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)
    raw = _raw_payload(sample)
    certificate = _certificate_payload(sample, raw)
    scientific = certificate["scientific_certificate"]
    assert isinstance(scientific, dict)
    if defect == "failed-check":
        checks = scientific["checks"]
        assert isinstance(checks, dict)
        checks["branch"] = False
    elif defect == "multiplier":
        multipliers = scientific["multipliers"]
        assert isinstance(multipliers, list)
        multipliers[0] = 1.0
    else:
        scientific["unexpected"] = True

    with pytest.raises(ValueError, match=message):
        validator._validate_sqp_certificate(
            certificate, sample, sample.source_identity, raw
        )


@pytest.mark.parametrize("defect", ("single-term", "ledger-boolean"))
def test_sqp_certificate_rejects_objective_ledger_tamper(defect: str) -> None:
    sample = _sample(CompleteSample.COLD, promotion_eligible=True)
    raw = _raw_payload(sample)
    certificate = _certificate_payload(sample, raw)
    scientific = certificate["scientific_certificate"]
    assert isinstance(scientific, dict)
    pre = scientific["pre_projection"]
    assert isinstance(pre, dict)
    if defect == "single-term":
        raw_terms = pre["raw_objective_terms"]
        assert isinstance(raw_terms, dict)
        raw_terms["residual"] = 1.0
    else:
        pre["objective_ledger_consistent"] = False

    with pytest.raises(ValueError, match="objective ledger Boolean differs"):
        validator._validate_sqp_certificate(
            certificate, sample, sample.source_identity, raw
        )


def test_campaign_v2_parser_requires_exact_route_order_and_cold_reference() -> None:
    cold_ref = _artifact("samples/cold.json", SQP_SAMPLE_SCHEMA_VERSION)
    receipt = _campaign((cold_ref,))
    payload = campaign_receipt_v2_payload(receipt)

    parsed = campaign_receipt_v2_from_payload(payload)
    assert tuple(outcome.route for outcome in parsed.route_outcomes) == V2_ROUTES

    outcomes = payload["route_outcomes"]
    assert isinstance(outcomes, list)
    outcomes[0], outcomes[1] = outcomes[1], outcomes[0]
    with pytest.raises(ValueError, match="canonical route order"):
        campaign_receipt_v2_from_payload(payload)


def test_sqp_collector_writes_canonical_complete_speed_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_collector_gate_loader(monkeypatch)
    collector = _passed_gates(_collector(tmp_path))
    order = (
        CompleteSample.COLD,
        CompleteSample.WARM_1,
        CompleteSample.WARM_2,
        CompleteSample.WARM_3,
    )
    original = collector
    for sample_name in order:
        sample = _sample(sample_name, promotion_eligible=True)
        collector = collector.record_sample(
            _sample_artifact(f"collector/{sample_name}.json", sample),
            sample,
        )
    output = tmp_path / "campaign.json"

    reference = collector.write(output)
    payload = load_canonical_json_bytes(output.read_bytes())
    receipt = campaign_receipt_v2_from_payload(payload)

    assert original.samples == ()
    assert receipt.disposition is CampaignDisposition.ENGINEERING_SPEED_SUCCESS
    assert receipt.contract_sha256 == contract_sha256_v2()
    assert tuple(
        evidence.gate_id for evidence in receipt.route_outcomes[-1].gate_evidence
    ) == tuple(SqpGate)
    assert reference.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        collector.write(output)


def test_sqp_collector_enforces_gate_order_and_terminal_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_collector_gate_loader(monkeypatch)
    collector = _collector(tmp_path)
    with pytest.raises(ValueError, match="derivative, 1-step, 10-step order"):
        collector.record_gate(SqpGate.CANARY_1, _artifact("gates/early.json"))
    with pytest.raises(ValueError, match="passed derivative"):
        early_sample = _sample(CompleteSample.COLD, promotion_eligible=True)
        collector.record_sample(
            _sample_artifact("samples/early.json", early_sample),
            early_sample,
        )
    collector = collector.record_gate(
        SqpGate.DERIVATIVE, _artifact("gates/failed.json")
    )

    receipt = collector.finalize_receipt()

    assert receipt.disposition is CampaignDisposition.BOUNDED_NEGATIVE
    assert (
        receipt.route_outcomes[-1].disposition is RouteDisposition.NOT_SELECTED_BY_GATE
    )
    assert receipt.route_outcomes[-1].upstream_gate == SqpGate.DERIVATIVE
    with pytest.raises(ValueError, match="execution stopped"):
        collector.record_gate(SqpGate.CANARY_1, _artifact("gates/after-failure.json"))


def test_sqp_collector_rejects_sample_reference_not_bound_to_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_collector_gate_loader(monkeypatch)
    collector = _passed_gates(_collector(tmp_path))
    cold = _sample(CompleteSample.COLD, promotion_eligible=True)

    with pytest.raises(ValueError, match="differs from receipt bytes"):
        collector.record_sample(
            _artifact("samples/unbound.json", SQP_SAMPLE_SCHEMA_VERSION), cold
        )


def test_sqp_collector_requires_certified_cold_before_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_collector_gate_loader(monkeypatch)
    cold = _sample(CompleteSample.COLD, promotion_eligible=False)
    collector = _passed_gates(_collector(tmp_path)).record_sample(
        _sample_artifact("samples/cold-failed.json", cold),
        cold,
    )

    assert (
        collector.finalize_receipt().disposition is CampaignDisposition.BOUNDED_NEGATIVE
    )
    with pytest.raises(ValueError, match="failed or final result"):
        warm = _sample(CompleteSample.WARM_1, promotion_eligible=True)
        collector.record_sample(
            _sample_artifact("samples/warm-after-cold-failure.json", warm),
            warm,
        )


def test_sqp_collector_stops_after_slow_or_failed_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_collector_gate_loader(monkeypatch)
    cold = _sample(CompleteSample.COLD, promotion_eligible=True)
    base = _passed_gates(_collector(tmp_path)).record_sample(
        _sample_artifact("samples/cold.json", cold),
        cold,
    )
    slow = replace(
        _sample(CompleteSample.WARM_1, promotion_eligible=True),
        synchronized_solve_seconds=SQP_WARM_SOLVE_MAX_SECONDS,
        total_child_wall_seconds=300.0,
    )
    slow_collector = base.record_sample(
        _sample_artifact("samples/warm-1-slow.json", slow), slow
    )
    failed_warm = _sample(CompleteSample.WARM_1, promotion_eligible=False)
    failed_collector = base.record_sample(
        _sample_artifact("samples/warm-1-failed.json", failed_warm),
        failed_warm,
    )

    for collector in (slow_collector, failed_collector):
        assert (
            collector.finalize_receipt().disposition
            is CampaignDisposition.BOUNDED_NEGATIVE
        )
        with pytest.raises(ValueError, match="failed or final result"):
            warm_2 = _sample(CompleteSample.WARM_2, promotion_eligible=True)
            collector.record_sample(
                _sample_artifact("samples/warm-2-forbidden.json", warm_2),
                warm_2,
            )


def test_sqp_collector_refuses_to_finalize_nonterminal_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_collector_gate_loader(monkeypatch)
    cold = _sample(CompleteSample.COLD, promotion_eligible=True)
    collector = _passed_gates(_collector(tmp_path)).record_sample(
        _sample_artifact("samples/cold-only.json", cold),
        cold,
    )

    with pytest.raises(ValueError, match="not terminal"):
        collector.finalize_receipt()


@pytest.mark.parametrize(
    "defect",
    (
        "reordered",
        "cold-not-certified",
        "mixed-source",
        "mixed-device",
        "mixed-bootstrap",
    ),
)
def test_v2_sample_chain_rejects_replacement_order_and_mixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    samples = [
        _sample(CompleteSample.COLD, promotion_eligible=True),
        _sample(CompleteSample.WARM_1, promotion_eligible=True),
        _sample(CompleteSample.WARM_2, promotion_eligible=True),
        _sample(CompleteSample.WARM_3, promotion_eligible=True),
    ]
    if defect == "reordered":
        samples[1], samples[2] = samples[2], samples[1]
    elif defect == "cold-not-certified":
        samples[0] = _sample(CompleteSample.COLD, promotion_eligible=False)
    elif defect == "mixed-source":
        samples[2] = replace(samples[2], source_identity=_source("f"))
    elif defect == "mixed-device":
        samples[2] = replace(samples[2], runtime_identity=_runtime("GPU-other"))
    elif defect == "mixed-bootstrap":
        samples[2] = replace(samples[2], bootstrap_identity_sha256="f" * 64)
    references = tuple(
        _artifact(f"samples/{index}.json", SQP_SAMPLE_SCHEMA_VERSION)
        for index in range(4)
    )
    _write_sample_placeholders(tmp_path, references)
    by_name = dict(
        zip(
            (str(tmp_path / ref.relative_path) for ref in references),
            samples,
            strict=True,
        )
    )
    monkeypatch.setattr(
        validator,
        "_validate_sqp_sample",
        lambda path, _root: by_name[str(path)],
    )

    with pytest.raises(ValueError):
        validator._validate_sqp_sample_chain(_campaign(references), tmp_path)


def test_v2_sample_chain_accepts_one_failed_cold_or_exact_certified_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cold_failed = _sample(CompleteSample.COLD, promotion_eligible=False)
    complete = (
        _sample(CompleteSample.COLD, promotion_eligible=True),
        _sample(CompleteSample.WARM_1, promotion_eligible=True),
        _sample(CompleteSample.WARM_2, promotion_eligible=True),
        _sample(CompleteSample.WARM_3, promotion_eligible=True),
    )
    all_samples = (cold_failed, *complete)
    references = tuple(
        _artifact(f"samples/{index}.json", SQP_SAMPLE_SCHEMA_VERSION)
        for index in range(len(all_samples))
    )
    _write_sample_placeholders(tmp_path, references)
    by_name = dict(
        zip(
            (str(tmp_path / ref.relative_path) for ref in references),
            all_samples,
            strict=True,
        )
    )
    monkeypatch.setattr(
        validator,
        "_validate_sqp_sample",
        lambda path, _root: by_name[str(path)],
    )

    validator._validate_sqp_sample_chain(
        _campaign((references[0],), terminal_status="ITERATION_LIMIT"), tmp_path
    )
    validator._validate_sqp_sample_chain(
        replace(
            _campaign(references[1:]),
            disposition=CampaignDisposition.ENGINEERING_SPEED_SUCCESS,
        ),
        tmp_path,
    )


@pytest.mark.parametrize(
    "promotion_flags",
    ((True, False), (True, True, False), (True, True, True, False)),
)
def test_bounded_negative_retains_exact_failed_warm_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    promotion_flags: tuple[bool, ...],
) -> None:
    order = (
        CompleteSample.COLD,
        CompleteSample.WARM_1,
        CompleteSample.WARM_2,
        CompleteSample.WARM_3,
    )
    samples = tuple(
        _sample(sample, promotion_eligible=promotion)
        for sample, promotion in zip(order, promotion_flags, strict=False)
    )
    references = tuple(
        _artifact(f"prefix/{index}.json", SQP_SAMPLE_SCHEMA_VERSION)
        for index in range(len(samples))
    )
    _write_sample_placeholders(tmp_path, references)
    by_name = dict(
        zip(
            (str(tmp_path / reference.relative_path) for reference in references),
            samples,
            strict=True,
        )
    )
    monkeypatch.setattr(
        validator,
        "_validate_sqp_sample",
        lambda path, _root: by_name[str(path)],
    )

    validator._validate_sqp_sample_chain(_campaign(references), tmp_path)


def test_sample_chain_rejects_evidence_after_failed_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = (
        _sample(CompleteSample.COLD, promotion_eligible=True),
        _sample(CompleteSample.WARM_1, promotion_eligible=False),
        _sample(CompleteSample.WARM_2, promotion_eligible=True),
    )
    references = tuple(
        _artifact(f"after-failure/{index}.json", SQP_SAMPLE_SCHEMA_VERSION)
        for index in range(3)
    )
    _write_sample_placeholders(tmp_path, references)
    by_name = dict(
        zip(
            (str(tmp_path / reference.relative_path) for reference in references),
            samples,
            strict=True,
        )
    )
    monkeypatch.setattr(
        validator,
        "_validate_sqp_sample",
        lambda path, _root: by_name[str(path)],
    )

    with pytest.raises(ValueError, match="continued after"):
        validator._validate_sqp_sample_chain(_campaign(references), tmp_path)


def test_speed_success_requires_every_warm_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = (
        _sample(CompleteSample.COLD, promotion_eligible=True),
        _sample(CompleteSample.WARM_1, promotion_eligible=True),
        replace(
            _sample(CompleteSample.WARM_2, promotion_eligible=True),
            synchronized_solve_seconds=287.30421751597896,
            total_child_wall_seconds=300.0,
        ),
        _sample(CompleteSample.WARM_3, promotion_eligible=True),
    )
    references = tuple(
        _artifact(f"speed/{index}.json", SQP_SAMPLE_SCHEMA_VERSION)
        for index in range(4)
    )
    _write_sample_placeholders(tmp_path, references)
    by_name = dict(
        zip(
            (str(tmp_path / reference.relative_path) for reference in references),
            samples,
            strict=True,
        )
    )
    monkeypatch.setattr(
        validator,
        "_validate_sqp_sample",
        lambda path, _root: by_name[str(path)],
    )
    success = replace(
        _campaign(references),
        disposition=CampaignDisposition.ENGINEERING_SPEED_SUCCESS,
    )

    with pytest.raises(ValueError, match="losing warm"):
        validator._validate_sqp_sample_chain(success, tmp_path)


def test_v2_sample_chain_rejects_duplicate_or_replacement_reference(
    tmp_path: Path,
) -> None:
    cold = _artifact("samples/cold.json", SQP_SAMPLE_SCHEMA_VERSION)
    _write_sample_placeholders(tmp_path, (cold,))

    with pytest.raises(ValueError, match="replace or duplicate"):
        validator._validate_sqp_sample_chain(
            _campaign((cold, cold, cold, cold)), tmp_path
        )


def test_schema_dispatched_v2_campaign_accepts_gate_closure_and_rejects_v1_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _artifact(
        "gates/sqp-not-selected.json",
        "single-stage-fullspace-cfs-sqp1-derivative-gate-receipt-v1",
    )
    _write_sample_placeholders(tmp_path, (gate,))
    legacy_evidence = GateEvidence("SQP-PHASE0-GATE", gate)
    legacy_outcomes = tuple(
        RouteOutcome(
            route=route,
            disposition=RouteDisposition.NOT_SELECTED_BY_GATE,
            terminal_status=None,
            receipt=None,
            upstream_gate="SQP-PHASE0-GATE",
            gate_evidence=(legacy_evidence,),
        )
        for route in LEGACY_V1_ROUTES
    )
    sqp_evidence = GateEvidence(SqpGate.DERIVATIVE, gate)
    sqp_outcome = RouteOutcome(
        route=FullSpaceRoute.CFS_SQP1,
        disposition=RouteDisposition.NOT_SELECTED_BY_GATE,
        terminal_status=None,
        receipt=None,
        upstream_gate=SqpGate.DERIVATIVE,
        gate_evidence=(sqp_evidence,),
    )
    monkeypatch.setattr(
        validator,
        "load_sqp_gate_result",
        lambda _root, requested_gate, artifact: SqpGateResult(
            requested_gate, artifact, False, ("FIXTURE_GATE_FAILURE",)
        ),
    )
    receipt = CampaignReceiptV2(
        disposition=CampaignDisposition.BOUNDED_NEGATIVE,
        baseline_classification=BaselineClassification.HISTORICAL_ENGINEERING_ONLY,
        contract_sha256=contract_sha256_v2(),
        route_outcomes=(*legacy_outcomes, sqp_outcome),
        sqp_samples=(),
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_bytes(
        canonical_json_bytes(campaign_receipt_v2_payload(receipt))
    )

    validator.validate_campaign(tmp_path)

    wrong = replace(receipt, contract_sha256=contract_sha256_v1())
    campaign_path.write_bytes(canonical_json_bytes(campaign_receipt_v2_payload(wrong)))
    with pytest.raises(ValueError, match="frozen contract"):
        validator.validate_campaign(tmp_path)
