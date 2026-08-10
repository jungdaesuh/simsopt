"""Fail-closed validator for single-stage full-space campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simsopt_jax.solve.fullspace import FullSpaceRoute
from simsopt_jax.solve.fullspace_certificate import (
    CFS_SQP1_CERTIFICATE_SCHEMA_VERSION,
)

from benchmarks.single_stage_fullspace_bootstrap import validate_bootstrap_artifact
from benchmarks.single_stage_fullspace_ftr_receipt import (
    gate_artifact_from_path,
    load_ftr_gate_result,
)
from benchmarks.single_stage_fullspace_receipt import (
    SCHEMA_VERSION,
    SQP_BUDGET_SHA256,
    SQP_CERTIFICATE_ENVELOPE_SCHEMA_VERSION,
    SQP_KKT_FORWARD_ERROR_MAXIMUM,
    SQP_KKT_RELATIVE_RESIDUAL_MAXIMUM,
    SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM,
    SQP_MAXIMUM_MEMORY_FRACTION,
    SQP_MEMORY_SCHEMA_VERSION,
    SQP_PLAN_SHA256,
    SQP_R1_BUDGET_SHA256,
    SQP_R1_CONTRACT_SHA256,
    SQP_R1_PLAN_SHA256,
    SQP_R2_BUDGET_SHA256,
    SQP_R2_PLAN_SHA256,
    SQP_RESULT_SCHEMA_VERSION,
    SQP_SCHUR_RELATIVE_RESIDUAL_MAXIMUM,
    SQP_WARM_SOLVE_MAX_SECONDS,
    CampaignDisposition,
    CampaignReceipt,
    CampaignReceiptV2,
    CompleteSample,
    JsonValue,
    RouteDisposition,
    RunPhase,
    SqpGate,
    SqpSampleReceipt,
    artifact_ref_from_payload,
    bootstrap_identity_sha256_from_payload,
    campaign_receipt_from_payload_dispatch,
    canonical_json_bytes,
    contract_sha256_v1,
    contract_sha256_v2,
    expect_exact_keys,
    expect_mapping,
    expect_string,
    load_canonical_json_bytes,
    load_sqp_gate_result,
    run_request_from_payload,
    run_request_v2_from_payload,
    sqp_sample_receipt_from_payload,
    validate_sqp_convergence_telemetry,
)
from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    RuntimeIdentity,
    SourceIdentity,
    runtime_identity_from_payload,
    source_identity_from_payload,
    validate_runtime_evidence,
)

_RUN_KEYS = frozenset(
    (
        "schema_version",
        "contract_sha256",
        "request",
        "source_identity",
        "runtime_identity",
        "runtime_evidence",
        "terminal_status",
        "timing",
        "trajectory_equivalence_required",
        "transfer_audit",
        "endpoint_certificate",
    )
)
_SQP_RAW_KEYS = frozenset(
    (
        "schema_version",
        "contract_sha256",
        "plan_sha256",
        "budget_sha256",
        "request",
        "source_identity",
        "runtime_evidence",
        "bootstrap_artifact",
        "optimizer_result",
        "endpoint",
        "timing",
        "transfer_audit",
        "endpoint_certificate",
        "promotion_eligible",
        "trajectory_equivalence_required",
        "terminal_status",
    )
)
_SQP_OPTIMIZER_KEYS = frozenset(
    (
        "status",
        "fatal",
        "failed",
        "converged",
        "all_finite",
        "all_accepted_states_finite",
        "iterations",
        "joint_evaluations",
        "derivative_builds",
        "kkt_solves",
        "line_search_evaluations",
        "rejected_nonfinite_trials",
        "bfgs_resets",
        "regularization_uses",
        "regularization_candidates_tested",
        "final_kkt_relative_residual",
        "final_kkt_reciprocal_condition",
        "final_kkt_solution_scaled_residual",
        "final_schur_relative_residual",
        "final_bfgs_cholesky_relative_pivot",
        "final_schur_cholesky_relative_pivot",
        "selected_regularization",
        "merit_penalty",
        "initial_physical_objective",
        "initial_scaled_constraint_infinity_norm",
        "initial_raw_kkt_stationarity_infinity_norm",
        "physical_objective",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
        "physical_state_sha256",
        "optimizer_coordinates_sha256",
        "scaled_multipliers_sha256",
        "raw_multipliers_sha256",
        "history",
        "history_sha256",
    )
)
_SQP_R3_OPTIMIZER_KEYS = _SQP_OPTIMIZER_KEYS | {
    "convergence_telemetry",
    "restoration_numerical_failures",
}
_SQP_HISTORY_KEYS = frozenset(
    (
        "accepted_length",
        "objective",
        "feasibility_infinity_norm",
        "stationarity_infinity_norm",
        "step_length",
        "kkt_relative_residual",
        "status",
    )
)
_SQP_ENDPOINT_KEYS = frozenset(
    (
        "physical_state",
        "physical_state_sha256",
        "optimizer_coordinates",
        "optimizer_coordinates_sha256",
        "scaled_multipliers",
        "scaled_multipliers_sha256",
        "raw_multipliers",
        "raw_multipliers_sha256",
        "physical_objective",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
        "all_finite",
    )
)
_SQP_MEMORY_KEYS = frozenset(
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
_SQP_CERTIFICATE_KEYS = frozenset(
    (
        "schema_version",
        "scientific_certificate",
        "endpoint_sha256",
        "raw_result_sha256",
        "source_manifest_sha256",
        "runtime_evidence_sha256",
        "bootstrap_identity_sha256",
    )
)
_SQP_SCIENTIFIC_CERTIFICATE_KEYS = frozenset(
    (
        "schema_version",
        "route",
        "termination",
        "pre_projection",
        "post_projection",
        "multipliers",
        "inactive_hardware",
        "objective_reference",
        "cross_evaluator",
        "field_line",
        "branch",
        "projection",
        "checks",
        "certified",
    )
)
_SQP_CERTIFICATE_ENDPOINT_KEYS = frozenset(
    (
        "state",
        "objective",
        "raw_objective_terms",
        "objective_ledger_consistent",
        "constraints",
        "scaled_constraints",
        "objective_gradient",
        "stationarity_gradient",
        "boozer_residual_infinity_norm",
        "volume_residual_absolute",
        "scaled_feasibility_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
        "iota",
        "major_radius",
        "one_sided_length_penalty",
        "all_finite_fp64",
    )
)
_SQP_CERTIFICATE_CHECK_KEYS = frozenset(
    (
        "optimizer_termination",
        "solver_result_consistent",
        "finite_fp64",
        "objective_ledger_consistent",
        "scaled_feasibility",
        "raw_kkt_stationarity",
        "fixed_state_preserved",
        "inactive_hardware_terms_valid",
        "objective_threshold",
        "objective_reference_valid",
        "cross_evaluator",
        "field_line",
        "branch",
        "projection_bound_to_solver_endpoint",
        "projection_immaterial",
        "pre_projection_certifiable",
        "post_projection_certifiable",
    )
)
_SQP_REGULARIZATION_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)


def _validate_executed_run(
    path: Path, expected_route: FullSpaceRoute, campaign_root: Path
) -> None:
    payload = load_canonical_json_bytes(path.read_bytes())
    run = expect_mapping(payload, context="run receipt")
    expect_exact_keys(run, _RUN_KEYS, context="run receipt")
    if expect_string(run["schema_version"], context="schema_version") != SCHEMA_VERSION:
        raise ValueError("run receipt schema mismatch")
    if run["contract_sha256"] != contract_sha256_v1():
        raise ValueError("run receipt contract digest does not match live contract")
    if run["trajectory_equivalence_required"] is not False:
        raise ValueError("fullspace run may not require trajectory equivalence")
    request = run_request_from_payload(run["request"])
    if request.route is not expected_route:
        raise ValueError("executed receipt route mismatch")
    source = source_identity_from_payload(run["source_identity"])
    runtime = runtime_identity_from_payload(run["runtime_identity"])
    if runtime.backend != "gpu":
        raise ValueError("executed run is not GPU-backed")
    manifest_path = source.snapshot_manifest.resolve_and_validate(campaign_root)
    runtime_ref = artifact_ref_from_payload(run["runtime_evidence"])
    runtime_path = runtime_ref.resolve_and_validate(campaign_root)
    evidence = validate_runtime_evidence(
        runtime_path,
        snapshot_root=manifest_path.parent,
        campaign_root=campaign_root,
    )
    if evidence.source_identity != source:
        raise ValueError("run source identity differs from runtime evidence")
    if evidence.observation.runtime_identity != runtime:
        raise ValueError("run runtime identity differs from runtime evidence")
    if run["terminal_status"] in (None, "PARTIAL", "RUNNING"):
        raise ValueError("executed run is partial")
    endpoint = expect_mapping(
        run["endpoint_certificate"], context="endpoint_certificate"
    )
    if endpoint.get("certified") is not True:
        raise ValueError("executed run lacks endpoint certification")


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a Boolean")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a nonnegative integer")
    return value


def _finite_nonnegative(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return result


def _optional_finite_nonnegative(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, context)


def _finite_vector(value: JsonValue, size: int, context: str) -> list[JsonValue]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{context} must be a length-{size} vector")
    for index, item in enumerate(value):
        _finite_number(item, f"{context}[{index}]")
    return value


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _sha256_string(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _validate_sqp_optimizer_result(
    value: JsonValue,
    sample: SqpSampleReceipt,
    *,
    require_convergence_telemetry: bool,
) -> None:
    optimizer = expect_mapping(value, context="SQP optimizer result")
    expected_keys = (
        _SQP_R3_OPTIMIZER_KEYS if require_convergence_telemetry else _SQP_OPTIMIZER_KEYS
    )
    expect_exact_keys(optimizer, expected_keys, context="SQP optimizer result")
    status = expect_string(optimizer["status"], context="optimizer.status")
    fatal = _boolean(optimizer["fatal"], "optimizer.fatal")
    failed = _boolean(optimizer["failed"], "optimizer.failed")
    converged = _boolean(optimizer["converged"], "optimizer.converged")
    all_finite = _boolean(optimizer["all_finite"], "optimizer.all_finite")
    accepted_finite = _boolean(
        optimizer["all_accepted_states_finite"],
        "optimizer.all_accepted_states_finite",
    )
    counters: dict[str, int] = {}
    for key in (
        "iterations",
        "joint_evaluations",
        "derivative_builds",
        "kkt_solves",
        "line_search_evaluations",
        "rejected_nonfinite_trials",
        "bfgs_resets",
        "regularization_uses",
        "regularization_candidates_tested",
    ):
        counters[key] = _nonnegative_integer(optimizer[key], f"optimizer.{key}")
    diagnostics: dict[str, float | None] = {}
    for key in (
        "final_kkt_relative_residual",
        "final_kkt_reciprocal_condition",
        "final_kkt_solution_scaled_residual",
        "final_schur_relative_residual",
        "final_bfgs_cholesky_relative_pivot",
        "final_schur_cholesky_relative_pivot",
        "selected_regularization",
    ):
        diagnostics[key] = _optional_finite_nonnegative(
            optimizer[key], f"optimizer.{key}"
        )
    for key in (
        "merit_penalty",
        "initial_physical_objective",
        "initial_scaled_constraint_infinity_norm",
        "initial_raw_kkt_stationarity_infinity_norm",
        "physical_objective",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
    ):
        _finite_nonnegative(optimizer[key], f"optimizer.{key}")
    selected_regularization = diagnostics["selected_regularization"]
    if counters["kkt_solves"] == 0:
        if selected_regularization is not None:
            raise ValueError("zero-KKT SQP result requires null regularization")
        if (
            counters["regularization_candidates_tested"] != 0
            or counters["regularization_uses"] != 0
        ):
            raise ValueError("zero-KKT SQP regularization counters are inconsistent")
    else:
        if selected_regularization not in _SQP_REGULARIZATION_LADDER:
            raise ValueError("SQP selected regularization is not in the frozen ladder")
        if (
            counters["regularization_candidates_tested"] < counters["kkt_solves"]
            or counters["regularization_uses"] > counters["kkt_solves"]
        ):
            raise ValueError("SQP regularization counters are inconsistent")
    if status == "CONVERGED" and counters["kkt_solves"] > 0:
        if any(value is None for value in diagnostics.values()):
            raise ValueError("converged SQP result requires finite KKT diagnostics")
        reciprocal_condition = diagnostics["final_kkt_reciprocal_condition"]
        solution_residual = diagnostics["final_kkt_solution_scaled_residual"]
        kkt_relative_residual = diagnostics["final_kkt_relative_residual"]
        schur_relative_residual = diagnostics["final_schur_relative_residual"]
        assert reciprocal_condition is not None
        assert solution_residual is not None
        assert kkt_relative_residual is not None
        assert schur_relative_residual is not None
        if reciprocal_condition <= solution_residual:
            raise ValueError("SQP KKT reciprocal condition does not exceed residual")
        if solution_residual > SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM:
            raise ValueError("SQP KKT solution-scaled residual exceeds the gate")
        if kkt_relative_residual > SQP_KKT_RELATIVE_RESIDUAL_MAXIMUM:
            raise ValueError("SQP KKT relative residual exceeds the gate")
        if schur_relative_residual > SQP_SCHUR_RELATIVE_RESIDUAL_MAXIMUM:
            raise ValueError("SQP Schur relative residual exceeds the gate")
        forward_error_bound = solution_residual / (
            reciprocal_condition - solution_residual
        )
        if forward_error_bound >= SQP_KKT_FORWARD_ERROR_MAXIMUM:
            raise ValueError("SQP KKT forward-error bound exceeds the gate")
    for key in (
        "physical_state_sha256",
        "optimizer_coordinates_sha256",
        "scaled_multipliers_sha256",
        "raw_multipliers_sha256",
    ):
        _sha256_string(optimizer[key], f"optimizer.{key}")
    history = expect_mapping(optimizer["history"], context="optimizer.history")
    expect_exact_keys(history, _SQP_HISTORY_KEYS, context="optimizer.history")
    accepted_length = _nonnegative_integer(
        history["accepted_length"], "optimizer.history.accepted_length"
    )
    if accepted_length != counters["iterations"] or accepted_length > 100:
        raise ValueError("SQP accepted history length differs from iterations")
    for key in _SQP_HISTORY_KEYS - {"accepted_length"}:
        values = history[key]
        if not isinstance(values, list) or len(values) != accepted_length:
            raise ValueError(
                f"optimizer.history.{key} must match accepted history length"
            )
        for index, item in enumerate(values):
            if key == "status":
                _nonnegative_integer(item, f"optimizer.history.{key}[{index}]")
            else:
                _finite_nonnegative(item, f"optimizer.history.{key}[{index}]")
    expected_history_sha = hashlib.sha256(canonical_json_bytes(history)).hexdigest()
    if (
        _sha256_string(optimizer["history_sha256"], "optimizer.history_sha256")
        != expected_history_sha
    ):
        raise ValueError("optimizer history digest differs from inline history")
    if counters["joint_evaluations"] < counters["iterations"]:
        raise ValueError("SQP joint-evaluation counter is inconsistent")
    if require_convergence_telemetry:
        validate_sqp_convergence_telemetry(
            optimizer,
            accepted_length=accepted_length,
        )
    if status != sample.terminal_status:
        raise ValueError("SQP optimizer status differs from sample receipt")
    if converged and (fatal or failed):
        raise ValueError("SQP optimizer cannot be converged and failed")
    if sample.promotion_eligible and not (
        converged and not fatal and not failed and all_finite and accepted_finite
    ):
        raise ValueError("promoting SQP sample has inconsistent optimizer state")


def _validate_sqp_raw_result(
    raw: dict[str, JsonValue], sample: SqpSampleReceipt, source: SourceIdentity
) -> None:
    expect_exact_keys(raw, _SQP_RAW_KEYS, context="SQP raw result")
    if raw["schema_version"] != SQP_RESULT_SCHEMA_VERSION:
        raise ValueError("SQP raw result schema mismatch")
    if raw["contract_sha256"] not in (
        contract_sha256_v2(),
        SQP_R1_CONTRACT_SHA256,
    ):
        raise ValueError("SQP raw result contract digest mismatch")
    current_identity = (
        raw["plan_sha256"] == SQP_PLAN_SHA256
        and raw["budget_sha256"] == SQP_BUDGET_SHA256
    )
    revision2_identity = (
        raw["plan_sha256"] == SQP_R2_PLAN_SHA256
        and raw["budget_sha256"] == SQP_R2_BUDGET_SHA256
    )
    revision1_identity = (
        raw["plan_sha256"] == SQP_R1_PLAN_SHA256
        and raw["budget_sha256"] == SQP_R1_BUDGET_SHA256
        and raw["contract_sha256"] == SQP_R1_CONTRACT_SHA256
    )
    if not (current_identity or revision2_identity or revision1_identity):
        valid_plans = (SQP_PLAN_SHA256, SQP_R2_PLAN_SHA256, SQP_R1_PLAN_SHA256)
        if raw["plan_sha256"] not in valid_plans:
            raise ValueError("SQP raw result plan digest mismatch")
        raise ValueError("SQP raw result budget digest mismatch")
    request = run_request_v2_from_payload(raw["request"])
    if request != sample.request:
        raise ValueError("SQP raw result request differs from sample receipt")
    if source_identity_from_payload(raw["source_identity"]) != source:
        raise ValueError("SQP raw result source identity differs")
    if artifact_ref_from_payload(raw["runtime_evidence"]) != sample.runtime_evidence:
        raise ValueError("SQP raw result runtime evidence differs")
    if (
        artifact_ref_from_payload(raw["bootstrap_artifact"])
        != sample.bootstrap_artifact
    ):
        raise ValueError("SQP raw result bootstrap artifact differs")
    if raw["promotion_eligible"] is not False:
        raise ValueError("SQP raw solver output must be non-promoting")
    if raw["endpoint_certificate"] is not None:
        raise ValueError("SQP raw solver output may not embed endpoint certification")
    if raw["trajectory_equivalence_required"] is not False:
        raise ValueError("SQP raw result may not require trajectory equivalence")
    if raw["terminal_status"] != sample.terminal_status:
        raise ValueError("SQP sample terminal status differs from raw result")
    _validate_sqp_optimizer_result(
        raw["optimizer_result"],
        sample,
        require_convergence_telemetry=(
            current_identity
            and request.phase is RunPhase.CANARY
            and request.steps == 10
        ),
    )
    optimizer = expect_mapping(raw["optimizer_result"], context="SQP optimizer result")
    endpoint = expect_mapping(raw["endpoint"], context="SQP raw endpoint")
    expect_exact_keys(endpoint, _SQP_ENDPOINT_KEYS, context="SQP raw endpoint")
    vector_bindings = (
        ("physical_state", "physical_state_sha256", 716),
        ("optimizer_coordinates", "optimizer_coordinates_sha256", 716),
        ("scaled_multipliers", "scaled_multipliers_sha256", 255),
        ("raw_multipliers", "raw_multipliers_sha256", 255),
    )
    for vector_key, digest_key, size in vector_bindings:
        vector = _finite_vector(endpoint[vector_key], size, f"endpoint.{vector_key}")
        digest = hashlib.sha256(canonical_json_bytes(vector)).hexdigest()
        if (
            _sha256_string(endpoint[digest_key], f"endpoint.{digest_key}") != digest
            or optimizer[digest_key] != digest
        ):
            raise ValueError(f"SQP endpoint {vector_key} digest differs")
    for key in (
        "physical_objective",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
    ):
        if _finite_nonnegative(endpoint[key], f"endpoint.{key}") != optimizer[key]:
            raise ValueError(f"SQP endpoint {key} differs from optimizer result")
    if (
        _boolean(endpoint["all_finite"], "endpoint.all_finite")
        != optimizer["all_finite"]
    ):
        raise ValueError("SQP endpoint finiteness differs from optimizer result")
    timing = expect_mapping(raw["timing"], context="SQP raw timing")
    solve_seconds = _finite_nonnegative(
        timing.get("synchronized_solve_seconds"),
        "timing.synchronized_solve_seconds",
    )
    child_seconds = _finite_nonnegative(
        timing.get("total_child_wall_seconds"), "timing.total_child_wall_seconds"
    )
    if (
        solve_seconds != sample.synchronized_solve_seconds
        or child_seconds != sample.total_child_wall_seconds
    ):
        raise ValueError("SQP sample timing differs from raw result")
    transfers = expect_mapping(raw["transfer_audit"], context="SQP transfer audit")
    transfer_bindings = (
        ("hot_h2d_calls", sample.hot_h2d_transfers),
        ("hot_d2h_calls", sample.hot_d2h_transfers),
        ("initial_h2d_calls", sample.initial_h2d_transfers),
        ("final_d2h_calls", sample.final_d2h_transfers),
    )
    for key, expected in transfer_bindings:
        if (
            _nonnegative_integer(transfers.get(key), f"transfer_audit.{key}")
            != expected
        ):
            raise ValueError("SQP sample transfer counts differ from raw result")


def _validate_sqp_memory(
    memory: dict[str, JsonValue],
    sample: SqpSampleReceipt,
    runtime: RuntimeIdentity,
) -> None:
    expect_exact_keys(memory, _SQP_MEMORY_KEYS, context="SQP GPU memory")
    if memory["schema_version"] != SQP_MEMORY_SCHEMA_VERSION:
        raise ValueError("SQP GPU memory schema mismatch")
    if memory["monitor_scope"] != "whole-child-exact-pid-exact-device":
        raise ValueError("SQP GPU memory scope is invalid")
    if _nonnegative_integer(memory["parent_pid"], "memory.parent_pid") == 0:
        raise ValueError("SQP GPU memory parent PID must be positive")
    if _nonnegative_integer(memory["child_pid"], "memory.child_pid") == 0:
        raise ValueError("SQP GPU memory child PID must be positive")
    if (
        _nonnegative_integer(
            memory["child_start_time_ticks"], "memory.child_start_time_ticks"
        )
        == 0
    ):
        raise ValueError("SQP GPU memory child start identity must be positive")
    if (
        _sha256_string(memory["child_argv_sha256"], "memory.child_argv_sha256")
        != hashlib.sha256(canonical_json_bytes(list(runtime.argv))).hexdigest()
    ):
        raise ValueError("SQP GPU memory child argv identity differs from runtime")
    if _nonnegative_integer(memory["sample_count"], "memory.sample_count") == 0:
        raise ValueError("SQP GPU memory requires at least one sample")
    if memory["device_uuid"] != runtime.device_uuid:
        raise ValueError("SQP GPU memory device differs from runtime")
    memory_bytes = _nonnegative_integer(
        memory["peak_memory_bytes"], "memory.peak_memory_bytes"
    )
    if memory_bytes == 0:
        raise ValueError("SQP GPU memory peak bytes must be positive")
    memory_fraction = _finite_nonnegative(
        memory["peak_memory_fraction"], "memory.peak_memory_fraction"
    )
    if (
        memory_bytes != sample.peak_memory_bytes
        or memory_fraction != sample.peak_memory_fraction
    ):
        raise ValueError("SQP sample peak memory differs from monitor artifact")
    if memory_fraction >= SQP_MAXIMUM_MEMORY_FRACTION:
        raise ValueError("SQP sample peak memory exceeds the frozen budget")


def _validate_certificate_endpoint(
    value: JsonValue, context: str
) -> dict[str, JsonValue]:
    endpoint = expect_mapping(value, context=context)
    expect_exact_keys(endpoint, _SQP_CERTIFICATE_ENDPOINT_KEYS, context=context)
    _finite_vector(endpoint["state"], 716, f"{context}.state")
    _finite_vector(endpoint["constraints"], 255, f"{context}.constraints")
    _finite_vector(endpoint["scaled_constraints"], 255, f"{context}.scaled_constraints")
    _finite_vector(endpoint["objective_gradient"], 716, f"{context}.objective_gradient")
    _finite_vector(
        endpoint["stationarity_gradient"],
        716,
        f"{context}.stationarity_gradient",
    )
    objective = _finite_number(endpoint["objective"], f"{context}.objective")
    raw_terms = expect_mapping(
        endpoint["raw_objective_terms"], context=f"{context}.raw_objective_terms"
    )
    expect_exact_keys(
        raw_terms,
        frozenset(("non_qs", "residual", "iota", "major_radius", "length")),
        context=f"{context}.raw_objective_terms",
    )
    term_values = tuple(
        _finite_nonnegative(raw_terms[key], f"{context}.raw_objective_terms.{key}")
        for key in ("non_qs", "residual", "iota", "major_radius", "length")
    )
    recomposed_objective = (
        term_values[0]
        + term_values[1]
        + term_values[2]
        + term_values[3]
        + term_values[4]
    )
    ledger_consistent = _boolean(
        endpoint["objective_ledger_consistent"],
        f"{context}.objective_ledger_consistent",
    )
    if ledger_consistent != (recomposed_objective == objective):
        raise ValueError(f"{context} objective ledger Boolean differs from raw terms")
    for key in (
        "boozer_residual_infinity_norm",
        "volume_residual_absolute",
        "scaled_feasibility_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
        "iota",
        "major_radius",
        "one_sided_length_penalty",
    ):
        _finite_number(endpoint[key], f"{context}.{key}")
    if term_values[4] != endpoint["one_sided_length_penalty"]:
        raise ValueError(f"{context} length term differs from length penalty")
    _boolean(endpoint["all_finite_fp64"], f"{context}.all_finite_fp64")
    return endpoint


def _validate_sqp_scientific_certificate(
    certificate: dict[str, JsonValue], raw: dict[str, JsonValue]
) -> bool:
    expect_exact_keys(
        certificate,
        _SQP_SCIENTIFIC_CERTIFICATE_KEYS,
        context="SQP scientific certificate",
    )
    if certificate["schema_version"] != CFS_SQP1_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("SQP scientific certificate schema mismatch")
    if certificate["route"] != FullSpaceRoute.CFS_SQP1:
        raise ValueError("SQP scientific certificate route mismatch")
    termination = expect_string(
        certificate["termination"], context="certificate.termination"
    )
    if termination not in (
        "CONVERGED",
        "INCOMPLETE",
        "NONFINITE",
        "OPTIMIZER_REJECTED",
    ):
        raise ValueError("SQP endpoint certificate termination is not normalized")
    pre = _validate_certificate_endpoint(
        certificate["pre_projection"], "certificate.pre_projection"
    )
    post = _validate_certificate_endpoint(
        certificate["post_projection"], "certificate.post_projection"
    )
    multipliers = _finite_vector(
        certificate["multipliers"], 255, "certificate.multipliers"
    )
    raw_endpoint = expect_mapping(raw["endpoint"], context="SQP raw endpoint")
    if pre["state"] != raw_endpoint["physical_state"]:
        raise ValueError(
            "SQP certificate pre-projection state differs from raw endpoint"
        )
    if multipliers != raw_endpoint["raw_multipliers"]:
        raise ValueError("SQP certificate multipliers differ from raw endpoint")

    inactive = expect_mapping(
        certificate["inactive_hardware"], context="certificate.inactive_hardware"
    )
    expect_exact_keys(
        inactive,
        frozenset(("names", "metrics", "weights")),
        context="certificate.inactive_hardware",
    )
    if inactive["names"] != [
        "curvature",
        "curve_curve",
        "curve_surface",
        "surface_vessel",
    ]:
        raise ValueError("SQP certificate inactive hardware names differ")
    _finite_vector(inactive["metrics"], 4, "certificate.inactive_hardware.metrics")
    _finite_vector(inactive["weights"], 4, "certificate.inactive_hardware.weights")

    objective_reference = expect_mapping(
        certificate["objective_reference"],
        context="certificate.objective_reference",
    )
    expect_exact_keys(
        objective_reference,
        frozenset(("native_reference_objective",)),
        context="certificate.objective_reference",
    )
    _finite_number(
        objective_reference["native_reference_objective"],
        "certificate.objective_reference.native_reference_objective",
    )

    cross = expect_mapping(
        certificate["cross_evaluator"], context="certificate.cross_evaluator"
    )
    expect_exact_keys(
        cross,
        frozenset(
            (
                "performed",
                "native_on_jax_endpoint_objective",
                "jax_on_native_endpoint_objective",
            )
        ),
        context="certificate.cross_evaluator",
    )
    _boolean(cross["performed"], "certificate.cross_evaluator.performed")
    _finite_number(
        cross["native_on_jax_endpoint_objective"],
        "certificate.cross_evaluator.native_on_jax_endpoint_objective",
    )
    _finite_number(
        cross["jax_on_native_endpoint_objective"],
        "certificate.cross_evaluator.jax_on_native_endpoint_objective",
    )

    field_line = expect_mapping(
        certificate["field_line"], context="certificate.field_line"
    )
    expect_exact_keys(
        field_line,
        frozenset(("performed", "poincare_closed", "traced_iota")),
        context="certificate.field_line",
    )
    _boolean(field_line["performed"], "certificate.field_line.performed")
    _boolean(field_line["poincare_closed"], "certificate.field_line.poincare_closed")
    _finite_number(field_line["traced_iota"], "certificate.field_line.traced_iota")

    branch = expect_mapping(certificate["branch"], context="certificate.branch")
    expect_exact_keys(
        branch,
        frozenset(
            (
                "performed",
                "exact_solve_succeeded",
                "material_branch_switch",
                "reproduced_state_infinity_difference",
                "basin_classification",
            )
        ),
        context="certificate.branch",
    )
    for key in ("performed", "exact_solve_succeeded", "material_branch_switch"):
        _boolean(branch[key], f"certificate.branch.{key}")
    _finite_number(
        branch["reproduced_state_infinity_difference"],
        "certificate.branch.reproduced_state_infinity_difference",
    )
    expect_string(
        branch["basin_classification"],
        context="certificate.branch.basin_classification",
    )

    projection = expect_mapping(
        certificate["projection"], context="certificate.projection"
    )
    expect_exact_keys(
        projection,
        frozenset(("evaluated", "used", "pre_state", "post_state")),
        context="certificate.projection",
    )
    _boolean(projection["evaluated"], "certificate.projection.evaluated")
    _boolean(projection["used"], "certificate.projection.used")
    _finite_vector(projection["pre_state"], 716, "certificate.projection.pre_state")
    _finite_vector(projection["post_state"], 716, "certificate.projection.post_state")
    if (
        projection["pre_state"] != pre["state"]
        or projection["post_state"] != post["state"]
    ):
        raise ValueError(
            "SQP certificate projection states differ from endpoint evidence"
        )

    checks = expect_mapping(certificate["checks"], context="certificate.checks")
    expect_exact_keys(checks, _SQP_CERTIFICATE_CHECK_KEYS, context="certificate.checks")
    check_values = tuple(
        _boolean(checks[key], f"certificate.checks.{key}")
        for key in _SQP_CERTIFICATE_CHECK_KEYS
    )
    certified = _boolean(certificate["certified"], "certificate.certified")
    expected_ledger_check = bool(
        pre["objective_ledger_consistent"] and post["objective_ledger_consistent"]
    )
    if checks["objective_ledger_consistent"] is not expected_ledger_check:
        raise ValueError(
            "SQP certificate objective ledger check differs from endpoints"
        )
    if certified and (termination != "CONVERGED" or not all(check_values)):
        raise ValueError("certified SQP scientific certificate has failed checks")
    return certified


def _validate_sqp_certificate(
    certificate: dict[str, JsonValue],
    sample: SqpSampleReceipt,
    source: SourceIdentity,
    raw: dict[str, JsonValue],
) -> bool:
    expect_exact_keys(
        certificate,
        _SQP_CERTIFICATE_KEYS,
        context="SQP endpoint certificate",
    )
    if certificate["schema_version"] != SQP_CERTIFICATE_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("SQP endpoint certificate schema mismatch")
    scientific = expect_mapping(
        certificate["scientific_certificate"],
        context="SQP scientific certificate",
    )
    certified = _validate_sqp_scientific_certificate(scientific, raw)
    bindings = (
        (
            "endpoint_sha256",
            hashlib.sha256(canonical_json_bytes(raw["endpoint"])).hexdigest(),
        ),
        ("raw_result_sha256", sample.raw_result.sha256),
        ("source_manifest_sha256", source.snapshot_manifest.sha256),
        ("runtime_evidence_sha256", sample.runtime_evidence.sha256),
        ("bootstrap_identity_sha256", sample.bootstrap_identity_sha256),
    )
    for key, expected in bindings:
        if _sha256_string(certificate[key], f"certificate.{key}") != expected:
            raise ValueError(f"SQP endpoint certificate {key} differs")
    return certified


def _validate_sqp_sample(path: Path, campaign_root: Path) -> SqpSampleReceipt:
    payload = load_canonical_json_bytes(path.read_bytes())
    sample = sqp_sample_receipt_from_payload(payload)
    source = sample.source_identity
    runtime = sample.runtime_identity
    if runtime.backend != "gpu":
        raise ValueError("SQP sample is not GPU-backed")
    manifest_path = source.snapshot_manifest.resolve_and_validate(campaign_root)
    runtime_path = sample.runtime_evidence.resolve_and_validate(campaign_root)
    evidence = validate_runtime_evidence(
        runtime_path,
        snapshot_root=manifest_path.parent,
        campaign_root=campaign_root,
    )
    if evidence.source_identity != source:
        raise ValueError("SQP sample source identity differs from runtime evidence")
    if evidence.observation.runtime_identity != runtime:
        raise ValueError("SQP sample runtime identity differs from runtime evidence")

    bootstrap = validate_bootstrap_artifact(
        sample.bootstrap_artifact.resolve_and_validate(campaign_root),
        campaign_root=campaign_root,
        snapshot_root=manifest_path.parent,
    )
    if (
        source_identity_from_payload(bootstrap["source_identity"]) != source
        or runtime_identity_from_payload(bootstrap["runtime_identity"]) != runtime
    ):
        raise ValueError("SQP sample mixes bootstrap and sample provenance")
    if (
        bootstrap_identity_sha256_from_payload(bootstrap)
        != sample.bootstrap_identity_sha256
    ):
        raise ValueError("SQP sample bootstrap identity differs")
    raw_path = sample.raw_result.resolve_and_validate(campaign_root)
    raw = expect_mapping(
        load_canonical_json_bytes(raw_path.read_bytes()), context="SQP raw result"
    )
    _validate_sqp_raw_result(raw, sample, source)

    memory_path = sample.gpu_memory.resolve_and_validate(campaign_root)
    memory = expect_mapping(
        load_canonical_json_bytes(memory_path.read_bytes()), context="SQP GPU memory"
    )
    _validate_sqp_memory(memory, sample, runtime)

    certified = False
    if sample.endpoint_certificate is not None:
        certificate_path = sample.endpoint_certificate.resolve_and_validate(
            campaign_root
        )
        certificate = expect_mapping(
            load_canonical_json_bytes(certificate_path.read_bytes()),
            context="SQP endpoint certificate",
        )
        certified = _validate_sqp_certificate(certificate, sample, source, raw)
    if sample.terminal_status == "CONVERGED":
        if sample.endpoint_certificate is None or not certified:
            raise ValueError("converged SQP campaign sample requires certification")
    elif sample.endpoint_certificate is not None:
        raise ValueError("nonconverged SQP sample cannot contain a certificate")
    if sample.promotion_eligible != certified:
        raise ValueError("SQP sample promotion eligibility differs from certificate")
    return sample


def load_sqp_sample_receipt(
    campaign_root: Path, artifact: ArtifactRef
) -> SqpSampleReceipt:
    """Resolve and fully validate one immutable campaign-local SQP sample."""

    root = campaign_root.resolve(strict=True)
    return _validate_sqp_sample(artifact.resolve_and_validate(root), root)


def _validate_sqp_sample_chain(receipt: CampaignReceiptV2, campaign_root: Path) -> None:
    if not receipt.sqp_samples:
        return
    if len({reference.relative_path for reference in receipt.sqp_samples}) != len(
        receipt.sqp_samples
    ):
        raise ValueError("SQP samples may not replace or duplicate a sample path")
    if len({reference.sha256 for reference in receipt.sqp_samples}) != len(
        receipt.sqp_samples
    ):
        raise ValueError("SQP samples may not reuse one sample's bytes")
    samples = tuple(
        _validate_sqp_sample(
            reference.resolve_and_validate(campaign_root), campaign_root
        )
        for reference in receipt.sqp_samples
    )
    complete_order = (
        CompleteSample.COLD,
        CompleteSample.WARM_1,
        CompleteSample.WARM_2,
        CompleteSample.WARM_3,
    )
    expected_order = complete_order[: len(samples)]
    if tuple(sample.request.sample for sample in samples) != expected_order:
        raise ValueError("SQP samples are missing, reordered, or replaced")
    cold = samples[0]
    if receipt.route_outcomes[-1].terminal_status != cold.terminal_status:
        raise ValueError("CFS-SQP1 route status differs from its cold sample")
    if len(samples) > 1 and not cold.promotion_eligible:
        raise ValueError("SQP warm samples are forbidden before cold certification")
    losing_warm_indices = tuple(
        index
        for index, sample in enumerate(samples[1:], start=1)
        if not sample.promotion_eligible
        or sample.synchronized_solve_seconds >= SQP_WARM_SOLVE_MAX_SECONDS
    )
    if losing_warm_indices and losing_warm_indices[0] != len(samples) - 1:
        raise ValueError("SQP samples continued after a losing warm sample")

    source_identity = cold.source_identity
    device = cold.request.device
    runtime_environment = (
        cold.runtime_identity.cwd,
        cold.runtime_identity.python_executable,
        cold.runtime_identity.python_version,
        cold.runtime_identity.jax_version,
        cold.runtime_identity.jaxlib_version,
        cold.runtime_identity.simsopt_module_path,
        cold.runtime_identity.simsopt_jax_module_path,
        cold.runtime_identity.native_extension_path,
        cold.runtime_identity.backend,
        cold.runtime_identity.device_uuid,
        cold.runtime_identity.driver_version,
        cold.runtime_identity.effective_environment_sha256,
    )
    bootstrap_identity = cold.bootstrap_identity_sha256
    for sample in samples[1:]:
        if sample.source_identity != source_identity:
            raise ValueError("SQP samples mix source identities")
        if (
            sample.request.device is not device
            or (
                sample.runtime_identity.cwd,
                sample.runtime_identity.python_executable,
                sample.runtime_identity.python_version,
                sample.runtime_identity.jax_version,
                sample.runtime_identity.jaxlib_version,
                sample.runtime_identity.simsopt_module_path,
                sample.runtime_identity.simsopt_jax_module_path,
                sample.runtime_identity.native_extension_path,
                sample.runtime_identity.backend,
                sample.runtime_identity.device_uuid,
                sample.runtime_identity.driver_version,
                sample.runtime_identity.effective_environment_sha256,
            )
            != runtime_environment
        ):
            raise ValueError("SQP samples mix device identities")
        if sample.bootstrap_identity_sha256 != bootstrap_identity:
            raise ValueError("SQP samples mix bootstrap identities")

    speed_success = (
        len(samples) == 4
        and all(sample.promotion_eligible for sample in samples)
        and all(
            sample.synchronized_solve_seconds < SQP_WARM_SOLVE_MAX_SECONDS
            for sample in samples[1:]
        )
    )
    bounded_negative = not cold.promotion_eligible or bool(losing_warm_indices)
    if receipt.disposition is CampaignDisposition.ENGINEERING_SPEED_SUCCESS:
        if not speed_success:
            raise ValueError(
                "SQP speed success requires four certified samples below the warm limit"
            )
    elif not bounded_negative:
        raise ValueError("SQP bounded-negative disposition is not terminal evidence")


def validate_campaign(campaign_root: Path) -> None:
    root = campaign_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("campaign path is not a directory")
    campaign_path = root / "campaign.json"
    if campaign_path.is_symlink() or not campaign_path.is_file():
        raise ValueError("campaign.json must be a regular file")
    receipt = campaign_receipt_from_payload_dispatch(
        load_canonical_json_bytes(campaign_path.read_bytes())
    )
    expected_contract = (
        contract_sha256_v1()
        if isinstance(receipt, CampaignReceipt)
        else contract_sha256_v2()
    )
    historical_sqp_revision = (
        isinstance(receipt, CampaignReceiptV2)
        and receipt.contract_sha256 == SQP_R1_CONTRACT_SHA256
    )
    if receipt.contract_sha256 != expected_contract and not historical_sqp_revision:
        raise ValueError("campaign contract digest does not match its frozen contract")
    for outcome in receipt.route_outcomes:
        if outcome.disposition is RouteDisposition.EXECUTED:
            if outcome.receipt is None:
                raise ValueError("executed route is missing its receipt")
            if outcome.route is FullSpaceRoute.CFS_SQP1:
                if not isinstance(receipt, CampaignReceiptV2):
                    raise ValueError("campaign-v1 cannot execute CFS-SQP1")
            else:
                run_path = outcome.receipt.resolve_and_validate(root)
                _validate_executed_run(run_path, outcome.route, root)
        else:
            for evidence in outcome.gate_evidence:
                evidence.artifact.resolve_and_validate(root)
    if isinstance(receipt, CampaignReceiptV2):
        sqp_outcome = receipt.route_outcomes[-1]
        gate_order = tuple(SqpGate)
        gate_ids = tuple(evidence.gate_id for evidence in sqp_outcome.gate_evidence)
        if gate_ids != gate_order[: len(gate_ids)]:
            raise ValueError("SQP gate evidence is missing, reordered, or replaced")
        gate_results = tuple(
            load_sqp_gate_result(root, SqpGate(evidence.gate_id), evidence.artifact)
            for evidence in sqp_outcome.gate_evidence
        )
        failed_indices = tuple(
            index
            for index, gate_result in enumerate(gate_results)
            if not gate_result.passed
        )
        if failed_indices and failed_indices[0] != len(gate_results) - 1:
            raise ValueError("SQP gate evidence continued after a failed gate")
        if sqp_outcome.disposition is RouteDisposition.NOT_SELECTED_BY_GATE:
            if not failed_indices:
                raise ValueError("unselected SQP route requires a failed gate")
            if sqp_outcome.upstream_gate != gate_results[-1].gate:
                raise ValueError("SQP upstream gate differs from failed evidence")
        elif len(gate_results) != 3 or failed_indices:
            raise ValueError("executed SQP route requires three passed gates")
        _validate_sqp_sample_chain(receipt, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--campaign", type=Path)
    input_group.add_argument("--ftr-gate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ftr_gate is not None:
        gate_path = args.ftr_gate.resolve(strict=True)
        campaign_root = gate_path.parents[2]
        artifact = gate_artifact_from_path(campaign_root, gate_path)
        load_ftr_gate_result(campaign_root, artifact)
    else:
        assert args.campaign is not None
        validate_campaign(args.campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
