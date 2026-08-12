from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from benchmarks.single_stage_fullspace_gntr_receipt import (
    GPU_UUID,
    PLAN_SHA256,
    ROUTE,
    SCHEMA_VERSION,
    canonical_json_bytes,
    derive_gntr_gate,
    load_and_validate_gntr_artifact,
)


def _attempt(index: int) -> dict[str, object]:
    radius = min(2.0**-4, 2.0 ** (-10 + index))
    return {
        "attempt": index + 1,
        "outcome": "ACCEPTED",
        "accepted_step_number": index + 1,
        "current_objective": 1.0 - 0.01 * index,
        "current_feasibility_inf": 1.0e-12,
        "current_stationarity_inf": 1.0 - 0.01 * index,
        "candidate_objective": 0.99 - 0.01 * index,
        "candidate_feasibility_inf": 1.0e-12,
        "actual_reduction": 0.01,
        "predicted_reduction": 0.01,
        "reduction_ratio": 1.0,
        "trust_radius": radius,
        "next_trust_radius": min(2.0**-4, 2.0 * radius),
        "tangent_step_norm": radius,
        "correction_norm": 0.0,
        "applied_step_norm": radius,
        "correction_step_ratio": 0.0,
        "corrected_radius_ratio": 1.0,
        "steihaug_iterations": 2,
        "steihaug_hvp_evaluations": 2,
        "steihaug_termination": "TRUST_BOUNDARY",
        "terminal_normalized_curvature": 0.1,
        "residual_value_defect": 1.0e-15,
        "residual_gradient_defect": 1.0e-15,
        "hvp_symmetry_defect": 1.0e-15,
        "probe_normalized_curvature": 0.1,
        "direction_rotation": 1.0e-2,
        "correction_relative_residual": 1.0e-15,
        "correction_forward_error_bound": 1.0e-12,
        "trial_gram_factorization_relative_residual": 1.0e-15,
        "trial_gram_solve_relative_residual": 1.0e-15,
        "current_projection_tangency_relative_residual": 1.0e-15,
        "current_projection_solve_relative_residual": 1.0e-15,
        "current_projection_forward_error_bound": 1.0e-12,
        "steihaug_tangency_relative_residual": 1.0e-15,
        "steihaug_final_projected_residual_norm": 1.0,
        "steihaug_projected_residual_target": 1.0e-10,
        "steihaug_hit_boundary": True,
        "steihaug_residual_projection_tangency_relative_residual": 1.0e-15,
        "steihaug_residual_projection_solve_relative_residual": 1.0e-15,
        "steihaug_residual_projection_forward_error_bound": 1.0e-12,
    }


def _passing_receipt() -> dict[str, object]:
    attempts = [_attempt(index) for index in range(8)]
    accepted_states = [
        {
            "accepted_step": index,
            "physical_objective": 1.0 - 0.01 * index,
            "scaled_feasibility_inf": 1.0e-12,
            "scaled_stationarity_inf": 1.0 - 0.01 * index,
        }
        for index in range(9)
    ]
    route_result = {
        "schema_version": "single-stage-fullspace-cfs-gntr1-result-v1",
        "route": "CFS-GNTR1",
        "optimizer": {
            "fatal": False,
            "bounded_complete": True,
            "mechanism_exercised": True,
            "accepted_steps": 8,
            "attempts": 8,
            "final_scaled_stationarity_inf": 0.92,
            "all_finite": True,
            "all_accepted_states_finite": True,
            "usable": True,
        },
        "final_certificate": {
            "coordinates_finite": True,
            "residual_value_defect": 1.0e-15,
            "residual_gradient_defect": 1.0e-15,
            "hvp_symmetry_defect": 1.0e-15,
            "probe_normalized_curvature": 0.1,
            "gram_factorization_relative_residual": 1.0e-15,
            "multiplier_relative_residual": 1.0e-15,
            "multiplier_forward_error_bound": 1.0e-12,
            "projection_tangency_relative_residual": 1.0e-15,
            "projection_solve_relative_residual": 1.0e-15,
            "projection_forward_error_bound": 1.0e-12,
            "all_finite": True,
            "certified": True,
        },
        "attempts": attempts,
        "accepted_states": accepted_states,
        "initial_endpoint": {
            "physical_objective": 1.0,
            "raw_kkt_stationarity_inf": 1.0,
            "all_finite": True,
        },
        "final_endpoint": {
            "physical_objective": 0.92,
            "scaled_feasibility_inf": 1.0e-12,
            "raw_kkt_stationarity_inf": 0.8,
            "all_finite": True,
        },
        "residual_value_defect": 1.0e-15,
        "residual_gradient_relative_defect": 1.0e-15,
        "stationarity_scaling_relative_defect": 1.0e-15,
        "objective_residual_size": 2110,
        "state_size": 716,
        "equality_size": 255,
        "all_finite": True,
        "residual_contract_valid": True,
        "current_state_certificates_valid": True,
        "solver_result_consistent": True,
        "bootstrap_matches_initial": True,
        "dimensions_valid": True,
        "fp64_valid": True,
        "canary_usable_before_resource_gate": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "route": ROUTE,
        "plan_sha256": PLAN_SHA256,
        "terminal_status": "SELECTED_FOR_ENDPOINT_CAMPAIGN",
        "promotion_eligible": False,
        "source": {
            "pre_post_manifest_identical": True,
            "manifest_sha256": "0" * 64,
            "post_manifest_sha256": "0" * 64,
        },
        "runtime": {
            "backend": "gpu",
            "jax_enable_x64": True,
            "gpu": {"uuid": GPU_UUID, "name": "NVIDIA GeForce RTX 5090"},
        },
        "bootstrap": {
            "state_size": 716,
            "equality_size": 255,
            "objective_residual_size": 2110,
            "dtype": "float64",
        },
        "policy": {
            "maximum_accepted_steps": 8,
            "maximum_attempts": 12,
            "initial_trust_radius": 2.0**-10,
            "minimum_trust_radius": 2.0**-20,
            "maximum_trust_radius": 2.0**-4,
            "maximum_steihaug_iterations": 32,
            "projected_residual_tolerance": 1.0e-10,
            "linear_residual_tolerance": 1.0e-10,
            "corrected_feasibility_tolerance": 1.0e-10,
            "forward_error_tolerance": 1.0e-7,
            "residual_value_defect_tolerance": 1.0e-12,
            "residual_gradient_defect_tolerance": 1.0e-10,
            "normalized_curvature_tolerance": 1.0e-10,
            "maximum_correction_step_ratio": 1.0e-3,
            "maximum_corrected_radius_excess": 1.0e-6,
            "mechanism_rotation_threshold": 1.0e-3,
        },
        "execution": {
            "lower_compile_succeeded": True,
            "bounded_solve_completed": True,
            "endpoint_finalization_completed": True,
        },
        "timing": {},
        "transfer_audit": {
            "initial_h2d_calls": 1,
            "hot_h2d_calls": 0,
            "hot_d2h_calls": 0,
            "final_d2h_calls": 1,
            "timed_execution_transfer_guard": "disallow",
        },
        "memory": {
            "gpu_uuid": GPU_UUID,
            "target_pid_observed": True,
            "peak_memory_fraction": 0.5,
        },
        "route_result": route_result,
        "gate": {},
    }


def test_selected_gate_is_recomputed_from_raw_evidence() -> None:
    detail = derive_gntr_gate(_passing_receipt())

    assert detail["gate_status"] == "PASS"
    assert detail["terminal_status"] == "SELECTED_FOR_ENDPOINT_CAMPAIGN"
    assert detail["selected_for_endpoint_campaign"] is True


def test_mechanism_not_exercised_is_not_an_algorithm_negative() -> None:
    receipt = _passing_receipt()
    route_result = receipt["route_result"]
    assert isinstance(route_result, dict)
    optimizer = route_result["optimizer"]
    assert isinstance(optimizer, dict)
    optimizer["mechanism_exercised"] = False
    attempts = route_result["attempts"]
    assert isinstance(attempts, list)
    for attempt in attempts:
        assert isinstance(attempt, dict)
        attempt["direction_rotation"] = 0.0

    detail = derive_gntr_gate(receipt)

    assert detail["gate_status"] == "PASS"
    assert detail["terminal_status"] == "MECHANISM_NOT_EXERCISED"


def test_bootstrap_ledger_accepts_sub_ulp_independent_endpoint_difference() -> None:
    receipt = _passing_receipt()
    route_result = receipt["route_result"]
    assert isinstance(route_result, dict)
    accepted_states = route_result["accepted_states"]
    attempts = route_result["attempts"]
    assert isinstance(accepted_states, list)
    assert isinstance(attempts, list)
    bootstrap_state = accepted_states[0]
    first_attempt = attempts[0]
    assert isinstance(bootstrap_state, dict)
    assert isinstance(first_attempt, dict)
    first_attempt["current_feasibility_inf"] = 1.3375024930542707e-15
    bootstrap_state["scaled_feasibility_inf"] = 1.2260439519664147e-15

    detail = derive_gntr_gate(receipt)

    assert detail["gate_status"] == "PASS"
    assert "ACCEPTED_STATE_LEDGER" not in detail["failure_reasons"]


def test_compile_failure_is_a_truthful_terminal_receipt() -> None:
    receipt = _passing_receipt()
    execution = receipt["execution"]
    assert isinstance(execution, dict)
    execution["lower_compile_succeeded"] = False
    execution["bounded_solve_completed"] = False
    execution["endpoint_finalization_completed"] = False
    receipt["route_result"] = None

    detail = derive_gntr_gate(receipt)

    assert detail["terminal_status"] == "CANARY_NOT_USABLE"
    assert detail["failure_reasons"] == ["LOWER_OR_COMPILE_FAILURE"]


def test_semantic_mutations_fail_closed() -> None:
    cases = (
        ("memory", "peak_memory_fraction", 0.8, "MEMORY_BUDGET"),
        ("transfer_audit", "hot_d2h_calls", 1, "TRANSFER_BUDGET"),
        ("source", "pre_post_manifest_identical", False, "SOURCE_CHANGED"),
    )
    for section, key, value, reason in cases:
        receipt = deepcopy(_passing_receipt())
        target = receipt[section]
        assert isinstance(target, dict)
        target[key] = value

        detail = derive_gntr_gate(receipt)

        assert reason in detail["failure_reasons"]
        assert detail["terminal_status"] == "CANARY_NOT_USABLE"


def test_raw_kkt_and_scaled_trend_are_both_required_for_selection() -> None:
    receipt = _passing_receipt()
    route_result = receipt["route_result"]
    assert isinstance(route_result, dict)
    final = route_result["final_endpoint"]
    assert isinstance(final, dict)
    final["raw_kkt_stationarity_inf"] = 0.95

    detail = derive_gntr_gate(receipt)

    assert detail["gate_status"] == "PASS"
    assert detail["terminal_status"] == "NOT_SELECTED_BY_BOUNDED_CONVERGENCE_CANARY"


def test_sealed_artifact_is_rehashed_and_semantically_revalidated(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = [
        {
            "path": "benchmarks/runner.py",
            "sha256": "1" * 64,
            "size_bytes": 123,
        }
    ]
    manifest_bytes = canonical_json_bytes(manifest)
    receipt = _passing_receipt()
    source = receipt["source"]
    assert isinstance(source, dict)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    source.update(
        {
            "manifest_sha256": digest,
            "post_manifest_sha256": digest,
            "manifest_entry_count": 1,
        }
    )
    gate = derive_gntr_gate(receipt)
    receipt["gate"] = gate
    receipt["terminal_status"] = gate["terminal_status"]
    result_path = artifact / "result.json"
    manifest_path = artifact / "source-manifest.json"
    result_path.write_bytes(canonical_json_bytes(receipt))
    manifest_path.write_bytes(manifest_bytes)
    result_path.chmod(0o444)
    manifest_path.chmod(0o444)
    artifact.chmod(0o555)
    try:
        loaded = load_and_validate_gntr_artifact(artifact)
        assert loaded["terminal_status"] == "SELECTED_FOR_ENDPOINT_CAMPAIGN"
    finally:
        artifact.chmod(0o755)
        result_path.chmod(0o644)
        manifest_path.chmod(0o644)
