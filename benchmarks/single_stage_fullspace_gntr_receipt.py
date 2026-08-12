"""Semantic validator for the sealed CFS-GNTR1 convergence canary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Final

SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-gntr1-receipt-v1"
PLAN_SHA256: Final = "c2b3c88024e4d86045d220195ccb2f5bcff2ed2696f62b4090637e65298652f7"
ROUTE: Final = "CFS-GNTR1"
GPU_UUID: Final = "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
STATE_SIZE: Final = 716
EQUALITY_SIZE: Final = 255
OBJECTIVE_RESIDUAL_SIZE: Final = 2110
MAXIMUM_ACCEPTED_STEPS: Final = 8
MAXIMUM_ATTEMPTS: Final = 12
MAXIMUM_MEMORY_FRACTION: Final = 0.8
_EXPECTED_POLICY: Final = {
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
}

_RECEIPT_KEYS: Final = frozenset(
    (
        "schema_version",
        "route",
        "plan_sha256",
        "terminal_status",
        "promotion_eligible",
        "source",
        "runtime",
        "bootstrap",
        "policy",
        "execution",
        "timing",
        "transfer_audit",
        "memory",
        "route_result",
        "gate",
    )
)


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict, deterministic JSON suitable for content addressing."""

    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be a string-keyed object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return value


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _finite_close(left: object, right: object, *, absolute_tolerance: float) -> bool:
    left_value = _finite(left)
    right_value = _finite(right)
    return bool(
        left_value is not None
        and right_value is not None
        and math.isclose(
            left_value,
            right_value,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
    )


def _least_squares_slope(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    tail = values[-3:]
    mean = sum(tail) / 3.0
    numerator = sum((index - 1.0) * (value - mean) for index, value in enumerate(tail))
    denominator = sum((index - 1.0) ** 2 for index in range(3))
    return numerator / denominator


def _ratio_radius_update(attempt: dict[str, object]) -> float | None:
    radius = _finite(attempt.get("trust_radius"))
    step_norm = _finite(attempt.get("applied_step_norm"))
    ratio = _finite(attempt.get("reduction_ratio"))
    if radius is None or step_norm is None:
        return None
    if ratio is None or ratio < 0.25:
        proposed = 0.25 * step_norm
    elif ratio > 0.75:
        proposed = max(2.0 * step_norm, radius)
    else:
        proposed = radius
    return min(2.0**-4, max(2.0**-20, proposed))


def derive_gntr_gate(receipt: dict[str, object]) -> dict[str, object]:
    """Recompute the frozen terminal disposition from raw receipt evidence."""

    reasons: list[str] = []
    source = _mapping(receipt.get("source"), "source")
    runtime = _mapping(receipt.get("runtime"), "runtime")
    bootstrap = _mapping(receipt.get("bootstrap"), "bootstrap")
    policy = _mapping(receipt.get("policy"), "policy")
    execution = _mapping(receipt.get("execution"), "execution")
    transfers = _mapping(receipt.get("transfer_audit"), "transfer audit")
    memory = _mapping(receipt.get("memory"), "memory")
    gpu = _mapping(runtime.get("gpu"), "runtime GPU")

    if receipt.get("schema_version") != SCHEMA_VERSION:
        reasons.append("SCHEMA_IDENTITY")
    if receipt.get("route") != ROUTE or receipt.get("plan_sha256") != PLAN_SHA256:
        reasons.append("ROUTE_OR_PLAN_IDENTITY")
    if receipt.get("promotion_eligible") is not False:
        reasons.append("PROMOTION_SCOPE")
    if policy != _EXPECTED_POLICY:
        reasons.append("POLICY_IDENTITY")
    if (
        runtime.get("backend") != "gpu"
        or runtime.get("jax_enable_x64") is not True
        or gpu.get("uuid") != GPU_UUID
        or "RTX 5090" not in str(gpu.get("name"))
    ):
        reasons.append("RUNTIME_IDENTITY")
    if (
        bootstrap.get("state_size") != STATE_SIZE
        or bootstrap.get("equality_size") != EQUALITY_SIZE
        or bootstrap.get("objective_residual_size") != OBJECTIVE_RESIDUAL_SIZE
        or bootstrap.get("dtype") != "float64"
    ):
        reasons.append("BOOTSTRAP_IDENTITY")
    if source.get("pre_post_manifest_identical") is not True or source.get(
        "manifest_sha256"
    ) != source.get("post_manifest_sha256"):
        reasons.append("SOURCE_CHANGED")

    compile_succeeded = execution.get("lower_compile_succeeded") is True
    solve_completed = execution.get("bounded_solve_completed") is True
    finalization_completed = execution.get("endpoint_finalization_completed") is True
    if not compile_succeeded:
        reasons.append("LOWER_OR_COMPILE_FAILURE")
    elif not solve_completed:
        reasons.append("EXECUTION_FAILURE")
    elif not finalization_completed:
        reasons.append("ENDPOINT_FINALIZATION_FAILURE")

    route_result_value = receipt.get("route_result")
    initial_raw_kkt: float | None = None
    final_raw_kkt: float | None = None
    initial_objective: float | None = None
    final_objective: float | None = None
    final_scaled_feasibility: float | None = None
    accepted_steps: int | None = None
    attempts: int | None = None
    mechanism_exercised = False
    normalized_slope: float | None = None

    if finalization_completed:
        result = _mapping(route_result_value, "route result")
        optimizer = _mapping(result.get("optimizer"), "optimizer")
        final_certificate = _mapping(
            result.get("final_certificate"), "final certificate"
        )
        initial = _mapping(result.get("initial_endpoint"), "initial endpoint")
        final = _mapping(result.get("final_endpoint"), "final endpoint")
        attempts_evidence = _list(result.get("attempts"), "attempt history")
        accepted_states = _list(result.get("accepted_states"), "accepted states")
        accepted_steps = _integer(optimizer.get("accepted_steps"))
        attempts = _integer(optimizer.get("attempts"))
        mechanism_exercised = optimizer.get("mechanism_exercised") is True
        if not (
            result.get("schema_version") == "single-stage-fullspace-cfs-gntr1-result-v1"
            and result.get("route") == ROUTE
            and result.get("all_finite") is True
            and result.get("residual_contract_valid") is True
            and result.get("current_state_certificates_valid") is True
            and result.get("solver_result_consistent") is True
            and result.get("bootstrap_matches_initial") is True
            and result.get("dimensions_valid") is True
            and result.get("fp64_valid") is True
            and optimizer.get("fatal") is False
            and optimizer.get("bounded_complete") is True
            and optimizer.get("all_finite") is True
            and optimizer.get("all_accepted_states_finite") is True
            and optimizer.get("usable") is True
            and final_certificate.get("coordinates_finite") is True
            and final_certificate.get("all_finite") is True
            and final_certificate.get("certified") is True
            and initial.get("all_finite") is True
            and final.get("all_finite") is True
            and result.get("canary_usable_before_resource_gate") is True
        ):
            reasons.append("NUMERICAL_OR_CONTRACT_GATE")
        value_defect = _finite(result.get("residual_value_defect"))
        gradient_defect = _finite(result.get("residual_gradient_relative_defect"))
        scaling_defect = _finite(result.get("stationarity_scaling_relative_defect"))
        if (
            result.get("state_size") != STATE_SIZE
            or result.get("equality_size") != EQUALITY_SIZE
            or result.get("objective_residual_size") != OBJECTIVE_RESIDUAL_SIZE
            or value_defect is None
            or value_defect > 1.0e-12
            or gradient_defect is None
            or gradient_defect > 1.0e-10
            or scaling_defect is None
            or scaling_defect > 1.0e-10
        ):
            reasons.append("ROUTE_RESULT_IDENTITY")
        final_certificate_limits = (
            ("residual_value_defect", 1.0e-12, False),
            ("residual_gradient_defect", 1.0e-10, False),
            ("hvp_symmetry_defect", 1.0e-10, False),
            ("gram_factorization_relative_residual", 1.0e-10, False),
            ("multiplier_relative_residual", 1.0e-10, False),
            ("multiplier_forward_error_bound", 1.0e-7, True),
            ("projection_tangency_relative_residual", 1.0e-10, False),
            ("projection_solve_relative_residual", 1.0e-10, False),
            ("projection_forward_error_bound", 1.0e-7, True),
        )
        if any(
            (value := _finite(final_certificate.get(field))) is None
            or (value >= limit if strict else value > limit)
            for field, limit, strict in final_certificate_limits
        ) or (
            (
                final_probe := _finite(
                    final_certificate.get("probe_normalized_curvature")
                )
            )
            is None
            or final_probe < -1.0e-10
        ):
            reasons.append("FINAL_CERTIFICATE")
        if (
            attempts is None
            or accepted_steps is None
            or not 1 <= attempts <= MAXIMUM_ATTEMPTS
            or accepted_steps != MAXIMUM_ACCEPTED_STEPS
            or len(attempts_evidence) != attempts
            or len(accepted_states) != accepted_steps + 1
        ):
            reasons.append("BOUNDED_HISTORY")
        else:
            accepted_count = sum(
                _mapping(item, "attempt").get("outcome") == "ACCEPTED"
                for item in attempts_evidence
            )
            if accepted_count != accepted_steps:
                reasons.append("ACCEPTANCE_LEDGER")
            recomputed_mechanism = False
            expected_accepted_states: list[tuple[float, float]] = []
            expected_accepted_stationarity: list[float] = []
            for item_index, item in enumerate(attempts_evidence):
                attempt = _mapping(item, "attempt")
                outcome = attempt.get("outcome")
                terminal_curvature = _finite(
                    attempt.get("terminal_normalized_curvature")
                )
                value_defect = _finite(attempt.get("residual_value_defect"))
                gradient_defect = _finite(attempt.get("residual_gradient_defect"))
                symmetry_defect = _finite(attempt.get("hvp_symmetry_defect"))
                probe_curvature = _finite(attempt.get("probe_normalized_curvature"))
                projection_tangency = _finite(
                    attempt.get("current_projection_tangency_relative_residual")
                )
                projection_solve = _finite(
                    attempt.get("current_projection_solve_relative_residual")
                )
                projection_forward_error = _finite(
                    attempt.get("current_projection_forward_error_bound")
                )
                if not (
                    value_defect is not None
                    and value_defect <= 1.0e-12
                    and gradient_defect is not None
                    and gradient_defect <= 1.0e-10
                    and symmetry_defect is not None
                    and symmetry_defect <= 1.0e-10
                    and probe_curvature is not None
                    and probe_curvature >= -1.0e-10
                    and projection_tangency is not None
                    and projection_tangency <= 1.0e-10
                    and projection_solve is not None
                    and projection_solve <= 1.0e-10
                    and projection_forward_error is not None
                    and projection_forward_error < 1.0e-7
                ):
                    reasons.append("CURRENT_STATE_CERTIFICATE")
                    break
                steihaug_termination = attempt.get("steihaug_termination")
                steihaug_tangency = _finite(
                    attempt.get("steihaug_tangency_relative_residual")
                )
                steihaug_final_residual = _finite(
                    attempt.get("steihaug_final_projected_residual_norm")
                )
                steihaug_target = _finite(
                    attempt.get("steihaug_projected_residual_target")
                )
                steihaug_hit_boundary = attempt.get("steihaug_hit_boundary")
                residual_projection_tangency = _finite(
                    attempt.get(
                        "steihaug_residual_projection_tangency_relative_residual"
                    )
                )
                residual_projection_solve = _finite(
                    attempt.get("steihaug_residual_projection_solve_relative_residual")
                )
                residual_projection_forward_error = _finite(
                    attempt.get("steihaug_residual_projection_forward_error_bound")
                )
                steihaug_certificate = (
                    steihaug_tangency is not None
                    and steihaug_tangency <= 1.0e-10
                    and steihaug_final_residual is not None
                    and steihaug_target is not None
                    and steihaug_termination
                    in {
                        "INTERIOR_CONVERGED",
                        "TRUST_BOUNDARY",
                        "NONPOSITIVE_CURVATURE",
                    }
                    and (
                        (
                            steihaug_termination == "INTERIOR_CONVERGED"
                            and steihaug_hit_boundary is False
                            and steihaug_final_residual <= steihaug_target
                        )
                        or (
                            steihaug_termination
                            in {"TRUST_BOUNDARY", "NONPOSITIVE_CURVATURE"}
                            and steihaug_hit_boundary is True
                        )
                    )
                    and terminal_curvature is not None
                    and terminal_curvature >= -1.0e-10
                    and residual_projection_tangency is not None
                    and residual_projection_tangency <= 1.0e-10
                    and residual_projection_solve is not None
                    and residual_projection_solve <= 1.0e-10
                    and residual_projection_forward_error is not None
                    and residual_projection_forward_error < 1.0e-7
                )
                if not steihaug_certificate:
                    reasons.append("STEIHAUG_CERTIFICATE")
                    break
                if outcome == "ACCEPTED":
                    actual_reduction = _finite(attempt.get("actual_reduction"))
                    predicted_reduction = _finite(attempt.get("predicted_reduction"))
                    candidate_feasibility = _finite(
                        attempt.get("candidate_feasibility_inf")
                    )
                    correction_ratio = _finite(attempt.get("correction_step_ratio"))
                    corrected_radius_ratio = _finite(
                        attempt.get("corrected_radius_ratio")
                    )
                    correction_residual = _finite(
                        attempt.get("correction_relative_residual")
                    )
                    correction_forward_error = _finite(
                        attempt.get("correction_forward_error_bound")
                    )
                    trial_gram_residual = _finite(
                        attempt.get("trial_gram_factorization_relative_residual")
                    )
                    trial_gram_solve_residual = _finite(
                        attempt.get("trial_gram_solve_relative_residual")
                    )
                    next_radius = _finite(attempt.get("next_trust_radius"))
                    expected_radius = _ratio_radius_update(attempt)
                    if not (
                        actual_reduction is not None
                        and actual_reduction > 0.0
                        and predicted_reduction is not None
                        and predicted_reduction > 0.0
                        and candidate_feasibility is not None
                        and candidate_feasibility <= 1.0e-10
                        and correction_ratio is not None
                        and correction_ratio <= 1.0e-3
                        and corrected_radius_ratio is not None
                        and corrected_radius_ratio <= 1.0 + 1.0e-6
                        and correction_residual is not None
                        and correction_residual <= 1.0e-10
                        and correction_forward_error is not None
                        and correction_forward_error < 1.0e-7
                        and trial_gram_residual is not None
                        and trial_gram_residual <= 1.0e-10
                        and trial_gram_solve_residual is not None
                        and trial_gram_solve_residual <= 1.0e-10
                        and next_radius is not None
                        and expected_radius is not None
                        and math.isclose(
                            next_radius,
                            expected_radius,
                            rel_tol=0.0,
                            abs_tol=1.0e-15,
                        )
                    ):
                        reasons.append("ACCEPTED_STEP_GATE")
                        break
                    assert candidate_feasibility is not None
                    candidate_objective = _finite(attempt.get("candidate_objective"))
                    if candidate_objective is None:
                        reasons.append("ACCEPTED_STEP_GATE")
                        break
                    expected_accepted_states.append(
                        (candidate_objective, candidate_feasibility)
                    )
                    accepted_number = _integer(attempt.get("accepted_step_number"))
                    hvp_evaluations = _integer(attempt.get("steihaug_hvp_evaluations"))
                    rotation = _finite(attempt.get("direction_rotation"))
                    next_stationarity = (
                        _finite(
                            _mapping(
                                attempts_evidence[item_index + 1],
                                "next attempt",
                            ).get("current_stationarity_inf")
                        )
                        if item_index + 1 < len(attempts_evidence)
                        else _finite(optimizer.get("final_scaled_stationarity_inf"))
                    )
                    if next_stationarity is None:
                        reasons.append("ACCEPTED_STATE_LEDGER")
                        break
                    expected_accepted_stationarity.append(next_stationarity)
                    recomputed_mechanism = recomputed_mechanism or bool(
                        accepted_number is not None
                        and accepted_number >= 2
                        and hvp_evaluations is not None
                        and hvp_evaluations >= 2
                        and rotation is not None
                        and rotation >= 1.0e-3
                    )
                elif isinstance(outcome, str) and outcome.startswith("RETRY_"):
                    radius = _finite(attempt.get("trust_radius"))
                    next_radius = _finite(attempt.get("next_trust_radius"))
                    expected_radius = (
                        _ratio_radius_update(attempt)
                        if outcome == "RETRY_OBJECTIVE"
                        else (
                            None
                            if radius is None
                            else min(2.0**-4, max(2.0**-20, 0.25 * radius))
                        )
                    )
                    if (
                        radius is None
                        or next_radius is None
                        or expected_radius is None
                        or not math.isclose(
                            next_radius,
                            expected_radius,
                            rel_tol=0.0,
                            abs_tol=1.0e-15,
                        )
                    ):
                        reasons.append("RETRY_RADIUS_RULE")
                        break
                else:
                    reasons.append("ATTEMPT_OUTCOME")
                    break
            if optimizer.get("mechanism_exercised") is not recomputed_mechanism:
                reasons.append("MECHANISM_LEDGER")
            if accepted_states:
                initial_state = _mapping(accepted_states[0], "bootstrap accepted state")
                initial_current = _mapping(attempts_evidence[0], "first attempt")
                accepted_state_pairs = [
                    (
                        _finite(
                            _mapping(state, "accepted state").get("physical_objective")
                        ),
                        _finite(
                            _mapping(state, "accepted state").get(
                                "scaled_feasibility_inf"
                            )
                        ),
                    )
                    for state in accepted_states[1:]
                ]
                accepted_state_stationarity = [
                    _finite(
                        _mapping(state, "accepted state").get("scaled_stationarity_inf")
                    )
                    for state in accepted_states[1:]
                ]
                pairs_match = len(accepted_state_pairs) == len(
                    expected_accepted_states
                ) and all(
                    observed_objective is not None
                    and observed_feasibility is not None
                    and math.isclose(
                        observed_objective,
                        expected_objective,
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    )
                    and math.isclose(
                        observed_feasibility,
                        expected_feasibility,
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    )
                    for (
                        observed_objective,
                        observed_feasibility,
                    ), (
                        expected_objective,
                        expected_feasibility,
                    ) in zip(
                        accepted_state_pairs,
                        expected_accepted_states,
                        strict=True,
                    )
                )
                if (
                    not _finite_close(
                        initial_state.get("physical_objective"),
                        initial_current.get("current_objective"),
                        absolute_tolerance=1.0e-15,
                    )
                    or not _finite_close(
                        initial_state.get("scaled_feasibility_inf"),
                        initial_current.get("current_feasibility_inf"),
                        absolute_tolerance=1.0e-15,
                    )
                    or not _finite_close(
                        initial_state.get("scaled_stationarity_inf"),
                        initial_current.get("current_stationarity_inf"),
                        absolute_tolerance=1.0e-15,
                    )
                    or not pairs_match
                    or accepted_state_stationarity != expected_accepted_stationarity
                ):
                    reasons.append("ACCEPTED_STATE_LEDGER")
        initial_objective = _finite(initial.get("physical_objective"))
        final_objective = _finite(final.get("physical_objective"))
        initial_raw_kkt = _finite(initial.get("raw_kkt_stationarity_inf"))
        final_raw_kkt = _finite(final.get("raw_kkt_stationarity_inf"))
        final_scaled_feasibility = _finite(final.get("scaled_feasibility_inf"))
        scaled_stationarity = [
            _finite(_mapping(item, "accepted state").get("scaled_stationarity_inf"))
            for item in accepted_states
        ]
        if (
            initial_objective is None
            or final_objective is None
            or initial_raw_kkt is None
            or final_raw_kkt is None
            or final_scaled_feasibility is None
            or any(value is None for value in scaled_stationarity)
        ):
            reasons.append("ENDPOINT_OR_TREND_EVIDENCE")
        else:
            stationarity_values = [
                value for value in scaled_stationarity if value is not None
            ]
            slope = _least_squares_slope(stationarity_values)
            bootstrap_stationarity = stationarity_values[0]
            if slope is not None and bootstrap_stationarity > 0.0:
                normalized_slope = slope / bootstrap_stationarity

    if finalization_completed and (
        transfers.get("initial_h2d_calls") != 1
        or transfers.get("hot_h2d_calls") != 0
        or transfers.get("hot_d2h_calls") != 0
        or transfers.get("final_d2h_calls") != 1
        or transfers.get("timed_execution_transfer_guard") != "disallow"
    ):
        reasons.append("TRANSFER_BUDGET")
    peak_fraction = _finite(memory.get("peak_memory_fraction"))
    if finalization_completed and (
        memory.get("gpu_uuid") != GPU_UUID
        or memory.get("target_pid_observed") is not True
        or peak_fraction is None
        or not 0.0 <= peak_fraction < MAXIMUM_MEMORY_FRACTION
    ):
        reasons.append("MEMORY_BUDGET")

    usable = not reasons
    selected = bool(
        usable
        and mechanism_exercised
        and initial_raw_kkt is not None
        and final_raw_kkt is not None
        and final_raw_kkt <= 0.9 * initial_raw_kkt
        and initial_objective is not None
        and final_objective is not None
        and final_objective <= initial_objective
        and normalized_slope is not None
        and normalized_slope <= -5.0e-3
    )
    if not usable:
        terminal_status = "CANARY_NOT_USABLE"
    elif not mechanism_exercised:
        terminal_status = "MECHANISM_NOT_EXERCISED"
    elif selected:
        terminal_status = "SELECTED_FOR_ENDPOINT_CAMPAIGN"
    else:
        terminal_status = "NOT_SELECTED_BY_BOUNDED_CONVERGENCE_CANARY"
    return {
        "gate_status": "PASS" if usable else "FAIL",
        "terminal_status": terminal_status,
        "failure_reasons": reasons,
        "selected_for_endpoint_campaign": selected,
        "mechanism_exercised": mechanism_exercised,
        "accepted_steps": accepted_steps,
        "attempts": attempts,
        "initial_raw_kkt_stationarity_inf": initial_raw_kkt,
        "final_raw_kkt_stationarity_inf": final_raw_kkt,
        "initial_physical_objective": initial_objective,
        "final_physical_objective": final_objective,
        "final_scaled_feasibility_inf": final_scaled_feasibility,
        "normalized_last_three_scaled_stationarity_slope": normalized_slope,
    }


def load_and_validate_gntr_artifact(output_root: Path) -> dict[str, object]:
    """Rehash, strictly parse, and semantically validate one sealed artifact."""

    root = output_root.resolve(strict=True)
    if root.stat().st_mode & 0o222:
        raise ValueError("GNTR artifact directory must be read-only")
    if {path.name for path in root.iterdir()} != {
        "result.json",
        "source-manifest.json",
    }:
        raise ValueError("GNTR artifact must contain exactly two sealed files")
    result_path = root / "result.json"
    manifest_path = root / "source-manifest.json"
    for path in (result_path, manifest_path):
        if path.stat().st_mode & 0o222:
            raise ValueError(f"GNTR artifact file must be read-only: {path.name}")
    encoded = result_path.read_bytes()
    receipt = _mapping(json.loads(encoded), "GNTR receipt")
    if canonical_json_bytes(receipt) != encoded:
        raise ValueError("GNTR receipt is not canonical strict JSON")
    if frozenset(receipt) != _RECEIPT_KEYS:
        raise ValueError("GNTR receipt has unexpected top-level keys")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _list(json.loads(manifest_bytes), "source manifest")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("GNTR source manifest is not canonical strict JSON")
    source = _mapping(receipt["source"], "source")
    if source.get("manifest_sha256") != hashlib.sha256(
        manifest_bytes
    ).hexdigest() or source.get("manifest_entry_count") != len(manifest):
        raise ValueError("GNTR source manifest identity mismatch")
    seen_paths: set[str] = set()
    for raw_entry in manifest:
        entry = _mapping(raw_entry, "source manifest entry")
        if frozenset(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError("GNTR source manifest entry has unexpected keys")
        relative_path = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith("/")
            or ".." in Path(relative_path).parts
            or relative_path in seen_paths
            or not isinstance(digest, str)
            or len(digest) != 64
            or _integer(size) is None
            or int(size) < 0
        ):
            raise ValueError("GNTR source manifest entry is invalid")
        name = Path(relative_path).name
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            raise ValueError("GNTR source manifest includes a forbidden env file")
        seen_paths.add(relative_path)
    detail = derive_gntr_gate(receipt)
    if (
        receipt["gate"] != detail
        or receipt["terminal_status"] != detail["terminal_status"]
    ):
        raise ValueError("GNTR terminal gate differs from recomputed evidence")
    return receipt


__all__ = (
    "EQUALITY_SIZE",
    "GPU_UUID",
    "MAXIMUM_ACCEPTED_STEPS",
    "MAXIMUM_ATTEMPTS",
    "MAXIMUM_MEMORY_FRACTION",
    "OBJECTIVE_RESIDUAL_SIZE",
    "PLAN_SHA256",
    "ROUTE",
    "SCHEMA_VERSION",
    "STATE_SIZE",
    "canonical_json_bytes",
    "derive_gntr_gate",
    "load_and_validate_gntr_artifact",
)
