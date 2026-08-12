from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from benchmarks.single_stage_compute_graph_c0_evaluator import EvaluationResult
from benchmarks.single_stage_compute_graph_canary_evaluator import (
    CANARY_CHILD_SCHEMA_ID,
    CanaryEvaluation,
    CanaryEvaluatorError,
    GpuMemoryEvidence,
    blocked_observation,
    evaluate_canary_once,
)


def _numerical() -> EvaluationResult:
    return EvaluationResult(
        objective=1.25,
        gradient=np.linspace(0.0, 1.0, 461, dtype=np.float64),
        inner_newton_success=True,
        adjoint_success=True,
        residual_certificates={"boozer_exact_residual_l2": 1.0e-13},
    )


def _telemetry(variant: str) -> dict[str, int | bool]:
    values: dict[str, int | bool] = {
        "exact_newton_variant_dense_linearization_used": True,
        "exact_newton_variant_linear_solve_attempt_count": 2,
        "exact_newton_variant_dense_materialization_count": 2,
        "exact_newton_variant_lu_factorization_count": 2,
        "exact_newton_variant_lu_solve_count": 3,
        "exact_newton_variant_refinement_correction_count": 1,
        "exact_newton_variant_stop_reason_code": 0,
        "exact_newton_variant_numerical_failure": False,
    }
    if variant == "C2":
        values.update(
            {
                "exact_newton_variant_applied_update_count": 1,
                "exact_newton_variant_rollback_branch_taken": False,
                "exact_newton_variant_rollback_recompute_count": 0,
                "exact_newton_variant_native_persist_predicate": True,
                "exact_newton_variant_persist_solved_state": True,
                "exact_newton_variant_initial_norm": 1,
                "exact_newton_variant_assessed_norm": 0,
                "exact_newton_variant_returned_norm": 0,
            }
        )
    else:
        values.update(
            {
                "exact_newton_variant_backtracking_iteration_count": 1,
                "exact_newton_variant_stalled": False,
                "exact_newton_variant_retry_linear_solve_at_strict_cap": False,
            }
        )
    return values


@dataclass
class _Prepared:
    result: CanaryEvaluation

    def evaluate_once(self) -> CanaryEvaluation:
        return self.result


@pytest.mark.parametrize("variant", ["C1", "C2"])
def test_gate_requires_full_variant_telemetry(variant: str) -> None:
    observation = evaluate_canary_once(
        variant=variant,  # type: ignore[arg-type]
        mode="gate",
        sample_index=None,
        parameter_sha256="1" * 64,
        prepared=_Prepared(CanaryEvaluation(_numerical(), _telemetry(variant))),
        clock=iter((10, 30)).__next__,
        peak_rss=lambda: 40,
        gpu_memory=lambda: GpuMemoryEvidence(1, "GPU-test", 2, 10_000_000, 80),
    )

    assert observation["schema_id"] == CANARY_CHILD_SCHEMA_ID
    assert observation["status"] == "PASS"
    assert observation["variant"] == variant
    assert observation["gradient_dtype"] == "float64"
    assert observation["gpu_memory"]["peak_bytes"] == 80  # type: ignore[index]
    assert len(observation["gradient"]) == 461  # type: ignore[arg-type]
    assert observation["telemetry"] == _telemetry(variant)


def test_initial_gate_records_parameter_identity() -> None:
    observation = evaluate_canary_once(
        variant="C1",
        mode="initial_gate",
        sample_index=None,
        parameter_sha256="9" * 64,
        prepared=_Prepared(CanaryEvaluation(_numerical(), _telemetry("C1"))),
        clock=iter((10, 30)).__next__,
        peak_rss=lambda: 40,
        gpu_memory=lambda: GpuMemoryEvidence(1, "GPU-test", 2, 10_000_000, 80),
    )

    assert observation["mode"] == "initial_gate"
    assert observation["parameter_sha256"] == "9" * 64


def test_missing_production_telemetry_fails_closed_after_evaluation() -> None:
    prepared = _Prepared(CanaryEvaluation(_numerical(), {}))

    with pytest.raises(CanaryEvaluatorError, match="did not export"):
        evaluate_canary_once(
            variant="C1",
            mode="gate",
            sample_index=None,
            parameter_sha256="1" * 64,
            prepared=prepared,
            clock=iter((10, 30)).__next__,
        )


def test_blocked_observation_is_not_a_c0_receipt_or_speed_verdict() -> None:
    observation = blocked_observation("C2", "warm", "telemetry unavailable")

    assert observation == {
        "schema_id": CANARY_CHILD_SCHEMA_ID,
        "status": "BLOCKED",
        "variant": "C2",
        "mode": "warm",
        "blocker": {
            "code": "PRODUCTION_VARIANT_TELEMETRY_UNAVAILABLE",
            "reason": "telemetry unavailable",
        },
    }
    assert "verdict" not in observation


def test_warm_index_is_required_and_gate_index_is_forbidden() -> None:
    prepared = _Prepared(CanaryEvaluation(_numerical(), _telemetry("C1")))

    with pytest.raises(CanaryEvaluatorError, match="only warm"):
        evaluate_canary_once(
            variant="C1",
            mode="warm",
            sample_index=None,
            parameter_sha256="1" * 64,
            prepared=prepared,
        )
