from __future__ import annotations

import hashlib
from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest
from benchmarks.single_stage_fullspace_ftr_receipt import (
    FTR_PRIMARY_DEVICE_UUID,
    derive_ftr_gate_detail,
    validate_ftr_device_binding,
)
from benchmarks.single_stage_fullspace_receipt import (
    DeviceLane,
    RunPhase,
    RunRequest,
    contract_sha256_v1,
    contract_sha256_v2,
    run_request_payload_v3,
    run_request_v3_from_payload,
)
from benchmarks.single_stage_fullspace_snapshot import (
    JsonValue,
    RuntimeEvidence,
    RuntimeIdentity,
    canonical_json_bytes,
)
from simsopt_jax.solve.fullspace import FullSpaceRoute


def _passing_evidence() -> tuple[dict[str, object], dict[str, object]]:
    optimizer = {
        "status": "ITERATION_LIMIT",
        "fatal": False,
        "failed": True,
        "all_finite": True,
        "all_accepted_states_finite": True,
        "solver_result_consistent": True,
        "solve_certificates_valid": True,
        "iterations": 10,
        "joint_evaluations": 30,
        "coordinate_count": 716,
        "equality_count": 255,
        "dtype": "float64",
        "history": {
            "attempted_length": 10,
            "accepted_length": 6,
            "accepted": [True] * 6 + [False] * 4,
            "filter_accepted": [True] * 10,
            "selected_radius_index": [0] * 10,
        },
        "initial_physical_objective": 2.0,
        "initial_scaled_constraint_infinity_norm": 1.0e-11,
        "initial_raw_kkt_stationarity_infinity_norm": 1.0e-2,
        "final_normal_relative_residual": 1.0e-12,
        "final_normal_forward_error_bound": 1.0e-9,
        "final_tangency_relative_residual": 1.0e-12,
        "final_multiplier_projection_relative_residual": 1.0e-12,
        "final_multiplier_projection_forward_error_bound": 1.0e-9,
    }
    raw = {
        "optimizer_result": optimizer,
        "endpoint": {
            "all_finite": True,
            "physical_objective": 1.0,
            "scaled_constraint_infinity_norm": 5.0e-11,
            "raw_kkt_stationarity_infinity_norm": 1.0e-3,
        },
        "independent_kkt": {
            "valid": True,
            "all_finite": True,
            "kkt_relative_residual": 1.0e-12,
            "schur_relative_residual": 1.0e-12,
            "kkt_solution_scaled_residual": 1.0e-12,
            "kkt_forward_error_bound": 1.0e-9,
        },
        "transfer_audit": {
            "initial_h2d_calls": 1,
            "hot_h2d_calls": 0,
            "hot_d2h_calls": 0,
            "final_d2h_calls": 1,
        },
        "timing": {"synchronized_solve_seconds": 20.0},
    }
    return raw, {"peak_memory_fraction": 0.5}


def _runtime_and_memory(
    *,
    runtime_uuid: str = FTR_PRIMARY_DEVICE_UUID,
    device_name: str = "NVIDIA GeForce RTX 5090",
    memory_uuid: str = FTR_PRIMARY_DEVICE_UUID,
) -> tuple[RuntimeEvidence, dict[str, object]]:
    identity = RuntimeIdentity(
        argv=("python", "runner.py", "--snapshot-child"),
        cwd="/snapshot",
        python_executable="/venv/bin/python",
        python_version="3.12",
        jax_version="0.7",
        jaxlib_version="0.7",
        simsopt_module_path="/snapshot/src/simsopt/__init__.py",
        simsopt_jax_module_path="/snapshot/src/simsopt_jax/__init__.py",
        native_extension_path="/snapshot/simsoptpp.so",
        backend="gpu",
        device_uuid=runtime_uuid,
        driver_version="test",
        effective_environment_sha256="0" * 64,
    )
    runtime = cast(
        "RuntimeEvidence",
        SimpleNamespace(
            observation=SimpleNamespace(
                runtime_identity=identity,
                device_name=device_name,
            )
        ),
    )
    memory: dict[str, object] = {
        "schema_version": "single-stage-fullspace-cfs-ftr1-gpu-memory-v1",
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "parent_pid": 1,
        "child_pid": 2,
        "child_start_time_ticks": 3,
        "child_argv_sha256": hashlib.sha256(
            canonical_json_bytes(list(identity.argv))
        ).hexdigest(),
        "device_uuid": memory_uuid,
        "sample_count": 1,
        "peak_memory_bytes": 1024,
        "peak_memory_fraction": 0.5,
    }
    return runtime, memory


def test_ftr_request_is_additive_and_exact() -> None:
    v1_before = contract_sha256_v1()
    v2_before = contract_sha256_v2()
    request = RunRequest(
        RunPhase.CANARY,
        FullSpaceRoute.CFS_FTR1,
        DeviceLane.RTX5090,
        10,
        None,
    )

    payload = run_request_payload_v3(request)

    assert run_request_v3_from_payload(payload["request"]) == request
    assert contract_sha256_v1() == v1_before
    assert contract_sha256_v2() == v2_before


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (
            RunRequest(
                RunPhase.CANARY,
                FullSpaceRoute.CFS_FTR1,
                DeviceLane.A100,
                10,
                None,
            ),
            "RTX 5090 ten-step",
        ),
        (
            RunRequest(
                RunPhase.CANARY,
                FullSpaceRoute.CFS_FTR1,
                DeviceLane.RTX5090,
                100,
                None,
            ),
            "RTX 5090 ten-step",
        ),
    ),
)
def test_ftr_request_rejects_non_gate2_shapes(
    replacement: RunRequest, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replacement.validate_v3()


def test_gate2_allows_iteration_limit_when_progress_and_certificates_pass() -> None:
    raw, memory = _passing_evidence()

    detail = derive_ftr_gate_detail(raw, memory)

    assert detail["gate_status"] == "PASS"
    assert detail["failure_reasons"] == []


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    (
        ("optimizer_result", "coordinate_count", 715, "LEDGER_OR_DTYPE"),
        ("optimizer_result", "joint_evaluations", 1201, "EVALUATION_BUDGET"),
        (
            "optimizer_result",
            "final_multiplier_projection_relative_residual",
            1.1e-10,
            "MULTIPLIER_PROJECTION_RESIDUAL",
        ),
        ("independent_kkt", "kkt_relative_residual", 1.1e-10, "KKT_RESIDUAL"),
        ("independent_kkt", "schur_relative_residual", 1.1e-10, "SCHUR_RESIDUAL"),
        (
            "endpoint",
            "raw_kkt_stationarity_infinity_norm",
            2.0e-2,
            "RAW_KKT_NOT_DECREASED",
        ),
        ("transfer_audit", "hot_d2h_calls", 1, "TRANSFER_BUDGET"),
        ("timing", "synchronized_solve_seconds", 29.0, "PROJECTED_TIME_EXCEEDED"),
    ),
)
def test_gate2_fails_closed_for_independent_tampering(
    section: str, key: str, value: object, reason: str
) -> None:
    raw, memory = _passing_evidence()
    tampered = deepcopy(raw)
    target = tampered[section]
    assert isinstance(target, dict)
    target[key] = value

    detail = derive_ftr_gate_detail(tampered, memory)

    assert detail["gate_status"] == "FAIL"
    assert reason in detail["failure_reasons"]


def test_gate2_memory_budget_fails_closed() -> None:
    raw, memory = _passing_evidence()
    memory["peak_memory_fraction"] = 0.8

    detail = derive_ftr_gate_detail(raw, memory)

    assert "MEMORY_BUDGET" in detail["failure_reasons"]


def test_gate2_device_binding_accepts_only_frozen_rtx5090() -> None:
    runtime, memory = _runtime_and_memory()

    validate_ftr_device_binding(runtime, cast("dict[str, JsonValue]", memory))


@pytest.mark.parametrize(
    ("runtime_uuid", "device_name", "memory_uuid"),
    (
        ("GPU-a100", "NVIDIA A100", FTR_PRIMARY_DEVICE_UUID),
        (FTR_PRIMARY_DEVICE_UUID, "NVIDIA A100", FTR_PRIMARY_DEVICE_UUID),
        (FTR_PRIMARY_DEVICE_UUID, "NVIDIA GeForce RTX 5090", "GPU-a100"),
    ),
)
def test_gate2_device_binding_rejects_wrong_or_crossed_device(
    runtime_uuid: str,
    device_name: str,
    memory_uuid: str,
) -> None:
    runtime, memory = _runtime_and_memory(
        runtime_uuid=runtime_uuid,
        device_name=device_name,
        memory_uuid=memory_uuid,
    )

    with pytest.raises(ValueError, match="RTX 5090|device differs"):
        validate_ftr_device_binding(runtime, cast("dict[str, JsonValue]", memory))
