from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import run_single_stage_fullspace_gpu as runner
from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryResult,
    ProcessGpuMemorySample,
)
from benchmarks.single_stage_fullspace_process_gpu_monitor import (
    LinuxProcessIdentity,
    bound_gpu_memory_payload,
    read_linux_process_identity,
)
from benchmarks.single_stage_fullspace_receipt import (
    SCHEMA_VERSION_V2,
    SQP_MEMORY_SCHEMA_VERSION,
    SQP_R2_PLAN_SHA256,
    CompleteSample,
    DeviceLane,
    RunPhase,
    RunRequest,
    canonical_json_bytes,
)
from simsopt_jax.solve.fullspace import FullSpaceRoute


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--phase",
        "first-eval",
        "--route",
        "CFS-SQP1",
        "--device",
        "rtx5090",
        "--output",
        str(tmp_path / "campaign"),
        *extra,
    ]


@pytest.mark.parametrize(
    ("phase", "steps", "sample", "maximum_iterations", "relative"),
    (
        (RunPhase.FIRST_EVAL, None, None, None, "gates/derivative"),
        (RunPhase.CANARY, 1, None, 1, "gates/canary-1"),
        (RunPhase.CANARY, 10, None, 10, "gates/canary-10"),
        (RunPhase.COMPLETE, None, CompleteSample.COLD, 100, "samples/cold"),
        (
            RunPhase.COMPLETE,
            None,
            CompleteSample.WARM_3,
            100,
            "samples/warm-3",
        ),
    ),
)
def test_sqp_seams_have_exhaustive_iteration_and_directory_routing(
    phase: RunPhase,
    steps: int | None,
    sample: CompleteSample | None,
    maximum_iterations: int | None,
    relative: str,
) -> None:
    request = RunRequest(
        phase,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        steps,
        sample,
    )
    if maximum_iterations is None:
        with pytest.raises(ValueError, match="no optimizer iteration"):
            runner._sqp_maximum_iterations(request)
    else:
        assert runner._sqp_maximum_iterations(request) == maximum_iterations
    assert runner._sqp_run_relative_directory(request) == Path(relative)


def test_sqp_preflight_is_v2_and_does_not_require_cuda_or_new_output(
    tmp_path: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    output = tmp_path / "campaign"
    output.mkdir()
    assert runner.main(_argv(tmp_path, "--preflight-only")) == 0
    payload = json.loads(capsysbinary.readouterr().out)
    assert payload["schema_version"] == SCHEMA_VERSION_V2
    assert payload["request"]["route"] == "CFS-SQP1"


def test_sqp_main_dispatches_directly_without_al_fallthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    observed: list[RunRequest] = []

    def fake_sqp(request: RunRequest, *_args: object, **_kwargs: object) -> bytes:
        observed.append(request)
        return b'{"route":"CFS-SQP1"}'

    monkeypatch.setattr(runner, "run_cfs_sqp1_campaign", fake_sqp)
    monkeypatch.setattr(runner.simsoptpp, "__file__", "/tmp/simsoptpp.so")
    monkeypatch.setattr(
        runner,
        "run_cfs_al2_campaign",
        lambda *_args, **_kwargs: pytest.fail("SQP fell through to AL2"),
    )
    assert runner.main(_argv(tmp_path)) == 0
    assert observed[0].route is FullSpaceRoute.CFS_SQP1
    assert capsysbinary.readouterr().out == b'{"route":"CFS-SQP1"}'


def test_first_eval_dispatches_derivative_gate_without_optimizer_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = RunRequest(
        RunPhase.FIRST_EVAL,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        None,
        None,
    )
    bootstrap = SimpleNamespace()
    monkeypatch.setattr(
        runner,
        "run_cfs_sqp1_derivative_gate",
        lambda value: (
            {"optimizer_steps_executed": 0, "bootstrap_matches": value is bootstrap},
            {"synchronized_derivative_kkt_seconds": 1.0},
            {"hot_h2d_calls": 0},
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_cfs_sqp1_probe",
        lambda *_args, **_kwargs: pytest.fail(
            "FIRST_EVAL dispatched an optimizer solve"
        ),
    )
    payload = runner._cfs_sqp1_child_probe_payload(request, bootstrap)
    assert payload["schema_version"] == runner.CFS_SQP1_DERIVATIVE_GATE_SCHEMA_VERSION
    assert payload["derivative_kkt_gate"] == {
        "optimizer_steps_executed": 0,
        "bootstrap_matches": True,
    }


@pytest.mark.parametrize(
    ("maximum_iterations", "changed_expected"), ((1, True), (10, False), (100, False))
)
def test_sqp_probe_selects_changed_state_only_for_one_step_canary(
    maximum_iterations: int,
    changed_expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_state = object()
    changed_state = object()
    problem = SimpleNamespace()
    observed: list[object] = []
    monkeypatch.setattr(
        runner,
        "_deterministic_cfs_sqp1_changed_state",
        lambda z0, candidate_problem: (
            changed_state
            if z0 is bootstrap_state and candidate_problem is problem
            else pytest.fail("changed-state inputs differ")
        ),
    )
    monkeypatch.setattr(
        runner,
        "prepare_cfs_sqp1",
        lambda candidate_problem, z0, initial, *, maximum_iterations: (
            observed.append(initial),
            SimpleNamespace(),
        )[1],
    )
    runner._prepare_cfs_sqp1_probe(
        problem,
        bootstrap_state,
        maximum_iterations=maximum_iterations,
    )
    assert observed == [changed_state if changed_expected else bootstrap_state]


def test_nonfinite_selected_regularization_serializes_as_null() -> None:
    assert runner._optional_finite_float(np.asarray(np.nan)) is None


def test_derivative_gate_kkt_uses_changed_state_linearization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = SimpleNamespace(
        constraint_jacobian=np.asarray([[2.0, 3.0]]),
        objective_gradient=np.asarray([5.0, 7.0]),
        scaled_constraints=np.asarray([11.0]),
    )
    policy = SimpleNamespace(
        regularization_ladder=(0.0,),
        kkt_relative_residual_tolerance=1.0e-10,
        schur_relative_residual_tolerance=1.0e-10,
        kkt_forward_error_tolerance=1.0e-7,
        kkt_solution_scaled_residual_tolerance=1.0e-10,
    )
    identity = np.eye(2)
    observed: dict[str, object] = {}
    sentinel = object()

    def fake_solve(
        bfgs: object,
        jacobian: object,
        dual_residual: object,
        constraints: object,
        **kwargs: object,
    ) -> object:
        observed.update(
            bfgs=bfgs,
            jacobian=jacobian,
            dual_residual=dual_residual,
            constraints=constraints,
            kwargs=kwargs,
        )
        return sentinel

    monkeypatch.setattr(runner, "solve_dense_sqp_kkt", fake_solve)

    result, dual_residual = runner._solve_changed_state_gate_kkt(
        changed, identity, policy
    )

    assert result is sentinel
    np.testing.assert_array_equal(dual_residual, [5.0, 7.0])
    np.testing.assert_array_equal(observed["bfgs"], identity)
    np.testing.assert_array_equal(observed["jacobian"], changed.constraint_jacobian)
    np.testing.assert_array_equal(observed["dual_residual"], [5.0, 7.0])
    np.testing.assert_array_equal(observed["constraints"], [11.0])


def _fake_derivative_state(*, numerical_rank: int = 255) -> dict[str, object]:
    return {
        "physical_objective": np.asarray(1.0),
        "scaled_constraints": np.zeros((255,)),
        "objective_gradient": np.zeros((716,)),
        "constraint_jacobian": np.zeros((255, 716)),
        "joint_vjp_rows": np.zeros((256, 716)),
        "all_finite": np.asarray(True),
        "singular_values": np.ones((255,)),
        "numerical_rank": np.asarray(numerical_rank),
        "rank_cutoff": np.asarray(1.0e-12),
        "av": np.zeros((255,)),
        "atw": np.zeros((716,)),
        "transpose_lhs": np.asarray(0.0),
        "transpose_rhs": np.asarray(0.0),
        "transpose_absolute_error": np.asarray(0.0),
        "transpose_relative_error": np.asarray(0.0),
    }


def _fake_derivative_device_evidence(
    *, numerical_rank: int = 255, kkt_valid: bool = True
) -> dict[str, object]:
    kkt_scalar = 0.0 if kkt_valid else np.nan
    return {
        "bootstrap": _fake_derivative_state(numerical_rank=numerical_rank),
        "changed": _fake_derivative_state(),
        "changed_physical_state": np.zeros((716,)),
        "changed_optimizer_coordinates": np.zeros((716,)),
        "kkt": {
            "primal_step": np.zeros((716,)) if kkt_valid else np.full((716,), np.nan),
            "multiplier_step": (
                np.zeros((255,)) if kkt_valid else np.full((255,), np.nan)
            ),
            "valid": np.asarray(kkt_valid),
            "selected_regularization": np.asarray(kkt_scalar),
            "rho_k": np.asarray(0.01 if kkt_valid else np.nan),
            "zeta_2": np.asarray(1.0e-12 if kkt_valid else np.nan),
            "kkt_relative_residual": np.asarray(kkt_scalar),
            "schur_relative_residual": np.asarray(kkt_scalar),
            "bfgs_cholesky_relative_pivot": np.asarray(1.0 if kkt_valid else np.nan),
            "schur_cholesky_relative_pivot": np.asarray(1.0 if kkt_valid else np.nan),
            "regularization_candidates_tested": np.asarray(1),
            "reconstructed_residual_inf": np.asarray(kkt_scalar),
            "reconstructed_residual_two": np.asarray(kkt_scalar),
            "certified_relative_error_bound": np.asarray(
                1.0e-10 if kkt_valid else np.nan
            ),
            "all_finite": np.asarray(kkt_valid),
        },
    }


def _run_fake_derivative_gate(
    monkeypatch: pytest.MonkeyPatch, evidence: dict[str, object]
) -> dict[str, object]:
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: (object(),))
    monkeypatch.setattr(runner.jax, "device_get", lambda value: value)
    monkeypatch.setattr(runner.jax, "device_put", lambda value, *, device: value)
    monkeypatch.setattr(runner.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(runner.jax, "transfer_guard", lambda _mode: nullcontext())
    monkeypatch.setattr(runner.jax, "jit", lambda _function: lambda *_args: evidence)
    monkeypatch.setattr(runner, "_array_tree_size", lambda _value: (1, 8))
    gate, _timing, _transfers = runner.run_cfs_sqp1_derivative_gate(
        SimpleNamespace(z0=np.zeros((716,)), problem=np.zeros((1,)))
    )
    return gate


def test_derivative_gate_retains_rank_failure_as_canonical_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _run_fake_derivative_gate(
        monkeypatch, _fake_derivative_device_evidence(numerical_rank=254)
    )
    assert gate["gate_status"] == "FAIL"
    assert "BOOTSTRAP_RANK" in gate["failure_reasons"]
    canonical_json_bytes(gate)


def test_derivative_gate_retains_invalid_kkt_as_null_canonical_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _run_fake_derivative_gate(
        monkeypatch, _fake_derivative_device_evidence(kkt_valid=False)
    )
    assert gate["gate_status"] == "FAIL"
    assert "KKT_INVALID" in gate["failure_reasons"]
    assert gate["kkt"]["selected_regularization"] is None
    assert gate["kkt"]["primal_step"] is None
    canonical_json_bytes(gate)


def test_physical_gpu_memory_query_has_one_nvidia_smi_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_run(argv: tuple[str, ...], **_kwargs: object) -> CompletedProcess[str]:
        observed.append(argv)
        return CompletedProcess(argv, 0, "0, GPU-test, 1024\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner._physical_gpu_memory_identity({}) == (
        "GPU-test",
        1024 * 1024 * 1024,
    )
    assert observed == [
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        )
    ]


def test_linux_process_identity_handles_parentheses_and_spaces(tmp_path: Path) -> None:
    process_root = tmp_path / "37"
    process_root.mkdir()
    fields_after_comm = ["S", *[str(index) for index in range(1, 20)]]
    fields_after_comm[19] = "987654"
    (process_root / "stat").write_text(
        f"37 (python worker) {' '.join(fields_after_comm)}\n", encoding="utf-8"
    )
    argv = ("/usr/bin/python", "-I", "/tmp/runner with space.py")
    (process_root / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv))
    assert read_linux_process_identity(37, proc_root=tmp_path) == LinuxProcessIdentity(
        37, 987654, argv
    )


def test_bound_memory_payload_matches_exact_validator_schema() -> None:
    runtime_argv = ("/snapshot/runner.py", "--phase", "complete")
    monitor = SimpleNamespace(
        identity=SimpleNamespace(pid=41, start_ticks=1234, argv=("python", "runner")),
        gpu_uuid="GPU-test",
    )
    measurement = ProcessGpuMemoryResult(
        gpu_uuid="GPU-test",
        provider_pid=41,
        samples=(ProcessGpuMemorySample(100, 256),),
        peak_used_memory_mib=256,
    )
    payload = bound_gpu_memory_payload(
        monitor,
        measurement,
        parent_pid=7,
        physical_device_memory_bytes=1024 * 1024 * 1024,
        runtime_argv=runtime_argv,
    )
    assert set(payload) == {
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
    }
    assert payload["schema_version"] == SQP_MEMORY_SCHEMA_VERSION
    assert (
        payload["child_argv_sha256"]
        == hashlib.sha256(canonical_json_bytes(list(runtime_argv))).hexdigest()
    )
    assert payload["peak_memory_fraction"] == 0.25


def test_prepare_or_load_reuses_the_existing_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "campaign"
    snapshot = campaign / runner.SNAPSHOT_DIRECTORY
    snapshot.mkdir(parents=True)
    publication = SimpleNamespace(root=snapshot)
    monkeypatch.setattr(runner, "load_snapshot", lambda path: publication)
    monkeypatch.setattr(
        runner,
        "prepare_execution_snapshot",
        lambda *_args, **_kwargs: pytest.fail("continuation republished snapshot"),
    )
    assert (
        runner.prepare_or_load_execution_snapshot(
            campaign, native_extension_path=tmp_path / "native.so"
        )
        is publication
    )


def _persist_passed_gate(
    campaign: Path,
    request: RunRequest,
) -> None:
    run_root = campaign / runner._sqp_run_relative_directory(request)
    raw: dict[str, object] = {
        "contract_sha256": runner.contract_sha256_v2(),
        "plan_sha256": runner.SQP_PLAN_SHA256,
        "budget_sha256": runner.SQP_BUDGET_SHA256,
        "request": {
            "phase": request.phase.value,
            "route": request.route.value,
            "device": request.device.value,
            "steps": request.steps,
            "sample": None,
        },
        "source_identity": {"identity": "test"},
        "runtime_evidence": {"identity": "test"},
        "bootstrap_artifact": {"identity": "test"},
    }
    raw_schema = runner.SQP_RESULT_SCHEMA_VERSION
    if request.phase is RunPhase.FIRST_EVAL:
        raw_schema = runner.CFS_SQP1_DERIVATIVE_GATE_SCHEMA_VERSION
        raw["derivative_kkt_gate"] = {
            "gate_status": "PASS",
            "failure_reasons": [],
        }
    else:
        assert request.steps in (1, 10)
        raw.update(
            {
                "optimizer_result": {
                    "all_finite": True,
                    "all_accepted_states_finite": True,
                    "fatal": False,
                    "failed": False,
                    "iterations": request.steps,
                    "history": {"accepted_length": request.steps},
                    "final_kkt_reciprocal_condition": 0.01,
                    "final_kkt_solution_scaled_residual": 1.0e-12,
                    "final_kkt_relative_residual": 1.0e-12,
                    "final_schur_relative_residual": 1.0e-12,
                    "initial_physical_objective": 2.0,
                    "initial_scaled_constraint_infinity_norm": 2.0e-10,
                    "initial_raw_kkt_stationarity_infinity_norm": 2.0e-6,
                },
                "endpoint": {
                    "all_finite": True,
                    "physical_objective": 1.0,
                    "scaled_constraint_infinity_norm": 1.0e-10,
                    "raw_kkt_stationarity_infinity_norm": 1.0e-6,
                },
                "timing": {"synchronized_solve_seconds": 1.0},
                "transfer_audit": {
                    "hot_h2d_calls": 0,
                    "hot_d2h_calls": 0,
                    "initial_h2d_calls": 1,
                    "final_d2h_calls": 1,
                },
            }
        )
        if request.steps == 10:
            raw_optimizer = raw["optimizer_result"]
            assert isinstance(raw_optimizer, dict)
            raw_optimizer["convergence_telemetry"] = {
                "merit": [1.0] * 10,
                "penalty": [2.0] * 10,
                "multiplier_update_infinity_norm": [0.5] * 10,
                "bfgs_reset": [0] * 10,
                "restoration_applied": [1] * 10,
                "restoration_numerical_failures": 0,
            }
            raw_optimizer["restoration_numerical_failures"] = 0
    raw["schema_version"] = raw_schema
    memory: dict[str, object] = {
        "schema_version": SQP_MEMORY_SCHEMA_VERSION,
        "peak_memory_fraction": 0.25,
    }
    raw_ref = runner._publish_immutable_json(
        run_root / "raw-result.json",
        raw,
        campaign_root=campaign,
        schema_version=raw_schema,
    )
    memory_ref = runner._publish_immutable_json(
        run_root / "gpu-memory.json",
        memory,
        campaign_root=campaign,
        schema_version=SQP_MEMORY_SCHEMA_VERSION,
    )
    gate_receipt = runner._sqp_gate_receipt_payload(
        request,
        raw,
        raw_ref,
        memory,
        memory_ref,
    )
    runner._publish_immutable_json(
        run_root / "gate-receipt.json",
        gate_receipt,
        campaign_root=campaign,
        schema_version=str(gate_receipt["schema_version"]),
    )


def test_sqp_revision3_authorizes_only_standalone_ten_step_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    derivative = RunRequest(
        RunPhase.FIRST_EVAL,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        None,
        None,
    )
    canary_1 = RunRequest(
        RunPhase.CANARY,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        1,
        None,
    )
    canary_10 = RunRequest(
        RunPhase.CANARY,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        10,
        None,
    )
    cold = RunRequest(
        RunPhase.COMPLETE,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        None,
        CompleteSample.COLD,
    )
    validated_gates: list[runner.SqpGate] = []

    def validate_gate(
        _campaign_root: Path,
        gate: runner.SqpGate,
        _artifact: object,
    ) -> object:
        validated_gates.append(gate)
        return SimpleNamespace(passed=True, failure_reasons=())

    monkeypatch.setattr(runner, "load_sqp_gate_result", validate_gate)

    with pytest.raises(runner.PhaseGateError, match="prohibits"):
        runner._enforce_sqp_prerequisite_chain(derivative, campaign)
    with pytest.raises(runner.PhaseGateError, match="prohibits"):
        runner._enforce_sqp_prerequisite_chain(canary_1, campaign)
    runner._enforce_sqp_prerequisite_chain(canary_10, campaign)
    with pytest.raises(runner.PhaseGateError, match="missing SQP prerequisite"):
        runner._enforce_sqp_prerequisite_chain(cold, campaign)
    _persist_passed_gate(campaign, canary_10)
    runner._enforce_sqp_prerequisite_chain(cold, campaign)
    assert validated_gates == [runner.SqpGate.CANARY_10]


def test_sqp_gate_reader_binds_raw_revision_identity(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    request = RunRequest(
        RunPhase.CANARY,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        10,
        None,
    )
    _persist_passed_gate(campaign, request)
    run_root = campaign / runner._sqp_run_relative_directory(request)
    raw_path = run_root / "raw-result.json"
    gate_path = run_root / "gate-receipt.json"

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["plan_sha256"] = SQP_R2_PLAN_SHA256
    raw_bytes = canonical_json_bytes(raw)
    raw_path.chmod(0o644)
    raw_path.write_bytes(raw_bytes)
    raw_path.chmod(0o444)

    gate_receipt = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_receipt["raw_result"]["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    gate_receipt["raw_result"]["size_bytes"] = len(raw_bytes)
    gate_bytes = canonical_json_bytes(gate_receipt)
    gate_path.chmod(0o644)
    gate_path.write_bytes(gate_bytes)
    gate_path.chmod(0o444)
    artifact = runner.ArtifactRef(
        relative_path=gate_path.relative_to(campaign).as_posix(),
        sha256=hashlib.sha256(gate_bytes).hexdigest(),
        size_bytes=len(gate_bytes),
        schema_version=runner.CFS_SQP1_CANARY_10_GATE_SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="raw plan_sha256 identity differ"):
        runner.load_sqp_gate_result(
            campaign,
            runner.SqpGate.CANARY_10,
            artifact,
        )


def test_complete_sqp_refuses_before_child_without_endpoint_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = RunRequest(
        RunPhase.COMPLETE,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        None,
        CompleteSample.COLD,
    )
    monkeypatch.setattr(
        runner,
        "_enforce_sqp_prerequisite_chain",
        lambda candidate, root: None,
    )
    monkeypatch.setattr(
        runner,
        "prepare_or_load_execution_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "snapshot prepared before authority gate"
        ),
    )

    with pytest.raises(
        runner.PhaseGateError,
        match="endpoint-audit authority producer",
    ):
        runner.run_cfs_sqp1_campaign(
            request,
            tmp_path / "campaign",
            native_extension_path=tmp_path / "native.so",
            interpreter=Path("/usr/bin/python3"),
            environment={},
        )
