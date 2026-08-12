from __future__ import annotations

import hashlib
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest
from benchmarks.single_stage_compute_graph_c0_runner import (
    PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
    PROCESS_TREE_RSS_SOURCE,
    CommandResult,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CANARY_ARTIFACT_SCHEMA_ID,
    CanarySpec,
    build_artifact,
    run_canary,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    RoleRoot,
    publish_immutable_snapshot,
)


def _digest(character: str) -> str:
    return character * 64


def _spec(tmp_path: Path) -> CanarySpec:
    native = tmp_path / "native.json"
    native.write_text(
        '{"gradient":[' + ",".join("0.0" for _ in range(461)) + '],"objective":1.0}',
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.npy"
    candidate.write_bytes(b"candidate")
    return CanarySpec(
        variant="C1",
        solver_graph_sha256=_digest("1"),
        source_state_sha256=_digest("2"),
        specimen_sha256=_digest("3"),
        candidate_file_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        parameter_sha256=_digest("8"),
        device_identity_sha256=_digest("4"),
        gpu_uuid="GPU-test",
        c0_gate_checkpoint_sha256=_digest("5"),
        c0_warm_checkpoint_sha256=_digest("6"),
        native_reference_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        runtime_identity_sha256=_digest("7"),
        input_root=tmp_path,
        candidate_path=candidate,
        native_reference_path=native,
        snapshot_root=tmp_path,
        interpreter_path=Path(sys.executable),
        cache_directory=tmp_path / "cache-c1",
        output_root=tmp_path / "artifact-c1",
        c0_p50_ns=200.0,
        c0_p95_ns=200.0,
        c0_peak_rss_bytes=20,
        c0_peak_gpu_memory_bytes=40,
        runtime_contract_json='{"policies":{},"route_environment":{},"runtime":{},"static_environment":{}}',
        native_reference={"native_objective": 1.0, "native_gradient": [0.0] * 461},
        native_initial_reference={
            "native_objective": 1.0,
            "native_gradient": [0.0] * 461,
            "parameter_sha256": _digest("9"),
        },
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _runnable_spec(tmp_path: Path) -> CanarySpec:
    spec = _spec(tmp_path)
    source = tmp_path / "snapshot-source"
    execution = _write(
        source / "execution" / "simsopt_jax" / "__init__.py", "\n"
    ).parent
    configuration = _write(source / "configuration" / "input.json", "{}\n")
    benchmark_root = source / "benchmark"
    _write(benchmark_root / "__init__.py", "\n")
    _write(
        benchmark_root / "single_stage_compute_graph_canary_evaluator.py",
        "raise SystemExit(0)\n",
    )
    test = _write(source / "test" / "test_canary.py", "def test_canary(): pass\n")
    native = _write(source / "native" / "simsoptpp.py", "NATIVE = True\n")
    snapshot = tmp_path / "snapshot"
    publication = publish_immutable_snapshot(
        snapshot,
        (
            RoleRoot("execution_source", execution, "src/simsopt_jax"),
            RoleRoot("configuration", configuration, "inputs/input.json"),
            RoleRoot("benchmark", benchmark_root, "benchmarks"),
            RoleRoot("test", test, "tests/test_canary.py"),
            RoleRoot("native_extension", native, "src/simsoptpp.py"),
        ),
    )
    return replace(
        spec,
        snapshot_root=snapshot,
        snapshot_manifest_sha256=publication.manifest_sha256,
    )


def _valid_telemetry() -> dict[str, int | bool]:
    return {
        "exact_newton_variant_dense_linearization_used": True,
        "exact_newton_variant_linear_solve_attempt_count": 1,
        "exact_newton_variant_dense_materialization_count": 1,
        "exact_newton_variant_lu_factorization_count": 1,
        "exact_newton_variant_lu_solve_count": 1,
        "exact_newton_variant_refinement_correction_count": 0,
        "exact_newton_variant_stop_reason_code": 0,
        "exact_newton_variant_numerical_failure": False,
        "exact_newton_variant_backtracking_iteration_count": 1,
        "exact_newton_variant_stalled": False,
        "exact_newton_variant_retry_linear_solve_at_strict_cap": False,
    }


def _memory() -> dict[str, object]:
    return {
        "peak_self_rss_bytes": 10,
        "peak_process_tree_rss_bytes": 10,
        "process_tree_rss_sample_count": 2,
        "process_tree_rss_sample_interval_ns": PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
        "process_tree_rss_source": PROCESS_TREE_RSS_SOURCE,
        "process_tree_rss_root_pid": 123,
        "process_tree_rss_root_starttime_ticks": 456,
        "gpu_memory": {
            "provider_pid": 12,
            "gpu_uuid": "GPU-test",
            "sample_count": 2,
            "sample_interval_ns": 10_000_000,
            "peak_bytes": 20,
            "source": "nvidia-smi_direct_pid_gpu_uuid",
        },
    }


def _gate_observation() -> dict[str, object]:
    return {
        "schema_id": "single-stage-compute-graph-canary-child-v1",
        "status": "PASS",
        "variant": "C1",
        "mode": "gate",
        "sample_index": None,
        "wall_ns": 100,
        "objective_dtype": "float64",
        "objective": 1.0,
        "gradient_dtype": "float64",
        "gradient": [0.0] * 461,
        "inner_newton_success": True,
        "adjoint_success": True,
        "residual_certificates": {"boozer_exact_residual_l2": 1.0e-13},
        "telemetry": _valid_telemetry(),
        **_memory(),
    }


def _initial_gate_observation() -> dict[str, object]:
    return {
        **_gate_observation(),
        "mode": "initial_gate",
        "parameter_sha256": _digest("9"),
    }


def _warm_observation(index: int) -> dict[str, object]:
    return {
        "schema_id": "single-stage-compute-graph-canary-child-v1",
        "status": "PASS",
        "variant": "C1",
        "mode": "warm",
        "sample_index": index,
        "wall_ns": 100 + index,
        "telemetry": _valid_telemetry(),
        **_memory(),
    }


def test_incomplete_warm_route_is_machine_readable_blocked(tmp_path: Path) -> None:
    artifact = build_artifact(_spec(tmp_path), ())

    assert artifact["schema_id"] == CANARY_ARTIFACT_SCHEMA_ID
    assert artifact["status"] == "BLOCKED"
    assert artifact["blocker"]["code"] == "INCOMPLETE_WARM_ROUTE"  # type: ignore[index]
    assert artifact["warm_measurement"] is None
    assert artifact["identity"]["c0_gate_checkpoint_sha256"] == _digest("5")  # type: ignore[index]
    assert artifact["identity"]["c0_warm_checkpoint_sha256"] == _digest("6")  # type: ignore[index]


def test_child_blocker_prevents_warm_promotion(tmp_path: Path) -> None:
    artifact = build_artifact(
        _spec(tmp_path),
        (
            {
                "status": "BLOCKED",
                "blocker": {"code": "NO_TELEMETRY", "reason": "missing"},
            },
        ),
    )

    assert artifact["status"] == "BLOCKED"
    assert artifact["blocker"] == {"code": "NO_TELEMETRY", "reason": "missing"}
    assert "gate" not in artifact


def test_complete_gate_and_ten_warms_remain_nonpromoting_without_trajectory(
    tmp_path: Path,
) -> None:
    telemetry = {
        "exact_newton_variant_dense_linearization_used": True,
        "exact_newton_variant_linear_solve_attempt_count": 1,
        "exact_newton_variant_dense_materialization_count": 1,
        "exact_newton_variant_lu_factorization_count": 1,
        "exact_newton_variant_lu_solve_count": 1,
        "exact_newton_variant_refinement_correction_count": 0,
        "exact_newton_variant_stop_reason_code": 0,
        "exact_newton_variant_numerical_failure": False,
        "exact_newton_variant_backtracking_iteration_count": 1,
        "exact_newton_variant_stalled": False,
        "exact_newton_variant_retry_linear_solve_at_strict_cap": False,
    }
    gate = {
        "schema_id": "single-stage-compute-graph-canary-child-v1",
        "status": "PASS",
        "variant": "C1",
        "mode": "gate",
        "objective_dtype": "float64",
        "objective": 1.0,
        "gradient_dtype": "float64",
        "gradient": [0.0] * 461,
        "inner_newton_success": True,
        "adjoint_success": True,
        "residual_certificates": {"boozer_exact_residual_l2": 1.0e-13},
        "telemetry": telemetry,
        **_memory(),
    }
    warms = tuple(
        {
            "schema_id": "single-stage-compute-graph-canary-child-v1",
            "status": "PASS",
            "variant": "C1",
            "mode": "warm",
            "sample_index": index,
            "wall_ns": 100 + index,
            "telemetry": telemetry,
            **_memory(),
        }
        for index in range(10)
    )

    initial_gate = {
        **gate,
        "mode": "initial_gate",
        "parameter_sha256": _digest("9"),
    }
    artifact = build_artifact(_spec(tmp_path), (initial_gate, gate, *warms))

    assert artifact["status"] == "MEASURED_NONPROMOTING"
    assert artifact["performance_passed"] is True
    assert (
        artifact["performance_gates"][  # type: ignore[index]
            "process_tree_rss_evidence_available"
        ]
        is True
    )
    assert artifact["promotion_blocker"]["code"] == (  # type: ignore[index]
        "PROMOTION_FINALIZER_REQUIRED"
    )
    assert artifact["warm_measurement"]["sample_count"] == 10  # type: ignore[index]
    assert artifact["warm_measurement"]["p50_wall_ns"] == 104.5  # type: ignore[index]
    assert artifact["warm_measurement"]["p95_wall_ns"] == 109  # type: ignore[index]


def test_direct_api_has_no_caller_controlled_trajectory_pass_flag(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)

    assert "trajectory_gate_passed" not in {field.name for field in fields(CanarySpec)}
    with pytest.raises(AttributeError):
        object.__setattr__(spec, "trajectory_gate_passed", True)


def test_base_builder_cannot_emit_promotable_status(tmp_path: Path) -> None:
    artifact = build_artifact(
        _spec(tmp_path),
        (
            _initial_gate_observation(),
            _gate_observation(),
            *(_warm_observation(index) for index in range(10)),
        ),
    )

    assert artifact["performance_passed"] is True
    assert artifact["status"] == "MEASURED_NONPROMOTING"
    assert artifact["promotion_blocker"]["code"] == "PROMOTION_FINALIZER_REQUIRED"  # type: ignore[index]


def test_artifact_compares_matched_process_tree_rss_scope(tmp_path: Path) -> None:
    warm = [_warm_observation(index) for index in range(10)]
    for observation in warm:
        observation["peak_process_tree_rss_bytes"] = 23

    artifact = build_artifact(
        _spec(tmp_path), (_initial_gate_observation(), _gate_observation(), *warm)
    )

    gates = artifact["performance_gates"]
    assert gates["process_tree_rss_evidence_available"] is True  # type: ignore[index]
    assert (
        gates["peak_process_tree_rss_at_most_10_percent_regression"]  # type: ignore[index]
        is False
    )
    assert artifact["performance_passed"] is False


def test_gate_timeout_writes_exclusive_raw_and_terminal_blocked(
    tmp_path: Path,
) -> None:
    spec = _runnable_spec(tmp_path)
    calls = 0

    def timeout_executor(argv, environment, cwd, timeout_seconds):
        nonlocal calls
        del argv, environment, cwd
        calls += 1
        assert timeout_seconds == 900.0
        return CommandResult(124, "", "timeout", 900_000_000_000, timed_out=True)

    artifact = run_canary(spec, executor=timeout_executor, environment={})

    assert calls == 1
    assert artifact["status"] == "BLOCKED"
    assert "timed out" in artifact["blocker"]["reason"]  # type: ignore[index]
    assert (spec.output_root / "children/initial_gate/raw.json").is_file()
    assert (spec.output_root / "state-terminal.json").is_file()
    assert (spec.output_root / "canary.json").is_file()


def test_snapshot_drift_blocks_before_child_executor(tmp_path: Path) -> None:
    spec = _runnable_spec(tmp_path)
    evaluator = (
        spec.snapshot_root / "benchmarks/single_stage_compute_graph_canary_evaluator.py"
    )
    evaluator.chmod(0o644)
    evaluator.write_text("print('drift')\n", encoding="utf-8")
    calls = 0

    def forbidden_executor(argv, environment, cwd, timeout_seconds):
        nonlocal calls
        del argv, environment, cwd, timeout_seconds
        calls += 1
        raise AssertionError("drifted snapshot must block before execution")

    artifact = run_canary(spec, executor=forbidden_executor, environment={})

    assert calls == 0
    assert artifact["status"] == "BLOCKED"
    assert "snapshot is invalid" in artifact["blocker"]["reason"]  # type: ignore[index]
