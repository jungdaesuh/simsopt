from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.single_stage_compute_graph_c0_runner import CommandResult
from benchmarks.single_stage_compute_graph_canary_profile_runner import (
    PROFILE_COUNT_SCHEMA_ID,
    PROFILE_SCHEMA_ID,
    CanaryProfileRunnerError,
    build_profile_artifact,
    build_profile_launch,
    execute_profile_launch,
    validate_profile_count_evidence,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CanarySpec,
    _spec_identity,
)
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    ApiActivity,
    SqliteLaneEvidence,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    HLO_MODULE_SET_IDENTITY_SOURCE,
)
from benchmarks.single_stage_compute_graph_profile import ComputeGraphProfile
from benchmarks.single_stage_compute_graph_snapshot import (
    RoleRoot,
    publish_immutable_snapshot,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import Interval


def _spec(tmp_path: Path) -> CanarySpec:
    return CanarySpec(
        variant="C1",
        solver_graph_sha256="1" * 64,
        source_state_sha256="2" * 64,
        specimen_sha256="3" * 64,
        candidate_file_sha256="4" * 64,
        parameter_sha256="5" * 64,
        device_identity_sha256="6" * 64,
        gpu_uuid="GPU-test",
        c0_gate_checkpoint_sha256="7" * 64,
        c0_warm_checkpoint_sha256="8" * 64,
        native_reference_sha256="9" * 64,
        runtime_identity_sha256="a" * 64,
        input_root=tmp_path,
        candidate_path=tmp_path / "candidate.npy",
        native_reference_path=tmp_path / "native.json",
        snapshot_root=tmp_path,
        interpreter_path=Path("/usr/bin/python3"),
        cache_directory=tmp_path / "cache",
        output_root=tmp_path / "canary",
        native_reference={"native_objective": 1.0, "native_gradient": [0.0] * 461},
        native_initial_reference={"parameter_sha256": "e" * 64},
    )


def _telemetry() -> dict[str, int | bool]:
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


def _numerical() -> dict[str, object]:
    return {
        "objective_dtype": "float64",
        "objective": 1.0,
        "gradient_dtype": "float64",
        "gradient": [0.0] * 461,
        "inner_newton_success": True,
        "adjoint_success": True,
        "residual_certificates": {"residual": 0.0},
    }


def _profile() -> ComputeGraphProfile:
    intervals = (Interval(10, 20),)
    return ComputeGraphProfile(
        evaluation_envelope_ns=100,
        device_active_ns=10,
        phase_interval_unions=(("newton.residual_jvp", intervals),),
        attributed_union_ns=10,
        unattributed_ns=0,
        attribution_coverage=1.0,
        pjrt_execute_count=3,
        kernel_launch_count=4,
        kernel_duration_ns=(1, 2, 3, 4),
        inter_launch_gap_ns=2,
        hlo_module_set_identity="b" * 64,
        hlo_module_set_identity_source=HLO_MODULE_SET_IDENTITY_SOURCE,
        device_active_share=0.1,
        inter_launch_gap_share=0.2,
        command_buffer=None,
        command_buffer_unavailable_reason="not classified",
    )


def _nsys() -> SqliteLaneEvidence:
    zero = ApiActivity(0, 0)
    total = ApiActivity(4, 10)
    return SqliteLaneEvidence(1, 101, zero, zero, zero, zero, total, total)


def _files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    values = []
    for name in (
        "trace.json.gz",
        "report.nsys-rep",
        "report.sqlite",
        "nsys",
        "nvtx.so",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        values.append(path)
    return tuple(values)  # type: ignore[return-value]


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _manifested_spec(tmp_path: Path) -> CanarySpec:
    source = tmp_path / "source"
    execution = _write(source / "execution/simsopt_jax/__init__.py", "\n").parent
    configuration = _write(source / "configuration/input.json", "{}\n")
    benchmark = source / "benchmark"
    _write(benchmark / "__init__.py", "\n")
    _write(benchmark / "single_stage_compute_graph_canary_evaluator.py", "\n")
    _write(benchmark / "single_stage_compute_graph_canary_profile.py", "\n")
    test = _write(source / "test/test_profile.py", "def test_profile(): pass\n")
    native = _write(source / "native/simsoptpp.py", "NATIVE = True\n")
    snapshot = tmp_path / "snapshot"
    publication = publish_immutable_snapshot(
        snapshot,
        (
            RoleRoot("execution_source", execution, "src/simsopt_jax"),
            RoleRoot("configuration", configuration, "inputs/input.json"),
            RoleRoot("benchmark", benchmark, "benchmarks"),
            RoleRoot("test", test, "tests/test_profile.py"),
            RoleRoot("native_extension", native, "src/simsoptpp.py"),
        ),
    )
    return replace(
        _spec(tmp_path),
        snapshot_root=snapshot,
        interpreter_path=Path(sys.executable),
        snapshot_manifest_sha256=publication.manifest_sha256,
        runtime_contract_json=(
            '{"policies":{},"route_environment":{},"runtime":{},'
            '"static_environment":{}}'
        ),
    )


def test_profile_launch_is_exact_manifested_nsys_command(tmp_path: Path) -> None:
    spec = _manifested_spec(tmp_path)
    nsys = _write(tmp_path / "nsys", "#!/bin/sh\n")
    nsys.chmod(0o755)
    nvtx = _write(tmp_path / "libnvtx.so", "library")

    launch = build_profile_launch(
        spec,
        nsys_binary=nsys,
        nvtx_library=nvtx,
        output_root=tmp_path / "profile-output",
        base_environment={},
    )

    assert launch.command[0] == str(nsys.resolve())
    assert "--trace=cuda,nvtx" in launch.command
    assert "benchmarks.single_stage_compute_graph_canary_profile" in launch.command
    assert "--variant" in launch.command
    assert "C1" in launch.command
    initial_index = launch.command.index("--initial-parameter-sha256")
    assert launch.command[initial_index + 1] == "e" * 64


def test_profile_launch_has_fixed_900_second_timeout(tmp_path: Path) -> None:
    spec = _manifested_spec(tmp_path)
    nsys = _write(tmp_path / "nsys", "#!/bin/sh\n")
    nsys.chmod(0o755)
    nvtx = _write(tmp_path / "libnvtx.so", "library")
    launch = build_profile_launch(
        spec,
        nsys_binary=nsys,
        nvtx_library=nvtx,
        output_root=tmp_path / "profile-output",
        base_environment={},
    )

    def executor(argv, environment, cwd, timeout_seconds):
        assert tuple(argv) == launch.command
        assert environment == launch.environment
        assert cwd == launch.cwd
        assert timeout_seconds == 900.0
        return CommandResult(124, "", "timeout", 900_000_000_000, timed_out=True)

    result = execute_profile_launch(launch, executor)
    assert result.timed_out is True


def test_separate_count_evidence_is_bound_to_canary_bytes(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    document = {
        "schema_id": PROFILE_COUNT_SCHEMA_ID,
        "identity": {
            **_spec_identity(spec),
            "canary_artifact_sha256": "c" * 64,
        },
        "counts": {
            "residual_evaluation_count": 3,
            "dense_primal_traversal_count": 1,
            "dense_tangent_batch_count": 4,
            "dense_tangent_direction_count": 255,
        },
    }

    assert (
        validate_profile_count_evidence(
            document, spec=spec, canary_artifact_sha256="c" * 64
        )["dense_tangent_direction_count"]
        == 255
    )
    with pytest.raises(CanaryProfileRunnerError, match="identity differs"):
        validate_profile_count_evidence(
            document, spec=spec, canary_artifact_sha256="d" * 64
        )


def test_artifact_fails_closed_and_names_missing_source_hooks(
    monkeypatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    gate = {"status": "PASS", "mode": "gate", "variant": "C1", **_numerical()}
    child = {
        "status": "PASS",
        "mode": "profile",
        "variant": "C1",
        "parameter_sha256": spec.parameter_sha256,
        **_numerical(),
        "telemetry": _telemetry(),
        "capture": {
            "hlo_ir_sha256": "d" * 64,
            "hlo_module_set_identity": "b" * 64,
            "hlo_modules": ["jit_graph"],
            "pjrt_execute_count": 3,
            "kernel_launch_count": 4,
        },
    }
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile_runner._gate_parity",
        lambda *_args, **_kwargs: None,
    )
    trace, report, sqlite, nsys_binary, nvtx = _files(tmp_path)
    artifact = build_profile_artifact(
        spec=spec,
        canary_artifact={
            "status": "MEASURED_NONPROMOTING",
            "identity": _spec_identity(spec),
            "gate": gate,
        },
        canary_artifact_sha256="c" * 64,
        child=child,
        profile=_profile(),
        nsys=_nsys(),
        trace_path=trace,
        report_path=report,
        sqlite_path=sqlite,
        nsys_binary=nsys_binary,
        nsys_version="2026.1",
        nvtx_library=nvtx,
        profile_counts={
            "residual_evaluation_count": 3,
            "dense_primal_traversal_count": 1,
            "dense_tangent_batch_count": 4,
            "dense_tangent_direction_count": 255,
        },
    )

    assert artifact["schema_id"] == PROFILE_SCHEMA_ID
    assert artifact["status"] == "BLOCKED"
    missing = artifact["missing_required_source_hooks"]
    assert {row["field"] for row in missing} >= {  # type: ignore[union-attr]
        "lu_factorization.device_interval_union_ns",
    }
    assert artifact["required_operations"]["residual"]["count"] == 3  # type: ignore[index]
    assert artifact["newton_only"]["device_interval_union_ns"] == 10  # type: ignore[index]


def test_profile_numerical_drift_is_rejected(monkeypatch, tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    gate = {"status": "PASS", "mode": "gate", "variant": "C1", **_numerical()}
    child = {
        "status": "PASS",
        "mode": "profile",
        "variant": "C1",
        "parameter_sha256": spec.parameter_sha256,
        **_numerical(),
        "telemetry": _telemetry(),
        "capture": {},
    }
    child["objective"] = 2.0
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile_runner._gate_parity",
        lambda *_args, **_kwargs: None,
    )
    trace, report, sqlite, nsys_binary, nvtx = _files(tmp_path)
    with pytest.raises(CanaryProfileRunnerError, match="differs"):
        build_profile_artifact(
            spec=spec,
            canary_artifact={
                "status": "MEASURED_NONPROMOTING",
                "identity": _spec_identity(spec),
                "gate": gate,
            },
            canary_artifact_sha256="c" * 64,
            child=child,
            profile=_profile(),
            nsys=_nsys(),
            trace_path=trace,
            report_path=report,
            sqlite_path=sqlite,
            nsys_binary=nsys_binary,
            nsys_version="2026.1",
            nvtx_library=nvtx,
        )
