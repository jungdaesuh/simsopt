from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from benchmarks.single_stage_compute_graph_c0_capture import (
    IDENTITY_ANCHOR_SCHEMA_ID,
)
from benchmarks.single_stage_compute_graph_c0_runner import (
    C0_RUNNER_SPEC_SCHEMA_ID,
    PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
    PROCESS_TREE_RSS_SOURCE,
    CommandResult,
    ProcessTreeRssEvidence,
)
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    ApiActivity,
    SqliteLaneEvidence,
    build_control_evidence,
)
from benchmarks.single_stage_compute_graph_phase0_post_gate import (
    PROFILE_MODULE,
    TELEMETRY_MODULE,
    Phase0PostGateError,
    assemble_phase0_receipt,
    collect_attribution_control_stage,
    collect_command_buffer_stage,
    collect_newton_telemetry_stage,
    collect_profile_stage,
    load_post_gate_context,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    HLO_MODULE_SET_IDENTITY_SOURCE,
    canonical_hlo_module_set_identity,
    canonical_json_bytes,
    canonical_sha256,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    RoleRoot,
    publish_immutable_snapshot,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rss_evidence() -> ProcessTreeRssEvidence:
    return ProcessTreeRssEvidence(
        peak_bytes=1_000_000,
        sample_count=2,
        sample_interval_ns=PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
        source=PROCESS_TREE_RSS_SOURCE,
        root_pid=123,
        root_starttime_ticks=456,
    )


def _snapshot(tmp_path: Path) -> Path:
    source = tmp_path / "snapshot-source"
    roles: list[RoleRoot] = []
    for role, relative in (
        ("execution_source", "src/simsopt_jax/__init__.py"),
        ("configuration", "inputs/configuration.json"),
        ("benchmark", f"{PROFILE_MODULE.replace('.', '/')}.py"),
        (
            "benchmark",
            "benchmarks/single_stage_compute_graph_command_buffer_control.py",
        ),
        ("benchmark", f"{TELEMETRY_MODULE.replace('.', '/')}.py"),
        ("test", "tests/test_evaluator.py"),
        ("native_extension", "src/simsoptpp.py"),
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"ROLE = {role!r}\n", encoding="utf-8")
        roles.append(RoleRoot(role, path, relative))
    snapshot = tmp_path / "snapshot"
    publish_immutable_snapshot(snapshot, tuple(roles))
    return snapshot


def _pending_spec(tmp_path: Path) -> Path:
    output_root = tmp_path / "c0"
    output_root.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    input_bundle = input_root / "input_bundle.json"
    input_bundle.write_bytes(canonical_json_bytes({"case": "native-default"}))
    candidate = tmp_path / "candidate.npy"
    candidate.write_bytes(b"candidate")
    native = tmp_path / "native-reference.json"
    native.write_bytes(canonical_json_bytes({"native": True}))
    specimen_sha = "a" * 64
    candidate_sha = "b" * 64
    source_sha = "c" * 64
    runtime_sha = "d" * 64
    gate = {
        "schema_id": "single-stage-compute-graph-c0-gate-checkpoint-v1",
        "state": "PASSED",
        "lane_id": "rtx5090",
        "gpu_uuid": "GPU-test",
        "specimen_sha256": specimen_sha,
        "input_bundle_sha256": _sha256(input_bundle),
        "parameter_sha256": candidate_sha,
        "initial_parameter_sha256": "9" * 64,
        "source_state_sha256": source_sha,
        "interpreter_path": sys.executable,
        "runtime_identity_sha256": runtime_sha,
        "native_reference_sha256": _sha256(native),
    }
    gate_path = output_root / "gate-checkpoint.json"
    gate_path.write_bytes(canonical_json_bytes(gate))
    warm = {
        "schema_id": "single-stage-compute-graph-c0-warm-checkpoint-v1",
        "state": "COMPLETE",
        "gate_checkpoint_sha256": _sha256(gate_path),
        **{
            field: gate[field]
            for field in (
                "lane_id",
                "gpu_uuid",
                "specimen_sha256",
                "input_bundle_sha256",
                "parameter_sha256",
                "source_state_sha256",
                "interpreter_path",
                "runtime_identity_sha256",
            )
        },
        "warm_measurement": {"p50_ns": 100.0},
    }
    warm_path = output_root / "warm-checkpoint.json"
    warm_path.write_bytes(canonical_json_bytes(warm))
    state = {
        "schema_id": "single-stage-compute-graph-c0-state-v1",
        "state": "POST_GATE_PENDING",
        "gate_checkpoint_sha256": _sha256(gate_path),
        "warm_checkpoint_sha256": _sha256(warm_path),
        "warm_p50_ns": 100.0,
        "lane_id": "rtx5090",
        "runtime_identity_sha256": runtime_sha,
    }
    (output_root / "state.json").write_bytes(canonical_json_bytes(state))
    spec = {
        "schema_id": C0_RUNNER_SPEC_SCHEMA_ID,
        "output_root": str(output_root),
        "input_root": str(input_root),
        "candidate_path": str(candidate),
        "native_reference_path": str(native),
        "lane_id": "rtx5090",
        "provenance": {
            "interpreter_path": sys.executable,
            "immutable_root": str(_snapshot(tmp_path)),
            "runtime": {"jax_backend": "gpu"},
            "environment": {
                "JAX_ENABLE_X64": "true",
                "JAX_PLATFORMS": "cuda",
                "JAX_TRANSFER_GUARD": "disallow",
            },
            "policies": {"quadrature_block_sizes": [128, 122]},
        },
        "receipt_template": {
            "specimen": {"input_bundle_sha256": _sha256(input_bundle)}
        },
    }
    spec_path = tmp_path / "c0-spec.json"
    spec_path.write_bytes(canonical_json_bytes(spec))
    return spec_path


def _trace(candidate_sha256: str) -> dict[str, object]:
    return {
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": True},
        "traceEvents": [
            {
                "ph": "M",
                "pid": 11,
                "name": "process_name",
                "args": {"name": "/host:CPU"},
            },
            {
                "ph": "M",
                "pid": 12,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            *(
                {
                    "ph": "X",
                    "pid": 11,
                    "tid": 1,
                    "ts": timestamp,
                    "dur": 1.0,
                    "name": f"optimizer.lifecycle.{event}",
                    "args": {
                        "evaluation_id": candidate_sha256,
                        "parameter_sha256": candidate_sha256,
                        "evaluation_kind": "trial",
                        "outer_iteration_id": None,
                    },
                }
                for event, timestamp in (
                    ("evaluator_entry", 10.0),
                    ("device_ready", 80.0),
                    ("evaluator_return", 90.0),
                )
            ),
            {
                "ph": "X",
                "pid": 11,
                "tid": 1,
                "ts": 20.0,
                "dur": 30.0,
                "name": "CommonPjRtLoadedExecutable::Execute (jit_forward)",
                "args": {
                    "name": "jit_forward",
                    "execution_mode": "command_buffer",
                },
            },
            {
                "ph": "X",
                "pid": 12,
                "tid": 1,
                "ts": 25.0,
                "dur": 20.0,
                "name": "jit_forward_kernel",
                "args": {
                    "context_id": "$$1",
                    "correlation_id": "forward",
                    "hlo_module": "jit_forward",
                    "hlo_op": "jit_forward/fusion",
                    "kernel_details": "regs:16",
                    "name": "jit(jit_forward)/newton.residual_jvp",
                    "scope_range_id": "forward",
                    "tf_op": "XlaModule:",
                },
            },
            {},
        ],
    }


def test_profile_stage_runs_isolated_profile_mode_and_reuses_valid_evidence(
    tmp_path: Path,
) -> None:
    context = load_post_gate_context(_pending_spec(tmp_path), base_environment={})
    calls: list[tuple[Sequence[str], Mapping[str, str], Path]] = []

    def executor(
        argv: Sequence[str],
        environment: Mapping[str, str],
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        del timeout_seconds
        calls.append((argv, environment, cwd))
        trace_root = Path(argv[argv.index("--trace-root") + 1])
        trace_root.mkdir(parents=True)
        with gzip.open(trace_root / "profile.trace.json.gz", "wb") as stream:
            stream.write(canonical_json_bytes(_trace(context.binding.candidate_sha256)))
        identity_anchor = Path(argv[argv.index("--identity-anchor") + 1])
        identity_anchor.write_bytes(
            canonical_json_bytes(
                {
                    "schema_id": IDENTITY_ANCHOR_SCHEMA_ID,
                    "hlo_module_set_identity": canonical_hlo_module_set_identity(
                        ("jit_forward",)
                    ),
                    "hlo_module_set_identity_source": (HLO_MODULE_SET_IDENTITY_SOURCE),
                }
            )
        )
        return CommandResult(
            0,
            json.dumps({"mode": "profile"}),
            "",
            10,
            process_tree_rss=_rss_evidence(),
        )

    evidence_path = collect_profile_stage(context, executor=executor)
    assert evidence_path.is_file()
    assert calls[0][1]["SINGLE_STAGE_COMPUTE_GRAPH_MODE"] == "profile"
    assert calls[0][1]["SINGLE_STAGE_COMPUTE_GRAPH_LANE"] == "rtx5090"
    assert calls[0][1]["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY"] == (
        context.binding.runtime_identity_sha256
    )
    assert calls[0][1]["TF_PROFILER_TRACE_VIEWER_MAX_EVENTS"] == "67108864"
    runtime_contract = json.loads(
        calls[0][1]["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"]
    )
    assert runtime_contract == {
        "runtime": {"jax_backend": "gpu"},
        "static_environment": {
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
        },
        "route_environment": {
            "JAX_COMPILATION_CACHE_DIR": str(context.paths.profile_cache_root),
            "SINGLE_STAGE_COMPUTE_GRAPH_LANE": "rtx5090",
            "SINGLE_STAGE_COMPUTE_GRAPH_MODE": "profile",
            "SINGLE_STAGE_COMPUTE_GRAPH_VARIANT": "C0",
        },
        "policies": {"quadrature_block_sizes": [128, 122]},
        "expected_runtime_identity_sha256": context.binding.runtime_identity_sha256,
    }
    assert calls[0][1]["JAX_COMPILATION_CACHE_DIR"] == str(
        context.paths.profile_cache_root
    )
    assert tuple(calls[0][0][1:3]) == ("-P", "-s")
    assert "benchmarks.single_stage_compute_graph_c0_evaluator" in calls[0][0]
    initial_flag_index = calls[0][0].index("--initial-parameter-sha256")
    assert calls[0][0][initial_flag_index + 1] == context.initial_parameter_sha256
    assert collect_profile_stage(context, executor=executor) == evidence_path
    assert len(calls) == 1
    evidence_path.unlink()
    assert collect_profile_stage(context, executor=executor) == evidence_path
    assert len(calls) == 1


def test_attribution_control_runs_six_isolated_attempts_and_validates_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = load_post_gate_context(_pending_spec(tmp_path), base_environment={})

    def runtime_identity(provenance: Mapping[str, object]) -> str:
        environment = provenance["environment"]
        assert isinstance(environment, dict)
        return (
            "e" * 64
            if environment.get("XLA_FLAGS") == "--xla_gpu_enable_command_buffer="
            else context.binding.runtime_identity_sha256
        )

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate._runtime_identity",
        runtime_identity,
    )
    calls: list[tuple[Sequence[str], Mapping[str, str], Path]] = []

    def executor(
        argv: Sequence[str],
        environment: Mapping[str, str],
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        del timeout_seconds
        calls.append((argv, environment, cwd))
        trace_root = Path(argv[argv.index("--trace-root") + 1])
        trace_root.mkdir(parents=True)
        with gzip.open(trace_root / "profile.trace.json.gz", "wb") as stream:
            stream.write(canonical_json_bytes(_trace(context.binding.candidate_sha256)))
        identity_anchor = Path(argv[argv.index("--identity-anchor") + 1])
        identity_anchor.write_bytes(
            canonical_json_bytes(
                {
                    "schema_id": IDENTITY_ANCHOR_SCHEMA_ID,
                    "hlo_module_set_identity": canonical_hlo_module_set_identity(
                        ("jit_forward",)
                    ),
                    "hlo_module_set_identity_source": HLO_MODULE_SET_IDENTITY_SOURCE,
                }
            )
        )
        observation = {
            "mode": "profile",
            "objective": 1.0,
            "gradient": [1.0, 2.0],
            "inner_newton_success": True,
            "adjoint_success": True,
            "residual_certificates": {"boozer": 1e-13, "adjoint": 1e-12},
        }
        return CommandResult(
            0,
            json.dumps(observation),
            "",
            10,
            process_tree_rss=_rss_evidence(),
        )

    path = collect_attribution_control_stage(context, executor=executor)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["state"] == "PRODUCED"
    assert document["promotion_eligible"] is True
    assert document["equivalence"]["module_topology_claim"] == (
        "exact_hlo_module_name_set_and_frozen_solver_specimen_only"
    )
    first_attempt = document["direct_default_measurement"]["attempts"][0]
    assert first_attempt["profile_derivation_version"] == (
        "compute-graph-profile-attribution-v1"
    )
    assert first_attempt["raw_trace_path"].startswith(
        "post-gate/attribution-control/default_control/attempt-00/"
    )
    assert len(first_attempt["raw_trace_sha256"]) == 64
    assert len(calls) == 6
    assert all(
        call[0][call[0].index("--initial-parameter-sha256") + 1]
        == context.initial_parameter_sha256
        for call in calls
    )
    cache_roots = {call[1]["JAX_COMPILATION_CACHE_DIR"] for call in calls}
    assert len(cache_roots) == 6
    assert (
        sum(
            call[1].get("XLA_FLAGS") == "--xla_gpu_enable_command_buffer="
            for call in calls
        )
        == 3
    )
    assert collect_attribution_control_stage(context, executor=executor) == path
    assert len(calls) == 6

    child = (
        context.paths.attribution_root
        / "command_buffer_disabled"
        / "attempt-01"
        / "child-observation.json"
    )
    tampered = json.loads(child.read_text(encoding="utf-8"))
    tampered["objective"] = 2.0
    child.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(Phase0PostGateError, match="differs from raw attempts"):
        collect_attribution_control_stage(context, executor=executor)


def test_command_buffer_stage_uses_existing_planner_executor_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    context = load_post_gate_context(_pending_spec(tmp_path), base_environment={})
    nsys = tmp_path / "nsys"
    nsys.write_bytes(b"nsys")
    nsys.chmod(0o755)
    nvtx_library = tmp_path / "libnvToolsExt.so.1"
    nvtx_library.write_bytes(b"nvtx")
    calls = 0

    def executor(plan, output_path: Path) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        for lane in plan.lanes:
            lane.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            lane.sqlite_path.write_bytes(b"raw")
        default = SqliteLaneEvidence(
            0,
            100,
            ApiActivity(1, 1, "api_calls"),
            ApiActivity(1, 2, "api_calls"),
            ApiActivity(0, 0, "api_calls"),
            ApiActivity(1, 50, "device_activity_records"),
            ApiActivity(1, 20, "device_activity_records"),
            ApiActivity(2, 70, "device_activity_records"),
            0,
        )
        disabled = SqliteLaneEvidence(
            0,
            100,
            ApiActivity(0, 0, "api_calls"),
            ApiActivity(0, 0, "api_calls"),
            ApiActivity(0, 0, "api_calls"),
            ApiActivity(0, 0, "device_activity_records"),
            ApiActivity(2, 60, "device_activity_records"),
            ApiActivity(2, 60, "device_activity_records"),
            0,
        )
        document = build_control_evidence(
            plan,
            nsys_version=plan.expected_nsys_version,
            default_evidence=default,
            disabled_evidence=disabled,
            default_wall_ns=101,
            disabled_wall_ns=102,
        )
        for lane in plan.lanes:
            lane.sqlite_path.unlink()
        output_path.write_bytes(canonical_json_bytes(document))
        return document

    path = collect_command_buffer_stage(
        context,
        nsys_binary=nsys,
        nvtx_library=nvtx_library,
        expected_nsys_version="Nsight test",
        current_xla_flags="--xla_gpu_triton_gemm_any=true",
        executor=executor,
    )
    assert (
        collect_command_buffer_stage(
            context,
            nsys_binary=nsys,
            nvtx_library=nvtx_library,
            expected_nsys_version="Nsight test",
            current_xla_flags="--xla_gpu_triton_gemm_any=true",
            executor=executor,
        )
        == path
    )
    assert calls == 1
    document = json.loads(path.read_text(encoding="utf-8"))
    document["identity"]["warm_checkpoint_sha256"] = "0" * 64
    path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(RuntimeError, match="identity"):
        collect_command_buffer_stage(
            context,
            nsys_binary=nsys,
            nvtx_library=nvtx_library,
            expected_nsys_version="Nsight test",
            current_xla_flags="--xla_gpu_triton_gemm_any=true",
            executor=executor,
        )


def test_pending_checkpoint_tamper_is_rejected_before_any_stage(tmp_path: Path) -> None:
    spec_path = _pending_spec(tmp_path)
    state_path = tmp_path / "c0" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["warm_checkpoint_sha256"] = "0" * 64
    state_path.write_bytes(canonical_json_bytes(state))
    with pytest.raises(Phase0PostGateError, match="checkpoint hashes"):
        load_post_gate_context(spec_path)


def test_newton_telemetry_stage_uses_shared_isolated_launcher_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    context = load_post_gate_context(_pending_spec(tmp_path), base_environment={})
    calls: list[tuple[Sequence[str], Mapping[str, str], Path]] = []

    def executor(
        argv: Sequence[str],
        environment: Mapping[str, str],
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        del timeout_seconds
        calls.append((argv, environment, cwd))
        output = Path(argv[argv.index("--output") + 1])
        equality = {
            "objective_exact": True,
            "raw_objective_exact": True,
            "gradient_exact": True,
            "solved_state_exact": True,
            "newton_success_exact": True,
            "newton_iterations_exact": True,
        }
        document = {
            "schema_id": "single-stage-compute-graph-newton-telemetry-v2",
            "state": "PRODUCED",
            "evidence_kind": ("observer_bearing_exact_newton_outside_promotion_timing"),
            "identity": context.telemetry_identity.to_json(),
            "route_id": "production-exact-newton",
            "warmup_executions_per_lane": 1,
            "numerical_equality": equality,
            "observer": {
                "api": "device_resident_fixed_shape_exact_newton_counts",
                "device_resident_fixed_shape_counts": True,
                "host_callback_used": False,
                "promotion_timing_included": False,
            },
            "newton_telemetry": {
                "telemetry_schema_id": (
                    "single-stage-compute-graph-newton-telemetry-v2"
                ),
                "route_id": "production-exact-newton",
                "measurement_method": (
                    "device_resident_fixed_shape_exact_newton_counts"
                ),
                "host_callback_used": False,
                "residual_evaluations": 7,
                "linear_operator_applications": 5,
                "observed_wall_ns": 120,
                "unobserved_wall_ns": 100,
                "observer_effect_ratio": 1.2,
                "collected_outside_timed_samples": True,
            },
        }
        document["newton_telemetry"]["raw_evidence_sha256"] = canonical_sha256(document)
        output.write_bytes(canonical_json_bytes(document))
        return CommandResult(0, "", "", 120)

    path = collect_newton_telemetry_stage(context, executor=executor)
    assert "benchmarks.single_stage_compute_graph_newton_telemetry" in calls[0][0]
    assert calls[0][1]["JAX_COMPILATION_CACHE_DIR"] == str(
        context.paths.telemetry_cache_root
    )
    assert collect_newton_telemetry_stage(context, executor=executor) == path
    assert len(calls) == 1
    document = json.loads(path.read_text(encoding="utf-8"))
    document["identity"]["candidate_sha256"] = "0" * 64
    path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(RuntimeError, match="identity"):
        collect_newton_telemetry_stage(context, executor=executor)


def test_final_assembly_publishes_canonical_resume_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = load_post_gate_context(_pending_spec(tmp_path), base_environment={})
    for path in (
        context.paths.attribution_evidence_path,
        context.paths.command_buffer_evidence_path,
        context.paths.newton_telemetry_path,
        context.paths.complete_path_evidence_path,
        context.paths.gap_inputs_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes({}))
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate.parse_profile_evidence",
        lambda path, expected_identity: object(),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate.require_promoting_attribution_evidence",
        lambda document: None,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate.validate_command_buffer_control_evidence",
        lambda document, expected_identity: {},
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate.validate_newton_telemetry_evidence",
        lambda document, expected_identity: {},
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate._validate_complete_path_identity",
        lambda document, context: None,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate.validate_gap_budget_inputs_artifact",
        lambda document, complete: {},
    )
    received: list[Mapping[str, object]] = []

    def assembler(spec: Mapping[str, object]) -> Mapping[str, object]:
        received.append(spec)
        return {"state": "assembled"}

    result = assemble_phase0_receipt(context, assembler=assembler)
    assert result == {"state": "assembled"}
    assert received[0]["resume"] == {
        "gate_checkpoint_path": str(tmp_path / "c0" / "gate-checkpoint.json"),
        "warm_checkpoint_path": str(tmp_path / "c0" / "warm-checkpoint.json"),
        "profile_evidence_path": str(context.paths.profile_evidence_path),
        "attribution_control_evidence_path": str(
            context.paths.attribution_evidence_path
        ),
        "command_buffer_evidence_path": str(context.paths.command_buffer_evidence_path),
        "newton_telemetry_evidence_path": str(context.paths.newton_telemetry_path),
        "complete_path_evidence_path": str(context.paths.complete_path_evidence_path),
        "gap_budget_inputs_path": str(context.paths.gap_inputs_path),
    }
    persisted = context.paths.resume_spec_path.read_bytes()
    assert persisted == canonical_json_bytes(received[0])


def test_final_assembly_refuses_explicitly_non_promoting_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = load_post_gate_context(_pending_spec(tmp_path), base_environment={})
    context.paths.attribution_evidence_path.parent.mkdir(parents=True)
    context.paths.attribution_evidence_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_id": "single-stage-compute-graph-attribution-evidence-v4",
                "state": "NON_PROMOTING",
                "promotion_eligible": False,
                "blockers": ["module_topology_identity_mismatch"],
                "selected_attribution": None,
            }
        )
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_phase0_post_gate.parse_profile_evidence",
        lambda path, expected_identity: object(),
    )

    with pytest.raises(Phase0PostGateError, match="explicitly non-promoting"):
        assemble_phase0_receipt(context, assembler=lambda spec: spec)
