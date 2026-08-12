from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from benchmarks import single_stage_compute_graph_complete_path as complete_path
from benchmarks.single_stage_compute_graph_attribution_control import (
    AttributionAttempt,
    AttributionBinding,
    AttributionControlError,
    build_attribution_evidence,
    canonical_module_topology_identity,
)
from benchmarks.single_stage_compute_graph_c0_runner import (
    C0_CHILD_OBSERVATION_SCHEMA_ID,
    C0_RUNNER_SPEC_SCHEMA_ID,
    EVALUATOR_MODULE,
    FAILURE_FILENAME,
    PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
    PROCESS_TREE_RSS_SOURCE,
    RECEIPT_FILENAME,
    C0RunnerError,
    CommandResult,
    ProcessTreeRssEvidence,
    _compute_gap_budget,
    _linux_process_tree_rss_bytes,
    _resume_c0_measurement,
    _runtime_identity,
    _subprocess_executor,
    run_c0_measurement,
)
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    build_control_evidence,
    build_control_plan,
    parse_nsys_sqlite,
)
from benchmarks.single_stage_compute_graph_complete_path import (
    CompletePathBinding,
    FaithfulLever,
    GapBudgetPolicyInput,
    PhaseReductionAssumption,
    ProtocolSample,
    build_complete_path_document,
    build_complete_path_plan,
    build_gap_budget_inputs_artifact,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    ISOLATED_MODULE_BOOTSTRAP,
)
from benchmarks.single_stage_compute_graph_newton_telemetry import (
    CandidateEvaluation,
    ExecutionCounts,
    TelemetryIdentity,
    collect_newton_telemetry,
    write_newton_telemetry,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    A100_LANE_ID,
    FORMAL_COMPLETE_PATH_FACTOR,
    LANE_AGGREGATION_POLICY,
    PHASE0_SCHEMA_ID,
    RTX_LANE_ID,
    SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    canonical_json_bytes,
    canonical_sha256,
    load_phase0_receipt,
    validate_phase0_receipt,
)
from benchmarks.single_stage_compute_graph_profile import (
    build_attribution_control_profile_evidence,
    build_profile_evidence,
    write_profile_evidence,
)


def _digest(character: str) -> str:
    return character * 64


def _manifest() -> dict[str, object]:
    return {
        "schema_id": "single-stage-compute-graph-source-manifest-v1",
        "entries": [
            {
                "role": "execution_source",
                "relative_path": "src/simsopt_jax/solver.py",
                "size_bytes": 123,
                "sha256": _digest("1"),
            }
        ],
    }


def _specimen() -> dict[str, object]:
    return {
        "specimen_id": "native-default-changed-state-0",
        "input_bundle_sha256": _digest("2"),
        "parameter_sha256": _digest("3"),
        "state_dimension": 255,
        "coil_dof_count": 461,
        "grids": {
            "inner_surface_points": 169,
            "non_qs_surface_points": 1600,
            "quadrature_nodes": 250,
            "physical_coil_contributions": 18,
        },
        "weights": {"iota": 1.0},
        "tolerances": {"newton": 1e-12},
        "solver_graph_id": "exact-boozer-newton-direct-adjoint",
        "solver_graph_sha256": _digest("4"),
    }


def _qualification(lane_id: str, *, blocked: bool = False) -> dict[str, object]:
    checks = {
        "source_snapshot",
        "import_bindings",
        "native_extension",
        "device_identity",
        "runtime_backend",
        "fp64_policy",
        "cpu_affinity",
        "strict_transfer_smoke",
    }
    if lane_id == A100_LANE_ID:
        checks |= {
            "slurm_allocation",
            "cuda_12_6_compatibility",
            "dependency_overlay",
            "resolved_cuda_libraries",
        }
    failed_check = "slurm_allocation"
    return {
        "outcome": "blocked" if blocked else "qualified",
        "attempted_identity": {"hostname": "host", "requested_device": lane_id},
        "checks": [
            {
                "check_id": check_id,
                "passed": not (blocked and check_id == failed_check),
                "evidence": f"evidence for {check_id}",
            }
            for check_id in sorted(checks)
        ],
        "blocker": (
            {
                "code": "NO_SLURM",
                "check_id": failed_check,
                "reason": "no allocation",
                "evidence_sha256": _digest("5"),
            }
            if blocked
            else None
        ),
    }


def _provenance(cache: Path) -> dict[str, object]:
    manifest = _manifest()
    return {
        "repository_commit": _digest("6"),
        "source_state_sha256": _digest("7"),
        "git_status_short": [" M src/simsopt_jax/solver.py"],
        "tracked_diff_sha256": _digest("8"),
        "untracked_manifest_sha256": _digest("9"),
        "immutable_root": "/immutable/rtx5090",
        "immutable_tree_sha256": _digest("a"),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "interpreter_path": sys.executable,
        "runtime": {
            "python_version": "3.13.5",
            "jax_version": "0.10.2",
            "jaxlib_version": "0.10.2",
            "cuda_runtime": "12.9",
            "cuda_driver": "570.0",
            "jax_backend": "gpu",
            "fp64_x64_enabled": True,
            "resolved_cuda_libraries": ["libcuda.so.1"],
        },
        "allocation": {
            "hostname": "playstation",
            "scheduler": "local",
            "allocation_id": "local",
            "job_id": "local",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_uuid": "GPU-rtx5090",
            "gpu_memory_bytes": 32_000_000_000,
            "cpu_affinity": "0-31",
            "cuda_compatibility_version": "native",
            "cuda_compatibility_path": "not-applicable",
        },
        "import_bindings": {
            package: {"path": f"/immutable/rtx5090/{package}", "sha256": _digest(char)}
            for package, char in (
                ("simsopt", "b"),
                ("simsopt_jax", "c"),
                ("simsopt_jax_adapters", "d"),
                ("simsoptpp", "e"),
            )
        },
        "package_overlay": {"lineax": "0.1.1"},
        "environment": {"JAX_ENABLE_X64": "1"},
        "policies": {
            "dense_batch_width": 8,
            "point_chunk_size": None,
            "coil_chunk_size": None,
            "quadrature_block_sizes": [128, 122],
        },
        "compilation_cache_directory": str(cache.resolve()),
    }


def _profile() -> dict[str, object]:
    return {
        "evaluation_envelope_ns": 1100,
        "device_active_ns": 1000,
        "phase_interval_unions": [
            {"phase_id": "newton.residual_jvp", "intervals": [[0, 920]]}
        ],
        "attributed_union_ns": 920,
        "unattributed_ns": 80,
        "attribution_coverage": 0.92,
        "pjrt_execute_count": 11,
        "kernel_launch_count": 40,
        "kernel_duration_ns": [10, 20],
        "inter_launch_gap_ns": 25,
    }


def _gap_budget() -> dict[str, object]:
    conservative = 0.5 * 0.1
    optimistic = 0.5 * 0.2 + 0.08 * 0.5
    candidate_c0 = 95.5
    complete_c0 = 955.0
    candidate_conservative = candidate_c0 * (1.0 - conservative)
    candidate_optimistic = candidate_c0 * (1.0 - optimistic)
    return {
        "candidate_value_and_gradient_reference_timings_ns": {
            "c0_warm_p50": candidate_c0
        },
        "matched_complete_path_reference_timings_ns": {
            "native": 800.0,
            "c0": complete_c0,
            "optax": 900.0,
        },
        "formal_target_factor": FORMAL_COMPLETE_PATH_FACTOR,
        "formal_target_ns": FORMAL_COMPLETE_PATH_FACTOR * 800.0,
        "projection_method": (
            "candidate_value_and_gradient_savings_subtracted_from_matched_c0_complete_path"
        ),
        "candidate_phases": [
            {
                "phase_id": "newton.residual_jvp",
                "measured_share": 0.5,
                "conservative_reduction": 0.1,
                "optimistic_reduction": 0.2,
                "overlap_disposition": "disjoint",
            }
        ],
        "unattributed_share": 0.08,
        "unattributed_conservative_reduction": 0.0,
        "unattributed_optimistic_reduction": 0.5,
        "candidate_value_and_gradient_conservative_projected_ns": (
            candidate_conservative
        ),
        "candidate_value_and_gradient_optimistic_projected_ns": candidate_optimistic,
        "conservative_complete_path_projected_ns": complete_c0
        - (candidate_c0 - candidate_conservative),
        "optimistic_complete_path_projected_ns": complete_c0
        - (candidate_c0 - candidate_optimistic),
        "faithful_levers": [
            {
                "lever_id": "dense_newton",
                "disposition": "bounded",
                "evidence_sha256": _digest("f"),
            }
        ],
        "all_faithful_levers_bounded": True,
        "target_reachable_optimistically": False,
        "pivot_fired": True,
    }


def _spec(tmp_path: Path) -> dict[str, object]:
    specimen = _specimen()
    cache = tmp_path / "cache-rtx5090"
    provenance = _provenance(cache)
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True)
    input_bundle = {
        "input_fingerprint": _digest("5"),
        "configuration_fingerprint": _digest("6"),
    }
    input_bundle_bytes = canonical_json_bytes(input_bundle)
    (input_root / "input_bundle.json").write_bytes(input_bundle_bytes)
    specimen["input_bundle_sha256"] = hashlib.sha256(input_bundle_bytes).hexdigest()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot_files = {
        f"{EVALUATOR_MODULE.replace('.', '/')}.py": (
            "benchmark",
            b"benchmark\n",
        ),
        "benchmarks/single_stage_compute_graph_command_buffer_control.py": (
            "benchmark",
            b"command buffer control\n",
        ),
        "configuration.json": ("configuration", b"{}\n"),
        "execution.py": ("execution_source", b"execution\n"),
        "native.so": ("native_extension", b"native\n"),
        "test.py": ("test", b"test\n"),
    }
    entries = []
    for relative_path, (role, payload) in snapshot_files.items():
        snapshot_path = snapshot / relative_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(payload)
        entries.append(
            {
                "role": role,
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (snapshot / "phase0-source-manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_id": "single-stage-compute-graph-source-manifest-v1",
                "entries": entries,
            }
        )
    )
    provenance["immutable_root"] = str(snapshot.resolve())
    candidate = tmp_path / "candidate.npy"
    candidate.write_bytes(b"candidate")
    native_reference = tmp_path / "native-reference.json"
    native_reference.write_text(
        json.dumps(
            {
                "schema_id": "single-stage-compute-graph-native-reference-v3",
                "parameter_sha256": specimen["parameter_sha256"],
                "identity": {
                    "input_bundle_sha256": specimen["input_bundle_sha256"],
                    "input_fingerprint": input_bundle["input_fingerprint"],
                    "configuration_fingerprint": input_bundle[
                        "configuration_fingerprint"
                    ],
                    "specimen_sha256": canonical_sha256(specimen),
                    "source_sha256": provenance["source_state_sha256"],
                    "interpreter_path": provenance["interpreter_path"],
                    "native_simsoptpp_path": provenance["import_bindings"]["simsoptpp"][
                        "path"
                    ],
                    "native_simsoptpp_sha256": provenance["import_bindings"][
                        "simsoptpp"
                    ]["sha256"],
                    "runtime_identity_sha256": _runtime_identity(provenance),
                },
                "objective_dtype": "float64",
                "objective": 1.25,
                "gradient_dtype": "float64",
                "gradient": [index / 1000.0 for index in range(461)],
                "inner_newton_success": True,
                "residual_certificates": {
                    "solver_residual_l2": 1.0e-13,
                    "solver_residual_inf": 1.0e-13,
                },
                "elapsed_ns": 1,
                "initial_evaluation": {
                    "parameter_sha256": _digest("a"),
                    "objective_dtype": "float64",
                    "objective": 1.25,
                    "gradient_dtype": "float64",
                    "gradient": [index / 1000.0 for index in range(461)],
                    "inner_newton_success": True,
                    "residual_certificates": {
                        "solver_residual_l2": 1.0e-13,
                        "solver_residual_inf": 1.0e-13,
                    },
                    "elapsed_ns": 1,
                },
                "baseline_anchor": {
                    "parameter_sha256": _digest("a"),
                    "surface_sha256": _digest("b"),
                    "inner_solver_success": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema_id": C0_RUNNER_SPEC_SCHEMA_ID,
        "lane_id": RTX_LANE_ID,
        "output_root": str(tmp_path / "artifact-rtx5090"),
        "warm_sample_count": 10,
        "input_root": str(input_root),
        "candidate_path": str(candidate),
        "native_reference_path": str(native_reference),
        "receipt_template": {
            "schema_id": PHASE0_SCHEMA_ID,
            "artifact_id": "phase0-test",
            "evidence_kind": "compute_graph_engineering_phase0_noncampaign",
            "lane_aggregation_policy": LANE_AGGREGATION_POLICY,
            "specimen": specimen,
            "specimen_sha256": canonical_sha256(specimen),
            "lanes": [
                {
                    "lane_id": RTX_LANE_ID,
                    "device_class": "NVIDIA GeForce RTX 5090",
                    "qualification": _qualification(RTX_LANE_ID),
                    "measurement": None,
                },
                {
                    "lane_id": A100_LANE_ID,
                    "device_class": "NVIDIA A100",
                    "qualification": _qualification(A100_LANE_ID, blocked=True),
                    "measurement": None,
                },
            ],
        },
        "provenance": provenance,
    }


def _first_observation(*, gradient_count: int = 461) -> dict[str, object]:
    return {
        "schema_id": C0_CHILD_OBSERVATION_SCHEMA_ID,
        "mode": "gate",
        "parameter_sha256": _digest("3"),
        "objective_dtype": "float64",
        "objective": 1.25,
        "gradient_dtype": "float64",
        "gradient": [index / 1000.0 for index in range(gradient_count)],
        "inner_newton_success": True,
        "adjoint_success": True,
        "residual_certificates": {"boozer": 1e-13, "adjoint": 1e-12},
        "peak_self_rss_bytes": 1_000_000,
        "sampled_process_gpu_memory_peak_bytes": 2_000_000,
        "sampled_process_gpu_memory_source": SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    }


def _initial_observation(*, gradient_count: int = 461) -> dict[str, object]:
    observation = _first_observation(gradient_count=gradient_count)
    observation["mode"] = "initial-gate"
    observation["parameter_sha256"] = _digest("a")
    return observation


def _warm_observation(index: int) -> dict[str, object]:
    return {
        "schema_id": C0_CHILD_OBSERVATION_SCHEMA_ID,
        "mode": "warm",
        "sample_index": index,
        "wall_ns": 91 + index,
        "peak_self_rss_bytes": 1_000_000 + index,
        "sampled_process_gpu_memory_peak_bytes": 2_000_000 + index,
        "sampled_process_gpu_memory_source": SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    }


class FakeExecutor:
    def __init__(
        self,
        first: Mapping[str, object],
        *,
        initial: Mapping[str, object] | None = None,
        timeout: bool = False,
        initial_elapsed_ns: int = 5_000_000,
        gate_elapsed_ns: int = 5_000_000,
    ) -> None:
        self.first = first
        self.initial = _initial_observation() if initial is None else initial
        self.timeout = timeout
        self.initial_elapsed_ns = initial_elapsed_ns
        self.gate_elapsed_ns = gate_elapsed_ns
        self.calls: list[tuple[tuple[str, ...], dict[str, str], Path, float]] = []

    def __call__(
        self,
        argv: Sequence[str],
        environment: Mapping[str, str],
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        self.calls.append((tuple(argv), dict(environment), cwd, timeout_seconds))
        if len(self.calls) == 1:
            if self.timeout:
                return CommandResult(124, "", "timeout", 900_000_000_001, True)
            observation = self.initial
        elif len(self.calls) == 2:
            observation = self.first
        else:
            observation = _warm_observation(len(self.calls) - 3)
        if len(self.calls) == 1:
            elapsed_ns = self.initial_elapsed_ns
        elif len(self.calls) == 2:
            elapsed_ns = self.gate_elapsed_ns
        else:
            elapsed_ns = 5_000_000
        return CommandResult(
            0,
            json.dumps(observation),
            "",
            elapsed_ns,
            process_tree_rss=ProcessTreeRssEvidence(
                peak_bytes=1_000_000 + len(self.calls),
                sample_count=2,
                sample_interval_ns=PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
                source=PROCESS_TREE_RSS_SOURCE,
                root_pid=123,
                root_starttime_ticks=456,
            ),
        )


def test_runner_gates_then_collects_ten_isolated_warm_samples(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    executor = FakeExecutor(_first_observation())

    state = run_c0_measurement(
        spec,
        executor=executor,
        environment={
            "JAX_ENABLE_X64": "false",
            "JAX_PLATFORMS": "cpu",
            "JAX_TRANSFER_GUARD": "allow",
        },
    )
    assert state["state"] == "POST_GATE_PENDING"
    assert state["warm_p50_ns"] == 95.5
    assert len(executor.calls) == 12
    assert [call[1]["SINGLE_STAGE_COMPUTE_GRAPH_MODE"] for call in executor.calls] == [
        "initial-gate",
        "gate",
        *("warm" for _ in range(10)),
    ]
    assert "SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX" not in executor.calls[0][1]
    assert [
        call[1]["SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX"]
        for call in executor.calls[2:]
    ] == [str(index) for index in range(10)]
    assert all(
        call[1]["SINGLE_STAGE_COMPUTE_GRAPH_LANE"] == RTX_LANE_ID
        for call in executor.calls
    )
    assert len({call[1]["JAX_COMPILATION_CACHE_DIR"] for call in executor.calls}) == 1
    assert all(call[1]["JAX_ENABLE_X64"] == "true" for call in executor.calls)
    assert all(call[1]["JAX_PLATFORMS"] == "cuda" for call in executor.calls)
    assert all(call[1]["JAX_TRANSFER_GUARD"] == "disallow" for call in executor.calls)
    snapshot_root = Path(spec["provenance"]["immutable_root"])
    assert all(call[2] == snapshot_root for call in executor.calls)
    assert all(
        call[1]["PYTHONPATH"] == f"{snapshot_root / 'src'}:{snapshot_root}"
        for call in executor.calls
    )
    assert all(call[1]["PYTHONNOUSERSITE"] == "1" for call in executor.calls)
    assert all(call[1]["PYTHONSAFEPATH"] == "1" for call in executor.calls)
    assert (Path(spec["output_root"]) / "gate-checkpoint.json").is_file()
    assert (Path(spec["output_root"]) / "warm-checkpoint.json").is_file()
    assert not (Path(spec["output_root"]) / RECEIPT_FILENAME).exists()
    gate_checkpoint = json.loads(
        (Path(spec["output_root"]) / "gate-checkpoint.json").read_text()
    )
    assert (
        gate_checkpoint["native_reference_sha256"]
        == hashlib.sha256(Path(spec["native_reference_path"]).read_bytes()).hexdigest()
    )
    child_root = Path(spec["output_root"]) / "children"
    assert len(list(child_root.glob("*/raw.json"))) == 12
    assert len({call[0] for call in executor.calls}) == 12
    assert all(
        call[0][1:6]
        == (
            "-P",
            "-s",
            "-c",
            ISOLATED_MODULE_BOOTSTRAP,
            "benchmarks.single_stage_compute_graph_c0_evaluator",
        )
        for call in executor.calls
    )
    assert all(
        "--input-root" in call[0] and "--candidate" in call[0]
        for call in executor.calls
    )
    commands = [call[0] for call in executor.calls]

    def argument(command: tuple[str, ...], flag: str) -> str:
        return command[command.index(flag) + 1]

    assert {argument(command, "--input-root") for command in commands} == {
        str(Path(spec["input_root"]).resolve())
    }
    assert {argument(command, "--candidate") for command in commands} == {
        str(Path(spec["candidate_path"]).resolve())
    }
    assert {argument(command, "--parameter-sha256") for command in commands} == {
        spec["receipt_template"]["specimen"]["parameter_sha256"]
    }
    assert {
        argument(command, "--initial-parameter-sha256") for command in commands
    } == {_digest("a")}
    assert {argument(command, "--gpu-uuid") for command in commands} == {
        spec["provenance"]["allocation"]["gpu_uuid"]
    }
    assert len({argument(command, "--trace-root") for command in commands}) == 12
    assert {argument(command, "--identity-anchor") for command in commands} == {
        str(Path(spec["output_root"]) / "hlo-module-set-identity-anchor.json")
    }


def test_first_evaluation_timeout_stops_before_warm_sampling(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    executor = FakeExecutor(_first_observation(), timeout=True)

    with pytest.raises(C0RunnerError, match="900-second|timed out"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert len(executor.calls) == 1
    output_root = Path(spec["output_root"])
    assert (output_root / FAILURE_FILENAME).is_file()
    assert not (output_root / RECEIPT_FILENAME).exists()


def test_proc_sampler_includes_descendants_and_excludes_unrelated(
    tmp_path: Path,
) -> None:
    def write_process(pid: int, parent: int, starttime: int, rss_kib: int) -> None:
        process_root = tmp_path / str(pid)
        task_root = process_root / "task" / str(pid)
        task_root.mkdir(parents=True)
        fields = ["S", str(parent), *("0" for _ in range(17)), str(starttime)]
        (process_root / "stat").write_text(
            f"{pid} (process {pid}) {' '.join(fields)}\n", encoding="utf-8"
        )
        (process_root / "status").write_text(
            f"Name:\tprocess-{pid}\nVmRSS:\t{rss_kib} kB\n", encoding="utf-8"
        )
        (task_root / "children").write_text("", encoding="utf-8")

    write_process(100, 1, 1000, 10)
    write_process(101, 100, 1001, 20)
    write_process(200, 1, 2000, 1000)
    (tmp_path / "100/task/100/children").write_text("101\n", encoding="utf-8")

    assert _linux_process_tree_rss_bytes(100, 1000, tmp_path) == 30 * 1024
    assert _linux_process_tree_rss_bytes(100, 9999, tmp_path) == 0

    started_ns = time.monotonic_ns()
    for _ in range(100):
        _linux_process_tree_rss_bytes(100, 1000, tmp_path)
    mean_sample_ns = (time.monotonic_ns() - started_ns) / 100
    assert mean_sample_ns < PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS


def test_subprocess_executor_samples_descendant_rss_from_proc(tmp_path: Path) -> None:
    child_code = "import time; payload = bytearray(24 * 1024 * 1024); time.sleep(0.15)"
    parent_code = (
        "import subprocess, sys; child = subprocess.Popen([sys.executable, '-c', "
        f"{child_code!r}]); child.wait()"
    )

    result = _subprocess_executor(
        (sys.executable, "-c", parent_code), os.environ, tmp_path, 5.0
    )

    assert result.returncode == 0
    assert result.process_tree_rss is not None
    assert result.process_tree_rss.source == PROCESS_TREE_RSS_SOURCE
    assert (
        result.process_tree_rss.sample_interval_ns
        == PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
    )
    assert result.process_tree_rss.sample_count >= 2
    assert result.process_tree_rss.peak_bytes >= 24 * 1024 * 1024


def test_initial_gate_parity_failure_stops_before_changed_state_timing(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    initial = _initial_observation()
    initial["objective"] = 2.0
    executor = FakeExecutor(_first_observation(), initial=initial)

    with pytest.raises(C0RunnerError, match="initial evaluation gate.*parity"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert len(executor.calls) == 1
    child_root = Path(spec["output_root"]) / "children"
    assert (child_root / "initial-gate" / "raw.json").is_file()
    assert not (child_root / "gate").exists()
    assert not list(child_root.glob("warm-*"))


def test_native_reference_v2_is_rejected_before_initial_gate(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    native_path = Path(spec["native_reference_path"])
    native = json.loads(native_path.read_text(encoding="utf-8"))
    native["schema_id"] = "single-stage-compute-graph-native-reference-v2"
    native_path.write_text(json.dumps(native), encoding="utf-8")
    executor = FakeExecutor(_first_observation())

    with pytest.raises(C0RunnerError, match="unsupported native-reference schema"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert executor.calls == []


@pytest.mark.parametrize(
    ("elapsed_ns", "passes"),
    [(899_000_000_000, True), (901_000_000_000, False)],
)
def test_parent_subprocess_wall_is_gate_authority(
    tmp_path: Path, elapsed_ns: int, passes: bool
) -> None:
    spec = _spec(tmp_path)
    executor = FakeExecutor(_first_observation(), gate_elapsed_ns=elapsed_ns)
    if passes:
        state = run_c0_measurement(spec, executor=executor, environment={})
        assert state["state"] == "POST_GATE_PENDING"
        checkpoint = json.loads(
            (Path(spec["output_root"]) / "gate-checkpoint.json").read_text()
        )
        assert checkpoint["first_evaluation_gate"]["elapsed_ns"] == elapsed_ns
    else:
        with pytest.raises(C0RunnerError, match="900-second"):
            run_c0_measurement(spec, executor=executor, environment={})
        assert len(executor.calls) == 2


def test_wrong_gradient_count_fails_before_warm_sampling(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    executor = FakeExecutor(_first_observation(gradient_count=460))

    with pytest.raises(C0RunnerError, match="exactly 461"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert len(executor.calls) == 2


def test_precomputed_derived_evidence_is_rejected_before_launch(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["profile"] = _profile()
    executor = FakeExecutor(_first_observation())
    with pytest.raises(C0RunnerError, match="precomputed derived evidence"):
        run_c0_measurement(spec, executor=executor, environment={})
    assert executor.calls == []


def test_failed_status_or_parity_fails_before_warm_sampling(tmp_path: Path) -> None:
    for mutation, match in (
        ("status", "adjoint solve failed"),
        ("parity", "objective parity failed"),
    ):
        spec = _spec(tmp_path / mutation)
        first = copy.deepcopy(_first_observation())
        if mutation == "status":
            first["adjoint_success"] = False
        else:
            first["objective"] = 2.0
        executor = FakeExecutor(first)

        with pytest.raises(C0RunnerError, match=match):
            run_c0_measurement(spec, executor=executor, environment={})

        assert len(executor.calls) == 2


def test_blocked_qualification_prevents_child_launch(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["receipt_template"]["lanes"][0]["qualification"] = _qualification(
        RTX_LANE_ID, blocked=True
    )
    executor = FakeExecutor(_first_observation())

    with pytest.raises(C0RunnerError, match="qualification"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert executor.calls == []


def test_artifact_and_cache_roots_must_be_disjoint(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["provenance"]["compilation_cache_directory"] = str(
        Path(spec["output_root"]) / "cache"
    )
    executor = FakeExecutor(_first_observation())

    with pytest.raises(C0RunnerError, match="disjoint"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert executor.calls == []


def test_preexisting_cache_blocks_before_child_launch(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    Path(spec["provenance"]["compilation_cache_directory"]).mkdir(parents=True)
    executor = FakeExecutor(_first_observation())

    with pytest.raises(C0RunnerError, match="must not exist"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert executor.calls == []


def test_gap_budget_uses_runner_warm_p50_last() -> None:
    profile = _profile()
    budget = _compute_gap_budget(
        {
            "matched_complete_path_reference_timings_ns": {
                "native": 800.0,
                "c0": 955.0,
                "optax": 900.0,
            },
            "c0_complete_path_value_and_gradient_evaluation_count": 5,
            "c0_complete_path_value_and_gradient_evaluation_count_semantics": (
                "scipy_optimize_result_nfev_for_combined_objective_and_gradient_"
                "callable_within_complete_path_boundary"
            ),
            "phase_reduction_assumptions": {
                "newton.residual_jvp": {
                    "conservative_reduction": 0.1,
                    "optimistic_reduction": 0.2,
                    "overlap_disposition": "disjoint",
                }
            },
            "unattributed_conservative_reduction": 0.0,
            "unattributed_optimistic_reduction": 0.5,
            "faithful_levers": [
                {
                    "lever_id": "dense_newton",
                    "disposition": "bounded",
                    "evidence_sha256": "f" * 64,
                }
            ],
        },
        warm_p50_ns=123.0,
        profile=profile,
    )
    assert budget["candidate_value_and_gradient_reference_timings_ns"] == {
        "c0_warm_p50": 123.0
    }
    assert budget["candidate_phases"][0]["measured_share"] == pytest.approx(920 / 1100)
    assert budget["unattributed_share"] == pytest.approx(180 / 1100)
    per_call_saving = (
        123.0 - budget["candidate_value_and_gradient_conservative_projected_ns"]
    )
    assert budget["conservative_complete_path_projected_ns"] == pytest.approx(
        955.0 - 5 * per_call_saving
    )


def test_gap_budget_does_not_project_device_coverage_over_full_wall() -> None:
    profile = _profile()
    profile["evaluation_envelope_ns"] = 10_000
    budget = _compute_gap_budget(
        {
            "matched_complete_path_reference_timings_ns": {
                "native": 800.0,
                "c0": 955.0,
                "optax": 900.0,
            },
            "c0_complete_path_value_and_gradient_evaluation_count": 1,
            "c0_complete_path_value_and_gradient_evaluation_count_semantics": (
                "scipy_optimize_result_nfev_for_combined_objective_and_gradient_"
                "callable_within_complete_path_boundary"
            ),
            "phase_reduction_assumptions": {
                "newton.residual_jvp": {
                    "conservative_reduction": 1.0,
                    "optimistic_reduction": 1.0,
                    "overlap_disposition": "disjoint",
                }
            },
            "unattributed_conservative_reduction": 0.0,
            "unattributed_optimistic_reduction": 0.0,
            "faithful_levers": [
                {
                    "lever_id": "dense_newton",
                    "disposition": "bounded",
                    "evidence_sha256": "f" * 64,
                }
            ],
        },
        warm_p50_ns=100.0,
        profile=profile,
    )
    assert budget["candidate_phases"][0]["measured_share"] == pytest.approx(0.092)
    assert budget[
        "candidate_value_and_gradient_optimistic_projected_ns"
    ] == pytest.approx(90.8)


def test_resume_uses_checkpoints_without_rerunning_gate_or_warm(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    executor = FakeExecutor(_first_observation())
    run_c0_measurement(spec, executor=executor, environment={})
    assert len(executor.calls) == 12
    root = Path(spec["output_root"])
    spec["resume"] = {
        "gate_checkpoint_path": str(root / "gate-checkpoint.json"),
        "warm_checkpoint_path": str(root / "warm-checkpoint.json"),
        "profile_evidence_path": str(root / "missing-profile.json"),
        "command_buffer_evidence_path": str(root / "missing-command.json"),
        "newton_telemetry_evidence_path": str(root / "missing-telemetry.json"),
        "gap_budget_inputs_path": str(root / "missing-gap.json"),
    }

    with pytest.raises(FileNotFoundError):
        run_c0_measurement(spec, executor=executor, environment={})

    assert len(executor.calls) == 12


def test_resume_rejects_source_drift_before_loading_post_gate_evidence(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    executor = FakeExecutor(_first_observation())
    run_c0_measurement(spec, executor=executor, environment={})
    root = Path(spec["output_root"])
    spec["resume"] = {
        "gate_checkpoint_path": str(root / "gate-checkpoint.json"),
        "warm_checkpoint_path": str(root / "warm-checkpoint.json"),
        "profile_evidence_path": str(root / "missing-profile.json"),
        "command_buffer_evidence_path": str(root / "missing-command.json"),
        "newton_telemetry_evidence_path": str(root / "missing-telemetry.json"),
        "gap_budget_inputs_path": str(root / "missing-gap.json"),
    }
    spec["provenance"]["source_state_sha256"] = "9" * 64

    with pytest.raises(C0RunnerError, match="source state differs"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert len(executor.calls) == 12


def test_resume_rejects_pending_state_hash_drift(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    executor = FakeExecutor(_first_observation())
    run_c0_measurement(spec, executor=executor, environment={})
    root = Path(spec["output_root"])
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["warm_checkpoint_sha256"] = "0" * 64
    state_path.write_bytes(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    spec["resume"] = {
        "gate_checkpoint_path": str(root / "gate-checkpoint.json"),
        "warm_checkpoint_path": str(root / "warm-checkpoint.json"),
    }

    with pytest.raises(C0RunnerError, match="checkpoint hashes"):
        run_c0_measurement(spec, executor=executor, environment={})


def test_native_reference_binding_drift_blocks_before_gate(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    native_path = Path(spec["native_reference_path"])
    native = json.loads(native_path.read_text(encoding="utf-8"))
    native["identity"]["source_sha256"] = "0" * 64
    native_path.write_text(json.dumps(native), encoding="utf-8")
    executor = FakeExecutor(_first_observation())

    with pytest.raises(C0RunnerError, match="source_sha256 binding mismatch"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert executor.calls == []


def test_wrong_raw_input_bundle_sha_blocks_before_gate(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (Path(spec["input_root"]) / "input_bundle.json").write_bytes(b"drifted\n")
    executor = FakeExecutor(_first_observation())

    with pytest.raises(C0RunnerError, match="raw input_bundle.json SHA-256"):
        run_c0_measurement(spec, executor=executor, environment={})

    assert executor.calls == []


def test_real_subprocess_fake_cli_receives_constructed_sample_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    spec["provenance"]["interpreter_path"] = sys.executable
    snapshot = Path(spec["provenance"]["immutable_root"])
    module = snapshot / "fake_c0_child.py"
    first_document = _first_observation()
    warm_document = _warm_observation(0)
    module.write_text(
        f"""import json, os, time
mode = os.environ['SINGLE_STAGE_COMPUTE_GRAPH_MODE']
if mode == 'initial-gate':
    document = {_initial_observation()!r}
elif mode == 'gate':
    document = {first_document!r}
else:
    document = {warm_document!r}
    index = int(os.environ['SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX'])
    document['sample_index'] = index
    document['wall_ns'] = 91 + index
time.sleep(0.03)
print(json.dumps(document))
""",
        encoding="utf-8",
    )
    manifest_path = snapshot / "phase0-source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module_bytes = module.read_bytes()
    manifest["entries"].append(
        {
            "role": "benchmark",
            "relative_path": module.name,
            "size_bytes": len(module_bytes),
            "sha256": hashlib.sha256(module_bytes).hexdigest(),
        }
    )
    manifest["entries"].sort(key=lambda entry: entry["relative_path"])
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_c0_runner.EVALUATOR_MODULE",
        "fake_c0_child",
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_isolated_launch.ALLOWED_MODULES",
        frozenset({"fake_c0_child"}),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)

    state = run_c0_measurement(spec, environment=environment)

    assert state["state"] == "POST_GATE_PENDING"


def test_resume_with_real_producer_artifacts_writes_valid_receipt(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    candidate = np.linspace(0.1, 0.2, 461, dtype=np.float64)
    candidate_path = Path(spec["candidate_path"])
    np.save(candidate_path, candidate)
    candidate_sha256 = hashlib.sha256(
        np.ascontiguousarray(candidate, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    specimen = spec["receipt_template"]["specimen"]
    specimen["parameter_sha256"] = candidate_sha256
    specimen_sha256 = canonical_sha256(specimen)
    spec["receipt_template"]["specimen_sha256"] = specimen_sha256
    native_path = Path(spec["native_reference_path"])
    native = json.loads(native_path.read_text(encoding="utf-8"))
    native["parameter_sha256"] = candidate_sha256
    native["identity"]["specimen_sha256"] = specimen_sha256
    native_path.write_text(json.dumps(native), encoding="utf-8")

    changed_observation = _first_observation()
    changed_observation["parameter_sha256"] = candidate_sha256
    executor = FakeExecutor(changed_observation)
    run_c0_measurement(spec, executor=executor, environment={})
    root = Path(spec["output_root"])
    gate_path = root / "gate-checkpoint.json"
    warm_path = root / "warm-checkpoint.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    warm = json.loads(warm_path.read_text(encoding="utf-8"))
    gate_sha256 = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    warm_sha256 = hashlib.sha256(warm_path.read_bytes()).hexdigest()
    warm_p50_ns = warm["warm_measurement"]["p50_ns"]
    identity = {
        "candidate_sha256": candidate_sha256,
        "specimen_sha256": specimen_sha256,
        "input_bundle_sha256": gate["input_bundle_sha256"],
        "source_sha256": gate["source_state_sha256"],
        "runtime_identity_sha256": gate["runtime_identity_sha256"],
        "lane_id": gate["lane_id"],
        "gpu_uuid": gate["gpu_uuid"],
        "gate_checkpoint_sha256": gate_sha256,
        "warm_checkpoint_sha256": warm_sha256,
        "warm_p50_ns": warm_p50_ns,
    }

    trace_root = root / "profile"
    trace_root.mkdir()
    trace_path = trace_root / "profile.trace.json"
    trace_path.write_bytes(
        canonical_json_bytes(
            {
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
                    *(
                        {
                            "ph": "X",
                            "pid": 11,
                            "tid": 1,
                            "ts": timestamp,
                            "dur": 20.0,
                            "name": f"CommonPjRtLoadedExecutable::Execute ({module})",
                            "args": {"name": module, "execution_mode": mode},
                        }
                        for module, timestamp, mode in (
                            ("jit_forward", 20.0, "command_buffer"),
                            ("jit_gradient", 45.0, "uncaptured"),
                        )
                    ),
                    *(
                        {
                            "ph": "X",
                            "pid": 12,
                            "tid": 1,
                            "ts": timestamp,
                            "dur": 10.0,
                            "name": f"{module}_kernel",
                            "args": {
                                "context_id": "$$1",
                                "correlation_id": module,
                                "hlo_module": module,
                                "hlo_op": f"{module}/fusion",
                                "kernel_details": "regs:16",
                                "name": f"jit({module})/{phase}",
                                "scope_range_id": module,
                                "tf_op": "XlaModule:",
                            },
                        }
                        for module, phase, timestamp in (
                            ("jit_forward", "newton.residual_jvp", 25.0),
                            ("jit_gradient", "adjoint.lu_solve", 45.0),
                        )
                    ),
                    {},
                ],
            }
        )
    )
    profile = build_profile_evidence(
        trace_path=trace_path,
        artifact_root=root,
        **identity,
    )
    profile_path = root / "profile-evidence.json"
    write_profile_evidence(profile_path, profile)

    nsys = tmp_path / "nsys"
    nsys.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nsys.chmod(0o755)
    nvtx_library = tmp_path / "libnvToolsExt.so.1"
    nvtx_library.write_bytes(b"nvtx")
    plan = build_control_plan(
        nsys_binary=nsys,
        nvtx_library=nvtx_library,
        expected_nsys_version="NVIDIA Nsight Systems version 2026.1",
        python_binary=Path(spec["provenance"]["interpreter_path"]),
        snapshot_root=Path(spec["provenance"]["immutable_root"]),
        artifact_root=tmp_path / "command-artifacts",
        cache_root=tmp_path / "command-cache",
        input_root=Path(spec["input_root"]),
        candidate_path=candidate_path,
        specimen_sha256=specimen_sha256,
        candidate_sha256=candidate_sha256,
        source_sha256=identity["source_sha256"],
        gate_checkpoint_sha256=gate_sha256,
        warm_checkpoint_sha256=warm_sha256,
        warm_p50_ns=warm_p50_ns,
        lane_id="rtx5090",
        gpu_uuid=identity["gpu_uuid"],
        runtime_identity_sha256=identity["runtime_identity_sha256"],
        input_bundle_sha256=identity["input_bundle_sha256"],
        current_xla_flags="--xla_gpu_triton_gemm_any=true",
        base_environment={},
    )
    for lane, graph in zip(plan.lanes, (True, False), strict=True):
        lane.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(lane.sqlite_path) as connection:
            connection.execute(
                "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT)"
            )
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES(100, 1000, ?)",
                ("single_stage.compute_graph.evaluation:" + candidate_sha256,),
            )
            connection.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
            connection.executemany(
                "INSERT INTO StringIds VALUES(?, ?)",
                enumerate(
                    (
                        "cudaGraphInstantiateWithFlags",
                        "cudaGraphLaunch",
                        "cudaGraphExecUpdate",
                    ),
                    start=1,
                ),
            )
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME("
                "start INTEGER, end INTEGER, nameId INTEGER, "
                "correlationId INTEGER)"
            )
            if graph:
                connection.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?, ?, ?, ?)",
                    ((120, 130, 1, 120), (200, 215, 2, 200), (300, 307, 3, 300)),
                )
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL("
                "start INTEGER, end INTEGER, correlationId INTEGER, "
                "graphNodeId INTEGER)"
            )
            rows = (
                ((400, 500, 200, 7), (550, 650, 550, None))
                if graph
                else ((400, 600, 400, None),)
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL VALUES(?, ?, ?, ?)",
                rows,
            )
    default_evidence = parse_nsys_sqlite(plan.lanes[0].sqlite_path, candidate_sha256)
    disabled_evidence = parse_nsys_sqlite(plan.lanes[1].sqlite_path, candidate_sha256)
    command_document = build_control_evidence(
        plan,
        nsys_version=plan.expected_nsys_version,
        default_evidence=default_evidence,
        disabled_evidence=disabled_evidence,
        default_wall_ns=100,
        disabled_wall_ns=110,
    )
    command_path = root / "command-buffer.json"
    command_path.write_bytes(canonical_json_bytes(command_document))

    telemetry_identity = TelemetryIdentity(**identity)

    class Prepared:
        def __init__(self, observed: bool) -> None:
            self.observed = observed

        def evaluate(self) -> CandidateEvaluation:
            return CandidateEvaluation(
                objective=1.0,
                raw_objective=1.0,
                gradient=np.ones(461, dtype=np.float64),
                solved_state=np.ones(255, dtype=np.float64),
                newton_success=True,
                newton_iterations=2,
                observer_bearing=self.observed,
                execution_counts=(
                    ExecutionCounts(7, 5) if self.observed else ExecutionCounts(0, 0)
                ),
            )

    telemetry_document = collect_newton_telemetry(
        telemetry_identity,
        candidate,
        lambda values: Prepared(
            os.environ.get("SIMSOPT_TRACEABLE_EXACT_NEWTON_EXECUTION_COUNTS") == "1"
        ),
        clock=iter((1, 21, 30, 80)).__next__,
    )
    telemetry_path = root / "newton-telemetry.json"
    write_newton_telemetry(telemetry_path, telemetry_document)

    binding = CompletePathBinding(
        specimen_sha256=specimen_sha256,
        candidate_sha256=candidate_sha256,
        source_sha256=identity["source_sha256"],
        runtime_identity_sha256=identity["runtime_identity_sha256"],
        native_reference_sha256=gate["native_reference_sha256"],
        gate_checkpoint_sha256=gate_sha256,
        warm_checkpoint_sha256=warm_sha256,
        warm_p50_ns=warm_p50_ns,
        lane_id="rtx5090",
        gpu_uuid=identity["gpu_uuid"],
    )
    lane_timings = {"native_cpu": 800, "jax_gpu_fast": 955, "jax_gpu_optax": 900}
    samples = []
    for run in build_complete_path_plan():
        parity_rows = (
            ()
            if run.profile_id == "native_cpu"
            else tuple(
                complete_path.measurement_runner.ParityRow(name, 1.0, 1.0, 1e-12)
                for name, _ in complete_path.measurement_runner._SINGLE_STAGE_PARITY_OBSERVABLES
            )
        )
        samples.append(
            ProtocolSample(
                profile_id=run.profile_id,
                phase=run.phase,
                sample_index=run.sample_index,
                optimization_wall_ns=lane_timings[run.profile_id],
                subprocess_wall_ns=lane_timings[run.profile_id] + 10,
                driver=complete_path.EXPECTED_DRIVERS[run.profile_id],
                backend_mode=complete_path.EXPECTED_BACKENDS[run.profile_id],
                input_fingerprint="5" * 64,
                configuration_fingerprint="6" * 64,
                effective_construction_fingerprint="7" * 64,
                input_bundle_sha256=identity["input_bundle_sha256"],
                source_sha256=identity["source_sha256"],
                runtime_identity_sha256=identity["runtime_identity_sha256"],
                nit=2,
                nfev=3,
                njev=3,
                endpoint_certificate={
                    "success": True,
                    "initial_stationary": False,
                    "terminal_stationary": False,
                    "constraints_satisfied": True,
                    "outer_status": 1,
                },
                parity_rows=parity_rows,
                    snapshot_source_manifest_sha256="8" * 64,
                    snapshot_import_attestation_sha256="9" * 64,
                    snapshot_lane_identity_sha256="b" * 64,
                    provenance={
                    "repository_commit": "commit",
                    "executed_sources": [{"path": "source.py", "sha256": "a" * 64}],
                },
            )
        )
    complete_document = build_complete_path_document(binding, samples)
    complete_path_file = root / "complete-path.json"
    complete_path_file.write_bytes(canonical_json_bytes(complete_document))
    gap_document = build_gap_budget_inputs_artifact(
        complete_document,
        GapBudgetPolicyInput(
            phase_reduction_assumptions={
                phase: PhaseReductionAssumption(0.0, 0.1, "disjoint")
                for phase in ("newton.residual_jvp", "adjoint.lu_solve")
            },
            unattributed_conservative_reduction=0.0,
            unattributed_optimistic_reduction=0.0,
            faithful_levers=(FaithfulLever("diagnostic", "bounded", "a" * 64),),
        ),
    )
    gap_path = root / "gap-inputs.json"
    gap_path.write_bytes(canonical_json_bytes(gap_document))

    attribution_binding = AttributionBinding(
        candidate_sha256=candidate_sha256,
        specimen_sha256=specimen_sha256,
        input_bundle_sha256=identity["input_bundle_sha256"],
        source_sha256=identity["source_sha256"],
        production_runtime_identity_sha256=identity["runtime_identity_sha256"],
        lane_id="rtx5090",
        gpu_uuid=identity["gpu_uuid"],
        gate_checkpoint_sha256=gate_sha256,
        warm_checkpoint_sha256=warm_sha256,
        warm_p50_ns=warm_p50_ns,
    )

    def attribution_attempt(
        mode: Literal["default_control", "command_buffer_disabled"], index: int
    ) -> AttributionAttempt:
        disabled = mode == "command_buffer_disabled"
        artifact_root = (
            root / "post-gate" / "attribution-control" / mode / f"attempt-{index:02d}"
        )
        artifact_root.mkdir(parents=True)
        trace = artifact_root / "trace.json"
        child = artifact_root / "child.json"
        anchor = artifact_root / "anchor.json"
        trace.write_bytes(trace_path.read_bytes())
        child.write_bytes(
            canonical_json_bytes(
                {
                    "schema_id": C0_CHILD_OBSERVATION_SCHEMA_ID,
                    "mode": "profile",
                    "sample_index": None,
                    "parameter_sha256": candidate_sha256,
                    "objective_dtype": "float64",
                    "objective": gate["gate_observation"]["objective"],
                    "gradient_dtype": "float64",
                    "gradient": gate["gate_observation"]["gradient"],
                    "inner_newton_success": gate["gate_observation"][
                        "inner_newton_success"
                    ],
                    "adjoint_success": gate["gate_observation"]["adjoint_success"],
                    "residual_certificates": gate["gate_observation"][
                        "residual_certificates"
                    ],
                    "cold_compile": {
                        "wall_ns": profile.profile.evaluation_envelope_ns + 1,
                        "peak_process_tree_rss_bytes": 1,
                        "process_tree_rss_sample_count": 1,
                        "process_tree_rss_sample_interval_ns": (
                            PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
                        ),
                        "process_tree_rss_source": PROCESS_TREE_RSS_SOURCE,
                        "process_tree_rss_root_pid": 123,
                        "process_tree_rss_root_starttime_ticks": 456,
                        "sampled_process_gpu_memory_peak_bytes": 0,
                        "sampled_process_gpu_memory_source": (
                            SAMPLED_PROCESS_GPU_MEMORY_SOURCE
                        ),
                        "hlo_module_set_identity": (
                            profile.profile.hlo_module_set_identity
                        ),
                        "hlo_module_set_identity_source": (
                            profile.profile.hlo_module_set_identity_source
                        ),
                    },
                    "pjrt_execute_count": profile.profile.pjrt_execute_count,
                    "kernel_launch_count": profile.profile.kernel_launch_count,
                }
            )
        )
        anchor.write_bytes(
            canonical_json_bytes(
                {
                    "schema_id": (
                        "single-stage-compute-graph-c0-hlo-module-set-identity-anchor-v2"
                    ),
                    "hlo_module_set_identity": (
                        profile.profile.hlo_module_set_identity
                    ),
                    "hlo_module_set_identity_source": (
                        profile.profile.hlo_module_set_identity_source
                    ),
                }
            )
        )
        disabled_environment = dict(spec["provenance"]["environment"])
        disabled_environment["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
        disabled_runtime_identity = _runtime_identity(
            {
                "interpreter_path": spec["provenance"]["interpreter_path"],
                "runtime": spec["provenance"]["runtime"],
                "environment": disabled_environment,
                "policies": spec["provenance"]["policies"],
            }
        )
        return AttributionAttempt(
            mode=mode,
            attempt_index=index,
            binding=attribution_binding,
            runtime_identity_sha256=(
                disabled_runtime_identity
                if disabled
                else identity["runtime_identity_sha256"]
            ),
            xla_flag_tokens=("--xla_gpu_enable_command_buffer=",) if disabled else (),
            compilation_cache_root=str(
                root
                / "post-gate"
                / "attribution-control-cache"
                / mode
                / f"attempt-{index:02d}"
            ),
            artifact_root=str(artifact_root),
            raw_trace_path=trace.relative_to(root).as_posix(),
            raw_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            child_observation_path=child.relative_to(root).as_posix(),
            child_observation_sha256=hashlib.sha256(child.read_bytes()).hexdigest(),
            hlo_anchor_path=anchor.relative_to(root).as_posix(),
            hlo_anchor_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest(),
            profile_derivation_version="compute-graph-profile-attribution-v1",
            objective=float(gate["gate_observation"]["objective"]),
            gradient=tuple(gate["gate_observation"]["gradient"]),
            solve_certificate={
                "inner_newton_success": gate["gate_observation"][
                    "inner_newton_success"
                ],
                "adjoint_success": gate["gate_observation"]["adjoint_success"],
                "residual_certificates": gate["gate_observation"][
                    "residual_certificates"
                ],
            },
            module_topology_identity_sha256=(
                canonical_sha256(
                    {
                        "hlo_module_set_identity": (
                            profile.profile.hlo_module_set_identity
                        ),
                        "hlo_module_set_identity_source": (
                            profile.profile.hlo_module_set_identity_source
                        ),
                        "solver_graph_specimen_sha256": specimen_sha256,
                    }
                )
            ),
            evaluation_envelope_ns=profile.profile.evaluation_envelope_ns,
            device_active_ns=profile.profile.device_active_ns,
            phase_device_ns=tuple(
                (
                    phase_id,
                    sum(interval.end_ns - interval.start_ns for interval in intervals),
                )
                for phase_id, intervals in profile.profile.phase_interval_unions
            ),
        )

    default_attribution_attempts = tuple(
        attribution_attempt("default_control", index) for index in range(3)
    )
    disabled_attribution_attempts = tuple(
        attribution_attempt("command_buffer_disabled", index) for index in range(3)
    )
    attribution_document = build_attribution_evidence(
        default_attribution_attempts,
        disabled_attribution_attempts,
    )
    attribution_path = (
        root / "post-gate" / "attribution-control" / "attribution-control-evidence.json"
    )
    attribution_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_path.write_bytes(canonical_json_bytes(attribution_document))

    spec["resume"] = {
        "gate_checkpoint_path": str(gate_path),
        "warm_checkpoint_path": str(warm_path),
        "profile_evidence_path": str(profile_path),
        "attribution_control_evidence_path": str(attribution_path),
        "command_buffer_evidence_path": str(command_path),
        "newton_telemetry_evidence_path": str(telemetry_path),
        "complete_path_evidence_path": str(complete_path_file),
        "gap_budget_inputs_path": str(gap_path),
    }
    mismatched_tokens = copy.deepcopy(attribution_document)
    for attempt in mismatched_tokens["direct_default_measurement"]["attempts"]:
        attempt["xla_flag_tokens"] = ["--unexpected=true"]
    for attempt in mismatched_tokens["attribution_replay"]["attempts"]:
        attempt["xla_flag_tokens"] = [
            "--unexpected=true",
            "--xla_gpu_enable_command_buffer=",
        ]
    attribution_path.write_bytes(canonical_json_bytes(mismatched_tokens))
    with pytest.raises(C0RunnerError, match="differ from production provenance"):
        _resume_c0_measurement(spec)
    attribution_path.write_bytes(canonical_json_bytes(attribution_document))
    bound_child = (
        root
        / attribution_document["direct_default_measurement"]["attempts"][0][
            "child_observation_path"
        ]
    )
    original_child = bound_child.read_bytes()
    bound_child.write_bytes(canonical_json_bytes({"mode": "tampered"}))
    with pytest.raises(C0RunnerError, match="child_observation_path hash mismatch"):
        _resume_c0_measurement(spec)
    bound_child.write_bytes(original_child)

    original_attempts = default_attribution_attempts + disabled_attribution_attempts
    original_child_bytes = {
        attempt.child_observation_path: (
            root / attempt.child_observation_path
        ).read_bytes()
        for attempt in original_attempts
    }
    impostor_child = json.loads(bound_child.read_text(encoding="utf-8"))
    del impostor_child["schema_id"]
    bound_child.write_bytes(canonical_json_bytes(impostor_child))
    impostor_attempt = replace(
        original_attempts[0],
        child_observation_sha256=hashlib.sha256(bound_child.read_bytes()).hexdigest(),
    )
    attribution_path.write_bytes(
        canonical_json_bytes(
            build_attribution_evidence(
                (impostor_attempt, *default_attribution_attempts[1:]),
                disabled_attribution_attempts,
            )
        )
    )
    with pytest.raises(C0RunnerError, match="profile observation schema"):
        _resume_c0_measurement(spec)
    bound_child.write_bytes(original_child)
    attribution_path.write_bytes(canonical_json_bytes(attribution_document))

    type_confused_attempts: list[AttributionAttempt] = []
    for attempt in original_attempts:
        child_path = root / attempt.child_observation_path
        child_document = json.loads(child_path.read_text(encoding="utf-8"))
        child_document["inner_newton_success"] = 1
        residual_name = next(iter(child_document["residual_certificates"]))
        child_document["residual_certificates"][residual_name] = True
        child_path.write_bytes(canonical_json_bytes(child_document))
        type_confused_attempts.append(
            replace(
                attempt,
                child_observation_sha256=hashlib.sha256(
                    child_path.read_bytes()
                ).hexdigest(),
                solve_certificate={
                    "inner_newton_success": 1,
                    "adjoint_success": child_document["adjoint_success"],
                    "residual_certificates": child_document["residual_certificates"],
                },
            )
        )
    with pytest.raises(
        AttributionControlError,
        match="solve_certificate success fields must be bool",
    ):
        build_attribution_evidence(
            tuple(type_confused_attempts[:3]),
            tuple(type_confused_attempts[3:]),
        )
    for relative_path, original_bytes in original_child_bytes.items():
        (root / relative_path).write_bytes(original_bytes)
    attribution_path.write_bytes(canonical_json_bytes(attribution_document))

    numerically_drifted_attempts: list[AttributionAttempt] = []
    for attempt in original_attempts:
        child_path = root / attempt.child_observation_path
        child_document = json.loads(child_path.read_text(encoding="utf-8"))
        child_document["objective"] = float(child_document["objective"]) + 1.0
        child_path.write_bytes(canonical_json_bytes(child_document))
        numerically_drifted_attempts.append(
            replace(
                attempt,
                objective=float(child_document["objective"]),
                child_observation_sha256=hashlib.sha256(
                    child_path.read_bytes()
                ).hexdigest(),
            )
        )
    attribution_path.write_bytes(
        canonical_json_bytes(
            build_attribution_evidence(
                tuple(numerically_drifted_attempts[:3]),
                tuple(numerically_drifted_attempts[3:]),
            )
        )
    )
    with pytest.raises(C0RunnerError, match="differs from gate"):
        _resume_c0_measurement(spec)
    for relative_path, original_bytes in original_child_bytes.items():
        (root / relative_path).write_bytes(original_bytes)
    attribution_path.write_bytes(canonical_json_bytes(attribution_document))

    original_trace_bytes = {
        attempt.raw_trace_path: (root / attempt.raw_trace_path).read_bytes()
        for attempt in original_attempts
    }
    original_anchor_bytes = {
        attempt.hlo_anchor_path: (root / attempt.hlo_anchor_path).read_bytes()
        for attempt in original_attempts
    }
    trace_drifted_attempts: list[AttributionAttempt] = []
    for attempt in original_attempts:
        raw_trace_path = root / attempt.raw_trace_path
        raw_trace = json.loads(raw_trace_path.read_text(encoding="utf-8"))
        first_kernel = next(
            event
            for event in raw_trace["traceEvents"]
            if isinstance(event, dict) and event.get("name") == "jit_forward_kernel"
        )
        first_kernel["dur"] = float(first_kernel["dur"]) + 1.0
        first_kernel["args"]["hlo_module"] = "jit_forward_drifted"
        raw_trace_path.write_bytes(canonical_json_bytes(raw_trace))
        rebuilt_profile = build_attribution_control_profile_evidence(
            trace_path=raw_trace_path,
            artifact_root=Path(attempt.artifact_root),
            candidate_sha256=attempt.binding.candidate_sha256,
            specimen_sha256=attempt.binding.specimen_sha256,
            input_bundle_sha256=attempt.binding.input_bundle_sha256,
            source_sha256=attempt.binding.source_sha256,
            runtime_identity_sha256=attempt.runtime_identity_sha256,
            lane_id=attempt.binding.lane_id,
            gpu_uuid=attempt.binding.gpu_uuid,
            gate_checkpoint_sha256=attempt.binding.gate_checkpoint_sha256,
            warm_checkpoint_sha256=attempt.binding.warm_checkpoint_sha256,
            warm_p50_ns=attempt.binding.warm_p50_ns,
        )
        child_path = root / attempt.child_observation_path
        child_document = json.loads(child_path.read_text(encoding="utf-8"))
        child_document["cold_compile"]["hlo_module_set_identity"] = (
            rebuilt_profile.profile.hlo_module_set_identity
        )
        child_document["cold_compile"]["hlo_module_set_identity_source"] = (
            rebuilt_profile.profile.hlo_module_set_identity_source
        )
        child_document["pjrt_execute_count"] = (
            rebuilt_profile.profile.pjrt_execute_count
        )
        child_document["kernel_launch_count"] = (
            rebuilt_profile.profile.kernel_launch_count
        )
        child_path.write_bytes(canonical_json_bytes(child_document))
        anchor_path = root / attempt.hlo_anchor_path
        anchor_path.write_bytes(
            canonical_json_bytes(
                {
                    "schema_id": (
                        "single-stage-compute-graph-c0-hlo-module-set-identity-anchor-v2"
                    ),
                    "hlo_module_set_identity": (
                        rebuilt_profile.profile.hlo_module_set_identity
                    ),
                    "hlo_module_set_identity_source": (
                        rebuilt_profile.profile.hlo_module_set_identity_source
                    ),
                }
            )
        )
        actual_phase_durations = tuple(
            (
                phase_id,
                sum(interval.end_ns - interval.start_ns for interval in intervals),
            )
            for phase_id, intervals in rebuilt_profile.profile.phase_interval_unions
        )
        guessed_phase_durations = (
            (actual_phase_durations[0][0], actual_phase_durations[0][1] + 1),
            *actual_phase_durations[1:],
        )
        trace_drifted_attempts.append(
            replace(
                attempt,
                raw_trace_sha256=hashlib.sha256(
                    raw_trace_path.read_bytes()
                ).hexdigest(),
                child_observation_sha256=hashlib.sha256(
                    child_path.read_bytes()
                ).hexdigest(),
                hlo_anchor_sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
                module_topology_identity_sha256=canonical_module_topology_identity(
                    rebuilt_profile.profile.hlo_module_set_identity,
                    rebuilt_profile.profile.hlo_module_set_identity_source,
                    specimen_sha256,
                ),
                evaluation_envelope_ns=(rebuilt_profile.profile.evaluation_envelope_ns),
                device_active_ns=rebuilt_profile.profile.device_active_ns + 1,
                phase_device_ns=guessed_phase_durations,
            )
        )
    attribution_path.write_bytes(
        canonical_json_bytes(
            build_attribution_evidence(
                tuple(trace_drifted_attempts[:3]),
                tuple(trace_drifted_attempts[3:]),
            )
        )
    )
    with pytest.raises(C0RunnerError, match="raw-recomputed evidence"):
        _resume_c0_measurement(spec)
    for relative_path, original_bytes in original_trace_bytes.items():
        (root / relative_path).write_bytes(original_bytes)
    for relative_path, original_bytes in original_anchor_bytes.items():
        (root / relative_path).write_bytes(original_bytes)
    for relative_path, original_bytes in original_child_bytes.items():
        (root / relative_path).write_bytes(original_bytes)
    attribution_path.write_bytes(canonical_json_bytes(attribution_document))

    receipt = run_c0_measurement(spec, executor=executor, environment={})

    audit = validate_phase0_receipt(receipt)
    assert audit.rtx.outcome == "qualified"
    assert audit.rtx.warm_p50_ns == warm_p50_ns
    assert audit.rtx.pivot_fired is False
    receipt_path = root / RECEIPT_FILENAME
    assert receipt_path.is_file()
    telemetry = receipt["lanes"][0]["measurement"]["newton_telemetry"]
    bound_telemetry_path = root / telemetry["raw_evidence_relative_path"]
    assert bound_telemetry_path.read_bytes() == telemetry_path.read_bytes()
    assert (
        hashlib.sha256(bound_telemetry_path.read_bytes()).hexdigest()
        == telemetry["raw_evidence_file_sha256"]
    )
    loaded_receipt, loaded_audit = load_phase0_receipt(receipt_path)
    assert loaded_receipt == receipt
    assert loaded_audit == audit
    assert len(executor.calls) == 12

    command_document["identity"]["warm_checkpoint_sha256"] = "0" * 64
    command_path.write_bytes(canonical_json_bytes(command_document))
    with pytest.raises(C0RunnerError, match="command-buffer"):
        run_c0_measurement(spec, executor=executor, environment={})
    assert len(executor.calls) == 12
