from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    CUDA_GRAPH_TRACE_GRANULARITY,
    DISABLE_COMMAND_BUFFER_FLAG,
    PROFILING_LIMITATION,
    CommandBufferControlError,
    build_control_evidence,
    build_control_plan,
    execute_control_plan,
    parse_nsys_sqlite,
    run_probe,
    validate_command_buffer_control_evidence,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    ISOLATED_MODULE_BOOTSTRAP,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    _validate_command_buffer,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    RoleRoot,
    publish_immutable_snapshot,
)

_CANDIDATE_SHA256 = "a" * 64
_SOURCE_SHA256 = "b" * 64
_GATE_CHECKPOINT_SHA256 = "c" * 64
_WARM_CHECKPOINT_SHA256 = "d" * 64
_SPECIMEN_SHA256 = "e" * 64
_RUNTIME_IDENTITY_SHA256 = "f" * 64
_INPUT_BUNDLE_SHA256 = (
    "866861ba022a1121c797ab61551bf79dc2dce2e92328a27025e1930a8d9af4f7"
)
_WARM_P50_NS = 95.5


def _executable(path: Path) -> Path:
    path.write_bytes(b"tool")
    path.chmod(0o755)
    return path


def _snapshot(root: Path) -> Path:
    source = root / "snapshot-source"
    files = (
        ("execution_source", "src/simsopt_jax/__init__.py"),
        ("configuration", "inputs/configuration.json"),
        (
            "benchmark",
            "benchmarks/single_stage_compute_graph_command_buffer_control.py",
        ),
        ("test", "tests/test_control.py"),
        ("native_extension", "src/simsoptpp.py"),
    )
    roles: list[RoleRoot] = []
    for role, relative_path in files:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"ROLE = {role!r}\n", encoding="utf-8")
        roles.append(RoleRoot(role, path, relative_path))
    snapshot = root / "snapshot"
    publish_immutable_snapshot(snapshot, tuple(roles))
    return snapshot


def _plan(
    tmp_path: Path,
    *,
    flags: str = "--xla_gpu_triton_gemm_any=true",
    gate_checkpoint_sha256: str = _GATE_CHECKPOINT_SHA256,
    warm_checkpoint_sha256: str = _WARM_CHECKPOINT_SHA256,
    warm_p50_ns: float = _WARM_P50_NS,
    specimen_sha256: str = _SPECIMEN_SHA256,
    lane_id: str = "rtx5090",
    runtime_identity_sha256: str = _RUNTIME_IDENTITY_SHA256,
    input_bundle_sha256: str = _INPUT_BUNDLE_SHA256,
    base_environment: dict[str, str] | None = None,
    nvtx_library: Path | None = None,
):
    nsys = _executable(tmp_path / "nsys")
    if nvtx_library is None:
        nvtx_library = tmp_path / "libnvToolsExt.so.1"
        nvtx_library.write_bytes(b"nvtx")
    python = _executable(tmp_path / "python")
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "input_bundle.json").write_bytes(b"input-bundle")
    candidate = tmp_path / "candidate.npy"
    candidate.write_bytes(b"candidate")
    snapshot_root = _snapshot(tmp_path)
    return build_control_plan(
        nsys_binary=nsys,
        nvtx_library=nvtx_library,
        expected_nsys_version="NVIDIA Nsight Systems version 2026.1",
        python_binary=python,
        snapshot_root=snapshot_root,
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        input_root=input_root,
        candidate_path=candidate,
        specimen_sha256=specimen_sha256,
        candidate_sha256=_CANDIDATE_SHA256,
        source_sha256=_SOURCE_SHA256,
        gate_checkpoint_sha256=gate_checkpoint_sha256,
        warm_checkpoint_sha256=warm_checkpoint_sha256,
        warm_p50_ns=warm_p50_ns,
        lane_id=lane_id,
        gpu_uuid="GPU-1",
        runtime_identity_sha256=runtime_identity_sha256,
        input_bundle_sha256=input_bundle_sha256,
        current_xla_flags=flags,
        base_environment=(
            {
                "PATH": "/bin",
                "PYTHONHOME": "/ambient-python",
                "PYTHONPATH": "/ambient-pythonpath",
            }
            if base_environment is None
            else base_environment
        ),
    )


def _sqlite_fixture(
    path: Path,
    *,
    graph: bool,
    candidate_sha256: str = _CANDIDATE_SHA256,
    graph_node_column: bool = True,
    optional_device_tables: bool = True,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT)"
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES(100, 1000, ?)",
            ("single_stage.compute_graph.evaluation:" + candidate_sha256,),
        )
        connection.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
        names = (
            "cudaGraphInstantiateWithFlags",
            "cudaGraphLaunch",
            "cudaGraphExecUpdate",
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES(?, ?)", enumerate(names, start=1)
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME("
            "start INTEGER, end INTEGER, nameId INTEGER, correlationId INTEGER)"
        )
        if graph:
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?, ?, ?, ?)",
                ((120, 130, 1, 120), (200, 215, 2, 200), (300, 307, 3, 300)),
            )
        if graph_node_column:
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL("
                "start INTEGER, end INTEGER, correlationId INTEGER, "
                "graphNodeId INTEGER, graphId INTEGER)"
            )
            rows = (
                ((400, 500, 200, 7, 11), (550, 650, 550, None, None))
                if graph
                else ((400, 600, 400, None, None),)
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
                "VALUES(?, ?, ?, ?, ?)",
                rows,
            )
        else:
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL("
                "start INTEGER, end INTEGER, correlationId INTEGER)"
            )
        if optional_device_tables:
            for table in (
                "CUPTI_ACTIVITY_KIND_MEMCPY",
                "CUPTI_ACTIVITY_KIND_MEMSET",
            ):
                connection.execute(
                    f"CREATE TABLE {table}(start INTEGER, end INTEGER, "
                    "correlationId INTEGER, graphNodeId INTEGER)"
                )


def test_control_plan_builds_exact_external_nsys_commands_and_fresh_caches(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, flags="--xla_gpu_triton_gemm_any=true")

    default, disabled = plan.lanes
    cache_directories = {
        dict(lane.environment)["JAX_COMPILATION_CACHE_DIR"] for lane in plan.lanes
    }
    assert len(cache_directories) == 2
    for lane in plan.lanes:
        assert lane.command[:6] == (
            str(plan.nsys_binary),
            "profile",
            "--trace=cuda,nvtx",
            "--cuda-graph-trace=node",
            "--export=sqlite",
            "--force-overwrite=false",
        )
        assert (
            "benchmarks.single_stage_compute_graph_command_buffer_control"
            in lane.command
        )
        assert lane.sqlite_path.parent == plan.artifact_root / lane.lane_id
        assert not Path(dict(lane.environment)["JAX_COMPILATION_CACHE_DIR"]).exists()
        assert dict(lane.environment)["JAX_PLATFORMS"] == "cuda"
        assert dict(lane.environment)["LD_LIBRARY_PATH"] == str(
            plan.nvtx_library.parent
        )
        assert lane.cwd == plan.snapshot_root
        assert lane.command[7:13] == (
            str(tmp_path / "python"),
            "-P",
            "-s",
            "-c",
            ISOLATED_MODULE_BOOTSTRAP,
            "benchmarks.single_stage_compute_graph_command_buffer_control",
        )
        assert dict(lane.environment)["PYTHONPATH"] == (
            f"{plan.snapshot_root / 'src'}:{plan.snapshot_root}"
        )
        assert "PYTHONHOME" not in dict(lane.environment)
        assert lane.command[-2:] == ("--nvtx-library", str(plan.nvtx_library))
    assert DISABLE_COMMAND_BUFFER_FLAG not in default.xla_flags
    assert DISABLE_COMMAND_BUFFER_FLAG in disabled.xla_flags
    assert CUDA_GRAPH_TRACE_GRANULARITY == "node"
    assert default.xla_flags == "--xla_gpu_triton_gemm_any=true"


def test_nvtx_preflight_fails_before_artifact_or_cache_creation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan.nvtx_library.write_bytes(b"changed-after-plan")

    with pytest.raises(CommandBufferControlError, match="changed after plan"):
        execute_control_plan(plan, tmp_path / "evidence.json")

    assert not plan.artifact_root.exists()
    for lane in plan.lanes:
        assert not Path(dict(lane.environment)["JAX_COMPILATION_CACHE_DIR"]).exists()


def test_plan_rejects_missing_or_nonregular_explicit_nvtx_dependency(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="nvtx_library must exist"):
        _plan(missing_root, nvtx_library=missing_root / "missing-nvtx.so")

    nonregular_root = tmp_path / "nonregular"
    nonregular_root.mkdir()
    nvtx_directory = nonregular_root / "nvtx-directory"
    nvtx_directory.mkdir()
    with pytest.raises(CommandBufferControlError, match="regular file"):
        _plan(nonregular_root, nvtx_library=nvtx_directory)


def test_control_plan_rejects_preselected_or_nonfresh_configuration(
    tmp_path: Path,
) -> None:
    with pytest.raises(CommandBufferControlError, match="already selects"):
        _plan(tmp_path, flags="--xla_gpu_enable_command_buffer=kernel")

    second_root = tmp_path / "second"
    second_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="fresh"):
        build_control_plan(
            nsys_binary=_executable(tmp_path / "nsys-2"),
            nvtx_library=tmp_path / "libnvToolsExt.so.1",
            expected_nsys_version="version",
            python_binary=_executable(tmp_path / "python-2"),
            snapshot_root=_snapshot(tmp_path / "second-snapshot"),
            artifact_root=second_root,
            cache_root=tmp_path / "cache-2",
            input_root=tmp_path / "inputs",
            candidate_path=tmp_path / "candidate.npy",
            specimen_sha256=_SPECIMEN_SHA256,
            candidate_sha256=_CANDIDATE_SHA256,
            source_sha256=_SOURCE_SHA256,
            gate_checkpoint_sha256=_GATE_CHECKPOINT_SHA256,
            warm_checkpoint_sha256=_WARM_CHECKPOINT_SHA256,
            warm_p50_ns=_WARM_P50_NS,
            lane_id="rtx5090",
            gpu_uuid="GPU-1",
            runtime_identity_sha256=_RUNTIME_IDENTITY_SHA256,
            input_bundle_sha256=_INPUT_BUNDLE_SHA256,
            current_xla_flags="",
        )


def test_control_plan_rejects_invalid_staged_runner_bindings(tmp_path: Path) -> None:
    invalid_gate_root = tmp_path / "invalid-gate"
    invalid_gate_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="gate_checkpoint_sha256"):
        _plan(invalid_gate_root, gate_checkpoint_sha256="not-a-sha256")

    invalid_warm_root = tmp_path / "invalid-warm"
    invalid_warm_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="warm_checkpoint_sha256"):
        _plan(invalid_warm_root, warm_checkpoint_sha256="not-a-sha256")

    invalid_p50_root = tmp_path / "invalid-p50"
    invalid_p50_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="positive and finite"):
        _plan(invalid_p50_root, warm_p50_ns=float("nan"))

    invalid_specimen_root = tmp_path / "invalid-specimen"
    invalid_specimen_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="specimen_sha256"):
        _plan(invalid_specimen_root, specimen_sha256="not-a-sha256")

    invalid_lane_root = tmp_path / "invalid-lane"
    invalid_lane_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="lane_id"):
        _plan(invalid_lane_root, lane_id="gpu")

    invalid_runtime_root = tmp_path / "invalid-runtime"
    invalid_runtime_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="runtime_identity_sha256"):
        _plan(invalid_runtime_root, runtime_identity_sha256="not-a-sha256")


def test_plan_and_probe_reject_wrong_input_bundle_binding(tmp_path: Path) -> None:
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    with pytest.raises(CommandBufferControlError, match="does not match input_root"):
        _plan(plan_root, input_bundle_sha256="0" * 64)

    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    input_root = probe_root / "inputs"
    input_root.mkdir()
    (input_root / "input_bundle.json").write_bytes(b"input-bundle")
    candidate = probe_root / "candidate.npy"
    candidate.write_bytes(b"not-reached")
    with pytest.raises(CommandBufferControlError, match="input bundle binding"):
        run_probe(
            input_root,
            candidate,
            _CANDIDATE_SHA256,
            "0" * 64,
            probe_root / "missing-nvtx.so",
        )


def test_sqlite_parser_records_graph_runtime_and_device_activity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph.sqlite"
    _sqlite_fixture(path, graph=True)

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_instantiate.to_json() == {
        "count": 1,
        "count_unit": "api_calls",
        "duration_ns": 10,
    }
    assert evidence.graph_launch.to_json() == {
        "count": 1,
        "count_unit": "api_calls",
        "duration_ns": 15,
    }
    assert evidence.graph_update.to_json() == {
        "count": 1,
        "count_unit": "api_calls",
        "duration_ns": 7,
    }
    assert evidence.graph_device_activity.to_json() == {
        "count": 1,
        "count_unit": "device_activity_records",
        "duration_ns": 100,
    }
    assert evidence.uncaptured_device_activity.to_json() == {
        "count": 1,
        "count_unit": "device_activity_records",
        "duration_ns": 100,
    }
    assert evidence.total_device_activity.to_json() == {
        "count": 2,
        "count_unit": "device_activity_records",
        "duration_ns": 200,
    }
    assert evidence.graph_uncaptured_device_overlap_ns == 0


def test_sqlite_parser_accepts_v14_driver_graph_api_names(tmp_path: Path) -> None:
    path = tmp_path / "driver-api.sqlite"
    _sqlite_fixture(path, graph=True)
    with sqlite3.connect(path) as connection:
        connection.executemany(
            "UPDATE StringIds SET value = ? WHERE id = ?",
            (
                ("cuGraphInstantiateWithFlags", 1),
                ("cuGraphLaunch", 2),
                ("cuGraphExecUpdate", 3),
            ),
        )

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_instantiate.count == 1
    assert evidence.graph_launch.count == 1
    assert evidence.graph_update.count == 1


def test_sqlite_parser_fails_closed_on_missing_graph_node_or_candidate_binding(
    tmp_path: Path,
) -> None:
    missing_column = tmp_path / "missing-column.sqlite"
    _sqlite_fixture(missing_column, graph=False, graph_node_column=False)
    with pytest.raises(CommandBufferControlError, match="graphNodeId"):
        parse_nsys_sqlite(missing_column, _CANDIDATE_SHA256)

    wrong_candidate = tmp_path / "wrong-candidate.sqlite"
    _sqlite_fixture(wrong_candidate, graph=False, candidate_sha256="c" * 64)
    with pytest.raises(CommandBufferControlError, match="exactly one"):
        parse_nsys_sqlite(wrong_candidate, _CANDIDATE_SHA256)


def test_sqlite_parser_accepts_zero_as_uncaptured_graph_node_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "zero-graph-node.sqlite"
    _sqlite_fixture(path, graph=False)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL SET graphNodeId = 0"
        )

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_device_activity.count == 0
    assert evidence.uncaptured_device_activity.count == 1


def test_sqlite_parser_accepts_graph_memcpy_only_launch(tmp_path: Path) -> None:
    path = tmp_path / "graph-memcpy-only.sqlite"
    _sqlite_fixture(path, graph=False)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(200, 215, 2, 77)"
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(400, 500, 77, 9)"
        )

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_launch.count == 1
    assert evidence.graph_device_activity.count == 1
    assert evidence.graph_device_activity.duration_ns == 100
    assert evidence.total_device_activity == evidence.graph_device_activity


def test_sqlite_parser_counts_one_launch_to_many_graph_activities(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one-launch-many-activities.sqlite"
    _sqlite_fixture(path, graph=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(500, 530, 200, 8)"
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
            "VALUES(530, 570, 200, 9, 11)"
        )

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_launch.to_json()["count_unit"] == "api_calls"
    assert evidence.graph_launch.count == 1
    assert evidence.graph_device_activity.to_json()["count_unit"] == (
        "device_activity_records"
    )
    assert evidence.graph_device_activity.count == 3
    assert evidence.graph_device_activity.duration_ns == 170


def test_sqlite_parser_includes_graph_tagged_memset_when_schema_supports_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph-memset.sqlite"
    _sqlite_fixture(path, graph=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(500, 525, 200, 8)"
        )

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_device_activity.count == 2
    assert evidence.graph_device_activity.duration_ns == 125


def test_sqlite_parser_allows_reused_graph_and_launch_correlation_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reused-graph-identity.sqlite"
    _sqlite_fixture(path, graph=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
            "VALUES(500, 540, 200, 8, 11)"
        )

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_launch.count == 1
    assert evidence.graph_device_activity.count == 2


def test_sqlite_parser_reports_graph_direct_temporal_overlap(tmp_path: Path) -> None:
    path = tmp_path / "overlap.sqlite"
    _sqlite_fixture(path, graph=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL SET end = 600 "
            "WHERE graphNodeId = 7"
        )
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
            "SET start = 500, end = 700 WHERE graphNodeId IS NULL"
        )

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_device_activity.duration_ns == 200
    assert evidence.uncaptured_device_activity.duration_ns == 200
    assert evidence.graph_uncaptured_device_overlap_ns == 100
    assert evidence.total_device_activity.duration_ns == 300
    assert (
        evidence.total_device_activity.duration_ns
        == evidence.graph_device_activity.duration_ns
        + evidence.uncaptured_device_activity.duration_ns
        - evidence.graph_uncaptured_device_overlap_ns
    )


def test_sqlite_parser_rejects_unbound_graph_activity_and_launch(
    tmp_path: Path,
) -> None:
    unbound_activity = tmp_path / "unbound-activity.sqlite"
    _sqlite_fixture(unbound_activity, graph=True)
    with sqlite3.connect(unbound_activity) as connection:
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
            "SET correlationId = 999 WHERE graphNodeId = 7"
        )
    with pytest.raises(CommandBufferControlError, match="not bound"):
        parse_nsys_sqlite(unbound_activity, _CANDIDATE_SHA256)

    unbound_launch = tmp_path / "unbound-launch.sqlite"
    _sqlite_fixture(unbound_launch, graph=True)
    with sqlite3.connect(unbound_launch) as connection:
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
            "SET graphNodeId = NULL, graphId = NULL WHERE graphNodeId = 7"
        )
    with pytest.raises(CommandBufferControlError, match="no bound"):
        parse_nsys_sqlite(unbound_launch, _CANDIDATE_SHA256)


def test_sqlite_parser_accepts_schema_without_optional_device_tables(
    tmp_path: Path,
) -> None:
    path = tmp_path / "required-tables-only.sqlite"
    _sqlite_fixture(path, graph=True, optional_device_tables=False)

    evidence = parse_nsys_sqlite(path, _CANDIDATE_SHA256)

    assert evidence.graph_device_activity.count == 1
    assert evidence.total_device_activity.count == 2


@pytest.mark.parametrize("graph_node_id", (-1, 1.5, "graph"))
def test_sqlite_parser_rejects_ambiguous_graph_node_ids(
    tmp_path: Path, graph_node_id: object
) -> None:
    path = tmp_path / "invalid-graph-node.sqlite"
    _sqlite_fixture(path, graph=False)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL SET graphNodeId = ?",
            (graph_node_id,),
        )

    with pytest.raises(CommandBufferControlError, match="graphNodeId"):
        parse_nsys_sqlite(path, _CANDIDATE_SHA256)


@pytest.mark.parametrize("flags", ("", "--xla_gpu_triton_gemm_any=true"))
def test_zero_default_graph_launches_emit_stopped_branch(
    tmp_path: Path, flags: str
) -> None:
    plan = _plan(tmp_path, flags=flags)
    plan.artifact_root.mkdir()
    default_path = plan.lanes[0].sqlite_path
    disabled_path = plan.lanes[1].sqlite_path
    default_path.parent.mkdir()
    disabled_path.parent.mkdir()
    _sqlite_fixture(default_path, graph=False)
    _sqlite_fixture(disabled_path, graph=False)
    default = parse_nsys_sqlite(default_path, _CANDIDATE_SHA256)
    disabled = parse_nsys_sqlite(disabled_path, _CANDIDATE_SHA256)

    document = build_control_evidence(
        plan,
        nsys_version=plan.expected_nsys_version,
        default_evidence=default,
        disabled_evidence=disabled,
        default_wall_ns=1000,
        disabled_wall_ns=1100,
    )

    assert document["outcome"] == "stopped_default_zero_cuda_graph_launches"
    assert document["state"] == "PRODUCED"
    assert document["promotion_eligible"] is False
    assert document["control_included_in_promotion_timing"] is False
    assert document["profiling_limitation"] == PROFILING_LIMITATION
    tool = document["tool"]
    assert isinstance(tool, dict)
    assert tool["cuda_graph_trace"] == "node"
    assert tool["nvtx_library_path"] == str(plan.nvtx_library)
    assert tool["nvtx_library_sha256"] == plan.nvtx_library_sha256
    identity = document["identity"]
    assert isinstance(identity, dict)
    assert identity["specimen_sha256"] == _SPECIMEN_SHA256
    assert identity["gate_checkpoint_sha256"] == _GATE_CHECKPOINT_SHA256
    assert identity["warm_checkpoint_sha256"] == _WARM_CHECKPOINT_SHA256
    assert identity["warm_p50_ns"] == _WARM_P50_NS
    assert identity["lane_id"] == "rtx5090"
    assert identity["gpu_uuid"] == "GPU-1"
    assert identity["runtime_identity_sha256"] == _RUNTIME_IDENTITY_SHA256
    assert identity["input_bundle_sha256"] == _INPUT_BUNDLE_SHA256
    assert "device_uuid" not in identity
    command_buffer = document["command_buffer"]
    assert isinstance(command_buffer, dict)
    assert command_buffer["resolved_configuration"] == flags
    expected_identity = {
        "candidate_sha256": _CANDIDATE_SHA256,
        "specimen_sha256": _SPECIMEN_SHA256,
        "source_sha256": _SOURCE_SHA256,
        "lane_id": "rtx5090",
        "gpu_uuid": "GPU-1",
        "gate_checkpoint_sha256": _GATE_CHECKPOINT_SHA256,
        "warm_checkpoint_sha256": _WARM_CHECKPOINT_SHA256,
        "warm_p50_ns": _WARM_P50_NS,
        "runtime_identity_sha256": _RUNTIME_IDENTITY_SHA256,
        "input_bundle_sha256": _INPUT_BUNDLE_SHA256,
    }
    evidence_path = tmp_path / "command-buffer-evidence.json"
    evidence_path.write_text(json.dumps(document), encoding="utf-8")
    assert (
        validate_command_buffer_control_evidence(document, expected_identity)
        == command_buffer
    )
    serialized = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert (
        validate_command_buffer_control_evidence(serialized, expected_identity)
        == command_buffer
    )
    assert command_buffer["observed_capture_participation"] is False
    assert command_buffer["captured_launch_count"] == 0
    assert command_buffer["ab_control"]["included_in_promotion_timing"] is False
    _validate_command_buffer(command_buffer, "command_buffer")


def test_public_validator_rejects_outer_tool_and_disable_corruption(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    plan.artifact_root.mkdir()
    default_path = plan.lanes[0].sqlite_path
    disabled_path = plan.lanes[1].sqlite_path
    default_path.parent.mkdir()
    disabled_path.parent.mkdir()
    _sqlite_fixture(default_path, graph=False)
    _sqlite_fixture(disabled_path, graph=False)
    document = build_control_evidence(
        plan,
        nsys_version=plan.expected_nsys_version,
        default_evidence=parse_nsys_sqlite(default_path, _CANDIDATE_SHA256),
        disabled_evidence=parse_nsys_sqlite(disabled_path, _CANDIDATE_SHA256),
        default_wall_ns=1000,
        disabled_wall_ns=1100,
    )
    identity = document["identity"]
    assert isinstance(identity, dict)

    outer_corruption = json.loads(json.dumps(document))
    outer_corruption["unexpected"] = True
    with pytest.raises(CommandBufferControlError, match="unexpected keys"):
        validate_command_buffer_control_evidence(outer_corruption, identity)

    plan.nsys_binary.write_bytes(b"tampered")
    with pytest.raises(CommandBufferControlError, match="binary hash mismatch"):
        validate_command_buffer_control_evidence(document, identity)
    plan.nsys_binary.write_bytes(b"tool")

    nvtx_corruption = json.loads(json.dumps(document))
    nvtx_corruption["tool"]["nvtx_library_sha256"] = "0" * 64
    with pytest.raises(CommandBufferControlError, match="NVTX library hash mismatch"):
        validate_command_buffer_control_evidence(nvtx_corruption, identity)

    environment_corruption = json.loads(json.dumps(document))
    environment_corruption["lanes"][0]["environment"]["LD_LIBRARY_PATH"] = "/ambient"
    with pytest.raises(CommandBufferControlError, match="isolated launch"):
        validate_command_buffer_control_evidence(environment_corruption, identity)

    command_corruption = json.loads(json.dumps(document))
    nvtx_option = command_corruption["lanes"][0]["command"].index("--nvtx-library")
    command_corruption["lanes"][0]["command"][nvtx_option + 1] = "/ambient/lib.so"
    with pytest.raises(CommandBufferControlError, match="NVTX binding differs"):
        validate_command_buffer_control_evidence(command_corruption, identity)

    disable_corruption = json.loads(json.dumps(document))
    disabled_lane = disable_corruption["lanes"][1]
    disabled_lane["sqlite_path"] = str(tmp_path / "absent.sqlite")
    evidence = disabled_lane["evidence"]
    evidence["cuda_graph_launch_api"] = {
        "count": 1,
        "count_unit": "api_calls",
        "duration_ns": 10,
    }
    evidence["graph_device_activity"] = {
        "count": 1,
        "count_unit": "device_activity_records",
        "duration_ns": 10,
    }
    evidence["total_device_activity"] = {
        "count": 2,
        "count_unit": "device_activity_records",
        "duration_ns": 210,
    }
    with pytest.raises(CommandBufferControlError, match="explicit-disable"):
        validate_command_buffer_control_evidence(disable_corruption, identity)


def test_graph_default_is_observed_enabled_but_disable_control_must_be_clean(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    plan.artifact_root.mkdir()
    default_path = plan.lanes[0].sqlite_path
    disabled_path = plan.lanes[1].sqlite_path
    default_path.parent.mkdir()
    disabled_path.parent.mkdir()
    _sqlite_fixture(default_path, graph=True)
    _sqlite_fixture(disabled_path, graph=False)
    with sqlite3.connect(default_path) as connection:
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
            "VALUES(700, 750, 200, 8, 11)"
        )
    default = parse_nsys_sqlite(default_path, _CANDIDATE_SHA256)
    disabled = parse_nsys_sqlite(disabled_path, _CANDIDATE_SHA256)

    document = build_control_evidence(
        plan,
        nsys_version=plan.expected_nsys_version,
        default_evidence=default,
        disabled_evidence=disabled,
        default_wall_ns=1000,
        disabled_wall_ns=1100,
    )

    assert document["outcome"] == "observed_default_cuda_graph_launches"
    lanes = document["lanes"]
    assert isinstance(lanes, list)
    assert lanes[0]["command_buffer_state"] == "observed_enabled"
    assert lanes[1]["command_buffer_state"] == "explicitly_disabled"
    command_buffer = document["command_buffer"]
    assert isinstance(command_buffer, dict)
    assert command_buffer["captured_launch_count"] == 1
    assert lanes[0]["evidence"]["graph_device_activity"]["count"] == 2
    _validate_command_buffer(command_buffer, "command_buffer")

    with pytest.raises(CommandBufferControlError, match="disable control"):
        build_control_evidence(
            plan,
            nsys_version=plan.expected_nsys_version,
            default_evidence=default,
            disabled_evidence=default,
            default_wall_ns=1000,
            disabled_wall_ns=1100,
        )


def test_command_buffer_payload_uses_exclusive_direct_device_union(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    plan.artifact_root.mkdir()
    default_path = plan.lanes[0].sqlite_path
    disabled_path = plan.lanes[1].sqlite_path
    default_path.parent.mkdir()
    disabled_path.parent.mkdir()
    _sqlite_fixture(default_path, graph=True)
    _sqlite_fixture(disabled_path, graph=False)
    with sqlite3.connect(default_path) as connection:
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL SET end = 600 "
            "WHERE graphNodeId = 7"
        )
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
            "SET start = 500, end = 700 WHERE graphNodeId IS NULL"
        )
    default = parse_nsys_sqlite(default_path, _CANDIDATE_SHA256)
    disabled = parse_nsys_sqlite(disabled_path, _CANDIDATE_SHA256)

    document = build_control_evidence(
        plan,
        nsys_version=plan.expected_nsys_version,
        default_evidence=default,
        disabled_evidence=disabled,
        default_wall_ns=1000,
        disabled_wall_ns=1100,
    )

    command_buffer = document["command_buffer"]
    assert isinstance(command_buffer, dict)
    assert command_buffer["graph_launched_device_ns"] == 200
    assert command_buffer["uncaptured_device_ns"] == 100
    assert (
        command_buffer["graph_launched_device_ns"]
        + command_buffer["uncaptured_device_ns"]
        == default.total_device_activity.duration_ns
    )


def test_cli_probe_help_does_not_import_or_launch_jax(tmp_path: Path) -> None:
    del tmp_path
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "benchmarks.single_stage_compute_graph_command_buffer_control",
            "probe",
            "--help",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--input-root" in completed.stdout
    assert "--parameter-sha256" in completed.stdout
    assert "--input-bundle-sha256" in completed.stdout


class _NegativeNvtx:
    def __init__(self) -> None:
        self.labels: list[bytes] = []
        self.pop_count = 0

    def nvtxRangePushA(self, label: bytes) -> int:
        self.labels.append(label)
        return -1

    def nvtxRangePop(self) -> int:
        self.pop_count += 1
        return -1


class _Replay:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.evaluation_count = 0

    def evaluate_once(self) -> object:
        self.evaluation_count += 1
        if self.error is not None:
            raise self.error
        return "profiled"


class _PreparedProbe:
    def __init__(self, replay: _Replay) -> None:
        self.replay = replay
        self.warm_evaluation_count = 0

    def evaluate_once(self) -> object:
        self.warm_evaluation_count += 1
        return "warm"

    def fresh_replay(self) -> _Replay:
        return self.replay


def _run_fake_probe(monkeypatch, tmp_path: Path, replay: _Replay) -> _NegativeNvtx:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "input_bundle.json").write_bytes(b"input-bundle")
    candidate_path = tmp_path / "candidate.npy"
    candidate_path.write_bytes(b"candidate")
    prepared = _PreparedProbe(replay)
    nvtx = _NegativeNvtx()
    validated: list[object] = []
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._nvtx_library",
        lambda _path: nvtx,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._canonical_candidate",
        lambda _path, _sha256: "candidate",
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._native_prepare",
        lambda _input_root: lambda candidate: prepared,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._validate_result",
        validated.append,
    )

    run_probe(
        input_root,
        candidate_path,
        _CANDIDATE_SHA256,
        _INPUT_BUNDLE_SHA256,
        tmp_path / "libnvtx.so",
    )

    assert prepared.warm_evaluation_count == 1
    assert validated == ["warm", "profiled"]
    return nvtx


def test_probe_ignores_negative_nvtx_returns_and_always_pops(
    monkeypatch, tmp_path: Path
) -> None:
    replay = _Replay()

    nvtx = _run_fake_probe(monkeypatch, tmp_path, replay)

    assert replay.evaluation_count == 1
    assert nvtx.labels == [
        ("single_stage.compute_graph.evaluation:" + _CANDIDATE_SHA256).encode("ascii")
    ]
    assert nvtx.pop_count == 1


def test_probe_propagates_evaluation_error_after_popping(
    monkeypatch, tmp_path: Path
) -> None:
    class ProbeFailure(RuntimeError):
        pass

    failure = ProbeFailure("evaluation failed")
    replay = _Replay(failure)
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "input_bundle.json").write_bytes(b"input-bundle")
    candidate_path = tmp_path / "candidate.npy"
    candidate_path.write_bytes(b"candidate")
    prepared = _PreparedProbe(replay)
    nvtx = _NegativeNvtx()
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._nvtx_library",
        lambda _path: nvtx,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._canonical_candidate",
        lambda _path, _sha256: "candidate",
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._native_prepare",
        lambda _input_root: lambda candidate: prepared,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_command_buffer_control._validate_result",
        lambda _result: None,
    )

    with pytest.raises(ProbeFailure, match="evaluation failed"):
        run_probe(
            input_root,
            candidate_path,
            _CANDIDATE_SHA256,
            _INPUT_BUNDLE_SHA256,
            tmp_path / "libnvtx.so",
        )

    assert replay.evaluation_count == 1
    assert nvtx.pop_count == 1
