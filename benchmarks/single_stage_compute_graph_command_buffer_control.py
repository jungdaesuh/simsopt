"""External Nsight Systems CUDA-command-buffer control for Phase 0."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from benchmarks.single_stage_compute_graph_c0_evaluator import (
    _canonical_candidate,
    _native_prepare,
    _validate_result,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    ISOLATED_MODULE_BOOTSTRAP,
    build_snapshot_module_launch,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    A100_LANE_ID,
    RTX_LANE_ID,
    canonical_json_bytes,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    LaneId as Phase0LaneId,
)

SCHEMA_ID: Final = "single-stage-compute-graph-command-buffer-control-v3"
NVTX_RANGE_PREFIX: Final = "single_stage.compute_graph.evaluation:"
DISABLE_COMMAND_BUFFER_FLAG: Final = "--xla_gpu_enable_command_buffer="
CUDA_GRAPH_TRACE_GRANULARITY: Final = "node"
PROFILING_OVERRIDE_FLAG: Final = "--xla_enable_command_buffers_during_profiling=true"
PROFILING_LIMITATION: Final = (
    "The local XLA binary disables command buffers while profiling unless "
    "--xla_enable_command_buffers_during_profiling=true is set. This control "
    "does not add that override; SQLite CUDA graph activity is authoritative."
)
LaneId = Literal["current_default", "explicit_disable"]
_IDENTITY_KEYS: Final = frozenset(
    {
        "candidate_sha256",
        "specimen_sha256",
        "source_sha256",
        "lane_id",
        "gpu_uuid",
        "gate_checkpoint_sha256",
        "warm_checkpoint_sha256",
        "warm_p50_ns",
        "runtime_identity_sha256",
        "input_bundle_sha256",
    }
)


class CommandBufferControlError(RuntimeError):
    """The external command-buffer control is incomplete or contradictory."""


@dataclass(frozen=True, slots=True)
class ApiActivity:
    """Explicitly-unitized count and interval-union activity duration."""

    count: int
    duration_ns: int
    count_unit: Literal["api_calls", "device_activity_records"]

    def to_json(self) -> dict[str, int | str]:
        return {
            "count": self.count,
            "count_unit": self.count_unit,
            "duration_ns": self.duration_ns,
        }


@dataclass(frozen=True, slots=True)
class SqliteLaneEvidence:
    """CUDA graph and device activity inside one candidate-bound NVTX range."""

    envelope_start_ns: int
    envelope_end_ns: int
    graph_instantiate: ApiActivity
    graph_launch: ApiActivity
    graph_update: ApiActivity
    graph_device_activity: ApiActivity
    uncaptured_device_activity: ApiActivity
    total_device_activity: ApiActivity
    graph_uncaptured_device_overlap_ns: int

    def to_json(self) -> dict[str, object]:
        return {
            "evaluation_envelope_ns": self.envelope_end_ns - self.envelope_start_ns,
            "cuda_graph_instantiate_api": self.graph_instantiate.to_json(),
            "cuda_graph_launch_api": self.graph_launch.to_json(),
            "cuda_graph_update_api": self.graph_update.to_json(),
            "graph_device_activity": self.graph_device_activity.to_json(),
            "uncaptured_device_activity": self.uncaptured_device_activity.to_json(),
            "total_device_activity": self.total_device_activity.to_json(),
            "graph_uncaptured_device_overlap_ns": (
                self.graph_uncaptured_device_overlap_ns
            ),
        }


@dataclass(frozen=True, slots=True)
class LanePlan:
    """One fresh-process Nsight invocation and its exact environment."""

    lane_id: LaneId
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: Path
    sqlite_path: Path
    xla_flags: str

    def to_json(self) -> dict[str, object]:
        recorded_environment_keys = frozenset(
            {
                "CUDA_VISIBLE_DEVICES",
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "JAX_TRANSFER_GUARD",
                "LD_LIBRARY_PATH",
                "PHASE0_EXPECTED_GPU_UUID",
                "PYTHONNOUSERSITE",
                "PYTHONPATH",
                "PYTHONSAFEPATH",
                "XLA_FLAGS",
            }
        )
        return {
            "lane_id": self.lane_id,
            "command": list(self.command),
            "environment": {
                key: value
                for key, value in self.environment
                if key in recorded_environment_keys
            },
            "cwd": str(self.cwd),
            "sqlite_path": str(self.sqlite_path),
            "resolved_xla_flags": self.xla_flags,
        }


@dataclass(frozen=True, slots=True)
class ControlPlan:
    """Immutable two-lane external profiling plan."""

    nsys_binary: Path
    nvtx_library: Path
    nvtx_library_sha256: str
    expected_nsys_version: str
    artifact_root: Path
    snapshot_root: Path
    specimen_sha256: str
    candidate_sha256: str
    source_sha256: str
    gate_checkpoint_sha256: str
    warm_checkpoint_sha256: str
    warm_p50_ns: float
    lane_id: Phase0LaneId
    gpu_uuid: str
    runtime_identity_sha256: str
    input_bundle_sha256: str
    lanes: tuple[LanePlan, LanePlan]


def _sha256_text(value: str, context: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CommandBufferControlError(f"{context} must be a lowercase SHA-256")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_regular_file(path: Path, context: str) -> Path:
    """Resolve one explicit dependency path and reject absent/non-files."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CommandBufferControlError(f"{context} must exist") from error
    if not resolved.is_file():
        raise CommandBufferControlError(f"{context} must be a regular file")
    return resolved


def _positive_finite_number(value: float, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandBufferControlError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise CommandBufferControlError(f"{context} must be positive and finite")
    return number


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CommandBufferControlError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CommandBufferControlError(f"{context} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        raise CommandBufferControlError(f"{context} has unexpected keys")


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CommandBufferControlError(f"{context} must be a non-empty string")
    return value


def _xla_flags_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise CommandBufferControlError(f"{context} must be a string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CommandBufferControlError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _normalized_xla_flags(value: str) -> tuple[str, ...]:
    try:
        tokens = tuple(shlex.split(value))
    except ValueError as error:
        raise CommandBufferControlError(
            "XLA_FLAGS is not valid shell-like text"
        ) from error
    if any(token.startswith("--xla_gpu_enable_command_buffer") for token in tokens):
        raise CommandBufferControlError(
            "current/default XLA_FLAGS already selects command-buffer operations"
        )
    return tokens


def _lane_plan(
    lane_id: LaneId,
    *,
    nsys_binary: Path,
    nvtx_library: Path,
    python_binary: Path,
    snapshot_root: Path,
    artifact_root: Path,
    cache_root: Path,
    input_root: Path,
    candidate_path: Path,
    candidate_sha256: str,
    input_bundle_sha256: str,
    xla_flags: str,
    gpu_uuid: str,
    base_environment: Mapping[str, str],
) -> LanePlan:
    lane_root = artifact_root / lane_id
    output_prefix = lane_root / "nsys-report"
    sqlite_path = output_prefix.with_suffix(".sqlite")
    cache_path = cache_root / lane_id
    module_args = (
        "probe",
        "--input-root",
        str(input_root),
        "--candidate",
        str(candidate_path),
        "--parameter-sha256",
        candidate_sha256,
        "--input-bundle-sha256",
        input_bundle_sha256,
        "--nvtx-library",
        str(nvtx_library),
    )
    child_environment = dict(base_environment)
    child_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": base_environment.get("CUDA_VISIBLE_DEVICES", "0"),
            "JAX_COMPILATION_CACHE_DIR": str(cache_path),
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
            "LD_LIBRARY_PATH": str(nvtx_library.parent),
            "PHASE0_EXPECTED_GPU_UUID": gpu_uuid,
            "XLA_FLAGS": xla_flags,
        }
    )
    launch = build_snapshot_module_launch(
        python_binary,
        snapshot_root,
        "benchmarks.single_stage_compute_graph_command_buffer_control",
        module_args,
        child_environment,
    )
    command = (
        str(nsys_binary),
        "profile",
        "--trace=cuda,nvtx",
        f"--cuda-graph-trace={CUDA_GRAPH_TRACE_GRANULARITY}",
        "--export=sqlite",
        "--force-overwrite=false",
        f"--output={output_prefix}",
        *launch.argv,
    )
    environment = tuple(sorted(launch.environment.items()))
    return LanePlan(lane_id, command, environment, launch.cwd, sqlite_path, xla_flags)


def build_control_plan(
    *,
    nsys_binary: Path,
    nvtx_library: Path,
    expected_nsys_version: str,
    python_binary: Path,
    snapshot_root: Path,
    artifact_root: Path,
    cache_root: Path,
    input_root: Path,
    candidate_path: Path,
    specimen_sha256: str,
    candidate_sha256: str,
    source_sha256: str,
    gate_checkpoint_sha256: str,
    warm_checkpoint_sha256: str,
    warm_p50_ns: float,
    lane_id: Phase0LaneId,
    gpu_uuid: str,
    runtime_identity_sha256: str,
    input_bundle_sha256: str,
    current_xla_flags: str,
    base_environment: Mapping[str, str] | None = None,
) -> ControlPlan:
    """Build fresh default/disable Nsight invocations without executing them."""

    nsys_binary = nsys_binary.resolve()
    nvtx_library = _resolved_regular_file(nvtx_library, "nvtx_library")
    nvtx_library_sha256 = _sha256_path(nvtx_library)
    if not python_binary.is_absolute():
        raise CommandBufferControlError("python_binary path must be absolute")
    # Do not resolve a virtualenv interpreter symlink into its base runtime.
    python_binary = Path(os.path.abspath(python_binary))
    snapshot_root = snapshot_root.resolve()
    artifact_root = artifact_root.resolve()
    cache_root = cache_root.resolve()
    input_root = input_root.resolve()
    candidate_path = candidate_path.resolve()
    if not nsys_binary.is_file() or not os.access(nsys_binary, os.X_OK):
        raise CommandBufferControlError("nsys_binary must be an executable file")
    if not python_binary.is_file() or not os.access(python_binary, os.X_OK):
        raise CommandBufferControlError("python_binary must be an executable file")
    if not input_root.is_dir() or not candidate_path.is_file():
        raise CommandBufferControlError("input root and candidate file must exist")
    if not expected_nsys_version.strip() or not gpu_uuid.strip():
        raise CommandBufferControlError("Nsight version and GPU UUID are required")
    specimen_sha256 = _sha256_text(specimen_sha256, "specimen_sha256")
    candidate_sha256 = _sha256_text(candidate_sha256, "candidate_sha256")
    source_sha256 = _sha256_text(source_sha256, "source_sha256")
    gate_checkpoint_sha256 = _sha256_text(
        gate_checkpoint_sha256, "gate_checkpoint_sha256"
    )
    warm_checkpoint_sha256 = _sha256_text(
        warm_checkpoint_sha256, "warm_checkpoint_sha256"
    )
    runtime_identity_sha256 = _sha256_text(
        runtime_identity_sha256, "runtime_identity_sha256"
    )
    input_bundle_sha256 = _sha256_text(input_bundle_sha256, "input_bundle_sha256")
    input_bundle_path = input_root / "input_bundle.json"
    if not input_bundle_path.is_file():
        raise CommandBufferControlError("input_root/input_bundle.json must exist")
    if _sha256_path(input_bundle_path) != input_bundle_sha256:
        raise CommandBufferControlError("input_bundle_sha256 does not match input_root")
    warm_p50_ns = _positive_finite_number(warm_p50_ns, "warm_p50_ns")
    if lane_id not in (RTX_LANE_ID, A100_LANE_ID):
        raise CommandBufferControlError(
            f"lane_id must be {RTX_LANE_ID!r} or {A100_LANE_ID!r}"
        )
    if artifact_root.exists() or cache_root.exists():
        raise CommandBufferControlError("artifact and cache roots must both be fresh")
    if (
        artifact_root == cache_root
        or artifact_root.is_relative_to(cache_root)
        or cache_root.is_relative_to(artifact_root)
    ):
        raise CommandBufferControlError("artifact and cache roots must be disjoint")
    current_tokens = _normalized_xla_flags(current_xla_flags)
    resolved_base_environment = (
        os.environ if base_environment is None else base_environment
    )
    current_flags = shlex.join(current_tokens)
    disabled_flags = shlex.join((*current_tokens, DISABLE_COMMAND_BUFFER_FLAG))
    current = _lane_plan(
        "current_default",
        nsys_binary=nsys_binary,
        nvtx_library=nvtx_library,
        python_binary=python_binary,
        snapshot_root=snapshot_root,
        artifact_root=artifact_root,
        cache_root=cache_root,
        input_root=input_root,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        input_bundle_sha256=input_bundle_sha256,
        xla_flags=current_flags,
        gpu_uuid=gpu_uuid,
        base_environment=resolved_base_environment,
    )
    disabled = _lane_plan(
        "explicit_disable",
        nsys_binary=nsys_binary,
        nvtx_library=nvtx_library,
        python_binary=python_binary,
        snapshot_root=snapshot_root,
        artifact_root=artifact_root,
        cache_root=cache_root,
        input_root=input_root,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        input_bundle_sha256=input_bundle_sha256,
        xla_flags=disabled_flags,
        gpu_uuid=gpu_uuid,
        base_environment=resolved_base_environment,
    )
    return ControlPlan(
        nsys_binary,
        nvtx_library,
        nvtx_library_sha256,
        expected_nsys_version.strip(),
        artifact_root,
        snapshot_root,
        specimen_sha256,
        candidate_sha256,
        source_sha256,
        gate_checkpoint_sha256,
        warm_checkpoint_sha256,
        warm_p50_ns,
        lane_id,
        gpu_uuid,
        runtime_identity_sha256,
        input_bundle_sha256,
        (current, disabled),
    )


def _table_columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return frozenset(str(row[1]) for row in rows)


def _require_table(
    connection: sqlite3.Connection, table: str, columns: frozenset[str]
) -> None:
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if table not in present:
        raise CommandBufferControlError(f"Nsight SQLite is missing table {table}")
    missing = columns - _table_columns(connection, table)
    if missing:
        raise CommandBufferControlError(
            f"Nsight SQLite table {table} is missing columns {sorted(missing)}"
        )


def _interval_union_ns(rows: Sequence[tuple[int, int]]) -> int:
    ordered = sorted(rows)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    if end <= start:
        raise CommandBufferControlError("Nsight activity interval is not positive")
    for next_start, next_end in ordered[1:]:
        if next_end <= next_start:
            raise CommandBufferControlError("Nsight activity interval is not positive")
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _activity(
    rows: Sequence[tuple[int, int]],
    count_unit: Literal["api_calls", "device_activity_records"],
) -> ApiActivity:
    return ApiActivity(len(rows), _interval_union_ns(rows), count_unit)


def _graph_node_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CommandBufferControlError(
            "Nsight graphNodeId must be NULL or a non-negative integer"
        )
    return value


def _correlation_id(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CommandBufferControlError(f"{context} must be a non-negative integer")
    return value


def _optional_correlation_id(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _correlation_id(value, context)


def _activity_rows(
    connection: sqlite3.Connection,
    table: str,
    envelope_start_ns: int,
    envelope_end_ns: int,
) -> tuple[tuple[int, int, int | None, int | None], ...]:
    _require_table(
        connection,
        table,
        frozenset({"start", "end", "correlationId", "graphNodeId"}),
    )
    rows = connection.execute(
        f'SELECT start, end, correlationId, graphNodeId FROM "{table}" '
        "WHERE start >= ? AND end <= ? ORDER BY start",
        (envelope_start_ns, envelope_end_ns),
    ).fetchall()
    return tuple(
        (
            int(row[0]),
            int(row[1]),
            _optional_correlation_id(row[2], f"{table}.correlationId"),
            _graph_node_id(row[3]),
        )
        for row in rows
    )


def _single_nvtx_envelope(
    connection: sqlite3.Connection, candidate_sha256: str
) -> tuple[int, int]:
    _require_table(connection, "NVTX_EVENTS", frozenset({"start", "end", "text"}))
    expected = NVTX_RANGE_PREFIX + candidate_sha256
    rows = connection.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = ?", (expected,)
    ).fetchall()
    if len(rows) != 1:
        raise CommandBufferControlError(
            "Nsight SQLite must contain exactly one candidate-bound NVTX range"
        )
    start, end = (int(rows[0][0]), int(rows[0][1]))
    if end <= start:
        raise CommandBufferControlError("candidate-bound NVTX range is not positive")
    return start, end


def parse_nsys_sqlite(path: Path, candidate_sha256: str) -> SqliteLaneEvidence:
    """Parse one Nsight SQLite export and fail closed on schema ambiguity."""

    candidate_sha256 = _sha256_text(candidate_sha256, "candidate_sha256")
    if not path.is_file():
        raise CommandBufferControlError("Nsight SQLite export does not exist")
    uri = f"file:{path.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            start, end = _single_nvtx_envelope(connection, candidate_sha256)
            _require_table(connection, "StringIds", frozenset({"id", "value"}))
            _require_table(
                connection,
                "CUPTI_ACTIVITY_KIND_RUNTIME",
                frozenset({"start", "end", "nameId", "correlationId"}),
            )
            kernel_tables = tuple(
                table
                for table in (
                    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                )
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                is not None
            )
            if len(kernel_tables) != 1:
                raise CommandBufferControlError(
                    "Nsight SQLite must expose exactly one CUDA kernel activity table"
                )
            kernel_table = kernel_tables[0]
            _require_table(
                connection,
                kernel_table,
                frozenset({"start", "end", "correlationId", "graphNodeId"}),
            )
            runtime_rows = connection.execute(
                "SELECT r.start, r.end, s.value, r.correlationId "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r "
                "JOIN StringIds AS s ON s.id = r.nameId "
                "WHERE r.start >= ? AND r.end <= ? ORDER BY r.start",
                (start, end),
            ).fetchall()
            device_tables = (
                kernel_table,
                *(
                    table
                    for table in (
                        "CUPTI_ACTIVITY_KIND_MEMCPY",
                        "CUPTI_ACTIVITY_KIND_MEMSET",
                    )
                    if connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    is not None
                ),
            )
            device_rows = tuple(
                row
                for table in device_tables
                for row in _activity_rows(connection, table, start, end)
            )
    except sqlite3.Error as error:
        raise CommandBufferControlError(
            f"invalid Nsight SQLite export: {error}"
        ) from error
    if not device_rows:
        raise CommandBufferControlError(
            "NVTX envelope contains no CUDA device activity"
        )

    def api_rows(
        prefixes: tuple[str, ...],
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (
                int(row[0]),
                int(row[1]),
                _correlation_id(row[3], "CUDA graph runtime correlationId"),
            )
            for row in runtime_rows
            if any(str(row[2]).startswith(prefix) for prefix in prefixes)
        )

    launch_rows = api_rows(("cudaGraphLaunch", "cuGraphLaunch"))
    launch_correlations = tuple(row[2] for row in launch_rows)
    if len(launch_correlations) != len(set(launch_correlations)):
        raise CommandBufferControlError(
            "CUDA graph launch correlationId values must be unique"
        )
    graph_rows = tuple(
        (
            start_ns,
            end_ns,
            _correlation_id(correlation_id, "graph device correlationId"),
        )
        for start_ns, end_ns, correlation_id, graph_node_id in device_rows
        if graph_node_id is not None and graph_node_id > 0
    )
    uncaptured_rows = tuple(
        (start_ns, end_ns)
        for start_ns, end_ns, _correlation_id_value, graph_node_id in device_rows
        if graph_node_id is None or graph_node_id == 0
    )
    launch_correlation_set = frozenset(launch_correlations)
    graph_correlation_set = frozenset(row[2] for row in graph_rows)
    unbound_graph_correlations = graph_correlation_set - launch_correlation_set
    unbound_launch_correlations = launch_correlation_set - graph_correlation_set
    if unbound_graph_correlations:
        raise CommandBufferControlError(
            "graph-tagged device activity is not bound to a CUDA graph launch"
        )
    if unbound_launch_correlations:
        raise CommandBufferControlError(
            "CUDA graph launch has no bound graph-tagged device activity"
        )
    graph_intervals = tuple((row[0], row[1]) for row in graph_rows)
    all_device_rows = tuple((row[0], row[1]) for row in device_rows)
    graph_device = _activity(graph_intervals, "device_activity_records")
    uncaptured_device = _activity(uncaptured_rows, "device_activity_records")
    total_device = _activity(all_device_rows, "device_activity_records")
    graph_uncaptured_overlap_ns = (
        graph_device.duration_ns
        + uncaptured_device.duration_ns
        - total_device.duration_ns
    )
    if graph_uncaptured_overlap_ns < 0:
        raise CommandBufferControlError(
            "CUDA device activity unions are internally inconsistent"
        )
    return SqliteLaneEvidence(
        start,
        end,
        _activity(
            tuple(
                (row[0], row[1])
                for row in api_rows(("cudaGraphInstantiate", "cuGraphInstantiate"))
            ),
            "api_calls",
        ),
        _activity(tuple((row[0], row[1]) for row in launch_rows), "api_calls"),
        _activity(
            tuple(
                (row[0], row[1])
                for row in api_rows(
                    (
                        "cudaGraphExecUpdate",
                        "cudaGraphExecKernelNodeSetParams",
                        "cuGraphExecUpdate",
                        "cuGraphExecKernelNodeSetParams",
                    )
                )
            ),
            "api_calls",
        ),
        graph_device,
        uncaptured_device,
        total_device,
        graph_uncaptured_overlap_ns,
    )


def build_control_evidence(
    plan: ControlPlan,
    *,
    nsys_version: str,
    default_evidence: SqliteLaneEvidence,
    disabled_evidence: SqliteLaneEvidence,
    default_wall_ns: int,
    disabled_wall_ns: int,
) -> dict[str, object]:
    """Bind two parsed lanes into receipt-compatible stopped/A-B evidence."""

    if nsys_version.strip() != plan.expected_nsys_version:
        raise CommandBufferControlError(
            "observed Nsight Systems version differs from plan"
        )
    if default_wall_ns <= 0 or disabled_wall_ns <= 0:
        raise CommandBufferControlError("lane wall times must be positive")
    if (
        disabled_evidence.graph_launch.count
        or disabled_evidence.graph_device_activity.count
    ):
        raise CommandBufferControlError(
            "explicit command-buffer disable control still contains CUDA graph activity"
        )
    default_observed = default_evidence.graph_launch.count > 0
    default_plan, disabled_plan = plan.lanes
    outcome = (
        "observed_default_cuda_graph_launches"
        if default_observed
        else "stopped_default_zero_cuda_graph_launches"
    )
    return {
        "schema_id": SCHEMA_ID,
        "state": "PRODUCED",
        "outcome": outcome,
        "promotion_eligible": False,
        "control_included_in_promotion_timing": False,
        "profiling_limitation": PROFILING_LIMITATION,
        "tool": {
            "binary_path": str(plan.nsys_binary),
            "binary_sha256": _sha256_path(plan.nsys_binary),
            "nvtx_library_path": str(plan.nvtx_library),
            "nvtx_library_sha256": plan.nvtx_library_sha256,
            "version": nsys_version.strip(),
            "trace": "cuda,nvtx",
            "cuda_graph_trace": CUDA_GRAPH_TRACE_GRANULARITY,
            "export": "sqlite",
        },
        "identity": {
            "specimen_sha256": plan.specimen_sha256,
            "candidate_sha256": plan.candidate_sha256,
            "source_sha256": plan.source_sha256,
            "gate_checkpoint_sha256": plan.gate_checkpoint_sha256,
            "warm_checkpoint_sha256": plan.warm_checkpoint_sha256,
            "warm_p50_ns": plan.warm_p50_ns,
            "lane_id": plan.lane_id,
            "gpu_uuid": plan.gpu_uuid,
            "runtime_identity_sha256": plan.runtime_identity_sha256,
            "input_bundle_sha256": plan.input_bundle_sha256,
        },
        "lanes": [
            {
                **default_plan.to_json(),
                "command_buffer_state": (
                    "observed_enabled" if default_observed else "no_graph_launches"
                ),
                "sqlite_sha256": _sha256_path(default_plan.sqlite_path),
                "sample_wall_ns": default_wall_ns,
                "evidence": default_evidence.to_json(),
            },
            {
                **disabled_plan.to_json(),
                "command_buffer_state": "explicitly_disabled",
                "sqlite_sha256": _sha256_path(disabled_plan.sqlite_path),
                "sample_wall_ns": disabled_wall_ns,
                "evidence": disabled_evidence.to_json(),
            },
        ],
        "command_buffer": {
            "resolved_configuration": default_plan.xla_flags,
            "observed_capture_participation": default_observed,
            "graph_launched_device_ns": (
                default_evidence.graph_device_activity.duration_ns
            ),
            "uncaptured_device_ns": (
                default_evidence.uncaptured_device_activity.duration_ns
                - default_evidence.graph_uncaptured_device_overlap_ns
            ),
            "captured_launch_count": default_evidence.graph_launch.count,
            "uncaptured_launch_count": (
                default_evidence.uncaptured_device_activity.count
            ),
            "ab_control": {
                "control_id": "explicit-command-buffer-disable",
                "resolved_configuration": disabled_plan.xla_flags,
                "sample_wall_ns": [disabled_wall_ns],
                "included_in_promotion_timing": False,
            },
        },
    }


def _validated_activity(
    value: object,
    context: str,
    expected_count_unit: Literal["api_calls", "device_activity_records"],
) -> tuple[int, int]:
    activity = _mapping(value, context)
    _exact_keys(
        activity,
        frozenset({"count", "count_unit", "duration_ns"}),
        context,
    )
    if activity["count_unit"] != expected_count_unit:
        raise CommandBufferControlError(
            f"{context}.count_unit must be {expected_count_unit!r}"
        )
    return (
        _integer(activity["count"], f"{context}.count"),
        _integer(activity["duration_ns"], f"{context}.duration_ns"),
    )


def _validated_lane_evidence(value: object, context: str) -> Mapping[str, object]:
    evidence = _mapping(value, context)
    _exact_keys(
        evidence,
        frozenset(
            {
                "evaluation_envelope_ns",
                "cuda_graph_instantiate_api",
                "cuda_graph_launch_api",
                "cuda_graph_update_api",
                "graph_device_activity",
                "uncaptured_device_activity",
                "total_device_activity",
                "graph_uncaptured_device_overlap_ns",
            }
        ),
        context,
    )
    _integer(
        evidence["evaluation_envelope_ns"],
        f"{context}.evaluation_envelope_ns",
        minimum=1,
    )
    _validated_activity(
        evidence["cuda_graph_instantiate_api"],
        f"{context}.cuda_graph_instantiate_api",
        "api_calls",
    )
    launch_count, _ = _validated_activity(
        evidence["cuda_graph_launch_api"],
        f"{context}.cuda_graph_launch_api",
        "api_calls",
    )
    _validated_activity(
        evidence["cuda_graph_update_api"],
        f"{context}.cuda_graph_update_api",
        "api_calls",
    )
    graph_count, graph_ns = _validated_activity(
        evidence["graph_device_activity"],
        f"{context}.graph_device_activity",
        "device_activity_records",
    )
    uncaptured_count, uncaptured_ns = _validated_activity(
        evidence["uncaptured_device_activity"],
        f"{context}.uncaptured_device_activity",
        "device_activity_records",
    )
    total_count, total_ns = _validated_activity(
        evidence["total_device_activity"],
        f"{context}.total_device_activity",
        "device_activity_records",
    )
    overlap_ns = _integer(
        evidence["graph_uncaptured_device_overlap_ns"],
        f"{context}.graph_uncaptured_device_overlap_ns",
    )
    if (launch_count > 0) != (graph_count > 0):
        raise CommandBufferControlError(
            f"{context} graph launch and graph device evidence contradict"
        )
    if total_count != graph_count + uncaptured_count:
        raise CommandBufferControlError(f"{context} device activity counts contradict")
    if overlap_ns > min(graph_ns, uncaptured_ns):
        raise CommandBufferControlError(
            f"{context} graph/direct overlap exceeds an activity union"
        )
    if total_ns != graph_ns + uncaptured_ns - overlap_ns:
        raise CommandBufferControlError(
            f"{context} device activity interval unions contradict"
        )
    return evidence


def validate_command_buffer_control_evidence(
    value: object, expected_identity: Mapping[str, object]
) -> dict[str, object]:
    """Validate a complete external control artifact and return its receipt payload."""

    document = _mapping(value, "command-buffer control evidence")
    _exact_keys(
        document,
        frozenset(
            {
                "schema_id",
                "state",
                "outcome",
                "promotion_eligible",
                "control_included_in_promotion_timing",
                "profiling_limitation",
                "tool",
                "identity",
                "lanes",
                "command_buffer",
            }
        ),
        "command-buffer control evidence",
    )
    if document["schema_id"] != SCHEMA_ID or document["state"] != "PRODUCED":
        raise CommandBufferControlError(
            "command-buffer control schema or state is unsupported"
        )
    if (
        document["promotion_eligible"] is not False
        or document["control_included_in_promotion_timing"] is not False
        or document["profiling_limitation"] != PROFILING_LIMITATION
    ):
        raise CommandBufferControlError(
            "command-buffer control promotion or profiling disposition is invalid"
        )

    identity = _mapping(document["identity"], "command-buffer identity")
    _exact_keys(identity, _IDENTITY_KEYS, "command-buffer identity")
    _exact_keys(expected_identity, _IDENTITY_KEYS, "expected command-buffer identity")
    if dict(identity) != dict(expected_identity):
        raise CommandBufferControlError("command-buffer identity binding mismatch")
    for key in (
        "candidate_sha256",
        "specimen_sha256",
        "source_sha256",
        "gate_checkpoint_sha256",
        "warm_checkpoint_sha256",
        "runtime_identity_sha256",
        "input_bundle_sha256",
    ):
        _sha256_text(_string(identity[key], f"identity.{key}"), f"identity.{key}")
    _positive_finite_number(identity["warm_p50_ns"], "identity.warm_p50_ns")
    if identity["lane_id"] not in (RTX_LANE_ID, A100_LANE_ID):
        raise CommandBufferControlError("identity.lane_id is not a Phase 0 lane")
    _string(identity["gpu_uuid"], "identity.gpu_uuid")

    tool = _mapping(document["tool"], "command-buffer tool")
    _exact_keys(
        tool,
        frozenset(
            {
                "binary_path",
                "binary_sha256",
                "nvtx_library_path",
                "nvtx_library_sha256",
                "version",
                "trace",
                "cuda_graph_trace",
                "export",
            }
        ),
        "command-buffer tool",
    )
    binary_path = Path(_string(tool["binary_path"], "tool.binary_path"))
    binary_sha256 = _sha256_text(
        _string(tool["binary_sha256"], "tool.binary_sha256"), "tool.binary_sha256"
    )
    nvtx_library_path = Path(
        _string(tool["nvtx_library_path"], "tool.nvtx_library_path")
    )
    nvtx_library_sha256 = _sha256_text(
        _string(tool["nvtx_library_sha256"], "tool.nvtx_library_sha256"),
        "tool.nvtx_library_sha256",
    )
    _string(tool["version"], "tool.version")
    if (
        tool["trace"] != "cuda,nvtx"
        or tool["cuda_graph_trace"] != CUDA_GRAPH_TRACE_GRANULARITY
        or tool["export"] != "sqlite"
    ):
        raise CommandBufferControlError("command-buffer tool configuration is invalid")
    if binary_path.is_file() and _sha256_path(binary_path) != binary_sha256:
        raise CommandBufferControlError("Nsight binary hash mismatch")
    if not nvtx_library_path.is_absolute():
        raise CommandBufferControlError("NVTX library evidence path must be absolute")
    if nvtx_library_path.exists():
        if not nvtx_library_path.is_file():
            raise CommandBufferControlError(
                "NVTX library evidence path is not a regular file"
            )
        if _sha256_path(nvtx_library_path) != nvtx_library_sha256:
            raise CommandBufferControlError("NVTX library hash mismatch")

    lane_values = _sequence(document["lanes"], "command-buffer lanes")
    if len(lane_values) != 2:
        raise CommandBufferControlError("command-buffer control requires two lanes")
    lanes: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    cache_paths: set[str] = set()
    for index, lane_value in enumerate(lane_values):
        context = f"command-buffer lanes[{index}]"
        lane = _mapping(lane_value, context)
        _exact_keys(
            lane,
            frozenset(
                {
                    "lane_id",
                    "command",
                    "environment",
                    "cwd",
                    "sqlite_path",
                    "resolved_xla_flags",
                    "command_buffer_state",
                    "sqlite_sha256",
                    "sample_wall_ns",
                    "evidence",
                }
            ),
            context,
        )
        lane_id = _string(lane["lane_id"], f"{context}.lane_id")
        if lane_id not in ("current_default", "explicit_disable") or lane_id in lanes:
            raise CommandBufferControlError("command-buffer A/B lane IDs are invalid")
        command = _sequence(lane["command"], f"{context}.command")
        if not all(isinstance(argument, str) for argument in command):
            raise CommandBufferControlError(f"{context}.command must contain strings")
        required_arguments = {
            "--trace=cuda,nvtx",
            f"--cuda-graph-trace={CUDA_GRAPH_TRACE_GRANULARITY}",
            "--export=sqlite",
        }
        if not required_arguments.issubset(command) or command[0] != str(binary_path):
            raise CommandBufferControlError(f"{context}.command is inconsistent")
        if len(command) < 13 or tuple(command[8:13]) != (
            "-P",
            "-s",
            "-c",
            ISOLATED_MODULE_BOOTSTRAP,
            "benchmarks.single_stage_compute_graph_command_buffer_control",
        ):
            raise CommandBufferControlError(f"{context}.command is not isolated")
        if "--nvtx-library" not in command:
            raise CommandBufferControlError(f"{context}.command lacks NVTX binding")
        nvtx_argument_index = command.index("--nvtx-library")
        if nvtx_argument_index + 1 >= len(command) or command[
            nvtx_argument_index + 1
        ] != str(nvtx_library_path):
            raise CommandBufferControlError(f"{context}.command NVTX binding differs")
        environment = _mapping(lane["environment"], f"{context}.environment")
        cwd = Path(_string(lane["cwd"], f"{context}.cwd"))
        pythonpath = _string(
            environment.get("PYTHONPATH"), f"{context}.environment.PYTHONPATH"
        )
        if (
            pythonpath != f"{cwd / 'src'}:{cwd}"
            or environment.get("PYTHONNOUSERSITE") != "1"
            or environment.get("PYTHONSAFEPATH") != "1"
            or environment.get("LD_LIBRARY_PATH") != str(nvtx_library_path.parent)
        ):
            raise CommandBufferControlError(f"{context} isolated launch is invalid")
        cache_path = _string(
            environment.get("JAX_COMPILATION_CACHE_DIR"),
            f"{context}.environment.JAX_COMPILATION_CACHE_DIR",
        )
        cache_paths.add(cache_path)
        resolved_flags = lane["resolved_xla_flags"]
        if environment.get("XLA_FLAGS") != resolved_flags:
            raise CommandBufferControlError(f"{context} XLA_FLAGS contradict")
        sqlite_path = Path(_string(lane["sqlite_path"], f"{context}.sqlite_path"))
        sqlite_sha256 = _sha256_text(
            _string(lane["sqlite_sha256"], f"{context}.sqlite_sha256"),
            f"{context}.sqlite_sha256",
        )
        _integer(lane["sample_wall_ns"], f"{context}.sample_wall_ns", minimum=1)
        evidence = _validated_lane_evidence(lane["evidence"], f"{context}.evidence")
        if sqlite_path.is_file():
            if _sha256_path(sqlite_path) != sqlite_sha256:
                raise CommandBufferControlError(f"{context} SQLite hash mismatch")
            parsed = parse_nsys_sqlite(
                sqlite_path, _string(identity["candidate_sha256"], "candidate_sha256")
            )
            if parsed.to_json() != dict(evidence):
                raise CommandBufferControlError(
                    f"{context} SQLite evidence does not match the export"
                )
        lanes[lane_id] = (lane, evidence)
    if len(cache_paths) != 2:
        raise CommandBufferControlError("command-buffer A/B caches must be distinct")

    current_lane, current = lanes["current_default"]
    disabled_lane, disabled = lanes["explicit_disable"]
    current_flags = _xla_flags_string(
        current_lane["resolved_xla_flags"], "current/default XLA_FLAGS"
    )
    if DISABLE_COMMAND_BUFFER_FLAG in current_flags:
        raise CommandBufferControlError(
            "current/default lane cannot explicitly disable command buffers"
        )
    current_launch_count, _ = _validated_activity(
        current["cuda_graph_launch_api"],
        "current cuda_graph_launch_api",
        "api_calls",
    )
    _, current_graph_ns = _validated_activity(
        current["graph_device_activity"],
        "current graph_device_activity",
        "device_activity_records",
    )
    current_uncaptured_count, current_uncaptured_ns = _validated_activity(
        current["uncaptured_device_activity"],
        "current uncaptured_device_activity",
        "device_activity_records",
    )
    current_overlap_ns = _integer(
        current["graph_uncaptured_device_overlap_ns"],
        "current graph_uncaptured_device_overlap_ns",
    )
    current_uncaptured_exclusive_ns = current_uncaptured_ns - current_overlap_ns
    disabled_launch_count, disabled_launch_ns = _validated_activity(
        disabled["cuda_graph_launch_api"],
        "disabled cuda_graph_launch_api",
        "api_calls",
    )
    disabled_graph_count, disabled_graph_ns = _validated_activity(
        disabled["graph_device_activity"],
        "disabled graph_device_activity",
        "device_activity_records",
    )
    if (
        disabled_launch_count
        or disabled_launch_ns
        or disabled_graph_count
        or disabled_graph_ns
    ):
        raise CommandBufferControlError(
            "explicit-disable lane contains CUDA graph activity"
        )
    if disabled_lane["command_buffer_state"] != "explicitly_disabled" or (
        DISABLE_COMMAND_BUFFER_FLAG
        not in _string(disabled_lane["resolved_xla_flags"], "disabled XLA_FLAGS")
    ):
        raise CommandBufferControlError(
            "explicit-disable lane is not explicitly disabled"
        )
    default_observed = current_launch_count > 0
    expected_outcome = (
        "observed_default_cuda_graph_launches"
        if default_observed
        else "stopped_default_zero_cuda_graph_launches"
    )
    expected_state = "observed_enabled" if default_observed else "no_graph_launches"
    if (
        document["outcome"] != expected_outcome
        or current_lane["command_buffer_state"] != expected_state
    ):
        raise CommandBufferControlError(
            "default lane outcome contradicts trace evidence"
        )

    command_buffer = _mapping(document["command_buffer"], "command_buffer payload")
    _exact_keys(
        command_buffer,
        frozenset(
            {
                "resolved_configuration",
                "observed_capture_participation",
                "graph_launched_device_ns",
                "uncaptured_device_ns",
                "captured_launch_count",
                "uncaptured_launch_count",
                "ab_control",
            }
        ),
        "command_buffer payload",
    )
    if (
        command_buffer["resolved_configuration"] != current_lane["resolved_xla_flags"]
        or command_buffer["observed_capture_participation"] is not default_observed
        or command_buffer["graph_launched_device_ns"] != current_graph_ns
        or command_buffer["uncaptured_device_ns"] != current_uncaptured_exclusive_ns
        or command_buffer["captured_launch_count"] != current_launch_count
        or command_buffer["uncaptured_launch_count"] != current_uncaptured_count
    ):
        raise CommandBufferControlError(
            "command_buffer payload contradicts lane evidence"
        )
    control = _mapping(command_buffer["ab_control"], "command_buffer.ab_control")
    _exact_keys(
        control,
        frozenset(
            {
                "control_id",
                "resolved_configuration",
                "sample_wall_ns",
                "included_in_promotion_timing",
            }
        ),
        "command_buffer.ab_control",
    )
    samples = _sequence(
        control["sample_wall_ns"], "command_buffer.ab_control.sample_wall_ns"
    )
    if (
        control["control_id"] != "explicit-command-buffer-disable"
        or control["resolved_configuration"] != disabled_lane["resolved_xla_flags"]
        or samples != [disabled_lane["sample_wall_ns"]]
        or control["included_in_promotion_timing"] is not False
    ):
        raise CommandBufferControlError("command-buffer A/B control is inconsistent")
    return dict(command_buffer)


def execute_control_plan(plan: ControlPlan, output_path: Path) -> dict[str, object]:
    """Run both fresh Nsight children and write one exclusive evidence document."""

    resolved_nvtx_library = _resolved_regular_file(plan.nvtx_library, "nvtx_library")
    if (
        resolved_nvtx_library != plan.nvtx_library
        or _sha256_path(resolved_nvtx_library) != plan.nvtx_library_sha256
    ):
        raise CommandBufferControlError("NVTX library changed after plan construction")
    _nvtx_library(resolved_nvtx_library)
    version = subprocess.run(
        (str(plan.nsys_binary), "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != plan.expected_nsys_version:
        raise CommandBufferControlError("Nsight Systems version preflight failed")
    plan.artifact_root.mkdir(parents=True, exist_ok=False)
    evidences: list[SqliteLaneEvidence] = []
    wall_times: list[int] = []
    for lane in plan.lanes:
        lane.sqlite_path.parent.mkdir(parents=True, exist_ok=False)
        cache_path = Path(dict(lane.environment)["JAX_COMPILATION_CACHE_DIR"])
        cache_path.mkdir(parents=True, exist_ok=False)
        environment = dict(lane.environment)
        started_ns = time.perf_counter_ns()
        subprocess.run(lane.command, check=True, env=environment, cwd=lane.cwd)
        wall_times.append(time.perf_counter_ns() - started_ns)
        evidences.append(parse_nsys_sqlite(lane.sqlite_path, plan.candidate_sha256))
    document = build_control_evidence(
        plan,
        nsys_version=version,
        default_evidence=evidences[0],
        disabled_evidence=evidences[1],
        default_wall_ns=wall_times[0],
        disabled_wall_ns=wall_times[1],
    )
    with output_path.open("xb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    return document


def _nvtx_library(path: Path) -> ctypes.CDLL:
    path = _resolved_regular_file(path, "nvtx_library")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise CommandBufferControlError(
            "explicit NVTX library is unloadable"
        ) from error
    library.nvtxRangePushA.argtypes = (ctypes.c_char_p,)
    library.nvtxRangePushA.restype = ctypes.c_int
    library.nvtxRangePop.argtypes = ()
    library.nvtxRangePop.restype = ctypes.c_int
    return library


def run_probe(
    input_root: Path,
    candidate_path: Path,
    candidate_sha256: str,
    input_bundle_sha256: str,
    nvtx_library_path: Path,
) -> None:
    """Run one prewarmed evaluation inside an exact NVTX range, without JAX profiler."""

    candidate_sha256 = _sha256_text(candidate_sha256, "candidate_sha256")
    input_bundle_sha256 = _sha256_text(input_bundle_sha256, "input_bundle_sha256")
    input_bundle_path = input_root / "input_bundle.json"
    if (
        not input_bundle_path.is_file()
        or _sha256_path(input_bundle_path) != input_bundle_sha256
    ):
        raise CommandBufferControlError("probe input bundle binding mismatch")
    nvtx = _nvtx_library(nvtx_library_path)
    candidate = _canonical_candidate(candidate_path, candidate_sha256)
    prepared = _native_prepare(input_root)(candidate)
    _validate_result(prepared.evaluate_once())
    replay = prepared.fresh_replay()
    label = (NVTX_RANGE_PREFIX + candidate_sha256).encode("ascii")
    nvtx.nvtxRangePushA(label)
    try:
        result = replay.evaluate_once()
    finally:
        nvtx.nvtxRangePop()
    _validate_result(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--input-root", type=Path, required=True)
    probe.add_argument("--candidate", type=Path, required=True)
    probe.add_argument("--parameter-sha256", required=True)
    probe.add_argument("--input-bundle-sha256", required=True)
    probe.add_argument("--nvtx-library", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_probe(
            args.input_root,
            args.candidate,
            args.parameter_sha256,
            args.input_bundle_sha256,
            args.nvtx_library,
        )
    except (OSError, ValueError, RuntimeError) as error:
        sys.stderr.write(f"command-buffer probe failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
