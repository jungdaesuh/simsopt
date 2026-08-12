"""Isolated C0 measurement child orchestrator for compute-graph Phase 0.

The runner constructs the canonical evaluator command itself.  Every sample is
an isolated process with a unique trace directory and a shared identity anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmarks.single_stage_compute_graph_isolated_launch import (
    SnapshotModuleLaunch,
    build_snapshot_module_launch,
    normalize_route_environment,
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    A100_LANE_ID,
    FIRST_EVALUATION_LIMIT_NS,
    FORMAL_COMPLETE_PATH_FACTOR,
    PHASE0_GRADIENT_ATOL,
    PHASE0_GRADIENT_RTOL,
    PHASE0_OBJECTIVE_ATOL,
    PHASE0_OBJECTIVE_RTOL,
    REQUIRED_WARM_SAMPLES,
    RTX_LANE_ID,
    SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    LaneId,
    Phase0ReceiptError,
    _interval_union_ns,
    _validate_attribution_control,
    _validate_first_evaluation,
    _validate_qualification,
    canonical_json_bytes,
    canonical_sha256,
    validate_phase0_receipt,
    write_phase0_receipt,
)

C0_RUNNER_SPEC_SCHEMA_ID: Final = "single-stage-compute-graph-c0-runner-spec-v3"
C0_CHILD_OBSERVATION_SCHEMA_ID: Final = (
    "single-stage-compute-graph-c0-child-observation-v3"
)
C0_FAILURE_SCHEMA_ID: Final = "single-stage-compute-graph-c0-failure-v2"
RECEIPT_FILENAME: Final = "phase0-receipt.json"
FAILURE_FILENAME: Final = "failure.json"
CHILD_DIRECTORY_NAME: Final = "children"
EVALUATOR_MODULE: Final = "benchmarks.single_stage_compute_graph_c0_evaluator"
IDENTITY_ANCHOR_FILENAME: Final = "hlo-module-set-identity-anchor.json"
RAW_CHILD_FILENAME: Final = "raw.json"
GATE_CHECKPOINT_FILENAME: Final = "gate-checkpoint.json"
WARM_CHECKPOINT_FILENAME: Final = "warm-checkpoint.json"
STATE_FILENAME: Final = "state.json"
RAW_TELEMETRY_DIRECTORY_NAME: Final = "raw-telemetry"
GATE_CHECKPOINT_SCHEMA_ID: Final = "single-stage-compute-graph-c0-gate-checkpoint-v1"
WARM_CHECKPOINT_SCHEMA_ID: Final = "single-stage-compute-graph-c0-warm-checkpoint-v1"
PENDING_SCHEMA_ID: Final = "single-stage-compute-graph-c0-state-v1"
NATIVE_REFERENCE_SCHEMA_ID: Final = "single-stage-compute-graph-native-reference-v3"
NFEV_SEMANTICS: Final = (
    "scipy_optimize_result_nfev_for_combined_objective_and_gradient_callable_within_"
    "complete_path_boundary"
)
PROCESS_TREE_RSS_SOURCE: Final = (
    "linux-proc-task-children-status-vmrss-root-starttime-v1"
)
PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS: Final = 10_000_000


class C0RunnerError(RuntimeError):
    """The C0 measurement could not produce a valid Phase 0 receipt."""


@dataclass(frozen=True, slots=True)
class ProcessTreeRssEvidence:
    """Parent-observed RSS sum for one child process tree."""

    peak_bytes: int
    sample_count: int
    sample_interval_ns: int
    source: str
    root_pid: int
    root_starttime_ticks: int


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured evaluator-child process result."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_ns: int
    timed_out: bool = False
    process_tree_rss: ProcessTreeRssEvidence | None = None


@dataclass(frozen=True, slots=True)
class C0ExecutionInputs:
    """Validated paths and identities propagated to every child."""

    interpreter_path: str
    snapshot_root: Path
    input_root: Path
    input_bundle_sha256: str
    input_fingerprint: str
    configuration_fingerprint: str
    candidate_path: Path
    parameter_sha256: str
    gpu_uuid: str
    cache_directory: Path
    native_reference_path: Path
    runtime_contract_json: str
    runtime_identity_sha256: str


CommandExecutor = Callable[
    [Sequence[str], Mapping[str, str], Path, float], CommandResult
]


def _linux_proc_identity(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    remainder = stat.rsplit(")", maxsplit=1)[1].split()
    return int(remainder[1]), int(remainder[19])


def _linux_process_tree_rss_bytes(
    root_pid: int, root_starttime: int, proc_root: Path = Path("/proc")
) -> int:
    total = 0
    pending: list[tuple[int, int | None]] = [(root_pid, None)]
    seen: set[int] = set()
    while pending:
        pid, expected_parent = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            parent_pid, starttime = _linux_proc_identity(pid, proc_root)
            if (pid == root_pid and starttime != root_starttime) or (
                expected_parent is not None and parent_pid != expected_parent
            ):
                continue
            process_root = proc_root / str(pid)
            child_pids: set[int] = set()
            with os.scandir(process_root / "task") as tasks:
                for task in tasks:
                    if task.name.isdecimal():
                        child_pids.update(
                            int(child)
                            for child in (Path(task.path) / "children")
                            .read_text(encoding="utf-8")
                            .split()
                        )
            status = (process_root / "status").read_text(encoding="utf-8")
            if _linux_proc_identity(pid, proc_root) != (parent_pid, starttime):
                continue
        except (OSError, IndexError, ValueError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
                break
        pending.extend((child_pid, pid) for child_pid in child_pids)
    return total


class _LinuxProcessTreeRssSampler:
    def __init__(self, root_pid: int) -> None:
        self._root_pid = root_pid
        _, self._root_starttime = _linux_proc_identity(root_pid)
        self._stop = threading.Event()
        self._peak_bytes = 0
        self._sample_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self._peak_bytes = max(
            self._peak_bytes,
            _linux_process_tree_rss_bytes(self._root_pid, self._root_starttime),
        )
        self._sample_count += 1

    def _run(self) -> None:
        next_sample_ns = time.monotonic_ns() + PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
        while True:
            remaining_seconds = max(
                0.0, (next_sample_ns - time.monotonic_ns()) / 1_000_000_000
            )
            if self._stop.wait(remaining_seconds):
                return
            self._sample()
            next_sample_ns += PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def finish(self) -> ProcessTreeRssEvidence:
        self._stop.set()
        self._thread.join()
        return ProcessTreeRssEvidence(
            peak_bytes=self._peak_bytes,
            sample_count=self._sample_count,
            sample_interval_ns=PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
            source=PROCESS_TREE_RSS_SOURCE,
            root_pid=self._root_pid,
            root_starttime_ticks=self._root_starttime,
        )


def _subprocess_executor(
    argv: Sequence[str],
    environment: Mapping[str, str],
    cwd: Path,
    timeout_seconds: float,
) -> CommandResult:
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        tuple(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(environment),
        cwd=cwd,
        start_new_session=True,
    )
    sampler = _LinuxProcessTreeRssSampler(process.pid)
    sampler.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        elapsed_ns = time.monotonic_ns() - started_ns
        return CommandResult(
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            elapsed_ns=elapsed_ns,
            timed_out=True,
            process_tree_rss=sampler.finish(),
        )
    return CommandResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_ns=time.monotonic_ns() - started_ns,
        process_tree_rss=sampler.finish(),
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise C0RunnerError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise C0RunnerError(f"{context} must be a JSON array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise C0RunnerError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise C0RunnerError(f"{context} must be an integer of at least {minimum}")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise C0RunnerError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _lane_id(value: object) -> LaneId:
    if value == RTX_LANE_ID:
        return RTX_LANE_ID
    if value == A100_LANE_ID:
        return A100_LANE_ID
    raise C0RunnerError(f"lane_id must be {RTX_LANE_ID!r} or {A100_LANE_ID!r}")


def _load_json_object(path: Path, context: str) -> Mapping[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise C0RunnerError(f"{context} contains duplicate key {key!r}")
            document[key] = value
        return document

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                C0RunnerError(f"{context} contains non-finite constant {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise C0RunnerError(f"{context} is not valid JSON: {error}") from error
    return _mapping(value, context)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_json(path: Path, document: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(document)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _runtime_identity(provenance: Mapping[str, object]) -> str:
    lexical_interpreter = Path(
        _string(provenance.get("interpreter_path"), "provenance.interpreter_path")
    )
    interpreter = lexical_interpreter.resolve()
    runtime = _mapping(provenance.get("runtime"), "provenance.runtime")
    environment = _mapping(provenance.get("environment"), "provenance.environment")
    policies = _mapping(provenance.get("policies"), "provenance.policies")
    return canonical_sha256(
        {
            "lexical_interpreter_path": str(lexical_interpreter),
            "resolved_interpreter_path": str(interpreter),
            "resolved_interpreter_sha256": _sha256_path(interpreter),
            "runtime_provenance": dict(runtime),
            "effective_environment": dict(environment),
            "numerical_policies": dict(policies),
        }
    )


def _load_canonical_json_object(path: Path, context: str) -> Mapping[str, object]:
    raw = path.read_bytes()
    document = _load_json_object(path, context)
    if raw != canonical_json_bytes(document):
        raise C0RunnerError(f"{context} is not canonical JSON")
    return document


def _native_reference(
    path: Path,
    parameter_sha256: str,
    *,
    expected_bindings: Mapping[str, object],
) -> tuple[Mapping[str, object], str, Mapping[str, object]]:
    document = _load_json_object(path, "native reference")
    if document.get("schema_id") != NATIVE_REFERENCE_SCHEMA_ID:
        raise C0RunnerError("unsupported native-reference schema")
    expected_fields = frozenset(
        {
            "schema_id",
            "identity",
            "parameter_sha256",
            "objective_dtype",
            "objective",
            "gradient_dtype",
            "gradient",
            "inner_newton_success",
            "residual_certificates",
            "elapsed_ns",
            "initial_evaluation",
            "baseline_anchor",
        }
    )
    if frozenset(document) != expected_fields:
        raise C0RunnerError("native reference has unexpected or missing fields")
    if _sha256(document.get("parameter_sha256"), "native parameter_sha256") != (
        parameter_sha256
    ):
        raise C0RunnerError("native reference is bound to a different candidate")
    identity = _mapping(document.get("identity"), "native reference identity")
    if frozenset(identity) != frozenset(expected_bindings):
        raise C0RunnerError(
            "native reference identity has unexpected or missing fields"
        )
    for field, expected in expected_bindings.items():
        if identity.get(field) != expected:
            raise C0RunnerError(f"native reference {field} binding mismatch")
    if (
        document.get("objective_dtype") != "float64"
        or document.get("gradient_dtype") != "float64"
    ):
        raise C0RunnerError("native reference must contain FP64 objective and gradient")
    gradient = _sequence(document.get("gradient"), "native reference gradient")
    if len(gradient) != 461:
        raise C0RunnerError("native reference gradient must contain exactly 461 values")
    anchor = _mapping(document.get("baseline_anchor"), "native baseline_anchor")
    initial_parameter_sha256 = _sha256(
        anchor.get("parameter_sha256"), "native baseline parameter_sha256"
    )
    _sha256(anchor.get("surface_sha256"), "native baseline surface_sha256")
    if anchor.get("inner_solver_success") is not True:
        raise C0RunnerError("native baseline anchor did not solve successfully")
    if document.get("inner_newton_success") is not True:
        raise C0RunnerError("native candidate reference did not solve successfully")
    initial = _mapping(document.get("initial_evaluation"), "native initial_evaluation")
    if frozenset(initial) != frozenset(
        {
            "parameter_sha256",
            "objective_dtype",
            "objective",
            "gradient_dtype",
            "gradient",
            "inner_newton_success",
            "residual_certificates",
            "elapsed_ns",
        }
    ):
        raise C0RunnerError(
            "native initial_evaluation has unexpected or missing fields"
        )
    if (
        _sha256(
            initial.get("parameter_sha256"),
            "native initial_evaluation.parameter_sha256",
        )
        != initial_parameter_sha256
    ):
        raise C0RunnerError(
            "native initial evaluation is not bound to the baseline anchor"
        )
    if (
        initial.get("objective_dtype") != "float64"
        or initial.get("gradient_dtype") != "float64"
    ):
        raise C0RunnerError(
            "native initial evaluation must contain FP64 objective and gradient"
        )
    initial_objective = initial.get("objective")
    if (
        isinstance(initial_objective, bool)
        or not isinstance(initial_objective, float)
        or not math.isfinite(initial_objective)
    ):
        raise C0RunnerError("native initial objective must be a finite float")
    initial_gradient = _sequence(
        initial.get("gradient"), "native initial_evaluation.gradient"
    )
    if len(initial_gradient) != 461 or any(
        isinstance(value, bool)
        or not isinstance(value, float)
        or not math.isfinite(value)
        for value in initial_gradient
    ):
        raise C0RunnerError(
            "native initial gradient must contain exactly 461 finite floats"
        )
    if initial.get("inner_newton_success") is not True:
        raise C0RunnerError("native initial evaluation did not solve successfully")
    initial_residuals = _mapping(
        initial.get("residual_certificates"),
        "native initial_evaluation.residual_certificates",
    )
    if not initial_residuals or any(
        isinstance(value, bool)
        or not isinstance(value, float)
        or not math.isfinite(value)
        or value < 0.0
        for value in initial_residuals.values()
    ):
        raise C0RunnerError(
            "native initial residual certificates must be finite nonnegative floats"
        )
    _integer(
        initial.get("elapsed_ns"),
        "native initial_evaluation.elapsed_ns",
        minimum=1,
    )
    reference = {
        "native_objective": document.get("objective"),
        "native_gradient": list(gradient),
        "objective_atol": PHASE0_OBJECTIVE_ATOL,
        "objective_rtol": PHASE0_OBJECTIVE_RTOL,
        "gradient_atol": PHASE0_GRADIENT_ATOL,
        "gradient_rtol": PHASE0_GRADIENT_RTOL,
        "initial_parameter_sha256": initial_parameter_sha256,
        "native_initial_objective": initial_objective,
        "native_initial_gradient": list(initial_gradient),
    }
    return document, _sha256_path(path), reference


def _child_observation(result: CommandResult, context: str) -> Mapping[str, object]:
    if result.timed_out:
        raise C0RunnerError(f"{context} timed out")
    if result.returncode != 0:
        raise C0RunnerError(
            f"{context} failed with return code {result.returncode}: {result.stderr.strip()}"
        )
    try:
        value = json.loads(
            result.stdout,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                C0RunnerError(f"{context} contains non-finite constant {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise C0RunnerError(f"{context} did not emit valid JSON: {error}") from error
    document = _mapping(value, context)
    if document.get("schema_id") != C0_CHILD_OBSERVATION_SCHEMA_ID:
        raise C0RunnerError(f"{context} has an unsupported schema_id")
    forbidden = frozenset(
        {
            "peak_process_tree_rss_bytes",
            "process_tree_rss_sample_count",
            "process_tree_rss_sample_interval_ns",
            "process_tree_rss_source",
            "process_tree_rss_root_pid",
            "process_tree_rss_root_starttime_ticks",
        }
    )
    if forbidden.intersection(document):
        raise C0RunnerError(
            f"{context} must not self-report parent-owned process-tree RSS"
        )
    return {**dict(document), **_command_rss_document(result, context)}


def _command_rss_document(result: CommandResult, context: str) -> dict[str, object]:
    evidence = result.process_tree_rss
    if (
        evidence is None
        or evidence.source != PROCESS_TREE_RSS_SOURCE
        or evidence.sample_interval_ns != PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
        or evidence.sample_count <= 0
        or evidence.peak_bytes <= 0
        or evidence.root_pid <= 0
        or evidence.root_starttime_ticks <= 0
    ):
        raise C0RunnerError(f"{context} process-tree RSS evidence is invalid")
    return {
        "peak_process_tree_rss_bytes": evidence.peak_bytes,
        "process_tree_rss_sample_count": evidence.sample_count,
        "process_tree_rss_sample_interval_ns": evidence.sample_interval_ns,
        "process_tree_rss_source": evidence.source,
        "process_tree_rss_root_pid": evidence.root_pid,
        "process_tree_rss_root_starttime_ticks": evidence.root_starttime_ticks,
    }


def _target_lane(
    receipt: Mapping[str, object], lane_id: LaneId
) -> tuple[Mapping[str, object], int]:
    lanes = _sequence(receipt.get("lanes"), "receipt_template.lanes")
    matches: list[tuple[Mapping[str, object], int]] = []
    for index, raw_lane in enumerate(lanes):
        lane = _mapping(raw_lane, f"receipt_template.lanes[{index}]")
        if lane.get("lane_id") == lane_id:
            matches.append((lane, index))
    if len(matches) != 1:
        raise C0RunnerError(f"receipt template must contain one {lane_id} lane")
    return matches[0]


def _validated_spec(
    spec: Mapping[str, object],
) -> tuple[LaneId, int, C0ExecutionInputs]:
    if spec.get("schema_id") != C0_RUNNER_SPEC_SCHEMA_ID:
        raise C0RunnerError("unsupported runner spec schema")
    forbidden = frozenset(
        {
            "first_evaluation_reference",
            "profile",
            "command_buffer",
            "newton_telemetry",
            "gap_budget",
        }
    )
    present = forbidden.intersection(spec)
    if present:
        raise C0RunnerError(
            f"runner spec must not contain precomputed derived evidence: {sorted(present)}"
        )
    lane_id = _lane_id(spec.get("lane_id"))
    warm_sample_count = _integer(
        spec.get("warm_sample_count"),
        "warm_sample_count",
        minimum=REQUIRED_WARM_SAMPLES,
    )
    receipt = _mapping(spec.get("receipt_template"), "receipt_template")
    lane, index = _target_lane(receipt, lane_id)
    qualification = lane.get("qualification")
    try:
        outcome = _validate_qualification(
            qualification, lane_id, f"receipt_template.lanes[{index}].qualification"
        )
    except Phase0ReceiptError as error:
        raise C0RunnerError(f"invalid target qualification: {error}") from error
    if outcome != "qualified":
        raise C0RunnerError(f"{lane_id} qualification is blocked")
    if lane.get("measurement") is not None:
        raise C0RunnerError(
            f"{lane_id} receipt template already contains a measurement"
        )
    provenance = _mapping(spec.get("provenance"), "provenance")
    runtime_identity_sha256 = _runtime_identity(provenance)
    runtime_contract_json = json.dumps(
        {
            "runtime": provenance["runtime"],
            "static_environment": provenance["environment"],
            "route_environment": {},
            "policies": provenance["policies"],
            "expected_runtime_identity_sha256": runtime_identity_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_directory = _string(
        provenance.get("compilation_cache_directory"),
        "provenance.compilation_cache_directory",
    )
    output_root = Path(_string(spec.get("output_root"), "output_root")).resolve()
    cache_root = Path(cache_directory).resolve()
    if (
        output_root == cache_root
        or output_root.is_relative_to(cache_root)
        or cache_root.is_relative_to(output_root)
    ):
        raise C0RunnerError("artifact root and compilation cache must be disjoint")
    if cache_root.exists():
        raise C0RunnerError("compilation cache must not exist before the first child")
    interpreter_path = _string(
        provenance.get("interpreter_path"), "provenance.interpreter_path"
    )
    interpreter = Path(interpreter_path)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise C0RunnerError("provenance interpreter_path must be executable")
    snapshot_root = Path(
        _string(provenance.get("immutable_root"), "provenance.immutable_root")
    ).resolve()
    if not snapshot_root.is_dir():
        raise C0RunnerError("provenance immutable_root must be an existing directory")
    allocation = _mapping(provenance.get("allocation"), "provenance.allocation")
    gpu_uuid = _string(allocation.get("gpu_uuid"), "provenance.allocation.gpu_uuid")
    specimen = _mapping(receipt.get("specimen"), "receipt_template.specimen")
    parameter_sha256 = _sha256(
        specimen.get("parameter_sha256"),
        "receipt_template.specimen.parameter_sha256",
    )
    input_root = Path(_string(spec.get("input_root"), "input_root")).resolve()
    candidate_path = Path(
        _string(spec.get("candidate_path"), "candidate_path")
    ).resolve()
    native_reference_path = Path(
        _string(spec.get("native_reference_path"), "native_reference_path")
    ).resolve()
    if not input_root.is_dir():
        raise C0RunnerError("input_root must be an existing directory")
    input_bundle_path = input_root / "input_bundle.json"
    if not input_bundle_path.is_file():
        raise C0RunnerError("input_root must contain input_bundle.json")
    input_bundle_sha256 = _sha256(
        specimen.get("input_bundle_sha256"),
        "receipt_template.specimen.input_bundle_sha256",
    )
    if _sha256_path(input_bundle_path) != input_bundle_sha256:
        raise C0RunnerError("raw input_bundle.json SHA-256 differs from specimen")
    input_bundle = _load_json_object(input_bundle_path, "input_bundle.json")
    input_fingerprint = _sha256(
        input_bundle.get("input_fingerprint"), "input_bundle.input_fingerprint"
    )
    configuration_fingerprint = _sha256(
        input_bundle.get("configuration_fingerprint"),
        "input_bundle.configuration_fingerprint",
    )
    if not candidate_path.is_file():
        raise C0RunnerError("candidate_path must be an existing file")
    if not native_reference_path.is_file():
        raise C0RunnerError("native_reference_path must be an existing file")
    return (
        lane_id,
        warm_sample_count,
        C0ExecutionInputs(
            interpreter_path=interpreter_path,
            snapshot_root=snapshot_root,
            input_root=input_root,
            input_bundle_sha256=input_bundle_sha256,
            input_fingerprint=input_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
            candidate_path=candidate_path,
            parameter_sha256=parameter_sha256,
            gpu_uuid=gpu_uuid,
            cache_directory=cache_root,
            native_reference_path=native_reference_path,
            runtime_contract_json=runtime_contract_json,
            runtime_identity_sha256=runtime_identity_sha256,
        ),
    )


def _child_launch(
    inputs: C0ExecutionInputs,
    *,
    initial_parameter_sha256: str,
    sample_root: Path,
    identity_anchor: Path,
    environment: Mapping[str, str],
) -> SnapshotModuleLaunch:
    """Construct one shell-free canonical evaluator invocation."""

    module_args = (
        "--input-root",
        str(inputs.input_root),
        "--input-bundle-sha256",
        inputs.input_bundle_sha256,
        "--snapshot-root",
        str(inputs.snapshot_root),
        "--candidate",
        str(inputs.candidate_path),
        "--parameter-sha256",
        inputs.parameter_sha256,
        "--initial-parameter-sha256",
        initial_parameter_sha256,
        "--trace-root",
        str(sample_root / "trace"),
        "--identity-anchor",
        str(identity_anchor),
        "--gpu-uuid",
        inputs.gpu_uuid,
    )
    return build_snapshot_module_launch(
        Path(inputs.interpreter_path),
        inputs.snapshot_root,
        EVALUATOR_MODULE,
        module_args,
        environment,
    )


def _first_gate(
    observation: Mapping[str, object],
    *,
    elapsed_ns: int,
    reference: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if observation.get("mode") != "first":
        raise C0RunnerError("first child observation has the wrong mode")
    gate = {
        "variant": "C0",
        "wall_time_limit_ns": FIRST_EVALUATION_LIMIT_NS,
        "elapsed_ns": elapsed_ns,
        "completed": True,
        "objective_dtype": observation.get("objective_dtype"),
        "objective": observation.get("objective"),
        "gradient_dtype": observation.get("gradient_dtype"),
        "gradient": observation.get("gradient"),
        "native_objective": reference.get("native_objective"),
        "native_gradient": reference.get("native_gradient"),
        "objective_atol": reference.get("objective_atol"),
        "objective_rtol": reference.get("objective_rtol"),
        "gradient_atol": reference.get("gradient_atol"),
        "gradient_rtol": reference.get("gradient_rtol"),
        "inner_newton_success": observation.get("inner_newton_success"),
        "adjoint_success": observation.get("adjoint_success"),
        "residual_certificates": observation.get("residual_certificates"),
    }
    cold_compile = dict(
        _mapping(observation.get("cold_compile"), "first observation cold_compile")
    )
    return gate, cold_compile


def _warm_sample(
    observation: Mapping[str, object], sample_index: int
) -> dict[str, object]:
    if observation.get("mode") != "warm":
        raise C0RunnerError(f"warm sample {sample_index} has the wrong mode")
    observed_index = _integer(
        observation.get("sample_index"), f"warm sample {sample_index}.sample_index"
    )
    if observed_index != sample_index:
        raise C0RunnerError(
            f"warm child index {observed_index} does not match requested {sample_index}"
        )
    if (
        observation.get("sampled_process_gpu_memory_source")
        != SAMPLED_PROCESS_GPU_MEMORY_SOURCE
    ):
        raise C0RunnerError(
            f"warm child {sample_index} has an unsupported GPU-memory source"
        )
    return {
        "sample_index": sample_index,
        "wall_ns": observation.get("wall_ns"),
        "peak_process_tree_rss_bytes": observation.get("peak_process_tree_rss_bytes"),
        "sampled_process_gpu_memory_peak_bytes": observation.get(
            "sampled_process_gpu_memory_peak_bytes"
        ),
        "sampled_process_gpu_memory_source": observation.get(
            "sampled_process_gpu_memory_source"
        ),
        "profiled": False,
    }


def _child_environment(
    base: Mapping[str, str],
    *,
    lane_id: LaneId,
    cache_directory: Path,
    mode: str,
    sample_index: int | None,
    child_output: Path,
    runtime_contract_json: str,
    runtime_identity_sha256: str,
) -> dict[str, str]:
    environment = normalize_static_timing_environment(base)
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(cache_directory),
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
            "SINGLE_STAGE_COMPUTE_GRAPH_LANE": lane_id,
            "SINGLE_STAGE_COMPUTE_GRAPH_VARIANT": "C0",
            "SINGLE_STAGE_COMPUTE_GRAPH_MODE": mode,
            "SINGLE_STAGE_COMPUTE_GRAPH_CHILD_OUTPUT": str(child_output),
            "SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT": runtime_contract_json,
            "SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY": runtime_identity_sha256,
        }
    )
    if sample_index is not None:
        environment["SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX"] = str(sample_index)
    else:
        environment.pop("SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX", None)
    contract = json.loads(runtime_contract_json)
    if not isinstance(contract, dict):
        raise C0RunnerError("runtime contract must be an object")
    contract["route_environment"] = normalize_route_environment(environment)
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"] = json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    )
    return environment


def _write_raw_child(path: Path, result: CommandResult) -> None:
    document = {
        "returncode": result.returncode,
        "elapsed_ns": result.elapsed_ns,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "process_tree_rss": (
            None
            if result.process_tree_rss is None
            else {
                "peak_bytes": result.process_tree_rss.peak_bytes,
                "sample_count": result.process_tree_rss.sample_count,
                "sample_interval_ns": result.process_tree_rss.sample_interval_ns,
                "source": result.process_tree_rss.source,
                "root_pid": result.process_tree_rss.root_pid,
                "root_starttime_ticks": (result.process_tree_rss.root_starttime_ticks),
            }
        ),
    }
    path.write_bytes(canonical_json_bytes(document))


def _write_failure(
    output_root: Path, *, code: str, reason: str, lane_id: LaneId | None
) -> None:
    document = {
        "schema_id": C0_FAILURE_SCHEMA_ID,
        "state": "BLOCKED",
        "lane_id": lane_id,
        "code": code,
        "reason": reason,
    }
    (output_root / FAILURE_FILENAME).write_bytes(canonical_json_bytes(document))


def _validate_command_buffer_evidence(
    path: Path, *, expected_identity: Mapping[str, object]
) -> Mapping[str, object]:
    document = _load_canonical_json_object(path, "command-buffer evidence")
    from benchmarks.single_stage_compute_graph_command_buffer_control import (
        CommandBufferControlError,
        validate_command_buffer_control_evidence,
    )

    try:
        return validate_command_buffer_control_evidence(document, expected_identity)
    except CommandBufferControlError as error:
        raise C0RunnerError(f"invalid command-buffer evidence: {error}") from error


def _bound_post_gate_document(
    path: Path,
    *,
    expected_identity: Mapping[str, object],
    payload_key: str,
) -> Mapping[str, object]:
    """Compatibility helper for producer tests; the runner uses typed validators."""

    document = _load_json_object(path, f"{payload_key} evidence")
    identity = _mapping(document.get("identity"), f"{payload_key} identity")
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise C0RunnerError(f"{payload_key} artifact binding mismatch for {key}")
    return _mapping(document.get(payload_key), f"{payload_key} payload")


def _compute_gap_budget(
    inputs: Mapping[str, object],
    *,
    warm_p50_ns: float,
    profile: Mapping[str, object],
    attribution_control: Mapping[str, object] | None = None,
) -> dict[str, object]:
    phase_wall_shares: dict[str, float] = {}
    if attribution_control is None:
        envelope_ns = _integer(
            profile.get("evaluation_envelope_ns"),
            "profile.evaluation_envelope_ns",
            minimum=1,
        )
        for raw_row in _sequence(
            profile.get("phase_interval_unions"), "profile.phase_interval_unions"
        ):
            row = _mapping(raw_row, "profile phase interval union")
            phase_id = _string(row.get("phase_id"), "profile phase_id")
            intervals = [
                (
                    _integer(interval[0], f"{phase_id} interval start"),
                    _integer(interval[1], f"{phase_id} interval end", minimum=1),
                )
                for interval in (
                    _sequence(value, f"{phase_id} interval")
                    for value in _sequence(
                        row.get("intervals"), f"{phase_id} intervals"
                    )
                )
            ]
            phase_wall_shares[phase_id] = _interval_union_ns(intervals) / envelope_ns
    else:
        selection = _mapping(
            attribution_control.get("selected_attribution"),
            "selected attribution",
        )
        for raw_row in _sequence(
            selection.get("phase_shares"), "selected attribution phase shares"
        ):
            row = _mapping(raw_row, "selected attribution phase share")
            phase_id = _string(row.get("phase_id"), "attribution phase_id")
            raw_share = row.get("selected_default_envelope_share")
            if (
                isinstance(raw_share, bool)
                or not isinstance(raw_share, (int, float))
                or not math.isfinite(raw_share)
                or not 0.0 <= raw_share <= 1.0
            ):
                raise C0RunnerError(
                    f"attribution {phase_id} transferred share must be a fraction"
                )
            measured_share = float(raw_share)
            if phase_id in phase_wall_shares:
                raise C0RunnerError("attribution transfer contains duplicate phase IDs")
            phase_wall_shares[phase_id] = measured_share
    assumptions = _mapping(
        inputs.get("phase_reduction_assumptions"), "phase_reduction_assumptions"
    )
    if frozenset(assumptions) != frozenset(phase_wall_shares):
        raise C0RunnerError(
            "phase reduction assumptions must cover exactly the profiled phases"
        )
    phases = []
    for phase_id, measured_share in phase_wall_shares.items():
        assumption = dict(
            _mapping(assumptions[phase_id], f"phase assumption {phase_id}")
        )
        phases.append(
            {
                "phase_id": phase_id,
                "measured_share": measured_share,
                **assumption,
            }
        )
    conservative_saving = sum(
        float(phase["measured_share"]) * float(phase["conservative_reduction"])
        for phase in phases
    )
    optimistic_saving = sum(
        float(phase["measured_share"]) * float(phase["optimistic_reduction"])
        for phase in phases
    )
    unattributed_share = 1.0 - sum(phase_wall_shares.values())
    unattributed_conservative = float(inputs["unattributed_conservative_reduction"])
    unattributed_optimistic = float(inputs["unattributed_optimistic_reduction"])
    conservative_saving += unattributed_share * unattributed_conservative
    optimistic_saving += unattributed_share * unattributed_optimistic
    complete = dict(
        _mapping(
            inputs.get("matched_complete_path_reference_timings_ns"),
            "matched complete-path references",
        )
    )
    complete_c0 = float(complete["c0"])
    evaluation_count = _integer(
        inputs.get("c0_complete_path_value_and_gradient_evaluation_count"),
        "c0 complete-path value-and-gradient evaluation count",
        minimum=1,
    )
    count_semantics = _string(
        inputs.get("c0_complete_path_value_and_gradient_evaluation_count_semantics"),
        "value-and-gradient evaluation count semantics",
    )
    if count_semantics != NFEV_SEMANTICS:
        raise C0RunnerError("unsupported value-and-gradient evaluation count semantics")
    if warm_p50_ns * evaluation_count > complete_c0:
        raise C0RunnerError(
            "C0 V&G evaluation envelope exceeds matched complete-path timing"
        )
    formal_target = FORMAL_COMPLETE_PATH_FACTOR * min(
        float(complete["native"]), float(complete["optax"])
    )
    candidate_conservative = warm_p50_ns * (1.0 - conservative_saving)
    candidate_optimistic = warm_p50_ns * (1.0 - optimistic_saving)
    conservative_complete = complete_c0 - evaluation_count * (
        warm_p50_ns - candidate_conservative
    )
    optimistic_complete = complete_c0 - evaluation_count * (
        warm_p50_ns - candidate_optimistic
    )
    levers = [
        dict(_mapping(value, "gap-budget faithful lever"))
        for value in _sequence(inputs.get("faithful_levers"), "faithful_levers")
    ]
    all_bounded = all(lever.get("disposition") != "unbounded" for lever in levers)
    reachable = optimistic_complete <= formal_target
    return {
        "candidate_value_and_gradient_reference_timings_ns": {
            "c0_warm_p50": warm_p50_ns
        },
        "matched_complete_path_reference_timings_ns": complete,
        "c0_complete_path_value_and_gradient_evaluation_count": evaluation_count,
        "c0_complete_path_value_and_gradient_evaluation_count_semantics": count_semantics,
        "formal_target_factor": FORMAL_COMPLETE_PATH_FACTOR,
        "formal_target_ns": formal_target,
        "projection_method": (
            "candidate_value_and_gradient_savings_subtracted_from_matched_c0_complete_path"
        ),
        "candidate_phases": phases,
        "unattributed_share": unattributed_share,
        "unattributed_conservative_reduction": unattributed_conservative,
        "unattributed_optimistic_reduction": unattributed_optimistic,
        "candidate_value_and_gradient_conservative_projected_ns": candidate_conservative,
        "candidate_value_and_gradient_optimistic_projected_ns": candidate_optimistic,
        "conservative_complete_path_projected_ns": conservative_complete,
        "optimistic_complete_path_projected_ns": optimistic_complete,
        "faithful_levers": levers,
        "all_faithful_levers_bounded": all_bounded,
        "target_reachable_optimistically": reachable,
        "reduction_evidence_kind": "theoretical_policy_assumptions",
        "claim_ceiling": "DIAGNOSTIC_ONLY_NO_PIVOT_AUTHORITY",
        "empirical_canary_bindings": [],
        "pivot_fired": False,
    }


def _resume_c0_measurement(spec: Mapping[str, object]) -> Mapping[str, object]:
    """Ingest independently produced, checkpoint-bound post-gate evidence."""

    output_root = Path(_string(spec.get("output_root"), "output_root")).resolve()
    resume = _mapping(spec.get("resume"), "resume")
    gate_path = Path(_string(resume.get("gate_checkpoint_path"), "gate path")).resolve()
    warm_path = Path(_string(resume.get("warm_checkpoint_path"), "warm path")).resolve()
    if gate_path != output_root / GATE_CHECKPOINT_FILENAME:
        raise C0RunnerError("gate checkpoint path is not owned by output_root")
    if warm_path != output_root / WARM_CHECKPOINT_FILENAME:
        raise C0RunnerError("warm checkpoint path is not owned by output_root")
    state_path = output_root / STATE_FILENAME
    state = _load_canonical_json_object(state_path, "pending state")
    if state.get("schema_id") != PENDING_SCHEMA_ID or state.get("state") != (
        "POST_GATE_PENDING"
    ):
        raise C0RunnerError("canonical pending state is not resumable")
    gate = _load_canonical_json_object(gate_path, "gate checkpoint")
    warm = _load_canonical_json_object(warm_path, "warm checkpoint")
    if (
        gate.get("schema_id") != GATE_CHECKPOINT_SCHEMA_ID
        or gate.get("state") != "PASSED"
    ):
        raise C0RunnerError("gate checkpoint is not a passing canonical checkpoint")
    if (
        warm.get("schema_id") != WARM_CHECKPOINT_SCHEMA_ID
        or warm.get("state") != "COMPLETE"
    ):
        raise C0RunnerError("warm checkpoint is not complete")
    gate_sha256 = _sha256_path(gate_path)
    warm_sha256 = _sha256_path(warm_path)
    if (
        state.get("gate_checkpoint_sha256") != gate_sha256
        or state.get("warm_checkpoint_sha256") != warm_sha256
    ):
        raise C0RunnerError("pending state checkpoint hashes do not match resume paths")
    if warm.get("gate_checkpoint_sha256") != gate_sha256:
        raise C0RunnerError("warm checkpoint is bound to a different gate")
    requested_lane = _lane_id(spec.get("lane_id"))
    if requested_lane != gate.get("lane_id") or state.get("lane_id") != requested_lane:
        raise C0RunnerError("requested lane differs from staged checkpoint lane")
    duplicate_identity_fields = (
        "lane_id",
        "gpu_uuid",
        "specimen_sha256",
        "input_bundle_sha256",
        "parameter_sha256",
        "source_state_sha256",
        "interpreter_path",
        "runtime_identity_sha256",
    )
    for field in duplicate_identity_fields:
        if warm.get(field) != gate.get(field):
            raise C0RunnerError(f"warm checkpoint {field} differs from gate checkpoint")
    receipt_template = _mapping(spec.get("receipt_template"), "receipt_template")
    specimen = _mapping(receipt_template.get("specimen"), "receipt_template.specimen")
    specimen_sha256 = _sha256(
        receipt_template.get("specimen_sha256"), "receipt_template.specimen_sha256"
    )
    if specimen_sha256 != canonical_sha256(specimen):
        raise C0RunnerError("receipt template specimen hash is inconsistent")
    if specimen_sha256 != gate.get("specimen_sha256"):
        raise C0RunnerError("receipt template specimen differs from gate checkpoint")
    if specimen.get("parameter_sha256") != gate.get("parameter_sha256"):
        raise C0RunnerError("receipt template parameter differs from gate checkpoint")
    lane_id = _lane_id(gate.get("lane_id"))
    _target_lane(receipt_template, lane_id)
    provenance = _mapping(spec.get("provenance"), "provenance")
    if provenance.get("source_state_sha256") != gate.get("source_state_sha256"):
        raise C0RunnerError("provenance source state differs from gate checkpoint")
    allocation = _mapping(provenance.get("allocation"), "provenance.allocation")
    if allocation.get("gpu_uuid") != gate.get("gpu_uuid"):
        raise C0RunnerError("provenance GPU differs from gate checkpoint")
    if provenance.get("interpreter_path") != gate.get("interpreter_path"):
        raise C0RunnerError("provenance interpreter differs from gate checkpoint")
    if _runtime_identity(provenance) != gate.get("runtime_identity_sha256"):
        raise C0RunnerError("provenance runtime identity differs from gate checkpoint")
    warm_measurement = _mapping(warm.get("warm_measurement"), "warm measurement")
    warm_p50_ns = float(warm_measurement["p50_ns"])
    if state.get("warm_p50_ns") != warm_p50_ns:
        raise C0RunnerError("pending state warm p50 differs from warm checkpoint")
    expected_identity = {
        "candidate_sha256": gate.get("parameter_sha256"),
        "specimen_sha256": gate.get("specimen_sha256"),
        "input_bundle_sha256": gate.get("input_bundle_sha256"),
        "source_sha256": gate.get("source_state_sha256"),
        "runtime_identity_sha256": gate.get("runtime_identity_sha256"),
        "lane_id": gate.get("lane_id"),
        "gpu_uuid": gate.get("gpu_uuid"),
        "gate_checkpoint_sha256": gate_sha256,
        "warm_checkpoint_sha256": warm_sha256,
        "warm_p50_ns": warm_p50_ns,
    }
    from benchmarks.single_stage_compute_graph_profile import (
        ProfileEvidenceIdentity,
        build_attribution_control_profile_evidence,
        parse_profile_evidence,
    )

    profile_path = Path(_string(resume.get("profile_evidence_path"), "profile path"))
    profile_evidence = parse_profile_evidence(
        profile_path,
        expected_identity=ProfileEvidenceIdentity(
            candidate_sha256=str(expected_identity["candidate_sha256"]),
            specimen_sha256=str(expected_identity["specimen_sha256"]),
            input_bundle_sha256=str(expected_identity["input_bundle_sha256"]),
            source_sha256=str(expected_identity["source_sha256"]),
            runtime_identity_sha256=str(expected_identity["runtime_identity_sha256"]),
            lane_id=_lane_id(expected_identity["lane_id"]),
            gpu_uuid=str(expected_identity["gpu_uuid"]),
            gate_checkpoint_sha256=gate_sha256,
            warm_checkpoint_sha256=warm_sha256,
            warm_p50_ns=warm_p50_ns,
        ),
    )
    profile = profile_evidence.profile.profile_phase0_json()
    from benchmarks.single_stage_compute_graph_attribution_control import (
        PROFILE_DERIVATION_VERSION,
        AttributionAttempt,
        AttributionBinding,
        AttributionControlError,
        build_attribution_evidence,
        canonical_attribution_attempt_row,
        canonical_module_topology_identity,
        require_promoting_attribution_evidence,
    )
    from benchmarks.single_stage_compute_graph_c0_capture import (
        IDENTITY_ANCHOR_SCHEMA_ID,
    )
    from benchmarks.single_stage_compute_graph_c0_evaluator import (
        CHILD_SCHEMA_ID,
        EXPECTED_PARAMETER_COUNT,
    )

    if CHILD_SCHEMA_ID != C0_CHILD_OBSERVATION_SCHEMA_ID:
        raise C0RunnerError("runner and evaluator child schema identifiers differ")
    if (
        _integer(specimen.get("coil_dof_count"), "specimen coil_dof_count", minimum=1)
        != EXPECTED_PARAMETER_COUNT
    ):
        raise C0RunnerError("specimen and evaluator parameter counts differ")

    attribution_path = Path(
        _string(
            resume.get("attribution_control_evidence_path"),
            "attribution-control path",
        )
    ).resolve()
    expected_attribution_path = (
        output_root
        / "post-gate"
        / "attribution-control"
        / "attribution-control-evidence.json"
    )
    if attribution_path != expected_attribution_path:
        raise C0RunnerError(
            "attribution-control evidence path is not owned by output_root"
        )
    attribution_control = _load_canonical_json_object(
        attribution_path, "attribution-control evidence"
    )
    try:
        require_promoting_attribution_evidence(attribution_control)
    except AttributionControlError as error:
        raise C0RunnerError(
            "attribution-control evidence is explicitly non-promoting"
        ) from error
    attribution_binding = _mapping(
        attribution_control.get("production_binding"),
        "attribution-control production binding",
    )
    for field in (
        "candidate_sha256",
        "specimen_sha256",
        "input_bundle_sha256",
        "source_sha256",
        "lane_id",
        "gpu_uuid",
        "gate_checkpoint_sha256",
        "warm_checkpoint_sha256",
        "warm_p50_ns",
    ):
        expected = expected_identity[field]
        if attribution_binding.get(field) != expected:
            raise C0RunnerError(
                f"attribution-control artifact {field} differs from checkpoints"
            )
    if (
        attribution_binding.get("production_runtime_identity_sha256")
        != expected_identity["runtime_identity_sha256"]
    ):
        raise C0RunnerError(
            "attribution-control production runtime differs from checkpoints"
        )
    selected_phase_shares = _validate_attribution_control(
        attribution_control,
        lane_id=lane_id,
        specimen=specimen,
        specimen_sha256=specimen_sha256,
        source_state_sha256=str(expected_identity["source_sha256"]),
        device_uuid=str(expected_identity["gpu_uuid"]),
        warm_p50_ns=warm_p50_ns,
        context="attribution-control evidence",
    )
    del selected_phase_shares

    direct_attempts = _sequence(
        _mapping(
            attribution_control.get("direct_default_measurement"),
            "attribution direct default",
        ).get("attempts"),
        "attribution direct attempts",
    )
    disabled_attempts = _sequence(
        _mapping(
            attribution_control.get("attribution_replay"),
            "attribution replay",
        ).get("attempts"),
        "attribution replay attempts",
    )
    default_tokens = tuple(
        str(token)
        for token in _sequence(
            _mapping(direct_attempts[0], "default attempt").get("xla_flag_tokens"),
            "default XLA tokens",
        )
    )
    production_environment = _mapping(
        provenance.get("environment"), "provenance.environment"
    )
    production_tokens = tuple(
        shlex.split(str(production_environment.get("XLA_FLAGS", "")))
    )
    if default_tokens != production_tokens:
        raise C0RunnerError(
            "default attribution XLA tokens differ from production provenance"
        )
    disabled_tokens = (*default_tokens, "--xla_gpu_enable_command_buffer=")
    disabled_environment = dict(production_environment)
    disabled_environment["XLA_FLAGS"] = shlex.join(disabled_tokens)
    disabled_runtime_identity = _runtime_identity(
        {
            "interpreter_path": provenance.get("interpreter_path"),
            "runtime": provenance.get("runtime"),
            "environment": disabled_environment,
            "policies": provenance.get("policies"),
        }
    )
    if any(
        _mapping(attempt, "disabled attribution attempt").get("runtime_identity_sha256")
        != disabled_runtime_identity
        for attempt in disabled_attempts
    ):
        raise C0RunnerError(
            "disabled attribution runtime is not derived from production provenance"
        )

    reconstructed_defaults: list[AttributionAttempt] = []
    reconstructed_disabled: list[AttributionAttempt] = []
    reconstructed_binding = AttributionBinding(
        candidate_sha256=str(expected_identity["candidate_sha256"]),
        specimen_sha256=str(expected_identity["specimen_sha256"]),
        input_bundle_sha256=str(expected_identity["input_bundle_sha256"]),
        source_sha256=str(expected_identity["source_sha256"]),
        production_runtime_identity_sha256=str(
            expected_identity["runtime_identity_sha256"]
        ),
        lane_id=lane_id,
        gpu_uuid=str(expected_identity["gpu_uuid"]),
        gate_checkpoint_sha256=gate_sha256,
        warm_checkpoint_sha256=warm_sha256,
        warm_p50_ns=warm_p50_ns,
    )
    gate_observation = _mapping(gate.get("gate_observation"), "gate observation")
    expected_result_certificate = {
        "objective": gate_observation.get("objective"),
        "gradient": gate_observation.get("gradient"),
        "inner_newton_success": gate_observation.get("inner_newton_success"),
        "adjoint_success": gate_observation.get("adjoint_success"),
        "residual_certificates": gate_observation.get("residual_certificates"),
    }

    for raw_attempt in (*direct_attempts, *disabled_attempts):
        attempt = _mapping(raw_attempt, "attribution attempt")
        mode = _string(attempt.get("mode"), "attribution attempt mode")
        attempt_index = _integer(
            attempt.get("attempt_index"), "attribution attempt index"
        )
        artifact_root = Path(
            _string(attempt.get("artifact_root"), "attribution artifact root")
        ).resolve()
        expected_attempt_root = (
            expected_attribution_path.parent / mode / f"attempt-{attempt_index:02d}"
        )
        if artifact_root != expected_attempt_root:
            raise C0RunnerError(
                "attribution attempt artifact root is not the canonical owned root"
            )
        cache_root = Path(
            _string(attempt.get("compilation_cache_root"), "attribution cache root")
        ).resolve()
        expected_cache_root = (
            output_root
            / "post-gate"
            / "attribution-control-cache"
            / mode
            / f"attempt-{attempt_index:02d}"
        )
        if cache_root != expected_cache_root:
            raise C0RunnerError(
                "attribution attempt cache root is not the canonical owned root"
            )
        bound_paths: dict[str, Path] = {}
        canonical_documents: dict[str, Mapping[str, object]] = {}
        for path_field, sha_field, canonical in (
            ("raw_trace_path", "raw_trace_sha256", False),
            ("child_observation_path", "child_observation_sha256", True),
            ("hlo_anchor_path", "hlo_anchor_sha256", True),
        ):
            relative = Path(
                _string(attempt.get(path_field), f"attribution {path_field}")
            )
            bound_path = (output_root / relative).resolve()
            if not bound_path.is_relative_to(artifact_root) or not bound_path.is_file():
                raise C0RunnerError(
                    f"attribution {path_field} is outside its artifact root"
                )
            if _sha256_path(bound_path) != attempt.get(sha_field):
                raise C0RunnerError(f"attribution {path_field} hash mismatch")
            bound_paths[path_field] = bound_path
            if canonical:
                canonical_documents[path_field] = _load_canonical_json_object(
                    bound_path, f"attribution {path_field}"
                )

        runtime_identity = (
            str(expected_identity["runtime_identity_sha256"])
            if mode == "default_control"
            else disabled_runtime_identity
        )
        tokens = default_tokens if mode == "default_control" else disabled_tokens
        recomputed_profile = build_attribution_control_profile_evidence(
            trace_path=bound_paths["raw_trace_path"],
            artifact_root=artifact_root,
            candidate_sha256=str(expected_identity["candidate_sha256"]),
            specimen_sha256=str(expected_identity["specimen_sha256"]),
            input_bundle_sha256=str(expected_identity["input_bundle_sha256"]),
            source_sha256=str(expected_identity["source_sha256"]),
            runtime_identity_sha256=runtime_identity,
            lane_id=lane_id,
            gpu_uuid=str(expected_identity["gpu_uuid"]),
            gate_checkpoint_sha256=gate_sha256,
            warm_checkpoint_sha256=warm_sha256,
            warm_p50_ns=warm_p50_ns,
        )
        expected_anchor = {
            "schema_id": IDENTITY_ANCHOR_SCHEMA_ID,
            "hlo_module_set_identity": (
                recomputed_profile.profile.hlo_module_set_identity
            ),
            "hlo_module_set_identity_source": (
                recomputed_profile.profile.hlo_module_set_identity_source
            ),
        }
        if canonical_documents["hlo_anchor_path"] != expected_anchor:
            raise C0RunnerError(
                "attribution HLO anchor differs from recomputed raw trace topology"
            )
        child = canonical_documents["child_observation_path"]
        expected_child_fields = {
            "schema_id",
            "mode",
            "sample_index",
            "parameter_sha256",
            "objective_dtype",
            "objective",
            "gradient_dtype",
            "gradient",
            "inner_newton_success",
            "adjoint_success",
            "residual_certificates",
            "cold_compile",
            "pjrt_execute_count",
            "kernel_launch_count",
        }
        if set(child) != expected_child_fields:
            raise C0RunnerError(
                "attribution child does not match the profile observation schema"
            )
        if child.get("schema_id") != CHILD_SCHEMA_ID:
            raise C0RunnerError("attribution child has an unsupported schema_id")
        if child.get("mode") != "profile":
            raise C0RunnerError("attribution child is not profile mode")
        if child.get("sample_index") is not None:
            raise C0RunnerError(
                "attribution profile child must not have a sample index"
            )
        if child.get("parameter_sha256") != expected_identity["candidate_sha256"]:
            raise C0RunnerError("attribution child parameter differs from checkpoints")
        if (
            child.get("objective_dtype") != "float64"
            or child.get("gradient_dtype") != "float64"
        ):
            raise C0RunnerError("attribution child has non-FP64 numerical outputs")
        cold_compile = _mapping(
            child.get("cold_compile"), "attribution child cold_compile"
        )
        expected_cold_compile_fields = {
            "wall_ns",
            "peak_process_tree_rss_bytes",
            "process_tree_rss_sample_count",
            "process_tree_rss_sample_interval_ns",
            "process_tree_rss_source",
            "process_tree_rss_root_pid",
            "process_tree_rss_root_starttime_ticks",
            "sampled_process_gpu_memory_peak_bytes",
            "sampled_process_gpu_memory_source",
            "hlo_module_set_identity",
            "hlo_module_set_identity_source",
        }
        if set(cold_compile) != expected_cold_compile_fields:
            raise C0RunnerError(
                "attribution child cold_compile does not match the profile schema"
            )
        if (
            cold_compile.get("hlo_module_set_identity")
            != recomputed_profile.profile.hlo_module_set_identity
            or cold_compile.get("hlo_module_set_identity_source")
            != recomputed_profile.profile.hlo_module_set_identity_source
        ):
            raise C0RunnerError(
                "attribution child HLO identity differs from the raw trace"
            )
        if (
            child.get("pjrt_execute_count")
            != recomputed_profile.profile.pjrt_execute_count
            or child.get("kernel_launch_count")
            != recomputed_profile.profile.kernel_launch_count
        ):
            raise C0RunnerError(
                "attribution child launch counts differ from the raw trace"
            )
        _integer(cold_compile.get("wall_ns"), "cold compile wall", minimum=1)
        _integer(
            cold_compile.get("peak_process_tree_rss_bytes"),
            "cold compile RSS",
            minimum=1,
        )
        _integer(
            cold_compile.get("process_tree_rss_sample_count"),
            "cold compile RSS sample count",
            minimum=1,
        )
        if (
            cold_compile.get("process_tree_rss_sample_interval_ns")
            != PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS
            or cold_compile.get("process_tree_rss_source") != PROCESS_TREE_RSS_SOURCE
        ):
            raise C0RunnerError("cold compile RSS sampler binding is invalid")
        _integer(
            cold_compile.get("process_tree_rss_root_pid"),
            "cold compile RSS root PID",
            minimum=1,
        )
        _integer(
            cold_compile.get("process_tree_rss_root_starttime_ticks"),
            "cold compile RSS root starttime",
            minimum=1,
        )
        _integer(
            cold_compile.get("sampled_process_gpu_memory_peak_bytes"),
            "cold compile GPU memory",
        )
        if (
            cold_compile.get("sampled_process_gpu_memory_source")
            != SAMPLED_PROCESS_GPU_MEMORY_SOURCE
        ):
            raise C0RunnerError("attribution child cold_compile payload is invalid")
        child_result_certificate = {
            "objective": child.get("objective"),
            "gradient": child.get("gradient"),
            "inner_newton_success": child.get("inner_newton_success"),
            "adjoint_success": child.get("adjoint_success"),
            "residual_certificates": child.get("residual_certificates"),
        }
        objective = child.get("objective")
        gradient = child.get("gradient")
        residuals = child.get("residual_certificates")
        if (
            not isinstance(objective, float)
            or not math.isfinite(objective)
            or not isinstance(gradient, list)
            or len(gradient) != EXPECTED_PARAMETER_COUNT
            or any(
                not isinstance(value, float) or not math.isfinite(value)
                for value in gradient
            )
            or not isinstance(residuals, dict)
            or not residuals
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, float)
                or not math.isfinite(value)
                or value < 0.0
                for name, value in residuals.items()
            )
            or not isinstance(child.get("inner_newton_success"), bool)
            or not isinstance(child.get("adjoint_success"), bool)
        ):
            raise C0RunnerError("attribution child numerical payload is invalid")
        if child_result_certificate != expected_result_certificate:
            raise C0RunnerError(
                "attribution child result or solve certificate differs from gate"
            )
        phase_device_ns = tuple(
            (
                phase_id,
                sum(interval.end_ns - interval.start_ns for interval in intervals),
            )
            for phase_id, intervals in recomputed_profile.profile.phase_interval_unions
        )
        reconstructed = AttributionAttempt(
            mode=(
                "default_control"
                if mode == "default_control"
                else "command_buffer_disabled"
            ),
            attempt_index=attempt_index,
            binding=reconstructed_binding,
            runtime_identity_sha256=runtime_identity,
            xla_flag_tokens=tokens,
            compilation_cache_root=str(cache_root),
            artifact_root=str(artifact_root),
            raw_trace_path=bound_paths["raw_trace_path"]
            .relative_to(output_root)
            .as_posix(),
            raw_trace_sha256=_sha256_path(bound_paths["raw_trace_path"]),
            child_observation_path=bound_paths["child_observation_path"]
            .relative_to(output_root)
            .as_posix(),
            child_observation_sha256=_sha256_path(
                bound_paths["child_observation_path"]
            ),
            hlo_anchor_path=bound_paths["hlo_anchor_path"]
            .relative_to(output_root)
            .as_posix(),
            hlo_anchor_sha256=_sha256_path(bound_paths["hlo_anchor_path"]),
            profile_derivation_version=PROFILE_DERIVATION_VERSION,
            objective=float(objective),
            gradient=tuple(float(value) for value in gradient),
            solve_certificate={
                "inner_newton_success": child["inner_newton_success"],
                "adjoint_success": child["adjoint_success"],
                "residual_certificates": residuals,
            },
            module_topology_identity_sha256=canonical_module_topology_identity(
                recomputed_profile.profile.hlo_module_set_identity,
                recomputed_profile.profile.hlo_module_set_identity_source,
                specimen_sha256,
            ),
            evaluation_envelope_ns=recomputed_profile.profile.evaluation_envelope_ns,
            device_active_ns=recomputed_profile.profile.device_active_ns,
            phase_device_ns=phase_device_ns,
        )
        if canonical_attribution_attempt_row(reconstructed) != dict(attempt):
            raise C0RunnerError(
                "attribution attempt row differs from raw-recomputed evidence"
            )
        target = (
            reconstructed_defaults
            if mode == "default_control"
            else reconstructed_disabled
        )
        target.append(reconstructed)

    rebuilt_attribution = build_attribution_evidence(
        tuple(reconstructed_defaults), tuple(reconstructed_disabled)
    )
    if rebuilt_attribution != attribution_control:
        raise C0RunnerError(
            "attribution-control evidence differs from six raw-recomputed attempts"
        )
    command_buffer = _validate_command_buffer_evidence(
        Path(
            _string(resume.get("command_buffer_evidence_path"), "command-buffer path")
        ),
        expected_identity=expected_identity,
    )
    from benchmarks.single_stage_compute_graph_newton_telemetry import (
        NewtonTelemetryError,
        TelemetryIdentity,
        validate_newton_telemetry_evidence,
    )

    telemetry_evidence_path = Path(
        _string(resume.get("newton_telemetry_evidence_path"), "telemetry path")
    )
    telemetry_document = _load_canonical_json_object(
        telemetry_evidence_path, "newton telemetry evidence"
    )
    try:
        newton_telemetry = validate_newton_telemetry_evidence(
            telemetry_document,
            TelemetryIdentity(
                candidate_sha256=str(expected_identity["candidate_sha256"]),
                specimen_sha256=str(expected_identity["specimen_sha256"]),
                input_bundle_sha256=str(expected_identity["input_bundle_sha256"]),
                source_sha256=str(expected_identity["source_sha256"]),
                runtime_identity_sha256=str(
                    expected_identity["runtime_identity_sha256"]
                ),
                lane_id=lane_id,
                gpu_uuid=str(expected_identity["gpu_uuid"]),
                gate_checkpoint_sha256=gate_sha256,
                warm_checkpoint_sha256=warm_sha256,
                warm_p50_ns=warm_p50_ns,
            ),
        )
    except NewtonTelemetryError as error:
        raise C0RunnerError(f"invalid Newton telemetry evidence: {error}") from error
    telemetry_bytes = telemetry_evidence_path.read_bytes()
    telemetry_relative_path = (
        Path(RAW_TELEMETRY_DIRECTORY_NAME) / lane_id / "newton-telemetry.json"
    )
    bound_telemetry_path = output_root / telemetry_relative_path
    bound_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    if bound_telemetry_path.exists():
        if bound_telemetry_path.read_bytes() != telemetry_bytes:
            raise C0RunnerError("bound raw telemetry artifact already differs")
    else:
        with bound_telemetry_path.open("xb") as stream:
            stream.write(telemetry_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    newton_telemetry = {
        **dict(newton_telemetry),
        "raw_evidence_relative_path": telemetry_relative_path.as_posix(),
        "raw_evidence_file_sha256": hashlib.sha256(telemetry_bytes).hexdigest(),
    }
    from benchmarks.single_stage_compute_graph_complete_path import (
        CompletePathEvidenceError,
        validate_gap_budget_inputs_artifact,
    )

    complete_path_document = _load_canonical_json_object(
        Path(
            _string(
                resume.get("complete_path_evidence_path"), "complete-path evidence path"
            )
        ),
        "complete-path evidence",
    )
    complete_identity = _mapping(
        complete_path_document.get("identity"), "complete-path identity"
    )
    for field, expected in expected_identity.items():
        if complete_identity.get(field) != expected:
            raise C0RunnerError(
                f"complete-path artifact {field} differs from checkpoints"
            )
    gap_document = _load_canonical_json_object(
        Path(_string(resume.get("gap_budget_inputs_path"), "gap inputs path")),
        "gap-budget input evidence",
    )
    try:
        gap_inputs = validate_gap_budget_inputs_artifact(
            gap_document,
            complete_path_document,
        )
    except CompletePathEvidenceError as error:
        raise C0RunnerError(f"invalid gap-budget input evidence: {error}") from error
    gap_budget = _compute_gap_budget(
        gap_inputs,
        warm_p50_ns=warm_p50_ns,
        profile=profile,
        attribution_control=attribution_control,
    )
    gate_observation = _mapping(gate.get("gate_observation"), "gate observation")
    cold_compile = {
        "wall_ns": _mapping(gate.get("first_evaluation_gate"), "gate")["elapsed_ns"],
        "peak_process_tree_rss_bytes": gate_observation["peak_process_tree_rss_bytes"],
        "sampled_process_gpu_memory_peak_bytes": gate_observation[
            "sampled_process_gpu_memory_peak_bytes"
        ],
        "sampled_process_gpu_memory_source": gate_observation[
            "sampled_process_gpu_memory_source"
        ],
        "hlo_module_set_identity": profile["hlo_module_set_identity"],
        "hlo_module_set_identity_source": profile["hlo_module_set_identity_source"],
    }
    measurement = {
        "variant": "C0",
        "specimen_sha256": gate["specimen_sha256"],
        "provenance": dict(provenance),
        "first_evaluation_gate": dict(
            _mapping(gate.get("first_evaluation_gate"), "first gate")
        ),
        "cold_compile": cold_compile,
        "warm_measurement": dict(warm_measurement),
        "profile": profile,
        "attribution_control": dict(attribution_control),
        "command_buffer": dict(command_buffer),
        "newton_telemetry": dict(newton_telemetry),
        "gap_budget": gap_budget,
    }
    receipt = dict(receipt_template)
    lanes = [
        dict(_mapping(item, "receipt lane"))
        for item in _sequence(receipt["lanes"], "receipt lanes")
    ]
    target_lane, target_index = _target_lane(receipt, lane_id)
    lanes[target_index] = {**dict(target_lane), "measurement": measurement}
    receipt["lanes"] = lanes
    try:
        validate_phase0_receipt(receipt)
    except Phase0ReceiptError as error:
        raise C0RunnerError(
            f"assembled Phase 0 receipt failed validation: {error}"
        ) from error
    write_phase0_receipt(output_root / RECEIPT_FILENAME, receipt)
    return receipt


def run_c0_measurement(
    spec: Mapping[str, object],
    *,
    executor: CommandExecutor = _subprocess_executor,
    environment: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Run gate plus unprofiled warm timing, or resume post-gate assembly."""

    if "resume" in spec:
        return _resume_c0_measurement(spec)
    lane_id: LaneId | None = None
    output_root = Path(_string(spec.get("output_root"), "output_root")).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    child_root = output_root / CHILD_DIRECTORY_NAME
    child_root.mkdir()
    try:
        lane_id, warm_sample_count, execution_inputs = _validated_spec(spec)
        execution_inputs.cache_directory.mkdir(parents=True, exist_ok=False)
        provenance = _mapping(spec.get("provenance"), "provenance")
        receipt_template = _mapping(spec.get("receipt_template"), "receipt_template")
        specimen = _mapping(
            receipt_template.get("specimen"), "receipt_template.specimen"
        )
        specimen_sha256 = _sha256(
            receipt_template.get("specimen_sha256"), "receipt_template.specimen_sha256"
        )
        if specimen_sha256 != canonical_sha256(specimen):
            raise C0RunnerError("receipt template specimen hash is inconsistent")
        parameter_sha256 = _sha256(
            specimen.get("parameter_sha256"), "specimen.parameter_sha256"
        )
        source_state_sha256 = _sha256(
            provenance.get("source_state_sha256"), "provenance.source_state_sha256"
        )
        runtime_identity_sha256 = _runtime_identity(provenance)
        import_bindings = _mapping(
            provenance.get("import_bindings"), "provenance.import_bindings"
        )
        native_simsoptpp = _mapping(
            import_bindings.get("simsoptpp"), "provenance.import_bindings.simsoptpp"
        )
        native, native_sha256, reference = _native_reference(
            execution_inputs.native_reference_path,
            parameter_sha256,
            expected_bindings={
                "input_bundle_sha256": execution_inputs.input_bundle_sha256,
                "input_fingerprint": execution_inputs.input_fingerprint,
                "configuration_fingerprint": execution_inputs.configuration_fingerprint,
                "specimen_sha256": specimen_sha256,
                "source_sha256": source_state_sha256,
                "interpreter_path": execution_inputs.interpreter_path,
                "native_simsoptpp_path": native_simsoptpp.get("path"),
                "native_simsoptpp_sha256": native_simsoptpp.get("sha256"),
                "runtime_identity_sha256": runtime_identity_sha256,
            },
        )
        base_environment = os.environ if environment is None else environment
        identity_anchor = output_root / IDENTITY_ANCHOR_FILENAME
        initial_parameter_sha256 = _sha256(
            reference.get("initial_parameter_sha256"),
            "native initial parameter_sha256",
        )

        initial_gate_root = child_root / "initial-gate"
        initial_gate_root.mkdir()
        initial_gate_path = initial_gate_root / RAW_CHILD_FILENAME
        initial_gate_environment = _child_environment(
            base_environment,
            lane_id=lane_id,
            cache_directory=execution_inputs.cache_directory,
            mode="initial-gate",
            sample_index=None,
            child_output=initial_gate_path,
            runtime_contract_json=execution_inputs.runtime_contract_json,
            runtime_identity_sha256=execution_inputs.runtime_identity_sha256,
        )
        initial_gate_launch = _child_launch(
            execution_inputs,
            initial_parameter_sha256=initial_parameter_sha256,
            sample_root=initial_gate_root,
            identity_anchor=identity_anchor,
            environment=initial_gate_environment,
        )
        initial_gate_result = executor(
            initial_gate_launch.argv,
            initial_gate_launch.environment,
            initial_gate_launch.cwd,
            FIRST_EVALUATION_LIMIT_NS / 1_000_000_000,
        )
        _write_raw_child(initial_gate_path, initial_gate_result)
        if (
            initial_gate_result.timed_out
            or initial_gate_result.elapsed_ns > FIRST_EVALUATION_LIMIT_NS
        ):
            raise C0RunnerError(
                "initial evaluation exceeded the 900-second fresh-process gate"
            )
        initial_gate_observation = _child_observation(
            initial_gate_result, "initial-gate child"
        )
        if initial_gate_observation.get("mode") != "initial-gate":
            raise C0RunnerError("initial-gate child observation has the wrong mode")
        if (
            _sha256(
                initial_gate_observation.get("parameter_sha256"),
                "initial-gate parameter_sha256",
            )
            != initial_parameter_sha256
        ):
            raise C0RunnerError(
                "initial-gate observation is bound to different parameters"
            )
        initial_evaluation_gate = {
            "variant": "C0",
            "wall_time_limit_ns": FIRST_EVALUATION_LIMIT_NS,
            "elapsed_ns": initial_gate_result.elapsed_ns,
            "completed": True,
            "objective_dtype": initial_gate_observation.get("objective_dtype"),
            "objective": initial_gate_observation.get("objective"),
            "gradient_dtype": initial_gate_observation.get("gradient_dtype"),
            "gradient": initial_gate_observation.get("gradient"),
            "native_objective": reference.get("native_initial_objective"),
            "native_gradient": reference.get("native_initial_gradient"),
            "objective_atol": reference.get("objective_atol"),
            "objective_rtol": reference.get("objective_rtol"),
            "gradient_atol": reference.get("gradient_atol"),
            "gradient_rtol": reference.get("gradient_rtol"),
            "inner_newton_success": initial_gate_observation.get(
                "inner_newton_success"
            ),
            "adjoint_success": initial_gate_observation.get("adjoint_success"),
            "residual_certificates": initial_gate_observation.get(
                "residual_certificates"
            ),
        }
        try:
            _validate_first_evaluation(
                initial_evaluation_gate, specimen, "initial_evaluation_gate"
            )
        except Phase0ReceiptError as error:
            raise C0RunnerError(f"initial evaluation gate failed: {error}") from error

        gate_root = child_root / "gate"
        gate_root.mkdir()
        gate_path = gate_root / RAW_CHILD_FILENAME
        gate_environment = _child_environment(
            base_environment,
            lane_id=lane_id,
            cache_directory=execution_inputs.cache_directory,
            mode="gate",
            sample_index=None,
            child_output=gate_path,
            runtime_contract_json=execution_inputs.runtime_contract_json,
            runtime_identity_sha256=execution_inputs.runtime_identity_sha256,
        )
        gate_launch = _child_launch(
            execution_inputs,
            initial_parameter_sha256=initial_parameter_sha256,
            sample_root=gate_root,
            identity_anchor=identity_anchor,
            environment=gate_environment,
        )
        gate_result = executor(
            gate_launch.argv,
            gate_launch.environment,
            gate_launch.cwd,
            FIRST_EVALUATION_LIMIT_NS / 1_000_000_000,
        )
        _write_raw_child(gate_path, gate_result)
        if gate_result.timed_out or gate_result.elapsed_ns > FIRST_EVALUATION_LIMIT_NS:
            raise C0RunnerError("first evaluation exceeded the 900-second gate")
        gate_observation = _child_observation(gate_result, "gate child")
        if gate_observation.get("mode") != "gate":
            raise C0RunnerError("gate child observation has the wrong mode")
        if (
            _sha256(
                gate_observation.get("parameter_sha256"),
                "gate parameter_sha256",
            )
            != parameter_sha256
        ):
            raise C0RunnerError("gate observation is bound to different parameters")
        first_gate = {
            "variant": "C0",
            "wall_time_limit_ns": FIRST_EVALUATION_LIMIT_NS,
            "elapsed_ns": gate_result.elapsed_ns,
            "completed": True,
            "objective_dtype": gate_observation.get("objective_dtype"),
            "objective": gate_observation.get("objective"),
            "gradient_dtype": gate_observation.get("gradient_dtype"),
            "gradient": gate_observation.get("gradient"),
            "native_objective": reference.get("native_objective"),
            "native_gradient": reference.get("native_gradient"),
            "objective_atol": reference.get("objective_atol"),
            "objective_rtol": reference.get("objective_rtol"),
            "gradient_atol": reference.get("gradient_atol"),
            "gradient_rtol": reference.get("gradient_rtol"),
            "inner_newton_success": gate_observation.get("inner_newton_success"),
            "adjoint_success": gate_observation.get("adjoint_success"),
            "residual_certificates": gate_observation.get("residual_certificates"),
        }
        try:
            _validate_first_evaluation(first_gate, specimen, "first_evaluation_gate")
        except Phase0ReceiptError as error:
            raise C0RunnerError(f"first evaluation gate failed: {error}") from error
        allocation = _mapping(provenance.get("allocation"), "provenance.allocation")
        gpu_uuid = _string(allocation.get("gpu_uuid"), "provenance.allocation.gpu_uuid")
        gate_checkpoint = {
            "schema_id": GATE_CHECKPOINT_SCHEMA_ID,
            "state": "PASSED",
            "lane_id": lane_id,
            "gpu_uuid": gpu_uuid,
            "specimen_sha256": specimen_sha256,
            "input_bundle_sha256": execution_inputs.input_bundle_sha256,
            "parameter_sha256": parameter_sha256,
            "initial_parameter_sha256": initial_parameter_sha256,
            "source_state_sha256": source_state_sha256,
            "interpreter_path": execution_inputs.interpreter_path,
            "runtime_identity_sha256": runtime_identity_sha256,
            "native_reference_sha256": native_sha256,
            "native_baseline_anchor": dict(
                _mapping(native.get("baseline_anchor"), "native baseline")
            ),
            "parity_tolerances": {
                "objective_atol": PHASE0_OBJECTIVE_ATOL,
                "objective_rtol": PHASE0_OBJECTIVE_RTOL,
                "gradient_atol": PHASE0_GRADIENT_ATOL,
                "gradient_rtol": PHASE0_GRADIENT_RTOL,
            },
            "first_evaluation_gate": first_gate,
            "initial_evaluation_gate": initial_evaluation_gate,
            "initial_gate_observation": dict(initial_gate_observation),
            "gate_observation": dict(gate_observation),
        }
        gate_sha256 = _write_exclusive_json(
            output_root / GATE_CHECKPOINT_FILENAME, gate_checkpoint
        )

        samples: list[dict[str, object]] = []
        warm_observations: list[Mapping[str, object]] = []
        for sample_index in range(warm_sample_count):
            sample_root = child_root / f"warm-{sample_index:02d}"
            sample_root.mkdir()
            sample_path = sample_root / RAW_CHILD_FILENAME
            warm_environment = _child_environment(
                base_environment,
                lane_id=lane_id,
                cache_directory=execution_inputs.cache_directory,
                mode="warm",
                sample_index=sample_index,
                child_output=sample_path,
                runtime_contract_json=execution_inputs.runtime_contract_json,
                runtime_identity_sha256=execution_inputs.runtime_identity_sha256,
            )
            warm_launch = _child_launch(
                execution_inputs,
                initial_parameter_sha256=initial_parameter_sha256,
                sample_root=sample_root,
                identity_anchor=identity_anchor,
                environment=warm_environment,
            )
            result = executor(
                warm_launch.argv,
                warm_launch.environment,
                warm_launch.cwd,
                FIRST_EVALUATION_LIMIT_NS / 1_000_000_000,
            )
            _write_raw_child(sample_path, result)
            observation = _child_observation(result, f"warm child {sample_index}")
            warm_observations.append(observation)
            samples.append(_warm_sample(observation, sample_index))
        wall_times = [
            _integer(sample["wall_ns"], "warm sample wall_ns", minimum=1)
            for sample in samples
        ]
        ordered_times = sorted(wall_times)
        warm_measurement = {
            "samples": samples,
            "p50_ns": float(statistics.median(wall_times)),
            "p95_ns": float(ordered_times[math.ceil(0.95 * len(ordered_times)) - 1]),
        }
        cache_identity = canonical_sha256(
            {
                "role": "promotion_warm_cache",
                "path": str(execution_inputs.cache_directory),
                "source_state_sha256": source_state_sha256,
                "specimen_sha256": specimen_sha256,
                "lane_id": lane_id,
                "gpu_uuid": gpu_uuid,
            }
        )
        warm_checkpoint = {
            "schema_id": WARM_CHECKPOINT_SCHEMA_ID,
            "state": "COMPLETE",
            "gate_checkpoint_sha256": gate_sha256,
            "lane_id": lane_id,
            "gpu_uuid": gpu_uuid,
            "specimen_sha256": specimen_sha256,
            "input_bundle_sha256": execution_inputs.input_bundle_sha256,
            "parameter_sha256": parameter_sha256,
            "source_state_sha256": source_state_sha256,
            "interpreter_path": execution_inputs.interpreter_path,
            "runtime_identity_sha256": runtime_identity_sha256,
            "promotion_cache_identity": cache_identity,
            "process_tree_rss_sampler": {
                "source": PROCESS_TREE_RSS_SOURCE,
                "sample_interval_ns": PROCESS_TREE_RSS_SAMPLE_INTERVAL_NS,
                "samples": [
                    {
                        "sample_index": index,
                        "sample_count": _integer(
                            observation.get("process_tree_rss_sample_count"),
                            f"warm observation {index} RSS sample_count",
                            minimum=1,
                        ),
                        "peak_bytes": _integer(
                            observation.get("peak_process_tree_rss_bytes"),
                            f"warm observation {index} peak RSS",
                            minimum=1,
                        ),
                        "root_pid": _integer(
                            observation.get("process_tree_rss_root_pid"),
                            f"warm observation {index} RSS root PID",
                            minimum=1,
                        ),
                        "root_starttime_ticks": _integer(
                            observation.get("process_tree_rss_root_starttime_ticks"),
                            f"warm observation {index} RSS root starttime",
                            minimum=1,
                        ),
                    }
                    for index, observation in enumerate(warm_observations)
                ],
            },
            "warm_measurement": warm_measurement,
        }
        warm_sha256 = _write_exclusive_json(
            output_root / WARM_CHECKPOINT_FILENAME, warm_checkpoint
        )
        state = {
            "schema_id": PENDING_SCHEMA_ID,
            "state": "POST_GATE_PENDING",
            "gate_checkpoint_sha256": gate_sha256,
            "warm_checkpoint_sha256": warm_sha256,
            "warm_p50_ns": warm_measurement["p50_ns"],
            "lane_id": lane_id,
            "runtime_identity_sha256": runtime_identity_sha256,
        }
        _write_exclusive_json(output_root / STATE_FILENAME, state)
        return state
    except C0RunnerError as error:
        _write_failure(
            output_root,
            code="C0_MEASUREMENT_BLOCKED",
            reason=str(error),
            lane_id=lane_id,
        )
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    spec = _load_json_object(options.spec, "runner spec")
    try:
        run_c0_measurement(spec)
    except C0RunnerError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
