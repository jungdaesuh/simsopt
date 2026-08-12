"""Restartable post-gate orchestration for a validated Phase 0 C0 lane.

This module owns no numerical or evidence schema.  It derives runner-owned
identity from the immutable C0 checkpoints and delegates collection and
validation to the profile, command-buffer, telemetry, complete-path, and C0
runner SSOT modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from benchmarks.single_stage_changed_state_profiler_policy import (
    TRACE_VIEWER_MAX_EVENTS,
    TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT,
)
from benchmarks.single_stage_compute_graph_attribution_control import (
    ATTRIBUTION_ATTEMPT_COUNT,
    COMMAND_BUFFER_DISABLE_FLAG,
    PROFILE_DERIVATION_VERSION,
    AttributionAttempt,
    AttributionBinding,
    build_attribution_evidence,
    canonical_module_topology_identity,
    require_promoting_attribution_evidence,
)
from benchmarks.single_stage_compute_graph_c0_capture import (
    IDENTITY_ANCHOR_SCHEMA_ID,
)
from benchmarks.single_stage_compute_graph_c0_runner import (
    C0_RUNNER_SPEC_SCHEMA_ID,
    GATE_CHECKPOINT_FILENAME,
    PENDING_SCHEMA_ID,
    RECEIPT_FILENAME,
    STATE_FILENAME,
    WARM_CHECKPOINT_FILENAME,
    CommandExecutor,
    _command_rss_document,
    _runtime_identity,
    run_c0_measurement,
)
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    ControlPlan,
    build_control_plan,
    execute_control_plan,
    validate_command_buffer_control_evidence,
)
from benchmarks.single_stage_compute_graph_complete_path import (
    DOCUMENT_PATH,
    CompletePathBinding,
    FaithfulLever,
    GapBudgetPolicyInput,
    PhaseReductionAssumption,
    ProfileId,
    RunExecutor,
    binding_from_phase0_checkpoints,
    build_gap_budget_inputs_artifact,
    build_staged_gap_budget_timing_input,
    collect_complete_path_evidence,
    validate_gap_budget_inputs_artifact,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    build_snapshot_module_launch,
    normalize_route_environment,
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_newton_telemetry import (
    TelemetryIdentity,
    validate_newton_telemetry_evidence,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    A100_LANE_ID,
    RTX_LANE_ID,
    LaneId,
    canonical_json_bytes,
    load_phase0_receipt,
)
from benchmarks.single_stage_compute_graph_profile import (
    ComputeGraphProfileEvidence,
    ProfileEvidenceIdentity,
    build_attribution_control_profile_evidence,
    build_profile_evidence,
    parse_profile_evidence,
    write_profile_evidence,
)

PROFILE_MODULE: Final = "benchmarks.single_stage_compute_graph_c0_evaluator"
TELEMETRY_MODULE: Final = "benchmarks.single_stage_compute_graph_newton_telemetry"
PROFILE_EVIDENCE_FILENAME: Final = "profile-evidence.json"
ATTRIBUTION_EVIDENCE_FILENAME: Final = "attribution-control-evidence.json"
COMMAND_BUFFER_EVIDENCE_FILENAME: Final = "command-buffer-evidence.json"
NEWTON_TELEMETRY_FILENAME: Final = "newton-telemetry.json"
GAP_INPUTS_FILENAME: Final = "gap-budget-inputs.json"
RESUME_SPEC_FILENAME: Final = "resume-spec.json"
POST_GATE_DIRECTORY_NAME: Final = "post-gate"


class Phase0PostGateError(RuntimeError):
    """Post-gate state or evidence is incomplete, stale, or contradictory."""


class ControlExecutor(Protocol):
    def __call__(
        self, plan: ControlPlan, output_path: Path
    ) -> Mapping[str, object]: ...


FinalAssembler = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class PostGatePaths:
    """Canonical artifact paths owned by one C0 output root."""

    c0_output_root: Path
    post_gate_root: Path
    profile_root: Path
    profile_cache_root: Path
    profile_evidence_path: Path
    attribution_root: Path
    attribution_cache_root: Path
    attribution_evidence_path: Path
    command_buffer_root: Path
    command_buffer_cache_root: Path
    command_buffer_evidence_path: Path
    telemetry_cache_root: Path
    newton_telemetry_path: Path
    complete_path_root: Path
    complete_path_evidence_path: Path
    gap_inputs_path: Path
    resume_spec_path: Path

    @classmethod
    def from_output_root(cls, output_root: Path) -> PostGatePaths:
        resolved = output_root.resolve()
        post_gate = resolved / POST_GATE_DIRECTORY_NAME
        profile_root = post_gate / "profile"
        command_root = post_gate / "command-buffer"
        complete_root = post_gate / "complete-path"
        return cls(
            c0_output_root=resolved,
            post_gate_root=post_gate,
            profile_root=profile_root,
            profile_cache_root=post_gate / "profile-cache",
            profile_evidence_path=profile_root / PROFILE_EVIDENCE_FILENAME,
            attribution_root=post_gate / "attribution-control",
            attribution_cache_root=post_gate / "attribution-control-cache",
            attribution_evidence_path=(
                post_gate / "attribution-control" / ATTRIBUTION_EVIDENCE_FILENAME
            ),
            command_buffer_root=command_root,
            command_buffer_cache_root=post_gate / "command-buffer-cache",
            command_buffer_evidence_path=(
                command_root / COMMAND_BUFFER_EVIDENCE_FILENAME
            ),
            telemetry_cache_root=post_gate / "telemetry-cache",
            newton_telemetry_path=post_gate / NEWTON_TELEMETRY_FILENAME,
            complete_path_root=complete_root,
            complete_path_evidence_path=complete_root / DOCUMENT_PATH,
            gap_inputs_path=post_gate / GAP_INPUTS_FILENAME,
            resume_spec_path=post_gate / RESUME_SPEC_FILENAME,
        )


@dataclass(frozen=True, slots=True)
class PostGateContext:
    """Validated pending state and exact identities used by every later stage."""

    c0_spec: Mapping[str, object]
    paths: PostGatePaths
    binding: CompletePathBinding
    input_root: Path
    input_bundle_sha256: str
    candidate_path: Path
    initial_parameter_sha256: str
    interpreter: Path
    snapshot_root: Path
    base_environment: Mapping[str, str]

    @property
    def profile_identity(self) -> ProfileEvidenceIdentity:
        return ProfileEvidenceIdentity(
            candidate_sha256=self.binding.candidate_sha256,
            specimen_sha256=self.binding.specimen_sha256,
            input_bundle_sha256=self.input_bundle_sha256,
            source_sha256=self.binding.source_sha256,
            runtime_identity_sha256=self.binding.runtime_identity_sha256,
            lane_id=_lane_id(self.binding.lane_id),
            gpu_uuid=self.binding.gpu_uuid,
            gate_checkpoint_sha256=self.binding.gate_checkpoint_sha256,
            warm_checkpoint_sha256=self.binding.warm_checkpoint_sha256,
            warm_p50_ns=self.binding.warm_p50_ns,
        )

    @property
    def telemetry_identity(self) -> TelemetryIdentity:
        identity = self.profile_identity
        return TelemetryIdentity(
            candidate_sha256=identity.candidate_sha256,
            specimen_sha256=identity.specimen_sha256,
            input_bundle_sha256=identity.input_bundle_sha256,
            source_sha256=identity.source_sha256,
            runtime_identity_sha256=identity.runtime_identity_sha256,
            lane_id=identity.lane_id,
            gpu_uuid=identity.gpu_uuid,
            gate_checkpoint_sha256=identity.gate_checkpoint_sha256,
            warm_checkpoint_sha256=identity.warm_checkpoint_sha256,
            warm_p50_ns=identity.warm_p50_ns,
        )

    @property
    def command_buffer_identity(self) -> dict[str, object]:
        return self.profile_identity.to_json()

    @property
    def attribution_binding(self) -> AttributionBinding:
        identity = self.profile_identity
        return AttributionBinding(
            candidate_sha256=identity.candidate_sha256,
            specimen_sha256=identity.specimen_sha256,
            input_bundle_sha256=identity.input_bundle_sha256,
            source_sha256=identity.source_sha256,
            production_runtime_identity_sha256=identity.runtime_identity_sha256,
            lane_id=identity.lane_id,
            gpu_uuid=identity.gpu_uuid,
            gate_checkpoint_sha256=identity.gate_checkpoint_sha256,
            warm_checkpoint_sha256=identity.warm_checkpoint_sha256,
            warm_p50_ns=identity.warm_p50_ns,
        )


def _lane_id(value: str) -> LaneId:
    if value == RTX_LANE_ID:
        return RTX_LANE_ID
    if value == A100_LANE_ID:
        return A100_LANE_ID
    raise Phase0PostGateError("checkpoint lane_id is unsupported")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_document(path: Path, context: str) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                Phase0PostGateError(
                    f"{context} contains non-finite constant {constant}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase0PostGateError(f"{context} is not readable JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Phase0PostGateError(f"{context} must be a JSON object")
    if raw != canonical_json_bytes(value):
        raise Phase0PostGateError(f"{context} is not canonical JSON")
    return value


def _write_exclusive(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def load_post_gate_context(
    c0_spec_path: Path,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> PostGateContext:
    """Validate the canonical C0 pending state and derive post-gate identity."""

    spec = _canonical_document(c0_spec_path, "C0 runner spec")
    if spec.get("schema_id") != C0_RUNNER_SPEC_SCHEMA_ID or "resume" in spec:
        raise Phase0PostGateError("C0 spec must be an original v3 measurement spec")
    output_value = spec.get("output_root")
    input_value = spec.get("input_root")
    candidate_value = spec.get("candidate_path")
    native_value = spec.get("native_reference_path")
    provenance = spec.get("provenance")
    if not all(
        isinstance(value, str) and value
        for value in (
            output_value,
            input_value,
            candidate_value,
            native_value,
        )
    ) or not isinstance(provenance, dict):
        raise Phase0PostGateError("C0 spec paths or provenance are invalid")
    paths = PostGatePaths.from_output_root(Path(cast(str, output_value)))
    gate_path = paths.c0_output_root / GATE_CHECKPOINT_FILENAME
    warm_path = paths.c0_output_root / WARM_CHECKPOINT_FILENAME
    state_path = paths.c0_output_root / STATE_FILENAME
    state = _canonical_document(state_path, "C0 pending state")
    gate_checkpoint = _canonical_document(gate_path, "C0 gate checkpoint")
    if state.get("schema_id") != PENDING_SCHEMA_ID or state.get("state") != (
        "POST_GATE_PENDING"
    ):
        raise Phase0PostGateError("C0 state is not POST_GATE_PENDING")
    if state.get("gate_checkpoint_sha256") != _sha256_path(gate_path) or state.get(
        "warm_checkpoint_sha256"
    ) != _sha256_path(warm_path):
        raise Phase0PostGateError("pending state checkpoint hashes do not match bytes")
    try:
        binding = binding_from_phase0_checkpoints(
            gate_path,
            warm_path,
            Path(cast(str, native_value)),
        )
    except (OSError, RuntimeError) as error:
        raise Phase0PostGateError(f"invalid C0 checkpoints: {error}") from error
    if (
        state.get("lane_id") != binding.lane_id
        or state.get("runtime_identity_sha256") != binding.runtime_identity_sha256
        or state.get("warm_p50_ns") != (binding.warm_p50_ns)
    ):
        raise Phase0PostGateError("pending state identity differs from checkpoints")
    initial_parameter_sha256 = gate_checkpoint.get("initial_parameter_sha256")
    if (
        not isinstance(initial_parameter_sha256, str)
        or len(initial_parameter_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in initial_parameter_sha256
        )
    ):
        raise Phase0PostGateError(
            "C0 gate checkpoint lacks the native-v3 initial parameter binding"
        )
    interpreter_value = provenance.get("interpreter_path")
    snapshot_value = provenance.get("immutable_root")
    if not isinstance(interpreter_value, str) or not isinstance(snapshot_value, str):
        raise Phase0PostGateError("C0 provenance lacks interpreter or snapshot root")
    specimen = spec.get("receipt_template")
    specimen_value = specimen.get("specimen") if isinstance(specimen, dict) else None
    input_sha = (
        specimen_value.get("input_bundle_sha256")
        if isinstance(specimen_value, dict)
        else None
    )
    if not isinstance(input_sha, str) or input_sha != _sha256_path(
        Path(cast(str, input_value)) / "input_bundle.json"
    ):
        raise Phase0PostGateError("C0 specimen input bundle binding is invalid")
    return PostGateContext(
        c0_spec=spec,
        paths=paths,
        binding=binding,
        input_root=Path(cast(str, input_value)).resolve(),
        input_bundle_sha256=input_sha,
        candidate_path=Path(cast(str, candidate_value)).resolve(),
        initial_parameter_sha256=initial_parameter_sha256,
        # Preserve the lexical venv entry point: resolving its symlink selects the
        # base interpreter and loses the venv's JAX/site-packages identity.
        interpreter=Path(interpreter_value).absolute(),
        snapshot_root=Path(snapshot_value).resolve(),
        base_environment=(os.environ if base_environment is None else base_environment),
    )


def _profile_environment(context: PostGateContext) -> dict[str, str]:
    environment = normalize_static_timing_environment(context.base_environment)
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(context.paths.profile_cache_root),
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
            "SINGLE_STAGE_COMPUTE_GRAPH_LANE": context.binding.lane_id,
            "SINGLE_STAGE_COMPUTE_GRAPH_MODE": "profile",
            "SINGLE_STAGE_COMPUTE_GRAPH_VARIANT": "C0",
            "SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY": (
                context.binding.runtime_identity_sha256
            ),
            TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT: str(TRACE_VIEWER_MAX_EVENTS),
        }
    )
    environment.pop("SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX", None)
    provenance = context.c0_spec["provenance"]
    if not isinstance(provenance, dict):
        raise Phase0PostGateError("C0 provenance is invalid")
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"] = json.dumps(
        {
            "runtime": provenance["runtime"],
            "static_environment": provenance["environment"],
            "route_environment": normalize_route_environment(environment),
            "policies": provenance["policies"],
            "expected_runtime_identity_sha256": (
                context.binding.runtime_identity_sha256
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return environment


def collect_profile_stage(
    context: PostGateContext,
    *,
    executor: CommandExecutor,
    timeout_seconds: float = 900.0,
) -> Path:
    """Run one isolated profile replay and derive evidence from its raw trace."""

    path = context.paths.profile_evidence_path
    if path.exists():
        parse_profile_evidence(path, expected_identity=context.profile_identity)
        return path
    trace_root = context.paths.profile_root / "raw-trace"
    raw_child_path = context.paths.profile_root / "child-observation.json"
    identity_anchor = context.paths.profile_root / "hlo-identity-anchor.json"
    if context.paths.profile_root.exists() or context.paths.profile_cache_root.exists():
        if not (
            context.paths.profile_root.is_dir()
            and context.paths.profile_cache_root.is_dir()
        ):
            raise Phase0PostGateError(
                "profile artifact and cache roots must both exist for resume"
            )
        return _finalize_profile_stage(
            context,
            trace_root=trace_root,
            raw_child_path=raw_child_path,
            identity_anchor=identity_anchor,
        )
    context.paths.profile_root.mkdir(parents=True)
    context.paths.profile_cache_root.mkdir(parents=True)
    module_args = (
        "--input-root",
        str(context.input_root),
        "--input-bundle-sha256",
        context.input_bundle_sha256,
        "--snapshot-root",
        str(context.snapshot_root),
        "--candidate",
        str(context.candidate_path),
        "--parameter-sha256",
        context.binding.candidate_sha256,
        "--initial-parameter-sha256",
        context.initial_parameter_sha256,
        "--trace-root",
        str(trace_root),
        "--identity-anchor",
        str(identity_anchor),
        "--gpu-uuid",
        context.binding.gpu_uuid,
    )
    launch = build_snapshot_module_launch(
        context.interpreter,
        context.snapshot_root,
        PROFILE_MODULE,
        module_args,
        _profile_environment(context),
    )
    result = executor(launch.argv, launch.environment, launch.cwd, timeout_seconds)
    if result.timed_out or result.returncode != 0:
        raise Phase0PostGateError(
            "isolated profile evaluator failed: " + result.stderr.strip()
        )
    try:
        observation = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise Phase0PostGateError("profile evaluator did not emit JSON") from error
    if not isinstance(observation, dict) or observation.get("mode") != "profile":
        raise Phase0PostGateError("profile evaluator observation is not profile mode")
    cold_compile = observation.get("cold_compile")
    if cold_compile is None:
        cold_compile = {}
        observation["cold_compile"] = cold_compile
    if not isinstance(cold_compile, dict):
        raise Phase0PostGateError("profile evaluator cold-compile evidence is invalid")
    cold_compile.pop("peak_self_rss_bytes", None)
    try:
        cold_compile.update(_command_rss_document(result, "profile child"))
    except RuntimeError as error:
        raise Phase0PostGateError(str(error)) from error
    _write_exclusive(raw_child_path, observation)
    return _finalize_profile_stage(
        context,
        trace_root=trace_root,
        raw_child_path=raw_child_path,
        identity_anchor=identity_anchor,
    )


def _finalize_profile_stage(
    context: PostGateContext,
    *,
    trace_root: Path,
    raw_child_path: Path,
    identity_anchor: Path,
) -> Path:
    """Derive evidence from complete evaluator-owned raw outputs, including resume."""

    observation = _canonical_document(raw_child_path, "profile child observation")
    if observation.get("mode") != "profile":
        raise Phase0PostGateError("profile child observation is not profile mode")
    anchor = _canonical_document(identity_anchor, "profile HLO identity anchor")
    trace_paths = tuple(trace_root.rglob("*.trace.json.gz"))
    if len(trace_paths) != 1:
        raise Phase0PostGateError("profile replay must produce exactly one raw trace")
    evidence = build_profile_evidence(
        trace_path=trace_paths[0],
        artifact_root=context.paths.profile_root,
        **context.profile_identity.to_json(),
    )
    expected_anchor = {
        "schema_id": IDENTITY_ANCHOR_SCHEMA_ID,
        "hlo_module_set_identity": evidence.profile.hlo_module_set_identity,
        "hlo_module_set_identity_source": (
            evidence.profile.hlo_module_set_identity_source
        ),
    }
    if anchor != expected_anchor:
        raise Phase0PostGateError(
            "profile HLO identity anchor differs from the raw trace"
        )
    path = context.paths.profile_evidence_path
    write_profile_evidence(path, evidence)
    parse_profile_evidence(path, expected_identity=context.profile_identity)
    return path


def _attribution_static_environment(
    context: PostGateContext,
    mode: Literal["default_control", "command_buffer_disabled"],
) -> tuple[dict[str, str], tuple[str, ...], str]:
    provenance = context.c0_spec["provenance"]
    if not isinstance(provenance, dict):
        raise Phase0PostGateError("C0 provenance is invalid")
    raw_static = provenance.get("environment")
    if not isinstance(raw_static, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_static.items()
    ):
        raise Phase0PostGateError("C0 static environment is invalid")
    static = dict(raw_static)
    default_flags = static.get("XLA_FLAGS", "")
    default_tokens = tuple(shlex.split(default_flags))
    command_buffer_tokens = tuple(
        token
        for token in default_tokens
        if token.startswith("--xla_gpu_enable_command_buffer")
    )
    if command_buffer_tokens:
        raise Phase0PostGateError(
            "authoritative default XLA_FLAGS already override command buffers"
        )
    if mode == "command_buffer_disabled":
        tokens = (*default_tokens, COMMAND_BUFFER_DISABLE_FLAG)
        static["XLA_FLAGS"] = shlex.join(tokens)
    else:
        tokens = default_tokens
    identity_provenance = {
        "interpreter_path": str(context.interpreter),
        "runtime": provenance.get("runtime"),
        "environment": static,
        "policies": provenance.get("policies"),
    }
    runtime_identity = _runtime_identity(identity_provenance)
    if (
        mode == "default_control"
        and runtime_identity != context.binding.runtime_identity_sha256
    ):
        raise Phase0PostGateError(
            "authoritative default runtime identity differs from the C0 checkpoint"
        )
    if (
        mode == "command_buffer_disabled"
        and runtime_identity == context.binding.runtime_identity_sha256
    ):
        raise Phase0PostGateError(
            "disabled replay runtime identity is not distinct from the default"
        )
    return static, tokens, runtime_identity


def _attribution_profile_identity(
    context: PostGateContext, runtime_identity_sha256: str
) -> ProfileEvidenceIdentity:
    identity = context.profile_identity
    return ProfileEvidenceIdentity(
        candidate_sha256=identity.candidate_sha256,
        specimen_sha256=identity.specimen_sha256,
        input_bundle_sha256=identity.input_bundle_sha256,
        source_sha256=identity.source_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        lane_id=identity.lane_id,
        gpu_uuid=identity.gpu_uuid,
        gate_checkpoint_sha256=identity.gate_checkpoint_sha256,
        warm_checkpoint_sha256=identity.warm_checkpoint_sha256,
        warm_p50_ns=identity.warm_p50_ns,
    )


def _attribution_environment(
    context: PostGateContext,
    *,
    mode: Literal["default_control", "command_buffer_disabled"],
    cache_root: Path,
) -> tuple[dict[str, str], tuple[str, ...], str]:
    static, tokens, runtime_identity = _attribution_static_environment(context, mode)
    environment = dict(static)
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(cache_root),
            "SINGLE_STAGE_COMPUTE_GRAPH_LANE": context.binding.lane_id,
            "SINGLE_STAGE_COMPUTE_GRAPH_MODE": "profile",
            "SINGLE_STAGE_COMPUTE_GRAPH_VARIANT": "C0",
            "SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY": runtime_identity,
            TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT: str(TRACE_VIEWER_MAX_EVENTS),
        }
    )
    provenance = context.c0_spec["provenance"]
    if not isinstance(provenance, dict):
        raise Phase0PostGateError("C0 provenance is invalid")
    environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"] = json.dumps(
        {
            "runtime": provenance["runtime"],
            "static_environment": static,
            "route_environment": normalize_route_environment(environment),
            "policies": provenance["policies"],
            "expected_runtime_identity_sha256": runtime_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return environment, tokens, runtime_identity


def _profile_topology_identity(
    context: PostGateContext, evidence: ComputeGraphProfileEvidence
) -> str:
    profile = evidence.profile
    return canonical_module_topology_identity(
        profile.hlo_module_set_identity,
        profile.hlo_module_set_identity_source,
        context.binding.specimen_sha256,
    )


def _attribution_attempt_from_outputs(
    context: PostGateContext,
    *,
    mode: Literal["default_control", "command_buffer_disabled"],
    attempt_index: int,
    attempt_root: Path,
    cache_root: Path,
    xla_flag_tokens: tuple[str, ...],
    runtime_identity_sha256: str,
    evidence: ComputeGraphProfileEvidence,
) -> AttributionAttempt:
    observation = _canonical_document(
        attempt_root / "child-observation.json",
        f"{mode} attribution child observation {attempt_index}",
    )
    if observation.get("mode") != "profile":
        raise Phase0PostGateError("attribution child observation is not profile mode")
    objective = observation.get("objective")
    gradient = observation.get("gradient")
    residuals = observation.get("residual_certificates")
    if (
        isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(objective)
        or not isinstance(gradient, list)
        or not gradient
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in gradient
        )
        or not isinstance(residuals, dict)
        or not residuals
    ):
        raise Phase0PostGateError("attribution child numerical evidence is invalid")
    solve_certificate = {
        "inner_newton_success": observation.get("inner_newton_success"),
        "adjoint_success": observation.get("adjoint_success"),
        "residual_certificates": residuals,
    }
    if (
        solve_certificate["inner_newton_success"] is not True
        or solve_certificate["adjoint_success"] is not True
    ):
        raise Phase0PostGateError("attribution child solve certificate is not passing")
    phase_device_ns = tuple(
        (
            phase_id,
            sum(interval.end_ns - interval.start_ns for interval in intervals),
        )
        for phase_id, intervals in evidence.profile.phase_interval_unions
    )
    return AttributionAttempt(
        mode=mode,
        attempt_index=attempt_index,
        binding=context.attribution_binding,
        runtime_identity_sha256=runtime_identity_sha256,
        xla_flag_tokens=xla_flag_tokens,
        compilation_cache_root=str(cache_root),
        artifact_root=str(attempt_root),
        raw_trace_path=(attempt_root / evidence.trace.path)
        .relative_to(context.paths.c0_output_root)
        .as_posix(),
        raw_trace_sha256=evidence.trace.sha256,
        child_observation_path=(attempt_root / "child-observation.json")
        .relative_to(context.paths.c0_output_root)
        .as_posix(),
        child_observation_sha256=_sha256_path(attempt_root / "child-observation.json"),
        hlo_anchor_path=(attempt_root / "hlo-identity-anchor.json")
        .relative_to(context.paths.c0_output_root)
        .as_posix(),
        hlo_anchor_sha256=_sha256_path(attempt_root / "hlo-identity-anchor.json"),
        profile_derivation_version=PROFILE_DERIVATION_VERSION,
        objective=float(objective),
        gradient=tuple(float(value) for value in gradient),
        solve_certificate=solve_certificate,
        module_topology_identity_sha256=_profile_topology_identity(context, evidence),
        evaluation_envelope_ns=evidence.profile.evaluation_envelope_ns,
        device_active_ns=evidence.profile.device_active_ns,
        phase_device_ns=phase_device_ns,
    )


def _collect_attribution_attempt(
    context: PostGateContext,
    *,
    mode: Literal["default_control", "command_buffer_disabled"],
    attempt_index: int,
    executor: CommandExecutor,
    timeout_seconds: float,
) -> AttributionAttempt:
    attempt_root = (
        context.paths.attribution_root / mode / f"attempt-{attempt_index:02d}"
    )
    cache_root = (
        context.paths.attribution_cache_root / mode / f"attempt-{attempt_index:02d}"
    )
    trace_root = attempt_root / "raw-trace"
    child_path = attempt_root / "child-observation.json"
    anchor_path = attempt_root / "hlo-identity-anchor.json"
    environment, xla_flag_tokens, runtime_identity = _attribution_environment(
        context, mode=mode, cache_root=cache_root
    )
    expected_identity = _attribution_profile_identity(context, runtime_identity)
    if attempt_root.exists() or cache_root.exists():
        if not (attempt_root.is_dir() and cache_root.is_dir()):
            raise Phase0PostGateError(
                "attribution artifact and cache roots must both exist for resume"
            )
    else:
        attempt_root.mkdir(parents=True)
        cache_root.mkdir(parents=True)
        module_args = (
            "--input-root",
            str(context.input_root),
            "--input-bundle-sha256",
            context.input_bundle_sha256,
            "--snapshot-root",
            str(context.snapshot_root),
            "--candidate",
            str(context.candidate_path),
            "--parameter-sha256",
            context.binding.candidate_sha256,
            "--initial-parameter-sha256",
            context.initial_parameter_sha256,
            "--trace-root",
            str(trace_root),
            "--identity-anchor",
            str(anchor_path),
            "--gpu-uuid",
            context.binding.gpu_uuid,
        )
        launch = build_snapshot_module_launch(
            context.interpreter,
            context.snapshot_root,
            PROFILE_MODULE,
            module_args,
            environment,
        )
        result = executor(launch.argv, launch.environment, launch.cwd, timeout_seconds)
        if result.timed_out or result.returncode != 0:
            raise Phase0PostGateError(
                f"isolated {mode} attribution evaluator failed: {result.stderr.strip()}"
            )
        try:
            observation = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise Phase0PostGateError(
                f"{mode} attribution evaluator did not emit JSON"
            ) from error
        if not isinstance(observation, dict) or observation.get("mode") != "profile":
            raise Phase0PostGateError(
                f"{mode} attribution observation is not profile mode"
            )
        cold_compile = observation.get("cold_compile")
        if cold_compile is None:
            cold_compile = {}
            observation["cold_compile"] = cold_compile
        if not isinstance(cold_compile, dict):
            raise Phase0PostGateError(
                f"{mode} attribution cold-compile evidence is invalid"
            )
        cold_compile.pop("peak_self_rss_bytes", None)
        try:
            cold_compile.update(_command_rss_document(result, f"{mode} child"))
        except RuntimeError as error:
            raise Phase0PostGateError(str(error)) from error
        _write_exclusive(child_path, observation)
    observation = _canonical_document(child_path, f"{mode} child observation")
    if observation.get("mode") != "profile":
        raise Phase0PostGateError("attribution child observation is not profile mode")
    anchor = _canonical_document(anchor_path, f"{mode} HLO identity anchor")
    trace_paths = tuple(trace_root.rglob("*.trace.json.gz"))
    if len(trace_paths) != 1:
        raise Phase0PostGateError(
            f"{mode} attribution replay must produce exactly one raw trace"
        )
    evidence = build_attribution_control_profile_evidence(
        trace_path=trace_paths[0],
        artifact_root=attempt_root,
        **expected_identity.to_json(),
    )
    expected_anchor = {
        "schema_id": IDENTITY_ANCHOR_SCHEMA_ID,
        "hlo_module_set_identity": evidence.profile.hlo_module_set_identity,
        "hlo_module_set_identity_source": evidence.profile.hlo_module_set_identity_source,
    }
    if anchor != expected_anchor:
        raise Phase0PostGateError(
            f"{mode} attribution HLO anchor differs from the raw trace"
        )
    return _attribution_attempt_from_outputs(
        context,
        mode=mode,
        attempt_index=attempt_index,
        attempt_root=attempt_root,
        cache_root=cache_root,
        xla_flag_tokens=xla_flag_tokens,
        runtime_identity_sha256=runtime_identity,
        evidence=evidence,
    )


def collect_attribution_control_stage(
    context: PostGateContext,
    *,
    executor: CommandExecutor,
    timeout_seconds: float = 900.0,
) -> Path:
    """Collect six isolated profiles and persist promoting or negative evidence."""

    defaults = tuple(
        _collect_attribution_attempt(
            context,
            mode="default_control",
            attempt_index=index,
            executor=executor,
            timeout_seconds=timeout_seconds,
        )
        for index in range(ATTRIBUTION_ATTEMPT_COUNT)
    )
    disabled = tuple(
        _collect_attribution_attempt(
            context,
            mode="command_buffer_disabled",
            attempt_index=index,
            executor=executor,
            timeout_seconds=timeout_seconds,
        )
        for index in range(ATTRIBUTION_ATTEMPT_COUNT)
    )
    document = build_attribution_evidence(defaults, disabled)
    path = context.paths.attribution_evidence_path
    if path.exists():
        persisted = _canonical_document(path, "attribution-control evidence")
        if persisted != document:
            raise Phase0PostGateError(
                "existing attribution-control evidence differs from raw attempts"
            )
        return path
    _write_exclusive(path, document)
    persisted = _canonical_document(path, "attribution-control evidence")
    if persisted != document:
        raise Phase0PostGateError(
            "persisted attribution-control evidence differs from constructed evidence"
        )
    return path


def collect_command_buffer_stage(
    context: PostGateContext,
    *,
    nsys_binary: Path,
    nvtx_library: Path,
    expected_nsys_version: str,
    current_xla_flags: str,
    executor: ControlExecutor = execute_control_plan,
) -> Path:
    """Run or validate the matched current-default/explicit-disable control."""

    path = context.paths.command_buffer_evidence_path
    if path.exists():
        validate_command_buffer_control_evidence(
            _canonical_document(path, "command-buffer evidence"),
            context.command_buffer_identity,
        )
        return path
    plan = build_control_plan(
        nsys_binary=nsys_binary,
        nvtx_library=nvtx_library,
        expected_nsys_version=expected_nsys_version,
        python_binary=context.interpreter,
        snapshot_root=context.snapshot_root,
        artifact_root=context.paths.command_buffer_root / "raw",
        cache_root=context.paths.command_buffer_cache_root,
        input_root=context.input_root,
        candidate_path=context.candidate_path,
        specimen_sha256=context.binding.specimen_sha256,
        candidate_sha256=context.binding.candidate_sha256,
        source_sha256=context.binding.source_sha256,
        gate_checkpoint_sha256=context.binding.gate_checkpoint_sha256,
        warm_checkpoint_sha256=context.binding.warm_checkpoint_sha256,
        warm_p50_ns=context.binding.warm_p50_ns,
        lane_id=_lane_id(context.binding.lane_id),
        gpu_uuid=context.binding.gpu_uuid,
        runtime_identity_sha256=context.binding.runtime_identity_sha256,
        input_bundle_sha256=context.input_bundle_sha256,
        current_xla_flags=current_xla_flags,
        base_environment=context.base_environment,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    document = executor(plan, path)
    if not path.exists():
        _write_exclusive(path, document)
    persisted = _canonical_document(path, "command-buffer evidence")
    if dict(persisted) != dict(document):
        raise Phase0PostGateError("command-buffer executor return differs from bytes")
    validate_command_buffer_control_evidence(persisted, context.command_buffer_identity)
    return path


def collect_newton_telemetry_stage(
    context: PostGateContext,
    *,
    executor: CommandExecutor,
    timeout_seconds: float = 900.0,
) -> Path:
    """Launch exact-route telemetry in an isolated snapshot process."""

    path = context.paths.newton_telemetry_path
    if path.exists():
        validate_newton_telemetry_evidence(
            _canonical_document(path, "Newton telemetry evidence"),
            context.telemetry_identity,
        )
        return path
    if context.paths.telemetry_cache_root.exists():
        raise Phase0PostGateError("telemetry cache root must be fresh")
    context.paths.post_gate_root.mkdir(parents=True, exist_ok=True)
    context.paths.telemetry_cache_root.mkdir()
    identity = context.telemetry_identity
    provenance = context.c0_spec["provenance"]
    if not isinstance(provenance, dict):
        raise Phase0PostGateError("C0 provenance is invalid")
    environment = normalize_static_timing_environment(context.base_environment)
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(context.paths.telemetry_cache_root),
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
            "JAX_TRANSFER_GUARD": "disallow",
        }
    )
    module_args = (
        "--input-root",
        str(context.input_root),
        "--candidate",
        str(context.candidate_path),
        "--candidate-sha256",
        identity.candidate_sha256,
        "--specimen-sha256",
        identity.specimen_sha256,
        "--input-bundle-sha256",
        identity.input_bundle_sha256,
        "--source-sha256",
        identity.source_sha256,
        "--runtime-identity-sha256",
        identity.runtime_identity_sha256,
        "--runtime-contract-json",
        json.dumps(
            {
                "runtime": provenance.get("runtime", {}),
                "static_environment": normalize_static_timing_environment(environment),
                "route_environment": normalize_route_environment(environment),
                "policies": provenance.get("policies", {}),
                "expected_runtime_identity_sha256": identity.runtime_identity_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--lane-id",
        identity.lane_id,
        "--gpu-uuid",
        identity.gpu_uuid,
        "--gate-checkpoint-sha256",
        identity.gate_checkpoint_sha256,
        "--warm-checkpoint-sha256",
        identity.warm_checkpoint_sha256,
        "--warm-p50-ns",
        str(identity.warm_p50_ns),
        "--output",
        str(path),
    )
    launch = build_snapshot_module_launch(
        context.interpreter,
        context.snapshot_root,
        TELEMETRY_MODULE,
        module_args,
        environment,
    )
    result = executor(launch.argv, launch.environment, launch.cwd, timeout_seconds)
    if result.timed_out or result.returncode != 0:
        raise Phase0PostGateError(
            "isolated Newton telemetry failed: " + result.stderr.strip()
        )
    validate_newton_telemetry_evidence(
        _canonical_document(path, "Newton telemetry evidence"), identity
    )
    return path


def collect_complete_path_stage(
    context: PostGateContext,
    *,
    specimen_document_path: Path,
    immutable_snapshot_provenance_paths: Mapping[ProfileId, Path] | None = None,
    executor: RunExecutor | None = None,
) -> Path:
    """Collect or validate exactly one native, C0, and Optax complete path."""

    path = context.paths.complete_path_evidence_path
    if path.exists():
        document = _canonical_document(path, "complete-path evidence")
        _validate_complete_path_identity(document, context)
        build_staged_gap_budget_timing_input(document)
        return path
    produced = collect_complete_path_evidence(
        artifact_root=context.paths.complete_path_root,
        specimen_document_path=specimen_document_path,
        input_bundle_path=context.input_root / "input_bundle.json",
        candidate_path=context.candidate_path,
        binding=context.binding,
        python_executable=str(context.interpreter),
        repo_root=context.snapshot_root,
        base_environment=context.base_environment,
        immutable_snapshot_provenance_paths=immutable_snapshot_provenance_paths,
        executor=executor,
    )
    document = _canonical_document(produced, "complete-path evidence")
    _validate_complete_path_identity(document, context)
    build_staged_gap_budget_timing_input(document)
    return produced


def _validate_complete_path_identity(
    document: Mapping[str, object], context: PostGateContext
) -> None:
    identity = document.get("identity")
    if not isinstance(identity, dict):
        raise Phase0PostGateError("complete-path evidence lacks identity")
    expected = context.command_buffer_identity
    for field, value in expected.items():
        if identity.get(field) != value:
            raise Phase0PostGateError(f"complete-path identity mismatch for {field}")


def build_gap_inputs_stage(
    context: PostGateContext,
    policy: GapBudgetPolicyInput,
) -> Path:
    """Build or validate typed gap inputs directly from complete-path evidence."""

    complete = _canonical_document(
        context.paths.complete_path_evidence_path, "complete-path evidence"
    )
    _validate_complete_path_identity(complete, context)
    path = context.paths.gap_inputs_path
    if path.exists():
        validate_gap_budget_inputs_artifact(
            _canonical_document(path, "gap-budget inputs"), complete
        )
        return path
    document = build_gap_budget_inputs_artifact(complete, policy)
    validate_gap_budget_inputs_artifact(document, complete)
    _write_exclusive(path, document)
    return path


def build_resume_spec(context: PostGateContext) -> Mapping[str, object]:
    """Publish the only C0 resume spec accepted by final assembly."""

    profile = parse_profile_evidence(
        context.paths.profile_evidence_path,
        expected_identity=context.profile_identity,
    )
    del profile
    attribution = _canonical_document(
        context.paths.attribution_evidence_path, "attribution-control evidence"
    )
    try:
        require_promoting_attribution_evidence(attribution)
    except RuntimeError as error:
        blockers = attribution.get("blockers")
        raise Phase0PostGateError(
            f"attribution control is explicitly non-promoting: {blockers}"
        ) from error
    validate_command_buffer_control_evidence(
        _canonical_document(
            context.paths.command_buffer_evidence_path, "command-buffer evidence"
        ),
        context.command_buffer_identity,
    )
    validate_newton_telemetry_evidence(
        _canonical_document(
            context.paths.newton_telemetry_path, "Newton telemetry evidence"
        ),
        context.telemetry_identity,
    )
    complete = _canonical_document(
        context.paths.complete_path_evidence_path, "complete-path evidence"
    )
    _validate_complete_path_identity(complete, context)
    validate_gap_budget_inputs_artifact(
        _canonical_document(context.paths.gap_inputs_path, "gap-budget inputs"),
        complete,
    )
    resume = {
        "gate_checkpoint_path": str(
            context.paths.c0_output_root / GATE_CHECKPOINT_FILENAME
        ),
        "warm_checkpoint_path": str(
            context.paths.c0_output_root / WARM_CHECKPOINT_FILENAME
        ),
        "profile_evidence_path": str(context.paths.profile_evidence_path),
        "attribution_control_evidence_path": str(
            context.paths.attribution_evidence_path
        ),
        "command_buffer_evidence_path": str(context.paths.command_buffer_evidence_path),
        "newton_telemetry_evidence_path": str(context.paths.newton_telemetry_path),
        "complete_path_evidence_path": str(context.paths.complete_path_evidence_path),
        "gap_budget_inputs_path": str(context.paths.gap_inputs_path),
    }
    spec = {**dict(context.c0_spec), "resume": resume}
    path = context.paths.resume_spec_path
    if path.exists():
        persisted = _canonical_document(path, "resume spec")
        if persisted != spec:
            raise Phase0PostGateError("existing resume spec differs from evidence")
        return persisted
    _write_exclusive(path, spec)
    return spec


def assemble_phase0_receipt(
    context: PostGateContext,
    *,
    assembler: FinalAssembler = run_c0_measurement,
) -> Mapping[str, object]:
    """Invoke the C0 runner for its canonical final validation and assembly."""

    receipt_path = context.paths.c0_output_root / RECEIPT_FILENAME
    if receipt_path.exists():
        document, _audit = load_phase0_receipt(receipt_path)
        return document
    return assembler(build_resume_spec(context))


def _load_policy(path: Path) -> GapBudgetPolicyInput:
    document = _canonical_document(path, "gap policy")
    assumptions_value = document.get("phase_reduction_assumptions")
    levers_value = document.get("faithful_levers")
    if not isinstance(assumptions_value, dict) or not isinstance(levers_value, list):
        raise Phase0PostGateError("gap policy assumptions or levers are invalid")
    assumptions: dict[str, PhaseReductionAssumption] = {}
    for phase_id, value in assumptions_value.items():
        if not isinstance(phase_id, str) or not isinstance(value, dict):
            raise Phase0PostGateError("gap policy phase assumption is invalid")
        assumptions[phase_id] = PhaseReductionAssumption(
            conservative_reduction=float(value["conservative_reduction"]),
            optimistic_reduction=float(value["optimistic_reduction"]),
            overlap_disposition=cast(
                Literal["disjoint", "excluded_overlap"],
                value["overlap_disposition"],
            ),
        )
    levers = tuple(
        FaithfulLever(
            lever_id=str(value["lever_id"]),
            disposition=cast(Literal["bounded", "unbounded"], value["disposition"]),
            evidence_sha256=str(value["evidence_sha256"]),
        )
        for value in levers_value
        if isinstance(value, dict)
    )
    if len(levers) != len(levers_value):
        raise Phase0PostGateError("gap policy lever is invalid")
    return GapBudgetPolicyInput(
        phase_reduction_assumptions=assumptions,
        unattributed_conservative_reduction=float(
            document["unattributed_conservative_reduction"]
        ),
        unattributed_optimistic_reduction=float(
            document["unattributed_optimistic_reduction"]
        ),
        faithful_levers=levers,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("profile")
    commands.add_parser("attribution-control")
    control = commands.add_parser("command-buffer")
    control.add_argument("--nsys-binary", type=Path, required=True)
    control.add_argument("--nvtx-library", type=Path, required=True)
    control.add_argument("--nsys-version", required=True)
    control.add_argument("--current-xla-flags", default="")
    complete = commands.add_parser("complete-path")
    complete.add_argument("--specimen-document", type=Path, required=True)
    complete.add_argument("--native-snapshot-provenance", type=Path, required=True)
    complete.add_argument("--c0-snapshot-provenance", type=Path, required=True)
    complete.add_argument("--optax-snapshot-provenance", type=Path, required=True)
    gap = commands.add_parser("gap-inputs")
    gap.add_argument("--policy", type=Path, required=True)
    commands.add_parser("telemetry")
    commands.add_parser("assemble")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        context = load_post_gate_context(options.spec)
        if options.command == "profile":
            from benchmarks.single_stage_compute_graph_c0_runner import (
                _subprocess_executor,
            )

            collect_profile_stage(context, executor=_subprocess_executor)
        elif options.command == "attribution-control":
            from benchmarks.single_stage_compute_graph_c0_runner import (
                _subprocess_executor,
            )

            collect_attribution_control_stage(context, executor=_subprocess_executor)
        elif options.command == "command-buffer":
            collect_command_buffer_stage(
                context,
                nsys_binary=options.nsys_binary,
                nvtx_library=options.nvtx_library,
                expected_nsys_version=options.nsys_version,
                current_xla_flags=options.current_xla_flags,
            )
        elif options.command == "telemetry":
            from benchmarks.single_stage_compute_graph_c0_runner import (
                _subprocess_executor,
            )

            collect_newton_telemetry_stage(context, executor=_subprocess_executor)
        elif options.command == "complete-path":
            provenance_paths: dict[ProfileId, Path] = {
                "native_cpu": options.native_snapshot_provenance,
                "jax_gpu_fast": options.c0_snapshot_provenance,
                "jax_gpu_optax": options.optax_snapshot_provenance,
            }
            collect_complete_path_stage(
                context,
                specimen_document_path=options.specimen_document,
                immutable_snapshot_provenance_paths=provenance_paths,
            )
        elif options.command == "gap-inputs":
            build_gap_inputs_stage(context, _load_policy(options.policy))
        else:
            assemble_phase0_receipt(context)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Phase 0 post-gate orchestration failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
