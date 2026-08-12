"""Nonclaim GPU trace-schema and device-scope survival preflight.

This preflight compiles and warms two tiny, independent JAX kernels before
profiling them in two sequential sessions. It validates both Chrome traces
with the campaign parser, including the exact ``/device:GPU:0`` process and
required scopes. Its result is an instrumentation compatibility check only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
for _import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

import jax
import jax.numpy as jnp
from simsopt_jax.runtime.trace_annotations import PhaseId, device_scope, trace_session

from benchmarks.single_stage_changed_state_profiler_policy import (
    PROFILED_PROFILER_POLICY,
    build_jax_profiler_options,
    profiler_policy_document,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    TRACE_SCHEMA_ID,
    TraceSummaryError,
    _device_intervals,
    _host_transfer_spans,
    _parse_trace_document,
    _scope_paths_from_span,
    load_trace_document,
)

PREFLIGHT_SCHEMA_ID: Final = "single-stage-changed-state-trace-preflight-v2"
EVIDENCE_FILENAME: Final = "preflight.json"
TRACE_DIRECTORY_NAME: Final = "trace"
REQUIRED_SCOPE_PHASES: Final = (
    PhaseId.NEWTON_RESIDUAL_JVP,
    PhaseId.ADJOINT_LU_SOLVE,
)
OBSERVED_EVIDENCE_COUNT_FIELDS: Final = (
    "device_kernel_intervals_containing_scope",
    "uniquely_attributed_device_kernel_intervals",
    "ambiguous_device_kernel_intervals",
)
OBSERVED_EVIDENCE_FIELDS: Final = ("phase_id", *OBSERVED_EVIDENCE_COUNT_FIELDS)
PREFLIGHT_EVIDENCE_FIELDS: Final = (
    "schema_id",
    "state",
    "trace_schema_id",
    "required_scopes",
    "observed_evidence",
    "device_identity",
    "profiler_policy",
    "session_evidence",
    "failure_reason",
)
PREFLIGHT_SESSION_COUNT: Final = 2
EXPECTED_DEVICE_PROCESS: Final = "/device:GPU:0"


class PreflightEvidenceError(ValueError):
    """Persisted preflight evidence does not satisfy the producer contract."""


@dataclass(frozen=True)
class DeviceIdentity:
    """Caller-supplied identity for the GPU whose trace is being preflighted."""

    name: str
    uuid: str

    def __post_init__(self) -> None:
        if not self.name or not self.uuid:
            raise ValueError("device name and UUID must be non-empty")

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "uuid": self.uuid}


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _scope_evidence(
    document: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    spans, device_pids = _parse_trace_document(document)
    device_spans = tuple(span for span in spans if span.pid in device_pids)
    device_intervals = _device_intervals(
        spans,
        device_pids,
        _host_transfer_spans(spans),
    )
    if len(device_spans) != len(device_intervals):
        raise TraceSummaryError(
            "unknown_trace_schema",
            "device span and parsed device interval counts differ",
        )

    evidence: list[dict[str, object]] = []
    for required_phase in REQUIRED_SCOPE_PHASES:
        containing_count = 0
        unique_count = 0
        ambiguous_count = 0
        for span, interval in zip(device_spans, device_intervals, strict=True):
            if interval.kind != "kernel":
                continue
            scope_paths = _scope_paths_from_span(span)
            if not any(required_phase in path for path in scope_paths):
                continue
            containing_count += 1
            if interval.attribution.ambiguous:
                ambiguous_count += 1
            elif interval.attribution.phase is required_phase:
                unique_count += 1
        evidence.append(
            {
                "phase_id": required_phase.value,
                OBSERVED_EVIDENCE_COUNT_FIELDS[0]: containing_count,
                OBSERVED_EVIDENCE_COUNT_FIELDS[1]: unique_count,
                OBSERVED_EVIDENCE_COUNT_FIELDS[2]: ambiguous_count,
            }
        )
    return tuple(evidence)


def evaluate_trace_scope_survival(
    document: Mapping[str, object],
    *,
    device_identity: DeviceIdentity,
) -> dict[str, object]:
    """Return canonicalizable fail-closed evidence for one parsed trace."""

    try:
        observed_evidence = _scope_evidence(document)
        failed = tuple(
            row
            for row in observed_evidence
            if row[OBSERVED_EVIDENCE_COUNT_FIELDS[1]] == 0
            or row[OBSERVED_EVIDENCE_COUNT_FIELDS[2]] != 0
        )
        if failed:
            failures = ", ".join(str(row["phase_id"]) for row in failed)
            failure_reason = (
                "required scopes did not survive with exclusively unique device "
                f"attribution: {failures}"
            )
            state = "failed"
        else:
            failure_reason = None
            state = "pass"
    except TraceSummaryError as error:
        observed_evidence = ()
        failure_reason = f"{error.code}: {error}"
        state = "failed"

    return {
        "schema_id": PREFLIGHT_SCHEMA_ID,
        "state": state,
        "trace_schema_id": TRACE_SCHEMA_ID,
        "required_scopes": [phase.value for phase in REQUIRED_SCOPE_PHASES],
        "observed_evidence": list(observed_evidence),
        "device_identity": device_identity.to_json(),
        "profiler_policy": profiler_policy_document(PROFILED_PROFILER_POLICY),
        "session_evidence": [],
        "failure_reason": failure_reason,
    }


def _device_process_names(document: Mapping[str, object]) -> tuple[str, ...]:
    """Return exact device process names declared by one Chrome trace."""

    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise TraceSummaryError("unknown_trace_schema", "traceEvents must be an array")
    names: list[str] = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "M":
            continue
        if event.get("name") != "process_name":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            raise TraceSummaryError(
                "unknown_trace_schema", "process_name args must be an object"
            )
        name = args.get("name")
        if not isinstance(name, str):
            raise TraceSummaryError(
                "unknown_trace_schema", "process_name must be a string"
            )
        if name.startswith("/device:"):
            names.append(name)
    return tuple(sorted(set(names)))


def evaluate_preflight_sessions(
    documents: Sequence[Mapping[str, object]],
    *,
    device_identity: DeviceIdentity,
) -> dict[str, object]:
    """Require two sequential traces to retain GPU kernels and exact scopes."""

    session_evidence: list[dict[str, object]] = []
    try:
        if len(documents) != PREFLIGHT_SESSION_COUNT:
            raise TraceSummaryError(
                "unknown_trace_schema",
                f"expected {PREFLIGHT_SESSION_COUNT} sequential traces",
            )
        for index, document in enumerate(documents, start=1):
            evidence = evaluate_trace_scope_survival(
                document, device_identity=device_identity
            )
            device_processes = _device_process_names(document)
            if device_processes != (EXPECTED_DEVICE_PROCESS,):
                raise TraceSummaryError(
                    "unknown_trace_schema",
                    "trace does not declare exactly /device:GPU:0",
                )
            if evidence["state"] != "pass":
                raise TraceSummaryError(
                    "scope_survival_failed", str(evidence["failure_reason"])
                )
            session_evidence.append(
                {
                    "session_id": f"session-{index:02d}",
                    "device_processes": list(device_processes),
                    "observed_evidence": evidence["observed_evidence"],
                }
            )
        state = "pass"
        failure_reason = None
    except TraceSummaryError as error:
        state = "failed"
        failure_reason = f"{error.code}: {error}"

    observed_evidence = []
    if session_evidence:
        for required_phase in REQUIRED_SCOPE_PHASES:
            rows = tuple(
                row
                for session in session_evidence
                for row in session["observed_evidence"]
                if row["phase_id"] == required_phase.value
            )
            observed_evidence.append(
                {
                    "phase_id": required_phase.value,
                    **{
                        field: sum(int(row[field]) for row in rows)
                        for field in OBSERVED_EVIDENCE_COUNT_FIELDS
                    },
                }
            )
    return {
        "schema_id": PREFLIGHT_SCHEMA_ID,
        "state": state,
        "trace_schema_id": TRACE_SCHEMA_ID,
        "required_scopes": [phase.value for phase in REQUIRED_SCOPE_PHASES],
        "observed_evidence": observed_evidence,
        "device_identity": device_identity.to_json(),
        "profiler_policy": profiler_policy_document(PROFILED_PROFILER_POLICY),
        "session_evidence": session_evidence,
        "failure_reason": failure_reason,
    }


def validate_passing_preflight_evidence(
    evidence: Mapping[str, object],
    *,
    trace_schema_id: str,
    device_identity: DeviceIdentity,
) -> None:
    """Validate one passing document against the producer-owned exact schema."""

    if set(evidence) != set(PREFLIGHT_EVIDENCE_FIELDS):
        raise PreflightEvidenceError("preflight evidence schema fields drifted")
    if evidence.get("schema_id") != PREFLIGHT_SCHEMA_ID:
        raise PreflightEvidenceError("preflight schema identity drifted")
    if evidence.get("trace_schema_id") != trace_schema_id:
        raise PreflightEvidenceError("preflight parser schema identity drifted")
    if evidence.get("required_scopes") != [
        phase.value for phase in REQUIRED_SCOPE_PHASES
    ]:
        raise PreflightEvidenceError("preflight required scope contract drifted")
    if evidence.get("device_identity") != device_identity.to_json():
        raise PreflightEvidenceError("preflight device identity drifted")
    if evidence.get("profiler_policy") != profiler_policy_document(
        PROFILED_PROFILER_POLICY
    ):
        raise PreflightEvidenceError("preflight profiler policy drifted")
    sessions = evidence.get("session_evidence")
    if not isinstance(sessions, list) or len(sessions) != PREFLIGHT_SESSION_COUNT:
        raise PreflightEvidenceError("preflight session evidence is incomplete")
    for index, session in enumerate(sessions, start=1):
        if not isinstance(session, dict) or set(session) != {
            "session_id",
            "device_processes",
            "observed_evidence",
        }:
            raise PreflightEvidenceError("preflight session schema drifted")
        if session["session_id"] != f"session-{index:02d}":
            raise PreflightEvidenceError("preflight session order drifted")
        if session["device_processes"] != [EXPECTED_DEVICE_PROCESS]:
            raise PreflightEvidenceError("preflight device process drifted")
        _validate_scope_observations(session["observed_evidence"])
    if evidence.get("state") != "pass" or evidence.get("failure_reason") is not None:
        raise PreflightEvidenceError(
            "preflight did not pass: " + str(evidence.get("failure_reason"))
        )

    _validate_scope_observations(evidence.get("observed_evidence"))


def _validate_scope_observations(observations: object) -> None:
    """Validate complete, unique and unambiguous evidence for all scopes."""

    if not isinstance(observations, list) or len(observations) != len(
        REQUIRED_SCOPE_PHASES
    ):
        raise PreflightEvidenceError("preflight scope evidence is incomplete")
    by_phase: dict[str, Mapping[str, object]] = {}
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != set(
            OBSERVED_EVIDENCE_FIELDS
        ):
            raise PreflightEvidenceError("preflight scope evidence schema drifted")
        phase_id = observation.get("phase_id")
        if not isinstance(phase_id, str) or phase_id in by_phase:
            raise PreflightEvidenceError("preflight phase identity is invalid")
        by_phase[phase_id] = observation
    if set(by_phase) != {phase.value for phase in REQUIRED_SCOPE_PHASES}:
        raise PreflightEvidenceError("preflight phase identities drifted")
    for phase_id, observation in by_phase.items():
        counts = tuple(observation[field] for field in OBSERVED_EVIDENCE_COUNT_FIELDS)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counts
        ):
            raise PreflightEvidenceError(f"preflight counts are invalid for {phase_id}")
        containing, unique, ambiguous = counts
        if containing < 1 or unique < 1 or ambiguous != 0 or unique > containing:
            raise PreflightEvidenceError(
                f"preflight attribution is invalid for {phase_id}"
            )


def _build_jitted_canary() -> tuple[
    Callable[[jax.Array], jax.Array],
    Callable[[jax.Array], jax.Array],
]:
    @jax.jit
    def residual_jvp_kernel(values: jax.Array) -> jax.Array:
        with device_scope(PhaseId.NEWTON_RESIDUAL_JVP):
            return jnp.sin(values) + values * values

    @jax.jit
    def adjoint_lu_solve_kernel(values: jax.Array) -> jax.Array:
        with device_scope(PhaseId.ADJOINT_LU_SOLVE):
            return jnp.cos(values) - values

    return residual_jvp_kernel, adjoint_lu_solve_kernel


def _execute_canary(trace_root: Path) -> tuple[Path, ...]:
    residual_jvp_kernel, adjoint_lu_solve_kernel = _build_jitted_canary()
    gpu_devices = tuple(jax.devices("gpu"))
    if len(gpu_devices) != 1:
        raise RuntimeError(
            f"trace preflight requires exactly one visible GPU, found {len(gpu_devices)}"
        )
    values = jax.device_put(jnp.arange(32, dtype=jnp.float32), gpu_devices[0])
    with trace_session():
        jax.block_until_ready(residual_jvp_kernel(values))
        jax.block_until_ready(adjoint_lu_solve_kernel(values))
    trace_paths: list[Path] = []
    for session_index in range(1, PREFLIGHT_SESSION_COUNT + 1):
        session_root = trace_root / f"session-{session_index:02d}"
        session_root.mkdir()
        with trace_session():
            jax.profiler.start_trace(
                str(session_root),
                profiler_options=build_jax_profiler_options(
                    jax.profiler.ProfileOptions, PROFILED_PROFILER_POLICY
                ),
            )
            try:
                jax.block_until_ready(residual_jvp_kernel(values))
                jax.block_until_ready(adjoint_lu_solve_kernel(values))
            finally:
                jax.profiler.stop_trace()
        trace_paths.append(_single_trace_path(session_root))
    return tuple(trace_paths)


def _single_trace_path(trace_root: Path) -> Path:
    paths = tuple(sorted(trace_root.rglob("*.trace.json.gz")))
    if len(paths) != 1:
        raise TraceSummaryError(
            "unknown_trace_schema",
            f"expected exactly one Chrome trace, found {len(paths)}",
        )
    return paths[0]


def run_trace_preflight(
    output_root: Path,
    *,
    device_identity: DeviceIdentity,
) -> dict[str, object]:
    """Create a fresh preflight directory and return its persisted evidence."""

    output_root.mkdir(parents=True, exist_ok=False)
    trace_root = output_root / TRACE_DIRECTORY_NAME
    trace_root.mkdir()
    # This is the top-level integrity boundary: every preflight failure must
    # leave canonical evidence instead of an absent or partially claimed pass.
    try:
        trace_paths = _execute_canary(trace_root)
        evidence = evaluate_preflight_sessions(
            [load_trace_document(trace_path) for trace_path in trace_paths],
            device_identity=device_identity,
        )
    except Exception as error:  # noqa: BLE001
        evidence = {
            "schema_id": PREFLIGHT_SCHEMA_ID,
            "state": "failed",
            "trace_schema_id": TRACE_SCHEMA_ID,
            "required_scopes": [phase.value for phase in REQUIRED_SCOPE_PHASES],
            "observed_evidence": [],
            "device_identity": device_identity.to_json(),
            "profiler_policy": profiler_policy_document(PROFILED_PROFILER_POLICY),
            "session_evidence": [],
            "failure_reason": f"{type(error).__name__}: {error}",
        }
    (output_root / EVIDENCE_FILENAME).write_bytes(_canonical_json_bytes(evidence))
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the nonclaim changed-state GPU trace preflight."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--device-uuid", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = run_trace_preflight(
        args.output_root,
        device_identity=DeviceIdentity(args.device_name, args.device_uuid),
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence["state"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
