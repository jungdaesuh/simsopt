"""Matched native/C0/Optax complete-optimization timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
from examples.jax.parity.artifacts import canonical_json_bytes, write_bytes_exclusive
from examples.jax.parity.input_bundle import read_input_bundle
from examples.jax.parity.provenance import (
    SNAPSHOT_LANE_IDENTITY_SCHEMA_ID,
    lane_provenance_payload,
    load_snapshot_lane_identity,
    normalize_snapshot_lane_environment,
)

from benchmarks import run_jax_native_example_measurements as measurement_runner
from benchmarks.single_stage_compute_graph_snapshot import (
    IMPORT_ATTESTATION_SCHEMA_ID,
    MANIFEST_FILENAME,
    SnapshotError,
    load_snapshot_manifest,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    canonical_json_bytes as snapshot_canonical_json_bytes,
)
from benchmarks.single_stage_speed_campaign_receipt import _validate_endpoint_audit

SCHEMA_ID: Final = "single-stage-compute-graph-complete-path-v2"
GAP_BUDGET_INPUTS_SCHEMA_ID: Final = "single-stage-compute-graph-gap-budget-inputs-v1"
LANE_SNAPSHOT_PROVENANCE_SCHEMA_ID: Final = SNAPSHOT_LANE_IDENTITY_SCHEMA_ID
DOCUMENT_PATH: Final = "complete_path.json"
ProfileId = Literal["native_cpu", "jax_gpu_fast", "jax_gpu_optax"]
LaneId = Literal["native", "c0", "optax"]
Phase = Literal["cold"]
PROFILE_IDS: Final[tuple[ProfileId, ...]] = (
    "native_cpu",
    "jax_gpu_fast",
    "jax_gpu_optax",
)
LANE_IDS: Final[dict[ProfileId, LaneId]] = {
    "native_cpu": "native",
    "jax_gpu_fast": "c0",
    "jax_gpu_optax": "optax",
}
EXPECTED_DRIVERS: Final = {
    profile_id: measurement_runner._SINGLE_STAGE_EXPECTED_DRIVERS[profile_id]
    for profile_id in PROFILE_IDS
}
EXPECTED_BACKENDS: Final = {
    "native_cpu": "native_cpu",
    "jax_gpu_fast": "jax_gpu_fast",
    "jax_gpu_optax": "jax_gpu_fast",
}
GAP_BUDGET_COUNT_SEMANTICS: Final = (
    "scipy_optimize_result_nfev_for_combined_objective_and_gradient_callable_"
    "within_complete_path_boundary"
)


class CompletePathEvidenceError(RuntimeError):
    """Complete-path evidence is incomplete, unmatched, or misbound."""


def _sha256(value: str, context: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CompletePathEvidenceError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _positive_number(value: float, context: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise CompletePathEvidenceError(f"{context} must be finite and positive")
    return value


@dataclass(frozen=True, slots=True)
class CompletePathBinding:
    """Immutable Phase 0 identities shared by every complete-path sample."""

    specimen_sha256: str
    candidate_sha256: str
    source_sha256: str
    runtime_identity_sha256: str
    native_reference_sha256: str
    gate_checkpoint_sha256: str
    warm_checkpoint_sha256: str
    warm_p50_ns: float
    lane_id: str
    gpu_uuid: str

    def __post_init__(self) -> None:
        for name in (
            "specimen_sha256",
            "candidate_sha256",
            "source_sha256",
            "runtime_identity_sha256",
            "native_reference_sha256",
            "gate_checkpoint_sha256",
            "warm_checkpoint_sha256",
        ):
            _sha256(getattr(self, name), name)
        _positive_number(self.warm_p50_ns, "warm_p50_ns")
        for name in ("lane_id", "gpu_uuid"):
            if not getattr(self, name):
                raise CompletePathEvidenceError(f"{name} must be non-empty")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(path: Path, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletePathEvidenceError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CompletePathEvidenceError(f"{context} must be a JSON object")
    return value


def _required_string(document: Mapping[str, object], field: str, context: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise CompletePathEvidenceError(f"{context}.{field} must be a string")
    return value


def binding_from_phase0_checkpoints(
    gate_checkpoint_path: Path,
    warm_checkpoint_path: Path,
    native_reference_path: Path,
) -> CompletePathBinding:
    """Recompute the checkpoint links used by complete-path evidence."""

    gate = _json_mapping(gate_checkpoint_path, "gate checkpoint")
    warm = _json_mapping(warm_checkpoint_path, "warm checkpoint")
    if gate.get("state") != "PASSED" or warm.get("state") != "COMPLETE":
        raise CompletePathEvidenceError("Phase 0 checkpoints are not complete")
    gate_sha256 = _sha256_path(gate_checkpoint_path)
    warm_sha256 = _sha256_path(warm_checkpoint_path)
    if warm.get("gate_checkpoint_sha256") != gate_sha256:
        raise CompletePathEvidenceError("warm checkpoint does not bind the gate")
    for field in (
        "lane_id",
        "gpu_uuid",
        "specimen_sha256",
        "parameter_sha256",
        "source_state_sha256",
        "runtime_identity_sha256",
    ):
        if warm.get(field) != gate.get(field):
            raise CompletePathEvidenceError(
                f"warm checkpoint does not match gate field {field}"
            )
    native_reference_sha256 = _sha256_path(native_reference_path)
    if gate.get("native_reference_sha256") != native_reference_sha256:
        raise CompletePathEvidenceError(
            "native reference bytes do not match the gate checkpoint"
        )
    warm_measurement = warm.get("warm_measurement")
    if not isinstance(warm_measurement, dict):
        raise CompletePathEvidenceError("warm checkpoint lacks warm_measurement")
    warm_p50 = warm_measurement.get("p50_ns")
    if isinstance(warm_p50, bool) or not isinstance(warm_p50, (int, float)):
        raise CompletePathEvidenceError("warm checkpoint p50_ns must be numeric")
    return CompletePathBinding(
        specimen_sha256=_sha256(
            _required_string(gate, "specimen_sha256", "gate checkpoint"),
            "specimen_sha256",
        ),
        candidate_sha256=_sha256(
            _required_string(gate, "parameter_sha256", "gate checkpoint"),
            "candidate_sha256",
        ),
        source_sha256=_sha256(
            _required_string(gate, "source_state_sha256", "gate checkpoint"),
            "source_sha256",
        ),
        runtime_identity_sha256=_sha256(
            _required_string(gate, "runtime_identity_sha256", "gate checkpoint"),
            "runtime_identity_sha256",
        ),
        native_reference_sha256=native_reference_sha256,
        gate_checkpoint_sha256=gate_sha256,
        warm_checkpoint_sha256=warm_sha256,
        warm_p50_ns=float(warm_p50),
        lane_id=_required_string(gate, "lane_id", "gate checkpoint"),
        gpu_uuid=_required_string(gate, "gpu_uuid", "gate checkpoint"),
    )


def _validate_specimen_binding(
    binding: CompletePathBinding,
    specimen_document_path: Path,
    input_bundle_path: Path,
    candidate_path: Path,
) -> None:
    specimen_document = _json_mapping(specimen_document_path, "specimen document")
    if specimen_document.get("specimen_sha256") != binding.specimen_sha256:
        raise CompletePathEvidenceError("specimen document SHA identity mismatch")
    input_reference = specimen_document.get("input_bundle")
    candidate_reference = specimen_document.get("candidate")
    specimen = specimen_document.get("specimen")
    if (
        not isinstance(input_reference, dict)
        or not isinstance(candidate_reference, dict)
        or not isinstance(specimen, dict)
    ):
        raise CompletePathEvidenceError("specimen document lacks input references")
    input_relative = input_reference.get("relative_path")
    candidate_relative = candidate_reference.get("relative_path")
    if not isinstance(input_relative, str) or not isinstance(candidate_relative, str):
        raise CompletePathEvidenceError("specimen relative paths are invalid")
    expected_input = (
        specimen_document_path.parent / input_relative / "input_bundle.json"
    ).resolve()
    expected_candidate = (specimen_document_path.parent / candidate_relative).resolve()
    if input_bundle_path.resolve() != expected_input:
        raise CompletePathEvidenceError("input bundle is not the specimen input")
    if _sha256_path(input_bundle_path) != specimen.get("input_bundle_sha256"):
        raise CompletePathEvidenceError("input bundle bytes do not match the specimen")
    if candidate_path.resolve() != expected_candidate or not candidate_path.is_file():
        raise CompletePathEvidenceError("candidate is not the specimen candidate")
    candidate = np.load(candidate_path, allow_pickle=False)
    if candidate.dtype != np.dtype(np.float64) or candidate.shape != (461,):
        raise CompletePathEvidenceError("specimen candidate is not an FP64 461-vector")
    candidate_sha256 = hashlib.sha256(
        np.ascontiguousarray(candidate, dtype=np.dtype("<f8")).tobytes(order="C")
    ).hexdigest()
    if (
        candidate_sha256 != binding.candidate_sha256
        or specimen.get("parameter_sha256") != binding.candidate_sha256
    ):
        raise CompletePathEvidenceError("specimen candidate SHA identity mismatch")


@dataclass(frozen=True, slots=True)
class ProtocolSample:
    """One isolated child result under the existing optimization-window boundary."""

    profile_id: ProfileId
    phase: Phase
    sample_index: int | None
    optimization_wall_ns: int
    subprocess_wall_ns: int
    driver: str
    backend_mode: str
    input_fingerprint: str
    configuration_fingerprint: str
    effective_construction_fingerprint: str
    input_bundle_sha256: str
    source_sha256: str
    runtime_identity_sha256: str
    nit: int
    nfev: int
    njev: int
    endpoint_certificate: Mapping[str, object]
    parity_rows: tuple[measurement_runner.ParityRow, ...]
    snapshot_source_manifest_sha256: str
    snapshot_import_attestation_sha256: str
    snapshot_lane_identity_sha256: str
    provenance: Mapping[str, object]


RunExecutor = Callable[
    [
        measurement_runner.CollectionRun,
        int,
        Mapping[str, str],
        Path,
    ],
    ProtocolSample,
]


def build_complete_path_plan() -> tuple[measurement_runner.CollectionRun, ...]:
    """Return one fresh measured optimization run per matched lane."""

    return tuple(
        measurement_runner.CollectionRun(profile, "cold", None, position, True, False)
        for position, profile in enumerate(PROFILE_IDS)
    )


def build_complete_path_lane_environment(
    profile_id: ProfileId,
    bound_environment: Mapping[str, str],
    *,
    gpu_uuid: str,
    repo_root: Path,
) -> dict[str, str]:
    """Construct one complete-path launch environment from its bound runtime base."""

    environment = measurement_runner.build_measurement_environment(
        profile_id,
        allocation_sensitive=False,
        base_environment=bound_environment,
        gpu_index=0,
        repo_root=repo_root,
    )
    environment.update(
        {
            "SIMSOPT_BACKEND_MODE": EXPECTED_BACKENDS[profile_id],
            "SIMSOPT_BACKEND_STRICT": "1",
            "SIMSOPT_PRECISION": "fp64",
            "JAX_ENABLE_X64": "1",
        }
    )
    if profile_id == "native_cpu":
        environment.update(
            {
                "SIMSOPT_JAX_TRANSFER_GUARD": "log",
                "JAX_TRANSFER_GUARD": "allow",
                "JAX_PLATFORMS": "cpu",
            }
        )
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.pop(measurement_runner._EXACT_ADJOINT_ENVIRONMENT_NAME, None)
        environment.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        return environment
    environment.update(
        {
            "SIMSOPT_JAX_TRANSFER_GUARD": "log",
            "JAX_TRANSFER_GUARD": "log",
            "JAX_PLATFORMS": "cuda",
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            measurement_runner._EXACT_ADJOINT_ENVIRONMENT_NAME: "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        }
    )
    return environment


def _validate_sample(sample: ProtocolSample) -> None:
    if sample.profile_id not in PROFILE_IDS:
        raise CompletePathEvidenceError("sample has an unsupported profile_id")
    if sample.phase != "cold" or sample.sample_index is not None:
        raise CompletePathEvidenceError("sample must be one fresh measured run")
    if sample.optimization_wall_ns < 1 or sample.subprocess_wall_ns < 1:
        raise CompletePathEvidenceError("sample timings must be positive")
    if sample.optimization_wall_ns > sample.subprocess_wall_ns:
        raise CompletePathEvidenceError(
            "optimization-window timing exceeds subprocess timing"
        )
    if sample.driver != EXPECTED_DRIVERS[sample.profile_id]:
        raise CompletePathEvidenceError("sample driver does not match its lane")
    if sample.backend_mode != EXPECTED_BACKENDS[sample.profile_id]:
        raise CompletePathEvidenceError("sample backend_mode does not match its lane")
    for name in ("nit", "nfev", "njev"):
        value = getattr(sample, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CompletePathEvidenceError(f"sample {name} must be a nonnegative int")
    expected_certificate_keys = frozenset(
        {
            "success",
            "initial_stationary",
            "terminal_stationary",
            "constraints_satisfied",
            "outer_status",
        }
    )
    if frozenset(sample.endpoint_certificate) != expected_certificate_keys:
        raise CompletePathEvidenceError("sample endpoint certificate keys are invalid")
    if any(
        not isinstance(sample.endpoint_certificate[key], bool)
        for key in expected_certificate_keys - {"outer_status"}
    ) or (
        isinstance(sample.endpoint_certificate["outer_status"], bool)
        or not isinstance(sample.endpoint_certificate["outer_status"], int)
    ):
        raise CompletePathEvidenceError("sample endpoint certificate types are invalid")
    expected_observables = tuple(
        observable
        for observable, _ in measurement_runner._SINGLE_STAGE_PARITY_OBSERVABLES
    )
    if sample.profile_id == "native_cpu":
        if sample.parity_rows:
            raise CompletePathEvidenceError(
                "native sample must not contain parity rows"
            )
    elif tuple(row.observable for row in sample.parity_rows) != expected_observables:
        raise CompletePathEvidenceError(
            "JAX sample parity rows do not cover the frozen observables"
        )
    for row in sample.parity_rows:
        if (
            not all(
                math.isfinite(value)
                for value in (row.native_value, row.lane_value, row.tolerance)
            )
            or row.tolerance < 0.0
        ):
            raise CompletePathEvidenceError("sample parity row is invalid")
    _sha256(
        sample.snapshot_source_manifest_sha256,
        "snapshot_source_manifest_sha256",
    )
    _sha256(
        sample.snapshot_import_attestation_sha256,
        "snapshot_import_attestation_sha256",
    )
    _sha256(sample.snapshot_lane_identity_sha256, "snapshot_lane_identity_sha256")
    for name in ("input_bundle_sha256", "source_sha256", "runtime_identity_sha256"):
        _sha256(getattr(sample, name), name)
    if not all(
        (
            sample.input_fingerprint,
            sample.configuration_fingerprint,
            sample.effective_construction_fingerprint,
            sample.provenance,
        )
    ):
        raise CompletePathEvidenceError("sample identity or provenance is empty")
    for name in (
        "input_fingerprint",
        "configuration_fingerprint",
        "effective_construction_fingerprint",
    ):
        _sha256(getattr(sample, name), name)


def _staged_gap_budget_timing_input(
    *,
    warm_p50_ns: float,
    matched_complete_path_timings_ns: Mapping[str, int],
    c0_value_and_gradient_evaluation_count: int,
) -> dict[str, object]:
    _positive_number(warm_p50_ns, "warm_p50_ns")
    if (
        isinstance(c0_value_and_gradient_evaluation_count, bool)
        or not isinstance(c0_value_and_gradient_evaluation_count, int)
        or c0_value_and_gradient_evaluation_count < 1
    ):
        raise CompletePathEvidenceError("C0 value-and-gradient count must be positive")
    if frozenset(matched_complete_path_timings_ns) != frozenset(LANE_IDS.values()):
        raise CompletePathEvidenceError(
            "matched complete-path timing keys must be native, c0, and optax"
        )
    for lane_id, timing_ns in matched_complete_path_timings_ns.items():
        _positive_number(float(timing_ns), f"{lane_id} complete-path timing")
    return {
        "matched_complete_path_reference_timings_ns": dict(
            matched_complete_path_timings_ns
        ),
        "c0_complete_path_value_and_gradient_evaluation_count": (
            c0_value_and_gradient_evaluation_count
        ),
        "c0_complete_path_value_and_gradient_evaluation_count_semantics": (
            GAP_BUDGET_COUNT_SEMANTICS
        ),
    }


def build_staged_gap_budget_timing_input(
    complete_path_document: Mapping[str, object],
) -> dict[str, object]:
    """Build the timing portion of staged gap-budget inputs without transcription."""

    if complete_path_document.get("schema_id") != SCHEMA_ID:
        raise CompletePathEvidenceError("complete-path document schema mismatch")
    identity = complete_path_document.get("identity")
    matched = complete_path_document.get("matched_complete_path_reference_timings_ns")
    if not isinstance(identity, dict) or not isinstance(matched, dict):
        raise CompletePathEvidenceError(
            "complete-path document lacks identity or matched timings"
        )
    warm_p50_ns = identity.get("warm_p50_ns")
    lanes = complete_path_document.get("lanes")
    c0_lane = lanes.get("c0") if isinstance(lanes, dict) else None
    counts = c0_lane.get("optimizer_counts") if isinstance(c0_lane, dict) else None
    c0_nfev = counts.get("nfev") if isinstance(counts, dict) else None
    if (
        isinstance(warm_p50_ns, bool)
        or not isinstance(warm_p50_ns, (int, float))
        or isinstance(c0_nfev, bool)
        or not isinstance(c0_nfev, int)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in matched.values()
        )
    ):
        raise CompletePathEvidenceError("complete-path timing types are invalid")
    timing_input = _staged_gap_budget_timing_input(
        warm_p50_ns=float(warm_p50_ns),
        matched_complete_path_timings_ns=cast(Mapping[str, int], matched),
        c0_value_and_gradient_evaluation_count=c0_nfev,
    )
    staged = complete_path_document.get("staged_gap_budget_timing_input")
    if staged is not None and staged != timing_input:
        raise CompletePathEvidenceError(
            "staged gap-budget timing input disagrees with authoritative timings"
        )
    return timing_input


@dataclass(frozen=True, slots=True)
class PhaseReductionAssumption:
    conservative_reduction: float
    optimistic_reduction: float
    overlap_disposition: Literal["disjoint", "excluded_overlap"]

    def __post_init__(self) -> None:
        if self.overlap_disposition not in ("disjoint", "excluded_overlap"):
            raise CompletePathEvidenceError("overlap_disposition is invalid")
        for name in ("conservative_reduction", "optimistic_reduction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise CompletePathEvidenceError(f"{name} must be a finite fraction")
        if self.conservative_reduction > self.optimistic_reduction:
            raise CompletePathEvidenceError(
                "conservative reduction exceeds optimistic reduction"
            )


@dataclass(frozen=True, slots=True)
class FaithfulLever:
    lever_id: str
    disposition: Literal["bounded", "unbounded"]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not self.lever_id:
            raise CompletePathEvidenceError("lever_id must be non-empty")
        if self.disposition not in ("bounded", "unbounded"):
            raise CompletePathEvidenceError("lever disposition is invalid")
        _sha256(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True, slots=True)
class GapBudgetPolicyInput:
    phase_reduction_assumptions: Mapping[str, PhaseReductionAssumption]
    unattributed_conservative_reduction: float
    unattributed_optimistic_reduction: float
    faithful_levers: tuple[FaithfulLever, ...]

    def __post_init__(self) -> None:
        if not self.phase_reduction_assumptions or any(
            not phase_id for phase_id in self.phase_reduction_assumptions
        ):
            raise CompletePathEvidenceError("phase assumptions must be non-empty")
        for name in (
            "unattributed_conservative_reduction",
            "unattributed_optimistic_reduction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise CompletePathEvidenceError(f"{name} must be a finite fraction")
        if (
            self.unattributed_conservative_reduction
            > self.unattributed_optimistic_reduction
        ):
            raise CompletePathEvidenceError(
                "unattributed conservative reduction exceeds optimistic reduction"
            )
        lever_ids = tuple(lever.lever_id for lever in self.faithful_levers)
        if not lever_ids or len(lever_ids) != len(set(lever_ids)):
            raise CompletePathEvidenceError("faithful lever IDs must be unique")


def _gap_budget_identity(
    complete_path_document: Mapping[str, object],
) -> dict[str, object]:
    identity = complete_path_document.get("identity")
    if not isinstance(identity, dict):
        raise CompletePathEvidenceError("complete-path identity is missing")
    fields = (
        "candidate_sha256",
        "specimen_sha256",
        "source_sha256",
        "lane_id",
        "gpu_uuid",
        "runtime_identity_sha256",
        "gate_checkpoint_sha256",
        "warm_checkpoint_sha256",
        "warm_p50_ns",
    )
    result = {field: identity.get(field) for field in fields}
    for field in fields[:-1]:
        value = result[field]
        if not isinstance(value, str) or not value:
            raise CompletePathEvidenceError(
                f"complete-path identity {field} is invalid"
            )
    warm_p50_ns = result["warm_p50_ns"]
    if isinstance(warm_p50_ns, bool) or not isinstance(warm_p50_ns, (int, float)):
        raise CompletePathEvidenceError("complete-path warm_p50_ns is invalid")
    _positive_number(float(warm_p50_ns), "warm_p50_ns")
    return result


def build_gap_budget_inputs_artifact(
    complete_path_document: Mapping[str, object],
    policy: GapBudgetPolicyInput,
) -> dict[str, object]:
    """Wrap authoritative complete-path values with typed diagnostic policy."""

    timing_input = build_staged_gap_budget_timing_input(complete_path_document)
    payload = {
        **timing_input,
        "phase_reduction_assumptions": {
            phase_id: {
                "conservative_reduction": assumption.conservative_reduction,
                "optimistic_reduction": assumption.optimistic_reduction,
                "overlap_disposition": assumption.overlap_disposition,
            }
            for phase_id, assumption in policy.phase_reduction_assumptions.items()
        },
        "unattributed_conservative_reduction": (
            policy.unattributed_conservative_reduction
        ),
        "unattributed_optimistic_reduction": (policy.unattributed_optimistic_reduction),
        "faithful_levers": [
            {
                "lever_id": lever.lever_id,
                "disposition": lever.disposition,
                "evidence_sha256": lever.evidence_sha256,
            }
            for lever in policy.faithful_levers
        ],
    }
    return {
        "schema_id": GAP_BUDGET_INPUTS_SCHEMA_ID,
        "identity": _gap_budget_identity(complete_path_document),
        "gap_budget_inputs": payload,
    }


def validate_gap_budget_inputs_artifact(
    document: Mapping[str, object],
    complete_path_document: Mapping[str, object],
) -> Mapping[str, object]:
    """Reject a wrapped gap-input artifact that drifts from complete-path evidence."""

    if (
        frozenset(document) != frozenset({"schema_id", "identity", "gap_budget_inputs"})
        or document.get("schema_id") != GAP_BUDGET_INPUTS_SCHEMA_ID
    ):
        raise CompletePathEvidenceError("gap-budget input artifact schema is invalid")
    if document.get("identity") != _gap_budget_identity(complete_path_document):
        raise CompletePathEvidenceError("gap-budget input identity is not bound")
    payload = document.get("gap_budget_inputs")
    if not isinstance(payload, dict):
        raise CompletePathEvidenceError("gap_budget_inputs must be an object")
    expected_timing = build_staged_gap_budget_timing_input(complete_path_document)
    for key, expected in expected_timing.items():
        if payload.get(key) != expected:
            raise CompletePathEvidenceError(f"gap-budget input {key} drifted")
    expected_keys = frozenset(expected_timing) | frozenset(
        {
            "phase_reduction_assumptions",
            "unattributed_conservative_reduction",
            "unattributed_optimistic_reduction",
            "faithful_levers",
        }
    )
    if frozenset(payload) != expected_keys:
        raise CompletePathEvidenceError("gap-budget input payload keys are invalid")
    assumptions = payload["phase_reduction_assumptions"]
    if not isinstance(assumptions, dict) or not assumptions:
        raise CompletePathEvidenceError("phase reduction assumptions are invalid")
    for phase_id, assumption in assumptions.items():
        if (
            not isinstance(phase_id, str)
            or not phase_id
            or not isinstance(assumption, dict)
        ):
            raise CompletePathEvidenceError("phase reduction assumption is invalid")
        if frozenset(assumption) != frozenset(
            {
                "conservative_reduction",
                "optimistic_reduction",
                "overlap_disposition",
            }
        ):
            raise CompletePathEvidenceError(
                "phase reduction assumption keys are invalid"
            )
        if any(
            isinstance(assumption[key], bool)
            or not isinstance(assumption[key], (int, float))
            for key in ("conservative_reduction", "optimistic_reduction")
        ) or not isinstance(assumption["overlap_disposition"], str):
            raise CompletePathEvidenceError(
                "phase reduction assumption types are invalid"
            )
        PhaseReductionAssumption(
            conservative_reduction=float(assumption["conservative_reduction"]),
            optimistic_reduction=float(assumption["optimistic_reduction"]),
            overlap_disposition=cast(
                Literal["disjoint", "excluded_overlap"],
                assumption["overlap_disposition"],
            ),
        )
    for key in (
        "unattributed_conservative_reduction",
        "unattributed_optimistic_reduction",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompletePathEvidenceError(f"{key} is invalid")
    levers = payload["faithful_levers"]
    if not isinstance(levers, list) or not levers:
        raise CompletePathEvidenceError("faithful_levers is invalid")
    parsed_levers: list[FaithfulLever] = []
    for lever in levers:
        if not isinstance(lever, dict) or frozenset(lever) != frozenset(
            {"lever_id", "disposition", "evidence_sha256"}
        ):
            raise CompletePathEvidenceError("faithful lever is invalid")
        if any(
            not isinstance(lever[key], str)
            for key in ("lever_id", "disposition", "evidence_sha256")
        ):
            raise CompletePathEvidenceError("faithful lever types are invalid")
        parsed_levers.append(
            FaithfulLever(
                lever_id=str(lever["lever_id"]),
                disposition=cast(Literal["bounded", "unbounded"], lever["disposition"]),
                evidence_sha256=str(lever["evidence_sha256"]),
            )
        )
    GapBudgetPolicyInput(
        phase_reduction_assumptions={
            phase_id: PhaseReductionAssumption(
                conservative_reduction=float(assumption["conservative_reduction"]),
                optimistic_reduction=float(assumption["optimistic_reduction"]),
                overlap_disposition=cast(
                    Literal["disjoint", "excluded_overlap"],
                    assumption["overlap_disposition"],
                ),
            )
            for phase_id, assumption in assumptions.items()
        },
        unattributed_conservative_reduction=float(
            payload["unattributed_conservative_reduction"]
        ),
        unattributed_optimistic_reduction=float(
            payload["unattributed_optimistic_reduction"]
        ),
        faithful_levers=tuple(parsed_levers),
    )
    return payload


def build_complete_path_document(
    binding: CompletePathBinding,
    samples: Sequence[ProtocolSample],
) -> dict[str, object]:
    """Validate three matched samples and publish rough single-run references."""

    expected_plan = build_complete_path_plan()
    if len(samples) != len(expected_plan):
        raise CompletePathEvidenceError("sample count does not match the protocol plan")
    for expected, sample in zip(expected_plan, samples, strict=True):
        _validate_sample(sample)
        if (
            sample.profile_id != expected.profile_id
            or sample.phase != expected.phase
            or sample.sample_index != expected.sample_index
        ):
            raise CompletePathEvidenceError("sample ordering does not match the plan")

    first = samples[0]
    if (
        first.source_sha256 != binding.source_sha256
        or first.runtime_identity_sha256 != binding.runtime_identity_sha256
    ):
        raise CompletePathEvidenceError(
            "sample provenance does not match the complete-path binding"
        )
    for sample in samples[1:]:
        if (
            sample.input_fingerprint != first.input_fingerprint
            or sample.configuration_fingerprint != first.configuration_fingerprint
            or sample.effective_construction_fingerprint
            != first.effective_construction_fingerprint
            or sample.snapshot_source_manifest_sha256
            != first.snapshot_source_manifest_sha256
            or sample.snapshot_import_attestation_sha256
            != first.snapshot_import_attestation_sha256
            or sample.input_bundle_sha256 != first.input_bundle_sha256
            or sample.source_sha256 != binding.source_sha256
            or sample.runtime_identity_sha256 != binding.runtime_identity_sha256
        ):
            raise CompletePathEvidenceError(
                "native, C0, and Optax samples are not identity matched"
            )

    lanes: dict[str, object] = {}
    matched_single_runs: dict[str, int] = {}
    for profile_id in PROFILE_IDS:
        lane_samples = tuple(
            sample for sample in samples if sample.profile_id == profile_id
        )
        if len(lane_samples) != 1:
            raise CompletePathEvidenceError(
                f"{profile_id} does not contain exactly one measured run"
            )
        sample = lane_samples[0]
        lane_id = LANE_IDS[profile_id]
        matched_single_runs[lane_id] = sample.optimization_wall_ns
        lanes[lane_id] = {
            "profile_id": profile_id,
            "device": "cpu" if profile_id == "native_cpu" else "gpu",
            "gpu_uuid": None if profile_id == "native_cpu" else binding.gpu_uuid,
            "driver": EXPECTED_DRIVERS[profile_id],
            "backend_mode": EXPECTED_BACKENDS[profile_id],
            "optimizer_counts": {
                "nit": sample.nit,
                "nfev": sample.nfev,
                "njev": sample.njev,
            },
            "endpoint_certificate": dict(sample.endpoint_certificate),
            "parity_rows": [
                {
                    "observable": row.observable,
                    "native_value": row.native_value,
                    "lane_value": row.lane_value,
                    "tolerance": row.tolerance,
                }
                for row in sample.parity_rows
            ],
            "raw_optimization_wall_ns": [sample.optimization_wall_ns],
            "single_run_optimization_wall_ns": sample.optimization_wall_ns,
            "statistical_summary": "not_produced_single_sample",
            "compilation_context": (
                "native_not_applicable"
                if profile_id == "native_cpu"
                else "fresh_empty_persistent_cache_objective_gradient_may_compile"
            ),
            "samples": [
                {
                    "phase": sample.phase,
                    "sample_index": sample.sample_index,
                    "optimization_wall_ns": sample.optimization_wall_ns,
                    "subprocess_wall_ns": sample.subprocess_wall_ns,
                    "optimizer_counts": {
                        "nit": sample.nit,
                        "nfev": sample.nfev,
                        "njev": sample.njev,
                    },
                    "provenance_binding": {
                        "source_sha256": sample.source_sha256,
                        "runtime_identity_sha256": sample.runtime_identity_sha256,
                        "snapshot_source_manifest_sha256": (
                            sample.snapshot_source_manifest_sha256
                        ),
                        "snapshot_import_attestation_sha256": (
                            sample.snapshot_import_attestation_sha256
                        ),
                        "snapshot_lane_identity_sha256": (
                            sample.snapshot_lane_identity_sha256
                        ),
                    },
                    "provenance": dict(sample.provenance),
                }
                for sample in lane_samples
            ],
        }

    staged_gap_budget_timing_input = _staged_gap_budget_timing_input(
        warm_p50_ns=binding.warm_p50_ns,
        matched_complete_path_timings_ns=matched_single_runs,
        c0_value_and_gradient_evaluation_count=next(
            sample.nfev for sample in samples if sample.profile_id == "jax_gpu_fast"
        ),
    )
    return {
        "schema_id": SCHEMA_ID,
        "identity": {
            "specimen_sha256": binding.specimen_sha256,
            "candidate_sha256": binding.candidate_sha256,
            "source_sha256": binding.source_sha256,
            "runtime_identity_sha256": binding.runtime_identity_sha256,
            "native_reference_sha256": binding.native_reference_sha256,
            "gate_checkpoint_sha256": binding.gate_checkpoint_sha256,
            "warm_checkpoint_sha256": binding.warm_checkpoint_sha256,
            "warm_p50_ns": binding.warm_p50_ns,
            "lane_id": binding.lane_id,
            "gpu_uuid": binding.gpu_uuid,
            "input_fingerprint": first.input_fingerprint,
            "input_bundle_sha256": first.input_bundle_sha256,
            "configuration_fingerprint": first.configuration_fingerprint,
            "effective_construction_fingerprint": (
                first.effective_construction_fingerprint
            ),
            "snapshot_source_manifest_sha256": (first.snapshot_source_manifest_sha256),
            "snapshot_import_attestation_sha256": (
                first.snapshot_import_attestation_sha256
            ),
        },
        "protocol": {
            "boundary": (
                "required initial objective-and-gradient evaluation through "
                "outer optimizer return"
            ),
            "problem_construction_excluded": True,
            "initial_baseline_boozer_solve_excluded": True,
            "fresh_process_per_lane": True,
            "sample_count_per_lane": 1,
            "timing_claim": "rough_non_statistical_single_run",
            "candidate_warm_sample_requirement_applies": False,
            "closed_r5_receipt_extended": False,
        },
        "matched_complete_path_reference_timings_ns": matched_single_runs,
        "staged_gap_budget_timing_input": staged_gap_budget_timing_input,
        "lanes": lanes,
    }


def _nanoseconds(seconds: float, context: str) -> int:
    _positive_number(seconds, context)
    nanoseconds = round(seconds * 1_000_000_000)
    if nanoseconds < 1:
        raise CompletePathEvidenceError(f"{context} rounds below one nanosecond")
    return nanoseconds


def build_lane_snapshot_provenance_document(
    *,
    binding: CompletePathBinding,
    profile_id: ProfileId,
    snapshot_publication_path: Path,
    snapshot_manifest_path: Path,
    import_attestation_path: Path,
    runner_spec_path: Path,
    runtime_provenance_path: Path,
    device_probe_path: Path,
) -> dict[str, object]:
    """Build static lane identity solely from validated producer artifacts."""

    if snapshot_manifest_path.name != MANIFEST_FILENAME:
        raise CompletePathEvidenceError(
            f"snapshot manifest must be named {MANIFEST_FILENAME}"
        )
    try:
        manifest_entries, manifest_sha256 = load_snapshot_manifest(
            snapshot_manifest_path.parent
        )
    except (OSError, SnapshotError) as error:
        raise CompletePathEvidenceError(
            "immutable snapshot manifest is invalid"
        ) from error
    if snapshot_manifest_path.resolve() != (
        snapshot_manifest_path.parent.resolve() / MANIFEST_FILENAME
    ):
        raise CompletePathEvidenceError("snapshot manifest path is not canonical")

    try:
        attestation_bytes = import_attestation_path.read_bytes()
        attestation = _json_mapping(import_attestation_path, "import attestation")
    except OSError as error:
        raise CompletePathEvidenceError("import attestation is unreadable") from error
    if attestation_bytes != snapshot_canonical_json_bytes(attestation):
        raise CompletePathEvidenceError("import attestation bytes are not canonical")
    if (
        frozenset(attestation)
        != frozenset(
            {
                "schema_id",
                "state",
                "snapshot_manifest_sha256",
                "interpreter_path",
                "python_version",
                "bindings",
            }
        )
        or attestation.get("schema_id") != IMPORT_ATTESTATION_SCHEMA_ID
    ):
        raise CompletePathEvidenceError("import attestation schema is invalid")
    if (
        attestation.get("state") != "pass"
        or attestation.get("snapshot_manifest_sha256") != manifest_sha256
    ):
        raise CompletePathEvidenceError(
            "import attestation does not bind the validated manifest"
        )
    manifest_by_path = {entry.relative_path: entry for entry in manifest_entries}
    raw_bindings = attestation.get("bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise CompletePathEvidenceError("import attestation bindings are invalid")
    modules: list[str] = []
    for index, raw_binding in enumerate(raw_bindings):
        if not isinstance(raw_binding, dict) or frozenset(raw_binding) != frozenset(
            {"module", "relative_path", "size_bytes", "sha256"}
        ):
            raise CompletePathEvidenceError(
                f"import attestation binding {index} is invalid"
            )
        module = raw_binding.get("module")
        relative_path = raw_binding.get("relative_path")
        size_bytes = raw_binding.get("size_bytes")
        digest = raw_binding.get("sha256")
        if (
            not isinstance(module, str)
            or not module
            or not isinstance(relative_path, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(digest, str)
        ):
            raise CompletePathEvidenceError(
                f"import attestation binding {index} types are invalid"
            )
        _sha256(digest, f"import attestation binding {index} sha256")
        entry = manifest_by_path.get(relative_path)
        if entry is None or (entry.size_bytes, entry.sha256) != (size_bytes, digest):
            raise CompletePathEvidenceError(
                f"import attestation binding {index} is absent from the manifest"
            )
        modules.append(module)
    if (
        frozenset(modules)
        != frozenset({"simsopt", "simsopt_jax", "simsopt_jax_adapters", "simsoptpp"})
        or len(modules) != 4
    ):
        raise CompletePathEvidenceError(
            "import attestation must bind each required production module once"
        )

    evidence_paths = {
        "publication": snapshot_publication_path,
        "manifest": snapshot_manifest_path,
        "import_attestation": import_attestation_path,
        "runner_spec": runner_spec_path,
        "runtime_provenance": runtime_provenance_path,
        "device_probe": device_probe_path,
    }
    evidence_documents: dict[str, Mapping[str, object]] = {}
    evidence: dict[str, dict[str, str]] = {}
    for name, path in evidence_paths.items():
        document = _json_mapping(path, f"snapshot {name}")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise CompletePathEvidenceError(
                f"snapshot {name} evidence is unreadable"
            ) from error
        if payload != snapshot_canonical_json_bytes(document):
            raise CompletePathEvidenceError(
                f"snapshot {name} evidence is not canonical"
            )
        evidence_documents[name] = document
        evidence[name] = {
            "path": str(path.absolute()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    publication = evidence_documents["publication"]
    worktree = publication.get("worktree")
    if (
        publication.get("schema_id")
        != "single-stage-compute-graph-snapshot-publication-v1"
        or publication.get("snapshot_root")
        != str(snapshot_manifest_path.parent.absolute())
        or publication.get("snapshot_manifest_sha256") != manifest_sha256
        or not isinstance(worktree, dict)
        or worktree.get("source_state_sha256") != binding.source_sha256
    ):
        raise CompletePathEvidenceError("snapshot publication binding is invalid")
    runner_spec = evidence_documents["runner_spec"]
    runtime_provenance = evidence_documents["runtime_provenance"]
    device_probe = evidence_documents["device_probe"]
    allocation = runtime_provenance.get("allocation")
    bound_environment = runtime_provenance.get("environment")
    probe_gpu = device_probe.get("gpu")
    if (
        set(runner_spec)
        != {
            "schema_id",
            "lane_id",
            "warm_sample_count",
            "output_root",
            "input_root",
            "candidate_path",
            "native_reference_path",
            "provenance",
            "receipt_template",
        }
        or runner_spec.get("schema_id")
        != "single-stage-compute-graph-c0-runner-spec-v3"
        or runner_spec.get("provenance") != runtime_provenance
    ):
        raise CompletePathEvidenceError("runner spec runtime provenance mismatch")
    if runner_spec.get("lane_id") != binding.lane_id:
        raise CompletePathEvidenceError("runner spec lane identity mismatch")
    if (
        runtime_provenance.get("source_state_sha256") != binding.source_sha256
        or not isinstance(allocation, dict)
        or allocation.get("gpu_uuid") != binding.gpu_uuid
        or not isinstance(bound_environment, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in bound_environment.items()
        )
    ):
        raise CompletePathEvidenceError("runtime provenance binding is invalid")
    if (
        set(device_probe)
        != {
            "schema_id",
            "lane_id",
            "source_state_sha256",
            "runtime_identity_sha256",
            "qualification_sha256",
            "gpu",
            "native_binary",
        }
        or device_probe.get("schema_id") != "single-stage-compute-graph-device-probe-v1"
        or device_probe.get("lane_id") != binding.lane_id
        or device_probe.get("source_state_sha256") != binding.source_sha256
        or device_probe.get("runtime_identity_sha256")
        != binding.runtime_identity_sha256
        or not isinstance(probe_gpu, dict)
        or probe_gpu.get("uuid") != binding.gpu_uuid
    ):
        raise CompletePathEvidenceError("device probe binding is invalid")
    lane = "native-cpu" if profile_id == "native_cpu" else "jax-gpu"
    platform = "cpu" if profile_id == "native_cpu" else "gpu"
    static_environment = normalize_snapshot_lane_environment(
        build_complete_path_lane_environment(
            profile_id,
            cast(Mapping[str, str], bound_environment),
            gpu_uuid=binding.gpu_uuid,
            repo_root=snapshot_manifest_path.parent,
        )
    )
    return {
        "schema_id": LANE_SNAPSHOT_PROVENANCE_SCHEMA_ID,
        "profile_id": profile_id,
        "lane": lane,
        "backend_mode": EXPECTED_BACKENDS[profile_id],
        "driver": EXPECTED_DRIVERS[profile_id],
        "execution_platform": platform,
        "runtime_identity_sha256": binding.runtime_identity_sha256,
        "source_sha256": binding.source_sha256,
        "gpu_uuid": binding.gpu_uuid,
        "snapshot_root": str(snapshot_manifest_path.parent.absolute()),
        "static_environment": static_environment,
        "evidence": evidence,
    }


def write_lane_snapshot_provenance(
    artifact_root: Path,
    relative_path: str,
    *,
    binding: CompletePathBinding,
    profile_id: ProfileId,
    snapshot_publication_path: Path,
    snapshot_manifest_path: Path,
    import_attestation_path: Path,
    runner_spec_path: Path,
    runtime_provenance_path: Path,
    device_probe_path: Path,
) -> Path:
    """Write and self-validate one exclusive static lane identity."""

    document = build_lane_snapshot_provenance_document(
        binding=binding,
        profile_id=profile_id,
        snapshot_publication_path=snapshot_publication_path,
        snapshot_manifest_path=snapshot_manifest_path,
        import_attestation_path=import_attestation_path,
        runner_spec_path=runner_spec_path,
        runtime_provenance_path=runtime_provenance_path,
        device_probe_path=device_probe_path,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_bytes_exclusive(artifact_root, relative_path, canonical_json_bytes(document))
    output = artifact_root / relative_path
    identity = load_snapshot_lane_identity(output)
    if identity.profile_id != profile_id:
        raise CompletePathEvidenceError(
            "written snapshot lane identity changed profile"
        )
    return output


def write_lane_snapshot_provenance_set(
    output_root: Path,
    *,
    binding: CompletePathBinding,
    snapshot_publication_path: Path,
    snapshot_manifest_path: Path,
    import_attestation_path: Path,
    runner_spec_path: Path,
    runtime_provenance_path: Path,
    device_probe_path: Path,
) -> Mapping[ProfileId, Path]:
    """Publish all three static lane identities as one exclusive artifact set."""

    if output_root.exists():
        raise CompletePathEvidenceError("snapshot provenance output already exists")
    staging = output_root.parent / f".{output_root.name}.partial"
    if staging.exists():
        raise CompletePathEvidenceError("snapshot provenance staging already exists")
    documents = {
        profile_id: build_lane_snapshot_provenance_document(
            binding=binding,
            profile_id=profile_id,
            snapshot_publication_path=snapshot_publication_path,
            snapshot_manifest_path=snapshot_manifest_path,
            import_attestation_path=import_attestation_path,
            runner_spec_path=runner_spec_path,
            runtime_provenance_path=runtime_provenance_path,
            device_probe_path=device_probe_path,
        )
        for profile_id in PROFILE_IDS
    }
    staging.mkdir(parents=True)
    relative_paths: dict[ProfileId, str] = {
        profile_id: f"{LANE_IDS[profile_id]}.json" for profile_id in PROFILE_IDS
    }
    for profile_id in PROFILE_IDS:
        write_bytes_exclusive(
            staging,
            relative_paths[profile_id],
            canonical_json_bytes(documents[profile_id]),
        )
        identity = load_snapshot_lane_identity(staging / relative_paths[profile_id])
        if identity.profile_id != profile_id:
            raise CompletePathEvidenceError(
                "written snapshot provenance set changed lane profile"
            )
    staging.rename(output_root)
    return {
        profile_id: output_root / relative_paths[profile_id]
        for profile_id in PROFILE_IDS
    }


def _snapshot_provenance_identity(
    path: Path,
    *,
    profile_id: ProfileId,
    runtime_identity_sha256: str,
    source_sha256: str,
    gpu_uuid: str,
    snapshot_root: Path,
) -> tuple[str, str, str, Mapping[str, str], Mapping[str, str]]:
    document = _json_mapping(path, "immutable snapshot lane provenance")
    try:
        identity = load_snapshot_lane_identity(path)
    except ValueError as error:
        raise CompletePathEvidenceError(
            "immutable snapshot lane provenance is invalid"
        ) from error
    if identity.profile_id != profile_id:
        raise CompletePathEvidenceError(
            "immutable snapshot lane provenance profile mismatch"
        )
    if identity.runtime_identity_sha256 != runtime_identity_sha256:
        raise CompletePathEvidenceError(
            "immutable snapshot lane provenance runtime identity mismatch"
        )
    if identity.source_sha256 != source_sha256:
        raise CompletePathEvidenceError(
            "immutable snapshot lane provenance source identity mismatch"
        )
    if identity.gpu_uuid != gpu_uuid:
        raise CompletePathEvidenceError(
            "immutable snapshot lane provenance device identity mismatch"
        )
    if identity.snapshot_root.resolve() != snapshot_root.resolve():
        raise CompletePathEvidenceError(
            "immutable snapshot lane provenance root mismatch"
        )
    evidence = document.get("evidence")
    if not isinstance(evidence, dict):
        raise CompletePathEvidenceError("immutable snapshot evidence is invalid")
    digests: list[str] = []
    for name in ("manifest", "import_attestation"):
        reference = evidence.get(name)
        if not isinstance(reference, dict):
            raise CompletePathEvidenceError("immutable snapshot evidence is invalid")
        digests.append(
            _sha256(
                _required_string(reference, "sha256", f"snapshot {name}"),
                f"snapshot {name} SHA",
            )
        )
    return (
        digests[0],
        digests[1],
        _sha256_path(path),
        identity.static_environment,
        identity.bound_environment,
    )


def collect_complete_path_evidence(
    *,
    artifact_root: Path,
    specimen_document_path: Path,
    input_bundle_path: Path,
    candidate_path: Path,
    binding: CompletePathBinding,
    python_executable: str,
    isolated_site: bool = False,
    repo_root: Path = measurement_runner._REPO_ROOT,
    base_environment: Mapping[str, str] = os.environ,
    gpu_index: int = 0,
    poll_interval_seconds: float = measurement_runner._DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: float = measurement_runner._DEFAULT_TIMEOUT_SECONDS,
    immutable_snapshot_provenance_paths: Mapping[ProfileId, Path] | None = None,
    executor: RunExecutor | None = None,
) -> Path:
    """Run the existing isolated protocol and publish a separate Phase 0 artifact."""

    if artifact_root.exists():
        raise CompletePathEvidenceError("artifact_root must not already exist")
    _validate_specimen_binding(
        binding,
        specimen_document_path,
        input_bundle_path,
        candidate_path,
    )
    if not input_bundle_path.is_file():
        raise CompletePathEvidenceError("input_bundle_path must be an existing file")
    if input_bundle_path.name != "input_bundle.json":
        raise CompletePathEvidenceError("input_bundle_path must name input_bundle.json")
    bundle, _arrays = read_input_bundle(input_bundle_path.parent)
    if (
        bundle.case_id != measurement_runner._SINGLE_STAGE_CASE_ID
        or bundle.scale != "native_default"
    ):
        raise CompletePathEvidenceError(
            "input bundle must be the native-default single-stage case"
        )
    iteration_budget_value = bundle.configuration.get("outer_maxiter")
    if (
        isinstance(iteration_budget_value, bool)
        or not isinstance(iteration_budget_value, int)
        or iteration_budget_value < 1
    ):
        raise CompletePathEvidenceError("input bundle has no valid iteration budget")
    iteration_budget = iteration_budget_value
    if executor is None:
        if immutable_snapshot_provenance_paths is None or frozenset(
            immutable_snapshot_provenance_paths
        ) != frozenset(PROFILE_IDS):
            raise CompletePathEvidenceError(
                "one immutable snapshot provenance path is required per lane"
            )
        measurement_runner._gpu_concurrent_use_preflight(gpu_index)
        _gpu_name, observed_uuid, _driver, _cuda = measurement_runner._gpu_identity(
            gpu_index
        )
        if observed_uuid != binding.gpu_uuid:
            raise CompletePathEvidenceError("qualified GPU UUID does not match binding")

    snapshot_identities: dict[
        ProfileId,
        tuple[str, str, str, Mapping[str, str], Mapping[str, str]],
    ] = {}
    if immutable_snapshot_provenance_paths is not None:
        for profile_id in PROFILE_IDS:
            path = immutable_snapshot_provenance_paths.get(profile_id)
            if path is None:
                raise CompletePathEvidenceError(
                    f"immutable snapshot provenance is missing for {profile_id}"
                )
            snapshot_identities[profile_id] = _snapshot_provenance_identity(
                path,
                profile_id=profile_id,
                runtime_identity_sha256=binding.runtime_identity_sha256,
                source_sha256=binding.source_sha256,
                gpu_uuid=binding.gpu_uuid,
                snapshot_root=repo_root,
            )
    workspace = artifact_root.parent / f".{artifact_root.name}.partial"
    if workspace.exists():
        raise CompletePathEvidenceError("partial artifact workspace already exists")
    workspace.mkdir(parents=True)
    environments: dict[ProfileId, dict[str, str]] = {}
    for profile_id in PROFILE_IDS:
        bound_environment = (
            snapshot_identities[profile_id][4]
            if profile_id in snapshot_identities
            else base_environment
        )
        environment = build_complete_path_lane_environment(
            profile_id,
            bound_environment,
            gpu_uuid=binding.gpu_uuid,
            repo_root=repo_root,
        )
        environments[profile_id] = environment
        if profile_id in snapshot_identities:
            expected_environment = snapshot_identities[profile_id][3]
            if normalize_snapshot_lane_environment(environments[profile_id]) != dict(
                expected_environment
            ):
                raise CompletePathEvidenceError(
                    "complete-path launch environment differs from snapshot identity"
                )
    protocol_samples: list[ProtocolSample] = []
    for profile_id in PROFILE_IDS:
        if profile_id != "native_cpu":
            cache_directory = workspace / "caches" / profile_id
            cache_directory.mkdir(parents=True)
            environments[profile_id]["JAX_COMPILATION_CACHE_DIR"] = str(
                cache_directory.resolve()
            )
            environments[profile_id]["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
            environments[profile_id]["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    observations: dict[ProfileId, measurement_runner.LaneObservation] = {}
    measured_runs: list[
        tuple[
            ProfileId,
            Phase,
            measurement_runner.CollectionRun,
            measurement_runner.MonitoredCommandResult,
            measurement_runner.LaneObservation,
            float,
        ]
    ] = []
    for sequence_index, run in enumerate(build_complete_path_plan()):
        if run.profile_id not in PROFILE_IDS or run.phase == "allocation_memory":
            raise CompletePathEvidenceError("protocol plan contains an invalid run")
        profile_id = cast(ProfileId, run.profile_id)
        phase = cast(Phase, run.phase)
        if executor is not None:
            sample = executor(
                run,
                sequence_index,
                environments[profile_id],
                workspace,
            )
            protocol_samples.append(sample)
            continue
        monitored, observation, trajectory_path, optimization_seconds = (
            measurement_runner._execute_single_stage_speed_run(
                bundle_path=input_bundle_path,
                workspace=workspace,
                run=run,
                sequence_index=sequence_index,
                environment=environments[profile_id],
                python_executable=python_executable,
                isolated_site=isolated_site,
                repo_root=repo_root,
                gpu_index=gpu_index,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
                immutable_snapshot_provenance_path=(
                    immutable_snapshot_provenance_paths[profile_id]
                    if immutable_snapshot_provenance_paths is not None
                    else None
                ),
            )
        )
        measurement_runner._validate_single_stage_campaign_observation(
            observation,
            profile_id=profile_id,
            input_fingerprint=bundle.input_fingerprint,
        )
        observations[profile_id] = observation
        trajectory = measurement_runner._single_stage_trajectory(trajectory_path)
        measurement_runner._validate_single_stage_trajectory_count(
            profile_id=profile_id,
            phase=phase,
            trajectory=trajectory,
            observation=observation,
            iteration_budget=iteration_budget,
        )
        measured_runs.append(
            (profile_id, phase, run, monitored, observation, optimization_seconds)
        )

    if executor is None:
        if frozenset(observations) != frozenset(PROFILE_IDS):
            raise CompletePathEvidenceError("complete-path observations are incomplete")
        measurement_runner._validate_single_stage_campaign_identity(
            cast(
                Mapping[
                    measurement_runner.RunnerProfileId,
                    measurement_runner.LaneObservation,
                ],
                observations,
            )
        )
        for (
            profile_id,
            phase,
            run,
            monitored,
            observation,
            optimization_seconds,
        ) in measured_runs:
            endpoint_audit = measurement_runner._single_stage_endpoint_audit(
                profile_id, observation
            )
            _validate_endpoint_audit(endpoint_audit)
            parity_rows = measurement_runner._single_stage_campaign_parity_rows(
                profile_id=profile_id,
                observations=cast(
                    Mapping[
                        measurement_runner.RunnerProfileId,
                        measurement_runner.LaneObservation,
                    ],
                    observations,
                ),
            )
            provenance = observation.provenance
            if provenance is None:
                raise CompletePathEvidenceError("observation omitted provenance")
            certificate = endpoint_audit.certificate
            (
                source_manifest_sha256,
                import_attestation_sha256,
                lane_identity_sha256,
                _static_environment,
                _bound_environment,
            ) = snapshot_identities[profile_id]
            protocol_samples.append(
                ProtocolSample(
                    profile_id=profile_id,
                    phase=phase,
                    sample_index=run.sample_index,
                    optimization_wall_ns=_nanoseconds(
                        optimization_seconds, "optimization_wall_seconds"
                    ),
                    subprocess_wall_ns=_nanoseconds(
                        monitored.wall_seconds, "subprocess_wall_seconds"
                    ),
                    driver=observation.driver,
                    backend_mode=observation.backend_mode,
                    input_fingerprint=observation.input_fingerprint,
                    configuration_fingerprint=observation.configuration_fingerprint,
                    effective_construction_fingerprint=(
                        observation.effective_construction_fingerprint
                    ),
                    input_bundle_sha256=_sha256_path(input_bundle_path),
                    source_sha256=binding.source_sha256,
                    runtime_identity_sha256=binding.runtime_identity_sha256,
                    nit=endpoint_audit.nit,
                    nfev=endpoint_audit.nfev,
                    njev=endpoint_audit.njev,
                    endpoint_certificate={
                        "success": certificate.success,
                        "initial_stationary": certificate.initial_stationary,
                        "terminal_stationary": certificate.terminal_stationary,
                        "constraints_satisfied": certificate.constraints_satisfied,
                        "outer_status": certificate.outer_status,
                    },
                    parity_rows=parity_rows,
                    snapshot_source_manifest_sha256=source_manifest_sha256,
                    snapshot_import_attestation_sha256=import_attestation_sha256,
                    snapshot_lane_identity_sha256=lane_identity_sha256,
                    provenance=lane_provenance_payload(provenance),
                )
            )

    input_bundle_sha256 = _sha256_path(input_bundle_path)
    if any(
        sample.input_bundle_sha256 != input_bundle_sha256
        or sample.input_fingerprint != bundle.input_fingerprint
        or sample.configuration_fingerprint != bundle.configuration_fingerprint
        for sample in protocol_samples
    ):
        raise CompletePathEvidenceError(
            "sample input identity does not match the immutable input bundle"
        )
    document = build_complete_path_document(binding, protocol_samples)
    write_bytes_exclusive(workspace, DOCUMENT_PATH, canonical_json_bytes(document))
    workspace.rename(artifact_root)
    return artifact_root / DOCUMENT_PATH


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser(
        "snapshot-provenance",
        help="produce native/C0/Optax immutable static identity documents",
    )
    snapshot.add_argument("--gate-checkpoint", type=Path, required=True)
    snapshot.add_argument("--warm-checkpoint", type=Path, required=True)
    snapshot.add_argument("--native-reference", type=Path, required=True)
    snapshot.add_argument("--snapshot-publication", type=Path, required=True)
    snapshot.add_argument("--snapshot-manifest", type=Path, required=True)
    snapshot.add_argument("--import-attestation", type=Path, required=True)
    snapshot.add_argument("--runner-spec", type=Path, required=True)
    snapshot.add_argument("--runtime-provenance", type=Path, required=True)
    snapshot.add_argument("--device-probe", type=Path, required=True)
    snapshot.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    if options.command != "snapshot-provenance":
        raise CompletePathEvidenceError("unsupported complete-path command")
    binding = binding_from_phase0_checkpoints(
        options.gate_checkpoint,
        options.warm_checkpoint,
        options.native_reference,
    )
    paths = write_lane_snapshot_provenance_set(
        options.output_root,
        binding=binding,
        snapshot_publication_path=options.snapshot_publication,
        snapshot_manifest_path=options.snapshot_manifest,
        import_attestation_path=options.import_attestation,
        runner_spec_path=options.runner_spec,
        runtime_provenance_path=options.runtime_provenance,
        device_probe_path=options.device_probe,
    )
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {profile_id: str(paths[profile_id]) for profile_id in PROFILE_IDS}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
