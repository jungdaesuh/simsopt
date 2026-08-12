"""Independent fail-closed receipt for the NEQ-GNTR1-DIAG1 replay.

The producer supplies only canonical raw-evidence artifacts.  This module owns
their wire schemas, resolves their bytes, and derives every diagnostic outcome.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Final, Iterable, Mapping, Self

import numpy as np
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256

from benchmarks.single_stage_fullspace_snapshot import (
    DIAG4_GPU_SNAPSHOT_ROLES,
    DIAG5_GPU_SNAPSHOT_ROLES,
    RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ArtifactRef,
    JsonValue,
    SnapshotIdentity,
    SnapshotPublication,
    canonical_json_bytes,
    load_canonical_json_bytes,
    load_snapshot,
    validate_diag5_runtime_evidence_v2_bytes,
    validate_runtime_evidence,
    validate_runtime_evidence_v2,
)
from benchmarks.single_stage_native_equivalent_endpoint_audit import (
    NativeEquivalentEndpointAudit,
    endpoint_audit_from_payload,
    endpoint_audit_payload,
)
from benchmarks.single_stage_native_equivalent_reference import (
    SCHEMA_VERSION as NATIVE_REFERENCE_SCHEMA_VERSION,
)
from benchmarks.single_stage_native_equivalent_reference import (
    validate_native_equivalent_reference,
)

PLAN_SHA256: Final = "e6871072a7011d64e511aa8e8cf7db17d36acedbb33dbbce22b18cd0ae2c6d59"
ROUTE: Final = "NEQ-GNTR1-DIAG1"
NUMERICAL_ROUTE: Final = "NEQ-GNTR1"
SCHEMA_VERSION: Final = "single-stage-neq-gntr1-no-hit-diagnostic-v1"
MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-no-hit-diagnostic-artifact-manifest-v1"
)
RECEIPT_FILENAME: Final = "diagnostic.json"
MANIFEST_FILENAME: Final = "artifact-manifest.json"
FIXED_ARTIFACT_ROLES: Final = {
    RECEIPT_FILENAME: "diagnostic_receipt",
    "execution.json": "execution_evidence",
    "preflight/producer.json": "preflight_producer",
    "preflight/terminal.json": "preflight_terminal",
    "preflight/process.json": "preflight_process",
    "preflight/stdout.bin": "preflight_stdout",
    "preflight/stderr.bin": "preflight_stderr",
    "preflight/gpu-memory.json": "preflight_memory",
    "preflight/gpu-memory-samples.json": "preflight_memory_samples",
    "preflight/runtime-evidence.json": "preflight_runtime",
    "preflight/policy.json": "preflight_policy",
    "policy-authority.json": "policy_authority",
    "cold/producer.json": "cold_producer",
    "cold/terminal.json": "cold_terminal",
    "cold/process.json": "cold_process",
    "cold/stdout.bin": "cold_stdout",
    "cold/stderr.bin": "cold_stderr",
    "cold/gpu-memory.json": "cold_memory",
    "cold/gpu-memory-samples.json": "cold_memory_samples",
    "cold/runtime-evidence.json": "cold_runtime",
    "cold/history.json": "history",
    "cold/terminal-numerical.json": "terminal_numerical",
    "cold/policy.json": "policy",
    "cold/trace-intervals.json": "trace_intervals",
    "source-snapshot/source-manifest.json": "source_snapshot",
    "native-reference/reference.json": "native_reference",
}
EVIDENCE_ROLE_PATHS: Final = {
    "history": "cold/history.json",
    "terminal_numerical": "cold/terminal-numerical.json",
    "trace_intervals": "cold/trace-intervals.json",
    "execution": "execution.json",
    "preflight": "preflight/producer.json",
    "preflight_child_terminal": "preflight/terminal.json",
    "preflight_memory": "preflight/gpu-memory.json",
    "preflight_memory_samples": "preflight/gpu-memory-samples.json",
    "preflight_process": "preflight/process.json",
    "preflight_runtime": "preflight/runtime-evidence.json",
    "preflight_policy": "preflight/policy.json",
    "policy_authority": "policy-authority.json",
    "producer": "cold/producer.json",
    "child_terminal": "cold/terminal.json",
    "runtime": "cold/runtime-evidence.json",
    "process": "cold/process.json",
    "memory": "cold/gpu-memory.json",
    "memory_samples": "cold/gpu-memory-samples.json",
    "source_manifest": "source-snapshot/source-manifest.json",
    "native_reference": "native-reference/reference.json",
    "policy": "cold/policy.json",
}
MAXIMUM_ATTEMPTS: Final = 300
MAXIMUM_ACCEPTED_STEPS: Final = 256
STATE_SIZE: Final = 716
EQUALITY_SIZE: Final = 255
LEDGER_SIZE: Final = 257
OBJECTIVE_MAXIMUM: Final = 4.4822246533126125e-08
FEASIBILITY_MAXIMUM: Final = 1.0e-10
RAW_EQUALITY_ABSOLUTE_TOLERANCE: Final = 1.0e-12
RAW_EQUALITY_RELATIVE_TOLERANCE: Final = 1.0e-10
RESIDUAL_VALUE_DEFECT_MAXIMUM: Final = 1.0e-12
RESIDUAL_GRADIENT_DEFECT_MAXIMUM: Final = 1.0e-10
TRANSPOSE_DEFECT_MAXIMUM: Final = 1.0e-10
GPU_UUID: Final = "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
MINIMUM_PHASE_COVERAGE: Final = 0.90
MODEL_REUSE_SELECTION_MINIMUM: Final = 0.05
RETRACTION_RETRY_FRACTION_MINIMUM: Final = 0.10

FROZEN_GNTR_OPTIONS: Final = {
    "corrected_feasibility_tolerance": 1.0e-10,
    "forward_error_tolerance": 1.0e-7,
    "initial_trust_radius": 2.0**-10,
    "linear_residual_tolerance": 1.0e-10,
    "maximum_accepted_steps": 256,
    "maximum_attempts": 300,
    "maximum_corrected_radius_excess": 1.0e-6,
    "maximum_correction_step_ratio": 1.0e-3,
    "maximum_steihaug_iterations": 32,
    "maximum_trust_radius": 2.0**-4,
    "mechanism_rotation_threshold": 1.0e-3,
    "minimum_trust_radius": 2.0**-20,
    "normalized_curvature_tolerance": 1.0e-10,
    "projected_residual_tolerance": 1.0e-10,
    "residual_gradient_defect_tolerance": 1.0e-10,
    "residual_value_defect_tolerance": 1.0e-12,
}
GNTR_OPTION_ORDER: Final = (
    "maximum_accepted_steps",
    "maximum_attempts",
    "initial_trust_radius",
    "minimum_trust_radius",
    "maximum_trust_radius",
    "maximum_steihaug_iterations",
    "projected_residual_tolerance",
    "linear_residual_tolerance",
    "corrected_feasibility_tolerance",
    "forward_error_tolerance",
    "residual_value_defect_tolerance",
    "residual_gradient_defect_tolerance",
    "normalized_curvature_tolerance",
    "maximum_correction_step_ratio",
    "maximum_corrected_radius_excess",
    "mechanism_rotation_threshold",
)
FINAL_CERTIFICATE_FIELDS: Final = (
    "residual_value_defect",
    "residual_gradient_defect",
    "hvp_symmetry_defect",
    "probe_normalized_curvature",
    "gram_factorization_relative_residual",
    "multiplier_relative_residual",
    "multiplier_forward_error_bound",
    "projection_tangency_relative_residual",
    "projection_solve_relative_residual",
    "projection_forward_error_bound",
)

PHASE_IDS: Final = (
    "gntr.current_linearization",
    "gntr.current_certificates",
    "gntr.steihaug",
    "gntr.trial_evaluation",
    "gntr.nonlinear_correction",
    "gntr.corrected_candidate_evaluation",
    "gntr.acceptance_radius_update",
)
PHASE_SCHEMA_SHA256: Final = (
    "f7cd595701cc206ae0c87286c3af052879f1d8db782d81362f2cc6d7337e7569"
)
TRACE_LOOP_ENVELOPE_NAME: Final = "neq_gntr1_diag_loop_envelope_v1"
REQUIRED_SOURCE_ROLE_BINDINGS: Final = {
    "benchmarks/run_single_stage_native_equivalent_quality_campaign.py": "benchmark",
    "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py": (
        "benchmark"
    ),
    "docs/single_stage_jax_gpu_native_equivalent_quality_no_hit_diagnostic_implementation_plan.md": (
        "configuration"
    ),
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag2_implementation_plan.md": (
        "configuration"
    ),
    "src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py": (
        "execution_source"
    ),
    "src/simsopt_jax/runtime/trace_annotations.py": "execution_source",
    "src/simsopt_jax/solve/fullspace_native_equivalent_quality.py": (
        "execution_source"
    ),
    "src/simsopt_jax_adapters/geo/single_stage_fullspace.py": "execution_source",
    "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py": (
        "test"
    ),
    "tests/benchmarks/_diag2_fixture.py": "test",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py": (
        "test"
    ),
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py": (
        "test"
    ),
    "tests/geo/test_fullspace_native_equivalent_quality.py": "test",
    "tests/geo/test_projected_gauss_newton_trust_region.py": "test",
}
CURRENT_MODEL_PHASES: Final = frozenset(PHASE_IDS[:2])

HISTORY_FLOAT_FIELDS: Final = (
    "current_objective",
    "current_feasibility_inf",
    "current_stationarity_inf",
    "candidate_objective",
    "candidate_feasibility_inf",
    "actual_reduction",
    "predicted_reduction",
    "reduction_ratio",
    "trust_radius",
    "next_trust_radius",
    "tangent_step_norm",
    "correction_norm",
    "applied_step_norm",
    "correction_step_ratio",
    "corrected_radius_ratio",
    "terminal_normalized_curvature",
    "residual_value_defect",
    "residual_gradient_defect",
    "hvp_symmetry_defect",
    "probe_normalized_curvature",
    "direction_rotation",
    "correction_relative_residual",
    "correction_forward_error_bound",
    "trial_gram_factorization_relative_residual",
    "trial_gram_solve_relative_residual",
    "current_projection_tangency_relative_residual",
    "current_projection_solve_relative_residual",
    "current_projection_forward_error_bound",
    "steihaug_tangency_relative_residual",
    "steihaug_final_projected_residual_norm",
    "steihaug_projected_residual_target",
    "steihaug_residual_projection_tangency_relative_residual",
    "steihaug_residual_projection_solve_relative_residual",
    "steihaug_residual_projection_forward_error_bound",
)
HISTORY_INTEGER_FIELDS: Final = (
    "steihaug_iterations",
    "steihaug_hvp_evaluations",
    "steihaug_termination",
)
HISTORY_ROW_RAW_FIELDS: Final = (
    "outcome",
    "accepted_step_number",
    "steihaug_hit_boundary",
    *HISTORY_INTEGER_FIELDS,
    *HISTORY_FLOAT_FIELDS,
)

TERMINAL_RAW_SCALAR_FIELDS: Final = (
    "objective",
    "reconstructed_objective",
    "authoritative_objective",
    "raw_kkt_inf",
    "scaled_stationarity_inf",
    "residual_value_defect",
    "residual_gradient_defect",
    "transpose_primal_dot",
    "transpose_adjoint_dot",
    "transpose_denominator",
    "transpose_defect",
    "terminal_endpoint_diagnostics_seconds",
)
DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS: Final = (
    "non_qs",
    "residual",
    "iota",
    "major_radius",
    "length",
)
DIAG4_ENDPOINT_OBSERVABLE_FIELDS: Final = (
    "iota",
    "G",
    "volume",
    "major_radius",
    "total_length",
    "non_qs_ratio",
    "boozer_residual_value",
    "boozer_residual_rms",
)
POLICY_RAW_VECTOR_FIELDS: Final = (
    "native_raw_equalities",
    "constraint_inverse_scale",
)
POLICY_RAW_HASH_FIELDS: Final = (
    "policy_sha256",
    "native_raw_equalities_sha256",
    "constraint_inverse_scale_sha256",
)
POLICY_RAW_SCALAR_FIELDS: Final = (
    "objective_target",
    "state_size",
    "equality_size",
    "objective_residual_size",
    "component_absolute_tolerance",
    "component_relative_tolerance",
    "scaled_feasibility_tolerance",
    "residual_value_defect_tolerance",
    "residual_gradient_defect_tolerance",
    "transpose_defect_tolerance",
)

ARRAY_SPECS: Final = {
    "optimizer_coordinates": ("<f8", (STATE_SIZE,)),
    "physical_state": ("<f8", (STATE_SIZE,)),
    "raw_equalities": ("<f8", (EQUALITY_SIZE,)),
    "scaled_equalities": ("<f8", (EQUALITY_SIZE,)),
    "objective_gradient": ("<f8", (STATE_SIZE,)),
    "multipliers": ("<f8", (EQUALITY_SIZE,)),
    "raw_stationarity": ("<f8", (STATE_SIZE,)),
    "native_equalities": ("<f8", (EQUALITY_SIZE,)),
    "constraint_inverse_scale": ("<f8", (EQUALITY_SIZE,)),
    "accepted_optimizer_ledger": ("<f8", (LEDGER_SIZE, STATE_SIZE)),
    "accepted_physical_ledger": ("<f8", (LEDGER_SIZE, STATE_SIZE)),
    "accepted_mask": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_objectives": ("<f8", (LEDGER_SIZE,)),
    "accepted_quality_raw_equalities": ("<f8", (LEDGER_SIZE, EQUALITY_SIZE)),
    "accepted_quality_scaled_equalities": ("<f8", (LEDGER_SIZE, EQUALITY_SIZE)),
    "accepted_quality_mask": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_coordinates_finite": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_objective_finite": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_raw_equalities_finite": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_scaled_equalities_finite": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_objective_satisfied": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_component_bounds_satisfied": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_scaled_feasibility_satisfied": ("|b1", (LEDGER_SIZE,)),
    "accepted_quality_satisfied": ("|b1", (LEDGER_SIZE,)),
    "authoritative_objective_gradient": ("<f8", (STATE_SIZE,)),
    "bootstrap_anchor": ("<f8", (STATE_SIZE,)),
    "constraint_jacobian": ("<f8", (EQUALITY_SIZE, STATE_SIZE)),
    "objective_residual_vector": ("<f8", (2110,)),
    "reconstructed_objective_gradient": ("<f8", (STATE_SIZE,)),
    "transpose_equality_probe": ("<f8", (EQUALITY_SIZE,)),
    "transpose_jvp_action": ("<f8", (EQUALITY_SIZE,)),
    "transpose_state_probe": ("<f8", (STATE_SIZE,)),
    "transpose_vjp_action": ("<f8", (STATE_SIZE,)),
    "variable_scale": ("<f8", (STATE_SIZE,)),
}
NONFINITE_ARRAYS: Final = frozenset(
    {
        "raw_stationarity",
        "accepted_optimizer_ledger",
        "accepted_quality_objectives",
        "accepted_quality_raw_equalities",
        "accepted_quality_scaled_equalities",
    }
)

EVIDENCE_REF_KEYS: Final = frozenset(
    {
        "history",
        "terminal_numerical",
        "raw_trace",
        "trace_intervals",
        "execution",
        "preflight",
        "preflight_child_terminal",
        "preflight_memory",
        "preflight_memory_samples",
        "preflight_process",
        "preflight_runtime",
        "preflight_policy",
        "policy_authority",
        "producer",
        "child_terminal",
        "runtime",
        "process",
        "memory",
        "memory_samples",
        "source_manifest",
        "native_reference",
        "policy",
    }
)
PREFLIGHT_EVIDENCE_REF_KEYS: Final = frozenset(
    {
        "producer",
        "child_terminal",
        "process",
        "memory",
        "memory_samples",
        "runtime",
        "preflight_policy",
        "policy_authority",
        "source_manifest",
        "native_reference",
    }
)

_LOWER_HEX: Final = frozenset("0123456789abcdef")


class AttemptOutcome(StrEnum):
    INACTIVE = "INACTIVE"
    ACCEPTED = "ACCEPTED"
    RETRY_OBJECTIVE = "RETRY_OBJECTIVE"
    RETRY_NONFINITE = "RETRY_NONFINITE"
    RETRY_CORRECTION_CERTIFICATE = "RETRY_CORRECTION_CERTIFICATE"
    RETRY_FEASIBILITY = "RETRY_FEASIBILITY"
    RETRY_STEP_BOUNDS = "RETRY_STEP_BOUNDS"
    FATAL_CURRENT_STATE = "FATAL_CURRENT_STATE"
    FATAL_STEIHAUG = "FATAL_STEIHAUG"
    FATAL_CURVATURE = "FATAL_CURVATURE"


RETRY_OUTCOMES: Final = frozenset(
    {
        AttemptOutcome.RETRY_OBJECTIVE,
        AttemptOutcome.RETRY_NONFINITE,
        AttemptOutcome.RETRY_CORRECTION_CERTIFICATE,
        AttemptOutcome.RETRY_FEASIBILITY,
        AttemptOutcome.RETRY_STEP_BOUNDS,
    }
)
RETRACTION_OUTCOMES: Final = frozenset(
    {
        AttemptOutcome.RETRY_CORRECTION_CERTIFICATE,
        AttemptOutcome.RETRY_FEASIBILITY,
        AttemptOutcome.RETRY_STEP_BOUNDS,
    }
)
FATAL_OUTCOMES: Final = frozenset(
    {
        AttemptOutcome.FATAL_CURRENT_STATE,
        AttemptOutcome.FATAL_STEIHAUG,
        AttemptOutcome.FATAL_CURVATURE,
    }
)


class LoopStatus(StrEnum):
    BOUNDED_COMPLETE = "BOUNDED_COMPLETE"
    ATTEMPT_LIMIT = "ATTEMPT_LIMIT"
    FATAL_CURRENT_STATE = "FATAL_CURRENT_STATE"
    FATAL_STEIHAUG = "FATAL_STEIHAUG"
    FATAL_CURVATURE = "FATAL_CURVATURE"
    DEVICE_QUALITY_CANDIDATE = "DEVICE_QUALITY_CANDIDATE"


class KktStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NONFINITE = "NONFINITE"


class PhaseTimingStatus(StrEnum):
    PRODUCED = "PRODUCED"
    NOT_PRODUCED = "PHASE_TIMING_NOT_PRODUCED"


class DiagnosticVerdict(StrEnum):
    INCOMPLETE = "DIAGNOSTIC_INCOMPLETE"
    QUALITY_HIT = "DIAGNOSTIC_COMPLETE_QUALITY_HIT_NONPROMOTING"
    NO_HIT = "DIAGNOSTIC_COMPLETE_NO_HIT"


class HistoricalAggregateRelation(StrEnum):
    MATCHES = "MATCHES_RETAINED_AGGREGATES"
    DIVERGES = "DIVERGES_FROM_RETAINED_AGGREGATES"


class NextRoute(StrEnum):
    RETRY_MODEL_REUSE = "RETRY_MODEL_REUSE"
    RADIUS_RETRACTION = "RADIUS_RETRACTION"
    CONDITIONING_MODEL_CHANGE = "CONDITIONING_MODEL_CHANGE"
    NOT_SELECTED = "NOT_SELECTED_DIAGNOSTIC_INCOMPLETE"


class FailureStage(StrEnum):
    PREFLIGHT = "PREFLIGHT_FAILED"
    COLD_TIMEOUT = "COLD_TIMEOUT"
    COLD_CRASH = "COLD_CRASH"
    COLD_PROTOCOL = "COLD_PROTOCOL_FAILURE"
    COLD_MONITOR = "COLD_MONITOR_FAILURE"
    COLD_RESOURCE = "COLD_RESOURCE_FAILURE"
    COLD_SOURCE = "COLD_SOURCE_FAILURE"
    NUMERICAL_EVIDENCE = "NUMERICAL_EVIDENCE_INCOMPLETE"


def _mapping(value: JsonValue, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _array(value: JsonValue, context: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, object], keys: frozenset[str], context: str
) -> None:
    if frozenset(value) != keys:
        raise ValueError(f"{context} keys differ from the frozen schema")


def _string(value: JsonValue, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _boolean(value: JsonValue, context: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{context} must be a Boolean")
    return value


def _integer(value: JsonValue, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} is below its minimum")
    return value


def _number(value: JsonValue, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _optional_number(value: JsonValue, context: str) -> float | None:
    return None if value is None else _number(value, context)


def _sha256(value: JsonValue, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(character not in _LOWER_HEX for character in result):
        raise ValueError(f"{context} must be a lower-case SHA-256")
    return result


def _artifact_ref(value: JsonValue, context: str) -> ArtifactRef:
    payload = _mapping(value, context)
    _exact_keys(
        payload,
        frozenset({"relative_path", "sha256", "size_bytes", "schema_version"}),
        context,
    )
    relative_path = _string(payload["relative_path"], f"{context}.relative_path")
    path = Path(relative_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative_path
        or not path.parts
    ):
        raise ValueError(f"{context}.relative_path is not canonical")
    return ArtifactRef(
        relative_path,
        _sha256(payload["sha256"], f"{context}.sha256"),
        _integer(payload["size_bytes"], f"{context}.size_bytes"),
        _string(payload["schema_version"], f"{context}.schema_version"),
    )


def _artifact_ref_payload(reference: ArtifactRef) -> dict[str, JsonValue]:
    return {
        "relative_path": reference.relative_path,
        "schema_version": reference.schema_version,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
    }


def _resolve_artifact(root: Path, reference: ArtifactRef) -> Path:
    held = _DIAG5_HELD_TREE.get()
    if held is not None and root.resolve(strict=True) == held.root.resolve(strict=True):
        data = held.file_bytes(reference.relative_path)
        if (
            len(data) != reference.size_bytes
            or hashlib.sha256(data).hexdigest() != reference.sha256
        ):
            raise ValueError(f"artifact held bytes differ: {reference.relative_path}")
        if Path(reference.relative_path).suffix == ".json":
            document = _mapping(
                load_canonical_json_bytes(data),
                f"artifact {reference.relative_path}",
            )
            if document.get("schema_version") != reference.schema_version:
                raise ValueError(f"artifact schema differs: {reference.relative_path}")
        return held.validation_path(reference.relative_path)
    base = root.resolve(strict=True)
    relative = Path(reference.relative_path)
    path = base.joinpath(relative)
    component = path
    while component != base:
        if component.is_symlink():
            raise ValueError(
                f"artifact path contains a symlink: {reference.relative_path}"
            )
        component = component.parent
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(base):
        raise ValueError(
            f"artifact is not a regular local file: {reference.relative_path}"
        )
    data = resolved.read_bytes()
    if (
        len(data) != reference.size_bytes
        or hashlib.sha256(data).hexdigest() != reference.sha256
    ):
        raise ValueError(f"artifact bytes differ: {reference.relative_path}")
    if resolved.suffix == ".json":
        document = _mapping(
            load_canonical_json_bytes(data), f"artifact {reference.relative_path}"
        )
        if document.get("schema_version") != reference.schema_version:
            raise ValueError(f"artifact schema differs: {reference.relative_path}")
    return resolved


@dataclass(frozen=True, slots=True)
class HistoryRow:
    outcome: AttemptOutcome
    accepted_step_number: int
    integer_values: tuple[int, int, int]
    steihaug_hit_boundary: bool
    floating_values: tuple[float | None, ...]

    def floating(self, name: str) -> float | None:
        return self.floating_values[HISTORY_FLOAT_FIELDS.index(name)]

    def to_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "outcome": self.outcome.value,
            "accepted_step_number": self.accepted_step_number,
            "steihaug_hit_boundary": self.steihaug_hit_boundary,
        }
        payload.update(zip(HISTORY_INTEGER_FIELDS, self.integer_values, strict=True))
        payload.update(zip(HISTORY_FLOAT_FIELDS, self.floating_values, strict=True))
        return payload


@dataclass(frozen=True, slots=True)
class HistoryEvidence:
    rows: tuple[HistoryRow, ...]
    attempts: int
    accepted_steps: int
    retryable_rejections: int
    status: LoopStatus
    fatal: bool
    bounded_complete: bool
    quality_latch: bool
    first_quality_attempt: int
    first_quality_accepted_step: int


@dataclass(frozen=True, slots=True)
class ArrayEvidence:
    reference: ArtifactRef
    dtype: str
    shape: tuple[int, ...]
    content_sha256: str
    values: np.ndarray

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact": _artifact_ref_payload(self.reference),
            "content_sha256": self.content_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class PolicyEvidence:
    policy_sha256: str
    native_raw_equalities_sha256: str
    constraint_inverse_scale_sha256: str
    objective_target: float
    component_absolute_tolerance: float
    component_relative_tolerance: float
    scaled_feasibility_tolerance: float
    residual_value_defect_tolerance: float
    residual_gradient_defect_tolerance: float
    transpose_defect_tolerance: float
    native_raw_equalities: tuple[float, ...]
    constraint_inverse_scale: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TerminalEvidence:
    arrays: tuple[tuple[str, ArrayEvidence], ...]
    objective: float
    objective_terms: tuple[tuple[str, float], ...]
    objective_weights: tuple[tuple[str, float], ...]
    reconstructed_objective: float
    authoritative_objective: float
    final_certificate: tuple[tuple[str, float], ...]
    final_certificate_passes: bool
    kkt_status: KktStatus
    raw_kkt_inf: float | None
    scaled_stationarity_inf: float | None
    residual_value_defect: float
    residual_gradient_defect: float
    transpose_primal_dot: float
    transpose_adjoint_dot: float
    transpose_denominator: float
    transpose_defect: float
    terminal_endpoint_diagnostics_seconds: float

    def array(self, name: str) -> ArrayEvidence:
        return dict(self.arrays)[name]


@dataclass(frozen=True, slots=True)
class TerminalEvidenceV4:
    terminal: TerminalEvidence
    numerical_identity: NativeEquivalentNumericalIdentity
    endpoint_state_sha256: str
    terminal_observables: tuple[tuple[str, float], ...]
    endpoint_objective_terms: tuple[tuple[str, float], ...]
    endpoint_observables: tuple[tuple[str, float], ...]


class ScientificOutcome(StrEnum):
    INCOMPLETE = "INCOMPLETE"
    NO_HIT = "NO_HIT"
    QUALITY_HIT = "QUALITY_HIT"


@dataclass(frozen=True, slots=True)
class NativeEquivalentNumericalIdentity:
    numerical_route: str
    numerical_result_schema_version: str
    problem_sha256: str
    optimizer_options_sha256: str
    base_neq_gntr1_policy_sha256: str
    scaling_sha256: str
    bootstrap_state_sha256: str
    initial_physical_state_sha256: str
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class NativeEquivalentScientificEvidence:
    backend: str
    numerical_identity: NativeEquivalentNumericalIdentity
    history: HistoryEvidence
    safeguard_telemetry: SafeguardTelemetryV4
    terminal: TerminalEvidenceV4
    policy: PolicyEvidence
    endpoint_audit: NativeEquivalentEndpointAudit
    quality: QualityEvidence
    outcome: ScientificOutcome


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    objective_margin: float
    component_margins: tuple[float, ...]
    minimum_component_margin: float
    minimum_component_index: int
    scaled_feasibility_margin: float
    objective_usage_ratio: float
    component_usage_ratio: float
    feasibility_usage_ratio: float
    residual_value_margin: float
    residual_gradient_margin: float
    transpose_margin: float

    @property
    def passes(self) -> bool:
        return bool(
            self.objective_margin >= 0.0
            and self.minimum_component_margin >= 0.0
            and self.scaled_feasibility_margin >= 0.0
        )


@dataclass(frozen=True, slots=True)
class Interval:
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    status: PhaseTimingStatus
    durations_ns: tuple[tuple[str, int], ...]
    overlaps_ns: tuple[tuple[str, str, int], ...]
    device_active_ns: int
    total_attributed_ns: int
    unattributed_ns: int
    current_model_ns: int
    coverage: float
    trace_start_ns: int
    trace_stop_ns: int


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    supporting_evidence: tuple[tuple[str, ArtifactRef], ...]
    preflight_status: str
    preflight_compile_success: bool
    preflight_solver_dispatched: bool
    preflight_finalizer_called: bool
    preflight_endpoint_audit_called: bool
    preflight_campaign_authorized: bool
    preflight_callbacks: int
    cold_status: str
    child_pid: int
    child_start_time_ticks: int
    backend: str
    gpu_uuid: str
    jax_enable_x64: bool
    state_size: int
    equality_size: int
    residual_size: int
    policy_sha256: str
    phase_schema_sha256: str
    source_pre_sha256: str
    source_post_sha256: str
    runtime_environment_sha256: str
    interpreter: str
    argv: tuple[str, ...]
    physical_memory_bytes: int
    peak_memory_bytes: int
    reported_peak_memory_fraction: float
    hot_h2d_transfers: int
    hot_d2h_transfers: int
    python_callbacks: int
    final_d2h_transfers: int
    timestamps_ns: tuple[tuple[str, int], ...]
    stdout_sha256: str
    stdout_size_bytes: int
    stderr_sha256: str
    stderr_size_bytes: int

    @property
    def peak_memory_fraction(self) -> float:
        return self.peak_memory_bytes / self.physical_memory_bytes

    def passes(self) -> bool:
        timestamps = dict(self.timestamps_ns)
        ordered = (
            "process_started",
            "compile_started",
            "compile_completed",
            "state_ready",
            "profiler_started",
            "solve_started",
            "solve_stopped",
            "profiler_stopped",
            "finalizer_started",
            "finalizer_stopped",
            "quality_replay_started",
            "quality_replay_stopped",
            "endpoint_diagnostics_started",
            "endpoint_diagnostics_stopped",
            "final_d2h",
            "trace_exported",
            "serialized",
            "process_stopped",
        )
        return bool(
            self.preflight_status == "COMPLETE"
            and self.preflight_compile_success
            and not self.preflight_solver_dispatched
            and not self.preflight_finalizer_called
            and not self.preflight_endpoint_audit_called
            and not self.preflight_campaign_authorized
            and self.preflight_callbacks == 0
            and self.cold_status == "COMPLETE"
            and self.child_pid > 0
            and self.child_start_time_ticks > 0
            and self.backend == "gpu"
            and self.gpu_uuid == GPU_UUID
            and self.jax_enable_x64
            and self.state_size == STATE_SIZE
            and self.equality_size == EQUALITY_SIZE
            and self.residual_size == 2110
            and self.source_pre_sha256 == self.source_post_sha256
            and self.physical_memory_bytes > 0
            and self.peak_memory_bytes >= 0
            and math.isclose(
                self.reported_peak_memory_fraction,
                self.peak_memory_fraction,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            and self.peak_memory_fraction < 0.8
            and self.hot_h2d_transfers == 0
            and self.hot_d2h_transfers == 0
            and self.python_callbacks == 0
            and self.final_d2h_transfers == 1
            and frozenset(timestamps) == frozenset(ordered)
            and all(
                timestamps[left] < timestamps[right]
                for left, right in zip(ordered[:-1], ordered[1:], strict=True)
            )
        )


@dataclass(frozen=True, slots=True)
class DiagnosticReceipt:
    evidence_refs: tuple[tuple[str, ArtifactRef], ...]
    policy: PolicyEvidence
    history: HistoryEvidence
    terminal: TerminalEvidence
    quality: QualityEvidence
    phases: PhaseEvidence
    execution: ExecutionEvidence
    verdict: DiagnosticVerdict
    historical_relation: HistoricalAggregateRelation
    next_route: NextRoute
    reuse_opportunity_estimate: float | None


@dataclass(frozen=True, slots=True)
class IncompleteDiagnosticReceipt:
    evidence_refs: tuple[tuple[str, ArtifactRef | None], ...]
    failure_stage: FailureStage
    failure_reasons: tuple[str, ...]


def _history_row(
    value: JsonValue, index: int, *, defer_step_bounds: bool = False
) -> HistoryRow:
    context = f"history.rows[{index}]"
    payload = _mapping(value, context)
    keys = frozenset(
        {
            "outcome",
            "accepted_step_number",
            "steihaug_hit_boundary",
            *HISTORY_INTEGER_FIELDS,
            *HISTORY_FLOAT_FIELDS,
        }
    )
    _exact_keys(payload, keys, context)
    outcome = AttemptOutcome(_string(payload["outcome"], f"{context}.outcome"))
    integers = tuple(
        _integer(payload[name], f"{context}.{name}") for name in HISTORY_INTEGER_FIELDS
    )
    if any(value < 0 for value in integers) or integers[2] > 3:
        raise ValueError(f"{context}.steihaug_termination is invalid")
    row = HistoryRow(
        outcome=outcome,
        accepted_step_number=_integer(
            payload["accepted_step_number"], f"{context}.accepted_step_number"
        ),
        integer_values=(integers[0], integers[1], integers[2]),
        steihaug_hit_boundary=_boolean(
            payload["steihaug_hit_boundary"], f"{context}.steihaug_hit_boundary"
        ),
        floating_values=tuple(
            _optional_number(payload[name], f"{context}.{name}")
            for name in HISTORY_FLOAT_FIELDS
        ),
    )
    if outcome is AttemptOutcome.INACTIVE:
        if (
            row.accepted_step_number != 0
            or any(row.integer_values)
            or row.steihaug_hit_boundary
            or any(item is not None for item in row.floating_values)
        ):
            raise ValueError(f"{context} inactive padding is not canonical")
    else:
        for required in ("trust_radius", "next_trust_radius"):
            if row.floating(required) is None:
                raise ValueError(f"{context}.{required} is required for an active row")
        if (
            row.floating("trust_radius") <= 0.0
            or row.floating("next_trust_radius") <= 0.0
        ):  # type: ignore[operator]
            raise ValueError(f"{context} trust radii must be positive")
        base_fields = (
            "current_objective",
            "current_feasibility_inf",
            "current_stationarity_inf",
            "residual_value_defect",
            "residual_gradient_defect",
            "hvp_symmetry_defect",
            "probe_normalized_curvature",
            "current_projection_tangency_relative_residual",
            "current_projection_solve_relative_residual",
            "current_projection_forward_error_bound",
        )
        step_fields = (
            "tangent_step_norm",
            "terminal_normalized_curvature",
            "direction_rotation",
            "steihaug_tangency_relative_residual",
            "steihaug_final_projected_residual_norm",
            "steihaug_projected_residual_target",
            "steihaug_residual_projection_tangency_relative_residual",
            "steihaug_residual_projection_solve_relative_residual",
            "steihaug_residual_projection_forward_error_bound",
        )
        trial_fields = (
            "candidate_objective",
            "candidate_feasibility_inf",
            "actual_reduction",
            "predicted_reduction",
            "reduction_ratio",
            "correction_norm",
            "applied_step_norm",
            "correction_step_ratio",
            "corrected_radius_ratio",
            "correction_relative_residual",
            "correction_forward_error_bound",
            "trial_gram_factorization_relative_residual",
            "trial_gram_solve_relative_residual",
        )

        def require_finite(names: tuple[str, ...]) -> None:
            if any(row.floating(name) is None for name in names):
                raise ValueError(f"{context} omits required stage telemetry")

        def require_null(names: tuple[str, ...]) -> None:
            if any(row.floating(name) is not None for name in names):
                raise ValueError(f"{context} populates telemetry before its stage")

        if outcome is AttemptOutcome.FATAL_CURRENT_STATE:
            require_null((*step_fields, *trial_fields))
            if any(row.integer_values) or row.steihaug_hit_boundary:
                raise ValueError(f"{context} populates Steihaug telemetry")
        elif outcome is AttemptOutcome.FATAL_STEIHAUG:
            require_finite(base_fields)
            require_null(trial_fields)
        elif outcome is AttemptOutcome.FATAL_CURVATURE:
            require_finite(base_fields)
            require_null(trial_fields)
            populated = tuple(row.floating(name) is not None for name in step_fields)
            if any(populated) and not all(populated):
                raise ValueError(f"{context} has a partial curvature-stage pattern")
        elif outcome is AttemptOutcome.RETRY_NONFINITE:
            require_finite((*base_fields, *step_fields))
            if all(row.floating(name) is not None for name in trial_fields):
                raise ValueError(f"{context} nonfinite retry has no null raw telemetry")
        else:
            finite_trial_fields = tuple(
                name for name in trial_fields if name != "reduction_ratio"
            )
            require_finite((*base_fields, *step_fields, *finite_trial_fields))
            predicted_reduction = row.floating("predicted_reduction")
            actual_reduction = row.floating("actual_reduction")
            reduction_ratio = row.floating("reduction_ratio")
            if predicted_reduction == 0.0:
                if reduction_ratio is not None:
                    raise ValueError(
                        f"{context}.reduction_ratio must be null for zero prediction"
                    )
            elif reduction_ratio is None or not math.isclose(
                reduction_ratio,
                actual_reduction / predicted_reduction,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError(f"{context}.reduction_ratio differs from reductions")
            correction_valid = bool(
                row.floating("trial_gram_factorization_relative_residual")
                <= FROZEN_GNTR_OPTIONS["linear_residual_tolerance"]
                and row.floating("correction_relative_residual")
                <= FROZEN_GNTR_OPTIONS["linear_residual_tolerance"]
                and row.floating("trial_gram_solve_relative_residual")
                <= FROZEN_GNTR_OPTIONS["linear_residual_tolerance"]
                and row.floating("correction_forward_error_bound")
                < FROZEN_GNTR_OPTIONS["forward_error_tolerance"]
            )
            feasibility_valid = bool(
                row.floating("candidate_feasibility_inf")
                <= FROZEN_GNTR_OPTIONS["corrected_feasibility_tolerance"]
            )
            bounds_valid = bool(
                row.floating("correction_norm")
                <= FROZEN_GNTR_OPTIONS["maximum_correction_step_ratio"]
                * min(row.floating("tangent_step_norm"), row.floating("trust_radius"))
                and row.floating("applied_step_norm")
                <= (1.0 + FROZEN_GNTR_OPTIONS["maximum_corrected_radius_excess"])
                * row.floating("trust_radius")
            )
            predicted_positive = predicted_reduction > 0.0
            actual_positive = actual_reduction > 0.0
            expected_before_step_bounds = (
                AttemptOutcome.RETRY_CORRECTION_CERTIFICATE
                if not correction_valid
                else (
                    AttemptOutcome.RETRY_FEASIBILITY if not feasibility_valid else None
                )
            )
            if expected_before_step_bounds is not None:
                valid_outcome = outcome is expected_before_step_bounds
            elif defer_step_bounds:
                valid_outcome = outcome in (
                    AttemptOutcome.RETRY_STEP_BOUNDS,
                    AttemptOutcome.ACCEPTED
                    if predicted_positive and actual_positive
                    else AttemptOutcome.RETRY_OBJECTIVE,
                )
            else:
                expected = (
                    AttemptOutcome.RETRY_STEP_BOUNDS
                    if not bounds_valid
                    else (
                        AttemptOutcome.ACCEPTED
                        if predicted_positive and actual_positive
                        else AttemptOutcome.RETRY_OBJECTIVE
                    )
                )
                valid_outcome = outcome is expected
            if not valid_outcome:
                raise ValueError(f"{context} outcome differs from raw certificates")
    return row


def _parse_history(
    value: JsonValue,
    *,
    defer_step_bounds: bool = False,
    expected_schema_version: str = f"{SCHEMA_VERSION}-history",
) -> HistoryEvidence:
    payload = _mapping(value, "history evidence")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "rows",
                "attempts",
                "accepted_steps",
                "retryable_rejections",
                "status",
                "fatal",
                "bounded_complete",
                "quality_latch",
                "first_quality_attempt",
                "first_quality_accepted_step",
            }
        ),
        "history evidence",
    )
    if payload["schema_version"] != expected_schema_version:
        raise ValueError("history schema differs")
    rows = tuple(
        _history_row(row, index, defer_step_bounds=defer_step_bounds)
        for index, row in enumerate(_array(payload["rows"], "history.rows"))
    )
    if len(rows) != MAXIMUM_ATTEMPTS:
        raise ValueError("history must retain exactly 300 rows")
    outcomes = tuple(row.outcome for row in rows)
    attempts = sum(outcome is not AttemptOutcome.INACTIVE for outcome in outcomes)
    if any(
        outcome is AttemptOutcome.INACTIVE for outcome in outcomes[:attempts]
    ) or any(outcome is not AttemptOutcome.INACTIVE for outcome in outcomes[attempts:]):
        raise ValueError("history active rows are not an exact prefix")
    accepted = sum(outcome is AttemptOutcome.ACCEPTED for outcome in outcomes)
    retries = sum(outcome in RETRY_OUTCOMES for outcome in outcomes)
    accepted_number = 0
    for index, row in enumerate(rows[:attempts]):
        if row.outcome is AttemptOutcome.ACCEPTED:
            accepted_number += 1
            if row.accepted_step_number != accepted_number:
                raise ValueError(f"history.rows[{index}] accepted numbering differs")
        elif row.accepted_step_number != 0:
            raise ValueError(f"history.rows[{index}] rejection has accepted number")
    fatal = attempts > 0 and outcomes[attempts - 1] in FATAL_OUTCOMES
    if any(outcome in FATAL_OUTCOMES for outcome in outcomes[: max(0, attempts - 1)]):
        raise ValueError("history contains a nonterminal fatal outcome")
    quality_latch = _boolean(payload["quality_latch"], "history.quality_latch")
    bounded = accepted == MAXIMUM_ACCEPTED_STEPS
    if fatal:
        status = LoopStatus(outcomes[attempts - 1].value)
    elif quality_latch:
        status = LoopStatus.DEVICE_QUALITY_CANDIDATE
    elif bounded:
        status = LoopStatus.BOUNDED_COMPLETE
    else:
        status = LoopStatus.ATTEMPT_LIMIT
    first_attempt = _integer(payload["first_quality_attempt"], "first quality attempt")
    first_accepted = _integer(
        payload["first_quality_accepted_step"], "first quality accepted step"
    )
    if quality_latch:
        if not (1 <= first_attempt <= attempts and 1 <= first_accepted <= accepted):
            raise ValueError("quality latch counters are invalid")
        if first_attempt != attempts or first_accepted != accepted:
            raise ValueError("quality latch must terminate at the first-hit row")
    elif first_attempt != 0 or first_accepted != 0:
        raise ValueError("no-hit history must use zero first-hit counters")
    claimed = (
        _integer(payload["attempts"], "history.attempts"),
        _integer(payload["accepted_steps"], "history.accepted_steps"),
        _integer(payload["retryable_rejections"], "history.retryable_rejections"),
        LoopStatus(_string(payload["status"], "history.status")),
        _boolean(payload["fatal"], "history.fatal"),
        _boolean(payload["bounded_complete"], "history.bounded_complete"),
    )
    if claimed != (attempts, accepted, retries, status, fatal, bounded):
        raise ValueError("history terminal summaries differ from raw rows")
    return HistoryEvidence(
        rows,
        attempts,
        accepted,
        retries,
        status,
        fatal,
        bounded,
        quality_latch,
        first_attempt,
        first_accepted,
    )


def _load_array(artifact_root: Path, value: JsonValue, name: str) -> ArrayEvidence:
    context = f"terminal.arrays.{name}"
    payload = _mapping(value, context)
    _exact_keys(
        payload,
        frozenset({"artifact", "content_sha256", "dtype", "shape"}),
        context,
    )
    reference = _artifact_ref(payload["artifact"], f"{context}.artifact")
    path = _resolve_artifact(artifact_root, reference)
    dtype, shape = ARRAY_SPECS[name]
    observed_shape = tuple(
        _integer(item, f"{context}.shape")
        for item in _array(payload["shape"], f"{context}.shape")
    )
    if payload["dtype"] != dtype or observed_shape != shape:
        raise ValueError(f"{context} dtype or shape differs")
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
    if (
        values.dtype.str != dtype
        or values.shape != shape
        or not values.flags.c_contiguous
    ):
        raise ValueError(f"{context} array representation differs")
    if (
        values.dtype.kind == "f"
        and name not in NONFINITE_ARRAYS
        and not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{context} contains nonfinite values")
    canonical = np.ascontiguousarray(values).tobytes()
    content_sha256 = hashlib.sha256(canonical).hexdigest()
    if content_sha256 != _sha256(
        payload["content_sha256"], f"{context}.content_sha256"
    ):
        raise ValueError(f"{context} content digest differs")
    frozen = np.array(values, copy=True)
    frozen.flags.writeable = False
    return ArrayEvidence(reference, dtype, shape, content_sha256, frozen)


def _named_numbers(value: JsonValue, context: str) -> tuple[tuple[str, float], ...]:
    payload = _mapping(value, context)
    expected = frozenset({"non_qs", "residual", "iota", "major_radius", "length"})
    _exact_keys(payload, expected, context)
    return tuple(
        (name, _number(payload[name], f"{context}.{name}")) for name in sorted(expected)
    )


def _parse_policy(
    value: JsonValue,
    terminal: TerminalEvidence | None = None,
    *,
    expected_schema_version: str = f"{SCHEMA_VERSION}-policy",
) -> PolicyEvidence:
    payload = _mapping(value, "quality policy evidence")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "policy_sha256",
                "native_raw_equalities_sha256",
                "native_raw_equalities",
                "constraint_inverse_scale_sha256",
                "constraint_inverse_scale",
                "objective_target",
                "state_size",
                "equality_size",
                "objective_residual_size",
                "component_absolute_tolerance",
                "component_relative_tolerance",
                "scaled_feasibility_tolerance",
                "residual_value_defect_tolerance",
                "residual_gradient_defect_tolerance",
                "transpose_defect_tolerance",
                "gntr_options",
            }
        ),
        "quality policy evidence",
    )
    if payload["schema_version"] != expected_schema_version:
        raise ValueError("quality policy schema differs")
    options_payload = _mapping(payload["gntr_options"], "GNTR options")
    _exact_keys(options_payload, frozenset(FROZEN_GNTR_OPTIONS), "GNTR options")
    options: dict[str, int | float] = {}
    for name, expected in FROZEN_GNTR_OPTIONS.items():
        observed = (
            _integer(options_payload[name], f"GNTR options.{name}")
            if isinstance(expected, int)
            else _number(options_payload[name], f"GNTR options.{name}")
        )
        if observed != expected:
            raise ValueError(f"GNTR options.{name} differs")
        options[name] = observed
    native = np.asarray(
        [
            _number(item, "policy native equality")
            for item in _array(
                payload["native_raw_equalities"], "policy native equalities"
            )
        ],
        dtype=np.dtype("<f8"),
    )
    scale = np.asarray(
        [
            _number(item, "policy constraint scale")
            for item in _array(
                payload["constraint_inverse_scale"], "policy constraint scale"
            )
        ],
        dtype=np.dtype("<f8"),
    )
    policy = PolicyEvidence(
        _sha256(payload["policy_sha256"], "quality policy SHA"),
        _sha256(payload["native_raw_equalities_sha256"], "native equalities SHA"),
        _sha256(payload["constraint_inverse_scale_sha256"], "constraint scale SHA"),
        _number(payload["objective_target"], "objective target"),
        _number(
            payload["component_absolute_tolerance"], "component absolute tolerance"
        ),
        _number(
            payload["component_relative_tolerance"], "component relative tolerance"
        ),
        _number(
            payload["scaled_feasibility_tolerance"],
            "scaled feasibility tolerance",
        ),
        _number(
            payload["residual_value_defect_tolerance"],
            "residual value tolerance",
        ),
        _number(
            payload["residual_gradient_defect_tolerance"],
            "residual gradient tolerance",
        ),
        _number(payload["transpose_defect_tolerance"], "transpose tolerance"),
        tuple(float(item) for item in native),
        tuple(float(item) for item in scale),
    )
    if (
        native.shape != (EQUALITY_SIZE,)
        or scale.shape != (EQUALITY_SIZE,)
        or not np.all(np.isfinite(native))
        or not np.all(np.isfinite(scale))
        or np.any(scale == 0.0)
        or _integer(payload["state_size"], "policy state size") != STATE_SIZE
        or _integer(payload["equality_size"], "policy equality size") != EQUALITY_SIZE
        or _integer(payload["objective_residual_size"], "policy residual size") != 2110
        or policy.objective_target != OBJECTIVE_MAXIMUM
        or policy.component_absolute_tolerance != RAW_EQUALITY_ABSOLUTE_TOLERANCE
        or policy.component_relative_tolerance != RAW_EQUALITY_RELATIVE_TOLERANCE
        or policy.scaled_feasibility_tolerance != FEASIBILITY_MAXIMUM
        or policy.residual_value_defect_tolerance != RESIDUAL_VALUE_DEFECT_MAXIMUM
        or policy.residual_gradient_defect_tolerance != RESIDUAL_GRADIENT_DEFECT_MAXIMUM
        or policy.transpose_defect_tolerance != TRANSPOSE_DEFECT_MAXIMUM
        or policy.native_raw_equalities_sha256 != exact_numeric_tree_sha256(native)
        or policy.constraint_inverse_scale_sha256
        != hashlib.sha256(scale.tobytes()).hexdigest()
    ):
        raise ValueError("quality policy raw evidence differs")
    if terminal is not None and (
        not np.array_equal(native, terminal.array("native_equalities").values)
        or not np.array_equal(scale, terminal.array("constraint_inverse_scale").values)
    ):
        raise ValueError("terminal policy arrays differ from raw policy evidence")
    reconstructed = exact_numeric_tree_sha256(
        (
            "single-stage-native-equivalent-quality-policy-v1",
            "single-stage-fullspace-neq-gntr1-result-v1",
            NUMERICAL_ROUTE,
            native,
            policy.native_raw_equalities_sha256,
            scale,
            policy.objective_target,
            STATE_SIZE,
            EQUALITY_SIZE,
            2110,
            policy.component_absolute_tolerance,
            policy.component_relative_tolerance,
            policy.scaled_feasibility_tolerance,
            policy.residual_value_defect_tolerance,
            policy.residual_gradient_defect_tolerance,
            policy.transpose_defect_tolerance,
            tuple((name, options[name]) for name in GNTR_OPTION_ORDER),
        )
    )
    if reconstructed != policy.policy_sha256:
        raise ValueError("quality policy identity differs from raw evidence")
    return policy


def _parse_terminal(artifact_root: Path, value: JsonValue) -> TerminalEvidence:
    payload = _mapping(value, "terminal numerical evidence")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "arrays",
                "objective",
                "objective_terms",
                "objective_weights",
                "reconstructed_objective",
                "authoritative_objective",
                "final_certificate",
                "kkt_status",
                "raw_kkt_inf",
                "scaled_stationarity_inf",
                "residual_value_defect",
                "residual_gradient_defect",
                "transpose_primal_dot",
                "transpose_adjoint_dot",
                "transpose_denominator",
                "transpose_defect",
                "terminal_endpoint_diagnostics_seconds",
            }
        ),
        "terminal numerical evidence",
    )
    if payload["schema_version"] != f"{SCHEMA_VERSION}-terminal":
        raise ValueError("terminal numerical schema differs")
    raw_arrays = _mapping(payload["arrays"], "terminal.arrays")
    _exact_keys(raw_arrays, frozenset(ARRAY_SPECS), "terminal.arrays")
    arrays = tuple(
        (name, _load_array(artifact_root, raw_arrays[name], name))
        for name in sorted(ARRAY_SPECS)
    )
    kkt_status = KktStatus(_string(payload["kkt_status"], "terminal.kkt_status"))
    raw_kkt = _optional_number(payload["raw_kkt_inf"], "terminal.raw_kkt_inf")
    scaled_kkt = _optional_number(
        payload["scaled_stationarity_inf"], "terminal.scaled_stationarity_inf"
    )
    if kkt_status is KktStatus.AVAILABLE:
        if raw_kkt is None or scaled_kkt is None:
            raise ValueError("available KKT evidence requires finite values")
    elif raw_kkt is not None or scaled_kkt is not None:
        raise ValueError("unavailable/nonfinite KKT evidence must use null values")
    primal = _number(payload["transpose_primal_dot"], "transpose primal dot")
    adjoint = _number(payload["transpose_adjoint_dot"], "transpose adjoint dot")
    denominator = _number(payload["transpose_denominator"], "transpose denominator")
    if denominator <= 0.0:
        raise ValueError("transpose denominator must be positive")
    defect = abs(primal - adjoint) / denominator
    reported = _number(payload["transpose_defect"], "transpose defect")
    if not math.isclose(reported, defect, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("transpose defect differs from raw dots")
    certificate_payload = _mapping(payload["final_certificate"], "final certificate")
    _exact_keys(
        certificate_payload,
        frozenset(FINAL_CERTIFICATE_FIELDS),
        "final certificate",
    )
    certificate = tuple(
        (name, _number(certificate_payload[name], f"final certificate.{name}"))
        for name in FINAL_CERTIFICATE_FIELDS
    )
    certificate_values = dict(certificate)
    certificate_passes = bool(
        certificate_values["residual_value_defect"] <= RESIDUAL_VALUE_DEFECT_MAXIMUM
        and certificate_values["residual_gradient_defect"]
        <= RESIDUAL_GRADIENT_DEFECT_MAXIMUM
        and certificate_values["hvp_symmetry_defect"]
        <= RESIDUAL_GRADIENT_DEFECT_MAXIMUM
        and certificate_values["probe_normalized_curvature"]
        >= -FROZEN_GNTR_OPTIONS["normalized_curvature_tolerance"]
        and certificate_values["gram_factorization_relative_residual"]
        <= FROZEN_GNTR_OPTIONS["linear_residual_tolerance"]
        and certificate_values["multiplier_relative_residual"]
        <= FROZEN_GNTR_OPTIONS["linear_residual_tolerance"]
        and certificate_values["multiplier_forward_error_bound"]
        < FROZEN_GNTR_OPTIONS["forward_error_tolerance"]
        and certificate_values["projection_tangency_relative_residual"]
        <= FROZEN_GNTR_OPTIONS["linear_residual_tolerance"]
        and certificate_values["projection_solve_relative_residual"]
        <= FROZEN_GNTR_OPTIONS["linear_residual_tolerance"]
        and certificate_values["projection_forward_error_bound"]
        < FROZEN_GNTR_OPTIONS["forward_error_tolerance"]
    )
    terminal = TerminalEvidence(
        arrays=arrays,
        objective=_number(payload["objective"], "terminal.objective"),
        objective_terms=_named_numbers(payload["objective_terms"], "objective terms"),
        objective_weights=_named_numbers(
            payload["objective_weights"], "objective weights"
        ),
        reconstructed_objective=_number(
            payload["reconstructed_objective"], "reconstructed objective"
        ),
        authoritative_objective=_number(
            payload["authoritative_objective"], "authoritative objective"
        ),
        final_certificate=certificate,
        final_certificate_passes=certificate_passes,
        kkt_status=kkt_status,
        raw_kkt_inf=raw_kkt,
        scaled_stationarity_inf=scaled_kkt,
        residual_value_defect=_number(
            payload["residual_value_defect"], "residual value defect"
        ),
        residual_gradient_defect=_number(
            payload["residual_gradient_defect"], "residual gradient defect"
        ),
        transpose_primal_dot=primal,
        transpose_adjoint_dot=adjoint,
        transpose_denominator=denominator,
        transpose_defect=reported,
        terminal_endpoint_diagnostics_seconds=_number(
            payload["terminal_endpoint_diagnostics_seconds"],
            "terminal endpoint diagnostic seconds",
        ),
    )
    weighted = sum(
        dict(terminal.objective_terms)[name] * weight
        for name, weight in terminal.objective_weights
    )
    if not math.isclose(terminal.objective, weighted, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("objective differs from raw term ledger and weights")
    return terminal


def _quality(terminal: TerminalEvidence) -> QualityEvidence:
    raw = terminal.array("raw_equalities").values
    scaled = terminal.array("scaled_equalities").values
    native = terminal.array("native_equalities").values
    inverse_scale = terminal.array("constraint_inverse_scale").values
    reconstructed = inverse_scale * raw
    if not np.allclose(scaled, reconstructed, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("scaled equalities differ from D*q")
    limits = (
        np.abs(native)
        + RAW_EQUALITY_ABSOLUTE_TOLERANCE
        + RAW_EQUALITY_RELATIVE_TOLERANCE * np.abs(native)
    )
    margins = limits - np.abs(raw)
    minimum_index = int(np.argmin(margins))
    maximum_scaled = float(np.max(np.abs(reconstructed)))
    return QualityEvidence(
        objective_margin=OBJECTIVE_MAXIMUM - terminal.objective,
        component_margins=tuple(float(item) for item in margins),
        minimum_component_margin=float(margins[minimum_index]),
        minimum_component_index=minimum_index,
        scaled_feasibility_margin=FEASIBILITY_MAXIMUM - maximum_scaled,
        objective_usage_ratio=terminal.objective / OBJECTIVE_MAXIMUM,
        component_usage_ratio=float(np.max(np.abs(raw) / limits)),
        feasibility_usage_ratio=maximum_scaled / FEASIBILITY_MAXIMUM,
        residual_value_margin=(
            RESIDUAL_VALUE_DEFECT_MAXIMUM - terminal.residual_value_defect
        ),
        residual_gradient_margin=(
            RESIDUAL_GRADIENT_DEFECT_MAXIMUM - terminal.residual_gradient_defect
        ),
        transpose_margin=TRANSPOSE_DEFECT_MAXIMUM - terminal.transpose_defect,
    )


def _validate_terminal_raw_evidence(
    terminal: TerminalEvidence, history: HistoryEvidence, policy: PolicyEvidence
) -> None:
    arrays = {name: evidence.values for name, evidence in terminal.arrays}
    optimizer = arrays["optimizer_coordinates"]
    anchor = arrays["bootstrap_anchor"]
    scale = arrays["variable_scale"]
    physical = arrays["physical_state"]
    if np.any(scale == 0.0) or not np.array_equal(physical, anchor + scale * optimizer):
        raise ValueError("terminal physical state differs from z0 + S*u")
    valid = history.accepted_steps + 1
    reconstructed_ledger = anchor[None, :] + (
        scale[None, :] * arrays["accepted_optimizer_ledger"][:valid]
    )
    if not np.array_equal(
        arrays["accepted_physical_ledger"][:valid], reconstructed_ledger
    ):
        raise ValueError("accepted physical ledger differs from z0 + S*u")
    if not np.array_equal(
        arrays["scaled_equalities"],
        arrays["constraint_inverse_scale"] * arrays["raw_equalities"],
    ):
        raise ValueError("terminal scaled equalities differ exactly from D*q")
    if not np.array_equal(
        arrays["authoritative_objective_gradient"], arrays["objective_gradient"]
    ):
        raise ValueError("authoritative objective gradients differ")

    residual = arrays["objective_residual_vector"]
    reconstructed_objective = 0.5 * float(np.dot(residual, residual))
    reconstructed_gradient = arrays["reconstructed_objective_gradient"]
    authoritative_gradient = arrays["authoritative_objective_gradient"]
    value_denominator = max(
        1.0, abs(reconstructed_objective), abs(terminal.authoritative_objective)
    )
    value_defect = (
        abs(reconstructed_objective - terminal.authoritative_objective)
        / value_denominator
    )
    gradient_denominator = max(
        1.0,
        float(np.linalg.norm(reconstructed_gradient, ord=np.inf)),
        float(np.linalg.norm(authoritative_gradient, ord=np.inf)),
    )
    gradient_defect = (
        float(
            np.linalg.norm(reconstructed_gradient - authoritative_gradient, ord=np.inf)
        )
        / gradient_denominator
    )
    if (
        reconstructed_objective != terminal.reconstructed_objective
        or terminal.authoritative_objective != terminal.objective
        or dict(terminal.final_certificate)["residual_value_defect"]
        != terminal.residual_value_defect
        or dict(terminal.final_certificate)["residual_gradient_defect"]
        != terminal.residual_gradient_defect
        or not math.isclose(
            value_defect,
            terminal.residual_value_defect,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            gradient_defect,
            terminal.residual_gradient_defect,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("residual objective reconstruction differs")

    jacobian = arrays["constraint_jacobian"]
    state_probe = arrays["transpose_state_probe"]
    equality_probe = arrays["transpose_equality_probe"]
    jvp_action = jacobian @ state_probe
    vjp_action = jacobian.T @ equality_probe
    primal = float(np.dot(equality_probe, jvp_action))
    adjoint = float(np.dot(state_probe, vjp_action))
    denominator = max(1.0, abs(primal), abs(adjoint))
    defect = abs(primal - adjoint) / denominator
    if (
        not np.allclose(
            arrays["transpose_jvp_action"], jvp_action, rtol=1.0e-12, atol=1.0e-15
        )
        or not np.allclose(
            arrays["transpose_vjp_action"], vjp_action, rtol=1.0e-12, atol=1.0e-15
        )
        or not math.isclose(
            terminal.transpose_primal_dot, primal, rel_tol=1e-12, abs_tol=1e-15
        )
        or not math.isclose(
            terminal.transpose_adjoint_dot, adjoint, rel_tol=1e-12, abs_tol=1e-15
        )
        or not math.isclose(
            terminal.transpose_denominator, denominator, rel_tol=1e-12, abs_tol=1e-15
        )
        or not math.isclose(
            terminal.transpose_defect, defect, rel_tol=1e-12, abs_tol=1e-15
        )
    ):
        raise ValueError("transpose raw actions or reported defect differ")

    gradient_u = scale * authoritative_gradient
    stationarity_u = gradient_u + jacobian.T @ arrays["multipliers"]
    scaled_stationarity_inf = float(np.linalg.norm(stationarity_u, ord=np.inf))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_stationarity = stationarity_u / scale
    raw_finite = bool(np.all(np.isfinite(raw_stationarity)))
    if raw_finite:
        raw_kkt_inf = float(np.linalg.norm(raw_stationarity, ord=np.inf))
        if (
            terminal.kkt_status is not KktStatus.AVAILABLE
            or terminal.raw_kkt_inf is None
            or terminal.scaled_stationarity_inf is None
            or not np.allclose(
                arrays["raw_stationarity"],
                raw_stationarity,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            or not math.isclose(
                terminal.raw_kkt_inf, raw_kkt_inf, rel_tol=1e-12, abs_tol=1e-15
            )
            or not math.isclose(
                terminal.scaled_stationarity_inf,
                scaled_stationarity_inf,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("finite KKT telemetry differs from raw evidence")
    elif terminal.kkt_status is not KktStatus.NONFINITE or not np.array_equal(
        np.isfinite(arrays["raw_stationarity"]), np.isfinite(raw_stationarity)
    ):
        raise ValueError("nonfinite KKT status differs from raw evidence")
    if not terminal.final_certificate_passes:
        raise ValueError("final certificate raw scalars do not pass")
    if policy.policy_sha256 == "":
        raise AssertionError("unreachable empty policy identity")


def _validate_quality_replay(
    history: HistoryEvidence, terminal: TerminalEvidence, policy: PolicyEvidence
) -> None:
    arrays = {name: evidence.values for name, evidence in terminal.arrays}
    mask = arrays["accepted_mask"]
    replay_mask = arrays["accepted_quality_mask"]
    replay_raw = arrays["accepted_quality_raw_equalities"]
    replay_scaled = arrays["accepted_quality_scaled_equalities"]
    replay_objectives = arrays["accepted_quality_objectives"]
    inverse_scale = arrays["constraint_inverse_scale"]
    expected_scaled = replay_raw * inverse_scale[None, :]
    if not np.array_equal(replay_scaled, expected_scaled, equal_nan=True):
        raise ValueError("accepted quality replay scaled equalities differ from D*q")
    coordinates_finite = np.all(
        np.isfinite(arrays["accepted_optimizer_ledger"]), axis=1
    )
    objective_finite = np.isfinite(replay_objectives)
    raw_finite = np.all(np.isfinite(replay_raw), axis=1)
    scaled_finite = np.all(np.isfinite(replay_scaled), axis=1)
    last = history.accepted_steps
    if (
        replay_objectives[last] != terminal.objective
        or not np.array_equal(replay_raw[last], arrays["raw_equalities"])
        or not np.array_equal(replay_scaled[last], arrays["scaled_equalities"])
    ):
        raise ValueError("terminal quality differs from the last accepted replay row")
    bounds = (
        np.abs(arrays["native_equalities"])
        + policy.component_absolute_tolerance
        + policy.component_relative_tolerance * np.abs(arrays["native_equalities"])
    )
    objective_satisfied = objective_finite & (
        replay_objectives <= policy.objective_target
    )
    component_satisfied = raw_finite & np.all(
        np.abs(replay_raw) <= bounds[None, :], axis=1
    )
    scaled_satisfied = scaled_finite & (
        np.max(np.abs(replay_scaled), axis=1) <= policy.scaled_feasibility_tolerance
    )
    quality = replay_mask & objective_satisfied & component_satisfied & scaled_satisfied
    replay_claims = {
        "accepted_quality_mask": mask,
        "accepted_quality_coordinates_finite": coordinates_finite,
        "accepted_quality_objective_finite": objective_finite,
        "accepted_quality_raw_equalities_finite": raw_finite,
        "accepted_quality_scaled_equalities_finite": scaled_finite,
        "accepted_quality_objective_satisfied": objective_satisfied,
        "accepted_quality_component_bounds_satisfied": component_satisfied,
        "accepted_quality_scaled_feasibility_satisfied": scaled_satisfied,
        "accepted_quality_satisfied": quality,
    }
    if any(
        not np.array_equal(arrays[name], expected)
        for name, expected in replay_claims.items()
    ):
        raise ValueError("device accepted-quality predicates differ from raw replay")
    active = slice(0, history.accepted_steps + 1)
    if not (
        np.all(coordinates_finite[active])
        and np.all(objective_finite[active])
        and np.all(raw_finite[active])
        and np.all(scaled_finite[active])
    ):
        raise ValueError("active accepted-quality replay contains nonfinite evidence")
    accepted_hits = np.flatnonzero(quality[1 : history.accepted_steps + 1]) + 1
    latch = accepted_hits.size > 0
    first_accepted = int(accepted_hits[0]) if latch else 0
    first_attempt = 0
    if latch:
        first_attempt = next(
            index + 1
            for index, row in enumerate(history.rows[: history.attempts])
            if row.accepted_step_number == first_accepted
        )
    if (
        latch != history.quality_latch
        or first_accepted != history.first_quality_accepted_step
        or first_attempt != history.first_quality_attempt
    ):
        raise ValueError(
            "device quality latch/counters differ from raw accepted replay"
        )


def _union(intervals: tuple[Interval, ...]) -> tuple[tuple[Interval, ...], int]:
    if not intervals:
        return (), 0
    ordered = sorted(intervals, key=lambda item: (item.start_ns, item.end_ns))
    merged: list[Interval] = []
    start, end = ordered[0].start_ns, ordered[0].end_ns
    for interval in ordered[1:]:
        if interval.start_ns <= end:
            end = max(end, interval.end_ns)
        else:
            merged.append(Interval(start, end))
            start, end = interval.start_ns, interval.end_ns
    merged.append(Interval(start, end))
    return tuple(merged), sum(item.duration_ns for item in merged)


def _intersection_ns(left: tuple[Interval, ...], right: tuple[Interval, ...]) -> int:
    return sum(
        max(0, min(a.end_ns, b.end_ns) - max(a.start_ns, b.start_ns))
        for a in left
        for b in right
    )


def _parse_phases(value: JsonValue, execution_phase_schema: str) -> PhaseEvidence:
    payload = _mapping(value, "raw profiler trace")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "phase_schema_sha256",
                "trace_start_ns",
                "trace_stop_ns",
                "device_intervals",
            }
        ),
        "raw profiler trace",
    )
    if payload["schema_version"] != f"{SCHEMA_VERSION}-raw-trace":
        raise ValueError("raw trace schema differs")
    phase_schema = _sha256(payload["phase_schema_sha256"], "trace phase schema")
    if phase_schema != PHASE_SCHEMA_SHA256 or phase_schema != execution_phase_schema:
        raise ValueError("preflight/cold trace phase schema differs")
    trace_start = _integer(payload["trace_start_ns"], "trace-local start")
    trace_stop = _integer(payload["trace_stop_ns"], "trace-local stop")
    if trace_stop <= trace_start:
        raise ValueError("trace-local device envelope is invalid")
    all_intervals: list[Interval] = []
    assigned: dict[str, list[Interval]] = {phase: [] for phase in PHASE_IDS}
    ambiguous = False
    for index, item in enumerate(
        _array(payload["device_intervals"], "trace.device_intervals")
    ):
        context = f"trace.device_intervals[{index}]"
        row = _mapping(item, context)
        _exact_keys(row, frozenset({"start_ns", "end_ns", "scope_paths"}), context)
        interval = Interval(
            _integer(row["start_ns"], f"{context}.start_ns"),
            _integer(row["end_ns"], f"{context}.end_ns"),
        )
        if (
            interval.end_ns <= interval.start_ns
            or interval.start_ns < trace_start
            or interval.end_ns > trace_stop
        ):
            raise ValueError(f"{context} is outside the solve envelope")
        all_intervals.append(interval)
        candidates: list[tuple[int, str]] = []
        for raw_path in _array(row["scope_paths"], f"{context}.scope_paths"):
            path = _array(raw_path, f"{context}.scope_path")
            for depth, raw_scope in enumerate(path, start=1):
                scope = _string(raw_scope, f"{context}.scope")
                if scope in PHASE_IDS:
                    candidates.append((depth, scope))
        if not candidates:
            continue
        deepest = max(depth for depth, _scope in candidates)
        owners = frozenset(scope for depth, scope in candidates if depth == deepest)
        if len(owners) != 1:
            ambiguous = True
            continue
        assigned[next(iter(owners))].append(interval)
    device_union, device_ns = _union(tuple(all_intervals))
    if not all_intervals:
        raise ValueError("raw trace has no device intervals")
    phase_unions = {name: _union(tuple(rows))[0] for name, rows in assigned.items()}
    durations = tuple(
        (name, sum(item.duration_ns for item in phase_unions[name]))
        for name in PHASE_IDS
    )
    attributed_intervals = tuple(
        interval for name in PHASE_IDS for interval in phase_unions[name]
    )
    _attributed_union, attributed_ns = _union(attributed_intervals)
    overlaps = tuple(
        (left, right, _intersection_ns(phase_unions[left], phase_unions[right]))
        for left_index, left in enumerate(PHASE_IDS)
        for right in PHASE_IDS[left_index + 1 :]
    )
    current_intervals = tuple(
        interval
        for name in PHASE_IDS
        if name in CURRENT_MODEL_PHASES
        for interval in phase_unions[name]
    )
    _current_union, current_ns = _union(current_intervals)
    coverage = attributed_ns / device_ns if device_ns > 0 else 0.0
    status = (
        PhaseTimingStatus.PRODUCED
        if not ambiguous
        and device_ns > 0
        and all(dict(durations)[phase] > 0 for phase in PHASE_IDS)
        and coverage >= MINIMUM_PHASE_COVERAGE
        and 0 <= current_ns <= attributed_ns
        else PhaseTimingStatus.NOT_PRODUCED
    )
    del device_union
    return PhaseEvidence(
        status,
        durations,
        overlaps,
        device_ns,
        attributed_ns,
        device_ns - attributed_ns,
        current_ns,
        coverage,
        trace_start,
        trace_stop,
    )


def _parse_execution(
    value: JsonValue,
    refs: Mapping[str, ArtifactRef],
    *,
    expected_schema_version: str = f"{SCHEMA_VERSION}-execution",
) -> ExecutionEvidence:
    payload = _mapping(value, "execution evidence")
    keys = frozenset(
        {
            "schema_version",
            "supporting_evidence",
            "preflight_status",
            "preflight_compile_success",
            "preflight_solver_dispatched",
            "preflight_finalizer_called",
            "preflight_endpoint_audit_called",
            "preflight_campaign_authorized",
            "preflight_callbacks",
            "cold_status",
            "child_pid",
            "child_start_time_ticks",
            "backend",
            "gpu_uuid",
            "jax_enable_x64",
            "state_size",
            "equality_size",
            "residual_size",
            "policy_sha256",
            "phase_schema_sha256",
            "source_pre_sha256",
            "source_post_sha256",
            "runtime_environment_sha256",
            "interpreter",
            "argv",
            "physical_memory_bytes",
            "peak_memory_bytes",
            "peak_memory_fraction",
            "hot_h2d_transfers",
            "hot_d2h_transfers",
            "python_callbacks",
            "final_d2h_transfers",
            "timestamps_ns",
            "stdout_sha256",
            "stdout_size_bytes",
            "stderr_sha256",
            "stderr_size_bytes",
        }
    )
    _exact_keys(payload, keys, "execution evidence")
    if payload["schema_version"] != expected_schema_version:
        raise ValueError("execution schema differs")
    if (
        _sha256(payload["phase_schema_sha256"], "phase schema SHA")
        != PHASE_SCHEMA_SHA256
    ):
        raise ValueError("execution phase schema differs from the frozen phases")
    supporting_payload = _mapping(payload["supporting_evidence"], "supporting evidence")
    supporting_keys = EVIDENCE_REF_KEYS - frozenset(
        {"history", "terminal_numerical", "raw_trace", "trace_intervals", "execution"}
    )
    _exact_keys(supporting_payload, supporting_keys, "supporting evidence")
    supporting = tuple(
        (name, _artifact_ref(supporting_payload[name], f"supporting evidence.{name}"))
        for name in sorted(supporting_keys)
    )
    if any(reference != refs[name] for name, reference in supporting):
        raise ValueError("execution supporting references differ from receipt")
    raw_timestamps = _mapping(payload["timestamps_ns"], "execution timestamps")
    timestamps = tuple(
        (name, _integer(raw_timestamps[name], f"timestamps.{name}"))
        for name in sorted(raw_timestamps)
    )
    argv = tuple(
        _string(item, "execution.argv")
        for item in _array(payload["argv"], "execution.argv")
    )
    return ExecutionEvidence(
        supporting,
        _string(payload["preflight_status"], "preflight status"),
        _boolean(payload["preflight_compile_success"], "preflight compile"),
        _boolean(payload["preflight_solver_dispatched"], "preflight dispatch"),
        _boolean(payload["preflight_finalizer_called"], "preflight finalizer"),
        _boolean(payload["preflight_endpoint_audit_called"], "preflight audit"),
        _boolean(payload["preflight_campaign_authorized"], "preflight authorization"),
        _integer(payload["preflight_callbacks"], "preflight callbacks"),
        _string(payload["cold_status"], "cold status"),
        _integer(payload["child_pid"], "child pid"),
        _integer(payload["child_start_time_ticks"], "child start ticks"),
        _string(payload["backend"], "backend"),
        _string(payload["gpu_uuid"], "GPU UUID"),
        _boolean(payload["jax_enable_x64"], "JAX x64"),
        _integer(payload["state_size"], "state size"),
        _integer(payload["equality_size"], "equality size"),
        _integer(payload["residual_size"], "residual size"),
        _sha256(payload["policy_sha256"], "policy SHA"),
        _sha256(payload["phase_schema_sha256"], "phase schema SHA"),
        _sha256(payload["source_pre_sha256"], "source pre SHA"),
        _sha256(payload["source_post_sha256"], "source post SHA"),
        _sha256(payload["runtime_environment_sha256"], "runtime environment SHA"),
        _string(payload["interpreter"], "interpreter"),
        argv,
        _integer(payload["physical_memory_bytes"], "physical memory"),
        _integer(payload["peak_memory_bytes"], "peak memory"),
        _number(payload["peak_memory_fraction"], "peak memory fraction"),
        _integer(payload["hot_h2d_transfers"], "hot H2D"),
        _integer(payload["hot_d2h_transfers"], "hot D2H"),
        _integer(payload["python_callbacks"], "Python callbacks"),
        _integer(payload["final_d2h_transfers"], "final D2H"),
        timestamps,
        _sha256(payload["stdout_sha256"], "stdout SHA"),
        _integer(payload["stdout_size_bytes"], "stdout size"),
        _sha256(payload["stderr_sha256"], "stderr SHA"),
        _integer(payload["stderr_size_bytes"], "stderr size"),
    )


def _terminal_semantics(history: HistoryEvidence, terminal: TerminalEvidence) -> bool:
    mask = terminal.array("accepted_mask").values
    expected_mask = np.arange(LEDGER_SIZE) < history.accepted_steps + 1
    if not np.array_equal(mask, expected_mask):
        return False
    last = history.accepted_steps
    return bool(
        np.array_equal(
            terminal.array("accepted_optimizer_ledger").values[last],
            terminal.array("optimizer_coordinates").values,
        )
        and np.array_equal(
            terminal.array("accepted_physical_ledger").values[last],
            terminal.array("physical_state").values,
        )
        and terminal.final_certificate_passes
        and terminal.terminal_endpoint_diagnostics_seconds >= 0.0
    )


def _derive(
    refs: tuple[tuple[str, ArtifactRef], ...],
    policy: PolicyEvidence,
    history: HistoryEvidence,
    terminal: TerminalEvidence,
    phases: PhaseEvidence,
    execution: ExecutionEvidence,
) -> DiagnosticReceipt:
    _validate_terminal_raw_evidence(terminal, history, policy)
    _validate_quality_replay(history, terminal, policy)
    quality = _quality(terminal)
    numerical_complete = bool(
        not history.fatal
        and history.attempts > 0
        and (
            history.attempts == MAXIMUM_ATTEMPTS
            or history.accepted_steps == MAXIMUM_ACCEPTED_STEPS
            or history.quality_latch
        )
        and _terminal_semantics(history, terminal)
        and quality.residual_value_margin >= 0.0
        and quality.residual_gradient_margin >= 0.0
        and quality.transpose_margin >= 0.0
    )
    complete = bool(
        numerical_complete
        and execution.passes()
        and phases.status is PhaseTimingStatus.PRODUCED
    )
    hit = bool(
        history.quality_latch
        and quality.passes
        and history.first_quality_attempt > 0
        and history.first_quality_accepted_step > 0
    )
    if not complete:
        verdict = DiagnosticVerdict.INCOMPLETE
    elif hit:
        verdict = DiagnosticVerdict.QUALITY_HIT
    elif not history.quality_latch:
        verdict = DiagnosticVerdict.NO_HIT
    else:
        verdict = DiagnosticVerdict.INCOMPLETE
    relation = (
        HistoricalAggregateRelation.MATCHES
        if history.attempts == 300
        and history.accepted_steps == 203
        and not history.quality_latch
        else HistoricalAggregateRelation.DIVERGES
    )
    reuse = (
        history.retryable_rejections
        / history.attempts
        * phases.current_model_ns
        / phases.total_attributed_ns
        if complete and history.attempts > 0 and phases.total_attributed_ns > 0
        else None
    )
    retraction_retries = sum(
        row.outcome in RETRACTION_OUTCOMES for row in history.rows[: history.attempts]
    )
    reuse_selected = bool(
        complete
        and history.attempts > 0
        and phases.total_attributed_ns > 0
        and 20 * history.retryable_rejections * phases.current_model_ns
        >= history.attempts * phases.total_attributed_ns
    )
    retraction_selected = bool(
        history.attempts > 0 and 10 * retraction_retries >= history.attempts
    )
    if not complete:
        next_route = NextRoute.NOT_SELECTED
    elif reuse_selected:
        next_route = NextRoute.RETRY_MODEL_REUSE
    elif (
        quality.minimum_component_margin < 0.0
        or quality.scaled_feasibility_margin < 0.0
        or retraction_selected
    ):
        next_route = NextRoute.RADIUS_RETRACTION
    else:
        next_route = NextRoute.CONDITIONING_MODEL_CHANGE
    return DiagnosticReceipt(
        refs,
        policy,
        history,
        terminal,
        quality,
        phases,
        execution,
        verdict,
        relation,
        next_route,
        reuse,
    )


@dataclass(frozen=True, slots=True)
class _Diag5HeldEntry:
    descriptor: int
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    is_directory: bool
    sha256: str | None


class _Diag5HeldTree:
    def __init__(self, root: Path, *, require_sealed: bool) -> None:
        self.root = root
        self.require_sealed = require_sealed
        self.root_descriptor = -1
        self._root_identity: tuple[int, int, int, int] | None = None
        self.entries: dict[str, _Diag5HeldEntry] = {}
        self._descriptors: list[int] = []
        self._validation_directory: tempfile.TemporaryDirectory[str] | None = None
        self._materialized_validation_paths: set[str] = set()

    def __enter__(self) -> Self:
        try:
            self.root_descriptor = os.open(
                self.root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            self._descriptors.append(self.root_descriptor)
            root_stat = os.fstat(self.root_descriptor)
            path_stat = self.root.lstat()
            root_identity = (
                root_stat.st_dev,
                root_stat.st_ino,
                stat.S_IMODE(root_stat.st_mode),
                root_stat.st_nlink,
            )
            if root_identity != (
                path_stat.st_dev,
                path_stat.st_ino,
                stat.S_IMODE(path_stat.st_mode),
                path_stat.st_nlink,
            ):
                raise ValueError(
                    "DIAG5 artifact root changed during descriptor binding"
                )
            if self.require_sealed and root_identity[2] != 0o555:
                raise ValueError("DIAG5 artifact root mode differs")
            self._root_identity = root_identity
            self._scan_directory(self.root_descriptor, "")
            rebound_root = os.fstat(self.root_descriptor)
            rebound_path = self.root.lstat()
            if (
                rebound_root.st_dev,
                rebound_root.st_ino,
                stat.S_IMODE(rebound_root.st_mode),
                rebound_root.st_nlink,
            ) != root_identity or (
                rebound_path.st_dev,
                rebound_path.st_ino,
                stat.S_IMODE(rebound_path.st_mode),
                rebound_path.st_nlink,
            ) != root_identity:
                raise ValueError("DIAG5 artifact root changed during descriptor scan")
            self._revalidate_namespace()
        except BaseException:
            self._close_descriptors()
            raise
        return self

    def _close_descriptors(self) -> None:
        for descriptor in reversed(self._descriptors):
            os.close(descriptor)
        self._descriptors.clear()
        self.root_descriptor = -1

    def __exit__(self, *_exc: object) -> None:
        if self._validation_directory is not None:
            validation_root = Path(self._validation_directory.name)
            for path in sorted(
                validation_root.rglob("*"),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                path.chmod(0o700 if path.is_dir() else 0o600)
            validation_root.chmod(0o700)
            self._validation_directory.cleanup()
            self._validation_directory = None
        self._close_descriptors()

    def _scan_directory(self, directory_descriptor: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_descriptor)):
            relative = f"{prefix}/{name}" if prefix else name
            before = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(before.st_mode):
                raise ValueError("DIAG5 artifact contains a symlink")
            if stat.S_ISDIR(before.st_mode):
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
                self._descriptors.append(descriptor)
                after = os.fstat(descriptor)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise ValueError(
                        "DIAG5 directory changed during descriptor binding"
                    )
                mode = stat.S_IMODE(after.st_mode)
                if self.require_sealed and mode != 0o555:
                    raise ValueError("DIAG5 artifact directory mode differs")
                self.entries[relative] = _Diag5HeldEntry(
                    descriptor,
                    after.st_dev,
                    after.st_ino,
                    mode,
                    after.st_nlink,
                    after.st_size,
                    True,
                    None,
                )
                self._scan_directory(descriptor, relative)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("DIAG5 artifact contains a special file")
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            self._descriptors.append(descriptor)
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError("DIAG5 file changed during descriptor binding")
            mode = stat.S_IMODE(after.st_mode)
            if after.st_nlink != 1:
                raise ValueError("DIAG5 artifact contains hardlink ambiguity")
            if self.require_sealed and mode != 0o444:
                raise ValueError("DIAG5 artifact file mode differs")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            hashed = os.fstat(descriptor)
            if (after.st_dev, after.st_ino, after.st_size, after.st_nlink) != (
                hashed.st_dev,
                hashed.st_ino,
                hashed.st_size,
                hashed.st_nlink,
            ) or stat.S_IMODE(after.st_mode) != stat.S_IMODE(hashed.st_mode):
                raise ValueError("DIAG5 file changed during descriptor hashing")
            self.entries[relative] = _Diag5HeldEntry(
                descriptor,
                after.st_dev,
                after.st_ino,
                mode,
                after.st_nlink,
                after.st_size,
                False,
                digest.hexdigest(),
            )

    def file_bytes(self, relative: str) -> bytes:
        entry = self.entries.get(relative)
        if entry is None or entry.is_directory:
            raise ValueError(f"DIAG5 held file is absent: {relative}")
        os.lseek(entry.descriptor, 0, os.SEEK_SET)
        data = bytearray()
        while chunk := os.read(entry.descriptor, 1024 * 1024):
            data.extend(chunk)
        after = os.fstat(entry.descriptor)
        if (
            len(data) != entry.size_bytes
            or after.st_size != entry.size_bytes
            or (after.st_dev, after.st_ino) != (entry.device, entry.inode)
            or stat.S_IMODE(after.st_mode) != entry.mode
            or after.st_nlink != entry.link_count
            or hashlib.sha256(data).hexdigest() != entry.sha256
        ):
            raise ValueError("DIAG5 file changed while reading held bytes")
        return bytes(data)

    def _copy_file_to_validation_tree(
        self, relative: str, target: Path, entry: _Diag5HeldEntry
    ) -> None:
        os.lseek(entry.descriptor, 0, os.SEEK_SET)
        copied = 0
        digest = hashlib.sha256()
        with target.open("wb") as stream:
            while chunk := os.read(entry.descriptor, 1024 * 1024):
                stream.write(chunk)
                copied += len(chunk)
                digest.update(chunk)
        after = os.fstat(entry.descriptor)
        if (
            copied != entry.size_bytes
            or after.st_size != entry.size_bytes
            or (after.st_dev, after.st_ino) != (entry.device, entry.inode)
            or stat.S_IMODE(after.st_mode) != entry.mode
            or after.st_nlink != entry.link_count
            or digest.hexdigest() != entry.sha256
        ):
            raise ValueError(
                f"DIAG5 file changed while materializing held bytes: {relative}"
            )
        target.chmod(entry.mode)

    def descriptor_path(self, relative: str = "") -> Path:
        entry = self.entries.get(relative) if relative else None
        descriptor = (
            self.root_descriptor
            if relative == ""
            else (entry.descriptor if entry is not None else -1)
        )
        if descriptor < 0:
            raise ValueError(f"DIAG5 held descriptor is absent: {relative}")
        return Path(f"/proc/self/fd/{descriptor}")

    def validation_path(self, relative: str = "") -> Path:
        """Return an immutable semantic mirror built only from retained bytes."""

        if self._validation_directory is None:
            self._validation_directory = tempfile.TemporaryDirectory(
                prefix="diag5-held-validation-"
            )
        validation_root = Path(self._validation_directory.name)
        if relative:
            selected = self.entries.get(relative)
            if selected is None:
                raise ValueError(f"DIAG5 held validation path is absent: {relative}")
            selected_paths = (
                tuple(
                    path
                    for path in self.entries
                    if path == relative or path.startswith(f"{relative}/")
                )
                if selected.is_directory
                else (relative,)
            )
            for path in sorted(selected_paths, key=lambda item: item.count("/")):
                if path in self._materialized_validation_paths:
                    continue
                entry = self.entries[path]
                target = validation_root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if entry.is_directory:
                    target.mkdir(exist_ok=True)
                else:
                    self._copy_file_to_validation_tree(path, target, entry)
                self._materialized_validation_paths.add(path)
            for path in sorted(
                selected_paths, key=lambda item: item.count("/"), reverse=True
            ):
                entry = self.entries[path]
                if entry.is_directory:
                    (validation_root / path).chmod(entry.mode)
        return validation_root / relative

    def revalidate_path_bindings(self) -> None:
        if self._root_identity is None:
            raise ValueError("DIAG5 artifact root was not descriptor-bound")
        root_stat = self.root.lstat()
        held_root = os.fstat(self.root_descriptor)
        if (
            root_stat.st_dev,
            root_stat.st_ino,
            stat.S_IMODE(root_stat.st_mode),
            root_stat.st_nlink,
        ) != self._root_identity or (
            held_root.st_dev,
            held_root.st_ino,
            stat.S_IMODE(held_root.st_mode),
            held_root.st_nlink,
        ) != self._root_identity:
            raise ValueError("DIAG5 artifact root changed during validation")
        for relative, entry in self.entries.items():
            observed = (self.root / relative).lstat()
            held = os.fstat(entry.descriptor)
            if (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_nlink,
                stat.S_IMODE(observed.st_mode),
            ) != (
                entry.device,
                entry.inode,
                entry.size_bytes,
                entry.link_count,
                entry.mode,
            ) or (
                held.st_dev,
                held.st_ino,
                held.st_size,
                held.st_nlink,
                stat.S_IMODE(held.st_mode),
            ) != (
                entry.device,
                entry.inode,
                entry.size_bytes,
                entry.link_count,
                entry.mode,
            ):
                raise ValueError("DIAG5 artifact path changed during validation")
            if not entry.is_directory:
                os.lseek(entry.descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                while chunk := os.read(entry.descriptor, 1024 * 1024):
                    digest.update(chunk)
                after_hash = os.fstat(entry.descriptor)
                if digest.hexdigest() != entry.sha256 or (
                    after_hash.st_dev,
                    after_hash.st_ino,
                    after_hash.st_size,
                    after_hash.st_nlink,
                    stat.S_IMODE(after_hash.st_mode),
                ) != (
                    entry.device,
                    entry.inode,
                    entry.size_bytes,
                    entry.link_count,
                    entry.mode,
                ):
                    raise ValueError("DIAG5 artifact content changed during validation")
        self._revalidate_namespace()

    def _revalidate_namespace(self) -> None:
        expected_children: dict[str, set[str]] = {"": set()}
        for relative, entry in self.entries.items():
            parent, _, name = relative.rpartition("/")
            expected_children.setdefault(parent, set()).add(name)
            if entry.is_directory:
                expected_children.setdefault(relative, set())
        directories = {"": self.root_descriptor}
        directories.update(
            {
                relative: entry.descriptor
                for relative, entry in self.entries.items()
                if entry.is_directory
            }
        )
        for relative, descriptor in directories.items():
            if set(os.listdir(descriptor)) != expected_children[relative]:
                raise ValueError("DIAG5 artifact namespace changed during validation")


_DIAG5_HELD_TREE: ContextVar[_Diag5HeldTree | None] = ContextVar(
    "diag5_held_tree", default=None
)


def _diag5_held_file_bytes(root: Path, relative: str) -> bytes:
    held = _DIAG5_HELD_TREE.get()
    return (
        held.file_bytes(relative)
        if held is not None
        and root.resolve(strict=True) == held.root.resolve(strict=True)
        else (root / relative).read_bytes()
    )


def _diag5_held_path(root: Path, relative: str = "") -> Path:
    held = _DIAG5_HELD_TREE.get()
    return (
        held.validation_path(relative)
        if held is not None
        and root.resolve(strict=True) == held.root.resolve(strict=True)
        else root / relative
    )


def _load_ref_json(root: Path, reference: ArtifactRef, context: str) -> JsonValue:
    held = _DIAG5_HELD_TREE.get()
    if held is not None and root.resolve(strict=True) == held.root.resolve(strict=True):
        data = held.file_bytes(reference.relative_path)
        if (
            len(data) != reference.size_bytes
            or hashlib.sha256(data).hexdigest() != reference.sha256
        ):
            raise ValueError(f"{context} held bytes differ from reference")
        return load_canonical_json_bytes(data)
    path = _resolve_artifact(root, reference)
    return load_canonical_json_bytes(path.read_bytes())


def _validate_child_terminal(
    value: JsonValue,
    context: str,
    *,
    expected_schema_version: str = f"{SCHEMA_VERSION}-child-terminal",
) -> None:
    payload = _mapping(value, context)
    keys = {"schema_version", "terminal_status", "failure_reasons"}
    if expected_schema_version in {
        DIAG2_CHILD_TERMINAL_SCHEMA_VERSION,
        DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
    }:
        keys.add("monitor_failure_kind")
    _exact_keys(
        payload,
        frozenset(keys),
        context,
    )
    if (
        payload["schema_version"] != expected_schema_version
        or payload["terminal_status"] != "COMPLETE"
        or _array(payload["failure_reasons"], f"{context}.failure_reasons") != []
        or (
            expected_schema_version
            in {
                DIAG2_CHILD_TERMINAL_SCHEMA_VERSION,
                DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
            }
            and payload["monitor_failure_kind"] != "NONE"
        )
    ):
        raise ValueError(f"{context} does not prove successful termination")


def _validate_process(
    root: Path,
    value: JsonValue,
    *,
    expected_pid: int,
    expected_start_ticks: int,
    expected_argv: tuple[str, ...],
    expected_stdout_sha256: str,
    expected_stdout_size: int,
    expected_stderr_sha256: str,
    expected_stderr_size: int,
    expected_source_sha256: str,
    context: str,
    expected_schema_version: str = f"{SCHEMA_VERSION}-process",
    require_parent_monotonic_interval: bool = False,
) -> None:
    payload = _mapping(value, context)
    keys = {
        "schema_version",
        "child_pid",
        "child_start_time_ticks",
        "argv",
        "stdout",
        "stderr",
        "process_seconds",
        "process_diagnostics",
        "pre_source_identity",
        "post_source_identity",
    }
    if require_parent_monotonic_interval:
        keys.update({"process_started_monotonic_ns", "process_stopped_monotonic_ns"})
    if expected_schema_version in {
        DIAG2_PROCESS_SCHEMA_VERSION,
        DIAG5_PROCESS_SCHEMA_VERSION,
    }:
        keys.add("monitor_failure_kind")
    _exact_keys(
        payload,
        frozenset(keys),
        context,
    )
    if payload["schema_version"] != expected_schema_version:
        raise ValueError(f"{context} schema differs")
    if expected_schema_version in {
        DIAG2_PROCESS_SCHEMA_VERSION,
        DIAG5_PROCESS_SCHEMA_VERSION,
    }:
        monitor_kind = _string(
            payload["monitor_failure_kind"], f"{context}.monitor_failure_kind"
        )
        if monitor_kind not in {"NONE", "BINDING", "FINALIZATION"}:
            raise ValueError(f"{context}.monitor_failure_kind differs")
    if require_parent_monotonic_interval:
        process_started = _integer(
            payload["process_started_monotonic_ns"], f"{context}.process start"
        )
        process_stopped = _integer(
            payload["process_stopped_monotonic_ns"], f"{context}.process stop"
        )
        if process_started >= process_stopped:
            raise ValueError(f"{context} parent-monotonic interval differs")
    argv = tuple(
        _string(item, f"{context}.argv")
        for item in _array(payload["argv"], f"{context}.argv")
    )
    stdout = _artifact_ref(payload["stdout"], f"{context}.stdout")
    stderr = _artifact_ref(payload["stderr"], f"{context}.stderr")
    _resolve_artifact(root, stdout)
    _resolve_artifact(root, stderr)
    seconds = _number(payload["process_seconds"], f"{context}.process_seconds")
    _mapping(payload["process_diagnostics"], f"{context}.process_diagnostics")
    pre_source_payload = _mapping(
        payload["pre_source_identity"], f"{context}.pre_source_identity"
    )
    post_source_payload = _mapping(
        payload["post_source_identity"], f"{context}.post_source_identity"
    )
    source_keys = frozenset(
        {
            "git_head",
            "tracked_diff_sha256",
            "untracked_bytes_manifest_sha256",
            "source_manifest_sha256",
            "source_manifest_size_bytes",
        }
    )
    _exact_keys(pre_source_payload, source_keys, f"{context}.pre_source_identity")
    _exact_keys(post_source_payload, source_keys, f"{context}.post_source_identity")
    if (
        _integer(payload["child_pid"], f"{context}.child_pid") != expected_pid
        or _integer(
            payload["child_start_time_ticks"], f"{context}.child_start_time_ticks"
        )
        != expected_start_ticks
        or argv != expected_argv
        or seconds <= 0.0
        or stdout.sha256 != expected_stdout_sha256
        or stdout.size_bytes != expected_stdout_size
        or stderr.sha256 != expected_stderr_sha256
        or stderr.size_bytes != expected_stderr_size
        or pre_source_payload != post_source_payload
        or hashlib.sha256(canonical_json_bytes(pre_source_payload)).hexdigest()
        != expected_source_sha256
        or hashlib.sha256(canonical_json_bytes(post_source_payload)).hexdigest()
        != expected_source_sha256
    ):
        raise ValueError(f"{context} differs from independently bound execution")


def _validate_memory(
    memory_value: JsonValue,
    samples_value: JsonValue,
    *,
    expected_pid: int,
    expected_start_ticks: int,
    expected_argv: tuple[str, ...],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
    expected_peak_bytes: int,
    expected_peak_fraction: float,
    context: str,
    expected_memory_schema_version: str = "single-stage-neq-gntr1-memory-v1",
    expected_samples_schema_version: str = f"{SCHEMA_VERSION}-memory-samples",
) -> None:
    memory = _mapping(memory_value, context)
    _exact_keys(
        memory,
        frozenset(
            {
                "schema_version",
                "monitor_scope",
                "parent_pid",
                "child_pid",
                "child_start_time_ticks",
                "child_argv_sha256",
                "device_uuid",
                "sample_count",
                "peak_memory_bytes",
                "peak_memory_fraction",
            }
        ),
        context,
    )
    if (
        memory["schema_version"] != expected_memory_schema_version
        or memory["monitor_scope"] != "whole-child-exact-pid-exact-device"
    ):
        raise ValueError(f"{context} identity differs")
    samples = _mapping(samples_value, f"{context} samples")
    _exact_keys(
        samples,
        frozenset({"schema_version", "samples"}),
        f"{context} samples",
    )
    if samples["schema_version"] != expected_samples_schema_version:
        raise ValueError(f"{context} sample schema differs")
    raw_samples = _array(samples["samples"], f"{context}.samples")
    parsed_samples: list[tuple[int, int]] = []
    for index, item in enumerate(raw_samples):
        row_context = f"{context}.samples[{index}]"
        row = _mapping(item, row_context)
        _exact_keys(
            row,
            frozenset({"sampled_at_unix_ns", "used_memory_mib"}),
            row_context,
        )
        parsed_samples.append(
            (
                _integer(row["sampled_at_unix_ns"], f"{row_context}.time"),
                _integer(row["used_memory_mib"], f"{row_context}.memory"),
            )
        )
    if not parsed_samples or any(
        left[0] >= right[0]
        for left, right in zip(parsed_samples[:-1], parsed_samples[1:], strict=True)
    ):
        raise ValueError(f"{context} samples are absent or unordered")
    peak = max(memory_mib for _sampled_at, memory_mib in parsed_samples) * 1024 * 1024
    reported_fraction = _number(
        memory["peak_memory_fraction"], f"{context}.peak_memory_fraction"
    )
    expected_argv_sha256 = hashlib.sha256(
        canonical_json_bytes(list(expected_argv[2:]))
    ).hexdigest()
    if (
        _integer(memory["parent_pid"], f"{context}.parent_pid") <= 0
        or _integer(memory["child_pid"], f"{context}.child_pid") != expected_pid
        or _integer(
            memory["child_start_time_ticks"], f"{context}.child_start_time_ticks"
        )
        != expected_start_ticks
        or _sha256(memory["child_argv_sha256"], f"{context}.child_argv_sha256")
        != expected_argv_sha256
        or memory["device_uuid"] != expected_gpu_uuid
        or _integer(memory["sample_count"], f"{context}.sample_count")
        != len(parsed_samples)
        or _integer(memory["peak_memory_bytes"], f"{context}.peak_memory_bytes") != peak
        or peak != expected_peak_bytes
        or not math.isclose(
            reported_fraction,
            peak / physical_memory_bytes,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            reported_fraction,
            expected_peak_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError(f"{context} differs from raw samples or execution")


def _runtime_mapping(value: JsonValue, context: str) -> dict[str, JsonValue]:
    payload = _mapping(value, context)
    _exact_keys(
        payload,
        frozenset(
            {
                "backend",
                "device",
                "device_uuid",
                "jax",
                "jax_enable_x64",
                "jaxlib",
                "python",
            }
        ),
        context,
    )
    return payload


def _validate_diagnostic_environment(
    environment: tuple[tuple[str, str | None], ...], context: str
) -> None:
    observed = dict(environment)
    expected = {
        "JAX_COMPILATION_CACHE_DIR": None,
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS": "67108864",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }
    if any(observed.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{context} diagnostic environment differs")


def _source_observation_sha256(snapshot: SnapshotPublication) -> str:
    observed_roles = {entry.relative_path: entry.role for entry in snapshot.entries}
    if any(
        observed_roles.get(path) != role
        for path, role in REQUIRED_SOURCE_ROLE_BINDINGS.items()
    ):
        raise ValueError("source snapshot diagnostic path/role binding differs")
    manifest = snapshot.manifest_path.read_bytes()
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "git_head": snapshot.worktree.git_head,
                "source_manifest_sha256": snapshot.manifest_sha256,
                "source_manifest_size_bytes": len(manifest),
                "tracked_diff_sha256": snapshot.worktree.tracked_diff_sha256,
                "untracked_bytes_manifest_sha256": (
                    snapshot.worktree.untracked_bytes_manifest_sha256
                ),
            }
        )
    ).hexdigest()


def validate_diagnostic_preflight_gate(
    artifact_root: Path,
    *,
    evidence_refs: Mapping[str, ArtifactRef],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
    expected_interpreter: str,
    expected_argv: tuple[str, ...],
    expected_route: str = ROUTE,
    expected_plan_sha256: str = PLAN_SHA256,
    policy_authority_override: PolicyEvidence | None = None,
    process_schema_version: str = f"{SCHEMA_VERSION}-process",
    child_terminal_schema_version: str = f"{SCHEMA_VERSION}-child-terminal",
    require_parent_monotonic_interval: bool = False,
) -> bool:
    """Independently authorize the sole cold child from retained raw evidence."""

    if frozenset(evidence_refs) != PREFLIGHT_EVIDENCE_REF_KEYS:
        raise ValueError("preflight evidence references differ from the frozen schema")
    for reference in evidence_refs.values():
        _resolve_artifact(artifact_root, reference)
    snapshot = load_snapshot(artifact_root / "source-snapshot")
    expected_source_sha256 = _source_observation_sha256(snapshot)
    if (
        evidence_refs["source_manifest"].sha256 != snapshot.manifest_sha256
        or _resolve_artifact(artifact_root, evidence_refs["source_manifest"])
        != snapshot.manifest_path
    ):
        raise ValueError("preflight source reference differs from validated snapshot")
    validate_native_equivalent_reference(artifact_root / "native-reference")
    if (
        _resolve_artifact(artifact_root, evidence_refs["native_reference"])
        != artifact_root.resolve(strict=True) / "native-reference" / "reference.json"
    ):
        raise ValueError("preflight native reference path differs")
    runtime = validate_runtime_evidence(
        _resolve_artifact(artifact_root, evidence_refs["runtime"]),
        snapshot_root=snapshot.root,
        campaign_root=artifact_root,
    )
    identity = runtime.observation.runtime_identity
    _validate_diagnostic_environment(
        runtime.observation.effective_environment, "preflight"
    )
    if (
        identity.backend != "gpu"
        or identity.device_uuid != expected_gpu_uuid
        or identity.python_executable != expected_interpreter
    ):
        raise ValueError("preflight runtime identity differs")
    producer = _mapping(
        _load_ref_json(artifact_root, evidence_refs["producer"], "preflight producer"),
        "preflight producer",
    )
    _exact_keys(
        producer,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "mode",
                "execution_status",
                "policy_sha256",
                "policy_evidence",
                "phase_schema_sha256",
                "state_size",
                "equality_size",
                "residual_size",
                "campaign_authorized",
                "solver_dispatched",
                "finalizer_called",
                "endpoint_audit_called",
                "python_callbacks",
                "runtime",
                "runtime_evidence",
                "timing",
                "failure_reasons",
            }
        ),
        "preflight producer",
    )
    runtime_mapping = _runtime_mapping(producer["runtime"], "preflight runtime")
    preflight_policy = _parse_policy(
        _load_ref_json(
            artifact_root,
            evidence_refs["preflight_policy"],
            "preflight policy",
        )
    )
    policy_authority = (
        policy_authority_override
        if policy_authority_override is not None
        else _parse_policy(
            _load_ref_json(
                artifact_root,
                evidence_refs["policy_authority"],
                "parent policy authority",
            )
        )
    )
    native_authority = _validated_native_equalities(artifact_root)
    if preflight_policy != policy_authority or not np.array_equal(
        np.asarray(policy_authority.native_raw_equalities, dtype=np.dtype("<f8")),
        native_authority,
    ):
        raise ValueError("preflight policy differs from independent parent authority")
    timing = _mapping(producer["timing"], "preflight timing")
    _exact_keys(
        timing,
        frozenset(
            {
                "compile_started_ns",
                "compile_completed_ns",
                "process_seconds_before_serialization",
            }
        ),
        "preflight timing",
    )
    if (
        producer["schema_version"] != "single-stage-neq-gntr1-preflight-worker-v1"
        or producer["route"] != expected_route
        or producer["plan_sha256"] != expected_plan_sha256
        or producer["mode"] != "ANNOTATED_LOWER_COMPILE_ONLY"
        or producer["execution_status"] != "SUCCESS"
        or _sha256(producer["policy_sha256"], "preflight policy SHA")
        != preflight_policy.policy_sha256
        or _artifact_ref(producer["policy_evidence"], "preflight policy ref")
        != evidence_refs["preflight_policy"]
        or producer["phase_schema_sha256"] != PHASE_SCHEMA_SHA256
        or _integer(producer["state_size"], "preflight state size") != STATE_SIZE
        or _integer(producer["equality_size"], "preflight equality size")
        != EQUALITY_SIZE
        or _integer(producer["residual_size"], "preflight residual size") != 2110
        or producer["campaign_authorized"] is not False
        or producer["solver_dispatched"] is not False
        or producer["finalizer_called"] is not False
        or producer["endpoint_audit_called"] is not False
        or _integer(producer["python_callbacks"], "preflight callbacks") != 0
        or _artifact_ref(producer["runtime_evidence"], "preflight runtime ref")
        != evidence_refs["runtime"]
        or runtime_mapping["backend"] != "gpu"
        or runtime_mapping["device_uuid"] != expected_gpu_uuid
        or _boolean(runtime_mapping["jax_enable_x64"], "preflight x64") is not True
        or _array(producer["failure_reasons"], "preflight failures") != []
        or _integer(timing["compile_started_ns"], "preflight compile start")
        >= _integer(timing["compile_completed_ns"], "preflight compile stop")
        or _number(
            timing["process_seconds_before_serialization"],
            "preflight process seconds",
        )
        <= 0.0
    ):
        raise ValueError("preflight producer does not authorize cold execution")
    _validate_child_terminal(
        _load_ref_json(
            artifact_root, evidence_refs["child_terminal"], "preflight terminal"
        ),
        "preflight terminal",
        expected_schema_version=child_terminal_schema_version,
    )
    process = _mapping(
        _load_ref_json(artifact_root, evidence_refs["process"], "preflight process"),
        "preflight process",
    )
    pid = _integer(process.get("child_pid"), "preflight process.child_pid")
    start_ticks = _integer(
        process.get("child_start_time_ticks"), "preflight process.start_ticks"
    )
    stderr = _artifact_ref(process.get("stderr"), "preflight process.stderr")
    _validate_process(
        artifact_root,
        process,
        expected_pid=pid,
        expected_start_ticks=start_ticks,
        expected_argv=expected_argv,
        expected_stdout_sha256=evidence_refs["producer"].sha256,
        expected_stdout_size=evidence_refs["producer"].size_bytes,
        expected_stderr_sha256=stderr.sha256,
        expected_stderr_size=stderr.size_bytes,
        expected_source_sha256=expected_source_sha256,
        context="preflight process",
        expected_schema_version=process_schema_version,
        require_parent_monotonic_interval=require_parent_monotonic_interval,
    )
    memory = _mapping(
        _load_ref_json(artifact_root, evidence_refs["memory"], "preflight memory"),
        "preflight memory",
    )
    peak = _integer(memory.get("peak_memory_bytes"), "preflight memory peak")
    fraction = _number(memory.get("peak_memory_fraction"), "preflight memory fraction")
    _validate_memory(
        memory,
        _load_ref_json(
            artifact_root, evidence_refs["memory_samples"], "preflight memory samples"
        ),
        expected_pid=pid,
        expected_start_ticks=start_ticks,
        expected_argv=expected_argv,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
        expected_peak_bytes=peak,
        expected_peak_fraction=fraction,
        context="preflight memory",
    )
    if fraction >= 0.8:
        raise ValueError("preflight memory fraction does not authorize cold execution")
    return True


def _validate_execution_authorities(
    root: Path,
    refs: Mapping[str, ArtifactRef],
    execution: ExecutionEvidence,
    *,
    expected_route: str = ROUTE,
    expected_plan_sha256: str = PLAN_SHA256,
    policy_authority_override: PolicyEvidence | None = None,
) -> None:
    snapshot = load_snapshot(root / "source-snapshot")
    expected_source = _source_observation_sha256(snapshot)
    if (
        refs["source_manifest"].sha256 != snapshot.manifest_sha256
        or _resolve_artifact(root, refs["source_manifest"]) != snapshot.manifest_path
    ):
        raise ValueError("source manifest reference differs from validated snapshot")
    validate_native_equivalent_reference(root / "native-reference")
    if (
        _resolve_artifact(root, refs["native_reference"])
        != root.resolve(strict=True) / "native-reference" / "reference.json"
    ):
        raise ValueError("native reference path differs")

    raw_preflight_process = _mapping(
        _load_ref_json(root, refs["preflight_process"], "preflight process"),
        "preflight process",
    )
    preflight_argv = tuple(
        _string(item, "preflight process.argv")
        for item in _array(raw_preflight_process.get("argv"), "preflight process.argv")
    )
    validate_diagnostic_preflight_gate(
        root,
        evidence_refs={
            "producer": refs["preflight"],
            "child_terminal": refs["preflight_child_terminal"],
            "process": refs["preflight_process"],
            "memory": refs["preflight_memory"],
            "memory_samples": refs["preflight_memory_samples"],
            "runtime": refs["preflight_runtime"],
            "preflight_policy": refs["preflight_policy"],
            "policy_authority": refs["policy_authority"],
            "source_manifest": refs["source_manifest"],
            "native_reference": refs["native_reference"],
        },
        expected_gpu_uuid=execution.gpu_uuid,
        physical_memory_bytes=execution.physical_memory_bytes,
        expected_interpreter=execution.interpreter,
        expected_argv=preflight_argv,
        expected_route=expected_route,
        expected_plan_sha256=expected_plan_sha256,
        policy_authority_override=policy_authority_override,
        process_schema_version=(
            DIAG2_PROCESS_SCHEMA_VERSION
            if expected_route == DIAG2_ROUTE
            else f"{SCHEMA_VERSION}-process"
        ),
        child_terminal_schema_version=(
            DIAG2_CHILD_TERMINAL_SCHEMA_VERSION
            if expected_route == DIAG2_ROUTE
            else f"{SCHEMA_VERSION}-child-terminal"
        ),
        require_parent_monotonic_interval=expected_route == DIAG2_ROUTE,
    )

    cold_runtime = validate_runtime_evidence(
        _resolve_artifact(root, refs["runtime"]),
        snapshot_root=snapshot.root,
        campaign_root=root,
    )
    preflight_runtime = validate_runtime_evidence(
        _resolve_artifact(root, refs["preflight_runtime"]),
        snapshot_root=snapshot.root,
        campaign_root=root,
    )
    cold_identity = cold_runtime.observation.runtime_identity
    preflight_identity = preflight_runtime.observation.runtime_identity
    _validate_diagnostic_environment(
        cold_runtime.observation.effective_environment, "cold"
    )
    _validate_diagnostic_environment(
        preflight_runtime.observation.effective_environment, "preflight"
    )
    if (
        cold_identity.effective_environment_sha256
        != execution.runtime_environment_sha256
        or preflight_identity.effective_environment_sha256
        != execution.runtime_environment_sha256
        or cold_identity.backend != execution.backend
        or cold_identity.device_uuid != execution.gpu_uuid
        or preflight_identity.backend != execution.backend
        or preflight_identity.device_uuid != execution.gpu_uuid
        or cold_identity.python_executable != execution.interpreter
        or preflight_identity.python_executable != execution.interpreter
        or execution.source_pre_sha256 != expected_source
        or execution.source_post_sha256 != expected_source
    ):
        raise ValueError("runtime or source authority differs from execution")

    preflight = _mapping(
        _load_ref_json(root, refs["preflight"], "preflight producer"),
        "preflight producer",
    )
    _exact_keys(
        preflight,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "mode",
                "execution_status",
                "policy_sha256",
                "policy_evidence",
                "phase_schema_sha256",
                "state_size",
                "equality_size",
                "residual_size",
                "campaign_authorized",
                "solver_dispatched",
                "finalizer_called",
                "endpoint_audit_called",
                "python_callbacks",
                "runtime",
                "runtime_evidence",
                "timing",
                "failure_reasons",
            }
        ),
        "preflight producer",
    )
    preflight_runtime_mapping = _runtime_mapping(
        preflight["runtime"], "preflight runtime"
    )
    preflight_policy = _parse_policy(
        _load_ref_json(root, refs["preflight_policy"], "preflight policy")
    )
    if (
        preflight["schema_version"] != "single-stage-neq-gntr1-preflight-worker-v1"
        or preflight["route"] != expected_route
        or preflight["plan_sha256"] != expected_plan_sha256
        or preflight["mode"] != "ANNOTATED_LOWER_COMPILE_ONLY"
        or preflight["execution_status"] != "SUCCESS"
        or _sha256(preflight["policy_sha256"], "preflight policy SHA")
        != execution.policy_sha256
        or preflight_policy.policy_sha256 != execution.policy_sha256
        or _artifact_ref(preflight["policy_evidence"], "preflight policy ref")
        != refs["preflight_policy"]
        or preflight["phase_schema_sha256"] != PHASE_SCHEMA_SHA256
        or _integer(preflight["state_size"], "preflight state size") != STATE_SIZE
        or _integer(preflight["equality_size"], "preflight equality size")
        != EQUALITY_SIZE
        or _integer(preflight["residual_size"], "preflight residual size") != 2110
        or preflight["campaign_authorized"] is not False
        or preflight["solver_dispatched"] is not False
        or preflight["finalizer_called"] is not False
        or preflight["endpoint_audit_called"] is not False
        or _integer(preflight["python_callbacks"], "preflight callbacks") != 0
        or _artifact_ref(preflight["runtime_evidence"], "preflight runtime ref")
        != refs["preflight_runtime"]
        or preflight_runtime_mapping["backend"] != execution.backend
        or preflight_runtime_mapping["device_uuid"] != execution.gpu_uuid
        or _boolean(preflight_runtime_mapping["jax_enable_x64"], "preflight x64")
        is not True
        or _array(preflight["failure_reasons"], "preflight failures") != []
    ):
        raise ValueError("preflight raw authority does not pass")
    timing = _mapping(preflight["timing"], "preflight timing")
    _exact_keys(
        timing,
        frozenset(
            {
                "compile_started_ns",
                "compile_completed_ns",
                "process_seconds_before_serialization",
            }
        ),
        "preflight timing",
    )
    if (
        _integer(timing["compile_started_ns"], "preflight compile start")
        >= _integer(timing["compile_completed_ns"], "preflight compile stop")
        or _number(
            timing["process_seconds_before_serialization"],
            "preflight process seconds",
        )
        <= 0.0
    ):
        raise ValueError("preflight timing is invalid")

    producer = _mapping(
        _load_ref_json(root, refs["producer"], "cold producer"), "cold producer"
    )
    _exact_keys(
        producer,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "execution_status",
                "runtime",
                "runtime_evidence",
                "policy_sha256",
                "phase_schema_sha256",
                "history_evidence",
                "terminal_numerical_evidence",
                "policy_evidence",
                "raw_trace_evidence",
                "trace_intervals_evidence",
                "timestamps_ns",
                "transfer_audit",
                "endpoint_audit_called",
                "campaign_authorized",
                "failure_reasons",
            }
        ),
        "cold producer",
    )
    cold_runtime_mapping = _runtime_mapping(producer["runtime"], "cold runtime")
    producer_ref_bindings = {
        "runtime_evidence": "runtime",
        "history_evidence": "history",
        "terminal_numerical_evidence": "terminal_numerical",
        "policy_evidence": "policy",
        "raw_trace_evidence": "raw_trace",
        "trace_intervals_evidence": "trace_intervals",
    }
    if any(
        _artifact_ref(producer[field], f"cold producer.{field}") != refs[ref_name]
        for field, ref_name in producer_ref_bindings.items()
    ):
        raise ValueError("cold producer raw references differ from receipt")
    transfers = _mapping(producer["transfer_audit"], "cold transfer audit")
    _exact_keys(
        transfers,
        frozenset(
            {
                "hot_h2d_transfers",
                "hot_d2h_transfers",
                "python_callbacks",
                "final_d2h_transfers",
            }
        ),
        "cold transfer audit",
    )
    producer_timestamps = _mapping(producer["timestamps_ns"], "cold timestamps")
    if (
        producer["schema_version"] != f"{SCHEMA_VERSION}-producer"
        or producer["route"] != expected_route
        or producer["plan_sha256"] != expected_plan_sha256
        or producer["execution_status"] != "COMPLETE"
        or producer["endpoint_audit_called"] is not False
        or producer["campaign_authorized"] is not False
        or _array(producer["failure_reasons"], "cold failures") != []
        or _sha256(producer["policy_sha256"], "cold policy SHA")
        != execution.policy_sha256
        or producer["phase_schema_sha256"] != PHASE_SCHEMA_SHA256
        or cold_runtime_mapping["backend"] != execution.backend
        or cold_runtime_mapping["device_uuid"] != execution.gpu_uuid
        or _boolean(cold_runtime_mapping["jax_enable_x64"], "cold x64")
        != execution.jax_enable_x64
        or tuple(producer_timestamps)
        != tuple(name for name, _ in execution.timestamps_ns)
        or any(
            _integer(producer_timestamps[name], f"cold timestamps.{name}") != value
            for name, value in execution.timestamps_ns
        )
        or _integer(transfers["hot_h2d_transfers"], "hot H2D")
        != execution.hot_h2d_transfers
        or _integer(transfers["hot_d2h_transfers"], "hot D2H")
        != execution.hot_d2h_transfers
        or _integer(transfers["python_callbacks"], "callbacks")
        != execution.python_callbacks
        or _integer(transfers["final_d2h_transfers"], "final D2H")
        != execution.final_d2h_transfers
    ):
        raise ValueError("cold producer raw authority differs from execution")

    terminal_schema_version = (
        DIAG2_CHILD_TERMINAL_SCHEMA_VERSION
        if expected_route == DIAG2_ROUTE
        else f"{SCHEMA_VERSION}-child-terminal"
    )
    _validate_child_terminal(
        _load_ref_json(root, refs["preflight_child_terminal"], "preflight terminal"),
        "preflight terminal",
        expected_schema_version=terminal_schema_version,
    )
    _validate_child_terminal(
        _load_ref_json(root, refs["child_terminal"], "cold terminal"),
        "cold terminal",
        expected_schema_version=terminal_schema_version,
    )
    if (
        execution.stdout_sha256 != refs["producer"].sha256
        or execution.stdout_size_bytes != refs["producer"].size_bytes
    ):
        raise ValueError("cold stdout differs from canonical producer bytes")
    _validate_process(
        root,
        _load_ref_json(root, refs["process"], "cold process"),
        expected_pid=execution.child_pid,
        expected_start_ticks=execution.child_start_time_ticks,
        expected_argv=execution.argv,
        expected_stdout_sha256=refs["producer"].sha256,
        expected_stdout_size=refs["producer"].size_bytes,
        expected_stderr_sha256=execution.stderr_sha256,
        expected_stderr_size=execution.stderr_size_bytes,
        expected_source_sha256=expected_source,
        context="cold process",
        expected_schema_version=(
            DIAG2_PROCESS_SCHEMA_VERSION
            if expected_route == DIAG2_ROUTE
            else f"{SCHEMA_VERSION}-process"
        ),
        require_parent_monotonic_interval=expected_route == DIAG2_ROUTE,
    )
    _validate_memory(
        _load_ref_json(root, refs["memory"], "cold memory"),
        _load_ref_json(root, refs["memory_samples"], "cold memory samples"),
        expected_pid=execution.child_pid,
        expected_start_ticks=execution.child_start_time_ticks,
        expected_argv=execution.argv,
        expected_gpu_uuid=execution.gpu_uuid,
        physical_memory_bytes=execution.physical_memory_bytes,
        expected_peak_bytes=execution.peak_memory_bytes,
        expected_peak_fraction=execution.reported_peak_memory_fraction,
        context="cold memory",
    )


def _validate_native_equalities_authority(
    root: Path, terminal: TerminalEvidence
) -> None:
    values = _validated_native_equalities(root)
    if not np.array_equal(values, terminal.array("native_equalities").values):
        raise ValueError("terminal native equalities differ from validated reference")


def _validated_native_equalities(root: Path) -> np.ndarray:
    reference = _mapping(
        load_canonical_json_bytes(
            (root / "native-reference" / "reference.json").read_bytes()
        ),
        "native reference",
    )
    evidence = _mapping(reference.get("evidence"), "native reference evidence")
    arrays = _mapping(evidence.get("arrays"), "native reference arrays")
    equality_ref = _mapping(
        arrays.get("raw_equalities"), "native reference raw equalities"
    )
    relative = _string(
        equality_ref.get("relative_path"), "native reference equality path"
    )
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("native reference equality path is invalid")
    path = (root / "native-reference" / relative_path).resolve(strict=True)
    if not path.is_relative_to((root / "native-reference").resolve(strict=True)):
        raise ValueError("native reference equality path escapes")
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
    if values.dtype.str != "<f8" or values.shape != (EQUALITY_SIZE,):
        raise ValueError("validated reference native equalities differ from schema")
    return values


def build_diagnostic_receipt(
    *, artifact_root: Path, evidence_refs: Mapping[str, ArtifactRef]
) -> DiagnosticReceipt:
    """Resolve canonical raw artifacts and derive the complete diagnostic receipt."""

    if frozenset(evidence_refs) != EVIDENCE_REF_KEYS:
        raise ValueError("diagnostic evidence references differ from the frozen schema")
    for reference in evidence_refs.values():
        _resolve_artifact(artifact_root, reference)
    refs = tuple((name, evidence_refs[name]) for name in sorted(EVIDENCE_REF_KEYS))
    history = _parse_history(
        _load_ref_json(artifact_root, evidence_refs["history"], "history")
    )
    terminal = _parse_terminal(
        artifact_root,
        _load_ref_json(artifact_root, evidence_refs["terminal_numerical"], "terminal"),
    )
    policy = _parse_policy(
        _load_ref_json(artifact_root, evidence_refs["policy"], "quality policy"),
        terminal,
    )
    _validate_native_equalities_authority(artifact_root, terminal)
    execution = _parse_execution(
        _load_ref_json(artifact_root, evidence_refs["execution"], "execution"),
        evidence_refs,
    )
    if policy.policy_sha256 != execution.policy_sha256:
        raise ValueError("terminal policy differs from preflight/cold policy")
    _validate_execution_authorities(artifact_root, evidence_refs, execution)
    normalized_trace = _load_ref_json(
        artifact_root, evidence_refs["trace_intervals"], "trace intervals"
    )
    independently_normalized = normalize_chrome_trace(
        _resolve_artifact(artifact_root, evidence_refs["raw_trace"]),
        phase_schema_sha256=execution.phase_schema_sha256,
    )
    if independently_normalized != normalized_trace:
        raise ValueError("normalized trace intervals differ from raw profiler bytes")
    phases = _parse_phases(normalized_trace, execution.phase_schema_sha256)
    return _derive(refs, policy, history, terminal, phases, execution)


def history_row(
    *,
    outcome: AttemptOutcome,
    accepted_step_number: int,
    integer_values: Mapping[str, int],
    steihaug_hit_boundary: bool,
    floating_values: Mapping[str, float | None],
) -> HistoryRow:
    """Construct one adapter-independent row under the frozen 300-slot schema."""

    if frozenset(integer_values) != frozenset(HISTORY_INTEGER_FIELDS):
        raise ValueError("history integer fields differ from the frozen schema")
    if frozenset(floating_values) != frozenset(HISTORY_FLOAT_FIELDS):
        raise ValueError("history floating fields differ from the frozen schema")
    payload: dict[str, JsonValue] = {
        "outcome": outcome.value,
        "accepted_step_number": accepted_step_number,
        "steihaug_hit_boundary": steihaug_hit_boundary,
    }
    payload.update({name: integer_values[name] for name in HISTORY_INTEGER_FIELDS})
    payload.update({name: floating_values[name] for name in HISTORY_FLOAT_FIELDS})
    return _history_row(payload, 0)


def _history_evidence_payload(
    rows: tuple[HistoryRow, ...],
    *,
    quality_latch: bool,
    first_quality_attempt: int,
    first_quality_accepted_step: int,
    schema_version: str,
) -> dict[str, JsonValue]:
    """Serialize rows with terminal counters derived exclusively from outcomes."""

    if len(rows) != MAXIMUM_ATTEMPTS:
        raise ValueError("history must retain exactly 300 rows")
    outcomes = tuple(row.outcome for row in rows)
    attempts = sum(outcome is not AttemptOutcome.INACTIVE for outcome in outcomes)
    accepted = sum(outcome is AttemptOutcome.ACCEPTED for outcome in outcomes)
    retries = sum(outcome in RETRY_OUTCOMES for outcome in outcomes)
    fatal = attempts > 0 and outcomes[attempts - 1] in FATAL_OUTCOMES
    bounded = accepted == MAXIMUM_ACCEPTED_STEPS
    if fatal:
        status = LoopStatus(outcomes[attempts - 1].value)
    elif quality_latch:
        status = LoopStatus.DEVICE_QUALITY_CANDIDATE
    elif bounded:
        status = LoopStatus.BOUNDED_COMPLETE
    else:
        status = LoopStatus.ATTEMPT_LIMIT
    result: dict[str, JsonValue] = {
        "schema_version": schema_version,
        "rows": [row.to_payload() for row in rows],
        "attempts": attempts,
        "accepted_steps": accepted,
        "retryable_rejections": retries,
        "status": status.value,
        "fatal": fatal,
        "bounded_complete": bounded,
        "quality_latch": quality_latch,
        "first_quality_attempt": first_quality_attempt,
        "first_quality_accepted_step": first_quality_accepted_step,
    }
    _parse_history(result, expected_schema_version=schema_version)
    return result


def history_evidence_payload(
    rows: tuple[HistoryRow, ...],
    *,
    quality_latch: bool,
    first_quality_attempt: int,
    first_quality_accepted_step: int,
) -> dict[str, JsonValue]:
    return _history_evidence_payload(
        rows,
        quality_latch=quality_latch,
        first_quality_attempt=first_quality_attempt,
        first_quality_accepted_step=first_quality_accepted_step,
        schema_version=f"{SCHEMA_VERSION}-history",
    )


def _history_evidence_from_arrays(
    history: object,
    *,
    quality_latch: bool,
    first_quality_attempt: int,
    first_quality_accepted_step: int,
    schema_version: str,
) -> dict[str, JsonValue]:
    """Materialize the public fixed-shape history arrays without class coupling."""

    def array(name: str) -> np.ndarray:
        try:
            value = getattr(history, name)
        except AttributeError as error:
            raise ValueError(f"history arrays omit {name}") from error
        result = np.asarray(value)
        if result.shape != (MAXIMUM_ATTEMPTS,):
            raise ValueError(f"history.{name} must have shape (300,)")
        return result

    integer_names = ("outcome", "accepted_step_number", *HISTORY_INTEGER_FIELDS)
    integer_arrays: dict[str, np.ndarray] = {}
    for name in integer_names:
        values = array(name)
        if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        ):
            raise TypeError(f"history.{name} must have an integer dtype")
        integer_arrays[name] = values
    boundary = array("steihaug_hit_boundary")
    if not np.issubdtype(boundary.dtype, np.bool_):
        raise TypeError("history.steihaug_hit_boundary must have a Boolean dtype")
    floating_arrays: dict[str, np.ndarray] = {}
    for name in HISTORY_FLOAT_FIELDS:
        values = array(name)
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"history.{name} must have a floating dtype")
        floating_arrays[name] = values

    outcome_by_code = tuple(AttemptOutcome)
    rows: list[HistoryRow] = []
    for index in range(MAXIMUM_ATTEMPTS):
        outcome_code = int(integer_arrays["outcome"][index])
        if not 0 <= outcome_code < len(outcome_by_code):
            raise ValueError(f"history.outcome[{index}] is invalid")
        rows.append(
            history_row(
                outcome=outcome_by_code[outcome_code],
                accepted_step_number=int(integer_arrays["accepted_step_number"][index]),
                integer_values={
                    name: int(integer_arrays[name][index])
                    for name in HISTORY_INTEGER_FIELDS
                },
                steihaug_hit_boundary=bool(boundary[index]),
                floating_values={
                    name: (
                        None
                        if not np.isfinite(floating_arrays[name][index])
                        else float(floating_arrays[name][index])
                    )
                    for name in HISTORY_FLOAT_FIELDS
                },
            )
        )
    return _history_evidence_payload(
        tuple(rows),
        quality_latch=quality_latch,
        first_quality_attempt=first_quality_attempt,
        first_quality_accepted_step=first_quality_accepted_step,
        schema_version=schema_version,
    )


def history_evidence_from_arrays(
    history: object,
    *,
    quality_latch: bool,
    first_quality_attempt: int,
    first_quality_accepted_step: int,
) -> dict[str, JsonValue]:
    return _history_evidence_from_arrays(
        history,
        quality_latch=quality_latch,
        first_quality_attempt=first_quality_attempt,
        first_quality_accepted_step=first_quality_accepted_step,
        schema_version=f"{SCHEMA_VERSION}-history",
    )


def diag5_history_evidence_from_arrays(
    history: object,
    *,
    quality_latch: bool,
    first_quality_attempt: int,
    first_quality_accepted_step: int,
) -> dict[str, JsonValue]:
    return _history_evidence_from_arrays(
        history,
        quality_latch=quality_latch,
        first_quality_attempt=first_quality_attempt,
        first_quality_accepted_step=first_quality_accepted_step,
        schema_version="single-stage-fullspace-neq-gntr3-history-v1",
    )


def validate_diag5_history_evidence_payload(
    value: JsonValue, *, defer_step_bounds: bool = False
) -> HistoryEvidence:
    return _parse_history(
        value,
        defer_step_bounds=defer_step_bounds,
        expected_schema_version="single-stage-fullspace-neq-gntr3-history-v1",
    )


def array_evidence_payload(
    *, reference: ArtifactRef, name: str, values: np.ndarray
) -> dict[str, JsonValue]:
    """Bind one canonical NPY reference to its typed little-endian raw values."""

    if name not in ARRAY_SPECS:
        raise ValueError("array name differs from the frozen schema")
    dtype, shape = ARRAY_SPECS[name]
    observed = np.asarray(values)
    if (
        observed.dtype.str != dtype
        or observed.shape != shape
        or not observed.flags.c_contiguous
        or (
            observed.dtype.kind == "f"
            and name not in NONFINITE_ARRAYS
            and not np.all(np.isfinite(observed))
        )
    ):
        raise ValueError(f"{name} representation differs from the frozen schema")
    return {
        "artifact": _artifact_ref_payload(reference),
        "content_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        "dtype": dtype,
        "shape": list(shape),
    }


def terminal_numerical_payload(
    *,
    arrays: Mapping[str, Mapping[str, JsonValue]],
    objective: float,
    objective_terms: Mapping[str, float],
    objective_weights: Mapping[str, float],
    reconstructed_objective: float,
    authoritative_objective: float,
    final_certificate: Mapping[str, float],
    kkt_status: KktStatus,
    raw_kkt_inf: float | None,
    scaled_stationarity_inf: float | None,
    residual_value_defect: float,
    residual_gradient_defect: float,
    transpose_primal_dot: float,
    transpose_adjoint_dot: float,
    transpose_denominator: float,
    transpose_defect: float,
    terminal_endpoint_diagnostics_seconds: float,
) -> dict[str, JsonValue]:
    """Serialize the post-timing terminal diagnostic without summary booleans."""

    if frozenset(arrays) != frozenset(ARRAY_SPECS):
        raise ValueError("terminal arrays differ from the frozen schema")
    return {
        "schema_version": f"{SCHEMA_VERSION}-terminal",
        "arrays": {name: dict(arrays[name]) for name in sorted(arrays)},
        "objective": objective,
        "objective_terms": dict(objective_terms),
        "objective_weights": dict(objective_weights),
        "reconstructed_objective": reconstructed_objective,
        "authoritative_objective": authoritative_objective,
        "final_certificate": dict(final_certificate),
        "kkt_status": kkt_status.value,
        "raw_kkt_inf": raw_kkt_inf,
        "scaled_stationarity_inf": scaled_stationarity_inf,
        "residual_value_defect": residual_value_defect,
        "residual_gradient_defect": residual_gradient_defect,
        "transpose_primal_dot": transpose_primal_dot,
        "transpose_adjoint_dot": transpose_adjoint_dot,
        "transpose_denominator": transpose_denominator,
        "transpose_defect": transpose_defect,
        "terminal_endpoint_diagnostics_seconds": terminal_endpoint_diagnostics_seconds,
    }


def _diag4_ordered_finite_mapping(
    value: JsonValue,
    *,
    fields: tuple[str, ...],
    context: str,
) -> tuple[tuple[str, float], ...]:
    payload = _mapping(value, context)
    _exact_keys(payload, frozenset(fields), context)
    return tuple((name, _number(payload[name], f"{context}.{name}")) for name in fields)


def diag4_terminal_numerical_payload(
    *,
    terminal_numerical: Mapping[str, JsonValue],
    numerical_identity: NativeEquivalentNumericalIdentity,
    endpoint_state_sha256: str,
    terminal_observables: Mapping[str, float],
    endpoint_objective_terms: Mapping[str, float],
    endpoint_observables: Mapping[str, float],
) -> dict[str, JsonValue]:
    """Extend one legacy terminal document with source-bound same-state evidence."""

    terminal = dict(terminal_numerical)
    if terminal.get("schema_version") != f"{SCHEMA_VERSION}-terminal":
        raise ValueError("DIAG4 terminal base schema differs")
    payload: dict[str, JsonValue] = {
        **terminal,
        "schema_version": f"{DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal",
        "numerical_route": numerical_identity.numerical_route,
        "numerical_result_schema_version": (
            numerical_identity.numerical_result_schema_version
        ),
        "problem_sha256": numerical_identity.problem_sha256,
        "optimizer_options_sha256": numerical_identity.optimizer_options_sha256,
        "base_neq_gntr1_policy_sha256": (
            numerical_identity.base_neq_gntr1_policy_sha256
        ),
        "scaling_sha256": numerical_identity.scaling_sha256,
        "bootstrap_state_sha256": numerical_identity.bootstrap_state_sha256,
        "initial_physical_state_sha256": (
            numerical_identity.initial_physical_state_sha256
        ),
        "identity_sha256": numerical_identity.identity_sha256,
        "endpoint_state_sha256": endpoint_state_sha256,
        "terminal_observables": dict(terminal_observables),
        "endpoint_objective_terms": dict(endpoint_objective_terms),
        "endpoint_observables": dict(endpoint_observables),
    }
    _validate_gntr3_terminal_numerical_structure(payload)
    return payload


def _validate_gntr3_terminal_numerical_structure(
    value: JsonValue,
) -> tuple[
    dict[str, JsonValue],
    NativeEquivalentNumericalIdentity,
    str,
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
]:
    payload = _mapping(value, "DIAG4 terminal numerical evidence")
    extension_fields = frozenset(
        {
            "endpoint_state_sha256",
            "numerical_route",
            "numerical_result_schema_version",
            "problem_sha256",
            "optimizer_options_sha256",
            "base_neq_gntr1_policy_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
            "terminal_observables",
            "endpoint_objective_terms",
            "endpoint_observables",
        }
    )
    legacy_fields = frozenset(
        {
            "schema_version",
            "arrays",
            "objective",
            "objective_terms",
            "objective_weights",
            "reconstructed_objective",
            "authoritative_objective",
            "final_certificate",
            "kkt_status",
            "raw_kkt_inf",
            "scaled_stationarity_inf",
            "residual_value_defect",
            "residual_gradient_defect",
            "transpose_primal_dot",
            "transpose_adjoint_dot",
            "transpose_denominator",
            "transpose_defect",
            "terminal_endpoint_diagnostics_seconds",
        }
    )
    _exact_keys(payload, legacy_fields | extension_fields, "DIAG4 terminal numerical")
    if payload["schema_version"] != f"{DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal":
        raise ValueError("DIAG4 terminal numerical schema differs")
    legacy = {name: payload[name] for name in legacy_fields}
    legacy["schema_version"] = f"{SCHEMA_VERSION}-terminal"
    terminal_observables = _diag4_ordered_finite_mapping(
        payload["terminal_observables"],
        fields=DIAG4_ENDPOINT_OBSERVABLE_FIELDS,
        context="DIAG4 terminal observables",
    )
    endpoint_terms = _diag4_ordered_finite_mapping(
        payload["endpoint_objective_terms"],
        fields=DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS,
        context="DIAG4 endpoint objective terms",
    )
    endpoint_observables = _diag4_ordered_finite_mapping(
        payload["endpoint_observables"],
        fields=DIAG4_ENDPOINT_OBSERVABLE_FIELDS,
        context="DIAG4 endpoint observables",
    )
    terminal_terms = _diag4_ordered_finite_mapping(
        payload["objective_terms"],
        fields=DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS,
        context="DIAG4 terminal objective terms",
    )
    weights = _diag4_ordered_finite_mapping(
        payload["objective_weights"],
        fields=DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS,
        context="DIAG4 terminal objective weights",
    )
    arrays = _mapping(payload["arrays"], "DIAG4 terminal arrays")
    physical_state = _mapping(
        arrays.get("physical_state"), "DIAG4 terminal physical-state array"
    )
    endpoint_state_sha256 = _sha256(
        payload["endpoint_state_sha256"], "DIAG4 endpoint state SHA"
    )
    if (
        endpoint_terms != terminal_terms
        or endpoint_observables != terminal_observables
        or endpoint_state_sha256
        != _sha256(
            physical_state.get("content_sha256"),
            "DIAG4 terminal physical-state content SHA",
        )
        or sum(
            term * weight
            for (_, term), (_, weight) in zip(endpoint_terms, weights, strict=True)
        )
        != _number(payload["objective"], "DIAG4 terminal objective")
    ):
        raise ValueError("DIAG4 terminal endpoint evidence differs")
    return (
        legacy,
        NativeEquivalentNumericalIdentity(
            _string(payload["numerical_route"], "DIAG4 terminal numerical route"),
            _string(
                payload["numerical_result_schema_version"],
                "DIAG4 terminal numerical result schema",
            ),
            _sha256(payload["problem_sha256"], "DIAG4 terminal problem SHA"),
            _sha256(payload["optimizer_options_sha256"], "DIAG4 terminal options SHA"),
            _sha256(
                payload["base_neq_gntr1_policy_sha256"],
                "DIAG4 terminal base policy SHA",
            ),
            _sha256(payload["scaling_sha256"], "DIAG4 terminal scaling SHA"),
            _sha256(
                payload["bootstrap_state_sha256"],
                "DIAG4 terminal bootstrap-state SHA",
            ),
            _sha256(
                payload["initial_physical_state_sha256"],
                "DIAG4 terminal initial-state SHA",
            ),
            _sha256(payload["identity_sha256"], "DIAG4 terminal identity SHA"),
        ),
        endpoint_state_sha256,
        terminal_observables,
        endpoint_terms,
        endpoint_observables,
    )


def _validate_gntr3_terminal_numerical_payload(
    artifact_root: Path, value: JsonValue
) -> TerminalEvidenceV4:
    """Deep-load the sole GNTR3 terminal artifact and its endpoint evaluation."""

    (
        legacy,
        numerical_identity,
        state_sha256,
        terminal_observables,
        endpoint_terms,
        endpoint_observables,
    ) = _validate_gntr3_terminal_numerical_structure(value)
    if (
        numerical_identity.numerical_route != DIAG4_NUMERICAL_ROUTE
        or numerical_identity.numerical_result_schema_version
        != DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("DIAG4 terminal numerical identity differs")
    terminal = _parse_terminal(artifact_root, legacy)
    if state_sha256 != terminal.array("physical_state").content_sha256:
        raise ValueError("DIAG4 endpoint state differs from terminal physical state")
    if endpoint_terms != tuple(
        (name, dict(terminal.objective_terms)[name])
        for name in DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS
    ):
        raise ValueError("DIAG4 endpoint objective terms differ from terminal")
    if endpoint_observables != terminal_observables:
        raise ValueError("DIAG4 endpoint observables differ from terminal")
    endpoint_weighted = sum(
        value * dict(terminal.objective_weights)[name] for name, value in endpoint_terms
    )
    if endpoint_weighted != terminal.objective:
        raise ValueError("DIAG4 endpoint objective differs from terminal")
    return TerminalEvidenceV4(
        terminal,
        numerical_identity,
        state_sha256,
        terminal_observables,
        endpoint_terms,
        endpoint_observables,
    )


def validate_diag4_terminal_numerical_payload(
    artifact_root: Path, value: JsonValue
) -> TerminalEvidenceV4:
    """Validate the shared GNTR3 terminal through the frozen DIAG4 API."""

    return _validate_gntr3_terminal_numerical_payload(artifact_root, value)


def _policy_evidence_payload(
    *,
    policy_sha256: str,
    native_raw_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
    schema_version: str,
) -> dict[str, JsonValue]:
    """Serialize the entire frozen policy; the parser recomputes its identity."""

    native = np.ascontiguousarray(native_raw_equalities, dtype=np.dtype("<f8"))
    scale = np.ascontiguousarray(constraint_inverse_scale, dtype=np.dtype("<f8"))
    if native.shape != (EQUALITY_SIZE,) or scale.shape != (EQUALITY_SIZE,):
        raise ValueError("quality policy arrays must have shape (255,)")
    if not np.all(np.isfinite(native)) or not np.all(np.isfinite(scale)):
        raise ValueError("quality policy arrays must be finite")
    return {
        "schema_version": schema_version,
        "policy_sha256": _sha256(policy_sha256, "quality policy SHA"),
        "native_raw_equalities_sha256": exact_numeric_tree_sha256(native),
        "native_raw_equalities": native.tolist(),
        "constraint_inverse_scale_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
        "constraint_inverse_scale": scale.tolist(),
        "objective_target": OBJECTIVE_MAXIMUM,
        "state_size": STATE_SIZE,
        "equality_size": EQUALITY_SIZE,
        "objective_residual_size": 2110,
        "component_absolute_tolerance": RAW_EQUALITY_ABSOLUTE_TOLERANCE,
        "component_relative_tolerance": RAW_EQUALITY_RELATIVE_TOLERANCE,
        "scaled_feasibility_tolerance": FEASIBILITY_MAXIMUM,
        "residual_value_defect_tolerance": RESIDUAL_VALUE_DEFECT_MAXIMUM,
        "residual_gradient_defect_tolerance": RESIDUAL_GRADIENT_DEFECT_MAXIMUM,
        "transpose_defect_tolerance": TRANSPOSE_DEFECT_MAXIMUM,
        "gntr_options": dict(FROZEN_GNTR_OPTIONS),
    }


def policy_evidence_payload(
    *,
    policy_sha256: str,
    native_raw_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
) -> dict[str, JsonValue]:
    return _policy_evidence_payload(
        policy_sha256=policy_sha256,
        native_raw_equalities=native_raw_equalities,
        constraint_inverse_scale=constraint_inverse_scale,
        schema_version=f"{SCHEMA_VERSION}-policy",
    )


def diag5_policy_evidence_payload(
    *,
    policy_sha256: str,
    native_raw_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
) -> dict[str, JsonValue]:
    payload = _policy_evidence_payload(
        policy_sha256=policy_sha256,
        native_raw_equalities=native_raw_equalities,
        constraint_inverse_scale=constraint_inverse_scale,
        schema_version="single-stage-native-equivalent-quality-policy-v1",
    )
    _parse_policy(
        payload,
        expected_schema_version="single-stage-native-equivalent-quality-policy-v1",
    )
    return payload


def validate_diag5_policy_evidence_payload(value: JsonValue) -> PolicyEvidence:
    return _parse_policy(
        value,
        expected_schema_version="single-stage-native-equivalent-quality-policy-v1",
    )


def raw_trace_payload(
    *,
    phase_schema_sha256: str,
    trace_start_ns: int,
    trace_stop_ns: int,
    device_intervals: tuple[Mapping[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    """Serialize profiler-local device intervals; attribution remains validator-owned."""

    if _sha256(phase_schema_sha256, "phase schema SHA") != PHASE_SCHEMA_SHA256:
        raise ValueError("phase schema SHA differs from the frozen diagnostic phases")
    if not device_intervals:
        raise ValueError("raw trace must retain at least one device interval")
    start = _integer(trace_start_ns, "trace-local envelope start")
    stop = _integer(trace_stop_ns, "trace-local envelope stop")
    if stop <= start:
        raise ValueError("trace-local envelope is invalid")
    return {
        "schema_version": f"{SCHEMA_VERSION}-raw-trace",
        "phase_schema_sha256": phase_schema_sha256,
        "trace_start_ns": start,
        "trace_stop_ns": stop,
        "device_intervals": [dict(interval) for interval in device_intervals],
    }


def normalize_chrome_trace(
    path: Path,
    *,
    phase_schema_sha256: str,
) -> dict[str, JsonValue]:
    """Normalize one gzip Chrome trace into raw device intervals or fail closed."""

    if _sha256(phase_schema_sha256, "phase schema SHA") != PHASE_SCHEMA_SHA256:
        raise ValueError("phase schema SHA differs from the frozen diagnostic phases")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"raw Chrome trace contains duplicate key {key!r}")
            result[key] = value
        return result

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        document = json.load(
            stream,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"raw Chrome trace contains {token}")
            ),
        )
    events_value = (
        document.get("traceEvents") if isinstance(document, dict) else document
    )
    if not isinstance(events_value, list):
        raise TypeError("raw Chrome trace has no traceEvents array")

    def ns(value: object, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{context} must be a numeric microsecond timestamp")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{context} must be finite and nonnegative")
        return round(result * 1000.0)

    def recognized_path(arguments: object) -> tuple[str, ...]:
        if not isinstance(arguments, dict):
            return ()
        name = arguments.get("name")
        if not isinstance(name, str):
            return ()
        return tuple(segment for segment in name.split("/") if segment in PHASE_IDS)

    envelope_events = []
    for index, event in enumerate(events_value):
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        arguments = event.get("args", {})
        annotation_name = arguments.get("name") if isinstance(arguments, dict) else None
        if event.get("name") == TRACE_LOOP_ENVELOPE_NAME or (
            annotation_name == TRACE_LOOP_ENVELOPE_NAME
        ):
            envelope_events.append(
                (
                    ns(event.get("ts"), f"traceEvents[{index}].ts"),
                    ns(event.get("dur"), f"traceEvents[{index}].dur"),
                )
            )
    if len(envelope_events) != 1:
        raise ValueError("raw Chrome trace must contain exactly one loop envelope")
    envelope_start, envelope_duration = envelope_events[0]
    if envelope_duration <= 0:
        raise ValueError("raw Chrome trace loop envelope has zero duration")
    envelope_stop = envelope_start + envelope_duration

    intervals: list[dict[str, JsonValue]] = []
    for index, event in enumerate(events_value):
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        arguments = event.get("args", {})
        path = recognized_path(arguments)
        device_argument_keys = (
            frozenset(arguments) if isinstance(arguments, dict) else frozenset()
        )
        is_device = (
            any(marker in category for marker in ("gpu", "kernel", "device", "cuda"))
            or bool(
                device_argument_keys
                & frozenset({"kernel_details", "memcpy_details", "memset_details"})
            )
            or {"cuda_graph_id", "cuda_graph_node_id"}.issubset(device_argument_keys)
        )
        if not is_device:
            continue
        start = ns(event.get("ts"), f"traceEvents[{index}].ts")
        duration = ns(event.get("dur"), f"traceEvents[{index}].dur")
        if duration <= 0:
            continue
        end = start + duration
        if start < envelope_start or end > envelope_stop:
            raise ValueError("raw trace device event lies outside the loop envelope")
        intervals.append(
            {
                "start_ns": start,
                "end_ns": end,
                "scope_paths": [list(path)] if path else [],
            }
        )
    if not intervals:
        raise ValueError("raw Chrome trace contains no in-envelope device intervals")
    intervals.sort(key=lambda row: (int(row["start_ns"]), int(row["end_ns"])))
    return raw_trace_payload(
        phase_schema_sha256=phase_schema_sha256,
        trace_start_ns=envelope_start,
        trace_stop_ns=envelope_stop,
        device_intervals=tuple(intervals),
    )


def execution_evidence_payload(
    *,
    supporting_evidence: Mapping[str, ArtifactRef],
    preflight: Mapping[str, JsonValue],
    cold: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Flatten the two-child raw observations into the strict execution schema."""

    support_keys = EVIDENCE_REF_KEYS - frozenset(
        {"history", "terminal_numerical", "raw_trace", "trace_intervals", "execution"}
    )
    if frozenset(supporting_evidence) != support_keys:
        raise ValueError("execution supporting evidence differs from the frozen schema")
    preflight_keys = frozenset(
        {
            "status",
            "compile_success",
            "solver_dispatched",
            "finalizer_called",
            "endpoint_audit_called",
            "campaign_authorized",
            "callbacks",
        }
    )
    cold_keys = frozenset(
        {
            "status",
            "child_pid",
            "child_start_time_ticks",
            "backend",
            "gpu_uuid",
            "jax_enable_x64",
            "state_size",
            "equality_size",
            "residual_size",
            "policy_sha256",
            "phase_schema_sha256",
            "source_pre_sha256",
            "source_post_sha256",
            "runtime_environment_sha256",
            "interpreter",
            "argv",
            "physical_memory_bytes",
            "peak_memory_bytes",
            "peak_memory_fraction",
            "hot_h2d_transfers",
            "hot_d2h_transfers",
            "python_callbacks",
            "final_d2h_transfers",
            "timestamps_ns",
            "stdout_sha256",
            "stdout_size_bytes",
            "stderr_sha256",
            "stderr_size_bytes",
        }
    )
    _exact_keys(preflight, preflight_keys, "preflight observation")
    _exact_keys(cold, cold_keys, "cold observation")
    return {
        "schema_version": f"{SCHEMA_VERSION}-execution",
        "supporting_evidence": {
            name: _artifact_ref_payload(supporting_evidence[name])
            for name in sorted(support_keys)
        },
        "preflight_status": preflight["status"],
        "preflight_compile_success": preflight["compile_success"],
        "preflight_solver_dispatched": preflight["solver_dispatched"],
        "preflight_finalizer_called": preflight["finalizer_called"],
        "preflight_endpoint_audit_called": preflight["endpoint_audit_called"],
        "preflight_campaign_authorized": preflight["campaign_authorized"],
        "preflight_callbacks": preflight["callbacks"],
        "cold_status": cold["status"],
        **{name: cold[name] for name in cold_keys - {"status"}},
    }


def _quality_payload(quality: QualityEvidence) -> dict[str, JsonValue]:
    return {
        "objective_margin": quality.objective_margin,
        "component_margins": list(quality.component_margins),
        "minimum_component_margin": quality.minimum_component_margin,
        "minimum_component_index": quality.minimum_component_index,
        "scaled_feasibility_margin": quality.scaled_feasibility_margin,
        "objective_usage_ratio": quality.objective_usage_ratio,
        "component_usage_ratio": quality.component_usage_ratio,
        "feasibility_usage_ratio": quality.feasibility_usage_ratio,
        "residual_value_margin": quality.residual_value_margin,
        "residual_gradient_margin": quality.residual_gradient_margin,
        "transpose_margin": quality.transpose_margin,
    }


def _phase_payload(phases: PhaseEvidence) -> dict[str, JsonValue]:
    return {
        "status": phases.status.value,
        "durations_ns": {name: duration for name, duration in phases.durations_ns},
        "overlaps_ns": [
            {"left": left, "right": right, "duration_ns": duration}
            for left, right, duration in phases.overlaps_ns
        ],
        "device_active_ns": phases.device_active_ns,
        "total_attributed_ns": phases.total_attributed_ns,
        "unattributed_ns": phases.unattributed_ns,
        "current_model_ns": phases.current_model_ns,
        "coverage": phases.coverage,
        "trace_start_ns": phases.trace_start_ns,
        "trace_stop_ns": phases.trace_stop_ns,
    }


def build_incomplete_diagnostic_receipt(
    *,
    artifact_root: Path,
    evidence_refs: Mapping[str, ArtifactRef | None],
) -> IncompleteDiagnosticReceipt:
    """Derive a truthful nonpromoting failure solely from available raw evidence."""

    if frozenset(evidence_refs) != EVIDENCE_REF_KEYS:
        raise ValueError("incomplete evidence references differ from the frozen schema")
    for reference in evidence_refs.values():
        if reference is not None:
            _resolve_artifact(artifact_root, reference)

    def terminal_observation(name: str) -> tuple[str | None, tuple[str, ...]]:
        reference = evidence_refs[name]
        if reference is None:
            return None, ()
        payload = _mapping(
            _load_ref_json(artifact_root, reference, f"{name} evidence"),
            f"{name} evidence",
        )
        value = payload.get("terminal_status")
        reasons = tuple(
            _string(item, f"{name}.failure_reasons")
            for item in _array(payload.get("failure_reasons", []), f"{name}.reasons")
        )
        return (value if isinstance(value, str) else None), reasons

    missing = tuple(
        name for name in sorted(EVIDENCE_REF_KEYS) if evidence_refs[name] is None
    )
    preflight_status, preflight_reasons = terminal_observation(
        "preflight_child_terminal"
    )
    cold_status, cold_reasons = terminal_observation("child_terminal")
    source_failures = tuple(
        reason
        for reason in (*preflight_reasons, *cold_reasons)
        if reason.startswith(("SOURCE_PRE:", "SOURCE_POST:"))
    )
    if evidence_refs["source_manifest"] is None and source_failures:
        stage = FailureStage.COLD_SOURCE
        reason = (
            "SOURCE_CAPTURE:"
            + hashlib.sha256(canonical_json_bytes(list(source_failures))).hexdigest()
        )
    elif preflight_status is not None and preflight_status != "COMPLETE":
        stage = FailureStage.PREFLIGHT
        reason = f"PREFLIGHT_TERMINAL:{preflight_status}"
    elif evidence_refs["preflight"] is None or any(
        evidence_refs[name] is None
        for name in (
            "preflight_child_terminal",
            "preflight_process",
            "preflight_memory",
            "preflight_memory_samples",
            "preflight_runtime",
            "preflight_policy",
        )
    ):
        stage = FailureStage.PREFLIGHT
        reason = "MISSING_PREFLIGHT_EVIDENCE:" + ",".join(missing)
    elif cold_status == "TIMEOUT":
        stage = FailureStage.COLD_TIMEOUT
        reason = "COLD_TERMINAL:TIMEOUT"
    elif cold_status == "CRASH":
        stage = FailureStage.COLD_CRASH
        reason = "COLD_TERMINAL:CRASH"
    elif cold_status == "MONITOR_FAILURE":
        stage = FailureStage.COLD_MONITOR
        reason = "COLD_TERMINAL:MONITOR_FAILURE"
    elif cold_status == "PROTOCOL_FAILURE":
        stage = FailureStage.COLD_PROTOCOL
        reason = "COLD_TERMINAL:PROTOCOL_FAILURE"
    elif missing:
        stage = FailureStage.NUMERICAL_EVIDENCE
        reason = "MISSING_COLD_EVIDENCE:" + ",".join(missing)
    else:
        complete_refs = {
            name: reference
            for name, reference in evidence_refs.items()
            if reference is not None
        }
        try:
            build_diagnostic_receipt(
                artifact_root=artifact_root,
                evidence_refs=complete_refs,
            )
        except (OSError, TypeError, ValueError) as error:
            text = str(error).lower()
            if "source" in text or "snapshot" in text:
                stage = FailureStage.COLD_SOURCE
            elif "memory" in text:
                stage = FailureStage.COLD_RESOURCE
            elif "process" in text or "runtime" in text or "producer" in text:
                stage = FailureStage.COLD_PROTOCOL
            else:
                stage = FailureStage.NUMERICAL_EVIDENCE
            reason = (
                f"SEMANTIC_VALIDATION:{type(error).__name__}:"
                + hashlib.sha256(str(error).encode()).hexdigest()
            )
        else:
            raise ValueError(
                "complete valid evidence must use build_diagnostic_receipt"
            )
    return IncompleteDiagnosticReceipt(
        tuple((name, evidence_refs[name]) for name in sorted(EVIDENCE_REF_KEYS)),
        stage,
        (reason,),
    )


def diagnostic_receipt_payload(
    receipt: DiagnosticReceipt | IncompleteDiagnosticReceipt,
) -> dict[str, JsonValue]:
    """Serialize only derived claims plus content-addressed raw authorities."""

    if isinstance(receipt, IncompleteDiagnosticReceipt):
        return {
            "schema_version": SCHEMA_VERSION,
            "route": ROUTE,
            "numerical_route": NUMERICAL_ROUTE,
            "plan_sha256": PLAN_SHA256,
            "evidence_refs": {
                name: None if reference is None else _artifact_ref_payload(reference)
                for name, reference in receipt.evidence_refs
            },
            "quality": None,
            "phase_attribution": None,
            "verdict": DiagnosticVerdict.INCOMPLETE.value,
            "historical_aggregate_relation": HistoricalAggregateRelation.DIVERGES.value,
            "next_route": NextRoute.NOT_SELECTED.value,
            "reuse_opportunity_estimate": None,
            "failure": {
                "stage": receipt.failure_stage.value,
                "reasons": list(receipt.failure_reasons),
            },
            "engineering_campaign_receipt_produced": False,
            "promotion_authorized": False,
            "formal_comparison": "NOT_PRODUCED",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "route": ROUTE,
        "numerical_route": NUMERICAL_ROUTE,
        "plan_sha256": PLAN_SHA256,
        "evidence_refs": {
            name: _artifact_ref_payload(reference)
            for name, reference in receipt.evidence_refs
        },
        "quality": _quality_payload(receipt.quality),
        "phase_attribution": _phase_payload(receipt.phases),
        "verdict": receipt.verdict.value,
        "historical_aggregate_relation": receipt.historical_relation.value,
        "next_route": receipt.next_route.value,
        "reuse_opportunity_estimate": receipt.reuse_opportunity_estimate,
        "failure": None,
        "engineering_campaign_receipt_produced": False,
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }


def diagnostic_receipt_bytes(
    receipt: DiagnosticReceipt | IncompleteDiagnosticReceipt,
) -> bytes:
    return canonical_json_bytes(diagnostic_receipt_payload(receipt))


def diagnostic_receipt_from_payload(
    value: JsonValue, *, artifact_root: Path
) -> DiagnosticReceipt | IncompleteDiagnosticReceipt:
    """Rebuild a receipt from raw referenced bytes and reject every false claim."""

    payload = _mapping(value, "diagnostic receipt")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "plan_sha256",
                "evidence_refs",
                "quality",
                "phase_attribution",
                "verdict",
                "historical_aggregate_relation",
                "next_route",
                "reuse_opportunity_estimate",
                "failure",
                "engineering_campaign_receipt_produced",
                "promotion_authorized",
                "formal_comparison",
            }
        ),
        "diagnostic receipt",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["route"] != ROUTE
        or payload["numerical_route"] != NUMERICAL_ROUTE
        or payload["plan_sha256"] != PLAN_SHA256
        or payload["engineering_campaign_receipt_produced"] is not False
        or payload["promotion_authorized"] is not False
        or payload["formal_comparison"] != "NOT_PRODUCED"
    ):
        raise ValueError("diagnostic identity or nonpromotion literals differ")
    raw_refs = _mapping(payload["evidence_refs"], "diagnostic evidence refs")
    _exact_keys(raw_refs, EVIDENCE_REF_KEYS, "diagnostic evidence refs")
    if payload["failure"] is not None:
        failure = _mapping(payload["failure"], "diagnostic failure")
        _exact_keys(failure, frozenset({"stage", "reasons"}), "diagnostic failure")
        _claimed_stage = FailureStage(_string(failure["stage"], "failure stage"))
        _claimed_reasons = tuple(
            _string(item, "diagnostic failure reason")
            for item in _array(failure["reasons"], "diagnostic failure reasons")
        )
        optional_refs = {
            name: (
                None
                if raw_refs[name] is None
                else _artifact_ref(raw_refs[name], f"diagnostic evidence refs.{name}")
            )
            for name in EVIDENCE_REF_KEYS
        }
        for reference in optional_refs.values():
            if reference is not None:
                _resolve_artifact(artifact_root, reference)
        rebuilt_failure = build_incomplete_diagnostic_receipt(
            artifact_root=artifact_root,
            evidence_refs=optional_refs,
        )
        if payload != diagnostic_receipt_payload(rebuilt_failure):
            raise ValueError("incomplete diagnostic claims differ from raw evidence")
        return rebuilt_failure
    refs = {
        name: _artifact_ref(raw_refs[name], f"diagnostic evidence refs.{name}")
        for name in EVIDENCE_REF_KEYS
    }
    rebuilt = build_diagnostic_receipt(artifact_root=artifact_root, evidence_refs=refs)
    if payload != diagnostic_receipt_payload(rebuilt):
        raise ValueError("diagnostic receipt claims differ from raw evidence")
    return rebuilt


def load_diagnostic_receipt_bytes(
    data: bytes, *, artifact_root: Path
) -> DiagnosticReceipt | IncompleteDiagnosticReceipt:
    return diagnostic_receipt_from_payload(
        load_canonical_json_bytes(data), artifact_root=artifact_root
    )


def _receipt_evidence_references(
    root: Path,
) -> dict[str, ArtifactRef | None]:
    payload = _mapping(
        load_canonical_json_bytes((root / RECEIPT_FILENAME).read_bytes()),
        "diagnostic receipt",
    )
    raw_refs = _mapping(payload.get("evidence_refs"), "diagnostic evidence refs")
    _exact_keys(raw_refs, EVIDENCE_REF_KEYS, "diagnostic evidence refs")
    return {
        name: (
            None
            if raw_refs[name] is None
            else _artifact_ref(raw_refs[name], f"diagnostic evidence refs.{name}")
        )
        for name in EVIDENCE_REF_KEYS
    }


def _diagnostic_artifact_roles(root: Path) -> dict[str, str]:
    refs = _receipt_evidence_references(root)
    roles = {RECEIPT_FILENAME: FIXED_ARTIFACT_ROLES[RECEIPT_FILENAME]}
    for name, reference in refs.items():
        if reference is None:
            continue
        if name == "raw_trace":
            relative = Path(reference.relative_path)
            expected_prefix = Path("cold/raw-trace/plugins/profile")
            if (
                relative.suffixes[-3:] != [".trace", ".json", ".gz"]
                or relative.parent.parent != expected_prefix
            ):
                raise ValueError("raw Chrome trace path differs from the frozen layout")
            stem = relative.name.removesuffix(".trace.json.gz")
            if not stem:
                raise ValueError("raw Chrome trace basename is empty")
            xplane = relative.with_name(f"{stem}.xplane.pb").as_posix()
            roles[reference.relative_path] = "raw_trace_chrome"
            roles[xplane] = "raw_trace_xplane"
            continue
        expected_path = EVIDENCE_ROLE_PATHS[name]
        if reference.relative_path != expected_path:
            raise ValueError(f"{name} evidence path differs from the frozen layout")
        roles[expected_path] = FIXED_ARTIFACT_ROLES[expected_path]

    trace_root = root / "cold/raw-trace"
    retained_trace_paths = (
        sorted(path for path in trace_root.rglob("*") if path.is_file())
        if trace_root.is_dir()
        else []
    )
    if refs["raw_trace"] is None and retained_trace_paths:
        chrome = [
            path
            for path in retained_trace_paths
            if path.name.endswith(".trace.json.gz")
        ]
        xplane = [
            path for path in retained_trace_paths if path.name.endswith(".xplane.pb")
        ]
        if (
            len(chrome) > 1
            or len(xplane) > 1
            or len(chrome) + len(xplane) != len(retained_trace_paths)
        ):
            raise ValueError("partial raw trace files differ from the frozen layout")
        for path in retained_trace_paths:
            relative = path.relative_to(root)
            if relative.parent.parent != Path("cold/raw-trace/plugins/profile"):
                raise ValueError(
                    "partial raw trace path differs from the frozen layout"
                )
        if chrome and xplane:
            chrome_stem = chrome[0].name.removesuffix(".trace.json.gz")
            xplane_stem = xplane[0].name.removesuffix(".xplane.pb")
            if chrome[0].parent != xplane[0].parent or chrome_stem != xplane_stem:
                raise ValueError("partial raw trace siblings do not share a basename")
        for path in chrome:
            roles[path.relative_to(root).as_posix()] = "raw_trace_chrome"
        for path in xplane:
            roles[path.relative_to(root).as_posix()] = "raw_trace_xplane"

    for process_name, prefix in (
        ("preflight_process", "preflight"),
        ("process", "cold"),
    ):
        reference = refs[process_name]
        if reference is None:
            continue
        process = _mapping(
            _load_ref_json(root, reference, f"{prefix} process"),
            f"{prefix} process",
        )
        for stream in ("stdout", "stderr"):
            stream_ref = _artifact_ref(process[stream], f"{prefix} process.{stream}")
            expected = f"{prefix}/{stream}.bin"
            if stream_ref.relative_path != expected:
                raise ValueError(
                    f"{prefix} {stream} path differs from the frozen layout"
                )
            _resolve_artifact(root, stream_ref)
            roles[expected] = FIXED_ARTIFACT_ROLES[expected]

    terminal_reference = refs["terminal_numerical"]
    if terminal_reference is not None:
        terminal = _mapping(
            _load_ref_json(root, terminal_reference, "terminal numerical evidence"),
            "terminal numerical evidence",
        )
        arrays = _mapping(terminal["arrays"], "terminal arrays")
        _exact_keys(arrays, frozenset(ARRAY_SPECS), "terminal arrays")
        for name in ARRAY_SPECS:
            row = _mapping(arrays[name], f"terminal arrays.{name}")
            reference = _artifact_ref(row["artifact"], f"terminal arrays.{name}")
            expected = f"cold/arrays/{name}.npy"
            if reference.relative_path != expected:
                raise ValueError(f"terminal array {name} path differs")
            _resolve_artifact(root, reference)
            roles[expected] = "terminal_array"

    if refs["source_manifest"] is not None:
        load_snapshot(root / "source-snapshot")
        roles.update(
            {
                path.relative_to(root).as_posix(): "source_snapshot"
                for path in (root / "source-snapshot").rglob("*")
                if path.is_file()
            }
        )
    else:
        source_root = root / "source-snapshot"
        retained_source_paths = (
            sorted(path for path in source_root.rglob("*") if path.is_file())
            if source_root.is_dir()
            else []
        )
        if retained_source_paths:
            receipt_payload = _mapping(
                load_canonical_json_bytes((root / RECEIPT_FILENAME).read_bytes()),
                "diagnostic receipt",
            )
            failure = _mapping(receipt_payload.get("failure"), "diagnostic failure")
            if failure.get("stage") != FailureStage.COLD_SOURCE.value:
                raise ValueError(
                    "opaque source snapshot bytes require a derived source failure"
                )
            roles.update(
                {
                    path.relative_to(root).as_posix(): "source_snapshot_opaque_failure"
                    for path in retained_source_paths
                }
            )
    if refs["native_reference"] is not None:
        validate_native_equivalent_reference(root / "native-reference")
        roles.update(
            {
                path.relative_to(root).as_posix(): "native_reference"
                for path in (root / "native-reference").rglob("*")
                if path.is_file()
            }
        )
    return roles


def diagnostic_artifact_manifest_payload(root: Path) -> dict[str, JsonValue]:
    """Build the exact role-bearing manifest from receipt-bound raw authorities."""

    roles = _diagnostic_artifact_roles(root)
    entries: list[dict[str, JsonValue]] = []
    for relative, role in sorted(roles.items()):
        data = (root / relative).read_bytes()
        entries.append(
            {
                "relative_path": relative,
                "role": role,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "entries": entries}


def _validate_manifest(root: Path) -> frozenset[str]:
    if root.is_symlink():
        raise ValueError("diagnostic root must not be a symlink")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("diagnostic root must be a directory")
    for path in (resolved, *resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError("diagnostic artifact contains a symlink")
        if not path.is_file() and not path.is_dir():
            raise ValueError("diagnostic artifact contains a non-regular path")
        mode = stat.S_IMODE(path.stat().st_mode)
        expected_mode = 0o444 if path.is_file() else 0o555
        if mode != expected_mode:
            raise ValueError("diagnostic artifact modes differ from the seal contract")
    manifest_path = resolved / MANIFEST_FILENAME
    manifest = _mapping(
        load_canonical_json_bytes(manifest_path.read_bytes()), "artifact manifest"
    )
    _exact_keys(manifest, frozenset({"schema_version", "entries"}), "artifact manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("artifact manifest schema differs")
    expected_roles = _diagnostic_artifact_roles(resolved)
    declared: list[str] = []
    for index, item in enumerate(_array(manifest["entries"], "manifest.entries")):
        context = f"manifest.entries[{index}]"
        row = _mapping(item, context)
        _exact_keys(
            row,
            frozenset({"relative_path", "role", "sha256", "size_bytes"}),
            context,
        )
        relative = _string(row["relative_path"], f"{context}.relative_path")
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or relative == MANIFEST_FILENAME
        ):
            raise ValueError(f"{context}.relative_path is not canonical")
        role = _string(row["role"], f"{context}.role")
        if expected_roles.get(relative) != role:
            raise ValueError(f"{context} logical role differs")
        data = (resolved / path).read_bytes()
        if len(data) != _integer(
            row["size_bytes"], f"{context}.size_bytes"
        ) or hashlib.sha256(data).hexdigest() != _sha256(
            row["sha256"], f"{context}.sha256"
        ):
            raise ValueError(f"{context} bytes differ")
        declared.append(relative)
    if declared != sorted(declared) or len(declared) != len(set(declared)):
        raise ValueError("artifact manifest paths must be sorted and unique")
    observed = frozenset(
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if observed != frozenset(declared) or observed != frozenset(expected_roles):
        raise ValueError("diagnostic artifact has missing or extra files")
    if RECEIPT_FILENAME not in observed:
        raise ValueError("diagnostic artifact omits its receipt")
    if any(
        "warm" in Path(path).parts or Path(path).name == "campaign.json"
        for path in observed
    ):
        raise ValueError("diagnostic artifact contains a forbidden campaign sample")
    return observed


def load_and_validate_diagnostic_artifact(
    artifact_root: Path,
) -> DiagnosticReceipt | IncompleteDiagnosticReceipt:
    """Validate the exact sealed tree and independently recompute its receipt."""

    _validate_manifest(artifact_root)
    return load_diagnostic_receipt_bytes(
        (artifact_root / RECEIPT_FILENAME).read_bytes(), artifact_root=artifact_root
    )


# DIAG2 is deliberately additive.  The v1 constants and entry points above remain
# byte-compatible so sealed historical readers never acquire v2 fallback behavior.
DIAG2_PLAN_SHA256: Final = (
    "38bf768c8c851347e9178596f6dcec8f3fb43ff88030a3dd953066999df97f78"
)
DIAG2_ROUTE: Final = "NEQ-GNTR1-DIAG2"
DIAG2_NUMERICAL_ROUTE: Final = NUMERICAL_ROUTE
DIAG2_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-no-hit-diagnostic-v2"
DIAG2_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-no-hit-diagnostic-artifact-manifest-v2"
)
DIAG2_RECEIPT_FILENAME: Final = RECEIPT_FILENAME
DIAG2_MANIFEST_FILENAME: Final = MANIFEST_FILENAME
DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-policy-authority-v2"
)
DIAG2_PROCESS_SCHEMA_VERSION: Final = f"{DIAG2_SCHEMA_VERSION}-process"
DIAG2_CHILD_TERMINAL_SCHEMA_VERSION: Final = f"{DIAG2_SCHEMA_VERSION}-child-terminal"
DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-supervisor-gpu-zero-v1"
)
DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-supervisor-terminal-v2"
)
DIAG2_FROZEN_SUBSET_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-frozen-numerical-subset-v1"
)
DIAG3_COLD_RESULT_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-command-buffer-recovery-cold-result-v1"
)
DIAG3_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-no-hit-diagnostic-v3"
DIAG3_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-no-hit-diagnostic-artifact-manifest-v3"
)
DIAG3_COMMITTED_NUMERICAL_DIRECTORY: Final = "cold/numerical-result"
DIAG3_PENDING_NUMERICAL_DIRECTORY: Final = "cold/.numerical-result.pending"
DIAG3_UNCOMMITTED_NUMERICAL_DIRECTORY: Final = "cold/uncommitted-numerical-result"
DIAG4_ROUTE: Final = "NEQ-GNTR3-DIAG4"
DIAG4_NUMERICAL_ROUTE: Final = "NEQ-GNTR3"
DIAG4_PLAN_SHA256: Final = (
    "987dd67227431a90dd851d4d8ab78f639f9964c57de5ac093fc15f5aac504e5c"
)
DIAG4_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-trace-free-diagnostic-v1"
DIAG4_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-artifact-manifest-v1"
)
DIAG4_COLD_RESULT_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-cold-result-v1"
)
DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-neq-gntr3-result-v1"
)
DIAG4_NUMERICAL_BUNDLE_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-numerical-bundle-v1"
)
DIAG4_SOLVE_TIMING_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-solve-timing-v1"
DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-step-bound-safeguard-telemetry-v1"
)
DIAG4_EXECUTION_SCHEMA_VERSION: Final = f"{DIAG4_SCHEMA_VERSION}-execution"
DIAG4_PREFLIGHT_SCHEMA_VERSION: Final = f"{DIAG4_SCHEMA_VERSION}-preflight"
DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION: Final = (
    f"{DIAG4_SCHEMA_VERSION}-supervisor-terminal"
)
DIAG4_ENGINEERING_THRESHOLD_SECONDS: Final = 287.30421751597896
DIAG4_MAXIMUM_NONLINEAR_CORRECTIONS: Final = 2
DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS: Final = 3
DIAG4_OUTER_TELEMETRY_FIELDS: Final = (
    "nonlinear_corrections",
    "maximum_individual_correction_step_ratio",
    "correction_path_step_ratio",
    "steihaug_solve_calls",
)
DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS: Final = (
    "subtrial_trust_radius",
    "subtrial_actual_reduction",
    "subtrial_predicted_reduction",
    "subtrial_maximum_individual_correction_step_ratio",
    "subtrial_correction_path_step_ratio",
    "subtrial_corrected_radius_ratio",
)
DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS: Final = (
    "subtrial_steihaug_iterations",
    "subtrial_steihaug_hvp_evaluations",
    "subtrial_steihaug_solve_calls",
    "subtrial_total_hvp_evaluations",
    "subtrial_nonlinear_corrections",
    "subtrial_joint_evaluations",
    "subtrial_joint_linearizations",
    "subtrial_joint_value_evaluations",
    "subtrial_objective_residual_linearizations",
    "subtrial_gram_factorizations",
    "subtrial_gram_solves",
)
DIAG4_SUBTRIAL_MATRIX_FIELDS: Final = (
    DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS[0],
    "subtrial_outcome",
    *DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS[1:],
    *DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS,
)
DIAG4_SUBTRIAL_SUMMARY_FIELDS: Final = (
    "total_subtrials",
    "total_shadow_subtrials",
    "maximum_subtrial_count",
    "logical_attempts_with_1_subtrial",
    "logical_attempts_with_2_subtrials",
    "logical_attempts_with_3_subtrials",
    "recovered_step_bound_attempts",
    "exhausted_step_bound_attempts",
    "total_steihaug_iterations",
    "total_steihaug_hvp_evaluations",
    "total_steihaug_solve_calls",
    "total_hvp_evaluations",
    "total_nonlinear_corrections",
    "total_joint_evaluations",
    "total_joint_linearizations",
    "total_joint_value_evaluations",
    "total_objective_residual_linearizations",
    "total_gram_factorizations",
    "total_gram_solves",
)
DIAG4_CONDITIONAL_TIMING_ROUTE: Final = "CONDITIONAL_ENGINEERING_TIMING"

# DIAG5 is an additive wire generation. Shared scientific v1 documents retain
# their original schemas; only the identity-bearing documents listed here move
# to v2. Legacy loaders remain intentionally unaware of every constant below.
DIAG5_PLAN_SHA256: Final = (
    "786a7d3da6252aa04704d0532158e8fde96833bcf07d81b639f6313d72e857a4"
)
DIAG5_ROUTE: Final = "NEQ-GNTR3-DIAG5"
DIAG5_NUMERICAL_ROUTE: Final = DIAG4_NUMERICAL_ROUTE
DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-neq-gntr3-result-v1"
)
DIAG5_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-trace-free-diagnostic-v2"
DIAG5_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-artifact-manifest-v2"
)
DIAG5_FROZEN_SUBSET_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-frozen-numerical-subset-v2"
)
DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-policy-authority-v2"
)
DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag5-supervisor-gpu-zero-v1"
)
DIAG5_PREFLIGHT_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-trace-free-preflight-v2"
DIAG5_COLD_RESULT_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-cold-result-v2"
)
DIAG5_NUMERICAL_BUNDLE_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-numerical-bundle-v2"
)
DIAG5_CHILD_TERMINAL_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-child-terminal-v2"
)
DIAG5_PROCESS_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-trace-free-process-v2"
DIAG5_MEMORY_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-memory-v2"
DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-memory-samples-v2"
)
DIAG5_SOLVE_TIMING_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-solve-timing-v2"
DIAG5_SAFEGUARD_TELEMETRY_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-step-bound-safeguard-telemetry-v2"
)
DIAG5_EXECUTION_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-trace-free-execution-v2"
DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-trace-free-supervisor-terminal-v2"
)
DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag4-independent-postmortem-v1"
)
DIAG5_PREDECESSOR_POSTMORTEM_PATH: Final = "control/predecessor-postmortem.json"
DIAG5_PENDING_NUMERICAL_DIRECTORY: Final = "cold/.numerical-result.pending"
DIAG5_COMMITTED_NUMERICAL_DIRECTORY: Final = "cold/numerical-result"
DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY: Final = "cold/uncommitted-numerical-result"
DIAG5_EMPTY_QUARANTINE_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-empty-quarantine-v1"
)
DIAG5_EMPTY_QUARANTINE_PATH: Final = "cold/uncommitted-numerical-result.empty.json"


def validate_diag5_predecessor_postmortem_payload(
    value: JsonValue,
) -> dict[str, JsonValue]:
    payload = _mapping(value, "DIAG5 predecessor postmortem")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "session_reference",
                "original_process_receipt",
                "original_stdout_retained",
                "original_stderr_retained",
                "reconstruction",
            }
        ),
        "DIAG5 predecessor postmortem",
    )
    reconstruction = _mapping(
        payload["reconstruction"], "DIAG5 predecessor reconstruction"
    )
    required = frozenset(
        {
            "command_text",
            "copied_qualifier_predicate",
            "copied_tree_entry_count",
            "exception_class",
            "exception_message",
            "execution_entries_sha256",
            "execution_manifest_sha256",
            "execution_source_entry_count",
            "failed_stage",
            "final_root_absent",
            "native_binding",
            "partial_root",
            "predecessor_full_tree_sha256",
            "prior_reviews_retracted",
            "qualifier_sha256",
            "retracted_reviews_sha256",
            "scientific_paths_absent",
        }
    )
    _exact_keys(reconstruction, required, "DIAG5 predecessor reconstruction")
    if (
        payload["schema_version"] != DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
        or payload["session_reference"] != "74963"
        or payload["original_process_receipt"] != "NOT_PRODUCED"
        or payload["original_stdout_retained"] is not False
        or payload["original_stderr_retained"] is not False
        or reconstruction["failed_stage"] != "NATIVE_EXTENSION_RUNTIME_BINDING"
        or reconstruction["final_root_absent"] is not True
        or reconstruction["scientific_paths_absent"] is not True
        or reconstruction["command_text"]
        != "env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv-qn-cpu/bin/python benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py --output-root /home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-cpu-qualification-20260811T214932Z"
        or reconstruction["copied_qualifier_predicate"] != "observed.st_nlink != 1"
        or reconstruction["copied_tree_entry_count"] != 604
        or reconstruction["execution_source_entry_count"] != 603
        or reconstruction["exception_class"] != "QualificationError"
        or reconstruction["exception_message"]
        != "native extension runtime binding differs"
        or reconstruction["execution_entries_sha256"]
        != "7b921fed75c8a0154833ee4acf16a82922ce11b4d93dc52154dc54cc71d248b2"
        or reconstruction["execution_manifest_sha256"]
        != "386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604"
        or reconstruction["predecessor_full_tree_sha256"]
        != "c04cbbb79650990ab38e497bd48d6d7ab9cc2714941c58e3ce91e4147997436a"
        or reconstruction["qualifier_sha256"]
        != "fbe302885c5b392958fb69ed5081edc0d69104573f19843c5be480c37af44c51"
        or reconstruction["partial_root"]
        != "/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-cpu-qualification-20260811T214932Z.partial-claim"
    ):
        raise ValueError("DIAG5 predecessor postmortem semantics differ")
    for name in (
        "execution_entries_sha256",
        "execution_manifest_sha256",
        "predecessor_full_tree_sha256",
        "qualifier_sha256",
        "retracted_reviews_sha256",
    ):
        _sha256(reconstruction[name], f"DIAG5 predecessor {name}")
    binding = _mapping(reconstruction["native_binding"], "predecessor native binding")
    _exact_keys(
        binding,
        frozenset(
            {"path", "loader", "sha256", "size_bytes", "device", "inode", "link_count"}
        ),
        "predecessor native binding",
    )
    _sha256(binding["sha256"], "predecessor native SHA")
    if binding != {
        "path": "/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-cpu/lib/python3.11/site-packages/simsoptpp.cpython-311-x86_64-linux-gnu.so",
        "loader": "_ScikitBuildLoaderWrapper",
        "sha256": "41b2ca791a720f325ffa9b382b31d29bade73f6516693805d41adc0de6f6ed4b",
        "size_bytes": 2883776,
        "device": 66306,
        "inode": 50480769,
        "link_count": 2,
    }:
        raise ValueError("DIAG5 predecessor native topology differs")
    reviews = _array(reconstruction["prior_reviews_retracted"], "retracted reviews")
    if len(reviews) != 4:
        raise ValueError("DIAG5 predecessor retracted review count differs")
    review_keys = frozenset(
        {
            "reviewed_execution_source_entries_sha256",
            "reviewed_execution_source_manifest_sha256",
            "reviewed_frozen_numerical_entries_sha256",
            "reviewed_plan_full_sha256",
            "reviewed_plan_prefix_sha256",
            "reviewed_qualified_files_sha256",
            "reviewer",
            "role",
            "session",
            "verdict",
        }
    )
    for review in reviews:
        _exact_keys(
            _mapping(review, "retracted review"), review_keys, "retracted review"
        )
    shared_review_fields: dict[str, JsonValue] = {
        "reviewed_execution_source_entries_sha256": (
            "7b921fed75c8a0154833ee4acf16a82922ce11b4d93dc52154dc54cc71d248b2"
        ),
        "reviewed_execution_source_manifest_sha256": (
            "386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604"
        ),
        "reviewed_frozen_numerical_entries_sha256": (
            "57a3bf08fad41871812322b516f994a8e66abe2104c0e8ed0055688e3209f7e0"
        ),
        "reviewed_plan_full_sha256": (
            "5c27a90047291774955858f1b86502bfeb0aec900c733f53d8a29c0dbe41a770"
        ),
        "reviewed_plan_prefix_sha256": (
            "987dd67227431a90dd851d4d8ab78f639f9964c57de5ac093fc15f5aac504e5c"
        ),
        "reviewed_qualified_files_sha256": (
            "e1938b81503c696bd5dc796045cdd8164e14453420b48fb38fb0f89b35ddbcc8"
        ),
        "verdict": "RETRACTED",
    }
    expected_reviews: tuple[dict[str, JsonValue], ...] = tuple(
        {
            **shared_review_fields,
            "role": role,
            "reviewer": reviewer,
            "session": session,
        }
        for role, reviewer, session in (
            (
                "numerical-controller",
                "codex-numerical-controller-current-manifest",
                "numerical-controller-20260811T220006-manifest386698c5",
            ),
            (
                "receipt-schema",
                "codex-receipt-schema-a55a4fac",
                "5c87cc42-3234-4b9f-bcd8-3eee3e0ea01d",
            ),
            (
                "source-snapshot",
                "/root/ftr_runner_receipt",
                "source-snapshot-final-20260811-ftr01",
            ),
            (
                "atomic-lifecycle",
                "codex-atomic-lifecycle-current-manifest",
                "/root/diag_runner_map/ssot_atomic_review@2026-08-12T02:01:09Z",
            ),
        )
    )
    if tuple(reviews) != expected_reviews:
        raise ValueError("DIAG5 predecessor retracted review rows differ")
    if hashlib.sha256(canonical_json_bytes(reviews)).hexdigest() != _sha256(
        reconstruction["retracted_reviews_sha256"], "retracted reviews SHA"
    ) or reconstruction["retracted_reviews_sha256"] != (
        "062e35d183f9618d5b0ca6cf7011c0c500ed3f6c7c0c1685e01262feeb5a4111"
    ):
        raise ValueError("DIAG5 predecessor retracted reviews differ")
    return payload


DIAG2_EVIDENCE_SLOT_PATHS: Final = {
    "source_manifest": "source-snapshot/source-manifest.json",
    "frozen_numerical_subset": "frozen-numerical-subset.json",
    "native_reference": "native-reference/reference.json",
    "policy_authority": "policy-authority.json",
    "supervisor_before_preflight": "supervisor/before-preflight.json",
    "preflight_producer": "preflight/producer.json",
    "preflight_terminal": "preflight/terminal.json",
    "preflight_process": "preflight/process.json",
    "preflight_memory": "preflight/gpu-memory.json",
    "preflight_memory_samples": "preflight/gpu-memory-samples.json",
    "preflight_runtime": "preflight/runtime-evidence.json",
    "preflight_policy": "preflight/policy.json",
    "supervisor_before_cold": "supervisor/before-cold.json",
    "cold_producer": "cold/producer.json",
    "cold_terminal": "cold/terminal.json",
    "cold_process": "cold/process.json",
    "cold_memory": "cold/gpu-memory.json",
    "cold_memory_samples": "cold/gpu-memory-samples.json",
    "cold_runtime": "cold/runtime-evidence.json",
    "cold_policy": "cold/policy.json",
    "cold_history": "cold/history.json",
    "cold_terminal_numerical": "cold/terminal-numerical.json",
    "cold_raw_trace": "cold/raw-trace/plugins/profile/<run>/<base>.trace.json.gz",
    "cold_trace_intervals": "cold/trace-intervals.json",
    "execution": "execution.json",
    "supervisor_terminal": "supervisor-terminal.json",
}
DIAG2_EVIDENCE_SLOT_NAMES: Final = frozenset(DIAG2_EVIDENCE_SLOT_PATHS)
DIAG3_EVIDENCE_SLOT_PATHS: Final = {
    **DIAG2_EVIDENCE_SLOT_PATHS,
    "cold_history": f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/history.json",
    "cold_terminal_numerical": (
        f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/terminal-numerical.json"
    ),
    "cold_raw_trace": (
        f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/raw-trace/plugins/profile"
    ),
    "cold_trace_intervals": (
        f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/trace-intervals.json"
    ),
}
DIAG4_EVIDENCE_SLOT_PATHS: Final = {
    name: path
    for name, path in DIAG3_EVIDENCE_SLOT_PATHS.items()
    if name
    not in {
        "cold_raw_trace",
        "cold_trace_intervals",
        "execution",
        "supervisor_terminal",
    }
}
DIAG4_EVIDENCE_SLOT_PATHS.update(
    {
        "cold_solve_timing": (
            f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/solve-timing.json"
        ),
        "cold_safeguard_telemetry": (
            f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/safeguard-telemetry.json"
        ),
        "execution": DIAG3_EVIDENCE_SLOT_PATHS["execution"],
        "supervisor_terminal": DIAG3_EVIDENCE_SLOT_PATHS["supervisor_terminal"],
    }
)
DIAG4_EVIDENCE_SLOT_NAMES: Final = frozenset(DIAG4_EVIDENCE_SLOT_PATHS)
DIAG5_EVIDENCE_SLOT_PATHS: Final = {
    "source_manifest": "source-snapshot/source-manifest.json",
    "frozen_numerical_subset": "frozen-numerical-subset.json",
    "native_reference": "native-reference/reference.json",
    "policy_authority": "policy-authority.json",
    "supervisor_before_preflight": "supervisor/before-preflight.json",
    "preflight_producer": "preflight/producer.json",
    "preflight_terminal": "preflight/terminal.json",
    "preflight_process": "preflight/process.json",
    "preflight_memory": "preflight/gpu-memory.json",
    "preflight_memory_samples": "preflight/gpu-memory-samples.json",
    "preflight_runtime": "preflight/runtime-evidence.json",
    "preflight_policy": "preflight/policy.json",
    "supervisor_before_cold": "supervisor/before-cold.json",
    "cold_producer": "cold/producer.json",
    "cold_terminal": "cold/terminal.json",
    "cold_process": "cold/process.json",
    "cold_memory": "cold/gpu-memory.json",
    "cold_memory_samples": "cold/gpu-memory-samples.json",
    "cold_runtime": "cold/runtime-evidence.json",
    "cold_policy": "cold/policy.json",
    "cold_history": f"{DIAG5_COMMITTED_NUMERICAL_DIRECTORY}/history.json",
    "cold_terminal_numerical": f"{DIAG5_COMMITTED_NUMERICAL_DIRECTORY}/terminal-numerical.json",
    "cold_solve_timing": f"{DIAG5_COMMITTED_NUMERICAL_DIRECTORY}/solve-timing.json",
    "cold_safeguard_telemetry": f"{DIAG5_COMMITTED_NUMERICAL_DIRECTORY}/safeguard-telemetry.json",
    "execution": "execution.json",
    "supervisor_terminal": "supervisor-terminal.json",
}
DIAG5_EVIDENCE_SLOT_NAMES: Final = frozenset(DIAG5_EVIDENCE_SLOT_PATHS)
DIAG5_EVIDENCE_SLOT_SCHEMAS: Final = {
    "source_manifest": SOURCE_MANIFEST_SCHEMA_VERSION,
    "frozen_numerical_subset": DIAG5_FROZEN_SUBSET_SCHEMA_VERSION,
    "native_reference": NATIVE_REFERENCE_SCHEMA_VERSION,
    "policy_authority": DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION,
    "supervisor_before_preflight": DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION,
    "preflight_producer": DIAG5_PREFLIGHT_SCHEMA_VERSION,
    "preflight_terminal": DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
    "preflight_process": DIAG5_PROCESS_SCHEMA_VERSION,
    "preflight_memory": DIAG5_MEMORY_SCHEMA_VERSION,
    "preflight_memory_samples": DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
    "preflight_runtime": RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
    "preflight_policy": "single-stage-native-equivalent-quality-policy-v1",
    "supervisor_before_cold": DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION,
    "cold_producer": DIAG5_COLD_RESULT_SCHEMA_VERSION,
    "cold_terminal": DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
    "cold_process": DIAG5_PROCESS_SCHEMA_VERSION,
    "cold_memory": DIAG5_MEMORY_SCHEMA_VERSION,
    "cold_memory_samples": DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
    "cold_runtime": RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
    "cold_policy": "single-stage-native-equivalent-quality-policy-v1",
    "cold_history": "single-stage-fullspace-neq-gntr3-history-v1",
    "cold_terminal_numerical": f"{DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal",
    "cold_solve_timing": DIAG5_SOLVE_TIMING_SCHEMA_VERSION,
    "cold_safeguard_telemetry": DIAG5_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
    "execution": DIAG5_EXECUTION_SCHEMA_VERSION,
    "supervisor_terminal": DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
}
DIAG2_FROZEN_NUMERICAL_ENTRIES: Final = (
    (
        "benchmarks/single_stage_native_equivalent_reference.py",
        "faf7614ad827e3603b1ba8e4a792394e50fb8be2146bff5bb34f002cb41d96e6",
    ),
    (
        "examples/jax/parity/cases/native_boozerqa.py",
        "3bf7c04ec64b340a7dbb8c08b0cd55cbc0bbd0cb41942976cd33451087894832",
    ),
    (
        "examples/jax/parity/input_bundle.py",
        "303439ea4dcf9b444ad3410c088fb17bc25cd4701dabc88bcf5106faf9a8e87b",
    ),
    (
        "src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py",
        "c28a598a56eae109b3e61f846ae58c34b97a2cdc5fe92fdb15af0a668eb380de",
    ),
    (
        "src/simsopt_jax/objectives/single_stage_fullspace.py",
        "ca3a09f57fcabe4e448b9c50256bf28cc3750005cf52199ace2061d3e55f19fd",
    ),
    (
        "src/simsopt_jax/runtime/trace_annotations.py",
        "9d50e5fca9dddc8b933f5039beb0ed5f25339dea78e2c5a12bacf67489881ea7",
    ),
    (
        "src/simsopt_jax/solve/fullspace.py",
        "475cb63ddc183e343c1ae40faf7e0abf8bad5e6c288eabe38d31ce416e18cde4",
    ),
    (
        "src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py",
        "62b7dec2194f7c381d676abeed852ff1c4acba9e1a5f8d764a845abcd040f436",
    ),
    (
        "src/simsopt_jax/solve/fullspace_native_equivalent_quality.py",
        "abf9726e487eb4bda9f82c6092415e988e5a346383c89cec732fe7185b6e6fac",
    ),
    (
        "src/simsopt_jax_adapters/geo/single_stage_fullspace.py",
        "910b59131cc9137fee65a8d14222eeccbc0cf3d61d300a63250a95469c413e4e",
    ),
    (
        "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py",
        "bad745833c598072e3b205599fd55eb4e35dec61e87fa6679552b8343d9d2934",
    ),
)
DIAG2_EXECUTED_DIAG1_SOURCE_MANIFEST_SHA256: Final = (
    "d33001f37fadd3b06d04a1fa3ac6f51075afe9da9c400efe2c3558c9c2ba6cfd"
)
DIAG2_SOURCE_DELTA_ALLOWLIST: Final = frozenset(
    {
        "benchmarks/process_gpu_monitor.py",
        "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
        "benchmarks/run_single_stage_native_equivalent_quality_campaign.py",
        "benchmarks/single_stage_fullspace_process_gpu_monitor.py",
        "benchmarks/single_stage_fullspace_snapshot.py",
        "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py",
        "benchmarks/single_stage_native_equivalent_quality_successor_authority.py",
        "docs/single_stage_jax_gpu_native_equivalent_quality_diag2_implementation_plan.md",
        "docs/single_stage_jax_gpu_native_equivalent_quality_diag3_command_buffer_recovery_plan.md",
        "docs/single_stage_jax_gpu_native_equivalent_quality_diag3_command_buffer_recovery_authorization.json",
        "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md",
        "docs/single_stage_jax_gpu_native_equivalent_quality_no_hit_diagnostic_implementation_plan.md",
        "tests/benchmarks/test_process_gpu_monitor.py",
        "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py",
        "tests/benchmarks/_diag2_fixture.py",
        "tests/benchmarks/test_single_stage_fullspace_snapshot.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py",
    }
)
DIAG2_BASELINE_FILTERED_ENTRY_COUNT: Final = 576
DIAG2_BASELINE_FILTERED_ENTRIES_SHA256: Final = (
    "bb96669db5145f4dec681ae2ecb5a51d71fce1a480cb8d5c372a422bdedef2bb"
)
DIAG2_VOLUME_TARGET_HEX: Final = "-0x1.296a9ce4a271dp-2"
DIAG2_BOOZER_SCALE_HEX: Final = "0x1.0101828467ee9p-4"
DIAG2_VOLUME_SCALE_HEX: Final = "0x1.b8b3b0469c959p+1"
DIAG2_SCALE_SHA256: Final = (
    "ee71932a5d6a0dfb0ca4dc9d852bf1f32e669dbc81ced26903db956027e1155e"
)
DIAG2_POLICY_SHA256: Final = (
    "6face7116d36d2eae954bb5b3bde465f37c990ceb9b319ba2c20bb62ad1a6f99"
)


class EvidenceState(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class FailureReasonCodeV2(StrEnum):
    SOURCE_PRE = "SOURCE_PRE"
    SOURCE_POST = "SOURCE_POST"
    FROZEN_SUBSET_INVALID = "FROZEN_SUBSET_INVALID"
    REFERENCE_INVALID = "REFERENCE_INVALID"
    POLICY_DERIVATION_INVALID = "POLICY_DERIVATION_INVALID"
    GPU_QUERY_FAILED = "GPU_QUERY_FAILED"
    GPU_PARENT_PID_PRESENT = "GPU_PARENT_PID_PRESENT"
    CHILD_LAUNCH_FAILED = "CHILD_LAUNCH_FAILED"
    CHILD_TIMEOUT = "CHILD_TIMEOUT"
    CHILD_COMPILE_FAILED = "CHILD_COMPILE_FAILED"
    CHILD_COMPILE_OOM = "CHILD_COMPILE_OOM"
    CHILD_EXIT_NONZERO = "CHILD_EXIT_NONZERO"
    PRODUCER_DECODE_FAILED = "PRODUCER_DECODE_FAILED"
    PRODUCER_SCHEMA_INVALID = "PRODUCER_SCHEMA_INVALID"
    RUNTIME_SCHEMA_INVALID = "RUNTIME_SCHEMA_INVALID"
    POLICY_SCHEMA_INVALID = "POLICY_SCHEMA_INVALID"
    NUMERICAL_SCHEMA_INVALID = "NUMERICAL_SCHEMA_INVALID"
    MONITOR_BINDING_FAILED = "MONITOR_BINDING_FAILED"
    MONITOR_FINALIZATION_FAILED = "MONITOR_FINALIZATION_FAILED"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    TRACE_NORMALIZATION_FAILED = "TRACE_NORMALIZATION_FAILED"
    SEMANTIC_VALIDATION_FAILED = "SEMANTIC_VALIDATION_FAILED"


class AbsenceReason(StrEnum):
    NOT_REACHED = "NOT_REACHED"
    SOURCE_PRE = "SOURCE_PRE"
    SOURCE_POST = "SOURCE_POST"
    FROZEN_SUBSET_INVALID = "FROZEN_SUBSET_INVALID"
    REFERENCE_INVALID = "REFERENCE_INVALID"
    POLICY_DERIVATION_INVALID = "POLICY_DERIVATION_INVALID"
    GPU_QUERY_FAILED = "GPU_QUERY_FAILED"
    GPU_PARENT_PID_PRESENT = "GPU_PARENT_PID_PRESENT"
    CHILD_LAUNCH_FAILED = "CHILD_LAUNCH_FAILED"
    CHILD_TIMEOUT = "CHILD_TIMEOUT"
    CHILD_COMPILE_FAILED = "CHILD_COMPILE_FAILED"
    CHILD_COMPILE_OOM = "CHILD_COMPILE_OOM"
    CHILD_EXIT_NONZERO = "CHILD_EXIT_NONZERO"
    PRODUCER_DECODE_FAILED = "PRODUCER_DECODE_FAILED"
    PRODUCER_SCHEMA_INVALID = "PRODUCER_SCHEMA_INVALID"
    RUNTIME_SCHEMA_INVALID = "RUNTIME_SCHEMA_INVALID"
    POLICY_SCHEMA_INVALID = "POLICY_SCHEMA_INVALID"
    NUMERICAL_SCHEMA_INVALID = "NUMERICAL_SCHEMA_INVALID"
    MONITOR_BINDING_FAILED = "MONITOR_BINDING_FAILED"
    MONITOR_FINALIZATION_FAILED = "MONITOR_FINALIZATION_FAILED"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    TRACE_NORMALIZATION_FAILED = "TRACE_NORMALIZATION_FAILED"
    SEMANTIC_VALIDATION_FAILED = "SEMANTIC_VALIDATION_FAILED"


class FailureStageV2(StrEnum):
    SOURCE_PUBLICATION_FAILURE = "SOURCE_PUBLICATION_FAILURE"
    NATIVE_REFERENCE_FAILURE = "NATIVE_REFERENCE_FAILURE"
    POLICY_AUTHORITY_FAILURE = "POLICY_AUTHORITY_FAILURE"
    GPU_ZERO_BEFORE_PREFLIGHT_FAILURE = "GPU_ZERO_BEFORE_PREFLIGHT_FAILURE"
    PREFLIGHT_SOURCE_FAILURE = "PREFLIGHT_SOURCE_FAILURE"
    PREFLIGHT_SUPERVISOR_FAILURE = "PREFLIGHT_SUPERVISOR_FAILURE"
    PREFLIGHT_TIMEOUT = "PREFLIGHT_TIMEOUT"
    PREFLIGHT_MONITOR_FAILURE = "PREFLIGHT_MONITOR_FAILURE"
    PREFLIGHT_PROTOCOL_FAILURE = "PREFLIGHT_PROTOCOL_FAILURE"
    PREFLIGHT_COMPILE_FAILURE = "PREFLIGHT_COMPILE_FAILURE"
    PREFLIGHT_CRASH = "PREFLIGHT_CRASH"
    PREFLIGHT_RESOURCE_FAILURE = "PREFLIGHT_RESOURCE_FAILURE"
    GPU_ZERO_BEFORE_COLD_FAILURE = "GPU_ZERO_BEFORE_COLD_FAILURE"
    COLD_SOURCE_FAILURE = "COLD_SOURCE_FAILURE"
    COLD_SUPERVISOR_FAILURE = "COLD_SUPERVISOR_FAILURE"
    COLD_TIMEOUT = "COLD_TIMEOUT"
    COLD_MONITOR_FAILURE = "COLD_MONITOR_FAILURE"
    COLD_PROTOCOL_FAILURE = "COLD_PROTOCOL_FAILURE"
    COLD_COMPILE_FAILURE = "COLD_COMPILE_FAILURE"
    COLD_CRASH = "COLD_CRASH"
    COLD_RESOURCE_FAILURE = "COLD_RESOURCE_FAILURE"
    NUMERICAL_EVIDENCE_INCOMPLETE = "NUMERICAL_EVIDENCE_INCOMPLETE"


class FailureStageV4(StrEnum):
    AUTHORITY = "AUTHORITY"
    SETUP = "SETUP"
    BEFORE_PREFLIGHT = "BEFORE_PREFLIGHT"
    PREFLIGHT = "PREFLIGHT"
    BEFORE_COLD = "BEFORE_COLD"
    COLD = "COLD"
    NUMERICAL_COMMIT = "NUMERICAL_COMMIT"
    RECEIPT = "RECEIPT"
    PUBLICATION = "PUBLICATION"
    SCIENTIFIC = "SCIENTIFIC"


class FailureReasonCodeV4(StrEnum):
    AUTHORITY_INVALID = "AUTHORITY_INVALID"
    OUTPUT_ROOT_NOT_ABSENT = "OUTPUT_ROOT_NOT_ABSENT"
    LOCK_CLAIM_FAILED = "LOCK_CLAIM_FAILED"
    IDENTITY_REVALIDATION_FAILED = "IDENTITY_REVALIDATION_FAILED"
    AUTHORITY_ALREADY_CONSUMED = "AUTHORITY_ALREADY_CONSUMED"
    SOURCE_PUBLICATION_FAILED = "SOURCE_PUBLICATION_FAILED"
    FROZEN_NUMERICAL_SUBSET_INVALID = "FROZEN_NUMERICAL_SUBSET_INVALID"
    NATIVE_REFERENCE_INVALID = "NATIVE_REFERENCE_INVALID"
    POLICY_AUTHORITY_INVALID = "POLICY_AUTHORITY_INVALID"
    SETUP_DEEP_LOAD_FAILED = "SETUP_DEEP_LOAD_FAILED"
    SUPERVISOR_GPU_OBSERVATION_INVALID = "SUPERVISOR_GPU_OBSERVATION_INVALID"
    SUPERVISOR_GPU_NONZERO = "SUPERVISOR_GPU_NONZERO"
    AUTHORITY_CONSUMPTION_FAILED = "AUTHORITY_CONSUMPTION_FAILED"
    AUTHORITY_CONSUMPTION_UNCERTAIN = "AUTHORITY_CONSUMPTION_UNCERTAIN"
    PREFLIGHT_LAUNCH_FAILED = "PREFLIGHT_LAUNCH_FAILED"
    PREFLIGHT_TIMEOUT = "PREFLIGHT_TIMEOUT"
    PREFLIGHT_MONITOR_FAILED = "PREFLIGHT_MONITOR_FAILED"
    PREFLIGHT_EXIT_NONZERO = "PREFLIGHT_EXIT_NONZERO"
    PREFLIGHT_PROTOCOL_INVALID = "PREFLIGHT_PROTOCOL_INVALID"
    PREFLIGHT_PRODUCER_INVALID = "PREFLIGHT_PRODUCER_INVALID"
    PREFLIGHT_GATE_FAILED = "PREFLIGHT_GATE_FAILED"
    SOURCE_REVALIDATION_FAILED = "SOURCE_REVALIDATION_FAILED"
    CONSUMPTION_MARKER_INVALID = "CONSUMPTION_MARKER_INVALID"
    COLD_LAUNCH_FAILED = "COLD_LAUNCH_FAILED"
    COLD_TIMEOUT = "COLD_TIMEOUT"
    COLD_MONITOR_FAILED = "COLD_MONITOR_FAILED"
    COLD_EXIT_NONZERO = "COLD_EXIT_NONZERO"
    COLD_PROTOCOL_INVALID = "COLD_PROTOCOL_INVALID"
    COLD_PRODUCER_INVALID = "COLD_PRODUCER_INVALID"
    PENDING_RESULT_ABSENT = "PENDING_RESULT_ABSENT"
    TIMING_INVALID = "TIMING_INVALID"
    SAFEGUARD_TELEMETRY_INVALID = "SAFEGUARD_TELEMETRY_INVALID"
    NUMERICAL_IDENTITY_MISMATCH = "NUMERICAL_IDENTITY_MISMATCH"
    PENDING_RESULT_INVALID = "PENDING_RESULT_INVALID"
    QUARANTINE_FAILED = "QUARANTINE_FAILED"
    COMMIT_COLLISION = "COMMIT_COLLISION"
    COMMIT_RENAME_FAILED = "COMMIT_RENAME_FAILED"
    COMMIT_FSYNC_FAILED = "COMMIT_FSYNC_FAILED"
    COMMITTED_DEEP_LOAD_FAILED = "COMMITTED_DEEP_LOAD_FAILED"
    EVIDENCE_VECTOR_INVALID = "EVIDENCE_VECTOR_INVALID"
    GROUP_PREFIX_INVALID = "GROUP_PREFIX_INVALID"
    SCIENTIFIC_RECONSTRUCTION_FAILED = "SCIENTIFIC_RECONSTRUCTION_FAILED"
    RECEIPT_SCHEMA_INVALID = "RECEIPT_SCHEMA_INVALID"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    MODE_OR_LINK_INVALID = "MODE_OR_LINK_INVALID"
    STAGING_DEEP_LOAD_FAILED = "STAGING_DEEP_LOAD_FAILED"
    FINAL_COLLISION = "FINAL_COLLISION"
    FINAL_RENAME_FAILED = "FINAL_RENAME_FAILED"
    INCOMPLETE = "INCOMPLETE"
    NO_HIT = "NO_HIT"
    QUALITY_HIT = "QUALITY_HIT"


DIAG4_STAGE_REASON_ORDER: Final = {
    FailureStageV4.AUTHORITY: (
        FailureReasonCodeV4.AUTHORITY_INVALID,
        FailureReasonCodeV4.OUTPUT_ROOT_NOT_ABSENT,
        FailureReasonCodeV4.LOCK_CLAIM_FAILED,
        FailureReasonCodeV4.IDENTITY_REVALIDATION_FAILED,
        FailureReasonCodeV4.AUTHORITY_ALREADY_CONSUMED,
    ),
    FailureStageV4.SETUP: (
        FailureReasonCodeV4.SOURCE_PUBLICATION_FAILED,
        FailureReasonCodeV4.FROZEN_NUMERICAL_SUBSET_INVALID,
        FailureReasonCodeV4.NATIVE_REFERENCE_INVALID,
        FailureReasonCodeV4.POLICY_AUTHORITY_INVALID,
        FailureReasonCodeV4.SETUP_DEEP_LOAD_FAILED,
    ),
    FailureStageV4.BEFORE_PREFLIGHT: (
        FailureReasonCodeV4.SUPERVISOR_GPU_OBSERVATION_INVALID,
        FailureReasonCodeV4.SUPERVISOR_GPU_NONZERO,
        FailureReasonCodeV4.AUTHORITY_CONSUMPTION_FAILED,
        FailureReasonCodeV4.AUTHORITY_CONSUMPTION_UNCERTAIN,
    ),
    FailureStageV4.PREFLIGHT: (
        FailureReasonCodeV4.PREFLIGHT_LAUNCH_FAILED,
        FailureReasonCodeV4.PREFLIGHT_TIMEOUT,
        FailureReasonCodeV4.PREFLIGHT_MONITOR_FAILED,
        FailureReasonCodeV4.PREFLIGHT_EXIT_NONZERO,
        FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID,
        FailureReasonCodeV4.PREFLIGHT_PRODUCER_INVALID,
        FailureReasonCodeV4.PREFLIGHT_GATE_FAILED,
    ),
    FailureStageV4.BEFORE_COLD: (
        FailureReasonCodeV4.SUPERVISOR_GPU_OBSERVATION_INVALID,
        FailureReasonCodeV4.SUPERVISOR_GPU_NONZERO,
        FailureReasonCodeV4.SOURCE_REVALIDATION_FAILED,
        FailureReasonCodeV4.IDENTITY_REVALIDATION_FAILED,
        FailureReasonCodeV4.CONSUMPTION_MARKER_INVALID,
    ),
    FailureStageV4.COLD: (
        FailureReasonCodeV4.COLD_LAUNCH_FAILED,
        FailureReasonCodeV4.COLD_TIMEOUT,
        FailureReasonCodeV4.COLD_MONITOR_FAILED,
        FailureReasonCodeV4.COLD_EXIT_NONZERO,
        FailureReasonCodeV4.COLD_PROTOCOL_INVALID,
        FailureReasonCodeV4.COLD_PRODUCER_INVALID,
    ),
    FailureStageV4.NUMERICAL_COMMIT: (
        FailureReasonCodeV4.PENDING_RESULT_ABSENT,
        FailureReasonCodeV4.TIMING_INVALID,
        FailureReasonCodeV4.SAFEGUARD_TELEMETRY_INVALID,
        FailureReasonCodeV4.NUMERICAL_IDENTITY_MISMATCH,
        FailureReasonCodeV4.QUARANTINE_FAILED,
        FailureReasonCodeV4.PENDING_RESULT_INVALID,
        FailureReasonCodeV4.COMMIT_COLLISION,
        FailureReasonCodeV4.COMMIT_RENAME_FAILED,
        FailureReasonCodeV4.COMMIT_FSYNC_FAILED,
        FailureReasonCodeV4.COMMITTED_DEEP_LOAD_FAILED,
    ),
    FailureStageV4.RECEIPT: (
        FailureReasonCodeV4.EVIDENCE_VECTOR_INVALID,
        FailureReasonCodeV4.GROUP_PREFIX_INVALID,
        FailureReasonCodeV4.SCIENTIFIC_RECONSTRUCTION_FAILED,
        FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
    ),
    FailureStageV4.PUBLICATION: (
        FailureReasonCodeV4.MANIFEST_INVALID,
        FailureReasonCodeV4.MODE_OR_LINK_INVALID,
        FailureReasonCodeV4.STAGING_DEEP_LOAD_FAILED,
        FailureReasonCodeV4.FINAL_COLLISION,
        FailureReasonCodeV4.FINAL_RENAME_FAILED,
    ),
    FailureStageV4.SCIENTIFIC: (
        FailureReasonCodeV4.INCOMPLETE,
        FailureReasonCodeV4.NO_HIT,
        FailureReasonCodeV4.QUALITY_HIT,
    ),
}
DIAG4_FAILURE_STAGE_ORDER: Final = tuple(FailureStageV4)


DIAG2_FAILURE_STAGE_ORDER: Final = (
    FailureStageV2.SOURCE_PUBLICATION_FAILURE,
    FailureStageV2.NATIVE_REFERENCE_FAILURE,
    FailureStageV2.POLICY_AUTHORITY_FAILURE,
    FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE,
    FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
    FailureStageV2.PREFLIGHT_SUPERVISOR_FAILURE,
    FailureStageV2.PREFLIGHT_TIMEOUT,
    FailureStageV2.PREFLIGHT_MONITOR_FAILURE,
    FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE,
    FailureStageV2.PREFLIGHT_COMPILE_FAILURE,
    FailureStageV2.PREFLIGHT_CRASH,
    FailureStageV2.PREFLIGHT_RESOURCE_FAILURE,
    FailureStageV2.GPU_ZERO_BEFORE_COLD_FAILURE,
    FailureStageV2.COLD_SOURCE_FAILURE,
    FailureStageV2.COLD_SUPERVISOR_FAILURE,
    FailureStageV2.COLD_TIMEOUT,
    FailureStageV2.COLD_MONITOR_FAILURE,
    FailureStageV2.COLD_PROTOCOL_FAILURE,
    FailureStageV2.COLD_COMPILE_FAILURE,
    FailureStageV2.COLD_CRASH,
    FailureStageV2.COLD_RESOURCE_FAILURE,
    FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE,
)


DIAG2_STAGE_REASON_CODES: Final = {
    FailureStageV2.SOURCE_PUBLICATION_FAILURE: frozenset(
        {
            FailureReasonCodeV2.SOURCE_PRE,
            FailureReasonCodeV2.SOURCE_POST,
            FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
        }
    ),
    FailureStageV2.NATIVE_REFERENCE_FAILURE: frozenset(
        {FailureReasonCodeV2.REFERENCE_INVALID}
    ),
    FailureStageV2.POLICY_AUTHORITY_FAILURE: frozenset(
        {FailureReasonCodeV2.POLICY_DERIVATION_INVALID}
    ),
    FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE: frozenset(
        {
            FailureReasonCodeV2.GPU_QUERY_FAILED,
            FailureReasonCodeV2.GPU_PARENT_PID_PRESENT,
        }
    ),
    FailureStageV2.PREFLIGHT_SOURCE_FAILURE: frozenset(
        {
            FailureReasonCodeV2.SOURCE_POST,
            FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
            FailureReasonCodeV2.REFERENCE_INVALID,
            FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
        }
    ),
    FailureStageV2.PREFLIGHT_SUPERVISOR_FAILURE: frozenset(
        {FailureReasonCodeV2.CHILD_LAUNCH_FAILED}
    ),
    FailureStageV2.PREFLIGHT_TIMEOUT: frozenset({FailureReasonCodeV2.CHILD_TIMEOUT}),
    FailureStageV2.PREFLIGHT_MONITOR_FAILURE: frozenset(
        {
            FailureReasonCodeV2.MONITOR_BINDING_FAILED,
            FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
        }
    ),
    FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE: frozenset(
        {
            FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
            FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
            FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
            FailureReasonCodeV2.POLICY_SCHEMA_INVALID,
        }
    ),
    FailureStageV2.PREFLIGHT_COMPILE_FAILURE: frozenset(
        {
            FailureReasonCodeV2.CHILD_COMPILE_FAILED,
            FailureReasonCodeV2.CHILD_COMPILE_OOM,
        }
    ),
    FailureStageV2.PREFLIGHT_CRASH: frozenset({FailureReasonCodeV2.CHILD_EXIT_NONZERO}),
    FailureStageV2.PREFLIGHT_RESOURCE_FAILURE: frozenset(
        {FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED}
    ),
    FailureStageV2.GPU_ZERO_BEFORE_COLD_FAILURE: frozenset(
        {
            FailureReasonCodeV2.GPU_QUERY_FAILED,
            FailureReasonCodeV2.GPU_PARENT_PID_PRESENT,
        }
    ),
    FailureStageV2.COLD_SOURCE_FAILURE: frozenset(
        {
            FailureReasonCodeV2.SOURCE_POST,
            FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
            FailureReasonCodeV2.REFERENCE_INVALID,
            FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
        }
    ),
    FailureStageV2.COLD_SUPERVISOR_FAILURE: frozenset(
        {FailureReasonCodeV2.CHILD_LAUNCH_FAILED}
    ),
    FailureStageV2.COLD_TIMEOUT: frozenset({FailureReasonCodeV2.CHILD_TIMEOUT}),
    FailureStageV2.COLD_MONITOR_FAILURE: frozenset(
        {
            FailureReasonCodeV2.MONITOR_BINDING_FAILED,
            FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
        }
    ),
    FailureStageV2.COLD_PROTOCOL_FAILURE: frozenset(
        {
            FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
            FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
            FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
            FailureReasonCodeV2.POLICY_SCHEMA_INVALID,
        }
    ),
    FailureStageV2.COLD_COMPILE_FAILURE: frozenset(
        {
            FailureReasonCodeV2.CHILD_COMPILE_FAILED,
            FailureReasonCodeV2.CHILD_COMPILE_OOM,
        }
    ),
    FailureStageV2.COLD_CRASH: frozenset({FailureReasonCodeV2.CHILD_EXIT_NONZERO}),
    FailureStageV2.COLD_RESOURCE_FAILURE: frozenset(
        {FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED}
    ),
    FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE: frozenset(
        {
            FailureReasonCodeV2.NUMERICAL_SCHEMA_INVALID,
            FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED,
            FailureReasonCodeV2.SEMANTIC_VALIDATION_FAILED,
        }
    ),
}
_DIAG2_POSTLAUNCH_SETUP_SLOT: Final = {
    FailureReasonCodeV2.SOURCE_POST: "source_manifest",
    FailureReasonCodeV2.FROZEN_SUBSET_INVALID: "frozen_numerical_subset",
    FailureReasonCodeV2.REFERENCE_INVALID: "native_reference",
    FailureReasonCodeV2.POLICY_DERIVATION_INVALID: "policy_authority",
}
_DIAG2_POSTLAUNCH_SETUP_REASON: Final = {
    slot: reason for reason, slot in _DIAG2_POSTLAUNCH_SETUP_SLOT.items()
}
_DIAG2_INITIAL_SETUP_STAGE: Final = {
    FailureStageV2.SOURCE_PUBLICATION_FAILURE,
    FailureStageV2.NATIVE_REFERENCE_FAILURE,
    FailureStageV2.POLICY_AUTHORITY_FAILURE,
}
_DIAG2_INITIAL_SETUP_SLOT: Final = {
    FailureReasonCodeV2.SOURCE_PRE: "source_manifest",
    **_DIAG2_POSTLAUNCH_SETUP_SLOT,
}


@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    """One exact physical authority or a typed assertion that its path is absent."""

    state: EvidenceState
    artifact: ArtifactRef | None
    reason: AbsenceReason | None

    @classmethod
    def present(cls, artifact: ArtifactRef) -> EvidenceSlot:
        return cls(EvidenceState.PRESENT, artifact, None)

    @classmethod
    def absent(cls, reason: AbsenceReason) -> EvidenceSlot:
        return cls(EvidenceState.ABSENT, None, reason)


@dataclass(frozen=True, slots=True)
class EvidenceSlotV4:
    """One v4 artifact or one exact terminal reason; None means NOT_REACHED."""

    state: EvidenceState
    artifact: ArtifactRef | None
    reason: FailureReasonCodeV4 | None

    @classmethod
    def present(cls, artifact: ArtifactRef) -> EvidenceSlotV4:
        return cls(EvidenceState.PRESENT, artifact, None)

    @classmethod
    def absent(cls, reason: FailureReasonCodeV4 | None = None) -> EvidenceSlotV4:
        return cls(EvidenceState.ABSENT, None, reason)


@dataclass(frozen=True, slots=True)
class StructuredFailureV2:
    stage: FailureStageV2
    reason: FailureReasonCodeV2
    detail_sha256: str


@dataclass(frozen=True, slots=True)
class StructuredFailureV4:
    stage: FailureStageV4
    reason: FailureReasonCodeV4
    detail_sha256: str


@dataclass(frozen=True, slots=True)
class Diag4ProfilerCallAudit:
    """Route-owned proof that the trace-free route invoked no profiler surface."""

    profiler_enabled: bool
    profiler_start_calls: int
    profiler_stop_calls: int
    trace_normalization_calls: int


DIAG4_PROFILER_CALL_AUDIT: Final = Diag4ProfilerCallAudit(False, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class Diag5ProfilerCallAudit:
    """DIAG5-owned proof that no profiler surface was invoked."""

    profiler_enabled: bool
    profiler_start_calls: int
    profiler_stop_calls: int
    trace_normalization_calls: int


DIAG5_PROFILER_CALL_AUDIT: Final = Diag5ProfilerCallAudit(False, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class DiagnosticReceiptV2:
    evidence_slots: tuple[tuple[str, EvidenceSlot], ...]
    verdict: str
    historical_relation: str
    quality: dict[str, JsonValue] | None
    phase_attribution: dict[str, JsonValue] | None
    next_route: str
    failure: StructuredFailureV2 | None


@dataclass(frozen=True, slots=True)
class _SolveTimingEvidence:
    child_pid: int
    child_start_time_ticks: int
    backend: str
    gpu_uuid: str
    numerical_route: str
    numerical_result_schema_version: str
    problem_sha256: str
    optimizer_options_sha256: str
    base_neq_gntr1_policy_sha256: str
    scaling_sha256: str
    bootstrap_state_sha256: str
    initial_physical_state_sha256: str
    identity_sha256: str
    source_manifest_sha256: str
    process_started_monotonic_ns: int
    state_ready_monotonic_ns: int
    solve_started_monotonic_ns: int
    solve_stopped_monotonic_ns: int
    finalizer_completed_monotonic_ns: int
    endpoint_audit_completed_monotonic_ns: int
    serialization_started_monotonic_ns: int
    synchronized_solve_seconds: float
    hot_h2d_transfers: int
    hot_d2h_transfers: int
    python_callbacks: int
    final_d2h_transfers: int
    profiler_start_calls: int
    profiler_stop_calls: int
    trace_normalization_calls: int


@dataclass(frozen=True, slots=True)
class _SafeguardTelemetry:
    history_evidence: ArtifactRef
    numerical_route: str
    numerical_result_schema_version: str
    problem_sha256: str
    optimizer_options_sha256: str
    base_neq_gntr1_policy_sha256: str
    scaling_sha256: str
    bootstrap_state_sha256: str
    initial_physical_state_sha256: str
    identity_sha256: str
    loop_attempts: int
    accepted_steps: int
    retryable_rejections: int
    terminal_status: LoopStatus
    quality_latch: bool
    nonlinear_corrections: tuple[int, ...]
    nonlinear_corrections_sha256: str
    maximum_individual_correction_step_ratio: tuple[float | None, ...]
    maximum_individual_correction_step_ratio_sha256: str
    correction_path_step_ratio: tuple[float | None, ...]
    correction_path_step_ratio_sha256: str
    steihaug_solve_calls: tuple[int, ...]
    steihaug_solve_calls_sha256: str
    history_outcomes_sha256: str
    subtrial_count: tuple[int, ...]
    subtrial_count_sha256: str
    selected_subtrial_index: tuple[int, ...]
    selected_subtrial_index_sha256: str
    subtrial_trust_radius: tuple[tuple[float | None, ...], ...]
    subtrial_trust_radius_sha256: str
    subtrial_outcome: tuple[tuple[AttemptOutcome, ...], ...]
    subtrial_outcome_sha256: str
    subtrial_actual_reduction: tuple[tuple[float | None, ...], ...]
    subtrial_actual_reduction_sha256: str
    subtrial_predicted_reduction: tuple[tuple[float | None, ...], ...]
    subtrial_predicted_reduction_sha256: str
    subtrial_maximum_individual_correction_step_ratio: tuple[
        tuple[float | None, ...], ...
    ]
    subtrial_maximum_individual_correction_step_ratio_sha256: str
    subtrial_correction_path_step_ratio: tuple[tuple[float | None, ...], ...]
    subtrial_correction_path_step_ratio_sha256: str
    subtrial_corrected_radius_ratio: tuple[tuple[float | None, ...], ...]
    subtrial_corrected_radius_ratio_sha256: str
    subtrial_steihaug_iterations: tuple[tuple[int, ...], ...]
    subtrial_steihaug_iterations_sha256: str
    subtrial_steihaug_hvp_evaluations: tuple[tuple[int, ...], ...]
    subtrial_steihaug_hvp_evaluations_sha256: str
    subtrial_steihaug_solve_calls: tuple[tuple[int, ...], ...]
    subtrial_steihaug_solve_calls_sha256: str
    subtrial_total_hvp_evaluations: tuple[tuple[int, ...], ...]
    subtrial_total_hvp_evaluations_sha256: str
    subtrial_nonlinear_corrections: tuple[tuple[int, ...], ...]
    subtrial_nonlinear_corrections_sha256: str
    subtrial_joint_evaluations: tuple[tuple[int, ...], ...]
    subtrial_joint_evaluations_sha256: str
    subtrial_joint_linearizations: tuple[tuple[int, ...], ...]
    subtrial_joint_linearizations_sha256: str
    subtrial_joint_value_evaluations: tuple[tuple[int, ...], ...]
    subtrial_joint_value_evaluations_sha256: str
    subtrial_objective_residual_linearizations: tuple[tuple[int, ...], ...]
    subtrial_objective_residual_linearizations_sha256: str
    subtrial_gram_factorizations: tuple[tuple[int, ...], ...]
    subtrial_gram_factorizations_sha256: str
    subtrial_gram_solves: tuple[tuple[int, ...], ...]
    subtrial_gram_solves_sha256: str
    subtrial_summary: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class SolveTimingEvidenceV4(_SolveTimingEvidence):
    """Public DIAG4 timing semantics; never returned by a successor parser."""


@dataclass(frozen=True, slots=True)
class SolveTimingEvidenceV5(_SolveTimingEvidence):
    """Public DIAG5 timing semantics with an identity-distinct Python type."""


@dataclass(frozen=True, slots=True)
class SafeguardTelemetryV4(_SafeguardTelemetry):
    """Public DIAG4 safeguard semantics; never returned by a successor parser."""


@dataclass(frozen=True, slots=True)
class SafeguardTelemetryV5(_SafeguardTelemetry):
    """Public DIAG5 safeguard semantics with an identity-distinct Python type."""


@dataclass(frozen=True, slots=True)
class DiagnosticReceiptV4:
    evidence_slots: tuple[tuple[str, EvidenceSlotV4], ...]
    verdict: str
    historical_relation: str
    quality: dict[str, JsonValue] | None
    phase_attribution: dict[str, JsonValue]
    next_route: str
    speed_comparison: JsonValue
    failure: StructuredFailureV4


class FailureStageV5(StrEnum):
    AUTHORITY = "AUTHORITY"
    SETUP = "SETUP"
    BEFORE_PREFLIGHT = "BEFORE_PREFLIGHT"
    PREFLIGHT = "PREFLIGHT"
    BEFORE_COLD = "BEFORE_COLD"
    COLD = "COLD"
    NUMERICAL_COMMIT = "NUMERICAL_COMMIT"
    RECEIPT = "RECEIPT"
    PUBLICATION = "PUBLICATION"
    SCIENTIFIC = "SCIENTIFIC"


class FailureReasonCodeV5(StrEnum):
    AUTHORITY_INVALID = "AUTHORITY_INVALID"
    OUTPUT_ROOT_NOT_ABSENT = "OUTPUT_ROOT_NOT_ABSENT"
    LOCK_CLAIM_FAILED = "LOCK_CLAIM_FAILED"
    IDENTITY_REVALIDATION_FAILED = "IDENTITY_REVALIDATION_FAILED"
    AUTHORITY_ALREADY_CONSUMED = "AUTHORITY_ALREADY_CONSUMED"
    SOURCE_PUBLICATION_FAILED = "SOURCE_PUBLICATION_FAILED"
    FROZEN_NUMERICAL_SUBSET_INVALID = "FROZEN_NUMERICAL_SUBSET_INVALID"
    NATIVE_REFERENCE_INVALID = "NATIVE_REFERENCE_INVALID"
    POLICY_AUTHORITY_INVALID = "POLICY_AUTHORITY_INVALID"
    SETUP_DEEP_LOAD_FAILED = "SETUP_DEEP_LOAD_FAILED"
    SUPERVISOR_GPU_OBSERVATION_INVALID = "SUPERVISOR_GPU_OBSERVATION_INVALID"
    SUPERVISOR_GPU_NONZERO = "SUPERVISOR_GPU_NONZERO"
    AUTHORITY_CONSUMPTION_FAILED = "AUTHORITY_CONSUMPTION_FAILED"
    AUTHORITY_CONSUMPTION_UNCERTAIN = "AUTHORITY_CONSUMPTION_UNCERTAIN"
    PREFLIGHT_LAUNCH_FAILED = "PREFLIGHT_LAUNCH_FAILED"
    PREFLIGHT_TIMEOUT = "PREFLIGHT_TIMEOUT"
    PREFLIGHT_MONITOR_FAILED = "PREFLIGHT_MONITOR_FAILED"
    PREFLIGHT_EXIT_NONZERO = "PREFLIGHT_EXIT_NONZERO"
    PREFLIGHT_PROTOCOL_INVALID = "PREFLIGHT_PROTOCOL_INVALID"
    PREFLIGHT_PRODUCER_INVALID = "PREFLIGHT_PRODUCER_INVALID"
    PREFLIGHT_GATE_FAILED = "PREFLIGHT_GATE_FAILED"
    SOURCE_REVALIDATION_FAILED = "SOURCE_REVALIDATION_FAILED"
    CONSUMPTION_MARKER_INVALID = "CONSUMPTION_MARKER_INVALID"
    COLD_LAUNCH_FAILED = "COLD_LAUNCH_FAILED"
    COLD_TIMEOUT = "COLD_TIMEOUT"
    COLD_MONITOR_FAILED = "COLD_MONITOR_FAILED"
    COLD_EXIT_NONZERO = "COLD_EXIT_NONZERO"
    COLD_PROTOCOL_INVALID = "COLD_PROTOCOL_INVALID"
    COLD_PRODUCER_INVALID = "COLD_PRODUCER_INVALID"
    PENDING_RESULT_ABSENT = "PENDING_RESULT_ABSENT"
    TIMING_INVALID = "TIMING_INVALID"
    SAFEGUARD_TELEMETRY_INVALID = "SAFEGUARD_TELEMETRY_INVALID"
    NUMERICAL_IDENTITY_MISMATCH = "NUMERICAL_IDENTITY_MISMATCH"
    QUARANTINE_FAILED = "QUARANTINE_FAILED"
    PENDING_RESULT_INVALID = "PENDING_RESULT_INVALID"
    COMMIT_COLLISION = "COMMIT_COLLISION"
    COMMIT_RENAME_FAILED = "COMMIT_RENAME_FAILED"
    COMMIT_FSYNC_FAILED = "COMMIT_FSYNC_FAILED"
    COMMITTED_DEEP_LOAD_FAILED = "COMMITTED_DEEP_LOAD_FAILED"
    EVIDENCE_VECTOR_INVALID = "EVIDENCE_VECTOR_INVALID"
    GROUP_PREFIX_INVALID = "GROUP_PREFIX_INVALID"
    SCIENTIFIC_RECONSTRUCTION_FAILED = "SCIENTIFIC_RECONSTRUCTION_FAILED"
    RECEIPT_SCHEMA_INVALID = "RECEIPT_SCHEMA_INVALID"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    MODE_OR_LINK_INVALID = "MODE_OR_LINK_INVALID"
    STAGING_DEEP_LOAD_FAILED = "STAGING_DEEP_LOAD_FAILED"
    FINAL_COLLISION = "FINAL_COLLISION"
    FINAL_RENAME_FAILED = "FINAL_RENAME_FAILED"
    INCOMPLETE = "INCOMPLETE"
    NO_HIT = "NO_HIT"
    QUALITY_HIT = "QUALITY_HIT"


@dataclass(frozen=True, slots=True)
class EvidenceSlotV5:
    state: EvidenceState
    artifact: ArtifactRef | None
    reason: FailureReasonCodeV5 | None

    @classmethod
    def present(cls, artifact: ArtifactRef) -> EvidenceSlotV5:
        return cls(EvidenceState.PRESENT, artifact, None)

    @classmethod
    def absent(cls, reason: FailureReasonCodeV5 | None = None) -> EvidenceSlotV5:
        return cls(EvidenceState.ABSENT, None, reason)


@dataclass(frozen=True, slots=True)
class StructuredFailureV5:
    stage: FailureStageV5
    reason: FailureReasonCodeV5
    detail_sha256: str


DIAG5_STAGE_REASON_ORDER: Final = {
    FailureStageV5.AUTHORITY: (
        FailureReasonCodeV5.AUTHORITY_INVALID,
        FailureReasonCodeV5.OUTPUT_ROOT_NOT_ABSENT,
        FailureReasonCodeV5.LOCK_CLAIM_FAILED,
        FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
        FailureReasonCodeV5.AUTHORITY_ALREADY_CONSUMED,
    ),
    FailureStageV5.SETUP: (
        FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED,
        FailureReasonCodeV5.FROZEN_NUMERICAL_SUBSET_INVALID,
        FailureReasonCodeV5.NATIVE_REFERENCE_INVALID,
        FailureReasonCodeV5.POLICY_AUTHORITY_INVALID,
        FailureReasonCodeV5.SETUP_DEEP_LOAD_FAILED,
    ),
    FailureStageV5.BEFORE_PREFLIGHT: (
        FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID,
        FailureReasonCodeV5.SUPERVISOR_GPU_NONZERO,
        FailureReasonCodeV5.AUTHORITY_CONSUMPTION_FAILED,
        FailureReasonCodeV5.AUTHORITY_CONSUMPTION_UNCERTAIN,
        FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
    ),
    FailureStageV5.PREFLIGHT: (
        FailureReasonCodeV5.PREFLIGHT_LAUNCH_FAILED,
        FailureReasonCodeV5.PREFLIGHT_TIMEOUT,
        FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED,
        FailureReasonCodeV5.PREFLIGHT_EXIT_NONZERO,
        FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
        FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
        FailureReasonCodeV5.PREFLIGHT_GATE_FAILED,
    ),
    FailureStageV5.BEFORE_COLD: (
        FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID,
        FailureReasonCodeV5.SUPERVISOR_GPU_NONZERO,
        FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
        FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
        FailureReasonCodeV5.CONSUMPTION_MARKER_INVALID,
    ),
    FailureStageV5.COLD: (
        FailureReasonCodeV5.COLD_LAUNCH_FAILED,
        FailureReasonCodeV5.COLD_TIMEOUT,
        FailureReasonCodeV5.COLD_MONITOR_FAILED,
        FailureReasonCodeV5.COLD_EXIT_NONZERO,
        FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
        FailureReasonCodeV5.COLD_PRODUCER_INVALID,
    ),
    FailureStageV5.NUMERICAL_COMMIT: (
        FailureReasonCodeV5.PENDING_RESULT_ABSENT,
        FailureReasonCodeV5.TIMING_INVALID,
        FailureReasonCodeV5.SAFEGUARD_TELEMETRY_INVALID,
        FailureReasonCodeV5.NUMERICAL_IDENTITY_MISMATCH,
        FailureReasonCodeV5.QUARANTINE_FAILED,
        FailureReasonCodeV5.PENDING_RESULT_INVALID,
        FailureReasonCodeV5.COMMIT_COLLISION,
        FailureReasonCodeV5.COMMIT_RENAME_FAILED,
        FailureReasonCodeV5.COMMIT_FSYNC_FAILED,
        FailureReasonCodeV5.COMMITTED_DEEP_LOAD_FAILED,
    ),
    FailureStageV5.RECEIPT: (
        FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID,
        FailureReasonCodeV5.GROUP_PREFIX_INVALID,
        FailureReasonCodeV5.SCIENTIFIC_RECONSTRUCTION_FAILED,
        FailureReasonCodeV5.RECEIPT_SCHEMA_INVALID,
    ),
    FailureStageV5.PUBLICATION: (
        FailureReasonCodeV5.MANIFEST_INVALID,
        FailureReasonCodeV5.MODE_OR_LINK_INVALID,
        FailureReasonCodeV5.STAGING_DEEP_LOAD_FAILED,
        FailureReasonCodeV5.FINAL_COLLISION,
        FailureReasonCodeV5.FINAL_RENAME_FAILED,
    ),
    FailureStageV5.SCIENTIFIC: (
        FailureReasonCodeV5.INCOMPLETE,
        FailureReasonCodeV5.NO_HIT,
        FailureReasonCodeV5.QUALITY_HIT,
    ),
}
DIAG5_FAILURE_STAGE_ORDER: Final = tuple(FailureStageV5)
DIAG5_STAGE_REASON_PRESENT_PREFIXES: Final = MappingProxyType(
    {
        (FailureStageV5.AUTHORITY, FailureReasonCodeV5.AUTHORITY_INVALID): (0,),
        (FailureStageV5.AUTHORITY, FailureReasonCodeV5.OUTPUT_ROOT_NOT_ABSENT): (0,),
        (FailureStageV5.AUTHORITY, FailureReasonCodeV5.LOCK_CLAIM_FAILED): (0,),
        (
            FailureStageV5.AUTHORITY,
            FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
        ): (0,),
        (
            FailureStageV5.AUTHORITY,
            FailureReasonCodeV5.AUTHORITY_ALREADY_CONSUMED,
        ): (0,),
        (FailureStageV5.SETUP, FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED): (0,),
        (
            FailureStageV5.SETUP,
            FailureReasonCodeV5.FROZEN_NUMERICAL_SUBSET_INVALID,
        ): (1,),
        (FailureStageV5.SETUP, FailureReasonCodeV5.NATIVE_REFERENCE_INVALID): (2,),
        (FailureStageV5.SETUP, FailureReasonCodeV5.POLICY_AUTHORITY_INVALID): (3,),
        (FailureStageV5.SETUP, FailureReasonCodeV5.SETUP_DEEP_LOAD_FAILED): (4,),
        (
            FailureStageV5.BEFORE_PREFLIGHT,
            FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID,
        ): (4, 5),
        (
            FailureStageV5.BEFORE_PREFLIGHT,
            FailureReasonCodeV5.SUPERVISOR_GPU_NONZERO,
        ): (5,),
        (
            FailureStageV5.BEFORE_PREFLIGHT,
            FailureReasonCodeV5.AUTHORITY_CONSUMPTION_FAILED,
        ): (5,),
        (
            FailureStageV5.BEFORE_PREFLIGHT,
            FailureReasonCodeV5.AUTHORITY_CONSUMPTION_UNCERTAIN,
        ): (5,),
        (
            FailureStageV5.BEFORE_PREFLIGHT,
            FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
        ): (5,),
        (FailureStageV5.PREFLIGHT, FailureReasonCodeV5.PREFLIGHT_LAUNCH_FAILED): (5,),
        (FailureStageV5.PREFLIGHT, FailureReasonCodeV5.PREFLIGHT_TIMEOUT): (8,),
        (FailureStageV5.PREFLIGHT, FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED): (8,),
        (FailureStageV5.PREFLIGHT, FailureReasonCodeV5.PREFLIGHT_EXIT_NONZERO): (8,),
        (
            FailureStageV5.PREFLIGHT,
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
        ): (8,),
        (
            FailureStageV5.PREFLIGHT,
            FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
        ): (8,),
        (FailureStageV5.PREFLIGHT, FailureReasonCodeV5.PREFLIGHT_GATE_FAILED): (12,),
        (
            FailureStageV5.BEFORE_COLD,
            FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID,
        ): (12, 13),
        (FailureStageV5.BEFORE_COLD, FailureReasonCodeV5.SUPERVISOR_GPU_NONZERO): (13,),
        (
            FailureStageV5.BEFORE_COLD,
            FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
        ): (13,),
        (
            FailureStageV5.BEFORE_COLD,
            FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
        ): (13,),
        (
            FailureStageV5.BEFORE_COLD,
            FailureReasonCodeV5.CONSUMPTION_MARKER_INVALID,
        ): (13,),
        (FailureStageV5.COLD, FailureReasonCodeV5.COLD_LAUNCH_FAILED): (13,),
        (FailureStageV5.COLD, FailureReasonCodeV5.COLD_TIMEOUT): (16,),
        (FailureStageV5.COLD, FailureReasonCodeV5.COLD_MONITOR_FAILED): (16,),
        (FailureStageV5.COLD, FailureReasonCodeV5.COLD_EXIT_NONZERO): (16,),
        (FailureStageV5.COLD, FailureReasonCodeV5.COLD_PROTOCOL_INVALID): (16,),
        (FailureStageV5.COLD, FailureReasonCodeV5.COLD_PRODUCER_INVALID): (16,),
        (
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.PENDING_RESULT_ABSENT,
        ): (20,),
        (FailureStageV5.NUMERICAL_COMMIT, FailureReasonCodeV5.TIMING_INVALID): (20,),
        (
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.SAFEGUARD_TELEMETRY_INVALID,
        ): (20,),
        (
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.NUMERICAL_IDENTITY_MISMATCH,
        ): (20,),
        (FailureStageV5.NUMERICAL_COMMIT, FailureReasonCodeV5.QUARANTINE_FAILED): (20,),
        (
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.PENDING_RESULT_INVALID,
        ): (20,),
        (FailureStageV5.NUMERICAL_COMMIT, FailureReasonCodeV5.COMMIT_COLLISION): (20,),
        (
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.COMMIT_RENAME_FAILED,
        ): (20,),
        (
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.COMMIT_FSYNC_FAILED,
        ): (20,),
        (
            FailureStageV5.NUMERICAL_COMMIT,
            FailureReasonCodeV5.COMMITTED_DEEP_LOAD_FAILED,
        ): (20,),
        (FailureStageV5.RECEIPT, FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID): (
            24,
            25,
        ),
        (FailureStageV5.RECEIPT, FailureReasonCodeV5.GROUP_PREFIX_INVALID): (25,),
        (
            FailureStageV5.RECEIPT,
            FailureReasonCodeV5.SCIENTIFIC_RECONSTRUCTION_FAILED,
        ): (25,),
        (FailureStageV5.RECEIPT, FailureReasonCodeV5.RECEIPT_SCHEMA_INVALID): (25,),
        (FailureStageV5.PUBLICATION, FailureReasonCodeV5.MANIFEST_INVALID): (25,),
        (FailureStageV5.PUBLICATION, FailureReasonCodeV5.MODE_OR_LINK_INVALID): (25,),
        (
            FailureStageV5.PUBLICATION,
            FailureReasonCodeV5.STAGING_DEEP_LOAD_FAILED,
        ): (25,),
        (FailureStageV5.PUBLICATION, FailureReasonCodeV5.FINAL_COLLISION): (25,),
        (FailureStageV5.PUBLICATION, FailureReasonCodeV5.FINAL_RENAME_FAILED): (25,),
        (FailureStageV5.SCIENTIFIC, FailureReasonCodeV5.INCOMPLETE): (25,),
        (FailureStageV5.SCIENTIFIC, FailureReasonCodeV5.NO_HIT): (25,),
        (FailureStageV5.SCIENTIFIC, FailureReasonCodeV5.QUALITY_HIT): (25,),
    }
)
DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256: Final = (
    "b0201988e5421a54500000ee56d2a836585f49b62a7a8d689d0c7f516316222e"
)
DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256: Final = (
    "320b43d84c82b9be812cdf389da4c89f74e548748922d8356a35d51a09192fa4"
)


def _diag5_allowed_present_prefixes(
    failure: StructuredFailureV5,
) -> tuple[int, ...] | None:
    if (
        failure.stage is FailureStageV5.PREFLIGHT
        and failure.reason is FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
        and failure.detail_sha256
        == DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256
    ):
        return (8, 12)
    if (
        failure.stage is FailureStageV5.COLD
        and failure.reason is FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        and failure.detail_sha256 == DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256
    ):
        return (16,)
    if failure.detail_sha256 in {
        DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256,
        DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256,
    }:
        return None
    return DIAG5_STAGE_REASON_PRESENT_PREFIXES.get((failure.stage, failure.reason))


@dataclass(frozen=True, slots=True)
class NativeBindingV5:
    role: str
    path: str
    sha256: str
    size_bytes: int
    link_count: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class DiagnosticReceiptV5:
    evidence_slots: tuple[tuple[str, EvidenceSlotV5], ...]
    verdict: str
    historical_relation: str
    quality: dict[str, JsonValue] | None
    phase_attribution: dict[str, JsonValue]
    next_route: str
    speed_comparison: JsonValue
    failure: StructuredFailureV5
    native_bindings: tuple[tuple[str, NativeBindingV5], ...]
    predecessor_postmortem: ArtifactRef


def diag2_evidence_slot_payload(slot: EvidenceSlot) -> dict[str, JsonValue]:
    if slot.state is EvidenceState.PRESENT:
        if slot.artifact is None or slot.reason is not None:
            raise ValueError("PRESENT evidence slot has invalid union members")
        return {
            "state": slot.state.value,
            "artifact": _artifact_ref_payload(slot.artifact),
        }
    if slot.artifact is not None or slot.reason is None:
        raise ValueError("ABSENT evidence slot has invalid union members")
    return {"state": slot.state.value, "reason": slot.reason.value}


def parse_diag2_evidence_slot(value: JsonValue, *, name: str) -> EvidenceSlot:
    """Parse the discriminated union and enforce the slot's frozen path."""

    return _parse_evidence_slot(
        value,
        name=name,
        slot_paths=DIAG2_EVIDENCE_SLOT_PATHS,
        trace_path_predicate=_diag2_trace_path,
        context="DIAG2",
    )


def parse_diag3_evidence_slot(value: JsonValue, *, name: str) -> EvidenceSlot:
    """Parse one successor slot against the atomic numerical-result layout."""

    return _parse_evidence_slot(
        value,
        name=name,
        slot_paths=DIAG3_EVIDENCE_SLOT_PATHS,
        trace_path_predicate=_diag3_trace_path,
        context="DIAG3",
    )


def diag4_evidence_slot_payload(slot: EvidenceSlotV4) -> dict[str, JsonValue]:
    if slot.state is EvidenceState.PRESENT:
        if slot.artifact is None or slot.reason is not None:
            raise ValueError("DIAG4 PRESENT slot union differs")
        return {
            "state": EvidenceState.PRESENT.value,
            "artifact": _artifact_ref_payload(slot.artifact),
        }
    if slot.artifact is not None:
        raise ValueError("DIAG4 ABSENT slot retains an artifact")
    return {
        "state": EvidenceState.ABSENT.value,
        "reason": "NOT_REACHED" if slot.reason is None else slot.reason.value,
    }


def parse_diag4_evidence_slot(value: JsonValue, *, name: str) -> EvidenceSlotV4:
    """Parse one trace-free successor slot without widening legacy layouts."""

    if name not in DIAG4_EVIDENCE_SLOT_PATHS:
        raise ValueError(f"unknown DIAG4 evidence slot: {name}")
    payload = _mapping(value, f"DIAG4 evidence_slots.{name}")
    state = EvidenceState(
        _string(payload.get("state"), f"DIAG4 evidence_slots.{name}.state")
    )
    if state is EvidenceState.PRESENT:
        _exact_keys(
            payload,
            frozenset({"state", "artifact"}),
            f"DIAG4 evidence_slots.{name}",
        )
        reference = _artifact_ref(
            payload["artifact"], f"DIAG4 evidence_slots.{name}.artifact"
        )
        if reference.relative_path != DIAG4_EVIDENCE_SLOT_PATHS[name]:
            raise ValueError(f"DIAG4 {name} path differs from the frozen layout")
        return EvidenceSlotV4.present(reference)
    _exact_keys(
        payload,
        frozenset({"state", "reason"}),
        f"DIAG4 evidence_slots.{name}",
    )
    raw_reason = _string(payload["reason"], f"DIAG4 evidence_slots.{name}.reason")
    return EvidenceSlotV4.absent(
        None if raw_reason == "NOT_REACHED" else FailureReasonCodeV4(raw_reason)
    )


def diag5_evidence_slot_payload(slot: EvidenceSlotV5) -> dict[str, JsonValue]:
    """Serialize one v5 slot without changing the generation-neutral union."""

    if slot.state is EvidenceState.PRESENT:
        if slot.artifact is None or slot.reason is not None:
            raise ValueError("DIAG5 PRESENT slot union differs")
        return {
            "state": EvidenceState.PRESENT.value,
            "artifact": _artifact_ref_payload(slot.artifact),
        }
    if slot.artifact is not None:
        raise ValueError("DIAG5 ABSENT slot retains an artifact")
    return {
        "state": EvidenceState.ABSENT.value,
        "reason": "NOT_REACHED" if slot.reason is None else slot.reason.value,
    }


def parse_diag5_evidence_slot(value: JsonValue, *, name: str) -> EvidenceSlotV5:
    """Parse one v5 slot and enforce its exact path and mixed child schema."""

    if name not in DIAG5_EVIDENCE_SLOT_PATHS:
        raise ValueError(f"unknown DIAG5 evidence slot: {name}")
    payload = _mapping(value, f"DIAG5 evidence_slots.{name}")
    state = EvidenceState(
        _string(payload.get("state"), f"DIAG5 evidence_slots.{name}.state")
    )
    if state is EvidenceState.PRESENT:
        _exact_keys(
            payload,
            frozenset({"state", "artifact"}),
            f"DIAG5 evidence_slots.{name}",
        )
        reference = _artifact_ref(
            payload["artifact"], f"DIAG5 evidence_slots.{name}.artifact"
        )
        if reference.relative_path != DIAG5_EVIDENCE_SLOT_PATHS[name]:
            raise ValueError(f"DIAG5 {name} path differs from the frozen layout")
        if reference.schema_version != DIAG5_EVIDENCE_SLOT_SCHEMAS[name]:
            raise ValueError(f"DIAG5 {name} schema differs from the frozen map")
        return EvidenceSlotV5.present(reference)
    _exact_keys(
        payload,
        frozenset({"state", "reason"}),
        f"DIAG5 evidence_slots.{name}",
    )
    raw_reason = _string(payload["reason"], f"DIAG5 evidence_slots.{name}.reason")
    return EvidenceSlotV5.absent(
        None if raw_reason == "NOT_REACHED" else FailureReasonCodeV5(raw_reason)
    )


def _parse_evidence_slot(
    value: JsonValue,
    *,
    name: str,
    slot_paths: Mapping[str, str],
    trace_path_predicate: Callable[[str], bool],
    context: str,
) -> EvidenceSlot:
    """Parse a slot using one explicitly selected immutable layout."""

    if name not in slot_paths:
        raise ValueError(f"unknown {context} evidence slot: {name}")
    payload = _mapping(value, f"evidence_slots.{name}")
    state = EvidenceState(_string(payload.get("state"), f"evidence_slots.{name}.state"))
    if state is EvidenceState.PRESENT:
        _exact_keys(payload, frozenset({"state", "artifact"}), f"evidence_slots.{name}")
        reference = _artifact_ref(
            payload["artifact"], f"evidence_slots.{name}.artifact"
        )
        frozen = slot_paths[name]
        if name != "cold_raw_trace" and reference.relative_path != frozen:
            raise ValueError(f"{name} path differs from the frozen layout")
        if name == "cold_raw_trace" and not trace_path_predicate(
            reference.relative_path
        ):
            raise ValueError("cold_raw_trace path differs from the frozen layout")
        return EvidenceSlot.present(reference)
    _exact_keys(payload, frozenset({"state", "reason"}), f"evidence_slots.{name}")
    return EvidenceSlot.absent(
        AbsenceReason(_string(payload["reason"], f"evidence_slots.{name}.reason"))
    )


def _diag2_trace_path(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        path.as_posix() == relative_path
        and not path.is_absolute()
        and ".." not in path.parts
        and path.name.endswith(".trace.json.gz")
        and path.parent.parent == Path("cold/raw-trace/plugins/profile")
        and bool(path.name.removesuffix(".trace.json.gz"))
    )


def _diag3_trace_path(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        path.as_posix() == relative_path
        and not path.is_absolute()
        and ".." not in path.parts
        and path.name.endswith(".trace.json.gz")
        and path.parent.parent
        == Path(DIAG3_COMMITTED_NUMERICAL_DIRECTORY) / "raw-trace/plugins/profile"
        and bool(path.name.removesuffix(".trace.json.gz"))
    )


def _present_artifact(slot: EvidenceSlot, name: str) -> ArtifactRef:
    if slot.state is not EvidenceState.PRESENT or slot.artifact is None:
        raise ValueError(f"PRESENT evidence slot omits artifact: {name}")
    return slot.artifact


def _present_reference(reference: ArtifactRef | None, name: str) -> ArtifactRef:
    if reference is None:
        raise ValueError(f"required artifact reference is absent: {name}")
    return reference


def _diag2_setup_path_is_minimum_typed(root: Path, name: str) -> bool:
    path = root / DIAG2_EVIDENCE_SLOT_PATHS[name]
    if not path.is_file() or path.is_symlink():
        return False
    expected_schema = {
        "source_manifest": SOURCE_MANIFEST_SCHEMA_VERSION,
        "frozen_numerical_subset": DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
        "native_reference": NATIVE_REFERENCE_SCHEMA_VERSION,
        "policy_authority": DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION,
    }[name]
    try:
        document = _mapping(
            load_canonical_json_bytes(path.read_bytes()), f"minimum-typed {name}"
        )
    except (OSError, TypeError, ValueError):
        return False
    return document.get("schema_version") == expected_schema


def _validate_diag2_slots(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlot],
    *,
    failure: StructuredFailureV2 | None = None,
    _slot_paths: Mapping[str, str] = DIAG2_EVIDENCE_SLOT_PATHS,
    _trace_path_predicate: Callable[[str], bool] = _diag2_trace_path,
) -> None:
    if frozenset(evidence_slots) != DIAG2_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG2 evidence slots differ from the frozen schema")
    root = artifact_root.resolve(strict=True)
    for name, slot in evidence_slots.items():
        diag2_evidence_slot_payload(slot)
        canonical = _slot_paths[name]
        if slot.state is EvidenceState.ABSENT:
            path_exists = (
                any(path.is_file() for path in (root / canonical).rglob("*"))
                if name == "cold_raw_trace" and (root / canonical).is_dir()
                else (root / canonical).exists()
            )
            retained_untyped = slot.reason in _DIAG2_UNTYPED_REASONS.get(
                name, frozenset()
            )
            retained_opaque = (
                (
                    name == "source_manifest"
                    and slot.reason
                    in {AbsenceReason.SOURCE_PRE, AbsenceReason.SOURCE_POST}
                )
                or (
                    name == "native_reference"
                    and slot.reason is AbsenceReason.REFERENCE_INVALID
                )
                or (
                    name == "frozen_numerical_subset"
                    and slot.reason is AbsenceReason.FROZEN_SUBSET_INVALID
                )
                or (
                    name == "policy_authority"
                    and slot.reason is AbsenceReason.POLICY_DERIVATION_INVALID
                )
                or name == "cold_raw_trace"
            )
            if path_exists and not retained_untyped and not retained_opaque:
                raise ValueError(f"ABSENT evidence path exists: {canonical}")
            continue
        if slot.artifact is None:
            raise ValueError(f"PRESENT evidence slot omits artifact: {name}")
        if name != "cold_raw_trace" and slot.artifact.relative_path != canonical:
            raise ValueError(f"{name} path differs from the frozen layout")
        if name == "cold_raw_trace" and not _trace_path_predicate(
            slot.artifact.relative_path
        ):
            raise ValueError("cold_raw_trace path differs from the frozen layout")
        _resolve_artifact(root, slot.artifact)
    source_slot = evidence_slots["source_manifest"]
    subset_slot = evidence_slots["frozen_numerical_subset"]
    postlaunch_setup_failure = (
        failure
        if failure is not None
        and failure.stage
        in {
            FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            FailureStageV2.COLD_SOURCE_FAILURE,
        }
        and failure.reason in _DIAG2_POSTLAUNCH_SETUP_SLOT
        else None
    )
    initial_setup_failure = (
        failure
        if failure is not None and failure.stage in _DIAG2_INITIAL_SETUP_STAGE
        else None
    )
    setup_integrity_failure = postlaunch_setup_failure or initial_setup_failure
    if setup_integrity_failure is not None:
        if setup_integrity_failure.reason is FailureReasonCodeV2.SOURCE_PRE:
            source = evidence_slots["source_manifest"]
            if (
                source.state is not EvidenceState.ABSENT
                or source.reason is not AbsenceReason.SOURCE_PRE
            ):
                raise ValueError("source-pre setup vector differs")
            if _diag2_setup_path_is_minimum_typed(root, "source_manifest"):
                raise ValueError("source-pre authority passes minimum typing")
            for name in tuple(_DIAG2_POSTLAUNCH_SETUP_REASON)[1:]:
                slot = evidence_slots[name]
                if (
                    slot.state is not EvidenceState.ABSENT
                    or slot.reason is not AbsenceReason.NOT_REACHED
                ):
                    raise ValueError("source-pre later setup vector differs")
            _validate_diag2_supervisor_sequence(
                root, evidence_slots, failure=setup_integrity_failure
            )
            return
        validators = (
            (
                "source_manifest",
                FailureReasonCodeV2.SOURCE_POST,
                lambda: validate_diag2_source_snapshot_authority(root),
            ),
            (
                "frozen_numerical_subset",
                FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
                lambda: validate_diag2_frozen_numerical_subset_payload(
                    _load_ref_json(
                        root,
                        _present_artifact(
                            evidence_slots["frozen_numerical_subset"],
                            "frozen_numerical_subset",
                        ),
                        "frozen numerical subset",
                    ),
                    artifact_root=root,
                ),
            ),
            (
                "native_reference",
                FailureReasonCodeV2.REFERENCE_INVALID,
                lambda: validate_native_equivalent_reference(root / "native-reference"),
            ),
            (
                "policy_authority",
                FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
                lambda: validate_diag2_policy_authority_payload(
                    _load_ref_json(
                        root,
                        _present_artifact(
                            evidence_slots["policy_authority"],
                            "policy_authority",
                        ),
                        "policy authority",
                    ),
                    artifact_root=root,
                ),
            ),
        )
        derived_reason: FailureReasonCodeV2 | None = None
        failed_index: int | None = None
        for index, (name, reason, validator) in enumerate(validators):
            slot = evidence_slots[name]
            if slot.state is EvidenceState.ABSENT:
                if slot.reason is not AbsenceReason(reason.value):
                    raise ValueError("post-launch setup absence reason differs")
                if _diag2_setup_path_is_minimum_typed(root, name):
                    raise ValueError("ABSENT setup authority passes minimum typing")
                derived_reason = reason
                failed_index = index
                break
            try:
                validator()
            except (OSError, TypeError, ValueError):
                derived_reason = reason
                failed_index = index
                break
        if derived_reason is not setup_integrity_failure.reason:
            raise ValueError("setup first failure differs")
        if failed_index is None:
            raise AssertionError("post-launch setup failure index narrowing failed")
        for name, reason, _ in validators[failed_index + 1 :]:
            slot = evidence_slots[name]
            if slot.state is EvidenceState.PRESENT:
                continue
            expected_absence = (
                AbsenceReason(reason.value)
                if postlaunch_setup_failure is not None
                else AbsenceReason.NOT_REACHED
            )
            if slot.reason is not expected_absence:
                raise ValueError("later setup absence reason differs")
            if _diag2_setup_path_is_minimum_typed(root, name):
                raise ValueError("later ABSENT setup authority passes minimum typing")
        _validate_diag2_supervisor_sequence(
            root, evidence_slots, failure=setup_integrity_failure
        )
        return
    snapshot: SnapshotPublication | None = None
    if source_slot.state is EvidenceState.PRESENT and source_slot.artifact is not None:
        snapshot = validate_diag2_source_snapshot_authority(root)
        if source_slot.artifact.sha256 != snapshot.manifest_sha256:
            raise ValueError("source manifest reference differs from loaded snapshot")
    if subset_slot.state is EvidenceState.PRESENT and subset_slot.artifact is not None:
        if snapshot is None:
            if source_slot.reason is not AbsenceReason.SOURCE_POST:
                raise ValueError(
                    "frozen numerical subset cannot precede source authority"
                )
            if (
                _load_ref_json(root, subset_slot.artifact, "frozen numerical subset")
                != build_diag2_frozen_numerical_subset_payload()
            ):
                raise ValueError("preserved frozen subset payload differs from SSOT")
        else:
            validate_diag2_frozen_numerical_subset_payload(
                _load_ref_json(root, subset_slot.artifact, "frozen numerical subset"),
                artifact_root=root,
            )
    _validate_diag2_supervisor_sequence(root, evidence_slots, failure=failure)


def _validate_diag3_slots(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlot],
    *,
    failure: StructuredFailureV2 | None = None,
) -> None:
    _validate_diag2_slots(
        artifact_root,
        evidence_slots,
        failure=failure,
        _slot_paths=DIAG3_EVIDENCE_SLOT_PATHS,
        _trace_path_predicate=_diag3_trace_path,
    )
    producer_slot = evidence_slots["cold_producer"]
    if producer_slot.artifact is None:
        return
    producer = validate_diag3_producer_payload(
        _load_ref_json(artifact_root, producer_slot.artifact, "DIAG3 cold producer"),
        mode="cold",
    )
    if producer["schema_version"] != DIAG3_COLD_RESULT_SCHEMA_VERSION:
        return
    for field, slot_name in (
        ("runtime_evidence", "cold_runtime"),
        ("policy_evidence", "cold_policy"),
        ("history_evidence", "cold_history"),
        ("terminal_numerical_evidence", "cold_terminal_numerical"),
        ("raw_trace_evidence", "cold_raw_trace"),
        ("trace_intervals_evidence", "cold_trace_intervals"),
    ):
        value = producer[field]
        slot = evidence_slots[slot_name]
        if value is None:
            if slot.artifact is not None:
                raise ValueError(f"DIAG3 producer omits PRESENT {slot_name}")
            continue
        reference = _artifact_ref(value, f"DIAG3 producer.{field}")
        if slot.artifact != reference:
            raise ValueError(f"DIAG3 producer binding differs for {slot_name}")
        _resolve_artifact(artifact_root, reference)
    if producer["execution_status"] == "TRACE_NORMALIZATION_FAILED" and (
        failure is None
        or failure.stage is not FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE
        or failure.reason is not FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED
    ):
        raise ValueError("DIAG3 trace-normalization producer contradicts receipt")
    if producer["execution_status"] == "TRACE_NORMALIZATION_FAILED":
        raw_trace = _artifact_ref(
            producer["raw_trace_evidence"], "DIAG3 producer.raw_trace_evidence"
        )
        try:
            normalize_chrome_trace(
                _resolve_artifact(artifact_root, raw_trace),
                phase_schema_sha256=_sha256(
                    producer["phase_schema_sha256"], "DIAG3 phase schema"
                ),
            )
        except (OSError, TypeError, ValueError):
            pass
        else:
            raise ValueError(
                "DIAG3 trace-normalization failure contradicts raw trace bytes"
            )


def _validate_diag4_present_prefix_documents(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV4],
) -> None:
    """Deep-parse every typed setup/preflight document even without a cold producer."""

    present = {
        name: slot.artifact
        for name, slot in evidence_slots.items()
        if slot.artifact is not None
    }
    loaded: dict[str, JsonValue] = {}
    for name, reference in present.items():
        if name != "native_reference":
            loaded[name] = _load_ref_json(artifact_root, reference, f"DIAG4 {name}")
    snapshot: SnapshotPublication | None = None
    source_reference = present.get("source_manifest")
    if source_reference is not None:
        snapshot = load_snapshot(
            artifact_root / "source-snapshot",
            required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
        )
        if source_reference.sha256 != snapshot.manifest_sha256:
            raise ValueError("DIAG4 source-manifest reference differs")
    frozen_value = loaded.get("frozen_numerical_subset")
    if frozen_value is not None:
        _, entries = _parse_diag4_frozen_numerical_subset_payload(frozen_value)
        validate_diag4_frozen_numerical_subset_payload(
            frozen_value,
            artifact_root=artifact_root,
            expected_entries=entries,
        )
    if "native_reference" in present:
        validate_native_equivalent_reference(artifact_root / "native-reference")
    authority: dict[str, JsonValue] | None = None
    if "policy_authority" in present:
        authority = validate_diag2_policy_authority_payload(
            loaded["policy_authority"], artifact_root=artifact_root
        )
    for name, stage in (
        ("supervisor_before_preflight", "BEFORE_PREFLIGHT"),
        ("supervisor_before_cold", "BEFORE_COLD"),
    ):
        if name in present:
            validate_diag2_supervisor_zero_payload(
                loaded[name],
                artifact_root=artifact_root,
                expected_stage=stage,
                allow_failure=True,
            )
    for mode in ("preflight", "cold"):
        producer_name = f"{mode}_producer"
        if producer_name in present:
            validate_diag4_producer_payload(loaded[producer_name], mode=mode)
    if "preflight_runtime" in present:
        if snapshot is None:
            raise ValueError("DIAG4 preflight runtime omits source authority")
        validate_runtime_evidence(
            _resolve_artifact(artifact_root, present["preflight_runtime"]),
            snapshot_root=snapshot.root,
            campaign_root=artifact_root,
            required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
        )
    preflight_policy: PolicyEvidence | None = None
    if "preflight_policy" in present:
        preflight_policy = _parse_policy(loaded["preflight_policy"])
    if (
        authority is not None
        and preflight_policy is not None
        and preflight_policy != _diag2_policy_evidence(authority)
    ):
        raise ValueError("DIAG4 preflight policy differs from authority")
    preflight_producer = loaded.get("preflight_producer")
    if preflight_producer is not None:
        for field, slot_name in (
            ("runtime_evidence", "preflight_runtime"),
            ("policy_evidence", "preflight_policy"),
        ):
            if _artifact_ref(
                _mapping(preflight_producer, "DIAG4 preflight producer")[field],
                f"DIAG4 preflight producer.{field}",
            ) != present.get(slot_name):
                raise ValueError(f"DIAG4 preflight producer {field} binding differs")


def _validate_diag4_stage_vector(
    evidence_slots: Mapping[str, EvidenceSlotV4],
    *,
    failure: StructuredFailureV4,
) -> None:
    """Validate exact ordering, stage state bounds, and absence-reason placement."""

    ordered_names = tuple(DIAG4_EVIDENCE_SLOT_PATHS)
    if tuple(evidence_slots) != ordered_names:
        raise ValueError("DIAG4 evidence slots differ from the frozen schema")
    if evidence_slots["supervisor_terminal"].state is not EvidenceState.PRESENT:
        raise ValueError("DIAG4 evidence vector omits supervisor terminal closure")
    minimum_present = {
        FailureStageV4.AUTHORITY: 0,
        FailureStageV4.SETUP: 0,
        FailureStageV4.BEFORE_PREFLIGHT: 4,
        FailureStageV4.PREFLIGHT: 5,
        FailureStageV4.BEFORE_COLD: 12,
        FailureStageV4.COLD: 13,
        FailureStageV4.NUMERICAL_COMMIT: 20,
        FailureStageV4.RECEIPT: 25,
        FailureStageV4.PUBLICATION: 26,
        FailureStageV4.SCIENTIFIC: 26,
    }[failure.stage]
    maximum_present = {
        FailureStageV4.AUTHORITY: 0,
        FailureStageV4.SETUP: 4,
        FailureStageV4.BEFORE_PREFLIGHT: 5,
        FailureStageV4.PREFLIGHT: 12,
        FailureStageV4.BEFORE_COLD: 13,
        FailureStageV4.COLD: 20,
        FailureStageV4.NUMERICAL_COMMIT: 24,
        FailureStageV4.RECEIPT: 25,
        FailureStageV4.PUBLICATION: 26,
        FailureStageV4.SCIENTIFIC: 26,
    }[failure.stage]
    nonterminal_names = ordered_names[:-1]
    if any(
        evidence_slots[name].state is not EvidenceState.PRESENT
        for name in nonterminal_names[:minimum_present]
    ) or any(
        evidence_slots[name].state is not EvidenceState.ABSENT
        for name in nonterminal_names[maximum_present:]
    ):
        raise ValueError("DIAG4 stage-specific evidence vector differs")
    if failure.stage is FailureStageV4.SETUP:
        setup_states = tuple(evidence_slots[name].state for name in ordered_names[:4])
        first_absent = next(
            (
                index
                for index, state in enumerate(setup_states)
                if state is EvidenceState.ABSENT
            ),
            len(setup_states),
        )
        if any(state is EvidenceState.PRESENT for state in setup_states[first_absent:]):
            raise ValueError("DIAG4 setup evidence prefix differs")
    absent_names = tuple(
        name
        for name in ordered_names
        if evidence_slots[name].state is EvidenceState.ABSENT
    )
    if absent_names and (
        evidence_slots[absent_names[0]].reason is not failure.reason
        or any(evidence_slots[name].reason is not None for name in absent_names[1:])
    ):
        raise ValueError("DIAG4 absence reasons differ from terminal outcome")
    numerical_states = tuple(
        evidence_slots[name].state
        for name in (
            "cold_history",
            "cold_terminal_numerical",
            "cold_solve_timing",
            "cold_safeguard_telemetry",
        )
    )
    if len(frozenset(numerical_states)) != 1:
        raise ValueError("DIAG4 atomic scientific subgroup differs")
    groups = (
        ordered_names[:4],
        ordered_names[4:12],
        ordered_names[12:24],
    )
    earlier_group_incomplete = False
    for group in groups:
        group_complete = all(
            evidence_slots[name].state is EvidenceState.PRESENT for name in group
        )
        if earlier_group_incomplete and any(
            evidence_slots[name].state is EvidenceState.PRESENT for name in group
        ):
            raise ValueError("DIAG4 evidence group prefix differs")
        earlier_group_incomplete = earlier_group_incomplete or not group_complete


def _validate_diag4_slots(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV4],
    *,
    failure: StructuredFailureV4,
) -> None:
    """Validate the trace-free v4 layout and every committed scientific join."""

    _validate_diag4_stage_vector(evidence_slots, failure=failure)
    root = artifact_root.resolve(strict=True)
    for name, slot in evidence_slots.items():
        diag4_evidence_slot_payload(slot)
        if slot.artifact is not None:
            if slot.artifact.relative_path != DIAG4_EVIDENCE_SLOT_PATHS[name]:
                raise ValueError(f"DIAG4 {name} path differs from the frozen layout")
            _resolve_artifact(root, slot.artifact)
    _validate_diag4_present_prefix_documents(artifact_root, evidence_slots)
    producer_slot = evidence_slots["cold_producer"]
    if producer_slot.artifact is None:
        return
    producer = validate_diag4_producer_payload(
        _load_ref_json(artifact_root, producer_slot.artifact, "DIAG4 cold producer"),
        mode="cold",
    )
    producer_bindings = {
        "runtime_evidence": "cold_runtime",
        "policy_evidence": "cold_policy",
        "history_evidence": "cold_history",
        "terminal_numerical_evidence": "cold_terminal_numerical",
        "solve_timing_evidence": "cold_solve_timing",
        "safeguard_telemetry_evidence": "cold_safeguard_telemetry",
    }
    for field, slot_name in producer_bindings.items():
        reference = _artifact_ref(producer[field], f"DIAG4 producer.{field}")
        if evidence_slots[slot_name].artifact != reference:
            raise ValueError(f"DIAG4 producer binding differs for {slot_name}")
        _resolve_artifact(artifact_root, reference)
    required = (
        "cold_history",
        "cold_terminal_numerical",
        "cold_solve_timing",
        "cold_safeguard_telemetry",
        "cold_process",
        "cold_policy",
        "source_manifest",
        "execution",
    )
    if any(evidence_slots[name].artifact is None for name in required):
        return
    refs = {
        name: _present_artifact(evidence_slots[name], name)
        for name in DIAG4_EVIDENCE_SLOT_PATHS
        if evidence_slots[name].artifact is not None
    }
    history = _parse_history(
        _load_ref_json(artifact_root, refs["cold_history"], "DIAG4 history"),
        defer_step_bounds=True,
    )
    timing_payload = _load_ref_json(
        artifact_root, refs["cold_solve_timing"], "DIAG4 solve timing"
    )
    timing = validate_solve_timing_evidence_payload(timing_payload)
    telemetry = validate_safeguard_telemetry_payload(
        _load_ref_json(
            artifact_root,
            refs["cold_safeguard_telemetry"],
            "DIAG4 safeguard telemetry",
        ),
        history=history,
        expected_history_evidence=refs["cold_history"],
    )
    identities = (
        "problem_sha256",
        "optimizer_options_sha256",
        "base_neq_gntr1_policy_sha256",
        "scaling_sha256",
        "bootstrap_state_sha256",
        "initial_physical_state_sha256",
        "identity_sha256",
    )
    producer_identity = tuple(
        _sha256(producer[name], f"DIAG4 producer.{name}") for name in identities
    )
    timing_identity = (
        timing.problem_sha256,
        timing.optimizer_options_sha256,
        timing.base_neq_gntr1_policy_sha256,
        timing.scaling_sha256,
        timing.bootstrap_state_sha256,
        timing.initial_physical_state_sha256,
        timing.identity_sha256,
    )
    telemetry_identity = (
        telemetry.problem_sha256,
        telemetry.optimizer_options_sha256,
        telemetry.base_neq_gntr1_policy_sha256,
        telemetry.scaling_sha256,
        telemetry.bootstrap_state_sha256,
        telemetry.initial_physical_state_sha256,
        telemetry.identity_sha256,
    )
    policy = _parse_policy(
        _load_ref_json(artifact_root, refs["cold_policy"], "DIAG4 policy")
    )
    if (
        producer_identity != timing_identity
        or producer_identity != telemetry_identity
        or policy.policy_sha256 != timing.base_neq_gntr1_policy_sha256
        or refs["source_manifest"].sha256 != timing.source_manifest_sha256
    ):
        raise ValueError("DIAG4 producer/timing/telemetry authority join differs")
    supporting_evidence = {
        name: reference
        for name, reference in refs.items()
        if name not in {"execution", "supervisor_terminal"}
    }
    validate_diag4_execution_evidence_payload(
        _load_ref_json(artifact_root, refs["execution"], "DIAG4 execution"),
        supporting_evidence=supporting_evidence,
        solve_timing=timing_payload,
        producer=producer,
        process=_load_ref_json(
            artifact_root, refs["cold_process"], "DIAG4 cold process"
        ),
    )


def _diag2_failure_payload(failure: StructuredFailureV2) -> dict[str, JsonValue]:
    return {
        "stage": failure.stage.value,
        "reason": {
            "code": failure.reason.value,
            "detail_sha256": failure.detail_sha256,
        },
    }


def _parse_diag2_failure(value: JsonValue) -> StructuredFailureV2:
    payload = _mapping(value, "diagnostic failure")
    _exact_keys(payload, frozenset({"stage", "reason"}), "diagnostic failure")
    reason = _mapping(payload["reason"], "diagnostic failure.reason")
    _exact_keys(
        reason, frozenset({"code", "detail_sha256"}), "diagnostic failure.reason"
    )
    stage = FailureStageV2(_string(payload["stage"], "diagnostic failure.stage"))
    code = FailureReasonCodeV2(
        _string(reason["code"], "diagnostic failure.reason.code")
    )
    if code not in DIAG2_STAGE_REASON_CODES[stage]:
        raise ValueError("diagnostic failure stage/reason pairing differs")
    return StructuredFailureV2(
        stage, code, _sha256(reason["detail_sha256"], "diagnostic failure detail SHA")
    )


def diag4_terminal_outcome_payload(
    outcome: StructuredFailureV4,
) -> dict[str, JsonValue]:
    if outcome.reason not in DIAG4_STAGE_REASON_ORDER[outcome.stage]:
        raise ValueError("DIAG4 terminal stage/reason pairing differs")
    return {
        "stage": outcome.stage.value,
        "reason": {
            "code": outcome.reason.value,
            "detail_sha256": _sha256(
                outcome.detail_sha256, "DIAG4 terminal detail SHA"
            ),
        },
    }


def parse_diag4_terminal_outcome(value: JsonValue) -> StructuredFailureV4:
    payload = _mapping(value, "DIAG4 terminal outcome")
    _exact_keys(payload, frozenset({"stage", "reason"}), "DIAG4 terminal outcome")
    reason_payload = _mapping(payload["reason"], "DIAG4 terminal outcome.reason")
    _exact_keys(
        reason_payload,
        frozenset({"code", "detail_sha256"}),
        "DIAG4 terminal outcome.reason",
    )
    stage = FailureStageV4(_string(payload["stage"], "DIAG4 terminal stage"))
    reason = FailureReasonCodeV4(
        _string(reason_payload["code"], "DIAG4 terminal reason")
    )
    outcome = StructuredFailureV4(
        stage,
        reason,
        _sha256(reason_payload["detail_sha256"], "DIAG4 terminal detail SHA"),
    )
    if diag4_terminal_outcome_payload(outcome) != payload:
        raise ValueError("DIAG4 terminal outcome differs")
    return outcome


def select_diag4_terminal_outcome(
    candidates: Iterable[StructuredFailureV4],
) -> StructuredFailureV4:
    """Select the exact first v4 stage/reason independent of input order."""

    unique = frozenset(candidates)
    if not unique:
        raise ValueError("DIAG4 terminal outcome candidates are empty")
    for stage in DIAG4_FAILURE_STAGE_ORDER:
        for reason in DIAG4_STAGE_REASON_ORDER[stage]:
            matches = tuple(
                candidate
                for candidate in unique
                if candidate.stage is stage and candidate.reason is reason
            )
            if len(matches) > 1:
                raise ValueError("DIAG4 duplicate stage/reason candidates differ")
            if matches:
                return matches[0]
    raise AssertionError("DIAG4 terminal outcome selection exhausted its schema")


def _diag4_expected_launched_children(
    outcome: StructuredFailureV4,
) -> tuple[str, ...]:
    if outcome.stage in {
        FailureStageV4.AUTHORITY,
        FailureStageV4.SETUP,
        FailureStageV4.BEFORE_PREFLIGHT,
    } or (
        outcome.stage is FailureStageV4.PREFLIGHT
        and outcome.reason is FailureReasonCodeV4.PREFLIGHT_LAUNCH_FAILED
    ):
        return ()
    if outcome.stage in {FailureStageV4.PREFLIGHT, FailureStageV4.BEFORE_COLD} or (
        outcome.stage is FailureStageV4.COLD
        and outcome.reason is FailureReasonCodeV4.COLD_LAUNCH_FAILED
    ):
        return ("preflight",)
    return ("preflight", "cold")


def build_diag4_supervisor_terminal_payload(
    *,
    outcome: StructuredFailureV4,
    launched_children: tuple[str, ...],
    staging_root: Path,
    final_root: Path,
    nonce: str,
) -> dict[str, JsonValue]:
    """Construct the sole v4 terminal from one receipt-owned ordered outcome."""

    if launched_children != _diag4_expected_launched_children(outcome):
        raise ValueError("DIAG4 child sequence differs from terminal stage/reason")
    if len(nonce) != 32 or any(character not in _LOWER_HEX for character in nonce):
        raise ValueError("DIAG4 nonce must be 32 lower-case hex digits")
    staging = staging_root.resolve(strict=False)
    final = final_root.resolve(strict=False)
    if (
        staging.parent != final.parent
        or staging.name != f"{final.name}.partial-{nonce}"
    ):
        raise ValueError("DIAG4 publication roots differ")
    scientific = outcome.stage is FailureStageV4.SCIENTIFIC
    quality_hit = outcome.reason is FailureReasonCodeV4.QUALITY_HIT
    return {
        "schema_version": DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
        "route": DIAG4_ROUTE,
        "numerical_route": DIAG4_NUMERICAL_ROUTE,
        "plan_sha256": DIAG4_PLAN_SHA256,
        "disposition": "COMPLETE" if scientific else "INCOMPLETE",
        "terminal_outcome": diag4_terminal_outcome_payload(outcome),
        "launched_children": list(launched_children),
        "publication": {
            "staging_root": str(staging),
            "final_root": str(final),
            "nonce": nonce,
        },
        "phase_attribution": "NOT_PRODUCED",
        "next_route": (
            DIAG4_CONDITIONAL_TIMING_ROUTE if quality_hit else "NOT_PRODUCED"
        ),
        "speed_comparison": (
            "CONDITIONAL_ENGINEERING_CONTEXT" if quality_hit else "NOT_PRODUCED"
        ),
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }


def parse_diag4_supervisor_terminal_payload(
    value: JsonValue,
) -> tuple[dict[str, JsonValue], StructuredFailureV4]:
    payload = _mapping(value, "DIAG4 supervisor terminal")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "plan_sha256",
                "disposition",
                "terminal_outcome",
                "launched_children",
                "publication",
                "phase_attribution",
                "next_route",
                "speed_comparison",
                "promotion_authorized",
                "formal_comparison",
            }
        ),
        "DIAG4 supervisor terminal",
    )
    publication = _mapping(payload["publication"], "DIAG4 publication")
    _exact_keys(
        publication,
        frozenset({"staging_root", "final_root", "nonce"}),
        "DIAG4 publication",
    )
    outcome = parse_diag4_terminal_outcome(payload["terminal_outcome"])
    rebuilt = build_diag4_supervisor_terminal_payload(
        outcome=outcome,
        launched_children=tuple(
            _string(item, "DIAG4 launched child")
            for item in _array(payload["launched_children"], "DIAG4 children")
        ),
        staging_root=Path(_string(publication["staging_root"], "DIAG4 staging root")),
        final_root=Path(_string(publication["final_root"], "DIAG4 final root")),
        nonce=_string(publication["nonce"], "DIAG4 nonce"),
    )
    if payload != rebuilt:
        raise ValueError("DIAG4 supervisor terminal differs")
    return payload, outcome


def select_diag2_failure(
    candidates: Iterable[StructuredFailureV2],
) -> StructuredFailureV2:
    """Select the sole earliest legal candidate in the frozen stage precedence."""

    by_stage: dict[FailureStageV2, StructuredFailureV2] = {}
    for candidate in candidates:
        if not isinstance(candidate, StructuredFailureV2):
            raise TypeError("DIAG2 failure candidate type differs")
        if (
            not isinstance(candidate.stage, FailureStageV2)
            or not isinstance(candidate.reason, FailureReasonCodeV2)
            or candidate.reason not in DIAG2_STAGE_REASON_CODES[candidate.stage]
        ):
            raise ValueError("DIAG2 failure candidate stage/reason pairing differs")
        _sha256(candidate.detail_sha256, "DIAG2 failure candidate detail SHA")
        if candidate.stage in by_stage:
            raise ValueError("DIAG2 failure candidates duplicate a stage")
        by_stage[candidate.stage] = candidate
    if not by_stage:
        raise ValueError("DIAG2 failure candidates are empty")
    return next(
        by_stage[stage] for stage in DIAG2_FAILURE_STAGE_ORDER if stage in by_stage
    )


@dataclass(frozen=True, slots=True)
class SupervisorQueryV2:
    argv: tuple[str, ...]
    query_executable_sha256: str
    launched: bool
    timed_out: bool
    returncode: int | None
    stdout: ArtifactRef
    stderr: ArtifactRef


class Diag2PreflightGateError(ValueError):
    """Typed nonauthorization carrying the receipt-owned reason and authority slot."""

    def __init__(
        self,
        reason: FailureReasonCodeV2,
        offending_slot: str,
        detail: str,
    ) -> None:
        self.reason = reason
        self.offending_slot = offending_slot
        self.detail_sha256 = hashlib.sha256(detail.encode()).hexdigest()
        super().__init__(f"{reason.value}:{offending_slot}:{self.detail_sha256}")


class Diag2SetupGateError(ValueError):
    """Typed setup nonauthorization used before any supervised child launch."""

    def __init__(
        self,
        reason: FailureReasonCodeV2,
        offending_slot: str,
        detail: str,
    ) -> None:
        self.reason = reason
        self.offending_slot = offending_slot
        self.detail_sha256 = hashlib.sha256(detail.encode()).hexdigest()
        super().__init__(f"{reason.value}:{offending_slot}:{self.detail_sha256}")


class Diag4NumericalDocumentError(ValueError):
    """Typed four-document rejection owned by NUMERICAL_COMMIT convergence."""

    def __init__(self, reason: FailureReasonCodeV4, detail: str) -> None:
        if reason not in {
            FailureReasonCodeV4.TIMING_INVALID,
            FailureReasonCodeV4.SAFEGUARD_TELEMETRY_INVALID,
            FailureReasonCodeV4.NUMERICAL_IDENTITY_MISMATCH,
            FailureReasonCodeV4.PENDING_RESULT_INVALID,
        }:
            raise ValueError("DIAG4 numerical document reason differs")
        self.reason = reason
        self.detail_sha256 = hashlib.sha256(detail.encode()).hexdigest()
        super().__init__(f"{reason.value}:{self.detail_sha256}")


class Diag5NumericalDocumentError(ValueError):
    """Typed V5 four-document rejection owned by numerical convergence."""

    def __init__(self, reason: FailureReasonCodeV5, detail: str) -> None:
        if reason not in {
            FailureReasonCodeV5.TIMING_INVALID,
            FailureReasonCodeV5.SAFEGUARD_TELEMETRY_INVALID,
            FailureReasonCodeV5.NUMERICAL_IDENTITY_MISMATCH,
            FailureReasonCodeV5.PENDING_RESULT_INVALID,
        }:
            raise ValueError("DIAG5 numerical document reason differs")
        self.reason = reason
        self.detail_sha256 = hashlib.sha256(detail.encode()).hexdigest()
        super().__init__(f"{reason.value}:{self.detail_sha256}")


class Diag5ReceiptConstructionError(ValueError):
    """Typed V5 receipt rejection independent of exception message text."""

    def __init__(self, reason: FailureReasonCodeV5, detail: str) -> None:
        if reason not in {
            FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID,
            FailureReasonCodeV5.GROUP_PREFIX_INVALID,
            FailureReasonCodeV5.RECEIPT_SCHEMA_INVALID,
        }:
            raise ValueError("DIAG5 receipt construction reason differs")
        self.reason = reason
        self.detail_sha256 = hashlib.sha256(detail.encode()).hexdigest()
        super().__init__(f"{reason.value}:{self.detail_sha256}")


def classify_diag5_receipt_construction_error(
    error: BaseException,
) -> FailureReasonCodeV5:
    """Classify a receipt build error without inspecting human-readable text."""

    if isinstance(error, Diag5ReceiptConstructionError):
        return error.reason
    return FailureReasonCodeV5.RECEIPT_SCHEMA_INVALID


@dataclass(frozen=True, slots=True)
class Diag2ColdEvidenceClassification:
    """Minimum-typed cold prefix and its first physical schema failure."""

    typed_slots: tuple[str, ...]
    failure: StructuredFailureV2 | None
    offending_slot: str | None


@dataclass(frozen=True, slots=True)
class Diag4ColdEvidenceClassification:
    typed_prefix: tuple[str, ...]
    outcome: StructuredFailureV4 | None
    offending_slot: str | None


def diag2_postlaunch_setup_failure(
    *,
    after_mode: str,
    reason: FailureReasonCodeV2,
    detail_sha256: str,
) -> StructuredFailureV2:
    """Map a setup-integrity drift observed after one child to its sealed stage."""

    if reason not in _DIAG2_POSTLAUNCH_SETUP_SLOT:
        raise ValueError("post-launch setup failure reason differs")
    stage = {
        "preflight": FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
        "cold": FailureStageV2.COLD_SOURCE_FAILURE,
    }.get(after_mode)
    if stage is None:
        raise ValueError("post-launch setup failure mode differs")
    return StructuredFailureV2(
        stage, reason, _sha256(detail_sha256, "post-launch setup detail SHA")
    )


def build_diag2_frozen_numerical_subset_payload(
    entries: Mapping[str, str] | None = None,
) -> dict[str, JsonValue]:
    """Bind the reviewed numerical-source subset without accepting caller order."""

    expected = dict(DIAG2_FROZEN_NUMERICAL_ENTRIES)
    if entries is not None and dict(entries) != expected:
        raise ValueError("frozen numerical subset entries differ from the SSOT")
    rows: list[dict[str, JsonValue]] = []
    for relative_path, digest in DIAG2_FROZEN_NUMERICAL_ENTRIES:
        path = Path(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative_path
            or not path.parts
        ):
            raise ValueError("frozen numerical subset path is not canonical")
        rows.append(
            {
                "relative_path": relative_path,
                "sha256": _sha256(digest, f"frozen subset {relative_path} SHA"),
            }
        )
    if not rows:
        raise ValueError("frozen numerical subset must not be empty")
    return {
        "schema_version": DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
        "plan_sha256": DIAG2_PLAN_SHA256,
        "entries": rows,
    }


def _diag2_filtered_source_entries(snapshot: SnapshotPublication) -> list[JsonValue]:
    return [
        entry.to_payload()
        for entry in snapshot.entries
        if entry.relative_path not in DIAG2_SOURCE_DELTA_ALLOWLIST
    ]


def validate_diag2_source_snapshot_authority(
    artifact_root: Path,
) -> SnapshotPublication:
    """Reject every source-manifest delta outside the reviewed DIAG2 allowlist."""

    snapshot = load_snapshot(artifact_root / "source-snapshot")
    observed_roles = {entry.relative_path: entry.role for entry in snapshot.entries}
    if any(
        observed_roles.get(path) != role
        for path, role in REQUIRED_SOURCE_ROLE_BINDINGS.items()
    ):
        raise ValueError("source snapshot diagnostic path/role binding differs")
    filtered = _diag2_filtered_source_entries(snapshot)
    if len(filtered) != DIAG2_BASELINE_FILTERED_ENTRY_COUNT:
        raise ValueError("source snapshot filtered entry count differs from DIAG1")
    digest = hashlib.sha256(canonical_json_bytes(filtered)).hexdigest()
    if digest != DIAG2_BASELINE_FILTERED_ENTRIES_SHA256:
        raise ValueError("source snapshot contains a non-allowlisted DIAG1 delta")
    return snapshot


def validate_diag2_frozen_numerical_subset_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
) -> dict[str, JsonValue]:
    """Recompute every frozen numerical entry from the immutable source snapshot."""

    expected = build_diag2_frozen_numerical_subset_payload()
    if value != expected:
        raise ValueError("frozen numerical subset payload differs from the SSOT")
    authority = validate_diag2_source_snapshot_authority(artifact_root)
    entries = {entry.relative_path: entry for entry in authority.entries}
    for relative_path, digest in DIAG2_FROZEN_NUMERICAL_ENTRIES:
        entry = entries.get(relative_path)
        if entry is None:
            raise ValueError(f"source snapshot omits frozen path: {relative_path}")
        expected_role = (
            "benchmark"
            if relative_path.startswith("benchmarks/")
            else "execution_source"
        )
        if entry.role != expected_role or entry.sha256 != digest:
            raise ValueError(f"source snapshot frozen entry differs: {relative_path}")
    return expected


def _build_frozen_numerical_subset_payload(
    entries: Mapping[str, str], *, schema_version: str, plan_sha256: str
) -> dict[str, JsonValue]:
    """Bind a numerical subset under one explicit wire generation."""

    rows: list[dict[str, JsonValue]] = []
    for relative_path, digest in sorted(entries.items()):
        path = Path(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative_path
            or not path.parts
        ):
            raise ValueError("DIAG4 frozen numerical subset path is not canonical")
        rows.append(
            {
                "relative_path": relative_path,
                "sha256": _sha256(digest, f"DIAG4 frozen subset {relative_path} SHA"),
            }
        )
    if not rows:
        raise ValueError("DIAG4 frozen numerical subset must not be empty")
    return {
        "schema_version": schema_version,
        "plan_sha256": plan_sha256,
        "entries": rows,
    }


def build_diag4_frozen_numerical_subset_payload(
    entries: Mapping[str, str],
) -> dict[str, JsonValue]:
    """Bind the DIAG4 numerical subset supplied by the held typed authority."""

    return _build_frozen_numerical_subset_payload(
        entries,
        schema_version=DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
        plan_sha256=DIAG4_PLAN_SHA256,
    )


def _parse_diag4_frozen_numerical_subset_payload(
    value: JsonValue,
) -> tuple[dict[str, JsonValue], dict[str, str]]:
    payload = _mapping(value, "DIAG4 frozen numerical subset")
    _exact_keys(
        payload,
        frozenset({"schema_version", "plan_sha256", "entries"}),
        "DIAG4 frozen numerical subset",
    )
    entries: dict[str, str] = {}
    for index, item in enumerate(
        _array(payload["entries"], "DIAG4 frozen numerical subset.entries")
    ):
        row = _mapping(item, f"DIAG4 frozen numerical subset.entries[{index}]")
        _exact_keys(
            row,
            frozenset({"relative_path", "sha256"}),
            f"DIAG4 frozen numerical subset.entries[{index}]",
        )
        relative_path = _string(row["relative_path"], "DIAG4 frozen subset path")
        if relative_path in entries:
            raise ValueError("DIAG4 frozen numerical subset duplicates a path")
        entries[relative_path] = _sha256(
            row["sha256"], f"DIAG4 frozen subset {relative_path} SHA"
        )
    rebuilt = build_diag4_frozen_numerical_subset_payload(entries)
    if payload != rebuilt:
        raise ValueError("DIAG4 frozen numerical subset is not canonical")
    return payload, entries


def validate_diag4_frozen_numerical_subset_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
    expected_entries: Mapping[str, str],
) -> dict[str, JsonValue]:
    """Join the subset artifact to authority-qualified source-snapshot bytes."""

    payload, _ = _parse_diag4_frozen_numerical_subset_payload(value)
    expected = build_diag4_frozen_numerical_subset_payload(expected_entries)
    if payload != expected:
        raise ValueError("DIAG4 frozen numerical subset differs from authority")
    snapshot = load_snapshot(
        artifact_root / "source-snapshot",
        required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
    )
    observed = {entry.relative_path: entry for entry in snapshot.entries}
    for relative_path, digest in sorted(expected_entries.items()):
        entry = observed.get(relative_path)
        if entry is None:
            raise ValueError(
                f"DIAG4 source snapshot omits frozen path: {relative_path}"
            )
        expected_role = (
            "benchmark"
            if relative_path.startswith("benchmarks/")
            else ("test" if relative_path.startswith("tests/") else "execution_source")
        )
        if entry.role != expected_role or entry.sha256 != digest:
            raise ValueError(
                f"DIAG4 source snapshot frozen entry differs: {relative_path}"
            )
    return expected


def build_diag5_frozen_numerical_subset_payload(
    entries: Mapping[str, str],
) -> dict[str, JsonValue]:
    """Bind the DIAG5 numerical subset under its independent v2 identity."""

    return _build_frozen_numerical_subset_payload(
        entries,
        schema_version=DIAG5_FROZEN_SUBSET_SCHEMA_VERSION,
        plan_sha256=DIAG5_PLAN_SHA256,
    )


def validate_diag5_frozen_numerical_subset_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
    expected_entries: Mapping[str, str],
) -> dict[str, JsonValue]:
    payload = _mapping(value, "DIAG5 frozen numerical subset")
    _exact_keys(
        payload,
        frozenset({"schema_version", "plan_sha256", "entries"}),
        "DIAG5 frozen numerical subset",
    )
    entries: dict[str, str] = {}
    for index, item in enumerate(
        _array(payload["entries"], "DIAG5 frozen numerical subset.entries")
    ):
        row = _mapping(item, f"DIAG5 frozen numerical subset.entries[{index}]")
        _exact_keys(row, frozenset({"relative_path", "sha256"}), "DIAG5 subset row")
        relative_path = _string(row["relative_path"], "DIAG5 frozen subset path")
        if relative_path in entries:
            raise ValueError("DIAG5 frozen numerical subset duplicates a path")
        entries[relative_path] = _sha256(row["sha256"], "DIAG5 frozen subset SHA")
    expected = build_diag5_frozen_numerical_subset_payload(expected_entries)
    if payload != expected or entries != dict(expected_entries):
        raise ValueError("DIAG5 frozen numerical subset differs from authority")
    snapshot = load_snapshot(
        _diag5_held_path(artifact_root, "source-snapshot"),
        required_roles=DIAG5_GPU_SNAPSHOT_ROLES,
    )
    observed = {entry.relative_path: entry for entry in snapshot.entries}
    for relative_path, digest in sorted(expected_entries.items()):
        entry = observed.get(relative_path)
        expected_role = (
            "benchmark"
            if relative_path.startswith("benchmarks/")
            else "test"
            if relative_path.startswith("tests/")
            else "execution_source"
        )
        if entry is None or entry.role != expected_role or entry.sha256 != digest:
            raise ValueError(
                f"DIAG5 source snapshot frozen entry differs: {relative_path}"
            )
    return expected


def _build_policy_authority_payload(
    *,
    native_reference: ArtifactRef,
    reference_volume: float,
    volume_target: float,
    native_raw_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
    schema_version: str,
    route: str,
    plan_sha256: str,
) -> dict[str, JsonValue]:
    """Construct one generation's NumPy authority and derive all hashes locally."""

    native = np.ascontiguousarray(native_raw_equalities, dtype=np.dtype("<f8"))
    scale = np.ascontiguousarray(constraint_inverse_scale, dtype=np.dtype("<f8"))
    if native.shape != (EQUALITY_SIZE,) or scale.shape != (EQUALITY_SIZE,):
        raise ValueError("DIAG2 policy arrays must both have shape (255,)")
    if (
        not np.all(np.isfinite(native))
        or not np.all(np.isfinite(scale))
        or np.any(scale == 0.0)
    ):
        raise ValueError("DIAG2 policy arrays must be finite and scale nonzero")
    reference_volume_value = _number(reference_volume, "reference volume")
    volume_target_value = _number(volume_target, "volume target")
    native_sha = exact_numeric_tree_sha256(native)
    scale_sha = hashlib.sha256(scale.tobytes()).hexdigest()
    if (
        volume_target_value != reference_volume_value - float(native[254])
        or volume_target_value.hex() != DIAG2_VOLUME_TARGET_HEX
        or float(scale[0]).hex() != DIAG2_BOOZER_SCALE_HEX
        or not np.array_equal(
            scale[:254],
            np.full(254, 1.0 / np.sqrt(np.float64(254.0)), dtype=np.dtype("<f8")),
        )
        or float(scale[254]).hex() != DIAG2_VOLUME_SCALE_HEX
        or scale_sha != DIAG2_SCALE_SHA256
    ):
        raise ValueError("DIAG2 policy scale differs from the frozen NumPy derivation")
    policy_identity = (
        "single-stage-native-equivalent-quality-policy-v1",
        "single-stage-fullspace-neq-gntr1-result-v1",
        NUMERICAL_ROUTE,
        native,
        native_sha,
        scale,
        OBJECTIVE_MAXIMUM,
        STATE_SIZE,
        EQUALITY_SIZE,
        2110,
        RAW_EQUALITY_ABSOLUTE_TOLERANCE,
        RAW_EQUALITY_RELATIVE_TOLERANCE,
        FEASIBILITY_MAXIMUM,
        RESIDUAL_VALUE_DEFECT_MAXIMUM,
        RESIDUAL_GRADIENT_DEFECT_MAXIMUM,
        TRANSPOSE_DEFECT_MAXIMUM,
        tuple((name, FROZEN_GNTR_OPTIONS[name]) for name in GNTR_OPTION_ORDER),
    )
    result: dict[str, JsonValue] = {
        "schema_version": schema_version,
        "route": route,
        "plan_sha256": plan_sha256,
        "derivation_kind": "VALIDATED_REFERENCE_NUMPY_FP64",
        "native_reference": _artifact_ref_payload(native_reference),
        "reference_volume": reference_volume_value,
        "volume_target": volume_target_value,
        "native_raw_equalities": native.tolist(),
        "native_raw_equalities_sha256": native_sha,
        "constraint_inverse_scale": scale.tolist(),
        "constraint_inverse_scale_sha256": scale_sha,
        "objective_target": OBJECTIVE_MAXIMUM,
        "state_size": STATE_SIZE,
        "equality_size": EQUALITY_SIZE,
        "objective_residual_size": 2110,
        "component_absolute_tolerance": RAW_EQUALITY_ABSOLUTE_TOLERANCE,
        "component_relative_tolerance": RAW_EQUALITY_RELATIVE_TOLERANCE,
        "scaled_feasibility_tolerance": FEASIBILITY_MAXIMUM,
        "residual_value_defect_tolerance": RESIDUAL_VALUE_DEFECT_MAXIMUM,
        "residual_gradient_defect_tolerance": RESIDUAL_GRADIENT_DEFECT_MAXIMUM,
        "transpose_defect_tolerance": TRANSPOSE_DEFECT_MAXIMUM,
        "gntr_options": dict(FROZEN_GNTR_OPTIONS),
        "policy_sha256": exact_numeric_tree_sha256(policy_identity),
    }
    if result["policy_sha256"] != DIAG2_POLICY_SHA256:
        raise ValueError("DIAG2 policy identity differs from the frozen authority")
    return result


def build_diag2_policy_authority_payload(
    *,
    native_reference: ArtifactRef,
    reference_volume: float,
    volume_target: float,
    native_raw_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
) -> dict[str, JsonValue]:
    """Construct only the frozen DIAG2 policy-authority generation."""

    return _build_policy_authority_payload(
        native_reference=native_reference,
        reference_volume=reference_volume,
        volume_target=volume_target,
        native_raw_equalities=native_raw_equalities,
        constraint_inverse_scale=constraint_inverse_scale,
        schema_version=DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION,
        route=DIAG2_ROUTE,
        plan_sha256=DIAG2_PLAN_SHA256,
    )


def _validate_policy_authority_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
    schema_version: str,
    route: str,
    plan_sha256: str,
) -> dict[str, JsonValue]:
    """Reconstruct one explicit policy-authority wire generation."""

    payload = _mapping(value, "DIAG2 policy authority")
    expected_keys = frozenset(
        {
            "schema_version",
            "route",
            "plan_sha256",
            "derivation_kind",
            "native_reference",
            "reference_volume",
            "volume_target",
            "native_raw_equalities",
            "native_raw_equalities_sha256",
            "constraint_inverse_scale",
            "constraint_inverse_scale_sha256",
            "objective_target",
            "state_size",
            "equality_size",
            "objective_residual_size",
            "component_absolute_tolerance",
            "component_relative_tolerance",
            "scaled_feasibility_tolerance",
            "residual_value_defect_tolerance",
            "residual_gradient_defect_tolerance",
            "transpose_defect_tolerance",
            "gntr_options",
            "policy_sha256",
        }
    )
    _exact_keys(payload, expected_keys, "DIAG2 policy authority")
    native_reference = _artifact_ref(
        payload["native_reference"], "policy native reference"
    )
    if native_reference.relative_path != DIAG2_EVIDENCE_SLOT_PATHS["native_reference"]:
        raise ValueError("policy native reference path differs")
    _resolve_artifact(artifact_root, native_reference)
    validation = validate_native_equivalent_reference(
        artifact_root / "native-reference"
    )
    if not validation.usable:
        raise ValueError("native reference is not usable for DIAG2 policy derivation")
    reference_document = _mapping(
        load_canonical_json_bytes(
            (artifact_root / "native-reference" / "reference.json").read_bytes()
        ),
        "native reference",
    )
    evidence = _mapping(reference_document.get("evidence"), "native reference evidence")
    observables = _mapping(evidence.get("observables"), "native reference observables")
    reference_volume = _number(observables.get("volume"), "native reference volume")
    native = np.ascontiguousarray(
        _validated_native_equalities(artifact_root), dtype=np.dtype("<f8")
    )
    volume_target = float(reference_volume - native[254])
    scale = np.empty(EQUALITY_SIZE, dtype=np.dtype("<f8"))
    scale[:254] = 1.0 / np.sqrt(np.float64(254.0))
    scale[254] = 1.0 / abs(volume_target)
    rebuilt = _build_policy_authority_payload(
        native_reference=native_reference,
        reference_volume=reference_volume,
        volume_target=volume_target,
        native_raw_equalities=native,
        constraint_inverse_scale=scale,
        schema_version=schema_version,
        route=route,
        plan_sha256=plan_sha256,
    )
    if payload != rebuilt:
        raise ValueError("DIAG2 policy authority differs from NumPy reconstruction")
    return rebuilt


def validate_diag2_policy_authority_payload(
    value: JsonValue, *, artifact_root: Path
) -> dict[str, JsonValue]:
    """Validate only the frozen DIAG2 policy-authority generation."""

    return _validate_policy_authority_payload(
        value,
        artifact_root=artifact_root,
        schema_version=DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION,
        route=DIAG2_ROUTE,
        plan_sha256=DIAG2_PLAN_SHA256,
    )


def build_diag5_policy_authority_payload(
    *,
    native_reference: ArtifactRef,
    reference_volume: float,
    volume_target: float,
    native_raw_equalities: np.ndarray,
    constraint_inverse_scale: np.ndarray,
) -> dict[str, JsonValue]:
    return _build_policy_authority_payload(
        native_reference=native_reference,
        reference_volume=reference_volume,
        volume_target=volume_target,
        native_raw_equalities=native_raw_equalities,
        constraint_inverse_scale=constraint_inverse_scale,
        schema_version=DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
    )


def validate_diag5_policy_authority_payload(
    value: JsonValue, *, artifact_root: Path
) -> dict[str, JsonValue]:
    return _validate_policy_authority_payload(
        value,
        artifact_root=artifact_root,
        schema_version=DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
    )


def validate_diag2_setup_authorities(
    artifact_root: Path,
    *,
    evidence_slots: Mapping[str, EvidenceSlot],
) -> bool:
    """Authorize child launch only after every immutable setup authority validates."""

    required = (
        "source_manifest",
        "frozen_numerical_subset",
        "native_reference",
        "policy_authority",
    )
    reasons = {
        "source_manifest": FailureReasonCodeV2.SOURCE_PRE,
        "frozen_numerical_subset": FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
        "native_reference": FailureReasonCodeV2.REFERENCE_INVALID,
        "policy_authority": FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
    }
    references: dict[str, ArtifactRef] = {}
    for name in required:
        slot = evidence_slots.get(name)
        if (
            slot is None
            or slot.state is not EvidenceState.PRESENT
            or slot.artifact is None
        ):
            raise Diag2SetupGateError(
                reasons[name], name, f"DIAG2 setup requires PRESENT {name}"
            )
        references[name] = slot.artifact
    try:
        snapshot = validate_diag2_source_snapshot_authority(artifact_root)
        if references["source_manifest"].sha256 != snapshot.manifest_sha256:
            raise ValueError("setup source-manifest reference differs")
    except (OSError, TypeError, ValueError) as error:
        raise Diag2SetupGateError(
            FailureReasonCodeV2.SOURCE_PRE, "source_manifest", str(error)
        ) from error
    try:
        validate_diag2_frozen_numerical_subset_payload(
            _load_ref_json(
                artifact_root,
                references["frozen_numerical_subset"],
                "frozen numerical subset",
            ),
            artifact_root=artifact_root,
        )
    except (OSError, TypeError, ValueError) as error:
        raise Diag2SetupGateError(
            FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
            "frozen_numerical_subset",
            str(error),
        ) from error
    try:
        validation = validate_native_equivalent_reference(
            artifact_root / "native-reference"
        )
        if not validation.usable:
            raise ValueError("setup native reference is unusable")
    except (OSError, TypeError, ValueError) as error:
        raise Diag2SetupGateError(
            FailureReasonCodeV2.REFERENCE_INVALID, "native_reference", str(error)
        ) from error
    try:
        authority = validate_diag2_policy_authority_payload(
            _load_ref_json(
                artifact_root, references["policy_authority"], "policy authority"
            ),
            artifact_root=artifact_root,
        )
        if (
            _artifact_ref(authority["native_reference"], "policy native reference")
            != references["native_reference"]
        ):
            raise ValueError("policy authority native-reference join differs")
    except (OSError, TypeError, ValueError) as error:
        raise Diag2SetupGateError(
            FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
            "policy_authority",
            str(error),
        ) from error
    return True


def _diag2_query_payload(query: SupervisorQueryV2) -> dict[str, JsonValue]:
    _sha256(query.query_executable_sha256, "query executable SHA")
    if not query.argv:
        raise ValueError("supervisor query argv must not be empty")
    if query.launched:
        if query.timed_out:
            if query.returncode is not None:
                raise ValueError("timed-out query must have null return code")
        elif query.returncode is None:
            raise ValueError("completed query must have a return code")
    elif query.timed_out or query.returncode is not None:
        raise ValueError("unlaunched query state is inconsistent")
    return {
        "argv": list(query.argv),
        "query_executable_sha256": query.query_executable_sha256,
        "launched": query.launched,
        "timed_out": query.timed_out,
        "returncode": query.returncode,
        "stdout": _artifact_ref_payload(query.stdout),
        "stderr": _artifact_ref_payload(query.stderr),
    }


_DIAG2_GPU_INVENTORY_ARGV: Final = (
    "nvidia-smi",
    "--query-gpu=uuid,memory.total",
    "--format=csv,noheader,nounits",
)
_DIAG2_COMPUTE_APPS_ARGV: Final = (
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
    "--format=csv,noheader,nounits",
)


def _build_supervisor_zero_payload(
    *,
    stage: str,
    captured_at_monotonic_ns: int,
    captured_at_unix_ns: int,
    supervisor_pid: int,
    supervisor_start_ticks: int,
    gpu_uuid: str,
    visible_device: str,
    gpu_inventory_query: SupervisorQueryV2,
    compute_apps_query: SupervisorQueryV2,
    matching_rows: tuple[Mapping[str, JsonValue], ...],
    schema_version: str,
    route: str,
    plan_sha256: str,
) -> dict[str, JsonValue]:
    """Construct one parent-GPU-zero authority from raw management-query refs."""

    if stage not in {"BEFORE_PREFLIGHT", "BEFORE_COLD"}:
        raise ValueError("supervisor zero stage differs")
    if gpu_inventory_query.argv != _DIAG2_GPU_INVENTORY_ARGV:
        raise ValueError("GPU inventory argv differs")
    if compute_apps_query.argv != _DIAG2_COMPUTE_APPS_ARGV:
        raise ValueError("compute-apps argv differs")
    rows: list[dict[str, JsonValue]] = []
    for index, raw_row in enumerate(matching_rows):
        row = dict(raw_row)
        _exact_keys(
            row,
            frozenset({"pid", "gpu_uuid", "used_memory_mib"}),
            f"matching_rows[{index}]",
        )
        rows.append(
            {
                "pid": _integer(row["pid"], f"matching_rows[{index}].pid", minimum=1),
                "gpu_uuid": _string(
                    row["gpu_uuid"], f"matching_rows[{index}].gpu_uuid"
                ),
                "used_memory_mib": _integer(
                    row["used_memory_mib"], f"matching_rows[{index}].used_memory_mib"
                ),
            }
        )
    successful = (
        gpu_inventory_query.launched
        and not gpu_inventory_query.timed_out
        and gpu_inventory_query.returncode == 0
        and compute_apps_query.launched
        and not compute_apps_query.timed_out
        and compute_apps_query.returncode == 0
    )
    if not successful and rows:
        raise ValueError("failed supervisor query cannot claim matching rows")
    return {
        "schema_version": schema_version,
        "route": route,
        "plan_sha256": plan_sha256,
        "stage": stage,
        "captured_at_monotonic_ns": _integer(
            captured_at_monotonic_ns, "capture monotonic ns"
        ),
        "captured_at_unix_ns": _integer(captured_at_unix_ns, "capture unix ns"),
        "supervisor_pid": _integer(supervisor_pid, "supervisor PID", minimum=1),
        "supervisor_start_ticks": _integer(
            supervisor_start_ticks, "supervisor start ticks", minimum=1
        ),
        "gpu_uuid": _string(gpu_uuid, "GPU UUID"),
        "visible_device": _string(visible_device, "visible device"),
        "gpu_inventory_query": _diag2_query_payload(gpu_inventory_query),
        "compute_apps_query": _diag2_query_payload(compute_apps_query),
        "matching_rows": rows,
    }


def build_diag2_supervisor_zero_payload(
    *,
    stage: str,
    captured_at_monotonic_ns: int,
    captured_at_unix_ns: int,
    supervisor_pid: int,
    supervisor_start_ticks: int,
    gpu_uuid: str,
    visible_device: str,
    gpu_inventory_query: SupervisorQueryV2,
    compute_apps_query: SupervisorQueryV2,
    matching_rows: tuple[Mapping[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    return _build_supervisor_zero_payload(
        stage=stage,
        captured_at_monotonic_ns=captured_at_monotonic_ns,
        captured_at_unix_ns=captured_at_unix_ns,
        supervisor_pid=supervisor_pid,
        supervisor_start_ticks=supervisor_start_ticks,
        gpu_uuid=gpu_uuid,
        visible_device=visible_device,
        gpu_inventory_query=gpu_inventory_query,
        compute_apps_query=compute_apps_query,
        matching_rows=matching_rows,
        schema_version=DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION,
        route=DIAG2_ROUTE,
        plan_sha256=DIAG2_PLAN_SHA256,
    )


def build_diag5_supervisor_zero_payload(
    *,
    stage: str,
    captured_at_monotonic_ns: int,
    captured_at_unix_ns: int,
    supervisor_pid: int,
    supervisor_start_ticks: int,
    gpu_uuid: str,
    visible_device: str,
    gpu_inventory_query: SupervisorQueryV2,
    compute_apps_query: SupervisorQueryV2,
    matching_rows: tuple[Mapping[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    return _build_supervisor_zero_payload(
        stage=stage,
        captured_at_monotonic_ns=captured_at_monotonic_ns,
        captured_at_unix_ns=captured_at_unix_ns,
        supervisor_pid=supervisor_pid,
        supervisor_start_ticks=supervisor_start_ticks,
        gpu_uuid=gpu_uuid,
        visible_device=visible_device,
        gpu_inventory_query=gpu_inventory_query,
        compute_apps_query=compute_apps_query,
        matching_rows=matching_rows,
        schema_version=DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
    )


def _parse_diag2_supervisor_query(
    value: JsonValue, *, context: str
) -> SupervisorQueryV2:
    payload = _mapping(value, context)
    _exact_keys(
        payload,
        frozenset(
            {
                "argv",
                "query_executable_sha256",
                "launched",
                "timed_out",
                "returncode",
                "stdout",
                "stderr",
            }
        ),
        context,
    )
    returncode_value = payload["returncode"]
    returncode = (
        None
        if returncode_value is None
        else _integer(returncode_value, f"{context}.returncode", minimum=-2147483648)
    )
    query = SupervisorQueryV2(
        tuple(
            _string(item, f"{context}.argv")
            for item in _array(payload["argv"], f"{context}.argv")
        ),
        _sha256(payload["query_executable_sha256"], f"{context}.executable SHA"),
        _boolean(payload["launched"], f"{context}.launched"),
        _boolean(payload["timed_out"], f"{context}.timed_out"),
        returncode,
        _artifact_ref(payload["stdout"], f"{context}.stdout"),
        _artifact_ref(payload["stderr"], f"{context}.stderr"),
    )
    _diag2_query_payload(query)
    return query


def _parse_diag2_inventory_rows(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 2 or not columns[0]:
            raise ValueError(f"GPU inventory row {index} is malformed")
        try:
            memory_mib = int(columns[1])
        except ValueError as error:
            raise ValueError(f"GPU inventory row {index} memory is invalid") from error
        if memory_mib <= 0:
            raise ValueError(f"GPU inventory row {index} memory is invalid")
        rows.append((columns[0], memory_mib))
    return rows


def _parse_diag2_compute_rows(path: Path) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 3:
            raise ValueError(f"compute-apps row {index} is malformed")
        try:
            pid = int(columns[0])
            used_memory_mib = int(columns[2])
        except ValueError as error:
            raise ValueError(f"compute-apps row {index} is invalid") from error
        if pid <= 0 or used_memory_mib < 0 or not columns[1]:
            raise ValueError(f"compute-apps row {index} is invalid")
        rows.append((pid, columns[1], used_memory_mib))
    return rows


def _validate_supervisor_zero_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
    expected_stage: str,
    allow_failure: bool = False,
    schema_version: str,
    route: str,
    plan_sha256: str,
) -> dict[str, JsonValue]:
    """Validate both raw query bindings and independently reject a parent GPU PID."""

    payload = _mapping(value, "DIAG2 supervisor zero")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "stage",
                "captured_at_monotonic_ns",
                "captured_at_unix_ns",
                "supervisor_pid",
                "supervisor_start_ticks",
                "gpu_uuid",
                "visible_device",
                "gpu_inventory_query",
                "compute_apps_query",
                "matching_rows",
            }
        ),
        "DIAG2 supervisor zero",
    )
    inventory = _parse_diag2_supervisor_query(
        payload["gpu_inventory_query"], context="gpu_inventory_query"
    )
    compute = _parse_diag2_supervisor_query(
        payload["compute_apps_query"], context="compute_apps_query"
    )
    stage_slug = (
        "before-preflight" if expected_stage == "BEFORE_PREFLIGHT" else "before-cold"
    )
    if expected_stage not in {"BEFORE_PREFLIGHT", "BEFORE_COLD"}:
        raise ValueError("expected supervisor-zero stage differs")
    expected_streams = (
        (
            inventory.stdout,
            f"supervisor/{stage_slug}-gpu-inventory.stdout.bin",
            "raw-supervisor-gpu-inventory-stdout-v1",
        ),
        (
            inventory.stderr,
            f"supervisor/{stage_slug}-gpu-inventory.stderr.bin",
            "raw-supervisor-gpu-inventory-stderr-v1",
        ),
        (
            compute.stdout,
            f"supervisor/{stage_slug}-compute-apps.stdout.bin",
            "raw-supervisor-compute-apps-stdout-v1",
        ),
        (
            compute.stderr,
            f"supervisor/{stage_slug}-compute-apps.stderr.bin",
            "raw-supervisor-compute-apps-stderr-v1",
        ),
    )
    if any(
        reference.relative_path != relative or reference.schema_version != schema
        for reference, relative, schema in expected_streams
    ):
        raise ValueError("supervisor raw query binding differs")
    if (
        inventory.query_executable_sha256 != compute.query_executable_sha256
        or payload["gpu_uuid"] != GPU_UUID
        or payload["visible_device"] != GPU_UUID
    ):
        raise ValueError("supervisor GPU/query identity differs")
    for reference in (
        inventory.stdout,
        inventory.stderr,
        compute.stdout,
        compute.stderr,
    ):
        _resolve_artifact(artifact_root, reference)
    inventory_success = (
        inventory.launched and not inventory.timed_out and inventory.returncode == 0
    )
    compute_success = (
        compute.launched and not compute.timed_out and compute.returncode == 0
    )
    if inventory_success:
        try:
            inventory_rows = _parse_diag2_inventory_rows(
                _resolve_artifact(artifact_root, inventory.stdout)
            )
            frozen_uuid = _string(payload["gpu_uuid"], "GPU UUID")
            frozen_rows = [row for row in inventory_rows if row[0] == frozen_uuid]
            if len(frozen_rows) != 1:
                raise ValueError(
                    "GPU inventory does not identify the frozen UUID exactly once"
                )
        except (UnicodeDecodeError, ValueError):
            if not allow_failure:
                raise
            inventory_success = False
    elif not allow_failure:
        raise ValueError("GPU inventory query did not succeed")
    derived_rows: list[dict[str, JsonValue]] = []
    if compute_success:
        supervisor_pid = _integer(
            payload["supervisor_pid"], "supervisor PID", minimum=1
        )
        frozen_uuid = _string(payload["gpu_uuid"], "GPU UUID")
        try:
            compute_rows = _parse_diag2_compute_rows(
                _resolve_artifact(artifact_root, compute.stdout)
            )
        except (UnicodeDecodeError, ValueError):
            if not allow_failure:
                raise
            compute_success = False
        else:
            for pid, row_uuid, used_memory_mib in compute_rows:
                if pid == supervisor_pid and row_uuid == frozen_uuid:
                    derived_rows.append(
                        {
                            "pid": pid,
                            "gpu_uuid": row_uuid,
                            "used_memory_mib": used_memory_mib,
                        }
                    )
    elif not allow_failure:
        raise ValueError("compute-apps query did not succeed")
    claimed_rows = tuple(
        _mapping(row, f"matching_rows[{index}]")
        for index, row in enumerate(_array(payload["matching_rows"], "matching_rows"))
    )
    rows: tuple[Mapping[str, JsonValue], ...] = (
        tuple(derived_rows) if inventory_success and compute_success else ()
    )
    if claimed_rows != rows:
        raise ValueError("matching rows differ from raw compute-app output")
    rebuilt = _build_supervisor_zero_payload(
        stage=_string(payload["stage"], "supervisor zero stage"),
        captured_at_monotonic_ns=_integer(
            payload["captured_at_monotonic_ns"], "capture monotonic ns"
        ),
        captured_at_unix_ns=_integer(payload["captured_at_unix_ns"], "capture unix ns"),
        supervisor_pid=_integer(payload["supervisor_pid"], "supervisor PID", minimum=1),
        supervisor_start_ticks=_integer(
            payload["supervisor_start_ticks"], "supervisor start ticks", minimum=1
        ),
        gpu_uuid=_string(payload["gpu_uuid"], "GPU UUID"),
        visible_device=_string(payload["visible_device"], "visible device"),
        gpu_inventory_query=inventory,
        compute_apps_query=compute,
        matching_rows=rows,
        schema_version=schema_version,
        route=route,
        plan_sha256=plan_sha256,
    )
    if payload != rebuilt or rebuilt["stage"] != expected_stage:
        raise ValueError("DIAG2 supervisor zero claims differ from raw authority")
    if rebuilt["matching_rows"] and not allow_failure:
        raise ValueError("supervisor PID is present on the frozen GPU")
    return rebuilt


def validate_diag2_supervisor_zero_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
    expected_stage: str,
    allow_failure: bool = False,
) -> dict[str, JsonValue]:
    return _validate_supervisor_zero_payload(
        value,
        artifact_root=artifact_root,
        expected_stage=expected_stage,
        allow_failure=allow_failure,
        schema_version=DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION,
        route=DIAG2_ROUTE,
        plan_sha256=DIAG2_PLAN_SHA256,
    )


def validate_diag5_supervisor_zero_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
    expected_stage: str,
    allow_failure: bool = False,
) -> dict[str, JsonValue]:
    return _validate_supervisor_zero_payload(
        value,
        artifact_root=artifact_root,
        expected_stage=expected_stage,
        allow_failure=allow_failure,
        schema_version=DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
    )


def _validate_diag2_supervisor_sequence(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlot],
    *,
    failure: StructuredFailureV2 | None,
) -> None:
    observations: dict[str, dict[str, JsonValue]] = {}
    for slot_name, stage in (
        ("supervisor_before_preflight", "BEFORE_PREFLIGHT"),
        ("supervisor_before_cold", "BEFORE_COLD"),
    ):
        reference = evidence_slots[slot_name].artifact
        if reference is not None:
            matching_failure_stage = {
                "BEFORE_PREFLIGHT": FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE,
                "BEFORE_COLD": FailureStageV2.GPU_ZERO_BEFORE_COLD_FAILURE,
            }[stage]
            observations[stage] = validate_diag2_supervisor_zero_payload(
                _load_ref_json(artifact_root, reference, slot_name),
                artifact_root=artifact_root,
                expected_stage=stage,
                allow_failure=(
                    failure is not None and failure.stage is matching_failure_stage
                ),
            )
    child_identities: dict[str, tuple[int, int]] = {}
    child_intervals: dict[str, tuple[int, int]] = {}
    for mode in ("preflight", "cold"):
        reference = evidence_slots[f"{mode}_process"].artifact
        if reference is None:
            continue
        process = _mapping(
            _load_ref_json(artifact_root, reference, f"{mode} process"),
            f"{mode} process",
        )
        _exact_keys(
            process,
            frozenset(
                {
                    "schema_version",
                    "monitor_failure_kind",
                    "child_pid",
                    "child_start_time_ticks",
                    "argv",
                    "stdout",
                    "stderr",
                    "process_seconds",
                    "process_diagnostics",
                    "pre_source_identity",
                    "post_source_identity",
                    "process_started_monotonic_ns",
                    "process_stopped_monotonic_ns",
                }
            ),
            f"{mode} process",
        )
        if process["schema_version"] != DIAG2_PROCESS_SCHEMA_VERSION:
            raise ValueError(f"{mode} process schema differs")
        monitor_failure_kind = _string(
            process["monitor_failure_kind"], f"{mode} monitor failure kind"
        )
        if monitor_failure_kind not in {"NONE", "BINDING", "FINALIZATION"}:
            raise ValueError(f"{mode} monitor failure kind differs")
        process_started = _integer(
            process["process_started_monotonic_ns"], f"{mode} process start"
        )
        process_stopped = _integer(
            process["process_stopped_monotonic_ns"], f"{mode} process stop"
        )
        if process_started >= process_stopped:
            raise ValueError(f"{mode} process parent-monotonic interval differs")
        child_pid = _integer(process.get("child_pid"), f"{mode} child PID", minimum=1)
        child_start_ticks = _integer(
            process.get("child_start_time_ticks"),
            f"{mode} child start ticks",
            minimum=0,
        )
        if (monitor_failure_kind == "BINDING") != (child_start_ticks == 0):
            raise ValueError(f"{mode} monitor binding sentinel differs")
        if child_start_ticks > 0:
            child_identities[mode] = (child_pid, child_start_ticks)
        child_intervals[mode] = (process_started, process_stopped)
        terminal_ref = evidence_slots[f"{mode}_terminal"].artifact
        if terminal_ref is None:
            raise ValueError(f"{mode} process omits child terminal")
        terminal = _mapping(
            _load_ref_json(artifact_root, terminal_ref, f"{mode} terminal"),
            f"{mode} terminal",
        )
        if terminal["schema_version"] != DIAG2_CHILD_TERMINAL_SCHEMA_VERSION:
            raise ValueError(f"{mode} terminal schema differs")
        _exact_keys(
            terminal,
            frozenset(
                {
                    "schema_version",
                    "terminal_status",
                    "failure_reasons",
                    "monitor_failure_kind",
                }
            ),
            f"{mode} terminal",
        )
        if terminal.get("monitor_failure_kind") != monitor_failure_kind:
            raise ValueError(f"{mode} terminal/process monitor kind differs")
        terminal_status = _string(
            terminal["terminal_status"], f"{mode} terminal status"
        )
        if monitor_failure_kind == "BINDING" and terminal_status != "MONITOR_FAILURE":
            raise ValueError(f"{mode} binding terminal status differs")
        if (
            monitor_failure_kind == "FINALIZATION"
            and terminal_status == "MONITOR_FAILURE"
        ):
            raise ValueError(f"{mode} finalization terminal status differs")
        memory_present = evidence_slots[f"{mode}_memory"].artifact is not None
        samples_present = evidence_slots[f"{mode}_memory_samples"].artifact is not None
        if (
            memory_present != samples_present
            or (monitor_failure_kind == "NONE") != memory_present
        ):
            raise ValueError(f"{mode} monitor evidence pairing differs")
    for observation in observations.values():
        supervisor_pid = _integer(
            observation["supervisor_pid"], "supervisor PID", minimum=1
        )
        if any(identity[0] == supervisor_pid for identity in child_identities.values()):
            raise ValueError("supervisor PID aliases a supervised child PID")
    if (
        len(child_identities) == 2
        and len({identity[0] for identity in child_identities.values()}) != 2
    ):
        raise ValueError("preflight and cold child PIDs are not distinct")
    before = observations.get("BEFORE_PREFLIGHT")
    preflight_interval = child_intervals.get("preflight")
    if (
        before is not None
        and preflight_interval is not None
        and not (
            _integer(before["captured_at_monotonic_ns"], "before monotonic")
            < preflight_interval[0]
            < preflight_interval[1]
        )
    ):
        raise ValueError("preflight child interval precedes its GPU-zero gate")
    cold = observations.get("BEFORE_COLD")
    cold_interval = child_intervals.get("cold")
    if (
        cold is not None
        and cold_interval is not None
        and not (
            _integer(cold["captured_at_monotonic_ns"], "cold monotonic")
            < cold_interval[0]
            < cold_interval[1]
        )
    ):
        raise ValueError("cold child interval precedes the before-cold gate")
    if frozenset(observations) == {"BEFORE_PREFLIGHT", "BEFORE_COLD"}:
        before = observations["BEFORE_PREFLIGHT"]
        cold = observations["BEFORE_COLD"]
        for key in ("supervisor_pid", "supervisor_start_ticks", "gpu_uuid"):
            if before[key] != cold[key]:
                raise ValueError(f"supervisor observations disagree on {key}")
        before_inventory = _mapping(
            before["gpu_inventory_query"], "before-preflight inventory query"
        )
        cold_inventory = _mapping(
            cold["gpu_inventory_query"], "before-cold inventory query"
        )
        if (
            before_inventory["query_executable_sha256"]
            != cold_inventory["query_executable_sha256"]
        ):
            raise ValueError("supervisor observations disagree on query executable")
        if not (
            _integer(before["captured_at_monotonic_ns"], "before monotonic")
            < _integer(cold["captured_at_monotonic_ns"], "cold monotonic")
            and _integer(before["captured_at_unix_ns"], "before unix")
            < _integer(cold["captured_at_unix_ns"], "cold unix")
        ):
            raise ValueError("supervisor GPU-zero observation order differs")
        preflight_interval = child_intervals.get("preflight")
        if preflight_interval is None or not (
            _integer(before["captured_at_monotonic_ns"], "before monotonic")
            < preflight_interval[0]
            < preflight_interval[1]
            < _integer(cold["captured_at_monotonic_ns"], "cold monotonic")
        ):
            raise ValueError("preflight child interval is outside GPU-zero gates")


def _validate_diag5_supervisor_sequence(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV5],
    *,
    failure: StructuredFailureV5,
    expected_gpu_uuid: str,
) -> None:
    observations: dict[str, dict[str, JsonValue]] = {}
    for slot_name, stage in (
        ("supervisor_before_preflight", "BEFORE_PREFLIGHT"),
        ("supervisor_before_cold", "BEFORE_COLD"),
    ):
        reference = evidence_slots[slot_name].artifact
        if reference is not None:
            observations[stage] = validate_diag5_supervisor_zero_payload(
                _load_ref_json(artifact_root, reference, slot_name),
                artifact_root=artifact_root,
                expected_stage=stage,
                allow_failure=(
                    failure.reason
                    in {
                        FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID,
                        FailureReasonCodeV5.SUPERVISOR_GPU_NONZERO,
                    }
                ),
            )
    child_identities: dict[str, tuple[int, int]] = {}
    child_intervals: dict[str, tuple[int, int]] = {}
    for mode in ("preflight", "cold"):
        process_reference = evidence_slots[f"{mode}_process"].artifact
        if process_reference is None:
            continue
        refs = {name: slot.artifact for name, slot in evidence_slots.items()}
        terminal, process, monitor_kind, _returncode = _diag2_child_documents(
            artifact_root,
            refs,
            mode=mode,
            child_terminal_schema_version=DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
            process_schema_version=DIAG5_PROCESS_SCHEMA_VERSION,
        )
        process_started = _integer(
            process["process_started_monotonic_ns"], f"DIAG5 {mode} process start"
        )
        process_stopped = _integer(
            process["process_stopped_monotonic_ns"], f"DIAG5 {mode} process stop"
        )
        if process_started >= process_stopped:
            raise ValueError(f"DIAG5 {mode} process interval differs")
        child_pid = _integer(process["child_pid"], f"DIAG5 {mode} child PID", minimum=1)
        child_start_ticks = _integer(
            process["child_start_time_ticks"],
            f"DIAG5 {mode} child start ticks",
        )
        if (monitor_kind == "BINDING") != (child_start_ticks == 0):
            raise ValueError(f"DIAG5 {mode} monitor binding sentinel differs")
        if child_start_ticks > 0:
            child_identities[mode] = (child_pid, child_start_ticks)
        child_intervals[mode] = (process_started, process_stopped)
        terminal_status = _string(
            terminal["terminal_status"], f"DIAG5 {mode} terminal status"
        )
        if monitor_kind == "BINDING" and terminal_status != "MONITOR_FAILURE":
            raise ValueError(f"DIAG5 {mode} binding terminal status differs")
        if monitor_kind == "FINALIZATION" and terminal_status == "MONITOR_FAILURE":
            raise ValueError(f"DIAG5 {mode} finalization terminal status differs")
        memory_present = evidence_slots[f"{mode}_memory"].artifact is not None
        samples_present = evidence_slots[f"{mode}_memory_samples"].artifact is not None
        if memory_present != samples_present or (
            monitor_kind == "NONE"
            and failure.reason
            not in {
                FailureReasonCodeV5.PREFLIGHT_TIMEOUT,
                FailureReasonCodeV5.PREFLIGHT_EXIT_NONZERO,
                FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
                FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
                FailureReasonCodeV5.COLD_TIMEOUT,
                FailureReasonCodeV5.COLD_EXIT_NONZERO,
                FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
                FailureReasonCodeV5.COLD_PRODUCER_INVALID,
            }
            and not memory_present
        ):
            raise ValueError(f"DIAG5 {mode} monitor evidence pairing differs")
    supervisor_identity: tuple[int, int] | None = None
    query_sha256: str | None = None
    for observation in observations.values():
        if observation["gpu_uuid"] != expected_gpu_uuid:
            raise ValueError("DIAG5 supervisor GPU UUID differs from authority")
        observed_supervisor = (
            _integer(observation["supervisor_pid"], "DIAG5 supervisor PID", minimum=1),
            _integer(
                observation["supervisor_start_ticks"],
                "DIAG5 supervisor start ticks",
                minimum=1,
            ),
        )
        if supervisor_identity is None:
            supervisor_identity = observed_supervisor
        elif supervisor_identity != observed_supervisor:
            raise ValueError("DIAG5 supervisor observations disagree on identity")
        for query_name in ("gpu_inventory_query", "compute_apps_query"):
            query = _mapping(observation[query_name], f"DIAG5 {query_name}")
            observed_query = _sha256(
                query["query_executable_sha256"], "DIAG5 query executable SHA"
            )
            if query_sha256 is None:
                query_sha256 = observed_query
            elif query_sha256 != observed_query:
                raise ValueError("DIAG5 supervisor query executable differs")
    if supervisor_identity is not None and any(
        identity[0] == supervisor_identity[0] for identity in child_identities.values()
    ):
        raise ValueError("DIAG5 supervisor PID aliases a child PID")
    if (
        len(child_identities) == 2
        and len({identity[0] for identity in child_identities.values()}) != 2
    ):
        raise ValueError("DIAG5 child PIDs are not distinct")
    before = observations.get("BEFORE_PREFLIGHT")
    preflight = child_intervals.get("preflight")
    cold_observation = observations.get("BEFORE_COLD")
    cold = child_intervals.get("cold")
    if (
        before is not None
        and preflight is not None
        and not (
            _integer(before["captured_at_monotonic_ns"], "DIAG5 before preflight")
            < preflight[0]
            < preflight[1]
        )
    ):
        raise ValueError("DIAG5 preflight chronology differs")
    if (
        cold_observation is not None
        and cold is not None
        and not (
            _integer(cold_observation["captured_at_monotonic_ns"], "DIAG5 before cold")
            < cold[0]
            < cold[1]
        )
    ):
        raise ValueError("DIAG5 cold chronology differs")
    if before is not None and cold_observation is not None:
        before_time = _integer(
            before["captured_at_monotonic_ns"], "DIAG5 before preflight"
        )
        cold_time = _integer(
            cold_observation["captured_at_monotonic_ns"], "DIAG5 before cold"
        )
        if (
            before_time >= cold_time
            or (
                _integer(before["captured_at_unix_ns"], "DIAG5 before preflight unix")
                >= _integer(
                    cold_observation["captured_at_unix_ns"], "DIAG5 before cold unix"
                )
            )
            or (
                preflight is not None
                and not (before_time < preflight[0] < preflight[1] < cold_time)
            )
        ):
            raise ValueError("DIAG5 supervisor/child chronology differs")


def build_diag2_supervisor_terminal_payload(
    *,
    disposition: str,
    failure: StructuredFailureV2 | None,
    launched_children: tuple[str, ...],
    policy_authority_produced: bool,
    preflight_authorized: bool,
    cold_authorized: bool,
    staging_root: Path,
    final_root: Path,
    nonce: str,
    algorithm_route_selection: str,
) -> dict[str, JsonValue]:
    """Construct the sole parent terminal with exact complete/incomplete invariants."""

    if disposition not in {"COMPLETE", "INCOMPLETE"}:
        raise ValueError("supervisor terminal disposition differs")
    if launched_children not in ((), ("preflight",), ("preflight", "cold")):
        raise ValueError("supervisor terminal child sequence differs")
    if len(nonce) != 32 or any(character not in _LOWER_HEX for character in nonce):
        raise ValueError("supervisor terminal nonce must be 32 lower-case hex digits")
    staging = staging_root.resolve(strict=False)
    final = final_root.resolve(strict=False)
    if (
        staging.parent != final.parent
        or staging.name != f"{final.name}.partial-{nonce}"
    ):
        raise ValueError("supervisor publication roots differ from staging layout")
    routes = {
        NextRoute.RETRY_MODEL_REUSE.value,
        NextRoute.RADIUS_RETRACTION.value,
        NextRoute.CONDITIONING_MODEL_CHANGE.value,
    }
    if disposition == "COMPLETE":
        if (
            failure is not None
            or launched_children != ("preflight", "cold")
            or not policy_authority_produced
            or not preflight_authorized
            or not cold_authorized
            or algorithm_route_selection not in routes
        ):
            raise ValueError("complete supervisor terminal invariants differ")
    elif failure is None or algorithm_route_selection != "NOT_PRODUCED":
        raise ValueError("incomplete supervisor terminal invariants differ")
    if (
        disposition == "INCOMPLETE"
        and failure is not None
        and failure.reason in _DIAG2_POSTLAUNCH_SETUP_SLOT
        and failure.stage
        in {
            FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            FailureStageV2.COLD_SOURCE_FAILURE,
        }
    ):
        expected = (
            (("preflight",), True, False, False)
            if failure.stage is FailureStageV2.PREFLIGHT_SOURCE_FAILURE
            else (("preflight", "cold"), True, True, True)
        )
        if (
            launched_children,
            policy_authority_produced,
            preflight_authorized,
            cold_authorized,
        ) != expected:
            raise ValueError("post-launch setup-drift terminal invariants differ")
    return {
        "schema_version": DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
        "route": DIAG2_ROUTE,
        "plan_sha256": DIAG2_PLAN_SHA256,
        "disposition": disposition,
        "failure_stage": None if failure is None else failure.stage.value,
        "failure_reason": (
            None
            if failure is None
            else {"code": failure.reason.value, "detail_sha256": failure.detail_sha256}
        ),
        "launched_children": list(launched_children),
        "policy_authority_produced": policy_authority_produced,
        "preflight_authorized": preflight_authorized,
        "cold_authorized": cold_authorized,
        "publication": {
            "staging_root": str(staging),
            "final_root": str(final),
            "nonce": nonce,
        },
        "engineering_campaign_receipt_produced": False,
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
        "algorithm_route_selection": algorithm_route_selection,
    }


def _parse_diag2_supervisor_terminal(
    value: JsonValue,
) -> tuple[dict[str, JsonValue], StructuredFailureV2 | None]:
    payload = _mapping(value, "DIAG2 supervisor terminal")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "disposition",
                "failure_stage",
                "failure_reason",
                "launched_children",
                "policy_authority_produced",
                "preflight_authorized",
                "cold_authorized",
                "publication",
                "engineering_campaign_receipt_produced",
                "promotion_authorized",
                "formal_comparison",
                "algorithm_route_selection",
            }
        ),
        "DIAG2 supervisor terminal",
    )
    failure: StructuredFailureV2 | None
    if payload["failure_stage"] is None and payload["failure_reason"] is None:
        failure = None
    elif payload["failure_stage"] is None or payload["failure_reason"] is None:
        raise ValueError("supervisor terminal failure fields differ")
    else:
        failure = _parse_diag2_failure(
            {"stage": payload["failure_stage"], "reason": payload["failure_reason"]}
        )
    publication = _mapping(payload["publication"], "supervisor publication")
    _exact_keys(
        publication,
        frozenset({"staging_root", "final_root", "nonce"}),
        "supervisor publication",
    )
    rebuilt = build_diag2_supervisor_terminal_payload(
        disposition=_string(payload["disposition"], "supervisor disposition"),
        failure=failure,
        launched_children=tuple(
            _string(item, "launched child")
            for item in _array(payload["launched_children"], "launched children")
        ),
        policy_authority_produced=_boolean(
            payload["policy_authority_produced"], "policy produced"
        ),
        preflight_authorized=_boolean(
            payload["preflight_authorized"], "preflight authorized"
        ),
        cold_authorized=_boolean(payload["cold_authorized"], "cold authorized"),
        staging_root=Path(_string(publication["staging_root"], "staging root")),
        final_root=Path(_string(publication["final_root"], "final root")),
        nonce=_string(publication["nonce"], "publication nonce"),
        algorithm_route_selection=_string(
            payload["algorithm_route_selection"], "algorithm route"
        ),
    )
    if payload != rebuilt:
        raise ValueError("supervisor terminal differs from reconstructed payload")
    return rebuilt, failure


def validate_diag2_producer_payload(
    value: JsonValue, *, mode: str
) -> dict[str, JsonValue]:
    """Validate the minimum canonical producer object before it may be published."""

    payload = _mapping(value, f"{mode} producer")
    status = _string(payload.get("execution_status"), f"{mode} producer status")
    compile_failure = status in {"COMPILE_FAILURE", "COMPILE_OOM"}
    if mode == "preflight":
        success_keys = frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "mode",
                "execution_status",
                "policy_sha256",
                "policy_evidence",
                "phase_schema_sha256",
                "state_size",
                "equality_size",
                "residual_size",
                "campaign_authorized",
                "solver_dispatched",
                "finalizer_called",
                "endpoint_audit_called",
                "python_callbacks",
                "runtime",
                "runtime_evidence",
                "timing",
                "failure_reasons",
            }
        )
        failure_keys = frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "mode",
                "execution_status",
                "campaign_authorized",
                "solver_dispatched",
                "finalizer_called",
                "endpoint_audit_called",
                "runtime",
                "runtime_evidence",
                "timing",
                "failure_reasons",
            }
        )
        keys = failure_keys if compile_failure else success_keys
        expected_schema = "single-stage-neq-gntr1-preflight-worker-v1"
    elif mode == "cold":
        success_keys = frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "execution_status",
                "runtime",
                "runtime_evidence",
                "policy_sha256",
                "phase_schema_sha256",
                "history_evidence",
                "terminal_numerical_evidence",
                "policy_evidence",
                "raw_trace_evidence",
                "trace_intervals_evidence",
                "timestamps_ns",
                "transfer_audit",
                "endpoint_audit_called",
                "campaign_authorized",
                "failure_reasons",
            }
        )
        failure_keys = frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "execution_status",
                "runtime",
                "runtime_evidence",
                "timing",
                "failure_reasons",
            }
        )
        keys = failure_keys if compile_failure else success_keys
        expected_schema = f"{SCHEMA_VERSION}-producer"
    else:
        raise ValueError("producer mode must be preflight or cold")
    _exact_keys(payload, keys, f"{mode} producer")
    if (
        payload["schema_version"] != expected_schema
        or payload["route"] != DIAG2_ROUTE
        or payload["plan_sha256"] != DIAG2_PLAN_SHA256
    ):
        raise ValueError(f"{mode} producer identity differs")
    if compile_failure:
        runtime = _mapping(payload["runtime"], f"{mode} compile runtime")
        _exact_keys(
            runtime,
            frozenset(
                {
                    "backend",
                    "device",
                    "device_uuid",
                    "jax",
                    "jax_enable_x64",
                    "jaxlib",
                    "python",
                }
            ),
            f"{mode} compile runtime",
        )
        if runtime["backend"] != "gpu" or runtime["device_uuid"] != GPU_UUID:
            raise ValueError(f"{mode} compile runtime differs from frozen GPU")
        _artifact_ref(payload["runtime_evidence"], f"{mode} runtime evidence")
        timing = _mapping(payload["timing"], f"{mode} compile timing")
        _exact_keys(
            timing,
            frozenset(
                {
                    "compile_started_ns",
                    "compile_completed_ns",
                    "process_seconds_before_serialization",
                }
            ),
            f"{mode} compile timing",
        )
        started = _integer(timing["compile_started_ns"], "compile started")
        completed = _integer(timing["compile_completed_ns"], "compile completed")
        seconds = _number(
            timing["process_seconds_before_serialization"], "compile process seconds"
        )
        if started < 0 or completed < started or seconds < 0.0:
            raise ValueError(f"{mode} compile timing is invalid")
        reasons = _array(payload["failure_reasons"], f"{mode} failure reasons")
        if not reasons or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError(f"{mode} compile failure reasons are invalid")
        if mode == "preflight" and any(
            payload[key] is not False
            for key in (
                "campaign_authorized",
                "solver_dispatched",
                "finalizer_called",
                "endpoint_audit_called",
            )
        ):
            raise ValueError("preflight compile failure claims dispatched work")
    elif (mode == "preflight" and status != "SUCCESS") or (
        mode == "cold" and status != "COMPLETE"
    ):
        raise ValueError(f"{mode} producer execution status differs")
    else:
        reference_fields = (
            ("runtime_evidence", "policy_evidence")
            if mode == "preflight"
            else (
                "runtime_evidence",
                "history_evidence",
                "terminal_numerical_evidence",
                "policy_evidence",
                "raw_trace_evidence",
                "trace_intervals_evidence",
            )
        )
        for field in reference_fields:
            _artifact_ref(payload[field], f"{mode} producer.{field}")
    return payload


def validate_diag3_producer_payload(
    value: JsonValue, *, mode: str
) -> dict[str, JsonValue]:
    """Validate the additive atomic-result producer without widening v2."""

    payload = _mapping(value, f"{mode} successor producer")
    if (
        mode != "cold"
        or payload.get("schema_version") != DIAG3_COLD_RESULT_SCHEMA_VERSION
    ):
        return validate_diag2_producer_payload(payload, mode=mode)
    status = _string(payload.get("execution_status"), "cold successor status")
    if status not in {"COMPLETE", "TRACE_NORMALIZATION_FAILED"}:
        raise ValueError("cold successor execution status differs")
    normalized = dict(payload)
    normalized["schema_version"] = f"{SCHEMA_VERSION}-producer"
    normalized["execution_status"] = "COMPLETE"
    normalized["failure_reasons"] = []
    if payload.get("trace_intervals_evidence") is None:
        normalized["trace_intervals_evidence"] = payload.get("raw_trace_evidence")
    validate_diag2_producer_payload(normalized, mode="cold")
    expected_paths = {
        "runtime_evidence": "cold/runtime-evidence.json",
        "policy_evidence": "cold/policy.json",
        "history_evidence": f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/history.json",
        "terminal_numerical_evidence": (
            f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/terminal-numerical.json"
        ),
    }
    for field, expected in expected_paths.items():
        if _artifact_ref(payload[field], field).relative_path != expected:
            raise ValueError(f"cold successor {field} path differs")
    raw_trace = _artifact_ref(payload["raw_trace_evidence"], "raw trace evidence")
    if not _diag3_trace_path(raw_trace.relative_path):
        raise ValueError("cold successor raw trace path differs")
    intervals_value = payload["trace_intervals_evidence"]
    reasons = _array(payload["failure_reasons"], "cold successor failure reasons")
    if status == "COMPLETE":
        intervals = _artifact_ref(intervals_value, "trace intervals evidence")
        if (
            intervals.relative_path
            != f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/trace-intervals.json"
            or reasons
        ):
            raise ValueError("complete cold successor result differs")
    elif (
        intervals_value is not None
        or len(reasons) != 1
        or not isinstance(reasons[0], str)
        or not reasons[0].startswith("TRACE_NORMALIZATION_FAILED:")
        or len(reasons[0].removeprefix("TRACE_NORMALIZATION_FAILED:")) != 64
    ):
        raise ValueError("trace-normalization successor result differs")
    return payload


def _diag4_history_outcomes_sha256(outcomes: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(outcomes))).hexdigest()


def _diag4_subtrial_integer_vector(
    value: JsonValue | tuple[int, ...], *, context: str, minimum: int = 0
) -> tuple[int, ...]:
    raw = value if isinstance(value, tuple) else _array(value, context)
    parsed = tuple(
        _integer(item, f"{context}[{index}]", minimum=minimum)
        for index, item in enumerate(raw)
    )
    if len(parsed) != MAXIMUM_ATTEMPTS:
        raise ValueError(f"{context} vector extent differs")
    return parsed


def _diag4_subtrial_integer_matrix(
    value: JsonValue | tuple[tuple[int, ...], ...], *, context: str
) -> tuple[tuple[int, ...], ...]:
    raw_rows = value if isinstance(value, tuple) else _array(value, context)
    rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(raw_rows):
        row = (
            raw_row
            if isinstance(raw_row, tuple)
            else _array(raw_row, f"{context}[{row_index}]")
        )
        parsed = tuple(
            _integer(item, f"{context}[{row_index}][{column_index}]")
            for column_index, item in enumerate(row)
        )
        if len(parsed) != DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS:
            raise ValueError(f"{context} matrix extent differs")
        rows.append(parsed)
    if len(rows) != MAXIMUM_ATTEMPTS:
        raise ValueError(f"{context} matrix extent differs")
    return tuple(rows)


def _diag4_subtrial_float_matrix(
    value: JsonValue, *, context: str
) -> tuple[tuple[float | None, ...], ...]:
    raw_rows = _array(value, context)
    rows: list[tuple[float | None, ...]] = []
    for row_index, raw_row in enumerate(raw_rows):
        row = _array(raw_row, f"{context}[{row_index}]")
        if len(row) != DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS:
            raise ValueError(f"{context} matrix extent differs")
        parsed: list[float | None] = []
        for column_index, item in enumerate(row):
            if item is None:
                parsed.append(None)
            else:
                if isinstance(item, bool) or not isinstance(item, float):
                    raise TypeError(
                        f"{context}[{row_index}][{column_index}] must be a float"
                    )
                if not math.isfinite(item):
                    raise ValueError(
                        f"{context}[{row_index}][{column_index}] must be finite"
                    )
                parsed.append(item)
        rows.append(tuple(parsed))
    if len(rows) != MAXIMUM_ATTEMPTS:
        raise ValueError(f"{context} matrix extent differs")
    return tuple(rows)


def _diag4_subtrial_summary(
    *,
    attempts: int,
    subtrial_count: tuple[int, ...],
    subtrial_outcome: tuple[tuple[AttemptOutcome, ...], ...],
    integer_work: Mapping[str, tuple[tuple[int, ...], ...]],
) -> tuple[tuple[str, int], ...]:
    active_counts = subtrial_count[:attempts]
    total_subtrials = sum(active_counts)
    selected_outcomes = tuple(
        subtrial_outcome[index][count - 1] for index, count in enumerate(active_counts)
    )
    structural = (
        total_subtrials,
        total_subtrials - attempts,
        max(active_counts, default=0),
        active_counts.count(1),
        active_counts.count(2),
        active_counts.count(3),
        sum(
            count > 1 and outcome is AttemptOutcome.ACCEPTED
            for count, outcome in zip(active_counts, selected_outcomes, strict=True)
        ),
        sum(
            count == DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS
            and outcome is AttemptOutcome.RETRY_STEP_BOUNDS
            for count, outcome in zip(active_counts, selected_outcomes, strict=True)
        ),
    )
    work_totals = tuple(
        sum(
            sum(integer_work[field][row_index][:count])
            for row_index, count in enumerate(active_counts)
        )
        for field in DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS
    )
    values = (*structural, *work_totals)
    return tuple(zip(DIAG4_SUBTRIAL_SUMMARY_FIELDS, values, strict=True))


def _validate_diag4_subtrial_structure(
    *,
    attempts: int,
    subtrial_count: tuple[int, ...],
    selected_subtrial_index: tuple[int, ...],
    subtrial_outcome: tuple[tuple[AttemptOutcome, ...], ...],
    float_matrices: Mapping[str, tuple[tuple[float | None, ...], ...]],
    integer_work: Mapping[str, tuple[tuple[int, ...], ...]],
    history: HistoryEvidence | None,
    nonlinear_corrections: tuple[int, ...],
    individual_ratios: tuple[float | None, ...],
    path_ratios: tuple[float | None, ...],
    steihaug_solve_calls: tuple[int, ...],
) -> None:
    minimum_radius = FROZEN_GNTR_OPTIONS["minimum_trust_radius"]
    maximum_radius = FROZEN_GNTR_OPTIONS["maximum_trust_radius"]
    for row_index in range(MAXIMUM_ATTEMPTS):
        count = subtrial_count[row_index]
        selected = selected_subtrial_index[row_index]
        active_row = row_index < attempts
        if active_row:
            if not 1 <= count <= DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS:
                raise ValueError("DIAG4 active subtrial count differs")
            if selected != count - 1:
                raise ValueError("DIAG4 selected subtrial index differs")
        elif count != 0 or selected != -1:
            raise ValueError("DIAG4 inactive subtrial scalar padding differs")
        for column_index in range(DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS):
            executed = active_row and column_index < count
            outcome = subtrial_outcome[row_index][column_index]
            float_values = {
                field: float_matrices[field][row_index][column_index]
                for field in DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS
            }
            work_values = {
                field: integer_work[field][row_index][column_index]
                for field in DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS
            }
            if not executed:
                if (
                    outcome is not AttemptOutcome.INACTIVE
                    or any(value is not None for value in float_values.values())
                    or any(work_values.values())
                ):
                    raise ValueError("DIAG4 subtrial matrix padding differs")
                continue
            if outcome is AttemptOutcome.INACTIVE:
                raise ValueError("DIAG4 executed subtrial outcome is inactive")
            trust_radius = float_values["subtrial_trust_radius"]
            if (
                trust_radius is None
                or trust_radius <= 0.0
                or trust_radius < minimum_radius
                or trust_radius > maximum_radius
            ):
                raise ValueError("DIAG4 executed subtrial trust radius differs")
            for field, value in work_values.items():
                if value < 0:
                    raise ValueError(f"DIAG4 {field} contains a negative work count")
            nonlinear_count = work_values["subtrial_nonlinear_corrections"]
            if nonlinear_count > DIAG4_MAXIMUM_NONLINEAR_CORRECTIONS:
                raise ValueError("DIAG4 subtrial correction count differs")
            trial_evaluated = nonlinear_count > 0
            iterations = work_values["subtrial_steihaug_iterations"]
            hvp_evaluations = work_values["subtrial_steihaug_hvp_evaluations"]
            solve_calls = work_values["subtrial_steihaug_solve_calls"]
            if (
                solve_calls not in (0, 1)
                or iterations > FROZEN_GNTR_OPTIONS["maximum_steihaug_iterations"]
                or hvp_evaluations not in (iterations, iterations + 1)
                or (solve_calls == 0 and (iterations != 0 or hvp_evaluations != 0))
            ):
                raise ValueError("DIAG4 subtrial Steihaug work bounds differ")
            expected_work = {
                "subtrial_total_hvp_evaluations": (
                    3 + hvp_evaluations + int(trial_evaluated)
                ),
                "subtrial_joint_evaluations": (
                    1
                    + (
                        DIAG4_MAXIMUM_NONLINEAR_CORRECTIONS + nonlinear_count
                        if trial_evaluated
                        else 0
                    )
                ),
                "subtrial_joint_linearizations": 1 + nonlinear_count,
                "subtrial_joint_value_evaluations": (
                    DIAG4_MAXIMUM_NONLINEAR_CORRECTIONS if trial_evaluated else 0
                ),
                "subtrial_objective_residual_linearizations": 1,
                "subtrial_gram_factorizations": 1 + nonlinear_count,
                "subtrial_gram_solves": (
                    2 + nonlinear_count + solve_calls * (3 + hvp_evaluations)
                ),
            }
            if any(
                work_values[field] != value for field, value in expected_work.items()
            ):
                raise ValueError("DIAG4 subtrial work recurrence differs")
            required_finite = outcome not in (
                AttemptOutcome.FATAL_CURRENT_STATE,
                AttemptOutcome.FATAL_STEIHAUG,
                AttemptOutcome.FATAL_CURVATURE,
                AttemptOutcome.RETRY_NONFINITE,
                AttemptOutcome.RETRY_CORRECTION_CERTIFICATE,
            )
            computed_fields = DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS[1:]
            if required_finite and any(
                float_values[field] is None for field in computed_fields
            ):
                raise ValueError("DIAG4 subtrial floating availability differs")
            if column_index > 0:
                previous_outcome = subtrial_outcome[row_index][column_index - 1]
                previous_actual = float_matrices["subtrial_actual_reduction"][
                    row_index
                ][column_index - 1]
                previous_predicted = float_matrices["subtrial_predicted_reduction"][
                    row_index
                ][column_index - 1]
                previous_radius = float_matrices["subtrial_trust_radius"][row_index][
                    column_index - 1
                ]
                if previous_radius is None:
                    raise AssertionError("executed subtrial radius narrowing failed")
                expected_radius = min(
                    max(0.25 * previous_radius, minimum_radius), maximum_radius
                )
                if (
                    previous_outcome is not AttemptOutcome.RETRY_STEP_BOUNDS
                    or previous_actual is None
                    or previous_actual <= 0.0
                    or previous_predicted is None
                    or previous_predicted <= 0.0
                    or expected_radius >= previous_radius
                    or trust_radius != expected_radius
                ):
                    raise ValueError("DIAG4 subtrial continuation trigger differs")
        if active_row and count < DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS:
            last = count - 1
            last_radius = float_matrices["subtrial_trust_radius"][row_index][last]
            if last_radius is None:
                raise AssertionError("selected subtrial radius narrowing failed")
            forced_radius = min(max(0.25 * last_radius, minimum_radius), maximum_radius)
            trigger = bool(
                subtrial_outcome[row_index][last] is AttemptOutcome.RETRY_STEP_BOUNDS
                and float_matrices["subtrial_actual_reduction"][row_index][last]
                is not None
                and float_matrices["subtrial_actual_reduction"][row_index][last] > 0.0
                and float_matrices["subtrial_predicted_reduction"][row_index][last]
                is not None
                and float_matrices["subtrial_predicted_reduction"][row_index][last]
                > 0.0
                and forced_radius < last_radius
            )
            if trigger:
                raise ValueError("DIAG4 subtrial sequence stops before a true trigger")
        if history is not None and active_row:
            row = history.rows[row_index]
            selected_outcome = subtrial_outcome[row_index][selected]
            if selected_outcome is not row.outcome:
                raise ValueError("DIAG4 selected subtrial outcome differs from history")
            selected_joins = (
                (
                    float_matrices["subtrial_trust_radius"][row_index][selected],
                    row.floating("trust_radius"),
                ),
                (
                    float_matrices["subtrial_actual_reduction"][row_index][selected],
                    row.floating("actual_reduction"),
                ),
                (
                    float_matrices["subtrial_predicted_reduction"][row_index][selected],
                    row.floating("predicted_reduction"),
                ),
                (
                    float_matrices["subtrial_maximum_individual_correction_step_ratio"][
                        row_index
                    ][selected],
                    individual_ratios[row_index],
                ),
                (
                    float_matrices["subtrial_correction_path_step_ratio"][row_index][
                        selected
                    ],
                    path_ratios[row_index],
                ),
                (
                    float_matrices["subtrial_corrected_radius_ratio"][row_index][
                        selected
                    ],
                    row.floating("corrected_radius_ratio"),
                ),
            )
            if any(observed != expected for observed, expected in selected_joins):
                raise ValueError(
                    "DIAG4 selected floating subtrial differs from history"
                )
            if (
                integer_work["subtrial_steihaug_iterations"][row_index][selected]
                != row.integer_values[0]
                or integer_work["subtrial_steihaug_hvp_evaluations"][row_index][
                    selected
                ]
                != row.integer_values[1]
                or integer_work["subtrial_nonlinear_corrections"][row_index][selected]
                != nonlinear_corrections[row_index]
                or integer_work["subtrial_steihaug_solve_calls"][row_index][selected]
                != steihaug_solve_calls[row_index]
            ):
                raise ValueError("DIAG4 selected integer subtrial differs from history")


def _diag4_safeguard_envelope_payload(
    value: np.ndarray, *, context: str, dtype: str, shape: tuple[int, ...]
) -> dict[str, JsonValue]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{context} must be a NumPy array")
    if value.dtype.str != dtype or value.shape != shape:
        raise ValueError(f"{context} dtype or shape differs")
    if dtype == "<i4":
        values: JsonValue = value.tolist()
    elif value.ndim == 1:
        values = [float(item) if np.isfinite(item) else None for item in value]
    else:
        values = [
            [float(item) if np.isfinite(item) else None for item in row]
            for row in value
        ]
    core: dict[str, JsonValue] = {
        "dtype": dtype,
        "shape": list(shape),
        "values": values,
    }
    return {
        **core,
        "sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }


def _parse_diag4_safeguard_envelope(
    value: JsonValue, *, context: str, dtype: str, shape: tuple[int, ...]
) -> tuple[JsonValue, str]:
    payload = _mapping(value, context)
    _exact_keys(payload, frozenset({"dtype", "shape", "values", "sha256"}), context)
    if payload["dtype"] != dtype or payload["shape"] != list(shape):
        raise ValueError(f"{context} dtype or shape differs")
    core: dict[str, JsonValue] = {
        "dtype": payload["dtype"],
        "shape": payload["shape"],
        "values": payload["values"],
    }
    digest = _sha256(payload["sha256"], f"{context} SHA")
    if digest != hashlib.sha256(canonical_json_bytes(core)).hexdigest():
        raise ValueError(f"{context} envelope hash differs")
    return payload["values"], digest


def _diag4_correction_ratio_payload(
    values: tuple[float, ...],
    nonlinear_corrections: tuple[int, ...],
    history_outcomes: tuple[str, ...],
    *,
    context: str,
) -> tuple[float | None, ...]:
    if (
        len(values) != MAXIMUM_ATTEMPTS
        or len(nonlinear_corrections) != len(values)
        or len(history_outcomes) != len(values)
    ):
        raise ValueError(f"{context} vector extent differs")
    normalized: list[float | None] = []
    for index, (value, count, raw_outcome) in enumerate(
        zip(values, nonlinear_corrections, history_outcomes, strict=True)
    ):
        if isinstance(value, bool) or not isinstance(value, float):
            raise TypeError(f"{context}[{index}] must be a float")
        outcome = AttemptOutcome(raw_outcome)
        if count == 0:
            if not math.isnan(value):
                raise ValueError(f"{context}[{index}] zero-count value must be NaN")
            normalized.append(None)
        else:
            if not math.isfinite(value):
                if outcome not in (
                    AttemptOutcome.RETRY_CORRECTION_CERTIFICATE,
                    AttemptOutcome.RETRY_NONFINITE,
                ):
                    raise ValueError(f"{context}[{index}] must be finite")
                normalized.append(None)
            else:
                if value < 0.0:
                    raise ValueError(f"{context}[{index}] must be nonnegative")
                normalized.append(value)
    return tuple(normalized)


def _parse_diag4_correction_ratio_vector(
    value: JsonValue,
    nonlinear_corrections: tuple[int, ...],
    *,
    context: str,
) -> tuple[float | None, ...]:
    raw = _array(value, context)
    if len(raw) != MAXIMUM_ATTEMPTS:
        raise ValueError(f"{context} vector extent differs")
    parsed: list[float | None] = []
    for index, (item, count) in enumerate(zip(raw, nonlinear_corrections, strict=True)):
        if count == 0:
            if item is not None:
                raise ValueError(f"{context}[{index}] zero-count value must be null")
            parsed.append(None)
        else:
            if item is None:
                parsed.append(None)
                continue
            if isinstance(item, bool) or not isinstance(item, float):
                raise TypeError(f"{context}[{index}] must be a float")
            if not math.isfinite(item) or item < 0.0:
                raise ValueError(f"{context}[{index}] must be finite and nonnegative")
            parsed.append(item)
    return tuple(parsed)


def _profiler_call_audit_payload(
    audit: Diag4ProfilerCallAudit | Diag5ProfilerCallAudit = (
        DIAG4_PROFILER_CALL_AUDIT
    ),
) -> dict[str, JsonValue]:
    """Serialize the generation-neutral frozen no-profiler call audit."""

    if (
        audit.profiler_enabled is not False
        or audit.profiler_start_calls != 0
        or audit.profiler_stop_calls != 0
        or audit.trace_normalization_calls != 0
    ):
        raise ValueError("profiler call audit differs from the frozen route")
    return {
        "profiler_enabled": audit.profiler_enabled,
        "profiler_start_calls": audit.profiler_start_calls,
        "profiler_stop_calls": audit.profiler_stop_calls,
        "trace_normalization_calls": audit.trace_normalization_calls,
    }


def diag4_profiler_call_audit_payload(
    audit: Diag4ProfilerCallAudit = DIAG4_PROFILER_CALL_AUDIT,
) -> dict[str, JsonValue]:
    """Serialize the no-profiler audit through the frozen DIAG4 API."""

    return _profiler_call_audit_payload(audit)


def _validate_profiler_call_audit(
    payload: Mapping[str, JsonValue], *, context: str
) -> None:
    expected = _profiler_call_audit_payload()
    observed = {
        "profiler_enabled": _boolean(
            payload["profiler_enabled"], f"{context} profiler enabled"
        ),
        "profiler_start_calls": _integer(
            payload["profiler_start_calls"], f"{context} profiler start calls"
        ),
        "profiler_stop_calls": _integer(
            payload["profiler_stop_calls"], f"{context} profiler stop calls"
        ),
        "trace_normalization_calls": _integer(
            payload["trace_normalization_calls"],
            f"{context} trace normalization calls",
        ),
    }
    if observed != expected:
        raise ValueError(f"{context} profiler call audit differs")


def _validate_diag4_profiler_call_audit(
    payload: Mapping[str, JsonValue], *, context: str
) -> None:
    """Compatibility wrapper for the frozen DIAG4 private surface."""

    _validate_profiler_call_audit(payload, context=context)


def _solve_timing_evidence_payload(
    *,
    child_pid: int,
    child_start_time_ticks: int,
    backend: str,
    gpu_uuid: str,
    problem_sha256: str,
    optimizer_options_sha256: str,
    base_neq_gntr1_policy_sha256: str,
    scaling_sha256: str,
    bootstrap_state_sha256: str,
    initial_physical_state_sha256: str,
    identity_sha256: str,
    source_manifest_sha256: str,
    process_started_monotonic_ns: int,
    state_ready_monotonic_ns: int,
    solve_started_monotonic_ns: int,
    solve_stopped_monotonic_ns: int,
    finalizer_completed_monotonic_ns: int,
    endpoint_audit_completed_monotonic_ns: int,
    serialization_started_monotonic_ns: int,
    hot_h2d_transfers: int,
    hot_d2h_transfers: int,
    python_callbacks: int,
    final_d2h_transfers: int,
    profiler_call_audit: Diag4ProfilerCallAudit | Diag5ProfilerCallAudit,
    schema_version: str,
    route: str,
    plan_sha256: str,
    evidence_type: type[_SolveTimingEvidence],
) -> dict[str, JsonValue]:
    """Build synchronized timing for one explicit wire generation."""

    call_audit = _profiler_call_audit_payload(profiler_call_audit)
    payload: dict[str, JsonValue] = {
        "schema_version": schema_version,
        "route": route,
        "numerical_route": DIAG4_NUMERICAL_ROUTE,
        "numerical_result_schema_version": DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "child_pid": child_pid,
        "child_start_time_ticks": child_start_time_ticks,
        "backend": backend,
        "gpu_uuid": gpu_uuid,
        **call_audit,
        "problem_sha256": problem_sha256,
        "optimizer_options_sha256": optimizer_options_sha256,
        "base_neq_gntr1_policy_sha256": base_neq_gntr1_policy_sha256,
        "scaling_sha256": scaling_sha256,
        "bootstrap_state_sha256": bootstrap_state_sha256,
        "initial_physical_state_sha256": initial_physical_state_sha256,
        "identity_sha256": identity_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "process_started_monotonic_ns": process_started_monotonic_ns,
        "state_ready_monotonic_ns": state_ready_monotonic_ns,
        "solve_started_monotonic_ns": solve_started_monotonic_ns,
        "solve_stopped_monotonic_ns": solve_stopped_monotonic_ns,
        "finalizer_completed_monotonic_ns": finalizer_completed_monotonic_ns,
        "endpoint_audit_completed_monotonic_ns": endpoint_audit_completed_monotonic_ns,
        "serialization_started_monotonic_ns": serialization_started_monotonic_ns,
        "synchronized_solve_seconds": (
            solve_stopped_monotonic_ns - solve_started_monotonic_ns
        )
        / 1.0e9,
        "transfer_audit": {
            "hot_h2d_transfers": hot_h2d_transfers,
            "hot_d2h_transfers": hot_d2h_transfers,
            "python_callbacks": python_callbacks,
            "final_d2h_transfers": final_d2h_transfers,
        },
    }
    _validate_solve_timing_evidence_payload(
        payload,
        schema_version=schema_version,
        route=route,
        plan_sha256=plan_sha256,
        evidence_type=evidence_type,
    )
    return payload


def solve_timing_evidence_payload(
    *,
    child_pid: int,
    child_start_time_ticks: int,
    backend: str,
    gpu_uuid: str,
    problem_sha256: str,
    optimizer_options_sha256: str,
    base_neq_gntr1_policy_sha256: str,
    scaling_sha256: str,
    bootstrap_state_sha256: str,
    initial_physical_state_sha256: str,
    identity_sha256: str,
    source_manifest_sha256: str,
    process_started_monotonic_ns: int,
    state_ready_monotonic_ns: int,
    solve_started_monotonic_ns: int,
    solve_stopped_monotonic_ns: int,
    finalizer_completed_monotonic_ns: int,
    endpoint_audit_completed_monotonic_ns: int,
    serialization_started_monotonic_ns: int,
    hot_h2d_transfers: int,
    hot_d2h_transfers: int,
    python_callbacks: int,
    final_d2h_transfers: int,
    profiler_call_audit: Diag4ProfilerCallAudit = DIAG4_PROFILER_CALL_AUDIT,
) -> dict[str, JsonValue]:
    """Build only the frozen DIAG4 timing generation."""

    return _solve_timing_evidence_payload(
        child_pid=child_pid,
        child_start_time_ticks=child_start_time_ticks,
        backend=backend,
        gpu_uuid=gpu_uuid,
        problem_sha256=problem_sha256,
        optimizer_options_sha256=optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=base_neq_gntr1_policy_sha256,
        scaling_sha256=scaling_sha256,
        bootstrap_state_sha256=bootstrap_state_sha256,
        initial_physical_state_sha256=initial_physical_state_sha256,
        identity_sha256=identity_sha256,
        source_manifest_sha256=source_manifest_sha256,
        process_started_monotonic_ns=process_started_monotonic_ns,
        state_ready_monotonic_ns=state_ready_monotonic_ns,
        solve_started_monotonic_ns=solve_started_monotonic_ns,
        solve_stopped_monotonic_ns=solve_stopped_monotonic_ns,
        finalizer_completed_monotonic_ns=finalizer_completed_monotonic_ns,
        endpoint_audit_completed_monotonic_ns=endpoint_audit_completed_monotonic_ns,
        serialization_started_monotonic_ns=serialization_started_monotonic_ns,
        hot_h2d_transfers=hot_h2d_transfers,
        hot_d2h_transfers=hot_d2h_transfers,
        python_callbacks=python_callbacks,
        final_d2h_transfers=final_d2h_transfers,
        profiler_call_audit=profiler_call_audit,
        schema_version=DIAG4_SOLVE_TIMING_SCHEMA_VERSION,
        route=DIAG4_ROUTE,
        plan_sha256=DIAG4_PLAN_SHA256,
        evidence_type=SolveTimingEvidenceV4,
    )


def _validate_solve_timing_evidence_payload(
    value: JsonValue,
    *,
    schema_version: str,
    route: str,
    plan_sha256: str,
    evidence_type: type[_SolveTimingEvidence],
) -> _SolveTimingEvidence:
    """Parse exact synchronized timing under one explicit wire generation."""

    payload = _mapping(value, "DIAG4 solve timing")
    timestamp_fields = (
        "process_started_monotonic_ns",
        "state_ready_monotonic_ns",
        "solve_started_monotonic_ns",
        "solve_stopped_monotonic_ns",
        "finalizer_completed_monotonic_ns",
        "endpoint_audit_completed_monotonic_ns",
        "serialization_started_monotonic_ns",
    )
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "numerical_result_schema_version",
                "plan_sha256",
                "child_pid",
                "child_start_time_ticks",
                "backend",
                "gpu_uuid",
                "profiler_enabled",
                "profiler_start_calls",
                "profiler_stop_calls",
                "trace_normalization_calls",
                "problem_sha256",
                "optimizer_options_sha256",
                "base_neq_gntr1_policy_sha256",
                "scaling_sha256",
                "bootstrap_state_sha256",
                "initial_physical_state_sha256",
                "identity_sha256",
                "source_manifest_sha256",
                *timestamp_fields,
                "synchronized_solve_seconds",
                "transfer_audit",
            }
        ),
        "DIAG4 solve timing",
    )
    if (
        payload["schema_version"] != schema_version
        or payload["route"] != route
        or payload["numerical_route"] != DIAG4_NUMERICAL_ROUTE
        or payload["numerical_result_schema_version"]
        != DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
        or payload["plan_sha256"] != plan_sha256
        or payload["backend"] != "gpu"
        or payload["gpu_uuid"] != GPU_UUID
    ):
        raise ValueError("DIAG4 solve timing identity differs")
    _validate_profiler_call_audit(payload, context="DIAG4 solve timing")
    timestamps = tuple(
        _integer(payload[name], f"DIAG4 solve timing.{name}")
        for name in timestamp_fields
    )
    if any(left >= right for left, right in zip(timestamps[:-1], timestamps[1:])):
        raise ValueError("DIAG4 solve timing order differs")
    synchronized_seconds = _number(
        payload["synchronized_solve_seconds"], "synchronized solve seconds"
    )
    expected_seconds = (timestamps[3] - timestamps[2]) / 1.0e9
    if synchronized_seconds <= 0.0 or synchronized_seconds != expected_seconds:
        raise ValueError("DIAG4 synchronized solve arithmetic differs")
    transfers = _mapping(payload["transfer_audit"], "DIAG4 transfer audit")
    _exact_keys(
        transfers,
        frozenset(
            {
                "hot_h2d_transfers",
                "hot_d2h_transfers",
                "python_callbacks",
                "final_d2h_transfers",
            }
        ),
        "DIAG4 transfer audit",
    )
    transfer_counts = tuple(
        _integer(transfers[name], f"DIAG4 transfer audit.{name}")
        for name in (
            "hot_h2d_transfers",
            "hot_d2h_transfers",
            "python_callbacks",
            "final_d2h_transfers",
        )
    )
    if transfer_counts != (0, 0, 0, 1):
        raise ValueError("DIAG4 transfer audit differs from the frozen boundary")
    return evidence_type(
        _integer(payload["child_pid"], "DIAG4 child PID", minimum=1),
        _integer(
            payload["child_start_time_ticks"],
            "DIAG4 child start ticks",
            minimum=1,
        ),
        _string(payload["backend"], "DIAG4 backend"),
        _string(payload["gpu_uuid"], "DIAG4 GPU UUID"),
        _string(payload["numerical_route"], "DIAG4 numerical route"),
        _string(
            payload["numerical_result_schema_version"],
            "DIAG4 numerical result schema version",
        ),
        _sha256(payload["problem_sha256"], "DIAG4 problem SHA"),
        _sha256(payload["optimizer_options_sha256"], "DIAG4 options SHA"),
        _sha256(payload["base_neq_gntr1_policy_sha256"], "DIAG4 base policy SHA"),
        _sha256(payload["scaling_sha256"], "DIAG4 scaling SHA"),
        _sha256(payload["bootstrap_state_sha256"], "DIAG4 bootstrap-state SHA"),
        _sha256(
            payload["initial_physical_state_sha256"],
            "DIAG4 initial-physical-state SHA",
        ),
        _sha256(payload["identity_sha256"], "DIAG4 identity SHA"),
        _sha256(payload["source_manifest_sha256"], "DIAG4 source-manifest SHA"),
        *timestamps,
        synchronized_seconds,
        *transfer_counts,
        0,
        0,
        0,
    )


def validate_solve_timing_evidence_payload(
    value: JsonValue,
) -> SolveTimingEvidenceV4:
    """Parse the frozen DIAG4 timing generation without successor fallback."""

    return _validate_solve_timing_evidence_payload(
        value,
        schema_version=DIAG4_SOLVE_TIMING_SCHEMA_VERSION,
        route=DIAG4_ROUTE,
        plan_sha256=DIAG4_PLAN_SHA256,
        evidence_type=SolveTimingEvidenceV4,
    )


def _safeguard_telemetry_payload(
    *,
    history_evidence: ArtifactRef,
    problem_sha256: str,
    optimizer_options_sha256: str,
    base_neq_gntr1_policy_sha256: str,
    scaling_sha256: str,
    bootstrap_state_sha256: str,
    initial_physical_state_sha256: str,
    identity_sha256: str,
    loop_attempts: int,
    accepted_steps: int,
    retryable_rejections: int,
    terminal_status: str,
    quality_latch: bool,
    history_outcomes: tuple[str, ...],
    nonlinear_corrections: np.ndarray,
    maximum_individual_correction_step_ratio: np.ndarray,
    correction_path_step_ratio: np.ndarray,
    steihaug_solve_calls: np.ndarray,
    subtrial_count: np.ndarray,
    selected_subtrial_index: np.ndarray,
    subtrial_trust_radius: np.ndarray,
    subtrial_outcome: np.ndarray,
    subtrial_actual_reduction: np.ndarray,
    subtrial_predicted_reduction: np.ndarray,
    subtrial_maximum_individual_correction_step_ratio: np.ndarray,
    subtrial_correction_path_step_ratio: np.ndarray,
    subtrial_corrected_radius_ratio: np.ndarray,
    subtrial_steihaug_iterations: np.ndarray,
    subtrial_steihaug_hvp_evaluations: np.ndarray,
    subtrial_steihaug_solve_calls: np.ndarray,
    subtrial_total_hvp_evaluations: np.ndarray,
    subtrial_nonlinear_corrections: np.ndarray,
    subtrial_joint_evaluations: np.ndarray,
    subtrial_joint_linearizations: np.ndarray,
    subtrial_joint_value_evaluations: np.ndarray,
    subtrial_objective_residual_linearizations: np.ndarray,
    subtrial_gram_factorizations: np.ndarray,
    subtrial_gram_solves: np.ndarray,
    schema_version: str,
    route: str,
    plan_sha256: str,
    evidence_type: type[_SafeguardTelemetry],
) -> dict[str, JsonValue]:
    """Build correction telemetry under one explicit wire generation."""

    vector_shape = (MAXIMUM_ATTEMPTS,)
    matrix_shape = (MAXIMUM_ATTEMPTS, DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS)
    outer_envelopes = {
        "nonlinear_corrections": _diag4_safeguard_envelope_payload(
            nonlinear_corrections,
            context="DIAG4 nonlinear corrections",
            dtype="<i4",
            shape=vector_shape,
        ),
        "maximum_individual_correction_step_ratio": (
            _diag4_safeguard_envelope_payload(
                maximum_individual_correction_step_ratio,
                context="DIAG4 maximum individual correction-step ratio",
                dtype="<f8",
                shape=vector_shape,
            )
        ),
        "correction_path_step_ratio": _diag4_safeguard_envelope_payload(
            correction_path_step_ratio,
            context="DIAG4 correction-path step ratio",
            dtype="<f8",
            shape=vector_shape,
        ),
        "steihaug_solve_calls": _diag4_safeguard_envelope_payload(
            steihaug_solve_calls,
            context="DIAG4 Steihaug solve calls",
            dtype="<i4",
            shape=vector_shape,
        ),
    }
    corrections = tuple(int(item) for item in nonlinear_corrections)
    solve_calls = tuple(int(item) for item in steihaug_solve_calls)
    individual_ratios = _diag4_correction_ratio_payload(
        tuple(float(item) for item in maximum_individual_correction_step_ratio),
        corrections,
        history_outcomes,
        context="DIAG4 maximum individual correction-step ratio",
    )
    path_ratios = _diag4_correction_ratio_payload(
        tuple(float(item) for item in correction_path_step_ratio),
        corrections,
        history_outcomes,
        context="DIAG4 correction-path step ratio",
    )
    subtrial_envelopes = {
        "subtrial_count": _diag4_safeguard_envelope_payload(
            subtrial_count,
            context="DIAG4 subtrial count",
            dtype="<i4",
            shape=vector_shape,
        ),
        "selected_subtrial_index": _diag4_safeguard_envelope_payload(
            selected_subtrial_index,
            context="DIAG4 selected subtrial index",
            dtype="<i4",
            shape=vector_shape,
        ),
        "subtrial_outcome": _diag4_safeguard_envelope_payload(
            subtrial_outcome,
            context="DIAG4 subtrial outcome",
            dtype="<i4",
            shape=matrix_shape,
        ),
    }
    for field, array in (
        ("subtrial_trust_radius", subtrial_trust_radius),
        ("subtrial_actual_reduction", subtrial_actual_reduction),
        ("subtrial_predicted_reduction", subtrial_predicted_reduction),
        (
            "subtrial_maximum_individual_correction_step_ratio",
            subtrial_maximum_individual_correction_step_ratio,
        ),
        (
            "subtrial_correction_path_step_ratio",
            subtrial_correction_path_step_ratio,
        ),
        ("subtrial_corrected_radius_ratio", subtrial_corrected_radius_ratio),
    ):
        subtrial_envelopes[field] = _diag4_safeguard_envelope_payload(
            array,
            context=f"DIAG4 {field}",
            dtype="<f8",
            shape=matrix_shape,
        )
    for field, array in (
        ("subtrial_steihaug_iterations", subtrial_steihaug_iterations),
        (
            "subtrial_steihaug_hvp_evaluations",
            subtrial_steihaug_hvp_evaluations,
        ),
        ("subtrial_steihaug_solve_calls", subtrial_steihaug_solve_calls),
        ("subtrial_total_hvp_evaluations", subtrial_total_hvp_evaluations),
        ("subtrial_nonlinear_corrections", subtrial_nonlinear_corrections),
        ("subtrial_joint_evaluations", subtrial_joint_evaluations),
        ("subtrial_joint_linearizations", subtrial_joint_linearizations),
        (
            "subtrial_joint_value_evaluations",
            subtrial_joint_value_evaluations,
        ),
        (
            "subtrial_objective_residual_linearizations",
            subtrial_objective_residual_linearizations,
        ),
        ("subtrial_gram_factorizations", subtrial_gram_factorizations),
        ("subtrial_gram_solves", subtrial_gram_solves),
    ):
        subtrial_envelopes[field] = _diag4_safeguard_envelope_payload(
            array,
            context=f"DIAG4 {field}",
            dtype="<i4",
            shape=matrix_shape,
        )
    subtrial_counts = tuple(int(item) for item in subtrial_count)
    selected_subtrial_indices = tuple(int(item) for item in selected_subtrial_index)
    outcome_members = tuple(AttemptOutcome)
    if np.any(subtrial_outcome < 0) or np.any(subtrial_outcome >= len(outcome_members)):
        raise ValueError("DIAG4 subtrial outcome code differs")
    subtrial_outcomes = tuple(
        tuple(outcome_members[int(item)] for item in row) for row in subtrial_outcome
    )
    subtrial_float_matrices = {
        field: _diag4_subtrial_float_matrix(
            subtrial_envelopes[field]["values"], context=f"DIAG4 {field}"
        )
        for field in DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS
    }
    subtrial_integer_matrices = {
        field: tuple(tuple(int(item) for item in row) for row in array)
        for field, array in (
            ("subtrial_steihaug_iterations", subtrial_steihaug_iterations),
            (
                "subtrial_steihaug_hvp_evaluations",
                subtrial_steihaug_hvp_evaluations,
            ),
            ("subtrial_steihaug_solve_calls", subtrial_steihaug_solve_calls),
            ("subtrial_total_hvp_evaluations", subtrial_total_hvp_evaluations),
            ("subtrial_nonlinear_corrections", subtrial_nonlinear_corrections),
            ("subtrial_joint_evaluations", subtrial_joint_evaluations),
            ("subtrial_joint_linearizations", subtrial_joint_linearizations),
            (
                "subtrial_joint_value_evaluations",
                subtrial_joint_value_evaluations,
            ),
            (
                "subtrial_objective_residual_linearizations",
                subtrial_objective_residual_linearizations,
            ),
            ("subtrial_gram_factorizations", subtrial_gram_factorizations),
            ("subtrial_gram_solves", subtrial_gram_solves),
        )
    }
    for index, (count, raw_outcome, individual_ratio, path_ratio) in enumerate(
        zip(
            corrections[:loop_attempts],
            history_outcomes[:loop_attempts],
            individual_ratios[:loop_attempts],
            path_ratios[:loop_attempts],
            strict=True,
        )
    ):
        outcome = AttemptOutcome(raw_outcome)
        if count == 0:
            continue
        if (
            individual_ratio is not None
            and path_ratio is not None
            and (
                individual_ratio > path_ratio
                or (count == 1 and individual_ratio != path_ratio)
            )
        ):
            raise ValueError(f"DIAG4 correction ratios differ at row {index}")
        if outcome is AttemptOutcome.ACCEPTED and (
            individual_ratio is None
            or path_ratio is None
            or individual_ratio > FROZEN_GNTR_OPTIONS["maximum_correction_step_ratio"]
            or path_ratio > count * FROZEN_GNTR_OPTIONS["maximum_correction_step_ratio"]
        ):
            raise ValueError(f"DIAG4 accepted correction bounds differ at row {index}")
    _validate_diag4_subtrial_structure(
        attempts=loop_attempts,
        subtrial_count=subtrial_counts,
        selected_subtrial_index=selected_subtrial_indices,
        subtrial_outcome=subtrial_outcomes,
        float_matrices=subtrial_float_matrices,
        integer_work=subtrial_integer_matrices,
        history=None,
        nonlinear_corrections=corrections,
        individual_ratios=individual_ratios,
        path_ratios=path_ratios,
        steihaug_solve_calls=solve_calls,
    )
    for index, count in enumerate(subtrial_counts[:loop_attempts]):
        selected = count - 1
        if subtrial_outcomes[index][selected] is not AttemptOutcome(
            history_outcomes[index]
        ):
            raise ValueError(
                "DIAG4 selected subtrial outcome differs from history hash"
            )
        if (
            subtrial_float_matrices[
                "subtrial_maximum_individual_correction_step_ratio"
            ][index][selected]
            != individual_ratios[index]
            or subtrial_float_matrices["subtrial_correction_path_step_ratio"][index][
                selected
            ]
            != path_ratios[index]
            or subtrial_integer_matrices["subtrial_nonlinear_corrections"][index][
                selected
            ]
            != corrections[index]
            or subtrial_integer_matrices["subtrial_steihaug_solve_calls"][index][
                selected
            ]
            != solve_calls[index]
        ):
            raise ValueError("DIAG4 selected correction subtrial differs")
    subtrial_summary = dict(
        _diag4_subtrial_summary(
            attempts=loop_attempts,
            subtrial_count=subtrial_counts,
            subtrial_outcome=subtrial_outcomes,
            integer_work=subtrial_integer_matrices,
        )
    )
    payload: dict[str, JsonValue] = {
        "schema_version": schema_version,
        "route": route,
        "numerical_route": DIAG4_NUMERICAL_ROUTE,
        "numerical_result_schema_version": DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "history_evidence": _artifact_ref_payload(history_evidence),
        "problem_sha256": problem_sha256,
        "optimizer_options_sha256": optimizer_options_sha256,
        "base_neq_gntr1_policy_sha256": base_neq_gntr1_policy_sha256,
        "scaling_sha256": scaling_sha256,
        "bootstrap_state_sha256": bootstrap_state_sha256,
        "initial_physical_state_sha256": initial_physical_state_sha256,
        "identity_sha256": identity_sha256,
        "loop_attempts": loop_attempts,
        "accepted_steps": accepted_steps,
        "retryable_rejections": retryable_rejections,
        "terminal_status": terminal_status,
        "quality_latch": quality_latch,
        **outer_envelopes,
        "history_outcomes_sha256": _diag4_history_outcomes_sha256(history_outcomes),
        **subtrial_envelopes,
        "subtrial_summary": subtrial_summary,
    }
    _validate_safeguard_telemetry_payload(
        payload,
        schema_version=schema_version,
        route=route,
        plan_sha256=plan_sha256,
        evidence_type=evidence_type,
    )
    return payload


def safeguard_telemetry_payload(
    *,
    history_evidence: ArtifactRef,
    problem_sha256: str,
    optimizer_options_sha256: str,
    base_neq_gntr1_policy_sha256: str,
    scaling_sha256: str,
    bootstrap_state_sha256: str,
    initial_physical_state_sha256: str,
    identity_sha256: str,
    loop_attempts: int,
    accepted_steps: int,
    retryable_rejections: int,
    terminal_status: str,
    quality_latch: bool,
    history_outcomes: tuple[str, ...],
    nonlinear_corrections: np.ndarray,
    maximum_individual_correction_step_ratio: np.ndarray,
    correction_path_step_ratio: np.ndarray,
    steihaug_solve_calls: np.ndarray,
    subtrial_count: np.ndarray,
    selected_subtrial_index: np.ndarray,
    subtrial_trust_radius: np.ndarray,
    subtrial_outcome: np.ndarray,
    subtrial_actual_reduction: np.ndarray,
    subtrial_predicted_reduction: np.ndarray,
    subtrial_maximum_individual_correction_step_ratio: np.ndarray,
    subtrial_correction_path_step_ratio: np.ndarray,
    subtrial_corrected_radius_ratio: np.ndarray,
    subtrial_steihaug_iterations: np.ndarray,
    subtrial_steihaug_hvp_evaluations: np.ndarray,
    subtrial_steihaug_solve_calls: np.ndarray,
    subtrial_total_hvp_evaluations: np.ndarray,
    subtrial_nonlinear_corrections: np.ndarray,
    subtrial_joint_evaluations: np.ndarray,
    subtrial_joint_linearizations: np.ndarray,
    subtrial_joint_value_evaluations: np.ndarray,
    subtrial_objective_residual_linearizations: np.ndarray,
    subtrial_gram_factorizations: np.ndarray,
    subtrial_gram_solves: np.ndarray,
) -> dict[str, JsonValue]:
    """Build the frozen DIAG4 correction telemetry generation."""

    return _safeguard_telemetry_payload(
        history_evidence=history_evidence,
        problem_sha256=problem_sha256,
        optimizer_options_sha256=optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=base_neq_gntr1_policy_sha256,
        scaling_sha256=scaling_sha256,
        bootstrap_state_sha256=bootstrap_state_sha256,
        initial_physical_state_sha256=initial_physical_state_sha256,
        identity_sha256=identity_sha256,
        loop_attempts=loop_attempts,
        accepted_steps=accepted_steps,
        retryable_rejections=retryable_rejections,
        terminal_status=terminal_status,
        quality_latch=quality_latch,
        history_outcomes=history_outcomes,
        nonlinear_corrections=nonlinear_corrections,
        maximum_individual_correction_step_ratio=maximum_individual_correction_step_ratio,
        correction_path_step_ratio=correction_path_step_ratio,
        steihaug_solve_calls=steihaug_solve_calls,
        subtrial_count=subtrial_count,
        selected_subtrial_index=selected_subtrial_index,
        subtrial_trust_radius=subtrial_trust_radius,
        subtrial_outcome=subtrial_outcome,
        subtrial_actual_reduction=subtrial_actual_reduction,
        subtrial_predicted_reduction=subtrial_predicted_reduction,
        subtrial_maximum_individual_correction_step_ratio=subtrial_maximum_individual_correction_step_ratio,
        subtrial_correction_path_step_ratio=subtrial_correction_path_step_ratio,
        subtrial_corrected_radius_ratio=subtrial_corrected_radius_ratio,
        subtrial_steihaug_iterations=subtrial_steihaug_iterations,
        subtrial_steihaug_hvp_evaluations=subtrial_steihaug_hvp_evaluations,
        subtrial_steihaug_solve_calls=subtrial_steihaug_solve_calls,
        subtrial_total_hvp_evaluations=subtrial_total_hvp_evaluations,
        subtrial_nonlinear_corrections=subtrial_nonlinear_corrections,
        subtrial_joint_evaluations=subtrial_joint_evaluations,
        subtrial_joint_linearizations=subtrial_joint_linearizations,
        subtrial_joint_value_evaluations=subtrial_joint_value_evaluations,
        subtrial_objective_residual_linearizations=subtrial_objective_residual_linearizations,
        subtrial_gram_factorizations=subtrial_gram_factorizations,
        subtrial_gram_solves=subtrial_gram_solves,
        schema_version=DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
        route=DIAG4_ROUTE,
        plan_sha256=DIAG4_PLAN_SHA256,
        evidence_type=SafeguardTelemetryV4,
    )


def _validate_safeguard_telemetry_payload(
    value: JsonValue,
    *,
    history: HistoryEvidence | None = None,
    expected_history_evidence: ArtifactRef | None = None,
    schema_version: str,
    route: str,
    plan_sha256: str,
    evidence_type: type[_SafeguardTelemetry],
) -> _SafeguardTelemetry:
    """Parse correction telemetry under one explicit wire generation."""

    payload = _mapping(value, "DIAG4 safeguard telemetry")
    keys = frozenset(
        {
            "schema_version",
            "route",
            "numerical_route",
            "numerical_result_schema_version",
            "plan_sha256",
            "history_evidence",
            "problem_sha256",
            "optimizer_options_sha256",
            "base_neq_gntr1_policy_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
            "loop_attempts",
            "accepted_steps",
            "retryable_rejections",
            "terminal_status",
            "quality_latch",
            "nonlinear_corrections",
            "maximum_individual_correction_step_ratio",
            "correction_path_step_ratio",
            "steihaug_solve_calls",
            "history_outcomes_sha256",
            "subtrial_count",
            "selected_subtrial_index",
            *DIAG4_SUBTRIAL_MATRIX_FIELDS,
            "subtrial_summary",
        }
    )
    _exact_keys(payload, keys, "DIAG4 safeguard telemetry")
    if (
        payload["schema_version"] != schema_version
        or payload["route"] != route
        or payload["numerical_route"] != DIAG4_NUMERICAL_ROUTE
        or payload["numerical_result_schema_version"]
        != DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
        or payload["plan_sha256"] != plan_sha256
    ):
        raise ValueError("DIAG4 safeguard telemetry identity differs")
    history_reference = _artifact_ref(
        payload["history_evidence"], "DIAG4 telemetry history evidence"
    )
    if (
        expected_history_evidence is not None
        and history_reference != expected_history_evidence
    ):
        raise ValueError("DIAG4 telemetry history reference differs")
    attempts = _integer(payload["loop_attempts"], "DIAG4 loop attempts", minimum=0)
    accepted_steps = _integer(
        payload["accepted_steps"], "DIAG4 accepted steps", minimum=0
    )
    retryable_rejections = _integer(
        payload["retryable_rejections"],
        "DIAG4 retryable rejections",
        minimum=0,
    )
    terminal_status = LoopStatus(
        _string(payload["terminal_status"], "DIAG4 terminal status")
    )
    quality_latch = _boolean(payload["quality_latch"], "DIAG4 quality latch")
    vector_shape = (MAXIMUM_ATTEMPTS,)
    matrix_shape = (MAXIMUM_ATTEMPTS, DIAG4_MAXIMUM_SAFEGUARD_SUBTRIALS)
    correction_values, corrections_sha256 = _parse_diag4_safeguard_envelope(
        payload["nonlinear_corrections"],
        context="DIAG4 nonlinear corrections",
        dtype="<i4",
        shape=vector_shape,
    )
    corrections = _diag4_subtrial_integer_vector(
        correction_values, context="DIAG4 nonlinear correction values"
    )
    if (
        len(corrections) != MAXIMUM_ATTEMPTS
        or attempts > MAXIMUM_ATTEMPTS
        or accepted_steps > attempts
        or retryable_rejections > attempts
    ):
        raise ValueError("DIAG4 correction vector extent differs")
    active = corrections[:attempts]
    if any(
        value < 0 or value > DIAG4_MAXIMUM_NONLINEAR_CORRECTIONS for value in active
    ) or any(corrections[attempts:]):
        raise ValueError("DIAG4 correction vector range or padding differs")
    individual_values, individual_sha256 = _parse_diag4_safeguard_envelope(
        payload["maximum_individual_correction_step_ratio"],
        context="DIAG4 maximum individual correction-step ratio",
        dtype="<f8",
        shape=vector_shape,
    )
    individual_ratios = _parse_diag4_correction_ratio_vector(
        individual_values,
        corrections,
        context="DIAG4 maximum individual correction-step ratio",
    )
    path_values, path_sha256 = _parse_diag4_safeguard_envelope(
        payload["correction_path_step_ratio"],
        context="DIAG4 correction-path step ratio",
        dtype="<f8",
        shape=vector_shape,
    )
    path_ratios = _parse_diag4_correction_ratio_vector(
        path_values,
        corrections,
        context="DIAG4 correction-path step ratio",
    )
    solve_call_values, steihaug_solve_calls_sha256 = _parse_diag4_safeguard_envelope(
        payload["steihaug_solve_calls"],
        context="DIAG4 Steihaug solve calls",
        dtype="<i4",
        shape=vector_shape,
    )
    steihaug_solve_calls = _diag4_subtrial_integer_vector(
        solve_call_values, context="DIAG4 Steihaug solve-call values"
    )
    if any(value not in (0, 1) for value in steihaug_solve_calls[:attempts]) or any(
        steihaug_solve_calls[attempts:]
    ):
        raise ValueError("DIAG4 Steihaug solve-call range or padding differs")
    count_values, subtrial_count_sha256 = _parse_diag4_safeguard_envelope(
        payload["subtrial_count"],
        context="DIAG4 subtrial count",
        dtype="<i4",
        shape=vector_shape,
    )
    selected_values, selected_subtrial_index_sha256 = _parse_diag4_safeguard_envelope(
        payload["selected_subtrial_index"],
        context="DIAG4 selected subtrial index",
        dtype="<i4",
        shape=vector_shape,
    )
    subtrial_counts = _diag4_subtrial_integer_vector(
        count_values, context="DIAG4 subtrial count values"
    )
    selected_subtrial_indices = _diag4_subtrial_integer_vector(
        selected_values,
        context="DIAG4 selected subtrial index values",
        minimum=-1,
    )
    outcome_values, subtrial_outcome_sha256 = _parse_diag4_safeguard_envelope(
        payload["subtrial_outcome"],
        context="DIAG4 subtrial outcome",
        dtype="<i4",
        shape=matrix_shape,
    )
    outcome_codes = _diag4_subtrial_integer_matrix(
        outcome_values, context="DIAG4 subtrial outcome values"
    )
    outcome_members = tuple(AttemptOutcome)
    if any(
        item < 0 or item >= len(outcome_members)
        for row in outcome_codes
        for item in row
    ):
        raise ValueError("DIAG4 subtrial outcome code differs")
    subtrial_outcomes = tuple(
        tuple(outcome_members[item] for item in row) for row in outcome_codes
    )
    subtrial_float_matrices: dict[str, tuple[tuple[float | None, ...], ...]] = {}
    subtrial_float_sha256: dict[str, str] = {}
    for field in DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS:
        values, digest = _parse_diag4_safeguard_envelope(
            payload[field],
            context=f"DIAG4 {field}",
            dtype="<f8",
            shape=matrix_shape,
        )
        subtrial_float_matrices[field] = _diag4_subtrial_float_matrix(
            values, context=f"DIAG4 {field} values"
        )
        subtrial_float_sha256[field] = digest
    subtrial_integer_matrices: dict[str, tuple[tuple[int, ...], ...]] = {}
    subtrial_integer_sha256: dict[str, str] = {}
    for field in DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS:
        values, digest = _parse_diag4_safeguard_envelope(
            payload[field],
            context=f"DIAG4 {field}",
            dtype="<i4",
            shape=matrix_shape,
        )
        subtrial_integer_matrices[field] = _diag4_subtrial_integer_matrix(
            values, context=f"DIAG4 {field} values"
        )
        subtrial_integer_sha256[field] = digest
    _validate_diag4_subtrial_structure(
        attempts=attempts,
        subtrial_count=subtrial_counts,
        selected_subtrial_index=selected_subtrial_indices,
        subtrial_outcome=subtrial_outcomes,
        float_matrices=subtrial_float_matrices,
        integer_work=subtrial_integer_matrices,
        history=history,
        nonlinear_corrections=corrections,
        individual_ratios=individual_ratios,
        path_ratios=path_ratios,
        steihaug_solve_calls=steihaug_solve_calls,
    )
    summary_payload = _mapping(payload["subtrial_summary"], "DIAG4 subtrial summary")
    _exact_keys(
        summary_payload,
        frozenset(DIAG4_SUBTRIAL_SUMMARY_FIELDS),
        "DIAG4 subtrial summary",
    )
    declared_subtrial_summary = tuple(
        (field, _integer(summary_payload[field], f"DIAG4 subtrial summary.{field}"))
        for field in DIAG4_SUBTRIAL_SUMMARY_FIELDS
    )
    expected_subtrial_summary = _diag4_subtrial_summary(
        attempts=attempts,
        subtrial_count=subtrial_counts,
        subtrial_outcome=subtrial_outcomes,
        integer_work=subtrial_integer_matrices,
    )
    if declared_subtrial_summary != expected_subtrial_summary:
        raise ValueError("DIAG4 subtrial summary differs from matrices")
    outcomes_sha256 = _sha256(
        payload["history_outcomes_sha256"], "DIAG4 history outcomes SHA"
    )
    if history is not None:
        outcomes = tuple(row.outcome.value for row in history.rows)
        if (
            attempts != history.attempts
            or accepted_steps != history.accepted_steps
            or retryable_rejections != history.retryable_rejections
            or terminal_status is not history.status
            or quality_latch is not history.quality_latch
            or outcomes_sha256 != _diag4_history_outcomes_sha256(outcomes)
        ):
            raise ValueError("DIAG4 telemetry differs from legacy history")
        for index, row in enumerate(history.rows[:attempts]):
            expected_solve_calls = int(
                row.outcome is not AttemptOutcome.FATAL_CURRENT_STATE
            )
            if steihaug_solve_calls[index] != expected_solve_calls:
                raise ValueError("DIAG4 Steihaug solve calls differ from history")
            if row.outcome not in FATAL_OUTCOMES and corrections[index] == 0:
                raise ValueError("DIAG4 active trial row omits a correction")
            individual_ratio = individual_ratios[index]
            path_ratio = path_ratios[index]
            if corrections[index] == 0:
                continue
            permits_nonfinite_ratio = row.outcome in (
                AttemptOutcome.RETRY_CORRECTION_CERTIFICATE,
                AttemptOutcome.RETRY_NONFINITE,
            )
            if (
                individual_ratio is None or path_ratio is None
            ) and not permits_nonfinite_ratio:
                raise ValueError(
                    "DIAG4 correction ratio finiteness differs from outcome"
                )
            net_ratio = row.floating("correction_step_ratio")
            if (
                net_ratio is not None
                and individual_ratio is not None
                and path_ratio is not None
                and (
                    net_ratio < 0.0
                    or net_ratio > path_ratio
                    or (
                        corrections[index] == 1
                        and not (net_ratio == individual_ratio == path_ratio)
                    )
                )
            ) or (
                individual_ratio is not None
                and path_ratio is not None
                and individual_ratio > path_ratio
            ):
                raise ValueError("DIAG4 correction ratios differ from legacy history")
            corrected_radius_ratio = row.floating("corrected_radius_ratio")
            if row.outcome is AttemptOutcome.ACCEPTED:
                if (
                    individual_ratio is None
                    or path_ratio is None
                    or corrected_radius_ratio is None
                    or individual_ratio
                    > FROZEN_GNTR_OPTIONS["maximum_correction_step_ratio"]
                    or path_ratio
                    > corrections[index]
                    * FROZEN_GNTR_OPTIONS["maximum_correction_step_ratio"]
                    or corrected_radius_ratio
                    > 1.0 + FROZEN_GNTR_OPTIONS["maximum_corrected_radius_excess"]
                ):
                    raise ValueError("DIAG4 accepted correction bounds differ")
            elif row.outcome is AttemptOutcome.RETRY_STEP_BOUNDS:
                if (
                    individual_ratio is None
                    or path_ratio is None
                    or corrected_radius_ratio is None
                ):
                    raise ValueError("DIAG4 step-bound retry omits finite ratios")
                if (
                    individual_ratio
                    <= FROZEN_GNTR_OPTIONS["maximum_correction_step_ratio"]
                    and path_ratio
                    <= corrections[index]
                    * FROZEN_GNTR_OPTIONS["maximum_correction_step_ratio"]
                    and corrected_radius_ratio
                    <= 1.0 + FROZEN_GNTR_OPTIONS["maximum_corrected_radius_excess"]
                ):
                    raise ValueError("DIAG4 step-bound retry has no failed bound")
    return evidence_type(
        history_reference,
        DIAG4_NUMERICAL_ROUTE,
        DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
        _sha256(payload["problem_sha256"], "DIAG4 problem SHA"),
        _sha256(payload["optimizer_options_sha256"], "DIAG4 options SHA"),
        _sha256(payload["base_neq_gntr1_policy_sha256"], "DIAG4 base policy SHA"),
        _sha256(payload["scaling_sha256"], "DIAG4 scaling SHA"),
        _sha256(payload["bootstrap_state_sha256"], "DIAG4 bootstrap-state SHA"),
        _sha256(
            payload["initial_physical_state_sha256"],
            "DIAG4 initial-physical-state SHA",
        ),
        _sha256(payload["identity_sha256"], "DIAG4 identity SHA"),
        attempts,
        accepted_steps,
        retryable_rejections,
        terminal_status,
        quality_latch,
        corrections,
        corrections_sha256,
        individual_ratios,
        individual_sha256,
        path_ratios,
        path_sha256,
        steihaug_solve_calls,
        steihaug_solve_calls_sha256,
        outcomes_sha256,
        subtrial_counts,
        subtrial_count_sha256,
        selected_subtrial_indices,
        selected_subtrial_index_sha256,
        subtrial_float_matrices["subtrial_trust_radius"],
        subtrial_float_sha256["subtrial_trust_radius"],
        subtrial_outcomes,
        subtrial_outcome_sha256,
        subtrial_float_matrices["subtrial_actual_reduction"],
        subtrial_float_sha256["subtrial_actual_reduction"],
        subtrial_float_matrices["subtrial_predicted_reduction"],
        subtrial_float_sha256["subtrial_predicted_reduction"],
        subtrial_float_matrices["subtrial_maximum_individual_correction_step_ratio"],
        subtrial_float_sha256["subtrial_maximum_individual_correction_step_ratio"],
        subtrial_float_matrices["subtrial_correction_path_step_ratio"],
        subtrial_float_sha256["subtrial_correction_path_step_ratio"],
        subtrial_float_matrices["subtrial_corrected_radius_ratio"],
        subtrial_float_sha256["subtrial_corrected_radius_ratio"],
        subtrial_integer_matrices["subtrial_steihaug_iterations"],
        subtrial_integer_sha256["subtrial_steihaug_iterations"],
        subtrial_integer_matrices["subtrial_steihaug_hvp_evaluations"],
        subtrial_integer_sha256["subtrial_steihaug_hvp_evaluations"],
        subtrial_integer_matrices["subtrial_steihaug_solve_calls"],
        subtrial_integer_sha256["subtrial_steihaug_solve_calls"],
        subtrial_integer_matrices["subtrial_total_hvp_evaluations"],
        subtrial_integer_sha256["subtrial_total_hvp_evaluations"],
        subtrial_integer_matrices["subtrial_nonlinear_corrections"],
        subtrial_integer_sha256["subtrial_nonlinear_corrections"],
        subtrial_integer_matrices["subtrial_joint_evaluations"],
        subtrial_integer_sha256["subtrial_joint_evaluations"],
        subtrial_integer_matrices["subtrial_joint_linearizations"],
        subtrial_integer_sha256["subtrial_joint_linearizations"],
        subtrial_integer_matrices["subtrial_joint_value_evaluations"],
        subtrial_integer_sha256["subtrial_joint_value_evaluations"],
        subtrial_integer_matrices["subtrial_objective_residual_linearizations"],
        subtrial_integer_sha256["subtrial_objective_residual_linearizations"],
        subtrial_integer_matrices["subtrial_gram_factorizations"],
        subtrial_integer_sha256["subtrial_gram_factorizations"],
        subtrial_integer_matrices["subtrial_gram_solves"],
        subtrial_integer_sha256["subtrial_gram_solves"],
        declared_subtrial_summary,
    )


def validate_safeguard_telemetry_payload(
    value: JsonValue,
    *,
    history: HistoryEvidence | None = None,
    expected_history_evidence: ArtifactRef | None = None,
) -> SafeguardTelemetryV4:
    """Parse only the frozen DIAG4 safeguard generation."""

    return _validate_safeguard_telemetry_payload(
        value,
        history=history,
        expected_history_evidence=expected_history_evidence,
        schema_version=DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
        route=DIAG4_ROUTE,
        plan_sha256=DIAG4_PLAN_SHA256,
        evidence_type=SafeguardTelemetryV4,
    )


def _validate_trace_free_producer_payload(
    value: JsonValue,
    *,
    mode: str,
    route: str,
    plan_sha256: str,
    preflight_schema_version: str,
    cold_schema_version: str,
    numerical_bundle_schema_version: str,
) -> dict[str, JsonValue]:
    """Validate one explicitly selected trace-free producer generation."""

    payload = _mapping(value, f"{mode} DIAG4 producer")
    if mode == "preflight":
        if route == DIAG5_ROUTE and payload.get("execution_status") in {
            "COMPILE_FAILURE",
            "COMPILE_OOM",
        }:
            failure_keys = frozenset(
                {
                    "schema_version",
                    "route",
                    "numerical_route",
                    "numerical_result_schema_version",
                    "plan_sha256",
                    "mode",
                    "execution_status",
                    "runtime",
                    "runtime_evidence",
                    "campaign_authorized",
                    "solver_dispatched",
                    "finalizer_called",
                    "endpoint_audit_called",
                    "profiler_enabled",
                    "profiler_start_calls",
                    "profiler_stop_calls",
                    "trace_normalization_calls",
                    "timing",
                    "failure_reasons",
                }
            )
            _exact_keys(payload, failure_keys, "preflight DIAG5 failure producer")
            if (
                payload["schema_version"] != preflight_schema_version
                or payload["route"] != route
                or payload["numerical_route"] != DIAG5_NUMERICAL_ROUTE
                or payload["numerical_result_schema_version"]
                != DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION
                or payload["plan_sha256"] != plan_sha256
                or payload["mode"] != "TRACE_FREE_COMPILE_ONLY"
                or any(
                    _boolean(payload[name], f"preflight DIAG5 producer.{name}")
                    for name in (
                        "campaign_authorized",
                        "solver_dispatched",
                        "finalizer_called",
                        "endpoint_audit_called",
                        "profiler_enabled",
                    )
                )
                or any(
                    _integer(payload[name], f"preflight DIAG5 producer.{name}") != 0
                    for name in (
                        "profiler_start_calls",
                        "profiler_stop_calls",
                        "trace_normalization_calls",
                    )
                )
                or len(_array(payload["failure_reasons"], "failure reasons")) != 1
            ):
                raise ValueError("preflight DIAG5 failure producer differs")
            _runtime_mapping(payload["runtime"], "preflight DIAG5 failure runtime")
            _artifact_ref(payload["runtime_evidence"], "DIAG5 runtime evidence")
            timing = _mapping(payload["timing"], "preflight DIAG5 failure timing")
            _exact_keys(
                timing,
                frozenset(
                    {
                        "compile_started_ns",
                        "compile_completed_ns",
                        "process_seconds_before_serialization",
                    }
                ),
                "preflight DIAG5 failure timing",
            )
            if (
                _integer(timing["compile_started_ns"], "compile start")
                >= _integer(timing["compile_completed_ns"], "compile completion")
                or _number(
                    timing["process_seconds_before_serialization"],
                    "process seconds before serialization",
                )
                <= 0.0
            ):
                raise ValueError("preflight DIAG5 failure timing differs")
            return payload
        keys = frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "numerical_result_schema_version",
                "plan_sha256",
                "mode",
                "execution_status",
                "runtime",
                "runtime_evidence",
                "base_neq_gntr1_policy_sha256",
                "policy_evidence",
                "problem_sha256",
                "optimizer_options_sha256",
                "scaling_sha256",
                "bootstrap_state_sha256",
                "initial_physical_state_sha256",
                "identity_sha256",
                "source_manifest_sha256",
                "state_size",
                "equality_size",
                "residual_size",
                "campaign_authorized",
                "solver_dispatched",
                "finalizer_called",
                "endpoint_audit_called",
                "python_callbacks",
                "profiler_enabled",
                "profiler_start_calls",
                "profiler_stop_calls",
                "trace_normalization_calls",
                "timing",
                "failure_reasons",
            }
        )
        _exact_keys(payload, keys, "preflight DIAG4 producer")
        if (
            payload["schema_version"] != preflight_schema_version
            or payload["route"] != route
            or payload["numerical_route"] != DIAG4_NUMERICAL_ROUTE
            or payload["numerical_result_schema_version"]
            != DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
            or payload["plan_sha256"] != plan_sha256
            or payload["mode"] != "TRACE_FREE_COMPILE_ONLY"
            or payload["execution_status"] != "SUCCESS"
            or any(
                _boolean(payload[name], f"preflight DIAG4 producer.{name}")
                for name in (
                    "campaign_authorized",
                    "solver_dispatched",
                    "finalizer_called",
                    "endpoint_audit_called",
                )
            )
            or _integer(payload["python_callbacks"], "preflight callbacks") != 0
            or _array(payload["failure_reasons"], "preflight failure reasons") != []
        ):
            raise ValueError("preflight DIAG4 producer identity differs")
        _validate_profiler_call_audit(payload, context="preflight DIAG4 producer")
        for field in ("runtime_evidence", "policy_evidence"):
            _artifact_ref(payload[field], f"preflight DIAG4 producer.{field}")
        for field in (
            "base_neq_gntr1_policy_sha256",
            "problem_sha256",
            "optimizer_options_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
            "source_manifest_sha256",
        ):
            _sha256(payload[field], f"preflight DIAG4 producer.{field}")
        if (
            _integer(payload["state_size"], "preflight state size") != STATE_SIZE
            or _integer(payload["equality_size"], "preflight equality size")
            != EQUALITY_SIZE
            or _integer(payload["residual_size"], "preflight residual size") != 2110
        ):
            raise ValueError("preflight DIAG4 dimensions differ")
        return payload
    if mode != "cold":
        raise ValueError("DIAG4 producer mode must be preflight or cold")
    keys = frozenset(
        {
            "schema_version",
            "numerical_bundle_schema_version",
            "route",
            "numerical_route",
            "numerical_result_schema_version",
            "plan_sha256",
            "execution_status",
            "runtime",
            "runtime_evidence",
            "base_neq_gntr1_policy_sha256",
            "policy_evidence",
            "problem_sha256",
            "optimizer_options_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
            "source_manifest_sha256",
            "history_evidence",
            "terminal_numerical_evidence",
            "solve_timing_evidence",
            "safeguard_telemetry_evidence",
            "profiler_enabled",
            "profiler_start_calls",
            "profiler_stop_calls",
            "trace_normalization_calls",
            "endpoint_audit_called",
            "campaign_authorized",
            "failure_reasons",
        }
    )
    _exact_keys(payload, keys, "cold DIAG4 producer")
    if (
        payload["schema_version"] != cold_schema_version
        or payload["numerical_bundle_schema_version"] != numerical_bundle_schema_version
        or payload["route"] != route
        or payload["numerical_route"] != DIAG4_NUMERICAL_ROUTE
        or payload["numerical_result_schema_version"]
        != DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
        or payload["plan_sha256"] != plan_sha256
        or payload["execution_status"] != "COMPLETE"
        or _boolean(payload["endpoint_audit_called"], "endpoint audit called")
        is not True
        or _boolean(payload["campaign_authorized"], "campaign authorized") is not False
        or _array(payload["failure_reasons"], "cold DIAG4 failures") != []
    ):
        raise ValueError("cold DIAG4 producer identity differs")
    _validate_profiler_call_audit(payload, context="cold DIAG4 producer")
    expected_paths = {
        "runtime_evidence": DIAG4_EVIDENCE_SLOT_PATHS["cold_runtime"],
        "policy_evidence": DIAG4_EVIDENCE_SLOT_PATHS["cold_policy"],
        "history_evidence": DIAG4_EVIDENCE_SLOT_PATHS["cold_history"],
        "terminal_numerical_evidence": DIAG4_EVIDENCE_SLOT_PATHS[
            "cold_terminal_numerical"
        ],
        "solve_timing_evidence": DIAG4_EVIDENCE_SLOT_PATHS["cold_solve_timing"],
        "safeguard_telemetry_evidence": DIAG4_EVIDENCE_SLOT_PATHS[
            "cold_safeguard_telemetry"
        ],
    }
    for field, expected_path in expected_paths.items():
        if _artifact_ref(payload[field], field).relative_path != expected_path:
            raise ValueError(f"cold DIAG4 {field} path differs")
    expected_schemas = {
        "terminal_numerical_evidence": (
            f"{DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal"
        ),
        "solve_timing_evidence": DIAG4_SOLVE_TIMING_SCHEMA_VERSION,
        "safeguard_telemetry_evidence": (DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION),
    }
    for field, expected_schema in expected_schemas.items():
        if _artifact_ref(payload[field], field).schema_version != expected_schema:
            raise ValueError(f"cold DIAG4 {field} schema differs")
    for field in (
        "base_neq_gntr1_policy_sha256",
        "problem_sha256",
        "optimizer_options_sha256",
        "scaling_sha256",
        "bootstrap_state_sha256",
        "initial_physical_state_sha256",
        "identity_sha256",
        "source_manifest_sha256",
    ):
        _sha256(payload[field], f"cold DIAG4 producer.{field}")
    return payload


def validate_diag4_producer_payload(
    value: JsonValue, *, mode: str
) -> dict[str, JsonValue]:
    """Validate only the frozen DIAG4 trace-free producer generation."""

    return _validate_trace_free_producer_payload(
        value,
        mode=mode,
        route=DIAG4_ROUTE,
        plan_sha256=DIAG4_PLAN_SHA256,
        preflight_schema_version=DIAG4_PREFLIGHT_SCHEMA_VERSION,
        cold_schema_version=DIAG4_COLD_RESULT_SCHEMA_VERSION,
        numerical_bundle_schema_version=DIAG4_NUMERICAL_BUNDLE_SCHEMA_VERSION,
    )


def diag4_execution_evidence_payload(
    *,
    supporting_evidence: Mapping[str, ArtifactRef],
    solve_timing: JsonValue,
    producer: JsonValue,
    process: JsonValue,
) -> dict[str, JsonValue]:
    """Derive the parent execution join from immutable child and process evidence."""

    timing = validate_solve_timing_evidence_payload(solve_timing)
    producer_payload = validate_diag4_producer_payload(producer, mode="cold")
    process_payload = _mapping(process, "DIAG4 cold process")
    supporting_names = DIAG4_EVIDENCE_SLOT_NAMES - frozenset(
        {"execution", "supervisor_terminal"}
    )
    if frozenset(supporting_evidence) != supporting_names:
        raise ValueError("DIAG4 execution supporting evidence keys differ")
    for name, payload_value in (
        ("cold_producer", producer_payload),
        ("cold_process", process_payload),
        ("cold_solve_timing", _mapping(solve_timing, "DIAG4 solve timing")),
    ):
        encoded = canonical_json_bytes(payload_value)
        reference = supporting_evidence[name]
        if reference.sha256 != hashlib.sha256(
            encoded
        ).hexdigest() or reference.size_bytes != len(encoded):
            raise ValueError(f"DIAG4 execution {name} bytes differ from its reference")
    producer_reference_fields = {
        "runtime_evidence": "cold_runtime",
        "policy_evidence": "cold_policy",
        "history_evidence": "cold_history",
        "terminal_numerical_evidence": "cold_terminal_numerical",
        "solve_timing_evidence": "cold_solve_timing",
        "safeguard_telemetry_evidence": "cold_safeguard_telemetry",
    }
    if any(
        _artifact_ref(producer_payload[field], f"cold DIAG4 producer.{field}")
        != supporting_evidence[name]
        for field, name in producer_reference_fields.items()
    ):
        raise ValueError("DIAG4 execution producer references differ")
    process_started = _integer(
        process_payload.get("process_started_monotonic_ns"),
        "DIAG4 process start",
    )
    process_stopped = _integer(
        process_payload.get("process_stopped_monotonic_ns"),
        "DIAG4 process stop",
    )
    if (
        _integer(process_payload.get("child_pid"), "DIAG4 process child PID")
        != timing.child_pid
        or _integer(
            process_payload.get("child_start_time_ticks"),
            "DIAG4 process child start ticks",
        )
        != timing.child_start_time_ticks
        or process_started != timing.process_started_monotonic_ns
        or not (
            process_started
            < timing.state_ready_monotonic_ns
            < timing.solve_started_monotonic_ns
            < timing.solve_stopped_monotonic_ns
            < timing.finalizer_completed_monotonic_ns
            < timing.endpoint_audit_completed_monotonic_ns
            < timing.serialization_started_monotonic_ns
            < process_stopped
        )
    ):
        raise ValueError("DIAG4 timing/process identity or containment differs")
    producer_identity = tuple(
        _sha256(producer_payload[name], f"cold DIAG4 producer.{name}")
        for name in (
            "problem_sha256",
            "optimizer_options_sha256",
            "base_neq_gntr1_policy_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
        )
    )
    timing_identity = (
        timing.problem_sha256,
        timing.optimizer_options_sha256,
        timing.base_neq_gntr1_policy_sha256,
        timing.scaling_sha256,
        timing.bootstrap_state_sha256,
        timing.initial_physical_state_sha256,
        timing.identity_sha256,
    )
    if (
        producer_identity != timing_identity
        or _sha256(
            producer_payload["source_manifest_sha256"],
            "cold DIAG4 producer source-manifest SHA",
        )
        != timing.source_manifest_sha256
        or timing.source_manifest_sha256
        != supporting_evidence["source_manifest"].sha256
    ):
        raise ValueError("DIAG4 timing/producer numerical identity differs")
    _validate_diag4_profiler_call_audit(producer_payload, context="cold DIAG4 producer")
    if (
        timing.profiler_start_calls,
        timing.profiler_stop_calls,
        timing.trace_normalization_calls,
    ) != (
        DIAG4_PROFILER_CALL_AUDIT.profiler_start_calls,
        DIAG4_PROFILER_CALL_AUDIT.profiler_stop_calls,
        DIAG4_PROFILER_CALL_AUDIT.trace_normalization_calls,
    ):
        raise ValueError("DIAG4 profiler call audit differs")
    return {
        "schema_version": DIAG4_EXECUTION_SCHEMA_VERSION,
        "route": DIAG4_ROUTE,
        "numerical_route": DIAG4_NUMERICAL_ROUTE,
        "numerical_result_schema_version": DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
        "plan_sha256": DIAG4_PLAN_SHA256,
        "supporting_evidence": {
            name: _artifact_ref_payload(supporting_evidence[name])
            for name in sorted(supporting_names)
        },
        "phase_attribution": "NOT_PRODUCED",
        **_profiler_call_audit_payload(),
        "process_started_monotonic_ns": process_started,
        "state_ready_monotonic_ns": timing.state_ready_monotonic_ns,
        "solve_started_monotonic_ns": timing.solve_started_monotonic_ns,
        "solve_stopped_monotonic_ns": timing.solve_stopped_monotonic_ns,
        "finalizer_completed_monotonic_ns": timing.finalizer_completed_monotonic_ns,
        "endpoint_audit_completed_monotonic_ns": (
            timing.endpoint_audit_completed_monotonic_ns
        ),
        "serialization_started_monotonic_ns": (
            timing.serialization_started_monotonic_ns
        ),
        "process_stopped_monotonic_ns": process_stopped,
        "synchronized_solve_seconds": timing.synchronized_solve_seconds,
    }


def validate_diag4_execution_evidence_payload(
    value: JsonValue,
    *,
    supporting_evidence: Mapping[str, ArtifactRef],
    solve_timing: JsonValue,
    producer: JsonValue,
    process: JsonValue,
) -> dict[str, JsonValue]:
    """Rebuild the complete parent execution join and reject authored claims."""

    payload = _mapping(value, "DIAG4 execution evidence")
    rebuilt = diag4_execution_evidence_payload(
        supporting_evidence=supporting_evidence,
        solve_timing=solve_timing,
        producer=producer,
        process=process,
    )
    if payload != rebuilt:
        raise ValueError("DIAG4 execution evidence differs from raw authorities")
    return payload


def validate_diag4_numerical_documents(
    *,
    history: JsonValue,
    solve_timing: JsonValue,
    safeguard_telemetry: JsonValue,
    terminal_numerical: JsonValue,
    producer: JsonValue,
    artifact_root: Path | None = None,
) -> tuple[
    HistoryEvidence,
    SolveTimingEvidenceV4,
    SafeguardTelemetryV4,
    dict[str, JsonValue],
]:
    """Deep-join the numerical documents before and after atomic commit."""

    try:
        history_evidence = _parse_history(history, defer_step_bounds=True)
        producer_payload = validate_diag4_producer_payload(producer, mode="cold")
        (
            _terminal_legacy,
            terminal_identity,
            _endpoint_state_sha256,
            _terminal_observables,
            _endpoint_terms,
            _endpoint_observables,
        ) = _validate_gntr3_terminal_numerical_structure(terminal_numerical)
        if artifact_root is not None:
            terminal_identity = validate_diag4_terminal_numerical_payload(
                artifact_root, terminal_numerical
            ).numerical_identity
    except (TypeError, ValueError) as error:
        raise Diag4NumericalDocumentError(
            FailureReasonCodeV4.PENDING_RESULT_INVALID, str(error)
        ) from error
    try:
        timing_evidence = validate_solve_timing_evidence_payload(solve_timing)
    except (TypeError, ValueError) as error:
        raise Diag4NumericalDocumentError(
            FailureReasonCodeV4.TIMING_INVALID, str(error)
        ) from error
    history_reference = _artifact_ref(
        producer_payload["history_evidence"], "DIAG4 producer history evidence"
    )
    try:
        telemetry_evidence = validate_safeguard_telemetry_payload(
            safeguard_telemetry,
            history=history_evidence,
            expected_history_evidence=history_reference,
        )
    except (TypeError, ValueError) as error:
        raise Diag4NumericalDocumentError(
            FailureReasonCodeV4.SAFEGUARD_TELEMETRY_INVALID, str(error)
        ) from error
    producer_identity = (
        _string(producer_payload["numerical_route"], "DIAG4 producer numerical route"),
        _string(
            producer_payload["numerical_result_schema_version"],
            "DIAG4 producer numerical result schema",
        ),
        *(
            _sha256(producer_payload[name], f"DIAG4 producer.{name}")
            for name in (
                "problem_sha256",
                "optimizer_options_sha256",
                "base_neq_gntr1_policy_sha256",
                "scaling_sha256",
                "bootstrap_state_sha256",
                "initial_physical_state_sha256",
                "identity_sha256",
            )
        ),
    )
    timing_identity = (
        timing_evidence.numerical_route,
        timing_evidence.numerical_result_schema_version,
        timing_evidence.problem_sha256,
        timing_evidence.optimizer_options_sha256,
        timing_evidence.base_neq_gntr1_policy_sha256,
        timing_evidence.scaling_sha256,
        timing_evidence.bootstrap_state_sha256,
        timing_evidence.initial_physical_state_sha256,
        timing_evidence.identity_sha256,
    )
    telemetry_identity = (
        telemetry_evidence.numerical_route,
        telemetry_evidence.numerical_result_schema_version,
        telemetry_evidence.problem_sha256,
        telemetry_evidence.optimizer_options_sha256,
        telemetry_evidence.base_neq_gntr1_policy_sha256,
        telemetry_evidence.scaling_sha256,
        telemetry_evidence.bootstrap_state_sha256,
        telemetry_evidence.initial_physical_state_sha256,
        telemetry_evidence.identity_sha256,
    )
    terminal_identity_values = (
        terminal_identity.numerical_route,
        terminal_identity.numerical_result_schema_version,
        terminal_identity.problem_sha256,
        terminal_identity.optimizer_options_sha256,
        terminal_identity.base_neq_gntr1_policy_sha256,
        terminal_identity.scaling_sha256,
        terminal_identity.bootstrap_state_sha256,
        terminal_identity.initial_physical_state_sha256,
        terminal_identity.identity_sha256,
    )
    _validate_diag4_profiler_call_audit(producer_payload, context="cold DIAG4 producer")
    timing_call_counts = (
        timing_evidence.profiler_start_calls,
        timing_evidence.profiler_stop_calls,
        timing_evidence.trace_normalization_calls,
    )
    if (
        producer_identity != timing_identity
        or producer_identity != telemetry_identity
        or producer_identity != terminal_identity_values
        or _sha256(
            producer_payload["source_manifest_sha256"],
            "DIAG4 producer source-manifest SHA",
        )
        != timing_evidence.source_manifest_sha256
        or timing_call_counts
        != (
            DIAG4_PROFILER_CALL_AUDIT.profiler_start_calls,
            DIAG4_PROFILER_CALL_AUDIT.profiler_stop_calls,
            DIAG4_PROFILER_CALL_AUDIT.trace_normalization_calls,
        )
    ):
        raise Diag4NumericalDocumentError(
            FailureReasonCodeV4.NUMERICAL_IDENTITY_MISMATCH,
            "DIAG4 numerical document join differs",
        )
    return (
        history_evidence,
        timing_evidence,
        telemetry_evidence,
        producer_payload,
    )


def validate_history_evidence_payload(
    value: JsonValue, *, defer_step_bounds: bool = False
) -> HistoryEvidence:
    """Public typed parser shared by GPU receipts and CPU qualification."""

    return _parse_history(value, defer_step_bounds=defer_step_bounds)


def validate_policy_evidence_payload(
    value: JsonValue, *, terminal: TerminalEvidence | None = None
) -> PolicyEvidence:
    """Public typed parser for the frozen scientific policy."""

    return _parse_policy(value, terminal)


def validate_terminal_numerical_payload(
    artifact_root: Path, value: JsonValue
) -> TerminalEvidence:
    """Public legacy terminal parser retained for v1-v3 consumers."""

    return _parse_terminal(artifact_root, value)


def validate_terminal_endpoint_audit(
    *, terminal: TerminalEvidenceV4, endpoint_audit: NativeEquivalentEndpointAudit
) -> None:
    """Join the independent endpoint audit to the sole GNTR3 terminal artifact."""

    terminal_values = terminal.terminal
    if (
        endpoint_audit.audited_state_sha256 != terminal.endpoint_state_sha256
        or endpoint_audit.gpu_quality.physical_objective != terminal_values.objective
        or endpoint_audit.gpu_quality.gpu_raw_objective_terms
        != tuple(value for _, value in terminal.endpoint_objective_terms)
        or endpoint_audit.gpu_quality.gpu_raw_equalities
        != tuple(float(item) for item in terminal_values.array("raw_equalities").values)
        or endpoint_audit.gpu_quality.constraint_inverse_scale
        != tuple(
            float(item)
            for item in terminal_values.array("constraint_inverse_scale").values
        )
    ):
        raise ValueError("GNTR3 endpoint audit differs from terminal evidence")


def validate_endpoint_audit_evidence_payload(
    value: JsonValue,
) -> NativeEquivalentEndpointAudit:
    """Parse structurally valid endpoint evidence without requiring a HIT."""

    evidence = endpoint_audit_from_payload(value)
    if canonical_json_bytes(endpoint_audit_payload(evidence)) != canonical_json_bytes(
        value
    ):
        raise ValueError("GNTR3 endpoint audit canonical round trip differs")
    return evidence


def build_native_equivalent_scientific_evidence(
    *,
    history: HistoryEvidence,
    safeguard_telemetry: SafeguardTelemetryV4,
    terminal: TerminalEvidenceV4,
    policy: PolicyEvidence,
    endpoint_audit: NativeEquivalentEndpointAudit,
    expected_numerical_identity: NativeEquivalentNumericalIdentity,
    backend: str = "cpu",
) -> NativeEquivalentScientificEvidence:
    """Derive scientific disposition without GPU process or receipt assumptions."""

    if backend != "cpu":
        raise ValueError("native-equivalent scientific evidence backend must be cpu")
    expected = expected_numerical_identity
    if (
        expected.numerical_route != DIAG4_NUMERICAL_ROUTE
        or expected.numerical_result_schema_version
        != DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("GNTR3 expected numerical identity route differs")
    telemetry_identity = NativeEquivalentNumericalIdentity(
        safeguard_telemetry.numerical_route,
        safeguard_telemetry.numerical_result_schema_version,
        safeguard_telemetry.problem_sha256,
        safeguard_telemetry.optimizer_options_sha256,
        safeguard_telemetry.base_neq_gntr1_policy_sha256,
        safeguard_telemetry.scaling_sha256,
        safeguard_telemetry.bootstrap_state_sha256,
        safeguard_telemetry.initial_physical_state_sha256,
        safeguard_telemetry.identity_sha256,
    )
    if telemetry_identity != expected or terminal.numerical_identity != expected:
        raise ValueError("GNTR3 scientific numerical identity differs")
    if (
        safeguard_telemetry.loop_attempts != history.attempts
        or safeguard_telemetry.accepted_steps != history.accepted_steps
        or safeguard_telemetry.retryable_rejections != history.retryable_rejections
        or safeguard_telemetry.terminal_status is not history.status
        or safeguard_telemetry.quality_latch is not history.quality_latch
    ):
        raise ValueError("GNTR3 safeguard telemetry differs from history")
    validate_terminal_endpoint_audit(terminal=terminal, endpoint_audit=endpoint_audit)
    terminal_values = terminal.terminal
    _validate_terminal_raw_evidence(terminal_values, history, policy)
    _validate_quality_replay(history, terminal_values, policy)
    quality = _quality(terminal_values)
    numerical_complete = bool(
        not history.fatal
        and history.attempts > 0
        and (
            history.attempts == MAXIMUM_ATTEMPTS
            or history.accepted_steps == MAXIMUM_ACCEPTED_STEPS
            or history.quality_latch
        )
        and _terminal_semantics(history, terminal_values)
        and quality.residual_value_margin >= 0.0
        and quality.residual_gradient_margin >= 0.0
        and quality.transpose_margin >= 0.0
    )
    quality_hit = bool(
        numerical_complete
        and history.quality_latch
        and quality.passes
        and endpoint_audit.passes()
        and history.first_quality_attempt > 0
        and history.first_quality_accepted_step > 0
    )
    outcome = (
        ScientificOutcome.QUALITY_HIT
        if quality_hit
        else ScientificOutcome.NO_HIT
        if numerical_complete
        else ScientificOutcome.INCOMPLETE
    )
    return NativeEquivalentScientificEvidence(
        backend,
        expected,
        history,
        safeguard_telemetry,
        terminal,
        policy,
        endpoint_audit,
        quality,
        outcome,
    )


def validate_native_equivalent_scientific_evidence(
    *,
    artifact_root: Path,
    history: JsonValue,
    safeguard_telemetry: JsonValue,
    terminal_numerical: JsonValue,
    policy: JsonValue,
    endpoint_audit: JsonValue,
    expected_history_evidence: ArtifactRef,
    expected_numerical_identity: NativeEquivalentNumericalIdentity,
    backend: str = "cpu",
) -> NativeEquivalentScientificEvidence:
    """Parse and join the complete CPU-usable GNTR3 scientific evidence set."""

    history_evidence = validate_history_evidence_payload(
        history, defer_step_bounds=True
    )
    terminal_evidence = _validate_gntr3_terminal_numerical_payload(
        artifact_root, terminal_numerical
    )
    policy_evidence = validate_policy_evidence_payload(
        policy, terminal=terminal_evidence.terminal
    )
    telemetry_evidence = validate_safeguard_telemetry_payload(
        safeguard_telemetry,
        history=history_evidence,
        expected_history_evidence=expected_history_evidence,
    )
    endpoint_evidence = validate_endpoint_audit_evidence_payload(endpoint_audit)
    return build_native_equivalent_scientific_evidence(
        history=history_evidence,
        safeguard_telemetry=telemetry_evidence,
        terminal=terminal_evidence,
        policy=policy_evidence,
        endpoint_audit=endpoint_evidence,
        expected_numerical_identity=expected_numerical_identity,
        backend=backend,
    )


def build_diag2_compile_failure_producer_payload(
    *,
    mode: str,
    execution_status: str,
    runtime: Mapping[str, JsonValue],
    runtime_evidence: ArtifactRef,
    compile_started_ns: int,
    compile_completed_ns: int,
    process_seconds_before_serialization: float,
    failure_reasons: tuple[str, ...],
) -> dict[str, JsonValue]:
    """Construct the sole minimum-typed producer variant for compile failure/OOM."""

    common: dict[str, JsonValue] = {
        "schema_version": (
            "single-stage-neq-gntr1-preflight-worker-v1"
            if mode == "preflight"
            else f"{SCHEMA_VERSION}-producer"
        ),
        "route": DIAG2_ROUTE,
        "plan_sha256": DIAG2_PLAN_SHA256,
        "execution_status": execution_status,
        "runtime": dict(runtime),
        "runtime_evidence": _artifact_ref_payload(runtime_evidence),
        "timing": {
            "compile_started_ns": compile_started_ns,
            "compile_completed_ns": compile_completed_ns,
            "process_seconds_before_serialization": process_seconds_before_serialization,
        },
        "failure_reasons": list(failure_reasons),
    }
    if mode == "preflight":
        common.update(
            {
                "mode": "ANNOTATED_LOWER_COMPILE_ONLY",
                "campaign_authorized": False,
                "solver_dispatched": False,
                "finalizer_called": False,
                "endpoint_audit_called": False,
            }
        )
    return validate_diag2_producer_payload(common, mode=mode)


def classify_diag2_cold_evidence(
    artifact_root: Path,
    *,
    artifact_refs: Mapping[str, ArtifactRef | None],
    _producer_validator: Callable[..., dict[str, JsonValue]] | None = None,
) -> Diag2ColdEvidenceClassification:
    """Type the cold evidence prefix in SSOT offending-path order."""

    if _producer_validator is None:
        _producer_validator = validate_diag2_producer_payload
    if frozenset(artifact_refs) != DIAG2_EVIDENCE_SLOT_NAMES:
        raise ValueError("cold classifier refs differ from the frozen slot schema")
    producer_reference = artifact_refs["cold_producer"]
    if producer_reference is None:
        raise ValueError("cold classifier requires a minimum-typed producer")
    try:
        producer = _producer_validator(
            _load_ref_json(artifact_root, producer_reference, "cold producer"),
            mode="cold",
        )
    except (OSError, TypeError, ValueError) as error:
        return Diag2ColdEvidenceClassification(
            (),
            StructuredFailureV2(
                FailureStageV2.COLD_PROTOCOL_FAILURE,
                FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
                hashlib.sha256(str(error).encode()).hexdigest(),
            ),
            "cold_producer",
        )
    producer_bindings = {
        "runtime_evidence": "cold_runtime",
        "policy_evidence": "cold_policy",
        "history_evidence": "cold_history",
        "terminal_numerical_evidence": "cold_terminal_numerical",
        "raw_trace_evidence": "cold_raw_trace",
        "trace_intervals_evidence": "cold_trace_intervals",
    }
    try:
        for field, slot_name in producer_bindings.items():
            reference = artifact_refs[slot_name]
            if (
                reference is not None
                and _artifact_ref(producer[field], f"cold producer.{field}")
                != reference
            ):
                raise ValueError(f"cold producer {field} binding differs")
    except (TypeError, ValueError) as error:
        return Diag2ColdEvidenceClassification(
            (),
            StructuredFailureV2(
                FailureStageV2.COLD_PROTOCOL_FAILURE,
                FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
                hashlib.sha256(str(error).encode()).hexdigest(),
            ),
            "cold_producer",
        )
    ordered = (
        "cold_runtime",
        "cold_policy",
        "cold_history",
        "cold_terminal_numerical",
        "cold_raw_trace",
        "cold_trace_intervals",
        "execution",
    )
    typed: list[str] = []
    terminal: TerminalEvidence | None = None
    for name in ordered:
        reference = artifact_refs[name]
        if reference is None:
            reason = (
                FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID
                if name == "cold_runtime"
                else (
                    FailureReasonCodeV2.POLICY_SCHEMA_INVALID
                    if name == "cold_policy"
                    else FailureReasonCodeV2.NUMERICAL_SCHEMA_INVALID
                )
            )
            stage = (
                FailureStageV2.COLD_PROTOCOL_FAILURE
                if name in {"cold_runtime", "cold_policy"}
                else FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE
            )
            return Diag2ColdEvidenceClassification(
                tuple(typed),
                StructuredFailureV2(
                    stage,
                    reason,
                    hashlib.sha256(f"missing:{name}".encode()).hexdigest(),
                ),
                name,
            )
        try:
            if name == "cold_runtime":
                snapshot = validate_diag2_source_snapshot_authority(artifact_root)
                validate_runtime_evidence(
                    _resolve_artifact(artifact_root, reference),
                    snapshot_root=snapshot.root,
                    campaign_root=artifact_root,
                )
            elif name == "cold_policy":
                _parse_policy(_load_ref_json(artifact_root, reference, "cold policy"))
            elif name == "cold_history":
                _parse_history(_load_ref_json(artifact_root, reference, "cold history"))
            elif name == "cold_terminal_numerical":
                terminal = _parse_terminal(
                    artifact_root,
                    _load_ref_json(artifact_root, reference, "cold terminal numerical"),
                )
            elif name == "cold_raw_trace":
                normalize_chrome_trace(
                    _resolve_artifact(artifact_root, reference),
                    phase_schema_sha256=PHASE_SCHEMA_SHA256,
                )
            elif name == "cold_trace_intervals":
                _parse_phases(
                    _load_ref_json(artifact_root, reference, "cold trace intervals"),
                    PHASE_SCHEMA_SHA256,
                )
            else:
                refs = _diag2_present_refs(
                    {
                        slot_name: EvidenceSlot.present(slot_reference)
                        for slot_name, slot_reference in artifact_refs.items()
                        if slot_reference is not None
                    }
                )
                _parse_execution(
                    _load_ref_json(artifact_root, reference, "execution"), refs
                )
            typed.append(name)
        except (OSError, TypeError, ValueError) as error:
            reason = (
                FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID
                if name == "cold_runtime"
                else (
                    FailureReasonCodeV2.POLICY_SCHEMA_INVALID
                    if name == "cold_policy"
                    else (
                        FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED
                        if name in {"cold_raw_trace", "cold_trace_intervals"}
                        else FailureReasonCodeV2.NUMERICAL_SCHEMA_INVALID
                    )
                )
            )
            stage = (
                FailureStageV2.COLD_PROTOCOL_FAILURE
                if name in {"cold_runtime", "cold_policy"}
                else FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE
            )
            return Diag2ColdEvidenceClassification(
                tuple(typed),
                StructuredFailureV2(
                    stage, reason, hashlib.sha256(str(error).encode()).hexdigest()
                ),
                name,
            )
    if terminal is None:
        raise AssertionError("cold terminal prefix narrowing failed")
    return Diag2ColdEvidenceClassification(tuple(typed), None, None)


def classify_diag3_cold_evidence(
    artifact_root: Path,
    *,
    artifact_refs: Mapping[str, ArtifactRef | None],
) -> Diag2ColdEvidenceClassification:
    return classify_diag2_cold_evidence(
        artifact_root,
        artifact_refs=artifact_refs,
        _producer_validator=validate_diag3_producer_payload,
    )


def _diag4_cold_slot_failure(
    name: str,
) -> tuple[FailureStageV4, FailureReasonCodeV4]:
    if name in {"cold_runtime", "cold_policy"}:
        return FailureStageV4.COLD, FailureReasonCodeV4.COLD_PROTOCOL_INVALID
    if name == "cold_solve_timing":
        return FailureStageV4.NUMERICAL_COMMIT, FailureReasonCodeV4.TIMING_INVALID
    if name == "cold_safeguard_telemetry":
        return (
            FailureStageV4.NUMERICAL_COMMIT,
            FailureReasonCodeV4.SAFEGUARD_TELEMETRY_INVALID,
        )
    if name == "execution":
        return FailureStageV4.RECEIPT, FailureReasonCodeV4.EVIDENCE_VECTOR_INVALID
    return FailureStageV4.NUMERICAL_COMMIT, FailureReasonCodeV4.PENDING_RESULT_INVALID


def classify_diag4_cold_evidence(
    artifact_root: Path,
    *,
    artifact_refs: Mapping[str, ArtifactRef | None],
) -> Diag4ColdEvidenceClassification:
    """Type the trace-free committed prefix without any profiler fallback."""

    if frozenset(artifact_refs) != DIAG4_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG4 cold classifier refs differ from the frozen schema")
    producer_reference = artifact_refs["cold_producer"]
    if producer_reference is None:
        raise ValueError("DIAG4 cold classifier requires a typed producer")
    try:
        producer = validate_diag4_producer_payload(
            _load_ref_json(artifact_root, producer_reference, "DIAG4 cold producer"),
            mode="cold",
        )
        for field, slot_name in {
            "runtime_evidence": "cold_runtime",
            "policy_evidence": "cold_policy",
            "history_evidence": "cold_history",
            "terminal_numerical_evidence": "cold_terminal_numerical",
            "solve_timing_evidence": "cold_solve_timing",
            "safeguard_telemetry_evidence": "cold_safeguard_telemetry",
        }.items():
            reference = artifact_refs[slot_name]
            if (
                reference is not None
                and _artifact_ref(producer[field], f"DIAG4 producer.{field}")
                != reference
            ):
                raise ValueError(f"DIAG4 producer {field} binding differs")
    except (OSError, TypeError, ValueError) as error:
        return Diag4ColdEvidenceClassification(
            (),
            StructuredFailureV4(
                FailureStageV4.COLD,
                FailureReasonCodeV4.COLD_PRODUCER_INVALID,
                hashlib.sha256(str(error).encode()).hexdigest(),
            ),
            "cold_producer",
        )
    ordered = (
        "cold_runtime",
        "cold_policy",
        "cold_history",
        "cold_terminal_numerical",
        "cold_solve_timing",
        "cold_safeguard_telemetry",
        "execution",
    )
    typed: list[str] = []
    history: HistoryEvidence | None = None
    timing_payload: JsonValue | None = None
    for name in ordered:
        reference = artifact_refs[name]
        if reference is None:
            stage, reason = _diag4_cold_slot_failure(name)
            return Diag4ColdEvidenceClassification(
                tuple(typed),
                StructuredFailureV4(
                    stage,
                    reason,
                    hashlib.sha256(f"missing:{name}".encode()).hexdigest(),
                ),
                name,
            )
        try:
            if name == "cold_runtime":
                snapshot = load_snapshot(
                    artifact_root / "source-snapshot",
                    required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
                )
                validate_runtime_evidence(
                    _resolve_artifact(artifact_root, reference),
                    snapshot_root=snapshot.root,
                    campaign_root=artifact_root,
                    required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
                )
            elif name == "cold_policy":
                _parse_policy(_load_ref_json(artifact_root, reference, "DIAG4 policy"))
            elif name == "cold_history":
                history = _parse_history(
                    _load_ref_json(artifact_root, reference, "DIAG4 history"),
                    defer_step_bounds=True,
                )
            elif name == "cold_terminal_numerical":
                validate_diag4_terminal_numerical_payload(
                    artifact_root,
                    _load_ref_json(artifact_root, reference, "DIAG4 terminal"),
                )
            elif name == "cold_solve_timing":
                timing_payload = _load_ref_json(
                    artifact_root, reference, "DIAG4 solve timing"
                )
                validate_solve_timing_evidence_payload(timing_payload)
            elif name == "cold_safeguard_telemetry":
                if history is None:
                    raise AssertionError("DIAG4 history prefix narrowing failed")
                validate_safeguard_telemetry_payload(
                    _load_ref_json(artifact_root, reference, "DIAG4 telemetry"),
                    history=history,
                    expected_history_evidence=_present_reference(
                        artifact_refs["cold_history"], "cold_history"
                    ),
                )
            else:
                if timing_payload is None:
                    raise AssertionError("DIAG4 timing prefix narrowing failed")
                supporting = {
                    slot_name: _present_reference(slot_reference, slot_name)
                    for slot_name, slot_reference in artifact_refs.items()
                    if slot_name not in {"execution", "supervisor_terminal"}
                }
                validate_diag4_execution_evidence_payload(
                    _load_ref_json(artifact_root, reference, "DIAG4 execution"),
                    supporting_evidence=supporting,
                    solve_timing=timing_payload,
                    producer=producer,
                    process=_load_ref_json(
                        artifact_root,
                        _present_reference(
                            artifact_refs["cold_process"], "cold_process"
                        ),
                        "DIAG4 cold process",
                    ),
                )
            typed.append(name)
        except (OSError, TypeError, ValueError) as error:
            stage, reason = _diag4_cold_slot_failure(name)
            return Diag4ColdEvidenceClassification(
                tuple(typed),
                StructuredFailureV4(
                    stage, reason, hashlib.sha256(str(error).encode()).hexdigest()
                ),
                name,
            )
    return Diag4ColdEvidenceClassification(tuple(typed), None, None)


def _policy_evidence_from_authority(
    payload: Mapping[str, JsonValue],
) -> PolicyEvidence:
    v1_payload = {
        key: payload[key]
        for key in (
            "policy_sha256",
            "native_raw_equalities_sha256",
            "native_raw_equalities",
            "constraint_inverse_scale_sha256",
            "constraint_inverse_scale",
            "objective_target",
            "state_size",
            "equality_size",
            "objective_residual_size",
            "component_absolute_tolerance",
            "component_relative_tolerance",
            "scaled_feasibility_tolerance",
            "residual_value_defect_tolerance",
            "residual_gradient_defect_tolerance",
            "transpose_defect_tolerance",
            "gntr_options",
        )
    }
    v1_payload["schema_version"] = f"{SCHEMA_VERSION}-policy"
    return _parse_policy(v1_payload)


def _diag2_policy_evidence(payload: Mapping[str, JsonValue]) -> PolicyEvidence:
    """Compatibility wrapper for the frozen v2 private test surface."""

    return _policy_evidence_from_authority(payload)


_DIAG2_V1_REF_NAMES: Final = {
    "history": "cold_history",
    "terminal_numerical": "cold_terminal_numerical",
    "raw_trace": "cold_raw_trace",
    "trace_intervals": "cold_trace_intervals",
    "execution": "execution",
    "preflight": "preflight_producer",
    "preflight_child_terminal": "preflight_terminal",
    "preflight_memory": "preflight_memory",
    "preflight_memory_samples": "preflight_memory_samples",
    "preflight_process": "preflight_process",
    "preflight_runtime": "preflight_runtime",
    "preflight_policy": "preflight_policy",
    "policy_authority": "policy_authority",
    "producer": "cold_producer",
    "child_terminal": "cold_terminal",
    "runtime": "cold_runtime",
    "process": "cold_process",
    "memory": "cold_memory",
    "memory_samples": "cold_memory_samples",
    "source_manifest": "source_manifest",
    "native_reference": "native_reference",
    "policy": "cold_policy",
}
_DIAG2_AUTHORITY_GROUPS: Final = (
    ("SETUP_SOURCE", ("source_manifest", "frozen_numerical_subset")),
    ("SETUP_REFERENCE", ("native_reference",)),
    ("SETUP_POLICY", ("policy_authority",)),
    ("ZERO_PREFLIGHT", ("supervisor_before_preflight",)),
    (
        "PREFLIGHT",
        (
            "preflight_producer",
            "preflight_terminal",
            "preflight_process",
            "preflight_memory",
            "preflight_memory_samples",
            "preflight_runtime",
            "preflight_policy",
        ),
    ),
    ("ZERO_COLD", ("supervisor_before_cold",)),
    (
        "COLD_SUPERVISION",
        (
            "cold_producer",
            "cold_terminal",
            "cold_process",
            "cold_memory",
            "cold_memory_samples",
            "cold_runtime",
            "cold_policy",
        ),
    ),
    (
        "COLD_NUMERICAL",
        (
            "cold_history",
            "cold_terminal_numerical",
            "cold_raw_trace",
            "cold_trace_intervals",
            "execution",
        ),
    ),
)
_DIAG2_STAGE_GROUP: Final = {
    FailureStageV2.SOURCE_PUBLICATION_FAILURE: "SETUP_SOURCE",
    FailureStageV2.NATIVE_REFERENCE_FAILURE: "SETUP_REFERENCE",
    FailureStageV2.POLICY_AUTHORITY_FAILURE: "SETUP_POLICY",
    FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE: "ZERO_PREFLIGHT",
    FailureStageV2.PREFLIGHT_SOURCE_FAILURE: "PREFLIGHT",
    FailureStageV2.PREFLIGHT_SUPERVISOR_FAILURE: "PREFLIGHT",
    FailureStageV2.PREFLIGHT_TIMEOUT: "PREFLIGHT",
    FailureStageV2.PREFLIGHT_MONITOR_FAILURE: "PREFLIGHT",
    FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE: "PREFLIGHT",
    FailureStageV2.PREFLIGHT_COMPILE_FAILURE: "PREFLIGHT",
    FailureStageV2.PREFLIGHT_CRASH: "PREFLIGHT",
    FailureStageV2.PREFLIGHT_RESOURCE_FAILURE: "PREFLIGHT",
    FailureStageV2.GPU_ZERO_BEFORE_COLD_FAILURE: "ZERO_COLD",
    FailureStageV2.COLD_SOURCE_FAILURE: "COLD_SUPERVISION",
    FailureStageV2.COLD_SUPERVISOR_FAILURE: "COLD_SUPERVISION",
    FailureStageV2.COLD_TIMEOUT: "COLD_SUPERVISION",
    FailureStageV2.COLD_MONITOR_FAILURE: "COLD_SUPERVISION",
    FailureStageV2.COLD_PROTOCOL_FAILURE: "COLD_SUPERVISION",
    FailureStageV2.COLD_COMPILE_FAILURE: "COLD_SUPERVISION",
    FailureStageV2.COLD_CRASH: "COLD_SUPERVISION",
    FailureStageV2.COLD_RESOURCE_FAILURE: "COLD_SUPERVISION",
    FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE: "COLD_NUMERICAL",
}


def _diag2_child_documents(
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    *,
    mode: str,
    child_terminal_schema_version: str = DIAG2_CHILD_TERMINAL_SCHEMA_VERSION,
    process_schema_version: str = DIAG2_PROCESS_SCHEMA_VERSION,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], str, int]:
    terminal_ref = artifact_refs[f"{mode}_terminal"]
    process_ref = artifact_refs[f"{mode}_process"]
    if terminal_ref is None or process_ref is None:
        raise ValueError(f"launched {mode} child omits terminal/process evidence")
    terminal = _mapping(
        _load_ref_json(artifact_root, terminal_ref, f"{mode} terminal"),
        f"{mode} terminal",
    )
    _exact_keys(
        terminal,
        frozenset(
            {
                "schema_version",
                "terminal_status",
                "failure_reasons",
                "monitor_failure_kind",
            }
        ),
        f"{mode} terminal",
    )
    if terminal["schema_version"] != child_terminal_schema_version:
        raise ValueError(f"{mode} terminal schema differs")
    process = _mapping(
        _load_ref_json(artifact_root, process_ref, f"{mode} process"),
        f"{mode} process",
    )
    expected_process_keys = frozenset(
        {
            "schema_version",
            "monitor_failure_kind",
            "child_pid",
            "child_start_time_ticks",
            "argv",
            "stdout",
            "stderr",
            "process_seconds",
            "process_diagnostics",
            "pre_source_identity",
            "post_source_identity",
            "process_started_monotonic_ns",
            "process_stopped_monotonic_ns",
        }
    )
    _exact_keys(process, expected_process_keys, f"{mode} process")
    if process["schema_version"] != process_schema_version:
        raise ValueError(f"{mode} process schema differs")
    monitor_kind = _string(
        terminal["monitor_failure_kind"], f"{mode} terminal monitor kind"
    )
    if (
        monitor_kind not in {"NONE", "BINDING", "FINALIZATION"}
        or process["monitor_failure_kind"] != monitor_kind
    ):
        raise ValueError(f"{mode} terminal/process monitor kind differs")
    diagnostics = _mapping(
        process["process_diagnostics"], f"{mode} process diagnostics"
    )
    returncode = _integer(
        diagnostics.get("returncode"),
        f"{mode} process return code",
        minimum=-2147483648,
    )
    return terminal, process, monitor_kind, returncode


def classify_diag2_subordinate_child_outcome(
    artifact_root: Path,
    *,
    artifact_refs: Mapping[str, ArtifactRef | None],
    mode: str,
    _producer_validator: Callable[..., dict[str, JsonValue]] | None = None,
) -> FailureReasonCodeV2 | None:
    """Recompute the subordinate raw child outcome under an outer setup failure."""

    if _producer_validator is None:
        _producer_validator = validate_diag2_producer_payload
    if mode not in {"preflight", "cold"}:
        raise ValueError("subordinate child mode must be preflight or cold")
    terminal, process, monitor_kind, returncode = _diag2_child_documents(
        artifact_root, artifact_refs, mode=mode
    )
    terminal_status = _string(terminal["terminal_status"], f"{mode} terminal status")
    child_start_ticks = _integer(
        process["child_start_time_ticks"], f"{mode} child start ticks", minimum=0
    )
    memory = artifact_refs[f"{mode}_memory"]
    samples = artifact_refs[f"{mode}_memory_samples"]
    if (memory is None) != (samples is None):
        raise ValueError(f"{mode} monitor evidence pairing differs")
    producer_ref = artifact_refs[f"{mode}_producer"]
    if monitor_kind == "BINDING":
        if (
            child_start_ticks != 0
            or terminal_status != "MONITOR_FAILURE"
            or producer_ref is not None
            or memory is not None
        ):
            raise ValueError(f"{mode} binding outcome evidence differs")
        return FailureReasonCodeV2.MONITOR_BINDING_FAILED
    if child_start_ticks <= 0:
        raise ValueError(f"{mode} bound child start identity differs")
    if monitor_kind == "NONE" and memory is None:
        raise ValueError(f"{mode} successful monitor omits memory evidence")
    if monitor_kind == "FINALIZATION" and memory is not None:
        raise ValueError(f"{mode} finalization retains memory evidence")
    if terminal_status == "TIMEOUT":
        return FailureReasonCodeV2.CHILD_TIMEOUT
    if monitor_kind == "FINALIZATION":
        if producer_ref is not None:
            if returncode != 0 or terminal_status != "COMPLETE":
                raise ValueError(
                    f"{mode} finalization producer contradicts terminal/process"
                )
            _producer_validator(
                _load_ref_json(
                    artifact_root, producer_ref, f"{mode} finalization producer"
                ),
                mode=mode,
            )
        return FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
    if terminal_status == "CRASH" or returncode != 0:
        if terminal_status != "CRASH" or returncode == 0:
            raise ValueError(f"{mode} crash terminal/process evidence differs")
        return FailureReasonCodeV2.CHILD_EXIT_NONZERO
    if producer_ref is None:
        if terminal_status != "PROTOCOL_FAILURE":
            raise ValueError(
                f"{mode} absent producer contradicts terminal/process evidence"
            )
        stdout_ref = _artifact_ref(process["stdout"], f"{mode} process.stdout")
        stdout = _resolve_artifact(artifact_root, stdout_ref).read_bytes()
        try:
            decoded = load_canonical_json_bytes(stdout)
        except (UnicodeDecodeError, ValueError):
            return FailureReasonCodeV2.PRODUCER_DECODE_FAILED
        try:
            _producer_validator(decoded, mode=mode)
        except (TypeError, ValueError):
            return FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID
        return None
    producer = _producer_validator(
        _load_ref_json(artifact_root, producer_ref, f"{mode} producer"), mode=mode
    )
    status = _string(producer["execution_status"], f"{mode} producer status")
    if status == "COMPILE_OOM":
        expected_terminal = (
            "COMPLETE" if monitor_kind == "FINALIZATION" else "COMPILE_FAILURE"
        )
        if returncode != 0 or terminal_status != expected_terminal:
            raise ValueError(f"{mode} compile-OOM terminal/process evidence differs")
        return FailureReasonCodeV2.CHILD_COMPILE_OOM
    if status == "COMPILE_FAILURE":
        expected_terminal = (
            "COMPLETE" if monitor_kind == "FINALIZATION" else "COMPILE_FAILURE"
        )
        if returncode != 0 or terminal_status != expected_terminal:
            raise ValueError(f"{mode} compile terminal/process evidence differs")
        return FailureReasonCodeV2.CHILD_COMPILE_FAILED
    if status == "TRACE_NORMALIZATION_FAILED":
        if terminal_status != "COMPLETE" or returncode != 0:
            raise ValueError(
                f"{mode} trace-normalization result contradicts terminal/process"
            )
        return FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED
    if terminal_status != "COMPLETE" or returncode != 0:
        raise ValueError(f"{mode} success producer contradicts terminal/process")
    if memory is None:
        raise ValueError(f"{mode} success producer omits memory evidence")
    memory_payload = _mapping(
        _load_ref_json(artifact_root, memory, f"{mode} memory"), f"{mode} memory"
    )
    if (
        _number(memory_payload.get("peak_memory_fraction"), f"{mode} memory fraction")
        >= 0.8
    ):
        return FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED
    return None


def classify_diag3_subordinate_child_outcome(
    artifact_root: Path,
    *,
    artifact_refs: Mapping[str, ArtifactRef | None],
    mode: str,
) -> FailureReasonCodeV2 | None:
    """Recompute a successor child outcome without widening the v2 parser."""

    return classify_diag2_subordinate_child_outcome(
        artifact_root,
        artifact_refs=artifact_refs,
        mode=mode,
        _producer_validator=validate_diag3_producer_payload,
    )


def _derive_diag2_subordinate_child_slots(
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    *,
    mode: str,
    _classifier: Callable[..., FailureReasonCodeV2 | None] | None = None,
) -> dict[str, EvidenceSlot]:
    if _classifier is None:
        _classifier = classify_diag2_subordinate_child_outcome
    outcome = _classifier(artifact_root, artifact_refs=artifact_refs, mode=mode)
    terminal, process, monitor_kind, returncode = _diag2_child_documents(
        artifact_root, artifact_refs, mode=mode
    )
    del terminal, process
    slots: dict[str, EvidenceSlot] = {}
    for suffix in ("terminal", "process"):
        name = f"{mode}_{suffix}"
        slots[name] = EvidenceSlot.present(
            _present_reference(artifact_refs[name], name)
        )
    monitor_reason = {
        "BINDING": AbsenceReason.MONITOR_BINDING_FAILED,
        "FINALIZATION": AbsenceReason.MONITOR_FINALIZATION_FAILED,
    }.get(monitor_kind)
    for suffix in ("memory", "memory_samples"):
        name = f"{mode}_{suffix}"
        reference = artifact_refs[name]
        if reference is not None:
            slots[name] = EvidenceSlot.present(reference)
        elif monitor_reason is not None:
            slots[name] = EvidenceSlot.absent(monitor_reason)
        else:
            raise ValueError(f"{mode} monitor evidence is absent without failure")
    producer_name = f"{mode}_producer"
    producer = artifact_refs[producer_name]
    producer_required = outcome in {
        None,
        FailureReasonCodeV2.CHILD_COMPILE_FAILED,
        FailureReasonCodeV2.CHILD_COMPILE_OOM,
        FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED,
        FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED,
    } or (
        outcome is FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
        and producer is not None
        and returncode == 0
    )
    if producer_required:
        slots[producer_name] = EvidenceSlot.present(
            _present_reference(producer, producer_name)
        )
    else:
        if producer is not None:
            raise ValueError(f"{mode} subordinate outcome retains forbidden producer")
        if outcome is None:
            raise AssertionError("subordinate producer absence narrowing failed")
        slots[producer_name] = EvidenceSlot.absent(AbsenceReason(outcome.value))
    documents_required = outcome in {
        None,
        FailureReasonCodeV2.CHILD_COMPILE_FAILED,
        FailureReasonCodeV2.CHILD_COMPILE_OOM,
        FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED,
        FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED,
    } or (
        outcome is FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
        and producer is not None
        and returncode == 0
    )
    for suffix, schema_reason in (
        ("runtime", AbsenceReason.RUNTIME_SCHEMA_INVALID),
        ("policy", AbsenceReason.POLICY_SCHEMA_INVALID),
    ):
        name = f"{mode}_{suffix}"
        reference = artifact_refs[name]
        if reference is not None:
            slots[name] = EvidenceSlot.present(reference)
        elif (
            artifact_root / DIAG2_EVIDENCE_SLOT_PATHS[name]
        ).is_file() or documents_required:
            slots[name] = EvidenceSlot.absent(schema_reason)
        else:
            if outcome is None:
                raise AssertionError("subordinate document absence narrowing failed")
            slots[name] = EvidenceSlot.absent(AbsenceReason(outcome.value))
    return slots


def derive_diag2_evidence_slots(
    *,
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    failure: StructuredFailureV2 | None,
    _slot_validator: Callable[..., None] | None = None,
    _subordinate_classifier: Callable[..., FailureReasonCodeV2 | None] | None = None,
) -> dict[str, EvidenceSlot]:
    """Derive the total slot vector from physical refs and one structured outcome."""

    if _slot_validator is None:
        _slot_validator = _validate_diag2_slots
    if _subordinate_classifier is None:
        _subordinate_classifier = classify_diag2_subordinate_child_outcome
    if frozenset(artifact_refs) != DIAG2_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG2 artifact refs differ from the frozen slot schema")
    if failure is None:
        if any(reference is None for reference in artifact_refs.values()):
            raise ValueError("complete DIAG2 evidence requires every artifact ref")
        return {
            name: EvidenceSlot.present(reference)
            for name, reference in artifact_refs.items()
            if reference is not None
        }
    if failure.reason not in DIAG2_STAGE_REASON_CODES[failure.stage]:
        raise ValueError("DIAG2 failure stage/reason pairing differs")
    if failure.stage in _DIAG2_INITIAL_SETUP_STAGE:
        offending = _DIAG2_INITIAL_SETUP_SLOT[failure.reason]
        setup_order = tuple(_DIAG2_POSTLAUNCH_SETUP_REASON)
        offending_index = setup_order.index(offending)
        setup_slots: dict[str, EvidenceSlot] = {}
        for index, name in enumerate(setup_order):
            reference = artifact_refs[name]
            if index < offending_index:
                if reference is None:
                    raise ValueError(
                        f"initial setup failure omits earlier authority: {name}"
                    )
                setup_slots[name] = EvidenceSlot.present(reference)
            elif index == offending_index:
                setup_slots[name] = (
                    EvidenceSlot.present(reference)
                    if reference is not None
                    else EvidenceSlot.absent(AbsenceReason(failure.reason.value))
                )
            else:
                if reference is not None:
                    raise ValueError(
                        f"initial setup failure retains later authority: {name}"
                    )
                setup_slots[name] = EvidenceSlot.absent(AbsenceReason.NOT_REACHED)
        for _, names in _DIAG2_AUTHORITY_GROUPS[3:]:
            for name in names:
                if artifact_refs[name] is not None:
                    raise ValueError(
                        f"initial setup failure retains downstream authority: {name}"
                    )
                setup_slots[name] = EvidenceSlot.absent(AbsenceReason.NOT_REACHED)
        terminal = artifact_refs["supervisor_terminal"]
        if terminal is None:
            raise ValueError("handled DIAG2 evidence omits supervisor terminal")
        setup_slots["supervisor_terminal"] = EvidenceSlot.present(terminal)
        _slot_validator(artifact_root, setup_slots, failure=failure)
        return setup_slots
    if (
        failure.stage
        in {
            FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            FailureStageV2.COLD_SOURCE_FAILURE,
        }
        and failure.reason in _DIAG2_POSTLAUNCH_SETUP_SLOT
    ):
        offending = _DIAG2_POSTLAUNCH_SETUP_SLOT[failure.reason]
        setup_order = tuple(_DIAG2_POSTLAUNCH_SETUP_REASON)
        offending_index = setup_order.index(offending)
        preserved_groups = (
            {
                "SETUP_SOURCE",
                "SETUP_REFERENCE",
                "SETUP_POLICY",
                "ZERO_PREFLIGHT",
                "PREFLIGHT",
            }
            if failure.stage is FailureStageV2.PREFLIGHT_SOURCE_FAILURE
            else {
                "SETUP_SOURCE",
                "SETUP_REFERENCE",
                "SETUP_POLICY",
                "ZERO_PREFLIGHT",
                "PREFLIGHT",
                "ZERO_COLD",
                "COLD_SUPERVISION",
            }
        )
        subordinate_group = (
            "PREFLIGHT"
            if failure.stage is FailureStageV2.PREFLIGHT_SOURCE_FAILURE
            else "COLD_SUPERVISION"
        )
        subordinate_mode = (
            "preflight"
            if failure.stage is FailureStageV2.PREFLIGHT_SOURCE_FAILURE
            else "cold"
        )
        subordinate_slots = _derive_diag2_subordinate_child_slots(
            artifact_root,
            artifact_refs,
            mode=subordinate_mode,
            _classifier=_subordinate_classifier,
        )
        drift_slots: dict[str, EvidenceSlot] = {}
        for group_name, names in _DIAG2_AUTHORITY_GROUPS:
            if group_name == subordinate_group:
                drift_slots.update(subordinate_slots)
                continue
            for name in names:
                reference = artifact_refs[name]
                if name in _DIAG2_POSTLAUNCH_SETUP_REASON:
                    setup_index = setup_order.index(name)
                    if setup_index < offending_index:
                        if reference is None:
                            raise ValueError(
                                f"post-launch setup drift omits earlier authority: {name}"
                            )
                        drift_slots[name] = EvidenceSlot.present(reference)
                    elif reference is not None:
                        drift_slots[name] = EvidenceSlot.present(reference)
                    else:
                        reason = _DIAG2_POSTLAUNCH_SETUP_REASON[name]
                        drift_slots[name] = EvidenceSlot.absent(
                            AbsenceReason(reason.value)
                        )
                elif group_name in preserved_groups:
                    if reference is None:
                        raise ValueError(
                            f"post-launch setup drift omits preserved authority: {name}"
                        )
                    drift_slots[name] = EvidenceSlot.present(reference)
                else:
                    if reference is not None:
                        raise ValueError(
                            f"post-launch setup drift retains downstream authority: {name}"
                        )
                    drift_slots[name] = EvidenceSlot.absent(AbsenceReason.NOT_REACHED)
        terminal = artifact_refs["supervisor_terminal"]
        if terminal is None:
            raise ValueError("handled DIAG2 evidence omits supervisor terminal")
        drift_slots["supervisor_terminal"] = EvidenceSlot.present(terminal)
        if failure.stage is FailureStageV2.COLD_SOURCE_FAILURE:
            preflight_producer = artifact_refs["preflight_producer"]
            validate_diag2_producer_payload(
                _load_ref_json(
                    artifact_root,
                    _present_reference(preflight_producer, "preflight_producer"),
                    "preflight producer",
                ),
                mode="preflight",
            )
        _slot_validator(artifact_root, drift_slots, failure=failure)
        return drift_slots
    group_names = tuple(name for name, _ in _DIAG2_AUTHORITY_GROUPS)
    own_index = group_names.index(_DIAG2_STAGE_GROUP[failure.stage])
    slots: dict[str, EvidenceSlot] = {}
    for index, (_, names) in enumerate(_DIAG2_AUTHORITY_GROUPS):
        if index < own_index:
            for name in names:
                reference = artifact_refs[name]
                if reference is None:
                    raise ValueError(f"earlier authority is absent: {name}")
                slots[name] = EvidenceSlot.present(reference)
        elif index > own_index:
            for name in names:
                if artifact_refs[name] is not None:
                    raise ValueError(
                        f"downstream authority is unexpectedly present: {name}"
                    )
                slots[name] = EvidenceSlot.absent(AbsenceReason.NOT_REACHED)
    own_name, own_slots = _DIAG2_AUTHORITY_GROUPS[own_index]
    direct_absence = AbsenceReason(failure.reason.value)
    if own_name in {"ZERO_PREFLIGHT", "ZERO_COLD"}:
        reference = artifact_refs[own_slots[0]]
        if reference is None:
            raise ValueError("GPU-zero failure requires its typed authority")
        slots[own_slots[0]] = EvidenceSlot.present(reference)
    elif own_name in {"SETUP_REFERENCE", "SETUP_POLICY"}:
        name = own_slots[0]
        reference = artifact_refs[name]
        slots[name] = (
            EvidenceSlot.present(reference)
            if reference is not None
            else EvidenceSlot.absent(direct_absence)
        )
    elif own_name == "SETUP_SOURCE":
        encountered_absence = False
        for name in own_slots:
            reference = artifact_refs[name]
            if reference is not None:
                if encountered_absence:
                    raise ValueError(
                        "setup source authority is present after a failed prefix"
                    )
                slots[name] = EvidenceSlot.present(reference)
            else:
                slots[name] = EvidenceSlot.absent(
                    direct_absence
                    if not encountered_absence
                    else AbsenceReason.NOT_REACHED
                )
                encountered_absence = True
    elif own_name in {"PREFLIGHT", "COLD_SUPERVISION"}:
        prefix = "preflight" if own_name == "PREFLIGHT" else "cold"
        if failure.reason is FailureReasonCodeV2.CHILD_LAUNCH_FAILED:
            for name in own_slots:
                if artifact_refs[name] is not None:
                    raise ValueError("prelaunch child failure retains a child artifact")
                slots[name] = EvidenceSlot.absent(direct_absence)
        else:
            for name in (f"{prefix}_terminal", f"{prefix}_process"):
                reference = artifact_refs[name]
                if reference is None:
                    raise ValueError(f"launched child omits {name}")
                slots[name] = EvidenceSlot.present(reference)
            for name in (f"{prefix}_memory", f"{prefix}_memory_samples"):
                reference = artifact_refs[name]
                if reference is not None:
                    slots[name] = EvidenceSlot.present(reference)
                else:
                    monitor_reason = (
                        direct_absence
                        if failure.reason
                        in {
                            FailureReasonCodeV2.MONITOR_BINDING_FAILED,
                            FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
                        }
                        else AbsenceReason.MONITOR_FINALIZATION_FAILED
                    )
                    slots[name] = EvidenceSlot.absent(monitor_reason)
            producer_name = f"{prefix}_producer"
            producer = artifact_refs[producer_name]
            producer_permitted = failure.reason in {
                FailureReasonCodeV2.SOURCE_POST,
                FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
                FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
                FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
                FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
                FailureReasonCodeV2.POLICY_SCHEMA_INVALID,
                FailureReasonCodeV2.CHILD_COMPILE_FAILED,
                FailureReasonCodeV2.CHILD_COMPILE_OOM,
                FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED,
            }
            if failure.reason in {
                FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
                FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
            }:
                producer_permitted = False
            if producer is not None and not producer_permitted:
                raise ValueError("failure stage does not permit retained producer")
            if producer is not None:
                producer_payload = validate_diag2_producer_payload(
                    _load_ref_json(artifact_root, producer, f"{prefix} producer"),
                    mode=prefix,
                )
                expected_producer_status = {
                    FailureReasonCodeV2.CHILD_COMPILE_FAILED: "COMPILE_FAILURE",
                    FailureReasonCodeV2.CHILD_COMPILE_OOM: "COMPILE_OOM",
                }.get(
                    failure.reason,
                    "SUCCESS" if prefix == "preflight" else "COMPLETE",
                )
                if producer_payload["execution_status"] != expected_producer_status:
                    raise ValueError(
                        "retained producer status contradicts failure reason"
                    )
                if failure.reason is FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED:
                    process_ref = artifact_refs[f"{prefix}_process"]
                    if process_ref is None:
                        raise ValueError("monitor-finalization producer omits process")
                    process_payload = _mapping(
                        _load_ref_json(artifact_root, process_ref, f"{prefix} process"),
                        f"{prefix} process",
                    )
                    diagnostics = _mapping(
                        process_payload.get("process_diagnostics"),
                        f"{prefix} process diagnostics",
                    )
                    if (
                        _integer(
                            diagnostics.get("returncode"),
                            f"{prefix} process return code",
                            minimum=-2147483648,
                        )
                        != 0
                    ):
                        raise ValueError(
                            "monitor-finalization producer requires zero process exit"
                        )
            terminal_reference = artifact_refs[f"{prefix}_terminal"]
            if terminal_reference is None:
                raise AssertionError("launched child terminal narrowing failed")
            terminal_payload = _mapping(
                _load_ref_json(
                    artifact_root,
                    terminal_reference,
                    f"{prefix} terminal",
                ),
                f"{prefix} terminal",
            )
            _exact_keys(
                terminal_payload,
                frozenset(
                    {
                        "schema_version",
                        "terminal_status",
                        "failure_reasons",
                        "monitor_failure_kind",
                    }
                ),
                f"{prefix} terminal",
            )
            if (
                terminal_payload["schema_version"]
                != DIAG2_CHILD_TERMINAL_SCHEMA_VERSION
            ):
                raise ValueError(f"{prefix} terminal schema differs")
            terminal_status = _string(
                terminal_payload["terminal_status"], f"{prefix} terminal status"
            )
            expected_terminal_statuses = {
                FailureReasonCodeV2.CHILD_TIMEOUT: frozenset({"TIMEOUT"}),
                FailureReasonCodeV2.CHILD_EXIT_NONZERO: frozenset({"CRASH"}),
                FailureReasonCodeV2.CHILD_COMPILE_FAILED: frozenset(
                    {"COMPILE_FAILURE"}
                ),
                FailureReasonCodeV2.CHILD_COMPILE_OOM: frozenset({"COMPILE_FAILURE"}),
                FailureReasonCodeV2.PRODUCER_DECODE_FAILED: frozenset(
                    {"PROTOCOL_FAILURE"}
                ),
                FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: frozenset(
                    {"PROTOCOL_FAILURE"}
                ),
                FailureReasonCodeV2.MONITOR_BINDING_FAILED: frozenset(
                    {"MONITOR_FAILURE"}
                ),
                FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: frozenset(
                    {"COMPLETE", "CRASH", "PROTOCOL_FAILURE"}
                ),
            }.get(failure.reason)
            if (
                expected_terminal_statuses is not None
                and terminal_status not in expected_terminal_statuses
            ):
                raise ValueError("child terminal status contradicts failure reason")
            slots[producer_name] = (
                EvidenceSlot.present(producer)
                if producer is not None
                else EvidenceSlot.absent(direct_absence)
            )
            for suffix, schema_reason in (
                ("runtime", AbsenceReason.RUNTIME_SCHEMA_INVALID),
                ("policy", AbsenceReason.POLICY_SCHEMA_INVALID),
            ):
                name = f"{prefix}_{suffix}"
                reference = artifact_refs[name]
                if reference is not None:
                    slots[name] = EvidenceSlot.present(reference)
                else:
                    path_exists = (
                        artifact_root / DIAG2_EVIDENCE_SLOT_PATHS[name]
                    ).is_file()
                    slots[name] = EvidenceSlot.absent(
                        schema_reason if path_exists else direct_absence
                    )
    else:
        if failure.reason is FailureReasonCodeV2.SEMANTIC_VALIDATION_FAILED and all(
            artifact_refs[name] is not None for name in own_slots
        ):
            for name in own_slots:
                reference = artifact_refs[name]
                if reference is None:
                    raise AssertionError("unreachable complete semantic prefix")
                slots[name] = EvidenceSlot.present(reference)
        else:
            failed = False
            for name in own_slots:
                reference = artifact_refs[name]
                if reference is not None and not failed:
                    slots[name] = EvidenceSlot.present(reference)
                elif reference is None and not failed:
                    slots[name] = EvidenceSlot.absent(direct_absence)
                    failed = True
                elif reference is not None:
                    raise ValueError("numerical authority exists after failed prefix")
                else:
                    slots[name] = EvidenceSlot.absent(AbsenceReason.NOT_REACHED)
    terminal = artifact_refs["supervisor_terminal"]
    if terminal is None:
        raise ValueError("handled DIAG2 evidence omits supervisor terminal")
    slots["supervisor_terminal"] = EvidenceSlot.present(terminal)
    _slot_validator(artifact_root, slots, failure=failure)
    return slots


def derive_diag3_evidence_slots(
    *,
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    failure: StructuredFailureV2 | None,
) -> dict[str, EvidenceSlot]:
    """Derive the successor vector against the committed result directory."""

    return derive_diag2_evidence_slots(
        artifact_root=artifact_root,
        artifact_refs=artifact_refs,
        failure=failure,
        _slot_validator=_validate_diag3_slots,
        _subordinate_classifier=classify_diag3_subordinate_child_outcome,
    )


def derive_diag4_evidence_slots(
    *,
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    outcome: StructuredFailureV4,
) -> dict[str, EvidenceSlotV4]:
    """Derive the ordered v4 vector while preserving terminal closure slots."""

    if tuple(artifact_refs) != tuple(DIAG4_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG4 artifact refs differ from the frozen slot schema")
    numerical_names = (
        "cold_history",
        "cold_terminal_numerical",
        "cold_solve_timing",
        "cold_safeguard_telemetry",
    )
    numerical_present = tuple(
        artifact_refs[name] is not None for name in numerical_names
    )
    if len(frozenset(numerical_present)) != 1:
        raise ValueError("DIAG4 atomic scientific subgroup refs differ")
    if outcome.stage is FailureStageV4.SCIENTIFIC and any(
        reference is None for reference in artifact_refs.values()
    ):
        raise ValueError("DIAG4 scientific outcome requires every artifact ref")
    slots: dict[str, EvidenceSlotV4] = {}
    terminal_names = {"execution", "supervisor_terminal"}
    first_absence = True
    for name in DIAG4_EVIDENCE_SLOT_PATHS:
        reference = artifact_refs[name]
        if reference is not None:
            slots[name] = EvidenceSlotV4.present(reference)
        else:
            slots[name] = EvidenceSlotV4.absent(
                outcome.reason if first_absence else None
            )
            if name not in terminal_names:
                first_absence = False
    _validate_diag4_slots(artifact_root, slots, failure=outcome)
    return slots


def validate_diag2_preflight_gate(
    artifact_root: Path,
    *,
    evidence_slots: Mapping[str, EvidenceSlot],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
    expected_interpreter: str,
    expected_argv: tuple[str, ...],
) -> bool:
    """Authorize cold only from parent-zero, NumPy policy, and raw preflight evidence."""

    required = {
        "source_manifest",
        "frozen_numerical_subset",
        "native_reference",
        "policy_authority",
        "supervisor_before_preflight",
        "preflight_producer",
        "preflight_terminal",
        "preflight_process",
        "preflight_memory",
        "preflight_memory_samples",
        "preflight_runtime",
        "preflight_policy",
    }
    if not required.issubset(evidence_slots):
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.CHILD_LAUNCH_FAILED,
            "preflight_terminal",
            "DIAG2 preflight gate omits required slot keys",
        )
    missing_reasons = {
        "source_manifest": FailureReasonCodeV2.SOURCE_POST,
        "frozen_numerical_subset": FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
        "native_reference": FailureReasonCodeV2.REFERENCE_INVALID,
        "policy_authority": FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
        "supervisor_before_preflight": FailureReasonCodeV2.GPU_QUERY_FAILED,
        "preflight_producer": FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
        "preflight_terminal": FailureReasonCodeV2.CHILD_EXIT_NONZERO,
        "preflight_process": FailureReasonCodeV2.CHILD_EXIT_NONZERO,
        "preflight_memory": FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
        "preflight_memory_samples": FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
        "preflight_runtime": FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
        "preflight_policy": FailureReasonCodeV2.POLICY_SCHEMA_INVALID,
    }
    references: dict[str, ArtifactRef] = {}
    for name in required:
        slot = evidence_slots[name]
        if slot.state is not EvidenceState.PRESENT or slot.artifact is None:
            raise Diag2PreflightGateError(
                missing_reasons[name],
                name,
                f"DIAG2 preflight gate requires PRESENT {name}",
            )
        references[name] = slot.artifact
    if expected_gpu_uuid != GPU_UUID:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.GPU_QUERY_FAILED,
            "supervisor_before_preflight",
            "DIAG2 preflight GPU UUID differs from the frozen device",
        )
    try:
        validate_diag2_supervisor_zero_payload(
            _load_ref_json(
                artifact_root,
                references["supervisor_before_preflight"],
                "before-preflight GPU-zero authority",
            ),
            artifact_root=artifact_root,
            expected_stage="BEFORE_PREFLIGHT",
        )
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.GPU_QUERY_FAILED,
            "supervisor_before_preflight",
            str(error),
        ) from error
    try:
        authority_payload = validate_diag2_policy_authority_payload(
            _load_ref_json(
                artifact_root, references["policy_authority"], "policy authority"
            ),
            artifact_root=artifact_root,
        )
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
            "policy_authority",
            str(error),
        ) from error
    try:
        authority = _diag2_policy_evidence(authority_payload)
    except (TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
            "policy_authority",
            str(error),
        ) from error
    try:
        validate_diag2_producer_payload(
            _load_ref_json(
                artifact_root, references["preflight_producer"], "preflight producer"
            ),
            mode="preflight",
        )
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
            "preflight_producer",
            str(error),
        ) from error
    try:
        snapshot = validate_diag2_source_snapshot_authority(artifact_root)
        if references["source_manifest"].sha256 != snapshot.manifest_sha256:
            raise ValueError("source manifest reference differs")
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.SOURCE_POST, "source_manifest", str(error)
        ) from error
    try:
        validate_diag2_frozen_numerical_subset_payload(
            _load_ref_json(
                artifact_root,
                references["frozen_numerical_subset"],
                "frozen numerical subset",
            ),
            artifact_root=artifact_root,
        )
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
            "frozen_numerical_subset",
            str(error),
        ) from error
    try:
        validate_native_equivalent_reference(artifact_root / "native-reference")
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.REFERENCE_INVALID, "native_reference", str(error)
        ) from error
    try:
        validate_runtime_evidence(
            _resolve_artifact(artifact_root, references["preflight_runtime"]),
            snapshot_root=snapshot.root,
            campaign_root=artifact_root,
        )
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
            "preflight_runtime",
            str(error),
        ) from error
    try:
        preflight_policy = _parse_policy(
            _load_ref_json(
                artifact_root, references["preflight_policy"], "preflight policy"
            )
        )
        if preflight_policy != authority:
            raise ValueError("preflight policy differs from parent authority")
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.POLICY_SCHEMA_INVALID,
            "preflight_policy",
            str(error),
        ) from error
    try:
        memory_payload = _mapping(
            _load_ref_json(
                artifact_root, references["preflight_memory"], "preflight memory"
            ),
            "preflight memory",
        )
        if (
            _number(
                memory_payload.get("peak_memory_fraction"), "preflight memory fraction"
            )
            >= 0.8
        ):
            raise Diag2PreflightGateError(
                FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED,
                "preflight_memory",
                "preflight memory fraction reached the frozen limit",
            )
        return validate_diagnostic_preflight_gate(
            artifact_root,
            evidence_refs={
                "producer": references["preflight_producer"],
                "child_terminal": references["preflight_terminal"],
                "process": references["preflight_process"],
                "memory": references["preflight_memory"],
                "memory_samples": references["preflight_memory_samples"],
                "runtime": references["preflight_runtime"],
                "preflight_policy": references["preflight_policy"],
                "policy_authority": references["policy_authority"],
                "source_manifest": references["source_manifest"],
                "native_reference": references["native_reference"],
            },
            expected_gpu_uuid=expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
            expected_interpreter=expected_interpreter,
            expected_argv=expected_argv,
            expected_route=DIAG2_ROUTE,
            expected_plan_sha256=DIAG2_PLAN_SHA256,
            policy_authority_override=authority,
            process_schema_version=DIAG2_PROCESS_SCHEMA_VERSION,
            child_terminal_schema_version=DIAG2_CHILD_TERMINAL_SCHEMA_VERSION,
            require_parent_monotonic_interval=True,
        )
    except Diag2PreflightGateError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise Diag2PreflightGateError(
            FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
            "preflight_process",
            str(error),
        ) from error


def validate_diag4_preflight_gate(
    artifact_root: Path,
    *,
    evidence_slots: Mapping[str, EvidenceSlotV4],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
    expected_interpreter: str,
    expected_argv: tuple[str, ...],
    expected_identity: Mapping[str, str],
    expected_frozen_numerical_entries: Mapping[str, str],
) -> bool:
    """Authorize the sole cold from exact setup and trace-free compile evidence."""

    required = (
        "source_manifest",
        "frozen_numerical_subset",
        "native_reference",
        "policy_authority",
        "supervisor_before_preflight",
        "preflight_producer",
        "preflight_terminal",
        "preflight_process",
        "preflight_memory",
        "preflight_memory_samples",
        "preflight_runtime",
        "preflight_policy",
    )
    refs = {
        name: _present_artifact(EvidenceSlot(slot.state, slot.artifact, None), name)
        for name, slot in evidence_slots.items()
        if name in required
    }
    if frozenset(refs) != frozenset(required):
        raise ValueError("DIAG4 preflight gate omits a required authority")
    snapshot = load_snapshot(
        artifact_root / "source-snapshot",
        required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
    )
    observed_roles = {entry.relative_path: entry.role for entry in snapshot.entries}
    if any(
        observed_roles.get(path) != role
        for path, role in REQUIRED_SOURCE_ROLE_BINDINGS.items()
    ):
        raise ValueError("DIAG4 source snapshot path/role binding differs")
    if refs["source_manifest"].sha256 != snapshot.manifest_sha256:
        raise ValueError("DIAG4 preflight source-manifest binding differs")
    validate_diag4_frozen_numerical_subset_payload(
        _load_ref_json(
            artifact_root,
            refs["frozen_numerical_subset"],
            "DIAG4 frozen numerical subset",
        ),
        artifact_root=artifact_root,
        expected_entries=expected_frozen_numerical_entries,
    )
    validate_native_equivalent_reference(artifact_root / "native-reference")
    zero_payload = validate_diag2_supervisor_zero_payload(
        _load_ref_json(
            artifact_root,
            refs["supervisor_before_preflight"],
            "DIAG4 supervisor before preflight",
        ),
        artifact_root=artifact_root,
        expected_stage="BEFORE_PREFLIGHT",
    )
    authority = validate_diag2_policy_authority_payload(
        _load_ref_json(
            artifact_root, refs["policy_authority"], "DIAG4 policy authority"
        ),
        artifact_root=artifact_root,
    )
    producer = validate_diag4_producer_payload(
        _load_ref_json(
            artifact_root, refs["preflight_producer"], "DIAG4 preflight producer"
        ),
        mode="preflight",
    )
    if (
        _artifact_ref(producer["runtime_evidence"], "DIAG4 preflight runtime ref")
        != refs["preflight_runtime"]
        or _artifact_ref(producer["policy_evidence"], "DIAG4 preflight policy ref")
        != refs["preflight_policy"]
        or _sha256(producer["source_manifest_sha256"], "DIAG4 preflight source SHA")
        != snapshot.manifest_sha256
    ):
        raise ValueError("DIAG4 preflight producer authority binding differs")
    policy = _parse_policy(
        _load_ref_json(
            artifact_root, refs["preflight_policy"], "DIAG4 preflight policy"
        )
    )
    if policy != _diag2_policy_evidence(authority) or policy.policy_sha256 != _sha256(
        producer["base_neq_gntr1_policy_sha256"],
        "DIAG4 preflight base policy SHA",
    ):
        raise ValueError("DIAG4 preflight policy identity differs")
    identity_fields = frozenset(
        {
            "problem_sha256",
            "optimizer_options_sha256",
            "base_neq_gntr1_policy_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
        }
    )
    if frozenset(expected_identity) != identity_fields or any(
        _sha256(expected_identity[name], f"expected DIAG4 identity.{name}")
        != _sha256(producer[name], f"preflight DIAG4 identity.{name}")
        for name in identity_fields
    ):
        raise ValueError("DIAG4 preflight identity differs from authority")
    runtime = validate_runtime_evidence(
        _resolve_artifact(artifact_root, refs["preflight_runtime"]),
        snapshot_root=snapshot.root,
        campaign_root=artifact_root,
        required_roles=DIAG4_GPU_SNAPSHOT_ROLES,
    )
    runtime_identity = runtime.observation.runtime_identity
    producer_runtime = _runtime_mapping(producer["runtime"], "DIAG4 preflight runtime")
    if (
        expected_gpu_uuid != GPU_UUID
        or runtime_identity.backend != "gpu"
        or runtime_identity.device_uuid != expected_gpu_uuid
        or runtime_identity.python_executable != expected_interpreter
        or producer_runtime["backend"] != "gpu"
        or producer_runtime["device_uuid"] != expected_gpu_uuid
        or _boolean(producer_runtime["jax_enable_x64"], "DIAG4 preflight x64")
        is not True
    ):
        raise ValueError("DIAG4 preflight runtime identity differs")
    terminal, process, monitor_kind, returncode = _diag2_child_documents(
        artifact_root,
        {name: slot.artifact for name, slot in evidence_slots.items()},
        mode="preflight",
    )
    process_argv = tuple(
        _string(item, "DIAG4 preflight argv")
        for item in _array(process["argv"], "DIAG4 preflight argv")
    )
    child_pid = _integer(process["child_pid"], "DIAG4 preflight PID", minimum=1)
    child_start_ticks = _integer(
        process["child_start_time_ticks"], "DIAG4 preflight start ticks", minimum=1
    )
    if (
        terminal["terminal_status"] != "COMPLETE"
        or monitor_kind != "NONE"
        or returncode != 0
        or process_argv != expected_argv
        or expected_argv[0] != expected_interpreter
        or _integer(
            zero_payload["captured_at_monotonic_ns"],
            "DIAG4 preflight GPU-zero capture",
        )
        >= _integer(
            process["process_started_monotonic_ns"],
            "DIAG4 preflight process start",
        )
    ):
        raise ValueError("DIAG4 preflight terminal/process gate differs")
    memory_payload = _mapping(
        _load_ref_json(artifact_root, refs["preflight_memory"], "DIAG4 memory"),
        "DIAG4 memory",
    )
    peak = _integer(memory_payload["peak_memory_bytes"], "DIAG4 preflight peak")
    fraction = _number(
        memory_payload["peak_memory_fraction"], "DIAG4 preflight memory fraction"
    )
    _validate_memory(
        memory_payload,
        _load_ref_json(
            artifact_root,
            refs["preflight_memory_samples"],
            "DIAG4 memory samples",
        ),
        expected_pid=child_pid,
        expected_start_ticks=child_start_ticks,
        expected_argv=expected_argv,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
        expected_peak_bytes=peak,
        expected_peak_fraction=fraction,
        context="DIAG4 preflight memory",
    )
    if fraction >= 0.8:
        raise ValueError("DIAG4 preflight memory does not authorize cold")
    return True


def _diag2_present_refs(
    evidence_slots: Mapping[str, EvidenceSlot],
) -> dict[str, ArtifactRef]:
    result: dict[str, ArtifactRef] = {}
    for old_name, slot_name in _DIAG2_V1_REF_NAMES.items():
        slot = evidence_slots[slot_name]
        if slot.artifact is None:
            raise ValueError(f"complete DIAG2 evidence omits {slot_name}")
        result[old_name] = slot.artifact
    return result


def _build_complete_diag2_receipt(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlot],
    supervisor_terminal: Mapping[str, JsonValue] | None,
) -> DiagnosticReceiptV2:
    refs = _diag2_present_refs(evidence_slots)
    history = _parse_history(_load_ref_json(artifact_root, refs["history"], "history"))
    terminal = _parse_terminal(
        artifact_root,
        _load_ref_json(artifact_root, refs["terminal_numerical"], "terminal"),
    )
    policy = _parse_policy(
        _load_ref_json(artifact_root, refs["policy"], "quality policy"), terminal
    )
    _validate_native_equalities_authority(artifact_root, terminal)
    authority_payload = validate_diag2_policy_authority_payload(
        _load_ref_json(artifact_root, refs["policy_authority"], "policy authority"),
        artifact_root=artifact_root,
    )
    authority = _diag2_policy_evidence(authority_payload)
    if policy != authority:
        raise ValueError("cold policy differs from parent NumPy authority")
    execution = _parse_execution(
        _load_ref_json(artifact_root, refs["execution"], "execution"), refs
    )
    if policy.policy_sha256 != execution.policy_sha256:
        raise ValueError("terminal policy differs from execution")
    _validate_execution_authorities(
        artifact_root,
        refs,
        execution,
        expected_route=DIAG2_ROUTE,
        expected_plan_sha256=DIAG2_PLAN_SHA256,
        policy_authority_override=authority,
    )
    normalized = _load_ref_json(
        artifact_root, refs["trace_intervals"], "trace intervals"
    )
    if (
        normalize_chrome_trace(
            _resolve_artifact(artifact_root, refs["raw_trace"]),
            phase_schema_sha256=execution.phase_schema_sha256,
        )
        != normalized
    ):
        raise ValueError("normalized trace differs from raw profiler bytes")
    phases = _parse_phases(normalized, execution.phase_schema_sha256)
    derived = _derive(
        tuple((name, refs[name]) for name in sorted(refs)),
        policy,
        history,
        terminal,
        phases,
        execution,
    )
    verdict = (
        "DIAGNOSTIC_COMPLETE_QUALITY_HIT"
        if derived.verdict is DiagnosticVerdict.QUALITY_HIT
        else "DIAGNOSTIC_COMPLETE_NO_HIT"
    )
    next_route = derived.next_route.value
    if (
        supervisor_terminal is not None
        and supervisor_terminal["algorithm_route_selection"] != next_route
    ):
        raise ValueError("supervisor terminal route differs from recomputation")
    return DiagnosticReceiptV2(
        tuple((name, evidence_slots[name]) for name in sorted(evidence_slots)),
        verdict,
        derived.historical_relation.value,
        _quality_payload(derived.quality),
        _phase_payload(derived.phases),
        next_route,
        None,
    )


def derive_diag2_algorithm_route(
    *,
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    _receipt_builder: Callable[..., DiagnosticReceiptV2] | None = None,
) -> str:
    """Derive the complete numerical route before supervisor-terminal publication."""

    if _receipt_builder is None:
        _receipt_builder = _build_complete_diag2_receipt
    if frozenset(artifact_refs) != DIAG2_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG2 artifact refs differ from the frozen slot schema")
    slots: dict[str, EvidenceSlot] = {}
    for name, reference in artifact_refs.items():
        if name == "supervisor_terminal":
            slots[name] = EvidenceSlot.absent(AbsenceReason.NOT_REACHED)
        elif reference is None:
            raise ValueError(f"complete route derivation omits {name}")
        else:
            slots[name] = EvidenceSlot.present(reference)
    return _receipt_builder(artifact_root, slots, None).next_route


def derive_diag3_algorithm_route(
    *, artifact_root: Path, artifact_refs: Mapping[str, ArtifactRef | None]
) -> str:
    return derive_diag2_algorithm_route(
        artifact_root=artifact_root,
        artifact_refs=artifact_refs,
    )


def build_diag2_diagnostic_receipt(
    *,
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlot],
    _slot_validator: Callable[..., None] | None = None,
    _slot_deriver: Callable[..., dict[str, EvidenceSlot]] | None = None,
) -> DiagnosticReceiptV2:
    """Recompute the sole v2 verdict from physical evidence and typed absences."""

    if _slot_validator is None:
        _slot_validator = _validate_diag2_slots
    if _slot_deriver is None:
        _slot_deriver = derive_diag2_evidence_slots
    if frozenset(evidence_slots) != DIAG2_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG2 evidence slots differ from the frozen schema")
    terminal_slot = evidence_slots["supervisor_terminal"]
    terminal_ref = _present_artifact(terminal_slot, "supervisor_terminal")
    if terminal_ref.relative_path != DIAG2_EVIDENCE_SLOT_PATHS["supervisor_terminal"]:
        raise ValueError("supervisor_terminal path differs from the frozen layout")
    terminal_payload, failure = _parse_diag2_supervisor_terminal(
        _load_ref_json(artifact_root, terminal_ref, "DIAG2 supervisor terminal")
    )
    _slot_validator(artifact_root, evidence_slots, failure=failure)
    if terminal_payload["disposition"] == "COMPLETE":
        if failure is not None or any(
            slot.state is not EvidenceState.PRESENT for slot in evidence_slots.values()
        ):
            raise ValueError("complete DIAG2 terminal requires every slot present")
        return _build_complete_diag2_receipt(
            artifact_root, evidence_slots, terminal_payload
        )
    if failure is None:
        raise ValueError("incomplete DIAG2 terminal omits structured failure")
    zero_slot_name = {
        FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE: (
            "supervisor_before_preflight",
            "BEFORE_PREFLIGHT",
        ),
        FailureStageV2.GPU_ZERO_BEFORE_COLD_FAILURE: (
            "supervisor_before_cold",
            "BEFORE_COLD",
        ),
    }.get(failure.stage)
    if zero_slot_name is not None:
        slot_name, stage_name = zero_slot_name
        zero_ref = evidence_slots[slot_name].artifact
        if zero_ref is None:
            raise ValueError("GPU-zero failure omits typed query authority")
        zero_payload = validate_diag2_supervisor_zero_payload(
            _load_ref_json(artifact_root, zero_ref, "failed GPU-zero authority"),
            artifact_root=artifact_root,
            expected_stage=stage_name,
            allow_failure=True,
        )
        rows = _array(zero_payload["matching_rows"], "failed GPU matching rows")
        if failure.reason is FailureReasonCodeV2.GPU_PARENT_PID_PRESENT:
            if not rows:
                raise ValueError("GPU parent-PID failure lacks a matching raw row")
        elif rows:
            raise ValueError("GPU query failure cannot contain a matching parent row")
        else:
            try:
                validate_diag2_supervisor_zero_payload(
                    zero_payload,
                    artifact_root=artifact_root,
                    expected_stage=stage_name,
                )
            except (UnicodeDecodeError, ValueError):
                pass
            else:
                raise ValueError("GPU query failure contradicts successful zero query")
    derived_slots = _slot_deriver(
        artifact_root=artifact_root,
        artifact_refs={name: slot.artifact for name, slot in evidence_slots.items()},
        failure=failure,
    )
    if dict(evidence_slots) != derived_slots:
        raise ValueError("DIAG2 evidence slot vector differs from physical outcome")
    return DiagnosticReceiptV2(
        tuple((name, evidence_slots[name]) for name in sorted(evidence_slots)),
        "DIAGNOSTIC_INCOMPLETE",
        "NOT_COMPARABLE_INCOMPLETE",
        None,
        None,
        "NOT_PRODUCED",
        failure,
    )


def build_diag3_diagnostic_receipt(
    *, artifact_root: Path, evidence_slots: Mapping[str, EvidenceSlot]
) -> DiagnosticReceiptV2:
    """Recompute the additive successor receipt from its atomic-result layout."""

    return build_diag2_diagnostic_receipt(
        artifact_root=artifact_root,
        evidence_slots=evidence_slots,
        _slot_validator=_validate_diag3_slots,
        _slot_deriver=derive_diag3_evidence_slots,
    )


def _diag4_scientific_reconstruction(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV4],
) -> tuple[FailureReasonCodeV4, QualityEvidence, SolveTimingEvidenceV4]:
    refs = {
        name: _present_artifact(EvidenceSlot(slot.state, slot.artifact, None), name)
        for name, slot in evidence_slots.items()
        if name != "supervisor_terminal"
    }
    history_payload = _load_ref_json(
        artifact_root, refs["cold_history"], "DIAG4 history"
    )
    history = _parse_history(history_payload, defer_step_bounds=True)
    terminal_evidence = validate_diag4_terminal_numerical_payload(
        artifact_root,
        _load_ref_json(
            artifact_root, refs["cold_terminal_numerical"], "DIAG4 terminal"
        ),
    )
    terminal = terminal_evidence.terminal
    telemetry = validate_safeguard_telemetry_payload(
        _load_ref_json(
            artifact_root,
            refs["cold_safeguard_telemetry"],
            "DIAG4 safeguard telemetry",
        ),
        history=history,
        expected_history_evidence=refs["cold_history"],
    )
    telemetry_identity = NativeEquivalentNumericalIdentity(
        telemetry.numerical_route,
        telemetry.numerical_result_schema_version,
        telemetry.problem_sha256,
        telemetry.optimizer_options_sha256,
        telemetry.base_neq_gntr1_policy_sha256,
        telemetry.scaling_sha256,
        telemetry.bootstrap_state_sha256,
        telemetry.initial_physical_state_sha256,
        telemetry.identity_sha256,
    )
    if terminal_evidence.numerical_identity != telemetry_identity:
        raise ValueError("DIAG4 terminal/safeguard numerical identity differs")
    policy = _parse_policy(
        _load_ref_json(artifact_root, refs["cold_policy"], "DIAG4 policy"),
        terminal,
    )
    authority_payload = validate_diag2_policy_authority_payload(
        _load_ref_json(
            artifact_root, refs["policy_authority"], "DIAG4 policy authority"
        ),
        artifact_root=artifact_root,
    )
    if policy != _diag2_policy_evidence(authority_payload):
        raise ValueError("DIAG4 cold policy differs from parent authority")
    _validate_native_equalities_authority(artifact_root, terminal)
    _validate_terminal_raw_evidence(terminal, history, policy)
    _validate_quality_replay(history, terminal, policy)
    quality = _quality(terminal)
    numerical_complete = bool(
        not history.fatal
        and history.attempts > 0
        and (
            history.attempts == MAXIMUM_ATTEMPTS
            or history.accepted_steps == MAXIMUM_ACCEPTED_STEPS
            or history.quality_latch
        )
        and _terminal_semantics(history, terminal)
        and quality.residual_value_margin >= 0.0
        and quality.residual_gradient_margin >= 0.0
        and quality.transpose_margin >= 0.0
    )
    hit = bool(
        numerical_complete
        and history.quality_latch
        and quality.passes
        and history.first_quality_attempt > 0
        and history.first_quality_accepted_step > 0
    )
    reason = (
        FailureReasonCodeV4.QUALITY_HIT
        if hit
        else (
            FailureReasonCodeV4.NO_HIT
            if numerical_complete
            else FailureReasonCodeV4.INCOMPLETE
        )
    )
    timing = validate_solve_timing_evidence_payload(
        _load_ref_json(artifact_root, refs["cold_solve_timing"], "DIAG4 solve timing")
    )
    return reason, quality, timing


def derive_diag4_algorithm_route(
    *, artifact_root: Path, artifact_refs: Mapping[str, ArtifactRef | None]
) -> str:
    """Gate conditional timing before the supervisor terminal is published."""

    outcome = derive_diag4_scientific_outcome(
        artifact_root=artifact_root, artifact_refs=artifact_refs
    )
    return (
        DIAG4_CONDITIONAL_TIMING_ROUTE
        if outcome.reason is FailureReasonCodeV4.QUALITY_HIT
        else "NOT_PRODUCED"
    )


def derive_diag4_scientific_outcome(
    *, artifact_root: Path, artifact_refs: Mapping[str, ArtifactRef | None]
) -> StructuredFailureV4:
    """Reconstruct the committed science before supervisor-terminal publication."""

    if tuple(artifact_refs) != tuple(DIAG4_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG4 scientific refs differ from the frozen slot schema")
    if artifact_refs["supervisor_terminal"] is not None or any(
        artifact_refs[name] is None
        for name in DIAG4_EVIDENCE_SLOT_PATHS
        if name != "supervisor_terminal"
    ):
        raise ValueError(
            "DIAG4 scientific reconstruction requires the committed prefix"
        )
    slots = {
        name: (
            EvidenceSlotV4.absent()
            if name == "supervisor_terminal"
            else EvidenceSlotV4.present(_present_reference(reference, name))
        )
        for name, reference in artifact_refs.items()
    }
    reason, _, _ = _diag4_scientific_reconstruction(artifact_root, slots)
    detail_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "reason": reason.value,
                "evidence": {
                    name: _artifact_ref_payload(_present_reference(reference, name))
                    for name, reference in artifact_refs.items()
                    if name != "supervisor_terminal"
                },
            }
        )
    ).hexdigest()
    return StructuredFailureV4(
        FailureStageV4.SCIENTIFIC,
        reason,
        detail_sha256,
    )


def build_diag4_diagnostic_receipt(
    *, artifact_root: Path, evidence_slots: Mapping[str, EvidenceSlotV4]
) -> DiagnosticReceiptV4:
    """Recompute v4 science first and expose timing only after a parity hit."""

    if frozenset(evidence_slots) != DIAG4_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG4 evidence slots differ from the frozen schema")
    terminal_slot = evidence_slots["supervisor_terminal"]
    if terminal_slot.artifact is None:
        raise ValueError("DIAG4 receipt omits supervisor terminal")
    terminal_payload, outcome = parse_diag4_supervisor_terminal_payload(
        _load_ref_json(
            artifact_root, terminal_slot.artifact, "DIAG4 supervisor terminal"
        )
    )
    _validate_diag4_slots(artifact_root, evidence_slots, failure=outcome)
    if outcome.stage is not FailureStageV4.SCIENTIFIC:
        return DiagnosticReceiptV4(
            tuple((name, evidence_slots[name]) for name in DIAG4_EVIDENCE_SLOT_PATHS),
            "DIAGNOSTIC_INCOMPLETE",
            "NOT_COMPARABLE_INCOMPLETE",
            None,
            {"status": "NOT_PRODUCED"},
            "NOT_PRODUCED",
            "NOT_PRODUCED",
            outcome,
        )
    derived_reason, quality, timing = _diag4_scientific_reconstruction(
        artifact_root, evidence_slots
    )
    if outcome.reason is not derived_reason:
        raise ValueError("DIAG4 scientific terminal differs from reconstruction")
    quality_hit = derived_reason is FailureReasonCodeV4.QUALITY_HIT
    next_route = DIAG4_CONDITIONAL_TIMING_ROUTE if quality_hit else "NOT_PRODUCED"
    if terminal_payload["next_route"] != next_route:
        raise ValueError("DIAG4 supervisor next route differs")
    speed_comparison: JsonValue = (
        {
            "status": "NON_FORMAL_ENGINEERING_CONTEXT",
            "synchronized_solve_seconds": timing.synchronized_solve_seconds,
            "historical_threshold_seconds": DIAG4_ENGINEERING_THRESHOLD_SECONDS,
            "observed_ratio_to_historical_threshold": (
                timing.synchronized_solve_seconds / DIAG4_ENGINEERING_THRESHOLD_SECONDS
            ),
        }
        if quality_hit
        else "NOT_PRODUCED"
    )
    return DiagnosticReceiptV4(
        tuple((name, evidence_slots[name]) for name in DIAG4_EVIDENCE_SLOT_PATHS),
        (
            "DIAGNOSTIC_COMPLETE_QUALITY_HIT"
            if quality_hit
            else (
                "DIAGNOSTIC_COMPLETE_NO_HIT"
                if derived_reason is FailureReasonCodeV4.NO_HIT
                else "DIAGNOSTIC_INCOMPLETE"
            )
        ),
        "NOT_COMPARABLE_SUCCESSOR",
        _quality_payload(quality),
        {"status": "NOT_PRODUCED"},
        next_route,
        speed_comparison,
        outcome,
    )


def _diagnostic_receipt_payload_for_schema(
    receipt: DiagnosticReceiptV2,
    *,
    schema_version: str,
) -> dict[str, JsonValue]:
    """Serialize derived v2 claims; every authority remains a content-addressed slot."""

    return {
        "schema_version": schema_version,
        "route": DIAG2_ROUTE,
        "numerical_route": DIAG2_NUMERICAL_ROUTE,
        "plan_sha256": DIAG2_PLAN_SHA256,
        "evidence_slots": {
            name: diag2_evidence_slot_payload(slot)
            for name, slot in receipt.evidence_slots
        },
        "verdict": receipt.verdict,
        "historical_relation": receipt.historical_relation,
        "quality": receipt.quality,
        "phase_attribution": receipt.phase_attribution,
        "next_route": receipt.next_route,
        "failure": (
            None if receipt.failure is None else _diag2_failure_payload(receipt.failure)
        ),
        "engineering_campaign_receipt_produced": False,
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }


def diag2_diagnostic_receipt_payload(
    receipt: DiagnosticReceiptV2,
) -> dict[str, JsonValue]:
    """Serialize derived v2 claims without accepting a successor layout."""

    return _diagnostic_receipt_payload_for_schema(
        receipt, schema_version=DIAG2_SCHEMA_VERSION
    )


def diag2_diagnostic_receipt_bytes(receipt: DiagnosticReceiptV2) -> bytes:
    return canonical_json_bytes(diag2_diagnostic_receipt_payload(receipt))


def diag3_diagnostic_receipt_payload(
    receipt: DiagnosticReceiptV2,
) -> dict[str, JsonValue]:
    return _diagnostic_receipt_payload_for_schema(
        receipt, schema_version=DIAG3_SCHEMA_VERSION
    )


def diag3_diagnostic_receipt_bytes(receipt: DiagnosticReceiptV2) -> bytes:
    return canonical_json_bytes(diag3_diagnostic_receipt_payload(receipt))


def diag4_diagnostic_receipt_payload(
    receipt: DiagnosticReceiptV4,
) -> dict[str, JsonValue]:
    return {
        "schema_version": DIAG4_SCHEMA_VERSION,
        "route": DIAG4_ROUTE,
        "numerical_route": DIAG4_NUMERICAL_ROUTE,
        "plan_sha256": DIAG4_PLAN_SHA256,
        "evidence_slots": {
            name: diag4_evidence_slot_payload(slot)
            for name, slot in receipt.evidence_slots
        },
        "verdict": receipt.verdict,
        "historical_relation": receipt.historical_relation,
        "quality": receipt.quality,
        "phase_attribution": receipt.phase_attribution,
        "next_route": receipt.next_route,
        "speed_comparison": receipt.speed_comparison,
        "terminal_outcome": diag4_terminal_outcome_payload(receipt.failure),
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }


def diag4_diagnostic_receipt_bytes(receipt: DiagnosticReceiptV4) -> bytes:
    return canonical_json_bytes(diag4_diagnostic_receipt_payload(receipt))


def _diagnostic_receipt_from_payload_for_layout(
    value: JsonValue,
    *,
    artifact_root: Path,
    schema_version: str,
    slot_parser: Callable[..., EvidenceSlot],
    receipt_builder: Callable[..., DiagnosticReceiptV2],
    payload_builder: Callable[[DiagnosticReceiptV2], dict[str, JsonValue]],
) -> DiagnosticReceiptV2:
    """Rebuild every v2 claim and reject a producer-authored verdict or route."""

    payload = _mapping(value, "DIAG2 diagnostic receipt")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "plan_sha256",
                "evidence_slots",
                "verdict",
                "historical_relation",
                "quality",
                "phase_attribution",
                "next_route",
                "failure",
                "engineering_campaign_receipt_produced",
                "promotion_authorized",
                "formal_comparison",
            }
        ),
        "DIAG2 diagnostic receipt",
    )
    if (
        payload["schema_version"] != schema_version
        or payload["route"] != DIAG2_ROUTE
        or payload["numerical_route"] != DIAG2_NUMERICAL_ROUTE
        or payload["plan_sha256"] != DIAG2_PLAN_SHA256
        or payload["engineering_campaign_receipt_produced"] is not False
        or payload["promotion_authorized"] is not False
        or payload["formal_comparison"] != "NOT_PRODUCED"
    ):
        raise ValueError("DIAG2 diagnostic identity or nonpromotion literals differ")
    raw_slots = _mapping(payload["evidence_slots"], "DIAG2 evidence slots")
    _exact_keys(raw_slots, DIAG2_EVIDENCE_SLOT_NAMES, "DIAG2 evidence slots")
    slots = {
        name: slot_parser(raw_slots[name], name=name)
        for name in DIAG2_EVIDENCE_SLOT_PATHS
    }
    rebuilt = receipt_builder(artifact_root=artifact_root, evidence_slots=slots)
    if payload != payload_builder(rebuilt):
        raise ValueError("DIAG2 diagnostic claims differ from raw evidence")
    return rebuilt


def diag2_diagnostic_receipt_from_payload(
    value: JsonValue, *, artifact_root: Path
) -> DiagnosticReceiptV2:
    """Rebuild v2 claims using only the frozen v2 parser and layout."""

    return _diagnostic_receipt_from_payload_for_layout(
        value,
        artifact_root=artifact_root,
        schema_version=DIAG2_SCHEMA_VERSION,
        slot_parser=parse_diag2_evidence_slot,
        receipt_builder=build_diag2_diagnostic_receipt,
        payload_builder=diag2_diagnostic_receipt_payload,
    )


def load_diag2_diagnostic_receipt_bytes(
    data: bytes, *, artifact_root: Path
) -> DiagnosticReceiptV2:
    return diag2_diagnostic_receipt_from_payload(
        load_canonical_json_bytes(data), artifact_root=artifact_root
    )


def diag3_diagnostic_receipt_from_payload(
    value: JsonValue, *, artifact_root: Path
) -> DiagnosticReceiptV2:
    return _diagnostic_receipt_from_payload_for_layout(
        value,
        artifact_root=artifact_root,
        schema_version=DIAG3_SCHEMA_VERSION,
        slot_parser=parse_diag3_evidence_slot,
        receipt_builder=build_diag3_diagnostic_receipt,
        payload_builder=diag3_diagnostic_receipt_payload,
    )


def load_diag3_diagnostic_receipt_bytes(
    data: bytes, *, artifact_root: Path
) -> DiagnosticReceiptV2:
    return diag3_diagnostic_receipt_from_payload(
        load_canonical_json_bytes(data), artifact_root=artifact_root
    )


def diag4_diagnostic_receipt_from_payload(
    value: JsonValue, *, artifact_root: Path
) -> DiagnosticReceiptV4:
    payload = _mapping(value, "DIAG4 diagnostic receipt")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "plan_sha256",
                "evidence_slots",
                "verdict",
                "historical_relation",
                "quality",
                "phase_attribution",
                "next_route",
                "speed_comparison",
                "terminal_outcome",
                "promotion_authorized",
                "formal_comparison",
            }
        ),
        "DIAG4 diagnostic receipt",
    )
    if (
        payload["schema_version"] != DIAG4_SCHEMA_VERSION
        or payload["route"] != DIAG4_ROUTE
        or payload["numerical_route"] != DIAG4_NUMERICAL_ROUTE
        or payload["plan_sha256"] != DIAG4_PLAN_SHA256
        or payload["promotion_authorized"] is not False
        or payload["formal_comparison"] != "NOT_PRODUCED"
    ):
        raise ValueError("DIAG4 diagnostic identity differs")
    raw_slots = _mapping(payload["evidence_slots"], "DIAG4 evidence slots")
    if frozenset(raw_slots) != DIAG4_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG4 evidence slot keys differ")
    slots = {
        name: parse_diag4_evidence_slot(raw_slots[name], name=name)
        for name in DIAG4_EVIDENCE_SLOT_PATHS
    }
    if (
        parse_diag4_terminal_outcome(payload["terminal_outcome"])
        != (
            rebuilt := build_diag4_diagnostic_receipt(
                artifact_root=artifact_root, evidence_slots=slots
            )
        ).failure
    ):
        raise ValueError("DIAG4 receipt terminal outcome differs")
    if payload != diag4_diagnostic_receipt_payload(rebuilt):
        raise ValueError("DIAG4 diagnostic claims differ from raw evidence")
    return rebuilt


def load_diag4_diagnostic_receipt_bytes(
    data: bytes, *, artifact_root: Path
) -> DiagnosticReceiptV4:
    return diag4_diagnostic_receipt_from_payload(
        load_canonical_json_bytes(data), artifact_root=artifact_root
    )


def _diag2_receipt_slots(root: Path) -> dict[str, EvidenceSlot]:
    payload = _mapping(
        load_canonical_json_bytes((root / DIAG2_RECEIPT_FILENAME).read_bytes()),
        "DIAG2 diagnostic receipt",
    )
    if payload.get("schema_version") != DIAG2_SCHEMA_VERSION:
        raise ValueError("DIAG2 receipt schema differs")
    raw_slots = _mapping(payload.get("evidence_slots"), "DIAG2 evidence slots")
    _exact_keys(raw_slots, DIAG2_EVIDENCE_SLOT_NAMES, "DIAG2 evidence slots")
    return {
        name: parse_diag2_evidence_slot(raw_slots[name], name=name)
        for name in DIAG2_EVIDENCE_SLOT_PATHS
    }


def _diag3_receipt_slots(root: Path) -> dict[str, EvidenceSlot]:
    payload = _mapping(
        load_canonical_json_bytes((root / DIAG2_RECEIPT_FILENAME).read_bytes()),
        "DIAG3 diagnostic receipt",
    )
    if payload.get("schema_version") != DIAG3_SCHEMA_VERSION:
        raise ValueError("DIAG3 receipt schema differs")
    raw_slots = _mapping(payload.get("evidence_slots"), "DIAG3 evidence slots")
    _exact_keys(raw_slots, DIAG2_EVIDENCE_SLOT_NAMES, "DIAG3 evidence slots")
    return {
        name: parse_diag3_evidence_slot(raw_slots[name], name=name)
        for name in DIAG3_EVIDENCE_SLOT_PATHS
    }


def _diag4_receipt_slots(root: Path) -> dict[str, EvidenceSlotV4]:
    payload = _mapping(
        load_canonical_json_bytes((root / DIAG2_RECEIPT_FILENAME).read_bytes()),
        "DIAG4 diagnostic receipt",
    )
    if payload.get("schema_version") != DIAG4_SCHEMA_VERSION:
        raise ValueError("DIAG4 receipt schema differs")
    raw_slots = _mapping(payload.get("evidence_slots"), "DIAG4 evidence slots")
    if frozenset(raw_slots) != DIAG4_EVIDENCE_SLOT_NAMES:
        raise ValueError("DIAG4 evidence slot keys differ")
    return {
        name: parse_diag4_evidence_slot(raw_slots[name], name=name)
        for name in DIAG4_EVIDENCE_SLOT_PATHS
    }


def _diag2_add_process_roles(
    root: Path, roles: dict[str, str], reference: ArtifactRef, prefix: str
) -> None:
    process = _mapping(
        _load_ref_json(root, reference, f"{prefix} process"), f"{prefix} process"
    )
    for stream in ("stdout", "stderr"):
        stream_ref = _artifact_ref(process[stream], f"{prefix} process.{stream}")
        expected = f"{prefix}/{stream}.bin"
        if stream_ref.relative_path != expected:
            raise ValueError(f"{prefix} {stream} path differs")
        _resolve_artifact(root, stream_ref)
        roles[expected] = f"{prefix}_{stream}"


def _diag2_add_zero_roles(
    root: Path, roles: dict[str, str], reference: ArtifactRef, stage: str
) -> None:
    payload = validate_diag2_supervisor_zero_payload(
        _load_ref_json(root, reference, f"{stage} supervisor zero"),
        artifact_root=root,
        expected_stage="BEFORE_PREFLIGHT"
        if stage == "before_preflight"
        else "BEFORE_COLD",
        allow_failure=True,
    )
    for query_name, query_role in (
        ("gpu_inventory_query", "gpu_inventory"),
        ("compute_apps_query", "compute_apps"),
    ):
        query = _mapping(payload[query_name], query_name)
        for stream in ("stdout", "stderr"):
            ref = _artifact_ref(query[stream], f"{query_name}.{stream}")
            expected = f"supervisor/{stage.replace('_', '-')}-{query_role.replace('_', '-')}.{stream}.bin"
            if ref.relative_path != expected:
                raise ValueError("supervisor raw query path differs")
            roles[expected] = f"{stage}_{query_role}_{stream}"


_DIAG2_UNTYPED_PATHS: Final = frozenset(
    {
        "preflight/runtime-evidence.json",
        "preflight/policy.json",
        "cold/runtime-evidence.json",
        "cold/policy.json",
        "cold/history.json",
        "cold/terminal-numerical.json",
        "cold/trace-intervals.json",
        "execution.json",
    }
)
_DIAG2_UNTYPED_REASONS: Final = {
    "preflight_runtime": frozenset({AbsenceReason.RUNTIME_SCHEMA_INVALID}),
    "preflight_policy": frozenset({AbsenceReason.POLICY_SCHEMA_INVALID}),
    "cold_runtime": frozenset({AbsenceReason.RUNTIME_SCHEMA_INVALID}),
    "cold_policy": frozenset({AbsenceReason.POLICY_SCHEMA_INVALID}),
    "cold_history": frozenset({AbsenceReason.NUMERICAL_SCHEMA_INVALID}),
    "cold_terminal_numerical": frozenset({AbsenceReason.NUMERICAL_SCHEMA_INVALID}),
    "cold_trace_intervals": frozenset(
        {
            AbsenceReason.NUMERICAL_SCHEMA_INVALID,
            AbsenceReason.TRACE_NORMALIZATION_FAILED,
        }
    ),
    "execution": frozenset({AbsenceReason.NUMERICAL_SCHEMA_INVALID}),
}


def _diag2_artifact_roles(
    root: Path,
    *,
    _slots_loader: Callable[[Path], dict[str, EvidenceSlot]] | None = None,
    _slot_paths: Mapping[str, str] = DIAG2_EVIDENCE_SLOT_PATHS,
    _receipt_filename: str = DIAG2_RECEIPT_FILENAME,
    _trace_root: str = "cold/raw-trace/plugins/profile",
    _allow_uncommitted_result: bool = False,
) -> dict[str, str]:
    if _slots_loader is None:
        _slots_loader = _diag2_receipt_slots
    slots = _slots_loader(root)
    supervisor_terminal = _present_artifact(
        slots["supervisor_terminal"], "supervisor_terminal"
    )
    _, failure = _parse_diag2_supervisor_terminal(
        _load_ref_json(root, supervisor_terminal, "DIAG2 supervisor terminal")
    )
    setup_failure_reason = (
        failure.reason
        if failure is not None
        and (
            failure.stage in _DIAG2_INITIAL_SETUP_STAGE
            or failure.stage
            in {
                FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
                FailureStageV2.COLD_SOURCE_FAILURE,
            }
        )
        else None
    )
    roles = {_receipt_filename: "diagnostic_receipt"}
    for name, slot in slots.items():
        if slot.artifact is None:
            continue
        role = "raw_trace_chrome" if name == "cold_raw_trace" else name
        roles[slot.artifact.relative_path] = role
    for name, prefix in (("preflight_process", "preflight"), ("cold_process", "cold")):
        reference = slots[name].artifact
        if reference is not None:
            _diag2_add_process_roles(root, roles, reference, prefix)
    for name, stage in (
        ("supervisor_before_preflight", "before_preflight"),
        ("supervisor_before_cold", "before_cold"),
    ):
        reference = slots[name].artifact
        if reference is not None:
            _diag2_add_zero_roles(root, roles, reference, stage)
    terminal_ref = slots["cold_terminal_numerical"].artifact
    if terminal_ref is not None:
        terminal = _mapping(
            _load_ref_json(root, terminal_ref, "cold terminal numerical"),
            "cold terminal numerical",
        )
        arrays = _mapping(terminal["arrays"], "terminal arrays")
        _exact_keys(arrays, frozenset(ARRAY_SPECS), "terminal arrays")
        for name in ARRAY_SPECS:
            row = _mapping(arrays[name], f"terminal arrays.{name}")
            ref = _artifact_ref(row["artifact"], f"terminal arrays.{name}.artifact")
            _resolve_artifact(root, ref)
            roles[ref.relative_path] = "terminal_array"
    trace_ref = slots["cold_raw_trace"].artifact
    if trace_ref is not None:
        trace_path = Path(trace_ref.relative_path)
        xplane = trace_path.with_name(
            f"{trace_path.name.removesuffix('.trace.json.gz')}.xplane.pb"
        ).as_posix()
        if not (root / xplane).is_file():
            raise ValueError("complete raw trace omits XPlane sibling")
        roles[xplane] = "raw_trace_xplane"
    trace_root = root / _trace_root
    if trace_ref is None and trace_root.is_dir():
        retained = [path for path in trace_root.rglob("*") if path.is_file()]
        for path in retained:
            relative = path.relative_to(root).as_posix()
            if path.name.endswith(".trace.json.gz"):
                roles[relative] = "raw_trace_chrome"
            elif path.name.endswith(".xplane.pb"):
                roles[relative] = "raw_trace_xplane"
            else:
                raise ValueError("retained trace path differs")
    source_root = root / "source-snapshot"
    source_semantic_failure = (
        slots["source_manifest"].artifact is not None
        and setup_failure_reason is FailureReasonCodeV2.SOURCE_POST
    )
    source_role = (
        "source_snapshot_opaque_failure"
        if slots["source_manifest"].artifact is None or source_semantic_failure
        else "source_snapshot"
    )
    if source_root.is_dir():
        for path in source_root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if not (
                    slots["source_manifest"].artifact is not None
                    and relative == _slot_paths["source_manifest"]
                ):
                    roles[relative] = source_role
    native_root = root / "native-reference"
    native_semantic_failure = (
        slots["native_reference"].artifact is not None
        and setup_failure_reason is FailureReasonCodeV2.REFERENCE_INVALID
    )
    native_role = (
        "native_reference_opaque_failure"
        if slots["native_reference"].artifact is None or native_semantic_failure
        else "native_reference"
    )
    if native_root.is_dir():
        for path in native_root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if not (
                    native_semantic_failure
                    and relative == _slot_paths["native_reference"]
                ):
                    roles[relative] = native_role
    for slot_name in ("frozen_numerical_subset", "policy_authority"):
        slot = slots[slot_name]
        path = root / _slot_paths[slot_name]
        if path.is_file() and slot.artifact is None:
            expected_reason = (
                AbsenceReason.FROZEN_SUBSET_INVALID
                if slot_name == "frozen_numerical_subset"
                else AbsenceReason.POLICY_DERIVATION_INVALID
            )
            if slot.reason is not expected_reason:
                raise ValueError(f"retained {slot_name} reason differs")
            roles[path.relative_to(root).as_posix()] = "invalid_setup_authority_failure"
    for relative in _DIAG2_UNTYPED_PATHS:
        path = root / relative
        slot_name = next(
            (name for name, frozen in _slot_paths.items() if frozen == relative),
            None,
        )
        if (
            path.is_file()
            and slot_name is not None
            and slots[slot_name].artifact is None
            and slots[slot_name].reason in _DIAG2_UNTYPED_REASONS[slot_name]
        ):
            roles[relative] = "untyped_evidence_failure"
    uncommitted_root = root / DIAG3_UNCOMMITTED_NUMERICAL_DIRECTORY
    if uncommitted_root.is_dir() and _allow_uncommitted_result:
        if failure is None or not failure.stage.value.startswith("COLD_"):
            raise ValueError("uncommitted cold numerical bytes require a cold failure")
        for path in uncommitted_root.rglob("*"):
            if path.is_file():
                roles[path.relative_to(root).as_posix()] = (
                    "uncommitted_cold_numerical_result"
                )
    return roles


def _diag3_artifact_roles(root: Path) -> dict[str, str]:
    if (root / DIAG3_PENDING_NUMERICAL_DIRECTORY).exists():
        raise ValueError("DIAG3 pending numerical result reached receipt sealing")
    roles = _diag2_artifact_roles(
        root,
        _slots_loader=_diag3_receipt_slots,
        _slot_paths=DIAG3_EVIDENCE_SLOT_PATHS,
        _trace_root=(
            f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/raw-trace/plugins/profile"
        ),
        _allow_uncommitted_result=True,
    )
    committed = root / DIAG3_COMMITTED_NUMERICAL_DIRECTORY
    if committed.is_dir() and not any(
        relative.startswith(f"{DIAG3_COMMITTED_NUMERICAL_DIRECTORY}/")
        for relative in roles
    ):
        raise ValueError("DIAG3 committed numerical result is untyped")
    return roles


def _forbidden_trace_path(relative: str) -> bool:
    path = Path(relative)
    lowered_parts = tuple(part.lower() for part in path.parts)
    return bool(
        "raw-trace" in lowered_parts
        or "trace-intervals.json" in lowered_parts
        or any("trace-interval" in part for part in lowered_parts)
        or any("xplane" in part for part in lowered_parts)
        or any(
            left == "plugins" and right == "profile"
            for left, right in zip(lowered_parts[:-1], lowered_parts[1:])
        )
    )


def _diag4_forbidden_trace_path(relative: str) -> bool:
    """Compatibility wrapper for the frozen v4 private test surface."""

    return _forbidden_trace_path(relative)


def _diag4_artifact_roles(root: Path) -> dict[str, str]:
    if (root / DIAG3_PENDING_NUMERICAL_DIRECTORY).exists():
        raise ValueError("DIAG4 pending numerical result reached receipt sealing")
    slots = _diag4_receipt_slots(root)
    supervisor_terminal = slots["supervisor_terminal"].artifact
    if supervisor_terminal is None:
        raise ValueError("DIAG4 artifact omits supervisor terminal")
    _, outcome = parse_diag4_supervisor_terminal_payload(
        _load_ref_json(root, supervisor_terminal, "DIAG4 supervisor terminal")
    )
    roles = {DIAG2_RECEIPT_FILENAME: "diagnostic_receipt"}
    for name, slot in slots.items():
        if slot.artifact is not None:
            roles[slot.artifact.relative_path] = name
    for name, prefix in (("preflight_process", "preflight"), ("cold_process", "cold")):
        reference = slots[name].artifact
        if reference is not None:
            _diag2_add_process_roles(root, roles, reference, prefix)
    for name, stage in (
        ("supervisor_before_preflight", "before-preflight"),
        ("supervisor_before_cold", "before-cold"),
    ):
        reference = slots[name].artifact
        if reference is None:
            continue
        payload = _mapping(_load_ref_json(root, reference, name), name)
        for query_name, query_role in (
            ("gpu_inventory_query", "gpu-inventory"),
            ("compute_apps_query", "compute-apps"),
        ):
            query = _mapping(payload[query_name], f"{name}.{query_name}")
            for stream in ("stdout", "stderr"):
                stream_ref = _artifact_ref(
                    query[stream], f"{name}.{query_name}.{stream}"
                )
                expected = f"supervisor/{stage}-{query_role}.{stream}.bin"
                if stream_ref.relative_path != expected:
                    raise ValueError("DIAG4 supervisor raw query path differs")
                _resolve_artifact(root, stream_ref)
                roles[expected] = f"{name}_{query_role}_{stream}"
    terminal_reference = slots["cold_terminal_numerical"].artifact
    if terminal_reference is not None:
        terminal = _mapping(
            _load_ref_json(root, terminal_reference, "DIAG4 terminal"),
            "DIAG4 terminal",
        )
        arrays = _mapping(terminal["arrays"], "DIAG4 terminal arrays")
        _exact_keys(arrays, frozenset(ARRAY_SPECS), "DIAG4 terminal arrays")
        for name in ARRAY_SPECS:
            row = _mapping(arrays[name], f"DIAG4 terminal arrays.{name}")
            reference = _artifact_ref(
                row["artifact"], f"DIAG4 terminal arrays.{name}.artifact"
            )
            _resolve_artifact(root, reference)
            roles[reference.relative_path] = "terminal_array"
    for tree, role in (
        ("source-snapshot", "source_snapshot"),
        ("native-reference", "native_reference"),
    ):
        tree_root = root / tree
        if tree_root.is_dir():
            for path in tree_root.rglob("*"):
                if path.is_file():
                    roles.setdefault(path.relative_to(root).as_posix(), role)
    uncommitted = root / DIAG3_UNCOMMITTED_NUMERICAL_DIRECTORY
    if uncommitted.is_dir():
        quarantine_allowed = (
            outcome.stage is FailureStageV4.COLD
            and outcome.reason
            in {
                FailureReasonCodeV4.COLD_TIMEOUT,
                FailureReasonCodeV4.COLD_EXIT_NONZERO,
                FailureReasonCodeV4.COLD_PROTOCOL_INVALID,
                FailureReasonCodeV4.COLD_PRODUCER_INVALID,
            }
        ) or (
            outcome.stage is FailureStageV4.NUMERICAL_COMMIT
            and outcome.reason
            in {
                FailureReasonCodeV4.PENDING_RESULT_INVALID,
                FailureReasonCodeV4.QUARANTINE_FAILED,
            }
        )
        if not quarantine_allowed:
            raise ValueError("DIAG4 opaque quarantine contradicts terminal outcome")
        for path in uncommitted.rglob("*"):
            if path.is_file():
                roles[path.relative_to(root).as_posix()] = (
                    "uncommitted_cold_numerical_result"
                )
    held = _DIAG5_HELD_TREE.get()
    observed = (
        {
            relative
            for relative, entry in held.entries.items()
            if not entry.is_directory and relative != DIAG2_MANIFEST_FILENAME
        }
        if held is not None
        else {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != root / DIAG2_MANIFEST_FILENAME
        }
    )
    if any(_forbidden_trace_path(relative) for relative in observed):
        raise ValueError("DIAG4 artifact contains forbidden trace evidence")
    unknown = observed - frozenset(roles)
    if unknown:
        raise ValueError("DIAG4 artifact contains an unknown path or trace alias")
    forbidden_roles = {"raw_trace_chrome", "raw_trace_xplane", "trace_intervals"}
    if any(role in forbidden_roles for role in roles.values()):
        raise ValueError("DIAG4 artifact contains a forbidden trace role")
    return roles


def diag2_artifact_manifest_payload(root: Path) -> dict[str, JsonValue]:
    """Build the exact v2 role manifest, excluding the manifest's own bytes."""

    roles = _diag2_artifact_roles(root.resolve(strict=True))
    entries: list[dict[str, JsonValue]] = []
    for relative, role in sorted(roles.items()):
        data = (root / relative).read_bytes()
        entries.append(
            {
                "relative_path": relative,
                "role": role,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    return {"schema_version": DIAG2_MANIFEST_SCHEMA_VERSION, "entries": entries}


def diag3_artifact_manifest_payload(root: Path) -> dict[str, JsonValue]:
    """Build the exact successor role manifest for the atomic result layout."""

    roles = _diag3_artifact_roles(root.resolve(strict=True))
    entries: list[dict[str, JsonValue]] = []
    for relative, role in sorted(roles.items()):
        data = (root / relative).read_bytes()
        entries.append(
            {
                "relative_path": relative,
                "role": role,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    return {"schema_version": DIAG3_MANIFEST_SCHEMA_VERSION, "entries": entries}


def diag4_artifact_manifest_payload(root: Path) -> dict[str, JsonValue]:
    """Close every regular v4 byte and reject all profiler-path aliases."""

    roles = _diag4_artifact_roles(root.resolve(strict=True))
    entries: list[dict[str, JsonValue]] = []
    for relative, role in sorted(roles.items()):
        data = (root / relative).read_bytes()
        entries.append(
            {
                "relative_path": relative,
                "role": role,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    return {"schema_version": DIAG4_MANIFEST_SCHEMA_VERSION, "entries": entries}


def _validate_diag2_publication_root(
    root: Path,
    *,
    staging: bool,
    _slots_loader: Callable[[Path], dict[str, EvidenceSlot]] | None = None,
) -> None:
    if _slots_loader is None:
        _slots_loader = _diag2_receipt_slots
    slots = _slots_loader(root)
    terminal_ref = slots["supervisor_terminal"].artifact
    if terminal_ref is None:
        raise ValueError("DIAG2 artifact omits supervisor terminal")
    terminal, _ = _parse_diag2_supervisor_terminal(
        _load_ref_json(root, terminal_ref, "DIAG2 supervisor terminal")
    )
    publication = _mapping(terminal["publication"], "supervisor publication")
    expected_key = "staging_root" if staging else "final_root"
    expected = Path(_string(publication[expected_key], expected_key)).resolve(
        strict=False
    )
    if root.resolve(strict=True) != expected:
        raise ValueError(f"DIAG2 artifact root differs from publication.{expected_key}")


def _validate_diag2_manifest(
    root: Path,
    *,
    require_sealed: bool,
    _manifest_schema: str = DIAG2_MANIFEST_SCHEMA_VERSION,
    _roles_builder: Callable[[Path], dict[str, str]] | None = None,
) -> frozenset[str]:
    if _roles_builder is None:
        _roles_builder = _diag2_artifact_roles
    resolved = root.resolve(strict=True)
    if root.is_symlink() or not resolved.is_dir():
        raise ValueError("DIAG2 artifact root must be a nonsymlink directory")
    for path in (resolved, *resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError("DIAG2 artifact contains a symlink")
        if not path.is_file() and not path.is_dir():
            raise ValueError("DIAG2 artifact contains a special file")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError("DIAG2 artifact contains hardlink ambiguity")
        if require_sealed:
            mode = stat.S_IMODE(path.stat().st_mode)
            expected_mode = 0o444 if path.is_file() else 0o555
            if mode != expected_mode:
                raise ValueError("DIAG2 artifact modes differ from seal contract")
    manifest_path = resolved / DIAG2_MANIFEST_FILENAME
    manifest = _mapping(
        load_canonical_json_bytes(manifest_path.read_bytes()), "DIAG2 manifest"
    )
    _exact_keys(manifest, frozenset({"schema_version", "entries"}), "DIAG2 manifest")
    if manifest["schema_version"] != _manifest_schema:
        raise ValueError("DIAG2 manifest schema differs")
    expected_roles = _roles_builder(resolved)
    declared: list[str] = []
    for index, item in enumerate(_array(manifest["entries"], "manifest.entries")):
        context = f"manifest.entries[{index}]"
        row = _mapping(item, context)
        _exact_keys(
            row,
            frozenset({"relative_path", "role", "sha256", "size_bytes"}),
            context,
        )
        relative = _string(row["relative_path"], f"{context}.relative_path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or relative == DIAG2_MANIFEST_FILENAME
        ):
            raise ValueError(f"{context}.relative_path is not canonical")
        if expected_roles.get(relative) != _string(row["role"], f"{context}.role"):
            raise ValueError(f"{context}.role differs")
        data = (resolved / relative_path).read_bytes()
        if len(data) != _integer(
            row["size_bytes"], f"{context}.size_bytes"
        ) or hashlib.sha256(data).hexdigest() != _sha256(
            row["sha256"], f"{context}.sha256"
        ):
            raise ValueError(f"{context} bytes differ")
        declared.append(relative)
    if declared != sorted(declared) or len(declared) != len(set(declared)):
        raise ValueError("DIAG2 manifest paths must be sorted and unique")
    observed = frozenset(
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if observed != frozenset(declared) or observed != frozenset(expected_roles):
        raise ValueError("DIAG2 artifact has missing or extra files")
    return observed


def _validate_diag2_live_supervisor_identity(
    artifact_root: Path,
    *,
    _slots_loader: Callable[[Path], dict[str, EvidenceSlot]] | None = None,
) -> None:
    if _slots_loader is None:
        _slots_loader = _diag2_receipt_slots
    slots = _slots_loader(artifact_root)
    identities: set[tuple[int, int]] = set()
    for name in ("supervisor_before_preflight", "supervisor_before_cold"):
        reference = slots[name].artifact
        if reference is None:
            continue
        payload = _mapping(_load_ref_json(artifact_root, reference, name), name)
        identities.add(
            (
                _integer(
                    payload.get("supervisor_pid"), "live supervisor PID", minimum=1
                ),
                _integer(
                    payload.get("supervisor_start_ticks"),
                    "live supervisor start ticks",
                    minimum=1,
                ),
            )
        )
    if len(identities) > 1:
        raise ValueError("staging supervisor identities differ")
    for pid, expected_start_ticks in identities:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat_text.rsplit(")", 1)[1].split()
        if len(fields) <= 19 or int(fields[19]) != expected_start_ticks:
            raise ValueError("live supervisor PID/start identity differs")


def validate_diag2_writable_staging(artifact_root: Path) -> DiagnosticReceiptV2:
    """Prevalidate a writable staging tree before its irreversible chmod/rename."""

    _validate_diag2_publication_root(artifact_root, staging=True)
    _validate_diag2_manifest(artifact_root, require_sealed=False)
    _validate_diag2_live_supervisor_identity(artifact_root)
    return load_diag2_diagnostic_receipt_bytes(
        (artifact_root / DIAG2_RECEIPT_FILENAME).read_bytes(),
        artifact_root=artifact_root,
    )


def load_and_validate_diag2_staging(artifact_root: Path) -> DiagnosticReceiptV2:
    """Deep-load the sealed staging identity immediately before atomic rename."""

    _validate_diag2_publication_root(artifact_root, staging=True)
    _validate_diag2_manifest(artifact_root, require_sealed=True)
    _validate_diag2_live_supervisor_identity(artifact_root)
    return load_diag2_diagnostic_receipt_bytes(
        (artifact_root / DIAG2_RECEIPT_FILENAME).read_bytes(),
        artifact_root=artifact_root,
    )


def load_and_validate_diag2_artifact(artifact_root: Path) -> DiagnosticReceiptV2:
    """Deep-load a sealed final v2 tree and recompute all receipt claims."""

    _validate_diag2_publication_root(artifact_root, staging=False)
    _validate_diag2_manifest(artifact_root, require_sealed=True)
    return load_diag2_diagnostic_receipt_bytes(
        (artifact_root / DIAG2_RECEIPT_FILENAME).read_bytes(),
        artifact_root=artifact_root,
    )


def _validate_diag3_tree(
    artifact_root: Path, *, staging: bool, require_sealed: bool
) -> DiagnosticReceiptV2:
    _validate_diag2_publication_root(
        artifact_root, staging=staging, _slots_loader=_diag3_receipt_slots
    )
    _validate_diag2_manifest(
        artifact_root,
        require_sealed=require_sealed,
        _manifest_schema=DIAG3_MANIFEST_SCHEMA_VERSION,
        _roles_builder=_diag3_artifact_roles,
    )
    if staging:
        _validate_diag2_live_supervisor_identity(
            artifact_root, _slots_loader=_diag3_receipt_slots
        )
    return load_diag3_diagnostic_receipt_bytes(
        (artifact_root / DIAG2_RECEIPT_FILENAME).read_bytes(),
        artifact_root=artifact_root,
    )


def validate_diag3_writable_staging(artifact_root: Path) -> DiagnosticReceiptV2:
    return _validate_diag3_tree(artifact_root, staging=True, require_sealed=False)


def load_and_validate_diag3_staging(artifact_root: Path) -> DiagnosticReceiptV2:
    return _validate_diag3_tree(artifact_root, staging=True, require_sealed=True)


def load_and_validate_diag3_artifact(artifact_root: Path) -> DiagnosticReceiptV2:
    return _validate_diag3_tree(artifact_root, staging=False, require_sealed=True)


def _validate_diag4_publication_root(root: Path, *, staging: bool) -> None:
    slots = _diag4_receipt_slots(root)
    terminal_reference = slots["supervisor_terminal"].artifact
    if terminal_reference is None:
        raise ValueError("DIAG4 artifact omits supervisor terminal")
    terminal, _ = parse_diag4_supervisor_terminal_payload(
        _load_ref_json(root, terminal_reference, "DIAG4 supervisor terminal")
    )
    publication = _mapping(terminal["publication"], "DIAG4 publication")
    expected_key = "staging_root" if staging else "final_root"
    expected = Path(_string(publication[expected_key], expected_key)).resolve(
        strict=False
    )
    if root.resolve(strict=True) != expected:
        raise ValueError(f"DIAG4 artifact root differs from publication.{expected_key}")


def _validate_diag4_tree(
    artifact_root: Path, *, staging: bool, require_sealed: bool
) -> DiagnosticReceiptV4:
    _validate_diag4_publication_root(artifact_root, staging=staging)
    _validate_diag2_manifest(
        artifact_root,
        require_sealed=require_sealed,
        _manifest_schema=DIAG4_MANIFEST_SCHEMA_VERSION,
        _roles_builder=_diag4_artifact_roles,
    )
    if staging:
        _validate_diag2_live_supervisor_identity(
            artifact_root, _slots_loader=_diag4_receipt_slots
        )
    return load_diag4_diagnostic_receipt_bytes(
        (artifact_root / DIAG2_RECEIPT_FILENAME).read_bytes(),
        artifact_root=artifact_root,
    )


def validate_diag4_writable_staging(artifact_root: Path) -> DiagnosticReceiptV4:
    return _validate_diag4_tree(artifact_root, staging=True, require_sealed=False)


def load_and_validate_diag4_staging(artifact_root: Path) -> DiagnosticReceiptV4:
    return _validate_diag4_tree(artifact_root, staging=True, require_sealed=True)


def load_and_validate_diag4_artifact(artifact_root: Path) -> DiagnosticReceiptV4:
    return _validate_diag4_tree(artifact_root, staging=False, require_sealed=True)


def _native_binding_v5(value: JsonValue, *, role: str) -> NativeBindingV5:
    context = f"DIAG5 native_bindings.{role}"
    payload = _mapping(value, context)
    if role not in {"cpu", "gpu"}:
        raise ValueError("DIAG5 native binding role is unsupported")
    prefix = f"{role}_native_extension"
    _exact_keys(
        payload,
        frozenset(
            {
                f"{prefix}_path",
                "native_extension_sha256",
                "native_extension_size_bytes",
                f"{prefix}_link_count",
                f"{prefix}_device",
                f"{prefix}_inode",
            }
        ),
        context,
    )
    path = _string(payload[f"{prefix}_path"], f"{context}.{prefix}_path")
    if not Path(path).is_absolute() or Path(path) != Path(path).resolve(strict=False):
        raise ValueError(f"{context} path must be absolute and resolved")
    return NativeBindingV5(
        role,
        path,
        _sha256(payload["native_extension_sha256"], f"{context}.sha256"),
        _integer(
            payload["native_extension_size_bytes"],
            f"{context}.size_bytes",
            minimum=1,
        ),
        _integer(payload[f"{prefix}_link_count"], f"{context}.link_count", minimum=1),
        _integer(payload[f"{prefix}_device"], f"{context}.device"),
        _integer(payload[f"{prefix}_inode"], f"{context}.inode", minimum=1),
    )


def native_binding_v5_payload(binding: NativeBindingV5) -> dict[str, JsonValue]:
    if binding.role not in {"cpu", "gpu"}:
        raise ValueError("DIAG5 native binding role is unsupported")
    prefix = f"{binding.role}_native_extension"
    payload: dict[str, JsonValue] = {
        f"{prefix}_path": binding.path,
        "native_extension_sha256": binding.sha256,
        "native_extension_size_bytes": binding.size_bytes,
        f"{prefix}_link_count": binding.link_count,
        f"{prefix}_device": binding.device,
        f"{prefix}_inode": binding.inode,
    }
    if _native_binding_v5(payload, role=binding.role) != binding:
        raise ValueError("DIAG5 native binding differs from its canonical payload")
    return payload


def parse_diag5_native_bindings(
    value: JsonValue,
) -> tuple[tuple[str, NativeBindingV5], ...]:
    payload = _mapping(value, "DIAG5 native_bindings")
    _exact_keys(payload, frozenset({"cpu", "gpu"}), "DIAG5 native_bindings")
    bindings = tuple(
        (role, _native_binding_v5(payload[role], role=role)) for role in ("cpu", "gpu")
    )
    cpu = bindings[0][1]
    gpu = bindings[1][1]
    if (cpu.sha256, cpu.size_bytes) != (gpu.sha256, gpu.size_bytes):
        raise ValueError("DIAG5 CPU/GPU native binary identity differs")
    return bindings


def diag5_native_bindings_payload(
    bindings: tuple[tuple[str, NativeBindingV5], ...],
) -> dict[str, JsonValue]:
    if tuple(role for role, _binding in bindings) != ("cpu", "gpu"):
        raise ValueError("DIAG5 native binding order differs")
    result = {role: native_binding_v5_payload(binding) for role, binding in bindings}
    parse_diag5_native_bindings(result)
    return result


def diag5_solve_timing_evidence_payload(
    *,
    child_pid: int,
    child_start_time_ticks: int,
    backend: str,
    gpu_uuid: str,
    problem_sha256: str,
    optimizer_options_sha256: str,
    base_neq_gntr1_policy_sha256: str,
    scaling_sha256: str,
    bootstrap_state_sha256: str,
    initial_physical_state_sha256: str,
    identity_sha256: str,
    source_manifest_sha256: str,
    process_started_monotonic_ns: int,
    state_ready_monotonic_ns: int,
    solve_started_monotonic_ns: int,
    solve_stopped_monotonic_ns: int,
    finalizer_completed_monotonic_ns: int,
    endpoint_audit_completed_monotonic_ns: int,
    serialization_started_monotonic_ns: int,
    hot_h2d_transfers: int,
    hot_d2h_transfers: int,
    python_callbacks: int,
    final_d2h_transfers: int,
    profiler_call_audit: Diag5ProfilerCallAudit = DIAG5_PROFILER_CALL_AUDIT,
) -> dict[str, JsonValue]:
    """Build v2 timing without requiring the runner to author a schema map."""

    if not isinstance(profiler_call_audit, Diag5ProfilerCallAudit):
        raise TypeError("DIAG5 timing requires a DIAG5 profiler call audit")
    return _solve_timing_evidence_payload(
        child_pid=child_pid,
        child_start_time_ticks=child_start_time_ticks,
        backend=backend,
        gpu_uuid=gpu_uuid,
        problem_sha256=problem_sha256,
        optimizer_options_sha256=optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=base_neq_gntr1_policy_sha256,
        scaling_sha256=scaling_sha256,
        bootstrap_state_sha256=bootstrap_state_sha256,
        initial_physical_state_sha256=initial_physical_state_sha256,
        identity_sha256=identity_sha256,
        source_manifest_sha256=source_manifest_sha256,
        process_started_monotonic_ns=process_started_monotonic_ns,
        state_ready_monotonic_ns=state_ready_monotonic_ns,
        solve_started_monotonic_ns=solve_started_monotonic_ns,
        solve_stopped_monotonic_ns=solve_stopped_monotonic_ns,
        finalizer_completed_monotonic_ns=finalizer_completed_monotonic_ns,
        endpoint_audit_completed_monotonic_ns=endpoint_audit_completed_monotonic_ns,
        serialization_started_monotonic_ns=serialization_started_monotonic_ns,
        hot_h2d_transfers=hot_h2d_transfers,
        hot_d2h_transfers=hot_d2h_transfers,
        python_callbacks=python_callbacks,
        final_d2h_transfers=final_d2h_transfers,
        profiler_call_audit=profiler_call_audit,
        schema_version=DIAG5_SOLVE_TIMING_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
        evidence_type=SolveTimingEvidenceV5,
    )


def validate_diag5_solve_timing_evidence_payload(
    value: JsonValue,
) -> SolveTimingEvidenceV5:
    evidence = _validate_solve_timing_evidence_payload(
        value,
        schema_version=DIAG5_SOLVE_TIMING_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
        evidence_type=SolveTimingEvidenceV5,
    )
    if not isinstance(evidence, SolveTimingEvidenceV5):
        raise TypeError("DIAG5 timing parser returned the wrong generation")
    return evidence


def diag5_safeguard_telemetry_payload(
    *,
    history_evidence: ArtifactRef,
    problem_sha256: str,
    optimizer_options_sha256: str,
    base_neq_gntr1_policy_sha256: str,
    scaling_sha256: str,
    bootstrap_state_sha256: str,
    initial_physical_state_sha256: str,
    identity_sha256: str,
    loop_attempts: int,
    accepted_steps: int,
    retryable_rejections: int,
    terminal_status: str,
    quality_latch: bool,
    history_outcomes: tuple[str, ...],
    nonlinear_corrections: np.ndarray,
    maximum_individual_correction_step_ratio: np.ndarray,
    correction_path_step_ratio: np.ndarray,
    steihaug_solve_calls: np.ndarray,
    subtrial_count: np.ndarray,
    selected_subtrial_index: np.ndarray,
    subtrial_trust_radius: np.ndarray,
    subtrial_outcome: np.ndarray,
    subtrial_actual_reduction: np.ndarray,
    subtrial_predicted_reduction: np.ndarray,
    subtrial_maximum_individual_correction_step_ratio: np.ndarray,
    subtrial_correction_path_step_ratio: np.ndarray,
    subtrial_corrected_radius_ratio: np.ndarray,
    subtrial_steihaug_iterations: np.ndarray,
    subtrial_steihaug_hvp_evaluations: np.ndarray,
    subtrial_steihaug_solve_calls: np.ndarray,
    subtrial_total_hvp_evaluations: np.ndarray,
    subtrial_nonlinear_corrections: np.ndarray,
    subtrial_joint_evaluations: np.ndarray,
    subtrial_joint_linearizations: np.ndarray,
    subtrial_joint_value_evaluations: np.ndarray,
    subtrial_objective_residual_linearizations: np.ndarray,
    subtrial_gram_factorizations: np.ndarray,
    subtrial_gram_solves: np.ndarray,
) -> dict[str, JsonValue]:
    """Build DIAG5 correction telemetry from typed live loop vectors."""

    return _safeguard_telemetry_payload(
        history_evidence=history_evidence,
        problem_sha256=problem_sha256,
        optimizer_options_sha256=optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=base_neq_gntr1_policy_sha256,
        scaling_sha256=scaling_sha256,
        bootstrap_state_sha256=bootstrap_state_sha256,
        initial_physical_state_sha256=initial_physical_state_sha256,
        identity_sha256=identity_sha256,
        loop_attempts=loop_attempts,
        accepted_steps=accepted_steps,
        retryable_rejections=retryable_rejections,
        terminal_status=terminal_status,
        quality_latch=quality_latch,
        history_outcomes=history_outcomes,
        nonlinear_corrections=nonlinear_corrections,
        maximum_individual_correction_step_ratio=maximum_individual_correction_step_ratio,
        correction_path_step_ratio=correction_path_step_ratio,
        steihaug_solve_calls=steihaug_solve_calls,
        subtrial_count=subtrial_count,
        selected_subtrial_index=selected_subtrial_index,
        subtrial_trust_radius=subtrial_trust_radius,
        subtrial_outcome=subtrial_outcome,
        subtrial_actual_reduction=subtrial_actual_reduction,
        subtrial_predicted_reduction=subtrial_predicted_reduction,
        subtrial_maximum_individual_correction_step_ratio=subtrial_maximum_individual_correction_step_ratio,
        subtrial_correction_path_step_ratio=subtrial_correction_path_step_ratio,
        subtrial_corrected_radius_ratio=subtrial_corrected_radius_ratio,
        subtrial_steihaug_iterations=subtrial_steihaug_iterations,
        subtrial_steihaug_hvp_evaluations=subtrial_steihaug_hvp_evaluations,
        subtrial_steihaug_solve_calls=subtrial_steihaug_solve_calls,
        subtrial_total_hvp_evaluations=subtrial_total_hvp_evaluations,
        subtrial_nonlinear_corrections=subtrial_nonlinear_corrections,
        subtrial_joint_evaluations=subtrial_joint_evaluations,
        subtrial_joint_linearizations=subtrial_joint_linearizations,
        subtrial_joint_value_evaluations=subtrial_joint_value_evaluations,
        subtrial_objective_residual_linearizations=subtrial_objective_residual_linearizations,
        subtrial_gram_factorizations=subtrial_gram_factorizations,
        subtrial_gram_solves=subtrial_gram_solves,
        schema_version=DIAG5_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
        evidence_type=SafeguardTelemetryV5,
    )


def validate_diag5_safeguard_telemetry_payload(
    value: JsonValue,
    *,
    history: HistoryEvidence,
    expected_history_evidence: ArtifactRef,
) -> SafeguardTelemetryV5:
    evidence = _validate_safeguard_telemetry_payload(
        value,
        history=history,
        expected_history_evidence=expected_history_evidence,
        schema_version=DIAG5_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
        evidence_type=SafeguardTelemetryV5,
    )
    if not isinstance(evidence, SafeguardTelemetryV5):
        raise TypeError("DIAG5 telemetry parser returned the wrong generation")
    return evidence


def validate_diag5_producer_payload(
    value: JsonValue, *, mode: str
) -> dict[str, JsonValue]:
    payload = _mapping(value, "DIAG5 producer")
    if payload.get("document_origin") == "PARENT_SUPERVISOR":
        return validate_diag5_supervisor_failure_producer_payload(payload, mode=mode)
    return _validate_trace_free_producer_payload(
        value,
        mode=mode,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
        preflight_schema_version=DIAG5_PREFLIGHT_SCHEMA_VERSION,
        cold_schema_version=DIAG5_COLD_RESULT_SCHEMA_VERSION,
        numerical_bundle_schema_version=DIAG5_NUMERICAL_BUNDLE_SCHEMA_VERSION,
    )


def build_diag5_supervisor_failure_producer_payload(
    *,
    mode: str,
    selected_failure_reason: FailureReasonCodeV5,
    child_pid: int,
    child_start_time_ticks: int,
    process_started_monotonic_ns: int,
    process_stopped_monotonic_ns: int,
    process_evidence: ArtifactRef,
    child_terminal_evidence: ArtifactRef,
) -> dict[str, JsonValue]:
    if mode not in {"preflight", "cold"}:
        raise ValueError("DIAG5 supervisor producer mode differs")
    allowed = {
        "preflight": {
            FailureReasonCodeV5.PREFLIGHT_TIMEOUT,
            FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED,
            FailureReasonCodeV5.PREFLIGHT_EXIT_NONZERO,
            FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
            FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
        },
        "cold": {
            FailureReasonCodeV5.COLD_TIMEOUT,
            FailureReasonCodeV5.COLD_MONITOR_FAILED,
            FailureReasonCodeV5.COLD_EXIT_NONZERO,
            FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
            FailureReasonCodeV5.COLD_PRODUCER_INVALID,
        },
    }[mode]
    if selected_failure_reason not in allowed:
        raise ValueError("DIAG5 supervisor producer reason differs")
    prefix = mode
    if (
        process_evidence.relative_path != f"{prefix}/process.json"
        or process_evidence.schema_version != DIAG5_PROCESS_SCHEMA_VERSION
        or child_terminal_evidence.relative_path != f"{prefix}/terminal.json"
        or child_terminal_evidence.schema_version != DIAG5_CHILD_TERMINAL_SCHEMA_VERSION
    ):
        raise ValueError("DIAG5 supervisor producer evidence references differ")
    started = _integer(
        process_started_monotonic_ns,
        "DIAG5 supervisor producer process start",
        minimum=1,
    )
    stopped = _integer(
        process_stopped_monotonic_ns,
        "DIAG5 supervisor producer process stop",
        minimum=1,
    )
    if started > stopped:
        raise ValueError("DIAG5 supervisor producer process interval differs")
    payload: dict[str, JsonValue] = {
        "schema_version": (
            DIAG5_PREFLIGHT_SCHEMA_VERSION
            if mode == "preflight"
            else DIAG5_COLD_RESULT_SCHEMA_VERSION
        ),
        "route": DIAG5_ROUTE,
        "numerical_route": DIAG5_NUMERICAL_ROUTE,
        "numerical_result_schema_version": DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION,
        "plan_sha256": DIAG5_PLAN_SHA256,
        "child_mode": mode.upper(),
        "document_origin": "PARENT_SUPERVISOR",
        "execution_status": "SUPERVISION_FAILURE",
        "promotion_eligible": False,
        "selected_failure_reason": selected_failure_reason.value,
        "child_pid": _integer(
            child_pid, "DIAG5 supervisor producer child PID", minimum=1
        ),
        "child_start_time_ticks": _integer(
            child_start_time_ticks, "DIAG5 supervisor producer child start ticks"
        ),
        "process_started_monotonic_ns": started,
        "process_stopped_monotonic_ns": stopped,
        "process_evidence": _artifact_ref_payload(process_evidence),
        "child_terminal_evidence": _artifact_ref_payload(child_terminal_evidence),
    }
    return payload


def validate_diag5_supervisor_failure_producer_payload(
    value: JsonValue, *, mode: str
) -> dict[str, JsonValue]:
    payload = _mapping(value, "DIAG5 supervisor failure producer")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "numerical_result_schema_version",
                "plan_sha256",
                "child_mode",
                "document_origin",
                "execution_status",
                "promotion_eligible",
                "selected_failure_reason",
                "child_pid",
                "child_start_time_ticks",
                "process_started_monotonic_ns",
                "process_stopped_monotonic_ns",
                "process_evidence",
                "child_terminal_evidence",
            }
        ),
        "DIAG5 supervisor failure producer",
    )
    reason = FailureReasonCodeV5(
        _string(payload["selected_failure_reason"], "DIAG5 selected failure reason")
    )
    rebuilt = build_diag5_supervisor_failure_producer_payload(
        mode=mode,
        selected_failure_reason=reason,
        child_pid=_integer(payload["child_pid"], "DIAG5 child PID", minimum=1),
        child_start_time_ticks=_integer(
            payload["child_start_time_ticks"], "DIAG5 child start ticks"
        ),
        process_started_monotonic_ns=_integer(
            payload["process_started_monotonic_ns"], "DIAG5 process start", minimum=1
        ),
        process_stopped_monotonic_ns=_integer(
            payload["process_stopped_monotonic_ns"], "DIAG5 process stop", minimum=1
        ),
        process_evidence=_artifact_ref(
            payload["process_evidence"], "DIAG5 process ref"
        ),
        child_terminal_evidence=_artifact_ref(
            payload["child_terminal_evidence"], "DIAG5 terminal ref"
        ),
    )
    if payload != rebuilt:
        raise ValueError("DIAG5 supervisor failure producer differs")
    return payload


def build_diag5_compile_failure_producer_payload(
    *,
    execution_status: str,
    runtime: Mapping[str, JsonValue],
    runtime_evidence: ArtifactRef,
    compile_started_ns: int,
    compile_completed_ns: int,
    process_seconds_before_serialization: float,
    failure_reason: str,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "schema_version": DIAG5_PREFLIGHT_SCHEMA_VERSION,
        "route": DIAG5_ROUTE,
        "numerical_route": DIAG5_NUMERICAL_ROUTE,
        "numerical_result_schema_version": DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION,
        "plan_sha256": DIAG5_PLAN_SHA256,
        "mode": "TRACE_FREE_COMPILE_ONLY",
        "execution_status": execution_status,
        "runtime": dict(runtime),
        "runtime_evidence": _artifact_ref_payload(runtime_evidence),
        "campaign_authorized": False,
        "solver_dispatched": False,
        "finalizer_called": False,
        "endpoint_audit_called": False,
        "profiler_enabled": False,
        "profiler_start_calls": 0,
        "profiler_stop_calls": 0,
        "trace_normalization_calls": 0,
        "timing": {
            "compile_started_ns": compile_started_ns,
            "compile_completed_ns": compile_completed_ns,
            "process_seconds_before_serialization": process_seconds_before_serialization,
        },
        "failure_reasons": [failure_reason],
    }
    return validate_diag5_producer_payload(payload, mode="preflight")


def validate_diag5_preflight_gate(
    artifact_root: Path,
    *,
    evidence_slots: Mapping[str, EvidenceSlotV5],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
    expected_interpreter: str,
    expected_argv: tuple[str, ...],
    expected_identity: Mapping[str, str],
    expected_frozen_numerical_entries: Mapping[str, str],
    gpu_native_binding: Mapping[str, JsonValue],
) -> bool:
    """Authorize the DIAG5 cold child from the complete v2 preflight chain."""

    required = tuple(DIAG5_EVIDENCE_SLOT_PATHS)[:12]
    refs = {
        name: _present_artifact(EvidenceSlot(slot.state, slot.artifact, None), name)
        for name, slot in evidence_slots.items()
        if name in required
    }
    if tuple(refs) != required:
        raise ValueError("DIAG5 preflight gate omits a required authority")
    binding = _native_binding_v5(gpu_native_binding, role="gpu")
    snapshot = load_snapshot(
        artifact_root / "source-snapshot", required_roles=DIAG5_GPU_SNAPSHOT_ROLES
    )
    if refs["source_manifest"].sha256 != snapshot.manifest_sha256:
        raise ValueError("DIAG5 preflight source-manifest binding differs")
    validate_diag5_frozen_numerical_subset_payload(
        _load_ref_json(
            artifact_root,
            refs["frozen_numerical_subset"],
            "DIAG5 frozen numerical subset",
        ),
        artifact_root=artifact_root,
        expected_entries=expected_frozen_numerical_entries,
    )
    validate_native_equivalent_reference(artifact_root / "native-reference")
    zero = validate_diag5_supervisor_zero_payload(
        _load_ref_json(
            artifact_root,
            refs["supervisor_before_preflight"],
            "DIAG5 supervisor before preflight",
        ),
        artifact_root=artifact_root,
        expected_stage="BEFORE_PREFLIGHT",
    )
    authority = validate_diag5_policy_authority_payload(
        _load_ref_json(
            artifact_root, refs["policy_authority"], "DIAG5 policy authority"
        ),
        artifact_root=artifact_root,
    )
    producer = validate_diag5_producer_payload(
        _load_ref_json(artifact_root, refs["preflight_producer"], "DIAG5 producer"),
        mode="preflight",
    )
    if (
        _artifact_ref(producer["runtime_evidence"], "DIAG5 runtime ref")
        != refs["preflight_runtime"]
        or _artifact_ref(producer["policy_evidence"], "DIAG5 policy ref")
        != refs["preflight_policy"]
        or _sha256(producer["source_manifest_sha256"], "DIAG5 source SHA")
        != snapshot.manifest_sha256
    ):
        raise ValueError("DIAG5 preflight producer authority binding differs")
    policy = validate_diag5_policy_evidence_payload(
        _load_ref_json(artifact_root, refs["preflight_policy"], "DIAG5 policy")
    )
    if policy != _policy_evidence_from_authority(authority):
        raise ValueError("DIAG5 preflight policy differs from authority")
    identity_fields = frozenset(
        {
            "problem_sha256",
            "optimizer_options_sha256",
            "base_neq_gntr1_policy_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
        }
    )
    if frozenset(expected_identity) != identity_fields or any(
        _sha256(expected_identity[name], f"expected DIAG5 identity.{name}")
        != _sha256(producer[name], f"DIAG5 producer identity.{name}")
        for name in identity_fields
    ):
        raise ValueError("DIAG5 preflight numerical identity differs")
    runtime = validate_runtime_evidence_v2(
        _resolve_artifact(artifact_root, refs["preflight_runtime"]),
        snapshot_root=snapshot.root,
        campaign_root=artifact_root,
        expected_native_extension_path=Path(binding.path),
        expected_native_extension_sha256=binding.sha256,
        expected_native_extension_size_bytes=binding.size_bytes,
        expected_native_extension_link_count=binding.link_count,
        required_roles=DIAG5_GPU_SNAPSHOT_ROLES,
    )
    runtime_identity = runtime.observation.runtime_identity
    if (
        expected_gpu_uuid != GPU_UUID
        or runtime_identity.backend != "gpu"
        or runtime_identity.device_uuid != expected_gpu_uuid
        or runtime_identity.python_executable != expected_interpreter
    ):
        raise ValueError("DIAG5 preflight runtime identity differs")
    terminal, process, monitor_kind, returncode = _diag2_child_documents(
        artifact_root,
        {name: slot.artifact for name, slot in evidence_slots.items()},
        mode="preflight",
        child_terminal_schema_version=DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
        process_schema_version=DIAG5_PROCESS_SCHEMA_VERSION,
    )
    process_argv = tuple(
        _string(item, "DIAG5 preflight argv")
        for item in _array(process["argv"], "DIAG5 preflight argv")
    )
    child_pid = _integer(process["child_pid"], "DIAG5 child PID", minimum=1)
    child_start_ticks = _integer(
        process["child_start_time_ticks"], "DIAG5 child start ticks", minimum=1
    )
    if (
        terminal["terminal_status"] != "COMPLETE"
        or monitor_kind != "NONE"
        or returncode != 0
        or process_argv != expected_argv
        or expected_argv[0] != expected_interpreter
        or _integer(zero["captured_at_monotonic_ns"], "DIAG5 zero capture")
        >= _integer(process["process_started_monotonic_ns"], "DIAG5 process start")
    ):
        raise ValueError("DIAG5 preflight terminal/process gate differs")
    memory = _mapping(
        _load_ref_json(artifact_root, refs["preflight_memory"], "DIAG5 memory"),
        "DIAG5 memory",
    )
    peak = _integer(memory["peak_memory_bytes"], "DIAG5 preflight peak")
    fraction = _number(memory["peak_memory_fraction"], "DIAG5 preflight fraction")
    _validate_memory(
        memory,
        _load_ref_json(
            artifact_root,
            refs["preflight_memory_samples"],
            "DIAG5 memory samples",
        ),
        expected_pid=child_pid,
        expected_start_ticks=child_start_ticks,
        expected_argv=expected_argv,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
        expected_peak_bytes=peak,
        expected_peak_fraction=fraction,
        context="DIAG5 preflight memory",
        expected_memory_schema_version=DIAG5_MEMORY_SCHEMA_VERSION,
        expected_samples_schema_version=DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
    )
    if fraction >= 0.8:
        raise ValueError("DIAG5 preflight memory does not authorize cold")
    return True


def validate_diag5_numerical_documents(
    *,
    history: JsonValue,
    solve_timing: JsonValue,
    safeguard_telemetry: JsonValue,
    terminal_numerical: JsonValue,
    producer: JsonValue,
    artifact_root: Path | None = None,
) -> tuple[
    HistoryEvidence,
    SolveTimingEvidenceV5,
    SafeguardTelemetryV5,
    dict[str, JsonValue],
]:
    """Deep-join DIAG5 numerical documents without predecessor dispatch."""

    try:
        history_evidence = validate_diag5_history_evidence_payload(
            history, defer_step_bounds=True
        )
        producer_payload = validate_diag5_producer_payload(producer, mode="cold")
        (
            _terminal_legacy,
            terminal_identity,
            _endpoint_state_sha256,
            _terminal_observables,
            _endpoint_terms,
            _endpoint_observables,
        ) = _validate_gntr3_terminal_numerical_structure(terminal_numerical)
        if artifact_root is not None:
            terminal_identity = _validate_gntr3_terminal_numerical_payload(
                artifact_root, terminal_numerical
            ).numerical_identity
    except (TypeError, ValueError) as error:
        raise Diag5NumericalDocumentError(
            FailureReasonCodeV5.PENDING_RESULT_INVALID, str(error)
        ) from error
    try:
        timing = validate_diag5_solve_timing_evidence_payload(solve_timing)
    except (TypeError, ValueError) as error:
        raise Diag5NumericalDocumentError(
            FailureReasonCodeV5.TIMING_INVALID, str(error)
        ) from error
    history_reference = _artifact_ref(
        producer_payload["history_evidence"], "DIAG5 producer history evidence"
    )
    try:
        telemetry = validate_diag5_safeguard_telemetry_payload(
            safeguard_telemetry,
            history=history_evidence,
            expected_history_evidence=history_reference,
        )
    except (TypeError, ValueError) as error:
        raise Diag5NumericalDocumentError(
            FailureReasonCodeV5.SAFEGUARD_TELEMETRY_INVALID, str(error)
        ) from error
    producer_identity = (
        _string(producer_payload["numerical_route"], "DIAG5 numerical route"),
        _string(
            producer_payload["numerical_result_schema_version"],
            "DIAG5 numerical result schema",
        ),
        *(
            _sha256(producer_payload[name], f"DIAG5 producer.{name}")
            for name in (
                "problem_sha256",
                "optimizer_options_sha256",
                "base_neq_gntr1_policy_sha256",
                "scaling_sha256",
                "bootstrap_state_sha256",
                "initial_physical_state_sha256",
                "identity_sha256",
            )
        ),
    )
    timing_identity = (
        timing.numerical_route,
        timing.numerical_result_schema_version,
        timing.problem_sha256,
        timing.optimizer_options_sha256,
        timing.base_neq_gntr1_policy_sha256,
        timing.scaling_sha256,
        timing.bootstrap_state_sha256,
        timing.initial_physical_state_sha256,
        timing.identity_sha256,
    )
    telemetry_identity = (
        telemetry.numerical_route,
        telemetry.numerical_result_schema_version,
        telemetry.problem_sha256,
        telemetry.optimizer_options_sha256,
        telemetry.base_neq_gntr1_policy_sha256,
        telemetry.scaling_sha256,
        telemetry.bootstrap_state_sha256,
        telemetry.initial_physical_state_sha256,
        telemetry.identity_sha256,
    )
    terminal_identity_values = (
        terminal_identity.numerical_route,
        terminal_identity.numerical_result_schema_version,
        terminal_identity.problem_sha256,
        terminal_identity.optimizer_options_sha256,
        terminal_identity.base_neq_gntr1_policy_sha256,
        terminal_identity.scaling_sha256,
        terminal_identity.bootstrap_state_sha256,
        terminal_identity.initial_physical_state_sha256,
        terminal_identity.identity_sha256,
    )
    if (
        producer_identity != timing_identity
        or producer_identity != telemetry_identity
        or producer_identity != terminal_identity_values
        or _sha256(
            producer_payload["source_manifest_sha256"], "DIAG5 source manifest SHA"
        )
        != timing.source_manifest_sha256
        or (
            timing.profiler_start_calls,
            timing.profiler_stop_calls,
            timing.trace_normalization_calls,
        )
        != (0, 0, 0)
    ):
        raise Diag5NumericalDocumentError(
            FailureReasonCodeV5.NUMERICAL_IDENTITY_MISMATCH,
            "DIAG5 numerical document join differs",
        )
    return history_evidence, timing, telemetry, producer_payload


def diag5_execution_evidence_payload(
    *,
    supporting_evidence: Mapping[str, ArtifactRef],
    solve_timing: JsonValue,
    producer: JsonValue,
    process: JsonValue,
    gpu_native_binding: Mapping[str, JsonValue],
    authority_sha256: str,
) -> dict[str, JsonValue]:
    binding = _native_binding_v5(gpu_native_binding, role="gpu")
    timing = validate_diag5_solve_timing_evidence_payload(solve_timing)
    producer_payload = validate_diag5_producer_payload(producer, mode="cold")
    process_payload = _mapping(process, "DIAG5 cold process")
    process_keys = frozenset(
        {
            "schema_version",
            "monitor_failure_kind",
            "child_pid",
            "child_start_time_ticks",
            "argv",
            "stdout",
            "stderr",
            "process_seconds",
            "process_diagnostics",
            "pre_source_identity",
            "post_source_identity",
            "process_started_monotonic_ns",
            "process_stopped_monotonic_ns",
        }
    )
    _exact_keys(process_payload, process_keys, "DIAG5 cold process")
    if process_payload["schema_version"] != DIAG5_PROCESS_SCHEMA_VERSION:
        raise ValueError("DIAG5 cold process schema differs")
    supporting_names = DIAG5_EVIDENCE_SLOT_NAMES - frozenset(
        {"execution", "supervisor_terminal"}
    )
    if frozenset(supporting_evidence) != supporting_names:
        raise ValueError("DIAG5 execution supporting evidence keys differ")
    for name, payload_value in (
        ("cold_producer", producer_payload),
        ("cold_process", process_payload),
        ("cold_solve_timing", _mapping(solve_timing, "DIAG5 solve timing")),
    ):
        encoded = canonical_json_bytes(payload_value)
        reference = supporting_evidence[name]
        if reference.sha256 != hashlib.sha256(
            encoded
        ).hexdigest() or reference.size_bytes != len(encoded):
            raise ValueError(f"DIAG5 execution {name} bytes differ from reference")
    producer_reference_fields = {
        "runtime_evidence": "cold_runtime",
        "policy_evidence": "cold_policy",
        "history_evidence": "cold_history",
        "terminal_numerical_evidence": "cold_terminal_numerical",
        "solve_timing_evidence": "cold_solve_timing",
        "safeguard_telemetry_evidence": "cold_safeguard_telemetry",
    }
    if any(
        _artifact_ref(producer_payload[field], f"DIAG5 producer.{field}")
        != supporting_evidence[name]
        for field, name in producer_reference_fields.items()
    ):
        raise ValueError("DIAG5 execution producer references differ")
    process_started = _integer(
        process_payload["process_started_monotonic_ns"], "DIAG5 process start"
    )
    process_stopped = _integer(
        process_payload["process_stopped_monotonic_ns"], "DIAG5 process stop"
    )
    if (
        _integer(process_payload["child_pid"], "DIAG5 process child PID")
        != timing.child_pid
        or _integer(
            process_payload["child_start_time_ticks"], "DIAG5 child start ticks"
        )
        != timing.child_start_time_ticks
        or process_started != timing.process_started_monotonic_ns
        or not (
            process_started
            < timing.state_ready_monotonic_ns
            < timing.solve_started_monotonic_ns
            < timing.solve_stopped_monotonic_ns
            < timing.finalizer_completed_monotonic_ns
            < timing.endpoint_audit_completed_monotonic_ns
            < timing.serialization_started_monotonic_ns
            < process_stopped
        )
    ):
        raise ValueError("DIAG5 timing/process containment differs")
    producer_identity = tuple(
        _sha256(producer_payload[name], f"DIAG5 producer.{name}")
        for name in (
            "problem_sha256",
            "optimizer_options_sha256",
            "base_neq_gntr1_policy_sha256",
            "scaling_sha256",
            "bootstrap_state_sha256",
            "initial_physical_state_sha256",
            "identity_sha256",
        )
    )
    timing_identity = (
        timing.problem_sha256,
        timing.optimizer_options_sha256,
        timing.base_neq_gntr1_policy_sha256,
        timing.scaling_sha256,
        timing.bootstrap_state_sha256,
        timing.initial_physical_state_sha256,
        timing.identity_sha256,
    )
    if (
        producer_identity != timing_identity
        or _sha256(
            producer_payload["source_manifest_sha256"],
            "DIAG5 producer source-manifest SHA",
        )
        != timing.source_manifest_sha256
        or timing.source_manifest_sha256
        != supporting_evidence["source_manifest"].sha256
    ):
        raise ValueError("DIAG5 timing/producer numerical identity differs")
    return {
        "schema_version": DIAG5_EXECUTION_SCHEMA_VERSION,
        "route": DIAG5_ROUTE,
        "numerical_route": DIAG5_NUMERICAL_ROUTE,
        "numerical_result_schema_version": DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION,
        "plan_sha256": DIAG5_PLAN_SHA256,
        "supporting_evidence": {
            name: _artifact_ref_payload(supporting_evidence[name])
            for name in sorted(supporting_names)
        },
        "phase_attribution": "NOT_PRODUCED",
        **_profiler_call_audit_payload(),
        "process_started_monotonic_ns": process_started,
        "state_ready_monotonic_ns": timing.state_ready_monotonic_ns,
        "solve_started_monotonic_ns": timing.solve_started_monotonic_ns,
        "solve_stopped_monotonic_ns": timing.solve_stopped_monotonic_ns,
        "finalizer_completed_monotonic_ns": timing.finalizer_completed_monotonic_ns,
        "endpoint_audit_completed_monotonic_ns": timing.endpoint_audit_completed_monotonic_ns,
        "serialization_started_monotonic_ns": timing.serialization_started_monotonic_ns,
        "process_stopped_monotonic_ns": process_stopped,
        "synchronized_solve_seconds": timing.synchronized_solve_seconds,
        "gpu_native_binding": native_binding_v5_payload(binding),
        "authority_sha256": _sha256(authority_sha256, "DIAG5 authority SHA"),
    }


def validate_diag5_execution_evidence_payload(
    value: JsonValue,
    *,
    supporting_evidence: Mapping[str, ArtifactRef],
    solve_timing: JsonValue,
    producer: JsonValue,
    process: JsonValue,
    gpu_native_binding: Mapping[str, JsonValue],
    authority_sha256: str,
) -> dict[str, JsonValue]:
    payload = _mapping(value, "DIAG5 execution evidence")
    rebuilt = diag5_execution_evidence_payload(
        supporting_evidence=supporting_evidence,
        solve_timing=solve_timing,
        producer=producer,
        process=process,
        gpu_native_binding=gpu_native_binding,
        authority_sha256=authority_sha256,
    )
    if payload != rebuilt:
        raise ValueError("DIAG5 execution evidence differs from raw authorities")
    return payload


def diag5_terminal_outcome_payload(
    outcome: StructuredFailureV5,
) -> dict[str, JsonValue]:
    if outcome.reason not in DIAG5_STAGE_REASON_ORDER[outcome.stage]:
        raise ValueError("DIAG5 terminal stage/reason pairing differs")
    return {
        "stage": outcome.stage.value,
        "reason": {
            "code": outcome.reason.value,
            "detail_sha256": _sha256(
                outcome.detail_sha256, "DIAG5 terminal detail SHA"
            ),
        },
    }


def parse_diag5_terminal_outcome(value: JsonValue) -> StructuredFailureV5:
    payload = _mapping(value, "DIAG5 terminal outcome")
    _exact_keys(payload, frozenset({"stage", "reason"}), "DIAG5 terminal outcome")
    reason_payload = _mapping(payload["reason"], "DIAG5 terminal outcome.reason")
    _exact_keys(
        reason_payload,
        frozenset({"code", "detail_sha256"}),
        "DIAG5 terminal outcome.reason",
    )
    outcome = StructuredFailureV5(
        FailureStageV5(_string(payload["stage"], "DIAG5 terminal stage")),
        FailureReasonCodeV5(_string(reason_payload["code"], "DIAG5 terminal reason")),
        _sha256(reason_payload["detail_sha256"], "DIAG5 terminal detail SHA"),
    )
    if diag5_terminal_outcome_payload(outcome) != payload:
        raise ValueError("DIAG5 terminal outcome differs")
    return outcome


def select_diag5_terminal_outcome(
    candidates: Iterable[StructuredFailureV5],
) -> StructuredFailureV5:
    unique = frozenset(candidates)
    if not unique:
        raise ValueError("DIAG5 terminal outcome candidates are empty")
    for stage in DIAG5_FAILURE_STAGE_ORDER:
        for reason in DIAG5_STAGE_REASON_ORDER[stage]:
            matches = tuple(
                candidate
                for candidate in unique
                if candidate.stage is stage and candidate.reason is reason
            )
            if len(matches) > 1:
                raise ValueError("DIAG5 duplicate stage/reason candidates differ")
            if matches:
                return matches[0]
    raise AssertionError("DIAG5 terminal outcome selection exhausted its schema")


def _diag5_expected_launched_children(
    outcome: StructuredFailureV5,
) -> tuple[str, ...]:
    if (
        outcome.stage is FailureStageV5.PREFLIGHT
        and outcome.reason is FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
        and outcome.detail_sha256
        == DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256
    ):
        return ("preflight",)
    if (
        outcome.stage is FailureStageV5.COLD
        and outcome.reason is FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        and outcome.detail_sha256 == DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256
    ):
        return ("preflight", "cold")
    if outcome.stage in {
        FailureStageV5.AUTHORITY,
        FailureStageV5.SETUP,
        FailureStageV5.BEFORE_PREFLIGHT,
    } or (
        outcome.stage is FailureStageV5.PREFLIGHT
        and outcome.reason is FailureReasonCodeV5.PREFLIGHT_LAUNCH_FAILED
    ):
        return ()
    if outcome.stage in {FailureStageV5.PREFLIGHT, FailureStageV5.BEFORE_COLD} or (
        outcome.stage is FailureStageV5.COLD
        and outcome.reason is FailureReasonCodeV5.COLD_LAUNCH_FAILED
    ):
        return ("preflight",)
    return ("preflight", "cold")


def build_diag5_supervisor_terminal_payload(
    *,
    outcome: StructuredFailureV5,
    launched_children: tuple[str, ...],
    staging_root: Path,
    final_root: Path,
) -> dict[str, JsonValue]:
    """Build the v2 terminal over DIAG5's fixed `.partial-claim` lifecycle."""

    if launched_children != _diag5_expected_launched_children(outcome):
        raise ValueError("DIAG5 child sequence differs from terminal stage/reason")
    staging = staging_root.resolve(strict=False)
    final = final_root.resolve(strict=False)
    if staging.parent != final.parent or staging.name != f"{final.name}.partial-claim":
        raise ValueError("DIAG5 publication roots differ")
    scientific = outcome.stage is FailureStageV5.SCIENTIFIC
    quality_hit = outcome.reason is FailureReasonCodeV5.QUALITY_HIT
    return {
        "schema_version": DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
        "route": DIAG5_ROUTE,
        "numerical_route": DIAG5_NUMERICAL_ROUTE,
        "plan_sha256": DIAG5_PLAN_SHA256,
        "disposition": "COMPLETE" if scientific else "INCOMPLETE",
        "terminal_outcome": diag5_terminal_outcome_payload(outcome),
        "launched_children": list(launched_children),
        "publication": {
            "staging_root": str(staging),
            "final_root": str(final),
            "nonce": "claim",
        },
        "phase_attribution": "NOT_PRODUCED",
        "next_route": (
            DIAG4_CONDITIONAL_TIMING_ROUTE if quality_hit else "NOT_PRODUCED"
        ),
        "speed_comparison": (
            "CONDITIONAL_ENGINEERING_CONTEXT" if quality_hit else "NOT_PRODUCED"
        ),
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }


def parse_diag5_supervisor_terminal_payload(
    value: JsonValue,
) -> tuple[dict[str, JsonValue], StructuredFailureV5]:
    payload = _mapping(value, "DIAG5 supervisor terminal")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "plan_sha256",
                "disposition",
                "terminal_outcome",
                "launched_children",
                "publication",
                "phase_attribution",
                "next_route",
                "speed_comparison",
                "promotion_authorized",
                "formal_comparison",
            }
        ),
        "DIAG5 supervisor terminal",
    )
    publication = _mapping(payload["publication"], "DIAG5 publication")
    _exact_keys(
        publication,
        frozenset({"staging_root", "final_root", "nonce"}),
        "DIAG5 publication",
    )
    if publication["nonce"] != "claim":
        raise ValueError("DIAG5 publication nonce differs")
    outcome = parse_diag5_terminal_outcome(payload["terminal_outcome"])
    rebuilt = build_diag5_supervisor_terminal_payload(
        outcome=outcome,
        launched_children=tuple(
            _string(item, "DIAG5 launched child")
            for item in _array(payload["launched_children"], "DIAG5 children")
        ),
        staging_root=Path(_string(publication["staging_root"], "DIAG5 staging root")),
        final_root=Path(_string(publication["final_root"], "DIAG5 final root")),
    )
    if payload != rebuilt:
        raise ValueError("DIAG5 supervisor terminal differs")
    return payload, outcome


def _validate_diag5_document_schemas(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV5],
) -> dict[str, JsonValue]:
    loaded: dict[str, JsonValue] = {}
    for name, slot in evidence_slots.items():
        if slot.artifact is None:
            continue
        if slot.artifact.schema_version != DIAG5_EVIDENCE_SLOT_SCHEMAS[name]:
            raise ValueError(f"DIAG5 {name} reference schema differs")
        document = _load_ref_json(artifact_root, slot.artifact, f"DIAG5 {name}")
        payload = _mapping(document, f"DIAG5 {name}")
        if payload.get("schema_version") != DIAG5_EVIDENCE_SLOT_SCHEMAS[name]:
            raise ValueError(f"DIAG5 {name} document schema differs")
        loaded[name] = document
    return loaded


def _validate_diag5_held_source_snapshot(
    observed: SnapshotPublication,
    expected: SnapshotIdentity,
) -> None:
    if observed.identity() != expected:
        raise ValueError("DIAG5 source snapshot differs from held authority")


def _validate_diag5_process_source_identity(
    process: Mapping[str, JsonValue],
    *,
    snapshot: SnapshotPublication,
    context: str,
) -> None:
    expected: dict[str, JsonValue] = {
        "git_head": snapshot.worktree.git_head,
        "tracked_diff_sha256": snapshot.worktree.tracked_diff_sha256,
        "untracked_bytes_manifest_sha256": (
            snapshot.worktree.untracked_bytes_manifest_sha256
        ),
        "source_manifest_sha256": snapshot.manifest_sha256,
        "source_manifest_size_bytes": snapshot.manifest_path.stat().st_size,
    }
    if (
        process["pre_source_identity"] != expected
        or process["post_source_identity"] != expected
    ):
        raise ValueError(f"{context} source identity differs from authority")


def _validate_diag5_child_outcome(
    terminal: Mapping[str, JsonValue],
    process: Mapping[str, JsonValue],
    *,
    mode: str,
    failure: StructuredFailureV5,
) -> None:
    terminal_status = _string(
        terminal["terminal_status"], f"DIAG5 {mode} terminal status"
    )
    monitor_kind = _string(
        terminal["monitor_failure_kind"], f"DIAG5 {mode} terminal monitor kind"
    )
    terminal_failures = _array(
        terminal["failure_reasons"], f"DIAG5 {mode} terminal failures"
    )
    for index, item in enumerate(terminal_failures):
        if not _string(item, f"DIAG5 {mode} terminal failures[{index}]"):
            raise ValueError(f"DIAG5 {mode} terminal failure reason is empty")
    returncode = _integer(
        _mapping(process["process_diagnostics"], f"DIAG5 {mode} process diagnostics")[
            "returncode"
        ],
        f"DIAG5 {mode} process return code",
        minimum=-2147483648,
    )
    selected_reason = (
        failure.reason
        if failure.stage
        is (FailureStageV5.PREFLIGHT if mode == "preflight" else FailureStageV5.COLD)
        else None
    )
    reserved_source_revalidation = (
        failure.stage is FailureStageV5.PREFLIGHT
        and failure.reason is FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
        and failure.detail_sha256
        == DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256
    ) or (
        failure.stage is FailureStageV5.COLD
        and failure.reason is FailureReasonCodeV5.COLD_PROTOCOL_INVALID
        and failure.detail_sha256 == DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256
    )
    if reserved_source_revalidation:
        if (
            terminal_status != "COMPLETE"
            or monitor_kind != "NONE"
            or terminal_failures != []
            or returncode != 0
        ):
            raise ValueError(f"DIAG5 {mode} reserved source revalidation child differs")
        return
    expected: dict[FailureReasonCodeV5, tuple[frozenset[str], frozenset[str], bool]] = {
        FailureReasonCodeV5.PREFLIGHT_TIMEOUT: (
            frozenset({"TIMEOUT"}),
            frozenset({"NONE"}),
            True,
        ),
        FailureReasonCodeV5.COLD_TIMEOUT: (
            frozenset({"TIMEOUT"}),
            frozenset({"NONE"}),
            True,
        ),
        FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED: (
            frozenset({"COMPLETE", "CRASH", "MONITOR_FAILURE", "PROTOCOL_FAILURE"}),
            frozenset({"BINDING", "FINALIZATION"}),
            False,
        ),
        FailureReasonCodeV5.COLD_MONITOR_FAILED: (
            frozenset({"COMPLETE", "CRASH", "MONITOR_FAILURE", "PROTOCOL_FAILURE"}),
            frozenset({"BINDING", "FINALIZATION"}),
            False,
        ),
        FailureReasonCodeV5.PREFLIGHT_EXIT_NONZERO: (
            frozenset({"CRASH"}),
            frozenset({"NONE"}),
            True,
        ),
        FailureReasonCodeV5.COLD_EXIT_NONZERO: (
            frozenset({"CRASH"}),
            frozenset({"NONE"}),
            True,
        ),
        FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID: (
            frozenset({"COMPILE_FAILURE", "COMPLETE", "PROTOCOL_FAILURE"}),
            frozenset({"NONE"}),
            False,
        ),
        FailureReasonCodeV5.COLD_PROTOCOL_INVALID: (
            frozenset({"COMPILE_FAILURE", "COMPLETE", "PROTOCOL_FAILURE"}),
            frozenset({"NONE"}),
            False,
        ),
        FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID: (
            frozenset({"PROTOCOL_FAILURE"}),
            frozenset({"NONE"}),
            False,
        ),
        FailureReasonCodeV5.COLD_PRODUCER_INVALID: (
            frozenset({"PROTOCOL_FAILURE"}),
            frozenset({"NONE"}),
            False,
        ),
    }
    contract = expected.get(selected_reason)
    if contract is None:
        if (
            terminal_status != "COMPLETE"
            or terminal_failures != []
            or monitor_kind != "NONE"
            or returncode != 0
        ):
            raise ValueError(
                f"DIAG5 {mode} child does not prove successful termination"
            )
        return
    statuses, monitor_kinds, require_nonzero_return = contract
    if (
        terminal_status not in statuses
        or monitor_kind not in monitor_kinds
        or (terminal_status != "COMPLETE" and not terminal_failures)
        or (require_nonzero_return and returncode == 0)
        or (
            selected_reason
            in {
                FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED,
                FailureReasonCodeV5.COLD_MONITOR_FAILED,
            }
            and returncode != 0
        )
        or (
            selected_reason
            in {
                FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
                FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
                FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
                FailureReasonCodeV5.COLD_PRODUCER_INVALID,
            }
            and returncode != 0
        )
    ):
        raise ValueError(f"DIAG5 {mode} child contradicts selected failure")


def _validate_diag5_memory_sample_rows(
    samples: Mapping[str, JsonValue], *, context: str
) -> None:
    for index, item in enumerate(_array(samples["samples"], f"{context} samples")):
        row = _mapping(item, f"{context} samples[{index}]")
        _integer(
            row["sampled_at_unix_ns"],
            f"{context} samples[{index}].time",
            minimum=1,
        )
        _integer(
            row["used_memory_mib"],
            f"{context} samples[{index}].memory",
            minimum=0,
        )


def _validate_diag5_child_producer_origin(
    producer: Mapping[str, JsonValue],
    *,
    failure: StructuredFailureV5,
    mode: str,
) -> None:
    if producer.get(
        "document_origin"
    ) == "PARENT_SUPERVISOR" and failure.detail_sha256 in {
        DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256,
        DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256,
    }:
        raise ValueError(
            f"DIAG5 {mode} reserved source revalidation uses supervisor producer"
        )


def _validate_diag5_manifest_only_auxiliaries(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV5],
    loaded: Mapping[str, JsonValue],
    *,
    mode: str,
    authority: Mapping[str, JsonValue] | None,
    snapshot: SnapshotPublication | None,
    expected_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    gpu_native_binding: NativeBindingV5,
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> None:
    auxiliary_paths = {
        suffix: artifact_root / f"{mode}/{filename}"
        for suffix, filename in (
            ("memory", "gpu-memory.json"),
            ("memory_samples", "gpu-memory-samples.json"),
            ("runtime", "runtime-evidence.json"),
            ("policy", "policy.json"),
        )
    }
    held = _DIAG5_HELD_TREE.get()
    manifest_only = {
        suffix: path
        for suffix, path in auxiliary_paths.items()
        if evidence_slots[f"{mode}_{suffix}"].artifact is None
        and (
            f"{mode}/{path.name}" in held.entries
            if held is not None
            else os.path.lexists(path)
        )
    }
    if not manifest_only:
        return
    if held is None and any(
        path.is_symlink() or not path.is_file() for path in manifest_only.values()
    ):
        raise ValueError(f"DIAG5 {mode} auxiliary is not a regular file")
    process = loaded.get(f"{mode}_process")
    if process is None:
        raise ValueError(f"DIAG5 {mode} auxiliary omits child process")
    process_payload = _mapping(process, f"DIAG5 {mode} auxiliary process")
    if ("memory" in manifest_only) != ("memory_samples" in manifest_only):
        raise ValueError(f"DIAG5 {mode} auxiliary memory pairing differs")
    if "memory" in manifest_only:
        memory = _mapping(
            load_canonical_json_bytes(
                _diag5_held_file_bytes(
                    artifact_root, f"{mode}/{manifest_only['memory'].name}"
                )
            ),
            f"DIAG5 {mode} auxiliary memory",
        )
        samples = _mapping(
            load_canonical_json_bytes(
                _diag5_held_file_bytes(
                    artifact_root,
                    f"{mode}/{manifest_only['memory_samples'].name}",
                )
            ),
            f"DIAG5 {mode} auxiliary memory samples",
        )
        _validate_diag5_memory_sample_rows(samples, context=f"DIAG5 {mode} auxiliary")
        argv = tuple(
            _string(item, f"DIAG5 {mode} auxiliary argv")
            for item in _array(process_payload["argv"], f"DIAG5 {mode} auxiliary argv")
        )
        _validate_memory(
            memory,
            samples,
            expected_pid=_integer(
                process_payload["child_pid"], f"DIAG5 {mode} auxiliary child PID"
            ),
            expected_start_ticks=_integer(
                process_payload["child_start_time_ticks"],
                f"DIAG5 {mode} auxiliary child start ticks",
            ),
            expected_argv=argv,
            expected_gpu_uuid=expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
            expected_peak_bytes=_integer(
                memory["peak_memory_bytes"], f"DIAG5 {mode} auxiliary peak bytes"
            ),
            expected_peak_fraction=_number(
                memory["peak_memory_fraction"],
                f"DIAG5 {mode} auxiliary peak fraction",
            ),
            context=f"DIAG5 {mode} auxiliary memory",
            expected_memory_schema_version=DIAG5_MEMORY_SCHEMA_VERSION,
            expected_samples_schema_version=DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
        )
    if "runtime" in manifest_only:
        if snapshot is None:
            raise ValueError(f"DIAG5 {mode} auxiliary runtime omits source snapshot")
        validate_diag5_runtime_evidence_v2_bytes(
            _diag5_held_file_bytes(artifact_root, f"{mode}/runtime-evidence.json"),
            expected_snapshot_identity=expected_snapshot_identity,
            expected_logical_campaign_root=expected_logical_snapshot_root.parent,
            expected_logical_snapshot_root=expected_logical_snapshot_root,
            expected_native_extension_path=Path(gpu_native_binding.path),
            expected_native_extension_sha256=gpu_native_binding.sha256,
            expected_native_extension_size_bytes=gpu_native_binding.size_bytes,
            expected_native_extension_link_count=gpu_native_binding.link_count,
        )
    if "policy" in manifest_only:
        if authority is None:
            raise ValueError(f"DIAG5 {mode} auxiliary policy omits authority")
        policy = validate_diag5_policy_evidence_payload(
            load_canonical_json_bytes(
                _diag5_held_file_bytes(
                    artifact_root, f"{mode}/{manifest_only['policy'].name}"
                )
            )
        )
        if policy != _policy_evidence_from_authority(authority):
            raise ValueError(f"DIAG5 {mode} auxiliary policy differs from authority")


def _validate_diag5_stage_vector(
    evidence_slots: Mapping[str, EvidenceSlotV5],
    *,
    failure: StructuredFailureV5,
) -> None:
    ordered_names = tuple(DIAG5_EVIDENCE_SLOT_PATHS)
    if tuple(evidence_slots) != ordered_names:
        raise Diag5ReceiptConstructionError(
            FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID,
            "DIAG5 evidence slots differ from the frozen schema",
        )
    if evidence_slots["supervisor_terminal"].state is not EvidenceState.PRESENT:
        raise Diag5ReceiptConstructionError(
            FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID,
            "DIAG5 evidence vector omits supervisor terminal closure",
        )
    nonterminal_names = ordered_names[:-1]
    nonterminal_states = tuple(evidence_slots[name].state for name in nonterminal_names)
    present_count = sum(state is EvidenceState.PRESENT for state in nonterminal_states)
    if nonterminal_states != (
        (EvidenceState.PRESENT,) * present_count
        + (EvidenceState.ABSENT,) * (len(nonterminal_names) - present_count)
    ):
        raise Diag5ReceiptConstructionError(
            FailureReasonCodeV5.GROUP_PREFIX_INVALID,
            "DIAG5 evidence vector contains a hole",
        )
    allowed_prefixes = _diag5_allowed_present_prefixes(failure)
    if allowed_prefixes is None or present_count not in allowed_prefixes:
        raise Diag5ReceiptConstructionError(
            FailureReasonCodeV5.GROUP_PREFIX_INVALID,
            "DIAG5 stage-specific evidence prefix differs",
        )
    absent_names = tuple(
        name
        for name in ordered_names
        if evidence_slots[name].state is EvidenceState.ABSENT
    )
    if absent_names and (
        evidence_slots[absent_names[0]].reason is not failure.reason
        or any(evidence_slots[name].reason is not None for name in absent_names[1:])
    ):
        raise Diag5ReceiptConstructionError(
            FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID,
            "DIAG5 absence reasons differ from terminal outcome",
        )
    numerical_states = tuple(
        evidence_slots[name].state
        for name in (
            "cold_history",
            "cold_terminal_numerical",
            "cold_solve_timing",
            "cold_safeguard_telemetry",
        )
    )
    if len(frozenset(numerical_states)) != 1:
        raise Diag5ReceiptConstructionError(
            FailureReasonCodeV5.GROUP_PREFIX_INVALID,
            "DIAG5 atomic scientific subgroup differs",
        )


def _validate_diag5_slots(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV5],
    *,
    failure: StructuredFailureV5,
    gpu_native_binding: NativeBindingV5,
    authority_sha256: str,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> None:
    if tuple(evidence_slots) != tuple(DIAG5_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG5 evidence slots differ from the frozen schema")
    if (
        not expected_logical_snapshot_root.is_absolute()
        or expected_logical_snapshot_root
        != expected_logical_snapshot_root.resolve(strict=False)
        or expected_logical_snapshot_root.name != "source-snapshot"
    ):
        raise ValueError("DIAG5 logical snapshot root differs")
    _validate_diag5_stage_vector(evidence_slots, failure=failure)
    if (
        failure.stage is FailureStageV5.AUTHORITY
        and failure.reason is not FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED
    ):
        raise ValueError("DIAG5 pre-staging authority failure cannot be an artifact")
    _validate_diag5_supervisor_sequence(
        artifact_root,
        evidence_slots,
        failure=failure,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    loaded = _validate_diag5_document_schemas(artifact_root, evidence_slots)
    source = evidence_slots["source_manifest"].artifact
    snapshot: SnapshotPublication | None = None
    if source is not None:
        snapshot = load_snapshot(
            _diag5_held_path(artifact_root, "source-snapshot"),
            required_roles=DIAG5_GPU_SNAPSHOT_ROLES,
        )
        if source.sha256 != snapshot.manifest_sha256:
            raise ValueError("DIAG5 source manifest differs from snapshot")
        _validate_diag5_held_source_snapshot(
            snapshot, expected_source_snapshot_identity
        )
    if "frozen_numerical_subset" in loaded:
        validate_diag5_frozen_numerical_subset_payload(
            loaded["frozen_numerical_subset"],
            artifact_root=artifact_root,
            expected_entries=expected_frozen_numerical_entries,
        )
    if evidence_slots["native_reference"].artifact is not None:
        validation = validate_native_equivalent_reference(
            _diag5_held_path(artifact_root, "native-reference")
        )
        if not validation.usable:
            raise ValueError("DIAG5 native reference is not usable")
    authority: dict[str, JsonValue] | None = None
    if "policy_authority" in loaded:
        authority = validate_diag5_policy_authority_payload(
            loaded["policy_authority"], artifact_root=artifact_root
        )
    for slot_name, stage in (
        ("supervisor_before_preflight", "BEFORE_PREFLIGHT"),
        ("supervisor_before_cold", "BEFORE_COLD"),
    ):
        if slot_name in loaded:
            validate_diag5_supervisor_zero_payload(
                loaded[slot_name],
                artifact_root=artifact_root,
                expected_stage=stage,
                allow_failure=(
                    failure.reason
                    in {
                        FailureReasonCodeV5.SUPERVISOR_GPU_OBSERVATION_INVALID,
                        FailureReasonCodeV5.SUPERVISOR_GPU_NONZERO,
                    }
                ),
            )
    for mode in ("preflight", "cold"):
        _validate_diag5_manifest_only_auxiliaries(
            artifact_root,
            evidence_slots,
            loaded,
            mode=mode,
            authority=authority,
            snapshot=snapshot,
            expected_snapshot_identity=expected_source_snapshot_identity,
            expected_logical_snapshot_root=expected_logical_snapshot_root,
            gpu_native_binding=gpu_native_binding,
            expected_gpu_uuid=expected_gpu_uuid,
            physical_memory_bytes=physical_memory_bytes,
        )
        producer = loaded.get(f"{mode}_producer")
        producer_payload: dict[str, JsonValue] | None = None
        if producer is not None:
            producer_payload = validate_diag5_producer_payload(producer, mode=mode)
        terminal = loaded.get(f"{mode}_terminal")
        if terminal is not None:
            terminal_payload = _mapping(terminal, f"DIAG5 {mode} terminal")
            _exact_keys(
                terminal_payload,
                frozenset(
                    {
                        "schema_version",
                        "terminal_status",
                        "failure_reasons",
                        "monitor_failure_kind",
                    }
                ),
                f"DIAG5 {mode} terminal",
            )
            if (
                terminal_payload["schema_version"]
                != DIAG5_CHILD_TERMINAL_SCHEMA_VERSION
            ):
                raise ValueError(f"DIAG5 {mode} terminal schema differs")
        process = loaded.get(f"{mode}_process")
        if process is not None:
            process_payload = _mapping(process, f"DIAG5 {mode} process")
            expected_process_keys = frozenset(
                {
                    "schema_version",
                    "monitor_failure_kind",
                    "child_pid",
                    "child_start_time_ticks",
                    "argv",
                    "stdout",
                    "stderr",
                    "process_seconds",
                    "process_diagnostics",
                    "pre_source_identity",
                    "post_source_identity",
                    "process_started_monotonic_ns",
                    "process_stopped_monotonic_ns",
                }
            )
            _exact_keys(process_payload, expected_process_keys, f"DIAG5 {mode} process")
            if snapshot is None:
                raise ValueError(f"DIAG5 {mode} process omits its source snapshot")
            _validate_diag5_process_source_identity(
                process_payload,
                snapshot=snapshot,
                context=f"DIAG5 {mode} process",
            )
            terminal_payload = loaded.get(f"{mode}_terminal")
            if terminal_payload is None:
                raise ValueError(f"DIAG5 {mode} process omits child terminal")
            terminal_mapping = _mapping(terminal_payload, f"DIAG5 {mode} terminal")
            if (
                terminal_mapping["monitor_failure_kind"]
                != process_payload["monitor_failure_kind"]
            ):
                raise ValueError(f"DIAG5 {mode} terminal/process monitor kind differs")
            if producer_payload is not None:
                producer_mapping = producer_payload
                _validate_diag5_child_producer_origin(
                    producer_mapping, failure=failure, mode=mode
                )
                runtime_ref = evidence_slots[f"{mode}_runtime"].artifact
                policy_ref = evidence_slots[f"{mode}_policy"].artifact
                if (
                    runtime_ref is not None
                    and _artifact_ref(
                        producer_mapping["runtime_evidence"], "producer runtime"
                    )
                    != runtime_ref
                ) or (
                    policy_ref is not None
                    and "policy_evidence" in producer_mapping
                    and _artifact_ref(
                        producer_mapping["policy_evidence"], "producer policy"
                    )
                    != policy_ref
                ):
                    raise ValueError(f"DIAG5 {mode} producer reference join differs")
                if producer_mapping.get("document_origin") == "PARENT_SUPERVISOR":
                    process_ref = _present_reference(
                        evidence_slots[f"{mode}_process"].artifact,
                        f"{mode}_process",
                    )
                    terminal_ref = _present_reference(
                        evidence_slots[f"{mode}_terminal"].artifact,
                        f"{mode}_terminal",
                    )
                    selected_reason = FailureReasonCodeV5(
                        _string(
                            producer_mapping["selected_failure_reason"],
                            f"DIAG5 {mode} selected reason",
                        )
                    )
                    process_started = _integer(
                        process_payload["process_started_monotonic_ns"],
                        f"DIAG5 {mode} process start",
                    )
                    process_stopped = _integer(
                        process_payload["process_stopped_monotonic_ns"],
                        f"DIAG5 {mode} process stop",
                    )
                    if (
                        selected_reason is not failure.reason
                        or _artifact_ref(
                            producer_mapping["process_evidence"],
                            f"DIAG5 {mode} process ref",
                        )
                        != process_ref
                        or _artifact_ref(
                            producer_mapping["child_terminal_evidence"],
                            f"DIAG5 {mode} terminal ref",
                        )
                        != terminal_ref
                        or _integer(
                            producer_mapping["child_pid"],
                            f"DIAG5 {mode} producer PID",
                        )
                        != _integer(
                            process_payload["child_pid"], f"DIAG5 {mode} process PID"
                        )
                        or _integer(
                            producer_mapping["child_start_time_ticks"],
                            f"DIAG5 {mode} producer start ticks",
                        )
                        != _integer(
                            process_payload["child_start_time_ticks"],
                            f"DIAG5 {mode} process start ticks",
                        )
                        or producer_mapping["process_started_monotonic_ns"]
                        != process_started
                        or producer_mapping["process_stopped_monotonic_ns"]
                        != process_stopped
                    ):
                        raise ValueError(
                            f"DIAG5 {mode} supervisor producer join differs"
                        )
            _validate_diag5_child_outcome(
                terminal_mapping,
                process_payload,
                mode=mode,
                failure=failure,
            )
        runtime_slot = evidence_slots[f"{mode}_runtime"]
        if runtime_slot.artifact is not None:
            validate_diag5_runtime_evidence_v2_bytes(
                _diag5_held_file_bytes(
                    artifact_root, runtime_slot.artifact.relative_path
                ),
                expected_snapshot_identity=expected_source_snapshot_identity,
                expected_logical_campaign_root=expected_logical_snapshot_root.parent,
                expected_logical_snapshot_root=expected_logical_snapshot_root,
                expected_native_extension_path=Path(gpu_native_binding.path),
                expected_native_extension_sha256=gpu_native_binding.sha256,
                expected_native_extension_size_bytes=gpu_native_binding.size_bytes,
                expected_native_extension_link_count=gpu_native_binding.link_count,
            )
        policy_payload = loaded.get(f"{mode}_policy")
        if policy_payload is not None:
            policy = validate_diag5_policy_evidence_payload(policy_payload)
            if authority is not None and policy != _policy_evidence_from_authority(
                authority
            ):
                raise ValueError(f"DIAG5 {mode} policy differs from authority")
        memory_payload = loaded.get(f"{mode}_memory")
        samples_payload = loaded.get(f"{mode}_memory_samples")
        if memory_payload is not None and samples_payload is not None:
            memory = _mapping(memory_payload, f"DIAG5 {mode} memory")
            samples = _mapping(samples_payload, f"DIAG5 {mode} memory samples")
            _validate_diag5_memory_sample_rows(samples, context=f"DIAG5 {mode}")
            process_payload = _mapping(
                loaded[f"{mode}_process"], f"DIAG5 {mode} process"
            )
            argv = tuple(
                _string(item, f"DIAG5 {mode} argv")
                for item in _array(process_payload["argv"], f"DIAG5 {mode} argv")
            )
            _validate_memory(
                memory_payload,
                samples_payload,
                expected_pid=_integer(
                    process_payload["child_pid"], f"DIAG5 {mode} child PID"
                ),
                expected_start_ticks=_integer(
                    process_payload["child_start_time_ticks"],
                    f"DIAG5 {mode} child start ticks",
                ),
                expected_argv=argv,
                expected_gpu_uuid=expected_gpu_uuid,
                physical_memory_bytes=physical_memory_bytes,
                expected_peak_bytes=_integer(
                    memory["peak_memory_bytes"], f"DIAG5 {mode} peak bytes"
                ),
                expected_peak_fraction=_number(
                    memory["peak_memory_fraction"], f"DIAG5 {mode} peak fraction"
                ),
                context=f"DIAG5 {mode} memory",
                expected_memory_schema_version=DIAG5_MEMORY_SCHEMA_VERSION,
                expected_samples_schema_version=(DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION),
            )
    execution = loaded.get("execution")
    if execution is not None:
        execution_payload = _mapping(execution, "DIAG5 execution")
        if (
            _native_binding_v5(execution_payload["gpu_native_binding"], role="gpu")
            != gpu_native_binding
        ):
            raise ValueError("DIAG5 execution native binding differs")
        supporting_names = DIAG5_EVIDENCE_SLOT_NAMES - {
            "execution",
            "supervisor_terminal",
        }
        supporting = {
            name: _present_reference(evidence_slots[name].artifact, name)
            for name in supporting_names
        }
        validate_diag5_execution_evidence_payload(
            execution,
            supporting_evidence=supporting,
            solve_timing=loaded["cold_solve_timing"],
            producer=loaded["cold_producer"],
            process=loaded["cold_process"],
            gpu_native_binding=native_binding_v5_payload(gpu_native_binding),
            authority_sha256=authority_sha256,
        )


def _diag5_scientific_reconstruction(
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV5],
) -> tuple[FailureReasonCodeV5, QualityEvidence, SolveTimingEvidenceV5]:
    refs = {
        name: _present_artifact(EvidenceSlot(slot.state, slot.artifact, None), name)
        for name, slot in evidence_slots.items()
        if name != "supervisor_terminal"
    }
    history = _parse_history(
        _load_ref_json(artifact_root, refs["cold_history"], "DIAG5 history"),
        defer_step_bounds=True,
        expected_schema_version="single-stage-fullspace-neq-gntr3-history-v1",
    )
    terminal_evidence = _validate_gntr3_terminal_numerical_payload(
        artifact_root,
        _load_ref_json(
            artifact_root, refs["cold_terminal_numerical"], "DIAG5 terminal"
        ),
    )
    terminal = terminal_evidence.terminal
    telemetry = validate_diag5_safeguard_telemetry_payload(
        _load_ref_json(
            artifact_root,
            refs["cold_safeguard_telemetry"],
            "DIAG5 safeguard telemetry",
        ),
        history=history,
        expected_history_evidence=refs["cold_history"],
    )
    telemetry_identity = NativeEquivalentNumericalIdentity(
        telemetry.numerical_route,
        telemetry.numerical_result_schema_version,
        telemetry.problem_sha256,
        telemetry.optimizer_options_sha256,
        telemetry.base_neq_gntr1_policy_sha256,
        telemetry.scaling_sha256,
        telemetry.bootstrap_state_sha256,
        telemetry.initial_physical_state_sha256,
        telemetry.identity_sha256,
    )
    if terminal_evidence.numerical_identity != telemetry_identity:
        raise ValueError("DIAG5 terminal/safeguard numerical identity differs")
    policy = _parse_policy(
        _load_ref_json(artifact_root, refs["cold_policy"], "DIAG5 policy"),
        terminal,
        expected_schema_version="single-stage-native-equivalent-quality-policy-v1",
    )
    authority_payload = _validate_policy_authority_payload(
        _load_ref_json(
            artifact_root, refs["policy_authority"], "DIAG5 policy authority"
        ),
        artifact_root=artifact_root,
        schema_version=DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION,
        route=DIAG5_ROUTE,
        plan_sha256=DIAG5_PLAN_SHA256,
    )
    if policy != _policy_evidence_from_authority(authority_payload):
        raise ValueError("DIAG5 cold policy differs from parent authority")
    _validate_native_equalities_authority(artifact_root, terminal)
    _validate_terminal_raw_evidence(terminal, history, policy)
    _validate_quality_replay(history, terminal, policy)
    quality = _quality(terminal)
    numerical_complete = bool(
        not history.fatal
        and history.attempts > 0
        and (
            history.attempts == MAXIMUM_ATTEMPTS
            or history.accepted_steps == MAXIMUM_ACCEPTED_STEPS
            or history.quality_latch
        )
        and _terminal_semantics(history, terminal)
        and quality.residual_value_margin >= 0.0
        and quality.residual_gradient_margin >= 0.0
        and quality.transpose_margin >= 0.0
    )
    hit = bool(
        numerical_complete
        and history.quality_latch
        and quality.passes
        and history.first_quality_attempt > 0
        and history.first_quality_accepted_step > 0
    )
    reason = (
        FailureReasonCodeV5.QUALITY_HIT
        if hit
        else FailureReasonCodeV5.NO_HIT
        if numerical_complete
        else FailureReasonCodeV5.INCOMPLETE
    )
    timing = validate_diag5_solve_timing_evidence_payload(
        _load_ref_json(artifact_root, refs["cold_solve_timing"], "DIAG5 solve timing")
    )
    return reason, quality, timing


def build_diag5_diagnostic_receipt(
    *,
    artifact_root: Path,
    evidence_slots: Mapping[str, EvidenceSlotV5],
    native_bindings: Mapping[str, JsonValue],
    predecessor_postmortem: ArtifactRef,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    """Recompute DIAG5 claims and bind both live native identities."""

    if tuple(evidence_slots) != tuple(DIAG5_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG5 evidence slots differ from the frozen schema")
    parsed_bindings = parse_diag5_native_bindings(native_bindings)
    if parsed_bindings != parse_diag5_native_bindings(expected_native_bindings):
        raise ValueError("DIAG5 native bindings differ from held authority")
    gpu_binding = parsed_bindings[1][1]
    if (
        predecessor_postmortem.relative_path != DIAG5_PREDECESSOR_POSTMORTEM_PATH
        or predecessor_postmortem.schema_version
        != DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
        or predecessor_postmortem != expected_predecessor_postmortem
    ):
        raise ValueError("DIAG5 predecessor postmortem reference differs")
    validate_diag5_predecessor_postmortem_payload(
        _load_ref_json(
            artifact_root, predecessor_postmortem, "DIAG5 predecessor postmortem"
        )
    )
    authority_sha256 = _sha256(expected_authority_sha256, "DIAG5 held authority SHA")
    terminal_ref = _present_artifact(
        EvidenceSlot(
            evidence_slots["supervisor_terminal"].state,
            evidence_slots["supervisor_terminal"].artifact,
            None,
        ),
        "supervisor_terminal",
    )
    terminal_payload, outcome = parse_diag5_supervisor_terminal_payload(
        _load_ref_json(artifact_root, terminal_ref, "DIAG5 supervisor terminal")
    )
    _validate_diag5_slots(
        artifact_root,
        evidence_slots,
        failure=outcome,
        gpu_native_binding=gpu_binding,
        authority_sha256=authority_sha256,
        expected_source_snapshot_identity=expected_source_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        expected_frozen_numerical_entries=expected_frozen_numerical_entries,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
    )
    execution_payload = None
    if evidence_slots["execution"].artifact is not None:
        execution_payload = _mapping(
            _load_ref_json(
                artifact_root,
                evidence_slots["execution"].artifact,
                "DIAG5 execution",
            ),
            "DIAG5 execution",
        )
        if (
            _native_binding_v5(execution_payload.get("gpu_native_binding"), role="gpu")
            != gpu_binding
        ):
            raise ValueError("DIAG5 execution GPU native binding differs")
        if (
            _sha256(execution_payload.get("authority_sha256"), "DIAG5 authority SHA")
            != authority_sha256
        ):
            raise ValueError("DIAG5 execution authority binding differs")
    if outcome.stage is not FailureStageV5.SCIENTIFIC:
        return DiagnosticReceiptV5(
            tuple((name, evidence_slots[name]) for name in DIAG5_EVIDENCE_SLOT_PATHS),
            "DIAGNOSTIC_INCOMPLETE",
            "NOT_COMPARABLE_INCOMPLETE",
            None,
            {"status": "NOT_PRODUCED"},
            "NOT_PRODUCED",
            "NOT_PRODUCED",
            outcome,
            parsed_bindings,
            predecessor_postmortem,
        )
    if any(slot.artifact is None for slot in evidence_slots.values()):
        raise ValueError("DIAG5 scientific outcome requires every evidence slot")
    derived_reason, quality, timing = _diag5_scientific_reconstruction(
        artifact_root, evidence_slots
    )
    if outcome.reason is not derived_reason:
        raise ValueError("DIAG5 scientific terminal differs from reconstruction")
    quality_hit = derived_reason is FailureReasonCodeV5.QUALITY_HIT
    next_route = DIAG4_CONDITIONAL_TIMING_ROUTE if quality_hit else "NOT_PRODUCED"
    if terminal_payload["next_route"] != next_route:
        raise ValueError("DIAG5 supervisor next route differs")
    speed: JsonValue = (
        {
            "status": "NON_FORMAL_ENGINEERING_CONTEXT",
            "synchronized_solve_seconds": timing.synchronized_solve_seconds,
            "historical_threshold_seconds": DIAG4_ENGINEERING_THRESHOLD_SECONDS,
            "observed_ratio_to_historical_threshold": (
                timing.synchronized_solve_seconds / DIAG4_ENGINEERING_THRESHOLD_SECONDS
            ),
        }
        if quality_hit
        else "NOT_PRODUCED"
    )
    return DiagnosticReceiptV5(
        tuple((name, evidence_slots[name]) for name in DIAG5_EVIDENCE_SLOT_PATHS),
        (
            "DIAGNOSTIC_COMPLETE_QUALITY_HIT"
            if quality_hit
            else "DIAGNOSTIC_COMPLETE_NO_HIT"
            if derived_reason is FailureReasonCodeV5.NO_HIT
            else "DIAGNOSTIC_INCOMPLETE"
        ),
        "NOT_COMPARABLE_SUCCESSOR",
        _quality_payload(quality),
        {"status": "NOT_PRODUCED"},
        next_route,
        speed,
        outcome,
        parsed_bindings,
        predecessor_postmortem,
    )


def derive_diag5_scientific_outcome(
    *, artifact_root: Path, artifact_refs: Mapping[str, ArtifactRef | None]
) -> StructuredFailureV5:
    if tuple(artifact_refs) != tuple(DIAG5_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG5 scientific refs differ from the frozen slot schema")
    if artifact_refs["supervisor_terminal"] is not None or any(
        artifact_refs[name] is None
        for name in DIAG5_EVIDENCE_SLOT_PATHS
        if name != "supervisor_terminal"
    ):
        raise ValueError("DIAG5 scientific reconstruction requires committed prefix")
    slots = {
        name: (
            EvidenceSlotV5.absent()
            if name == "supervisor_terminal"
            else EvidenceSlotV5.present(_present_reference(reference, name))
        )
        for name, reference in artifact_refs.items()
    }
    reason, _, _ = _diag5_scientific_reconstruction(artifact_root, slots)
    detail_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "reason": reason.value,
                "evidence": {
                    name: _artifact_ref_payload(_present_reference(reference, name))
                    for name, reference in artifact_refs.items()
                    if name != "supervisor_terminal"
                },
            }
        )
    ).hexdigest()
    return StructuredFailureV5(FailureStageV5.SCIENTIFIC, reason, detail_sha256)


def diag5_diagnostic_receipt_payload(
    receipt: DiagnosticReceiptV5,
) -> dict[str, JsonValue]:
    return {
        "schema_version": DIAG5_SCHEMA_VERSION,
        "route": DIAG5_ROUTE,
        "numerical_route": DIAG5_NUMERICAL_ROUTE,
        "plan_sha256": DIAG5_PLAN_SHA256,
        "evidence_slots": {
            name: diag5_evidence_slot_payload(slot)
            for name, slot in receipt.evidence_slots
        },
        "verdict": receipt.verdict,
        "historical_relation": receipt.historical_relation,
        "quality": receipt.quality,
        "phase_attribution": receipt.phase_attribution,
        "next_route": receipt.next_route,
        "speed_comparison": receipt.speed_comparison,
        "terminal_outcome": diag5_terminal_outcome_payload(receipt.failure),
        "native_bindings": diag5_native_bindings_payload(receipt.native_bindings),
        "predecessor_postmortem": _artifact_ref_payload(receipt.predecessor_postmortem),
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }


def diag5_diagnostic_receipt_bytes(receipt: DiagnosticReceiptV5) -> bytes:
    return canonical_json_bytes(diag5_diagnostic_receipt_payload(receipt))


def diag5_diagnostic_receipt_from_payload(
    value: JsonValue,
    *,
    artifact_root: Path,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    payload = _mapping(value, "DIAG5 diagnostic receipt")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "plan_sha256",
                "evidence_slots",
                "verdict",
                "historical_relation",
                "quality",
                "phase_attribution",
                "next_route",
                "speed_comparison",
                "terminal_outcome",
                "native_bindings",
                "predecessor_postmortem",
                "promotion_authorized",
                "formal_comparison",
            }
        ),
        "DIAG5 diagnostic receipt",
    )
    if (
        payload["schema_version"] != DIAG5_SCHEMA_VERSION
        or payload["route"] != DIAG5_ROUTE
        or payload["numerical_route"] != DIAG5_NUMERICAL_ROUTE
        or payload["plan_sha256"] != DIAG5_PLAN_SHA256
        or payload["promotion_authorized"] is not False
        or payload["formal_comparison"] != "NOT_PRODUCED"
    ):
        raise ValueError("DIAG5 diagnostic identity differs")
    raw_slots = _mapping(payload["evidence_slots"], "DIAG5 evidence slots")
    if frozenset(raw_slots) != frozenset(DIAG5_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG5 evidence slot keys differ")
    slots = {
        name: parse_diag5_evidence_slot(raw_slots[name], name=name)
        for name in DIAG5_EVIDENCE_SLOT_PATHS
    }
    rebuilt = build_diag5_diagnostic_receipt(
        artifact_root=artifact_root,
        evidence_slots=slots,
        native_bindings=_mapping(payload["native_bindings"], "native bindings"),
        predecessor_postmortem=_artifact_ref(
            payload["predecessor_postmortem"], "predecessor postmortem"
        ),
        expected_native_bindings=expected_native_bindings,
        expected_authority_sha256=expected_authority_sha256,
        expected_predecessor_postmortem=expected_predecessor_postmortem,
        expected_source_snapshot_identity=expected_source_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        expected_frozen_numerical_entries=expected_frozen_numerical_entries,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
    )
    if parse_diag5_terminal_outcome(payload["terminal_outcome"]) != rebuilt.failure:
        raise ValueError("DIAG5 receipt terminal outcome differs")
    if payload != diag5_diagnostic_receipt_payload(rebuilt):
        raise ValueError("DIAG5 diagnostic claims differ from raw evidence")
    return rebuilt


def load_diag5_diagnostic_receipt_bytes(
    data: bytes,
    *,
    artifact_root: Path,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    return diag5_diagnostic_receipt_from_payload(
        load_canonical_json_bytes(data),
        artifact_root=artifact_root,
        expected_native_bindings=expected_native_bindings,
        expected_authority_sha256=expected_authority_sha256,
        expected_predecessor_postmortem=expected_predecessor_postmortem,
        expected_source_snapshot_identity=expected_source_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        expected_frozen_numerical_entries=expected_frozen_numerical_entries,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
    )


def _diag5_receipt_slots(root: Path) -> dict[str, EvidenceSlotV5]:
    held = _DIAG5_HELD_TREE.get()
    payload = _mapping(
        load_canonical_json_bytes(
            held.file_bytes(DIAG2_RECEIPT_FILENAME)
            if held is not None
            else (root / DIAG2_RECEIPT_FILENAME).read_bytes()
        ),
        "DIAG5 diagnostic receipt",
    )
    if payload.get("schema_version") != DIAG5_SCHEMA_VERSION:
        raise ValueError("DIAG5 receipt schema differs")
    raw_slots = _mapping(payload.get("evidence_slots"), "DIAG5 evidence slots")
    if frozenset(raw_slots) != frozenset(DIAG5_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG5 evidence slot keys differ")
    return {
        name: parse_diag5_evidence_slot(raw_slots[name], name=name)
        for name in DIAG5_EVIDENCE_SLOT_PATHS
    }


def _diag5_add_child_custody_roles(
    root: Path,
    roles: dict[str, str],
    slots: Mapping[str, EvidenceSlotV5],
    *,
    mode: str,
    outcome: StructuredFailureV5,
) -> None:
    producer_ref = slots[f"{mode}_producer"].artifact
    if producer_ref is None:
        return
    producer = validate_diag5_producer_payload(
        _load_ref_json(root, producer_ref, f"DIAG5 {mode} producer"), mode=mode
    )
    parent_supervisor = producer.get("document_origin") == "PARENT_SUPERVISOR"
    selected_reason = (
        FailureReasonCodeV5(
            _string(
                producer["selected_failure_reason"],
                f"DIAG5 {mode} producer selected reason",
            )
        )
        if parent_supervisor
        else None
    )
    if parent_supervisor and selected_reason is not outcome.reason:
        raise ValueError(f"DIAG5 {mode} custody reason differs")
    allowed_auxiliary_reason = outcome.reason in {
        FailureReasonCodeV5.PREFLIGHT_TIMEOUT,
        FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED,
        FailureReasonCodeV5.PREFLIGHT_EXIT_NONZERO,
        FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
        FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
        FailureReasonCodeV5.COLD_TIMEOUT,
        FailureReasonCodeV5.COLD_MONITOR_FAILED,
        FailureReasonCodeV5.COLD_EXIT_NONZERO,
        FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
        FailureReasonCodeV5.COLD_PRODUCER_INVALID,
    }
    invalid_relative = f"{mode}/invalid-producer.bin"
    invalid_path = root / invalid_relative
    held = _DIAG5_HELD_TREE.get()
    invalid_entry = held.entries.get(invalid_relative) if held is not None else None
    if invalid_entry is not None if held is not None else os.path.lexists(invalid_path):
        if not parent_supervisor or selected_reason not in {
            FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
            FailureReasonCodeV5.COLD_PRODUCER_INVALID,
        }:
            raise ValueError(f"DIAG5 {mode} invalid producer custody differs")
        if held is not None:
            invalid_regular = (
                invalid_entry is not None and not invalid_entry.is_directory
            )
            invalid_links = invalid_entry.link_count if invalid_entry is not None else 0
        else:
            metadata = invalid_path.lstat()
            invalid_regular = stat.S_ISREG(metadata.st_mode)
            invalid_links = metadata.st_nlink
        if not invalid_regular or invalid_links != 1:
            raise ValueError(f"DIAG5 {mode} invalid producer is not one regular inode")
        roles[invalid_relative] = f"{mode}_invalid_producer"
    auxiliary = (
        ("memory", "gpu-memory.json", DIAG5_MEMORY_SCHEMA_VERSION),
        (
            "memory_samples",
            "gpu-memory-samples.json",
            DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
        ),
        ("runtime", "runtime-evidence.json", RUNTIME_EVIDENCE_V2_SCHEMA_VERSION),
        (
            "policy",
            "policy.json",
            "single-stage-native-equivalent-quality-policy-v1",
        ),
    )
    present_auxiliary: dict[str, ArtifactRef] = {}
    for suffix, filename, schema in auxiliary:
        relative = f"{mode}/{filename}"
        path = root / relative
        typed = slots[f"{mode}_{suffix}"].artifact
        if typed is not None:
            continue
        path_exists = (
            relative in held.entries if held is not None else os.path.lexists(path)
        )
        if not path_exists:
            continue
        if not allowed_auxiliary_reason:
            raise ValueError(f"DIAG5 {mode} auxiliary contradicts terminal outcome")
        if held is None and (not path.is_file() or path.is_symlink()):
            raise ValueError(f"DIAG5 {mode} auxiliary path is not a regular file")
        payload = _mapping(
            load_canonical_json_bytes(_diag5_held_file_bytes(root, relative)),
            f"DIAG5 {mode} auxiliary {suffix}",
        )
        if payload.get("schema_version") != schema:
            raise ValueError(f"DIAG5 {mode} auxiliary {suffix} schema differs")
        if suffix == "policy":
            validate_diag5_policy_evidence_payload(payload)
        data = _diag5_held_file_bytes(root, relative)
        reference = ArtifactRef(
            relative,
            hashlib.sha256(data).hexdigest(),
            len(data),
            schema,
        )
        present_auxiliary[suffix] = reference
        roles[relative] = f"{mode}_{suffix}"
    if ("memory" in present_auxiliary) != ("memory_samples" in present_auxiliary):
        raise ValueError(f"DIAG5 {mode} auxiliary memory pairing differs")
    for suffix, field in (
        ("runtime", "runtime_evidence"),
        ("policy", "policy_evidence"),
    ):
        auxiliary_ref = present_auxiliary.get(suffix)
        if (
            auxiliary_ref is not None
            and field in producer
            and _artifact_ref(producer[field], f"DIAG5 {mode} producer {field}")
            != auxiliary_ref
        ):
            raise ValueError(f"DIAG5 {mode} producer auxiliary join differs")


def _diag5_add_quarantine_roles(
    root: Path,
    roles: dict[str, str],
    *,
    outcome: StructuredFailureV5,
) -> None:
    quarantine = root / DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY
    empty_marker = root / DIAG5_EMPTY_QUARANTINE_PATH
    held = _DIAG5_HELD_TREE.get()
    quarantine_entry = (
        held.entries.get(DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY)
        if held is not None
        else None
    )
    quarantine_exists = (
        quarantine_entry is not None
        if held is not None
        else os.path.lexists(quarantine)
    )
    marker_exists = (
        DIAG5_EMPTY_QUARANTINE_PATH in held.entries
        if held is not None
        else os.path.lexists(empty_marker)
    )
    if not quarantine_exists:
        if marker_exists:
            raise ValueError("DIAG5 empty quarantine marker omits its directory")
        return
    if (
        held is not None
        and (quarantine_entry is None or not quarantine_entry.is_directory)
    ) or (held is None and (quarantine.is_symlink() or not quarantine.is_dir())):
        raise ValueError("DIAG5 quarantine is not a nonsymlink directory")
    allowed = (
        outcome.stage is FailureStageV5.COLD
        and outcome.reason
        in {
            FailureReasonCodeV5.COLD_TIMEOUT,
            FailureReasonCodeV5.COLD_MONITOR_FAILED,
            FailureReasonCodeV5.COLD_EXIT_NONZERO,
            FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
            FailureReasonCodeV5.COLD_PRODUCER_INVALID,
        }
    ) or (
        outcome.stage is FailureStageV5.NUMERICAL_COMMIT
        and outcome.reason is FailureReasonCodeV5.PENDING_RESULT_INVALID
    )
    retained = (
        [
            relative
            for relative, entry in held.entries.items()
            if relative.startswith(f"{DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY}/")
            and not entry.is_directory
        ]
        if held is not None
        else [
            path.relative_to(root).as_posix()
            for path in quarantine.rglob("*")
            if path.is_file()
        ]
    )
    if not allowed:
        raise ValueError("DIAG5 opaque quarantine contradicts terminal outcome")
    if retained:
        if marker_exists:
            raise ValueError("DIAG5 nonempty quarantine retains an empty marker")
        for relative in retained:
            roles[relative] = "uncommitted_cold_numerical_result"
        return
    if not marker_exists or (
        held is None and (not empty_marker.is_file() or empty_marker.is_symlink())
    ):
        raise ValueError("DIAG5 empty quarantine omits its canonical marker")
    marker = _mapping(
        load_canonical_json_bytes(
            held.file_bytes(DIAG5_EMPTY_QUARANTINE_PATH)
            if held is not None
            else empty_marker.read_bytes()
        ),
        "DIAG5 empty quarantine marker",
    )
    _exact_keys(
        marker,
        frozenset(
            {
                "schema_version",
                "route",
                "quarantine_relative_path",
                "selected_failure_reason",
            }
        ),
        "DIAG5 empty quarantine marker",
    )
    if (
        marker["schema_version"] != DIAG5_EMPTY_QUARANTINE_SCHEMA_VERSION
        or marker["route"] != DIAG5_ROUTE
        or marker["quarantine_relative_path"] != DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY
        or marker["selected_failure_reason"] != outcome.reason.value
    ):
        raise ValueError("DIAG5 empty quarantine marker differs")
    roles[DIAG5_EMPTY_QUARANTINE_PATH] = "empty_uncommitted_cold_numerical_result"


def _diag5_artifact_roles(root: Path) -> dict[str, str]:
    pending = root / DIAG5_PENDING_NUMERICAL_DIRECTORY
    held = _DIAG5_HELD_TREE.get()
    if (
        DIAG5_PENDING_NUMERICAL_DIRECTORY in held.entries
        if held is not None
        else os.path.lexists(pending)
    ):
        raise ValueError("DIAG5 pending numerical result reached receipt sealing")
    slots = _diag5_receipt_slots(root)
    terminal_ref = slots["supervisor_terminal"].artifact
    if terminal_ref is None:
        raise ValueError("DIAG5 artifact omits supervisor terminal")
    _, outcome = parse_diag5_supervisor_terminal_payload(
        _load_ref_json(root, terminal_ref, "DIAG5 supervisor terminal")
    )
    roles = {
        DIAG2_RECEIPT_FILENAME: "diagnostic_receipt",
        DIAG5_PREDECESSOR_POSTMORTEM_PATH: "predecessor_postmortem",
    }
    for name, slot in slots.items():
        if slot.artifact is not None:
            roles[slot.artifact.relative_path] = name
    for name, prefix in (("preflight_process", "preflight"), ("cold_process", "cold")):
        reference = slots[name].artifact
        if reference is not None:
            _diag2_add_process_roles(root, roles, reference, prefix)
    for mode in ("preflight", "cold"):
        _diag5_add_child_custody_roles(root, roles, slots, mode=mode, outcome=outcome)
    for name, stage in (
        ("supervisor_before_preflight", "before-preflight"),
        ("supervisor_before_cold", "before-cold"),
    ):
        reference = slots[name].artifact
        if reference is None:
            continue
        payload = _mapping(_load_ref_json(root, reference, name), name)
        for query_name, query_role in (
            ("gpu_inventory_query", "gpu-inventory"),
            ("compute_apps_query", "compute-apps"),
        ):
            query = _mapping(payload[query_name], f"{name}.{query_name}")
            for stream in ("stdout", "stderr"):
                stream_ref = _artifact_ref(
                    query[stream], f"{name}.{query_name}.{stream}"
                )
                expected = f"supervisor/{stage}-{query_role}.{stream}.bin"
                if stream_ref.relative_path != expected:
                    raise ValueError("DIAG5 supervisor raw query path differs")
                _resolve_artifact(root, stream_ref)
                roles[expected] = f"{name}_{query_role}_{stream}"
    terminal_reference = slots["cold_terminal_numerical"].artifact
    if terminal_reference is not None:
        terminal = _mapping(
            _load_ref_json(root, terminal_reference, "DIAG5 terminal"),
            "DIAG5 terminal",
        )
        arrays = _mapping(terminal["arrays"], "DIAG5 terminal arrays")
        _exact_keys(arrays, frozenset(ARRAY_SPECS), "DIAG5 terminal arrays")
        for name in ARRAY_SPECS:
            row = _mapping(arrays[name], f"DIAG5 terminal arrays.{name}")
            reference = _artifact_ref(
                row["artifact"], f"DIAG5 terminal arrays.{name}.artifact"
            )
            _resolve_artifact(root, reference)
            roles[reference.relative_path] = "terminal_array"
    for tree, role in (
        ("source-snapshot", "source_snapshot"),
        ("native-reference", "native_reference"),
    ):
        tree_root = root / tree
        tree_exists = tree in held.entries if held is not None else tree_root.is_dir()
        if tree_exists:
            slot_name = (
                "source_manifest" if tree == "source-snapshot" else "native_reference"
            )
            if slots[slot_name].artifact is None:
                raise ValueError(f"DIAG5 absent {slot_name} retains a physical tree")
            relatives = (
                (
                    relative
                    for relative, entry in held.entries.items()
                    if relative.startswith(f"{tree}/") and not entry.is_directory
                )
                if held is not None
                else (
                    path.relative_to(root).as_posix()
                    for path in tree_root.rglob("*")
                    if path.is_file()
                )
            )
            for relative in relatives:
                roles.setdefault(relative, role)
    _diag5_add_quarantine_roles(root, roles, outcome=outcome)
    observed = (
        {
            relative
            for relative, entry in held.entries.items()
            if not entry.is_directory and relative != DIAG2_MANIFEST_FILENAME
        }
        if held is not None
        else {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != root / DIAG2_MANIFEST_FILENAME
        }
    )
    if any(_forbidden_trace_path(relative) for relative in observed):
        raise ValueError("DIAG5 artifact contains forbidden trace evidence")
    if observed != frozenset(roles):
        raise ValueError("DIAG5 artifact contains an unknown or missing typed path")
    directories = (
        (relative for relative, entry in held.entries.items() if entry.is_directory)
        if held is not None
        else (
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    for relative in directories:
        canonical_empty_quarantine = (
            relative == DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY
            and roles.get(DIAG5_EMPTY_QUARANTINE_PATH)
            == "empty_uncommitted_cold_numerical_result"
        )
        if not canonical_empty_quarantine and not any(
            path.startswith(f"{relative}/") for path in roles
        ):
            raise ValueError("DIAG5 artifact contains an empty or alternate directory")
    return roles


def diag5_artifact_manifest_payload(root: Path) -> dict[str, JsonValue]:
    roles = _diag5_artifact_roles(root.resolve(strict=True))
    entries: list[dict[str, JsonValue]] = []
    for relative, role in sorted(roles.items()):
        data = (root / relative).read_bytes()
        entries.append(
            {
                "relative_path": relative,
                "role": role,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    return {"schema_version": DIAG5_MANIFEST_SCHEMA_VERSION, "entries": entries}


def _validate_diag5_held_manifest(
    held: _Diag5HeldTree, roles: Mapping[str, str]
) -> None:
    manifest = _mapping(
        load_canonical_json_bytes(held.file_bytes(DIAG2_MANIFEST_FILENAME)),
        "DIAG5 held manifest",
    )
    _exact_keys(manifest, frozenset({"schema_version", "entries"}), "DIAG5 manifest")
    if manifest["schema_version"] != DIAG5_MANIFEST_SCHEMA_VERSION:
        raise ValueError("DIAG5 manifest schema differs")
    declared: list[str] = []
    for index, item in enumerate(_array(manifest["entries"], "manifest.entries")):
        context = f"manifest.entries[{index}]"
        row = _mapping(item, context)
        _exact_keys(
            row,
            frozenset({"relative_path", "role", "sha256", "size_bytes"}),
            context,
        )
        relative = _string(row["relative_path"], f"{context}.relative_path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or relative == DIAG2_MANIFEST_FILENAME
            or roles.get(relative) != _string(row["role"], f"{context}.role")
        ):
            raise ValueError(f"{context} identity differs")
        data = held.file_bytes(relative)
        if len(data) != _integer(
            row["size_bytes"], f"{context}.size_bytes"
        ) or hashlib.sha256(data).hexdigest() != _sha256(
            row["sha256"], f"{context}.sha256"
        ):
            raise ValueError(f"{context} held bytes differ")
        declared.append(relative)
    observed = frozenset(
        relative
        for relative, entry in held.entries.items()
        if not entry.is_directory and relative != DIAG2_MANIFEST_FILENAME
    )
    if (
        declared != sorted(declared)
        or len(declared) != len(set(declared))
        or observed != frozenset(declared)
        or observed != frozenset(roles)
    ):
        raise ValueError("DIAG5 held manifest path closure differs")


def _validate_diag5_publication_root(root: Path, *, staging: bool) -> None:
    slots = _diag5_receipt_slots(root)
    terminal_ref = slots["supervisor_terminal"].artifact
    if terminal_ref is None:
        raise ValueError("DIAG5 artifact omits supervisor terminal")
    terminal, _ = parse_diag5_supervisor_terminal_payload(
        _load_ref_json(root, terminal_ref, "DIAG5 supervisor terminal")
    )
    publication = _mapping(terminal["publication"], "DIAG5 publication")
    expected_key = "staging_root" if staging else "final_root"
    expected = Path(_string(publication[expected_key], expected_key)).resolve(
        strict=False
    )
    if root.resolve(strict=True) != expected:
        raise ValueError(f"DIAG5 artifact root differs from publication.{expected_key}")


def _validate_diag5_tree(
    artifact_root: Path,
    *,
    staging: bool,
    require_sealed: bool,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    resolved = artifact_root.resolve(strict=True)
    with _Diag5HeldTree(resolved, require_sealed=require_sealed) as held:
        token = _DIAG5_HELD_TREE.set(held)
        try:
            _validate_diag5_publication_root(resolved, staging=staging)
            roles = _diag5_artifact_roles(resolved)
            _validate_diag5_held_manifest(held, roles)
            if staging:
                _validate_diag2_live_supervisor_identity(
                    resolved, _slots_loader=_diag5_receipt_slots
                )
            receipt = load_diag5_diagnostic_receipt_bytes(
                held.file_bytes(DIAG2_RECEIPT_FILENAME),
                artifact_root=resolved,
                expected_native_bindings=expected_native_bindings,
                expected_authority_sha256=expected_authority_sha256,
                expected_predecessor_postmortem=expected_predecessor_postmortem,
                expected_source_snapshot_identity=expected_source_snapshot_identity,
                expected_logical_snapshot_root=expected_logical_snapshot_root,
                expected_frozen_numerical_entries=(expected_frozen_numerical_entries),
                expected_gpu_uuid=expected_gpu_uuid,
                physical_memory_bytes=physical_memory_bytes,
            )
            held.revalidate_path_bindings()
            return receipt
        finally:
            _DIAG5_HELD_TREE.reset(token)


def validate_diag5_writable_staging(
    artifact_root: Path,
    *,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    return _validate_diag5_tree(
        artifact_root,
        staging=True,
        require_sealed=False,
        expected_native_bindings=expected_native_bindings,
        expected_authority_sha256=expected_authority_sha256,
        expected_predecessor_postmortem=expected_predecessor_postmortem,
        expected_source_snapshot_identity=expected_source_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        expected_frozen_numerical_entries=expected_frozen_numerical_entries,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
    )


def load_and_validate_diag5_staging(
    artifact_root: Path,
    *,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    return _validate_diag5_tree(
        artifact_root,
        staging=True,
        require_sealed=True,
        expected_native_bindings=expected_native_bindings,
        expected_authority_sha256=expected_authority_sha256,
        expected_predecessor_postmortem=expected_predecessor_postmortem,
        expected_source_snapshot_identity=expected_source_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        expected_frozen_numerical_entries=expected_frozen_numerical_entries,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
    )


def load_and_validate_diag5_artifact(
    artifact_root: Path,
    *,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    return _validate_diag5_tree(
        artifact_root,
        staging=False,
        require_sealed=True,
        expected_native_bindings=expected_native_bindings,
        expected_authority_sha256=expected_authority_sha256,
        expected_predecessor_postmortem=expected_predecessor_postmortem,
        expected_source_snapshot_identity=expected_source_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        expected_frozen_numerical_entries=expected_frozen_numerical_entries,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
    )


def load_and_validate_diag5_rollback(
    rollback_root: Path,
    *,
    expected_rollback_root: Path,
    expected_final_root: Path,
    expected_native_bindings: Mapping[str, JsonValue],
    expected_authority_sha256: str,
    expected_predecessor_postmortem: ArtifactRef,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    """Deep-load sealed bytes moved from the final root into exact rollback."""

    if rollback_root.is_symlink():
        raise ValueError("DIAG5 rollback root must not be a symlink")
    resolved = rollback_root.resolve(strict=True)
    if resolved != expected_rollback_root.resolve(strict=False):
        raise ValueError("DIAG5 rollback root identity differs")
    with _Diag5HeldTree(resolved, require_sealed=True) as held:
        token = _DIAG5_HELD_TREE.set(held)
        try:
            slots = _diag5_receipt_slots(resolved)
            terminal_ref = slots["supervisor_terminal"].artifact
            if terminal_ref is None:
                raise ValueError("DIAG5 rollback omits supervisor terminal")
            terminal, _ = parse_diag5_supervisor_terminal_payload(
                _load_ref_json(
                    resolved, terminal_ref, "DIAG5 rollback supervisor terminal"
                )
            )
            publication = _mapping(
                terminal["publication"], "DIAG5 rollback publication"
            )
            if Path(_string(publication["final_root"], "DIAG5 final root")).resolve(
                strict=False
            ) != expected_final_root.resolve(strict=False):
                raise ValueError("DIAG5 rollback final-root identity differs")
            roles = _diag5_artifact_roles(resolved)
            _validate_diag5_held_manifest(held, roles)
            receipt = load_diag5_diagnostic_receipt_bytes(
                held.file_bytes(DIAG2_RECEIPT_FILENAME),
                artifact_root=resolved,
                expected_native_bindings=expected_native_bindings,
                expected_authority_sha256=expected_authority_sha256,
                expected_predecessor_postmortem=expected_predecessor_postmortem,
                expected_source_snapshot_identity=expected_source_snapshot_identity,
                expected_logical_snapshot_root=expected_logical_snapshot_root,
                expected_frozen_numerical_entries=(expected_frozen_numerical_entries),
                expected_gpu_uuid=expected_gpu_uuid,
                physical_memory_bytes=physical_memory_bytes,
            )
            held.revalidate_path_bindings()
            return receipt
        finally:
            _DIAG5_HELD_TREE.reset(token)


def derive_diag5_evidence_slots(
    *,
    artifact_root: Path,
    artifact_refs: Mapping[str, ArtifactRef | None],
    outcome: StructuredFailureV5,
    gpu_native_binding: Mapping[str, JsonValue],
    authority_sha256: str,
    expected_source_snapshot_identity: SnapshotIdentity,
    expected_logical_snapshot_root: Path,
    expected_frozen_numerical_entries: Mapping[str, str],
    expected_gpu_uuid: str,
    physical_memory_bytes: int,
) -> dict[str, EvidenceSlotV5]:
    if tuple(artifact_refs) != tuple(DIAG5_EVIDENCE_SLOT_PATHS):
        raise ValueError("DIAG5 artifact refs differ from the frozen slot schema")
    numerical_names = (
        "cold_history",
        "cold_terminal_numerical",
        "cold_solve_timing",
        "cold_safeguard_telemetry",
    )
    numerical_present = tuple(
        artifact_refs[name] is not None for name in numerical_names
    )
    if len(frozenset(numerical_present)) != 1:
        raise ValueError("DIAG5 atomic scientific subgroup refs differ")
    if outcome.stage is FailureStageV5.SCIENTIFIC and any(
        reference is None for reference in artifact_refs.values()
    ):
        raise ValueError("DIAG5 scientific outcome requires every artifact ref")
    slots: dict[str, EvidenceSlotV5] = {}
    first_absence = True
    for name in DIAG5_EVIDENCE_SLOT_PATHS:
        reference = artifact_refs[name]
        if reference is not None:
            if reference.schema_version != DIAG5_EVIDENCE_SLOT_SCHEMAS[name]:
                raise ValueError(f"DIAG5 {name} reference schema differs")
            slots[name] = EvidenceSlotV5.present(reference)
        else:
            slots[name] = EvidenceSlotV5.absent(
                outcome.reason if first_absence else None
            )
            if name not in {"execution", "supervisor_terminal"}:
                first_absence = False
    _validate_diag5_slots(
        artifact_root,
        slots,
        failure=outcome,
        gpu_native_binding=_native_binding_v5(gpu_native_binding, role="gpu"),
        authority_sha256=_sha256(authority_sha256, "DIAG5 authority SHA"),
        expected_source_snapshot_identity=expected_source_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        expected_frozen_numerical_entries=expected_frozen_numerical_entries,
        expected_gpu_uuid=expected_gpu_uuid,
        physical_memory_bytes=physical_memory_bytes,
    )
    return slots


__all__ = (
    "ARRAY_SPECS",
    "DIAG2_BASELINE_FILTERED_ENTRIES_SHA256",
    "DIAG2_BASELINE_FILTERED_ENTRY_COUNT",
    "DIAG2_CHILD_TERMINAL_SCHEMA_VERSION",
    "DIAG2_EVIDENCE_SLOT_NAMES",
    "DIAG2_EVIDENCE_SLOT_PATHS",
    "DIAG2_EXECUTED_DIAG1_SOURCE_MANIFEST_SHA256",
    "DIAG2_FAILURE_STAGE_ORDER",
    "DIAG2_FROZEN_NUMERICAL_ENTRIES",
    "DIAG2_FROZEN_SUBSET_SCHEMA_VERSION",
    "DIAG2_MANIFEST_FILENAME",
    "DIAG2_MANIFEST_SCHEMA_VERSION",
    "DIAG2_NUMERICAL_ROUTE",
    "DIAG2_PLAN_SHA256",
    "DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION",
    "DIAG2_POLICY_SHA256",
    "DIAG2_PROCESS_SCHEMA_VERSION",
    "DIAG2_RECEIPT_FILENAME",
    "DIAG2_ROUTE",
    "DIAG2_SCALE_SHA256",
    "DIAG2_SCHEMA_VERSION",
    "DIAG2_SOURCE_DELTA_ALLOWLIST",
    "DIAG2_STAGE_REASON_CODES",
    "DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION",
    "DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION",
    "DIAG2_VOLUME_TARGET_HEX",
    "DIAG3_COLD_RESULT_SCHEMA_VERSION",
    "DIAG3_COMMITTED_NUMERICAL_DIRECTORY",
    "DIAG3_EVIDENCE_SLOT_PATHS",
    "DIAG3_MANIFEST_SCHEMA_VERSION",
    "DIAG3_PENDING_NUMERICAL_DIRECTORY",
    "DIAG3_SCHEMA_VERSION",
    "DIAG3_UNCOMMITTED_NUMERICAL_DIRECTORY",
    "DIAG4_COLD_RESULT_SCHEMA_VERSION",
    "DIAG4_CONDITIONAL_TIMING_ROUTE",
    "DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS",
    "DIAG4_ENDPOINT_OBSERVABLE_FIELDS",
    "DIAG4_ENGINEERING_THRESHOLD_SECONDS",
    "DIAG4_EVIDENCE_SLOT_NAMES",
    "DIAG4_EVIDENCE_SLOT_PATHS",
    "DIAG4_EXECUTION_SCHEMA_VERSION",
    "DIAG4_FAILURE_STAGE_ORDER",
    "DIAG4_MANIFEST_SCHEMA_VERSION",
    "DIAG4_MAXIMUM_NONLINEAR_CORRECTIONS",
    "DIAG4_NUMERICAL_BUNDLE_SCHEMA_VERSION",
    "DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION",
    "DIAG4_NUMERICAL_ROUTE",
    "DIAG4_OUTER_TELEMETRY_FIELDS",
    "DIAG4_PLAN_SHA256",
    "DIAG4_PREFLIGHT_SCHEMA_VERSION",
    "DIAG4_PROFILER_CALL_AUDIT",
    "DIAG4_ROUTE",
    "DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION",
    "DIAG4_SCHEMA_VERSION",
    "DIAG4_SOLVE_TIMING_SCHEMA_VERSION",
    "DIAG4_STAGE_REASON_ORDER",
    "DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS",
    "DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS",
    "DIAG4_SUBTRIAL_MATRIX_FIELDS",
    "DIAG4_SUBTRIAL_SUMMARY_FIELDS",
    "DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION",
    "DIAG5_CHILD_TERMINAL_SCHEMA_VERSION",
    "DIAG5_COLD_RESULT_SCHEMA_VERSION",
    "DIAG5_EMPTY_QUARANTINE_PATH",
    "DIAG5_EMPTY_QUARANTINE_SCHEMA_VERSION",
    "DIAG5_EVIDENCE_SLOT_NAMES",
    "DIAG5_EVIDENCE_SLOT_PATHS",
    "DIAG5_EVIDENCE_SLOT_SCHEMAS",
    "DIAG5_EXECUTION_SCHEMA_VERSION",
    "DIAG5_FAILURE_STAGE_ORDER",
    "DIAG5_FROZEN_SUBSET_SCHEMA_VERSION",
    "DIAG5_MANIFEST_SCHEMA_VERSION",
    "DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION",
    "DIAG5_MEMORY_SCHEMA_VERSION",
    "DIAG5_NUMERICAL_BUNDLE_SCHEMA_VERSION",
    "DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION",
    "DIAG5_NUMERICAL_ROUTE",
    "DIAG5_PENDING_NUMERICAL_DIRECTORY",
    "DIAG5_PLAN_SHA256",
    "DIAG5_POLICY_AUTHORITY_SCHEMA_VERSION",
    "DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256",
    "DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256",
    "DIAG5_PREDECESSOR_POSTMORTEM_PATH",
    "DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION",
    "DIAG5_PREFLIGHT_SCHEMA_VERSION",
    "DIAG5_PROFILER_CALL_AUDIT",
    "DIAG5_ROUTE",
    "DIAG5_SAFEGUARD_TELEMETRY_SCHEMA_VERSION",
    "DIAG5_SCHEMA_VERSION",
    "DIAG5_SOLVE_TIMING_SCHEMA_VERSION",
    "DIAG5_STAGE_REASON_ORDER",
    "DIAG5_STAGE_REASON_PRESENT_PREFIXES",
    "DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION",
    "DIAG5_SUPERVISOR_ZERO_SCHEMA_VERSION",
    "DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY",
    "EVIDENCE_REF_KEYS",
    "EVIDENCE_ROLE_PATHS",
    "FINAL_CERTIFICATE_FIELDS",
    "FIXED_ARTIFACT_ROLES",
    "FROZEN_GNTR_OPTIONS",
    "GPU_UUID",
    "HISTORY_FLOAT_FIELDS",
    "HISTORY_INTEGER_FIELDS",
    "HISTORY_ROW_RAW_FIELDS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "MAXIMUM_ACCEPTED_STEPS",
    "MAXIMUM_ATTEMPTS",
    "NUMERICAL_ROUTE",
    "PHASE_IDS",
    "PHASE_SCHEMA_SHA256",
    "PLAN_SHA256",
    "POLICY_RAW_HASH_FIELDS",
    "POLICY_RAW_SCALAR_FIELDS",
    "POLICY_RAW_VECTOR_FIELDS",
    "PREFLIGHT_EVIDENCE_REF_KEYS",
    "RECEIPT_FILENAME",
    "REQUIRED_SOURCE_ROLE_BINDINGS",
    "ROUTE",
    "SCHEMA_VERSION",
    "TERMINAL_RAW_SCALAR_FIELDS",
    "TRACE_LOOP_ENVELOPE_NAME",
    "AbsenceReason",
    "ArrayEvidence",
    "AttemptOutcome",
    "Diag2ColdEvidenceClassification",
    "Diag2PreflightGateError",
    "Diag2SetupGateError",
    "Diag4ColdEvidenceClassification",
    "Diag4NumericalDocumentError",
    "Diag4ProfilerCallAudit",
    "Diag5NumericalDocumentError",
    "Diag5ProfilerCallAudit",
    "Diag5ReceiptConstructionError",
    "DiagnosticReceipt",
    "DiagnosticReceiptV2",
    "DiagnosticReceiptV4",
    "DiagnosticReceiptV5",
    "DiagnosticVerdict",
    "EvidenceSlot",
    "EvidenceSlotV4",
    "EvidenceSlotV5",
    "EvidenceState",
    "ExecutionEvidence",
    "FailureReasonCodeV2",
    "FailureReasonCodeV4",
    "FailureReasonCodeV5",
    "FailureStage",
    "FailureStageV2",
    "FailureStageV4",
    "FailureStageV5",
    "HistoricalAggregateRelation",
    "HistoryEvidence",
    "HistoryRow",
    "IncompleteDiagnosticReceipt",
    "KktStatus",
    "NativeBindingV5",
    "NativeEquivalentNumericalIdentity",
    "NativeEquivalentScientificEvidence",
    "NextRoute",
    "PhaseEvidence",
    "PhaseTimingStatus",
    "PolicyEvidence",
    "QualityEvidence",
    "SafeguardTelemetryV4",
    "SafeguardTelemetryV5",
    "ScientificOutcome",
    "SolveTimingEvidenceV5",
    "StructuredFailureV2",
    "StructuredFailureV4",
    "StructuredFailureV5",
    "SupervisorQueryV2",
    "TerminalEvidence",
    "TerminalEvidenceV4",
    "array_evidence_payload",
    "build_diag2_compile_failure_producer_payload",
    "build_diag2_diagnostic_receipt",
    "build_diag2_frozen_numerical_subset_payload",
    "build_diag2_policy_authority_payload",
    "build_diag2_supervisor_terminal_payload",
    "build_diag2_supervisor_zero_payload",
    "build_diag3_diagnostic_receipt",
    "build_diag4_diagnostic_receipt",
    "build_diag4_frozen_numerical_subset_payload",
    "build_diag4_supervisor_terminal_payload",
    "build_diag5_compile_failure_producer_payload",
    "build_diag5_diagnostic_receipt",
    "build_diag5_frozen_numerical_subset_payload",
    "build_diag5_policy_authority_payload",
    "build_diag5_supervisor_failure_producer_payload",
    "build_diag5_supervisor_terminal_payload",
    "build_diag5_supervisor_zero_payload",
    "build_diagnostic_receipt",
    "build_incomplete_diagnostic_receipt",
    "build_native_equivalent_scientific_evidence",
    "classify_diag2_cold_evidence",
    "classify_diag2_subordinate_child_outcome",
    "classify_diag3_cold_evidence",
    "classify_diag3_subordinate_child_outcome",
    "classify_diag4_cold_evidence",
    "classify_diag5_receipt_construction_error",
    "derive_diag2_algorithm_route",
    "derive_diag2_evidence_slots",
    "derive_diag3_algorithm_route",
    "derive_diag3_evidence_slots",
    "derive_diag4_algorithm_route",
    "derive_diag4_evidence_slots",
    "derive_diag4_scientific_outcome",
    "derive_diag5_evidence_slots",
    "derive_diag5_scientific_outcome",
    "diag2_artifact_manifest_payload",
    "diag2_diagnostic_receipt_bytes",
    "diag2_diagnostic_receipt_from_payload",
    "diag2_diagnostic_receipt_payload",
    "diag2_evidence_slot_payload",
    "diag2_postlaunch_setup_failure",
    "diag3_artifact_manifest_payload",
    "diag3_diagnostic_receipt_bytes",
    "diag3_diagnostic_receipt_from_payload",
    "diag3_diagnostic_receipt_payload",
    "diag4_artifact_manifest_payload",
    "diag4_diagnostic_receipt_bytes",
    "diag4_diagnostic_receipt_from_payload",
    "diag4_diagnostic_receipt_payload",
    "diag4_evidence_slot_payload",
    "diag4_execution_evidence_payload",
    "diag4_profiler_call_audit_payload",
    "diag4_terminal_numerical_payload",
    "diag4_terminal_outcome_payload",
    "diag5_artifact_manifest_payload",
    "diag5_diagnostic_receipt_bytes",
    "diag5_diagnostic_receipt_from_payload",
    "diag5_diagnostic_receipt_payload",
    "diag5_evidence_slot_payload",
    "diag5_execution_evidence_payload",
    "diag5_history_evidence_from_arrays",
    "diag5_native_bindings_payload",
    "diag5_policy_evidence_payload",
    "diag5_safeguard_telemetry_payload",
    "diag5_solve_timing_evidence_payload",
    "diag5_terminal_outcome_payload",
    "diagnostic_artifact_manifest_payload",
    "diagnostic_receipt_bytes",
    "diagnostic_receipt_from_payload",
    "diagnostic_receipt_payload",
    "execution_evidence_payload",
    "history_evidence_from_arrays",
    "history_evidence_payload",
    "history_row",
    "load_and_validate_diag2_artifact",
    "load_and_validate_diag2_staging",
    "load_and_validate_diag3_artifact",
    "load_and_validate_diag3_staging",
    "load_and_validate_diag4_artifact",
    "load_and_validate_diag4_staging",
    "load_and_validate_diag5_artifact",
    "load_and_validate_diag5_rollback",
    "load_and_validate_diag5_staging",
    "load_and_validate_diagnostic_artifact",
    "load_diag2_diagnostic_receipt_bytes",
    "load_diag3_diagnostic_receipt_bytes",
    "load_diag4_diagnostic_receipt_bytes",
    "load_diag5_diagnostic_receipt_bytes",
    "load_diagnostic_receipt_bytes",
    "native_binding_v5_payload",
    "normalize_chrome_trace",
    "parse_diag2_evidence_slot",
    "parse_diag3_evidence_slot",
    "parse_diag4_evidence_slot",
    "parse_diag4_supervisor_terminal_payload",
    "parse_diag4_terminal_outcome",
    "parse_diag5_evidence_slot",
    "parse_diag5_native_bindings",
    "parse_diag5_supervisor_terminal_payload",
    "parse_diag5_terminal_outcome",
    "policy_evidence_payload",
    "raw_trace_payload",
    "safeguard_telemetry_payload",
    "select_diag2_failure",
    "select_diag4_terminal_outcome",
    "select_diag5_terminal_outcome",
    "solve_timing_evidence_payload",
    "terminal_numerical_payload",
    "validate_diag2_frozen_numerical_subset_payload",
    "validate_diag2_policy_authority_payload",
    "validate_diag2_preflight_gate",
    "validate_diag2_producer_payload",
    "validate_diag2_setup_authorities",
    "validate_diag2_source_snapshot_authority",
    "validate_diag2_supervisor_zero_payload",
    "validate_diag2_writable_staging",
    "validate_diag3_producer_payload",
    "validate_diag3_writable_staging",
    "validate_diag4_execution_evidence_payload",
    "validate_diag4_frozen_numerical_subset_payload",
    "validate_diag4_numerical_documents",
    "validate_diag4_preflight_gate",
    "validate_diag4_producer_payload",
    "validate_diag4_terminal_numerical_payload",
    "validate_diag4_writable_staging",
    "validate_diag5_execution_evidence_payload",
    "validate_diag5_frozen_numerical_subset_payload",
    "validate_diag5_history_evidence_payload",
    "validate_diag5_numerical_documents",
    "validate_diag5_policy_authority_payload",
    "validate_diag5_policy_evidence_payload",
    "validate_diag5_predecessor_postmortem_payload",
    "validate_diag5_preflight_gate",
    "validate_diag5_producer_payload",
    "validate_diag5_safeguard_telemetry_payload",
    "validate_diag5_solve_timing_evidence_payload",
    "validate_diag5_supervisor_failure_producer_payload",
    "validate_diag5_supervisor_zero_payload",
    "validate_diag5_writable_staging",
    "validate_diagnostic_preflight_gate",
    "validate_endpoint_audit_evidence_payload",
    "validate_history_evidence_payload",
    "validate_native_equivalent_scientific_evidence",
    "validate_policy_evidence_payload",
    "validate_safeguard_telemetry_payload",
    "validate_solve_timing_evidence_payload",
    "validate_terminal_endpoint_audit",
    "validate_terminal_numerical_payload",
)
