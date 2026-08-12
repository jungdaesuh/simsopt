"""Fail-closed receipt contract for the additive NEQ-GNTR1 campaign.

The module owns evidence parsing and terminal adjudication, not solver objects.
Producers cross this boundary with canonical JSON mappings; consumers receive
frozen typed receipts whose verdicts are always recomputed from raw evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import asdict, dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Final

import numpy as np
from simsopt_jax.objectives.single_stage_fullspace import TERM_LEDGER
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NativeEquivalentQualityPolicy,
)

from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    JsonValue,
    canonical_json_bytes,
    load_canonical_json_bytes,
    load_snapshot,
    validate_runtime_evidence,
)
from benchmarks.single_stage_native_equivalent_endpoint_audit import (
    endpoint_audit_from_payload,
    validate_endpoint_audit_payload,
)
from benchmarks.single_stage_native_equivalent_reference import (
    REFERENCE_FILENAME,
    REFERENCE_NOT_PRODUCED,
    validate_native_equivalent_reference,
)
from benchmarks.single_stage_native_equivalent_reference import (
    SCHEMA_VERSION as NATIVE_REFERENCE_SCHEMA_VERSION,
)

PLAN_SHA256: Final = "d082baa587b9db580ac3ef8c99a3123ed83564586b605200f7c2cfa6feb909a9"
ROUTE: Final = "NEQ-GNTR1"
REFERENCE_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-reference-receipt-v2"
SAMPLE_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-sample-receipt-v2"
CAMPAIGN_SCHEMA_VERSION: Final = "single-stage-neq-gntr1-campaign-receipt-v2"
NATIVE_RECEIPT_SHA256: Final = (
    "8118529751f184f60f0c4d26f338cd1832aae579004d62866fb2a2f6617e9fe4"
)
NATIVE_TRAJECTORY_SHA256: Final = (
    "fa81b533b7bd8127b021bc2aa206c01914f91a3ef2e34eee6e0636e2031fed8f"
)
OBJECTIVE_MAXIMUM: Final = 4.4822246533126125e-08
SCALED_FEASIBILITY_MAXIMUM: Final = 1.0e-10
RESIDUAL_VALUE_DEFECT_MAXIMUM: Final = 1.0e-12
RESIDUAL_GRADIENT_DEFECT_MAXIMUM: Final = 1.0e-10
TRANSPOSE_DEFECT_MAXIMUM: Final = 1.0e-10
EQUALITY_ABSOLUTE_TOLERANCE: Final = 1.0e-12
EQUALITY_RELATIVE_TOLERANCE: Final = 1.0e-10
WARM_SOLVE_MAXIMUM_SECONDS: Final = 287.30421751597896
MAXIMUM_MEMORY_FRACTION: Final = 0.8
STATE_SIZE: Final = 716
EQUALITY_SIZE: Final = 255
GPU_UUID: Final = "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
CAMPAIGN_ARTIFACT_MANIFEST_FILENAME: Final = "artifact-manifest.json"
CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-campaign-artifact-manifest-v1"
)

_LOWER_HEX: Final = frozenset("0123456789abcdef")


class SampleName(StrEnum):
    COLD = "cold"
    WARM_1 = "warm-1"
    WARM_2 = "warm-2"
    WARM_3 = "warm-3"


class ExecutionStatus(StrEnum):
    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    COMPILE_FAILURE = "COMPILE_FAILURE"
    CRASH = "CRASH"
    INCOMPLETE = "INCOMPLETE"


class KktTelemetryStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NONFINITE = "NONFINITE"


class SampleQuality(StrEnum):
    NATIVE_EQUIVALENT_QUALITY = "NATIVE_EQUIVALENT_QUALITY"
    DEVICE_QUALITY_CANDIDATE = "DEVICE_QUALITY_CANDIDATE"
    QUALITY_NOT_REACHED = "QUALITY_NOT_REACHED"
    NOT_PRODUCED = "NOT_PRODUCED"


class EngineeringDisposition(StrEnum):
    ENGINEERING_SPEED_GOAL_ACHIEVED = "ENGINEERING_SPEED_GOAL_ACHIEVED"
    ENGINEERING_SPEED_GOAL_NOT_ACHIEVED = "ENGINEERING_SPEED_GOAL_NOT_ACHIEVED"
    QUALITY_NOT_REACHED_BOUNDED_NEGATIVE = "QUALITY_NOT_REACHED_BOUNDED_NEGATIVE"
    REFERENCE_NOT_PRODUCED = "REFERENCE_NOT_PRODUCED"
    ENDPOINT_AUDIT_FAILED_NONPROMOTING = "ENDPOINT_AUDIT_FAILED_NONPROMOTING"
    NOT_PRODUCED = "NOT_PRODUCED"


def _mapping(value: JsonValue, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _exact_keys(
    value: dict[str, JsonValue], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{context} keys differ from the frozen schema")


def _string(value: JsonValue, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _boolean(value: JsonValue, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a boolean")
    return value


def _integer(value: JsonValue, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
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
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return result


def _optional_sha256(value: JsonValue, context: str) -> str | None:
    return None if value is None else _sha256(value, context)


def _require_sha256(value: str, context: str) -> None:
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{context} must be a lowercase SHA-256")


def _require_git_head(value: str) -> None:
    if len(value) != 40 or any(character not in _LOWER_HEX for character in value):
        raise ValueError("source git head must be a lowercase 40-hex object ID")


def _git_head(value: JsonValue) -> str:
    result = _string(value, "source git head")
    _require_git_head(result)
    return result


def _canonical_digest(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _artifact_ref_from_payload(value: JsonValue, context: str) -> ArtifactRef:
    payload = _mapping(value, context)
    _exact_keys(
        payload,
        frozenset({"relative_path", "sha256", "size_bytes", "schema_version"}),
        context,
    )
    reference = ArtifactRef(
        relative_path=_string(payload["relative_path"], f"{context}.relative_path"),
        sha256=_sha256(payload["sha256"], f"{context}.sha256"),
        size_bytes=_integer(payload["size_bytes"], f"{context}.size_bytes"),
        schema_version=_string(payload["schema_version"], f"{context}.schema_version"),
    )
    _validate_artifact_ref(reference, context)
    return reference


def _validate_artifact_ref(reference: ArtifactRef, context: str) -> None:
    path = Path(reference.relative_path)
    if (
        not reference.relative_path
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != reference.relative_path
        or reference.size_bytes <= 0
        or not reference.schema_version
    ):
        raise ValueError(f"{context} is not a canonical artifact reference")
    _require_sha256(reference.sha256, f"{context}.sha256")


def _artifact_ref_payload(reference: ArtifactRef) -> dict[str, JsonValue]:
    return {
        "relative_path": reference.relative_path,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "schema_version": reference.schema_version,
    }


def _float_vector(value: JsonValue, size: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{context} must contain exactly {size} values")
    return tuple(
        _number(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _strings(value: JsonValue, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return tuple(
        _string(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class SourceIdentityEvidence:
    git_head: str
    tracked_diff_sha256: str
    untracked_bytes_manifest_sha256: str
    source_manifest_sha256: str
    source_manifest_size_bytes: int

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "git_head": self.git_head,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_manifest_size_bytes": self.source_manifest_size_bytes,
            "tracked_diff_sha256": self.tracked_diff_sha256,
            "untracked_bytes_manifest_sha256": self.untracked_bytes_manifest_sha256,
        }

    def validate(self) -> None:
        _require_git_head(self.git_head)
        for value, context in (
            (self.tracked_diff_sha256, "source tracked diff"),
            (self.untracked_bytes_manifest_sha256, "source untracked manifest"),
            (self.source_manifest_sha256, "source manifest"),
        ):
            _require_sha256(value, context)
        if self.source_manifest_size_bytes <= 0:
            raise ValueError("source manifest size must be positive")


@dataclass(frozen=True, slots=True)
class NativeBranchEvidence:
    path_256_successful_knots: int
    path_512_successful_knots: int
    common_knot_count: int
    maximum_common_knot_difference: float
    maximum_scaled_feasibility_256: float
    maximum_scaled_feasibility_512: float
    path_256_terminal_state_sha256: str
    path_512_terminal_state_sha256: str
    first_failing_index: int | None

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "common_knot_count": self.common_knot_count,
            "first_failing_index": self.first_failing_index,
            "maximum_common_knot_difference": self.maximum_common_knot_difference,
            "maximum_scaled_feasibility_256": self.maximum_scaled_feasibility_256,
            "maximum_scaled_feasibility_512": self.maximum_scaled_feasibility_512,
            "path_256_successful_knots": self.path_256_successful_knots,
            "path_256_terminal_state_sha256": self.path_256_terminal_state_sha256,
            "path_512_successful_knots": self.path_512_successful_knots,
            "path_512_terminal_state_sha256": self.path_512_terminal_state_sha256,
        }

    def passes(self, physical_state_sha256: str) -> bool:
        return bool(
            self.path_256_successful_knots == 257
            and self.path_512_successful_knots == 513
            and self.common_knot_count == 257
            and self.maximum_common_knot_difference >= 0.0
            and self.maximum_common_knot_difference <= 1.0e-10
            and self.maximum_scaled_feasibility_256 >= 0.0
            and self.maximum_scaled_feasibility_256 <= SCALED_FEASIBILITY_MAXIMUM
            and self.maximum_scaled_feasibility_512 >= 0.0
            and self.maximum_scaled_feasibility_512 <= SCALED_FEASIBILITY_MAXIMUM
            and self.path_256_terminal_state_sha256 == physical_state_sha256
            and self.path_512_terminal_state_sha256 == physical_state_sha256
            and self.first_failing_index is None
        )


@dataclass(frozen=True, slots=True)
class FiniteArrayEvidence:
    dtype: str
    size: int
    nonfinite_count: int

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "dtype": self.dtype,
            "nonfinite_count": self.nonfinite_count,
            "size": self.size,
        }

    def passes(self, expected_size: int) -> bool:
        return (
            self.dtype == "float64"
            and self.size == expected_size
            and self.nonfinite_count == 0
        )


@dataclass(frozen=True, slots=True)
class AcceptedLedgerEvidence:
    row_capacity: int
    valid_row_count: int
    accepted_step_count: int
    bootstrap_state_sha256: str
    latched_state_sha256: str
    rejected_rows_retained: int

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "accepted_step_count": self.accepted_step_count,
            "bootstrap_state_sha256": self.bootstrap_state_sha256,
            "latched_state_sha256": self.latched_state_sha256,
            "rejected_rows_retained": self.rejected_rows_retained,
            "row_capacity": self.row_capacity,
            "valid_row_count": self.valid_row_count,
        }

    def passes(self, candidate: CandidateEvidence, reference: ReferenceReceipt) -> bool:
        return bool(
            self.row_capacity == 257
            and self.valid_row_count == self.accepted_step_count + 1
            and self.accepted_step_count == candidate.accepted_step_count
            and self.bootstrap_state_sha256 == reference.bootstrap_state_sha256
            and self.latched_state_sha256 == candidate.state_sha256
            and self.rejected_rows_retained == 0
        )


@dataclass(frozen=True, slots=True)
class BranchReplayEvidence:
    replayed_row_count: int
    row0_state_sha256: str
    successful_direct_rows: int
    successful_midpoint_refined_rows: int
    maximum_direct_refined_difference: float
    maximum_gpu_native_difference: float
    maximum_scaled_feasibility: float
    first_failing_row: int | None

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "first_failing_row": self.first_failing_row,
            "maximum_direct_refined_difference": self.maximum_direct_refined_difference,
            "maximum_gpu_native_difference": self.maximum_gpu_native_difference,
            "maximum_scaled_feasibility": self.maximum_scaled_feasibility,
            "replayed_row_count": self.replayed_row_count,
            "row0_state_sha256": self.row0_state_sha256,
            "successful_direct_rows": self.successful_direct_rows,
            "successful_midpoint_refined_rows": self.successful_midpoint_refined_rows,
        }

    def passes(
        self, ledger: AcceptedLedgerEvidence, reference: ReferenceReceipt
    ) -> bool:
        return bool(
            self.replayed_row_count == ledger.valid_row_count
            and self.row0_state_sha256 == reference.bootstrap_state_sha256
            and self.successful_direct_rows == ledger.valid_row_count
            and self.successful_midpoint_refined_rows == ledger.valid_row_count - 1
            and self.maximum_direct_refined_difference >= 0.0
            and self.maximum_direct_refined_difference <= 1.0e-10
            and self.maximum_gpu_native_difference >= 0.0
            and self.maximum_gpu_native_difference <= 1.0e-10
            and self.maximum_scaled_feasibility >= 0.0
            and self.maximum_scaled_feasibility <= SCALED_FEASIBILITY_MAXIMUM
            and self.first_failing_row is None
        )


@dataclass(frozen=True, slots=True)
class CrossEvaluatorEvidence:
    native_on_gpu_objective: float
    jax_on_gpu_objective: float
    native_on_native_objective: float
    jax_on_native_objective: float
    gpu_raw_term_maximum_difference: float
    gpu_raw_term_maximum_magnitude: float
    native_raw_term_maximum_difference: float
    native_raw_term_maximum_magnitude: float

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "gpu_raw_term_maximum_difference": self.gpu_raw_term_maximum_difference,
            "gpu_raw_term_maximum_magnitude": self.gpu_raw_term_maximum_magnitude,
            "jax_on_gpu_objective": self.jax_on_gpu_objective,
            "jax_on_native_objective": self.jax_on_native_objective,
            "native_on_gpu_objective": self.native_on_gpu_objective,
            "native_on_native_objective": self.native_on_native_objective,
            "native_raw_term_maximum_difference": self.native_raw_term_maximum_difference,
            "native_raw_term_maximum_magnitude": self.native_raw_term_maximum_magnitude,
        }

    def passes(self) -> bool:
        objective_pairs = (
            (self.native_on_gpu_objective, self.jax_on_gpu_objective),
            (self.native_on_native_objective, self.jax_on_native_objective),
        )
        return bool(
            all(
                math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)
                for left, right in objective_pairs
            )
            and self.gpu_raw_term_maximum_difference >= 0.0
            and self.gpu_raw_term_maximum_magnitude >= 0.0
            and self.gpu_raw_term_maximum_difference
            <= 1.0e-15 + 1.0e-12 * self.gpu_raw_term_maximum_magnitude
            and self.native_raw_term_maximum_difference >= 0.0
            and self.native_raw_term_maximum_magnitude >= 0.0
            and self.native_raw_term_maximum_difference
            <= 1.0e-15 + 1.0e-12 * self.native_raw_term_maximum_magnitude
        )


@dataclass(frozen=True, slots=True)
class ReferenceReceipt:
    produced: bool
    reference_evidence: ArtifactRef
    reference_policy_sha256: str | None
    native_receipt_sha256: str
    native_trajectory_sha256: str
    bootstrap_state_sha256: str | None
    physical_state_sha256: str | None
    raw_equalities: tuple[float, ...]
    constraint_inverse_scale: tuple[float, ...]
    ledger_identity_sha256: str | None
    native_branch_evidence_sha256: str | None
    native_branch_evidence: NativeBranchEvidence | None
    failure_reasons: tuple[str, ...]

    def validate(self) -> None:
        _validate_artifact_ref(self.reference_evidence, "reference evidence")
        _require_sha256(self.native_receipt_sha256, "reference native receipt")
        _require_sha256(self.native_trajectory_sha256, "reference native trajectory")
        if self.native_receipt_sha256 != NATIVE_RECEIPT_SHA256:
            raise ValueError("reference native receipt identity differs")
        if self.native_trajectory_sha256 != NATIVE_TRAJECTORY_SHA256:
            raise ValueError("reference native trajectory identity differs")
        if self.produced:
            hashes = (
                self.reference_policy_sha256,
                self.bootstrap_state_sha256,
                self.physical_state_sha256,
                self.ledger_identity_sha256,
                self.native_branch_evidence_sha256,
            )
            if any(value is None for value in hashes):
                raise ValueError("produced reference requires every frozen identity")
            for value in hashes:
                assert value is not None
                _require_sha256(value, "produced reference identity")
            if len(self.raw_equalities) != EQUALITY_SIZE:
                raise ValueError("produced reference requires 255 raw equalities")
            if any(not math.isfinite(value) for value in self.raw_equalities):
                raise ValueError("produced reference raw equalities must be finite")
            if len(self.constraint_inverse_scale) != EQUALITY_SIZE or any(
                not math.isfinite(value) or value == 0.0
                for value in self.constraint_inverse_scale
            ):
                raise ValueError("produced reference requires finite nonzero scaling")
            if self.failure_reasons:
                raise ValueError("produced reference cannot contain failure reasons")
            if self.native_branch_evidence is None:
                raise ValueError(
                    "produced reference requires raw native branch evidence"
                )
            if self.native_branch_evidence_sha256 != _canonical_digest(
                self.native_branch_evidence.to_payload()
            ):
                raise ValueError(
                    "native branch evidence digest differs from raw evidence"
                )
            assert self.physical_state_sha256 is not None
            if not self.native_branch_evidence.passes(self.physical_state_sha256):
                raise ValueError(
                    "produced reference fails the independent 256/512 gate"
                )
        elif (
            self.reference_policy_sha256 is not None
            or self.bootstrap_state_sha256 is not None
            or self.physical_state_sha256 is not None
            or self.raw_equalities
            or self.constraint_inverse_scale
            or self.ledger_identity_sha256 is not None
            or self.native_branch_evidence_sha256 is not None
            or self.native_branch_evidence is not None
            or not self.failure_reasons
        ):
            raise ValueError("unproduced reference must contain only failure evidence")


def _reference_artifact_bytes(
    artifact_root: Path,
    value: JsonValue,
    context: str,
    *,
    digest_key: str,
    expected_keys: frozenset[str],
) -> bytes:
    reference = _mapping(value, context)
    _exact_keys(reference, expected_keys, context)
    relative_path = _string(reference["relative_path"], f"{context}.relative_path")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        raise ValueError(f"{context} path is not canonical")
    payload = artifact_root.joinpath(path).read_bytes()
    if len(payload) != _integer(
        reference["size_bytes"], f"{context}.size_bytes"
    ) or hashlib.sha256(payload).hexdigest() != _sha256(
        reference[digest_key], f"{context}.{digest_key}"
    ):
        raise ValueError(f"{context} bytes differ from their reference")
    return payload


def _reference_array(
    artifact_root: Path,
    value: JsonValue,
    context: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    reference = _mapping(value, context)
    payload = _reference_artifact_bytes(
        artifact_root,
        value,
        context,
        digest_key="file_sha256",
        expected_keys=frozenset(
            {
                "content_sha256",
                "dtype",
                "file_sha256",
                "order",
                "relative_path",
                "shape",
                "size_bytes",
            }
        ),
    )
    array = np.load(BytesIO(payload), allow_pickle=False)
    if (
        array.dtype != np.dtype("<f8")
        or array.shape != shape
        or not array.flags.c_contiguous
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{context} is not the required finite FP64 C array")
    content_sha256 = hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f8").tobytes()
    ).hexdigest()
    if reference.get("content_sha256") != content_sha256:
        raise ValueError(f"{context} content identity differs")
    return np.array(array, copy=True)


def _objective_ledger_sha256() -> str:
    payload = json.dumps(
        [asdict(row) for row in TERM_LEDGER],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping_steps(value: JsonValue, context: str) -> tuple[dict[str, JsonValue], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a nonempty array")
    return tuple(
        _mapping(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def reference_receipt_from_artifact(
    *,
    artifact_root: Path,
    reference_evidence: ArtifactRef,
    bootstrap_state: object,
    constraint_inverse_scale: object,
) -> ReferenceReceipt:
    """Build the receipt authority from one independently validated reference tree."""

    validation = validate_native_equivalent_reference(artifact_root)
    reference_bytes = artifact_root.joinpath(REFERENCE_FILENAME).read_bytes()
    _validate_artifact_ref(reference_evidence, "reference evidence")
    if (
        Path(reference_evidence.relative_path).name != REFERENCE_FILENAME
        or reference_evidence.sha256 != hashlib.sha256(reference_bytes).hexdigest()
        or reference_evidence.size_bytes != len(reference_bytes)
        or reference_evidence.schema_version != NATIVE_REFERENCE_SCHEMA_VERSION
    ):
        raise ValueError(
            "reference evidence does not identify validated reference.json"
        )
    if validation.disposition == REFERENCE_NOT_PRODUCED:
        reasons = validation.failure_reasons or (REFERENCE_NOT_PRODUCED,)
        receipt = ReferenceReceipt(
            produced=False,
            reference_evidence=reference_evidence,
            reference_policy_sha256=None,
            native_receipt_sha256=NATIVE_RECEIPT_SHA256,
            native_trajectory_sha256=NATIVE_TRAJECTORY_SHA256,
            bootstrap_state_sha256=None,
            physical_state_sha256=None,
            raw_equalities=(),
            constraint_inverse_scale=(),
            ledger_identity_sha256=None,
            native_branch_evidence_sha256=None,
            native_branch_evidence=None,
            failure_reasons=tuple(reasons),
        )
        receipt.validate()
        return receipt
    document = _mapping(load_canonical_json_bytes(reference_bytes), "reference.json")
    evidence = _mapping(document["evidence"], "reference evidence payload")
    arrays = _mapping(evidence["arrays"], "reference arrays")
    state = _reference_array(artifact_root, arrays["state"], "state", (STATE_SIZE,))
    raw_equalities = _reference_array(
        artifact_root,
        arrays["raw_equalities"],
        "raw equalities",
        (EQUALITY_SIZE,),
    )
    coarse = _reference_array(
        artifact_root, arrays["coarse_roots"], "coarse roots", (257, 255)
    )
    refined = _reference_array(
        artifact_root, arrays["refined_roots"], "refined roots", (513, 255)
    )
    observed_bootstrap = np.asarray(bootstrap_state)
    observed_inverse_scale = np.asarray(constraint_inverse_scale)
    if (
        observed_bootstrap.dtype != np.dtype(np.float64)
        or observed_inverse_scale.dtype != np.dtype(np.float64)
        or observed_bootstrap.shape != (STATE_SIZE,)
        or observed_inverse_scale.shape != (EQUALITY_SIZE,)
        or not np.all(np.isfinite(observed_bootstrap))
        or not np.all(np.isfinite(observed_inverse_scale))
        or np.any(observed_inverse_scale == 0.0)
    ):
        raise ValueError("bootstrap state or constraint inverse scale is invalid")
    bootstrap = np.ascontiguousarray(observed_bootstrap, dtype="<f8")
    inverse_scale = np.ascontiguousarray(observed_inverse_scale, dtype="<f8")
    derived_policy = NativeEquivalentQualityPolicy(
        raw_equalities,
        exact_numeric_tree_sha256(raw_equalities),
        inverse_scale,
    )
    diagnostics_bytes = _reference_artifact_bytes(
        artifact_root,
        document["diagnostics"],
        "reference diagnostics",
        digest_key="sha256",
        expected_keys=frozenset({"relative_path", "sha256", "size_bytes"}),
    )
    diagnostics = _mapping(
        load_canonical_json_bytes(diagnostics_bytes), "reference diagnostics"
    )
    coarse_steps = _mapping_steps(diagnostics["coarse_steps"], "coarse steps")
    refined_steps = _mapping_steps(diagnostics["refined_steps"], "refined steps")
    maximum_coarse_feasibility = max(
        _number(step["scaled_boozer_infinity_norm"], "scaled Boozer feasibility")
        for step in coarse_steps
    )
    maximum_refined_feasibility = max(
        _number(step["scaled_boozer_infinity_norm"], "scaled Boozer feasibility")
        for step in refined_steps
    )
    state_sha256 = hashlib.sha256(state.tobytes()).hexdigest()
    branch = NativeBranchEvidence(
        path_256_successful_knots=coarse.shape[0],
        path_512_successful_knots=refined.shape[0],
        common_knot_count=coarse.shape[0],
        maximum_common_knot_difference=float(np.max(np.abs(coarse - refined[::2]))),
        maximum_scaled_feasibility_256=maximum_coarse_feasibility,
        maximum_scaled_feasibility_512=maximum_refined_feasibility,
        path_256_terminal_state_sha256=state_sha256,
        path_512_terminal_state_sha256=state_sha256,
        first_failing_index=None,
    )
    receipt = ReferenceReceipt(
        produced=True,
        reference_evidence=reference_evidence,
        reference_policy_sha256=derived_policy.policy_sha256,
        native_receipt_sha256=NATIVE_RECEIPT_SHA256,
        native_trajectory_sha256=NATIVE_TRAJECTORY_SHA256,
        bootstrap_state_sha256=hashlib.sha256(bootstrap.tobytes()).hexdigest(),
        physical_state_sha256=state_sha256,
        raw_equalities=tuple(float(value) for value in raw_equalities),
        constraint_inverse_scale=tuple(float(value) for value in inverse_scale),
        ledger_identity_sha256=_objective_ledger_sha256(),
        native_branch_evidence_sha256=_canonical_digest(branch.to_payload()),
        native_branch_evidence=branch,
        failure_reasons=(),
    )
    receipt.validate()
    return receipt


@dataclass(frozen=True, slots=True)
class TimingEvidence:
    compile_completed_ns: int
    device_state_ready_ns: int
    timer_started_ns: int
    first_hit_synchronized_ns: int | None
    timer_stopped_ns: int
    audit_started_ns: int | None
    final_transfer_ns: int | None
    serialized_ns: int
    synchronized_solve_seconds: float
    endpoint_audit_seconds: float | None
    total_process_seconds: float

    def valid(self, candidate_reached: bool) -> bool:
        ordered_prefix = (
            0
            <= self.compile_completed_ns
            <= self.device_state_ready_ns
            <= self.timer_started_ns
            <= self.timer_stopped_ns
            <= self.serialized_ns
        )
        solve_matches = math.isclose(
            self.synchronized_solve_seconds,
            (self.timer_stopped_ns - self.timer_started_ns) / 1.0e9,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        total_valid = (
            self.synchronized_solve_seconds >= 0.0
            and self.total_process_seconds >= self.synchronized_solve_seconds
        )
        hit_valid = (
            self.first_hit_synchronized_ns is not None
            and self.first_hit_synchronized_ns == self.timer_stopped_ns
            if candidate_reached
            else self.first_hit_synchronized_ns is None
        )
        audit_valid = self.audit_started_ns is None or (
            self.audit_started_ns >= self.timer_stopped_ns
            and self.final_transfer_ns is not None
            and self.final_transfer_ns >= self.audit_started_ns
            and self.serialized_ns >= self.final_transfer_ns
            and self.endpoint_audit_seconds is not None
            and self.endpoint_audit_seconds >= 0.0
        )
        return (
            ordered_prefix
            and solve_matches
            and total_valid
            and hit_valid
            and audit_valid
        )


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    reached: bool
    first_hit_attempt: int | None
    first_hit_accepted_step: int | None
    accepted_step_count: int
    state_sha256: str | None
    physical_objective: float | None
    raw_equalities: tuple[float, ...]
    scaled_equalities: tuple[float, ...]
    scaled_feasibility_inf: float | None
    state_dtype: str | None
    equality_dtype: str | None
    correction_certified: bool

    def passes(self, reference: ReferenceReceipt) -> bool:
        if not self.reached or not reference.produced:
            return False
        assert self.first_hit_attempt is not None
        assert self.first_hit_accepted_step is not None
        assert self.physical_objective is not None
        assert self.scaled_feasibility_inf is not None
        return bool(
            self.first_hit_attempt >= 1
            and self.first_hit_attempt <= 300
            and 1 <= self.first_hit_accepted_step <= self.accepted_step_count <= 256
            and self.state_sha256 is not None
            and self.physical_objective <= OBJECTIVE_MAXIMUM
            and len(self.raw_equalities) == EQUALITY_SIZE
            and len(self.scaled_equalities) == EQUALITY_SIZE
            and all(
                abs(observed)
                <= abs(expected)
                + EQUALITY_ABSOLUTE_TOLERANCE
                + EQUALITY_RELATIVE_TOLERANCE * abs(expected)
                for observed, expected in zip(
                    self.raw_equalities, reference.raw_equalities, strict=True
                )
            )
            and all(
                math.isclose(
                    scaled,
                    inverse_scale * raw,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
                for scaled, inverse_scale, raw in zip(
                    self.scaled_equalities,
                    reference.constraint_inverse_scale,
                    self.raw_equalities,
                    strict=True,
                )
            )
            and math.isclose(
                self.scaled_feasibility_inf,
                max(abs(value) for value in self.scaled_equalities),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            and self.scaled_feasibility_inf <= SCALED_FEASIBILITY_MAXIMUM
            and self.state_dtype == "float64"
            and self.equality_dtype == "float64"
            and self.correction_certified
        )


@dataclass(frozen=True, slots=True)
class EndpointAuditEvidence:
    """Immutable canonical bytes for the independently parsed raw endpoint audit."""

    canonical_payload: bytes

    @classmethod
    def from_payload(cls, value: JsonValue) -> EndpointAuditEvidence:
        payload = _mapping(value, "endpoint audit")
        endpoint_audit_from_payload(payload)
        return cls(canonical_json_bytes(payload))

    def to_payload(self) -> dict[str, JsonValue]:
        return _mapping(
            load_canonical_json_bytes(self.canonical_payload), "endpoint audit"
        )

    def validate(self) -> None:
        if canonical_json_bytes(self.to_payload()) != self.canonical_payload:
            raise ValueError("endpoint audit bytes are not canonical")
        endpoint_audit_from_payload(self.to_payload())

    def passes(self, candidate: CandidateEvidence, reference: ReferenceReceipt) -> bool:
        payload = self.to_payload()
        audit = endpoint_audit_from_payload(payload)
        return bool(
            candidate.state_sha256 is not None
            and audit.audited_state_sha256 == candidate.state_sha256
            and audit.binding.accepted_step_count == candidate.accepted_step_count
            and audit.binding.bootstrap_state_sha256 == reference.bootstrap_state_sha256
            and audit.physics_contract.term_ledger_sha256
            == reference.ledger_identity_sha256
            and audit.native_reference_state_sha256 == reference.physical_state_sha256
            and validate_endpoint_audit_payload(payload)
        )


@dataclass(frozen=True, slots=True)
class ResourceEvidence:
    source_identity_sha256: str
    pre_source_identity: SourceIdentityEvidence
    post_source_identity: SourceIdentityEvidence
    source_manifest: ArtifactRef
    runtime_environment_sha256: str
    runtime_evidence: ArtifactRef
    reference_policy_sha256: str
    backend: str
    device_uuid: str
    jax_enable_x64: bool
    child_pid: int
    child_start_time_ticks: int
    hot_h2d_transfers: int
    hot_d2h_transfers: int
    python_callbacks: int
    final_d2h_transfers: int
    peak_memory_fraction: float

    def passes(
        self, reference: ReferenceReceipt, execution_status: ExecutionStatus
    ) -> bool:
        final_transfer_valid = (
            self.final_d2h_transfers in {0, 1}
            if execution_status is ExecutionStatus.TIMEOUT
            else self.final_d2h_transfers == 1
        )
        return bool(
            self.pre_source_identity == self.post_source_identity
            and self.source_identity_sha256
            == _canonical_digest(self.pre_source_identity.to_payload())
            and self.source_manifest.sha256
            == self.pre_source_identity.source_manifest_sha256
            and self.source_manifest.size_bytes
            == self.pre_source_identity.source_manifest_size_bytes
            and self.reference_policy_sha256 == reference.reference_policy_sha256
            and self.backend == "gpu"
            and self.device_uuid == GPU_UUID
            and self.jax_enable_x64
            and self.child_pid > 0
            and self.child_start_time_ticks > 0
            and self.hot_h2d_transfers == 0
            and self.hot_d2h_transfers == 0
            and self.python_callbacks == 0
            and final_transfer_valid
            and 0.0 <= self.peak_memory_fraction < MAXIMUM_MEMORY_FRACTION
        )


@dataclass(frozen=True, slots=True)
class KktTelemetry:
    status: KktTelemetryStatus
    raw_stationarity_inf: float | None
    scaled_stationarity_inf: float | None

    def validate(self) -> None:
        if self.status is KktTelemetryStatus.AVAILABLE:
            if (
                self.raw_stationarity_inf is None
                or self.scaled_stationarity_inf is None
                or not math.isfinite(self.raw_stationarity_inf)
                or not math.isfinite(self.scaled_stationarity_inf)
            ):
                raise ValueError("available KKT telemetry requires finite values")
        elif (
            self.raw_stationarity_inf is not None
            or self.scaled_stationarity_inf is not None
        ):
            raise ValueError("unavailable/nonfinite KKT status must use null values")


@dataclass(frozen=True, slots=True)
class SampleReceipt:
    sample: SampleName
    producer_evidence: ArtifactRef
    execution_status: ExecutionStatus
    timing: TimingEvidence
    candidate: CandidateEvidence
    endpoint_audit: EndpointAuditEvidence | None
    resources: ResourceEvidence
    kkt_telemetry: KktTelemetry
    failure_reasons: tuple[str, ...]

    def validate(self) -> None:
        _validate_artifact_ref(self.producer_evidence, "producer evidence")
        self.resources.pre_source_identity.validate()
        self.resources.post_source_identity.validate()
        _validate_artifact_ref(self.resources.source_manifest, "source manifest")
        _validate_artifact_ref(self.resources.runtime_evidence, "runtime evidence")
        for value, context in (
            (self.resources.source_identity_sha256, "source identity"),
            (self.resources.runtime_environment_sha256, "runtime environment"),
            (self.resources.reference_policy_sha256, "resource reference policy"),
        ):
            _require_sha256(value, context)
        if self.candidate.state_sha256 is not None:
            _require_sha256(self.candidate.state_sha256, "candidate state")
        if self.endpoint_audit is not None:
            self.endpoint_audit.validate()
        self.kkt_telemetry.validate()
        if not 0 <= self.candidate.accepted_step_count <= 256:
            raise ValueError("accepted-step count is outside the bounded solve")
        if (
            self.execution_status
            in {
                ExecutionStatus.TIMEOUT,
                ExecutionStatus.COMPILE_FAILURE,
                ExecutionStatus.CRASH,
                ExecutionStatus.INCOMPLETE,
            }
            and not self.failure_reasons
        ):
            raise ValueError("failed execution requires explicit failure reasons")
        if self.execution_status is ExecutionStatus.COMPLETED and self.failure_reasons:
            raise ValueError("completed execution cannot contain failure reasons")
        if self.candidate.reached:
            if (
                self.candidate.first_hit_attempt is None
                or self.candidate.first_hit_accepted_step is None
                or self.candidate.state_sha256 is None
                or self.candidate.physical_objective is None
                or self.candidate.scaled_feasibility_inf is None
            ):
                raise ValueError("reached candidate is missing first-hit evidence")
        elif (
            self.candidate.first_hit_attempt is not None
            or self.candidate.first_hit_accepted_step is not None
            or self.candidate.state_sha256 is not None
            or self.candidate.physical_objective is not None
            or self.candidate.raw_equalities
            or self.candidate.scaled_equalities
            or self.candidate.scaled_feasibility_inf is not None
            or self.candidate.state_dtype is not None
            or self.candidate.equality_dtype is not None
            or self.candidate.correction_certified
            or self.endpoint_audit is not None
        ):
            raise ValueError("unreached candidate cannot contain endpoint evidence")
        if self.candidate.reached and self.endpoint_audit is not None:
            if self.timing.audit_started_ns is None:
                raise ValueError(
                    "completed endpoint audit requires post-timing evidence"
                )
        elif self.timing.audit_started_ns is not None:
            raise ValueError("post-timing audit evidence requires a completed audit")
        if not self.timing.valid(self.candidate.reached):
            raise ValueError("sample timing boundary is invalid")

    def quality(self, reference: ReferenceReceipt) -> SampleQuality:
        self.validate()
        if self.execution_status is ExecutionStatus.TIMEOUT:
            return SampleQuality.QUALITY_NOT_REACHED
        if self.execution_status in {
            ExecutionStatus.COMPILE_FAILURE,
            ExecutionStatus.CRASH,
            ExecutionStatus.INCOMPLETE,
        }:
            return SampleQuality.NOT_PRODUCED
        if not self.candidate.passes(reference):
            return SampleQuality.QUALITY_NOT_REACHED
        if self.endpoint_audit is None:
            return SampleQuality.NOT_PRODUCED
        if not self.endpoint_audit.passes(self.candidate, reference):
            return SampleQuality.DEVICE_QUALITY_CANDIDATE
        return SampleQuality.NATIVE_EQUIVALENT_QUALITY

    def provenance_and_resources_pass(self, reference: ReferenceReceipt) -> bool:
        return self.resources.passes(reference, self.execution_status)


@dataclass(frozen=True, slots=True)
class CampaignReceipt:
    reference: ReferenceReceipt
    samples: tuple[SampleReceipt, ...]

    def validate(self) -> None:
        self.reference.validate()
        if not self.reference.produced and self.samples:
            raise ValueError("unproduced reference forbids campaign samples")
        for sample in self.samples:
            sample.validate()
        if len(
            {
                (
                    sample.producer_evidence.relative_path,
                    sample.producer_evidence.sha256,
                )
                for sample in self.samples
            }
        ) != len(self.samples):
            raise ValueError("campaign samples may not replace or reuse evidence")
        if len(
            {sample.resources.runtime_evidence.sha256 for sample in self.samples}
        ) != len(self.samples):
            raise ValueError("campaign samples may not reuse runtime evidence")
        if len(
            {
                (sample.resources.child_pid, sample.resources.child_start_time_ticks)
                for sample in self.samples
            }
        ) != len(self.samples):
            raise ValueError("campaign samples must use isolated child processes")
        order = (
            SampleName.COLD,
            SampleName.WARM_1,
            SampleName.WARM_2,
            SampleName.WARM_3,
        )
        if (
            tuple(sample.sample for sample in self.samples)
            != order[: len(self.samples)]
        ):
            raise ValueError("campaign samples are missing, reordered, or replaced")
        if not self.samples:
            return
        cold = self.samples[0]
        if len(self.samples) > 1 and (
            cold.quality(self.reference) is not SampleQuality.NATIVE_EQUIVALENT_QUALITY
            or not cold.provenance_and_resources_pass(self.reference)
        ):
            raise ValueError("warm samples require a fully audited valid cold sample")
        if len(self.samples) not in (1, 4):
            raise ValueError(
                "campaign requires a cold-only terminal or cold plus three warms"
            )
        source_identity = cold.resources.source_identity_sha256
        runtime_environment = cold.resources.runtime_environment_sha256
        for sample in self.samples[1:]:
            if (
                sample.resources.source_identity_sha256 != source_identity
                or sample.resources.runtime_environment_sha256 != runtime_environment
            ):
                raise ValueError("campaign samples mix source or runtime identities")

    def disposition(self) -> EngineeringDisposition:
        self.validate()
        if not self.reference.produced:
            return EngineeringDisposition.REFERENCE_NOT_PRODUCED
        if not self.samples:
            return EngineeringDisposition.NOT_PRODUCED
        qualities = tuple(sample.quality(self.reference) for sample in self.samples)
        if any(quality is SampleQuality.NOT_PRODUCED for quality in qualities):
            return EngineeringDisposition.NOT_PRODUCED
        if any(
            not sample.provenance_and_resources_pass(self.reference)
            for sample in self.samples
        ):
            return EngineeringDisposition.NOT_PRODUCED
        if any(
            quality is SampleQuality.DEVICE_QUALITY_CANDIDATE for quality in qualities
        ):
            return EngineeringDisposition.ENDPOINT_AUDIT_FAILED_NONPROMOTING
        if any(quality is SampleQuality.QUALITY_NOT_REACHED for quality in qualities):
            return EngineeringDisposition.QUALITY_NOT_REACHED_BOUNDED_NEGATIVE
        if len(self.samples) != 4:
            return EngineeringDisposition.NOT_PRODUCED
        warm_samples = self.samples[1:]
        if all(
            sample.timing.synchronized_solve_seconds < WARM_SOLVE_MAXIMUM_SECONDS
            for sample in warm_samples
        ):
            return EngineeringDisposition.ENGINEERING_SPEED_GOAL_ACHIEVED
        return EngineeringDisposition.ENGINEERING_SPEED_GOAL_NOT_ACHIEVED


def _source_identity_from_payload(
    value: JsonValue, context: str
) -> SourceIdentityEvidence:
    payload = _mapping(value, context)
    _exact_keys(
        payload,
        frozenset(
            {
                "git_head",
                "tracked_diff_sha256",
                "untracked_bytes_manifest_sha256",
                "source_manifest_sha256",
                "source_manifest_size_bytes",
            }
        ),
        context,
    )
    evidence = SourceIdentityEvidence(
        git_head=_git_head(payload["git_head"]),
        tracked_diff_sha256=_sha256(
            payload["tracked_diff_sha256"], f"{context}.tracked_diff_sha256"
        ),
        untracked_bytes_manifest_sha256=_sha256(
            payload["untracked_bytes_manifest_sha256"],
            f"{context}.untracked_bytes_manifest_sha256",
        ),
        source_manifest_sha256=_sha256(
            payload["source_manifest_sha256"], f"{context}.source_manifest_sha256"
        ),
        source_manifest_size_bytes=_integer(
            payload["source_manifest_size_bytes"],
            f"{context}.source_manifest_size_bytes",
        ),
    )
    evidence.validate()
    return evidence


def _native_branch_from_payload(value: JsonValue) -> NativeBranchEvidence:
    payload = _mapping(value, "native branch evidence")
    _exact_keys(
        payload,
        frozenset(
            {
                "path_256_successful_knots",
                "path_512_successful_knots",
                "common_knot_count",
                "maximum_common_knot_difference",
                "maximum_scaled_feasibility_256",
                "maximum_scaled_feasibility_512",
                "path_256_terminal_state_sha256",
                "path_512_terminal_state_sha256",
                "first_failing_index",
            }
        ),
        "native branch evidence",
    )
    failure = payload["first_failing_index"]
    return NativeBranchEvidence(
        path_256_successful_knots=_integer(
            payload["path_256_successful_knots"], "256 knot count"
        ),
        path_512_successful_knots=_integer(
            payload["path_512_successful_knots"], "512 knot count"
        ),
        common_knot_count=_integer(payload["common_knot_count"], "common knot count"),
        maximum_common_knot_difference=_number(
            payload["maximum_common_knot_difference"], "common-knot difference"
        ),
        maximum_scaled_feasibility_256=_number(
            payload["maximum_scaled_feasibility_256"], "256 feasibility"
        ),
        maximum_scaled_feasibility_512=_number(
            payload["maximum_scaled_feasibility_512"], "512 feasibility"
        ),
        path_256_terminal_state_sha256=_sha256(
            payload["path_256_terminal_state_sha256"], "256 terminal state"
        ),
        path_512_terminal_state_sha256=_sha256(
            payload["path_512_terminal_state_sha256"], "512 terminal state"
        ),
        first_failing_index=None
        if failure is None
        else _integer(failure, "native first failing index"),
    )


def _reference_from_payload(value: JsonValue) -> ReferenceReceipt:
    payload = _mapping(value, "reference receipt")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "produced",
                "reference_evidence",
                "reference_policy_sha256",
                "native_receipt_sha256",
                "native_trajectory_sha256",
                "bootstrap_state_sha256",
                "physical_state_sha256",
                "raw_equalities",
                "constraint_inverse_scale",
                "ledger_identity_sha256",
                "native_branch_evidence_sha256",
                "native_branch_evidence",
                "failure_reasons",
            }
        ),
        "reference receipt",
    )
    if (
        payload["schema_version"] != REFERENCE_SCHEMA_VERSION
        or payload["route"] != ROUTE
    ):
        raise ValueError("reference schema or route identity differs")
    if payload["plan_sha256"] != PLAN_SHA256:
        raise ValueError("reference plan identity differs")
    raw_equalities_value = payload["raw_equalities"]
    scale_value = payload["constraint_inverse_scale"]
    produced = _boolean(payload["produced"], "reference.produced")
    native_branch_value = payload["native_branch_evidence"]
    raw_equalities = (
        _float_vector(raw_equalities_value, EQUALITY_SIZE, "reference.raw_equalities")
        if produced
        else _float_vector(raw_equalities_value, 0, "reference.raw_equalities")
    )
    scale = (
        _float_vector(scale_value, EQUALITY_SIZE, "reference.constraint_inverse_scale")
        if produced
        else _float_vector(scale_value, 0, "reference.constraint_inverse_scale")
    )
    receipt = ReferenceReceipt(
        produced=produced,
        reference_evidence=_artifact_ref_from_payload(
            payload["reference_evidence"], "reference evidence"
        ),
        reference_policy_sha256=_optional_sha256(
            payload["reference_policy_sha256"], "reference policy"
        ),
        native_receipt_sha256=_sha256(
            payload["native_receipt_sha256"], "native receipt"
        ),
        native_trajectory_sha256=_sha256(
            payload["native_trajectory_sha256"], "native trajectory"
        ),
        bootstrap_state_sha256=_optional_sha256(
            payload["bootstrap_state_sha256"], "bootstrap state"
        ),
        physical_state_sha256=_optional_sha256(
            payload["physical_state_sha256"], "reference state"
        ),
        raw_equalities=raw_equalities,
        constraint_inverse_scale=scale,
        ledger_identity_sha256=_optional_sha256(
            payload["ledger_identity_sha256"], "ledger identity"
        ),
        native_branch_evidence_sha256=_optional_sha256(
            payload["native_branch_evidence_sha256"], "native branch evidence"
        ),
        native_branch_evidence=(
            _native_branch_from_payload(native_branch_value)
            if produced
            else (
                None
                if native_branch_value is None
                else _native_branch_from_payload(native_branch_value)
            )
        ),
        failure_reasons=_strings(
            payload["failure_reasons"], "reference.failure_reasons"
        ),
    )
    receipt.validate()
    return receipt


def _timing_from_payload(value: JsonValue) -> TimingEvidence:
    payload = _mapping(value, "timing")
    keys = frozenset(
        {
            "compile_completed_ns",
            "device_state_ready_ns",
            "timer_started_ns",
            "first_hit_synchronized_ns",
            "timer_stopped_ns",
            "audit_started_ns",
            "final_transfer_ns",
            "serialized_ns",
            "synchronized_solve_seconds",
            "endpoint_audit_seconds",
            "total_process_seconds",
        }
    )
    _exact_keys(payload, keys, "timing")
    optional_integer = lambda item, context: (
        None if item is None else _integer(item, context)
    )
    return TimingEvidence(
        compile_completed_ns=_integer(payload["compile_completed_ns"], "compile time"),
        device_state_ready_ns=_integer(
            payload["device_state_ready_ns"], "device-ready time"
        ),
        timer_started_ns=_integer(payload["timer_started_ns"], "timer start"),
        first_hit_synchronized_ns=optional_integer(
            payload["first_hit_synchronized_ns"], "first-hit time"
        ),
        timer_stopped_ns=_integer(payload["timer_stopped_ns"], "timer stop"),
        audit_started_ns=optional_integer(payload["audit_started_ns"], "audit start"),
        final_transfer_ns=optional_integer(
            payload["final_transfer_ns"], "final transfer"
        ),
        serialized_ns=_integer(payload["serialized_ns"], "serialization time"),
        synchronized_solve_seconds=_number(
            payload["synchronized_solve_seconds"], "solve seconds"
        ),
        endpoint_audit_seconds=_optional_number(
            payload["endpoint_audit_seconds"], "audit seconds"
        ),
        total_process_seconds=_number(
            payload["total_process_seconds"], "process seconds"
        ),
    )


def _candidate_from_payload(value: JsonValue) -> CandidateEvidence:
    payload = _mapping(value, "candidate")
    _exact_keys(
        payload,
        frozenset(
            {
                "reached",
                "first_hit_attempt",
                "first_hit_accepted_step",
                "accepted_step_count",
                "state_sha256",
                "physical_objective",
                "raw_equalities",
                "scaled_equalities",
                "scaled_feasibility_inf",
                "state_dtype",
                "equality_dtype",
                "correction_certified",
            }
        ),
        "candidate",
    )
    reached = _boolean(payload["reached"], "candidate.reached")
    optional_integer = lambda item, context: (
        None if item is None else _integer(item, context)
    )
    return CandidateEvidence(
        reached=reached,
        first_hit_attempt=optional_integer(
            payload["first_hit_attempt"], "first-hit attempt"
        ),
        first_hit_accepted_step=optional_integer(
            payload["first_hit_accepted_step"], "first-hit accepted step"
        ),
        accepted_step_count=_integer(
            payload["accepted_step_count"], "accepted-step count"
        ),
        state_sha256=_optional_sha256(payload["state_sha256"], "candidate state"),
        physical_objective=_optional_number(
            payload["physical_objective"], "candidate objective"
        ),
        raw_equalities=_float_vector(
            payload["raw_equalities"],
            EQUALITY_SIZE if reached else 0,
            "candidate.raw_equalities",
        ),
        scaled_equalities=_float_vector(
            payload["scaled_equalities"],
            EQUALITY_SIZE if reached else 0,
            "candidate.scaled_equalities",
        ),
        scaled_feasibility_inf=_optional_number(
            payload["scaled_feasibility_inf"], "scaled feasibility"
        ),
        state_dtype=None
        if payload["state_dtype"] is None
        else _string(payload["state_dtype"], "state dtype"),
        equality_dtype=None
        if payload["equality_dtype"] is None
        else _string(payload["equality_dtype"], "equality dtype"),
        correction_certified=_boolean(
            payload["correction_certified"], "correction certificate"
        ),
    )


def _finite_array_from_payload(value: JsonValue, context: str) -> FiniteArrayEvidence:
    payload = _mapping(value, context)
    _exact_keys(payload, frozenset({"dtype", "size", "nonfinite_count"}), context)
    return FiniteArrayEvidence(
        dtype=_string(payload["dtype"], f"{context}.dtype"),
        size=_integer(payload["size"], f"{context}.size"),
        nonfinite_count=_integer(
            payload["nonfinite_count"], f"{context}.nonfinite_count"
        ),
    )


def _accepted_ledger_from_payload(value: JsonValue) -> AcceptedLedgerEvidence:
    payload = _mapping(value, "accepted ledger")
    _exact_keys(
        payload,
        frozenset(
            {
                "row_capacity",
                "valid_row_count",
                "accepted_step_count",
                "bootstrap_state_sha256",
                "latched_state_sha256",
                "rejected_rows_retained",
            }
        ),
        "accepted ledger",
    )
    return AcceptedLedgerEvidence(
        row_capacity=_integer(payload["row_capacity"], "ledger row capacity"),
        valid_row_count=_integer(payload["valid_row_count"], "ledger valid rows"),
        accepted_step_count=_integer(
            payload["accepted_step_count"], "ledger accepted steps"
        ),
        bootstrap_state_sha256=_sha256(
            payload["bootstrap_state_sha256"], "ledger bootstrap state"
        ),
        latched_state_sha256=_sha256(
            payload["latched_state_sha256"], "ledger latched state"
        ),
        rejected_rows_retained=_integer(
            payload["rejected_rows_retained"], "ledger rejected rows"
        ),
    )


def _branch_replay_from_payload(value: JsonValue) -> BranchReplayEvidence:
    payload = _mapping(value, "branch replay")
    _exact_keys(
        payload,
        frozenset(
            {
                "replayed_row_count",
                "row0_state_sha256",
                "successful_direct_rows",
                "successful_midpoint_refined_rows",
                "maximum_direct_refined_difference",
                "maximum_gpu_native_difference",
                "maximum_scaled_feasibility",
                "first_failing_row",
            }
        ),
        "branch replay",
    )
    failure = payload["first_failing_row"]
    return BranchReplayEvidence(
        replayed_row_count=_integer(payload["replayed_row_count"], "replayed rows"),
        row0_state_sha256=_sha256(payload["row0_state_sha256"], "row0 state"),
        successful_direct_rows=_integer(
            payload["successful_direct_rows"], "direct rows"
        ),
        successful_midpoint_refined_rows=_integer(
            payload["successful_midpoint_refined_rows"], "midpoint rows"
        ),
        maximum_direct_refined_difference=_number(
            payload["maximum_direct_refined_difference"], "direct/refined difference"
        ),
        maximum_gpu_native_difference=_number(
            payload["maximum_gpu_native_difference"], "GPU/native difference"
        ),
        maximum_scaled_feasibility=_number(
            payload["maximum_scaled_feasibility"], "replay feasibility"
        ),
        first_failing_row=None
        if failure is None
        else _integer(failure, "first failing row"),
    )


def _cross_evaluator_from_payload(value: JsonValue) -> CrossEvaluatorEvidence:
    payload = _mapping(value, "cross evaluator")
    keys = frozenset(
        {
            "native_on_gpu_objective",
            "jax_on_gpu_objective",
            "native_on_native_objective",
            "jax_on_native_objective",
            "gpu_raw_term_maximum_difference",
            "gpu_raw_term_maximum_magnitude",
            "native_raw_term_maximum_difference",
            "native_raw_term_maximum_magnitude",
        }
    )
    _exact_keys(payload, keys, "cross evaluator")
    return CrossEvaluatorEvidence(
        native_on_gpu_objective=_number(
            payload["native_on_gpu_objective"], "cross evaluator.native GPU objective"
        ),
        jax_on_gpu_objective=_number(
            payload["jax_on_gpu_objective"], "cross evaluator.JAX GPU objective"
        ),
        native_on_native_objective=_number(
            payload["native_on_native_objective"],
            "cross evaluator.native native objective",
        ),
        jax_on_native_objective=_number(
            payload["jax_on_native_objective"],
            "cross evaluator.JAX native objective",
        ),
        gpu_raw_term_maximum_difference=_number(
            payload["gpu_raw_term_maximum_difference"],
            "cross evaluator GPU term difference",
        ),
        gpu_raw_term_maximum_magnitude=_number(
            payload["gpu_raw_term_maximum_magnitude"],
            "cross evaluator GPU term magnitude",
        ),
        native_raw_term_maximum_difference=_number(
            payload["native_raw_term_maximum_difference"],
            "cross evaluator native term difference",
        ),
        native_raw_term_maximum_magnitude=_number(
            payload["native_raw_term_maximum_magnitude"],
            "cross evaluator native term magnitude",
        ),
    )


def _audit_from_payload(value: JsonValue) -> EndpointAuditEvidence:
    return EndpointAuditEvidence.from_payload(value)


def _resources_from_payload(value: JsonValue) -> ResourceEvidence:
    payload = _mapping(value, "resources")
    _exact_keys(
        payload,
        frozenset(
            {
                "source_identity_sha256",
                "pre_source_identity",
                "post_source_identity",
                "source_manifest",
                "runtime_environment_sha256",
                "runtime_evidence",
                "reference_policy_sha256",
                "backend",
                "device_uuid",
                "jax_enable_x64",
                "child_pid",
                "child_start_time_ticks",
                "hot_h2d_transfers",
                "hot_d2h_transfers",
                "python_callbacks",
                "final_d2h_transfers",
                "peak_memory_fraction",
            }
        ),
        "resources",
    )
    return ResourceEvidence(
        source_identity_sha256=_sha256(
            payload["source_identity_sha256"], "source identity"
        ),
        pre_source_identity=_source_identity_from_payload(
            payload["pre_source_identity"], "pre source identity"
        ),
        post_source_identity=_source_identity_from_payload(
            payload["post_source_identity"], "post source identity"
        ),
        source_manifest=_artifact_ref_from_payload(
            payload["source_manifest"], "source manifest"
        ),
        runtime_environment_sha256=_sha256(
            payload["runtime_environment_sha256"], "runtime environment"
        ),
        runtime_evidence=_artifact_ref_from_payload(
            payload["runtime_evidence"], "runtime evidence"
        ),
        reference_policy_sha256=_sha256(
            payload["reference_policy_sha256"], "resource reference policy"
        ),
        backend=_string(payload["backend"], "backend"),
        device_uuid=_string(payload["device_uuid"], "device UUID"),
        jax_enable_x64=_boolean(payload["jax_enable_x64"], "JAX x64"),
        child_pid=_integer(payload["child_pid"], "child PID"),
        child_start_time_ticks=_integer(
            payload["child_start_time_ticks"], "child start ticks"
        ),
        hot_h2d_transfers=_integer(payload["hot_h2d_transfers"], "hot H2D count"),
        hot_d2h_transfers=_integer(payload["hot_d2h_transfers"], "hot D2H count"),
        python_callbacks=_integer(payload["python_callbacks"], "callback count"),
        final_d2h_transfers=_integer(payload["final_d2h_transfers"], "final D2H count"),
        peak_memory_fraction=_number(
            payload["peak_memory_fraction"], "memory fraction"
        ),
    )


def _kkt_from_payload(value: JsonValue) -> KktTelemetry:
    payload = _mapping(value, "KKT telemetry")
    _exact_keys(
        payload,
        frozenset({"status", "raw_stationarity_inf", "scaled_stationarity_inf"}),
        "KKT telemetry",
    )
    telemetry = KktTelemetry(
        status=KktTelemetryStatus(_string(payload["status"], "KKT status")),
        raw_stationarity_inf=_optional_number(
            payload["raw_stationarity_inf"], "raw KKT"
        ),
        scaled_stationarity_inf=_optional_number(
            payload["scaled_stationarity_inf"], "scaled KKT"
        ),
    )
    telemetry.validate()
    return telemetry


def _sample_from_payload(value: JsonValue) -> SampleReceipt:
    payload = _mapping(value, "sample receipt")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "sample",
                "producer_evidence",
                "execution_status",
                "timing",
                "candidate",
                "endpoint_audit",
                "resources",
                "kkt_telemetry",
                "failure_reasons",
            }
        ),
        "sample receipt",
    )
    if payload["schema_version"] != SAMPLE_SCHEMA_VERSION or payload["route"] != ROUTE:
        raise ValueError("sample schema or route identity differs")
    if payload["plan_sha256"] != PLAN_SHA256:
        raise ValueError("sample plan identity differs")
    receipt = SampleReceipt(
        sample=SampleName(_string(payload["sample"], "sample name")),
        producer_evidence=_artifact_ref_from_payload(
            payload["producer_evidence"], "producer evidence"
        ),
        execution_status=ExecutionStatus(
            _string(payload["execution_status"], "execution status")
        ),
        timing=_timing_from_payload(payload["timing"]),
        candidate=_candidate_from_payload(payload["candidate"]),
        endpoint_audit=(
            None
            if payload["endpoint_audit"] is None
            else _audit_from_payload(payload["endpoint_audit"])
        ),
        resources=_resources_from_payload(payload["resources"]),
        kkt_telemetry=_kkt_from_payload(payload["kkt_telemetry"]),
        failure_reasons=_strings(payload["failure_reasons"], "sample.failure_reasons"),
    )
    receipt.validate()
    return receipt


def campaign_receipt_from_payload(value: JsonValue) -> CampaignReceipt:
    """Parse one exact-schema campaign mapping and recompute its disposition."""

    payload = _mapping(value, "campaign receipt")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "reference",
                "samples",
                "terminal_disposition",
            }
        ),
        "campaign receipt",
    )
    if (
        payload["schema_version"] != CAMPAIGN_SCHEMA_VERSION
        or payload["route"] != ROUTE
    ):
        raise ValueError("campaign schema or route identity differs")
    if payload["plan_sha256"] != PLAN_SHA256:
        raise ValueError("campaign plan identity differs")
    raw_samples = payload["samples"]
    if not isinstance(raw_samples, list):
        raise TypeError("campaign samples must be an array")
    receipt = CampaignReceipt(
        reference=_reference_from_payload(payload["reference"]),
        samples=tuple(_sample_from_payload(item) for item in raw_samples),
    )
    derived = receipt.disposition()
    claimed = EngineeringDisposition(
        _string(payload["terminal_disposition"], "terminal disposition")
    )
    if claimed is not derived:
        raise ValueError("campaign terminal disposition differs from raw evidence")
    return receipt


def load_campaign_receipt(path: Path) -> CampaignReceipt:
    """Parse one receipt file; use the artifact loader for promotion authority."""

    return campaign_receipt_from_payload(load_canonical_json_bytes(path.read_bytes()))


def _validate_campaign_artifact_manifest(root: Path) -> frozenset[str]:
    if root.is_symlink():
        raise ValueError("campaign root must not be a symlink")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("campaign root must be a directory")
    for path in (resolved, *resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError("campaign artifact tree contains a symlink")
        if not path.is_file() and not path.is_dir():
            raise ValueError("campaign artifact tree contains a non-regular path")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError("campaign artifact tree is not sealed read-only")
    manifest_path = resolved / CAMPAIGN_ARTIFACT_MANIFEST_FILENAME
    manifest = _mapping(
        load_canonical_json_bytes(manifest_path.read_bytes()),
        "campaign artifact manifest",
    )
    _exact_keys(
        manifest,
        frozenset({"entries", "schema_version"}),
        "campaign artifact manifest",
    )
    if manifest["schema_version"] != CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("campaign artifact manifest schema differs")
    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("campaign artifact manifest entries must be an array")
    declared: list[str] = []
    for index, value in enumerate(raw_entries):
        context = f"campaign artifact manifest entries[{index}]"
        entry = _mapping(value, context)
        _exact_keys(
            entry,
            frozenset({"relative_path", "sha256", "size_bytes"}),
            context,
        )
        relative_path = _string(entry["relative_path"], f"{context}.relative_path")
        relative = Path(relative_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_path
            or relative_path == CAMPAIGN_ARTIFACT_MANIFEST_FILENAME
        ):
            raise ValueError(f"{context} path is not canonical")
        path = resolved.joinpath(relative)
        payload = path.read_bytes()
        if len(payload) != _integer(
            entry["size_bytes"], f"{context}.size_bytes"
        ) or hashlib.sha256(payload).hexdigest() != _sha256(
            entry["sha256"], f"{context}.sha256"
        ):
            raise ValueError(f"{context} bytes differ")
        declared.append(relative_path)
    if declared != sorted(declared) or len(declared) != len(set(declared)):
        raise ValueError("campaign artifact manifest paths are not sorted and unique")
    observed = frozenset(
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if observed != frozenset(declared):
        raise ValueError("campaign artifact contains missing or unmanifested files")
    if "campaign.json" not in observed:
        raise ValueError("campaign artifact omits campaign.json")
    return observed


def _candidate_payload(candidate: CandidateEvidence) -> dict[str, JsonValue]:
    return {
        "accepted_step_count": candidate.accepted_step_count,
        "correction_certified": candidate.correction_certified,
        "equality_dtype": candidate.equality_dtype,
        "first_hit_accepted_step": candidate.first_hit_accepted_step,
        "first_hit_attempt": candidate.first_hit_attempt,
        "physical_objective": candidate.physical_objective,
        "raw_equalities": list(candidate.raw_equalities),
        "reached": candidate.reached,
        "scaled_equalities": list(candidate.scaled_equalities),
        "scaled_feasibility_inf": candidate.scaled_feasibility_inf,
        "state_dtype": candidate.state_dtype,
        "state_sha256": candidate.state_sha256,
    }


def _validate_producer_binding(
    root: Path,
    sample: SampleReceipt,
    snapshot_root: Path,
) -> dict[str, JsonValue]:
    producer_path = sample.producer_evidence.resolve_and_validate(root)
    producer = _mapping(
        load_canonical_json_bytes(producer_path.read_bytes()), "sample producer"
    )
    if (
        producer.get("route") != ROUTE
        or producer.get("plan_sha256") != PLAN_SHA256
        or producer.get("sample") != sample.sample.value
        or producer.get("execution_status") != sample.execution_status.value
    ):
        raise ValueError("sample receipt differs from producer identity/status")
    runtime_value = producer.get("runtime_evidence")
    if runtime_value != _artifact_ref_payload(sample.resources.runtime_evidence):
        raise ValueError("sample runtime reference differs from producer")
    runtime_path = sample.resources.runtime_evidence.resolve_and_validate(root)
    runtime = validate_runtime_evidence(
        runtime_path,
        snapshot_root=snapshot_root,
        campaign_root=root,
    )
    runtime_identity = runtime.observation.runtime_identity
    if (
        runtime_identity.effective_environment_sha256
        != sample.resources.runtime_environment_sha256
        or runtime_identity.backend != sample.resources.backend
        or runtime_identity.device_uuid != sample.resources.device_uuid
    ):
        raise ValueError("sample resources differ from runtime evidence")
    if sample.candidate.reached:
        if producer.get("candidate") != _candidate_payload(sample.candidate):
            raise ValueError("sample candidate differs from producer raw evidence")
        if (
            sample.endpoint_audit is None
            or producer.get("endpoint_audit") != sample.endpoint_audit.to_payload()
        ):
            raise ValueError("sample endpoint audit differs from producer raw evidence")
    elif (
        producer.get("candidate_reached") is not False
        or producer.get("endpoint_audit") is not None
        or (
            producer.get("candidate") is not None
            and producer.get("candidate") != _candidate_payload(sample.candidate)
        )
        or (
            producer.get("candidate") is None
            and sample.candidate.accepted_step_count != 0
        )
    ):
        raise ValueError("unreached sample differs from producer raw evidence")
    timing = _mapping(producer.get("timing"), "producer timing")
    compile_completed_ns = timing.get("compile_completed_ns", 0)
    device_state_ready_ns = timing.get("device_state_ready_ns", compile_completed_ns)
    timer_started_ns = timing.get("timer_started_ns", device_state_ready_ns)
    timer_stopped_ns = timing.get("timer_stopped_ns", 0)
    serialized_ns = timing.get(
        "serialized_ns",
        max(
            _integer(timer_started_ns, "timer start"),
            _integer(timer_stopped_ns, "timer stop"),
        ),
    )
    expected_timing = {
        "audit_started_ns": timing.get("audit_started_ns")
        if sample.candidate.reached
        else None,
        "compile_completed_ns": compile_completed_ns,
        "device_state_ready_ns": device_state_ready_ns,
        "endpoint_audit_seconds": timing.get("endpoint_audit_seconds")
        if sample.candidate.reached
        else None,
        "final_transfer_ns": timing.get("final_transfer_ns")
        if sample.candidate.reached
        else None,
        "serialized_ns": serialized_ns,
        "synchronized_solve_seconds": timing.get("synchronized_solve_seconds", 0.0),
        "timer_started_ns": timer_started_ns,
        "timer_stopped_ns": timer_stopped_ns,
    }
    observed_timing = {
        "audit_started_ns": sample.timing.audit_started_ns,
        "compile_completed_ns": sample.timing.compile_completed_ns,
        "device_state_ready_ns": sample.timing.device_state_ready_ns,
        "endpoint_audit_seconds": sample.timing.endpoint_audit_seconds,
        "final_transfer_ns": sample.timing.final_transfer_ns,
        "serialized_ns": sample.timing.serialized_ns,
        "synchronized_solve_seconds": sample.timing.synchronized_solve_seconds,
        "timer_started_ns": sample.timing.timer_started_ns,
        "timer_stopped_ns": sample.timing.timer_stopped_ns,
    }
    if observed_timing != expected_timing:
        raise ValueError("sample timing differs from producer raw evidence")
    if sample.candidate.reached and (
        sample.timing.first_hit_synchronized_ns != timing.get("timer_stopped_ns")
    ):
        raise ValueError("sample first-hit time differs from producer raw evidence")
    runtime_summary = _mapping(producer.get("runtime"), "producer runtime")
    if (
        runtime_summary.get("backend") != sample.resources.backend
        or runtime_summary.get("device_uuid") != sample.resources.device_uuid
        or runtime_summary.get("jax_enable_x64") is not sample.resources.jax_enable_x64
    ):
        raise ValueError("sample resource summary differs from producer")
    transfer_value = producer.get("transfer_audit")
    if transfer_value is not None:
        transfer = _mapping(transfer_value, "producer transfer audit")
        expected_transfers = {
            "final_d2h_transfers": sample.resources.final_d2h_transfers,
            "hot_d2h_transfers": sample.resources.hot_d2h_transfers,
            "hot_h2d_transfers": sample.resources.hot_h2d_transfers,
            "python_callbacks": sample.resources.python_callbacks,
        }
        if any(transfer.get(key) != value for key, value in expected_transfers.items()):
            raise ValueError("sample transfer counts differ from producer")
    return producer


def _validate_terminal_binding(root: Path, sample: SampleReceipt) -> str:
    relative_path = f"samples/{sample.sample.value}/terminal.json"
    terminal = _mapping(
        load_canonical_json_bytes((root / relative_path).read_bytes()),
        "sample terminal",
    )
    _exact_keys(
        terminal,
        frozenset(
            {
                "child_pid",
                "child_start_time_ticks",
                "failure_reasons",
                "process_seconds",
                "sample",
                "schema_version",
                "terminal_status",
            }
        ),
        "sample terminal",
    )
    terminal_statuses = {
        ExecutionStatus.COMPLETED: frozenset({"COMPLETE"}),
        ExecutionStatus.TIMEOUT: frozenset({"TIMEOUT"}),
        ExecutionStatus.COMPILE_FAILURE: frozenset({"COMPILE_FAILURE"}),
        ExecutionStatus.CRASH: frozenset({"CRASH"}),
        ExecutionStatus.INCOMPLETE: frozenset({"MONITOR_FAILURE", "PROTOCOL_FAILURE"}),
    }
    if (
        terminal.get("schema_version") != "single-stage-neq-gntr1-terminal-v1"
        or terminal.get("sample") != sample.sample.value
        or terminal.get("terminal_status")
        not in terminal_statuses[sample.execution_status]
        or terminal.get("child_pid") != sample.resources.child_pid
        or terminal.get("child_start_time_ticks")
        != sample.resources.child_start_time_ticks
        or terminal.get("process_seconds") != sample.timing.total_process_seconds
        or terminal.get("failure_reasons") != list(sample.failure_reasons)
    ):
        raise ValueError("sample receipt differs from terminal raw evidence")
    return relative_path


def _validate_memory_binding(
    root: Path,
    sample: SampleReceipt,
    observed: frozenset[str],
) -> str | None:
    relative_path = f"samples/{sample.sample.value}/gpu-memory.json"
    if relative_path not in observed:
        if sample.resources.peak_memory_fraction != 1.0:
            raise ValueError("sample memory receipt lacks raw monitor evidence")
        return None
    memory = _mapping(
        load_canonical_json_bytes((root / relative_path).read_bytes()),
        "sample GPU memory",
    )
    if (
        memory.get("schema_version") != "single-stage-neq-gntr1-memory-v1"
        or memory.get("child_pid") != sample.resources.child_pid
        or memory.get("child_start_time_ticks")
        != sample.resources.child_start_time_ticks
        or memory.get("device_uuid") != sample.resources.device_uuid
        or memory.get("peak_memory_fraction") != sample.resources.peak_memory_fraction
    ):
        raise ValueError("sample receipt differs from GPU-memory raw evidence")
    return relative_path


def load_and_validate_campaign_artifact(campaign_root: Path) -> CampaignReceipt:
    """Load the sealed tree, resolve every authority, and recompute the verdict."""

    if campaign_root.is_symlink():
        raise ValueError("campaign root must not be a symlink")
    root = campaign_root.resolve(strict=True)
    observed = _validate_campaign_artifact_manifest(root)
    receipt = campaign_receipt_from_payload(
        load_canonical_json_bytes((root / "campaign.json").read_bytes())
    )
    reference_path = receipt.reference.reference_evidence.resolve_and_validate(root)
    reference_root = reference_path.parent
    validate_native_equivalent_reference(reference_root)
    snapshot_root = root / "source-snapshot"
    snapshot = load_snapshot(snapshot_root)
    producers: list[dict[str, JsonValue]] = []
    explicit_paths = {"campaign.json"}
    for sample in receipt.samples:
        if sample.resources.source_manifest.resolve_and_validate(root) != (
            snapshot.manifest_path
        ):
            raise ValueError("sample source manifest differs from sealed snapshot")
        source = sample.resources.pre_source_identity
        if (
            source.git_head != snapshot.worktree.git_head
            or source.tracked_diff_sha256 != snapshot.worktree.tracked_diff_sha256
            or source.untracked_bytes_manifest_sha256
            != snapshot.worktree.untracked_bytes_manifest_sha256
            or source.source_manifest_sha256 != snapshot.manifest_sha256
            or source.source_manifest_size_bytes
            != snapshot.manifest_path.stat().st_size
        ):
            raise ValueError("sample source identity differs from sealed snapshot")
        producers.append(_validate_producer_binding(root, sample, snapshot_root))
        terminal_path = _validate_terminal_binding(root, sample)
        memory_path = _validate_memory_binding(root, sample, observed)
        explicit_paths.update(
            {
                sample.producer_evidence.relative_path,
                sample.resources.runtime_evidence.relative_path,
                terminal_path,
            }
        )
        if memory_path is not None:
            explicit_paths.add(memory_path)
    reference_inputs = _mapping(
        producers[0].get("reference_inputs"), "producer reference inputs"
    )
    bootstrap = np.asarray(reference_inputs.get("bootstrap_state"))
    inverse_scale = np.asarray(reference_inputs.get("constraint_inverse_scale"))
    rebuilt_reference = reference_receipt_from_artifact(
        artifact_root=reference_root,
        reference_evidence=receipt.reference.reference_evidence,
        bootstrap_state=bootstrap,
        constraint_inverse_scale=inverse_scale,
    )
    if rebuilt_reference != receipt.reference:
        raise ValueError("campaign reference differs from sealed raw authority")
    protected_prefixes = ("native-reference/", "source-snapshot/")
    unexplained = {
        path
        for path in observed
        if path not in explicit_paths
        and not any(path.startswith(prefix) for prefix in protected_prefixes)
    }
    if unexplained:
        raise ValueError("campaign artifact contains unexplained evidence files")
    return receipt


def campaign_payload(receipt: CampaignReceipt) -> dict[str, JsonValue]:
    """Serialize a typed receipt with its freshly derived terminal disposition."""

    receipt.validate()

    def reference_payload(reference: ReferenceReceipt) -> dict[str, JsonValue]:
        return {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "route": ROUTE,
            "plan_sha256": PLAN_SHA256,
            "produced": reference.produced,
            "reference_evidence": _artifact_ref_payload(reference.reference_evidence),
            "reference_policy_sha256": reference.reference_policy_sha256,
            "native_receipt_sha256": reference.native_receipt_sha256,
            "native_trajectory_sha256": reference.native_trajectory_sha256,
            "bootstrap_state_sha256": reference.bootstrap_state_sha256,
            "physical_state_sha256": reference.physical_state_sha256,
            "raw_equalities": list(reference.raw_equalities),
            "constraint_inverse_scale": list(reference.constraint_inverse_scale),
            "ledger_identity_sha256": reference.ledger_identity_sha256,
            "native_branch_evidence_sha256": reference.native_branch_evidence_sha256,
            "native_branch_evidence": (
                None
                if reference.native_branch_evidence is None
                else reference.native_branch_evidence.to_payload()
            ),
            "failure_reasons": list(reference.failure_reasons),
        }

    def sample_payload(sample: SampleReceipt) -> dict[str, JsonValue]:
        timing = sample.timing
        candidate = sample.candidate
        audit = sample.endpoint_audit
        resources = sample.resources
        kkt = sample.kkt_telemetry
        return {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "route": ROUTE,
            "plan_sha256": PLAN_SHA256,
            "sample": sample.sample,
            "producer_evidence": _artifact_ref_payload(sample.producer_evidence),
            "execution_status": sample.execution_status,
            "timing": {
                "compile_completed_ns": timing.compile_completed_ns,
                "device_state_ready_ns": timing.device_state_ready_ns,
                "timer_started_ns": timing.timer_started_ns,
                "first_hit_synchronized_ns": timing.first_hit_synchronized_ns,
                "timer_stopped_ns": timing.timer_stopped_ns,
                "audit_started_ns": timing.audit_started_ns,
                "final_transfer_ns": timing.final_transfer_ns,
                "serialized_ns": timing.serialized_ns,
                "synchronized_solve_seconds": timing.synchronized_solve_seconds,
                "endpoint_audit_seconds": timing.endpoint_audit_seconds,
                "total_process_seconds": timing.total_process_seconds,
            },
            "candidate": {
                "reached": candidate.reached,
                "first_hit_attempt": candidate.first_hit_attempt,
                "first_hit_accepted_step": candidate.first_hit_accepted_step,
                "accepted_step_count": candidate.accepted_step_count,
                "state_sha256": candidate.state_sha256,
                "physical_objective": candidate.physical_objective,
                "raw_equalities": list(candidate.raw_equalities),
                "scaled_equalities": list(candidate.scaled_equalities),
                "scaled_feasibility_inf": candidate.scaled_feasibility_inf,
                "state_dtype": candidate.state_dtype,
                "equality_dtype": candidate.equality_dtype,
                "correction_certified": candidate.correction_certified,
            },
            "endpoint_audit": None if audit is None else audit.to_payload(),
            "resources": {
                "post_source_identity": resources.post_source_identity.to_payload(),
                "pre_source_identity": resources.pre_source_identity.to_payload(),
                "source_identity_sha256": resources.source_identity_sha256,
                "source_manifest": _artifact_ref_payload(resources.source_manifest),
                "runtime_environment_sha256": resources.runtime_environment_sha256,
                "runtime_evidence": _artifact_ref_payload(resources.runtime_evidence),
                "reference_policy_sha256": resources.reference_policy_sha256,
                "backend": resources.backend,
                "device_uuid": resources.device_uuid,
                "jax_enable_x64": resources.jax_enable_x64,
                "child_pid": resources.child_pid,
                "child_start_time_ticks": resources.child_start_time_ticks,
                "hot_h2d_transfers": resources.hot_h2d_transfers,
                "hot_d2h_transfers": resources.hot_d2h_transfers,
                "python_callbacks": resources.python_callbacks,
                "final_d2h_transfers": resources.final_d2h_transfers,
                "peak_memory_fraction": resources.peak_memory_fraction,
            },
            "kkt_telemetry": {
                "status": kkt.status,
                "raw_stationarity_inf": kkt.raw_stationarity_inf,
                "scaled_stationarity_inf": kkt.scaled_stationarity_inf,
            },
            "failure_reasons": list(sample.failure_reasons),
        }

    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "route": ROUTE,
        "plan_sha256": PLAN_SHA256,
        "reference": reference_payload(receipt.reference),
        "samples": [sample_payload(sample) for sample in receipt.samples],
        "terminal_disposition": receipt.disposition(),
    }


def campaign_sha256(receipt: CampaignReceipt) -> str:
    """Return the digest of the only canonical encoding of a campaign receipt."""

    return hashlib.sha256(canonical_json_bytes(campaign_payload(receipt))).hexdigest()


__all__ = (
    "CAMPAIGN_ARTIFACT_MANIFEST_FILENAME",
    "CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "CAMPAIGN_SCHEMA_VERSION",
    "GPU_UUID",
    "NATIVE_RECEIPT_SHA256",
    "NATIVE_TRAJECTORY_SHA256",
    "OBJECTIVE_MAXIMUM",
    "PLAN_SHA256",
    "REFERENCE_SCHEMA_VERSION",
    "ROUTE",
    "SAMPLE_SCHEMA_VERSION",
    "WARM_SOLVE_MAXIMUM_SECONDS",
    "CampaignReceipt",
    "CandidateEvidence",
    "EndpointAuditEvidence",
    "EngineeringDisposition",
    "ExecutionStatus",
    "KktTelemetry",
    "KktTelemetryStatus",
    "NativeBranchEvidence",
    "ReferenceReceipt",
    "ResourceEvidence",
    "SampleName",
    "SampleQuality",
    "SampleReceipt",
    "SourceIdentityEvidence",
    "TimingEvidence",
    "campaign_payload",
    "campaign_receipt_from_payload",
    "campaign_sha256",
    "load_and_validate_campaign_artifact",
    "load_campaign_receipt",
    "reference_receipt_from_artifact",
)
