"""Promotion-safe selection of direct or transferred phase attribution.

The default command-buffer-enabled evaluation remains the timing authority.  This
module selects its stable direct attribution when coverage is sufficient, otherwise
it transfers stable disabled-control fractions. Both routes require three matched
attempts per mode, quantitative parity, and exact solve/topology/runtime identity.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Literal

from benchmarks.single_stage_compute_graph_phase0_receipt import (
    PHASE0_GRADIENT_ATOL,
    PHASE0_GRADIENT_RTOL,
    PHASE0_OBJECTIVE_ATOL,
    PHASE0_OBJECTIVE_RTOL,
    canonical_json_bytes,
)

ATTRIBUTION_EVIDENCE_SCHEMA_ID: Final = (
    "single-stage-compute-graph-attribution-evidence-v4"
)
PROFILE_DERIVATION_VERSION: Final = "compute-graph-profile-attribution-v1"
ATTRIBUTION_ATTEMPT_COUNT: Final = 3
COMMAND_BUFFER_DISABLE_FLAG: Final = "--xla_gpu_enable_command_buffer="
MAX_PHASE_TOTAL_VARIATION_DISTANCE: Final = 0.02
MAX_DEFAULT_DEVICE_ACTIVE_SHARE_SPREAD: Final = 0.05
MIN_ATTRIBUTION_COVERAGE: Final = 0.90
NUMERICAL_PARITY_TOLERANCE_SOURCE: Final = "parity_ladder.gpu_runtime.same_state"

Mode = Literal["default_control", "command_buffer_disabled"]
_SHA256_HEX_LENGTH = 64
_LOWER_HEX = frozenset("0123456789abcdef")


class AttributionControlError(RuntimeError):
    """Attribution-control evidence is malformed and cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class AttributionBinding:
    """Production identities that must be identical in all six attempts."""

    candidate_sha256: str
    specimen_sha256: str
    input_bundle_sha256: str
    source_sha256: str
    production_runtime_identity_sha256: str
    lane_id: str
    gpu_uuid: str
    gate_checkpoint_sha256: str
    warm_checkpoint_sha256: str
    warm_p50_ns: float

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "specimen_sha256": self.specimen_sha256,
            "input_bundle_sha256": self.input_bundle_sha256,
            "source_sha256": self.source_sha256,
            "production_runtime_identity_sha256": (
                self.production_runtime_identity_sha256
            ),
            "lane_id": self.lane_id,
            "gpu_uuid": self.gpu_uuid,
            "gate_checkpoint_sha256": self.gate_checkpoint_sha256,
            "warm_checkpoint_sha256": self.warm_checkpoint_sha256,
            "warm_p50_ns": self.warm_p50_ns,
        }


@dataclass(frozen=True, slots=True)
class AttributionAttempt:
    """One immutable, already synchronized changed-state profile observation."""

    mode: Mode
    attempt_index: int
    binding: AttributionBinding
    runtime_identity_sha256: str
    xla_flag_tokens: tuple[str, ...]
    compilation_cache_root: str
    artifact_root: str
    raw_trace_path: str
    raw_trace_sha256: str
    child_observation_path: str
    child_observation_sha256: str
    hlo_anchor_path: str
    hlo_anchor_sha256: str
    profile_derivation_version: str
    objective: float
    gradient: tuple[float, ...]
    solve_certificate: Mapping[str, object]
    module_topology_identity_sha256: str
    evaluation_envelope_ns: int
    device_active_ns: int
    phase_device_ns: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _ProfileObservation:
    evaluation_envelope_ns: int
    device_active_ns: int
    phase_device_ns: tuple[tuple[str, int], ...]


def _sha256(value: str, context: str) -> str:
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in _LOWER_HEX for character in value
    ):
        raise AttributionControlError(f"{context} must be a lowercase SHA-256")
    return value


def _nonempty(value: str, context: str) -> str:
    if not value or value.strip() != value:
        raise AttributionControlError(f"{context} must be a non-empty exact string")
    return value


def _root(value: str, context: str) -> str:
    root = PurePosixPath(_nonempty(value, context))
    if not root.is_absolute() or ".." in root.parts:
        raise AttributionControlError(f"{context} must be a normalized absolute path")
    return str(root)


def _validate_binding(binding: AttributionBinding) -> None:
    for field in (
        "candidate_sha256",
        "specimen_sha256",
        "input_bundle_sha256",
        "source_sha256",
        "production_runtime_identity_sha256",
        "gate_checkpoint_sha256",
        "warm_checkpoint_sha256",
    ):
        _sha256(getattr(binding, field), f"binding.{field}")
    _nonempty(binding.lane_id, "binding.lane_id")
    _nonempty(binding.gpu_uuid, "binding.gpu_uuid")
    if not math.isfinite(binding.warm_p50_ns) or binding.warm_p50_ns <= 0.0:
        raise AttributionControlError("binding.warm_p50_ns must be finite and positive")


def _relative_path(value: str, context: str) -> str:
    path = PurePosixPath(_nonempty(value, context))
    if path.is_absolute() or ".." in path.parts:
        raise AttributionControlError(f"{context} must be a safe relative path")
    return path.as_posix()


def _phase_fractions(attempt: _ProfileObservation) -> dict[str, float]:
    fractions = {
        phase_id: duration_ns / attempt.device_active_ns
        for phase_id, duration_ns in attempt.phase_device_ns
    }
    fractions["__unattributed__"] = 1.0 - sum(fractions.values())
    return fractions


def _phase_envelope_shares(attempt: _ProfileObservation) -> dict[str, float]:
    shares = {
        phase_id: duration_ns / attempt.evaluation_envelope_ns
        for phase_id, duration_ns in attempt.phase_device_ns
    }
    shares["__unattributed__"] = 1.0 - sum(shares.values())
    return shares


def _profile_observation(attempt: AttributionAttempt) -> _ProfileObservation:
    return _ProfileObservation(
        evaluation_envelope_ns=attempt.evaluation_envelope_ns,
        device_active_ns=attempt.device_active_ns,
        phase_device_ns=attempt.phase_device_ns,
    )


def _validate_attempt(attempt: AttributionAttempt) -> None:
    _validate_binding(attempt.binding)
    if attempt.attempt_index < 0:
        raise AttributionControlError("attempt_index must be non-negative")
    _sha256(attempt.runtime_identity_sha256, "runtime_identity_sha256")
    _sha256(
        attempt.module_topology_identity_sha256,
        "module_topology_identity_sha256",
    )
    if any(not token or token.strip() != token for token in attempt.xla_flag_tokens):
        raise AttributionControlError("xla_flag_tokens must be exact non-empty tokens")
    for path_field, sha_field in (
        ("raw_trace_path", "raw_trace_sha256"),
        ("child_observation_path", "child_observation_sha256"),
        ("hlo_anchor_path", "hlo_anchor_sha256"),
    ):
        _relative_path(getattr(attempt, path_field), path_field)
        _sha256(getattr(attempt, sha_field), sha_field)
    if attempt.profile_derivation_version != PROFILE_DERIVATION_VERSION:
        raise AttributionControlError("profile_derivation_version is unsupported")
    cache_root = _root(attempt.compilation_cache_root, "compilation_cache_root")
    artifact_root = _root(attempt.artifact_root, "artifact_root")
    if cache_root == artifact_root:
        raise AttributionControlError("cache and artifact roots must be distinct")
    if not math.isfinite(attempt.objective):
        raise AttributionControlError("objective must be finite")
    if not attempt.gradient or any(
        not math.isfinite(value) for value in attempt.gradient
    ):
        raise AttributionControlError("gradient must be non-empty and finite")
    _solve_certificate_parts(attempt.solve_certificate)
    if attempt.evaluation_envelope_ns <= 0 or attempt.device_active_ns <= 0:
        raise AttributionControlError("profile durations must be positive")
    if attempt.device_active_ns > attempt.evaluation_envelope_ns:
        raise AttributionControlError("device-active time exceeds evaluation envelope")
    phase_ids = tuple(phase_id for phase_id, _ in attempt.phase_device_ns)
    if not phase_ids or len(phase_ids) != len(set(phase_ids)):
        raise AttributionControlError("phase IDs must be non-empty and unique")
    if any(
        not phase_id or duration_ns <= 0
        for phase_id, duration_ns in attempt.phase_device_ns
    ):
        raise AttributionControlError("phase IDs and durations must be positive")
    if (
        sum(duration_ns for _, duration_ns in attempt.phase_device_ns)
        > attempt.device_active_ns
    ):
        raise AttributionControlError("phase durations exceed device-active time")


def _solve_certificate_parts(
    certificate: Mapping[str, object],
) -> tuple[dict[str, object], tuple[tuple[str, float], ...]]:
    expected_keys = frozenset(
        {"inner_newton_success", "adjoint_success", "residual_certificates"}
    )
    if frozenset(certificate) != expected_keys:
        raise AttributionControlError(
            "solve_certificate must contain exact success and residual fields"
        )
    inner_newton_success = certificate["inner_newton_success"]
    adjoint_success = certificate["adjoint_success"]
    if not isinstance(inner_newton_success, bool) or not isinstance(
        adjoint_success, bool
    ):
        raise AttributionControlError("solve_certificate success fields must be bool")
    raw_residuals = certificate["residual_certificates"]
    if not isinstance(raw_residuals, Mapping) or not raw_residuals:
        raise AttributionControlError(
            "solve_certificate residual_certificates must be a non-empty mapping"
        )
    residuals: list[tuple[str, float]] = []
    for residual_name, raw_value in sorted(raw_residuals.items()):
        if not isinstance(residual_name, str) or not residual_name:
            raise AttributionControlError(
                "solve_certificate residual names must be non-empty strings"
            )
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(raw_value)
        ):
            raise AttributionControlError(
                "solve_certificate residual values must be finite numbers"
            )
        residuals.append((residual_name, float(raw_value)))
    categorical = {
        "inner_newton_success": inner_newton_success,
        "adjoint_success": adjoint_success,
        "residual_certificate_names": [name for name, _ in residuals],
    }
    return categorical, tuple(residuals)


def _solve_certificate_categorical_digest(
    certificate: Mapping[str, object],
) -> str:
    categorical, _ = _solve_certificate_parts(certificate)
    return hashlib.sha256(canonical_json_bytes(categorical)).hexdigest()


def _comparison_summary(
    reference: Sequence[float],
    observations: Sequence[Sequence[float]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    same_shape = all(len(values) == len(reference) for values in observations)
    comparisons = [
        (abs(actual - expected), atol + rtol * abs(expected))
        for values in observations
        for actual, expected in zip(values, reference, strict=False)
    ]
    max_absolute_difference = max((item[0] for item in comparisons), default=0.0)
    max_allowed_difference = max((item[1] for item in comparisons), default=atol)
    max_tolerance_ratio = max(
        (difference / allowed for difference, allowed in comparisons), default=0.0
    )
    return {
        "atol": atol,
        "rtol": rtol,
        "value_count": len(reference),
        "same_shape": same_shape,
        "observed_max_absolute_difference": max_absolute_difference,
        "observed_max_allowed_absolute_difference": max_allowed_difference,
        "observed_max_tolerance_ratio": max_tolerance_ratio,
        "within_tolerance": same_shape and max_tolerance_ratio <= 1.0,
    }


def _numerical_equivalence_from_values(
    objectives: Sequence[float],
    gradients: Sequence[Sequence[float]],
    solve_certificates: Sequence[Mapping[str, object]],
    *,
    topology_identities: set[str],
) -> dict[str, object]:
    reference_objective = objectives[0]
    reference_gradient = gradients[0]
    reference_categorical, reference_residuals = _solve_certificate_parts(
        solve_certificates[0]
    )
    certificate_parts = [
        _solve_certificate_parts(certificate) for certificate in solve_certificates
    ]
    objective = _comparison_summary(
        (reference_objective,),
        tuple((value,) for value in objectives),
        atol=PHASE0_OBJECTIVE_ATOL,
        rtol=PHASE0_OBJECTIVE_RTOL,
    )
    gradient = _comparison_summary(
        reference_gradient,
        gradients,
        atol=PHASE0_GRADIENT_ATOL,
        rtol=PHASE0_GRADIENT_RTOL,
    )
    residual_certificates = _comparison_summary(
        tuple(value for _, value in reference_residuals),
        tuple(
            tuple(value for _, value in residuals) for _, residuals in certificate_parts
        ),
        atol=PHASE0_OBJECTIVE_ATOL,
        rtol=PHASE0_OBJECTIVE_RTOL,
    )
    exact_categorical = all(
        categorical == reference_categorical for categorical, _ in certificate_parts
    )
    passing_solve_status = all(
        categorical["inner_newton_success"] is True
        and categorical["adjoint_success"] is True
        for categorical, _ in certificate_parts
    )
    quantitative_parity = all(
        comparison["within_tolerance"] is True
        for comparison in (objective, gradient, residual_certificates)
    )
    return {
        "quantitative_result_and_residual_parity": quantitative_parity,
        "exact_solve_status_and_residual_names": exact_categorical,
        "passing_solve_status": passing_solve_status,
        "exact_module_topology_identity": len(topology_identities) == 1,
        "tolerance_source": NUMERICAL_PARITY_TOLERANCE_SOURCE,
        "objective": objective,
        "gradient": gradient,
        "residual_certificates": residual_certificates,
        "module_topology_claim": (
            "exact_hlo_module_name_set_and_frozen_solver_specimen_only"
        ),
    }


def _numerical_equivalence(
    attempts: Sequence[AttributionAttempt],
    *,
    topology_identities: set[str],
) -> dict[str, object]:
    return _numerical_equivalence_from_values(
        tuple(attempt.objective for attempt in attempts),
        tuple(attempt.gradient for attempt in attempts),
        tuple(attempt.solve_certificate for attempt in attempts),
        topology_identities=topology_identities,
    )


def _document_attempt_rows(
    document: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    sections: list[tuple[Mapping[str, object], ...]] = []
    for section_name in ("direct_default_measurement", "attribution_replay"):
        section = document.get(section_name)
        if not isinstance(section, Mapping):
            raise AttributionControlError(
                f"{section_name} must be a mapping for equivalence validation"
            )
        raw_rows = section.get("attempts")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise AttributionControlError(f"{section_name}.attempts must be a sequence")
        if any(not isinstance(row, Mapping) for row in raw_rows):
            raise AttributionControlError(
                f"{section_name}.attempts must contain mappings"
            )
        sections.append(tuple(raw_rows))
    return sections[0], sections[1]


def _recompute_document_equivalence(
    document: Mapping[str, object],
) -> dict[str, object]:
    default_rows, disabled_rows = _document_attempt_rows(document)
    rows = (*default_rows, *disabled_rows)
    if not rows:
        raise AttributionControlError("attribution document contains no attempts")

    objectives: list[float] = []
    gradients: list[tuple[float, ...]] = []
    certificates: list[Mapping[str, object]] = []
    topology_identities: set[str] = set()
    for row in rows:
        objective = row.get("objective")
        raw_gradient = row.get("gradient")
        certificate = row.get("solve_certificate")
        topology_identity = row.get("module_topology_identity_sha256")
        if (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(objective)
        ):
            raise AttributionControlError("attempt objective must be finite")
        if (
            not isinstance(raw_gradient, Sequence)
            or isinstance(raw_gradient, (str, bytes))
            or not raw_gradient
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in raw_gradient
            )
        ):
            raise AttributionControlError("attempt gradient must be finite")
        if not isinstance(certificate, Mapping):
            raise AttributionControlError("attempt solve_certificate must be a mapping")
        raw_categorical_digest = row.get(
            "solve_certificate_categorical_identity_sha256"
        )
        if not isinstance(raw_categorical_digest, str):
            raise AttributionControlError(
                "solve certificate categorical identity must be a string"
            )
        recorded_categorical_digest = _sha256(
            raw_categorical_digest, "solve certificate categorical identity"
        )
        if recorded_categorical_digest != _solve_certificate_categorical_digest(
            certificate
        ):
            raise AttributionControlError(
                "solve certificate categorical identity differs from evidence"
            )
        if not isinstance(topology_identity, str):
            raise AttributionControlError("module topology identity must be a string")
        objectives.append(float(objective))
        gradients.append(tuple(float(value) for value in raw_gradient))
        certificates.append(certificate)
        topology_identities.add(_sha256(topology_identity, "module topology identity"))
    return _numerical_equivalence_from_values(
        objectives,
        gradients,
        certificates,
        topology_identities=topology_identities,
    )


def _profile_observation_from_row(
    row: Mapping[str, object],
) -> _ProfileObservation:
    envelope = row.get("evaluation_envelope_ns")
    active = row.get("device_active_ns")
    raw_phases = row.get("phase_device_ns")
    if (
        isinstance(envelope, bool)
        or not isinstance(envelope, int)
        or envelope <= 0
        or isinstance(active, bool)
        or not isinstance(active, int)
        or active <= 0
        or active > envelope
    ):
        raise AttributionControlError("attempt profile durations are invalid")
    if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)):
        raise AttributionControlError("attempt phase_device_ns must be a sequence")
    phases: list[tuple[str, int]] = []
    for raw_phase in raw_phases:
        if not isinstance(raw_phase, Mapping):
            raise AttributionControlError("attempt phase row must be a mapping")
        phase_id = raw_phase.get("phase_id")
        duration = raw_phase.get("duration_ns")
        if (
            not isinstance(phase_id, str)
            or not phase_id
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration <= 0
        ):
            raise AttributionControlError("attempt phase row is invalid")
        phases.append((phase_id, duration))
    if not phases or len(phases) != len({phase_id for phase_id, _ in phases}):
        raise AttributionControlError("attempt phase IDs must be non-empty and unique")
    if sum(duration for _, duration in phases) > active:
        raise AttributionControlError(
            "attempt phase durations exceed device-active time"
        )
    return _ProfileObservation(
        evaluation_envelope_ns=envelope,
        device_active_ns=active,
        phase_device_ns=tuple(phases),
    )


def _recompute_document_route(
    document: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    default_rows, disabled_rows = _document_attempt_rows(document)
    stability, selected, _ = _route_analysis(
        tuple(_profile_observation_from_row(row) for row in default_rows),
        tuple(_profile_observation_from_row(row) for row in disabled_rows),
    )
    return stability, selected


def canonical_module_topology_identity(
    hlo_module_set_identity: str,
    hlo_module_set_identity_source: str,
    solver_graph_specimen_sha256: str,
) -> str:
    """Hash the narrow module-name topology claim and frozen solver specimen."""

    payload = {
        "hlo_module_set_identity": _nonempty(
            hlo_module_set_identity, "hlo_module_set_identity"
        ),
        "hlo_module_set_identity_source": _nonempty(
            hlo_module_set_identity_source, "hlo_module_set_identity_source"
        ),
        "solver_graph_specimen_sha256": _sha256(
            solver_graph_specimen_sha256, "solver_graph_specimen_sha256"
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    phases = frozenset(left) | frozenset(right)
    return 0.5 * sum(
        abs(left.get(phase, 0.0) - right.get(phase, 0.0)) for phase in phases
    )


def _maximum_total_variation(rows: Sequence[Mapping[str, float]]) -> float:
    return max(
        (
            _total_variation(left, right)
            for index, left in enumerate(rows)
            for right in rows[index + 1 :]
        ),
        default=0.0,
    )


def _route_analysis(
    default_attempts: Sequence[_ProfileObservation],
    disabled_attempts: Sequence[_ProfileObservation],
) -> tuple[dict[str, object], dict[str, object] | None, tuple[str, ...]]:
    default_fraction_rows = [_phase_fractions(attempt) for attempt in default_attempts]
    disabled_fraction_rows = [
        _phase_fractions(attempt) for attempt in disabled_attempts
    ]
    default_envelope_rows = [
        _phase_envelope_shares(attempt) for attempt in default_attempts
    ]
    default_phase_set_mismatch = (
        len({frozenset(row) for row in default_fraction_rows}) > 1
    )
    disabled_phase_set_mismatch = (
        len({frozenset(row) for row in disabled_fraction_rows}) > 1
    )
    default_phase_variation = _maximum_total_variation(default_fraction_rows)
    disabled_phase_variation = _maximum_total_variation(disabled_fraction_rows)
    default_envelope_variation = _maximum_total_variation(default_envelope_rows)
    default_coverages = [
        sum(duration for _, duration in attempt.phase_device_ns)
        / attempt.device_active_ns
        for attempt in default_attempts
    ]
    disabled_coverages = [
        sum(duration for _, duration in attempt.phase_device_ns)
        / attempt.device_active_ns
        for attempt in disabled_attempts
    ]
    default_active_shares = [
        attempt.device_active_ns / attempt.evaluation_envelope_ns
        for attempt in default_attempts
    ]
    default_share_spread = (
        max(default_active_shares) - min(default_active_shares)
        if default_active_shares
        else 0.0
    )

    direct_blockers: list[str] = []
    if len(default_attempts) != ATTRIBUTION_ATTEMPT_COUNT:
        direct_blockers.append("default_attempt_count_mismatch")
    if default_phase_set_mismatch:
        direct_blockers.append("default_phase_set_mismatch")
    if default_phase_variation > MAX_PHASE_TOTAL_VARIATION_DISTANCE:
        direct_blockers.append("default_phase_fraction_instability")
    if default_envelope_variation > MAX_PHASE_TOTAL_VARIATION_DISTANCE:
        direct_blockers.append("default_phase_envelope_share_instability")
    if default_share_spread > MAX_DEFAULT_DEVICE_ACTIVE_SHARE_SPREAD:
        direct_blockers.append("default_device_active_share_instability")
    if len(default_coverages) != ATTRIBUTION_ATTEMPT_COUNT or any(
        coverage < MIN_ATTRIBUTION_COVERAGE for coverage in default_coverages
    ):
        direct_blockers.append("default_attribution_coverage_below_threshold")

    fallback_blockers: list[str] = []
    if len(disabled_attempts) != ATTRIBUTION_ATTEMPT_COUNT:
        fallback_blockers.append("disabled_attempt_count_mismatch")
    if disabled_phase_set_mismatch:
        fallback_blockers.append("disabled_phase_set_mismatch")
    if disabled_phase_variation > MAX_PHASE_TOTAL_VARIATION_DISTANCE:
        fallback_blockers.append("disabled_phase_fraction_instability")
    if default_share_spread > MAX_DEFAULT_DEVICE_ACTIVE_SHARE_SPREAD:
        fallback_blockers.append("default_device_active_share_instability")
    if len(default_active_shares) != ATTRIBUTION_ATTEMPT_COUNT:
        fallback_blockers.append("default_attempt_count_mismatch")
    if len(disabled_coverages) != ATTRIBUTION_ATTEMPT_COUNT or any(
        coverage < MIN_ATTRIBUTION_COVERAGE for coverage in disabled_coverages
    ):
        fallback_blockers.append("disabled_attribution_coverage_below_threshold")

    selected: dict[str, object] | None = None
    if not direct_blockers:
        phase_ids = tuple(
            phase_id
            for phase_id in sorted(default_envelope_rows[0])
            if phase_id != "__unattributed__"
        )
        phase_rows = [
            {
                "phase_id": phase_id,
                "selected_default_envelope_share": statistics.median(
                    row[phase_id] for row in default_envelope_rows
                ),
            }
            for phase_id in phase_ids
        ]
        selected_sum = sum(
            float(row["selected_default_envelope_share"]) for row in phase_rows
        )
        selected = {
            "route": "direct_default",
            "method": "direct-default-median-phase-envelope-share",
            "phase_shares": phase_rows,
            "unattributed_default_envelope_share": 1.0 - selected_sum,
        }
    elif not fallback_blockers:
        default_active_share = statistics.median(default_active_shares)
        phase_ids = tuple(
            phase_id
            for phase_id in sorted(disabled_fraction_rows[0])
            if phase_id != "__unattributed__"
        )
        phase_rows = []
        selected_sum = 0.0
        for phase_id in phase_ids:
            disabled_fraction = statistics.median(
                row[phase_id] for row in disabled_fraction_rows
            )
            selected_share = disabled_fraction * default_active_share
            selected_sum += selected_share
            phase_rows.append(
                {
                    "phase_id": phase_id,
                    "disabled_device_fraction": disabled_fraction,
                    "selected_default_envelope_share": selected_share,
                }
            )
        selected = {
            "route": "disabled_transfer_fallback",
            "method": "disabled-device-fraction-times-default-device-active-share",
            "default_device_active_share": default_active_share,
            "phase_shares": phase_rows,
            "unattributed_default_envelope_share": 1.0 - selected_sum,
        }

    stability = {
        "required_attempts_per_mode": ATTRIBUTION_ATTEMPT_COUNT,
        "max_phase_total_variation_distance": MAX_PHASE_TOTAL_VARIATION_DISTANCE,
        "observed_default_phase_total_variation_distance": default_phase_variation,
        "observed_disabled_phase_total_variation_distance": disabled_phase_variation,
        "max_default_phase_envelope_total_variation_distance": (
            MAX_PHASE_TOTAL_VARIATION_DISTANCE
        ),
        "observed_default_phase_envelope_total_variation_distance": (
            default_envelope_variation
        ),
        "max_default_device_active_share_spread": (
            MAX_DEFAULT_DEVICE_ACTIVE_SHARE_SPREAD
        ),
        "observed_default_device_active_share_spread": default_share_spread,
        "minimum_attribution_coverage": MIN_ATTRIBUTION_COVERAGE,
        "observed_minimum_default_attribution_coverage": (
            min(default_coverages) if default_coverages else 0.0
        ),
        "observed_minimum_disabled_attribution_coverage": (
            min(disabled_coverages) if disabled_coverages else 0.0
        ),
        "direct_default_route": {
            "eligible": not direct_blockers,
            "blockers": direct_blockers,
        },
        "disabled_transfer_fallback": {
            "eligible": not fallback_blockers,
            "blockers": fallback_blockers,
        },
    }
    route_blockers = tuple(dict.fromkeys((*direct_blockers, *fallback_blockers)))
    return stability, selected, route_blockers


def canonical_attribution_attempt_row(
    attempt: AttributionAttempt,
) -> dict[str, object]:
    """Validate and encode one attempt exactly as transfer evidence."""

    _validate_attempt(attempt)
    attributed_ns = sum(duration_ns for _, duration_ns in attempt.phase_device_ns)
    return {
        "mode": attempt.mode,
        "attempt_index": attempt.attempt_index,
        "runtime_identity_sha256": attempt.runtime_identity_sha256,
        "xla_flag_tokens": list(attempt.xla_flag_tokens),
        "compilation_cache_root": attempt.compilation_cache_root,
        "artifact_root": attempt.artifact_root,
        "raw_trace_path": attempt.raw_trace_path,
        "raw_trace_sha256": attempt.raw_trace_sha256,
        "child_observation_path": attempt.child_observation_path,
        "child_observation_sha256": attempt.child_observation_sha256,
        "hlo_anchor_path": attempt.hlo_anchor_path,
        "hlo_anchor_sha256": attempt.hlo_anchor_sha256,
        "profile_derivation_version": attempt.profile_derivation_version,
        "objective": attempt.objective,
        "gradient": list(attempt.gradient),
        "solve_certificate": attempt.solve_certificate,
        "solve_certificate_categorical_identity_sha256": (
            _solve_certificate_categorical_digest(attempt.solve_certificate)
        ),
        "module_topology_identity_sha256": (attempt.module_topology_identity_sha256),
        "evaluation_envelope_ns": attempt.evaluation_envelope_ns,
        "device_active_ns": attempt.device_active_ns,
        "device_active_share": attempt.device_active_ns
        / attempt.evaluation_envelope_ns,
        "phase_device_ns": [
            {"phase_id": phase_id, "duration_ns": duration_ns}
            for phase_id, duration_ns in attempt.phase_device_ns
        ],
        "attribution_coverage": attributed_ns / attempt.device_active_ns,
    }


def build_attribution_evidence(
    default_attempts: Sequence[AttributionAttempt],
    disabled_attempts: Sequence[AttributionAttempt],
) -> dict[str, object]:
    """Select direct attribution or a disabled-control fallback, fail closed."""

    attempts = tuple(default_attempts) + tuple(disabled_attempts)
    for attempt in attempts:
        _validate_attempt(attempt)

    blockers: list[str] = []
    if len(default_attempts) != ATTRIBUTION_ATTEMPT_COUNT:
        blockers.append("default_attempt_count_mismatch")
    if len(disabled_attempts) != ATTRIBUTION_ATTEMPT_COUNT:
        blockers.append("disabled_attempt_count_mismatch")
    if not attempts:
        raise AttributionControlError("at least one attempt is required")

    for expected_mode, mode_attempts in (
        ("default_control", default_attempts),
        ("command_buffer_disabled", disabled_attempts),
    ):
        if tuple(attempt.attempt_index for attempt in mode_attempts) != tuple(
            range(len(mode_attempts))
        ):
            blockers.append(f"{expected_mode}_attempt_indices_mismatch")
        if any(attempt.mode != expected_mode for attempt in mode_attempts):
            blockers.append(f"{expected_mode}_mode_mismatch")

    reference = attempts[0]
    if any(attempt.binding != reference.binding for attempt in attempts[1:]):
        blockers.append("production_binding_mismatch")
    topology_identities = {
        attempt.module_topology_identity_sha256 for attempt in attempts
    }
    equivalence = _numerical_equivalence(
        attempts,
        topology_identities=topology_identities,
    )
    if equivalence["quantitative_result_and_residual_parity"] is not True:
        blockers.append("result_or_solve_certificate_mismatch")
    if equivalence["exact_solve_status_and_residual_names"] is not True:
        blockers.append("solve_status_or_residual_names_mismatch")
    if equivalence["passing_solve_status"] is not True:
        blockers.append("solve_status_not_passing")
    if len(topology_identities) != 1:
        blockers.append("module_topology_identity_mismatch")

    default_runtime_ids = {
        attempt.runtime_identity_sha256 for attempt in default_attempts
    }
    disabled_runtime_ids = {
        attempt.runtime_identity_sha256 for attempt in disabled_attempts
    }
    if len(default_runtime_ids) != 1:
        blockers.append("default_runtime_identity_instability")
    if default_runtime_ids != {reference.binding.production_runtime_identity_sha256}:
        blockers.append("default_runtime_identity_not_production")
    if len(disabled_runtime_ids) != 1:
        blockers.append("disabled_runtime_identity_instability")
    if default_runtime_ids & disabled_runtime_ids:
        blockers.append("disabled_runtime_identity_not_distinct")
    default_token_sequences = {attempt.xla_flag_tokens for attempt in default_attempts}
    if len(default_token_sequences) != 1:
        blockers.append("default_xla_token_sequence_instability")
    else:
        default_tokens = next(iter(default_token_sequences))
        if any(
            token.startswith("--xla_gpu_enable_command_buffer")
            for token in default_tokens
        ):
            blockers.append("default_command_buffer_configuration_not_authoritative")
        expected_disabled_tokens = (*default_tokens, COMMAND_BUFFER_DISABLE_FLAG)
        if any(
            attempt.xla_flag_tokens != expected_disabled_tokens
            for attempt in disabled_attempts
        ):
            blockers.append("disabled_xla_token_sequence_not_exact_extension")

    cache_roots = [attempt.compilation_cache_root for attempt in attempts]
    artifact_roots = [attempt.artifact_root for attempt in attempts]
    if len(cache_roots) != len(set(cache_roots)):
        blockers.append("compilation_cache_root_reused")
    if len(artifact_roots) != len(set(artifact_roots)):
        blockers.append("artifact_root_reused")
    if set(cache_roots) & set(artifact_roots):
        blockers.append("cache_and_artifact_roots_overlap")

    stability, selected_candidate, route_blockers = _route_analysis(
        tuple(_profile_observation(attempt) for attempt in default_attempts),
        tuple(_profile_observation(attempt) for attempt in disabled_attempts),
    )
    if selected_candidate is None:
        blockers.extend(route_blockers)
    blockers = list(dict.fromkeys(blockers))
    selected_attribution = selected_candidate if not blockers else None

    return {
        "schema_id": ATTRIBUTION_EVIDENCE_SCHEMA_ID,
        "state": "PRODUCED" if selected_attribution is not None else "NON_PROMOTING",
        "promotion_eligible": selected_attribution is not None,
        "blockers": blockers,
        "production_binding": reference.binding.to_json(),
        "direct_default_measurement": {
            "authoritative_for_timing": True,
            "attempts": [
                canonical_attribution_attempt_row(attempt)
                for attempt in default_attempts
            ],
        },
        "attribution_replay": {
            "authoritative_for_timing": False,
            "attempts": [
                canonical_attribution_attempt_row(attempt)
                for attempt in disabled_attempts
            ],
        },
        "equivalence": equivalence,
        "stability": stability,
        "selected_attribution": selected_attribution,
    }


def require_promoting_attribution_evidence(document: Mapping[str, object]) -> None:
    """Recompute and require a promotion-safe selected attribution route."""

    if document.get("schema_id") != ATTRIBUTION_EVIDENCE_SCHEMA_ID:
        raise AttributionControlError("unexpected attribution-evidence schema")
    if (
        document.get("state") != "PRODUCED"
        or document.get("promotion_eligible") is not True
    ):
        raise AttributionControlError(
            "attribution evidence is explicitly non-promoting"
        )
    if document.get("blockers") != [] or document.get("selected_attribution") is None:
        raise AttributionControlError(
            "promoting attribution evidence is internally inconsistent"
        )
    recorded_equivalence = document.get("equivalence")
    if not isinstance(recorded_equivalence, Mapping):
        raise AttributionControlError("equivalence must be a mapping")
    recomputed_equivalence = _recompute_document_equivalence(document)
    if canonical_json_bytes(recorded_equivalence) != canonical_json_bytes(
        recomputed_equivalence
    ):
        raise AttributionControlError(
            "recorded numerical equivalence differs from attempt evidence"
        )
    if (
        recomputed_equivalence["quantitative_result_and_residual_parity"] is not True
        or recomputed_equivalence["exact_solve_status_and_residual_names"] is not True
        or recomputed_equivalence["passing_solve_status"] is not True
        or recomputed_equivalence["exact_module_topology_identity"] is not True
    ):
        raise AttributionControlError("attribution equivalence checks did not pass")
    recorded_stability = document.get("stability")
    recorded_selection = document.get("selected_attribution")
    recomputed_stability, recomputed_selection = _recompute_document_route(document)
    if canonical_json_bytes(recorded_stability) != canonical_json_bytes(
        recomputed_stability
    ):
        raise AttributionControlError(
            "recorded attribution stability differs from attempt evidence"
        )
    if canonical_json_bytes(recorded_selection) != canonical_json_bytes(
        recomputed_selection
    ):
        raise AttributionControlError(
            "selected attribution differs from attempt evidence"
        )


__all__ = [
    "ATTRIBUTION_ATTEMPT_COUNT",
    "ATTRIBUTION_EVIDENCE_SCHEMA_ID",
    "COMMAND_BUFFER_DISABLE_FLAG",
    "MAX_DEFAULT_DEVICE_ACTIVE_SHARE_SPREAD",
    "MAX_PHASE_TOTAL_VARIATION_DISTANCE",
    "MIN_ATTRIBUTION_COVERAGE",
    "NUMERICAL_PARITY_TOLERANCE_SOURCE",
    "PROFILE_DERIVATION_VERSION",
    "AttributionAttempt",
    "AttributionBinding",
    "AttributionControlError",
    "build_attribution_evidence",
    "canonical_attribution_attempt_row",
    "canonical_module_topology_identity",
    "require_promoting_attribution_evidence",
]
