"""Fail-closed authority preflight for the CFS-SQP1 endpoint audit.

This module performs no physics evaluation.  It validates that a future audit
runner has complete, content-addressed authority before native, JAX, branch,
projection, or field-line work starts.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Final

from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    JsonValue,
    canonical_json_bytes,
    load_canonical_json_bytes,
)
from simsopt_jax.objectives.single_stage_fullspace import FullSpaceRawTerms
from simsopt_jax.solve.fullspace import FullSpaceRoute
from simsopt_jax.solve.fullspace_certificate import (
    BRANCH_STATE_INFINITY_TOLERANCE,
    CFS_SQP1_CERTIFICATE_SCHEMA_VERSION,
    CROSS_EVALUATOR_ATOL,
    CROSS_EVALUATOR_RTOL,
    PROJECTION_CONSTRAINT_INFINITY_TOLERANCE,
    PROJECTION_OBJECTIVE_TOLERANCE,
    PROJECTION_STATE_INFINITY_TOLERANCE,
    TRACED_IOTA_TOLERANCE,
    CertificateChecks,
    CfsSqp1EndpointCertificate,
    EndpointNumerics,
)

AUTHORITY_SCHEMA_VERSION: Final = "single-stage-fullspace-endpoint-audit-authority-v1"
PREFLIGHT_SCHEMA_VERSION: Final = "single-stage-fullspace-endpoint-audit-preflight-v1"
OUTPUT_SCHEMA_VERSION: Final = "single-stage-fullspace-endpoint-audit-output-v1"
EXPECTED_STATE_SIZE: Final = 716
EXACT_BRANCH_NEWTON_TOLERANCE: Final = 1.0e-13
EXACT_BRANCH_MAXIMUM_NEWTON_ITERATIONS: Final = 20


class EndpointAuditDisposition(StrEnum):
    READY = "READY"
    NOT_PRODUCED = "NOT_PRODUCED"


class EndpointAuditOutputDisposition(StrEnum):
    PRODUCED = "PRODUCED"
    NOT_PRODUCED = "NOT_PRODUCED"


class MissingAuthority(StrEnum):
    HISTORICAL_NATIVE_ENDPOINT = "HISTORICAL_NATIVE_ENDPOINT"
    FIELDLINE = "FIELDLINE"


class EndpointAuditAuthorityError(ValueError):
    """An asserted endpoint-audit authority failed an integrity boundary."""


def _artifact_payload(reference: ArtifactRef) -> dict[str, JsonValue]:
    return asdict(reference)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EndpointAuditAuthorityError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EndpointAuditAuthorityError(f"{name} must be a finite number")
    return normalized


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise EndpointAuditAuthorityError(f"{name} must be finite and positive")


def _extract(document: JsonValue, path: tuple[str, ...], name: str) -> JsonValue:
    current = document
    for component in path:
        if not isinstance(current, dict) or component not in current:
            raise EndpointAuditAuthorityError(
                f"{name} JSON path does not exist: {'.'.join(path)}"
            )
        current = current[component]
    return current


@dataclass(frozen=True, slots=True)
class EndpointVectorAuthority:
    """Content-addressed FP64 JSON endpoint and its exact vector location."""

    artifact: ArtifactRef
    vector_path: tuple[str, ...]
    dtype_path: tuple[str, ...]
    vector_sha256: str
    coordinate_count: int = EXPECTED_STATE_SIZE

    def validate(self, campaign_root: Path, name: str) -> None:
        path = self.artifact.resolve_and_validate(campaign_root)
        document = load_canonical_json_bytes(path.read_bytes())
        vector = _extract(document, self.vector_path, name)
        dtype = _extract(document, self.dtype_path, f"{name} dtype")
        if dtype != "float64":
            raise EndpointAuditAuthorityError(f"{name} dtype must be float64")
        if self.coordinate_count != EXPECTED_STATE_SIZE:
            raise EndpointAuditAuthorityError(
                f"{name} coordinate_count must be {EXPECTED_STATE_SIZE}"
            )
        if not isinstance(vector, list) or len(vector) != self.coordinate_count:
            raise EndpointAuditAuthorityError(
                f"{name} must resolve to {self.coordinate_count} coordinates"
            )
        for index, value in enumerate(vector):
            _finite_number(value, f"{name}[{index}]")
        observed = hashlib.sha256(canonical_json_bytes(vector)).hexdigest()
        if observed != self.vector_sha256:
            raise EndpointAuditAuthorityError(f"{name} vector digest mismatch")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact": _artifact_payload(self.artifact),
            "coordinate_count": self.coordinate_count,
            "dtype_path": list(self.dtype_path),
            "vector_path": list(self.vector_path),
            "vector_sha256": self.vector_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluatorAuthority:
    """Exact source, runtime, and configuration identity for one evaluator."""

    evaluator: str
    source_manifest: ArtifactRef
    runtime_evidence: ArtifactRef
    configuration: ArtifactRef

    def validate(self, campaign_root: Path, expected: str) -> None:
        if self.evaluator != expected:
            raise EndpointAuditAuthorityError(
                f"expected {expected} evaluator, got {self.evaluator}"
            )
        self.source_manifest.resolve_and_validate(campaign_root)
        self.runtime_evidence.resolve_and_validate(campaign_root)
        self.configuration.resolve_and_validate(campaign_root)

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "configuration": _artifact_payload(self.configuration),
            "evaluator": self.evaluator,
            "runtime_evidence": _artifact_payload(self.runtime_evidence),
            "source_manifest": _artifact_payload(self.source_manifest),
        }


@dataclass(frozen=True, slots=True)
class CrossEvaluatorAuthority:
    """Both evaluators needed for native-on-JAX and JAX-on-native checks."""

    native: EvaluatorAuthority
    jax: EvaluatorAuthority
    relative_tolerance: float = CROSS_EVALUATOR_RTOL
    absolute_tolerance: float = CROSS_EVALUATOR_ATOL

    def validate(self, campaign_root: Path) -> None:
        self.native.validate(campaign_root, "native")
        self.jax.validate(campaign_root, "jax")
        if self.relative_tolerance != CROSS_EVALUATOR_RTOL:
            raise EndpointAuditAuthorityError("cross-evaluator rtol is not frozen")
        if self.absolute_tolerance != CROSS_EVALUATOR_ATOL:
            raise EndpointAuditAuthorityError("cross-evaluator atol is not frozen")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "absolute_tolerance": self.absolute_tolerance,
            "jax": self.jax.to_payload(),
            "native": self.native.to_payload(),
            "relative_tolerance": self.relative_tolerance,
        }


@dataclass(frozen=True, slots=True)
class ExactBranchAuthority:
    """Frozen exact Newton branch-reproduction policy and provenance."""

    configuration: ArtifactRef
    newton_tolerance: float
    maximum_newton_iterations: int
    reproduced_state_infinity_tolerance: float
    basin_classification_rule: str
    material_branch_switch_rule: str

    def validate(self, campaign_root: Path) -> None:
        self.configuration.resolve_and_validate(campaign_root)
        _positive(self.newton_tolerance, "branch newton_tolerance")
        if self.newton_tolerance != EXACT_BRANCH_NEWTON_TOLERANCE:
            raise EndpointAuditAuthorityError("branch Newton tolerance is not frozen")
        if self.maximum_newton_iterations != EXACT_BRANCH_MAXIMUM_NEWTON_ITERATIONS:
            raise EndpointAuditAuthorityError(
                "branch maximum Newton iterations are not frozen"
            )
        if self.reproduced_state_infinity_tolerance != BRANCH_STATE_INFINITY_TOLERANCE:
            raise EndpointAuditAuthorityError("branch state tolerance is not frozen")
        if not self.basin_classification_rule.strip():
            raise EndpointAuditAuthorityError("basin classification rule is empty")
        if not self.material_branch_switch_rule.strip():
            raise EndpointAuditAuthorityError("material branch-switch rule is empty")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "basin_classification_rule": self.basin_classification_rule,
            "configuration": _artifact_payload(self.configuration),
            "material_branch_switch_rule": self.material_branch_switch_rule,
            "maximum_newton_iterations": self.maximum_newton_iterations,
            "newton_tolerance": self.newton_tolerance,
            "reproduced_state_infinity_tolerance": (
                self.reproduced_state_infinity_tolerance
            ),
        }


@dataclass(frozen=True, slots=True)
class ProjectionAuthority:
    """Frozen immateriality thresholds for optional exact projection."""

    state_infinity_tolerance: float = PROJECTION_STATE_INFINITY_TOLERANCE
    objective_tolerance: float = PROJECTION_OBJECTIVE_TOLERANCE
    constraint_infinity_tolerance: float = PROJECTION_CONSTRAINT_INFINITY_TOLERANCE

    def validate(self) -> None:
        expected = (
            PROJECTION_STATE_INFINITY_TOLERANCE,
            PROJECTION_OBJECTIVE_TOLERANCE,
            PROJECTION_CONSTRAINT_INFINITY_TOLERANCE,
        )
        observed = (
            self.state_infinity_tolerance,
            self.objective_tolerance,
            self.constraint_infinity_tolerance,
        )
        if observed != expected:
            raise EndpointAuditAuthorityError("projection tolerances are not frozen")

    def to_payload(self) -> dict[str, JsonValue]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FieldLineAuthority:
    """Exact seed, integration, closure, and traced-iota protocol."""

    configuration: ArtifactRef
    radial_seeds: tuple[float, ...]
    vertical_seeds: tuple[float, ...]
    phi_sections: tuple[float, ...]
    integrator: str
    integration_tolerance: float
    maximum_integration_iterations: int
    turn_count: int
    closure_tolerance: float
    traced_iota_estimator: str
    traced_iota_tolerance: float = TRACED_IOTA_TOLERANCE

    def validate(self, campaign_root: Path) -> None:
        self.configuration.resolve_and_validate(campaign_root)
        if not self.radial_seeds or len(self.radial_seeds) != len(self.vertical_seeds):
            raise EndpointAuditAuthorityError(
                "field-line R/Z seed vectors must have equal nonzero length"
            )
        if not self.phi_sections:
            raise EndpointAuditAuthorityError("field-line phi sections are empty")
        for name, values in (
            ("radial_seeds", self.radial_seeds),
            ("vertical_seeds", self.vertical_seeds),
            ("phi_sections", self.phi_sections),
        ):
            for index, value in enumerate(values):
                _finite_number(value, f"field_line.{name}[{index}]")
        if not self.integrator.strip() or not self.traced_iota_estimator.strip():
            raise EndpointAuditAuthorityError(
                "field-line integrator and iota estimator must be named"
            )
        _positive(self.integration_tolerance, "field-line integration_tolerance")
        _positive(self.closure_tolerance, "field-line closure_tolerance")
        if self.maximum_integration_iterations <= 0 or self.turn_count <= 0:
            raise EndpointAuditAuthorityError(
                "field-line iteration and turn counts must be positive"
            )
        if self.traced_iota_tolerance != TRACED_IOTA_TOLERANCE:
            raise EndpointAuditAuthorityError("traced-iota tolerance is not frozen")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "closure_tolerance": self.closure_tolerance,
            "configuration": _artifact_payload(self.configuration),
            "integration_tolerance": self.integration_tolerance,
            "integrator": self.integrator,
            "maximum_integration_iterations": self.maximum_integration_iterations,
            "phi_sections": list(self.phi_sections),
            "radial_seeds": list(self.radial_seeds),
            "traced_iota_estimator": self.traced_iota_estimator,
            "traced_iota_tolerance": self.traced_iota_tolerance,
            "turn_count": self.turn_count,
            "vertical_seeds": list(self.vertical_seeds),
        }


@dataclass(frozen=True, slots=True)
class ProvenanceAuthority:
    """Campaign-wide source/runtime/bootstrap evidence for endpoint work."""

    source_manifest: ArtifactRef
    runtime_evidence: ArtifactRef
    bootstrap_artifact: ArtifactRef

    def validate(self, campaign_root: Path) -> None:
        self.source_manifest.resolve_and_validate(campaign_root)
        self.runtime_evidence.resolve_and_validate(campaign_root)
        self.bootstrap_artifact.resolve_and_validate(campaign_root)

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "bootstrap_artifact": _artifact_payload(self.bootstrap_artifact),
            "runtime_evidence": _artifact_payload(self.runtime_evidence),
            "source_manifest": _artifact_payload(self.source_manifest),
        }


@dataclass(frozen=True, slots=True)
class CertificatePayloadBinding:
    """Exact live scientific-certificate payload shape expected downstream."""

    schema_version: str
    certificate_fields: tuple[str, ...]
    endpoint_fields: tuple[str, ...]
    raw_objective_term_fields: tuple[str, ...]
    check_fields: tuple[str, ...]

    @classmethod
    def current(cls) -> CertificatePayloadBinding:
        return cls(
            schema_version=CFS_SQP1_CERTIFICATE_SCHEMA_VERSION,
            certificate_fields=tuple(
                field.name for field in fields(CfsSqp1EndpointCertificate)
            ),
            endpoint_fields=tuple(field.name for field in fields(EndpointNumerics)),
            raw_objective_term_fields=tuple(
                field.name for field in fields(FullSpaceRawTerms)
            ),
            check_fields=tuple(field.name for field in fields(CertificateChecks)),
        )

    def validate(self) -> None:
        if self != self.current():
            raise EndpointAuditAuthorityError(
                "certificate payload binding differs from the live contract"
            )

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "certificate_fields": list(self.certificate_fields),
            "check_fields": list(self.check_fields),
            "endpoint_fields": list(self.endpoint_fields),
            "raw_objective_term_fields": list(self.raw_objective_term_fields),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class EndpointAuditAuthority:
    """Complete immutable authority needed before endpoint audit execution."""

    schema_version: str
    route: FullSpaceRoute
    candidate_endpoint: EndpointVectorAuthority
    historical_native_endpoint: EndpointVectorAuthority | None
    cross_evaluator: CrossEvaluatorAuthority
    exact_branch: ExactBranchAuthority
    projection: ProjectionAuthority
    field_line: FieldLineAuthority | None
    provenance: ProvenanceAuthority
    certificate_payload: CertificatePayloadBinding

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "candidate_endpoint": self.candidate_endpoint.to_payload(),
            "certificate_payload": self.certificate_payload.to_payload(),
            "cross_evaluator": self.cross_evaluator.to_payload(),
            "exact_branch": self.exact_branch.to_payload(),
            "field_line": (
                None if self.field_line is None else self.field_line.to_payload()
            ),
            "historical_native_endpoint": (
                None
                if self.historical_native_endpoint is None
                else self.historical_native_endpoint.to_payload()
            ),
            "projection": self.projection.to_payload(),
            "provenance": self.provenance.to_payload(),
            "route": self.route.value,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class EndpointAuditPreflight:
    """Canonical non-promoting authorization result for a future runner."""

    schema_version: str
    disposition: EndpointAuditDisposition
    missing_authority: tuple[MissingAuthority, ...]
    endpoint_work_authorized: bool
    promotion_eligible: bool
    authority_sha256: str
    certificate_payload_sha256: str

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "authority_sha256": self.authority_sha256,
            "certificate_payload_sha256": self.certificate_payload_sha256,
            "disposition": self.disposition.value,
            "endpoint_work_authorized": self.endpoint_work_authorized,
            "missing_authority": [item.value for item in self.missing_authority],
            "promotion_eligible": self.promotion_eligible,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class RawObjectiveTermValues:
    """The five raw physical objective terms in frozen ledger order."""

    non_qs: float
    residual: float
    iota: float
    major_radius: float
    length: float

    def validate(self, name: str) -> None:
        for field in fields(self):
            _finite_number(getattr(self, field.name), f"{name}.{field.name}")

    def to_payload(self) -> dict[str, JsonValue]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateStateOutput:
    """Validated candidate state identity consumed by every endpoint audit."""

    artifact: ArtifactRef
    vector_sha256: str
    coordinate_count: int
    dtype: str

    def validate(self, campaign_root: Path) -> None:
        self.artifact.resolve_and_validate(campaign_root)
        if self.coordinate_count != EXPECTED_STATE_SIZE or self.dtype != "float64":
            raise EndpointAuditAuthorityError("candidate output shape/dtype mismatch")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact": _artifact_payload(self.artifact),
            "coordinate_count": self.coordinate_count,
            "dtype": self.dtype,
            "vector_sha256": self.vector_sha256,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluationOutput:
    """One evaluator's independently produced objective and raw-term ledger."""

    artifact: ArtifactRef
    evaluator: str
    endpoint_vector_sha256: str
    objective: float
    raw_objective_terms: RawObjectiveTermValues

    def validate(self, campaign_root: Path, expected_evaluator: str) -> None:
        self.artifact.resolve_and_validate(campaign_root)
        if self.evaluator != expected_evaluator:
            raise EndpointAuditAuthorityError("objective evaluator identity mismatch")
        _finite_number(self.objective, f"{self.evaluator}.objective")
        self.raw_objective_terms.validate(f"{self.evaluator}.raw_objective_terms")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact": _artifact_payload(self.artifact),
            "endpoint_vector_sha256": self.endpoint_vector_sha256,
            "evaluator": self.evaluator,
            "objective": self.objective,
            "raw_objective_terms": self.raw_objective_terms.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class ExactBranchOutput:
    artifact: ArtifactRef
    exact_solve_succeeded: bool
    material_branch_switch: bool
    reproduced_state_infinity_difference: float
    basin_classification: str

    def validate(self, campaign_root: Path) -> None:
        self.artifact.resolve_and_validate(campaign_root)
        difference = _finite_number(
            self.reproduced_state_infinity_difference,
            "exact_branch.reproduced_state_infinity_difference",
        )
        if difference < 0.0 or not self.basin_classification.strip():
            raise EndpointAuditAuthorityError("exact-branch output is incomplete")

    def to_payload(self) -> dict[str, JsonValue]:
        return {"artifact": _artifact_payload(self.artifact), **asdict(self)}


@dataclass(frozen=True, slots=True)
class ProjectionOutput:
    artifact: ArtifactRef
    evaluated: bool
    used: bool
    pre_state_sha256: str
    post_state_sha256: str
    state_infinity_change: float
    objective_absolute_change: float
    constraint_infinity_change: float

    def validate(self, campaign_root: Path) -> None:
        self.artifact.resolve_and_validate(campaign_root)
        for name in (
            "state_infinity_change",
            "objective_absolute_change",
            "constraint_infinity_change",
        ):
            if _finite_number(getattr(self, name), f"projection.{name}") < 0.0:
                raise EndpointAuditAuthorityError(
                    "projection changes must be nonnegative"
                )

    def to_payload(self) -> dict[str, JsonValue]:
        return {"artifact": _artifact_payload(self.artifact), **asdict(self)}


@dataclass(frozen=True, slots=True)
class FieldLineOutput:
    artifact: ArtifactRef
    poincare_closed: bool
    traced_iota: float
    maximum_closure_error: float
    turns_completed: int

    def validate(self, campaign_root: Path) -> None:
        self.artifact.resolve_and_validate(campaign_root)
        _finite_number(self.traced_iota, "field_line.traced_iota")
        closure = _finite_number(
            self.maximum_closure_error, "field_line.maximum_closure_error"
        )
        if closure < 0.0 or self.turns_completed <= 0:
            raise EndpointAuditAuthorityError("field-line output is incomplete")

    def to_payload(self) -> dict[str, JsonValue]:
        return {"artifact": _artifact_payload(self.artifact), **asdict(self)}


@dataclass(frozen=True, slots=True)
class EndpointAuditOutput:
    """Typed audit result; NOT_PRODUCED forbids all scientific evidence."""

    schema_version: str
    disposition: EndpointAuditOutputDisposition
    promotion_eligible: bool
    preflight_sha256: str
    missing_authority: tuple[MissingAuthority, ...]
    candidate_state: CandidateStateOutput | None
    native_on_jax: ObjectiveEvaluationOutput | None
    jax_on_historical_native: ObjectiveEvaluationOutput | None
    exact_branch: ExactBranchOutput | None
    projection: ProjectionOutput | None
    field_line: FieldLineOutput | None
    provenance: ProvenanceAuthority | None
    scientific_certificate: ArtifactRef | None

    def validate(self, campaign_root: Path) -> None:
        if self.schema_version != OUTPUT_SCHEMA_VERSION:
            raise EndpointAuditAuthorityError("endpoint-audit output schema mismatch")
        scientific = (
            self.candidate_state,
            self.native_on_jax,
            self.jax_on_historical_native,
            self.exact_branch,
            self.projection,
            self.field_line,
            self.provenance,
            self.scientific_certificate,
        )
        if self.disposition is EndpointAuditOutputDisposition.NOT_PRODUCED:
            if (
                self.promotion_eligible
                or not self.missing_authority
                or any(value is not None for value in scientific)
            ):
                raise EndpointAuditAuthorityError(
                    "NOT_PRODUCED output must be nonpromoting and evidence-free"
                )
            return
        if self.missing_authority or any(value is None for value in scientific):
            raise EndpointAuditAuthorityError(
                "PRODUCED output requires complete evidence"
            )
        assert self.candidate_state is not None
        assert self.native_on_jax is not None
        assert self.jax_on_historical_native is not None
        assert self.exact_branch is not None
        assert self.projection is not None
        assert self.field_line is not None
        assert self.provenance is not None
        assert self.scientific_certificate is not None
        self.candidate_state.validate(campaign_root)
        self.native_on_jax.validate(campaign_root, "native")
        self.jax_on_historical_native.validate(campaign_root, "jax")
        self.exact_branch.validate(campaign_root)
        self.projection.validate(campaign_root)
        self.field_line.validate(campaign_root)
        self.provenance.validate(campaign_root)
        self.scientific_certificate.resolve_and_validate(campaign_root)

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "candidate_state": (
                None
                if self.candidate_state is None
                else self.candidate_state.to_payload()
            ),
            "disposition": self.disposition.value,
            "exact_branch": (
                None if self.exact_branch is None else self.exact_branch.to_payload()
            ),
            "field_line": (
                None if self.field_line is None else self.field_line.to_payload()
            ),
            "jax_on_historical_native": (
                None
                if self.jax_on_historical_native is None
                else self.jax_on_historical_native.to_payload()
            ),
            "missing_authority": [item.value for item in self.missing_authority],
            "native_on_jax": (
                None if self.native_on_jax is None else self.native_on_jax.to_payload()
            ),
            "preflight_sha256": self.preflight_sha256,
            "projection": (
                None if self.projection is None else self.projection.to_payload()
            ),
            "promotion_eligible": self.promotion_eligible,
            "provenance": (
                None if self.provenance is None else self.provenance.to_payload()
            ),
            "schema_version": self.schema_version,
            "scientific_certificate": (
                None
                if self.scientific_certificate is None
                else _artifact_payload(self.scientific_certificate)
            ),
        }


def preflight_endpoint_audit_authority(
    campaign_root: Path,
    authority: EndpointAuditAuthority,
) -> EndpointAuditPreflight:
    """Validate asserted bytes and authorize work only for complete authority."""

    if authority.schema_version != AUTHORITY_SCHEMA_VERSION:
        raise EndpointAuditAuthorityError("endpoint-audit authority schema mismatch")
    if authority.route is not FullSpaceRoute.CFS_SQP1:
        raise EndpointAuditAuthorityError("endpoint audit requires CFS-SQP1")

    authority.candidate_endpoint.validate(campaign_root, "candidate endpoint")
    authority.cross_evaluator.validate(campaign_root)
    authority.exact_branch.validate(campaign_root)
    authority.projection.validate()
    authority.provenance.validate(campaign_root)
    authority.certificate_payload.validate()

    missing: list[MissingAuthority] = []
    if authority.historical_native_endpoint is None:
        missing.append(MissingAuthority.HISTORICAL_NATIVE_ENDPOINT)
    else:
        authority.historical_native_endpoint.validate(
            campaign_root, "historical native endpoint"
        )
    if authority.field_line is None:
        missing.append(MissingAuthority.FIELDLINE)
    else:
        authority.field_line.validate(campaign_root)

    authority_payload = authority.to_payload()
    certificate_payload = authority.certificate_payload.to_payload()
    ready = not missing
    return EndpointAuditPreflight(
        schema_version=PREFLIGHT_SCHEMA_VERSION,
        disposition=(
            EndpointAuditDisposition.READY
            if ready
            else EndpointAuditDisposition.NOT_PRODUCED
        ),
        missing_authority=tuple(missing),
        endpoint_work_authorized=ready,
        promotion_eligible=False,
        authority_sha256=hashlib.sha256(
            canonical_json_bytes(authority_payload)
        ).hexdigest(),
        certificate_payload_sha256=hashlib.sha256(
            canonical_json_bytes(certificate_payload)
        ).hexdigest(),
    )


def write_preflight(path: Path, preflight: EndpointAuditPreflight) -> None:
    """Write one canonical preflight result without implying audit completion."""

    path.write_bytes(canonical_json_bytes(preflight.to_payload()))


def not_produced_output(preflight: EndpointAuditPreflight) -> EndpointAuditOutput:
    """Convert an incomplete preflight into an evidence-free audit result."""

    if preflight.disposition is not EndpointAuditDisposition.NOT_PRODUCED:
        raise EndpointAuditAuthorityError(
            "only an incomplete preflight can produce NOT_PRODUCED output"
        )
    preflight_sha256 = hashlib.sha256(
        canonical_json_bytes(preflight.to_payload())
    ).hexdigest()
    return EndpointAuditOutput(
        schema_version=OUTPUT_SCHEMA_VERSION,
        disposition=EndpointAuditOutputDisposition.NOT_PRODUCED,
        promotion_eligible=False,
        preflight_sha256=preflight_sha256,
        missing_authority=preflight.missing_authority,
        candidate_state=None,
        native_on_jax=None,
        jax_on_historical_native=None,
        exact_branch=None,
        projection=None,
        field_line=None,
        provenance=None,
        scientific_certificate=None,
    )


def write_output(path: Path, output: EndpointAuditOutput, campaign_root: Path) -> None:
    """Validate then write one canonical endpoint-audit output."""

    output.validate(campaign_root)
    path.write_bytes(canonical_json_bytes(output.to_payload()))
