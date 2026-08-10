"""Canonical evidence contracts for the single-stage full-space campaign."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
from simsopt_jax.objectives.single_stage_fullspace import (
    frozen_problem_contract_payload,
)
from simsopt_jax.solve.fullspace import (
    LEGACY_V1_ROUTES,
    FullSpaceRoute,
    frozen_route_contract_payload_v1,
    frozen_route_contract_payload_v2,
    frozen_route_contract_payload_v3,
)

from benchmarks import single_stage_fullspace_snapshot as _snapshot_contract
from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    JsonValue,
    canonical_json_bytes,
)

SCHEMA_VERSION = "single-stage-fullspace-campaign-v1"
SCHEMA_VERSION_V2 = "single-stage-fullspace-campaign-v2"
SCHEMA_VERSION_V3 = "single-stage-fullspace-campaign-v3"
SQP_SAMPLE_SCHEMA_VERSION = "single-stage-fullspace-cfs-sqp1-sample-receipt-v1"
SQP_RESULT_SCHEMA_VERSION = "single-stage-fullspace-cfs-sqp1-result-v1"
SQP_MEMORY_SCHEMA_VERSION = "single-stage-fullspace-cfs-sqp1-gpu-memory-v1"
SQP_CERTIFICATE_ENVELOPE_SCHEMA_VERSION = (
    "single-stage-fullspace-cfs-sqp1-endpoint-certificate-envelope-v1"
)
SQP_DERIVATIVE_GATE_SCHEMA_VERSION = (
    "single-stage-fullspace-cfs-sqp1-derivative-gate-v1"
)
SQP_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION = (
    "single-stage-fullspace-cfs-sqp1-derivative-gate-receipt-v1"
)
SQP_CANARY_1_GATE_SCHEMA_VERSION = "single-stage-fullspace-cfs-sqp1-canary-1-gate-v1"
SQP_CANARY_10_GATE_SCHEMA_VERSION = "single-stage-fullspace-cfs-sqp1-canary-10-gate-v1"
SQP_PLAN_SHA256 = "e8ba9fe0513163038fd587427cc5199a00be954d9d9c3f9f51a79641136c9f4e"
SQP_BUDGET_SHA256 = "d51c87c55793ebed63acf01e87ef3837f5abdccb95e8c61827758a8961482082"
SQP_R2_PLAN_SHA256 = "3024b82b272dd72349c8c814b7b547dc6335357c1155b95cf60c1c5d252d0b78"
SQP_R2_BUDGET_SHA256 = SQP_BUDGET_SHA256
SQP_R1_PLAN_SHA256 = "1baf0a1c487e7f985ff5017e3762a19a5d4c69fdc27f140bdcd652cd384dce44"
SQP_R1_BUDGET_SHA256 = (
    "c8fab1b74588cd9256020d84133739c930c28b8a6741181f4f2f9c667dbb86c1"
)
SQP_R1_CONTRACT_SHA256 = (
    "ef1728fb1f402c0cebd05ab6adbb46833757338103dbf3c06a94ff2d9c603a81"
)
SQP_MAXIMUM_MEMORY_FRACTION = 0.8
SQP_WARM_SOLVE_MAX_SECONDS = 287.30421751597896
# Retained only to validate the immutable revision-1 campaign evidence.
SQP_KKT_RECIPROCAL_CONDITION_MINIMUM = 0.0010000001
SQP_KKT_FORWARD_ERROR_MAXIMUM = 1.0e-7
SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM = 1.0e-10
SQP_KKT_RELATIVE_RESIDUAL_MAXIMUM = 1.0e-10
SQP_SCHUR_RELATIVE_RESIDUAL_MAXIMUM = 1.0e-10
SQP_TERMINAL_STATUSES = frozenset(
    (
        "CONVERGED",
        "RANK_DEFICIENT_OR_UNSTABLE_KKT",
        "GLOBALIZATION_FAILED",
        "BFGS_UPDATE_FAILED",
        "OBJECTIVE_QUALITY_REJECTED",
        "ITERATION_LIMIT",
        "EVALUATION_LIMIT",
    )
)
_SQP_CONVERGENCE_TELEMETRY_KEYS = frozenset(
    (
        "merit",
        "penalty",
        "multiplier_update_infinity_norm",
        "bfgs_reset",
        "restoration_applied",
        "restoration_numerical_failures",
    )
)
V2_ROUTES = (*LEGACY_V1_ROUTES, FullSpaceRoute.CFS_SQP1)
RuntimeIdentity = _snapshot_contract.RuntimeIdentity
SourceIdentity = _snapshot_contract.SourceIdentity
load_canonical_json_bytes = _snapshot_contract.load_canonical_json_bytes


def _is_sqp_revision1_identity(
    plan_sha256: JsonValue,
    budget_sha256: JsonValue,
    contract_sha256: JsonValue | None = None,
) -> bool:
    """Identify immutable Revision 1 evidence without changing its meaning."""

    return (
        plan_sha256 == SQP_R1_PLAN_SHA256
        and budget_sha256 == SQP_R1_BUDGET_SHA256
        and (contract_sha256 is None or contract_sha256 == SQP_R1_CONTRACT_SHA256)
    )


def _is_sqp_revision2_identity(
    plan_sha256: JsonValue,
    budget_sha256: JsonValue,
    contract_sha256: JsonValue | None = None,
) -> bool:
    """Identify immutable Revision 2 evidence without changing its meaning."""

    return (
        plan_sha256 == SQP_R2_PLAN_SHA256
        and budget_sha256 == SQP_R2_BUDGET_SHA256
        and (contract_sha256 is None or contract_sha256 == contract_sha256_v2())
    )


class RunPhase(StrEnum):
    FIRST_EVAL = "first-eval"
    CANARY = "canary"
    COMPLETE = "complete"


class DeviceLane(StrEnum):
    RTX5090 = "rtx5090"
    A100 = "a100"


class CompleteSample(StrEnum):
    COLD = "cold"
    WARM_1 = "warm-1"
    WARM_2 = "warm-2"
    WARM_3 = "warm-3"


class SqpGate(StrEnum):
    DERIVATIVE = "CFS-SQP1-DERIVATIVE"
    CANARY_1 = "CFS-SQP1-CANARY-1"
    CANARY_10 = "CFS-SQP1-CANARY-10"


class RouteDisposition(StrEnum):
    EXECUTED = "EXECUTED"
    NOT_SELECTED_BY_GATE = "NOT_SELECTED_BY_GATE"


class CampaignDisposition(StrEnum):
    ENGINEERING_SPEED_SUCCESS = "ENGINEERING_SPEED_SUCCESS"
    BOUNDED_NEGATIVE = "BOUNDED_NEGATIVE"


class BaselineClassification(StrEnum):
    HISTORICAL_ENGINEERING_ONLY = "HISTORICAL_ENGINEERING_ONLY"
    CLAIM_COMPATIBLE = "CLAIM_COMPATIBLE"


@dataclass(frozen=True, slots=True)
class RunRequest:
    phase: RunPhase
    route: FullSpaceRoute
    device: DeviceLane
    steps: int | None
    sample: CompleteSample | None

    def validate(self) -> None:
        """Validate the legacy-v1 request surface."""

        if self.route not in LEGACY_V1_ROUTES:
            raise ValueError("campaign-v1 does not authorize CFS-SQP1")
        self._validate_legacy_shape()

    def validate_v2(self) -> None:
        """Validate an additive campaign-v2 legacy or SQP request."""

        if self.route is FullSpaceRoute.CFS_SQP1:
            if self.device is not DeviceLane.RTX5090:
                raise ValueError("CFS-SQP1 campaign-v2 requests require RTX 5090")
            if self.phase is RunPhase.FIRST_EVAL:
                if self.steps is not None or self.sample is not None:
                    raise ValueError("CFS-SQP1 first-eval forbids --steps and --sample")
            elif self.phase is RunPhase.CANARY:
                if self.steps not in (1, 10) or self.sample is not None:
                    raise ValueError(
                        "CFS-SQP1 canary requires --steps=1 or 10 and forbids --sample"
                    )
            elif self.phase is RunPhase.COMPLETE:
                if self.steps is not None or self.sample is None:
                    raise ValueError(
                        "CFS-SQP1 complete requires --sample and forbids --steps"
                    )
            else:
                raise ValueError("CFS-SQP1 campaign-v2 request phase is unsupported")
            return
        if self.route not in LEGACY_V1_ROUTES:
            raise ValueError("campaign-v2 request route is unsupported")
        self._validate_legacy_shape()

    def validate_v3(self) -> None:
        """Validate an additive campaign-v3 legacy, SQP, or FTR request."""

        if self.route is FullSpaceRoute.CFS_FTR1:
            if (
                self.phase is not RunPhase.CANARY
                or self.device is not DeviceLane.RTX5090
                or self.steps != 10
                or self.sample is not None
            ):
                raise ValueError(
                    "CFS-FTR1 Gate 2 requires exactly one RTX 5090 ten-step canary"
                )
            return
        self.validate_v2()

    def _validate_legacy_shape(self) -> None:
        if self.phase is RunPhase.FIRST_EVAL:
            if self.route is not FullSpaceRoute.CFS_P0 or self.steps is not None:
                raise ValueError("first-eval requires CFS-P0 and forbids --steps")
            if self.sample is not None:
                raise ValueError("first-eval forbids --sample")
        elif self.phase is RunPhase.CANARY:
            if self.steps not in (10, 100) or self.sample is not None:
                raise ValueError(
                    "canary requires --steps=10 or 100 and forbids --sample"
                )
        elif self.phase is RunPhase.COMPLETE:
            if self.route is FullSpaceRoute.CFS_P0:
                raise ValueError("complete forbids diagnostic route CFS-P0")
            if self.steps is not None or self.sample is None:
                raise ValueError("complete requires --sample and forbids --steps")


@dataclass(frozen=True, slots=True)
class GateEvidence:
    gate_id: str
    artifact: ArtifactRef


@dataclass(frozen=True, slots=True)
class RouteOutcome:
    route: FullSpaceRoute
    disposition: RouteDisposition
    terminal_status: str | None
    receipt: ArtifactRef | None
    upstream_gate: str | None
    gate_evidence: tuple[GateEvidence, ...]

    def validate(self) -> None:
        if self.disposition is RouteDisposition.EXECUTED:
            if self.terminal_status is None or self.receipt is None:
                raise ValueError("executed route requires terminal status and receipt")
            if self.upstream_gate is not None:
                raise ValueError(
                    "executed route cannot name an upstream selection gate"
                )
        else:
            if self.terminal_status is not None or self.receipt is not None:
                raise ValueError("unselected route cannot claim execution evidence")
            if self.upstream_gate is None or not self.gate_evidence:
                raise ValueError("unselected route requires gate name and raw evidence")


@dataclass(frozen=True, slots=True)
class CampaignReceipt:
    disposition: CampaignDisposition
    baseline_classification: BaselineClassification
    contract_sha256: str
    route_outcomes: tuple[RouteOutcome, ...]

    def validate(self) -> None:
        if tuple(outcome.route for outcome in self.route_outcomes) != LEGACY_V1_ROUTES:
            raise ValueError(
                "campaign requires exactly one outcome in canonical route order"
            )
        for outcome in self.route_outcomes:
            outcome.validate()


@dataclass(frozen=True, slots=True)
class SqpSampleReceipt:
    """One immutable SQP complete sample and its independent evidence chain."""

    request: RunRequest
    source_identity: SourceIdentity
    runtime_identity: RuntimeIdentity
    runtime_evidence: ArtifactRef
    bootstrap_artifact: ArtifactRef
    bootstrap_identity_sha256: str
    raw_result: ArtifactRef
    gpu_memory: ArtifactRef
    endpoint_certificate: ArtifactRef | None
    promotion_eligible: bool
    terminal_status: str
    synchronized_solve_seconds: float
    total_child_wall_seconds: float
    hot_h2d_transfers: int
    hot_d2h_transfers: int
    initial_h2d_transfers: int
    final_d2h_transfers: int
    peak_memory_bytes: int
    peak_memory_fraction: float

    def validate(self) -> None:
        self.request.validate_v2()
        if (
            self.request.route is not FullSpaceRoute.CFS_SQP1
            or self.request.phase is not RunPhase.COMPLETE
        ):
            raise ValueError("SQP sample receipt requires a complete CFS-SQP1 request")
        if self.promotion_eligible and self.endpoint_certificate is None:
            raise ValueError(
                "SQP promotion eligibility requires an endpoint certificate"
            )
        if len(self.bootstrap_identity_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.bootstrap_identity_sha256
        ):
            raise ValueError("SQP bootstrap identity must be a lowercase SHA-256")
        if self.terminal_status not in SQP_TERMINAL_STATUSES:
            raise ValueError("SQP sample receipt terminal status is not normalized")
        if self.promotion_eligible and self.terminal_status != "CONVERGED":
            raise ValueError("promoting SQP sample must have CONVERGED status")
        if (
            not math.isfinite(self.synchronized_solve_seconds)
            or self.synchronized_solve_seconds < 0.0
            or not math.isfinite(self.total_child_wall_seconds)
            or self.total_child_wall_seconds < self.synchronized_solve_seconds
        ):
            raise ValueError("SQP sample timing is invalid")
        transfer_counts = (
            self.hot_h2d_transfers,
            self.hot_d2h_transfers,
            self.initial_h2d_transfers,
            self.final_d2h_transfers,
        )
        if any(isinstance(count, bool) or count < 0 for count in transfer_counts):
            raise ValueError("SQP sample transfer counts must be nonnegative integers")
        if self.hot_h2d_transfers != 0 or self.hot_d2h_transfers != 0:
            raise ValueError("SQP sample hot transfers must be zero")
        if self.initial_h2d_transfers != 1 or self.final_d2h_transfers != 1:
            raise ValueError("SQP sample requires one initial H2D and one final D2H")
        if isinstance(self.peak_memory_bytes, bool) or self.peak_memory_bytes < 0:
            raise ValueError("SQP sample peak memory bytes are invalid")
        if (
            not math.isfinite(self.peak_memory_fraction)
            or self.peak_memory_fraction < 0.0
            or self.peak_memory_fraction >= SQP_MAXIMUM_MEMORY_FRACTION
        ):
            raise ValueError("SQP sample peak memory exceeds the frozen budget")


@dataclass(frozen=True, slots=True)
class CampaignReceiptV2:
    """Additive campaign envelope binding legacy routes and ordered SQP samples."""

    disposition: CampaignDisposition
    baseline_classification: BaselineClassification
    contract_sha256: str
    route_outcomes: tuple[RouteOutcome, ...]
    sqp_samples: tuple[ArtifactRef, ...]

    def validate(self) -> None:
        if tuple(outcome.route for outcome in self.route_outcomes) != V2_ROUTES:
            raise ValueError(
                "campaign-v2 requires exactly one outcome in canonical route order"
            )
        for outcome in self.route_outcomes:
            outcome.validate()
        sqp_outcome = self.route_outcomes[-1]
        if sqp_outcome.disposition is RouteDisposition.NOT_SELECTED_BY_GATE:
            if self.sqp_samples:
                raise ValueError("unselected CFS-SQP1 route cannot contain samples")
            if self.disposition is CampaignDisposition.ENGINEERING_SPEED_SUCCESS:
                raise ValueError("unselected CFS-SQP1 cannot claim speed success")
            return
        if len(self.sqp_samples) not in (1, 2, 3, 4):
            raise ValueError("executed CFS-SQP1 requires an exact sample prefix")
        if sqp_outcome.receipt != self.sqp_samples[0]:
            raise ValueError("CFS-SQP1 route receipt must identify its cold sample")
        if (
            self.disposition is CampaignDisposition.ENGINEERING_SPEED_SUCCESS
            and len(self.sqp_samples) != 4
        ):
            raise ValueError("campaign-v2 speed success requires cold and three warms")


@dataclass(frozen=True, slots=True)
class SqpGateResult:
    gate: SqpGate
    artifact: ArtifactRef
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SqpCollectedSample:
    artifact: ArtifactRef
    receipt: SqpSampleReceipt


@dataclass(frozen=True, slots=True)
class SqpCampaignV2Collector:
    """Immutable derivative-to-warm campaign-v2 evidence collector."""

    baseline_classification: BaselineClassification
    legacy_route_outcomes: tuple[RouteOutcome, ...]
    campaign_root: Path
    gates: tuple[SqpGateResult, ...] = ()
    samples: tuple[SqpCollectedSample, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        baseline_classification: BaselineClassification,
        legacy_route_outcomes: tuple[RouteOutcome, ...],
        campaign_root: Path,
    ) -> SqpCampaignV2Collector:
        root = campaign_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("SQP campaign root must be a directory")
        collector = cls(
            baseline_classification=baseline_classification,
            legacy_route_outcomes=legacy_route_outcomes,
            campaign_root=root,
        )
        collector._validate_legacy_outcomes()
        return collector

    def record_gate(
        self, gate: SqpGate, artifact: ArtifactRef
    ) -> SqpCampaignV2Collector:
        """Load immutable gate bytes and derive their status from semantics."""

        self._validate_legacy_outcomes()
        if self.samples or self._terminal_reason() is not None:
            raise ValueError(
                "SQP campaign cannot record a gate after execution stopped"
            )
        gate_order = (SqpGate.DERIVATIVE, SqpGate.CANARY_1, SqpGate.CANARY_10)
        if (
            len(self.gates) >= len(gate_order)
            or gate is not gate_order[len(self.gates)]
        ):
            raise ValueError("SQP gates must follow derivative, 1-step, 10-step order")
        if any(
            result.artifact.relative_path == artifact.relative_path
            or result.artifact.sha256 == artifact.sha256
            for result in self.gates
        ):
            raise ValueError("SQP gates may not reuse one evidence artifact")
        gate_result = load_sqp_gate_result(self.campaign_root, gate, artifact)
        return replace(
            self,
            gates=(*self.gates, gate_result),
        )

    def record_sample(
        self, artifact: ArtifactRef, receipt: SqpSampleReceipt
    ) -> SqpCampaignV2Collector:
        self._validate_legacy_outcomes()
        if self._terminal_reason() is not None:
            raise ValueError(
                "SQP campaign cannot continue after a failed or final result"
            )
        if len(self.gates) != 3 or not all(result.passed for result in self.gates):
            raise ValueError(
                "SQP samples require passed derivative, 1-step, and 10-step gates"
            )
        if artifact.schema_version != SQP_SAMPLE_SCHEMA_VERSION:
            raise ValueError("SQP sample reference has the wrong schema")
        receipt.validate()
        receipt_bytes = canonical_json_bytes(sqp_sample_receipt_payload(receipt))
        if (
            artifact.size_bytes != len(receipt_bytes)
            or artifact.sha256 != hashlib.sha256(receipt_bytes).hexdigest()
        ):
            raise ValueError("SQP sample reference differs from receipt bytes")
        sample_order = (
            CompleteSample.COLD,
            CompleteSample.WARM_1,
            CompleteSample.WARM_2,
            CompleteSample.WARM_3,
        )
        if len(self.samples) >= len(sample_order) or (
            receipt.request.sample is not sample_order[len(self.samples)]
        ):
            raise ValueError("SQP samples must follow cold then warm-1/2/3 order")
        if any(
            existing.artifact.relative_path == artifact.relative_path
            or existing.artifact.sha256 == artifact.sha256
            for existing in self.samples
        ):
            raise ValueError("SQP samples may not replace or duplicate evidence")
        if self.samples:
            cold = self.samples[0].receipt
            if not cold.promotion_eligible or cold.endpoint_certificate is None:
                raise ValueError("SQP warm samples require a certified cold sample")
            if (
                receipt.source_identity != cold.source_identity
                or receipt.request.device is not cold.request.device
                or _runtime_environment_identity(receipt.runtime_identity)
                != _runtime_environment_identity(cold.runtime_identity)
                or receipt.bootstrap_identity_sha256 != cold.bootstrap_identity_sha256
            ):
                raise ValueError("SQP sample provenance differs from the cold sample")
        return replace(
            self,
            samples=(*self.samples, SqpCollectedSample(artifact, receipt)),
        )

    def finalize_receipt(self) -> CampaignReceiptV2:
        self._validate_legacy_outcomes()
        terminal_reason = self._terminal_reason()
        if terminal_reason is None:
            raise ValueError("SQP campaign evidence prefix is not terminal")
        gate_evidence = tuple(
            GateEvidence(result.gate, result.artifact) for result in self.gates
        )
        failed_gate = next((result for result in self.gates if not result.passed), None)
        if failed_gate is not None:
            sqp_outcome = RouteOutcome(
                route=FullSpaceRoute.CFS_SQP1,
                disposition=RouteDisposition.NOT_SELECTED_BY_GATE,
                terminal_status=None,
                receipt=None,
                upstream_gate=failed_gate.gate,
                gate_evidence=gate_evidence,
            )
            disposition = CampaignDisposition.BOUNDED_NEGATIVE
        else:
            if not self.samples:
                raise ValueError("executed SQP campaign requires a cold sample")
            cold = self.samples[0]
            sqp_outcome = RouteOutcome(
                route=FullSpaceRoute.CFS_SQP1,
                disposition=RouteDisposition.EXECUTED,
                terminal_status=cold.receipt.terminal_status,
                receipt=cold.artifact,
                upstream_gate=None,
                gate_evidence=gate_evidence,
            )
            disposition = (
                CampaignDisposition.ENGINEERING_SPEED_SUCCESS
                if terminal_reason == "complete-success"
                else CampaignDisposition.BOUNDED_NEGATIVE
            )
        receipt = CampaignReceiptV2(
            disposition=disposition,
            baseline_classification=self.baseline_classification,
            contract_sha256=contract_sha256_v2(),
            route_outcomes=(*self.legacy_route_outcomes, sqp_outcome),
            sqp_samples=tuple(sample.artifact for sample in self.samples),
        )
        receipt.validate()
        return receipt

    def write(self, path: Path) -> ArtifactRef:
        output = path.absolute()
        if output != self.campaign_root / "campaign.json":
            raise ValueError("SQP campaign receipt must be campaign_root/campaign.json")
        payload = canonical_json_bytes(
            campaign_receipt_v2_payload(self.finalize_receipt())
        )
        with output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        output.chmod(0o444)
        return ArtifactRef(
            relative_path=output.relative_to(self.campaign_root).as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            schema_version=SCHEMA_VERSION_V2,
        )

    def write_and_validate(self, path: Path) -> ArtifactRef:
        """Exclusively write the terminal receipt, then run the full validator."""

        reference = self.write(path)
        from benchmarks.validate_single_stage_fullspace_campaign import (
            validate_campaign,
        )

        validate_campaign(self.campaign_root)
        return reference

    def _validate_legacy_outcomes(self) -> None:
        if tuple(outcome.route for outcome in self.legacy_route_outcomes) != (
            LEGACY_V1_ROUTES
        ):
            raise ValueError("collector requires exact legacy route outcomes")
        for outcome in self.legacy_route_outcomes:
            outcome.validate()

    def _terminal_reason(self) -> str | None:
        if self.gates and not self.gates[-1].passed:
            return "gate-failure"
        if not self.samples:
            return None
        latest = self.samples[-1].receipt
        if not latest.promotion_eligible:
            return "sample-failure"
        if (
            latest.request.sample is not CompleteSample.COLD
            and latest.synchronized_solve_seconds >= SQP_WARM_SOLVE_MAX_SECONDS
        ):
            return "speed-failure"
        if len(self.samples) == 4:
            return "complete-success"
        return None


def _runtime_environment_identity(runtime: RuntimeIdentity) -> tuple[str, ...]:
    """Identity fields that must remain invariant across fresh sample processes."""

    return (
        runtime.cwd,
        runtime.python_executable,
        runtime.python_version,
        runtime.jax_version,
        runtime.jaxlib_version,
        runtime.simsopt_module_path,
        runtime.simsopt_jax_module_path,
        runtime.native_extension_path,
        runtime.backend,
        runtime.device_uuid,
        runtime.driver_version,
        runtime.effective_environment_sha256,
    )


def write_derivative_gate_failed_campaign_receipt(
    *,
    campaign_root: Path,
    derivative_gate: ArtifactRef,
    baseline_classification: BaselineClassification,
    legacy_route_outcomes: tuple[RouteOutcome, ...],
) -> ArtifactRef:
    """Seal and independently validate one derivative-gate-failed campaign."""

    collector = SqpCampaignV2Collector.create(
        baseline_classification=baseline_classification,
        legacy_route_outcomes=legacy_route_outcomes,
        campaign_root=campaign_root,
    ).record_gate(SqpGate.DERIVATIVE, derivative_gate)
    if collector.gates[-1].passed:
        raise ValueError(
            "derivative gate passed; bounded-negative closure is forbidden"
        )
    return collector.write_and_validate(collector.campaign_root / "campaign.json")


def expect_mapping(value: JsonValue, *, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def expect_exact_keys(
    value: dict[str, JsonValue], expected: frozenset[str], *, context: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{context} keys do not match schema")


def expect_string(value: JsonValue, *, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def expect_integer(value: JsonValue, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def expect_number(value: JsonValue, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a number")
    return float(value)


_SQP_GATE_TOP_LEVEL_KEYS = frozenset(
    (
        "bootstrap_artifact",
        "budget_sha256",
        "contract_sha256",
        "plan_sha256",
        "request",
        "runtime_evidence",
        "schema_version",
        "source_identity",
        "terminal_status",
        "timing",
        "transfer_audit",
    )
)
_SQP_DERIVATIVE_GATE_KEYS = frozenset(
    (
        "bootstrap",
        "changed",
        "changed_optimizer_coordinates",
        "changed_physical_state",
        "failure_reasons",
        "gate_status",
        "kkt",
        "optimizer_steps_executed",
        "schema_version",
    )
)
_SQP_DERIVATIVE_STATE_KEYS = frozenset(
    (
        "all_finite",
        "atw",
        "av",
        "constraint_jacobian",
        "joint_vjp_rows",
        "joint_vjp_rows_dtype",
        "joint_vjp_rows_sha256",
        "joint_vjp_rows_shape",
        "numerical_rank",
        "objective_gradient",
        "physical_objective",
        "rank_cutoff",
        "rank_relative_threshold",
        "scaled_constraints",
        "scaled_constraints_sha256",
        "sigma_maximum",
        "sigma_minimum",
        "singular_values",
        "transpose_absolute_error",
        "transpose_lhs",
        "transpose_relative_error",
        "transpose_rhs",
    )
)
_SQP_DERIVATIVE_KKT_KEYS = frozenset(
    (
        "all_finite",
        "bfgs_cholesky_relative_pivot",
        "certified_relative_error_bound",
        "kkt_relative_residual",
        "multiplier_step",
        "primal_step",
        "reconstructed_residual_inf",
        "reconstructed_residual_two",
        "regularization_candidates_tested",
        "rho_k",
        "schur_cholesky_relative_pivot",
        "schur_relative_residual",
        "selected_regularization",
        "valid",
        "zeta_2",
    )
)
_SQP_GATE_RECEIPT_COMMON_KEYS = frozenset(
    (
        "bootstrap_artifact",
        "budget_sha256",
        "contract_sha256",
        "failure_reasons",
        "gate_status",
        "gpu_memory",
        "plan_sha256",
        "raw_result",
        "request",
        "runtime_evidence",
        "schema_version",
        "source_identity",
    )
)
_SQP_CANARY_GATE_KEYS = frozenset(
    (
        "expected_iterations",
        "failure_reasons",
        "gate_status",
        "initial_state",
        "schema_version",
    )
)
_SQP_CANARY_10_EXTRA_KEYS = frozenset(
    (
        "final_physical_objective",
        "final_raw_kkt_stationarity_inf",
        "final_scaled_feasibility_inf",
        "initial_physical_objective",
        "initial_raw_kkt_stationarity_inf",
        "initial_scaled_feasibility_inf",
        "projected_100_iteration_s",
        "projection_formula",
        "synchronized_solve_seconds",
    )
)


def _expect_boolean(value: JsonValue, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a Boolean")
    return value


def _expect_finite_number(value: JsonValue, *, context: str) -> float:
    result = expect_number(value, context=context)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def validate_sqp_convergence_telemetry(
    optimizer: dict[str, JsonValue], *, accepted_length: int
) -> None:
    """Validate the Revision 3 fixed-shape diagnostic prefix."""

    restoration_failures = optimizer.get("restoration_numerical_failures")
    if (
        isinstance(restoration_failures, bool)
        or not isinstance(restoration_failures, int)
        or restoration_failures < 0
    ):
        raise ValueError("optimizer.restoration_numerical_failures must be nonnegative")
    telemetry = expect_mapping(
        optimizer.get("convergence_telemetry"),
        context="optimizer.convergence_telemetry",
    )
    expect_exact_keys(
        telemetry,
        _SQP_CONVERGENCE_TELEMETRY_KEYS,
        context="optimizer.convergence_telemetry",
    )
    if telemetry["restoration_numerical_failures"] != restoration_failures:
        raise ValueError("SQP restoration failure telemetry is inconsistent")
    for key in (
        "merit",
        "penalty",
        "multiplier_update_infinity_norm",
        "bfgs_reset",
        "restoration_applied",
    ):
        values = telemetry[key]
        if not isinstance(values, list) or len(values) != accepted_length:
            raise ValueError(
                f"optimizer.convergence_telemetry.{key} must match accepted history length"
            )
        for index, item in enumerate(values):
            context = f"optimizer.convergence_telemetry.{key}[{index}]"
            if key in {"bfgs_reset", "restoration_applied"}:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item not in (0, 1)
                ):
                    raise ValueError(f"{context} must be a zero-or-one indicator")
            else:
                value = _expect_finite_number(item, context=context)
                if value < 0.0 or (key == "penalty" and value == 0.0):
                    raise ValueError(f"{context} must be positive telemetry")
    if optimizer.get("all_finite") is True and restoration_failures != 0:
        raise ValueError("finite SQP result reports restoration numerical failures")


def _expect_failure_reasons(value: JsonValue) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("SQP gate failure_reasons must be an array")
    reasons = tuple(expect_string(item, context="failure reason") for item in value)
    if len(set(reasons)) != len(reasons) or any(not reason for reason in reasons):
        raise ValueError("SQP gate failure reasons must be nonempty and unique")
    return reasons


def _validate_derivative_state(
    value: JsonValue, *, name: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    state = expect_mapping(value, context=f"{name} derivative state")
    expect_exact_keys(
        state, _SQP_DERIVATIVE_STATE_KEYS, context=f"{name} derivative state"
    )
    reasons: list[str] = []
    prefix = name.upper()
    if not _expect_boolean(state["all_finite"], context=f"{name}.all_finite"):
        reasons.append(f"{prefix}_NONFINITE")
    shape = state["joint_vjp_rows_shape"]
    if shape != [256, 716]:
        reasons.append(f"{prefix}_ROW_SHAPE")
    if state["joint_vjp_rows_dtype"] != "float64":
        reasons.append(f"{prefix}_DTYPE")
    if expect_integer(state["numerical_rank"], context=f"{name}.numerical_rank") != 255:
        reasons.append(f"{prefix}_RANK")
    rows = state["joint_vjp_rows"]
    constraints = state["scaled_constraints"]
    if (
        hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
        != state["joint_vjp_rows_sha256"]
    ):
        raise ValueError(f"{name} joint VJP rows digest differs")
    if (
        hashlib.sha256(canonical_json_bytes(constraints)).hexdigest()
        != state["scaled_constraints_sha256"]
    ):
        raise ValueError(f"{name} scaled constraints digest differs")
    joint_rows = np.asarray(rows, dtype=np.float64)
    objective_gradient = np.asarray(state["objective_gradient"], dtype=np.float64)
    constraint_jacobian = np.asarray(state["constraint_jacobian"], dtype=np.float64)
    scaled_constraints = np.asarray(constraints, dtype=np.float64)
    if joint_rows.shape != (256, 716) or objective_gradient.shape != (716,):
        raise ValueError(f"{name} retained derivative shapes differ")
    if constraint_jacobian.shape != (255, 716) or scaled_constraints.shape != (255,):
        raise ValueError(f"{name} retained constraint shapes differ")
    if not np.array_equal(joint_rows[0], objective_gradient) or not np.array_equal(
        joint_rows[1:], constraint_jacobian
    ):
        raise ValueError(f"{name} joint rows do not equal concatenated g and A")
    retained_arrays = (
        joint_rows,
        objective_gradient,
        constraint_jacobian,
        scaled_constraints,
    )
    if not all(np.all(np.isfinite(array)) for array in retained_arrays):
        raise ValueError(f"{name} retained derivative arrays are nonfinite")
    singular_values = np.linalg.svd(constraint_jacobian, compute_uv=False)
    recorded_singular_values = np.asarray(state["singular_values"], dtype=np.float64)
    if recorded_singular_values.shape != (255,) or not np.allclose(
        recorded_singular_values, singular_values, rtol=5.0e-13, atol=0.0
    ):
        raise ValueError(f"{name} singular values differ from retained A")
    sigma_maximum = float(singular_values[0])
    rank_relative_threshold = _expect_finite_number(
        state["rank_relative_threshold"], context=f"{name}.rank_relative_threshold"
    )
    rank_cutoff = rank_relative_threshold * sigma_maximum
    numerical_rank = int(np.sum(singular_values > rank_cutoff))
    for key, computed in (
        ("sigma_maximum", sigma_maximum),
        ("sigma_minimum", float(singular_values[-1])),
        ("rank_cutoff", rank_cutoff),
    ):
        if not np.isclose(
            _expect_finite_number(state[key], context=f"{name}.{key}"),
            computed,
            rtol=5.0e-13,
            atol=0.0,
        ):
            raise ValueError(f"{name} {key} differs from retained A")
    if (
        expect_integer(state["numerical_rank"], context=f"{name}.numerical_rank")
        != numerical_rank
    ):
        raise ValueError(f"{name} numerical rank differs from retained A")
    direction = np.linspace(-0.75, 1.0, 716, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    cotangent = np.linspace(0.5, -1.25, 255, dtype=np.float64)
    cotangent /= np.linalg.norm(cotangent)
    av = constraint_jacobian @ direction
    atw = constraint_jacobian.T @ cotangent
    recorded_av = np.asarray(state["av"], dtype=np.float64)
    recorded_atw = np.asarray(state["atw"], dtype=np.float64)
    if not np.allclose(recorded_av, av, rtol=5.0e-13, atol=1.0e-15) or not np.allclose(
        recorded_atw, atw, rtol=5.0e-13, atol=1.0e-15
    ):
        raise ValueError(f"{name} retained Av or A^T w differs")
    lhs = float(np.vdot(cotangent, av))
    rhs = float(np.vdot(atw, direction))
    absolute_error = abs(lhs - rhs)
    denominator = max(abs(lhs), abs(rhs), np.finfo(np.float64).tiny)
    relative_error = absolute_error / denominator
    for key, computed in (
        ("transpose_lhs", lhs),
        ("transpose_rhs", rhs),
        ("transpose_absolute_error", absolute_error),
        ("transpose_relative_error", relative_error),
    ):
        if not np.isclose(
            _expect_finite_number(state[key], context=f"{name}.{key}"),
            computed,
            rtol=5.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError(f"{name} {key} differs from retained A")
    if relative_error > 1.0e-9 and absolute_error > 1.0e-10:
        reasons.append(f"{prefix}_TRANSPOSE_IDENTITY")
    return tuple(reasons), objective_gradient, constraint_jacobian, scaled_constraints


def _validate_derivative_gate(value: JsonValue) -> SqpGateResult:
    raw = expect_mapping(value, context="SQP derivative gate artifact")
    expect_exact_keys(
        raw,
        _SQP_GATE_TOP_LEVEL_KEYS | {"derivative_kkt_gate"},
        context="SQP derivative gate artifact",
    )
    if raw["schema_version"] != SQP_DERIVATIVE_GATE_SCHEMA_VERSION:
        raise ValueError("SQP derivative gate schema mismatch")
    revision1 = _is_sqp_revision1_identity(
        raw["plan_sha256"], raw["budget_sha256"], raw["contract_sha256"]
    )
    revision2 = _is_sqp_revision2_identity(
        raw["plan_sha256"], raw["budget_sha256"], raw["contract_sha256"]
    )
    if not (
        (
            raw["contract_sha256"] == contract_sha256_v2()
            and raw["plan_sha256"] == SQP_PLAN_SHA256
            and raw["budget_sha256"] == SQP_BUDGET_SHA256
        )
        or revision1
        or revision2
    ):
        raise ValueError("SQP derivative gate plan or budget digest differs")
    request = run_request_v2_from_payload(raw["request"])
    expected_request = RunRequest(
        RunPhase.FIRST_EVAL,
        FullSpaceRoute.CFS_SQP1,
        DeviceLane.RTX5090,
        None,
        None,
    )
    if request != expected_request:
        raise ValueError("SQP derivative gate request differs from the frozen request")
    if raw["terminal_status"] != "DERIVATIVE_KKT_GATE_COMPLETED":
        raise ValueError("SQP derivative gate terminal status differs")
    transfer = expect_mapping(raw["transfer_audit"], context="SQP derivative transfers")
    for key, expected in (
        ("hot_h2d_calls", 0),
        ("hot_d2h_calls", 0),
        ("initial_h2d_calls", 1),
        ("final_d2h_calls", 1),
    ):
        if (
            expect_integer(transfer.get(key), context=f"transfer_audit.{key}")
            != expected
        ):
            raise ValueError(
                f"SQP derivative gate {key} differs from the frozen boundary"
            )
    gate = expect_mapping(raw["derivative_kkt_gate"], context="derivative_kkt_gate")
    expect_exact_keys(gate, _SQP_DERIVATIVE_GATE_KEYS, context="derivative_kkt_gate")
    if gate["schema_version"] != SQP_DERIVATIVE_GATE_SCHEMA_VERSION:
        raise ValueError("nested SQP derivative gate schema mismatch")
    if (
        expect_integer(
            gate["optimizer_steps_executed"], context="optimizer_steps_executed"
        )
        != 0
    ):
        raise ValueError("SQP derivative gate executed optimizer steps")
    bootstrap_state = _validate_derivative_state(gate["bootstrap"], name="bootstrap")
    changed_state = _validate_derivative_state(gate["changed"], name="changed")
    reasons = [*bootstrap_state[0], *changed_state[0]]
    kkt = expect_mapping(gate["kkt"], context="derivative KKT result")
    expect_exact_keys(kkt, _SQP_DERIVATIVE_KKT_KEYS, context="derivative KKT result")
    objective_gradient, constraint_jacobian, scaled_constraints = changed_state[1:]
    selected_regularization = _expect_finite_number(
        kkt["selected_regularization"], context="kkt.selected_regularization"
    )
    if selected_regularization not in (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
        raise ValueError("derivative KKT regularization is outside the frozen ladder")
    primal_step = np.asarray(kkt["primal_step"], dtype=np.float64)
    multiplier_step = np.asarray(kkt["multiplier_step"], dtype=np.float64)
    if primal_step.shape != (716,) or multiplier_step.shape != (255,):
        raise ValueError("derivative KKT step shapes differ")
    regularized_bfgs = np.eye(716, dtype=np.float64) * (1.0 + selected_regularization)
    kkt_matrix = np.block(
        [
            [regularized_bfgs, constraint_jacobian.T],
            [constraint_jacobian, np.zeros((255, 255), dtype=np.float64)],
        ]
    )
    right_hand_side = -np.concatenate((objective_gradient, scaled_constraints))
    solution = np.concatenate((primal_step, multiplier_step))
    residual = kkt_matrix @ solution - right_hand_side
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    residual_two = float(np.linalg.norm(residual, ord=2))
    matrix_norm_inf = float(np.linalg.norm(kkt_matrix, ord=np.inf))
    solution_norm_inf = float(np.linalg.norm(solution, ord=np.inf))
    rhs_norm_inf = float(np.linalg.norm(right_hand_side, ord=np.inf))
    kkt_relative_residual = residual_inf / max(
        1.0, matrix_norm_inf * solution_norm_inf + rhs_norm_inf
    )
    eigenvalue_magnitudes = np.abs(np.linalg.eigvalsh(kkt_matrix))
    sigma_maximum = float(np.max(eigenvalue_magnitudes))
    rho_k = (
        float(np.min(eigenvalue_magnitudes)) / sigma_maximum
        if sigma_maximum > 0.0
        else 0.0
    )
    solution_two = float(np.linalg.norm(solution, ord=2))
    zeta_denominator = sigma_maximum * solution_two
    zeta_2 = (
        residual_two / zeta_denominator
        if zeta_denominator > 0.0
        else 0.0
        if residual_two == 0.0
        else math.inf
    )
    certified_error_bound = zeta_2 / (rho_k - zeta_2) if rho_k > zeta_2 else math.inf
    inverse_scale = 1.0 / (1.0 + selected_regularization)
    schur = constraint_jacobian @ (constraint_jacobian.T * inverse_scale)
    schur_rhs = scaled_constraints - constraint_jacobian @ (
        objective_gradient * inverse_scale
    )
    schur_residual_vector = schur @ multiplier_step - schur_rhs
    schur_relative_residual = float(
        np.linalg.norm(schur_residual_vector, ord=np.inf)
    ) / max(
        1.0,
        float(np.linalg.norm(schur, ord=np.inf))
        * float(np.linalg.norm(multiplier_step, ord=np.inf))
        + float(np.linalg.norm(schur_rhs, ord=np.inf)),
    )
    recomputed_kkt = (
        ("reconstructed_residual_inf", residual_inf),
        ("reconstructed_residual_two", residual_two),
        ("kkt_relative_residual", kkt_relative_residual),
        ("rho_k", rho_k),
        ("zeta_2", zeta_2),
        ("certified_relative_error_bound", certified_error_bound),
        ("schur_relative_residual", schur_relative_residual),
    )
    for key, computed in recomputed_kkt:
        recorded = kkt[key]
        if not math.isfinite(computed):
            if recorded is not None:
                raise ValueError(
                    f"derivative KKT {key} must be null for nonfinite evidence"
                )
            continue
        recorded_number = _expect_finite_number(recorded, context=f"kkt.{key}")
        absolute_tolerance = (
            1.0e-12 if key == "certified_relative_error_bound" else 1.0e-14
        )
        if not np.isclose(
            recorded_number, computed, rtol=2.0e-9, atol=absolute_tolerance
        ):
            raise ValueError(f"derivative KKT {key} differs from retained arrays")
    computed_finite = all(
        np.all(np.isfinite(array))
        for array in (kkt_matrix, solution, residual, eigenvalue_magnitudes)
    ) and all(math.isfinite(value) for _, value in recomputed_kkt)
    computed_valid = (
        computed_finite
        and kkt_relative_residual <= SQP_KKT_RELATIVE_RESIDUAL_MAXIMUM
        and (
            rho_k > SQP_KKT_RECIPROCAL_CONDITION_MINIMUM
            if revision1
            else rho_k > zeta_2
        )
        and zeta_2 <= SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM
        and certified_error_bound < SQP_KKT_FORWARD_ERROR_MAXIMUM
        and schur_relative_residual <= SQP_SCHUR_RELATIVE_RESIDUAL_MAXIMUM
    )
    if _expect_boolean(kkt["all_finite"], context="kkt.all_finite") != computed_finite:
        raise ValueError("derivative KKT finite status differs from retained arrays")
    if _expect_boolean(kkt["valid"], context="kkt.valid") != computed_valid:
        raise ValueError("derivative KKT valid status differs from retained arrays")
    if not _expect_boolean(kkt["valid"], context="kkt.valid"):
        reasons.append("KKT_INVALID")
    if not _expect_boolean(kkt["all_finite"], context="kkt.all_finite"):
        reasons.append("KKT_NONFINITE")
    if not revision1 and (
        not math.isfinite(rho_k) or not math.isfinite(zeta_2) or rho_k <= zeta_2
    ):
        reasons.append("KKT_RHO")
    thresholds = (
        (
            "rho_k",
            SQP_KKT_RECIPROCAL_CONDITION_MINIMUM,
            lambda actual, limit: actual > limit,
            "KKT_RHO",
        )
        if revision1
        else None,
        (
            "zeta_2",
            SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM,
            lambda actual, limit: actual <= limit,
            "KKT_ZETA",
        ),
        (
            "kkt_relative_residual",
            SQP_KKT_RELATIVE_RESIDUAL_MAXIMUM,
            lambda actual, limit: actual <= limit,
            "KKT_RESIDUAL",
        ),
        (
            "schur_relative_residual",
            SQP_SCHUR_RELATIVE_RESIDUAL_MAXIMUM,
            lambda actual, limit: actual <= limit,
            "SCHUR_RESIDUAL",
        ),
        (
            "certified_relative_error_bound",
            SQP_KKT_FORWARD_ERROR_MAXIMUM,
            lambda actual, limit: actual < limit,
            "KKT_ERROR_BOUND",
        ),
    )
    for threshold in thresholds:
        if threshold is None:
            continue
        key, limit, predicate, reason = threshold
        computed = dict(recomputed_kkt)[key]
        if not math.isfinite(computed):
            reasons.append(reason)
            continue
        actual = _expect_finite_number(kkt[key], context=f"kkt.{key}")
        if not predicate(actual, limit):
            reasons.append(reason)
    recorded_reasons = _expect_failure_reasons(gate["failure_reasons"])
    if recorded_reasons != tuple(reasons):
        raise ValueError("SQP derivative gate failure reasons differ from evidence")
    expected_status = "PASS" if not reasons else "FAIL"
    if gate["gate_status"] != expected_status:
        raise ValueError("SQP derivative gate status differs from evidence")
    return SqpGateResult(
        SqpGate.DERIVATIVE,
        ArtifactRef("", "", 0, SQP_DERIVATIVE_GATE_SCHEMA_VERSION),
        not reasons,
        tuple(reasons),
    )


def _canary_failure_reasons(
    request: RunRequest,
    raw: dict[str, JsonValue],
    memory: dict[str, JsonValue],
) -> tuple[tuple[str, ...], dict[str, JsonValue]]:
    optimizer = expect_mapping(raw.get("optimizer_result"), context="canary optimizer")
    endpoint = expect_mapping(raw.get("endpoint"), context="canary endpoint")
    transfers = expect_mapping(raw.get("transfer_audit"), context="canary transfers")
    timing = expect_mapping(raw.get("timing"), context="canary timing")
    assert request.steps in (1, 10)
    reasons: list[str] = []
    if (
        optimizer.get("all_finite") is not True
        or optimizer.get("all_accepted_states_finite") is not True
    ):
        reasons.append("OPTIMIZER_NOT_FINITE")
    if endpoint.get("all_finite") is not True:
        reasons.append("ENDPOINT_NOT_FINITE")
    if optimizer.get("fatal") is not False or optimizer.get("failed") is True:
        reasons.append("FATAL_STATUS")
    if optimizer.get("iterations") != request.steps:
        reasons.append("ITERATION_COUNT")
    history = optimizer.get("history")
    if not isinstance(history, dict) or history.get("accepted_length") != request.steps:
        reasons.append("ACCEPTED_HISTORY_LENGTH")
    if request.steps == 10 and raw.get("plan_sha256") == SQP_PLAN_SHA256:
        validate_sqp_convergence_telemetry(
            optimizer,
            accepted_length=request.steps,
        )
    if transfers.get("hot_h2d_calls") != 0 or transfers.get("hot_d2h_calls") != 0:
        reasons.append("HOT_TRANSFER")
    if transfers.get("initial_h2d_calls") != 1 or transfers.get("final_d2h_calls") != 1:
        reasons.append("BOUNDARY_TRANSFER")
    peak_fraction = memory.get("peak_memory_fraction")
    if (
        isinstance(peak_fraction, bool)
        or not isinstance(peak_fraction, (int, float))
        or not math.isfinite(float(peak_fraction))
        or float(peak_fraction) >= SQP_MAXIMUM_MEMORY_FRACTION
    ):
        reasons.append("MEMORY_BUDGET")
    kkt_values = (
        optimizer.get("final_kkt_reciprocal_condition"),
        optimizer.get("final_kkt_solution_scaled_residual"),
        optimizer.get("final_kkt_relative_residual"),
        optimizer.get("final_schur_relative_residual"),
    )
    revision1 = _is_sqp_revision1_identity(
        raw.get("plan_sha256"),
        raw.get("budget_sha256"),
        raw.get("contract_sha256"),
    )
    if not (
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in kkt_values
        )
        and (
            float(kkt_values[0]) > SQP_KKT_RECIPROCAL_CONDITION_MINIMUM
            if revision1
            else float(kkt_values[0]) > float(kkt_values[1])
        )
        and float(kkt_values[1]) <= SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM
        and (
            revision1
            or float(kkt_values[1]) / (float(kkt_values[0]) - float(kkt_values[1]))
            < SQP_KKT_FORWARD_ERROR_MAXIMUM
        )
        and float(kkt_values[2]) <= 1.0e-10
        and float(kkt_values[3]) <= 1.0e-10
    ):
        reasons.append("KKT_CERTIFICATE")
    detail: dict[str, JsonValue] = {
        "schema_version": (
            SQP_CANARY_1_GATE_SCHEMA_VERSION
            if request.steps == 1
            else SQP_CANARY_10_GATE_SCHEMA_VERSION
        ),
        "gate_status": "PASS" if not reasons else "FAIL",
        "failure_reasons": list(reasons),
        "expected_iterations": request.steps,
        "initial_state": "changed" if request.steps == 1 else "bootstrap",
    }
    if request.steps == 10:
        synchronized_seconds = _expect_finite_number(
            timing.get("synchronized_solve_seconds"),
            context="canary synchronized_solve_seconds",
        )
        progress = (
            optimizer.get("initial_physical_objective"),
            endpoint.get("physical_objective"),
            optimizer.get("initial_scaled_constraint_infinity_norm"),
            endpoint.get("scaled_constraint_infinity_norm"),
            optimizer.get("initial_raw_kkt_stationarity_infinity_norm"),
            endpoint.get("raw_kkt_stationarity_infinity_norm"),
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in progress
        ):
            reasons.append("PROGRESS_EVIDENCE_NONFINITE")
        else:
            if not float(progress[1]) < float(progress[0]):
                reasons.append("OBJECTIVE_NOT_DECREASED")
            if not (
                float(progress[3]) <= 1.0e-10 or float(progress[3]) < float(progress[2])
            ):
                reasons.append("FEASIBILITY_NOT_MAINTAINED_OR_DECREASED")
            if not float(progress[5]) < float(progress[4]):
                reasons.append("RAW_KKT_NOT_DECREASED")
        projected_seconds = 10.0 * synchronized_seconds
        if projected_seconds >= SQP_WARM_SOLVE_MAX_SECONDS:
            reasons.append("PROJECTED_TIME_EXCEEDED")
        detail.update(
            {
                "initial_physical_objective": progress[0],
                "final_physical_objective": progress[1],
                "initial_scaled_feasibility_inf": progress[2],
                "final_scaled_feasibility_inf": progress[3],
                "initial_raw_kkt_stationarity_inf": progress[4],
                "final_raw_kkt_stationarity_inf": progress[5],
                "projected_100_iteration_s": projected_seconds,
                "projection_formula": "10 * synchronized_solve_seconds",
                "synchronized_solve_seconds": synchronized_seconds,
            }
        )
        detail["failure_reasons"] = list(reasons)
        detail["gate_status"] = "PASS" if not reasons else "FAIL"
    return tuple(reasons), detail


def load_sqp_gate_result(
    campaign_root: Path, gate: SqpGate, artifact: ArtifactRef
) -> SqpGateResult:
    """Resolve one campaign-local gate artifact and derive its immutable verdict."""

    root = campaign_root.resolve(strict=True)
    path = artifact.resolve_and_validate(root)
    receipt = expect_mapping(
        load_canonical_json_bytes(path.read_bytes()), context="SQP gate receipt"
    )
    detail_key = "derivative_kkt_gate" if gate is SqpGate.DERIVATIVE else "canary_gate"
    expect_exact_keys(
        receipt,
        _SQP_GATE_RECEIPT_COMMON_KEYS | {detail_key},
        context="SQP gate receipt",
    )
    expected_schema = {
        SqpGate.DERIVATIVE: SQP_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION,
        SqpGate.CANARY_1: SQP_CANARY_1_GATE_SCHEMA_VERSION,
        SqpGate.CANARY_10: SQP_CANARY_10_GATE_SCHEMA_VERSION,
    }[gate]
    if (
        receipt["schema_version"] != expected_schema
        or artifact.schema_version != expected_schema
    ):
        raise ValueError("SQP gate receipt schema differs from the requested gate")
    if not (
        (
            receipt["contract_sha256"] == contract_sha256_v2()
            and receipt["plan_sha256"] == SQP_PLAN_SHA256
            and receipt["budget_sha256"] == SQP_BUDGET_SHA256
        )
        or _is_sqp_revision1_identity(
            receipt["plan_sha256"],
            receipt["budget_sha256"],
            receipt["contract_sha256"],
        )
        or _is_sqp_revision2_identity(
            receipt["plan_sha256"],
            receipt["budget_sha256"],
            receipt["contract_sha256"],
        )
    ):
        raise ValueError("SQP gate receipt plan or budget digest differs")
    request = run_request_v2_from_payload(receipt["request"])
    expected_request = {
        SqpGate.DERIVATIVE: RunRequest(
            RunPhase.FIRST_EVAL, FullSpaceRoute.CFS_SQP1, DeviceLane.RTX5090, None, None
        ),
        SqpGate.CANARY_1: RunRequest(
            RunPhase.CANARY, FullSpaceRoute.CFS_SQP1, DeviceLane.RTX5090, 1, None
        ),
        SqpGate.CANARY_10: RunRequest(
            RunPhase.CANARY, FullSpaceRoute.CFS_SQP1, DeviceLane.RTX5090, 10, None
        ),
    }[gate]
    if request != expected_request:
        raise ValueError("SQP gate receipt request differs from the frozen request")
    raw_ref = artifact_ref_from_payload(receipt["raw_result"])
    memory_ref = artifact_ref_from_payload(receipt["gpu_memory"])
    raw = expect_mapping(
        load_canonical_json_bytes(raw_ref.resolve_and_validate(root).read_bytes()),
        context="SQP gate raw result",
    )
    memory = expect_mapping(
        load_canonical_json_bytes(memory_ref.resolve_and_validate(root).read_bytes()),
        context="SQP gate GPU memory",
    )
    for identity_key in ("contract_sha256", "plan_sha256", "budget_sha256"):
        if raw.get(identity_key) != receipt[identity_key]:
            raise ValueError(f"SQP gate receipt and raw {identity_key} identity differ")
    runtime_ref = artifact_ref_from_payload(receipt["runtime_evidence"])
    bootstrap_ref = artifact_ref_from_payload(receipt["bootstrap_artifact"])
    if raw.get("request") != receipt["request"]:
        raise ValueError("SQP gate receipt and raw request differ")
    if raw.get("source_identity") != receipt["source_identity"]:
        raise ValueError("SQP gate receipt and raw source identity differ")
    if raw.get("runtime_evidence") != receipt["runtime_evidence"]:
        raise ValueError("SQP gate receipt and raw runtime evidence differ")
    if raw.get("bootstrap_artifact") != receipt["bootstrap_artifact"]:
        raise ValueError("SQP gate receipt and raw bootstrap differ")
    source = _snapshot_contract.source_identity_from_payload(receipt["source_identity"])
    manifest_path = source.snapshot_manifest.resolve_and_validate(root)
    evidence = _snapshot_contract.validate_runtime_evidence(
        runtime_ref.resolve_and_validate(root),
        snapshot_root=manifest_path.parent,
        campaign_root=root,
    )
    if evidence.source_identity != source:
        raise ValueError("SQP gate source identity differs from runtime evidence")
    from benchmarks.single_stage_fullspace_bootstrap import validate_bootstrap_artifact

    bootstrap = validate_bootstrap_artifact(
        bootstrap_ref.resolve_and_validate(root),
        campaign_root=root,
        snapshot_root=manifest_path.parent,
    )
    if bootstrap.get("runtime_evidence") != receipt["runtime_evidence"]:
        raise ValueError("SQP gate bootstrap and runtime evidence differ")
    if memory_ref.schema_version != SQP_MEMORY_SCHEMA_VERSION:
        raise ValueError("SQP gate memory artifact schema differs")
    if memory.get("schema_version") != SQP_MEMORY_SCHEMA_VERSION:
        raise ValueError("SQP gate memory payload schema differs")
    if gate is SqpGate.DERIVATIVE:
        raw_result = _validate_derivative_gate(raw)
        detail = expect_mapping(receipt[detail_key], context=detail_key)
        if detail != raw.get("derivative_kkt_gate"):
            raise ValueError(
                "SQP derivative gate receipt detail differs from raw bytes"
            )
        reasons = raw_result.failure_reasons
        passed = raw_result.passed
    else:
        if raw.get("schema_version") != SQP_RESULT_SCHEMA_VERSION:
            raise ValueError("SQP canary raw result schema differs")
        reasons, expected_detail = _canary_failure_reasons(request, raw, memory)
        detail = expect_mapping(receipt[detail_key], context=detail_key)
        expected_detail_keys = _SQP_CANARY_GATE_KEYS | (
            _SQP_CANARY_10_EXTRA_KEYS if gate is SqpGate.CANARY_10 else frozenset()
        )
        expect_exact_keys(detail, expected_detail_keys, context=detail_key)
        if detail != expected_detail:
            raise ValueError("SQP canary gate detail differs from raw evidence")
        passed = not reasons
    outer_reasons = _expect_failure_reasons(receipt["failure_reasons"])
    if outer_reasons != reasons:
        raise ValueError("SQP gate receipt failure reasons differ from evidence")
    if receipt["gate_status"] != ("PASS" if passed else "FAIL"):
        raise ValueError("SQP gate receipt status differs from evidence")
    return SqpGateResult(gate, artifact, passed, reasons)


def artifact_ref_from_payload(value: JsonValue) -> ArtifactRef:
    mapping = expect_mapping(value, context="artifact")
    expect_exact_keys(
        mapping,
        frozenset(("relative_path", "sha256", "size_bytes", "schema_version")),
        context="artifact",
    )
    return ArtifactRef(
        relative_path=expect_string(mapping["relative_path"], context="relative_path"),
        sha256=expect_string(mapping["sha256"], context="sha256"),
        size_bytes=expect_integer(mapping["size_bytes"], context="size_bytes"),
        schema_version=expect_string(
            mapping["schema_version"], context="schema_version"
        ),
    )


def gate_evidence_from_payload(value: JsonValue) -> GateEvidence:
    mapping = expect_mapping(value, context="gate evidence")
    expect_exact_keys(
        mapping, frozenset(("gate_id", "artifact")), context="gate evidence"
    )
    return GateEvidence(
        gate_id=expect_string(mapping["gate_id"], context="gate_id"),
        artifact=artifact_ref_from_payload(mapping["artifact"]),
    )


def route_outcome_from_payload(value: JsonValue) -> RouteOutcome:
    mapping = expect_mapping(value, context="route outcome")
    expect_exact_keys(
        mapping,
        frozenset(
            (
                "route",
                "disposition",
                "terminal_status",
                "receipt",
                "upstream_gate",
                "gate_evidence",
            )
        ),
        context="route outcome",
    )
    terminal_value = mapping["terminal_status"]
    receipt_value = mapping["receipt"]
    upstream_value = mapping["upstream_gate"]
    evidence_value = mapping["gate_evidence"]
    if not isinstance(evidence_value, list):
        raise TypeError("gate_evidence must be an array")
    outcome = RouteOutcome(
        route=FullSpaceRoute(expect_string(mapping["route"], context="route")),
        disposition=RouteDisposition(
            expect_string(mapping["disposition"], context="disposition")
        ),
        terminal_status=(
            None
            if terminal_value is None
            else expect_string(terminal_value, context="terminal_status")
        ),
        receipt=None
        if receipt_value is None
        else artifact_ref_from_payload(receipt_value),
        upstream_gate=(
            None
            if upstream_value is None
            else expect_string(upstream_value, context="upstream_gate")
        ),
        gate_evidence=tuple(
            gate_evidence_from_payload(item) for item in evidence_value
        ),
    )
    outcome.validate()
    return outcome


def campaign_receipt_from_payload(value: JsonValue) -> CampaignReceipt:
    mapping = expect_mapping(value, context="campaign receipt")
    expect_exact_keys(
        mapping,
        frozenset(
            (
                "schema_version",
                "disposition",
                "baseline_classification",
                "contract_sha256",
                "route_outcomes",
            )
        ),
        context="campaign receipt",
    )
    if (
        expect_string(mapping["schema_version"], context="schema_version")
        != SCHEMA_VERSION
    ):
        raise ValueError("campaign schema version mismatch")
    outcomes_value = mapping["route_outcomes"]
    if not isinstance(outcomes_value, list):
        raise TypeError("route_outcomes must be an array")
    receipt = CampaignReceipt(
        disposition=CampaignDisposition(
            expect_string(mapping["disposition"], context="disposition")
        ),
        baseline_classification=BaselineClassification(
            expect_string(
                mapping["baseline_classification"], context="baseline_classification"
            )
        ),
        contract_sha256=expect_string(
            mapping["contract_sha256"], context="contract_sha256"
        ),
        route_outcomes=tuple(
            route_outcome_from_payload(item) for item in outcomes_value
        ),
    )
    receipt.validate()
    return receipt


def campaign_receipt_v2_from_payload(value: JsonValue) -> CampaignReceiptV2:
    """Parse the additive SQP campaign envelope without weakening v1."""

    mapping = expect_mapping(value, context="campaign-v2 receipt")
    expect_exact_keys(
        mapping,
        frozenset(
            (
                "schema_version",
                "disposition",
                "baseline_classification",
                "contract_sha256",
                "route_outcomes",
                "sqp_samples",
            )
        ),
        context="campaign-v2 receipt",
    )
    if (
        expect_string(mapping["schema_version"], context="schema_version")
        != SCHEMA_VERSION_V2
    ):
        raise ValueError("campaign-v2 schema version mismatch")
    outcomes_value = mapping["route_outcomes"]
    samples_value = mapping["sqp_samples"]
    if not isinstance(outcomes_value, list) or not isinstance(samples_value, list):
        raise TypeError("campaign-v2 outcomes and samples must be arrays")
    receipt = CampaignReceiptV2(
        disposition=CampaignDisposition(
            expect_string(mapping["disposition"], context="disposition")
        ),
        baseline_classification=BaselineClassification(
            expect_string(
                mapping["baseline_classification"], context="baseline_classification"
            )
        ),
        contract_sha256=expect_string(
            mapping["contract_sha256"], context="contract_sha256"
        ),
        route_outcomes=tuple(
            route_outcome_from_payload(item) for item in outcomes_value
        ),
        sqp_samples=tuple(artifact_ref_from_payload(item) for item in samples_value),
    )
    receipt.validate()
    return receipt


def campaign_receipt_v2_payload(receipt: CampaignReceiptV2) -> dict[str, JsonValue]:
    """Encode the exact additive campaign envelope after route-chain validation."""

    receipt.validate()
    return {
        "baseline_classification": receipt.baseline_classification,
        "contract_sha256": receipt.contract_sha256,
        "disposition": receipt.disposition,
        "route_outcomes": [
            {
                "disposition": outcome.disposition,
                "gate_evidence": [
                    {
                        "artifact": asdict(evidence.artifact),
                        "gate_id": evidence.gate_id,
                    }
                    for evidence in outcome.gate_evidence
                ],
                "receipt": (
                    None if outcome.receipt is None else asdict(outcome.receipt)
                ),
                "route": outcome.route,
                "terminal_status": outcome.terminal_status,
                "upstream_gate": outcome.upstream_gate,
            }
            for outcome in receipt.route_outcomes
        ],
        "schema_version": SCHEMA_VERSION_V2,
        "sqp_samples": [asdict(reference) for reference in receipt.sqp_samples],
    }


def campaign_receipt_from_payload_dispatch(
    value: JsonValue,
) -> CampaignReceipt | CampaignReceiptV2:
    """Dispatch only the two exact campaign schemas."""

    mapping = expect_mapping(value, context="campaign receipt")
    schema_version = expect_string(
        mapping.get("schema_version"), context="schema_version"
    )
    if schema_version == SCHEMA_VERSION:
        return campaign_receipt_from_payload(value)
    if schema_version == SCHEMA_VERSION_V2:
        return campaign_receipt_v2_from_payload(value)
    raise ValueError("campaign schema version mismatch")


def contract_payload_v1() -> dict[str, JsonValue]:
    """Return the byte-frozen legacy Phase-0 contract envelope."""

    return {
        "schema_version": SCHEMA_VERSION,
        "physics": cast(JsonValue, frozen_problem_contract_payload()),
        "routes": cast(JsonValue, frozen_route_contract_payload_v1()),
    }


def contract_payload_v2() -> dict[str, JsonValue]:
    """Return the additive SQP campaign contract envelope."""

    return {
        "schema_version": SCHEMA_VERSION_V2,
        "physics": cast(JsonValue, frozen_problem_contract_payload()),
        "routes": cast(JsonValue, frozen_route_contract_payload_v2()),
    }


def contract_payload_v3() -> dict[str, JsonValue]:
    """Return the additive FTR campaign contract envelope."""

    return {
        "schema_version": SCHEMA_VERSION_V3,
        "physics": cast(JsonValue, frozen_problem_contract_payload()),
        "routes": cast(JsonValue, frozen_route_contract_payload_v3()),
    }


def contract_payload() -> dict[str, JsonValue]:
    """Compatibility alias for the byte-frozen campaign-v1 contract."""

    return contract_payload_v1()


def contract_sha256_v1() -> str:
    return hashlib.sha256(canonical_json_bytes(contract_payload_v1())).hexdigest()


def contract_sha256_v2() -> str:
    return hashlib.sha256(canonical_json_bytes(contract_payload_v2())).hexdigest()


def contract_sha256_v3() -> str:
    return hashlib.sha256(canonical_json_bytes(contract_payload_v3())).hexdigest()


def contract_sha256() -> str:
    """Compatibility alias for the byte-frozen campaign-v1 digest."""

    return contract_sha256_v1()


def bootstrap_identity_sha256_from_payload(value: JsonValue) -> str:
    """Hash only the bootstrap's physics/state identity, excluding runtime paths."""

    mapping = expect_mapping(value, context="bootstrap artifact")
    required = ("exact_mask", "layout", "state", "targets")
    if any(key not in mapping for key in required):
        raise ValueError("bootstrap artifact lacks its physics identity")
    identity: dict[str, JsonValue] = {key: mapping[key] for key in required}
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def write_contract(path: Path) -> ArtifactRef:
    """Create the contract once and return its content-addressed reference."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(contract_payload())
    path.write_bytes(payload)
    return ArtifactRef(
        relative_path=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=SCHEMA_VERSION,
    )


def write_contract_v2(path: Path) -> ArtifactRef:
    """Create the additive SQP contract once without touching the v1 writer."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(contract_payload_v2())
    path.write_bytes(payload)
    return ArtifactRef(
        relative_path=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=SCHEMA_VERSION_V2,
    )


def run_request_payload(request: RunRequest) -> dict[str, JsonValue]:
    """Encode a legacy-v1 request, rejecting the additive SQP route."""

    request.validate()
    payload = asdict(request)
    return {"schema_version": SCHEMA_VERSION, "request": payload}


def run_request_payload_v2(request: RunRequest) -> dict[str, JsonValue]:
    """Encode an exact campaign-v2 legacy or SQP request."""

    request.validate_v2()
    payload = asdict(request)
    return {"schema_version": SCHEMA_VERSION_V2, "request": payload}


def run_request_payload_v3(request: RunRequest) -> dict[str, JsonValue]:
    """Encode one exact campaign-v3 request."""

    request.validate_v3()
    return {"schema_version": SCHEMA_VERSION_V3, "request": asdict(request)}


def run_request_from_payload(value: JsonValue) -> RunRequest:
    """Parse and cross-validate the exact request embedded in a run receipt."""

    request = _run_request_from_payload_unvalidated(value)
    request.validate()
    return request


def run_request_v2_from_payload(value: JsonValue) -> RunRequest:
    """Parse and cross-validate one campaign-v2 request."""

    request = _run_request_from_payload_unvalidated(value)
    request.validate_v2()
    return request


def run_request_v3_from_payload(value: JsonValue) -> RunRequest:
    """Parse and cross-validate one campaign-v3 request."""

    request = _run_request_from_payload_unvalidated(value)
    request.validate_v3()
    return request


def _run_request_from_payload_unvalidated(value: JsonValue) -> RunRequest:
    mapping = expect_mapping(value, context="run request")
    expect_exact_keys(
        mapping,
        frozenset(("phase", "route", "device", "steps", "sample")),
        context="run request",
    )
    steps_value = mapping["steps"]
    sample_value = mapping["sample"]
    return RunRequest(
        phase=RunPhase(expect_string(mapping["phase"], context="request.phase")),
        route=FullSpaceRoute(expect_string(mapping["route"], context="request.route")),
        device=DeviceLane(expect_string(mapping["device"], context="request.device")),
        steps=(
            None
            if steps_value is None
            else expect_integer(steps_value, context="request.steps")
        ),
        sample=(
            None
            if sample_value is None
            else CompleteSample(expect_string(sample_value, context="request.sample"))
        ),
    )


def sqp_sample_receipt_payload(receipt: SqpSampleReceipt) -> dict[str, JsonValue]:
    """Encode one immutable SQP sample receipt after semantic validation."""

    receipt.validate()
    runtime_identity = asdict(receipt.runtime_identity)
    runtime_identity["argv"] = list(receipt.runtime_identity.argv)
    return {
        "bootstrap_artifact": asdict(receipt.bootstrap_artifact),
        "bootstrap_identity_sha256": receipt.bootstrap_identity_sha256,
        "contract_sha256": contract_sha256_v2(),
        "endpoint_certificate": (
            None
            if receipt.endpoint_certificate is None
            else asdict(receipt.endpoint_certificate)
        ),
        "final_d2h_transfers": receipt.final_d2h_transfers,
        "gpu_memory": asdict(receipt.gpu_memory),
        "hot_d2h_transfers": receipt.hot_d2h_transfers,
        "hot_h2d_transfers": receipt.hot_h2d_transfers,
        "initial_h2d_transfers": receipt.initial_h2d_transfers,
        "peak_memory_bytes": receipt.peak_memory_bytes,
        "peak_memory_fraction": receipt.peak_memory_fraction,
        "promotion_eligible": receipt.promotion_eligible,
        "raw_result": asdict(receipt.raw_result),
        "request": asdict(receipt.request),
        "runtime_evidence": asdict(receipt.runtime_evidence),
        "runtime_identity": runtime_identity,
        "schema_version": SQP_SAMPLE_SCHEMA_VERSION,
        "source_identity": asdict(receipt.source_identity),
        "synchronized_solve_seconds": receipt.synchronized_solve_seconds,
        "terminal_status": receipt.terminal_status,
        "total_child_wall_seconds": receipt.total_child_wall_seconds,
    }


def sqp_sample_receipt_from_payload(value: JsonValue) -> SqpSampleReceipt:
    """Parse the exact SQP sample schema and reject raw-result promotion."""

    mapping = expect_mapping(value, context="SQP sample receipt")
    expect_exact_keys(
        mapping,
        frozenset(
            (
                "schema_version",
                "contract_sha256",
                "request",
                "source_identity",
                "runtime_identity",
                "runtime_evidence",
                "bootstrap_artifact",
                "bootstrap_identity_sha256",
                "synchronized_solve_seconds",
                "total_child_wall_seconds",
                "hot_h2d_transfers",
                "hot_d2h_transfers",
                "initial_h2d_transfers",
                "final_d2h_transfers",
                "peak_memory_bytes",
                "peak_memory_fraction",
                "raw_result",
                "gpu_memory",
                "endpoint_certificate",
                "promotion_eligible",
                "terminal_status",
            )
        ),
        context="SQP sample receipt",
    )
    if mapping["schema_version"] != SQP_SAMPLE_SCHEMA_VERSION:
        raise ValueError("SQP sample receipt schema mismatch")
    if mapping["contract_sha256"] != contract_sha256_v2():
        raise ValueError("SQP sample receipt contract digest mismatch")
    promotion_eligible = mapping["promotion_eligible"]
    if not isinstance(promotion_eligible, bool):
        raise TypeError("SQP sample promotion_eligible must be a Boolean")
    certificate_value = mapping["endpoint_certificate"]
    receipt = SqpSampleReceipt(
        request=run_request_v2_from_payload(mapping["request"]),
        source_identity=_snapshot_contract.source_identity_from_payload(
            mapping["source_identity"]
        ),
        runtime_identity=_snapshot_contract.runtime_identity_from_payload(
            mapping["runtime_identity"]
        ),
        runtime_evidence=artifact_ref_from_payload(mapping["runtime_evidence"]),
        bootstrap_artifact=artifact_ref_from_payload(mapping["bootstrap_artifact"]),
        bootstrap_identity_sha256=expect_string(
            mapping["bootstrap_identity_sha256"],
            context="bootstrap_identity_sha256",
        ),
        raw_result=artifact_ref_from_payload(mapping["raw_result"]),
        gpu_memory=artifact_ref_from_payload(mapping["gpu_memory"]),
        endpoint_certificate=(
            None
            if certificate_value is None
            else artifact_ref_from_payload(certificate_value)
        ),
        promotion_eligible=promotion_eligible,
        terminal_status=expect_string(
            mapping["terminal_status"], context="terminal_status"
        ),
        synchronized_solve_seconds=expect_number(
            mapping["synchronized_solve_seconds"],
            context="synchronized_solve_seconds",
        ),
        total_child_wall_seconds=expect_number(
            mapping["total_child_wall_seconds"],
            context="total_child_wall_seconds",
        ),
        hot_h2d_transfers=expect_integer(
            mapping["hot_h2d_transfers"], context="hot_h2d_transfers"
        ),
        hot_d2h_transfers=expect_integer(
            mapping["hot_d2h_transfers"], context="hot_d2h_transfers"
        ),
        initial_h2d_transfers=expect_integer(
            mapping["initial_h2d_transfers"], context="initial_h2d_transfers"
        ),
        final_d2h_transfers=expect_integer(
            mapping["final_d2h_transfers"], context="final_d2h_transfers"
        ),
        peak_memory_bytes=expect_integer(
            mapping["peak_memory_bytes"], context="peak_memory_bytes"
        ),
        peak_memory_fraction=expect_number(
            mapping["peak_memory_fraction"], context="peak_memory_fraction"
        ),
    )
    receipt.validate()
    return receipt
