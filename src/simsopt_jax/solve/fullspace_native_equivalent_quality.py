"""NEQ-GNTR1 adapter for device-timed native-equivalent quality search.

The compiled loop owns only optimizer work and the route-local quality latch.
Independent endpoint derivatives and evidence are formed by the separate
post-timing finalizer; continuation and cross-evaluator audits remain external.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonAcceptedState,
    ProjectedGaussNewtonLoopResult,
    ProjectedGaussNewtonOptions,
    ProjectedGaussNewtonResult,
    finalize_projected_gauss_newton_trust_region,
    run_projected_gauss_newton_trust_region_loop,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceEvaluation,
    FullSpaceProblem,
    evaluate_fullspace,
    flatten_fullspace_constraints,
    fullspace_constraint_vector,
    fullspace_objective_residual_vector,
    fullspace_value_and_grad,
)
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    ObjectiveResidualReconstruction,
    certify_fullspace_objective_residuals,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256

from .fullspace import (
    FullSpaceScaling,
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
)
from .fullspace_gauss_newton_trust_region import CFS_GNTR1_OPTIONS
from .fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_value_constraints,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-neq-gntr1-result-v1"
ROUTE: Final = "NEQ-GNTR1"
NEQ_GNTR1_STATE_SIZE: Final = 716
NEQ_GNTR1_EQUALITY_SIZE: Final = 255
NEQ_GNTR1_OBJECTIVE_RESIDUAL_SIZE: Final = 2110
NATIVE_OBJECTIVE_TARGET: Final = 4.4822246533126125e-08
NEQ_GNTR1_OPTIONS: Final = replace(
    CFS_GNTR1_OPTIONS,
    maximum_accepted_steps=256,
    maximum_attempts=300,
)
NEQ_GNTR2_SCHEMA_VERSION: Final = "single-stage-fullspace-neq-gntr2-result-v1"
NEQ_GNTR2_ROUTE: Final = "NEQ-GNTR2"
NEQ_GNTR2_OPTIONS: Final = replace(
    NEQ_GNTR1_OPTIONS,
    maximum_nonlinear_corrections=2,
)
NEQ_GNTR3_SCHEMA_VERSION: Final = "single-stage-fullspace-neq-gntr3-result-v1"
NEQ_GNTR3_ROUTE: Final = "NEQ-GNTR3"
NEQ_GNTR3_OPTIONS: Final = replace(
    NEQ_GNTR2_OPTIONS,
    enable_step_bound_safeguard=True,
)
_RETRACTION_CANARY_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-neq-gntr-retraction-canary-result-v1"
)
_RETRACTION_CANARY_ROUTE: Final = "NEQ-GNTR-RETRACTION-CANARY"
_NEQ_GNTR1_POLICY_OPTION_FIELDS: Final = (
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
_PROJECTED_GNTR_OPTIONS_V1_FIELDS: Final = (
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
    "maximum_nonlinear_corrections",
)
_PROJECTED_GNTR_OPTIONS_V2_FIELDS: Final = (
    *_PROJECTED_GNTR_OPTIONS_V1_FIELDS,
    "enable_step_bound_safeguard",
)


class KktTelemetryStatus(IntEnum):
    """Availability state for informational, non-gating KKT telemetry."""

    AVAILABLE_FINITE = 0
    AVAILABLE_NONFINITE = 1
    UNAVAILABLE = 2


class DerivativeTransposeCertificate(NamedTuple):
    """Deterministic scaled-coordinate constraint transpose certificate."""

    primal_dot: jax.Array
    transpose_dot: jax.Array
    denominator: jax.Array
    defect: jax.Array
    state_probe: jax.Array
    equality_probe: jax.Array
    jvp_action: jax.Array
    vjp_action: jax.Array
    all_finite: jax.Array
    certified: jax.Array


class KktTelemetry(NamedTuple):
    """Non-gating endpoint KKT values with an explicit availability status."""

    status: jax.Array
    scaled_stationarity_infinity_norm: jax.Array
    scaled_feasibility_infinity_norm: jax.Array


class NativeEquivalentEndpointEvidence(NamedTuple):
    """Independent same-state evidence produced after synchronized timing."""

    physical_state: jax.Array
    evaluation: FullSpaceEvaluation
    objective_gradient: jax.Array
    raw_equalities: jax.Array
    scaled_equalities: jax.Array
    residual_reconstruction: ObjectiveResidualReconstruction
    transpose_certificate: DerivativeTransposeCertificate
    kkt_telemetry: KktTelemetry
    objective_quality_valid: jax.Array
    equality_quality_valid: jax.Array
    scaled_feasibility_valid: jax.Array
    gradient_valid: jax.Array
    residual_contract_valid: jax.Array
    derivative_contract_valid: jax.Array
    all_finite: jax.Array
    local_audit_passed: jax.Array


class NativeEquivalentQualityResult(NamedTuple):
    """Timed candidate plus untimed evidence for external campaign audits."""

    schema_version: str
    route: str
    loop_result: ProjectedGaussNewtonLoopResult
    optimizer_result: ProjectedGaussNewtonResult
    accepted_physical_coordinates: jax.Array
    accepted_state_mask: jax.Array
    endpoint: NativeEquivalentEndpointEvidence
    latched_state_exact: jax.Array
    candidate_ready_for_external_audit: jax.Array


class NativeEquivalentQualityMargins(NamedTuple):
    """Signed terminal quality distances; nonnegative values meet the gate."""

    objective_margin: jax.Array
    component_margins: jax.Array
    minimum_component_margin: jax.Array
    minimum_component_index: jax.Array
    scaled_feasibility_margin: jax.Array
    residual_value_margin: jax.Array
    residual_gradient_margin: jax.Array
    transpose_margin: jax.Array
    objective_usage_ratio: jax.Array
    component_usage_ratio: jax.Array
    scaled_feasibility_usage_ratio: jax.Array
    all_finite: jax.Array


class NativeEquivalentTerminalDiagnosticResult(NamedTuple):
    """Additive no-hit terminal evidence, independent of the device latch."""

    base_result: NativeEquivalentQualityResult
    raw_endpoint: CfsSqp1EndpointDiagnostics
    quality_margins: NativeEquivalentQualityMargins
    raw_kkt_status: jax.Array
    terminal_state_bound: jax.Array
    terminal_quality_satisfied: jax.Array


class NeqGntrRetractionCanaryIdentity(NamedTuple):
    """Exact non-production identity for one iterative-retraction canary."""

    schema_version: str
    route: str
    base_neq_gntr1_policy_sha256: str
    problem_sha256: str
    optimizer_options_sha256: str
    scaling_sha256: str
    initial_physical_state_sha256: str
    identity_sha256: str


class NeqGntrRetractionCanaryResult(NamedTuple):
    """Numerical canary evidence that is never promotion eligible."""

    identity: NeqGntrRetractionCanaryIdentity
    loop_result: ProjectedGaussNewtonLoopResult
    optimizer_result: ProjectedGaussNewtonResult
    accepted_physical_coordinates: jax.Array
    accepted_state_mask: jax.Array
    endpoint: NativeEquivalentEndpointEvidence
    terminal_state_exact: jax.Array
    native_quality_observed: jax.Array
    promotion_eligible: jax.Array


class NeqGntr2Identity(NamedTuple):
    """Exact numerical identity for one authoritative NEQ-GNTR2 execution."""

    schema_version: str
    route: str
    base_neq_gntr1_policy_sha256: str
    problem_sha256: str
    optimizer_options_sha256: str
    scaling_sha256: str
    bootstrap_state_sha256: str
    initial_physical_state_sha256: str
    identity_sha256: str


class NeqGntr2Result(NamedTuple):
    """Audited NEQ-GNTR2 candidate for an external receipt decision."""

    schema_version: str
    route: str
    identity: NeqGntr2Identity
    loop_result: ProjectedGaussNewtonLoopResult
    optimizer_result: ProjectedGaussNewtonResult
    accepted_physical_coordinates: jax.Array
    accepted_state_mask: jax.Array
    endpoint: NativeEquivalentEndpointEvidence
    latched_state_exact: jax.Array
    candidate_ready_for_external_audit: jax.Array


class NeqGntr3Identity(NamedTuple):
    """Exact numerical identity for one authoritative NEQ-GNTR3 execution."""

    schema_version: str
    route: str
    base_neq_gntr1_policy_sha256: str
    problem_sha256: str
    optimizer_options_sha256: str
    scaling_sha256: str
    bootstrap_state_sha256: str
    initial_physical_state_sha256: str
    identity_sha256: str


class NeqGntr3Result(NamedTuple):
    """Audited NEQ-GNTR3 candidate for an external receipt decision."""

    schema_version: str
    route: str
    identity: NeqGntr3Identity
    loop_result: ProjectedGaussNewtonLoopResult
    optimizer_result: ProjectedGaussNewtonResult
    accepted_physical_coordinates: jax.Array
    accepted_state_mask: jax.Array
    endpoint: NativeEquivalentEndpointEvidence
    latched_state_exact: jax.Array
    candidate_ready_for_external_audit: jax.Array


@dataclass(frozen=True, slots=True)
class NativeEquivalentQualityPolicy:
    """Immutable native endpoint and numerical quality contract."""

    native_raw_equalities: jax.Array
    native_raw_equalities_sha256: str
    constraint_inverse_scale: jax.Array
    objective_target: float = NATIVE_OBJECTIVE_TARGET
    state_size: int = NEQ_GNTR1_STATE_SIZE
    equality_size: int = NEQ_GNTR1_EQUALITY_SIZE
    objective_residual_size: int = NEQ_GNTR1_OBJECTIVE_RESIDUAL_SIZE
    component_absolute_tolerance: float = 1.0e-12
    component_relative_tolerance: float = 1.0e-10
    scaled_feasibility_tolerance: float = 1.0e-10
    residual_value_defect_tolerance: float = 1.0e-12
    residual_gradient_defect_tolerance: float = 1.0e-10
    transpose_defect_tolerance: float = 1.0e-10
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        raw = np.asarray(self.native_raw_equalities)
        scale = np.asarray(self.constraint_inverse_scale)
        if self.state_size != NEQ_GNTR1_STATE_SIZE:
            raise ValueError("NEQ-GNTR1 requires exactly 716 optimizer coordinates")
        if self.equality_size != NEQ_GNTR1_EQUALITY_SIZE:
            raise ValueError("NEQ-GNTR1 requires exactly 255 equalities")
        if self.objective_residual_size != NEQ_GNTR1_OBJECTIVE_RESIDUAL_SIZE:
            raise ValueError("NEQ-GNTR1 requires exactly 2110 objective residuals")
        if raw.shape != (self.equality_size,):
            raise ValueError("native_raw_equalities must have shape (255,)")
        if scale.shape != (self.equality_size,):
            raise ValueError("constraint_inverse_scale must have shape (255,)")
        if raw.dtype != np.dtype(np.float64) or scale.dtype != np.dtype(np.float64):
            raise TypeError("native equalities and constraint scale must be FP64")
        if not np.all(np.isfinite(raw)):
            raise ValueError("native_raw_equalities must be finite")
        if not np.all(np.isfinite(scale)) or np.any(scale == 0.0):
            raise ValueError("constraint_inverse_scale must be finite and nonzero")
        if not np.isfinite(self.objective_target):
            raise ValueError("objective_target must be finite")
        tolerances = np.asarray(
            (
                self.component_absolute_tolerance,
                self.component_relative_tolerance,
                self.scaled_feasibility_tolerance,
                self.residual_value_defect_tolerance,
                self.residual_gradient_defect_tolerance,
                self.transpose_defect_tolerance,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
            raise ValueError("quality thresholds must be finite and nonnegative")
        actual_hash = exact_numeric_tree_sha256(self.native_raw_equalities)
        if self.native_raw_equalities_sha256 != actual_hash:
            raise ValueError("native_raw_equalities_sha256 does not match vector bytes")
        policy_payload = (
            "single-stage-native-equivalent-quality-policy-v1",
            SCHEMA_VERSION,
            ROUTE,
            self.native_raw_equalities,
            self.native_raw_equalities_sha256,
            self.constraint_inverse_scale,
            self.objective_target,
            self.state_size,
            self.equality_size,
            self.objective_residual_size,
            self.component_absolute_tolerance,
            self.component_relative_tolerance,
            self.scaled_feasibility_tolerance,
            self.residual_value_defect_tolerance,
            self.residual_gradient_defect_tolerance,
            self.transpose_defect_tolerance,
            tuple(
                (field_name, getattr(NEQ_GNTR1_OPTIONS, field_name))
                for field_name in _NEQ_GNTR1_POLICY_OPTION_FIELDS
            ),
        )
        object.__setattr__(
            self,
            "policy_sha256",
            exact_numeric_tree_sha256(policy_payload),
        )


_PreparedLoop = Callable[[jax.Array], ProjectedGaussNewtonLoopResult]
_PreparedFinalize = Callable[
    [ProjectedGaussNewtonLoopResult],
    tuple[ProjectedGaussNewtonResult, NativeEquivalentEndpointEvidence],
]
_PreparedLedgerMap = Callable[[jax.Array], jax.Array]
_PreparedAcceptedQualityReplay = Callable[
    [jax.Array, jax.Array], "NativeEquivalentAcceptedQualityReplay"
]
_PreparedTerminalEndpoint = Callable[
    [jax.Array, jax.Array], "NativeEquivalentTerminalEndpointEvidence"
]


class NativeEquivalentAcceptedQualityReplay(NamedTuple):
    """Independent post-timing quality replay over the accepted-state ledger."""

    objectives: jax.Array
    raw_equalities: jax.Array
    scaled_equalities: jax.Array
    accepted_state_mask: jax.Array
    coordinates_finite: jax.Array
    objective_finite: jax.Array
    raw_equalities_finite: jax.Array
    scaled_equalities_finite: jax.Array
    objective_satisfied: jax.Array
    component_bounds_satisfied: jax.Array
    scaled_feasibility_satisfied: jax.Array
    quality_satisfied: jax.Array
    first_quality_accepted_step: jax.Array
    quality_candidate_reached: jax.Array


class NativeEquivalentTerminalEndpointEvidence(NamedTuple):
    """One-call raw KKT and independent objective-residual evidence."""

    raw_endpoint: CfsSqp1EndpointDiagnostics
    objective_residual_vector: jax.Array
    reconstructed_objective: jax.Array
    reconstructed_objective_gradient: jax.Array
    authoritative_objective: jax.Array
    authoritative_objective_gradient: jax.Array
    value_scaled_defect: jax.Array
    gradient_scaled_defect: jax.Array
    all_finite: jax.Array


@dataclass(frozen=True, slots=True)
class PreparedNeqTerminalEndpointDiagnostics:
    """One separately compiled raw endpoint diagnostic executable."""

    _run_endpoint: _PreparedTerminalEndpoint = field(repr=False, compare=False)

    def run(
        self,
        optimizer_coordinates: jax.Array,
        scaled_multipliers: jax.Array,
    ) -> CfsSqp1EndpointDiagnostics:
        """Dispatch exactly one post-finalizer raw endpoint diagnostic."""

        return self.run_evidence(
            optimizer_coordinates,
            scaled_multipliers,
        ).raw_endpoint

    def run_evidence(
        self,
        optimizer_coordinates: jax.Array,
        scaled_multipliers: jax.Array,
    ) -> NativeEquivalentTerminalEndpointEvidence:
        """Dispatch raw KKT and residual reconstruction in one executable."""

        return self._run_endpoint(optimizer_coordinates, scaled_multipliers)


@dataclass(frozen=True, slots=True)
class PreparedNeqAcceptedQualityDiagnostics:
    """One separately compiled accepted-ledger quality replay executable."""

    _run_quality: _PreparedAcceptedQualityReplay = field(
        repr=False,
        compare=False,
    )

    def run(
        self,
        accepted_optimizer_coordinates: jax.Array,
        accepted_state_mask: jax.Array,
    ) -> NativeEquivalentAcceptedQualityReplay:
        """Replay native-quality gates after synchronized solver timing."""

        return self._run_quality(
            accepted_optimizer_coordinates,
            accepted_state_mask,
        )


class _FinalizedNeqGntrExecution(NamedTuple):
    optimizer_result: ProjectedGaussNewtonResult
    accepted_physical_coordinates: jax.Array
    endpoint: NativeEquivalentEndpointEvidence
    terminal_state_exact: jax.Array


def _finalize_neq_gntr_execution(
    loop_result: ProjectedGaussNewtonLoopResult,
    run_finalizer: _PreparedFinalize,
    map_ledger: _PreparedLedgerMap,
) -> _FinalizedNeqGntrExecution:
    """Finalize and verify exactly the supplied accepted-state ledger."""

    accepted_physical_coordinates = map_ledger(
        loop_result.accepted_optimizer_coordinates
    )
    optimizer_result, endpoint = run_finalizer(loop_result)
    endpoint_row = loop_result.accepted_optimizer_coordinates[
        loop_result.accepted_steps
    ]
    terminal_state_exact = (
        jnp.array_equal(endpoint_row, loop_result.optimizer_coordinates)
        & loop_result.accepted_state_mask[loop_result.accepted_steps]
        & (
            ~loop_result.device_quality_candidate_reached
            | (loop_result.first_quality_accepted_step == loop_result.accepted_steps)
        )
    )
    return _FinalizedNeqGntrExecution(
        optimizer_result=optimizer_result,
        accepted_physical_coordinates=accepted_physical_coordinates,
        endpoint=endpoint,
        terminal_state_exact=terminal_state_exact,
    )


@dataclass(frozen=True, slots=True)
class PreparedNeqGntr1:
    """Compiled timed loop and separately compiled post-timing audit."""

    problem: FullSpaceProblem
    scaling: FullSpaceScaling
    policy: NativeEquivalentQualityPolicy
    initial_physical_state: jax.Array
    initial_optimizer_coordinates: jax.Array
    options: ProjectedGaussNewtonOptions
    _run_loop: _PreparedLoop = field(repr=False, compare=False)
    _finalize: _PreparedFinalize = field(repr=False, compare=False)
    _map_ledger: _PreparedLedgerMap = field(repr=False, compare=False)

    def run_solver_loop(self) -> ProjectedGaussNewtonLoopResult:
        """Launch only the compiled device loop; the caller owns synchronization.

        Timing ends when the returned pytree is passed to
        :func:`jax.block_until_ready`. This method never invokes endpoint
        finalization.
        """

        return self._run_loop(self.initial_optimizer_coordinates)

    def finalize_result(
        self,
        loop_result: ProjectedGaussNewtonLoopResult,
    ) -> NativeEquivalentQualityResult:
        """Run untimed audit after caller synchronization of ``loop_result``.

        The finalizer audits exactly the returned candidate without projection
        or replacement and is intentionally unreachable from
        :meth:`run_solver_loop`.
        """

        finalized = _finalize_neq_gntr_execution(
            loop_result,
            self._finalize,
            self._map_ledger,
        )
        candidate_ready_for_external_audit = (
            loop_result.device_quality_candidate_reached
            & finalized.terminal_state_exact
            & finalized.endpoint.local_audit_passed
        )
        return NativeEquivalentQualityResult(
            schema_version=SCHEMA_VERSION,
            route=ROUTE,
            loop_result=loop_result,
            optimizer_result=finalized.optimizer_result,
            accepted_physical_coordinates=finalized.accepted_physical_coordinates,
            accepted_state_mask=loop_result.accepted_state_mask,
            endpoint=finalized.endpoint,
            latched_state_exact=finalized.terminal_state_exact,
            candidate_ready_for_external_audit=candidate_ready_for_external_audit,
        )


@dataclass(frozen=True, slots=True)
class PreparedNeqGntr2:
    """Compiled authoritative GNTR2 loop and separately compiled audit."""

    identity: NeqGntr2Identity
    problem: FullSpaceProblem
    scaling: FullSpaceScaling
    policy: NativeEquivalentQualityPolicy
    initial_physical_state: jax.Array
    initial_optimizer_coordinates: jax.Array
    options: ProjectedGaussNewtonOptions
    _run_loop: _PreparedLoop = field(repr=False, compare=False)
    _finalize: _PreparedFinalize = field(repr=False, compare=False)
    _map_ledger: _PreparedLedgerMap = field(repr=False, compare=False)

    def run_solver_loop(self) -> ProjectedGaussNewtonLoopResult:
        """Launch only the compiled device loop; the caller synchronizes it."""

        return self._run_loop(self.initial_optimizer_coordinates)

    def finalize_result(
        self,
        loop_result: ProjectedGaussNewtonLoopResult,
    ) -> NeqGntr2Result:
        """Audit the exact result without authorizing scientific promotion."""

        finalized = _finalize_neq_gntr_execution(
            loop_result,
            self._finalize,
            self._map_ledger,
        )
        candidate_ready_for_external_audit = (
            loop_result.device_quality_candidate_reached
            & finalized.terminal_state_exact
            & finalized.endpoint.local_audit_passed
        )
        return NeqGntr2Result(
            schema_version=NEQ_GNTR2_SCHEMA_VERSION,
            route=NEQ_GNTR2_ROUTE,
            identity=self.identity,
            loop_result=loop_result,
            optimizer_result=finalized.optimizer_result,
            accepted_physical_coordinates=finalized.accepted_physical_coordinates,
            accepted_state_mask=loop_result.accepted_state_mask,
            endpoint=finalized.endpoint,
            latched_state_exact=finalized.terminal_state_exact,
            candidate_ready_for_external_audit=candidate_ready_for_external_audit,
        )


@dataclass(frozen=True, slots=True)
class PreparedNeqGntr3:
    """Compiled authoritative GNTR3 loop and separately compiled audit."""

    identity: NeqGntr3Identity
    problem: FullSpaceProblem
    scaling: FullSpaceScaling
    policy: NativeEquivalentQualityPolicy
    initial_physical_state: jax.Array
    initial_optimizer_coordinates: jax.Array
    options: ProjectedGaussNewtonOptions
    _run_loop: _PreparedLoop = field(repr=False, compare=False)
    _finalize: _PreparedFinalize = field(repr=False, compare=False)
    _map_ledger: _PreparedLedgerMap = field(repr=False, compare=False)

    def run_solver_loop(self) -> ProjectedGaussNewtonLoopResult:
        """Launch only the compiled device loop; the caller synchronizes it."""

        return self._run_loop(self.initial_optimizer_coordinates)

    def finalize_result(
        self,
        loop_result: ProjectedGaussNewtonLoopResult,
    ) -> NeqGntr3Result:
        """Audit the exact result without authorizing scientific promotion."""

        finalized = _finalize_neq_gntr_execution(
            loop_result,
            self._finalize,
            self._map_ledger,
        )
        candidate_ready_for_external_audit = (
            loop_result.device_quality_candidate_reached
            & finalized.terminal_state_exact
            & finalized.endpoint.local_audit_passed
        )
        return NeqGntr3Result(
            schema_version=NEQ_GNTR3_SCHEMA_VERSION,
            route=NEQ_GNTR3_ROUTE,
            identity=self.identity,
            loop_result=loop_result,
            optimizer_result=finalized.optimizer_result,
            accepted_physical_coordinates=finalized.accepted_physical_coordinates,
            accepted_state_mask=loop_result.accepted_state_mask,
            endpoint=finalized.endpoint,
            latched_state_exact=finalized.terminal_state_exact,
            candidate_ready_for_external_audit=candidate_ready_for_external_audit,
        )


@dataclass(frozen=True, slots=True)
class PreparedNeqGntrRetractionCanary:
    """Compiled iterative-retraction canary with non-production identity."""

    identity: NeqGntrRetractionCanaryIdentity
    problem: FullSpaceProblem
    scaling: FullSpaceScaling
    policy: NativeEquivalentQualityPolicy
    initial_physical_state: jax.Array
    initial_optimizer_coordinates: jax.Array
    options: ProjectedGaussNewtonOptions
    _run_loop: _PreparedLoop = field(repr=False, compare=False)
    _finalize: _PreparedFinalize = field(repr=False, compare=False)
    _map_ledger: _PreparedLedgerMap = field(repr=False, compare=False)

    def run_solver_loop(self) -> ProjectedGaussNewtonLoopResult:
        """Launch only the compiled canary loop; the caller synchronizes it."""

        return self._run_loop(self.initial_optimizer_coordinates)

    def finalize_result(
        self,
        loop_result: ProjectedGaussNewtonLoopResult,
    ) -> NeqGntrRetractionCanaryResult:
        """Audit the exact canary terminal state without granting promotion."""

        finalized = _finalize_neq_gntr_execution(
            loop_result,
            self._finalize,
            self._map_ledger,
        )
        native_quality_observed = (
            loop_result.device_quality_candidate_reached
            & finalized.terminal_state_exact
            & finalized.endpoint.local_audit_passed
        )
        return NeqGntrRetractionCanaryResult(
            identity=self.identity,
            loop_result=loop_result,
            optimizer_result=finalized.optimizer_result,
            accepted_physical_coordinates=finalized.accepted_physical_coordinates,
            accepted_state_mask=loop_result.accepted_state_mask,
            endpoint=finalized.endpoint,
            terminal_state_exact=finalized.terminal_state_exact,
            native_quality_observed=native_quality_observed,
            promotion_eligible=jnp.asarray(False),
        )


def accepted_state_meets_native_quality(
    accepted_state: ProjectedGaussNewtonAcceptedState,
    policy: NativeEquivalentQualityPolicy,
) -> jax.Array:
    """Apply the frozen route-local predicate using retained accepted values."""

    raw_equalities = accepted_state.constraints / policy.constraint_inverse_scale
    component_bound = (
        jnp.abs(policy.native_raw_equalities)
        + policy.component_absolute_tolerance
        + policy.component_relative_tolerance * jnp.abs(policy.native_raw_equalities)
    )
    return (
        jnp.all(jnp.isfinite(accepted_state.optimizer_coordinates))
        & jnp.isfinite(accepted_state.objective)
        & (accepted_state.objective <= policy.objective_target)
        & jnp.all(jnp.isfinite(accepted_state.constraints))
        & jnp.isfinite(accepted_state.scaled_feasibility_inf)
        & (accepted_state.scaled_feasibility_inf <= policy.scaled_feasibility_tolerance)
        & jnp.all(jnp.isfinite(raw_equalities))
        & jnp.all(jnp.abs(raw_equalities) <= component_bound)
    )


def deterministic_constraint_transpose_certificate(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    *,
    tolerance: float = 1.0e-10,
) -> DerivativeTransposeCertificate:
    """Independently certify JVP/VJP orientation for ``D*q(z0 + S*u)``."""

    def scaled_constraints(candidate: jax.Array) -> jax.Array:
        physical = fullspace_physical_coordinates(candidate, scaling)
        return (
            fullspace_constraint_vector(physical, problem)
            * scaling.constraint_inverse_scale
        )

    state_indices = jnp.arange(
        1,
        optimizer_coordinates.size + 1,
        dtype=optimizer_coordinates.dtype,
    )
    state_probe = jnp.sin(state_indices)
    state_probe = state_probe / jnp.linalg.norm(state_probe)
    equality_indices = jnp.arange(
        1,
        scaling.constraint_inverse_scale.size + 1,
        dtype=optimizer_coordinates.dtype,
    )
    equality_probe = jnp.cos(0.5 * equality_indices)
    equality_probe = equality_probe / jnp.linalg.norm(equality_probe)
    _, jvp_action = jax.jvp(
        scaled_constraints,
        (optimizer_coordinates,),
        (state_probe,),
    )
    _, pullback = jax.vjp(scaled_constraints, optimizer_coordinates)
    vjp_action = pullback(equality_probe)[0]
    primal_dot = jnp.vdot(equality_probe, jvp_action)
    transpose_dot = jnp.vdot(state_probe, vjp_action)
    denominator = jnp.maximum(
        jnp.asarray(jnp.finfo(optimizer_coordinates.dtype).tiny),
        jnp.linalg.norm(equality_probe) * jnp.linalg.norm(jvp_action)
        + jnp.linalg.norm(state_probe) * jnp.linalg.norm(vjp_action),
    )
    defect = jnp.abs(primal_dot - transpose_dot) / denominator
    all_finite = (
        jnp.all(jnp.isfinite(state_probe))
        & jnp.all(jnp.isfinite(equality_probe))
        & jnp.all(jnp.isfinite(jvp_action))
        & jnp.all(jnp.isfinite(vjp_action))
        & jnp.isfinite(primal_dot)
        & jnp.isfinite(transpose_dot)
        & jnp.isfinite(denominator)
        & jnp.isfinite(defect)
    )
    return DerivativeTransposeCertificate(
        primal_dot=primal_dot,
        transpose_dot=transpose_dot,
        denominator=denominator,
        defect=defect,
        state_probe=state_probe,
        equality_probe=equality_probe,
        jvp_action=jvp_action,
        vjp_action=vjp_action,
        all_finite=all_finite,
        certified=all_finite & (defect <= tolerance),
    )


def classify_kkt_telemetry(
    optimizer_result: ProjectedGaussNewtonResult,
    *,
    available: jax.Array | bool = True,
) -> KktTelemetry:
    """Encode informational KKT availability without making it a quality gate."""

    finite = (
        jnp.isfinite(optimizer_result.scaled_stationarity_inf)
        & jnp.isfinite(optimizer_result.scaled_feasibility_inf)
        & jnp.all(jnp.isfinite(optimizer_result.multipliers))
        & jnp.all(jnp.isfinite(optimizer_result.stationarity))
    )
    status = jnp.where(
        jnp.asarray(available),
        jnp.where(
            finite,
            int(KktTelemetryStatus.AVAILABLE_FINITE),
            int(KktTelemetryStatus.AVAILABLE_NONFINITE),
        ),
        int(KktTelemetryStatus.UNAVAILABLE),
    ).astype(jnp.int32)
    return KktTelemetry(
        status=status,
        scaled_stationarity_infinity_norm=jnp.where(
            jnp.asarray(available),
            optimizer_result.scaled_stationarity_inf,
            jnp.asarray(0.0, dtype=optimizer_result.optimizer_coordinates.dtype),
        ),
        scaled_feasibility_infinity_norm=jnp.where(
            jnp.asarray(available),
            optimizer_result.scaled_feasibility_inf,
            jnp.asarray(0.0, dtype=optimizer_result.optimizer_coordinates.dtype),
        ),
    )


def native_equivalent_quality_margins(
    endpoint: NativeEquivalentEndpointEvidence,
    policy: NativeEquivalentQualityPolicy,
) -> NativeEquivalentQualityMargins:
    """Compute the frozen signed distances and usage ratios at any endpoint."""

    component_bounds = (
        jnp.abs(policy.native_raw_equalities)
        + policy.component_absolute_tolerance
        + policy.component_relative_tolerance * jnp.abs(policy.native_raw_equalities)
    )
    component_margins = component_bounds - jnp.abs(endpoint.raw_equalities)
    minimum_component_index = jnp.argmin(component_margins).astype(jnp.int32)
    minimum_component_margin = component_margins[minimum_component_index]
    scaled_feasibility = jnp.max(jnp.abs(endpoint.scaled_equalities))
    objective_margin = policy.objective_target - endpoint.evaluation.weighted_total
    scaled_feasibility_margin = policy.scaled_feasibility_tolerance - scaled_feasibility
    residual_value_margin = (
        policy.residual_value_defect_tolerance
        - endpoint.residual_reconstruction.value_scaled_defect
    )
    residual_gradient_margin = (
        policy.residual_gradient_defect_tolerance
        - endpoint.residual_reconstruction.gradient_scaled_defect
    )
    transpose_margin = (
        policy.transpose_defect_tolerance - endpoint.transpose_certificate.defect
    )
    objective_usage_ratio = jnp.where(
        policy.objective_target > 0.0,
        endpoint.evaluation.weighted_total / policy.objective_target,
        jnp.asarray(jnp.inf, dtype=endpoint.evaluation.weighted_total.dtype),
    )
    component_usage_ratio = jnp.max(
        jnp.where(
            component_bounds > 0.0,
            jnp.abs(endpoint.raw_equalities) / component_bounds,
            jnp.full_like(component_bounds, jnp.inf),
        )
    )
    scaled_feasibility_usage_ratio = jnp.where(
        policy.scaled_feasibility_tolerance > 0.0,
        scaled_feasibility / policy.scaled_feasibility_tolerance,
        jnp.asarray(jnp.inf, dtype=scaled_feasibility.dtype),
    )
    all_finite = (
        jnp.isfinite(objective_margin)
        & jnp.all(jnp.isfinite(component_margins))
        & jnp.isfinite(minimum_component_margin)
        & jnp.isfinite(scaled_feasibility_margin)
        & jnp.isfinite(residual_value_margin)
        & jnp.isfinite(residual_gradient_margin)
        & jnp.isfinite(transpose_margin)
        & jnp.isfinite(objective_usage_ratio)
        & jnp.isfinite(component_usage_ratio)
        & jnp.isfinite(scaled_feasibility_usage_ratio)
    )
    return NativeEquivalentQualityMargins(
        objective_margin=objective_margin,
        component_margins=component_margins,
        minimum_component_margin=minimum_component_margin,
        minimum_component_index=minimum_component_index,
        scaled_feasibility_margin=scaled_feasibility_margin,
        residual_value_margin=residual_value_margin,
        residual_gradient_margin=residual_gradient_margin,
        transpose_margin=transpose_margin,
        objective_usage_ratio=objective_usage_ratio,
        component_usage_ratio=component_usage_ratio,
        scaled_feasibility_usage_ratio=scaled_feasibility_usage_ratio,
        all_finite=all_finite,
    )


def build_native_equivalent_terminal_diagnostic(
    base_result: NativeEquivalentQualityResult,
    raw_endpoint: CfsSqp1EndpointDiagnostics,
    policy: NativeEquivalentQualityPolicy,
) -> NativeEquivalentTerminalDiagnosticResult:
    """Bind independently computed raw KKT and margins to the unchanged state."""

    margins = native_equivalent_quality_margins(base_result.endpoint, policy)
    raw_kkt_finite = (
        jnp.all(jnp.isfinite(raw_endpoint.scaled_multipliers))
        & jnp.all(jnp.isfinite(raw_endpoint.raw_multipliers))
        & jnp.all(jnp.isfinite(raw_endpoint.raw_stationarity_residual))
        & jnp.isfinite(raw_endpoint.raw_kkt_stationarity_infinity_norm)
    )
    raw_kkt_status = jnp.where(
        raw_kkt_finite,
        int(KktTelemetryStatus.AVAILABLE_FINITE),
        int(KktTelemetryStatus.AVAILABLE_NONFINITE),
    ).astype(jnp.int32)
    accepted_steps = base_result.loop_result.accepted_steps
    accepted_optimizer_coordinates = (
        base_result.loop_result.accepted_optimizer_coordinates[accepted_steps]
    )
    accepted_physical_coordinates = base_result.accepted_physical_coordinates[
        accepted_steps
    ]
    expected_mask = (
        jnp.arange(base_result.accepted_state_mask.size, dtype=jnp.int32)
        <= accepted_steps
    )
    expected_latched_state_exact = (
        jnp.array_equal(
            base_result.loop_result.optimizer_coordinates,
            accepted_optimizer_coordinates,
        )
        & base_result.loop_result.accepted_state_mask[accepted_steps]
        & (
            ~base_result.loop_result.device_quality_candidate_reached
            | (base_result.loop_result.first_quality_accepted_step == accepted_steps)
        )
    )
    terminal_state_bound = (
        jnp.array_equal(
            base_result.optimizer_result.optimizer_coordinates,
            accepted_optimizer_coordinates,
        )
        & jnp.array_equal(
            base_result.optimizer_result.optimizer_coordinates,
            base_result.loop_result.optimizer_coordinates,
        )
        & expected_latched_state_exact
        & jnp.array_equal(
            base_result.latched_state_exact,
            expected_latched_state_exact,
        )
        & jnp.array_equal(
            base_result.loop_result.accepted_state_mask,
            expected_mask,
        )
        & jnp.array_equal(
            base_result.accepted_state_mask,
            base_result.loop_result.accepted_state_mask,
        )
        & jnp.array_equal(
            base_result.endpoint.physical_state,
            accepted_physical_coordinates,
        )
        & jnp.array_equal(
            raw_endpoint.physical_state,
            accepted_physical_coordinates,
        )
        & jnp.array_equal(
            raw_endpoint.scaled_multipliers,
            base_result.optimizer_result.multipliers,
        )
        & jnp.array_equal(
            raw_endpoint.physical_objective,
            base_result.endpoint.evaluation.weighted_total,
        )
        & jnp.array_equal(
            raw_endpoint.raw_constraints, base_result.endpoint.raw_equalities
        )
        & jnp.array_equal(
            raw_endpoint.scaled_constraints, base_result.endpoint.scaled_equalities
        )
    )
    terminal_quality_satisfied = (
        terminal_state_bound
        & margins.all_finite
        & (margins.objective_margin >= 0.0)
        & (margins.minimum_component_margin >= 0.0)
        & (margins.scaled_feasibility_margin >= 0.0)
        & (margins.residual_value_margin >= 0.0)
        & (margins.residual_gradient_margin >= 0.0)
        & (margins.transpose_margin >= 0.0)
        & base_result.endpoint.gradient_valid
    )
    return NativeEquivalentTerminalDiagnosticResult(
        base_result=base_result,
        raw_endpoint=raw_endpoint,
        quality_margins=margins,
        raw_kkt_status=raw_kkt_status,
        terminal_state_bound=terminal_state_bound,
        terminal_quality_satisfied=terminal_quality_satisfied,
    )


def prepare_neq_terminal_endpoint_diagnostics(
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    example_optimizer_coordinates: jax.Array,
    example_scaled_multipliers: jax.Array,
) -> PreparedNeqTerminalEndpointDiagnostics:
    """Compile the one state-bound raw-KKT computation used after finalization."""

    def terminal_endpoint(
        optimizer_coordinates: jax.Array,
        scaled_multipliers: jax.Array,
    ) -> NativeEquivalentTerminalEndpointEvidence:
        raw_endpoint = cfs_sqp1_endpoint_diagnostics(
            optimizer_coordinates,
            scaled_multipliers,
            problem,
            scaling,
        )
        physical_state = fullspace_physical_coordinates(
            optimizer_coordinates,
            scaling,
        )
        objective_residual_vector = fullspace_objective_residual_vector(
            physical_state,
            problem,
        )
        reconstructed_objective = 0.5 * jnp.vdot(
            objective_residual_vector,
            objective_residual_vector,
        )
        _, residual_pullback = jax.vjp(
            lambda state: fullspace_objective_residual_vector(state, problem),
            physical_state,
        )
        reconstructed_objective_gradient = residual_pullback(objective_residual_vector)[
            0
        ]
        authoritative_objective, authoritative_objective_gradient = (
            fullspace_value_and_grad(physical_state, problem)
        )
        value_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=physical_state.dtype),
            jnp.maximum(
                jnp.abs(reconstructed_objective),
                jnp.abs(authoritative_objective),
            ),
        )
        value_scaled_defect = (
            jnp.abs(reconstructed_objective - authoritative_objective) / value_scale
        )
        gradient_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=physical_state.dtype),
            jnp.maximum(
                jnp.linalg.norm(reconstructed_objective_gradient, ord=jnp.inf),
                jnp.linalg.norm(authoritative_objective_gradient, ord=jnp.inf),
            ),
        )
        gradient_scaled_defect = (
            jnp.linalg.norm(
                reconstructed_objective_gradient - authoritative_objective_gradient,
                ord=jnp.inf,
            )
            / gradient_scale
        )
        all_finite = (
            raw_endpoint.all_finite
            & jnp.all(jnp.isfinite(objective_residual_vector))
            & jnp.isfinite(reconstructed_objective)
            & jnp.all(jnp.isfinite(reconstructed_objective_gradient))
            & jnp.isfinite(authoritative_objective)
            & jnp.all(jnp.isfinite(authoritative_objective_gradient))
            & jnp.isfinite(value_scaled_defect)
            & jnp.isfinite(gradient_scaled_defect)
        )
        return NativeEquivalentTerminalEndpointEvidence(
            raw_endpoint=raw_endpoint,
            objective_residual_vector=objective_residual_vector,
            reconstructed_objective=reconstructed_objective,
            reconstructed_objective_gradient=reconstructed_objective_gradient,
            authoritative_objective=authoritative_objective,
            authoritative_objective_gradient=authoritative_objective_gradient,
            value_scaled_defect=value_scaled_defect,
            gradient_scaled_defect=gradient_scaled_defect,
            all_finite=all_finite,
        )

    executable = (
        jax.jit(terminal_endpoint)
        .lower(example_optimizer_coordinates, example_scaled_multipliers)
        .compile()
    )
    return PreparedNeqTerminalEndpointDiagnostics(_run_endpoint=executable)


def prepare_neq_accepted_quality_diagnostics(
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    policy: NativeEquivalentQualityPolicy,
    example_optimizer_ledger: jax.Array,
    example_accepted_state_mask: jax.Array,
) -> PreparedNeqAcceptedQualityDiagnostics:
    """Compile independent post-timing quality replay for accepted states."""

    component_bounds = (
        jnp.abs(policy.native_raw_equalities)
        + policy.component_absolute_tolerance
        + policy.component_relative_tolerance * jnp.abs(policy.native_raw_equalities)
    )

    def replay_accepted_quality(
        optimizer_ledger: jax.Array,
        accepted_state_mask: jax.Array,
    ) -> NativeEquivalentAcceptedQualityReplay:
        def evaluate_row(
            optimizer_coordinates: jax.Array,
        ) -> tuple[jax.Array, ...]:
            objective, scaled_equalities = cfs_sqp1_joint_value_constraints(
                optimizer_coordinates,
                problem,
                scaling,
            )
            raw_equalities = scaled_equalities / policy.constraint_inverse_scale
            scaled_feasibility = jnp.max(jnp.abs(scaled_equalities))
            coordinates_finite = jnp.all(jnp.isfinite(optimizer_coordinates))
            objective_finite = jnp.isfinite(objective)
            raw_equalities_finite = jnp.all(jnp.isfinite(raw_equalities))
            scaled_equalities_finite = jnp.all(jnp.isfinite(scaled_equalities))
            objective_satisfied = objective_finite & (
                objective <= policy.objective_target
            )
            component_bounds_satisfied = raw_equalities_finite & jnp.all(
                jnp.abs(raw_equalities) <= component_bounds
            )
            scaled_feasibility_satisfied = (
                scaled_equalities_finite
                & jnp.isfinite(scaled_feasibility)
                & (scaled_feasibility <= policy.scaled_feasibility_tolerance)
            )
            predicate = accepted_state_meets_native_quality(
                ProjectedGaussNewtonAcceptedState(
                    optimizer_coordinates=optimizer_coordinates,
                    objective=objective,
                    constraints=scaled_equalities,
                    scaled_feasibility_inf=scaled_feasibility,
                ),
                policy,
            )
            return (
                objective,
                raw_equalities,
                scaled_equalities,
                coordinates_finite,
                objective_finite,
                raw_equalities_finite,
                scaled_equalities_finite,
                objective_satisfied,
                component_bounds_satisfied,
                scaled_feasibility_satisfied,
                predicate,
            )

        (
            objectives,
            raw_equalities,
            scaled_equalities,
            coordinates_finite,
            objective_finite,
            raw_equalities_finite,
            scaled_equalities_finite,
            objective_satisfied,
            component_bounds_satisfied,
            scaled_feasibility_satisfied,
            predicate,
        ) = jax.lax.map(evaluate_row, optimizer_ledger)
        row_indices = jnp.arange(accepted_state_mask.size, dtype=jnp.int32)
        candidate_mask = accepted_state_mask & (row_indices > 0)
        quality_satisfied = candidate_mask & predicate
        quality_candidate_reached = jnp.any(quality_satisfied)
        first_quality_accepted_step = jnp.where(
            quality_candidate_reached,
            jnp.argmax(quality_satisfied).astype(jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        return NativeEquivalentAcceptedQualityReplay(
            objectives=objectives,
            raw_equalities=raw_equalities,
            scaled_equalities=scaled_equalities,
            accepted_state_mask=accepted_state_mask,
            coordinates_finite=coordinates_finite,
            objective_finite=objective_finite,
            raw_equalities_finite=raw_equalities_finite,
            scaled_equalities_finite=scaled_equalities_finite,
            objective_satisfied=objective_satisfied,
            component_bounds_satisfied=component_bounds_satisfied,
            scaled_feasibility_satisfied=scaled_feasibility_satisfied,
            quality_satisfied=quality_satisfied,
            first_quality_accepted_step=first_quality_accepted_step,
            quality_candidate_reached=quality_candidate_reached,
        )

    executable = (
        jax.jit(replay_accepted_quality)
        .lower(example_optimizer_ledger, example_accepted_state_mask)
        .compile()
    )
    return PreparedNeqAcceptedQualityDiagnostics(_run_quality=executable)


def _independent_endpoint_evidence(
    optimizer_result: ProjectedGaussNewtonResult,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    policy: NativeEquivalentQualityPolicy,
) -> NativeEquivalentEndpointEvidence:
    optimizer_coordinates = optimizer_result.optimizer_coordinates
    physical_state = fullspace_physical_coordinates(optimizer_coordinates, scaling)
    evaluation = evaluate_fullspace(physical_state, problem)
    raw_equalities = flatten_fullspace_constraints(evaluation.constraints)
    scaled_equalities = raw_equalities * scaling.constraint_inverse_scale
    _, objective_gradient = fullspace_value_and_grad(physical_state, problem)
    reconstruction = certify_fullspace_objective_residuals(physical_state, problem)
    transpose = deterministic_constraint_transpose_certificate(
        optimizer_coordinates,
        problem,
        scaling,
        tolerance=policy.transpose_defect_tolerance,
    )
    kkt = classify_kkt_telemetry(optimizer_result)
    component_bound = (
        jnp.abs(policy.native_raw_equalities)
        + policy.component_absolute_tolerance
        + policy.component_relative_tolerance * jnp.abs(policy.native_raw_equalities)
    )
    objective_quality_valid = jnp.isfinite(evaluation.weighted_total) & (
        evaluation.weighted_total <= policy.objective_target
    )
    equality_quality_valid = jnp.all(
        jnp.isfinite(raw_equalities) & (jnp.abs(raw_equalities) <= component_bound)
    )
    scaled_feasibility_valid = jnp.all(jnp.isfinite(scaled_equalities)) & (
        jnp.linalg.norm(scaled_equalities, ord=jnp.inf)
        <= policy.scaled_feasibility_tolerance
    )
    gradient_valid = jnp.all(jnp.isfinite(objective_gradient))
    residual_contract_valid = (
        reconstruction.residual_valid
        & reconstruction.all_finite
        & (reconstruction.value_scaled_defect <= policy.residual_value_defect_tolerance)
        & (
            reconstruction.gradient_scaled_defect
            <= policy.residual_gradient_defect_tolerance
        )
    )
    derivative_contract_valid = transpose.certified
    evaluation_all_finite = jnp.all(
        jnp.stack(
            tuple(
                jnp.all(jnp.isfinite(leaf))
                for leaf in jax.tree_util.tree_leaves(evaluation)
            )
        )
    )
    all_finite = (
        jnp.all(jnp.isfinite(physical_state))
        & evaluation_all_finite
        & jnp.all(jnp.isfinite(raw_equalities))
        & jnp.all(jnp.isfinite(scaled_equalities))
        & gradient_valid
        & reconstruction.all_finite
        & transpose.all_finite
    )
    local_audit_passed = (
        objective_quality_valid
        & equality_quality_valid
        & scaled_feasibility_valid
        & gradient_valid
        & residual_contract_valid
        & derivative_contract_valid
        & all_finite
    )
    return NativeEquivalentEndpointEvidence(
        physical_state=physical_state,
        evaluation=evaluation,
        objective_gradient=objective_gradient,
        raw_equalities=raw_equalities,
        scaled_equalities=scaled_equalities,
        residual_reconstruction=reconstruction,
        transpose_certificate=transpose,
        kkt_telemetry=kkt,
        objective_quality_valid=objective_quality_valid,
        equality_quality_valid=equality_quality_valid,
        scaled_feasibility_valid=scaled_feasibility_valid,
        gradient_valid=gradient_valid,
        residual_contract_valid=residual_contract_valid,
        derivative_contract_valid=derivative_contract_valid,
        all_finite=all_finite,
        local_audit_passed=local_audit_passed,
    )


class _PreparedNeqGntrExecution(NamedTuple):
    scaling: FullSpaceScaling
    initial_optimizer_coordinates: jax.Array
    options: ProjectedGaussNewtonOptions
    run_loop: _PreparedLoop
    finalize: _PreparedFinalize
    map_ledger: _PreparedLedgerMap


def _projected_gntr_options_sha256(
    options: ProjectedGaussNewtonOptions,
) -> str:
    return exact_numeric_tree_sha256(
        (
            "projected-gauss-newton-options-v1",
            tuple(
                (field_name, getattr(options, field_name))
                for field_name in _PROJECTED_GNTR_OPTIONS_V1_FIELDS
            ),
        )
    )


def _projected_gntr3_options_sha256(
    options: ProjectedGaussNewtonOptions,
) -> str:
    return exact_numeric_tree_sha256(
        (
            "projected-gauss-newton-options-v2",
            tuple(
                (field_name, getattr(options, field_name))
                for field_name in _PROJECTED_GNTR_OPTIONS_V2_FIELDS
            ),
        )
    )


def _prepare_neq_gntr_execution(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    policy: NativeEquivalentQualityPolicy,
    options: ProjectedGaussNewtonOptions,
) -> _PreparedNeqGntrExecution:
    """Compile one route-owned GNTR execution under an explicit option set."""

    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    if not np.array_equal(
        np.asarray(scaling.constraint_inverse_scale),
        np.asarray(policy.constraint_inverse_scale),
    ):
        raise ValueError("policy constraint scale does not match canonical scaling")
    initial_optimizer_coordinates = fullspace_optimizer_coordinates(
        initial_physical_state,
        scaling,
    )
    if initial_optimizer_coordinates.shape != (policy.state_size,):
        raise ValueError("initial optimizer coordinates do not match policy dimension")
    if initial_optimizer_coordinates.dtype != jnp.float64:
        raise TypeError("NEQ-GNTR1 requires FP64 coordinates")

    def joint_value_constraints(
        optimizer_coordinates: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return cfs_sqp1_joint_value_constraints(
            optimizer_coordinates,
            problem,
            scaling,
        )

    def objective_residuals(optimizer_coordinates: jax.Array) -> jax.Array:
        physical_state = fullspace_physical_coordinates(
            optimizer_coordinates,
            scaling,
        )
        return fullspace_objective_residual_vector(physical_state, problem)

    residual_shape = jax.eval_shape(objective_residuals, initial_optimizer_coordinates)
    if residual_shape.shape != (policy.objective_residual_size,):
        raise ValueError("objective residual vector does not match policy dimension")
    if residual_shape.dtype != jnp.float64:
        raise TypeError("NEQ-GNTR1 objective residual must be FP64")

    def route_quality(accepted_state: ProjectedGaussNewtonAcceptedState) -> jax.Array:
        return accepted_state_meets_native_quality(accepted_state, policy)

    def run_loop(coordinates: jax.Array) -> ProjectedGaussNewtonLoopResult:
        return run_projected_gauss_newton_trust_region_loop(
            joint_value_constraints,
            objective_residuals,
            coordinates,
            options=options,
            accepted_state_quality_predicate=route_quality,
        )

    def finalize(
        loop_result: ProjectedGaussNewtonLoopResult,
    ) -> tuple[ProjectedGaussNewtonResult, NativeEquivalentEndpointEvidence]:
        optimizer_result = finalize_projected_gauss_newton_trust_region(
            joint_value_constraints,
            objective_residuals,
            loop_result,
            options=options,
        )
        evidence = _independent_endpoint_evidence(
            optimizer_result,
            problem,
            scaling,
            policy,
        )
        return optimizer_result, evidence

    def map_ledger(optimizer_ledger: jax.Array) -> jax.Array:
        return scaling.bootstrap_anchor[None, :] + (
            optimizer_ledger * scaling.variable_scale[None, :]
        )

    compiled_loop = jax.jit(run_loop).lower(initial_optimizer_coordinates).compile()
    loop_shape = jax.eval_shape(run_loop, initial_optimizer_coordinates)
    compiled_finalize = jax.jit(finalize).lower(loop_shape).compile()
    ledger_shape = jax.ShapeDtypeStruct(
        (options.maximum_accepted_steps + 1, policy.state_size),
        jnp.float64,
    )
    compiled_map = jax.jit(map_ledger).lower(ledger_shape).compile()
    return _PreparedNeqGntrExecution(
        scaling=scaling,
        initial_optimizer_coordinates=initial_optimizer_coordinates,
        options=options,
        run_loop=compiled_loop,
        finalize=compiled_finalize,
        map_ledger=compiled_map,
    )


def prepare_neq_gntr1(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    policy: NativeEquivalentQualityPolicy,
) -> PreparedNeqGntr1:
    """Compile the frozen 256-accepted/300-attempt NEQ-GNTR1 route."""

    execution = _prepare_neq_gntr_execution(
        problem,
        bootstrap_state,
        initial_physical_state,
        policy,
        NEQ_GNTR1_OPTIONS,
    )
    return PreparedNeqGntr1(
        problem=problem,
        scaling=execution.scaling,
        policy=policy,
        initial_physical_state=initial_physical_state,
        initial_optimizer_coordinates=execution.initial_optimizer_coordinates,
        options=execution.options,
        _run_loop=execution.run_loop,
        _finalize=execution.finalize,
        _map_ledger=execution.map_ledger,
    )


def prepare_neq_gntr2(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    policy: NativeEquivalentQualityPolicy,
) -> PreparedNeqGntr2:
    """Compile the authoritative max-two NEQ-GNTR2 numerical route."""

    execution = _prepare_neq_gntr_execution(
        problem,
        bootstrap_state,
        initial_physical_state,
        policy,
        NEQ_GNTR2_OPTIONS,
    )
    problem_sha256 = exact_numeric_tree_sha256(problem)
    optimizer_options_sha256 = _projected_gntr_options_sha256(NEQ_GNTR2_OPTIONS)
    scaling_sha256 = exact_numeric_tree_sha256(execution.scaling)
    bootstrap_state_sha256 = exact_numeric_tree_sha256(bootstrap_state)
    initial_physical_state_sha256 = exact_numeric_tree_sha256(initial_physical_state)
    identity_payload = (
        NEQ_GNTR2_SCHEMA_VERSION,
        NEQ_GNTR2_ROUTE,
        policy.policy_sha256,
        problem_sha256,
        optimizer_options_sha256,
        scaling_sha256,
        bootstrap_state_sha256,
        initial_physical_state_sha256,
    )
    identity = NeqGntr2Identity(
        schema_version=NEQ_GNTR2_SCHEMA_VERSION,
        route=NEQ_GNTR2_ROUTE,
        base_neq_gntr1_policy_sha256=policy.policy_sha256,
        problem_sha256=problem_sha256,
        optimizer_options_sha256=optimizer_options_sha256,
        scaling_sha256=scaling_sha256,
        bootstrap_state_sha256=bootstrap_state_sha256,
        initial_physical_state_sha256=initial_physical_state_sha256,
        identity_sha256=exact_numeric_tree_sha256(identity_payload),
    )
    return PreparedNeqGntr2(
        identity=identity,
        problem=problem,
        scaling=execution.scaling,
        policy=policy,
        initial_physical_state=initial_physical_state,
        initial_optimizer_coordinates=execution.initial_optimizer_coordinates,
        options=execution.options,
        _run_loop=execution.run_loop,
        _finalize=execution.finalize,
        _map_ledger=execution.map_ledger,
    )


def prepare_neq_gntr3(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    policy: NativeEquivalentQualityPolicy,
) -> PreparedNeqGntr3:
    """Compile the safeguarded max-two NEQ-GNTR3 numerical route."""

    execution = _prepare_neq_gntr_execution(
        problem,
        bootstrap_state,
        initial_physical_state,
        policy,
        NEQ_GNTR3_OPTIONS,
    )
    problem_sha256 = exact_numeric_tree_sha256(problem)
    optimizer_options_sha256 = _projected_gntr3_options_sha256(NEQ_GNTR3_OPTIONS)
    scaling_sha256 = exact_numeric_tree_sha256(execution.scaling)
    bootstrap_state_sha256 = exact_numeric_tree_sha256(bootstrap_state)
    initial_physical_state_sha256 = exact_numeric_tree_sha256(initial_physical_state)
    identity_payload = (
        NEQ_GNTR3_SCHEMA_VERSION,
        NEQ_GNTR3_ROUTE,
        policy.policy_sha256,
        problem_sha256,
        optimizer_options_sha256,
        scaling_sha256,
        bootstrap_state_sha256,
        initial_physical_state_sha256,
    )
    identity = NeqGntr3Identity(
        schema_version=NEQ_GNTR3_SCHEMA_VERSION,
        route=NEQ_GNTR3_ROUTE,
        base_neq_gntr1_policy_sha256=policy.policy_sha256,
        problem_sha256=problem_sha256,
        optimizer_options_sha256=optimizer_options_sha256,
        scaling_sha256=scaling_sha256,
        bootstrap_state_sha256=bootstrap_state_sha256,
        initial_physical_state_sha256=initial_physical_state_sha256,
        identity_sha256=exact_numeric_tree_sha256(identity_payload),
    )
    return PreparedNeqGntr3(
        identity=identity,
        problem=problem,
        scaling=execution.scaling,
        policy=policy,
        initial_physical_state=initial_physical_state,
        initial_optimizer_coordinates=execution.initial_optimizer_coordinates,
        options=execution.options,
        _run_loop=execution.run_loop,
        _finalize=execution.finalize,
        _map_ledger=execution.map_ledger,
    )


def prepare_neq_gntr_retraction_canary(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    policy: NativeEquivalentQualityPolicy,
    *,
    options: ProjectedGaussNewtonOptions,
) -> PreparedNeqGntrRetractionCanary:
    """Compile a bounded retraction comparator with separate identity."""

    execution = _prepare_neq_gntr_execution(
        problem,
        bootstrap_state,
        initial_physical_state,
        policy,
        options,
    )
    optimizer_options_sha256 = _projected_gntr_options_sha256(options)
    problem_sha256 = exact_numeric_tree_sha256(problem)
    scaling_sha256 = exact_numeric_tree_sha256(execution.scaling)
    initial_physical_state_sha256 = exact_numeric_tree_sha256(initial_physical_state)
    identity_payload = (
        _RETRACTION_CANARY_SCHEMA_VERSION,
        _RETRACTION_CANARY_ROUTE,
        policy.policy_sha256,
        problem_sha256,
        optimizer_options_sha256,
        scaling_sha256,
        initial_physical_state_sha256,
    )
    identity = NeqGntrRetractionCanaryIdentity(
        schema_version=_RETRACTION_CANARY_SCHEMA_VERSION,
        route=_RETRACTION_CANARY_ROUTE,
        base_neq_gntr1_policy_sha256=policy.policy_sha256,
        problem_sha256=problem_sha256,
        optimizer_options_sha256=optimizer_options_sha256,
        scaling_sha256=scaling_sha256,
        initial_physical_state_sha256=initial_physical_state_sha256,
        identity_sha256=exact_numeric_tree_sha256(identity_payload),
    )
    return PreparedNeqGntrRetractionCanary(
        identity=identity,
        problem=problem,
        scaling=execution.scaling,
        policy=policy,
        initial_physical_state=initial_physical_state,
        initial_optimizer_coordinates=execution.initial_optimizer_coordinates,
        options=execution.options,
        _run_loop=execution.run_loop,
        _finalize=execution.finalize,
        _map_ledger=execution.map_ledger,
    )


__all__ = (
    "NATIVE_OBJECTIVE_TARGET",
    "NEQ_GNTR1_EQUALITY_SIZE",
    "NEQ_GNTR1_OBJECTIVE_RESIDUAL_SIZE",
    "NEQ_GNTR1_OPTIONS",
    "NEQ_GNTR1_STATE_SIZE",
    "NEQ_GNTR2_OPTIONS",
    "NEQ_GNTR2_ROUTE",
    "NEQ_GNTR2_SCHEMA_VERSION",
    "NEQ_GNTR3_OPTIONS",
    "NEQ_GNTR3_ROUTE",
    "NEQ_GNTR3_SCHEMA_VERSION",
    "ROUTE",
    "SCHEMA_VERSION",
    "KktTelemetry",
    "KktTelemetryStatus",
    "NativeEquivalentAcceptedQualityReplay",
    "NativeEquivalentEndpointEvidence",
    "NativeEquivalentQualityMargins",
    "NativeEquivalentQualityPolicy",
    "NativeEquivalentQualityResult",
    "NativeEquivalentTerminalDiagnosticResult",
    "NativeEquivalentTerminalEndpointEvidence",
    "NeqGntr2Identity",
    "NeqGntr2Result",
    "NeqGntr3Identity",
    "NeqGntr3Result",
    "NeqGntrRetractionCanaryIdentity",
    "NeqGntrRetractionCanaryResult",
    "PreparedNeqAcceptedQualityDiagnostics",
    "PreparedNeqGntr1",
    "PreparedNeqGntr2",
    "PreparedNeqGntr3",
    "PreparedNeqGntrRetractionCanary",
    "PreparedNeqTerminalEndpointDiagnostics",
    "accepted_state_meets_native_quality",
    "build_native_equivalent_terminal_diagnostic",
    "classify_kkt_telemetry",
    "deterministic_constraint_transpose_certificate",
    "native_equivalent_quality_margins",
    "prepare_neq_accepted_quality_diagnostics",
    "prepare_neq_gntr1",
    "prepare_neq_gntr2",
    "prepare_neq_gntr3",
    "prepare_neq_gntr_retraction_canary",
    "prepare_neq_terminal_endpoint_diagnostics",
)
