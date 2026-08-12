"""Independent endpoint audit for the NEQ-GNTR1 performance route.

The timed solver result is evidence, not authority.  This module rebinds the
latched state to the accepted ledger, replays its native Boozer branch, and
evaluates both native and JAX physics at both the GPU and native endpoints.
KKT values are deliberately outside this engineering-quality boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.objectives.single_stage_fullspace import (
    FROZEN_PROBLEM_CONTRACT,
    TERM_LEDGER,
    FullSpaceEvaluation,
    FullSpaceProblem,
    TermClassification,
    evaluate_fullspace,
    flatten_fullspace_constraints,
    fullspace_objective_residual_vector,
    fullspace_value_and_grad,
)
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    certify_fullspace_objective_residuals,
)
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NATIVE_OBJECTIVE_TARGET,
    NativeEquivalentQualityPolicy,
    NativeEquivalentQualityResult,
    deterministic_constraint_transpose_certificate,
)
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    BRANCH_ROOT_TOLERANCE,
    SCALED_BOOZER_TOLERANCE,
    NativeAcceptedIntervalEvidence,
    NativeAcceptedPathEvidence,
    NativeContinuationStep,
    NativeExplicitStateEvaluation,
    NativeFullSpaceObjectiveContract,
    NativeSingleStageEndpointRuntime,
)

from benchmarks.single_stage_fullspace_snapshot import (
    JsonValue,
    canonical_json_bytes,
    load_canonical_json_bytes,
)

SCHEMA_VERSION: Final = "single-stage-neq-gntr1-endpoint-audit-v1"
PLAN_SHA256: Final = "d082baa587b9db580ac3ef8c99a3123ed83564586b605200f7c2cfa6feb909a9"
_STATE_SIZE: Final = 716
_EQUALITY_SIZE: Final = 255
_RAW_TERM_NAMES: Final = ("non_qs", "residual", "iota", "major_radius", "length")
_OBSERVABLE_NAMES: Final = (
    "iota",
    "G",
    "volume",
    "major_radius",
    "total_length",
    "non_qs_ratio",
    "boozer_residual_value",
    "boozer_residual_rms",
)
_JSON_STRING_FIELDS: Final = frozenset(
    {
        "schema_version",
        "gradient_dtype",
        "jvp_dtype",
        "vjp_dtype",
        "state_dtype",
        "native_state_dtype",
        "native_equality_dtype",
        "jax_equality_dtype",
        "failure_reason",
    }
)
_JSON_BOOLEAN_FIELDS: Final = frozenset(
    {"usable", "inner_solver_success", "weight_inv_modB"}
)


def _fp64(value: object) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(_fp64(value), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: object) -> JsonValue:
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported endpoint-audit payload value: {type(value).__name__}")


def _finite_fp64(value: object, shape: tuple[int, ...]) -> bool:
    array = _fp64(value)
    return bool(
        array.dtype == np.dtype(np.float64)
        and array.shape == shape
        and np.all(np.isfinite(array))
    )


def _deterministic_state_probe() -> np.ndarray:
    indices = np.arange(1, _STATE_SIZE + 1, dtype=np.float64)
    probe = np.sin(indices)
    return probe / np.linalg.norm(probe)


def _deterministic_equality_probe() -> np.ndarray:
    indices = np.arange(1, _EQUALITY_SIZE + 1, dtype=np.float64)
    probe = np.cos(np.float64(0.5) * indices)
    return probe / np.linalg.norm(probe)


@dataclass(frozen=True, slots=True)
class LatchedStateBindingEvidence:
    accepted_step_count: int
    valid_row_count: int
    row_capacity: int
    mask_true_count: int
    observed_mask_sha256: str
    expected_mask_sha256: str
    optimizer_ledger_sha256: str
    physical_ledger_sha256: str
    recomputed_physical_ledger_sha256: str
    valid_physical_row_sha256s: tuple[str, ...]
    bootstrap_state_sha256: str
    expected_bootstrap_state_sha256: str
    latched_state_sha256: str
    latched_optimizer_row_sha256: str
    loop_latched_optimizer_state_sha256: str
    finalized_optimizer_state_sha256: str
    endpoint_physical_state_sha256: str
    observed_mask: np.ndarray
    expected_mask: np.ndarray
    optimizer_ledger: np.ndarray
    physical_ledger: np.ndarray
    recomputed_physical_ledger: np.ndarray
    bootstrap_anchor: np.ndarray
    variable_scale: np.ndarray

    def passes(self) -> bool:
        raw_valid = bool(
            self.observed_mask.dtype == np.dtype(np.bool_)
            and self.expected_mask.dtype == np.dtype(np.bool_)
            and self.observed_mask.shape == (257,)
            and self.expected_mask.shape == (257,)
            and _finite_fp64(self.optimizer_ledger, (257, _STATE_SIZE))
            and _finite_fp64(self.physical_ledger, (257, _STATE_SIZE))
            and _finite_fp64(self.recomputed_physical_ledger, (257, _STATE_SIZE))
            and _finite_fp64(self.bootstrap_anchor, (_STATE_SIZE,))
            and _finite_fp64(self.variable_scale, (_STATE_SIZE,))
            and self.observed_mask_sha256 == _array_sha256(self.observed_mask)
            and self.expected_mask_sha256 == _array_sha256(self.expected_mask)
            and self.optimizer_ledger_sha256 == _array_sha256(self.optimizer_ledger)
            and self.physical_ledger_sha256 == _array_sha256(self.physical_ledger)
            and self.recomputed_physical_ledger_sha256
            == _array_sha256(self.recomputed_physical_ledger)
        )
        recomputed_row_hashes = tuple(
            _array_sha256(row) for row in self.physical_ledger[: self.valid_row_count]
        )
        independently_recomputed_physical = self.bootstrap_anchor[None, :] + (
            self.optimizer_ledger * self.variable_scale[None, :]
        )
        return bool(
            raw_valid
            and 1 <= self.accepted_step_count <= 256
            and self.valid_row_count == self.accepted_step_count + 1
            and self.row_capacity == 257
            and self.mask_true_count == self.valid_row_count
            and self.observed_mask_sha256 == self.expected_mask_sha256
            and np.array_equal(
                self.expected_mask,
                np.arange(self.row_capacity) < self.valid_row_count,
            )
            and np.array_equal(
                self.optimizer_ledger[0],
                np.zeros((_STATE_SIZE,), dtype=np.float64),
            )
            and self.physical_ledger_sha256 == self.recomputed_physical_ledger_sha256
            and np.array_equal(
                self.recomputed_physical_ledger,
                independently_recomputed_physical,
            )
            and self.valid_physical_row_sha256s == recomputed_row_hashes
            and self.valid_physical_row_sha256s[0] == self.bootstrap_state_sha256
            and self.valid_physical_row_sha256s[-1] == self.latched_state_sha256
            and self.bootstrap_state_sha256 == self.expected_bootstrap_state_sha256
            and self.latched_state_sha256 == self.endpoint_physical_state_sha256
            and self.loop_latched_optimizer_state_sha256
            == self.finalized_optimizer_state_sha256
            and self.loop_latched_optimizer_state_sha256
            == self.latched_optimizer_row_sha256
        )


@dataclass(frozen=True, slots=True)
class DerivativeResidualEvidence:
    gradient_dtype: str
    gradient_size: int
    gradient_nonfinite_count: int
    jvp_dtype: str
    jvp_size: int
    jvp_nonfinite_count: int
    vjp_dtype: str
    vjp_size: int
    vjp_nonfinite_count: int
    residual_value_defect: float
    residual_gradient_defect: float
    transpose_primal_dot: float
    transpose_adjoint_dot: float
    transpose_denominator: float
    transpose_defect: float
    gradient: tuple[float, ...]
    state_probe: tuple[float, ...]
    equality_probe: tuple[float, ...]
    jvp_action: tuple[float, ...]
    vjp_action: tuple[float, ...]
    reconstructed_objective_value: float
    authoritative_objective_value: float
    reconstructed_objective_gradient: tuple[float, ...]
    authoritative_objective_gradient: tuple[float, ...]

    def passes(self) -> bool:
        gradient = np.asarray(self.gradient, dtype=np.float64)
        state_probe = np.asarray(self.state_probe, dtype=np.float64)
        equality_probe = np.asarray(self.equality_probe, dtype=np.float64)
        jvp_action = np.asarray(self.jvp_action, dtype=np.float64)
        vjp_action = np.asarray(self.vjp_action, dtype=np.float64)
        reconstructed_gradient = np.asarray(
            self.reconstructed_objective_gradient, dtype=np.float64
        )
        authoritative_gradient = np.asarray(
            self.authoritative_objective_gradient, dtype=np.float64
        )
        value_scale = max(
            1.0,
            abs(self.reconstructed_objective_value),
            abs(self.authoritative_objective_value),
        )
        recomputed_value_defect = (
            abs(self.reconstructed_objective_value - self.authoritative_objective_value)
            / value_scale
        )
        gradient_scale = max(
            1.0,
            float(np.linalg.norm(reconstructed_gradient, ord=np.inf)),
            float(np.linalg.norm(authoritative_gradient, ord=np.inf)),
        )
        recomputed_gradient_defect = float(
            np.linalg.norm(
                reconstructed_gradient - authoritative_gradient,
                ord=np.inf,
            )
            / gradient_scale
        )
        recomputed_primal_dot = float(np.vdot(equality_probe, jvp_action))
        recomputed_adjoint_dot = float(np.vdot(state_probe, vjp_action))
        recomputed_denominator = max(
            np.finfo(np.float64).tiny,
            float(
                np.linalg.norm(equality_probe) * np.linalg.norm(jvp_action)
                + np.linalg.norm(state_probe) * np.linalg.norm(vjp_action)
            ),
        )
        recomputed_transpose_defect = (
            abs(recomputed_primal_dot - recomputed_adjoint_dot) / recomputed_denominator
        )
        scalars = np.asarray(
            (
                self.residual_value_defect,
                self.residual_gradient_defect,
                self.transpose_primal_dot,
                self.transpose_adjoint_dot,
                self.transpose_denominator,
                self.transpose_defect,
                self.reconstructed_objective_value,
                self.authoritative_objective_value,
            ),
            dtype=np.float64,
        )
        return bool(
            self.gradient_dtype == "float64"
            and gradient.shape == (_STATE_SIZE,)
            and self.gradient_size == gradient.size
            and self.gradient_nonfinite_count
            == gradient.size - np.count_nonzero(np.isfinite(gradient))
            and self.jvp_dtype == "float64"
            and jvp_action.shape == (_EQUALITY_SIZE,)
            and self.jvp_size == jvp_action.size
            and self.jvp_nonfinite_count
            == jvp_action.size - np.count_nonzero(np.isfinite(jvp_action))
            and self.vjp_dtype == "float64"
            and vjp_action.shape == (_STATE_SIZE,)
            and self.vjp_size == vjp_action.size
            and self.vjp_nonfinite_count
            == vjp_action.size - np.count_nonzero(np.isfinite(vjp_action))
            and state_probe.shape == (_STATE_SIZE,)
            and equality_probe.shape == (_EQUALITY_SIZE,)
            and reconstructed_gradient.shape == (_STATE_SIZE,)
            and authoritative_gradient.shape == (_STATE_SIZE,)
            and np.all(np.isfinite(gradient))
            and np.all(np.isfinite(state_probe))
            and np.all(np.isfinite(equality_probe))
            and np.all(np.isfinite(jvp_action))
            and np.all(np.isfinite(vjp_action))
            and np.all(np.isfinite(reconstructed_gradient))
            and np.all(np.isfinite(authoritative_gradient))
            and np.array_equal(gradient, authoritative_gradient)
            and np.all(np.isfinite(scalars))
            and self.transpose_denominator > 0.0
            and self.residual_value_defect == recomputed_value_defect
            and self.residual_gradient_defect == recomputed_gradient_defect
            and recomputed_value_defect <= 1.0e-12
            and recomputed_gradient_defect <= 1.0e-10
            and self.transpose_primal_dot == recomputed_primal_dot
            and self.transpose_adjoint_dot == recomputed_adjoint_dot
            and self.transpose_denominator == recomputed_denominator
            and np.isclose(
                self.transpose_defect,
                recomputed_transpose_defect,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            and recomputed_transpose_defect <= 1.0e-10
            and np.array_equal(
                state_probe,
                _deterministic_state_probe(),
            )
            and np.array_equal(
                equality_probe,
                _deterministic_equality_probe(),
            )
        )


@dataclass(frozen=True, slots=True)
class GpuQualityEvidence:
    physical_objective: float
    gpu_raw_objective_terms: tuple[float, ...]
    objective_weights: tuple[float, ...]
    gpu_raw_equalities: tuple[float, ...]
    native_raw_equalities: tuple[float, ...]
    policy_native_raw_equalities: tuple[float, ...]
    constraint_inverse_scale: tuple[float, ...]
    objective_target: float
    component_absolute_tolerance: float
    component_relative_tolerance: float
    scaled_feasibility_tolerance: float
    reported_scaled_feasibility_infinity_norm: float

    def passes(self) -> bool:
        gpu = np.asarray(self.gpu_raw_equalities, dtype=np.float64)
        raw_terms = np.asarray(self.gpu_raw_objective_terms, dtype=np.float64)
        weights = np.asarray(self.objective_weights, dtype=np.float64)
        native = np.asarray(self.native_raw_equalities, dtype=np.float64)
        policy_native = np.asarray(
            self.policy_native_raw_equalities,
            dtype=np.float64,
        )
        scale = np.asarray(self.constraint_inverse_scale, dtype=np.float64)
        scalars = np.asarray(
            (
                self.physical_objective,
                self.objective_target,
                self.component_absolute_tolerance,
                self.component_relative_tolerance,
                self.scaled_feasibility_tolerance,
                self.reported_scaled_feasibility_infinity_norm,
            ),
            dtype=np.float64,
        )
        scaled = gpu * scale
        scaled_feasibility = float(np.linalg.norm(scaled, ord=np.inf))
        component_bound = (
            np.abs(native)
            + self.component_absolute_tolerance
            + self.component_relative_tolerance * np.abs(native)
        )
        recomputed_objective = float(np.vdot(weights, raw_terms))
        return bool(
            gpu.shape == (_EQUALITY_SIZE,)
            and native.shape == (_EQUALITY_SIZE,)
            and policy_native.shape == (_EQUALITY_SIZE,)
            and scale.shape == (_EQUALITY_SIZE,)
            and raw_terms.shape == (5,)
            and weights.shape == (5,)
            and np.all(np.isfinite(gpu))
            and np.all(np.isfinite(native))
            and np.all(np.isfinite(policy_native))
            and np.all(np.isfinite(scale))
            and np.all(np.isfinite(raw_terms))
            and np.all(np.isfinite(weights))
            and np.all(scale != 0.0)
            and np.all(np.isfinite(scalars))
            and np.all(scalars[2:5] >= 0.0)
            and self.objective_target == NATIVE_OBJECTIVE_TARGET
            and self.component_absolute_tolerance == 1.0e-12
            and self.component_relative_tolerance == 1.0e-10
            and self.scaled_feasibility_tolerance == 1.0e-10
            and self.physical_objective == recomputed_objective
            and recomputed_objective <= self.objective_target
            and np.array_equal(native, policy_native)
            and np.all(np.abs(gpu) <= component_bound)
            and self.reported_scaled_feasibility_infinity_norm == scaled_feasibility
            and scaled_feasibility <= self.scaled_feasibility_tolerance
        )


@dataclass(frozen=True, slots=True)
class SameStateCrossEvaluationEvidence:
    requested_state: tuple[float, ...]
    native_returned_state: tuple[float, ...]
    requested_state_sha256: str
    native_returned_state_sha256: str
    native_objective: float
    jax_objective: float
    native_raw_terms: tuple[float, ...]
    jax_raw_terms: tuple[float, ...]
    native_raw_equalities: tuple[float, ...]
    jax_raw_equalities: tuple[float, ...]
    native_observables: tuple[float, ...]
    jax_observables: tuple[float, ...]
    state_dtype: str
    state_size: int
    state_nonfinite_count: int
    native_state_dtype: str
    native_state_size: int
    native_state_nonfinite_count: int
    native_equality_dtype: str
    jax_equality_dtype: str
    jax_scalar_dtypes: tuple[str, ...]

    def passes(self) -> bool:
        requested_state = np.asarray(self.requested_state, dtype=np.float64)
        native_returned_state = np.asarray(self.native_returned_state, dtype=np.float64)
        native_terms = np.asarray(self.native_raw_terms, dtype=np.float64)
        jax_terms = np.asarray(self.jax_raw_terms, dtype=np.float64)
        native_equalities = np.asarray(self.native_raw_equalities, dtype=np.float64)
        jax_equalities = np.asarray(self.jax_raw_equalities, dtype=np.float64)
        native_observables = np.asarray(self.native_observables, dtype=np.float64)
        jax_observables = np.asarray(self.jax_observables, dtype=np.float64)
        scalar_values = np.concatenate(
            (
                np.asarray((self.native_objective, self.jax_objective)),
                native_terms,
                jax_terms,
                native_observables,
                jax_observables,
            )
        )
        return bool(
            requested_state.shape == (_STATE_SIZE,)
            and native_returned_state.shape == (_STATE_SIZE,)
            and np.all(np.isfinite(requested_state))
            and np.all(np.isfinite(native_returned_state))
            and self.requested_state_sha256 == _array_sha256(requested_state)
            and self.native_returned_state_sha256
            == _array_sha256(native_returned_state)
            and np.array_equal(requested_state, native_returned_state)
            and self.state_dtype == "float64"
            and self.state_size == _STATE_SIZE
            and self.state_nonfinite_count == 0
            and self.native_state_dtype == "float64"
            and self.native_state_size == _STATE_SIZE
            and self.native_state_nonfinite_count == 0
            and self.native_equality_dtype == "float64"
            and self.jax_equality_dtype == "float64"
            and len(native_equalities) == _EQUALITY_SIZE
            and len(jax_equalities) == _EQUALITY_SIZE
            and all(dtype == "float64" for dtype in self.jax_scalar_dtypes)
            and np.all(np.isfinite(scalar_values))
            and np.all(np.isfinite(native_equalities))
            and np.all(np.isfinite(jax_equalities))
            and np.isclose(
                self.native_objective,
                self.jax_objective,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            and _maximum_tolerance_ratio(
                native_terms, jax_terms, rtol=1.0e-12, atol=1.0e-15
            )
            <= 1.0
            and _maximum_tolerance_ratio(
                native_equalities, jax_equalities, rtol=1.0e-10, atol=1.0e-12
            )
            <= 1.0
            and _maximum_tolerance_ratio(
                native_observables, jax_observables, rtol=1.0e-12, atol=1.0e-15
            )
            <= 1.0
        )


@dataclass(frozen=True, slots=True)
class ObjectiveContractEvidence:
    iota_target: float
    major_radius_target: float
    length_target: float
    volume_target: float
    non_qs_weight: float
    residual_weight: float
    iota_weight: float
    major_radius_weight: float
    length_weight: float
    non_qs_axis: int
    weight_inv_modB: bool


@dataclass(frozen=True, slots=True)
class PhysicsContractEvidence:
    state_ordering: tuple[str, ...]
    endpoint_state_sha256: str
    repacked_endpoint_state_sha256: str
    coil_dof_count: int
    surface_dof_count: int
    scalar_dof_count: int
    total_dof_count: int
    equality_count: int
    length_coil_indices: tuple[int, ...]
    objective_term_ids: tuple[str, ...]
    equality_term_ids: tuple[str, ...]
    fixed_term_ids: tuple[str, ...]
    term_ledger_sha256: str
    objective_contract_sha256: str
    native_objective_contract_sha256: str
    exact_mask_indices_sha256: str
    native_exact_mask_indices_sha256: str
    fixed_first_base_current: float
    gpu_native_fixed_first_base_current: float
    reference_native_fixed_first_base_current: float
    jax_objective_contract: ObjectiveContractEvidence
    native_objective_contract: ObjectiveContractEvidence
    exact_mask_indices: np.ndarray
    native_exact_mask_indices: np.ndarray

    def passes(self) -> bool:
        frozen = FROZEN_PROBLEM_CONTRACT.layout
        objective_rows = tuple(
            row
            for row in TERM_LEDGER
            if row.classification
            in {TermClassification.OBJECTIVE, TermClassification.OBJECTIVE_PENALTY}
        )
        masks_valid = bool(
            self.exact_mask_indices.dtype == np.dtype(np.int64)
            and self.native_exact_mask_indices.dtype == np.dtype(np.int64)
            and self.exact_mask_indices.shape == (254,)
            and self.native_exact_mask_indices.shape == (254,)
            and np.array_equal(self.exact_mask_indices, self.native_exact_mask_indices)
            and np.all(np.diff(self.exact_mask_indices) > 0)
            and self.exact_mask_indices_sha256 == _array_sha256(self.exact_mask_indices)
            and self.native_exact_mask_indices_sha256
            == _array_sha256(self.native_exact_mask_indices)
        )
        return bool(
            self.state_ordering == frozen.ordering
            and self.endpoint_state_sha256 == self.repacked_endpoint_state_sha256
            and self.coil_dof_count == frozen.coil_dof_count
            and self.surface_dof_count == frozen.surface_dof_count
            and self.scalar_dof_count == frozen.scalar_dof_count
            and self.total_dof_count == frozen.total_dof_count
            and self.equality_count == frozen.equality_count
            and self.length_coil_indices == (0, 1, 2)
            and self.jax_objective_contract == self.native_objective_contract
            and (
                self.jax_objective_contract.non_qs_weight,
                self.jax_objective_contract.residual_weight,
                self.jax_objective_contract.iota_weight,
                self.jax_objective_contract.major_radius_weight,
                self.jax_objective_contract.length_weight,
            )
            == tuple(row.weight for row in objective_rows)
            and self.jax_objective_contract.non_qs_axis == 0
            and not self.jax_objective_contract.weight_inv_modB
            and self.objective_term_ids
            == tuple(
                row.term_id
                for row in TERM_LEDGER
                if row.classification
                in {TermClassification.OBJECTIVE, TermClassification.OBJECTIVE_PENALTY}
            )
            and self.equality_term_ids
            == tuple(
                row.term_id
                for row in TERM_LEDGER
                if row.classification is TermClassification.EQUALITY
            )
            and self.fixed_term_ids
            == tuple(
                row.term_id
                for row in TERM_LEDGER
                if row.classification is TermClassification.FIXED_STATE
            )
            and self.term_ledger_sha256
            == _canonical_sha256([asdict(row) for row in TERM_LEDGER])
            and self.objective_contract_sha256 == self.native_objective_contract_sha256
            and self.objective_contract_sha256
            == _canonical_sha256(asdict(self.jax_objective_contract))
            and self.native_objective_contract_sha256
            == _canonical_sha256(asdict(self.native_objective_contract))
            and self.exact_mask_indices_sha256 == self.native_exact_mask_indices_sha256
            and masks_valid
            and self.fixed_first_base_current
            == self.gpu_native_fixed_first_base_current
            == self.reference_native_fixed_first_base_current
        )


@dataclass(frozen=True, slots=True)
class BranchReplayAuditEvidence:
    raw: NativeAcceptedPathEvidence
    evidence_sha256: str
    replayed_row_count: int
    successful_direct_rows: int
    successful_midpoint_refined_rows: int
    maximum_direct_refined_difference: float
    maximum_gpu_native_difference: float
    maximum_scaled_boozer_feasibility: float
    accepted_state_sha256s: tuple[str, ...]
    accepted_coil_sha256s: tuple[str, ...]
    accepted_root_sha256s: tuple[str, ...]
    accepted_roots: tuple[tuple[float, ...], ...]
    midpoint_coil_sha256s: tuple[str, ...]

    def passes(self, valid_row_count: int) -> bool:
        interval_count = valid_row_count - 1
        if (
            len(self.raw.intervals) != interval_count
            or len(self.accepted_state_sha256s) != valid_row_count
            or len(self.accepted_coil_sha256s) != valid_row_count
            or len(self.accepted_root_sha256s) != valid_row_count
            or len(self.accepted_roots) != valid_row_count
            or len(self.midpoint_coil_sha256s) != interval_count
        ):
            return False
        bootstrap = self.raw.bootstrap_step
        accepted_root_arrays = tuple(
            np.asarray(root, dtype=np.float64) for root in self.accepted_roots
        )
        steps_finite = all(
            root.shape == (255,)
            and np.all(np.isfinite(root))
            and _array_sha256(root) == expected_sha256
            for root, expected_sha256 in zip(
                accepted_root_arrays,
                self.accepted_root_sha256s,
                strict=True,
            )
        ) and _continuation_step_passes(
            bootstrap,
            segment_count=interval_count,
            index=0,
            predecessor_index=None,
            coil_sha256=self.accepted_coil_sha256s[0],
            seed_root_sha256=self.accepted_root_sha256s[0],
            root_sha256=self.accepted_root_sha256s[0],
            maximum_iterations=0,
        )
        previous_root_sha256 = self.accepted_root_sha256s[0]
        direct_refined_differences: list[float] = []
        gpu_native_differences: list[float] = []
        for index, interval in enumerate(self.raw.intervals, start=1):
            direct_root_sha256 = _array_sha256(interval.direct_root)
            midpoint_root_sha256 = _array_sha256(interval.midpoint_root)
            refined_root_sha256 = _array_sha256(interval.refined_root)
            interval_valid = bool(
                interval.index == index
                and interval.supplied_state_little_endian_sha256
                == self.accepted_state_sha256s[index]
                and _finite_fp64(interval.direct_root, (255,))
                and _finite_fp64(interval.midpoint_root, (255,))
                and _finite_fp64(interval.refined_root, (255,))
                and _continuation_step_passes(
                    interval.direct_step,
                    segment_count=interval_count,
                    index=index,
                    predecessor_index=index - 1,
                    coil_sha256=self.accepted_coil_sha256s[index],
                    seed_root_sha256=previous_root_sha256,
                    root_sha256=direct_root_sha256,
                )
                and _continuation_step_passes(
                    interval.midpoint_step,
                    segment_count=2 * interval_count,
                    index=2 * index - 1,
                    predecessor_index=2 * index - 2,
                    coil_sha256=self.midpoint_coil_sha256s[index - 1],
                    seed_root_sha256=previous_root_sha256,
                    root_sha256=midpoint_root_sha256,
                )
                and _continuation_step_passes(
                    interval.refined_step,
                    segment_count=2 * interval_count,
                    index=2 * index,
                    predecessor_index=2 * index - 1,
                    coil_sha256=self.accepted_coil_sha256s[index],
                    seed_root_sha256=midpoint_root_sha256,
                    root_sha256=refined_root_sha256,
                )
                and refined_root_sha256 == self.accepted_root_sha256s[index]
                and np.isfinite(interval.direct_refined_infinity_difference)
                and np.isfinite(interval.supplied_refined_infinity_difference)
            )
            accepted_root = accepted_root_arrays[index]
            direct_refined_difference = float(
                np.max(np.abs(interval.direct_root - interval.refined_root))
            )
            gpu_native_difference = float(
                np.max(np.abs(accepted_root - interval.refined_root))
            )
            interval_valid = bool(
                interval_valid
                and interval.direct_refined_infinity_difference
                == direct_refined_difference
                and interval.supplied_refined_infinity_difference
                == gpu_native_difference
            )
            steps_finite = steps_finite and interval_valid
            direct_refined_differences.append(direct_refined_difference)
            gpu_native_differences.append(gpu_native_difference)
            previous_root_sha256 = refined_root_sha256
        raw_steps = (bootstrap,) + tuple(
            step
            for interval in self.raw.intervals
            for step in (
                interval.direct_step,
                interval.midpoint_step,
                interval.refined_step,
            )
        )
        maximum_scaled = max(
            (step.scaled_boozer_infinity_norm for step in raw_steps),
            default=float("inf"),
        )
        maximum_direct_refined = max(direct_refined_differences, default=0.0)
        maximum_gpu_native = max(gpu_native_differences, default=0.0)
        return bool(
            self.raw.first_failing_index is None
            and self.raw.failure_reason is None
            and self.evidence_sha256 == _canonical_sha256(_branch_payload(self.raw))
            and steps_finite
            and self.replayed_row_count == valid_row_count
            and self.successful_direct_rows == valid_row_count
            and self.successful_midpoint_refined_rows == valid_row_count - 1
            and self.maximum_direct_refined_difference == maximum_direct_refined
            and self.maximum_gpu_native_difference == maximum_gpu_native
            and self.maximum_scaled_boozer_feasibility == maximum_scaled
            and maximum_direct_refined <= BRANCH_ROOT_TOLERANCE
            and maximum_gpu_native <= BRANCH_ROOT_TOLERANCE
            and maximum_scaled <= SCALED_BOOZER_TOLERANCE
        )


@dataclass(frozen=True, slots=True)
class NativeReducedEndpointTelemetry:
    parameter_sha256: str
    objective: float
    gradient: tuple[float, ...]
    gradient_dtype: str
    gradient_infinity_norm: float
    gradient_l2_norm: float
    solver_residual_l2: float
    solver_residual_infinity_norm: float
    inner_solver_success: bool

    def passes(self, expected_parameter_sha256: str) -> bool:
        gradient = np.asarray(self.gradient, dtype=np.float64)
        scalars = np.asarray(
            (
                self.objective,
                self.gradient_infinity_norm,
                self.gradient_l2_norm,
                self.solver_residual_l2,
                self.solver_residual_infinity_norm,
            )
        )
        return bool(
            self.parameter_sha256 == expected_parameter_sha256
            and self.gradient_dtype == "float64"
            and gradient.shape == (461,)
            and np.all(np.isfinite(gradient))
            and np.all(np.isfinite(scalars))
            and np.isclose(
                self.gradient_infinity_norm,
                np.linalg.norm(gradient, ord=np.inf),
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            and np.isclose(
                self.gradient_l2_norm,
                np.linalg.norm(gradient),
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        )


@dataclass(frozen=True, slots=True)
class NativeEquivalentEndpointAudit:
    schema_version: str
    plan_sha256: str
    audited_state_sha256: str
    audited_coil_parameters_sha256: str
    binding: LatchedStateBindingEvidence
    gpu_quality: GpuQualityEvidence
    derivative_residual: DerivativeResidualEvidence
    physics_contract: PhysicsContractEvidence
    branch_replay: BranchReplayAuditEvidence
    gpu_endpoint_cross_evaluation: SameStateCrossEvaluationEvidence
    native_endpoint_cross_evaluation: SameStateCrossEvaluationEvidence
    native_reference_state_sha256: str
    native_reduced_endpoint: NativeReducedEndpointTelemetry

    def passes(self) -> bool:
        valid_physical = self.binding.physical_ledger[: self.binding.valid_row_count]
        valid_coils = valid_physical[:, :461]
        recomputed_midpoint_coil_hashes = tuple(
            _array_sha256(
                valid_coils[index - 1]
                + np.float64(0.5) * (valid_coils[index] - valid_coils[index - 1])
            )
            for index in range(1, valid_coils.shape[0])
        )
        return bool(
            self.schema_version == SCHEMA_VERSION
            and self.plan_sha256 == PLAN_SHA256
            and self.audited_state_sha256 == self.binding.latched_state_sha256
            and self.binding.passes()
            and self.gpu_quality.passes()
            and self.derivative_residual.passes()
            and self.physics_contract.passes()
            and self.branch_replay.passes(self.binding.valid_row_count)
            and self.branch_replay.accepted_state_sha256s
            == self.binding.valid_physical_row_sha256s
            and self.branch_replay.accepted_coil_sha256s
            == tuple(_array_sha256(row) for row in valid_coils)
            and self.branch_replay.accepted_root_sha256s
            == tuple(_array_sha256(row) for row in valid_physical[:, 461:])
            and self.branch_replay.midpoint_coil_sha256s
            == recomputed_midpoint_coil_hashes
            and self.gpu_endpoint_cross_evaluation.passes()
            and self.native_endpoint_cross_evaluation.passes()
            and self.gpu_quality.gpu_raw_objective_terms
            == self.gpu_endpoint_cross_evaluation.jax_raw_terms
            and self.gpu_quality.gpu_raw_equalities
            == self.gpu_endpoint_cross_evaluation.jax_raw_equalities
            and self.gpu_quality.native_raw_equalities
            == self.native_endpoint_cross_evaluation.native_raw_equalities
            and self.gpu_quality.objective_weights
            == (
                self.physics_contract.jax_objective_contract.non_qs_weight,
                self.physics_contract.jax_objective_contract.residual_weight,
                self.physics_contract.jax_objective_contract.iota_weight,
                self.physics_contract.jax_objective_contract.major_radius_weight,
                self.physics_contract.jax_objective_contract.length_weight,
            )
            and self.gpu_endpoint_cross_evaluation.requested_state_sha256
            == self.audited_state_sha256
            and self.native_endpoint_cross_evaluation.requested_state_sha256
            == self.native_reference_state_sha256
            and self.native_reduced_endpoint.passes(self.audited_coil_parameters_sha256)
        )


def _continuation_step_passes(
    step: NativeContinuationStep,
    *,
    segment_count: int,
    index: int,
    predecessor_index: int | None,
    coil_sha256: str,
    seed_root_sha256: str,
    root_sha256: str,
    maximum_iterations: int = 20,
) -> bool:
    raw_equalities = np.asarray(step.raw_equalities)
    recomputed_l2 = float(np.linalg.norm(raw_equalities))
    recomputed_infinity = float(np.linalg.norm(raw_equalities, ord=np.inf))
    recomputed_scaled_boozer = float(
        np.linalg.norm(raw_equalities[:254], ord=np.inf) / np.sqrt(254.0)
    )
    scalars = np.asarray(
        (
            step.residual_l2,
            step.residual_infinity_norm,
            step.scaled_boozer_infinity_norm,
        ),
        dtype=np.float64,
    )
    return bool(
        step.segment_count == segment_count
        and step.index == index
        and step.predecessor_index == predecessor_index
        and step.coil_little_endian_sha256 == coil_sha256
        and step.seed_root_little_endian_sha256 == seed_root_sha256
        and step.root_little_endian_sha256 == root_sha256
        and 0 <= step.newton_iterations <= maximum_iterations
        and raw_equalities.dtype == np.dtype(np.float64)
        and raw_equalities.shape == (255,)
        and np.all(np.isfinite(raw_equalities))
        and step.raw_equalities_little_endian_sha256 == _array_sha256(raw_equalities)
        and np.all(np.isfinite(scalars))
        and step.residual_l2 == recomputed_l2
        and step.residual_infinity_norm == recomputed_infinity
        and step.scaled_boozer_infinity_norm == recomputed_scaled_boozer
        and recomputed_scaled_boozer <= SCALED_BOOZER_TOLERANCE
    )


def _objective_contract_payload(problem: FullSpaceProblem) -> dict[str, object]:
    config = problem.config
    return {
        "iota_target": float(config.iota_target),
        "major_radius_target": float(config.major_radius_target),
        "length_target": float(config.length_target),
        "volume_target": float(config.volume_target),
        "non_qs_weight": float(config.non_qs_weight),
        "residual_weight": float(config.residual_weight),
        "iota_weight": float(config.iota_weight),
        "major_radius_weight": float(config.major_radius_weight),
        "length_weight": float(config.length_weight),
        "non_qs_axis": config.non_qs_axis,
        "weight_inv_modB": config.weight_inv_modB,
    }


def _native_objective_contract_payload(
    contract: NativeFullSpaceObjectiveContract,
) -> dict[str, object]:
    return asdict(contract)


def _jax_objective_contract(problem: FullSpaceProblem) -> ObjectiveContractEvidence:
    config = problem.config
    return ObjectiveContractEvidence(
        iota_target=float(config.iota_target),
        major_radius_target=float(config.major_radius_target),
        length_target=float(config.length_target),
        volume_target=float(config.volume_target),
        non_qs_weight=float(config.non_qs_weight),
        residual_weight=float(config.residual_weight),
        iota_weight=float(config.iota_weight),
        major_radius_weight=float(config.major_radius_weight),
        length_weight=float(config.length_weight),
        non_qs_axis=config.non_qs_axis,
        weight_inv_modB=config.weight_inv_modB,
    )


def _native_objective_contract(
    contract: NativeFullSpaceObjectiveContract,
) -> ObjectiveContractEvidence:
    return ObjectiveContractEvidence(**asdict(contract))


def _maximum_tolerance_ratio(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> float:
    if reference.shape != candidate.shape:
        return float("inf")
    allowed = atol + rtol * np.abs(reference)
    difference = np.abs(candidate - reference)
    ratio = np.divide(
        difference,
        allowed,
        out=np.zeros_like(difference),
        where=allowed != 0.0,
    )
    if np.any((allowed == 0.0) & (difference != 0.0)):
        return float("inf")
    return float(np.max(ratio, initial=0.0))


def _jax_raw_terms(evaluation: FullSpaceEvaluation) -> np.ndarray:
    return np.asarray(
        [float(getattr(evaluation.raw_terms, name)) for name in _RAW_TERM_NAMES],
        dtype=np.float64,
    )


def _native_raw_terms(evaluation: NativeExplicitStateEvaluation) -> np.ndarray:
    return np.asarray(
        [float(getattr(evaluation.objective_terms, name)) for name in _RAW_TERM_NAMES],
        dtype=np.float64,
    )


def _jax_observables(evaluation: FullSpaceEvaluation) -> np.ndarray:
    names = (
        "iota",
        "G",
        "volume",
        "major_radius",
        "total_length",
        "non_qs_ratio",
        "boozer_residual_scalar",
        "boozer_residual_rms",
    )
    return np.asarray(
        [float(getattr(evaluation.observables, name)) for name in names],
        dtype=np.float64,
    )


def _native_observables(evaluation: NativeExplicitStateEvaluation) -> np.ndarray:
    return np.asarray(
        [float(getattr(evaluation.observables, name)) for name in _OBSERVABLE_NAMES],
        dtype=np.float64,
    )


def _cross_evaluate(
    state: np.ndarray,
    native: NativeExplicitStateEvaluation,
    jax_evaluation: FullSpaceEvaluation,
) -> SameStateCrossEvaluationEvidence:
    native_terms = _native_raw_terms(native)
    jax_terms = _jax_raw_terms(jax_evaluation)
    native_equalities = np.asarray(native.raw_equalities, dtype=np.float64)
    jax_equalities = _fp64(flatten_fullspace_constraints(jax_evaluation.constraints))
    native_observables = _native_observables(native)
    jax_observables = _jax_observables(jax_evaluation)
    jax_raw_leaves = tuple(
        _fp64(getattr(jax_evaluation.raw_terms, name)) for name in _RAW_TERM_NAMES
    )
    jax_observable_names = (
        "iota",
        "G",
        "volume",
        "major_radius",
        "total_length",
        "non_qs_ratio",
        "boozer_residual_scalar",
        "boozer_residual_rms",
    )
    jax_observable_leaves = tuple(
        _fp64(getattr(jax_evaluation.observables, name))
        for name in jax_observable_names
    )
    state_array = np.asarray(state)
    native_state = np.asarray(native.state)
    return SameStateCrossEvaluationEvidence(
        requested_state=tuple(float(value) for value in state_array),
        native_returned_state=tuple(float(value) for value in native_state),
        requested_state_sha256=_array_sha256(state),
        native_returned_state_sha256=_array_sha256(native.state),
        native_objective=float(native.objective),
        jax_objective=float(jax_evaluation.weighted_total),
        native_raw_terms=tuple(float(value) for value in native_terms),
        jax_raw_terms=tuple(float(value) for value in jax_terms),
        native_raw_equalities=tuple(float(value) for value in native_equalities),
        jax_raw_equalities=tuple(float(value) for value in jax_equalities),
        native_observables=tuple(float(value) for value in native_observables),
        jax_observables=tuple(float(value) for value in jax_observables),
        state_dtype=str(state_array.dtype),
        state_size=state_array.size,
        state_nonfinite_count=int(
            state_array.size - np.count_nonzero(np.isfinite(state_array))
        ),
        native_state_dtype=str(native_state.dtype),
        native_state_size=native_state.size,
        native_state_nonfinite_count=int(
            native_state.size - np.count_nonzero(np.isfinite(native_state))
        ),
        native_equality_dtype=str(np.asarray(native.raw_equalities).dtype),
        jax_equality_dtype=str(jax_equalities.dtype),
        jax_scalar_dtypes=tuple(
            str(leaf.dtype)
            for leaf in (
                _fp64(jax_evaluation.weighted_total),
                *jax_raw_leaves,
                *jax_observable_leaves,
            )
        ),
    )


def _branch_payload(evidence: NativeAcceptedPathEvidence) -> dict[str, object]:
    def step_payload(step: NativeContinuationStep) -> dict[str, JsonValue]:
        payload = _json_value(step)
        if not isinstance(payload, dict):
            raise TypeError("native continuation step must serialize as an object")
        return payload

    return {
        "bootstrap": step_payload(evidence.bootstrap_step),
        "intervals": [
            {
                "index": interval.index,
                "supplied_state_little_endian_sha256": (
                    interval.supplied_state_little_endian_sha256
                ),
                "direct_step": step_payload(interval.direct_step),
                "midpoint_step": step_payload(interval.midpoint_step),
                "refined_step": step_payload(interval.refined_step),
                "direct_root_sha256": _array_sha256(interval.direct_root),
                "midpoint_root_sha256": _array_sha256(interval.midpoint_root),
                "refined_root_sha256": _array_sha256(interval.refined_root),
                "direct_refined_infinity_difference": (
                    interval.direct_refined_infinity_difference
                ),
                "supplied_refined_infinity_difference": (
                    interval.supplied_refined_infinity_difference
                ),
            }
            for interval in evidence.intervals
        ],
        "first_failing_index": evidence.first_failing_index,
        "failure_reason": evidence.failure_reason,
        "usable": evidence.usable,
    }


def _branch_audit(
    evidence: NativeAcceptedPathEvidence,
    accepted_states: np.ndarray,
) -> BranchReplayAuditEvidence:
    steps = (evidence.bootstrap_step,) + tuple(
        step
        for interval in evidence.intervals
        for step in (
            interval.direct_step,
            interval.midpoint_step,
            interval.refined_step,
        )
    )
    maximum_scaled = max(
        (step.scaled_boozer_infinity_norm for step in steps),
        default=float("inf"),
    )
    coil_blocks = accepted_states[:, :461]
    root_blocks = accepted_states[:, 461:]
    midpoint_coils = tuple(
        coil_blocks[index - 1]
        + np.float64(0.5) * (coil_blocks[index] - coil_blocks[index - 1])
        for index in range(1, accepted_states.shape[0])
    )
    return BranchReplayAuditEvidence(
        raw=evidence,
        evidence_sha256=_canonical_sha256(_branch_payload(evidence)),
        replayed_row_count=1 + len(evidence.intervals),
        successful_direct_rows=1 + len(evidence.intervals),
        successful_midpoint_refined_rows=len(evidence.intervals),
        maximum_direct_refined_difference=max(
            (
                interval.direct_refined_infinity_difference
                for interval in evidence.intervals
            ),
            default=0.0,
        ),
        maximum_gpu_native_difference=max(
            (
                interval.supplied_refined_infinity_difference
                for interval in evidence.intervals
            ),
            default=0.0,
        ),
        maximum_scaled_boozer_feasibility=float(maximum_scaled),
        accepted_state_sha256s=tuple(_array_sha256(state) for state in accepted_states),
        accepted_coil_sha256s=tuple(_array_sha256(coil) for coil in coil_blocks),
        accepted_root_sha256s=tuple(_array_sha256(root) for root in root_blocks),
        accepted_roots=tuple(
            tuple(float(value) for value in root) for root in root_blocks
        ),
        midpoint_coil_sha256s=tuple(_array_sha256(coil) for coil in midpoint_coils),
    )


def produce_native_equivalent_endpoint_audit(
    result: NativeEquivalentQualityResult,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    policy: NativeEquivalentQualityPolicy,
    native_runtime: NativeSingleStageEndpointRuntime,
    native_reference_state: np.ndarray,
) -> NativeEquivalentEndpointAudit:
    """Recompute the complete post-timing quality audit from raw endpoint state."""

    loop = result.loop_result
    accepted_steps = int(jax.device_get(loop.accepted_steps))
    optimizer_ledger = _fp64(loop.accepted_optimizer_coordinates)
    physical_ledger = _fp64(result.accepted_physical_coordinates)
    mask = np.asarray(jax.device_get(result.accepted_state_mask), dtype=np.bool_)
    valid_row_count = accepted_steps + 1
    valid_physical = np.array(physical_ledger[:valid_row_count], copy=True)
    recomputed_physical = _fp64(scaling.bootstrap_anchor)[None, :] + (
        optimizer_ledger * _fp64(scaling.variable_scale)[None, :]
    )
    expected_mask = np.arange(mask.size) < valid_row_count
    endpoint_state = _fp64(result.endpoint.physical_state)
    latched_state = valid_physical[-1]
    binding = LatchedStateBindingEvidence(
        accepted_step_count=accepted_steps,
        valid_row_count=valid_row_count,
        row_capacity=optimizer_ledger.shape[0],
        mask_true_count=int(np.count_nonzero(mask)),
        observed_mask_sha256=_array_sha256(mask),
        expected_mask_sha256=_array_sha256(expected_mask),
        optimizer_ledger_sha256=_array_sha256(optimizer_ledger),
        physical_ledger_sha256=_array_sha256(physical_ledger),
        recomputed_physical_ledger_sha256=_array_sha256(recomputed_physical),
        valid_physical_row_sha256s=tuple(_array_sha256(row) for row in valid_physical),
        bootstrap_state_sha256=_array_sha256(valid_physical[0]),
        expected_bootstrap_state_sha256=_array_sha256(native_runtime.bootstrap_state),
        latched_state_sha256=_array_sha256(latched_state),
        latched_optimizer_row_sha256=_array_sha256(optimizer_ledger[accepted_steps]),
        loop_latched_optimizer_state_sha256=_array_sha256(loop.optimizer_coordinates),
        finalized_optimizer_state_sha256=_array_sha256(
            result.optimizer_result.optimizer_coordinates
        ),
        endpoint_physical_state_sha256=_array_sha256(endpoint_state),
        observed_mask=np.array(mask, copy=True),
        expected_mask=np.array(expected_mask, copy=True),
        optimizer_ledger=np.array(optimizer_ledger, copy=True),
        physical_ledger=np.array(physical_ledger, copy=True),
        recomputed_physical_ledger=np.array(recomputed_physical, copy=True),
        bootstrap_anchor=np.array(_fp64(scaling.bootstrap_anchor), copy=True),
        variable_scale=np.array(_fp64(scaling.variable_scale), copy=True),
    )

    optimizer_endpoint = _fp64(result.optimizer_result.optimizer_coordinates)
    residual = certify_fullspace_objective_residuals(endpoint_state, problem)
    endpoint_jax_state = jnp.asarray(endpoint_state)
    objective_residuals = fullspace_objective_residual_vector(
        endpoint_jax_state,
        problem,
    )
    reconstructed_value = 0.5 * jnp.vdot(objective_residuals, objective_residuals)
    authoritative_value, authoritative_gradient = fullspace_value_and_grad(
        endpoint_jax_state,
        problem,
    )
    _, residual_pullback = jax.vjp(
        lambda candidate: fullspace_objective_residual_vector(candidate, problem),
        endpoint_jax_state,
    )
    reconstructed_gradient = residual_pullback(objective_residuals)[0]
    transpose = deterministic_constraint_transpose_certificate(
        jnp.asarray(optimizer_endpoint),
        problem,
        scaling,
    )
    gradient = _fp64(authoritative_gradient)
    jvp = _fp64(transpose.jvp_action)
    vjp = _fp64(transpose.vjp_action)
    derivative_residual = DerivativeResidualEvidence(
        gradient_dtype=str(gradient.dtype),
        gradient_size=gradient.size,
        gradient_nonfinite_count=int(
            gradient.size - np.count_nonzero(np.isfinite(gradient))
        ),
        jvp_dtype=str(jvp.dtype),
        jvp_size=jvp.size,
        jvp_nonfinite_count=int(jvp.size - np.count_nonzero(np.isfinite(jvp))),
        vjp_dtype=str(vjp.dtype),
        vjp_size=vjp.size,
        vjp_nonfinite_count=int(vjp.size - np.count_nonzero(np.isfinite(vjp))),
        residual_value_defect=float(residual.value_scaled_defect),
        residual_gradient_defect=float(residual.gradient_scaled_defect),
        transpose_primal_dot=float(transpose.primal_dot),
        transpose_adjoint_dot=float(transpose.transpose_dot),
        transpose_denominator=float(transpose.denominator),
        transpose_defect=float(transpose.defect),
        gradient=tuple(float(value) for value in gradient),
        state_probe=tuple(float(value) for value in _fp64(transpose.state_probe)),
        equality_probe=tuple(float(value) for value in _fp64(transpose.equality_probe)),
        jvp_action=tuple(float(value) for value in jvp),
        vjp_action=tuple(float(value) for value in vjp),
        reconstructed_objective_value=float(reconstructed_value),
        authoritative_objective_value=float(authoritative_value),
        reconstructed_objective_gradient=tuple(
            float(value) for value in _fp64(reconstructed_gradient)
        ),
        authoritative_objective_gradient=tuple(
            float(value) for value in _fp64(authoritative_gradient)
        ),
    )

    native_gpu = native_runtime.evaluate_state(latched_state)
    checked_native_reference_state = np.asarray(native_reference_state)
    native_native = native_runtime.evaluate_state(checked_native_reference_state)
    native_reduced = native_runtime.evaluate_reduced(latched_state[:461])
    jax_gpu = evaluate_fullspace(jnp.asarray(latched_state), problem)
    jax_native = evaluate_fullspace(
        jnp.asarray(checked_native_reference_state), problem
    )
    gpu_cross = _cross_evaluate(latched_state, native_gpu, jax_gpu)
    native_cross = _cross_evaluate(
        checked_native_reference_state,
        native_native,
        jax_native,
    )
    reduced_gradient = np.asarray(native_reduced.gradient)
    reduced_telemetry = NativeReducedEndpointTelemetry(
        parameter_sha256=_array_sha256(native_reduced.parameters),
        objective=native_reduced.objective,
        gradient=tuple(float(value) for value in reduced_gradient),
        gradient_dtype=str(reduced_gradient.dtype),
        gradient_infinity_norm=native_reduced.gradient_infinity_norm,
        gradient_l2_norm=native_reduced.gradient_l2_norm,
        solver_residual_l2=native_reduced.solver_residual_l2,
        solver_residual_infinity_norm=native_reduced.solver_residual_infinity_norm,
        inner_solver_success=native_reduced.inner_solver_success,
    )

    layout = FROZEN_PROBLEM_CONTRACT.layout
    unpacked_endpoint = problem.layout.unpack(jnp.asarray(endpoint_state))
    repacked_endpoint = _fp64(problem.layout.pack(unpacked_endpoint))
    contract = PhysicsContractEvidence(
        state_ordering=layout.ordering,
        endpoint_state_sha256=_array_sha256(endpoint_state),
        repacked_endpoint_state_sha256=_array_sha256(repacked_endpoint),
        coil_dof_count=problem.layout.coil_dof_count,
        surface_dof_count=problem.layout.surface_dof_count,
        scalar_dof_count=problem.layout.total_dof_count
        - problem.layout.coil_dof_count
        - problem.layout.surface_dof_count,
        total_dof_count=problem.layout.total_dof_count,
        equality_count=native_gpu.raw_equalities.size,
        length_coil_indices=problem.config.length_coil_indices,
        objective_term_ids=tuple(
            row.term_id
            for row in TERM_LEDGER
            if row.classification
            in {TermClassification.OBJECTIVE, TermClassification.OBJECTIVE_PENALTY}
        ),
        equality_term_ids=tuple(
            row.term_id
            for row in TERM_LEDGER
            if row.classification is TermClassification.EQUALITY
        ),
        fixed_term_ids=tuple(
            row.term_id
            for row in TERM_LEDGER
            if row.classification is TermClassification.FIXED_STATE
        ),
        term_ledger_sha256=_canonical_sha256([asdict(row) for row in TERM_LEDGER]),
        objective_contract_sha256=_canonical_sha256(
            _objective_contract_payload(problem)
        ),
        native_objective_contract_sha256=_canonical_sha256(
            _native_objective_contract_payload(native_runtime.objective_contract)
        ),
        exact_mask_indices_sha256=_array_sha256(problem.exact_mask_indices),
        native_exact_mask_indices_sha256=_array_sha256(
            native_runtime.exact_mask_indices
        ),
        fixed_first_base_current=native_runtime.fixed_first_base_current,
        gpu_native_fixed_first_base_current=(
            native_gpu.observables.fixed_first_base_current
        ),
        reference_native_fixed_first_base_current=(
            native_native.observables.fixed_first_base_current
        ),
        jax_objective_contract=_jax_objective_contract(problem),
        native_objective_contract=_native_objective_contract(
            native_runtime.objective_contract
        ),
        exact_mask_indices=np.array(
            _fp64(problem.exact_mask_indices),
            dtype=np.int64,
            copy=True,
        ),
        native_exact_mask_indices=np.array(
            native_runtime.exact_mask_indices,
            dtype=np.int64,
            copy=True,
        ),
    )
    branch = _branch_audit(
        native_runtime.audit_accepted_states(valid_physical),
        valid_physical,
    )
    gpu_quality = GpuQualityEvidence(
        physical_objective=float(jax_gpu.weighted_total),
        gpu_raw_objective_terms=tuple(
            float(value) for value in _jax_raw_terms(jax_gpu)
        ),
        objective_weights=tuple(
            float(value)
            for value in (
                problem.config.non_qs_weight,
                problem.config.residual_weight,
                problem.config.iota_weight,
                problem.config.major_radius_weight,
                problem.config.length_weight,
            )
        ),
        gpu_raw_equalities=tuple(
            float(value)
            for value in _fp64(flatten_fullspace_constraints(jax_gpu.constraints))
        ),
        native_raw_equalities=tuple(
            float(value) for value in np.asarray(native_native.raw_equalities)
        ),
        policy_native_raw_equalities=tuple(
            float(value) for value in _fp64(policy.native_raw_equalities)
        ),
        constraint_inverse_scale=tuple(
            float(value) for value in _fp64(policy.constraint_inverse_scale)
        ),
        objective_target=policy.objective_target,
        component_absolute_tolerance=policy.component_absolute_tolerance,
        component_relative_tolerance=policy.component_relative_tolerance,
        scaled_feasibility_tolerance=policy.scaled_feasibility_tolerance,
        reported_scaled_feasibility_infinity_norm=float(
            np.linalg.norm(
                _fp64(flatten_fullspace_constraints(jax_gpu.constraints))
                * _fp64(policy.constraint_inverse_scale),
                ord=np.inf,
            )
        ),
    )
    return NativeEquivalentEndpointAudit(
        schema_version=SCHEMA_VERSION,
        plan_sha256=PLAN_SHA256,
        audited_state_sha256=_array_sha256(endpoint_state),
        audited_coil_parameters_sha256=_array_sha256(endpoint_state[:461]),
        binding=binding,
        gpu_quality=gpu_quality,
        derivative_residual=derivative_residual,
        physics_contract=contract,
        branch_replay=branch,
        gpu_endpoint_cross_evaluation=gpu_cross,
        native_endpoint_cross_evaluation=native_cross,
        native_reference_state_sha256=_array_sha256(checked_native_reference_state),
        native_reduced_endpoint=reduced_telemetry,
    )


def endpoint_audit_payload(
    audit: NativeEquivalentEndpointAudit,
) -> dict[str, JsonValue]:
    """Return the complete adapter-independent raw audit as canonical JSON data."""

    payload = _json_value(audit)
    if not isinstance(payload, dict):
        raise TypeError("endpoint audit did not serialize as an object")
    return payload


def endpoint_audit_bytes(audit: NativeEquivalentEndpointAudit) -> bytes:
    """Serialize the audit using the protocol's sole canonical representation."""

    return canonical_json_bytes(endpoint_audit_payload(audit))


def _object(
    value: JsonValue,
    expected_type: type[object],
    context: str,
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    expected_keys = frozenset(field.name for field in fields(expected_type))
    if frozenset(value) != expected_keys:
        raise ValueError(f"{context} keys differ from the frozen schema")
    return value


def _array(
    value: JsonValue,
    shape: tuple[int, ...],
    dtype: np.dtype[np.float64] | np.dtype[np.bool_] | np.dtype[np.int64],
    context: str,
) -> np.ndarray:
    def contains_boolean(item: JsonValue) -> bool:
        return isinstance(item, bool) or (
            isinstance(item, list) and any(contains_boolean(child) for child in item)
        )

    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{context} shape differs from the frozen schema")
    if dtype == np.dtype(np.bool_):
        if array.dtype != np.dtype(np.bool_):
            raise TypeError(f"{context} must contain booleans")
        return np.array(array, dtype=np.bool_, copy=True)
    if dtype == np.dtype(np.int64):
        if array.dtype.kind not in {"i", "u"} or array.dtype == np.dtype(np.bool_):
            raise TypeError(f"{context} must contain JSON integers")
        return np.array(array, dtype=np.int64, copy=True)
    if contains_boolean(value):
        raise TypeError(f"{context} must not contain booleans")
    if array.dtype.kind not in {"f", "i"}:
        raise TypeError(f"{context} must contain JSON numbers")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context} must be finite")
    return result


def _tuple_numbers(value: JsonValue, size: int, context: str) -> tuple[float, ...]:
    return tuple(
        float(item) for item in _array(value, (size,), np.dtype(np.float64), context)
    )


def _tuple_strings(value: JsonValue, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{context} must be an array of strings")
    return tuple(value)


def _tuple_sha256(value: JsonValue, context: str) -> tuple[str, ...]:
    return tuple(
        _sha256(item, f"{context}[{index}]")
        for index, item in enumerate(_tuple_strings(value, context))
    )


def _number(value: JsonValue, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a JSON number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer(value: JsonValue, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be a JSON integer")
    if value < 0:
        raise ValueError(f"{context} must be nonnegative")
    return value


def _boolean(value: JsonValue, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a JSON boolean")
    return value


def _string(value: JsonValue, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _sha256(value: JsonValue, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return result


def _strict_scalar_fields(
    payload: dict[str, JsonValue],
    context: str,
    *,
    numbers: tuple[str, ...] = (),
    integers: tuple[str, ...] = (),
    strings: tuple[str, ...] = (),
    hashes: tuple[str, ...] = (),
    booleans: tuple[str, ...] = (),
) -> dict[str, JsonValue]:
    result = dict(payload)
    for name in numbers:
        result[name] = _number(payload[name], f"{context}.{name}")
    for name in integers:
        result[name] = _integer(payload[name], f"{context}.{name}")
    for name in strings:
        result[name] = _string(payload[name], f"{context}.{name}")
    for name in hashes:
        result[name] = _sha256(payload[name], f"{context}.{name}")
    for name in booleans:
        result[name] = _boolean(payload[name], f"{context}.{name}")
    return result


def _validate_json_tree(value: JsonValue, context: str = "endpoint audit") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} keys must be strings")
            if key.endswith("sha256") and (
                not isinstance(item, str)
                or len(item) != 64
                or any(character not in "0123456789abcdef" for character in item)
            ):
                raise ValueError(f"{context}.{key} must be a lowercase SHA-256")
            if isinstance(item, str) and not (
                key.endswith("sha256") or key in _JSON_STRING_FIELDS
            ):
                raise TypeError(f"{context}.{key} has an invalid string value")
            if isinstance(item, bool) and key not in _JSON_BOOLEAN_FIELDS:
                raise TypeError(f"{context}.{key} has an invalid boolean value")
            _validate_json_tree(item, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{context}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{context} must be finite")
        return
    raise TypeError(f"{context} is not a JSON value")


def _step_from_payload(value: JsonValue, context: str) -> NativeContinuationStep:
    payload = _object(value, NativeContinuationStep, context)
    return NativeContinuationStep(
        segment_count=_integer(payload["segment_count"], f"{context}.segment_count"),
        index=_integer(payload["index"], f"{context}.index"),
        predecessor_index=(
            None
            if payload["predecessor_index"] is None
            else _integer(payload["predecessor_index"], f"{context}.predecessor_index")
        ),
        coil_little_endian_sha256=_sha256(
            payload["coil_little_endian_sha256"], f"{context}.coil hash"
        ),
        seed_root_little_endian_sha256=_sha256(
            payload["seed_root_little_endian_sha256"], f"{context}.seed hash"
        ),
        root_little_endian_sha256=_sha256(
            payload["root_little_endian_sha256"], f"{context}.root hash"
        ),
        newton_iterations=_integer(
            payload["newton_iterations"], f"{context}.newton_iterations"
        ),
        residual_l2=_number(payload["residual_l2"], f"{context}.residual_l2"),
        residual_infinity_norm=_number(
            payload["residual_infinity_norm"], f"{context}.residual_infinity_norm"
        ),
        scaled_boozer_infinity_norm=_number(
            payload["scaled_boozer_infinity_norm"],
            f"{context}.scaled_boozer_infinity_norm",
        ),
        raw_equalities=_array(
            payload["raw_equalities"],
            (_EQUALITY_SIZE,),
            np.dtype(np.float64),
            f"{context}.raw_equalities",
        ),
        raw_equalities_little_endian_sha256=_sha256(
            payload["raw_equalities_little_endian_sha256"],
            f"{context}.raw_equalities hash",
        ),
    )


def _accepted_path_from_payload(value: JsonValue) -> NativeAcceptedPathEvidence:
    payload = _object(value, NativeAcceptedPathEvidence, "branch.raw")
    raw_intervals = payload["intervals"]
    if not isinstance(raw_intervals, list):
        raise TypeError("branch.raw.intervals must be an array")
    intervals = []
    for index, raw_interval in enumerate(raw_intervals):
        interval = _object(
            raw_interval,
            NativeAcceptedIntervalEvidence,
            f"branch.raw.intervals[{index}]",
        )
        intervals.append(
            NativeAcceptedIntervalEvidence(
                index=_integer(interval["index"], f"branch.interval[{index}].index"),
                supplied_state_little_endian_sha256=_sha256(
                    interval["supplied_state_little_endian_sha256"],
                    f"branch.interval[{index}].state hash",
                ),
                direct_step=_step_from_payload(
                    interval["direct_step"], f"branch.direct[{index}]"
                ),
                midpoint_step=_step_from_payload(
                    interval["midpoint_step"], f"branch.midpoint[{index}]"
                ),
                refined_step=_step_from_payload(
                    interval["refined_step"], f"branch.refined[{index}]"
                ),
                direct_root=_array(
                    interval["direct_root"],
                    (255,),
                    np.dtype(np.float64),
                    f"branch.direct_root[{index}]",
                ),
                midpoint_root=_array(
                    interval["midpoint_root"],
                    (255,),
                    np.dtype(np.float64),
                    f"branch.midpoint_root[{index}]",
                ),
                refined_root=_array(
                    interval["refined_root"],
                    (255,),
                    np.dtype(np.float64),
                    f"branch.refined_root[{index}]",
                ),
                direct_refined_infinity_difference=_number(
                    interval["direct_refined_infinity_difference"],
                    f"branch.interval[{index}].direct_refined_difference",
                ),
                supplied_refined_infinity_difference=_number(
                    interval["supplied_refined_infinity_difference"],
                    f"branch.interval[{index}].supplied_refined_difference",
                ),
            )
        )
    return NativeAcceptedPathEvidence(
        bootstrap_step=_step_from_payload(
            payload["bootstrap_step"], "branch.bootstrap"
        ),
        intervals=tuple(intervals),
        first_failing_index=(
            None
            if payload["first_failing_index"] is None
            else _integer(payload["first_failing_index"], "branch.first_failing_index")
        ),
        failure_reason=(
            None
            if payload["failure_reason"] is None
            else _string(payload["failure_reason"], "branch.failure_reason")
        ),
        usable=_boolean(payload["usable"], "branch.usable"),
    )


def endpoint_audit_from_payload(value: JsonValue) -> NativeEquivalentEndpointAudit:
    """Strictly parse raw JSON evidence and recompute its semantic verdict."""

    _validate_json_tree(value)
    payload = _object(value, NativeEquivalentEndpointAudit, "endpoint audit")
    binding_payload = _object(
        payload["binding"], LatchedStateBindingEvidence, "endpoint audit.binding"
    )
    binding = LatchedStateBindingEvidence(
        **{
            **_strict_scalar_fields(
                binding_payload,
                "binding",
                integers=(
                    "accepted_step_count",
                    "valid_row_count",
                    "row_capacity",
                    "mask_true_count",
                ),
                hashes=tuple(
                    field.name
                    for field in fields(LatchedStateBindingEvidence)
                    if field.name.endswith("sha256")
                ),
            ),
            "valid_physical_row_sha256s": _tuple_sha256(
                binding_payload["valid_physical_row_sha256s"], "binding row hashes"
            ),
            "observed_mask": _array(
                binding_payload["observed_mask"],
                (257,),
                np.dtype(np.bool_),
                "observed mask",
            ),
            "expected_mask": _array(
                binding_payload["expected_mask"],
                (257,),
                np.dtype(np.bool_),
                "expected mask",
            ),
            "optimizer_ledger": _array(
                binding_payload["optimizer_ledger"],
                (257, _STATE_SIZE),
                np.dtype(np.float64),
                "optimizer ledger",
            ),
            "physical_ledger": _array(
                binding_payload["physical_ledger"],
                (257, _STATE_SIZE),
                np.dtype(np.float64),
                "physical ledger",
            ),
            "recomputed_physical_ledger": _array(
                binding_payload["recomputed_physical_ledger"],
                (257, _STATE_SIZE),
                np.dtype(np.float64),
                "recomputed physical ledger",
            ),
            "bootstrap_anchor": _array(
                binding_payload["bootstrap_anchor"],
                (_STATE_SIZE,),
                np.dtype(np.float64),
                "bootstrap anchor",
            ),
            "variable_scale": _array(
                binding_payload["variable_scale"],
                (_STATE_SIZE,),
                np.dtype(np.float64),
                "variable scale",
            ),
        }
    )
    quality_payload = _object(
        payload["gpu_quality"], GpuQualityEvidence, "GPU quality evidence"
    )
    gpu_quality = GpuQualityEvidence(
        **{
            **_strict_scalar_fields(
                quality_payload,
                "GPU quality",
                numbers=(
                    "physical_objective",
                    "objective_target",
                    "component_absolute_tolerance",
                    "component_relative_tolerance",
                    "scaled_feasibility_tolerance",
                    "reported_scaled_feasibility_infinity_norm",
                ),
            ),
            "gpu_raw_equalities": _tuple_numbers(
                quality_payload["gpu_raw_equalities"], 255, "GPU raw equalities"
            ),
            "gpu_raw_objective_terms": _tuple_numbers(
                quality_payload["gpu_raw_objective_terms"],
                5,
                "GPU raw objective terms",
            ),
            "objective_weights": _tuple_numbers(
                quality_payload["objective_weights"], 5, "objective weights"
            ),
            "native_raw_equalities": _tuple_numbers(
                quality_payload["native_raw_equalities"],
                255,
                "native raw equalities",
            ),
            "policy_native_raw_equalities": _tuple_numbers(
                quality_payload["policy_native_raw_equalities"],
                255,
                "policy native raw equalities",
            ),
            "constraint_inverse_scale": _tuple_numbers(
                quality_payload["constraint_inverse_scale"],
                255,
                "constraint inverse scale",
            ),
        }
    )
    derivative_payload = _object(
        payload["derivative_residual"],
        DerivativeResidualEvidence,
        "derivative evidence",
    )
    derivative = DerivativeResidualEvidence(
        **{
            **_strict_scalar_fields(
                derivative_payload,
                "derivative",
                integers=(
                    "gradient_size",
                    "gradient_nonfinite_count",
                    "jvp_size",
                    "jvp_nonfinite_count",
                    "vjp_size",
                    "vjp_nonfinite_count",
                ),
                strings=("gradient_dtype", "jvp_dtype", "vjp_dtype"),
                numbers=(
                    "residual_value_defect",
                    "residual_gradient_defect",
                    "transpose_primal_dot",
                    "transpose_adjoint_dot",
                    "transpose_denominator",
                    "transpose_defect",
                    "reconstructed_objective_value",
                    "authoritative_objective_value",
                ),
            ),
            "gradient": _tuple_numbers(derivative_payload["gradient"], 716, "gradient"),
            "state_probe": _tuple_numbers(
                derivative_payload["state_probe"], 716, "state probe"
            ),
            "equality_probe": _tuple_numbers(
                derivative_payload["equality_probe"], 255, "equality probe"
            ),
            "jvp_action": _tuple_numbers(derivative_payload["jvp_action"], 255, "JVP"),
            "vjp_action": _tuple_numbers(derivative_payload["vjp_action"], 716, "VJP"),
            "reconstructed_objective_gradient": _tuple_numbers(
                derivative_payload["reconstructed_objective_gradient"],
                716,
                "reconstructed objective gradient",
            ),
            "authoritative_objective_gradient": _tuple_numbers(
                derivative_payload["authoritative_objective_gradient"],
                716,
                "authoritative objective gradient",
            ),
        }
    )

    def cross(raw_value: JsonValue, context: str) -> SameStateCrossEvaluationEvidence:
        raw = _object(raw_value, SameStateCrossEvaluationEvidence, context)
        return SameStateCrossEvaluationEvidence(
            **{
                **_strict_scalar_fields(
                    raw,
                    context,
                    integers=(
                        "state_size",
                        "state_nonfinite_count",
                        "native_state_size",
                        "native_state_nonfinite_count",
                    ),
                    strings=(
                        "state_dtype",
                        "native_state_dtype",
                        "native_equality_dtype",
                        "jax_equality_dtype",
                    ),
                    hashes=("requested_state_sha256", "native_returned_state_sha256"),
                    numbers=("native_objective", "jax_objective"),
                ),
                "requested_state": _tuple_numbers(
                    raw["requested_state"], 716, f"{context} requested state"
                ),
                "native_returned_state": _tuple_numbers(
                    raw["native_returned_state"],
                    716,
                    f"{context} returned state",
                ),
                "native_raw_terms": _tuple_numbers(raw["native_raw_terms"], 5, context),
                "jax_raw_terms": _tuple_numbers(raw["jax_raw_terms"], 5, context),
                "native_raw_equalities": _tuple_numbers(
                    raw["native_raw_equalities"], 255, context
                ),
                "jax_raw_equalities": _tuple_numbers(
                    raw["jax_raw_equalities"], 255, context
                ),
                "native_observables": _tuple_numbers(
                    raw["native_observables"], 8, context
                ),
                "jax_observables": _tuple_numbers(raw["jax_observables"], 8, context),
                "jax_scalar_dtypes": _tuple_strings(raw["jax_scalar_dtypes"], context),
            }
        )

    physics_payload = _object(
        payload["physics_contract"], PhysicsContractEvidence, "physics contract"
    )

    def objective_contract(
        raw_value: JsonValue, context: str
    ) -> ObjectiveContractEvidence:
        raw = _object(raw_value, ObjectiveContractEvidence, context)
        return ObjectiveContractEvidence(
            iota_target=_number(raw["iota_target"], f"{context}.iota_target"),
            major_radius_target=_number(
                raw["major_radius_target"], f"{context}.major_radius_target"
            ),
            length_target=_number(raw["length_target"], f"{context}.length_target"),
            volume_target=_number(raw["volume_target"], f"{context}.volume_target"),
            non_qs_weight=_number(raw["non_qs_weight"], f"{context}.non_qs_weight"),
            residual_weight=_number(
                raw["residual_weight"], f"{context}.residual_weight"
            ),
            iota_weight=_number(raw["iota_weight"], f"{context}.iota_weight"),
            major_radius_weight=_number(
                raw["major_radius_weight"], f"{context}.major_radius_weight"
            ),
            length_weight=_number(raw["length_weight"], f"{context}.length_weight"),
            non_qs_axis=_integer(raw["non_qs_axis"], f"{context}.non_qs_axis"),
            weight_inv_modB=_boolean(
                raw["weight_inv_modB"], f"{context}.weight_inv_modB"
            ),
        )

    physics = PhysicsContractEvidence(
        **{
            **_strict_scalar_fields(
                physics_payload,
                "physics",
                integers=(
                    "coil_dof_count",
                    "surface_dof_count",
                    "scalar_dof_count",
                    "total_dof_count",
                    "equality_count",
                ),
                numbers=(
                    "fixed_first_base_current",
                    "gpu_native_fixed_first_base_current",
                    "reference_native_fixed_first_base_current",
                ),
                hashes=tuple(
                    field.name
                    for field in fields(PhysicsContractEvidence)
                    if field.name.endswith("sha256")
                ),
            ),
            "state_ordering": _tuple_strings(
                physics_payload["state_ordering"], "state ordering"
            ),
            "length_coil_indices": tuple(
                _integer(item, "physics.length_coil_indices")
                for item in physics_payload["length_coil_indices"]
            ),
            "objective_term_ids": _tuple_strings(
                physics_payload["objective_term_ids"], "objective terms"
            ),
            "equality_term_ids": _tuple_strings(
                physics_payload["equality_term_ids"], "equality terms"
            ),
            "fixed_term_ids": _tuple_strings(
                physics_payload["fixed_term_ids"], "fixed terms"
            ),
            "jax_objective_contract": objective_contract(
                physics_payload["jax_objective_contract"], "JAX objective contract"
            ),
            "native_objective_contract": objective_contract(
                physics_payload["native_objective_contract"],
                "native objective contract",
            ),
            "exact_mask_indices": _array(
                physics_payload["exact_mask_indices"],
                (254,),
                np.dtype(np.int64),
                "exact mask indices",
            ),
            "native_exact_mask_indices": _array(
                physics_payload["native_exact_mask_indices"],
                (254,),
                np.dtype(np.int64),
                "native exact mask indices",
            ),
        }
    )
    branch_payload = _object(
        payload["branch_replay"], BranchReplayAuditEvidence, "branch replay"
    )
    accepted_roots_value = branch_payload["accepted_roots"]
    if not isinstance(accepted_roots_value, list):
        raise TypeError("accepted roots must be an array")
    branch = BranchReplayAuditEvidence(
        **{
            **_strict_scalar_fields(
                branch_payload,
                "branch",
                integers=(
                    "replayed_row_count",
                    "successful_direct_rows",
                    "successful_midpoint_refined_rows",
                ),
                numbers=(
                    "maximum_direct_refined_difference",
                    "maximum_gpu_native_difference",
                    "maximum_scaled_boozer_feasibility",
                ),
                hashes=("evidence_sha256",),
            ),
            "raw": _accepted_path_from_payload(branch_payload["raw"]),
            "accepted_state_sha256s": _tuple_sha256(
                branch_payload["accepted_state_sha256s"], "accepted state hashes"
            ),
            "accepted_coil_sha256s": _tuple_sha256(
                branch_payload["accepted_coil_sha256s"], "accepted coil hashes"
            ),
            "accepted_root_sha256s": _tuple_sha256(
                branch_payload["accepted_root_sha256s"], "accepted root hashes"
            ),
            "midpoint_coil_sha256s": _tuple_sha256(
                branch_payload["midpoint_coil_sha256s"], "midpoint coil hashes"
            ),
            "accepted_roots": tuple(
                _tuple_numbers(root, 255, f"accepted root {index}")
                for index, root in enumerate(accepted_roots_value)
            ),
        }
    )
    reduced_payload = _object(
        payload["native_reduced_endpoint"],
        NativeReducedEndpointTelemetry,
        "native reduced endpoint",
    )
    reduced = NativeReducedEndpointTelemetry(
        **{
            **_strict_scalar_fields(
                reduced_payload,
                "reduced endpoint",
                strings=("gradient_dtype",),
                hashes=("parameter_sha256",),
                booleans=("inner_solver_success",),
                numbers=(
                    "objective",
                    "gradient_infinity_norm",
                    "gradient_l2_norm",
                    "solver_residual_l2",
                    "solver_residual_infinity_norm",
                ),
            ),
            "gradient": _tuple_numbers(
                reduced_payload["gradient"], 461, "reduced gradient"
            ),
        }
    )
    audit = NativeEquivalentEndpointAudit(
        schema_version=_string(payload["schema_version"], "schema version"),
        plan_sha256=_sha256(payload["plan_sha256"], "plan hash"),
        audited_state_sha256=_sha256(payload["audited_state_sha256"], "state hash"),
        audited_coil_parameters_sha256=_sha256(
            payload["audited_coil_parameters_sha256"], "coil parameter hash"
        ),
        binding=binding,
        gpu_quality=gpu_quality,
        derivative_residual=derivative,
        physics_contract=physics,
        branch_replay=branch,
        gpu_endpoint_cross_evaluation=cross(
            payload["gpu_endpoint_cross_evaluation"], "GPU cross evaluation"
        ),
        native_endpoint_cross_evaluation=cross(
            payload["native_endpoint_cross_evaluation"], "native cross evaluation"
        ),
        native_reference_state_sha256=_sha256(
            payload["native_reference_state_sha256"], "native reference state hash"
        ),
        native_reduced_endpoint=reduced,
    )
    return audit


def validate_endpoint_audit_payload(value: JsonValue) -> bool:
    """Parse complete raw evidence and independently recompute its disposition."""

    return endpoint_audit_from_payload(value).passes()


def load_endpoint_audit_bytes(payload: bytes) -> NativeEquivalentEndpointAudit:
    """Load the sole canonical JSON encoding, rejecting duplicates and nonfinite data."""

    value = load_canonical_json_bytes(payload)
    return endpoint_audit_from_payload(value)


__all__ = (
    "PLAN_SHA256",
    "SCHEMA_VERSION",
    "BranchReplayAuditEvidence",
    "DerivativeResidualEvidence",
    "GpuQualityEvidence",
    "LatchedStateBindingEvidence",
    "NativeEquivalentEndpointAudit",
    "PhysicsContractEvidence",
    "SameStateCrossEvaluationEvidence",
    "endpoint_audit_bytes",
    "endpoint_audit_from_payload",
    "endpoint_audit_payload",
    "load_endpoint_audit_bytes",
    "produce_native_equivalent_endpoint_audit",
    "validate_endpoint_audit_payload",
)
