from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import cast

import benchmarks.single_stage_native_equivalent_endpoint_audit as audit_module
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.single_stage_fullspace_snapshot import SnapshotValidationError
from simsopt_jax.objectives.single_stage_fullspace import (
    FROZEN_LAYOUT,
    FullSpaceConstraints,
    FullSpaceEvaluation,
    FullSpaceObjectiveConfig,
    FullSpaceObservables,
    FullSpaceProblem,
    FullSpaceRawTerms,
)
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    ObjectiveResidualReconstruction,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    DerivativeTransposeCertificate,
    NativeEquivalentQualityPolicy,
    NativeEquivalentQualityResult,
)
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    NativeAcceptedIntervalEvidence,
    NativeAcceptedPathEvidence,
    NativeContinuationStep,
    NativeExplicitStateEvaluation,
    NativeFullSpaceObjectiveContract,
    NativeObjectiveTerms,
    NativeReducedEvaluation,
    NativeSingleStageEndpointRuntime,
    NativeStateObservables,
)

jax.config.update("jax_enable_x64", True)

_STATE_SIZE = 716
_EQUALITY_SIZE = 255


@dataclass(frozen=True, slots=True)
class _FakeLoop:
    accepted_steps: jax.Array
    accepted_optimizer_coordinates: jax.Array
    accepted_state_mask: jax.Array
    optimizer_coordinates: jax.Array


@dataclass(frozen=True, slots=True)
class _FakeOptimizer:
    optimizer_coordinates: jax.Array


@dataclass(frozen=True, slots=True)
class _FakeEndpoint:
    physical_state: jax.Array
    objective_gradient: jax.Array


@dataclass(frozen=True, slots=True)
class _FakeResult:
    loop_result: _FakeLoop
    optimizer_result: _FakeOptimizer
    accepted_physical_coordinates: jax.Array
    accepted_state_mask: jax.Array
    endpoint: _FakeEndpoint


@dataclass(frozen=True, slots=True)
class _FakeProblem:
    layout: object
    config: FullSpaceObjectiveConfig
    exact_mask_indices: jax.Array


class _FakeRuntime:
    def __init__(self, state: np.ndarray) -> None:
        self.bootstrap_state = np.array(state, copy=True)
        self.exact_mask_indices = np.arange(254, dtype=np.int64)
        self.fixed_first_base_current = 7.0
        self.returned_state_offset = 0.0
        self.objective_contract = NativeFullSpaceObjectiveContract(
            iota_target=0.0,
            major_radius_target=0.0,
            length_target=0.0,
            volume_target=0.0,
            non_qs_weight=1.0,
            residual_weight=1.0,
            iota_weight=1.0,
            major_radius_weight=1.0,
            length_weight=1.0,
            non_qs_axis=0,
            weight_inv_modB=False,
        )

    def evaluate_state(self, state: np.ndarray) -> NativeExplicitStateEvaluation:
        zero_terms = NativeObjectiveTerms(0.0, 0.0, 0.0, 0.0, 0.0)
        observables = NativeStateObservables(
            iota=0.0,
            G=0.0,
            volume=0.0,
            major_radius=0.0,
            total_length=0.0,
            non_qs_ratio=0.0,
            boozer_residual_value=0.0,
            boozer_residual_rms=0.0,
            fixed_first_base_current=7.0,
        )
        return NativeExplicitStateEvaluation(
            state=np.array(state, dtype=np.float64, copy=True)
            + self.returned_state_offset,
            state_little_endian_sha256="0" * 64,
            objective_terms=zero_terms,
            objective=0.0,
            observables=observables,
            masked_boozer_equalities=np.zeros((254,), dtype=np.float64),
            volume_equality=0.0,
            raw_equalities=np.zeros((_EQUALITY_SIZE,), dtype=np.float64),
            all_finite=False,
        )

    def audit_accepted_states(
        self,
        physical_states: np.ndarray,
    ) -> NativeAcceptedPathEvidence:
        coil_sha = _sha(physical_states[0, :461])
        root_sha = _sha(physical_states[0, 461:])
        bootstrap = replace(
            _step(1, 0, None, coil_sha, root_sha, root_sha),
            newton_iterations=0,
        )
        intervals = tuple(
            NativeAcceptedIntervalEvidence(
                index=index,
                supplied_state_little_endian_sha256=_sha(physical_states[index]),
                direct_step=_step(1, index, index - 1, coil_sha, root_sha, root_sha),
                midpoint_step=_step(
                    2, 2 * index - 1, 2 * index - 2, coil_sha, root_sha, root_sha
                ),
                refined_step=_step(
                    2, 2 * index, 2 * index - 1, coil_sha, root_sha, root_sha
                ),
                direct_root=np.zeros((255,), dtype=np.float64),
                midpoint_root=np.zeros((255,), dtype=np.float64),
                refined_root=np.zeros((255,), dtype=np.float64),
                direct_refined_infinity_difference=0.0,
                supplied_refined_infinity_difference=0.0,
            )
            for index in range(1, physical_states.shape[0])
        )
        return NativeAcceptedPathEvidence(
            bootstrap_step=bootstrap,
            intervals=intervals,
            first_failing_index=None,
            failure_reason=None,
            usable=False,
        )

    def evaluate_reduced(self, parameters: np.ndarray) -> NativeReducedEvaluation:
        gradient = np.zeros((461,), dtype=np.float64)
        return NativeReducedEvaluation(
            parameters=np.array(parameters, copy=True),
            objective=0.0,
            gradient=gradient,
            gradient_infinity_norm=0.0,
            gradient_l2_norm=0.0,
            inner_solver_success=False,
            solver_residual_l2=0.0,
            solver_residual_infinity_norm=0.0,
            all_finite=False,
        )


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def _step(
    segment_count: int,
    index: int,
    predecessor: int | None,
    coil_sha256: str,
    seed_sha256: str,
    root_sha256: str,
) -> NativeContinuationStep:
    raw_equalities = np.zeros((_EQUALITY_SIZE,), dtype=np.float64)
    return NativeContinuationStep(
        segment_count=segment_count,
        index=index,
        predecessor_index=predecessor,
        coil_little_endian_sha256=coil_sha256,
        seed_root_little_endian_sha256=seed_sha256,
        root_little_endian_sha256=root_sha256,
        newton_iterations=1,
        residual_l2=0.0,
        residual_infinity_norm=0.0,
        scaled_boozer_infinity_norm=0.0,
        raw_equalities=raw_equalities,
        raw_equalities_little_endian_sha256=_sha(raw_equalities),
    )


def _evaluation() -> FullSpaceEvaluation:
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return FullSpaceEvaluation(
        raw_terms=FullSpaceRawTerms(zero, zero, zero, zero, zero),
        weighted_total=zero,
        constraints=FullSpaceConstraints(
            jnp.zeros((254,), dtype=jnp.float64),
            zero,
        ),
        observables=FullSpaceObservables(
            zero, zero, zero, zero, zero, zero, zero, zero
        ),
    )


def _transpose() -> DerivativeTransposeCertificate:
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return DerivativeTransposeCertificate(
        primal_dot=zero,
        transpose_dot=zero,
        denominator=jnp.asarray(np.finfo(np.float64).tiny, dtype=jnp.float64),
        defect=zero,
        state_probe=jnp.asarray(
            audit_module._deterministic_state_probe(), dtype=jnp.float64
        ),
        equality_probe=jnp.asarray(
            audit_module._deterministic_equality_probe(), dtype=jnp.float64
        ),
        jvp_action=jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        vjp_action=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        all_finite=jnp.asarray(True),
        certified=jnp.asarray(True),
    )


def _fixtures() -> tuple[
    NativeEquivalentQualityResult,
    FullSpaceProblem,
    FullSpaceScaling,
    NativeEquivalentQualityPolicy,
    NativeSingleStageEndpointRuntime,
    np.ndarray,
]:
    state = jnp.zeros((_STATE_SIZE,), dtype=jnp.float64)
    ledger = jnp.zeros((257, _STATE_SIZE), dtype=jnp.float64)
    mask = jnp.arange(257) < 2
    result = _FakeResult(
        loop_result=_FakeLoop(
            accepted_steps=jnp.asarray(1, dtype=jnp.int32),
            accepted_optimizer_coordinates=ledger,
            accepted_state_mask=mask,
            optimizer_coordinates=state,
        ),
        optimizer_result=_FakeOptimizer(optimizer_coordinates=state),
        accepted_physical_coordinates=ledger,
        accepted_state_mask=mask,
        endpoint=_FakeEndpoint(
            physical_state=state,
            objective_gradient=state,
        ),
    )
    scalar = jnp.asarray(0.0, dtype=jnp.float64)
    config = FullSpaceObjectiveConfig(
        iota_target=scalar,
        major_radius_target=scalar,
        length_target=scalar,
        volume_target=scalar,
        non_qs_weight=jnp.asarray(1.0, dtype=jnp.float64),
        residual_weight=jnp.asarray(1.0, dtype=jnp.float64),
        iota_weight=jnp.asarray(1.0, dtype=jnp.float64),
        major_radius_weight=jnp.asarray(1.0, dtype=jnp.float64),
        length_weight=jnp.asarray(1.0, dtype=jnp.float64),
        non_qs_axis=0,
        weight_inv_modB=False,
        length_coil_indices=(0, 1, 2),
    )
    problem = _FakeProblem(
        layout=FROZEN_LAYOUT,
        config=config,
        exact_mask_indices=jnp.arange(254, dtype=jnp.int64),
    )
    scaling = FullSpaceScaling(
        bootstrap_anchor=state,
        variable_scale=jnp.ones_like(state),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    runtime = _FakeRuntime(np.zeros((_STATE_SIZE,), dtype=np.float64))
    raw_equalities = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
    policy = NativeEquivalentQualityPolicy(
        native_raw_equalities=raw_equalities,
        native_raw_equalities_sha256=exact_numeric_tree_sha256(raw_equalities),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    reference_state = np.zeros((_STATE_SIZE,), dtype=np.float64)
    return (
        cast(NativeEquivalentQualityResult, result),
        cast(FullSpaceProblem, problem),
        scaling,
        policy,
        cast(NativeSingleStageEndpointRuntime, runtime),
        reference_state,
    )


@pytest.fixture(autouse=True)
def _independent_numeric_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    reconstruction = ObjectiveResidualReconstruction(
        reconstructed_value=zero,
        authoritative_value=zero,
        value_scaled_defect=zero,
        gradient_scaled_defect=zero,
        residual_valid=jnp.asarray(True),
        all_finite=jnp.asarray(True),
    )
    monkeypatch.setattr(
        audit_module, "evaluate_fullspace", lambda _state, _problem: _evaluation()
    )
    monkeypatch.setattr(
        audit_module,
        "certify_fullspace_objective_residuals",
        lambda _state, _problem: reconstruction,
    )
    monkeypatch.setattr(
        audit_module,
        "fullspace_objective_residual_vector",
        lambda _state, _problem: jnp.zeros((1,), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        audit_module,
        "fullspace_value_and_grad",
        lambda _state, _problem: (
            zero,
            jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        audit_module,
        "deterministic_constraint_transpose_certificate",
        lambda _state, _problem, _scaling: _transpose(),
    )


def test_complete_raw_audit_passes_without_consulting_kkt() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())

    assert evidence.passes()
    assert evidence.binding.valid_row_count == 2
    assert evidence.branch_replay.successful_midpoint_refined_rows == 1
    assert evidence.gpu_endpoint_cross_evaluation.passes()
    assert evidence.native_endpoint_cross_evaluation.passes()


def test_latched_state_must_be_exactly_the_last_valid_ledger_row() -> None:
    result, problem, scaling, policy, runtime, reference = _fixtures()
    fake = cast(_FakeResult, result)
    changed_endpoint = replace(
        fake.endpoint,
        physical_state=fake.endpoint.physical_state.at[0].set(1.0),
    )
    changed = replace(fake, endpoint=changed_endpoint)

    evidence = audit_module.produce_native_equivalent_endpoint_audit(
        cast(NativeEquivalentQualityResult, changed),
        problem,
        scaling,
        policy,
        runtime,
        reference,
    )

    assert (
        evidence.binding.latched_state_sha256
        != evidence.binding.endpoint_physical_state_sha256
    )
    assert not evidence.passes()


def test_canonical_payload_round_trip_recomputes_the_raw_verdict() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    payload = audit_module.endpoint_audit_payload(evidence)

    parsed = audit_module.endpoint_audit_from_payload(payload)

    assert parsed.passes()
    derivative = payload["derivative_residual"]
    assert isinstance(derivative, dict)
    derivative["transpose_defect"] = 0.5
    assert not audit_module.endpoint_audit_from_payload(payload).passes()
    assert not audit_module.validate_endpoint_audit_payload(payload)


def test_timed_result_gradient_is_not_derivative_authority() -> None:
    result, problem, scaling, policy, runtime, reference = _fixtures()
    fake = cast(_FakeResult, result)
    changed = replace(
        fake,
        endpoint=replace(
            fake.endpoint,
            objective_gradient=fake.endpoint.objective_gradient.at[5].set(jnp.nan),
        ),
    )

    evidence = audit_module.produce_native_equivalent_endpoint_audit(
        cast(NativeEquivalentQualityResult, changed),
        problem,
        scaling,
        policy,
        runtime,
        reference,
    )

    assert evidence.derivative_residual.gradient_nonfinite_count == 0
    assert evidence.derivative_residual.passes()
    assert evidence.passes()


def test_cross_evaluator_recomputes_parity_from_raw_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, problem, scaling, policy, runtime, reference = _fixtures()
    mismatched = replace(
        _evaluation(), weighted_total=jnp.asarray(1.0, dtype=jnp.float64)
    )
    monkeypatch.setattr(
        audit_module,
        "evaluate_fullspace",
        lambda _state, _problem: mismatched,
    )

    evidence = audit_module.produce_native_equivalent_endpoint_audit(
        result,
        problem,
        scaling,
        policy,
        runtime,
        reference,
    )

    assert not evidence.gpu_endpoint_cross_evaluation.passes()
    assert not evidence.native_endpoint_cross_evaluation.passes()
    assert not evidence.passes()


def test_cross_evaluator_is_bound_to_the_native_returned_state() -> None:
    result, problem, scaling, policy, runtime, reference = _fixtures()
    fake_runtime = cast(_FakeRuntime, runtime)
    fake_runtime.returned_state_offset = 1.0

    evidence = audit_module.produce_native_equivalent_endpoint_audit(
        result, problem, scaling, policy, runtime, reference
    )

    assert (
        evidence.gpu_endpoint_cross_evaluation.requested_state_sha256
        != evidence.gpu_endpoint_cross_evaluation.native_returned_state_sha256
    )
    assert not evidence.passes()


def test_branch_predecessor_mutation_is_recomputed_from_raw_steps() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    interval = evidence.branch_replay.raw.intervals[0]
    bad_direct = replace(interval.direct_step, predecessor_index=None)
    bad_interval = replace(interval, direct_step=bad_direct)
    bad_raw = replace(evidence.branch_replay.raw, intervals=(bad_interval,))
    bad_branch = replace(
        evidence.branch_replay,
        raw=bad_raw,
        evidence_sha256=audit_module._canonical_sha256(
            audit_module._branch_payload(bad_raw)
        ),
    )

    assert not bad_branch.passes(evidence.binding.valid_row_count)


def test_transpose_defect_is_recomputed_from_raw_dots() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    fabricated = replace(
        evidence.derivative_residual,
        transpose_primal_dot=1.0,
        transpose_adjoint_dot=0.0,
        transpose_defect=0.0,
    )

    assert not fabricated.passes()

    changed_audited_gradient = list(evidence.derivative_residual.gradient)
    changed_audited_gradient[0] = 1.0
    assert not replace(
        evidence.derivative_residual,
        gradient=tuple(changed_audited_gradient),
    ).passes()


def test_mask_digest_is_compared_with_the_expected_prefix() -> None:
    result, problem, scaling, policy, runtime, reference = _fixtures()
    fake = cast(_FakeResult, result)
    bad_mask = fake.accepted_state_mask.at[5].set(True)
    changed = replace(fake, accepted_state_mask=bad_mask)

    evidence = audit_module.produce_native_equivalent_endpoint_audit(
        cast(NativeEquivalentQualityResult, changed),
        problem,
        scaling,
        policy,
        runtime,
        reference,
    )

    assert (
        evidence.binding.observed_mask_sha256 != evidence.binding.expected_mask_sha256
    )
    assert not evidence.passes()


def test_physical_ledger_is_recomputed_from_raw_anchor_and_scale() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    changed_anchor = np.array(evidence.binding.bootstrap_anchor, copy=True)
    changed_anchor[0] = 1.0
    fabricated = replace(evidence.binding, bootstrap_anchor=changed_anchor)

    assert not fabricated.passes()


def test_objective_and_gradient_defects_are_recomputed_from_independent_values() -> (
    None
):
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    changed_gradient = list(
        evidence.derivative_residual.reconstructed_objective_gradient
    )
    changed_gradient[0] = 1.0
    fabricated = replace(
        evidence.derivative_residual,
        reconstructed_objective_gradient=tuple(changed_gradient),
        residual_gradient_defect=0.0,
    )

    assert not fabricated.passes()


def test_deterministic_probe_formula_cannot_be_replaced() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    changed_probe = list(evidence.derivative_residual.state_probe)
    changed_probe[0] = -changed_probe[0]
    fabricated = replace(
        evidence.derivative_residual,
        state_probe=tuple(changed_probe),
    )

    assert not fabricated.passes()


def test_gpu_objective_is_recomputed_from_raw_terms_and_weights() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    changed_terms = list(evidence.gpu_quality.gpu_raw_objective_terms)
    changed_terms[0] = 1.0
    fabricated = replace(
        evidence.gpu_quality,
        gpu_raw_objective_terms=tuple(changed_terms),
    )

    assert not fabricated.passes()


def test_branch_step_summaries_are_bound_to_raw_equalities() -> None:
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    interval = evidence.branch_replay.raw.intervals[0]
    changed_raw = np.array(interval.direct_step.raw_equalities, copy=True)
    changed_raw[0] = 1.0e-15
    bad_step = replace(interval.direct_step, raw_equalities=changed_raw)
    bad_interval = replace(interval, direct_step=bad_step)
    bad_raw = replace(evidence.branch_replay.raw, intervals=(bad_interval,))
    bad_branch = replace(
        evidence.branch_replay,
        raw=bad_raw,
        evidence_sha256=audit_module._canonical_sha256(
            audit_module._branch_payload(bad_raw)
        ),
    )

    assert not bad_branch.passes(evidence.binding.valid_row_count)


def test_canonical_loader_rejects_duplicates_noncanonical_bytes_and_bool_counts() -> (
    None
):
    evidence = audit_module.produce_native_equivalent_endpoint_audit(*_fixtures())
    encoded = audit_module.endpoint_audit_bytes(evidence)

    assert audit_module.load_endpoint_audit_bytes(encoded).passes()
    with pytest.raises(SnapshotValidationError):
        audit_module.load_endpoint_audit_bytes(encoded + b" ")
    plan_entry = f'"plan_sha256":"{audit_module.PLAN_SHA256}",'.encode()
    with pytest.raises(SnapshotValidationError):
        audit_module.load_endpoint_audit_bytes(b"{" + plan_entry + encoded[1:])
    payload = audit_module.endpoint_audit_payload(evidence)
    binding = payload["binding"]
    assert isinstance(binding, dict)
    binding["accepted_step_count"] = True
    with pytest.raises(TypeError):
        audit_module.endpoint_audit_from_payload(payload)
