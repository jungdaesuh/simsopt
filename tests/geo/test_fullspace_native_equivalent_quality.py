from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.solve.fullspace_native_equivalent_quality as neq_module
from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonAcceptedState,
    ProjectedGaussNewtonFinalCertificate,
    ProjectedGaussNewtonHistory,
    ProjectedGaussNewtonLoopResult,
    ProjectedGaussNewtonOptions,
    ProjectedGaussNewtonResult,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceConstraints,
    FullSpaceEvaluation,
    FullSpaceObservables,
    FullSpaceProblem,
    FullSpaceRawTerms,
)
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    ObjectiveResidualReconstruction,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_gauss_newton_trust_region import (
    CFS_GNTR1_OPTIONS,
)
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NATIVE_OBJECTIVE_TARGET,
    NEQ_GNTR1_OPTIONS,
    NEQ_GNTR2_OPTIONS,
    NEQ_GNTR2_ROUTE,
    NEQ_GNTR2_SCHEMA_VERSION,
    NEQ_GNTR3_OPTIONS,
    NEQ_GNTR3_ROUTE,
    NEQ_GNTR3_SCHEMA_VERSION,
    ROUTE,
    SCHEMA_VERSION,
    KktTelemetryStatus,
    NativeEquivalentEndpointEvidence,
    NativeEquivalentQualityPolicy,
    NativeEquivalentQualityResult,
    PreparedNeqAcceptedQualityDiagnostics,
    PreparedNeqGntr1,
    PreparedNeqGntr2,
    PreparedNeqGntr3,
    PreparedNeqGntrRetractionCanary,
    accepted_state_meets_native_quality,
    build_native_equivalent_terminal_diagnostic,
    classify_kkt_telemetry,
    deterministic_constraint_transpose_certificate,
    native_equivalent_quality_margins,
    prepare_neq_accepted_quality_diagnostics,
    prepare_neq_gntr1,
    prepare_neq_gntr2,
    prepare_neq_gntr3,
    prepare_neq_gntr_retraction_canary,
    prepare_neq_terminal_endpoint_diagnostics,
)
from simsopt_jax.solve.fullspace_sqp import CfsSqp1EndpointDiagnostics

jax.config.update("jax_enable_x64", True)

_STATE_SIZE = 716
_EQUALITY_SIZE = 255


def _policy(
    *,
    raw_equalities: jax.Array | None = None,
    scale: jax.Array | None = None,
) -> NativeEquivalentQualityPolicy:
    raw = (
        jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
        if raw_equalities is None
        else raw_equalities
    )
    constraint_scale = (
        jnp.linspace(0.5, 1.5, _EQUALITY_SIZE, dtype=jnp.float64)
        if scale is None
        else scale
    )
    return NativeEquivalentQualityPolicy(
        native_raw_equalities=raw,
        native_raw_equalities_sha256=exact_numeric_tree_sha256(raw),
        constraint_inverse_scale=constraint_scale,
    )


def _accepted_state(
    policy: NativeEquivalentQualityPolicy,
    *,
    objective: float = NATIVE_OBJECTIVE_TARGET,
    raw_equalities: jax.Array | None = None,
    coordinates: jax.Array | None = None,
) -> ProjectedGaussNewtonAcceptedState:
    raw = (
        jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
        if raw_equalities is None
        else raw_equalities
    )
    scaled = raw * policy.constraint_inverse_scale
    return ProjectedGaussNewtonAcceptedState(
        optimizer_coordinates=(
            jnp.zeros((_STATE_SIZE,), dtype=jnp.float64)
            if coordinates is None
            else coordinates
        ),
        objective=jnp.asarray(objective, dtype=jnp.float64),
        constraints=scaled,
        scaled_feasibility_inf=jnp.linalg.norm(scaled, ord=jnp.inf),
    )


def _history() -> ProjectedGaussNewtonHistory:
    floating = jnp.zeros((1,), dtype=jnp.float64)
    integer = jnp.zeros((1,), dtype=jnp.int32)
    boolean = jnp.zeros((1,), dtype=jnp.bool_)
    values = {
        name: (
            boolean
            if name == "steihaug_hit_boundary"
            else integer
            if name
            in {
                "outcome",
                "accepted_step_number",
                "steihaug_iterations",
                "steihaug_hvp_evaluations",
                "steihaug_termination",
            }
            else floating
        )
        for name in ProjectedGaussNewtonHistory._fields
    }
    return ProjectedGaussNewtonHistory(**values)


def _optimizer_result(*, stationarity: float = 0.0) -> ProjectedGaussNewtonResult:
    coordinates = jnp.zeros((_STATE_SIZE,), dtype=jnp.float64)
    equalities = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
    scalar = jnp.asarray(stationarity, dtype=jnp.float64)
    certificate = ProjectedGaussNewtonFinalCertificate(
        coordinates_finite=jnp.asarray(True),
        residual_value_defect=jnp.asarray(0.0, dtype=jnp.float64),
        residual_gradient_defect=jnp.asarray(0.0, dtype=jnp.float64),
        hvp_symmetry_defect=jnp.asarray(0.0, dtype=jnp.float64),
        probe_normalized_curvature=jnp.asarray(0.0, dtype=jnp.float64),
        gram_factorization_relative_residual=jnp.asarray(0.0, dtype=jnp.float64),
        multiplier_relative_residual=jnp.asarray(0.0, dtype=jnp.float64),
        multiplier_forward_error_bound=jnp.asarray(0.0, dtype=jnp.float64),
        projection_tangency_relative_residual=jnp.asarray(0.0, dtype=jnp.float64),
        projection_solve_relative_residual=jnp.asarray(0.0, dtype=jnp.float64),
        projection_forward_error_bound=jnp.asarray(0.0, dtype=jnp.float64),
        all_finite=jnp.asarray(True),
        certified=jnp.asarray(True),
    )
    return ProjectedGaussNewtonResult(
        optimizer_coordinates=coordinates,
        multipliers=equalities,
        objective=jnp.asarray(NATIVE_OBJECTIVE_TARGET, dtype=jnp.float64),
        constraints=equalities,
        objective_gradient=coordinates,
        constraint_jacobian=jnp.zeros((_EQUALITY_SIZE, _STATE_SIZE), dtype=jnp.float64),
        stationarity=jnp.full_like(coordinates, scalar),
        scaled_stationarity_inf=scalar,
        scaled_feasibility_inf=jnp.asarray(0.0, dtype=jnp.float64),
        trust_radius=jnp.asarray(2.0**-10, dtype=jnp.float64),
        accepted_steps=jnp.asarray(1, dtype=jnp.int32),
        attempts=jnp.asarray(1, dtype=jnp.int32),
        retryable_rejections=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(6, dtype=jnp.int32),
        fatal=jnp.asarray(False),
        bounded_complete=jnp.asarray(False),
        mechanism_exercised=jnp.asarray(True),
        all_accepted_states_finite=jnp.asarray(True),
        all_finite=jnp.asarray(True),
        final_certificate=certificate,
        usable=jnp.asarray(True),
        history=_history(),
        device_quality_candidate_reached=jnp.asarray(True),
        first_quality_attempt=jnp.asarray(1, dtype=jnp.int32),
        first_quality_accepted_step=jnp.asarray(1, dtype=jnp.int32),
    )


def _loop_result() -> ProjectedGaussNewtonLoopResult:
    ledger = jnp.zeros((257, _STATE_SIZE), dtype=jnp.float64).at[1, 0].set(2.0)
    mask = jnp.zeros((257,), dtype=jnp.bool_).at[:2].set(True)
    optimizer = _optimizer_result()
    return ProjectedGaussNewtonLoopResult(
        optimizer_coordinates=ledger[1],
        trust_radius=optimizer.trust_radius,
        accepted_steps=jnp.asarray(1, dtype=jnp.int32),
        attempts=jnp.asarray(1, dtype=jnp.int32),
        retryable_rejections=jnp.asarray(0, dtype=jnp.int32),
        status=optimizer.status,
        fatal=jnp.asarray(False),
        bounded_complete=jnp.asarray(False),
        device_quality_candidate_reached=jnp.asarray(True),
        first_quality_attempt=jnp.asarray(1, dtype=jnp.int32),
        first_quality_accepted_step=jnp.asarray(1, dtype=jnp.int32),
        accepted_optimizer_coordinates=ledger,
        accepted_state_mask=mask,
        mechanism_exercised=jnp.asarray(True),
        all_accepted_states_finite=jnp.asarray(True),
        history=_history(),
    )


def _endpoint_evidence() -> NativeEquivalentEndpointEvidence:
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    raw_terms = FullSpaceRawTerms(zero, zero, zero, zero, zero)
    constraints = FullSpaceConstraints(jnp.zeros((254,), dtype=jnp.float64), zero)
    observables = FullSpaceObservables(zero, zero, zero, zero, zero, zero, zero, zero)
    evaluation = FullSpaceEvaluation(raw_terms, zero, constraints, observables)
    reconstruction = ObjectiveResidualReconstruction(
        zero, zero, zero, zero, jnp.asarray(True), jnp.asarray(True)
    )
    transpose = neq_module.DerivativeTransposeCertificate(
        zero,
        zero,
        jnp.asarray(1.0, dtype=jnp.float64),
        zero,
        jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    kkt = neq_module.KktTelemetry(
        jnp.asarray(int(KktTelemetryStatus.AVAILABLE_FINITE), dtype=jnp.int32),
        zero,
        zero,
    )
    truth = jnp.asarray(True)
    return NativeEquivalentEndpointEvidence(
        physical_state=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        evaluation=evaluation,
        objective_gradient=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        raw_equalities=jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        scaled_equalities=jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        residual_reconstruction=reconstruction,
        transpose_certificate=transpose,
        kkt_telemetry=kkt,
        objective_quality_valid=truth,
        equality_quality_valid=truth,
        scaled_feasibility_valid=truth,
        gradient_valid=truth,
        residual_contract_valid=truth,
        derivative_contract_valid=truth,
        all_finite=truth,
        local_audit_passed=truth,
    )


def _base_result(
    *,
    endpoint: NativeEquivalentEndpointEvidence | None = None,
    quality_latched: bool = False,
) -> NativeEquivalentQualityResult:
    selected_endpoint = _endpoint_evidence() if endpoint is None else endpoint
    loop = _loop_result()._replace(
        device_quality_candidate_reached=jnp.asarray(quality_latched),
        first_quality_attempt=jnp.asarray(1 if quality_latched else 0, dtype=jnp.int32),
        first_quality_accepted_step=jnp.asarray(
            1 if quality_latched else 0, dtype=jnp.int32
        ),
    )
    return NativeEquivalentQualityResult(
        schema_version="single-stage-fullspace-neq-gntr1-result-v1",
        route="NEQ-GNTR1",
        loop_result=loop,
        optimizer_result=_optimizer_result()._replace(
            optimizer_coordinates=loop.optimizer_coordinates
        ),
        accepted_physical_coordinates=(
            loop.accepted_optimizer_coordinates.at[loop.accepted_steps].set(
                selected_endpoint.physical_state
            )
        ),
        accepted_state_mask=loop.accepted_state_mask,
        endpoint=selected_endpoint,
        latched_state_exact=jnp.asarray(True),
        candidate_ready_for_external_audit=jnp.asarray(quality_latched),
    )


def _raw_endpoint(
    endpoint: NativeEquivalentEndpointEvidence,
    *,
    raw_stationarity: jax.Array | None = None,
) -> CfsSqp1EndpointDiagnostics:
    stationarity = (
        jnp.zeros((_STATE_SIZE,), dtype=jnp.float64)
        if raw_stationarity is None
        else raw_stationarity
    )
    multipliers = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
    return CfsSqp1EndpointDiagnostics(
        physical_state=endpoint.physical_state,
        physical_objective=endpoint.evaluation.weighted_total,
        raw_constraints=endpoint.raw_equalities,
        scaled_constraints=endpoint.scaled_equalities,
        scaled_multipliers=multipliers,
        raw_multipliers=multipliers,
        raw_stationarity_residual=stationarity,
        raw_constraint_infinity_norm=jnp.max(jnp.abs(endpoint.raw_equalities)),
        scaled_constraint_infinity_norm=jnp.max(jnp.abs(endpoint.scaled_equalities)),
        raw_kkt_stationarity_infinity_norm=jnp.max(jnp.abs(stationarity)),
        all_finite=jnp.all(jnp.isfinite(stationarity)),
    )


def test_policy_is_exact_hashed_fp64_and_dimension_bound() -> None:
    policy = _policy()

    assert policy.objective_target == NATIVE_OBJECTIVE_TARGET
    assert policy.state_size == _STATE_SIZE
    assert policy.equality_size == _EQUALITY_SIZE
    assert policy.native_raw_equalities_sha256 == exact_numeric_tree_sha256(
        policy.native_raw_equalities
    )
    assert (
        policy.policy_sha256
        == "3541ce9b200f81bb94ded5d32667f11d12528ce4ae206f2b1b6f2ed34e879169"
    )

    with pytest.raises(ValueError, match="does not match"):
        NativeEquivalentQualityPolicy(
            native_raw_equalities=policy.native_raw_equalities,
            native_raw_equalities_sha256="0" * 64,
            constraint_inverse_scale=policy.constraint_inverse_scale,
        )


def test_policy_identity_binds_scale_and_threshold_bytes() -> None:
    baseline = _policy()
    changed_scale = _policy(
        scale=baseline.constraint_inverse_scale.at[13].set(
            jnp.nextafter(
                baseline.constraint_inverse_scale[13],
                jnp.asarray(jnp.inf, dtype=jnp.float64),
            )
        )
    )
    changed_tolerance = NativeEquivalentQualityPolicy(
        native_raw_equalities=baseline.native_raw_equalities,
        native_raw_equalities_sha256=baseline.native_raw_equalities_sha256,
        constraint_inverse_scale=baseline.constraint_inverse_scale,
        component_relative_tolerance=np.nextafter(
            baseline.component_relative_tolerance,
            np.inf,
        ),
    )

    assert changed_scale.policy_sha256 != baseline.policy_sha256
    assert changed_tolerance.policy_sha256 != baseline.policy_sha256


@pytest.mark.parametrize("bad_value", [0.0, np.nan, np.inf])
def test_policy_rejects_zero_or_nonfinite_constraint_scale(bad_value: float) -> None:
    scale = jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64).at[17].set(bad_value)

    with pytest.raises(ValueError, match="finite and nonzero"):
        _policy(scale=scale)


def test_policy_rejects_nonfinite_reference_and_threshold() -> None:
    raw = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64).at[3].set(jnp.nan)
    scale = jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64)

    with pytest.raises(ValueError, match="native_raw_equalities must be finite"):
        NativeEquivalentQualityPolicy(
            native_raw_equalities=raw,
            native_raw_equalities_sha256="0" * 64,
            constraint_inverse_scale=scale,
        )
    finite_raw = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="thresholds"):
        NativeEquivalentQualityPolicy(
            native_raw_equalities=finite_raw,
            native_raw_equalities_sha256=exact_numeric_tree_sha256(finite_raw),
            constraint_inverse_scale=scale,
            transpose_defect_tolerance=np.nan,
        )


@pytest.mark.parametrize("objective_target", [0.0, -1.0])
def test_policy_preserves_finite_nonpositive_objective_compatibility(
    objective_target: float,
) -> None:
    raw = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)

    policy = NativeEquivalentQualityPolicy(
        native_raw_equalities=raw,
        native_raw_equalities_sha256=exact_numeric_tree_sha256(raw),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
        objective_target=objective_target,
    )

    assert policy.objective_target == objective_target


def test_policy_preserves_zero_defect_threshold_compatibility_and_ratios_fail_closed() -> (
    None
):
    raw = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
    policy = NativeEquivalentQualityPolicy(
        native_raw_equalities=raw,
        native_raw_equalities_sha256=exact_numeric_tree_sha256(raw),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
        objective_target=0.0,
        component_absolute_tolerance=0.0,
        component_relative_tolerance=0.0,
        scaled_feasibility_tolerance=0.0,
        residual_value_defect_tolerance=0.0,
        residual_gradient_defect_tolerance=0.0,
        transpose_defect_tolerance=0.0,
    )

    margins = native_equivalent_quality_margins(_endpoint_evidence(), policy)

    assert jnp.isinf(margins.objective_usage_ratio)
    assert jnp.isinf(margins.component_usage_ratio)
    assert jnp.isinf(margins.scaled_feasibility_usage_ratio)
    assert not margins.all_finite


def test_neq_options_change_only_bounded_loop_limits() -> None:
    assert NEQ_GNTR1_OPTIONS.maximum_accepted_steps == 256
    assert NEQ_GNTR1_OPTIONS.maximum_attempts == 300
    assert NEQ_GNTR1_OPTIONS.initial_trust_radius == (
        CFS_GNTR1_OPTIONS.initial_trust_radius
    )
    assert NEQ_GNTR1_OPTIONS.minimum_trust_radius == (
        CFS_GNTR1_OPTIONS.minimum_trust_radius
    )
    assert NEQ_GNTR1_OPTIONS.maximum_trust_radius == (
        CFS_GNTR1_OPTIONS.maximum_trust_radius
    )
    assert NEQ_GNTR1_OPTIONS.maximum_steihaug_iterations == (
        CFS_GNTR1_OPTIONS.maximum_steihaug_iterations
    )


def test_quality_predicate_accepts_exact_objective_and_component_boundary() -> None:
    policy = _policy()
    boundary = policy.component_absolute_tolerance
    raw = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64).at[9].set(boundary)

    assert accepted_state_meets_native_quality(
        _accepted_state(policy, raw_equalities=raw), policy
    )
    assert not accepted_state_meets_native_quality(
        _accepted_state(
            policy,
            objective=np.nextafter(NATIVE_OBJECTIVE_TARGET, np.inf),
            raw_equalities=raw,
        ),
        policy,
    )


def test_quality_component_rule_is_ulp_sharp_and_recovers_raw_q() -> None:
    policy = _policy()
    boundary = policy.component_absolute_tolerance
    below = np.nextafter(boundary, 0.0)
    above = np.nextafter(boundary, np.inf)
    raw_below = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64).at[41].set(below)
    raw_above = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64).at[41].set(above)

    assert accepted_state_meets_native_quality(
        _accepted_state(policy, raw_equalities=raw_below), policy
    )
    assert not accepted_state_meets_native_quality(
        _accepted_state(policy, raw_equalities=raw_above), policy
    )


def test_quality_predicate_rejects_nonfinite_corrected_state() -> None:
    policy = _policy()
    coordinates = jnp.zeros((_STATE_SIZE,), dtype=jnp.float64).at[5].set(jnp.nan)

    compiled_predicate = jax.jit(
        lambda state: accepted_state_meets_native_quality(state, policy)
    )
    assert not compiled_predicate(_accepted_state(policy, coordinates=coordinates))


def test_transpose_certificate_uses_frozen_scaled_coordinate_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = jnp.reshape(
        jnp.sin(jnp.arange(_EQUALITY_SIZE * _STATE_SIZE, dtype=jnp.float64) * 1.0e-3),
        (_EQUALITY_SIZE, _STATE_SIZE),
    )
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.linspace(-0.1, 0.1, _STATE_SIZE, dtype=jnp.float64),
        variable_scale=jnp.linspace(0.5, 1.5, _STATE_SIZE, dtype=jnp.float64),
        constraint_inverse_scale=jnp.linspace(
            0.75, 1.25, _EQUALITY_SIZE, dtype=jnp.float64
        ),
    )
    monkeypatch.setattr(
        neq_module,
        "fullspace_constraint_vector",
        lambda physical, _problem: matrix @ physical,
    )
    problem = cast("FullSpaceProblem", object())
    coordinates = jnp.linspace(-0.2, 0.2, _STATE_SIZE, dtype=jnp.float64)

    compiled_certificate = jax.jit(
        lambda candidate: deterministic_constraint_transpose_certificate(
            candidate,
            problem,
            scaling,
        )
    )
    certificate = compiled_certificate(coordinates)
    expected_jacobian = (
        scaling.constraint_inverse_scale[:, None]
        * matrix
        * scaling.variable_scale[None, :]
    )

    np.testing.assert_allclose(
        certificate.jvp_action,
        expected_jacobian @ certificate.state_probe,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        certificate.vjp_action,
        expected_jacobian.T @ certificate.equality_probe,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert certificate.all_finite
    assert certificate.certified


def test_kkt_status_is_typed_and_nonfinite_or_unavailable_is_representable() -> None:
    finite = classify_kkt_telemetry(_optimizer_result(stationarity=10.0))
    nonfinite = classify_kkt_telemetry(_optimizer_result(stationarity=np.nan))
    unavailable = classify_kkt_telemetry(_optimizer_result(), available=False)

    assert finite.status == int(KktTelemetryStatus.AVAILABLE_FINITE)
    assert finite.scaled_stationarity_infinity_norm == 10.0
    assert nonfinite.status == int(KktTelemetryStatus.AVAILABLE_NONFINITE)
    assert unavailable.status == int(KktTelemetryStatus.UNAVAILABLE)
    assert jnp.isfinite(unavailable.scaled_stationarity_infinity_norm)


def test_post_timing_finalizer_maps_only_and_does_not_replace_latched_state() -> None:
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.ones((_STATE_SIZE,), dtype=jnp.float64),
        variable_scale=jnp.full((_STATE_SIZE,), 0.5, dtype=jnp.float64),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    policy = _policy(scale=scaling.constraint_inverse_scale)
    loop_result = _loop_result()
    optimizer_result = _optimizer_result()._replace(
        optimizer_coordinates=loop_result.optimizer_coordinates
    )
    finalize_calls: list[ProjectedGaussNewtonLoopResult] = []

    def finalize(
        supplied_loop: ProjectedGaussNewtonLoopResult,
    ) -> tuple[ProjectedGaussNewtonResult, NativeEquivalentEndpointEvidence]:
        finalize_calls.append(supplied_loop)
        return optimizer_result, _endpoint_evidence()

    prepared = PreparedNeqGntr1(
        problem=cast("FullSpaceProblem", object()),
        scaling=scaling,
        policy=policy,
        initial_physical_state=scaling.bootstrap_anchor,
        initial_optimizer_coordinates=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        options=NEQ_GNTR1_OPTIONS,
        _run_loop=lambda _coordinates: loop_result,
        _finalize=finalize,
        _map_ledger=jax.jit(
            lambda ledger: (
                scaling.bootstrap_anchor[None, :]
                + scaling.variable_scale[None, :] * ledger
            )
        ),
    )

    timed = prepared.run_solver_loop()
    synchronized = jax.block_until_ready(timed)
    assert not finalize_calls
    result = prepared.finalize_result(synchronized)

    assert len(finalize_calls) == 1
    assert finalize_calls[0] is synchronized
    assert timed.first_quality_attempt == 1
    assert timed.first_quality_accepted_step == 1
    assert result.latched_state_exact
    assert result.candidate_ready_for_external_audit
    np.testing.assert_array_equal(result.accepted_state_mask, timed.accepted_state_mask)
    np.testing.assert_array_equal(
        result.accepted_physical_coordinates[1],
        scaling.bootstrap_anchor + scaling.variable_scale * timed.optimizer_coordinates,
    )
    np.testing.assert_array_equal(
        result.optimizer_result.optimizer_coordinates,
        timed.optimizer_coordinates,
    )


def test_post_timing_audit_does_not_gate_on_high_kkt() -> None:
    evidence = _endpoint_evidence()._replace(
        kkt_telemetry=classify_kkt_telemetry(_optimizer_result(stationarity=1.0e3))
    )

    assert evidence.local_audit_passed
    assert evidence.kkt_telemetry.scaled_stationarity_infinity_norm == 1.0e3


def test_prepare_exposes_separate_compiled_loop_and_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        variable_scale=jnp.ones((_STATE_SIZE,), dtype=jnp.float64),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    policy = _policy(scale=scaling.constraint_inverse_scale)
    expected_loop = _loop_result()
    expected_optimizer = _optimizer_result()._replace(
        optimizer_coordinates=expected_loop.optimizer_coordinates
    )
    loop_options: list[ProjectedGaussNewtonOptions] = []
    finalizer_options: list[ProjectedGaussNewtonOptions] = []

    def run_loop(
        _joint: object,
        _residual: object,
        _coordinates: jax.Array,
        **kwargs: object,
    ) -> ProjectedGaussNewtonLoopResult:
        loop_options.append(cast("ProjectedGaussNewtonOptions", kwargs["options"]))
        return expected_loop

    def finalize(
        _joint: object,
        _residual: object,
        _loop: ProjectedGaussNewtonLoopResult,
        **kwargs: object,
    ) -> ProjectedGaussNewtonResult:
        finalizer_options.append(cast("ProjectedGaussNewtonOptions", kwargs["options"]))
        return expected_optimizer

    monkeypatch.setattr(
        neq_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem: scaling,
    )
    monkeypatch.setattr(
        neq_module,
        "fullspace_objective_residual_vector",
        lambda _physical, _problem: jnp.zeros((2110,), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        neq_module,
        "cfs_sqp1_joint_value_constraints",
        lambda coordinates, _problem, _scaling: (
            jnp.vdot(coordinates, coordinates),
            jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        neq_module,
        "run_projected_gauss_newton_trust_region_loop",
        run_loop,
    )
    monkeypatch.setattr(
        neq_module,
        "finalize_projected_gauss_newton_trust_region",
        finalize,
    )
    monkeypatch.setattr(
        neq_module,
        "_independent_endpoint_evidence",
        lambda _optimizer, _problem, _scaling, _policy: _endpoint_evidence(),
    )
    problem = cast("FullSpaceProblem", object())

    prepared = prepare_neq_gntr1(
        problem,
        scaling.bootstrap_anchor,
        scaling.bootstrap_anchor,
        policy,
    )
    timed = prepared.run_solver_loop()
    result = prepared.finalize_result(timed)

    assert timed.device_quality_candidate_reached
    assert timed.first_quality_attempt == 1
    assert prepared.options is NEQ_GNTR1_OPTIONS
    assert loop_options == [NEQ_GNTR1_OPTIONS]
    assert finalizer_options == [NEQ_GNTR1_OPTIONS]
    assert result.schema_version == "single-stage-fullspace-neq-gntr1-result-v1"
    assert result.route == "NEQ-GNTR1"
    assert result.candidate_ready_for_external_audit
    assert result.accepted_physical_coordinates.shape == (257, _STATE_SIZE)


def test_neq_gntr2_binds_max_two_identity_and_remains_receipt_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        variable_scale=jnp.ones((_STATE_SIZE,), dtype=jnp.float64),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    policy = _policy(scale=scaling.constraint_inverse_scale)
    legacy_policy_sha256 = policy.policy_sha256
    bootstrap_state = scaling.bootstrap_anchor.at[3].set(-0.5)
    initial_physical_state = scaling.bootstrap_anchor.at[19].set(0.25)
    problem = cast("FullSpaceProblem", ("fullspace-problem", 1))
    expected_loop = _loop_result()._replace(
        history=_history()._replace(
            nonlinear_corrections=jnp.asarray([2], dtype=jnp.int32),
            correction_step_ratio=jnp.asarray([8.0e-4], dtype=jnp.float64),
            maximum_individual_correction_step_ratio=jnp.asarray(
                [9.0e-4], dtype=jnp.float64
            ),
            correction_path_step_ratio=jnp.asarray([1.7e-3], dtype=jnp.float64),
        )
    )
    expected_optimizer = _optimizer_result()._replace(
        optimizer_coordinates=expected_loop.optimizer_coordinates,
        history=expected_loop.history,
    )
    observed_options: list[ProjectedGaussNewtonOptions] = []

    def run_loop(
        _joint: object,
        _residual: object,
        _coordinates: jax.Array,
        **kwargs: object,
    ) -> ProjectedGaussNewtonLoopResult:
        selected_options = cast("ProjectedGaussNewtonOptions", kwargs["options"])
        observed_options.append(selected_options)
        return expected_loop._replace(
            history=expected_loop.history._replace(
                nonlinear_corrections=jnp.asarray(
                    [selected_options.maximum_nonlinear_corrections],
                    dtype=jnp.int32,
                )
            )
        )

    def finalize(
        _joint: object,
        _residual: object,
        _loop: ProjectedGaussNewtonLoopResult,
        **kwargs: object,
    ) -> ProjectedGaussNewtonResult:
        observed_options.append(cast("ProjectedGaussNewtonOptions", kwargs["options"]))
        return expected_optimizer

    monkeypatch.setattr(
        neq_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem: scaling,
    )
    monkeypatch.setattr(
        neq_module,
        "fullspace_objective_residual_vector",
        lambda _physical, _problem: jnp.zeros((2110,), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        neq_module,
        "cfs_sqp1_joint_value_constraints",
        lambda coordinates, _problem, _scaling: (
            jnp.vdot(coordinates, coordinates),
            jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        neq_module,
        "run_projected_gauss_newton_trust_region_loop",
        run_loop,
    )
    monkeypatch.setattr(
        neq_module,
        "finalize_projected_gauss_newton_trust_region",
        finalize,
    )
    monkeypatch.setattr(
        neq_module,
        "_independent_endpoint_evidence",
        lambda _optimizer, _problem, _scaling, _policy: _endpoint_evidence(),
    )

    prepared = prepare_neq_gntr2(
        problem,
        bootstrap_state,
        initial_physical_state,
        policy,
    )
    timed = prepared.run_solver_loop()
    result = prepared.finalize_result(timed)

    assert isinstance(prepared, PreparedNeqGntr2)
    assert prepared.options is NEQ_GNTR2_OPTIONS
    assert NEQ_GNTR2_OPTIONS.maximum_nonlinear_corrections == 2
    for option_field in NEQ_GNTR1_OPTIONS.__dataclass_fields__:
        if option_field == "maximum_nonlinear_corrections":
            continue
        assert getattr(NEQ_GNTR2_OPTIONS, option_field) == getattr(
            NEQ_GNTR1_OPTIONS, option_field
        )
    assert observed_options
    assert all(observed is NEQ_GNTR2_OPTIONS for observed in observed_options)
    assert timed.history.nonlinear_corrections[0] == 2
    assert timed.history.correction_step_ratio[0] == 8.0e-4
    assert timed.history.maximum_individual_correction_step_ratio[0] == 9.0e-4
    assert timed.history.correction_path_step_ratio[0] == 1.7e-3
    assert result.schema_version == NEQ_GNTR2_SCHEMA_VERSION
    assert result.route == NEQ_GNTR2_ROUTE
    assert result.schema_version != SCHEMA_VERSION
    assert result.route != ROUTE
    assert result.identity is prepared.identity
    assert result.identity.schema_version == result.schema_version
    assert result.identity.route == result.route
    assert result.identity.base_neq_gntr1_policy_sha256 == legacy_policy_sha256
    assert result.identity.problem_sha256 == exact_numeric_tree_sha256(problem)
    assert result.identity.scaling_sha256 == exact_numeric_tree_sha256(scaling)
    assert result.identity.bootstrap_state_sha256 == exact_numeric_tree_sha256(
        bootstrap_state
    )
    assert result.identity.initial_physical_state_sha256 == exact_numeric_tree_sha256(
        initial_physical_state
    )
    assert result.latched_state_exact
    assert result.candidate_ready_for_external_audit
    assert "promotion_eligible" not in result._fields
    assert policy.policy_sha256 == legacy_policy_sha256

    changed_problem = prepare_neq_gntr2(
        cast("FullSpaceProblem", ("fullspace-problem", 2)),
        bootstrap_state,
        initial_physical_state,
        policy,
    )
    assert changed_problem.identity.problem_sha256 != prepared.identity.problem_sha256
    assert changed_problem.identity.identity_sha256 != prepared.identity.identity_sha256

    legacy = prepare_neq_gntr1(
        problem,
        bootstrap_state,
        initial_physical_state,
        policy,
    )
    legacy_timed = legacy.run_solver_loop()
    legacy_result = legacy.finalize_result(legacy_timed)
    assert legacy.options is NEQ_GNTR1_OPTIONS
    assert legacy.options.maximum_nonlinear_corrections == 1
    assert legacy_result.schema_version == SCHEMA_VERSION
    assert legacy_result.route == ROUTE
    assert policy.policy_sha256 == legacy_policy_sha256


def test_legacy_gntr_options_identity_domains_remain_byte_exact() -> None:
    assert (
        neq_module._projected_gntr_options_sha256(NEQ_GNTR1_OPTIONS)
        == "dcd481184681563551a0631d1da93b8ec0f12aacc81d4dd2e4a1e55cdc9787f7"
    )
    assert (
        neq_module._projected_gntr_options_sha256(NEQ_GNTR2_OPTIONS)
        == "ed187997094f68888c2c36009551a10dec510ac6661766306c2832f621d74460"
    )
    assert neq_module._projected_gntr_options_sha256(
        replace(NEQ_GNTR1_OPTIONS, enable_step_bound_safeguard=True)
    ) == neq_module._projected_gntr_options_sha256(NEQ_GNTR1_OPTIONS)
    assert neq_module._projected_gntr_options_sha256(
        replace(NEQ_GNTR2_OPTIONS, enable_step_bound_safeguard=True)
    ) == neq_module._projected_gntr_options_sha256(NEQ_GNTR2_OPTIONS)


def test_neq_gntr3_binds_safeguarded_identity_and_remains_receipt_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        variable_scale=jnp.ones((_STATE_SIZE,), dtype=jnp.float64),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    policy = _policy(scale=scaling.constraint_inverse_scale)
    policy_sha256 = policy.policy_sha256
    bootstrap_state = scaling.bootstrap_anchor.at[3].set(-0.5)
    initial_physical_state = scaling.bootstrap_anchor.at[19].set(0.25)
    problem = cast("FullSpaceProblem", ("fullspace-problem", 1))
    expected_history = _history()._replace(
        nonlinear_corrections=jnp.asarray([2], dtype=jnp.int32),
        subtrial_count=jnp.asarray([2], dtype=jnp.int32),
        selected_subtrial_index=jnp.asarray([1], dtype=jnp.int32),
    )
    expected_loop = _loop_result()._replace(history=expected_history)
    expected_optimizer = _optimizer_result()._replace(
        optimizer_coordinates=expected_loop.optimizer_coordinates,
        history=expected_history,
    )
    observed_options: list[ProjectedGaussNewtonOptions] = []

    def run_loop(
        _joint: object,
        _residual: object,
        _coordinates: jax.Array,
        **kwargs: object,
    ) -> ProjectedGaussNewtonLoopResult:
        selected_options = cast("ProjectedGaussNewtonOptions", kwargs["options"])
        observed_options.append(selected_options)
        return expected_loop._replace(
            history=expected_loop.history._replace(
                nonlinear_corrections=jnp.asarray(
                    [selected_options.maximum_nonlinear_corrections],
                    dtype=jnp.int32,
                ),
                subtrial_count=jnp.asarray(
                    [2 if selected_options.enable_step_bound_safeguard else 0],
                    dtype=jnp.int32,
                ),
            )
        )

    def finalize(
        _joint: object,
        _residual: object,
        _loop: ProjectedGaussNewtonLoopResult,
        **kwargs: object,
    ) -> ProjectedGaussNewtonResult:
        observed_options.append(cast("ProjectedGaussNewtonOptions", kwargs["options"]))
        return expected_optimizer

    monkeypatch.setattr(
        neq_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem: scaling,
    )
    monkeypatch.setattr(
        neq_module,
        "fullspace_objective_residual_vector",
        lambda _physical, _problem: jnp.zeros((2110,), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        neq_module,
        "cfs_sqp1_joint_value_constraints",
        lambda coordinates, _problem, _scaling: (
            jnp.vdot(coordinates, coordinates),
            jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        neq_module,
        "run_projected_gauss_newton_trust_region_loop",
        run_loop,
    )
    monkeypatch.setattr(
        neq_module,
        "finalize_projected_gauss_newton_trust_region",
        finalize,
    )
    monkeypatch.setattr(
        neq_module,
        "_independent_endpoint_evidence",
        lambda _optimizer, _problem, _scaling, _policy: _endpoint_evidence(),
    )

    prepared = prepare_neq_gntr3(
        problem,
        bootstrap_state,
        initial_physical_state,
        policy,
    )
    timed = prepared.run_solver_loop()
    result = prepared.finalize_result(timed)

    assert isinstance(prepared, PreparedNeqGntr3)
    assert prepared.options is NEQ_GNTR3_OPTIONS
    assert NEQ_GNTR3_OPTIONS.maximum_nonlinear_corrections == 2
    assert NEQ_GNTR3_OPTIONS.enable_step_bound_safeguard
    changed_option_fields = [
        option_field
        for option_field in NEQ_GNTR2_OPTIONS.__dataclass_fields__
        if getattr(NEQ_GNTR2_OPTIONS, option_field)
        != getattr(NEQ_GNTR3_OPTIONS, option_field)
    ]
    assert changed_option_fields == ["enable_step_bound_safeguard"]
    assert observed_options
    assert all(observed is NEQ_GNTR3_OPTIONS for observed in observed_options)
    assert timed.history.nonlinear_corrections[0] == 2
    assert timed.history.subtrial_count[0] == 2
    assert result.schema_version == NEQ_GNTR3_SCHEMA_VERSION
    assert result.route == NEQ_GNTR3_ROUTE
    assert result.schema_version not in {SCHEMA_VERSION, NEQ_GNTR2_SCHEMA_VERSION}
    assert result.route not in {ROUTE, NEQ_GNTR2_ROUTE}
    assert result.identity is prepared.identity
    assert result.identity.schema_version == result.schema_version
    assert result.identity.route == result.route
    assert result.identity.base_neq_gntr1_policy_sha256 == policy_sha256
    assert result.identity.problem_sha256 == exact_numeric_tree_sha256(problem)
    assert result.identity.optimizer_options_sha256 == (
        neq_module._projected_gntr3_options_sha256(NEQ_GNTR3_OPTIONS)
    )
    assert result.identity.scaling_sha256 == exact_numeric_tree_sha256(scaling)
    assert result.identity.bootstrap_state_sha256 == exact_numeric_tree_sha256(
        bootstrap_state
    )
    assert result.identity.initial_physical_state_sha256 == exact_numeric_tree_sha256(
        initial_physical_state
    )
    assert result.latched_state_exact
    assert result.candidate_ready_for_external_audit
    assert "promotion_eligible" not in result._fields
    assert policy.policy_sha256 == policy_sha256
    assert not NEQ_GNTR1_OPTIONS.enable_step_bound_safeguard
    assert not NEQ_GNTR2_OPTIONS.enable_step_bound_safeguard

    changed_problem = prepare_neq_gntr3(
        cast("FullSpaceProblem", ("fullspace-problem", 2)),
        bootstrap_state,
        initial_physical_state,
        policy,
    )
    assert changed_problem.identity.problem_sha256 != prepared.identity.problem_sha256
    assert changed_problem.identity.identity_sha256 != prepared.identity.identity_sha256

    changed_bootstrap = prepare_neq_gntr3(
        problem,
        bootstrap_state.at[7].set(0.125),
        initial_physical_state,
        policy,
    )
    assert changed_bootstrap.identity.bootstrap_state_sha256 != (
        prepared.identity.bootstrap_state_sha256
    )
    assert (
        changed_bootstrap.identity.identity_sha256 != prepared.identity.identity_sha256
    )

    changed_initial = prepare_neq_gntr3(
        problem,
        bootstrap_state,
        initial_physical_state.at[11].set(-0.25),
        policy,
    )
    assert changed_initial.identity.initial_physical_state_sha256 != (
        prepared.identity.initial_physical_state_sha256
    )
    assert changed_initial.identity.identity_sha256 != prepared.identity.identity_sha256

    changed_policy = NativeEquivalentQualityPolicy(
        native_raw_equalities=policy.native_raw_equalities,
        native_raw_equalities_sha256=policy.native_raw_equalities_sha256,
        constraint_inverse_scale=policy.constraint_inverse_scale,
        component_relative_tolerance=np.nextafter(
            policy.component_relative_tolerance, np.inf
        ),
    )
    changed_policy_prepared = prepare_neq_gntr3(
        problem,
        bootstrap_state,
        initial_physical_state,
        changed_policy,
    )
    assert changed_policy_prepared.identity.base_neq_gntr1_policy_sha256 != (
        prepared.identity.base_neq_gntr1_policy_sha256
    )
    assert changed_policy_prepared.identity.identity_sha256 != (
        prepared.identity.identity_sha256
    )

    changed_scaling = FullSpaceScaling(
        bootstrap_anchor=scaling.bootstrap_anchor,
        variable_scale=scaling.variable_scale.at[13].set(
            jnp.nextafter(
                scaling.variable_scale[13],
                jnp.asarray(jnp.inf, dtype=jnp.float64),
            )
        ),
        constraint_inverse_scale=scaling.constraint_inverse_scale,
    )
    with monkeypatch.context() as scaling_patch:
        scaling_patch.setattr(
            neq_module,
            "fullspace_scaling_from_bootstrap",
            lambda _bootstrap, _problem: changed_scaling,
        )
        changed_scaling_prepared = prepare_neq_gntr3(
            problem,
            bootstrap_state,
            initial_physical_state,
            policy,
        )
    assert changed_scaling_prepared.identity.scaling_sha256 != (
        prepared.identity.scaling_sha256
    )
    assert changed_scaling_prepared.identity.identity_sha256 != (
        prepared.identity.identity_sha256
    )

    mutated_options = replace(
        NEQ_GNTR3_OPTIONS,
        enable_step_bound_safeguard=False,
    )
    with monkeypatch.context() as options_patch:
        options_patch.setattr(neq_module, "NEQ_GNTR3_OPTIONS", mutated_options)
        changed_options_prepared = prepare_neq_gntr3(
            problem,
            bootstrap_state,
            initial_physical_state,
            policy,
        )
    assert changed_options_prepared.identity.optimizer_options_sha256 != (
        prepared.identity.optimizer_options_sha256
    )
    assert changed_options_prepared.identity.identity_sha256 != (
        prepared.identity.identity_sha256
    )


def test_retraction_canary_binds_options_state_telemetry_and_stays_non_promoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        variable_scale=jnp.ones((_STATE_SIZE,), dtype=jnp.float64),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    policy = _policy(scale=scaling.constraint_inverse_scale)
    policy_sha256 = policy.policy_sha256
    initial_physical_state = scaling.bootstrap_anchor.at[19].set(0.25)
    options = replace(NEQ_GNTR1_OPTIONS, maximum_nonlinear_corrections=2)
    problem = cast("FullSpaceProblem", ("fullspace-problem", 1))
    canary_history = _history()._replace(
        nonlinear_corrections=jnp.asarray([2], dtype=jnp.int32)
    )
    expected_loop = _loop_result()._replace(history=canary_history)
    expected_optimizer = _optimizer_result()._replace(
        optimizer_coordinates=expected_loop.optimizer_coordinates,
        history=canary_history,
    )
    observed_options: list[ProjectedGaussNewtonOptions] = []
    observed_quality_predicates: list[object] = []

    def run_loop(
        _joint: object,
        _residual: object,
        _coordinates: jax.Array,
        **kwargs: object,
    ) -> ProjectedGaussNewtonLoopResult:
        selected_options = cast("ProjectedGaussNewtonOptions", kwargs["options"])
        observed_options.append(selected_options)
        observed_quality_predicates.append(kwargs["accepted_state_quality_predicate"])
        return expected_loop._replace(
            history=expected_loop.history._replace(
                nonlinear_corrections=jnp.asarray(
                    [selected_options.maximum_nonlinear_corrections],
                    dtype=jnp.int32,
                )
            )
        )

    def finalize(
        _joint: object,
        _residual: object,
        _loop: ProjectedGaussNewtonLoopResult,
        **kwargs: object,
    ) -> ProjectedGaussNewtonResult:
        observed_options.append(cast("ProjectedGaussNewtonOptions", kwargs["options"]))
        return expected_optimizer

    monkeypatch.setattr(
        neq_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem: scaling,
    )
    monkeypatch.setattr(
        neq_module,
        "fullspace_objective_residual_vector",
        lambda _physical, _problem: jnp.zeros((2110,), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        neq_module,
        "cfs_sqp1_joint_value_constraints",
        lambda coordinates, _problem, _scaling: (
            jnp.vdot(coordinates, coordinates),
            jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        neq_module,
        "run_projected_gauss_newton_trust_region_loop",
        run_loop,
    )
    monkeypatch.setattr(
        neq_module,
        "finalize_projected_gauss_newton_trust_region",
        finalize,
    )
    monkeypatch.setattr(
        neq_module,
        "_independent_endpoint_evidence",
        lambda _optimizer, _problem, _scaling, _policy: _endpoint_evidence(),
    )

    prepared = prepare_neq_gntr_retraction_canary(
        problem,
        scaling.bootstrap_anchor,
        initial_physical_state,
        policy,
        options=options,
    )
    timed = prepared.run_solver_loop()
    result = prepared.finalize_result(timed)

    assert isinstance(prepared, PreparedNeqGntrRetractionCanary)
    assert prepared.options is options
    np.testing.assert_array_equal(
        prepared.initial_physical_state,
        initial_physical_state,
    )
    np.testing.assert_array_equal(
        prepared.initial_optimizer_coordinates,
        initial_physical_state,
    )
    assert observed_options
    assert all(observed is options for observed in observed_options)
    assert observed_quality_predicates
    quality_predicate = cast(
        "Callable[[ProjectedGaussNewtonAcceptedState], jax.Array]",
        observed_quality_predicates[0],
    )
    assert quality_predicate(_accepted_state(policy))
    assert timed.history.nonlinear_corrections[0] == 2
    assert (
        timed.history.nonlinear_corrections[0] <= options.maximum_nonlinear_corrections
    )
    assert result.identity is prepared.identity
    assert result.identity.route == "NEQ-GNTR-RETRACTION-CANARY"
    assert result.identity.schema_version != SCHEMA_VERSION
    assert result.identity.base_neq_gntr1_policy_sha256 == policy_sha256
    assert result.identity.problem_sha256 == exact_numeric_tree_sha256(problem)
    assert len(result.identity.optimizer_options_sha256) == 64
    assert len(result.identity.identity_sha256) == 64
    assert policy.policy_sha256 == policy_sha256
    assert NEQ_GNTR1_OPTIONS.maximum_nonlinear_corrections == 1
    assert result.accepted_physical_coordinates.shape == (257, _STATE_SIZE)
    assert result.accepted_state_mask.shape == (257,)
    assert result.terminal_state_exact
    assert result.native_quality_observed
    assert not result.promotion_eligible
    control = prepare_neq_gntr_retraction_canary(
        problem,
        scaling.bootstrap_anchor,
        initial_physical_state,
        policy,
        options=NEQ_GNTR1_OPTIONS,
    )
    control_timed = control.run_solver_loop()
    control_result = control.finalize_result(control_timed)
    assert control.options is NEQ_GNTR1_OPTIONS
    assert control.identity.route != ROUTE
    assert control.identity.schema_version != SCHEMA_VERSION
    assert control.identity.optimizer_options_sha256 != (
        prepared.identity.optimizer_options_sha256
    )
    assert control.identity.identity_sha256 != prepared.identity.identity_sha256
    assert control_timed.history.nonlinear_corrections[0] == 1
    assert not control_result.promotion_eligible
    changed_problem = prepare_neq_gntr_retraction_canary(
        cast("FullSpaceProblem", ("fullspace-problem", 2)),
        scaling.bootstrap_anchor,
        initial_physical_state,
        policy,
        options=options,
    )
    assert changed_problem.identity.problem_sha256 != prepared.identity.problem_sha256
    assert changed_problem.identity.identity_sha256 != prepared.identity.identity_sha256


def test_separate_accepted_quality_diagnostic_is_compiled_and_callback_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        variable_scale=jnp.ones((_STATE_SIZE,), dtype=jnp.float64),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    policy = _policy(scale=scaling.constraint_inverse_scale)
    monkeypatch.setattr(
        neq_module,
        "cfs_sqp1_joint_value_constraints",
        lambda coordinates, _problem, _scaling: (
            jnp.vdot(coordinates, coordinates),
            jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64),
        ),
    )
    loop = _loop_result()
    prepared = prepare_neq_accepted_quality_diagnostics(
        cast("FullSpaceProblem", object()),
        scaling,
        policy,
        loop.accepted_optimizer_coordinates,
        loop.accepted_state_mask,
    )

    assert isinstance(prepared, PreparedNeqAcceptedQualityDiagnostics)
    no_hit = prepared.run(
        loop.accepted_optimizer_coordinates,
        loop.accepted_state_mask,
    )
    first_hit_ledger = loop.accepted_optimizer_coordinates.at[1].set(
        jnp.zeros((_STATE_SIZE,), dtype=jnp.float64)
    )
    first_hit = prepared.run(first_hit_ledger, loop.accepted_state_mask)

    assert no_hit.objectives.shape == (257,)
    assert no_hit.raw_equalities.shape == (257, _EQUALITY_SIZE)
    assert no_hit.scaled_equalities.shape == (257, _EQUALITY_SIZE)
    assert no_hit.accepted_state_mask.shape == (257,)
    assert not no_hit.quality_candidate_reached
    assert no_hit.first_quality_accepted_step == 0
    assert first_hit.quality_candidate_reached
    assert first_hit.first_quality_accepted_step == 1
    assert first_hit.quality_satisfied[1]
    assert not first_hit.quality_satisfied[2]
    compiled_text = (
        prepared._run_quality.runtime_executable().hlo_modules()[0].to_string()
    )
    assert "host_callback" not in compiled_text
    assert "io_callback" not in compiled_text


def test_signed_quality_margins_cover_exact_and_ulp_objective_boundaries() -> None:
    policy = _policy()
    baseline = _endpoint_evidence()

    def evidence_at(objective: float) -> NativeEquivalentEndpointEvidence:
        return baseline._replace(
            evaluation=replace(
                baseline.evaluation,
                weighted_total=jnp.asarray(objective, dtype=jnp.float64),
            )
        )

    exact = native_equivalent_quality_margins(
        evidence_at(policy.objective_target), policy
    )
    below = native_equivalent_quality_margins(
        evidence_at(np.nextafter(policy.objective_target, 0.0)), policy
    )
    above = native_equivalent_quality_margins(
        evidence_at(np.nextafter(policy.objective_target, np.inf)), policy
    )

    assert exact.objective_margin == 0.0
    assert below.objective_margin > 0.0
    assert above.objective_margin < 0.0
    assert exact.objective_usage_ratio == 1.0


def test_component_margins_use_lowest_index_for_equal_worst_values() -> None:
    policy = _policy()
    endpoint = _endpoint_evidence()
    violation = np.nextafter(policy.component_absolute_tolerance, np.inf)
    raw = (
        jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
        .at[3]
        .set(violation)
        .at[8]
        .set(violation)
    )
    endpoint = endpoint._replace(
        raw_equalities=raw,
        scaled_equalities=raw * policy.constraint_inverse_scale,
    )

    margins = native_equivalent_quality_margins(endpoint, policy)

    assert margins.minimum_component_index == 3
    assert margins.minimum_component_margin < 0.0
    assert margins.component_usage_ratio > 1.0


def test_signed_quality_margins_fail_closed_on_nonfinite_endpoint() -> None:
    policy = _policy()
    endpoint = _endpoint_evidence()._replace(
        raw_equalities=jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)
        .at[7]
        .set(jnp.nan)
    )

    margins = native_equivalent_quality_margins(endpoint, policy)

    assert not margins.all_finite


def test_terminal_diagnostic_is_unconditional_on_no_hit_and_kkt_is_non_gating() -> None:
    policy = _policy()
    endpoint = _endpoint_evidence()
    base_result = _base_result(endpoint=endpoint, quality_latched=False)
    nonfinite_stationarity = (
        jnp.zeros((_STATE_SIZE,), dtype=jnp.float64).at[19].set(jnp.nan)
    )

    diagnostic = build_native_equivalent_terminal_diagnostic(
        base_result,
        _raw_endpoint(endpoint, raw_stationarity=nonfinite_stationarity),
        policy,
    )

    assert not diagnostic.base_result.loop_result.device_quality_candidate_reached
    assert diagnostic.raw_kkt_status == int(KktTelemetryStatus.AVAILABLE_NONFINITE)
    assert diagnostic.terminal_state_bound
    assert diagnostic.terminal_quality_satisfied


def test_terminal_diagnostic_rejects_state_replacement() -> None:
    policy = _policy()
    endpoint = _endpoint_evidence()
    raw_endpoint = _raw_endpoint(endpoint)
    replaced_endpoint = replace(
        raw_endpoint,
        physical_state=raw_endpoint.physical_state.at[0].set(1.0),
    )

    diagnostic = build_native_equivalent_terminal_diagnostic(
        _base_result(endpoint=endpoint),
        replaced_endpoint,
        policy,
    )

    assert not diagnostic.terminal_state_bound
    assert not diagnostic.terminal_quality_satisfied


def test_terminal_diagnostic_rejects_wrong_scaled_multipliers() -> None:
    policy = _policy()
    base_result = _base_result()
    raw_endpoint = replace(
        _raw_endpoint(base_result.endpoint),
        scaled_multipliers=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )

    diagnostic = build_native_equivalent_terminal_diagnostic(
        base_result, raw_endpoint, policy
    )

    assert not diagnostic.terminal_state_bound


def test_terminal_diagnostic_rejects_wrong_physical_objective() -> None:
    policy = _policy()
    base_result = _base_result()
    raw_endpoint = replace(
        _raw_endpoint(base_result.endpoint),
        physical_objective=jnp.asarray(1.0, dtype=jnp.float64),
    )

    diagnostic = build_native_equivalent_terminal_diagnostic(
        base_result, raw_endpoint, policy
    )

    assert not diagnostic.terminal_state_bound


def test_terminal_diagnostic_rejects_wrong_last_optimizer_ledger_row() -> None:
    policy = _policy()
    base_result = _base_result()
    loop = base_result.loop_result
    corrupted_loop = loop._replace(
        accepted_optimizer_coordinates=(
            loop.accepted_optimizer_coordinates.at[loop.accepted_steps, 0].add(1.0)
        )
    )
    corrupted = base_result._replace(loop_result=corrupted_loop)

    diagnostic = build_native_equivalent_terminal_diagnostic(
        corrupted, _raw_endpoint(corrupted.endpoint), policy
    )

    assert not diagnostic.terminal_state_bound


def test_terminal_diagnostic_rejects_wrong_loop_terminal_coordinates() -> None:
    policy = _policy()
    base_result = _base_result()
    corrupted = base_result._replace(
        loop_result=base_result.loop_result._replace(
            optimizer_coordinates=(
                base_result.loop_result.optimizer_coordinates.at[0].add(1.0)
            )
        )
    )

    diagnostic = build_native_equivalent_terminal_diagnostic(
        corrupted, _raw_endpoint(corrupted.endpoint), policy
    )

    assert not diagnostic.terminal_state_bound


def test_terminal_diagnostic_rejects_false_latched_state_claim() -> None:
    policy = _policy()
    base_result = _base_result()._replace(latched_state_exact=jnp.asarray(False))

    diagnostic = build_native_equivalent_terminal_diagnostic(
        base_result, _raw_endpoint(base_result.endpoint), policy
    )

    assert not diagnostic.terminal_state_bound


def test_terminal_diagnostic_rejects_wrong_last_accepted_mask() -> None:
    policy = _policy()
    base_result = _base_result()
    corrupted = base_result._replace(
        accepted_state_mask=base_result.accepted_state_mask.at[
            base_result.loop_result.accepted_steps
        ].set(False)
    )

    diagnostic = build_native_equivalent_terminal_diagnostic(
        corrupted, _raw_endpoint(corrupted.endpoint), policy
    )

    assert not diagnostic.terminal_state_bound


def test_compiled_terminal_endpoint_surface_dispatches_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint_evidence()
    expected = _raw_endpoint(endpoint)
    trace_calls = 0

    def endpoint_diagnostics(
        _coordinates: jax.Array,
        _multipliers: jax.Array,
        _problem: FullSpaceProblem,
        _scaling: FullSpaceScaling,
    ) -> CfsSqp1EndpointDiagnostics:
        nonlocal trace_calls
        trace_calls += 1
        return expected

    monkeypatch.setattr(
        neq_module,
        "cfs_sqp1_endpoint_diagnostics",
        endpoint_diagnostics,
    )
    monkeypatch.setattr(
        neq_module,
        "fullspace_objective_residual_vector",
        lambda state, _problem: jnp.pad(state, (0, 2110 - state.size)),
    )
    monkeypatch.setattr(
        neq_module,
        "fullspace_value_and_grad",
        lambda state, _problem: (0.5 * jnp.vdot(state, state), state),
    )
    scaling = FullSpaceScaling(
        bootstrap_anchor=jnp.zeros((_STATE_SIZE,), dtype=jnp.float64),
        variable_scale=jnp.ones((_STATE_SIZE,), dtype=jnp.float64),
        constraint_inverse_scale=jnp.ones((_EQUALITY_SIZE,), dtype=jnp.float64),
    )
    coordinates = jnp.zeros((_STATE_SIZE,), dtype=jnp.float64)
    multipliers = jnp.zeros((_EQUALITY_SIZE,), dtype=jnp.float64)

    prepared = prepare_neq_terminal_endpoint_diagnostics(
        cast("FullSpaceProblem", object()),
        scaling,
        coordinates,
        multipliers,
    )
    assert trace_calls == 1
    result = prepared.run(coordinates, multipliers)
    evidence = prepared.run_evidence(coordinates, multipliers)

    assert trace_calls == 1
    np.testing.assert_array_equal(result.physical_state, expected.physical_state)
    assert evidence.objective_residual_vector.shape == (2110,)
    assert evidence.reconstructed_objective_gradient.shape == (_STATE_SIZE,)
    assert evidence.authoritative_objective_gradient.shape == (_STATE_SIZE,)
    assert evidence.all_finite
