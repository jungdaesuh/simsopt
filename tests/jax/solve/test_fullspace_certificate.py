from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
import simsopt_jax.solve.fullspace_certificate as certificate_module
from simsopt_jax.geo.optimizers.dense_sqp import (
    DenseSQPConvergenceTelemetry,
    DenseSQPHistory,
    DenseSQPResult,
    DenseSQPStatus,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceConstraints,
    FullSpaceEvaluation,
    FullSpaceKKTPrimitives,
    FullSpaceObservables,
    FullSpaceRawTerms,
)
from simsopt_jax.solve.fullspace import (
    CfsAl1Result,
    CfsAl1StageHistory,
    FullSpaceRoute,
    FullSpaceScaling,
)
from simsopt_jax.solve.fullspace_certificate import (
    CFS_SQP1_CERTIFICATE_SCHEMA_VERSION,
    BranchEvidence,
    CfsAl1EndpointCertificate,
    CfsSqp1EndpointCertificate,
    CrossEvaluatorEvidence,
    FieldLineEvidence,
    FixedStateEvidence,
    InactiveHardwareEvidence,
    ObjectiveReferenceEvidence,
    OptimizerTermination,
    ProjectionEvidence,
    certify_cfs_al1_endpoint,
    certify_cfs_al2_endpoint,
    certify_cfs_sqp1_endpoint,
    certify_fullspace_endpoint,
)
from simsopt_jax.solve.fullspace_sqp import (
    SCHEMA_VERSION as CFS_SQP1_RESULT_SCHEMA_VERSION,
)
from simsopt_jax.solve.fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    CfsSqp1Result,
)

jax.config.update("jax_enable_x64", True)


def _evaluation(state: jax.Array, _problem: object) -> FullSpaceEvaluation:
    objective = jnp.asarray(0.5e-8, dtype=jnp.float64) * (state[0] - 1.0) ** 2
    boozer = jnp.reshape(state[0] - 1.0, (1,))
    volume = state[1] - 2.0
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return FullSpaceEvaluation(
        raw_terms=FullSpaceRawTerms(
            non_qs=objective,
            residual=zero,
            iota=zero,
            major_radius=zero,
            length=zero,
        ),
        weighted_total=objective,
        constraints=FullSpaceConstraints(boozer=boozer, volume=volume),
        observables=FullSpaceObservables(
            iota=state[0],
            G=state[1],
            volume=state[1],
            major_radius=jnp.asarray(1.5, dtype=jnp.float64),
            total_length=jnp.asarray(3.0, dtype=jnp.float64),
            non_qs_ratio=objective,
            boozer_residual_scalar=boozer[0] ** 2,
            boozer_residual_rms=jnp.abs(boozer[0]),
        ),
    )


def _kkt(
    state: jax.Array,
    multipliers: jax.Array,
    problem: object,
) -> FullSpaceKKTPrimitives:
    evaluation = _evaluation(state, problem)
    constraints = jnp.concatenate(
        (
            evaluation.constraints.boozer,
            jnp.reshape(evaluation.constraints.volume, (1,)),
        )
    )
    objective_gradient = jax.grad(
        lambda candidate: _evaluation(candidate, problem).weighted_total
    )(state)
    stationarity = objective_gradient + multipliers
    return FullSpaceKKTPrimitives(
        objective_value=evaluation.weighted_total,
        constraint_residual=constraints,
        stationarity_residual=stationarity,
        primal_feasibility_inf=jnp.max(jnp.abs(constraints)),
        stationarity_inf=jnp.max(jnp.abs(stationarity)),
        all_finite=(
            jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(multipliers))
            & jnp.all(jnp.isfinite(stationarity))
            & jnp.all(jnp.isfinite(constraints))
        ),
    )


@pytest.fixture(autouse=True)
def _install_pure_test_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(certificate_module, "evaluate_fullspace", _evaluation)
    monkeypatch.setattr(certificate_module, "fullspace_kkt_primitives", _kkt)


def _result(
    *,
    state: jax.Array | None = None,
    inner_iterations_per_stage: int = 1,
    function_evaluations_per_stage: int = 2,
) -> CfsAl1Result:
    endpoint = jnp.asarray([1.0, 2.0], dtype=jnp.float64) if state is None else state
    stages = 10
    history = CfsAl1StageHistory(
        penalty=jnp.ones(stages, dtype=jnp.float64),
        physical_objective=jnp.zeros(stages, dtype=jnp.float64),
        scaled_feasibility_infinity_norm=jnp.zeros(stages, dtype=jnp.float64),
        augmented_stationarity_infinity_norm=jnp.zeros(stages, dtype=jnp.float64),
        optimizer_status=jnp.ones(stages, dtype=jnp.int32),
        inner_iterations=jnp.full(stages, inner_iterations_per_stage, dtype=jnp.int32),
        function_evaluations=jnp.full(
            stages, function_evaluations_per_stage, dtype=jnp.int32
        ),
        stage_completed=jnp.ones(stages, dtype=jnp.bool_),
    )
    return CfsAl1Result(
        optimizer_coordinates=endpoint,
        physical_state=endpoint,
        scaled_multipliers=jnp.zeros(2, dtype=jnp.float64),
        raw_multipliers=jnp.zeros(2, dtype=jnp.float64),
        next_penalty=jnp.asarray(250.0, dtype=jnp.float64),
        physical_objective=jnp.asarray(0.0, dtype=jnp.float64),
        raw_constraint_infinity_norm=jnp.asarray(0.0, dtype=jnp.float64),
        scaled_constraint_infinity_norm=jnp.asarray(0.0, dtype=jnp.float64),
        raw_kkt_stationarity_infinity_norm=jnp.asarray(0.0, dtype=jnp.float64),
        completed_outer_stages=jnp.asarray(stages, dtype=jnp.int32),
        total_inner_iterations=jnp.asarray(
            inner_iterations_per_stage * stages, dtype=jnp.int32
        ),
        total_function_evaluations=jnp.asarray(
            function_evaluations_per_stage * stages, dtype=jnp.int32
        ),
        total_gradient_evaluations=jnp.asarray(
            function_evaluations_per_stage * stages, dtype=jnp.int32
        ),
        nonfinite_evaluation_count=jnp.asarray(0, dtype=jnp.int32),
        all_accepted_states_finite=jnp.asarray(True),
        converged=jnp.asarray(True),
        fatal=jnp.asarray(False),
        all_finite=jnp.asarray(True),
        stage_history=history,
    )


def _inputs() -> dict[str, object]:
    endpoint = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    return {
        "result": _result(),
        "problem": SimpleNamespace(
            layout=SimpleNamespace(total_dof_count=2),
            config=SimpleNamespace(
                non_qs_weight=jnp.asarray(1.0, dtype=jnp.float64),
                residual_weight=jnp.asarray(1.0, dtype=jnp.float64),
                iota_weight=jnp.asarray(1.0, dtype=jnp.float64),
                major_radius_weight=jnp.asarray(1.0, dtype=jnp.float64),
                length_weight=jnp.asarray(1.0, dtype=jnp.float64),
            ),
        ),
        "scaling": FullSpaceScaling(
            bootstrap_anchor=jnp.zeros(2, dtype=jnp.float64),
            variable_scale=jnp.ones(2, dtype=jnp.float64),
            constraint_inverse_scale=jnp.ones(2, dtype=jnp.float64),
        ),
        "fixed_state": FixedStateEvidence(
            expected_first_current=jnp.asarray(1.2e5, dtype=jnp.float64),
            observed_first_current=jnp.asarray(1.2e5, dtype=jnp.float64),
            expected_fixed_dofs=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
            observed_fixed_dofs=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
        ),
        "inactive_hardware": InactiveHardwareEvidence(
            names=certificate_module.INACTIVE_HARDWARE_TERMS,
            metrics=jnp.asarray([20.0, 0.1, 0.2, 0.3], dtype=jnp.float64),
            weights=jnp.zeros(4, dtype=jnp.float64),
        ),
        "objective_reference": ObjectiveReferenceEvidence(
            native_reference_objective=jnp.asarray(
                certificate_module.OBJECTIVE_MAXIMUM, dtype=jnp.float64
            )
        ),
        "cross_evaluator": CrossEvaluatorEvidence(
            performed=True,
            native_on_jax_endpoint_objective=jnp.asarray(0.0, dtype=jnp.float64),
            jax_on_native_endpoint_objective=jnp.asarray(
                certificate_module.OBJECTIVE_MAXIMUM, dtype=jnp.float64
            ),
        ),
        "field_line": FieldLineEvidence(
            performed=True,
            poincare_closed=True,
            traced_iota=jnp.asarray(1.0, dtype=jnp.float64),
        ),
        "branch": BranchEvidence(
            performed=True,
            exact_solve_succeeded=True,
            material_branch_switch=False,
            reproduced_state_infinity_difference=jnp.asarray(
                1.0e-12, dtype=jnp.float64
            ),
            basin_classification="continuation-connected-native-basin",
        ),
        "projection": ProjectionEvidence(
            evaluated=True,
            used=False,
            pre_state=endpoint,
            post_state=endpoint,
        ),
    }


def _sqp_result(
    *,
    status: DenseSQPStatus = DenseSQPStatus.CONVERGED,
    iterations: int = 2,
    joint_evaluations: int = 5,
    kkt_solves: int | None = None,
) -> CfsSqp1Result:
    endpoint = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    scaled_multipliers = jnp.zeros(2, dtype=jnp.float64)
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    history_size = 100
    accepted = jnp.arange(history_size) < iterations
    history = DenseSQPHistory(
        objective=jnp.where(accepted, zero, jnp.nan),
        feasibility_infinity_norm=jnp.where(accepted, zero, jnp.nan),
        stationarity_infinity_norm=jnp.where(accepted, zero, jnp.nan),
        step_length=jnp.where(accepted, jnp.asarray(1.0), jnp.nan),
        kkt_relative_residual=jnp.where(accepted, zero, jnp.nan),
        status=jnp.where(
            accepted,
            jnp.asarray(int(status), dtype=jnp.int32),
            jnp.asarray(int(DenseSQPStatus.RUNNING), dtype=jnp.int32),
        ),
    )
    convergence_telemetry = DenseSQPConvergenceTelemetry(
        merit=jnp.where(accepted, zero, jnp.nan),
        penalty=jnp.where(accepted, jnp.asarray(1.0), jnp.nan),
        multiplier_update_infinity_norm=jnp.where(accepted, zero, jnp.nan),
        bfgs_reset=jnp.zeros(history_size, dtype=jnp.int32),
        restoration_applied=jnp.zeros(history_size, dtype=jnp.int32),
    )
    converged = status is DenseSQPStatus.CONVERGED
    solve_count = iterations if kkt_solves is None else kkt_solves
    unavailable = jnp.asarray(jnp.nan, dtype=jnp.float64)
    final_residual = zero if solve_count > 0 else unavailable
    final_condition = (
        jnp.asarray(1.0, dtype=jnp.float64) if solve_count > 0 else unavailable
    )
    final_pivot = (
        jnp.asarray(1.0, dtype=jnp.float64) if solve_count > 0 else unavailable
    )
    final_regularization = zero if solve_count > 0 else unavailable
    optimizer = DenseSQPResult(
        optimizer_coordinates=endpoint,
        multipliers=scaled_multipliers,
        bfgs_matrix=jnp.eye(2, dtype=jnp.float64),
        objective=zero,
        constraints=jnp.zeros(2, dtype=jnp.float64),
        objective_gradient=jnp.zeros(2, dtype=jnp.float64),
        constraint_jacobian=jnp.eye(2, dtype=jnp.float64),
        stationarity=jnp.zeros(2, dtype=jnp.float64),
        converged=jnp.asarray(converged),
        fatal=jnp.asarray(False),
        failed=jnp.asarray(not converged),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        joint_evaluations=jnp.asarray(joint_evaluations, dtype=jnp.int32),
        derivative_builds=jnp.asarray(iterations + 1, dtype=jnp.int32),
        kkt_solves=jnp.asarray(solve_count, dtype=jnp.int32),
        line_search_evaluations=jnp.asarray(iterations, dtype=jnp.int32),
        rejected_nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
        bfgs_resets=jnp.asarray(0, dtype=jnp.int32),
        regularization_uses=jnp.asarray(0, dtype=jnp.int32),
        final_kkt_relative_residual=final_residual,
        final_kkt_reciprocal_condition=final_condition,
        final_kkt_solution_scaled_residual=final_residual,
        final_schur_relative_residual=final_residual,
        final_bfgs_cholesky_relative_pivot=final_pivot,
        final_schur_cholesky_relative_pivot=final_pivot,
        selected_regularization=final_regularization,
        regularization_candidates_tested=jnp.asarray(solve_count, dtype=jnp.int32),
        merit_penalty=jnp.asarray(1.0, dtype=jnp.float64),
        restoration_numerical_failures=jnp.asarray(0, dtype=jnp.int32),
        all_accepted_states_finite=jnp.asarray(True),
        all_finite=jnp.asarray(True),
        history=history,
        convergence_telemetry=convergence_telemetry,
    )
    diagnostics = CfsSqp1EndpointDiagnostics(
        physical_state=endpoint,
        physical_objective=zero,
        raw_constraints=jnp.zeros(2, dtype=jnp.float64),
        scaled_constraints=jnp.zeros(2, dtype=jnp.float64),
        scaled_multipliers=scaled_multipliers,
        raw_multipliers=scaled_multipliers,
        raw_stationarity_residual=jnp.zeros(2, dtype=jnp.float64),
        raw_constraint_infinity_norm=zero,
        scaled_constraint_infinity_norm=zero,
        raw_kkt_stationarity_infinity_norm=zero,
        all_finite=jnp.asarray(True),
    )
    return CfsSqp1Result(
        schema_version=CFS_SQP1_RESULT_SCHEMA_VERSION,
        route=FullSpaceRoute.CFS_SQP1,
        optimizer=optimizer,
        endpoint=diagnostics,
        optimizer_stationarity_tolerance=jnp.asarray(1.0e-7, dtype=jnp.float64),
        stationarity_scaling_error_infinity_norm=zero,
        solver_result_consistent=jnp.asarray(True),
        all_finite=jnp.asarray(True),
        converged=jnp.asarray(converged),
    )


def _certify(inputs: dict[str, object]):
    return certify_cfs_al1_endpoint(**inputs)  # type: ignore[arg-type]


def test_complete_cfs_al1_evidence_certifies_without_nested_inner_success() -> None:
    certificate = _certify(_inputs())

    assert type(certificate) is CfsAl1EndpointCertificate
    assert certificate.schema_version == certificate_module.SCHEMA_VERSION
    assert certificate.certified
    assert certificate.termination is OptimizerTermination.CONVERGED
    assert certificate.pre_projection.boozer_residual_infinity_norm == 0.0
    assert certificate.pre_projection.volume_residual_absolute == 0.0
    assert certificate.post_projection.iota == 1.0
    assert certificate.post_projection.major_radius == 1.5
    assert certificate.post_projection.one_sided_length_penalty == 0.0
    assert not hasattr(certificate, "inner_success")


@pytest.mark.parametrize(
    ("field", "replacement", "failed_check"),
    (
        (
            "fixed_state",
            lambda value: replace(
                value,
                observed_first_current=jnp.asarray(1.2e5 + 1.0, dtype=jnp.float64),
            ),
            "fixed_state_preserved",
        ),
        (
            "inactive_hardware",
            lambda value: replace(
                value,
                weights=jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.float64),
            ),
            "inactive_hardware_terms_valid",
        ),
        (
            "cross_evaluator",
            lambda value: replace(value, performed=False),
            "cross_evaluator",
        ),
        (
            "field_line",
            lambda value: replace(value, poincare_closed=False),
            "field_line",
        ),
        (
            "branch",
            lambda value: replace(value, material_branch_switch=True),
            "branch",
        ),
    ),
)
def test_required_external_evidence_tampering_fails_closed(
    field: str,
    replacement: object,
    failed_check: str,
) -> None:
    inputs = _inputs()
    inputs[field] = replacement(inputs[field])  # type: ignore[operator]

    certificate = _certify(inputs)

    assert not certificate.certified
    assert not getattr(certificate.checks, failed_check)


def test_material_projection_cannot_repair_failed_unprojected_solve() -> None:
    inputs = _inputs()
    pre_state = jnp.asarray([1.0 + 1.0e-4, 2.0], dtype=jnp.float64)
    inputs["result"] = _result(state=pre_state)
    inputs["projection"] = ProjectionEvidence(
        evaluated=True,
        used=True,
        pre_state=pre_state,
        post_state=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
    )

    certificate = _certify(inputs)

    assert not certificate.certified
    assert not certificate.checks.pre_projection_certifiable
    assert not certificate.checks.projection_immaterial
    assert certificate.checks.post_projection_certifiable


def test_optimizer_nonfinite_and_incomplete_histories_are_normalized() -> None:
    inputs = _inputs()
    result = inputs["result"]
    assert isinstance(result, CfsAl1Result)
    inputs["result"] = replace(
        result,
        nonfinite_evaluation_count=jnp.asarray(1, dtype=jnp.int32),
    )
    nonfinite = _certify(inputs)
    assert not nonfinite.certified
    assert nonfinite.termination is OptimizerTermination.NONFINITE

    inputs = _inputs()
    result = inputs["result"]
    assert isinstance(result, CfsAl1Result)
    inputs["result"] = replace(
        result,
        completed_outer_stages=jnp.asarray(9, dtype=jnp.int32),
    )
    incomplete = _certify(inputs)
    assert not incomplete.certified
    assert incomplete.termination is OptimizerTermination.INCOMPLETE

    inputs = _inputs()
    result = inputs["result"]
    assert isinstance(result, CfsAl1Result)
    tampered_history = replace(
        result.stage_history,
        inner_iterations=result.stage_history.inner_iterations.at[0].set(101),
    )
    inputs["result"] = replace(
        result,
        stage_history=tampered_history,
        total_inner_iterations=jnp.asarray(110, dtype=jnp.int32),
    )
    over_budget = _certify(inputs)
    assert not over_budget.certified
    assert over_budget.termination is OptimizerTermination.INCOMPLETE


def test_solver_reported_endpoint_metrics_are_recomputed_not_trusted() -> None:
    inputs = _inputs()
    result = inputs["result"]
    assert isinstance(result, CfsAl1Result)
    inputs["result"] = replace(
        result,
        physical_objective=jnp.asarray(-1.0, dtype=jnp.float64),
    )

    certificate = _certify(inputs)

    assert not certificate.certified
    assert not certificate.checks.solver_result_consistent


def test_fp32_or_nonfinite_authoritative_evidence_fails_closed() -> None:
    inputs = _inputs()
    inputs["objective_reference"] = ObjectiveReferenceEvidence(
        native_reference_objective=jnp.asarray(
            certificate_module.OBJECTIVE_MAXIMUM, dtype=jnp.float32
        )
    )
    wrong_dtype = _certify(inputs)
    assert not wrong_dtype.certified
    assert not wrong_dtype.checks.objective_reference_valid

    inputs = _inputs()
    inputs["branch"] = replace(
        inputs["branch"],  # type: ignore[arg-type]
        reproduced_state_infinity_difference=jnp.asarray(jnp.nan, dtype=jnp.float64),
    )
    nonfinite = _certify(inputs)
    assert not nonfinite.certified
    assert not nonfinite.checks.branch


def test_projection_pre_state_must_be_exact_solver_endpoint() -> None:
    inputs = _inputs()
    projection = inputs["projection"]
    assert isinstance(projection, ProjectionEvidence)
    inputs["projection"] = replace(
        projection,
        pre_state=projection.pre_state.at[0].add(1.0e-13),
        post_state=projection.post_state.at[0].add(1.0e-13),
    )

    certificate = _certify(inputs)

    assert not certificate.certified
    assert not certificate.checks.projection_bound_to_solver_endpoint


@pytest.mark.parametrize(
    "route",
    (FullSpaceRoute.CFS_P0, FullSpaceRoute.CFS_AL1_B),
)
def test_other_routes_are_rejected_before_scientific_evaluation(
    route: FullSpaceRoute,
) -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="restricted to CFS-AL1 and CFS-AL2"):
        certify_fullspace_endpoint(
            route=route,
            **inputs,  # type: ignore[arg-type]
        )


def test_cfs_al2_uses_its_1000_per_stage_and_10000_total_budget() -> None:
    inputs = _inputs()
    inputs["result"] = _result(
        inner_iterations_per_stage=1000,
        function_evaluations_per_stage=1001,
    )

    certificate = certify_cfs_al2_endpoint(**inputs)  # type: ignore[arg-type]

    assert certificate.certified
    assert certificate.route is FullSpaceRoute.CFS_AL2
    assert certificate.termination is OptimizerTermination.CONVERGED


def test_identical_stage_history_is_route_validated_not_globally_accepted() -> None:
    inputs = _inputs()
    inputs["result"] = _result(
        inner_iterations_per_stage=1000,
        function_evaluations_per_stage=1001,
    )

    al1 = _certify(inputs)

    assert not al1.certified
    assert al1.termination is OptimizerTermination.INCOMPLETE


@pytest.mark.parametrize(
    ("inner_iterations", "function_evaluations"),
    (
        (1001, 1002),
        (1000, 999),
        (1000, 15001),
    ),
)
def test_cfs_al2_rejects_per_stage_iteration_and_maxfun_tampering(
    inner_iterations: int,
    function_evaluations: int,
) -> None:
    inputs = _inputs()
    inputs["result"] = _result(
        inner_iterations_per_stage=inner_iterations,
        function_evaluations_per_stage=function_evaluations,
    )

    certificate = certify_cfs_al2_endpoint(**inputs)  # type: ignore[arg-type]

    assert not certificate.certified
    assert certificate.termination is OptimizerTermination.INCOMPLETE


def test_complete_cfs_sqp1_evidence_uses_shared_scientific_recomputation() -> None:
    inputs = _inputs()
    inputs["result"] = _sqp_result()

    certificate = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]

    assert certificate.certified
    assert type(certificate) is CfsSqp1EndpointCertificate
    assert certificate.schema_version == CFS_SQP1_CERTIFICATE_SCHEMA_VERSION
    assert certificate.route is FullSpaceRoute.CFS_SQP1
    assert certificate.termination is OptimizerTermination.CONVERGED
    assert certificate.checks.scaled_feasibility
    assert certificate.checks.raw_kkt_stationarity
    assert certificate.checks.objective_threshold
    assert certificate.checks.cross_evaluator
    assert certificate.checks.field_line
    assert certificate.checks.branch
    assert certificate.checks.objective_ledger_consistent
    assert certificate.pre_projection.raw_objective_terms.non_qs == 0.0
    assert certificate.pre_projection.raw_objective_terms.residual == 0.0
    assert certificate.pre_projection.raw_objective_terms.iota == 0.0
    assert certificate.pre_projection.raw_objective_terms.major_radius == 0.0
    assert certificate.pre_projection.raw_objective_terms.length == 0.0
    assert not hasattr(certificate, "stage_history")


@pytest.mark.parametrize(
    "field",
    ("non_qs", "residual", "iota", "major_radius", "length"),
)
def test_cfs_sqp1_raw_objective_term_tampering_fails_closed(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tampered_evaluation(state: jax.Array, problem: object) -> FullSpaceEvaluation:
        evaluation = _evaluation(state, problem)
        return replace(
            evaluation,
            raw_terms=replace(
                evaluation.raw_terms,
                **{field: getattr(evaluation.raw_terms, field) + 1.0e-9},
            ),
        )

    monkeypatch.setattr(certificate_module, "evaluate_fullspace", tampered_evaluation)
    inputs = _inputs()
    inputs["result"] = _sqp_result()

    certificate = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]

    assert not certificate.certified
    assert not certificate.checks.objective_ledger_consistent


def test_cfs_sqp1_initially_converged_endpoint_requires_unavailable_solve_diagnostics() -> (
    None
):
    inputs = _inputs()
    inputs["result"] = _sqp_result(
        iterations=0,
        joint_evaluations=1,
        kkt_solves=0,
    )

    certificate = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]

    assert certificate.certified
    assert certificate.termination is OptimizerTermination.CONVERGED


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("final_kkt_relative_residual", 1.1e-10),
        ("final_kkt_reciprocal_condition", 0.0),
        ("final_kkt_solution_scaled_residual", 1.1e-10),
        ("final_schur_relative_residual", 1.1e-10),
        ("selected_regularization", 2.0e-12),
        ("final_bfgs_cholesky_relative_pivot", jnp.nan),
        ("final_schur_cholesky_relative_pivot", jnp.nan),
    ),
)
def test_cfs_sqp1_kkt_diagnostic_tampering_fails_closed(
    field: str,
    value: float,
) -> None:
    inputs = _inputs()
    result = _sqp_result()
    inputs["result"] = replace(
        result,
        optimizer=result.optimizer._replace(
            **{field: jnp.asarray(value, dtype=jnp.float64)}
        ),
    )

    certificate = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]

    assert not certificate.certified
    assert not certificate.checks.solver_result_consistent


def test_cfs_sqp1_zero_solve_rejects_fabricated_available_diagnostics() -> None:
    inputs = _inputs()
    result = _sqp_result(iterations=0, joint_evaluations=1, kkt_solves=0)
    inputs["result"] = replace(
        result,
        optimizer=result.optimizer._replace(
            final_kkt_reciprocal_condition=jnp.asarray(1.0, dtype=jnp.float64)
        ),
    )

    certificate = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]

    assert not certificate.certified
    assert not certificate.checks.solver_result_consistent


@pytest.mark.parametrize(
    ("status", "iterations", "joint_evaluations", "termination"),
    (
        (DenseSQPStatus.RUNNING, 2, 5, OptimizerTermination.INCOMPLETE),
        (DenseSQPStatus.ITERATION_LIMIT, 100, 201, OptimizerTermination.INCOMPLETE),
        (DenseSQPStatus.EVALUATION_LIMIT, 10, 1200, OptimizerTermination.INCOMPLETE),
        (
            DenseSQPStatus.GLOBALIZATION_FAILED,
            2,
            5,
            OptimizerTermination.OPTIMIZER_REJECTED,
        ),
    ),
)
def test_cfs_sqp1_statuses_are_normalized_without_al_stage_fabrication(
    status: DenseSQPStatus,
    iterations: int,
    joint_evaluations: int,
    termination: OptimizerTermination,
) -> None:
    inputs = _inputs()
    inputs["result"] = _sqp_result(
        status=status,
        iterations=iterations,
        joint_evaluations=joint_evaluations,
    )

    certificate = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]

    assert not certificate.certified
    assert certificate.termination is termination
    assert not certificate.checks.optimizer_termination


def test_cfs_sqp1_rejects_budget_nonfinite_and_reported_endpoint_tampering() -> None:
    inputs = _inputs()
    inputs["result"] = _sqp_result(iterations=101, joint_evaluations=202)
    over_budget = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]
    assert not over_budget.certified
    assert over_budget.termination is OptimizerTermination.INCOMPLETE

    inputs = _inputs()
    result = _sqp_result()
    inputs["result"] = replace(
        result,
        optimizer=result.optimizer._replace(
            bfgs_matrix=result.optimizer.bfgs_matrix.at[0, 0].set(jnp.nan)
        ),
    )
    nonfinite = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]
    assert not nonfinite.certified
    assert nonfinite.termination is OptimizerTermination.NONFINITE

    inputs = _inputs()
    result = _sqp_result()
    inputs["result"] = replace(
        result,
        endpoint=replace(
            result.endpoint,
            physical_objective=jnp.asarray(-1.0, dtype=jnp.float64),
        ),
    )
    tampered = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]
    assert not tampered.certified
    assert not tampered.checks.solver_result_consistent


def test_cfs_sqp1_retains_all_external_audit_requirements() -> None:
    inputs = _inputs()
    inputs["result"] = _sqp_result()
    inputs["cross_evaluator"] = replace(
        inputs["cross_evaluator"],  # type: ignore[arg-type]
        performed=False,
    )

    certificate = certify_cfs_sqp1_endpoint(**inputs)  # type: ignore[arg-type]

    assert not certificate.certified
    assert not certificate.checks.cross_evaluator
