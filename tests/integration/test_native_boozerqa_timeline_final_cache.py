"""Regression tests for timeline-only optimizer final-value reuse."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from examples.jax.parity.cases import native_boozerqa
from simsopt_jax.runtime.trace_annotations import (
    EvaluationKind,
    HostEvent,
    PhaseId,
    evaluation_context,
    trace_session,
)
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    TraceableObjectiveCandidateEvaluation,
    TraceableObjectiveInnerState,
    TraceableObjectiveSession,
)


def _parameter_sha256(parameters: np.ndarray) -> str:
    return native_boozerqa.hashlib.sha256(
        np.ascontiguousarray(parameters, dtype=np.dtype("<f8"))
        .reshape(-1)
        .tobytes(order="C")
    ).hexdigest()


def test_compute_graph_variant_is_selected_at_runtime_construction() -> None:
    with pytest.raises(ValueError, match="must be C0, C1, or C2"):
        native_boozerqa._prepare_jax_variant_runtime(
            None,  # type: ignore[arg-type]
            {},
            None,  # type: ignore[arg-type]
            None,
            exact_newton_variant="c1",  # type: ignore[arg-type]
        )


def test_timeline_optimizer_final_reuses_exact_accepted_value_and_gradient() -> None:
    parameters = np.asarray([1.25, -0.5], dtype=np.float64)
    gradient = np.asarray([0.75, -0.125], dtype=np.float64)
    accepted = native_boozerqa._AcceptedTimelineValueAndGradient(
        parameter_sha256=_parameter_sha256(parameters),
        value=3.5,
        gradient=gradient,
    )
    objective_calls = 0

    def evaluate(_parameters: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal objective_calls
        objective_calls += 1
        return -1.0, np.zeros_like(gradient)

    value, returned_gradient = native_boozerqa._optimizer_final_value_and_gradient(
        parameters,
        timeline_enabled=True,
        accepted_timeline_evaluation=accepted,
        parameter_sha256=_parameter_sha256,
        evaluate=evaluate,
    )

    assert objective_calls == 0
    assert value == accepted.value
    assert returned_gradient is gradient


def test_optimizer_final_without_timeline_preserves_evaluation() -> None:
    parameters = np.asarray([1.25, -0.5], dtype=np.float64)
    expected_gradient = np.asarray([0.25, 0.5], dtype=np.float64)
    objective_calls = 0

    def evaluate(actual_parameters: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal objective_calls
        objective_calls += 1
        np.testing.assert_array_equal(actual_parameters, parameters)
        return 2.0, expected_gradient

    value, gradient = native_boozerqa._optimizer_final_value_and_gradient(
        parameters,
        timeline_enabled=False,
        accepted_timeline_evaluation=None,
        parameter_sha256=_parameter_sha256,
        evaluate=evaluate,
    )

    assert objective_calls == 1
    assert value == 2.0
    assert gradient is expected_gradient


def test_timeline_has_one_final_reporting_evaluation_after_measurement_window() -> None:
    source = inspect.getsource(native_boozerqa._jax)

    assert source.count("EvaluationKind.FINAL_REPORTING") == 1
    window_end = source.index("    final_result = timeline_evaluate(")
    assert source.index("EvaluationKind.FINAL_REPORTING") > window_end


def test_deferred_timeline_evidence_materializes_complete_immutable_record() -> None:
    deferred = native_boozerqa._DeferredChangedStateTimelineObservation(
        trace_context=SimpleNamespace(
            evaluation_id="evaluation-000001",
            kind=SimpleNamespace(value="trial"),
            outer_iteration_id=2,
            parameter_sha256="a" * 64,
        ),
        parameters=(1.0, 2.0),
        parameter_shape=(2,),
        objective=4.0,
        gradient=(0.25, -0.5),
        gradient_shape=(2,),
        forward_success=np.asarray(True),
        primal_success=np.asarray(True),
        actual_adjoint_success=np.asarray(True),
        gradient_source="candidate",
        candidate_gradient_source=True,
        eligible=np.asarray(True),
        newton_iterations=np.asarray(2, dtype=np.int32),
        newton_attempted_iterations=np.asarray(2, dtype=np.int32),
        newton_trace_active_present=np.asarray(True),
        inner_residual_trace_present=np.asarray(True),
        newton_step_accepted_trace_present=np.asarray(True),
        newton_linear_solve_success_trace_present=np.asarray(True),
        newton_trace_active=np.asarray([True, True, False]),
        inner_residual_trace=np.asarray([1.0e-4, 1.0e-8, np.nan]),
        newton_step_accepted_trace=np.asarray([True, True, False]),
        newton_linear_solve_success_trace=np.asarray([True, True, False]),
        adjoint_output=np.asarray([0.5, -0.75]),
        adjoint_residual=np.asarray(1.0e-11),
        adjoint_residual_relative=np.asarray(1.0e-12),
        execution_counts=SimpleNamespace(
            dense_materialization_count=np.asarray(1, dtype=np.int32),
            lu_factorization_count=np.asarray(1, dtype=np.int32),
            lu_solve_count=np.asarray(12, dtype=np.int32),
            refinement_correction_count=np.asarray(1, dtype=np.int32),
            adjoint_execution_count=np.asarray(1, dtype=np.int32),
        ),
        observables=native_boozerqa._TIMELINE_UNAVAILABLE_PHYSICS_OBSERVABLES,
    )

    observation = native_boozerqa._materialize_changed_state_timeline_observation(
        deferred
    )

    assert observation.parameters == (1.0, 2.0)
    assert observation.parameter_shape == (2,)
    assert observation.inner_residual_trace == (1.0e-4, 1.0e-8)
    assert observation.newton_step_accepted_trace == (True, True)
    assert observation.newton_linear_solve_success_trace == (True, True)
    assert observation.newton_iterations == 2
    assert observation.newton_attempted_iterations == 2
    assert observation.newton_trace_available is True
    assert observation.adjoint_output == (0.5, -0.75)
    assert observation.adjoint_residual_relative == 1.0e-12
    assert observation.dense_materializations == 1
    assert observation.lu_factorizations == 1
    assert observation.lu_solves == 12
    assert observation.refinement_corrections == 1
    assert observation.adjoint_executions == 1
    assert observation.observables == (
        ("objective", None),
        ("iota", None),
        ("volume", None),
        ("non_qs_ratio", None),
        ("boozer_residual", None),
    )

    unavailable = replace(
        deferred,
        newton_iterations=np.asarray(3, dtype=np.int32),
        newton_attempted_iterations=np.asarray(np.iinfo(np.int32).min, dtype=np.int32),
        newton_trace_active_present=np.asarray(False),
        inner_residual_trace_present=np.asarray(False),
        newton_step_accepted_trace_present=np.asarray(False),
        newton_linear_solve_success_trace_present=np.asarray(False),
    )
    unavailable_observation = (
        native_boozerqa._materialize_changed_state_timeline_observation(unavailable)
    )

    assert unavailable_observation.newton_iterations == 3
    assert unavailable_observation.newton_attempted_iterations is None
    assert unavailable_observation.newton_trace_available is False
    assert unavailable_observation.inner_residual_trace == ()
    assert unavailable_observation.newton_step_accepted_trace == ()
    assert unavailable_observation.newton_linear_solve_success_trace == ()

    with pytest.raises(RuntimeError, match="presence is inconsistent"):
        native_boozerqa._materialize_changed_state_timeline_observation(
            replace(unavailable, newton_trace_active_present=np.asarray(True))
        )


def test_final_candidate_host_seam_emits_exact_lifecycle_and_transfer_spans(
    monkeypatch,
) -> None:
    parameters = np.asarray([1.25, -0.5], dtype=np.float64)
    evaluation = TraceableObjectiveCandidateEvaluation(
        forward_result={
            "value": np.asarray(3.5, dtype=np.float64),
            "raw_value": np.asarray(99.0, dtype=np.float64),
        },
        gradient=np.asarray([0.75, -0.125], dtype=np.float64),
        actual_adjoint_success=np.asarray(True),
        gradient_source="candidate",
        candidate_inner_state=TraceableObjectiveInnerState(
            coil_dofs=parameters,
            solved_x=np.asarray([0.25], dtype=np.float64),
            objective_value=np.asarray(99.0, dtype=np.float64),
            eligible=np.asarray(True),
        ),
    )
    assert not hasattr(evaluation, "value")
    session = object.__new__(TraceableObjectiveSession)
    device_calls = 0

    def evaluate_device(_session, candidate, incumbent_state):
        nonlocal device_calls
        device_calls += 1
        np.testing.assert_array_equal(np.asarray(candidate), parameters)
        assert incumbent_state == "anchor"
        return evaluation

    monkeypatch.setattr(
        TraceableObjectiveSession,
        "_evaluate_candidate_device",
        evaluate_device,
    )
    parameter_sha256 = _parameter_sha256(parameters)
    with trace_session() as audit, evaluation_context(
        "evaluation-final",
        parameter_sha256,
        EvaluationKind.FINAL_REPORTING,
    ):
        result, value, gradient = session._evaluate_candidate_from_anchor_host(
            parameters,
            "anchor",
        )

    assert result is evaluation
    assert value == 3.5
    np.testing.assert_array_equal(gradient, evaluation.gradient)
    assert device_calls == 1
    assert tuple(record.event for record in audit.events()) == tuple(HostEvent)
    assert tuple(record.phase for record in audit.records()) == (
        PhaseId.HOST_H2D_SUBMIT,
        PhaseId.HOST_D2H_MATERIALIZE,
    )
    for record in (*audit.events(), *audit.records()):
        attributes = dict(record.attributes)
        assert attributes["evaluation_id"] == "evaluation-final"
        assert attributes["parameter_sha256"] == parameter_sha256
        assert attributes["evaluation_kind"] == "final_reporting"
        assert "outer_iteration_id" not in attributes
