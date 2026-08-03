from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax_adapters.geo import surface_objectives_traceable


def _forward_result(*, success: bool, primal_success: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "value": jnp.asarray(91.0, dtype=jnp.float64),
        "raw_value": jnp.asarray(7.5, dtype=jnp.float64),
        "x": jnp.asarray([3.0, 4.0], dtype=jnp.float64),
        "linear_solve_factors": None,
        "success": jnp.asarray(success, dtype=jnp.bool_),
        "predictor_success": jnp.asarray(True, dtype=jnp.bool_),
        "primal_success": jnp.asarray(primal_success, dtype=jnp.bool_),
        "adjoint_linear_solve_available": jnp.asarray(True, dtype=jnp.bool_),
        "newton_success": jnp.asarray(False, dtype=jnp.bool_),
        "newton_iterations": jnp.asarray(6, dtype=jnp.int32),
        "newton_attempted_iterations": jnp.asarray(8, dtype=jnp.int32),
        "newton_stop_reason_code": jnp.asarray(2, dtype=jnp.int32),
        "newton_last_linear_solve_success": jnp.asarray(False, dtype=jnp.bool_),
        "inner_penalty_residual_l2": jnp.asarray(2.5e-8, dtype=jnp.float64),
        "final_gradient_inf_norm": jnp.asarray(3.5e-9, dtype=jnp.float64),
        "newton_linear_solve_backend_code": jnp.asarray(4, dtype=jnp.int32),
        "newton_linear_solve_backend_code_present": jnp.asarray(True, dtype=jnp.bool_),
        "dense_hessian_bytes": jnp.asarray(4096, dtype=jnp.int64),
        "dense_hessian_bytes_present": jnp.asarray(True, dtype=jnp.bool_),
        "max_dense_hessian_bytes": jnp.asarray(8192, dtype=jnp.int64),
        "max_dense_hessian_bytes_present": jnp.asarray(True, dtype=jnp.bool_),
    }
    for index, key in enumerate(
        surface_objectives_traceable._TRACEABLE_NEWTON_TRACE_KEYS
    ):
        if "dtype_bits" in key:
            value = jnp.asarray([64 + index, 65 + index], dtype=jnp.int32)
        elif key in {
            "newton_trace_active",
            "newton_trace_step_accepted",
            "newton_trace_linear_solve_success",
            "newton_trace_linear_live_operator_certificate",
        }:
            value = jnp.asarray([True, False], dtype=jnp.bool_)
        else:
            value = jnp.asarray([index + 0.25, index + 0.5], dtype=jnp.float64)
        result[key] = value
        result[f"{key}_present"] = jnp.asarray(True, dtype=jnp.bool_)
    return result


def _compiled_bundle(
    forward_result: dict[str, object],
    *,
    adjoint_success: bool = True,
) -> tuple[dict[str, object], list[tuple[np.ndarray, np.ndarray, object]], list[int]]:
    gradient_calls: list[tuple[np.ndarray, np.ndarray, object]] = []
    forward_calls: list[int] = []

    def compiled_forward_result_for(coil_dofs):
        forward_calls.append(1)
        return forward_result

    def compiled_total_gradient_for(coil_dofs, solved_x, factors):
        gradient_calls.append(
            (
                np.asarray(coil_dofs),
                np.asarray(solved_x),
                factors,
            )
        )
        return (
            jnp.asarray([1.25, -2.5], dtype=jnp.float64),
            jnp.asarray(adjoint_success, dtype=jnp.bool_),
        )

    bundle = {
        "state": {
            "baseline_coil_dofs": np.asarray([-1.0, -2.0], dtype=np.float64),
            "baseline_x": np.asarray([-3.0, -4.0], dtype=np.float64),
            "baseline_linear_solve_factors": None,
        },
        "compiled_forward_result_for": compiled_forward_result_for,
        "compiled_total_gradient_for": compiled_total_gradient_for,
    }
    return bundle, gradient_calls, forward_calls


def test_trial_evaluator_uses_one_forward_and_retains_candidate_telemetry() -> None:
    forward_result = _forward_result(success=True, primal_success=True)
    bundle, gradient_calls, forward_calls = _compiled_bundle(forward_result)
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )

    evaluator = session.trial_evaluator()
    trial = evaluator(np.asarray([8.0, 9.0], dtype=np.float64))

    assert evaluator is session.trial_evaluator()
    assert forward_calls == [1]
    assert len(gradient_calls) == 1
    np.testing.assert_array_equal(gradient_calls[0][0], np.asarray([8.0, 9.0]))
    np.testing.assert_array_equal(gradient_calls[0][1], np.asarray([3.0, 4.0]))
    assert gradient_calls[0][2] is None
    assert trial.raw_objective_value == 7.5
    assert trial.filtered_objective_value == 91.0
    np.testing.assert_array_equal(trial.gradient, np.asarray([1.25, -2.5]))
    assert trial.gradient_source == "candidate"
    assert trial.gradient_is_finite
    assert trial.gradient_inf_norm == 2.5
    assert trial.predictor_success
    assert trial.primal_success
    assert trial.actual_adjoint_success
    assert not trial.newton_success
    assert trial.newton_iterations == 6
    assert trial.newton_attempted_iterations == 8
    assert trial.newton_stop_reason_code == 2
    assert not trial.newton_last_linear_solve_success
    assert trial.inner_penalty_residual_l2 == 2.5e-8
    assert trial.final_gradient_inf_norm == 3.5e-9
    np.testing.assert_array_equal(
        trial.newton_trace_active,
        forward_result["newton_trace_active"],
    )
    np.testing.assert_array_equal(
        trial.newton_trace_certificate_gradient_dtype_bits,
        forward_result["newton_trace_certificate_gradient_dtype_bits"],
    )
    assert trial.newton_trace_active_present
    assert trial.newton_trace_certificate_gradient_dtype_bits_present
    assert trial.newton_linear_solve_backend_code == 4
    assert trial.newton_linear_solve_backend_code_present
    assert trial.dense_hessian_bytes == 4096
    assert trial.dense_hessian_bytes_present
    assert trial.max_dense_hessian_bytes == 8192
    assert trial.max_dense_hessian_bytes_present
    with pytest.raises(AttributeError):
        trial.gradient_source = "baseline"


def test_trial_evaluator_uses_baseline_gradient_only_when_primal_failed() -> None:
    forward_result = _forward_result(success=False, primal_success=False)
    forward_result["predictor_success"] = jnp.asarray(False, dtype=jnp.bool_)
    bundle, gradient_calls, forward_calls = _compiled_bundle(forward_result)
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )

    trial = session.trial_evaluator()(np.asarray([8.0, 9.0], dtype=np.float64))

    assert forward_calls == [1]
    assert len(gradient_calls) == 1
    np.testing.assert_array_equal(gradient_calls[0][0], np.asarray([-1.0, -2.0]))
    np.testing.assert_array_equal(gradient_calls[0][1], np.asarray([-3.0, -4.0]))
    assert trial.gradient_source == "baseline"
    assert not trial.predictor_success
    assert not trial.primal_success


def test_trial_evaluator_keeps_candidate_gradient_for_filtered_primal() -> None:
    forward_result = _forward_result(success=False, primal_success=True)
    bundle, gradient_calls, _forward_calls = _compiled_bundle(forward_result)
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )

    trial = session.trial_evaluator()(np.asarray([8.0, 9.0], dtype=np.float64))

    np.testing.assert_array_equal(gradient_calls[0][0], np.asarray([8.0, 9.0]))
    np.testing.assert_array_equal(gradient_calls[0][1], np.asarray([3.0, 4.0]))
    assert trial.gradient_source == "candidate"
    assert not bool(forward_result["success"])
    assert trial.primal_success


def test_trial_evaluator_retains_actual_adjoint_failure() -> None:
    forward_result = _forward_result(success=True, primal_success=True)
    bundle, _gradient_calls, _forward_calls = _compiled_bundle(
        forward_result,
        adjoint_success=False,
    )
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )

    trial = session.trial_evaluator()(np.asarray([8.0, 9.0], dtype=np.float64))

    assert not trial.actual_adjoint_success
    assert not trial.gradient_is_finite
    assert np.isnan(trial.gradient).all()
    assert np.isnan(trial.gradient_inf_norm)


def test_trial_evaluator_returns_immutable_candidate_inner_state() -> None:
    forward_result = _forward_result(success=True, primal_success=True)
    bundle, _gradient_calls, _forward_calls = _compiled_bundle(forward_result)
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )
    candidate_coil_dofs = np.asarray([8.0, 9.0], dtype=np.float64)

    trial = session.trial_evaluator()(candidate_coil_dofs)

    candidate_inner_state = trial.candidate_inner_state
    assert isinstance(
        candidate_inner_state,
        surface_objectives_traceable.TraceableObjectiveInnerState,
    )
    np.testing.assert_array_equal(
        candidate_inner_state.coil_dofs,
        candidate_coil_dofs,
    )
    np.testing.assert_array_equal(
        candidate_inner_state.solved_x,
        forward_result["x"],
    )
    assert float(candidate_inner_state.objective_value) == 7.5
    assert bool(candidate_inner_state.eligible)
    with pytest.raises(AttributeError):
        candidate_inner_state.eligible = False


def test_trial_evaluator_threads_explicit_incumbent_into_forward_path() -> None:
    forward_result = _forward_result(success=False, primal_success=False)
    forward_calls: list[tuple[object, ...]] = []
    gradient_calls: list[tuple[np.ndarray, np.ndarray, object]] = []

    def compiled_forward_result_for(*args):
        forward_calls.append(args)
        return forward_result

    def compiled_total_gradient_for(coil_dofs, solved_x, factors):
        gradient_calls.append(
            (
                np.asarray(coil_dofs),
                np.asarray(solved_x),
                factors,
            )
        )
        return (
            jnp.asarray([1.25, -2.5], dtype=jnp.float64),
            jnp.asarray(True, dtype=jnp.bool_),
        )

    bundle = {
        "state": {
            "baseline_coil_dofs": np.asarray([-1.0, -2.0], dtype=np.float64),
            "baseline_x": np.asarray([-3.0, -4.0], dtype=np.float64),
            "baseline_linear_solve_factors": None,
        },
        "compiled_forward_result_for": compiled_forward_result_for,
        "compiled_forward_result_from_anchor_for": compiled_forward_result_for,
        "compiled_total_gradient_for": compiled_total_gradient_for,
    }
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )
    incumbent = surface_objectives_traceable.TraceableObjectiveInnerState(
        coil_dofs=jnp.asarray([0.25, -0.75], dtype=jnp.float64),
        solved_x=jnp.asarray([1.5, -2.5], dtype=jnp.float64),
        objective_value=jnp.asarray(4.5, dtype=jnp.float64),
        eligible=jnp.asarray(True, dtype=jnp.bool_),
    )

    trial = session.trial_evaluator()(
        np.asarray([8.0, 9.0], dtype=np.float64),
        incumbent,
    )

    assert len(forward_calls) == 1
    forwarded_anchor = forward_calls[0][1:]
    if len(forwarded_anchor) == 1 and isinstance(
        forwarded_anchor[0],
        surface_objectives_traceable.TraceableObjectiveInnerState,
    ):
        forwarded_state = forwarded_anchor[0]
        np.testing.assert_array_equal(forwarded_state.coil_dofs, incumbent.coil_dofs)
        np.testing.assert_array_equal(forwarded_state.solved_x, incumbent.solved_x)
        np.testing.assert_array_equal(
            forwarded_state.objective_value,
            incumbent.objective_value,
        )
    else:
        assert any(
            np.array_equal(np.asarray(argument), np.asarray(incumbent.coil_dofs))
            for argument in forwarded_anchor
        )
        assert any(
            np.array_equal(np.asarray(argument), np.asarray(incumbent.solved_x))
            for argument in forwarded_anchor
        )
        assert any(
            np.array_equal(np.asarray(argument), np.asarray(incumbent.objective_value))
            for argument in forwarded_anchor
        )
    assert len(gradient_calls) == 1
    np.testing.assert_array_equal(gradient_calls[0][0], incumbent.coil_dofs)
    np.testing.assert_array_equal(gradient_calls[0][1], incumbent.solved_x)
    assert trial.gradient_source == "incumbent"


def test_trial_evaluator_marks_rejected_candidate_inner_state_ineligible() -> None:
    forward_result = _forward_result(success=False, primal_success=True)
    bundle, _gradient_calls, _forward_calls = _compiled_bundle(forward_result)
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )

    trial = session.trial_evaluator()(np.asarray([8.0, 9.0], dtype=np.float64))

    candidate_inner_state = trial.candidate_inner_state
    np.testing.assert_array_equal(
        candidate_inner_state.coil_dofs,
        np.asarray([8.0, 9.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(candidate_inner_state.solved_x, forward_result["x"])
    assert not bool(candidate_inner_state.eligible)


def _inner_state(
    coil_dofs: tuple[float, float],
    solved_x: tuple[float, float],
    objective_value: float,
    *,
    eligible: bool,
) -> surface_objectives_traceable.TraceableObjectiveInnerState:
    return surface_objectives_traceable.TraceableObjectiveInnerState(
        coil_dofs=jnp.asarray(coil_dofs, dtype=jnp.float64),
        solved_x=jnp.asarray(solved_x, dtype=jnp.float64),
        objective_value=jnp.asarray(objective_value, dtype=jnp.float64),
        eligible=jnp.asarray(eligible, dtype=jnp.bool_),
    )


def _incumbent_evaluation(
    candidate_inner_state: surface_objectives_traceable.TraceableObjectiveInnerState,
) -> surface_objectives_traceable.TraceableObjectiveIncumbentEvaluation:
    return surface_objectives_traceable.TraceableObjectiveIncumbentEvaluation(
        value=jnp.asarray(candidate_inner_state.objective_value + 10.0),
        gradient=jnp.asarray([1.25, -2.5], dtype=jnp.float64),
        candidate_inner_state=candidate_inner_state,
    )


def test_accepted_incumbent_controller_rejected_evaluation_keeps_incumbent() -> None:
    initial_state = _inner_state(
        (-1.0, -2.0),
        (-3.0, -4.0),
        5.0,
        eligible=True,
    )
    rejected_state = _inner_state(
        (8.0, 9.0),
        (3.0, 4.0),
        7.5,
        eligible=False,
    )
    observed_incumbents = []

    def compiled_evaluate(_parameters, incumbent):
        observed_incumbents.append(incumbent)
        return _incumbent_evaluation(rejected_state)

    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        compiled_evaluate,
        initial_state,
    )

    value, gradient = controller.value_and_grad(
        np.asarray([8.0, 9.0], dtype=np.float64)
    )
    controller.value_and_grad(np.asarray([10.0, 11.0], dtype=np.float64))

    assert callable(controller.value_and_grad)
    assert callable(controller.accept)
    assert value == 17.5
    np.testing.assert_array_equal(gradient, np.asarray([1.25, -2.5]))
    assert len(observed_incumbents) == 2
    assert all(incumbent is initial_state for incumbent in observed_incumbents)
    assert controller.current_inner_state is initial_state


def test_accepted_incumbent_controller_accepts_exact_candidate_state() -> None:
    initial_state = _inner_state(
        (-1.0, -2.0),
        (-3.0, -4.0),
        5.0,
        eligible=True,
    )
    accepted_state = _inner_state(
        (8.0, 9.0),
        (3.0, 4.0),
        7.5,
        eligible=True,
    )
    observed_incumbents = []

    def compiled_evaluate(_parameters, incumbent):
        observed_incumbents.append(incumbent)
        return _incumbent_evaluation(accepted_state)

    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        compiled_evaluate,
        initial_state,
    )
    parameters = np.asarray([8.0, 9.0], dtype=np.float64)

    controller.value_and_grad(parameters)
    controller.accept(parameters)
    controller.value_and_grad(np.asarray([10.0, 11.0], dtype=np.float64))

    assert controller.current_inner_state is accepted_state
    assert observed_incumbents[0] is initial_state
    assert observed_incumbents[1] is accepted_state


def test_accepted_incumbent_controller_hash_mismatch_fails_closed() -> None:
    initial_state = _inner_state(
        (-1.0, -2.0),
        (-3.0, -4.0),
        5.0,
        eligible=True,
    )
    candidate_state = _inner_state(
        (8.0, 9.0),
        (3.0, 4.0),
        7.5,
        eligible=True,
    )
    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        lambda _parameters, _incumbent: _incumbent_evaluation(candidate_state),
        initial_state,
    )
    controller.value_and_grad(np.asarray([8.0, 9.0], dtype=np.float64))

    with pytest.raises(RuntimeError, match="do not match"):
        controller.accept(np.asarray([8.0, 10.0], dtype=np.float64))

    assert controller.current_inner_state is initial_state


def test_accepted_incumbent_controller_ineligible_promotion_fails_closed() -> None:
    initial_state = _inner_state(
        (-1.0, -2.0),
        (-3.0, -4.0),
        5.0,
        eligible=True,
    )
    ineligible_state = _inner_state(
        (8.0, 9.0),
        (3.0, 4.0),
        7.5,
        eligible=False,
    )
    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        lambda _parameters, _incumbent: _incumbent_evaluation(ineligible_state),
        initial_state,
    )
    parameters = np.asarray([8.0, 9.0], dtype=np.float64)
    controller.value_and_grad(parameters)

    with pytest.raises(RuntimeError, match="uncertified"):
        controller.accept(parameters)

    assert controller.current_inner_state is initial_state


def test_accepted_incumbent_controller_accepts_earlier_trial_by_identity() -> None:
    initial_state = _inner_state((-1.0, -2.0), (-3.0, -4.0), 5.0, eligible=True)
    first_state = _inner_state((8.0, 9.0), (3.0, 4.0), 7.5, eligible=True)
    second_state = _inner_state((10.0, 11.0), (5.0, 6.0), 9.5, eligible=True)
    states = iter((first_state, second_state))

    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        lambda _parameters, _incumbent: _incumbent_evaluation(next(states)),
        initial_state,
    )
    first_parameters = np.asarray([8.0, 9.0], dtype=np.float64)
    second_parameters = np.asarray([10.0, 11.0], dtype=np.float64)

    controller.value_and_grad(first_parameters)
    controller.value_and_grad(second_parameters)
    controller.accept(first_parameters)

    assert controller.current_inner_state is first_state
    with pytest.raises(RuntimeError, match="do not match"):
        controller.accept(second_parameters)


def test_accepted_incumbent_controller_discards_candidate_across_acceptance() -> None:
    initial_state = _inner_state((-1.0, -2.0), (-3.0, -4.0), 5.0, eligible=True)
    first_state = _inner_state((8.0, 9.0), (3.0, 4.0), 7.5, eligible=True)
    stale_state = _inner_state((10.0, 11.0), (5.0, 6.0), 9.5, eligible=True)
    first_parameters = np.asarray([8.0, 9.0], dtype=np.float64)
    second_parameters = np.asarray([10.0, 11.0], dtype=np.float64)

    def compiled_evaluate(parameters, incumbent):
        if bool(np.all(np.asarray(parameters) == first_parameters)):
            return _incumbent_evaluation(first_state)
        # The second evaluation overlaps an acceptance: the incumbent it was
        # evaluated from is promoted away before its result is inserted.
        controller.accept(first_parameters)
        assert incumbent is initial_state
        return _incumbent_evaluation(stale_state)

    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        compiled_evaluate,
        initial_state,
    )

    controller.value_and_grad(first_parameters)
    controller.value_and_grad(second_parameters)

    assert controller.current_inner_state is first_state
    with pytest.raises(RuntimeError, match="do not match"):
        controller.accept(second_parameters)
    assert controller.current_inner_state is first_state


def test_accepted_incumbent_controller_is_transfer_guard_safe() -> None:
    initial_state = _inner_state((-1.0, -2.0), (-3.0, -4.0), 5.0, eligible=True)
    accepted_state = _inner_state((8.0, 9.0), (3.0, 4.0), 7.5, eligible=True)
    evaluation = _incumbent_evaluation(accepted_state)

    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        lambda _parameters, _incumbent: evaluation,
        initial_state,
    )
    parameters = np.asarray([8.0, 9.0], dtype=np.float64)

    with jax.transfer_guard("disallow"):
        value, gradient = controller.value_and_grad(parameters)
        controller.accept(parameters)

    assert controller.current_inner_state is accepted_state
    assert value == 17.5
    np.testing.assert_array_equal(gradient, np.asarray([1.25, -2.5]))


def test_accepted_incumbent_session_factory_is_transfer_guard_safe() -> None:
    forward_result = _forward_result(success=True, primal_success=True)
    bundle, _gradient_calls, _forward_calls = _compiled_bundle(forward_result)
    bundle["state"]["baseline_value"] = jnp.asarray(5.0, dtype=jnp.float64)
    bundle["compiled_forward_result_from_anchor_for"] = (
        lambda _parameters, _incumbent: forward_result
    )
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )

    with jax.transfer_guard("disallow"):
        controller = session.accepted_incumbent_host_value_and_grad()

    assert bool(np.asarray(jax.device_get(controller.current_inner_state.eligible)))


def test_session_candidate_evaluation_uses_explicit_anchor_under_transfer_guard() -> None:
    forward_result = _forward_result(success=True, primal_success=True)
    anchored_calls: list[tuple[object, object]] = []
    gradient_calls: list[tuple[object, object, object]] = []
    gradient = jnp.asarray([1.25, -2.5], dtype=jnp.float64)
    adjoint_success = jnp.asarray(True, dtype=jnp.bool_)

    def baseline_forward(_parameters):
        pytest.fail("anchored candidate evaluation must not call baseline forward")

    def anchored_forward(parameters, incumbent_state):
        anchored_calls.append((parameters, incumbent_state))
        return forward_result

    def total_gradient(coil_dofs, solved_x, factors):
        gradient_calls.append((coil_dofs, solved_x, factors))
        return gradient, adjoint_success

    bundle = {
        "state": {
            "baseline_coil_dofs": jnp.asarray([-1.0, -2.0], dtype=jnp.float64),
            "baseline_x": jnp.asarray([-3.0, -4.0], dtype=jnp.float64),
            "baseline_linear_solve_factors": None,
        },
        "compiled_forward_result_for": baseline_forward,
        "compiled_forward_result_from_anchor_for": anchored_forward,
        "compiled_total_gradient_for": total_gradient,
    }
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )
    incumbent_state = _inner_state(
        (0.25, -0.75),
        (1.5, -2.5),
        4.5,
        eligible=True,
    )
    parameters = np.asarray([8.0, 9.0], dtype=np.float64)

    with jax.transfer_guard("disallow"):
        evaluation = session.evaluate_candidate_from_anchor(
            parameters,
            incumbent_state,
        )

    assert isinstance(
        evaluation,
        surface_objectives_traceable.TraceableObjectiveCandidateEvaluation,
    )
    assert len(anchored_calls) == 1
    np.testing.assert_array_equal(anchored_calls[0][0], parameters)
    assert anchored_calls[0][1] is incumbent_state
    assert len(gradient_calls) == 1
    np.testing.assert_array_equal(evaluation.gradient, gradient)
    assert evaluation.forward_result["x"] is forward_result["x"]


def test_session_candidate_evaluation_skips_fallback_gradient() -> None:
    forward_result = _forward_result(success=False, primal_success=False)
    anchored_calls: list[tuple[object, object]] = []
    gradient_calls: list[tuple[object, object, object]] = []

    def baseline_forward(_parameters):
        pytest.fail("anchored candidate evaluation must not call baseline forward")

    def anchored_forward(parameters, incumbent_state):
        anchored_calls.append((parameters, incumbent_state))
        return forward_result

    def total_gradient(coil_dofs, solved_x, factors):
        gradient_calls.append((coil_dofs, solved_x, factors))
        pytest.fail("failed terminal forward must not evaluate a fallback gradient")

    bundle = {
        "state": {
            "baseline_coil_dofs": jnp.asarray([-1.0, -2.0], dtype=jnp.float64),
            "baseline_x": jnp.asarray([-3.0, -4.0], dtype=jnp.float64),
            "baseline_linear_solve_factors": None,
        },
        "compiled_forward_result_for": baseline_forward,
        "compiled_forward_result_from_anchor_for": anchored_forward,
        "compiled_total_gradient_for": total_gradient,
    }
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )
    incumbent_state = _inner_state(
        (0.25, -0.75),
        (1.5, -2.5),
        4.5,
        eligible=True,
    )

    with jax.transfer_guard("disallow"):
        evaluation = session.evaluate_candidate_from_anchor(
            np.asarray([8.0, 9.0], dtype=np.float64),
            incumbent_state,
        )

    assert len(anchored_calls) == 1
    assert anchored_calls[0][1] is incumbent_state
    assert gradient_calls == []
    assert evaluation.gradient_source == "unavailable"
    assert np.isnan(np.asarray(jax.device_get(evaluation.gradient))).all()
    assert not bool(np.asarray(jax.device_get(evaluation.candidate_inner_state.eligible)))


def test_session_candidate_evaluation_rejects_ineligible_anchor() -> None:
    forward_result = _forward_result(success=True, primal_success=True)
    bundle, _gradient_calls, _forward_calls = _compiled_bundle(forward_result)
    bundle["compiled_forward_result_from_anchor_for"] = (
        lambda _parameters, _incumbent: forward_result
    )
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )
    ineligible_state = _inner_state(
        (0.25, -0.75),
        (1.5, -2.5),
        4.5,
        eligible=False,
    )

    with jax.transfer_guard("disallow"), pytest.raises(
        RuntimeError,
        match="eligible anchor",
    ):
        session.evaluate_candidate_from_anchor(
            np.asarray([8.0, 9.0], dtype=np.float64),
            ineligible_state,
        )


def test_accepted_incumbent_controllers_from_same_session_are_isolated() -> None:
    forward_result = _forward_result(success=True, primal_success=True)
    bundle, _gradient_calls, _forward_calls = _compiled_bundle(forward_result)
    bundle["state"]["baseline_value"] = jnp.asarray(5.0, dtype=jnp.float64)
    bundle["compiled_forward_result_from_anchor_for"] = (
        lambda _parameters, _incumbent: forward_result
    )
    session = surface_objectives_traceable.TraceableObjectiveSession.from_optimizer_compiled_bundle(
        bundle
    )

    first = session.accepted_incumbent_host_value_and_grad()
    second = session.accepted_incumbent_host_value_and_grad()
    parameters = np.asarray([8.0, 9.0], dtype=np.float64)

    assert first is not second
    first.value_and_grad(parameters)
    first.accept(parameters)

    assert first.current_inner_state is not second.current_inner_state
    np.testing.assert_array_equal(
        second.current_inner_state.coil_dofs,
        bundle["state"]["baseline_coil_dofs"],
    )
    np.testing.assert_array_equal(
        second.current_inner_state.solved_x,
        bundle["state"]["baseline_x"],
    )


def test_accepted_incumbent_controller_drives_host_bfgs_via_callback() -> None:
    import simsopt_jax.geo.optimizer_host_lbfgs as _host_lbfgs

    def _rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
        value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
        gradient = np.asarray(
            (
                -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] ** 2),
            ),
            dtype=np.float64,
        )
        return float(value), gradient

    def compiled_evaluate(candidate, incumbent):
        x = np.asarray(candidate, dtype=np.float64)
        value, gradient = _rosenbrock(x)
        candidate_state = _inner_state(
            (float(x[0]), float(x[1])),
            (2.0 * float(x[0]), 2.0 * float(x[1])),
            value,
            eligible=True,
        )
        return surface_objectives_traceable.TraceableObjectiveIncumbentEvaluation(
            value=jnp.asarray(value, dtype=jnp.float64),
            gradient=jnp.asarray(gradient, dtype=jnp.float64),
            candidate_inner_state=candidate_state,
        )

    x0 = np.asarray((-1.2, 1.0), dtype=np.float64)
    initial_value, _ = _rosenbrock(x0)
    controller = surface_objectives_traceable.AcceptedIncumbentHostValueAndGrad(
        compiled_evaluate,
        _inner_state(
            (float(x0[0]), float(x0[1])),
            (2.0 * float(x0[0]), 2.0 * float(x0[1])),
            initial_value,
            eligible=True,
        ),
    )
    accepted_parameters = []

    def promote(x_accepted):
        controller.accept(x_accepted)
        accepted_parameters.append(np.asarray(x_accepted, dtype=np.float64).copy())

    result = _host_lbfgs.minimize_bfgs_host_core(
        controller.value_and_grad,
        x0,
        maxiter=100,
        callback=promote,
    )

    assert result.converged
    assert result.k == len(accepted_parameters) > 0
    np.testing.assert_array_equal(
        np.asarray(controller.current_inner_state.coil_dofs),
        accepted_parameters[-1],
    )
    promoted_value, _ = _rosenbrock(accepted_parameters[-1])
    assert float(controller.current_inner_state.objective_value) == promoted_value
    assert bool(controller.current_inner_state.eligible)


def test_general_forward_retries_newton_from_incumbent_when_predictor_fails(
    monkeypatch,
) -> None:
    incumbent_x = jnp.asarray([1.5, -2.5], dtype=jnp.float64)
    failed_prediction = jnp.asarray([99.0, 101.0], dtype=jnp.float64)
    solved_x = jnp.asarray([1.25, -2.25], dtype=jnp.float64)
    monkeypatch.setattr(
        surface_objectives_traceable,
        "_traceable_predict_warmstart_x",
        lambda *_args, **_kwargs: (
            failed_prediction,
            jnp.asarray(False, dtype=jnp.bool_),
        ),
    )
    monkeypatch.setattr(
        surface_objectives_traceable,
        "_evaluate_traceable_total_objective_with_raw_terms",
        lambda *_args, **_kwargs: (jnp.asarray(7.5, dtype=jnp.float64), None),
    )

    def unpack(x, _optimize_G, coil_set_spec):
        del coil_set_spec
        return x[:-1], x[-1], None

    def run_code_traceable(_coil_set_spec, sdofs, iota, _G, **_kwargs):
        warmstart_x = jnp.concatenate((sdofs, jnp.reshape(iota, (1,))))
        retried_x = warmstart_x + jnp.asarray([-0.25, 0.25], dtype=jnp.float64)
        return {
            "x": retried_x,
            "sdofs": retried_x[:-1],
            "iota": retried_x[-1],
            "G": None,
            "success": jnp.asarray(True, dtype=jnp.bool_),
            "primal_success": jnp.asarray(True, dtype=jnp.bool_),
            "adjoint_linear_solve_available": jnp.asarray(True, dtype=jnp.bool_),
            "newton_iter": jnp.asarray(1, dtype=jnp.int32),
        }

    booz_jax = types.SimpleNamespace(
        _unpack_decision_vector_jax=unpack,
        run_code_traceable=run_code_traceable,
    )

    result = surface_objectives_traceable._traceable_general_forward_result(
        booz_jax,
        lambda coil_dofs: coil_dofs,
        coil_dofs=jnp.asarray([8.0, 9.0], dtype=jnp.float64),
        baseline_x=incumbent_x,
        baseline_value=jnp.asarray(4.5, dtype=jnp.float64),
        baseline_linear_solve_factors=None,
        linearization_kind="exact_jacobian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        optimize_G=False,
        baseline_coil_dofs=jnp.asarray([0.25, -0.75], dtype=jnp.float64),
        predictor_kind="exact",
        objective_kwargs={},
        success_filter=None,
        newton_trace_capacity=2,
    )

    np.testing.assert_array_equal(result["x"], solved_x)
    assert bool(result["primal_success"])


def test_general_forward_propagates_raw_inner_and_newton_trial_fields(
    monkeypatch,
) -> None:
    solved_x = jnp.asarray([3.0, 4.0], dtype=jnp.float64)
    monkeypatch.setattr(
        surface_objectives_traceable,
        "_traceable_predict_warmstart_x",
        lambda *_args, **_kwargs: (solved_x, jnp.asarray(True, dtype=jnp.bool_)),
    )
    monkeypatch.setattr(
        surface_objectives_traceable,
        "_evaluate_traceable_total_objective_with_raw_terms",
        lambda *_args, **_kwargs: (jnp.asarray(7.5, dtype=jnp.float64), None),
    )
    booz_jax = types.SimpleNamespace(
        _unpack_decision_vector_jax=lambda x, _optimize_G, coil_set_spec: (
            x[:-1],
            x[-1],
            None,
        ),
        run_code_traceable=lambda *_args, **_kwargs: {
            "x": solved_x,
            "sdofs": solved_x[:-1],
            "iota": solved_x[-1],
            "G": None,
            "success": jnp.asarray(False, dtype=jnp.bool_),
            "primal_success": jnp.asarray(True, dtype=jnp.bool_),
            "adjoint_linear_solve_available": jnp.asarray(True, dtype=jnp.bool_),
            "newton_iter": jnp.asarray(6, dtype=jnp.int32),
            "newton_attempted_iterations": jnp.asarray(8, dtype=jnp.int32),
            "newton_stop_reason_code": jnp.asarray(2, dtype=jnp.int32),
            "newton_last_linear_solve_success": jnp.asarray(False, dtype=jnp.bool_),
            "inner_penalty_residual_l2": jnp.asarray(4.25, dtype=jnp.float64),
            "final_gradient_inf_norm": jnp.asarray(0.125, dtype=jnp.float64),
        },
    )

    forward_result = surface_objectives_traceable._traceable_general_forward_result(
        booz_jax,
        lambda coil_dofs: coil_dofs,
        coil_dofs=jnp.asarray([8.0, 9.0], dtype=jnp.float64),
        baseline_x=jnp.asarray([-3.0, -4.0], dtype=jnp.float64),
        baseline_value=jnp.asarray(5.0, dtype=jnp.float64),
        baseline_linear_solve_factors=None,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        optimize_G=False,
        baseline_coil_dofs=jnp.asarray([-1.0, -2.0], dtype=jnp.float64),
        predictor_kind="ls",
        objective_kwargs={},
        success_filter=lambda *_args: jnp.asarray(False, dtype=jnp.bool_),
        newton_trace_capacity=2,
    )

    assert float(forward_result["raw_value"]) == 7.5
    assert float(forward_result["value"]) != 7.5
    assert bool(forward_result["predictor_success"])
    assert bool(forward_result["primal_success"])
    assert not bool(forward_result["success"])
    assert not bool(forward_result["newton_success"])
    assert int(forward_result["newton_iterations"]) == 6
    assert int(forward_result["newton_attempted_iterations"]) == 8
    assert int(forward_result["newton_stop_reason_code"]) == 2
    assert not bool(forward_result["newton_last_linear_solve_success"])
    assert float(forward_result["inner_penalty_residual_l2"]) == 4.25
    assert float(forward_result["final_gradient_inf_norm"]) == 0.125
