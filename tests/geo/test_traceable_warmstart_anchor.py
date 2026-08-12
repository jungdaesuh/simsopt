"""Contracts for caller-authorized traceable warm-start anchors."""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jax_core
from simsopt_jax.geo.optimizers import optimizer as _optimizer
from simsopt_jax.runtime.trace_annotations import PhaseId, trace_session
from simsopt_jax_adapters.geo import surface_objectives as _compatibility
from simsopt_jax_adapters.geo import surface_objectives_traceable as _traceable


def _linear_solve_status(success):
    value = jnp.asarray(0.0, dtype=jnp.float64)
    return _optimizer._LinearSolveStatus(
        success=jnp.asarray(success, dtype=bool),
        residual=value,
        residual_relative=value,
        iterations=jnp.asarray(1, dtype=jnp.int32),
    )


def _jaxpr_name_stacks(value: object) -> list[str]:
    names: list[str] = []
    if isinstance(value, jax_core.ClosedJaxpr):
        return _jaxpr_name_stacks(value.jaxpr)
    if isinstance(value, jax_core.Jaxpr):
        for equation in value.eqns:
            name = str(equation.source_info.name_stack)
            if name:
                names.append(name)
            for parameter in equation.params.values():
                names.extend(_jaxpr_name_stacks(parameter))
        return names
    if isinstance(value, dict):
        for item in value.values():
            names.extend(_jaxpr_name_stacks(item))
        return names
    if isinstance(value, (tuple, list)):
        for item in value:
            names.extend(_jaxpr_name_stacks(item))
    return names


def test_legacy_adapter_path_reexports_anchor_helpers():
    assert (
        _compatibility._traceable_predict_warmstart_from_anchor
        is _traceable._traceable_predict_warmstart_from_anchor
    )
    assert (
        _compatibility._traceable_select_predictor_linear_solve_factors
        is _traceable._traceable_select_predictor_linear_solve_factors
    )


def test_warmstart_prediction_uses_the_explicit_anchor(monkeypatch):
    anchor_x = jnp.asarray([1.5, -2.5], dtype=jnp.float64)
    anchor_coil_dofs = jnp.asarray([0.25, -0.75], dtype=jnp.float64)
    coil_dofs = jnp.asarray([0.75, -0.25], dtype=jnp.float64)
    anchor_factors = (
        jnp.eye(2, dtype=jnp.float64),
        2.0 * jnp.eye(2, dtype=jnp.float64),
        jnp.asarray([0, 1], dtype=jnp.int32),
    )
    dx = jnp.asarray([0.125, -0.375], dtype=jnp.float32)
    calls = []

    monkeypatch.setattr(
        _traceable,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {},
    )

    def stationarity_jvp(
        x_inner,
        current_coil_dofs,
        coil_dofs_tangent,
        _coil_set_spec_from_dofs,
        **_objective_kwargs,
    ):
        np.testing.assert_allclose(x_inner, anchor_x)
        np.testing.assert_allclose(current_coil_dofs, anchor_coil_dofs)
        np.testing.assert_allclose(
            coil_dofs_tangent,
            coil_dofs - anchor_coil_dofs,
        )
        calls.append("jvp")
        return jnp.asarray([0.25, -0.5], dtype=jnp.float64)

    def solve_linearization(
        _booz_jax,
        solved_x,
        rhs,
        coil_set_spec,
        _objective_kwargs,
        *,
        linear_solve_factors,
        linearization_kind,
        linear_solve_tol,
        linear_solve_stab,
        transpose,
    ):
        np.testing.assert_allclose(solved_x, anchor_x)
        np.testing.assert_allclose(rhs, [-0.25, 0.5])
        np.testing.assert_allclose(coil_set_spec, anchor_coil_dofs)
        assert linear_solve_factors is anchor_factors
        assert linearization_kind == "hessian"
        assert linear_solve_tol == 1.0e-7
        assert linear_solve_stab == 0.25
        assert transpose is False
        calls.append("solve")
        return dx, _linear_solve_status(True)

    monkeypatch.setattr(
        _traceable,
        "_traceable_inner_stationarity_coil_jvp",
        stationarity_jvp,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_solve_linearization",
        solve_linearization,
    )

    predicted, success = _traceable._traceable_predict_warmstart_from_anchor(
        object(),
        lambda current_coil_dofs: current_coil_dofs,
        coil_dofs=coil_dofs,
        anchor_coil_dofs=anchor_coil_dofs,
        anchor_x=anchor_x,
        anchor_linear_solve_factors=anchor_factors,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-7,
        linear_solve_stab=0.25,
        predictor_kind="ls",
        objective_kwargs={},
    )

    assert calls == ["jvp", "solve"]
    assert bool(np.asarray(success))
    assert predicted.dtype == anchor_x.dtype
    np.testing.assert_allclose(predicted, anchor_x + dx)


def test_factor_selection_uses_only_an_eligible_anchor():
    baseline_factors = (
        jnp.eye(2, dtype=jnp.float64),
        2.0 * jnp.eye(2, dtype=jnp.float64),
        jnp.asarray([0, 1], dtype=jnp.int32),
    )
    anchor_factors = (
        3.0 * jnp.eye(2, dtype=jnp.float64),
        4.0 * jnp.eye(2, dtype=jnp.float64),
        jnp.asarray([1, 0], dtype=jnp.int32),
    )

    for eligible, expected_factors in (
        (False, baseline_factors),
        (True, anchor_factors),
    ):
        selected = _traceable._traceable_select_predictor_linear_solve_factors(
            jnp.asarray(eligible, dtype=bool),
            baseline_linear_solve_factors=baseline_factors,
            anchor_linear_solve_factors=anchor_factors,
        )
        for actual, expected in zip(selected, expected_factors):
            np.testing.assert_allclose(actual, expected)


def test_mixed_predictor_uses_fp32_forcing_but_fp64_linear_certificate(monkeypatch):
    anchor_x = jnp.asarray([1.5, -2.5], dtype=jnp.float64)
    anchor_coil_dofs = jnp.asarray([0.25, -0.75], dtype=jnp.float64)
    certificate_anchor_coil_dofs = jnp.asarray(
        [0.25000001, -0.75000001],
        dtype=jnp.float64,
    )
    coil_dofs = jnp.asarray([0.75, -0.25], dtype=jnp.float64)
    observed = {}

    monkeypatch.setattr(
        _traceable,
        "_as_compute_array",
        lambda value: jnp.asarray(value, dtype=jnp.float32),
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {},
    )

    def stationarity_jvp(
        x_inner,
        current_coil_dofs,
        coil_dofs_tangent,
        _coil_set_spec_from_dofs,
        **_objective_kwargs,
    ):
        observed["proposal_state_dtype"] = x_inner.dtype
        observed["proposal_coil_dtype"] = current_coil_dofs.dtype
        observed["proposal_delta_dtype"] = coil_dofs_tangent.dtype
        return jnp.asarray([0.25, -0.5], dtype=jnp.float32)

    def certificate_spec_from_dofs(current_coil_dofs):
        observed["certificate_coil_dofs"] = current_coil_dofs
        return current_coil_dofs

    def solve_linearization(
        _booz_jax,
        solved_x,
        rhs,
        coil_set_spec,
        _objective_kwargs,
        **_solve_kwargs,
    ):
        observed["solve_state_dtype"] = solved_x.dtype
        observed["solve_rhs_dtype"] = rhs.dtype
        observed["solve_spec_dtype"] = coil_set_spec.dtype
        np.testing.assert_array_equal(coil_set_spec, certificate_anchor_coil_dofs)
        return jnp.asarray([0.125, -0.375], dtype=jnp.float64), _linear_solve_status(
            True
        )

    monkeypatch.setattr(
        _traceable,
        "_traceable_inner_stationarity_coil_jvp",
        stationarity_jvp,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_solve_linearization",
        solve_linearization,
    )

    forcing, predicted, _status, success = (
        _traceable._traceable_predict_warmstart_result_from_anchor(
            object(),
            lambda current_coil_dofs: current_coil_dofs,
            certificate_coil_set_spec_from_dofs=certificate_spec_from_dofs,
            anchor_certificate_coil_dofs=certificate_anchor_coil_dofs,
            coil_dofs=coil_dofs,
            anchor_coil_dofs=anchor_coil_dofs,
            anchor_x=anchor_x,
            anchor_linear_solve_factors=None,
            linearization_kind="hessian",
            linear_solve_tol=1.0e-7,
            linear_solve_stab=0.0,
            predictor_kind="ls",
            objective_kwargs={},
        )
    )

    assert forcing.dtype == jnp.float32
    assert predicted.dtype == jnp.float64
    assert bool(np.asarray(success))
    assert observed.pop("proposal_state_dtype") == jnp.float64
    assert observed.pop("proposal_coil_dtype") == jnp.float32
    assert observed.pop("proposal_delta_dtype") == jnp.float32
    np.testing.assert_array_equal(
        observed.pop("certificate_coil_dofs"),
        certificate_anchor_coil_dofs,
    )
    assert observed.pop("solve_state_dtype") == jnp.float64
    assert observed.pop("solve_rhs_dtype") == jnp.float64
    assert observed.pop("solve_spec_dtype") == jnp.float64
    assert observed == {}


def test_exact_warmstart_scopes_jvp_and_solve_under_warmstart_without_changing_result(
    monkeypatch,
):
    monkeypatch.setattr(_traceable, "_traceable_exact_residual_kwargs", lambda _: {})
    monkeypatch.setattr(
        _traceable,
        "_boozer_exact_residual",
        lambda x_inner, coil_set_spec, **_kwargs: x_inner * coil_set_spec,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_solve_linearization",
        lambda _booz, _x, rhs, *_args, **_kwargs: (
            0.5 * rhs,
            _linear_solve_status(True),
        ),
    )
    anchor_x = jnp.asarray([1.5, -2.5], dtype=jnp.float64)
    anchor_coils = jnp.asarray([0.25, -0.75], dtype=jnp.float64)

    def predict(coil_dofs):
        return _traceable._traceable_predict_warmstart_result_from_anchor(
            object(),
            lambda current_coils: current_coils,
            coil_dofs=coil_dofs,
            anchor_coil_dofs=anchor_coils,
            anchor_x=anchor_x,
            anchor_linear_solve_factors=None,
            linearization_kind="exact_jacobian",
            linear_solve_tol=1.0e-10,
            linear_solve_stab=0.0,
            predictor_kind="exact",
            objective_kwargs={},
        )

    coil_dofs = jnp.asarray([0.75, -0.25], dtype=jnp.float64)
    unannotated_result = predict(coil_dofs)
    unannotated_jaxpr = jax.make_jaxpr(predict)(coil_dofs)
    with trace_session():
        annotated_result = predict(coil_dofs)
        annotated_jaxpr = jax.make_jaxpr(lambda current: predict(current))(coil_dofs)

    for unannotated_leaf, annotated_leaf in zip(
        jax.tree.leaves(unannotated_result),
        jax.tree.leaves(annotated_result),
        strict=True,
    ):
        np.testing.assert_array_equal(annotated_leaf, unannotated_leaf)

    unannotated_names = _jaxpr_name_stacks(unannotated_jaxpr)
    annotated_names = _jaxpr_name_stacks(annotated_jaxpr)
    nested_jvp = (
        f"{PhaseId.NEWTON_WARM_START.value}/{PhaseId.NEWTON_RESIDUAL_JVP.value}"
    )
    nested_solve = (
        f"{PhaseId.NEWTON_WARM_START.value}/{PhaseId.NEWTON_LINEAR_SOLVE.value}"
    )
    assert all(
        phase not in name
        for phase in (nested_jvp, nested_solve)
        for name in unannotated_names
    )
    assert any(nested_jvp in name for name in annotated_names)
    assert any(nested_solve in name for name in annotated_names)


def test_exact_general_forward_scopes_full_boozer_solve_without_changing_result(
    monkeypatch,
):
    baseline_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    monkeypatch.setattr(
        _traceable,
        "_traceable_predict_warmstart_x",
        lambda *_args, coil_dofs, **_kwargs: (
            baseline_x + 0.25 * coil_dofs,
            jnp.asarray(True),
        ),
    )
    monkeypatch.setattr(
        _traceable,
        "_evaluate_traceable_total_objective_with_raw_terms",
        lambda solved_x, *_args, **_kwargs: (jnp.sum(solved_x**2), None),
    )

    def unpack(x, _optimize_G, coil_set_spec):
        del coil_set_spec
        return x[:-1], x[-1], None

    def run_code_traceable(coil_set_spec, sdofs, iota, _G, **_kwargs):
        warmstart_x = jnp.concatenate((sdofs, jnp.reshape(iota, (1,))))
        solved_x = warmstart_x + 0.125 * coil_set_spec
        return {
            "x": solved_x,
            "sdofs": solved_x[:-1],
            "iota": solved_x[-1],
            "G": None,
            "success": jnp.asarray(True),
            "primal_success": jnp.asarray(True),
            "adjoint_linear_solve_available": jnp.asarray(True),
            "newton_iter": jnp.asarray(1, dtype=jnp.int32),
        }

    booz_jax = types.SimpleNamespace(
        _unpack_decision_vector_jax=unpack,
        run_code_traceable=run_code_traceable,
    )

    def solve(coil_dofs):
        return _traceable._traceable_general_forward_result(
            booz_jax,
            lambda current_coils: current_coils,
            coil_dofs=coil_dofs,
            baseline_x=baseline_x,
            baseline_value=jnp.asarray(10.0, dtype=jnp.float64),
            baseline_linear_solve_factors=None,
            linearization_kind="exact_jacobian",
            linear_solve_tol=1.0e-10,
            linear_solve_stab=0.0,
            optimize_G=False,
            baseline_coil_dofs=jnp.zeros_like(coil_dofs),
            predictor_kind="exact",
            objective_kwargs={},
            success_filter=None,
            newton_trace_capacity=2,
        )

    coil_dofs = jnp.asarray([0.75, -0.5], dtype=jnp.float64)
    unannotated_result = solve(coil_dofs)
    unannotated_jaxpr = jax.make_jaxpr(solve)(coil_dofs)
    with trace_session():
        annotated_result = solve(coil_dofs)
        annotated_jaxpr = jax.make_jaxpr(lambda current: solve(current))(coil_dofs)

    for unannotated_leaf, annotated_leaf in zip(
        jax.tree.leaves(unannotated_result),
        jax.tree.leaves(annotated_result),
        strict=True,
    ):
        np.testing.assert_array_equal(annotated_leaf, unannotated_leaf)

    unannotated_names = _jaxpr_name_stacks(unannotated_jaxpr)
    annotated_names = _jaxpr_name_stacks(annotated_jaxpr)
    solver_control = PhaseId.NEWTON_SOLVER_CONTROL.value
    assert all(solver_control not in name for name in unannotated_names)
    assert any(solver_control in name for name in annotated_names)


def test_profile_inner_solve_retains_solver_control_scope(monkeypatch):
    baseline_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    monkeypatch.setattr(
        _traceable,
        "_traceable_predict_warmstart_x",
        lambda *_args, coil_dofs, **_kwargs: (
            baseline_x + 0.25 * coil_dofs,
            jnp.asarray(True),
        ),
    )
    monkeypatch.setattr(
        _traceable,
        "_make_traceable_batched_value_and_grad_pipeline",
        lambda compiled_value_and_grad_for: compiled_value_and_grad_for,
    )
    monkeypatch.setattr(
        _traceable,
        "_evaluate_traceable_total_objective",
        lambda solved_x, *_args, **_kwargs: jnp.sum(solved_x**2),
    )

    def unpack(x, _optimize_G, coil_set_spec):
        del coil_set_spec
        return x[:-1], x[-1], None

    def run_code_traceable(coil_set_spec, sdofs, iota, _G, **_kwargs):
        warmstart_x = jnp.concatenate((sdofs, jnp.reshape(iota, (1,))))
        solved_x = warmstart_x + 0.125 * coil_set_spec
        return {
            "x": solved_x,
            "fun": jnp.sum(solved_x**2),
            "success": jnp.asarray(True),
            "nit": jnp.asarray(1, dtype=jnp.int32),
        }

    booz_jax = types.SimpleNamespace(
        boozer_type="exact",
        _unpack_decision_vector_jax=unpack,
        run_code_traceable=run_code_traceable,
    )
    compiled_bundle = {
        "compiled_forward_result_for": lambda coil_dofs: {"value": jnp.sum(coil_dofs)},
        "compiled_value_and_grad_for": lambda coil_dofs: (
            jnp.sum(coil_dofs),
            jnp.ones_like(coil_dofs),
        ),
        "state": {
            "objective_kwargs": {},
            "baseline_coil_dofs": jnp.zeros(2, dtype=jnp.float64),
            "baseline_x": baseline_x,
            "baseline_linear_solve_factors": None,
            "optimize_G": False,
            "predictor_kind": "exact",
            "linearization_kind": "exact_jacobian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
            "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
        },
    }

    def inner_solve():
        profile = (
            _traceable._make_traceable_objective_profile_suite_from_compiled_bundle(
                compiled_bundle,
                booz_jax,
                object(),
            )
        )
        return profile["inner_solve"]

    coil_dofs = jnp.asarray([0.75, -0.5], dtype=jnp.float64)
    unannotated_solve = inner_solve()
    unannotated_result = unannotated_solve(coil_dofs)
    unannotated_jaxpr = jax.make_jaxpr(unannotated_solve)(coil_dofs)
    with trace_session():
        annotated_solve = inner_solve()
        annotated_result = annotated_solve(coil_dofs)
        annotated_jaxpr = jax.make_jaxpr(annotated_solve)(coil_dofs)

    for unannotated_leaf, annotated_leaf in zip(
        jax.tree.leaves(unannotated_result),
        jax.tree.leaves(annotated_result),
        strict=True,
    ):
        np.testing.assert_array_equal(annotated_leaf, unannotated_leaf)

    unannotated_names = _jaxpr_name_stacks(unannotated_jaxpr)
    annotated_names = _jaxpr_name_stacks(annotated_jaxpr)
    solver_control = PhaseId.NEWTON_SOLVER_CONTROL.value
    assert all(solver_control not in name for name in unannotated_names)
    assert any(solver_control in name for name in annotated_names)


def test_factor_selection_cannot_inject_factors_into_a_factorless_route():
    selected = _traceable._traceable_select_predictor_linear_solve_factors(
        jnp.asarray(True, dtype=bool),
        baseline_linear_solve_factors=None,
        anchor_linear_solve_factors=(jnp.eye(2, dtype=jnp.float64),),
    )

    assert selected is None


def test_profile_suite_falls_back_to_baseline_when_anchor_is_ineligible(monkeypatch):
    baseline_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    anchor_x = jnp.asarray([1.5, -1.25], dtype=jnp.float64)
    failed_dx = jnp.asarray([0.125, -0.375], dtype=jnp.float64)
    monkeypatch.setattr(_traceable, "_traceable_exact_residual_kwargs", lambda _: {})
    monkeypatch.setattr(
        _traceable,
        "_boozer_exact_residual",
        lambda x_inner, coil_set_spec, **_kwargs: x_inner + coil_set_spec,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_solve_linearization",
        lambda *_args, **_kwargs: (failed_dx, _linear_solve_status(False)),
    )
    monkeypatch.setattr(
        _traceable,
        "_make_traceable_batched_value_and_grad_pipeline",
        lambda compiled_value_and_grad_for: compiled_value_and_grad_for,
    )
    compiled_bundle = {
        "compiled_forward_result_for": object(),
        "compiled_value_and_grad_for": object(),
        "state": {
            "objective_kwargs": {},
            "baseline_coil_dofs": jnp.asarray([0.0, 0.0], dtype=jnp.float64),
            "baseline_x": baseline_x,
            "baseline_linear_solve_factors": None,
            "optimize_G": False,
            "predictor_kind": "exact",
            "linearization_kind": "exact_jacobian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
            "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
        },
    }
    booz_jax = types.SimpleNamespace(
        boozer_type="exact",
        _unpack_decision_vector_jax=lambda x, optimize_G, coil_set_spec: (
            x[:-1],
            x[-1],
            None,
        ),
    )
    profile_suite = (
        _traceable._make_traceable_objective_profile_suite_from_compiled_bundle(
            compiled_bundle,
            booz_jax,
            object(),
        )
    )
    predict = profile_suite["current_incumbent_warmstart_predict"]
    coil_dofs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    anchor_coil_dofs = jnp.asarray([0.25, -0.5], dtype=jnp.float64)

    eligible = predict(coil_dofs, anchor_coil_dofs, anchor_x, None, True)
    ineligible = predict(coil_dofs, anchor_coil_dofs, anchor_x, None, False)

    assert bool(np.asarray(eligible["anchor_used"]))
    assert not bool(np.asarray(ineligible["anchor_used"]))
    np.testing.assert_allclose(eligible["x"], anchor_x + failed_dx)
    np.testing.assert_allclose(ineligible["x"], baseline_x + failed_dx)
