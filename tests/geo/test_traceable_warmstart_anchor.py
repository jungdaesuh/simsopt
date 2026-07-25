"""Contracts for caller-authorized traceable warm-start anchors."""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np

from simsopt_jax.geo.optimizers import optimizer as _optimizer
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
