"""Contracts for caller-authorized traceable warm-start anchors."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

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
