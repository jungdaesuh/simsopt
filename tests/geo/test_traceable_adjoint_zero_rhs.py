"""Device-safe exact-zero handling for traceable adjoint right-hand sides."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax.geo.optimizers import optimizer as _optimizer
from simsopt_jax_adapters.geo import surface_objectives_traceable as _traceable


@pytest.mark.parametrize(
    ("rhs_values", "expected"),
    [([0.0, 0.0], True), ([0.0, 1.0e-30], False)],
)
def test_traceable_adjoint_zero_rhs_gate_is_strict_transfer_safe(
    rhs_values,
    expected,
):
    device = jax.devices()[0]
    rhs = jax.device_put(np.asarray(rhs_values, dtype=np.float64), device)
    gate = jax.jit(_traceable._traceable_adjoint_rhs_exactly_zero)

    with jax.transfer_guard("disallow"):
        result = gate(rhs)
        result.block_until_ready()

    assert bool(np.asarray(jax.device_get(result))) is expected


@pytest.mark.parametrize("x_coefficient", [0.0, 1.0e-12])
def test_traceable_gradient_skips_only_exact_zero_adjoint_rhs(
    monkeypatch,
    x_coefficient,
):
    coil_dofs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    solved_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)

    def scalar_objective_fn(
        x_inner,
        current_coil_dofs,
        _coil_set_spec,
        *,
        objective_kwargs,
    ):
        del _coil_set_spec, objective_kwargs
        x_term = jnp.asarray(x_coefficient, dtype=jnp.float64) * jnp.sum(x_inner)
        return jnp.dot(current_coil_dofs, current_coil_dofs) + x_term

    def solve_linearization(*args, **_kwargs):
        rhs = args[2]
        adjoint = jnp.full_like(rhs, 1.0e6)
        status = _optimizer._linear_solve_status(
            adjoint,
            jnp.zeros_like(rhs),
            rhs,
            tol=1.0e-10,
            iterations=1,
        )._replace(success=jnp.asarray(True))
        return adjoint, status

    def directional_inner_stationarity(_x_inner, tangent, coil_set_spec, **_kwargs):
        return jnp.dot(tangent[: coil_set_spec.shape[0]], coil_set_spec)

    monkeypatch.setattr(
        _traceable,
        "_traceable_solve_linearization",
        solve_linearization,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_directional_inner_stationarity",
        directional_inner_stationarity,
    )

    (
        direct_grad,
        implicit_grad,
        total_grad,
        linear_solve_success,
        _trust,
        _execution_counts,
        _adjoint_evidence,
    ) = _traceable._traceable_objective_gradient_parts(
        object(),
        lambda current_coil_dofs: current_coil_dofs,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        solved_linear_solve_factors=None,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        objective_kwargs={},
        scalar_objective_fn=scalar_objective_fn,
    )

    expected_implicit = np.zeros(2) if x_coefficient == 0.0 else np.full(2, 1.0e6)
    expected_direct = np.asarray([2.0, -4.0])
    np.testing.assert_allclose(np.asarray(direct_grad), expected_direct)
    np.testing.assert_allclose(np.asarray(implicit_grad), expected_implicit)
    np.testing.assert_allclose(
        np.asarray(total_grad),
        expected_direct - expected_implicit,
    )
    assert bool(np.asarray(linear_solve_success))

    (
        fused_total_grad,
        fused_linear_solve_success,
        _fused_trust,
        fused_execution_counts,
        _fused_adjoint_evidence,
    ) = _traceable._traceable_fused_total_gradient_canary(
        object(),
        lambda current_coil_dofs: current_coil_dofs,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        solved_linear_solve_factors=None,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        objective_kwargs={},
        scalar_objective_fn=scalar_objective_fn,
    )

    np.testing.assert_allclose(
        np.asarray(fused_total_grad),
        expected_direct - expected_implicit,
    )
    assert bool(np.asarray(fused_linear_solve_success))
    expected_adjoint_execution_count = 0 if x_coefficient == 0.0 else 1
    assert (
        int(np.asarray(fused_execution_counts.adjoint_execution_count))
        == expected_adjoint_execution_count
    )


def test_traceable_fused_total_gradient_masks_failed_adjoint(monkeypatch):
    coil_dofs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    solved_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)

    def scalar_objective_fn(
        x_inner,
        current_coil_dofs,
        _coil_set_spec,
        *,
        objective_kwargs,
    ):
        del _coil_set_spec, objective_kwargs
        return jnp.sum(x_inner) + jnp.dot(current_coil_dofs, current_coil_dofs)

    def failed_solve(*args, **_kwargs):
        rhs = args[2]
        adjoint = jnp.ones_like(rhs)
        status = _optimizer._linear_solve_status(
            adjoint,
            jnp.ones_like(rhs),
            rhs,
            tol=1.0e-10,
            iterations=1,
        )._replace(success=jnp.asarray(False))
        return adjoint, status

    monkeypatch.setattr(_traceable, "_traceable_solve_linearization", failed_solve)
    monkeypatch.setattr(
        _traceable,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_directional_inner_stationarity",
        lambda _x_inner, tangent, coil_set_spec, **_kwargs: jnp.dot(
            tangent,
            coil_set_spec,
        ),
    )

    total_grad, success, _trust, execution_counts, evidence = (
        _traceable._traceable_fused_total_gradient_canary(
            object(),
            lambda current_coil_dofs: current_coil_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=None,
            linearization_kind="hessian",
            linear_solve_tol=1.0e-10,
            linear_solve_stab=0.0,
            objective_kwargs={},
            scalar_objective_fn=scalar_objective_fn,
        )
    )

    assert np.isnan(np.asarray(total_grad)).all()
    assert not bool(np.asarray(success))
    assert int(np.asarray(execution_counts.adjoint_execution_count)) == 1
    np.testing.assert_allclose(np.asarray(evidence.adjoint_output), np.ones(2))
