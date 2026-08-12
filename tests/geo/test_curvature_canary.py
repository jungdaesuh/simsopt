from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import simsopt_jax.solve.fullspace_curvature_canary as fullspace_module
from simsopt_jax.geo.optimizers.curvature_canary import (
    materialize_exact_lagrangian_hessian,
    project_equality_multipliers,
    run_dense_curvature_canary,
    solve_dense_primal_dual_direction,
)
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_sqp import CfsSqp1EndpointDiagnostics

jax.config.update("jax_enable_x64", True)


def test_exact_hessian_materialization_preserves_tail_and_orientation() -> None:
    matrix = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 2.0, 0.25, 0.0],
            [0.0, 0.0, 0.25, 1.5, 0.1],
            [0.0, 0.0, 0.0, 0.1, 1.0],
        ],
        dtype=jnp.float64,
    )

    def quadratic(x: jax.Array) -> jax.Array:
        return 0.5 * x @ matrix @ x

    dense, symmetry_defect, action_defect = materialize_exact_lagrangian_hessian(
        quadratic,
        jnp.linspace(-0.2, 0.3, 5, dtype=jnp.float64),
        batch_width=2,
    )

    np.testing.assert_allclose(dense, matrix, rtol=0.0, atol=1.0e-14)
    assert float(symmetry_defect) <= 1.0e-14
    assert float(action_defect) <= 1.0e-14


def test_multiplier_projection_is_stationary_in_constraint_range() -> None:
    gradient = jnp.asarray([2.0, -1.0, 3.0], dtype=jnp.float64)
    jacobian = jnp.asarray([[1.0, 2.0, -1.0]], dtype=jnp.float64)

    projection = project_equality_multipliers(gradient, jacobian)
    stationarity = gradient + jacobian.T @ projection.multipliers

    np.testing.assert_allclose(jacobian @ stationarity, 0.0, atol=1.0e-14)
    assert float(projection.relative_residual) <= 1.0e-14
    assert float(projection.forward_error_bound) <= 1.0e-14
    assert bool(projection.all_finite)


def test_exact_lagrangian_hessian_includes_nonlinear_constraint_curvature() -> None:
    coordinates = jnp.asarray([0.5, 0.75], dtype=jnp.float64)

    def joint(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        objective = 0.5 * (x[0] ** 2 + 4.0 * x[1] ** 2)
        constraints = jnp.asarray([x[0] ** 2 + x[1] - 1.0])
        return objective, constraints

    objective_gradient = jax.grad(lambda x: joint(x)[0])(coordinates)
    constraint_jacobian = jax.jacrev(lambda x: joint(x)[1])(coordinates)
    projection = project_equality_multipliers(
        objective_gradient,
        constraint_jacobian,
    )

    def lagrangian(x: jax.Array) -> jax.Array:
        objective, constraints = joint(x)
        return objective + jnp.vdot(projection.multipliers, constraints)

    dense, symmetry_defect, action_defect = materialize_exact_lagrangian_hessian(
        lagrangian,
        coordinates,
        batch_width=1,
    )
    expected = jnp.diag(
        jnp.asarray(
            [1.0 + 2.0 * projection.multipliers[0], 4.0],
            dtype=jnp.float64,
        )
    )
    probe = jnp.asarray([0.3, -0.7], dtype=jnp.float64)
    epsilon = 1.0e-6
    finite_difference_action = (
        jax.grad(lagrangian)(coordinates + epsilon * probe)
        - jax.grad(lagrangian)(coordinates - epsilon * probe)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(dense, expected, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        dense @ probe,
        finite_difference_action,
        rtol=1.0e-9,
        atol=1.0e-10,
    )
    assert float(symmetry_defect) <= 1.0e-14
    assert float(action_defect) <= 1.0e-14


def test_dense_primal_dual_direction_solves_indefinite_kkt_system() -> None:
    curvature = jnp.asarray([[2.0, 0.0], [0.0, -1.0]], dtype=jnp.float64)
    jacobian = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    stationarity = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    constraints = jnp.asarray([0.25], dtype=jnp.float64)

    direction = solve_dense_primal_dual_direction(
        curvature,
        jacobian,
        stationarity,
        constraints,
    )

    np.testing.assert_allclose(
        curvature @ direction.primal + jacobian.T @ direction.dual,
        -stationarity,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        jacobian @ direction.primal,
        -constraints,
        atol=1.0e-14,
    )
    assert bool(direction.all_finite)
    assert float(direction.relative_residual) <= 1.0e-14
    assert float(direction.condition_estimate) >= 1.0
    assert float(direction.forward_error_bound) <= 1.0e-12


def test_exact_curvature_beats_identity_on_anisotropic_quadratic() -> None:
    curvature = jnp.diag(jnp.asarray([1.0, 100.0, 4.0], dtype=jnp.float64))
    target = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)

    def joint(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        delta = x - target
        objective = 0.5 * delta @ curvature @ delta
        constraints = jnp.reshape(x[2], (1,))
        return objective, constraints

    result = run_dense_curvature_canary(
        joint,
        jnp.asarray([0.01, 0.01, 0.0], dtype=jnp.float64),
        trust_radius=1.0 / 64.0,
        hessian_batch_width=2,
    )

    assert bool(result.both_variants_usable)
    assert bool(result.exact_scaled_stationarity_improved)
    assert float(result.exact.scaled_stationarity_inf) < float(
        result.identity.scaled_stationarity_inf
    )
    assert float(result.exact.scaled_feasibility_inf) <= 1.0e-14


def test_fullspace_adapter_preserves_scaling_and_recomputes_endpoints(
    monkeypatch,
) -> None:
    anchor = jnp.asarray([0.2, -0.1, 0.0], dtype=jnp.float64)
    scaling = FullSpaceScaling(
        bootstrap_anchor=anchor,
        variable_scale=jnp.asarray([2.0, 0.5, 1.0], dtype=jnp.float64),
        constraint_inverse_scale=jnp.asarray([3.0], dtype=jnp.float64),
    )

    monkeypatch.setattr(
        fullspace_module,
        "fullspace_scaling_from_bootstrap",
        lambda _state, _problem: scaling,
    )

    def joint(
        optimizer_coordinates: jax.Array,
        _problem: object,
        active_scaling: FullSpaceScaling,
    ) -> tuple[jax.Array, jax.Array]:
        physical = (
            active_scaling.bootstrap_anchor
            + active_scaling.variable_scale * optimizer_coordinates
        )
        objective = 0.5 * jnp.vdot(physical, physical)
        constraints = jnp.asarray(
            [active_scaling.constraint_inverse_scale[0] * physical[2]]
        )
        return objective, constraints

    endpoint_calls: list[np.ndarray] = []

    def diagnostics(
        optimizer_coordinates: jax.Array,
        multipliers: jax.Array,
        _problem: object,
        active_scaling: FullSpaceScaling,
    ) -> CfsSqp1EndpointDiagnostics:
        physical = (
            active_scaling.bootstrap_anchor
            + active_scaling.variable_scale * optimizer_coordinates
        )
        endpoint_calls.append(np.asarray(physical))
        raw_constraints = jnp.asarray([physical[2]])
        raw_stationarity = physical.at[2].add(3.0 * multipliers[0])
        return CfsSqp1EndpointDiagnostics(
            physical_state=physical,
            physical_objective=0.5 * jnp.vdot(physical, physical),
            raw_constraints=raw_constraints,
            scaled_constraints=3.0 * raw_constraints,
            scaled_multipliers=multipliers,
            raw_multipliers=3.0 * multipliers,
            raw_stationarity_residual=raw_stationarity,
            raw_constraint_infinity_norm=jnp.linalg.norm(raw_constraints, ord=jnp.inf),
            scaled_constraint_infinity_norm=jnp.linalg.norm(
                3.0 * raw_constraints, ord=jnp.inf
            ),
            raw_kkt_stationarity_infinity_norm=jnp.linalg.norm(
                raw_stationarity, ord=jnp.inf
            ),
            all_finite=jnp.asarray(True),
        )

    monkeypatch.setattr(fullspace_module, "cfs_sqp1_joint_value_constraints", joint)
    monkeypatch.setattr(fullspace_module, "cfs_sqp1_endpoint_diagnostics", diagnostics)

    result = fullspace_module.run_fullspace_curvature_canary(
        object(),
        anchor,
        hessian_batch_width=2,
    )

    np.testing.assert_array_equal(result.scaling.bootstrap_anchor, anchor)
    assert len(endpoint_calls) == 3
    np.testing.assert_allclose(endpoint_calls[0], anchor)
    assert bool(result.all_finite)


def test_complete_generic_canary_lowers_and_compiles() -> None:
    curvature = jnp.diag(jnp.asarray([1.0, 3.0, 5.0], dtype=jnp.float64))

    def run(initial_coordinates: jax.Array):
        def joint(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            return 0.5 * x @ curvature @ x, jnp.reshape(x[2], (1,))

        return run_dense_curvature_canary(
            joint,
            initial_coordinates,
            hessian_batch_width=2,
        )

    executable = (
        jax.jit(run).lower(jnp.asarray([0.01, -0.02, 0.0], dtype=jnp.float64)).compile()
    )
    result = executable(jnp.asarray([0.01, -0.02, 0.0], dtype=jnp.float64))

    assert bool(result.all_finite)
    assert bool(result.both_variants_usable)
