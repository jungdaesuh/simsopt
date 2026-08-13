"""Closed-form checks for the reduced-Lagrangian Newton--CG step.

Every claim here is checked against the sphere problem's analytic Lagrangian,
not against the implementation.  For ``min 0.5 x' D x`` subject to
``x'x = 1`` the least-squares multiplier at a feasible point is
``lambda = -(x' D x) / 2`` and the Lagrangian Hessian is exactly
``D - (x' D x) I``, so the constraint curvature term is a known shift of the
objective's own curvature -- which is the whole quantity this route exists to
carry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.lagrangian_newton_cg import (
    TangentNewtonCgTermination,
    lagrangian_hessian_vector_product,
    least_squares_multipliers,
    solve_tangent_newton_cg,
)
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    factor_certified_gram_projector,
    materialize_certified_projection,
    project_with_certified_gram,
)
from simsopt_jax.geo.optimizers.projected_lbfgs import evaluate_projected_point

jax.config.update("jax_enable_x64", True)

_CURVATURES = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)


def _sphere_problem(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
    objective = 0.5 * jnp.sum(_CURVATURES * coordinates**2)
    return objective, jnp.reshape(coordinates @ coordinates - 1.0, (1,))


def _feasible_point() -> jax.Array:
    point = jnp.asarray([0.3, 0.4, 0.5, 0.5, 0.5], dtype=jnp.float64)
    return point / jnp.linalg.norm(point)


def _positive_curvature_point() -> jax.Array:
    """A feasible point whose REDUCED Lagrangian Hessian is positive definite.

    The reduced Hessian is ``P (D - (x' D x) I) P``, so definiteness depends on
    where on the sphere the point sits: at ``_feasible_point`` the tangent
    eigenvalues run -2.23 to 1.20 and a Newton solve is entitled to refuse to
    move, while near the minimizing axis they run 0.94 to 3.94.
    """

    point = jnp.asarray([1.0, 0.12, 0.08, 0.06, 0.05], dtype=jnp.float64)
    return point / jnp.linalg.norm(point)


def _analytic_multiplier(coordinates: jax.Array) -> float:
    return -0.5 * float(jnp.sum(_CURVATURES * coordinates**2))


def test_multipliers_match_the_analytic_least_squares_value() -> None:
    coordinates = _feasible_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)

    multipliers = least_squares_multipliers(point.projector, point.gradient)

    assert bool(multipliers.all_finite)
    np.testing.assert_allclose(
        float(multipliers.multipliers[0]), _analytic_multiplier(coordinates), rtol=1e-12
    )
    # The multipliers are exactly what makes the projected gradient the
    # Lagrangian gradient: g + A' lambda must equal P g.
    stationarity = (
        point.gradient
        + point.projector.constraint_jacobian.T @ multipliers.multipliers
    )
    np.testing.assert_allclose(
        np.asarray(stationarity), np.asarray(point.projected_gradient), atol=1e-14
    )


def test_hessian_vector_product_is_the_analytic_lagrangian_hessian() -> None:
    coordinates = _feasible_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    multipliers = least_squares_multipliers(point.projector, point.gradient)
    product = lagrangian_hessian_vector_product(
        _sphere_problem, coordinates, multipliers.multipliers
    )
    vector = jnp.asarray([0.1, -0.7, 0.3, 0.2, -0.5], dtype=jnp.float64)

    analytic = _CURVATURES * vector + 2.0 * multipliers.multipliers[0] * vector

    np.testing.assert_allclose(
        np.asarray(product(vector)), np.asarray(analytic), rtol=1e-12, atol=1e-14
    )


def test_hessian_vector_product_with_zero_multipliers_drops_constraint_curvature() -> (
    None
):
    coordinates = _feasible_point()
    vector = jnp.asarray([0.1, -0.7, 0.3, 0.2, -0.5], dtype=jnp.float64)

    objective_only = lagrangian_hessian_vector_product(
        _sphere_problem, coordinates, jnp.zeros((1,), dtype=jnp.float64)
    )(vector)

    np.testing.assert_allclose(
        np.asarray(objective_only), np.asarray(_CURVATURES * vector), rtol=1e-12
    )


def test_newton_cg_solves_the_reduced_newton_system() -> None:
    coordinates = _positive_curvature_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    multipliers = least_squares_multipliers(point.projector, point.gradient)
    product = lagrangian_hessian_vector_product(
        _sphere_problem, coordinates, multipliers.multipliers
    )

    step = solve_tangent_newton_cg(
        product,
        point.projector,
        point.projected_gradient,
        maximum_iterations=20,
        forcing_maximum=0.5,
    )

    assert bool(step.usable)
    assert int(step.termination) == int(TangentNewtonCgTermination.FORCING_SATISFIED)
    assert 1 <= int(step.iterations) <= 20
    # The direction stays in the tangent space and solves the projected system
    # to the forcing tolerance it reported.
    assert float(step.tangency_relative_residual) < 1e-12
    projection = materialize_certified_projection(point.projector)
    reduced_hessian = jnp.stack(
        [projection @ product(projection @ column) for column in jnp.eye(5)]
    ).T
    residual = reduced_hessian @ step.direction + point.projected_gradient
    assert float(jnp.linalg.norm(residual)) <= float(step.residual_target)
    assert float(step.predicted_reduction) > 0.0


def test_newton_cg_direction_is_a_descent_direction_on_the_true_objective() -> None:
    coordinates = _positive_curvature_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    multipliers = least_squares_multipliers(point.projector, point.gradient)

    step = solve_tangent_newton_cg(
        lagrangian_hessian_vector_product(
            _sphere_problem, coordinates, multipliers.multipliers
        ),
        point.projector,
        point.projected_gradient,
        maximum_iterations=20,
    )

    assert float(point.projected_gradient @ step.direction) < 0.0


def test_first_iteration_negative_curvature_returns_no_step() -> None:
    # At the largest-curvature axis the reduced Lagrangian Hessian is
    # D - 5I, whose tangent eigenvalues are -4 to -1, so the very first
    # conjugate-gradient iteration must refuse to move.
    coordinates = jnp.asarray([1e-8, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
    coordinates = coordinates / jnp.linalg.norm(coordinates)
    point = evaluate_projected_point(_sphere_problem, coordinates)
    multipliers = least_squares_multipliers(point.projector, point.gradient)

    step = solve_tangent_newton_cg(
        lagrangian_hessian_vector_product(
            _sphere_problem, coordinates, multipliers.multipliers
        ),
        point.projector,
        point.projected_gradient,
        maximum_iterations=20,
    )

    assert bool(step.negative_curvature)
    assert bool(step.negative_curvature_before_any_step)
    assert not bool(step.usable)
    assert float(step.direction_norm) == 0.0
    assert int(step.iterations) == 1


def test_a_preconditioner_does_not_move_the_solution() -> None:
    """The seed changes the conjugate directions, never the point they reach.

    On this problem the tangent space is four-dimensional and CG exhausts it
    either way, so the claim under test is the solution's invariance, not an
    iteration count.
    """

    coordinates = _positive_curvature_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    multipliers = least_squares_multipliers(point.projector, point.gradient)
    product = lagrangian_hessian_vector_product(
        _sphere_problem, coordinates, multipliers.multipliers
    )
    tight = 1.0e-13

    plain = solve_tangent_newton_cg(
        product,
        point.projector,
        point.projected_gradient,
        maximum_iterations=40,
        forcing_maximum=tight,
    )
    preconditioned = solve_tangent_newton_cg(
        product,
        point.projector,
        point.projected_gradient,
        maximum_iterations=40,
        forcing_maximum=tight,
        preconditioner=lambda vector: vector / _CURVATURES,
    )

    assert bool(plain.usable) and bool(preconditioned.usable)
    np.testing.assert_allclose(
        np.asarray(preconditioned.direction), np.asarray(plain.direction), atol=1e-10
    )


def test_a_preconditioner_leaving_the_tangent_space_is_projected_back() -> None:
    coordinates = _positive_curvature_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    multipliers = least_squares_multipliers(point.projector, point.gradient)

    step = solve_tangent_newton_cg(
        lagrangian_hessian_vector_product(
            _sphere_problem, coordinates, multipliers.multipliers
        ),
        point.projector,
        point.projected_gradient,
        maximum_iterations=40,
        forcing_maximum=1.0e-13,
        # A preconditioner deliberately built to push mass along the normal.
        preconditioner=lambda vector: vector + 3.0 * coordinates,
    )

    assert float(step.tangency_relative_residual) < 1e-11
    normal_component = float(
        jnp.linalg.norm(
            step.direction - project_with_certified_gram(
                point.projector, step.direction
            ).projected
        )
    )
    assert normal_component < 1e-11 * float(step.direction_norm)


def test_solver_rejects_an_empty_budget_and_a_degenerate_forcing() -> None:
    coordinates = _feasible_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    product = lagrangian_hessian_vector_product(
        _sphere_problem, coordinates, jnp.zeros((1,), dtype=jnp.float64)
    )

    for keywords in (
        {"maximum_iterations": 0},
        {"maximum_iterations": 5, "forcing_maximum": 0.0},
        {"maximum_iterations": 5, "forcing_maximum": 1.0},
    ):
        with pytest.raises(ValueError):
            solve_tangent_newton_cg(
                product, point.projector, point.projected_gradient, **keywords
            )


def test_solve_is_jittable_with_frozen_multipliers() -> None:
    coordinates = _feasible_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    multipliers = least_squares_multipliers(point.projector, point.gradient)

    @jax.jit
    def solve(values, frozen, projector, projected_gradient):
        return solve_tangent_newton_cg(
            lagrangian_hessian_vector_product(_sphere_problem, values, frozen),
            projector,
            projected_gradient,
            maximum_iterations=20,
        )

    jitted = solve(
        coordinates,
        multipliers.multipliers,
        point.projector,
        point.projected_gradient,
    )
    eager = solve_tangent_newton_cg(
        lagrangian_hessian_vector_product(
            _sphere_problem, coordinates, multipliers.multipliers
        ),
        point.projector,
        point.projected_gradient,
        maximum_iterations=20,
    )

    np.testing.assert_allclose(
        np.asarray(jitted.direction), np.asarray(eager.direction), rtol=1e-12
    )


def test_factored_projector_is_the_one_the_multipliers_reuse() -> None:
    # The multiplier solve must consume the caller's factorization rather than
    # build a second one, so an independently factored projector at the same
    # point has to give the identical answer.
    coordinates = _feasible_point()
    point = evaluate_projected_point(_sphere_problem, coordinates)
    rebuilt = factor_certified_gram_projector(point.projector.constraint_jacobian)

    np.testing.assert_allclose(
        np.asarray(least_squares_multipliers(rebuilt, point.gradient).multipliers),
        np.asarray(
            least_squares_multipliers(point.projector, point.gradient).multipliers
        ),
        rtol=1e-15,
    )
