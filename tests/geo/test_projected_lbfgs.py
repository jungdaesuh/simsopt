from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    ProjectedLbfgsOptions,
    ProjectedLbfgsStatus,
    evaluate_projected_point,
    retract_to_manifold,
    run_projected_lbfgs,
)

jax.config.update("jax_enable_x64", True)

# Minimizing 0.5 x' diag(1..5) x on the unit sphere has the unique minimum
# 0.5 at the first coordinate axis, which makes every claim below checkable
# against a closed form rather than against the implementation.
_CURVATURES = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
_SPHERE_MINIMUM = 0.5


def _sphere_problem(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
    objective = 0.5 * jnp.sum(_CURVATURES * coordinates**2)
    return objective, jnp.reshape(coordinates @ coordinates - 1.0, (1,))


def _feasible_start() -> jax.Array:
    start = jnp.asarray([0.3, 0.4, 0.5, 0.5, 0.5], dtype=jnp.float64)
    return start / jnp.linalg.norm(start)


def test_retraction_restores_feasibility_from_an_off_manifold_point() -> None:
    off_manifold = jnp.asarray([0.9, 0.6, 0.2, 0.1, 0.0], dtype=jnp.float64)
    assert float(abs(off_manifold @ off_manifold - 1.0)) > 1.0e-2

    retraction = retract_to_manifold(
        _sphere_problem,
        off_manifold,
        feasibility_tolerance=1.0e-12,
        maximum_corrections=8,
    )

    assert bool(retraction.feasible), "retraction left the point off the manifold"
    assert float(retraction.feasibility_inf) <= 1.0e-12
    assert 1 <= int(retraction.corrections) <= 8
    # The correction is normal to the manifold, so it only rescales the radius.
    direction = np.asarray(retraction.coordinates) / np.linalg.norm(
        np.asarray(retraction.coordinates)
    )
    np.testing.assert_allclose(
        direction,
        np.asarray(off_manifold) / np.linalg.norm(np.asarray(off_manifold)),
        atol=1.0e-12,
    )


def test_feasible_point_needs_no_correction() -> None:
    retraction = retract_to_manifold(
        _sphere_problem,
        _feasible_start(),
        feasibility_tolerance=1.0e-12,
        maximum_corrections=8,
    )

    assert int(retraction.corrections) == 0
    assert bool(retraction.feasible)


def test_projected_gradient_is_tangent_and_vanishes_at_the_minimizer() -> None:
    interior = evaluate_projected_point(_sphere_problem, _feasible_start())
    assert float(interior.projected_gradient_inf) > 1.0e-3

    minimizer = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    stationary = evaluate_projected_point(_sphere_problem, minimizer)
    assert float(stationary.projected_gradient_inf) <= 1.0e-12
    # Tangency: the projected gradient carries no component along the normal.
    assert (
        abs(float(interior.projected_gradient @ _feasible_start())) <= 1.0e-12
    )


def test_run_descends_monotonically_to_the_closed_form_minimum() -> None:
    banked: list[float] = []
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=60,
            memory=6,
            feasibility_tolerance=1.0e-12,
            maximum_step_norm=0.5,
        ),
        observer=lambda record: banked.append(record.objective),
    )

    assert run.status is ProjectedLbfgsStatus.ITERATION_LIMIT
    assert len(banked) == len(run.iterations)
    objectives = [record.objective for record in run.iterations]
    assert objectives == sorted(objectives, reverse=True), (
        "Armijo acceptance must make the objective sequence nonincreasing"
    )
    assert run.objective == min(objectives + [run.objective])
    assert abs(run.objective - _SPHERE_MINIMUM) <= 1.0e-10, (
        f"converged to {run.objective}, not the closed-form minimum "
        f"{_SPHERE_MINIMUM}"
    )
    assert run.projected_gradient_inf <= 1.0e-8


def test_run_holds_feasibility_at_every_accepted_iteration() -> None:
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=40, feasibility_tolerance=1.0e-12
        ),
    )

    for record in run.iterations:
        assert record.feasibility_inf <= 1.0e-12, (
            f"iteration {record.index} started at feasibility "
            f"{record.feasibility_inf}, above the enforced tolerance"
        )
        assert record.candidate_feasibility_inf <= 1.0e-12
    assert run.feasibility_inf <= 1.0e-12


def test_curvature_pairs_accumulate_and_bound_at_the_memory() -> None:
    memory = 4
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(maximum_iterations=30, memory=memory),
    )

    admitted = [record for record in run.iterations if record.pair_admitted]
    assert len(admitted) >= memory, "no curvature was ever admitted"
    assert all(record.curvature > 0.0 for record in admitted)
    assert max(record.stored_pairs for record in run.iterations) == memory
    assert run.stored_pairs == memory


def test_curvature_beats_steepest_descent_on_the_same_budget() -> None:
    budget = 25
    quasi_newton = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(maximum_iterations=budget, memory=8),
    )
    steepest = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(maximum_iterations=budget, memory=1),
    )

    quasi_newton_gap = quasi_newton.objective - _SPHERE_MINIMUM
    steepest_gap = steepest.objective - _SPHERE_MINIMUM
    assert quasi_newton_gap < steepest_gap, (
        "stored curvature must close the gap faster than a one-pair store: "
        f"{quasi_newton_gap} vs {steepest_gap}"
    )


def test_infeasible_start_fails_closed() -> None:
    run = run_projected_lbfgs(
        _sphere_problem,
        jnp.asarray([0.9, 0.6, 0.2, 0.1, 0.0], dtype=jnp.float64),
        options=ProjectedLbfgsOptions(maximum_iterations=5),
    )

    assert run.status is ProjectedLbfgsStatus.INFEASIBLE_START
    assert run.iterations == ()


def test_objective_target_stops_the_run() -> None:
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=200, objective_target=0.55
        ),
    )

    assert run.status is ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED
    assert run.objective <= 0.55
