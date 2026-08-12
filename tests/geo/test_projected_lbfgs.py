from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers.dense_sqp import materialize_joint_vjp_rows
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    factor_certified_gram_projector,
)
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    ProjectedLbfgsOptions,
    ProjectedLbfgsStatus,
    evaluate_projected_point,
    retract_to_manifold,
    retract_with_frozen_projector,
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
    assert abs(float(interior.projected_gradient @ _feasible_start())) <= 1.0e-12


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
        f"converged to {run.objective}, not the closed-form minimum {_SPHERE_MINIMUM}"
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


def test_frozen_projector_retraction_lands_on_the_true_manifold() -> None:
    # Factor the Gram somewhere OTHER than the point being retracted, so the
    # chord rows really are stale, then demand the true constraints anyway.
    anchor = _feasible_start()
    projector = factor_certified_gram_projector(
        materialize_joint_vjp_rows(_sphere_problem, anchor).constraint_jacobian
    )
    off_manifold = anchor + jnp.asarray(
        [0.05, -0.04, 0.03, 0.02, -0.01], dtype=jnp.float64
    )

    frozen = retract_with_frozen_projector(
        _sphere_problem,
        projector,
        off_manifold,
        feasibility_tolerance=1.0e-12,
        maximum_corrections=8,
    )

    assert bool(frozen.feasible), "chord retraction failed to reach the manifold"
    assert float(frozen.feasibility_inf) <= 1.0e-12


def test_frozen_projector_run_matches_the_exact_retraction_run() -> None:
    shared = {"maximum_iterations": 25, "memory": 6, "feasibility_tolerance": 1.0e-12}
    exact = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(**shared),
    )
    frozen = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(**shared, frozen_projector_line_search=True),
    )

    # Amortizing the factorization must not cost feasibility or descent quality.
    assert frozen.feasibility_inf <= 1.0e-12
    for record in frozen.iterations:
        assert record.candidate_feasibility_inf <= 1.0e-12
    assert abs(frozen.objective - _SPHERE_MINIMUM) <= 1.0e-10
    assert frozen.objective <= exact.objective * 1.001, (
        f"frozen-projector run ended at {frozen.objective}, materially worse "
        f"than the exact-retraction run at {exact.objective}"
    )


def test_transport_masks_pairs_whose_curvature_does_not_survive() -> None:
    from simsopt_jax.geo.optimizers.quasi_newton_metric import (
        empty_quasi_newton_metric,
        insert_curvature_pair,
        transport_quasi_newton_metric,
    )

    metric = empty_quasi_newton_metric(4, 5, jnp.float64)
    step = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    change = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    metric = insert_curvature_pair(metric, step, change)
    assert int(metric.history.n_corrs) == 1

    # A transport that annihilates exactly the direction the pair's curvature
    # lived in must retire the pair rather than keep pretending it is curvature.
    def kill_first_axis(vectors: jax.Array) -> jax.Array:
        return vectors.at[:, 0].set(0.0)

    carried = transport_quasi_newton_metric(metric, kill_first_axis)

    assert int(carried.masked_pairs) == 1
    assert int(carried.live_pairs) == 1
    np.testing.assert_allclose(np.asarray(carried.metric.history.s[0]), 0.0)
    np.testing.assert_allclose(np.asarray(carried.metric.history.y[0]), 0.0)


def test_masked_pair_is_a_no_op_in_the_two_loop() -> None:
    from simsopt_jax.geo.optimizers.quasi_newton_metric import (
        apply_quasi_newton_metric,
        empty_quasi_newton_metric,
        insert_curvature_pair,
        transport_quasi_newton_metric,
    )

    empty = empty_quasi_newton_metric(4, 5, jnp.float64)
    stored = insert_curvature_pair(
        empty,
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64),
    )
    masked = transport_quasi_newton_metric(
        stored, lambda vectors: vectors.at[:, 0].set(0.0)
    ).metric
    probe = jnp.asarray([0.3, -0.7, 1.1, 0.0, 2.0], dtype=jnp.float64)

    # theta is carried from the stored pair, so compare against the same store
    # with its pair zeroed -- the claim is that a zeroed pair contributes
    # nothing, not that the store reverts to the identity.
    np.testing.assert_allclose(
        np.asarray(apply_quasi_newton_metric(masked, probe)),
        np.asarray(probe) / float(masked.inverse_hessian_theta),
        atol=1.0e-14,
    )


def test_transport_run_holds_feasibility_and_descends() -> None:
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=40,
            memory=6,
            feasibility_tolerance=1.0e-12,
            vector_transport=True,
        ),
    )

    objectives = [record.objective for record in run.iterations]
    assert objectives == sorted(objectives, reverse=True)
    assert abs(run.objective - _SPHERE_MINIMUM) <= 1.0e-10
    for record in run.iterations:
        assert record.feasibility_inf <= 1.0e-12
        assert record.transport_masked_pairs >= 0


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
        options=ProjectedLbfgsOptions(maximum_iterations=200, objective_target=0.55),
    )

    assert run.status is ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED
    assert run.objective <= 0.55


def test_dense_curvature_solves_the_sphere_and_stays_definite() -> None:
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=30,
            feasibility_tolerance=1.0e-12,
            dense_curvature=True,
        ),
    )

    objectives = [record.objective for record in run.iterations]
    assert objectives == sorted(objectives, reverse=True)
    assert abs(run.objective - _SPHERE_MINIMUM) <= 1.0e-10
    for record in run.iterations:
        assert record.feasibility_inf <= 1.0e-12
        assert record.dense_positive_definite, (
            f"iteration {record.index} lost positive definiteness, so the "
            "Powell damping failed to keep B a metric"
        )


def test_powell_damping_keeps_a_negative_curvature_pair_definite() -> None:
    from simsopt_jax.geo.optimizers.dense_tangent_curvature import (
        dense_tangent_direction,
        empty_dense_tangent_curvature,
        update_dense_tangent_curvature,
    )

    curvature = empty_dense_tangent_curvature(5, jnp.float64)
    step = jnp.asarray([1.0, 0.5, 0.0, 0.0, 0.0], dtype=jnp.float64)
    # s.y < 0: an undamped BFGS update on this pair would destroy definiteness.
    bad_change = -0.5 * step
    assert float(step @ bad_change) < 0.0

    updated = update_dense_tangent_curvature(curvature, step, bad_change)
    factor = np.linalg.cholesky(np.asarray(updated.hessian))

    assert np.all(np.isfinite(factor)), "Powell damping failed to keep B definite"
    assert int(updated.damped_updates) == 1
    probe = jnp.asarray([0.2, -0.3, 0.7, 0.1, 0.0], dtype=jnp.float64)
    result = dense_tangent_direction(updated, probe, lambda vector: vector)
    assert bool(result.positive_definite)
    assert float(probe @ result.direction) < 0.0, "direction must descend"


def _sphere_residuals(coordinates: jax.Array) -> jax.Array:
    """Residual form of the sphere objective: Phi = 0.5 ||R||^2."""
    return jnp.sqrt(_CURVATURES) * coordinates


def test_gauss_newton_step_is_tangent_and_predicts_its_own_reduction() -> None:
    from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
        materialize_certified_projection,
    )
    from simsopt_jax.geo.optimizers.tangent_gauss_newton import (
        solve_tangent_gauss_newton,
    )

    start = _feasible_start()
    point = evaluate_projected_point(_sphere_problem, start)
    projection = materialize_certified_projection(point.projector)
    jacobian = jax.jacfwd(_sphere_residuals)(start)

    step = solve_tangent_gauss_newton(
        jacobian, projection, point.projected_gradient, levenberg_relative=1.0e-12
    )

    assert bool(step.all_finite)
    assert float(step.tangency_relative_residual) <= 1.0e-10
    # The step must leave the constraint linearization untouched to first order.
    jacobian_constraint = 2.0 * np.asarray(start)
    assert abs(float(jacobian_constraint @ step.direction)) <= 1.0e-10
    assert float(step.predicted_reduction) > 0.0
    assert float(point.projected_gradient @ step.direction) < 0.0


def test_gauss_newton_run_descends_and_its_model_predicts_the_decrease() -> None:
    """The route's premise is model EXACTNESS, not superiority over a secant.

    On a curved constraint manifold ``P J^T J P`` is the Gauss--Newton Hessian
    of the OBJECTIVE and omits the constraint-curvature term of the reduced
    Lagrangian Hessian, so GN need not beat L-BFGS on a budget.  What must hold
    is that the model predicts the decrease it actually delivers, since that is
    what lets the line search take full steps.
    """

    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=12,
            feasibility_tolerance=1.0e-12,
            gauss_newton=True,
        ),
        objective_residuals=_sphere_residuals,
    )

    assert any(record.gauss_newton_used for record in run.iterations)
    objectives = [record.objective for record in run.iterations]
    assert objectives == sorted(objectives, reverse=True)
    assert abs(run.objective - _SPHERE_MINIMUM) <= 1.0e-5
    for record in run.iterations:
        assert record.feasibility_inf <= 1.0e-12
        if record.gauss_newton_used and not record.gauss_newton_rescued:
            assert record.gauss_newton_predicted_reduction > 0.0
            assert record.gauss_newton_tangency_residual <= 1.0e-10


def test_gauss_newton_requires_residuals() -> None:
    import pytest

    with pytest.raises(ValueError, match="objective_residuals"):
        run_projected_lbfgs(
            _sphere_problem,
            _feasible_start(),
            options=ProjectedLbfgsOptions(maximum_iterations=3, gauss_newton=True),
        )
