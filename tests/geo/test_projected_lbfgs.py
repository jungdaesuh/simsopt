from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.dense_sqp import materialize_joint_vjp_rows
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    factor_certified_gram_projector,
)
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    ProjectedLbfgsOptions,
    ProjectedLbfgsStatus,
    evaluate_projected_point,
    evaluate_projected_point_with_projector,
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


def _shifted_sphere_problem(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The same problem with its objective shifted to run strictly negative."""

    objective, constraints = _sphere_problem(coordinates)
    return objective - 10.0, constraints


def _feasible_start() -> jax.Array:
    start = jnp.asarray([0.3, 0.4, 0.5, 0.5, 0.5], dtype=jnp.float64)
    return start / jnp.linalg.norm(start)


def _near_minimizer_start() -> jax.Array:
    """A start whose REDUCED Lagrangian Hessian is positive definite.

    The reduced Hessian is ``P (D - (x' D x) I) P``, which is indefinite over
    most of the sphere -- at ``_feasible_start`` its tangent eigenvalues run
    -2.23 to 1.20 -- and positive definite only near the minimizing axis.  A
    Newton claim is a claim about the definite regime; the indefinite one is
    what the negative-curvature fallback exists for.
    """

    start = jnp.asarray([1.0, 0.12, 0.08, 0.06, 0.05], dtype=jnp.float64)
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
    # Every carried pair kept admissible curvature on this manifold, so the
    # store the transported run applies is the whole store -- the run is
    # evidence about transport, not about a store that emptied itself.
    assert all(record.transport_masked_pairs == 0 for record in run.iterations)

    untransported = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=40,
            memory=6,
            feasibility_tolerance=1.0e-12,
        ),
    )

    # And the transport is not a relabelling: re-forming the pairs in the new
    # tangent space has to move the trajectory off the untransported one.
    assert objectives != [record.objective for record in untransported.iterations]


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
    # The reported conditioning must describe the tangent block of the solve,
    # not the Levenberg floor the normal space sits at exactly.
    hessian = np.asarray(jacobian).T @ np.asarray(jacobian)
    reduced = np.asarray(projection) @ hessian @ np.asarray(projection)
    reduced = 0.5 * (reduced + reduced.T)
    regularized = reduced + float(step.levenberg) * np.eye(reduced.shape[0])
    eigenvalues = np.linalg.eigvalsh(regularized)
    tangent_rank = round(float(np.trace(np.asarray(projection))))
    expected = eigenvalues[eigenvalues.size - tangent_rank] / eigenvalues[-1]
    assert float(step.reduced_hessian_reciprocal_condition) == pytest.approx(
        expected, rel=1.0e-9
    )
    assert float(step.reduced_hessian_reciprocal_condition) > (
        1.0e3 * float(step.levenberg) / eigenvalues[-1]
    )


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
    with pytest.raises(ValueError, match="objective_residuals"):
        run_projected_lbfgs(
            _sphere_problem,
            _feasible_start(),
            options=ProjectedLbfgsOptions(maximum_iterations=3, gauss_newton=True),
        )


def test_lagrangian_newton_run_reaches_the_closed_form_minimum() -> None:
    run = run_projected_lbfgs(
        _sphere_problem,
        _near_minimizer_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=40,
            lagrangian_newton=True,
            objective_target=_SPHERE_MINIMUM + 1.0e-12,
        ),
    )

    assert run.status is ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED
    # The point of a ball-free Newton step: unit scale is accepted as proposed.
    # A rescued iteration's scale belongs to the secant direction that placed
    # the step, so it is not evidence either way.
    assert all(
        iteration.step_scale == 1.0
        for iteration in run.iterations
        if iteration.lagrangian_newton_used and not iteration.lagrangian_newton_rescued
    )
    assert float(run.objective) <= _SPHERE_MINIMUM + 1.0e-12
    assert float(run.feasibility_inf) <= 1.0e-10
    assert any(iteration.lagrangian_newton_used for iteration in run.iterations)
    # Every Newton iteration must bank the curvature it explored.
    for iteration in run.iterations:
        if iteration.lagrangian_newton_used:
            assert len(iteration.newton_curvature_rayleigh_history) == (
                iteration.newton_cg_iterations
            )
            assert iteration.newton_cg_iterations >= 1
            assert math.isfinite(iteration.multiplier_norm)


def test_lagrangian_newton_beats_the_secant_store_on_the_same_budget() -> None:
    budget = 4
    secant = run_projected_lbfgs(
        _sphere_problem,
        _near_minimizer_start(),
        options=ProjectedLbfgsOptions(maximum_iterations=budget),
    )
    newton = run_projected_lbfgs(
        _sphere_problem,
        _near_minimizer_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=budget, lagrangian_newton=True
        ),
    )

    # Four Newton steps land on the closed-form minimum to rounding; four
    # secant steps are still three digits away from it.
    assert float(newton.objective) - _SPHERE_MINIMUM <= 1.0e-14
    assert float(secant.objective) - _SPHERE_MINIMUM > 1.0e-6


def test_negative_curvature_iteration_falls_back_to_the_store() -> None:
    # From this start the reduced Lagrangian Hessian is indefinite, so the
    # first Newton solve must refuse and the run must still descend.
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=3, lagrangian_newton=True
        ),
    )

    first = run.iterations[0]
    assert first.newton_cg_negative_curvature_before_any_step
    assert not first.lagrangian_newton_used
    assert float(run.objective) < float(run.iterations[0].objective)


def test_hybrid_schedule_opens_on_the_secant_step() -> None:
    window = 3
    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=12,
            lagrangian_newton=True,
            hybrid_plateau_window=window,
            # A factor of one arms the Newton phase only on a true plateau.
            hybrid_plateau_factor=1.0,
        ),
    )

    for iteration in run.iterations[:window]:
        assert iteration.newton_cg_iterations == 0
        assert not iteration.lagrangian_newton_used
    assert all(
        math.isnan(iteration.trailing_objective_factor)
        for iteration in run.iterations[:window]
    )
    assert any(
        math.isfinite(iteration.trailing_objective_factor)
        for iteration in run.iterations[window:]
    )


def test_hybrid_window_requires_the_newton_arm() -> None:
    with pytest.raises(ValueError, match="hybrid_plateau_window requires"):
        run_projected_lbfgs(
            _sphere_problem,
            _feasible_start(),
            options=ProjectedLbfgsOptions(hybrid_plateau_window=5),
        )


def test_direction_models_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="are exclusive"):
        run_projected_lbfgs(
            _sphere_problem,
            _feasible_start(),
            options=ProjectedLbfgsOptions(lagrangian_newton=True, dense_curvature=True),
        )


def test_transport_is_rejected_with_a_model_direction() -> None:
    with pytest.raises(ValueError, match="secant direction only"):
        run_projected_lbfgs(
            _sphere_problem,
            _feasible_start(),
            options=ProjectedLbfgsOptions(
                lagrangian_newton=True, vector_transport=True
            ),
        )


def test_objective_target_is_off_by_default() -> None:
    """A nonpositive objective must not read the default as a met target."""

    run = run_projected_lbfgs(
        _shifted_sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(maximum_iterations=3),
    )

    assert run.status is not ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED
    assert len(run.iterations) == 3


def test_carried_projector_evaluation_matches_the_fresh_one_at_its_own_point() -> None:
    """At the point its rows came from, the carried form must be the fresh one.

    The two paths reach the objective, the constraints and the gradient through
    different derivative programs -- a batched VJP over 1 + m outputs versus one
    scalar reverse pass -- so their agreement is what says the cheap path is the
    same evaluation and not an approximation of it.
    """

    coordinates = _feasible_start()
    fresh = evaluate_projected_point(_sphere_problem, coordinates)
    carried = evaluate_projected_point_with_projector(
        _sphere_problem, fresh.projector, coordinates
    )

    np.testing.assert_allclose(
        np.asarray(carried.gradient), np.asarray(fresh.gradient), rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(carried.projected_gradient),
        np.asarray(fresh.projected_gradient),
        rtol=0.0,
        atol=1e-14,
    )
    assert float(carried.objective) == float(fresh.objective)
    assert float(carried.feasibility_inf) == float(fresh.feasibility_inf)
    # The rows ARE this point's, so the measured true tangency has to agree
    # with the projector's own certificate rather than merely be small.
    assert float(carried.true_tangency_relative_residual) <= 1.0e-10
    assert bool(carried.all_finite)


def test_carried_projector_reports_the_drift_it_accumulates() -> None:
    """The measurement must react to distance, or it cannot gate the carry."""

    near = _feasible_start()
    projector = evaluate_projected_point(_sphere_problem, near).projector
    far = retract_to_manifold(
        _sphere_problem,
        near + jnp.asarray([0.2, -0.15, 0.1, 0.0, 0.0], dtype=jnp.float64),
        feasibility_tolerance=1.0e-12,
        maximum_corrections=8,
    ).coordinates

    here = evaluate_projected_point_with_projector(_sphere_problem, projector, near)
    there = evaluate_projected_point_with_projector(_sphere_problem, projector, far)

    assert float(there.true_tangency_relative_residual) > 1.0e-3
    assert float(there.true_tangency_relative_residual) > 1.0e6 * float(
        here.true_tangency_relative_residual
    )
    # The carried projector never touches the constraint values, so the point
    # it reports is still exactly on the manifold.
    assert float(there.feasibility_inf) <= 1.0e-12


def test_refresh_period_reaches_the_same_minimum_with_fewer_materializations() -> None:
    period = 4
    budget = 30
    carried = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=budget,
            feasibility_tolerance=1.0e-12,
            frozen_projector_line_search=True,
            projector_refresh_period=period,
        ),
    )

    assert carried.projector_materializations <= len(carried.iterations)
    assert abs(float(carried.objective) - _SPHERE_MINIMUM) <= 1.0e-6
    # Feasibility is judged against the TRUE constraints whatever the rows are.
    for iteration in carried.iterations:
        assert iteration.feasibility_inf <= 1.0e-12
        assert 0 <= iteration.projector_age < period
    assert any(iteration.projector_age > 0 for iteration in carried.iterations)


def test_tangency_tolerance_forces_the_carry_to_end() -> None:
    """A tolerance no drift can clear must materialize at every point."""

    run = run_projected_lbfgs(
        _sphere_problem,
        _feasible_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=8,
            feasibility_tolerance=1.0e-12,
            frozen_projector_line_search=True,
            projector_refresh_period=8,
            projector_tangency_tolerance=1.0e-300,
        ),
    )

    assert run.tangency_forced_refreshes == len(run.iterations)
    assert all(iteration.projector_age == 0 for iteration in run.iterations)
    assert all(iteration.projector_refreshed for iteration in run.iterations)


def test_tangent_fraction_gate_skips_the_newton_solve() -> None:
    """Below the threshold no curvature solve may be spent at all."""

    run = run_projected_lbfgs(
        _sphere_problem,
        _near_minimizer_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=6,
            lagrangian_newton=True,
            newton_tangent_fraction_threshold=0.999,
        ),
    )

    assert all(iteration.newton_gate_skipped for iteration in run.iterations)
    assert all(iteration.newton_cg_iterations == 0 for iteration in run.iterations)
    assert not any(iteration.lagrangian_newton_used for iteration in run.iterations)
    objectives = [iteration.objective for iteration in run.iterations]
    assert objectives == sorted(objectives, reverse=True)
    for iteration in run.iterations:
        assert 0.0 <= iteration.tangent_gradient_fraction <= 1.0 + 1.0e-12


def test_a_gate_that_admits_everything_leaves_the_newton_arm_alone() -> None:
    budget = 6
    ungated = run_projected_lbfgs(
        _sphere_problem,
        _near_minimizer_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=budget, lagrangian_newton=True
        ),
    )
    gated = run_projected_lbfgs(
        _sphere_problem,
        _near_minimizer_start(),
        options=ProjectedLbfgsOptions(
            maximum_iterations=budget,
            lagrangian_newton=True,
            newton_tangent_fraction_threshold=1.0e-12,
        ),
    )

    assert not any(iteration.newton_gate_skipped for iteration in gated.iterations)
    assert float(gated.objective) == float(ungated.objective)


def test_newton_gate_requires_the_newton_arm() -> None:
    with pytest.raises(ValueError, match="newton_tangent_fraction_threshold requires"):
        run_projected_lbfgs(
            _sphere_problem,
            _feasible_start(),
            options=ProjectedLbfgsOptions(newton_tangent_fraction_threshold=0.3),
        )


def test_refresh_period_must_be_positive() -> None:
    with pytest.raises(ValueError, match="projector_refresh_period must be positive"):
        run_projected_lbfgs(
            _sphere_problem,
            _feasible_start(),
            options=ProjectedLbfgsOptions(projector_refresh_period=0),
        )


def test_committing_the_start_point_is_numerically_inert() -> None:
    """The loop binds its start to one device; that must change no number.

    A jitted output is committed to the device it ran on and a caller's array
    need not be, and ``jax.jit`` compiles a second executable for the second
    of those it sees.  The loop therefore commits its start point once.  What
    that must never do is move the run, so the same start supplied both ways
    has to produce the identical trajectory.
    """

    uncommitted = _feasible_start()
    committed = jax.device_put(uncommitted, next(iter(uncommitted.devices())))
    options = ProjectedLbfgsOptions(maximum_iterations=6, feasibility_tolerance=1e-12)

    loose = run_projected_lbfgs(_sphere_problem, uncommitted, options=options)
    bound = run_projected_lbfgs(_sphere_problem, committed, options=options)

    assert [record.objective for record in loose.iterations] == [
        record.objective for record in bound.iterations
    ]
    np.testing.assert_array_equal(
        np.asarray(loose.coordinates), np.asarray(bound.coordinates)
    )
