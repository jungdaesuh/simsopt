"""Projected L-BFGS on an equality-constraint manifold.

Curvature stepping on the true objective, with the constraints enforced by
projection and retraction rather than by penalties: the objective gradient is
projected into the tangent space of the linearized equalities, the
limited-memory inverse Hessian turns that tangent gradient into a search
direction, and every trial point is pulled back onto the manifold by certified
minimum-norm Newton corrections before its objective is compared.

The module owns one piece of knowledge: how a projected curvature step is
proposed, retracted, and accepted.  The certified tangent projector and the
certified Newton correction live in ``projected_hvp_trust_region``, the
correction store lives in ``quasi_newton_metric``, the reduced-Lagrangian
Newton direction lives in ``lagrangian_newton_cg``, and the exact derivative
rows come from ``dense_sqp``.

Which curvature model proposes the direction is an option, and the models can
be scheduled against each other: the limited-memory step is cheap but settles
into a linear rate, while a Lagrangian Newton step costs several curvature
products and earns that back only where the linear rate has stalled.
``hybrid_plateau_window`` measures the realized rate and switches on it.

The constraint Jacobian's materialization dominates everything else here by
two orders of magnitude -- one primal traversal per row against one for the
whole objective gradient -- so the factored Gram it yields is treated as an
asset with a lifetime rather than as a per-point quantity.
``frozen_projector_line_search`` spends it across one line search;
``projector_refresh_period`` spends it across several ITERATIONS, with the
intervening points evaluated for value, constraints and gradient only.  What
that buys is speed and what it costs is exactness of the tangent projection --
never feasibility, because the retraction's verdict is taken against the true
constraints in both cases.  Each carried point measures its own true tangency
and hands the carry back when it has drifted, and a line search that can place
nothing against carried rows refreshes and retries instead of collapsing.

``newton_tangent_fraction_threshold`` decides on the same evidence whether the
Newton solve is worth running at all: the tangent fraction ``||P g|| / ||g||``
is free once the point is evaluated, and it separates the neighbourhoods where
a reduced model earns its cost from the ones where it does not.

Curvature pairs are formed from the realized on-manifold displacement ``s``
and the projected-gradient change ``y``.  Those two vectors are only
comparable inside one tangent space, so ``vector_transport`` decides how the
manifold's rotation is handled: without it they are differenced across two
tangent spaces, which is exact only in the small-step limit; with it both are
formed in the new point's tangent space and the stored pairs are carried into
the current one before each apply, with pairs whose curvature does not survive
the carry masked out.  The curvature admission guard keeps the resulting
operator positive definite either way.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dense_sqp import JointValueConstraints, materialize_joint_vjp_rows
from .dense_tangent_curvature import (
    dense_tangent_direction,
    empty_dense_tangent_curvature,
    update_dense_tangent_curvature,
)
from .lagrangian_newton_cg import (
    TangentNewtonCgStep,
    lagrangian_hessian_vector_product,
    least_squares_multipliers,
    solve_tangent_newton_cg,
)
from .projected_hvp_trust_region import (
    CertifiedGramProjector,
    certified_correction_with_projector,
    certified_minimum_norm_correction,
    factor_certified_gram_projector,
    materialize_certified_projection,
    project_with_certified_gram,
)
from .quasi_newton_metric import (
    QuasiNewtonMetric,
    apply_quasi_newton_metric,
    curvature_pair_admissible,
    empty_quasi_newton_metric,
    insert_curvature_pair,
    transport_quasi_newton_metric,
    valid_pair_count,
)
from .tangent_gauss_newton import (
    materialize_residual_jacobian,
    solve_tangent_gauss_newton,
)


class ProjectedLbfgsStatus(IntEnum):
    """Terminal disposition of one bounded projected L-BFGS run."""

    RUNNING = 0
    ITERATION_LIMIT = 1
    OBJECTIVE_TARGET_REACHED = 2
    LINE_SEARCH_COLLAPSE = 3
    NONFINITE_STATE = 4
    NON_DESCENT_DIRECTION = 5
    INFEASIBLE_START = 6


@dataclass(frozen=True, slots=True)
class ProjectedLbfgsOptions:
    """Immutable numerical policy for one projected L-BFGS run.

    ``feasibility_tolerance`` is stated in the units of the supplied constraint
    function, and ``maximum_step_norm`` in the units of the supplied
    coordinates.
    """

    maximum_iterations: int = 50
    memory: int = 10
    feasibility_tolerance: float = 1.0e-10
    maximum_retraction_corrections: int = 8
    maximum_step_norm: float = 0.25
    armijo_coefficient: float = 1.0e-4
    backtracking_factor: float = 0.5
    maximum_line_search_trials: int = 24
    minimum_step_scale: float = 1.0e-6
    objective_target: float | None = None
    """Objective at or below which the run stops and reports success.

    ``None`` runs no target check at all.  A numeric default would be a target
    every nonpositive objective meets at the start point, which is a success
    status handed to a run that has taken no step.
    """

    gauss_newton: bool = False
    """Take reduced Gauss--Newton steps, with the pair store as rescue.

    Requires ``objective_residuals``.  For a small-residual objective the GN
    model is locally near-exact, so the step is taken at full length and the
    Armijo line search -- not a trust region -- decides how far along it to go.
    When the GN step cannot be accepted above ``model_step_rescue_scale`` the
    iteration falls back to the limited-memory direction.
    """

    gauss_newton_levenberg_relative: float = 1.0e-12

    model_step_rescue_scale: float = 1.0e-2
    """Scale below which a full-length model step is abandoned for the store.

    A Gauss--Newton or Newton direction is a MODEL-scale step: either the model
    describes this neighbourhood, in which case a step near unit scale is
    accepted, or it does not, in which case backtracking it into the ground
    only spends retractions to arrive somewhere the secant direction would have
    reached sooner.
    """

    lagrangian_newton: bool = False
    """Take ball-free reduced-Lagrangian Newton--CG steps, pair store as rescue.

    The curvature carried is ``P (grad2 Phi + sum_i lambda_i grad2 c_i) P`` at
    the least-squares multipliers, applied matrix-free by forward-over-reverse.
    That constraint-curvature term is exactly what an objective-only model
    omits, and when most of the gradient lies in the constraint row space the
    multipliers are large enough for it to dominate at Newton scale.  Like the
    Gauss--Newton arm the step is tried at full length and rescued by the pair
    store below ``model_step_rescue_scale``.
    """

    newton_cg_maximum_iterations: int = 40
    newton_cg_forcing_maximum: float = 0.5

    hybrid_plateau_window: int = 0
    """Trailing iterations whose decrease rate arms the Newton phase.

    Zero runs one direction model for the whole run.  Positive runs the
    limited-memory step -- several times the cheaper one -- until its geometric
    per-iteration decrease factor over this window stops clearing
    ``hybrid_plateau_factor``, then switches to Newton steps, and switches back
    for another window whenever a Newton step cannot be accepted.  Requires
    ``lagrangian_newton``.
    """

    hybrid_plateau_factor: float = 0.995

    dense_curvature: bool = False
    """Carry a dense tangent-space Hessian approximation instead of pairs.

    At this dimension a dense Powell-damped BFGS model costs microseconds per
    iteration against a Jacobian materialization, and it is the curvature model
    a native dense-BFGS solve uses.  The direction comes from the dense model
    alone; the ``memory`` pair store is still fed, at a cost of one rank-two
    update per iteration, so a run can be compared against the secant arm on
    its own banked pairs.
    """

    vector_transport: bool = False
    """Carry curvature pairs between tangent spaces by projection.

    Without this, ``s`` and ``y`` are differences of quantities living in the
    tangent spaces of two different iterates, so a rotating manifold injects
    curvature that is an artifact of the rotation.  With it, both are formed in
    the NEW point's tangent space, and the stored pairs are re-projected into
    the current one before each two-loop apply, with pairs whose curvature does
    not survive the carry masked out.

    The transport reaches the SECANT direction only.  A model arm applies the
    store untransported -- as a Newton preconditioner, or as the rescue after a
    refused model step -- so pairing transport with one would write the store in
    one convention and read it in another; the combination is rejected.
    """

    frozen_projector_line_search: bool = False
    """Retract line-search trials against the accepted point's factored Gram.

    Amortizes the Jacobian materialization -- by far the dominant per-iteration
    cost -- over the whole line search, leaving one materialization per accepted
    iteration.  Feasibility is still certified against the true constraints, so
    a trial the chord iteration cannot land is rejected and the step halves.
    """

    projector_refresh_period: int = 1
    """Accepted iterations one materialized constraint Jacobian serves.

    One leaves the Jacobian materialized at every accepted point.  A larger
    period carries the factored Gram ACROSS iterations: the intervening points
    are evaluated for their objective, constraints and objective gradient only
    -- two orders of magnitude cheaper than the row materialization -- and the
    tangent projection and the multipliers spend the carried factor.  The
    retraction spends it as well only under ``frozen_projector_line_search``;
    the exact retraction this option leaves in place materializes its own rows
    at every correction whatever the period is.  The projection is then
    inexact, so every stale point
    measures its own TRUE tangency with one constraint JVP and refreshes when
    ``projector_tangency_tolerance`` is exceeded; a line search that can place
    no step against carried rows refreshes and retries rather than collapsing.
    Feasibility is unaffected either way: the retraction's verdict is taken
    against the true constraints, so no stale factor can admit an off-manifold
    point -- it can only cost trials.
    """

    projector_tangency_tolerance: float = 0.0
    """Measured true tangency above which a carried projector is refreshed.

    Zero carries the projector for the full period regardless.  The measured
    quantity is ``||A d||_inf / ||d||_inf`` at the CURRENT point for the
    projected gradient ``d``, which is zero to rounding when the rows were
    materialized here and grows with the distance travelled since.
    """

    newton_tangent_fraction_threshold: float = 0.0
    """Tangent gradient fraction below which the Newton solve is skipped.

    Zero runs the scheduled Newton solve unconditionally.  Positive compares
    ``||P g|| / ||g||`` -- already in hand from the point evaluation, at the
    cost of no additional work -- against the threshold before the solve, and
    below it takes the limited-memory direction directly.  A gradient that lies
    almost entirely in the constraint row space leaves a tangent component too
    small for the reduced model to describe, and the banked telemetry separates
    the two populations cleanly: the Newton directions that were accepted sat
    at fraction 0.85, the ones the line search refused at 0.15.
    """


_DEFAULT_OPTIONS = ProjectedLbfgsOptions()


class ProjectedPoint(NamedTuple):
    """One on-manifold point with its tangent gradient and projector evidence.

    ``projector`` holds the factored Gram of the constraint Jacobian the point
    was evaluated WITH, so a caller can spend the factorization again -- on a
    line search's trial retractions, for instance -- without materializing the
    Jacobian.  Those rows are this point's own only when the evaluation
    materialized them here; under ``projector_refresh_period`` they belong to
    an earlier iterate.

    ``tangency_relative_residual`` certifies the projection against the rows
    the projector CARRIES; ``true_tangency_relative_residual`` certifies it
    against the constraint Jacobian at these coordinates.  The two coincide
    when the rows were materialized here and separate when a carried projector
    is spent at a later point.
    """

    projector: CertifiedGramProjector
    objective: jax.Array
    gradient: jax.Array
    constraints: jax.Array
    feasibility_inf: jax.Array
    projected_gradient: jax.Array
    projected_gradient_inf: jax.Array
    projected_gradient_norm: jax.Array
    gradient_norm: jax.Array
    tangency_relative_residual: jax.Array
    true_tangency_relative_residual: jax.Array
    gram_reciprocal_condition: jax.Array
    solve_forward_error_bound: jax.Array
    all_finite: jax.Array


class ManifoldRetraction(NamedTuple):
    """One pullback onto the manifold and the evidence that it landed there."""

    coordinates: jax.Array
    objective: jax.Array
    constraints: jax.Array
    feasibility_inf: jax.Array
    corrections: jax.Array
    correction_path_norm: jax.Array
    feasible: jax.Array
    solve_relative_residual: jax.Array
    solve_forward_error_bound: jax.Array
    all_finite: jax.Array


class _RetractionState(NamedTuple):
    coordinates: jax.Array
    correction_path_norm: jax.Array
    corrections: jax.Array
    solve_relative_residual: jax.Array
    solve_forward_error_bound: jax.Array
    all_finite: jax.Array


class ProjectedLbfgsIteration(NamedTuple):
    """One iteration's accepted coordinates and host-side scalar evidence.

    ``coordinates`` are the iterate this record opens at, so that callers can
    take whatever projection of the trajectory they track — a distance to a
    reference state, a per-block norm — without the optimizer knowing about it.

    ``retraction_corrections`` belongs to the ACCEPTED trial alone; the
    corrections the rejected trials spent are inside ``retraction_seconds`` and
    are counted nowhere else.  ``line_search_trials`` is the whole search.
    """

    index: int
    coordinates: jax.Array
    objective: float
    projected_gradient_inf: float
    projected_gradient_norm: float
    gradient_norm: float
    feasibility_inf: float
    tangency_relative_residual: float
    true_tangency_relative_residual: float
    projector_age: int
    projector_refreshed: bool
    tangent_gradient_fraction: float
    newton_gate_skipped: bool
    gram_reciprocal_condition: float
    direction_norm: float
    directional_derivative: float
    step_scale: float
    step_norm: float
    step_norm_capped: bool
    line_search_trials: int
    rejected_for_feasibility: int
    retraction_corrections: int
    candidate_objective: float
    candidate_feasibility_inf: float
    curvature: float
    pair_admitted: bool
    stored_pairs: int
    transport_masked_pairs: int
    dense_damped_updates: int
    dense_positive_definite: bool
    gauss_newton_used: bool
    gauss_newton_rescued: bool
    gauss_newton_predicted_reduction: float
    gauss_newton_reciprocal_condition: float
    gauss_newton_tangency_residual: float
    gauss_newton_direction_norm: float
    lagrangian_newton_used: bool
    lagrangian_newton_rescued: bool
    newton_cg_iterations: int
    newton_cg_termination: int
    newton_cg_negative_curvature: bool
    newton_cg_negative_curvature_before_any_step: bool
    newton_cg_forcing_eta: float
    newton_cg_initial_residual_norm: float
    newton_cg_final_residual_norm: float
    newton_curvature: float
    newton_normalized_curvature: float
    newton_curvature_rayleigh_history: tuple[float, ...]
    newton_direction_norm: float
    newton_predicted_reduction: float
    multiplier_norm: float
    multiplier_forward_error_bound: float
    trailing_objective_factor: float
    model_exactness_ratio: float
    direction_seconds: float
    line_search_seconds: float
    retraction_seconds: float
    point_evaluation_seconds: float
    pair_update_seconds: float


class _AbandonedAttempt(NamedTuple):
    """What an iteration attempt spent before carried rows sent it back.

    A line search that placed nothing against a carried projector is retried
    against rows materialized here, and the retry opens the counters again.
    Carrying what the abandoned attempt spent -- rather than dropping it -- is
    what keeps the per-phase seconds summing to the solve time and the trial
    counts summing to the retractions actually performed.
    """

    direction_seconds: float = 0.0
    line_search_seconds: float = 0.0
    retraction_seconds: float = 0.0
    point_evaluation_seconds: float = 0.0
    trials: int = 0
    rejected_for_feasibility: int = 0


_NO_ABANDONED_ATTEMPT = _AbandonedAttempt()


class ProjectedLbfgsRun(NamedTuple):
    """Terminal state of one run plus every banked iteration record.

    ``projector_materializations`` counts the constraint-Jacobian rows the LOOP
    materialized -- one per point evaluation that did not spend a carried
    factor.  The exact retraction materializes its own rows per correction and
    those are not counted here; ``retraction_corrections`` on each record is
    what reports them.
    """

    status: ProjectedLbfgsStatus
    coordinates: jax.Array
    objective: float
    feasibility_inf: float
    projected_gradient_inf: float
    stored_pairs: int
    iterations: tuple[ProjectedLbfgsIteration, ...]
    compile_seconds: float
    solve_seconds: float
    projector_materializations: int
    tangency_forced_refreshes: int
    line_search_forced_refreshes: int


def evaluate_projected_point(
    joint_value_constraints: JointValueConstraints,
    coordinates: jax.Array,
) -> ProjectedPoint:
    """Evaluate the objective and its tangent gradient at one point.

    The tangent gradient is the objective gradient with its component in the
    row space of the constraint Jacobian removed, which is the projected
    stationarity the least-squares multipliers would leave behind.
    """

    rows = materialize_joint_vjp_rows(joint_value_constraints, coordinates)
    projector = factor_certified_gram_projector(rows.constraint_jacobian)
    projection = project_with_certified_gram(projector, rows.objective_gradient)
    projected_gradient = projection.projected
    return ProjectedPoint(
        projector=projector,
        objective=rows.objective,
        gradient=rows.objective_gradient,
        constraints=rows.constraints,
        feasibility_inf=jnp.linalg.norm(rows.constraints, ord=jnp.inf),
        projected_gradient=projected_gradient,
        projected_gradient_inf=jnp.linalg.norm(projected_gradient, ord=jnp.inf),
        projected_gradient_norm=jnp.linalg.norm(projected_gradient),
        gradient_norm=jnp.linalg.norm(rows.objective_gradient),
        tangency_relative_residual=projection.tangency_relative_residual,
        # The rows were materialized HERE, so the projector's own certificate
        # is already the true one; measuring it again would buy nothing.
        true_tangency_relative_residual=projection.tangency_relative_residual,
        gram_reciprocal_condition=projector.reciprocal_condition,
        solve_forward_error_bound=projection.solve_forward_error_bound,
        all_finite=(
            jnp.all(jnp.isfinite(coordinates))
            & jnp.isfinite(rows.objective)
            & jnp.all(jnp.isfinite(rows.constraints))
            & jnp.all(jnp.isfinite(rows.objective_gradient))
            & projection.all_finite
        ),
    )


def evaluate_projected_point_with_projector(
    joint_value_constraints: JointValueConstraints,
    projector: CertifiedGramProjector,
    coordinates: jax.Array,
) -> ProjectedPoint:
    """Evaluate one point against an already factored constraint Jacobian.

    This is the carried-projector form of :func:`evaluate_projected_point`.
    The objective, the constraints and the objective gradient are exact here --
    one primal traversal and one reverse pass, against the 256 the row
    materialization spends -- and only the tangent projection is inexact,
    because the rows it removes are the manifold's at an earlier point.

    That inexactness is measured rather than assumed: one constraint JVP
    through the projected gradient gives the tangency residual the step will
    actually be judged by, in the same relative form the fresh evaluation
    certifies, so a caller can decide when the carry has to end.
    """

    (objective, constraints), gradient = jax.value_and_grad(
        joint_value_constraints, has_aux=True
    )(coordinates)
    projection = project_with_certified_gram(projector, gradient)
    projected_gradient = projection.projected
    true_rows_action = jax.jvp(
        lambda values: joint_value_constraints(values)[1],
        (coordinates,),
        (projected_gradient,),
    )[1]
    tiny = jnp.asarray(jnp.finfo(coordinates.dtype).tiny, dtype=coordinates.dtype)
    true_tangency_relative_residual = jnp.linalg.norm(
        true_rows_action, ord=jnp.inf
    ) / jnp.maximum(tiny, jnp.linalg.norm(projected_gradient, ord=jnp.inf))
    return ProjectedPoint(
        projector=projector,
        objective=objective,
        gradient=gradient,
        constraints=constraints,
        feasibility_inf=jnp.linalg.norm(constraints, ord=jnp.inf),
        projected_gradient=projected_gradient,
        projected_gradient_inf=jnp.linalg.norm(projected_gradient, ord=jnp.inf),
        projected_gradient_norm=jnp.linalg.norm(projected_gradient),
        gradient_norm=jnp.linalg.norm(gradient),
        tangency_relative_residual=projection.tangency_relative_residual,
        true_tangency_relative_residual=true_tangency_relative_residual,
        gram_reciprocal_condition=projector.reciprocal_condition,
        solve_forward_error_bound=projection.solve_forward_error_bound,
        all_finite=(
            jnp.all(jnp.isfinite(coordinates))
            & jnp.isfinite(objective)
            & jnp.all(jnp.isfinite(constraints))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.isfinite(true_tangency_relative_residual)
            & projection.all_finite
        ),
    )


def retract_to_manifold(
    joint_value_constraints: JointValueConstraints,
    coordinates: jax.Array,
    *,
    feasibility_tolerance: float,
    maximum_corrections: int,
) -> ManifoldRetraction:
    """Pull one point back onto the equality manifold, then evaluate it there.

    Each correction is the certified minimum-norm solution of the linearized
    equalities, so the pullback is normal to the manifold and leaves the
    incoming tangential displacement intact.  Corrections stop as soon as the
    constraint infinity norm is within ``feasibility_tolerance``; ``feasible``
    reports whether that was reached inside the budget.
    """

    if maximum_corrections < 1:
        raise ValueError("maximum_corrections must be positive")

    dtype = coordinates.dtype

    def correct(state: _RetractionState) -> _RetractionState:
        rows = materialize_joint_vjp_rows(joint_value_constraints, state.coordinates)
        correction = certified_minimum_norm_correction(
            rows.constraint_jacobian, rows.constraints
        )
        return _RetractionState(
            coordinates=state.coordinates + correction.correction,
            correction_path_norm=state.correction_path_norm
            + jnp.linalg.norm(correction.correction),
            corrections=state.corrections + jnp.asarray(1, dtype=jnp.int32),
            solve_relative_residual=jnp.maximum(
                state.solve_relative_residual, correction.relative_residual
            ),
            solve_forward_error_bound=jnp.maximum(
                state.solve_forward_error_bound, correction.forward_error_bound
            ),
            all_finite=state.all_finite & correction.all_finite,
        )

    def extend(_iteration: jax.Array, state: _RetractionState) -> _RetractionState:
        _, constraints = joint_value_constraints(state.coordinates)
        feasibility = jnp.linalg.norm(constraints, ord=jnp.inf)
        required = (
            state.all_finite
            & jnp.isfinite(feasibility)
            & (feasibility > feasibility_tolerance)
        )
        return jax.lax.cond(required, correct, lambda active: active, state)

    retracted = jax.lax.fori_loop(
        0,
        maximum_corrections,
        extend,
        _RetractionState(
            coordinates=coordinates,
            correction_path_norm=jnp.zeros((), dtype=dtype),
            corrections=jnp.asarray(0, dtype=jnp.int32),
            solve_relative_residual=jnp.zeros((), dtype=dtype),
            solve_forward_error_bound=jnp.zeros((), dtype=dtype),
            all_finite=jnp.asarray(True),
        ),
    )
    return _finish_retraction(joint_value_constraints, retracted, feasibility_tolerance)


def _finish_retraction(
    joint_value_constraints: JointValueConstraints,
    retracted: _RetractionState,
    feasibility_tolerance: float,
) -> ManifoldRetraction:
    """Evaluate the landed point and certify that it reached the manifold.

    The verdict is always taken against the TRUE constraints, so a chord
    retraction is held to the same standard as an exact one.
    """

    objective, constraints = joint_value_constraints(retracted.coordinates)
    feasibility_inf = jnp.linalg.norm(constraints, ord=jnp.inf)
    all_finite = (
        retracted.all_finite
        & jnp.all(jnp.isfinite(retracted.coordinates))
        & jnp.isfinite(objective)
        & jnp.all(jnp.isfinite(constraints))
    )
    return ManifoldRetraction(
        coordinates=retracted.coordinates,
        objective=objective,
        constraints=constraints,
        feasibility_inf=feasibility_inf,
        corrections=retracted.corrections,
        correction_path_norm=retracted.correction_path_norm,
        feasible=all_finite & (feasibility_inf <= feasibility_tolerance),
        solve_relative_residual=retracted.solve_relative_residual,
        solve_forward_error_bound=retracted.solve_forward_error_bound,
        all_finite=all_finite,
    )


def retract_with_frozen_projector(
    joint_value_constraints: JointValueConstraints,
    projector: CertifiedGramProjector,
    coordinates: jax.Array,
    *,
    feasibility_tolerance: float,
    maximum_corrections: int,
) -> ManifoldRetraction:
    """Pull a point back onto the manifold reusing an already factored Jacobian.

    This is the chord form of :func:`retract_to_manifold`: the corrections are
    normal to the frozen rows in ``projector`` instead of to the manifold at the
    current point, which costs one primal evaluation per correction rather than
    a full Jacobian materialization.  The frozen rows change only the RATE and
    the direction of approach -- a converged point still satisfies the true
    constraints to ``feasibility_tolerance``, and ``feasible`` reports whether
    convergence happened inside the budget, so a caller can fall back.
    """

    if maximum_corrections < 1:
        raise ValueError("maximum_corrections must be positive")

    dtype = coordinates.dtype

    def extend(_iteration: jax.Array, state: _RetractionState) -> _RetractionState:
        _, constraints = joint_value_constraints(state.coordinates)
        feasibility = jnp.linalg.norm(constraints, ord=jnp.inf)
        required = (
            state.all_finite
            & jnp.isfinite(feasibility)
            & (feasibility > feasibility_tolerance)
        )

        def correct(active: _RetractionState) -> _RetractionState:
            correction = certified_correction_with_projector(projector, constraints)
            return _RetractionState(
                coordinates=active.coordinates + correction.correction,
                correction_path_norm=active.correction_path_norm
                + jnp.linalg.norm(correction.correction),
                corrections=active.corrections + jnp.asarray(1, dtype=jnp.int32),
                solve_relative_residual=jnp.maximum(
                    active.solve_relative_residual, correction.relative_residual
                ),
                solve_forward_error_bound=jnp.maximum(
                    active.solve_forward_error_bound, correction.forward_error_bound
                ),
                all_finite=active.all_finite & correction.all_finite,
            )

        return jax.lax.cond(required, correct, lambda active: active, state)

    retracted = jax.lax.fori_loop(
        0,
        maximum_corrections,
        extend,
        _RetractionState(
            coordinates=coordinates,
            correction_path_norm=jnp.zeros((), dtype=dtype),
            corrections=jnp.asarray(0, dtype=jnp.int32),
            solve_relative_residual=jnp.zeros((), dtype=dtype),
            solve_forward_error_bound=jnp.zeros((), dtype=dtype),
            all_finite=jnp.asarray(True),
        ),
    )
    return _finish_retraction(joint_value_constraints, retracted, feasibility_tolerance)


def project_into_tangent_space(
    projector: CertifiedGramProjector,
    vectors: jax.Array,
) -> jax.Array:
    """Project a ``(batch, dimension)`` stack into ``null(A)`` row by row.

    This is the vector transport for an embedded manifold: the ambient vector
    is simply re-projected into the tangent space at the new point.
    """

    return jax.vmap(
        lambda vector: project_with_certified_gram(projector, vector).projected
    )(vectors)


def _transported_pair(
    projector: CertifiedGramProjector,
    step: jax.Array,
    previous_gradient: jax.Array,
    projected_gradient: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Form one curvature pair entirely inside the NEW point's tangent space.

    ``y`` subtracts the previous RAW gradient projected here from the gradient
    already projected here, so both terms are measured in one space; ``s`` is
    the realized displacement projected into that same space.
    """

    carried = project_into_tangent_space(
        projector, jnp.stack((step, previous_gradient))
    )
    return carried[0], projected_gradient - carried[1]


def _admit_curvature_pair(
    metric: QuasiNewtonMetric,
    step: jax.Array,
    gradient_change: jax.Array,
) -> tuple[QuasiNewtonMetric, jax.Array, jax.Array, jax.Array]:
    """Store one pair and report the curvature that decided its admission."""

    updated = insert_curvature_pair(metric, step, gradient_change)
    return (
        updated,
        step @ gradient_change,
        curvature_pair_admissible(step, gradient_change),
        valid_pair_count(updated),
    )


def run_projected_lbfgs(
    joint_value_constraints: JointValueConstraints,
    initial_coordinates: jax.Array,
    *,
    options: ProjectedLbfgsOptions = _DEFAULT_OPTIONS,
    objective_residuals: Callable[[jax.Array], jax.Array] | None = None,
    observer: Callable[[ProjectedLbfgsIteration], None] | None = None,
) -> ProjectedLbfgsRun:
    """Run the bounded projected L-BFGS loop from one feasible starting point.

    The host drives one jitted kernel per phase — point evaluation, retraction,
    the correction store, and whichever direction model the options select — so
    that every phase is separately timed; ``observer`` receives each iteration
    record as it is produced, before the run returns.
    """

    _validate_options(options)
    # Every coordinate array the loop feeds back into these kernels comes out
    # of one of them, and a jitted output is COMMITTED to the device it ran on
    # while a caller's array need not be.  ``jax.jit`` caches on that
    # difference, so an uncommitted start point compiles the point evaluation,
    # the retraction and the direction solve a SECOND time on their first
    # in-loop call -- 31 s of the measured 92 s fixed cost on the coupled
    # full-space problem.  Committing the start once is numerically inert and
    # leaves one executable per kernel.
    initial_coordinates = jax.device_put(
        initial_coordinates, next(iter(initial_coordinates.devices()))
    )
    dimension = int(initial_coordinates.shape[0])
    dtype = initial_coordinates.dtype

    evaluate = jax.jit(
        lambda coordinates: evaluate_projected_point(
            joint_value_constraints, coordinates
        )
    )
    evaluate_carried = jax.jit(
        lambda projector, coordinates: evaluate_projected_point_with_projector(
            joint_value_constraints, projector, coordinates
        )
    )
    exact_retract = jax.jit(
        lambda coordinates: retract_to_manifold(
            joint_value_constraints,
            coordinates,
            feasibility_tolerance=options.feasibility_tolerance,
            maximum_corrections=options.maximum_retraction_corrections,
        )
    )
    frozen_retract = jax.jit(
        lambda projector, coordinates: retract_with_frozen_projector(
            joint_value_constraints,
            projector,
            coordinates,
            feasibility_tolerance=options.feasibility_tolerance,
            maximum_corrections=options.maximum_retraction_corrections,
        )
    )

    def retract(projector: CertifiedGramProjector, coordinates: jax.Array):
        if options.frozen_projector_line_search:
            return frozen_retract(projector, coordinates)
        return exact_retract(coordinates)

    apply_metric = jax.jit(apply_quasi_newton_metric)
    admit_pair = jax.jit(_admit_curvature_pair)
    transport_metric = jax.jit(
        lambda metric, projector: transport_quasi_newton_metric(
            metric, lambda vectors: project_into_tangent_space(projector, vectors)
        )
    )
    transported_pair = jax.jit(_transported_pair)
    dense_direction = jax.jit(
        lambda dense, projector, projected_gradient: dense_tangent_direction(
            dense,
            projected_gradient,
            lambda vector: project_with_certified_gram(projector, vector).projected,
        )
    )
    update_curvature = jax.jit(update_dense_tangent_curvature)
    if options.lagrangian_newton:

        def _lagrangian_newton_direction(
            coordinates, projector, gradient, projected_gradient, metric
        ):
            """Multipliers, then a truncated Newton solve on their Lagrangian.

            Both live in one jitted region so the multiplier solve reuses the
            already factored Gram without a host round trip, and the frozen
            multipliers reach the curvature operator as a device value.
            """

            multipliers = least_squares_multipliers(projector, gradient)
            return multipliers, solve_tangent_newton_cg(
                lagrangian_hessian_vector_product(
                    joint_value_constraints, coordinates, multipliers.multipliers
                ),
                projector,
                projected_gradient,
                maximum_iterations=options.newton_cg_maximum_iterations,
                forcing_maximum=options.newton_cg_forcing_maximum,
                preconditioner=lambda vector: apply_quasi_newton_metric(metric, vector),
            )

        lagrangian_newton_direction = jax.jit(_lagrangian_newton_direction)
    if objective_residuals is not None and not options.gauss_newton:
        raise ValueError("objective_residuals requires gauss_newton")
    if options.gauss_newton:
        if objective_residuals is None:
            raise ValueError("gauss_newton requires objective_residuals")

        def _gauss_newton_direction(coordinates, projector, projected_gradient):
            return solve_tangent_gauss_newton(
                materialize_residual_jacobian(objective_residuals, coordinates),
                materialize_certified_projection(projector),
                projected_gradient,
                levenberg_relative=options.gauss_newton_levenberg_relative,
            )

        gauss_newton_direction = jax.jit(_gauss_newton_direction)

    compile_started = time.perf_counter()
    point = jax.block_until_ready(evaluate(initial_coordinates))
    compile_seconds = time.perf_counter() - compile_started

    coordinates = initial_coordinates
    metric = empty_quasi_newton_metric(options.memory, dimension, dtype)
    curvature = empty_dense_tangent_curvature(dimension, dtype)
    records: list[ProjectedLbfgsIteration] = []
    status = ProjectedLbfgsStatus.RUNNING
    stored_pairs = 0
    # Accepted iterations the projector in hand has already served, plus the
    # seconds a forced refresh spent outside any record's own accounting.
    projector_age = 0
    projector_materializations = 1
    tangency_forced_refreshes = 0
    line_search_forced_refreshes = 0
    abandoned = _NO_ABANDONED_ATTEMPT
    # Without a hybrid window the selected model runs from the first iteration;
    # with one the run opens on the cheap secant step and earns its way in.
    newton_active = options.lagrangian_newton and options.hybrid_plateau_window == 0
    newton_rearm_index = 0

    solve_started = time.perf_counter()
    if not bool(point.all_finite):
        status = ProjectedLbfgsStatus.NONFINITE_STATE
    elif float(point.feasibility_inf) > options.feasibility_tolerance:
        status = ProjectedLbfgsStatus.INFEASIBLE_START

    while status is ProjectedLbfgsStatus.RUNNING:
        index = len(records)
        if (
            options.objective_target is not None
            and float(point.objective) <= options.objective_target
        ):
            status = ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED
            break
        if index >= options.maximum_iterations:
            status = ProjectedLbfgsStatus.ITERATION_LIMIT
            break

        trailing_factor = _trailing_objective_factor(
            records, float(point.objective), options.hybrid_plateau_window
        )
        if (
            options.lagrangian_newton
            and options.hybrid_plateau_window > 0
            and not newton_active
            and index - newton_rearm_index >= options.hybrid_plateau_window
            and math.isfinite(trailing_factor)
        ):
            newton_active = trailing_factor >= options.hybrid_plateau_factor

        direction_started = time.perf_counter()
        applied_metric = metric
        masked_pairs = 0
        dense_pd = True
        gauss_newton_used = False
        gauss_newton_rescued = False
        gn_predicted = float("nan")
        gn_condition = float("nan")
        gn_tangency = float("nan")
        gn_direction_norm = float("nan")
        lagrangian_newton_used = False
        lagrangian_newton_rescued = False
        newton_step: TangentNewtonCgStep | None = None
        model_predicted = float("nan")
        multiplier_norm = float("nan")
        multiplier_forward_error_bound = float("nan")
        # The age of the rows THIS iteration's direction and line search are
        # spending, fixed before the acceptance can advance it.
        direction_projector_age = projector_age
        # The fraction of the gradient that survives the projection is already
        # in hand; where it is small the reduced model has almost nothing to
        # describe and the solve would be spent to be refused downstream.
        gradient_norm_value = float(point.gradient_norm)
        tangent_fraction = (
            float(point.projected_gradient_norm) / gradient_norm_value
            if gradient_norm_value > 0.0
            else float("nan")
        )
        newton_gate_skipped = (
            options.lagrangian_newton
            and newton_active
            and options.newton_tangent_fraction_threshold > 0.0
            and math.isfinite(tangent_fraction)
            and tangent_fraction < options.newton_tangent_fraction_threshold
        )
        if options.lagrangian_newton and newton_active and not newton_gate_skipped:
            multipliers, newton_step = jax.block_until_ready(
                lagrangian_newton_direction(
                    coordinates,
                    point.projector,
                    point.gradient,
                    point.projected_gradient,
                    metric,
                )
            )
            multiplier_norm = float(multipliers.norm)
            multiplier_forward_error_bound = float(multipliers.forward_error_bound)
            if bool(newton_step.usable):
                direction = newton_step.direction
                lagrangian_newton_used = True
                model_predicted = float(newton_step.predicted_reduction)
            else:
                # A refused Newton solve is evidence, not a failure: the
                # telemetry is kept and the secant store takes this iteration.
                direction = jax.block_until_ready(
                    apply_metric(metric, -point.projected_gradient)
                )
        elif options.gauss_newton:
            gn_step = jax.block_until_ready(
                gauss_newton_direction(
                    coordinates, point.projector, point.projected_gradient
                )
            )
            if bool(gn_step.all_finite):
                direction = gn_step.direction
                gauss_newton_used = True
                gn_predicted = float(gn_step.predicted_reduction)
                model_predicted = gn_predicted
                gn_condition = float(gn_step.reduced_hessian_reciprocal_condition)
                gn_tangency = float(gn_step.tangency_relative_residual)
                gn_direction_norm = float(jnp.linalg.norm(gn_step.direction))
            else:
                direction = jax.block_until_ready(
                    apply_metric(metric, -point.projected_gradient)
                )
        elif options.dense_curvature:
            dense_step = jax.block_until_ready(
                dense_direction(curvature, point.projector, point.projected_gradient)
            )
            direction = dense_step.direction
            dense_pd = bool(dense_step.positive_definite)
        else:
            if options.vector_transport:
                carried = jax.block_until_ready(
                    transport_metric(metric, point.projector)
                )
                applied_metric = carried.metric
                masked_pairs = int(carried.masked_pairs)
            direction = jax.block_until_ready(
                apply_metric(applied_metric, -point.projected_gradient)
            )
        directional_derivative = float(point.projected_gradient @ direction)
        direction_norm = float(jnp.linalg.norm(direction))
        direction_seconds = time.perf_counter() - direction_started

        if not math.isfinite(direction_norm) or not math.isfinite(
            directional_derivative
        ):
            status = ProjectedLbfgsStatus.NONFINITE_STATE
            break
        if directional_derivative >= 0.0:
            status = ProjectedLbfgsStatus.NON_DESCENT_DIRECTION
            break

        # A curvature-model step is a MODEL-scale step, not a locally safe one.
        # Capping it to maximum_step_norm reimposes exactly the trust ball this
        # route exists to remove -- the projected GN trust-region arm measured
        # its directions at 152-1950x that ball -- and with the cap in place the
        # initial scale lands below the rescue floor, so the step is never even
        # tried.  The cap therefore applies only to secant directions.
        model_step_used = gauss_newton_used or lagrangian_newton_used
        if model_step_used:
            capped = False
            step_scale = 1.0
        else:
            capped = direction_norm > options.maximum_step_norm
            step_scale = options.maximum_step_norm / direction_norm if capped else 1.0

        line_search_started = time.perf_counter()
        retraction_seconds = 0.0
        trials = 0
        rejected_for_feasibility = 0
        accepted: ManifoldRetraction | None = None

        def search(
            search_direction: jax.Array,
            slope: float,
            initial_scale: float,
            floor_scale: float,
            *,
            point: ProjectedPoint = point,
            coordinates: jax.Array = coordinates,
        ):
            """Backtrack from ``initial_scale`` until Armijo holds or ``floor_scale``.

            ``point`` and ``coordinates`` are bound as defaults so the closure
            cannot outlive the iteration that created it.
            """

            nonlocal trials, rejected_for_feasibility, retraction_seconds
            scale = initial_scale
            while trials < options.maximum_line_search_trials and scale >= floor_scale:
                trials += 1
                started = time.perf_counter()
                candidate = jax.block_until_ready(
                    retract(point.projector, coordinates + scale * search_direction)
                )
                retraction_seconds += time.perf_counter() - started
                if bool(candidate.feasible):
                    sufficient = float(point.objective) + (
                        options.armijo_coefficient * scale * slope
                    )
                    if float(candidate.objective) <= sufficient:
                        return candidate, scale
                else:
                    rejected_for_feasibility += 1
                scale *= options.backtracking_factor
            return None, scale

        # A model step is tried at full length and abandoned early: if it is
        # not accepted well above the rescue scale the model is not describing
        # this neighbourhood, and the pair store is the better bet.
        rescue_floor = (
            options.model_step_rescue_scale
            if model_step_used
            else options.minimum_step_scale
        )
        accepted, step_scale = search(
            direction, directional_derivative, step_scale, rescue_floor
        )
        if accepted is None and model_step_used:
            rescue_direction = jax.block_until_ready(
                apply_metric(metric, -point.projected_gradient)
            )
            rescue_slope = float(point.projected_gradient @ rescue_direction)
            rescue_norm = float(jnp.linalg.norm(rescue_direction))
            if (
                math.isfinite(rescue_slope)
                and rescue_slope < 0.0
                and math.isfinite(rescue_norm)
                and rescue_norm > 0.0
            ):
                rescue_capped = rescue_norm > options.maximum_step_norm
                initial = (
                    options.maximum_step_norm / rescue_norm if rescue_capped else 1.0
                )
                rescued, rescue_scale = search(
                    rescue_direction,
                    rescue_slope,
                    initial,
                    options.minimum_step_scale,
                )
                # The rescue owns the iteration only if it placed a step.  A
                # rescue the shared trial budget starved to zero attempts
                # placed nothing, and stamping the direction fields with a
                # candidate that was never retracted would report the model
                # arm as rescued on evidence that does not exist.
                if rescued is not None:
                    accepted = rescued
                    step_scale = rescue_scale
                    gauss_newton_rescued = gauss_newton_used
                    lagrangian_newton_rescued = lagrangian_newton_used
                    capped = rescue_capped
                    direction = rescue_direction
                    directional_derivative = rescue_slope
                    direction_norm = rescue_norm
        line_search_seconds = time.perf_counter() - line_search_started

        # A line search that placed nothing against CARRIED rows has not proved
        # anything about this point: the trials it rejected were retracted by a
        # chord whose rows belong somewhere else.  Materialize here and take the
        # iteration again.  The retry cannot repeat, because the refreshed
        # projector is not stale, so a second failure is the real verdict.  It
        # runs BEFORE the hybrid disarm below for the same reason: an attempt
        # that proved nothing is not evidence against the model it used.  Only
        # the seconds and trials it spent are carried; its own direction and
        # curvature telemetry belong to a tangent space this point does not
        # have, and the retry produces the record's telemetry afresh.
        if accepted is None and projector_age > 0:
            refresh_started = time.perf_counter()
            point = jax.block_until_ready(evaluate(coordinates))
            abandoned = _AbandonedAttempt(
                direction_seconds=abandoned.direction_seconds + direction_seconds,
                line_search_seconds=abandoned.line_search_seconds
                + line_search_seconds,
                retraction_seconds=abandoned.retraction_seconds + retraction_seconds,
                point_evaluation_seconds=abandoned.point_evaluation_seconds
                + (time.perf_counter() - refresh_started),
                trials=abandoned.trials + trials,
                rejected_for_feasibility=abandoned.rejected_for_feasibility
                + rejected_for_feasibility,
            )
            projector_age = 0
            projector_materializations += 1
            line_search_forced_refreshes += 1
            if not bool(point.all_finite):
                status = ProjectedLbfgsStatus.NONFINITE_STATE
                break
            continue

        # A Newton phase that could not place a step -- because the solve
        # refused one or because the line search would not take it -- hands the
        # run back to the secant store for another window rather than paying
        # for a curvature solve every iteration to be told the same thing.  It
        # runs after the retry so that ``index`` is the index of an iteration
        # that will be RECORDED, which is what makes the window that counts
        # from ``index + 1`` exactly as many secant iterations as the option
        # names.
        if (
            options.hybrid_plateau_window > 0
            and newton_active
            and not newton_gate_skipped
            and (lagrangian_newton_rescued or not lagrangian_newton_used)
        ):
            newton_active = False
            newton_rearm_index = index + 1

        if accepted is None:
            status = ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE
            records.append(
                _collapsed_record(
                    index=index,
                    coordinates=coordinates,
                    point=point,
                    direction_norm=direction_norm,
                    directional_derivative=directional_derivative,
                    step_scale=step_scale,
                    capped=capped,
                    trials=trials + abandoned.trials,
                    rejected_for_feasibility=rejected_for_feasibility
                    + abandoned.rejected_for_feasibility,
                    stored_pairs=stored_pairs,
                    masked_pairs=masked_pairs,
                    dense_damped_updates=(
                        int(curvature.damped_updates) if options.dense_curvature else 0
                    ),
                    dense_positive_definite=dense_pd,
                    gauss_newton_used=gauss_newton_used,
                    gauss_newton_rescued=gauss_newton_rescued,
                    gauss_newton_predicted_reduction=gn_predicted,
                    gauss_newton_reciprocal_condition=gn_condition,
                    gauss_newton_tangency_residual=gn_tangency,
                    gauss_newton_direction_norm=gn_direction_norm,
                    lagrangian_newton_used=lagrangian_newton_used,
                    lagrangian_newton_rescued=lagrangian_newton_rescued,
                    newton_step=newton_step,
                    multiplier_norm=multiplier_norm,
                    multiplier_forward_error_bound=multiplier_forward_error_bound,
                    trailing_objective_factor=trailing_factor,
                    projector_age=direction_projector_age,
                    tangent_fraction=tangent_fraction,
                    newton_gate_skipped=newton_gate_skipped,
                    direction_seconds=direction_seconds + abandoned.direction_seconds,
                    line_search_seconds=line_search_seconds
                    + abandoned.line_search_seconds,
                    retraction_seconds=retraction_seconds
                    + abandoned.retraction_seconds,
                    point_evaluation_seconds=abandoned.point_evaluation_seconds,
                )
            )
            if observer is not None:
                observer(records[-1])
            break

        step = accepted.coordinates - coordinates
        point_started = time.perf_counter()
        if projector_age + 1 < options.projector_refresh_period:
            next_point = jax.block_until_ready(
                evaluate_carried(point.projector, accepted.coordinates)
            )
            carry_tangency = float(next_point.true_tangency_relative_residual)
            # The carry is kept only while its own measurement says the tangent
            # space it names is still this point's.
            if options.projector_tangency_tolerance > 0.0 and not (
                carry_tangency <= options.projector_tangency_tolerance
            ):
                next_point = jax.block_until_ready(evaluate(accepted.coordinates))
                projector_age = 0
                projector_materializations += 1
                tangency_forced_refreshes += 1
            else:
                projector_age += 1
        else:
            next_point = jax.block_until_ready(evaluate(accepted.coordinates))
            projector_age = 0
            projector_materializations += 1
        point_evaluation_seconds = (
            time.perf_counter() - point_started + abandoned.point_evaluation_seconds
        )

        pair_started = time.perf_counter()
        if options.vector_transport:
            pair_step, pair_gradient_change = jax.block_until_ready(
                transported_pair(
                    next_point.projector,
                    step,
                    point.gradient,
                    next_point.projected_gradient,
                )
            )
        else:
            pair_step = step
            pair_gradient_change = (
                next_point.projected_gradient - point.projected_gradient
            )
        metric, pair_curvature, admitted, live_pairs = jax.block_until_ready(
            admit_pair(metric, pair_step, pair_gradient_change)
        )
        if options.dense_curvature:
            curvature = jax.block_until_ready(
                update_curvature(curvature, pair_step, pair_gradient_change)
            )
            dense_damped = int(curvature.damped_updates)
        else:
            dense_damped = 0
        pair_update_seconds = time.perf_counter() - pair_started
        # The model's own decrease along the step that was ACCEPTED.
        # ``model_predicted`` is ``-(g.d + 0.5 d.H d)`` at full length, so the
        # same quadratic predicts ``-(t g.d + t^2 0.5 d.H d)`` at scale ``t``,
        # which rearranges into these two numbers the iteration already holds.
        # Dividing the realized decrease by the FULL-step prediction instead
        # reports every backtracked iteration as an inexact model: a step at
        # scale 1/8 buys roughly an eighth of the full-step decrease and would
        # score near 0.125 however exact the model is.
        scaled_model_predicted = (
            step_scale * step_scale * model_predicted
            + (step_scale * step_scale - step_scale) * directional_derivative
        )
        stored_pairs = int(live_pairs)

        record = ProjectedLbfgsIteration(
            index=index,
            coordinates=coordinates,
            objective=float(point.objective),
            projected_gradient_inf=float(point.projected_gradient_inf),
            projected_gradient_norm=float(point.projected_gradient_norm),
            gradient_norm=float(point.gradient_norm),
            feasibility_inf=float(point.feasibility_inf),
            tangency_relative_residual=float(point.tangency_relative_residual),
            true_tangency_relative_residual=float(
                point.true_tangency_relative_residual
            ),
            projector_age=direction_projector_age,
            projector_refreshed=direction_projector_age == 0,
            tangent_gradient_fraction=tangent_fraction,
            newton_gate_skipped=newton_gate_skipped,
            gram_reciprocal_condition=float(point.gram_reciprocal_condition),
            direction_norm=direction_norm,
            directional_derivative=directional_derivative,
            step_scale=step_scale,
            step_norm=float(jnp.linalg.norm(step)),
            step_norm_capped=bool(capped),
            line_search_trials=trials + abandoned.trials,
            rejected_for_feasibility=rejected_for_feasibility
            + abandoned.rejected_for_feasibility,
            retraction_corrections=int(accepted.corrections),
            candidate_objective=float(accepted.objective),
            candidate_feasibility_inf=float(accepted.feasibility_inf),
            curvature=float(pair_curvature),
            pair_admitted=bool(admitted),
            stored_pairs=stored_pairs,
            transport_masked_pairs=masked_pairs,
            dense_damped_updates=dense_damped,
            dense_positive_definite=dense_pd,
            gauss_newton_used=gauss_newton_used,
            gauss_newton_rescued=gauss_newton_rescued,
            gauss_newton_predicted_reduction=gn_predicted,
            gauss_newton_reciprocal_condition=gn_condition,
            gauss_newton_tangency_residual=gn_tangency,
            gauss_newton_direction_norm=gn_direction_norm,
            lagrangian_newton_used=lagrangian_newton_used,
            lagrangian_newton_rescued=lagrangian_newton_rescued,
            **_newton_telemetry(newton_step),
            multiplier_norm=multiplier_norm,
            multiplier_forward_error_bound=multiplier_forward_error_bound,
            trailing_objective_factor=trailing_factor,
            model_exactness_ratio=(
                (float(point.objective) - float(accepted.objective))
                / scaled_model_predicted
                if model_step_used
                and not (gauss_newton_rescued or lagrangian_newton_rescued)
                and math.isfinite(scaled_model_predicted)
                and scaled_model_predicted != 0.0
                else float("nan")
            ),
            direction_seconds=direction_seconds + abandoned.direction_seconds,
            line_search_seconds=line_search_seconds + abandoned.line_search_seconds,
            retraction_seconds=retraction_seconds + abandoned.retraction_seconds,
            point_evaluation_seconds=point_evaluation_seconds,
            pair_update_seconds=pair_update_seconds,
        )
        abandoned = _NO_ABANDONED_ATTEMPT
        records.append(record)
        if observer is not None:
            observer(record)

        coordinates = accepted.coordinates
        point = next_point
        if not bool(point.all_finite):
            status = ProjectedLbfgsStatus.NONFINITE_STATE
            break

    solve_seconds = time.perf_counter() - solve_started
    return ProjectedLbfgsRun(
        status=status,
        coordinates=coordinates,
        objective=float(point.objective),
        feasibility_inf=float(point.feasibility_inf),
        projected_gradient_inf=float(point.projected_gradient_inf),
        stored_pairs=stored_pairs,
        iterations=tuple(records),
        compile_seconds=compile_seconds,
        solve_seconds=solve_seconds,
        projector_materializations=projector_materializations,
        tangency_forced_refreshes=tangency_forced_refreshes,
        line_search_forced_refreshes=line_search_forced_refreshes,
    )


def _trailing_objective_factor(
    records: list[ProjectedLbfgsIteration],
    current_objective: float,
    window: int,
) -> float:
    """Geometric per-iteration objective decrease over the trailing window.

    A value approaching one is a plateau: the model in use is buying less per
    iteration than a more expensive one would have to buy to be worth its cost.

    The ratio is only a rate for a positive objective, so a window that reaches
    a nonpositive one reports NaN.  ``hybrid_plateau_window`` requires a finite
    factor to rearm, which means an objective that can go nonpositive leaves the
    hybrid arm on the secant store for good; such a run wants the Newton arm
    selected outright rather than scheduled.
    """

    if window < 1 or len(records) < window:
        return float("nan")
    earlier = records[-window].objective
    if not (earlier > 0.0 and current_objective > 0.0):
        return float("nan")
    return (current_objective / earlier) ** (1.0 / window)


def _newton_telemetry(step: TangentNewtonCgStep | None) -> dict[str, object]:
    """Host-side scalars for one Newton--CG solve, blank when none was run."""

    if step is None:
        return {
            "newton_cg_iterations": 0,
            "newton_cg_termination": -1,
            "newton_cg_negative_curvature": False,
            "newton_cg_negative_curvature_before_any_step": False,
            "newton_cg_forcing_eta": float("nan"),
            "newton_cg_initial_residual_norm": float("nan"),
            "newton_cg_final_residual_norm": float("nan"),
            "newton_curvature": float("nan"),
            "newton_normalized_curvature": float("nan"),
            "newton_curvature_rayleigh_history": (),
            "newton_direction_norm": float("nan"),
            "newton_predicted_reduction": float("nan"),
        }
    return {
        "newton_cg_iterations": int(step.iterations),
        "newton_cg_termination": int(step.termination),
        "newton_cg_negative_curvature": bool(step.negative_curvature),
        "newton_cg_negative_curvature_before_any_step": bool(
            step.negative_curvature_before_any_step
        ),
        "newton_cg_forcing_eta": float(step.forcing_eta),
        "newton_cg_initial_residual_norm": float(step.initial_residual_norm),
        "newton_cg_final_residual_norm": float(step.final_residual_norm),
        "newton_curvature": float(step.terminal_curvature),
        "newton_normalized_curvature": float(step.terminal_normalized_curvature),
        "newton_curvature_rayleigh_history": tuple(
            float(value)
            for value in step.curvature_rayleigh_history[: int(step.iterations)]
        ),
        "newton_direction_norm": float(step.direction_norm),
        "newton_predicted_reduction": float(step.predicted_reduction),
    }


def _collapsed_record(
    *,
    index: int,
    coordinates: jax.Array,
    point: ProjectedPoint,
    direction_norm: float,
    directional_derivative: float,
    step_scale: float,
    capped: bool,
    trials: int,
    rejected_for_feasibility: int,
    stored_pairs: int,
    masked_pairs: int,
    dense_damped_updates: int,
    dense_positive_definite: bool,
    gauss_newton_used: bool,
    gauss_newton_rescued: bool,
    gauss_newton_predicted_reduction: float,
    gauss_newton_reciprocal_condition: float,
    gauss_newton_tangency_residual: float,
    gauss_newton_direction_norm: float,
    lagrangian_newton_used: bool,
    lagrangian_newton_rescued: bool,
    newton_step: TangentNewtonCgStep | None,
    multiplier_norm: float,
    multiplier_forward_error_bound: float,
    trailing_objective_factor: float,
    projector_age: int,
    tangent_fraction: float,
    newton_gate_skipped: bool,
    direction_seconds: float,
    line_search_seconds: float,
    retraction_seconds: float,
    point_evaluation_seconds: float,
) -> ProjectedLbfgsIteration:
    """Record the iteration whose line search found no acceptable step.

    The direction and curvature telemetry is carried through unchanged: an
    iteration that could place no step is the one whose direction evidence
    matters most.  Only the fields an ACCEPTED step would have produced -- the
    candidate, the admitted pair, the realized decrease -- are absent, and they
    are absent rather than defaulted.
    """

    return ProjectedLbfgsIteration(
        index=index,
        coordinates=coordinates,
        objective=float(point.objective),
        projected_gradient_inf=float(point.projected_gradient_inf),
        projected_gradient_norm=float(point.projected_gradient_norm),
        gradient_norm=float(point.gradient_norm),
        feasibility_inf=float(point.feasibility_inf),
        tangency_relative_residual=float(point.tangency_relative_residual),
        true_tangency_relative_residual=float(point.true_tangency_relative_residual),
        projector_age=projector_age,
        projector_refreshed=projector_age == 0,
        tangent_gradient_fraction=tangent_fraction,
        newton_gate_skipped=newton_gate_skipped,
        gram_reciprocal_condition=float(point.gram_reciprocal_condition),
        direction_norm=direction_norm,
        directional_derivative=directional_derivative,
        step_scale=step_scale,
        step_norm=float("nan"),
        step_norm_capped=capped,
        line_search_trials=trials,
        rejected_for_feasibility=rejected_for_feasibility,
        retraction_corrections=-1,
        candidate_objective=float("nan"),
        candidate_feasibility_inf=float("nan"),
        curvature=float("nan"),
        pair_admitted=False,
        dense_damped_updates=dense_damped_updates,
        dense_positive_definite=dense_positive_definite,
        gauss_newton_used=gauss_newton_used,
        gauss_newton_rescued=gauss_newton_rescued,
        gauss_newton_predicted_reduction=gauss_newton_predicted_reduction,
        gauss_newton_reciprocal_condition=gauss_newton_reciprocal_condition,
        gauss_newton_tangency_residual=gauss_newton_tangency_residual,
        gauss_newton_direction_norm=gauss_newton_direction_norm,
        lagrangian_newton_used=lagrangian_newton_used,
        lagrangian_newton_rescued=lagrangian_newton_rescued,
        **_newton_telemetry(newton_step),
        multiplier_norm=multiplier_norm,
        multiplier_forward_error_bound=multiplier_forward_error_bound,
        trailing_objective_factor=trailing_objective_factor,
        model_exactness_ratio=float("nan"),
        stored_pairs=stored_pairs,
        transport_masked_pairs=masked_pairs,
        direction_seconds=direction_seconds,
        line_search_seconds=line_search_seconds,
        retraction_seconds=retraction_seconds,
        point_evaluation_seconds=point_evaluation_seconds,
        pair_update_seconds=0.0,
    )


def _validate_options(options: ProjectedLbfgsOptions) -> None:
    if options.maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if options.memory < 1:
        raise ValueError("memory must be positive")
    if options.maximum_line_search_trials < 1:
        raise ValueError("maximum_line_search_trials must be positive")
    if not 0.0 < options.backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must lie strictly inside (0, 1)")
    if not 0.0 < options.armijo_coefficient < 1.0:
        raise ValueError("armijo_coefficient must lie strictly inside (0, 1)")
    positive = (
        options.feasibility_tolerance,
        options.maximum_step_norm,
        options.minimum_step_scale,
        options.model_step_rescue_scale,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("tolerances and bounds must be finite and positive")
    if options.objective_target is not None and not math.isfinite(
        options.objective_target
    ):
        raise ValueError("objective_target must be finite")
    if options.newton_cg_maximum_iterations < 1:
        raise ValueError("newton_cg_maximum_iterations must be positive")
    if not 0.0 < options.newton_cg_forcing_maximum < 1.0:
        raise ValueError("newton_cg_forcing_maximum must lie strictly inside (0, 1)")
    if options.hybrid_plateau_window < 0:
        raise ValueError("hybrid_plateau_window must not be negative")
    if not 0.0 < options.hybrid_plateau_factor <= 1.0:
        raise ValueError("hybrid_plateau_factor must lie inside (0, 1]")
    if options.hybrid_plateau_window > 0 and not options.lagrangian_newton:
        raise ValueError("hybrid_plateau_window requires lagrangian_newton")
    if options.projector_refresh_period < 1:
        raise ValueError("projector_refresh_period must be positive")
    if (
        not math.isfinite(options.projector_tangency_tolerance)
        or options.projector_tangency_tolerance < 0.0
    ):
        raise ValueError("projector_tangency_tolerance must be finite and nonnegative")
    if not 0.0 <= options.newton_tangent_fraction_threshold < 1.0:
        raise ValueError("newton_tangent_fraction_threshold must lie inside [0, 1)")
    if options.newton_tangent_fraction_threshold > 0.0 and not (
        options.lagrangian_newton
    ):
        raise ValueError("newton_tangent_fraction_threshold requires lagrangian_newton")
    if options.maximum_retraction_corrections < 1:
        raise ValueError("maximum_retraction_corrections must be positive")
    if (
        not math.isfinite(options.gauss_newton_levenberg_relative)
        or options.gauss_newton_levenberg_relative < 0.0
    ):
        raise ValueError(
            "gauss_newton_levenberg_relative must be finite and nonnegative"
        )
    selected = sum(
        (options.gauss_newton, options.dense_curvature, options.lagrangian_newton)
    )
    if selected > 1:
        raise ValueError(
            "gauss_newton, dense_curvature and lagrangian_newton are exclusive"
        )
    if options.vector_transport and selected:
        raise ValueError("vector_transport applies to the secant direction only")


__all__ = (
    "ManifoldRetraction",
    "ProjectedLbfgsIteration",
    "ProjectedLbfgsOptions",
    "ProjectedLbfgsRun",
    "ProjectedLbfgsStatus",
    "ProjectedPoint",
    "evaluate_projected_point",
    "evaluate_projected_point_with_projector",
    "retract_to_manifold",
    "retract_with_frozen_projector",
    "run_projected_lbfgs",
)
