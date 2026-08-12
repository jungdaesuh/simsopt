"""Projected L-BFGS on an equality-constraint manifold.

Curvature stepping on the true objective, with the constraints enforced by
projection and retraction rather than by penalties: the objective gradient is
projected into the tangent space of the linearized equalities, the
limited-memory inverse Hessian turns that tangent gradient into a search
direction, and every trial point is pulled back onto the manifold by certified
minimum-norm Newton corrections before its objective is compared.

The module owns one piece of knowledge: how a projected quasi-Newton step is
proposed, retracted, and accepted.  The certified tangent projector and the
certified Newton correction live in ``projected_hvp_trust_region``, the
correction store lives in ``quasi_newton_metric``, and the exact derivative
rows come from ``dense_sqp``.

The pair rule is the standard Riemannian-L-BFGS-with-projection
approximation: ``s`` is the realized on-manifold displacement (tangent step
plus normal correction) and ``y`` is the change in the *projected* gradient,
compared without vector transport.  Both vectors live in different tangent
spaces, which is exact only in the limit of small steps; the curvature
admission guard is what keeps the resulting operator positive definite.
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
from .projected_hvp_trust_region import (
    certified_minimum_norm_correction,
    factor_certified_gram_projector,
    project_with_certified_gram,
)
from .quasi_newton_metric import (
    QuasiNewtonMetric,
    apply_quasi_newton_metric,
    curvature_pair_admissible,
    empty_quasi_newton_metric,
    insert_curvature_pair,
    valid_pair_count,
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
    objective_target: float = 0.0


class ProjectedPoint(NamedTuple):
    """One on-manifold point with its tangent gradient and projector evidence."""

    objective: jax.Array
    constraints: jax.Array
    feasibility_inf: jax.Array
    projected_gradient: jax.Array
    projected_gradient_inf: jax.Array
    projected_gradient_norm: jax.Array
    gradient_norm: jax.Array
    tangency_relative_residual: jax.Array
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
    """Host-side scalar record of one iteration, complete enough to bank."""

    index: int
    objective: float
    projected_gradient_inf: float
    projected_gradient_norm: float
    gradient_norm: float
    feasibility_inf: float
    tangency_relative_residual: float
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
    direction_seconds: float
    line_search_seconds: float
    retraction_seconds: float
    point_evaluation_seconds: float
    pair_update_seconds: float


class ProjectedLbfgsRun(NamedTuple):
    """Terminal state of one run plus every banked iteration record."""

    status: ProjectedLbfgsStatus
    coordinates: jax.Array
    objective: float
    feasibility_inf: float
    projected_gradient_inf: float
    stored_pairs: int
    iterations: tuple[ProjectedLbfgsIteration, ...]
    compile_seconds: float
    solve_seconds: float


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
        objective=rows.objective,
        constraints=rows.constraints,
        feasibility_inf=jnp.linalg.norm(rows.constraints, ord=jnp.inf),
        projected_gradient=projected_gradient,
        projected_gradient_inf=jnp.linalg.norm(projected_gradient, ord=jnp.inf),
        projected_gradient_norm=jnp.linalg.norm(projected_gradient),
        gradient_norm=jnp.linalg.norm(rows.objective_gradient),
        tangency_relative_residual=projection.tangency_relative_residual,
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
    options: ProjectedLbfgsOptions = ProjectedLbfgsOptions(),
    observer: Callable[[ProjectedLbfgsIteration], None] | None = None,
) -> ProjectedLbfgsRun:
    """Run the bounded projected L-BFGS loop from one feasible starting point.

    The host drives three jitted kernels — point evaluation, retraction, and
    the correction store — so that every phase is separately timed; ``observer``
    receives each iteration record as it is produced, before the run returns.
    """

    _validate_options(options)
    dimension = int(initial_coordinates.shape[0])
    dtype = initial_coordinates.dtype

    evaluate = jax.jit(
        lambda coordinates: evaluate_projected_point(
            joint_value_constraints, coordinates
        )
    )
    retract = jax.jit(
        lambda coordinates: retract_to_manifold(
            joint_value_constraints,
            coordinates,
            feasibility_tolerance=options.feasibility_tolerance,
            maximum_corrections=options.maximum_retraction_corrections,
        )
    )
    apply_metric = jax.jit(apply_quasi_newton_metric)
    admit_pair = jax.jit(_admit_curvature_pair)

    compile_started = time.perf_counter()
    point = jax.block_until_ready(evaluate(initial_coordinates))
    compile_seconds = time.perf_counter() - compile_started

    coordinates = initial_coordinates
    metric = empty_quasi_newton_metric(options.memory, dimension, dtype)
    records: list[ProjectedLbfgsIteration] = []
    status = ProjectedLbfgsStatus.RUNNING
    stored_pairs = 0

    solve_started = time.perf_counter()
    if not bool(point.all_finite):
        status = ProjectedLbfgsStatus.NONFINITE_STATE
    elif float(point.feasibility_inf) > options.feasibility_tolerance:
        status = ProjectedLbfgsStatus.INFEASIBLE_START

    while status is ProjectedLbfgsStatus.RUNNING:
        index = len(records)
        if float(point.objective) <= options.objective_target:
            status = ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED
            break
        if index >= options.maximum_iterations:
            status = ProjectedLbfgsStatus.ITERATION_LIMIT
            break

        direction_started = time.perf_counter()
        direction = jax.block_until_ready(
            apply_metric(metric, -point.projected_gradient)
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

        capped = direction_norm > options.maximum_step_norm
        step_scale = (
            options.maximum_step_norm / direction_norm if capped else 1.0
        )

        line_search_started = time.perf_counter()
        retraction_seconds = 0.0
        trials = 0
        rejected_for_feasibility = 0
        accepted: ManifoldRetraction | None = None
        while (
            trials < options.maximum_line_search_trials
            and step_scale >= options.minimum_step_scale
        ):
            trials += 1
            retraction_started = time.perf_counter()
            candidate = jax.block_until_ready(
                retract(coordinates + step_scale * direction)
            )
            retraction_seconds += time.perf_counter() - retraction_started
            if bool(candidate.feasible):
                sufficient = float(point.objective) + (
                    options.armijo_coefficient * step_scale * directional_derivative
                )
                if float(candidate.objective) <= sufficient:
                    accepted = candidate
                    break
            else:
                rejected_for_feasibility += 1
            step_scale *= options.backtracking_factor
        line_search_seconds = time.perf_counter() - line_search_started

        if accepted is None:
            status = ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE
            records.append(
                _collapsed_record(
                    index=index,
                    point=point,
                    direction_norm=direction_norm,
                    directional_derivative=directional_derivative,
                    step_scale=step_scale,
                    capped=capped,
                    trials=trials,
                    rejected_for_feasibility=rejected_for_feasibility,
                    stored_pairs=stored_pairs,
                    direction_seconds=direction_seconds,
                    line_search_seconds=line_search_seconds,
                    retraction_seconds=retraction_seconds,
                )
            )
            if observer is not None:
                observer(records[-1])
            break

        step = accepted.coordinates - coordinates
        point_started = time.perf_counter()
        next_point = jax.block_until_ready(evaluate(accepted.coordinates))
        point_evaluation_seconds = time.perf_counter() - point_started

        pair_started = time.perf_counter()
        metric, curvature, admitted, live_pairs = jax.block_until_ready(
            admit_pair(
                metric,
                step,
                next_point.projected_gradient - point.projected_gradient,
            )
        )
        pair_update_seconds = time.perf_counter() - pair_started
        stored_pairs = int(live_pairs)

        record = ProjectedLbfgsIteration(
            index=index,
            objective=float(point.objective),
            projected_gradient_inf=float(point.projected_gradient_inf),
            projected_gradient_norm=float(point.projected_gradient_norm),
            gradient_norm=float(point.gradient_norm),
            feasibility_inf=float(point.feasibility_inf),
            tangency_relative_residual=float(point.tangency_relative_residual),
            gram_reciprocal_condition=float(point.gram_reciprocal_condition),
            direction_norm=direction_norm,
            directional_derivative=directional_derivative,
            step_scale=step_scale,
            step_norm=float(jnp.linalg.norm(step)),
            step_norm_capped=bool(capped),
            line_search_trials=trials,
            rejected_for_feasibility=rejected_for_feasibility,
            retraction_corrections=int(accepted.corrections),
            candidate_objective=float(accepted.objective),
            candidate_feasibility_inf=float(accepted.feasibility_inf),
            curvature=float(curvature),
            pair_admitted=bool(admitted),
            stored_pairs=stored_pairs,
            direction_seconds=direction_seconds,
            line_search_seconds=line_search_seconds,
            retraction_seconds=retraction_seconds,
            point_evaluation_seconds=point_evaluation_seconds,
            pair_update_seconds=pair_update_seconds,
        )
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
    )


def _collapsed_record(
    *,
    index: int,
    point: ProjectedPoint,
    direction_norm: float,
    directional_derivative: float,
    step_scale: float,
    capped: bool,
    trials: int,
    rejected_for_feasibility: int,
    stored_pairs: int,
    direction_seconds: float,
    line_search_seconds: float,
    retraction_seconds: float,
) -> ProjectedLbfgsIteration:
    """Record the iteration whose line search found no acceptable step."""

    return ProjectedLbfgsIteration(
        index=index,
        objective=float(point.objective),
        projected_gradient_inf=float(point.projected_gradient_inf),
        projected_gradient_norm=float(point.projected_gradient_norm),
        gradient_norm=float(point.gradient_norm),
        feasibility_inf=float(point.feasibility_inf),
        tangency_relative_residual=float(point.tangency_relative_residual),
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
        stored_pairs=stored_pairs,
        direction_seconds=direction_seconds,
        line_search_seconds=line_search_seconds,
        retraction_seconds=retraction_seconds,
        point_evaluation_seconds=0.0,
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
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("tolerances and bounds must be finite and positive")
    if not math.isfinite(options.objective_target):
        raise ValueError("objective_target must be finite")


__all__ = (
    "ManifoldRetraction",
    "ProjectedLbfgsIteration",
    "ProjectedLbfgsOptions",
    "ProjectedLbfgsRun",
    "ProjectedLbfgsStatus",
    "ProjectedPoint",
    "evaluate_projected_point",
    "retract_to_manifold",
    "run_projected_lbfgs",
)
