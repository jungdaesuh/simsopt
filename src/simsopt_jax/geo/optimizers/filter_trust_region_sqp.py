"""Device-resident filter/trust-region SQP for equality-constrained JAX programs.

The module owns only optimizer mechanics. Callers provide a pure FP64 function
returning one scalar objective and one nonempty equality-constraint vector.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .dense_sqp import (
    JointValueConstraints,
    materialize_joint_vjp_rows,
    powell_damped_bfgs_update,
)

_STEIHAUG_RESIDUAL_TOLERANCE = 1.0e-12


class FilterTrustRegionSQPStatus(IntEnum):
    """Terminal status codes emitted by the prepared optimizer."""

    RUNNING = 0
    CONVERGED = 1
    RANK_DEFICIENT_GRAM = 2
    MULTIPLIER_PROJECTION_FAILED = 3
    BFGS_UPDATE_FAILED = 4
    OBJECTIVE_QUALITY_REJECTED = 5
    ITERATION_LIMIT = 6
    EVALUATION_LIMIT = 7
    RADIUS_LIMIT = 8


@dataclass(frozen=True, slots=True)
class FilterTrustRegionSQPOptions:
    """Immutable externally owned limits and numerical certificate tolerances."""

    maximum_iterations: int = 100
    maximum_joint_evaluations: int = 1200
    reverse_row_batch_width: int = 8
    objective_maximum: float = float("inf")
    feasibility_tolerance: float = 1.0e-10
    stationarity_tolerance: float = 1.0e-7
    gram_relative_residual_tolerance: float = 1.0e-10
    linear_solve_forward_error_tolerance: float = 1.0e-7
    multiplier_projection_relative_residual_tolerance: float = 1.0e-10
    initial_bfgs_identity_scale: float = 1.0
    curvature_fraction: float = 0.2
    maximum_consecutive_bfgs_resets: int = 2
    initial_trust_radius: float = 1.0
    minimum_trust_radius: float = 2.0**-20
    maximum_trust_radius: float = 8.0
    normal_radius_fraction: float = 0.8
    maximum_tangential_cg_iterations: int = 64
    filter_gamma_feasibility: float = 1.0e-4
    filter_gamma_objective: float = 1.0e-4
    objective_step_threshold: float = 1.0e-4
    acceptance_ratio: float = 0.1
    radius_shrink_ratio: float = 0.25
    expansion_ratio: float = 0.75
    radius_contraction: float = 0.25
    radius_expansion: float = 2.0
    boundary_fraction: float = 0.8
    tangency_relative_residual_tolerance: float = 1.0e-10


_DEFAULT_OPTIONS = FilterTrustRegionSQPOptions()


class FilterTrustRegionSQPHistory(NamedTuple):
    """Fixed-capacity per-attempt optimizer diagnostics."""

    objective: jax.Array
    feasibility_infinity_norm: jax.Array
    stationarity_infinity_norm: jax.Array
    normal_step_norm: jax.Array
    tangential_step_norm: jax.Array
    combined_step_norm: jax.Array
    radius: jax.Array
    selected_radius_index: jax.Array
    predicted_reduction: jax.Array
    actual_reduction: jax.Array
    reduction_ratio: jax.Array
    objective_type: jax.Array
    filter_accepted: jax.Array
    accepted: jax.Array
    multiplier_projection_relative_residual: jax.Array
    normal_relative_residual: jax.Array
    normal_forward_error_bound: jax.Array
    tangency_relative_residual: jax.Array
    bfgs_reset: jax.Array
    joint_evaluations: jax.Array
    status: jax.Array


class FilterTrustRegionSQPFailureCounters(NamedTuple):
    """Causal counts for every fail-closed candidate or state rejection."""

    factor: jax.Array
    nonfinite: jax.Array
    projection: jax.Array
    model: jax.Array
    filter: jax.Array
    radius: jax.Array
    budget: jax.Array


class FilterTrustRegionSQPResult(NamedTuple):
    """Device-array result of one prepared filter/trust-region SQP solve."""

    optimizer_coordinates: jax.Array
    multipliers: jax.Array
    bfgs_matrix: jax.Array
    objective: jax.Array
    constraints: jax.Array
    objective_gradient: jax.Array
    constraint_jacobian: jax.Array
    stationarity: jax.Array
    converged: jax.Array
    fatal: jax.Array
    failed: jax.Array
    status: jax.Array
    iterations: jax.Array
    accepted_iterations: jax.Array
    joint_evaluations: jax.Array
    derivative_builds: jax.Array
    radius: jax.Array
    final_normal_relative_residual: jax.Array
    final_normal_forward_error_bound: jax.Array
    final_tangency_relative_residual: jax.Array
    final_multiplier_projection_relative_residual: jax.Array
    final_multiplier_projection_forward_error_bound: jax.Array
    all_accepted_states_finite: jax.Array
    all_finite: jax.Array
    history: FilterTrustRegionSQPHistory
    failure_counters: FilterTrustRegionSQPFailureCounters


_PreparedRun = Callable[[jax.Array, jax.Array], FilterTrustRegionSQPResult]


@dataclass(frozen=True, slots=True)
class PreparedFilterTrustRegionSQP:
    """One fixed-shape compiled filter/trust-region SQP program."""

    coordinate_shape: tuple[int, ...]
    coordinate_dtype: str
    equality_count: int
    options: FilterTrustRegionSQPOptions
    _run_prepared: _PreparedRun = field(repr=False, compare=False)

    def run(
        self,
        x0: jax.Array,
        multipliers0: jax.Array | None = None,
    ) -> FilterTrustRegionSQPResult:
        """Run from primal and optional scaled-dual inputs matching preparation."""

        coordinates = jnp.asarray(x0)
        if coordinates.shape != self.coordinate_shape:
            raise ValueError(
                "filter/trust-region SQP coordinates must retain prepared shape "
                f"{self.coordinate_shape}, got {coordinates.shape}"
            )
        if str(coordinates.dtype) != self.coordinate_dtype:
            raise TypeError(
                "filter/trust-region SQP coordinates must retain prepared dtype "
                f"{self.coordinate_dtype}, got {coordinates.dtype}"
            )
        multipliers = (
            jnp.zeros((self.equality_count,), dtype=coordinates.dtype)
            if multipliers0 is None
            else jnp.asarray(multipliers0)
        )
        if multipliers.shape != (self.equality_count,):
            raise ValueError(
                "filter/trust-region SQP multipliers must have shape "
                f"({self.equality_count},), got {multipliers.shape}"
            )
        if multipliers.dtype != coordinates.dtype:
            raise TypeError(
                "filter/trust-region SQP primal and multiplier dtypes must match"
            )
        return self._run_prepared(coordinates, multipliers)


class _GramFactor(NamedTuple):
    factor: jax.Array
    reciprocal_condition: jax.Array
    valid: jax.Array


class _CandidateBatch(NamedTuple):
    coordinates: jax.Array
    objectives: jax.Array
    constraints: jax.Array
    normal_steps: jax.Array
    tangential_steps: jax.Array
    combined_steps: jax.Array
    predicted_objective_reduction: jax.Array
    predicted_feasibility_reduction: jax.Array
    actual_objective_reduction: jax.Array
    actual_feasibility_reduction: jax.Array
    reduction_ratio: jax.Array
    objective_type: jax.Array
    filter_accepted: jax.Array
    accepted: jax.Array
    finite: jax.Array
    normal_relative_residual: jax.Array
    normal_forward_error_bound: jax.Array
    tangency_relative_residual: jax.Array


class _State(NamedTuple):
    coordinates: jax.Array
    multipliers: jax.Array
    bfgs_matrix: jax.Array
    objective: jax.Array
    constraints: jax.Array
    objective_gradient: jax.Array
    constraint_jacobian: jax.Array
    stationarity: jax.Array
    radius: jax.Array
    status: jax.Array
    iterations: jax.Array
    accepted_iterations: jax.Array
    joint_evaluations: jax.Array
    derivative_builds: jax.Array
    consecutive_bfgs_resets: jax.Array
    filter_theta: jax.Array
    filter_objective: jax.Array
    filter_active: jax.Array
    final_normal_relative_residual: jax.Array
    final_normal_forward_error_bound: jax.Array
    final_tangency_relative_residual: jax.Array
    final_multiplier_projection_relative_residual: jax.Array
    final_multiplier_projection_forward_error_bound: jax.Array
    all_accepted_states_finite: jax.Array
    all_finite: jax.Array
    history: FilterTrustRegionSQPHistory
    failure_counters: FilterTrustRegionSQPFailureCounters


def _validate_options(options: FilterTrustRegionSQPOptions) -> None:
    if options.maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if options.maximum_joint_evaluations < 1:
        raise ValueError("maximum_joint_evaluations must be positive")
    if options.reverse_row_batch_width < 1:
        raise ValueError("reverse_row_batch_width must be positive")
    for name, tolerance in (
        ("feasibility_tolerance", options.feasibility_tolerance),
        ("stationarity_tolerance", options.stationarity_tolerance),
        ("gram_relative_residual_tolerance", options.gram_relative_residual_tolerance),
        (
            "linear_solve_forward_error_tolerance",
            options.linear_solve_forward_error_tolerance,
        ),
        (
            "multiplier_projection_relative_residual_tolerance",
            options.multiplier_projection_relative_residual_tolerance,
        ),
        (
            "tangency_relative_residual_tolerance",
            options.tangency_relative_residual_tolerance,
        ),
    ):
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if options.initial_bfgs_identity_scale <= 0.0 or not math.isfinite(
        options.initial_bfgs_identity_scale
    ):
        raise ValueError("initial_bfgs_identity_scale must be finite and positive")
    if not 0.0 < options.curvature_fraction < 1.0:
        raise ValueError("curvature_fraction must lie strictly between zero and one")
    if options.maximum_consecutive_bfgs_resets < 1:
        raise ValueError("maximum_consecutive_bfgs_resets must be positive")
    if options.maximum_tangential_cg_iterations < 1:
        raise ValueError("maximum_tangential_cg_iterations must be positive")
    if not (
        0.0
        < options.minimum_trust_radius
        <= options.initial_trust_radius
        <= options.maximum_trust_radius
    ):
        raise ValueError("trust radii must be positive and ordered")
    if not 0.0 < options.normal_radius_fraction < 1.0:
        raise ValueError(
            "normal_radius_fraction must lie strictly between zero and one"
        )
    if not 0.0 < options.filter_gamma_feasibility < 1.0:
        raise ValueError(
            "filter_gamma_feasibility must lie strictly between zero and one"
        )
    if options.filter_gamma_objective <= 0.0:
        raise ValueError("filter_gamma_objective must be positive")
    if options.objective_step_threshold <= 0.0:
        raise ValueError("objective_step_threshold must be positive")
    if not (
        0.0
        < options.acceptance_ratio
        < options.radius_shrink_ratio
        < options.expansion_ratio
        < 1.0
    ):
        raise ValueError(
            "acceptance, radius-shrink, and expansion ratios must be strictly ordered"
        )
    if not 0.0 < options.radius_contraction < 1.0:
        raise ValueError("radius_contraction must lie strictly between zero and one")
    if options.radius_expansion <= 1.0:
        raise ValueError("radius_expansion must exceed one")
    if not 0.0 < options.boundary_fraction <= 1.0:
        raise ValueError("boundary_fraction must lie in (0, 1]")


def _factor_gram(constraint_jacobian: jax.Array) -> _GramFactor:
    gram = constraint_jacobian @ constraint_jacobian.T
    gram = 0.5 * (gram + gram.T)
    factor = jnp.linalg.cholesky(gram)
    eigenvalues = jnp.linalg.eigvalsh(gram)
    largest = jnp.max(eigenvalues)
    reciprocal_condition = jnp.where(largest > 0.0, jnp.min(eigenvalues) / largest, 0.0)
    valid = (
        jnp.all(jnp.isfinite(factor))
        & jnp.all(jnp.diag(factor) > 0.0)
        & jnp.all(jnp.isfinite(eigenvalues))
        & (reciprocal_condition > 0.0)
    )
    return _GramFactor(
        factor=factor, reciprocal_condition=reciprocal_condition, valid=valid
    )


def _forward_error_bound(
    relative_residual: jax.Array, reciprocal_condition: jax.Array
) -> jax.Array:
    return jnp.where(
        reciprocal_condition > relative_residual,
        relative_residual / (reciprocal_condition - relative_residual),
        jnp.asarray(jnp.inf, dtype=relative_residual.dtype),
    )


def _relative_residual(
    matrix: jax.Array,
    solution: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    residual = matrix @ solution - right_hand_side
    denominator = jnp.maximum(
        jnp.asarray(1.0, dtype=matrix.dtype),
        jnp.linalg.norm(matrix, ord=jnp.inf) * jnp.linalg.norm(solution, ord=jnp.inf)
        + jnp.linalg.norm(right_hand_side, ord=jnp.inf),
    )
    return jnp.linalg.norm(residual, ord=jnp.inf) / denominator


def _boundary_root(
    offset: jax.Array, direction: jax.Array, radius: jax.Array
) -> jax.Array:
    quadratic = direction @ direction
    linear = 2.0 * (offset @ direction)
    constant = offset @ offset - radius * radius
    discriminant = jnp.maximum(linear * linear - 4.0 * quadratic * constant, 0.0)
    safe_quadratic = jnp.where(quadratic > 0.0, quadratic, 1.0)
    return jnp.maximum((-linear + jnp.sqrt(discriminant)) / (2.0 * safe_quadratic), 0.0)


def _normal_dogleg(
    constraint_jacobian: jax.Array,
    constraints: jax.Array,
    gram_factor: jax.Array,
    radius: jax.Array,
    normal_radius_fraction: float,
) -> tuple[jax.Array, jax.Array]:
    normal_rhs = jsp.linalg.cho_solve((gram_factor, True), constraints)
    newton_step = -constraint_jacobian.T @ normal_rhs
    gradient = constraint_jacobian.T @ constraints
    mapped_gradient = constraint_jacobian @ gradient
    gradient_norm_squared = gradient @ gradient
    mapped_norm_squared = mapped_gradient @ mapped_gradient
    cauchy_scale = jnp.where(
        mapped_norm_squared > 0.0,
        gradient_norm_squared / mapped_norm_squared,
        jnp.asarray(0.0, dtype=constraints.dtype),
    )
    cauchy_step = -cauchy_scale * gradient
    normal_radius = normal_radius_fraction * radius
    newton_norm = jnp.linalg.norm(newton_step)
    cauchy_norm = jnp.linalg.norm(cauchy_step)
    scaled_cauchy = cauchy_step * jnp.where(
        cauchy_norm > 0.0, normal_radius / cauchy_norm, 0.0
    )
    dogleg_direction = newton_step - cauchy_step
    dogleg_scale = _boundary_root(cauchy_step, dogleg_direction, normal_radius)
    dogleg_step = cauchy_step + jnp.minimum(dogleg_scale, 1.0) * dogleg_direction
    normal_step = jnp.where(
        newton_norm <= normal_radius,
        newton_step,
        jnp.where(cauchy_norm >= normal_radius, scaled_cauchy, dogleg_step),
    )
    normal_step = jnp.where(
        jnp.all(constraints == 0.0), jnp.zeros_like(normal_step), normal_step
    )
    gram = constraint_jacobian @ constraint_jacobian.T
    residual = _relative_residual(gram, normal_rhs, constraints)
    return normal_step, residual


class _SteihaugState(NamedTuple):
    tangential_step: jax.Array
    residual: jax.Array
    direction: jax.Array
    active: jax.Array


def _projected_steihaug_step(
    bfgs_matrix: jax.Array,
    constraint_jacobian: jax.Array,
    gram_factor: jax.Array,
    objective_gradient: jax.Array,
    normal_step: jax.Array,
    radius: jax.Array,
    maximum_iterations: int,
) -> tuple[jax.Array, jax.Array]:
    def project(vector: jax.Array) -> jax.Array:
        coefficients = jsp.linalg.cho_solve(
            (gram_factor, True), constraint_jacobian @ vector
        )
        return vector - constraint_jacobian.T @ coefficients

    initial_residual = project(objective_gradient + bfgs_matrix @ normal_step)
    initial = _SteihaugState(
        tangential_step=jnp.zeros_like(normal_step),
        residual=initial_residual,
        direction=-initial_residual,
        active=jnp.linalg.norm(initial_residual) > _STEIHAUG_RESIDUAL_TOLERANCE,
    )

    def iteration(_index: int, state: _SteihaugState) -> _SteihaugState:
        projected_matrix_direction = project(bfgs_matrix @ state.direction)
        curvature = state.direction @ projected_matrix_direction
        residual_squared = state.residual @ state.residual
        safe_curvature = jnp.where(curvature > 0.0, curvature, 1.0)
        alpha = residual_squared / safe_curvature
        unconstrained_step = state.tangential_step + alpha * state.direction
        crosses_boundary = jnp.linalg.norm(normal_step + unconstrained_step) >= radius
        hits_boundary = (curvature <= 0.0) | crosses_boundary
        tau = _boundary_root(
            normal_step + state.tangential_step, state.direction, radius
        )
        boundary_step = state.tangential_step + tau * state.direction
        next_step = jnp.where(hits_boundary, boundary_step, unconstrained_step)
        next_residual = state.residual + alpha * projected_matrix_direction
        next_residual_squared = next_residual @ next_residual
        beta = next_residual_squared / jnp.maximum(
            residual_squared, jnp.finfo(normal_step.dtype).tiny
        )
        next_direction = -next_residual + beta * state.direction
        next_active = (
            ~hits_boundary
            & jnp.isfinite(next_residual_squared)
            & (jnp.sqrt(next_residual_squared) > _STEIHAUG_RESIDUAL_TOLERANCE)
        )
        updated = _SteihaugState(next_step, next_residual, next_direction, next_active)
        return jax.tree.map(
            lambda new, old: jnp.where(state.active, new, old), updated, state
        )

    trip_count = min(normal_step.shape[0], maximum_iterations)
    solved = jax.lax.fori_loop(0, trip_count, iteration, initial)
    tangential_step = project(solved.tangential_step)
    tangency_residual = jnp.linalg.norm(
        constraint_jacobian @ tangential_step, ord=jnp.inf
    ) / jnp.maximum(
        jnp.asarray(1.0, dtype=normal_step.dtype),
        jnp.linalg.norm(tangential_step, ord=jnp.inf),
    )
    return tangential_step, tangency_residual


def _initial_history(
    maximum_iterations: int, dtype: jnp.dtype
) -> FilterTrustRegionSQPHistory:
    floating = lambda: jnp.full((maximum_iterations,), jnp.nan, dtype=dtype)
    integer = lambda value: jnp.full((maximum_iterations,), value, dtype=jnp.int32)
    return FilterTrustRegionSQPHistory(
        objective=floating(),
        feasibility_infinity_norm=floating(),
        stationarity_infinity_norm=floating(),
        normal_step_norm=floating(),
        tangential_step_norm=floating(),
        combined_step_norm=floating(),
        radius=floating(),
        selected_radius_index=integer(-1),
        predicted_reduction=floating(),
        actual_reduction=floating(),
        reduction_ratio=floating(),
        objective_type=integer(0),
        filter_accepted=integer(0),
        accepted=integer(0),
        multiplier_projection_relative_residual=floating(),
        normal_relative_residual=floating(),
        normal_forward_error_bound=floating(),
        tangency_relative_residual=floating(),
        bfgs_reset=integer(0),
        joint_evaluations=integer(0),
        status=integer(int(FilterTrustRegionSQPStatus.RUNNING)),
    )


def _zero_failure_counters() -> FilterTrustRegionSQPFailureCounters:
    zero = jnp.asarray(0, dtype=jnp.int32)
    return FilterTrustRegionSQPFailureCounters(zero, zero, zero, zero, zero, zero, zero)


def _terminal_status(
    objective: jax.Array,
    constraints: jax.Array,
    stationarity: jax.Array,
    options: FilterTrustRegionSQPOptions,
) -> jax.Array:
    finite = (
        jnp.isfinite(objective)
        & jnp.all(jnp.isfinite(constraints))
        & jnp.all(jnp.isfinite(stationarity))
    )
    satisfies_kkt = (
        finite
        & (jnp.linalg.norm(constraints, ord=jnp.inf) <= options.feasibility_tolerance)
        & (jnp.linalg.norm(stationarity, ord=jnp.inf) <= options.stationarity_tolerance)
    )
    return jnp.where(
        satisfies_kkt & (objective <= options.objective_maximum),
        int(FilterTrustRegionSQPStatus.CONVERGED),
        jnp.where(
            satisfies_kkt,
            int(FilterTrustRegionSQPStatus.OBJECTIVE_QUALITY_REJECTED),
            int(FilterTrustRegionSQPStatus.RUNNING),
        ),
    ).astype(jnp.int32)


def _build_program(
    joint_value_constraints: JointValueConstraints,
    options: FilterTrustRegionSQPOptions,
) -> Callable[[jax.Array, jax.Array], FilterTrustRegionSQPResult]:
    def program(x0: jax.Array, multipliers0: jax.Array) -> FilterTrustRegionSQPResult:
        initial_rows = materialize_joint_vjp_rows(
            joint_value_constraints, x0, batch_width=options.reverse_row_batch_width
        )
        dimension = x0.shape[0]
        initial_stationarity = (
            initial_rows.objective_gradient
            + initial_rows.constraint_jacobian.T @ multipliers0
        )
        requested_initial_status = _terminal_status(
            initial_rows.objective,
            initial_rows.constraints,
            initial_stationarity,
            options,
        )
        initial_factor = _factor_gram(initial_rows.constraint_jacobian)
        initial_gram = (
            initial_rows.constraint_jacobian @ initial_rows.constraint_jacobian.T
        )
        initial_normal_solution = jsp.linalg.cho_solve(
            (initial_factor.factor, True), initial_rows.constraints
        )
        initial_normal_residual = _relative_residual(
            initial_gram, initial_normal_solution, initial_rows.constraints
        )
        initial_normal_forward_error = _forward_error_bound(
            initial_normal_residual, initial_factor.reciprocal_condition
        )
        initial_projection_rhs = -(
            initial_rows.constraint_jacobian @ initial_rows.objective_gradient
        )
        initial_projected_multipliers = jsp.linalg.cho_solve(
            (initial_factor.factor, True), initial_projection_rhs
        )
        initial_projection_residual = _relative_residual(
            initial_gram, initial_projected_multipliers, initial_projection_rhs
        )
        initial_projection_forward_error = _forward_error_bound(
            initial_projection_residual, initial_factor.reciprocal_condition
        )
        initial_projection_valid = (
            initial_factor.valid
            & (initial_normal_residual <= options.gram_relative_residual_tolerance)
            & (
                initial_normal_forward_error
                < options.linear_solve_forward_error_tolerance
            )
            & (
                initial_projection_residual
                <= options.multiplier_projection_relative_residual_tolerance
            )
            & (
                initial_projection_forward_error
                < options.linear_solve_forward_error_tolerance
            )
        )
        initially_terminal = requested_initial_status != int(
            FilterTrustRegionSQPStatus.RUNNING
        )
        certified_initial_multipliers = jnp.where(
            initially_terminal, initial_projected_multipliers, multipliers0
        )
        certified_initial_stationarity = (
            initial_rows.objective_gradient
            + initial_rows.constraint_jacobian.T @ certified_initial_multipliers
        )
        certified_initial_status = _terminal_status(
            initial_rows.objective,
            initial_rows.constraints,
            certified_initial_stationarity,
            options,
        )
        initial_status = jnp.where(
            initially_terminal & ~initial_factor.valid,
            int(FilterTrustRegionSQPStatus.RANK_DEFICIENT_GRAM),
            jnp.where(
                initially_terminal & ~initial_projection_valid,
                int(FilterTrustRegionSQPStatus.MULTIPLIER_PROJECTION_FAILED),
                jnp.where(
                    initially_terminal,
                    certified_initial_status,
                    requested_initial_status,
                ),
            ),
        ).astype(jnp.int32)
        initial_finite = (
            jnp.all(jnp.isfinite(x0))
            & jnp.all(jnp.isfinite(multipliers0))
            & jnp.isfinite(initial_rows.objective)
            & jnp.all(jnp.isfinite(initial_rows.constraints))
            & jnp.all(jnp.isfinite(initial_rows.joint_rows))
        )
        filter_capacity = options.maximum_iterations
        state = _State(
            coordinates=x0,
            multipliers=certified_initial_multipliers,
            bfgs_matrix=options.initial_bfgs_identity_scale
            * jnp.eye(dimension, dtype=x0.dtype),
            objective=initial_rows.objective,
            constraints=initial_rows.constraints,
            objective_gradient=initial_rows.objective_gradient,
            constraint_jacobian=initial_rows.constraint_jacobian,
            stationarity=certified_initial_stationarity,
            radius=jnp.asarray(options.initial_trust_radius, dtype=x0.dtype),
            status=initial_status,
            iterations=jnp.asarray(0, dtype=jnp.int32),
            accepted_iterations=jnp.asarray(0, dtype=jnp.int32),
            joint_evaluations=jnp.asarray(1, dtype=jnp.int32),
            derivative_builds=jnp.asarray(1, dtype=jnp.int32),
            consecutive_bfgs_resets=jnp.asarray(0, dtype=jnp.int32),
            filter_theta=jnp.full((filter_capacity,), jnp.inf, dtype=x0.dtype),
            filter_objective=jnp.full((filter_capacity,), jnp.inf, dtype=x0.dtype),
            filter_active=jnp.zeros((filter_capacity,), dtype=jnp.bool_),
            final_normal_relative_residual=jnp.where(
                initially_terminal,
                initial_normal_residual,
                jnp.asarray(jnp.nan, dtype=x0.dtype),
            ),
            final_normal_forward_error_bound=jnp.where(
                initially_terminal,
                initial_normal_forward_error,
                jnp.asarray(jnp.nan, dtype=x0.dtype),
            ),
            final_tangency_relative_residual=jnp.where(
                initially_terminal,
                jnp.asarray(0.0, dtype=x0.dtype),
                jnp.asarray(jnp.nan, dtype=x0.dtype),
            ),
            final_multiplier_projection_relative_residual=jnp.where(
                initially_terminal,
                initial_projection_residual,
                jnp.asarray(jnp.nan, dtype=x0.dtype),
            ),
            final_multiplier_projection_forward_error_bound=jnp.where(
                initially_terminal,
                initial_projection_forward_error,
                jnp.asarray(jnp.nan, dtype=x0.dtype),
            ),
            all_accepted_states_finite=initial_finite,
            all_finite=initial_finite
            & (~initially_terminal | initial_projection_valid),
            history=_initial_history(options.maximum_iterations, x0.dtype),
            failure_counters=FilterTrustRegionSQPFailureCounters(
                factor=(initially_terminal & ~initial_factor.valid).astype(jnp.int32),
                nonfinite=jnp.asarray(0, dtype=jnp.int32),
                projection=(initially_terminal & ~initial_projection_valid).astype(
                    jnp.int32
                ),
                model=jnp.asarray(0, dtype=jnp.int32),
                filter=jnp.asarray(0, dtype=jnp.int32),
                radius=jnp.asarray(0, dtype=jnp.int32),
                budget=jnp.asarray(0, dtype=jnp.int32),
            ),
        )

        def continue_loop(current: _State) -> jax.Array:
            return (
                (current.status == int(FilterTrustRegionSQPStatus.RUNNING))
                & (current.iterations < options.maximum_iterations)
                & (current.joint_evaluations + 2 <= options.maximum_joint_evaluations)
            )

        def iteration(current: _State) -> _State:
            factor = _factor_gram(current.constraint_jacobian)
            normal_step, normal_residual = _normal_dogleg(
                current.constraint_jacobian,
                current.constraints,
                factor.factor,
                current.radius,
                options.normal_radius_fraction,
            )
            tangential_step, tangency_residual = _projected_steihaug_step(
                current.bfgs_matrix,
                current.constraint_jacobian,
                factor.factor,
                current.objective_gradient,
                normal_step,
                current.radius,
                options.maximum_tangential_cg_iterations,
            )
            combined_step = normal_step + tangential_step
            normal_forward_error_bound = _forward_error_bound(
                normal_residual, factor.reciprocal_condition
            )
            candidate_coordinates = current.coordinates + combined_step
            candidate_objective, candidate_constraints = joint_value_constraints(
                candidate_coordinates
            )
            predicted_objective = -(
                current.objective_gradient @ combined_step
                + 0.5 * combined_step @ current.bfgs_matrix @ combined_step
            )
            current_theta = jnp.linalg.norm(current.constraints, ord=2)
            linearized_constraints = (
                current.constraints + current.constraint_jacobian @ combined_step
            )
            predicted_feasibility = current_theta - jnp.linalg.norm(
                linearized_constraints, ord=2
            )
            candidate_theta = jnp.linalg.norm(candidate_constraints, ord=2)
            actual_objective = current.objective - candidate_objective
            actual_feasibility = current_theta - candidate_theta
            objective_type = (
                predicted_objective
                >= options.objective_step_threshold * current_theta**2
            )
            applicable_prediction = jnp.where(
                objective_type, predicted_objective, predicted_feasibility
            )
            applicable_actual = jnp.where(
                objective_type, actual_objective, actual_feasibility
            )
            ratio = applicable_actual / jnp.where(
                applicable_prediction > 0.0, applicable_prediction, 1.0
            )
            against_filter = (
                (
                    candidate_theta
                    <= (1.0 - options.filter_gamma_feasibility) * current.filter_theta
                )
                | (
                    candidate_objective
                    <= current.filter_objective
                    - options.filter_gamma_objective * current.filter_theta
                )
                | ~current.filter_active
            )
            filter_accepted = jnp.all(against_filter)
            finite = (
                jnp.all(jnp.isfinite(candidate_coordinates))
                & jnp.isfinite(candidate_objective)
                & jnp.all(jnp.isfinite(candidate_constraints))
                & jnp.all(jnp.isfinite(normal_step))
                & jnp.all(jnp.isfinite(tangential_step))
                & jnp.isfinite(applicable_prediction)
                & jnp.isfinite(applicable_actual)
                & jnp.isfinite(ratio)
                & jnp.isfinite(normal_residual)
                & jnp.isfinite(tangency_residual)
            )
            accepted = (
                factor.valid
                & finite
                & filter_accepted
                & (applicable_prediction > 0.0)
                & (ratio >= options.acceptance_ratio)
                & (normal_residual <= options.gram_relative_residual_tolerance)
                & (
                    normal_forward_error_bound
                    < options.linear_solve_forward_error_tolerance
                )
                & (tangency_residual <= options.tangency_relative_residual_tolerance)
            )
            selected = _CandidateBatch(
                coordinates=candidate_coordinates,
                objectives=candidate_objective,
                constraints=candidate_constraints,
                normal_steps=normal_step,
                tangential_steps=tangential_step,
                combined_steps=combined_step,
                predicted_objective_reduction=predicted_objective,
                predicted_feasibility_reduction=predicted_feasibility,
                actual_objective_reduction=actual_objective,
                actual_feasibility_reduction=actual_feasibility,
                reduction_ratio=ratio,
                objective_type=objective_type,
                filter_accepted=filter_accepted,
                accepted=accepted,
                finite=finite,
                normal_relative_residual=normal_residual,
                normal_forward_error_bound=normal_forward_error_bound,
                tangency_relative_residual=tangency_residual,
            )
            selected_index = jnp.asarray(0, dtype=jnp.int32)
            selected_radius = current.radius
            attempts_evaluations = jnp.asarray(1, dtype=jnp.int32)

            def accept_step() -> _State:
                next_rows = materialize_joint_vjp_rows(
                    joint_value_constraints,
                    selected.coordinates,
                    batch_width=options.reverse_row_batch_width,
                )
                next_factor = _factor_gram(next_rows.constraint_jacobian)
                next_gram = (
                    next_rows.constraint_jacobian @ next_rows.constraint_jacobian.T
                )
                projection_rhs = -(
                    next_rows.constraint_jacobian @ next_rows.objective_gradient
                )
                next_multipliers = jsp.linalg.cho_solve(
                    (next_factor.factor, True), projection_rhs
                )
                projection_residual = _relative_residual(
                    next_gram, next_multipliers, projection_rhs
                )
                projection_forward_error_bound = _forward_error_bound(
                    projection_residual, next_factor.reciprocal_condition
                )
                projection_valid = (
                    next_factor.valid
                    & jnp.all(jnp.isfinite(next_multipliers))
                    & jnp.isfinite(projection_residual)
                    & (
                        projection_residual
                        <= options.multiplier_projection_relative_residual_tolerance
                    )
                    & (
                        projection_forward_error_bound
                        < options.linear_solve_forward_error_tolerance
                    )
                )
                next_stationarity = (
                    next_rows.objective_gradient
                    + next_rows.constraint_jacobian.T @ next_multipliers
                )
                old_lagrangian_gradient = (
                    current.objective_gradient
                    + current.constraint_jacobian.T @ next_multipliers
                )
                bfgs = powell_damped_bfgs_update(
                    current.bfgs_matrix,
                    selected.combined_steps,
                    next_stationarity - old_lagrangian_gradient,
                    curvature_fraction=options.curvature_fraction,
                )
                accepted_finite = (
                    jnp.all(jnp.isfinite(selected.coordinates))
                    & jnp.isfinite(next_rows.objective)
                    & jnp.all(jnp.isfinite(next_rows.constraints))
                    & jnp.all(jnp.isfinite(next_rows.joint_rows))
                    & jnp.all(jnp.isfinite(next_multipliers))
                    & jnp.all(jnp.isfinite(next_stationarity))
                    & bfgs.all_finite
                )
                endpoint_status = _terminal_status(
                    next_rows.objective,
                    next_rows.constraints,
                    next_stationarity,
                    options,
                )
                consecutive_bfgs_resets = jnp.where(
                    bfgs.reset, current.consecutive_bfgs_resets + 1, 0
                )
                endpoint_status = jnp.where(
                    ~projection_valid,
                    int(FilterTrustRegionSQPStatus.MULTIPLIER_PROJECTION_FAILED),
                    jnp.where(
                        consecutive_bfgs_resets
                        >= options.maximum_consecutive_bfgs_resets,
                        int(FilterTrustRegionSQPStatus.BFGS_UPDATE_FAILED),
                        endpoint_status,
                    ),
                ).astype(jnp.int32)
                ratio_for_radius = selected.reduction_ratio
                boundary_active = (
                    jnp.linalg.norm(selected.combined_steps)
                    >= options.boundary_fraction * selected_radius
                )
                next_radius = jnp.where(
                    ratio_for_radius < options.radius_shrink_ratio,
                    options.radius_contraction * selected_radius,
                    jnp.where(
                        (ratio_for_radius > options.expansion_ratio) & boundary_active,
                        jnp.minimum(
                            options.radius_expansion * selected_radius,
                            options.maximum_trust_radius,
                        ),
                        selected_radius,
                    ),
                )
                history_index = current.iterations
                history = FilterTrustRegionSQPHistory(
                    objective=current.history.objective.at[history_index].set(
                        next_rows.objective
                    ),
                    feasibility_infinity_norm=current.history.feasibility_infinity_norm.at[
                        history_index
                    ].set(jnp.linalg.norm(next_rows.constraints, ord=jnp.inf)),
                    stationarity_infinity_norm=current.history.stationarity_infinity_norm.at[
                        history_index
                    ].set(jnp.linalg.norm(next_stationarity, ord=jnp.inf)),
                    normal_step_norm=current.history.normal_step_norm.at[
                        history_index
                    ].set(jnp.linalg.norm(selected.normal_steps)),
                    tangential_step_norm=current.history.tangential_step_norm.at[
                        history_index
                    ].set(jnp.linalg.norm(selected.tangential_steps)),
                    combined_step_norm=current.history.combined_step_norm.at[
                        history_index
                    ].set(jnp.linalg.norm(selected.combined_steps)),
                    radius=current.history.radius.at[history_index].set(
                        selected_radius
                    ),
                    selected_radius_index=current.history.selected_radius_index.at[
                        history_index
                    ].set(selected_index),
                    predicted_reduction=current.history.predicted_reduction.at[
                        history_index
                    ].set(
                        jnp.where(
                            selected.objective_type,
                            selected.predicted_objective_reduction,
                            selected.predicted_feasibility_reduction,
                        )
                    ),
                    actual_reduction=current.history.actual_reduction.at[
                        history_index
                    ].set(
                        jnp.where(
                            selected.objective_type,
                            selected.actual_objective_reduction,
                            selected.actual_feasibility_reduction,
                        )
                    ),
                    reduction_ratio=current.history.reduction_ratio.at[
                        history_index
                    ].set(selected.reduction_ratio),
                    objective_type=current.history.objective_type.at[history_index].set(
                        selected.objective_type.astype(jnp.int32)
                    ),
                    filter_accepted=current.history.filter_accepted.at[
                        history_index
                    ].set(selected.filter_accepted.astype(jnp.int32)),
                    accepted=current.history.accepted.at[history_index].set(1),
                    multiplier_projection_relative_residual=current.history.multiplier_projection_relative_residual.at[
                        history_index
                    ].set(projection_residual),
                    normal_relative_residual=current.history.normal_relative_residual.at[
                        history_index
                    ].set(selected.normal_relative_residual),
                    normal_forward_error_bound=current.history.normal_forward_error_bound.at[
                        history_index
                    ].set(selected.normal_forward_error_bound),
                    tangency_relative_residual=current.history.tangency_relative_residual.at[
                        history_index
                    ].set(selected.tangency_relative_residual),
                    bfgs_reset=current.history.bfgs_reset.at[history_index].set(
                        bfgs.reset.astype(jnp.int32)
                    ),
                    joint_evaluations=current.history.joint_evaluations.at[
                        history_index
                    ].set(current.joint_evaluations + attempts_evaluations + 1),
                    status=current.history.status.at[history_index].set(
                        endpoint_status
                    ),
                )
                insert_filter = ~selected.objective_type
                old_theta = jnp.linalg.norm(current.constraints, ord=2)
                dominated_by_old = (
                    current.filter_active
                    & (current.filter_theta >= old_theta)
                    & (current.filter_objective >= current.objective)
                )
                retained_filter_active = current.filter_active & ~dominated_by_old
                filter_index = jnp.argmax((~retained_filter_active).astype(jnp.int32))
                updated_filter_theta = current.filter_theta.at[filter_index].set(
                    old_theta
                )
                updated_filter_objective = current.filter_objective.at[
                    filter_index
                ].set(current.objective)
                updated_filter_active = retained_filter_active.at[filter_index].set(
                    True
                )
                return current._replace(
                    coordinates=selected.coordinates,
                    multipliers=next_multipliers,
                    bfgs_matrix=bfgs.matrix,
                    objective=next_rows.objective,
                    constraints=next_rows.constraints,
                    objective_gradient=next_rows.objective_gradient,
                    constraint_jacobian=next_rows.constraint_jacobian,
                    stationarity=next_stationarity,
                    radius=jnp.maximum(next_radius, options.minimum_trust_radius),
                    status=endpoint_status,
                    iterations=current.iterations + 1,
                    accepted_iterations=current.accepted_iterations + 1,
                    joint_evaluations=current.joint_evaluations
                    + attempts_evaluations
                    + 1,
                    derivative_builds=current.derivative_builds + 1,
                    consecutive_bfgs_resets=consecutive_bfgs_resets,
                    filter_theta=jnp.where(
                        insert_filter, updated_filter_theta, current.filter_theta
                    ),
                    filter_objective=jnp.where(
                        insert_filter,
                        updated_filter_objective,
                        current.filter_objective,
                    ),
                    filter_active=jnp.where(
                        insert_filter, updated_filter_active, current.filter_active
                    ),
                    final_normal_relative_residual=selected.normal_relative_residual,
                    final_normal_forward_error_bound=selected.normal_forward_error_bound,
                    final_tangency_relative_residual=selected.tangency_relative_residual,
                    final_multiplier_projection_relative_residual=projection_residual,
                    final_multiplier_projection_forward_error_bound=projection_forward_error_bound,
                    all_accepted_states_finite=current.all_accepted_states_finite
                    & accepted_finite,
                    all_finite=current.all_finite & accepted_finite & projection_valid,
                    history=history,
                    failure_counters=current.failure_counters._replace(
                        projection=current.failure_counters.projection
                        + (~projection_valid).astype(jnp.int32)
                    ),
                )

            def reject_step() -> _State:
                next_radius = options.radius_contraction * current.radius
                radius_failed = next_radius < options.minimum_trust_radius
                factor_failed = ~factor.valid
                status = jnp.where(
                    factor_failed,
                    int(FilterTrustRegionSQPStatus.RANK_DEFICIENT_GRAM),
                    jnp.where(
                        radius_failed,
                        int(FilterTrustRegionSQPStatus.RADIUS_LIMIT),
                        int(FilterTrustRegionSQPStatus.RUNNING),
                    ),
                ).astype(jnp.int32)
                history_index = current.iterations
                history = current.history._replace(
                    radius=current.history.radius.at[history_index].set(current.radius),
                    selected_radius_index=current.history.selected_radius_index.at[
                        history_index
                    ].set(-1),
                    filter_accepted=current.history.filter_accepted.at[
                        history_index
                    ].set(jnp.any(filter_accepted).astype(jnp.int32)),
                    accepted=current.history.accepted.at[history_index].set(0),
                    joint_evaluations=current.history.joint_evaluations.at[
                        history_index
                    ].set(current.joint_evaluations + attempts_evaluations),
                    status=current.history.status.at[history_index].set(status),
                )
                return current._replace(
                    radius=jnp.maximum(next_radius, options.minimum_trust_radius),
                    status=status,
                    iterations=current.iterations + 1,
                    joint_evaluations=current.joint_evaluations + attempts_evaluations,
                    all_finite=current.all_finite & factor.valid,
                    history=history,
                    failure_counters=FilterTrustRegionSQPFailureCounters(
                        factor=current.failure_counters.factor
                        + factor_failed.astype(jnp.int32),
                        nonfinite=current.failure_counters.nonfinite
                        + jnp.sum((~finite).astype(jnp.int32), dtype=jnp.int32),
                        projection=current.failure_counters.projection,
                        model=current.failure_counters.model
                        + jnp.sum(
                            (finite & (applicable_prediction <= 0.0)).astype(jnp.int32),
                            dtype=jnp.int32,
                        ),
                        filter=current.failure_counters.filter
                        + jnp.sum(
                            (finite & ~filter_accepted).astype(jnp.int32),
                            dtype=jnp.int32,
                        ),
                        radius=current.failure_counters.radius
                        + radius_failed.astype(jnp.int32),
                        budget=current.failure_counters.budget,
                    ),
                )

            return jax.lax.cond(accepted & factor.valid, accept_step, reject_step)

        state = jax.lax.while_loop(continue_loop, iteration, state)
        budget_exhausted = (state.status == int(FilterTrustRegionSQPStatus.RUNNING)) & (
            state.joint_evaluations + 2 > options.maximum_joint_evaluations
        )
        status = jnp.where(
            budget_exhausted,
            int(FilterTrustRegionSQPStatus.EVALUATION_LIMIT),
            jnp.where(
                (state.status == int(FilterTrustRegionSQPStatus.RUNNING))
                & (state.iterations >= options.maximum_iterations),
                int(FilterTrustRegionSQPStatus.ITERATION_LIMIT),
                state.status,
            ),
        ).astype(jnp.int32)
        failure_counters = state.failure_counters._replace(
            budget=state.failure_counters.budget + budget_exhausted.astype(jnp.int32)
        )
        converged = status == int(FilterTrustRegionSQPStatus.CONVERGED)
        fatal = (
            (status == int(FilterTrustRegionSQPStatus.RANK_DEFICIENT_GRAM))
            | (status == int(FilterTrustRegionSQPStatus.MULTIPLIER_PROJECTION_FAILED))
            | (status == int(FilterTrustRegionSQPStatus.BFGS_UPDATE_FAILED))
        )
        return FilterTrustRegionSQPResult(
            optimizer_coordinates=state.coordinates,
            multipliers=state.multipliers,
            bfgs_matrix=state.bfgs_matrix,
            objective=state.objective,
            constraints=state.constraints,
            objective_gradient=state.objective_gradient,
            constraint_jacobian=state.constraint_jacobian,
            stationarity=state.stationarity,
            converged=converged,
            fatal=fatal,
            failed=~converged,
            status=status,
            iterations=state.iterations,
            accepted_iterations=state.accepted_iterations,
            joint_evaluations=state.joint_evaluations,
            derivative_builds=state.derivative_builds,
            radius=state.radius,
            final_normal_relative_residual=state.final_normal_relative_residual,
            final_normal_forward_error_bound=state.final_normal_forward_error_bound,
            final_tangency_relative_residual=state.final_tangency_relative_residual,
            final_multiplier_projection_relative_residual=state.final_multiplier_projection_relative_residual,
            final_multiplier_projection_forward_error_bound=state.final_multiplier_projection_forward_error_bound,
            all_accepted_states_finite=state.all_accepted_states_finite,
            all_finite=state.all_finite,
            history=state.history,
            failure_counters=failure_counters,
        )

    return program


def prepare_filter_trust_region_sqp(
    joint_value_constraints: JointValueConstraints,
    x0: jax.Array,
    *,
    options: FilterTrustRegionSQPOptions = _DEFAULT_OPTIONS,
) -> PreparedFilterTrustRegionSQP:
    """Compile one fixed-shape callback-free FP64 filter/trust-region program."""

    _validate_options(options)
    coordinates = jnp.asarray(x0)
    if coordinates.ndim != 1:
        raise ValueError("filter/trust-region SQP coordinates must be a vector")
    if coordinates.dtype != jnp.float64:
        raise TypeError(
            "filter/trust-region SQP authoritative coordinates must use float64"
        )
    objective, constraints = jax.eval_shape(joint_value_constraints, coordinates)
    if objective.shape != ():
        raise ValueError(
            "filter/trust-region SQP objective callback output must be scalar"
        )
    if constraints.ndim != 1 or constraints.shape[0] < 1:
        raise ValueError(
            "filter/trust-region SQP constraints must be a nonempty vector"
        )
    if objective.dtype != coordinates.dtype or constraints.dtype != coordinates.dtype:
        raise TypeError(
            "filter/trust-region SQP callback outputs must match coordinate dtype"
        )
    multipliers = jnp.zeros(constraints.shape, dtype=coordinates.dtype)
    compiled = (
        jax.jit(_build_program(joint_value_constraints, options))
        .lower(coordinates, multipliers)
        .compile()
    )
    return PreparedFilterTrustRegionSQP(
        coordinate_shape=coordinates.shape,
        coordinate_dtype=str(coordinates.dtype),
        equality_count=constraints.shape[0],
        options=options,
        _run_prepared=compiled,
    )


__all__ = (
    "FilterTrustRegionSQPFailureCounters",
    "FilterTrustRegionSQPHistory",
    "FilterTrustRegionSQPOptions",
    "FilterTrustRegionSQPResult",
    "FilterTrustRegionSQPStatus",
    "PreparedFilterTrustRegionSQP",
    "prepare_filter_trust_region_sqp",
)
