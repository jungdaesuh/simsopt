"""Callback-free dense equality-constrained SQP for JAX programs.

The solver owns optimizer mechanics only.  Callers supply one pure function
returning a scalar objective and an equality-constraint vector in the
coordinates and scaling that they want the solver to use.
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

JointValueConstraints = Callable[[jax.Array], tuple[jax.Array, jax.Array]]


class DenseSQPStatus(IntEnum):
    """Terminal status codes emitted by :class:`PreparedDenseSQP`."""

    RUNNING = 0
    CONVERGED = 1
    RANK_DEFICIENT_OR_UNSTABLE_KKT = 2
    GLOBALIZATION_FAILED = 3
    BFGS_UPDATE_FAILED = 4
    OBJECTIVE_QUALITY_REJECTED = 5
    ITERATION_LIMIT = 6
    EVALUATION_LIMIT = 7


@dataclass(frozen=True, slots=True)
class DenseSQPOptions:
    """Immutable numerical policy for a prepared dense SQP program."""

    maximum_iterations: int = 100
    maximum_joint_evaluations: int = 1200
    reverse_row_batch_width: int = 8
    objective_maximum: float = float("inf")
    feasibility_tolerance: float = 1.0e-10
    stationarity_tolerance: float = 1.0e-7
    kkt_relative_residual_tolerance: float = 1.0e-10
    schur_relative_residual_tolerance: float = 1.0e-10
    kkt_forward_error_tolerance: float = 1.0e-7
    kkt_solution_scaled_residual_tolerance: float = 1.0e-10
    regularization_ladder: tuple[float, ...] = (
        0.0,
        1.0e-12,
        1.0e-10,
        1.0e-8,
        1.0e-6,
    )
    curvature_fraction: float = 0.2
    initial_bfgs_identity_scale: float = 1.0
    merit_initial: float = 1.0
    merit_multiplier_margin: float = 1.0
    armijo_coefficient: float = 1.0e-4
    candidate_steps: tuple[float, ...] = (
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
        0.00390625,
        0.001953125,
        0.0009765625,
    )
    maximum_consecutive_bfgs_resets: int = 2
    maximum_identity_retries: int = 1


_DEFAULT_DENSE_SQP_OPTIONS = DenseSQPOptions()


class JointVJPRows(NamedTuple):
    """One primal evaluation and its exact reverse derivative rows."""

    objective: jax.Array
    constraints: jax.Array
    objective_gradient: jax.Array
    constraint_jacobian: jax.Array
    joint_rows: jax.Array


class DenseSQPKKTStep(NamedTuple):
    """Certified solution of one regularized SQP KKT system."""

    primal_step: jax.Array
    multiplier_step: jax.Array
    valid: jax.Array
    selected_regularization: jax.Array
    kkt_relative_residual: jax.Array
    kkt_reciprocal_condition: jax.Array
    kkt_solution_scaled_residual: jax.Array
    kkt_forward_error_bound: jax.Array
    schur_relative_residual: jax.Array
    bfgs_cholesky_relative_pivot: jax.Array
    schur_cholesky_relative_pivot: jax.Array
    regularization_candidates_tested: jax.Array
    all_finite: jax.Array


class PowellBFGSUpdate(NamedTuple):
    """One Powell-damped full-BFGS update or an identity reset."""

    matrix: jax.Array
    reset: jax.Array
    theta: jax.Array
    all_finite: jax.Array


class DenseSQPHistory(NamedTuple):
    """Fixed-shape accepted-iteration diagnostics."""

    objective: jax.Array
    feasibility_infinity_norm: jax.Array
    stationarity_infinity_norm: jax.Array
    step_length: jax.Array
    kkt_relative_residual: jax.Array
    status: jax.Array


class DenseSQPResult(NamedTuple):
    """Device-array result of one prepared dense equality-SQP solve."""

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
    joint_evaluations: jax.Array
    derivative_builds: jax.Array
    kkt_solves: jax.Array
    line_search_evaluations: jax.Array
    rejected_nonfinite_trials: jax.Array
    bfgs_resets: jax.Array
    regularization_uses: jax.Array
    final_kkt_relative_residual: jax.Array
    final_kkt_reciprocal_condition: jax.Array
    final_kkt_solution_scaled_residual: jax.Array
    final_schur_relative_residual: jax.Array
    final_bfgs_cholesky_relative_pivot: jax.Array
    final_schur_cholesky_relative_pivot: jax.Array
    selected_regularization: jax.Array
    regularization_candidates_tested: jax.Array
    merit_penalty: jax.Array
    all_accepted_states_finite: jax.Array
    all_finite: jax.Array
    history: DenseSQPHistory


_PreparedRun = Callable[[jax.Array, jax.Array], DenseSQPResult]


@dataclass(frozen=True, slots=True)
class PreparedDenseSQP:
    """One fixed-shape compiled dense equality-SQP program."""

    coordinate_shape: tuple[int, ...]
    coordinate_dtype: str
    equality_count: int
    options: DenseSQPOptions
    _run_prepared: _PreparedRun = field(repr=False, compare=False)

    def run(
        self,
        x0: jax.Array,
        multipliers0: jax.Array | None = None,
    ) -> DenseSQPResult:
        """Run the compiled program from matching primal and scaled-dual inputs."""

        coordinates = jnp.asarray(x0)
        if coordinates.shape != self.coordinate_shape:
            raise ValueError(
                "dense SQP coordinates must retain prepared shape "
                f"{self.coordinate_shape}, got {coordinates.shape}"
            )
        if str(coordinates.dtype) != self.coordinate_dtype:
            raise TypeError(
                "dense SQP coordinates must retain prepared dtype "
                f"{self.coordinate_dtype}, got {coordinates.dtype}"
            )
        multipliers = (
            jnp.zeros((self.equality_count,), dtype=coordinates.dtype)
            if multipliers0 is None
            else jnp.asarray(multipliers0)
        )
        if multipliers.shape != (self.equality_count,):
            raise ValueError(
                "dense SQP multipliers must have shape "
                f"({self.equality_count},), got {multipliers.shape}"
            )
        if multipliers.dtype != coordinates.dtype:
            raise TypeError("dense SQP primal and multiplier dtypes must match")
        return self._run_prepared(coordinates, multipliers)


class _SolverState(NamedTuple):
    coordinates: jax.Array
    multipliers: jax.Array
    bfgs_matrix: jax.Array
    objective: jax.Array
    constraints: jax.Array
    objective_gradient: jax.Array
    constraint_jacobian: jax.Array
    stationarity: jax.Array
    merit_penalty: jax.Array
    status: jax.Array
    iterations: jax.Array
    joint_evaluations: jax.Array
    derivative_builds: jax.Array
    kkt_solves: jax.Array
    line_search_evaluations: jax.Array
    rejected_nonfinite_trials: jax.Array
    bfgs_resets: jax.Array
    consecutive_bfgs_resets: jax.Array
    regularization_uses: jax.Array
    final_kkt_relative_residual: jax.Array
    final_kkt_reciprocal_condition: jax.Array
    final_kkt_solution_scaled_residual: jax.Array
    final_schur_relative_residual: jax.Array
    final_bfgs_cholesky_relative_pivot: jax.Array
    final_schur_cholesky_relative_pivot: jax.Array
    selected_regularization: jax.Array
    regularization_candidates_tested: jax.Array
    all_accepted_states_finite: jax.Array
    all_finite: jax.Array
    history: DenseSQPHistory


class _LineSearchResult(NamedTuple):
    accepted: jax.Array
    coordinates: jax.Array
    objective: jax.Array
    constraints: jax.Array
    step_length: jax.Array
    evaluations: jax.Array
    rejected_nonfinite: jax.Array
    evaluation_limit: jax.Array


def _validate_options(options: DenseSQPOptions) -> None:
    if options.maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if options.maximum_joint_evaluations < 1:
        raise ValueError("maximum_joint_evaluations must be positive")
    if options.reverse_row_batch_width < 1:
        raise ValueError("reverse_row_batch_width must be positive")
    if not options.regularization_ladder:
        raise ValueError("regularization_ladder must not be empty")
    if any(delta < 0.0 for delta in options.regularization_ladder):
        raise ValueError("regularization values must be nonnegative")
    if any(not math.isfinite(delta) for delta in options.regularization_ladder):
        raise ValueError("regularization values must be finite")
    if options.feasibility_tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    if options.stationarity_tolerance < 0.0:
        raise ValueError("stationarity_tolerance must be nonnegative")
    if options.kkt_relative_residual_tolerance < 0.0:
        raise ValueError("kkt_relative_residual_tolerance must be nonnegative")
    if options.schur_relative_residual_tolerance < 0.0:
        raise ValueError("schur_relative_residual_tolerance must be nonnegative")
    if options.kkt_forward_error_tolerance < 0.0:
        raise ValueError("kkt_forward_error_tolerance must be nonnegative")
    if options.kkt_solution_scaled_residual_tolerance < 0.0:
        raise ValueError("kkt_solution_scaled_residual_tolerance must be nonnegative")
    if options.initial_bfgs_identity_scale <= 0.0:
        raise ValueError("initial_bfgs_identity_scale must be positive")
    if options.merit_multiplier_margin <= 0.0:
        raise ValueError("merit_multiplier_margin must be positive")
    if options.merit_initial <= 0.0:
        raise ValueError("merit_initial must be positive")
    if not options.candidate_steps or any(
        alpha <= 0.0 for alpha in options.candidate_steps
    ):
        raise ValueError("candidate_steps must contain positive values")
    if not 0.0 < options.curvature_fraction < 1.0:
        raise ValueError("curvature_fraction must lie strictly between zero and one")
    if not 0.0 < options.armijo_coefficient < 1.0:
        raise ValueError("armijo_coefficient must lie strictly between zero and one")
    if options.maximum_consecutive_bfgs_resets < 1:
        raise ValueError("maximum_consecutive_bfgs_resets must be positive")
    if options.maximum_identity_retries not in (0, 1):
        raise ValueError("maximum_identity_retries must be zero or one")


def materialize_joint_vjp_rows(
    joint_value_constraints: JointValueConstraints,
    coordinates: jax.Array,
    *,
    batch_width: int = 8,
) -> JointVJPRows:
    """Materialize objective and equality rows from one VJP primal traversal."""

    if batch_width < 1:
        raise ValueError("batch_width must be positive")

    def joined(values: jax.Array) -> jax.Array:
        objective, constraints = joint_value_constraints(values)
        return jnp.concatenate((jnp.reshape(objective, (1,)), constraints))

    joint_values, pullback = jax.vjp(joined, coordinates)
    output_count = joint_values.shape[0]
    basis = jnp.eye(output_count, dtype=joint_values.dtype)
    complete_count = output_count // batch_width
    complete_width = complete_count * batch_width

    def pull_rows(cotangents: jax.Array) -> jax.Array:
        return jax.vmap(lambda cotangent: pullback(cotangent)[0])(cotangents)

    row_parts: list[jax.Array] = []
    if complete_count:
        complete_basis = basis[:complete_width].reshape(
            (complete_count, batch_width, output_count)
        )
        complete_rows = jax.lax.map(pull_rows, complete_basis).reshape(
            (complete_width, coordinates.shape[0])
        )
        row_parts.append(complete_rows)
    if complete_width < output_count:
        row_parts.append(pull_rows(basis[complete_width:]))
    joint_rows = row_parts[0] if len(row_parts) == 1 else jnp.concatenate(row_parts)
    return JointVJPRows(
        objective=joint_values[0],
        constraints=joint_values[1:],
        objective_gradient=joint_rows[0],
        constraint_jacobian=joint_rows[1:],
        joint_rows=joint_rows,
    )


def _relative_linear_residual(
    matrix: jax.Array,
    solution: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    residual = matrix @ solution - right_hand_side
    numerator = jnp.linalg.norm(residual, ord=jnp.inf)
    denominator = jnp.maximum(
        jnp.asarray(1.0, dtype=matrix.dtype),
        jnp.linalg.norm(matrix, ord=jnp.inf) * jnp.linalg.norm(solution, ord=jnp.inf)
        + jnp.linalg.norm(right_hand_side, ord=jnp.inf),
    )
    return numerator / denominator


def _kkt_relative_residual(
    regularized_bfgs: jax.Array,
    constraint_jacobian: jax.Array,
    primal_step: jax.Array,
    multiplier_step: jax.Array,
    dual_residual: jax.Array,
    constraints: jax.Array,
) -> jax.Array:
    primal_residual = (
        regularized_bfgs @ primal_step
        + constraint_jacobian.T @ multiplier_step
        + dual_residual
    )
    constraint_residual = constraint_jacobian @ primal_step + constraints
    numerator = jnp.maximum(
        jnp.linalg.norm(primal_residual, ord=jnp.inf),
        jnp.linalg.norm(constraint_residual, ord=jnp.inf),
    )
    primal_row_sums = jnp.sum(jnp.abs(regularized_bfgs), axis=1) + jnp.sum(
        jnp.abs(constraint_jacobian), axis=0
    )
    constraint_row_sums = jnp.sum(jnp.abs(constraint_jacobian), axis=1)
    matrix_norm = jnp.maximum(jnp.max(primal_row_sums), jnp.max(constraint_row_sums))
    solution_norm = jnp.maximum(
        jnp.linalg.norm(primal_step, ord=jnp.inf),
        jnp.linalg.norm(multiplier_step, ord=jnp.inf),
    )
    right_hand_side_norm = jnp.maximum(
        jnp.linalg.norm(dual_residual, ord=jnp.inf),
        jnp.linalg.norm(constraints, ord=jnp.inf),
    )
    denominator = jnp.maximum(
        jnp.asarray(1.0, dtype=regularized_bfgs.dtype),
        matrix_norm * solution_norm + right_hand_side_norm,
    )
    return numerator / denominator


def solve_dense_sqp_kkt(
    bfgs_matrix: jax.Array,
    constraint_jacobian: jax.Array,
    dual_residual: jax.Array,
    constraints: jax.Array,
    *,
    regularization_ladder: tuple[float, ...] = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6),
    relative_residual_tolerance: float = 1.0e-10,
    schur_relative_residual_tolerance: float = 1.0e-10,
    kkt_forward_error_tolerance: float = 1.0e-7,
    kkt_solution_scaled_residual_tolerance: float = 1.0e-10,
) -> DenseSQPKKTStep:
    """Solve the first certified Cholesky/Schur SQP system in the ladder."""

    dtype = bfgs_matrix.dtype
    dimension = bfgs_matrix.shape[0]
    equality_count = constraint_jacobian.shape[0]
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    initial = DenseSQPKKTStep(
        primal_step=jnp.full((dimension,), nan, dtype=dtype),
        multiplier_step=jnp.full((equality_count,), nan, dtype=dtype),
        valid=jnp.asarray(False),
        selected_regularization=nan,
        kkt_relative_residual=nan,
        kkt_reciprocal_condition=nan,
        kkt_solution_scaled_residual=nan,
        kkt_forward_error_bound=nan,
        schur_relative_residual=nan,
        bfgs_cholesky_relative_pivot=nan,
        schur_cholesky_relative_pivot=nan,
        regularization_candidates_tested=jnp.asarray(0, dtype=jnp.int32),
        all_finite=jnp.asarray(False),
    )
    identity = jnp.eye(dimension, dtype=dtype)
    kkt_zeros = jnp.zeros((equality_count, equality_count), dtype=dtype)

    def candidate(delta: jax.Array) -> DenseSQPKKTStep:
        symmetric_bfgs = 0.5 * (bfgs_matrix + bfgs_matrix.T)
        regularized = symmetric_bfgs + delta * identity
        factor = jnp.linalg.cholesky(regularized)
        solve_b = lambda rhs: jsp.linalg.cho_solve((factor, True), rhs)
        inverse_times_at = solve_b(constraint_jacobian.T)
        inverse_times_dual = solve_b(dual_residual)
        raw_schur = constraint_jacobian @ inverse_times_at
        schur = 0.5 * (raw_schur + raw_schur.T)
        schur_factor = jnp.linalg.cholesky(schur)
        schur_rhs = constraints - constraint_jacobian @ inverse_times_dual
        multiplier_step = jsp.linalg.cho_solve((schur_factor, True), schur_rhs)
        primal_step = -solve_b(dual_residual + constraint_jacobian.T @ multiplier_step)
        kkt_residual = _kkt_relative_residual(
            regularized,
            constraint_jacobian,
            primal_step,
            multiplier_step,
            dual_residual,
            constraints,
        )
        kkt_matrix = jnp.block(
            [[regularized, constraint_jacobian.T], [constraint_jacobian, kkt_zeros]]
        )
        kkt_right_hand_side = -jnp.concatenate((dual_residual, constraints))
        kkt_solution = jnp.concatenate((primal_step, multiplier_step))
        kkt_eigenvalues = jnp.linalg.eigvalsh(kkt_matrix)
        kkt_eigenvalue_magnitudes = jnp.abs(kkt_eigenvalues)
        kkt_sigma_maximum = jnp.max(kkt_eigenvalue_magnitudes)
        kkt_reciprocal_condition = jnp.where(
            kkt_sigma_maximum > 0.0,
            jnp.min(kkt_eigenvalue_magnitudes) / kkt_sigma_maximum,
            jnp.asarray(0.0, dtype=dtype),
        )
        kkt_residual_norm_two = jnp.linalg.norm(
            kkt_matrix @ kkt_solution - kkt_right_hand_side, ord=2
        )
        kkt_scaled_residual_denominator = kkt_sigma_maximum * jnp.linalg.norm(
            kkt_solution, ord=2
        )
        kkt_solution_scaled_residual = jnp.where(
            kkt_scaled_residual_denominator > 0.0,
            kkt_residual_norm_two / kkt_scaled_residual_denominator,
            jnp.where(
                kkt_residual_norm_two == 0.0,
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
            ),
        )
        kkt_forward_error_bound = jnp.where(
            kkt_reciprocal_condition > kkt_solution_scaled_residual,
            kkt_solution_scaled_residual
            / (kkt_reciprocal_condition - kkt_solution_scaled_residual),
            jnp.asarray(jnp.inf, dtype=dtype),
        )
        schur_residual = _relative_linear_residual(schur, multiplier_step, schur_rhs)
        bfgs_factor_diagonal = jnp.diag(factor)
        schur_factor_diagonal = jnp.diag(schur_factor)
        bfgs_relative_pivot = jnp.min(jnp.abs(bfgs_factor_diagonal)) / jnp.max(
            jnp.abs(bfgs_factor_diagonal)
        )
        schur_relative_pivot = jnp.min(jnp.abs(schur_factor_diagonal)) / jnp.max(
            jnp.abs(schur_factor_diagonal)
        )
        finite = (
            jnp.all(jnp.isfinite(factor))
            & jnp.all(jnp.isfinite(schur_factor))
            & jnp.all(jnp.isfinite(primal_step))
            & jnp.all(jnp.isfinite(multiplier_step))
            & jnp.isfinite(kkt_residual)
            & jnp.all(jnp.isfinite(kkt_eigenvalues))
            & jnp.isfinite(kkt_reciprocal_condition)
            & jnp.isfinite(kkt_solution_scaled_residual)
            & jnp.isfinite(kkt_forward_error_bound)
            & jnp.isfinite(schur_residual)
            & jnp.isfinite(bfgs_relative_pivot)
            & jnp.isfinite(schur_relative_pivot)
        )
        valid = (
            finite
            & (kkt_residual <= relative_residual_tolerance)
            & (kkt_reciprocal_condition > kkt_solution_scaled_residual)
            & (kkt_solution_scaled_residual <= kkt_solution_scaled_residual_tolerance)
            & (kkt_forward_error_bound < kkt_forward_error_tolerance)
            & (schur_residual <= schur_relative_residual_tolerance)
        )
        return DenseSQPKKTStep(
            primal_step=primal_step,
            multiplier_step=multiplier_step,
            valid=valid,
            selected_regularization=delta,
            kkt_relative_residual=kkt_residual,
            kkt_reciprocal_condition=kkt_reciprocal_condition,
            kkt_solution_scaled_residual=kkt_solution_scaled_residual,
            kkt_forward_error_bound=kkt_forward_error_bound,
            schur_relative_residual=schur_residual,
            bfgs_cholesky_relative_pivot=bfgs_relative_pivot,
            schur_cholesky_relative_pivot=schur_relative_pivot,
            regularization_candidates_tested=jnp.asarray(1, dtype=jnp.int32),
            all_finite=finite,
        )

    def select(
        selected: DenseSQPKKTStep, delta: jax.Array
    ) -> tuple[DenseSQPKKTStep, None]:
        tested = selected.regularization_candidates_tested + jnp.asarray(
            ~selected.valid, dtype=jnp.int32
        )
        trial = jax.lax.cond(selected.valid, lambda: selected, lambda: candidate(delta))
        trial = trial._replace(regularization_candidates_tested=tested)
        return trial, None

    selected, _ = jax.lax.scan(
        select, initial, jnp.asarray(regularization_ladder, dtype=dtype)
    )
    return selected


def powell_damped_bfgs_update(
    bfgs_matrix: jax.Array,
    step: jax.Array,
    lagrangian_gradient_difference: jax.Array,
    *,
    curvature_fraction: float = 0.2,
) -> PowellBFGSUpdate:
    """Apply the frozen Powell-damped full-BFGS formula or reset to identity."""

    matrix_step = bfgs_matrix @ step
    quadratic_curvature = step @ matrix_step
    secant_curvature = step @ lagrangian_gradient_difference
    fraction = jnp.asarray(curvature_fraction, dtype=bfgs_matrix.dtype)
    theta = jnp.where(
        secant_curvature >= fraction * quadratic_curvature,
        jnp.asarray(1.0, dtype=bfgs_matrix.dtype),
        (1.0 - fraction)
        * quadratic_curvature
        / (quadratic_curvature - secant_curvature),
    )
    damped_difference = (
        theta * lagrangian_gradient_difference + (1.0 - theta) * matrix_step
    )
    damped_curvature = step @ damped_difference
    candidate = (
        bfgs_matrix
        - jnp.outer(matrix_step, matrix_step) / quadratic_curvature
        + jnp.outer(damped_difference, damped_difference) / damped_curvature
    )
    candidate = 0.5 * (candidate + candidate.T)
    valid = (
        jnp.isfinite(quadratic_curvature)
        & jnp.isfinite(secant_curvature)
        & jnp.isfinite(theta)
        & jnp.isfinite(damped_curvature)
        & (quadratic_curvature > 0.0)
        & (damped_curvature > 0.0)
        & jnp.all(jnp.isfinite(candidate))
    )
    returned_matrix = jnp.where(
        valid, candidate, jnp.eye(step.shape[0], dtype=step.dtype)
    )
    return PowellBFGSUpdate(
        matrix=returned_matrix,
        reset=~valid,
        theta=jnp.where(valid, theta, jnp.asarray(jnp.nan, dtype=step.dtype)),
        all_finite=jnp.all(jnp.isfinite(returned_matrix)),
    )


def _line_search(
    joint_value_constraints: JointValueConstraints,
    state: _SolverState,
    kkt_step: DenseSQPKKTStep,
    penalty: jax.Array,
    options: DenseSQPOptions,
) -> _LineSearchResult:
    dtype = state.coordinates.dtype
    current_merit = state.objective + penalty * jnp.linalg.norm(
        state.constraints, ord=1
    )
    linearized_constraints = (
        state.constraints + state.constraint_jacobian @ kkt_step.primal_step
    )
    merit_derivative = state.objective_gradient @ kkt_step.primal_step + penalty * (
        jnp.linalg.norm(linearized_constraints, ord=1)
        - jnp.linalg.norm(state.constraints, ord=1)
    )
    descent = jnp.isfinite(merit_derivative) & (merit_derivative < 0.0)
    initial = _LineSearchResult(
        accepted=jnp.asarray(False),
        coordinates=state.coordinates,
        objective=state.objective,
        constraints=state.constraints,
        step_length=jnp.asarray(0.0, dtype=dtype),
        evaluations=jnp.asarray(0, dtype=jnp.int32),
        rejected_nonfinite=jnp.asarray(0, dtype=jnp.int32),
        evaluation_limit=jnp.asarray(False),
    )

    def try_candidate(result: _LineSearchResult, alpha: jax.Array) -> _LineSearchResult:
        trial_coordinates = state.coordinates + alpha * kkt_step.primal_step
        trial_objective, trial_constraints = joint_value_constraints(trial_coordinates)
        finite = jnp.isfinite(trial_objective) & jnp.all(
            jnp.isfinite(trial_constraints)
        )
        trial_merit = trial_objective + penalty * jnp.linalg.norm(
            trial_constraints, ord=1
        )
        accepted = finite & (
            trial_merit
            <= current_merit + options.armijo_coefficient * alpha * merit_derivative
        )
        return _LineSearchResult(
            accepted=accepted,
            coordinates=jnp.where(accepted, trial_coordinates, result.coordinates),
            objective=jnp.where(accepted, trial_objective, result.objective),
            constraints=jnp.where(accepted, trial_constraints, result.constraints),
            step_length=jnp.where(accepted, alpha, result.step_length),
            evaluations=result.evaluations + 1,
            rejected_nonfinite=result.rejected_nonfinite
            + jnp.asarray(~finite, dtype=jnp.int32),
            evaluation_limit=result.evaluation_limit,
        )

    def scan_step(
        result: _LineSearchResult, alpha: jax.Array
    ) -> tuple[_LineSearchResult, None]:
        used = state.joint_evaluations + result.evaluations
        has_budget = used < options.maximum_joint_evaluations - 1
        should_try = descent & ~result.accepted & has_budget
        updated = jax.lax.cond(
            should_try, lambda: try_candidate(result, alpha), lambda: result
        )
        updated = updated._replace(
            evaluation_limit=updated.evaluation_limit | (~result.accepted & ~has_budget)
        )
        return updated, None

    result, _ = jax.lax.scan(
        scan_step, initial, jnp.asarray(options.candidate_steps, dtype=dtype)
    )
    return result


def _initial_history(maximum_iterations: int, dtype: jnp.dtype) -> DenseSQPHistory:
    return DenseSQPHistory(
        objective=jnp.full((maximum_iterations,), jnp.nan, dtype=dtype),
        feasibility_infinity_norm=jnp.full((maximum_iterations,), jnp.nan, dtype=dtype),
        stationarity_infinity_norm=jnp.full(
            (maximum_iterations,), jnp.nan, dtype=dtype
        ),
        step_length=jnp.full((maximum_iterations,), jnp.nan, dtype=dtype),
        kkt_relative_residual=jnp.full((maximum_iterations,), jnp.nan, dtype=dtype),
        status=jnp.full(
            (maximum_iterations,), int(DenseSQPStatus.RUNNING), dtype=jnp.int32
        ),
    )


def _terminal_status(
    coordinates: jax.Array,
    multipliers: jax.Array,
    objective: jax.Array,
    constraints: jax.Array,
    stationarity: jax.Array,
    options: DenseSQPOptions,
) -> jax.Array:
    finite = (
        jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(multipliers))
        & jnp.isfinite(objective)
        & jnp.all(jnp.isfinite(constraints))
        & jnp.all(jnp.isfinite(stationarity))
    )
    kkt_satisfied = (
        finite
        & (jnp.linalg.norm(constraints, ord=jnp.inf) <= options.feasibility_tolerance)
        & (jnp.linalg.norm(stationarity, ord=jnp.inf) <= options.stationarity_tolerance)
    )
    return jnp.where(
        kkt_satisfied & (objective <= options.objective_maximum),
        int(DenseSQPStatus.CONVERGED),
        jnp.where(
            kkt_satisfied,
            int(DenseSQPStatus.OBJECTIVE_QUALITY_REJECTED),
            int(DenseSQPStatus.RUNNING),
        ),
    ).astype(jnp.int32)


def _build_program(
    joint_value_constraints: JointValueConstraints,
    options: DenseSQPOptions,
) -> Callable[[jax.Array, jax.Array], DenseSQPResult]:
    def program(x0: jax.Array, multipliers0: jax.Array) -> DenseSQPResult:
        initial_rows = materialize_joint_vjp_rows(
            joint_value_constraints,
            x0,
            batch_width=options.reverse_row_batch_width,
        )
        dimension = x0.shape[0]
        initial_stationarity = (
            initial_rows.objective_gradient
            + initial_rows.constraint_jacobian.T @ multipliers0
        )
        initial_status = _terminal_status(
            x0,
            multipliers0,
            initial_rows.objective,
            initial_rows.constraints,
            initial_stationarity,
            options,
        )
        initial_all_finite = (
            jnp.all(jnp.isfinite(x0))
            & jnp.all(jnp.isfinite(multipliers0))
            & jnp.isfinite(initial_rows.objective)
            & jnp.all(jnp.isfinite(initial_rows.constraints))
            & jnp.all(jnp.isfinite(initial_rows.joint_rows))
            & jnp.all(jnp.isfinite(initial_stationarity))
        )
        state = _SolverState(
            coordinates=x0,
            multipliers=multipliers0,
            bfgs_matrix=options.initial_bfgs_identity_scale
            * jnp.eye(dimension, dtype=x0.dtype),
            objective=initial_rows.objective,
            constraints=initial_rows.constraints,
            objective_gradient=initial_rows.objective_gradient,
            constraint_jacobian=initial_rows.constraint_jacobian,
            stationarity=initial_stationarity,
            merit_penalty=jnp.asarray(options.merit_initial, dtype=x0.dtype),
            status=initial_status,
            iterations=jnp.asarray(0, dtype=jnp.int32),
            joint_evaluations=jnp.asarray(1, dtype=jnp.int32),
            derivative_builds=jnp.asarray(1, dtype=jnp.int32),
            kkt_solves=jnp.asarray(0, dtype=jnp.int32),
            line_search_evaluations=jnp.asarray(0, dtype=jnp.int32),
            rejected_nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
            bfgs_resets=jnp.asarray(0, dtype=jnp.int32),
            consecutive_bfgs_resets=jnp.asarray(0, dtype=jnp.int32),
            regularization_uses=jnp.asarray(0, dtype=jnp.int32),
            final_kkt_relative_residual=jnp.asarray(jnp.nan, dtype=x0.dtype),
            final_kkt_reciprocal_condition=jnp.asarray(jnp.nan, dtype=x0.dtype),
            final_kkt_solution_scaled_residual=jnp.asarray(jnp.nan, dtype=x0.dtype),
            final_schur_relative_residual=jnp.asarray(jnp.nan, dtype=x0.dtype),
            final_bfgs_cholesky_relative_pivot=jnp.asarray(jnp.nan, dtype=x0.dtype),
            final_schur_cholesky_relative_pivot=jnp.asarray(jnp.nan, dtype=x0.dtype),
            selected_regularization=jnp.asarray(jnp.nan, dtype=x0.dtype),
            regularization_candidates_tested=jnp.asarray(0, dtype=jnp.int32),
            all_accepted_states_finite=initial_all_finite,
            all_finite=initial_all_finite,
            history=_initial_history(options.maximum_iterations, x0.dtype),
        )

        def continue_loop(current: _SolverState) -> jax.Array:
            return (
                (current.status == int(DenseSQPStatus.RUNNING))
                & (current.iterations < options.maximum_iterations)
                & (current.joint_evaluations < options.maximum_joint_evaluations)
            )

        def iteration(current: _SolverState) -> _SolverState:
            dual_residual = current.stationarity
            primary_step = solve_dense_sqp_kkt(
                current.bfgs_matrix,
                current.constraint_jacobian,
                dual_residual,
                current.constraints,
                regularization_ladder=options.regularization_ladder,
                relative_residual_tolerance=options.kkt_relative_residual_tolerance,
                schur_relative_residual_tolerance=options.schur_relative_residual_tolerance,
                kkt_forward_error_tolerance=options.kkt_forward_error_tolerance,
                kkt_solution_scaled_residual_tolerance=options.kkt_solution_scaled_residual_tolerance,
            )
            primary_penalty = jnp.maximum(
                current.merit_penalty,
                jnp.linalg.norm(
                    current.multipliers + primary_step.multiplier_step, ord=jnp.inf
                )
                + options.merit_multiplier_margin,
            )
            primary_search = jax.lax.cond(
                primary_step.valid,
                lambda: _line_search(
                    joint_value_constraints,
                    current,
                    primary_step,
                    primary_penalty,
                    options,
                ),
                lambda: _LineSearchResult(
                    accepted=jnp.asarray(False),
                    coordinates=current.coordinates,
                    objective=current.objective,
                    constraints=current.constraints,
                    step_length=jnp.asarray(0.0, dtype=x0.dtype),
                    evaluations=jnp.asarray(0, dtype=jnp.int32),
                    rejected_nonfinite=jnp.asarray(0, dtype=jnp.int32),
                    evaluation_limit=jnp.asarray(False),
                ),
            )
            retry_needed = (
                primary_step.valid
                & ~primary_search.accepted
                & ~primary_search.evaluation_limit
                & (options.maximum_identity_retries == 1)
            )
            retry_step = jax.lax.cond(
                retry_needed,
                lambda: solve_dense_sqp_kkt(
                    jnp.eye(dimension, dtype=x0.dtype),
                    current.constraint_jacobian,
                    dual_residual,
                    current.constraints,
                    regularization_ladder=options.regularization_ladder,
                    relative_residual_tolerance=options.kkt_relative_residual_tolerance,
                    schur_relative_residual_tolerance=options.schur_relative_residual_tolerance,
                    kkt_forward_error_tolerance=options.kkt_forward_error_tolerance,
                    kkt_solution_scaled_residual_tolerance=options.kkt_solution_scaled_residual_tolerance,
                ),
                lambda: primary_step,
            )
            retry_penalty = jnp.maximum(
                current.merit_penalty,
                jnp.linalg.norm(
                    current.multipliers + retry_step.multiplier_step, ord=jnp.inf
                )
                + options.merit_multiplier_margin,
            )
            state_after_primary_counts = current._replace(
                joint_evaluations=current.joint_evaluations
                + primary_search.evaluations,
            )
            retry_search_executed = retry_needed & retry_step.valid
            retry_search = jax.lax.cond(
                retry_search_executed,
                lambda: _line_search(
                    joint_value_constraints,
                    state_after_primary_counts,
                    retry_step,
                    retry_penalty,
                    options,
                ),
                lambda: _LineSearchResult(
                    accepted=jnp.asarray(False),
                    coordinates=current.coordinates,
                    objective=current.objective,
                    constraints=current.constraints,
                    step_length=jnp.asarray(0.0, dtype=x0.dtype),
                    evaluations=jnp.asarray(0, dtype=jnp.int32),
                    rejected_nonfinite=jnp.asarray(0, dtype=jnp.int32),
                    evaluation_limit=jnp.asarray(False),
                ),
            )
            use_retry = retry_needed
            chosen_step = jax.tree.map(
                lambda retry, primary: jnp.where(use_retry, retry, primary),
                retry_step,
                primary_step,
            )
            chosen_search = jax.tree.map(
                lambda retry, primary: jnp.where(use_retry, retry, primary),
                retry_search,
                primary_search,
            )
            chosen_penalty = jnp.where(use_retry, retry_penalty, primary_penalty)
            total_search_evaluations = primary_search.evaluations + jnp.where(
                retry_search_executed, retry_search.evaluations, 0
            )
            total_nonfinite = primary_search.rejected_nonfinite + jnp.where(
                retry_search_executed, retry_search.rejected_nonfinite, 0
            )
            total_kkt_solves = jnp.asarray(1, dtype=jnp.int32) + jnp.asarray(
                use_retry, dtype=jnp.int32
            )
            total_regularization_candidates = (
                primary_step.regularization_candidates_tested
                + jnp.where(use_retry, retry_step.regularization_candidates_tested, 0)
            )
            regularization_use_count = jnp.asarray(
                primary_step.valid & (primary_step.selected_regularization > 0.0),
                dtype=jnp.int32,
            ) + jnp.asarray(
                use_retry
                & retry_step.valid
                & (retry_step.selected_regularization > 0.0),
                dtype=jnp.int32,
            )
            accepted = chosen_step.valid & chosen_search.accepted
            evaluation_limit = (
                primary_search.evaluation_limit
                | (retry_search_executed & retry_search.evaluation_limit)
                | (
                    current.joint_evaluations + total_search_evaluations
                    >= options.maximum_joint_evaluations
                )
            )

            def accept_step() -> _SolverState:
                next_multipliers = (
                    current.multipliers
                    + chosen_search.step_length * chosen_step.multiplier_step
                )
                next_rows = materialize_joint_vjp_rows(
                    joint_value_constraints,
                    chosen_search.coordinates,
                    batch_width=options.reverse_row_batch_width,
                )
                old_lagrangian_gradient = (
                    current.objective_gradient
                    + current.constraint_jacobian.T @ next_multipliers
                )
                next_stationarity = (
                    next_rows.objective_gradient
                    + next_rows.constraint_jacobian.T @ next_multipliers
                )
                bfgs = powell_damped_bfgs_update(
                    jnp.where(
                        use_retry,
                        jnp.eye(dimension, dtype=x0.dtype),
                        current.bfgs_matrix,
                    ),
                    chosen_search.coordinates - current.coordinates,
                    next_stationarity - old_lagrangian_gradient,
                    curvature_fraction=options.curvature_fraction,
                )
                consecutive_resets = jnp.where(
                    bfgs.reset, current.consecutive_bfgs_resets + 1, 0
                )
                endpoint_status = _terminal_status(
                    chosen_search.coordinates,
                    next_multipliers,
                    next_rows.objective,
                    next_rows.constraints,
                    next_stationarity,
                    options,
                )
                accepted_state_finite = (
                    jnp.all(jnp.isfinite(chosen_search.coordinates))
                    & jnp.all(jnp.isfinite(next_multipliers))
                    & jnp.isfinite(next_rows.objective)
                    & jnp.all(jnp.isfinite(next_rows.constraints))
                    & jnp.all(jnp.isfinite(next_rows.joint_rows))
                    & jnp.all(jnp.isfinite(next_stationarity))
                )
                endpoint_status = jnp.where(
                    consecutive_resets >= options.maximum_consecutive_bfgs_resets,
                    int(DenseSQPStatus.BFGS_UPDATE_FAILED),
                    endpoint_status,
                ).astype(jnp.int32)
                history_index = current.iterations
                history = DenseSQPHistory(
                    objective=current.history.objective.at[history_index].set(
                        next_rows.objective
                    ),
                    feasibility_infinity_norm=current.history.feasibility_infinity_norm.at[
                        history_index
                    ].set(jnp.linalg.norm(next_rows.constraints, ord=jnp.inf)),
                    stationarity_infinity_norm=current.history.stationarity_infinity_norm.at[
                        history_index
                    ].set(jnp.linalg.norm(next_stationarity, ord=jnp.inf)),
                    step_length=current.history.step_length.at[history_index].set(
                        chosen_search.step_length
                    ),
                    kkt_relative_residual=current.history.kkt_relative_residual.at[
                        history_index
                    ].set(chosen_step.kkt_relative_residual),
                    status=current.history.status.at[history_index].set(
                        endpoint_status
                    ),
                )
                return current._replace(
                    coordinates=chosen_search.coordinates,
                    multipliers=next_multipliers,
                    bfgs_matrix=bfgs.matrix,
                    objective=next_rows.objective,
                    constraints=next_rows.constraints,
                    objective_gradient=next_rows.objective_gradient,
                    constraint_jacobian=next_rows.constraint_jacobian,
                    stationarity=next_stationarity,
                    merit_penalty=chosen_penalty,
                    status=endpoint_status,
                    iterations=current.iterations + 1,
                    joint_evaluations=current.joint_evaluations
                    + total_search_evaluations
                    + 1,
                    derivative_builds=current.derivative_builds + 1,
                    kkt_solves=current.kkt_solves + total_kkt_solves,
                    line_search_evaluations=current.line_search_evaluations
                    + total_search_evaluations,
                    rejected_nonfinite_trials=current.rejected_nonfinite_trials
                    + total_nonfinite,
                    bfgs_resets=current.bfgs_resets
                    + jnp.asarray(bfgs.reset, dtype=jnp.int32),
                    consecutive_bfgs_resets=consecutive_resets,
                    regularization_uses=current.regularization_uses
                    + regularization_use_count,
                    final_kkt_relative_residual=chosen_step.kkt_relative_residual,
                    final_kkt_reciprocal_condition=chosen_step.kkt_reciprocal_condition,
                    final_kkt_solution_scaled_residual=chosen_step.kkt_solution_scaled_residual,
                    final_schur_relative_residual=chosen_step.schur_relative_residual,
                    final_bfgs_cholesky_relative_pivot=chosen_step.bfgs_cholesky_relative_pivot,
                    final_schur_cholesky_relative_pivot=chosen_step.schur_cholesky_relative_pivot,
                    selected_regularization=chosen_step.selected_regularization,
                    regularization_candidates_tested=current.regularization_candidates_tested
                    + total_regularization_candidates,
                    all_accepted_states_finite=current.all_accepted_states_finite
                    & accepted_state_finite,
                    all_finite=current.all_finite
                    & accepted_state_finite
                    & chosen_step.all_finite
                    & bfgs.all_finite,
                    history=history,
                )

            def reject_step() -> _SolverState:
                status = jnp.where(
                    ~chosen_step.valid,
                    int(DenseSQPStatus.RANK_DEFICIENT_OR_UNSTABLE_KKT),
                    jnp.where(
                        evaluation_limit,
                        int(DenseSQPStatus.EVALUATION_LIMIT),
                        int(DenseSQPStatus.GLOBALIZATION_FAILED),
                    ),
                ).astype(jnp.int32)
                return current._replace(
                    status=status,
                    joint_evaluations=current.joint_evaluations
                    + total_search_evaluations,
                    kkt_solves=current.kkt_solves + total_kkt_solves,
                    line_search_evaluations=current.line_search_evaluations
                    + total_search_evaluations,
                    rejected_nonfinite_trials=current.rejected_nonfinite_trials
                    + total_nonfinite,
                    regularization_uses=current.regularization_uses
                    + regularization_use_count,
                    final_kkt_relative_residual=chosen_step.kkt_relative_residual,
                    final_kkt_reciprocal_condition=chosen_step.kkt_reciprocal_condition,
                    final_kkt_solution_scaled_residual=chosen_step.kkt_solution_scaled_residual,
                    final_schur_relative_residual=chosen_step.schur_relative_residual,
                    final_bfgs_cholesky_relative_pivot=chosen_step.bfgs_cholesky_relative_pivot,
                    final_schur_cholesky_relative_pivot=chosen_step.schur_cholesky_relative_pivot,
                    selected_regularization=chosen_step.selected_regularization,
                    regularization_candidates_tested=current.regularization_candidates_tested
                    + total_regularization_candidates,
                    all_finite=current.all_finite & chosen_step.all_finite,
                )

            return jax.lax.cond(accepted, accept_step, reject_step)

        state = jax.lax.while_loop(continue_loop, iteration, state)
        status = jnp.where(
            (state.status == int(DenseSQPStatus.RUNNING))
            & (state.joint_evaluations >= options.maximum_joint_evaluations),
            int(DenseSQPStatus.EVALUATION_LIMIT),
            jnp.where(
                (state.status == int(DenseSQPStatus.RUNNING))
                & (state.iterations >= options.maximum_iterations),
                int(DenseSQPStatus.ITERATION_LIMIT),
                state.status,
            ),
        ).astype(jnp.int32)
        final_telemetry_finite = jax.lax.cond(
            state.kkt_solves == 0,
            lambda: jnp.asarray(True),
            lambda: (
                jnp.isfinite(state.final_kkt_relative_residual)
                & jnp.isfinite(state.final_kkt_reciprocal_condition)
                & jnp.isfinite(state.final_kkt_solution_scaled_residual)
                & jnp.isfinite(state.final_schur_relative_residual)
                & jnp.isfinite(state.final_bfgs_cholesky_relative_pivot)
                & jnp.isfinite(state.final_schur_cholesky_relative_pivot)
                & jnp.isfinite(state.selected_regularization)
            ),
        )
        authoritative_finite = (
            state.all_accepted_states_finite
            & state.all_finite
            & final_telemetry_finite
            & jnp.all(jnp.isfinite(state.coordinates))
            & jnp.all(jnp.isfinite(state.multipliers))
            & jnp.all(jnp.isfinite(state.bfgs_matrix))
        )
        status = jnp.where(
            (status == int(DenseSQPStatus.CONVERGED)) & ~authoritative_finite,
            int(DenseSQPStatus.RANK_DEFICIENT_OR_UNSTABLE_KKT),
            status,
        ).astype(jnp.int32)
        converged = status == int(DenseSQPStatus.CONVERGED)
        fatal = (
            (status == int(DenseSQPStatus.RANK_DEFICIENT_OR_UNSTABLE_KKT))
            | (status == int(DenseSQPStatus.GLOBALIZATION_FAILED))
            | (status == int(DenseSQPStatus.BFGS_UPDATE_FAILED))
        )
        return DenseSQPResult(
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
            failed=fatal,
            status=status,
            iterations=state.iterations,
            joint_evaluations=state.joint_evaluations,
            derivative_builds=state.derivative_builds,
            kkt_solves=state.kkt_solves,
            line_search_evaluations=state.line_search_evaluations,
            rejected_nonfinite_trials=state.rejected_nonfinite_trials,
            bfgs_resets=state.bfgs_resets,
            regularization_uses=state.regularization_uses,
            final_kkt_relative_residual=state.final_kkt_relative_residual,
            final_kkt_reciprocal_condition=state.final_kkt_reciprocal_condition,
            final_kkt_solution_scaled_residual=state.final_kkt_solution_scaled_residual,
            final_schur_relative_residual=state.final_schur_relative_residual,
            final_bfgs_cholesky_relative_pivot=state.final_bfgs_cholesky_relative_pivot,
            final_schur_cholesky_relative_pivot=state.final_schur_cholesky_relative_pivot,
            selected_regularization=state.selected_regularization,
            regularization_candidates_tested=state.regularization_candidates_tested,
            merit_penalty=state.merit_penalty,
            all_accepted_states_finite=state.all_accepted_states_finite,
            all_finite=state.all_finite & final_telemetry_finite,
            history=state.history,
        )

    return program


def prepare_dense_sqp(
    joint_value_constraints: JointValueConstraints,
    x0: jax.Array,
    *,
    options: DenseSQPOptions = _DEFAULT_DENSE_SQP_OPTIONS,
) -> PreparedDenseSQP:
    """Compile one fixed-shape callback-free dense equality-SQP program."""

    _validate_options(options)
    coordinates = jnp.asarray(x0)
    if coordinates.ndim != 1:
        raise ValueError("dense SQP coordinates must be a vector")
    if coordinates.dtype != jnp.float64:
        raise TypeError("dense SQP authoritative coordinates must use float64")
    objective, constraints = jax.eval_shape(joint_value_constraints, coordinates)
    if objective.shape != ():
        raise ValueError("dense SQP objective callback output must be scalar")
    if constraints.ndim != 1 or constraints.shape[0] < 1:
        raise ValueError(
            "dense SQP constraints callback output must be a nonempty vector"
        )
    if objective.dtype != coordinates.dtype or constraints.dtype != coordinates.dtype:
        raise TypeError("dense SQP callback outputs must match coordinate dtype")
    multipliers = jnp.zeros(constraints.shape, dtype=coordinates.dtype)
    compiled = (
        jax.jit(_build_program(joint_value_constraints, options))
        .lower(coordinates, multipliers)
        .compile()
    )
    return PreparedDenseSQP(
        coordinate_shape=coordinates.shape,
        coordinate_dtype=str(coordinates.dtype),
        equality_count=constraints.shape[0],
        options=options,
        _run_prepared=compiled,
    )


__all__ = (
    "DenseSQPHistory",
    "DenseSQPKKTStep",
    "DenseSQPOptions",
    "DenseSQPResult",
    "DenseSQPStatus",
    "JointVJPRows",
    "PowellBFGSUpdate",
    "PreparedDenseSQP",
    "materialize_joint_vjp_rows",
    "powell_damped_bfgs_update",
    "prepare_dense_sqp",
    "solve_dense_sqp_kkt",
)
