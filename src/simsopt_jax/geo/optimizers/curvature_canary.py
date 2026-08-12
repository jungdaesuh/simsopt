"""One-step dense-curvature A/B diagnostic for equality-constrained problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from .dense_sqp import materialize_joint_vjp_rows
from .linear_solve import (
    _dense_matrix_condition_estimate_with_telemetry,
    _relative_residual_1_norm,
)

JointValueConstraints = Callable[[jax.Array], tuple[jax.Array, jax.Array]]


class CurvatureCanaryEndpoint(NamedTuple):
    coordinates: jax.Array
    multipliers: jax.Array
    objective: jax.Array
    constraints: jax.Array
    stationarity: jax.Array
    scaled_feasibility_inf: jax.Array
    scaled_stationarity_inf: jax.Array
    raw_direction: jax.Array
    applied_step: jax.Array
    raw_direction_norm: jax.Array
    applied_step_norm: jax.Array
    kkt_relative_residual: jax.Array
    kkt_condition_estimate: jax.Array
    kkt_forward_error_bound: jax.Array
    multiplier_projection_relative_residual: jax.Array
    multiplier_projection_reciprocal_condition: jax.Array
    multiplier_projection_forward_error_bound: jax.Array
    correction_relative_residual: jax.Array
    correction_forward_error_bound: jax.Array
    all_finite: jax.Array


class DenseCurvatureCanaryResult(NamedTuple):
    initial: CurvatureCanaryEndpoint
    identity: CurvatureCanaryEndpoint
    exact: CurvatureCanaryEndpoint
    exact_hessian: jax.Array
    exact_hessian_symmetry_relative_defect: jax.Array
    exact_hessian_action_relative_defect: jax.Array
    both_variants_usable: jax.Array
    exact_scaled_stationarity_improved: jax.Array
    all_finite: jax.Array


class _KKTDirection(NamedTuple):
    primal: jax.Array
    dual: jax.Array
    relative_residual: jax.Array
    condition_estimate: jax.Array
    forward_error_bound: jax.Array
    all_finite: jax.Array


class EqualityMultiplierProjection(NamedTuple):
    multipliers: jax.Array
    relative_residual: jax.Array
    reciprocal_condition: jax.Array
    forward_error_bound: jax.Array
    all_finite: jax.Array


class _GramFactor(NamedTuple):
    matrix: jax.Array
    cholesky: jax.Array
    reciprocal_condition: jax.Array
    all_finite: jax.Array


def _relative_inf_residual(
    matrix: jax.Array,
    solution: jax.Array,
    residual: jax.Array,
    reference: jax.Array,
) -> jax.Array:
    numerator = jnp.linalg.norm(residual, ord=jnp.inf)
    denominator = jnp.maximum(
        jnp.asarray(jnp.finfo(reference.dtype).tiny, dtype=reference.dtype),
        jnp.linalg.norm(matrix, ord=jnp.inf) * jnp.linalg.norm(solution, ord=jnp.inf)
        + jnp.linalg.norm(reference, ord=jnp.inf),
    )
    return numerator / denominator


def _forward_error_bound(
    relative_residual: jax.Array,
    reciprocal_condition: jax.Array,
) -> jax.Array:
    return jnp.where(
        reciprocal_condition > relative_residual,
        relative_residual / (reciprocal_condition - relative_residual),
        jnp.asarray(jnp.inf, dtype=relative_residual.dtype),
    )


def _estimated_forward_error_bound(
    relative_residual: jax.Array,
    condition_estimate: jax.Array,
) -> jax.Array:
    scaled = condition_estimate * relative_residual
    return jnp.where(
        scaled < 1.0,
        scaled / (1.0 - scaled),
        jnp.asarray(jnp.inf, dtype=relative_residual.dtype),
    )


def _factor_gram(constraint_jacobian: jax.Array) -> _GramFactor:
    gram = constraint_jacobian @ constraint_jacobian.T
    gram = 0.5 * (gram + gram.T)
    cholesky = jnp.linalg.cholesky(gram)
    eigenvalues = jnp.linalg.eigvalsh(gram)
    largest = jnp.max(eigenvalues)
    reciprocal_condition = jnp.where(
        largest > 0.0,
        jnp.min(eigenvalues) / largest,
        jnp.asarray(0.0, dtype=gram.dtype),
    )
    all_finite = (
        jnp.all(jnp.isfinite(cholesky))
        & jnp.all(jnp.diag(cholesky) > 0.0)
        & jnp.all(jnp.isfinite(eigenvalues))
        & jnp.isfinite(reciprocal_condition)
        & (reciprocal_condition > 0.0)
    )
    return _GramFactor(
        matrix=gram,
        cholesky=cholesky,
        reciprocal_condition=reciprocal_condition,
        all_finite=all_finite,
    )


def _solve_gram(
    factor: _GramFactor,
    right_hand_side: jax.Array,
) -> EqualityMultiplierProjection:
    solution = jsp_linalg.cho_solve((factor.cholesky, True), right_hand_side)
    first_residual = right_hand_side - factor.matrix @ solution
    solution = solution + jsp_linalg.cho_solve((factor.cholesky, True), first_residual)
    residual = factor.matrix @ solution - right_hand_side
    relative_residual = _relative_inf_residual(
        factor.matrix,
        solution,
        residual,
        right_hand_side,
    )
    forward_error_bound = _forward_error_bound(
        relative_residual,
        factor.reciprocal_condition,
    )
    return EqualityMultiplierProjection(
        multipliers=solution,
        relative_residual=relative_residual,
        reciprocal_condition=factor.reciprocal_condition,
        forward_error_bound=forward_error_bound,
        all_finite=(
            factor.all_finite
            & jnp.all(jnp.isfinite(solution))
            & jnp.isfinite(relative_residual)
            & jnp.isfinite(forward_error_bound)
        ),
    )


def project_equality_multipliers(
    objective_gradient: jax.Array,
    constraint_jacobian: jax.Array,
) -> EqualityMultiplierProjection:
    """Return a condition-certified minimum-stationarity multiplier."""

    right_hand_side = -(constraint_jacobian @ objective_gradient)
    return _solve_gram(_factor_gram(constraint_jacobian), right_hand_side)


def materialize_exact_lagrangian_hessian(
    lagrangian: Callable[[jax.Array], jax.Array],
    coordinates: jax.Array,
    *,
    batch_width: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Materialize one exact Hessian from a linearized gradient in exact tails."""

    if batch_width < 1:
        raise ValueError("batch_width must be positive")

    _gradient, hessian_vector_product = jax.linearize(
        jax.grad(lagrangian),
        coordinates,
    )
    dimension = coordinates.shape[0]
    basis = jnp.eye(dimension, dtype=coordinates.dtype)
    complete_count = dimension // batch_width
    complete_width = complete_count * batch_width

    def apply_batch(tangents: jax.Array) -> jax.Array:
        return jax.vmap(hessian_vector_product)(tangents)

    parts: list[jax.Array] = []
    if complete_count:
        complete_basis = basis[:complete_width].reshape(
            (complete_count, batch_width, dimension)
        )
        complete_columns = jax.lax.map(apply_batch, complete_basis).reshape(
            (complete_width, dimension)
        )
        parts.append(complete_columns)
    if complete_width < dimension:
        parts.append(apply_batch(basis[complete_width:]))
    columns = parts[0] if len(parts) == 1 else jnp.concatenate(parts)
    dense = jnp.swapaxes(columns, 0, 1)
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=dense.dtype),
        jnp.linalg.norm(dense, ord=jnp.inf),
    )
    symmetry_defect = jnp.linalg.norm(dense - dense.T, ord=jnp.inf) / scale
    symmetric = 0.5 * (dense + dense.T)

    probe = jnp.linspace(-1.0, 1.0, dimension, dtype=coordinates.dtype)
    probe = probe / jnp.linalg.norm(probe)
    direct_action = hessian_vector_product(probe)
    dense_action = symmetric @ probe
    action_defect = _relative_inf_residual(
        symmetric,
        probe,
        dense_action - direct_action,
        direct_action,
    )
    return symmetric, symmetry_defect, action_defect


def solve_dense_primal_dual_direction(
    curvature: jax.Array,
    constraint_jacobian: jax.Array,
    stationarity: jax.Array,
    constraints: jax.Array,
) -> _KKTDirection:
    """Solve one dense equality-KKT direction without assuming positive curvature."""

    coordinate_count = curvature.shape[0]
    equality_count = constraint_jacobian.shape[0]
    zero_block = jnp.zeros(
        (equality_count, equality_count),
        dtype=curvature.dtype,
    )
    kkt_matrix = jnp.block(
        [
            [curvature, constraint_jacobian.T],
            [constraint_jacobian, zero_block],
        ]
    )
    right_hand_side = -jnp.concatenate((stationarity, constraints))
    factors = jsp_linalg.lu_factor(kkt_matrix)
    solution = jsp_linalg.lu_solve(factors, right_hand_side)
    first_residual = right_hand_side - kkt_matrix @ solution
    solution = solution + jsp_linalg.lu_solve(factors, first_residual)
    residual = kkt_matrix @ solution - right_hand_side
    relative_residual = _relative_residual_1_norm(residual, right_hand_side)
    condition_estimate, _, _ = _dense_matrix_condition_estimate_with_telemetry(
        kkt_matrix,
        lu_piv=factors,
        transpose_operator=False,
    )
    forward_error_bound = _estimated_forward_error_bound(
        relative_residual,
        condition_estimate,
    )
    return _KKTDirection(
        primal=solution[:coordinate_count],
        dual=solution[coordinate_count:],
        relative_residual=relative_residual,
        condition_estimate=condition_estimate,
        forward_error_bound=forward_error_bound,
        all_finite=(
            jnp.all(jnp.isfinite(kkt_matrix))
            & jnp.all(jnp.isfinite(solution))
            & jnp.isfinite(relative_residual)
            & jnp.isfinite(condition_estimate)
            & (condition_estimate >= 1.0)
            & jnp.isfinite(forward_error_bound)
        ),
    )


def _cap_step(direction: jax.Array, trust_radius: float) -> jax.Array:
    direction_norm = jnp.linalg.norm(direction)
    radius = jnp.asarray(trust_radius, dtype=direction.dtype)
    scale = jnp.minimum(
        jnp.asarray(1.0, dtype=direction.dtype),
        radius / jnp.maximum(direction_norm, jnp.finfo(direction.dtype).tiny),
    )
    return scale * direction


def run_dense_curvature_canary(
    joint_value_constraints: JointValueConstraints,
    initial_coordinates: jax.Array,
    *,
    trust_radius: float = 1.0 / 64.0,
    hessian_batch_width: int = 1,
    feasibility_tolerance: float = 1.0e-10,
    linear_residual_tolerance: float = 1.0e-10,
) -> DenseCurvatureCanaryResult:
    """Compare identity and exact-Lagrangian curvature in one corrected step."""

    initial_rows = materialize_joint_vjp_rows(
        joint_value_constraints,
        initial_coordinates,
        batch_width=8,
    )
    initial_gram_factor = _factor_gram(initial_rows.constraint_jacobian)
    initial_projection = _solve_gram(
        initial_gram_factor,
        -(initial_rows.constraint_jacobian @ initial_rows.objective_gradient),
    )
    initial_multipliers = initial_projection.multipliers
    initial_stationarity = (
        initial_rows.objective_gradient
        + initial_rows.constraint_jacobian.T @ initial_multipliers
    )

    def lagrangian(coordinates: jax.Array) -> jax.Array:
        objective, constraints = joint_value_constraints(coordinates)
        return objective + jnp.vdot(initial_multipliers, constraints)

    exact_hessian, symmetry_defect, action_defect = (
        materialize_exact_lagrangian_hessian(
            lagrangian,
            initial_coordinates,
            batch_width=hessian_batch_width,
        )
    )
    identity_hessian = jnp.eye(
        initial_coordinates.shape[0],
        dtype=initial_coordinates.dtype,
    )
    identity_direction = solve_dense_primal_dual_direction(
        identity_hessian,
        initial_rows.constraint_jacobian,
        initial_stationarity,
        initial_rows.constraints,
    )
    exact_direction = solve_dense_primal_dual_direction(
        exact_hessian,
        initial_rows.constraint_jacobian,
        initial_stationarity,
        initial_rows.constraints,
    )

    def corrected_endpoint(
        direction: _KKTDirection,
    ) -> CurvatureCanaryEndpoint:
        applied_step = _cap_step(direction.primal, trust_radius)
        trial_coordinates = initial_coordinates + applied_step
        _, trial_constraints = joint_value_constraints(trial_coordinates)
        correction_rhs = -trial_constraints
        correction_solve = _solve_gram(initial_gram_factor, correction_rhs)
        correction_dual = correction_solve.multipliers
        correction = initial_rows.constraint_jacobian.T @ correction_dual
        corrected_coordinates = trial_coordinates + correction
        corrected_rows = materialize_joint_vjp_rows(
            joint_value_constraints,
            corrected_coordinates,
            batch_width=8,
        )
        projection = project_equality_multipliers(
            corrected_rows.objective_gradient,
            corrected_rows.constraint_jacobian,
        )
        corrected_multipliers = projection.multipliers
        corrected_stationarity = (
            corrected_rows.objective_gradient
            + corrected_rows.constraint_jacobian.T @ corrected_multipliers
        )
        all_finite = (
            direction.all_finite
            & correction_solve.all_finite
            & projection.all_finite
            & jnp.all(jnp.isfinite(corrected_coordinates))
            & jnp.all(jnp.isfinite(corrected_multipliers))
            & jnp.all(jnp.isfinite(corrected_stationarity))
        )
        return CurvatureCanaryEndpoint(
            coordinates=corrected_coordinates,
            multipliers=corrected_multipliers,
            objective=corrected_rows.objective,
            constraints=corrected_rows.constraints,
            stationarity=corrected_stationarity,
            scaled_feasibility_inf=jnp.linalg.norm(
                corrected_rows.constraints,
                ord=jnp.inf,
            ),
            scaled_stationarity_inf=jnp.linalg.norm(
                corrected_stationarity,
                ord=jnp.inf,
            ),
            raw_direction=direction.primal,
            applied_step=applied_step + correction,
            raw_direction_norm=jnp.linalg.norm(direction.primal),
            applied_step_norm=jnp.linalg.norm(applied_step + correction),
            kkt_relative_residual=direction.relative_residual,
            kkt_condition_estimate=direction.condition_estimate,
            kkt_forward_error_bound=direction.forward_error_bound,
            multiplier_projection_relative_residual=projection.relative_residual,
            multiplier_projection_reciprocal_condition=projection.reciprocal_condition,
            multiplier_projection_forward_error_bound=projection.forward_error_bound,
            correction_relative_residual=correction_solve.relative_residual,
            correction_forward_error_bound=correction_solve.forward_error_bound,
            all_finite=all_finite,
        )

    zero_direction = jnp.zeros_like(initial_coordinates)
    initial_endpoint = CurvatureCanaryEndpoint(
        coordinates=initial_coordinates,
        multipliers=initial_multipliers,
        objective=initial_rows.objective,
        constraints=initial_rows.constraints,
        stationarity=initial_stationarity,
        scaled_feasibility_inf=jnp.linalg.norm(initial_rows.constraints, ord=jnp.inf),
        scaled_stationarity_inf=jnp.linalg.norm(initial_stationarity, ord=jnp.inf),
        raw_direction=zero_direction,
        applied_step=zero_direction,
        raw_direction_norm=jnp.asarray(0.0, dtype=initial_coordinates.dtype),
        applied_step_norm=jnp.asarray(0.0, dtype=initial_coordinates.dtype),
        kkt_relative_residual=jnp.asarray(0.0, dtype=initial_coordinates.dtype),
        kkt_condition_estimate=jnp.asarray(1.0, dtype=initial_coordinates.dtype),
        kkt_forward_error_bound=jnp.asarray(0.0, dtype=initial_coordinates.dtype),
        multiplier_projection_relative_residual=(initial_projection.relative_residual),
        multiplier_projection_reciprocal_condition=(
            initial_projection.reciprocal_condition
        ),
        multiplier_projection_forward_error_bound=(
            initial_projection.forward_error_bound
        ),
        correction_relative_residual=jnp.asarray(0.0, dtype=initial_coordinates.dtype),
        correction_forward_error_bound=jnp.asarray(
            0.0, dtype=initial_coordinates.dtype
        ),
        all_finite=(
            jnp.isfinite(initial_rows.objective)
            & jnp.all(jnp.isfinite(initial_rows.constraints))
            & jnp.all(jnp.isfinite(initial_stationarity))
            & jnp.all(jnp.isfinite(initial_multipliers))
            & initial_projection.all_finite
        ),
    )
    identity_endpoint = corrected_endpoint(identity_direction)
    exact_endpoint = corrected_endpoint(exact_direction)
    initial_projection_usable = (
        initial_projection.all_finite
        & (initial_projection.relative_residual <= linear_residual_tolerance)
        & (initial_projection.forward_error_bound <= 1.0e-7)
    )
    identity_usable = (
        identity_endpoint.all_finite
        & (identity_endpoint.kkt_relative_residual <= linear_residual_tolerance)
        & (identity_endpoint.kkt_forward_error_bound <= 1.0e-7)
        & (identity_endpoint.correction_relative_residual <= linear_residual_tolerance)
        & (identity_endpoint.correction_forward_error_bound <= 1.0e-7)
        & (
            identity_endpoint.multiplier_projection_relative_residual
            <= linear_residual_tolerance
        )
        & (identity_endpoint.multiplier_projection_forward_error_bound <= 1.0e-7)
        & (identity_endpoint.scaled_feasibility_inf <= feasibility_tolerance)
    )
    exact_usable = (
        exact_endpoint.all_finite
        & (exact_endpoint.kkt_relative_residual <= linear_residual_tolerance)
        & (exact_endpoint.kkt_forward_error_bound <= 1.0e-7)
        & (exact_endpoint.correction_relative_residual <= linear_residual_tolerance)
        & (exact_endpoint.correction_forward_error_bound <= 1.0e-7)
        & (
            exact_endpoint.multiplier_projection_relative_residual
            <= linear_residual_tolerance
        )
        & (exact_endpoint.multiplier_projection_forward_error_bound <= 1.0e-7)
        & (exact_endpoint.scaled_feasibility_inf <= feasibility_tolerance)
        & (symmetry_defect <= linear_residual_tolerance)
        & (action_defect <= linear_residual_tolerance)
    )
    both_variants_usable = initial_projection_usable & identity_usable & exact_usable
    exact_scaled_stationarity_improved = (
        both_variants_usable
        & (
            exact_endpoint.scaled_stationarity_inf
            < initial_endpoint.scaled_stationarity_inf
        )
        & (
            exact_endpoint.scaled_stationarity_inf
            < identity_endpoint.scaled_stationarity_inf
        )
    )
    all_finite = (
        initial_endpoint.all_finite
        & identity_endpoint.all_finite
        & exact_endpoint.all_finite
        & jnp.all(jnp.isfinite(exact_hessian))
        & jnp.isfinite(symmetry_defect)
        & jnp.isfinite(action_defect)
    )
    return DenseCurvatureCanaryResult(
        initial=initial_endpoint,
        identity=identity_endpoint,
        exact=exact_endpoint,
        exact_hessian=exact_hessian,
        exact_hessian_symmetry_relative_defect=symmetry_defect,
        exact_hessian_action_relative_defect=action_defect,
        both_variants_usable=both_variants_usable,
        exact_scaled_stationarity_improved=exact_scaled_stationarity_improved,
        all_finite=all_finite,
    )


__all__ = [
    "CurvatureCanaryEndpoint",
    "DenseCurvatureCanaryResult",
    "EqualityMultiplierProjection",
    "materialize_exact_lagrangian_hessian",
    "project_equality_multipliers",
    "run_dense_curvature_canary",
    "solve_dense_primal_dual_direction",
]
