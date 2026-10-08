"""One exact boozerQA solve, objective and implicit gradient on one device.

The objective is native NonQuasiSymmetricRatio plus identity quadratic
penalties on iota and major radius and the upper quadratic penalty on total
CurveLength. The exact Newton and residual coil VJP are PR8's kernels; this
module supplies their composition, not another solver or optimizer.

All numeric settings, coil reconstruction templates and the last successful
inner solution are operands. A failed Newton solve evaluates the objective
gradient at its returned iterate, reports 1e3, and returns the incoming warm
start unchanged, as upstream boozerQA.py does before its next line search.
The returned failed iterate is available separately for comparison/inspection.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_factor, lu_solve

from simsopt_jax.pytree import pytree_dataclass

from .boozer_problem import BoozerProblem
from .boozer_solvers import boozer_exact_newton, boozer_exact_residual_coil_vjp
from .curve_geometry import curve_length_from_dofs, optimizable_input_dofs_from_map_spec
from .field import coil_set_spec_from_dof_extraction_spec
from .quasisymmetry import non_quasi_symmetric_ratio
from .specs import CoilSetDofExtractionSpec, GroupedCoilSetSpec, SurfaceSpec
from .surface_fourier_series import surface_spec_with_dofs
from .surface_geometry import (
    surface_gamma,
    surface_gammadash1,
    surface_gammadash2,
    surface_volume,
)

__all__ = [
    "ExactSingleStageProblem",
    "ExactSingleStageResult",
    "ExactSingleStageState",
    "exact_single_stage_evaluate",
]


@pytree_dataclass(
    data=("boozer", "extraction", "nonqs_surface", "residual_rows", "targets", "tol", "maxiter"),
    meta=("length_indices", "axis"),
)
class ExactSingleStageProblem:
    """Immutable boozerQA snapshot; targets are [iota, major radius, length].

    The length indices select original coil curves in the extraction snapshot
    (each occurrence contributes once). Fixed DOF templates and free-vector
    mappings belong to PR3. Rebuild the snapshot when that layout, fixed values
    or structural grids change; free values and numeric settings are traced.
    G is always an explicit inner variable, as in native boozerQA.
    """

    boozer: BoozerProblem
    extraction: CoilSetDofExtractionSpec
    nonqs_surface: SurfaceSpec
    residual_rows: jax.Array
    targets: jax.Array
    tol: jax.Array
    maxiter: jax.Array
    length_indices: tuple[int, ...]
    axis: int


@pytree_dataclass(data=("x",))
class ExactSingleStageState:
    """Last successful [all surface DOFs, iota, G], passed in and returned.

    JAX arrays are immutable. Keeping this state to retry or branch an outer
    evaluation is safe; neither the evaluator nor another state owns a cache.
    """

    x: jax.Array


@pytree_dataclass(
    data=(
        "value", "gradient", "state", "solved_x", "success", "iterations",
        "norm", "singular", "adjoint_finite", "radius_singular", "terms",
    )
)
class ExactSingleStageResult:
    """Device result, including the failed iterate and native error flags.

    ``terms`` contains raw [NonQS, iota, major radius, total length]. The
    adapter raises native singular/nonfinite-adjoint errors; pure callers must
    inspect these flags. ``state`` is restored on failure, ``solved_x`` is not.
    The gradient on failure is not the derivative of the constant 1e3 penalty.
    """

    value: jax.Array
    gradient: jax.Array
    state: ExactSingleStageState
    solved_x: jax.Array
    success: jax.Array
    iterations: jax.Array
    norm: jax.Array
    singular: jax.Array
    adjoint_finite: jax.Array
    radius_singular: jax.Array
    terms: jax.Array


@jax.custom_jvp
def _absolute(value: jax.Array) -> jax.Array:
    """Native abs with its sign(0)=0 derivative and IEEE nonfinite values."""
    return jnp.abs(value)


@_absolute.defjvp
def _absolute_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    (value,), (tangent,) = primals, tangents
    return _absolute(value), jnp.sign(value) * tangent


def _major_radius(surface: SurfaceSpec) -> tuple[jax.Array, jax.Array]:
    """Native volume/(2*pi**2*minor_radius**2), with its singular-map flag.

    The cylindrical-area expression is PR7's closed form of native's det/inv
    evaluation. The flag preserves native's error on an exactly singular
    finite map, where the closed form alone could otherwise remain finite.
    Float64 underflow/overflow limits of PR6 surface geometry still apply.
    """
    gamma = surface_gamma(surface)
    xphi, xtheta = surface_gammadash1(surface), surface_gammadash2(surface)
    x, y = gamma[..., 0], gamma[..., 1]
    radius_squared = x * x + y * y
    phi_numerator = x * xphi[..., 1] - y * xphi[..., 0]
    phi_derivative = phi_numerator / radius_squared
    singular = jnp.any(jnp.isfinite(phi_derivative) & (phi_derivative == 0))
    section = (
        xtheta[..., 2] * phi_numerator
        - xphi[..., 2] * (x * xtheta[..., 1] - y * xtheta[..., 0])
    ) / jnp.sqrt(radius_squared)
    mean_area = _absolute(jnp.mean(section)) / (2 * np.pi)
    minor_radius = jnp.sqrt(mean_area / np.pi)
    major_radius = _absolute(surface_volume(surface)) / (2 * np.pi**2 * minor_radius**2)
    return major_radius, singular


def _length_penalty(
    problem: ExactSingleStageProblem, parameters: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    total_length = jnp.zeros((), parameters.dtype)
    for index in problem.length_indices:
        coil = problem.extraction.coils[index]
        dofs = optimizable_input_dofs_from_map_spec(coil.curve_map, parameters)
        total_length = total_length + curve_length_from_dofs(coil.curve, dofs)
    return 0.5 * jnp.maximum(total_length - problem.targets[2], 0)**2, total_length


@jax.jit
def exact_single_stage_evaluate(
    problem: ExactSingleStageProblem,
    parameters: jax.Array,
    state: ExactSingleStageState,
) -> ExactSingleStageResult:
    """Solve from state, evaluate boozerQA with a shared adjoint factorization.

    One jitted dispatch; no differentiation through the Newton iteration.
    New free DOFs, warm starts, targets, tolerance and cap reuse the program.
    Callers place all arguments on a single device before calling.
    Finite penalty weights share one coil VJP; nonfinite weights are applied
    after component coil VJPs, preserving native's inf/NaN propagation.
    """
    def reconstruct(parameters: jax.Array) -> GroupedCoilSetSpec:
        return coil_set_spec_from_dof_extraction_spec(problem.extraction, parameters)

    coils, coil_pullback = jax.vjp(reconstruct, parameters)
    boozer = replace(problem.boozer, coils=coils)
    solved = boozer_exact_newton(boozer, state.x, problem.residual_rows, problem.tol, problem.maxiter)
    nonqs_surface = surface_spec_with_dofs(problem.nonqs_surface, solved.x[:-2])
    nonqs, dnonqs, direct_coils = non_quasi_symmetric_ratio(
        nonqs_surface, coils, axis=problem.axis,
    )

    def radius(dofs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _major_radius(surface_spec_with_dofs(problem.boozer.surface, dofs))

    (major_radius, radius_singular), dradius = jax.value_and_grad(radius, has_aux=True)(solved.x[:-2])
    iota = solved.x[-2]
    multipliers = jnp.stack((jnp.ones_like(iota), iota - problem.targets[0], major_radius - problem.targets[1]))
    value = nonqs + 0.5 * multipliers[1]**2 + 0.5 * multipliers[2]**2
    surface_terms = jnp.stack((nonqs, iota, major_radius))
    component_rhs = jnp.stack((
        jnp.pad(dnonqs, (0, 2)),
        jnp.zeros_like(solved.x).at[-2].set(1),
        jnp.pad(dradius, (0, 2)),
    ), axis=1)
    # Native component objectives solve before QuadraticPenalty multiplies
    # their derivatives. One LU and a matrix RHS retain that error boundary.
    component_adjoints = lu_solve(lu_factor(solved.jacobian), component_rhs, trans=1)
    direct_gradient = coil_pullback(direct_coils)[0]

    def implicit_gradient(adjoint: jax.Array) -> jax.Array:
        cotangent = boozer_exact_residual_coil_vjp(boozer, solved.x, problem.residual_rows, adjoint)
        return coil_pullback(cotangent)[0]

    def finite_gradient() -> jax.Array:
        return direct_gradient - implicit_gradient(component_adjoints @ multipliers)

    def nonfinite_gradient() -> jax.Array:
        # Applying infinity before a solve/VJP spreads NaNs to unrelated
        # entries. Native scales each completed free-vector derivative.
        components = jax.lax.map(implicit_gradient, component_adjoints.T)
        return (
            direct_gradient - components[0]
            - multipliers[1] * components[1] - multipliers[2] * components[2]
        )

    gradient = jax.lax.cond(jnp.all(jnp.isfinite(multipliers)), finite_gradient, nonfinite_gradient)
    (length_penalty, total_length), length_gradient = jax.value_and_grad(
        _length_penalty, argnums=1, has_aux=True,
    )(problem, parameters)
    success = solved.norm <= problem.tol
    return ExactSingleStageResult(
        value=jnp.where(success, value + length_penalty, 1e3),
        gradient=gradient + length_gradient,
        state=ExactSingleStageState(jnp.where(success, solved.x, state.x)),
        solved_x=solved.x,
        success=success,
        iterations=solved.iterations,
        norm=solved.norm,
        singular=solved.singular,
        adjoint_finite=(
            jnp.all(jnp.isfinite(solved.jacobian))
            & jnp.all(jnp.isfinite(component_rhs))
            & jnp.all(jnp.isfinite(component_adjoints))
        ),
        radius_singular=radius_singular,
        terms=jnp.concatenate((surface_terms, total_length[None])),
    )
