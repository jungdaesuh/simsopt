"""One exact boozerQA solve, objective and implicit gradient on one device.

The objective is native NonQuasiSymmetricRatio plus identity quadratic
penalties on iota and major radius and the upper quadratic penalty on total
CurveLength. The exact Newton solve and residual coil VJP are ``boozer_solvers``' kernels; this
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
from .quasisymmetry import _ratio
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
    mappings come from the ``CoilSetDofExtractionSpec``. Rebuild the snapshot
    when that layout, fixed values or structural grids change; free values and
    numeric settings are traced. G is always an explicit inner variable, as in
    native boozerQA.

    Args:
        boozer (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering. Exact formulation with explicit G.
        extraction (CoilSetDofExtractionSpec): Fixed-value templates and maps from the
            outer free coil DOF vector of shape (nfree,).
        nonqs_surface (SurfaceSpec): Immutable surface coefficients and quadrature grid;
            all native DOFs are included. Auxiliary grid for the non-QS objective.
        residual_rows (jax.Array): Shape (nrows,) integer indices of the flattened
            Boozer residual, in native mask order.
        targets (jax.Array): Shape (3,) targets [dimensionless iota, major radius in
            meters, total selected curve length in meters].
        tol (jax.Array): Scalar shape () stopping tolerance in the norm's units.
        maxiter (jax.Array): Scalar shape () float64 iteration cap, which may be
            infinite.
        length_indices (tuple[int, ...]): Original coil indices selecting length terms;
            each occurrence contributes once.
        axis (int): Non-QS averaging axis, 0 for phi or 1 for theta.
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

    Args:
        x (jax.Array): Shape (nsurface + 2,) last successful [all surface DOFs in
            meters, dimensionless iota, G in tesla meters]; caller-owned immutable warm
            start.
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

    Args:
        value (jax.Array): Scalar shape () sum of the non-QS ratio and unit-weight
            native quadratic penalties, or 1e3 on failed convergence.
        gradient (jax.Array): Shape (nfree,) outer free coil DOF gradient; evaluated at
            solved_x even on failed convergence.
        state (ExactSingleStageState): Next warm start; retains the incoming state on
            failed convergence.
        solved_x (jax.Array): Shape (nsurface + 2,) final inner [surface DOFs, iota, G],
            including a failed iterate.
        success (jax.Array): Scalar shape () bool indicating the stopping residual norm
            is within tol.
        iterations (jax.Array): Scalar shape () completed int32 inner step count.
        norm (jax.Array): Scalar shape () native exact-solve stopping residual norm.
        singular (jax.Array): Scalar shape () bool indicating an exactly singular finite
            Newton matrix.
        adjoint_finite (jax.Array): Scalar shape () bool indicating finite inner
            Jacobian, adjoint right-hand sides and adjoints.
        radius_singular (jax.Array): Scalar shape () bool indicating a singular
            cylindrical map in the major-radius calculation.
        terms (jax.Array): Shape (4,) raw [dimensionless non-QS ratio, dimensionless
            iota, major radius in meters, selected total length in meters].
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

    The cylindrical-angle derivatives along the two surface tangents are
    forward-mode derivatives of ``atan2(y, x)``. The flag preserves native's
    error on an exactly singular finite map, where the cross-section area alone
    could otherwise remain finite. Float64 underflow/overflow limits of
    ``surface_geometry`` still apply.
    """
    gamma = surface_gamma(surface)
    xphi, xtheta = surface_gammadash1(surface), surface_gammadash2(surface)
    x, y = gamma[..., 0], gamma[..., 1]
    _, phi_derivative = jax.jvp(jnp.arctan2, (y, x), (xphi[..., 1], xphi[..., 0]))
    _, theta_phi_derivative = jax.jvp(jnp.arctan2, (y, x), (xtheta[..., 1], xtheta[..., 0]))
    singular = jnp.any(jnp.isfinite(phi_derivative) & (phi_derivative == 0))
    section = jnp.sqrt(x * x + y * y) * (
        xtheta[..., 2] * phi_derivative - xphi[..., 2] * theta_phi_derivative
    )
    mean_area = _absolute(jnp.mean(section)) / (2 * np.pi)
    minor_radius = jnp.sqrt(mean_area / np.pi)
    major_radius = _absolute(surface_volume(surface)) / (2 * np.pi**2 * minor_radius**2)
    return major_radius, singular


def _length_penalty(
    problem: ExactSingleStageProblem, dofs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    total_length = jnp.zeros((), dofs.dtype)
    for index in problem.length_indices:
        coil = problem.extraction.coils[index]
        curve_dofs = optimizable_input_dofs_from_map_spec(coil.curve_map, dofs)
        total_length = total_length + curve_length_from_dofs(coil.curve, curve_dofs)
    return 0.5 * jnp.maximum(total_length - problem.targets[2], 0)**2, total_length


@jax.jit
def exact_single_stage_evaluate(
    problem: ExactSingleStageProblem,
    dofs: jax.Array,
    state: ExactSingleStageState,
) -> ExactSingleStageResult:
    """Solve from state, evaluate boozerQA with a shared adjoint factorization.

    One jitted dispatch; no differentiation through the Newton iteration.
    New free DOFs, warm starts, targets, tolerance and cap reuse the program.
    Callers place all arguments on a single device before calling.
    The gradient is one reverse-mode derivative of the composed objective,
    one adjoint solve and one implicit coil VJP. When a penalty multiplier is
    nonfinite, each component is solved and pulled back on its own and then
    scaled, preserving native's inf/NaN propagation.

    Successful values add the non-QS ratio to half the squared iota and
    major-radius target errors, plus ``0.5 * max(total_length - length_target, 0)**2``.

    Args:
        problem (ExactSingleStageProblem): Immutable problem snapshot with an explicit
            inner G variable.
        dofs (jax.Array): Shape (nfree,) free coil DOFs in the extraction snapshot's
            native order, including geometry coefficients and current DOFs.
        state (ExactSingleStageState): Caller-owned warm start for the inner exact
            solve.

    Returns:
        ExactSingleStageResult: Device result object with value/gradient, solve facts and next warm start.
            Failed convergence returns value 1e3 and the gradient at the failed iterate,
            while preserving the incoming state.
    """
    def reconstruct(dofs: jax.Array) -> GroupedCoilSetSpec:
        return coil_set_spec_from_dof_extraction_spec(problem.extraction, dofs)

    coils, coil_pullback = jax.vjp(reconstruct, dofs)
    boozer = replace(problem.boozer, coils=coils)
    solved = boozer_exact_newton(boozer, state.x, problem.residual_rows, problem.tol, problem.maxiter)

    def terms_of(x: jax.Array, coils: GroupedCoilSetSpec) -> tuple[jax.Array, jax.Array]:
        nonqs = _ratio(surface_spec_with_dofs(problem.nonqs_surface, x[:-2]), coils, problem.axis)
        major_radius, singular = _major_radius(surface_spec_with_dofs(problem.boozer.surface, x[:-2]))
        return jnp.stack((nonqs, x[-2], major_radius)), singular

    def scalarization(terms: jax.Array) -> jax.Array:
        return (
            terms[0] + 0.5 * (terms[1] - problem.targets[0])**2
            + 0.5 * (terms[2] - problem.targets[1])**2
        )

    def objective(
        x: jax.Array, coils: GroupedCoilSetSpec,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        terms, singular = terms_of(x, coils)
        return scalarization(terms), (terms, singular)

    (value, (terms, radius_singular)), (rhs, direct_coils) = jax.value_and_grad(
        objective, argnums=(0, 1), has_aux=True,
    )(solved.x, coils)
    factor = lu_factor(solved.jacobian)

    def implicit_gradient(adjoint: jax.Array) -> jax.Array:
        cotangent = boozer_exact_residual_coil_vjp(boozer, solved.x, problem.residual_rows, adjoint)
        return coil_pullback(cotangent)[0]

    def adjoint_finite(rhs: jax.Array, adjoint: jax.Array) -> jax.Array:
        return (
            jnp.all(jnp.isfinite(solved.jacobian))
            & jnp.all(jnp.isfinite(rhs))
            & jnp.all(jnp.isfinite(adjoint))
        )

    def composed_gradient() -> tuple[jax.Array, jax.Array]:
        adjoint = lu_solve(factor, rhs, trans=1)
        return coil_pullback(direct_coils)[0] - implicit_gradient(adjoint), adjoint_finite(rhs, adjoint)

    multipliers = jax.grad(scalarization)(terms)

    def component_gradient() -> tuple[jax.Array, jax.Array]:
        # Sending an infinite multiplier into the solve or the VJPs spreads
        # NaNs to unrelated entries. Native solves each component objective,
        # then scales its completed free-vector derivative by the multiplier;
        # its error flag reads the component right-hand sides and adjoints.
        _, terms_pullback, _ = jax.vjp(terms_of, solved.x, coils, has_aux=True)
        component_rhs, component_direct = jax.vmap(terms_pullback)(jnp.eye(3, dtype=solved.x.dtype))
        component_adjoints = lu_solve(factor, component_rhs.T, trans=1)
        components = (
            jax.vmap(lambda cotangent: coil_pullback(cotangent)[0])(component_direct)
            - jax.lax.map(implicit_gradient, component_adjoints.T)
        )
        gradient = components[0] + multipliers[1] * components[1] + multipliers[2] * components[2]
        return gradient, adjoint_finite(component_rhs, component_adjoints)

    gradient, finite = jax.lax.cond(jnp.all(jnp.isfinite(multipliers)), composed_gradient, component_gradient)
    (length_penalty, total_length), length_gradient = jax.value_and_grad(
        _length_penalty, argnums=1, has_aux=True,
    )(problem, dofs)
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
        adjoint_finite=finite,
        radius_singular=radius_singular,
        terms=jnp.concatenate((terms, total_length[None])),
    )
