"""Reduced nested Boozer LS: exact ``(ι, G)`` QR projection, Newton on ``s``.

The nested surface is the branch ``g(c, s) = ∇_s Φ̂ = 0`` where
``y*(c, s) = argmin_y Φ(c, s, y)`` is the two-column residual QR and
``Φ̂(c, s) = Φ(c, s, y*)``. Volume and the axis-z penalty do not depend
on ``y``, and the Boozer residual is affine in ``y``, so the QR is exact
variable projection.

This is not F3 flat-675 QR-in-``J``. It does not change the fused
L-BFGS-B objective or in-graph ``newton_polish``. Physics knobs are the
reconstruct bar in :mod:`nested_ls_contract`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres
from simsopt.geo.boozersurface import _boozer_iterate_is_persistable
from simsopt_jax.geo.optimizers.linear_solve import _hessian_vector_product_fn
from simsopt_jax.numerical_policy import NEWTON_ARMIJO_C1

from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.flat675.y_solve import (
    FLAT675_Y_COLUMN_COUNT,
    solve_flat675_y_qr,
)
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_STAB,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_WEIGHT_INV_MODB,
)

_Y_SIZE = FLAT675_Y_COLUMN_COUNT
NESTED_LS_SCHUR_GMRES_RESTART: Final[int] = 8
NESTED_LS_SCHUR_GMRES_MAXITER: Final[int] = 1
NESTED_LS_SCHUR_BACKTRACKING_MAX_STEPS: Final[int] = 8
NESTED_LS_SCHUR_GMRES_RTOL: Final[float] = 1.0e-10


class NestedLsReducedRankError(ValueError):
    """Raised when the ``(ι, G)`` residual columns are rank-deficient."""


@dataclass(frozen=True, slots=True)
class NestedLsYSolution:
    """Two-column QR result for ``y* = (ι, G)`` at one frozen surface."""

    solution: jax.Array
    singular_values: jax.Array
    numerical_rank: jax.Array
    numerics_finite: jax.Array
    design_matrix: jax.Array
    right_hand_side: jax.Array


@dataclass(frozen=True, slots=True)
class NestedLsReducedNewtonResult:
    """One reduced Newton polish: surface DOFs plus the projected ``y*``."""

    success: bool
    persisted: bool
    iteration_count: int
    iota: float
    G: float
    surface_dofs: NDArray[np.float64]
    reduced_gradient: NDArray[np.float64]
    objective: float
    coil_delta_inf: float
    y_rank: int
    y_singular_values: tuple[float, float]


@dataclass(frozen=True, slots=True)
class NestedLsReducedSchurOperator:
    """Cached ``Ĥ_ss v = Φ_ss v − Φ_sy Φ_yy⁻¹ Φ_ys v`` at one ``(s, y*)``.

    ``Φ_yy`` is the exact 2×2 y-block, not Gauss–Newton ``AᵀA``. Packed
    HVPs differentiate ``Φ(s, y)`` only; they do not go through QR.
    """

    packed: jax.Array
    surface_size: int
    phi_sy: jax.Array
    phi_yy: jax.Array
    y_star: jax.Array
    y_rank: int
    phi_yy_condition: float
    _packed_hvp: object

    def apply(self, tangent: object) -> jax.Array:
        """Return ``Ĥ_ss v`` using one packed ``Φ`` HVP and the cached 2×2."""

        vector = jnp.asarray(tangent, dtype=jnp.float64).reshape(-1)
        if vector.shape != (self.surface_size,):
            raise ValueError(
                "Schur HVP tangent shape "
                f"{vector.shape} does not match surface shape "
                f"({self.surface_size},)."
            )
        packed_tangent = jnp.concatenate(
            (vector, jnp.zeros((_Y_SIZE,), dtype=jnp.float64))
        )
        packed_hvp = self._packed_hvp(self.packed, packed_tangent)
        phi_ss_v = packed_hvp[: self.surface_size]
        phi_ys_v = packed_hvp[self.surface_size :]
        correction = jnp.linalg.solve(self.phi_yy, phi_ys_v)
        return phi_ss_v - self.phi_sy @ correction


@dataclass(frozen=True, slots=True)
class NestedLsHvpComparison:
    """AD-through-QR HVP versus Schur HVP on the same tangent."""

    ad_through_qr: NDArray[np.float64]
    schur: NDArray[np.float64]
    max_abs: float
    rel_l2: float
    ad_through_qr_seconds: float
    schur_apply_seconds: float
    schur_factor_seconds: float
    phi_yy_condition: float


@dataclass(frozen=True, slots=True)
class NestedLsSchurNewtonResult:
    """One capped Schur-Newton polish on ``s``, with Krylov telemetry."""

    success: bool
    persisted: bool
    iteration_count: int
    iota: float
    G: float
    surface_dofs: NDArray[np.float64]
    reduced_gradient: NDArray[np.float64]
    objective: float
    coil_delta_inf: float
    y_rank: int
    y_singular_values: tuple[float, float]
    step_accepted: bool
    step_alpha: float
    gmres_info: int
    gmres_matvecs: int
    gmres_residual_l2: float
    gmres_restart: int
    gmres_maxiter: int
    factor_seconds: float
    gmres_seconds: float
    phi_yy_condition: float


def split_surface_and_y(decision: object) -> tuple[jax.Array, jax.Array]:
    """Split ``[surface_dofs, iota, G]`` into ``s`` and ``y``."""

    packed = jnp.asarray(decision, dtype=jnp.float64).reshape(-1)
    if packed.size < _Y_SIZE + 1:
        raise ValueError("nested-LS decision must be [surface_dofs, iota, G].")
    return packed[:-_Y_SIZE], packed[-_Y_SIZE:]


def pack_surface_and_y(surface_dofs: object, y: object) -> jax.Array:
    """Pack surface DOFs with ``y = (ι, G)``."""

    surface = jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1)
    y_vec = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    if y_vec.shape != (_Y_SIZE,):
        raise ValueError("y must be (iota, G) with shape (2,).")
    if surface.size < 1:
        raise ValueError("surface_dofs must contain at least one coordinate.")
    return jnp.concatenate((surface, y_vec))


def projected_y_system(
    residual_fn,
    surface_dofs: object,
    y_probe: object,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(A, b)`` such that ``r(s, y) = A y - b`` at this surface."""

    surface = jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1)
    probe = jnp.asarray(y_probe, dtype=jnp.float64).reshape(-1)
    if probe.shape != (_Y_SIZE,):
        raise ValueError("y probe must be (iota, G) with shape (2,).")

    def residual_of_y(y: jax.Array) -> jax.Array:
        return jnp.asarray(residual_fn(pack_surface_and_y(surface, y)))

    residual = residual_of_y(probe)
    design_matrix = jax.jacfwd(residual_of_y)(probe)
    right_hand_side = design_matrix @ probe - residual
    return design_matrix, right_hand_side


def solve_projected_y(
    residual_fn,
    surface_dofs: object,
    y_probe: object | None = None,
) -> NestedLsYSolution:
    """Solve ``y*(s)`` by economy QR of the two residual columns."""

    probe = jnp.zeros((_Y_SIZE,), dtype=jnp.float64) if y_probe is None else y_probe
    design_matrix, right_hand_side = projected_y_system(
        residual_fn, surface_dofs, probe
    )
    raw = solve_flat675_y_qr(design_matrix, right_hand_side)
    return NestedLsYSolution(
        solution=raw.solution,
        singular_values=raw.singular_values,
        numerical_rank=raw.numerical_rank,
        numerics_finite=raw.numerics_finite,
        design_matrix=design_matrix,
        right_hand_side=right_hand_side,
    )


def require_full_y_rank(solution: NestedLsYSolution) -> None:
    """Refuse a projected ``y*`` whose design matrix is not finite rank 2."""

    finite = bool(np.asarray(jax.device_get(solution.numerics_finite)))
    rank = int(np.asarray(jax.device_get(solution.numerical_rank)))
    if (not finite) or rank != _Y_SIZE:
        singular = np.asarray(
            jax.device_get(solution.singular_values), dtype=np.float64
        )
        raise NestedLsReducedRankError(
            "nested-LS (iota, G) design matrix is not finite rank "
            f"{_Y_SIZE}; rank={rank}, finite={finite}, singular values {singular}."
        )


def make_reduced_penalty_objective(residual_fn, objective_fn):
    """Return ``Φ̂(s) = Φ(s, y*(s))`` as a scalar JAX function of ``s``."""

    def phi_hat(surface_dofs: jax.Array) -> jax.Array:
        surface = jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1)
        solution = solve_projected_y(residual_fn, surface)
        y_star = solution.solution
        projected = (solution.numerical_rank == _Y_SIZE) & solution.numerics_finite
        y_star = jnp.where(projected, y_star, jnp.full_like(y_star, jnp.nan))
        return objective_fn(pack_surface_and_y(surface, y_star))

    return phi_hat


def reduced_penalty_gradient(phi_hat, surface_dofs: object) -> jax.Array:
    """Exact reduced gradient ``g = ∇_s Φ̂``."""

    surface = jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1)
    return jax.grad(phi_hat)(surface)


def reduced_penalty_hvp(phi_hat, surface_dofs: object, tangent: object) -> jax.Array:
    """Exact reduced Hessian-vector product ``H_ss v``."""

    surface = jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1)
    vector = jnp.asarray(tangent, dtype=jnp.float64).reshape(-1)
    if vector.shape != surface.shape:
        raise ValueError(
            "reduced HVP tangent shape "
            f"{vector.shape} does not match surface shape {surface.shape}."
        )
    return jax.jvp(jax.grad(phi_hat), (surface,), (vector,))[1]


def reduced_penalty_gradient_envelope(
    objective_fn,
    surface_dofs: object,
    y_star: object,
) -> jax.Array:
    """Envelope reduced gradient: the ``s``-block of ``∇Φ`` at ``(s, y*)``."""

    packed = pack_surface_and_y(surface_dofs, y_star)
    return jax.grad(objective_fn)(packed)[:-_Y_SIZE]


def _packed_objective_hvp(objective_fn):
    return _hessian_vector_product_fn(objective_fn)


def _require_full_phi_yy(phi_yy: jax.Array) -> tuple[NDArray[np.float64], float]:
    matrix = np.asarray(jax.device_get(phi_yy), dtype=np.float64)
    if matrix.shape != (_Y_SIZE, _Y_SIZE) or not np.all(np.isfinite(matrix)):
        raise NestedLsReducedRankError(
            "nested-LS Φ_yy is not a finite 2×2 Hessian block; "
            f"shape={matrix.shape}, finite={np.all(np.isfinite(matrix))}."
        )
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank_threshold = (
        float(_Y_SIZE) * float(np.finfo(np.float64).eps) * float(abs(singular[0]))
    )
    rank = int(np.sum(singular > rank_threshold))
    if rank != _Y_SIZE:
        raise NestedLsReducedRankError(
            "nested-LS Φ_yy is not finite rank "
            f"{_Y_SIZE}; rank={rank}, singular values {singular}."
        )
    condition = float(abs(singular[0]) / abs(singular[-1]))
    return np.array(singular, dtype=np.float64, copy=True), condition


def factor_reduced_nested_ls_schur(
    residual_fn,
    objective_fn,
    surface_dofs: object,
    y_probe: object | None = None,
) -> NestedLsReducedSchurOperator:
    """Factor ``Ĥ_ss`` at ``y*(s)`` from two packed ``Φ`` HVPs of the y-basis."""

    surface = jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1)
    solution = solve_projected_y(residual_fn, surface, y_probe)
    require_full_y_rank(solution)
    packed = pack_surface_and_y(surface, solution.solution)
    packed_hvp = _packed_objective_hvp(objective_fn)
    surface_size = int(surface.size)
    sy_columns = []
    yy_columns = []
    zeros_surface = jnp.zeros((surface_size,), dtype=jnp.float64)
    for index in range(_Y_SIZE):
        e_y = jnp.zeros((_Y_SIZE,), dtype=jnp.float64).at[index].set(1.0)
        packed_hvp_column = packed_hvp(packed, jnp.concatenate((zeros_surface, e_y)))
        sy_columns.append(packed_hvp_column[:surface_size])
        yy_columns.append(packed_hvp_column[surface_size:])
    phi_sy = jnp.stack(sy_columns, axis=1)
    phi_yy = jnp.stack(yy_columns, axis=1)
    _singular, condition = _require_full_phi_yy(phi_yy)
    del _singular
    return NestedLsReducedSchurOperator(
        packed=packed,
        surface_size=surface_size,
        phi_sy=phi_sy,
        phi_yy=phi_yy,
        y_star=solution.solution,
        y_rank=int(np.asarray(jax.device_get(solution.numerical_rank))),
        phi_yy_condition=condition,
        _packed_hvp=packed_hvp,
    )


def compare_ad_qr_and_schur_hvp(
    residual_fn,
    objective_fn,
    phi_hat,
    surface_dofs: object,
    tangent: object,
) -> NestedLsHvpComparison:
    """Compare the full AD-through-QR HVP vector with the Schur HVP."""

    surface = jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1)
    vector = jnp.asarray(tangent, dtype=jnp.float64).reshape(-1)
    ad_started = time.perf_counter()
    ad_hvp = np.asarray(reduced_penalty_hvp(phi_hat, surface, vector), dtype=np.float64)
    ad_seconds = time.perf_counter() - ad_started
    factor_started = time.perf_counter()
    operator = factor_reduced_nested_ls_schur(residual_fn, objective_fn, surface)
    factor_seconds = time.perf_counter() - factor_started
    apply_started = time.perf_counter()
    schur_hvp = np.asarray(operator.apply(vector), dtype=np.float64)
    apply_seconds = time.perf_counter() - apply_started
    delta = schur_hvp - ad_hvp
    ad_norm = float(np.linalg.norm(ad_hvp))
    rel_l2 = (
        float(np.linalg.norm(delta) / ad_norm)
        if ad_norm > 0.0
        else float(np.linalg.norm(delta))
    )
    return NestedLsHvpComparison(
        ad_through_qr=ad_hvp,
        schur=schur_hvp,
        max_abs=float(np.max(np.abs(delta))),
        rel_l2=rel_l2,
        ad_through_qr_seconds=float(ad_seconds),
        schur_apply_seconds=float(apply_seconds),
        schur_factor_seconds=float(factor_seconds),
        phi_yy_condition=float(operator.phi_yy_condition),
    )


def nested_ls_reduced_closures(
    jax_boozer: BoozerSurfaceJAX,
    *,
    constraint_weight: float = NESTED_LS_CONSTRAINT_WEIGHT,
    weight_inv_modB: bool = NESTED_LS_WEIGHT_INV_MODB,
):
    """Residual, full ``Φ``, and ``Φ̂`` closures on one JAX Boozer surface."""

    residual_fn = jax_boozer._make_penalty_residual_with(
        True,
        weight_inv_modB,
        constraint_weight,
        decision_split_mode="jvp",
    )
    objective_fn = jax_boozer._make_penalty_objective_with(
        True,
        weight_inv_modB,
        constraint_weight,
        decision_split_mode="jvp",
    )
    return (
        residual_fn,
        objective_fn,
        make_reduced_penalty_objective(residual_fn, objective_fn),
    )


def _host_vector(values) -> NDArray[np.float64]:
    return np.array(jax.device_get(values), dtype=np.float64, copy=True).reshape(-1)


def _finite_vector(values) -> bool:
    return bool(np.all(np.isfinite(_host_vector(values))))


def _coil_coordinates(biotsavart) -> NDArray[np.float64]:
    return np.array(biotsavart.x, dtype=np.float64, copy=True)


def run_reduced_nested_ls_newton(
    jax_boozer: BoozerSurfaceJAX,
    *,
    iota: float,
    G: float,
    constraint_weight: float = NESTED_LS_CONSTRAINT_WEIGHT,
    weight_inv_modB: bool = NESTED_LS_WEIGHT_INV_MODB,
    stab: float = NESTED_LS_NEWTON_STAB,
    tol: float = NESTED_LS_NEWTON_TOL,
    maxiter: int = NESTED_LS_NEWTON_MAXITER,
) -> NestedLsReducedNewtonResult:
    """Newton on ``s`` of ``Φ̂``, with reconstruct-bar persist/rollback.

    Mutates ``jax_boozer.surface``. Does not write ``self.res``. Never
    materializes dense ``H_ss`` (autodiff-through-QR at 661 is a ~66 GB
    RSS path). Coils stay frozen. Trajectories need not match full-state
    Newton; a persisted iterate is the ``(s, y*)`` branch.
    """

    residual_fn, _objective_fn, phi_hat = nested_ls_reduced_closures(
        jax_boozer,
        constraint_weight=constraint_weight,
        weight_inv_modB=weight_inv_modB,
    )
    surface_start = _host_vector(jax_boozer.surface.get_dofs())
    y_start = np.array([float(iota), float(G)], dtype=np.float64)
    coil_before = _coil_coordinates(jax_boozer.biotsavart)
    start_solution = solve_projected_y(residual_fn, surface_start, y_start)
    require_full_y_rank(start_solution)

    surface_start_jax = jnp.asarray(surface_start, dtype=jnp.float64)
    initial_value, initial_grad = jax.value_and_grad(phi_hat)(surface_start_jax)
    initial_norm = float(np.linalg.norm(_host_vector(initial_grad)))

    method = jax_boozer._resolve_optimizer_method(optimize_G=True)
    polish = jax_boozer._run_newton_polish_for_method(
        method,
        phi_hat,
        surface_start_jax,
        maxiter=maxiter,
        tol=tol,
        stab=stab,
        materialize_hessian=False,
        max_dense_hessian_bytes=jax_boozer.options["max_dense_linearization_bytes"],
        progress_callback=jax_boozer._resolve_newton_progress_callback(method),
        objective_args=(),
    )
    polished_surface = _host_vector(polish["x"])
    polished_grad = _host_vector(polish["grad"])
    finite_iterate = _finite_vector(polished_surface) and _finite_vector(polished_grad)
    final_norm = (
        float(np.linalg.norm(polished_grad)) if finite_iterate else float("inf")
    )
    effective_success = bool(finite_iterate) and bool(polish["success"])
    unmoved = bool(finite_iterate) and np.array_equal(polished_surface, surface_start)
    persist = unmoved or _boozer_iterate_is_persistable(
        effective_success,
        final_norm,
        initial_norm,
    )
    if persist:
        surface_final = polished_surface
        reduced_gradient = polished_grad
        objective = float(np.asarray(jax.device_get(polish["fun"])))
        committed_success = effective_success
        final_solution = solve_projected_y(residual_fn, surface_final)
        require_full_y_rank(final_solution)
        y_final = _host_vector(final_solution.solution)
        y_rank = int(np.asarray(jax.device_get(final_solution.numerical_rank)))
        singular = _host_vector(final_solution.singular_values)
    else:
        surface_final = surface_start
        reduced_gradient = _host_vector(initial_grad)
        objective = float(np.asarray(jax.device_get(initial_value)))
        committed_success = False
        y_final = _host_vector(start_solution.solution)
        y_rank = int(np.asarray(jax.device_get(start_solution.numerical_rank)))
        singular = _host_vector(start_solution.singular_values)

    jax_boozer.surface.set_dofs(surface_final)
    coil_after = _coil_coordinates(jax_boozer.biotsavart)
    return NestedLsReducedNewtonResult(
        success=committed_success,
        persisted=bool(persist),
        iteration_count=int(np.asarray(jax.device_get(polish["nit"]))),
        iota=float(y_final[0]),
        G=float(y_final[1]),
        surface_dofs=np.array(surface_final, dtype=np.float64, copy=True),
        reduced_gradient=np.array(reduced_gradient, dtype=np.float64, copy=True),
        objective=objective,
        coil_delta_inf=float(np.linalg.norm(coil_after - coil_before, ord=np.inf)),
        y_rank=y_rank,
        y_singular_values=(float(singular[0]), float(singular[1])),
    )


def _envelope_value_and_grad(residual_fn, objective_fn, surface_dofs):
    solution = solve_projected_y(residual_fn, surface_dofs)
    require_full_y_rank(solution)
    packed = pack_surface_and_y(surface_dofs, solution.solution)
    value, full_grad = jax.value_and_grad(objective_fn)(packed)
    return (
        float(np.asarray(jax.device_get(value))),
        _host_vector(full_grad[:-_Y_SIZE]),
        solution,
    )


def _schur_armijo_step(
    residual_fn,
    objective_fn,
    surface,
    value,
    gradient,
    newton_direction,
    current_solution,
):
    descent = float(np.real(np.vdot(gradient, newton_direction)))
    alpha = 1.0
    current_norm = float(np.linalg.norm(gradient))
    for _ in range(NESTED_LS_SCHUR_BACKTRACKING_MAX_STEPS):
        trial = surface - alpha * newton_direction
        if not np.all(np.isfinite(trial)):
            alpha *= 0.5
            continue
        trial_value, trial_grad, trial_solution = _envelope_value_and_grad(
            residual_fn, objective_fn, trial
        )
        trial_norm = float(np.linalg.norm(trial_grad))
        finite = np.isfinite(trial_value) and np.all(np.isfinite(trial_grad))
        armijo = descent > 0.0 and trial_value <= (
            value - NEWTON_ARMIJO_C1 * alpha * descent
        )
        if finite and (armijo or trial_norm <= current_norm):
            return trial, trial_value, trial_grad, trial_solution, alpha, True
        alpha *= 0.5
    return surface, value, gradient, current_solution, alpha, False


def run_reduced_nested_ls_schur_newton(
    jax_boozer: BoozerSurfaceJAX,
    *,
    iota: float,
    G: float,
    constraint_weight: float = NESTED_LS_CONSTRAINT_WEIGHT,
    weight_inv_modB: bool = NESTED_LS_WEIGHT_INV_MODB,
    stab: float = NESTED_LS_NEWTON_STAB,
    tol: float = NESTED_LS_NEWTON_TOL,
    maxiter: int = 1,
    gmres_restart: int = NESTED_LS_SCHUR_GMRES_RESTART,
    gmres_maxiter: int = NESTED_LS_SCHUR_GMRES_MAXITER,
    gmres_rtol: float = NESTED_LS_SCHUR_GMRES_RTOL,
) -> NestedLsSchurNewtonResult:
    """Capped GMRES Newton on ``s`` using the Schur ``Ĥ_ss`` operator.

    Mutates ``jax_boozer.surface``. Does not write ``self.res``. Packed
    HVPs never differentiate through QR. Default Krylov budget is one
    restart cycle of 8 matvecs — not a full walk.
    """

    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(
        jax_boozer,
        constraint_weight=constraint_weight,
        weight_inv_modB=weight_inv_modB,
    )
    surface = _host_vector(jax_boozer.surface.get_dofs())
    y_start = np.array([float(iota), float(G)], dtype=np.float64)
    coil_before = _coil_coordinates(jax_boozer.biotsavart)
    start_solution = solve_projected_y(residual_fn, surface, y_start)
    require_full_y_rank(start_solution)
    value, gradient, current_solution = _envelope_value_and_grad(
        residual_fn, objective_fn, surface
    )
    initial_norm = float(np.linalg.norm(gradient))
    iteration_count = 0
    step_accepted = False
    step_alpha = 1.0
    gmres_info = -1
    gmres_matvecs = 0
    gmres_residual_l2 = 0.0
    factor_seconds = 0.0
    gmres_seconds = 0.0
    phi_yy_condition = 0.0
    working_surface = surface
    working_value = value
    working_grad = gradient
    working_solution = current_solution

    while iteration_count < maxiter:
        grad_norm = float(np.linalg.norm(working_grad))
        if grad_norm <= tol:
            break
        factor_started = time.perf_counter()
        operator = factor_reduced_nested_ls_schur(
            residual_fn,
            objective_fn,
            working_surface,
            y_probe=working_solution.solution,
        )
        factor_seconds += time.perf_counter() - factor_started
        phi_yy_condition = float(operator.phi_yy_condition)
        matvec_count = 0

        def matvec(tangent, schur=operator, stab_value=float(stab)):
            nonlocal matvec_count
            matvec_count += 1
            vector = np.asarray(tangent, dtype=np.float64).reshape(-1)
            hv = np.asarray(
                jax.device_get(schur.apply(vector)), dtype=np.float64
            ).reshape(-1)
            return hv + stab_value * vector

        linear = LinearOperator(
            (operator.surface_size, operator.surface_size),
            matvec=matvec,
            dtype=np.float64,
        )
        gmres_started = time.perf_counter()
        newton_direction, gmres_info = gmres(
            linear,
            working_grad,
            rtol=float(gmres_rtol),
            atol=0.0,
            restart=int(gmres_restart),
            maxiter=int(gmres_maxiter),
        )
        gmres_seconds += time.perf_counter() - gmres_started
        gmres_matvecs += matvec_count
        newton_direction = np.asarray(newton_direction, dtype=np.float64).reshape(-1)
        residual = matvec(newton_direction) - working_grad
        gmres_residual_l2 = float(np.linalg.norm(residual))
        if not np.all(np.isfinite(newton_direction)):
            step_accepted = False
            break
        (
            working_surface,
            working_value,
            working_grad,
            working_solution,
            step_alpha,
            step_accepted,
        ) = _schur_armijo_step(
            residual_fn,
            objective_fn,
            working_surface,
            working_value,
            working_grad,
            newton_direction,
            working_solution,
        )
        if not step_accepted:
            break
        iteration_count += 1

    finite_iterate = _finite_vector(working_surface) and _finite_vector(working_grad)
    final_norm = float(np.linalg.norm(working_grad)) if finite_iterate else float("inf")
    effective_success = bool(finite_iterate) and final_norm <= float(tol)
    unmoved = bool(finite_iterate) and np.array_equal(working_surface, surface)
    persist = unmoved or _boozer_iterate_is_persistable(
        effective_success,
        final_norm,
        initial_norm,
    )
    if persist:
        surface_final = working_surface
        reduced_gradient = working_grad
        objective = working_value
        committed_success = effective_success
        final_solution = working_solution
    else:
        surface_final = surface
        reduced_gradient = gradient
        objective = value
        committed_success = False
        final_solution = start_solution
        iteration_count = 0
        step_accepted = False

    y_final = _host_vector(final_solution.solution)
    y_rank = int(np.asarray(jax.device_get(final_solution.numerical_rank)))
    singular = _host_vector(final_solution.singular_values)
    jax_boozer.surface.set_dofs(surface_final)
    coil_after = _coil_coordinates(jax_boozer.biotsavart)
    return NestedLsSchurNewtonResult(
        success=committed_success,
        persisted=bool(persist),
        iteration_count=iteration_count,
        iota=float(y_final[0]),
        G=float(y_final[1]),
        surface_dofs=np.array(surface_final, dtype=np.float64, copy=True),
        reduced_gradient=np.array(reduced_gradient, dtype=np.float64, copy=True),
        objective=objective,
        coil_delta_inf=float(np.linalg.norm(coil_after - coil_before, ord=np.inf)),
        y_rank=y_rank,
        y_singular_values=(float(singular[0]), float(singular[1])),
        step_accepted=bool(step_accepted),
        step_alpha=float(step_alpha),
        gmres_info=int(gmres_info),
        gmres_matvecs=int(gmres_matvecs),
        gmres_residual_l2=float(gmres_residual_l2),
        gmres_restart=int(gmres_restart),
        gmres_maxiter=int(gmres_maxiter),
        factor_seconds=float(factor_seconds),
        gmres_seconds=float(gmres_seconds),
        phi_yy_condition=float(phi_yy_condition),
    )


__all__ = [
    "NESTED_LS_SCHUR_GMRES_MAXITER",
    "NESTED_LS_SCHUR_GMRES_RESTART",
    "NestedLsHvpComparison",
    "NestedLsReducedNewtonResult",
    "NestedLsReducedRankError",
    "NestedLsReducedSchurOperator",
    "NestedLsSchurNewtonResult",
    "NestedLsYSolution",
    "compare_ad_qr_and_schur_hvp",
    "factor_reduced_nested_ls_schur",
    "make_reduced_penalty_objective",
    "nested_ls_reduced_closures",
    "pack_surface_and_y",
    "projected_y_system",
    "reduced_penalty_gradient",
    "reduced_penalty_gradient_envelope",
    "reduced_penalty_hvp",
    "require_full_y_rank",
    "run_reduced_nested_ls_newton",
    "run_reduced_nested_ls_schur_newton",
    "solve_projected_y",
    "split_surface_and_y",
]
