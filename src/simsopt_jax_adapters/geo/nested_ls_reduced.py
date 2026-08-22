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

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from numpy.typing import NDArray
from simsopt.geo.boozersurface import BoozerSurface, _boozer_iterate_is_persistable
from simsopt_jax.geo.optimizers.linear_solve import (
    _hessian_vector_product_fn,
    _materialize_dense_linear_operator,
    _run_operator_gmres,
)
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
PackedPenaltyHvp = Callable[[jax.Array, jax.Array], jax.Array]
NESTED_LS_SCHUR_GMRES_RESTART: Final[int] = 8
NESTED_LS_SCHUR_GMRES_MAXITER: Final[int] = 1
# Outer GMRES restart-cycle cap for the matrix-free lane. Doubling
# ``maxiter`` repeats an eight-vector Krylov space; it does not
# enlarge it. F3 B37 step 4 needed 512 cycles at restart=8 to meet
# live Choice 2 η≈0.0407; step 6 still missed η≈0.027 at 1024.
# Dense LU of the 661×661 Ĥ_ss+stab I is the canary replacement.
# Fourier-block Jacobi ``M`` stays opt-in.
NESTED_LS_SCHUR_GMRES_MAXITER_CAP: Final[int] = 512
NESTED_LS_SCHUR_BACKTRACKING_MAX_STEPS: Final[int] = 8
# Inexact-Newton forcing η_k = ‖(Ĥ_ss+stab I)δs − g‖₂ / ‖g‖₂. 0.24 is
# η_max (F3 B37 SciPy-cap observation). Choice 2 of Eisenstat–Walker
# 1996 adapts η_k downward from this cap.
NESTED_LS_SCHUR_GMRES_RTOL: Final[float] = 0.24
NESTED_LS_LINEAR_SOLVERS: Final[tuple[str, ...]] = ("gmres", "dense_lu", "shamanskii")
NESTED_LS_SHAMANSKII_REFINE_PASSES: Final[int] = 3
NESTED_LS_EW_GAMMA: Final[float] = 0.9
NESTED_LS_EW_ALPHA: Final[float] = 2.0
NESTED_LS_EW_SAFEGUARD: Final[float] = 0.1
NESTED_LS_DENSE_FLOAT64_BYTES: Final[int] = 8
_TENSOR_FOURIER_DOF_NAME = re.compile(r"^([xyz])\((\d+),(\d+)\)$")


class NestedLsReducedRankError(ValueError):
    """Raised when projected ``(ι, G)`` or a Fourier block of ``Ĥ_ss`` is rank-deficient."""


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

    Because the residual is affine in ``y=(ι, G)``, the exact inner
    block is ``Φ_yy = AᵀA``. Surface and reduced blocks still carry
    residual-curvature terms that Gauss–Newton would drop. Packed HVPs
    differentiate ``Φ(s, y)`` only; they do not go through QR.
    """

    packed: jax.Array
    surface_size: int
    phi_sy: jax.Array
    phi_yy: jax.Array
    y_star: jax.Array
    y_rank: int
    phi_yy_condition: float
    _packed_hvp: PackedPenaltyHvp

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
class NestedLsSchurNewtonStepRecord:
    """One Schur-Newton linear solve plus Armijo attempt."""

    iteration: int
    iota: float
    G: float
    objective: float
    grad_l2: float
    gmres_forcing_eta: float
    gmres_residual_l2: float
    gmres_rtol: float
    gmres_maxiter: int
    factor_seconds: float
    gmres_seconds: float
    step_alpha: float
    step_accepted: bool
    assembled: bool = False
    shamanskii_reused: bool = False
    shamanskii_reassembled: bool = False
    shamanskii_attempt_eta: float | None = None
    shamanskii_attempt_eta_reason: str | None = None
    shamanskii_refine_passes: int = 0


@dataclass(frozen=True, slots=True)
class NestedLsBananaRunCodeLane:
    """One banana ``run_code`` (BFGS then Newton, ``stab=0``)."""

    success: bool
    iteration_count: int
    iota: float
    G: float
    surface_dofs: NDArray[np.float64]
    coil_delta_inf: float
    seconds: float


@dataclass(frozen=True, slots=True)
class NestedLsBananaRunCodePair:
    """Native and JAX banana ``run_code`` from the same start."""

    native: NestedLsBananaRunCodeLane
    jax: NestedLsBananaRunCodeLane
    physics_matched: bool


class NestedLsB37TimingBlocked(RuntimeError):
    """Raised when B37 nested timing is requested before B3 matches."""


@dataclass(frozen=True, slots=True)
class NestedLsSchurNewtonResult:
    """One capped Schur-Newton polish on ``s``, with Krylov telemetry.

    ``gmres_rtol`` is η_max passed by the caller. Each step's requested
    η_k is Eisenstat–Walker Choice 2, recorded on the step. Armijo runs
    only when the independent unpreconditioned certificate satisfies
    ``gmres_forcing_eta ≤ η_k``. ``gmres_maxiter`` is the last Krylov
    restart-cycle count actually used. ``gmres_info`` is JAX's 0/−1
    NaN placeholder, not SciPy's iteration count. ``gmres_matvecs``
    stays 0: JAX incremental GMRES does not report operator applications.
    """

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
    gmres_forcing_eta: float
    gmres_solution: NDArray[np.float64]
    gmres_rtol: float
    gmres_restart: int
    gmres_maxiter: int
    factor_seconds: float
    gmres_seconds: float
    phi_yy_condition: float
    steps: tuple[NestedLsSchurNewtonStepRecord, ...]


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


def _packed_objective_hvp(objective_fn) -> PackedPenaltyHvp:
    hvp = _hessian_vector_product_fn(objective_fn)

    def packed_hvp(packed: jax.Array, tangent: jax.Array) -> jax.Array:
        return hvp(packed, tangent)

    return packed_hvp


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
    hostify_inputs: bool = True,
):
    """Residual, full ``Φ``, and ``Φ̂`` closures on one JAX Boozer surface.

    Default ``hostify_inputs=True`` captures coils as host constants (frozen
    coils). Pass ``False`` to keep device coil specs without making coil
    DOFs arguments; runtime coil DOFs use
    :func:`nested_ls_runtime_coil_closures`.
    """

    residual_fn = jax_boozer._make_penalty_residual_with(
        True,
        weight_inv_modB,
        constraint_weight,
        hostify_inputs=hostify_inputs,
        decision_split_mode="jvp",
    )
    objective_fn = jax_boozer._make_penalty_objective_with(
        True,
        weight_inv_modB,
        constraint_weight,
        hostify_inputs=hostify_inputs,
        decision_split_mode="jvp",
    )
    return (
        residual_fn,
        objective_fn,
        make_reduced_penalty_objective(residual_fn, objective_fn),
    )


def nested_ls_runtime_coil_closures(
    jax_boozer: BoozerSurfaceJAX,
    *,
    constraint_weight: float = NESTED_LS_CONSTRAINT_WEIGHT,
    weight_inv_modB: bool = NESTED_LS_WEIGHT_INV_MODB,
):
    """Residual, ``Φ``, and ``Φ̂`` as functions of ``(packed, coil_dofs)``.

    Uses the binary ``(x, coil_set_spec)`` penalty kernels. Coil DOFs
    stay runtime arguments through
    ``BiotSavartJAX.coil_set_spec_from_dofs``. Frozen-coil Newton still
    uses :func:`nested_ls_reduced_closures`.
    """

    biotsavart = jax_boozer.biotsavart
    residual_kernel = jax_boozer._get_traceable_penalty_residual(
        True,
        weight_inv_modB,
        constraint_weight,
    )
    objective_kernel = jax_boozer._get_traceable_penalty_objective(
        True,
        weight_inv_modB,
        constraint_weight,
    )

    def residual_fn(packed: jax.Array, coil_dofs: jax.Array) -> jax.Array:
        spec = biotsavart.coil_set_spec_from_dofs(coil_dofs)
        return residual_kernel(packed, spec)

    def objective_fn(packed: jax.Array, coil_dofs: jax.Array) -> jax.Array:
        spec = biotsavart.coil_set_spec_from_dofs(coil_dofs)
        return objective_kernel(packed, spec)

    def residual_at_coil(coil_dofs: jax.Array):
        coil = jnp.asarray(coil_dofs, dtype=jnp.float64).reshape(-1)

        def residual_of_packed(packed: jax.Array) -> jax.Array:
            return residual_fn(packed, coil)

        return residual_of_packed

    def objective_at_coil(coil_dofs: jax.Array):
        coil = jnp.asarray(coil_dofs, dtype=jnp.float64).reshape(-1)

        def objective_of_packed(packed: jax.Array) -> jax.Array:
            return objective_fn(packed, coil)

        return objective_of_packed

    def phi_hat(surface_dofs: jax.Array, coil_dofs: jax.Array) -> jax.Array:
        return make_reduced_penalty_objective(
            residual_at_coil(coil_dofs),
            objective_at_coil(coil_dofs),
        )(surface_dofs)

    return residual_fn, objective_fn, phi_hat


def _host_vector(values) -> NDArray[np.float64]:
    return np.array(jax.device_get(values), dtype=np.float64, copy=True).reshape(-1)


def _finite_vector(values) -> bool:
    return bool(np.all(np.isfinite(_host_vector(values))))


def _coil_coordinates(biotsavart) -> NDArray[np.float64]:
    return np.array(biotsavart.x, dtype=np.float64, copy=True)


def eisenstat_walker_forcing_eta(
    grad_norm: float,
    *,
    previous_grad_norm: float | None,
    previous_eta: float | None,
    eta_max: float,
    nonlinear_tol: float,
) -> float:
    """Eisenstat–Walker 1996 Choice 2 forcing, capped at ``eta_max``.

    ``η = γ (‖g_k‖/‖g_{k-1}‖)^α`` with ``γ=0.9``, ``α=2``. If
    ``γ η_{k-1}^α > 0.1``, η cannot drop faster than that freeze.
    Floor ``½ τ / ‖g_k‖`` prevents oversolving near the nonlinear
    tolerance. The first iteration returns ``eta_max``.
    """

    cap = float(eta_max)
    current = float(grad_norm)
    if (
        previous_grad_norm is None
        or not np.isfinite(previous_grad_norm)
        or float(previous_grad_norm) <= 0.0
    ):
        eta = cap
    else:
        ratio = current / float(previous_grad_norm)
        eta = NESTED_LS_EW_GAMMA * ratio**NESTED_LS_EW_ALPHA
        if previous_eta is not None and np.isfinite(previous_eta):
            freeze = NESTED_LS_EW_GAMMA * float(previous_eta) ** NESTED_LS_EW_ALPHA
            if freeze > NESTED_LS_EW_SAFEGUARD:
                eta = max(eta, freeze)
        eta = min(cap, eta)
    if current > 0.0:
        eta = min(cap, max(eta, 0.5 * float(nonlinear_tol) / current))
    return float(eta)


def linear_solve_meets_forcing(eta_achieved: float, eta_requested: float) -> bool:
    """True when the unpreconditioned certificate is at most η_requested."""

    return bool(np.isfinite(eta_achieved) and eta_achieved <= float(eta_requested))


def receipt_float(value: float) -> tuple[float | None, str | None]:
    """JSON-safe float: non-finite values become ``None`` plus a reason."""

    if np.isfinite(value):
        return float(value), None
    return None, "nonfinite"


def solve_operator_gmres_with_forcing(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    eta_requested: float,
    restart: int,
    maxiter: int,
    maxiter_cap: int,
    preconditioner: Callable[[jax.Array], jax.Array] | None = None,
    x0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, float, float, int]:
    """Device GMRES with doubling restart-cycles until η ≤ requested.

    The accept/reject certificate is the independent unpreconditioned
    ``‖Aδ − b‖₂ / ‖b‖₂``, not JAX ``info`` and not a comparison to the
    zero guess. If that certificate misses, retry from the current ``δ``
    with a tighter JAX ``tol`` so incremental GMRES cannot stop early
    on its internal residual estimate. ``x0`` continues from a prior
    ``δ`` when raising ``maxiter_cap``.
    """

    rhs_host = _host_vector(rhs)
    rhs_norm = float(np.linalg.norm(rhs_host))
    used = max(1, int(maxiter))
    cap = max(int(maxiter_cap), used)
    jax_tol = float(eta_requested)
    newton_jax = (
        jnp.zeros_like(rhs) if x0 is None else jnp.asarray(x0, dtype=jnp.float64)
    )
    residual = -rhs
    info: object = jnp.asarray(-1, dtype=jnp.int32)
    residual_l2 = rhs_norm
    eta = 1.0 if rhs_norm > 0.0 else 0.0
    guess = None if x0 is None else newton_jax
    while True:
        newton_jax, info = _run_operator_gmres(
            matvec,
            rhs,
            tol=float(jax_tol),
            restart=int(restart),
            maxiter=int(used),
            M=preconditioner,
            x0=guess,
        )
        residual = matvec(newton_jax) - rhs
        residual_l2 = float(np.linalg.norm(_host_vector(residual)))
        eta = residual_l2 / rhs_norm if rhs_norm > 0.0 else 0.0
        if linear_solve_meets_forcing(eta, eta_requested) or used >= cap:
            break
        guess = newton_jax
        used = min(cap, used * 2)
        if np.isfinite(eta) and eta > 0.0:
            jax_tol = min(float(eta_requested), 0.25 * float(eta))
    return newton_jax, residual, jnp.asarray(info), residual_l2, float(eta), int(used)


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


def _stabilized_schur_matvec(
    operator: NestedLsReducedSchurOperator,
    stab: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """Return the device map ``v ↦ Ĥ_ss v + stab v``."""

    def matvec(tangent: jax.Array) -> jax.Array:
        return operator.apply(tangent) + stab * tangent

    return matvec


def tensor_fourier_mode_blocks(
    dof_names: Sequence[str],
    *,
    mpol: int,
    ntor: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition TensorFourier DOFs into physical ``(m, n)`` harmonic blocks.

    Names are ``x(i,j)``, ``y(i,j)``, ``z(i,j)``. Tensor indices map to
    ``m = i`` or ``i-mpol`` and ``n = j`` or ``j-ntor``. Stellsym keeps
    the live xyz companions of each harmonic in one block.
    """

    if int(mpol) < 0 or int(ntor) < 0:
        raise ValueError(f"mpol and ntor must be nonnegative; got {mpol}, {ntor}.")
    groups: dict[tuple[int, int], list[int]] = {}
    for index, raw_name in enumerate(dof_names):
        matched = _TENSOR_FOURIER_DOF_NAME.fullmatch(str(raw_name))
        if matched is None:
            raise ValueError(
                "Fourier-block partition requires TensorFourier names "
                f"'x|y|z(i,j)'; got {raw_name!r} at index {index}."
            )
        i_index = int(matched.group(2))
        j_index = int(matched.group(3))
        mode_m = i_index if i_index <= mpol else i_index - mpol
        mode_n = j_index if j_index <= ntor else j_index - ntor
        groups.setdefault((mode_m, mode_n), []).append(index)
    if not groups:
        raise ValueError("Fourier-block partition received no TensorFourier names.")
    return tuple(tuple(groups[key]) for key in sorted(groups))


@dataclass(frozen=True, slots=True)
class NestedLsFourierBlockPreconditioner:
    """Left block-Jacobi inverse of the exact Fourier blocks of ``Ĥ_ss+stab I``.

    Factoring applies the Schur matvec once per live DOF and stores only
    the ``(m, n)`` slices, not the full ``n×n`` matrix.
    """

    blocks: tuple[tuple[int, ...], ...]
    inverses: tuple[jax.Array, ...]

    def apply(self, vector: jax.Array) -> jax.Array:
        result = jnp.zeros_like(vector)
        for indices, inverse in zip(self.blocks, self.inverses, strict=True):
            index_array = jnp.asarray(indices, dtype=jnp.int32)
            result = result.at[index_array].set(inverse @ vector[index_array])
        return result


def factor_schur_fourier_block_preconditioner(
    operator: NestedLsReducedSchurOperator,
    stab: float,
    dof_names: Sequence[str],
    *,
    mpol: int,
    ntor: int,
) -> NestedLsFourierBlockPreconditioner:
    """Factor exact physical ``(m, n)`` blocks of ``Ĥ_ss+stab I`` for GMRES ``M``."""

    blocks = tensor_fourier_mode_blocks(dof_names, mpol=mpol, ntor=ntor)
    if sum(len(indices) for indices in blocks) != operator.surface_size:
        raise ValueError(
            "Fourier-block DOF names do not cover the Schur surface vector; "
            f"covered={sum(len(indices) for indices in blocks)}, "
            f"surface_size={operator.surface_size}."
        )
    matvec = _stabilized_schur_matvec(operator, jnp.asarray(stab, dtype=jnp.float64))
    dimension = operator.surface_size
    inverses: list[jax.Array] = []
    for indices in blocks:
        index_array = jnp.asarray(indices, dtype=jnp.int32)
        columns = []
        for dof_index in indices:
            unit = (
                jnp.zeros((dimension,), dtype=jnp.float64).at[int(dof_index)].set(1.0)
            )
            columns.append(matvec(unit)[index_array])
        block = jnp.stack(columns, axis=1)
        inverse = jnp.linalg.inv(block)
        if not np.all(np.isfinite(np.asarray(jax.device_get(inverse)))):
            raise NestedLsReducedRankError(
                "Fourier-block of Ĥ_ss+stab I is not finite-invertible; "
                f"indices={indices}."
            )
        inverses.append(inverse)
    return NestedLsFourierBlockPreconditioner(blocks=blocks, inverses=tuple(inverses))


def schur_dense_operator_bytes(dimension: int) -> int:
    """Stored ``n×n`` float64 bytes for a dense stabilized Schur matrix."""

    size = int(dimension)
    return size * size * NESTED_LS_DENSE_FLOAT64_BYTES


def materialize_stabilized_schur_dense(
    operator: NestedLsReducedSchurOperator,
    stab: float,
    *,
    max_dense_linearization_bytes: int | None = None,
    chunk_batch_size: int | None = None,
) -> jax.Array:
    """Chunked dense ``Ĥ_ss+stab I`` via the linear-solve materializer.

    Peak HVP parallelism is the SSOT chunk batch, not ``n``. The stored
    matrix is ``n²`` float64 entries; refuse when that exceeds the cap.
    Optional ``chunk_batch_size`` overrides the import-time SSOT for a
    canary; ``None`` keeps the production batch.
    """

    dimension = int(operator.surface_size)
    stored_bytes = schur_dense_operator_bytes(dimension)
    if max_dense_linearization_bytes is not None and stored_bytes > int(
        max_dense_linearization_bytes
    ):
        raise MemoryError(
            "dense Ĥ_ss+stab I needs "
            f"{stored_bytes} bytes for dimension {dimension}; "
            f"max_dense_linearization_bytes={int(max_dense_linearization_bytes)}."
        )
    matvec = _stabilized_schur_matvec(operator, jnp.asarray(stab, dtype=jnp.float64))
    dummy = jnp.zeros((dimension,), dtype=jnp.float64)

    def linear_operator_fn(_linearization: jax.Array, tangent: jax.Array) -> jax.Array:
        return matvec(tangent)

    return _materialize_dense_linear_operator(
        linear_operator_fn, dummy, batch_width=chunk_batch_size
    )


def solve_stabilized_schur_dense_lu(dense: jax.Array, rhs: jax.Array) -> jax.Array:
    """A: direct LU of the chunked dense stabilized Schur matrix."""

    return jnp.linalg.solve(dense, rhs)


def dense_schur_lu_preconditioner(
    dense: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """Left GMRES ``M`` from LU of stored ``Ĥ_ss+stab I``. Not an explicit inverse."""

    lu_and_pivots = jsp_linalg.lu_factor(dense)

    def apply_m(vector: jax.Array) -> jax.Array:
        return jsp_linalg.lu_solve(lu_and_pivots, vector)

    return apply_m


def attempt_shamanskii_schur_reuse(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    eta_requested: float,
    apply_stale: Callable[[jax.Array], jax.Array],
    refine_passes: int = NESTED_LS_SHAMANSKII_REFINE_PASSES,
) -> tuple[jax.Array, jax.Array, float, bool, int]:
    """Stale-LU Newton: ``δ = H_stale⁻¹ g``, live η, optional refinement.

    ``apply_stale`` is a cached LU apply. Direct apply avoids JAX GMRES
    ``M``+``x0=0`` returning zeros. After the first live HVP, up to
    ``refine_passes`` corrections ``δ ← δ + H_stale⁻¹(g − H_live δ)``
    run; bail and let the caller reassemble if η is non-finite or does
    not decrease. Certificate is unpreconditioned ``‖Aδ − b‖₂ / ‖b‖₂``.
    """

    rhs_norm = float(np.linalg.norm(_host_vector(rhs)))
    newton_jax = apply_stale(rhs)
    residual = matvec(newton_jax) - rhs
    residual_l2 = float(np.linalg.norm(_host_vector(residual)))
    eta = residual_l2 / rhs_norm if rhs_norm > 0.0 else 0.0
    used_refines = 0
    if linear_solve_meets_forcing(eta, eta_requested):
        return newton_jax, residual, float(eta), True, used_refines
    limit = max(0, int(refine_passes))
    while used_refines < limit:
        if not np.isfinite(eta):
            break
        candidate = newton_jax + apply_stale(-residual)
        new_residual = matvec(candidate) - rhs
        new_l2 = float(np.linalg.norm(_host_vector(new_residual)))
        new_eta = new_l2 / rhs_norm if rhs_norm > 0.0 else 0.0
        if (not np.isfinite(new_eta)) or new_eta >= eta:
            break
        newton_jax = candidate
        residual = new_residual
        eta = new_eta
        used_refines += 1
        if linear_solve_meets_forcing(eta, eta_requested):
            return newton_jax, residual, float(eta), True, used_refines
    return (
        newton_jax,
        residual,
        float(eta),
        linear_solve_meets_forcing(eta, eta_requested),
        used_refines,
    )


def dense_schur_inverse_preconditioner(
    dense: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """B: left GMRES ``M`` from the dense inverse of ``Ĥ_ss+stab I``."""

    inverse = jnp.linalg.inv(dense)

    def apply_m(vector: jax.Array) -> jax.Array:
        return inverse @ vector

    return apply_m


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
    gmres_maxiter_cap: int = NESTED_LS_SCHUR_GMRES_MAXITER_CAP,
    gmres_rtol: float = NESTED_LS_SCHUR_GMRES_RTOL,
    gmres_preconditioner: Callable[[jax.Array], jax.Array] | None = None,
    linear_solver: str = "gmres",
    max_dense_linearization_bytes: int | None = None,
) -> NestedLsSchurNewtonResult:
    """Capped inexact Newton on ``s`` using Schur ``Ĥ_ss`` and device GMRES.

    Mutates ``jax_boozer.surface``. Does not write ``self.res``. Packed
    HVPs never differentiate through QR. ``linear_solver`` is ``gmres``,
    ``dense_lu``, or opt-in ``shamanskii``. Optional
    ``gmres_preconditioner`` is JAX GMRES ``M`` and is incompatible
    with Shamanskii. Armijo runs only when the independent
    unpreconditioned ``η = ‖(Ĥ+stab I)δs − g‖₂ / ‖g‖₂`` is at most the
    Eisenstat–Walker request. Weak GMRES retries by doubling
    ``maxiter`` up to ``gmres_maxiter_cap``, not by raising
    ``restart``. Shamanskii applies the stale LU (``δ = H_old⁻¹ g``),
    certifies with a live HVP, then up to
    ``NESTED_LS_SHAMANSKII_REFINE_PASSES`` residual corrections
    ``δ ← δ + H_old⁻¹(g − H_live δ)``. Bail and reassemble if η is
    non-finite or does not decrease.
    Inner dense LU has no default
    stored-matrix cap; the adjoint 1 MiB default is a different budget
    and would refuse the 661 matrix (3,495,368 bytes).
    """

    if linear_solver not in NESTED_LS_LINEAR_SOLVERS:
        raise ValueError(
            "linear_solver must be 'gmres', 'dense_lu', or 'shamanskii'; "
            f"got {linear_solver!r}."
        )
    if linear_solver == "shamanskii" and gmres_preconditioner is not None:
        raise ValueError(
            "linear_solver='shamanskii' cannot combine with gmres_preconditioner."
        )

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
    gmres_forcing_eta = 0.0
    gmres_solution = np.zeros(surface.size, dtype=np.float64)
    used_gmres_maxiter = int(gmres_maxiter)
    requested_eta = float(gmres_rtol)
    previous_grad_norm: float | None = None
    previous_eta: float | None = None
    factor_seconds = 0.0
    gmres_seconds = 0.0
    phi_yy_condition = 0.0
    working_surface = surface
    working_value = value
    working_grad = gradient
    working_solution = current_solution
    step_records: list[NestedLsSchurNewtonStepRecord] = []
    stale_apply: Callable[[jax.Array], jax.Array] | None = None

    while iteration_count < maxiter:
        grad_norm = float(np.linalg.norm(working_grad))
        if grad_norm <= tol:
            break
        requested_eta = eisenstat_walker_forcing_eta(
            grad_norm,
            previous_grad_norm=previous_grad_norm,
            previous_eta=previous_eta,
            eta_max=float(gmres_rtol),
            nonlinear_tol=float(tol),
        )
        factor_started = time.perf_counter()
        operator = factor_reduced_nested_ls_schur(
            residual_fn,
            objective_fn,
            working_surface,
            y_probe=working_solution.solution,
        )
        step_factor_seconds = time.perf_counter() - factor_started
        factor_seconds += step_factor_seconds
        phi_yy_condition = float(operator.phi_yy_condition)
        matvec = _stabilized_schur_matvec(
            operator, jnp.asarray(stab, dtype=jnp.float64)
        )
        rhs = jnp.asarray(working_grad, dtype=jnp.float64)
        gmres_started = time.perf_counter()
        assembled = False
        shamanskii_reused = False
        shamanskii_reassembled = False
        shamanskii_attempt_eta: float | None = None
        shamanskii_attempt_eta_reason: str | None = None
        shamanskii_refine_passes = 0
        newton_jax = jnp.zeros_like(rhs)
        residual = -rhs
        info = jnp.asarray(-1, dtype=jnp.int32)
        gmres_residual_l2 = float(grad_norm)
        gmres_forcing_eta = 1.0 if grad_norm > 0.0 else 0.0
        used_gmres_maxiter = int(gmres_maxiter)
        if linear_solver == "gmres":
            newton_jax, residual, info, gmres_residual_l2, gmres_forcing_eta, used = (
                solve_operator_gmres_with_forcing(
                    matvec,
                    rhs,
                    eta_requested=float(requested_eta),
                    restart=int(gmres_restart),
                    maxiter=int(gmres_maxiter),
                    maxiter_cap=int(gmres_maxiter_cap),
                    preconditioner=gmres_preconditioner,
                )
            )
            used_gmres_maxiter = int(used)
        elif linear_solver in ("dense_lu", "shamanskii"):
            if linear_solver == "shamanskii" and stale_apply is not None:
                (
                    newton_jax,
                    residual,
                    gmres_forcing_eta,
                    shamanskii_reused,
                    shamanskii_refine_passes,
                ) = attempt_shamanskii_schur_reuse(
                    matvec,
                    rhs,
                    eta_requested=float(requested_eta),
                    apply_stale=stale_apply,
                )
                shamanskii_attempt_eta, shamanskii_attempt_eta_reason = receipt_float(
                    gmres_forcing_eta
                )
                gmres_residual_l2 = float(np.linalg.norm(_host_vector(residual)))
                info = jnp.asarray(0, dtype=jnp.int32)
                used_gmres_maxiter = 0
            if not shamanskii_reused:
                shamanskii_reassembled = (
                    linear_solver == "shamanskii" and stale_apply is not None
                )
                dense = materialize_stabilized_schur_dense(
                    operator,
                    stab,
                    max_dense_linearization_bytes=max_dense_linearization_bytes,
                )
                newton_jax = solve_stabilized_schur_dense_lu(dense, rhs)
                info = jnp.asarray(0, dtype=jnp.int32)
                residual = matvec(newton_jax) - rhs
                gmres_residual_l2 = float(np.linalg.norm(_host_vector(residual)))
                gmres_forcing_eta = (
                    gmres_residual_l2 / grad_norm if grad_norm > 0.0 else 0.0
                )
                used_gmres_maxiter = int(gmres_maxiter)
                stale_apply = dense_schur_lu_preconditioner(dense)
                assembled = True
        else:
            raise ValueError(
                "linear_solver must be 'gmres', 'dense_lu', or 'shamanskii'; "
                f"got {linear_solver!r}."
            )
        step_linear_seconds = time.perf_counter() - gmres_started
        gmres_seconds += step_linear_seconds
        newton_direction = _host_vector(newton_jax)
        gmres_solution = np.array(newton_direction, dtype=np.float64, copy=True)
        gmres_info = int(_host_vector(jnp.reshape(jnp.asarray(info), (1,)))[0])
        if gmres_preconditioner is None:
            quality_residual = gmres_residual_l2
            quality_scale = grad_norm
        else:
            quality_residual = float(
                np.linalg.norm(_host_vector(gmres_preconditioner(residual)))
            )
            quality_scale = float(
                np.linalg.norm(_host_vector(gmres_preconditioner(rhs)))
            )
        forcing_ok = linear_solve_meets_forcing(gmres_forcing_eta, requested_eta)
        finite_direction = bool(np.all(np.isfinite(newton_direction)))
        if not finite_direction or quality_residual > quality_scale or not forcing_ok:
            step_accepted = False
            y_now = _host_vector(working_solution.solution)
            step_records.append(
                NestedLsSchurNewtonStepRecord(
                    iteration=iteration_count + 1,
                    iota=float(y_now[0]),
                    G=float(y_now[1]),
                    objective=float(working_value),
                    grad_l2=float(grad_norm),
                    gmres_forcing_eta=float(gmres_forcing_eta),
                    gmres_residual_l2=float(gmres_residual_l2),
                    gmres_rtol=float(requested_eta),
                    gmres_maxiter=int(used_gmres_maxiter),
                    factor_seconds=float(step_factor_seconds),
                    gmres_seconds=float(step_linear_seconds),
                    step_alpha=float(step_alpha),
                    step_accepted=False,
                    assembled=bool(assembled),
                    shamanskii_reused=bool(shamanskii_reused),
                    shamanskii_reassembled=bool(shamanskii_reassembled),
                    shamanskii_attempt_eta=shamanskii_attempt_eta,
                    shamanskii_attempt_eta_reason=shamanskii_attempt_eta_reason,
                    shamanskii_refine_passes=int(shamanskii_refine_passes),
                )
            )
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
        y_now = _host_vector(working_solution.solution)
        step_records.append(
            NestedLsSchurNewtonStepRecord(
                iteration=iteration_count + 1,
                iota=float(y_now[0]),
                G=float(y_now[1]),
                objective=float(working_value),
                grad_l2=float(np.linalg.norm(working_grad)),
                gmres_forcing_eta=float(gmres_forcing_eta),
                gmres_residual_l2=float(gmres_residual_l2),
                gmres_rtol=float(requested_eta),
                gmres_maxiter=int(used_gmres_maxiter),
                factor_seconds=float(step_factor_seconds),
                gmres_seconds=float(step_linear_seconds),
                step_alpha=float(step_alpha),
                step_accepted=bool(step_accepted),
                assembled=bool(assembled),
                shamanskii_reused=bool(shamanskii_reused),
                shamanskii_reassembled=bool(shamanskii_reassembled),
                shamanskii_attempt_eta=shamanskii_attempt_eta,
                shamanskii_attempt_eta_reason=shamanskii_attempt_eta_reason,
                shamanskii_refine_passes=int(shamanskii_refine_passes),
            )
        )
        if not step_accepted:
            break
        previous_grad_norm = grad_norm
        previous_eta = float(requested_eta)
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
        gmres_forcing_eta=float(gmres_forcing_eta),
        gmres_solution=np.array(gmres_solution, dtype=np.float64, copy=True),
        gmres_rtol=float(gmres_rtol),
        gmres_restart=int(gmres_restart),
        gmres_maxiter=int(used_gmres_maxiter),
        factor_seconds=float(factor_seconds),
        gmres_seconds=float(gmres_seconds),
        phi_yy_condition=float(phi_yy_condition),
        steps=tuple(step_records),
    )


def apply_reduced_mixed_schur_coil_tangent(
    residual_fn,
    objective_fn,
    surface_dofs: object,
    coil_dofs: object,
    coil_tangent: object,
    *,
    operator: NestedLsReducedSchurOperator | None = None,
) -> jax.Array:
    """Return ``Ĥ_sc v_c = Φ_sc v_c − Φ_sy Φ_yy⁻¹ Φ_yc v_c`` at frozen ``s``.

    ``residual_fn`` and ``objective_fn`` take ``(packed, coil_dofs)``.
    """

    coil = jnp.asarray(coil_dofs, dtype=jnp.float64).reshape(-1)
    tangent = jnp.asarray(coil_tangent, dtype=jnp.float64).reshape(-1)
    if tangent.shape != coil.shape:
        raise ValueError(
            "coil tangent shape "
            f"{tangent.shape} does not match coil DOF shape {coil.shape}."
        )
    if operator is None:
        operator = factor_reduced_nested_ls_schur(
            lambda packed: residual_fn(packed, coil),
            lambda packed: objective_fn(packed, coil),
            surface_dofs,
        )
    packed = operator.packed

    def packed_gradient(coil_vector: jax.Array) -> jax.Array:
        return jax.grad(lambda packed_state: objective_fn(packed_state, coil_vector))(
            packed
        )

    full_jvp = jax.jvp(packed_gradient, (coil,), (tangent,))[1]
    phi_sc_v = full_jvp[: operator.surface_size]
    phi_yc_v = full_jvp[operator.surface_size :]
    return phi_sc_v - operator.phi_sy @ jnp.linalg.solve(operator.phi_yy, phi_yc_v)


NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES: Final[int] = 1_048_576


def implicit_adjoint_coil_gradient(
    residual_fn,
    objective_fn,
    surface_dofs: object,
    coil_dofs: object,
    surface_cotangent: object,
    *,
    stab: float = 0.0,
    linear_solver: str = "gmres",
    gmres_rtol: float = 1.0e-10,
    gmres_restart: int = 64,
    gmres_maxiter: int = 10,
    gmres_maxiter_cap: int = NESTED_LS_SCHUR_GMRES_MAXITER_CAP,
    gmres_preconditioner: Callable[[jax.Array], jax.Array] | None = None,
    operator: NestedLsReducedSchurOperator | None = None,
    max_dense_linearization_bytes: int | None = (
        NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES
    ),
) -> jax.Array:
    """Return ``−Ĥ_scᵀ λ`` with ``Ĥ_ss λ = v_s`` (IFT, ``stab=0``).

    ``Ĥ_ss`` is a Hessian, so the adjoint solve is the forward Schur
    operator. Stabilization is not part of the IFT Jacobian; pass a
    positive ``stab`` only as a regularized canary or fold it into
    ``gmres_preconditioner``. Default GMRES is matrix-free. Dense LU
    remains the 7×7 path and still honours the 1 MiB stored-matrix cap.
    That 1 MiB default is the adjoint budget, not the reconstruct-inner
    Newton budget.
    """

    coil = jnp.asarray(coil_dofs, dtype=jnp.float64).reshape(-1)
    cotangent = jnp.asarray(surface_cotangent, dtype=jnp.float64).reshape(-1)
    if operator is None:
        operator = factor_reduced_nested_ls_schur(
            lambda packed: residual_fn(packed, coil),
            lambda packed: objective_fn(packed, coil),
            surface_dofs,
        )
    if cotangent.shape != (operator.surface_size,):
        raise ValueError(
            "surface cotangent shape "
            f"{cotangent.shape} does not match surface shape "
            f"({operator.surface_size},)."
        )
    if linear_solver == "dense_lu":
        dense = materialize_stabilized_schur_dense(
            operator,
            stab,
            max_dense_linearization_bytes=max_dense_linearization_bytes,
        )
        adjoint_state = solve_stabilized_schur_dense_lu(dense, cotangent)
    elif linear_solver == "gmres":
        matvec = _stabilized_schur_matvec(
            operator, jnp.asarray(stab, dtype=jnp.float64)
        )
        adjoint_state, _, _, _, eta, _ = solve_operator_gmres_with_forcing(
            matvec,
            cotangent,
            eta_requested=float(gmres_rtol),
            restart=int(gmres_restart),
            maxiter=int(gmres_maxiter),
            maxiter_cap=int(gmres_maxiter_cap),
            preconditioner=gmres_preconditioner,
        )
        if not linear_solve_meets_forcing(eta, gmres_rtol):
            raise RuntimeError(
                f"implicit adjoint GMRES forcing {eta} exceeded requested {gmres_rtol}."
            )
    else:
        raise ValueError(
            f"linear_solver must be 'gmres' or 'dense_lu'; got {linear_solver!r}."
        )

    def mixed_map(coil_tangent: jax.Array) -> jax.Array:
        return apply_reduced_mixed_schur_coil_tangent(
            residual_fn,
            objective_fn,
            surface_dofs,
            coil,
            coil_tangent,
            operator=operator,
        )

    # Reverse-mode at a nonzero probe so coil reconstruction is not traced
    # at the origin. The map is linear in the tangent.
    probe = jnp.zeros_like(coil).at[0].set(1.0)
    _primal, pullback = jax.vjp(mixed_map, probe)
    del _primal
    return -pullback(adjoint_state)[0]


def _banana_run_code_lane(
    boozer, *, iota: float, G: float
) -> NestedLsBananaRunCodeLane:
    coil_before = _coil_coordinates(boozer.biotsavart)
    boozer.need_to_run_code = True
    started = time.perf_counter()
    result = boozer.run_code(float(iota), G=float(G))
    seconds = time.perf_counter() - started
    if result is None:
        raise RuntimeError("banana run_code returned None; need_to_run_code was false.")
    coil_after = _coil_coordinates(boozer.biotsavart)
    return NestedLsBananaRunCodeLane(
        success=bool(result["success"]),
        iteration_count=int(result["iter"]),
        iota=float(result["iota"]),
        G=float(result["G"]),
        surface_dofs=np.array(boozer.surface.get_dofs(), dtype=np.float64, copy=True),
        coil_delta_inf=float(np.linalg.norm(coil_after - coil_before, ord=np.inf)),
        seconds=float(seconds),
    )


def run_banana_run_code_pair(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    *,
    iota: float,
    G: float,
    iota_rtol: float = 1.0e-8,
    iota_atol: float = 1.0e-10,
    surface_rtol: float = 1.0e-8,
    surface_atol: float = 1.0e-10,
) -> NestedLsBananaRunCodePair:
    """Native vs JAX banana ``run_code``. Coils stay frozen.

    This is the timing bar, not reconstruct Newton. Trajectories may
    differ from the physics bar. ``physics_matched`` is both successes,
    frozen coils, and ``(ι, G, s)`` within the supplied tolerances.
    """

    native_lane = _banana_run_code_lane(native, iota=iota, G=G)
    jax_lane = _banana_run_code_lane(jax_boozer, iota=iota, G=G)
    iota_close = np.allclose(
        jax_lane.iota, native_lane.iota, rtol=iota_rtol, atol=iota_atol
    )
    g_close = np.allclose(jax_lane.G, native_lane.G, rtol=iota_rtol, atol=iota_atol)
    surface_close = np.allclose(
        jax_lane.surface_dofs,
        native_lane.surface_dofs,
        rtol=surface_rtol,
        atol=surface_atol,
    )
    physics_matched = (
        native_lane.success
        and jax_lane.success
        and native_lane.coil_delta_inf == 0.0
        and jax_lane.coil_delta_inf == 0.0
        and iota_close
        and g_close
        and surface_close
    )
    return NestedLsBananaRunCodePair(
        native=native_lane,
        jax=jax_lane,
        physics_matched=bool(physics_matched),
    )


__all__ = [
    "NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES",
    "NESTED_LS_LINEAR_SOLVERS",
    "NESTED_LS_SCHUR_GMRES_MAXITER",
    "NESTED_LS_SCHUR_GMRES_MAXITER_CAP",
    "NESTED_LS_SCHUR_GMRES_RESTART",
    "NESTED_LS_SCHUR_GMRES_RTOL",
    "NESTED_LS_SHAMANSKII_REFINE_PASSES",
    "NestedLsB37TimingBlocked",
    "NestedLsBananaRunCodeLane",
    "NestedLsBananaRunCodePair",
    "NestedLsFourierBlockPreconditioner",
    "NestedLsHvpComparison",
    "NestedLsReducedNewtonResult",
    "NestedLsReducedRankError",
    "NestedLsSchurNewtonResult",
    "NestedLsSchurNewtonStepRecord",
    "NestedLsYSolution",
    "apply_reduced_mixed_schur_coil_tangent",
    "attempt_shamanskii_schur_reuse",
    "compare_ad_qr_and_schur_hvp",
    "dense_schur_inverse_preconditioner",
    "dense_schur_lu_preconditioner",
    "eisenstat_walker_forcing_eta",
    "factor_reduced_nested_ls_schur",
    "factor_schur_fourier_block_preconditioner",
    "implicit_adjoint_coil_gradient",
    "linear_solve_meets_forcing",
    "make_reduced_penalty_objective",
    "materialize_stabilized_schur_dense",
    "nested_ls_reduced_closures",
    "nested_ls_runtime_coil_closures",
    "pack_surface_and_y",
    "projected_y_system",
    "receipt_float",
    "reduced_penalty_gradient",
    "reduced_penalty_gradient_envelope",
    "reduced_penalty_hvp",
    "require_full_y_rank",
    "run_banana_run_code_pair",
    "run_reduced_nested_ls_newton",
    "run_reduced_nested_ls_schur_newton",
    "schur_dense_operator_bytes",
    "solve_operator_gmres_with_forcing",
    "solve_projected_y",
    "solve_stabilized_schur_dense_lu",
    "split_surface_and_y",
    "tensor_fourier_mode_blocks",
]
