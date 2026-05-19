"""Solve-level JAX wrappers for fixed-state permanent-magnet optimization."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..geo.permanent_magnet_grid_jax import PermanentMagnetGridJAX
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.pm_optimization import (
    GPMOArbVecBacktrackingSpec,
    GPMOArbVecSpec,
    GPMOBacktrackingSpec,
    GPMOBaselineSpec,
    GPMOMultiSpec,
    PMOptimizationSpec,
    gpmo_arbvec_backtracking_solve,
    gpmo_arbvec_solve,
    gpmo_backtracking_solve,
    gpmo_baseline_solve,
    gpmo_multi_solve,
    mwpgp_solve,
    projection_l2_balls,
)

__all__ = [
    "GPMOArbVecBacktrackingResult",
    "GPMOArbVecResult",
    "GPMOBacktrackingResult",
    "GPMOBaselineResult",
    "GPMOMultiResult",
    "GPMO_ArbVec_backtracking_jax",
    "GPMO_ArbVec_jax",
    "GPMO_backtracking_jax",
    "GPMO_baseline_jax",
    "GPMO_multi_jax",
    "PMRelaxAndSplitResult",
    "projection_L2_balls_jax",
    "prox_l0_jax",
    "prox_l1_jax",
    "relax_and_split_jax",
    "setup_initial_condition_jax",
]


def _moments_as_matrix(name: str, value: object, ndipoles: int) -> jax.Array:
    moments = _as_jax_float64(value)
    if moments.shape == (ndipoles * 3,):
        return jnp.reshape(moments, (ndipoles, 3))
    if moments.shape != (ndipoles, 3):
        raise ValueError(f"{name} must have shape ({ndipoles}, 3).")
    return moments


def _flatten_like(reference: jax.Array, moments: jax.Array) -> jax.Array:
    return jnp.reshape(moments, reference.shape)


def _normalized_moment_magnitudes(matrix: jax.Array, m_maxima: jax.Array) -> jax.Array:
    positive_mmax = m_maxima > 0.0
    safe_mmax = jnp.where(positive_mmax, m_maxima, 1.0)
    return jnp.where(
        positive_mmax[:, None],
        jnp.abs(matrix) / safe_mmax[:, None],
        0.0,
    )


def _is_tracing(value: object) -> bool:
    return isinstance(value, jax.core.Tracer)


def _has_tracer_leaf(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _raise_if_infeasible_initial_condition(
    moments: jax.Array, projected: jax.Array
) -> None:
    moments_host = np.asarray(jax.device_get(moments))
    projected_host = np.asarray(jax.device_get(projected))
    if not np.allclose(moments_host, projected_host):
        raise ValueError(
            "Initial dipole guess must contain values that satisfy the "
            "maximum bound constraints."
        )


def _host_scalar(name: str, value: jax.Array) -> float:
    array = np.asarray(jax.device_get(value))
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return float(array)


@dataclass(frozen=True)
class PMRelaxAndSplitResult:
    """Immutable result from ``relax_and_split_jax``."""

    errors: jax.Array
    m_history: jax.Array
    m_proxy_history: jax.Array
    m: jax.Array
    m_proxy: jax.Array
    residual_history: jax.Array


jax.tree_util.register_dataclass(
    PMRelaxAndSplitResult,
    data_fields=[
        "errors",
        "m_history",
        "m_proxy_history",
        "m",
        "m_proxy",
        "residual_history",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class GPMOBaselineResult:
    """Immutable result from ``GPMO_baseline_jax``."""

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array


jax.tree_util.register_dataclass(
    GPMOBaselineResult,
    data_fields=[
        "m",
        "m_history",
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_components",
        "selected_signs",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class GPMOMultiResult:
    """Immutable result from ``GPMO_multi_jax``."""

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_seed_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array
    selected_groups: jax.Array


jax.tree_util.register_dataclass(
    GPMOMultiResult,
    data_fields=[
        "m",
        "m_history",
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_seed_dipoles",
        "selected_components",
        "selected_signs",
        "selected_groups",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class GPMOBacktrackingResult:
    """Immutable result from ``GPMO_backtracking_jax``."""

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array
    num_nonzeros_history: jax.Array
    removed_pair_count_history: jax.Array
    done_history: jax.Array


jax.tree_util.register_dataclass(
    GPMOBacktrackingResult,
    data_fields=[
        "m",
        "m_history",
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_components",
        "selected_signs",
        "num_nonzeros_history",
        "removed_pair_count_history",
        "done_history",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class GPMOArbVecResult:
    """Immutable result from ``GPMO_ArbVec_jax``."""

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_vector_indices: jax.Array
    selected_signs: jax.Array


jax.tree_util.register_dataclass(
    GPMOArbVecResult,
    data_fields=[
        "m",
        "m_history",
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_vector_indices",
        "selected_signs",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class GPMOArbVecBacktrackingResult:
    """Immutable result from ``GPMO_ArbVec_backtracking_jax``."""

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_vector_indices: jax.Array
    selected_signs: jax.Array
    num_nonzeros_history: jax.Array
    removed_pair_count_history: jax.Array
    done_history: jax.Array
    initial_x: jax.Array
    initial_residual: jax.Array
    initial_num_nonzero: jax.Array


jax.tree_util.register_dataclass(
    GPMOArbVecBacktrackingResult,
    data_fields=[
        "m",
        "m_history",
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_vector_indices",
        "selected_signs",
        "num_nonzeros_history",
        "removed_pair_count_history",
        "done_history",
        "initial_x",
        "initial_residual",
        "initial_num_nonzero",
    ],
    meta_fields=[],
)


def projection_L2_balls_jax(x: object, mmax: object) -> jax.Array:
    """Project dipole moments onto per-dipole L2 balls, matching CPU shape."""

    moments = _as_jax_float64(x)
    m_maxima = _as_jax_float64(mmax)
    projected = projection_l2_balls(
        jnp.reshape(moments, (m_maxima.shape[0], 3)), m_maxima
    )
    return _flatten_like(moments, projected)


def prox_l0_jax(m: object, mmax: object, reg_l0: float, nu: float) -> jax.Array:
    """JAX port of the CPU ``prox_l0`` thresholding rule."""

    moments = _as_jax_float64(m)
    m_maxima = _as_jax_float64(mmax)
    matrix = jnp.reshape(moments, (m_maxima.shape[0], 3))
    normalized = _normalized_moment_magnitudes(matrix, m_maxima)
    thresholded = matrix * (normalized > (2.0 * reg_l0 * nu))
    return _flatten_like(moments, thresholded)


def prox_l1_jax(m: object, mmax: object, reg_l1: float, nu: float) -> jax.Array:
    """JAX port of the CPU ``prox_l1`` soft-thresholding rule."""

    moments = _as_jax_float64(m)
    m_maxima = _as_jax_float64(mmax)
    matrix = jnp.reshape(moments, (m_maxima.shape[0], 3))
    normalized = _normalized_moment_magnitudes(matrix, m_maxima)
    thresholded = (
        jnp.sign(matrix)
        * jnp.maximum(normalized - reg_l1 * nu, 0.0)
        * m_maxima[:, None]
    )
    return _flatten_like(moments, thresholded)


def setup_initial_condition_jax(
    grid: PermanentMagnetGridJAX, m0: object | None = None
) -> jax.Array:
    """Return the initial moment matrix for a fixed JAX PM grid.

    Explicit ``m0`` is an eager host-boundary input so infeasible values are
    rejected before a fixed-state solve starts. JIT callers should stage the
    validated initial condition in ``PermanentMagnetGridJAX.m0`` and pass
    ``m0=None``.
    """

    if m0 is None:
        return grid.m0
    moments = _moments_as_matrix("m0", m0, grid.ndipoles)
    if _is_tracing(moments):
        raise ValueError(
            "Explicit m0 validation requires an eager host-boundary call; "
            "stage the validated initial condition in PermanentMagnetGridJAX.m0 "
            "before JIT compilation."
        )
    projected = projection_l2_balls(moments, grid.m_maxima)
    _raise_if_infeasible_initial_condition(moments, projected)
    return moments


def _component_mmax(grid: PermanentMagnetGridJAX) -> jax.Array:
    return jnp.repeat(grid.m_maxima, 3)


def GPMO_baseline_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    single_direction: int = -1,
) -> GPMOBaselineResult:
    """Run the baseline greedy GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit baseline-only wrapper. Use ``GPMO_multi_jax`` for
    the multi-neighbour variant, ``GPMO_ArbVec_jax`` for arbitrary-vector
    placement, ``GPMO_backtracking_jax`` for baseline backtracking, and
    ``GPMO_ArbVec_backtracking_jax`` for the arbitrary-vector backtracking
    variant.
    """

    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_baseline_solve(
        GPMOBaselineSpec(
            m_maxima=grid.m_maxima,
            reg_l2=jnp.asarray(np.float64(reg_l2), dtype=grid.A_obj.dtype),
            single_direction=single_direction,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
    )
    m = core.x * grid.m_maxima[:, None]
    m_history = core.x_history * grid.m_maxima[None, :, None]
    return GPMOBaselineResult(
        m=m,
        m_history=m_history,
        x=core.x,
        x_history=core.x_history,
        residual=core.residual,
        residual_history=core.residual_history,
        selected_dipoles=core.selected_dipoles,
        selected_components=core.selected_components,
        selected_signs=core.selected_signs,
    )


def GPMO_multi_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    single_direction: int = -1,
    Nadjacent: int = 7,
) -> GPMOMultiResult:
    """Run the multi-neighbour greedy GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit multi-only wrapper. Use ``GPMO_ArbVec_jax`` for
    arbitrary-vector placement, ``GPMO_backtracking_jax`` for baseline
    backtracking, and ``GPMO_ArbVec_backtracking_jax`` for the
    arbitrary-vector backtracking variant.
    """

    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_multi_solve(
        GPMOMultiSpec(
            m_maxima=grid.m_maxima,
            reg_l2=jnp.asarray(np.float64(reg_l2), dtype=grid.A_obj.dtype),
            dipole_grid_xyz=grid.dipole_grid_xyz,
            single_direction=single_direction,
            Nadjacent=Nadjacent,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
    )
    m = core.x * grid.m_maxima[:, None]
    m_history = core.x_history * grid.m_maxima[None, :, None]
    return GPMOMultiResult(
        m=m,
        m_history=m_history,
        x=core.x,
        x_history=core.x_history,
        residual=core.residual,
        residual_history=core.residual_history,
        selected_seed_dipoles=core.selected_seed_dipoles,
        selected_components=core.selected_components,
        selected_signs=core.selected_signs,
        selected_groups=core.selected_groups,
    )


def GPMO_backtracking_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    single_direction: int = -1,
    Nadjacent: int = 7,
    backtracking: int = 100,
    max_nMagnets: int = 1000,
) -> GPMOBacktrackingResult:
    """Run the baseline backtracking GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit backtracking-only wrapper. The arbitrary-vector
    backtracking variant lives in ``GPMO_ArbVec_backtracking_jax``. Neither
    wrapper mutates a CPU grid.
    """

    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_backtracking_solve(
        GPMOBacktrackingSpec(
            m_maxima=grid.m_maxima,
            reg_l2=jnp.asarray(np.float64(reg_l2), dtype=grid.A_obj.dtype),
            dipole_grid_xyz=grid.dipole_grid_xyz,
            single_direction=single_direction,
            Nadjacent=Nadjacent,
            backtracking=backtracking,
            max_nMagnets=max_nMagnets,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
    )
    m = core.x * grid.m_maxima[:, None]
    m_history = core.x_history * grid.m_maxima[None, :, None]
    return GPMOBacktrackingResult(
        m=m,
        m_history=m_history,
        x=core.x,
        x_history=core.x_history,
        residual=core.residual,
        residual_history=core.residual_history,
        selected_dipoles=core.selected_dipoles,
        selected_components=core.selected_components,
        selected_signs=core.selected_signs,
        num_nonzeros_history=core.num_nonzeros_history,
        removed_pair_count_history=core.removed_pair_count_history,
        done_history=core.done_history,
    )


def _gpmo_pol_vectors(
    grid: PermanentMagnetGridJAX, pol_vectors: object | None
) -> jax.Array:
    if pol_vectors is not None:
        return _as_jax_float64(pol_vectors)
    if grid.pol_vectors is None:
        raise ValueError("GPMO_ArbVec_jax requires pol_vectors.")
    return grid.pol_vectors


def GPMO_ArbVec_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    pol_vectors: object | None = None,
) -> GPMOArbVecResult:
    """Run the arbitrary-vector greedy GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit arbitrary-vector-only wrapper. Use
    ``GPMO_ArbVec_backtracking_jax`` for the angle-thresholded backtracking
    variant.
    """

    pol_vectors_arr = _gpmo_pol_vectors(grid, pol_vectors)
    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_arbvec_solve(
        GPMOArbVecSpec(
            m_maxima=grid.m_maxima,
            reg_l2=jnp.asarray(np.float64(reg_l2), dtype=grid.A_obj.dtype),
            pol_vectors=pol_vectors_arr,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
    )
    m = core.x * grid.m_maxima[:, None]
    m_history = core.x_history * grid.m_maxima[None, :, None]
    return GPMOArbVecResult(
        m=m,
        m_history=m_history,
        x=core.x,
        x_history=core.x_history,
        residual=core.residual,
        residual_history=core.residual_history,
        selected_dipoles=core.selected_dipoles,
        selected_vector_indices=core.selected_vector_indices,
        selected_signs=core.selected_signs,
    )


def GPMO_ArbVec_backtracking_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    Nadjacent: int = 7,
    backtracking: int = 100,
    thresh_angle: float = 3.141592653589793,
    max_nMagnets: int = 1000,
    pol_vectors: object | None = None,
    m_init: object | None = None,
) -> GPMOArbVecBacktrackingResult:
    """Run the arbitrary-vector backtracking GPMO algorithm on a fixed JAX PM grid.

    Mirrors the ``algorithm='ArbVec_backtracking'`` branch of the CPU
    ``GPMO`` wrapper (see ``simsopt.solve.permanent_magnet_optimization``).
    The optional ``m_init`` input is converted to normalized ``x_init``
    coordinates via ``m_init / repeat(m_maxima, 3).reshape(N, 3)``,
    matching the upstream ``contig(kwargs['m_init'] / mmax_vec.reshape(N, 3))``
    transform.
    """

    pol_vectors_arr = _gpmo_pol_vectors(grid, pol_vectors)
    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    if m_init is None:
        x_init_arr = jnp.zeros((grid.ndipoles, 3), dtype=grid.A_obj.dtype)
    else:
        m_init_arr = _moments_as_matrix("m_init", m_init, grid.ndipoles)
        x_init_arr = m_init_arr / grid.m_maxima[:, None]
    core = gpmo_arbvec_backtracking_solve(
        GPMOArbVecBacktrackingSpec(
            m_maxima=grid.m_maxima,
            reg_l2=jnp.asarray(np.float64(reg_l2), dtype=grid.A_obj.dtype),
            dipole_grid_xyz=grid.dipole_grid_xyz,
            pol_vectors=pol_vectors_arr,
            Nadjacent=Nadjacent,
            backtracking=backtracking,
            thresh_angle=thresh_angle,
            max_nMagnets=max_nMagnets,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
        x_init=x_init_arr,
    )
    m = core.x * grid.m_maxima[:, None]
    m_history = core.x_history * grid.m_maxima[None, :, None]
    return GPMOArbVecBacktrackingResult(
        m=m,
        m_history=m_history,
        x=core.x,
        x_history=core.x_history,
        residual=core.residual,
        residual_history=core.residual_history,
        selected_dipoles=core.selected_dipoles,
        selected_vector_indices=core.selected_vector_indices,
        selected_signs=core.selected_signs,
        num_nonzeros_history=core.num_nonzeros_history,
        removed_pair_count_history=core.removed_pair_count_history,
        done_history=core.done_history,
        initial_x=core.initial_x,
        initial_residual=core.initial_residual,
        initial_num_nonzero=core.initial_num_nonzero,
    )


def _mwpgp_spec(
    grid: PermanentMagnetGridJAX,
    m_proxy: jax.Array,
    *,
    alpha: object | None,
    nu: float,
    reg_l2: float,
) -> PMOptimizationSpec:
    hessian_scale = grid.ATA_scale + jnp.asarray(
        np.float64(2.0 * reg_l2 + 1.0 / nu), dtype=grid.A_obj.dtype
    )
    if alpha is None:
        alpha_value = (
            jnp.asarray(2.0 * (1.0 - 1.0e-5), dtype=grid.A_obj.dtype) / hessian_scale
        )
    else:
        alpha_value = _as_jax_float64(alpha)
        if _has_tracer_leaf((alpha_value, hessian_scale)):
            raise ValueError(
                "Explicit alpha validation requires an eager host-boundary call."
            )
        alpha_host = _host_scalar("alpha", alpha_value)
        bound_host = _host_scalar("2/lambda_max(H)", 2.0 / hessian_scale)
        if alpha_host > bound_host:
            raise ValueError(
                "alpha must be <= 2/lambda_max(H) for MwPGP fixed-step "
                f"contraction; got {alpha_host} > {bound_host}."
            )
    return PMOptimizationSpec(
        m_maxima=grid.m_maxima,
        m_proxy=m_proxy,
        nu=jnp.asarray(np.float64(nu), dtype=grid.A_obj.dtype),
        reg_l2=jnp.asarray(np.float64(reg_l2), dtype=grid.A_obj.dtype),
        alpha=alpha_value,
    )


def _run_mwpgp(
    grid: PermanentMagnetGridJAX,
    m0: jax.Array,
    m_proxy: jax.Array,
    *,
    alpha: object | None,
    nu: float,
    reg_l2: float,
    max_iter: int,
) -> tuple[jax.Array, jax.Array]:
    spec = _mwpgp_spec(grid, m_proxy, alpha=alpha, nu=nu, reg_l2=reg_l2)
    return mwpgp_solve(spec, grid.A_obj, grid.ATb, m0, n_steps=max_iter)


def _last_error(
    residual_history: jax.Array, dtype: jnp.dtype, max_iter: int
) -> jax.Array:
    if max_iter == 0:
        return jnp.asarray(0.0, dtype=dtype)
    return residual_history[-1]


def _relax_and_split_cost(
    grid: PermanentMagnetGridJAX,
    m: jax.Array,
    m_proxy: jax.Array,
    *,
    nu: float,
    reg_l2: float,
) -> jax.Array:
    m_flat = jnp.reshape(m, (-1,))
    residual = grid.A_obj @ m_flat - grid.b_obj
    r2 = 0.5 * jnp.sum(residual * residual)
    n2 = (
        0.5
        * jnp.sum((m - m_proxy) * (m - m_proxy))
        / jnp.asarray(np.float64(nu), dtype=m.dtype)
    )
    l2 = jnp.asarray(np.float64(reg_l2), dtype=m.dtype) * jnp.sum(m * m)
    return r2 + n2 + l2


def relax_and_split_jax(
    grid: PermanentMagnetGridJAX,
    m0: object | None = None,
    *,
    alpha: object | None = None,
    max_iter: int = 100,
    max_iter_RS: int = 1,
    nu: float = 1.0e100,
    reg_l0: float = 0.0,
    reg_l1: float = 0.0,
    reg_l2: float = 0.0,
) -> PMRelaxAndSplitResult:
    """Run the fixed-step JAX relax-and-split PM solve wrapper.

    This is the solve-level adapter for the already-ported MwPGP kernel. It
    consumes an immutable ``PermanentMagnetGridJAX`` fixed-state payload and
    returns immutable arrays; it does not mutate a CPU ``PermanentMagnetGrid``.
    ``GPMO_baseline_jax``, ``GPMO_multi_jax``, ``GPMO_ArbVec_jax``,
    ``GPMO_backtracking_jax``, and ``GPMO_ArbVec_backtracking_jax`` cover the
    baseline, multi-neighbour, arbitrary-vector, baseline backtracking, and
    arbitrary-vector backtracking greedy variants respectively.
    """

    if max_iter_RS < 1:
        raise ValueError(f"max_iter_RS must be positive; got {max_iter_RS}")
    if (not np.isclose(reg_l0, 0.0, atol=1.0e-16)) and (
        not np.isclose(reg_l1, 0.0, atol=1.0e-16)
    ):
        raise ValueError("L0 and L1 loss terms cannot be used concurrently.")

    initial = setup_initial_condition_jax(grid, m0)
    no_l0 = np.isclose(reg_l0, 0.0, atol=1.0e-16)
    no_l1 = np.isclose(reg_l1, 0.0, atol=1.0e-16)
    if no_l0 and no_l1:
        m, residual_history = _run_mwpgp(
            grid,
            initial,
            initial,
            alpha=alpha,
            nu=nu,
            reg_l2=reg_l2,
            max_iter=max_iter,
        )
        return PMRelaxAndSplitResult(
            errors=jnp.reshape(_last_error(residual_history, m.dtype, max_iter), (1,)),
            m_history=m[None, :, :],
            m_proxy_history=jnp.empty((0, grid.ndipoles, 3), dtype=m.dtype),
            m=m,
            m_proxy=m,
            residual_history=residual_history[None, :],
        )

    if no_l0:
        prox = prox_l1_jax
        reg_rs = reg_l1
    else:
        prox = prox_l0_jax
        reg_rs = reg_l0

    m = initial
    m_proxy = _moments_as_matrix(
        "m_proxy", prox(initial, grid.m_maxima, reg_rs, nu), grid.ndipoles
    )
    errors = []
    m_history = []
    m_proxy_history = []
    residual_histories = []
    for _ in range(max_iter_RS):
        m_proxy_current = m_proxy
        m, residual_history = _run_mwpgp(
            grid,
            m,
            m_proxy_current,
            alpha=alpha,
            nu=nu,
            reg_l2=reg_l2,
            max_iter=max_iter,
        )
        m_history.append(m)
        errors.append(
            _relax_and_split_cost(
                grid,
                m,
                m_proxy_current,
                nu=nu,
                reg_l2=reg_l2,
            )
        )
        m_proxy = _moments_as_matrix(
            "m_proxy", prox(m, grid.m_maxima, reg_rs, nu), grid.ndipoles
        )
        m_proxy_history.append(m_proxy)
        residual_histories.append(residual_history)

    return PMRelaxAndSplitResult(
        errors=jnp.stack(errors),
        m_history=jnp.stack(m_history),
        m_proxy_history=jnp.stack(m_proxy_history),
        m=m,
        m_proxy=m_proxy,
        residual_history=jnp.stack(residual_histories),
    )
