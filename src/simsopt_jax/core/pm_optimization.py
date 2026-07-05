"""JAX port of ``simsoptpp::MwPGP_algorithm`` (Tier P4 item 25).

This module implements the Mixed-Active-Set Projected Gradient (MwPGP) solver
for the convex permanent-magnet optimization sub-problem

``min_m  1/2 * ||A m - b||^2 + (reg_l2 + 1/(2 nu)) ||m||^2
         - (1/nu) <m_proxy, m>  s.t.  ||m_i||_2 <= m_maxima_i for each dipole``.

The matrix ``A`` has shape ``(M, 3 N)`` (``M`` plasma-surface quadrature
points, ``N`` dipoles each carrying a 3-vector moment). ``b`` has shape
``(M,)``. The decision variable ``m`` is stored as shape ``(N, 3)``.

The **MwPGP_algorithm** kernel is ported for the convex inner solver used
inside the relax-and-split outer loop. The baseline, multi-neighbour,
backtracking, arbitrary-vector, and arbitrary-vector backtracking GPMO greedy
variants are all ported as fixed-step, pure-JAX state machines.

Reference: Bouchala et al., *On the solution of convex QPQC problems with
elliptic and other separable constraints with strong curvature*, Applied
Mathematics and Computation 247 (2014) 848-864.

Upstream oracle: ``src/simsoptpp/permanent_magnet_optimization.cpp`` lines 11-324.

Algorithm contract
------------------

The C++ kernel exposes a dynamic-stopping API
(``epsilon`` / ``max_iter``) plus an internal history buffer. JAX
``jax.lax.scan`` requires a static iteration count and rejects host-side
break logic. The port therefore exposes the **fixed-step** kernel
``mwpgp_solve(spec, A, b, m0, n_steps=...)`` that mirrors the C++ algebra
exactly for the first ``n_steps`` iterations and never short-circuits.
``mwpgp_step`` exposes the single-iteration transition for testing and
manual unrolling.

Per-dipole L2-ball projection, the ``phi``/``beta_tilde``/``g_reduced_*``
helpers, and the quadratic ``find_max_alphaf`` are implemented as pure
batched ``jax.numpy`` operations so the entire solver is JIT-compatible
(no Python-level branching on tracer values).

Shape contract
--------------

The ordinary kernels are fixed-shape kernels: ``N`` (dipoles), ``M`` (surface
quadrature points), and the arbitrary-vector count ``P`` are array dimensions,
so changing them produces a separate JAX compilation. Batch callers should
group ordinary runs by exact shape or keep a compiled callable per geometry.

``gpmo_arbvec_solve_bucketed`` is the reusable-shape path for arbitrary-vector
GPMO batches whose active ``N``/``P`` vary within a fixed bucket. Callers stage
zero-padded arrays to the bucket dimensions and pass active counts as tensor
inputs; inactive dipoles/vectors are masked inside the greedy candidate scan so
the same compiled executable can serve smaller PM grids without changing array
shapes.

The Hessian action is the closed form
``H v = A^T A v + 2 (reg_l2 + 1/(2 nu)) v``
matching ``permanent_magnet_optimization.cpp:187`` and ``:216``. ``A`` is
treated as a static input (shape captured at trace time); we never
materialise ``A^T A``, which would cost ``O((3 N)^2)`` memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._math_utils import has_tracer_leaf as _has_tracer_leaf
from ._math_utils import (
    as_runtime_array as _as_runtime_array,
    as_runtime_value as _as_runtime_value,
    as_jax_int32 as _as_jax_int32,
)

__all__ = [
    "GPMOArbVecBacktrackingResult",
    "GPMOArbVecBacktrackingSpec",
    "GPMOBaselineResult",
    "GPMOBaselineSpec",
    "GPMOArbVecResult",
    "GPMOArbVecSpec",
    "GPMOBacktrackingResult",
    "GPMOBacktrackingSpec",
    "GPMOMultiResult",
    "GPMOMultiSpec",
    "PMOptimizationSpec",
    "find_max_alphaf",
    "gpmo_connectivity_matrix",
    "gpmo_baseline_candidate_costs",
    "gpmo_baseline_solve",
    "gpmo_baseline_step",
    "gpmo_arbvec_candidate_costs",
    "gpmo_arbvec_solve_bucketed",
    "gpmo_arbvec_solve",
    "gpmo_arbvec_step",
    "gpmo_arbvec_backtracking_solve",
    "gpmo_arbvec_backtracking_step",
    "gpmo_backtracking_solve",
    "gpmo_backtracking_step",
    "gpmo_multi_candidate_costs",
    "gpmo_multi_solve",
    "gpmo_multi_step",
    "g_reduced_gradient",
    "g_reduced_projected_gradient",
    "initialize_gpmo_arbvec",
    "mwpgp_initial_state",
    "mwpgp_solve",
    "mwpgp_step",
    "phi_mwpgp",
    "projection_l2_balls",
]


# Tolerance constants matching ``permanent_magnet_optimization.cpp``:
# ``abs(xmag2 - mmax^2) > 1.0e-8 + 1.0e-5 * mmax^2`` decides whether a
# dipole is "on" or "off" the L2 ball. See ``phi_MwPGP`` (line 22) and
# ``beta_tilde`` (line 42).
_BALL_ACTIVE_ABS_TOL: float = 1.0e-8
_BALL_ACTIVE_REL_TOL: float = 1.0e-5
# Floor for ``a = ||p||^2`` in the quadratic ``find_max_alphaf``. Matches
# ``double tol = 1e-20;`` in ``permanent_magnet_optimization.cpp:80``.
_FIND_MAX_ALPHAF_TOL: float = 1.0e-20
# Sentinel value used in the C++ kernel for "no boundary hit". Matches
# ``alphaf_plus = 1e100;`` in ``permanent_magnet_optimization.cpp:90``.
_FIND_MAX_ALPHAF_SENTINEL: float = 1.0e100
_UNAVAILABLE_CANDIDATE_COST: float = float("inf")


def _scalar_like(reference: jax.Array, value: float) -> jax.Array:
    return _as_runtime_value(value, reference=reference, dtype=reference.dtype)


def _device_scalar(value: object, dtype) -> jax.Array:
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))


def _bool_scalar(value: bool) -> jax.Array:
    return _device_scalar(value, np.bool_)


def _find_max_alphaf_sentinel(dtype) -> float:
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.floating) and _FIND_MAX_ALPHAF_SENTINEL > float(
        np.finfo(target_dtype).max
    ):
        return float("inf")
    return _FIND_MAX_ALPHAF_SENTINEL


def _zeros_like_shape(reference: jax.Array, shape: tuple[int, ...], *, dtype=None):
    target_dtype = reference.dtype if dtype is None else dtype
    zero = jnp.sum(reference - reference).astype(target_dtype)
    return jnp.broadcast_to(zero, shape)


def _false_like_shape(reference: jax.Array, shape: tuple[int, ...]):
    zero = jnp.sum(reference - reference)
    return jnp.broadcast_to(zero != zero, shape)


def _negative_ones_int_like_shape(reference: jax.Array, shape: tuple[int, ...]):
    zero = jnp.sum(reference - reference).astype(reference.dtype)
    one = (zero == zero).astype(reference.dtype)
    return jnp.broadcast_to(zero - one, shape)


def _int_scalar_like(reference: jax.Array, value: int) -> jax.Array:
    return _device_scalar(value, reference.dtype)


def _argmin_finite_cost(costs: jax.Array, *, axis: int | None = None) -> jax.Array:
    """Return the minimum finite candidate, sorting NaNs after valid costs."""
    sentinel = _scalar_like(costs, _UNAVAILABLE_CANDIDATE_COST)
    finite_costs = jnp.where(jnp.isfinite(costs), costs, sentinel)
    return jnp.argmin(finite_costs, axis=axis)


def _host_scalar_for_validation(name: str, value, *, dtype=None):
    if _has_tracer_leaf(value):
        return None
    host_value = np.asarray(jax.device_get(value), dtype=dtype)
    if host_value.shape != ():
        raise ValueError(f"{name} must be a scalar; got shape {host_value.shape}.")
    return host_value


def _validate_positive_scalar(name: str, value) -> None:
    host_value = _host_scalar_for_validation(name, value, dtype=np.float64)
    if host_value is None:
        return
    if not bool(host_value > 0.0):
        raise ValueError(f"{name} must be positive; got {host_value.item()}.")


@dataclass(frozen=True)
class PMOptimizationSpec:
    """Immutable payload for the JAX MwPGP solver.

    Parameters
    ----------
    m_maxima
        Per-dipole L2-ball radii, shape ``(N,)``. Units Am^2.
    m_proxy
        Proxy moments used by the relax-and-split term, shape ``(N, 3)``.
        Set to zero (or pass-through) when the relax-and-split term is
        not used (``nu`` very large).
    nu
        Relax-and-split coupling. Large ``nu`` (``1e100`` upstream
        default) effectively turns the relax-and-split contribution off.
    reg_l2
        L2 regularisation weight on ``||m||^2``.
    alpha
        Fixed step size for the projected-gradient sweep. Must satisfy
        ``alpha < 2 / lambda_max(A^T A + 2 (reg_l2 + 1/(2 nu)) I)`` for
        the underlying algorithm to be a contraction. The C++ caller is
        responsible for choosing ``alpha``; see ``PermanentMagnetGrid``
        upstream which uses ``2 / sigma_1(A)^2``.
    """

    m_maxima: jax.Array
    m_proxy: jax.Array
    nu: jax.Array
    reg_l2: jax.Array
    alpha: jax.Array

    def __post_init__(self) -> None:
        _validate_positive_scalar("nu", self.nu)
        _validate_positive_scalar("alpha", self.alpha)


jax.tree_util.register_dataclass(
    PMOptimizationSpec,
    data_fields=["m_maxima", "m_proxy", "nu", "reg_l2", "alpha"],
    meta_fields=[],
)


@dataclass(frozen=True)
class GPMOBaselineSpec:
    """Immutable payload for the JAX baseline GPMO solver.

    Parameters
    ----------
    m_maxima
        Per-dipole physical moment scales, shape ``(N,)``. The baseline
        algorithm works in normalized coordinates ``x_i in {-1, 0, 1}`` and
        chooses columns of ``A * repeat(m_maxima, 3)``.
    reg_l2
        L2 regularization weight used in the C++ baseline candidate score.
    single_direction
        ``-1`` allows all Cartesian components. ``0``, ``1``, or ``2`` limits
        placement to one component per dipole, matching the C++ loop stride.
    """

    m_maxima: jax.Array
    reg_l2: jax.Array
    single_direction: int = -1


jax.tree_util.register_dataclass(
    GPMOBaselineSpec,
    data_fields=["m_maxima", "reg_l2"],
    meta_fields=["single_direction"],
)


@dataclass(frozen=True)
class GPMOMultiSpec:
    """Immutable payload for the JAX multi-neighbour GPMO solver.

    ``Nadjacent`` is the number of nearest available dipoles placed at each
    greedy iteration. The nearest-neighbour order includes the seed dipole
    itself, matching ``connectivity_matrix`` in the C++ oracle.
    """

    m_maxima: jax.Array
    reg_l2: jax.Array
    dipole_grid_xyz: jax.Array
    single_direction: int = -1
    Nadjacent: int = 7


jax.tree_util.register_dataclass(
    GPMOMultiSpec,
    data_fields=["m_maxima", "reg_l2", "dipole_grid_xyz"],
    meta_fields=["single_direction", "Nadjacent"],
)


@dataclass(frozen=True)
class GPMOBacktrackingSpec:
    """Immutable payload for the JAX backtracking GPMO solver.

    ``backtracking`` controls how often adjacent equal-and-opposite magnets are
    removed. ``max_nMagnets`` mirrors the CPU stopping limit; the fixed-scan
    JAX solver carries a ``done`` mask after this limit is reached rather than
    breaking out of the compiled loop.
    """

    m_maxima: jax.Array
    reg_l2: jax.Array
    dipole_grid_xyz: jax.Array
    single_direction: int = -1
    Nadjacent: int = 7
    backtracking: int = 100
    max_nMagnets: int = 1000


jax.tree_util.register_dataclass(
    GPMOBacktrackingSpec,
    data_fields=["m_maxima", "reg_l2", "dipole_grid_xyz"],
    meta_fields=["single_direction", "Nadjacent", "backtracking", "max_nMagnets"],
)


@dataclass(frozen=True)
class GPMOArbVecSpec:
    """Immutable payload for the JAX arbitrary-vector GPMO solver."""

    m_maxima: jax.Array
    reg_l2: jax.Array
    pol_vectors: jax.Array


jax.tree_util.register_dataclass(
    GPMOArbVecSpec,
    data_fields=["m_maxima", "reg_l2", "pol_vectors"],
    meta_fields=[],
)


@dataclass(frozen=True)
class GPMOArbVecBacktrackingSpec:
    """Immutable payload for the JAX arbitrary-vector backtracking GPMO solver.

    ``thresh_angle`` is the half-angle threshold (radians) used by the
    dewyrming pass: a placed pair is removed when ``cos_angle`` between the
    seed dipole and its most-anti-aligned placed neighbor drops below
    ``cos(thresh_angle)``. Matches the C++ test
    ``min_cos_angle <= cos_thresh_angle``.
    """

    m_maxima: jax.Array
    reg_l2: jax.Array
    dipole_grid_xyz: jax.Array
    pol_vectors: jax.Array
    Nadjacent: int = 7
    backtracking: int = 100
    thresh_angle: float = 3.141592653589793
    max_nMagnets: int = 1000


jax.tree_util.register_dataclass(
    GPMOArbVecBacktrackingSpec,
    data_fields=["m_maxima", "reg_l2", "dipole_grid_xyz", "pol_vectors"],
    meta_fields=["Nadjacent", "backtracking", "thresh_angle", "max_nMagnets"],
)


@dataclass(frozen=True)
class GPMOBaselineResult:
    """Result from ``gpmo_baseline_solve`` in normalized coordinates."""

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
class GPMOArbVecResult:
    """Result from ``gpmo_arbvec_solve`` in normalized coordinates."""

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
class GPMOMultiResult:
    """Result from ``gpmo_multi_solve`` in normalized coordinates."""

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
    """Result from ``gpmo_backtracking_solve`` in normalized coordinates."""

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
class GPMOArbVecBacktrackingResult:
    """Result from ``gpmo_arbvec_backtracking_solve`` in normalized coordinates.

    ``x`` is the per-dipole moment unit-vector (``sign * pol_vectors[j, m_j]``)
    so the physical moment is recovered as ``x * m_maxima[:, None]``.

    Initialization (when ``x_init`` is nonzero) places dipoles BEFORE the main
    K-iteration loop runs. ``selected_*`` traces describe the iterations only;
    ``initial_*`` mirrors the placement made by ``initialize_gpmo_arbvec``.
    """

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


def _validate_gpmo_static_args(K: int, single_direction: int, ndipoles: int) -> None:
    if not isinstance(K, int):
        raise TypeError(f"K must be a Python int; got {type(K).__name__}")
    if K < 0:
        raise ValueError(f"K must be non-negative; got {K}")
    if K > ndipoles:
        raise ValueError(f"K must be <= ndipoles ({ndipoles}); got {K}")
    if single_direction not in (-1, 0, 1, 2):
        raise ValueError(
            f"single_direction must be -1, 0, 1, or 2; got {single_direction}"
        )


def _validate_record_every(record_every: int | None) -> None:
    if record_every is None:
        return
    if not isinstance(record_every, int):
        raise TypeError(
            f"record_every must be a Python int; got {type(record_every).__name__}"
        )
    if record_every < 1:
        raise ValueError(f"record_every must be positive; got {record_every}.")


def _record_rows(K: int, record_every: int) -> np.ndarray:
    rows = np.arange(record_every - 1, K, record_every, dtype=np.int32)
    if rows.size == 0 or rows[-1] != K - 1:
        rows = np.concatenate([rows, np.asarray([K - 1], dtype=np.int32)])
    return rows


def _gpmo_recording_scan(
    step_fn,
    initial_state: tuple[jax.Array, ...],
    reference: jax.Array,
    *,
    K: int,
    record_every: int,
    ndipoles: int,
    extra_history_specs: tuple[tuple[tuple[int, ...], object], ...],
    extra_trace_indices: tuple[int, ...],
) -> tuple[
    tuple[jax.Array, ...],
    jax.Array,
    jax.Array,
    tuple[jax.Array, ...],
]:
    record_rows = _record_rows(K, record_every)
    record_count = int(record_rows.size)
    record_rows_arr = _as_jax_int32(record_rows)

    def _recording_scan_body(carry, iteration):
        state, next_record_slot, x_history, residual_history, extra_histories = carry
        next_state, trace = step_fn(state, iteration)
        safe_slot = jnp.minimum(next_record_slot, record_count - 1)
        should_record = iteration == record_rows_arr[safe_slot]
        x_history_candidate = x_history.at[safe_slot].set(next_state[0])
        residual_history_candidate = residual_history.at[safe_slot].set(trace[3])
        extra_history_candidates = tuple(
            history.at[safe_slot].set(trace[trace_index])
            for history, trace_index in zip(extra_histories, extra_trace_indices)
        )
        return (
            next_state,
            next_record_slot + should_record.astype(next_record_slot.dtype),
            jnp.where(should_record, x_history_candidate, x_history),
            jnp.where(should_record, residual_history_candidate, residual_history),
            tuple(
                jnp.where(should_record, candidate, history)
                for candidate, history in zip(extra_history_candidates, extra_histories)
            ),
        ), None

    final_carry, _ = jax.lax.scan(
        _recording_scan_body,
        (
            initial_state,
            _zeros_like_shape(reference, (), dtype=jnp.int32),
            _zeros_like_shape(reference, (record_count, ndipoles, 3)),
            _zeros_like_shape(reference, (record_count,)),
            tuple(
                _zeros_like_shape(
                    reference,
                    (record_count, *shape),
                    dtype=dtype,
                )
                for shape, dtype in extra_history_specs
            ),
        ),
        jax.lax.iota(jnp.int32, K),
    )
    final_state, _next_record_slot, x_history, residual_history, extra_histories = (
        final_carry
    )
    return final_state, x_history, residual_history, extra_histories


def _validate_gpmo_multi_static_args(
    K: int, single_direction: int, ndipoles: int, Nadjacent: int
) -> None:
    _validate_gpmo_static_args(K, single_direction, ndipoles)
    if not isinstance(Nadjacent, int):
        raise TypeError(
            f"Nadjacent must be a Python int; got {type(Nadjacent).__name__}"
        )
    if Nadjacent < 1:
        raise ValueError(f"Nadjacent must be positive; got {Nadjacent}")
    if Nadjacent > ndipoles:
        raise ValueError(f"Nadjacent must be <= ndipoles ({ndipoles}); got {Nadjacent}")
    if K * Nadjacent > ndipoles:
        raise ValueError(
            "K * Nadjacent must not exceed ndipoles for fixed-step GPMO_multi; "
            f"got K={K}, Nadjacent={Nadjacent}, ndipoles={ndipoles}"
        )


def _validate_gpmo_backtracking_static_args(
    K: int,
    single_direction: int,
    ndipoles: int,
    Nadjacent: int,
    backtracking: int,
    max_nMagnets: int,
) -> None:
    if not isinstance(K, int):
        raise TypeError(f"K must be a Python int; got {type(K).__name__}")
    if K < 0:
        raise ValueError(f"K must be non-negative; got {K}")
    if single_direction not in (-1, 0, 1, 2):
        raise ValueError(
            f"single_direction must be -1, 0, 1, or 2; got {single_direction}"
        )
    if not isinstance(Nadjacent, int):
        raise TypeError(
            f"Nadjacent must be a Python int; got {type(Nadjacent).__name__}"
        )
    if Nadjacent < 1:
        raise ValueError(f"Nadjacent must be positive; got {Nadjacent}")
    if Nadjacent > ndipoles:
        raise ValueError(f"Nadjacent must be <= ndipoles ({ndipoles}); got {Nadjacent}")
    if not isinstance(backtracking, int):
        raise TypeError(
            f"backtracking must be a Python int; got {type(backtracking).__name__}"
        )
    if backtracking < 1:
        raise ValueError(f"backtracking must be positive; got {backtracking}")
    if not isinstance(max_nMagnets, int):
        raise TypeError(
            f"max_nMagnets must be a Python int; got {type(max_nMagnets).__name__}"
        )
    if max_nMagnets < 1:
        raise ValueError(f"max_nMagnets must be positive; got {max_nMagnets}")


def _validate_gpmo_arbvec_static_args(
    K: int, ndipoles: int, pol_vectors: jax.Array
) -> None:
    _validate_gpmo_static_args(K, -1, ndipoles)
    if pol_vectors.ndim != 3:
        raise ValueError("pol_vectors must have shape (ndipoles, n_vectors, 3).")
    if pol_vectors.shape[0] != ndipoles:
        raise ValueError(
            "pol_vectors first dimension must match ndipoles; "
            f"got {pol_vectors.shape[0]} and {ndipoles}."
        )
    if pol_vectors.shape[2] != 3:
        raise ValueError("pol_vectors third dimension must be 3.")


def _validate_gpmo_bucket_count(name: str, value, bucket_size: int) -> None:
    host_value = _host_scalar_for_validation(name, value)
    if host_value is None:
        return
    if host_value.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must be an integer scalar; got {host_value.dtype}.")
    count = int(host_value.item())
    if count < 0 or count > bucket_size:
        raise ValueError(
            f"{name} must satisfy 0 <= {name} <= {bucket_size}; got {count}."
        )


def _validate_gpmo_arbvec_backtracking_static_args(
    K: int,
    ndipoles: int,
    pol_vectors: jax.Array,
    Nadjacent: int,
    backtracking: int,
    max_nMagnets: int,
    thresh_angle: float,
) -> None:
    if not isinstance(K, int):
        raise TypeError(f"K must be a Python int; got {type(K).__name__}")
    if K < 0:
        raise ValueError(f"K must be non-negative; got {K}")
    if pol_vectors.ndim != 3:
        raise ValueError("pol_vectors must have shape (ndipoles, n_vectors, 3).")
    if pol_vectors.shape[0] != ndipoles:
        raise ValueError(
            "pol_vectors first dimension must match ndipoles; "
            f"got {pol_vectors.shape[0]} and {ndipoles}."
        )
    if pol_vectors.shape[2] != 3:
        raise ValueError("pol_vectors third dimension must be 3.")
    if not isinstance(Nadjacent, int):
        raise TypeError(
            f"Nadjacent must be a Python int; got {type(Nadjacent).__name__}"
        )
    if Nadjacent < 1:
        raise ValueError(f"Nadjacent must be positive; got {Nadjacent}")
    if Nadjacent > ndipoles:
        raise ValueError(f"Nadjacent must be <= ndipoles ({ndipoles}); got {Nadjacent}")
    if not isinstance(backtracking, int):
        raise TypeError(
            f"backtracking must be a Python int; got {type(backtracking).__name__}"
        )
    if backtracking < 1:
        raise ValueError(f"backtracking must be positive; got {backtracking}")
    if not isinstance(max_nMagnets, int):
        raise TypeError(
            f"max_nMagnets must be a Python int; got {type(max_nMagnets).__name__}"
        )
    if max_nMagnets < 1:
        raise ValueError(f"max_nMagnets must be positive; got {max_nMagnets}")
    if not isinstance(thresh_angle, (int, float)):
        raise TypeError(
            f"thresh_angle must be a Python float; got {type(thresh_angle).__name__}"
        )


def _component_mmax(m_maxima: jax.Array) -> jax.Array:
    return jnp.repeat(m_maxima, 3)


def _single_direction_mask(n_components: int, single_direction: int) -> jax.Array:
    components = jax.lax.rem(jax.lax.iota(jnp.int32, n_components), np.int32(3))
    if single_direction < 0:
        return components == components
    return components == np.int32(single_direction)


def gpmo_baseline_candidate_costs(
    spec: GPMOBaselineSpec,
    A_scaled: jax.Array,
    residual: jax.Array,
    available: jax.Array,
    *,
    penalty: jax.Array | None = None,
    col_sq: jax.Array | None = None,
) -> jax.Array:
    """Return baseline GPMO plus/minus candidate costs.

    Mirrors ``GPMO_baseline`` in
    ``src/simsoptpp/permanent_magnet_optimization.cpp:1270-1292``. The returned
    vector has shape ``(6 N,)`` with all ``+`` candidates first followed by all
    ``-`` candidates, matching the C++ ``std::min_element`` tie order.
    """

    A_arr = _as_runtime_array(A_scaled)
    residual_arr = _as_runtime_array(residual)
    m_maxima = _as_runtime_array(spec.m_maxima)
    reg_l2 = _as_runtime_array(spec.reg_l2)

    n_components = A_arr.shape[1]
    penalty_arr = (
        reg_l2 * _component_mmax(m_maxima) ** 2
        if penalty is None
        else _as_runtime_array(penalty)
    )
    residual_sq = jnp.sum(residual_arr * residual_arr)
    dot = A_arr.T @ residual_arr
    col_sq_arr = (
        jnp.sum(A_arr * A_arr, axis=0)
        if col_sq is None
        else _as_runtime_array(col_sq)
    )
    plus = residual_sq + 2.0 * dot + col_sq_arr + penalty_arr
    minus = residual_sq - 2.0 * dot + col_sq_arr + penalty_arr

    available_components = jnp.reshape(available, (n_components,))
    direction_mask = _single_direction_mask(n_components, spec.single_direction)
    allowed = available_components & direction_mask
    sentinel = _scalar_like(A_arr, _UNAVAILABLE_CANDIDATE_COST)
    plus = jnp.where(allowed, plus, sentinel)
    minus = jnp.where(allowed, minus, sentinel)
    return jnp.concatenate([plus, minus])


def gpmo_baseline_step(
    spec: GPMOBaselineSpec,
    state: tuple[jax.Array, jax.Array, jax.Array],
    A_scaled: jax.Array,
    *,
    penalty: jax.Array | None = None,
    col_sq: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Run one normalized baseline GPMO placement step."""

    x, residual, available = state
    costs = gpmo_baseline_candidate_costs(
        spec,
        A_scaled,
        residual,
        available,
        penalty=penalty,
        col_sq=col_sq,
    )
    n_components = A_scaled.shape[1]
    choice = _argmin_finite_cost(costs)
    is_minus = choice >= n_components
    component_index = jnp.where(is_minus, choice - n_components, choice)
    sign = jnp.where(is_minus, -1.0, 1.0).astype(residual.dtype)
    dipole = component_index // 3
    component = component_index % 3

    x_new = x.at[dipole, component].set(sign)
    residual_new = residual + sign * A_scaled[:, component_index]
    available_new = available.at[dipole, :].set(False)
    residual_sq = jnp.sum(residual_new * residual_new)
    return (
        x_new,
        residual_new,
        available_new,
    ), (
        dipole,
        component,
        sign,
        residual_sq,
    )


def _baseline_x_history(
    selected_dipoles: jax.Array,
    selected_components: jax.Array,
    selected_signs: jax.Array,
    *,
    ndipoles: int,
) -> jax.Array:
    """Reconstruct post-step normalized states from the baseline trace."""
    x0 = _zeros_like_shape(selected_signs, (ndipoles, 3))

    def _scan_body(x_state, trace_entry):
        dipole, component, sign = trace_entry
        x_new = x_state.at[dipole, component].set(sign)
        return x_new, x_new

    _, x_history = jax.lax.scan(
        _scan_body, x0, (selected_dipoles, selected_components, selected_signs)
    )
    return x_history


@partial(jax.jit, static_argnames=("K", "record_every"))
def gpmo_baseline_solve(
    spec: GPMOBaselineSpec,
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    K: int,
    record_every: int | None = None,
) -> GPMOBaselineResult:
    """Run the baseline greedy permanent-magnet optimizer in JAX.

    ``A_scaled`` has shape ``(M, 3N)`` and must already include the physical
    moment scaling by ``repeat(m_maxima, 3)``. The returned ``x`` is normalized,
    with at most one nonzero Cartesian component per selected dipole.
    """

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    m_maxima = _as_runtime_array(spec.m_maxima)
    reg_l2 = _as_runtime_array(spec.reg_l2)
    ndipoles = int(m_maxima.shape[0])
    _validate_gpmo_static_args(K, spec.single_direction, ndipoles)
    _validate_record_every(record_every)

    x0 = _zeros_like_shape(A_arr, (ndipoles, 3))
    residual0 = -b_arr
    available0 = x0 == x0
    if K == 0:
        empty_int = _zeros_like_shape(A_arr, (0,), dtype=jnp.int64)
        empty_float = _zeros_like_shape(A_arr, (0,))
        empty_x_history = _zeros_like_shape(A_arr, (0, ndipoles, 3))
        return GPMOBaselineResult(
            x=x0,
            x_history=empty_x_history,
            residual=residual0,
            residual_history=empty_float,
            selected_dipoles=empty_int,
            selected_components=empty_int,
            selected_signs=empty_float,
        )
    penalty = reg_l2 * _component_mmax(m_maxima) ** 2
    col_sq = jnp.sum(A_arr * A_arr, axis=0)
    if record_every is not None:

        def _recording_step(state, _iteration):
            return gpmo_baseline_step(
                spec,
                state,
                A_arr,
                penalty=penalty,
                col_sq=col_sq,
            )

        final_state, x_history, residual_history, extra_histories = (
            _gpmo_recording_scan(
                _recording_step,
                (x0, residual0, available0),
                A_arr,
                K=K,
                record_every=record_every,
                ndipoles=ndipoles,
                extra_history_specs=(
                    ((), jnp.int64),
                    ((), jnp.int64),
                    ((), A_arr.dtype),
                ),
                extra_trace_indices=(0, 1, 2),
            )
        )
        selected_dipoles, selected_components, selected_signs = extra_histories
        return GPMOBaselineResult(
            x=final_state[0],
            x_history=x_history,
            residual=final_state[1],
            residual_history=residual_history,
            selected_dipoles=selected_dipoles,
            selected_components=selected_components,
            selected_signs=selected_signs,
        )

    def _scan_body(state, _):
        return gpmo_baseline_step(
            spec,
            state,
            A_arr,
            penalty=penalty,
            col_sq=col_sq,
        )

    final_state, trace = jax.lax.scan(
        _scan_body, (x0, residual0, available0), xs=None, length=K
    )
    selected_dipoles, selected_components, selected_signs, residual_history = trace
    return GPMOBaselineResult(
        x=final_state[0],
        x_history=_baseline_x_history(
            selected_dipoles,
            selected_components,
            selected_signs,
            ndipoles=ndipoles,
        ),
        residual=final_state[1],
        residual_history=residual_history,
        selected_dipoles=selected_dipoles,
        selected_components=selected_components,
        selected_signs=selected_signs,
    )


def gpmo_connectivity_matrix(dipole_grid_xyz: jax.Array) -> jax.Array:
    """Return nearest-neighbour dipole indices, including each dipole itself."""

    xyz = _as_runtime_array(dipole_grid_xyz)
    deltas = xyz[:, None, :] - xyz[None, :, :]
    distances = jnp.sqrt(jnp.sum(deltas * deltas, axis=2))
    return jnp.argsort(distances, axis=1, stable=True)


def _gpmo_arbvec_contributions(
    A_scaled: jax.Array, pol_vectors: jax.Array
) -> jax.Array:
    A_by_dipole = jnp.reshape(A_scaled, (A_scaled.shape[0], pol_vectors.shape[0], 3))
    return jnp.einsum("mnl,npl->mnp", A_by_dipole, pol_vectors)


def _gpmo_arbvec_candidate_costs_for_allowed(
    spec: GPMOArbVecSpec,
    contributions: jax.Array,
    residual: jax.Array,
    allowed: jax.Array,
    *,
    col_sq: jax.Array | None = None,
    penalty: jax.Array | None = None,
) -> jax.Array:
    residual_arr = _as_runtime_array(residual)
    m_maxima = _as_runtime_array(spec.m_maxima)
    reg_l2 = _as_runtime_array(spec.reg_l2)
    contributions_arr = _as_runtime_array(contributions)

    residual_sq = jnp.sum(residual_arr * residual_arr)
    dot = jnp.einsum("m,mnp->np", residual_arr, contributions_arr)
    col_sq_arr = (
        jnp.sum(contributions_arr * contributions_arr, axis=0)
        if col_sq is None
        else _as_runtime_array(col_sq)
    )
    plus = residual_sq + 2.0 * dot + col_sq_arr
    minus = residual_sq - 2.0 * dot + col_sq_arr

    # ``GPMO_ArbVec`` indexes the component-expanded regularization vector by
    # dipole id, matching the C++ quirk also covered by ``GPMO_multi``.
    penalty_arr = (
        (reg_l2 * _component_mmax(m_maxima)[: m_maxima.shape[0]] ** 2)[:, None]
        if penalty is None
        else _as_runtime_array(penalty)
    )
    plus = plus + penalty_arr
    minus = minus + penalty_arr

    sentinel = _scalar_like(contributions_arr, _UNAVAILABLE_CANDIDATE_COST)
    plus = jnp.where(allowed, plus, sentinel)
    minus = jnp.where(allowed, minus, sentinel)
    return jnp.concatenate([jnp.ravel(plus), jnp.ravel(minus)])


def gpmo_arbvec_candidate_costs(
    spec: GPMOArbVecSpec,
    A_scaled: jax.Array,
    residual: jax.Array,
    available: jax.Array,
) -> jax.Array:
    """Return arbitrary-vector GPMO plus/minus candidate costs.

    Mirrors ``GPMO_ArbVec`` in
    ``src/simsoptpp/permanent_magnet_optimization.cpp:1168-1195``. Candidate
    order is dipole-major, polarization-vector-minor, with all plus candidates
    followed by all minus candidates.
    """

    contributions = _gpmo_arbvec_contributions(A_scaled, spec.pol_vectors)
    return _gpmo_arbvec_candidate_costs_for_allowed(
        spec,
        contributions,
        residual,
        available[:, None],
    )


def _gpmo_arbvec_candidate_costs_masked(
    spec: GPMOArbVecSpec,
    contributions: jax.Array,
    residual: jax.Array,
    available: jax.Array,
    vector_available: jax.Array,
    *,
    col_sq: jax.Array | None = None,
    penalty: jax.Array | None = None,
) -> jax.Array:
    allowed = available[:, None] & vector_available[None, :]
    return _gpmo_arbvec_candidate_costs_for_allowed(
        spec,
        contributions,
        residual,
        allowed,
        col_sq=col_sq,
        penalty=penalty,
    )


def gpmo_arbvec_step(
    spec: GPMOArbVecSpec,
    state: tuple[jax.Array, jax.Array, jax.Array],
    A_scaled: jax.Array,
    contributions: jax.Array | None = None,
    *,
    col_sq: jax.Array | None = None,
    penalty: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Run one normalized arbitrary-vector GPMO placement step."""

    x, residual, available = state
    if contributions is None:
        contributions = _gpmo_arbvec_contributions(A_scaled, spec.pol_vectors)
    costs = _gpmo_arbvec_candidate_costs_for_allowed(
        spec,
        contributions,
        residual,
        available[:, None],
        col_sq=col_sq,
        penalty=penalty,
    )
    n_candidates = spec.pol_vectors.shape[0] * spec.pol_vectors.shape[1]
    choice = _argmin_finite_cost(costs)
    is_minus = choice >= n_candidates
    candidate = jnp.where(is_minus, choice - n_candidates, choice)
    sign = jnp.where(is_minus, -1.0, 1.0).astype(residual.dtype)
    dipole = candidate // spec.pol_vectors.shape[1]
    vector_index = candidate % spec.pol_vectors.shape[1]
    selected_vector = spec.pol_vectors[dipole, vector_index, :]

    x_new = x.at[dipole, :].set(sign * selected_vector)
    residual_new = residual + sign * contributions[:, dipole, vector_index]
    available_new = available.at[dipole].set(False)
    residual_sq = jnp.sum(residual_new * residual_new)
    return (
        x_new,
        residual_new,
        available_new,
    ), (
        dipole,
        vector_index,
        sign,
        residual_sq,
    )


def _arbvec_x_history(
    selected_dipoles: jax.Array,
    selected_vector_indices: jax.Array,
    selected_signs: jax.Array,
    *,
    pol_vectors: jax.Array,
) -> jax.Array:
    """Reconstruct post-step normalized states from an ArbVec trace."""
    x0 = jnp.zeros((pol_vectors.shape[0], 3), dtype=pol_vectors.dtype)

    def _scan_body(x_state, trace_entry):
        dipole, vector_index, sign = trace_entry
        active = dipole >= 0
        dipole_index = jnp.maximum(dipole, 0)
        vector_index = jnp.maximum(vector_index, 0)
        selected_vector = pol_vectors[dipole_index, vector_index, :]
        x_candidate = x_state.at[dipole_index, :].set(sign * selected_vector)
        x_new = jnp.where(active, x_candidate, x_state)
        return x_new, x_new

    _, x_history = jax.lax.scan(
        _scan_body,
        x0,
        (selected_dipoles, selected_vector_indices, selected_signs),
    )
    return x_history


@partial(jax.jit, static_argnames=("K", "record_every"))
def gpmo_arbvec_solve(
    spec: GPMOArbVecSpec,
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    K: int,
    record_every: int | None = None,
) -> GPMOArbVecResult:
    """Run the arbitrary-vector greedy permanent-magnet optimizer in JAX."""

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    m_maxima = _as_runtime_array(spec.m_maxima)
    pol_vectors = _as_runtime_array(spec.pol_vectors)
    ndipoles = int(m_maxima.shape[0])
    _validate_gpmo_arbvec_static_args(K, ndipoles, pol_vectors)
    _validate_record_every(record_every)

    x0 = jnp.zeros((ndipoles, 3), dtype=A_arr.dtype)
    residual0 = -b_arr
    available0 = jnp.ones((ndipoles,), dtype=bool)
    if K == 0:
        empty_int = jnp.zeros((0,), dtype=jnp.int64)
        empty_float = jnp.zeros((0,), dtype=A_arr.dtype)
        empty_x_history = jnp.zeros((0, ndipoles, 3), dtype=A_arr.dtype)
        return GPMOArbVecResult(
            x=x0,
            x_history=empty_x_history,
            residual=residual0,
            residual_history=empty_float,
            selected_dipoles=empty_int,
            selected_vector_indices=empty_int,
            selected_signs=empty_float,
        )

    scan_spec = GPMOArbVecSpec(
        m_maxima=m_maxima,
        reg_l2=_as_runtime_array(spec.reg_l2),
        pol_vectors=pol_vectors,
    )
    contributions = _gpmo_arbvec_contributions(A_arr, pol_vectors)
    col_sq = jnp.sum(contributions * contributions, axis=0)
    penalty = (
        _as_runtime_array(spec.reg_l2)
        * _component_mmax(m_maxima)[: m_maxima.shape[0]] ** 2
    )[:, None]

    if record_every is not None:

        def _recording_step(state, _iteration):
            return gpmo_arbvec_step(
                scan_spec,
                state,
                A_arr,
                contributions,
                col_sq=col_sq,
                penalty=penalty,
            )

        final_state, x_history, residual_history, extra_histories = (
            _gpmo_recording_scan(
                _recording_step,
                (x0, residual0, available0),
                A_arr,
                K=K,
                record_every=record_every,
                ndipoles=ndipoles,
                extra_history_specs=(
                    ((), jnp.int64),
                    ((), jnp.int64),
                    ((), A_arr.dtype),
                ),
                extra_trace_indices=(0, 1, 2),
            )
        )
        selected_dipoles, selected_vector_indices, selected_signs = extra_histories
        return GPMOArbVecResult(
            x=final_state[0],
            x_history=x_history,
            residual=final_state[1],
            residual_history=residual_history,
            selected_dipoles=selected_dipoles,
            selected_vector_indices=selected_vector_indices,
            selected_signs=selected_signs,
        )

    def _scan_body(state, _):
        return gpmo_arbvec_step(
            scan_spec,
            state,
            A_arr,
            contributions,
            col_sq=col_sq,
            penalty=penalty,
        )

    final_state, trace = jax.lax.scan(
        _scan_body, (x0, residual0, available0), xs=None, length=K
    )
    selected_dipoles, selected_vector_indices, selected_signs, residual_history = trace
    return GPMOArbVecResult(
        x=final_state[0],
        x_history=_arbvec_x_history(
            selected_dipoles,
            selected_vector_indices,
            selected_signs,
            pol_vectors=pol_vectors,
        ),
        residual=final_state[1],
        residual_history=residual_history,
        selected_dipoles=selected_dipoles,
        selected_vector_indices=selected_vector_indices,
        selected_signs=selected_signs,
    )


def gpmo_arbvec_solve_bucketed(
    spec: GPMOArbVecSpec,
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    K: int,
    active_ndipoles: jax.Array,
    active_nvectors: jax.Array,
) -> GPMOArbVecResult:
    """Run arbitrary-vector GPMO with fixed bucket shapes and active counts.

    ``A_scaled``, ``m_maxima``, and ``pol_vectors`` are staged to bucket
    dimensions ``(M, 3*N_bucket)``, ``(N_bucket,)``, and
    ``(N_bucket, P_bucket, 3)``. ``active_ndipoles`` and ``active_nvectors``
    are scalar tensor inputs that choose the active prefix of the bucket at
    runtime. Inactive dipoles/vectors are masked before candidate selection,
    so changing the active counts does not change the compiled array shapes.
    """

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    m_maxima = _as_runtime_array(spec.m_maxima)
    pol_vectors = _as_runtime_array(spec.pol_vectors)
    ndipoles = int(m_maxima.shape[0])
    _validate_gpmo_arbvec_static_args(K, ndipoles, pol_vectors)
    _validate_gpmo_bucket_count("active_ndipoles", active_ndipoles, ndipoles)
    _validate_gpmo_bucket_count(
        "active_nvectors",
        active_nvectors,
        int(pol_vectors.shape[1]),
    )
    return _gpmo_arbvec_solve_bucketed_impl(
        spec,
        A_arr,
        b_arr,
        K=K,
        active_ndipoles=active_ndipoles,
        active_nvectors=active_nvectors,
    )


@partial(jax.jit, static_argnames=("K",))
def _gpmo_arbvec_solve_bucketed_impl(
    spec: GPMOArbVecSpec,
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    K: int,
    active_ndipoles: jax.Array,
    active_nvectors: jax.Array,
) -> GPMOArbVecResult:
    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    m_maxima = _as_runtime_array(spec.m_maxima)
    pol_vectors = _as_runtime_array(spec.pol_vectors)
    ndipoles = int(m_maxima.shape[0])

    active_ndipoles_arr = jnp.asarray(active_ndipoles, dtype=jnp.int64)
    active_nvectors_arr = jnp.asarray(active_nvectors, dtype=jnp.int64)
    active_dipoles = jnp.arange(ndipoles, dtype=jnp.int64) < active_ndipoles_arr
    active_vectors = (
        jnp.arange(pol_vectors.shape[1], dtype=jnp.int64) < active_nvectors_arr
    )

    x0 = jnp.zeros((ndipoles, 3), dtype=A_arr.dtype)
    residual0 = -b_arr
    available0 = active_dipoles
    done0 = (active_ndipoles_arr == 0) | (active_nvectors_arr == 0)
    if K == 0:
        empty_int = jnp.zeros((0,), dtype=jnp.int64)
        empty_float = jnp.zeros((0,), dtype=A_arr.dtype)
        empty_x_history = jnp.zeros((0, ndipoles, 3), dtype=A_arr.dtype)
        return GPMOArbVecResult(
            x=x0,
            x_history=empty_x_history,
            residual=residual0,
            residual_history=empty_float,
            selected_dipoles=empty_int,
            selected_vector_indices=empty_int,
            selected_signs=empty_float,
        )

    scan_spec = GPMOArbVecSpec(
        m_maxima=m_maxima,
        reg_l2=_as_runtime_array(spec.reg_l2),
        pol_vectors=pol_vectors,
    )
    contributions = _gpmo_arbvec_contributions(A_arr, pol_vectors)
    col_sq = jnp.sum(contributions * contributions, axis=0)
    penalty = (
        _as_runtime_array(spec.reg_l2)
        * _component_mmax(m_maxima)[: m_maxima.shape[0]] ** 2
    )[:, None]

    def _scan_body(state, _):
        x, residual, available, done = state
        costs = _gpmo_arbvec_candidate_costs_masked(
            scan_spec,
            contributions,
            residual,
            available,
            active_vectors,
            col_sq=col_sq,
            penalty=penalty,
        )
        n_candidates = pol_vectors.shape[0] * pol_vectors.shape[1]
        choice = _argmin_finite_cost(costs)
        is_minus = choice >= n_candidates
        candidate = jnp.where(is_minus, choice - n_candidates, choice)
        sign = jnp.where(is_minus, -1.0, 1.0).astype(residual.dtype)
        dipole = candidate // pol_vectors.shape[1]
        vector_index = candidate % pol_vectors.shape[1]
        selected_vector = pol_vectors[dipole, vector_index, :]

        x_placed = x.at[dipole, :].set(sign * selected_vector)
        residual_placed = residual + sign * contributions[:, dipole, vector_index]
        available_placed = available.at[dipole].set(False)
        placed_count = jnp.sum((active_dipoles & (~available_placed)).astype(jnp.int64))
        done_new = done | (placed_count >= active_ndipoles_arr)
        x_new = jnp.where(done, x, x_placed)
        residual_new = jnp.where(done, residual, residual_placed)
        available_new = jnp.where(done, available, available_placed)
        residual_sq = jnp.sum(residual_new * residual_new)
        trace = (
            jnp.where(done, _device_scalar(-1, jnp.int64), dipole),
            jnp.where(done, _device_scalar(-1, jnp.int64), vector_index),
            jnp.where(done, _scalar_like(residual, 0.0), sign),
            residual_sq,
        )
        return (x_new, residual_new, available_new, done_new), trace

    final_state, trace = jax.lax.scan(
        _scan_body, (x0, residual0, available0, done0), xs=None, length=K
    )
    (
        selected_dipoles,
        selected_vector_indices,
        selected_signs,
        residual_history,
    ) = trace
    return GPMOArbVecResult(
        x=final_state[0],
        x_history=_arbvec_x_history(
            selected_dipoles,
            selected_vector_indices,
            selected_signs,
            pol_vectors=pol_vectors,
        ),
        residual=final_state[1],
        residual_history=residual_history,
        selected_dipoles=selected_dipoles,
        selected_vector_indices=selected_vector_indices,
        selected_signs=selected_signs,
    )


# ── Arbitrary-vector backtracking GPMO (item 25 deferred sub-item) ─────


def initialize_gpmo_arbvec(
    x_init: jax.Array,
    pol_vectors: jax.Array,
    A_scaled: jax.Array,
    b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Initialize the arbitrary-vector backtracking solver state.

    Mirrors ``initialize_GPMO_ArbVec`` in
    ``permanent_magnet_optimization.cpp:994-1117``. For each dipole ``j``
    whose ``x_init[j]`` is nonzero, find the nearest allowable polarization
    vector ``pol_vectors[j, m_min]`` and sign ``sign_min in {-1, 0, +1}``.
    Sign ``0`` is selected when the squared distance to the zero vector is
    smaller than to every signed polarization candidate, in which case the
    dipole remains available and ``x`` stays zero. Otherwise ``x[j]`` is set
    to ``sign_min * pol_vectors[j, m_min]`` and the running residual
    ``residual = A_scaled @ x_flat - b`` is updated accordingly.

    Parameters
    ----------
    x_init
        Per-dipole initial guess, shape ``(N, 3)``. Should be expressed in
        the same coordinates as ``pol_vectors`` (each row is a non-negative
        L2-bounded unit moment in normalized coordinates). Pass zeros when no
        initial guess is desired.
    pol_vectors
        Allowable polarization vectors, shape ``(N, n_vectors, 3)``.
    A_scaled
        Forward matrix, shape ``(M, 3 N)``, already scaled by
        ``repeat(m_maxima, 3)``.
    b
        Target field, shape ``(M,)``.

    Returns
    -------
    x
        Initial dipole moment matrix, shape ``(N, 3)``.
    residual
        Running residual ``A_scaled @ x_flat - b``, shape ``(M,)``.
    available
        Boolean mask of unplaced dipoles, shape ``(N,)``.
    current_vector_indices
        Per-dipole polarization-vector index for placed dipoles, shape
        ``(N,)``. Unplaced dipoles carry ``0`` (a placeholder consistent with
        the C++ ``x_vec[j] = 0`` default).
    current_signs
        Per-dipole sign factor for placed dipoles, shape ``(N,)``. Unplaced
        dipoles carry ``0.0``.
    num_nonzero
        Scalar count of placed dipoles.
    """

    x_init_arr = _as_runtime_array(x_init)
    pol_vectors_arr = _as_runtime_array(pol_vectors)
    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    ndipoles = pol_vectors_arr.shape[0]
    if x_init_arr.shape != (ndipoles, 3):
        raise ValueError(
            "x_init must have shape (ndipoles, 3); "
            f"got {tuple(x_init_arr.shape)} for ndipoles={ndipoles}."
        )

    # Per-dipole squared distance to each (sign, pol_vector) candidate.
    # ``diff_pos[j, m] = ||x_init[j] - pol_vectors[j, m]||^2``
    # ``diff_neg[j, m] = ||x_init[j] + pol_vectors[j, m]||^2``
    diff_pos = jnp.sum(
        (x_init_arr[:, None, :] - pol_vectors_arr) ** 2, axis=2
    )  # (N, n_vectors)
    diff_neg = jnp.sum(
        (x_init_arr[:, None, :] + pol_vectors_arr) ** 2, axis=2
    )  # (N, n_vectors)
    # Squared distance to the zero vector candidate (per dipole).
    diff_null = jnp.sum(x_init_arr * x_init_arr, axis=1)  # (N,)

    # The C++ tie order picks the FIRST minimum encountered in the scan:
    # plus-then-minus per ``m``, then fall through to the null candidate.
    # ``argmin`` over a stable per-row sequence reproduces that order.
    n_vectors = pol_vectors_arr.shape[1]
    candidates = jnp.concatenate(
        [diff_pos, diff_neg, diff_null[:, None]], axis=1
    )  # (N, 2*n_vectors + 1)
    choice = _argmin_finite_cost(candidates, axis=1)  # (N,)
    null_choice = jax.device_put(
        np.asarray(2 * n_vectors, dtype=np.dtype(choice.dtype))
    )
    vector_count = jax.device_put(np.asarray(n_vectors, dtype=np.dtype(choice.dtype)))
    chose_null = choice == null_choice
    chose_minus = (choice >= vector_count) & (~chose_null)
    zero_choice = choice - choice
    vector_idx = jnp.where(
        chose_null,
        zero_choice,
        jnp.where(chose_minus, choice - vector_count, choice),
    )
    zero = _scalar_like(A_arr, 0.0)
    sign = jnp.where(
        chose_null,
        zero,
        jnp.where(
            chose_minus,
            _scalar_like(A_arr, -1.0),
            _scalar_like(A_arr, 1.0),
        ),
    )

    # Only update dipoles whose ``x_init`` is nonzero (matches the C++
    # ``if (x_init(j,0) == 0 && ... ) continue`` guard).
    nonzero_init = jnp.any(x_init_arr != zero, axis=1)  # (N,)
    placed = nonzero_init & (~chose_null)
    sign = jnp.where(placed, sign, zero)
    selected_pol_vec = jnp.take_along_axis(
        pol_vectors_arr, vector_idx[:, None, None], axis=1
    )[:, 0, :]  # (N, 3)
    x = jnp.where(placed[:, None], sign[:, None] * selected_pol_vec, zero)

    # Running residual: A_scaled @ x.flatten() - b.
    residual = A_arr @ jnp.reshape(x, (ndipoles * 3,)) - b_arr
    available = ~placed
    current_vector_indices = jnp.where(placed, vector_idx, vector_idx - vector_idx)
    current_signs = sign
    num_nonzero = jnp.sum(placed.astype(jnp.int64))
    return (
        x,
        residual,
        available,
        current_vector_indices,
        current_signs,
        num_nonzero,
    )


def _gpmo_arbvec_backtracking_candidate_costs(
    A_scaled: jax.Array,
    residual: jax.Array,
    available: jax.Array,
    pol_vectors: jax.Array,
    m_maxima: jax.Array,
    reg_l2: jax.Array,
) -> jax.Array:
    """Candidate costs for the arbitrary-vector backtracking placement step.

    Matches the C++ formula
    ``R2s_ptr[mj] = sum_i (Aij_mj[i] + bnorm)^2 + mmax[j]^2``
    where ``bnorm = sum_l pol_vec[l, m, j] * A[i, l + 3*j]`` and ``mmax``
    is the kernel-level array ``sqrt(reg_l2) * repeat(m_maxima, 3)``. The
    penalty therefore expands to ``reg_l2 * repeat(m_maxima, 3)[j]^2`` for
    dipole ``j ∈ [0, N)``, indexing the first ``N`` entries of the C++
    component-expanded vector. Same index quirk as
    ``gpmo_arbvec_candidate_costs``. See
    ``permanent_magnet_optimization.cpp:821-822``.
    """

    contributions = _gpmo_arbvec_contributions(A_scaled, pol_vectors)
    return _gpmo_arbvec_backtracking_candidate_costs_from_contributions(
        contributions,
        residual,
        available,
        m_maxima,
        reg_l2,
    )


def _gpmo_arbvec_backtracking_candidate_costs_from_contributions(
    contributions: jax.Array,
    residual: jax.Array,
    available: jax.Array,
    m_maxima: jax.Array,
    reg_l2: jax.Array,
    *,
    col_sq: jax.Array | None = None,
    penalty: jax.Array | None = None,
) -> jax.Array:
    contributions_arr = _as_runtime_array(contributions)
    residual_arr = _as_runtime_array(residual)
    residual_sq = jnp.sum(residual_arr * residual_arr)
    dot = jnp.einsum("m,mnp->np", residual_arr, contributions_arr)
    col_sq_arr = (
        jnp.sum(contributions_arr * contributions_arr, axis=0)
        if col_sq is None
        else _as_runtime_array(col_sq)
    )
    plus = residual_sq + 2.0 * dot + col_sq_arr
    minus = residual_sq - 2.0 * dot + col_sq_arr

    component_mmax = _component_mmax(m_maxima)[: m_maxima.shape[0]]
    penalty_arr = (
        (reg_l2 * component_mmax * component_mmax)[:, None]
        if penalty is None
        else _as_runtime_array(penalty)
    )
    plus = plus + penalty_arr
    minus = minus + penalty_arr

    allowed = available[:, None]
    sentinel = _scalar_like(contributions_arr, _UNAVAILABLE_CANDIDATE_COST)
    plus = jnp.where(allowed, plus, sentinel)
    minus = jnp.where(allowed, minus, sentinel)
    return jnp.concatenate([jnp.ravel(plus), jnp.ravel(minus)])


def _gpmo_arbvec_remove_one_pair(
    state: tuple[jax.Array, ...],
    A_scaled: jax.Array,
    pol_vectors: jax.Array,
    jk: jax.Array,
    cj: jax.Array,
) -> tuple[jax.Array, ...]:
    (
        x,
        residual,
        available,
        current_vector_indices,
        current_signs,
        removed_for_seed,
        removed_count,
    ) = state
    sign_j = current_signs[jk]
    sign_c = current_signs[cj]
    vector_j = pol_vectors[jk, current_vector_indices[jk], :]
    vector_c = pol_vectors[cj, current_vector_indices[cj], :]
    # Strip the pair's contribution to the running residual:
    # ``A_scaled @ delta_x_flat`` where ``delta_x_flat[3*jk + l] = -sign_j * vector_j[l]``
    # and similarly for ``cj``. Equivalently subtract ``sign * pol_vec``
    # times the corresponding columns of ``A_scaled``. Dynamic-shape-stable
    # gather via ``lax.dynamic_slice`` because ``jk`` / ``cj`` are tracers.
    jk_cols = jax.lax.dynamic_slice(A_scaled, (0, 3 * jk), (A_scaled.shape[0], 3))
    cj_cols = jax.lax.dynamic_slice(A_scaled, (0, 3 * cj), (A_scaled.shape[0], 3))
    residual_new = (
        residual - sign_j * (jk_cols @ vector_j) - sign_c * (cj_cols @ vector_c)
    )
    return (
        x.at[jk]
        .set(jnp.zeros(3, dtype=x.dtype))
        .at[cj]
        .set(jnp.zeros(3, dtype=x.dtype)),
        residual_new,
        available.at[jk].set(True).at[cj].set(True),
        current_vector_indices.at[jk].set(0).at[cj].set(0),
        current_signs.at[jk].set(0.0).at[cj].set(0.0),
        _bool_scalar(True),
        removed_count + _device_scalar(1, removed_count.dtype),
    )


def _gpmo_arbvec_remove_pairs(
    x: jax.Array,
    residual: jax.Array,
    available: jax.Array,
    current_vector_indices: jax.Array,
    current_signs: jax.Array,
    connectivity: jax.Array,
    A_scaled: jax.Array,
    pol_vectors: jax.Array,
    cos_thresh_angle: jax.Array,
    *,
    ndipoles: int,
    Nadjacent: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the C++ dewyrming pass for the arbitrary-vector variant.

    Mirrors ``permanent_magnet_optimization.cpp:861-939`` exactly: the outer
    loop iterates over **dipole indices** ``j ∈ [0, N)`` (NOT over placement
    history). For each ``j`` that is currently placed (``current_signs[j] !=
    0``), search its ``Nadjacent`` nearest dipoles for the one with the
    smallest ``cos_angle`` (most anti-aligned). If that smallest cosine is
    ``<= cos_thresh_angle``, remove the pair: revert both placements and add
    both dipoles back to the available set. Cascaded removals are allowed
    within a single dewyrming pass because the state propagates between
    seed iterations.
    """

    removed_count0 = _device_scalar(0, jnp.int64)
    sentinel_cos = _scalar_like(A_scaled, 2.0)

    def _seed_body_clean(seed_state, jk):
        (
            x_seed,
            residual_seed,
            available_seed,
            vector_idx_seed,
            signs_seed,
            removed_count_seed,
        ) = seed_state
        sign_j = signs_seed[jk]
        seed_active = sign_j != 0.0
        vector_j_idx = vector_idx_seed[jk]
        moment_j = sign_j * pol_vectors[jk, vector_j_idx, :]

        def _neighbor_body_clean(neighbor_state, neighbor_index):
            min_cos_angle, cj_min_idx = neighbor_state
            cj = connectivity[jk, neighbor_index]
            sign_c = signs_seed[cj]
            vector_c_idx = vector_idx_seed[cj]
            moment_c = sign_c * pol_vectors[cj, vector_c_idx, :]
            cos_angle = jnp.sum(moment_j * moment_c)
            # ``if Gamma_complement(cj) continue`` -> skip when ``sign_c == 0``.
            neighbor_placed = sign_c != 0.0
            candidate_cos = jnp.where(neighbor_placed, cos_angle, sentinel_cos)
            update = candidate_cos < min_cos_angle
            return (
                jnp.where(update, candidate_cos, min_cos_angle),
                jnp.where(update, cj, cj_min_idx),
            ), None

        (min_cos_angle, cj_min), _ = jax.lax.scan(
            _neighbor_body_clean,
            (sentinel_cos, _device_scalar(0, connectivity.dtype)),
            jnp.arange(Nadjacent),
        )

        should_remove = seed_active & (min_cos_angle <= cos_thresh_angle)
        updated_state = jax.lax.cond(
            should_remove,
            lambda args: _gpmo_arbvec_remove_one_pair(*args),
            lambda args: args[0],
            (
                (
                    x_seed,
                    residual_seed,
                    available_seed,
                    vector_idx_seed,
                    signs_seed,
                    _bool_scalar(False),
                    removed_count_seed,
                ),
                A_scaled,
                pol_vectors,
                jk,
                cj_min,
            ),
        )
        # Drop the ``removed_for_seed`` scratch field before returning.
        return (
            updated_state[0],
            updated_state[1],
            updated_state[2],
            updated_state[3],
            updated_state[4],
            updated_state[6],
        ), None

    final_state, _ = jax.lax.scan(
        _seed_body_clean,
        (
            x,
            residual,
            available,
            current_vector_indices,
            current_signs,
            removed_count0,
        ),
        jnp.arange(ndipoles),
    )
    return final_state


def gpmo_arbvec_backtracking_step(
    spec: GPMOArbVecBacktrackingSpec,
    state: tuple[jax.Array, ...],
    A_scaled: jax.Array,
    connectivity: jax.Array,
    cos_thresh_angle: jax.Array,
    iteration: jax.Array,
    contributions: jax.Array | None = None,
    *,
    col_sq: jax.Array | None = None,
    penalty: jax.Array | None = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Run one arbitrary-vector backtracking GPMO placement step.

    The step is composed of:
      1. Pick the candidate ``(j, m, sign)`` that most reduces ``||residual + bnorm||^2``.
      2. Place it: update ``x``, residual, available, sign-and-vector traces.
      3. If ``iteration >= backtracking`` and ``iteration % backtracking == 0``,
         run the dewyrming pass: for each placed dipole ``j``, find the most
         anti-aligned placed adjacent dipole and remove the pair if their
         cosine angle is below ``cos_thresh_angle``.
      4. Carry-forward semantics for terminated runs (``num_nonzero >=
         ndipoles`` or ``num_nonzero >= max_nMagnets``).
    """

    (
        x,
        residual,
        available,
        current_vector_indices,
        current_signs,
        selected_dipoles,
        selected_vector_indices,
        selected_signs,
        done,
    ) = state

    if contributions is None:
        contributions = _gpmo_arbvec_contributions(A_scaled, spec.pol_vectors)
    costs = _gpmo_arbvec_backtracking_candidate_costs_from_contributions(
        contributions,
        residual,
        available,
        spec.m_maxima,
        spec.reg_l2,
        col_sq=col_sq,
        penalty=penalty,
    )
    n_candidates = spec.pol_vectors.shape[0] * spec.pol_vectors.shape[1]
    choice = _argmin_finite_cost(costs)
    is_minus = choice >= n_candidates
    candidate = jnp.where(is_minus, choice - n_candidates, choice)
    sign = jnp.where(is_minus, -1.0, 1.0).astype(residual.dtype)
    dipole = candidate // spec.pol_vectors.shape[1]
    vector_index = candidate % spec.pol_vectors.shape[1]
    selected_vector = spec.pol_vectors[dipole, vector_index, :]

    x_placed = x.at[dipole].set(sign * selected_vector)
    residual_placed = residual + sign * contributions[:, dipole, vector_index]
    available_placed = available.at[dipole].set(False)
    current_signs_placed = current_signs.at[dipole].set(sign)
    current_vector_indices_placed = current_vector_indices.at[dipole].set(vector_index)
    selected_dipoles_placed = selected_dipoles.at[iteration].set(dipole)
    selected_vector_indices_placed = selected_vector_indices.at[iteration].set(
        vector_index
    )
    selected_signs_placed = selected_signs.at[iteration].set(sign)

    # The C++ ``GPMO_ArbVec_backtracking`` gate is exactly
    # ``(k % backtracking) == 0`` (see ``permanent_magnet_optimization.cpp:861``)
    # — the very first iteration ``k=0`` triggers a dewyrming pass after the
    # initial placement. This differs from baseline ``GPMO_backtracking``
    # which adds the ``k >= backtracking`` guard at line 486.
    backtrack_due = (iteration % spec.backtracking) == 0
    ndipoles = x.shape[0]
    (
        x_backtracked,
        residual_backtracked,
        available_backtracked,
        current_vector_indices_backtracked,
        current_signs_backtracked,
        removed_count,
    ) = jax.lax.cond(
        backtrack_due,
        lambda args: _gpmo_arbvec_remove_pairs(
            *args, ndipoles=ndipoles, Nadjacent=spec.Nadjacent
        ),
        lambda args: (
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            _device_scalar(0, jnp.int64),
        ),
        (
            x_placed,
            residual_placed,
            available_placed,
            current_vector_indices_placed,
            current_signs_placed,
            connectivity,
            A_scaled,
            spec.pol_vectors,
            cos_thresh_angle,
        ),
    )
    num_nonzeros = jnp.sum((~available_backtracked).astype(jnp.int64))
    stop = (num_nonzeros >= x.shape[0]) | (num_nonzeros >= spec.max_nMagnets)
    done_new = done | stop
    residual_sq = jnp.sum(residual_backtracked * residual_backtracked)

    next_state = (
        jnp.where(done, x, x_backtracked),
        jnp.where(done, residual, residual_backtracked),
        jnp.where(done, available, available_backtracked),
        jnp.where(
            done,
            current_vector_indices,
            current_vector_indices_backtracked,
        ),
        jnp.where(done, current_signs, current_signs_backtracked),
        jnp.where(done, selected_dipoles, selected_dipoles_placed),
        jnp.where(done, selected_vector_indices, selected_vector_indices_placed),
        jnp.where(done, selected_signs, selected_signs_placed),
        done_new,
    )
    trace = (
        jnp.where(done, _device_scalar(-1, jnp.int64), dipole),
        jnp.where(done, _device_scalar(-1, jnp.int64), vector_index),
        jnp.where(done, _scalar_like(residual, 0.0), sign),
        jnp.where(done, jnp.sum(residual * residual), residual_sq),
        next_state[0],
        jnp.where(
            done,
            jnp.sum((~available).astype(jnp.int64)),
            num_nonzeros,
        ),
        jnp.where(done, _device_scalar(0, jnp.int64), removed_count),
        done_new,
    )
    return next_state, trace


@partial(jax.jit, static_argnames=("K", "record_every"))
def gpmo_arbvec_backtracking_solve(
    spec: GPMOArbVecBacktrackingSpec,
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    K: int,
    x_init: jax.Array | None = None,
    record_every: int | None = None,
) -> GPMOArbVecBacktrackingResult:
    """Run the arbitrary-vector backtracking greedy permanent-magnet optimizer.

    Mirrors ``GPMO_ArbVec_backtracking`` in
    ``permanent_magnet_optimization.cpp:729-987``. The scan length is fixed
    by ``K``. Once the CPU stopping condition
    ``num_nonzero >= ndipoles`` or ``num_nonzero >= max_nMagnets`` is reached,
    later scan iterations carry the final state unchanged. The ``x_init``
    input (optional) is folded into the initial state via
    ``initialize_gpmo_arbvec``; passing ``None`` matches the upstream default
    ``x_init = zeros((N, 3))``.
    """

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    m_maxima = _as_runtime_array(spec.m_maxima)
    pol_vectors = _as_runtime_array(spec.pol_vectors)
    ndipoles = int(m_maxima.shape[0])
    _validate_gpmo_arbvec_backtracking_static_args(
        K,
        ndipoles,
        pol_vectors,
        spec.Nadjacent,
        spec.backtracking,
        spec.max_nMagnets,
        spec.thresh_angle,
    )
    _validate_record_every(record_every)

    if x_init is None:
        x_init_arr = _zeros_like_shape(A_arr, (ndipoles, 3))
    else:
        x_init_arr = _as_runtime_array(x_init)
        if x_init_arr.shape != (ndipoles, 3):
            raise ValueError(
                "x_init must have shape (ndipoles, 3); "
                f"got {tuple(x_init_arr.shape)} for ndipoles={ndipoles}."
            )

    (
        x0,
        residual0,
        available0,
        current_vector_indices0,
        current_signs0,
        initial_num_nonzero,
    ) = initialize_gpmo_arbvec(x_init_arr, pol_vectors, A_arr, b_arr)

    selected_dipoles0 = _negative_ones_int_like_shape(current_vector_indices0, (K,))
    selected_vector_indices0 = _negative_ones_int_like_shape(
        current_vector_indices0, (K,)
    )
    selected_signs0 = _zeros_like_shape(A_arr, (K,))
    if K == 0:
        empty_int = _zeros_like_shape(current_vector_indices0, (0,))
        empty_float = _zeros_like_shape(A_arr, (0,))
        empty_x_history = _zeros_like_shape(A_arr, (0, ndipoles, 3))
        empty_bool = _false_like_shape(A_arr, (0,))
        return GPMOArbVecBacktrackingResult(
            x=x0,
            x_history=empty_x_history,
            residual=residual0,
            residual_history=empty_float,
            selected_dipoles=empty_int,
            selected_vector_indices=empty_int,
            selected_signs=empty_float,
            num_nonzeros_history=empty_int,
            removed_pair_count_history=empty_int,
            done_history=empty_bool,
            initial_x=x0,
            initial_residual=residual0,
            initial_num_nonzero=initial_num_nonzero,
        )

    connectivity = gpmo_connectivity_matrix(spec.dipole_grid_xyz)
    scan_spec = GPMOArbVecBacktrackingSpec(
        m_maxima=m_maxima,
        reg_l2=_as_runtime_array(spec.reg_l2),
        dipole_grid_xyz=_as_runtime_array(spec.dipole_grid_xyz),
        pol_vectors=pol_vectors,
        Nadjacent=spec.Nadjacent,
        backtracking=spec.backtracking,
        thresh_angle=spec.thresh_angle,
        max_nMagnets=spec.max_nMagnets,
    )
    cos_thresh_angle = jnp.cos(_scalar_like(A_arr, spec.thresh_angle))
    contributions = _gpmo_arbvec_contributions(A_arr, pol_vectors)
    col_sq = jnp.sum(contributions * contributions, axis=0)
    # Left-associated ``(reg_l2*cm)*cm`` to match the backtracking candidate-cost
    # fallback (and the pre-hoist in-function form); ``reg_l2*cm**2`` would
    # reassociate the product and diverge at the ULP level from HEAD.
    component_mmax = _component_mmax(m_maxima)[: m_maxima.shape[0]]
    penalty = (_as_runtime_array(spec.reg_l2) * component_mmax * component_mmax)[
        :, None
    ]

    # Stopping at the initial-state check: if initialization already
    # satisfies the C++ stop predicate, propagate a final-state carry.
    initial_stop = (
        initial_num_nonzero >= _int_scalar_like(initial_num_nonzero, ndipoles)
    ) | (
        initial_num_nonzero >= _int_scalar_like(initial_num_nonzero, spec.max_nMagnets)
    )

    if record_every is not None:

        def _recording_step(state, iteration):
            return gpmo_arbvec_backtracking_step(
                scan_spec,
                state,
                A_arr,
                connectivity,
                cos_thresh_angle,
                iteration,
                contributions,
                col_sq=col_sq,
                penalty=penalty,
            )

        final_state, x_history, residual_history, extra_histories = (
            _gpmo_recording_scan(
                _recording_step,
                (
                    x0,
                    residual0,
                    available0,
                    current_vector_indices0,
                    current_signs0,
                    selected_dipoles0,
                    selected_vector_indices0,
                    selected_signs0,
                    initial_stop,
                ),
                A_arr,
                K=K,
                record_every=record_every,
                ndipoles=ndipoles,
                extra_history_specs=(
                    ((), jnp.int64),
                    ((), jnp.int64),
                    ((), A_arr.dtype),
                    ((), jnp.int64),
                    ((), jnp.int64),
                    ((), bool),
                ),
                extra_trace_indices=(0, 1, 2, 5, 6, 7),
            )
        )
        (
            selected_dipoles,
            selected_vector_indices,
            selected_signs,
            num_nonzeros_history,
            removed_pair_count_history,
            done_history,
        ) = extra_histories
        return GPMOArbVecBacktrackingResult(
            x=final_state[0],
            x_history=x_history,
            residual=final_state[1],
            residual_history=residual_history,
            selected_dipoles=selected_dipoles,
            selected_vector_indices=selected_vector_indices,
            selected_signs=selected_signs,
            num_nonzeros_history=num_nonzeros_history,
            removed_pair_count_history=removed_pair_count_history,
            done_history=done_history,
            initial_x=x0,
            initial_residual=residual0,
            initial_num_nonzero=initial_num_nonzero,
        )

    def _scan_body(state, iteration):
        return gpmo_arbvec_backtracking_step(
            scan_spec,
            state,
            A_arr,
            connectivity,
            cos_thresh_angle,
            iteration,
            contributions,
            col_sq=col_sq,
            penalty=penalty,
        )

    final_state, trace = jax.lax.scan(
        _scan_body,
        (
            x0,
            residual0,
            available0,
            current_vector_indices0,
            current_signs0,
            selected_dipoles0,
            selected_vector_indices0,
            selected_signs0,
            initial_stop,
        ),
        jnp.arange(K),
    )
    (
        selected_dipoles,
        selected_vector_indices,
        selected_signs,
        residual_history,
        x_history,
        num_nonzeros_history,
        removed_pair_count_history,
        done_history,
    ) = trace
    return GPMOArbVecBacktrackingResult(
        x=final_state[0],
        x_history=x_history,
        residual=final_state[1],
        residual_history=residual_history,
        selected_dipoles=selected_dipoles,
        selected_vector_indices=selected_vector_indices,
        selected_signs=selected_signs,
        num_nonzeros_history=num_nonzeros_history,
        removed_pair_count_history=removed_pair_count_history,
        done_history=done_history,
        initial_x=x0,
        initial_residual=residual0,
        initial_num_nonzero=initial_num_nonzero,
    )


def _gpmo_multi_selected_mask(
    connectivity: jax.Array,
    available: jax.Array,
    seed_dipole: jax.Array,
    component: jax.Array,
    Nadjacent: int,
) -> jax.Array:
    ordered_dipoles = connectivity[seed_dipole]
    ordered_available = available[ordered_dipoles, component]
    available_rank = jnp.cumsum(ordered_available.astype(jnp.int64))
    ordered_selected = ordered_available & (available_rank <= Nadjacent)
    return (
        jnp.zeros_like(available[:, 0], dtype=bool)
        .at[ordered_dipoles]
        .set(ordered_selected)
    )


def _gpmo_multi_selected_group(
    connectivity: jax.Array,
    available: jax.Array,
    seed_dipole: jax.Array,
    component: jax.Array,
    Nadjacent: int,
) -> jax.Array:
    ordered_dipoles = connectivity[seed_dipole]
    ordered_available = available[ordered_dipoles, component]
    available_rank = jnp.cumsum(ordered_available.astype(jnp.int64))
    ordered_selected = ordered_available & (available_rank <= Nadjacent)
    selected_order = jnp.nonzero(ordered_selected, size=Nadjacent, fill_value=0)[0]
    return ordered_dipoles[selected_order]


def gpmo_multi_candidate_costs(
    spec: GPMOMultiSpec,
    A_scaled: jax.Array,
    residual: jax.Array,
    available: jax.Array,
    connectivity: jax.Array,
    *,
    col_sq: jax.Array | None = None,
    regularizer_sq: jax.Array | None = None,
) -> jax.Array:
    """Return multi-neighbour GPMO plus/minus candidate costs.

    Mirrors ``GPMO_multi`` in
    ``src/simsoptpp/permanent_magnet_optimization.cpp:630-667``. The candidate
    vector has shape ``(6 N,)`` with plus candidates followed by minus
    candidates, preserving the C++ ``std::min_element`` tie order.
    """

    A_arr = _as_runtime_array(A_scaled)
    residual_arr = _as_runtime_array(residual)
    m_maxima = _as_runtime_array(spec.m_maxima)
    reg_l2 = _as_runtime_array(spec.reg_l2)

    n_components = A_arr.shape[1]
    component_indices = jnp.arange(n_components)
    seed_dipoles = component_indices // 3
    components = component_indices % 3
    ordered_dipoles = connectivity[seed_dipoles]
    ordered_available = available[ordered_dipoles, components[:, None]]
    available_rank = jnp.cumsum(ordered_available.astype(jnp.int64), axis=1)
    ordered_selected = ordered_available & (available_rank <= spec.Nadjacent)
    has_enough = jnp.sum(ordered_selected, axis=1) == spec.Nadjacent

    selected_component_indices = 3 * ordered_dipoles + components[:, None]
    residual_sq = jnp.sum(residual_arr * residual_arr)
    dot = A_arr.T @ residual_arr
    col_sq_arr = (
        jnp.sum(A_arr * A_arr, axis=0)
        if col_sq is None
        else _as_runtime_array(col_sq)
    )
    selected_dot = dot[selected_component_indices]
    selected_col_sq = col_sq_arr[selected_component_indices]
    plus_per_neighbor = residual_sq + 2.0 * selected_dot + selected_col_sq
    minus_per_neighbor = residual_sq - 2.0 * selected_dot + selected_col_sq

    selected_float = ordered_selected.astype(A_arr.dtype)
    plus = jnp.sum(plus_per_neighbor * selected_float, axis=1)
    minus = jnp.sum(minus_per_neighbor * selected_float, axis=1)

    # ``GPMO_multi`` indexes the regularization vector by dipole id in the C++
    # loop. The vector passed from Python is component-expanded, so this uses
    # the same first-N entries rather than the baseline component index.
    regularizer_sq_arr = (
        _component_mmax(m_maxima)[ordered_dipoles] ** 2
        if regularizer_sq is None
        else _as_runtime_array(regularizer_sq)
    )
    # Keep ``reg_l2`` OUTSIDE the reduction to stay bit-identical to the
    # pre-hoist form ``reg_l2 * sum(reg*reg*sel)`` -- distributing it inside
    # (``sum(reg_l2*reg*reg*sel)``) reassociates the FP sum (c*Sx != S(c*x)).
    penalty = reg_l2 * jnp.sum(regularizer_sq_arr * selected_float, axis=1)
    plus = plus + penalty
    minus = minus + penalty

    direction_mask = _single_direction_mask(n_components, spec.single_direction)
    allowed = jnp.reshape(available, (n_components,)) & direction_mask & has_enough
    sentinel = _scalar_like(A_arr, _UNAVAILABLE_CANDIDATE_COST)
    plus = jnp.where(allowed, plus, sentinel)
    minus = jnp.where(allowed, minus, sentinel)
    return jnp.concatenate([plus, minus])


def gpmo_multi_step(
    spec: GPMOMultiSpec,
    state: tuple[jax.Array, jax.Array, jax.Array],
    A_scaled: jax.Array,
    connectivity: jax.Array,
    *,
    col_sq: jax.Array | None = None,
    regularizer_sq: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Run one normalized multi-neighbour GPMO placement step."""

    x, residual, available = state
    costs = gpmo_multi_candidate_costs(
        spec,
        A_scaled,
        residual,
        available,
        connectivity,
        col_sq=col_sq,
        regularizer_sq=regularizer_sq,
    )
    n_components = A_scaled.shape[1]
    choice = _argmin_finite_cost(costs)
    is_minus = choice >= n_components
    component_index = jnp.where(is_minus, choice - n_components, choice)
    sign = jnp.where(is_minus, -1.0, 1.0).astype(residual.dtype)
    seed_dipole = component_index // 3
    component = component_index % 3
    selected_mask = _gpmo_multi_selected_mask(
        connectivity, available, seed_dipole, component, spec.Nadjacent
    )
    selected_group = _gpmo_multi_selected_group(
        connectivity, available, seed_dipole, component, spec.Nadjacent
    )
    selected_component_indices = 3 * selected_group + component

    x_new = jnp.where(
        selected_mask[:, None],
        jnp.zeros_like(x).at[:, component].set(sign),
        x,
    )
    residual_new = residual + sign * jnp.sum(
        A_scaled[:, selected_component_indices], axis=1
    )
    available_new = jnp.where(selected_mask[:, None], False, available)
    residual_sq = jnp.sum(residual_new * residual_new)
    return (
        x_new,
        residual_new,
        available_new,
    ), (
        seed_dipole,
        component,
        sign,
        residual_sq,
        selected_group,
    )


def _multi_x_history(
    selected_groups: jax.Array,
    selected_components: jax.Array,
    selected_signs: jax.Array,
    *,
    ndipoles: int,
) -> jax.Array:
    """Reconstruct post-step normalized states from a multi-neighbour trace."""
    x0 = jnp.zeros((ndipoles, 3), dtype=selected_signs.dtype)
    dipole_ids = jnp.arange(ndipoles)

    def _scan_body(x_state, trace_entry):
        selected_group, component, sign = trace_entry
        selected_mask = jnp.any(dipole_ids[:, None] == selected_group[None, :], axis=1)
        x_updates = jnp.zeros_like(x_state).at[:, component].set(sign)
        x_new = jnp.where(selected_mask[:, None], x_updates, x_state)
        return x_new, x_new

    _, x_history = jax.lax.scan(
        _scan_body, x0, (selected_groups, selected_components, selected_signs)
    )
    return x_history


@partial(jax.jit, static_argnames=("K", "record_every"))
def gpmo_multi_solve(
    spec: GPMOMultiSpec,
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    K: int,
    record_every: int | None = None,
) -> GPMOMultiResult:
    """Run the multi-neighbour greedy permanent-magnet optimizer in JAX."""

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    m_maxima = _as_runtime_array(spec.m_maxima)
    ndipoles = int(m_maxima.shape[0])
    _validate_gpmo_multi_static_args(K, spec.single_direction, ndipoles, spec.Nadjacent)
    _validate_record_every(record_every)

    x0 = jnp.zeros((ndipoles, 3), dtype=A_arr.dtype)
    residual0 = -b_arr
    available0 = jnp.ones((ndipoles, 3), dtype=bool)
    if K == 0:
        empty_int = jnp.zeros((0,), dtype=jnp.int64)
        empty_float = jnp.zeros((0,), dtype=A_arr.dtype)
        empty_x_history = jnp.zeros((0, ndipoles, 3), dtype=A_arr.dtype)
        empty_groups = jnp.zeros((0, spec.Nadjacent), dtype=jnp.int64)
        return GPMOMultiResult(
            x=x0,
            x_history=empty_x_history,
            residual=residual0,
            residual_history=empty_float,
            selected_seed_dipoles=empty_int,
            selected_components=empty_int,
            selected_signs=empty_float,
            selected_groups=empty_groups,
        )

    connectivity = gpmo_connectivity_matrix(spec.dipole_grid_xyz)
    n_components = A_arr.shape[1]
    component_indices = jnp.arange(n_components)
    seed_dipoles = component_indices // 3
    ordered_dipoles = connectivity[seed_dipoles]
    col_sq = jnp.sum(A_arr * A_arr, axis=0)
    # Squared regularizer only (no reg_l2): gpmo_multi_candidate_costs applies
    # reg_l2 outside the reduction to preserve the pre-hoist FP association.
    regularizer_sq = _component_mmax(m_maxima)[ordered_dipoles] ** 2

    if record_every is not None:

        def _recording_step(state, _iteration):
            return gpmo_multi_step(
                spec,
                state,
                A_arr,
                connectivity,
                col_sq=col_sq,
                regularizer_sq=regularizer_sq,
            )

        final_state, x_history, residual_history, extra_histories = (
            _gpmo_recording_scan(
                _recording_step,
                (x0, residual0, available0),
                A_arr,
                K=K,
                record_every=record_every,
                ndipoles=ndipoles,
                extra_history_specs=(
                    ((), jnp.int64),
                    ((), jnp.int64),
                    ((), A_arr.dtype),
                    ((spec.Nadjacent,), jnp.int64),
                ),
                extra_trace_indices=(0, 1, 2, 4),
            )
        )
        (
            selected_seed_dipoles,
            selected_components,
            selected_signs,
            selected_groups,
        ) = extra_histories
        return GPMOMultiResult(
            x=final_state[0],
            x_history=x_history,
            residual=final_state[1],
            residual_history=residual_history,
            selected_seed_dipoles=selected_seed_dipoles,
            selected_components=selected_components,
            selected_signs=selected_signs,
            selected_groups=selected_groups,
        )

    def _scan_body(state, _):
        return gpmo_multi_step(
            spec,
            state,
            A_arr,
            connectivity,
            col_sq=col_sq,
            regularizer_sq=regularizer_sq,
        )

    final_state, trace = jax.lax.scan(
        _scan_body, (x0, residual0, available0), xs=None, length=K
    )
    (
        selected_seed_dipoles,
        selected_components,
        selected_signs,
        residual_history,
        selected_groups,
    ) = trace
    return GPMOMultiResult(
        x=final_state[0],
        x_history=_multi_x_history(
            selected_groups,
            selected_components,
            selected_signs,
            ndipoles=ndipoles,
        ),
        residual=final_state[1],
        residual_history=residual_history,
        selected_seed_dipoles=selected_seed_dipoles,
        selected_components=selected_components,
        selected_signs=selected_signs,
        selected_groups=selected_groups,
    )


def _gpmo_backtracking_remove_one_pair(
    state: tuple[jax.Array, ...],
    A_scaled: jax.Array,
    jk: jax.Array,
    cj: jax.Array,
) -> tuple[jax.Array, ...]:
    (
        x,
        residual,
        available,
        current_signs,
        current_components,
        removed_for_seed,
        removed_count,
    ) = state
    comp_j = current_components[jk]
    comp_c = current_components[cj]
    sign_j = current_signs[jk]
    sign_c = current_signs[cj]
    component_j = 3 * jk + comp_j
    component_c = 3 * cj + comp_c
    residual_new = residual - sign_j * A_scaled[:, component_j]
    residual_new = residual_new - sign_c * A_scaled[:, component_c]
    return (
        x.at[jk, comp_j].set(0.0).at[cj, comp_c].set(0.0),
        residual_new,
        available.at[jk, :].set(True).at[cj, :].set(True),
        current_signs.at[jk].set(0.0).at[cj].set(0.0),
        current_components,
        _bool_scalar(True),
        removed_count + _device_scalar(1, removed_count.dtype),
    )


def _gpmo_backtracking_remove_pairs(
    x: jax.Array,
    residual: jax.Array,
    available: jax.Array,
    current_signs: jax.Array,
    current_components: jax.Array,
    selected_dipoles: jax.Array,
    connectivity: jax.Array,
    A_scaled: jax.Array,
    iteration: jax.Array,
    *,
    K: int,
    Nadjacent: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Mirror the C++ dewyrming pass over prior selected dipoles."""

    removed_count0 = _device_scalar(0, jnp.int64)

    def _seed_body(seed_state, hist_index):
        (
            x_seed,
            residual_seed,
            available_seed,
            signs_seed,
            components_seed,
            removed_count_seed,
        ) = seed_state
        jk = selected_dipoles[hist_index]
        sign_j = signs_seed[jk]
        comp_j = components_seed[jk]
        seed_active = (hist_index < iteration) & (sign_j != 0.0)

        def _neighbor_body(neighbor_state, neighbor_index):
            (
                x_neighbor,
                residual_neighbor,
                available_neighbor,
                signs_neighbor,
                components_neighbor,
                removed_for_seed,
                removed_count_neighbor,
            ) = neighbor_state
            cj = connectivity[jk, neighbor_index]
            should_remove = (
                seed_active
                & (~removed_for_seed)
                & (signs_neighbor[jk] != 0.0)
                & (signs_neighbor[jk] == -signs_neighbor[cj])
                & (comp_j == components_neighbor[cj])
            )
            return jax.lax.cond(
                should_remove,
                lambda args: _gpmo_backtracking_remove_one_pair(*args),
                lambda args: args[0],
                (
                    (
                        x_neighbor,
                        residual_neighbor,
                        available_neighbor,
                        signs_neighbor,
                        components_neighbor,
                        removed_for_seed,
                        removed_count_neighbor,
                    ),
                    A_scaled,
                    jk,
                    cj,
                ),
            ), None

        neighbor_state0 = (
            x_seed,
            residual_seed,
            available_seed,
            signs_seed,
            components_seed,
            _bool_scalar(False),
            removed_count_seed,
        )
        neighbor_state, _ = jax.lax.scan(
            _neighbor_body, neighbor_state0, jnp.arange(Nadjacent)
        )
        (
            x_out,
            residual_out,
            available_out,
            signs_out,
            components_out,
            _removed_for_seed,
            removed_count_out,
        ) = neighbor_state
        return (
            x_out,
            residual_out,
            available_out,
            signs_out,
            components_out,
            removed_count_out,
        ), None

    final_state, _ = jax.lax.scan(
        _seed_body,
        (x, residual, available, current_signs, current_components, removed_count0),
        jnp.arange(K),
    )
    return final_state


def gpmo_backtracking_step(
    spec: GPMOBacktrackingSpec,
    state: tuple[jax.Array, ...],
    A_scaled: jax.Array,
    connectivity: jax.Array,
    iteration: jax.Array,
    *,
    K: int,
    penalty: jax.Array | None = None,
    col_sq: jax.Array | None = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Run one normalized backtracking GPMO placement step."""

    (
        x,
        residual,
        available,
        current_signs,
        current_components,
        selected_dipoles,
        selected_components,
        selected_signs,
        done,
    ) = state
    candidate_spec = GPMOBaselineSpec(
        m_maxima=spec.m_maxima,
        reg_l2=spec.reg_l2,
        single_direction=spec.single_direction,
    )
    costs = gpmo_baseline_candidate_costs(
        candidate_spec,
        A_scaled,
        residual,
        available,
        penalty=penalty,
        col_sq=col_sq,
    )
    n_components = A_scaled.shape[1]
    choice = _argmin_finite_cost(costs)
    is_minus = choice >= n_components
    component_index = jnp.where(is_minus, choice - n_components, choice)
    sign = jnp.where(is_minus, -1.0, 1.0).astype(residual.dtype)
    dipole = component_index // 3
    component = component_index % 3

    x_placed = x.at[dipole, component].set(sign)
    residual_placed = residual + sign * A_scaled[:, component_index]
    available_placed = available.at[dipole, :].set(False)
    current_signs_placed = current_signs.at[dipole].set(sign)
    current_components_placed = current_components.at[dipole].set(component)
    selected_dipoles_placed = selected_dipoles.at[iteration].set(dipole)
    selected_components_placed = selected_components.at[iteration].set(component)
    selected_signs_placed = selected_signs.at[iteration].set(sign)

    backtrack_due = (iteration >= spec.backtracking) & (
        (iteration % spec.backtracking) == 0
    )
    (
        x_backtracked,
        residual_backtracked,
        available_backtracked,
        current_signs_backtracked,
        current_components_backtracked,
        removed_count,
    ) = jax.lax.cond(
        backtrack_due,
        lambda args: _gpmo_backtracking_remove_pairs(
            *args, K=K, Nadjacent=spec.Nadjacent
        ),
        lambda args: (
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            _device_scalar(0, jnp.int64),
        ),
        (
            x_placed,
            residual_placed,
            available_placed,
            current_signs_placed,
            current_components_placed,
            selected_dipoles_placed,
            connectivity,
            A_scaled,
            iteration,
        ),
    )
    num_nonzeros = jnp.sum(jnp.any(~available_backtracked, axis=1).astype(jnp.int64))
    stop = (num_nonzeros >= x.shape[0]) | (num_nonzeros >= spec.max_nMagnets)
    done_new = done | stop
    residual_sq = jnp.sum(residual_backtracked * residual_backtracked)

    next_state = (
        jnp.where(done, x, x_backtracked),
        jnp.where(done, residual, residual_backtracked),
        jnp.where(done, available, available_backtracked),
        jnp.where(done, current_signs, current_signs_backtracked),
        jnp.where(done, current_components, current_components_backtracked),
        jnp.where(done, selected_dipoles, selected_dipoles_placed),
        jnp.where(done, selected_components, selected_components_placed),
        jnp.where(done, selected_signs, selected_signs_placed),
        done_new,
    )
    trace = (
        jnp.where(done, _device_scalar(-1, jnp.int64), dipole),
        jnp.where(done, _device_scalar(-1, jnp.int64), component),
        jnp.where(done, _scalar_like(residual, 0.0), sign),
        jnp.where(done, jnp.sum(residual * residual), residual_sq),
        next_state[0],
        jnp.where(
            done, jnp.sum(jnp.any(~available, axis=1).astype(jnp.int64)), num_nonzeros
        ),
        jnp.where(done, _device_scalar(0, jnp.int64), removed_count),
        done_new,
    )
    return next_state, trace


@partial(jax.jit, static_argnames=("K", "record_every"))
def gpmo_backtracking_solve(
    spec: GPMOBacktrackingSpec,
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    K: int,
    record_every: int | None = None,
) -> GPMOBacktrackingResult:
    """Run the backtracking greedy permanent-magnet optimizer in JAX.

    The scan length is fixed by ``K``. Once the CPU stopping condition
    ``num_nonzero >= ndipoles`` or ``num_nonzero >= max_nMagnets`` is reached,
    later scan iterations carry the final state unchanged.
    """

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    m_maxima = _as_runtime_array(spec.m_maxima)
    ndipoles = int(m_maxima.shape[0])
    _validate_gpmo_backtracking_static_args(
        K,
        spec.single_direction,
        ndipoles,
        spec.Nadjacent,
        spec.backtracking,
        spec.max_nMagnets,
    )
    _validate_record_every(record_every)

    x0 = jnp.zeros((ndipoles, 3), dtype=A_arr.dtype)
    residual0 = -b_arr
    available0 = jnp.ones((ndipoles, 3), dtype=bool)
    current_signs0 = jnp.zeros((ndipoles,), dtype=A_arr.dtype)
    current_components0 = jnp.zeros((ndipoles,), dtype=jnp.int64)
    selected_dipoles0 = -jnp.ones((K,), dtype=jnp.int64)
    selected_components0 = -jnp.ones((K,), dtype=jnp.int64)
    selected_signs0 = jnp.zeros((K,), dtype=A_arr.dtype)
    if K == 0:
        empty_int = jnp.zeros((0,), dtype=jnp.int64)
        empty_float = jnp.zeros((0,), dtype=A_arr.dtype)
        empty_x_history = jnp.zeros((0, ndipoles, 3), dtype=A_arr.dtype)
        empty_bool = jnp.zeros((0,), dtype=bool)
        return GPMOBacktrackingResult(
            x=x0,
            x_history=empty_x_history,
            residual=residual0,
            residual_history=empty_float,
            selected_dipoles=empty_int,
            selected_components=empty_int,
            selected_signs=empty_float,
            num_nonzeros_history=empty_int,
            removed_pair_count_history=empty_int,
            done_history=empty_bool,
        )

    connectivity = gpmo_connectivity_matrix(spec.dipole_grid_xyz)
    scan_spec = GPMOBacktrackingSpec(
        m_maxima=m_maxima,
        reg_l2=_as_runtime_array(spec.reg_l2),
        dipole_grid_xyz=_as_runtime_array(spec.dipole_grid_xyz),
        single_direction=spec.single_direction,
        Nadjacent=spec.Nadjacent,
        backtracking=spec.backtracking,
        max_nMagnets=spec.max_nMagnets,
    )
    penalty = scan_spec.reg_l2 * _component_mmax(m_maxima) ** 2
    col_sq = jnp.sum(A_arr * A_arr, axis=0)

    def _scan_body(state, iteration):
        return gpmo_backtracking_step(
            scan_spec,
            state,
            A_arr,
            connectivity,
            iteration,
            K=K,
            penalty=penalty,
            col_sq=col_sq,
        )

    if record_every is not None:

        def _recording_step(state, iteration):
            return gpmo_backtracking_step(
                scan_spec,
                state,
                A_arr,
                connectivity,
                iteration,
                K=K,
                penalty=penalty,
                col_sq=col_sq,
            )

        final_state, x_history, residual_history, extra_histories = (
            _gpmo_recording_scan(
                _recording_step,
                (
                    x0,
                    residual0,
                    available0,
                    current_signs0,
                    current_components0,
                    selected_dipoles0,
                    selected_components0,
                    selected_signs0,
                    _bool_scalar(False),
                ),
                A_arr,
                K=K,
                record_every=record_every,
                ndipoles=ndipoles,
                extra_history_specs=(
                    ((), jnp.int64),
                    ((), jnp.int64),
                    ((), A_arr.dtype),
                    ((), jnp.int64),
                    ((), jnp.int64),
                    ((), bool),
                ),
                extra_trace_indices=(0, 1, 2, 5, 6, 7),
            )
        )
        (
            selected_dipoles,
            selected_components,
            selected_signs,
            num_nonzeros_history,
            removed_pair_count_history,
            done_history,
        ) = extra_histories
        return GPMOBacktrackingResult(
            x=final_state[0],
            x_history=x_history,
            residual=final_state[1],
            residual_history=residual_history,
            selected_dipoles=selected_dipoles,
            selected_components=selected_components,
            selected_signs=selected_signs,
            num_nonzeros_history=num_nonzeros_history,
            removed_pair_count_history=removed_pair_count_history,
            done_history=done_history,
        )

    final_state, trace = jax.lax.scan(
        _scan_body,
        (
            x0,
            residual0,
            available0,
            current_signs0,
            current_components0,
            selected_dipoles0,
            selected_components0,
            selected_signs0,
            _bool_scalar(False),
        ),
        jnp.arange(K),
    )
    (
        selected_dipoles,
        selected_components,
        selected_signs,
        residual_history,
        x_history,
        num_nonzeros_history,
        removed_pair_count_history,
        done_history,
    ) = trace
    return GPMOBacktrackingResult(
        x=final_state[0],
        x_history=x_history,
        residual=final_state[1],
        residual_history=residual_history,
        selected_dipoles=selected_dipoles,
        selected_components=selected_components,
        selected_signs=selected_signs,
        num_nonzeros_history=num_nonzeros_history,
        removed_pair_count_history=removed_pair_count_history,
        done_history=done_history,
    )


# ── Per-dipole helper kernels (mirror C++ scalar helpers) ───────────────


def projection_l2_balls(m: jax.Array, m_maxima: jax.Array) -> jax.Array:
    """Project each row of ``m`` onto the L2 ball with radius ``m_maxima``.

    Matches the Python CPU wrapper for public parity. Positive zero radii project
    nonzero rows to the zero ball while keeping their JAX derivatives finite;
    zero-radius zero rows preserve the CPU wrapper's ``0 / 0`` NaN result.
    Finite negative radii and infinities pass through; NaN radii propagate a NaN
    row.

    Parameters
    ----------
    m
        Shape ``(N, 3)``.
    m_maxima
        Shape ``(N,)``.
    """
    norm_sq = jnp.sum(m * m, axis=1)
    norm = _row_norm_without_zero_sqrt_gradient(norm_sq)
    unit = m_maxima**0
    zero = unit - unit
    finite_radius = jnp.isfinite(m_maxima)
    nan_radius = jnp.isnan(m_maxima)
    positive_radius = finite_radius & (m_maxima > zero)
    zero_radius = finite_radius & (m_maxima == zero)
    zero_norm = norm_sq == zero
    zero_radius_zero_norm = zero_radius & zero_norm
    positive_zero_radius = zero_radius & ~jnp.signbit(m_maxima)
    positive_zero_nonzero_radius = positive_zero_radius & ~zero_norm
    safe_radius = jnp.where(positive_radius, m_maxima, unit)
    radius_ratio = norm / safe_radius
    denom = jnp.maximum(unit, radius_ratio)
    projected = m / denom[:, None]
    zero_rows = jnp.broadcast_to(zero[:, None], m.shape)
    nan_rows = jnp.broadcast_to((zero / zero)[:, None], m.shape)
    return jnp.where(
        nan_radius[:, None],
        nan_rows,
        jnp.where(
            zero_radius_zero_norm[:, None],
            nan_rows,
            jnp.where(
                positive_zero_nonzero_radius[:, None],
                zero_rows,
                jnp.where(
                    (~positive_radius)[:, None],
                    m,
                    projected,
                ),
            ),
        ),
    )


def _on_ball(m: jax.Array, m_maxima: jax.Array) -> jax.Array:
    """Return a boolean mask ``(N,)`` of dipoles that are on the L2 ball.

    A dipole is considered "active" (on the ball) when
    ``|||m_i||^2 - m_maxima_i^2| < 1e-8 + 1e-5 m_maxima_i^2``. Matches
    the predicate in ``phi_MwPGP`` and ``beta_tilde``.
    """
    xmag2 = jnp.sum(m * m, axis=1)  # (N,)
    mmax2 = m_maxima * m_maxima  # (N,)
    abs_tol = _scalar_like(mmax2, _BALL_ACTIVE_ABS_TOL)
    rel_tol = _scalar_like(mmax2, _BALL_ACTIVE_REL_TOL)
    return jnp.abs(xmag2 - mmax2) < (abs_tol + rel_tol * mmax2)


def _row_norm_without_zero_sqrt_gradient(norm_sq: jax.Array) -> jax.Array:
    one = norm_sq**0
    zero = one - one
    has_norm = norm_sq > zero
    safe_norm_sq = jnp.where(has_norm, norm_sq, one)
    return jnp.where(has_norm, jnp.sqrt(safe_norm_sq), zero)


def phi_mwpgp(m: jax.Array, g: jax.Array, m_maxima: jax.Array) -> jax.Array:
    """Free-component projection of ``g``: zero out rows where ``m`` is on
    the ball, pass through otherwise.

    Mirrors ``phi_MwPGP`` in ``permanent_magnet_optimization.cpp:18``.
    """
    on_ball = _on_ball(m, m_maxima)
    zero = _scalar_like(g, 0.0)
    return jnp.where(on_ball[:, None], zero, g)


def g_reduced_gradient(
    m: jax.Array,
    g: jax.Array,
    alpha: jax.Array,
    m_maxima: jax.Array,
) -> jax.Array:
    """Compute ``(m - proj_L2(m - alpha g)) / alpha`` (the reduced gradient).

    Mirrors ``g_reduced_gradient`` in ``permanent_magnet_optimization.cpp:60``.
    """
    proj = projection_l2_balls(m - alpha * g, m_maxima)
    return (m - proj) / alpha


def _beta_tilde(
    m: jax.Array,
    g: jax.Array,
    alpha: jax.Array,
    m_maxima: jax.Array,
) -> jax.Array:
    """Active-component contribution: matches ``beta_tilde`` in
    ``permanent_magnet_optimization.cpp:34``.

    For each row:
      - off-ball -> zero (handled by ``phi`` already).
      - on-ball and ``<m_i, g_i> > 0`` (gradient pointing out of ball) -> g_i.
      - on-ball and ``<m_i, g_i> <= 0`` -> ``g_reduced_gradient``.
    """
    on_ball = _on_ball(m, m_maxima)
    # Avoid divide-by-zero and sqrt-at-zero gradients when ||m|| ≈ 0.
    norm_sq = jnp.sum(m * m, axis=1)
    norm_unit = norm_sq**0
    has_norm = norm_sq > (norm_unit - norm_unit)
    norm = _row_norm_without_zero_sqrt_gradient(norm_sq)
    safe_norm = jnp.where(has_norm, norm, norm_unit)
    mg = jnp.sum(m * g, axis=1)  # <m_i, g_i>
    ng = mg / safe_norm  # (N,)
    grg = g_reduced_gradient(m, g, alpha, m_maxima)  # (N, 3)
    zero = _scalar_like(g, 0.0)
    on_active_grad = jnp.where((ng > _scalar_like(ng, 0.0))[:, None], g, grg)
    return jnp.where(on_ball[:, None], on_active_grad, zero)


def g_reduced_projected_gradient(
    m: jax.Array,
    g: jax.Array,
    alpha: jax.Array,
    m_maxima: jax.Array,
) -> jax.Array:
    """``phi + beta_tilde``. Mirrors
    ``g_reduced_projected_gradient`` in ``permanent_magnet_optimization.cpp:68``.
    """
    return phi_mwpgp(m, g, m_maxima) + _beta_tilde(m, g, alpha, m_maxima)


def find_max_alphaf(
    m: jax.Array,
    p: jax.Array,
    m_maxima: jax.Array,
) -> jax.Array:
    """Per-dipole largest ``alpha`` such that ``m - alpha p`` stays in the
    L2 ball of radius ``m_maxima``.

    Mirrors ``find_max_alphaf`` in ``permanent_magnet_optimization.cpp:78``.
    Solves ``a alpha^2 + b alpha + c = 0`` with
    ``a = ||p||^2, b = -2 <m, p>, c = ||m||^2 - m_maxima^2``. ``c <= 0`` on
    the feasible set so the positive root is always nonnegative. Returns
    ``1e100`` for rows where ``a < tol``.

    Parameters
    ----------
    m
        Shape ``(N, 3)``.
    p
        Shape ``(N, 3)``.
    m_maxima
        Shape ``(N,)``.
    """
    a = jnp.sum(p * p, axis=1)
    zero = _scalar_like(a, 0.0)
    one = _scalar_like(a, 1.0)
    two = _scalar_like(a, 2.0)
    four = _scalar_like(a, 4.0)
    b = -two * jnp.sum(m * p, axis=1)
    c = jnp.sum(m * m, axis=1) - m_maxima * m_maxima
    disc = b * b - four * a * c
    # Guard the sqrt against tiny negative values arising from FP rounding
    # of the very-close-to-boundary case (``c ≈ 0``). The C++ kernel does
    # not need this because dipoles strictly satisfy ``c <= 0``; in JAX
    # autodiff this can still hit ``sqrt(-0.0)`` which is fine but a
    # tracer-safe ``maximum`` is harmless.
    active = a > _scalar_like(a, _FIND_MAX_ALPHAF_TOL)
    safe_disc = jnp.where(active, jnp.maximum(disc, zero), one)
    sqrt_disc = jnp.sqrt(safe_disc)
    safe_a = jnp.where(active, a, one)
    alphaf = (-b + sqrt_disc) / (two * safe_a)
    return jnp.where(
        active,
        alphaf,
        _scalar_like(alphaf, _find_max_alphaf_sentinel(alphaf.dtype)),
    )


# ── Hessian action (closed form, never materialised) ───────────────────


def _flatten_moments(m: jax.Array) -> jax.Array:
    return m.reshape(m.shape[0] * 3)


def _hessian_action(
    v: jax.Array,
    A: jax.Array,
    reg_l2: jax.Array,
    nu: jax.Array,
) -> jax.Array:
    """Compute ``H v`` where ``H = A^T A + 2 (reg_l2 + 1/(2 nu)) I``.

    ``v`` has shape ``(N, 3)`` and the operator acts on the flattened
    ``(3 N,)`` view. The C++ kernel evaluates the same expression in
    two ``Eigen::Map`` matrix-matrix multiplies (line 187 and line 216).
    """
    v_flat = _flatten_moments(v)
    Av = A @ v_flat
    AtAv = A.T @ Av
    one = _scalar_like(nu, 1.0)
    two = _scalar_like(nu, 2.0)
    scale = two * (reg_l2 + one / (two * nu))
    return (AtAv + scale * v_flat).reshape(v.shape[0], 3)


# ── Single iteration (mirrors the C++ main loop body, lines 203-322) ───


def _step_body(
    state: tuple[jax.Array, jax.Array, jax.Array],
    A: jax.Array,
    ATb_rs: jax.Array,
    m_maxima: jax.Array,
    alpha: jax.Array,
    reg_l2: jax.Array,
    nu: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single MwPGP iteration. Returns the updated ``(x, g, p)``.

    The control flow uses three branches (matches C++ lines 234-301):
      branch_inner_cg: ``|g_alpha_p|^2 <= |phi|^2`` AND ``alpha_cg < alpha_f``.
      branch_inner_expand: ``|g_alpha_p|^2 <= |phi|^2`` AND ``alpha_cg >= alpha_f``.
      branch_outer_proj: ``|g_alpha_p|^2 > |phi|^2`` -- vanilla projected gradient.
    """
    x, g, p = state

    g_alpha_p = g_reduced_projected_gradient(x, g, alpha, m_maxima)
    phi_g = phi_mwpgp(x, g, m_maxima)
    norm_g_alpha_p = jnp.sum(g_alpha_p * g_alpha_p)
    norm_phi = jnp.sum(phi_g * phi_g)
    inner = norm_g_alpha_p <= norm_phi

    # This branch is per-geometry scalar control flow. If a future caller vmaps
    # whole geometries through this step, both arms will execute under batching
    # and this should be revisited as select-style masking.
    def inner_branch(_operand):
        ATAp = _hessian_action(p, A, reg_l2, nu)
        gp = jnp.sum(g * p)
        pATAp = jnp.sum(p * ATAp)
        alpha_cg = gp / pATAp
        alpha_f = jnp.min(find_max_alphaf(x, p, m_maxima))

        def cg_branch(__operand):
            x_cg = x - alpha_cg * p
            g_cg = g - alpha_cg * ATAp
            phi_cg = phi_mwpgp(x_cg, g_cg, m_maxima)
            gamma_num = jnp.sum(phi_cg * ATAp)
            gamma = gamma_num / pATAp
            return x_cg, g_cg, phi_cg - gamma * p

        def expand_branch(__operand):
            x_expand_raw = (x - alpha_f * p) - alpha * (g - alpha_f * ATAp)
            x_expand = projection_l2_balls(x_expand_raw, m_maxima)
            g_expand = _hessian_action(x_expand, A, reg_l2, nu) - ATb_rs
            return x_expand, g_expand, phi_mwpgp(x_expand, g_expand, m_maxima)

        return jax.lax.cond(alpha_cg < alpha_f, cg_branch, expand_branch, None)

    def projected_gradient_branch(_operand):
        x_proj = projection_l2_balls(x - alpha * g, m_maxima)
        g_proj = _hessian_action(x_proj, A, reg_l2, nu) - ATb_rs
        return x_proj, g_proj, phi_mwpgp(x_proj, g_proj, m_maxima)

    return jax.lax.cond(inner, inner_branch, projected_gradient_branch, None)


def mwpgp_step(
    spec: PMOptimizationSpec,
    state: tuple[jax.Array, jax.Array, jax.Array],
    A: jax.Array,
    ATb: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single MwPGP iteration.

    Parameters
    ----------
    spec
        Solver hyperparameters.
    state
        ``(x, g, p)`` tuple from the previous iteration (or initialised).
    A
        Forward matrix, shape ``(M, 3 N)``.
    ATb
        ``A^T b`` shaped as ``(N, 3)``.

    Returns
    -------
    (x_new, g_new, p_new)
        Updated solver state.
    """
    A_arr = _as_runtime_array(A)
    ATb_arr = _as_runtime_array(ATb)
    m_maxima = _as_runtime_array(spec.m_maxima)
    m_proxy = _as_runtime_array(spec.m_proxy)
    nu = _as_runtime_array(spec.nu)
    reg_l2 = _as_runtime_array(spec.reg_l2)
    alpha = _as_runtime_array(spec.alpha)
    ATb_rs = ATb_arr + m_proxy / nu
    return _step_body(state, A_arr, ATb_rs, m_maxima, alpha, reg_l2, nu)


# ── Fixed-step scan solver (the public solver entry point) ─────────────


def _initial_state(
    m0: jax.Array,
    A: jax.Array,
    ATb_rs: jax.Array,
    m_maxima: jax.Array,
    reg_l2: jax.Array,
    nu: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mirror C++ lines 169-196: build the initial ``(x, g, p)`` triplet.

    ``x_k1 = m0``; ``g = H m0 - ATb_rs``; ``p = phi(m0, g)``.
    """
    g0 = _hessian_action(m0, A, reg_l2, nu) - ATb_rs
    p0 = phi_mwpgp(m0, g0, m_maxima)
    return m0, g0, p0


def mwpgp_initial_state(
    spec: PMOptimizationSpec,
    A: jax.Array,
    ATb: jax.Array,
    m0: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the public ``(x, g, p)`` state consumed by ``mwpgp_step``.

    ``ATb`` is the unshifted ``A^T b`` term. The relax-and-split
    ``m_proxy / nu`` contribution is applied from ``spec`` to mirror the
    internal solver initialization.
    """
    A_arr = _as_runtime_array(A)
    ATb_arr = _as_runtime_array(ATb)
    m0_arr = _as_runtime_array(m0)
    m_maxima = _as_runtime_array(spec.m_maxima)
    m_proxy = _as_runtime_array(spec.m_proxy)
    nu = _as_runtime_array(spec.nu)
    reg_l2 = _as_runtime_array(spec.reg_l2)
    ATb_rs = ATb_arr + m_proxy / nu
    return _initial_state(m0_arr, A_arr, ATb_rs, m_maxima, reg_l2, nu)


@partial(jax.jit, static_argnames=("n_steps", "record_residual"))
def mwpgp_solve(
    spec: PMOptimizationSpec,
    A: jax.Array,
    ATb: jax.Array,
    m0: jax.Array,
    *,
    n_steps: int,
    record_residual: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Run ``n_steps`` MwPGP iterations under ``jax.lax.scan``.

    Parameters
    ----------
    spec
        Hyperparameters.
    A
        Forward matrix shape ``(M, 3 N)``.
    ATb
        ``A^T b`` reshaped as ``(N, 3)``.
    m0
        Initial guess, shape ``(N, 3)``. Must lie in the L2 ball.
    n_steps
        Static iteration count. The solver never short-circuits.
    record_residual
        When true, return the per-step residual proxy history used by the
        direct MwPGP path. When false, skip that diagnostic matvec and return a
        zero history with the same static shape.

    Returns
    -------
    m_final
        Solution after ``n_steps`` iterations, shape ``(N, 3)``.
    R2_history
        Per-iteration ``||A m - b||^2`` traced under the scan, shape
        ``(n_steps,)``. Useful for diagnostic monotonicity checks. This is
        a fixed-length JAX trace, not the upstream C++ ``objective_history``
        / ``m_history`` buffer, and callers that need C++ early-stop history
        must use the CPU implementation.

    Notes
    -----
    The upstream ``b`` vector enters only through the ``ATb`` reduction
    for state updates; the per-iteration residual norm is computed from
    ``A m`` and the implied ``b`` via ``||A m - b||^2 = m^T A^T A m
    - 2 m^T ATb + ||b||^2``. We omit the constant ``||b||^2`` from the
    reported history (it cancels for monotonicity checks).
    """
    if not isinstance(n_steps, int):
        raise TypeError(f"n_steps must be a Python int; got {type(n_steps).__name__}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative; got {n_steps}")

    A_arr = _as_runtime_array(A)
    ATb_arr = _as_runtime_array(ATb)
    m0_arr = _as_runtime_array(m0)
    m_maxima = _as_runtime_array(spec.m_maxima)
    m_proxy = _as_runtime_array(spec.m_proxy)
    nu = _as_runtime_array(spec.nu)
    reg_l2 = _as_runtime_array(spec.reg_l2)
    alpha = _as_runtime_array(spec.alpha)

    ATb_rs = ATb_arr + m_proxy / nu
    init_state = _initial_state(m0_arr, A_arr, ATb_rs, m_maxima, reg_l2, nu)

    def _scan_body(state, _):
        x_new, g_new, p_new = _step_body(
            state, A_arr, ATb_rs, m_maxima, alpha, reg_l2, nu
        )
        # Diagnostic: residual-squared minus the constant ||b||^2 term.
        x_flat = _flatten_moments(x_new)
        Ax = A_arr @ x_flat
        residual_sq_proxy = jnp.sum(Ax * Ax) - _scalar_like(Ax, 2.0) * jnp.sum(
            x_flat * ATb_arr.reshape(-1)
        )
        return (x_new, g_new, p_new), residual_sq_proxy

    if n_steps == 0:
        return m0_arr, _zeros_like_shape(m0_arr, (0,), dtype=m0_arr.dtype)

    if record_residual:
        final_state, residual_history = jax.lax.scan(
            _scan_body, init_state, xs=None, length=n_steps
        )
        return final_state[0], residual_history

    def _scan_body_without_residual(state, _):
        x_new, g_new, p_new = _step_body(
            state, A_arr, ATb_rs, m_maxima, alpha, reg_l2, nu
        )
        return (x_new, g_new, p_new), None

    final_state, _ = jax.lax.scan(
        _scan_body_without_residual, init_state, xs=None, length=n_steps
    )
    return final_state[0], _zeros_like_shape(m0_arr, (n_steps,), dtype=m0_arr.dtype)
