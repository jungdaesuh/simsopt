"""JAX fixed-state solve helpers for wireframe optimization."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..field.biotsavart_jax_backend import BiotSavartJAX
from ..field.magneticfield import MagneticField, _is_jax_native_field
from ..field.wireframefield_jax import WireframeFieldJAX
from ..geo.surface import Surface
from ..geo.wireframe_toroidal import ToroidalWireframe
from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
)

__all__ = [
    "WireframeGSCOResult",
    "WireframeRCLSResult",
    "bnorm_obj_matrices_jax",
    "get_gsco_iteration_jax",
    "greedy_stellarator_coil_optimization_jax",
    "gsco_wireframe_jax",
    "optimize_wireframe_jax",
    "rcls_wireframe_jax",
    "regularized_constrained_least_squares_jax",
]


@dataclass(frozen=True)
class WireframeRCLSResult:
    """Immutable result from ``rcls_wireframe_jax``."""

    x: jax.Array
    f_B: jax.Array
    f_R: jax.Array
    f: jax.Array


jax.tree_util.register_dataclass(
    WireframeRCLSResult,
    data_fields=["x", "f_B", "f_R", "f"],
    meta_fields=[],
)


@dataclass(frozen=True)
class WireframeGSCOResult:
    """Immutable fixed-shape result from the JAX GSCO kernel."""

    x: jax.Array
    loop_count: jax.Array
    history_length: jax.Array
    iter_history: jax.Array
    curr_history: jax.Array
    loop_history: jax.Array
    f_B_history: jax.Array
    f_S_history: jax.Array
    f_history: jax.Array


jax.tree_util.register_dataclass(
    WireframeGSCOResult,
    data_fields=[
        "x",
        "loop_count",
        "history_length",
        "iter_history",
        "curr_history",
        "loop_history",
        "f_B_history",
        "f_S_history",
        "f_history",
    ],
    meta_fields=[],
)


def _field_B_at_points(field, points):
    orig_points = field.get_points_cart_ref()
    field.set_points(points)
    B_ext = field.B()
    if orig_points is None and isinstance(field, BiotSavartJAX):
        field.clear_points()
    else:
        field.set_points(orig_points)
    return B_ext


def _jax_native_ext_field_B(ext_field, points):
    if isinstance(ext_field, MagneticField):
        if not _is_jax_native_field(ext_field):
            raise ValueError("Input `ext_field` must be a JAX-native MagneticField")
        return _field_B_at_points(ext_field, points)

    if isinstance(ext_field, BiotSavartJAX):
        return _field_B_at_points(ext_field, points)

    raise ValueError(
        "Input `ext_field` must be a JAX-native MagneticField or BiotSavartJAX"
    )


def _regularization_matrix(W: object, n: int) -> jax.Array:
    W_arr = jnp.squeeze(_as_jax_float64(W))
    if W_arr.ndim == 0:
        return W_arr * jnp.eye(n, dtype=W_arr.dtype)
    if W_arr.ndim == 1:
        if W_arr.shape[0] != n:
            raise ValueError(
                "Number of elements in vector-form W must match columns in A"
            )
        return jnp.diag(W_arr)
    if W_arr.ndim == 2:
        if W_arr.shape != (n, n):
            raise ValueError(
                "Number of rows and columns in matrix-form W must both equal "
                "number of columns in A"
            )
        return W_arr
    raise ValueError("W must be a scalar, 1d array, or 2d array")


def regularized_constrained_least_squares_jax(
    A: object,
    b: object,
    W: object,
    C: object,
    d: object,
) -> jax.Array:
    """JAX port of ``regularized_constrained_least_squares``.

    Solves ``min_x 0.5 * (||A x - b||^2 + ||W x||^2)`` subject to
    ``C x = d`` using the same QR basis construction as the CPU routine.
    """

    Amat = _as_jax_float64(A)
    bvec = jnp.reshape(_as_jax_float64(b), (-1, 1))
    Cmat = _as_jax_float64(C)
    dvec = jnp.reshape(_as_jax_float64(d), (-1, 1))

    m, n = Amat.shape
    if bvec.shape[0] != m:
        raise ValueError("Number of elements in b must match rows in A")
    if Cmat.shape[1] != n:
        raise ValueError("A and C must have the same number of columns")
    if dvec.shape[0] != Cmat.shape[0]:
        raise ValueError("Number of elements in d must match rows in C")

    Wmat = _regularization_matrix(W, n)
    Ctra = Cmat.T
    Qfull, Rtall = jnp.linalg.qr(Ctra, mode="complete")
    p = Cmat.shape[0]
    Q1mat = Qfull[:, :p]
    Q2mat = Qfull[:, p:]
    Rmat = Rtall[:p, :]

    uvec = jnp.linalg.solve(Rmat.T, dvec)

    AQ2mat = Amat @ Q2mat
    WQ2mat = Wmat @ Q2mat
    LHS = AQ2mat.T @ AQ2mat + WQ2mat.T @ WQ2mat

    AQ1mat = Amat @ Q1mat
    WQ1mat = Wmat @ Q1mat
    AQ1uvec = AQ1mat @ uvec
    WQ1uvec = WQ1mat @ uvec
    AQ2bvec = AQ2mat.T @ bvec
    RHS = AQ2bvec - AQ2mat.T @ AQ1uvec - WQ2mat.T @ WQ1uvec

    rcond = max(LHS.shape) * jnp.finfo(LHS.dtype).eps
    vvec = jnp.linalg.lstsq(LHS, RHS, rcond=rcond)[0]
    return Qfull @ jnp.concatenate((uvec, vvec), axis=0)


def _wireframe_regularization_value(W: object, x: jax.Array) -> jax.Array:
    W_arr = jnp.squeeze(_as_jax_float64(W))
    if W_arr.ndim == 0:
        return 0.5 * W_arr**2 * jnp.sum(x**2)
    if W_arr.ndim == 1:
        return 0.5 * jnp.sum((W_arr * jnp.ravel(x)) ** 2)
    return 0.5 * jnp.sum((W_arr @ x) ** 2)


def _host_array(array: object, *, dtype=None) -> np.ndarray:
    with jax.transfer_guard("allow"):
        out = np.asarray(jax.device_get(array))
    if dtype is None:
        return out
    return np.asarray(out, dtype=dtype)


def _host_scalar(array: object):
    return _host_array(array).reshape(()).item()


def _gsco_active_entries(x: jax.Array, tol: jax.Array) -> jax.Array:
    return jnp.where(jnp.abs(x) > tol, 1.0, 0.0)


def _gsco_two_f_s(x: jax.Array, tol: jax.Array) -> jax.Array:
    return jnp.sum(_gsco_active_entries(x, tol))


def _gsco_opposite_candidate_index(opt_ind: jax.Array, n_loops: int) -> jax.Array:
    return (opt_ind + n_loops) % (2 * n_loops)


def _gsco_candidate_currents(
    x: jax.Array,
    loop_count: jax.Array,
    loops: jax.Array,
    free_loops: jax.Array,
    segments: jax.Array,
    connections: jax.Array,
    default_current: jax.Array,
    max_current: jax.Array,
    max_loop_count: int,
    no_crossing: bool,
    no_new_coils: bool,
    match_current: bool,
    tol: jax.Array,
) -> jax.Array:
    n_loops = int(loops.shape[0])
    candidate_ids = jnp.arange(2 * n_loops, dtype=jnp.int32)
    loop_ids = candidate_ids % n_loops
    directions = jnp.where(candidate_ids < n_loops, 1.0, -1.0).astype(x.dtype)
    direction_counts = jnp.where(candidate_ids < n_loops, 1, -1).astype(
        loop_count.dtype
    )
    loop_inds = loops[loop_ids, :]
    loop_signs = jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=x.dtype)
    loop_x = x[loop_inds]

    eligible = free_loops[loop_ids] == 1
    if max_loop_count > 0:
        next_counts = loop_count[loop_ids] + direction_counts
        eligible = eligible & (jnp.abs(next_counts) <= max_loop_count)

    if no_new_coils:
        eligible = eligible & jnp.any(_gsco_active_entries(loop_x, tol) > 0.0, axis=1)

    if match_current:
        abs_loop_x = jnp.abs(loop_x)
        nonzero_currents = abs_loop_x > 0.0
        matched_abs_current = jnp.max(
            jnp.where(nonzero_currents, abs_loop_x, 0.0),
            axis=1,
        )
        mismatch = jnp.any(
            jnp.where(
                nonzero_currents,
                abs_loop_x != matched_abs_current[:, None],
                False,
            ),
            axis=1,
        )
        loop_current = jnp.where(
            matched_abs_current != 0.0,
            directions * matched_abs_current,
            directions * default_current,
        )
        eligible = eligible & ~mismatch
    else:
        loop_current = directions * default_current

    candidate_x = loop_x + loop_signs[None, :] * loop_current[:, None]
    eligible = eligible & jnp.all(jnp.abs(candidate_x) <= max_current, axis=1)

    if no_crossing:
        toroidal_segment_inds = loop_inds[:, [0, 2]]
        nodes = jnp.reshape(segments[toroidal_segment_inds, :], (2 * n_loops, 4))
        connected = connections[nodes, :]
        loop_deltas = loop_signs[None, :] * loop_current[:, None]
        matches = connected[:, :, :, None] == loop_inds[:, None, None, :]
        first_match = jnp.argmax(matches, axis=3)
        matched_delta = jnp.take_along_axis(
            jnp.broadcast_to(loop_deltas[:, None, None, :], matches.shape),
            first_match[:, :, :, None],
            axis=3,
        )[:, :, :, 0]
        current_to_add = jnp.where(jnp.any(matches, axis=3), matched_delta, 0.0)
        active_connections = jnp.abs(x[connected] + current_to_add) > tol
        crossing_found = jnp.any(jnp.sum(active_connections, axis=2) > 2, axis=1)
        eligible = eligible & ~crossing_found

    return jnp.where(eligible, loop_current, 0.0)


def _gsco_candidate_objectives(
    A: jax.Array,
    x: jax.Array,
    residual: jax.Array,
    loops: jax.Array,
    candidate_currents: jax.Array,
    lambda_s: jax.Array,
    two_f_s_latest: jax.Array,
    tol: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_loops = int(loops.shape[0])
    candidate_ids = jnp.arange(2 * n_loops, dtype=jnp.int32)
    loop_inds = loops[candidate_ids % n_loops, :]
    loop_signs = jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=x.dtype)
    loop_delta = loop_signs[None, :] * candidate_currents[:, None]

    field_delta = jnp.sum(A[:, loop_inds] * loop_delta[None, :, :], axis=2)
    two_f_b = jnp.sum((residual[:, None] + field_delta) ** 2, axis=0)

    loop_x = x[loop_inds]
    two_df_orig = jnp.sum(_gsco_active_entries(loop_x, tol), axis=1)
    two_df = jnp.sum(_gsco_active_entries(loop_x + loop_delta, tol), axis=1)
    two_f_s = two_f_s_latest + two_df - two_df_orig
    two_f = two_f_b + lambda_s * two_f_s
    inf = jnp.finfo(A.dtype).max
    eligible = candidate_currents != 0.0
    return (
        jnp.where(eligible, two_f_b, inf),
        jnp.where(eligible, two_f_s, inf),
        jnp.where(eligible, two_f, inf),
    )


def greedy_stellarator_coil_optimization_jax(
    no_crossing: bool,
    no_new_coils: bool,
    match_current: bool,
    A_obj: object,
    b_obj: object,
    default_current: float,
    max_current: float,
    max_loop_count: int,
    loops: object,
    free_loops: object,
    segments: object,
    connections: object,
    lambda_S: float,
    max_iter: int,
    x_init: object,
    loop_count_init: object,
) -> WireframeGSCOResult:
    """Pure-JAX fixed-state GSCO kernel matching the C++ greedy loop updates.

    The C++ implementation returns history arrays truncated to the accepted
    update count. This JAX kernel keeps fixed-size history arrays for JIT
    compatibility and returns ``history_length`` for slicing valid entries.
    A segment is considered active when ``abs(current) > tol`` with
    ``tol = 0.001 * abs(default_current)``. Thus ``default_current=0`` makes
    the active-current threshold exactly zero, matching the CPU edge case.
    """

    A = _as_jax_float64(A_obj)
    b = jnp.reshape(_as_jax_float64(b_obj), (-1,))
    x0 = jnp.reshape(_as_jax_float64(x_init), (-1,))
    loops_arr = _as_jax_int32(loops)
    free_loops_arr = _as_jax_int32(free_loops)
    segments_arr = _as_jax_int32(segments)
    connections_arr = _as_jax_int32(connections)
    loop_count0 = jnp.reshape(_as_jax_int32(loop_count_init), (-1,))

    n_loops = int(loops_arr.shape[0])
    n_iter = int(max_iter)
    default_current_abs = jnp.abs(jnp.asarray(default_current, dtype=A.dtype))
    max_current_abs = jnp.abs(jnp.asarray(max_current, dtype=A.dtype))
    lambda_s = jnp.asarray(lambda_S, dtype=A.dtype)
    max_loop_count_abs = abs(int(max_loop_count))
    tol = 0.001 * default_current_abs
    loop_signs = jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=A.dtype)

    residual0 = A @ x0 - b
    two_f_b0 = jnp.sum(residual0 * residual0)
    two_f_s0 = _gsco_two_f_s(x0, tol)
    two_f0 = two_f_b0 + lambda_s * two_f_s0

    iter_history0 = jnp.zeros((n_iter + 1,), dtype=jnp.int32)
    curr_history0 = jnp.zeros((n_iter + 1,), dtype=A.dtype)
    loop_history0 = jnp.zeros((n_iter + 1,), dtype=jnp.int32)
    f_b_history0 = jnp.zeros((n_iter + 1,), dtype=A.dtype).at[0].set(0.5 * two_f_b0)
    f_s_history0 = jnp.zeros((n_iter + 1,), dtype=A.dtype).at[0].set(0.5 * two_f_s0)
    f_history0 = jnp.zeros((n_iter + 1,), dtype=A.dtype).at[0].set(0.5 * two_f0)

    def _step(carry, iteration):
        (
            x,
            loop_count,
            residual,
            two_f_b_latest,
            two_f_s_latest,
            two_f_latest,
            opt_ind_prev,
            hist_ind,
            done,
            iter_history,
            curr_history,
            loop_history,
            f_b_history,
            f_s_history,
            f_history,
        ) = carry

        candidate_currents = _gsco_candidate_currents(
            x,
            loop_count,
            loops_arr,
            free_loops_arr,
            segments_arr,
            connections_arr,
            default_current_abs,
            max_current_abs,
            max_loop_count_abs,
            no_crossing,
            no_new_coils,
            match_current,
            tol,
        )
        two_f_bs, two_f_ss, two_fs = _gsco_candidate_objectives(
            A,
            x,
            residual,
            loops_arr,
            candidate_currents,
            lambda_s,
            two_f_s_latest,
            tol,
        )

        opt_ind = jnp.argmin(two_fs).astype(jnp.int32)
        current = candidate_currents[opt_ind]
        loop_ind = (opt_ind % n_loops).astype(jnp.int32)
        direction_count = jnp.where(opt_ind < n_loops, 1, -1).astype(loop_count.dtype)
        loop_inds = loops_arr[loop_ind, :]
        delta_x = loop_signs * current
        residual_delta = jnp.sum(A[:, loop_inds] * delta_x[None, :], axis=1)

        n_eligible = jnp.sum(candidate_currents != 0.0)
        stop_none_eligible = n_eligible < 1
        opposite_opt_ind = _gsco_opposite_candidate_index(opt_ind, n_loops)
        stop_undone_loop = (iteration > 0) & (opposite_opt_ind == opt_ind_prev)
        stop_last_iter = iteration + 1 == n_iter
        reject_undo = stop_undone_loop & (two_fs[opt_ind] > two_f_latest)
        accept_loop = (~done) & (~stop_none_eligible) & (~reject_undo)
        stop_now = (~done) & (stop_none_eligible | stop_undone_loop | stop_last_iter)

        x_candidate = x.at[loop_inds].add(delta_x)
        loop_count_candidate = loop_count.at[loop_ind].add(direction_count)
        residual_candidate = residual + residual_delta
        two_f_b_candidate = two_f_bs[opt_ind]
        two_f_s_candidate = two_f_ss[opt_ind]
        two_f_candidate = two_fs[opt_ind]

        x_next = jnp.where(accept_loop, x_candidate, x)
        loop_count_next = jnp.where(accept_loop, loop_count_candidate, loop_count)
        residual_next = jnp.where(accept_loop, residual_candidate, residual)
        two_f_b_next = jnp.where(accept_loop, two_f_b_candidate, two_f_b_latest)
        two_f_s_next = jnp.where(accept_loop, two_f_s_candidate, two_f_s_latest)
        two_f_next = jnp.where(accept_loop, two_f_candidate, two_f_latest)

        hist_ind_next = hist_ind + accept_loop.astype(hist_ind.dtype)
        iter_history_candidate = iter_history.at[hist_ind_next].set(hist_ind_next)
        curr_history_candidate = curr_history.at[hist_ind_next].set(current)
        loop_history_candidate = loop_history.at[hist_ind_next].set(loop_ind)
        f_b_history_candidate = f_b_history.at[hist_ind_next].set(0.5 * two_f_b_next)
        f_s_history_candidate = f_s_history.at[hist_ind_next].set(0.5 * two_f_s_next)
        f_history_candidate = f_history.at[hist_ind_next].set(0.5 * two_f_next)

        iter_history_next = jnp.where(accept_loop, iter_history_candidate, iter_history)
        curr_history_next = jnp.where(accept_loop, curr_history_candidate, curr_history)
        loop_history_next = jnp.where(accept_loop, loop_history_candidate, loop_history)
        f_b_history_next = jnp.where(accept_loop, f_b_history_candidate, f_b_history)
        f_s_history_next = jnp.where(accept_loop, f_s_history_candidate, f_s_history)
        f_history_next = jnp.where(accept_loop, f_history_candidate, f_history)

        opt_ind_prev_next = jnp.where(
            (~done) & (~stop_none_eligible) & (~stop_undone_loop),
            opt_ind,
            opt_ind_prev,
        )
        done_next = done | stop_now

        return (
            x_next,
            loop_count_next,
            residual_next,
            two_f_b_next,
            two_f_s_next,
            two_f_next,
            opt_ind_prev_next,
            hist_ind_next,
            done_next,
            iter_history_next,
            curr_history_next,
            loop_history_next,
            f_b_history_next,
            f_s_history_next,
            f_history_next,
        ), None

    initial_carry = (
        x0,
        loop_count0,
        residual0,
        two_f_b0,
        two_f_s0,
        two_f0,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        iter_history0,
        curr_history0,
        loop_history0,
        f_b_history0,
        f_s_history0,
        f_history0,
    )
    final_carry, _ = jax.lax.scan(
        _step,
        initial_carry,
        jnp.arange(n_iter, dtype=jnp.int32),
    )
    (
        x,
        loop_count,
        _residual,
        _two_f_b,
        _two_f_s,
        _two_f,
        _opt_ind_prev,
        hist_ind,
        _done,
        iter_history,
        curr_history,
        loop_history,
        f_b_history,
        f_s_history,
        f_history,
    ) = final_carry
    history_length = hist_ind + jnp.asarray(1, dtype=hist_ind.dtype)
    return WireframeGSCOResult(
        x=jnp.reshape(x, (-1, 1)),
        loop_count=loop_count,
        history_length=history_length,
        iter_history=iter_history,
        curr_history=curr_history,
        loop_history=loop_history,
        f_B_history=f_b_history,
        f_S_history=f_s_history,
        f_history=f_history,
    )


def gsco_wireframe_jax(
    wframe,
    A: object,
    c: object,
    lambda_S: float,
    no_crossing: bool,
    match_current: bool,
    default_current: float,
    max_current: float,
    max_iter: int,
    print_interval: int,
    no_new_coils: bool = False,
    max_loop_count: int = 0,
    x_init: object | None = None,
    loop_count_init: object | None = None,
    verbose: bool = True,
) -> WireframeGSCOResult:
    """Run fixed-state GSCO from a host wireframe without mutating currents."""

    del print_interval, verbose
    loops = np.asarray(wframe.get_cell_key(), dtype=np.int32)
    free_loops = np.asarray(wframe.get_free_cells(form="logical"), dtype=np.int32)
    segments = np.asarray(wframe.segments, dtype=np.int32)
    connections = np.asarray(wframe.connected_segments, dtype=np.int32)

    if x_init is None:
        x_init_arr = np.reshape(np.asarray(wframe.currents, dtype=np.float64), (-1, 1))
    else:
        x_init_arr = np.reshape(np.asarray(x_init, dtype=np.float64), (-1, 1))

    if loop_count_init is None:
        loop_count_arr = np.zeros(len(free_loops), dtype=np.int32)
    else:
        loop_count_arr = np.asarray(loop_count_init, dtype=np.int32)

    return greedy_stellarator_coil_optimization_jax(
        no_crossing,
        no_new_coils,
        match_current,
        A,
        c,
        abs(default_current),
        abs(max_current),
        abs(max_loop_count),
        loops,
        free_loops,
        segments,
        connections,
        lambda_S,
        max_iter,
        x_init_arr,
        loop_count_arr,
    )


def rcls_wireframe_jax(
    wframe,
    Amat: object,
    bvec: object,
    reg_W: object,
    assume_no_crossings: bool = False,
) -> WireframeRCLSResult:
    """Run the RCLS wireframe solve with immutable JAX result arrays.

    Host ``wframe`` geometry is used only to obtain the same constraint
    matrices and unconstrained segment indices as the CPU routine. The JAX path
    returns arrays and does not mutate ``wframe.currents``.
    """

    C, d = wframe.constraint_matrices(
        assume_no_crossings=assume_no_crossings,
        remove_constrained_segments=True,
    )
    free_segs = np.asarray(wframe.unconstrained_segments())
    A_arr = _as_jax_float64(Amat)
    b_arr = jnp.reshape(_as_jax_float64(bvec), (-1, 1))

    if A_arr.shape[1] != int(wframe.n_segments):
        raise ValueError("Input Amat has inconsistent dimensions with wframe")
    if np.shape(C)[0] >= len(free_segs):
        raise ValueError(
            "Least-squares problem has as many or more constraints than degrees of freedom."
        )

    if np.isscalar(reg_W):
        Wfree = reg_W
        Wfull = reg_W
    else:
        W_host = np.asarray(reg_W)
        if W_host.ndim == 1:
            Wfree = W_host[free_segs]
        elif W_host.ndim == 2:
            Wfree = W_host[free_segs, free_segs]
        else:
            raise ValueError("Input reg_W must be a scalar, 1d array, or 2d array")
        Wfull = W_host

    xfree = regularized_constrained_least_squares_jax(
        A_arr[:, free_segs],
        b_arr,
        Wfree,
        C,
        d,
    )
    x = jnp.zeros((int(wframe.n_segments), 1), dtype=A_arr.dtype)
    x = x.at[free_segs, :].set(xfree)

    residual = A_arr @ x - b_arr
    f_B = 0.5 * jnp.sum(residual * residual)
    f_R = _wireframe_regularization_value(Wfull, x)
    return WireframeRCLSResult(x=x, f_B=f_B, f_R=f_R, f=f_B + f_R)


def bnorm_obj_matrices_jax(
    wframe,
    surf_plas,
    ext_field=None,
    area_weighted: bool = True,
    bnorm_target=None,
    verbose: bool = True,
):
    """JAX-backed normal-field matrices for wireframe current optimization."""

    del verbose
    if not isinstance(surf_plas, Surface):
        raise ValueError("Input `surf_plas` must be a Surface class instance")

    normal = surf_plas.normal()
    absn = np.linalg.norm(normal, axis=2)[:, :, None]
    unitn = normal * (1.0 / absn)
    sqrt_area = np.sqrt(absn.reshape((-1, 1)) / float(absn.size))
    area_weight = sqrt_area if area_weighted else np.ones(sqrt_area.shape)

    wf_field = WireframeFieldJAX(wframe)
    wf_field.set_points(surf_plas.gamma().reshape((-1, 3)))
    A = wf_field.dBnormal_by_dsegmentcurrents_matrix(
        surf_plas,
        area_weighted=area_weighted,
    )

    if ext_field is not None:
        B_ext = np.asarray(
            _jax_native_ext_field_B(
                ext_field,
                surf_plas.gamma().reshape((-1, 3)),
            ),
            dtype=np.float64,
        ).reshape(normal.shape)
        bnorm_ext = np.sum(B_ext * unitn, axis=2)[:, :, None]
        bnorm_ext_weighted = bnorm_ext.reshape((-1, 1)) * area_weight
    else:
        bnorm_ext_weighted = 0 * area_weight

    if bnorm_target is not None:
        target = np.asarray(bnorm_target, dtype=np.float64)
        if target.size != area_weight.size:
            raise ValueError(
                "Input `bnorm_target` must have the same number of elements as "
                "the number of quadrature points of `surf_plas`"
            )
        bnorm_target_weighted = target.reshape((-1, 1)) * area_weight
    else:
        bnorm_target_weighted = 0 * area_weight

    b = np.ascontiguousarray(bnorm_target_weighted - bnorm_ext_weighted)
    return A, b


def _precomputed_wireframe_matrices(wframe, Amat: object, bvec: object):
    b = np.array(bvec, dtype=np.float64).reshape((-1, 1))
    A = np.array(Amat, dtype=np.float64)
    if np.shape(A) != (len(b), int(wframe.n_segments)):
        raise ValueError(
            "Input `Amat` has inconsistent dimensions with input `bvec` and/or `wframe`"
        )
    return A, b


def _gsco_initial_state(wframe, params: dict):
    if params.get("x_init") is not None:
        x_init = np.array(
            np.reshape(params["x_init"], (-1, 1)),
            dtype=np.float64,
            order="C",
            copy=True,
        )
    else:
        x_init = np.array(
            np.reshape(np.asarray(wframe.currents, dtype=np.float64), (-1, 1)),
            dtype=np.float64,
            order="C",
            copy=True,
        )

    free_loops = wframe.get_free_cells(form="logical")
    if params.get("loop_count_init") is not None:
        loop_count_init = np.ascontiguousarray(params["loop_count_init"]).astype(
            np.int64
        )
    else:
        loop_count_init = np.ascontiguousarray(
            np.zeros(len(free_loops), dtype=np.int64)
        )
    return x_init, loop_count_init


def _write_wireframe_currents(wframe, x: np.ndarray) -> None:
    wframe.currents[:] = 0
    wframe.currents[:] = x.reshape((-1))[:]


def optimize_wireframe_jax(
    wframe,
    algorithm: str,
    params: dict,
    surf_plas=None,
    ext_field=None,
    area_weighted: bool = True,
    bnorm_target=None,
    Amat=None,
    bvec=None,
    verbose: bool = True,
):
    """JAX-backed public wrapper for fixed-state wireframe current solves."""

    if not isinstance(wframe, ToroidalWireframe):
        raise ValueError("Input `wframe` must be a ToroidalWireframe class instance")

    if surf_plas is not None:
        if Amat is not None or bvec is not None:
            raise ValueError(
                "Inputs `Amat` and `bvec` must not be supplied if `surf_plas` is given"
            )
        A, b = bnorm_obj_matrices_jax(
            wframe,
            surf_plas,
            ext_field=ext_field,
            area_weighted=area_weighted,
            bnorm_target=bnorm_target,
            verbose=verbose,
        )
    elif Amat is not None and bvec is not None:
        if surf_plas is not None or ext_field is not None or bnorm_target is not None:
            raise ValueError(
                "If `Amat` and `bvec` are provided, `surf_plas`, `ext_field`, "
                "and `bnorm_target` must not be provided"
            )
        A, b = _precomputed_wireframe_matrices(wframe, Amat, bvec)
    else:
        raise ValueError("`surf_plas` or `Amat` and `bvec` must be supplied")

    results = {}
    algorithm_name = algorithm.lower()
    if algorithm_name == "rcls":
        if "reg_W" not in params:
            raise ValueError(
                "params dictionary must contain `reg_W` for the RCLS algorithm"
            )
        result = rcls_wireframe_jax(
            wframe,
            A,
            b,
            params["reg_W"],
            assume_no_crossings=params.get("assume_no_crossings", False),
        )
        x = _host_array(result.x, dtype=np.float64)
        _write_wireframe_currents(wframe, x)
        f_B = _host_scalar(result.f_B)
        f_R = _host_scalar(result.f_R)
        f = _host_scalar(result.f)
        results["f_R"] = f_R
    elif algorithm_name == "gsco":
        for key in ("lambda_S", "max_iter", "print_interval"):
            if key not in params:
                raise ValueError(
                    f"params dictionary must contain `{key}` for the GSCO algorithm"
                )
        x_init, loop_count_init = _gsco_initial_state(wframe, params)
        result = gsco_wireframe_jax(
            wframe,
            A,
            b,
            params["lambda_S"],
            params.get("no_crossing", False),
            params.get("match_current", False),
            params.get("default_current", 0.0),
            params.get("max_current", np.inf),
            params["max_iter"],
            params["print_interval"],
            no_new_coils=params.get("no_new_coils", False),
            max_loop_count=params.get("max_loop_count", 0),
            x_init=x_init,
            loop_count_init=loop_count_init,
            verbose=verbose,
        )
        x = _host_array(result.x, dtype=np.float64)
        _write_wireframe_currents(wframe, x)
        history_length = int(_host_scalar(result.history_length))
        history_slice = slice(0, history_length)
        f_B_hist = _host_array(result.f_B_history, dtype=np.float64)[history_slice]
        f_S_hist = _host_array(result.f_S_history, dtype=np.float64)[history_slice]
        f_hist = _host_array(result.f_history, dtype=np.float64)[history_slice]
        f_B = f_B_hist[-1]
        f_S = f_S_hist[-1]
        f = f_hist[-1]
        results.update(
            {
                "loop_count": _host_array(result.loop_count, dtype=np.int64),
                "iter_hist": _host_array(result.iter_history, dtype=np.int64)[
                    history_slice
                ],
                "curr_hist": _host_array(result.curr_history, dtype=np.float64)[
                    history_slice
                ],
                "loop_hist": _host_array(result.loop_history, dtype=np.int64)[
                    history_slice
                ],
                "f_B_hist": f_B_hist,
                "f_S_hist": f_S_hist,
                "f_hist": f_hist,
                "x_init": x_init,
                "f_S": f_S,
            }
        )
    else:
        raise ValueError(f"Unrecognized algorithm {algorithm}")

    results.update(
        {
            "x": x,
            "Amat": A,
            "bvec": b,
            "wframe_field": WireframeFieldJAX(wframe),
            "f_B": f_B,
            "f": f,
        }
    )
    return results


def get_gsco_iteration_jax(iteration: int, res: dict, wframe):
    """Return the intermediate fixed-state GSCO solution at one iteration."""

    if "loop_hist" not in res or "curr_hist" not in res or "x_init" not in res:
        raise ValueError("`res` does not appear to contain data from a GSCO procedure")
    if int(wframe.n_segments) != np.asarray(res["x_init"]).size:
        raise ValueError("Input `wframe` is not consistent with the solution in `res`")
    if iteration + 1 > np.asarray(res["loop_hist"]).size:
        raise ValueError("`iteration` exceeds number of iterations for solution")

    cells = wframe.get_cell_key()
    x_iter = np.array(res["x_init"], dtype=np.float64)
    for index in range(iteration + 1):
        curr_i = res["curr_hist"][index]
        cell_i = res["loop_hist"][index]
        x_iter[cells[cell_i, :2]] += curr_i
        x_iter[cells[cell_i, 2:]] -= curr_i
    return x_iter
