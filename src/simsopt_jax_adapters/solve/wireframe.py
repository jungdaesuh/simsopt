"""JAX fixed-state solve helpers for wireframe optimization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt.field.magneticfield import (
    MagneticField,
    MagneticFieldMultiply,
    MagneticFieldSum,
)
from simsopt_jax_adapters.field.wireframe import WireframeFieldJAX
from simsopt.geo.surface import Surface
from simsopt.geo.wireframe_toroidal import ToroidalWireframe
from simsopt_jax.core._math_utils import (
    as_runtime_array as _as_runtime_array,
    as_runtime_value as _as_runtime_value,
    as_jax_int32 as _as_jax_int32,
    runtime_host_dtype as _runtime_host_dtype,
)
from simsopt_jax.core.wireframe_workflow import (
    WireframeGSCOResult,
    _gsco_opposite_candidate_index as _gsco_opposite_candidate_index,
    greedy_stellarator_coil_optimization_jax,
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


def _is_jax_native_field(field) -> bool:
    """Return True iff ``field`` and every composite child are JAX-native."""
    if bool(getattr(type(field), "_simsopt_jax_native_field", False)):
        return True
    if isinstance(field, MagneticFieldMultiply):
        return _is_jax_native_field(field.Bfield)
    if isinstance(field, MagneticFieldSum):
        return all(_is_jax_native_field(child) for child in field.Bfields)
    return False


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
    W_arr = jnp.squeeze(_as_runtime_array(W))
    indices = jax.lax.iota(jnp.int32, n)
    identity = (indices[:, None] == indices[None, :]).astype(W_arr.dtype)
    if W_arr.ndim == 0:
        return W_arr * identity
    if W_arr.ndim == 1:
        if W_arr.shape[0] != n:
            raise ValueError(
                "Number of elements in vector-form W must match columns in A"
            )
        return W_arr[:, None] * identity
    if W_arr.ndim == 2:
        if W_arr.shape != (n, n):
            raise ValueError(
                "Number of rows and columns in matrix-form W must both equal "
                "number of columns in A"
            )
        return W_arr
    raise ValueError("W must be a scalar, 1d array, or 2d array")


def _half_like(reference: jax.Array) -> jax.Array:
    return _as_runtime_value(0.5, reference=reference, dtype=reference.dtype)


def _scipy_lstsq_rcond(dtype) -> np.floating:
    host_dtype = np.dtype(dtype)
    eps = host_dtype.type(np.finfo(host_dtype).eps)
    return np.nextafter(eps, host_dtype.type(np.inf))


@jax.jit
def _regularized_constrained_least_squares_core(
    Amat: jax.Array,
    bvec: jax.Array,
    W: jax.Array,
    Cmat: jax.Array,
    dvec: jax.Array,
) -> jax.Array:
    m, n = (int(dim) for dim in Amat.shape)
    Wmat = _regularization_matrix(W, n)
    Ctra = Cmat.T
    Qfull, Rtall = jnp.linalg.qr(Ctra, mode="complete")
    p = int(Cmat.shape[0])
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

    rcond = _as_runtime_value(
        _scipy_lstsq_rcond(LHS.dtype),
        reference=LHS,
        dtype=LHS.dtype,
    )
    vvec = jnp.linalg.lstsq(LHS, RHS, rcond=rcond)[0]
    return Qfull @ jnp.concatenate((uvec, vvec), axis=0)


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

    Amat = _as_runtime_array(A)
    bvec = jnp.reshape(_as_runtime_array(b), (-1, 1))
    W_arr = _as_runtime_array(W)
    Cmat = _as_runtime_array(C)
    dvec = jnp.reshape(_as_runtime_array(d), (-1, 1))

    m, n = (int(dim) for dim in Amat.shape)
    if bvec.shape[0] != m:
        raise ValueError("Number of elements in b must match rows in A")
    if Cmat.shape[1] != n:
        raise ValueError("A and C must have the same number of columns")
    if dvec.shape[0] != Cmat.shape[0]:
        raise ValueError("Number of elements in d must match rows in C")

    return _regularized_constrained_least_squares_core(
        Amat,
        bvec,
        W_arr,
        Cmat,
        dvec,
    )


def _wireframe_regularization_value(W: object, x: jax.Array) -> jax.Array:
    W_arr = jnp.squeeze(_as_runtime_array(W))
    half = _half_like(x)
    if W_arr.ndim == 0:
        return half * W_arr**2 * jnp.sum(x**2)
    if W_arr.ndim == 1:
        return half * jnp.sum((W_arr * jnp.ravel(x)) ** 2)
    return half * jnp.sum((W_arr @ x) ** 2)


@partial(jax.jit, static_argnames=("n_segments",))
def _insert_free_segment_solution(
    free_segments: jax.Array,
    xfree: jax.Array,
    *,
    n_segments: int,
) -> jax.Array:
    x = jnp.zeros((int(n_segments), 1), dtype=xfree.dtype)
    return x.at[free_segments, :].set(xfree)


def _host_pytree(value: object):
    with jax.transfer_guard_device_to_host("allow"):
        return jax.device_get(value)


def _host_array(value: object, *, dtype=None) -> np.ndarray:
    out = _host_pytree(value)
    if dtype is None:
        return np.asarray(out)
    return np.asarray(out, dtype=dtype)


def _host_scalar(value: object):
    return _host_array(value).reshape(()).item()


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
    record_every: int | None = None,
    verbose: bool = True,
) -> WireframeGSCOResult:
    """Run fixed-state GSCO from a host wireframe without mutating currents."""

    del print_interval, verbose
    loops = np.asarray(wframe.get_cell_key(), dtype=np.int32)
    free_loops = np.asarray(wframe.get_free_cells(form="logical"), dtype=np.int32)
    segments = np.asarray(wframe.segments, dtype=np.int32)
    connections = np.asarray(wframe.connected_segments, dtype=np.int32)

    if x_init is None:
        x_init_arr = np.reshape(
            np.asarray(wframe.currents, dtype=_runtime_host_dtype()), (-1, 1)
        )
    else:
        x_init_arr = np.reshape(
            np.asarray(x_init, dtype=_runtime_host_dtype()), (-1, 1)
        )

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
        record_every=record_every,
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
    free_segs_host = np.asarray(wframe.unconstrained_segments(), dtype=np.intp)
    free_segs_device = _as_jax_int32(free_segs_host)
    A_arr = _as_runtime_array(Amat)
    b_arr = jnp.reshape(_as_runtime_array(bvec), (-1, 1))

    if A_arr.shape[1] != int(wframe.n_segments):
        raise ValueError("Input Amat has inconsistent dimensions with wframe")
    if np.shape(C)[0] >= len(free_segs_host):
        raise ValueError(
            "Least-squares problem has as many or more constraints than degrees of freedom."
        )

    if np.isscalar(reg_W):
        Wfree = reg_W
        Wfull = reg_W
    else:
        W_host = np.asarray(reg_W)
        if W_host.ndim == 1:
            Wfree = W_host[free_segs_host]
        elif W_host.ndim == 2:
            Wfree = W_host[free_segs_host, free_segs_host]
        else:
            raise ValueError("Input reg_W must be a scalar, 1d array, or 2d array")
        Wfull = W_host

    xfree = regularized_constrained_least_squares_jax(
        jnp.take(A_arr, free_segs_device, axis=1),
        b_arr,
        Wfree,
        C,
        d,
    )
    x = _insert_free_segment_solution(
        free_segs_device,
        xfree,
        n_segments=int(wframe.n_segments),
    )

    residual = A_arr @ x - b_arr
    f_B = _half_like(A_arr) * jnp.sum(residual * residual)
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

    host_dtype = _runtime_host_dtype()
    normal = np.asarray(surf_plas.normal(), dtype=host_dtype)
    absn = np.linalg.norm(normal, axis=2)[:, :, None]
    unitn = normal * (1.0 / absn)
    sqrt_area = np.sqrt(absn.reshape((-1, 1)) / float(absn.size))
    area_weight = (
        sqrt_area if area_weighted else np.ones(sqrt_area.shape, dtype=host_dtype)
    )

    wf_field = WireframeFieldJAX(wframe)
    wf_field.set_points(surf_plas.gamma().reshape((-1, 3)))
    A = np.ascontiguousarray(
        wf_field.dBnormal_by_dsegmentcurrents_matrix(
            surf_plas,
            area_weighted=area_weighted,
        ),
        dtype=host_dtype,
    )

    if ext_field is not None:
        B_ext = _host_array(
            _jax_native_ext_field_B(
                ext_field,
                surf_plas.gamma().reshape((-1, 3)),
            ),
            dtype=host_dtype,
        ).reshape(normal.shape)
        bnorm_ext = np.sum(B_ext * unitn, axis=2)[:, :, None]
        bnorm_ext_weighted = bnorm_ext.reshape((-1, 1)) * area_weight
    else:
        bnorm_ext_weighted = 0 * area_weight

    if bnorm_target is not None:
        target = np.asarray(bnorm_target, dtype=host_dtype)
        if target.size != area_weight.size:
            raise ValueError(
                "Input `bnorm_target` must have the same number of elements as "
                "the number of quadrature points of `surf_plas`"
            )
        bnorm_target_weighted = target.reshape((-1, 1)) * area_weight
    else:
        bnorm_target_weighted = 0 * area_weight

    b = np.ascontiguousarray(
        bnorm_target_weighted - bnorm_ext_weighted,
        dtype=host_dtype,
    )
    return A, b


def _precomputed_wireframe_matrices(wframe, Amat: object, bvec: object):
    host_dtype = _runtime_host_dtype()
    b = np.array(bvec, dtype=host_dtype).reshape((-1, 1))
    A = np.array(Amat, dtype=host_dtype)
    if np.shape(A) != (len(b), int(wframe.n_segments)):
        raise ValueError(
            "Input `Amat` has inconsistent dimensions with input `bvec` and/or `wframe`"
        )
    return A, b


def _gsco_initial_state(wframe, params: dict):
    host_dtype = _runtime_host_dtype()
    if params.get("x_init") is not None:
        x_init = np.array(
            np.reshape(params["x_init"], (-1, 1)),
            dtype=host_dtype,
            order="C",
            copy=True,
        )
    else:
        x_init = np.array(
            np.reshape(np.asarray(wframe.currents, dtype=host_dtype), (-1, 1)),
            dtype=host_dtype,
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

    host_dtype = _runtime_host_dtype()
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
        result = _host_pytree(
            rcls_wireframe_jax(
                wframe,
                A,
                b,
                params["reg_W"],
                assume_no_crossings=params.get("assume_no_crossings", False),
            )
        )
        x = np.asarray(result.x, dtype=host_dtype)
        _write_wireframe_currents(wframe, x)
        f_B = np.asarray(result.f_B).reshape(()).item()
        f_R = np.asarray(result.f_R).reshape(()).item()
        f = np.asarray(result.f).reshape(()).item()
        results["f_R"] = f_R
    elif algorithm_name == "gsco":
        for key in ("lambda_S", "max_iter", "print_interval"):
            if key not in params:
                raise ValueError(
                    f"params dictionary must contain `{key}` for the GSCO algorithm"
                )
        x_init, loop_count_init = _gsco_initial_state(wframe, params)
        result = _host_pytree(
            gsco_wireframe_jax(
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
                record_every=params.get("record_every"),
                verbose=verbose,
            )
        )
        x = np.asarray(result.x, dtype=host_dtype)
        _write_wireframe_currents(wframe, x)
        history_length = int(np.asarray(result.history_length).reshape(()).item())
        history_slice = slice(0, history_length)
        f_B_hist = np.asarray(result.f_B_history, dtype=host_dtype)[history_slice]
        f_S_hist = np.asarray(result.f_S_history, dtype=host_dtype)[history_slice]
        f_hist = np.asarray(result.f_history, dtype=host_dtype)[history_slice]
        f_B = f_B_hist[-1]
        f_S = f_S_hist[-1]
        f = f_hist[-1]
        results.update(
            {
                "loop_count": np.asarray(result.loop_count, dtype=np.int64),
                "iter_hist": np.asarray(result.iter_history, dtype=np.int64)[
                    history_slice
                ],
                "curr_hist": np.asarray(result.curr_history, dtype=host_dtype)[
                    history_slice
                ],
                "loop_hist": np.asarray(result.loop_history, dtype=np.int64)[
                    history_slice
                ],
                "f_B_hist": f_B_hist,
                "f_S_hist": f_S_hist,
                "f_hist": f_hist,
                "x_init": x_init,
                "record_every": params.get("record_every"),
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
    if res.get("record_every") not in (None, 1):
        raise ValueError("`record_every` GSCO histories do not support replay")
    if int(wframe.n_segments) != np.asarray(res["x_init"]).size:
        raise ValueError("Input `wframe` is not consistent with the solution in `res`")
    if iteration + 1 > np.asarray(res["loop_hist"]).size:
        raise ValueError("`iteration` exceeds number of iterations for solution")

    cells = wframe.get_cell_key()
    x_iter = np.array(res["x_init"], copy=True)
    for index in range(iteration + 1):
        curr_i = res["curr_hist"][index]
        cell_i = res["loop_hist"][index]
        x_iter[cells[cell_i, :2]] += curr_i
        x_iter[cells[cell_i, 2:]] -= curr_i
    return x_iter
