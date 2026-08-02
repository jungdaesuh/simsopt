"""On-device BFGS solver implemented as a ``lax.while_loop``."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from simsopt_jax.runtime.host_boundary import (
    host_bool,
    host_int,
    host_value,
)
from simsopt_jax.runtime.jaxpr_closure import (
    closure_converted_value_and_grad,
    device_put_closure_consts,
)

from .._shared import _prepare_optimizer_callable_inputs
from ._common import (
    _as_jax_dtype,
    _bool_scalar,
    _cached_private_solver,
    _dot,
    _einsum,
    _emit_iteration_callbacks,
    _eye,
    _int_scalar,
    _norm,
    _require_private_optimizer_runtime,
    _scalar_value_and_grad,
)
from ._line_search import _line_search, _line_search_value_and_grad
from ._step_runtime import ContinueDecision, RuntimeLimits, StepOps, run_eager
from ._types import BFGS_STATUS_CALLBACK_STOP, _BFGSResults, _LineSearchResults


class _BFGSTransition(NamedTuple):
    state: _BFGSResults
    accepted: bool | jax.Array


class _BFGSObservation(NamedTuple):
    terminal: bool | jax.Array
    accepted: bool | jax.Array
    iteration: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array


class _BFGSObserverObservation(NamedTuple):
    terminal: bool | jax.Array
    accepted: bool | jax.Array
    iteration: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    x_k: jax.Array
    f_k: jax.Array
    g_k: jax.Array


class _BFGSHostObservation(NamedTuple):
    terminal: bool
    accepted: bool
    iteration: int
    nfev: int
    ngev: int


class _BFGSObserverHostObservation(NamedTuple):
    terminal: bool
    accepted: bool
    iteration: int
    nfev: int
    ngev: int
    x_k: np.ndarray
    f_k: float
    g_k: np.ndarray


_BFGSConsts = tuple[jax.Array, ...]
_BFGSValueAndGrad = Callable[
    [jax.Array, _BFGSConsts],
    tuple[jax.Array, jax.Array],
]
_BFGSScalarObjective = Callable[[jax.Array], jax.Array]
_BFGSScalarValueAndGrad = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
_BFGSInputObjective = _BFGSScalarObjective | _BFGSScalarValueAndGrad
_BFGSLineSearch = Callable[..., _LineSearchResults]
_BFGSCallback = Callable[[np.ndarray], object]
_BFGSProgressCallback = Callable[[int, float, float], object]
_BFGSStepKernel = Callable[[_BFGSResults, _BFGSConsts], _BFGSResults]
_BFGSFinalizeKernel = Callable[
    [_BFGSResults, _BFGSConsts, jax.Array],
    _BFGSResults,
]


def _state_dtype(value: int | jax.Array) -> jax.typing.DTypeLike:
    """Return a counter/status dtype without materializing a tracer."""

    if isinstance(value, jax.Array):
        return value.dtype
    return np.int32


class _CompiledMemoryAnalysis(Protocol):
    argument_size_in_bytes: int
    output_size_in_bytes: int
    alias_size_in_bytes: int
    temp_size_in_bytes: int


class _CompiledStep(Protocol):
    def memory_analysis(self) -> _CompiledMemoryAnalysis: ...


def _bfgs_memory_contract(
    dimension: int,
    dtype: jax.typing.DTypeLike,
) -> dict[str, int | bool]:
    """Return conservative logical byte accounting for one dense update.

    The accounting assumes no buffer donation: old and new inverse Hessians,
    four matrix-shaped update intermediates, and four vector-shaped operands
    may coexist. XLA fusion can reduce physical live bytes; this is an upper
    bound for receipts, not a routing threshold.
    """

    n = int(dimension)
    itemsize = int(np.dtype(dtype).itemsize)
    matrix_bytes = n * n * itemsize
    vector_bytes = n * itemsize
    return {
        "dimension": n,
        "dtype_itemsize": itemsize,
        "inverse_hessian_bytes": matrix_bytes,
        "simultaneous_old_new_hessian_bytes": 2 * matrix_bytes,
        "matrix_intermediate_bytes": 4 * matrix_bytes,
        "vector_intermediate_bytes": 4 * vector_bytes,
        "fixed_scalar_overhead_bytes": 1024,
        "derived_peak_live_upper_bound_bytes": (
            6 * matrix_bytes + 4 * vector_bytes + 1024
        ),
        "buffer_donation": False,
    }


def _compiled_step_memory_analysis(compiled: _CompiledStep) -> dict[str, int]:
    """Return XLA buffer-accounting fields for one compiled BFGS step."""

    analysis = compiled.memory_analysis()
    argument_bytes = int(analysis.argument_size_in_bytes)
    output_bytes = int(analysis.output_size_in_bytes)
    alias_bytes = int(analysis.alias_size_in_bytes)
    temp_bytes = int(analysis.temp_size_in_bytes)
    return {
        "compiled_step_argument_bytes": argument_bytes,
        "compiled_step_output_bytes": output_bytes,
        "compiled_step_alias_bytes": alias_bytes,
        "compiled_step_temp_bytes": temp_bytes,
        "compiled_step_peak_live_bytes": max(
            0,
            argument_bytes + output_bytes + temp_bytes - alias_bytes,
        ),
    }


def _closure_constants_signature(
    constants: _BFGSConsts,
) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (
            tuple(int(dimension) for dimension in getattr(leaf, "shape", ())),
            np.dtype(getattr(leaf, "dtype", np.float64)).str,
        )
        for leaf in jax.tree_util.tree_leaves(constants)
    )


def _cached_bfgs_value_and_grad_program(
    fun: _BFGSInputObjective,
    *,
    cache_owner: object | None,
    value_and_grad: bool,
    dtype: jax.typing.DTypeLike,
    shape: tuple[int, ...],
) -> _BFGSScalarValueAndGrad:
    if value_and_grad:
        return cast(_BFGSScalarValueAndGrad, fun)
    return _cached_private_solver(
        cache_owner,
        cache_key=(
            "bfgs-scalar-value-and-grad",
            np.dtype(dtype).str,
            tuple(int(dimension) for dimension in shape),
        ),
        builder=lambda: _scalar_value_and_grad(cast(_BFGSScalarObjective, fun)),
    )


def _bfgs_curvature_terms(
    s_k: jax.Array,
    y_k: jax.Array,
    *,
    x_dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return curvature terms for the dense BFGS inverse-Hessian update."""
    rho_k_inv = _dot(y_k, s_k)
    rho_k = jnp.reciprocal(rho_k_inv)
    machine_eps = _as_jax_dtype(np.finfo(np.dtype(x_dtype)).eps, x_dtype)
    curvature_eps = jnp.sqrt(machine_eps)
    curvature_scale = _norm(s_k) * _norm(y_k)
    curvature_tol = curvature_eps * _as_jax_dtype(
        curvature_scale,
        rho_k_inv.dtype,
    )
    valid_curvature = (
        jnp.isfinite(rho_k_inv)
        & jnp.isfinite(rho_k)
        & jnp.isfinite(curvature_scale)
        & (rho_k_inv > curvature_tol)
    )
    return rho_k_inv, rho_k, valid_curvature


def _bfgs_inverse_hessian_update(
    H_k: jax.Array,
    s_k: jax.Array,
    y_k: jax.Array,
    *,
    base_identity_host: np.ndarray,
) -> jax.Array:
    """Apply the dense inverse-Hessian update used by the BFGS step."""

    _rho_k_inv, rho_k, valid_curvature = _bfgs_curvature_terms(
        s_k,
        y_k,
        x_dtype=H_k.dtype,
    )
    sy_k = s_k[:, np.newaxis] * y_k[np.newaxis, :]
    identity = _as_jax_dtype(base_identity_host, rho_k.dtype)
    w = identity - rho_k * sy_k
    H_kp1 = (
        _einsum("ij,jk,lk", w, H_k, w) + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :]
    )
    return jnp.where(valid_curvature, H_kp1, H_k)


def _bfgs_accepted_step(
    state: _BFGSResults,
    explicit_value_and_grad_consts: _BFGSConsts,
    *,
    converted_value_and_grad: _BFGSValueAndGrad,
    line_search: _BFGSLineSearch,
    value_and_grad: bool,
    line_search_maxiter: int,
    norm: float,
    gtol_value: float,
    xrtol_value: float,
    base_identity_host: np.ndarray,
    allow_decreasing_fallback: bool,
) -> _BFGSResults:
    """Advance one BFGS step without a solve-length control loop or callback."""

    gtol_jax = _as_jax_dtype(gtol_value, state.g_k.dtype)
    wolfe_c1 = _as_jax_dtype(1e-4, state.f_k.dtype)
    wolfe_c2 = _as_jax_dtype(0.9, state.f_k.dtype)
    p_k = -_dot(state.H_k, state.g_k)

    def explicit_value_and_grad(
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return converted_value_and_grad(x, explicit_value_and_grad_consts)

    def explicit_objective(x: jax.Array) -> jax.Array:
        value, _gradient = explicit_value_and_grad(x)
        return value

    line_search_results = line_search(
        explicit_value_and_grad if value_and_grad else explicit_objective,
        state.x_k,
        p_k,
        old_fval=state.f_k,
        old_old_fval=state.old_old_fval,
        gfk=state.g_k,
        maxiter=line_search_maxiter,
    )
    line_search_status = _as_jax_dtype(
        line_search_results.status,
        _state_dtype(state.line_search_status),
    )
    next_nfev = state.nfev + line_search_results.nfev
    next_ngev = state.ngev + line_search_results.ngev
    s_k = line_search_results.a_k * p_k
    x_kp1 = state.x_k + s_k
    f_kp1 = line_search_results.f_k
    g_kp1 = line_search_results.g_k
    y_k = g_kp1 - state.g_k
    H_kp1 = _bfgs_inverse_hessian_update(
        state.H_k,
        s_k,
        y_k,
        base_identity_host=base_identity_host,
    )
    xrtol_jax = _as_jax_dtype(xrtol_value, state.x_k.dtype)
    relative_step_converged = (xrtol_jax > _as_jax_dtype(0.0, xrtol_jax.dtype)) & (
        _norm(s_k) <= xrtol_jax * (xrtol_jax + _norm(x_kp1))
    )
    converged = (_norm(g_kp1, ord=norm) < gtol_jax) | relative_step_converged
    next_k = state.k + _int_scalar(1)
    dphi_0 = jnp.real(_dot(state.g_k, p_k))
    dphi_kp1 = jnp.real(_dot(g_kp1, p_k))
    strong_wolfe = (
        jnp.isfinite(f_kp1)
        & jnp.all(jnp.isfinite(g_kp1))
        & (f_kp1 <= state.f_k + wolfe_c1 * line_search_results.a_k * dphi_0)
        & (jnp.abs(dphi_kp1) <= -wolfe_c2 * dphi_0)
    )
    decreasing_fallback = (
        allow_decreasing_fallback & jnp.isfinite(f_kp1) & (f_kp1 < state.f_k)
    )
    wolfe_failure = (~strong_wolfe) & (~decreasing_fallback)
    nonfinite_step = (~jnp.isfinite(f_kp1)) | (~jnp.all(jnp.isfinite(g_kp1)))
    nonfinite_line_search_status = _as_jax_dtype(
        -1,
        _state_dtype(state.line_search_status),
    )
    failure_line_search_status = jnp.where(
        line_search_results.failed,
        line_search_status,
        jnp.where(
            wolfe_failure,
            _as_jax_dtype(0, _state_dtype(state.line_search_status)),
            line_search_status,
        ),
    )

    def nonfinite_step_result(_operand: object) -> _BFGSResults:
        return state._replace(
            converged=_bool_scalar(False),
            failed=_bool_scalar(True),
            k=next_k,
            nfev=next_nfev,
            ngev=next_ngev,
            line_search_status=nonfinite_line_search_status,
        )

    def failed_step(_operand: object) -> _BFGSResults:
        return state._replace(
            converged=_bool_scalar(False),
            failed=_bool_scalar(True),
            k=next_k,
            nfev=next_nfev,
            ngev=next_ngev,
            line_search_status=failure_line_search_status,
        )

    def accepted_step(_operand: object) -> _BFGSResults:
        return state._replace(
            converged=converged,
            nfev=next_nfev,
            ngev=next_ngev,
            k=next_k,
            x_k=x_kp1,
            f_k=f_kp1,
            g_k=g_kp1,
            H_k=H_kp1,
            old_old_fval=state.f_k,
            line_search_status=line_search_status,
        )

    def failed_or_accepted_step(_operand: object) -> _BFGSResults:
        return lax.cond(
            line_search_results.failed | wolfe_failure,
            failed_step,
            accepted_step,
            operand=None,
        )

    return cast(
        _BFGSResults,
        lax.cond(
            nonfinite_step,
            nonfinite_step_result,
            failed_or_accepted_step,
            operand=None,
        ),
    )


def _bfgs_finalize_state(
    state: _BFGSResults,
    explicit_value_and_grad_consts: _BFGSConsts,
    *,
    converted_value_and_grad: _BFGSValueAndGrad,
    norm: float,
    gtol_value: float,
    maxiter_value: jax.Array,
) -> _BFGSResults:
    """Perform the final value/gradient evaluation and assign public status."""

    gtol_jax = _as_jax_dtype(gtol_value, state.g_k.dtype)
    f_final, g_final = converted_value_and_grad(
        state.x_k,
        explicit_value_and_grad_consts,
    )
    converged_final = jnp.logical_not(state.failed) & (
        state.converged | (_norm(g_final, ord=norm) < gtol_jax)
    )
    state = state._replace(
        converged=converged_final,
        nfev=state.nfev + _int_scalar(1),
        ngev=state.ngev + _int_scalar(1),
        f_k=_as_jax_dtype(f_final, state.f_k.dtype),
        g_k=_as_jax_dtype(g_final, state.g_k.dtype),
    )
    failed_status = jnp.where(
        state.line_search_status < _int_scalar(0),
        _int_scalar(2),
        _int_scalar(2) + state.line_search_status,
    )
    status = jnp.where(
        state.status == _int_scalar(BFGS_STATUS_CALLBACK_STOP),
        _int_scalar(BFGS_STATUS_CALLBACK_STOP),
        jnp.where(
            state.converged,
            _int_scalar(0),
            jnp.where(
                state.k == maxiter_value,
                _int_scalar(1),
                jnp.where(state.failed, failed_status, _int_scalar(-1)),
            ),
        ),
    )
    return state._replace(status=status)


def _run_bfgs_eager(
    initial_state: _BFGSResults,
    value_and_grad_consts: _BFGSConsts,
    *,
    fun: Callable[[jax.Array], jax.Array],
    converted_value_and_grad: _BFGSValueAndGrad,
    line_search: _BFGSLineSearch,
    value_and_grad: bool,
    line_search_maxiter: int,
    norm: float,
    gtol_value: float,
    xrtol_value: float,
    base_identity_host: np.ndarray,
    allow_decreasing_fallback: bool,
    maxiter: int,
    callback: _BFGSCallback | None,
    progress_callback: _BFGSProgressCallback | None,
    adapter: object | None,
    memory_analysis_callback: Callable[[dict[str, int]], None] | None,
) -> _BFGSResults:
    """Run BFGS transitions from a host loop while preserving public events."""

    cache_owner = fun if adapter is None else None

    def step_transition(
        state: _BFGSResults,
        consts: _BFGSConsts,
    ) -> _BFGSResults:
        return _bfgs_accepted_step(
            state,
            consts,
            converted_value_and_grad=converted_value_and_grad,
            line_search=line_search,
            value_and_grad=value_and_grad,
            line_search_maxiter=line_search_maxiter,
            norm=norm,
            gtol_value=gtol_value,
            xrtol_value=xrtol_value,
            base_identity_host=base_identity_host,
            allow_decreasing_fallback=allow_decreasing_fallback,
        )

    def finalize_transition(
        state: _BFGSResults,
        consts: _BFGSConsts,
        maxiter_limit: jax.Array,
    ) -> _BFGSResults:
        return _bfgs_finalize_state(
            state,
            consts,
            converted_value_and_grad=converted_value_and_grad,
            norm=norm,
            gtol_value=gtol_value,
            maxiter_value=maxiter_limit,
        )

    step_kernel, finalize_kernel = cast(
        tuple[_BFGSStepKernel, _BFGSFinalizeKernel],
        _cached_private_solver(
            cache_owner,
            cache_key=(
                "bfgs-eager-runtime",
                bool(value_and_grad),
                np.dtype(initial_state.x_k.dtype).str,
                tuple(int(dim) for dim in initial_state.x_k.shape),
                norm,
                int(line_search_maxiter),
                float(gtol_value),
                float(xrtol_value),
            ),
            builder=lambda: (
                jax.jit(step_transition),
                jax.jit(finalize_transition),
            ),
        ),
    )

    memory_analysis_reported = False

    def advance(
        state: _BFGSResults,
        _limits: RuntimeLimits,
    ) -> _BFGSTransition:
        nonlocal memory_analysis_reported
        next_state = step_kernel(state, value_and_grad_consts)
        if memory_analysis_callback is not None and not memory_analysis_reported:
            compiled = cast(
                _CompiledStep,
                step_kernel.lower(state, value_and_grad_consts).compile(),
            )
            memory_analysis_callback(_compiled_step_memory_analysis(compiled))
            memory_analysis_reported = True
        return _BFGSTransition(next_state, jnp.logical_not(next_state.failed))

    observing = callback is not None or progress_callback is not None

    def observe(
        transition: _BFGSTransition,
    ) -> _BFGSObservation | _BFGSObserverObservation:
        state = transition.state
        terminal = (
            state.converged
            | state.failed
            | (state.k >= _as_jax_dtype(maxiter, _state_dtype(state.k)))
        )
        if observing:
            return _BFGSObserverObservation(
                terminal=terminal,
                accepted=transition.accepted,
                iteration=state.k,
                nfev=state.nfev,
                ngev=state.ngev,
                x_k=state.x_k,
                f_k=state.f_k,
                g_k=state.g_k,
            )
        return _BFGSObservation(
            terminal=terminal,
            accepted=transition.accepted,
            iteration=state.k,
            nfev=state.nfev,
            ngev=state.ngev,
        )

    def host_observation(
        observation: _BFGSObservation | _BFGSObserverObservation,
    ) -> _BFGSHostObservation | _BFGSObserverHostObservation:
        if isinstance(observation, _BFGSObserverObservation):
            (
                terminal,
                accepted,
                iteration,
                nfev,
                ngev,
                x_k,
                f_k,
                g_k,
            ) = cast(
                tuple[bool, bool, int, int, int, np.ndarray, float, np.ndarray],
                host_value(observation),
            )
            return _BFGSObserverHostObservation(
                terminal=bool(terminal),
                accepted=bool(accepted),
                iteration=int(iteration),
                nfev=int(nfev),
                ngev=int(ngev),
                x_k=np.asarray(x_k, dtype=float),
                f_k=float(f_k),
                g_k=np.asarray(g_k, dtype=float),
            )
        terminal, accepted, iteration, nfev, ngev = cast(
            tuple[bool, bool, int, int, int],
            host_value(observation),
        )
        return _BFGSHostObservation(
            terminal=bool(terminal),
            accepted=bool(accepted),
            iteration=int(iteration),
            nfev=int(nfev),
            ngev=int(ngev),
        )

    def sink(
        observation: _BFGSHostObservation | _BFGSObserverHostObservation,
        payload: _BFGSResults | None,
    ) -> ContinueDecision:
        del payload
        if not observation.accepted:
            return ContinueDecision.CONTINUE
        if not isinstance(observation, _BFGSObserverHostObservation):
            return ContinueDecision.CONTINUE
        try:
            if callback is not None:
                callback(observation.x_k)
            if progress_callback is not None:
                progress_callback(
                    observation.iteration,
                    observation.f_k,
                    float(np.max(np.abs(observation.g_k))),
                )
        except StopIteration:
            callback_stopped[0] = True
            return ContinueDecision.STOP
        return ContinueDecision.CONTINUE

    callback_stopped = [False]

    ops: StepOps[
        _BFGSResults,
        _BFGSTransition,
        _BFGSObservation | _BFGSObserverObservation,
        _BFGSHostObservation | _BFGSObserverHostObservation,
        _BFGSResults,
        _BFGSResults,
    ] = StepOps(
        initialize=lambda value, _limits: value,
        advance=advance,
        state_of=lambda transition: transition.state,
        observe=observe,
        host_observation=host_observation,
        payload=lambda transition: transition.state,
        terminal=lambda observation: observation.terminal,
        initial_terminal=lambda state, limits: (
            host_bool(state.converged)
            or host_bool(state.failed)
            or not host_bool(jnp.isfinite(state.f_k))
            or not host_bool(jnp.all(jnp.isfinite(state.g_k)))
            or host_int(state.k) >= limits.maxiter
        ),
    )
    state = run_eager(
        ops,
        initial_state,
        RuntimeLimits(maxiter=maxiter),
        sink=sink if callback is not None or progress_callback is not None else None,
    )
    if callback_stopped[0]:
        state = state._replace(
            status=_as_jax_dtype(
                BFGS_STATUS_CALLBACK_STOP,
                _state_dtype(state.status),
            )
        )
    return finalize_kernel(
        state,
        value_and_grad_consts,
        _int_scalar(maxiter),
    )


def _minimize_bfgs_private(
    fun: _BFGSInputObjective,
    x0: jax.typing.ArrayLike,
    *,
    maxiter: int | None = None,
    norm: float = np.inf,
    gtol: float = 1e-5,
    xrtol: float = 0.0,
    line_search_maxiter: int = 10,
    initial_state: _BFGSResults | None = None,
    callback: _BFGSCallback | None = None,
    progress_callback: _BFGSProgressCallback | None = None,
    memory_analysis_callback: Callable[[dict[str, int]], None] | None = None,
    value_and_grad: bool = False,
    x_dtype: jax.typing.DTypeLike | None = None,
) -> _BFGSResults:
    prepared_fun, prepared_x0, prepared_callback, adapter = (
        _prepare_optimizer_callable_inputs(
            fun,
            x0,
            value_and_grad=value_and_grad,
            callback=callback,
        )
    )
    fun = cast(_BFGSInputObjective, prepared_fun)
    x0 = cast(jax.typing.ArrayLike, prepared_x0)
    callback = cast(_BFGSCallback | None, prepared_callback)
    x0 = _require_private_optimizer_runtime(x0, dtype=x_dtype)
    if maxiter is None:
        maxiter = np.size(x0) * 200
    maxiter = int(maxiter)

    d = x0.shape[0]
    cache_owner = fun if adapter is None else None
    value_and_grad_program = _cached_bfgs_value_and_grad_program(
        fun,
        cache_owner=cache_owner,
        value_and_grad=value_and_grad,
        dtype=x0.dtype,
        shape=x0.shape,
    )
    converted_value_and_grad, value_and_grad_consts = closure_converted_value_and_grad(
        value_and_grad_program, x0
    )
    value_and_grad_consts = device_put_closure_consts(value_and_grad_consts, x0)
    scalar_value_and_grad = cast(
        _BFGSValueAndGrad,
        _cached_private_solver(
            cache_owner,
            cache_key=(
                "bfgs-value-and-grad-closure-converted",
                bool(value_and_grad),
                np.dtype(x0.dtype).str,
                tuple(int(dimension) for dimension in x0.shape),
                _closure_constants_signature(value_and_grad_consts),
            ),
            builder=lambda: jax.jit(converted_value_and_grad),
        ),
    )
    line_search: _BFGSLineSearch
    if value_and_grad:
        line_search = _line_search_value_and_grad
    else:
        line_search = _line_search
    gtol_value = np.asarray(gtol, dtype=np.dtype(x0.dtype)).item()
    xrtol_value = np.asarray(xrtol, dtype=np.dtype(x0.dtype)).item()
    half_value = np.asarray(0.5, dtype=np.dtype(x0.dtype)).item()
    wolfe_c1_value = np.asarray(1e-4, dtype=np.dtype(x0.dtype)).item()
    wolfe_c2_value = np.asarray(0.9, dtype=np.dtype(x0.dtype)).item()
    allow_decreasing_fallback = np.dtype(x0.dtype).itemsize < 8
    maxiter_value = np.int32(maxiter)
    base_identity_host = np.eye(d, dtype=np.dtype(x0.dtype))
    if initial_state is None:
        initial_H = _eye(d, x0.dtype)
        f_0, g_0 = scalar_value_and_grad(x0, value_and_grad_consts)
        state = _BFGSResults(
            converged=_norm(g_0, ord=norm) < _as_jax_dtype(gtol_value, x0.dtype),
            failed=_bool_scalar(False),
            k=_int_scalar(0),
            nfev=_int_scalar(1),
            ngev=_int_scalar(1),
            nhev=_int_scalar(0),
            x_k=x0,
            f_k=f_0,
            g_k=g_0,
            H_k=initial_H,
            old_old_fval=f_0 + _norm(g_0) * _as_jax_dtype(half_value, x0.dtype),
            status=_int_scalar(0),
            line_search_status=_int_scalar(0),
        )
    else:
        state = initial_state._replace(
            x_k=_as_jax_dtype(initial_state.x_k, x0.dtype),
            f_k=_as_jax_dtype(initial_state.f_k, x0.dtype),
            g_k=_as_jax_dtype(initial_state.g_k, x0.dtype),
            H_k=_as_jax_dtype(initial_state.H_k, x0.dtype),
            old_old_fval=_as_jax_dtype(initial_state.old_old_fval, x0.dtype),
        )

    def cond_fun(
        carry: tuple[_BFGSResults, _BFGSConsts],
    ) -> jax.Array:
        state, _value_and_grad_consts = carry
        maxiter_jax = _as_jax_dtype(maxiter_value, _state_dtype(state.k))
        return (
            jnp.logical_not(state.converged)
            & jnp.logical_not(state.failed)
            & (state.k < maxiter_jax)
            & jnp.isfinite(state.f_k)
            & jnp.all(jnp.isfinite(state.g_k))
        )

    def body_fun(
        carry: tuple[_BFGSResults, _BFGSConsts],
    ) -> tuple[_BFGSResults, _BFGSConsts]:
        state, explicit_value_and_grad_consts = carry
        gtol_jax = _as_jax_dtype(gtol_value, state.g_k.dtype)
        wolfe_c1 = _as_jax_dtype(wolfe_c1_value, state.f_k.dtype)
        wolfe_c2 = _as_jax_dtype(wolfe_c2_value, state.f_k.dtype)
        p_k = -_dot(state.H_k, state.g_k)

        def explicit_value_and_grad(
            x: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return converted_value_and_grad(x, explicit_value_and_grad_consts)

        def explicit_objective(x: jax.Array) -> jax.Array:
            value, _gradient = explicit_value_and_grad(x)
            return value

        if value_and_grad:
            line_search_results = line_search(
                explicit_value_and_grad,
                state.x_k,
                p_k,
                old_fval=state.f_k,
                old_old_fval=state.old_old_fval,
                gfk=state.g_k,
                maxiter=line_search_maxiter,
            )
        else:
            line_search_results = line_search(
                explicit_objective,
                state.x_k,
                p_k,
                old_fval=state.f_k,
                old_old_fval=state.old_old_fval,
                gfk=state.g_k,
                maxiter=line_search_maxiter,
            )
        line_search_status = _as_jax_dtype(
            line_search_results.status,
            _state_dtype(state.line_search_status),
        )
        next_nfev = state.nfev + line_search_results.nfev
        next_ngev = state.ngev + line_search_results.ngev
        s_k = line_search_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        y_k = g_kp1 - state.g_k
        _rho_k_inv, rho_k, valid_curvature = _bfgs_curvature_terms(
            s_k,
            y_k,
            x_dtype=state.x_k.dtype,
        )

        sy_k = s_k[:, np.newaxis] * y_k[np.newaxis, :]
        identity = _as_jax_dtype(base_identity_host, rho_k.dtype)
        w = identity - rho_k * sy_k
        H_kp1 = (
            _einsum("ij,jk,lk", w, state.H_k, w)
            + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :]
        )
        H_kp1 = jnp.where(valid_curvature, H_kp1, state.H_k)
        xrtol_jax = _as_jax_dtype(xrtol_value, state.x_k.dtype)
        relative_step_converged = (xrtol_jax > _as_jax_dtype(0.0, xrtol_jax.dtype)) & (
            _norm(s_k) <= xrtol_jax * (xrtol_jax + _norm(x_kp1))
        )
        converged = (_norm(g_kp1, ord=norm) < gtol_jax) | relative_step_converged
        next_k = state.k + _int_scalar(1)
        dphi_0 = jnp.real(_dot(state.g_k, p_k))
        dphi_kp1 = jnp.real(_dot(g_kp1, p_k))
        strong_wolfe = (
            jnp.isfinite(f_kp1)
            & jnp.all(jnp.isfinite(g_kp1))
            & (f_kp1 <= state.f_k + wolfe_c1 * line_search_results.a_k * dphi_0)
            & (jnp.abs(dphi_kp1) <= -wolfe_c2 * dphi_0)
        )
        decreasing_fallback = (
            allow_decreasing_fallback & jnp.isfinite(f_kp1) & (f_kp1 < state.f_k)
        )
        wolfe_failure = (~strong_wolfe) & (~decreasing_fallback)
        nonfinite_step = (~jnp.isfinite(f_kp1)) | (~jnp.all(jnp.isfinite(g_kp1)))
        nonfinite_line_search_status = _as_jax_dtype(
            -1,
            _state_dtype(state.line_search_status),
        )
        failure_line_search_status = jnp.where(
            line_search_results.failed,
            line_search_status,
            jnp.where(
                wolfe_failure,
                _as_jax_dtype(0, _state_dtype(state.line_search_status)),
                line_search_status,
            ),
        )

        def nonfinite_step_result(_):
            return state._replace(
                converged=_bool_scalar(False),
                failed=_bool_scalar(True),
                k=next_k,
                nfev=next_nfev,
                ngev=next_ngev,
                line_search_status=nonfinite_line_search_status,
            )

        def failed_step(_):
            return state._replace(
                converged=_bool_scalar(False),
                failed=_bool_scalar(True),
                k=next_k,
                nfev=next_nfev,
                ngev=next_ngev,
                line_search_status=failure_line_search_status,
            )

        def accepted_step(_):
            _emit_iteration_callbacks(
                callback, progress_callback, x_kp1, next_k, f_kp1, g_kp1
            )
            return state._replace(
                converged=converged,
                nfev=next_nfev,
                ngev=next_ngev,
                k=next_k,
                x_k=x_kp1,
                f_k=f_kp1,
                g_k=g_kp1,
                H_k=H_kp1,
                old_old_fval=state.f_k,
                line_search_status=line_search_status,
            )

        next_state = lax.cond(
            nonfinite_step,
            nonfinite_step_result,
            lambda _: lax.cond(
                line_search_results.failed | wolfe_failure,
                failed_step,
                accepted_step,
                operand=None,
            ),
            operand=None,
        )
        return next_state, explicit_value_and_grad_consts

    def run_solver(
        initial_state: _BFGSResults,
        explicit_value_and_grad_consts: _BFGSConsts,
    ) -> _BFGSResults:
        gtol_jax = _as_jax_dtype(gtol_value, initial_state.g_k.dtype)
        maxiter_jax = _as_jax_dtype(maxiter_value, _state_dtype(initial_state.k))
        state, _ = cast(
            tuple[_BFGSResults, _BFGSConsts],
            lax.while_loop(
                cond_fun,
                body_fun,
                (initial_state, explicit_value_and_grad_consts),
            ),
        )
        f_final, g_final = converted_value_and_grad(
            state.x_k,
            explicit_value_and_grad_consts,
        )
        converged_final = jnp.logical_not(state.failed) & (
            state.converged | (_norm(g_final, ord=norm) < gtol_jax)
        )
        state = state._replace(
            converged=converged_final,
            nfev=state.nfev + _int_scalar(1),
            ngev=state.ngev + _int_scalar(1),
            f_k=_as_jax_dtype(f_final, state.f_k.dtype),
            g_k=_as_jax_dtype(g_final, state.g_k.dtype),
        )
        failed_status = jnp.where(
            state.line_search_status < _int_scalar(0),
            _int_scalar(2),
            _int_scalar(2) + state.line_search_status,
        )
        status = jnp.where(
            state.converged,
            _int_scalar(0),
            jnp.where(
                state.k == maxiter_jax,
                _int_scalar(1),
                jnp.where(
                    state.failed,
                    failed_status,
                    _int_scalar(-1),
                ),
            ),
        )
        return state._replace(status=status)

    if not isinstance(x0, jax.core.Tracer):
        return _run_bfgs_eager(
            state,
            value_and_grad_consts,
            fun=fun,
            converted_value_and_grad=converted_value_and_grad,
            line_search=line_search,
            value_and_grad=value_and_grad,
            line_search_maxiter=line_search_maxiter,
            norm=norm,
            gtol_value=gtol_value,
            xrtol_value=xrtol_value,
            base_identity_host=base_identity_host,
            allow_decreasing_fallback=allow_decreasing_fallback,
            maxiter=maxiter,
            callback=callback,
            progress_callback=progress_callback,
            adapter=adapter,
            memory_analysis_callback=memory_analysis_callback,
        )

    run_solver.__name__ = "bfgs_private_run_solver"
    can_cache_solver = (
        adapter is None and callback is None and progress_callback is None
    )
    solver = _cached_private_solver(
        fun if can_cache_solver else None,
        cache_key=(
            "bfgs",
            bool(value_and_grad),
            np.dtype(x0.dtype).str,
            tuple(int(dim) for dim in x0.shape),
            norm,
            int(line_search_maxiter),
            float(gtol_value),
            float(xrtol_value),
            int(maxiter_value),
        ),
        builder=lambda: jax.jit(run_solver),
    )
    return solver(state, value_and_grad_consts)
