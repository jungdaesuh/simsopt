"""SciPy-compatible L-BFGS-B over cached JAX value-and-gradient kernels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.geo.optimizer_host_lbfgs import (
    LINE_SEARCH_FAILURE_REASON_FAILED,
    line_search_failure_reason_to_code,
)
from simsopt_jax.runtime.host_boundary import (
    host_array,
    host_float,
    host_int,
    host_transfer_phase,
    host_value,
)
from simsopt_jax.runtime.jaxpr_closure import (
    closure_converted_value_and_grad,
    device_put_closure_consts,
)

from .._shared import (
    _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
    _prepare_optimizer_callable_inputs,
)
from . import _lbfgsb_scipy as lbfgsb
from ._common import (
    _as_jax_dtype,
    _cached_private_solver,
    _int_scalar,
    _normalize_lbfgs_counter_limit,
    _require_private_optimizer_runtime,
    _scalar_value_and_grad,
)
from ._types import (
    _LBFGSInvalidStepLog,
    _LBFGSResults,
)

_SCIPY_LBFGSB_DEFAULT_MAXITER = 15000
_SCIPY_LBFGSB_DEFAULT_MAXFUN = 15000
_DEFAULT_OPTIMIZER_STATE_TRACE_MAX_BYTES = 64 * 1024 * 1024
_TRACE_ARRAYS_PER_ENTRY = 2
_TRACE_SCALARS_PER_ENTRY = 5
_LBFGS_RUN_MODE_STEPWISE = "stepwise"
_LBFGS_RUN_MODE_MONOLITHIC_DEBUG = "monolithic_debug"
_LBFGSB_PRIVATE_BOUNDS = None
_LBFGS_STEP_ENTRY_START = "start"
_LBFGS_STEP_ENTRY_SEARCH = "search"
_LBFGS_STEP_ENTRY_NEW_X_REENTRY = "new_x_reentry"
_LBFGS_STEP_ENTRY_KIND_BY_CODE = {
    lbfgsb.LBFGSB_STEP_ENTRY_START: _LBFGS_STEP_ENTRY_START,
    lbfgsb.LBFGSB_STEP_ENTRY_SEARCH: _LBFGS_STEP_ENTRY_SEARCH,
    lbfgsb.LBFGSB_STEP_ENTRY_NEW_X_REENTRY: _LBFGS_STEP_ENTRY_NEW_X_REENTRY,
}


class _LbfgsbStepwiseHostStatus(NamedTuple):
    terminal: bool
    entry_kind: str
    accepted_new_x: bool
    x: np.ndarray | None = None
    f: float | None = None
    g: np.ndarray | None = None
    n_iterations: int | None = None
    nfev: int | None = None
    njev: int | None = None


_LBFGSConsts = tuple[jax.Array, ...]
_LBFGSValueAndGrad = Callable[
    [jax.Array, _LBFGSConsts],
    tuple[jax.Array, jax.Array],
]
_LBFGSInitialState = Callable[[jax.Array], lbfgsb.LbfgsbState]
_LBFGSAdvance = Callable[
    [
        lbfgsb.LbfgsbState,
        _LBFGSConsts,
        jax.Array,
        jax.Array,
    ],
    lbfgsb.LbfgsbMacroStepResult,
]
_LBFGSResultPayload = Callable[
    [lbfgsb.LbfgsbState, jax.Array, jax.Array],
    _LBFGSResults,
]
_LBFGSInitialValueAndGrad = tuple[jax.Array, jax.Array]
_LBFGSObserver = Callable[[_LbfgsbStepwiseHostStatus], object]
_LBFGSCallback = Callable[[np.ndarray], object]
_LBFGSProgressCallback = Callable[[int, float, float], object]
_LBFGSDiagnosticCallback = Callable[..., object]


def _lbfgsb_unconstrained_fast_path_enabled(bounds: object | None) -> bool:
    return bounds is None


# `lbfgs-ondevice` keeps the SciPy-compatible L-BFGS-B contract; only its
# execution boundary mirrors Optax-style init/update/state/result structure.
def _record_diagnostic_event(
    callback: Callable[..., object] | None,
    label: str,
    **fields: object,
) -> None:
    if callback is not None:
        callback(label, **fields)


def _closure_converted_lbfgs_value_and_grad(
    value_and_grad_fun: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    example_x: jax.Array,
) -> tuple[_LBFGSValueAndGrad, _LBFGSConsts]:
    converted, consts = closure_converted_value_and_grad(value_and_grad_fun, example_x)
    return converted, consts


def _call_lbfgs_value_and_grad(
    value_and_grad: _LBFGSValueAndGrad
    | Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x: jax.Array,
    value_and_grad_consts: _LBFGSConsts | None,
) -> tuple[jax.Array, jax.Array]:
    if value_and_grad_consts is None:
        return value_and_grad(x)
    return value_and_grad(x, value_and_grad_consts)


def _device_put_lbfgs_consts(
    value_and_grad_consts: _LBFGSConsts,
    example_x: jax.Array,
) -> _LBFGSConsts:
    return device_put_closure_consts(value_and_grad_consts, example_x)


def _cached_lbfgs_value_and_grad_kernel(
    value_and_grad_fun: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    *,
    cache_owner: object | None,
    adapter: object | None,
    objective_mode: str,
    dtype: jax.typing.DTypeLike,
    example_x: jax.Array,
) -> tuple[_LBFGSValueAndGrad, _LBFGSConsts]:
    converted_value_and_grad, value_and_grad_consts = (
        _closure_converted_lbfgs_value_and_grad(value_and_grad_fun, example_x)
    )
    cache_owner, cache_key_prefix = _lbfgsb_cache_context(
        cache_owner,
        adapter,
        objective_mode,
        dtype,
        example_x.shape,
    )

    def build_kernel():
        def lbfgs_private_value_and_grad(
            x: jax.Array,
            consts: _LBFGSConsts,
        ) -> tuple[jax.Array, jax.Array]:
            value, grad = converted_value_and_grad(x, consts)
            return (
                jnp.asarray(value, dtype=x.dtype),
                jnp.asarray(grad, dtype=x.dtype),
            )

        lbfgs_private_value_and_grad.__name__ = "lbfgs_private_value_and_grad"
        return jax.jit(lbfgs_private_value_and_grad)

    value_and_grad_kernel = _cached_private_solver(
        cache_owner,
        cache_key=("lbfgs-value-and-grad-closure-converted", *cache_key_prefix),
        builder=build_kernel,
    )
    value_and_grad_consts = _device_put_lbfgs_consts(value_and_grad_consts, example_x)
    return value_and_grad_kernel, value_and_grad_consts


def _lbfgsb_cache_context(
    cache_owner: object | None,
    adapter: object | None,
    objective_mode: str,
    dtype: jax.typing.DTypeLike,
    shape: tuple[int, ...],
) -> tuple[object | None, tuple[object, ...]]:
    structured_solver_cache_token = None
    if adapter is not None and cache_owner is not None:
        structured_solver_cache_token = getattr(
            cache_owner,
            _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
            None,
        )
    adapter_cache_key = None if adapter is None else adapter.solver_cache_key()
    can_cache_kernel = cache_owner is not None and (
        adapter is None or structured_solver_cache_token is not None
    )
    return (
        cache_owner if can_cache_kernel else None,
        (
            str(objective_mode),
            structured_solver_cache_token,
            adapter_cache_key,
            np.dtype(dtype).str,
            tuple(int(dim) for dim in shape),
        ),
    )


def _lbfgsb_initial_state_kernel(
    *,
    cache_owner: object | None = None,
    cache_key_prefix: tuple[object, ...] = (),
    m: int,
    ftol: float,
    gtol: float,
    maxls: int,
):
    def build(x0: jax.Array) -> lbfgsb.LbfgsbState:
        return lbfgsb.lbfgsb_initial_state(
            x0,
            m=m,
            bounds=_LBFGSB_PRIVATE_BOUNDS,
            ftol=ftol,
            gtol=gtol,
            maxls=maxls,
        )

    build.__name__ = "lbfgs_private_initial_state_solver"
    return _cached_private_solver(
        cache_owner,
        cache_key=(
            "lbfgsb-initial-state",
            *cache_key_prefix,
            int(m),
            float(ftol),
            float(gtol),
            int(maxls),
        ),
        builder=lambda: jax.jit(build),
    )


def _lbfgsb_mainlb_kernel(
    value_and_grad: _LBFGSValueAndGrad,
    *,
    cache_owner: object | None = None,
    cache_key_prefix: tuple[object, ...] = (),
    maxiter: int,
    maxfun: int,
    accepted_step_callback: _LBFGSObserver | None = None,
) -> Callable[[lbfgsb.LbfgsbState, _LBFGSConsts | None], _LBFGSResults]:
    """Legacy full-solve compile wrapper kept only for explicit debug runs."""

    def run(
        state: lbfgsb.LbfgsbState,
        value_and_grad_consts: _LBFGSConsts | None = None,
    ) -> _LBFGSResults:
        def value_and_grad_with_consts(x):
            return _call_lbfgs_value_and_grad(
                value_and_grad,
                x,
                value_and_grad_consts,
            )

        final_state = lbfgsb.lbfgsb_mainlb(
            value_and_grad_with_consts,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=accepted_step_callback,
        )
        history = lbfgsb.lbfgsb_inverse_hessian_history(final_state)
        return _lbfgsb_state_to_lbfgs_results(
            final_state,
            history=history,
            maxiter_limit=_int_scalar(maxiter),
            maxfun_limit=_int_scalar(maxfun),
        )

    run.__name__ = "lbfgs_private_monolithic_mainlb_solver"
    return _cached_private_solver(
        cache_owner if accepted_step_callback is None else None,
        cache_key=(
            "lbfgsb-mainlb",
            *cache_key_prefix,
            int(maxiter),
            int(maxfun),
        ),
        builder=lambda: jax.jit(run),
    )


def _lbfgsb_advance_to_next_observable_kernel(
    value_and_grad: _LBFGSValueAndGrad,
    *,
    cache_owner: object | None = None,
    cache_key_prefix: tuple[object, ...] = (),
    maxiter: int | None = None,
    maxfun: int | None = None,
    accepted_step_callback: _LBFGSObserver | None = None,
    entry_kind: str = _LBFGS_STEP_ENTRY_SEARCH,
    unconstrained_fast_path: bool = False,
    dynamic_limits: bool = False,
) -> Callable[..., lbfgsb.LbfgsbMacroStepResult]:
    if not dynamic_limits and (maxiter is None or maxfun is None):
        raise ValueError("static L-BFGS step kernels require maxiter and maxfun")

    def run_with_limits(
        state: lbfgsb.LbfgsbState,
        value_and_grad_consts: _LBFGSConsts | None,
        maxiter_limit: jax.Array,
        maxfun_limit: jax.Array,
    ) -> lbfgsb.LbfgsbMacroStepResult:
        def value_and_grad_with_consts(x):
            return _call_lbfgs_value_and_grad(
                value_and_grad,
                x,
                value_and_grad_consts,
            )

        if entry_kind == _LBFGS_STEP_ENTRY_START:
            return lbfgsb.lbfgsb_advance_from_start_to_next_observable(
                value_and_grad_with_consts,
                state,
                maxiter=maxiter_limit,
                maxfun=maxfun_limit,
                accepted_step_callback=accepted_step_callback,
                unconstrained_fast_path=unconstrained_fast_path,
            )
        if entry_kind == _LBFGS_STEP_ENTRY_NEW_X_REENTRY:
            return lbfgsb.lbfgsb_reenter_new_x(
                value_and_grad_with_consts,
                state,
                maxiter=maxiter_limit,
                maxfun=maxfun_limit,
                accepted_step_callback=accepted_step_callback,
                unconstrained_fast_path=unconstrained_fast_path,
            )
        return lbfgsb.lbfgsb_advance_from_search_to_next_observable(
            value_and_grad_with_consts,
            state,
            maxiter=maxiter_limit,
            maxfun=maxfun_limit,
            accepted_step_callback=accepted_step_callback,
            unconstrained_fast_path=unconstrained_fast_path,
        )

    if dynamic_limits:

        def run(
            state: lbfgsb.LbfgsbState,
            value_and_grad_consts: _LBFGSConsts,
            maxiter_limit: jax.Array,
            maxfun_limit: jax.Array,
        ) -> lbfgsb.LbfgsbMacroStepResult:
            return run_with_limits(
                state,
                value_and_grad_consts,
                maxiter_limit,
                maxfun_limit,
            )

    else:

        def run(
            state: lbfgsb.LbfgsbState,
            value_and_grad_consts: _LBFGSConsts | None = None,
        ) -> lbfgsb.LbfgsbMacroStepResult:
            return run_with_limits(
                state,
                value_and_grad_consts,
                _int_scalar(maxiter),
                _int_scalar(maxfun),
            )

    run.__name__ = "lbfgs_private_macro_step_solver"
    budget_key = ("dynamic",) if dynamic_limits else (int(maxiter), int(maxfun))
    return _cached_private_solver(
        cache_owner if accepted_step_callback is None else None,
        cache_key=(
            "lbfgsb-advance-to-next-observable",
            *cache_key_prefix,
            entry_kind,
            *budget_key,
            bool(unconstrained_fast_path),
        ),
        builder=lambda: jax.jit(run),
    )


def _lbfgsb_result_payload_kernel(
    *,
    cache_owner: object | None = None,
    cache_key_prefix: tuple[object, ...] = (),
    maxiter: int | None = None,
    maxfun: int | None = None,
) -> Callable[[lbfgsb.LbfgsbState, jax.Array, jax.Array], _LBFGSResults]:
    del maxiter, maxfun

    def run(
        state: lbfgsb.LbfgsbState,
        maxiter: jax.Array,
        maxfun: jax.Array,
    ) -> _LBFGSResults:
        history = lbfgsb.lbfgsb_inverse_hessian_history(state)
        return _lbfgsb_state_to_lbfgs_results(
            state,
            history=history,
            maxiter_limit=maxiter,
            maxfun_limit=maxfun,
        )

    run.__name__ = "lbfgs_private_result_payload_solver"
    return _cached_private_solver(
        cache_owner,
        cache_key=(
            "lbfgsb-result-payload",
            *cache_key_prefix,
        ),
        builder=lambda: jax.jit(run),
    )


def _lbfgsb_callback_stop_state_kernel(
    *,
    cache_owner: object | None = None,
    cache_key_prefix: tuple[object, ...] = (),
) -> Callable[[lbfgsb.LbfgsbState], lbfgsb.LbfgsbState]:
    def stop_for_callback(state: lbfgsb.LbfgsbState) -> lbfgsb.LbfgsbState:
        workspace = state.workspace._replace(
            task=lbfgsb._lbfgsb_task(lbfgsb.STOP, lbfgsb.STOP_CALLB)
        )
        return state._replace(workspace=workspace)

    return _cached_private_solver(
        cache_owner,
        cache_key=("lbfgsb-callback-stop-state", *cache_key_prefix),
        builder=lambda: jax.jit(stop_for_callback),
    )


def _lbfgsb_start_with_initial_value_and_grad_kernel(
    *,
    cache_owner: object | None = None,
    cache_key_prefix: tuple[object, ...] = (),
) -> Callable[[lbfgsb.LbfgsbState, jax.Array, jax.Array], lbfgsb.LbfgsbState]:
    def start_with_initial_value_and_grad(
        state: lbfgsb.LbfgsbState,
        value: jax.Array,
        grad: jax.Array,
    ) -> lbfgsb.LbfgsbState:
        return lbfgsb.lbfgsb_start_with_initial_value_and_grad(
            state,
            value,
            grad,
        )

    return _cached_private_solver(
        cache_owner,
        cache_key=(
            "lbfgsb-start-with-initial-value-and-grad",
            *cache_key_prefix,
        ),
        builder=lambda: jax.jit(start_with_initial_value_and_grad),
    )


def _resolve_scipy_lbfgsb_limits(
    maxiter: int | None,
    maxfun: int | None,
) -> tuple[np.int32, np.int32]:
    maxiter_limit = _normalize_lbfgs_counter_limit(
        _SCIPY_LBFGSB_DEFAULT_MAXITER if maxiter is None else maxiter
    )
    maxfun_limit = _normalize_lbfgs_counter_limit(
        _SCIPY_LBFGSB_DEFAULT_MAXFUN if maxfun is None else maxfun
    )
    return maxiter_limit, maxfun_limit


def _check_lbfgsb_run_mode(run_mode: str) -> str:
    if run_mode not in {
        _LBFGS_RUN_MODE_STEPWISE,
        _LBFGS_RUN_MODE_MONOLITHIC_DEBUG,
    }:
        raise ValueError(
            "lbfgs_run_mode must be 'stepwise' or 'monolithic_debug', "
            f"got {run_mode!r}."
        )
    return run_mode


def _resolve_lbfgsb_run_mode_for_runtime(
    run_mode: str,
    x0: jax.Array,
    *,
    callback: _LBFGSCallback | None,
    progress_callback: _LBFGSProgressCallback | None,
    record_optimizer_state_trace: bool,
) -> str:
    if not isinstance(x0, jax.core.Tracer):
        return run_mode
    if (
        callback is not None
        or progress_callback is not None
        or record_optimizer_state_trace
    ):
        raise ValueError(
            "lbfgs-ondevice host observation is not available while "
            "tracing a JAX function."
        )
    if run_mode == _LBFGS_RUN_MODE_STEPWISE:
        raise ValueError(
            "lbfgs-ondevice stepwise mode uses host-observed macro steps and "
            "cannot run while tracing a JAX function. Pass "
            "lbfgs_run_mode='monolithic_debug' explicitly for the full-solve "
            "debug path."
        )
    return run_mode


def _lbfgsb_workspace_bytes(
    n: int,
    maxcor: int,
    dtype: jax.typing.DTypeLike,
    *,
    record_optimizer_state_trace: bool = False,
    maxiter_limit: int | np.integer | jax.Array | None = None,
) -> int:
    float_slots = lbfgsb.lbfgsb_workspace_size(int(n), int(maxcor)) + 29
    int_slots = lbfgsb.lbfgsb_iwa_size(int(n)) + 2 + 2 + 4 + 44
    workspace_bytes = (
        float_slots * np.dtype(dtype).itemsize + int_slots * np.dtype(np.int32).itemsize
    )
    if record_optimizer_state_trace:
        iterations = max(1, int(maxiter_limit))
        workspace_bytes += iterations * (
            (_TRACE_ARRAYS_PER_ENTRY * int(n) + _TRACE_SCALARS_PER_ENTRY)
            * np.dtype(np.float64).itemsize
        )
    return int(workspace_bytes)


def _check_lbfgsb_trace_budget(
    d: int,
    maxiter_limit: int | np.integer | jax.Array,
    *,
    record_optimizer_state_trace: bool,
    max_optimizer_state_trace_bytes: int | None,
) -> None:
    if not record_optimizer_state_trace:
        return
    iterations = max(1, int(maxiter_limit))
    trace_bytes = iterations * (
        (_TRACE_ARRAYS_PER_ENTRY * int(d) + _TRACE_SCALARS_PER_ENTRY)
        * np.dtype(np.float64).itemsize
    )
    limit = (
        _DEFAULT_OPTIMIZER_STATE_TRACE_MAX_BYTES
        if max_optimizer_state_trace_bytes is None
        else int(max_optimizer_state_trace_bytes)
    )
    if trace_bytes > limit:
        raise ValueError(
            "optimizer_state_trace would allocate "
            f"{trace_bytes} bytes for d={int(d)} and iterations={iterations}, "
            f"exceeding max_optimizer_state_trace_bytes={limit}."
        )


def _resolve_lbfgs_history_size(
    maxcor: int,
    *,
    maxiter_limit: int | np.integer | jax.Array,
) -> int:
    del maxiter_limit
    return max(1, int(maxcor))


def _lbfgsb_public_status(
    state: lbfgsb.LbfgsbState,
    *,
    maxiter_limit: int | np.integer | jax.Array,
    maxfun_limit: int | np.integer | jax.Array,
) -> jax.Array:
    return lbfgsb.lbfgsb_public_status_from_state(
        state,
        maxiter=maxiter_limit,
        maxfun=maxfun_limit,
    )


def _lbfgsb_rho_history(history: lbfgsb.LbfgsbInverseHessianHistory) -> jax.Array:
    return jnp.zeros((history.s.shape[0],), dtype=history.s.dtype)


def _lbfgsb_invalid_step_log(state: lbfgsb.LbfgsbState) -> _LBFGSInvalidStepLog:
    int_zero = state.nfev - state.nfev
    int_slot = jnp.zeros_like(state.workspace.isave[:1])
    float_slot = jnp.zeros_like(state.workspace.dsave[:1])
    abnormal = state.workspace.task[0] == lbfgsb.ABNORMAL
    abnormal_slot = jnp.reshape(abnormal, (1,))
    false_slot = jnp.zeros_like(int_slot, dtype=bool)
    true_slot = jnp.ones_like(int_slot, dtype=bool)
    failed_reason = jnp.asarray(
        line_search_failure_reason_to_code(LINE_SEARCH_FAILURE_REASON_FAILED),
        dtype=state.workspace.isave.dtype,
    )
    step = jnp.where(abnormal, state.workspace.dsave[13], state.workspace.dsave[13] * 0)
    step_slot = jnp.reshape(step, (1,))
    nonfinite = (~jnp.isfinite(state.f)) | jnp.any(~jnp.isfinite(state.g))
    nonfinite_slot = jnp.reshape(abnormal & nonfinite, (1,))
    failure_reason_slot = jnp.where(abnormal_slot, failed_reason, int_slot)
    return _LBFGSInvalidStepLog(
        count=jnp.where(abnormal, int_zero + 1, int_zero),
        write_index=int_zero,
        iteration=jnp.where(
            abnormal_slot, jnp.reshape(state.n_iterations, (1,)), int_slot
        ),
        step_scale=step_slot,
        line_search_failed=abnormal_slot,
        nonfinite_step=nonfinite_slot,
        stalled_step=false_slot,
        valid_curvature=true_slot,
        trial_converged=false_slot,
        ls_status=jnp.where(abnormal_slot, state.workspace.isave[34:35], int_slot),
        requested_initial_step=step_slot,
        first_tested_alpha=step_slot,
        best_finite_alpha=float_slot,
        returned_alpha=float_slot,
        failure_reason=failure_reason_slot,
        armijo_margin=float_slot,
        curvature_margin=float_slot,
    )


def _lbfgsb_state_with_initial_value_and_grad(
    state: lbfgsb.LbfgsbState,
    initial_value_and_grad: _LBFGSInitialValueAndGrad,
    *,
    dtype: jax.typing.DTypeLike,
    cache_owner: object | None = None,
    cache_key_prefix: tuple[object, ...] = (),
) -> lbfgsb.LbfgsbState:
    value, grad = initial_value_and_grad
    value = _as_jax_dtype(value, dtype)
    grad = _as_jax_dtype(grad, dtype)
    if grad.shape != state.x.shape:
        raise ValueError(
            "initial_value_and_grad must provide a gradient matching "
            f"x.shape={state.x.shape}, got {grad.shape}."
        )
    return _lbfgsb_start_with_initial_value_and_grad_kernel(
        cache_owner=cache_owner,
        cache_key_prefix=cache_key_prefix,
    )(state, value, grad)


def _lbfgsb_accepted_step_observer(
    *,
    callback: _LBFGSCallback | None,
    progress_callback: _LBFGSProgressCallback | None,
    optimizer_state_trace: list[dict[str, object]],
    record_optimizer_state_trace: bool,
) -> _LBFGSObserver | None:
    if (
        callback is None
        and progress_callback is None
        and not record_optimizer_state_trace
    ):
        return None

    def observe(*args):
        if len(args) == 1 and isinstance(args[0], _LbfgsbStepwiseHostStatus):
            status = args[0]
            x_host = status.x
            g_host = status.g
            f_host = status.f
            iteration_host = status.n_iterations
            nfev_host = status.nfev
            njev_host = status.njev
            if (
                x_host is None
                or g_host is None
                or f_host is None
                or iteration_host is None
                or nfev_host is None
                or njev_host is None
            ):
                raise RuntimeError("incomplete packed L-BFGS observer status")
        else:
            iteration, x, f, g, nfev, njev = args
            x_host = host_array(x, dtype=float)
            g_host = host_array(g, dtype=float)
            f_host = host_float(f)
            iteration_host = host_int(iteration)
            nfev_host = host_int(nfev)
            njev_host = host_int(njev)
        if callback is not None:
            callback(x_host)
        if progress_callback is not None:
            progress_callback(
                iteration_host,
                f_host,
                float(np.linalg.norm(g_host, ord=np.inf)),
            )
        if record_optimizer_state_trace:
            optimizer_state_trace.append(
                {
                    "iteration": iteration_host,
                    "x": x_host,
                    "fun": f_host,
                    "jac": g_host,
                    "jac_inf_norm": float(np.linalg.norm(g_host, ord=np.inf)),
                    "nfev": nfev_host,
                    "njev": njev_host,
                }
            )

    return observe


def _lbfgsb_state_to_lbfgs_results(
    state: lbfgsb.LbfgsbState,
    *,
    history: lbfgsb.LbfgsbInverseHessianHistory,
    maxiter_limit: int | np.integer | jax.Array,
    maxfun_limit: int | np.integer | jax.Array,
    optimizer_state_trace: tuple[dict[str, object], ...] = (),
) -> _LBFGSResults:
    status = _lbfgsb_public_status(
        state,
        maxiter_limit=maxiter_limit,
        maxfun_limit=maxfun_limit,
    )
    return _LBFGSResults(
        converged=status == jnp.zeros_like(status),
        failed=status != jnp.zeros_like(status),
        k=state.n_iterations,
        nfev=state.nfev,
        ngev=state.njev,
        x_k=state.x,
        f_k=state.f,
        g_k=state.g,
        s_history=history.s,
        y_history=history.y,
        rho_history=_lbfgsb_rho_history(history),
        gamma=state.workspace.dsave[0],
        status=status,
        ls_status=state.workspace.isave[34],
        invalid_step_log=_lbfgsb_invalid_step_log(state),
        optimizer_state_trace=tuple(optimizer_state_trace),
        hess_inv_s=history.s,
        hess_inv_y=history.y,
        hess_inv_n_corrs=history.n_corrs,
        task=state.workspace.task,
    )


def _lbfgsb_stepwise_host_status(
    step_result: lbfgsb.LbfgsbMacroStepResult,
    *,
    include_observation: bool = False,
) -> _LbfgsbStepwiseHostStatus:
    if include_observation:
        (
            terminal,
            entry_code,
            accepted_new_x,
            x,
            f,
            g,
            n_iterations,
            nfev,
            njev,
        ) = host_value(
            (
                step_result.terminal,
                step_result.entry_kind,
                step_result.accepted_new_x,
                step_result.state.x,
                step_result.state.f,
                step_result.state.g,
                step_result.state.n_iterations,
                step_result.state.nfev,
                step_result.state.njev,
            )
        )
    else:
        terminal, entry_code, accepted_new_x = host_value(
            (
                step_result.terminal,
                step_result.entry_kind,
                step_result.accepted_new_x,
            )
        )
        x = f = g = n_iterations = nfev = njev = None
    entry_kind = _LBFGS_STEP_ENTRY_KIND_BY_CODE[int(entry_code)]
    return _LbfgsbStepwiseHostStatus(
        terminal=bool(terminal),
        entry_kind=entry_kind,
        accepted_new_x=bool(accepted_new_x),
        x=None if x is None else host_array(x, dtype=float),
        f=None if f is None else float(f),
        g=None if g is None else host_array(g, dtype=float),
        n_iterations=None if n_iterations is None else int(n_iterations),
        nfev=None if nfev is None else int(nfev),
        njev=None if njev is None else int(njev),
    )


def _lbfgsb_stepwise_driver(
    state: lbfgsb.LbfgsbState,
    value_and_grad_consts: _LBFGSConsts,
    advance_from_start: Callable[..., lbfgsb.LbfgsbMacroStepResult],
    advance_from_search: Callable[..., lbfgsb.LbfgsbMacroStepResult],
    reenter_new_x: Callable[..., lbfgsb.LbfgsbMacroStepResult],
    result_payload: Callable[..., _LBFGSResults],
    *,
    initial_entry_kind: str,
    maxiter: int,
    maxfun: int,
    accepted_step_callback: _LBFGSObserver | None = None,
    callback_stop_state: Callable[[lbfgsb.LbfgsbState], lbfgsb.LbfgsbState]
    | None = None,
) -> _LBFGSResults:
    host_status = _LbfgsbStepwiseHostStatus(
        terminal=False,
        entry_kind=initial_entry_kind,
        accepted_new_x=False,
    )
    maxiter_scalar = _int_scalar(maxiter)
    maxfun_scalar = _int_scalar(maxfun)
    while not host_status.terminal:
        if host_status.entry_kind == _LBFGS_STEP_ENTRY_START:
            step_result = advance_from_start(
                state,
                value_and_grad_consts,
                maxiter_scalar,
                maxfun_scalar,
            )
        elif host_status.entry_kind == _LBFGS_STEP_ENTRY_NEW_X_REENTRY:
            step_result = reenter_new_x(
                state,
                value_and_grad_consts,
                maxiter_scalar,
                maxfun_scalar,
            )
        else:
            step_result = advance_from_search(
                state,
                value_and_grad_consts,
                maxiter_scalar,
                maxfun_scalar,
            )
        state = step_result.state
        transfer_phase = "callback" if accepted_step_callback is not None else "advance"
        with host_transfer_phase(transfer_phase):
            step_host_status = _lbfgsb_stepwise_host_status(
                step_result,
                include_observation=accepted_step_callback is not None,
            )
            if accepted_step_callback is not None and step_host_status.accepted_new_x:
                try:
                    accepted_step_callback(step_host_status)
                except StopIteration:
                    if callback_stop_state is None:
                        raise
                    state = callback_stop_state(state)
                    break
        host_status = step_host_status
    return result_payload(
        state,
        _int_scalar(maxiter),
        _int_scalar(maxfun),
    )


class _LoweredPreparedProgram(Protocol):
    def compile(self) -> object: ...


class _JittedPreparedProgram(Protocol):
    def lower(self, *args: object) -> _LoweredPreparedProgram: ...


def _compile_prepared_program(program: object, *args: object) -> object:
    """Compile one fixed-shape optimizer program for a prepared run."""

    return cast(_JittedPreparedProgram, program).lower(*args).compile()


@dataclass(frozen=True)
class PreparedLBFGS:
    """Compiled fixed-shape L-BFGS programs with dynamic run budgets.

    The object follows the same init/update/result separation as Optax while
    retaining the in-tree SciPy-compatible L-BFGS-B state machine.  Compilation
    happens in :func:`prepare_lbfgs_private`; :meth:`run` only creates the
    initial state and drives already-compiled macro-step transitions.
    """

    initial_state: _LBFGSInitialState
    value_and_grad: _LBFGSValueAndGrad
    value_and_grad_consts: _LBFGSConsts
    advance_from_start: _LBFGSAdvance
    advance_from_search: _LBFGSAdvance
    reenter_new_x: _LBFGSAdvance
    result_payload: _LBFGSResultPayload
    initial_value: jax.Array
    initial_gradient: jax.Array
    history_size: int

    def run(
        self,
        x0: jax.Array,
        *,
        maxiter: int,
        maxfun: int | None = None,
    ) -> _LBFGSResults:
        maxiter_limit, maxfun_limit = _resolve_scipy_lbfgsb_limits(
            maxiter,
            maxfun,
        )
        state = self.initial_state(x0)
        return _lbfgsb_stepwise_driver(
            state,
            self.value_and_grad_consts,
            self.advance_from_start,
            self.advance_from_search,
            self.reenter_new_x,
            self.result_payload,
            initial_entry_kind=_LBFGS_STEP_ENTRY_START,
            maxiter=int(maxiter_limit),
            maxfun=int(maxfun_limit),
        )


def prepare_lbfgs_private(
    fun: Callable[[jax.Array], jax.Array],
    x0: jax.typing.ArrayLike,
    *,
    cache_owner: object | None = None,
    maxcor: int = 10,
    ftol: float = 0.0,
    gtol: float = 1e-5,
    maxls: int = 20,
    x_dtype: jax.typing.DTypeLike | None = None,
) -> PreparedLBFGS:
    """Prepare a scalar L-BFGS program for repeated fixed-shape runs.

    ``maxiter`` and ``maxfun`` remain dynamic inputs to the compiled macro
    steps.  This lets callers measure cold compilation once and then compare
    warm solver execution without rebuilding the state machine.
    """

    value_and_grad_fun = _scalar_value_and_grad(fun)
    prepared_fun, prepared_x0, _callback, adapter = _prepare_optimizer_callable_inputs(
        value_and_grad_fun,
        x0,
        value_and_grad=True,
        callback=None,
    )
    prepared_value_and_grad = cast(
        Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        prepared_fun,
    )
    params = _require_private_optimizer_runtime(
        cast(jax.typing.ArrayLike, prepared_x0),
        dtype=x_dtype,
    )
    value_and_grad_kernel, value_and_grad_consts = _cached_lbfgs_value_and_grad_kernel(
        prepared_value_and_grad,
        cache_owner=cache_owner,
        adapter=adapter,
        objective_mode="scalar",
        dtype=params.dtype,
        example_x=params,
    )
    solver_cache_owner, solver_cache_key_prefix = _lbfgsb_cache_context(
        cache_owner,
        adapter,
        "scalar",
        params.dtype,
        params.shape,
    )
    maxiter_limit, maxfun_limit = _resolve_scipy_lbfgsb_limits(1, None)
    history_size = _resolve_lbfgs_history_size(
        maxcor,
        maxiter_limit=maxiter_limit,
    )
    initial_state_kernel = _lbfgsb_initial_state_kernel(
        cache_owner=solver_cache_owner,
        cache_key_prefix=solver_cache_key_prefix,
        m=history_size,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
    )
    initial_state = cast(
        _LBFGSInitialState,
        _compile_prepared_program(initial_state_kernel, params),
    )
    sample_state = initial_state(params)
    value_and_grad = cast(
        _LBFGSValueAndGrad,
        _compile_prepared_program(
            value_and_grad_kernel,
            params,
            value_and_grad_consts,
        ),
    )
    initial_value, initial_gradient = value_and_grad(
        params,
        value_and_grad_consts,
    )
    unconstrained_fast_path = _lbfgsb_unconstrained_fast_path_enabled(
        _LBFGSB_PRIVATE_BOUNDS
    )
    compiled_transitions: list[_LBFGSAdvance] = []
    for entry_kind in (
        _LBFGS_STEP_ENTRY_START,
        _LBFGS_STEP_ENTRY_SEARCH,
        _LBFGS_STEP_ENTRY_NEW_X_REENTRY,
    ):
        transition = _lbfgsb_advance_to_next_observable_kernel(
            value_and_grad_kernel,
            cache_owner=solver_cache_owner,
            cache_key_prefix=solver_cache_key_prefix,
            accepted_step_callback=None,
            entry_kind=entry_kind,
            unconstrained_fast_path=unconstrained_fast_path,
            dynamic_limits=True,
        )
        compiled_transitions.append(
            cast(
                _LBFGSAdvance,
                _compile_prepared_program(
                    transition,
                    sample_state,
                    value_and_grad_consts,
                    _int_scalar(maxiter_limit),
                    _int_scalar(maxfun_limit),
                ),
            )
        )
    result_payload_kernel = _lbfgsb_result_payload_kernel(
        cache_owner=solver_cache_owner,
        cache_key_prefix=solver_cache_key_prefix,
    )
    result_payload = cast(
        _LBFGSResultPayload,
        _compile_prepared_program(
            result_payload_kernel,
            sample_state,
            _int_scalar(maxiter_limit),
            _int_scalar(maxfun_limit),
        ),
    )
    return PreparedLBFGS(
        initial_state=initial_state,
        value_and_grad=value_and_grad,
        value_and_grad_consts=value_and_grad_consts,
        advance_from_start=compiled_transitions[0],
        advance_from_search=compiled_transitions[1],
        reenter_new_x=compiled_transitions[2],
        result_payload=result_payload,
        initial_value=initial_value,
        initial_gradient=initial_gradient,
        history_size=history_size,
    )


def _minimize_lbfgs_private_impl(
    value_and_grad_fun: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x0: jax.typing.ArrayLike,
    *,
    cache_owner: object | None = None,
    objective_mode: str,
    maxiter: int | None = None,
    maxcor: int = 10,
    ftol: float = 0.0,
    gtol: float = 1e-5,
    maxfun: int | None = None,
    maxls: int = 20,
    callback: _LBFGSCallback | None = None,
    progress_callback: _LBFGSProgressCallback | None = None,
    initial_value_and_grad: _LBFGSInitialValueAndGrad | None = None,
    record_optimizer_state_trace: bool = False,
    max_optimizer_state_trace_bytes: int | None = None,
    diagnostic_event_callback: _LBFGSDiagnosticCallback | None = None,
    run_mode: str = _LBFGS_RUN_MODE_STEPWISE,
    x_dtype: jax.typing.DTypeLike | None = None,
) -> _LBFGSResults:
    run_mode = _check_lbfgsb_run_mode(str(run_mode))
    value_and_grad_fun, x0, callback, adapter = _prepare_optimizer_callable_inputs(
        value_and_grad_fun,
        x0,
        value_and_grad=True,
        callback=callback,
    )
    x0 = _require_private_optimizer_runtime(x0, dtype=x_dtype)
    run_mode = _resolve_lbfgsb_run_mode_for_runtime(
        run_mode,
        x0,
        callback=callback,
        progress_callback=progress_callback,
        record_optimizer_state_trace=bool(record_optimizer_state_trace),
    )
    dtype = x0.dtype
    maxiter_limit_value, maxfun_limit_value = _resolve_scipy_lbfgsb_limits(
        maxiter,
        maxfun,
    )
    history_size = _resolve_lbfgs_history_size(
        maxcor,
        maxiter_limit=maxiter_limit_value,
    )
    workspace_bytes = _lbfgsb_workspace_bytes(
        int(x0.size),
        history_size,
        dtype,
        record_optimizer_state_trace=bool(record_optimizer_state_trace),
        maxiter_limit=maxiter_limit_value,
    )
    _check_lbfgsb_trace_budget(
        int(x0.size),
        maxiter_limit_value,
        record_optimizer_state_trace=record_optimizer_state_trace,
        max_optimizer_state_trace_bytes=max_optimizer_state_trace_bytes,
    )

    value_and_grad_kernel, value_and_grad_consts = _cached_lbfgs_value_and_grad_kernel(
        value_and_grad_fun,
        cache_owner=cache_owner,
        adapter=adapter,
        objective_mode=objective_mode,
        dtype=dtype,
        example_x=x0,
    )
    solver_cache_owner, solver_cache_key_prefix = _lbfgsb_cache_context(
        cache_owner,
        adapter,
        objective_mode,
        dtype,
        x0.shape,
    )

    _record_diagnostic_event(
        diagnostic_event_callback,
        "lbfgs_initial_state_started",
        maxiter=int(maxiter_limit_value),
        maxfun=int(maxfun_limit_value),
        maxcor=int(history_size),
        maxls=int(maxls),
        n=int(x0.size),
        workspace_bytes=workspace_bytes,
        callback_enabled=callback is not None or progress_callback is not None,
        record_optimizer_state_trace=bool(record_optimizer_state_trace),
        run_mode=run_mode,
    )
    with host_transfer_phase("initialization"):
        state = _lbfgsb_initial_state_kernel(
            cache_owner=solver_cache_owner,
            cache_key_prefix=solver_cache_key_prefix,
            m=history_size,
            ftol=ftol,
            gtol=gtol,
            maxls=maxls,
        )(x0)
    _record_diagnostic_event(
        diagnostic_event_callback,
        "lbfgs_initial_state_returned",
    )
    if initial_value_and_grad is not None:
        _record_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_initial_value_and_grad_seed_started",
        )
        with host_transfer_phase("initialization"):
            state = _lbfgsb_state_with_initial_value_and_grad(
                state,
                initial_value_and_grad,
                dtype=dtype,
                cache_owner=solver_cache_owner,
                cache_key_prefix=solver_cache_key_prefix,
            )
        _record_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_initial_value_and_grad_seed_returned",
        )
    optimizer_state_trace = []
    accepted_step_callback = _lbfgsb_accepted_step_observer(
        callback=callback,
        progress_callback=progress_callback,
        optimizer_state_trace=optimizer_state_trace,
        record_optimizer_state_trace=record_optimizer_state_trace,
    )
    _record_diagnostic_event(
        diagnostic_event_callback,
        "lbfgs_main_kernel_started",
        accepted_step_callback=accepted_step_callback is not None,
        record_optimizer_state_trace=bool(record_optimizer_state_trace),
        run_mode=run_mode,
    )
    if run_mode == _LBFGS_RUN_MODE_MONOLITHIC_DEBUG:
        result = _lbfgsb_mainlb_kernel(
            value_and_grad_kernel,
            cache_owner=solver_cache_owner,
            cache_key_prefix=solver_cache_key_prefix,
            maxiter=int(maxiter_limit_value),
            maxfun=int(maxfun_limit_value),
            accepted_step_callback=accepted_step_callback,
        )(state, value_and_grad_consts)
    else:
        unconstrained_fast_path = _lbfgsb_unconstrained_fast_path_enabled(
            _LBFGSB_PRIVATE_BOUNDS
        )
        advance_from_start = _lbfgsb_advance_to_next_observable_kernel(
            value_and_grad_kernel,
            cache_owner=solver_cache_owner,
            cache_key_prefix=solver_cache_key_prefix,
            accepted_step_callback=None,
            entry_kind=_LBFGS_STEP_ENTRY_START,
            unconstrained_fast_path=unconstrained_fast_path,
            dynamic_limits=True,
        )
        advance_from_search = _lbfgsb_advance_to_next_observable_kernel(
            value_and_grad_kernel,
            cache_owner=solver_cache_owner,
            cache_key_prefix=solver_cache_key_prefix,
            accepted_step_callback=None,
            entry_kind=_LBFGS_STEP_ENTRY_SEARCH,
            unconstrained_fast_path=unconstrained_fast_path,
            dynamic_limits=True,
        )
        reenter_new_x = _lbfgsb_advance_to_next_observable_kernel(
            value_and_grad_kernel,
            cache_owner=solver_cache_owner,
            cache_key_prefix=solver_cache_key_prefix,
            accepted_step_callback=None,
            entry_kind=_LBFGS_STEP_ENTRY_NEW_X_REENTRY,
            unconstrained_fast_path=unconstrained_fast_path,
            dynamic_limits=True,
        )
        result_payload = _lbfgsb_result_payload_kernel(
            cache_owner=solver_cache_owner,
            cache_key_prefix=solver_cache_key_prefix,
        )
        callback_stop_state = None
        if accepted_step_callback is not None:
            callback_stop_state = _lbfgsb_callback_stop_state_kernel(
                cache_owner=solver_cache_owner,
                cache_key_prefix=solver_cache_key_prefix,
            )
        result = _lbfgsb_stepwise_driver(
            state,
            value_and_grad_consts,
            advance_from_start,
            advance_from_search,
            reenter_new_x,
            result_payload,
            initial_entry_kind=(
                _LBFGS_STEP_ENTRY_SEARCH
                if initial_value_and_grad is not None
                else _LBFGS_STEP_ENTRY_START
            ),
            maxiter=int(maxiter_limit_value),
            maxfun=int(maxfun_limit_value),
            accepted_step_callback=accepted_step_callback,
            callback_stop_state=callback_stop_state,
        )
    _record_diagnostic_event(
        diagnostic_event_callback,
        "lbfgs_main_kernel_returned",
    )
    if accepted_step_callback is not None:
        _record_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_effects_barrier_started",
        )
        jax.effects_barrier()
        _record_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_effects_barrier_returned",
        )
    return result._replace(optimizer_state_trace=tuple(optimizer_state_trace))


def _minimize_lbfgs_private(
    fun: Callable[[jax.Array], jax.Array],
    x0: jax.typing.ArrayLike,
    *,
    maxiter: int | None = None,
    maxcor: int = 10,
    ftol: float = 0.0,
    gtol: float = 1e-5,
    maxfun: int | None = None,
    maxls: int = 20,
    callback: _LBFGSCallback | None = None,
    progress_callback: _LBFGSProgressCallback | None = None,
    record_optimizer_state_trace: bool = False,
    max_optimizer_state_trace_bytes: int | None = None,
    diagnostic_event_callback: _LBFGSDiagnosticCallback | None = None,
    run_mode: str = _LBFGS_RUN_MODE_STEPWISE,
    x_dtype: jax.typing.DTypeLike | None = None,
) -> _LBFGSResults:
    return _minimize_lbfgs_private_impl(
        _scalar_value_and_grad(fun),
        x0,
        cache_owner=fun,
        objective_mode="scalar",
        maxiter=maxiter,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxls=maxls,
        callback=callback,
        progress_callback=progress_callback,
        record_optimizer_state_trace=record_optimizer_state_trace,
        max_optimizer_state_trace_bytes=max_optimizer_state_trace_bytes,
        diagnostic_event_callback=diagnostic_event_callback,
        run_mode=run_mode,
        x_dtype=x_dtype,
    )


def _minimize_lbfgs_private_value_and_grad(
    fun: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x0: jax.typing.ArrayLike,
    *,
    maxiter: int | None = None,
    maxcor: int = 10,
    ftol: float = 0.0,
    gtol: float = 1e-5,
    maxfun: int | None = None,
    maxls: int = 20,
    callback: _LBFGSCallback | None = None,
    progress_callback: _LBFGSProgressCallback | None = None,
    initial_value_and_grad: _LBFGSInitialValueAndGrad | None = None,
    record_optimizer_state_trace: bool = False,
    max_optimizer_state_trace_bytes: int | None = None,
    diagnostic_event_callback: _LBFGSDiagnosticCallback | None = None,
    run_mode: str = _LBFGS_RUN_MODE_STEPWISE,
    x_dtype: jax.typing.DTypeLike | None = None,
) -> _LBFGSResults:
    return _minimize_lbfgs_private_impl(
        fun,
        x0,
        cache_owner=fun,
        objective_mode="value_and_grad",
        maxiter=maxiter,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxls=maxls,
        callback=callback,
        progress_callback=progress_callback,
        initial_value_and_grad=initial_value_and_grad,
        record_optimizer_state_trace=record_optimizer_state_trace,
        max_optimizer_state_trace_bytes=max_optimizer_state_trace_bytes,
        diagnostic_event_callback=diagnostic_event_callback,
        run_mode=run_mode,
        x_dtype=x_dtype,
    )
