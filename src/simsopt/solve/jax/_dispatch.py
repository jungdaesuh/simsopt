"""Driver dispatch for the public ``simsopt.solve.jax`` API."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from threading import Lock
import time
from typing import Callable, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import lineax
import numpy as np
import optax
import optimistix as optx
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares as scipy_least_squares
from scipy.optimize import minimize as scipy_minimize

from simsopt.geo import optimizer_jax as legacy

from .contracts import (
    Callback,
    Driver,
    InvalidStepEvent,
    OptimizerResult,
    OptimizerCallbackEvent,
    OptimizerStateTraceEntry,
    OptionsBase,
    OptaxAdamCallbackEvent,
    OptaxLBFGSCallbackEvent,
    OptimistixLBFGSCallbackEvent,
    ResidualFn,
    ScipyBFGSCallbackEvent,
    ScipyLBFGSBCallbackEvent,
    SimsoptAdamCallbackEvent,
    SimsoptAdamHostCallbackEvent,
    SimsoptBFGSCallbackEvent,
    SimsoptLBFGSBCallbackEvent,
    SimsoptLMGMRESCallbackEvent,
    SimsoptLMGMRESHostCallbackEvent,
    SimsoptLMQRCallbackEvent,
    SimsoptTraceLBFGSCallbackEvent,
    ValueAndGradFn,
)
from ._shared import LineSearchStatus
from .._jax_driver import (
    legacy_reference_least_squares_method,
    legacy_reference_minimize_method,
    legacy_target_least_squares_method,
    legacy_target_minimize_method,
)
from .optimistix.contracts import (
    LinearSolver,
    OptimistixLBFGSOptions,
    OptimistixLMOptions,
)
from .optax.contracts import OptaxAdamOptions, OptaxLBFGSOptions, OptaxLineSearch
from .scipy.contracts import ScipyBFGSOptions, ScipyLBFGSBOptions, ScipyLMOptions
from .simsopt.contracts import (
    SimsoptAdamHostOptions,
    SimsoptAdamOptions,
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
    SimsoptLMGMRESHostOptions,
    SimsoptLMGMRESOptions,
    SimsoptLMQROptions,
    SimsoptTraceLBFGSOptions,
)

_OptionsT = TypeVar("_OptionsT", bound=OptionsBase)
_LegacyProgress: TypeAlias = tuple[int, float, float]
_LegacyEventFactory: TypeAlias = Callable[
    [np.ndarray, int, float, float, float],
    OptimizerCallbackEvent,
]


_MINIMIZE_OPTIONS: dict[Driver, type[OptionsBase]] = {
    Driver.SCIPY_LBFGSB: ScipyLBFGSBOptions,
    Driver.SCIPY_BFGS: ScipyBFGSOptions,
    Driver.OPTAX_LBFGS: OptaxLBFGSOptions,
    Driver.OPTAX_ADAM: OptaxAdamOptions,
    Driver.OPTIMISTIX_LBFGS: OptimistixLBFGSOptions,
    Driver.SIMSOPT_LBFGSB: SimsoptLBFGSBOptions,
    Driver.SIMSOPT_BFGS: SimsoptBFGSOptions,
    Driver.SIMSOPT_TRACE_LBFGS: SimsoptTraceLBFGSOptions,
    Driver.SIMSOPT_ADAM_HOST: SimsoptAdamHostOptions,
    Driver.SIMSOPT_ADAM: SimsoptAdamOptions,
}

_LEAST_SQUARES_OPTIONS: dict[Driver, type[OptionsBase]] = {
    Driver.SCIPY_LM: ScipyLMOptions,
    Driver.OPTIMISTIX_LM: OptimistixLMOptions,
    Driver.SIMSOPT_LM_GMRES_HOST: SimsoptLMGMRESHostOptions,
    Driver.SIMSOPT_LM_GMRES: SimsoptLMGMRESOptions,
    Driver.SIMSOPT_LM_QR: SimsoptLMQROptions,
}


def _resolve_options(
    driver: Driver,
    options: OptionsBase | None,
    options_by_driver: dict[Driver, type[_OptionsT]],
) -> _OptionsT:
    options_type = options_by_driver.get(driver)
    if options_type is None:
        allowed = ", ".join(sorted(item.value for item in options_by_driver))
        raise ValueError(
            f"Driver {driver.value!r} is not valid here. Choose: {allowed}."
        )
    if options is None:
        return options_type()
    if type(options) is not options_type:
        raise TypeError(
            f"Driver {driver.value!r} requires options of type "
            f"{options_type.__name__}, got {type(options).__name__}."
        )
    return options


def _host_array(value) -> np.ndarray:
    return np.asarray(jax.device_get(jax.block_until_ready(value)))


def _host_optional_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return _host_array(value)


def _host_float(value) -> float:
    return float(np.asarray(jax.device_get(jax.block_until_ready(value))).reshape(()))


def _device_scalar(value: float, dtype) -> jax.Array:
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))


def _optimistix_result_metadata(result) -> tuple[bool, str, str]:
    with jax.transfer_guard_host_to_device("allow"):
        result_text = str(result)
        result_message = str(optx.RESULTS[result])
        successful = bool(result == optx.RESULTS.successful)
    return successful, result_text, result_message


def _legacy_minimize_options(options: OptionsBase) -> dict[str, object]:
    return asdict(options)


def _legacy_lbfgsb_options(options: ScipyLBFGSBOptions | SimsoptLBFGSBOptions):
    return {
        "ftol": options.ftol,
        "maxcor": options.maxcor,
        "maxfun": options.maxfun,
        "maxls": options.maxls,
    }


def _legacy_bfgs_options(options: ScipyBFGSOptions | SimsoptBFGSOptions):
    payload = {"xrtol": options.xrtol}
    if isinstance(options, ScipyBFGSOptions):
        payload["norm"] = options.norm
    return payload


def _legacy_lm_options(
    options: SimsoptLMGMRESHostOptions | SimsoptLMGMRESOptions | SimsoptLMQROptions,
):
    payload: dict[str, object] = {
        "ftol": options.ftol,
        "xtol": options.xtol,
        "gtol": options.gtol,
    }
    if isinstance(options, SimsoptLMGMRESOptions | SimsoptLMQROptions):
        payload["max_dense_linearization_bytes"] = options.max_dense_linearization_bytes
    if isinstance(options, SimsoptLMGMRESOptions):
        payload["materialize_dense_linearization"] = (
            options.materialize_dense_linearization
        )
    return payload


def _invalid_step_events(result: OptimizeResult) -> list[InvalidStepEvent] | None:
    raw_log = getattr(result, "invalid_step_log", None)
    if raw_log is None:
        return None
    return [
        InvalidStepEvent(
            iteration=int(entry["iteration"]),
            step_scale=float(entry["step_scale"]),
            reason=str(entry["failure_reason"]),
        )
        for entry in raw_log
    ]


def _optimizer_state_trace(
    result: OptimizeResult,
) -> list[OptimizerStateTraceEntry] | None:
    raw_trace = getattr(result, "optimizer_state_trace", None)
    if raw_trace is None:
        return None
    return [
        OptimizerStateTraceEntry(
            iteration=int(entry["iteration"]),
            fun=float(entry["fun"]),
            grad_norm_inf=float(entry["jac_inf_norm"]),
        )
        for entry in raw_trace
    ]


def _public_result(
    result: OptimizeResult,
    *,
    driver: Driver,
    options_used: OptionsBase,
    wallclock_s: float,
) -> OptimizerResult:
    return OptimizerResult(
        x=_host_array(result.x),
        fun=_host_float(result.fun),
        jac=_host_optional_array(getattr(result, "jac", None)),
        nit=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)),
        status=int(getattr(result, "status", 0)),
        success=bool(getattr(result, "success", False)),
        message=str(getattr(result, "message", "")),
        driver=driver,
        options_used=options_used,
        wallclock_s=wallclock_s,
        residual=_host_optional_array(getattr(result, "residual", None)),
        residual_jacobian=_host_optional_array(
            getattr(result, "residual_jacobian", None)
        ),
        hessian=_host_optional_array(getattr(result, "hessian", None)),
        invalid_step_log=_invalid_step_events(result),
        optimizer_state_trace=_optimizer_state_trace(result),
        optimistix_result=getattr(result, "optimistix_result", None),
        optimistix_result_message=getattr(result, "optimistix_result_message", None),
    )


def _block_jax_leaves(value) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _scalar_value_from_value_and_grad(value_and_grad_fn: ValueAndGradFn):
    def scalar_value(x):
        value, _grad = value_and_grad_fn(x)
        return value

    return scalar_value


def _scalar_value_with_explicit_vjp(value_and_grad_fn: ValueAndGradFn):
    @jax.custom_vjp
    def scalar_value(x):
        value, _grad = value_and_grad_fn(x)
        return jnp.asarray(value)

    def scalar_value_fwd(x):
        value, grad = value_and_grad_fn(x)
        return jnp.asarray(value), jnp.asarray(grad)

    def scalar_value_bwd(explicit_grad, cotangent):
        return (jnp.asarray(cotangent) * explicit_grad,)

    scalar_value.defvjp(scalar_value_fwd, scalar_value_bwd)
    return scalar_value


def _scipy_lm_result(
    residual_fn: ResidualFn,
    x0,
    *,
    options: ScipyLMOptions,
) -> OptimizeResult:
    def residual_host(x_host):
        return np.asarray(residual_fn(np.asarray(x_host)), dtype=float).reshape(-1)

    result = scipy_least_squares(
        residual_host,
        np.asarray(x0, dtype=float),
        method="lm",
        max_nfev=options.max_nfev,
        ftol=options.ftol,
        xtol=options.xtol,
        gtol=options.gtol,
    )
    residual = np.asarray(result.fun, dtype=float).reshape(-1)
    jacobian = np.asarray(result.jac, dtype=float)
    gradient = jacobian.T @ residual
    hessian = jacobian.T @ jacobian
    njev = 0 if result.njev is None else int(result.njev)
    return OptimizeResult(
        x=np.asarray(result.x, dtype=float),
        fun=float(0.5 * np.dot(residual, residual)),
        jac=gradient,
        nit=njev,
        nfev=int(result.nfev),
        njev=njev,
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
        residual=residual,
        residual_jacobian=jacobian,
        hessian=hessian,
    )


def _run_scipy_minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    driver: Driver,
    options: ScipyLBFGSBOptions | ScipyBFGSOptions,
    callback: Callback | None,
) -> OptimizeResult:
    iteration = 0
    start = time.perf_counter()

    def scipy_fun(x_host):
        value, grad = value_and_grad_fn(np.asarray(x_host))
        return float(np.asarray(value).reshape(())), np.asarray(grad, dtype=float)

    def scipy_callback(x_host):
        nonlocal iteration
        if callback is None:
            return

        iteration += 1
        value, grad = value_and_grad_fn(np.asarray(x_host))
        grad_host = np.asarray(grad, dtype=float)
        event_fields = {
            "iteration": iteration,
            "x": np.asarray(x_host, dtype=float).copy(),
            "fun": float(np.asarray(value).reshape(())),
            "grad_norm_inf": float(np.linalg.norm(grad_host, ord=np.inf)),
            "wallclock_s": time.perf_counter() - start,
        }
        if driver == Driver.SCIPY_LBFGSB:
            callback(ScipyLBFGSBCallbackEvent(**event_fields))
        else:
            callback(ScipyBFGSCallbackEvent(**event_fields))

    if isinstance(options, ScipyLBFGSBOptions):
        scipy_method = "L-BFGS-B"
        scipy_options = {
            "maxiter": options.maxiter,
            "maxfun": options.maxfun,
            "gtol": options.gtol,
            "ftol": options.ftol,
            "maxcor": options.maxcor,
            "maxls": options.maxls,
        }
    else:
        scipy_method = "BFGS"
        scipy_options = {
            "maxiter": options.maxiter,
            "gtol": options.gtol,
            "xrtol": options.xrtol,
            "norm": options.norm,
        }
    result = scipy_minimize(
        scipy_fun,
        np.asarray(x0, dtype=float),
        jac=True,
        method=scipy_method,
        options=scipy_options,
        callback=scipy_callback if callback is not None else None,
    )
    return OptimizeResult(
        x=np.asarray(result.x, dtype=float),
        fun=float(np.asarray(result.fun).reshape(())),
        jac=np.asarray(result.jac, dtype=float),
        nit=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)),
        status=int(getattr(result, "status", 0)),
        success=bool(result.success),
        message=str(result.message),
        hessian=getattr(result, "hess_inv", None)
        if driver == Driver.SCIPY_BFGS
        else None,
    )


def _run_optax_minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    driver: Driver,
    options: OptaxLBFGSOptions | OptaxAdamOptions,
    callback: Callback | None,
) -> OptimizeResult:
    start = time.perf_counter()
    params = jnp.asarray(jax.device_put(x0))
    value_fn = _scalar_value_from_value_and_grad(value_and_grad_fn)
    maxiter = options.maxiter
    gtol = options.gtol
    if isinstance(options, OptaxLBFGSOptions):
        if options.line_search != OptaxLineSearch.ZOOM:
            raise ValueError(
                "OptaxLBFGSOptions.line_search must be OptaxLineSearch.ZOOM."
            )
        transform = optax.lbfgs(
            memory_size=options.memory_size,
            scale_init_precond=options.scale_init_precond,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=options.max_linesearch_steps,
                initial_guess_strategy="one",
            ),
        )
    else:
        transform = (
            optax.adamw(
                learning_rate=options.learning_rate,
                b1=options.b1,
                b2=options.b2,
                eps=options.eps,
                weight_decay=options.weight_decay,
            )
            if options.weight_decay > 0.0
            else optax.adam(
                learning_rate=options.learning_rate,
                b1=options.b1,
                b2=options.b2,
                eps=options.eps,
            )
        )

    state = jax.jit(transform.init)(params)

    @jax.jit
    def apply_optax_step(current_params, current_state, current_grad, current_value):
        updates, next_state = transform.update(
            current_grad,
            current_state,
            current_params,
            value=current_value,
            grad=current_grad,
            value_fn=value_fn,
        )
        return optax.apply_updates(current_params, updates), next_state

    nfev = 0
    njev = 0
    status = 1
    message = "Maximum number of iterations reached."
    success = False
    value, grad = value_and_grad_fn(params)
    value = jnp.asarray(value)
    grad = jnp.asarray(grad)
    _block_jax_leaves((value, grad))
    nfev += 1
    njev += 1
    completed_iterations = 0
    for iteration in range(maxiter):
        grad_norm_inf = _host_float(jnp.max(jnp.abs(grad)))
        if gtol is not None and grad_norm_inf <= float(gtol):
            status = 0
            message = "Optimization terminated successfully."
            success = True
            break
        params, state = apply_optax_step(params, state, grad, value)
        _block_jax_leaves((params, state))
        completed_iterations = iteration + 1
        value, grad = value_and_grad_fn(params)
        value = jnp.asarray(value)
        grad = jnp.asarray(grad)
        _block_jax_leaves((value, grad))
        nfev += 1
        njev += 1
        grad_norm_inf = _host_float(jnp.max(jnp.abs(grad)))
        if callback is not None:
            _emit_optax_callback(
                callback,
                driver=driver,
                iteration=completed_iterations,
                params=params,
                value=value,
                grad_norm_inf=grad_norm_inf,
                state=state,
                options=options,
                wallclock_s=time.perf_counter() - start,
            )
        if gtol is not None and grad_norm_inf <= float(gtol):
            status = 0
            message = "Optimization terminated successfully."
            success = True
            break
    else:
        if gtol is None:
            status = 0
            message = "Optimization terminated after the requested Adam iterations."
            success = driver == Driver.OPTAX_ADAM

    return OptimizeResult(
        x=_host_array(params),
        fun=_host_float(value),
        jac=_host_array(grad),
        nit=completed_iterations,
        nfev=nfev,
        njev=njev,
        status=status,
        success=success,
        message=message,
    )


def _emit_optax_callback(
    callback: Callback,
    *,
    driver: Driver,
    iteration: int,
    params,
    value,
    grad_norm_inf: float,
    state,
    options: OptaxLBFGSOptions | OptaxAdamOptions,
    wallclock_s: float,
) -> None:
    if isinstance(options, OptaxAdamOptions):
        callback(
            OptaxAdamCallbackEvent(
                iteration=iteration,
                x=_host_array(params),
                fun=_host_float(value),
                grad_norm_inf=grad_norm_inf,
                wallclock_s=wallclock_s,
                learning_rate=options.learning_rate,
            )
        )
        return
    line_search_state = state[-1]
    callback(
        OptaxLBFGSCallbackEvent(
            iteration=iteration,
            x=_host_array(params),
            fun=_host_float(value),
            grad_norm_inf=grad_norm_inf,
            wallclock_s=wallclock_s,
            learning_rate=_host_float(line_search_state.learning_rate),
            num_linesearch_steps=int(
                np.asarray(jax.device_get(line_search_state.info.num_linesearch_steps))
            ),
            decrease_error=_host_float(line_search_state.info.decrease_error),
            curvature_error=_host_float(line_search_state.info.curvature_error),
        )
    )


def _run_optimistix_minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    options: OptimistixLBFGSOptions,
    callback: Callback | None,
) -> OptimizeResult:
    start = time.perf_counter()
    params = jnp.asarray(jax.device_put(x0))
    solver_tol = float(options.tol)
    explicit_scalar_value = _scalar_value_with_explicit_vjp(value_and_grad_fn)

    def scalar_value(current, _args):
        return explicit_scalar_value(current)

    solver = optx.LBFGS(
        rtol=solver_tol,
        atol=solver_tol,
        history_length=options.history_length,
    )
    with jax.transfer_guard_host_to_device("allow"):
        solution = optx.minimise(
            scalar_value,
            solver,
            params,
            args=None,
            max_steps=options.maxiter,
            throw=False,
        )
    solution.value.block_until_ready()
    value, grad = value_and_grad_fn(solution.value)
    value = jnp.asarray(value)
    grad = jnp.asarray(grad)
    _block_jax_leaves((value, grad))
    x_host = _host_array(solution.value)
    fun_host = _host_float(value)
    jac_host = _host_array(grad)
    grad_norm_inf = float(np.max(np.abs(jac_host))) if jac_host.size else 0.0
    finite = (
        np.all(np.isfinite(x_host))
        and np.isfinite(fun_host)
        and np.all(np.isfinite(jac_host))
    )
    (
        optimistix_success,
        optimistix_result,
        optimistix_result_message,
    ) = _optimistix_result_metadata(solution.result)
    success = optimistix_success and bool(finite)
    status = 0 if success else 1 if finite else 2
    num_steps = int(np.asarray(jax.device_get(solution.stats["num_steps"])))
    wallclock_s = time.perf_counter() - start
    if callback is not None:
        callback(
            OptimistixLBFGSCallbackEvent(
                iteration=num_steps,
                x=x_host,
                fun=fun_host,
                grad_norm_inf=grad_norm_inf,
                wallclock_s=wallclock_s,
                history_length=options.history_length,
                optimistix_result=optimistix_result,
            )
        )
    return OptimizeResult(
        x=x_host,
        fun=fun_host,
        jac=jac_host,
        nit=num_steps,
        nfev=num_steps + 1,
        njev=num_steps + 1,
        status=status,
        success=success,
        message=optimistix_result,
        optimistix_result=optimistix_result,
        optimistix_result_message=optimistix_result_message,
    )


def _legacy_callback_pair(
    callback: Callback,
    make_event: _LegacyEventFactory,
):
    pending_x: deque[np.ndarray] = deque()
    pending_progress: deque[_LegacyProgress] = deque()
    pending_lock = Lock()
    start = time.perf_counter()

    def ready_event():
        if not pending_x or not pending_progress:
            return None
        x_host = pending_x.popleft()
        iteration, fun, grad_norm_inf = pending_progress.popleft()
        return make_event(
            x_host.copy(),
            iteration,
            fun,
            grad_norm_inf,
            time.perf_counter() - start,
        )

    def legacy_callback(x):
        x_host = np.asarray(jax.device_get(jax.block_until_ready(x)), dtype=float)
        with pending_lock:
            pending_x.append(x_host)
            event = ready_event()
        if event is not None:
            callback(event)

    def legacy_progress(iteration_value, fun_value, grad_norm_inf):
        progress = (
            int(np.asarray(iteration_value).reshape(())),
            float(np.asarray(fun_value).reshape(())),
            float(np.asarray(grad_norm_inf).reshape(())),
        )
        with pending_lock:
            pending_progress.append(progress)
            event = ready_event()
        if event is not None:
            callback(event)

    return legacy_callback, legacy_progress


def _legacy_minimize_callbacks(
    callback: Callback | None,
    *,
    driver: Driver,
    options: OptionsBase,
):
    if callback is None:
        return None, None

    def make_event(
        x_host: np.ndarray,
        iteration: int,
        fun: float,
        grad_norm_inf: float,
        wallclock_s: float,
    ) -> OptimizerCallbackEvent:
        base_fields = {
            "iteration": iteration,
            "x": x_host,
            "fun": fun,
            "grad_norm_inf": grad_norm_inf,
            "wallclock_s": wallclock_s,
        }
        if driver == Driver.SIMSOPT_LBFGSB:
            return SimsoptLBFGSBCallbackEvent(
                **base_fields,
                accepted_alpha=float("nan"),
                num_linesearch_steps=0,
            )
        if driver == Driver.SIMSOPT_BFGS:
            return SimsoptBFGSCallbackEvent(
                **base_fields,
                accepted_alpha=float("nan"),
                num_linesearch_steps=0,
            )
        if driver == Driver.SIMSOPT_TRACE_LBFGS:
            return SimsoptTraceLBFGSCallbackEvent(
                **base_fields,
                accepted_alpha=float("nan"),
                rejected_alphas=(),
                line_search_status=LineSearchStatus.SUCCESS,
                invalid_step_reason=None,
            )
        if isinstance(options, SimsoptAdamOptions):
            return SimsoptAdamCallbackEvent(
                **base_fields,
                learning_rate=options.learning_rate,
            )
        if isinstance(options, SimsoptAdamHostOptions):
            return SimsoptAdamHostCallbackEvent(
                **base_fields,
                learning_rate=options.learning_rate,
            )
        raise ValueError(f"driver={driver.value!r} does not support callbacks.")

    return _legacy_callback_pair(callback, make_event)


def _legacy_least_squares_callbacks(
    callback: Callback | None,
    *,
    driver: Driver,
    options: OptionsBase,
):
    if callback is None:
        return None, None

    def make_event(
        x_host: np.ndarray,
        iteration: int,
        fun: float,
        grad_norm_inf: float,
        wallclock_s: float,
    ) -> OptimizerCallbackEvent:
        residual_norm = float(np.sqrt(max(0.0, 2.0 * fun)))
        base_fields = {
            "iteration": iteration,
            "x": x_host,
            "fun": fun,
            "grad_norm_inf": grad_norm_inf,
            "wallclock_s": wallclock_s,
        }
        if isinstance(options, SimsoptLMGMRESOptions):
            return SimsoptLMGMRESCallbackEvent(
                **base_fields,
                residual_norm=residual_norm,
                damping=float("nan"),
                gmres_iterations=0,
            )
        if isinstance(options, SimsoptLMGMRESHostOptions):
            return SimsoptLMGMRESHostCallbackEvent(
                **base_fields,
                residual_norm=residual_norm,
                damping=float("nan"),
                gmres_iterations=0,
            )
        if isinstance(options, SimsoptLMQROptions):
            return SimsoptLMQRCallbackEvent(
                **base_fields,
                residual_norm=residual_norm,
                damping=float("nan"),
                minpack_info=None,
            )
        raise ValueError(f"driver={driver.value!r} does not support callbacks.")

    return _legacy_callback_pair(callback, make_event)


def _run_optimistix_lm(
    residual_fn: ResidualFn,
    x0,
    *,
    options: OptimistixLMOptions,
) -> OptimizeResult:
    params = jnp.asarray(jax.device_put(x0))
    solver_tol = float(options.tol)

    def residual_value(current, _args):
        return jnp.ravel(jnp.asarray(residual_fn(current)))

    if options.linear_solver == LinearSolver.LSMR:
        linear_solver = lineax.LSMR(rtol=solver_tol, atol=solver_tol)
    else:
        linear_solver = lineax.QR()
    solver = optx.LevenbergMarquardt(
        rtol=solver_tol,
        atol=solver_tol,
        linear_solver=linear_solver,
    )
    with jax.transfer_guard_host_to_device("allow"):
        solution = optx.least_squares(
            residual_value,
            solver,
            params,
            args=None,
            max_steps=options.maxiter,
            throw=False,
        )
    solution.value.block_until_ready()
    residual = jnp.ravel(jnp.asarray(residual_fn(solution.value)))
    half = _device_scalar(0.5, residual.dtype)
    cost = half * jnp.vdot(residual, residual)
    if options.materialize_dense_linearization:
        residual_size = int(residual.size)
        parameter_size = int(solution.value.size)
        dtype_size = np.dtype(residual.dtype).itemsize
        dense_linearization_bytes = residual_size * parameter_size * dtype_size
        dense_hessian_bytes = parameter_size * parameter_size * dtype_size
        dense_bytes = dense_linearization_bytes + dense_hessian_bytes
        if (
            options.max_dense_linearization_bytes is not None
            and dense_bytes > options.max_dense_linearization_bytes
        ):
            raise ValueError(
                "Optimistix dense linearization requires "
                f"{dense_bytes} bytes, exceeding "
                f"max_dense_linearization_bytes={options.max_dense_linearization_bytes}."
            )
        jacobian = jax.jacrev(lambda x: jnp.ravel(jnp.asarray(residual_fn(x))))(
            solution.value
        )
        jacobian = jnp.reshape(jacobian, (residual_size, parameter_size))
        hessian = jacobian.T @ jacobian
        gradient = jnp.reshape(jacobian.T @ residual, solution.value.shape)
        _block_jax_leaves((residual, jacobian, hessian, gradient, cost))
        residual_jacobian = _host_array(jacobian)
        hessian_host = _host_array(hessian)
    else:

        def residual_cost(x):
            residual_value = jnp.ravel(jnp.asarray(residual_fn(x)))
            return half * jnp.vdot(residual_value, residual_value)

        _cost_value, pullback = jax.vjp(residual_cost, solution.value)
        gradient = pullback(
            _device_scalar(1.0, residual.dtype),
        )[0]
        _block_jax_leaves((residual, gradient, cost))
        residual_jacobian = None
        hessian_host = None
    x_host = _host_array(solution.value)
    fun_host = _host_float(cost)
    jac_host = _host_array(gradient)
    residual_host = _host_array(residual)
    finite = (
        np.all(np.isfinite(x_host))
        and np.isfinite(fun_host)
        and np.all(np.isfinite(jac_host))
        and np.all(np.isfinite(residual_host))
    )
    if hessian_host is not None:
        finite = finite and np.all(np.isfinite(hessian_host))
    (
        optimistix_success,
        optimistix_result,
        optimistix_result_message,
    ) = _optimistix_result_metadata(solution.result)
    success = optimistix_success and bool(finite)
    status = 0 if success else 1 if finite else 2
    num_steps = int(np.asarray(jax.device_get(solution.stats["num_steps"])))
    return OptimizeResult(
        x=x_host,
        fun=fun_host,
        jac=jac_host,
        nit=num_steps,
        nfev=num_steps + 1,
        njev=num_steps + 1,
        status=status,
        success=success,
        message=optimistix_result,
        residual=residual_host,
        residual_jacobian=residual_jacobian,
        hessian=hessian_host,
        optimistix_result=optimistix_result,
        optimistix_result_message=optimistix_result_message,
    )


def minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    driver: Driver,
    options: OptionsBase | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize a scalar objective with a typed JAX-lane driver."""
    options_used = _resolve_options(driver, options, _MINIMIZE_OPTIONS)
    start = time.perf_counter()
    if isinstance(options_used, ScipyLBFGSBOptions | ScipyBFGSOptions):
        result = _run_scipy_minimize(
            value_and_grad_fn,
            x0,
            driver=driver,
            options=options_used,
            callback=callback,
        )
    elif isinstance(options_used, OptaxLBFGSOptions | OptaxAdamOptions):
        result = _run_optax_minimize(
            value_and_grad_fn,
            x0,
            driver=driver,
            options=options_used,
            callback=callback,
        )
    elif isinstance(options_used, OptimistixLBFGSOptions):
        result = _run_optimistix_minimize(
            value_and_grad_fn,
            x0,
            options=options_used,
            callback=callback,
        )
    elif isinstance(options_used, SimsoptLBFGSBOptions):
        legacy_callback, legacy_progress_callback = _legacy_minimize_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )
        result = legacy.target_minimize(
            value_and_grad_fn,
            x0,
            method=legacy_target_minimize_method(driver),
            tol=options_used.gtol,
            maxiter=options_used.maxiter,
            options=_legacy_lbfgsb_options(options_used),
            value_and_grad=True,
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    elif isinstance(options_used, SimsoptBFGSOptions):
        legacy_callback, legacy_progress_callback = _legacy_minimize_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )
        result = legacy.target_minimize(
            value_and_grad_fn,
            x0,
            method=legacy_target_minimize_method(driver),
            tol=options_used.gtol,
            maxiter=options_used.maxiter,
            options=_legacy_bfgs_options(options_used),
            value_and_grad=True,
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    elif isinstance(options_used, SimsoptTraceLBFGSOptions):
        trace_options = _legacy_minimize_options(options_used)
        trace_options.pop("gtol")
        trace_options.pop("maxiter")
        legacy_callback, legacy_progress_callback = _legacy_minimize_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )
        result = legacy.reference_minimize(
            value_and_grad_fn,
            x0,
            method=legacy_reference_minimize_method(driver),
            tol=options_used.gtol,
            maxiter=options_used.maxiter,
            options=trace_options,
            value_and_grad=True,
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    elif isinstance(options_used, SimsoptAdamHostOptions | SimsoptAdamOptions):
        is_host_adam = isinstance(
            options_used, SimsoptAdamHostOptions
        ) and not isinstance(options_used, SimsoptAdamOptions)
        method = (
            legacy_reference_minimize_method(driver)
            if is_host_adam
            else legacy_target_minimize_method(driver)
        )
        driver_fn = (
            legacy.reference_minimize if is_host_adam else legacy.target_minimize
        )
        legacy_callback, legacy_progress_callback = _legacy_minimize_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )
        result = driver_fn(
            value_and_grad_fn,
            x0,
            method=method,
            tol=options_used.gtol if options_used.gtol is not None else 0.0,
            maxiter=options_used.maxiter,
            options={
                "step_size": options_used.learning_rate,
                "beta1": options_used.b1,
                "beta2": options_used.b2,
                "eps": options_used.eps,
            },
            value_and_grad=True,
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    else:
        raise ValueError(f"Unsupported minimize options {type(options_used).__name__}.")
    wallclock_s = time.perf_counter() - start
    return _public_result(
        result,
        driver=driver,
        options_used=options_used,
        wallclock_s=wallclock_s,
    )


def least_squares(
    residual_fn: ResidualFn,
    x0,
    *,
    driver: Driver,
    options: OptionsBase | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Solve a residual least-squares problem with a typed JAX-lane driver."""
    options_used = _resolve_options(driver, options, _LEAST_SQUARES_OPTIONS)
    if callback is not None and driver in {Driver.SCIPY_LM, Driver.OPTIMISTIX_LM}:
        raise ValueError(f"driver={driver.value!r} does not support callbacks.")
    start = time.perf_counter()

    if isinstance(options_used, ScipyLMOptions):
        result = _scipy_lm_result(residual_fn, x0, options=options_used)
    elif isinstance(options_used, OptimistixLMOptions):
        result = _run_optimistix_lm(residual_fn, x0, options=options_used)
    elif isinstance(options_used, SimsoptLMGMRESHostOptions) and not isinstance(
        options_used, SimsoptLMGMRESOptions
    ):
        legacy_callback, legacy_progress_callback = _legacy_least_squares_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )
        result = legacy.reference_least_squares(
            residual_fn,
            x0,
            method=legacy_reference_least_squares_method(driver),
            maxiter=options_used.maxiter,
            options=_legacy_lm_options(options_used),
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    elif isinstance(options_used, SimsoptLMGMRESOptions | SimsoptLMQROptions):
        legacy_callback, legacy_progress_callback = _legacy_least_squares_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )
        result = legacy.target_least_squares(
            residual_fn,
            x0,
            method=legacy_target_least_squares_method(driver),
            maxiter=options_used.maxiter,
            options=_legacy_lm_options(options_used),
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    else:
        raise ValueError(
            f"Unsupported least-squares options {type(options_used).__name__}."
        )
    wallclock_s = time.perf_counter() - start
    return _public_result(
        result,
        driver=driver,
        options_used=options_used,
        wallclock_s=wallclock_s,
    )


__all__ = ["least_squares", "minimize"]
