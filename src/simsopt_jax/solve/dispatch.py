"""Driver dispatch for the public ``simsopt_jax.solve`` API."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict
from threading import Lock
from typing import Callable, TypeAlias, TypeVar, cast

import jax
import jax.numpy as jnp
import lineax
import numpy as np
import optimistix as optx
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares as scipy_least_squares
from scipy.optimize import minimize as scipy_minimize

from simsopt_jax.backend.dtypes import explicit_device_array, runtime_device_put_tree
from simsopt_jax.geo.optimizers import optimizer as legacy
from simsopt_jax.runtime.host_boundary import (
    block_until_ready,
    host_array_after_ready,
    host_int,
)

from .contracts import (
    Callback,
    Driver,
    HessianInverse,
    InvalidStepEvent,
    OptimizerCallbackEvent,
    OptimizerResult,
    OptimizerStateTraceEntry,
    OptionsBase,
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
from .driver import (
    legacy_reference_least_squares_method,
    legacy_reference_minimize_method,
    legacy_target_least_squares_method,
    legacy_target_minimize_method,
)
from .minimize_runtime import (
    block_jax_leaves as _block_jax_leaves,
)
from .minimize_runtime import (
    host_array as _host_array,
)
from .minimize_runtime import (
    host_float as _host_float,
)
from .minimize_runtime import (
    optimistix_result_metadata as _optimistix_result_metadata,
)
from .minimize_runtime import (
    run_optax_minimize as _run_optax_minimize,
)
from .minimize_runtime import (
    run_optimistix_minimize as _run_optimistix_minimize,
)
from .optax.contracts import OptaxAdamOptions, OptaxLBFGSOptions
from .optimistix.contracts import (
    LinearSolver,
    OptimistixLBFGSOptions,
    OptimistixLMOptions,
)
from .scipy.contracts import ScipyBFGSOptions, ScipyLBFGSBOptions, ScipyLMOptions
from .shared import LineSearchStatus
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


def _host_optional_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return _host_array(value)


def _device_scalar(value: float, dtype) -> jax.Array:
    return explicit_device_array(value, dtype=dtype)


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
    else:
        payload["line_search_maxiter"] = options.line_search_max_steps
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
    hess_inv = cast(HessianInverse | None, getattr(result, "hess_inv", None))
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
        hess_inv=hess_inv,
        invalid_step_log=_invalid_step_events(result),
        optimizer_state_trace=_optimizer_state_trace(result),
        optimistix_result=getattr(result, "optimistix_result", None),
        optimistix_result_message=getattr(result, "optimistix_result_message", None),
    )


def _scipy_lm_result(
    residual_fn: ResidualFn,
    x0,
    *,
    options: ScipyLMOptions,
    residual_args: tuple[object, ...] = (),
) -> OptimizeResult:
    def residual_host(x_host):
        return np.asarray(
            residual_fn(np.asarray(x_host), *residual_args),
            dtype=float,
        ).reshape(-1)

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
        x_host = host_array_after_ready(x, dtype=float)
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
    residual_args: tuple[object, ...] = (),
) -> OptimizeResult:
    params = jnp.asarray(runtime_device_put_tree(x0))
    solver_tol = float(options.tol)

    def residual_value(current, function_args):
        return jnp.ravel(jnp.asarray(residual_fn(current, *function_args)))

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
            args=residual_args,
            max_steps=options.maxiter,
            throw=False,
        )
    block_until_ready(solution.value)
    residual = jnp.ravel(jnp.asarray(residual_fn(solution.value, *residual_args)))
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

        def dense_residual(x, function_args):
            return jnp.ravel(jnp.asarray(residual_fn(x, *function_args)))

        jacobian = jax.jacrev(dense_residual, argnums=0)(
            solution.value,
            residual_args,
        )
        jacobian = jnp.reshape(jacobian, (residual_size, parameter_size))
        hessian = jacobian.T @ jacobian
        gradient = jnp.reshape(jacobian.T @ residual, solution.value.shape)
        _block_jax_leaves((residual, jacobian, hessian, gradient, cost))
        residual_jacobian = _host_array(jacobian)
        hessian_host = _host_array(hessian)
    else:

        def residual_cost(x, function_args):
            current_residual = jnp.ravel(jnp.asarray(residual_fn(x, *function_args)))
            return half * jnp.vdot(current_residual, current_residual)

        _cost_value, pullback = jax.vjp(
            residual_cost,
            solution.value,
            residual_args,
        )
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
    num_steps = host_int(solution.stats["num_steps"])
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
    residual_args: tuple[object, ...] = (),
) -> OptimizerResult:
    """Solve a residual least-squares problem with a typed JAX-lane driver."""
    options_used = _resolve_options(driver, options, _LEAST_SQUARES_OPTIONS)
    if callback is not None and driver in {Driver.SCIPY_LM, Driver.OPTIMISTIX_LM}:
        raise ValueError(f"driver={driver.value!r} does not support callbacks.")
    start = time.perf_counter()

    if isinstance(options_used, ScipyLMOptions):
        if residual_args:
            result = _scipy_lm_result(
                residual_fn,
                x0,
                options=options_used,
                residual_args=residual_args,
            )
        else:
            result = _scipy_lm_result(residual_fn, x0, options=options_used)
    elif isinstance(options_used, OptimistixLMOptions):
        if residual_args:
            result = _run_optimistix_lm(
                residual_fn,
                x0,
                options=options_used,
                residual_args=residual_args,
            )
        else:
            result = _run_optimistix_lm(residual_fn, x0, options=options_used)
    elif isinstance(options_used, SimsoptLMGMRESHostOptions) and not isinstance(
        options_used, SimsoptLMGMRESOptions
    ):
        legacy_callback, legacy_progress_callback = _legacy_least_squares_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )

        def residual_with_args(current_x):
            return residual_fn(current_x, *residual_args)

        result = legacy.reference_least_squares(
            residual_with_args,
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
            args=residual_args,
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
